"""Launch and read-only monitor a retail PopCap DMO replay.

This diagnostic deliberately leaves the replay at its normal 1.0 update
multiplier.  It binds the temporary, path-verified Steam runtime and samples
known ``SexyAppBase`` demo fields until the process exits or a timeout expires.
No target-process memory is written.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct
import subprocess
import sys
import time
from typing import Iterable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.inspect_popcap_replay import (
    close_process,
    open_process_readonly,
    read_process_bytes,
)
from tools.launch_popcap_replay import (
    DEFAULT_RUNTIME_EXE,
    DEFAULT_STEAM_EXE,
    process_ids_by_name,
    wait_for_new_runtime_window,
)


G_SEXY_APP_BASE_ADDRESS = 0x009FC740
STATE_START_OFFSET = 0x4C4
STATE_END_OFFSET = 0x631


def _decode_state(data: bytes) -> dict[str, int | float | bool]:
    """Decode the retail framework fields used to assess replay progress."""

    def i32(relative: int) -> int:
        return struct.unpack_from("<i", data, relative)[0]

    return {
        "update": i32(0x000),
        "multiplier": struct.unpack_from("<d", data, 0x00C)[0],
        "paused": bool(data[0x014]),
        "buffer_read_bit_position": i32(0x144),
        "demo_length": i32(0x14C),
        "last_demo_update": i32(0x158),
        "needs_command": bool(data[0x15C]),
        "is_short_command": bool(data[0x15D]),
        "command_number": i32(0x160),
        "command_order": i32(0x164),
        "command_bit_position": i32(0x168),
        "demo_loading_complete": bool(data[0x16C]),
    }


def _read_base(handle: int, timeout: float) -> int:
    deadline = time.monotonic() + timeout
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        try:
            base = struct.unpack(
                "<I", read_process_bytes(handle, G_SEXY_APP_BASE_ADDRESS, 4)
            )[0]
            if base:
                return base
        except (OSError, RuntimeError) as error:
            last_error = error
        time.sleep(0.002)
    detail = f": {last_error}" if last_error is not None else ""
    raise TimeoutError(f"timed out waiting for gSexyAppBase{detail}")


def monitor_replay(
    *,
    steam_exe: Path,
    app_id: int,
    dmo: Path,
    runtime_exe: Path,
    expected_updates: int,
    launch_timeout: float,
    timeout: float,
    sample_interval: float,
    milestone: int,
) -> tuple[dict[str, object], bool]:
    """Launch one replay and return its observed progress plus success flag."""

    if not steam_exe.is_file():
        raise FileNotFoundError(f"Steam executable does not exist: {steam_exe}")
    if not dmo.is_file():
        raise FileNotFoundError(f"DMO does not exist: {dmo}")

    existing = set(process_ids_by_name(runtime_exe.name))
    if existing:
        raise RuntimeError(
            "refusing to launch with an existing PopCap runtime: "
            f"{sorted(existing)}"
        )

    command = [
        str(steam_exe),
        "-silent",
        "-applaunch",
        str(app_id),
        "-play",
        f"-demofile={dmo.resolve()}",
    ]
    launched = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
    )
    del launched
    pid, hwnd = wait_for_new_runtime_window(
        existing_pids=existing,
        runtime_exe=runtime_exe,
        timeout=launch_timeout,
    )

    handle = open_process_readonly(pid)
    started = time.monotonic()
    base = _read_base(handle, launch_timeout)
    max_update = -1
    max_bit_position = -1
    last_state: dict[str, int | float | bool] | None = None
    samples: list[dict[str, int | float | bool]] = []
    next_milestone = 0
    previous_loading: bool | None = None
    exit_observed = False
    timed_out = False
    read_error: str | None = None
    try:
        while True:
            elapsed = time.monotonic() - started
            if elapsed >= timeout:
                timed_out = True
                break
            try:
                raw = read_process_bytes(
                    handle,
                    base + STATE_START_OFFSET,
                    STATE_END_OFFSET - STATE_START_OFFSET,
                )
            except (OSError, RuntimeError) as error:
                read_error = str(error)
                exit_observed = (
                    pid not in set(process_ids_by_name(runtime_exe.name))
                )
                break

            state = _decode_state(raw)
            update = int(state["update"])
            bit_position = int(state["buffer_read_bit_position"])
            max_update = max(max_update, update)
            max_bit_position = max(max_bit_position, bit_position)
            loading = bool(state["demo_loading_complete"])
            should_sample = (
                not samples
                or update >= next_milestone
                or loading != previous_loading
            )
            if should_sample:
                samples.append({"elapsed_seconds": round(elapsed, 6), **state})
                if len(samples) > 256:
                    samples.pop(0)
                next_milestone = ((max(update, 0) // milestone) + 1) * milestone
                previous_loading = loading
            last_state = state
            time.sleep(sample_interval)
    finally:
        close_process(handle)

    success = max_update >= expected_updates - 2
    result: dict[str, object] = {
        "pid": pid,
        "hwnd": hwnd,
        "base": f"0x{base:08X}",
        "dmo": str(dmo.resolve()),
        "expected_updates": expected_updates,
        "max_update": max_update,
        "max_buffer_read_bit_position": max_bit_position,
        "exit_observed": exit_observed,
        "timed_out": timed_out,
        "read_error": read_error,
        "last_state": last_state,
        "samples": samples,
        "success": success,
    }
    return result, success


def _positive_float(text: str) -> float:
    value = float(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dmo", required=True, type=Path)
    parser.add_argument("--expected-updates", required=True, type=_positive_int)
    parser.add_argument("--steam-exe", type=Path, default=DEFAULT_STEAM_EXE)
    parser.add_argument("--runtime-exe", type=Path, default=DEFAULT_RUNTIME_EXE)
    parser.add_argument("--app-id", type=_positive_int, default=3620)
    parser.add_argument("--launch-timeout", type=_positive_float, default=20.0)
    parser.add_argument("--timeout", type=_positive_float, default=30.0)
    parser.add_argument("--sample-interval", type=_positive_float, default=0.002)
    parser.add_argument("--milestone", type=_positive_int, default=250)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result, success = monitor_replay(
            steam_exe=args.steam_exe,
            app_id=args.app_id,
            dmo=args.dmo,
            runtime_exe=args.runtime_exe,
            expected_updates=args.expected_updates,
            launch_timeout=args.launch_timeout,
            timeout=args.timeout,
            sample_interval=args.sample_interval,
            milestone=args.milestone,
        )
    except (OSError, RuntimeError, TimeoutError, ValueError) as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
