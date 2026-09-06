"""Freeze a live PopCap DMO replay on one exact framework update.

The game exposes ``-`` as a documented replay-speed control.  Six presses
cross the framework's 0.1 update-multiplier threshold and stop autonomous
updates.  This helper waits at full speed until shortly before the requested
tick, applies the first five presses, and uses the sixth press exactly at the
target tick.  It only posts window messages and reads process memory.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Iterable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.control_popcap_replay import (
    expected_multiplier,
    freeze_at,
    main_window_for_pid,
    post_char,
    wait_for_update,
)
from tools.inspect_popcap_replay import (
    ReplayState,
    close_process,
    open_process_readonly,
    read_process_bytes,
    read_replay_state,
)


G_SEXY_APP_BASE_ADDRESS = 0x009FC740
UPDATE_MULTIPLIER_OFFSET = 0x4D0
DEMO_LENGTH_FROM_MULTIPLIER = 320


def _locate_replay_state_fast(
    pid: int,
    *,
    minus_count: int,
    demo_length: int | None,
) -> tuple[int, int, ReplayState]:
    """Resolve the fixed retail SexyApp fields without a process-wide scan."""

    handle = open_process_readonly(pid)
    try:
        base = int.from_bytes(
            read_process_bytes(handle, G_SEXY_APP_BASE_ADDRESS, 4),
            "little",
        )
        if not base:
            raise RuntimeError("gSexyAppBase is unavailable")
        address = base + UPDATE_MULTIPLIER_OFFSET
        state = read_replay_state(handle, address)
        observed_demo_length = int.from_bytes(
            read_process_bytes(
                handle,
                address + DEMO_LENGTH_FROM_MULTIPLIER,
                4,
            ),
            "little",
            signed=True,
        )
        if (
            state.update_multiplier != expected_multiplier(minus_count)
            or state.frame_time_ms != 10
            or (
                demo_length is not None
                and observed_demo_length != demo_length
            )
        ):
            raise RuntimeError("fixed retail replay state validation failed")
        return handle, address, state
    except BaseException:
        close_process(handle)
        raise


def _wait_for_multiplier(
    handle: int,
    multiplier_address: int,
    expected: float,
    *,
    timeout: float,
) -> None:
    deadline = time.monotonic() + timeout
    while True:
        state = read_replay_state(handle, multiplier_address)
        if state.update_multiplier == expected:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "timed out waiting for update multiplier "
                f"{expected!r}; observed {state.update_multiplier!r}"
            )
        time.sleep(0.002)


def freeze_replay(
    *,
    pid: int,
    slowdown_at: int,
    freeze_update: int,
    demo_length: int | None,
    timeout: float,
    allow_slowdown_overshoot: bool = False,
) -> ReplayState:
    if slowdown_at < 0 or freeze_update <= slowdown_at:
        raise ValueError("freeze_update must be after slowdown_at")
    hwnd = main_window_for_pid(pid)
    handle, address, state = _locate_replay_state_fast(
        pid,
        minus_count=0,
        demo_length=demo_length,
    )
    try:
        if state.update_count > slowdown_at:
            raise RuntimeError(
                f"replay already passed slowdown tick {slowdown_at}: "
                f"{state.update_count}"
            )
        if allow_slowdown_overshoot:
            deadline = time.monotonic() + timeout
            while state.update_count < slowdown_at:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "timed out before replay slowdown"
                    )
                time.sleep(0.002)
                state = read_replay_state(handle, address)
            if state.update_count >= freeze_update:
                raise RuntimeError(
                    "replay passed freeze tick before slowdown"
                )
        else:
            state = wait_for_update(
                handle,
                address,
                slowdown_at,
                timeout=timeout,
            )
        for minus_count in range(1, 6):
            post_char(hwnd, "-")
            _wait_for_multiplier(
                handle,
                address,
                expected_multiplier(minus_count),
                timeout=5.0,
            )
        state = freeze_at(
            hwnd,
            handle,
            address,
            freeze_update,
            minus_count=5,
            timeout=timeout,
        )
        print(
            f"pid={pid} multiplier_address=0x{address:X} "
            f"update={state.update_count} "
            f"multiplier={state.update_multiplier!r} "
            f"draw_count={state.draw_count}"
        )
        return state
    finally:
        close_process(handle)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--slowdown-at", type=int, required=True)
    parser.add_argument("--freeze-at", type=int, required=True)
    parser.add_argument("--demo-length", type=int)
    parser.add_argument("--timeout", type=float, default=180.0)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    freeze_replay(
        pid=args.pid,
        slowdown_at=args.slowdown_at,
        freeze_update=args.freeze_at,
        demo_length=args.demo_length,
        timeout=args.timeout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
