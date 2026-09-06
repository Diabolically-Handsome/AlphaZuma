"""Find the first live/offline command-boundary divergence in a PopCap DMO.

The offline scanner records every exact command start bit.  A path-verified
Steam replay is then sampled through read-only process access.  Stable live
``mDemoCmdBitPos`` states are checked against those starts and decoded command
headers.  The tool reports the first repeat-observed mismatch and surrounding
valid transitions; optional affinity and thread-priority controls affect only
Windows scheduling and never write target-process memory.
"""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import struct
import subprocess
import sys
import time
from typing import Iterable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.align_popcap_dmo_startup import (
    _command_stream_offset,
    _load_popcap_dmo_module,
    _scan_commands,
)
from tools.control_popcap_replay import expected_multiplier, post_char
from tools.inspect_popcap_replay import (
    close_process,
    open_process_readonly,
    read_process_bytes,
)
from tools.launch_popcap_replay import (
    DEFAULT_RUNTIME_EXE,
    DEFAULT_STEAM_EXE,
    process_ids_by_name,
    wait_for_pid_window,
)
from tools.monitor_popcap_replay import (
    G_SEXY_APP_BASE_ADDRESS,
    STATE_END_OFFSET,
    STATE_START_OFFSET,
    _decode_state,
    _read_base,
)
from tools.trace_popcap_demo_commands import wait_for_new_runtime


PROCESS_SET_INFORMATION = 0x0200
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
TH32CS_SNAPTHREAD = 0x00000004
THREAD_SET_INFORMATION = 0x0020
THREAD_QUERY_LIMITED_INFORMATION = 0x0800
THREAD_PRIORITY_ERROR_RETURN = 0x7FFFFFFF
THREAD_PRIORITY_LOWEST = -2
THREAD_PRIORITY_HIGHEST = 2
MAX_GATE_MULTIPLIER = 0.15
MAX_GATE_RECOVERY_BITS = 512
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
user32 = ctypes.WinDLL("user32", use_last_error=True)


class THREADENTRY32(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("cntUsage", wintypes.DWORD),
        ("th32ThreadID", wintypes.DWORD),
        ("th32OwnerProcessID", wintypes.DWORD),
        ("tpBasePri", wintypes.LONG),
        ("tpDeltaPri", wintypes.LONG),
        ("dwFlags", wintypes.DWORD),
    ]


kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
kernel32.OpenProcess.restype = wintypes.HANDLE
kernel32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
kernel32.SetProcessAffinityMask.restype = wintypes.BOOL
kernel32.GetProcessAffinityMask.argtypes = [
    wintypes.HANDLE,
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_size_t),
]
kernel32.GetProcessAffinityMask.restype = wintypes.BOOL
kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
kernel32.Thread32First.argtypes = [
    wintypes.HANDLE,
    ctypes.POINTER(THREADENTRY32),
]
kernel32.Thread32First.restype = wintypes.BOOL
kernel32.Thread32Next.argtypes = [
    wintypes.HANDLE,
    ctypes.POINTER(THREADENTRY32),
]
kernel32.Thread32Next.restype = wintypes.BOOL
kernel32.OpenThread.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
kernel32.OpenThread.restype = wintypes.HANDLE
kernel32.GetThreadPriority.argtypes = [wintypes.HANDLE]
kernel32.GetThreadPriority.restype = wintypes.INT
kernel32.SetThreadPriority.argtypes = [wintypes.HANDLE, wintypes.INT]
kernel32.SetThreadPriority.restype = wintypes.BOOL
kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
kernel32.CloseHandle.restype = wintypes.BOOL
user32.GetWindowThreadProcessId.argtypes = [
    wintypes.HWND,
    ctypes.POINTER(wintypes.DWORD),
]
user32.GetWindowThreadProcessId.restype = wintypes.DWORD


def _set_process_affinity(pid: int, mask: int) -> dict[str, int]:
    if mask <= 0:
        raise ValueError("affinity mask must be positive")
    handle = kernel32.OpenProcess(
        PROCESS_SET_INFORMATION | PROCESS_QUERY_LIMITED_INFORMATION,
        False,
        pid,
    )
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        previous_process = ctypes.c_size_t()
        system_mask = ctypes.c_size_t()
        if not kernel32.GetProcessAffinityMask(
            handle,
            ctypes.byref(previous_process),
            ctypes.byref(system_mask),
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        if mask & ~system_mask.value:
            raise ValueError(
                f"affinity mask 0x{mask:X} exceeds system mask "
                f"0x{system_mask.value:X}"
            )
        if not kernel32.SetProcessAffinityMask(handle, mask):
            raise ctypes.WinError(ctypes.get_last_error())
        observed_process = ctypes.c_size_t()
        observed_system = ctypes.c_size_t()
        if not kernel32.GetProcessAffinityMask(
            handle,
            ctypes.byref(observed_process),
            ctypes.byref(observed_system),
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        return {
            "previous_process_mask": previous_process.value,
            "requested_process_mask": mask,
            "observed_process_mask": observed_process.value,
            "system_mask": observed_system.value,
        }
    finally:
        kernel32.CloseHandle(handle)


def _thread_ids_for_pid(pid: int) -> tuple[int, ...]:
    snapshot = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0)
    invalid_handle = ctypes.c_void_p(-1).value
    if not snapshot or int(snapshot) == invalid_handle:
        raise ctypes.WinError(ctypes.get_last_error())
    found: list[int] = []
    try:
        entry = THREADENTRY32()
        entry.dwSize = ctypes.sizeof(entry)
        ok = kernel32.Thread32First(snapshot, ctypes.byref(entry))
        while ok:
            if int(entry.th32OwnerProcessID) == pid:
                found.append(int(entry.th32ThreadID))
            ok = kernel32.Thread32Next(snapshot, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snapshot)
    return tuple(sorted(found))


def _set_replay_thread_priority_bias(
    pid: int,
    hwnd: int,
) -> dict[str, object]:
    observed_pid = wintypes.DWORD()
    window_thread_id = int(
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(observed_pid))
    )
    if not window_thread_id or int(observed_pid.value) != pid:
        raise ctypes.WinError(ctypes.get_last_error())

    rows: list[dict[str, int | str]] = []
    main_applied = False
    for thread_id in _thread_ids_for_pid(pid):
        target = (
            THREAD_PRIORITY_LOWEST
            if thread_id == window_thread_id
            else THREAD_PRIORITY_HIGHEST
        )
        handle = kernel32.OpenThread(
            THREAD_SET_INFORMATION | THREAD_QUERY_LIMITED_INFORMATION,
            False,
            thread_id,
        )
        if not handle:
            rows.append(
                {
                    "thread_id": thread_id,
                    "target_priority": target,
                    "status": "thread_unavailable",
                }
            )
            continue
        try:
            previous = int(kernel32.GetThreadPriority(handle))
            if previous == THREAD_PRIORITY_ERROR_RETURN:
                rows.append(
                    {
                        "thread_id": thread_id,
                        "target_priority": target,
                        "status": "priority_query_failed",
                    }
                )
                continue
            if not kernel32.SetThreadPriority(handle, target):
                rows.append(
                    {
                        "thread_id": thread_id,
                        "previous_priority": previous,
                        "target_priority": target,
                        "status": "priority_set_failed",
                    }
                )
                continue
            observed = int(kernel32.GetThreadPriority(handle))
            rows.append(
                {
                    "thread_id": thread_id,
                    "previous_priority": previous,
                    "target_priority": target,
                    "observed_priority": observed,
                    "status": "applied",
                }
            )
            if thread_id == window_thread_id and observed == target:
                main_applied = True
        finally:
            kernel32.CloseHandle(handle)
    if not main_applied:
        raise RuntimeError("failed to bias the replay window thread priority")
    return {
        "window_thread_id": window_thread_id,
        "window_thread_priority": THREAD_PRIORITY_LOWEST,
        "other_thread_priority": THREAD_PRIORITY_HIGHEST,
        "threads": rows,
    }


def _offline_rows(dmo: Path) -> tuple[int, list[dict[str, object]]]:
    module = _load_popcap_dmo_module()
    data = dmo.read_bytes()
    offset = _command_stream_offset(data)
    length_updates = struct.unpack_from("<I", data, offset - 4)[0]
    rows, _ = _scan_commands(module, data[offset:], length_updates)
    return length_updates, rows


def _transition(
    *,
    state: dict[str, int | float | bool],
    row_by_start: dict[int, tuple[int, dict[str, object]]],
    elapsed: float,
) -> dict[str, object]:
    bit_position = int(state["command_bit_position"])
    match = row_by_start.get(bit_position)
    transition: dict[str, object] = {
        "elapsed_seconds": round(elapsed, 6),
        "live_update": int(state["update"]),
        "live_order": int(state["command_order"]),
        "live_command_bit_position": bit_position,
        "live_read_bit_position": int(state["buffer_read_bit_position"]),
        "live_short_form": bool(state["is_short_command"]),
        "live_command_number": int(state["command_number"]),
        "live_last_demo_update": int(state["last_demo_update"]),
        "live_needs_command": bool(state["needs_command"]),
    }
    if match is None:
        transition["boundary_match"] = False
        return transition

    index, row = match
    header_match = (
        bool(row["short_form"]) == bool(state["is_short_command"])
        and int(row["command_number"]) == int(state["command_number"])
    )
    transition.update(
        {
            "boundary_match": True,
            "header_match": header_match,
            "offline_index": index,
            "offline_update": int(row["update"]),
            "offline_kind": str(row["kind"]),
            "offline_end_bit_position": int(row["end"]),
        }
    )
    return transition


def compare_replay(
    *,
    steam_exe: Path,
    app_id: int,
    dmo: Path,
    runtime_exe: Path,
    launch_timeout: float,
    timeout: float,
    sample_interval: float,
    slow_at_update: int | None,
    minus_count: int,
    resume_after_bit_position: int | None,
    stop_after_update: int | None,
    affinity_mask: int | None,
    bias_replay_threads: bool,
    gate_bit_ranges: tuple[tuple[int, int], ...],
    gate_minus_count: int,
) -> dict[str, object]:
    if not steam_exe.is_file():
        raise FileNotFoundError(f"Steam executable does not exist: {steam_exe}")
    if not dmo.is_file():
        raise FileNotFoundError(f"DMO does not exist: {dmo}")
    if minus_count < 0:
        raise ValueError("minus_count must be non-negative")
    if (slow_at_update is None) != (minus_count == 0):
        raise ValueError(
            "slow_at_update and a positive minus_count must be used together"
        )
    if resume_after_bit_position is not None and minus_count == 0:
        raise ValueError("resume_after_bit_position requires slowdown controls")
    if gate_minus_count < 0:
        raise ValueError("gate_minus_count must be non-negative")
    if (
        gate_bit_ranges
        and expected_multiplier(gate_minus_count) > MAX_GATE_MULTIPLIER
    ):
        raise ValueError(
            "gate_minus_count must throttle updates at or below "
            f"{MAX_GATE_MULTIPLIER}"
        )

    length_updates, rows = _offline_rows(dmo)
    row_by_start = {
        int(row["start"]): (index, row) for index, row in enumerate(rows)
    }
    starts = sorted(row_by_start)

    existing = set(process_ids_by_name(runtime_exe.name))
    if existing:
        raise RuntimeError(
            "refusing to launch with an existing PopCap runtime: "
            f"{sorted(existing)}"
        )
    subprocess.Popen(
        [
            str(steam_exe),
            "-silent",
            "-applaunch",
            str(app_id),
            "-play",
            f"-demofile={dmo.resolve()}",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
    )
    pid = wait_for_new_runtime(
        existing_pids=existing,
        runtime_exe=runtime_exe,
        timeout=launch_timeout,
    )
    affinity = (
        _set_process_affinity(pid, affinity_mask)
        if affinity_mask is not None
        else None
    )
    hwnd = wait_for_pid_window(
        pid=pid,
        runtime_exe=runtime_exe,
        timeout=launch_timeout,
    )
    thread_priority_bias = (
        _set_replay_thread_priority_bias(pid, hwnd)
        if bias_replay_threads
        else None
    )

    handle = open_process_readonly(pid)
    base = _read_base(handle, launch_timeout)
    started = time.monotonic()
    previous_key: tuple[int, int, bool, int, bool] | None = None
    history: list[dict[str, object]] = []
    mismatch_candidate_key: tuple[int, int, bool, int, bool] | None = None
    mismatch_candidate: dict[str, object] | None = None
    mismatch_observations = 0
    first_mismatch: dict[str, object] | None = None
    history_at_mismatch: list[dict[str, object]] | None = None
    exit_observed = False
    timed_out = False
    stopped_at_update = False
    last_update = -1
    slowdown_sent = False
    resume_sent = False
    controls: list[dict[str, object]] = []
    gate_index = 0
    gate_active = False
    gate_armed: dict[str, object] | None = None
    gate_suppressed_samples = 0
    gate_recovery_until_bit_position: int | None = None
    gate_recovery_suppressed_samples = 0
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
            except (OSError, RuntimeError):
                exit_observed = (
                    pid not in set(process_ids_by_name(runtime_exe.name))
                )
                break
            state = _decode_state(raw)
            last_update = max(last_update, int(state["update"]))
            if gate_index < len(gate_bit_ranges):
                gate_start, gate_end = gate_bit_ranges[gate_index]
                live_read_position = int(
                    state["buffer_read_bit_position"]
                )
                live_command_position = int(
                    state["command_bit_position"]
                )
                live_row_match = row_by_start.get(live_command_position)
                live_row = (
                    live_row_match[1]
                    if live_row_match is not None
                    else None
                )
                gate_start_match = row_by_start.get(gate_start)
                gate_start_row = (
                    gate_start_match[1]
                    if gate_start_match is not None
                    else None
                )
                live_header_bits = (
                    4
                    if live_row is not None
                    and bool(live_row["short_form"])
                    else 10
                )
                live_header_matches = (
                    live_row is not None
                    and bool(live_row["short_form"])
                    == bool(state["is_short_command"])
                    and int(live_row["command_number"])
                    == int(state["command_number"])
                )
                exact_gate_arm = (
                    gate_start
                    <= live_command_position
                    < gate_end
                    and live_header_matches
                    and live_read_position
                    >= live_command_position + live_header_bits
                    and live_read_position <= int(live_row["end"])
                )
                gate_start_header_bits = (
                    4
                    if gate_start_row is not None
                    and bool(gate_start_row["short_form"])
                    else 10
                )
                range_fallback_gate_arm = (
                    gate_start_row is not None
                    and gate_start + gate_start_header_bits
                    <= live_read_position
                    < gate_end
                    and int(state["update"])
                    >= int(gate_start_row["update"])
                )
                if (
                    not gate_active
                    and gate_armed is None
                    and (exact_gate_arm or range_fallback_gate_arm)
                ):
                    armed_row = (
                        live_row if exact_gate_arm else gate_start_row
                    )
                    assert armed_row is not None
                    gate_armed = {
                        "arm_mode": (
                            "exact_header"
                            if exact_gate_arm
                            else "configured_range_fallback"
                        ),
                        "command_bit_position": (
                            live_command_position
                            if exact_gate_arm
                            else gate_start
                        ),
                        "observed_command_bit_position": (
                            live_command_position
                        ),
                        "command_kind": str(armed_row["kind"]),
                        "command_end_bit_position": int(armed_row["end"]),
                        "offline_update": int(armed_row["update"]),
                    }
                    controls.append(
                        {
                            "kind": "gate_arm",
                            "gate_index": gate_index,
                            "elapsed_seconds": round(elapsed, 6),
                            "update": int(state["update"]),
                            "read_bit_position": live_read_position,
                            **gate_armed,
                            "gate_end_bit_position": gate_end,
                        }
                    )
                if (
                    gate_armed is not None
                    and not gate_active
                    and int(state["update"])
                    >= int(gate_armed["offline_update"])
                ):
                    for _ in range(gate_minus_count):
                        post_char(hwnd, "-")
                    gate_active = True
                    controls.append(
                        {
                            "kind": "gate_hold",
                            "gate_index": gate_index,
                            "elapsed_seconds": round(elapsed, 6),
                            "update": int(state["update"]),
                            "read_bit_position": live_read_position,
                            **gate_armed,
                            "minus_count": gate_minus_count,
                            "target_multiplier": expected_multiplier(
                                gate_minus_count
                            ),
                            "gate_end_bit_position": gate_end,
                        }
                    )
                    gate_armed = None
                if gate_active and live_read_position >= gate_end:
                    recovery_start = max(
                        (
                            value
                            for value in starts
                            if value <= live_read_position
                        ),
                        default=None,
                    )
                    recovery_row_match = (
                        row_by_start.get(recovery_start)
                        if recovery_start is not None
                        else None
                    )
                    recovery_row = (
                        recovery_row_match[1]
                        if recovery_row_match is not None
                        else None
                    )
                    recovery_command_end = (
                        int(recovery_row["end"])
                        if recovery_row is not None
                        else live_read_position
                    )
                    gate_recovery_until_bit_position = min(
                        int(rows[-1]["end"]),
                        max(
                            recovery_command_end,
                            gate_end + MAX_GATE_RECOVERY_BITS,
                        ),
                    )
                    for _ in range(gate_minus_count):
                        post_char(hwnd, "+")
                    controls.append(
                        {
                            "kind": "gate_resume",
                            "gate_index": gate_index,
                            "elapsed_seconds": round(elapsed, 6),
                            "update": int(state["update"]),
                            "read_bit_position": live_read_position,
                            "plus_count": gate_minus_count,
                            "target_multiplier": 1.0,
                            "recovery_until_bit_position": (
                                gate_recovery_until_bit_position
                            ),
                        }
                    )
                    gate_active = False
                    gate_index += 1
            if (
                not slowdown_sent
                and slow_at_update is not None
                and int(state["update"]) >= slow_at_update
            ):
                for _ in range(minus_count):
                    post_char(hwnd, "-")
                slowdown_sent = True
                controls.append(
                    {
                        "kind": "slowdown",
                        "elapsed_seconds": round(elapsed, 6),
                        "update": int(state["update"]),
                        "read_bit_position": int(
                            state["buffer_read_bit_position"]
                        ),
                        "minus_count": minus_count,
                        "target_multiplier": expected_multiplier(minus_count),
                    }
                )
            if (
                slowdown_sent
                and not resume_sent
                and resume_after_bit_position is not None
                and int(state["buffer_read_bit_position"])
                >= resume_after_bit_position
            ):
                for _ in range(minus_count):
                    post_char(hwnd, "+")
                resume_sent = True
                controls.append(
                    {
                        "kind": "resume",
                        "elapsed_seconds": round(elapsed, 6),
                        "update": int(state["update"]),
                        "read_bit_position": int(
                            state["buffer_read_bit_position"]
                        ),
                        "plus_count": minus_count,
                        "target_multiplier": 1.0,
                    }
                )
            if int(state["demo_length"]) != length_updates:
                time.sleep(sample_interval)
                continue

            key = (
                int(state["command_order"]),
                int(state["command_bit_position"]),
                bool(state["is_short_command"]),
                int(state["command_number"]),
                bool(state["needs_command"]),
            )
            transition = _transition(
                state=state,
                row_by_start=row_by_start,
                elapsed=elapsed,
            )
            valid = bool(transition.get("boundary_match")) and bool(
                transition.get("header_match")
            )
            if (
                gate_recovery_until_bit_position is not None
                and int(state["buffer_read_bit_position"])
                > gate_recovery_until_bit_position
            ):
                gate_recovery_until_bit_position = None
            if gate_active and not valid:
                # Long service commands expose their partially consumed
                # payload as if it were a new command header while the game
                # waits for the asynchronous result.  A configured gate is
                # an explicit assertion that this interval is in-flight
                # service state, not a command boundary.  Keep the strict
                # mismatch detector armed for the first sample after the
                # gate, but do not condemn these transient payload bits.
                gate_suppressed_samples += 1
                mismatch_candidate_key = None
                mismatch_candidate = None
                mismatch_observations = 0
            elif (
                gate_recovery_until_bit_position is not None
                and not valid
            ):
                # Posting the resume keys can leave several queued window
                # messages while the parser consumes subsequent commands.
                # Recovery is bounded by MAX_GATE_RECOVERY_BITS; an invalid
                # stream that does not return to a verified boundary inside
                # that small horizon is still reported.
                gate_recovery_suppressed_samples += 1
                mismatch_candidate_key = None
                mismatch_candidate = None
                mismatch_observations = 0
            elif valid:
                gate_recovery_until_bit_position = None
                mismatch_candidate_key = None
                mismatch_candidate = None
                mismatch_observations = 0
                if key != previous_key and first_mismatch is None:
                    history.append(transition)
                    history = history[-16:]
                    previous_key = key
            elif key == mismatch_candidate_key:
                mismatch_observations += 1
                if mismatch_observations >= 2 and first_mismatch is None:
                    bit_position = int(state["command_bit_position"])
                    lower = max((value for value in starts if value < bit_position), default=None)
                    upper = min((value for value in starts if value > bit_position), default=None)
                    first_mismatch = {
                        **(mismatch_candidate or transition),
                        "repeat_observations": mismatch_observations,
                        "nearest_lower_offline_start": lower,
                        "nearest_upper_offline_start": upper,
                    }
                    history_at_mismatch = list(history)
            else:
                mismatch_candidate_key = key
                mismatch_candidate = transition
                mismatch_observations = 1
            if (
                stop_after_update is not None
                and int(state["update"]) >= stop_after_update
            ):
                stopped_at_update = True
                break
            time.sleep(sample_interval)
    finally:
        close_process(handle)

    return {
        "pid": pid,
        "hwnd": hwnd,
        "base": f"0x{base:08X}",
        "dmo": str(dmo.resolve()),
        "offline_commands": len(rows),
        "length_updates": length_updates,
        "affinity": affinity,
        "thread_priority_bias": thread_priority_bias,
        "last_live_update": last_update,
        "exit_observed": exit_observed,
        "timed_out": timed_out,
        "stopped_at_update": stopped_at_update,
        "controls": controls,
        "gate_suppressed_samples": gate_suppressed_samples,
        "gate_recovery_suppressed_samples": (
            gate_recovery_suppressed_samples
        ),
        "completed_gate_ranges": gate_index,
        "configured_gate_ranges": [
            {"start": start, "end": end}
            for start, end in gate_bit_ranges
        ],
        "first_mismatch": first_mismatch,
        "preceding_valid_transitions": history_at_mismatch or history,
    }


def _positive_float(text: str) -> float:
    value = float(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _non_negative_int(text: str) -> int:
    value = int(text, 0)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return value


def _bit_range(text: str) -> tuple[int, int]:
    try:
        start_text, end_text = text.split(":", 1)
        start = int(start_text, 0)
        end = int(end_text, 0)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "bit range must be START:END"
        ) from error
    if start < 0 or end <= start:
        raise argparse.ArgumentTypeError(
            "bit range must satisfy 0 <= START < END"
        )
    return start, end


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dmo", required=True, type=Path)
    parser.add_argument("--steam-exe", type=Path, default=DEFAULT_STEAM_EXE)
    parser.add_argument("--runtime-exe", type=Path, default=DEFAULT_RUNTIME_EXE)
    parser.add_argument("--app-id", type=int, default=3620)
    parser.add_argument("--launch-timeout", type=_positive_float, default=20.0)
    parser.add_argument("--timeout", type=_positive_float, default=30.0)
    parser.add_argument("--sample-interval", type=_positive_float, default=0.001)
    parser.add_argument("--slow-at-update", type=_non_negative_int)
    parser.add_argument("--minus-count", type=_non_negative_int, default=0)
    parser.add_argument(
        "--resume-after-bit-position",
        type=_non_negative_int,
    )
    parser.add_argument("--stop-after-update", type=_non_negative_int)
    parser.add_argument("--affinity-mask", type=_non_negative_int)
    parser.add_argument(
        "--bias-replay-threads",
        action="store_true",
        help=(
            "Lower the window thread and raise other replay threads; use "
            "with single-core affinity to test service-command scheduling."
        ),
    )
    parser.add_argument(
        "--gate-bit-range",
        type=_bit_range,
        action="append",
        default=[],
        metavar="START:END",
    )
    parser.add_argument(
        "--gate-minus-count",
        type=_non_negative_int,
        default=6,
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = compare_replay(
            steam_exe=args.steam_exe,
            app_id=args.app_id,
            dmo=args.dmo,
            runtime_exe=args.runtime_exe,
            launch_timeout=args.launch_timeout,
            timeout=args.timeout,
            sample_interval=args.sample_interval,
            slow_at_update=args.slow_at_update,
            minus_count=args.minus_count,
            resume_after_bit_position=args.resume_after_bit_position,
            stop_after_update=args.stop_after_update,
            affinity_mask=args.affinity_mask,
            bias_replay_threads=args.bias_replay_threads,
            gate_bit_ranges=tuple(args.gate_bit_range),
            gate_minus_count=args.gate_minus_count,
        )
    except (OSError, RuntimeError, TimeoutError, ValueError) as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["first_mismatch"] is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
