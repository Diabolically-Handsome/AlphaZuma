"""Deterministic control of the retail PopCap DMO replay interface.

The controller sends only documented demo-window messages and opens the
process with read-only memory rights to observe framework counters.  It never
writes target memory.
"""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import math
import os
from pathlib import Path
import sys
import time
from typing import Iterable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.inspect_popcap_replay import (
    ReplayState,
    close_process,
    locate_exact_float64,
    open_process_readonly,
    read_process_bytes,
    read_replay_state,
)


WM_CHAR = 0x0102
WM_COMMAND = 0x0111
BN_CLICKED = 0
IDOK = 1


if os.name == "nt":
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    EnumWindowsProc = ctypes.WINFUNCTYPE(
        wintypes.BOOL, wintypes.HWND, wintypes.LPARAM
    )
    user32.EnumWindows.argtypes = [EnumWindowsProc, wintypes.LPARAM]
    user32.EnumWindows.restype = wintypes.BOOL
    user32.GetWindowThreadProcessId.argtypes = [
        wintypes.HWND,
        ctypes.POINTER(wintypes.DWORD),
    ]
    user32.GetWindowThreadProcessId.restype = wintypes.DWORD
    user32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetClassNameW.restype = ctypes.c_int
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetWindowTextW.restype = ctypes.c_int
    user32.PostMessageW.argtypes = [
        wintypes.HWND,
        wintypes.UINT,
        wintypes.WPARAM,
        wintypes.LPARAM,
    ]
    user32.PostMessageW.restype = wintypes.BOOL
    user32.SetDlgItemTextA.argtypes = [
        wintypes.HWND,
        ctypes.c_int,
        wintypes.LPCSTR,
    ]
    user32.SetDlgItemTextA.restype = wintypes.BOOL
    user32.GetDlgItemTextA.argtypes = [
        wintypes.HWND,
        ctypes.c_int,
        wintypes.LPSTR,
        ctypes.c_int,
    ]
    user32.GetDlgItemTextA.restype = wintypes.UINT
    user32.IsWindow.argtypes = [wintypes.HWND]
    user32.IsWindow.restype = wintypes.BOOL


def _window_text(hwnd: int) -> str:
    buffer = ctypes.create_unicode_buffer(512)
    user32.GetWindowTextW(hwnd, buffer, len(buffer))
    return buffer.value


def _window_class(hwnd: int) -> str:
    buffer = ctypes.create_unicode_buffer(128)
    user32.GetClassNameW(hwnd, buffer, len(buffer))
    return buffer.value


def windows_for_pid(pid: int) -> tuple[tuple[int, str, str], ...]:
    if os.name != "nt":
        raise RuntimeError("live replay control requires Windows")
    found: list[tuple[int, str, str]] = []

    @EnumWindowsProc
    def callback(hwnd: int, _: int) -> bool:
        owner = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value == pid:
            found.append((int(hwnd), _window_class(hwnd), _window_text(hwnd)))
        return True

    if not user32.EnumWindows(callback, 0):
        raise ctypes.WinError(ctypes.get_last_error())
    return tuple(found)


def main_window_for_pid(pid: int) -> int:
    candidates = [
        hwnd
        for hwnd, class_name, title in windows_for_pid(pid)
        if class_name == "MainWindow" and title.startswith("Zuma's Revenge!")
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"expected one replay MainWindow, found {len(candidates)}")
    return candidates[0]


def dialog_for_pid(pid: int, title: str) -> int | None:
    candidates = [
        hwnd
        for hwnd, class_name, window_title in windows_for_pid(pid)
        if class_name == "#32770" and window_title == title
    ]
    if len(candidates) > 1:
        raise RuntimeError(f"multiple {title!r} dialogs found")
    return candidates[0] if candidates else None


def post_char(hwnd: int, character: str) -> None:
    if len(character) != 1:
        raise ValueError("character must have length one")
    if not user32.PostMessageW(hwnd, WM_CHAR, ord(character), 0):
        raise ctypes.WinError(ctypes.get_last_error())


def expected_multiplier(minus_count: int) -> float:
    if minus_count < 0:
        raise ValueError("minus_count must be non-negative")
    value = 1.0
    for _ in range(minus_count):
        value /= 1.5
    return value


def locate_replay_state(
    pid: int,
    minus_count: int,
    demo_length: int | None = None,
) -> tuple[int, int, ReplayState]:
    handle = open_process_readonly(pid)
    try:
        addresses = locate_exact_float64(handle, expected_multiplier(minus_count))
        candidates: list[tuple[int, ReplayState]] = []
        for address in addresses:
            try:
                state = read_replay_state(handle, address)
                observed_demo_length = int.from_bytes(
                    read_process_bytes(handle, address + 320, 4),
                    "little",
                    signed=True,
                )
            except RuntimeError:
                continue
            if (
                state.multiplier_address == address
                and state.update_multiplier == expected_multiplier(minus_count)
                and state.frame_time_ms == 10
                and state.non_draw_count >= 0
                and state.sleep_count >= 0
                and state.draw_count >= 0
                and state.update_count >= 0
                and 0 <= state.update_app_state <= 3
                and 0 <= state.update_app_depth <= 64
                and state.fast_forward_target >= 0
                and (
                    demo_length is None
                    or observed_demo_length == demo_length
                )
            ):
                candidates.append((address, state))
        if len(candidates) != 1:
            raise RuntimeError(
                "expected one plausible replay state, "
                f"found {len(candidates)} from {len(addresses)} multiplier matches"
            )
        address, state = candidates[0]
        return handle, address, state
    except BaseException:
        close_process(handle)
        raise


def wait_for_update(
    handle: int,
    multiplier_address: int,
    target: int,
    *,
    timeout: float,
) -> ReplayState:
    """Wait for the exact counter value, without claiming update completion.

    ``mUpdateCount`` advances inside ``DoUpdateFrames`` before the rest of the
    retail frame has necessarily finished.  Callers that need a coherent
    post-step sample must use :func:`wait_for_replay_step` instead.
    """

    deadline = time.monotonic() + timeout
    while True:
        state = read_replay_state(handle, multiplier_address)
        if state.update_count >= target:
            if state.update_count != target:
                raise RuntimeError(
                    f"update overshot target {target}: {state.update_count}"
                )
            return state
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"timed out at update {state.update_count}, target {target}"
            )
        time.sleep(0.002)


def wait_for_replay_step(
    handle: int,
    multiplier_address: int,
    target: int,
    *,
    timeout: float,
    stable_reads: int = 2,
) -> ReplayState:
    """Wait until an ``N`` replay step crosses its post-update barrier.

    The retail ``N`` command sets ``mFastForwardStep`` before calling
    ``DoUpdateFrames`` and clears it only after ``DoUpdateFramesF`` and the
    safe-delete pass complete.  The update counter alone is therefore an
    early, unsafe signal; the cleared step flag is the observable barrier.
    ``mUpdateAppDepth`` commonly remains one while the frozen main loop idles,
    so it is deliberately not used as a quiescence condition.
    """

    if stable_reads < 1:
        raise ValueError("stable_reads must be positive")
    deadline = time.monotonic() + timeout
    consecutive = 0
    last_state: ReplayState | None = None
    while True:
        state = read_replay_state(handle, multiplier_address)
        if state.update_count > target:
            raise RuntimeError(
                f"update overshot step target {target}: {state.update_count}"
            )
        complete = (
            state.update_count == target
            and state.fast_forward_target == target
            and not state.fast_forward_to_marker
            and not state.fast_forward_step
        )
        if complete:
            consecutive += 1
            last_state = state
            if consecutive >= stable_reads:
                return last_state
        else:
            consecutive = 0
            last_state = None
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "timed out waiting for replay-step barrier "
                f"at update {state.update_count}, target {target}, "
                f"fast_forward_target={state.fast_forward_target}, "
                f"fast_forward_to_marker={state.fast_forward_to_marker}, "
                f"fast_forward_step={state.fast_forward_step}"
            )
        time.sleep(0.001)


def step_to(
    hwnd: int,
    handle: int,
    multiplier_address: int,
    target: int,
    *,
    timeout_per_step: float,
) -> ReplayState:
    state = read_replay_state(handle, multiplier_address)
    if target < state.update_count:
        raise ValueError("step target is behind the current update")
    while state.update_count < target:
        next_update = state.update_count + 1
        post_char(hwnd, "N")
        state = wait_for_replay_step(
            handle,
            multiplier_address,
            next_update,
            timeout=timeout_per_step,
        )
    return state


def wait_loaded(
    handle: int,
    multiplier_address: int,
    *,
    timeout: float,
    stable_seconds: float = 0.5,
) -> ReplayState:
    deadline = time.monotonic() + timeout
    stable_since: float | None = None
    stable_update: int | None = None
    while True:
        state = read_replay_state(handle, multiplier_address)
        now = time.monotonic()
        if state.loaded and state.loading_thread_completed:
            if stable_update == state.update_count:
                if stable_since is not None and now - stable_since >= stable_seconds:
                    return state
            else:
                stable_update = state.update_count
                stable_since = now
        else:
            stable_update = None
            stable_since = None
        if now >= deadline:
            raise TimeoutError(
                "timed out waiting for a stable loaded replay state "
                f"at update {state.update_count}"
            )
        time.sleep(0.005)


def freeze_at(
    hwnd: int,
    handle: int,
    multiplier_address: int,
    target: int,
    *,
    minus_count: int,
    timeout: float,
) -> ReplayState:
    """Cross the framework's 0.1 multiplier threshold at a chosen update."""

    before_multiplier = expected_multiplier(minus_count)
    after_multiplier = expected_multiplier(minus_count + 1)
    if not (before_multiplier > 0.1 >= after_multiplier):
        raise ValueError(
            "freeze_at requires a minus count that crosses the 0.1 threshold"
        )
    state = read_replay_state(handle, multiplier_address)
    if state.update_multiplier != before_multiplier:
        raise RuntimeError("current multiplier does not match --minus-count")
    if target < state.update_count:
        raise ValueError("freeze target is behind the current update")

    deadline = time.monotonic() + timeout
    while state.update_count < target:
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"timed out at update {state.update_count}, freeze target {target}"
            )
        time.sleep(0.002)
        state = read_replay_state(handle, multiplier_address)

    post_char(hwnd, "-")
    while True:
        state = read_replay_state(handle, multiplier_address)
        if state.update_multiplier == after_multiplier:
            if state.update_count != target:
                raise RuntimeError(
                    "freeze barrier crossed at the wrong update: "
                    f"target {target}, observed {state.update_count}"
                )
            if state.fast_forward_step:
                raise RuntimeError(
                    "freeze barrier crossed during an unfinished replay step"
                )
            return state
        if time.monotonic() >= deadline:
            raise TimeoutError("timed out waiting for freeze multiplier")
        time.sleep(0.002)


def jump_to_remaining_seconds(
    *,
    pid: int,
    hwnd: int,
    handle: int,
    multiplier_address: int,
    seconds: int,
    demo_length: int,
    dialog_timeout: float,
    jump_timeout: float,
) -> ReplayState:
    if seconds < 0 or demo_length <= 0:
        raise ValueError("seconds must be non-negative and demo_length positive")
    before = read_replay_state(handle, multiplier_address)
    if before.frame_time_ms <= 0:
        raise RuntimeError("invalid frame time")
    target = demo_length - ((seconds + 1) * 1000 // before.frame_time_ms)
    if target <= before.update_count:
        raise ValueError(
            f"jump target {target} is not ahead of update {before.update_count}"
        )

    post_char(hwnd, "J")
    deadline = time.monotonic() + dialog_timeout
    dialog = None
    while dialog is None:
        dialog = dialog_for_pid(pid, "Jump To Time")
        if time.monotonic() >= deadline:
            raise TimeoutError("Jump To Time dialog did not appear")
        if dialog is None:
            time.sleep(0.01)

    value = str(seconds).encode("ascii")
    if not user32.SetDlgItemTextA(dialog, 100, value):
        raise ctypes.WinError(ctypes.get_last_error())
    readback = ctypes.create_string_buffer(64)
    user32.GetDlgItemTextA(dialog, 100, readback, len(readback))
    if readback.value != value:
        raise RuntimeError(
            f"jump dialog readback mismatch: {readback.value!r} != {value!r}"
        )
    wparam = IDOK | (BN_CLICKED << 16)
    if not user32.PostMessageW(dialog, WM_COMMAND, wparam, 0):
        raise ctypes.WinError(ctypes.get_last_error())

    return wait_for_update(
        handle,
        multiplier_address,
        target,
        timeout=jump_timeout,
    )


def format_state(state: ReplayState) -> str:
    return (
        f"multiplier_address=0x{state.multiplier_address:X}\n"
        f"frame_time_ms={state.frame_time_ms}\n"
        f"update_multiplier={state.update_multiplier!r}\n"
        f"update_count={state.update_count}\n"
        f"draw_count={state.draw_count}\n"
        f"sleep_count={state.sleep_count}\n"
        f"paused={state.paused}\n"
        f"fast_forward_target={state.fast_forward_target}\n"
        f"fast_forward_to_marker={state.fast_forward_to_marker}\n"
        f"fast_forward_step={state.fast_forward_step}\n"
        f"step_mode={state.step_mode}\n"
        f"update_app_state={state.update_app_state}\n"
        f"update_app_depth={state.update_app_depth}\n"
        f"loading_thread_started={state.loading_thread_started}\n"
        f"loading_thread_completed={state.loading_thread_completed}\n"
        f"loaded={state.loaded}"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step or jump a slowed PopCap DMO replay deterministically."
    )
    parser.add_argument("--pid", required=True, type=int)
    parser.add_argument("--minus-count", required=True, type=int)
    parser.add_argument("--step-to", type=int)
    parser.add_argument("--wait-loaded", action="store_true")
    parser.add_argument("--freeze-at", type=int)
    parser.add_argument("--jump-seconds-remaining", type=int)
    parser.add_argument("--demo-length", type=int)
    parser.add_argument("--timeout-per-step", type=float, default=2.0)
    parser.add_argument("--loaded-timeout", type=float, default=120.0)
    parser.add_argument("--freeze-timeout", type=float, default=180.0)
    parser.add_argument("--dialog-timeout", type=float, default=5.0)
    parser.add_argument("--jump-timeout", type=float, default=60.0)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.jump_seconds_remaining is not None and args.demo_length is None:
        raise SystemExit("--demo-length is required for a jump")
    hwnd = main_window_for_pid(args.pid)
    handle, address, state = locate_replay_state(
        args.pid,
        args.minus_count,
        args.demo_length,
    )
    try:
        if args.wait_loaded:
            state = wait_loaded(
                handle,
                address,
                timeout=args.loaded_timeout,
            )
        if args.freeze_at is not None:
            state = freeze_at(
                hwnd,
                handle,
                address,
                args.freeze_at,
                minus_count=args.minus_count,
                timeout=args.freeze_timeout,
            )
        if args.step_to is not None:
            state = step_to(
                hwnd,
                handle,
                address,
                args.step_to,
                timeout_per_step=args.timeout_per_step,
            )
        if args.jump_seconds_remaining is not None:
            state = jump_to_remaining_seconds(
                pid=args.pid,
                hwnd=hwnd,
                handle=handle,
                multiplier_address=address,
                seconds=args.jump_seconds_remaining,
                demo_length=args.demo_length,
                dialog_timeout=args.dialog_timeout,
                jump_timeout=args.jump_timeout,
            )
        print(format_state(state))
    finally:
        close_process(handle)
    return 0


if __name__ == "__main__":
    sys.exit(main())
