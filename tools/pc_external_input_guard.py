"""Privacy-preserving Windows keyboard/mouse contamination monitor.

The monitor records only device class, coarse action, injected status and a
monotonic timestamp.  It never records key identities, text, pointer
coordinates, wheel deltas or button identities.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Mapping

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from zuma_rl.pc_external_input import (
    EXTERNAL_INPUT_GUARD_SCHEMA,
    EXTERNAL_INPUT_GUARD_VERSION,
    EXTERNAL_INPUT_POLICY,
    EXTERNAL_INPUT_PRIVACY,
    REPAINT_INPUT_EXTRA_INFO,
    SELF_TEST_INPUT_EXTRA_INFO,
    validate_external_input_guard_receipt,
)


MAX_RECORDED_EXTERNAL_EVENTS = 64
HOOK_READY_TIMEOUT_SECONDS = 5.0
SELF_TEST_TIMEOUT_SECONDS = 2.0
THREAD_STOP_TIMEOUT_SECONDS = 5.0
START_QUIESCENCE_SECONDS = 0.5
START_QUIESCENCE_TIMEOUT_SECONDS = 5.0
START_QUIESCENCE_POLL_SECONDS = 0.01


class ExternalInputGuardError(RuntimeError):
    """The guard could not establish or prove complete input coverage."""


def _require_windows() -> None:
    if os.name != "nt":
        raise ExternalInputGuardError("windows_required")


class ExternalInputGuard:
    """Monitor all session keyboard/mouse events over one critical interval."""

    def __init__(self) -> None:
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._thread_id: int | None = None
        self._thread_error: str | None = None
        self._callback_error: str | None = None
        self._keyboard_installed = False
        self._mouse_installed = False
        self._keyboard_unhooked = False
        self._mouse_unhooked = False
        self._coverage_start_ns: int | None = None
        self._coverage_end_ns: int | None = None
        self._finished_receipt: Mapping[str, Any] | None = None
        self._counts = {
            "external": 0,
            "allowed_repaint": 0,
            "self_test": 0,
        }
        self._external_events: list[dict[str, Any]] = []
        self._self_test_counts = {
            "start": {"keyboard": 0, "mouse": 0},
            "end": {"keyboard": 0, "mouse": 0},
        }
        self._active_self_test: str | None = None
        self._self_tests: list[dict[str, Any]] = []
        self._state_lock = threading.Lock()
        self._precoverage_event_count = 0
        self._precoverage_last_activity_ns: int | None = None

    @property
    def coverage_start_perf_counter_ns(self) -> int:
        if self._coverage_start_ns is None:
            raise ExternalInputGuardError("guard_not_started")
        return self._coverage_start_ns

    @property
    def finished(self) -> bool:
        return self._finished_receipt is not None

    def _record(
        self,
        *,
        device: str,
        action: str,
        flags: int,
        extra_info: int,
        injected_flag: int,
    ) -> None:
        try:
            observed_ns = time.perf_counter_ns()
            injected = bool(flags & injected_flag)
            with self._state_lock:
                if injected and extra_info == SELF_TEST_INPUT_EXTRA_INFO:
                    self._counts["self_test"] += 1
                    label = self._active_self_test
                    if label in self._self_test_counts:
                        self._self_test_counts[label][device] += 1
                    return
                if (
                    device == "mouse"
                    and injected
                    and extra_info == REPAINT_INPUT_EXTRA_INFO
                ):
                    start = self._coverage_start_ns
                    end = self._coverage_end_ns
                    if start is not None and observed_ns >= start and (
                        end is None or observed_ns <= end
                    ):
                        self._counts["allowed_repaint"] += 1
                        return
                start = self._coverage_start_ns
                end = self._coverage_end_ns
                if start is None:
                    self._precoverage_event_count += 1
                    self._precoverage_last_activity_ns = observed_ns
                    return
                if observed_ns < start or (
                    end is not None and observed_ns > end
                ):
                    return
                self._counts["external"] += 1
                if len(self._external_events) < MAX_RECORDED_EXTERNAL_EVENTS:
                    self._external_events.append(
                        {
                            "perf_counter_ns": observed_ns,
                            "device": device,
                            "action": action,
                            "injected": injected,
                        }
                    )
        except Exception as error:  # pragma: no cover - callback fail-closed
            self._callback_error = type(error).__name__

    def _thread_main(self) -> None:  # pragma: no cover - Windows integration
        keyboard_hook = None
        mouse_hook = None
        try:
            user32 = ctypes.WinDLL("user32", use_last_error=True)
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            hook_proc = ctypes.WINFUNCTYPE(
                ctypes.c_ssize_t,
                ctypes.c_int,
                wintypes.WPARAM,
                wintypes.LPARAM,
            )

            class KeyboardEvent(ctypes.Structure):
                _fields_ = (
                    ("vk_code", wintypes.DWORD),
                    ("scan_code", wintypes.DWORD),
                    ("flags", wintypes.DWORD),
                    ("time", wintypes.DWORD),
                    ("extra_info", ctypes.c_size_t),
                )

            class MouseEvent(ctypes.Structure):
                _fields_ = (
                    ("point", wintypes.POINT),
                    ("mouse_data", wintypes.DWORD),
                    ("flags", wintypes.DWORD),
                    ("time", wintypes.DWORD),
                    ("extra_info", ctypes.c_size_t),
                )

            user32.SetWindowsHookExW.argtypes = (
                ctypes.c_int,
                hook_proc,
                wintypes.HINSTANCE,
                wintypes.DWORD,
            )
            user32.SetWindowsHookExW.restype = wintypes.HANDLE
            user32.CallNextHookEx.argtypes = (
                wintypes.HANDLE,
                ctypes.c_int,
                wintypes.WPARAM,
                wintypes.LPARAM,
            )
            user32.CallNextHookEx.restype = ctypes.c_ssize_t
            user32.UnhookWindowsHookEx.argtypes = (wintypes.HANDLE,)
            user32.UnhookWindowsHookEx.restype = wintypes.BOOL
            user32.GetMessageW.argtypes = (
                ctypes.POINTER(wintypes.MSG),
                wintypes.HWND,
                wintypes.UINT,
                wintypes.UINT,
            )
            user32.GetMessageW.restype = wintypes.BOOL
            kernel32.GetModuleHandleW.argtypes = (wintypes.LPCWSTR,)
            kernel32.GetModuleHandleW.restype = wintypes.HMODULE
            kernel32.GetCurrentThreadId.argtypes = ()
            kernel32.GetCurrentThreadId.restype = wintypes.DWORD

            keyboard_actions = {
                0x0100: "down",
                0x0101: "up",
                0x0104: "system_down",
                0x0105: "system_up",
            }
            mouse_actions = {
                0x0200: "move",
                0x0201: "button_down",
                0x0202: "button_up",
                0x0203: "button_double",
                0x0204: "button_down",
                0x0205: "button_up",
                0x0206: "button_double",
                0x0207: "button_down",
                0x0208: "button_up",
                0x0209: "button_double",
                0x020A: "wheel",
                0x020B: "button_down",
                0x020C: "button_up",
                0x020D: "button_double",
                0x020E: "horizontal_wheel",
            }

            @hook_proc
            def keyboard_callback(
                code: int,
                message: int,
                data_pointer: int,
            ) -> int:
                if code >= 0:
                    data = ctypes.cast(
                        data_pointer,
                        ctypes.POINTER(KeyboardEvent),
                    ).contents
                    self._record(
                        device="keyboard",
                        action=keyboard_actions.get(int(message), "other"),
                        flags=int(data.flags),
                        extra_info=int(data.extra_info),
                        injected_flag=0x10,
                    )
                return int(
                    user32.CallNextHookEx(
                        keyboard_hook,
                        code,
                        message,
                        data_pointer,
                    )
                )

            @hook_proc
            def mouse_callback(
                code: int,
                message: int,
                data_pointer: int,
            ) -> int:
                if code >= 0:
                    data = ctypes.cast(
                        data_pointer,
                        ctypes.POINTER(MouseEvent),
                    ).contents
                    self._record(
                        device="mouse",
                        action=mouse_actions.get(int(message), "other"),
                        flags=int(data.flags),
                        extra_info=int(data.extra_info),
                        injected_flag=0x01,
                    )
                return int(
                    user32.CallNextHookEx(
                        mouse_hook,
                        code,
                        message,
                        data_pointer,
                    )
                )

            module = kernel32.GetModuleHandleW(None)
            self._thread_id = int(kernel32.GetCurrentThreadId())
            keyboard_hook = user32.SetWindowsHookExW(
                13, keyboard_callback, module, 0
            )
            mouse_hook = user32.SetWindowsHookExW(
                14, mouse_callback, module, 0
            )
            self._keyboard_installed = bool(keyboard_hook)
            self._mouse_installed = bool(mouse_hook)
            if not keyboard_hook or not mouse_hook:
                self._thread_error = "hook_install_failed"
                return
            self._ready.set()
            message = wintypes.MSG()
            while True:
                result = int(
                    user32.GetMessageW(
                        ctypes.byref(message), None, 0, 0
                    )
                )
                if result == 0:
                    break
                if result < 0:
                    self._thread_error = "hook_message_loop_failed"
                    break
        except Exception as error:
            self._thread_error = type(error).__name__
        finally:
            try:
                if keyboard_hook:
                    self._keyboard_unhooked = bool(
                        user32.UnhookWindowsHookEx(keyboard_hook)
                    )
                if mouse_hook:
                    self._mouse_unhooked = bool(
                        user32.UnhookWindowsHookEx(mouse_hook)
                    )
            except Exception:
                self._thread_error = self._thread_error or "unhook_failed"
            self._ready.set()

    @staticmethod
    def _send_self_test_input() -> None:  # pragma: no cover - Windows KAT
        user32 = ctypes.WinDLL("user32", use_last_error=True)

        class MouseInput(ctypes.Structure):
            _fields_ = (
                ("dx", wintypes.LONG),
                ("dy", wintypes.LONG),
                ("mouse_data", wintypes.DWORD),
                ("flags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("extra_info", ctypes.c_size_t),
            )

        class KeyboardInput(ctypes.Structure):
            _fields_ = (
                ("virtual_key", wintypes.WORD),
                ("scan_code", wintypes.WORD),
                ("flags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("extra_info", ctypes.c_size_t),
            )

        class HardwareInput(ctypes.Structure):
            _fields_ = (
                ("message", wintypes.DWORD),
                ("parameter_low", wintypes.WORD),
                ("parameter_high", wintypes.WORD),
            )

        class InputPayload(ctypes.Union):
            _fields_ = (
                ("mouse", MouseInput),
                ("keyboard", KeyboardInput),
                ("hardware", HardwareInput),
            )

        class Input(ctypes.Structure):
            _fields_ = (
                ("input_type", wintypes.DWORD),
                ("payload", InputPayload),
            )

        user32.GetCursorPos.argtypes = (ctypes.POINTER(wintypes.POINT),)
        user32.GetCursorPos.restype = wintypes.BOOL
        user32.GetSystemMetrics.argtypes = (ctypes.c_int,)
        user32.GetSystemMetrics.restype = ctypes.c_int
        user32.SendInput.argtypes = (
            wintypes.UINT,
            ctypes.POINTER(Input),
            ctypes.c_int,
        )
        user32.SendInput.restype = wintypes.UINT
        cursor = wintypes.POINT()
        if not user32.GetCursorPos(ctypes.byref(cursor)):
            raise ExternalInputGuardError("self_test_cursor_unavailable")
        left = int(user32.GetSystemMetrics(76))
        top = int(user32.GetSystemMetrics(77))
        width = int(user32.GetSystemMetrics(78))
        height = int(user32.GetSystemMetrics(79))
        if width <= 1 or height <= 1:
            raise ExternalInputGuardError("self_test_desktop_invalid")
        neighbour_x = int(cursor.x) + (1 if int(cursor.x) < left + width - 1 else -1)

        def absolute_mouse(x: int, y: int) -> MouseInput:
            return MouseInput(
                dx=round((x - left) * 65535 / (width - 1)),
                dy=round((y - top) * 65535 / (height - 1)),
                mouse_data=0,
                flags=0x0001 | 0x2000 | 0x4000 | 0x8000,
                time=0,
                extra_info=SELF_TEST_INPUT_EXTRA_INFO,
            )

        items = (Input * 4)()
        items[0].input_type = 1
        items[0].payload.keyboard = KeyboardInput(
            virtual_key=0x87,
            scan_code=0,
            flags=0,
            time=0,
            extra_info=SELF_TEST_INPUT_EXTRA_INFO,
        )
        items[1].input_type = 1
        items[1].payload.keyboard = KeyboardInput(
            virtual_key=0x87,
            scan_code=0,
            flags=0x0002,
            time=0,
            extra_info=SELF_TEST_INPUT_EXTRA_INFO,
        )
        items[2].input_type = 0
        items[2].payload.mouse = absolute_mouse(neighbour_x, int(cursor.y))
        items[3].input_type = 0
        items[3].payload.mouse = absolute_mouse(
            int(cursor.x), int(cursor.y)
        )
        if user32.SendInput(4, items, ctypes.sizeof(Input)) != 4:
            raise ExternalInputGuardError("self_test_send_input_failed")

    def _run_self_test(self, label: str) -> Mapping[str, Any]:
        counts = self._self_test_counts[label]
        self._active_self_test = label
        try:
            self._send_self_test_input()
            deadline = time.monotonic() + SELF_TEST_TIMEOUT_SECONDS
            while time.monotonic() < deadline:
                if counts["keyboard"] >= 2 and counts["mouse"] >= 2:
                    break
                time.sleep(0.01)
        finally:
            self._active_self_test = None
        passed = counts["keyboard"] >= 2 and counts["mouse"] >= 2
        return {
            "label": label,
            "status": "PASS" if passed else "FAIL",
            "keyboard_event_count": counts["keyboard"],
            "mouse_event_count": counts["mouse"],
        }

    def _start_coverage_after_quiescence(self) -> int:
        quiet_ns = round(START_QUIESCENCE_SECONDS * 1_000_000_000)
        deadline = time.monotonic() + START_QUIESCENCE_TIMEOUT_SECONDS
        quiet_not_before_ns = time.perf_counter_ns()
        while True:
            now_ns = time.perf_counter_ns()
            with self._state_lock:
                last_activity_ns = self._precoverage_last_activity_ns
                quiet_since_ns = max(
                    quiet_not_before_ns,
                    (
                        last_activity_ns
                        if last_activity_ns is not None
                        else quiet_not_before_ns
                    ),
                )
                if now_ns - quiet_since_ns >= quiet_ns:
                    self._coverage_start_ns = now_ns
                    return now_ns
            if time.monotonic() >= deadline:
                raise ExternalInputGuardError("start_quiescence_timeout")
            time.sleep(START_QUIESCENCE_POLL_SECONDS)

    def start(self) -> int:
        _require_windows()
        if self._thread is not None:
            raise ExternalInputGuardError("guard_already_started")
        self._thread = threading.Thread(
            target=self._thread_main,
            name="zuma-external-input-guard",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(HOOK_READY_TIMEOUT_SECONDS):
            raise ExternalInputGuardError("hook_ready_timeout")
        if (
            self._thread_error is not None
            or not self._keyboard_installed
            or not self._mouse_installed
        ):
            raise ExternalInputGuardError(
                self._thread_error or "hook_install_failed"
            )
        try:
            start_test = self._run_self_test("start")
            self._self_tests.append(dict(start_test))
            if start_test["status"] != "PASS":
                raise ExternalInputGuardError("start_self_test_failed")
            return self._start_coverage_after_quiescence()
        except Exception:
            self._stop_thread()
            raise

    def _stop_thread(self) -> None:
        if self._thread is None:
            return
        if self._thread.is_alive() and self._thread_id is not None:
            user32 = ctypes.WinDLL("user32", use_last_error=True)
            user32.PostThreadMessageW.argtypes = (
                wintypes.DWORD,
                wintypes.UINT,
                wintypes.WPARAM,
                wintypes.LPARAM,
            )
            user32.PostThreadMessageW.restype = wintypes.BOOL
            if not user32.PostThreadMessageW(
                self._thread_id, 0x0012, 0, 0
            ):
                self._thread_error = self._thread_error or (
                    "hook_stop_message_failed"
                )
        self._thread.join(THREAD_STOP_TIMEOUT_SECONDS)
        if self._thread.is_alive():
            self._thread_error = self._thread_error or "hook_stop_timeout"

    def finish(self) -> Mapping[str, Any]:
        if self._finished_receipt is not None:
            return self._finished_receipt
        if self._coverage_start_ns is None:
            raise ExternalInputGuardError("guard_not_started")
        with self._state_lock:
            self._coverage_end_ns = time.perf_counter_ns()
        try:
            end_test = self._run_self_test("end")
        except Exception as error:
            end_test = {
                "label": "end",
                "status": "FAIL",
                "keyboard_event_count": self._self_test_counts["end"][
                    "keyboard"
                ],
                "mouse_event_count": self._self_test_counts["end"][
                    "mouse"
                ],
            }
            self._thread_error = self._thread_error or type(error).__name__
        self._self_tests.append(dict(end_test))
        self._stop_thread()
        protocol_errors = [
            error
            for error in (self._thread_error, self._callback_error)
            if error is not None
        ]
        hooks = {
            "keyboard_installed": self._keyboard_installed,
            "mouse_installed": self._mouse_installed,
            "keyboard_unhooked": self._keyboard_unhooked,
            "mouse_unhooked": self._mouse_unhooked,
        }
        passed = (
            not protocol_errors
            and all(hooks.values())
            and len(self._self_tests) == 2
            and all(row["status"] == "PASS" for row in self._self_tests)
            and self._counts["external"] == 0
        )
        receipt = {
            "schema": EXTERNAL_INPUT_GUARD_SCHEMA,
            "version": EXTERNAL_INPUT_GUARD_VERSION,
            "status": "PASS" if passed else "FAIL",
            "policy": EXTERNAL_INPUT_POLICY,
            "privacy": EXTERNAL_INPUT_PRIVACY,
            "coverage": {
                "start_perf_counter_ns": self._coverage_start_ns,
                "end_perf_counter_ns": self._coverage_end_ns,
            },
            "hooks": hooks,
            "self_tests": self._self_tests,
            "markers": {
                "repaint_input_extra_info_hex": (
                    f"0x{REPAINT_INPUT_EXTRA_INFO:08x}"
                ),
                "self_test_input_extra_info_hex": (
                    f"0x{SELF_TEST_INPUT_EXTRA_INFO:08x}"
                ),
            },
            "event_counts": dict(self._counts),
            "external_events": list(self._external_events),
            "protocol_errors": protocol_errors,
        }
        validate_external_input_guard_receipt(
            receipt,
            require_pass=False,
        )
        self._finished_receipt = receipt
        return receipt


__all__ = ["ExternalInputGuard", "ExternalInputGuardError"]
