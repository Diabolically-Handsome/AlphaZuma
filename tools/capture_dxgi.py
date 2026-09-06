"""Short, lossless DXGI capture for PC golden evidence.

This helper is Windows-only and deliberately refuses to capture unless the
target game process is running.  It stores raw BGRA frames plus the original
DXGI/QPC presentation timestamps; video encoding happens later so a codec
cannot hide acquisition loss.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import importlib.metadata
import json
import math
import os
import re
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

if __package__:
    from .embedded_pe_payload import (
        EmbeddedPeError,
        EmbeddedPeIdentity,
        inspect_embedded_signed_pe,
    )
else:
    from embedded_pe_payload import (
        EmbeddedPeError,
        EmbeddedPeIdentity,
        inspect_embedded_signed_pe,
    )

CAPTURE_SCHEMA = "zuma-rl.dxgi-bgra-capture"
CAPTURE_VERSION = 2
DEFAULT_PROCESS = "popcapgame1.exe"
DEFAULT_FRAME_BUDGET_FPS = 240
MIN_DURATION_SECONDS = 0.25
# A 5.5-second window is required for the formal 500-native-tick drift
# campaign.  The independent MAX_RAW_BYTES calculation remains the hard
# allocation bound, so raising the time limit does not permit an oversized
# capture at 240 fps.
MAX_DURATION_SECONDS = 6.0
WARMUP_TIMEOUT_SECONDS = 2.0
MAX_RAW_BYTES = 2 * 1024**3
PIXEL_FORMAT = "bgra"
BYTES_PER_PIXEL = 4
SUPPORTED_RUNTIME_VERSIONS = {
    "comtypes": "1.4.16",
    "dxcam": "0.3.0",
    "numpy": "2.5.1",
}
_PIXEL_HASH_DOMAIN = b"zuma-rl.dxgi-bgra.v1\0"
_PROCESS_PATTERN = re.compile(r"[A-Za-z0-9_.-]{1,128}\.exe", re.IGNORECASE)
FRAMEWORK_UPDATE_SCHEMA = "zuma-rl.pc-framework-update-map"
FRAMEWORK_UPDATE_VERSION = 1
FRAMEWORK_STATE_SCHEMA = "zuma-rl.pc-framework-state-diagnostic"
FRAMEWORK_STATE_VERSION = 1
G_SEXY_APP_BASE_ADDRESS = 0x009FC740
FRAMEWORK_UPDATE_OFFSET = 0x4C4
FRAMEWORK_STATE_OFFSET = 0x494
FRAMEWORK_STATE_BYTES = 92
_DPI_AWARENESS_VERIFIED = False
_WINDOW_GUARD_USER32: Any | None = None

Region = tuple[int, int, int, int]


@dataclass(frozen=True)
class WindowTarget:
    process_id: int
    window_handle: int
    client_region: Region


@dataclass
class _CapturedFrame:
    pixels: Any
    present_ticks: int
    qpc_frequency: int
    host_perf_counter_ns: int
    accumulated_frames: int
    framework_update_before: int | None = None
    framework_update_after: int | None = None


@dataclass(frozen=True)
class FrameworkUpdateSample:
    sequence: int
    present_ticks: int
    host_perf_counter_ns: int
    update_before: int
    update_after: int


@dataclass(frozen=True)
class FrameworkStateSnapshot:
    """One coherent read of retail ``SexyAppBase`` scheduling fields."""

    non_draw_count: int
    frame_time_ms: int
    is_drawing: bool
    last_draw_was_empty: bool
    has_pending_draw: bool
    pending_updates_acc: float
    update_f_time_acc: float
    last_time_check: int
    last_time: int
    last_user_input_tick: int
    sleep_count: int
    draw_count: int
    update_count: int
    update_app_state: int
    update_app_depth: int
    update_multiplier: float
    paused: bool
    fast_forward_target: int
    fast_forward_to_marker: bool
    fast_forward_step: bool
    last_draw_tick: int
    next_draw_tick: int
    step_mode: int


@dataclass(frozen=True)
class FrameworkStateSample:
    sequence: int
    present_ticks: int
    host_perf_counter_ns: int
    before: FrameworkStateSnapshot
    after: FrameworkStateSnapshot


class CaptureError(RuntimeError):
    """A fixed-code error safe to expose without a traceback."""

    def __init__(
        self,
        code: str,
        *,
        diagnostic: dict[str, Any] | None = None,
    ) -> None:
        if not re.fullmatch(r"[a-z0-9_]{1,80}", code):
            code = "capture_failed"
        self.code = code
        self.diagnostic = (
            None if diagnostic is None else dict(diagnostic)
        )
        super().__init__(code)


class SafeArgumentParser(argparse.ArgumentParser):
    """Argument parser that does not echo private argument values on error."""

    def error(self, message: str) -> None:
        del message
        raise CaptureError("invalid_arguments")


def parse_duration(value: Any) -> float:
    if isinstance(value, bool):
        raise CaptureError("invalid_duration")
    try:
        duration = float(value)
    except (OverflowError, TypeError, ValueError):
        raise CaptureError("invalid_duration") from None
    if (
        not math.isfinite(duration)
        or duration < MIN_DURATION_SECONDS
        or duration > MAX_DURATION_SECONDS
    ):
        raise CaptureError("invalid_duration")
    return duration


def parse_frame_budget_fps(value: Any) -> int:
    if isinstance(value, bool):
        raise CaptureError("invalid_frame_budget_fps")
    try:
        frame_budget_fps = int(value)
    except (OverflowError, TypeError, ValueError):
        raise CaptureError("invalid_frame_budget_fps") from None
    if (
        frame_budget_fps < 60
        or frame_budget_fps > 240
        or str(frame_budget_fps) != str(value)
    ):
        raise CaptureError("invalid_frame_budget_fps")
    return frame_budget_fps


def parse_capture_index(value: Any, *, error_code: str) -> int:
    if isinstance(value, bool):
        raise CaptureError(error_code)
    try:
        index = int(value)
    except (OverflowError, TypeError, ValueError):
        raise CaptureError(error_code) from None
    if index < 0 or index > 15 or str(index) != str(value):
        raise CaptureError(error_code)
    return index


def parse_framework_update(value: Any) -> int:
    if isinstance(value, bool):
        raise CaptureError("invalid_framework_update")
    try:
        update = int(value)
    except (OverflowError, TypeError, ValueError):
        raise CaptureError("invalid_framework_update") from None
    if update < 0 or update > 0x7FFFFFFF or str(update) != str(value):
        raise CaptureError("invalid_framework_update")
    return update


def parse_region(values: Sequence[Any]) -> Region:
    if len(values) != 4:
        raise CaptureError("invalid_region")
    parsed: list[int] = []
    for value in values:
        if isinstance(value, bool):
            raise CaptureError("invalid_region")
        try:
            coordinate = int(value)
        except (OverflowError, TypeError, ValueError):
            raise CaptureError("invalid_region") from None
        if str(coordinate) != str(value):
            raise CaptureError("invalid_region")
        parsed.append(coordinate)
    left, top, right, bottom = parsed
    if right <= left or bottom <= top:
        raise CaptureError("invalid_region")
    return left, top, right, bottom


def region_dimensions(region: Region) -> tuple[int, int]:
    left, top, right, bottom = region
    return right - left, bottom - top


def estimated_raw_bytes(
    region: Region,
    duration_seconds: float,
    frame_budget_fps: int,
) -> int:
    width, height = region_dimensions(region)
    frame_count = math.ceil(duration_seconds * frame_budget_fps) + 2
    estimated = width * height * BYTES_PER_PIXEL * frame_count
    if estimated > MAX_RAW_BYTES:
        raise CaptureError("raw_budget_exceeded")
    return estimated


def validate_process_name(process_name: Any) -> str:
    if (
        not isinstance(process_name, str)
        or _PROCESS_PATTERN.fullmatch(process_name) is None
    ):
        raise CaptureError("invalid_process_name")
    return process_name


def ensure_empty_output_directory(path: str | Path) -> Path:
    output = Path(path)
    if not output.is_absolute():
        raise CaptureError("output_directory_not_absolute")
    try:
        is_junction = getattr(output, "is_junction", lambda: False)
        if (
            output.is_symlink()
            or is_junction()
            or not output.is_dir()
        ):
            raise CaptureError("output_directory_invalid")
        if any(output.iterdir()):
            raise CaptureError("output_directory_not_empty")
        if os.name == "nt":
            str(output).encode("ascii", errors="strict")
    except CaptureError:
        raise
    except (OSError, UnicodeError):
        raise CaptureError("output_directory_invalid") from None
    return output


def _require_windows() -> None:
    if os.name != "nt":
        raise CaptureError("windows_required")


def prepare_framework_update_output(
    value: str | Path,
    *,
    capture_output: Path,
) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise CaptureError("framework_update_output_not_absolute")
    part = path.with_name(path.name + ".part")
    try:
        resolved_parent = path.parent.resolve()
        resolved_capture = capture_output.resolve()
        if (
            path.name in {"", ".", ".."}
            or path.suffix.casefold() != ".json"
            or not resolved_parent.is_dir()
            or resolved_parent.is_symlink()
            or path.exists()
            or part.exists()
            or resolved_parent == resolved_capture
            or resolved_parent.is_relative_to(resolved_capture)
        ):
            raise CaptureError("framework_update_output_invalid")
        if os.name == "nt":
            str(path).encode("ascii", errors="strict")
    except CaptureError:
        raise
    except (OSError, UnicodeError, ValueError):
        raise CaptureError("framework_update_output_invalid") from None
    return path


def prepare_framework_state_output(
    value: str | Path,
    *,
    capture_output: Path,
) -> Path:
    """Validate a new diagnostic sidecar path outside the raw capture."""

    path = Path(value)
    if not path.is_absolute():
        raise CaptureError("framework_state_output_not_absolute")
    part = path.with_name(path.name + ".part")
    try:
        resolved_parent = path.parent.resolve()
        resolved_capture = capture_output.resolve()
        if (
            path.name in {"", ".", ".."}
            or path.suffix.casefold() != ".json"
            or not resolved_parent.is_dir()
            or resolved_parent.is_symlink()
            or path.exists()
            or part.exists()
            or resolved_parent == resolved_capture
            or resolved_parent.is_relative_to(resolved_capture)
        ):
            raise CaptureError("framework_state_output_invalid")
        if os.name == "nt":
            str(path).encode("ascii", errors="strict")
    except CaptureError:
        raise
    except (OSError, UnicodeError, ValueError):
        raise CaptureError("framework_state_output_invalid") from None
    return path


class FrameworkUpdateReader:
    """Read the verified 32-bit SexyApp framework update counter."""

    def __init__(self, process_id: int) -> None:
        _require_windows()
        from ctypes import wintypes

        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._kernel32.OpenProcess.argtypes = (
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.DWORD,
        )
        self._kernel32.OpenProcess.restype = wintypes.HANDLE
        self._kernel32.ReadProcessMemory.argtypes = (
            wintypes.HANDLE,
            wintypes.LPCVOID,
            wintypes.LPVOID,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t),
        )
        self._kernel32.ReadProcessMemory.restype = wintypes.BOOL
        self._kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        self._kernel32.CloseHandle.restype = wintypes.BOOL
        self._handle = self._kernel32.OpenProcess(
            0x1000 | 0x0010,
            False,
            int(process_id),
        )
        if not self._handle:
            raise CaptureError("framework_update_process_open_failed")
        self._base = 0

    def _read_bytes(self, address: int, size: int) -> bytes:
        payload = (ctypes.c_ubyte * size)()
        transferred = ctypes.c_size_t()
        if not self._handle or not self._kernel32.ReadProcessMemory(
            self._handle,
            ctypes.c_void_p(address),
            ctypes.byref(payload),
            size,
            ctypes.byref(transferred),
        ):
            raise CaptureError("framework_update_read_failed")
        if transferred.value != size:
            raise CaptureError("framework_update_read_failed")
        return bytes(payload)

    def _read_u32(self, address: int) -> int:
        return int.from_bytes(
            self._read_bytes(address, 4),
            "little",
            signed=False,
        )

    def _base_address(self) -> int:
        if not self._base:
            self._base = self._read_u32(G_SEXY_APP_BASE_ADDRESS)
            if not self._base:
                raise CaptureError("framework_update_base_unavailable")
        return self._base

    @staticmethod
    def decode_state(data: bytes) -> FrameworkStateSnapshot:
        """Decode the verified 32-bit retail scheduling-field window."""

        if len(data) != FRAMEWORK_STATE_BYTES:
            raise CaptureError("framework_state_read_failed")

        def i32(offset: int) -> int:
            return struct.unpack_from("<i", data, offset)[0]

        def u32(offset: int) -> int:
            return struct.unpack_from("<I", data, offset)[0]

        snapshot = FrameworkStateSnapshot(
            non_draw_count=i32(0),
            frame_time_ms=i32(4),
            is_drawing=bool(data[8]),
            last_draw_was_empty=bool(data[9]),
            has_pending_draw=bool(data[10]),
            pending_updates_acc=struct.unpack_from("<d", data, 12)[0],
            update_f_time_acc=struct.unpack_from("<d", data, 20)[0],
            last_time_check=u32(28),
            last_time=u32(32),
            last_user_input_tick=u32(36),
            sleep_count=i32(40),
            draw_count=i32(44),
            update_count=i32(48),
            update_app_state=i32(52),
            update_app_depth=i32(56),
            update_multiplier=struct.unpack_from("<d", data, 60)[0],
            paused=bool(data[68]),
            fast_forward_target=i32(72),
            fast_forward_to_marker=bool(data[76]),
            fast_forward_step=bool(data[77]),
            last_draw_tick=u32(80),
            next_draw_tick=u32(84),
            step_mode=i32(88),
        )
        if (
            snapshot.non_draw_count < 0
            or not 1 <= snapshot.frame_time_ms <= 1000
            or snapshot.sleep_count < 0
            or snapshot.draw_count < 0
            or snapshot.update_count < 0
            or not math.isfinite(snapshot.pending_updates_acc)
            or not math.isfinite(snapshot.update_f_time_acc)
            or not -2.0 <= snapshot.pending_updates_acc <= 2.0
            or not -1000.0 <= snapshot.update_f_time_acc <= 1000.0
            or not 0 <= snapshot.update_app_state <= 3
            or not 0 <= snapshot.update_app_depth <= 64
            or not math.isfinite(snapshot.update_multiplier)
            or snapshot.update_multiplier <= 0.0
            or snapshot.fast_forward_target < 0
            or not 0 <= snapshot.step_mode <= 2
        ):
            raise CaptureError("framework_state_invalid")
        return snapshot

    def sample(self) -> int:
        value = self._read_u32(
            self._base_address() + FRAMEWORK_UPDATE_OFFSET
        )
        if value > 0x7FFFFFFF:
            raise CaptureError("framework_update_invalid")
        return value

    def sample_state(self) -> FrameworkStateSnapshot:
        return self.decode_state(
            self._read_bytes(
                self._base_address() + FRAMEWORK_STATE_OFFSET,
                FRAMEWORK_STATE_BYTES,
            )
        )

    def close(self) -> None:
        if self._handle:
            self._kernel32.CloseHandle(self._handle)
            self._handle = None

    def __enter__(self) -> "FrameworkUpdateReader":
        return self

    def __exit__(self, *unused: object) -> None:
        self.close()


def _set_dpi_awareness() -> None:
    global _DPI_AWARENESS_VERIFIED
    _require_windows()
    if _DPI_AWARENESS_VERIFIED:
        return
    try:
        from ctypes import wintypes

        shcore = ctypes.WinDLL("shcore", use_last_error=True)
        shcore.SetProcessDpiAwareness.argtypes = (ctypes.c_int,)
        shcore.SetProcessDpiAwareness.restype = ctypes.c_long
        shcore.GetProcessDpiAwareness.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(ctypes.c_int),
        )
        shcore.GetProcessDpiAwareness.restype = ctypes.c_long
        # E_ACCESSDENIED is expected after the context has already been set.
        shcore.SetProcessDpiAwareness(2)
        awareness = ctypes.c_int()
        result = shcore.GetProcessDpiAwareness(
            None,
            ctypes.byref(awareness),
        )
    except (AttributeError, OSError):
        raise CaptureError("dpi_awareness_unavailable") from None
    if result != 0 or awareness.value != 2:
        raise CaptureError("dpi_awareness_unavailable")
    _DPI_AWARENESS_VERIFIED = True


def _window_guard_user32() -> Any:
    global _WINDOW_GUARD_USER32
    _require_windows()
    if _WINDOW_GUARD_USER32 is not None:
        return _WINDOW_GUARD_USER32
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.IsWindow.argtypes = (wintypes.HWND,)
    user32.IsWindow.restype = wintypes.BOOL
    user32.IsWindowVisible.argtypes = (wintypes.HWND,)
    user32.IsWindowVisible.restype = wintypes.BOOL
    user32.IsIconic.argtypes = (wintypes.HWND,)
    user32.IsIconic.restype = wintypes.BOOL
    user32.GetWindowThreadProcessId.argtypes = (
        wintypes.HWND,
        ctypes.POINTER(wintypes.DWORD),
    )
    user32.GetWindowThreadProcessId.restype = wintypes.DWORD
    user32.GetClientRect.argtypes = (
        wintypes.HWND,
        ctypes.POINTER(wintypes.RECT),
    )
    user32.GetClientRect.restype = wintypes.BOOL
    user32.ClientToScreen.argtypes = (
        wintypes.HWND,
        ctypes.POINTER(wintypes.POINT),
    )
    user32.ClientToScreen.restype = wintypes.BOOL
    user32.GetForegroundWindow.argtypes = ()
    user32.GetForegroundWindow.restype = wintypes.HWND
    _WINDOW_GUARD_USER32 = user32
    return user32


def windows_process_ids(process_name: str) -> tuple[int, ...]:
    """Return exact executable-name matches using Toolhelp32."""

    _require_windows()
    validate_process_name(process_name)
    from ctypes import wintypes

    class PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", wintypes.LONG),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.WCHAR * 260),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateToolhelp32Snapshot.argtypes = (
        wintypes.DWORD,
        wintypes.DWORD,
    )
    kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel32.Process32FirstW.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(PROCESSENTRY32W),
    )
    kernel32.Process32FirstW.restype = wintypes.BOOL
    kernel32.Process32NextW.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(PROCESSENTRY32W),
    )
    kernel32.Process32NextW.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL
    snapshot = kernel32.CreateToolhelp32Snapshot(0x00000002, 0)
    invalid_handle = wintypes.HANDLE(-1).value
    if snapshot == invalid_handle:
        raise CaptureError("process_inventory_failed")
    entry = PROCESSENTRY32W()
    entry.dwSize = ctypes.sizeof(entry)
    matches: list[int] = []
    try:
        success = kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
        while success:
            if entry.szExeFile.casefold() == process_name.casefold():
                matches.append(int(entry.th32ProcessID))
            success = kernel32.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snapshot)
    return tuple(sorted(set(matches)))


def windows_client_windows(
    process_ids: Iterable[int],
) -> tuple[WindowTarget, ...]:
    """Return visible, non-minimized top-level client windows."""

    _require_windows()
    _set_dpi_awareness()
    from ctypes import wintypes

    wanted = {int(pid) for pid in process_ids}
    if not wanted:
        return ()
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    windows: list[WindowTarget] = []

    window_callback = ctypes.WINFUNCTYPE(
        wintypes.BOOL,
        wintypes.HWND,
        wintypes.LPARAM,
    )
    user32.IsWindowVisible.argtypes = (wintypes.HWND,)
    user32.IsWindowVisible.restype = wintypes.BOOL
    user32.IsIconic.argtypes = (wintypes.HWND,)
    user32.IsIconic.restype = wintypes.BOOL
    user32.GetWindowThreadProcessId.argtypes = (
        wintypes.HWND,
        ctypes.POINTER(wintypes.DWORD),
    )
    user32.GetWindowThreadProcessId.restype = wintypes.DWORD
    user32.GetClientRect.argtypes = (
        wintypes.HWND,
        ctypes.POINTER(wintypes.RECT),
    )
    user32.GetClientRect.restype = wintypes.BOOL
    user32.ClientToScreen.argtypes = (
        wintypes.HWND,
        ctypes.POINTER(wintypes.POINT),
    )
    user32.ClientToScreen.restype = wintypes.BOOL

    @window_callback
    def callback(hwnd: int, lparam: int) -> bool:
        del lparam
        if not user32.IsWindowVisible(hwnd) or user32.IsIconic(hwnd):
            return True
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if int(pid.value) not in wanted:
            return True
        rect = wintypes.RECT()
        if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
            return True
        origin = wintypes.POINT(0, 0)
        extent = wintypes.POINT(rect.right, rect.bottom)
        if not user32.ClientToScreen(hwnd, ctypes.byref(origin)):
            return True
        if not user32.ClientToScreen(hwnd, ctypes.byref(extent)):
            return True
        region = (
            int(origin.x),
            int(origin.y),
            int(extent.x),
            int(extent.y),
        )
        if region[2] > region[0] and region[3] > region[1]:
            windows.append(
                WindowTarget(
                    process_id=int(pid.value),
                    window_handle=int(hwnd),
                    client_region=region,
                )
            )
        return True

    user32.EnumWindows.argtypes = (window_callback, wintypes.LPARAM)
    user32.EnumWindows.restype = wintypes.BOOL
    if not user32.EnumWindows(callback, 0):
        raise CaptureError("window_inventory_failed")
    return tuple(
        sorted(
            windows,
            key=lambda target: (
                target.process_id,
                target.window_handle,
                target.client_region,
            ),
        )
    )


def windows_client_regions(process_ids: Iterable[int]) -> tuple[Region, ...]:
    return tuple(
        target.client_region
        for target in windows_client_windows(process_ids)
    )


def windows_foreground_window_handle() -> int:
    """Return the current foreground HWND, or zero when none exists."""

    _require_windows()
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.GetForegroundWindow.argtypes = ()
    user32.GetForegroundWindow.restype = wintypes.HWND
    return int(user32.GetForegroundWindow() or 0)


def resolve_windows_target(
    *,
    process_name: str,
    explicit_region: Region | None,
    process_ids_fn: Callable[[str], tuple[int, ...]] = windows_process_ids,
    client_windows_fn: Callable[
        [Iterable[int]], tuple[WindowTarget, ...]
    ] = windows_client_windows,
    foreground_window_fn: Callable[
        [], int
    ] = windows_foreground_window_handle,
) -> tuple[WindowTarget, Region]:
    process_name = validate_process_name(process_name)
    process_ids = process_ids_fn(process_name)
    if len(process_ids) != 1:
        raise CaptureError(
            "target_process_missing"
            if not process_ids
            else "target_process_not_unique"
        )
    windows = client_windows_fn(process_ids)
    if not windows:
        raise CaptureError("target_window_missing")
    if len(windows) > 1:
        foreground = foreground_window_fn()
        windows = tuple(
            target
            for target in windows
            if target.window_handle == foreground
        )
    if len(windows) != 1:
        raise CaptureError("target_window_not_unique")
    target = windows[0]
    if (
        target.process_id != process_ids[0]
        or target.process_id <= 0
        or target.window_handle <= 0
    ):
        raise CaptureError("target_window_identity_invalid")
    client_region = parse_region(target.client_region)
    if explicit_region is None:
        return target, client_region
    explicit_region = parse_region(explicit_region)
    if not (
        client_region[0] <= explicit_region[0] < explicit_region[2]
        <= client_region[2]
        and client_region[1] <= explicit_region[1] < explicit_region[3]
        <= client_region[3]
    ):
        raise CaptureError("region_outside_target_window")
    return target, explicit_region


def resolve_global_region(
    *,
    process_name: str,
    explicit_region: Region | None,
    process_ids_fn: Callable[[str], tuple[int, ...]] = windows_process_ids,
    client_regions_fn: Callable[[Iterable[int]], tuple[Region, ...]] = (
        windows_client_regions
    ),
) -> Region:
    process_name = validate_process_name(process_name)
    process_ids = process_ids_fn(process_name)
    if len(process_ids) != 1:
        raise CaptureError(
            "target_process_missing"
            if not process_ids
            else "target_process_not_unique"
        )
    regions = client_regions_fn(process_ids)
    if len(regions) != 1:
        raise CaptureError(
            "target_window_missing"
            if not regions
            else "target_window_not_unique"
        )
    client_region = regions[0]
    if explicit_region is None:
        return client_region
    explicit_region = parse_region(explicit_region)
    if not (
        client_region[0] <= explicit_region[0] < explicit_region[2]
        <= client_region[2]
        and client_region[1] <= explicit_region[1] < explicit_region[3]
        <= client_region[3]
    ):
        raise CaptureError("region_outside_target_window")
    return explicit_region


def verify_windows_target(
    target: WindowTarget,
    *,
    require_foreground: bool = True,
) -> None:
    """Fail if the bound HWND/PID/client rectangle has changed."""

    _require_windows()
    _set_dpi_awareness()
    from ctypes import wintypes

    user32 = _window_guard_user32()

    hwnd = wintypes.HWND(target.window_handle)
    if (
        not user32.IsWindow(hwnd)
        or not user32.IsWindowVisible(hwnd)
        or user32.IsIconic(hwnd)
    ):
        raise CaptureError("target_window_changed")
    pid = wintypes.DWORD()
    if (
        not user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        or int(pid.value) != target.process_id
    ):
        raise CaptureError("target_window_changed")
    rect = wintypes.RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        raise CaptureError("target_window_changed")
    origin = wintypes.POINT(0, 0)
    extent = wintypes.POINT(rect.right, rect.bottom)
    if (
        not user32.ClientToScreen(hwnd, ctypes.byref(origin))
        or not user32.ClientToScreen(hwnd, ctypes.byref(extent))
    ):
        raise CaptureError("target_window_changed")
    current_region = (
        int(origin.x),
        int(origin.y),
        int(extent.x),
        int(extent.y),
    )
    if current_region != target.client_region:
        raise CaptureError("target_window_changed")
    if (
        require_foreground
        and int(user32.GetForegroundWindow() or 0)
        != target.window_handle
    ):
        raise CaptureError("target_window_not_foreground")


def windows_process_identity(
    target: WindowTarget,
    *,
    process_name: str,
    allow_deferred_executable_hash: bool = False,
    runtime_source_executable: Path | None = None,
) -> dict[str, Any]:
    """Bind a PID to its creation time and executable bytes."""

    _require_windows()
    process_name = validate_process_name(process_name)
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.QueryFullProcessImageNameW.argtypes = (
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD),
    )
    kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
    kernel32.GetProcessTimes.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
    )
    kernel32.GetProcessTimes.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL

    process = kernel32.OpenProcess(0x1000, False, target.process_id)
    if not process:
        raise CaptureError("target_process_identity_unavailable")
    try:
        capacity = wintypes.DWORD(32768)
        image_buffer = ctypes.create_unicode_buffer(capacity.value)
        if not kernel32.QueryFullProcessImageNameW(
            process,
            0,
            image_buffer,
            ctypes.byref(capacity),
        ):
            raise CaptureError("target_process_identity_unavailable")
        creation = wintypes.FILETIME()
        exit_time = wintypes.FILETIME()
        kernel_time = wintypes.FILETIME()
        user_time = wintypes.FILETIME()
        if not kernel32.GetProcessTimes(
            process,
            ctypes.byref(creation),
            ctypes.byref(exit_time),
            ctypes.byref(kernel_time),
            ctypes.byref(user_time),
        ):
            raise CaptureError("target_process_identity_unavailable")
    finally:
        kernel32.CloseHandle(process)

    image_path = Path(image_buffer.value)
    if image_path.name.casefold() != process_name.casefold():
        raise CaptureError("target_process_identity_mismatch")
    try:
        image_size = image_path.stat().st_size
        if (
            image_size <= 0
            or image_size > 512 * 1024**2
            or not image_path.is_file()
        ):
            raise CaptureError("target_process_identity_invalid")
        executable_hash_fields = _target_executable_hash_fields(
            image_path,
            allow_deferred=allow_deferred_executable_hash,
            executable_bytes=image_size,
            runtime_source_executable=runtime_source_executable,
        )
    except CaptureError:
        raise
    except OSError:
        raise CaptureError("target_process_identity_unavailable") from None
    creation_filetime = (
        int(creation.dwHighDateTime) << 32
    ) | int(creation.dwLowDateTime)
    if creation_filetime <= 0:
        raise CaptureError("target_process_identity_invalid")
    return {
        "process_id": target.process_id,
        "process_creation_filetime_100ns": creation_filetime,
        "executable_bytes": image_size,
        "window_handle_hex": f"0x{target.window_handle:016x}",
        "window_client_region": list(target.client_region),
        **executable_hash_fields,
    }


def output_local_region(camera: Any, global_region: Region) -> Region:
    """Translate a screen-space rectangle into one DXGI output."""

    try:
        if int(camera._output.rotation_angle) != 0:
            raise CaptureError("rotated_output_unsupported")
        desktop = camera._output.desc.DesktopCoordinates
        output_left = int(desktop.left)
        output_top = int(desktop.top)
        output_right = int(desktop.right)
        output_bottom = int(desktop.bottom)
    except CaptureError:
        raise
    except (AttributeError, OverflowError, TypeError, ValueError):
        raise CaptureError("output_geometry_unavailable") from None
    left, top, right, bottom = global_region
    if not (
        output_left <= left < right <= output_right
        and output_top <= top < bottom <= output_bottom
    ):
        raise CaptureError("region_outside_capture_output")
    return (
        left - output_left,
        top - output_top,
        right - output_left,
        bottom - output_top,
    )


def _dxgi_counters(camera: Any) -> tuple[int, int, int]:
    try:
        duplicator = camera._duplicator
        ticks = int(duplicator.latest_frame_ticks)
        frequency = int(duplicator.performance_frequency)
        accumulated = int(duplicator.accumulated_frames)
    except (AttributeError, OverflowError, TypeError, ValueError):
        raise CaptureError("dxgi_clock_unavailable") from None
    if (
        ticks < 0
        or frequency <= 0
        or accumulated < 0
        or (accumulated > 0 and ticks <= 0)
    ):
        raise CaptureError("dxgi_present_clock_invalid")
    return ticks, frequency, accumulated


def _camera_duplicator(camera: Any) -> Any:
    try:
        return camera._duplicator
    except AttributeError:
        raise CaptureError("dxgi_clock_unavailable") from None


def _warm_up_capture(
    *,
    camera: Any,
    local_region: Region,
    width: int,
    height: int,
    destination: Any | None,
    monotonic_ns: Callable[[], int],
    idle_sleep: Callable[[float], None],
    source_guard: Callable[[], None] | None,
) -> tuple[Any, int, int, int, int, int]:
    """Drain the initial desktop frame and establish a present boundary."""

    duplicator = _camera_duplicator(camera)
    warmup_idle_polls = 0
    pointer_only_updates = 0
    start_ns = monotonic_ns()
    deadline_ns = start_ns + int(WARMUP_TIMEOUT_SECONDS * 1_000_000_000)
    if source_guard is not None:
        source_guard()
    while monotonic_ns() < deadline_ns:
        if _camera_duplicator(camera) is not duplicator:
            raise CaptureError("capture_source_changed")
        frame = _grab_owned_frame(
            camera=camera,
            region=local_region,
            width=width,
            height=height,
            destination=destination,
        )
        boundary_host_ns = monotonic_ns()
        if frame is None:
            warmup_idle_polls += 1
            idle_sleep(0.0005)
            continue
        if source_guard is not None:
            source_guard()
        ticks, frequency, accumulated = _dxgi_counters(camera)
        if accumulated == 0:
            pointer_only_updates += 1
            continue
        if accumulated != 1:
            # Warmup is outside the committed capture interval. Drain any
            # startup backlog until one presentation is acquired cleanly;
            # the recorded interval still rejects every missed presentation.
            continue
        _frame_bytes(frame, width, height)
        return (
            duplicator,
            ticks,
            frequency,
            boundary_host_ns,
            warmup_idle_polls,
            pointer_only_updates,
        )
    raise CaptureError("warmup_present_timeout")


def validate_runtime_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package, expected in SUPPORTED_RUNTIME_VERSIONS.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            raise CaptureError("capture_runtime_unavailable") from None
        if actual != expected:
            raise CaptureError("unsupported_capture_runtime")
        versions[package] = actual
    return versions


def capture_source_identity(camera: Any) -> dict[str, Any]:
    try:
        device = camera._device
        output = camera._output
        description = str(device.description)
        vendor_id = int(device.vendor_id)
        vram_bytes = int(device.vram_size)
        output_name = str(output.devicename)
        desktop = output.desc.DesktopCoordinates
        desktop_region = (
            int(desktop.left),
            int(desktop.top),
            int(desktop.right),
            int(desktop.bottom),
        )
        rotation = int(output.rotation_angle)
    except (AttributeError, OverflowError, TypeError, ValueError):
        raise CaptureError("capture_source_identity_unavailable") from None
    if (
        not description
        or len(description) > 256
        or any(ord(character) < 32 for character in description)
        or not output_name
        or len(output_name) > 256
        or any(ord(character) < 32 for character in output_name)
        or vendor_id < 0
        or vram_bytes < 0
        or region_dimensions(desktop_region)[0] <= 0
        or region_dimensions(desktop_region)[1] <= 0
    ):
        raise CaptureError("capture_source_identity_invalid")
    return {
        "adapter_description": description,
        "adapter_vendor_id": vendor_id,
        "adapter_vram_bytes": vram_bytes,
        "output_device_name": output_name,
        "output_desktop_region": list(desktop_region),
        "output_rotation_degrees": rotation,
    }


def _frame_bytes(frame: Any, width: int, height: int) -> memoryview:
    try:
        dtype_name = str(frame.dtype)
        shape = tuple(int(value) for value in frame.shape)
        contiguous = bool(frame.flags.c_contiguous)
        view = memoryview(frame).cast("B")
    except (AttributeError, BufferError, TypeError, ValueError):
        raise CaptureError("frame_layout_invalid") from None
    if (
        dtype_name != "uint8"
        or shape != (height, width, BYTES_PER_PIXEL)
        or not contiguous
        or view.nbytes != width * height * BYTES_PER_PIXEL
    ):
        raise CaptureError("frame_layout_invalid")
    return view


def _preallocate_capture_frames(
    *,
    camera: Any,
    width: int,
    height: int,
    frame_budget: int,
) -> list[Any] | None:
    """Pre-touch owned destinations for the pinned DXCam fast path."""

    if not callable(getattr(camera, "_grab_into", None)):
        return None
    try:
        import numpy as np
    except (ImportError, OSError, RuntimeError):
        raise CaptureError("capture_runtime_unavailable") from None
    frames: list[Any] = []
    try:
        for _ in range(frame_budget):
            frame = np.empty(
                (height, width, BYTES_PER_PIXEL),
                dtype=np.uint8,
            )
            frame.fill(0)
            _frame_bytes(frame, width, height)
            frames.append(frame)
    except (MemoryError, OverflowError, ValueError):
        raise CaptureError("frame_pool_allocation_failed") from None
    return frames


def _grab_owned_frame(
    *,
    camera: Any,
    region: Region,
    width: int,
    height: int,
    destination: Any | None,
) -> Any | None:
    if destination is None:
        return camera.grab(
            region=region,
            copy=True,
            new_frame_only=True,
        )
    grab_into = getattr(camera, "_grab_into", None)
    if not callable(grab_into):
        raise CaptureError("capture_runtime_incompatible")
    try:
        result = grab_into(region, destination)
        if not isinstance(result, tuple) or len(result) != 4:
            raise TypeError
        captured, frame_ticks, frame_width, frame_height = result
        frame_ticks = int(frame_ticks)
        frame_width = int(frame_width)
        frame_height = int(frame_height)
    except (AttributeError, OverflowError, TypeError, ValueError):
        raise CaptureError("capture_runtime_incompatible") from None
    if not isinstance(captured, bool):
        raise CaptureError("capture_runtime_incompatible")
    if not captured:
        if frame_ticks != 0 or frame_width != 0 or frame_height != 0:
            raise CaptureError("frame_layout_invalid")
        return None
    if (
        frame_ticks <= 0
        or frame_width != width
        or frame_height != height
    ):
        raise CaptureError("frame_layout_invalid")
    _frame_bytes(destination, width, height)
    return destination


def _open_exclusive_outputs(output: Path) -> tuple[Any, Any, Path, Path]:
    raw_part = output / "frames.bgra.raw.part"
    csv_part = output / "frames.csv.part"
    raw_stream = None
    try:
        raw_stream = raw_part.open("xb")
        csv_stream = csv_part.open("x", encoding="utf-8", newline="")
    except (FileExistsError, OSError):
        if raw_stream is not None:
            raw_stream.close()
        raise CaptureError("output_creation_failed") from None
    return raw_stream, csv_stream, raw_part, csv_part


def _sha256_file(
    path: Path,
    *,
    error_code: str = "output_hash_failed",
) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except OSError:
        raise CaptureError(error_code) from None
    return f"sha256:{digest.hexdigest()}"


def _target_executable_hash_fields(
    image_path: Path,
    *,
    allow_deferred: bool,
    executable_bytes: int | None = None,
    runtime_source_executable: Path | None = None,
    hash_file_fn: Callable[[Path], str] | None = None,
    inspect_embedded_fn: Callable[..., EmbeddedPeIdentity] | None = None,
) -> dict[str, Any]:
    """Bind target bytes directly or through an embedded signed PE."""

    if hash_file_fn is None:
        def hash_file_fn(path: Path) -> str:
            return _sha256_file(
                path,
                error_code="target_executable_hash_unavailable",
            )
    if inspect_embedded_fn is None:
        inspect_embedded_fn = inspect_embedded_signed_pe

    if runtime_source_executable is not None:
        if (
            executable_bytes is None
            or isinstance(executable_bytes, bool)
            or executable_bytes <= 0
            or allow_deferred
        ):
            raise CaptureError("runtime_source_identity_invalid")
        try:
            embedded = inspect_embedded_fn(
                runtime_source_executable,
                expected_payload_bytes=executable_bytes,
            )
        except EmbeddedPeError:
            raise CaptureError("runtime_source_identity_invalid") from None
        direct_verified = False
        try:
            direct_hash = hash_file_fn(image_path)
        except CaptureError as error:
            if error.code != "target_executable_hash_unavailable":
                raise
        else:
            if direct_hash != embedded.payload_sha256:
                raise CaptureError("runtime_source_identity_mismatch")
            direct_verified = True
        provenance = {
            "method": "embedded_signed_pe",
            **embedded.to_dict(),
            "runtime_file_direct_hash_verified": direct_verified,
        }
        return {
            "executable_sha256": embedded.payload_sha256,
            "executable_hash_provenance": provenance,
        }

    try:
        executable_sha256 = hash_file_fn(image_path)
    except CaptureError as error:
        if (
            not allow_deferred
            or error.code != "target_executable_hash_unavailable"
        ):
            raise
        return {
            "executable_sha256": None,
            "executable_sha256_status": (
                "deferred_locked_runtime_payload_probe_only"
            ),
        }
    return {
        "executable_sha256": executable_sha256,
        "executable_hash_provenance": {
            "method": "direct_file_sha256",
        },
    }


def write_framework_update_sidecar(
    path: Path,
    *,
    metadata_bytes: bytes,
    target_identity: dict[str, Any],
    samples: Sequence[FrameworkUpdateSample],
) -> str:
    """Exclusively publish a canonical per-frame framework-update map."""

    if not samples:
        raise CaptureError("framework_update_samples_missing")
    required_identity = {
        "process_id",
        "process_creation_filetime_100ns",
        "executable_sha256",
    }
    if not required_identity.issubset(target_identity):
        raise CaptureError("framework_update_identity_missing")
    records: list[dict[str, int]] = []
    previous_present: int | None = None
    previous_host: int | None = None
    previous_before: int | None = None
    previous_after: int | None = None
    for expected_sequence, sample in enumerate(samples):
        if (
            not isinstance(sample, FrameworkUpdateSample)
            or sample.sequence != expected_sequence
            or sample.present_ticks < 0
            or sample.host_perf_counter_ns <= 0
            or sample.update_before < 0
            or sample.update_after < sample.update_before
            or (
                previous_present is not None
                and sample.present_ticks <= previous_present
            )
            or (
                previous_host is not None
                and sample.host_perf_counter_ns <= previous_host
            )
            or (
                previous_before is not None
                and sample.update_before < previous_before
            )
            or (
                previous_after is not None
                and sample.update_after < previous_after
            )
        ):
            raise CaptureError("framework_update_samples_invalid")
        records.append(
            {
                "sequence": sample.sequence,
                "present_ticks": sample.present_ticks,
                "host_perf_counter_ns": sample.host_perf_counter_ns,
                "update_before": sample.update_before,
                "update_after": sample.update_after,
            }
        )
        previous_present = sample.present_ticks
        previous_host = sample.host_perf_counter_ns
        previous_before = sample.update_before
        previous_after = sample.update_after
    payload = {
        "schema": FRAMEWORK_UPDATE_SCHEMA,
        "version": FRAMEWORK_UPDATE_VERSION,
        "capture_metadata_sha256": (
            f"sha256:{hashlib.sha256(metadata_bytes).hexdigest()}"
        ),
        "process_id": int(target_identity["process_id"]),
        "process_creation_filetime_100ns": int(
            target_identity["process_creation_filetime_100ns"]
        ),
        "executable_sha256": target_identity["executable_sha256"],
        "records": records,
    }
    try:
        encoded = (
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")
    except (OverflowError, TypeError, UnicodeError, ValueError):
        raise CaptureError("framework_update_encode_failed") from None
    part = path.with_name(path.name + ".part")
    try:
        with part.open("xb") as stream:
            if stream.write(encoded) != len(encoded):
                raise CaptureError("framework_update_write_failed")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(part, path)
        part.unlink()
    except CaptureError:
        raise
    except (FileExistsError, OSError):
        raise CaptureError("framework_update_publish_failed") from None
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _framework_state_row(
    state: FrameworkStateSnapshot,
) -> dict[str, int | float | bool]:
    if not isinstance(state, FrameworkStateSnapshot):
        raise CaptureError("framework_state_samples_invalid")
    return {
        "non_draw_count": state.non_draw_count,
        "frame_time_ms": state.frame_time_ms,
        "is_drawing": state.is_drawing,
        "last_draw_was_empty": state.last_draw_was_empty,
        "has_pending_draw": state.has_pending_draw,
        "pending_updates_acc": state.pending_updates_acc,
        "update_f_time_acc": state.update_f_time_acc,
        "last_time_check": state.last_time_check,
        "last_time": state.last_time,
        "last_user_input_tick": state.last_user_input_tick,
        "sleep_count": state.sleep_count,
        "draw_count": state.draw_count,
        "update_count": state.update_count,
        "update_app_state": state.update_app_state,
        "update_app_depth": state.update_app_depth,
        "update_multiplier": state.update_multiplier,
        "paused": state.paused,
        "fast_forward_target": state.fast_forward_target,
        "fast_forward_to_marker": state.fast_forward_to_marker,
        "fast_forward_step": state.fast_forward_step,
        "last_draw_tick": state.last_draw_tick,
        "next_draw_tick": state.next_draw_tick,
        "step_mode": state.step_mode,
    }


def write_framework_state_sidecar(
    path: Path,
    *,
    metadata_bytes: bytes,
    framework_update_bytes: bytes,
    target_identity: dict[str, Any],
    samples: Sequence[FrameworkStateSample],
) -> str:
    """Publish diagnostic scheduler observations bound to the capture/map."""

    if not samples:
        raise CaptureError("framework_state_samples_missing")
    required_identity = {
        "process_id",
        "process_creation_filetime_100ns",
        "executable_sha256",
    }
    if not required_identity.issubset(target_identity):
        raise CaptureError("framework_state_identity_missing")
    records: list[dict[str, Any]] = []
    previous: FrameworkStateSample | None = None
    for expected_sequence, sample in enumerate(samples):
        if (
            not isinstance(sample, FrameworkStateSample)
            or sample.sequence != expected_sequence
            or sample.present_ticks < 0
            or sample.host_perf_counter_ns <= 0
            or sample.after.update_count < sample.before.update_count
            or sample.after.draw_count < sample.before.draw_count
            or (
                previous is not None
                and (
                    sample.present_ticks <= previous.present_ticks
                    or sample.host_perf_counter_ns
                    <= previous.host_perf_counter_ns
                    or sample.before.update_count
                    < previous.before.update_count
                    or sample.after.update_count
                    < previous.after.update_count
                    or sample.before.draw_count < previous.before.draw_count
                    or sample.after.draw_count < previous.after.draw_count
                )
            )
        ):
            raise CaptureError("framework_state_samples_invalid")
        records.append(
            {
                "sequence": sample.sequence,
                "present_ticks": sample.present_ticks,
                "host_perf_counter_ns": sample.host_perf_counter_ns,
                "before": _framework_state_row(sample.before),
                "after": _framework_state_row(sample.after),
            }
        )
        previous = sample
    payload = {
        "schema": FRAMEWORK_STATE_SCHEMA,
        "version": FRAMEWORK_STATE_VERSION,
        "diagnostic_only": True,
        "capture_metadata_sha256": (
            f"sha256:{hashlib.sha256(metadata_bytes).hexdigest()}"
        ),
        "framework_update_map_sha256": (
            f"sha256:{hashlib.sha256(framework_update_bytes).hexdigest()}"
        ),
        "process_id": int(target_identity["process_id"]),
        "process_creation_filetime_100ns": int(
            target_identity["process_creation_filetime_100ns"]
        ),
        "executable_sha256": target_identity["executable_sha256"],
        "records": records,
    }
    try:
        encoded = (
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")
    except (OverflowError, TypeError, UnicodeError, ValueError):
        raise CaptureError("framework_state_encode_failed") from None
    part = path.with_name(path.name + ".part")
    try:
        with part.open("xb") as stream:
            if stream.write(encoded) != len(encoded):
                raise CaptureError("framework_state_write_failed")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(part, path)
        part.unlink()
    except CaptureError:
        raise
    except (FileExistsError, OSError):
        raise CaptureError("framework_state_publish_failed") from None
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def capture_to_directory(
    *,
    camera: Any,
    output: Path,
    global_region: Region,
    local_region: Region,
    duration_seconds: float,
    frame_budget_fps: int,
    process_name: str,
    device_index: int = 0,
    output_index: int = 0,
    runtime_versions: dict[str, str] | None = None,
    source_identity: dict[str, Any] | None = None,
    target_identity: dict[str, Any] | None = None,
    source_guard: Callable[[], None] | None = None,
    framework_update_sampler: Callable[[], int] | None = None,
    framework_update_samples: list[FrameworkUpdateSample] | None = None,
    framework_state_sampler: (
        Callable[[], FrameworkStateSnapshot] | None
    ) = None,
    framework_state_samples: list[FrameworkStateSample] | None = None,
    capture_start_gate: Callable[[], None] | None = None,
    monotonic_ns: Callable[[], int] = time.perf_counter_ns,
    idle_sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Capture frames from an already-created camera and publish artifacts."""

    output = ensure_empty_output_directory(output)
    duration_seconds = parse_duration(duration_seconds)
    frame_budget_fps = parse_frame_budget_fps(frame_budget_fps)
    process_name = validate_process_name(process_name)
    global_region = parse_region(global_region)
    local_region = parse_region(local_region)
    if region_dimensions(global_region) != region_dimensions(local_region):
        raise CaptureError("capture_region_size_mismatch")
    device_index = parse_capture_index(
        device_index,
        error_code="invalid_device_index",
    )
    output_index = parse_capture_index(
        output_index,
        error_code="invalid_output_index",
    )
    state_capture = (
        framework_state_sampler is not None
        or framework_state_samples is not None
    )
    if state_capture:
        invalid_framework_capture = (
            framework_state_sampler is None
            or framework_state_samples is None
            or framework_update_sampler is not None
            or framework_update_samples is None
            or bool(framework_state_samples)
            or bool(framework_update_samples)
        )
    else:
        invalid_framework_capture = (
            (framework_update_sampler is None)
            != (framework_update_samples is None)
            or bool(framework_update_samples)
        )
    if invalid_framework_capture:
        raise CaptureError("framework_update_capture_invalid")
    width, height = region_dimensions(local_region)
    budget = estimated_raw_bytes(
        local_region,
        duration_seconds,
        frame_budget_fps,
    )
    frame_budget = math.ceil(duration_seconds * frame_budget_fps) + 2
    frame_pool = _preallocate_capture_frames(
        camera=camera,
        width=width,
        height=height,
        frame_budget=frame_budget + 1,
    )
    if capture_start_gate is not None:
        capture_start_gate()
    raw_stream, csv_stream, raw_part, csv_part = _open_exclusive_outputs(
        output
    )
    aggregate = hashlib.sha256(_PIXEL_HASH_DOMAIN)
    raw_digest = hashlib.sha256()
    captured_frames: list[_CapturedFrame] = []
    raw_bytes = 0
    frame_count = 0
    idle_polls = 0
    pointer_only_updates = 0
    first_ticks: int | None = None
    try:
        writer = csv.writer(csv_stream, lineterminator="\n")
        writer.writerow(
            (
                "sequence",
                "present_ticks",
                "qpc_frequency",
                "host_perf_counter_ns",
                "accumulated_frames",
                "raw_offset",
                "raw_bytes",
                "frame_sha256",
            )
        )
        (
            duplicator,
            baseline_ticks,
            qpc_frequency,
            start_ns,
            warmup_idle_polls,
            warmup_pointer_only_updates,
        ) = _warm_up_capture(
            camera=camera,
            local_region=local_region,
            width=width,
            height=height,
            destination=(
                None if frame_pool is None else frame_pool[0]
            ),
            monotonic_ns=monotonic_ns,
            idle_sleep=idle_sleep,
            source_guard=source_guard,
        )
        last_ticks = baseline_ticks
        deadline_ns = start_ns + int(
            duration_seconds * 1_000_000_000
        )
        end_ns = start_ns
        last_host_ns = start_ns
        while monotonic_ns() < deadline_ns:
            if _camera_duplicator(camera) is not duplicator:
                raise CaptureError("capture_source_changed")
            stage_start_ns = time.perf_counter_ns()
            state_before = (
                framework_state_sampler()
                if framework_state_sampler is not None
                else None
            )
            update_before = (
                state_before.update_count
                if state_before is not None
                else (
                    framework_update_sampler()
                    if framework_update_sampler is not None
                    else None
                )
            )
            update_before_done_ns = time.perf_counter_ns()
            frame = _grab_owned_frame(
                camera=camera,
                region=local_region,
                width=width,
                height=height,
                destination=(
                    None
                    if frame_pool is None
                    else frame_pool[frame_count]
                ),
            )
            host_ns = monotonic_ns()
            grab_done_ns = time.perf_counter_ns()
            state_after = (
                framework_state_sampler()
                if frame is not None
                and framework_state_sampler is not None
                else None
            )
            update_after = (
                state_after.update_count
                if state_after is not None
                else (
                    framework_update_sampler()
                    if frame is not None
                    and framework_update_sampler is not None
                    else None
                )
            )
            update_after_done_ns = time.perf_counter_ns()
            if frame is None:
                idle_polls += 1
                idle_sleep(0.0005)
                continue
            if source_guard is not None:
                source_guard()
            source_guard_done_ns = time.perf_counter_ns()
            ticks, frequency, accumulated = _dxgi_counters(camera)
            counters_done_ns = time.perf_counter_ns()
            if accumulated == 0:
                pointer_only_updates += 1
                continue
            if qpc_frequency != frequency:
                raise CaptureError("dxgi_frequency_changed")
            if ticks <= last_ticks:
                raise CaptureError("dxgi_ticks_not_increasing")
            if accumulated != 1:
                raise CaptureError(
                    "missed_presentations_detected",
                    diagnostic={
                        "sequence": frame_count,
                        "accumulated_frames": accumulated,
                        "present_ticks": ticks,
                        "last_present_ticks": last_ticks,
                        "framework_update_before": update_before,
                        "framework_update_after": update_after,
                        "sample_before_elapsed_ns": (
                            update_before_done_ns - stage_start_ns
                        ),
                        "grab_elapsed_ns": (
                            grab_done_ns - update_before_done_ns
                        ),
                        "sample_after_elapsed_ns": (
                            update_after_done_ns - grab_done_ns
                        ),
                        "source_guard_elapsed_ns": (
                            source_guard_done_ns
                            - update_after_done_ns
                        ),
                        "counter_read_elapsed_ns": (
                            counters_done_ns - source_guard_done_ns
                        ),
                    },
                )
            if host_ns <= last_host_ns:
                raise CaptureError("host_clock_not_increasing")
            payload = _frame_bytes(frame, width, height)
            if frame_count >= frame_budget:
                raise CaptureError("frame_rate_budget_exceeded")
            if (
                raw_bytes + payload.nbytes > budget
                or raw_bytes + payload.nbytes > MAX_RAW_BYTES
            ):
                raise CaptureError("raw_budget_exceeded")
            captured_frames.append(
                _CapturedFrame(
                    pixels=frame,
                    present_ticks=ticks,
                    qpc_frequency=frequency,
                    host_perf_counter_ns=host_ns,
                    accumulated_frames=accumulated,
                    framework_update_before=update_before,
                    framework_update_after=update_after,
                )
            )
            if framework_update_samples is not None:
                if update_before is None or update_after is None:
                    raise CaptureError("framework_update_capture_invalid")
                framework_update_samples.append(
                    FrameworkUpdateSample(
                        sequence=frame_count,
                        present_ticks=ticks,
                        host_perf_counter_ns=host_ns,
                        update_before=update_before,
                        update_after=update_after,
                    )
                )
            if framework_state_samples is not None:
                if state_before is None or state_after is None:
                    raise CaptureError("framework_update_capture_invalid")
                framework_state_samples.append(
                    FrameworkStateSample(
                        sequence=frame_count,
                        present_ticks=ticks,
                        host_perf_counter_ns=host_ns,
                        before=state_before,
                        after=state_after,
                    )
                )
            raw_bytes += payload.nbytes
            frame_count += 1
            if first_ticks is None:
                first_ticks = ticks
            last_ticks = ticks
            last_host_ns = host_ns
        end_ns = monotonic_ns()
        if end_ns < last_host_ns:
            raise CaptureError("host_clock_not_increasing")
        if frame_count == 0 or first_ticks is None:
            raise CaptureError("no_presented_frames")
        if source_guard is not None:
            source_guard()
        if _camera_duplicator(camera) is not duplicator:
            raise CaptureError("capture_source_changed")

        if frame_pool is not None:
            frame_pool.clear()
        raw_offset = 0
        for sequence, captured in enumerate(captured_frames):
            payload = _frame_bytes(captured.pixels, width, height)
            frame_digest = hashlib.sha256(payload).hexdigest()
            if raw_stream.write(payload) != payload.nbytes:
                raise CaptureError("capture_write_failed")
            raw_digest.update(payload)
            aggregate.update(payload.nbytes.to_bytes(8, "big"))
            aggregate.update(payload)
            writer.writerow(
                (
                    sequence,
                    captured.present_ticks,
                    captured.qpc_frequency,
                    captured.host_perf_counter_ns,
                    captured.accumulated_frames,
                    raw_offset,
                    payload.nbytes,
                    f"sha256:{frame_digest}",
                )
            )
            raw_offset += payload.nbytes
            captured.pixels = None
        if raw_offset != raw_bytes:
            raise CaptureError("capture_write_failed")
        raw_stream.flush()
        csv_stream.flush()
        os.fsync(raw_stream.fileno())
        os.fsync(csv_stream.fileno())
    finally:
        raw_stream.close()
        csv_stream.close()

    raw_final = output / "frames.bgra.raw"
    csv_final = output / "frames.csv"
    metadata_part = output / "metadata.json.part"
    metadata_final = output / "metadata.json"
    if any(path.exists() for path in (raw_final, csv_final, metadata_final)):
        raise CaptureError("output_publish_conflict")
    try:
        if set(output.iterdir()) != {raw_part, csv_part}:
            raise CaptureError("output_publish_conflict")
    except CaptureError:
        raise
    except OSError:
        raise CaptureError("output_publish_failed") from None
    present_span_ticks = last_ticks - first_ticks
    present_span_seconds = (
        present_span_ticks / qpc_frequency
        if frame_count >= 2
        else None
    )
    observed_mean_present_fps = (
        (frame_count - 1) / present_span_seconds
        if present_span_seconds is not None and present_span_seconds > 0
        else None
    )
    capture_elapsed_seconds = (end_ns - start_ns) / 1_000_000_000
    try:
        raw_file_bytes = raw_part.stat().st_size
        frames_csv_bytes = csv_part.stat().st_size
    except OSError:
        raise CaptureError("output_hash_failed") from None
    raw_file_sha256 = _sha256_file(raw_part)
    expected_raw_sha256 = f"sha256:{raw_digest.hexdigest()}"
    if (
        raw_file_bytes != raw_bytes
        or raw_file_sha256 != expected_raw_sha256
    ):
        raise CaptureError("output_verification_failed")
    metadata = {
        "schema": CAPTURE_SCHEMA,
        "version": CAPTURE_VERSION,
        "status": "acquisition_complete",
        "target_process": process_name,
        "device_index": device_index,
        "output_index": output_index,
        "global_region": list(global_region),
        "output_local_region": list(local_region),
        "width": width,
        "height": height,
        "pixel_format": PIXEL_FORMAT,
        "bytes_per_pixel": BYTES_PER_PIXEL,
        "frame_budget_fps": frame_budget_fps,
        "capture_mode": "every_new_present",
        "capture_storage_mode": "memory_then_publish",
        "capture_surface": "dxgi_desktop_client_region_crop",
        "foreground_geometry_guard": source_guard is not None,
        "topmost_or_injected_overlay_detection": False,
        "requires_full_frame_visual_review": True,
        "requested_duration_seconds": duration_seconds,
        "capture_start_perf_counter_ns": start_ns,
        "capture_end_perf_counter_ns": end_ns,
        "capture_elapsed_seconds": capture_elapsed_seconds,
        "estimated_raw_budget_bytes": budget,
        "frame_budget": frame_budget,
        "frame_count": frame_count,
        "warmup_baseline_present_ticks": baseline_ticks,
        "first_present_ticks": first_ticks,
        "last_present_ticks": last_ticks,
        "present_span_ticks": present_span_ticks,
        "present_span_seconds": present_span_seconds,
        "observed_mean_present_fps": observed_mean_present_fps,
        "qpc_frequency": qpc_frequency,
        "missed_presentations": 0,
        "idle_poll_count": idle_polls,
        "pointer_only_update_count": pointer_only_updates,
        "warmup_idle_poll_count": warmup_idle_polls,
        "warmup_pointer_only_update_count": (
            warmup_pointer_only_updates
        ),
        "runtime_versions": dict(sorted((runtime_versions or {}).items())),
        "runtime_versions_verified": runtime_versions is not None,
        "source_identity": source_identity or {},
        "target_identity": target_identity or {},
        "raw_bytes": raw_bytes,
        "raw_sha256": raw_file_sha256,
        "frames_csv_bytes": frames_csv_bytes,
        "frames_csv_rows": frame_count,
        "frames_csv_sha256": _sha256_file(csv_part),
        "aggregate_pixel_sha256": (
            f"sha256:{aggregate.hexdigest()}"
        ),
        "frames_csv": "frames.csv",
        "raw_frames": "frames.bgra.raw",
    }
    encoded_metadata = (
        json.dumps(
            metadata,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")
    try:
        with metadata_part.open("xb") as stream:
            if stream.write(encoded_metadata) != len(encoded_metadata):
                raise CaptureError("capture_write_failed")
            stream.flush()
            os.fsync(stream.fileno())
        raw_part.rename(raw_final)
        csv_part.rename(csv_final)
        metadata_part.rename(metadata_final)
    except (FileExistsError, OSError):
        raise CaptureError("output_publish_failed") from None
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = SafeArgumentParser(
        description=(
            "Capture short raw BGRA evidence from the Zuma client area."
        )
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--duration-seconds",
        default="3.0",
    )
    parser.add_argument(
        "--frame-budget-fps",
        default=str(DEFAULT_FRAME_BUDGET_FPS),
    )
    parser.add_argument(
        "--process",
        default=DEFAULT_PROCESS,
    )
    parser.add_argument(
        "--region",
        nargs=4,
        metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
    )
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--output-index", type=int, default=0)
    parser.add_argument(
        "--defer-locked-executable-hash",
        action="store_true",
    )
    parser.add_argument(
        "--runtime-source-executable",
        type=Path,
        help=(
            "Persistent launcher containing the signed runtime PE. "
            "Required for formal capture when the runtime file is locked."
        ),
    )
    parser.add_argument(
        "--framework-update-output",
        type=Path,
        help=(
            "Write a canonical per-frame framework-update sidecar outside "
            "the raw capture directory."
        ),
    )
    parser.add_argument(
        "--framework-state-output",
        type=Path,
        help=(
            "Write a diagnostic per-frame SexyApp scheduling-state sidecar. "
            "This requires --framework-update-output."
        ),
    )
    parser.add_argument(
        "--wait-until-framework-update",
        help=(
            "After camera allocation, begin warmup only when the native "
            "framework update reaches this value."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _dry_run_report(
    *,
    process_name: str,
    global_region: Region,
    duration_seconds: float,
    frame_budget_fps: int,
    target_identity: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": CAPTURE_SCHEMA,
        "version": CAPTURE_VERSION,
        "status": "dry_run",
        "target_process": process_name,
        "target_identity": target_identity,
        "global_region": list(global_region),
        "width": region_dimensions(global_region)[0],
        "height": region_dimensions(global_region)[1],
        "frame_budget_fps": frame_budget_fps,
        "capture_mode": "every_new_present",
        "requested_duration_seconds": duration_seconds,
        "estimated_raw_bytes": estimated_raw_bytes(
            global_region,
            duration_seconds,
            frame_budget_fps,
        ),
        "pixels_captured": False,
    }


def run(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = ensure_empty_output_directory(args.output_dir)
    framework_update_output = (
        None
        if args.framework_update_output is None
        else prepare_framework_update_output(
            args.framework_update_output,
            capture_output=output,
        )
    )
    framework_state_output = (
        None
        if args.framework_state_output is None
        else prepare_framework_state_output(
            args.framework_state_output,
            capture_output=output,
        )
    )
    wait_until_framework_update = (
        None
        if args.wait_until_framework_update is None
        else parse_framework_update(args.wait_until_framework_update)
    )
    duration = parse_duration(args.duration_seconds)
    frame_budget_fps = parse_frame_budget_fps(args.frame_budget_fps)
    device_index = parse_capture_index(
        args.device_index,
        error_code="invalid_device_index",
    )
    output_index = parse_capture_index(
        args.output_index,
        error_code="invalid_output_index",
    )
    process_name = validate_process_name(args.process)
    defer_locked_executable_hash = bool(
        args.defer_locked_executable_hash
    )
    runtime_source_executable = args.runtime_source_executable
    if defer_locked_executable_hash and (
        process_name.casefold() != "popcapgame1.exe"
        or duration != MIN_DURATION_SECONDS
        or args.dry_run
        or runtime_source_executable is not None
    ):
        raise CaptureError("deferred_executable_hash_probe_only")
    if framework_update_output is not None and (
        process_name.casefold() != "popcapgame1.exe"
        or args.dry_run
        or defer_locked_executable_hash
    ):
        raise CaptureError("framework_update_capture_invalid")
    if (
        wait_until_framework_update is not None
        and framework_update_output is None
    ):
        raise CaptureError("framework_update_capture_invalid")
    if framework_state_output is not None and (
        framework_update_output is None
        or framework_state_output == framework_update_output
        or process_name.casefold() != "popcapgame1.exe"
        or args.dry_run
        or defer_locked_executable_hash
    ):
        raise CaptureError("framework_state_capture_invalid")
    if runtime_source_executable is not None:
        if (
            process_name.casefold() != "popcapgame1.exe"
            or not runtime_source_executable.is_absolute()
        ):
            raise CaptureError("runtime_source_identity_invalid")
    explicit = None if args.region is None else parse_region(args.region)
    target, global_region = resolve_windows_target(
        process_name=process_name,
        explicit_region=explicit,
    )
    verify_windows_target(target)
    target_identity = windows_process_identity(
        target,
        process_name=process_name,
        allow_deferred_executable_hash=(
            defer_locked_executable_hash
        ),
        runtime_source_executable=runtime_source_executable,
    )
    estimated_raw_bytes(global_region, duration, frame_budget_fps)
    if args.dry_run:
        print(
            json.dumps(
                _dry_run_report(
                    process_name=process_name,
                    global_region=global_region,
                    duration_seconds=duration,
                    frame_budget_fps=frame_budget_fps,
                    target_identity=target_identity,
                ),
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        return 0

    _require_windows()
    try:
        import dxcam
    except (ImportError, OSError, RuntimeError):
        raise CaptureError("dxcam_unavailable") from None
    runtime_versions = validate_runtime_versions()
    camera = None
    update_reader: FrameworkUpdateReader | None = None
    update_samples: list[FrameworkUpdateSample] | None = (
        [] if framework_update_output is not None else None
    )
    state_samples: list[FrameworkStateSample] | None = (
        [] if framework_state_output is not None else None
    )
    try:
        if framework_update_output is not None:
            update_reader = FrameworkUpdateReader(target.process_id)
        camera = dxcam.create(
            device_idx=device_index,
            output_idx=output_index,
            output_color="BGRA",
            backend="dxgi",
            processor_backend="numpy",
        )
        local_region = output_local_region(camera, global_region)
        source_identity = capture_source_identity(camera)

        def source_guard() -> None:
            verify_windows_target(target)
            if output_local_region(camera, global_region) != local_region:
                raise CaptureError("capture_source_changed")

        capture_start_gate = None
        if wait_until_framework_update is not None:
            assert update_reader is not None

            def capture_start_gate() -> None:
                deadline = time.monotonic() + 120.0
                last_update = -1
                while time.monotonic() < deadline:
                    source_guard()
                    last_update = update_reader.sample()
                    if last_update >= wait_until_framework_update:
                        return
                    time.sleep(0.0005)
                raise CaptureError("framework_update_wait_timeout")

        metadata = capture_to_directory(
            camera=camera,
            output=output,
            global_region=global_region,
            local_region=local_region,
            duration_seconds=duration,
            frame_budget_fps=frame_budget_fps,
            process_name=process_name,
            device_index=device_index,
            output_index=output_index,
            runtime_versions=runtime_versions,
            source_identity=source_identity,
            target_identity=target_identity,
            source_guard=source_guard,
            framework_update_sampler=(
                None
                if update_reader is None
                or framework_state_output is not None
                else update_reader.sample
            ),
            framework_update_samples=update_samples,
            framework_state_sampler=(
                None
                if update_reader is None
                or framework_state_output is None
                else update_reader.sample_state
            ),
            framework_state_samples=state_samples,
            capture_start_gate=capture_start_gate,
        )
    finally:
        if update_reader is not None:
            update_reader.close()
        if camera is not None:
            try:
                camera.release()
            except Exception:
                pass
    framework_update_sha256 = None
    framework_state_sha256 = None
    if framework_update_output is not None:
        assert update_samples is not None
        try:
            metadata_bytes = (output / "metadata.json").read_bytes()
        except OSError:
            raise CaptureError("framework_update_metadata_read_failed") from None
        framework_update_sha256 = write_framework_update_sidecar(
            framework_update_output,
            metadata_bytes=metadata_bytes,
            target_identity=target_identity,
            samples=update_samples,
        )
    if framework_state_output is not None:
        assert state_samples is not None
        assert framework_update_output is not None
        try:
            framework_update_bytes = framework_update_output.read_bytes()
        except OSError:
            raise CaptureError("framework_state_update_map_read_failed") from None
        framework_state_sha256 = write_framework_state_sidecar(
            framework_state_output,
            metadata_bytes=metadata_bytes,
            framework_update_bytes=framework_update_bytes,
            target_identity=target_identity,
            samples=state_samples,
        )
    print(
        json.dumps(
            {
                "status": "acquisition_complete",
                "frame_count": metadata["frame_count"],
                "missed_presentations": metadata[
                    "missed_presentations"
                ],
                "raw_bytes": metadata["raw_bytes"],
                "framework_update_sha256": framework_update_sha256,
                "framework_state_sha256": framework_state_sha256,
            },
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(argv)
    except CaptureError as error:
        print(f"capture error: {error.code}", file=sys.stderr)
        if error.diagnostic is not None:
            print(
                "capture diagnostic: "
                + json.dumps(
                    error.diagnostic,
                    allow_nan=False,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
        return 1
    except KeyboardInterrupt:
        print("capture error: interrupted", file=sys.stderr)
        return 130
    except Exception:
        print("capture error: unexpected_failure", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
