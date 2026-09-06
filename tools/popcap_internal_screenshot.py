"""Probe the purported retail F11 screenshot path with safe cleanup.

Static analysis later established that F11 is not a gameplay-screenshot hotkey
in the pinned retail executable.  This module is retained only so the negative
live probes remain reproducible; it must not be used as PC Golden evidence.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import hashlib
import os
from pathlib import Path
import struct
import time
from typing import Any, Callable
import zlib


VK_F11 = 0x7A
WM_KEYDOWN = 0x0100
WM_KEYUP = 0x0101
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
DEFAULT_SCREENSHOT_ROOT = Path(
    r"C:\ProgramData\Steam\ZumasRevenge\_screenshots"
)
MAX_SCREENSHOT_BYTES = 64 * 1024 * 1024


class InternalScreenshotError(RuntimeError):
    """The retail screenshot command did not produce one valid image."""


def _sha256_bytes(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _png_facts(payload: bytes) -> dict[str, Any]:
    if len(payload) < 33 or len(payload) > MAX_SCREENSHOT_BYTES:
        raise InternalScreenshotError("internal_screenshot_size_invalid")
    if payload[:8] != PNG_SIGNATURE:
        raise InternalScreenshotError("internal_screenshot_not_png")
    offset = 8
    chunks: list[str] = []
    width = height = bit_depth = color_type = None
    saw_iend = False
    while offset < len(payload):
        if offset + 12 > len(payload):
            raise InternalScreenshotError("internal_screenshot_png_truncated")
        length = struct.unpack(">I", payload[offset : offset + 4])[0]
        chunk_type = payload[offset + 4 : offset + 8]
        data_start = offset + 8
        data_end = data_start + length
        crc_end = data_end + 4
        if crc_end > len(payload):
            raise InternalScreenshotError("internal_screenshot_png_truncated")
        stored_crc = struct.unpack(">I", payload[data_end:crc_end])[0]
        computed_crc = zlib.crc32(chunk_type)
        computed_crc = zlib.crc32(payload[data_start:data_end], computed_crc)
        if stored_crc != computed_crc & 0xFFFFFFFF:
            raise InternalScreenshotError("internal_screenshot_png_crc_invalid")
        try:
            chunk_name = chunk_type.decode("ascii")
        except UnicodeDecodeError:
            raise InternalScreenshotError(
                "internal_screenshot_png_chunk_invalid"
            ) from None
        chunks.append(chunk_name)
        if len(chunks) == 1:
            if chunk_type != b"IHDR" or length != 13:
                raise InternalScreenshotError(
                    "internal_screenshot_png_ihdr_invalid"
                )
            (
                width,
                height,
                bit_depth,
                color_type,
                compression,
                filter_method,
                interlace,
            ) = struct.unpack(">IIBBBBB", payload[data_start:data_end])
            if (
                width != 800
                or height != 600
                or bit_depth != 8
                or color_type not in (2, 6)
                or compression != 0
                or filter_method != 0
                or interlace not in (0, 1)
            ):
                raise InternalScreenshotError(
                    "internal_screenshot_png_geometry_invalid"
                )
        if chunk_type == b"IEND":
            if length != 0 or crc_end != len(payload):
                raise InternalScreenshotError(
                    "internal_screenshot_png_iend_invalid"
                )
            saw_iend = True
            break
        offset = crc_end
    if not saw_iend or "IDAT" not in chunks:
        raise InternalScreenshotError("internal_screenshot_png_incomplete")
    return {
        "width": width,
        "height": height,
        "bit_depth": bit_depth,
        "color_type": color_type,
        "chunk_count": len(chunks),
    }


def post_virtual_key(window_handle: int, virtual_key: int) -> None:
    if os.name != "nt":
        raise InternalScreenshotError("internal_screenshot_requires_windows")
    if window_handle <= 0 or not 1 <= virtual_key <= 0xFE:
        raise InternalScreenshotError("internal_screenshot_key_invalid")
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.PostMessageW.argtypes = (
        wintypes.HWND,
        wintypes.UINT,
        wintypes.WPARAM,
        wintypes.LPARAM,
    )
    user32.PostMessageW.restype = wintypes.BOOL
    user32.MapVirtualKeyW.argtypes = (wintypes.UINT, wintypes.UINT)
    user32.MapVirtualKeyW.restype = wintypes.UINT
    scan_code = int(user32.MapVirtualKeyW(virtual_key, 0)) & 0xFF
    down_lparam = 1 | (scan_code << 16)
    up_lparam = down_lparam | (1 << 30) | (1 << 31)
    if not user32.PostMessageW(
        wintypes.HWND(window_handle),
        WM_KEYDOWN,
        virtual_key,
        down_lparam,
    ):
        raise InternalScreenshotError("internal_screenshot_keydown_failed")
    if not user32.PostMessageW(
        wintypes.HWND(window_handle),
        WM_KEYUP,
        virtual_key,
        up_lparam,
    ):
        raise InternalScreenshotError("internal_screenshot_keyup_failed")


def send_virtual_key(window_handle: int, virtual_key: int) -> None:
    """Deliver one real keyboard edge after the caller verifies foreground."""

    if os.name != "nt":
        raise InternalScreenshotError("internal_screenshot_requires_windows")
    if window_handle <= 0 or not 1 <= virtual_key <= 0xFE:
        raise InternalScreenshotError("internal_screenshot_key_invalid")

    class KeyboardInput(ctypes.Structure):
        _fields_ = (
            ("wVk", wintypes.WORD),
            ("wScan", wintypes.WORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ctypes.c_size_t),
        )

    class MouseInput(ctypes.Structure):
        _fields_ = (
            ("dx", wintypes.LONG),
            ("dy", wintypes.LONG),
            ("mouseData", wintypes.DWORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ctypes.c_size_t),
        )

    class HardwareInput(ctypes.Structure):
        _fields_ = (
            ("uMsg", wintypes.DWORD),
            ("wParamL", wintypes.WORD),
            ("wParamH", wintypes.WORD),
        )

    class InputUnion(ctypes.Union):
        _fields_ = (
            ("mi", MouseInput),
            ("ki", KeyboardInput),
            ("hi", HardwareInput),
        )

    class Input(ctypes.Structure):
        _anonymous_ = ("payload",)
        _fields_ = (
            ("type", wintypes.DWORD),
            ("payload", InputUnion),
        )

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.GetForegroundWindow.argtypes = ()
    user32.GetForegroundWindow.restype = wintypes.HWND
    user32.SendInput.argtypes = (
        wintypes.UINT,
        ctypes.POINTER(Input),
        ctypes.c_int,
    )
    user32.SendInput.restype = wintypes.UINT
    if int(user32.GetForegroundWindow() or 0) != window_handle:
        raise InternalScreenshotError(
            "internal_screenshot_window_not_foreground"
        )
    items = (Input * 2)(
        Input(
            type=1,
            payload=InputUnion(
                ki=KeyboardInput(
                    wVk=virtual_key,
                    wScan=0,
                    dwFlags=0,
                    time=0,
                    dwExtraInfo=0,
                )
            ),
        ),
        Input(
            type=1,
            payload=InputUnion(
                ki=KeyboardInput(
                    wVk=virtual_key,
                    wScan=0,
                    dwFlags=0x0002,
                    time=0,
                    dwExtraInfo=0,
                )
            ),
        ),
    )
    if user32.SendInput(2, items, ctypes.sizeof(Input)) != 2:
        raise InternalScreenshotError("internal_screenshot_send_input_failed")


def _inventory(root: Path) -> dict[str, tuple[int, int]]:
    if not root.exists():
        return {}
    if not root.is_dir() or root.is_symlink():
        raise InternalScreenshotError("internal_screenshot_root_invalid")
    result: dict[str, tuple[int, int]] = {}
    for path in root.iterdir():
        if path.is_symlink() or not path.is_file():
            raise InternalScreenshotError("internal_screenshot_root_unsafe")
        stat = path.stat()
        result[path.name] = (stat.st_size, stat.st_mtime_ns)
    return result


def capture_internal_screenshot(
    *,
    window_handle: int,
    output: Path,
    screenshot_root: Path = DEFAULT_SCREENSHOT_ROOT,
    additional_screenshot_roots: tuple[Path, ...] = (),
    timeout: float = 5.0,
    post_key: Callable[[int, int], None] = post_virtual_key,
    monotonic: Callable[[], float] = time.monotonic,
    delay: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Request F11, copy exactly one new PNG, then remove only that PNG."""

    output = output.resolve()
    screenshot_roots = tuple(
        dict.fromkeys(
            root.resolve()
            for root in (screenshot_root, *additional_screenshot_roots)
        )
    )
    if output.exists() or not output.parent.is_dir():
        raise InternalScreenshotError("internal_screenshot_output_invalid")
    if timeout <= 0 or timeout > 30:
        raise InternalScreenshotError("internal_screenshot_timeout_invalid")
    root_preexisted = {
        root: root.exists() for root in screenshot_roots
    }
    before = {root: _inventory(root) for root in screenshot_roots}
    started_ns = time.perf_counter_ns()
    post_key(window_handle, VK_F11)
    deadline = monotonic() + timeout
    candidate: tuple[Path, Path] | None = None
    stable_identity: tuple[int, int] | None = None
    stable_count = 0
    while monotonic() < deadline:
        current = {root: _inventory(root) for root in screenshot_roots}
        new_files = sorted(
            (
                (
                    root,
                    name,
                    current[root][name],
                )
                for root in screenshot_roots
                for name in set(current[root]).difference(before[root])
            ),
            key=lambda row: (str(row[0]), row[1]),
        )
        if len(new_files) > 1:
            raise InternalScreenshotError(
                "internal_screenshot_multiple_new_files"
            )
        if len(new_files) == 1:
            root, name, identity = new_files[0]
            if Path(name).suffix.casefold() != ".png":
                raise InternalScreenshotError(
                    "internal_screenshot_extension_invalid"
                )
            if identity[0] > 0 and identity == stable_identity:
                stable_count += 1
            else:
                stable_identity = identity
                stable_count = 1
            if stable_count >= 2:
                candidate = (root, root / name)
                break
        delay(0.025)
    if candidate is None:
        raise InternalScreenshotError("internal_screenshot_not_created")
    source_root, source_path = candidate
    payload = source_path.read_bytes()
    facts = _png_facts(payload)
    with output.open("xb") as stream:
        if stream.write(payload) != len(payload):
            raise InternalScreenshotError("internal_screenshot_copy_failed")
        stream.flush()
        os.fsync(stream.fileno())
    if output.read_bytes() != payload:
        raise InternalScreenshotError("internal_screenshot_copy_mismatch")
    source_name = source_path.name
    source_path.unlink()
    source_removed = not source_path.exists()
    root_removed = False
    if not root_preexisted[source_root]:
        try:
            source_root.rmdir()
            root_removed = True
        except OSError:
            root_removed = False
    if not source_removed:
        raise InternalScreenshotError("internal_screenshot_cleanup_failed")
    finished_ns = time.perf_counter_ns()
    return {
        "schema": "zuma-rl.pc-internal-screenshot-diagnostic",
        "version": 1,
        "mechanism": "retail_f11_internal_backbuffer_png",
        "window_handle_hex": f"0x{window_handle:016x}",
        "source_directory": str(source_root),
        "monitored_source_directories": [
            str(root) for root in screenshot_roots
        ],
        "source_filename": source_name,
        "preexisting_file_count": sum(
            len(files) for files in before.values()
        ),
        "artifact": output.name,
        "artifact_bytes": len(payload),
        "artifact_sha256": _sha256_bytes(payload),
        "png": facts,
        "source_file_removed_after_verified_copy": source_removed,
        "new_source_directory_removed": root_removed,
        "started_perf_counter_ns": started_ns,
        "finished_perf_counter_ns": finished_ns,
    }


__all__ = [
    "DEFAULT_SCREENSHOT_ROOT",
    "InternalScreenshotError",
    "VK_F11",
    "capture_internal_screenshot",
    "post_virtual_key",
    "send_virtual_key",
]
