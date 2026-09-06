"""Exact, reversible Win32 display-mode control for PC evidence runs.

The PC Golden collector samples a legacy 100 Hz game through DXGI desktop
duplication.  Keeping display-mode selection in a small module makes the
temporary host mutation explicit, testable, and independently auditable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import ctypes
from ctypes import wintypes
import os
from typing import Iterable


ENUM_CURRENT_SETTINGS = -1

CDS_FULLSCREEN = 0x00000004
CDS_TEST = 0x00000002
DISP_CHANGE_SUCCESSFUL = 0

DM_BITSPERPEL = 0x00040000
DM_PELSWIDTH = 0x00080000
DM_PELSHEIGHT = 0x00100000
DM_DISPLAYFLAGS = 0x00200000
DM_DISPLAYFREQUENCY = 0x00400000
_DISPLAY_MODE_FIELDS = (
    DM_BITSPERPEL
    | DM_PELSWIDTH
    | DM_PELSHEIGHT
    | DM_DISPLAYFLAGS
    | DM_DISPLAYFREQUENCY
)


class DisplayModeError(RuntimeError):
    """A display mode could not be enumerated, selected, or applied."""


@dataclass(frozen=True, slots=True)
class DisplayMode:
    """The five display fields guaranteed by ``EnumDisplaySettingsW``."""

    bits_per_pixel: int
    width: int
    height: int
    display_flags: int
    refresh_rate_hz: int

    def __post_init__(self) -> None:
        if (
            self.bits_per_pixel <= 0
            or self.width <= 0
            or self.height <= 0
            or self.display_flags < 0
            or self.refresh_rate_hz < 0
        ):
            raise ValueError("display mode fields are invalid")

    def to_dict(self) -> dict[str, int]:
        """Return a stable JSON-ready identity."""

        return asdict(self)


def select_exact_display_mode(
    modes: Iterable[DisplayMode],
    *,
    width: int,
    height: int,
    refresh_rate_hz: int,
    current: DisplayMode | None = None,
) -> DisplayMode:
    """Select one exact geometry/rate mode without silently approximating."""

    if width <= 0 or height <= 0 or refresh_rate_hz <= 1:
        raise ValueError("requested display mode is invalid")
    candidates = {
        mode
        for mode in modes
        if mode.width == width
        and mode.height == height
        and mode.refresh_rate_hz == refresh_rate_hz
    }
    if not candidates:
        raise DisplayModeError("requested_display_mode_unavailable")
    return sorted(
        candidates,
        key=lambda mode: (
            0
            if current is not None
            and mode.bits_per_pixel == current.bits_per_pixel
            else 1,
            0
            if current is not None
            and mode.display_flags == current.display_flags
            else 1,
            -mode.bits_per_pixel,
            mode.display_flags,
        ),
    )[0]


if os.name == "nt":

    class _POINTL(ctypes.Structure):
        _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]


    class _PRINTER_FIELDS(ctypes.Structure):
        _fields_ = [
            ("dmOrientation", wintypes.SHORT),
            ("dmPaperSize", wintypes.SHORT),
            ("dmPaperLength", wintypes.SHORT),
            ("dmPaperWidth", wintypes.SHORT),
            ("dmScale", wintypes.SHORT),
            ("dmCopies", wintypes.SHORT),
            ("dmDefaultSource", wintypes.SHORT),
            ("dmPrintQuality", wintypes.SHORT),
        ]


    class _DISPLAY_FIELDS(ctypes.Structure):
        _fields_ = [
            ("dmPosition", _POINTL),
            ("dmDisplayOrientation", wintypes.DWORD),
            ("dmDisplayFixedOutput", wintypes.DWORD),
        ]


    class _DEVMODE_UNION_1(ctypes.Union):
        _anonymous_ = ("printer", "display")
        _fields_ = [
            ("printer", _PRINTER_FIELDS),
            ("display", _DISPLAY_FIELDS),
        ]


    class _DEVMODE_UNION_2(ctypes.Union):
        _fields_ = [
            ("dmDisplayFlags", wintypes.DWORD),
            ("dmNup", wintypes.DWORD),
        ]


    class _DEVMODEW(ctypes.Structure):
        _anonymous_ = ("mode_fields", "display_flags_union")
        _fields_ = [
            ("dmDeviceName", wintypes.WCHAR * 32),
            ("dmSpecVersion", wintypes.WORD),
            ("dmDriverVersion", wintypes.WORD),
            ("dmSize", wintypes.WORD),
            ("dmDriverExtra", wintypes.WORD),
            ("dmFields", wintypes.DWORD),
            ("mode_fields", _DEVMODE_UNION_1),
            ("dmColor", wintypes.SHORT),
            ("dmDuplex", wintypes.SHORT),
            ("dmYResolution", wintypes.SHORT),
            ("dmTTOption", wintypes.SHORT),
            ("dmCollate", wintypes.SHORT),
            ("dmFormName", wintypes.WCHAR * 32),
            ("dmLogPixels", wintypes.WORD),
            ("dmBitsPerPel", wintypes.DWORD),
            ("dmPelsWidth", wintypes.DWORD),
            ("dmPelsHeight", wintypes.DWORD),
            ("display_flags_union", _DEVMODE_UNION_2),
            ("dmDisplayFrequency", wintypes.DWORD),
            ("dmICMMethod", wintypes.DWORD),
            ("dmICMIntent", wintypes.DWORD),
            ("dmMediaType", wintypes.DWORD),
            ("dmDitherType", wintypes.DWORD),
            ("dmReserved1", wintypes.DWORD),
            ("dmReserved2", wintypes.DWORD),
            ("dmPanningWidth", wintypes.DWORD),
            ("dmPanningHeight", wintypes.DWORD),
        ]


    _user32 = ctypes.WinDLL("user32", use_last_error=True)
    _user32.EnumDisplaySettingsW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.POINTER(_DEVMODEW),
    ]
    _user32.EnumDisplaySettingsW.restype = wintypes.BOOL
    _user32.ChangeDisplaySettingsExW.argtypes = [
        wintypes.LPCWSTR,
        ctypes.POINTER(_DEVMODEW),
        wintypes.HWND,
        wintypes.DWORD,
        wintypes.LPVOID,
    ]
    _user32.ChangeDisplaySettingsExW.restype = wintypes.LONG


def _require_windows() -> None:
    if os.name != "nt":
        raise DisplayModeError("windows_required")


def _new_devmode() -> "_DEVMODEW":
    _require_windows()
    value = _DEVMODEW()
    value.dmSize = ctypes.sizeof(_DEVMODEW)
    value.dmDriverExtra = 0
    return value


def _display_mode(value: "_DEVMODEW") -> DisplayMode:
    return DisplayMode(
        bits_per_pixel=int(value.dmBitsPerPel),
        width=int(value.dmPelsWidth),
        height=int(value.dmPelsHeight),
        display_flags=int(value.dmDisplayFlags),
        refresh_rate_hz=int(value.dmDisplayFrequency),
    )


def current_display_mode() -> DisplayMode:
    """Read the primary display's current physical-pixel mode."""

    value = _new_devmode()
    if not _user32.EnumDisplaySettingsW(
        None,
        ctypes.c_uint32(ENUM_CURRENT_SETTINGS).value,
        ctypes.byref(value),
    ):
        raise DisplayModeError("current_display_mode_unavailable")
    return _display_mode(value)


def available_display_modes() -> tuple[DisplayMode, ...]:
    """Enumerate unique primary-display modes in driver order."""

    _require_windows()
    modes: list[DisplayMode] = []
    seen: set[DisplayMode] = set()
    index = 0
    while True:
        value = _new_devmode()
        if not _user32.EnumDisplaySettingsW(
            None,
            index,
            ctypes.byref(value),
        ):
            break
        mode = _display_mode(value)
        if mode not in seen:
            seen.add(mode)
            modes.append(mode)
        index += 1
    if not modes:
        raise DisplayModeError("display_mode_enumeration_empty")
    return tuple(modes)


def _native_mode(mode: DisplayMode) -> "_DEVMODEW":
    for index in range(100_000):
        value = _new_devmode()
        if not _user32.EnumDisplaySettingsW(
            None,
            index,
            ctypes.byref(value),
        ):
            break
        if _display_mode(value) == mode:
            value.dmFields = _DISPLAY_MODE_FIELDS
            return value
    raise DisplayModeError("exact_display_mode_disappeared")


def test_display_mode(mode: DisplayMode) -> None:
    """Ask the driver to validate an exact enumerated mode without applying it."""

    value = _native_mode(mode)
    result = int(
        _user32.ChangeDisplaySettingsExW(
            None,
            ctypes.byref(value),
            None,
            CDS_TEST,
            None,
        )
    )
    if result != DISP_CHANGE_SUCCESSFUL:
        raise DisplayModeError(f"display_mode_test_failed:{result}")


def apply_temporary_display_mode(mode: DisplayMode) -> DisplayMode:
    """Apply one enumerated mode temporarily and verify the exact result."""

    value = _native_mode(mode)
    result = int(
        _user32.ChangeDisplaySettingsExW(
            None,
            ctypes.byref(value),
            None,
            CDS_FULLSCREEN,
            None,
        )
    )
    if result != DISP_CHANGE_SUCCESSFUL:
        raise DisplayModeError(f"display_mode_apply_failed:{result}")
    observed = current_display_mode()
    if observed != mode:
        raise DisplayModeError("display_mode_apply_verification_failed")
    return observed


__all__ = [
    "DisplayMode",
    "DisplayModeError",
    "apply_temporary_display_mode",
    "available_display_modes",
    "current_display_mode",
    "select_exact_display_mode",
    "test_display_mode",
]
