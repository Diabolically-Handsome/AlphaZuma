"""Read-only inspection helpers for a running PopCap demo replay.

The retail game exposes useful demo controls but no machine-readable status.
This diagnostic locates an exact IEEE-754 update multiplier in readable
process memory and prints nearby primitive values.  It never requests process
write access and cannot alter the target.
"""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
from dataclasses import dataclass
import math
import os
import struct
import sys
from typing import Iterable, Iterator


PROCESS_QUERY_INFORMATION = 0x0400
PROCESS_VM_READ = 0x0010
MEM_COMMIT = 0x1000
PAGE_GUARD = 0x100
PAGE_NOACCESS = 0x01


@dataclass(frozen=True)
class ReplayState:
    """Known SexyAppBase fields surrounding ``mUpdateMultiplier``."""

    multiplier_address: int
    non_draw_count: int
    frame_time_ms: int
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
    step_mode: int
    loading_thread_started: bool
    loading_thread_completed: bool
    loaded: bool


def exact_value_offsets(data: bytes, value: float) -> tuple[int, ...]:
    """Return offsets of the exact little-endian float64 representation."""

    if not math.isfinite(value):
        raise ValueError("value must be finite")
    needle = struct.pack("<d", value)
    offsets: list[int] = []
    start = 0
    while True:
        offset = data.find(needle, start)
        if offset < 0:
            return tuple(offsets)
        offsets.append(offset)
        start = offset + 1


def primitive_window(data: bytes, center: int, radius: int = 64) -> list[dict[str, object]]:
    """Decode aligned int32/float64 values around a byte offset."""

    if center < 0 or radius < 0:
        raise ValueError("center and radius must be non-negative")
    start = max(0, center - radius)
    end = min(len(data), center + 8 + radius)
    rows: list[dict[str, object]] = []
    for offset in range(start - (start % 4), end - 3, 4):
        int32 = struct.unpack_from("<i", data, offset)[0]
        float64: float | None = None
        if offset + 8 <= len(data):
            candidate = struct.unpack_from("<d", data, offset)[0]
            if math.isfinite(candidate):
                float64 = candidate
        rows.append(
            {
                "relative_offset": offset - center,
                "int32": int32,
                "float64": float64,
            }
        )
    return rows


def decode_replay_state(data: bytes, multiplier_offset: int, address: int) -> ReplayState:
    """Decode stable framework fields around an exact multiplier match."""

    if multiplier_offset < 60 or multiplier_offset + 108 > len(data):
        raise ValueError("insufficient bytes around multiplier")

    def i32(relative: int) -> int:
        return struct.unpack_from("<i", data, multiplier_offset + relative)[0]

    return ReplayState(
        multiplier_address=address,
        non_draw_count=i32(-60),
        frame_time_ms=i32(-56),
        sleep_count=i32(-20),
        draw_count=i32(-16),
        update_count=i32(-12),
        update_app_state=i32(-8),
        update_app_depth=i32(-4),
        update_multiplier=struct.unpack_from("<d", data, multiplier_offset)[0],
        paused=bool(data[multiplier_offset + 8]),
        fast_forward_target=i32(12),
        fast_forward_to_marker=bool(data[multiplier_offset + 16]),
        fast_forward_step=bool(data[multiplier_offset + 17]),
        step_mode=i32(28),
        loading_thread_started=bool(data[multiplier_offset + 105]),
        loading_thread_completed=bool(data[multiplier_offset + 106]),
        loaded=bool(data[multiplier_offset + 107]),
    )


if os.name == "nt":
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class MEMORY_BASIC_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BaseAddress", ctypes.c_void_p),
            ("AllocationBase", ctypes.c_void_p),
            ("AllocationProtect", wintypes.DWORD),
            ("PartitionId", wintypes.WORD),
            ("RegionSize", ctypes.c_size_t),
            ("State", wintypes.DWORD),
            ("Protect", wintypes.DWORD),
            ("Type", wintypes.DWORD),
        ]

    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.VirtualQueryEx.argtypes = [
        wintypes.HANDLE,
        ctypes.c_void_p,
        ctypes.POINTER(MEMORY_BASIC_INFORMATION),
        ctypes.c_size_t,
    ]
    kernel32.VirtualQueryEx.restype = ctypes.c_size_t
    kernel32.ReadProcessMemory.argtypes = [
        wintypes.HANDLE,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    kernel32.ReadProcessMemory.restype = wintypes.BOOL


def _readable_regions(handle: int) -> Iterator[tuple[int, int]]:
    address = 0
    maximum = (1 << (ctypes.sizeof(ctypes.c_void_p) * 8)) - 1
    mbi = MEMORY_BASIC_INFORMATION()
    mbi_size = ctypes.sizeof(mbi)
    while address < maximum:
        result = kernel32.VirtualQueryEx(
            handle, ctypes.c_void_p(address), ctypes.byref(mbi), mbi_size
        )
        if result == 0:
            break
        base = int(mbi.BaseAddress or 0)
        size = int(mbi.RegionSize)
        if (
            mbi.State == MEM_COMMIT
            and not (mbi.Protect & PAGE_GUARD)
            and not (mbi.Protect & PAGE_NOACCESS)
            and size > 0
        ):
            yield base, size
        next_address = base + max(size, 0x1000)
        if next_address <= address:
            break
        address = next_address


def _read_region(handle: int, base: int, size: int) -> bytes | None:
    if size <= 0 or size > 512 * 1024 * 1024:
        return None
    buffer = ctypes.create_string_buffer(size)
    read = ctypes.c_size_t()
    ok = kernel32.ReadProcessMemory(
        handle,
        ctypes.c_void_p(base),
        buffer,
        size,
        ctypes.byref(read),
    )
    if not ok or read.value == 0:
        return None
    return buffer.raw[: read.value]


def open_process_readonly(pid: int) -> int:
    """Open a process with query/read rights only."""

    if os.name != "nt":
        raise RuntimeError("live process inspection requires Windows")
    handle = kernel32.OpenProcess(
        PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, pid
    )
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    return handle


def close_process(handle: int) -> None:
    if os.name == "nt" and handle:
        kernel32.CloseHandle(handle)


def read_process_bytes(handle: int, address: int, size: int) -> bytes:
    data = _read_region(handle, address, size)
    if data is None or len(data) != size:
        raise RuntimeError(f"could not read {size} bytes at 0x{address:X}")
    return data


def locate_exact_float64(handle: int, value: float) -> tuple[int, ...]:
    """Locate all exact float64 values in committed readable regions."""

    addresses: list[int] = []
    for base, size in _readable_regions(handle):
        data = _read_region(handle, base, size)
        if data is None:
            continue
        addresses.extend(base + offset for offset in exact_value_offsets(data, value))
    return tuple(addresses)


def read_replay_state(handle: int, multiplier_address: int) -> ReplayState:
    """Read the known field window without scanning the process again."""

    start = multiplier_address - 60
    data = read_process_bytes(handle, start, 176)
    return decode_replay_state(data, 60, multiplier_address)


def inspect_process(pid: int, value: float, radius: int) -> list[dict[str, object]]:
    """Find exact float64 matches in readable memory and decode their vicinity."""

    if os.name != "nt":
        raise RuntimeError("live process inspection requires Windows")
    handle = open_process_readonly(pid)
    matches: list[dict[str, object]] = []
    try:
        for base, size in _readable_regions(handle):
            data = _read_region(handle, base, size)
            if data is None:
                continue
            for offset in exact_value_offsets(data, value):
                matches.append(
                    {
                        "address": base + offset,
                        "region_base": base,
                        "region_size": len(data),
                        "nearby": primitive_window(data, offset, radius),
                    }
                )
    finally:
        kernel32.CloseHandle(handle)
    return matches


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only exact-value scan of a running PopCap replay."
    )
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--minus-count", type=int)
    parser.add_argument("--value", type=float)
    parser.add_argument("--radius", type=int, default=64)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if (args.minus_count is None) == (args.value is None):
        raise SystemExit("provide exactly one of --minus-count or --value")
    if args.minus_count is not None:
        if args.minus_count < 0:
            raise SystemExit("--minus-count must be non-negative")
        value = 1.0 / (1.5**args.minus_count)
    else:
        value = float(args.value)
    matches = inspect_process(args.pid, value, args.radius)
    print(f"target_float64={value!r}")
    print(f"matches={len(matches)}")
    for match in matches:
        print(
            f"address=0x{match['address']:X} "
            f"region=0x{match['region_base']:X}+0x{match['region_size']:X}"
        )
        for row in match["nearby"]:
            print(
                f"  {row['relative_offset']:+5d} "
                f"i32={row['int32']:12d} "
                f"f64={row['float64']!r}"
            )
    return 0 if matches else 2


if __name__ == "__main__":
    sys.exit(main())
