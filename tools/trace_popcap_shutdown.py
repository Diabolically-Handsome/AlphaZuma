"""Trace the caller of the retail PopCap framework ``Shutdown`` method.

This is a narrowly scoped Windows diagnostic for the verified 32-bit Zuma's
Revenge runtime.  It attaches through the Windows debugging API, replaces the
first byte of one caller-supplied function address with ``INT 3``, captures
the WOW64 register/stack state when that breakpoint is reached, restores the
original byte, and lets the process continue.

The executable path is checked before any process memory is changed.  The
breakpoint byte is restored both on a hit and on every normal error/timeout
cleanup path.
"""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import struct
import sys
import time
from typing import Iterable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.control_popcap_replay import main_window_for_pid, post_char
from tools.launch_popcap_replay import process_image_path, same_windows_path


DBG_CONTINUE = 0x00010002
DBG_EXCEPTION_NOT_HANDLED = 0x80010001
EXCEPTION_DEBUG_EVENT = 1
CREATE_PROCESS_DEBUG_EVENT = 3
EXIT_PROCESS_DEBUG_EVENT = 5
LOAD_DLL_DEBUG_EVENT = 6
EXCEPTION_BREAKPOINT = 0x80000003
STATUS_WX86_BREAKPOINT = 0x4000001F
MICROSOFT_CPP_EXCEPTION = 0xE06D7363

PROCESS_ACCESS = (
    0x00100000  # SYNCHRONIZE
    | 0x0400  # PROCESS_QUERY_INFORMATION
    | 0x0010  # PROCESS_VM_READ
    | 0x0020  # PROCESS_VM_WRITE
    | 0x0008  # PROCESS_VM_OPERATION
)
THREAD_ACCESS = (
    0x0008  # THREAD_GET_CONTEXT
    | 0x0010  # THREAD_SET_CONTEXT
    | 0x0040  # THREAD_QUERY_INFORMATION
)
CONTEXT_I386 = 0x00010000
CONTEXT_FULL = CONTEXT_I386 | 0x1 | 0x2 | 0x4
ERROR_SEM_TIMEOUT = 121


class EXCEPTION_RECORD(ctypes.Structure):
    _fields_ = [
        ("ExceptionCode", wintypes.DWORD),
        ("ExceptionFlags", wintypes.DWORD),
        ("ExceptionRecord", ctypes.c_void_p),
        ("ExceptionAddress", ctypes.c_void_p),
        ("NumberParameters", wintypes.DWORD),
        ("ExceptionInformation", ctypes.c_size_t * 15),
    ]


class EXCEPTION_DEBUG_INFO(ctypes.Structure):
    _fields_ = [
        ("ExceptionRecord", EXCEPTION_RECORD),
        ("dwFirstChance", wintypes.DWORD),
    ]


class CREATE_THREAD_DEBUG_INFO(ctypes.Structure):
    _fields_ = [
        ("hThread", wintypes.HANDLE),
        ("lpThreadLocalBase", ctypes.c_void_p),
        ("lpStartAddress", ctypes.c_void_p),
    ]


class CREATE_PROCESS_DEBUG_INFO(ctypes.Structure):
    _fields_ = [
        ("hFile", wintypes.HANDLE),
        ("hProcess", wintypes.HANDLE),
        ("hThread", wintypes.HANDLE),
        ("lpBaseOfImage", ctypes.c_void_p),
        ("dwDebugInfoFileOffset", wintypes.DWORD),
        ("nDebugInfoSize", wintypes.DWORD),
        ("lpThreadLocalBase", ctypes.c_void_p),
        ("lpStartAddress", ctypes.c_void_p),
        ("lpImageName", ctypes.c_void_p),
        ("fUnicode", wintypes.WORD),
    ]


class EXIT_DEBUG_INFO(ctypes.Structure):
    _fields_ = [("dwExitCode", wintypes.DWORD)]


class LOAD_DLL_DEBUG_INFO(ctypes.Structure):
    _fields_ = [
        ("hFile", wintypes.HANDLE),
        ("lpBaseOfDll", ctypes.c_void_p),
        ("dwDebugInfoFileOffset", wintypes.DWORD),
        ("nDebugInfoSize", wintypes.DWORD),
        ("lpImageName", ctypes.c_void_p),
        ("fUnicode", wintypes.WORD),
    ]


class UNLOAD_DLL_DEBUG_INFO(ctypes.Structure):
    _fields_ = [("lpBaseOfDll", ctypes.c_void_p)]


class OUTPUT_DEBUG_STRING_INFO(ctypes.Structure):
    _fields_ = [
        ("lpDebugStringData", ctypes.c_void_p),
        ("fUnicode", wintypes.WORD),
        ("nDebugStringLength", wintypes.WORD),
    ]


class RIP_INFO(ctypes.Structure):
    _fields_ = [
        ("dwError", wintypes.DWORD),
        ("dwType", wintypes.DWORD),
    ]


class DEBUG_EVENT_UNION(ctypes.Union):
    _fields_ = [
        ("Exception", EXCEPTION_DEBUG_INFO),
        ("CreateThread", CREATE_THREAD_DEBUG_INFO),
        ("CreateProcessInfo", CREATE_PROCESS_DEBUG_INFO),
        ("ExitThread", EXIT_DEBUG_INFO),
        ("ExitProcess", EXIT_DEBUG_INFO),
        ("LoadDll", LOAD_DLL_DEBUG_INFO),
        ("UnloadDll", UNLOAD_DLL_DEBUG_INFO),
        ("DebugString", OUTPUT_DEBUG_STRING_INFO),
        ("RipInfo", RIP_INFO),
    ]


class DEBUG_EVENT(ctypes.Structure):
    _fields_ = [
        ("dwDebugEventCode", wintypes.DWORD),
        ("dwProcessId", wintypes.DWORD),
        ("dwThreadId", wintypes.DWORD),
        ("u", DEBUG_EVENT_UNION),
    ]


class FLOATING_SAVE_AREA(ctypes.Structure):
    _fields_ = [
        ("ControlWord", wintypes.DWORD),
        ("StatusWord", wintypes.DWORD),
        ("TagWord", wintypes.DWORD),
        ("ErrorOffset", wintypes.DWORD),
        ("ErrorSelector", wintypes.DWORD),
        ("DataOffset", wintypes.DWORD),
        ("DataSelector", wintypes.DWORD),
        ("RegisterArea", ctypes.c_ubyte * 80),
        ("Cr0NpxState", wintypes.DWORD),
    ]


class WOW64_CONTEXT(ctypes.Structure):
    _fields_ = [
        ("ContextFlags", wintypes.DWORD),
        ("Dr0", wintypes.DWORD),
        ("Dr1", wintypes.DWORD),
        ("Dr2", wintypes.DWORD),
        ("Dr3", wintypes.DWORD),
        ("Dr6", wintypes.DWORD),
        ("Dr7", wintypes.DWORD),
        ("FloatSave", FLOATING_SAVE_AREA),
        ("SegGs", wintypes.DWORD),
        ("SegFs", wintypes.DWORD),
        ("SegEs", wintypes.DWORD),
        ("SegDs", wintypes.DWORD),
        ("Edi", wintypes.DWORD),
        ("Esi", wintypes.DWORD),
        ("Ebx", wintypes.DWORD),
        ("Edx", wintypes.DWORD),
        ("Ecx", wintypes.DWORD),
        ("Eax", wintypes.DWORD),
        ("Ebp", wintypes.DWORD),
        ("Eip", wintypes.DWORD),
        ("SegCs", wintypes.DWORD),
        ("EFlags", wintypes.DWORD),
        ("Esp", wintypes.DWORD),
        ("SegSs", wintypes.DWORD),
        ("ExtendedRegisters", ctypes.c_ubyte * 512),
    ]


if os.name == "nt":
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    def _declare(name: str, arguments: list[object], result: object) -> None:
        function = getattr(kernel32, name)
        function.argtypes = arguments
        function.restype = result

    _declare("DebugActiveProcess", [wintypes.DWORD], wintypes.BOOL)
    _declare("DebugSetProcessKillOnExit", [wintypes.BOOL], wintypes.BOOL)
    _declare("DebugActiveProcessStop", [wintypes.DWORD], wintypes.BOOL)
    _declare(
        "WaitForDebugEvent",
        [ctypes.POINTER(DEBUG_EVENT), wintypes.DWORD],
        wintypes.BOOL,
    )
    _declare(
        "ContinueDebugEvent",
        [wintypes.DWORD, wintypes.DWORD, wintypes.DWORD],
        wintypes.BOOL,
    )
    _declare(
        "OpenProcess",
        [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD],
        wintypes.HANDLE,
    )
    _declare(
        "OpenThread",
        [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD],
        wintypes.HANDLE,
    )
    _declare(
        "ReadProcessMemory",
        [
            wintypes.HANDLE,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t),
        ],
        wintypes.BOOL,
    )
    _declare(
        "WriteProcessMemory",
        [
            wintypes.HANDLE,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t),
        ],
        wintypes.BOOL,
    )
    _declare(
        "FlushInstructionCache",
        [wintypes.HANDLE, ctypes.c_void_p, ctypes.c_size_t],
        wintypes.BOOL,
    )
    _declare(
        "Wow64GetThreadContext",
        [wintypes.HANDLE, ctypes.POINTER(WOW64_CONTEXT)],
        wintypes.BOOL,
    )
    _declare(
        "Wow64SetThreadContext",
        [wintypes.HANDLE, ctypes.POINTER(WOW64_CONTEXT)],
        wintypes.BOOL,
    )
    _declare("CloseHandle", [wintypes.HANDLE], wintypes.BOOL)


def read_memory(handle: int, address: int, size: int) -> bytes:
    buffer = ctypes.create_string_buffer(size)
    transferred = ctypes.c_size_t()
    if not kernel32.ReadProcessMemory(
        handle,
        ctypes.c_void_p(address),
        buffer,
        size,
        ctypes.byref(transferred),
    ) or transferred.value != size:
        raise ctypes.WinError(ctypes.get_last_error())
    return buffer.raw


def write_memory(handle: int, address: int, data: bytes) -> None:
    buffer = ctypes.create_string_buffer(data)
    transferred = ctypes.c_size_t()
    if not kernel32.WriteProcessMemory(
        handle,
        ctypes.c_void_p(address),
        buffer,
        len(data),
        ctypes.byref(transferred),
    ) or transferred.value != len(data):
        raise ctypes.WinError(ctypes.get_last_error())


def demo_state_at_shutdown(process: int, base: int) -> dict[str, int]:
    """Read the known demo fields from the ``SexyAppBase`` being shut down."""

    data = read_memory(process, base + 0x4C4, 0x16D)

    def i32(relative: int) -> int:
        return struct.unpack_from("<i", data, relative)[0]

    return {
        "update": i32(0x000),
        "read_bitpos": i32(0x144),
        "demo_length": i32(0x14C),
        "last_update": i32(0x158),
        "needs": data[0x15C],
        "short": data[0x15D],
        "num": i32(0x160),
        "order": i32(0x164),
        "cmd_bitpos": i32(0x168),
        "loading_complete": data[0x16C],
    }


def frame_chain(
    process: int,
    frame_pointer: int,
    *,
    maximum: int = 16,
) -> tuple[tuple[int, int], ...]:
    frames: list[tuple[int, int]] = []
    current = frame_pointer
    for _ in range(maximum):
        if not current:
            break
        try:
            next_frame, return_address = struct.unpack(
                "<II", read_memory(process, current, 8)
            )
        except OSError:
            break
        frames.append((current, return_address))
        if next_frame <= current or next_frame - current > 0x100000:
            break
        current = next_frame
    return tuple(frames)


def trace_shutdown(
    *,
    pid: int,
    executable: Path,
    address: int,
    timeout: float,
    trigger_char: str | None,
) -> bool:
    if os.name != "nt":
        raise RuntimeError("shutdown tracing requires Windows")
    observed = process_image_path(pid)
    if observed is None or not same_windows_path(observed, executable):
        raise RuntimeError(
            f"PID {pid} path mismatch: observed={observed}, expected={executable}"
        )
    if address <= 0 or timeout <= 0:
        raise ValueError("address and timeout must be positive")

    attached = False
    exited = False
    hit = False
    process = 0
    original: bytes | None = None
    restored = False
    cpp_exceptions = 0
    if not kernel32.DebugActiveProcess(pid):
        raise ctypes.WinError(ctypes.get_last_error())
    attached = True
    try:
        if not kernel32.DebugSetProcessKillOnExit(False):
            raise ctypes.WinError(ctypes.get_last_error())
        process = kernel32.OpenProcess(PROCESS_ACCESS, False, pid)
        if not process:
            raise ctypes.WinError(ctypes.get_last_error())
        original = read_memory(process, address, 1)
        write_memory(process, address, b"\xCC")
        kernel32.FlushInstructionCache(process, ctypes.c_void_p(address), 1)
        if trigger_char is not None:
            post_char(main_window_for_pid(pid), trigger_char)
        print(
            f"armed pid={pid} address=0x{address:X} "
            f"original={original.hex()} trigger={trigger_char!r}",
            flush=True,
        )

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            event = DEBUG_EVENT()
            if not kernel32.WaitForDebugEvent(
                ctypes.byref(event), 250
            ):
                error = ctypes.get_last_error()
                if error == ERROR_SEM_TIMEOUT:
                    continue
                raise ctypes.WinError(error)
            status = DBG_CONTINUE
            if event.dwDebugEventCode == EXCEPTION_DEBUG_EVENT:
                exception = event.u.Exception
                code = int(exception.ExceptionRecord.ExceptionCode)
                exception_address = int(
                    exception.ExceptionRecord.ExceptionAddress or 0
                )
                if code == MICROSOFT_CPP_EXCEPTION:
                    cpp_exceptions += 1
                    if cpp_exceptions == 1 or cpp_exceptions % 25 == 0:
                        print(
                            f"cpp_exceptions={cpp_exceptions} "
                            f"tid={event.dwThreadId} "
                            f"address=0x{exception_address:X}",
                            flush=True,
                        )
                    status = DBG_EXCEPTION_NOT_HANDLED
                elif (
                    code in (EXCEPTION_BREAKPOINT, STATUS_WX86_BREAKPOINT)
                    and exception_address == address
                ):
                    thread = kernel32.OpenThread(
                        THREAD_ACCESS, False, event.dwThreadId
                    )
                    if not thread:
                        raise ctypes.WinError(ctypes.get_last_error())
                    try:
                        context = WOW64_CONTEXT()
                        context.ContextFlags = CONTEXT_FULL
                        if not kernel32.Wow64GetThreadContext(
                            thread, ctypes.byref(context)
                        ):
                            raise ctypes.WinError(ctypes.get_last_error())
                        words = struct.unpack(
                            "<32I",
                            read_memory(process, context.Esp, 128),
                        )
                        print(
                            "shutdown_break "
                            f"tid={event.dwThreadId} "
                            f"eip=0x{context.Eip:08X} "
                            f"esp=0x{context.Esp:08X} "
                            f"ebp=0x{context.Ebp:08X} "
                            f"ecx=0x{context.Ecx:08X} "
                            f"eax=0x{context.Eax:08X} "
                            f"return=0x{words[0]:08X}",
                            flush=True,
                        )
                        demo_state = demo_state_at_shutdown(
                            process, context.Ecx
                        )
                        print(
                            "shutdown_demo_state "
                            + " ".join(
                                f"{key}={value}"
                                for key, value in demo_state.items()
                            ),
                            flush=True,
                        )
                        print(
                            "stack_words="
                            + ",".join(f"0x{value:08X}" for value in words),
                            flush=True,
                        )
                        print(
                            "frame_chain="
                            + ",".join(
                                f"0x{frame:08X}:0x{return_address:08X}"
                                for frame, return_address in frame_chain(
                                    process, context.Ebp
                                )
                            ),
                            flush=True,
                        )
                        write_memory(process, address, original)
                        restored = True
                        kernel32.FlushInstructionCache(
                            process, ctypes.c_void_p(address), 1
                        )
                        context.Eip = address
                        if not kernel32.Wow64SetThreadContext(
                            thread, ctypes.byref(context)
                        ):
                            raise ctypes.WinError(ctypes.get_last_error())
                        hit = True
                    finally:
                        kernel32.CloseHandle(thread)
                elif code not in (
                    EXCEPTION_BREAKPOINT,
                    STATUS_WX86_BREAKPOINT,
                ):
                    status = DBG_EXCEPTION_NOT_HANDLED
            elif event.dwDebugEventCode == CREATE_PROCESS_DEBUG_EVENT:
                file_handle = event.u.CreateProcessInfo.hFile
                if file_handle:
                    kernel32.CloseHandle(file_handle)
            elif event.dwDebugEventCode == LOAD_DLL_DEBUG_EVENT:
                file_handle = event.u.LoadDll.hFile
                if file_handle:
                    kernel32.CloseHandle(file_handle)
            elif event.dwDebugEventCode == EXIT_PROCESS_DEBUG_EVENT:
                print(
                    f"exit_process code={event.u.ExitProcess.dwExitCode}",
                    flush=True,
                )
                exited = True

            if not kernel32.ContinueDebugEvent(
                event.dwProcessId, event.dwThreadId, status
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            if exited:
                break

        print(
            f"result hit={hit} exited={exited} "
            f"cpp_exceptions={cpp_exceptions}",
            flush=True,
        )
        return hit
    finally:
        if process:
            if original is not None and not restored:
                try:
                    write_memory(process, address, original)
                    kernel32.FlushInstructionCache(
                        process, ctypes.c_void_p(address), 1
                    )
                except OSError:
                    pass
            kernel32.CloseHandle(process)
        if attached and not exited:
            kernel32.DebugActiveProcessStop(pid)


def _positive_int(text: str) -> int:
    value = int(text, 0)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Trace one path-verified 32-bit PopCap shutdown function."
    )
    parser.add_argument("--pid", required=True, type=_positive_int)
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--address", required=True, type=_positive_int)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--trigger-char")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.trigger_char is not None and len(args.trigger_char) != 1:
        raise SystemExit("--trigger-char must contain exactly one character")
    hit = trace_shutdown(
        pid=args.pid,
        executable=args.executable.resolve(),
        address=args.address,
        timeout=args.timeout,
        trigger_char=args.trigger_char,
    )
    return 0 if hit else 2


if __name__ == "__main__":
    sys.exit(main())
