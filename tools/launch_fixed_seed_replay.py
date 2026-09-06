"""Launch the byte-verified retail runtime with a recorded C RNG seed.

PopCap's framework restores its Mersenne-Twister seed from a DMO, but the
retail initialization path separately executes ``srand(GetTickCount())``.
That second seed is not stored in the DMO and makes independent gameplay
replays diverge.

This module launches the extracted, byte-identical retail payload suspended,
attaches a debugger before its entry point runs, and places a one-shot
breakpoint immediately after the startup ``GetTickCount`` call.  At that
breakpoint only EAX is replaced with the declared seed; the original
instruction byte is restored before the process continues.  The debugger is
then detached.  No executable file or persistent game asset is modified.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass
import os
from pathlib import Path
import struct
import subprocess
import time
from typing import Any, Callable, Mapping

from tools.trace_popcap_shutdown import (
    CONTEXT_FULL,
    CREATE_PROCESS_DEBUG_EVENT,
    DBG_CONTINUE,
    DBG_EXCEPTION_NOT_HANDLED,
    DEBUG_EVENT,
    EXCEPTION_BREAKPOINT,
    EXCEPTION_DEBUG_EVENT,
    EXIT_PROCESS_DEBUG_EVENT,
    LOAD_DLL_DEBUG_EVENT,
    MICROSOFT_CPP_EXCEPTION,
    THREAD_ACCESS,
    WOW64_CONTEXT,
    read_memory,
    write_memory,
)

if os.name == "nt":
    from tools.trace_popcap_shutdown import kernel32
else:
    kernel32 = None


CREATE_SUSPENDED = 0x00000004
CREATE_UNICODE_ENVIRONMENT = 0x00000400
CREATE_THREAD_DEBUG_EVENT = 2
EXIT_THREAD_DEBUG_EVENT = 4
STATUS_WX86_BREAKPOINT = 0x4000001F
EXCEPTION_SINGLE_STEP = 0x80000004
ERROR_SEM_TIMEOUT = 121
ERROR_ACCESS_DENIED = 5
INVALID_SUSPEND_COUNT = 0xFFFFFFFF
STILL_ACTIVE = 259
MEM_COMMIT = 0x00001000
MEM_RESERVE = 0x00002000
MEM_RELEASE = 0x00008000
PAGE_READWRITE = 0x04
PAGE_EXECUTE_READWRITE = 0x40

# SHA-256 2181ce... runtime, fixed image base 0x00400000, no relocations.
DEFAULT_C_RAND_SEED_BREAKPOINT = 0x0068F577
DEFAULT_C_RAND_SEED_ORIGINAL = b"\x50"  # push eax
DEFAULT_C_RAND_SEED_RETURN_ADDRESS = DEFAULT_C_RAND_SEED_BREAKPOINT
DEFAULT_GET_TICK_COUNT_IAT = 0x0094B1D4
DEFAULT_RUNTIME_ENTRY_POINT = 0x008D1883
DEFAULT_RUNTIME_ENTRY_ORIGINAL = b"\xE8\x2E"
G_SEXY_APP_BASE_ADDRESS = 0x009FC740
SEXY_APP_UPDATE_OFFSET = 0x4C4
SEXY_APP_DEMO_READ_BIT_POSITION_OFFSET = 0x608
SEXY_APP_LAST_DEMO_UPDATE_OFFSET = 0x61C
SEXY_APP_NEEDS_COMMAND_OFFSET = 0x620
SEXY_APP_COMMAND_NUMBER_OFFSET = 0x624
SEXY_APP_COMMAND_ORDER_OFFSET = 0x628
SEXY_APP_COMMAND_BIT_POSITION_OFFSET = 0x62C
SEXY_APP_DEMO_LOADING_COMPLETE_OFFSET = 0x630


class STARTUPINFOW(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("lpReserved", wintypes.LPWSTR),
        ("lpDesktop", wintypes.LPWSTR),
        ("lpTitle", wintypes.LPWSTR),
        ("dwX", wintypes.DWORD),
        ("dwY", wintypes.DWORD),
        ("dwXSize", wintypes.DWORD),
        ("dwYSize", wintypes.DWORD),
        ("dwXCountChars", wintypes.DWORD),
        ("dwYCountChars", wintypes.DWORD),
        ("dwFillAttribute", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("wShowWindow", wintypes.WORD),
        ("cbReserved2", wintypes.WORD),
        ("lpReserved2", ctypes.POINTER(ctypes.c_ubyte)),
        ("hStdInput", wintypes.HANDLE),
        ("hStdOutput", wintypes.HANDLE),
        ("hStdError", wintypes.HANDLE),
    ]


class PROCESS_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("hProcess", wintypes.HANDLE),
        ("hThread", wintypes.HANDLE),
        ("dwProcessId", wintypes.DWORD),
        ("dwThreadId", wintypes.DWORD),
    ]


@dataclass(frozen=True, slots=True)
class FixedSeedLaunchEvidence:
    process_id: int
    thread_id: int
    seed: int
    breakpoint_address: int
    original_instruction_hex: str
    breakpoint_observed_perf_counter_ns: int
    debugger_detached_perf_counter_ns: int
    detach_pending_events_drained: int
    detach_attempts: int
    board_seed_observation: Mapping[str, Any] | None = None
    main_thread_suspended_on_detach: bool = False
    main_thread_suspend_previous_count: int | None = None
    process_affinity: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "process_id": self.process_id,
            "thread_id": self.thread_id,
            "seed": self.seed,
            "breakpoint_address_hex": (
                f"0x{self.breakpoint_address:08x}"
            ),
            "original_instruction_hex": self.original_instruction_hex,
            "breakpoint_observed_perf_counter_ns": (
                self.breakpoint_observed_perf_counter_ns
            ),
            "debugger_detached_perf_counter_ns": (
                self.debugger_detached_perf_counter_ns
            ),
            "detach_pending_events_drained": (
                self.detach_pending_events_drained
            ),
            "detach_attempts": self.detach_attempts,
            "main_thread_suspended_on_detach": (
                self.main_thread_suspended_on_detach
            ),
            "main_thread_suspend_previous_count": (
                self.main_thread_suspend_previous_count
            ),
            "register_override": "eax_before_push_to_srand",
            "compatibility_layer": "HIGHDPIAWARE",
            "persistent_file_modified": False,
            "process_affinity": (
                None
                if self.process_affinity is None
                else dict(self.process_affinity)
            ),
        }


@dataclass(frozen=True, slots=True)
class NaturalSeedLaunchEvidence:
    process_id: int
    thread_id: int
    observed_seed: int
    breakpoint_address: int
    original_instruction_hex: str
    breakpoint_observed_perf_counter_ns: int
    debugger_detached_perf_counter_ns: int
    detach_pending_events_drained: int
    detach_attempts: int
    main_thread_suspended_on_detach: bool = False
    main_thread_suspend_previous_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": "retail_natural_seed_observation",
            "process_id": self.process_id,
            "thread_id": self.thread_id,
            "observed_seed": self.observed_seed,
            "effective_seed": self.observed_seed,
            "breakpoint_address_hex": (
                f"0x{self.breakpoint_address:08x}"
            ),
            "original_instruction_hex": self.original_instruction_hex,
            "breakpoint_observed_perf_counter_ns": (
                self.breakpoint_observed_perf_counter_ns
            ),
            "debugger_detached_perf_counter_ns": (
                self.debugger_detached_perf_counter_ns
            ),
            "detach_pending_events_drained": (
                self.detach_pending_events_drained
            ),
            "detach_attempts": self.detach_attempts,
            "main_thread_suspended_on_detach": (
                self.main_thread_suspended_on_detach
            ),
            "main_thread_suspend_previous_count": (
                self.main_thread_suspend_previous_count
            ),
            "seed_source": "retail_eax_before_push_to_srand",
            "register_override": None,
            "rng_process_memory_writes": 0,
            "compatibility_layer": "HIGHDPIAWARE",
            "persistent_file_modified": False,
        }


@dataclass(frozen=True, slots=True)
class FixedSeedIatLaunchEvidence:
    process_id: int
    thread_id: int
    seed: int
    seed_return_address: int
    iat_address: int
    entry_point_address: int
    entry_original_hex: str
    original_pointer: int
    stub_address: int
    stub_machine_code_hex: str
    loader_gate_perf_counter_ns: int
    iat_patched_perf_counter_ns: int
    restoration_guard_perf_counter_ns: int
    iat_restored_perf_counter_ns: int
    restored_at_framework_update: int

    def to_dict(self) -> dict[str, int | str | bool]:
        return {
            "process_id": self.process_id,
            "thread_id": self.thread_id,
            "seed": self.seed,
            "seed_transport": "temporary_get_tick_count_iat_stub",
            "seed_call_return_address_hex": (
                f"0x{self.seed_return_address:08x}"
            ),
            "iat_address_hex": f"0x{self.iat_address:08x}",
            "entry_point_address_hex": (
                f"0x{self.entry_point_address:08x}"
            ),
            "entry_original_hex": self.entry_original_hex,
            "original_pointer_hex": (
                f"0x{self.original_pointer:08x}"
            ),
            "stub_address_hex": f"0x{self.stub_address:08x}",
            "stub_machine_code": (
                "caller_return_address_filter_then_forward"
            ),
            "stub_machine_code_hex": self.stub_machine_code_hex,
            "loader_gate_perf_counter_ns": (
                self.loader_gate_perf_counter_ns
            ),
            "iat_patched_perf_counter_ns": (
                self.iat_patched_perf_counter_ns
            ),
            "restoration_guard_perf_counter_ns": (
                self.restoration_guard_perf_counter_ns
            ),
            "iat_restored_perf_counter_ns": (
                self.iat_restored_perf_counter_ns
            ),
            "restored_at_framework_update": (
                self.restored_at_framework_update
            ),
            "iat_restored": True,
            "compatibility_layer": "HIGHDPIAWARE",
            "persistent_file_modified": False,
        }


def _declare_win32() -> None:
    kernel32.CreateProcessW.argtypes = (
        wintypes.LPCWSTR,
        wintypes.LPWSTR,
        wintypes.LPVOID,
        wintypes.LPVOID,
        wintypes.BOOL,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.LPCWSTR,
        ctypes.POINTER(STARTUPINFOW),
        ctypes.POINTER(PROCESS_INFORMATION),
    )
    kernel32.CreateProcessW.restype = wintypes.BOOL
    kernel32.ResumeThread.argtypes = (wintypes.HANDLE,)
    kernel32.ResumeThread.restype = wintypes.DWORD
    kernel32.SuspendThread.argtypes = (wintypes.HANDLE,)
    kernel32.SuspendThread.restype = wintypes.DWORD
    kernel32.TerminateProcess.argtypes = (wintypes.HANDLE, wintypes.UINT)
    kernel32.TerminateProcess.restype = wintypes.BOOL
    kernel32.GetExitCodeProcess.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.DWORD),
    )
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    kernel32.GetProcessAffinityMask.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
    )
    kernel32.GetProcessAffinityMask.restype = wintypes.BOOL
    kernel32.SetProcessAffinityMask.argtypes = (
        wintypes.HANDLE,
        ctypes.c_size_t,
    )
    kernel32.SetProcessAffinityMask.restype = wintypes.BOOL
    kernel32.VirtualAllocEx.argtypes = (
        wintypes.HANDLE,
        wintypes.LPVOID,
        ctypes.c_size_t,
        wintypes.DWORD,
        wintypes.DWORD,
    )
    kernel32.VirtualAllocEx.restype = wintypes.LPVOID
    kernel32.VirtualFreeEx.argtypes = (
        wintypes.HANDLE,
        wintypes.LPVOID,
        ctypes.c_size_t,
        wintypes.DWORD,
    )
    kernel32.VirtualFreeEx.restype = wintypes.BOOL
    kernel32.VirtualProtectEx.argtypes = (
        wintypes.HANDLE,
        wintypes.LPVOID,
        ctypes.c_size_t,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    )
    kernel32.VirtualProtectEx.restype = wintypes.BOOL


def _environment_block(
    overrides: Mapping[str, str],
) -> ctypes.Array[ctypes.c_wchar]:
    environment = dict(os.environ)
    environment.update(overrides)
    entries = (
        f"{key}={value}"
        for key, value in sorted(
            environment.items(),
            key=lambda item: item[0].casefold(),
        )
    )
    return ctypes.create_unicode_buffer("\0".join(entries) + "\0\0")


def _set_suspended_process_affinity(
    process_handle: wintypes.HANDLE,
    mask: int,
) -> dict[str, int | bool | str]:
    if mask <= 0 or mask > ctypes.c_size_t(-1).value:
        raise ValueError("process affinity mask must fit a positive uintptr")
    previous = ctypes.c_size_t()
    system = ctypes.c_size_t()
    if not kernel32.GetProcessAffinityMask(
        process_handle,
        ctypes.byref(previous),
        ctypes.byref(system),
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    if mask & ~int(system.value):
        raise ValueError(
            f"process affinity mask 0x{mask:X} exceeds system mask "
            f"0x{int(system.value):X}"
        )
    if not kernel32.SetProcessAffinityMask(
        process_handle,
        ctypes.c_size_t(mask),
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    observed = ctypes.c_size_t()
    observed_system = ctypes.c_size_t()
    if not kernel32.GetProcessAffinityMask(
        process_handle,
        ctypes.byref(observed),
        ctypes.byref(observed_system),
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    if int(observed.value) != mask:
        raise RuntimeError("suspended process affinity did not verify")
    return {
        "application_stage": "created_suspended_before_first_resume",
        "previous_process_mask": int(previous.value),
        "requested_process_mask": mask,
        "observed_process_mask": int(observed.value),
        "system_mask": int(observed_system.value),
        "single_logical_processor": mask.bit_count() == 1,
        "persistent_host_modification": False,
    }


def _close_debug_event_handles(event: DEBUG_EVENT) -> None:
    if event.dwDebugEventCode == CREATE_PROCESS_DEBUG_EVENT:
        info = event.u.CreateProcessInfo
        for handle in (info.hFile, info.hThread, info.hProcess):
            if handle:
                kernel32.CloseHandle(handle)
    elif event.dwDebugEventCode == CREATE_THREAD_DEBUG_EVENT:
        handle = event.u.CreateThread.hThread
        if handle:
            kernel32.CloseHandle(handle)
    elif event.dwDebugEventCode == LOAD_DLL_DEBUG_EVENT:
        handle = event.u.LoadDll.hFile
        if handle:
            kernel32.CloseHandle(handle)


def _debug_event_continue_status(event: DEBUG_EVENT) -> int:
    if event.dwDebugEventCode != EXCEPTION_DEBUG_EVENT:
        return DBG_CONTINUE
    code = int(event.u.Exception.ExceptionRecord.ExceptionCode)
    if code in (
        EXCEPTION_BREAKPOINT,
        STATUS_WX86_BREAKPOINT,
        EXCEPTION_SINGLE_STEP,
    ):
        return DBG_CONTINUE
    return DBG_EXCEPTION_NOT_HANDLED


def _process_exit_code(process_handle: wintypes.HANDLE) -> int:
    exit_code = wintypes.DWORD()
    if not kernel32.GetExitCodeProcess(
        process_handle,
        ctypes.byref(exit_code),
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    return int(exit_code.value)


def _startup_replay_state(
    process_handle: wintypes.HANDLE,
) -> dict[str, int | bool | str]:
    """Best-effort snapshot of the framework DMO reader during startup."""

    try:
        base = struct.unpack(
            "<I",
            read_memory(process_handle, G_SEXY_APP_BASE_ADDRESS, 4),
        )[0]
        if base == 0:
            return {
                "available": False,
                "reason": "gSexyAppBase_null",
            }

        def read_i32(offset: int) -> int:
            return struct.unpack(
                "<i",
                read_memory(process_handle, base + offset, 4),
            )[0]

        return {
            "available": True,
            "base_address_hex": f"0x{base:08x}",
            "framework_update": read_i32(SEXY_APP_UPDATE_OFFSET),
            "buffer_read_bit_position": read_i32(
                SEXY_APP_DEMO_READ_BIT_POSITION_OFFSET
            ),
            "last_demo_update": read_i32(
                SEXY_APP_LAST_DEMO_UPDATE_OFFSET
            ),
            "needs_command": bool(
                read_memory(
                    process_handle,
                    base + SEXY_APP_NEEDS_COMMAND_OFFSET,
                    1,
                )[0]
            ),
            "command_number": read_i32(SEXY_APP_COMMAND_NUMBER_OFFSET),
            "command_order": read_i32(SEXY_APP_COMMAND_ORDER_OFFSET),
            "command_bit_position": read_i32(
                SEXY_APP_COMMAND_BIT_POSITION_OFFSET
            ),
            "demo_loading_complete": bool(
                read_memory(
                    process_handle,
                    base + SEXY_APP_DEMO_LOADING_COMPLETE_OFFSET,
                    1,
                )[0]
            ),
        }
    except (OSError, RuntimeError, struct.error) as error:
        return {
            "available": False,
            "reason": (
                f"{type(error).__name__}: {error}"
            ),
        }


def _drain_pending_debug_events(process_id: int) -> tuple[int, int | None]:
    """Continue every event already queued before a detach attempt."""

    drained = 0
    exit_code: int | None = None
    while drained < 512:
        event = DEBUG_EVENT()
        if not kernel32.WaitForDebugEvent(ctypes.byref(event), 0):
            error = ctypes.get_last_error()
            if error == ERROR_SEM_TIMEOUT:
                return drained, exit_code
            raise ctypes.WinError(error)
        if int(event.dwProcessId) != process_id:
            raise RuntimeError("unexpected process in startup debug queue")
        status = _debug_event_continue_status(event)
        if event.dwDebugEventCode == EXIT_PROCESS_DEBUG_EVENT:
            exit_code = int(event.u.ExitProcess.dwExitCode)
        _close_debug_event_handles(event)
        if not kernel32.ContinueDebugEvent(
            event.dwProcessId,
            event.dwThreadId,
            status,
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        drained += 1
        if exit_code is not None:
            return drained, exit_code
    raise RuntimeError("startup debug event queue did not quiesce")


def _detach_debugger(
    *,
    process_id: int,
    process_handle: wintypes.HANDLE,
) -> tuple[int, int]:
    """Drain attachment events and detach without accepting a live failure."""

    total_drained = 0
    last_error = ERROR_ACCESS_DENIED
    for attempt in range(1, 9):
        drained, queued_exit_code = _drain_pending_debug_events(process_id)
        total_drained += drained
        if queued_exit_code is not None:
            raise RuntimeError(
                "runtime exited before debugger detach "
                f"(exit_code={queued_exit_code})"
            )
        if kernel32.DebugActiveProcessStop(process_id):
            return total_drained, attempt
        last_error = ctypes.get_last_error()
        exit_code = _process_exit_code(process_handle)
        if exit_code != STILL_ACTIVE:
            raise RuntimeError(
                "runtime exited before debugger detach "
                f"(exit_code={exit_code})"
            )
        if last_error != ERROR_ACCESS_DENIED:
            raise ctypes.WinError(last_error)
        time.sleep(0.005)
    raise ctypes.WinError(last_error)


def _write_pointer_with_temporary_protection(
    process_handle: wintypes.HANDLE,
    address: int,
    payload: bytes,
) -> None:
    old_protection = wintypes.DWORD()
    if not kernel32.VirtualProtectEx(
        process_handle,
        ctypes.c_void_p(address),
        len(payload),
        PAGE_READWRITE,
        ctypes.byref(old_protection),
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    write_error: BaseException | None = None
    try:
        write_memory(process_handle, address, payload)
    except BaseException as error:
        write_error = error
    restored_protection = wintypes.DWORD()
    if not kernel32.VirtualProtectEx(
        process_handle,
        ctypes.c_void_p(address),
        len(payload),
        old_protection.value,
        ctypes.byref(restored_protection),
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    if write_error is not None:
        raise write_error


def _fixed_seed_iat_stub_code(
    *,
    seed: int,
    original_pointer: int,
    seed_return_address: int,
) -> bytes:
    """Return a caller-filtered GetTickCount forwarding stub."""

    values = (seed, original_pointer, seed_return_address)
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= 0xFFFFFFFF
        for value in values
    ):
        raise ValueError("fixed-seed IAT stub values must fit uint32")
    if original_pointer == 0 or seed_return_address == 0:
        raise ValueError(
            "fixed-seed IAT forwarding and caller addresses must be nonzero"
        )

    # cmp dword ptr [esp], <seed caller return address>
    # jne forward_original
    # mov eax, <seed>
    # ret
    # forward_original:
    # mov eax, <original GetTickCount pointer>
    # jmp eax
    return (
        b"\x81\x3C\x24"
        + struct.pack("<I", seed_return_address)
        + b"\x75\x06"
        + b"\xB8"
        + struct.pack("<I", seed)
        + b"\xC3\xB8"
        + struct.pack("<I", original_pointer)
        + b"\xFF\xE0"
    )


def _fixed_seed_launch_command(
    *,
    runtime_executable: Path,
    changedir: Path,
    dmo: Path,
    launch_mode: str,
) -> str:
    """Build the exact direct-runtime command for play or record mode."""

    if launch_mode not in {"play", "record"}:
        raise ValueError("fixed-seed launch mode must be play or record")
    return subprocess.list2cmdline(
        [
            str(runtime_executable.resolve()),
            f"-{launch_mode}",
            f"-demofile={dmo.resolve()}",
            f"-changedir={changedir.resolve()}\\",
        ]
    )


def launch_fixed_seed_replay_iat_stub(
    *,
    runtime_executable: Path,
    changedir: Path,
    dmo: Path,
    app_id: int,
    seed: int,
    timeout: float = 15.0,
    iat_address: int = DEFAULT_GET_TICK_COUNT_IAT,
    seed_return_address: int = DEFAULT_C_RAND_SEED_RETURN_ADDRESS,
    entry_point_address: int = DEFAULT_RUNTIME_ENTRY_POINT,
    entry_expected_original: bytes = DEFAULT_RUNTIME_ENTRY_ORIGINAL,
    launch_mode: str = "play",
) -> FixedSeedIatLaunchEvidence:
    """Seed play/record startup using a restored temporary IAT stub."""

    if os.name != "nt":
        raise RuntimeError("fixed-seed replay launch requires Windows")
    if (
        not runtime_executable.is_file()
        or not changedir.is_dir()
        or (
            launch_mode == "play"
            and not dmo.is_file()
        )
        or (
            launch_mode == "record"
            and (
                dmo.exists()
                or not dmo.parent.is_dir()
            )
        )
    ):
        raise FileNotFoundError("runtime, changedir, or DMO is unavailable")
    if (
        launch_mode not in {"play", "record"}
        or
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed <= 0xFFFFFFFF
        or app_id <= 0
        or timeout <= 0
        or iat_address <= 0
        or seed_return_address <= 0
        or entry_point_address <= 0
        or len(entry_expected_original) != 2
    ):
        raise ValueError("invalid fixed-seed IAT launch parameter")

    _declare_win32()
    startup = STARTUPINFOW()
    startup.cb = ctypes.sizeof(startup)
    process = PROCESS_INFORMATION()
    command = _fixed_seed_launch_command(
        runtime_executable=runtime_executable,
        changedir=changedir,
        dmo=dmo,
        launch_mode=launch_mode,
    )
    command_buffer = ctypes.create_unicode_buffer(command)
    environment = _environment_block(
        {
            "__COMPAT_LAYER": "HIGHDPIAWARE",
            "SteamAppId": str(app_id),
            "SteamGameId": str(app_id),
        }
    )
    if not kernel32.CreateProcessW(
        str(runtime_executable.resolve()),
        command_buffer,
        None,
        None,
        False,
        CREATE_SUSPENDED | CREATE_UNICODE_ENVIRONMENT,
        environment,
        str(changedir.resolve()),
        ctypes.byref(startup),
        ctypes.byref(process),
    ):
        raise ctypes.WinError(ctypes.get_last_error())

    pid = int(process.dwProcessId)
    thread_id = int(process.dwThreadId)
    stub_address = 0
    original_pointer = 0
    entry_original = b""
    entry_patched = False
    iat_patched = False
    thread_suspended = True
    iat_patched_ns = 0
    loader_gate_ns = 0
    restoration_guard_ns = 0
    restored_ns = 0
    restored_update = -1
    stub_code = b""
    try:
        stub = kernel32.VirtualAllocEx(
            process.hProcess,
            None,
            32,
            MEM_COMMIT | MEM_RESERVE,
            PAGE_EXECUTE_READWRITE,
        )
        stub_address = (
            int(stub)
            if isinstance(stub, int)
            else int(getattr(stub, "value", 0) or 0)
        )
        if not stub_address or stub_address > 0xFFFFFFFF:
            raise RuntimeError(
                "fixed-seed return stub was not allocated below 4 GiB"
            )
        entry_original = read_memory(
            process.hProcess,
            entry_point_address,
            len(entry_expected_original),
        )
        if entry_original != entry_expected_original:
            raise RuntimeError("runtime entry-point instruction mismatch")
        unresolved_pointer = struct.unpack(
            "<I",
            read_memory(process.hProcess, iat_address, 4),
        )[0]
        _write_pointer_with_temporary_protection(
            process.hProcess,
            entry_point_address,
            b"\xEB\xFE",
        )
        kernel32.FlushInstructionCache(
            process.hProcess,
            ctypes.c_void_p(entry_point_address),
            2,
        )
        entry_patched = True
        previous_suspend_count = int(
            kernel32.ResumeThread(process.hThread)
        )
        if previous_suspend_count == INVALID_SUSPEND_COUNT:
            raise ctypes.WinError(ctypes.get_last_error())
        if previous_suspend_count < 1:
            raise RuntimeError("created thread was not suspended")
        thread_suspended = False

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            exit_code = _process_exit_code(process.hProcess)
            if exit_code != STILL_ACTIVE:
                raise RuntimeError(
                    "runtime exited before the loader IAT gate "
                    f"(exit_code={exit_code})"
                )
            candidate_pointer = struct.unpack(
                "<I",
                read_memory(process.hProcess, iat_address, 4),
            )[0]
            if (
                candidate_pointer
                and candidate_pointer != unresolved_pointer
                and candidate_pointer != stub_address
            ):
                previous_suspend_count = int(
                    kernel32.SuspendThread(process.hThread)
                )
                if previous_suspend_count == INVALID_SUSPEND_COUNT:
                    raise ctypes.WinError(ctypes.get_last_error())
                thread_suspended = True
                context = WOW64_CONTEXT()
                context.ContextFlags = CONTEXT_FULL
                if not kernel32.Wow64GetThreadContext(
                    process.hThread,
                    ctypes.byref(context),
                ):
                    raise ctypes.WinError(ctypes.get_last_error())
                if int(context.Eip) == entry_point_address:
                    original_pointer = candidate_pointer
                    loader_gate_ns = time.perf_counter_ns()
                    break
                if (
                    int(kernel32.ResumeThread(process.hThread))
                    == INVALID_SUSPEND_COUNT
                ):
                    raise ctypes.WinError(ctypes.get_last_error())
                thread_suspended = False
            time.sleep(0.0005)
        else:
            raise TimeoutError(
                "runtime did not reach the loader-resolved entry gate"
            )

        stub_code = _fixed_seed_iat_stub_code(
            seed=seed,
            original_pointer=original_pointer,
            seed_return_address=seed_return_address,
        )
        write_memory(process.hProcess, stub_address, stub_code)
        kernel32.FlushInstructionCache(
            process.hProcess,
            ctypes.c_void_p(stub_address),
            len(stub_code),
        )
        _write_pointer_with_temporary_protection(
            process.hProcess,
            iat_address,
            struct.pack("<I", stub_address),
        )
        if (
            struct.unpack(
                "<I",
                read_memory(process.hProcess, iat_address, 4),
            )[0]
            != stub_address
        ):
            raise RuntimeError("GetTickCount IAT patch did not verify")
        iat_patched = True
        _write_pointer_with_temporary_protection(
            process.hProcess,
            entry_point_address,
            entry_original,
        )
        kernel32.FlushInstructionCache(
            process.hProcess,
            ctypes.c_void_p(entry_point_address),
            len(entry_original),
        )
        if (
            read_memory(
                process.hProcess,
                entry_point_address,
                len(entry_original),
            )
            != entry_original
        ):
            raise RuntimeError("runtime entry-point restore did not verify")
        entry_patched = False
        iat_patched_ns = time.perf_counter_ns()
        if int(kernel32.ResumeThread(process.hThread)) == INVALID_SUSPEND_COUNT:
            raise ctypes.WinError(ctypes.get_last_error())
        thread_suspended = False

        deadline = time.monotonic() + timeout
        base = 0
        while time.monotonic() < deadline:
            exit_code = _process_exit_code(process.hProcess)
            if exit_code != STILL_ACTIVE:
                raise RuntimeError(
                    "runtime exited before the fixed-seed IAT restore "
                    f"(exit_code={exit_code})"
                )
            try:
                if not base:
                    base = struct.unpack(
                        "<I",
                        read_memory(
                            process.hProcess,
                            G_SEXY_APP_BASE_ADDRESS,
                            4,
                        ),
                    )[0]
                if base:
                    restored_update = struct.unpack(
                        "<i",
                        read_memory(
                            process.hProcess,
                            base + SEXY_APP_UPDATE_OFFSET,
                            4,
                        ),
                    )[0]
                    if restored_update >= 1:
                        restoration_guard_ns = time.perf_counter_ns()
                        break
            except OSError:
                base = 0
            time.sleep(0.0005)
        else:
            raise TimeoutError(
                "runtime did not reach the fixed-seed IAT restore guard"
            )

        previous_suspend_count = int(
            kernel32.SuspendThread(process.hThread)
        )
        if previous_suspend_count == INVALID_SUSPEND_COUNT:
            raise ctypes.WinError(ctypes.get_last_error())
        thread_suspended = True
        if (
            struct.unpack(
                "<I",
                read_memory(process.hProcess, iat_address, 4),
            )[0]
            != stub_address
        ):
            raise RuntimeError(
                "GetTickCount IAT changed before restoration"
            )
        _write_pointer_with_temporary_protection(
            process.hProcess,
            iat_address,
            struct.pack("<I", original_pointer),
        )
        if (
            struct.unpack(
                "<I",
                read_memory(process.hProcess, iat_address, 4),
            )[0]
            != original_pointer
        ):
            raise RuntimeError("GetTickCount IAT restore did not verify")
        iat_patched = False
        restored_ns = time.perf_counter_ns()
        if int(kernel32.ResumeThread(process.hThread)) == INVALID_SUSPEND_COUNT:
            raise ctypes.WinError(ctypes.get_last_error())
        thread_suspended = False
        if not kernel32.VirtualFreeEx(
            process.hProcess,
            ctypes.c_void_p(stub_address),
            0,
            MEM_RELEASE,
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        stub_address_evidence = stub_address
        stub_address = 0
        return FixedSeedIatLaunchEvidence(
            process_id=pid,
            thread_id=thread_id,
            seed=seed,
            seed_return_address=seed_return_address,
            iat_address=iat_address,
            entry_point_address=entry_point_address,
            entry_original_hex=entry_original.hex(),
            original_pointer=original_pointer,
            stub_address=stub_address_evidence,
            stub_machine_code_hex=stub_code.hex(),
            loader_gate_perf_counter_ns=loader_gate_ns,
            iat_patched_perf_counter_ns=iat_patched_ns,
            restoration_guard_perf_counter_ns=restoration_guard_ns,
            iat_restored_perf_counter_ns=restored_ns,
            restored_at_framework_update=restored_update,
        )
    except BaseException:
        if entry_patched and entry_original:
            try:
                if not thread_suspended:
                    suspended = int(
                        kernel32.SuspendThread(process.hThread)
                    )
                    thread_suspended = suspended != INVALID_SUSPEND_COUNT
                _write_pointer_with_temporary_protection(
                    process.hProcess,
                    entry_point_address,
                    entry_original,
                )
                kernel32.FlushInstructionCache(
                    process.hProcess,
                    ctypes.c_void_p(entry_point_address),
                    len(entry_original),
                )
            except OSError:
                pass
        if iat_patched and original_pointer:
            try:
                if not thread_suspended:
                    suspended = int(
                        kernel32.SuspendThread(process.hThread)
                    )
                    thread_suspended = suspended != INVALID_SUSPEND_COUNT
                _write_pointer_with_temporary_protection(
                    process.hProcess,
                    iat_address,
                    struct.pack("<I", original_pointer),
                )
            except OSError:
                pass
        kernel32.TerminateProcess(process.hProcess, 1)
        raise
    finally:
        if stub_address:
            kernel32.VirtualFreeEx(
                process.hProcess,
                ctypes.c_void_p(stub_address),
                0,
                MEM_RELEASE,
            )
        if process.hThread:
            kernel32.CloseHandle(process.hThread)
        if process.hProcess:
            kernel32.CloseHandle(process.hProcess)


def _launch_seed_observed_replay(
    *,
    runtime_executable: Path,
    changedir: Path,
    dmo: Path,
    app_id: int,
    seed: int | None,
    timeout: float = 15.0,
    breakpoint_address: int = DEFAULT_C_RAND_SEED_BREAKPOINT,
    expected_original: bytes = DEFAULT_C_RAND_SEED_ORIGINAL,
    board_seed_address: int | None = None,
    board_seed_expected_original: bytes = b"\xE8",
    board_seed_callback: Callable[
        [int, int, WOW64_CONTEXT, int, int],
        Mapping[str, Any],
    ]
    | None = None,
    suspend_main_thread_on_detach: bool = False,
    process_affinity_mask: int | None = None,
) -> FixedSeedLaunchEvidence | NaturalSeedLaunchEvidence:
    """Create a suspended replay and observe or replace its C RNG seed."""

    if os.name != "nt":
        raise RuntimeError("fixed-seed replay launch requires Windows")
    if (
        not runtime_executable.is_file()
        or not changedir.is_dir()
        or not dmo.is_file()
    ):
        raise FileNotFoundError("runtime, changedir, or DMO is unavailable")
    if (
        seed is not None
        and (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or not 0 <= seed <= 0xFFFFFFFF
        )
        or app_id <= 0
        or timeout <= 0
        or breakpoint_address <= 0
        or len(expected_original) != 1
        or not isinstance(suspend_main_thread_on_detach, bool)
        or (
            process_affinity_mask is not None
            and (
                isinstance(process_affinity_mask, bool)
                or process_affinity_mask <= 0
                or process_affinity_mask > ctypes.c_size_t(-1).value
            )
        )
        or (board_seed_address is None)
        != (board_seed_callback is None)
        or (
            board_seed_address is not None
            and (
                board_seed_address <= 0
                or board_seed_address == breakpoint_address
                or len(board_seed_expected_original) != 1
            )
        )
    ):
        raise ValueError("invalid fixed-seed launch parameter")

    _declare_win32()
    startup = STARTUPINFOW()
    startup.cb = ctypes.sizeof(startup)
    process = PROCESS_INFORMATION()
    command = subprocess.list2cmdline(
        [
            str(runtime_executable.resolve()),
            "-play",
            f"-demofile={dmo.resolve()}",
            f"-changedir={changedir.resolve()}\\",
        ]
    )
    command_buffer = ctypes.create_unicode_buffer(command)
    environment = _environment_block(
        {
            # The Steam-extracted path has the same per-user compatibility
            # entry. Use the process-local equivalent so the byte-identical
            # direct payload exposes the same unscaled 800x600 client.
            "__COMPAT_LAYER": "HIGHDPIAWARE",
            "SteamAppId": str(app_id),
            "SteamGameId": str(app_id),
        }
    )
    if not kernel32.CreateProcessW(
        str(runtime_executable.resolve()),
        command_buffer,
        None,
        None,
        False,
        CREATE_SUSPENDED | CREATE_UNICODE_ENVIRONMENT,
        environment,
        str(changedir.resolve()),
        ctypes.byref(startup),
        ctypes.byref(process),
    ):
        raise ctypes.WinError(ctypes.get_last_error())

    pid = int(process.dwProcessId)
    thread_id = int(process.dwThreadId)
    attached = False
    seeded = False
    restored = False
    board_seeded = board_seed_address is None
    board_restored = board_seed_address is None
    board_breakpoint_armed = False
    board_observation: Mapping[str, Any] | None = None
    board_original: bytes | None = None
    observed_ns = 0
    observed_seed: int | None = None
    detached_ns = 0
    main_thread_suspended_on_detach = False
    main_thread_suspend_previous_count: int | None = None
    process_affinity: Mapping[str, Any] | None = None
    try:
        if process_affinity_mask is not None:
            process_affinity = _set_suspended_process_affinity(
                process.hProcess,
                process_affinity_mask,
            )
            print(
                "startup_process_affinity_applied "
                f"pid={pid} mask=0x{process_affinity_mask:X} "
                f"previous=0x{int(process_affinity['previous_process_mask']):X} "
                f"observed=0x{int(process_affinity['observed_process_mask']):X} "
                f"system=0x{int(process_affinity['system_mask']):X} "
                "stage=created_suspended_before_first_resume",
                flush=True,
            )
        if not kernel32.DebugActiveProcess(pid):
            raise ctypes.WinError(ctypes.get_last_error())
        attached = True
        if not kernel32.DebugSetProcessKillOnExit(False):
            raise ctypes.WinError(ctypes.get_last_error())
        original = read_memory(
            process.hProcess,
            breakpoint_address,
            1,
        )
        if original != expected_original:
            raise RuntimeError("fixed-seed breakpoint instruction mismatch")
        if board_seed_address is not None:
            board_original = read_memory(
                process.hProcess,
                board_seed_address,
                1,
            )
            if board_original != board_seed_expected_original:
                raise RuntimeError(
                    "board-seed breakpoint instruction mismatch"
                )
        write_memory(process.hProcess, breakpoint_address, b"\xCC")
        kernel32.FlushInstructionCache(
            process.hProcess,
            ctypes.c_void_p(breakpoint_address),
            1,
        )
        previous_suspend_count = int(
            kernel32.ResumeThread(process.hThread)
        )
        if previous_suspend_count == INVALID_SUSPEND_COUNT:
            raise ctypes.WinError(ctypes.get_last_error())
        if previous_suspend_count < 1:
            raise RuntimeError("created thread was not suspended")

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            event = DEBUG_EVENT()
            if not kernel32.WaitForDebugEvent(ctypes.byref(event), 100):
                error = ctypes.get_last_error()
                if error == ERROR_SEM_TIMEOUT:
                    continue
                raise ctypes.WinError(error)
            status = DBG_CONTINUE
            if event.dwDebugEventCode == EXCEPTION_DEBUG_EVENT:
                exception = event.u.Exception
                code = int(exception.ExceptionRecord.ExceptionCode)
                address = int(
                    exception.ExceptionRecord.ExceptionAddress or 0
                )
                if (
                    code in (EXCEPTION_BREAKPOINT, STATUS_WX86_BREAKPOINT)
                    and address == breakpoint_address
                ):
                    thread = kernel32.OpenThread(
                        THREAD_ACCESS,
                        False,
                        event.dwThreadId,
                    )
                    if not thread:
                        raise ctypes.WinError(ctypes.get_last_error())
                    try:
                        context = WOW64_CONTEXT()
                        context.ContextFlags = CONTEXT_FULL
                        if not kernel32.Wow64GetThreadContext(
                            thread,
                            ctypes.byref(context),
                        ):
                            raise ctypes.WinError(ctypes.get_last_error())
                        observed_seed = int(context.Eax)
                        if seed is not None:
                            context.Eax = seed
                        context.Eip = breakpoint_address
                        write_memory(
                            process.hProcess,
                            breakpoint_address,
                            original,
                        )
                        restored = True
                        kernel32.FlushInstructionCache(
                            process.hProcess,
                            ctypes.c_void_p(breakpoint_address),
                            1,
                        )
                        if not kernel32.Wow64SetThreadContext(
                            thread,
                            ctypes.byref(context),
                        ):
                            raise ctypes.WinError(ctypes.get_last_error())
                        if board_seed_address is not None:
                            if board_original is None:
                                raise RuntimeError(
                                    "board-seed original byte is missing"
                                )
                            write_memory(
                                process.hProcess,
                                board_seed_address,
                                b"\xCC",
                            )
                            board_breakpoint_armed = True
                            kernel32.FlushInstructionCache(
                                process.hProcess,
                                ctypes.c_void_p(board_seed_address),
                                1,
                            )
                    finally:
                        kernel32.CloseHandle(thread)
                    observed_ns = time.perf_counter_ns()
                    seeded = True
                    print(
                        "startup_seed_breakpoint "
                        f"pid={pid} tid={event.dwThreadId} "
                        f"observed_seed={observed_seed} "
                        f"override_applied={seed is not None} "
                        f"board_breakpoint_armed="
                        f"{board_breakpoint_armed}",
                        flush=True,
                    )
                elif (
                    board_seed_address is not None
                    and code
                    in (EXCEPTION_BREAKPOINT, STATUS_WX86_BREAKPOINT)
                    and address == board_seed_address
                ):
                    if (
                        not seeded
                        or not board_breakpoint_armed
                        or board_original is None
                        or board_seed_callback is None
                    ):
                        raise RuntimeError(
                            "unexpected board-seed breakpoint state"
                        )
                    thread = kernel32.OpenThread(
                        THREAD_ACCESS,
                        False,
                        event.dwThreadId,
                    )
                    if not thread:
                        raise ctypes.WinError(ctypes.get_last_error())
                    try:
                        context = WOW64_CONTEXT()
                        context.ContextFlags = CONTEXT_FULL
                        if not kernel32.Wow64GetThreadContext(
                            thread,
                            ctypes.byref(context),
                        ):
                            raise ctypes.WinError(ctypes.get_last_error())
                        board_observation = dict(
                            board_seed_callback(
                                int(process.hProcess),
                                int(thread),
                                context,
                                pid,
                                int(event.dwThreadId),
                            )
                        )
                        context.Eip = board_seed_address
                        write_memory(
                            process.hProcess,
                            board_seed_address,
                            board_original,
                        )
                        board_restored = True
                        board_breakpoint_armed = False
                        kernel32.FlushInstructionCache(
                            process.hProcess,
                            ctypes.c_void_p(board_seed_address),
                            1,
                        )
                        if not kernel32.Wow64SetThreadContext(
                            thread,
                            ctypes.byref(context),
                        ):
                            raise ctypes.WinError(ctypes.get_last_error())
                    finally:
                        kernel32.CloseHandle(thread)
                    board_seeded = True
                    print(
                        "fixed_seed_board_breakpoint "
                        f"pid={pid} tid={event.dwThreadId}",
                        flush=True,
                    )
                elif code == MICROSOFT_CPP_EXCEPTION:
                    status = DBG_EXCEPTION_NOT_HANDLED
                elif code not in (
                    EXCEPTION_BREAKPOINT,
                    STATUS_WX86_BREAKPOINT,
                ):
                    status = DBG_EXCEPTION_NOT_HANDLED
            elif event.dwDebugEventCode == EXIT_PROCESS_DEBUG_EVENT:
                exit_code = int(event.u.ExitProcess.dwExitCode)
                replay_state = _startup_replay_state(process.hProcess)
                raise RuntimeError(
                    "runtime exited before required RNG seed breakpoints: "
                    f"exit_code={exit_code}, "
                    f"startup_reached={seeded}, "
                    f"board_reached={board_seeded}, "
                    f"replay_state={replay_state!r}"
                )

            if (
                suspend_main_thread_on_detach
                and seeded
                and board_seeded
                and not main_thread_suspended_on_detach
            ):
                # Increment the explicit suspend count while this debug
                # event still keeps every runtime thread stopped.  The
                # following ContinueDebugEvent therefore cannot create a
                # runnable gap before the successor debugger attaches.
                previous_suspend_count = int(
                    kernel32.SuspendThread(process.hThread)
                )
                if previous_suspend_count == INVALID_SUSPEND_COUNT:
                    raise ctypes.WinError(ctypes.get_last_error())
                if previous_suspend_count != 0:
                    raise RuntimeError(
                        "startup trace handoff expected an unsuspended main "
                        f"thread, observed count={previous_suspend_count}"
                    )
                main_thread_suspended_on_detach = True
                main_thread_suspend_previous_count = previous_suspend_count

            _close_debug_event_handles(event)
            if not kernel32.ContinueDebugEvent(
                event.dwProcessId,
                event.dwThreadId,
                status,
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            if seeded and board_seeded:
                break
        if not seeded or not board_seeded:
            exit_code = _process_exit_code(process.hProcess)
            replay_state = _startup_replay_state(process.hProcess)
            raise TimeoutError(
                "startup or board RNG seed breakpoint was not reached: "
                f"startup_reached={seeded}, "
                f"board_reached={board_seeded}, "
                f"process_exit_code={exit_code}, "
                f"replay_state={replay_state!r}"
            )
        if (
            suspend_main_thread_on_detach
            and not main_thread_suspended_on_detach
        ):
            raise RuntimeError(
                "startup trace handoff did not suspend the main thread "
                "before the final debug event was continued"
            )
        detach_drained, detach_attempts = _detach_debugger(
            process_id=pid,
            process_handle=process.hProcess,
        )
        attached = False
        detached_ns = time.perf_counter_ns()
        if process_affinity is not None:
            handoff_observed = ctypes.c_size_t()
            handoff_system = ctypes.c_size_t()
            if not kernel32.GetProcessAffinityMask(
                process.hProcess,
                ctypes.byref(handoff_observed),
                ctypes.byref(handoff_system),
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            handoff_observed_ns = time.perf_counter_ns()
            requested_mask = int(
                process_affinity["requested_process_mask"]
            )
            observed_mask = int(handoff_observed.value)
            process_affinity = {
                **process_affinity,
                "handoff_observed_process_mask": observed_mask,
                "handoff_system_mask": int(handoff_system.value),
                "handoff_verification_stage": (
                    "after_debugger_detach_before_tracer_handoff"
                ),
                "handoff_verification_perf_counter_ns": (
                    handoff_observed_ns
                ),
                "handoff_verified": observed_mask == requested_mask,
            }
            print(
                "startup_process_affinity_handoff "
                f"pid={pid} requested=0x{requested_mask:X} "
                f"observed=0x{observed_mask:X} "
                f"system=0x{int(handoff_system.value):X} "
                f"verified={observed_mask == requested_mask}",
                flush=True,
            )
            if observed_mask != requested_mask:
                raise RuntimeError(
                    "process affinity changed before tracer handoff: "
                    f"requested=0x{requested_mask:X}, "
                    f"observed=0x{observed_mask:X}"
                )
        if observed_seed is None:
            raise RuntimeError("startup RNG seed was not observed")
        if seed is None:
            if board_seed_address is not None or process_affinity is not None:
                raise RuntimeError("natural seed launch contract drifted")
            return NaturalSeedLaunchEvidence(
                process_id=pid,
                thread_id=thread_id,
                observed_seed=observed_seed,
                breakpoint_address=breakpoint_address,
                original_instruction_hex=original.hex(),
                breakpoint_observed_perf_counter_ns=observed_ns,
                debugger_detached_perf_counter_ns=detached_ns,
                detach_pending_events_drained=detach_drained,
                detach_attempts=detach_attempts,
                main_thread_suspended_on_detach=(
                    main_thread_suspended_on_detach
                ),
                main_thread_suspend_previous_count=(
                    main_thread_suspend_previous_count
                ),
            )
        return FixedSeedLaunchEvidence(
            process_id=pid,
            thread_id=thread_id,
            seed=seed,
            breakpoint_address=breakpoint_address,
            original_instruction_hex=original.hex(),
            breakpoint_observed_perf_counter_ns=observed_ns,
            debugger_detached_perf_counter_ns=detached_ns,
            detach_pending_events_drained=detach_drained,
            detach_attempts=detach_attempts,
            board_seed_observation=board_observation,
            main_thread_suspended_on_detach=(
                main_thread_suspended_on_detach
            ),
            main_thread_suspend_previous_count=(
                main_thread_suspend_previous_count
            ),
            process_affinity=process_affinity,
        )
    except BaseException:
        if attached:
            if not restored:
                try:
                    write_memory(
                        process.hProcess,
                        breakpoint_address,
                        expected_original,
                    )
                    kernel32.FlushInstructionCache(
                        process.hProcess,
                        ctypes.c_void_p(breakpoint_address),
                        1,
                    )
                except OSError:
                    pass
            if (
                board_seed_address is not None
                and board_original is not None
                and board_breakpoint_armed
                and not board_restored
            ):
                try:
                    write_memory(
                        process.hProcess,
                        board_seed_address,
                        board_original,
                    )
                    kernel32.FlushInstructionCache(
                        process.hProcess,
                        ctypes.c_void_p(board_seed_address),
                        1,
                    )
                except OSError:
                    pass
            kernel32.DebugActiveProcessStop(pid)
        kernel32.TerminateProcess(process.hProcess, 1)
        raise
    finally:
        if process.hThread:
            kernel32.CloseHandle(process.hThread)
        if process.hProcess:
            kernel32.CloseHandle(process.hProcess)


def launch_fixed_seed_replay(
    *,
    runtime_executable: Path,
    changedir: Path,
    dmo: Path,
    app_id: int,
    seed: int,
    timeout: float = 15.0,
    breakpoint_address: int = DEFAULT_C_RAND_SEED_BREAKPOINT,
    expected_original: bytes = DEFAULT_C_RAND_SEED_ORIGINAL,
    board_seed_address: int | None = None,
    board_seed_expected_original: bytes = b"\xE8",
    board_seed_callback: Callable[
        [int, int, WOW64_CONTEXT, int, int],
        Mapping[str, Any],
    ]
    | None = None,
    suspend_main_thread_on_detach: bool = False,
    process_affinity_mask: int | None = None,
) -> FixedSeedLaunchEvidence:
    """Create a suspended replay and replace only its startup C RNG seed."""

    result = _launch_seed_observed_replay(
        runtime_executable=runtime_executable,
        changedir=changedir,
        dmo=dmo,
        app_id=app_id,
        seed=seed,
        timeout=timeout,
        breakpoint_address=breakpoint_address,
        expected_original=expected_original,
        board_seed_address=board_seed_address,
        board_seed_expected_original=board_seed_expected_original,
        board_seed_callback=board_seed_callback,
        suspend_main_thread_on_detach=suspend_main_thread_on_detach,
        process_affinity_mask=process_affinity_mask,
    )
    if not isinstance(result, FixedSeedLaunchEvidence):
        raise RuntimeError("fixed seed launch evidence type mismatch")
    return result


def launch_natural_seed_replay(
    *,
    runtime_executable: Path,
    changedir: Path,
    dmo: Path,
    app_id: int,
    timeout: float = 15.0,
    breakpoint_address: int = DEFAULT_C_RAND_SEED_BREAKPOINT,
    expected_original: bytes = DEFAULT_C_RAND_SEED_ORIGINAL,
    suspend_main_thread_on_detach: bool = False,
) -> NaturalSeedLaunchEvidence:
    """Observe retail EAX at ``srand`` without replacing the seed."""

    result = _launch_seed_observed_replay(
        runtime_executable=runtime_executable,
        changedir=changedir,
        dmo=dmo,
        app_id=app_id,
        seed=None,
        timeout=timeout,
        breakpoint_address=breakpoint_address,
        expected_original=expected_original,
        suspend_main_thread_on_detach=suspend_main_thread_on_detach,
    )
    if not isinstance(result, NaturalSeedLaunchEvidence):
        raise RuntimeError("natural seed launch evidence type mismatch")
    return result
