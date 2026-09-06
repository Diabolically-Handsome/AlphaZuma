"""Trace callers of the retail global MTRand wrapper in an existing process."""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import struct
import sys
import time
from typing import Any, Iterable

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tools.launch_popcap_replay import process_image_path, same_windows_path
from tools.popcap_global_mtrand_restore import (
    GlobalMTRandRestoreState,
    load_global_mtrand_restore_state,
)
from tools.popcap_thread_crt_restore import (
    ThreadCrtRestoreState,
    load_source_bound_thread_crt_restore_state,
)
from tools.trace_popcap_demo_commands import (
    ERROR_ACCESS_DENIED,
    ERROR_INVALID_PARAMETER,
    ERROR_SEM_TIMEOUT,
    EXCEPTION_SINGLE_STEP,
    G_SEXY_APP_BASE_ADDRESS,
    MTRAND_STATE_BYTES,
    MTRAND_STATE_WORDS,
    STATUS_WX86_BREAKPOINT,
    STATUS_WX86_SINGLE_STEP,
    THREAD_SUSPEND_RESUME,
    TRAP_FLAG,
    _read_i32,
    _read_u32,
    _resume_suspended_threads,
    _stack_code_candidates,
    _suspend_other_threads,
    _thread_crt_rng_state,
)
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
    PROCESS_ACCESS,
    THREAD_ACCESS,
    WOW64_CONTEXT,
    kernel32,
    read_memory,
    write_memory,
)
from zuma_rl.revenge_core import PopCapMTRandom


DEFAULT_GLOBAL_RNG_WRAPPER = 0x00617490
DEFAULT_GLOBAL_RNG_STATE = 0x00A313C0
DEFAULT_BOARD_UPDATE_OFFSET = 0x4C4
TRACE_SCHEMA = "zuma-rl.pc-global-mtrand-call-trace"
TRACE_VERSION = 1


def _can_stop_without_pending_single_step(
    *,
    stop_requested: bool,
    stepping_thread: int | None,
) -> bool:
    """A frozen target needs no future RNG event to restore the breakpoint."""

    return stop_requested and stepping_thread is None


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _sha256_bytes(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _write_canonical(path: Path, value: dict[str, Any]) -> None:
    path.write_bytes(
        (
            json.dumps(
                value,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("ascii")
    )


def _predicted_output(payload: bytes) -> tuple[int, int]:
    if len(payload) != MTRAND_STATE_BYTES:
        raise RuntimeError("global MTRand state size mismatch")
    unpacked = struct.unpack("<625I", payload)
    index = unpacked[MTRAND_STATE_WORDS]
    if index > MTRAND_STATE_WORDS:
        raise RuntimeError(f"global MTRand index invalid: {index}")
    rng = PopCapMTRandom(1)
    rng.load_state(unpacked[:MTRAND_STATE_WORDS], index)
    return rng.next_u31(), rng.index


def _thread_crt_snapshot(
    *,
    process: int,
    thread: int,
    context: WOW64_CONTEXT,
    thread_id: int,
) -> dict[str, Any]:
    """Read one paused WOW64 thread's CRT rand state without mutation."""

    try:
        identity = _thread_crt_rng_state(
            process=process,
            thread=thread,
            context=context,
            expected_thread_id=thread_id,
        )
        return {
            "thread_crt_state": {
                key: int(value) for key, value in identity.items()
            },
            "thread_crt_rand_state": _read_u32(
                process,
                int(identity["rand_state_address"]),
            ),
            "thread_crt_snapshot_error": None,
        }
    except (OSError, RuntimeError, ValueError) as error:
        return {
            "thread_crt_state": None,
            "thread_crt_rand_state": None,
            "thread_crt_snapshot_error": (
                f"{type(error).__name__}: {error}"
            ),
        }


def _restore_source_evidence(
    state: GlobalMTRandRestoreState,
) -> dict[str, Any]:
    return {
        "source_path": str(state.source_path),
        "source_sha256": state.source_sha256,
        "source_process_id": state.source_process_id,
        "source_framework_update": state.source_framework_update,
        "source_native_game_time": state.source_native_game_time,
        **state.semantic_dict(),
        "semantic_sha256": state.semantic_sha256,
    }


def _thread_crt_restore_source_evidence(
    state: ThreadCrtRestoreState,
) -> dict[str, Any]:
    return {
        "source_kind": state.source_kind,
        "source_path": str(state.source_path),
        "source_sha256": state.source_sha256,
        "source_process_id": state.source_process_id,
        "source_thread_id": state.source_thread_id,
        "source_framework_update": state.source_framework_update,
        "source_native_game_time": state.source_native_game_time,
        "source_call_order": state.source_call_order,
        "source_caller": state.source_caller,
        "source_caller_hex": (
            f"0x{state.source_caller:08x}"
            if state.source_caller is not None
            else None
        ),
        "source_recording_report_path": (
            str(state.source_recording_report_path)
            if state.source_recording_report_path is not None
            else None
        ),
        "source_recording_report_sha256": (
            state.source_recording_report_sha256
        ),
        "source_dmo_path": (
            str(state.source_dmo_path)
            if state.source_dmo_path is not None
            else None
        ),
        "source_dmo_sha256": state.source_dmo_sha256,
        **state.semantic_dict(),
        "semantic_sha256": state.semantic_sha256,
    }


def _apply_thread_crt_restore(
    *,
    process: int,
    process_id: int,
    thread: int,
    thread_id: int,
    context: WOW64_CONTEXT,
    framework_update: int,
    caller: int,
    desired: ThreadCrtRestoreState,
) -> dict[str, Any]:
    identity = _thread_crt_rng_state(
        process=process,
        thread=thread,
        context=context,
        expected_thread_id=thread_id,
    )
    state_address = int(identity["rand_state_address"])
    live_before = _read_u32(process, state_address)
    changed = live_before != desired.state
    try:
        if changed:
            write_memory(process, state_address, struct.pack("<I", desired.state))
        observed = _read_u32(process, state_address)
        if observed != desired.state:
            raise RuntimeError("call-site thread CRT writeback did not verify")
    except Exception:
        try:
            write_memory(process, state_address, struct.pack("<I", live_before))
            if _read_u32(process, state_address) != live_before:
                raise RuntimeError("call-site thread CRT rollback did not verify")
        except Exception as rollback_error:
            raise RuntimeError(
                "call-site thread CRT restore and rollback failed"
            ) from rollback_error
        raise
    return {
        "classification": "diagnostic-source-bound-thread-crt-restore",
        "process_id": process_id,
        "thread_id": thread_id,
        "framework_update": framework_update,
        "caller": caller,
        "caller_hex": f"0x{caller:08x}",
        "state_address": state_address,
        "live_before_state": live_before,
        "bytes_written": 4 if changed else 0,
        "changed": changed,
        "writeback_verified": True,
        "transactional_rollback_on_failure": True,
        "persistent_file_modified": False,
        "restored": _thread_crt_restore_source_evidence(desired),
    }


def trace_global_rng_calls(
    *,
    pid: int,
    executable: Path,
    output_path: Path,
    ready_path: Path,
    stop_path: Path,
    start_update: int,
    end_update: int,
    address: int = DEFAULT_GLOBAL_RNG_WRAPPER,
    attach_timeout: float = 30.0,
    timeout: float = 60.0,
    restore_state: GlobalMTRandRestoreState | None = None,
    restore_framework_update: int | None = None,
    restore_caller: int | None = None,
    thread_crt_restore_state: ThreadCrtRestoreState | None = None,
    thread_crt_restore_framework_update: int | None = None,
    thread_crt_restore_caller: int | None = None,
) -> Path:
    """Attach, trace bounded wrapper entries, then detach without killing."""

    if (
        pid <= 0
        or address <= 0
        or start_update < 0
        or end_update < start_update
        or attach_timeout <= 0
        or timeout <= 0
    ):
        raise ValueError("invalid trace target or update range")
    restore_values = (
        restore_state,
        restore_framework_update,
        restore_caller,
    )
    if any(value is not None for value in restore_values) and any(
        value is None for value in restore_values
    ):
        raise ValueError("call-site restore options must be supplied together")
    if (
        restore_framework_update is not None
        and (
            restore_framework_update < start_update
            or restore_framework_update > end_update
            or restore_caller is None
            or restore_caller <= 0
        )
    ):
        raise ValueError("call-site restore target is outside the trace range")
    thread_crt_restore_values = (
        thread_crt_restore_state,
        thread_crt_restore_framework_update,
        thread_crt_restore_caller,
    )
    if any(value is not None for value in thread_crt_restore_values) and any(
        value is None for value in thread_crt_restore_values
    ):
        raise ValueError(
            "thread CRT call-site restore options must be supplied together"
        )
    if (
        thread_crt_restore_framework_update is not None
        and (
            thread_crt_restore_framework_update < start_update
            or thread_crt_restore_framework_update > end_update
            or thread_crt_restore_caller is None
            or thread_crt_restore_caller <= 0
        )
    ):
        raise ValueError(
            "thread CRT call-site restore target is outside the trace range"
        )
    for path in (output_path, ready_path, stop_path):
        if path.exists() or not path.parent.is_dir():
            raise FileExistsError(f"trace path is not new: {path}")
    observed = process_image_path(pid)
    if observed is None or not same_windows_path(observed, executable):
        raise RuntimeError(
            f"PID {pid} path mismatch: observed={observed}, expected={executable}"
        )

    attached = False
    process = 0
    original: bytes | None = None
    breakpoint_armed = False
    stepping_thread: int | None = None
    stepping_suspended_threads: list[tuple[int, int]] = []
    calls: list[dict[str, Any]] = []
    exited = False
    exit_code: int | None = None
    started_utc = datetime.now(timezone.utc).isoformat()
    started_ns = time.perf_counter_ns()
    stop_reason = "timeout"
    stop_requested = False
    graceful_detach_after_single_step = False
    access_denied_skip_events: list[int] = []
    restore_observation: dict[str, Any] | None = None
    thread_crt_restore_observation: dict[str, Any] | None = None

    attach_deadline = time.monotonic() + attach_timeout
    handoff_wait_reported = False
    while True:
        if kernel32.DebugActiveProcess(pid):
            attached = True
            break
        error = ctypes.get_last_error()
        retryable_handoff_error = error in (
            ERROR_ACCESS_DENIED,
            ERROR_INVALID_PARAMETER,
        )
        if (
            retryable_handoff_error
            and time.monotonic() < attach_deadline
            and (
                current_image := process_image_path(pid)
            ) is not None
            and same_windows_path(current_image, executable)
        ):
            if not handoff_wait_reported:
                print(
                    f"rng_trace_waiting_for_debugger_handoff pid={pid}",
                    flush=True,
                )
                handoff_wait_reported = True
            time.sleep(0.01)
            continue
        raise ctypes.WinError(error)
    try:
        if not kernel32.DebugSetProcessKillOnExit(False):
            raise ctypes.WinError(ctypes.get_last_error())
        process = kernel32.OpenProcess(PROCESS_ACCESS, False, pid)
        if not process:
            raise ctypes.WinError(ctypes.get_last_error())
        original = read_memory(process, address, 1)
        if original != b"\xBA":
            raise RuntimeError(
                "global MTRand wrapper does not start with expected MOV opcode"
            )
        write_memory(process, address, b"\xCC")
        breakpoint_armed = True
        kernel32.FlushInstructionCache(process, ctypes.c_void_p(address), 1)
        _write_canonical(
            ready_path,
            {
                "schema": "zuma-rl.pc-global-mtrand-call-trace-ready",
                "version": 1,
                "process_id": pid,
                "address": address,
                "start_update": start_update,
                "end_update": end_update,
            },
        )
        print(
            f"rng_trace_armed pid={pid} address=0x{address:08X} "
            f"updates={start_update}:{end_update}",
            flush=True,
        )

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if stop_path.exists():
                stop_requested = True
            if _can_stop_without_pending_single_step(
                stop_requested=stop_requested,
                stepping_thread=stepping_thread,
            ):
                stop_reason = "stop_requested"
                break
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
                exception_address = int(
                    exception.ExceptionRecord.ExceptionAddress or 0
                )
                if (
                    code in (EXCEPTION_BREAKPOINT, STATUS_WX86_BREAKPOINT)
                    and exception_address == address
                ):
                    if (
                        not breakpoint_armed
                        or original is None
                        or stepping_thread is not None
                        or stepping_suspended_threads
                    ):
                        raise RuntimeError("global RNG breakpoint state collision")
                    thread = kernel32.OpenThread(
                        THREAD_ACCESS | THREAD_SUSPEND_RESUME,
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
                        base = _read_u32(process, G_SEXY_APP_BASE_ADDRESS)
                        update = (
                            _read_i32(
                                process,
                                base + DEFAULT_BOARD_UPDATE_OFFSET,
                            )
                            if base
                            else -1
                        )
                        state = read_memory(
                            process,
                            DEFAULT_GLOBAL_RNG_STATE,
                            MTRAND_STATE_BYTES,
                        )
                        caller = _read_u32(process, int(context.Esp))
                        if (
                            thread_crt_restore_state is not None
                            and thread_crt_restore_observation is None
                            and update == thread_crt_restore_framework_update
                            and caller == thread_crt_restore_caller
                        ):
                            thread_crt_restore_observation = (
                                _apply_thread_crt_restore(
                                    process=process,
                                    process_id=pid,
                                    thread=thread,
                                    thread_id=int(event.dwThreadId),
                                    context=context,
                                    framework_update=update,
                                    caller=caller,
                                    desired=thread_crt_restore_state,
                                )
                            )
                            print(
                                "thread_crt_call_site_restore "
                                f"update={update} "
                                f"caller=0x{caller:08X} "
                                "before="
                                f"{thread_crt_restore_observation['live_before_state']} "
                                f"after={thread_crt_restore_state.state}",
                                flush=True,
                            )
                        if (
                            restore_state is not None
                            and restore_observation is None
                            and update == restore_framework_update
                            and caller == restore_caller
                        ):
                            live_state = state
                            live_index = struct.unpack_from(
                                "<I",
                                live_state,
                                MTRAND_STATE_WORDS * 4,
                            )[0]
                            changed = live_state != restore_state.payload
                            try:
                                if changed:
                                    write_memory(
                                        process,
                                        DEFAULT_GLOBAL_RNG_STATE,
                                        restore_state.payload,
                                    )
                                state = read_memory(
                                    process,
                                    DEFAULT_GLOBAL_RNG_STATE,
                                    MTRAND_STATE_BYTES,
                                )
                                if state != restore_state.payload:
                                    raise RuntimeError(
                                        "call-site global MTRand writeback "
                                        "verification failed"
                                    )
                            except Exception:
                                try:
                                    write_memory(
                                        process,
                                        DEFAULT_GLOBAL_RNG_STATE,
                                        live_state,
                                    )
                                    if (
                                        read_memory(
                                            process,
                                            DEFAULT_GLOBAL_RNG_STATE,
                                            MTRAND_STATE_BYTES,
                                        )
                                        != live_state
                                    ):
                                        raise RuntimeError(
                                            "call-site global MTRand rollback "
                                            "verification failed"
                                        )
                                except Exception as rollback_error:
                                    raise RuntimeError(
                                        "call-site global MTRand restore and "
                                        "rollback failed"
                                    ) from rollback_error
                                raise
                            restore_observation = {
                                "classification": (
                                    "diagnostic-ephemeral-call-site-state-"
                                    "restore"
                                ),
                                "process_id": pid,
                                "thread_id": int(event.dwThreadId),
                                "framework_update": update,
                                "caller": caller,
                                "caller_hex": f"0x{caller:08x}",
                                "state_address": DEFAULT_GLOBAL_RNG_STATE,
                                "live_before_index": live_index,
                                "live_before_state_sha256": _sha256_bytes(
                                    live_state
                                ),
                                "bytes_written": (
                                    MTRAND_STATE_BYTES if changed else 0
                                ),
                                "changed": changed,
                                "writeback_verified": True,
                                "transactional_rollback_on_failure": True,
                                "persistent_file_modified": False,
                                "restored": _restore_source_evidence(
                                    restore_state
                                ),
                            }
                            print(
                                "rng_call_site_restore "
                                f"update={update} "
                                f"caller=0x{caller:08X} "
                                f"before={_sha256_bytes(live_state)} "
                                f"after={restore_state.state_sha256}",
                                flush=True,
                            )
                        index_before = struct.unpack_from(
                            "<I",
                            state,
                            MTRAND_STATE_WORDS * 4,
                        )[0]
                        output, index_after = _predicted_output(state)
                        stack = _stack_code_candidates(
                            process,
                            int(context.Esp),
                        )
                        if start_update <= update <= end_update:
                            capture_thread_crt = (
                                thread_crt_restore_observation is not None
                                and int(event.dwThreadId)
                                == int(
                                    thread_crt_restore_observation["thread_id"]
                                )
                            )
                            thread_crt_evidence = (
                                {
                                    "thread_crt_snapshot_requested": True,
                                    **_thread_crt_snapshot(
                                        process=process,
                                        thread=thread,
                                        context=context,
                                        thread_id=int(event.dwThreadId),
                                    ),
                                }
                                if capture_thread_crt
                                else {
                                    "thread_crt_snapshot_requested": False,
                                    "thread_crt_state": None,
                                    "thread_crt_rand_state": None,
                                    "thread_crt_snapshot_error": None,
                                }
                            )
                            row = {
                                "order": len(calls),
                                "framework_update": update,
                                "thread_id": int(event.dwThreadId),
                                "caller": caller,
                                "caller_hex": f"0x{caller:08x}",
                                "mtrand_index_before": index_before,
                                "mtrand_index_after": index_after,
                                "mtrand_output": output,
                                "mtrand_state_sha256_before": _sha256_bytes(
                                    state
                                ),
                                **thread_crt_evidence,
                                "stack_code": [
                                    {
                                        "stack_offset": offset,
                                        "address": candidate,
                                        "address_hex": f"0x{candidate:08x}",
                                    }
                                    for offset, candidate in stack
                                ],
                            }
                            calls.append(row)
                            print(
                                "rng_call "
                                f"order={row['order']} update={update} "
                                f"tid={event.dwThreadId} "
                                f"caller=0x{caller:08X} "
                                f"index={index_before}->{index_after} "
                                f"output={output}",
                                flush=True,
                            )

                        write_memory(process, address, original)
                        breakpoint_armed = False
                        kernel32.FlushInstructionCache(
                            process,
                            ctypes.c_void_p(address),
                            1,
                        )
                        context.Eip = address
                        context.EFlags |= TRAP_FLAG
                        if not kernel32.Wow64SetThreadContext(
                            thread,
                            ctypes.byref(context),
                        ):
                            raise ctypes.WinError(ctypes.get_last_error())
                        stepping_suspended_threads = _suspend_other_threads(
                            pid,
                            excluded_thread_id=int(event.dwThreadId),
                            access_denied_skip_events=(
                                access_denied_skip_events
                            ),
                        )
                        stepping_thread = int(event.dwThreadId)
                    finally:
                        kernel32.CloseHandle(thread)
                elif (
                    code in (EXCEPTION_SINGLE_STEP, STATUS_WX86_SINGLE_STEP)
                    and stepping_thread == int(event.dwThreadId)
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
                        context.EFlags &= ~TRAP_FLAG
                        if not kernel32.Wow64SetThreadContext(
                            thread,
                            ctypes.byref(context),
                        ):
                            raise ctypes.WinError(ctypes.get_last_error())
                        if stop_requested:
                            graceful_detach_after_single_step = True
                        else:
                            write_memory(process, address, b"\xCC")
                            breakpoint_armed = True
                            kernel32.FlushInstructionCache(
                                process,
                                ctypes.c_void_p(address),
                                1,
                            )
                        stepping_thread = None
                        _resume_suspended_threads(stepping_suspended_threads)
                    finally:
                        kernel32.CloseHandle(thread)
                elif code == MICROSOFT_CPP_EXCEPTION:
                    status = DBG_EXCEPTION_NOT_HANDLED
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
                exited = True
                exit_code = int(event.u.ExitProcess.dwExitCode)
                stop_reason = "process_exit"

            if not kernel32.ContinueDebugEvent(
                event.dwProcessId,
                event.dwThreadId,
                status,
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            if exited:
                break
            if graceful_detach_after_single_step:
                stop_reason = "stop_requested"
                break
        else:
            stop_reason = "timeout"
    finally:
        if stepping_suspended_threads:
            try:
                _resume_suspended_threads(stepping_suspended_threads)
            except OSError:
                pass
        if process:
            if original is not None and breakpoint_armed:
                try:
                    write_memory(process, address, original)
                    kernel32.FlushInstructionCache(
                        process,
                        ctypes.c_void_p(address),
                        1,
                    )
                except OSError:
                    pass
            kernel32.CloseHandle(process)
        if attached:
            kernel32.DebugActiveProcessStop(pid)

    missed_restores: list[str] = []
    if restore_state is not None and restore_observation is None:
        missed_restores.append("global_mtrand")
    if (
        thread_crt_restore_state is not None
        and thread_crt_restore_observation is None
    ):
        missed_restores.append("thread_crt")
    status = "FAIL" if missed_restores else "PASS"
    result = {
        "schema": TRACE_SCHEMA,
        "version": TRACE_VERSION,
        "status": status,
        "failure": (
            "requested_call_site_restore_not_observed:"
            + ",".join(missed_restores)
            if missed_restores
            else None
        ),
        "process_id": pid,
        "runtime_executable": str(executable.resolve()),
        "runtime_executable_sha256": _sha256_path(executable),
        "address": address,
        "address_hex": f"0x{address:08x}",
        "global_mtrand_address": DEFAULT_GLOBAL_RNG_STATE,
        "global_mtrand_address_hex": f"0x{DEFAULT_GLOBAL_RNG_STATE:08x}",
        "start_update": start_update,
        "end_update": end_update,
        "attach_timeout_seconds": attach_timeout,
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "started_perf_counter_ns": started_ns,
        "finished_perf_counter_ns": time.perf_counter_ns(),
        "stop_reason": stop_reason,
        "graceful_detach_after_single_step": (
            graceful_detach_after_single_step
        ),
        "exited": exited,
        "exit_code": exit_code,
        "call_count": len(calls),
        "access_denied_thread_skip_event_count": len(
            access_denied_skip_events
        ),
        "access_denied_thread_skip_ids": sorted(
            set(access_denied_skip_events)
        ),
        "process_memory_mutation": (
            restore_observation is not None
            or thread_crt_restore_observation is not None
        ),
        "global_mtrand_call_site_restore": restore_observation,
        "thread_crt_call_site_restore": thread_crt_restore_observation,
        "calls": calls,
    }
    _write_canonical(output_path, result)
    print(
        f"rng_trace_result status={status} calls={len(calls)} "
        f"stop_reason={stop_reason}",
        flush=True,
    )
    print(output_path.resolve(), flush=True)
    return output_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", required=True, type=int)
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ready", required=True, type=Path)
    parser.add_argument("--stop", required=True, type=Path)
    parser.add_argument("--start-update", required=True, type=int)
    parser.add_argument("--end-update", required=True, type=int)
    parser.add_argument(
        "--address",
        type=lambda value: int(value, 0),
        default=DEFAULT_GLOBAL_RNG_WRAPPER,
    )
    parser.add_argument("--attach-timeout", type=float, default=30.0)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--restore-log", type=Path)
    parser.add_argument("--restore-source-update", type=int)
    parser.add_argument("--restore-seed", type=lambda value: int(value, 0))
    parser.add_argument("--restore-rewind-draws", type=int, default=0)
    parser.add_argument("--restore-maximum-draws", type=int, default=100_000)
    parser.add_argument("--restore-framework-update", type=int)
    parser.add_argument(
        "--restore-caller",
        type=lambda value: int(value, 0),
    )
    parser.add_argument("--thread-crt-restore-trace", type=Path)
    parser.add_argument("--thread-crt-restore-recording-report", type=Path)
    parser.add_argument("--thread-crt-restore-source-order", type=int)
    parser.add_argument("--thread-crt-restore-framework-update", type=int)
    parser.add_argument(
        "--thread-crt-restore-caller",
        type=lambda value: int(value, 0),
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    restore_group = (
        args.restore_log,
        args.restore_source_update,
        args.restore_seed,
        args.restore_framework_update,
        args.restore_caller,
    )
    restore_requested = any(value is not None for value in restore_group)
    if restore_requested and any(value is None for value in restore_group):
        parser.error(
            "all call-site restore target and source options are required"
        )
    restore_state = (
        load_global_mtrand_restore_state(
            args.restore_log.resolve(),
            framework_update=args.restore_source_update,
            seed=args.restore_seed,
            maximum_draws=args.restore_maximum_draws,
            rewind_draws=args.restore_rewind_draws,
        )
        if restore_requested
        else None
    )
    thread_crt_restore_group = (
        args.thread_crt_restore_trace,
        args.thread_crt_restore_recording_report,
        args.thread_crt_restore_source_order,
        args.thread_crt_restore_framework_update,
        args.thread_crt_restore_caller,
    )
    thread_crt_restore_requested = any(
        value is not None for value in thread_crt_restore_group
    )
    if thread_crt_restore_requested and any(
        value is None for value in thread_crt_restore_group
    ):
        parser.error(
            "all source-bound thread CRT restore options are required"
        )
    thread_crt_restore_state = (
        load_source_bound_thread_crt_restore_state(
            args.thread_crt_restore_trace.resolve(),
            args.thread_crt_restore_recording_report.resolve(),
            source_order=args.thread_crt_restore_source_order,
            framework_update=args.thread_crt_restore_framework_update,
            caller=args.thread_crt_restore_caller,
        )
        if thread_crt_restore_requested
        else None
    )
    trace_global_rng_calls(
        pid=args.pid,
        executable=args.executable.resolve(),
        output_path=args.output.resolve(),
        ready_path=args.ready.resolve(),
        stop_path=args.stop.resolve(),
        start_update=args.start_update,
        end_update=args.end_update,
        address=args.address,
        attach_timeout=args.attach_timeout,
        timeout=args.timeout,
        restore_state=restore_state,
        restore_framework_update=args.restore_framework_update,
        restore_caller=args.restore_caller,
        thread_crt_restore_state=thread_crt_restore_state,
        thread_crt_restore_framework_update=(
            args.thread_crt_restore_framework_update
        ),
        thread_crt_restore_caller=args.thread_crt_restore_caller,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
