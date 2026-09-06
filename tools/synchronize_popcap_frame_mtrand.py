"""Synchronize the retail gameplay MTRand call with a natural-run schedule."""

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
from tools.popcap_mtrand_frame_schedule import (
    FrameMTRandSyncSchedule,
    load_frame_mtrand_sync_schedule,
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
    _read_i32,
    _thread_ids_for_pid,
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


DEFAULT_GAMEPLAY_RNG_RETURN = 0x004B5ADF
DEFAULT_GLOBAL_RNG_STATE = 0x00A313C0
DEFAULT_BOARD_UPDATE_OFFSET = 0x4C4
CONTEXT_DEBUG_REGISTERS = 0x00010010
CONTEXT_FULL_AND_DEBUG = CONTEXT_FULL | CONTEXT_DEBUG_REGISTERS
RESUME_FLAG = 0x00010000
INVALID_SUSPEND_COUNT = 0xFFFFFFFF
TRACE_SCHEMA = "zuma-rl.pc-frame-mtrand-synchronizer"
TRACE_VERSION = 1


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


def _open_main_thread(main_thread_id: int) -> int:
    handle = kernel32.OpenThread(
        THREAD_ACCESS | THREAD_SUSPEND_RESUME,
        False,
        main_thread_id,
    )
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    return int(handle)


def _read_context(thread: int) -> WOW64_CONTEXT:
    context = WOW64_CONTEXT()
    context.ContextFlags = CONTEXT_FULL_AND_DEBUG
    if not kernel32.Wow64GetThreadContext(
        thread,
        ctypes.byref(context),
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    return context


def _write_context(thread: int, context: WOW64_CONTEXT) -> None:
    context.ContextFlags = CONTEXT_FULL_AND_DEBUG
    if not kernel32.Wow64SetThreadContext(
        thread,
        ctypes.byref(context),
    ):
        raise ctypes.WinError(ctypes.get_last_error())


def _arm_hardware_breakpoint(
    *,
    main_thread_id: int,
    address: int,
) -> tuple[int, int, int, int, int, int]:
    thread = _open_main_thread(main_thread_id)
    try:
        context = _read_context(thread)
        original = (
            int(context.Dr0),
            int(context.Dr1),
            int(context.Dr2),
            int(context.Dr3),
            int(context.Dr6),
            int(context.Dr7),
        )
        if context.Dr7 & 0x3:
            raise RuntimeError("hardware breakpoint slot 0 is already active")
        context.Dr0 = address
        context.Dr6 = 0
        context.Dr7 &= ~0x000F0003
        context.Dr7 |= 0x1
        _write_context(thread, context)
        verified = _read_context(thread)
        if (
            int(verified.Dr0) != address
            or int(verified.Dr7) & 0x1 != 0x1
            or int(verified.Dr7) & 0x000F0000
        ):
            raise RuntimeError("hardware breakpoint writeback mismatch")
        return original
    finally:
        kernel32.CloseHandle(thread)


def _restore_hardware_breakpoint(
    *,
    main_thread_id: int,
    original: tuple[int, int, int, int, int, int],
) -> None:
    thread = _open_main_thread(main_thread_id)
    suspended = False
    resume_error: OSError | None = None
    try:
        previous = int(kernel32.SuspendThread(thread))
        if previous == INVALID_SUSPEND_COUNT:
            raise ctypes.WinError(ctypes.get_last_error())
        suspended = True
        context = _read_context(thread)
        (
            context.Dr0,
            context.Dr1,
            context.Dr2,
            context.Dr3,
            context.Dr6,
            context.Dr7,
        ) = original
        _write_context(thread, context)
        verified = _read_context(thread)
        observed = (
            int(verified.Dr0),
            int(verified.Dr1),
            int(verified.Dr2),
            int(verified.Dr3),
            int(verified.Dr6),
            int(verified.Dr7),
        )
        if observed != original:
            raise RuntimeError("hardware breakpoint restore mismatch")
    finally:
        if suspended:
            result = int(kernel32.ResumeThread(thread))
            if result == INVALID_SUSPEND_COUNT:
                resume_error = ctypes.WinError(ctypes.get_last_error())
        kernel32.CloseHandle(thread)
        if resume_error is not None:
            raise resume_error


def synchronize_frame_mtrand(
    *,
    pid: int,
    main_thread_id: int,
    executable: Path,
    schedule: FrameMTRandSyncSchedule,
    output_path: Path,
    ready_path: Path,
    stop_path: Path,
    address: int = DEFAULT_GAMEPLAY_RNG_RETURN,
    attach_timeout: float = 60.0,
    timeout: float = 600.0,
) -> Path:
    """Apply one cryptographically derived post-call state per natural hit."""

    if (
        pid <= 0
        or main_thread_id <= 0
        or address <= 0
        or attach_timeout <= 0
        or timeout <= 0
    ):
        raise ValueError("invalid frame synchronizer target")
    for path in (output_path, ready_path, stop_path):
        if path.exists() or not path.parent.is_dir():
            raise FileExistsError(f"synchronizer path is not new: {path}")
    observed = process_image_path(pid)
    if observed is None or not same_windows_path(observed, executable):
        raise RuntimeError(
            f"PID {pid} path mismatch: observed={observed}, expected={executable}"
        )
    if main_thread_id not in _thread_ids_for_pid(pid):
        raise RuntimeError("declared main thread does not belong to target PID")

    attached = False
    process = 0
    hardware_armed = False
    hardware_original: tuple[int, int, int, int, int, int] | None = None
    hits: list[dict[str, Any]] = []
    hit_updates: set[int] = set()
    unexpected_hits: list[int] = []
    duplicate_hits: list[int] = []
    exited = False
    exit_code: int | None = None
    stop_reason = "timeout"
    failure: str | None = None
    hardware_restore_error: str | None = None
    started_utc = datetime.now(timezone.utc).isoformat()
    started_ns = time.perf_counter_ns()
    schedule_by_update = schedule.by_update

    attach_deadline = time.monotonic() + attach_timeout
    wait_reported = False
    while True:
        if kernel32.DebugActiveProcess(pid):
            attached = True
            break
        error = ctypes.get_last_error()
        retryable = error in (ERROR_ACCESS_DENIED, ERROR_INVALID_PARAMETER)
        current_image = process_image_path(pid)
        if (
            retryable
            and time.monotonic() < attach_deadline
            and current_image is not None
            and same_windows_path(current_image, executable)
        ):
            if not wait_reported:
                print(
                    f"frame_sync_waiting_for_debugger_handoff pid={pid}",
                    flush=True,
                )
                wait_reported = True
            time.sleep(0.01)
            continue
        raise ctypes.WinError(error)

    try:
        if not kernel32.DebugSetProcessKillOnExit(False):
            raise ctypes.WinError(ctypes.get_last_error())
        process = kernel32.OpenProcess(PROCESS_ACCESS, False, pid)
        if not process:
            raise ctypes.WinError(ctypes.get_last_error())

        # DebugActiveProcess leaves an initial debug event pending, so every
        # target thread is stopped while its hardware register is configured.
        hardware_original = _arm_hardware_breakpoint(
            main_thread_id=main_thread_id,
            address=address,
        )
        hardware_armed = True
        _write_canonical(
            ready_path,
            {
                "schema": "zuma-rl.pc-frame-mtrand-synchronizer-ready",
                "version": 1,
                "process_id": pid,
                "main_thread_id": main_thread_id,
                "address": address,
                "start_update": schedule.start_update,
                "end_update": schedule.end_update,
                "schedule_semantic_sha256": schedule.semantic_sha256,
            },
        )
        print(
            f"frame_sync_armed pid={pid} tid={main_thread_id} "
            f"address=0x{address:08X} "
            f"updates={schedule.start_update}:{schedule.end_update} "
            f"expected_hits={len(schedule.entries)}",
            flush=True,
        )

        deadline = time.monotonic() + timeout
        stop_after_continue = False
        while time.monotonic() < deadline:
            if stop_path.exists():
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
                is_single_step = code in (
                    EXCEPTION_SINGLE_STEP,
                    STATUS_WX86_SINGLE_STEP,
                )
                if (
                    is_single_step
                    and int(event.dwThreadId) == main_thread_id
                ):
                    thread = _open_main_thread(main_thread_id)
                    try:
                        context = _read_context(thread)
                        is_slot_zero_hit = bool(int(context.Dr6) & 0x1)
                        is_target = (
                            is_slot_zero_hit
                            or exception_address == address
                            or int(context.Eip) == address
                        )
                        if is_target:
                            base = struct.unpack(
                                "<I",
                                read_memory(
                                    process,
                                    G_SEXY_APP_BASE_ADDRESS,
                                    4,
                                ),
                            )[0]
                            update = (
                                _read_i32(
                                    process,
                                    base + DEFAULT_BOARD_UPDATE_OFFSET,
                                )
                                if base
                                else -1
                            )
                            if update > schedule.end_update:
                                (
                                    context.Dr0,
                                    context.Dr1,
                                    context.Dr2,
                                    context.Dr3,
                                    context.Dr6,
                                    context.Dr7,
                                ) = hardware_original
                                context.EFlags |= RESUME_FLAG
                                _write_context(thread, context)
                                hardware_armed = False
                                stop_reason = "end_update_reached"
                                stop_after_continue = True
                            elif update >= schedule.start_update:
                                entry = schedule_by_update.get(update)
                                if entry is None:
                                    unexpected_hits.append(update)
                                    failure = (
                                        "unexpected_gameplay_rng_hit"
                                    )
                                    stop_reason = "semantic_failure"
                                    context.Dr6 = 0
                                    context.EFlags |= RESUME_FLAG
                                    _write_context(thread, context)
                                    stop_after_continue = True
                                elif update in hit_updates:
                                    duplicate_hits.append(update)
                                    failure = "duplicate_gameplay_rng_hit"
                                    stop_reason = "semantic_failure"
                                    context.Dr6 = 0
                                    context.EFlags |= RESUME_FLAG
                                    _write_context(thread, context)
                                    stop_after_continue = True
                                else:
                                    live_state = read_memory(
                                        process,
                                        DEFAULT_GLOBAL_RNG_STATE,
                                        MTRAND_STATE_BYTES,
                                    )
                                    live_index = struct.unpack_from(
                                        "<I",
                                        live_state,
                                        MTRAND_STATE_WORDS * 4,
                                    )[0]
                                    live_eax = int(context.Eax)
                                    changed = (
                                        live_state != entry.post_payload
                                    )
                                    try:
                                        if changed:
                                            write_memory(
                                                process,
                                                DEFAULT_GLOBAL_RNG_STATE,
                                                entry.post_payload,
                                            )
                                        if (
                                            read_memory(
                                                process,
                                                DEFAULT_GLOBAL_RNG_STATE,
                                                MTRAND_STATE_BYTES,
                                            )
                                            != entry.post_payload
                                        ):
                                            raise RuntimeError(
                                                "frame sync writeback mismatch"
                                            )
                                    except Exception:
                                        try:
                                            write_memory(
                                                process,
                                                DEFAULT_GLOBAL_RNG_STATE,
                                                live_state,
                                            )
                                        except Exception as rollback_error:
                                            raise RuntimeError(
                                                "frame sync write and rollback "
                                                "both failed"
                                            ) from rollback_error
                                        raise
                                    context.Eax = entry.output
                                    context.Dr6 = 0
                                    context.EFlags |= RESUME_FLAG
                                    _write_context(thread, context)
                                    hit_updates.add(update)
                                    hits.append(
                                        {
                                            "framework_update": update,
                                            "thread_id": main_thread_id,
                                            "live_eax": live_eax,
                                            "restored_eax": entry.output,
                                            "live_index": live_index,
                                            "live_state_sha256": (
                                                _sha256_bytes(live_state)
                                            ),
                                            "bytes_written": (
                                                MTRAND_STATE_BYTES
                                                if changed
                                                else 0
                                            ),
                                            "changed": changed,
                                            **entry.evidence_dict(),
                                        }
                                    )
                                    if len(hits) % 250 == 0:
                                        print(
                                            "frame_sync_progress "
                                            f"hits={len(hits)} "
                                            f"update={update}",
                                            flush=True,
                                        )
                        else:
                            status = DBG_EXCEPTION_NOT_HANDLED
                    finally:
                        kernel32.CloseHandle(thread)
                elif code in (
                    EXCEPTION_BREAKPOINT,
                    STATUS_WX86_BREAKPOINT,
                ):
                    # Consume the normal debugger-attach breakpoint.
                    status = DBG_CONTINUE
                elif code == MICROSOFT_CPP_EXCEPTION:
                    status = DBG_EXCEPTION_NOT_HANDLED
                else:
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
            if exited or stop_after_continue:
                break
        else:
            stop_reason = "timeout"
    finally:
        if hardware_armed and hardware_original is not None and not exited:
            try:
                _restore_hardware_breakpoint(
                    main_thread_id=main_thread_id,
                    original=hardware_original,
                )
                hardware_armed = False
            except Exception as error:
                hardware_restore_error = (
                    f"{type(error).__name__}: {error}"
                )
        if process:
            kernel32.CloseHandle(process)
        if attached:
            kernel32.DebugActiveProcessStop(pid)

    expected_updates = {
        entry.framework_update for entry in schedule.entries
    }
    missing_updates = sorted(expected_updates.difference(hit_updates))
    if failure is None and missing_updates:
        failure = "missing_gameplay_rng_hits"
    if failure is None and hardware_restore_error is not None:
        failure = "hardware_breakpoint_restore_failed"
    status = "PASS" if failure is None else "FAIL"
    result = {
        "schema": TRACE_SCHEMA,
        "version": TRACE_VERSION,
        "status": status,
        "failure": failure,
        "classification": "diagnostic-ephemeral-frame-rng-synchronization",
        "process_id": pid,
        "main_thread_id": main_thread_id,
        "runtime_executable": str(executable.resolve()),
        "runtime_executable_sha256": _sha256_path(executable),
        "address": address,
        "address_hex": f"0x{address:08x}",
        "global_mtrand_address": DEFAULT_GLOBAL_RNG_STATE,
        "schedule": {
            "source_path": str(schedule.source_path),
            "source_sha256": schedule.source_sha256,
            "source_process_id": schedule.source_process_id,
            "seed": schedule.seed,
            "start_update": schedule.start_update,
            "end_update": schedule.end_update,
            "entry_count": len(schedule.entries),
            "semantic_sha256": schedule.semantic_sha256,
        },
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "started_perf_counter_ns": started_ns,
        "finished_perf_counter_ns": time.perf_counter_ns(),
        "stop_reason": stop_reason,
        "exited": exited,
        "exit_code": exit_code,
        "expected_hit_count": len(schedule.entries),
        "hit_count": len(hits),
        "missing_hit_count": len(missing_updates),
        "missing_updates": missing_updates,
        "unexpected_hits": unexpected_hits,
        "duplicate_hits": duplicate_hits,
        "process_memory_mutation": bool(hits),
        "persistent_file_modified": False,
        "hardware_breakpoint_restored": not hardware_armed,
        "hardware_breakpoint_restore_error": hardware_restore_error,
        "hits": hits,
    }
    _write_canonical(output_path, result)
    print(
        f"frame_sync_result status={status} hits={len(hits)}/"
        f"{len(schedule.entries)} stop_reason={stop_reason}",
        flush=True,
    )
    print(output_path.resolve(), flush=True)
    return output_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", required=True, type=int)
    parser.add_argument("--main-thread-id", required=True, type=int)
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--reference-log", required=True, type=Path)
    parser.add_argument(
        "--seed",
        required=True,
        type=lambda value: int(value, 0),
    )
    parser.add_argument("--start-update", required=True, type=int)
    parser.add_argument("--end-update", required=True, type=int)
    parser.add_argument("--maximum-draws", type=int, default=100_000)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ready", required=True, type=Path)
    parser.add_argument("--stop", required=True, type=Path)
    parser.add_argument(
        "--address",
        type=lambda value: int(value, 0),
        default=DEFAULT_GAMEPLAY_RNG_RETURN,
    )
    parser.add_argument("--attach-timeout", type=float, default=60.0)
    parser.add_argument("--timeout", type=float, default=600.0)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    schedule = load_frame_mtrand_sync_schedule(
        args.reference_log.resolve(),
        seed=args.seed,
        start_update=args.start_update,
        end_update=args.end_update,
        maximum_draws=args.maximum_draws,
    )
    result_path = synchronize_frame_mtrand(
        pid=args.pid,
        main_thread_id=args.main_thread_id,
        executable=args.executable.resolve(),
        schedule=schedule,
        output_path=args.output.resolve(),
        ready_path=args.ready.resolve(),
        stop_path=args.stop.resolve(),
        address=args.address,
        attach_timeout=args.attach_timeout,
        timeout=args.timeout,
    )
    result = json.loads(result_path.read_text(encoding="ascii"))
    return 0 if result.get("status") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
