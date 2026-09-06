"""Synchronize retail gameplay MTRand calls to an exact natural-run oracle."""

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
from tools.popcap_mtrand_call_oracle import (
    GameplayMTRandOracle,
    load_gameplay_mtrand_oracle,
)
from tools.synchronize_popcap_frame_mtrand import (
    DEFAULT_GAMEPLAY_RNG_RETURN,
    DEFAULT_GLOBAL_RNG_STATE,
    RESUME_FLAG,
    _arm_hardware_breakpoint,
    _open_main_thread,
    _read_context,
    _restore_hardware_breakpoint,
    _write_context,
)
from tools.trace_popcap_demo_commands import (
    ERROR_ACCESS_DENIED,
    ERROR_INVALID_PARAMETER,
    ERROR_SEM_TIMEOUT,
    EXCEPTION_SINGLE_STEP,
    MTRAND_STATE_BYTES,
    MTRAND_STATE_WORDS,
    STATUS_WX86_BREAKPOINT,
    STATUS_WX86_SINGLE_STEP,
    _thread_ids_for_pid,
)
from tools.trace_popcap_gameplay_mtrand import _board_snapshot
from tools.trace_popcap_shutdown import (
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
    kernel32,
    read_memory,
    write_memory,
)


TRACE_SCHEMA = "zuma-rl.pc-gameplay-mtrand-oracle-synchronizer"
TRACE_VERSION = 1
PROCESS_ALL_ACCESS = 0x001F0FFF

kernel32.DebugBreakProcess.argtypes = (wintypes.HANDLE,)
kernel32.DebugBreakProcess.restype = wintypes.BOOL


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


def synchronize_gameplay_mtrand(
    *,
    pid: int,
    main_thread_id: int,
    executable: Path,
    oracle: GameplayMTRandOracle,
    output_path: Path,
    ready_path: Path,
    stop_path: Path,
    address: int = DEFAULT_GAMEPLAY_RNG_RETURN,
    attach_timeout: float = 60.0,
    timeout: float = 900.0,
) -> Path:
    """Apply each verified natural post-call state and output in order."""

    if (
        pid <= 0
        or main_thread_id <= 0
        or address <= 0
        or attach_timeout <= 0
        or timeout <= 0
        or not oracle.entries
    ):
        raise ValueError("invalid gameplay MTRand synchronizer target")
    for path in (output_path, ready_path, stop_path):
        if path.exists() or not path.parent.is_dir():
            raise FileExistsError(
                f"synchronizer path is not new: {path}"
            )
    observed = process_image_path(pid)
    if observed is None or not same_windows_path(observed, executable):
        raise RuntimeError(
            f"PID {pid} path mismatch: observed={observed}, "
            f"expected={executable}"
        )
    if main_thread_id not in _thread_ids_for_pid(pid):
        raise RuntimeError("declared main thread does not belong to target PID")

    attached = False
    process = 0
    control_process = 0
    hardware_armed = False
    hardware_original: tuple[int, int, int, int, int, int] | None = None
    hardware_restore_error: str | None = None
    detach_error: str | None = None
    hits: list[dict[str, Any]] = []
    exited = False
    exit_code: int | None = None
    stop_reason = "timeout"
    failure: str | None = None
    exception_detail: str | None = None
    extra_call: dict[str, Any] | None = None
    stop_break_requested = False
    stop_break_observed = False
    started_utc = datetime.now(timezone.utc).isoformat()
    started_ns = time.perf_counter_ns()

    try:
        attach_deadline = time.monotonic() + attach_timeout
        wait_reported = False
        while True:
            if kernel32.DebugActiveProcess(pid):
                attached = True
                break
            error = ctypes.get_last_error()
            retryable = error in (
                ERROR_ACCESS_DENIED,
                ERROR_INVALID_PARAMETER,
            )
            current_image = process_image_path(pid)
            if (
                retryable
                and time.monotonic() < attach_deadline
                and current_image is not None
                and same_windows_path(current_image, executable)
            ):
                if not wait_reported:
                    print(
                        "gameplay_sync_waiting_for_debugger_handoff "
                        f"pid={pid}",
                        flush=True,
                    )
                    wait_reported = True
                time.sleep(0.01)
                continue
            raise ctypes.WinError(error)

        if not kernel32.DebugSetProcessKillOnExit(False):
            raise ctypes.WinError(ctypes.get_last_error())
        process = kernel32.OpenProcess(PROCESS_ACCESS, False, pid)
        if not process:
            raise ctypes.WinError(ctypes.get_last_error())
        control_process = kernel32.OpenProcess(
            PROCESS_ALL_ACCESS,
            False,
            pid,
        )
        if not control_process:
            raise ctypes.WinError(ctypes.get_last_error())

        hardware_original = _arm_hardware_breakpoint(
            main_thread_id=main_thread_id,
            address=address,
        )
        hardware_armed = True
        _write_canonical(
            ready_path,
            {
                "schema": (
                    "zuma-rl.pc-gameplay-mtrand-oracle-"
                    "synchronizer-ready"
                ),
                "version": 1,
                "process_id": pid,
                "main_thread_id": main_thread_id,
                "address": address,
                "expected_call_count": len(oracle.entries),
                "oracle_semantic_sha256": oracle.semantic_sha256,
            },
        )
        print(
            f"gameplay_sync_armed pid={pid} tid={main_thread_id} "
            f"address=0x{address:08X} "
            f"expected_calls={len(oracle.entries)}",
            flush=True,
        )

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if stop_path.exists() and not stop_break_requested:
                if not kernel32.DebugBreakProcess(control_process):
                    raise ctypes.WinError(ctypes.get_last_error())
                stop_break_requested = True
            event = DEBUG_EVENT()
            if not kernel32.WaitForDebugEvent(ctypes.byref(event), 100):
                error = ctypes.get_last_error()
                if error == ERROR_SEM_TIMEOUT:
                    if stop_break_observed:
                        break
                    continue
                raise ctypes.WinError(error)

            status = DBG_CONTINUE
            stop_after_continue = False
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
                    and stop_break_observed
                ):
                    thread = _open_main_thread(main_thread_id)
                    try:
                        context = _read_context(thread)
                        if hardware_original is not None:
                            context.Dr6 = hardware_original[4]
                        else:
                            context.Dr6 = 0
                        context.EFlags |= RESUME_FLAG
                        _write_context(thread, context)
                    finally:
                        kernel32.CloseHandle(thread)
                    status = DBG_CONTINUE
                elif (
                    is_single_step
                    and int(event.dwThreadId) == main_thread_id
                ):
                    thread = _open_main_thread(main_thread_id)
                    try:
                        context = _read_context(thread)
                        is_target = (
                            bool(int(context.Dr6) & 0x1)
                            or exception_address == address
                            or int(context.Eip) == address
                        )
                        if is_target:
                            board = _board_snapshot(process)
                            if len(hits) >= len(oracle.entries):
                                extra_call = {
                                    "order": len(hits),
                                    "thread_id": main_thread_id,
                                    "live_eax": int(context.Eax),
                                    **board,
                                }
                                failure = "unexpected_extra_gameplay_call"
                                stop_reason = "semantic_failure"
                                context.Dr6 = 0
                                context.EFlags |= RESUME_FLAG
                                _write_context(thread, context)
                                stop_after_continue = True
                            else:
                                entry = oracle.entries[len(hits)]
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
                                    observed_state = read_memory(
                                        process,
                                        DEFAULT_GLOBAL_RNG_STATE,
                                        MTRAND_STATE_BYTES,
                                    )
                                    if observed_state != entry.post_payload:
                                        raise RuntimeError(
                                            "gameplay sync writeback "
                                            "mismatch"
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
                                            "gameplay sync write and "
                                            "rollback both failed"
                                        ) from rollback_error
                                    raise
                                context.Eax = entry.output
                                context.Dr6 = 0
                                context.EFlags |= RESUME_FLAG
                                _write_context(thread, context)
                                observed_update = int(
                                    board["framework_update"]
                                )
                                observed_native_raw = board[
                                    "native_game_time"
                                ]
                                observed_native = (
                                    int(observed_native_raw)
                                    if observed_native_raw is not None
                                    else None
                                )
                                observed_score_raw = board["score"]
                                observed_score = (
                                    int(observed_score_raw)
                                    if observed_score_raw is not None
                                    else None
                                )
                                hits.append(
                                    {
                                        "order": entry.order,
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
                                        "observed_framework_update": (
                                            observed_update
                                        ),
                                        "reference_framework_update": (
                                            entry.framework_update
                                        ),
                                        "framework_update_delta": (
                                            observed_update
                                            - entry.framework_update
                                        ),
                                        "observed_native_game_time": (
                                            observed_native
                                        ),
                                        "reference_native_game_time": (
                                            entry.native_game_time
                                        ),
                                        "native_game_time_delta": (
                                            observed_native
                                            - entry.native_game_time
                                            if observed_native is not None
                                            else None
                                        ),
                                        "observed_score": observed_score,
                                        "reference_score": entry.score,
                                        "score_matches": (
                                            observed_score == entry.score
                                        ),
                                        "score_target": board[
                                            "score_target"
                                        ],
                                        "post_draw_count": (
                                            entry.post_draw_count
                                        ),
                                        "post_index": entry.post_index,
                                        "post_state_sha256": (
                                            entry.post_state_sha256
                                        ),
                                    }
                                )
                                if len(hits) % 100 == 0:
                                    print(
                                        "gameplay_sync_progress "
                                        f"hits={len(hits)}/"
                                        f"{len(oracle.entries)} "
                                        f"update={observed_update}",
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
                    status = DBG_CONTINUE
                    if (
                        stop_break_requested
                        and exception_address != address
                    ):
                        if (
                            hardware_armed
                            and hardware_original is not None
                        ):
                            _restore_hardware_breakpoint(
                                main_thread_id=main_thread_id,
                                original=hardware_original,
                            )
                            hardware_armed = False
                        stop_break_observed = True
                        stop_reason = "stop_requested"
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
            failure = "timeout"
    except Exception as error:
        failure = "synchronizer_exception"
        exception_detail = f"{type(error).__name__}: {error}"
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
        elif exited:
            hardware_armed = False
        if process:
            kernel32.CloseHandle(process)
        if control_process:
            kernel32.CloseHandle(control_process)
        if attached and not exited:
            if not kernel32.DebugActiveProcessStop(pid):
                detach_error = (
                    f"OSError: {ctypes.WinError(ctypes.get_last_error())}"
                )

    missing_count = len(oracle.entries) - len(hits)
    if failure is None and missing_count:
        failure = "missing_gameplay_calls"
    if failure is None and hardware_restore_error is not None:
        failure = "hardware_breakpoint_restore_failed"
    if failure is None and detach_error is not None:
        failure = "debugger_detach_failed"
    status = "PASS" if failure is None else "FAIL"
    result = {
        "schema": TRACE_SCHEMA,
        "version": TRACE_VERSION,
        "status": status,
        "failure": failure,
        "exception_detail": exception_detail,
        "classification": (
            "diagnostic-ephemeral-gameplay-rng-synchronization"
        ),
        "process_id": pid,
        "main_thread_id": main_thread_id,
        "runtime_executable": str(executable.resolve()),
        "runtime_executable_sha256": _sha256_path(executable),
        "address": address,
        "address_hex": f"0x{address:08x}",
        "global_mtrand_address": DEFAULT_GLOBAL_RNG_STATE,
        "oracle": {
            "source_path": str(oracle.source_path),
            "source_sha256": oracle.source_sha256,
            "source_process_id": oracle.source_process_id,
            "seed": oracle.seed,
            "entry_count": len(oracle.entries),
            "semantic_sha256": oracle.semantic_sha256,
        },
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "started_perf_counter_ns": started_ns,
        "finished_perf_counter_ns": time.perf_counter_ns(),
        "stop_reason": stop_reason,
        "stop_break_requested": stop_break_requested,
        "stop_break_observed": stop_break_observed,
        "exited": exited,
        "exit_code": exit_code,
        "expected_hit_count": len(oracle.entries),
        "hit_count": len(hits),
        "missing_hit_count": missing_count,
        "unexpected_extra_call": extra_call,
        "process_memory_mutation": bool(hits),
        "persistent_file_modified": False,
        "hardware_breakpoint_restored": not hardware_armed,
        "hardware_breakpoint_restore_error": hardware_restore_error,
        "debugger_detach_error": detach_error,
        "hits": hits,
    }
    _write_canonical(output_path, result)
    print(
        f"gameplay_sync_result status={status} hits={len(hits)}/"
        f"{len(oracle.entries)} stop_reason={stop_reason}",
        flush=True,
    )
    print(output_path.resolve(), flush=True)
    return output_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", required=True, type=int)
    parser.add_argument("--main-thread-id", required=True, type=int)
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--oracle", required=True, type=Path)
    parser.add_argument(
        "--seed",
        required=True,
        type=lambda value: int(value, 0),
    )
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
    parser.add_argument("--timeout", type=float, default=900.0)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    oracle = load_gameplay_mtrand_oracle(
        args.oracle.resolve(),
        seed=args.seed,
        maximum_draws=args.maximum_draws,
    )
    result_path = synchronize_gameplay_mtrand(
        pid=args.pid,
        main_thread_id=args.main_thread_id,
        executable=args.executable.resolve(),
        oracle=oracle,
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
