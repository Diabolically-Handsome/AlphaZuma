"""Synchronize every post-boundary retail global-MTRand wrapper call."""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import replace
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
from tools.popcap_global_mtrand_call_oracle import (
    GlobalMTRandCallOracle,
    load_global_mtrand_call_oracle,
    reconstruct_mtrand_draw_counts,
)
from tools.popcap_thread_runtime_snapshot import (
    capture_thread_runtime_snapshot,
    compute_thread_runtime_delta,
)
from tools.synchronize_popcap_frame_mtrand import (
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
    INVALID_SUSPEND_COUNT,
    MTRAND_STATE_BYTES,
    MTRAND_STATE_WORDS,
    STATUS_WX86_BREAKPOINT,
    STATUS_WX86_SINGLE_STEP,
    _read_u32,
    _thread_ids_for_pid,
)
from tools.trace_popcap_gameplay_mtrand import (
    DEFAULT_GLOBAL_RNG_WRAPPER,
    _board_snapshot,
)
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


TRACE_SCHEMA = "zuma-rl.pc-global-mtrand-oracle-synchronizer"
TRACE_VERSION = 1
READY_SCHEMA = "zuma-rl.pc-global-mtrand-oracle-synchronizer-ready"
READY_VERSION = 2
CREATE_THREAD_DEBUG_EVENT = 2
EXIT_THREAD_DEBUG_EVENT = 4


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _sha256_bytes(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _is_canonical_sha256(value: object) -> bool:
    if (
        not isinstance(value, str)
        or len(value) != 71
        or not value.startswith("sha256:")
        or value != value.lower()
    ):
        return False
    try:
        int(value[7:], 16)
    except ValueError:
        return False
    return True


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


def _publish_ready_after_verified_handoff(
    *,
    main_thread_id: int,
    ready_path: Path,
    ready_payload: dict[str, Any],
    resume_main_thread_on_ready: bool,
    open_main_thread: Any | None = None,
    resume_thread: Any | None = None,
    suspend_thread: Any | None = None,
    close_handle: Any | None = None,
    write_ready: Any | None = None,
    perf_counter_ns: Any | None = None,
) -> tuple[bool, int | None, int]:
    """Publish readiness only after the suspended-thread handoff commits."""

    open_main_thread = open_main_thread or _open_main_thread
    resume_thread = resume_thread or kernel32.ResumeThread
    suspend_thread = suspend_thread or kernel32.SuspendThread
    close_handle = close_handle or kernel32.CloseHandle
    write_ready = write_ready or _write_canonical
    perf_counter_ns = perf_counter_ns or time.perf_counter_ns

    handoff_resumed = False
    previous_suspend_count: int | None = None
    if resume_main_thread_on_ready:
        handoff_thread = open_main_thread(main_thread_id)
        try:
            previous_suspend_count = int(resume_thread(handoff_thread))
            if previous_suspend_count == INVALID_SUSPEND_COUNT:
                raise ctypes.WinError(ctypes.get_last_error())
            if previous_suspend_count != 1:
                if previous_suspend_count > 1:
                    restored_count = int(suspend_thread(handoff_thread))
                    if restored_count == INVALID_SUSPEND_COUNT:
                        raise ctypes.WinError(ctypes.get_last_error())
                raise RuntimeError(
                    "main-thread handoff expected suspend count 1, "
                    f"observed {previous_suspend_count}"
                )
            handoff_resumed = True
        finally:
            close_handle(handoff_thread)

    published_ns = int(perf_counter_ns())
    receipt = {
        **ready_payload,
        "schema": READY_SCHEMA,
        "version": READY_VERSION,
        "handoff_verified": True,
        "handoff_main_thread_resumed": handoff_resumed,
        "handoff_resume_previous_suspend_count": previous_suspend_count,
        "publication_order": "AFTER_VERIFIED_HANDOFF_RESUME",
        "published_perf_counter_ns": published_ns,
    }
    write_ready(ready_path, receipt)
    return handoff_resumed, previous_suspend_count, published_ns


def _restore_debug_registers_in_context(
    context: Any,
    original: tuple[int, int, int, int, int, int],
) -> None:
    (
        context.Dr0,
        context.Dr1,
        context.Dr2,
        context.Dr3,
        context.Dr6,
        context.Dr7,
    ) = original
    context.EFlags |= RESUME_FLAG


def _prepare_global_mtrand_pre_state(
    *,
    process: int,
    live_state: bytes,
    expected_state: bytes,
    observe_boundary_only: bool,
) -> tuple[bool, bool]:
    """Return natural match and whether a synchronization write occurred."""

    live_matches = live_state == expected_state
    if observe_boundary_only:
        return live_matches, False
    correction_applied = False
    try:
        if not live_matches:
            write_memory(
                process,
                DEFAULT_GLOBAL_RNG_STATE,
                expected_state,
            )
            correction_applied = True
        if (
            read_memory(
                process,
                DEFAULT_GLOBAL_RNG_STATE,
                MTRAND_STATE_BYTES,
            )
            != expected_state
        ):
            raise RuntimeError("global sync writeback mismatch")
    except Exception:
        try:
            write_memory(
                process,
                DEFAULT_GLOBAL_RNG_STATE,
                live_state,
            )
        except Exception as rollback_error:
            raise RuntimeError(
                "global sync write and rollback both failed"
            ) from rollback_error
        raise
    return live_matches, correction_applied


def _capture_handoff_rng_state(process: int) -> dict[str, Any]:
    """Read one immutable process-global MTRand handoff snapshot."""

    state = read_memory(
        process,
        DEFAULT_GLOBAL_RNG_STATE,
        MTRAND_STATE_BYTES,
    )
    index = struct.unpack_from(
        "<I",
        state,
        MTRAND_STATE_WORDS * 4,
    )[0]
    return {
        "schema": "zuma-rl.pc-global-mtrand-handoff-state",
        "version": 1,
        "classification": "diagnostic-read-only-process-memory-snapshot",
        "address": DEFAULT_GLOBAL_RNG_STATE,
        "address_hex": f"0x{DEFAULT_GLOBAL_RNG_STATE:08x}",
        "state_bytes": len(state),
        "state_sha256": _sha256_bytes(state),
        "words_sha256": _sha256_bytes(
            state[: MTRAND_STATE_WORDS * 4]
        ),
        "index": index,
        "draw_count": None,
        "captured_perf_counter_ns": time.perf_counter_ns(),
        "process_memory_reads": 1,
        "process_memory_read_bytes": len(state),
        "process_memory_writes": 0,
        "process_memory_mutation": False,
    }


def _wait_for_handoff_rng_readiness(
    *,
    process: int,
    target_words_sha256: str,
    minimum_index: int,
    timeout_seconds: float,
    poll_interval_seconds: float,
    capture_state: Any | None = None,
    perf_counter_ns: Any | None = None,
    sleep: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Wait read-only while the predecessor's main-thread suspend is held."""

    if (
        process <= 0
        or not _is_canonical_sha256(target_words_sha256)
        or isinstance(minimum_index, bool)
        or not 0 <= minimum_index <= MTRAND_STATE_WORDS
        or timeout_seconds <= 0
        or poll_interval_seconds <= 0
        or poll_interval_seconds > timeout_seconds
    ):
        raise ValueError("invalid handoff RNG readiness wait options")
    capture_state = capture_state or _capture_handoff_rng_state
    perf_counter_ns = perf_counter_ns or time.perf_counter_ns
    sleep = sleep or time.sleep

    started_ns = int(perf_counter_ns())
    timeout_ns = max(1, int(timeout_seconds * 1_000_000_000))
    deadline_ns = started_ns + timeout_ns
    initial = dict(capture_state(process))
    final = initial
    read_count = 1

    def classify(observation: dict[str, Any]) -> str | None:
        if observation.get("words_sha256") != target_words_sha256:
            return "BLOCK_MISMATCH"
        if int(observation.get("index", -1)) >= minimum_index:
            return "TARGET_REACHED"
        return None

    status = classify(final)
    while status is None:
        now_ns = int(perf_counter_ns())
        if now_ns >= deadline_ns:
            status = "TIMEOUT"
            break
        remaining_seconds = (deadline_ns - now_ns) / 1_000_000_000
        sleep(min(poll_interval_seconds, remaining_seconds))
        final = dict(capture_state(process))
        read_count += 1
        status = classify(final)

    finished_ns = int(perf_counter_ns())
    receipt = {
        "schema": "zuma-rl.pc-global-mtrand-handoff-readiness-wait",
        "version": 1,
        "classification": (
            "diagnostic-read-only-main-thread-suspended-rng-readiness-wait"
        ),
        "status": status,
        "target_words_sha256": target_words_sha256,
        "minimum_index": minimum_index,
        "timeout_seconds": timeout_seconds,
        "poll_interval_seconds": poll_interval_seconds,
        "target_reached": status == "TARGET_REACHED",
        "main_thread_suspended_during_wait": True,
        "started_perf_counter_ns": started_ns,
        "finished_perf_counter_ns": finished_ns,
        "elapsed_perf_counter_ns": max(0, finished_ns - started_ns),
        "initial_state_sha256": initial["state_sha256"],
        "initial_words_sha256": initial["words_sha256"],
        "initial_index": initial["index"],
        "initial_captured_perf_counter_ns": initial[
            "captured_perf_counter_ns"
        ],
        "initial_draw_count": None,
        "final_state_sha256": final["state_sha256"],
        "final_words_sha256": final["words_sha256"],
        "final_index": final["index"],
        "final_captured_perf_counter_ns": final[
            "captured_perf_counter_ns"
        ],
        "final_draw_count": None,
        "poll_count": read_count - 1,
        "process_memory_reads": read_count,
        "process_memory_read_bytes": read_count * MTRAND_STATE_BYTES,
        "process_memory_writes": 0,
        "process_memory_mutation": False,
    }
    return final, receipt


def synchronize_global_mtrand_calls(
    *,
    pid: int,
    main_thread_id: int,
    executable: Path,
    oracle: GlobalMTRandCallOracle,
    output_path: Path,
    ready_path: Path,
    stop_path: Path,
    address: int = DEFAULT_GLOBAL_RNG_WRAPPER,
    attach_timeout: float = 60.0,
    timeout: float = 900.0,
    resume_main_thread_on_ready: bool = False,
    observe_boundary_only: bool = False,
    observe_boundary_thread_runtime: bool = False,
    observe_handoff_thread_runtime: bool = False,
    observe_handoff_rng_state: bool = False,
    handoff_rng_wait_target_words_sha256: str | None = None,
    handoff_rng_wait_minimum_index: int | None = None,
    handoff_rng_wait_timeout_seconds: float | None = None,
    handoff_rng_wait_poll_interval_seconds: float | None = None,
    maximum_draws: int = 100_000,
) -> Path:
    """Synchronize a suffix or observe its first boundary without writes."""

    if (
        pid <= 0
        or main_thread_id <= 0
        or address <= 0
        or attach_timeout <= 0
        or timeout <= 0
        or maximum_draws <= 0
        or not oracle.entries
    ):
        raise ValueError("invalid global MTRand synchronizer target")
    if observe_boundary_thread_runtime and not observe_boundary_only:
        raise ValueError(
            "boundary thread runtime requires boundary observation mode"
        )
    if (
        observe_handoff_thread_runtime
        and not observe_boundary_thread_runtime
    ):
        raise ValueError(
            "handoff thread runtime requires boundary thread runtime"
        )
    if observe_handoff_rng_state and not observe_boundary_only:
        raise ValueError(
            "handoff RNG state requires boundary observation mode"
        )
    handoff_rng_wait_values = (
        handoff_rng_wait_target_words_sha256,
        handoff_rng_wait_minimum_index,
        handoff_rng_wait_timeout_seconds,
        handoff_rng_wait_poll_interval_seconds,
    )
    handoff_rng_wait_enabled = any(
        value is not None for value in handoff_rng_wait_values
    )
    if handoff_rng_wait_enabled and (
        any(value is None for value in handoff_rng_wait_values)
        or not observe_boundary_only
        or not observe_handoff_rng_state
        or not resume_main_thread_on_ready
        or not _is_canonical_sha256(
            handoff_rng_wait_target_words_sha256
        )
        or isinstance(handoff_rng_wait_minimum_index, bool)
        or not isinstance(handoff_rng_wait_minimum_index, int)
        or not 0
        <= handoff_rng_wait_minimum_index
        <= MTRAND_STATE_WORDS
        or isinstance(handoff_rng_wait_timeout_seconds, bool)
        or not isinstance(handoff_rng_wait_timeout_seconds, (int, float))
        or handoff_rng_wait_timeout_seconds <= 0
        or isinstance(handoff_rng_wait_poll_interval_seconds, bool)
        or not isinstance(
            handoff_rng_wait_poll_interval_seconds,
            (int, float),
        )
        or handoff_rng_wait_poll_interval_seconds <= 0
        or handoff_rng_wait_poll_interval_seconds
        > handoff_rng_wait_timeout_seconds
    ):
        raise ValueError("invalid handoff RNG readiness wait options")
    active_oracle = (
        replace(oracle, entries=oracle.entries[:1])
        if observe_boundary_only
        else oracle
    )
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
    hardware_armed = False
    hardware_original: tuple[int, int, int, int, int, int] | None = None
    hardware_restore_error: str | None = None
    detach_error: str | None = None
    hits: list[dict[str, Any]] = []
    semantic_mismatch: dict[str, Any] | None = None
    unexpected_extra_call: dict[str, Any] | None = None
    awaiting_final_post: dict[str, Any] | None = None
    final_post_verification: dict[str, Any] | None = None
    exited = False
    exit_code: int | None = None
    stop_reason = "timeout"
    failure: str | None = None
    exception_detail: str | None = None
    handoff_main_thread_resumed = False
    handoff_resume_previous_suspend_count: int | None = None
    ready_receipt_published = False
    ready_receipt_published_perf_counter_ns: int | None = None
    attach_event_count = 0
    attach_breakpoint_observed = False
    thread_lifecycle_events: list[dict[str, Any]] = []
    handoff_thread_runtime_snapshot: dict[str, Any] | None = None
    boundary_thread_runtime_snapshot: dict[str, Any] | None = None
    handoff_to_boundary_runtime_delta: dict[str, Any] | None = None
    handoff_rng_state_observation: dict[str, Any] | None = None
    handoff_rng_readiness_wait: dict[str, Any] | None = None
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
            current_image = process_image_path(pid)
            if (
                error in (ERROR_ACCESS_DENIED, ERROR_INVALID_PARAMETER)
                and time.monotonic() < attach_deadline
                and current_image is not None
                and same_windows_path(current_image, executable)
            ):
                if not wait_reported:
                    print(
                        "global_sync_waiting_for_debugger_handoff "
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

        attach_drain_deadline = time.monotonic() + min(
            attach_timeout,
            10.0,
        )
        while time.monotonic() < attach_drain_deadline:
            attach_event = DEBUG_EVENT()
            if not kernel32.WaitForDebugEvent(
                ctypes.byref(attach_event),
                50,
            ):
                error = ctypes.get_last_error()
                if (
                    error == ERROR_SEM_TIMEOUT
                    and attach_breakpoint_observed
                ):
                    break
                if error == ERROR_SEM_TIMEOUT:
                    continue
                raise ctypes.WinError(error)

            attach_status = DBG_CONTINUE
            if attach_event.dwDebugEventCode == EXCEPTION_DEBUG_EVENT:
                attach_code = int(
                    attach_event.u.Exception.
                    ExceptionRecord.ExceptionCode
                )
                if attach_code in (
                    EXCEPTION_BREAKPOINT,
                    STATUS_WX86_BREAKPOINT,
                ):
                    attach_breakpoint_observed = True
                else:
                    attach_status = DBG_EXCEPTION_NOT_HANDLED
            elif (
                attach_event.dwDebugEventCode
                == CREATE_PROCESS_DEBUG_EVENT
            ):
                if observe_boundary_thread_runtime:
                    create_info = attach_event.u.CreateProcessInfo
                    start_address = int(create_info.lpStartAddress or 0)
                    thread_lifecycle_events.append(
                        {
                            "phase": "attach_drain",
                            "event": "create_process_main_thread",
                            "thread_id": int(attach_event.dwThreadId),
                            "start_address": start_address,
                            "start_address_hex": (
                                f"0x{start_address:016x}"
                            ),
                            "thread_local_base": int(
                                create_info.lpThreadLocalBase or 0
                            ),
                        }
                    )
                file_handle = (
                    attach_event.u.CreateProcessInfo.hFile
                )
                if file_handle:
                    kernel32.CloseHandle(file_handle)
            elif (
                attach_event.dwDebugEventCode
                == CREATE_THREAD_DEBUG_EVENT
            ):
                if observe_boundary_thread_runtime:
                    create_info = attach_event.u.CreateThread
                    start_address = int(create_info.lpStartAddress or 0)
                    thread_lifecycle_events.append(
                        {
                            "phase": "attach_drain",
                            "event": "create_thread",
                            "thread_id": int(attach_event.dwThreadId),
                            "start_address": start_address,
                            "start_address_hex": (
                                f"0x{start_address:016x}"
                            ),
                            "thread_local_base": int(
                                create_info.lpThreadLocalBase or 0
                            ),
                        }
                    )
            elif (
                attach_event.dwDebugEventCode
                == EXIT_THREAD_DEBUG_EVENT
            ):
                if observe_boundary_thread_runtime:
                    thread_lifecycle_events.append(
                        {
                            "phase": "attach_drain",
                            "event": "exit_thread",
                            "thread_id": int(attach_event.dwThreadId),
                            "exit_code": int(
                                attach_event.u.ExitThread.dwExitCode
                            ),
                        }
                    )
            elif attach_event.dwDebugEventCode == LOAD_DLL_DEBUG_EVENT:
                file_handle = attach_event.u.LoadDll.hFile
                if file_handle:
                    kernel32.CloseHandle(file_handle)
            elif attach_event.dwDebugEventCode == EXIT_PROCESS_DEBUG_EVENT:
                exited = True
                exit_code = int(
                    attach_event.u.ExitProcess.dwExitCode
                )

            if not kernel32.ContinueDebugEvent(
                attach_event.dwProcessId,
                attach_event.dwThreadId,
                attach_status,
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            attach_event_count += 1
            if exited:
                raise RuntimeError(
                    "target exited during debugger handoff"
                )
        if not attach_breakpoint_observed:
            raise RuntimeError(
                "debugger attach breakpoint was not observed"
            )

        hardware_original = _arm_hardware_breakpoint(
            main_thread_id=main_thread_id,
            address=address,
        )
        hardware_armed = True
        if handoff_rng_wait_enabled:
            assert handoff_rng_wait_target_words_sha256 is not None
            assert handoff_rng_wait_minimum_index is not None
            assert handoff_rng_wait_timeout_seconds is not None
            assert handoff_rng_wait_poll_interval_seconds is not None
            (
                handoff_rng_state_observation,
                handoff_rng_readiness_wait,
            ) = _wait_for_handoff_rng_readiness(
                process=process,
                target_words_sha256=(
                    handoff_rng_wait_target_words_sha256
                ),
                minimum_index=handoff_rng_wait_minimum_index,
                timeout_seconds=float(
                    handoff_rng_wait_timeout_seconds
                ),
                poll_interval_seconds=float(
                    handoff_rng_wait_poll_interval_seconds
                ),
            )
        elif observe_handoff_rng_state:
            handoff_rng_state_observation = _capture_handoff_rng_state(
                process
            )
        if observe_handoff_thread_runtime:
            handoff_thread_runtime_snapshot = (
                capture_thread_runtime_snapshot(
                    process_id=pid,
                    main_thread_id=main_thread_id,
                    phase="global_mtrand_handoff_pre_resume",
                )
            )
        (
            handoff_main_thread_resumed,
            handoff_resume_previous_suspend_count,
            ready_receipt_published_perf_counter_ns,
        ) = _publish_ready_after_verified_handoff(
            main_thread_id=main_thread_id,
            ready_path=ready_path,
            ready_payload={
                "process_id": pid,
                "main_thread_id": main_thread_id,
                "address": address,
                "expected_call_count": len(active_oracle.entries),
                "oracle_semantic_sha256": active_oracle.semantic_sha256,
                "mode": (
                    "observe_boundary_only"
                    if observe_boundary_only
                    else "synchronize"
                ),
                "resume_main_thread_on_ready": (
                    resume_main_thread_on_ready
                ),
                "observe_boundary_thread_runtime": (
                    observe_boundary_thread_runtime
                ),
                "observe_handoff_thread_runtime": (
                    observe_handoff_thread_runtime
                ),
                "observe_handoff_rng_state": observe_handoff_rng_state,
                "handoff_rng_state_sha256": (
                    handoff_rng_state_observation["state_sha256"]
                    if handoff_rng_state_observation is not None
                    else None
                ),
                "handoff_rng_words_sha256": (
                    handoff_rng_state_observation["words_sha256"]
                    if handoff_rng_state_observation is not None
                    else None
                ),
                "handoff_rng_index": (
                    handoff_rng_state_observation["index"]
                    if handoff_rng_state_observation is not None
                    else None
                ),
                "handoff_rng_captured_perf_counter_ns": (
                    handoff_rng_state_observation[
                        "captured_perf_counter_ns"
                    ]
                    if handoff_rng_state_observation is not None
                    else None
                ),
                "handoff_rng_readiness_wait_enabled": (
                    handoff_rng_wait_enabled
                ),
                "handoff_rng_readiness_wait_status": (
                    handoff_rng_readiness_wait["status"]
                    if handoff_rng_readiness_wait is not None
                    else None
                ),
                "handoff_rng_readiness_wait_target_words_sha256": (
                    handoff_rng_wait_target_words_sha256
                ),
                "handoff_rng_readiness_wait_minimum_index": (
                    handoff_rng_wait_minimum_index
                ),
                "handoff_rng_readiness_wait_timeout_seconds": (
                    handoff_rng_wait_timeout_seconds
                ),
                "handoff_rng_readiness_wait_poll_interval_seconds": (
                    handoff_rng_wait_poll_interval_seconds
                ),
                "handoff_rng_readiness_wait_target_reached": (
                    handoff_rng_readiness_wait["target_reached"]
                    if handoff_rng_readiness_wait is not None
                    else None
                ),
                "handoff_rng_readiness_wait_final_state_sha256": (
                    handoff_rng_readiness_wait["final_state_sha256"]
                    if handoff_rng_readiness_wait is not None
                    else None
                ),
                "handoff_rng_readiness_wait_final_words_sha256": (
                    handoff_rng_readiness_wait["final_words_sha256"]
                    if handoff_rng_readiness_wait is not None
                    else None
                ),
                "handoff_rng_readiness_wait_final_index": (
                    handoff_rng_readiness_wait["final_index"]
                    if handoff_rng_readiness_wait is not None
                    else None
                ),
                "handoff_rng_readiness_wait_final_captured_perf_counter_ns": (
                    handoff_rng_readiness_wait[
                        "final_captured_perf_counter_ns"
                    ]
                    if handoff_rng_readiness_wait is not None
                    else None
                ),
            },
            resume_main_thread_on_ready=resume_main_thread_on_ready,
        )
        ready_receipt_published = True
        print(
            f"global_sync_armed pid={pid} tid={main_thread_id} "
            f"address=0x{address:08X} "
            f"expected_calls={len(active_oracle.entries)} "
            "handoff_resumed="
            f"{handoff_main_thread_resumed}",
            flush=True,
        )

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if stop_path.exists():
                failure = "external_stop_requested"
                stop_reason = "external_stop_requested"
                break
            event = DEBUG_EVENT()
            if not kernel32.WaitForDebugEvent(ctypes.byref(event), 100):
                error = ctypes.get_last_error()
                if error == ERROR_SEM_TIMEOUT:
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
                ):
                    thread = _open_main_thread(main_thread_id)
                    try:
                        context = _read_context(thread)
                        if awaiting_final_post is not None:
                            entry = awaiting_final_post["entry"]
                            return_address = int(
                                awaiting_final_post["return_address"]
                            )
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
                            observed_return = int(context.Eip)
                            observed_output = int(context.Eax)
                            return_address_matches = (
                                observed_return == return_address
                            )
                            output_matches = observed_output == entry.output
                            post_state_matches = (
                                live_state == entry.post_payload
                            )
                            post_matches = (
                                return_address_matches
                                and output_matches
                                and post_state_matches
                            )
                            source_exact = (
                                bool(hits)
                                and hits[-1].get("contract_match") is True
                                and hits[-1].get("live_pre_state_match")
                                is True
                                and output_matches
                                and post_state_matches
                            )
                            final_post_verification = {
                                "source_order": entry.source_order,
                                "order": entry.order,
                                "expected_return_address": (
                                    return_address
                                ),
                                "observed_return_address": (
                                    observed_return
                                ),
                                "expected_output": entry.output,
                                "observed_output": observed_output,
                                "expected_post_index": (
                                    entry.post_index
                                ),
                                "observed_post_index": live_index,
                                "expected_post_state_sha256": (
                                    entry.post_state_sha256
                                ),
                                "observed_post_state_sha256": (
                                    _sha256_bytes(live_state)
                                ),
                                "return_address_match": (
                                    return_address_matches
                                ),
                                "expected_output_match": output_matches,
                                "expected_post_state_match": (
                                    post_state_matches
                                ),
                                "source_exact": source_exact,
                                "verified": (
                                    return_address_matches
                                    if observe_boundary_only
                                    else post_matches
                                ),
                            }
                            if hardware_original is not None:
                                _restore_debug_registers_in_context(
                                    context,
                                    hardware_original,
                                )
                                hardware_armed = False
                            _write_context(thread, context)
                            awaiting_final_post = None
                            if (
                                return_address_matches
                                if observe_boundary_only
                                else post_matches
                            ):
                                stop_reason = (
                                    "boundary_observed"
                                    if observe_boundary_only
                                    else "oracle_complete"
                                )
                            else:
                                failure = (
                                    "boundary_return_not_observed"
                                    if observe_boundary_only
                                    else "final_global_call_post_mismatch"
                                )
                                stop_reason = (
                                    "observer_failure"
                                    if observe_boundary_only
                                    else "semantic_failure"
                                )
                            stop_after_continue = True
                        is_target = (
                            final_post_verification is None
                            and (
                                bool(int(context.Dr6) & 0x1)
                                or exception_address == address
                                or int(context.Eip) == address
                            )
                        )
                        if is_target:
                            board = _board_snapshot(process)
                            caller = _read_u32(
                                process,
                                int(context.Esp),
                            )
                            if len(hits) >= len(active_oracle.entries):
                                unexpected_extra_call = {
                                    "order": len(hits),
                                    "caller": caller,
                                    **board,
                                }
                                failure = "unexpected_extra_global_call"
                                stop_reason = "semantic_failure"
                                if hardware_original is not None:
                                    _restore_debug_registers_in_context(
                                        context,
                                        hardware_original,
                                    )
                                    hardware_armed = False
                                _write_context(thread, context)
                                stop_after_continue = True
                            else:
                                entry = active_oracle.entries[len(hits)]
                                observed_contract = {
                                    "framework_update": int(
                                        board["framework_update"]
                                    ),
                                    "native_game_time": board[
                                        "native_game_time"
                                    ],
                                    "score": board["score"],
                                    "score_target": board[
                                        "score_target"
                                    ],
                                    "caller": caller,
                                }
                                expected_contract = {
                                    "framework_update": (
                                        entry.framework_update
                                    ),
                                    "native_game_time": (
                                        entry.native_game_time
                                    ),
                                    "score": entry.score,
                                    "score_target": entry.score_target,
                                    "caller": entry.caller,
                                }
                                contract_match = (
                                    observed_contract == expected_contract
                                )
                                if not contract_match:
                                    semantic_mismatch = {
                                        "order": entry.order,
                                        "source_order": (
                                            entry.source_order
                                        ),
                                        "observed": observed_contract,
                                        "expected": expected_contract,
                                    }
                                if (
                                    not contract_match
                                    and not observe_boundary_only
                                ):
                                    failure = (
                                        "global_call_contract_mismatch"
                                    )
                                    stop_reason = "semantic_failure"
                                    if hardware_original is not None:
                                        _restore_debug_registers_in_context(
                                            context,
                                            hardware_original,
                                        )
                                        hardware_armed = False
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
                                    if observe_boundary_thread_runtime:
                                        boundary_thread_runtime_snapshot = (
                                            capture_thread_runtime_snapshot(
                                                process_id=pid,
                                                main_thread_id=(
                                                    main_thread_id
                                                ),
                                                phase=(
                                                    "global_mtrand_"
                                                    "boundary_pre_call"
                                                ),
                                            )
                                        )
                                        if (
                                            handoff_thread_runtime_snapshot
                                            is not None
                                        ):
                                            handoff_to_boundary_runtime_delta = (
                                                compute_thread_runtime_delta(
                                                    handoff_thread_runtime_snapshot,
                                                    (
                                                        boundary_thread_runtime_snapshot
                                                    ),
                                                )
                                            )
                                    (
                                        live_pre_state_match,
                                        correction_applied,
                                    ) = _prepare_global_mtrand_pre_state(
                                        process=process,
                                        live_state=live_state,
                                        expected_state=entry.pre_payload,
                                        observe_boundary_only=(
                                            observe_boundary_only
                                        ),
                                    )

                                    completed = (
                                        len(hits) + 1
                                        == len(active_oracle.entries)
                                    )
                                    if (
                                        completed
                                        and hardware_original is not None
                                    ):
                                        context.Dr0 = caller
                                        context.Dr6 = 0
                                        context.Dr7 &= ~0x000F0003
                                        context.Dr7 |= 0x1
                                        context.EFlags |= RESUME_FLAG
                                        awaiting_final_post = {
                                            "entry": entry,
                                            "return_address": caller,
                                        }
                                    else:
                                        context.Dr6 = 0
                                        context.EFlags |= RESUME_FLAG
                                    _write_context(thread, context)
                                    hits.append(
                                        {
                                            "order": entry.order,
                                            "source_order": (
                                                entry.source_order
                                            ),
                                            "framework_update": (
                                                entry.framework_update
                                            ),
                                            "observed_framework_update": (
                                                observed_contract[
                                                    "framework_update"
                                                ]
                                            ),
                                            "native_game_time": (
                                                entry.native_game_time
                                            ),
                                            "observed_native_game_time": (
                                                observed_contract[
                                                    "native_game_time"
                                                ]
                                            ),
                                            "score": entry.score,
                                            "observed_score": (
                                                observed_contract["score"]
                                            ),
                                            "score_target": (
                                                entry.score_target
                                            ),
                                            "observed_score_target": (
                                                observed_contract[
                                                    "score_target"
                                                ]
                                            ),
                                            "caller": entry.caller,
                                            "caller_hex": (
                                                f"0x{entry.caller:08x}"
                                            ),
                                            "observed_caller": caller,
                                            "observed_caller_hex": (
                                                f"0x{caller:08x}"
                                            ),
                                            "contract_match": (
                                                contract_match
                                            ),
                                            "output": entry.output,
                                            "live_pre_index": live_index,
                                            "live_pre_state_sha256": (
                                                _sha256_bytes(live_state)
                                            ),
                                            "live_pre_state_match": (
                                                live_pre_state_match
                                            ),
                                            "restored_pre_draw_count": (
                                                entry.pre_draw_count
                                            ),
                                            "restored_pre_index": (
                                                entry.pre_index
                                            ),
                                            "restored_pre_state_sha256": (
                                                entry.pre_state_sha256
                                            ),
                                            "expected_post_draw_count": (
                                                entry.post_draw_count
                                            ),
                                            "expected_post_index": (
                                                entry.post_index
                                            ),
                                            "expected_post_state_sha256": (
                                                entry.post_state_sha256
                                            ),
                                            "bytes_written": (
                                                MTRAND_STATE_BYTES
                                                if correction_applied
                                                else 0
                                            ),
                                            "changed": correction_applied,
                                            "state_correction_applied": (
                                                correction_applied
                                            ),
                                        }
                                    )
                                    if len(hits) % 250 == 0:
                                        print(
                                            "global_sync_progress "
                                            f"hits={len(hits)}/"
                                            f"{len(active_oracle.entries)} "
                                            f"update="
                                            f"{entry.framework_update}",
                                            flush=True,
                                        )
                                    if completed:
                                        print(
                                            (
                                                "global_observer_awaiting_"
                                                "boundary_return "
                                                if observe_boundary_only
                                                else (
                                                    "global_sync_awaiting_"
                                                    "final_post "
                                                )
                                            )
                                            + f"return=0x{caller:08X} "
                                            + f"output={entry.output}",
                                            flush=True,
                                        )
                        else:
                            if final_post_verification is None:
                                status = DBG_EXCEPTION_NOT_HANDLED
                    finally:
                        kernel32.CloseHandle(thread)
                elif code in (
                    EXCEPTION_BREAKPOINT,
                    STATUS_WX86_BREAKPOINT,
                ):
                    status = DBG_CONTINUE
                elif code == MICROSOFT_CPP_EXCEPTION:
                    status = DBG_EXCEPTION_NOT_HANDLED
                else:
                    status = DBG_EXCEPTION_NOT_HANDLED
            elif event.dwDebugEventCode == CREATE_PROCESS_DEBUG_EVENT:
                if observe_boundary_thread_runtime:
                    create_info = event.u.CreateProcessInfo
                    start_address = int(create_info.lpStartAddress or 0)
                    thread_lifecycle_events.append(
                        {
                            "phase": "active_observation",
                            "event": "create_process_main_thread",
                            "thread_id": int(event.dwThreadId),
                            "start_address": start_address,
                            "start_address_hex": (
                                f"0x{start_address:016x}"
                            ),
                            "thread_local_base": int(
                                create_info.lpThreadLocalBase or 0
                            ),
                        }
                    )
                file_handle = event.u.CreateProcessInfo.hFile
                if file_handle:
                    kernel32.CloseHandle(file_handle)
            elif event.dwDebugEventCode == CREATE_THREAD_DEBUG_EVENT:
                if observe_boundary_thread_runtime:
                    create_info = event.u.CreateThread
                    start_address = int(create_info.lpStartAddress or 0)
                    thread_lifecycle_events.append(
                        {
                            "phase": "active_observation",
                            "event": "create_thread",
                            "thread_id": int(event.dwThreadId),
                            "start_address": start_address,
                            "start_address_hex": (
                                f"0x{start_address:016x}"
                            ),
                            "thread_local_base": int(
                                create_info.lpThreadLocalBase or 0
                            ),
                        }
                    )
            elif event.dwDebugEventCode == EXIT_THREAD_DEBUG_EVENT:
                if observe_boundary_thread_runtime:
                    thread_lifecycle_events.append(
                        {
                            "phase": "active_observation",
                            "event": "exit_thread",
                            "thread_id": int(event.dwThreadId),
                            "exit_code": int(
                                event.u.ExitThread.dwExitCode
                            ),
                        }
                    )
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
        if attached and not exited:
            if not kernel32.DebugActiveProcessStop(pid):
                detach_error = (
                    f"OSError: {ctypes.WinError(ctypes.get_last_error())}"
                )

    missing_count = len(active_oracle.entries) - len(hits)
    state_correction_count = sum(
        1 for hit in hits if bool(hit["changed"])
    )
    bytes_written_total = sum(
        int(hit["bytes_written"]) for hit in hits
    )
    draw_count_reconstruction: dict[str, Any] = {
        "status": "NOT_REQUESTED",
        "maximum_draws": maximum_draws,
    }
    if observe_boundary_only:
        if hits and final_post_verification is not None:
            state_hashes = {
                str(hits[0]["live_pre_state_sha256"]),
                str(
                    final_post_verification[
                        "observed_post_state_sha256"
                    ]
                ),
            }
            if handoff_rng_state_observation is not None:
                state_hashes.add(
                    str(handoff_rng_state_observation["state_sha256"])
                )
            if handoff_rng_readiness_wait is not None:
                state_hashes.update(
                    {
                        str(
                            handoff_rng_readiness_wait[
                                "initial_state_sha256"
                            ]
                        ),
                        str(
                            handoff_rng_readiness_wait[
                                "final_state_sha256"
                            ]
                        ),
                    }
                )
            try:
                draw_counts = reconstruct_mtrand_draw_counts(
                    seed=active_oracle.seed,
                    state_sha256=state_hashes,
                    maximum_draws=maximum_draws,
                )
                hits[0]["live_pre_draw_count"] = draw_counts[
                    str(hits[0]["live_pre_state_sha256"])
                ]
                final_post_verification["observed_post_draw_count"] = (
                    draw_counts[
                        str(
                            final_post_verification[
                                "observed_post_state_sha256"
                            ]
                        )
                    ]
                )
                if handoff_rng_state_observation is not None:
                    handoff_rng_state_observation["draw_count"] = (
                        draw_counts[
                            str(
                                handoff_rng_state_observation[
                                    "state_sha256"
                                ]
                            )
                        ]
                    )
                if handoff_rng_readiness_wait is not None:
                    handoff_rng_readiness_wait["initial_draw_count"] = (
                        draw_counts[
                            str(
                                handoff_rng_readiness_wait[
                                    "initial_state_sha256"
                                ]
                            )
                        ]
                    )
                    handoff_rng_readiness_wait["final_draw_count"] = (
                        draw_counts[
                            str(
                                handoff_rng_readiness_wait[
                                    "final_state_sha256"
                                ]
                            )
                        ]
                    )
                draw_count_reconstruction = {
                    "status": "PASS",
                    "maximum_draws": maximum_draws,
                    "state_count": len(state_hashes),
                }
            except ValueError as error:
                draw_count_reconstruction = {
                    "status": "FAIL",
                    "maximum_draws": maximum_draws,
                    "failure": f"{type(error).__name__}: {error}",
                }
                if failure is None:
                    failure = "draw_count_reconstruction_failed"
        else:
            draw_count_reconstruction = {
                "status": "INCOMPLETE",
                "maximum_draws": maximum_draws,
            }
    if failure is None and missing_count:
        failure = "missing_global_calls"
    expected_stop_reason = (
        "boundary_observed"
        if observe_boundary_only
        else "oracle_complete"
    )
    if failure is None and stop_reason != expected_stop_reason:
        failure = "oracle_did_not_complete"
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
            "diagnostic-read-only-global-rng-boundary-observation"
            if observe_boundary_only
            else "diagnostic-ephemeral-global-rng-synchronization"
        ),
        "mode": (
            "observe_boundary_only"
            if observe_boundary_only
            else "synchronize"
        ),
        "process_id": pid,
        "main_thread_id": main_thread_id,
        "runtime_executable": str(executable.resolve()),
        "runtime_executable_sha256": _sha256_path(executable),
        "address": address,
        "address_hex": f"0x{address:08x}",
        "global_mtrand_address": DEFAULT_GLOBAL_RNG_STATE,
        "oracle": {
            "source_path": str(active_oracle.source_path),
            "source_sha256": active_oracle.source_sha256,
            "source_process_id": active_oracle.source_process_id,
            "source_main_thread_id": active_oracle.source_main_thread_id,
            "seed": active_oracle.seed,
            "start_after_update": active_oracle.start_after_update,
            "end_at_update": active_oracle.end_at_update,
            "entry_count": len(active_oracle.entries),
            "semantic_sha256": active_oracle.semantic_sha256,
        },
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "started_perf_counter_ns": started_ns,
        "finished_perf_counter_ns": time.perf_counter_ns(),
        "stop_reason": stop_reason,
        "exited": exited,
        "exit_code": exit_code,
        "expected_hit_count": len(active_oracle.entries),
        "hit_count": len(hits),
        "missing_hit_count": missing_count,
        "semantic_mismatch": semantic_mismatch,
        "unexpected_extra_call": unexpected_extra_call,
        "state_correction_count": state_correction_count,
        "bytes_written_total": bytes_written_total,
        "process_memory_writes": bytes_written_total,
        "process_memory_mutation": bool(bytes_written_total),
        "process_context_mutation": True,
        "persistent_file_modified": False,
        "hardware_breakpoint_restored": not hardware_armed,
        "hardware_breakpoint_restore_error": hardware_restore_error,
        "debugger_detach_error": detach_error,
        "resume_main_thread_on_ready": resume_main_thread_on_ready,
        "observe_handoff_thread_runtime": observe_handoff_thread_runtime,
        "observe_handoff_rng_state": observe_handoff_rng_state,
        "handoff_rng_state_observation": handoff_rng_state_observation,
        "handoff_rng_readiness_wait_enabled": handoff_rng_wait_enabled,
        "handoff_rng_readiness_wait": handoff_rng_readiness_wait,
        "handoff_main_thread_resumed": handoff_main_thread_resumed,
        "handoff_resume_previous_suspend_count": (
            handoff_resume_previous_suspend_count
        ),
        "ready_receipt_schema": READY_SCHEMA,
        "ready_receipt_version": READY_VERSION,
        "ready_receipt_published": ready_receipt_published,
        "ready_receipt_publication_order": (
            "AFTER_VERIFIED_HANDOFF_RESUME"
            if ready_receipt_published
            else None
        ),
        "ready_receipt_published_perf_counter_ns": (
            ready_receipt_published_perf_counter_ns
        ),
        "attach_event_count": attach_event_count,
        "attach_breakpoint_observed": attach_breakpoint_observed,
        "final_post_verification": final_post_verification,
        "draw_count_reconstruction": draw_count_reconstruction,
        "natural_boundary_source_exact": (
            final_post_verification.get("source_exact")
            if isinstance(final_post_verification, dict)
            else None
        ),
        "thread_runtime_observation": (
            {
                "schema": (
                    "zuma-rl.pc-global-mtrand-boundary-"
                    "thread-runtime-observation"
                ),
                "version": 1,
                "classification": (
                    "diagnostic-read-only-os-thread-metadata-"
                    "captured-after-boundary-arrival"
                ),
                "enabled": True,
                "process_memory_reads": 0,
                "process_memory_writes": 0,
                "process_context_reads": 0,
                "process_context_writes": 0,
                "lifecycle_event_count": len(thread_lifecycle_events),
                "lifecycle_events": thread_lifecycle_events,
                "handoff_snapshot_enabled": (
                    observe_handoff_thread_runtime
                ),
                "handoff_snapshot": handoff_thread_runtime_snapshot,
                "boundary_snapshot": (
                    boundary_thread_runtime_snapshot
                ),
                "handoff_to_boundary_delta": (
                    handoff_to_boundary_runtime_delta
                ),
            }
            if observe_boundary_thread_runtime
            else None
        ),
        "hits": hits,
    }
    _write_canonical(output_path, result)
    print(
        f"global_sync_result status={status} hits={len(hits)}/"
        f"{len(active_oracle.entries)} stop_reason={stop_reason}",
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
    parser.add_argument("--start-after-update", required=True, type=int)
    parser.add_argument(
        "--end-at-update",
        type=int,
        help=(
            "Verify only the contiguous oracle prefix through this "
            "framework update, then detach after its final post-state."
        ),
    )
    parser.add_argument("--maximum-draws", type=int, default=100_000)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ready", required=True, type=Path)
    parser.add_argument("--stop", required=True, type=Path)
    parser.add_argument(
        "--address",
        type=lambda value: int(value, 0),
        default=DEFAULT_GLOBAL_RNG_WRAPPER,
    )
    parser.add_argument("--attach-timeout", type=float, default=60.0)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument(
        "--resume-main-thread-on-ready",
        action="store_true",
        help=(
            "Resume the one explicit suspend count left by the predecessor "
            "debugger after this synchronizer is armed."
        ),
    )
    parser.add_argument(
        "--observe-boundary-only",
        action="store_true",
        help=(
            "observe only the first selected post-handoff wrapper call and "
            "its return without writing process memory"
        ),
    )
    parser.add_argument(
        "--observe-boundary-thread-runtime",
        action="store_true",
        help=(
            "after the selected boundary is reached, capture read-only OS "
            "thread lifecycle and runtime metadata"
        ),
    )
    parser.add_argument(
        "--observe-handoff-thread-runtime",
        action="store_true",
        help=(
            "capture a second read-only OS thread-runtime snapshot before "
            "the handoff resume and retain boundary-minus-handoff deltas; "
            "requires --observe-boundary-thread-runtime"
        ),
    )
    parser.add_argument(
        "--observe-handoff-rng-state",
        action="store_true",
        help=(
            "capture the process-global MTRand state once before the "
            "verified handoff resume and reconstruct its seeded draw count; "
            "requires --observe-boundary-only"
        ),
    )
    parser.add_argument(
        "--handoff-rng-wait-target-words-sha256",
        help=(
            "diagnostic only: while the inherited main-thread suspension is "
            "held, wait for this exact 624-word MTRand block"
        ),
    )
    parser.add_argument(
        "--handoff-rng-wait-min-index",
        dest="handoff_rng_wait_minimum_index",
        type=int,
        help="minimum natural MTRand index required before handoff resume",
    )
    parser.add_argument(
        "--handoff-rng-wait-timeout-seconds",
        type=float,
        help="finite maximum duration of the read-only handoff wait",
    )
    parser.add_argument(
        "--handoff-rng-wait-poll-interval-seconds",
        type=float,
        help="read-only MTRand polling interval during the handoff wait",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    oracle = load_global_mtrand_call_oracle(
        args.oracle.resolve(),
        seed=args.seed,
        start_after_update=args.start_after_update,
        end_at_update=args.end_at_update,
        maximum_draws=args.maximum_draws,
    )
    result_path = synchronize_global_mtrand_calls(
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
        resume_main_thread_on_ready=args.resume_main_thread_on_ready,
        observe_boundary_only=args.observe_boundary_only,
        observe_boundary_thread_runtime=(
            args.observe_boundary_thread_runtime
        ),
        observe_handoff_thread_runtime=(
            args.observe_handoff_thread_runtime
        ),
        observe_handoff_rng_state=args.observe_handoff_rng_state,
        handoff_rng_wait_target_words_sha256=(
            args.handoff_rng_wait_target_words_sha256
        ),
        handoff_rng_wait_minimum_index=(
            args.handoff_rng_wait_minimum_index
        ),
        handoff_rng_wait_timeout_seconds=(
            args.handoff_rng_wait_timeout_seconds
        ),
        handoff_rng_wait_poll_interval_seconds=(
            args.handoff_rng_wait_poll_interval_seconds
        ),
        maximum_draws=args.maximum_draws,
    )
    result = json.loads(result_path.read_text(encoding="ascii"))
    return 0 if result.get("status") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
