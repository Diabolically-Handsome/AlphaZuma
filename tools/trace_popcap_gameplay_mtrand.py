"""Trace the retail main-thread gameplay MTRand return with a HW breakpoint."""

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
from typing import Any, Iterable, Mapping

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tools.collect_pc_memory_probe import (
    ACTIVE_BOARD_OFFSET,
    BOARD_SCORE_OFFSET,
    BOARD_SCORE_TARGET_OFFSET,
)
from tools.launch_popcap_replay import process_image_path, same_windows_path
from tools.popcap_global_mtrand_restore import (
    GlobalMTRandRestoreState,
    load_source_bound_global_mtrand_restore_state,
)
from tools.popcap_thread_crt_restore import (
    ThreadCrtRestoreState,
    load_source_bound_thread_crt_restore_state,
)
from tools.synchronize_popcap_frame_mtrand import (
    DEFAULT_BOARD_UPDATE_OFFSET,
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
    G_SEXY_APP_BASE_ADDRESS,
    MTRAND_STATE_BYTES,
    MTRAND_STATE_WORDS,
    STATUS_WX86_BREAKPOINT,
    STATUS_WX86_SINGLE_STEP,
    _read_i32,
    _read_u32,
    _thread_crt_rng_state,
    _thread_ids_for_pid,
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
from zuma_rl.pc_memory_trajectory import (
    BOARD_NATIVE_GAME_TIME_OFFSET,
    G_CURVE_PLAN_EXHAUSTED_ADDRESS,
)
from zuma_rl.revenge_core import PopCapMTRandom


TRACE_SCHEMA = "zuma-rl.pc-gameplay-mtrand-call-trace"
TRACE_VERSION = 1
DEFAULT_GLOBAL_RNG_WRAPPER = 0x00617490
DEFAULT_BOARD_GLOBAL_PRECALL = 0x0065B81C
BOARD_GLOBAL_PRECALL_RETURN = 0x0065B821
BOARD_GLOBAL_PRECALL_INSTRUCTION = bytes.fromhex("e86fbcfbff")
BOARD_GLOBAL_PRECALL_AND_RESET_KIND = "board_global_precall_and_reset"
BOARD_GLOBAL_PRECALL_AND_TARGET_WRITE_KIND = (
    "board_global_precall_and_target_write"
)
BOARD_GLOBAL_PRECALL_AND_NATURAL_LOSS_TARGET_WRITE_KIND = (
    "board_global_precall_and_natural_loss_target_write"
)
BOARD_TARGET_WRITE_KINDS = frozenset(
    {
        BOARD_GLOBAL_PRECALL_AND_TARGET_WRITE_KIND,
        BOARD_GLOBAL_PRECALL_AND_NATURAL_LOSS_TARGET_WRITE_KIND,
    }
)
NATURAL_LOSS_TARGET_CLEAR_WRITER_ADDRESS = 0x00412EED
NATURAL_LOSS_TARGET_CLEAR_WRITER_INSTRUCTION = bytes.fromhex(
    "899f08010000"
)
NATURAL_LOSS_TARGET_CLEAR_POST_EIP = 0x00412EF3
NATURAL_LOSS_TARGET_POSITIVE_WRITER_ADDRESS = 0x00411B8D
NATURAL_LOSS_TARGET_POSITIVE_WRITER_INSTRUCTION = bytes.fromhex(
    "899608010000"
)
NATURAL_LOSS_TARGET_POSITIVE_POST_EIP = 0x00411B93
NATURAL_LOSS_TARGET_WRITE_MAX_INTERVAL_NS = 5_000_000
DEFAULT_BOARD_RESET_ENTRY = 0x00419A90
BOARD_RESET_ENTRY_INSTRUCTION = bytes.fromhex(
    "558bec6aff68285d940064a10000000050"
    "83ec205357a110509e0033c5508d45"
)
BOARD_RESET_NATIVE_CLOCK_ZERO_ADDRESS = 0x00419BAF
BOARD_RESET_NATIVE_CLOCK_ZERO_INSTRUCTION = bytes.fromhex(
    "c786c80e000000000000"
)
BOARD_RESET_RETURN_ADDRESS = 0x00419E24
BOARD_RESET_RETURN_INSTRUCTION = bytes.fromhex("c20800")
BOARD_RESET_DIRECT_CALL_SITES = (
    0x004163DD,
    0x00416485,
    0x00419F77,
    0x0041E627,
    0x00420E03,
    0x00422370,
    0x00423A64,
    0x004242F5,
    0x0043352C,
    0x00433917,
    0x004368F4,
    0x004382CC,
    0x00438458,
    0x00438BFE,
)
BOARD_RESET_DIRECT_RETURN_ADDRESSES = frozenset(
    address + 5 for address in BOARD_RESET_DIRECT_CALL_SITES
)
BOARD_SOURCE_BREAKPOINT_KINDS = frozenset(
    {
        "board_global_precall",
        BOARD_GLOBAL_PRECALL_AND_RESET_KIND,
        *BOARD_TARGET_WRITE_KINDS,
    }
)
BREAKPOINT_KINDS = (
    "gameplay_return",
    "global_wrapper_entry",
    "board_global_precall",
    BOARD_GLOBAL_PRECALL_AND_RESET_KIND,
    *BOARD_TARGET_WRITE_KINDS,
)
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


def _read_u32(process: int, address: int) -> int:
    return struct.unpack("<I", read_memory(process, address, 4))[0]


def _default_breakpoint_address(breakpoint_kind: str) -> int:
    if breakpoint_kind == "global_wrapper_entry":
        return DEFAULT_GLOBAL_RNG_WRAPPER
    if breakpoint_kind in BOARD_SOURCE_BREAKPOINT_KINDS:
        return DEFAULT_BOARD_GLOBAL_PRECALL
    if breakpoint_kind == "gameplay_return":
        return DEFAULT_GAMEPLAY_RNG_RETURN
    raise ValueError("MTRand breakpoint kind is invalid")


def _is_source_board_anchor_candidate(board: dict[str, Any]) -> bool:
    values = (
        board.get("framework_update"),
        board.get("board_address"),
        board.get("native_game_time"),
        board.get("score"),
        board.get("score_target"),
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in values
    ):
        return False
    update, address, native_time, score, score_target = values
    return (
        update >= 0
        and address > 0
        and native_time >= 0
        and score >= 0
        and score_target > 0
        and score < score_target
        and board.get("board_snapshot_error") is None
    )


def _is_natural_loss_source_anchor_candidate(
    board: dict[str, Any],
) -> bool:
    return bool(
        _is_source_board_anchor_candidate(board)
        and board.get("native_game_time") == 109
        and board.get("score") == 7950
        and board.get("score_target") == 9650
        and board.get("curve_plan_exhausted") is False
    )


def _is_board_reset_entry_candidate(
    *,
    board: dict[str, Any],
    esi: int,
    return_address: int,
    source_board_address: int,
) -> bool:
    """Accept only a statically linked reset on the anchored active Board."""

    values = (
        board.get("board_address"),
        esi,
        return_address,
        source_board_address,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in values
    ):
        return False
    board_address = int(board["board_address"])
    return (
        board.get("board_snapshot_error") is None
        and board_address > 0
        and board_address == esi == source_board_address
        and return_address in BOARD_RESET_DIRECT_RETURN_ADDRESSES
    )


def _is_board_target_write_completion(
    *,
    board: dict[str, Any],
    source_board_address: int,
    source_score_target: int,
) -> bool:
    """Stop only on a positive changed target on the anchored Board."""

    values = (
        board.get("board_address"),
        board.get("score_target"),
        source_board_address,
        source_score_target,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in values
    ):
        return False
    return (
        board.get("board_snapshot_error") is None
        and int(board["board_address"]) == source_board_address > 0
        and int(board["score_target"]) > 0
        and int(board["score_target"]) != source_score_target
    )


def _strict_event_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _is_natural_loss_target_clear_event(
    *,
    source: Mapping[str, Any],
    event: Mapping[str, Any],
) -> bool:
    """Match the preregistered in-place natural-loss target clearing write."""

    source_board = _strict_event_int(source.get("board_address"))
    source_update = _strict_event_int(source.get("framework_update"))
    source_perf = _strict_event_int(source.get("perf_counter_ns"))
    event_board = _strict_event_int(event.get("board_address"))
    event_update = _strict_event_int(event.get("framework_update"))
    event_perf = _strict_event_int(event.get("perf_counter_ns"))
    registers = event.get("registers")
    return bool(
        source.get("source_board_candidate") is True
        and source_board is not None
        and source_board > 0
        and source_update is not None
        and source_perf is not None
        and source.get("native_game_time") == 109
        and source.get("score") == 7950
        and source.get("score_target") == 9650
        and source.get("curve_plan_exhausted") is False
        and event.get("event_kind") == "board_score_target_write"
        and event.get("write_order") == 0
        and event_board == source_board
        and event.get("same_active_board_as_source") is True
        and event.get("watched_address")
        == source_board + BOARD_SCORE_TARGET_OFFSET
        and event.get("source_score_target") == 9650
        and event.get("previous_score_target") == 9650
        and event.get("post_score_target") == 0
        and event.get("native_game_time") == 0
        and event.get("score") == 7950
        and event.get("curve_plan_exhausted") is True
        and event.get("exception_address")
        == NATURAL_LOSS_TARGET_CLEAR_POST_EIP
        and event.get("post_instruction_eip")
        == NATURAL_LOSS_TARGET_CLEAR_POST_EIP
        and event_update is not None
        and event_update > source_update
        and event_perf is not None
        and event_perf > source_perf
        and isinstance(registers, Mapping)
        and registers.get("Edi") == source_board
        and registers.get("Ebx") == 0
        and registers.get("Eip") == NATURAL_LOSS_TARGET_CLEAR_POST_EIP
    )


def _is_natural_loss_target_write_pair(
    *,
    source: Mapping[str, Any],
    events: Iterable[Mapping[str, Any]],
) -> bool:
    """Match the exact clear-then-positive writer pair frozen for c155."""

    pair = tuple(events)
    if len(pair) != 2:
        return False
    clear, positive = pair
    if not _is_natural_loss_target_clear_event(
        source=source,
        event=clear,
    ):
        return False
    source_board = _strict_event_int(source.get("board_address"))
    clear_update = _strict_event_int(clear.get("framework_update"))
    clear_perf = _strict_event_int(clear.get("perf_counter_ns"))
    positive_perf = _strict_event_int(positive.get("perf_counter_ns"))
    positive_target = _strict_event_int(positive.get("post_score_target"))
    registers = positive.get("registers")
    return bool(
        source_board is not None
        and source_board > 0
        and clear_update is not None
        and clear_perf is not None
        and positive_perf is not None
        and positive_target is not None
        and positive_target > 0
        and positive_target != 9650
        and positive.get("event_kind") == "board_score_target_write"
        and positive.get("write_order") == 1
        and positive.get("board_address") == source_board
        and positive.get("same_active_board_as_source") is True
        and positive.get("watched_address")
        == source_board + BOARD_SCORE_TARGET_OFFSET
        and positive.get("source_score_target") == 9650
        and positive.get("previous_score_target") == 0
        and positive.get("native_game_time") == 0
        and positive.get("score") == 7950
        and positive.get("curve_plan_exhausted") is False
        and positive.get("exception_address")
        == NATURAL_LOSS_TARGET_POSITIVE_POST_EIP
        and positive.get("post_instruction_eip")
        == NATURAL_LOSS_TARGET_POSITIVE_POST_EIP
        and positive.get("framework_update") == clear_update
        and 0 < positive_perf - clear_perf
        <= NATURAL_LOSS_TARGET_WRITE_MAX_INTERVAL_NS
        and isinstance(registers, Mapping)
        and registers.get("Esi") == source_board
        and registers.get("Edx") == positive_target
        and registers.get("Eip")
        == NATURAL_LOSS_TARGET_POSITIVE_POST_EIP
    )


def _retarget_hardware_breakpoint_context(
    context: Any,
    address: int,
    *,
    access: str = "execute",
) -> None:
    """Retarget DR0 while the owning thread is paused at a debug event."""

    if isinstance(address, bool) or not isinstance(address, int) or address <= 0:
        raise ValueError("hardware breakpoint address is invalid")
    if access not in {"execute", "write4"}:
        raise ValueError("hardware breakpoint access is invalid")
    if access == "write4" and address % 4:
        raise ValueError("four-byte watchpoint address is not aligned")
    context.Dr0 = address
    context.Dr6 = 0
    context.Dr7 &= ~0x000F0003
    context.Dr7 |= 0x1 if access == "execute" else 0x000D0001
    context.EFlags |= RESUME_FLAG


def _restore_hardware_breakpoint_context(
    context: Any,
    original: tuple[int, int, int, int, int, int],
) -> None:
    """Restore all frozen debug registers in an already paused context."""

    (
        context.Dr0,
        context.Dr1,
        context.Dr2,
        context.Dr3,
        context.Dr6,
        context.Dr7,
    ) = original
    context.EFlags |= RESUME_FLAG


def _board_snapshot(process: int) -> dict[str, Any]:
    try:
        return _board_snapshot_strict(process)
    except (OSError, RuntimeError, ValueError) as error:
        return {
            "app_address": 0,
            "framework_update": -1,
            "board_address": 0,
            "native_game_time": None,
            "score": None,
            "score_target": None,
            "curve_plan_exhausted": None,
            "board_snapshot_error": (
                f"{type(error).__name__}: {error}"
            ),
        }


def _board_snapshot_strict(process: int) -> dict[str, int | bool | None]:
    app = _read_u32(process, G_SEXY_APP_BASE_ADDRESS)
    if app == 0:
        return {
            "app_address": 0,
            "framework_update": -1,
            "board_address": 0,
            "native_game_time": None,
            "score": None,
            "score_target": None,
            "curve_plan_exhausted": None,
        }
    update = _read_i32(
        process,
        app + DEFAULT_BOARD_UPDATE_OFFSET,
    )
    board = _read_u32(process, app + ACTIVE_BOARD_OFFSET)
    if board == 0:
        return {
            "app_address": app,
            "framework_update": update,
            "board_address": 0,
            "native_game_time": None,
            "score": None,
            "score_target": None,
            "curve_plan_exhausted": None,
        }
    return {
        "app_address": app,
        "framework_update": update,
        "board_address": board,
        "native_game_time": _read_i32(
            process,
            board + BOARD_NATIVE_GAME_TIME_OFFSET,
        ),
        "score": _read_i32(
            process,
            board + BOARD_SCORE_OFFSET,
        ),
        "score_target": _read_i32(
            process,
            board + BOARD_SCORE_TARGET_OFFSET,
        ),
        "curve_plan_exhausted": bool(
            read_memory(process, G_CURVE_PLAN_EXHAUSTED_ADDRESS, 1)[0]
        ),
    }


def _predicted_post_state(
    payload: bytes,
) -> tuple[int, int, bytes]:
    if len(payload) != MTRAND_STATE_BYTES:
        raise RuntimeError("global MTRand state size mismatch")
    unpacked = struct.unpack("<625I", payload)
    index = unpacked[MTRAND_STATE_WORDS]
    if index > MTRAND_STATE_WORDS:
        raise RuntimeError("global MTRand index is invalid")
    rng = PopCapMTRandom(1)
    rng.load_state(unpacked[:MTRAND_STATE_WORDS], index)
    output = rng.next_u31()
    post_state = struct.pack("<625I", *rng.words, rng.index)
    return output, rng.index, post_state


def _thread_crt_snapshot(
    *,
    process: int,
    thread: int,
    context: Any,
    thread_id: int,
) -> dict[str, Any]:
    """Read the paused main thread's CRT state at the HW boundary."""

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


def _global_mtrand_restore_source_evidence(
    state: GlobalMTRandRestoreState,
) -> dict[str, Any]:
    return {
        "source_kind": state.source_kind,
        "source_path": str(state.source_path),
        "source_sha256": state.source_sha256,
        "source_process_id": state.source_process_id,
        "source_framework_update": state.source_framework_update,
        "source_native_game_time": state.source_native_game_time,
        "source_trace_path": (
            str(state.source_trace_path)
            if state.source_trace_path is not None
            else None
        ),
        "source_trace_sha256": state.source_trace_sha256,
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
        "source_call_order": state.source_call_order,
        "source_caller": state.source_caller,
        "source_caller_hex": (
            f"0x{state.source_caller:08x}"
            if state.source_caller is not None
            else None
        ),
        **state.semantic_dict(),
        "semantic_sha256": state.semantic_sha256,
    }


def _apply_global_mtrand_restore(
    *,
    process: int,
    process_id: int,
    thread_id: int,
    framework_update: int,
    caller: int,
    live_before: bytes,
    desired: GlobalMTRandRestoreState,
) -> dict[str, Any]:
    """Transactionally restore one source-bound global MTRand state."""

    if len(live_before) != MTRAND_STATE_BYTES:
        raise RuntimeError("live global MTRand state size mismatch")
    live_before_index = struct.unpack_from(
        "<I",
        live_before,
        MTRAND_STATE_WORDS * 4,
    )[0]
    changed = live_before != desired.payload
    try:
        if changed:
            write_memory(
                process,
                DEFAULT_GLOBAL_RNG_STATE,
                desired.payload,
            )
        observed = read_memory(
            process,
            DEFAULT_GLOBAL_RNG_STATE,
            MTRAND_STATE_BYTES,
        )
        if observed != desired.payload:
            raise RuntimeError(
                "hardware call-site global MTRand writeback did not verify"
            )
    except Exception:
        try:
            write_memory(
                process,
                DEFAULT_GLOBAL_RNG_STATE,
                live_before,
            )
            if (
                read_memory(
                    process,
                    DEFAULT_GLOBAL_RNG_STATE,
                    MTRAND_STATE_BYTES,
                )
                != live_before
            ):
                raise RuntimeError(
                    "hardware call-site global MTRand rollback did not verify"
                )
        except Exception as rollback_error:
            raise RuntimeError(
                "hardware call-site global MTRand restore and rollback failed"
            ) from rollback_error
        raise
    return {
        "classification": (
            "diagnostic-source-bound-global-mtrand-hardware-restore"
        ),
        "process_id": process_id,
        "thread_id": thread_id,
        "framework_update": framework_update,
        "caller": caller,
        "caller_hex": f"0x{caller:08x}",
        "state_address": DEFAULT_GLOBAL_RNG_STATE,
        "live_before_index": live_before_index,
        "live_before_state_sha256": _sha256_bytes(live_before),
        "bytes_written": MTRAND_STATE_BYTES if changed else 0,
        "changed": changed,
        "writeback_verified": True,
        "transactional_rollback_on_failure": True,
        "persistent_file_modified": False,
        "restored": _global_mtrand_restore_source_evidence(desired),
    }


def _apply_thread_crt_restore(
    *,
    process: int,
    process_id: int,
    thread: int,
    thread_id: int,
    context: Any,
    framework_update: int,
    caller: int,
    desired: ThreadCrtRestoreState,
) -> dict[str, Any]:
    """Transactionally restore one source-bound main-thread CRT state."""

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
            write_memory(
                process,
                state_address,
                struct.pack("<I", desired.state),
            )
        observed = _read_u32(process, state_address)
        if observed != desired.state:
            raise RuntimeError(
                "hardware call-site thread CRT writeback did not verify"
            )
    except Exception:
        try:
            write_memory(
                process,
                state_address,
                struct.pack("<I", live_before),
            )
            if _read_u32(process, state_address) != live_before:
                raise RuntimeError(
                    "hardware call-site thread CRT rollback did not verify"
                )
        except Exception as rollback_error:
            raise RuntimeError(
                "hardware call-site thread CRT restore and rollback failed"
            ) from rollback_error
        raise
    return {
        "classification": (
            "diagnostic-source-bound-thread-crt-hardware-restore"
        ),
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


def trace_gameplay_mtrand(
    *,
    pid: int,
    main_thread_id: int,
    executable: Path,
    output_path: Path,
    ready_path: Path,
    stop_path: Path,
    address: int = DEFAULT_GAMEPLAY_RNG_RETURN,
    breakpoint_kind: str = "gameplay_return",
    attach_timeout: float = 60.0,
    timeout: float = 900.0,
    maximum_calls: int = 100_000,
    global_mtrand_restore_state: GlobalMTRandRestoreState | None = None,
    global_mtrand_restore_framework_update: int | None = None,
    global_mtrand_restore_caller: int | None = None,
    thread_crt_restore_state: ThreadCrtRestoreState | None = None,
    thread_crt_restore_framework_update: int | None = None,
    thread_crt_restore_caller: int | None = None,
) -> Path:
    """Capture exact outputs, with an optional source-bound CRT restore."""

    if (
        pid <= 0
        or main_thread_id <= 0
        or address <= 0
        or attach_timeout <= 0
        or timeout <= 0
        or maximum_calls <= 0
        or breakpoint_kind not in BREAKPOINT_KINDS
    ):
        raise ValueError("invalid gameplay MTRand trace target")
    global_mtrand_restore_values = (
        global_mtrand_restore_state,
        global_mtrand_restore_framework_update,
        global_mtrand_restore_caller,
    )
    if any(value is not None for value in global_mtrand_restore_values) and any(
        value is None for value in global_mtrand_restore_values
    ):
        raise ValueError(
            "global MTRand hardware restore options must be supplied together"
        )
    if global_mtrand_restore_state is not None and (
        breakpoint_kind != "global_wrapper_entry"
        or global_mtrand_restore_framework_update is None
        or global_mtrand_restore_framework_update < 0
        or global_mtrand_restore_caller is None
        or global_mtrand_restore_caller <= 0
    ):
        raise ValueError(
            "global MTRand hardware restore requires a valid global wrapper "
            "call-site target"
        )
    thread_crt_restore_values = (
        thread_crt_restore_state,
        thread_crt_restore_framework_update,
        thread_crt_restore_caller,
    )
    if any(value is not None for value in thread_crt_restore_values) and any(
        value is None for value in thread_crt_restore_values
    ):
        raise ValueError(
            "thread CRT hardware restore options must be supplied together"
        )
    if thread_crt_restore_state is not None and (
        breakpoint_kind != "global_wrapper_entry"
        or thread_crt_restore_framework_update is None
        or thread_crt_restore_framework_update < 0
        or thread_crt_restore_caller is None
        or thread_crt_restore_caller <= 0
    ):
        raise ValueError(
            "thread CRT hardware restore requires a valid global wrapper "
            "call-site target"
        )
    for path in (output_path, ready_path, stop_path):
        if path.exists() or not path.parent.is_dir():
            raise FileExistsError(f"trace path is not new: {path}")
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
    calls: list[dict[str, Any]] = []
    exited = False
    exit_code: int | None = None
    stop_reason = "timeout"
    failure: str | None = None
    exception_detail: str | None = None
    stop_break_requested = False
    stop_break_observed = False
    global_mtrand_restore_observation: dict[str, Any] | None = None
    thread_crt_restore_observation: dict[str, Any] | None = None
    hardware_released_by_process_exit = False
    observed_call_instruction_hex: str | None = None
    observed_reset_entry_instruction_hex: str | None = None
    observed_reset_native_clock_zero_instruction_hex: str | None = None
    observed_reset_return_instruction_hex: str | None = None
    observed_natural_loss_clear_instruction_hex: str | None = None
    observed_natural_loss_positive_instruction_hex: str | None = None
    active_breakpoint_address = address
    combined_phase = (
        "source_anchor"
        if breakpoint_kind == BOARD_GLOBAL_PRECALL_AND_RESET_KIND
        or breakpoint_kind in BOARD_TARGET_WRITE_KINDS
        else None
    )
    source_board_address: int | None = None
    source_score_target: int | None = None
    source_anchor_observation: dict[str, Any] | None = None
    target_watch_address: int | None = None
    target_write_events: list[dict[str, Any]] = []
    target_write_completion: dict[str, Any] | None = None
    reset_entry_attempts: list[dict[str, Any]] = []
    reset_entry_observation: dict[str, Any] | None = None
    reset_return_observation: dict[str, Any] | None = None
    breakpoint_transitions: list[dict[str, Any]] = []
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
                        "gameplay_mtrand_waiting_for_debugger_handoff "
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

        if breakpoint_kind in BOARD_SOURCE_BREAKPOINT_KINDS:
            observed_instruction = read_memory(
                process,
                address,
                len(BOARD_GLOBAL_PRECALL_INSTRUCTION),
            )
            observed_call_instruction_hex = observed_instruction.hex()
            if observed_instruction != BOARD_GLOBAL_PRECALL_INSTRUCTION:
                raise RuntimeError(
                    "Board global-MTRand pre-call instruction mismatch"
                )
        if breakpoint_kind == BOARD_GLOBAL_PRECALL_AND_RESET_KIND:
            observed_reset_entry = read_memory(
                process,
                DEFAULT_BOARD_RESET_ENTRY,
                len(BOARD_RESET_ENTRY_INSTRUCTION),
            )
            observed_reset_entry_instruction_hex = observed_reset_entry.hex()
            if observed_reset_entry != BOARD_RESET_ENTRY_INSTRUCTION:
                raise RuntimeError("Board reset entry instruction mismatch")
            observed_reset_zero = read_memory(
                process,
                BOARD_RESET_NATIVE_CLOCK_ZERO_ADDRESS,
                len(BOARD_RESET_NATIVE_CLOCK_ZERO_INSTRUCTION),
            )
            observed_reset_native_clock_zero_instruction_hex = (
                observed_reset_zero.hex()
            )
            if observed_reset_zero != BOARD_RESET_NATIVE_CLOCK_ZERO_INSTRUCTION:
                raise RuntimeError(
                    "Board reset native-clock-zero instruction mismatch"
                )
            observed_reset_return = read_memory(
                process,
                BOARD_RESET_RETURN_ADDRESS,
                len(BOARD_RESET_RETURN_INSTRUCTION),
            )
            observed_reset_return_instruction_hex = (
                observed_reset_return.hex()
            )
            if observed_reset_return != BOARD_RESET_RETURN_INSTRUCTION:
                raise RuntimeError("Board reset return instruction mismatch")
        if (
            breakpoint_kind
            == BOARD_GLOBAL_PRECALL_AND_NATURAL_LOSS_TARGET_WRITE_KIND
        ):
            observed_clear = read_memory(
                process,
                NATURAL_LOSS_TARGET_CLEAR_WRITER_ADDRESS,
                len(NATURAL_LOSS_TARGET_CLEAR_WRITER_INSTRUCTION),
            )
            observed_natural_loss_clear_instruction_hex = (
                observed_clear.hex()
            )
            if observed_clear != NATURAL_LOSS_TARGET_CLEAR_WRITER_INSTRUCTION:
                raise RuntimeError(
                    "natural-loss target-clear writer instruction mismatch"
                )
            observed_positive = read_memory(
                process,
                NATURAL_LOSS_TARGET_POSITIVE_WRITER_ADDRESS,
                len(NATURAL_LOSS_TARGET_POSITIVE_WRITER_INSTRUCTION),
            )
            observed_natural_loss_positive_instruction_hex = (
                observed_positive.hex()
            )
            if (
                observed_positive
                != NATURAL_LOSS_TARGET_POSITIVE_WRITER_INSTRUCTION
            ):
                raise RuntimeError(
                    "natural-loss positive-target writer instruction mismatch"
                )

        hardware_original = _arm_hardware_breakpoint(
            main_thread_id=main_thread_id,
            address=address,
        )
        hardware_armed = True
        _write_canonical(
            ready_path,
            {
                "schema": (
                    "zuma-rl.pc-gameplay-mtrand-call-trace-ready"
                ),
                "version": 1,
                "process_id": pid,
                "main_thread_id": main_thread_id,
                "address": address,
                "breakpoint_kind": breakpoint_kind,
                "board_reset_entry": (
                    DEFAULT_BOARD_RESET_ENTRY
                    if breakpoint_kind
                    == BOARD_GLOBAL_PRECALL_AND_RESET_KIND
                    else None
                ),
                "board_score_target_offset": (
                    BOARD_SCORE_TARGET_OFFSET
                    if breakpoint_kind in BOARD_TARGET_WRITE_KINDS
                    else None
                ),
                "natural_loss_target_clear_writer": (
                    NATURAL_LOSS_TARGET_CLEAR_WRITER_ADDRESS
                    if breakpoint_kind
                    == BOARD_GLOBAL_PRECALL_AND_NATURAL_LOSS_TARGET_WRITE_KIND
                    else None
                ),
                "natural_loss_target_positive_writer": (
                    NATURAL_LOSS_TARGET_POSITIVE_WRITER_ADDRESS
                    if breakpoint_kind
                    == BOARD_GLOBAL_PRECALL_AND_NATURAL_LOSS_TARGET_WRITE_KIND
                    else None
                ),
            },
        )
        print(
            f"gameplay_mtrand_armed pid={pid} tid={main_thread_id} "
            f"address=0x{address:08X} kind={breakpoint_kind}",
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
                    # A hardware exception can already be queued behind the
                    # controlled DebugBreakProcess event.  Consume it before
                    # detaching, after clearing stale debug status.
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
                            or exception_address == active_breakpoint_address
                            or int(context.Eip) == active_breakpoint_address
                        )
                        if is_target:
                            if (
                                breakpoint_kind
                                == BOARD_GLOBAL_PRECALL_AND_RESET_KIND
                                and combined_phase == "reset_entry"
                            ):
                                board = _board_snapshot(process)
                                stack = struct.unpack(
                                    "<III",
                                    read_memory(
                                        process,
                                        int(context.Esp),
                                        12,
                                    ),
                                )
                                return_address, argument_0, argument_1 = stack
                                entry = {
                                    "event_kind": "board_reset_entry",
                                    "order": len(calls),
                                    "thread_id": main_thread_id,
                                    "entry_address": DEFAULT_BOARD_RESET_ENTRY,
                                    "entry_address_hex": (
                                        f"0x{DEFAULT_BOARD_RESET_ENTRY:08x}"
                                    ),
                                    "entry_instruction_hex": (
                                        observed_reset_entry_instruction_hex
                                    ),
                                    "return_address": return_address,
                                    "return_address_hex": (
                                        f"0x{return_address:08x}"
                                    ),
                                    "call_site": return_address - 5,
                                    "call_site_hex": (
                                        f"0x{return_address - 5:08x}"
                                    ),
                                    "argument_0": argument_0,
                                    "argument_0_hex": f"0x{argument_0:08x}",
                                    "argument_1": argument_1,
                                    "argument_1_hex": f"0x{argument_1:08x}",
                                    "registers": {
                                        name: int(getattr(context, name))
                                        for name in (
                                            "Eax",
                                            "Ebx",
                                            "Ecx",
                                            "Edx",
                                            "Esi",
                                            "Edi",
                                            "Ebp",
                                            "Esp",
                                            "Eip",
                                            "EFlags",
                                        )
                                    },
                                    "candidate_matches_frozen_structure": (
                                        _is_board_reset_entry_candidate(
                                            board=board,
                                            esi=int(context.Esi),
                                            return_address=return_address,
                                            source_board_address=int(
                                                source_board_address or 0
                                            ),
                                        )
                                    ),
                                    "perf_counter_ns": time.perf_counter_ns(),
                                    **board,
                                }
                                reset_entry_attempts.append(entry)
                                if entry[
                                    "candidate_matches_frozen_structure"
                                ]:
                                    reset_entry_observation = entry
                                    calls.append(entry)
                                    old_address = active_breakpoint_address
                                    active_breakpoint_address = return_address
                                    combined_phase = "reset_return"
                                    breakpoint_transitions.append(
                                        {
                                            "from": old_address,
                                            "to": return_address,
                                            "reason": "accepted_board_reset_entry",
                                            "perf_counter_ns": (
                                                time.perf_counter_ns()
                                            ),
                                        }
                                    )
                                    _retarget_hardware_breakpoint_context(
                                        context,
                                        active_breakpoint_address,
                                    )
                                    _write_context(thread, context)
                                    print(
                                        "board_reset_entry "
                                        f"board=0x{int(context.Esi):08X} "
                                        f"caller=0x{return_address - 5:08X} "
                                        f"return=0x{return_address:08X}",
                                        flush=True,
                                    )
                                else:
                                    context.Dr6 = 0
                                    context.EFlags |= RESUME_FLAG
                                    _write_context(thread, context)
                            elif (
                                breakpoint_kind
                                == BOARD_GLOBAL_PRECALL_AND_RESET_KIND
                                and combined_phase == "reset_return"
                            ):
                                board = _board_snapshot(process)
                                pair_matches = bool(
                                    reset_entry_observation is not None
                                    and int(context.Eip)
                                    == active_breakpoint_address
                                    and board.get("board_address")
                                    == reset_entry_observation.get(
                                        "board_address"
                                    )
                                )
                                reset_return_observation = {
                                    "event_kind": "board_reset_return",
                                    "order": len(calls),
                                    "thread_id": main_thread_id,
                                    "return_address": (
                                        active_breakpoint_address
                                    ),
                                    "return_address_hex": (
                                        f"0x{active_breakpoint_address:08x}"
                                    ),
                                    "entry_order": (
                                        reset_entry_observation.get("order")
                                        if reset_entry_observation is not None
                                        else None
                                    ),
                                    "registers": {
                                        name: int(getattr(context, name))
                                        for name in (
                                            "Eax",
                                            "Ebx",
                                            "Ecx",
                                            "Edx",
                                            "Esi",
                                            "Edi",
                                            "Ebp",
                                            "Esp",
                                            "Eip",
                                            "EFlags",
                                        )
                                    },
                                    "pair_matches_frozen_structure": (
                                        pair_matches
                                    ),
                                    "perf_counter_ns": time.perf_counter_ns(),
                                    **board,
                                }
                                calls.append(reset_return_observation)
                                if not pair_matches:
                                    failure = "board_reset_return_pair_mismatch"
                                    stop_reason = "semantic_failure"
                                else:
                                    stop_reason = "board_reset_pair_complete"
                                if hardware_original is None:
                                    raise RuntimeError(
                                        "original hardware context is missing"
                                    )
                                _restore_hardware_breakpoint_context(
                                    context,
                                    hardware_original,
                                )
                                _write_context(thread, context)
                                hardware_armed = False
                                combined_phase = "complete"
                                stop_after_continue = True
                                print(
                                    "board_reset_return "
                                    f"board=0x{int(board.get('board_address') or 0):08X} "
                                    f"return=0x{active_breakpoint_address:08X} "
                                    f"pair={pair_matches}",
                                    flush=True,
                                )
                            elif (
                                breakpoint_kind in BOARD_TARGET_WRITE_KINDS
                                and combined_phase == "score_target_write"
                            ):
                                if (
                                    source_board_address is None
                                    or source_score_target is None
                                    or target_watch_address is None
                                ):
                                    raise RuntimeError(
                                        "target watchpoint source is incomplete"
                                    )
                                board = _board_snapshot(process)
                                post_eip = int(context.Eip)
                                code_window_address = max(0, post_eip - 16)
                                code_window = read_memory(
                                    process,
                                    code_window_address,
                                    32,
                                )
                                stack_bytes = read_memory(
                                    process,
                                    int(context.Esp),
                                    64,
                                )
                                previous_score_target = (
                                    int(
                                        target_write_events[-1][
                                            "post_score_target"
                                        ]
                                    )
                                    if target_write_events
                                    and isinstance(
                                        target_write_events[-1].get(
                                            "post_score_target"
                                        ),
                                        int,
                                    )
                                    else source_score_target
                                )
                                natural_loss_semantic_mode = (
                                    breakpoint_kind
                                    == BOARD_GLOBAL_PRECALL_AND_NATURAL_LOSS_TARGET_WRITE_KIND
                                )
                                completion = (
                                    False
                                    if natural_loss_semantic_mode
                                    else _is_board_target_write_completion(
                                        board=board,
                                        source_board_address=(
                                            source_board_address
                                        ),
                                        source_score_target=(
                                            source_score_target
                                        ),
                                    )
                                )
                                target_event = {
                                    "event_kind": "board_score_target_write",
                                    "order": len(calls),
                                    "write_order": len(target_write_events),
                                    "thread_id": main_thread_id,
                                    "watched_address": target_watch_address,
                                    "watched_address_hex": (
                                        f"0x{target_watch_address:08x}"
                                    ),
                                    "exception_address": exception_address,
                                    "exception_address_hex": (
                                        f"0x{exception_address:08x}"
                                    ),
                                    "post_instruction_eip": post_eip,
                                    "post_instruction_eip_hex": (
                                        f"0x{post_eip:08x}"
                                    ),
                                    "code_window_address": (
                                        code_window_address
                                    ),
                                    "code_window_address_hex": (
                                        f"0x{code_window_address:08x}"
                                    ),
                                    "code_window_hex": code_window.hex(),
                                    "stack_address": int(context.Esp),
                                    "stack_address_hex": (
                                        f"0x{int(context.Esp):08x}"
                                    ),
                                    "stack_64_bytes_hex": stack_bytes.hex(),
                                    "stack_dwords": list(
                                        struct.unpack("<16I", stack_bytes)
                                    ),
                                    "registers": {
                                        name: int(getattr(context, name))
                                        for name in (
                                            "Eax",
                                            "Ebx",
                                            "Ecx",
                                            "Edx",
                                            "Esi",
                                            "Edi",
                                            "Ebp",
                                            "Esp",
                                            "Eip",
                                            "EFlags",
                                        )
                                    },
                                    "source_score_target": (
                                        source_score_target
                                    ),
                                    "previous_score_target": (
                                        previous_score_target
                                    ),
                                    "post_score_target": board.get(
                                        "score_target"
                                    ),
                                    "same_active_board_as_source": (
                                        board.get("board_address")
                                        == source_board_address
                                    ),
                                    "completion": completion,
                                    "perf_counter_ns": time.perf_counter_ns(),
                                    **board,
                                }
                                semantic_failure: str | None = None
                                if natural_loss_semantic_mode:
                                    if source_anchor_observation is None:
                                        raise RuntimeError(
                                            "natural-loss source anchor is missing"
                                        )
                                    candidate_events = [
                                        *target_write_events,
                                        target_event,
                                    ]
                                    if len(candidate_events) == 1:
                                        if not _is_natural_loss_target_clear_event(
                                            source=source_anchor_observation,
                                            event=target_event,
                                        ):
                                            semantic_failure = (
                                                "natural_loss_target_clear_mismatch"
                                            )
                                    elif len(candidate_events) == 2:
                                        completion = (
                                            _is_natural_loss_target_write_pair(
                                                source=(
                                                    source_anchor_observation
                                                ),
                                                events=candidate_events,
                                            )
                                        )
                                        target_event["completion"] = completion
                                        if not completion:
                                            semantic_failure = (
                                                "natural_loss_target_pair_mismatch"
                                            )
                                    else:
                                        semantic_failure = (
                                            "natural_loss_target_write_overflow"
                                        )
                                target_write_events.append(target_event)
                                calls.append(target_event)
                                if semantic_failure is not None:
                                    failure = semantic_failure
                                    stop_reason = "semantic_failure"
                                    if hardware_original is None:
                                        raise RuntimeError(
                                            "original hardware context is missing"
                                        )
                                    _restore_hardware_breakpoint_context(
                                        context,
                                        hardware_original,
                                    )
                                    _write_context(thread, context)
                                    hardware_armed = False
                                    stop_after_continue = True
                                elif completion:
                                    target_write_completion = target_event
                                    stop_reason = (
                                        "natural_loss_target_write_complete"
                                        if natural_loss_semantic_mode
                                        else "board_score_target_write_complete"
                                    )
                                    if hardware_original is None:
                                        raise RuntimeError(
                                            "original hardware context is missing"
                                        )
                                    _restore_hardware_breakpoint_context(
                                        context,
                                        hardware_original,
                                    )
                                    _write_context(thread, context)
                                    hardware_armed = False
                                    combined_phase = "complete"
                                    stop_after_continue = True
                                    print(
                                        "board_score_target_write "
                                        f"board=0x{source_board_address:08X} "
                                        f"target={previous_score_target}->"
                                        f"{board.get('score_target')} "
                                        f"post_eip=0x{post_eip:08X} "
                                        "complete=True",
                                        flush=True,
                                    )
                                elif len(target_write_events) >= (
                                    2 if natural_loss_semantic_mode else 16
                                ):
                                    failure = "maximum_target_writes"
                                    stop_reason = "semantic_failure"
                                    if hardware_original is None:
                                        raise RuntimeError(
                                            "original hardware context is missing"
                                        )
                                    _restore_hardware_breakpoint_context(
                                        context,
                                        hardware_original,
                                    )
                                    _write_context(thread, context)
                                    hardware_armed = False
                                    stop_after_continue = True
                                else:
                                    context.Dr6 = 0
                                    _write_context(thread, context)
                            else:
                                live_state = read_memory(
                                    process,
                                    DEFAULT_GLOBAL_RNG_STATE,
                                    MTRAND_STATE_BYTES,
                                )
                                if (
                                    breakpoint_kind
                                    in BOARD_SOURCE_BREAKPOINT_KINDS
                                    or breakpoint_kind
                                    == "global_wrapper_entry"
                                ):
                                    is_board_source = (
                                        breakpoint_kind
                                        in BOARD_SOURCE_BREAKPOINT_KINDS
                                    )
                                    caller = (
                                        BOARD_GLOBAL_PRECALL_RETURN
                                        if is_board_source
                                        else _read_u32(
                                            process,
                                            int(context.Esp),
                                        )
                                    )
                                    board = _board_snapshot(process)
                                    update = int(board["framework_update"])
                                    if (
                                        global_mtrand_restore_state is not None
                                        and global_mtrand_restore_observation
                                        is None
                                        and update
                                        == global_mtrand_restore_framework_update
                                        and caller
                                        == global_mtrand_restore_caller
                                    ):
                                        global_mtrand_restore_observation = (
                                            _apply_global_mtrand_restore(
                                                process=process,
                                                process_id=pid,
                                                thread_id=main_thread_id,
                                                framework_update=update,
                                                caller=caller,
                                                live_before=live_state,
                                                desired=(
                                                    global_mtrand_restore_state
                                                ),
                                            )
                                        )
                                        live_state = (
                                            global_mtrand_restore_state.payload
                                        )
                                        print(
                                            "global_mtrand_hardware_restore "
                                            f"update={update} "
                                            f"caller=0x{caller:08X} "
                                            "before="
                                            f"{global_mtrand_restore_observation['live_before_state_sha256']} "
                                            "after="
                                            f"{global_mtrand_restore_state.state_sha256}",
                                            flush=True,
                                        )
                                    if (
                                        thread_crt_restore_state is not None
                                        and thread_crt_restore_observation
                                        is None
                                        and update
                                        == thread_crt_restore_framework_update
                                        and caller == thread_crt_restore_caller
                                    ):
                                        thread_crt_restore_observation = (
                                            _apply_thread_crt_restore(
                                                process=process,
                                                process_id=pid,
                                                thread=thread,
                                                thread_id=main_thread_id,
                                                context=context,
                                                framework_update=update,
                                                caller=caller,
                                                desired=(
                                                    thread_crt_restore_state
                                                ),
                                            )
                                        )
                                        print(
                                            "thread_crt_hardware_restore "
                                            f"update={update} "
                                            f"caller=0x{caller:08X} "
                                            "before="
                                            f"{thread_crt_restore_observation['live_before_state']} "
                                            "after="
                                            f"{thread_crt_restore_state.state}",
                                            flush=True,
                                        )
                                    (
                                        output,
                                        post_index,
                                        post_state,
                                    ) = _predicted_post_state(live_state)
                                    pre_index = struct.unpack_from(
                                        "<I",
                                        live_state,
                                        MTRAND_STATE_WORDS * 4,
                                    )[0]
                                    call = {
                                        "order": len(calls),
                                        "thread_id": main_thread_id,
                                        "wrapper_address": (
                                            DEFAULT_GLOBAL_RNG_WRAPPER
                                            if is_board_source
                                            else address
                                        ),
                                        "wrapper_address_hex": (
                                            "0x00617490"
                                            if is_board_source
                                            else f"0x{address:08x}"
                                        ),
                                        "caller": caller,
                                        "pre_index": pre_index,
                                        "pre_state_sha256": (
                                            _sha256_bytes(live_state)
                                        ),
                                        "output": output,
                                        "post_index": post_index,
                                        "post_state_sha256": (
                                            _sha256_bytes(post_state)
                                        ),
                                        "perf_counter_ns": (
                                            time.perf_counter_ns()
                                        ),
                                        **_thread_crt_snapshot(
                                            process=process,
                                            thread=thread,
                                            context=context,
                                            thread_id=main_thread_id,
                                        ),
                                        **board,
                                    }
                                    if is_board_source:
                                        call.update(
                                            {
                                                "event_kind": (
                                                    "board_global_precall"
                                                ),
                                                "call_address": address,
                                                "call_address_hex": (
                                                    f"0x{address:08x}"
                                                ),
                                                "call_instruction_hex": (
                                                    observed_call_instruction_hex
                                                ),
                                                "source_board_candidate": (
                                                    _is_natural_loss_source_anchor_candidate(
                                                        board
                                                    )
                                                    if breakpoint_kind
                                                    == BOARD_GLOBAL_PRECALL_AND_NATURAL_LOSS_TARGET_WRITE_KIND
                                                    else _is_source_board_anchor_candidate(
                                                        board
                                                    )
                                                ),
                                            }
                                        )
                                    call["caller_hex"] = (
                                        f"0x{int(call['caller']):08x}"
                                    )
                                else:
                                    post_index = struct.unpack_from(
                                        "<I",
                                        live_state,
                                        MTRAND_STATE_WORDS * 4,
                                    )[0]
                                    call = {
                                        "order": len(calls),
                                        "thread_id": main_thread_id,
                                        "return_address": address,
                                        "return_address_hex": (
                                            f"0x{address:08x}"
                                        ),
                                        "output": int(context.Eax),
                                        "post_index": post_index,
                                        "post_state_sha256": (
                                            _sha256_bytes(live_state)
                                        ),
                                        "perf_counter_ns": (
                                            time.perf_counter_ns()
                                        ),
                                        **_board_snapshot(process),
                                    }
                                calls.append(call)
                                if (
                                    (
                                        breakpoint_kind
                                        == BOARD_GLOBAL_PRECALL_AND_RESET_KIND
                                        or breakpoint_kind
                                        in BOARD_TARGET_WRITE_KINDS
                                    )
                                    and call.get("source_board_candidate")
                                    is True
                                ):
                                    source_board_address = int(
                                        call["board_address"]
                                    )
                                    source_score_target = int(
                                        call["score_target"]
                                    )
                                    source_anchor_observation = call
                                    old_address = active_breakpoint_address
                                    if (
                                        breakpoint_kind
                                        == BOARD_GLOBAL_PRECALL_AND_RESET_KIND
                                    ):
                                        active_breakpoint_address = (
                                            DEFAULT_BOARD_RESET_ENTRY
                                        )
                                        combined_phase = "reset_entry"
                                        transition_reason = (
                                            "source_board_anchor"
                                        )
                                        next_label = (
                                            f"0x{active_breakpoint_address:08X}"
                                        )
                                        watch_access = "execute"
                                    else:
                                        target_watch_address = (
                                            source_board_address
                                            + BOARD_SCORE_TARGET_OFFSET
                                        )
                                        active_breakpoint_address = (
                                            target_watch_address
                                        )
                                        combined_phase = (
                                            "score_target_write"
                                        )
                                        transition_reason = (
                                            "source_board_anchor_to_"
                                            "score_target_write_watch"
                                        )
                                        next_label = (
                                            f"0x{active_breakpoint_address:08X}"
                                            "/write4"
                                        )
                                        watch_access = "write4"
                                    breakpoint_transitions.append(
                                        {
                                            "from": old_address,
                                            "to": active_breakpoint_address,
                                            "reason": transition_reason,
                                            "perf_counter_ns": (
                                                time.perf_counter_ns()
                                            ),
                                        }
                                    )
                                    _retarget_hardware_breakpoint_context(
                                        context,
                                        active_breakpoint_address,
                                        access=watch_access,
                                    )
                                    _write_context(thread, context)
                                    print(
                                        "board_source_anchor "
                                        f"board=0x{source_board_address:08X} "
                                        f"next={next_label}",
                                        flush=True,
                                    )
                                else:
                                    context.Dr6 = 0
                                    context.EFlags |= RESUME_FLAG
                                    _write_context(thread, context)
                                    if (
                                        breakpoint_kind
                                        == "board_global_precall"
                                        and call.get(
                                            "source_board_candidate"
                                        )
                                        is True
                                    ):
                                        stop_reason = "source_board_anchor"
                                        stop_after_continue = True
                                    elif len(calls) >= maximum_calls:
                                        stop_reason = "maximum_calls"
                                        stop_after_continue = True
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
                        and exception_address != active_breakpoint_address
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
                hardware_armed = False
                hardware_released_by_process_exit = True

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
        failure = "trace_exception"
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
        if process:
            kernel32.CloseHandle(process)
        if control_process:
            kernel32.CloseHandle(control_process)
        if attached and not exited:
            if not kernel32.DebugActiveProcessStop(pid):
                detach_error = (
                    f"OSError: {ctypes.WinError(ctypes.get_last_error())}"
                )

    if failure is None and not calls:
        failure = "no_gameplay_mtrand_calls"
    if (
        failure is None
        and breakpoint_kind in BOARD_SOURCE_BREAKPOINT_KINDS
        and sum(
            row.get("source_board_candidate") is True
            for row in calls
        )
        != 1
    ):
        failure = "source_board_anchor_not_observed"
    if (
        failure is None
        and breakpoint_kind == BOARD_GLOBAL_PRECALL_AND_RESET_KIND
        and reset_entry_observation is None
    ):
        failure = "board_reset_entry_not_observed"
    if (
        failure is None
        and breakpoint_kind == BOARD_GLOBAL_PRECALL_AND_RESET_KIND
        and reset_return_observation is None
    ):
        failure = "board_reset_return_not_observed"
    if (
        failure is None
        and breakpoint_kind == BOARD_GLOBAL_PRECALL_AND_RESET_KIND
        and reset_return_observation is not None
        and reset_return_observation.get("pair_matches_frozen_structure")
        is not True
    ):
        failure = "board_reset_return_pair_mismatch"
    if (
        failure is None
        and breakpoint_kind in BOARD_TARGET_WRITE_KINDS
        and not target_write_events
    ):
        failure = (
            "natural_loss_target_write_not_observed"
            if breakpoint_kind
            == BOARD_GLOBAL_PRECALL_AND_NATURAL_LOSS_TARGET_WRITE_KIND
            else "board_score_target_write_not_observed"
        )
    if (
        failure is None
        and breakpoint_kind in BOARD_TARGET_WRITE_KINDS
        and target_write_completion is None
    ):
        failure = (
            "natural_loss_target_write_completion_not_observed"
            if breakpoint_kind
            == BOARD_GLOBAL_PRECALL_AND_NATURAL_LOSS_TARGET_WRITE_KIND
            else "board_score_target_write_completion_not_observed"
        )
    if (
        failure is None
        and global_mtrand_restore_state is not None
        and global_mtrand_restore_observation is None
    ):
        failure = "requested_global_mtrand_hardware_restore_not_observed"
    if (
        failure is None
        and thread_crt_restore_state is not None
        and thread_crt_restore_observation is None
    ):
        failure = "requested_thread_crt_hardware_restore_not_observed"
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
            "read-only-hardware-breakpoint-preregistered-natural-loss-observation"
            if breakpoint_kind
            == BOARD_GLOBAL_PRECALL_AND_NATURAL_LOSS_TARGET_WRITE_KIND
            else (
                "diagnostic-source-bound-hardware-breakpoint-replay"
                if (
                global_mtrand_restore_state is not None
                or thread_crt_restore_state is not None
                )
                else "read-only-hardware-breakpoint-diagnostic"
            )
        ),
        "process_id": pid,
        "main_thread_id": main_thread_id,
        "runtime_executable": str(executable.resolve()),
        "runtime_executable_sha256": _sha256_path(executable),
        "address": address,
        "address_hex": f"0x{address:08x}",
        "breakpoint_kind": breakpoint_kind,
        "board_global_precall_instruction_hex": (
            observed_call_instruction_hex
        ),
        "board_reset_entry_address": (
            DEFAULT_BOARD_RESET_ENTRY
            if breakpoint_kind == BOARD_GLOBAL_PRECALL_AND_RESET_KIND
            else None
        ),
        "board_reset_entry_instruction_hex": (
            observed_reset_entry_instruction_hex
        ),
        "board_reset_native_clock_zero_address": (
            BOARD_RESET_NATIVE_CLOCK_ZERO_ADDRESS
            if breakpoint_kind == BOARD_GLOBAL_PRECALL_AND_RESET_KIND
            else None
        ),
        "board_reset_native_clock_zero_instruction_hex": (
            observed_reset_native_clock_zero_instruction_hex
        ),
        "board_reset_return_instruction_address": (
            BOARD_RESET_RETURN_ADDRESS
            if breakpoint_kind == BOARD_GLOBAL_PRECALL_AND_RESET_KIND
            else None
        ),
        "board_reset_return_instruction_hex": (
            observed_reset_return_instruction_hex
        ),
        "board_score_target_offset": (
            BOARD_SCORE_TARGET_OFFSET
            if breakpoint_kind in BOARD_TARGET_WRITE_KINDS
            else None
        ),
        "natural_loss_target_clear_writer_address": (
            NATURAL_LOSS_TARGET_CLEAR_WRITER_ADDRESS
            if breakpoint_kind
            == BOARD_GLOBAL_PRECALL_AND_NATURAL_LOSS_TARGET_WRITE_KIND
            else None
        ),
        "natural_loss_target_clear_writer_instruction_hex": (
            observed_natural_loss_clear_instruction_hex
        ),
        "natural_loss_target_clear_post_eip": (
            NATURAL_LOSS_TARGET_CLEAR_POST_EIP
            if breakpoint_kind
            == BOARD_GLOBAL_PRECALL_AND_NATURAL_LOSS_TARGET_WRITE_KIND
            else None
        ),
        "natural_loss_target_positive_writer_address": (
            NATURAL_LOSS_TARGET_POSITIVE_WRITER_ADDRESS
            if breakpoint_kind
            == BOARD_GLOBAL_PRECALL_AND_NATURAL_LOSS_TARGET_WRITE_KIND
            else None
        ),
        "natural_loss_target_positive_writer_instruction_hex": (
            observed_natural_loss_positive_instruction_hex
        ),
        "natural_loss_target_positive_post_eip": (
            NATURAL_LOSS_TARGET_POSITIVE_POST_EIP
            if breakpoint_kind
            == BOARD_GLOBAL_PRECALL_AND_NATURAL_LOSS_TARGET_WRITE_KIND
            else None
        ),
        "natural_loss_target_write_max_interval_ns": (
            NATURAL_LOSS_TARGET_WRITE_MAX_INTERVAL_NS
            if breakpoint_kind
            == BOARD_GLOBAL_PRECALL_AND_NATURAL_LOSS_TARGET_WRITE_KIND
            else None
        ),
        "board_score_target_watch_address": target_watch_address,
        "board_score_target_watch_address_hex": (
            f"0x{target_watch_address:08x}"
            if target_watch_address is not None
            else None
        ),
        "global_mtrand_address": DEFAULT_GLOBAL_RNG_STATE,
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "started_perf_counter_ns": started_ns,
        "finished_perf_counter_ns": time.perf_counter_ns(),
        "stop_reason": stop_reason,
        "stop_break_requested": stop_break_requested,
        "stop_break_observed": stop_break_observed,
        "exited": exited,
        "exit_code": exit_code,
        "call_count": len(calls),
        "process_memory_writes": (
            (
                int(global_mtrand_restore_observation["bytes_written"])
                if global_mtrand_restore_observation is not None
                else 0
            )
            + (
                int(thread_crt_restore_observation["bytes_written"])
                if thread_crt_restore_observation is not None
                else 0
            )
        ),
        "process_memory_mutation": (
            global_mtrand_restore_observation is not None
            or
            thread_crt_restore_observation is not None
        ),
        "persistent_file_modified": False,
        "hardware_breakpoint_restored": not hardware_armed,
        "hardware_released_by_process_exit": (
            hardware_released_by_process_exit
        ),
        "hardware_breakpoint_restore_error": hardware_restore_error,
        "debugger_detach_error": detach_error,
        "global_mtrand_call_site_restore": (
            global_mtrand_restore_observation
        ),
        "thread_crt_call_site_restore": (
            thread_crt_restore_observation
        ),
        "source_anchor_count": sum(
            row.get("source_board_candidate") is True for row in calls
        ),
        "reset_entry_count": (
            1 if reset_entry_observation is not None else 0
        ),
        "reset_return_count": (
            1 if reset_return_observation is not None else 0
        ),
        "reset_entry_attempts": reset_entry_attempts,
        "reset_pair": (
            {
                "entry": reset_entry_observation,
                "return": reset_return_observation,
                "complete": bool(
                    reset_entry_observation is not None
                    and reset_return_observation is not None
                    and reset_return_observation.get(
                        "pair_matches_frozen_structure"
                    )
                    is True
                ),
            }
            if breakpoint_kind == BOARD_GLOBAL_PRECALL_AND_RESET_KIND
            else None
        ),
        "target_write_count": len(target_write_events),
        "target_write_events": target_write_events,
        "target_write_completion": target_write_completion,
        "natural_loss_target_write_pair": (
            {
                "clear": (
                    target_write_events[0]
                    if len(target_write_events) >= 1
                    else None
                ),
                "positive": (
                    target_write_events[1]
                    if len(target_write_events) >= 2
                    else None
                ),
                "complete": bool(
                    source_anchor_observation is not None
                    and _is_natural_loss_target_write_pair(
                        source=source_anchor_observation,
                        events=target_write_events,
                    )
                ),
            }
            if breakpoint_kind
            == BOARD_GLOBAL_PRECALL_AND_NATURAL_LOSS_TARGET_WRITE_KIND
            else None
        ),
        "breakpoint_transitions": breakpoint_transitions,
        "calls": calls,
    }
    _write_canonical(output_path, result)
    print(
        f"gameplay_mtrand_result status={status} calls={len(calls)} "
        f"stop_reason={stop_reason}",
        flush=True,
    )
    print(output_path.resolve(), flush=True)
    return output_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", required=True, type=int)
    parser.add_argument("--main-thread-id", required=True, type=int)
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ready", required=True, type=Path)
    parser.add_argument("--stop", required=True, type=Path)
    parser.add_argument(
        "--address",
        type=lambda value: int(value, 0),
    )
    parser.add_argument(
        "--breakpoint-kind",
        choices=BREAKPOINT_KINDS,
        default="gameplay_return",
    )
    parser.add_argument("--attach-timeout", type=float, default=60.0)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--maximum-calls", type=int, default=100_000)
    parser.add_argument("--global-mtrand-restore-monitor", type=Path)
    parser.add_argument("--global-mtrand-restore-trace", type=Path)
    parser.add_argument(
        "--global-mtrand-restore-recording-report",
        type=Path,
    )
    parser.add_argument("--global-mtrand-restore-monitor-update", type=int)
    parser.add_argument(
        "--global-mtrand-restore-seed",
        type=lambda value: int(value, 0),
    )
    parser.add_argument(
        "--global-mtrand-restore-rewind-draws",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--global-mtrand-restore-maximum-draws",
        type=int,
        default=100_000,
    )
    parser.add_argument("--global-mtrand-restore-source-order", type=int)
    parser.add_argument("--global-mtrand-restore-framework-update", type=int)
    parser.add_argument(
        "--global-mtrand-restore-caller",
        type=lambda value: int(value, 0),
    )
    parser.add_argument("--thread-crt-restore-trace", type=Path)
    parser.add_argument(
        "--thread-crt-restore-recording-report",
        type=Path,
    )
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
    address = args.address
    if address is None:
        address = _default_breakpoint_address(args.breakpoint_kind)
    global_mtrand_restore_group = (
        args.global_mtrand_restore_monitor,
        args.global_mtrand_restore_trace,
        args.global_mtrand_restore_recording_report,
        args.global_mtrand_restore_monitor_update,
        args.global_mtrand_restore_seed,
        args.global_mtrand_restore_source_order,
        args.global_mtrand_restore_framework_update,
        args.global_mtrand_restore_caller,
    )
    global_mtrand_restore_requested = any(
        value is not None for value in global_mtrand_restore_group
    )
    if global_mtrand_restore_requested and any(
        value is None for value in global_mtrand_restore_group
    ):
        parser.error(
            "all source-bound global MTRand hardware restore options are "
            "required"
        )
    global_mtrand_restore_state = (
        load_source_bound_global_mtrand_restore_state(
            args.global_mtrand_restore_monitor.resolve(),
            args.global_mtrand_restore_trace.resolve(),
            args.global_mtrand_restore_recording_report.resolve(),
            monitor_framework_update=(
                args.global_mtrand_restore_monitor_update
            ),
            seed=args.global_mtrand_restore_seed,
            rewind_draws=args.global_mtrand_restore_rewind_draws,
            source_order=args.global_mtrand_restore_source_order,
            framework_update=(
                args.global_mtrand_restore_framework_update
            ),
            caller=args.global_mtrand_restore_caller,
            maximum_draws=args.global_mtrand_restore_maximum_draws,
        )
        if global_mtrand_restore_requested
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
            "all source-bound thread CRT hardware restore options are "
            "required"
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
    result_path = trace_gameplay_mtrand(
        pid=args.pid,
        main_thread_id=args.main_thread_id,
        executable=args.executable.resolve(),
        output_path=args.output.resolve(),
        ready_path=args.ready.resolve(),
        stop_path=args.stop.resolve(),
        address=address,
        breakpoint_kind=args.breakpoint_kind,
        attach_timeout=args.attach_timeout,
        timeout=args.timeout,
        maximum_calls=args.maximum_calls,
        global_mtrand_restore_state=global_mtrand_restore_state,
        global_mtrand_restore_framework_update=(
            args.global_mtrand_restore_framework_update
        ),
        global_mtrand_restore_caller=args.global_mtrand_restore_caller,
        thread_crt_restore_state=thread_crt_restore_state,
        thread_crt_restore_framework_update=(
            args.thread_crt_restore_framework_update
        ),
        thread_crt_restore_caller=args.thread_crt_restore_caller,
    )
    result = json.loads(result_path.read_text(encoding="ascii"))
    return 0 if result.get("status") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
