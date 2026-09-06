"""Record a retail DMO while the read-only live controller plays one board.

The transaction captures and later restores the exact scoped Zuma users and
registry state.  The runtime is the byte-identical extracted retail payload;
gameplay input travels only through Win32 ``SendInput``.  A successful run
still produces a candidate recording, not certified PC Golden evidence.
"""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tools.autoplay_live_zuma import (
    LiveBoardUnavailable,
    _ensure_dpi_awareness,
    autoplay,
    read_framework_state,
    read_live_board,
    send_mouse_click,
)
from tools.launch_popcap_replay import (
    process_ids_by_name,
    wait_for_pid_window,
)
from tools.launch_fixed_seed_replay import (
    launch_fixed_seed_replay_iat_stub,
)
from tools.pc_state_transaction import (
    capture_state,
    overlay_snapshot_files,
    restore_state,
    write_snapshot_exclusive,
)
from tools.snapshot_dxgi_window import snapshot

EXPECTED_RETAIL_RUNTIME_SHA256 = (
    "sha256:2181ce2bfbfcb4678bf69a1474e08d3db941311aa768176a88453cc69692af20"
)
PROCESS_TERMINATE = 0x0001
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
SYNCHRONIZE = 0x00100000
WAIT_TIMEOUT = 258
STILL_ACTIVE = 259
DEFAULT_TITLE_START_UPDATE = 950
DEFAULT_ADVENTURE_UPDATE = 1280
DEFAULT_CONTINUE_GAME_UPDATE = 1420
DEFAULT_TITLE_START_POINT = (496.0, 477.0)
DEFAULT_ADVENTURE_POINT = (608.0, 279.0)
DEFAULT_CONTINUE_GAME_POINT = (442.0, 463.0)
BOARD_GLOBAL_PRECALL_KIND = "board_global_precall"
BOARD_GLOBAL_PRECALL_AND_RESET_KIND = "board_global_precall_and_reset"
BOARD_GLOBAL_PRECALL_AND_TARGET_WRITE_KIND = (
    "board_global_precall_and_target_write"
)
BOARD_GLOBAL_PRECALL_AND_NATURAL_LOSS_TARGET_WRITE_KIND = (
    "board_global_precall_and_natural_loss_target_write"
)
NATURAL_LOSS_TARGET_WRITE_RECEIPT_KIND = (
    "natural_loss_same_board_target_write_transition_v4"
)
NATURAL_LOSS_TARGET_CLEAR_WRITER_ADDRESS = 0x00412EED
NATURAL_LOSS_TARGET_CLEAR_WRITER_INSTRUCTION_HEX = "899f08010000"
NATURAL_LOSS_TARGET_CLEAR_POST_EIP = 0x00412EF3
NATURAL_LOSS_TARGET_POSITIVE_WRITER_ADDRESS = 0x00411B8D
NATURAL_LOSS_TARGET_POSITIVE_WRITER_INSTRUCTION_HEX = "899608010000"
NATURAL_LOSS_TARGET_POSITIVE_POST_EIP = 0x00411B93
NATURAL_LOSS_TARGET_WRITE_MAX_INTERVAL_NS = 5_000_000
NATURAL_LOSS_TARGET_WRITE_TRACE_CLASSIFICATION = (
    "read-only-hardware-breakpoint-preregistered-natural-loss-observation"
)
BOARD_SOURCE_TRACE_KINDS = frozenset(
    {
        BOARD_GLOBAL_PRECALL_KIND,
        BOARD_GLOBAL_PRECALL_AND_RESET_KIND,
        BOARD_GLOBAL_PRECALL_AND_TARGET_WRITE_KIND,
        BOARD_GLOBAL_PRECALL_AND_NATURAL_LOSS_TARGET_WRITE_KIND,
    }
)
BOARD_GLOBAL_PRECALL_ADDRESS = 0x0065B81C
BOARD_GLOBAL_PRECALL_RETURN = 0x0065B821


def _window_thread_id(window_handle: int, expected_pid: int) -> int:
    """Resolve the verified UI thread without mutating the retail process."""

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.GetWindowThreadProcessId.argtypes = (
        wintypes.HWND,
        ctypes.POINTER(wintypes.DWORD),
    )
    user32.GetWindowThreadProcessId.restype = wintypes.DWORD
    observed_pid = wintypes.DWORD()
    thread_id = int(
        user32.GetWindowThreadProcessId(
            wintypes.HWND(window_handle),
            ctypes.byref(observed_pid),
        )
    )
    if thread_id <= 0 or int(observed_pid.value) != expected_pid:
        raise RuntimeError("retail window thread identity mismatch")
    return thread_id


class _PidProcess:
    """Small Popen-compatible owner for a process launched by Win32."""

    def __init__(self, pid: int) -> None:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = (
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.DWORD,
        )
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetExitCodeProcess.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.DWORD),
        )
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.WaitForSingleObject.argtypes = (
            wintypes.HANDLE,
            wintypes.DWORD,
        )
        kernel32.WaitForSingleObject.restype = wintypes.DWORD
        kernel32.TerminateProcess.argtypes = (
            wintypes.HANDLE,
            wintypes.UINT,
        )
        kernel32.TerminateProcess.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(
            PROCESS_TERMINATE
            | PROCESS_QUERY_LIMITED_INFORMATION
            | SYNCHRONIZE,
            False,
            pid,
        )
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        self.pid = pid
        self._kernel32 = kernel32
        self._handle = handle

    def poll(self) -> int | None:
        exit_code = wintypes.DWORD()
        if not self._kernel32.GetExitCodeProcess(
            self._handle,
            ctypes.byref(exit_code),
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        return (
            None
            if int(exit_code.value) == STILL_ACTIVE
            else int(exit_code.value)
        )

    def wait(self, timeout: float | None = None) -> int:
        milliseconds = (
            0xFFFFFFFF
            if timeout is None
            else max(0, min(0xFFFFFFFE, int(timeout * 1000)))
        )
        status = int(
            self._kernel32.WaitForSingleObject(
                self._handle,
                milliseconds,
            )
        )
        if status == WAIT_TIMEOUT:
            raise subprocess.TimeoutExpired(str(self.pid), timeout)
        if status != 0:
            raise ctypes.WinError(ctypes.get_last_error())
        exit_code = self.poll()
        if exit_code is None:
            raise RuntimeError("signaled process still reports active")
        return exit_code

    def terminate(self) -> None:
        if self.poll() is not None:
            return
        if not self._kernel32.TerminateProcess(self._handle, 1):
            raise ctypes.WinError(ctypes.get_last_error())

    kill = terminate

    def close(self) -> None:
        if self._handle:
            self._kernel32.CloseHandle(self._handle)
            self._handle = None


_ManagedProcess = subprocess.Popen[bytes] | _PidProcess


def _write_json_exclusive(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(
            payload,
            stream,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        stream.write("\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _required_int(
    value: Any,
    name: str,
    *,
    minimum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise RuntimeError(f"{name} must be at least {minimum}")
    return value


def _writer_bytes_match(
    event: Mapping[str, Any],
    *,
    writer_address: int,
    instruction_hex: str,
) -> bool:
    start = event.get("code_window_address")
    payload_hex = event.get("code_window_hex")
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or not isinstance(payload_hex, str)
    ):
        return False
    try:
        payload = bytes.fromhex(payload_hex)
        expected = bytes.fromhex(instruction_hex)
    except ValueError:
        return False
    offset = writer_address - start
    return offset >= 0 and payload[offset : offset + len(expected)] == expected


def _promote_natural_loss_target_write_gameplay(
    gameplay: Mapping[str, Any],
    trace_result: Mapping[str, Any],
    *,
    trace_sha256: str,
) -> dict[str, Any]:
    """Apply the explicit c155 v4 receipt after independently validating it."""

    if (
        not isinstance(trace_sha256, str)
        or not trace_sha256.startswith("sha256:")
        or len(trace_sha256) != 71
    ):
        raise RuntimeError("natural-loss trace digest is invalid")
    if trace_result.get("breakpoint_kind") != (
        BOARD_GLOBAL_PRECALL_AND_NATURAL_LOSS_TARGET_WRITE_KIND
    ):
        raise RuntimeError("natural-loss trace kind is not certifying")
    if (
        trace_result.get("schema")
        != "zuma-rl.pc-gameplay-mtrand-call-trace"
        or trace_result.get("version") != 1
        or trace_result.get("status") != "PASS"
        or trace_result.get("failure") is not None
        or trace_result.get("classification")
        != NATURAL_LOSS_TARGET_WRITE_TRACE_CLASSIFICATION
        or trace_result.get("runtime_executable_sha256")
        != EXPECTED_RETAIL_RUNTIME_SHA256
        or trace_result.get("address") != BOARD_GLOBAL_PRECALL_ADDRESS
        or trace_result.get("board_global_precall_instruction_hex")
        != "e86fbcfbff"
        or trace_result.get("process_memory_writes") != 0
        or trace_result.get("process_memory_mutation") is not False
        or trace_result.get("persistent_file_modified") is not False
        or trace_result.get("hardware_breakpoint_restored") is not True
        or trace_result.get("hardware_breakpoint_restore_error") is not None
        or trace_result.get("debugger_detach_error") is not None
        or trace_result.get("source_anchor_count") != 1
        or trace_result.get("target_write_count") != 2
        or trace_result.get("stop_reason")
        != "natural_loss_target_write_complete"
        or trace_result.get("natural_loss_target_clear_writer_address")
        != NATURAL_LOSS_TARGET_CLEAR_WRITER_ADDRESS
        or trace_result.get(
            "natural_loss_target_clear_writer_instruction_hex"
        )
        != NATURAL_LOSS_TARGET_CLEAR_WRITER_INSTRUCTION_HEX
        or trace_result.get("natural_loss_target_clear_post_eip")
        != NATURAL_LOSS_TARGET_CLEAR_POST_EIP
        or trace_result.get("natural_loss_target_positive_writer_address")
        != NATURAL_LOSS_TARGET_POSITIVE_WRITER_ADDRESS
        or trace_result.get(
            "natural_loss_target_positive_writer_instruction_hex"
        )
        != NATURAL_LOSS_TARGET_POSITIVE_WRITER_INSTRUCTION_HEX
        or trace_result.get("natural_loss_target_positive_post_eip")
        != NATURAL_LOSS_TARGET_POSITIVE_POST_EIP
        or trace_result.get("natural_loss_target_write_max_interval_ns")
        != NATURAL_LOSS_TARGET_WRITE_MAX_INTERVAL_NS
    ):
        raise RuntimeError("natural-loss trace identity or safety mismatch")

    calls = trace_result.get("calls")
    events = trace_result.get("target_write_events")
    pair = trace_result.get("natural_loss_target_write_pair")
    if not isinstance(calls, list) or not isinstance(events, list):
        raise RuntimeError("natural-loss trace event arrays are invalid")
    sources = [
        row
        for row in calls
        if isinstance(row, Mapping)
        and row.get("source_board_candidate") is True
    ]
    if len(sources) != 1 or len(events) != 2:
        raise RuntimeError("natural-loss trace cardinality mismatch")
    if (
        not isinstance(pair, Mapping)
        or pair.get("complete") is not True
        or pair.get("clear") != events[0]
        or pair.get("positive") != events[1]
        or trace_result.get("target_write_completion") != events[1]
    ):
        raise RuntimeError("natural-loss trace pair binding mismatch")

    source = sources[0]
    clear = events[0]
    positive = events[1]
    if not isinstance(clear, Mapping) or not isinstance(positive, Mapping):
        raise RuntimeError("natural-loss target events are invalid")
    source_board = _required_int(
        source.get("board_address"),
        "natural-loss source Board",
        minimum=1,
    )
    source_update = _required_int(
        source.get("framework_update"),
        "natural-loss source update",
        minimum=0,
    )
    source_perf = _required_int(
        source.get("perf_counter_ns"),
        "natural-loss source time",
        minimum=1,
    )
    if (
        source.get("event_kind") != "board_global_precall"
        or source.get("call_address") != BOARD_GLOBAL_PRECALL_ADDRESS
        or source.get("caller") != BOARD_GLOBAL_PRECALL_RETURN
        or source.get("call_instruction_hex") != "e86fbcfbff"
        or source.get("native_game_time") != 109
        or source.get("score") != 7950
        or source.get("score_target") != 9650
        or source.get("curve_plan_exhausted") is not False
    ):
        raise RuntimeError("natural-loss source anchor mismatch")

    clear_update = _required_int(
        clear.get("framework_update"),
        "natural-loss clear update",
        minimum=0,
    )
    clear_perf = _required_int(
        clear.get("perf_counter_ns"),
        "natural-loss clear time",
        minimum=1,
    )
    positive_perf = _required_int(
        positive.get("perf_counter_ns"),
        "natural-loss positive time",
        minimum=1,
    )
    positive_target = _required_int(
        positive.get("post_score_target"),
        "natural-loss positive target",
        minimum=1,
    )
    clear_registers = clear.get("registers")
    positive_registers = positive.get("registers")
    watched_address = source_board + 0x108
    if (
        clear.get("event_kind") != "board_score_target_write"
        or clear.get("write_order") != 0
        or clear.get("board_address") != source_board
        or clear.get("same_active_board_as_source") is not True
        or clear.get("watched_address") != watched_address
        or clear.get("source_score_target") != 9650
        or clear.get("previous_score_target") != 9650
        or clear.get("post_score_target") != 0
        or clear.get("native_game_time") != 0
        or clear.get("score") != 7950
        or clear.get("curve_plan_exhausted") is not True
        or clear.get("exception_address")
        != NATURAL_LOSS_TARGET_CLEAR_POST_EIP
        or clear.get("post_instruction_eip")
        != NATURAL_LOSS_TARGET_CLEAR_POST_EIP
        or clear.get("completion") is not False
        or clear_update <= source_update
        or clear_perf <= source_perf
        or not isinstance(clear_registers, Mapping)
        or clear_registers.get("Edi") != source_board
        or clear_registers.get("Ebx") != 0
        or clear_registers.get("Eip")
        != NATURAL_LOSS_TARGET_CLEAR_POST_EIP
        or not _writer_bytes_match(
            clear,
            writer_address=NATURAL_LOSS_TARGET_CLEAR_WRITER_ADDRESS,
            instruction_hex=NATURAL_LOSS_TARGET_CLEAR_WRITER_INSTRUCTION_HEX,
        )
    ):
        raise RuntimeError("natural-loss target-clear event mismatch")
    interval_ns = positive_perf - clear_perf
    if (
        positive_target == 9650
        or positive.get("event_kind") != "board_score_target_write"
        or positive.get("write_order") != 1
        or positive.get("board_address") != source_board
        or positive.get("same_active_board_as_source") is not True
        or positive.get("watched_address") != watched_address
        or positive.get("source_score_target") != 9650
        or positive.get("previous_score_target") != 0
        or positive.get("native_game_time") != 0
        or positive.get("score") != 7950
        or positive.get("curve_plan_exhausted") is not False
        or positive.get("exception_address")
        != NATURAL_LOSS_TARGET_POSITIVE_POST_EIP
        or positive.get("post_instruction_eip")
        != NATURAL_LOSS_TARGET_POSITIVE_POST_EIP
        or positive.get("completion") is not True
        or positive.get("framework_update") != clear_update
        or not 0 < interval_ns <= NATURAL_LOSS_TARGET_WRITE_MAX_INTERVAL_NS
        or not isinstance(positive_registers, Mapping)
        or positive_registers.get("Esi") != source_board
        or positive_registers.get("Edx") != positive_target
        or positive_registers.get("Eip")
        != NATURAL_LOSS_TARGET_POSITIVE_POST_EIP
        or not _writer_bytes_match(
            positive,
            writer_address=NATURAL_LOSS_TARGET_POSITIVE_WRITER_ADDRESS,
            instruction_hex=(
                NATURAL_LOSS_TARGET_POSITIVE_WRITER_INSTRUCTION_HEX
            ),
        )
    ):
        raise RuntimeError("natural-loss positive-target event mismatch")

    if (
        gameplay.get("expected_outcome") != "natural_loss"
        or gameplay.get("gameplay_policy") != "idle"
        or gameplay.get("action_count") != 0
        or gameplay.get("swap_count") != 0
        or gameplay.get("fruit_shot_count") != 0
    ):
        raise RuntimeError("natural-loss idle gameplay contract mismatch")
    final_state = gameplay.get("final_state")
    if not isinstance(final_state, Mapping):
        raise RuntimeError("natural-loss final Board state is missing")
    if (
        final_state.get("board_address") != source_board
        or final_state.get("score") != 7950
        or final_state.get("score_target") != positive_target
    ):
        raise RuntimeError("natural-loss final Board binding mismatch")

    source_fields = (
        "order",
        "thread_id",
        "call_address",
        "caller",
        "framework_update",
        "board_address",
        "native_game_time",
        "score",
        "score_target",
        "curve_plan_exhausted",
        "pre_index",
        "pre_state_sha256",
        "output",
        "post_index",
        "post_state_sha256",
        "perf_counter_ns",
        "thread_crt_rand_state",
    )
    event_fields = (
        "order",
        "write_order",
        "thread_id",
        "watched_address",
        "exception_address",
        "post_instruction_eip",
        "framework_update",
        "board_address",
        "native_game_time",
        "score",
        "score_target",
        "curve_plan_exhausted",
        "source_score_target",
        "previous_score_target",
        "post_score_target",
        "same_active_board_as_source",
        "completion",
        "perf_counter_ns",
        "registers",
        "code_window_address",
        "code_window_hex",
    )
    receipt = {
        "kind": NATURAL_LOSS_TARGET_WRITE_RECEIPT_KIND,
        "outcome": "natural_loss",
        "observation_mode": (
            "read-only hardware watchpoint on anchored Board+0x108"
        ),
        "trace_sha256": trace_sha256,
        "trace_breakpoint_kind": (
            BOARD_GLOBAL_PRECALL_AND_NATURAL_LOSS_TARGET_WRITE_KIND
        ),
        "trace_classification": (
            NATURAL_LOSS_TARGET_WRITE_TRACE_CLASSIFICATION
        ),
        "trace_call_count": _required_int(
            trace_result.get("call_count"),
            "natural-loss trace call count",
            minimum=3,
        ),
        "source_anchor": {key: source.get(key) for key in source_fields},
        "clear_write": {key: clear.get(key) for key in event_fields},
        "positive_write": {
            key: positive.get(key) for key in event_fields
        },
        "target_write_interval_ns": interval_ns,
        "process_memory_writes": 0,
        "process_memory_mutation": False,
        "persistent_file_modified": False,
        "hardware_breakpoint_restored": True,
        "debugger_detach_error": None,
    }
    promoted = dict(gameplay)
    promoted.update(
        {
            "polling_status_before_target_write_receipt": gameplay.get(
                "status"
            ),
            "polling_outcome_before_target_write_receipt": gameplay.get(
                "outcome"
            ),
            "polling_terminal_observation_before_target_write_receipt": (
                gameplay.get("terminal_observation")
            ),
            "status": "PASS",
            "outcome": "natural_loss",
            "terminal_observation": receipt,
        }
    )
    return promoted


def _write_board_anchor_monitor(
    path: Path,
    trace_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive one transparent monitor row from the exact call-site hit."""

    calls = trace_result.get("calls")
    candidates = (
        [
            row
            for row in calls
            if isinstance(row, Mapping)
            and row.get("source_board_candidate") is True
        ]
        if isinstance(calls, list)
        else []
    )
    if (
        trace_result.get("schema")
        != "zuma-rl.pc-gameplay-mtrand-call-trace"
        or trace_result.get("version") != 1
        or trace_result.get("status") != "PASS"
        or trace_result.get("failure") is not None
        or trace_result.get("breakpoint_kind")
        not in BOARD_SOURCE_TRACE_KINDS
        or trace_result.get("address") != BOARD_GLOBAL_PRECALL_ADDRESS
        or trace_result.get("process_memory_writes") != 0
        or trace_result.get("process_memory_mutation") is not False
        or trace_result.get("persistent_file_modified") is not False
        or trace_result.get("hardware_breakpoint_restored") is not True
        or len(candidates) != 1
        or path.exists()
        or not path.parent.is_dir()
    ):
        raise RuntimeError("Board call-site trace cannot derive an anchor")
    row = candidates[0]
    integer_fields = (
        "order",
        "thread_id",
        "caller",
        "framework_update",
        "board_address",
        "native_game_time",
        "score",
        "score_target",
        "pre_index",
        "post_index",
        "output",
        "perf_counter_ns",
        "thread_crt_rand_state",
    )
    if any(
        isinstance(row.get(name), bool)
        or not isinstance(row.get(name), int)
        for name in integer_fields
    ):
        raise RuntimeError("Board call-site anchor fields are invalid")
    if (
        row.get("caller") != BOARD_GLOBAL_PRECALL_RETURN
        or row.get("call_address") != BOARD_GLOBAL_PRECALL_ADDRESS
        or row.get("call_instruction_hex") != "e86fbcfbff"
        or row.get("thread_crt_snapshot_error") is not None
        or not isinstance(row.get("thread_crt_state"), Mapping)
    ):
        raise RuntimeError("Board call-site anchor identity is invalid")
    pre_sha256 = row.get("pre_state_sha256")
    post_sha256 = row.get("post_state_sha256")
    if (
        not isinstance(pre_sha256, str)
        or not pre_sha256.startswith("sha256:")
        or len(pre_sha256) != 71
        or not isinstance(post_sha256, str)
        or not post_sha256.startswith("sha256:")
        or len(post_sha256) != 71
    ):
        raise RuntimeError("Board call-site state digest is invalid")
    process_id = trace_result.get("process_id")
    main_thread_id = trace_result.get("main_thread_id")
    if (
        isinstance(process_id, bool)
        or not isinstance(process_id, int)
        or process_id <= 0
        or main_thread_id != row["thread_id"]
    ):
        raise RuntimeError("Board call-site process identity is invalid")

    source_trace_kind = str(trace_result["breakpoint_kind"])
    header = {
        "schema": "zuma-rl.live-rng-change-monitor",
        "version": 1,
        "classification": (
            "read-only-board-global-precall-derived-anchor"
        ),
        "process_id": process_id,
        "process_memory_writes": 0,
        "source_trace_breakpoint_kind": source_trace_kind,
        "source_boundary_kind": BOARD_GLOBAL_PRECALL_KIND,
        "source_trace_call_order": row["order"],
        "thread_crt_state": dict(row["thread_crt_state"]),
    }
    monitor_row = {
        "type": "rng",
        "classification": (
            "exact-read-only-board-global-precall-state"
        ),
        "process_id": process_id,
        "framework_update": row["framework_update"],
        "native_game_time": row["native_game_time"],
        "score": row["score"],
        "score_target": row["score_target"],
        "board_address": row["board_address"],
        "global_mtrand_index": row["pre_index"],
        "global_mtrand_sha256": pre_sha256.removeprefix("sha256:"),
        "predicted_output": row["output"],
        "predicted_post_index": row["post_index"],
        "predicted_post_sha256": post_sha256.removeprefix("sha256:"),
        "thread_crt_rand_state": row["thread_crt_rand_state"],
        "source_call_order": row["order"],
        "source_caller": row["caller"],
        "source_call_address": row["call_address"],
        "perf_counter_ns": row["perf_counter_ns"],
    }
    with path.open("x", encoding="ascii", newline="\n") as stream:
        for value in (header, monitor_row):
            stream.write(
                json.dumps(
                    value,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            stream.write("\n")
    return {
        "exit_code": 0,
        "process_id": process_id,
        "ui_thread_id": main_thread_id,
        "thread_crt_capture_requested": True,
        "output": str(path),
        "output_bytes": path.stat().st_size,
        "output_sha256": _sha256(path),
        "stop_after_native_game_time": row["native_game_time"],
        "process_memory_writes": 0,
        "source": source_trace_kind,
        "source_boundary_kind": BOARD_GLOBAL_PRECALL_KIND,
        "source_call_order": row["order"],
        "source_caller": row["caller"],
        "source_framework_update": row["framework_update"],
    }


def _wait_until_update(
    pid: int,
    target: int,
    *,
    timeout: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        try:
            state = read_framework_state(pid)
            if int(state["framework_update"]) >= target:
                return state
        except (OSError, LiveBoardUnavailable) as error:
            last_error = error
        time.sleep(0.01)
    detail = f": {last_error}" if last_error is not None else ""
    raise TimeoutError(f"framework update {target} not reached{detail}")


def _validate_navigation_updates(
    title_start_update: int,
    adventure_update: int,
    continue_game_update: int,
) -> tuple[int, int, int]:
    values = (
        title_start_update,
        adventure_update,
        continue_game_update,
    )
    if any(isinstance(value, bool) or value < 0 for value in values):
        raise ValueError("navigation updates must be non-negative integers")
    if not title_start_update < adventure_update < continue_game_update:
        raise ValueError(
            "navigation updates must be strictly increasing: "
            "title start, adventure, continue game"
        )
    return values


def _validate_navigation_point(
    name: str,
    point: tuple[float, float],
) -> tuple[float, float]:
    if len(point) != 2:
        raise ValueError(f"{name} navigation point must contain x and y")
    x, y = point
    if (
        isinstance(x, bool)
        or isinstance(y, bool)
        or not math.isfinite(x)
        or not math.isfinite(y)
        or not 0.0 <= x < 800.0
        or not 0.0 <= y < 600.0
    ):
        raise ValueError(
            f"{name} navigation point must be inside the 800x600 canvas"
        )
    return float(x), float(y)


def _validate_prelaunch_cursor_screen_point(
    point: tuple[int, int],
) -> tuple[int, int]:
    if len(point) != 2:
        raise ValueError("prelaunch cursor point must contain x and y")
    x, y = point
    if (
        isinstance(x, bool)
        or isinstance(y, bool)
        or not -(2**31) <= x < 2**31
        or not -(2**31) <= y < 2**31
    ):
        raise ValueError("prelaunch cursor point must fit signed int32")
    return int(x), int(y)


def _park_cursor_before_launch(
    point: tuple[int, int],
) -> dict[str, Any]:
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.GetCursorPos.argtypes = (ctypes.POINTER(wintypes.POINT),)
    user32.GetCursorPos.restype = wintypes.BOOL
    user32.SetCursorPos.argtypes = (ctypes.c_int, ctypes.c_int)
    user32.SetCursorPos.restype = wintypes.BOOL

    before = wintypes.POINT()
    if not user32.GetCursorPos(ctypes.byref(before)):
        raise ctypes.WinError(ctypes.get_last_error())
    if not user32.SetCursorPos(point[0], point[1]):
        raise ctypes.WinError(ctypes.get_last_error())
    after = wintypes.POINT()
    if not user32.GetCursorPos(ctypes.byref(after)):
        raise ctypes.WinError(ctypes.get_last_error())
    observed = (int(after.x), int(after.y))
    if observed != point:
        raise RuntimeError(
            f"prelaunch cursor parking mismatch: {observed} != {point}"
        )
    return {
        "before_process_launch": True,
        "transport": "Win32 SetCursorPos",
        "before_screen_point": [int(before.x), int(before.y)],
        "requested_screen_point": list(point),
        "observed_screen_point": list(observed),
        "process_memory_writes": 0,
    }


def _capture_navigation_snapshot(
    output_root: Path,
    stage: str,
) -> dict[str, Any]:
    output = output_root / f"navigation-before-{stage}.bmp"
    try:
        return snapshot(
            output=output,
            process_name="popcapgame1.exe",
            device_index=0,
            output_index=0,
            timeout=2.0,
        )
    except Exception as error:
        return {
            "error": f"{type(error).__name__}: {error}",
            "output": str(output),
        }


def _normal_close(window_handle: int, process: _ManagedProcess) -> bool:
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.PostMessageW.argtypes = (
        wintypes.HWND,
        wintypes.UINT,
        wintypes.WPARAM,
        wintypes.LPARAM,
    )
    user32.PostMessageW.restype = wintypes.BOOL
    if not user32.PostMessageW(
        wintypes.HWND(window_handle),
        0x0010,
        0,
        0,
    ):
        raise OSError(ctypes.get_last_error(), "WM_CLOSE failed")
    try:
        return process.wait(timeout=10.0) == 0
    except subprocess.TimeoutExpired:
        return False


def _exit_after_natural_win(
    window_handle: int,
    process: _ManagedProcess,
    diagnostic_root: Path,
) -> tuple[list[dict[str, Any]], bool]:
    """Leave through retail UI so the ``-record`` DMO is committed."""

    steps = (
        # Adventure Continue advances directly into the next board.  The
        # remaining coordinates reproduce the exact normal-exit route from
        # the unmodified human reference recording:
        # in-game MENU -> Main Menu -> confirm -> Quit -> confirm.
        ("score_continue", 540.0, 560.0, 10.0),
        ("next_level_menu", 741.0, 16.0, 2.0),
        ("pause_main_menu", 262.0, 455.0, 1.25),
        ("return_to_main_confirm", 299.0, 396.0, 2.0),
        ("main_quit", 704.0, 516.0, 1.5),
        ("quit_confirm", 309.0, 399.0, 0.0),
    )
    evidence: list[dict[str, Any]] = []
    # The Board reaches its natural-win terminal state before the retail
    # "Level Aced" animation and the result-screen score count-up finish.
    # Inputs during either phase are ignored (or can merely skip the count-up),
    # so wait for the actual result screen to become fully interactive.
    time.sleep(13.0)
    for index, (name, x, y, delay_after) in enumerate(steps):
        if process.poll() is not None:
            break
        diagnostic_path = (
            diagnostic_root / f"exit-{index:02d}-before-{name}.bmp"
        )
        try:
            diagnostic = snapshot(
                output=diagnostic_path,
                process_name="popcapgame1.exe",
                device_index=0,
                output_index=0,
                timeout=2.0,
            )
        except Exception as error:
            diagnostic = {"error": str(error)}
        evidence.append(
            {
                "stage": name,
                "perf_counter_ns": time.perf_counter_ns(),
                "diagnostic_snapshot": diagnostic,
                "input": send_mouse_click(
                    window_handle,
                    logical_x=x,
                    logical_y=y,
                    button="left",
                ),
            }
        )
        if delay_after:
            time.sleep(delay_after)
    try:
        return evidence, process.wait(timeout=20.0) == 0
    except subprocess.TimeoutExpired:
        return evidence, False


def _exit_after_natural_loss_restart(
    window_handle: int,
    process: _ManagedProcess,
    diagnostic_root: Path,
) -> tuple[list[dict[str, Any]], bool]:
    """Leave a replacement Board through the normal retail menu route."""

    steps = (
        ("replacement_board_menu", 741.0, 16.0, 2.0),
        ("pause_main_menu", 262.0, 455.0, 1.25),
        ("return_to_main_confirm", 299.0, 396.0, 2.0),
        ("main_quit", 704.0, 516.0, 1.5),
        ("quit_confirm", 309.0, 399.0, 0.0),
    )
    evidence: list[dict[str, Any]] = []
    # The Board-only detector stops as soon as the replacement plan is present,
    # before Shooter reconstruction is observable.  The c142 source trace
    # reached the first established replacement-board frame about 4.3 seconds
    # later, so the preregistered five-second settle keeps exit clicks outside
    # the teardown animation while preserving the normal retail menu route.
    time.sleep(5.0)
    for index, (name, x, y, delay_after) in enumerate(steps):
        if process.poll() is not None:
            break
        diagnostic_path = (
            diagnostic_root / f"exit-{index:02d}-before-{name}.bmp"
        )
        try:
            diagnostic = snapshot(
                output=diagnostic_path,
                process_name="popcapgame1.exe",
                device_index=0,
                output_index=0,
                timeout=2.0,
            )
        except Exception as error:
            diagnostic = {"error": str(error)}
        evidence.append(
            {
                "stage": name,
                "perf_counter_ns": time.perf_counter_ns(),
                "diagnostic_snapshot": diagnostic,
                "input": send_mouse_click(
                    window_handle,
                    logical_x=x,
                    logical_y=y,
                    button="left",
                ),
            }
        )
        if delay_after:
            time.sleep(delay_after)
    try:
        return evidence, process.wait(timeout=20.0) == 0
    except subprocess.TimeoutExpired:
        return evidence, False


def record_autoplay(
    *,
    output_root: Path,
    runtime_executable: Path,
    changedir: Path,
    prestate_dir: Path,
    expected_score_target: int,
    minimum_entry_score: int,
    maximum_gameplay_seconds: float,
    expected_outcome: str = "natural_win",
    gameplay_policy: str = "autoplay",
    fruit_policy: str = "ignore",
    crt_rand_seed: int | None = None,
    rng_monitor_stop_native_time: int | None = None,
    gameplay_mtrand_trace: bool = False,
    mtrand_trace_kind: str = "gameplay_return",
    title_start_update: int = DEFAULT_TITLE_START_UPDATE,
    adventure_update: int = DEFAULT_ADVENTURE_UPDATE,
    continue_game_update: int = DEFAULT_CONTINUE_GAME_UPDATE,
    title_start_point: tuple[float, float] = DEFAULT_TITLE_START_POINT,
    adventure_point: tuple[float, float] = DEFAULT_ADVENTURE_POINT,
    continue_game_point: tuple[float, float] = DEFAULT_CONTINUE_GAME_POINT,
    prelaunch_cursor_screen_point: tuple[int, int] | None = None,
) -> dict[str, Any]:
    if os.name != "nt":
        raise RuntimeError("retail recording requires Windows")
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite: {output_root}")
    if not runtime_executable.is_file():
        raise FileNotFoundError(runtime_executable)
    if not changedir.is_dir():
        raise FileNotFoundError(changedir)
    if not prestate_dir.is_dir():
        raise FileNotFoundError(prestate_dir)
    if expected_score_target <= 0 or not (
        0 <= minimum_entry_score < expected_score_target
    ):
        raise ValueError("score expectations are invalid")
    if (
        expected_outcome not in {"natural_loss", "natural_win"}
        or gameplay_policy not in {"autoplay", "idle"}
        or fruit_policy not in {"ignore", "collect", "bomb"}
    ):
        raise ValueError("recording outcome contract is invalid")
    if (
        crt_rand_seed is not None
        and (
            isinstance(crt_rand_seed, bool)
            or not 0 <= crt_rand_seed <= 0xFFFFFFFF
        )
    ):
        raise ValueError("C rand seed must fit uint32")
    if (
        rng_monitor_stop_native_time is not None
        and rng_monitor_stop_native_time < 0
    ):
        raise ValueError("RNG monitor native stop must not be negative")
    if mtrand_trace_kind not in {
        "gameplay_return",
        "global_wrapper_entry",
        BOARD_GLOBAL_PRECALL_KIND,
        BOARD_GLOBAL_PRECALL_AND_RESET_KIND,
        BOARD_GLOBAL_PRECALL_AND_TARGET_WRITE_KIND,
        BOARD_GLOBAL_PRECALL_AND_NATURAL_LOSS_TARGET_WRITE_KIND,
    }:
        raise ValueError("MTRand trace kind is invalid")
    if (
        mtrand_trace_kind in BOARD_SOURCE_TRACE_KINDS
        and rng_monitor_stop_native_time is not None
    ):
        raise ValueError(
            "Board call-site tracing derives its own exact anchor monitor"
        )
    if mtrand_trace_kind == (
        BOARD_GLOBAL_PRECALL_AND_NATURAL_LOSS_TARGET_WRITE_KIND
    ) and (
        not gameplay_mtrand_trace
        or expected_outcome != "natural_loss"
        or gameplay_policy != "idle"
        or fruit_policy != "ignore"
    ):
        raise ValueError(
            "natural-loss target-write tracing requires the frozen idle "
            "recording contract"
        )
    navigation_updates = _validate_navigation_updates(
        title_start_update,
        adventure_update,
        continue_game_update,
    )
    navigation_points = {
        "title_start": _validate_navigation_point(
            "title start",
            title_start_point,
        ),
        "adventure": _validate_navigation_point(
            "adventure",
            adventure_point,
        ),
        "continue_game": _validate_navigation_point(
            "continue game",
            continue_game_point,
        ),
    }
    parked_cursor_point = (
        None
        if prelaunch_cursor_screen_point is None
        else _validate_prelaunch_cursor_screen_point(
            prelaunch_cursor_screen_point
        )
    )
    runtime_sha256 = _sha256(runtime_executable)
    if runtime_sha256 != EXPECTED_RETAIL_RUNTIME_SHA256:
        raise RuntimeError(
            "runtime executable is not the verified retail payload: "
            f"{runtime_sha256}"
        )
    _ensure_dpi_awareness()
    existing = process_ids_by_name(runtime_executable.name)
    if existing:
        raise RuntimeError(f"existing retail runtime(s): {existing}")

    output_root.mkdir(parents=False)
    safety = output_root / "safety"
    safety.mkdir()
    inputs = output_root / "inputs"
    inputs.mkdir()
    copied_prestate = inputs / "prestate"
    copied_prestate.mkdir()
    replacement_paths: list[tuple[str, Path]] = []
    prestate_rows: list[dict[str, Any]] = []
    for name in ("users.dat", "user2.dat", "adv_in_game2.sav"):
        source = prestate_dir / name
        provenance_source = prestate_dir / f"{name}.provenance.json"
        if not source.is_file() or not provenance_source.is_file():
            raise FileNotFoundError(
                f"prestate artifact or provenance is missing: {name}"
            )
        destination = copied_prestate / name
        provenance_destination = copied_prestate / (
            f"{name}.provenance.json"
        )
        shutil.copy2(source, destination)
        shutil.copy2(provenance_source, provenance_destination)
        replacement_paths.append((name, destination))
        prestate_rows.append(
            {
                "relative_path": name,
                "bytes": destination.stat().st_size,
                "sha256": _sha256(destination),
                "provenance_sha256": _sha256(provenance_destination),
            }
        )
    dmo = output_root / "input.dmo"
    actions = output_root / "autoplay.ndjson"
    session_nonce = secrets.token_hex(16)
    host_pre = capture_state(
        session_nonce=session_nonce,
        phase="host-pre",
    )
    write_snapshot_exclusive(host_pre, safety / "host-pre.json")
    record_pre = overlay_snapshot_files(
        host_pre,
        replacement_paths,
        phase="record-pre",
        captured_perf_counter_ns=time.perf_counter_ns(),
    )
    write_snapshot_exclusive(record_pre, safety / "record-pre.json")

    command = [
        str(runtime_executable),
        "-record",
        f"-demofile={dmo}",
        f"-changedir={changedir.resolve()}\\",
    ]
    environment = os.environ.copy()
    environment["SteamAppId"] = "3620"
    environment["SteamGameId"] = "3620"
    environment["__COMPAT_LAYER"] = "HIGHDPIAWARE"
    launched: _ManagedProcess | None = None
    startup_rng = None
    rng_monitor: subprocess.Popen[bytes] | None = None
    rng_monitor_stdout = None
    rng_monitor_stderr = None
    gameplay_mtrand: subprocess.Popen[bytes] | None = None
    gameplay_mtrand_stdout = None
    gameplay_mtrand_stderr = None
    gameplay_mtrand_output: Path | None = None
    gameplay_mtrand_ready: Path | None = None
    gameplay_mtrand_stop: Path | None = None
    window_handle: int | None = None
    main_thread_id: int | None = None
    result: dict[str, Any] = {
        "schema": "zuma-rl.retail-autoplay-recording",
        "version": 2,
        "classification": "candidate-recording-not-pc-golden",
        "status": "FAILED",
        "output_root": str(output_root),
        "runtime_executable": str(runtime_executable),
        "runtime_executable_bytes": runtime_executable.stat().st_size,
        "runtime_executable_sha256": runtime_sha256,
        "changedir": str(changedir),
        "prestate_source": str(prestate_dir),
        "prestate": prestate_rows,
        "record_pre_state_root": record_pre.state_root,
        "expected_score_target": expected_score_target,
        "minimum_entry_score": minimum_entry_score,
        "expected_outcome": expected_outcome,
        "gameplay_policy": gameplay_policy,
        "fruit_policy": fruit_policy,
        "dmo": str(dmo),
        "session_nonce": session_nonce,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "host_pre_state_root": host_pre.state_root,
        "process_memory_writes": 0,
        "input_transport": "Win32 SendInput",
        "controlled_crt_rand_seed": crt_rand_seed,
        "rng_monitor_stop_native_time": rng_monitor_stop_native_time,
        "gameplay_mtrand_trace_requested": gameplay_mtrand_trace,
        "mtrand_trace_kind": mtrand_trace_kind,
        "navigation_plan": {
            "title_start": {
                "framework_update": navigation_updates[0],
                "logical_point": list(navigation_points["title_start"]),
            },
            "adventure": {
                "framework_update": navigation_updates[1],
                "logical_point": list(navigation_points["adventure"]),
            },
            "continue_game": {
                "framework_update": navigation_updates[2],
                "logical_point": list(navigation_points["continue_game"]),
            },
        },
        "prelaunch_cursor_screen_point": (
            None
            if parked_cursor_point is None
            else list(parked_cursor_point)
        ),
    }
    try:
        applied_pre = restore_state(record_pre)
        if applied_pre.state_root != record_pre.state_root:
            raise RuntimeError("recording prestate restore did not verify")
        result["applied_pre_state_root"] = applied_pre.state_root
        if parked_cursor_point is not None:
            result["prelaunch_cursor"] = _park_cursor_before_launch(
                parked_cursor_point
            )
        if crt_rand_seed is None:
            launched = subprocess.Popen(
                command,
                cwd=changedir,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
            )
        else:
            startup_rng = launch_fixed_seed_replay_iat_stub(
                runtime_executable=runtime_executable,
                changedir=changedir,
                dmo=dmo,
                app_id=3620,
                seed=crt_rand_seed,
                launch_mode="record",
            )
            launched = _PidProcess(startup_rng.process_id)
            result["startup_rng"] = startup_rng.to_dict()
        result["process_id"] = launched.pid
        window_handle = wait_for_pid_window(
            pid=launched.pid,
            runtime_exe=runtime_executable,
            timeout=30.0,
        )
        result["window_handle_hex"] = f"0x{window_handle:016x}"
        main_thread_id = (
            startup_rng.thread_id
            if startup_rng is not None
            else _window_thread_id(window_handle, launched.pid)
        )
        result["main_thread"] = {
            "thread_id": main_thread_id,
            "source": (
                "controlled_startup_handoff"
                if startup_rng is not None
                else "verified_retail_window"
            ),
            "process_memory_writes": 0,
        }
        if gameplay_mtrand_trace:
            assert main_thread_id is not None
            trace_stem = {
                "global_wrapper_entry": "global-mtrand-calls",
                "board_global_precall": "board-global-precall",
                "board_global_precall_and_reset": (
                    "board-global-precall-and-reset"
                ),
                "board_global_precall_and_target_write": (
                    "board-global-precall-and-target-write"
                ),
                "board_global_precall_and_natural_loss_target_write": (
                    "board-global-precall-and-natural-loss-target-write"
                ),
                "gameplay_return": "gameplay-mtrand-calls",
            }[mtrand_trace_kind]
            gameplay_mtrand_output = output_root / f"{trace_stem}.json"
            gameplay_mtrand_ready = (
                output_root / f"{trace_stem}.ready.json"
            )
            gameplay_mtrand_stop = output_root / f"{trace_stem}.stop"
            gameplay_mtrand_stdout_path = (
                output_root / f"{trace_stem}.log"
            )
            gameplay_mtrand_stderr_path = (
                output_root / f"{trace_stem}.err.log"
            )
            gameplay_mtrand_stdout = (
                gameplay_mtrand_stdout_path.open("xb")
            )
            gameplay_mtrand_stderr = (
                gameplay_mtrand_stderr_path.open("xb")
            )
            gameplay_mtrand = subprocess.Popen(
                [
                    sys.executable,
                    str(
                        Path(__file__).resolve().parent
                        / "trace_popcap_gameplay_mtrand.py"
                    ),
                    "--pid",
                    str(launched.pid),
                    "--main-thread-id",
                    str(main_thread_id),
                    "--executable",
                    str(runtime_executable),
                    "--output",
                    str(gameplay_mtrand_output),
                    "--ready",
                    str(gameplay_mtrand_ready),
                    "--stop",
                    str(gameplay_mtrand_stop),
                    "--breakpoint-kind",
                    mtrand_trace_kind,
                    "--attach-timeout",
                    "60",
                    "--timeout",
                    str(maximum_gameplay_seconds + 180.0),
                    "--maximum-calls",
                    (
                        "64"
                        if mtrand_trace_kind in BOARD_SOURCE_TRACE_KINDS
                        else "100000"
                    ),
                ],
                cwd=Path(__file__).resolve().parents[1],
                env=os.environ.copy(),
                stdin=subprocess.DEVNULL,
                stdout=gameplay_mtrand_stdout,
                stderr=gameplay_mtrand_stderr,
                close_fds=True,
            )
            result["gameplay_mtrand_trace"] = {
                "process_id": gameplay_mtrand.pid,
                "thread_id": main_thread_id,
                "output": str(gameplay_mtrand_output),
                "ready": str(gameplay_mtrand_ready),
                "stop": str(gameplay_mtrand_stop),
                "stdout": str(gameplay_mtrand_stdout_path),
                "stderr": str(gameplay_mtrand_stderr_path),
                "process_memory_writes": 0,
                "breakpoint_kind": mtrand_trace_kind,
            }
            ready_deadline = time.monotonic() + 30.0
            while (
                time.monotonic() < ready_deadline
                and not gameplay_mtrand_ready.is_file()
                and gameplay_mtrand.poll() is None
            ):
                time.sleep(0.02)
            if not gameplay_mtrand_ready.is_file():
                raise RuntimeError(
                    "gameplay MTRand tracer did not become ready"
                )
        if rng_monitor_stop_native_time is not None:
            assert main_thread_id is not None
            rng_output = output_root / "rng.ndjson"
            rng_stdout_path = output_root / "rng-monitor.log"
            rng_stderr_path = output_root / "rng-monitor.err.log"
            rng_monitor_stdout = rng_stdout_path.open("xb")
            rng_monitor_stderr = rng_stderr_path.open("xb")
            rng_monitor_command = [
                sys.executable,
                str(
                    Path(__file__).resolve().parent
                    / "monitor_live_zuma_rng.py"
                ),
                "--pid",
                str(launched.pid),
                "--output",
                str(rng_output),
                "--interval-seconds",
                "0.002",
                "--maximum-seconds",
                "180",
                "--stop-after-native-game-time",
                str(rng_monitor_stop_native_time),
            ]
            if startup_rng is not None:
                rng_monitor_command += [
                    "--thread-id",
                    str(main_thread_id),
                ]
            rng_monitor = subprocess.Popen(
                rng_monitor_command,
                cwd=Path(__file__).resolve().parents[1],
                env=os.environ.copy(),
                stdin=subprocess.DEVNULL,
                stdout=rng_monitor_stdout,
                stderr=rng_monitor_stderr,
                close_fds=True,
            )
            result["rng_monitor"] = {
                "process_id": rng_monitor.pid,
                "ui_thread_id": main_thread_id,
                "thread_crt_capture_requested": startup_rng is not None,
                "output": str(rng_output),
                "stdout": str(rng_stdout_path),
                "stderr": str(rng_stderr_path),
                "stop_after_native_game_time": (
                    rng_monitor_stop_native_time
                ),
                "process_memory_writes": 0,
            }

        navigation: list[dict[str, Any]] = []
        startup = _wait_until_update(
            launched.pid,
            title_start_update,
            timeout=45.0,
        )
        title_start_input = send_mouse_click(
            window_handle,
            logical_x=navigation_points["title_start"][0],
            logical_y=navigation_points["title_start"][1],
            button="left",
        )
        navigation.append(
            {
                "stage": "title_start",
                "target_framework_update": title_start_update,
                "framework_update": startup["framework_update"],
                "input": title_start_input,
                "diagnostic_snapshot_after_input": _capture_navigation_snapshot(
                    output_root,
                    "title-start",
                ),
            }
        )
        main_menu = _wait_until_update(
            launched.pid,
            adventure_update,
            timeout=45.0,
        )
        adventure_input = send_mouse_click(
            window_handle,
            logical_x=navigation_points["adventure"][0],
            logical_y=navigation_points["adventure"][1],
            button="left",
        )
        navigation.append(
            {
                "stage": "adventure",
                "target_framework_update": adventure_update,
                "framework_update": main_menu["framework_update"],
                "input": adventure_input,
                "diagnostic_snapshot_after_input": _capture_navigation_snapshot(
                    output_root,
                    "adventure",
                ),
            }
        )
        map_state = _wait_until_update(
            launched.pid,
            continue_game_update,
            timeout=45.0,
        )
        continue_game_input = send_mouse_click(
            window_handle,
            logical_x=navigation_points["continue_game"][0],
            logical_y=navigation_points["continue_game"][1],
            button="left",
        )
        navigation.append(
            {
                "stage": "continue_game",
                "target_framework_update": continue_game_update,
                "framework_update": map_state["framework_update"],
                "input": continue_game_input,
                "diagnostic_snapshot_after_input": _capture_navigation_snapshot(
                    output_root,
                    "continue-game",
                ),
            }
        )
        result["navigation"] = navigation

        board_deadline = time.monotonic() + 30.0
        last_board_contract: tuple[int, int] | None = None
        while True:
            if time.monotonic() >= board_deadline:
                raise TimeoutError(
                    "expected active retail Board did not appear: "
                    f"{last_board_contract!r}"
                )
            try:
                board = read_live_board(launched.pid)
                last_board_contract = (
                    int(board["score"]),
                    int(board["score_target"]),
                )
                if (
                    int(board["score_target"]) != expected_score_target
                    or int(board["score"]) < minimum_entry_score
                ):
                    time.sleep(0.02)
                    continue
                result["board_entry"] = {
                    key: board[key]
                    for key in (
                        "native_game_time",
                        "score",
                        "displayed_score",
                        "score_target",
                        "runtime_active",
                        "active_ball_count",
                        "pending_ball_count",
                    )
                }
                break
            except LiveBoardUnavailable:
                time.sleep(0.02)

        gameplay = autoplay(
            launched.pid,
            output_path=actions,
            maximum_seconds=maximum_gameplay_seconds,
            shot_interval_seconds=0.16,
            swap_delay_seconds=0.04,
            idle_poll_seconds=0.01,
            expected_outcome=expected_outcome,
            gameplay_policy=gameplay_policy,
            fruit_policy=fruit_policy,
        )
        result["gameplay"] = gameplay
        if gameplay_mtrand is not None:
            if (
                gameplay_mtrand_stop is None
                or gameplay_mtrand_output is None
            ):
                raise RuntimeError(
                    "gameplay MTRand trace paths are unavailable"
                )
            gameplay_mtrand_stop.touch(exist_ok=False)
            try:
                trace_exit = gameplay_mtrand.wait(timeout=30.0)
            except subprocess.TimeoutExpired as error:
                raise RuntimeError(
                    "gameplay MTRand tracer did not stop"
                ) from error
            result["gameplay_mtrand_trace"]["exit_code"] = trace_exit
            if not gameplay_mtrand_output.is_file():
                raise RuntimeError(
                    "gameplay MTRand trace result is missing"
                )
            trace_result = json.loads(
                gameplay_mtrand_output.read_text(encoding="ascii")
            )
            trace_output_sha256 = _sha256(gameplay_mtrand_output)
            result["gameplay_mtrand_trace"].update(
                {
                    "status": trace_result.get("status"),
                    "failure": trace_result.get("failure"),
                    "call_count": trace_result.get("call_count"),
                    "hardware_breakpoint_restored": trace_result.get(
                        "hardware_breakpoint_restored"
                    ),
                    "hardware_breakpoint_restore_error": trace_result.get(
                        "hardware_breakpoint_restore_error"
                    ),
                    "debugger_detach_error": trace_result.get(
                        "debugger_detach_error"
                    ),
                    "process_memory_mutation": trace_result.get(
                        "process_memory_mutation"
                    ),
                    "persistent_file_modified": trace_result.get(
                        "persistent_file_modified"
                    ),
                    "source_anchor_count": trace_result.get(
                        "source_anchor_count"
                    ),
                    "target_write_count": trace_result.get(
                        "target_write_count"
                    ),
                    "stop_reason": trace_result.get("stop_reason"),
                    "output_bytes": (
                        gameplay_mtrand_output.stat().st_size
                    ),
                    "output_sha256": trace_output_sha256,
                }
            )
            if (
                trace_exit != 0
                or trace_result.get("status") != "PASS"
                or int(trace_result.get("call_count", 0)) <= 0
                or not trace_result.get(
                    "hardware_breakpoint_restored"
                )
            ):
                raise RuntimeError(
                    "gameplay MTRand trace structured result failed"
                )
            if mtrand_trace_kind == (
                BOARD_GLOBAL_PRECALL_AND_NATURAL_LOSS_TARGET_WRITE_KIND
            ):
                gameplay = _promote_natural_loss_target_write_gameplay(
                    gameplay,
                    trace_result,
                    trace_sha256=trace_output_sha256,
                )
                result["gameplay"] = gameplay
            if mtrand_trace_kind in BOARD_SOURCE_TRACE_KINDS:
                result["rng_monitor"] = _write_board_anchor_monitor(
                    output_root / "rng.ndjson",
                    trace_result,
                )
        if gameplay["outcome"] == "natural_win":
            exit_inputs, normal_exit = _exit_after_natural_win(
                window_handle,
                launched,
                output_root,
            )
            result["exit_inputs"] = exit_inputs
            result["normal_exit"] = normal_exit
            result["exit_method"] = "retail_ui"
        elif gameplay["outcome"] == "natural_loss":
            exit_inputs, normal_exit = _exit_after_natural_loss_restart(
                window_handle,
                launched,
                output_root,
            )
            result["exit_inputs"] = exit_inputs
            result["normal_exit"] = normal_exit
            result["exit_method"] = "retail_ui"
        else:
            result["normal_close_requested"] = True
            result["normal_exit"] = _normal_close(
                window_handle,
                launched,
            )
            result["exit_method"] = "retail_window_close"
        result["process_exit_code"] = launched.poll()
        if not result["normal_exit"]:
            raise RuntimeError("retail runtime did not accept normal close")
        if not dmo.is_file() or dmo.stat().st_size == 0:
            raise RuntimeError("retail runtime did not commit the DMO")
        result["dmo_bytes"] = dmo.stat().st_size
        result["dmo_sha256"] = _sha256(dmo)
        result["autoplay_log_bytes"] = actions.stat().st_size
        result["autoplay_log_sha256"] = _sha256(actions)
        result["status"] = (
            "PASS"
            if (
                gameplay["status"] == "PASS"
                and gameplay["outcome"] == expected_outcome
                and result["exit_method"] == "retail_ui"
            )
            else "INCOMPLETE"
        )
    except Exception as error:
        result["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        raise
    finally:
        if gameplay_mtrand is not None:
            if gameplay_mtrand.poll() is None:
                if (
                    gameplay_mtrand_stop is not None
                    and not gameplay_mtrand_stop.exists()
                ):
                    gameplay_mtrand_stop.touch(exist_ok=False)
                try:
                    gameplay_mtrand.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    gameplay_mtrand.terminate()
                    try:
                        gameplay_mtrand.wait(timeout=10.0)
                    except subprocess.TimeoutExpired:
                        gameplay_mtrand.kill()
                        gameplay_mtrand.wait(timeout=10.0)
            if "gameplay_mtrand_trace" in result:
                result["gameplay_mtrand_trace"]["exit_code"] = (
                    gameplay_mtrand.returncode
                )
        if gameplay_mtrand_stdout is not None:
            gameplay_mtrand_stdout.close()
        if gameplay_mtrand_stderr is not None:
            gameplay_mtrand_stderr.close()
        if (
            gameplay_mtrand_output is not None
            and gameplay_mtrand_output.is_file()
            and "gameplay_mtrand_trace" in result
        ):
            result["gameplay_mtrand_trace"].setdefault(
                "output_bytes",
                gameplay_mtrand_output.stat().st_size,
            )
            result["gameplay_mtrand_trace"].setdefault(
                "output_sha256",
                _sha256(gameplay_mtrand_output),
            )
        if rng_monitor is not None:
            if rng_monitor.poll() is None:
                rng_monitor.terminate()
                try:
                    rng_monitor.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    rng_monitor.kill()
                    rng_monitor.wait(timeout=10.0)
            result["rng_monitor"]["exit_code"] = rng_monitor.returncode
        if rng_monitor_stdout is not None:
            rng_monitor_stdout.close()
        if rng_monitor_stderr is not None:
            rng_monitor_stderr.close()
        rng_output = output_root / "rng.ndjson"
        if rng_output.is_file():
            result["rng_monitor"]["output_bytes"] = (
                rng_output.stat().st_size
            )
            result["rng_monitor"]["output_sha256"] = _sha256(
                rng_output
            )
        if launched is not None and launched.poll() is None:
            launched.terminate()
            try:
                launched.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                launched.kill()
                launched.wait(timeout=10.0)
        if isinstance(launched, _PidProcess):
            launched.close()
        host_end = capture_state(
            session_nonce=session_nonce,
            phase="host-end",
        )
        write_snapshot_exclusive(host_end, safety / "host-end.json")
        restored = restore_state(host_pre)
        write_snapshot_exclusive(
            restored,
            safety / "host-restored.json",
        )
        result["host_end_state_root"] = host_end.state_root
        result["host_restored_state_root"] = restored.state_root
        result["host_restored"] = restored.state_root == host_pre.state_root
        result["finished_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json_exclusive(output_root / "result.json", result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--runtime-executable", required=True, type=Path)
    parser.add_argument("--changedir", required=True, type=Path)
    parser.add_argument("--prestate-dir", required=True, type=Path)
    parser.add_argument("--expected-score-target", type=int, default=9650)
    parser.add_argument("--minimum-entry-score", type=int, default=7000)
    parser.add_argument(
        "--maximum-gameplay-seconds",
        type=float,
        default=900.0,
    )
    parser.add_argument(
        "--expected-outcome",
        choices=("natural_loss", "natural_win"),
        default="natural_win",
    )
    parser.add_argument(
        "--gameplay-policy",
        choices=("autoplay", "idle"),
        default="autoplay",
    )
    parser.add_argument(
        "--fruit-policy",
        choices=("ignore", "collect", "bomb"),
        default="ignore",
        help=(
            "Optionally record ordinary projectile fruit attempts or "
            "nearby active-chain proximity-bomb attempts."
        ),
    )
    parser.add_argument(
        "--crt-rand-seed",
        type=int,
        help=(
            "Fix only the startup srand(GetTickCount()) seed while "
            "recording; the temporary process IAT patch is restored at "
            "framework update one."
        ),
    )
    parser.add_argument(
        "--rng-monitor-stop-native-time",
        type=int,
        help=(
            "Capture read-only Board, QRand, global MT, and main-thread "
            "CRT rand state until this native game time."
        ),
    )
    parser.add_argument(
        "--gameplay-mtrand-trace",
        action="store_true",
        help=(
            "Capture the exact main-thread gameplay MTRand return "
            "sequence with a read-only hardware-breakpoint tracer."
        ),
    )
    parser.add_argument(
        "--mtrand-trace-kind",
        choices=(
            "gameplay_return",
            "global_wrapper_entry",
            BOARD_GLOBAL_PRECALL_KIND,
            BOARD_GLOBAL_PRECALL_AND_RESET_KIND,
            BOARD_GLOBAL_PRECALL_AND_TARGET_WRITE_KIND,
            BOARD_GLOBAL_PRECALL_AND_NATURAL_LOSS_TARGET_WRITE_KIND,
        ),
        default="gameplay_return",
        help=(
            "Select the exact retail MTRand instruction boundary to "
            "trace when --gameplay-mtrand-trace is enabled."
        ),
    )
    parser.add_argument(
        "--title-start-update",
        type=int,
        default=DEFAULT_TITLE_START_UPDATE,
        help="Absolute framework update for the title-screen Start click.",
    )
    parser.add_argument(
        "--adventure-update",
        type=int,
        default=DEFAULT_ADVENTURE_UPDATE,
        help="Absolute framework update for the Adventure click.",
    )
    parser.add_argument(
        "--continue-game-update",
        type=int,
        default=DEFAULT_CONTINUE_GAME_UPDATE,
        help="Absolute framework update for the map Continue click.",
    )
    parser.add_argument(
        "--title-start-point",
        nargs=2,
        type=float,
        metavar=("X", "Y"),
        default=DEFAULT_TITLE_START_POINT,
        help="Logical 800x600 point for the title-screen Start click.",
    )
    parser.add_argument(
        "--adventure-point",
        nargs=2,
        type=float,
        metavar=("X", "Y"),
        default=DEFAULT_ADVENTURE_POINT,
        help="Logical 800x600 point for the Adventure click.",
    )
    parser.add_argument(
        "--continue-game-point",
        nargs=2,
        type=float,
        metavar=("X", "Y"),
        default=DEFAULT_CONTINUE_GAME_POINT,
        help="Logical 800x600 point for the map Continue click.",
    )
    parser.add_argument(
        "--prelaunch-cursor-screen-point",
        nargs=2,
        type=int,
        metavar=("X", "Y"),
        help=(
            "Park the system cursor at this desktop point before the retail "
            "process starts, preventing an update-zero mouse-enter event."
        ),
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = record_autoplay(
        output_root=args.output_root.resolve(),
        runtime_executable=args.runtime_executable.resolve(),
        changedir=args.changedir.resolve(),
        prestate_dir=args.prestate_dir.resolve(),
        expected_score_target=args.expected_score_target,
        minimum_entry_score=args.minimum_entry_score,
        maximum_gameplay_seconds=args.maximum_gameplay_seconds,
        expected_outcome=args.expected_outcome,
        gameplay_policy=args.gameplay_policy,
        fruit_policy=args.fruit_policy,
        crt_rand_seed=args.crt_rand_seed,
        rng_monitor_stop_native_time=(
            args.rng_monitor_stop_native_time
        ),
        gameplay_mtrand_trace=args.gameplay_mtrand_trace,
        mtrand_trace_kind=args.mtrand_trace_kind,
        title_start_update=args.title_start_update,
        adventure_update=args.adventure_update,
        continue_game_update=args.continue_game_update,
        title_start_point=tuple(args.title_start_point),
        adventure_point=tuple(args.adventure_point),
        continue_game_point=tuple(args.continue_game_point),
        prelaunch_cursor_screen_point=(
            None
            if args.prelaunch_cursor_screen_point is None
            else tuple(args.prelaunch_cursor_screen_point)
        ),
    )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
