"""Validate one captured retail main-thread CRT rand state for restore."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from zuma_rl.retail_dmo_provenance import (
    RetailDmoProvenanceError,
    certifying_recording_outcome,
)


@dataclass(frozen=True, slots=True)
class ThreadCrtRestoreState:
    source_path: Path
    source_sha256: str
    source_process_id: int
    source_thread_id: int
    source_framework_update: int
    source_native_game_time: int
    captured_state: int
    rewind_draws: int
    state: int
    source_kind: str = "live_rng_monitor"
    source_recording_report_path: Path | None = None
    source_recording_report_sha256: str | None = None
    source_dmo_path: Path | None = None
    source_dmo_sha256: str | None = None
    source_call_order: int | None = None
    source_caller: int | None = None

    def semantic_dict(self) -> dict[str, int]:
        return {
            "captured_state": self.captured_state,
            "rewind_draws": self.rewind_draws,
            "state": self.state,
        }

    @property
    def semantic_sha256(self) -> str:
        payload = json.dumps(
            self.semantic_dict(),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        return "sha256:" + hashlib.sha256(payload).hexdigest()


def _strict_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context} must be an integer")
    return value


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _load_json_object(path: Path, context: str) -> tuple[bytes, Mapping[str, Any]]:
    data = path.read_bytes()
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{context} is not valid JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a JSON object")
    return data, value


def _resolved_path(value: Any, context: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} path is invalid")
    return Path(value).resolve()


def load_thread_crt_restore_state(
    path: Path,
    *,
    framework_update: int,
    rewind_draws: int = 0,
) -> ThreadCrtRestoreState:
    """Load exactly one thread-aware read-only monitor row."""

    if framework_update < 0:
        raise ValueError("thread CRT source update must not be negative")
    if rewind_draws < 0:
        raise ValueError("thread CRT rewind draws must not be negative")
    data = path.read_bytes()
    source_sha256 = "sha256:" + hashlib.sha256(data).hexdigest()
    try:
        lines = data.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("thread CRT monitor log is not UTF-8") from error
    if not lines:
        raise ValueError("thread CRT monitor log is empty")
    try:
        header = json.loads(lines[0])
    except json.JSONDecodeError as error:
        raise ValueError("thread CRT monitor header is invalid") from error
    if (
        not isinstance(header, Mapping)
        or header.get("schema") != "zuma-rl.live-rng-change-monitor"
        or header.get("version") != 1
        or header.get("process_memory_writes") != 0
    ):
        raise ValueError("thread CRT monitor header contract mismatch")
    process_id = _strict_int(
        header.get("process_id"),
        "thread CRT source process ID",
    )
    thread = header.get("thread_crt_state")
    if not isinstance(thread, Mapping):
        raise ValueError("thread CRT monitor lacks resolved thread state")
    thread_id = _strict_int(
        thread.get("thread_id"),
        "thread CRT source thread ID",
    )
    ptd_thread_id = _strict_int(
        thread.get("ptd_thread_id"),
        "thread CRT source PTD thread ID",
    )
    rand_state_address = _strict_int(
        thread.get("rand_state_address"),
        "thread CRT source state address",
    )
    if (
        process_id <= 0
        or thread_id <= 0
        or ptd_thread_id != thread_id
        or rand_state_address <= 0
    ):
        raise ValueError("thread CRT monitor thread identity is invalid")

    matches: list[Mapping[str, Any]] = []
    for line in lines[1:]:
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError("thread CRT monitor row is invalid") from error
        if (
            isinstance(row, Mapping)
            and row.get("type") == "rng"
            and row.get("framework_update") == framework_update
        ):
            matches.append(row)
    if len(matches) != 1:
        raise ValueError(
            "thread CRT monitor must contain exactly one row at the "
            "source update"
        )
    row = matches[0]
    native_game_time = _strict_int(
        row.get("native_game_time"),
        "thread CRT source native game time",
    )
    captured_state = _strict_int(
        row.get("thread_crt_rand_state"),
        "thread CRT rand state",
    )
    if (
        native_game_time < 0
        or not 0 <= captured_state <= 0xFFFFFFFF
    ):
        raise ValueError("thread CRT monitor state is invalid")
    inverse = pow(0x343FD, -1, 1 << 32)
    state = captured_state
    for _ in range(rewind_draws):
        state = ((state - 0x269EC3) * inverse) & 0xFFFFFFFF
    return ThreadCrtRestoreState(
        source_path=path.resolve(),
        source_sha256=source_sha256,
        source_process_id=process_id,
        source_thread_id=thread_id,
        source_framework_update=framework_update,
        source_native_game_time=native_game_time,
        captured_state=captured_state,
        rewind_draws=rewind_draws,
        state=state,
    )


def load_source_bound_thread_crt_restore_state(
    trace_path: Path,
    recording_report_path: Path,
    *,
    source_order: int,
    framework_update: int,
    caller: int,
) -> ThreadCrtRestoreState:
    """Load one CRT state from a natural, read-only retail source trace."""

    if source_order < 0 or framework_update < 0 or caller <= 0:
        raise ValueError("source-bound thread CRT selector is invalid")
    trace_path = trace_path.resolve()
    recording_report_path = recording_report_path.resolve()
    trace_data, trace = _load_json_object(
        trace_path,
        "source-bound thread CRT trace",
    )
    trace_sha256 = _sha256_bytes(trace_data)
    if (
        trace.get("schema") != "zuma-rl.pc-gameplay-mtrand-call-trace"
        or trace.get("version") != 1
        or trace.get("status") != "PASS"
        or trace.get("failure") is not None
        or trace.get("process_memory_writes") != 0
        or trace.get("persistent_file_modified") is not False
        or trace.get("hardware_breakpoint_restored") is not True
        or trace.get("hardware_breakpoint_restore_error") is not None
        or trace.get("debugger_detach_error") is not None
    ):
        raise ValueError("source-bound thread CRT trace contract mismatch")
    calls = trace.get("calls")
    if not isinstance(calls, list):
        raise ValueError("source-bound thread CRT trace calls are invalid")
    call_count = _strict_int(
        trace.get("call_count"),
        "source-bound thread CRT trace call count",
    )
    if call_count != len(calls) or source_order >= len(calls):
        raise ValueError("source-bound thread CRT trace call count mismatch")
    row = calls[source_order]
    if not isinstance(row, Mapping):
        raise ValueError("source-bound thread CRT row is invalid")
    row_order = _strict_int(row.get("order"), "source-bound CRT row order")
    row_update = _strict_int(
        row.get("framework_update"),
        "source-bound CRT row update",
    )
    row_caller = _strict_int(
        row.get("caller"),
        "source-bound CRT row caller",
    )
    source_process_id = _strict_int(
        trace.get("process_id"),
        "source-bound CRT process ID",
    )
    source_thread_id = _strict_int(
        row.get("thread_id"),
        "source-bound CRT thread ID",
    )
    source_native_game_time = _strict_int(
        row.get("native_game_time"),
        "source-bound CRT native game time",
    )
    captured_state = _strict_int(
        row.get("thread_crt_rand_state"),
        "source-bound CRT rand state",
    )
    thread_identity = row.get("thread_crt_state")
    if (
        row_order != source_order
        or row_update != framework_update
        or row_caller != caller
        or row.get("thread_crt_snapshot_error") is not None
        or not isinstance(thread_identity, Mapping)
        or source_process_id <= 0
        or source_thread_id <= 0
        or source_native_game_time < 0
        or not 0 <= captured_state <= 0xFFFFFFFF
        or _strict_int(
            trace.get("main_thread_id"),
            "source-bound CRT main thread ID",
        )
        != source_thread_id
        or _strict_int(
            thread_identity.get("ptd_thread_id"),
            "source-bound CRT PTD thread ID",
        )
        != source_thread_id
        or _strict_int(
            thread_identity.get("rand_state_address"),
            "source-bound CRT state address",
        )
        <= 0
    ):
        raise ValueError("source-bound thread CRT row contract mismatch")

    report_data, report = _load_json_object(
        recording_report_path,
        "source-bound retail recording report",
    )
    report_sha256 = _sha256_bytes(report_data)
    gameplay = report.get("gameplay")
    main_thread = report.get("main_thread")
    trace_receipt = report.get("gameplay_mtrand_trace")
    try:
        certifying_recording_outcome(report)
    except RetailDmoProvenanceError as error:
        raise ValueError(
            "source-bound retail recording contract mismatch"
        ) from error
    if (
        report.get("schema") != "zuma-rl.retail-autoplay-recording"
        or report.get("version") not in {1, 2}
        or report.get("status") != "PASS"
        or report.get("controlled_crt_rand_seed") is not None
        or report.get("process_memory_writes") != 0
        or report.get("normal_exit") is not True
        or report.get("exit_method") != "retail_ui"
        or report.get("process_exit_code") != 0
        or report.get("host_restored") is not True
        or report.get("host_pre_state_root")
        != report.get("host_restored_state_root")
        or not isinstance(gameplay, Mapping)
        or gameplay.get("status") != "PASS"
        or not isinstance(main_thread, Mapping)
        or main_thread.get("source") != "verified_retail_window"
        or main_thread.get("process_memory_writes") != 0
        or _strict_int(
            main_thread.get("thread_id"),
            "source-bound recording main thread ID",
        )
        != source_thread_id
        or _strict_int(
            report.get("process_id"),
            "source-bound recording process ID",
        )
        != source_process_id
        or not isinstance(trace_receipt, Mapping)
        or trace_receipt.get("status") != "PASS"
        or trace_receipt.get("failure") is not None
        or trace_receipt.get("process_memory_writes") != 0
        or trace_receipt.get("hardware_breakpoint_restored") is not True
        or trace_receipt.get("output_sha256") != trace_sha256
        or _resolved_path(
            trace_receipt.get("output"),
            "source-bound trace receipt",
        )
        != trace_path
        or report.get("runtime_executable_sha256")
        != trace.get("runtime_executable_sha256")
    ):
        raise ValueError("source-bound retail recording contract mismatch")
    dmo_path = _resolved_path(report.get("dmo"), "source-bound recording DMO")
    dmo_sha256 = _sha256_bytes(dmo_path.read_bytes())
    if report.get("dmo_sha256") != dmo_sha256:
        raise ValueError("source-bound retail recording DMO hash mismatch")

    return ThreadCrtRestoreState(
        source_path=trace_path,
        source_sha256=trace_sha256,
        source_process_id=source_process_id,
        source_thread_id=source_thread_id,
        source_framework_update=framework_update,
        source_native_game_time=source_native_game_time,
        captured_state=captured_state,
        rewind_draws=0,
        state=captured_state,
        source_kind="natural_retail_hardware_trace",
        source_recording_report_path=recording_report_path,
        source_recording_report_sha256=report_sha256,
        source_dmo_path=dmo_path,
        source_dmo_sha256=dmo_sha256,
        source_call_order=source_order,
        source_caller=caller,
    )
