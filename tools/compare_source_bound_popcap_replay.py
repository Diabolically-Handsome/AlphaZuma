"""Compare a replay's gameplay RNG suffix with one natural retail source."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tools.monitor_existing_popcap_replay import _load_strict_trace_receipt
from tools.popcap_thread_crt_restore import (
    load_source_bound_thread_crt_restore_state,
)
from zuma_rl.retail_dmo_provenance import certifying_recording_outcome


SCHEMA = "zuma-rl.source-bound-popcap-replay-parity"
VERSION = 2
EXACT_CALL_FIELDS = (
    "framework_update",
    "native_game_time",
    "caller",
    "output",
    "pre_index",
    "pre_state_sha256",
    "post_index",
    "post_state_sha256",
    "thread_crt_rand_state",
    "thread_crt_snapshot_error",
    "score",
    "score_target",
)


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _load_json(path: Path, context: str) -> tuple[bytes, Mapping[str, Any]]:
    path = path.resolve()
    data = path.read_bytes()
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{context} is not valid JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} is not a JSON object")
    return data, value


def _strict_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context} is not an integer")
    return value


def _resolved_path(value: Any, context: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} is not a path")
    return Path(value).resolve()


def compare_exact_call_suffix(
    source_calls: list[Any],
    replay_calls: list[Any],
    *,
    source_start_order: int,
) -> tuple[int, dict[str, Any] | None]:
    """Return exact compared rows and the first semantic mismatch."""

    if source_start_order < 0 or source_start_order >= len(source_calls):
        raise ValueError("source call suffix selector is invalid")
    suffix = source_calls[source_start_order:]
    compared = min(len(suffix), len(replay_calls))
    for index in range(compared):
        source = source_calls[source_start_order + index]
        replay = replay_calls[index]
        if not isinstance(source, Mapping) or not isinstance(replay, Mapping):
            return index, {
                "replay_order": index,
                "source_order": source_start_order + index,
                "fields": ["row_type"],
            }
        mismatched_fields = [
            field
            for field in EXACT_CALL_FIELDS
            if source.get(field) != replay.get(field)
        ]
        if _strict_int(source.get("order"), "source call order") != (
            source_start_order + index
        ):
            mismatched_fields.append("source_order")
        if _strict_int(replay.get("order"), "replay call order") != index:
            mismatched_fields.append("replay_order")
        if mismatched_fields:
            return index, {
                "replay_order": index,
                "source_order": source_start_order + index,
                "fields": mismatched_fields,
                "source": {
                    field: source.get(field) for field in mismatched_fields
                },
                "replay": {
                    field: replay.get(field) for field in mismatched_fields
                },
            }
    if len(replay_calls) < len(suffix):
        return compared, {
            "replay_order": compared,
            "source_order": source_start_order + compared,
            "fields": ["missing_replay_suffix"],
            "source_suffix_call_count": len(suffix),
            "replay_call_count": len(replay_calls),
        }
    return len(suffix), None


def _is_natural_win_terminal_state(
    state: Mapping[str, Any],
    *,
    minimum_score: int,
) -> bool:
    score = state.get("score")
    target = state.get("score_target")
    if (
        isinstance(score, bool)
        or not isinstance(score, int)
        or isinstance(target, bool)
        or not isinstance(target, int)
    ):
        return False
    return (
        score >= target
        and score >= minimum_score
        and state.get("runtime_active") is False
        and state.get("curve_plan_exhausted") is True
        and all(
            state.get(field) == 0
            for field in (
                "active_ball_count",
                "pending_ball_count",
                "inserting_ball_count",
                "fired_ball_count",
            )
        )
    )


def _is_natural_loss_terminal_state(
    state: Mapping[str, Any],
) -> bool:
    loss_counter = state.get("loss_counter")
    return (
        isinstance(loss_counter, int)
        and not isinstance(loss_counter, bool)
        and loss_counter > 0
        and state.get("runtime_active") is False
        and all(
            state.get(field) == 0
            for field in (
                "active_ball_count",
                "pending_ball_count",
                "inserting_ball_count",
                "fired_ball_count",
            )
        )
    )


def _replay_loss_observation(
    full_replay: Mapping[str, Any],
) -> dict[str, Any] | None:
    row = full_replay.get("natural_loss")
    loss_counter = (
        row.get("loss_counter")
        if isinstance(row, Mapping)
        else None
    )
    if (
        not isinstance(row, Mapping)
        or row.get("terminal_board_fallback") is not False
        or isinstance(loss_counter, bool)
        or not isinstance(loss_counter, int)
        or loss_counter <= 0
    ):
        return None
    return {
        "evidence": "live-retail-loss-counter-under-exact-source-bound-gameplay-call-parity",
        **dict(row),
    }


def _first_replay_win_score_call(
    replay_calls: list[Any],
    *,
    source_start_order: int,
    source_suffix_count: int,
    minimum_score: int,
) -> dict[str, Any] | None:
    for replay_order, row in enumerate(
        replay_calls[:source_suffix_count]
    ):
        if not isinstance(row, Mapping):
            continue
        score = row.get("score")
        target = row.get("score_target")
        if (
            isinstance(score, bool)
            or not isinstance(score, int)
            or isinstance(target, bool)
            or not isinstance(target, int)
            or score < target
            or score < minimum_score
        ):
            continue
        return {
            "evidence": "exact-source-bound-gameplay-call",
            "replay_order": replay_order,
            "source_order": source_start_order + replay_order,
            "framework_update": row.get("framework_update"),
            "native_game_time": row.get("native_game_time"),
            "score": score,
            "score_target": target,
        }
    return None


def compare_source_bound_replay(
    *,
    source_trace_path: Path,
    source_recording_report_path: Path,
    replay_trace_path: Path,
    strict_trace_result_path: Path,
    full_replay_result_path: Path,
    source_start_order: int,
    source_framework_update: int,
    source_caller: int,
) -> dict[str, Any]:
    """Validate provenance and exact gameplay-call parity through source EOF."""

    source_trace_path = source_trace_path.resolve()
    source_recording_report_path = source_recording_report_path.resolve()
    replay_trace_path = replay_trace_path.resolve()
    strict_trace_result_path = strict_trace_result_path.resolve()
    full_replay_result_path = full_replay_result_path.resolve()
    source_state = load_source_bound_thread_crt_restore_state(
        source_trace_path,
        source_recording_report_path,
        source_order=source_start_order,
        framework_update=source_framework_update,
        caller=source_caller,
    )
    source_data, source_trace = _load_json(
        source_trace_path,
        "source gameplay call trace",
    )
    report_data, source_report = _load_json(
        source_recording_report_path,
        "source retail recording report",
    )
    replay_data, replay_trace = _load_json(
        replay_trace_path,
        "replay gameplay call trace",
    )
    full_data, full_replay = _load_json(
        full_replay_result_path,
        "full replay monitor result",
    )
    source_calls = source_trace.get("calls")
    replay_calls = replay_trace.get("calls")
    gameplay = source_report.get("gameplay")
    source_final_state = (
        gameplay.get("final_state")
        if isinstance(gameplay, Mapping)
        else None
    )
    full_dmo = full_replay.get("dmo")
    full_strict_receipt = full_replay.get("strict_trace_receipt")
    minimum_win_score = _strict_int(
        full_replay.get("minimum_win_score"),
        "full replay minimum win score",
    )
    source_outcome = certifying_recording_outcome(source_report)
    full_replay_version = full_replay.get("version")
    if full_replay_version == 2:
        replay_expected_outcome = "natural_win"
        natural_outcome_deferred = full_replay.get(
            "natural_win_evidence_deferred"
        )
        natural_outcome_mode = full_replay.get(
            "natural_win_evidence_mode"
        )
    elif full_replay_version == 3:
        replay_expected_outcome = full_replay.get("expected_outcome")
        natural_outcome_deferred = full_replay.get(
            "natural_outcome_evidence_deferred"
        )
        natural_outcome_mode = full_replay.get(
            "natural_outcome_evidence_mode"
        )
    else:
        raise ValueError("full replay monitor version is unsupported")
    if (
        not isinstance(source_calls, list)
        or _strict_int(source_trace.get("call_count"), "source call count")
        != len(source_calls)
        or not isinstance(replay_calls, list)
        or _strict_int(replay_trace.get("call_count"), "replay call count")
        != len(replay_calls)
        or replay_trace.get("schema")
        != "zuma-rl.pc-gameplay-mtrand-call-trace"
        or replay_trace.get("version") != 1
        or replay_trace.get("status") != "PASS"
        or replay_trace.get("failure") is not None
        or replay_trace.get("stop_reason") != "process_exit"
        or replay_trace.get("exited") is not True
        or replay_trace.get("exit_code") != 0
        or replay_trace.get("hardware_breakpoint_restored") is not True
        or replay_trace.get("hardware_breakpoint_restore_error") is not None
        or replay_trace.get("debugger_detach_error") is not None
        or replay_trace.get("persistent_file_modified") is not False
        or not isinstance(gameplay, Mapping)
        or gameplay.get("status") != "PASS"
        or gameplay.get("outcome") != source_outcome
        or not isinstance(source_final_state, Mapping)
        or full_replay.get("schema")
        != "zuma-rl.existing-popcap-full-replay-monitor"
        or full_replay.get("status") != "PASS"
        or full_replay.get("failures") != []
        or full_replay.get("exit_observed") is not True
        or full_replay.get("exit_code") != 0
        or full_replay.get("timed_out") is not False
        or full_replay.get("process_memory_writes") != 0
        or full_replay.get("persistent_file_modified") is not False
        or replay_expected_outcome != source_outcome
        or (
            source_outcome == "natural_win"
            and minimum_win_score <= 0
        )
        or natural_outcome_deferred is not True
        or natural_outcome_mode
        != "source-bound-exact-gameplay-call-parity"
        or full_replay.get("terminal_board_fallback_count") != 0
        or not isinstance(full_dmo, Mapping)
        or not isinstance(full_strict_receipt, Mapping)
    ):
        raise ValueError("source or replay result contract mismatch")
    source_terminal_valid = (
        _is_natural_win_terminal_state(
            source_final_state,
            minimum_score=minimum_win_score,
        )
        if source_outcome == "natural_win"
        else _is_natural_loss_terminal_state(source_final_state)
    )
    if not source_terminal_valid:
        raise ValueError(
            f"source {source_outcome.replace('_', '-')} terminal state "
            "is invalid"
        )
    aligned_dmo_path = _resolved_path(full_dmo.get("path"), "aligned DMO")
    strict_receipt = _load_strict_trace_receipt(
        strict_trace_result_path,
        aligned_dmo_path,
    )
    if (
        full_strict_receipt.get("sha256")
        != strict_receipt["sha256"]
        or _strict_int(
            full_replay.get("command_order_offset"),
            "full replay command-order offset",
        )
        != strict_receipt["command_order_offset"]
        or _strict_int(
            full_replay.get("max_framework_update"),
            "full replay update",
        )
        < _strict_int(full_dmo.get("length_updates"), "DMO length")
        or _strict_int(
            full_replay.get("max_buffer_read_bit_position"),
            "full replay bit position",
        )
        < _strict_int(
            full_dmo.get("expected_final_bit_position"),
            "DMO final bit position",
        )
    ):
        raise ValueError("full replay strict completion receipt mismatch")

    global_restore = replay_trace.get("global_mtrand_call_site_restore")
    crt_restore = replay_trace.get("thread_crt_call_site_restore")
    source_trace_sha256 = _sha256_bytes(source_data)
    source_report_sha256 = _sha256_bytes(report_data)
    if not isinstance(global_restore, Mapping) or not isinstance(
        crt_restore,
        Mapping,
    ):
        raise ValueError("source-bound replay restore receipts are missing")
    global_restored = global_restore.get("restored")
    crt_restored = crt_restore.get("restored")
    if (
        not isinstance(global_restored, Mapping)
        or not isinstance(crt_restored, Mapping)
        or global_restore.get("framework_update") != source_framework_update
        or global_restore.get("caller") != source_caller
        or global_restore.get("bytes_written") != 0
        or global_restore.get("changed") is not False
        or global_restore.get("writeback_verified") is not True
        or crt_restore.get("framework_update") != source_framework_update
        or crt_restore.get("caller") != source_caller
        or crt_restore.get("bytes_written") != 4
        or crt_restore.get("changed") is not True
        or crt_restore.get("writeback_verified") is not True
        or replay_trace.get("process_memory_writes") != 4
        or global_restored.get("source_call_order") != source_start_order
        or global_restored.get("source_trace_sha256")
        != source_trace_sha256
        or global_restored.get("source_recording_report_sha256")
        != source_report_sha256
        or crt_restored.get("source_call_order") != source_start_order
        or crt_restored.get("source_sha256") != source_trace_sha256
        or crt_restored.get("source_recording_report_sha256")
        != source_report_sha256
        or crt_restored.get("state") != source_state.state
    ):
        raise ValueError("source-bound replay restore contract mismatch")

    compared_call_count, mismatch = compare_exact_call_suffix(
        source_calls,
        replay_calls,
        source_start_order=source_start_order,
    )
    source_suffix_count = len(source_calls) - source_start_order
    replay_natural_outcome = (
        _first_replay_win_score_call(
            replay_calls,
            source_start_order=source_start_order,
            source_suffix_count=source_suffix_count,
            minimum_score=minimum_win_score,
        )
        if source_outcome == "natural_win"
        else _replay_loss_observation(full_replay)
    )
    failures: list[str] = []
    if mismatch is not None:
        failures.append("gameplay_call_mismatch")
    if replay_natural_outcome is None:
        failures.append(f"{source_outcome}_evidence_not_observed")
    status = "PASS" if not failures else "FAIL"
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "status": status,
        "failures": failures,
        "classification": (
            "source-bound-exact-retail-gameplay-call-suffix-and-normal-exit"
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "trace_path": str(source_trace_path),
            "trace_sha256": source_trace_sha256,
            "recording_report_path": str(source_recording_report_path),
            "recording_report_sha256": source_report_sha256,
            "raw_dmo_path": str(source_state.source_dmo_path),
            "raw_dmo_sha256": source_state.source_dmo_sha256,
            "outcome": source_outcome,
            "terminal_state": dict(source_final_state),
            "natural_win_score": (
                source_final_state.get("score")
                if source_outcome == "natural_win"
                else None
            ),
            "natural_loss_counter": (
                source_final_state.get("loss_counter")
                if source_outcome == "natural_loss"
                else None
            ),
            "start_order": source_start_order,
            "suffix_call_count": source_suffix_count,
        },
        "replay": {
            "trace_path": str(replay_trace_path),
            "trace_sha256": _sha256_bytes(replay_data),
            "strict_trace_result_path": str(strict_trace_result_path),
            "strict_trace_receipt": strict_receipt,
            "full_replay_result_path": str(full_replay_result_path),
            "full_replay_result_sha256": _sha256_bytes(full_data),
            "call_count": len(replay_calls),
            "post_source_call_count": len(replay_calls) - source_suffix_count,
            "expected_outcome": source_outcome,
            "natural_outcome": replay_natural_outcome,
            "natural_outcome_evidence_mode": natural_outcome_mode,
            "natural_win": (
                replay_natural_outcome
                if source_outcome == "natural_win"
                else None
            ),
            "natural_loss": (
                replay_natural_outcome
                if source_outcome == "natural_loss"
                else None
            ),
            "normal_exit": True,
            "process_memory_write_bytes": 4,
        },
        "exact_fields": list(EXACT_CALL_FIELDS),
        "exact_compared_call_count": compared_call_count,
        "first_mismatch": mismatch,
        "persistent_file_modified": False,
    }


def _write_canonical(path: Path, value: Mapping[str, Any]) -> None:
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-trace", required=True, type=Path)
    parser.add_argument(
        "--source-recording-report",
        required=True,
        type=Path,
    )
    parser.add_argument("--replay-trace", required=True, type=Path)
    parser.add_argument("--strict-trace-result", required=True, type=Path)
    parser.add_argument("--full-replay-result", required=True, type=Path)
    parser.add_argument("--source-start-order", required=True, type=int)
    parser.add_argument("--source-framework-update", required=True, type=int)
    parser.add_argument("--source-caller", required=True, type=lambda x: int(x, 0))
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output.resolve()
    if output.exists() or not output.parent.is_dir():
        raise FileExistsError(f"parity output path is not new: {output}")
    try:
        result = compare_source_bound_replay(
            source_trace_path=args.source_trace,
            source_recording_report_path=args.source_recording_report,
            replay_trace_path=args.replay_trace,
            strict_trace_result_path=args.strict_trace_result,
            full_replay_result_path=args.full_replay_result,
            source_start_order=args.source_start_order,
            source_framework_update=args.source_framework_update,
            source_caller=args.source_caller,
        )
    except (OSError, RuntimeError, ValueError) as error:
        result = {
            "schema": SCHEMA,
            "version": VERSION,
            "status": "FAIL",
            "failures": ["parity_contract_error"],
            "classification": (
                "source-bound-exact-retail-gameplay-call-suffix-and-normal-exit"
            ),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "error": f"{type(error).__name__}: {error}",
            "persistent_file_modified": False,
        }
    _write_canonical(output, result)
    print(
        f"source_bound_replay_parity status={result['status']} "
        f"compared={result.get('exact_compared_call_count', 0)}",
        flush=True,
    )
    print(output, flush=True)
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
