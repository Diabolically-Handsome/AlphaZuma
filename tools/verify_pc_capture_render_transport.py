"""Verify one preregistered free-running render-transport holdout.

This verifier deliberately remains diagnostic-only.  It freezes the visual
mask, draw-settle delay, alignment rule, and thresholds before the holdout is
collected.  A PASS supports the narrow conclusion that residual differences
are bounded sampling-transport effects; it is not PC Golden evidence and does
not authorize Fidelity Gate claims.
"""

from __future__ import annotations

import argparse
from collections import Counter
from difflib import SequenceMatcher
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tools.analyze_pc_capture_draw_alignment import (
    _load_run,
    _sha256_file,
)
from tools.analyze_pc_capture_poll_alignment import (
    CollapsedVisual,
    _assigned_draws,
    _collapse,
    _input_row,
    _load_raw,
    _masked_run,
)
from tools.analyze_pc_capture_visual_alignment import _mask_sha256


PREREGISTRATION_SCHEMA = "zuma-rl.pc-render-transport-preregistration"
PREREGISTRATION_VERSION = 1
EXECUTION_BINDING_SCHEMA = "zuma-rl.pc-render-transport-execution-binding"
EXECUTION_BINDING_VERSION = 1
REPORT_SCHEMA = "zuma-rl.pc-render-transport-holdout-report"
REPORT_VERSION = 1


class RenderTransportError(ValueError):
    """Raised when inputs are malformed or not bound to the frozen plan."""


def _fail(reason: str) -> None:
    raise RenderTransportError(reason)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        _fail("render_transport_json_invalid")
    if not isinstance(value, dict):
        _fail("render_transport_json_invalid")
    return value


def _mapping(value: Any, reason: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(reason)
    return value


def _integer(value: Any, reason: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(reason)
    return value


def _finite_number(value: Any, reason: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(reason)
    result = float(value)
    if not np.isfinite(result) or result < minimum:
        _fail(reason)
    return result


def _frozen_mask(value: Mapping[str, Any]) -> np.ndarray:
    width = _integer(value.get("width"), "render_transport_mask_invalid", minimum=1)
    height = _integer(
        value.get("height"),
        "render_transport_mask_invalid",
        minimum=1,
    )
    pixel_count = _integer(
        value.get("pixel_count"),
        "render_transport_mask_invalid",
    )
    coordinates = value.get("coordinates_xy")
    if not isinstance(coordinates, list) or len(coordinates) != pixel_count:
        _fail("render_transport_mask_invalid")
    parsed: list[tuple[int, int]] = []
    for row in coordinates:
        if (
            not isinstance(row, list)
            or len(row) != 2
            or isinstance(row[0], bool)
            or not isinstance(row[0], int)
            or isinstance(row[1], bool)
            or not isinstance(row[1], int)
        ):
            _fail("render_transport_mask_invalid")
        x, y = row
        if not 0 <= x < width or not 0 <= y < height:
            _fail("render_transport_mask_invalid")
        parsed.append((x, y))
    if len(set(parsed)) != len(parsed):
        _fail("render_transport_mask_invalid")
    mask = np.zeros((height, width), dtype=bool)
    for x, y in parsed:
        mask[y, x] = True
    if (
        int(mask.sum()) != pixel_count
        or _mask_sha256(mask) != value.get("mask_sha256")
    ):
        _fail("render_transport_mask_invalid")
    return mask


def _expand_execution_binding(
    path: Path,
    value: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if value.get("schema") != EXECUTION_BINDING_SCHEMA:
        return value, [
            {
                "artifact": str(path),
                "sha256": _sha256_file(path),
                "role": "frozen_preregistration",
            }
        ]
    if (
        value.get("version") != EXECUTION_BINDING_VERSION
        or value.get("classification")
        != "preregistered-independent-diagnostic-not-pc-golden"
        or value.get("gate_effect") != "none"
        or value.get("status")
        != "FROZEN_RETRY_AFTER_INFRASTRUCTURE_INVALIDATION"
        or value.get("amendments") != ["holdout_session_root"]
    ):
        _fail("render_transport_execution_binding_invalid")
    parent = _mapping(
        value.get("parent"),
        "render_transport_execution_binding_invalid",
    )
    parent_path = Path(str(parent.get("artifact", "")))
    if (
        not parent_path.is_file()
        or _sha256_file(parent_path) != parent.get("sha256")
    ):
        _fail("render_transport_execution_binding_invalid")
    parent_value = _read_json(parent_path)
    if parent_value.get("schema") != PREREGISTRATION_SCHEMA:
        _fail("render_transport_execution_binding_invalid")
    invalidated = _mapping(
        value.get("invalidated_attempt"),
        "render_transport_execution_binding_invalid",
    )
    failed_root = Path(str(invalidated.get("session_root", ""))).resolve()
    parent_holdout = _mapping(
        parent_value.get("holdout"),
        "render_transport_execution_binding_invalid",
    )
    failure_path = Path(str(invalidated.get("failure_artifact", "")))
    if (
        failed_root
        != Path(str(parent_holdout.get("session_root", ""))).resolve()
        or invalidated.get("reason")
        != "collection_failed_before_second_capture"
        or not failure_path.is_file()
        or _sha256_file(failure_path) != invalidated.get("failure_sha256")
        or (failed_root / "collection.json").exists()
        or (failed_root / "run-r2" / "capture" / "metadata.json").exists()
    ):
        _fail("render_transport_execution_binding_invalid")
    failure = _read_json(failure_path)
    if (
        failure.get("status") != "failed"
        or failure.get("host_restored_state_root")
        != invalidated.get("expected_host_restored_state_root")
        or invalidated.get("comparison_performed") is not False
    ):
        _fail("render_transport_execution_binding_invalid")
    new_root = Path(str(value.get("holdout_session_root", ""))).resolve()
    if new_root == failed_root:
        _fail("render_transport_execution_binding_invalid")
    effective = json.loads(json.dumps(parent_value))
    effective["holdout"]["session_root"] = str(new_root)
    chain = [
        {
            "artifact": str(path),
            "sha256": _sha256_file(path),
            "role": "frozen_execution_binding",
        },
        {
            "artifact": str(parent_path),
            "sha256": _sha256_file(parent_path),
            "role": "frozen_parent_preregistration",
        },
        {
            "artifact": str(failure_path),
            "sha256": _sha256_file(failure_path),
            "role": "invalidated_attempt_receipt",
        },
    ]
    return effective, chain


def _load_preregistration(
    path: Path,
    *,
    session_root: Path,
) -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]]]:
    plan, chain = _expand_execution_binding(path, _read_json(path))
    if (
        plan.get("schema") != PREREGISTRATION_SCHEMA
        or plan.get("version") != PREREGISTRATION_VERSION
        or plan.get("classification")
        != "preregistered-independent-diagnostic-not-pc-golden"
        or plan.get("gate_effect") != "none"
        or plan.get("status") != "FROZEN_BEFORE_HOLDOUT_COLLECTION"
    ):
        _fail("render_transport_preregistration_invalid")
    pilot = _mapping(
        plan.get("pilot"),
        "render_transport_preregistration_invalid",
    )
    pilot_path = Path(str(pilot.get("artifact", "")))
    if (
        pilot.get("use") != "threshold_design_only"
        or pilot.get("holdout_reuse_forbidden") is not True
        or not pilot_path.is_file()
        or _sha256_file(pilot_path) != pilot.get("sha256")
    ):
        _fail("render_transport_pilot_binding_invalid")
    holdout = _mapping(
        plan.get("holdout"),
        "render_transport_preregistration_invalid",
    )
    try:
        expected_root = Path(str(holdout.get("session_root", ""))).resolve()
        if expected_root != session_root.resolve() or expected_root == pilot_path.parent:
            _fail("render_transport_holdout_root_invalid")
    except OSError:
        _fail("render_transport_holdout_root_invalid")
    comparison = _mapping(
        plan.get("comparison"),
        "render_transport_preregistration_invalid",
    )
    if (
        comparison.get("pixel_format") != "masked_bgr24"
        or comparison.get("ordered_alignment")
        != "sequence_matcher_exact_digest_blocks"
        or comparison.get("draw_event_sample")
        != "first_present_after_fixed_settle_delay"
    ):
        _fail("render_transport_comparison_contract_invalid")
    _integer(
        comparison.get("render_settle_delay_ns"),
        "render_transport_comparison_contract_invalid",
    )
    mask = _frozen_mask(
        _mapping(
            comparison.get("mask"),
            "render_transport_mask_invalid",
        )
    )
    _mapping(
        plan.get("acceptance"),
        "render_transport_preregistration_invalid",
    )
    return plan, mask, chain


def _validate_collection(
    session_root: Path,
    preregistration: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    plan = _read_json(session_root / "plan.json")
    collection = _read_json(session_root / "collection.json")
    holdout = _mapping(
        preregistration.get("holdout"),
        "render_transport_preregistration_invalid",
    )
    expected_capture = _mapping(
        holdout.get("capture"),
        "render_transport_preregistration_invalid",
    )
    capture = _mapping(plan.get("capture"), "render_transport_plan_invalid")
    runtime = _mapping(plan.get("runtime"), "render_transport_plan_invalid")
    trace = _mapping(plan.get("trace"), "render_transport_plan_invalid")
    prestate = _mapping(plan.get("prestate"), "render_transport_plan_invalid")
    dmo = _mapping(plan.get("dmo"), "render_transport_plan_invalid")
    expected_fields = (
        "duration_seconds",
        "frame_budget_fps",
        "wait_until_framework_update",
        "window_repaint_update",
        "framework_state_poll_arm_update",
        "framework_state_poll_target_hz",
    )
    if (
        plan.get("schema") != "zuma-rl.pc-golden-v4-collection-plan"
        or plan.get("version") != 5
        or dmo.get("sha256") != holdout.get("dmo_sha256")
        or prestate.get("template_state_root")
        != holdout.get("protocol_pre_state_root")
        or runtime.get("runtime_executable_sha256")
        != holdout.get("runtime_executable_sha256")
        or trace.get("startup_process_affinity_mask")
        != expected_capture.get("game_affinity_mask")
        or capture.get("framework_state_poll_affinity_mask")
        != expected_capture.get("observer_affinity_mask")
        or capture.get("framework_state_poll_diagnostic") is not True
        or capture.get("framework_state_diagnostic") is not True
        or any(
            capture.get(field) != expected_capture.get(field)
            for field in expected_fields
        )
    ):
        _fail("render_transport_plan_binding_invalid")
    runs = collection.get("runs")
    if (
        collection.get("schema")
        != "zuma-rl.pc-golden-v4-collection-result"
        or collection.get("version") != 5
        or collection.get("status") != "complete"
        or not isinstance(runs, list)
        or len(runs) != 2
        or collection.get("protocol_pre_state_root")
        != holdout.get("protocol_pre_state_root")
        or collection.get("protocol_restored_state_root")
        != collection.get("protocol_pre_state_root")
        or collection.get("host_restored_state_root")
        != collection.get("host_pre_state_root")
    ):
        _fail("render_transport_collection_invalid")
    identities: set[tuple[int, int]] = set()
    for expected_run, row in zip(("r1", "r2"), runs, strict=True):
        if not isinstance(row, Mapping):
            _fail("render_transport_collection_invalid")
        pid = _integer(
            row.get("process_id"),
            "render_transport_collection_invalid",
            minimum=1,
        )
        creation = _integer(
            row.get("process_creation_filetime_100ns"),
            "render_transport_collection_invalid",
            minimum=1,
        )
        if (
            row.get("run_id") != expected_run
            or row.get("executable_sha256")
            != holdout.get("runtime_executable_sha256")
            or row.get("end_state_root")
            != collection.get("protocol_pre_state_root")
        ):
            _fail("render_transport_collection_invalid")
        identities.add((pid, creation))
    if len(identities) != 2:
        _fail("render_transport_process_identity_invalid")
    return plan, collection


def _counter_rows(values: Counter[int]) -> list[list[int]]:
    return [[key, values[key]] for key in sorted(values)]


def _alignment_metrics(
    left_collapsed: Sequence[CollapsedVisual],
    right_collapsed: Sequence[CollapsedVisual],
    *,
    left_draws: Sequence[int],
    right_draws: Sequence[int],
    left_event_updates: Mapping[int, int],
    right_event_updates: Mapping[int, int],
) -> dict[str, Any]:
    matcher = SequenceMatcher(
        None,
        [row.digest for row in left_collapsed],
        [row.digest for row in right_collapsed],
        autojunk=False,
    )
    update_deltas: Counter[int] = Counter()
    draw_deltas: Counter[int] = Counter()
    matched_rows: list[dict[str, int]] = []
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            left_sequence = left_collapsed[
                block.a + offset
            ].first_sequence
            right_sequence = right_collapsed[
                block.b + offset
            ].first_sequence
            left_draw = left_draws[left_sequence]
            right_draw = right_draws[right_sequence]
            try:
                left_update = left_event_updates[left_draw]
                right_update = right_event_updates[right_draw]
            except KeyError:
                _fail("render_transport_draw_event_binding_invalid")
            update_delta = left_update - right_update
            draw_delta = left_draw - right_draw
            update_deltas[update_delta] += 1
            draw_deltas[draw_delta] += 1
            matched_rows.append(
                {
                    "left_sequence": left_sequence,
                    "right_sequence": right_sequence,
                    "left_draw_count": left_draw,
                    "right_draw_count": right_draw,
                    "left_poll_event_update": left_update,
                    "right_poll_event_update": right_update,
                }
            )
    non_equal: list[dict[str, Any]] = []
    maximum_left_gap = 0
    maximum_right_gap = 0
    for tag, left_start, left_stop, right_start, right_stop in matcher.get_opcodes():
        if tag == "equal":
            continue
        left_size = left_stop - left_start
        right_size = right_stop - right_start
        maximum_left_gap = max(maximum_left_gap, left_size)
        maximum_right_gap = max(maximum_right_gap, right_size)
        non_equal.append(
            {
                "tag": tag,
                "left_collapsed_range": [left_start, left_stop],
                "right_collapsed_range": [right_start, right_stop],
                "left_size": left_size,
                "right_size": right_size,
            }
        )
    matched_count = len(matched_rows)
    same_update_count = update_deltas[0]
    draw_span = (
        max(draw_deltas) - min(draw_deltas) if draw_deltas else 0
    )
    maximum_update_delta = max(map(abs, update_deltas), default=0)
    return {
        "left_collapsed_visual_count": len(left_collapsed),
        "right_collapsed_visual_count": len(right_collapsed),
        "matched_collapsed_visual_count": matched_count,
        "sequence_matcher_ratio": matcher.ratio(),
        "maximum_unmatched_collapsed_gap_left": maximum_left_gap,
        "maximum_unmatched_collapsed_gap_right": maximum_right_gap,
        "same_poll_event_update_count": same_update_count,
        "same_poll_event_update_fraction": (
            same_update_count / matched_count if matched_count else 0.0
        ),
        "maximum_absolute_poll_event_update_delta": maximum_update_delta,
        "local_draw_delta_span": draw_span,
        "poll_event_update_delta_histogram": _counter_rows(update_deltas),
        "draw_delta_histogram": _counter_rows(draw_deltas),
        "non_equal_opcodes": non_equal,
        "matched_rows": matched_rows,
    }


def _criterion(
    name: str,
    observed: Any,
    required: Any,
    passed: bool,
) -> dict[str, Any]:
    return {
        "name": name,
        "status": "PASS" if passed else "FAIL",
        "observed": observed,
        "required": required,
    }


def _criteria(
    metrics: Mapping[str, Any],
    runs: Sequence[Any],
    acceptance: Mapping[str, Any],
) -> list[dict[str, Any]]:
    minimum_hz = _finite_number(
        acceptance.get("minimum_mean_poll_hz_each_run"),
        "render_transport_acceptance_invalid",
    )
    maximum_p99 = _integer(
        acceptance.get("maximum_p99_poll_interval_ns_each_run"),
        "render_transport_acceptance_invalid",
    )
    maximum_skips = _integer(
        acceptance.get("maximum_skipped_draw_increments_each_run"),
        "render_transport_acceptance_invalid",
    )
    minimum_matches = _integer(
        acceptance.get("minimum_matched_collapsed_visual_count"),
        "render_transport_acceptance_invalid",
        minimum=1,
    )
    minimum_ratio = _finite_number(
        acceptance.get("minimum_sequence_matcher_ratio"),
        "render_transport_acceptance_invalid",
    )
    maximum_gap = _integer(
        acceptance.get("maximum_unmatched_collapsed_gap_each_run"),
        "render_transport_acceptance_invalid",
    )
    minimum_same_update = _finite_number(
        acceptance.get("minimum_same_poll_event_update_fraction"),
        "render_transport_acceptance_invalid",
    )
    maximum_update_delta = _integer(
        acceptance.get("maximum_absolute_poll_event_update_delta"),
        "render_transport_acceptance_invalid",
    )
    maximum_draw_span = _integer(
        acceptance.get("maximum_local_draw_delta_span"),
        "render_transport_acceptance_invalid",
    )
    poll_hz = [run.poll_summary.get("mean_polls_per_second") for run in runs]
    poll_p99 = [
        run.poll_summary.get("p99_sample_completion_interval_ns")
        for run in runs
    ]
    poll_skips = [
        run.poll_summary.get("skipped_draw_increment_count", 0)
        for run in runs
    ]
    topology = [
        run.poll_summary.get("affinity_topology") for run in runs
    ]
    rows = [
        _criterion(
            "mean_poll_hz_each_run",
            poll_hz,
            {"minimum_each": minimum_hz},
            all(isinstance(value, (int, float)) and value >= minimum_hz for value in poll_hz),
        ),
        _criterion(
            "p99_poll_interval_ns_each_run",
            poll_p99,
            {"maximum_each": maximum_p99},
            all(isinstance(value, int) and value <= maximum_p99 for value in poll_p99),
        ),
        _criterion(
            "skipped_draw_increments_each_run",
            poll_skips,
            {"maximum_each": maximum_skips},
            all(isinstance(value, int) and value <= maximum_skips for value in poll_skips),
        ),
        _criterion(
            "disjoint_physical_cores_each_run",
            [
                row.get("physical_cores_disjoint")
                if isinstance(row, Mapping)
                else None
                for row in topology
            ],
            True,
            acceptance.get("require_disjoint_physical_cores") is True
            and all(
                isinstance(row, Mapping)
                and row.get("physical_cores_disjoint") is True
                for row in topology
            ),
        ),
        _criterion(
            "matched_collapsed_visual_count",
            metrics["matched_collapsed_visual_count"],
            {"minimum": minimum_matches},
            metrics["matched_collapsed_visual_count"] >= minimum_matches,
        ),
        _criterion(
            "sequence_matcher_ratio",
            metrics["sequence_matcher_ratio"],
            {"minimum": minimum_ratio},
            metrics["sequence_matcher_ratio"] >= minimum_ratio,
        ),
        _criterion(
            "maximum_unmatched_collapsed_gap_each_run",
            [
                metrics["maximum_unmatched_collapsed_gap_left"],
                metrics["maximum_unmatched_collapsed_gap_right"],
            ],
            {"maximum_each": maximum_gap},
            metrics["maximum_unmatched_collapsed_gap_left"] <= maximum_gap
            and metrics["maximum_unmatched_collapsed_gap_right"] <= maximum_gap,
        ),
        _criterion(
            "same_poll_event_update_fraction",
            metrics["same_poll_event_update_fraction"],
            {"minimum": minimum_same_update},
            metrics["same_poll_event_update_fraction"] >= minimum_same_update,
        ),
        _criterion(
            "maximum_absolute_poll_event_update_delta",
            metrics["maximum_absolute_poll_event_update_delta"],
            {"maximum": maximum_update_delta},
            metrics["maximum_absolute_poll_event_update_delta"]
            <= maximum_update_delta,
        ),
        _criterion(
            "local_draw_delta_span",
            metrics["local_draw_delta_span"],
            {"maximum": maximum_draw_span},
            metrics["local_draw_delta_span"] <= maximum_draw_span,
        ),
    ]
    return rows


def verify(preregistration_path: Path, session_root: Path) -> dict[str, Any]:
    preregistration_path = preregistration_path.resolve()
    session_root = session_root.resolve()
    preregistration, mask, preregistration_chain = _load_preregistration(
        preregistration_path,
        session_root=session_root,
    )
    _, collection = _validate_collection(session_root, preregistration)
    comparison = _mapping(
        preregistration.get("comparison"),
        "render_transport_preregistration_invalid",
    )
    delay_ns = _integer(
        comparison.get("render_settle_delay_ns"),
        "render_transport_comparison_contract_invalid",
    )
    timelines = [
        _load_run(session_root / "run-r1"),
        _load_run(session_root / "run-r2"),
    ]
    loaded = [_load_raw(timeline) for timeline in timelines]
    if any(
        timeline.width != mask.shape[1]
        or timeline.height != mask.shape[0]
        for timeline in timelines
    ):
        _fail("render_transport_dimensions_invalid")
    runs = [
        _masked_run(timeline, raw, metadata, raw_sha, mask)
        for timeline, (raw, metadata, raw_sha) in zip(
            timelines,
            loaded,
            strict=True,
        )
    ]
    assigned = [_assigned_draws(run, delay_ns) for run in runs]
    collapsed = [_collapse(run.masked_digests) for run in runs]
    event_updates = [
        {event.draw_count: event.update_count for event in run.draw_events}
        for run in runs
    ]
    metrics = _alignment_metrics(
        collapsed[0],
        collapsed[1],
        left_draws=assigned[0],
        right_draws=assigned[1],
        left_event_updates=event_updates[0],
        right_event_updates=event_updates[1],
    )
    criteria = _criteria(
        metrics,
        runs,
        _mapping(
            preregistration.get("acceptance"),
            "render_transport_acceptance_invalid",
        ),
    )
    reasons = [row["name"] for row in criteria if row["status"] != "PASS"]
    status = "PASS" if not reasons else "FAIL"
    return {
        "schema": REPORT_SCHEMA,
        "version": REPORT_VERSION,
        "classification": "independent-diagnostic-not-pc-golden",
        "gate_effect": "none",
        "status": status,
        "reasons": reasons,
        "preregistration": {
            "artifact": str(preregistration_path),
            "sha256": _sha256_file(preregistration_path),
            "status": _read_json(preregistration_path).get("status"),
            "chain": preregistration_chain,
        },
        "session_root": str(session_root),
        "collection": {
            "artifact": str(session_root / "collection.json"),
            "sha256": _sha256_file(session_root / "collection.json"),
            "status": collection.get("status"),
        },
        "inputs": {
            "left": _input_row(runs[0]),
            "right": _input_row(runs[1]),
        },
        "comparison": {
            "render_settle_delay_ns": delay_ns,
            "mask_pixel_count": int(mask.sum()),
            "mask_sha256": _mask_sha256(mask),
            "metrics": metrics,
        },
        "criteria": criteria,
        "interpretation": preregistration.get("interpretation"),
    }


def _write_exclusive(path: Path, report: Mapping[str, Any]) -> str:
    if not path.is_absolute() or path.exists() or not path.parent.is_dir():
        _fail("render_transport_output_invalid")
    payload = (
        json.dumps(
            report,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")
    try:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError:
        _fail("render_transport_output_write_failed")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--session-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = verify(args.preregistration, args.session_root)
        digest = _write_exclusive(args.output.resolve(), report)
    except RenderTransportError as error:
        print(f"render transport error: {error}", file=sys.stderr)
        return 2
    print(f"{args.output.resolve()}\n{digest}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
