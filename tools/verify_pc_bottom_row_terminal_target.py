"""Verify a target under the PC Golden bounded final-raster-row contract.

Every requested update must have a monotonic pair of independently captured
frames that is byte-exact above the final native raster row.  Only a bounded
number of pixels in row 599 may differ, alpha remains exact, and the terminal
frame must independently bind to a reference frozen before collection.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tools.verify_pc_render_settled_target import (
    TargetVerificationError,
    _load_mask_chain,
    _load_run,
    _read_json,
    _sha256_path,
    _write_exclusive,
)
from tools.verify_pc_terminal_state_target import _reference_frame


REPORT_SCHEMA = "zuma-rl.pc-bottom-row-terminal-target-verification"
REPORT_VERSION = 1


def _fail(reason: str) -> None:
    raise TargetVerificationError(reason)


def _bottom_row_metrics(
    left: np.ndarray,
    right: np.ndarray,
) -> dict[str, int]:
    if left.shape != (600, 800, 4) or right.shape != left.shape:
        _fail("bottom_row_target_geometry_invalid")
    difference = left != right
    pixel_difference = np.any(difference, axis=2)
    return {
        "interior_mismatch_count": int(pixel_difference[:599].sum()),
        "edge_mismatch_count": int(pixel_difference[599].sum()),
        "edge_alpha_mismatch_count": int(difference[599, :, 3].sum()),
        "maximum_edge_bgr_channel_delta": int(
            np.abs(
                left[599, :, :3].astype(np.int16)
                - right[599, :, :3].astype(np.int16)
            ).max(initial=0)
        ),
    }


def _select_pairs(
    left: Any,
    right: Any,
    *,
    target_updates: Sequence[int],
    maximum_edge_mismatches: int,
) -> tuple[list[dict[str, Any]], list[int], list[int]]:
    selected: list[dict[str, Any]] = []
    absent: list[int] = []
    unmatched: list[int] = []
    for update in target_updates:
        left_sequences = left.records_by_update.get(update, ())
        right_sequences = right.records_by_update.get(update, ())
        if not left_sequences or not right_sequences:
            absent.append(update)
            continue
        accepted: list[tuple[int, int, int, int, dict[str, Any]]] = []
        for left_sequence in left_sequences:
            for right_sequence in right_sequences:
                left_frame = left.raw[left_sequence]
                right_frame = right.raw[right_sequence]
                metrics = _bottom_row_metrics(left_frame, right_frame)
                if (
                    metrics["interior_mismatch_count"]
                    or metrics["edge_mismatch_count"]
                    > maximum_edge_mismatches
                    or metrics["edge_alpha_mismatch_count"]
                ):
                    continue
                row = {
                    "framework_update": update,
                    "r1_sequence": left_sequence,
                    "r2_sequence": right_sequence,
                    "r1_frame_sha256": left.frame_hashes[left_sequence],
                    "r2_frame_sha256": right.frame_hashes[right_sequence],
                    "full_frame_exact": (
                        left.frame_hashes[left_sequence]
                        == right.frame_hashes[right_sequence]
                    ),
                    **metrics,
                }
                accepted.append(
                    (
                        metrics["edge_mismatch_count"],
                        metrics["maximum_edge_bgr_channel_delta"],
                        left_sequence,
                        right_sequence,
                        row,
                    )
                )
        if not accepted:
            unmatched.append(update)
            continue
        accepted.sort(key=lambda item: item[:4])
        selected.append(accepted[0][4])
    return selected, absent, unmatched


def _criteria(
    *,
    selected: Sequence[Mapping[str, Any]],
    required_count: int,
    absent: Sequence[int],
    unmatched: Sequence[int],
    maximum_edge_mismatches: int,
    terminal_comparisons: Mapping[str, Mapping[str, int]],
) -> list[dict[str, Any]]:
    complete = (
        len(selected) == required_count and not absent and not unmatched
    )
    monotonic = complete and all(
        after["r1_sequence"] > before["r1_sequence"]
        and after["r2_sequence"] > before["r2_sequence"]
        for before, after in zip(selected, selected[1:])
    )
    terminal_exact = len(terminal_comparisons) == 2 and all(
        row["interior_mismatch_count"] == 0
        and row["edge_mismatch_count"] <= maximum_edge_mismatches
        and row["edge_alpha_mismatch_count"] == 0
        for row in terminal_comparisons.values()
    )
    return [
        {
            "name": "complete_target_update_coverage",
            "observed": len(selected),
            "required": required_count,
            "absent_updates": list(absent),
            "unmatched_updates": list(unmatched),
            "status": "PASS" if complete else "FAIL",
        },
        {
            "name": "monotonic_selected_frame_pairs",
            "observed": monotonic if complete else None,
            "required": True,
            "status": "PASS" if monotonic else "FAIL",
        },
        {
            "name": "exact_above_final_raster_row",
            "observed_maximum": (
                max(
                    int(row["interior_mismatch_count"])
                    for row in selected
                )
                if selected
                else None
            ),
            "required_maximum": 0,
            "status": (
                "PASS"
                if complete
                and all(row["interior_mismatch_count"] == 0 for row in selected)
                else "FAIL"
            ),
        },
        {
            "name": "bounded_final_raster_row_pixels",
            "observed_maximum": (
                max(int(row["edge_mismatch_count"]) for row in selected)
                if selected
                else None
            ),
            "required_maximum": maximum_edge_mismatches,
            "status": (
                "PASS"
                if complete
                and all(
                    row["edge_mismatch_count"] <= maximum_edge_mismatches
                    for row in selected
                )
                else "FAIL"
            ),
        },
        {
            "name": "exact_final_raster_row_alpha",
            "observed_maximum": (
                max(
                    int(row["edge_alpha_mismatch_count"])
                    for row in selected
                )
                if selected
                else None
            ),
            "required_maximum": 0,
            "status": (
                "PASS"
                if complete
                and all(
                    row["edge_alpha_mismatch_count"] == 0
                    for row in selected
                )
                else "FAIL"
            ),
        },
        {
            "name": "terminal_reference_bounded_final_row_exactness",
            "observed": terminal_exact,
            "required": True,
            "status": "PASS" if terminal_exact else "FAIL",
        },
    ]


def verify(
    *,
    session_root: Path,
    calibration_preregistration_path: Path,
    calibration_execution_binding_path: Path,
    calibration_holdout_report_path: Path,
    target_start: int,
    target_end: int,
    terminal_update: int,
    terminal_reference_path: Path,
    terminal_reference_sha256: str,
    maximum_edge_mismatches: int,
) -> dict[str, Any]:
    if target_end < target_start or not target_start <= terminal_update <= target_end:
        _fail("bottom_row_target_update_range_invalid")
    if not 0 <= maximum_edge_mismatches <= 800:
        _fail("bottom_row_target_edge_budget_invalid")
    session_root = session_root.resolve()
    paths = (
        calibration_preregistration_path.resolve(),
        calibration_execution_binding_path.resolve(),
        calibration_holdout_report_path.resolve(),
    )
    collection_path = session_root / "collection.json"
    collection = _read_json(collection_path)
    if (
        collection.get("schema") != "zuma-rl.pc-golden-v4-collection-result"
        or collection.get("status") != "complete"
        or collection.get("protocol_pre_state_root")
        != collection.get("protocol_restored_state_root")
        or collection.get("host_pre_state_root")
        != collection.get("host_restored_state_root")
    ):
        _fail("bottom_row_target_collection_invalid")
    mask, _preregistration, chain = _load_mask_chain(*paths)
    left = _load_run(
        session_root,
        "r1",
        mask=mask,
        calibration_paths=paths,
    )
    right = _load_run(
        session_root,
        "r2",
        mask=mask,
        calibration_paths=paths,
    )
    if left.metadata.process_instance == right.metadata.process_instance:
        _fail("bottom_row_target_process_instances_not_distinct")
    target_updates = tuple(range(target_start, target_end + 1))
    selected, absent, unmatched = _select_pairs(
        left,
        right,
        target_updates=target_updates,
        maximum_edge_mismatches=maximum_edge_mismatches,
    )
    terminal_rows = [
        row for row in selected if row["framework_update"] == terminal_update
    ]
    terminal_comparisons: dict[str, Mapping[str, int]] = {}
    reference = _reference_frame(
        terminal_reference_path.resolve(),
        expected_sha256=terminal_reference_sha256,
    )
    if len(terminal_rows) == 1:
        terminal = terminal_rows[0]
        for run_id, run in (("r1", left), ("r2", right)):
            sequence = int(terminal[f"{run_id}_sequence"])
            terminal_comparisons[run_id] = _bottom_row_metrics(
                run.raw[sequence], reference
            )
    criteria = _criteria(
        selected=selected,
        required_count=len(target_updates),
        absent=absent,
        unmatched=unmatched,
        maximum_edge_mismatches=maximum_edge_mismatches,
        terminal_comparisons=terminal_comparisons,
    )
    reasons = [row["name"] for row in criteria if row["status"] != "PASS"]
    return {
        "schema": REPORT_SCHEMA,
        "version": REPORT_VERSION,
        "classification": "pre-packaging-bounded-bottom-row-terminal-target",
        "gate_effect": "none_until_independent_pc_golden_verification",
        "status": "PASS" if not reasons else "FAIL",
        "reasons": reasons,
        "session_root": str(session_root),
        "collection": {
            "artifact": str(collection_path),
            "sha256": _sha256_path(collection_path),
            "host_state_root": collection.get("host_pre_state_root"),
            "protocol_state_root": collection.get("protocol_pre_state_root"),
        },
        "calibration_chain": list(chain),
        "target": {
            "first_update": target_start,
            "last_update": target_end,
            "required_update_count": len(target_updates),
            "selected_update_count": len(selected),
            "absent_updates": absent,
            "unmatched_updates": unmatched,
            "excluded_bottom_rows": 1,
            "maximum_edge_mismatches": maximum_edge_mismatches,
            "selected_pairs": selected,
        },
        "terminal_state_binding": {
            "framework_update": terminal_update,
            "reference_artifact": str(terminal_reference_path.resolve()),
            "reference_sha256": terminal_reference_sha256,
            "runs": terminal_comparisons,
        },
        "runs": {"r1": dict(left.evidence), "r2": dict(right.evidence)},
        "criteria": criteria,
        "non_authorizations": [
            "pc_golden",
            "front_insertion",
            "fidelity_gate_open",
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-root", required=True, type=Path)
    parser.add_argument("--calibration-preregistration", required=True, type=Path)
    parser.add_argument("--calibration-execution-binding", required=True, type=Path)
    parser.add_argument("--calibration-holdout-report", required=True, type=Path)
    parser.add_argument("--target-start", required=True, type=int)
    parser.add_argument("--target-end", required=True, type=int)
    parser.add_argument("--terminal-update", required=True, type=int)
    parser.add_argument("--terminal-reference", required=True, type=Path)
    parser.add_argument("--terminal-reference-sha256", required=True)
    parser.add_argument("--maximum-edge-mismatches", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = verify(
            session_root=args.session_root,
            calibration_preregistration_path=args.calibration_preregistration,
            calibration_execution_binding_path=args.calibration_execution_binding,
            calibration_holdout_report_path=args.calibration_holdout_report,
            target_start=args.target_start,
            target_end=args.target_end,
            terminal_update=args.terminal_update,
            terminal_reference_path=args.terminal_reference,
            terminal_reference_sha256=args.terminal_reference_sha256,
            maximum_edge_mismatches=args.maximum_edge_mismatches,
        )
        digest = _write_exclusive(args.output.resolve(), report)
    except TargetVerificationError as error:
        print(f"bottom-row target verification error: {error}", file=sys.stderr)
        return 2
    print(f"{args.output.resolve()}\n{digest}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
