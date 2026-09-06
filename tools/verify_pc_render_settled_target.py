"""Verify a frozen render-settled target window under one sparse mask.

This is a fail-closed pre-packaging verifier.  It accepts only a mask frozen in
the render-transport preregistration and independently passed by its holdout.
Every requested framework update must have a monotonic frame pair that is
exact outside that mask.  The masked pixels remain bounded by both a mismatch
count and a per-channel delta.  A PASS is still not PC Golden evidence; the PC
Golden packager and verifier must independently enforce the same contract.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
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

from tools.analyze_pc_capture_visual_alignment import _mask_sha256
from zuma_rl.pc_protocol_evidence import PcDxgiCaptureMetadata
from zuma_rl.pc_render_settle import (
    PcRenderSettledUpdateMap,
    verify_render_settled_update_map,
)


REPORT_SCHEMA = "zuma-rl.pc-render-settled-sparse-target-verification"
REPORT_VERSION = 2
PREREGISTRATION_SCHEMA = "zuma-rl.pc-render-transport-preregistration"
EXECUTION_BINDING_SCHEMA = "zuma-rl.pc-render-transport-execution-binding"
HOLDOUT_SCHEMA = "zuma-rl.pc-render-transport-holdout-report"
_MASKED_FRAME_DOMAIN = b"zuma-rl.sparse-mask-bgra-frame.v1\0"


class TargetVerificationError(ValueError):
    """Raised when immutable inputs or the frozen calibration are malformed."""


def _fail(reason: str) -> None:
    raise TargetVerificationError(reason)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        _fail("target_verification_json_invalid")
    if not isinstance(value, dict):
        _fail("target_verification_json_invalid")
    return value


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
    except OSError:
        _fail("target_verification_artifact_unreadable")
    return f"sha256:{digest.hexdigest()}"


def _strict_int(value: Any, reason: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(reason)
    return value


def _mapping(value: Any, reason: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(reason)
    return value


def _load_mask_chain(
    preregistration_path: Path,
    execution_binding_path: Path,
    holdout_report_path: Path,
) -> tuple[np.ndarray, Mapping[str, Any], tuple[Mapping[str, Any], ...]]:
    preregistration = _read_json(preregistration_path)
    execution = _read_json(execution_binding_path)
    holdout = _read_json(holdout_report_path)
    preregistration_sha = _sha256_path(preregistration_path)
    execution_sha = _sha256_path(execution_binding_path)
    holdout_sha = _sha256_path(holdout_report_path)

    if (
        preregistration.get("schema") != PREREGISTRATION_SCHEMA
        or preregistration.get("version") != 1
        or preregistration.get("status")
        != "FROZEN_BEFORE_HOLDOUT_COLLECTION"
        or preregistration.get("classification")
        != "preregistered-independent-diagnostic-not-pc-golden"
        or preregistration.get("gate_effect") != "none"
    ):
        _fail("target_verification_mask_preregistration_invalid")
    if (
        execution.get("schema") != EXECUTION_BINDING_SCHEMA
        or execution.get("version") != 1
        or execution.get("status")
        != "FROZEN_RETRY_AFTER_INFRASTRUCTURE_INVALIDATION"
        or execution.get("classification")
        != "preregistered-independent-diagnostic-not-pc-golden"
        or execution.get("gate_effect") != "none"
    ):
        _fail("target_verification_mask_execution_binding_invalid")
    parent = _mapping(
        execution.get("parent"),
        "target_verification_mask_execution_binding_invalid",
    )
    if parent.get("sha256") != preregistration_sha:
        _fail("target_verification_mask_execution_parent_mismatch")
    if (
        holdout.get("schema") != HOLDOUT_SCHEMA
        or holdout.get("version") != 1
        or holdout.get("status") != "PASS"
        or holdout.get("classification")
        != "independent-diagnostic-not-pc-golden"
        or holdout.get("gate_effect") != "none"
    ):
        _fail("target_verification_mask_holdout_invalid")
    criteria = holdout.get("criteria")
    if (
        not isinstance(criteria, list)
        or not criteria
        or any(
            not isinstance(row, Mapping) or row.get("status") != "PASS"
            for row in criteria
        )
    ):
        _fail("target_verification_mask_holdout_criteria_invalid")
    holdout_preregistration = _mapping(
        holdout.get("preregistration"),
        "target_verification_mask_holdout_invalid",
    )
    chain = holdout_preregistration.get("chain")
    if not isinstance(chain, list):
        _fail("target_verification_mask_holdout_chain_invalid")
    chain_hashes = {
        row.get("sha256")
        for row in chain
        if isinstance(row, Mapping)
    }
    if preregistration_sha not in chain_hashes or execution_sha not in chain_hashes:
        _fail("target_verification_mask_holdout_chain_invalid")

    comparison = _mapping(
        preregistration.get("comparison"),
        "target_verification_mask_preregistration_invalid",
    )
    if (
        comparison.get("pixel_format") != "masked_bgr24"
        or comparison.get("ordered_alignment")
        != "sequence_matcher_exact_digest_blocks"
        or comparison.get("draw_event_sample")
        != "first_present_after_fixed_settle_delay"
    ):
        _fail("target_verification_mask_comparison_invalid")
    mask_value = _mapping(
        comparison.get("mask"),
        "target_verification_mask_invalid",
    )
    width = _strict_int(
        mask_value.get("width"),
        "target_verification_mask_invalid",
        minimum=1,
    )
    height = _strict_int(
        mask_value.get("height"),
        "target_verification_mask_invalid",
        minimum=1,
    )
    pixel_count = _strict_int(
        mask_value.get("pixel_count"),
        "target_verification_mask_invalid",
        minimum=1,
    )
    coordinates = mask_value.get("coordinates_xy")
    if not isinstance(coordinates, list) or len(coordinates) != pixel_count:
        _fail("target_verification_mask_invalid")
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
            _fail("target_verification_mask_invalid")
        x, y = row
        if not 0 <= x < width or not 0 <= y < height:
            _fail("target_verification_mask_invalid")
        parsed.append((x, y))
    if len(set(parsed)) != len(parsed):
        _fail("target_verification_mask_invalid")
    mask = np.zeros((height, width), dtype=bool)
    for x, y in parsed:
        mask[y, x] = True
    mask_sha = _mask_sha256(mask)
    if mask_sha != mask_value.get("mask_sha256"):
        _fail("target_verification_mask_invalid")

    holdout_comparison = _mapping(
        holdout.get("comparison"),
        "target_verification_mask_holdout_invalid",
    )
    if (
        holdout_comparison.get("mask_pixel_count") != pixel_count
        or holdout_comparison.get("mask_sha256") != mask_sha
    ):
        _fail("target_verification_mask_holdout_mask_mismatch")
    return mask, preregistration, (
        {
            "role": "frozen_mask_preregistration",
            "path": str(preregistration_path),
            "sha256": preregistration_sha,
        },
        {
            "role": "frozen_mask_execution_binding",
            "path": str(execution_binding_path),
            "sha256": execution_sha,
        },
        {
            "role": "independent_mask_holdout",
            "path": str(holdout_report_path),
            "sha256": holdout_sha,
        },
    )


@dataclass(frozen=True, slots=True)
class _RunFrames:
    run_id: str
    raw: np.memmap
    metadata: PcDxgiCaptureMetadata
    records_by_update: Mapping[int, tuple[int, ...]]
    frame_hashes: tuple[str, ...]
    evidence: Mapping[str, Any]


def _load_frame_hashes(path: Path, frame_count: int) -> tuple[str, ...]:
    try:
        with path.open("r", encoding="ascii", newline="") as stream:
            rows = list(csv.DictReader(stream))
    except (OSError, UnicodeError, csv.Error):
        _fail("target_verification_frames_csv_invalid")
    if len(rows) != frame_count:
        _fail("target_verification_frames_csv_invalid")
    hashes: list[str] = []
    for sequence, row in enumerate(rows):
        try:
            observed_sequence = int(row["sequence"])
            frame_sha = row["frame_sha256"]
        except (KeyError, TypeError, ValueError):
            _fail("target_verification_frames_csv_invalid")
        if observed_sequence != sequence or not isinstance(frame_sha, str):
            _fail("target_verification_frames_csv_invalid")
        hashes.append(frame_sha)
    return tuple(hashes)


def _load_run(
    session_root: Path,
    run_id: str,
    *,
    mask: np.ndarray,
    calibration_paths: tuple[Path, Path, Path],
) -> _RunFrames:
    run_root = session_root / f"run-{run_id}"
    capture_root = run_root / "capture"
    metadata_path = capture_root / "metadata.json"
    frames_path = capture_root / "frames.bgra.raw"
    frames_csv_path = capture_root / "frames.csv"
    framework_update_path = run_root / "framework-updates.json"
    framework_state_path = run_root / "framework-state-diagnostic.json"
    framework_poll_path = run_root / "framework-state-poll-diagnostic.json"
    settled_path = run_root / "render-settled-updates.json"
    metadata = PcDxgiCaptureMetadata.read(metadata_path)
    if (metadata.height, metadata.width) != mask.shape:
        _fail("target_verification_geometry_mismatch")
    expected_bytes = metadata.frame_count * metadata.width * metadata.height * 4
    try:
        if frames_path.stat().st_size != expected_bytes:
            _fail("target_verification_raw_size_mismatch")
    except OSError:
        _fail("target_verification_raw_unreadable")
    raw_sha = _sha256_path(frames_path)
    if raw_sha != metadata.raw_sha256:
        _fail("target_verification_raw_sha256_mismatch")
    frames_csv_sha = _sha256_path(frames_csv_path)
    if frames_csv_sha != metadata.frames_csv_sha256:
        _fail("target_verification_frames_csv_sha256_mismatch")
    declared = PcRenderSettledUpdateMap.read(settled_path)
    verified = verify_render_settled_update_map(
        declared,
        capture_metadata_path=metadata_path,
        frames_csv_path=frames_csv_path,
        framework_update_map_path=framework_update_path,
        framework_state_sidecar_path=framework_state_path,
        framework_poll_path=framework_poll_path,
        calibration_preregistration_path=calibration_paths[0],
        calibration_execution_binding_path=calibration_paths[1],
        calibration_holdout_report_path=calibration_paths[2],
    )
    grouped: dict[int, list[int]] = {}
    for row in verified.records:
        grouped.setdefault(row.assigned_framework_update, []).append(row.sequence)
    raw = np.memmap(
        frames_path,
        dtype=np.uint8,
        mode="r",
        shape=(metadata.frame_count, metadata.height, metadata.width, 4),
    )
    return _RunFrames(
        run_id=run_id,
        raw=raw,
        metadata=metadata,
        records_by_update={
            update: tuple(sequences) for update, sequences in grouped.items()
        },
        frame_hashes=_load_frame_hashes(frames_csv_path, metadata.frame_count),
        evidence={
            "process_id": metadata.process_id,
            "process_creation_filetime_100ns": (
                metadata.process_creation_filetime_100ns
            ),
            "frame_count": metadata.frame_count,
            "capture_metadata_sha256": _sha256_path(metadata_path),
            "raw_sha256": raw_sha,
            "frames_csv_sha256": frames_csv_sha,
            "framework_update_map_sha256": _sha256_path(
                framework_update_path
            ),
            "framework_state_sha256": _sha256_path(framework_state_path),
            "framework_poll_sha256": _sha256_path(framework_poll_path),
            "render_settled_map_sha256": _sha256_path(settled_path),
            "assigned_update_range": [
                min(grouped),
                max(grouped),
            ],
        },
    )


def _masked_frame_sha256(frame: np.ndarray, mask: np.ndarray) -> str:
    normalized = np.ascontiguousarray(frame).copy()
    normalized[mask, :3] = 0
    digest = hashlib.sha256(_MASKED_FRAME_DOMAIN)
    digest.update(np.asarray(normalized.shape, dtype="<u4").tobytes())
    digest.update(normalized.tobytes())
    return f"sha256:{digest.hexdigest()}"


def _frame_sha256(frame: np.ndarray) -> str:
    return f"sha256:{hashlib.sha256(memoryview(np.ascontiguousarray(frame)).cast('B')).hexdigest()}"


def _candidate_metrics(
    left: np.ndarray,
    right: np.ndarray,
    mask: np.ndarray,
) -> tuple[int, int, int, int]:
    difference = left != right
    pixel_difference = np.any(difference, axis=2)
    outside_mismatches = int(np.logical_and(pixel_difference, ~mask).sum())
    mask_mismatches = int(np.logical_and(pixel_difference, mask).sum())
    alpha_mismatches = int(np.logical_and(difference[:, :, 3], mask).sum())
    if np.any(mask):
        maximum_bgr_delta = int(
            np.abs(
                left[:, :, :3][mask].astype(np.int16)
                - right[:, :, :3][mask].astype(np.int16)
            ).max(initial=0)
        )
    else:
        maximum_bgr_delta = 0
    return (
        outside_mismatches,
        mask_mismatches,
        alpha_mismatches,
        maximum_bgr_delta,
    )


def _select_target_pairs(
    left: _RunFrames,
    right: _RunFrames,
    *,
    mask: np.ndarray,
    target_updates: Sequence[int],
    maximum_mask_pixel_mismatches: int,
    maximum_bgr_channel_delta: int,
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
        accepted: list[tuple[int, int, int, int, int, dict[str, Any]]] = []
        for left_sequence in left_sequences:
            left_frame = left.raw[left_sequence]
            left_sha = _frame_sha256(left_frame)
            if left_sha != left.frame_hashes[left_sequence]:
                _fail("target_verification_selected_frame_hash_mismatch")
            for right_sequence in right_sequences:
                right_frame = right.raw[right_sequence]
                right_sha = _frame_sha256(right_frame)
                if right_sha != right.frame_hashes[right_sequence]:
                    _fail("target_verification_selected_frame_hash_mismatch")
                (
                    outside_mismatches,
                    mask_mismatches,
                    alpha_mismatches,
                    maximum_delta,
                ) = _candidate_metrics(left_frame, right_frame, mask)
                if (
                    outside_mismatches
                    or alpha_mismatches
                    or mask_mismatches > maximum_mask_pixel_mismatches
                    or maximum_delta > maximum_bgr_channel_delta
                ):
                    continue
                left_masked_sha = _masked_frame_sha256(left_frame, mask)
                right_masked_sha = _masked_frame_sha256(right_frame, mask)
                if left_masked_sha != right_masked_sha:
                    _fail("target_verification_masked_digest_mismatch")
                row = {
                    "framework_update": update,
                    "r1_sequence": left_sequence,
                    "r2_sequence": right_sequence,
                    "r1_frame_sha256": left_sha,
                    "r2_frame_sha256": right_sha,
                    "comparison_bgra_sha256": left_masked_sha,
                    "outside_mask_mismatch_count": outside_mismatches,
                    "mask_mismatch_count": mask_mismatches,
                    "mask_alpha_mismatch_count": alpha_mismatches,
                    "maximum_mask_bgr_channel_delta": maximum_delta,
                    "full_frame_exact": left_sha == right_sha,
                }
                accepted.append(
                    (
                        mask_mismatches,
                        maximum_delta,
                        left_sequence,
                        right_sequence,
                        0 if left_sha == right_sha else 1,
                        row,
                    )
                )
        if not accepted:
            unmatched.append(update)
            continue
        accepted.sort(key=lambda item: item[:5])
        selected.append(accepted[0][5])
    return selected, absent, unmatched


def _target_criteria(
    *,
    selected: Sequence[Mapping[str, Any]],
    target_updates: Sequence[int],
    absent: Sequence[int],
    unmatched: Sequence[int],
    maximum_mask_pixel_mismatches: int,
    maximum_bgr_channel_delta: int,
) -> list[dict[str, Any]]:
    """Build fail-closed criteria for one complete target window.

    Every comparison criterion depends on complete coverage.  This prevents an
    empty or partial selection from passing through Python's vacuous ``all``
    semantics while the coverage criterion alone fails.
    """

    complete = (
        len(selected) == len(target_updates)
        and not absent
        and not unmatched
    )
    monotonic = complete and all(
        after["r1_sequence"] > before["r1_sequence"]
        and after["r2_sequence"] > before["r2_sequence"]
        for before, after in zip(selected, selected[1:])
    )

    def observed_maximum(field: str) -> int | None:
        if not selected:
            return None
        return max(int(row[field]) for row in selected)

    def complete_and_all(field: str, predicate: Any) -> bool:
        return complete and all(predicate(row[field]) for row in selected)

    return [
        {
            "name": "complete_target_update_coverage",
            "observed": len(selected),
            "required": len(target_updates),
            "absent_update_count": len(absent),
            "unmatched_update_count": len(unmatched),
            "status": "PASS" if complete else "FAIL",
        },
        {
            "name": "monotonic_selected_frame_pairs",
            "observed": monotonic if complete else None,
            "required": True,
            "evaluated_update_count": len(selected),
            "required_update_count": len(target_updates),
            "status": "PASS" if monotonic else "FAIL",
        },
        {
            "name": "exact_outside_frozen_sparse_mask",
            "observed_maximum_mismatches": observed_maximum(
                "outside_mask_mismatch_count"
            ),
            "required_maximum": 0,
            "evaluated_update_count": len(selected),
            "required_update_count": len(target_updates),
            "status": (
                "PASS"
                if complete_and_all(
                    "outside_mask_mismatch_count", lambda value: value == 0
                )
                else "FAIL"
            ),
        },
        {
            "name": "bounded_frozen_mask_pixels",
            "observed_maximum_mismatches": observed_maximum(
                "mask_mismatch_count"
            ),
            "required_maximum": maximum_mask_pixel_mismatches,
            "evaluated_update_count": len(selected),
            "required_update_count": len(target_updates),
            "status": (
                "PASS"
                if complete_and_all(
                    "mask_mismatch_count",
                    lambda value: value <= maximum_mask_pixel_mismatches,
                )
                else "FAIL"
            ),
        },
        {
            "name": "bounded_frozen_mask_bgr_delta",
            "observed_maximum": observed_maximum(
                "maximum_mask_bgr_channel_delta"
            ),
            "required_maximum": maximum_bgr_channel_delta,
            "evaluated_update_count": len(selected),
            "required_update_count": len(target_updates),
            "status": (
                "PASS"
                if complete_and_all(
                    "maximum_mask_bgr_channel_delta",
                    lambda value: value <= maximum_bgr_channel_delta,
                )
                else "FAIL"
            ),
        },
        {
            "name": "exact_mask_alpha",
            "observed_maximum_mismatches": observed_maximum(
                "mask_alpha_mismatch_count"
            ),
            "required_maximum": 0,
            "evaluated_update_count": len(selected),
            "required_update_count": len(target_updates),
            "status": (
                "PASS"
                if complete_and_all(
                    "mask_alpha_mismatch_count", lambda value: value == 0
                )
                else "FAIL"
            ),
        },
    ]


def verify(
    *,
    session_root: Path,
    mask_preregistration_path: Path,
    mask_execution_binding_path: Path,
    mask_holdout_report_path: Path,
    target_start: int,
    target_end: int,
    maximum_mask_pixel_mismatches: int,
    maximum_bgr_channel_delta: int,
) -> dict[str, Any]:
    session_root = session_root.resolve()
    mask_preregistration_path = mask_preregistration_path.resolve()
    mask_execution_binding_path = mask_execution_binding_path.resolve()
    mask_holdout_report_path = mask_holdout_report_path.resolve()
    if target_end < target_start:
        _fail("target_verification_update_range_invalid")
    if maximum_mask_pixel_mismatches <= 0:
        _fail("target_verification_mask_budget_invalid")
    if not 0 <= maximum_bgr_channel_delta <= 255:
        _fail("target_verification_channel_delta_invalid")
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
        _fail("target_verification_collection_invalid")
    mask, mask_preregistration, mask_chain = _load_mask_chain(
        mask_preregistration_path,
        mask_execution_binding_path,
        mask_holdout_report_path,
    )
    mask_pixel_count = int(mask.sum())
    if maximum_mask_pixel_mismatches > mask_pixel_count:
        _fail("target_verification_mask_budget_invalid")
    holdout_root = _read_json(mask_holdout_report_path).get("session_root")
    if isinstance(holdout_root, str):
        try:
            same_session = Path(holdout_root).resolve() == session_root
        except OSError:
            _fail("target_verification_mask_holdout_path_invalid")
        if same_session:
            _fail("target_verification_holdout_reuse_forbidden")
    calibration_paths = (
        mask_preregistration_path,
        mask_execution_binding_path,
        mask_holdout_report_path,
    )
    left = _load_run(
        session_root,
        "r1",
        mask=mask,
        calibration_paths=calibration_paths,
    )
    right = _load_run(
        session_root,
        "r2",
        mask=mask,
        calibration_paths=calibration_paths,
    )
    if left.metadata.process_instance == right.metadata.process_instance:
        _fail("target_verification_process_instances_not_distinct")
    target_updates = tuple(range(target_start, target_end + 1))
    selected, absent, unmatched = _select_target_pairs(
        left,
        right,
        mask=mask,
        target_updates=target_updates,
        maximum_mask_pixel_mismatches=maximum_mask_pixel_mismatches,
        maximum_bgr_channel_delta=maximum_bgr_channel_delta,
    )
    criteria = _target_criteria(
        selected=selected,
        target_updates=target_updates,
        absent=absent,
        unmatched=unmatched,
        maximum_mask_pixel_mismatches=maximum_mask_pixel_mismatches,
        maximum_bgr_channel_delta=maximum_bgr_channel_delta,
    )
    reasons = [row["name"] for row in criteria if row["status"] != "PASS"]
    return {
        "schema": REPORT_SCHEMA,
        "version": REPORT_VERSION,
        "classification": "pre-packaging-sparse-mask-target-verification",
        "gate_effect": "none_until_independent_pc_golden_verification",
        "status": "PASS" if not reasons else "FAIL",
        "reasons": reasons,
        "session_root": str(session_root),
        "collection": {
            "artifact": str(collection_path),
            "sha256": _sha256_path(collection_path),
            "protocol_state_root": collection.get("protocol_pre_state_root"),
            "host_state_root": collection.get("host_pre_state_root"),
        },
        "mask_calibration": {
            "pixel_count": mask_pixel_count,
            "mask_sha256": _mask_sha256(mask),
            "render_settle_delay_ns": _mapping(
                mask_preregistration.get("comparison"),
                "target_verification_mask_comparison_invalid",
            ).get("render_settle_delay_ns"),
            "chain": list(mask_chain),
        },
        "target": {
            "first_update": target_start,
            "last_update": target_end,
            "required_update_count": len(target_updates),
            "selected_update_count": len(selected),
            "absent_updates": absent,
            "unmatched_updates": unmatched,
            "maximum_mask_pixel_mismatches": (
                maximum_mask_pixel_mismatches
            ),
            "maximum_bgr_channel_delta": maximum_bgr_channel_delta,
            "full_frame_exact_update_count": sum(
                bool(row["full_frame_exact"]) for row in selected
            ),
            "selected_pairs": selected,
        },
        "runs": {
            "r1": dict(left.evidence),
            "r2": dict(right.evidence),
        },
        "criteria": criteria,
        "non_authorizations": [
            "pc_golden",
            "front_insertion",
            "fidelity_gate_open",
        ],
    }


def _write_exclusive(path: Path, report: Mapping[str, Any]) -> str:
    if not path.is_absolute() or path.exists() or not path.parent.is_dir():
        _fail("target_verification_output_invalid")
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
        _fail("target_verification_output_write_failed")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-root", required=True, type=Path)
    parser.add_argument("--mask-preregistration", required=True, type=Path)
    parser.add_argument("--mask-execution-binding", required=True, type=Path)
    parser.add_argument("--mask-holdout-report", required=True, type=Path)
    parser.add_argument("--target-start", required=True, type=int)
    parser.add_argument("--target-end", required=True, type=int)
    parser.add_argument(
        "--maximum-mask-pixel-mismatches",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--maximum-bgr-channel-delta",
        required=True,
        type=int,
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = verify(
            session_root=args.session_root,
            mask_preregistration_path=args.mask_preregistration,
            mask_execution_binding_path=args.mask_execution_binding,
            mask_holdout_report_path=args.mask_holdout_report,
            target_start=args.target_start,
            target_end=args.target_end,
            maximum_mask_pixel_mismatches=(
                args.maximum_mask_pixel_mismatches
            ),
            maximum_bgr_channel_delta=args.maximum_bgr_channel_delta,
        )
        digest = _write_exclusive(args.output.resolve(), report)
    except TargetVerificationError as error:
        print(f"target verification error: {error}", file=sys.stderr)
        return 2
    print(f"{args.output.resolve()}\n{digest}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
