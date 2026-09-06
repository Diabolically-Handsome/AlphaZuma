"""Verify a complete render-settled target and its frozen terminal state.

The base sparse-target verifier establishes independently recomputed update
labels and a monotonic pair for every requested update.  This wrapper adds two
fail-closed requirements used by the front-insertion campaign:

* every selected replay pair must be full-frame exact; and
* the selected terminal frame must match a reference frozen before collection,
  exactly outside the independently calibrated transport mask.

The reference is design evidence only.  A PASS remains pre-packaging evidence;
the PC Golden packager and verifier must still recompute the replay contract.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import numpy as np

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tools.verify_pc_render_settled_target import (
    TargetVerificationError,
    _frame_sha256,
    _load_mask_chain,
    _sha256_path,
    _write_exclusive,
    verify as verify_sparse_target,
)
from zuma_rl.pc_protocol_evidence import PcDxgiCaptureMetadata


REPORT_SCHEMA = "zuma-rl.pc-render-settled-terminal-state-verification"
REPORT_VERSION = 1


def _fail(reason: str) -> None:
    raise TargetVerificationError(reason)


def _reference_frame(path: Path, *, expected_sha256: str) -> np.ndarray:
    if _sha256_path(path) != expected_sha256:
        _fail("terminal_state_reference_sha256_mismatch")
    try:
        from PIL import Image

        with Image.open(path) as source:
            if source.size != (800, 600):
                _fail("terminal_state_reference_geometry_invalid")
            rgba = np.asarray(source.convert("RGBA"), dtype=np.uint8)
    except TargetVerificationError:
        raise
    except Exception:
        _fail("terminal_state_reference_decode_failed")
    return np.ascontiguousarray(rgba[:, :, [2, 1, 0, 3]])


def _selected_frame(
    session_root: Path,
    *,
    run_id: str,
    sequence: int,
    expected_sha256: str,
) -> np.ndarray:
    run_root = session_root / f"run-{run_id}"
    metadata = PcDxgiCaptureMetadata.read(
        run_root / "capture" / "metadata.json"
    )
    if (metadata.width, metadata.height) != (800, 600):
        _fail("terminal_state_capture_geometry_invalid")
    if not 0 <= sequence < metadata.frame_count:
        _fail("terminal_state_selected_sequence_invalid")
    raw_path = run_root / "capture" / "frames.bgra.raw"
    expected_bytes = metadata.frame_count * 800 * 600 * 4
    try:
        if raw_path.stat().st_size != expected_bytes:
            _fail("terminal_state_raw_size_mismatch")
        raw = np.memmap(
            raw_path,
            dtype=np.uint8,
            mode="r",
            shape=(metadata.frame_count, 600, 800, 4),
        )
        frame = np.ascontiguousarray(raw[sequence])
    except TargetVerificationError:
        raise
    except OSError:
        _fail("terminal_state_raw_unreadable")
    if _frame_sha256(frame) != expected_sha256:
        _fail("terminal_state_selected_frame_sha256_mismatch")
    return frame


def _reference_metrics(
    frame: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
) -> dict[str, int]:
    if frame.shape != reference.shape or frame.shape[:2] != mask.shape:
        _fail("terminal_state_comparison_geometry_mismatch")
    difference = frame != reference
    pixel_difference = np.any(difference, axis=2)
    outside_mismatches = int(np.logical_and(pixel_difference, ~mask).sum())
    mask_mismatches = int(np.logical_and(pixel_difference, mask).sum())
    mask_alpha_mismatches = int(
        np.logical_and(difference[:, :, 3], mask).sum()
    )
    maximum_mask_bgr_delta = int(
        np.abs(
            frame[:, :, :3][mask].astype(np.int16)
            - reference[:, :, :3][mask].astype(np.int16)
        ).max(initial=0)
    )
    return {
        "outside_mask_mismatch_count": outside_mismatches,
        "mask_mismatch_count": mask_mismatches,
        "mask_alpha_mismatch_count": mask_alpha_mismatches,
        "maximum_mask_bgr_channel_delta": maximum_mask_bgr_delta,
    }


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
    terminal_update: int,
    terminal_reference_path: Path,
    terminal_reference_sha256: str,
    maximum_reference_mask_bgr_delta: int,
) -> dict[str, Any]:
    if not target_start <= terminal_update <= target_end:
        _fail("terminal_state_update_outside_target")
    if not 0 <= maximum_reference_mask_bgr_delta <= 255:
        _fail("terminal_state_reference_delta_invalid")
    session_root = session_root.resolve()
    mask_preregistration_path = mask_preregistration_path.resolve()
    mask_execution_binding_path = mask_execution_binding_path.resolve()
    mask_holdout_report_path = mask_holdout_report_path.resolve()
    terminal_reference_path = terminal_reference_path.resolve()

    base = verify_sparse_target(
        session_root=session_root,
        mask_preregistration_path=mask_preregistration_path,
        mask_execution_binding_path=mask_execution_binding_path,
        mask_holdout_report_path=mask_holdout_report_path,
        target_start=target_start,
        target_end=target_end,
        maximum_mask_pixel_mismatches=maximum_mask_pixel_mismatches,
        maximum_bgr_channel_delta=maximum_bgr_channel_delta,
    )
    mask, _preregistration, _chain = _load_mask_chain(
        mask_preregistration_path,
        mask_execution_binding_path,
        mask_holdout_report_path,
    )
    target = base.get("target")
    if not isinstance(target, Mapping):
        _fail("terminal_state_base_target_invalid")
    selected = target.get("selected_pairs")
    if not isinstance(selected, list):
        _fail("terminal_state_base_target_invalid")
    terminal_rows = [
        row
        for row in selected
        if isinstance(row, Mapping)
        and row.get("framework_update") == terminal_update
    ]
    if len(terminal_rows) != 1:
        _fail("terminal_state_selected_pair_missing")
    terminal_row = terminal_rows[0]
    reference = _reference_frame(
        terminal_reference_path,
        expected_sha256=terminal_reference_sha256,
    )
    comparisons: dict[str, Mapping[str, Any]] = {}
    for run_id in ("r1", "r2"):
        sequence = terminal_row.get(f"{run_id}_sequence")
        frame_sha256 = terminal_row.get(f"{run_id}_frame_sha256")
        if (
            isinstance(sequence, bool)
            or not isinstance(sequence, int)
            or not isinstance(frame_sha256, str)
        ):
            _fail("terminal_state_selected_pair_invalid")
        frame = _selected_frame(
            session_root,
            run_id=run_id,
            sequence=sequence,
            expected_sha256=frame_sha256,
        )
        comparisons[run_id] = {
            "sequence": sequence,
            "frame_sha256": frame_sha256,
            **_reference_metrics(frame, reference, mask),
        }

    required_update_count = target_end - target_start + 1
    all_full_frame_exact = (
        len(selected) == required_update_count
        and all(
            isinstance(row, Mapping) and row.get("full_frame_exact") is True
            for row in selected
        )
    )
    criteria = [
        {
            "name": "base_sparse_target_verification",
            "observed": base.get("status"),
            "required": "PASS",
            "status": "PASS" if base.get("status") == "PASS" else "FAIL",
        },
        {
            "name": "all_selected_pairs_full_frame_exact",
            "observed": sum(
                isinstance(row, Mapping)
                and row.get("full_frame_exact") is True
                for row in selected
            ),
            "required": required_update_count,
            "status": "PASS" if all_full_frame_exact else "FAIL",
        },
        {
            "name": "terminal_reference_exact_outside_frozen_mask",
            "observed_maximum": max(
                int(row["outside_mask_mismatch_count"])
                for row in comparisons.values()
            ),
            "required_maximum": 0,
            "status": (
                "PASS"
                if all(
                    row["outside_mask_mismatch_count"] == 0
                    for row in comparisons.values()
                )
                else "FAIL"
            ),
        },
        {
            "name": "terminal_reference_bounded_frozen_mask_delta",
            "observed_maximum": max(
                int(row["maximum_mask_bgr_channel_delta"])
                for row in comparisons.values()
            ),
            "required_maximum": maximum_reference_mask_bgr_delta,
            "status": (
                "PASS"
                if all(
                    row["maximum_mask_bgr_channel_delta"]
                    <= maximum_reference_mask_bgr_delta
                    and row["mask_mismatch_count"] <= int(mask.sum())
                    for row in comparisons.values()
                )
                else "FAIL"
            ),
        },
        {
            "name": "terminal_reference_exact_mask_alpha",
            "observed_maximum": max(
                int(row["mask_alpha_mismatch_count"])
                for row in comparisons.values()
            ),
            "required_maximum": 0,
            "status": (
                "PASS"
                if all(
                    row["mask_alpha_mismatch_count"] == 0
                    for row in comparisons.values()
                )
                else "FAIL"
            ),
        },
    ]
    reasons = [row["name"] for row in criteria if row["status"] != "PASS"]
    return {
        **base,
        "schema": REPORT_SCHEMA,
        "version": REPORT_VERSION,
        "classification": (
            "pre-packaging-full-exact-terminal-state-verification"
        ),
        "status": "PASS" if not reasons else "FAIL",
        "reasons": reasons,
        "base_target_verifier": {
            "schema": base.get("schema"),
            "version": base.get("version"),
            "status": base.get("status"),
        },
        "terminal_state_binding": {
            "framework_update": terminal_update,
            "reference_artifact": str(terminal_reference_path),
            "reference_sha256": terminal_reference_sha256,
            "mask_pixel_count": int(mask.sum()),
            "maximum_reference_mask_bgr_channel_delta": (
                maximum_reference_mask_bgr_delta
            ),
            "runs": comparisons,
        },
        "criteria": [*base.get("criteria", []), *criteria],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-root", required=True, type=Path)
    parser.add_argument("--mask-preregistration", required=True, type=Path)
    parser.add_argument("--mask-execution-binding", required=True, type=Path)
    parser.add_argument("--mask-holdout-report", required=True, type=Path)
    parser.add_argument("--target-start", required=True, type=int)
    parser.add_argument("--target-end", required=True, type=int)
    parser.add_argument(
        "--maximum-mask-pixel-mismatches", required=True, type=int
    )
    parser.add_argument(
        "--maximum-bgr-channel-delta", required=True, type=int
    )
    parser.add_argument("--terminal-update", required=True, type=int)
    parser.add_argument("--terminal-reference", required=True, type=Path)
    parser.add_argument("--terminal-reference-sha256", required=True)
    parser.add_argument(
        "--maximum-reference-mask-bgr-delta", required=True, type=int
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
            terminal_update=args.terminal_update,
            terminal_reference_path=args.terminal_reference,
            terminal_reference_sha256=args.terminal_reference_sha256,
            maximum_reference_mask_bgr_delta=(
                args.maximum_reference_mask_bgr_delta
            ),
        )
        digest = _write_exclusive(args.output.resolve(), report)
    except TargetVerificationError as error:
        print(f"terminal target verification error: {error}", file=sys.stderr)
        return 2
    print(f"{args.output.resolve()}\n{digest}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
