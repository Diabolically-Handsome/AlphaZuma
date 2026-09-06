from __future__ import annotations

from pathlib import Path

import numpy as np

from tools import verify_pc_terminal_state_target as terminal


def test_reference_metrics_separates_mask_and_gameplay() -> None:
    reference = np.zeros((2, 3, 4), dtype=np.uint8)
    frame = reference.copy()
    mask = np.zeros((2, 3), dtype=bool)
    mask[1, 2] = True
    frame[1, 2, 0] = 3
    frame[0, 0, 1] = 4

    metrics = terminal._reference_metrics(frame, reference, mask)

    assert metrics == {
        "outside_mask_mismatch_count": 1,
        "mask_mismatch_count": 1,
        "mask_alpha_mismatch_count": 0,
        "maximum_mask_bgr_channel_delta": 3,
    }


def _base_report(*, full_exact: bool = True) -> dict[str, object]:
    return {
        "schema": "zuma-rl.pc-render-settled-sparse-target-verification",
        "version": 2,
        "status": "PASS",
        "criteria": [],
        "target": {
            "selected_pairs": [
                {
                    "framework_update": 10,
                    "r1_sequence": 2,
                    "r2_sequence": 3,
                    "r1_frame_sha256": "sha256:r1",
                    "r2_frame_sha256": "sha256:r2",
                    "full_frame_exact": full_exact,
                }
            ]
        },
    }


def _patch_sources(monkeypatch, *, frame_delta: int = 0) -> None:
    mask = np.zeros((600, 800), dtype=bool)
    mask[599, 799] = True
    reference = np.zeros((600, 800, 4), dtype=np.uint8)
    frame = reference.copy()
    frame[599, 799, 0] = frame_delta
    monkeypatch.setattr(
        terminal,
        "verify_sparse_target",
        lambda **_kwargs: _base_report(),
    )
    monkeypatch.setattr(
        terminal,
        "_load_mask_chain",
        lambda *_args: (mask, {}, ()),
    )
    monkeypatch.setattr(
        terminal,
        "_reference_frame",
        lambda *_args, **_kwargs: reference,
    )
    monkeypatch.setattr(
        terminal,
        "_selected_frame",
        lambda *_args, **_kwargs: frame,
    )


def test_terminal_verifier_passes_full_exact_bounded_reference(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _patch_sources(monkeypatch, frame_delta=3)

    report = terminal.verify(
        session_root=tmp_path,
        mask_preregistration_path=tmp_path / "pre.json",
        mask_execution_binding_path=tmp_path / "execution.json",
        mask_holdout_report_path=tmp_path / "holdout.json",
        target_start=10,
        target_end=10,
        maximum_mask_pixel_mismatches=1,
        maximum_bgr_channel_delta=255,
        terminal_update=10,
        terminal_reference_path=tmp_path / "reference.png",
        terminal_reference_sha256="sha256:reference",
        maximum_reference_mask_bgr_delta=3,
    )

    assert report["status"] == "PASS"
    assert report["reasons"] == []


def test_terminal_verifier_rejects_reference_delta_over_bound(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _patch_sources(monkeypatch, frame_delta=4)

    report = terminal.verify(
        session_root=tmp_path,
        mask_preregistration_path=tmp_path / "pre.json",
        mask_execution_binding_path=tmp_path / "execution.json",
        mask_holdout_report_path=tmp_path / "holdout.json",
        target_start=10,
        target_end=10,
        maximum_mask_pixel_mismatches=1,
        maximum_bgr_channel_delta=255,
        terminal_update=10,
        terminal_reference_path=tmp_path / "reference.png",
        terminal_reference_sha256="sha256:reference",
        maximum_reference_mask_bgr_delta=3,
    )

    assert report["status"] == "FAIL"
    assert report["reasons"] == [
        "terminal_reference_bounded_frozen_mask_delta"
    ]
