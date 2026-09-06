from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from tools.verify_pc_render_settled_target import (
    _RunFrames,
    _frame_sha256,
    _select_target_pairs,
    _target_criteria,
)


def _run(
    run_id: str,
    frames: np.ndarray,
    records: dict[int, tuple[int, ...]],
) -> _RunFrames:
    return _RunFrames(
        run_id=run_id,
        raw=frames,
        metadata=SimpleNamespace(),
        records_by_update=records,
        frame_hashes=tuple(_frame_sha256(frame) for frame in frames),
        evidence={},
    )


def _frames() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left = np.zeros((2, 4, 5, 4), dtype=np.uint8)
    left[:, :, :, 3] = 255
    right = left.copy()
    mask = np.zeros((4, 5), dtype=bool)
    mask[0, 0] = True
    mask[3, 4] = True
    right[:, 0, 0, :3] = (1, 2, 3)
    right[:, 3, 4, :3] = (3, 2, 1)
    return left, right, mask


def test_sparse_target_accepts_only_frozen_mask_differences() -> None:
    left, right, mask = _frames()
    records = {10: (0,), 11: (1,)}

    selected, absent, unmatched = _select_target_pairs(
        _run("r1", left, records),
        _run("r2", right, records),
        mask=mask,
        target_updates=(10, 11),
        maximum_mask_pixel_mismatches=2,
        maximum_bgr_channel_delta=3,
    )

    assert absent == []
    assert unmatched == []
    assert [row["framework_update"] for row in selected] == [10, 11]
    assert all(row["outside_mask_mismatch_count"] == 0 for row in selected)
    assert all(row["mask_mismatch_count"] == 2 for row in selected)
    assert all(row["maximum_mask_bgr_channel_delta"] == 3 for row in selected)
    assert all(row["mask_alpha_mismatch_count"] == 0 for row in selected)


def test_sparse_target_rejects_one_outside_mask_pixel() -> None:
    left, right, mask = _frames()
    right[0, 2, 2, 0] = 1

    selected, absent, unmatched = _select_target_pairs(
        _run("r1", left, {10: (0,)}),
        _run("r2", right, {10: (0,)}),
        mask=mask,
        target_updates=(10,),
        maximum_mask_pixel_mismatches=2,
        maximum_bgr_channel_delta=3,
    )

    assert selected == []
    assert absent == []
    assert unmatched == [10]


def test_sparse_target_rejects_mask_alpha_or_excess_delta() -> None:
    left, right, mask = _frames()
    right[0, 0, 0, 3] = 254
    right[1, 0, 0, 0] = 4

    selected, absent, unmatched = _select_target_pairs(
        _run("r1", left, {10: (0,), 11: (1,)}),
        _run("r2", right, {10: (0,), 11: (1,)}),
        mask=mask,
        target_updates=(10, 11),
        maximum_mask_pixel_mismatches=2,
        maximum_bgr_channel_delta=3,
    )

    assert selected == []
    assert absent == []
    assert unmatched == [10, 11]


def test_sparse_target_reports_a_missing_update_before_comparison() -> None:
    left, right, mask = _frames()

    selected, absent, unmatched = _select_target_pairs(
        _run("r1", left, {10: (0,), 11: (1,)}),
        _run("r2", right, {10: (0,)}),
        mask=mask,
        target_updates=(10, 11),
        maximum_mask_pixel_mismatches=2,
        maximum_bgr_channel_delta=3,
    )

    assert [row["framework_update"] for row in selected] == [10]
    assert absent == [11]
    assert unmatched == []


def test_partial_window_fails_all_dependent_criteria() -> None:
    selected = [
        {
            "framework_update": 10,
            "r1_sequence": 0,
            "r2_sequence": 0,
            "outside_mask_mismatch_count": 0,
            "mask_mismatch_count": 2,
            "mask_alpha_mismatch_count": 0,
            "maximum_mask_bgr_channel_delta": 3,
        }
    ]

    criteria = _target_criteria(
        selected=selected,
        target_updates=(10, 11),
        absent=(11,),
        unmatched=(),
        maximum_mask_pixel_mismatches=2,
        maximum_bgr_channel_delta=3,
    )

    assert {row["status"] for row in criteria} == {"FAIL"}
    assert criteria[1]["observed"] is None
    assert all(row.get("evaluated_update_count") == 1 for row in criteria[1:])


def test_complete_window_passes_all_comparison_criteria() -> None:
    selected = [
        {
            "framework_update": update,
            "r1_sequence": sequence,
            "r2_sequence": sequence + 1,
            "outside_mask_mismatch_count": 0,
            "mask_mismatch_count": 2,
            "mask_alpha_mismatch_count": 0,
            "maximum_mask_bgr_channel_delta": 255,
        }
        for sequence, update in enumerate((10, 11))
    ]

    criteria = _target_criteria(
        selected=selected,
        target_updates=(10, 11),
        absent=(),
        unmatched=(),
        maximum_mask_pixel_mismatches=2,
        maximum_bgr_channel_delta=255,
    )

    assert {row["status"] for row in criteria} == {"PASS"}
