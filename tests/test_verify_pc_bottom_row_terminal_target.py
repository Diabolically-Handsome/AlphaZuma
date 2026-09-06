from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from tools.verify_pc_bottom_row_terminal_target import (
    _bottom_row_metrics,
    _select_pairs,
)


def test_bottom_row_metrics_separate_gameplay_and_edge() -> None:
    left = np.zeros((600, 800, 4), dtype=np.uint8)
    right = left.copy()
    right[599, 799, :3] = 121
    right[10, 20, 0] = 1

    metrics = _bottom_row_metrics(left, right)

    assert metrics == {
        "interior_mismatch_count": 1,
        "edge_mismatch_count": 1,
        "edge_alpha_mismatch_count": 0,
        "maximum_edge_bgr_channel_delta": 121,
    }


def test_select_pairs_accepts_one_exact_alpha_edge_pixel() -> None:
    left_frame = np.zeros((600, 800, 4), dtype=np.uint8)
    right_frame = left_frame.copy()
    right_frame[599, 799, :3] = 121
    left = SimpleNamespace(
        records_by_update={10: (0,)},
        raw=np.asarray([left_frame]),
        frame_hashes=("sha256:left",),
    )
    right = SimpleNamespace(
        records_by_update={10: (0,)},
        raw=np.asarray([right_frame]),
        frame_hashes=("sha256:right",),
    )

    selected, absent, unmatched = _select_pairs(
        left,
        right,
        target_updates=(10,),
        maximum_edge_mismatches=1,
    )

    assert not absent
    assert not unmatched
    assert selected[0]["edge_mismatch_count"] == 1
    assert selected[0]["interior_mismatch_count"] == 0


def test_select_pairs_rejects_second_edge_pixel() -> None:
    left_frame = np.zeros((600, 800, 4), dtype=np.uint8)
    right_frame = left_frame.copy()
    right_frame[599, 798:800, :3] = 121
    left = SimpleNamespace(
        records_by_update={10: (0,)},
        raw=np.asarray([left_frame]),
        frame_hashes=("sha256:left",),
    )
    right = SimpleNamespace(
        records_by_update={10: (0,)},
        raw=np.asarray([right_frame]),
        frame_hashes=("sha256:right",),
    )

    selected, absent, unmatched = _select_pairs(
        left,
        right,
        target_updates=(10,),
        maximum_edge_mismatches=1,
    )

    assert not selected
    assert not absent
    assert unmatched == [10]
