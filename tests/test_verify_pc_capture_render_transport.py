from __future__ import annotations

import pytest

from tools.analyze_pc_capture_poll_alignment import CollapsedVisual
from tools.analyze_pc_capture_visual_alignment import _mask_sha256
from tools.verify_pc_capture_render_transport import (
    RenderTransportError,
    _alignment_metrics,
    _frozen_mask,
)


def _collapsed(*digests: str) -> tuple[CollapsedVisual, ...]:
    return tuple(
        CollapsedVisual(digest, index, index)
        for index, digest in enumerate(digests)
    )


def test_alignment_metrics_measure_bounded_insertions_and_local_phase() -> None:
    metrics = _alignment_metrics(
        _collapsed("a", "b", "left-only", "c"),
        _collapsed("a", "b", "right-1", "right-2", "c"),
        left_draws=(20, 21, 22, 24),
        right_draws=(10, 11, 12, 13, 13),
        left_event_updates={20: 100, 21: 101, 22: 102, 24: 104},
        right_event_updates={10: 100, 11: 101, 12: 102, 13: 103},
    )

    assert metrics["matched_collapsed_visual_count"] == 3
    assert metrics["maximum_unmatched_collapsed_gap_left"] == 1
    assert metrics["maximum_unmatched_collapsed_gap_right"] == 2
    assert metrics["same_poll_event_update_count"] == 2
    assert metrics["maximum_absolute_poll_event_update_delta"] == 1
    assert metrics["local_draw_delta_span"] == 1
    assert metrics["poll_event_update_delta_histogram"] == [[0, 2], [1, 1]]
    assert metrics["draw_delta_histogram"] == [[10, 2], [11, 1]]


def test_frozen_mask_requires_exact_coordinate_digest() -> None:
    mask = _frozen_mask(
        {
            "width": 3,
            "height": 2,
            "pixel_count": 1,
            "coordinates_xy": [[2, 1]],
            "mask_sha256": _mask_sha256(
                __import__("numpy").array(
                    [[False, False, False], [False, False, True]],
                    dtype=bool,
                )
            ),
        }
    )
    assert mask.shape == (2, 3)
    assert bool(mask[1, 2]) is True

    with pytest.raises(RenderTransportError, match="mask_invalid"):
        _frozen_mask(
            {
                "width": 3,
                "height": 2,
                "pixel_count": 1,
                "coordinates_xy": [[2, 1]],
                "mask_sha256": "sha256:" + "0" * 64,
            }
        )
