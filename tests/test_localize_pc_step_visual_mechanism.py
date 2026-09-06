from __future__ import annotations

import numpy as np

from tools.localize_pc_step_visual_mechanism import (
    _difference,
    _dominant_rgb,
    _mask_sha256,
)


def test_difference_reports_exact_mask_bounds_and_channel_delta() -> None:
    left = np.zeros((600, 800, 3), dtype=np.uint8)
    right = left.copy()
    right[588, 797] = (3, 4, 5)

    mask, report = _difference(left, right)

    assert mask.sum() == 1
    assert report["mismatched_rgb_pixel_count"] == 1
    assert report["difference_bounds_xyxy"] == [797, 588, 798, 589]
    assert report["maximum_absolute_delta_by_rgb_channel"] == [3, 4, 5]
    assert report["mismatch_mask_sha256"] == _mask_sha256(mask)


def test_dominant_rgb_counts_the_full_viewport() -> None:
    image = np.full((600, 800, 3), 255, dtype=np.uint8)
    image[0, 0] = (0, 0, 0)

    dominant = _dominant_rgb(image)

    assert dominant["rgb"] == [255, 255, 255]
    assert dominant["pixel_count"] == 800 * 600 - 1
