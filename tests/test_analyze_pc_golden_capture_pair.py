"""Tests for the read-only PC capture-pair inventory helper."""

from __future__ import annotations

from tools.analyze_pc_golden_capture_pair import _contiguous_ranges


def test_contiguous_ranges_are_sorted_deduplicated_and_inclusive() -> None:
    assert _contiguous_ranges([8, 7, 3, 3, 4, 10]) == (
        (3, 4),
        (7, 8),
        (10, 10),
    )


def test_contiguous_ranges_accept_empty_input() -> None:
    assert _contiguous_ranges([]) == ()
