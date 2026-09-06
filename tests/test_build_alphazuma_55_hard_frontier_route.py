import pytest

from tools import build_alphazuma_55_hard_frontier_route as builder


def test_require_disjoint_accepts_adjacent_ranges() -> None:
    builder._require_disjoint(
        ("new", 200, 299),
        [("old-a", 0, 99), ("old-b", 100, 199), ("old-c", 300, 399)],
    )


def test_require_disjoint_rejects_overlap() -> None:
    with pytest.raises(ValueError, match="overlaps old"):
        builder._require_disjoint(("new", 150, 249), [("old", 100, 199)])


def test_require_disjoint_rejects_duplicate_id() -> None:
    with pytest.raises(ValueError, match="duplicates old"):
        builder._require_disjoint(("old", 200, 299), [("old", 0, 99)])
