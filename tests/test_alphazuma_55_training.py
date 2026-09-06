from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pytest

from tools.build_alphazuma_55_training_route import _level_weight
from tools.train_alphazuma_55 import CachedOriginalGameCatalog
from zuma_rl.original_data import OriginalGameCatalog


def test_cached_catalog_reuses_and_evicts_decoded_levels(monkeypatch) -> None:
    calls: list[tuple[str, bool]] = []

    def fake_load(self, level_id: str, *, hard: bool = False):
        calls.append((level_id, hard))
        return (level_id, hard, len(calls))

    monkeypatch.setattr(OriginalGameCatalog, "load_level", fake_load)
    catalog = object.__new__(CachedOriginalGameCatalog)
    catalog.maximum_levels = 2
    catalog._loaded_levels = OrderedDict()

    first = catalog.load_level("Jungle1")
    assert catalog.load_level("jungle1") is first
    catalog.load_level("Jungle2")
    catalog.load_level("Jungle1")
    catalog.load_level("Jungle3")

    assert calls == [
        ("Jungle1", False),
        ("Jungle2", False),
        ("Jungle3", False),
    ]
    assert list(catalog._loaded_levels) == [
        ("jungle1", False),
        ("jungle3", False),
    ]


def test_frontier_curriculum_emphasizes_late_and_dual_levels() -> None:
    assert _level_weight("volcano10", mode="frontier") > _level_weight(
        "Jungle1",
        mode="frontier",
    )
    assert _level_weight("volcano3", mode="frontier") > _level_weight(
        "volcano4",
        mode="frontier",
    )
    assert _level_weight("village3", mode="balanced") > _level_weight(
        "village4",
        mode="balanced",
    )


def test_distillation_reservoir_is_bounded_and_deterministic() -> None:
    pytest.importorskip("torch")
    from tools.distill_alphazuma_55 import _reservoir_add

    left: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    right: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    left_rng = np.random.default_rng(17)
    right_rng = np.random.default_rng(17)
    for seen in range(1, 101):
        row = (
            np.asarray([seen], dtype=np.float32),
            np.asarray([1, seen % 180], dtype=np.int64),
            np.asarray([True, True], dtype=np.bool_),
        )
        _reservoir_add(
            left,
            seen=seen,
            capacity=8,
            row=row,
            rng=left_rng,
        )
        _reservoir_add(
            right,
            seen=seen,
            capacity=8,
            row=row,
            rng=right_rng,
        )

    assert len(left) == len(right) == 8
    assert [int(row[0][0]) for row in left] == [
        int(row[0][0]) for row in right
    ]
