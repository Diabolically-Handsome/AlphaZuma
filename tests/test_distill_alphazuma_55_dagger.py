from __future__ import annotations

import numpy as np

from tools.distill_alphazuma_55_dagger import _is_exact_masked_action_legal
from tools.materialize_alphazuma_55_dagger import _ranking_key


def _full_mask() -> np.ndarray:
    return np.ones(4 + 180, dtype=np.bool_)


def test_exact_masked_action_accepts_legal_factorized_action() -> None:
    mask = _full_mask()
    assert _is_exact_masked_action_legal(np.asarray([1, 179]), mask)


def test_exact_masked_action_rejects_disabled_factor() -> None:
    mask = _full_mask()
    mask[1] = False
    assert not _is_exact_masked_action_legal(np.asarray([1, 42]), mask)
    mask = _full_mask()
    mask[4 + 42] = False
    assert not _is_exact_masked_action_legal(np.asarray([1, 42]), mask)


def test_exact_masked_action_rejects_shape_and_bounds_errors() -> None:
    mask = _full_mask()
    assert not _is_exact_masked_action_legal(np.asarray([4, 0]), mask)
    assert not _is_exact_masked_action_legal(np.asarray([0, 180]), mask)
    assert not _is_exact_masked_action_legal(np.asarray([0]), mask)
    assert not _is_exact_masked_action_legal(np.asarray([0, 0]), mask[:-1])


def test_source_ranking_is_win_then_score_then_speed_then_hash() -> None:
    rows = [
        {"model_sha256": "sha256:c", "wins": 1, "total_score": 900, "median_winning_ticks": 90},
        {"model_sha256": "sha256:b", "wins": 2, "total_score": 100, "median_winning_ticks": 200},
        {"model_sha256": "sha256:a", "wins": 2, "total_score": 100, "median_winning_ticks": 100},
        {"model_sha256": "sha256:d", "wins": 0, "total_score": 5000, "median_winning_ticks": None},
    ]
    ranked = sorted(rows, key=_ranking_key)
    assert [row["model_sha256"] for row in ranked] == [
        "sha256:a",
        "sha256:b",
        "sha256:c",
        "sha256:d",
    ]
