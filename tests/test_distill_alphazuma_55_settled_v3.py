from __future__ import annotations

import numpy as np

from tools.distill_alphazuma_55_settled_v3 import (
    VERB_NAMES,
    _effective_label,
    _verb_balanced_indices,
)


def _row(verb: int):
    return (
        np.asarray([float(verb)], dtype=np.float32),
        np.asarray([verb, 17], dtype=np.int64),
        np.ones(184, dtype=np.bool_),
    )


def test_masked_verb_becomes_wait_without_losing_target():
    mask = np.ones(184, dtype=np.bool_)
    mask[1] = False

    action, forced = _effective_label(np.asarray([1, 93]), mask)

    assert forced is True
    assert action.tolist() == [0, 93]


def test_unmasked_label_is_unchanged():
    mask = np.ones(184, dtype=np.bool_)

    action, forced = _effective_label(np.asarray([3, 42]), mask)

    assert forced is False
    assert action.tolist() == [3, 42]


def test_verb_balanced_indices_follow_frozen_mix():
    dataset = [_row(0) for _ in range(100)]
    dataset += [_row(1) for _ in range(10)]
    dataset += [_row(2) for _ in range(3)]
    dataset += [_row(3)]
    indices = _verb_balanced_indices(
        dataset=dataset,
        batch_size=100,
        fractions={
            "wait": 0.70,
            "fire": 0.22,
            "swap": 0.07,
            "hop": 0.01,
        },
        rng=np.random.default_rng(7),
    )

    verbs = [int(dataset[int(index)][1][0]) for index in indices]

    assert len(indices) == 100
    assert [verbs.count(index) for index in range(len(VERB_NAMES))] == [70, 22, 7, 1]
