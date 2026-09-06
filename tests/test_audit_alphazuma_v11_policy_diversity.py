from __future__ import annotations

import numpy as np
import pytest

from tools.audit_alphazuma_v11_policy_diversity import (
    action_signature,
    summarize_action_pair,
    validate_actions,
)


def _all_valid_masks(rows: int) -> np.ndarray:
    return np.ones((rows, 183), dtype=np.bool_)


def test_pair_summary_distinguishes_exact_from_semantic_changes() -> None:
    left = np.asarray([[0, 10], [1, 10], [1, 179], [2, 5]], dtype=np.int64)
    right = np.asarray([[0, 20], [1, 12], [0, 179], [2, 5]], dtype=np.int64)

    result = summarize_action_pair(left, right)

    assert result["exact_action_difference"] == {"count": 3, "rate": 0.75}
    assert result["semantic_action_difference"] == {"count": 2, "rate": 0.5}
    assert result["verb_difference"] == {"count": 1, "rate": 0.25}
    assert result["both_fire_circular_aim_distance"] == {
        "mean": 2.0,
        "median": 2.0,
        "max": 2,
    }


def test_semantic_difference_is_symmetric() -> None:
    left = np.asarray([[0, 30], [1, 179], [2, 40]], dtype=np.int64)
    right = np.asarray([[1, 30], [1, 1], [2, 90]], dtype=np.int64)

    forward = summarize_action_pair(left, right)
    reverse = summarize_action_pair(right, left)

    assert forward["semantic_action_difference"] == reverse[
        "semantic_action_difference"
    ]
    assert forward["exact_action_difference"] == reverse[
        "exact_action_difference"
    ]
    assert forward["both_fire_circular_aim_distance"]["mean"] == 2.0


def test_action_validation_rejects_mask_violation() -> None:
    actions = np.asarray([[1, 42], [0, 7]], dtype=np.int64)
    masks = _all_valid_masks(2)
    masks[0, 3 + 42] = False

    with pytest.raises(ValueError, match="factor 1 violates"):
        validate_actions(actions, masks)


def test_action_signature_is_canonical_and_sensitive() -> None:
    first = np.asarray([[1, 2], [0, 179]], dtype=np.int32)
    same = np.asarray([[1, 2], [0, 179]], dtype=np.int64)
    changed = np.asarray([[1, 3], [0, 179]], dtype=np.int64)

    assert action_signature(first) == action_signature(same)
    assert action_signature(first) != action_signature(changed)
