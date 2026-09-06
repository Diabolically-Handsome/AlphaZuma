"""Tests for factor-level masked-policy drift summaries."""

from __future__ import annotations

import numpy as np
import pytest

from tools.audit_maskable_policy_drift import summarize_factor


def test_summarize_factor_counts_flips_and_small_reference_margins() -> None:
    result = summarize_factor(
        forward_kl=np.asarray([0.0, 0.2, 0.1]),
        total_variation=np.asarray([0.0, 0.3, 0.2]),
        reference_actions=np.asarray([0, 1, 2]),
        candidate_actions=np.asarray([0, 0, 1]),
        reference_margins=np.asarray([0.8, 0.005, 0.04]),
        candidate_margins=np.asarray([0.7, 0.002, 0.03]),
    )

    assert result["sample_count"] == 3
    assert result["deterministic_argmax_flip_count"] == 2
    assert result["deterministic_argmax_flip_rate"] == pytest.approx(2 / 3)
    assert result["flips_with_reference_margin_at_most_1e_3"] == 0
    assert result["flips_with_reference_margin_at_most_1e_2"] == 1
    assert result["flips_with_reference_margin_at_most_5e_2"] == 2
    assert result["forward_kl"]["max"] == pytest.approx(0.2)


def test_summarize_factor_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        summarize_factor(
            forward_kl=np.asarray([0.0, 0.2]),
            total_variation=np.asarray([0.0]),
            reference_actions=np.asarray([0, 1]),
            candidate_actions=np.asarray([0, 1]),
            reference_margins=np.asarray([0.8, 0.1]),
            candidate_margins=np.asarray([0.7, 0.2]),
        )


def test_summarize_factor_rejects_negative_divergence() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        summarize_factor(
            forward_kl=np.asarray([-0.01]),
            total_variation=np.asarray([0.0]),
            reference_actions=np.asarray([0]),
            candidate_actions=np.asarray([0]),
            reference_margins=np.asarray([0.8]),
            candidate_margins=np.asarray([0.7]),
        )
