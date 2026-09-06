"""Tests for identity-aware PPO boundary comparisons."""

from __future__ import annotations

import pytest

from zuma_rl.boundary_evidence import (
    BoundaryEvidenceError,
    compare_boundary_receipts,
)


def _receipt(
    *,
    phase: str,
    order: list[tuple[int, int]],
    rewards: list[float],
    lengths: list[int],
    outcomes: list[str | None] | None = None,
) -> dict[str, object]:
    seed = 1000
    if outcomes is None:
        outcomes = ["win"] * len(order)
    return {
        "schema": "zuma-rl.ppo-boundary-evaluation",
        "version": 4,
        "phase": phase,
        "deterministic": True,
        "seed": seed,
        "action_sampling_seed": None,
        "vector_env_count": 2,
        "episode_count": len(order),
        "episode_identity_protocol": (
            "vector_env_index_per_env_episode_ordinal_and_terminal_info"
        ),
        "episode_identities": [
            {
                "pairing_key": f"env-{env}:episode-{ordinal}",
                "vector_env_index": env,
                "episode_ordinal_within_env": ordinal,
                "initial_env_seed": seed + env,
                "terminal_outcome": outcome,
                "native_outcome": outcome,
                "terminal_score": 2000,
                "terminal_ticks": length,
                "time_limit_truncated": outcome is None,
            }
            for (env, ordinal), outcome, length in zip(
                order, outcomes, lengths, strict=True
            )
        ],
        "episode_rewards": rewards,
        "episode_lengths": lengths,
    }


def test_comparison_pairs_by_identity_not_completion_order() -> None:
    pretrain = _receipt(
        phase="pretrain",
        order=[(1, 0), (0, 0), (1, 1), (0, 1)],
        rewards=[20.0, 30.0, 40.0, 50.0],
        lengths=[100, 200, 300, 400],
    )
    posttrain = _receipt(
        phase="posttrain",
        order=[(0, 1), (1, 1), (0, 0), (1, 0)],
        rewards=[55.0, 35.0, 32.0, 18.0],
        lengths=[410, 310, 210, 110],
    )

    result = compare_boundary_receipts(
        pretrain, posttrain, max_ticks=12_000
    )

    assert [
        episode["pairing_key"] for episode in result["paired_episodes"]
    ] == [
        "env-0:episode-0",
        "env-0:episode-1",
        "env-1:episode-0",
        "env-1:episode-1",
    ]
    assert [
        episode["reward_delta"] for episode in result["paired_episodes"]
    ] == [2.0, 5.0, -2.0, -5.0]
    assert result["paired_reward_regressions_exceeding_threshold"] == 0
    assert result["mean_reward_delta"] == 0.0


def test_comparison_rejects_v3_receipts_without_terminal_info() -> None:
    pretrain = _receipt(
        phase="pretrain",
        order=[(0, 0)],
        rewards=[30.0],
        lengths=[100],
    )
    posttrain = _receipt(
        phase="posttrain",
        order=[(0, 0)],
        rewards=[31.0],
        lengths=[100],
    )
    pretrain["version"] = 3

    with pytest.raises(BoundaryEvidenceError, match="version 4"):
        compare_boundary_receipts(pretrain, posttrain, max_ticks=12_000)


def test_comparison_rejects_different_identity_sets() -> None:
    pretrain = _receipt(
        phase="pretrain",
        order=[(0, 0)],
        rewards=[30.0],
        lengths=[100],
    )
    posttrain = _receipt(
        phase="posttrain",
        order=[(1, 0)],
        rewards=[31.0],
        lengths=[100],
    )

    with pytest.raises(BoundaryEvidenceError, match="identity sets differ"):
        compare_boundary_receipts(pretrain, posttrain, max_ticks=12_000)


def test_comparison_uses_terminal_outcome_not_reward_threshold() -> None:
    pretrain = _receipt(
        phase="pretrain",
        order=[(0, 0)],
        rewards=[50.0],
        lengths=[100],
        outcomes=["loss"],
    )
    posttrain = _receipt(
        phase="posttrain",
        order=[(0, 0)],
        rewards=[5.0],
        lengths=[100],
        outcomes=["win"],
    )

    result = compare_boundary_receipts(
        pretrain, posttrain, max_ticks=12_000
    )

    assert result["outcome_source"] == "terminal_info"
    assert result["pretrain"]["losses"] == 1
    assert result["posttrain"]["wins"] == 1
