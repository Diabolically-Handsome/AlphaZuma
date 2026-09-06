"""Strict identity-aware comparison for PPO boundary receipts."""

from __future__ import annotations

import math
from typing import Any


class BoundaryEvidenceError(ValueError):
    """Raised when boundary receipts cannot support a paired comparison."""


def _indexed_episodes(
    receipt: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    if receipt.get("schema") != "zuma-rl.ppo-boundary-evaluation":
        raise BoundaryEvidenceError("boundary receipt schema is invalid")
    if receipt.get("version") != 4:
        raise BoundaryEvidenceError("boundary receipt version 4 is required")
    if receipt.get("episode_identity_protocol") != (
        "vector_env_index_per_env_episode_ordinal_and_terminal_info"
    ):
        raise BoundaryEvidenceError("episode identity protocol is invalid")

    identities = receipt.get("episode_identities")
    rewards = receipt.get("episode_rewards")
    lengths = receipt.get("episode_lengths")
    episode_count = receipt.get("episode_count")
    if not isinstance(episode_count, int) or episode_count < 1:
        raise BoundaryEvidenceError("episode count is invalid")
    if not all(isinstance(values, list) for values in (identities, rewards, lengths)):
        raise BoundaryEvidenceError("episode arrays are invalid")
    if not (
        len(identities)
        == len(rewards)
        == len(lengths)
        == episode_count
    ):
        raise BoundaryEvidenceError("episode arrays have different lengths")

    seed = receipt.get("seed")
    if not isinstance(seed, int):
        raise BoundaryEvidenceError("boundary seed is invalid")
    indexed: dict[str, dict[str, Any]] = {}
    for identity, reward, length in zip(
        identities, rewards, lengths, strict=True
    ):
        if not isinstance(identity, dict):
            raise BoundaryEvidenceError("episode identity is invalid")
        vector_env_index = identity.get("vector_env_index")
        episode_ordinal = identity.get("episode_ordinal_within_env")
        pairing_key = identity.get("pairing_key")
        initial_env_seed = identity.get("initial_env_seed")
        if (
            not isinstance(vector_env_index, int)
            or vector_env_index < 0
            or not isinstance(episode_ordinal, int)
            or episode_ordinal < 0
        ):
            raise BoundaryEvidenceError("episode identity indices are invalid")
        expected_key = f"env-{vector_env_index}:episode-{episode_ordinal}"
        if pairing_key != expected_key:
            raise BoundaryEvidenceError("episode pairing key is invalid")
        if initial_env_seed != seed + vector_env_index:
            raise BoundaryEvidenceError("initial environment seed is invalid")
        terminal_outcome = identity.get("terminal_outcome")
        time_limit_truncated = identity.get("time_limit_truncated")
        terminal_score = identity.get("terminal_score")
        terminal_ticks = identity.get("terminal_ticks")
        if terminal_outcome not in {"win", "loss", None}:
            raise BoundaryEvidenceError("terminal outcome is invalid")
        if not isinstance(time_limit_truncated, bool):
            raise BoundaryEvidenceError("truncation flag is invalid")
        if time_limit_truncated == (terminal_outcome is not None):
            raise BoundaryEvidenceError("terminal outcome is inconsistent")
        if not isinstance(terminal_score, int) or terminal_score < 0:
            raise BoundaryEvidenceError("terminal score is invalid")
        if not isinstance(terminal_ticks, int) or terminal_ticks < 1:
            raise BoundaryEvidenceError("terminal ticks are invalid")
        if pairing_key in indexed:
            raise BoundaryEvidenceError("episode pairing key is duplicated")
        if not isinstance(reward, (int, float)) or not math.isfinite(reward):
            raise BoundaryEvidenceError("episode reward is invalid")
        if not isinstance(length, int) or length < 1:
            raise BoundaryEvidenceError("episode length is invalid")
        indexed[pairing_key] = {
            "identity": identity,
            "reward": float(reward),
            "length": int(length),
        }
    return indexed


def _outcome(*, episode: dict[str, Any], max_ticks: int) -> str:
    if episode["length"] > max_ticks:
        raise BoundaryEvidenceError("episode length exceeds max_ticks")
    identity = episode["identity"]
    if identity["time_limit_truncated"]:
        return "truncation"
    outcome = identity["terminal_outcome"]
    if outcome not in {"win", "loss"}:
        raise BoundaryEvidenceError("terminal outcome is unavailable")
    return str(outcome)


def compare_boundary_receipts(
    pretrain: dict[str, Any],
    posttrain: dict[str, Any],
    *,
    max_ticks: int,
    severe_regression_threshold: float = 5.0,
) -> dict[str, Any]:
    """Compare pre/post receipts by stable vector-stream identity."""

    if max_ticks < 1:
        raise BoundaryEvidenceError("max_ticks must be positive")
    if severe_regression_threshold <= 0:
        raise BoundaryEvidenceError(
            "severe regression threshold must be positive"
        )
    if pretrain.get("phase") != "pretrain":
        raise BoundaryEvidenceError("pretrain receipt phase is invalid")
    if posttrain.get("phase") != "posttrain":
        raise BoundaryEvidenceError("posttrain receipt phase is invalid")
    for field in (
        "deterministic",
        "seed",
        "action_sampling_seed",
        "vector_env_count",
        "episode_count",
    ):
        if pretrain.get(field) != posttrain.get(field):
            raise BoundaryEvidenceError(f"boundary field differs: {field}")

    pre_index = _indexed_episodes(pretrain)
    post_index = _indexed_episodes(posttrain)
    if pre_index.keys() != post_index.keys():
        raise BoundaryEvidenceError("pre/post episode identity sets differ")

    ordered_keys = sorted(
        pre_index,
        key=lambda key: (
            pre_index[key]["identity"]["vector_env_index"],
            pre_index[key]["identity"]["episode_ordinal_within_env"],
        ),
    )
    paired_episodes = []
    pre_outcomes: list[str] = []
    post_outcomes: list[str] = []
    reward_deltas: list[float] = []
    for pairing_key in ordered_keys:
        pre = pre_index[pairing_key]
        post = post_index[pairing_key]
        pre_outcome = _outcome(episode=pre, max_ticks=max_ticks)
        post_outcome = _outcome(episode=post, max_ticks=max_ticks)
        reward_delta = post["reward"] - pre["reward"]
        pre_outcomes.append(pre_outcome)
        post_outcomes.append(post_outcome)
        reward_deltas.append(reward_delta)
        paired_episodes.append(
            {
                "pairing_key": pairing_key,
                "vector_env_index": pre["identity"]["vector_env_index"],
                "episode_ordinal_within_env": pre["identity"][
                    "episode_ordinal_within_env"
                ],
                "initial_env_seed": pre["identity"]["initial_env_seed"],
                "pretrain_terminal_score": pre["identity"][
                    "terminal_score"
                ],
                "posttrain_terminal_score": post["identity"][
                    "terminal_score"
                ],
                "pretrain_terminal_ticks": pre["identity"][
                    "terminal_ticks"
                ],
                "posttrain_terminal_ticks": post["identity"][
                    "terminal_ticks"
                ],
                "pretrain_reward": pre["reward"],
                "posttrain_reward": post["reward"],
                "reward_delta": reward_delta,
                "pretrain_length": pre["length"],
                "posttrain_length": post["length"],
                "pretrain_outcome": pre_outcome,
                "posttrain_outcome": post_outcome,
            }
        )

    episode_count = len(ordered_keys)
    pre_mean = math.fsum(
        pre_index[key]["reward"] for key in ordered_keys
    ) / episode_count
    post_mean = math.fsum(
        post_index[key]["reward"] for key in ordered_keys
    ) / episode_count
    return {
        "schema": "zuma-rl.ppo-boundary-comparison",
        "version": 2,
        "identity_aware": True,
        "outcome_source": "terminal_info",
        "deterministic": bool(pretrain["deterministic"]),
        "seed": int(pretrain["seed"]),
        "action_sampling_seed": pretrain["action_sampling_seed"],
        "episode_count": episode_count,
        "max_ticks": max_ticks,
        "severe_regression_threshold": severe_regression_threshold,
        "pretrain": {
            "wins": pre_outcomes.count("win"),
            "losses": pre_outcomes.count("loss"),
            "truncations": pre_outcomes.count("truncation"),
            "mean_reward": pre_mean,
        },
        "posttrain": {
            "wins": post_outcomes.count("win"),
            "losses": post_outcomes.count("loss"),
            "truncations": post_outcomes.count("truncation"),
            "mean_reward": post_mean,
        },
        "mean_reward_delta": post_mean - pre_mean,
        "paired_reward_improvements": sum(
            delta > 0.0 for delta in reward_deltas
        ),
        "paired_reward_regressions": sum(
            delta < 0.0 for delta in reward_deltas
        ),
        "paired_reward_regressions_exceeding_threshold": sum(
            delta < -severe_regression_threshold for delta in reward_deltas
        ),
        "paired_episodes": paired_episodes,
    }
