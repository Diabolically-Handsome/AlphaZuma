"""Capacity-safe version of the frozen AlphaZuma multilevel evaluator.

The original evaluator correctly refuses to silently truncate actor-visible
balls beyond the frozen 768-ball observation capacity.  A worker exception,
however, kills the complete vector batch.  This version preserves the exact
observation shape and turns only that exact capacity exception into an
explicit, auditable failure for the affected policy/seed attempt.  Every other
exception still propagates and fails the shard.

The mature matrix implementation remains imported from
``evaluate_zero_shot_multilevel.py``.  Its module ``__file__`` is rebound to
this version before execution so preregistrations and receipts bind these
bytes, not the legacy evaluator bytes.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import gymnasium as gym
import numpy as np

from tools import evaluate_zero_shot_multilevel as legacy
from zuma_rl.human_speedrun import (
    EliteHumanInputConfig,
    HumanSpeedrunWrapper,
    WinFirstRewardConfig,
)
from zuma_rl.revenge_env import FloatObservation, RevengeEnv, RevengeEnvConfig


CAPACITY_ERROR_PREFIX = "actor observation capacity exceeded:"


class ObservationCapacityFailClosed(gym.Wrapper):
    """Convert only actor-capacity overflow into one terminal policy failure."""

    def __init__(self, env: RevengeEnv):
        super().__init__(env)
        self._last_observation: FloatObservation | None = None

    @property
    def revenge_env(self) -> RevengeEnv:
        base = self.env.unwrapped
        if not isinstance(base, RevengeEnv):
            raise RuntimeError("capacity wrapper lost its RevengeEnv")
        return base

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[FloatObservation, dict[str, Any]]:
        observation, info = self.env.reset(seed=seed, options=options)
        self._last_observation = np.asarray(observation, dtype=np.float32).copy()
        return observation, info

    def step(
        self,
        action: Any,
    ) -> tuple[FloatObservation, float, bool, bool, dict[str, Any]]:
        try:
            observation, reward, terminated, truncated, info = self.env.step(action)
        except RuntimeError as error:
            message = str(error)
            if not message.startswith(CAPACITY_ERROR_PREFIX):
                raise
            if self._last_observation is None:
                raise RuntimeError(
                    "capacity overflow occurred before a valid observation"
                ) from error

            base = self.revenge_env
            # RevengeEnv stores this tick's complete info before constructing
            # the observation that raises.  Reuse it so score/tick/action
            # accounting remains exact at the fail-closed boundary.
            info = dict(base._last_info)
            visible_balls = int(info["visible_balls"])
            if visible_balls <= int(base.config.max_balls):
                raise RuntimeError(
                    "capacity exception did not contain an actual overflow"
                ) from error
            info.update(
                {
                    "outcome": "loss",
                    "observation_capacity_overflow": True,
                    "failure_reason": "actor_observation_capacity_overflow",
                    "capacity_limit_balls": int(base.config.max_balls),
                    "visible_balls_at_failure": visible_balls,
                    "capacity_error": message,
                    "evaluation_fail_closed": True,
                }
            )
            info.pop("TimeLimit.truncated", None)
            base._episode_done = True
            score_delta = int(info.get("score_delta", 0))
            native_reward = (
                base.config.score_reward_scale * score_delta
                + base.config.step_penalty
                + base.config.loss_reward
            )
            terminal_observation = self._last_observation.copy()
            return terminal_observation, float(native_reward), True, False, info

        self._last_observation = np.asarray(observation, dtype=np.float32).copy()
        return observation, reward, terminated, truncated, info


class CapacityAwareEpisodeTracker(legacy.EpisodeTracker):
    """Retain fail-closed capacity evidence in each completed attempt row."""

    def finish(
        self,
        *,
        info: dict[str, Any],
        model: dict[str, Any],
        threshold_ticks: int,
    ) -> dict[str, Any]:
        row = super().finish(
            info=info,
            model=model,
            threshold_ticks=threshold_ticks,
        )
        overflow = bool(info.get("observation_capacity_overflow", False))
        row["observation_capacity_overflow"] = overflow
        if overflow:
            row.update(
                {
                    "failure_reason": info["failure_reason"],
                    "capacity_limit_balls": int(info["capacity_limit_balls"]),
                    "visible_balls_at_failure": int(
                        info["visible_balls_at_failure"]
                    ),
                    "capacity_error": str(info["capacity_error"]),
                    "evaluation_fail_closed": True,
                }
            )
        return row


def _make_capacity_safe_env_factory(
    *,
    original_root: Path,
    level_id: str,
    profile_mode: str,
    base_config: RevengeEnvConfig,
    input_config: EliteHumanInputConfig,
    reward_config: WinFirstRewardConfig,
) -> Any:
    def make_env() -> HumanSpeedrunWrapper:
        base = RevengeEnv(
            config=base_config,
            level_id=level_id,
            root=original_root,
            hard=False,
            curve_index=0,
            profile_mode=profile_mode,
        )
        guarded = ObservationCapacityFailClosed(base)
        return HumanSpeedrunWrapper(
            guarded,
            input_config=input_config,
            reward_config=reward_config,
        )

    return make_env


def main(argv: list[str] | None = None) -> int:
    # Legacy validation and receipts resolve __file__ at runtime.  Rebinding it
    # makes every evaluator hash/path check refer to this versioned executable.
    legacy.__file__ = __file__
    legacy._make_env_factory = _make_capacity_safe_env_factory
    legacy.EpisodeTracker = CapacityAwareEpisodeTracker
    return legacy.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
