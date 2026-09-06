"""Credit-horizon-corrected successor to ``train_overnight_multilevel.py``.

Lineage.  The v1 trainer drives all seven current routes but bakes in four
verified pathologies: it hardcodes ``gamma=0.995`` (a ~2 s credit horizon at
the native 100 Hz tick against 12,000-20,000-tick episodes) and re-enforces
that value on resume; the frozen 12,000-tick ceiling is below realistic
long-route win lengths (12,800-20,400 ticks), so wins are unreachable there;
time-limit truncation pays the same ``-10`` terminal as death, flattening
long-level return variance to ~0; and the warm-start recipe (entropy
2e-4-5e-4, lr 1e-6) removes exploration entirely.

This v2 mirrors the v1 receipt-driven structure and fixes all four without
editing any frozen file:

* ``gamma`` is configurable (default ``0.999``) and resume no longer forces
  a hardcoded value over the user's choice — the configured
  hyperparameters are applied to the loaded checkpoint and both the stored
  and the applied values are recorded in the run receipt.
* ``max_ticks`` is a per-level map: long routes (id prefixes ``coast``,
  ``grotto``, ``volcano``, case-insensitive) default to 24,000 native
  ticks, everything else to 12,000, with exact-id overrides.
* Time-limit truncation pays no terminal penalty.  The episode ends with
  ``truncated=True`` and only the time/shaping terms, so SB3 2.9
  bootstraps the value target; a terminated loss still pays ``-10`` and a
  win ``+10``.
* Score shaping is configurable (cap default ``10.0`` per episode instead
  of the v1 ``0.01``), and fresh-start defaults are exploratory (entropy
  ``0.01``, lr ``3e-4``); documented warm-start defaults are gentler but
  still alive (entropy ``2e-3``, lr ``3e-5``).
* ``--use-park-settle-wrapper`` optionally trains through
  :class:`~zuma_rl.park_settle_action_wrapper.ParkSettleActionWrapper`.
  Models trained with it use the macro ``MultiDiscrete([3, aim_bins])``
  action interface and are NOT action-compatible with per-tick policies;
  such runs start from a fresh policy (``initial_model: null``).

Frozen v1 artifacts are reused by import only (helpers, callback builder);
nothing under the existing receipt roots is modified.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass, replace
from functools import partial
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from tools.train_overnight_multilevel import (
    _build_callback_class,
    _finite_positive,
    _mapping,
    _normalize_weights,
    _parse_utc,
    _positive_int,
    _read_json,
    _runtime,
    _sha256,
    _utc_now,
    _validate_catalog,
    _write_json_atomic,
)
from zuma_rl.human_speedrun import (
    EliteHumanInputConfig,
    HumanSpeedrunWrapper,
    WinFirstRewardConfig,
)
from zuma_rl.observable_human_speedrun import ObservableHumanSpeedrunWrapper
from zuma_rl.park_settle_action_wrapper import (
    ParkSettleActionConfig,
    ParkSettleActionWrapper,
    resolve_human_speedrun_wrapper,
)
from zuma_rl.revenge_core import SUPPORTED_PROFILE_MODE
from zuma_rl.revenge_env import RevengeEnv, RevengeEnvConfig


FloatObservation = NDArray[np.float32]

PREREGISTRATION_SCHEMA_V2 = "zuma-rl.overnight-multilevel-preregistration-v2"

# Fresh-start optimisation defaults.  The v1 warm-start recipe (entropy
# 2e-4-5e-4 with lr 1e-6 on top of a teacher-distilled policy) produced no
# exploration at all; fresh runs need a live entropy bonus and a normal
# PPO learning rate.
DEFAULT_GAMMA = 0.999
DEFAULT_GAE_LAMBDA = 0.95
DEFAULT_FRESH_LEARNING_RATE = 3e-4
DEFAULT_FRESH_ENTROPY_COEF = 0.01
# Documented warm-start defaults, used only when a resume run supplies
# neither value: gentle enough not to destroy a distilled policy, alive
# enough to keep exploring.
WARM_START_LEARNING_RATE = 3e-5
WARM_START_ENTROPY_COEF = 2e-3

DEFAULT_SCORE_PROGRESS_REWARD_CAP = 10.0
DEFAULT_MAX_TICKS = 12_000
DEFAULT_LONG_HORIZON_MAX_TICKS = 24_000
DEFAULT_LONG_HORIZON_PREFIXES = ("coast", "grotto", "volcano")

REWARD_PROFILE_ID_V2 = "win-time-score-truncation-neutral-v2"


@dataclass(frozen=True, slots=True)
class TrainerV2Config:
    """Trainer-level knobs that were hardcoded or absent in v1.

    ``max_ticks_overrides`` holds ``(casefolded_level_id, ticks)`` pairs and
    wins over the prefix rule; the prefix rule wins over
    ``default_max_ticks``.
    """

    gamma: float = DEFAULT_GAMMA
    gae_lambda: float = DEFAULT_GAE_LAMBDA
    score_progress_reward_cap: float = DEFAULT_SCORE_PROGRESS_REWARD_CAP
    default_max_ticks: int = DEFAULT_MAX_TICKS
    long_horizon_max_ticks: int = DEFAULT_LONG_HORIZON_MAX_TICKS
    long_horizon_level_prefixes: tuple[str, ...] = DEFAULT_LONG_HORIZON_PREFIXES
    max_ticks_overrides: tuple[tuple[str, int], ...] = ()
    use_park_settle_wrapper: bool = False
    use_motor_observable: bool = False

    def __post_init__(self) -> None:
        if not 0.0 < self.gamma < 1.0:
            raise ValueError("gamma must be inside (0, 1)")
        if not 0.0 < self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be inside (0, 1]")
        if not math.isfinite(self.score_progress_reward_cap):
            raise ValueError("score_progress_reward_cap must be finite")
        if self.score_progress_reward_cap < 0.0:
            raise ValueError("score_progress_reward_cap cannot be negative")
        for name in ("default_max_ticks", "long_horizon_max_ticks"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        for prefix in self.long_horizon_level_prefixes:
            if not prefix or prefix != prefix.casefold():
                raise ValueError(
                    "long-horizon prefixes must be non-empty and casefolded"
                )
        for level, ticks in self.max_ticks_overrides:
            if not level or level != level.casefold():
                raise ValueError(
                    "max_ticks_overrides keys must be non-empty and casefolded"
                )
            if int(ticks) < 1:
                raise ValueError("max_ticks_overrides values must be positive")

    def max_ticks_for_level(self, level_id: str) -> int:
        key = str(level_id).casefold()
        for level, ticks in self.max_ticks_overrides:
            if level == key:
                return int(ticks)
        for prefix in self.long_horizon_level_prefixes:
            if key.startswith(prefix):
                return int(self.long_horizon_max_ticks)
        return int(self.default_max_ticks)

    def contract(self) -> dict[str, Any]:
        return {
            "schema": "zuma-rl.overnight-multilevel-trainer-v2-config",
            "version": 1,
            **{key: value for key, value in asdict(self).items()},
        }


_CONFIG_FIELDS = (
    "gamma",
    "gae_lambda",
    "score_progress_reward_cap",
    "default_max_ticks",
    "long_horizon_max_ticks",
    "long_horizon_level_prefixes",
    "max_ticks_overrides",
    "use_park_settle_wrapper",
    "use_motor_observable",
)


def _normalized_max_ticks_overrides(
    value: Any,
) -> tuple[tuple[str, int], ...]:
    if isinstance(value, Mapping):
        items = list(value.items())
    else:
        items = [tuple(row) for row in value]
    return tuple(
        sorted((str(level).casefold(), int(ticks)) for level, ticks in items)
    )


def build_trainer_config(
    base: Mapping[str, Any] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> TrainerV2Config:
    """Merge preregistered values and explicit overrides over the defaults.

    ``overrides`` (typically CLI flags) win over ``base`` (typically the
    preregistration's ``trainer_v2`` mapping); ``None`` entries are ignored
    so absent CLI flags never mask preregistered values.
    """

    values: dict[str, Any] = {}
    for source in (base or {}, overrides or {}):
        for field in _CONFIG_FIELDS:
            if field in source and source[field] is not None:
                values[field] = source[field]
    if "long_horizon_level_prefixes" in values:
        values["long_horizon_level_prefixes"] = tuple(
            str(prefix).casefold()
            for prefix in values["long_horizon_level_prefixes"]
        )
    if "max_ticks_overrides" in values:
        values["max_ticks_overrides"] = _normalized_max_ticks_overrides(
            values["max_ticks_overrides"]
        )
    return TrainerV2Config(**values)


# ---------------------------------------------------------------------------
# Reward semantics: dense shaping + truncation-neutral terminal payments
# ---------------------------------------------------------------------------


class _DenseShapingRewardOrderingMixin:
    """Relax the v1 one-second shaping bound while keeping terminal sanity.

    :class:`HumanSpeedrunWrapper` rejects any reward profile whose score
    shaping can outweigh one second of elapsed time, which caps shaping at
    ``0.01`` per episode and starves long levels of dense signal.  The v2
    profile deliberately allows a larger cap; this override keeps the
    guarantee that a win's worst return beats a *terminated* failure's best
    return and drops only the one-second dominance rule.
    """

    reward_config: WinFirstRewardConfig
    revenge_env: RevengeEnv

    def _validate_reward_ordering(self) -> None:
        rewards = self.reward_config
        max_ticks = self.revenge_env.config.max_ticks
        minimum_win = (
            rewards.win_reward
            + rewards.time_penalty_per_native_tick * max_ticks
        )
        maximum_terminated_failure = (
            rewards.failure_reward + rewards.score_progress_reward_cap
        )
        if minimum_win <= maximum_terminated_failure:
            raise ValueError(
                "reward contract does not guarantee that every win outranks "
                "every terminated failure"
            )


class DenseShapingHumanSpeedrunWrapper(
    _DenseShapingRewardOrderingMixin,
    HumanSpeedrunWrapper,
):
    """Actuator wrapper accepting the dense v2 score-shaping cap."""


class DenseShapingObservableHumanSpeedrunWrapper(
    _DenseShapingRewardOrderingMixin,
    ObservableHumanSpeedrunWrapper,
):
    """Motor-observable actuator wrapper accepting the dense v2 cap."""


class TruncationNeutralRewardWrapper(gym.Wrapper):
    """Remove the terminal failure payment from time-limit truncation.

    ``HumanSpeedrunWrapper._speedrun_reward`` pays ``failure_reward`` on
    ``terminated or truncated``, so under v1 a truncated 12,000-tick episode
    and a death are indistinguishable to the critic and long-route return
    variance collapses to ~0.  This wrapper adds back the failure payment on
    the truncation step only (a terminated loss keeps its ``-10``, a win its
    ``+10``) and leaves ``truncated=True`` set, so SB3 2.9 bootstraps the
    value target from the terminal observation instead of treating the
    time limit as death.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        human = resolve_human_speedrun_wrapper(env)
        self._failure_reward = float(human.reward_config.failure_reward)

    def action_masks(self) -> NDArray[np.bool_]:
        return self.env.action_masks()

    def step(
        self,
        action: Any,
    ) -> tuple[FloatObservation, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(
            action
        )
        if truncated and not terminated and info.get("outcome") != "win":
            reward = float(reward) - self._failure_reward
            info = dict(info)
            terms = dict(info.get("reward_terms", {}))
            terms["outcome"] = 0.0
            info["reward_terms"] = terms
            info["truncation_terminal_reward_removed"] = True
        return observation, float(reward), terminated, truncated, info


def _reward_config_v2(
    trainer_config: TrainerV2Config,
    reward_profile: Mapping[str, Any] | None = None,
) -> WinFirstRewardConfig:
    """Build the v2 reward profile; the cap always follows the trainer.

    A preregistered ``environment.reward_profile`` may override the terminal
    and time terms, but ``score_progress_reward_cap`` is owned by
    :class:`TrainerV2Config` so that one receipt governs it.
    """

    values: dict[str, Any] = {
        "profile_id": REWARD_PROFILE_ID_V2,
        "win_reward": 10.0,
        "failure_reward": -10.0,
        "time_penalty_per_native_tick": -0.0001,
    }
    if reward_profile is not None:
        for key in (
            "profile_id",
            "win_reward",
            "failure_reward",
            "time_penalty_per_native_tick",
        ):
            if key in reward_profile:
                values[key] = reward_profile[key]
    return WinFirstRewardConfig(
        profile_id=str(values["profile_id"]),
        win_reward=float(values["win_reward"]),
        failure_reward=float(values["failure_reward"]),
        time_penalty_per_native_tick=float(
            values["time_penalty_per_native_tick"]
        ),
        score_progress_reward_cap=float(
            trainer_config.score_progress_reward_cap
        ),
    )


def _make_human_env_v2(
    *,
    level_id: str,
    base_config: RevengeEnvConfig,
    input_config: EliteHumanInputConfig,
    trainer_config: TrainerV2Config,
    reward_profile: Mapping[str, Any] | None = None,
    original_root: Path | None = None,
    park_settle_config: ParkSettleActionConfig | None = None,
) -> gym.Env:
    """Build the full v2 stack for one level.

    ``RevengeEnv`` (per-level ``max_ticks``) -> dense-shaping actuator
    wrapper -> truncation-neutral rewards -> optional park-settle macro
    actions.
    """

    config = replace(
        base_config,
        max_ticks=trainer_config.max_ticks_for_level(level_id),
    )
    base = RevengeEnv(
        config=config,
        level_id=level_id,
        root=original_root,
        hard=False,
        curve_index=0,
        profile_mode=SUPPORTED_PROFILE_MODE,
    )
    wrapper_class = (
        DenseShapingObservableHumanSpeedrunWrapper
        if trainer_config.use_motor_observable
        else DenseShapingHumanSpeedrunWrapper
    )
    human = wrapper_class(
        base,
        input_config=input_config,
        reward_config=_reward_config_v2(trainer_config, reward_profile),
    )
    env: gym.Env = TruncationNeutralRewardWrapper(human)
    if trainer_config.use_park_settle_wrapper:
        env = ParkSettleActionWrapper(env, config=park_settle_config)
    return env


# ---------------------------------------------------------------------------
# PPO hyperparameters
# ---------------------------------------------------------------------------


def _ppo_hyperparameters(
    trainer_config: TrainerV2Config,
    run_spec: Mapping[str, Any],
    *,
    resuming: bool,
) -> dict[str, Any]:
    """Resolve the algorithm hyperparameters for one run.

    Explicit run values always win; absent values fall back to the
    fresh-start defaults, or to the documented warm-start defaults when
    resuming an optimizer state.
    """

    learning_rate = run_spec.get("learning_rate")
    if learning_rate is None:
        learning_rate = (
            WARM_START_LEARNING_RATE if resuming else DEFAULT_FRESH_LEARNING_RATE
        )
    entropy_coef = run_spec.get("entropy_coef")
    if entropy_coef is None:
        entropy_coef = (
            WARM_START_ENTROPY_COEF if resuming else DEFAULT_FRESH_ENTROPY_COEF
        )
    return {
        "learning_rate": float(learning_rate),
        "n_steps": int(run_spec["rollout_steps"]),
        "batch_size": int(run_spec["batch_size"]),
        "n_epochs": int(run_spec["ppo_epochs"]),
        "gamma": float(trainer_config.gamma),
        "gae_lambda": float(trainer_config.gae_lambda),
        "ent_coef": float(entropy_coef),
    }


def _resume_custom_objects(
    hyperparameters: Mapping[str, Any],
) -> dict[str, Any]:
    """Custom objects applied when loading a resume checkpoint.

    v1 refused to resume unless the checkpoint carried its hardcoded
    ``gamma=0.995``; v2 instead applies the *configured* hyperparameters to
    the loaded model (SB3 replaces the stored attributes before
    ``_setup_model`` rebuilds schedules and buffers) and records both the
    stored and the applied values in the run receipt.
    """

    return {
        key: hyperparameters[key]
        for key in (
            "learning_rate",
            "n_steps",
            "batch_size",
            "n_epochs",
            "gamma",
            "gae_lambda",
            "ent_coef",
        )
    }


# ---------------------------------------------------------------------------
# Multi-level environment (v2: per-level max_ticks + optional macro actions)
# ---------------------------------------------------------------------------


class MultiLevelSpeedrunEnvV2(gym.Env[FloatObservation, Any]):
    """Select one preregistered compatible level at each episode reset.

    Mirrors v1's :class:`MultiLevelHumanSpeedrunEnv` (disjoint deterministic
    episode-seed namespaces, weighted level selection) but builds the v2
    stack, so each level carries its own ``max_ticks`` and the optional
    park-settle macro action interface.
    """

    metadata = RevengeEnv.metadata

    def __init__(
        self,
        *,
        original_root: Path | None,
        levels: Sequence[str],
        weights: Mapping[str, float],
        base_config: RevengeEnvConfig,
        input_config: EliteHumanInputConfig,
        trainer_config: TrainerV2Config,
        reward_profile: Mapping[str, Any] | None,
        selector_seed: int,
        episode_seed_base: int,
        episode_seed_stride: int,
        worker_rank: int,
    ) -> None:
        super().__init__()
        if not levels or len(set(levels)) != len(levels):
            raise ValueError("levels must be non-empty and unique")
        if worker_rank < 0:
            raise ValueError("worker_rank cannot be negative")
        if episode_seed_stride < 1:
            raise ValueError("episode_seed_stride must be positive")
        self.original_root = original_root
        self.levels = tuple(str(level) for level in levels)
        self.base_config = base_config
        self.input_config = input_config
        self.trainer_config = trainer_config
        self.reward_profile = (
            dict(reward_profile) if reward_profile is not None else None
        )
        self.selector_seed = int(selector_seed)
        self.episode_seed_base = int(episode_seed_base)
        self.episode_seed_stride = int(episode_seed_stride)
        self.worker_rank = int(worker_rank)
        self._weights, self._probabilities = _normalize_weights(
            self.levels,
            weights,
        )
        self._selector = np.random.default_rng(self.selector_seed)
        self._episode_index = 0
        self._active_level = self.levels[0]
        self._active_episode_seed: int | None = None
        self._active = self._make_env(self._active_level)
        self.observation_space = self._active.observation_space
        self.action_space = self._active.action_space

    def _make_env(self, level_id: str) -> gym.Env:
        return _make_human_env_v2(
            level_id=level_id,
            base_config=self.base_config,
            input_config=self.input_config,
            trainer_config=self.trainer_config,
            reward_profile=self.reward_profile,
            original_root=self.original_root,
        )

    def _replace_active(self, level_id: str) -> None:
        if level_id == self._active_level:
            return
        self._active.close()
        replacement = self._make_env(level_id)
        if replacement.observation_space != self.observation_space:
            replacement.close()
            raise RuntimeError(f"observation space changed for {level_id}")
        if replacement.action_space != self.action_space:
            replacement.close()
            raise RuntimeError(f"action space changed for {level_id}")
        self._active = replacement
        self._active_level = level_id

    def set_level_weights(self, weights: Mapping[str, float]) -> dict[str, float]:
        self._weights, self._probabilities = _normalize_weights(
            self.levels,
            weights,
        )
        return dict(self._weights)

    def get_curriculum_state(self) -> dict[str, Any]:
        return {
            "worker_rank": self.worker_rank,
            "episode_index": self._episode_index,
            "active_level": self._active_level,
            "active_episode_seed": self._active_episode_seed,
            "weights": dict(self._weights),
        }

    def action_masks(self) -> NDArray[np.bool_]:
        return self._active.action_masks()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[FloatObservation, dict[str, Any]]:
        if seed is not None:
            # SB3 supplies a deterministic per-worker seed on the first
            # reset; fold it into the preregistered selector namespace.
            self._selector = np.random.default_rng(
                np.random.SeedSequence([self.selector_seed, int(seed)])
            )
        selected_index = int(
            self._selector.choice(len(self.levels), p=self._probabilities)
        )
        level_id = self.levels[selected_index]
        self._replace_active(level_id)
        if self._episode_index >= self.episode_seed_stride:
            raise RuntimeError("worker exhausted its preregistered seed namespace")
        episode_seed = (
            self.episode_seed_base
            + self.worker_rank * self.episode_seed_stride
            + self._episode_index
        )
        self._active_episode_seed = episode_seed
        episode_index = self._episode_index
        self._episode_index += 1
        observation, info = self._active.reset(seed=episode_seed, options=options)
        result = dict(info)
        result.update(
            {
                "training_level_id": level_id,
                "training_episode_seed": episode_seed,
                "training_episode_index": episode_index,
                "training_worker_rank": self.worker_rank,
                "training_level_max_ticks": (
                    self.trainer_config.max_ticks_for_level(level_id)
                ),
            }
        )
        return observation, result

    def step(
        self,
        action: Any,
    ) -> tuple[FloatObservation, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self._active.step(
            action
        )
        result = dict(info)
        result.update(
            {
                "training_level_id": self._active_level,
                "training_episode_seed": self._active_episode_seed,
                "training_episode_index": self._episode_index - 1,
                "training_worker_rank": self.worker_rank,
            }
        )
        return observation, reward, terminated, truncated, result

    def render(self) -> Any:
        return self._active.render()

    def close(self) -> None:
        self._active.close()


def _make_worker_v2(
    *,
    original_root: Path | None,
    levels: Sequence[str],
    weights: Mapping[str, float],
    base_config: RevengeEnvConfig,
    input_config: EliteHumanInputConfig,
    trainer_config: TrainerV2Config,
    reward_profile: Mapping[str, Any] | None,
    selector_seed: int,
    episode_seed_base: int,
    episode_seed_stride: int,
    worker_rank: int,
) -> gym.Env:
    from stable_baselines3.common.monitor import Monitor

    environment = MultiLevelSpeedrunEnvV2(
        original_root=original_root,
        levels=levels,
        weights=weights,
        base_config=base_config,
        input_config=input_config,
        trainer_config=trainer_config,
        reward_profile=reward_profile,
        selector_seed=selector_seed + worker_rank,
        episode_seed_base=episode_seed_base,
        episode_seed_stride=episode_seed_stride,
        worker_rank=worker_rank,
    )
    return Monitor(environment)


# ---------------------------------------------------------------------------
# Preregistration validation (v2 schema)
# ---------------------------------------------------------------------------


def _configs_v2(
    prereg: dict[str, Any],
) -> tuple[RevengeEnvConfig, EliteHumanInputConfig, dict[str, Any] | None]:
    environment = _mapping(prereg.get("environment"), "environment")
    base = _mapping(environment.get("base_config"), "environment.base_config")
    input_profile = _mapping(
        environment.get("input_profile"),
        "environment.input_profile",
    )
    reward_profile_raw = environment.get("reward_profile")
    reward_profile = (
        dict(_mapping(reward_profile_raw, "environment.reward_profile"))
        if reward_profile_raw is not None
        else None
    )
    return (
        RevengeEnvConfig(**base),
        EliteHumanInputConfig(
            profile_id=str(input_profile["profile_id"]),
            reaction_delay_ticks=int(input_profile["reaction_delay_ticks"]),
            max_aim_speed_degrees_per_second=float(
                input_profile["max_aim_speed_degrees_per_second"]
            ),
            max_aim_acceleration_degrees_per_second_squared=float(
                input_profile[
                    "max_aim_acceleration_degrees_per_second_squared"
                ]
            ),
            min_button_interval_ticks=int(
                input_profile["min_button_interval_ticks"]
            ),
        ),
        reward_profile,
    )


def _validate_preregistration_v2(path: Path) -> dict[str, Any]:
    prereg = _read_json(path)
    if prereg.get("schema") != PREREGISTRATION_SCHEMA_V2:
        raise ValueError("unexpected preregistration schema")
    if prereg.get("version") != 2:
        raise ValueError("unsupported preregistration version")
    if prereg.get("status") != "FROZEN_BEFORE_TRAINING":
        raise ValueError("preregistration is not frozen before training")
    trainer = _mapping(prereg.get("trainer"), "trainer")
    if Path(str(trainer.get("path", ""))).resolve() != Path(__file__).resolve():
        raise ValueError("preregistration binds another trainer path")
    if trainer.get("sha256") != _sha256(Path(__file__).resolve()):
        raise ValueError("trainer bytes differ from preregistration")
    implementation_raw = prereg.get("implementation")
    if implementation_raw is not None:
        implementation = _mapping(implementation_raw, "implementation")
        for name, raw in implementation.items():
            artifact = _mapping(raw, f"implementation.{name}")
            artifact_path = Path(str(artifact.get("path", ""))).resolve(
                strict=True
            )
            if artifact.get("sha256") != _sha256(artifact_path):
                raise ValueError(f"implementation bytes differ: {name}")
    trainer_v2_raw = prereg.get("trainer_v2")
    if trainer_v2_raw is not None:
        # Raises on invalid values; the merged result is rebuilt in main().
        build_trainer_config(_mapping(trainer_v2_raw, "trainer_v2"))
    initial_model_raw = prereg.get("initial_model")
    if initial_model_raw is not None:
        initial_model = _mapping(initial_model_raw, "initial_model")
        model_path = Path(str(initial_model.get("path", ""))).resolve(
            strict=True
        )
        if initial_model.get("sha256") != _sha256(model_path):
            raise ValueError("initial model bytes differ from preregistration")
    levels = prereg.get("levels")
    if not isinstance(levels, list) or not levels:
        raise ValueError("levels must be a non-empty list")
    ids = [str(_mapping(row, "level").get("id", "")) for row in levels]
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("level ids are empty or duplicated")
    runs = prereg.get("runs")
    if not isinstance(runs, list) or len(runs) < 1:
        raise ValueError("runs must be a non-empty list")
    run_ids: set[str] = set()
    run_directories: set[str] = set()
    episode_ranges: list[tuple[int, int, str]] = []
    for index, raw in enumerate(runs):
        run = _mapping(raw, f"runs[{index}]")
        run_id = str(run.get("id", ""))
        if not run_id or run_id in run_ids:
            raise ValueError("run ids are empty or duplicated")
        run_ids.add(run_id)
        run_directory = str(Path(str(run.get("run_dir", ""))).resolve())
        if run_directory in run_directories:
            raise ValueError("run directories are duplicated")
        run_directories.add(run_directory)
        num_envs = _positive_int(run.get("num_envs"), "run.num_envs")
        rollout_steps = _positive_int(run.get("rollout_steps"), "rollout_steps")
        batch_size = _positive_int(run.get("batch_size"), "batch_size")
        if batch_size > num_envs * rollout_steps:
            raise ValueError("batch size exceeds rollout size")
        for name in (
            "total_steps",
            "ppo_epochs",
            "checkpoint_every",
            "episode_seed_base",
            "episode_seed_stride",
        ):
            _positive_int(run.get(name), f"run.{name}")
        # v2: optimisation coefficients are optional; defaults are the
        # fresh-start (or, on resume, documented warm-start) values.
        if run.get("learning_rate") is not None:
            _finite_positive(run.get("learning_rate"), "run.learning_rate")
        if run.get("entropy_coef") is not None:
            entropy = float(run["entropy_coef"])
            if not math.isfinite(entropy) or entropy < 0.0:
                raise ValueError(
                    "entropy coefficient must be finite and non-negative"
                )
        _mapping(run.get("adaptive"), "run.adaptive")
        run_trainer_v2 = run.get("trainer_v2")
        if run_trainer_v2 is not None:
            build_trainer_config(_mapping(run_trainer_v2, "run.trainer_v2"))
        weights = _mapping(run.get("initial_weights"), "run.initial_weights")
        _normalize_weights(ids, weights)
        resume_curriculum_raw = run.get("resume_curriculum")
        if resume_curriculum_raw is not None:
            resume_curriculum = _mapping(
                resume_curriculum_raw,
                "run.resume_curriculum",
            )
            _normalize_weights(
                ids,
                _mapping(
                    resume_curriculum.get("weights"),
                    "resume_curriculum.weights",
                ),
            )
        resume_raw = run.get("resume_model")
        if resume_raw is not None:
            resume = _mapping(resume_raw, "run.resume_model")
            resume_path = Path(str(resume.get("path", ""))).resolve(strict=True)
            if resume.get("sha256") != _sha256(resume_path):
                raise ValueError("resume model bytes differ from preregistration")
            _positive_int(resume.get("timesteps"), "run.resume_model.timesteps")
        base = int(run["episode_seed_base"])
        stride = int(run["episode_seed_stride"])
        episode_ranges.append((base, base + num_envs * stride - 1, run_id))
    for left_index, left in enumerate(episode_ranges):
        for right in episode_ranges[left_index + 1 :]:
            if max(left[0], right[0]) <= min(left[1], right[1]):
                raise ValueError(
                    f"training seed namespaces overlap: {left[2]} and {right[2]}"
                )
    schedule = _mapping(prereg.get("schedule"), "schedule")
    stop = _parse_utc(str(schedule.get("training_stop_utc", "")))
    end = _parse_utc(str(schedule.get("goal_end_utc", "")))
    if stop >= end:
        raise ValueError("training stop must precede goal end")
    return prereg


def _trainer_config_for_run(
    prereg: dict[str, Any],
    run_spec: Mapping[str, Any] | None,
    cli_overrides: Mapping[str, Any] | None,
) -> TrainerV2Config:
    """Merge defaults <- prereg.trainer_v2 <- run.trainer_v2 <- CLI flags."""

    merged: dict[str, Any] = {}
    prereg_level = prereg.get("trainer_v2")
    if prereg_level is not None:
        merged.update(_mapping(prereg_level, "trainer_v2"))
    if run_spec is not None and run_spec.get("trainer_v2") is not None:
        merged.update(_mapping(run_spec["trainer_v2"], "run.trainer_v2"))
    return build_trainer_config(merged, cli_overrides)


def _prevalidate_spaces_v2(
    *,
    prereg: dict[str, Any],
    original_root: Path | None,
    base_config: RevengeEnvConfig,
    input_config: EliteHumanInputConfig,
    reward_profile: Mapping[str, Any] | None,
    trainer_config: TrainerV2Config,
) -> dict[str, Any]:
    from sb3_contrib import MaskablePPO

    environments: list[dict[str, Any]] = []
    observation_space = None
    action_space = None
    contract: dict[str, Any] | None = None
    for frozen in prereg["levels"]:
        level_id = str(frozen["id"])
        environment = _make_human_env_v2(
            level_id=level_id,
            base_config=base_config,
            input_config=input_config,
            trainer_config=trainer_config,
            reward_profile=reward_profile,
            original_root=original_root,
        )
        try:
            if observation_space is None:
                observation_space = environment.observation_space
                action_space = environment.action_space
            elif (
                environment.observation_space != observation_space
                or environment.action_space != action_space
            ):
                raise ValueError(f"space mismatch for {level_id}")
            human = resolve_human_speedrun_wrapper(environment)
            level_contract = (
                environment.contract()
                if isinstance(environment, ParkSettleActionWrapper)
                else human.contract()
            )
            if contract is None:
                contract = level_contract
            elif level_contract != contract:
                raise ValueError(
                    "human-speedrun contract differs between levels"
                )
            environments.append(
                {
                    "level_id": level_id,
                    "max_ticks": trainer_config.max_ticks_for_level(level_id),
                }
            )
        finally:
            environment.close()
    if observation_space is None or action_space is None:
        raise RuntimeError("no level environment was validated")

    def check_model(path: Path, name: str) -> dict[str, Any]:
        model = MaskablePPO.load(path, device="cpu")
        try:
            if model.observation_space != observation_space:
                raise ValueError(
                    f"{name} observation space differs from the v2 stack"
                )
            if model.action_space != action_space:
                message = f"{name} action space differs from the v2 stack"
                if trainer_config.use_park_settle_wrapper:
                    message += (
                        "; park-settle runs use the macro MultiDiscrete"
                        "([3, aim_bins]) interface and must start fresh "
                        "(initial_model: null)"
                    )
                raise ValueError(message)
            return {
                "path": str(path),
                "sha256": _sha256(path),
                "timesteps": int(model.num_timesteps),
            }
        finally:
            del model

    initial_model_raw = prereg.get("initial_model")
    initial_model = (
        check_model(
            Path(str(_mapping(initial_model_raw, "initial_model")["path"]))
            .resolve(strict=True),
            "initial model",
        )
        if initial_model_raw is not None
        else None
    )
    resume_models: list[dict[str, Any]] = []
    for run in prereg["runs"]:
        resume_raw = run.get("resume_model")
        if resume_raw is None:
            continue
        resume = _mapping(resume_raw, "run.resume_model")
        checked = check_model(
            Path(str(resume["path"])).resolve(strict=True),
            f"resume model for {run['id']}",
        )
        if checked["timesteps"] != int(resume["timesteps"]):
            raise ValueError(
                "resume model timestep differs from preregistration"
            )
        checked["run_id"] = str(run["id"])
        resume_models.append(checked)
    return {
        "levels_verified": len(environments),
        "per_level_max_ticks": environments,
        "observation_shape": list(observation_space.shape),
        "action_nvec": (
            np.asarray(action_space.nvec).tolist()
            if hasattr(action_space, "nvec")
            else None
        ),
        "contract": contract,
        "trainer_config": trainer_config.contract(),
        "initial_model": initial_model,
        "resume_models": resume_models,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _run_training_v2(
    *,
    prereg_path: Path,
    prereg: dict[str, Any],
    run_spec: dict[str, Any],
    original_root: Path | None,
    trainer_config: TrainerV2Config,
    validation: dict[str, Any],
) -> dict[str, Any]:
    import torch
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
    from stable_baselines3.common.save_util import load_from_zip_file
    from stable_baselines3.common.vec_env import SubprocVecEnv

    torch.set_num_threads(1)
    levels = [str(row["id"]) for row in prereg["levels"]]
    base_config, input_config, reward_profile = _configs_v2(prereg)
    run_dir = Path(str(run_spec["run_dir"])).resolve()
    if run_dir.exists():
        raise FileExistsError(f"refusing to reuse run directory: {run_dir}")
    run_dir.mkdir(parents=True)
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir()

    factories = [
        partial(
            _make_worker_v2,
            original_root=original_root,
            levels=levels,
            weights=(
                _mapping(
                    _mapping(
                        run_spec["resume_curriculum"],
                        "run.resume_curriculum",
                    ).get("weights"),
                    "resume_curriculum.weights",
                )
                if run_spec.get("resume_curriculum") is not None
                else run_spec["initial_weights"]
            ),
            base_config=base_config,
            input_config=input_config,
            trainer_config=trainer_config,
            reward_profile=reward_profile,
            selector_seed=int(run_spec["seed"]),
            episode_seed_base=int(run_spec["episode_seed_base"]),
            episode_seed_stride=int(run_spec["episode_seed_stride"]),
            worker_rank=rank,
        )
        for rank in range(int(run_spec["num_envs"]))
    ]
    train_env = SubprocVecEnv(factories, start_method="forkserver")
    started_utc = _utc_now()
    started_monotonic = time.monotonic()
    resume_raw = run_spec.get("resume_model")
    resume_spec = (
        _mapping(resume_raw, "run.resume_model")
        if resume_raw is not None
        else None
    )
    hyperparameters = _ppo_hyperparameters(
        trainer_config,
        run_spec,
        resuming=resume_spec is not None,
    )
    initial_model_raw = prereg.get("initial_model")
    initial_path = (
        Path(str(_mapping(initial_model_raw, "initial_model")["path"]))
        .resolve(strict=True)
        if initial_model_raw is not None
        else None
    )
    source_path = (
        Path(str(resume_spec["path"])).resolve(strict=True)
        if resume_spec is not None
        else initial_path
    )
    source_timesteps = int(resume_spec["timesteps"]) if resume_spec else 0
    receipt = {
        "schema": "zuma-rl.overnight-multilevel-training-config-v2",
        "version": 2,
        "status": "STARTED",
        "started_utc": started_utc,
        "preregistration": {
            "path": str(prereg_path),
            "sha256": _sha256(prereg_path),
        },
        "trainer_sha256": _sha256(Path(__file__).resolve()),
        "run": run_spec,
        "trainer_config": trainer_config.contract(),
        "hyperparameters": hyperparameters,
        "initial_model": prereg.get("initial_model"),
        "training_source_model": (
            {
                "path": str(source_path),
                "sha256": _sha256(source_path),
                "timesteps": source_timesteps,
                "mode": "resume" if resume_spec is not None else "fresh",
            }
            if source_path is not None
            else {"mode": "fresh-random-init"}
        ),
        "levels": prereg["levels"],
        "environment": prereg["environment"],
        "prevalidation": validation,
        "runtime": _runtime(),
    }
    _write_json_atomic(run_dir / "config.json", receipt)

    callback_class = _build_callback_class()
    stop_at = _parse_utc(prereg["schedule"]["training_stop_utc"])
    telemetry_callback = callback_class(
        run_dir=run_dir,
        run_spec=run_spec,
        levels=levels,
        stop_at=stop_at,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(
            1,
            int(run_spec["checkpoint_every"]) // int(run_spec["num_envs"]),
        ),
        save_path=str(checkpoints),
        name_prefix=f"{run_spec['id']}",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    completion: dict[str, Any]
    try:
        initialization: dict[str, Any] = {
            "schema": "zuma-rl.overnight-multilevel-initialization-v2",
            "version": 2,
            "hyperparameters": hyperparameters,
        }
        if resume_spec is None:
            source = (
                MaskablePPO.load(initial_path, device="cpu")
                if initial_path is not None
                else None
            )
            model = MaskablePPO(
                "MlpPolicy",
                train_env,
                policy_kwargs=(
                    copy.deepcopy(source.policy_kwargs)
                    if source is not None
                    else None
                ),
                tensorboard_log=str(run_dir / "tensorboard"),
                seed=int(run_spec["seed"]),
                device=str(run_spec["device"]),
                verbose=1,
                **hyperparameters,
            )
            if source is not None:
                try:
                    source_state = source.policy.state_dict()
                    model.policy.load_state_dict(source_state, strict=True)
                    copied_exactly = all(
                        torch.equal(
                            source_state[name].detach().cpu(),
                            value.detach().cpu(),
                        )
                        for name, value in model.policy.state_dict().items()
                    )
                    if not copied_exactly:
                        raise RuntimeError(
                            "initial policy parameters did not copy bitwise"
                        )
                finally:
                    del source
                initialization["policy_copied_bitwise"] = True
                initialization["source_model_sha256"] = _sha256(initial_path)
            else:
                initialization["policy_copied_bitwise"] = False
                initialization["fresh_random_init"] = True
            initialization["optimizer_state_reset"] = True
        else:
            stored, _, _ = load_from_zip_file(source_path, device="cpu")
            stored_hyperparameters = {
                key: stored.get(key)
                for key in (
                    "learning_rate",
                    "n_steps",
                    "batch_size",
                    "n_epochs",
                    "gamma",
                    "gae_lambda",
                    "ent_coef",
                )
            }
            model = MaskablePPO.load(
                source_path,
                env=train_env,
                device=str(run_spec["device"]),
                tensorboard_log=str(run_dir / "tensorboard"),
                custom_objects=_resume_custom_objects(hyperparameters),
            )
            if int(model.num_timesteps) != source_timesteps:
                raise RuntimeError(
                    "resume checkpoint timestep differs from preregistration"
                )
            applied = {
                key: (
                    float(model.lr_schedule(1.0))
                    if key == "learning_rate"
                    else getattr(model, key)
                )
                for key in stored_hyperparameters
            }
            for key, expected in hyperparameters.items():
                if float(applied[key]) != float(expected):
                    raise RuntimeError(
                        f"configured hyperparameter did not apply: {key}"
                    )
            # v1 raised here unless the checkpoint matched its hardcoded
            # gamma=0.995; v2 records the difference instead of forcing it.
            initialization["checkpoint_hyperparameters"] = (
                stored_hyperparameters
            )
            initialization["hyperparameters_overridden"] = sorted(
                key
                for key, value in stored_hyperparameters.items()
                if value is not None and float(value) != float(applied[key])
            )
            initialization["optimizer_state_reset"] = False
            initialization["optimizer_state_entries"] = len(
                model.policy.optimizer.state_dict().get("state", {})
            )
            model.verbose = 1
            model.seed = int(run_spec["seed"])
            model.set_random_seed(int(run_spec["seed"]))
        initialization["policy_parameters"] = sum(
            parameter.numel() for parameter in model.policy.parameters()
        )
        _write_json_atomic(run_dir / "initialization.json", initialization)
        model.learn(
            total_timesteps=int(run_spec["total_steps"]),
            callback=CallbackList([checkpoint_callback, telemetry_callback]),
            reset_num_timesteps=resume_spec is None,
            progress_bar=False,
        )
        final_path = run_dir / "final_model.zip"
        model.save(final_path)
        completion = {
            "schema": "zuma-rl.overnight-multilevel-completion-v2",
            "version": 2,
            "status": "COMPLETE",
            "completed_utc": _utc_now(),
            "elapsed_seconds": time.monotonic() - started_monotonic,
            "stop_reason": telemetry_callback.stop_reason,
            "requested_steps": int(run_spec["total_steps"]),
            "source_steps": source_timesteps,
            "actual_steps": int(model.num_timesteps),
            "actual_steps_this_run": int(model.num_timesteps) - source_timesteps,
            "final_model": {
                "path": str(final_path),
                "sha256": _sha256(final_path),
                "bytes": final_path.stat().st_size,
            },
            "training_status": _read_json(run_dir / "training_status.json"),
        }
        _write_json_atomic(run_dir / "completion.json", completion)
    except BaseException as error:
        failure = {
            "schema": "zuma-rl.overnight-multilevel-failure-v2",
            "version": 2,
            "status": "FAILED",
            "failed_utc": _utc_now(),
            "elapsed_seconds": time.monotonic() - started_monotonic,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        _write_json_atomic(run_dir / "failure.json", failure)
        raise
    finally:
        train_env.close()
    return completion


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_max_ticks_override(rows: Sequence[str] | None) -> dict[str, int]:
    overrides: dict[str, int] = {}
    for row in rows or ():
        level, separator, ticks = str(row).partition("=")
        if not separator or not level or not ticks:
            raise SystemExit(f"--max-ticks expects LEVEL=TICKS, got: {row!r}")
        overrides[level] = int(ticks)
    return overrides


def _cli_trainer_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "score_progress_reward_cap": args.score_progress_cap,
        "default_max_ticks": args.default_max_ticks,
        "long_horizon_max_ticks": args.long_horizon_max_ticks,
        "use_park_settle_wrapper": args.use_park_settle_wrapper,
        "use_motor_observable": args.use_motor_observable,
    }
    max_ticks = _parse_max_ticks_override(args.max_ticks)
    if max_ticks:
        overrides["max_ticks_overrides"] = max_ticks
    return overrides


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--gamma",
        type=float,
        help=f"discount factor (default {DEFAULT_GAMMA})",
    )
    parser.add_argument("--gae-lambda", type=float)
    parser.add_argument(
        "--learning-rate",
        type=float,
        help=(
            f"fresh default {DEFAULT_FRESH_LEARNING_RATE}; warm-start "
            f"default {WARM_START_LEARNING_RATE}"
        ),
    )
    parser.add_argument(
        "--entropy-coef",
        type=float,
        help=(
            f"fresh default {DEFAULT_FRESH_ENTROPY_COEF}; warm-start "
            f"default {WARM_START_ENTROPY_COEF}"
        ),
    )
    parser.add_argument(
        "--score-progress-cap",
        type=float,
        help=f"per-episode shaping cap (default {DEFAULT_SCORE_PROGRESS_REWARD_CAP})",
    )
    parser.add_argument("--default-max-ticks", type=int)
    parser.add_argument("--long-horizon-max-ticks", type=int)
    parser.add_argument(
        "--max-ticks",
        action="append",
        metavar="LEVEL=TICKS",
        help="exact per-level override; repeatable",
    )
    parser.add_argument(
        "--use-park-settle-wrapper",
        action="store_true",
        default=None,
        help=(
            "train through the park-settle macro action wrapper; the "
            "resulting model uses the MultiDiscrete([3, aim_bins]) macro "
            "interface and is not compatible with per-tick policies"
        ),
    )
    parser.add_argument(
        "--use-motor-observable",
        action="store_true",
        default=None,
        help="use the motor-observable actuator observation",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prereg_path = args.preregistration.expanduser().resolve(strict=True)
    original_root = args.original_root.expanduser().resolve(strict=True)
    prereg = _validate_preregistration_v2(prereg_path)
    _validate_catalog(prereg=prereg, original_root=original_root)
    base_config, input_config, reward_profile = _configs_v2(prereg)
    cli_overrides = _cli_trainer_overrides(args)
    run_spec: dict[str, Any] | None = None
    if args.run_id:
        matches = [run for run in prereg["runs"] if run["id"] == args.run_id]
        if len(matches) != 1:
            raise SystemExit(f"unknown or duplicate run id: {args.run_id}")
        run_spec = dict(matches[0])
        if args.learning_rate is not None:
            run_spec["learning_rate"] = float(args.learning_rate)
        if args.entropy_coef is not None:
            run_spec["entropy_coef"] = float(args.entropy_coef)
    trainer_config = _trainer_config_for_run(prereg, run_spec, cli_overrides)
    validation = _prevalidate_spaces_v2(
        prereg=prereg,
        original_root=original_root,
        base_config=base_config,
        input_config=input_config,
        reward_profile=reward_profile,
        trainer_config=trainer_config,
    )
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "VALID",
                    "preregistration_sha256": _sha256(prereg_path),
                    "trainer_sha256": _sha256(Path(__file__).resolve()),
                    **validation,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if run_spec is None:
        raise SystemExit("--run-id is required unless --validate-only is used")
    completion = _run_training_v2(
        prereg_path=prereg_path,
        prereg=prereg,
        run_spec=run_spec,
        original_root=original_root,
        trainer_config=trainer_config,
        validation=validation,
    )
    print(json.dumps(completion, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
