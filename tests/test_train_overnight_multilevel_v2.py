from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from tools.train_overnight_multilevel_v2 import (
    DEFAULT_FRESH_ENTROPY_COEF,
    DEFAULT_FRESH_LEARNING_RATE,
    DEFAULT_GAMMA,
    DenseShapingHumanSpeedrunWrapper,
    TrainerV2Config,
    TruncationNeutralRewardWrapper,
    WARM_START_ENTROPY_COEF,
    WARM_START_LEARNING_RATE,
    _make_human_env_v2,
    _ppo_hyperparameters,
    _resume_custom_objects,
    _reward_config_v2,
    build_trainer_config,
)
from zuma_rl.alphazuma_55 import full55_environment_config
from zuma_rl.human_speedrun import (
    EliteHumanInputConfig,
    HumanSpeedrunWrapper,
    WinFirstRewardConfig,
)
from zuma_rl.revenge_env import RevengeEnv


def _wait(action_bins: int = 180) -> np.ndarray:
    return np.asarray((0, 0), dtype=np.int64)


def test_trainer_config_defaults_fix_the_verified_v1_pathologies() -> None:
    config = TrainerV2Config()
    assert config.gamma == 0.999
    assert config.score_progress_reward_cap == 10.0
    # Long routes get a ceiling that realistic win lengths fit under.
    assert config.max_ticks_for_level("coast3") == 24_000
    assert config.max_ticks_for_level("Coast3") == 24_000
    assert config.max_ticks_for_level("Grotto2") == 24_000
    assert config.max_ticks_for_level("volcano1") == 24_000
    assert config.max_ticks_for_level("jungle2") == 12_000
    assert config.max_ticks_for_level("boss1") == 12_000


def test_max_ticks_overrides_win_over_prefix_rule() -> None:
    config = build_trainer_config(
        None,
        {"max_ticks_overrides": {"Coast3": 30_000, "jungle2": 18_000}},
    )
    assert config.max_ticks_for_level("coast3") == 30_000
    assert config.max_ticks_for_level("coast4") == 24_000
    assert config.max_ticks_for_level("Jungle2") == 18_000


def test_build_trainer_config_precedence() -> None:
    assert build_trainer_config().gamma == DEFAULT_GAMMA
    assert build_trainer_config({"gamma": 0.997}).gamma == 0.997
    merged = build_trainer_config({"gamma": 0.997}, {"gamma": 0.9995})
    assert merged.gamma == 0.9995
    # None entries (absent CLI flags) never mask preregistered values.
    masked = build_trainer_config({"gamma": 0.997}, {"gamma": None})
    assert masked.gamma == 0.997
    with pytest.raises(ValueError):
        build_trainer_config({"gamma": 1.5})


def test_ppo_hyperparameters_defaults_and_explicit_values() -> None:
    run_spec = {"rollout_steps": 64, "batch_size": 32, "ppo_epochs": 4}
    config = TrainerV2Config()
    fresh = _ppo_hyperparameters(config, run_spec, resuming=False)
    assert fresh["gamma"] == 0.999
    assert fresh["learning_rate"] == DEFAULT_FRESH_LEARNING_RATE
    assert fresh["ent_coef"] == DEFAULT_FRESH_ENTROPY_COEF
    warm = _ppo_hyperparameters(config, run_spec, resuming=True)
    assert warm["learning_rate"] == WARM_START_LEARNING_RATE
    assert warm["ent_coef"] == WARM_START_ENTROPY_COEF
    explicit = _ppo_hyperparameters(
        config,
        {**run_spec, "learning_rate": 1e-4, "entropy_coef": 0.005},
        resuming=True,
    )
    assert explicit["learning_rate"] == 1e-4
    assert explicit["ent_coef"] == 0.005


def test_resume_does_not_force_a_hardcoded_gamma() -> None:
    config = build_trainer_config(None, {"gamma": 0.9995})
    run_spec = {"rollout_steps": 64, "batch_size": 32, "ppo_epochs": 4}
    hyperparameters = _ppo_hyperparameters(config, run_spec, resuming=True)
    custom = _resume_custom_objects(hyperparameters)
    # The user's gamma is what gets applied to the loaded checkpoint; the
    # v1 behavior of enforcing 0.995 is gone.
    assert custom["gamma"] == 0.9995
    assert custom["gamma"] != 0.995
    assert custom["ent_coef"] == WARM_START_ENTROPY_COEF
    assert set(custom) == {
        "learning_rate",
        "n_steps",
        "batch_size",
        "n_epochs",
        "gamma",
        "gae_lambda",
        "ent_coef",
    }


def test_dense_shaping_cap_requires_the_v2_wrapper_subclass() -> None:
    dense_reward = WinFirstRewardConfig(
        profile_id="dense-test",
        score_progress_reward_cap=10.0,
    )
    base = RevengeEnv(
        config=full55_environment_config(max_ticks=100),
        level_id="Jungle1",
        seed=3,
    )
    with pytest.raises(ValueError):
        HumanSpeedrunWrapper(base, reward_config=dense_reward)
    base.close()

    base = RevengeEnv(
        config=full55_environment_config(max_ticks=100),
        level_id="Jungle1",
        seed=3,
    )
    env = DenseShapingHumanSpeedrunWrapper(base, reward_config=dense_reward)
    assert env.reward_config.score_progress_reward_cap == 10.0
    env.close()


def test_truncation_pays_no_terminal_penalty_on_the_real_stack() -> None:
    trainer_config = build_trainer_config(
        None,
        {"max_ticks_overrides": {"jungle1": 25}},
    )
    env = _make_human_env_v2(
        level_id="Jungle1",
        base_config=full55_environment_config(),
        input_config=EliteHumanInputConfig(),
        trainer_config=trainer_config,
    )
    try:
        env.reset(seed=5)
        terminated = truncated = False
        reward = 0.0
        steps = 0
        while not (terminated or truncated):
            _, reward, terminated, truncated, info = env.step(_wait())
            steps += 1
            assert steps <= 25
        assert truncated is True
        assert terminated is False
        assert steps == 25
        assert info["TimeLimit.truncated"] is True
        # Only the per-tick time term remains; the -10 failure payment on
        # time-limit truncation is removed so SB3 2.9 bootstraps instead.
        assert info["reward_terms"]["outcome"] == 0.0
        assert info["truncation_terminal_reward_removed"] is True
        assert reward > -1.0
        assert reward == pytest.approx(-0.0001, abs=1e-6)
    finally:
        env.close()


def test_v1_semantics_paid_minus_ten_on_the_same_truncation() -> None:
    base = RevengeEnv(
        config=full55_environment_config(max_ticks=25),
        level_id="Jungle1",
        seed=5,
    )
    env = HumanSpeedrunWrapper(base)
    try:
        env.reset(seed=5)
        terminated = truncated = False
        reward = 0.0
        while not (terminated or truncated):
            _, reward, terminated, truncated, _ = env.step(_wait())
        assert truncated is True
        assert reward < -9.0
    finally:
        env.close()


class _StubTerminalEnv(gym.Env[Any, Any]):
    """Minimal env emitting one preset terminal transition."""

    observation_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
    action_space = spaces.MultiDiscrete(np.array((4, 180), dtype=np.int64))

    def __init__(
        self,
        *,
        reward: float,
        terminated: bool,
        truncated: bool,
        outcome: str | None,
    ) -> None:
        # Explicit gym.Env calls: the diamond with HumanSpeedrunWrapper in
        # _StubHumanTerminalEnv must never reach the wrapper's __init__.
        gym.Env.__init__(self)
        self.reward_config = WinFirstRewardConfig()
        self._result = (reward, terminated, truncated, outcome)

    def reset(self, *, seed=None, options=None):
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        reward, terminated, truncated, outcome = self._result
        info: dict[str, Any] = {
            "outcome": outcome,
            "reward_terms": {"outcome": -10.0 if outcome != "win" else 10.0},
        }
        if truncated:
            info["TimeLimit.truncated"] = True
        return (
            np.zeros(4, dtype=np.float32),
            reward,
            terminated,
            truncated,
            info,
        )


class _StubHumanTerminalEnv(_StubTerminalEnv, HumanSpeedrunWrapper):
    """Stand-in typed as a HumanSpeedrunWrapper for resolver purposes.

    Only the interface exercised by TruncationNeutralRewardWrapper is
    provided; HumanSpeedrunWrapper.__init__ is deliberately not called.
    """

    def __init__(self, **kwargs: Any) -> None:
        _StubTerminalEnv.__init__(self, **kwargs)


def _wrapped_stub(**kwargs: Any) -> TruncationNeutralRewardWrapper:
    return TruncationNeutralRewardWrapper(_StubHumanTerminalEnv(**kwargs))


def test_wrapper_death_keeps_minus_ten_with_terminated() -> None:
    env = _wrapped_stub(
        reward=-10.01,
        terminated=True,
        truncated=False,
        outcome="loss",
    )
    env.reset(seed=1)
    _, reward, terminated, truncated, info = env.step((0, 0))
    assert terminated is True
    assert truncated is False
    assert reward == pytest.approx(-10.01)
    assert info["reward_terms"]["outcome"] == -10.0


def test_wrapper_truncation_removes_only_the_terminal_payment() -> None:
    env = _wrapped_stub(
        reward=-10.01,
        terminated=False,
        truncated=True,
        outcome=None,
    )
    env.reset(seed=1)
    _, reward, terminated, truncated, info = env.step((0, 0))
    assert truncated is True
    assert terminated is False
    assert reward == pytest.approx(-0.01)
    assert info["reward_terms"]["outcome"] == 0.0
    assert info["truncation_terminal_reward_removed"] is True


def test_wrapper_win_payment_is_untouched() -> None:
    env = _wrapped_stub(
        reward=9.99,
        terminated=True,
        truncated=False,
        outcome="win",
    )
    env.reset(seed=1)
    _, reward, terminated, truncated, info = env.step((0, 0))
    assert terminated is True
    assert reward == pytest.approx(9.99)
    assert info["reward_terms"]["outcome"] == 10.0


def test_max_ticks_plumbs_into_the_level_environment() -> None:
    trainer_config = build_trainer_config(
        None,
        {"max_ticks_overrides": {"jungle1": 33}},
    )
    env = _make_human_env_v2(
        level_id="Jungle1",
        base_config=full55_environment_config(),
        input_config=EliteHumanInputConfig(),
        trainer_config=trainer_config,
    )
    try:
        assert env.unwrapped.config.max_ticks == 33
    finally:
        env.close()


def test_park_settle_flag_changes_the_action_interface() -> None:
    base_kwargs = {
        "level_id": "Jungle1",
        "base_config": full55_environment_config(),
        "input_config": EliteHumanInputConfig(),
    }
    per_tick = _make_human_env_v2(
        trainer_config=build_trainer_config(),
        **base_kwargs,
    )
    try:
        assert per_tick.action_space.nvec.tolist() == [4, 180]
    finally:
        per_tick.close()
    macro = _make_human_env_v2(
        trainer_config=build_trainer_config(
            None,
            {"use_park_settle_wrapper": True},
        ),
        **base_kwargs,
    )
    try:
        assert macro.action_space.nvec.tolist() == [3, 180]
        macro.reset(seed=9)
        assert macro.action_masks().shape == (3 + 180,)
    finally:
        macro.close()


def test_reward_config_v2_cap_follows_the_trainer_config() -> None:
    config = build_trainer_config(None, {"score_progress_reward_cap": 2.5})
    reward = _reward_config_v2(config)
    assert reward.score_progress_reward_cap == 2.5
    assert reward.win_reward == 10.0
    assert reward.failure_reward == -10.0
    profile = _reward_config_v2(
        config,
        {"win_reward": 12.0, "score_progress_reward_cap": 99.0},
    )
    # The preregistered profile may tune terminal terms, but the shaping
    # cap is owned by the trainer configuration.
    assert profile.win_reward == 12.0
    assert profile.score_progress_reward_cap == 2.5
