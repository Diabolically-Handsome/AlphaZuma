from __future__ import annotations

import numpy as np

from zuma_rl.alphazuma_55 import full55_environment_config
from zuma_rl.human_speedrun import HumanSpeedrunWrapper
from zuma_rl.observable_human_speedrun import (
    MOTOR_OBSERVATION_PROFILE,
    ObservableHumanSpeedrunWrapper,
)
from zuma_rl.revenge_env import RevengeEnv


def _base(seed: int) -> RevengeEnv:
    return RevengeEnv(
        config=full55_environment_config(max_ticks=200),
        level_id="Jungle1",
        seed=seed,
    )


def test_motor_observation_preserves_base_prefix_and_dynamics() -> None:
    ordinary = HumanSpeedrunWrapper(_base(11))
    observable = ObservableHumanSpeedrunWrapper(_base(11))
    try:
        ordinary_observation, ordinary_info = ordinary.reset(seed=17)
        observable_observation, observable_info = observable.reset(seed=17)
        width = observable.base_observation_size
        np.testing.assert_array_equal(
            observable_observation[:width], ordinary_observation
        )
        assert observable_info["motor_observation_profile"] == (
            MOTOR_OBSERVATION_PROFILE
        )
        assert "motor_observation_profile" not in ordinary_info
        assert observable_observation.shape == observable.observation_space.shape

        actions = (
            np.asarray((1, 23), dtype=np.int64),
            np.asarray((1, 25), dtype=np.int64),
            np.asarray((0, 27), dtype=np.int64),
            np.asarray((2, 29), dtype=np.int64),
        )
        for action in actions:
            left = ordinary.step(action)
            right = observable.step(action)
            np.testing.assert_array_equal(right[0][:width], left[0])
            assert right[1:4] == left[1:4]
            assert right[4]["outcome"] == left[4]["outcome"]
            assert right[4]["human_speedrun"] == left[4]["human_speedrun"]
    finally:
        ordinary.close()
        observable.close()


def test_motor_observation_exposes_hidden_delay_state() -> None:
    env = ObservableHumanSpeedrunWrapper(_base(19))
    try:
        observation, _ = env.reset(seed=23)
        motor = observation[env.base_observation_size :]
        names = env.motor_feature_names
        assert motor[names.index("motor_last_desired_wait")] == 1.0
        assert motor[names.index("motor_button_queue_0_present")] == 0.0

        observation, *_ = env.step(np.asarray((1, 41), dtype=np.int64))
        motor = observation[env.base_observation_size :]
        assert motor[names.index("motor_last_desired_fire")] == 1.0
        assert motor[names.index("motor_button_queue_0_present")] == 1.0
        assert motor[names.index("motor_button_queue_0_fire")] == 1.0
        assert motor[names.index("motor_aim_queue_0_present")] == 1.0
        assert np.all(np.isfinite(motor))
        assert np.max(motor) <= 1.0
        assert np.min(motor) >= -1.0
    finally:
        env.close()


def test_motor_features_extend_only_the_global_slice() -> None:
    env = ObservableHumanSpeedrunWrapper(_base(29))
    try:
        base = env.revenge_env
        assert env.observation_layout["balls"] == base.observation_layout["balls"]
        assert env.observation_layout["projectiles"] == (
            base.observation_layout["projectiles"]
        )
        assert env.observation_layout["globals"].start == (
            base.observation_layout["globals"].start
        )
        assert env.observation_layout["globals"].stop == (
            env.observation_space.shape[0]
        )
        assert env.global_feature_names[: base.global_feature_size] == tuple(
            base.global_feature_names
        )
        assert env.global_feature_size == (
            base.global_feature_size + len(env.motor_feature_names)
        )
        contract = env.contract()
        assert contract["base_observation_is_exact_prefix"] is True
        assert contract["actuator_dynamics_unchanged"] is True
        assert contract["motor_state_is_actor_visible"] is True
    finally:
        env.close()
