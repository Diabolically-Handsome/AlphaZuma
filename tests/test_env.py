import numpy as np
from gymnasium.utils.env_checker import check_env

from zuma_rl import ZumaConfig, ZumaEnv


def test_gymnasium_contract() -> None:
    check_env(ZumaEnv(), skip_render_check=False)


def test_seed_reproducibility() -> None:
    first = ZumaEnv()
    second = ZumaEnv()
    first_observation, _ = first.reset(seed=99)
    second_observation, _ = second.reset(seed=99)
    np.testing.assert_array_equal(first_observation, second_observation)

    actions = [1, 8, 72, 190, 15]
    for action in actions:
        first_step = first.step(action)
        second_step = second.step(action)
        np.testing.assert_array_equal(first_step[0], second_step[0])
        assert first_step[1:] == second_step[1:]


def test_different_seeds_change_the_chain() -> None:
    env = ZumaEnv()
    first, _ = env.reset(seed=1)
    second, _ = env.reset(seed=2)
    assert not np.array_equal(first, second)


def test_observation_is_in_space() -> None:
    env = ZumaEnv()
    observation, _ = env.reset(seed=3)
    assert env.observation_space.contains(observation)
    for _ in range(20):
        observation, _, terminated, truncated, _ = env.step(
            env.action_space.sample()
        )
        assert env.observation_space.contains(observation)
        if terminated or truncated:
            observation, _ = env.reset()
            assert env.observation_space.contains(observation)


def test_rgb_render() -> None:
    config = ZumaConfig(render_size=128)
    env = ZumaEnv(config=config, render_mode="rgb_array")
    env.reset(seed=4)
    frame = env.render()
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (128, 128, 3)
    assert frame.dtype == np.uint8


def test_reset_can_load_handcrafted_state() -> None:
    env = ZumaEnv()
    observation, info = env.reset(
        seed=5,
        options={
            "state": {
                "balls": [0, 1, 2],
                "front_progress": 0.25,
                "current_color": 2,
                "next_color": 1,
            }
        },
    )
    assert env.observation_space.contains(observation)
    assert info["chain_length"] == 3
    assert info["front_progress"] == 0.25

