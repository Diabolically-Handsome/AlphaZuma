"""Gymnasium-contract tests for the fidelity-first Revenge adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from zuma_rl.original_data import CurveParameters
from zuma_rl.human_speedrun import (
    EliteHumanInputConfig,
    HumanSpeedrunWrapper,
    WinFirstRewardConfig,
)
from zuma_rl.revenge_core import (
    ChainBall,
    FruitSpawnCalibration,
    PowerupType,
    RevengeSimulator,
)
from zuma_rl.revenge_env import RevengeEnv, RevengeEnvConfig
from zuma_rl.revenge_teacher import ActorObservableRevengeTeacher


@dataclass(slots=True)
class SyntheticCurve:
    points: np.ndarray
    in_tunnel: np.ndarray
    parameters: CurveParameters
    die_at_end: bool = True

    @property
    def end_waypoint(self) -> int:
        return len(self.points) - 1

    def point_at_waypoint(
        self,
        waypoint: float | Iterable[float],
        *,
        loop_at_end: bool = False,
    ) -> np.ndarray:
        values = np.asarray(waypoint, dtype=np.float64)
        if loop_at_end:
            values = np.mod(values, len(self.points))
        values = np.clip(values, 0.0, float(self.end_waypoint))
        lower = np.floor(values).astype(np.int64)
        upper = np.minimum(lower + 1, self.end_waypoint)
        fraction = values - lower
        return self.points[lower] + fraction[..., np.newaxis] * (
            self.points[upper] - self.points[lower]
        )

    def perpendicular_at_waypoint(
        self,
        waypoint: float | Iterable[float],
    ) -> np.ndarray:
        values = np.asarray(waypoint)
        return np.broadcast_to(
            np.array((0.0, -1.0), dtype=np.float64),
            values.shape + (2,),
        ).copy()

    def is_in_tunnel_at_waypoint(
        self,
        waypoint: float | Iterable[float],
    ) -> np.ndarray | np.bool_:
        values = np.asarray(waypoint, dtype=np.float64)
        indices = np.trunc(values).astype(np.int64)
        valid = (indices >= 0) & (indices <= self.end_waypoint)
        clipped = np.clip(indices, 0, self.end_waypoint)
        result = np.where(valid, self.in_tunnel[clipped], indices < 0)
        return np.asarray(result, dtype=np.bool_)


def _parameters(
    *,
    speed: float = 0.0,
    zuma_score: int = 1_000,
    colors: int = 4,
) -> CurveParameters:
    return CurveParameters(
        start_distance_percent=65,
        num_balls=0,
        ball_repeat_chance=45,
        max_single=2,
        colors=colors,
        speed=speed,
        slow_distance=200,
        acceleration_rate=0.0,
        max_speed=100.0,
        zuma_score=zuma_score,
        skull_rotation_degrees=75,
        zuma_back_distance=300,
        zuma_slow_duration=1_100,
        slow_factor=4.0,
        max_clump_size=6,
        powerup_records=(),
        powerup_chance=600,
    )


def _curve(
    *,
    length: int = 1_000,
    tunnel_waypoints: Iterable[int] = (),
    colors: int = 4,
) -> SyntheticCurve:
    x = np.linspace(0.0, 799.0, length + 1)
    points = np.column_stack((x, np.full_like(x, 180.0)))
    tunnels = np.zeros(length + 1, dtype=np.bool_)
    for waypoint in tunnel_waypoints:
        tunnels[waypoint] = True
    return SyntheticCurve(points, tunnels, _parameters(colors=colors))


def _env(
    *,
    config: RevengeEnvConfig | None = None,
    curve: SyntheticCurve | None = None,
    shooter_positions: tuple[tuple[float, float], ...] | None = None,
    seed: int = 7,
    render_mode: str | None = None,
) -> RevengeEnv:
    simulator = RevengeSimulator(
        curve or _curve(),
        shooter=(400.0, 300.0),
        shooter_positions=shooter_positions,
        seed=seed,
    )
    return RevengeEnv(
        config=config,
        simulator=simulator,
        render_mode=render_mode,
    )


def test_config_and_action_contract() -> None:
    config = RevengeEnvConfig(aim_bins=180)
    assert config.frame_skip == 1
    env = _env(config=config)
    assert isinstance(env.action_space, gym.spaces.MultiDiscrete)
    assert env.action_space.nvec.tolist() == [3, 180]

    verb, aim_bin, angle = env.decode_action(
        np.asarray((2, 17), dtype=np.int64)
    )
    assert (verb, aim_bin) == (2, 17)
    assert angle == pytest.approx(2.0 * np.pi * 17.5 / config.aim_bins)
    np.testing.assert_array_equal(
        env.encode_action(1, 42),
        np.asarray((1, 42), dtype=np.int64),
    )
    with pytest.raises(ValueError):
        env.decode_action(np.asarray((3, 0), dtype=np.int64))

    flat = _env(
        config=RevengeEnvConfig(
            aim_bins=180,
            action_mode="flat",
        )
    )
    assert isinstance(flat.action_space, gym.spaces.Discrete)
    assert flat.action_space.n == 540
    assert flat.decode_action(2 * config.aim_bins + 17)[:2] == (2, 17)
    assert flat.encode_action(1, 42) == config.aim_bins + 42


def test_full55_interface_fixes_six_colors_and_adds_masked_hop() -> None:
    config = RevengeEnvConfig(actor_interface="full55-v1", aim_bins=180)
    env = _env(config=config)
    observation, info = env.reset(seed=9)

    assert env.action_space.nvec.tolist() == [4, 180]
    assert env.num_colors == 4
    assert env.color_feature_count == 6
    assert tuple(
        name for name in env.ball_feature_names if name.startswith("color_")
    ) == tuple(f"color_{index}" for index in range(6))
    assert env.action_masks().shape == (4 + 180,)
    assert not bool(env.action_masks()[3])
    assert info["actor_interface"] == "full55-v1"
    assert info["active_num_colors"] == 4
    assert info["color_feature_count"] == 6
    assert env.observation_space.contains(observation)


def test_full55_observation_and_action_spaces_match_four_to_six_colors() -> None:
    config = RevengeEnvConfig(actor_interface="full55-v1", max_balls=160)
    environments = [
        _env(config=config, curve=_curve(colors=colors))
        for colors in (4, 5, 6)
    ]
    try:
        reference = environments[0]
        for colors, env in zip((4, 5, 6), environments, strict=True):
            observation, info = env.reset(seed=11)
            assert env.observation_space == reference.observation_space
            assert env.action_space == reference.action_space
            assert info["active_num_colors"] == colors
            assert env.observation_space.contains(observation)
    finally:
        for env in environments:
            env.close()


def test_full55_dual_position_hop_updates_actor_visible_shooter() -> None:
    positions = ((175.0, 310.0), (610.0, 310.0))
    env = _env(
        config=RevengeEnvConfig(
            actor_interface="full55-v1",
            aim_bins=180,
            frame_skip=1,
        ),
        shooter_positions=positions,
    )
    before, before_info = env.reset(seed=13)
    global_slice = env.observation_layout["globals"]
    before_globals = before[global_slice]
    global_index = {
        name: index for index, name in enumerate(env.global_feature_names)
    }

    assert before_info["frog_position_count"] == 2
    assert before_info["active_frog_position"] == 0
    assert bool(env.valid_verb_mask()[3])
    assert before_globals[global_index["frog_position_0"]] == 1.0
    assert before_globals[global_index["frog_position_1"]] == 0.0

    after, _, _, _, after_info = env.step(env.encode_action(3, 0))
    after_globals = after[global_slice]
    assert after_info["verb"] == "hop"
    assert after_info["action_accepted"] is True
    assert after_info["active_frog_position"] == 0
    assert after_info["hop_in_progress"] is True
    assert after_info["hop_ticks_remaining"] == 19
    assert after_info["hop_target_frog_position"] == 1
    assert after_globals[global_index["frog_position_0"]] == 1.0
    assert after_globals[global_index["frog_position_1"]] == 0.0
    assert sum(after_globals[global_index[name]] for name in (
        "gun_normal",
        "gun_firing",
        "gun_reloading",
    )) == 0.0
    assert env.valid_verb_mask().tolist() == [True, False, False, False]

    for _ in range(18):
        after, _, _, _, after_info = env.step(env.encode_action(0, 0))
    assert after_info["active_frog_position"] == 0
    assert after_info["hop_ticks_remaining"] == 1

    after, _, _, _, after_info = env.step(env.encode_action(0, 0))
    after_globals = after[global_slice]
    assert after_info["active_frog_position"] == 1
    assert after_info["hop_in_progress"] is False
    assert after_info["hop_ticks_remaining"] == 0
    assert after_info["hop_target_frog_position"] is None
    assert after_globals[global_index["frog_position_0"]] == 0.0
    assert after_globals[global_index["frog_position_1"]] == 1.0
    assert after_globals[global_index["shooter_x"]] == pytest.approx(
        2.0 * positions[1][0] / env.sim.config.logical_width - 1.0
    )


def test_frame_skip_sets_aim_and_reports_action() -> None:
    config = RevengeEnvConfig(aim_bins=36, frame_skip=7)
    env = _env(config=config)
    env.reset(seed=10)
    action = env.encode_action(0, 8)
    _, reward, terminated, truncated, info = env.step(action)

    assert env.sim.tick_count == 7
    assert env.sim.aim_angle == pytest.approx(np.pi * 2.0 * 8.5 / 36)
    assert isinstance(env.sim.aim_angle, np.float32)
    assert info["profile_mode"] == "tutorials_completed"
    assert info["ticks_advanced"] == 7
    assert info["verb"] == "wait"
    assert reward == pytest.approx(config.step_penalty)
    assert not terminated
    assert not truncated


def test_fire_and_swap_verbs_are_dispatched() -> None:
    env = _env(config=RevengeEnvConfig(frame_skip=1))
    env.reset(
        seed=4,
        options={
            "state": {
                "colors": [0, 1],
                "waypoints": [100.0, 136.0],
                "current_color": 0,
                "next_color": 1,
            }
        },
    )
    _, _, _, _, swap_info = env.step(env.encode_action(2, 0))
    assert swap_info["action_accepted"] is True
    assert (env.sim.current_color, env.sim.next_color) == (1, 0)

    # Reloading is an original simulator state; make the isolated dispatch
    # assertion independent from its animation by starting a fresh episode.
    env.reset(
        seed=4,
        options={
            "state": {
                "colors": [0, 1],
                "waypoints": [100.0, 136.0],
                "current_color": 0,
                "next_color": 1,
            }
        },
    )
    _, _, _, _, fire_info = env.step(env.encode_action(1, 9))
    assert fire_info["action_accepted"] is True
    assert env.sim.aim_angle == pytest.approx(2.0 * np.pi * 9.5 / 180)


def test_actor_observable_teacher_fires_current_match_and_swaps_for_next() -> None:
    env = _env(config=RevengeEnvConfig(aim_bins=180, max_balls=160))
    teacher = ActorObservableRevengeTeacher.from_env(env)
    current_observation, _ = env.reset(
        seed=4,
        options={
            "state": {
                "colors": [0, 0, 1],
                "waypoints": [400.0, 436.0, 472.0],
                "current_color": 0,
                "next_color": 1,
            }
        },
    )
    current_action = teacher.act(current_observation)
    assert env.action_space.contains(current_action)
    assert int(current_action[0]) == 1

    next_observation, _ = env.reset(
        seed=4,
        options={
            "state": {
                "colors": [0, 0, 1],
                "waypoints": [400.0, 436.0, 472.0],
                "current_color": 1,
                "next_color": 0,
            }
        },
    )
    next_action = teacher.act(next_observation)
    assert env.action_space.contains(next_action)
    assert int(next_action[0]) == 2

    # The teacher captures only immutable interface/level constants.  Dynamic
    # decisions remain a pure function of the supplied observation.
    env.sim.current_color = 3
    np.testing.assert_array_equal(teacher.act(next_observation), next_action)


def test_entity_feature_extractor_is_finite_with_and_without_entities() -> None:
    torch = pytest.importorskip("torch")
    from zuma_rl.revenge_features import revenge_entity_policy_kwargs

    env = _env(config=RevengeEnvConfig(aim_bins=180, max_balls=160))
    kwargs = revenge_entity_policy_kwargs(env)
    extractor_class = kwargs["features_extractor_class"]
    extractor = extractor_class(
        env.observation_space,
        **kwargs["features_extractor_kwargs"],
    )
    observation, _ = env.reset(seed=11)
    batch = torch.as_tensor(
        np.stack((observation, np.zeros_like(observation))),
        dtype=torch.float32,
    )
    features = extractor(batch)

    assert features.shape == (2, 256)
    assert torch.isfinite(features).all()
    features.sum().backward()
    assert any(
        parameter.grad is not None
        for parameter in extractor.parameters()
    )


def test_relational_feature_extractor_exposes_color_run_relations() -> None:
    torch = pytest.importorskip("torch")
    from zuma_rl.revenge_features import revenge_relational_policy_kwargs

    env = _env(config=RevengeEnvConfig(aim_bins=180, max_balls=160))
    observation, _ = env.reset(
        seed=4,
        options={
            "state": {
                "colors": [0, 0, 1],
                "waypoints": [400.0, 436.0, 472.0],
                "current_color": 1,
                "next_color": 0,
            }
        },
    )
    kwargs = revenge_relational_policy_kwargs(env)
    extractor_kwargs = kwargs["features_extractor_kwargs"]
    extractor = kwargs["features_extractor_class"](
        env.observation_space,
        **extractor_kwargs,
    )
    batch = torch.as_tensor(observation[None, :], dtype=torch.float32)
    balls = batch[
        :,
        extractor.balls_start : extractor.balls_stop,
    ].reshape(1, extractor.max_balls, extractor.ball_feature_size)
    globals_ = batch[:, extractor.globals_start : extractor.globals_stop]

    augmented, summary, current_match, next_match = (
        extractor._ball_relations(balls, globals_)
    )

    assert augmented.shape[-1] == extractor.ball_feature_size + 6
    assert current_match[0, :3].tolist() == [False, False, True]
    assert next_match[0, :3].tolist() == [True, True, False]
    np.testing.assert_allclose(
        summary.detach().numpy()[0],
        [1.0, 1.0, 0.0, 1.0, 1.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0],
        atol=1e-6,
    )
    features = extractor(batch)
    assert features.shape == (1, 256)
    assert torch.isfinite(features).all()


def test_polar_feature_extractor_maps_visible_target_to_aim_bins() -> None:
    torch = pytest.importorskip("torch")
    from zuma_rl.revenge_features import revenge_polar_policy_kwargs

    env = _env(config=RevengeEnvConfig(aim_bins=180, max_balls=160))
    observation, _ = env.reset(
        seed=4,
        options={
            "state": {
                "colors": [1],
                "waypoints": [400.0],
                "current_color": 1,
                "next_color": 0,
            }
        },
    )
    teacher = ActorObservableRevengeTeacher.from_env(env)
    teacher_action = teacher.act(observation)
    assert int(teacher_action[0]) == 1
    kwargs = revenge_polar_policy_kwargs(env)
    extractor = kwargs["features_extractor_class"](
        env.observation_space,
        **kwargs["features_extractor_kwargs"],
    )
    features = extractor(
        torch.as_tensor(observation[None, :], dtype=torch.float32)
    )

    assert features.shape == (1, 256 + 180)
    polar_bin = int(torch.argmax(features[0, 256:]).item())
    distance = abs(polar_bin - int(teacher_action[1]))
    distance = min(distance, 180 - distance)
    assert distance <= 1


def test_factorized_action_mask_matches_retail_request_acceptance() -> None:
    env = _env(config=RevengeEnvConfig(aim_bins=36, frame_skip=1))
    env.reset(
        seed=4,
        options={
            "state": {
                "colors": [0, 1],
                "waypoints": [100.0, 136.0],
                "current_color": 0,
                "next_color": 1,
            }
        },
    )

    mask = env.action_masks()
    assert mask.dtype == np.bool_
    assert mask.shape == (3 + 36,)
    np.testing.assert_array_equal(mask[:3], (True, True, True))
    assert mask[3:].all()

    env.step(env.encode_action(1, 9))
    np.testing.assert_array_equal(
        env.valid_verb_mask(),
        (True, False, False),
    )
    assert env.action_masks()[3:].all()

    env.reset(
        seed=4,
        options={
            "state": {
                "colors": [0, 1],
                "waypoints": [100.0, 136.0],
                "current_color": 0,
                "next_color": 0,
            }
        },
    )
    np.testing.assert_array_equal(
        env.valid_verb_mask(),
        (True, True, False),
    )


def test_flat_action_mask_repeats_each_verb_block() -> None:
    env = _env(
        config=RevengeEnvConfig(
            aim_bins=36,
            action_mode="flat",
            frame_skip=1,
        )
    )
    env.reset(
        seed=4,
        options={
            "state": {
                "colors": [0, 1],
                "waypoints": [100.0, 136.0],
                "current_color": 0,
                "next_color": 0,
            }
        },
    )
    mask = env.action_masks().reshape(3, 36)
    assert mask[0].all()
    assert mask[1].all()
    assert not mask[2].any()


def test_actor_view_hides_tunnel_balls_and_compacts_slots() -> None:
    curve = _curve(tunnel_waypoints=(100,))
    env = _env(curve=curve)
    env.reset(
        seed=2,
        options={
            "state": {
                "colors": [0, 1, 2],
                "waypoints": [64.0, 100.0, 136.0],
                "contacts": [True, True],
            }
        },
    )
    observation = env._observation()
    ball_values = observation[env.observation_layout["balls"]].reshape(
        env.config.max_balls,
        env.ball_feature_size,
    )

    assert ball_values[0, 0] == 1.0
    assert ball_values[1, 0] == 1.0
    assert ball_values[2, 0] == 0.0
    color_offset = len(env._BALL_BASE_FEATURES)
    assert ball_values[0, color_offset + 0] == 1.0
    assert ball_values[1, color_offset + 2] == 1.0
    # Packing visible balls must not turn two balls separated by a hidden
    # tunnel ball into apparent direct neighbours.
    assert ball_values[:2, 4].tolist() == [0.0, 0.0]
    assert ball_values[:2, 6].tolist() == [0.0, 0.0]

    privileged = _env(
        curve=curve,
        config=RevengeEnvConfig(privileged_debug=True),
    )
    privileged.reset(
        seed=2,
        options={
            "state": {
                "colors": [0, 1, 2],
                "waypoints": [64.0, 100.0, 136.0],
            }
        },
    )
    privileged_values = privileged._observation()[
        privileged.observation_layout["balls"]
    ].reshape(privileged.config.max_balls, privileged.ball_feature_size)
    assert privileged_values[:3, 0].tolist() == [1.0, 1.0, 1.0]
    assert privileged_values[:3, 4].tolist() == [1.0, 1.0, 0.0]
    assert privileged_values[1, 6] == 1.0


def test_multi_curve_observation_preserves_curve_identity() -> None:
    curves = (
        _curve(),
        _curve(),
    )
    simulator = RevengeSimulator(
        curves[0],
        curves=curves,
        shooter=(400.0, 300.0),
        seed=19,
    )
    for curve_index, color in enumerate((1, 2)):
        simulator._activate_curve(curve_index)
        simulator.balls = [
            ChainBall(
                id=simulator._new_id(),
                color=color,
                waypoint=np.float32(100.0 + 50.0 * curve_index),
                radius=simulator.config.ball_radius,
            )
        ]
        simulator.pending_colors.clear()
        simulator.stop_adding = True
    simulator._activate_curve(0)
    env = RevengeEnv(simulator=simulator)

    observation = env._observation()
    balls = observation[env.observation_layout["balls"]].reshape(
        env.config.max_balls,
        env.ball_feature_size,
    )
    curve_offset = (
        len(env._BALL_BASE_FEATURES)
        + env.num_colors
        + int(PowerupType.NONE)
    )

    assert balls[:2, 0].tolist() == [1.0, 1.0]
    assert balls[0, curve_offset : curve_offset + 2].tolist() == [1.0, 0.0]
    assert balls[1, curve_offset : curve_offset + 2].tolist() == [0.0, 1.0]
    assert env._info()["chain_lengths"] == (1, 1)

    simulator._activate_curve(1)
    projectile = simulator.launch_projectile(0.0, color=3)
    simulator.attach_projectile(projectile, hit_index=0, hit_in_front=True)
    simulator._activate_curve(0)
    projectiles = env._projectile_observation().reshape(
        env.config.max_projectiles,
        env.projectile_feature_size,
    )
    projectile_curve_offset = (
        len(env._PROJECTILE_BASE_FEATURES) + env.num_colors
    )
    assert projectiles[0, 6] == 1.0
    assert projectiles[
        0,
        projectile_curve_offset : projectile_curve_offset + 2,
    ].tolist() == [0.0, 1.0]


def test_actor_observation_exposes_visible_powerup_type() -> None:
    env = _env()
    env.reset(
        seed=9,
        options={
            "state": {
                "colors": [2],
                "waypoints": [80.0],
                "pending_colors": [],
                "current_color": 0,
                "next_color": 1,
            }
        },
    )
    env.sim.balls[0].powerup_primary_type = int(PowerupType.REVERSE)

    balls = env._observation()[
        env.observation_layout["balls"]
    ].reshape(env.config.max_balls, env.ball_feature_size)
    powerup_offset = len(env._BALL_BASE_FEATURES) + env.num_colors

    assert balls[0, powerup_offset + int(PowerupType.REVERSE)] == 1.0
    assert np.sum(
        balls[
            0,
            powerup_offset : powerup_offset + int(PowerupType.NONE),
        ]
    ) == 1.0


def test_actor_observation_exposes_fruit_presence_position_and_state() -> None:
    calibration = FruitSpawnCalibration(
        frequency_ticks=1_000,
        lifetime_ticks=1_000,
        points=((28.0, 105.0),),
        unlock_percentages=((1,),),
        provenance="unit-test retail fruit proof",
        fruit_type="pineapple",
    )
    simulator = RevengeSimulator(
        _curve(),
        shooter=(400.0, 300.0),
        seed=7,
        fruit_calibration=calibration,
    )
    env = RevengeEnv(simulator=simulator)
    simulator.fruit_active_point_index = 0
    simulator.fruit_vertical_offset = np.float32(2.0)

    globals_ = env._observation()[env.observation_layout["globals"]]
    indices = {
        name: env.global_feature_names.index(name)
        for name in (
            "fruit_present",
            "fruit_collectable",
            "fruit_x",
            "fruit_y",
            "fruit_collecting",
        )
    }
    assert globals_[indices["fruit_present"]] == 1.0
    assert globals_[indices["fruit_collectable"]] == 1.0
    assert globals_[indices["fruit_x"]] == pytest.approx(2 * 54 / 800 - 1)
    assert globals_[indices["fruit_y"]] == pytest.approx(2 * 133 / 600 - 1)
    assert globals_[indices["fruit_collecting"]] == 0.0

    simulator.fruit_collecting = True
    globals_ = env._observation()[env.observation_layout["globals"]]
    assert globals_[indices["fruit_collectable"]] == 0.0
    assert globals_[indices["fruit_collecting"]] == 1.0


def test_observation_capacity_fails_closed_instead_of_truncating() -> None:
    balls_env = _env(config=RevengeEnvConfig(max_balls=160))
    balls_env.sim.load_state(
        colors=[index % 4 for index in range(161)],
        waypoints=[float(index * 4) for index in range(161)],
    )
    with pytest.raises(RuntimeError, match="161 visible balls"):
        balls_env._observation()

    shots_env = _env(config=RevengeEnvConfig(max_projectiles=1))
    shots_env.sim.launch_projectile(0.0, color=0)
    shots_env.sim.launch_projectile(0.0, color=1)
    with pytest.raises(RuntimeError, match="2 projectiles"):
        shots_env._observation()


def test_large_actor_capacity_retains_all_490_visible_balls() -> None:
    env = _env(config=RevengeEnvConfig(max_balls=768))
    env.sim.load_state(
        colors=[index % 4 for index in range(490)],
        waypoints=[float(index * 2) for index in range(490)],
    )

    observation = env._observation()
    balls = observation[env.observation_layout["balls"]].reshape(
        env.config.max_balls,
        env.ball_feature_size,
    )

    assert int(np.sum(balls[:, 0])) == 490
    assert np.all(balls[:490, 0] == 1.0)
    assert np.all(balls[490:, 0] == 0.0)


def test_observation_contains_free_and_merging_projectiles() -> None:
    env = _env()
    env.reset(
        seed=2,
        options={
            "state": {
                "colors": [0, 1],
                "waypoints": [100.0, 136.0],
                "current_color": 0,
                "next_color": 1,
            }
        },
    )
    first = env.sim.launch_projectile(0.0, color=2)
    second = env.sim.launch_projectile(0.0, color=3)
    env.sim.attach_projectile(second, hit_index=0, hit_in_front=True)
    assert first in env.sim.free_projectiles
    assert second in env.sim.merging_projectiles

    observation = env._observation()
    projectiles = observation[
        env.observation_layout["projectiles"]
    ].reshape(env.config.max_projectiles, env.projectile_feature_size)
    assert projectiles[:2, 0].tolist() == [1.0, 1.0]
    assert projectiles[:2, 6].tolist() == [0.0, 1.0]
    assert env.observation_space.contains(observation)


def test_reset_seed_replays_identically() -> None:
    env = _env(config=RevengeEnvConfig(frame_skip=3))
    actions = [
        env.encode_action(0, 1),
        env.encode_action(0, 42),
        env.encode_action(1, 90),
        env.encode_action(0, 120),
    ]

    initial_a, _ = env.reset(seed=12345)
    trajectory_a = [initial_a.copy()]
    for action in actions:
        trajectory_a.append(env.step(action)[0].copy())

    initial_b, _ = env.reset(seed=12345)
    trajectory_b = [initial_b.copy()]
    for action in actions:
        trajectory_b.append(env.step(action)[0].copy())

    for left, right in zip(trajectory_a, trajectory_b, strict=True):
        np.testing.assert_array_equal(left, right)


def test_max_ticks_truncates_without_overshoot() -> None:
    config = RevengeEnvConfig(frame_skip=5, max_ticks=7)
    env = _env(config=config)
    env.reset(seed=8)
    assert env.step(env.encode_action(0, 0))[3] is False
    _, _, terminated, truncated, info = env.step(env.encode_action(0, 0))
    assert not terminated
    assert truncated
    assert env.sim.tick_count == 7
    assert info["ticks_advanced"] == 2
    assert info["TimeLimit.truncated"] is True


def test_env_terminates_at_irreversible_loss_without_forced_noop_tail() -> None:
    config = RevengeEnvConfig(frame_skip=8)
    env = _env(config=config)
    env.reset(
        seed=5,
        options={
            "state": {
                "colors": [0, 1],
                "waypoints": [30.0, 1_000.0],
                "contacts": [False],
                "pending_colors": [2],
                "current_color": 0,
                "next_color": 1,
            }
        },
    )

    observation, reward, terminated, truncated, info = env.step(
        env.encode_action(0, 0)
    )

    assert terminated
    assert not truncated
    assert env.sim.tick_count == 2
    assert env.sim.loss_started
    assert env.sim.outcome is None
    assert info["ticks_advanced"] == 2
    assert info["outcome"] == "loss"
    assert info["native_outcome"] is None
    assert info["loss_started"] is True
    assert info["loss_elapsed_ticks"] == 1
    assert reward == pytest.approx(config.step_penalty + config.loss_reward)
    global_values = observation[env.observation_layout["globals"]]
    assert global_values[15] == -1.0
    with pytest.raises(RuntimeError, match="after episode end"):
        env.step(env.encode_action(0, 0))


def test_env_terminates_at_input_locked_victory_transition() -> None:
    config = RevengeEnvConfig(frame_skip=8)
    env = _env(config=config)
    env.reset(
        seed=13,
        options={
            "state": {
                "colors": [0],
                "waypoints": [100.0],
                "pending_colors": [],
                "current_color": 0,
                "next_color": 1,
            }
        },
    )
    env.sim.balls[0].should_remove = True
    env.sim.stop_adding = True

    observation, reward, terminated, truncated, info = env.step(
        env.encode_action(0, 0)
    )

    assert terminated
    assert not truncated
    assert env.sim.tick_count == 1
    assert env.sim.win_pending
    assert env.sim.outcome is None
    assert info["ticks_advanced"] == 1
    assert info["outcome"] == "win"
    assert info["native_outcome"] is None
    assert info["win_pending"] is True
    assert reward == pytest.approx(config.step_penalty + config.win_reward)
    global_values = observation[env.observation_layout["globals"]]
    assert global_values[15] == 1.0
    with pytest.raises(RuntimeError, match="after episode end"):
        env.step(env.encode_action(0, 0))


def test_gymnasium_checker_and_observation_bounds() -> None:
    env = _env(config=RevengeEnvConfig(frame_skip=2, max_ticks=100))
    check_env(env, skip_render_check=True)
    observation, _ = env.reset(seed=99)
    assert observation.dtype == np.float32
    assert observation.shape == (env.observation_size,)
    assert np.all(np.isfinite(observation))
    assert env.observation_space.contains(observation)

    for _ in range(12):
        observation, _, terminated, truncated, _ = env.step(
            env.action_space.sample()
        )
        assert env.observation_space.contains(observation)
        if terminated or truncated:
            break


def _human_test_state() -> dict[str, object]:
    return {
        "colors": [0, 1],
        "waypoints": [100.0, 150.0],
        "contacts": [True],
        "pending_colors": [2],
        "current_color": 0,
        "next_color": 1,
    }


def test_human_speedrun_contract_preserves_spaces_and_guarantees_win_order() -> None:
    base = _env(
        config=RevengeEnvConfig(
            aim_bins=180,
            frame_skip=1,
            max_ticks=12_000,
        )
    )
    env = HumanSpeedrunWrapper(base)

    assert env.action_space is base.action_space
    assert env.observation_space is base.observation_space
    contract = env.contract()
    assert contract["input"]["profile_id"] == "elite-human-v1"
    assert contract["reward"]["profile_id"] == "win-time-score-v1"
    assert contract["base_environment_unchanged"] is True

    rewards = env.reward_config
    minimum_win = (
        rewards.win_reward
        + rewards.time_penalty_per_native_tick * base.config.max_ticks
    )
    maximum_failure = (
        rewards.failure_reward + rewards.score_progress_reward_cap
    )
    assert minimum_win > maximum_failure
    assert (
        -rewards.time_penalty_per_native_tick * base.tick_hz
        >= rewards.score_progress_reward_cap
    )


def test_human_speedrun_delays_button_and_limits_aim_motion() -> None:
    base = _env(config=RevengeEnvConfig(aim_bins=180, frame_skip=1))
    env = HumanSpeedrunWrapper(base)
    env.reset(seed=3, options={"state": _human_test_state()})
    target_bin = 45

    execution: list[tuple[int, int]] = []
    for tick in range(13):
        verb = 1 if tick == 0 else 0
        _, _, terminated, truncated, info = env.step(
            base.encode_action(verb, target_bin)
        )
        assert not terminated
        assert not truncated
        executed = info["human_speedrun"]["executed_verb"]
        if executed:
            execution.append((tick, executed))

    assert execution == [(12, 1)]
    assert info["action_accepted"] is True
    # Motion starts only after the 120 ms reaction delay, then accelerates.  It
    # therefore cannot teleport to the requested 91-degree bin on that tick.
    assert 0.0 < info["human_speedrun"]["aim_angle_degrees"] < 20.0
    assert (
        abs(info["human_speedrun"]["aim_velocity_degrees_per_second"])
        <= env.input_config.max_aim_speed_degrees_per_second
    )


def test_human_speedrun_enforces_swap_to_fire_button_interval() -> None:
    base = _env(config=RevengeEnvConfig(aim_bins=180, frame_skip=1))
    env = HumanSpeedrunWrapper(base)
    env.reset(seed=5, options={"state": _human_test_state()})

    executions: list[tuple[int, int]] = []
    for tick in range(20):
        verb = 2 if tick == 0 else (1 if tick == 1 else 0)
        _, _, terminated, truncated, info = env.step(base.encode_action(verb, 0))
        assert not terminated
        assert not truncated
        executed = info["human_speedrun"]["executed_verb"]
        if executed:
            executions.append((tick, executed))

    assert executions == [(12, 2), (17, 1)]
    assert executions[1][0] - executions[0][0] == 5


def test_human_speedrun_reward_is_bounded_progress_time_and_outcome() -> None:
    base = _env(
        config=RevengeEnvConfig(
            aim_bins=180,
            frame_skip=1,
            score_reward_scale=99.0,
            step_penalty=-7.0,
            win_reward=777.0,
            loss_reward=-777.0,
        )
    )
    env = HumanSpeedrunWrapper(base)
    env.reset(seed=7, options={"state": _human_test_state()})
    base.sim.score = 100

    _, reward, terminated, truncated, info = env.step(base.encode_action(0, 0))

    assert not terminated
    assert not truncated
    # The synthetic curve target is 1,000, so 100 points advances progress by
    # 0.1.  The score term is 0.01 * 0.1, followed by one native-tick cost.
    assert info["reward_terms"] == pytest.approx(
        {
            "score_progress": 0.001,
            "elapsed_time": -0.0001,
            "outcome": 0.0,
        }
    )
    assert reward == pytest.approx(0.0009)
    assert info["native_environment_reward"] == pytest.approx(-7.0)


def test_human_speedrun_rejects_non_lexicographic_reward_contract() -> None:
    base = _env(config=RevengeEnvConfig(frame_skip=1))
    with pytest.raises(ValueError, match="one second"):
        HumanSpeedrunWrapper(
            base,
            reward_config=WinFirstRewardConfig(
                score_progress_reward_cap=0.0101,
            ),
        )
    with pytest.raises(ValueError, match="frame_skip=1"):
        HumanSpeedrunWrapper(
            _env(config=RevengeEnvConfig(frame_skip=2)),
            input_config=EliteHumanInputConfig(),
        )


def test_debug_renderers_need_no_original_assets() -> None:
    ansi_env = _env(render_mode="ansi")
    ansi_env.reset(seed=3)
    assert "tick=" in ansi_env.render()

    rgb_env = _env(
        config=RevengeEnvConfig(render_width=160, render_height=120),
        render_mode="rgb_array",
    )
    rgb_env.reset(seed=3)
    frame = rgb_env.render()
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (120, 160, 3)
    assert frame.dtype == np.uint8
