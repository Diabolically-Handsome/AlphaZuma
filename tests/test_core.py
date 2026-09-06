import numpy as np
import pytest

from zuma_rl import SpiralTrack, ZumaConfig, ZumaSimulator, resolve_matches


def test_resolve_simple_match() -> None:
    remaining, removed, cascades = resolve_matches([0, 1, 1, 1, 2], 2)
    assert remaining == [0, 2]
    assert removed == 3
    assert cascades == 1


def test_resolve_boundary_cascade() -> None:
    remaining, removed, cascades = resolve_matches([2, 2, 1, 1, 1, 2], 3)
    assert remaining == []
    assert removed == 6
    assert cascades == 2


def test_nonmatching_insertion_does_not_scan_unrelated_runs() -> None:
    remaining, removed, cascades = resolve_matches([0, 0, 0, 1, 2], 4)
    assert remaining == [0, 0, 0, 1, 2]
    assert removed == 0
    assert cascades == 0


def test_invalid_insertion_index() -> None:
    with pytest.raises(IndexError):
        resolve_matches([0, 1], 2)


def test_initial_chain_never_contains_a_match() -> None:
    config = ZumaConfig(match_size=3)
    simulator = ZumaSimulator(config, rng=np.random.default_rng(123))
    for index in range(len(simulator.balls) - 2):
        assert len(set(simulator.balls[index : index + 3])) > 1


def test_track_uses_approximately_uniform_arc_progress() -> None:
    track = SpiralTrack(ZumaConfig())
    points = track.points_at(np.linspace(0.0, 1.0, 101))
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    assert np.max(distances) / np.min(distances) < 1.1


def test_action_towards_hits_a_ball() -> None:
    simulator = ZumaSimulator(ZumaConfig(aim_bins=360))
    positions = simulator.ball_positions()
    action = simulator.action_towards(positions[-1])
    angle, _ = simulator.action_to_angle(action)
    assert simulator._ray_hit(angle) is not None


def test_clearing_chain_wins_before_chain_advances() -> None:
    config = ZumaConfig(aim_bins=720)
    simulator = ZumaSimulator(config)
    simulator.load_state(
        [0, 0],
        front_progress=0.999,
        current_color=0,
        next_color=1,
    )
    action = simulator.action_towards(simulator.ball_positions()[-1])
    result = simulator.step(action)
    assert result.terminated
    assert not result.truncated
    assert result.info["outcome"] == "win"
    assert simulator.balls == []


def test_miss_can_reach_skull() -> None:
    config = ZumaConfig(
        aim_bins=180,
        initial_front_progress=0.9,
        chain_speed_per_shot=0.2,
    )
    simulator = ZumaSimulator(config)
    simulator.load_state([0, 1], front_progress=0.9)
    positions = simulator.ball_positions()
    ball_angles = np.arctan2(
        positions[:, 1] - config.shooter_y,
        positions[:, 0] - config.shooter_x,
    ) % (2 * np.pi)
    for action in range(config.aim_bins):
        angle, _ = simulator.action_to_angle(action)
        angular_distance = np.min(
            np.abs(np.angle(np.exp(1j * (angle - ball_angles))))
        )
        if angular_distance > 0.2:
            break
    result = simulator.step(action)
    assert result.terminated
    assert result.info["outcome"] == "skull"


def test_step_after_done_is_rejected() -> None:
    config = ZumaConfig(chain_speed_per_shot=1.0)
    simulator = ZumaSimulator(config)
    simulator.step(0)
    with pytest.raises(RuntimeError):
        simulator.step(0)

