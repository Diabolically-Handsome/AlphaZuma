"""Tick-level regression tests for the high-fidelity Revenge core.

These tests intentionally use a straight synthetic curve.  That keeps every
expected waypoint and collision position hand-checkable, while exercising the
same public interface used with decoded ``OriginalCurve`` instances.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import pytest

from zuma_rl.original_data import CurveParameters
from zuma_rl.revenge_core import (
    ChainBall,
    FruitSpawnCalibration,
    GunState,
    MsvcCRTRandom,
    PopCapMTRandom,
    PowerupSpawnCalibration,
    PowerupType,
    Projectile,
    RevengePhysicsConfig,
    RevengeSimulator,
)


@dataclass(slots=True)
class SyntheticCurve:
    """Minimal ``OriginalCurve``-compatible, one-pixel sampled straight path."""

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
        values = np.asarray(waypoint, dtype=np.float32)
        if loop_at_end:
            values = np.mod(values, len(self.points))
        clipped = np.clip(
            values,
            np.float32(0.0),
            np.float32(self.end_waypoint),
        )
        lower = np.trunc(clipped).astype(np.int64)
        upper = np.minimum(lower + 1, self.end_waypoint)
        fraction = np.subtract(
            clipped,
            lower.astype(np.float32),
            dtype=np.float32,
        )
        delta = np.subtract(
            self.points[upper],
            self.points[lower],
            dtype=np.float32,
        )
        return np.add(
            self.points[lower],
            np.multiply(
                fraction[..., np.newaxis],
                delta,
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

    def perpendicular_at_waypoint(
        self,
        waypoint: float | Iterable[float],
    ) -> np.ndarray:
        values = np.asarray(waypoint)
        shape = values.shape + (2,)
        return np.broadcast_to(
            np.array((0.0, -1.0), dtype=np.float32),
            shape,
        ).copy()

    def is_in_tunnel_at_waypoint(
        self,
        waypoint: float | Iterable[float],
    ) -> np.ndarray | np.bool_:
        values = np.asarray(waypoint, dtype=np.float32)
        indices = np.trunc(values).astype(np.int64)
        result = np.zeros(values.shape, dtype=np.bool_)
        result = np.where(indices < 0, True, result)
        valid = (indices >= 0) & (indices <= self.end_waypoint)
        result = np.where(
            valid,
            self.in_tunnel[np.clip(indices, 0, self.end_waypoint)],
            result,
        )
        return result


def _parameters(
    *,
    speed: float = 0.0,
    zuma_score: int = 100,
    zuma_back_distance: int = 300,
    zuma_slow_duration: int = 1_100,
    powerup_records: tuple[tuple[int, int], ...] = (),
) -> CurveParameters:
    return CurveParameters(
        start_distance_percent=65,
        num_balls=0,
        ball_repeat_chance=45,
        max_single=2,
        colors=4,
        speed=speed,
        slow_distance=200,
        acceleration_rate=0.0,
        max_speed=100.0,
        zuma_score=zuma_score,
        skull_rotation_degrees=75,
        zuma_back_distance=zuma_back_distance,
        zuma_slow_duration=zuma_slow_duration,
        slow_factor=4.0,
        max_clump_size=6,
        powerup_records=powerup_records,
        powerup_chance=600,
    )


def _line_curve(
    *,
    length: int = 1_000,
    y: float = 100.0,
    speed: float = 0.0,
    zuma_score: int = 100,
    zuma_back_distance: int = 300,
    zuma_slow_duration: int = 1_100,
    tunnel_waypoints: Iterable[int] = (),
    die_at_end: bool = True,
    powerup_records: tuple[tuple[int, int], ...] = (),
) -> SyntheticCurve:
    x = np.arange(length + 1, dtype=np.float32)
    points = np.column_stack((x, np.full_like(x, y)))
    in_tunnel = np.zeros(length + 1, dtype=np.bool_)
    for waypoint in tunnel_waypoints:
        in_tunnel[waypoint] = True
    return SyntheticCurve(
        points=points,
        in_tunnel=in_tunnel,
        parameters=_parameters(
            speed=speed,
            zuma_score=zuma_score,
            zuma_back_distance=zuma_back_distance,
            zuma_slow_duration=zuma_slow_duration,
            powerup_records=powerup_records,
        ),
        die_at_end=die_at_end,
    )


def _isolated_simulator(
    *,
    curve: SyntheticCurve | None = None,
    shooter: tuple[float, float] = (400.0, 300.0),
    seed: int = 1,
) -> RevengeSimulator:
    simulator = RevengeSimulator(
        curve or _line_curve(),
        shooter=shooter,
        seed=seed,
    )
    simulator.stop_adding = True
    simulator.pending_colors.clear()
    return simulator


def _load_isolated(
    simulator: RevengeSimulator,
    *,
    colors: list[int],
    waypoints: list[float],
    contacts: list[bool] | None = None,
) -> None:
    simulator.load_state(
        colors=colors,
        waypoints=waypoints,
        contacts=contacts,
    )
    # ``load_state`` is allowed to reset ordinary episode fields.  Restore the
    # two isolation controls explicitly so no generated ball obscures a
    # one-tick mechanics assertion.
    simulator.stop_adding = True
    simulator.pending_colors.clear()


def _set_curve_chain(
    simulator: RevengeSimulator,
    curve_index: int,
    *,
    colors: list[int],
    waypoints: list[float],
    contacts: list[bool] | None = None,
    stop_adding: bool = True,
    pending_colors: list[int] | None = None,
) -> None:
    if len(colors) != len(waypoints):
        raise ValueError("test chain colors and waypoints must align")
    original_curve = simulator.active_curve_index
    simulator._activate_curve(curve_index)
    simulator.balls = [
        ChainBall(
            id=simulator._new_id(),
            color=color,
            waypoint=np.float32(waypoint),
            radius=simulator.config.ball_radius,
        )
        for color, waypoint in zip(colors, waypoints, strict=True)
    ]
    for index, ball in enumerate(simulator.balls[:-1]):
        ball.contact_next = (
            True if contacts is None else bool(contacts[index])
        )
    if simulator.balls:
        simulator.balls[-1].contact_next = False
    simulator.pending_colors = list(pending_colors or ())
    simulator.merging_projectiles.clear()
    simulator.stop_adding = stop_adding
    simulator.num_balls_created = (
        len(simulator.balls) + len(simulator.pending_colors)
    )
    simulator.advance_speed = np.float32(0.0)
    simulator.current_acceleration = np.float32(0.0)
    simulator.slow_count = 0
    simulator.backward_count = 0
    simulator.stop_time = 0
    simulator._have_sets = False
    simulator._activate_curve(original_curve)


def _attach(
    simulator: RevengeSimulator,
    *,
    color: int,
    hit_index: int,
    hit_in_front: bool,
) -> Projectile:
    projectile = simulator.launch_projectile(0.0, color=color)
    simulator.attach_projectile(
        projectile,
        hit_index=hit_index,
        hit_in_front=hit_in_front,
    )
    return projectile


def _snapshot(simulator: RevengeSimulator) -> tuple[object, ...]:
    return (
        tuple(simulator.pending_colors),
        tuple(
            (
                ball.color,
                float(ball.waypoint),
                ball.contact_next,
                ball.exploding,
                ball.suck_count,
            )
            for ball in simulator.balls
        ),
        tuple(
            (
                projectile.color,
                tuple(map(float, projectile.position)),
                tuple(map(float, projectile.velocity)),
                float(projectile.hit_percent),
            )
            for projectile in simulator.free_projectiles
        ),
        simulator.score,
        simulator.stop_adding,
        simulator.zuma_reached,
        simulator.outcome,
    )


def test_pc_candidate_constants_are_explicit() -> None:
    config = RevengePhysicsConfig()
    assert config.tick_hz == 100
    assert config.ball_radius == pytest.approx(18.0)
    assert config.projectile_speed == pytest.approx(8.0)
    assert config.merge_speed == pytest.approx(0.025)
    assert config.explosion_ticks == 40
    assert config.zuma_bar_width == 330
    assert config.powerup_lifetime_ticks == 2_000
    assert config.powerup_transition_ticks == 100
    assert config.powerup_previous_retention_ticks == 150
    assert config.proximity_bomb_collision_pad == 56
    assert config.proximity_bomb_fruit_radius == 108.0
    assert config.reverse_powerup_ticks == 300
    assert config.reverse_speed == pytest.approx(1.0)
    assert config.slow_powerup_ticks == 800
    assert config.slow_powerup_replace_threshold == 1_000
    assert config.loss_initial_suck_count == 1
    assert config.loss_suction_speed_shift == 2


def _fruit_calibration(
    *,
    frequency_ticks: int = 1_000,
    lifetime_ticks: int = 1_000,
    unlock_percentages: tuple[tuple[int, ...], ...] = (
        (60,),
        (33,),
        (84,),
    ),
) -> FruitSpawnCalibration:
    return FruitSpawnCalibration(
        frequency_ticks=frequency_ticks,
        lifetime_ticks=lifetime_ticks,
        points=((28.0, 105.0), (716.0, 505.0), (602.0, 366.0)),
        unlock_percentages=unlock_percentages,
        provenance="unit-test retail static proof",
    )


def test_fruit_chance_starts_only_after_strict_tfreq_boundary() -> None:
    simulator = RevengeSimulator(
        _line_curve(),
        seed=5489,
        fruit_calibration=_fruit_calibration(),
    )
    simulator.stop_adding = True
    simulator.pending_colors.clear()
    initial_rng = simulator.rng.state

    simulator.native_game_time = 1_000
    simulator._update_fruit_scheduler()
    assert simulator.rng.state == initial_rng
    assert simulator.last_events.fruit_chance_draws == 0

    reference = PopCapMTRandom(1)
    reference.load_state(*initial_rng)
    reference.next_u31()
    simulator.native_game_time = 1_001
    simulator._update_fruit_scheduler()

    assert simulator.rng.state == reference.state
    assert simulator.last_events.fruit_chance_draws == 1
    assert simulator.fruit_active_point_index is None


def test_fruit_spawn_uses_front_ball_progress_and_second_mt_choice() -> None:
    simulator = RevengeSimulator(
        _line_curve(length=999),
        seed=5489,
        fruit_calibration=_fruit_calibration(frequency_ticks=1),
    )
    _load_isolated(simulator, colors=[0, 1], waypoints=[1.0, 400.0])
    simulator.native_game_time = 2
    reference = PopCapMTRandom(1)
    reference.load_state(*simulator.rng.state)
    reference.next_u31()
    reference.next_u31()

    simulator._update_fruit_scheduler()

    # The low-waypoint head remains at 0%, while the front at 400 truncates to
    # 40%.  Only Jungle2 point 1 (33%) is eligible, and retail still consumes
    # a second MT draw to select it.
    assert simulator._eligible_fruit_points() == (1,)
    assert simulator.fruit_active_point_index == 1
    assert simulator.fruit_expiry_time == 1_002
    assert simulator.fruit_spawn_count == 1
    assert simulator.last_events.fruit_chance_draws == 1
    assert simulator.last_events.fruits_spawned == 1
    assert simulator.rng.state == reference.state


def test_fruit_expiry_tick_returns_without_a_new_chance_draw() -> None:
    simulator = RevengeSimulator(
        _line_curve(),
        seed=5489,
        fruit_calibration=_fruit_calibration(frequency_ticks=1),
    )
    simulator.fruit_active_point_index = 0
    simulator.fruit_expiry_time = 10
    initial_rng = simulator.rng.state

    simulator.native_game_time = 10
    simulator._update_fruit_scheduler()

    assert simulator.fruit_active_point_index is None
    assert simulator.last_events.fruits_expired == 1
    assert simulator.last_events.fruit_chance_draws == 0
    assert simulator.rng.state == initial_rng

    simulator.native_game_time = 11
    simulator._update_fruit_scheduler()
    assert simulator.rng.state == initial_rng
    simulator.native_game_time = 12
    simulator._update_fruit_scheduler()
    assert simulator.last_events.fruit_chance_draws == 1
    assert simulator.rng.state != initial_rng


def test_fruit_bob_matches_c111_float32_extrema_and_bound_snap() -> None:
    simulator = RevengeSimulator(
        _line_curve(),
        seed=5489,
        fruit_calibration=_fruit_calibration(),
    )
    simulator.tick_count = 3_629
    simulator.native_game_time = 617
    simulator.fruit_velocity = np.float32(0.25)
    simulator.fruit_acceleration = np.float32(0.01)
    simulator.fruit_vertical_offset = np.float32(-4.351139068603516e-06)
    simulator.fruit_lower_bound = np.float32(-1.862645149230957e-06)
    simulator.fruit_upper_bound = np.float32(-4.351139068603516e-06)
    simulator.fruit_cell_index = 25

    simulator._update_fruit_visual()

    assert simulator.fruit_velocity == np.float32(0.25999999046325684)
    assert simulator.fruit_acceleration == np.float32(-0.01)
    assert simulator.fruit_vertical_offset == np.float32(
        0.25999563932418823
    )
    assert simulator.fruit_cell_index == 25

    simulator.tick_count = 3_680
    simulator.native_game_time = 668
    simulator.fruit_velocity = np.float32(-0.24000009894371033)
    simulator.fruit_acceleration = np.float32(-0.01)
    simulator.fruit_vertical_offset = np.float32(0.5099921226501465)
    simulator._update_fruit_visual()

    assert simulator.fruit_velocity == np.float32(-0.25000008940696716)
    assert simulator.fruit_acceleration == np.float32(0.01)
    assert simulator.fruit_vertical_offset == np.float32(
        -0.2500019669532776
    )


def test_projectile_collects_fruit_before_curve_collision_and_scores_tier() -> None:
    simulator = RevengeSimulator(
        _line_curve(),
        seed=5489,
        fruit_calibration=_fruit_calibration(),
    )
    _load_isolated(simulator, colors=[0], waypoints=[54.0])
    simulator.score = 4_800
    simulator.score_at_level_start = 1_200
    simulator.fruit_active_point_index = 0
    simulator.fruit_vertical_offset = np.float32(0.0)
    center = simulator.fruit_center()
    assert center is not None
    projectile = Projectile(
        id=simulator._new_id(),
        color=0,
        position=center,
        velocity=np.zeros(2, dtype=np.float32),
        radius=simulator.config.ball_radius,
        just_fired=True,
    )
    simulator.free_projectiles.append(projectile)

    simulator._update_free_projectiles()

    assert simulator.free_projectiles == []
    assert simulator.fruit_collecting
    assert simulator.fruit_collect_count == 1
    assert simulator.last_events.fruits_collected == 1
    assert simulator.last_events.hits == 0
    assert simulator.score == 5_400
    assert simulator.last_events.score_delta == 600
    assert simulator.fruit_collection_ticks_remaining == 26

    for _ in range(25):
        simulator._update_fruit_visual()
    assert simulator.fruit_active_point_index == 0
    simulator._update_fruit_visual()
    assert simulator.fruit_active_point_index is None
    assert simulator.fruit_collecting


def test_existing_projectile_moves_before_fruit_collision_test() -> None:
    simulator = RevengeSimulator(
        _line_curve(),
        seed=5489,
        fruit_calibration=_fruit_calibration(),
    )
    simulator.stop_adding = True
    simulator.pending_colors.clear()
    simulator.fruit_active_point_index = 0
    center = simulator.fruit_center()
    assert center is not None
    projectile = Projectile(
        id=simulator._new_id(),
        color=0,
        position=center,
        velocity=np.asarray((100.0, 0.0), dtype=np.float32),
        radius=simulator.config.ball_radius,
        just_fired=False,
    )
    simulator.free_projectiles.append(projectile)

    simulator._update_free_projectiles()

    assert simulator.fruit_collect_count == 0
    assert simulator.fruit_active_point_index == 0
    assert projectile.position[0] == np.float32(center[0] + 100.0)


@pytest.mark.parametrize(
    ("fruit_x", "expected_collection"),
    ((207.999, True), (208.0, False)),
)
def test_proximity_bomb_collects_fruit_with_strict_retail_radius(
    fruit_x: float,
    expected_collection: bool,
) -> None:
    calibration = FruitSpawnCalibration(
        frequency_ticks=1_000,
        lifetime_ticks=1_000,
        points=((fruit_x, 100.0),),
        unlock_percentages=((1,),),
        provenance="unit-test retail static proof",
    )
    simulator = RevengeSimulator(
        _line_curve(),
        seed=5489,
        fruit_calibration=calibration,
    )
    _load_isolated(simulator, colors=[0], waypoints=[100.0])
    simulator.score = 4_800
    simulator.score_at_level_start = 1_200
    simulator.fruit_active_point_index = 0
    ball = simulator.balls[0]
    ball.powerup_primary_type = int(PowerupType.PROXIMITY_BOMB)
    simulator.active_powerup_color_counts[ball.color] = 1

    simulator._begin_ball_explosion(ball)

    assert simulator.fruit_collecting is expected_collection
    assert simulator.last_events.fruits_collected == int(
        expected_collection
    )
    assert simulator.score == (5_400 if expected_collection else 4_800)


def test_tick_runs_board_fruit_scheduler_before_curve_updates() -> None:
    simulator = RevengeSimulator(
        _line_curve(),
        seed=5489,
        fruit_calibration=_fruit_calibration(),
    )
    _load_isolated(simulator, colors=[0], waypoints=[100.0])
    order: list[str] = []
    original_fruit = simulator._update_fruit_scheduler
    original_curves = simulator._tick_all_curves

    def fruit() -> None:
        order.append("fruit")
        original_fruit()

    def curves() -> None:
        order.append("curves")
        original_curves()

    simulator._update_fruit_scheduler = fruit  # type: ignore[method-assign]
    simulator._tick_all_curves = curves  # type: ignore[method-assign]
    simulator.tick()

    assert order == ["fruit", "curves"]


def test_powerup_activation_completion_starts_1999_sampled_ticks() -> None:
    simulator = _isolated_simulator()
    _load_isolated(simulator, colors=[1], waypoints=[100.0])
    ball = simulator.balls[0]
    ball.powerup_secondary_type = int(PowerupType.REVERSE)
    ball.powerup_transition_ticks = 1
    ball.powerup_visual_scale = np.float32(1.04)
    ball.powerup_visual_step = np.float32(0.04)
    ball.powerup_visual_index = 4
    simulator.active_powerup_color_counts[1] = 1

    simulator.native_game_time = 100
    simulator._update_ball_objects()

    assert ball.powerup_transition_ticks == 0
    assert ball.powerup_primary_type == int(PowerupType.REVERSE)
    assert ball.powerup_secondary_type == int(PowerupType.NONE)
    assert ball.powerup_lifetime_ticks == 1_999
    assert ball.powerup_visual_scale == pytest.approx(1.0, abs=1e-5)
    assert ball.powerup_visual_index == -1


def test_reverse_powerup_expiry_matches_retail_100_and_150_tick_phases() -> None:
    simulator = _isolated_simulator()
    _load_isolated(simulator, colors=[1], waypoints=[100.0])
    ball = simulator.balls[0]
    ball.powerup_primary_type = int(PowerupType.REVERSE)
    ball.powerup_lifetime_ticks = 3
    ball.powerup_visual_scale = np.float32(1.0000038)
    ball.powerup_visual_step = np.float32(0.04)
    simulator.active_powerup_color_counts[1] = 1
    simulator.powerup_cooldown_times[int(PowerupType.REVERSE)] = -1_000

    for native_time in (101, 102, 103):
        simulator.native_game_time = native_time
        simulator._update_ball_objects()

    assert ball.powerup_lifetime_ticks == 0
    assert ball.powerup_previous_type == int(PowerupType.REVERSE)
    assert ball.powerup_previous_ticks == 149
    assert ball.powerup_primary_type == int(PowerupType.REVERSE)
    assert ball.powerup_secondary_type == int(PowerupType.NONE)
    assert ball.powerup_transition_ticks == 100
    assert ball.powerup_visual_scale == pytest.approx(5.0)
    assert simulator.powerup_cooldown_times[
        int(PowerupType.REVERSE)
    ] == 103
    assert simulator.active_powerup_color_counts[1] == 0

    for native_time in range(104, 204):
        simulator.native_game_time = native_time
        simulator._update_ball_objects()

    assert ball.powerup_transition_ticks == 0
    assert ball.powerup_primary_type == int(PowerupType.NONE)
    assert ball.powerup_previous_type == int(PowerupType.REVERSE)
    assert ball.powerup_previous_ticks == 49
    assert ball.powerup_visual_scale == pytest.approx(
        1.0000038,
        abs=1e-5,
    )

    for native_time in range(204, 253):
        simulator.native_game_time = native_time
        simulator._update_ball_objects()

    assert ball.powerup_previous_ticks == 0
    assert ball.powerup_previous_type == int(PowerupType.NONE)


def test_exploding_powerup_ball_freezes_remaining_lifetime() -> None:
    simulator = _isolated_simulator()
    _load_isolated(simulator, colors=[1], waypoints=[100.0])
    ball = simulator.balls[0]
    ball.powerup_primary_type = int(PowerupType.REVERSE)
    ball.powerup_lifetime_ticks = 1_966
    ball.exploding = True
    ball.explode_frame = 1

    simulator._update_ball_objects()

    assert ball.update_count == 1
    assert ball.powerup_lifetime_ticks == 1_966


def test_calibrated_powerup_spawn_consumes_four_ordered_mt_choices() -> None:
    calibration = PowerupSpawnCalibration(
        chance_denominator=1,
        initial_delay_ticks=0,
        spawn_delay_ticks=0,
        cooldown_ticks=0,
        unique_color=True,
        supported_types=(
            int(PowerupType.PROXIMITY_BOMB),
            int(PowerupType.REVERSE),
        ),
        provenance="unit-test",
    )
    simulator = RevengeSimulator(
        _line_curve(
            powerup_records=(
                (100, 0),
                (0, 0),
                (0, 0),
                (100, 0),
            ),
        ),
        seed=5489,
        powerup_calibration=calibration,
    )
    _load_isolated(
        simulator,
        colors=[0, 1, 1, 1, 2, 3],
        waypoints=[100.0, 140.0, 180.0, 220.0, 260.0, 300.0],
    )
    simulator.native_game_time = 2_000
    reference = PopCapMTRandom(1)
    reference.load_state(*simulator.rng.state)
    outputs = tuple(reference.next_u31() for _ in range(4))
    selected_type = (
        int(PowerupType.PROXIMITY_BOMB)
        if outputs[1] % 200 < 100
        else int(PowerupType.REVERSE)
    )
    selected_color = outputs[2] % 4
    candidates = [
        ball for ball in simulator.balls if ball.color == selected_color
    ]
    expected_ball = candidates[outputs[3] % len(candidates)]

    selected_ball = simulator._maybe_spawn_powerup()

    assert selected_ball is expected_ball
    assert selected_ball.powerup_primary_type == int(PowerupType.NONE)
    assert selected_ball.powerup_secondary_type == selected_type
    assert selected_ball.powerup_transition_ticks == 100
    assert selected_ball.powerup_visual_scale == pytest.approx(5.0)
    assert selected_ball.powerup_visual_index == {
        int(PowerupType.PROXIMITY_BOMB): 3,
        int(PowerupType.REVERSE): 4,
    }[selected_type]
    assert simulator.active_powerup_color_counts[selected_color] == 1
    assert simulator.powerup_last_any_spawn_time == 2_000
    assert simulator.powerup_last_spawn_times[selected_type] == 2_000
    assert simulator.powerup_spawn_counts[selected_type] == 1
    assert simulator.last_events.powerups_spawned == 1
    assert simulator.rng.state == reference.state


def test_effective_powerup_uses_primary_then_live_previous_then_secondary() -> None:
    ball = ChainBall(
        id=1,
        color=0,
        waypoint=np.float32(100.0),
        radius=18,
        powerup_primary_type=int(PowerupType.REVERSE),
        powerup_previous_type=int(PowerupType.PROXIMITY_BOMB),
        powerup_previous_ticks=1,
        powerup_secondary_type=int(PowerupType.SLOW),
    )

    assert (
        RevengeSimulator._effective_ball_powerup(ball)
        == int(PowerupType.REVERSE)
    )
    ball.powerup_primary_type = int(PowerupType.NONE)
    assert (
        RevengeSimulator._effective_ball_powerup(ball)
        == int(PowerupType.PROXIMITY_BOMB)
    )
    ball.powerup_previous_ticks = 0
    assert (
        RevengeSimulator._effective_ball_powerup(ball)
        == int(PowerupType.SLOW)
    )


def test_bomb_uses_strict_spatial_radius_and_recursively_triggers_reverse() -> None:
    simulator = _isolated_simulator()
    _load_isolated(
        simulator,
        colors=[1, 1, 1, 2, 3],
        waypoints=[100.0, 136.0, 172.0, 244.0, 248.0],
        contacts=[True, True, False, False],
    )
    bomb = simulator.balls[0]
    reverse = simulator.balls[3]
    bomb.powerup_primary_type = int(PowerupType.PROXIMITY_BOMB)
    bomb.powerup_lifetime_ticks = 500
    reverse.powerup_primary_type = int(PowerupType.REVERSE)
    reverse.powerup_lifetime_ticks = 500
    simulator.active_powerup_color_counts[1] = 1
    simulator.active_powerup_color_counts[2] = 1
    simulator.native_game_time = 4_321
    bomb_fields = (
        bomb.powerup_primary_type,
        bomb.powerup_lifetime_ticks,
    )
    reverse_fields = (
        reverse.powerup_primary_type,
        reverse.powerup_lifetime_ticks,
    )

    assert simulator._check_set(1)

    # With 18 px balls and native pad 56, distance 144 is inside the strict
    # 148 px threshold while distance 148 is exactly outside it.
    assert [ball.exploding for ball in simulator.balls] == [
        True,
        True,
        True,
        True,
        False,
    ]
    assert simulator.score == 40
    assert simulator.last_events.balls_exploded == 4
    assert [ball.combo_score for ball in simulator.balls[:4]] == [40] * 4
    assert simulator.last_events.powerups_triggered == 2
    assert simulator.powerup_field_124_by_type[
        int(PowerupType.PROXIMITY_BOMB)
    ] == 1
    assert simulator.powerup_field_124_by_type[
        int(PowerupType.REVERSE)
    ] == 1
    assert simulator.powerup_cooldown_times[
        int(PowerupType.PROXIMITY_BOMB)
    ] == 4_321
    assert simulator.powerup_cooldown_times[
        int(PowerupType.REVERSE)
    ] == 4_321
    assert simulator.active_powerup_color_counts[1] == 0
    assert simulator.active_powerup_color_counts[2] == 0
    assert simulator.backward_count == 300
    assert simulator.powerup_triggered
    assert simulator.last_powerup_waypoint == 244
    assert (
        bomb.powerup_primary_type,
        bomb.powerup_lifetime_ticks,
    ) == bomb_fields
    assert (
        reverse.powerup_primary_type,
        reverse.powerup_lifetime_ticks,
    ) == reverse_fields


def test_check_gun_colors_replaces_only_absent_chamber_color() -> None:
    simulator = _isolated_simulator()
    simulator.load_state(
        colors=[1, 1],
        waypoints=[100.0, 136.0],
        current_color=0,
        next_color=1,
    )
    update_count = simulator._color_chooser.update_count

    simulator._check_gun_colors()

    assert simulator.current_color == 1
    assert simulator.next_color == 1
    assert simulator._color_chooser.update_count == update_count + 1


@pytest.mark.parametrize("locked_field", ("skull_entry_pending", "loss_started"))
def test_check_gun_colors_preserves_locked_loss_chamber(
    locked_field: str,
) -> None:
    simulator = _isolated_simulator()
    simulator.load_state(
        colors=[0],
        waypoints=[100.0],
        current_color=0,
        next_color=2,
    )
    simulator.load_shooter_random_state(
        crt_state=3_527_905_494,
        update_count=2,
        selected_index=2,
        weights=(0.25, 0.25, 0.25, 0.25, 0.0, 0.0),
        sways=(
            0.0832500010728836,
            0.109375,
            0.109375,
            0.109375,
            0.0,
            0.0,
        ),
        last_hit=(1, 0, 2, 0, 0, 0),
        previous_hit=(0, 0, 0, 0, 0, 0),
    )
    setattr(simulator, locked_field, True)
    chooser = simulator._color_chooser
    before = (
        simulator.current_color,
        simulator.next_color,
        simulator.crt_rng.state,
        chooser.update_count,
        chooser.selected_index,
        chooser.allowed_support,
        tuple(map(float, chooser.weights)),
        tuple(map(float, chooser.sways)),
        tuple(map(int, chooser.last_hit)),
        tuple(map(int, chooser.previous_hit)),
    )

    simulator._check_gun_colors()

    after = (
        simulator.current_color,
        simulator.next_color,
        simulator.crt_rng.state,
        chooser.update_count,
        chooser.selected_index,
        chooser.allowed_support,
        tuple(map(float, chooser.weights)),
        tuple(map(float, chooser.sways)),
        tuple(map(int, chooser.last_hit)),
        tuple(map(int, chooser.previous_hit)),
    )
    assert after == before


def test_reverse_trigger_moves_connected_chain_on_same_tick() -> None:
    simulator = _isolated_simulator()
    _load_isolated(
        simulator,
        colors=[3, 3, 3, 1],
        waypoints=[100.0, 136.0, 172.0, 208.0],
        contacts=[True, True, True],
    )
    reverse = simulator.balls[1]
    reverse.powerup_primary_type = int(PowerupType.REVERSE)
    reverse.powerup_lifetime_ticks = 446
    simulator.active_powerup_color_counts[3] = 1
    simulator.native_game_time = 7_656
    before = tuple(float(ball.waypoint) for ball in simulator.balls)

    assert simulator._check_set(1)
    simulator._advance_backward_balls()

    assert simulator.backward_count == 300
    assert simulator.score == 30
    assert simulator.powerup_triggered
    assert simulator.powerup_cooldown_times[int(PowerupType.REVERSE)] == 7_656
    assert tuple(float(ball.waypoint) for ball in simulator.balls) == tuple(
        waypoint - 1.0 for waypoint in before
    )
    assert simulator.balls[-1].backwards_count == 0
    assert simulator.balls[-1].backwards_speed == np.float32(1.0)

    simulator._tick_curve()

    assert simulator.backward_count == 299
    assert tuple(float(ball.waypoint) for ball in simulator.balls) == tuple(
        waypoint - 2.0 for waypoint in before
    )


def test_slow_powerup_only_replaces_counts_below_native_threshold() -> None:
    simulator = _isolated_simulator()
    _load_isolated(
        simulator,
        colors=[1, 1, 1],
        waypoints=[100.0, 136.0, 172.0],
        contacts=[True, True],
    )
    ball = simulator.balls[1]
    ball.powerup_primary_type = int(PowerupType.SLOW)
    simulator.active_powerup_color_counts[1] = 1
    simulator.slow_count = 999

    assert simulator._check_set(1)
    assert simulator.slow_count == 800

    simulator = _isolated_simulator()
    _load_isolated(
        simulator,
        colors=[1, 1, 1],
        waypoints=[100.0, 136.0, 172.0],
        contacts=[True, True],
    )
    ball = simulator.balls[1]
    ball.powerup_primary_type = int(PowerupType.SLOW)
    simulator.active_powerup_color_counts[1] = 1
    simulator.slow_count = 1_000

    assert simulator._check_set(1)
    assert simulator.slow_count == 1_000


def test_runtime_float_fields_and_geometry_are_float32() -> None:
    simulator = _isolated_simulator(
        curve=_line_curve(length=2_000, speed=0.5),
        shooter=(400.25, 300.5),
    )
    _load_isolated(simulator, colors=[0], waypoints=[100.125])
    ball = simulator.balls[0]

    assert simulator.shooter.dtype == np.dtype(np.float32)
    assert simulator._curve_points is not None
    assert simulator._curve_points.dtype == np.dtype(np.float32)
    assert isinstance(simulator.advance_speed, np.float32)
    assert isinstance(simulator.current_acceleration, np.float32)
    assert isinstance(simulator.aim_angle, np.float32)
    assert isinstance(ball.waypoint, np.float32)
    assert isinstance(ball.backwards_speed, np.float32)
    assert simulator.ball_position(ball).dtype == np.dtype(np.float32)
    assert all(
        isinstance(value, np.float32)
        for value in simulator._point_xy(ball.waypoint)
    )
    assert all(
        isinstance(value, np.float32)
        for value in simulator._perpendicular_xy(ball.waypoint)
    )
    assert isinstance(simulator._roll_in_speed(), np.float32)
    assert isinstance(simulator._target_chain_speed(), np.float32)
    simulator.set_aim(np.pi / 7.0)
    assert isinstance(simulator.aim_angle, np.float32)

    projectile = simulator.launch_projectile(0.25, color=1)
    assert projectile.position.dtype == np.dtype(np.float32)
    assert projectile.velocity.dtype == np.dtype(np.float32)
    assert isinstance(projectile.waypoint, np.float32)
    assert isinstance(projectile.hit_percent, np.float32)
    assert isinstance(projectile.merge_speed, np.float32)

    simulator.attach_projectile(
        projectile,
        hit_index=0,
        hit_in_front=True,
    )
    assert projectile.hit_position is not None
    assert projectile.target_position is not None
    assert projectile.hit_position.dtype == np.dtype(np.float32)
    assert projectile.target_position.dtype == np.dtype(np.float32)
    assert isinstance(projectile.waypoint, np.float32)


def test_chain_waypoint_rounds_after_every_tick_not_only_at_the_end() -> None:
    simulator = _isolated_simulator(
        curve=_line_curve(length=5_000, speed=0.01),
    )
    _load_isolated(simulator, colors=[0], waypoints=[1_000.0])
    simulator.advance_speed = np.float32(0.01)

    tick_count = 4_096
    simulator.tick(tick_count)

    sequential = np.float32(1_000.0)
    increment = np.float32(0.01)
    for _ in range(tick_count):
        sequential = np.add(
            sequential,
            increment,
            dtype=np.float32,
        )
    float64_then_cast = np.float32(
        1_000.0 + tick_count * float(increment),
    )

    assert sequential == np.float32(1_041.0)
    assert float64_then_cast == np.float32(1_040.96)
    assert sequential != float64_then_cast
    assert isinstance(simulator.balls[0].waypoint, np.float32)
    assert simulator.balls[0].waypoint == sequential


def test_physical_collision_squares_and_sums_as_float32() -> None:
    origin = np.zeros(2, dtype=np.float32)
    almost_tangent = np.array(
        (0.100358, 35.99986),
        dtype=np.float32,
    )
    exact_float64_distance_squared = (
        float(almost_tangent[0]) ** 2
        + float(almost_tangent[1]) ** 2
    )

    # In real arithmetic this point is just inside radius 36.  The original
    # float expression rounds its two squared terms to exactly 1296.0f, and
    # the strict comparison therefore rejects it as a collision.
    assert exact_float64_distance_squared < 36.0**2
    assert not RevengeSimulator._physical_collision(
        origin,
        18,
        almost_tangent,
        18,
    )


def test_native_entrance_cutoff_keeps_default_without_initial_tunnel() -> None:
    plain = _isolated_simulator(curve=_line_curve())
    tunneled = _isolated_simulator(
        curve=_line_curve(tunnel_waypoints=range(105)),
    )

    assert plain.entrance_cutoff == 15 - plain.config.ball_radius
    assert tunneled.entrance_cutoff == 105 - tunneled.config.ball_radius

    _load_isolated(plain, colors=[0], waypoints=[0.0])
    plain.stop_time = 2
    plain.tick()
    assert plain.stop_time == 1


@pytest.mark.parametrize(
    ("level_id", "curve_count", "gun_type", "positions", "reason"),
    (
        ("Jungle5", 1, "horiz", ((400.0, 300.0),), "moving-frog"),
        ("boss1", 1, "normal", ((400.0, 300.0),), "boss state machine"),
        (
            "ironfrog1",
            1,
            "normal",
            ((400.0, 300.0),),
            "Iron Frog mode",
        ),
    ),
)
def test_installed_factory_fails_closed_for_unsupported_level_structures(
    monkeypatch: pytest.MonkeyPatch,
    level_id: str,
    curve_count: int,
    gun_type: str,
    positions: tuple[tuple[float, float], ...],
    reason: str,
) -> None:
    import zuma_rl.original_data as original_data

    curves = tuple(_line_curve() for _ in range(curve_count))

    class FakeCatalog:
        def __init__(self, root: object = None):
            del root

        def load_level(self, requested: str, *, hard: bool = False) -> object:
            del hard
            assert requested == level_id
            return SimpleNamespace(
                definition=SimpleNamespace(
                    id=level_id,
                    gun=SimpleNamespace(type=gun_type, positions=positions),
                    attributes=(
                        {"ironfrog": "true"}
                        if level_id == "ironfrog1"
                        else {}
                    ),
                ),
                curves=curves,
            )

    monkeypatch.setattr(original_data, "OriginalGameCatalog", FakeCatalog)

    with pytest.raises(NotImplementedError, match=reason):
        RevengeSimulator.from_installed(level_id)

    diagnostic = RevengeSimulator.from_installed(
        level_id,
        curve_index=curve_count - 1,
        gun_index=len(positions) - 1,
        allow_partial_level=True,
    )
    assert diagnostic.shooter == pytest.approx(positions[-1])


def test_installed_factory_preserves_and_switches_dual_frog_positions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import zuma_rl.original_data as original_data

    positions = ((175.0, 310.0), (610.0, 310.0))

    class FakeCatalog:
        def __init__(self, root: object = None):
            del root

        def load_level(self, requested: str, *, hard: bool = False) -> object:
            del hard
            assert requested == "village3"
            return SimpleNamespace(
                definition=SimpleNamespace(
                    id="village3",
                    gun=SimpleNamespace(type="normal", positions=positions),
                    attributes={},
                ),
                curves=(_line_curve(),),
            )

    monkeypatch.setattr(original_data, "OriginalGameCatalog", FakeCatalog)
    simulator = RevengeSimulator.from_installed("village3", gun_index=1)

    assert simulator.shooter_position_count == 2
    assert simulator.active_shooter_index == 1
    np.testing.assert_allclose(np.asarray(simulator.shooter_positions), positions)
    assert simulator.shooter == pytest.approx(positions[1])
    assert simulator.request_hop()
    assert simulator.active_shooter_index == 1
    assert simulator.hop_target_index == 0
    assert simulator.hop_ticks_remaining == 20
    assert simulator.hop_in_progress
    assert not simulator.can_hop_shooter
    assert not simulator.request_hop()
    assert not simulator.request_fire(0.0)
    assert not simulator.swap_balls()

    simulator.tick(19)
    assert simulator.active_shooter_index == 1
    assert simulator.hop_ticks_remaining == 1
    simulator.tick()
    assert simulator.active_shooter_index == 0
    assert simulator.shooter == pytest.approx(positions[0])
    assert not simulator.hop_in_progress
    assert simulator.hop_target_index is None

    simulator.gun_state = GunState.FIRING
    assert not simulator.request_hop()
    assert simulator.active_shooter_index == 0
    simulator.reset(seed=7)
    assert simulator.active_shooter_index == 1
    assert not simulator.hop_in_progress
    assert simulator.hop_target_index is None


def test_installed_factory_builds_one_shared_board_for_two_curves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import zuma_rl.original_data as original_data

    curves = (
        _line_curve(y=100.0, zuma_score=120),
        _line_curve(y=300.0, zuma_score=180),
    )

    class FakeCatalog:
        def __init__(self, root: object = None):
            del root

        def load_level(self, requested: str, *, hard: bool = False) -> object:
            del hard
            assert requested == "Jungle9"
            return SimpleNamespace(
                definition=SimpleNamespace(
                    id="Jungle9",
                    gun=SimpleNamespace(
                        type="normal",
                        positions=((400.0, 300.0),),
                    ),
                    attributes={},
                ),
                curves=curves,
            )

    monkeypatch.setattr(original_data, "OriginalGameCatalog", FakeCatalog)

    simulator = RevengeSimulator.from_installed("Jungle9", seed=17)

    assert simulator.curve_count == 2
    assert simulator.curves == curves
    assert simulator.score_target == 300
    assert [len(state.pending_colors) for state in simulator.curve_states] == [
        10,
        10,
    ]
    assert simulator.free_projectiles == []
    assert all(not hasattr(state, "rng") for state in simulator.curve_states)


def test_two_curves_tick_in_one_shared_native_frame() -> None:
    curves = (
        _line_curve(y=100.0),
        _line_curve(y=300.0),
    )
    simulator = RevengeSimulator(
        curves[0],
        curves=curves,
        seed=23,
    )
    _set_curve_chain(
        simulator,
        0,
        colors=[0],
        waypoints=[100.0],
    )
    _set_curve_chain(
        simulator,
        1,
        colors=[1],
        waypoints=[200.0],
    )

    simulator.tick()

    states = simulator.curve_states
    assert simulator.tick_count == 1
    assert simulator.native_game_time == 1
    assert [state.balls[0].update_count for state in states] == [1, 1]
    assert simulator.active_curve_index == 0
    assert simulator.balls is states[0].balls


def test_shared_zuma_bar_applies_each_curves_own_timers() -> None:
    curves = (
        _line_curve(
            y=100.0,
            zuma_score=120,
            zuma_back_distance=250,
            zuma_slow_duration=900,
        ),
        _line_curve(
            y=300.0,
            zuma_score=180,
            zuma_back_distance=350,
            zuma_slow_duration=1_300,
        ),
    )
    simulator = RevengeSimulator(
        curves[0],
        curves=curves,
        seed=29,
    )
    simulator.score = 300
    simulator.current_bar_size = simulator.config.zuma_bar_width
    simulator.target_bar_size = simulator.config.zuma_bar_width

    simulator._update_zuma_bar()

    states = simulator.curve_states
    assert simulator.score_target == 300
    assert simulator.zuma_reached
    assert [state.backward_count for state in states] == [250, 350]
    assert [state.slow_count for state in states] == [900, 1_300]
    assert all(state.stop_adding for state in states)
    assert all(not state.pending_colors for state in states)


def test_free_projectile_collision_uses_retail_curve_array_order() -> None:
    curves = (
        _line_curve(y=100.0),
        _line_curve(y=100.0),
    )
    simulator = RevengeSimulator(
        curves[0],
        shooter=(100.0, 100.0),
        curves=curves,
        seed=31,
    )
    for curve_index in range(2):
        _set_curve_chain(
            simulator,
            curve_index,
            colors=[curve_index],
            waypoints=[100.0],
        )
    projectile = Projectile(
        id=simulator._new_id(),
        color=2,
        position=np.array((100.0, 100.0), dtype=np.float32),
        velocity=np.zeros(2, dtype=np.float32),
        radius=simulator.config.ball_radius,
        just_fired=False,
    )
    simulator.free_projectiles.append(projectile)

    simulator._update_free_projectiles()

    states = simulator.curve_states
    assert not simulator.free_projectiles
    assert projectile.curve_index == 0
    assert states[0].merging_projectiles == [projectile]
    assert not states[1].merging_projectiles


def test_multi_curve_victory_waits_for_every_curve_to_empty() -> None:
    curves = (
        _line_curve(y=100.0),
        _line_curve(y=300.0),
    )
    simulator = RevengeSimulator(
        curves[0],
        curves=curves,
        seed=37,
    )
    _set_curve_chain(simulator, 0, colors=[], waypoints=[])
    _set_curve_chain(
        simulator,
        1,
        colors=[1],
        waypoints=[100.0],
    )

    simulator.tick()
    assert not simulator.win_pending
    assert simulator.outcome is None

    simulator._activate_curve(1)
    simulator.balls[0].should_remove = True
    simulator._activate_curve(0)
    simulator.tick()

    assert simulator.win_pending
    assert simulator.outcome is None
    simulator.tick()
    assert simulator.outcome == "win"


def test_first_lethal_curve_arms_board_wide_loss_suction() -> None:
    curves = (
        _line_curve(y=100.0),
        _line_curve(y=300.0),
    )
    simulator = RevengeSimulator(
        curves[0],
        curves=curves,
        seed=41,
    )
    for curve_index in range(2):
        _set_curve_chain(
            simulator,
            curve_index,
            colors=[0, 1],
            waypoints=[100.0, 1_000.0],
            contacts=[False],
        )

    simulator.tick()

    assert simulator.skull_entry_pending
    assert simulator.loss_curve_index == 0
    assert not simulator.loss_started

    simulator.tick()

    states = simulator.curve_states
    assert simulator.loss_started
    assert simulator.loss_elapsed_ticks == 1
    assert simulator.outcome is None
    assert [len(state.balls) for state in states] == [1, 1]
    assert all(
        state.balls[0].suck_count
        == simulator.config.loss_initial_suck_count
        for state in states
    )


def test_fresh_profile_tutorial_mode_fails_closed() -> None:
    with pytest.raises(NotImplementedError, match="fresh-profile tutorial"):
        RevengeSimulator(
            _line_curve(),
            profile_mode="fresh_profile",
        )


def test_installed_iron_frog_level_fails_closed_when_available() -> None:
    from zuma_rl.original_data import (
        OriginalDataError,
        find_original_installation,
    )

    try:
        root = find_original_installation()
    except OriginalDataError:
        pytest.skip("original Zuma's Revenge installation is unavailable")

    with pytest.raises(NotImplementedError, match="Iron Frog mode"):
        RevengeSimulator.from_installed("ironfrog1", root=root)


def test_native_fire_and_reload_cadence_is_six_plus_fifteen_ticks() -> None:
    simulator = _isolated_simulator(
        curve=_line_curve(y=100.0),
        shooter=(400.0, 500.0),
    )
    simulator.load_state(
        colors=[0],
        waypoints=[100.0],
        current_color=0,
        next_color=1,
    )
    simulator.stop_adding = True
    simulator.pending_colors.clear()

    assert simulator.request_fire(np.pi / 2)
    for _ in range(5):
        simulator.tick()
    assert simulator.gun_state is GunState.FIRING
    assert not simulator.free_projectiles

    simulator.tick()
    assert simulator.gun_state is GunState.RELOADING
    assert len(simulator.free_projectiles) == 1
    for _ in range(14):
        simulator.tick()
    assert simulator.gun_state is GunState.RELOADING
    simulator.tick()
    assert simulator.gun_state is GunState.NORMAL


def test_repeated_clicks_are_accepted_every_twenty_one_ticks() -> None:
    simulator = _isolated_simulator(
        curve=_line_curve(y=100.0),
        shooter=(400.0, 500.0),
    )
    simulator.load_state(
        colors=[0],
        waypoints=[100.0],
        current_color=0,
        next_color=1,
    )
    simulator.stop_adding = True
    simulator.pending_colors.clear()

    down_updates = set(range(0, 61, 3))
    accepted_updates: list[int] = []
    release_updates: list[int] = []
    for update in range(61):
        if update > 0:
            events = simulator.tick()
            if events.fired:
                release_updates.append(update)
        if update in down_updates and simulator.request_fire(0.0):
            accepted_updates.append(update)

    assert accepted_updates == [0, 21, 42]
    assert release_updates == [6, 27, 48]


def test_release_tick_matches_retail_48_pixel_forward_position() -> None:
    simulator = _isolated_simulator(
        curve=_line_curve(y=100.0),
        shooter=(400.0, 500.0),
    )
    simulator.load_state(
        colors=[0],
        waypoints=[100.0],
        current_color=0,
        next_color=1,
    )
    simulator.stop_adding = True
    simulator.pending_colors.clear()

    assert simulator.request_fire(0.0)
    simulator.tick(6)

    assert len(simulator.free_projectiles) == 1
    projectile = simulator.free_projectiles[0]
    # Retail PC memory shows 48 px forward and 2 px lateral after the release
    # update: 8 px muzzle offset, four in-frog updates, then one free update.
    np.testing.assert_array_equal(
        projectile.position,
        np.array((448.0, 502.0), dtype=np.float32),
    )


def test_pc_scoring_is_per_wave_and_chain_bonus_starts_on_clear_six() -> None:
    simulator = _isolated_simulator()
    simulator.score = 0
    simulator.consecutive_clears = 4
    simulator._score_match(3, combo_count=0, gap_bonus=0)
    assert simulator.score == 30

    simulator.consecutive_clears = 5
    simulator._score_match(3, combo_count=0, gap_bonus=0)
    assert simulator.score == 30 + 130

    simulator._score_match(3, combo_count=1, gap_bonus=0)
    assert simulator.score == 30 + 130 + 130


def test_match_scores_only_balls_newly_entering_explosion() -> None:
    simulator = _isolated_simulator()
    _load_isolated(
        simulator,
        colors=[1, 1, 1],
        waypoints=[100.0, 136.0, 172.0],
        contacts=[True, True],
    )
    simulator.balls[0].exploding = True
    simulator.balls[1].exploding = True

    assert simulator._check_set(2)

    assert simulator.score == 10
    assert simulator.last_events.balls_exploded == 1


def test_suck_pending_clear_counts_streak_before_scoring() -> None:
    simulator = _isolated_simulator()
    _load_isolated(
        simulator,
        colors=[2, 2, 2],
        waypoints=[100.0, 136.0, 172.0],
        contacts=[True, True],
    )
    simulator.balls[1].suck_pending = True
    simulator.consecutive_clears = 4

    assert simulator._check_set(1)

    assert simulator.consecutive_clears == 5
    assert simulator.score == 130
    assert not simulator.balls[1].suck_pending
    assert {ball.combo_score for ball in simulator.balls} == {130}


def test_global_zuma_rollback_moves_for_299_ticks() -> None:
    simulator = _isolated_simulator(curve=_line_curve(length=2_000))
    _load_isolated(
        simulator,
        colors=[0, 1],
        waypoints=[1_000.0, 1_036.0],
        contacts=[True],
    )
    simulator.backward_count = 300

    simulator.tick(299)
    assert simulator.balls[0].waypoint == pytest.approx(701.0)
    assert simulator.balls[1].waypoint == pytest.approx(737.0)
    before = [ball.waypoint for ball in simulator.balls]
    simulator.tick()
    assert [ball.waypoint for ball in simulator.balls] == pytest.approx(before)


def test_danger_slowdown_waits_for_cruising_speed() -> None:
    simulator = _isolated_simulator(curve=_line_curve(speed=1.0))
    _load_isolated(simulator, colors=[0], waypoints=[350.0])
    simulator.first_chain_end = 350

    assert simulator._target_chain_speed() == pytest.approx(1.0)

    simulator.has_reached_cruising_speed = True
    assert simulator._target_chain_speed() < 1.0


def test_danger_slowdown_reads_previous_tick_first_chain_end_cache() -> None:
    simulator = _isolated_simulator(curve=_line_curve(speed=0.5))
    _load_isolated(simulator, colors=[0], waypoints=[400.0])
    simulator.has_reached_cruising_speed = True
    simulator.advance_speed = np.float32(0.5)

    assert simulator.first_chain_end == 0
    assert simulator._target_chain_speed() == np.float32(0.5)

    simulator._advance_balls()

    assert simulator.first_chain_end == 400
    assert simulator._target_chain_speed() < np.float32(0.5)


def test_speed_ramp_down_does_not_clamp_and_rollout_waits_for_rollback() -> None:
    simulator = _isolated_simulator(curve=_line_curve(speed=0.5))
    _load_isolated(simulator, colors=[0], waypoints=[700.0])
    simulator.advance_speed = 0.55
    simulator.backward_count = 1

    simulator._advance_balls()

    assert simulator.advance_speed == pytest.approx(0.45)
    assert not simulator.has_reached_rollout

    simulator.backward_count = 0
    simulator._advance_balls()
    assert simulator.has_reached_rollout


def test_initial_roll_in_snaps_to_cruise_without_float32_overshoot() -> None:
    simulator = _isolated_simulator(curve=_line_curve(speed=0.5))
    _load_isolated(simulator, colors=[0], waypoints=[100.0])
    simulator.advance_speed = np.float32(0.5999561548233032)
    # The native latch flips on the preceding tick (0.699956 -> 0.599956),
    # before the final subtraction crosses the 0.5 cruise target.
    simulator.has_reached_cruising_speed = True

    simulator._advance_balls()

    assert simulator.advance_speed == np.float32(0.5)


def test_match_records_last_exploded_waypoint_for_powerup_manager() -> None:
    simulator = _isolated_simulator()
    _load_isolated(
        simulator,
        colors=[2, 2, 2],
        waypoints=[587.5, 623.5, 659.5],
        contacts=[True, True],
    )

    # Retail CurveMgr::ExplodeBall writes this field for every ball in the
    # match traversal.  A middle seed therefore cannot stand in for the final
    # ball visited by that traversal.
    assert simulator._check_set(1)

    assert simulator.last_powerup_waypoint == 659


def test_only_rear_ball_advances_without_contact() -> None:
    simulator = _isolated_simulator(curve=_line_curve(speed=0.5))
    _load_isolated(
        simulator,
        colors=[0, 1],
        waypoints=[100.0, 200.0],
        contacts=[False],
    )

    simulator.tick()

    assert simulator.balls[0].waypoint > 100.0
    assert simulator.balls[1].waypoint == pytest.approx(200.0)


def test_contact_pushes_forward_but_stops_at_a_gap() -> None:
    simulator = _isolated_simulator(curve=_line_curve(speed=0.5))
    _load_isolated(
        simulator,
        colors=[0, 1, 2],
        waypoints=[100.0, 136.0, 220.0],
        contacts=[True, False],
    )

    simulator.tick()

    expected_rear = np.add(
        np.float32(100.0),
        np.float32(0.005),
        dtype=np.float32,
    )
    expected_front = np.add(
        np.add(expected_rear, np.float32(18.0), dtype=np.float32),
        np.float32(18.0),
        dtype=np.float32,
    )
    assert simulator.balls[0].waypoint == expected_rear
    assert simulator.balls[1].waypoint == expected_front
    assert simulator.balls[2].waypoint == pytest.approx(220.0)
    assert simulator.balls[0].contact_next
    assert not simulator.balls[1].contact_next


def test_high_speed_add_ball_recurses_within_the_same_tick() -> None:
    simulator = _isolated_simulator()
    _load_isolated(simulator, colors=[0], waypoints=[40.0])
    simulator.stop_adding = False
    simulator.pending_colors = [1, 2, 3]
    simulator.advance_speed = 36.0

    simulator._add_ball()

    assert [ball.color for ball in simulator.balls] == [2, 1, 0]
    assert simulator.pending_colors == [3]
    assert simulator.balls[1].waypoint > 1.0
    assert simulator.balls[0].waypoint < 1.0


def test_exhausted_feed_plan_does_not_synthesize_pending_ball() -> None:
    simulator = _isolated_simulator()
    _load_isolated(simulator, colors=[0], waypoints=[100.0])
    simulator.stop_adding = False
    simulator.feed_plan_exhausted = True
    simulator.pending_colors.clear()
    rng_state = simulator.rng.state
    next_id = simulator._next_id

    simulator._add_ball()

    assert [ball.color for ball in simulator.balls] == [0]
    assert simulator.pending_colors == []
    assert simulator.rng.state == rng_state
    assert simulator._next_id == next_id


def test_exhausted_feed_plan_refills_after_live_pending_ball_is_consumed() -> None:
    simulator = _isolated_simulator()
    _load_isolated(simulator, colors=[0], waypoints=[100.0])
    simulator.stop_adding = False
    simulator.feed_plan_exhausted = True
    simulator.pending_colors = [1]

    simulator._add_ball()

    assert [ball.color for ball in simulator.balls] == [1, 0]
    assert simulator.pending_colors == []
    assert simulator.feed_refill_armed
    rng_state = simulator.rng.state

    simulator._add_ball()

    assert len(simulator.pending_colors) == 1
    assert simulator.rng.state != rng_state
    assert not simulator.feed_refill_armed


def test_match_does_not_cross_a_topological_gap() -> None:
    simulator = _isolated_simulator()
    _load_isolated(
        simulator,
        colors=[1, 1],
        waypoints=[100.0, 200.0],
        contacts=[False],
    )
    _attach(
        simulator,
        color=1,
        hit_index=0,
        hit_in_front=True,
    )

    for _ in range(21):
        simulator.tick()

    assert len(simulator.balls) == 3
    assert not any(ball.exploding for ball in simulator.balls)
    assert any(not ball.contact_next for ball in simulator.balls[:-1])


def test_exploding_match_is_removed_after_about_40_ticks() -> None:
    simulator = _isolated_simulator()
    _load_isolated(
        simulator,
        colors=[1, 1],
        waypoints=[100.0, 136.0],
        contacts=[True],
    )
    _attach(
        simulator,
        color=1,
        hit_index=0,
        hit_in_front=True,
    )
    for _ in range(21):
        simulator.tick()

    assert len(simulator.balls) == 3
    assert all(ball.exploding for ball in simulator.balls)

    ticks_after_match = 0
    while simulator.balls and ticks_after_match < 45:
        simulator.tick()
        ticks_after_match += 1

    assert not simulator.balls
    assert 39 <= ticks_after_match <= 41


def test_same_color_gap_boundary_starts_suckback() -> None:
    simulator = _isolated_simulator()
    _load_isolated(
        simulator,
        colors=[2, 1, 1, 2],
        waypoints=[100.0, 136.0, 172.0, 208.0],
        contacts=[True, True, True],
    )
    _attach(
        simulator,
        color=1,
        hit_index=1,
        hit_in_front=True,
    )
    for _ in range(21):
        simulator.tick()

    for _ in range(45):
        if len(simulator.balls) == 2:
            break
        simulator.tick()

    assert [ball.color for ball in simulator.balls] == [2, 2]
    assert not simulator.balls[0].contact_next
    assert simulator.balls[1].suck_count >= 10

    before = simulator.balls[1].waypoint
    simulator.tick()
    assert simulator.balls[1].waypoint == pytest.approx(before - 1.0)
    assert simulator.balls[1].suck_count >= 11
    second = simulator.balls[1].waypoint
    simulator.tick()
    assert simulator.balls[1].waypoint == pytest.approx(second - 1.0)
    assert simulator.balls[1].suck_count >= 12


def test_multiple_suckback_segments_advance_in_one_tick() -> None:
    simulator = _isolated_simulator()
    _load_isolated(
        simulator,
        colors=[0, 0, 1, 1],
        waypoints=[100.0, 150.0, 300.0, 350.0],
        contacts=[False, False, False],
    )
    simulator.balls[1].suck_count = 8
    simulator.balls[3].suck_count = 8

    simulator._update_sucking_balls()

    assert simulator.balls[1].waypoint == pytest.approx(149.0)
    assert simulator.balls[3].waypoint == pytest.approx(349.0)
    assert simulator.balls[1].suck_count == 9
    assert simulator.balls[3].suck_count == 9


def test_suckback_ramp_is_scaled_by_retail_reverse_speed() -> None:
    simulator = RevengeSimulator(
        _line_curve(),
        config=replace(RevengePhysicsConfig(), reverse_speed=0.5),
    )
    simulator.stop_adding = True
    simulator.pending_colors.clear()
    _load_isolated(
        simulator,
        colors=[1, 1],
        waypoints=[100.0, 200.0],
        contacts=[False],
    )
    simulator.balls[1].suck_count = 16

    simulator._update_sucking_balls()

    # (16 >> 3) * 0.5 = 1.0.  The previous implementation moved 2.0.
    assert simulator.balls[1].waypoint == pytest.approx(199.0)
    assert simulator.balls[1].suck_count == 17


def test_set_removal_explicitly_selects_backward_suck_direction() -> None:
    simulator = _isolated_simulator()
    _load_isolated(
        simulator,
        colors=[2, 1, 2],
        waypoints=[100.0, 150.0, 250.0],
        contacts=[False, False],
    )
    simulator.balls[1].should_remove = True
    simulator.balls[1].combo_count = 3
    simulator.balls[2].suck_back = False

    simulator._update_sets()

    assert [ball.color for ball in simulator.balls] == [2, 2]
    assert simulator.balls[1].suck_count == 10
    assert simulator.balls[1].suck_back
    assert simulator.balls[1].combo_count == 4


def test_suckback_skips_all_exploding_balls_after_same_color_boundary() -> None:
    simulator = _isolated_simulator()
    _load_isolated(
        simulator,
        colors=[2, 1, 2, 2],
        waypoints=[100.0, 120.0, 140.0, 170.0],
        contacts=[False, False, False],
    )
    simulator.balls[1].exploding = True
    simulator.balls[2].exploding = True
    simulator.balls[3].suck_count = 8

    simulator._update_sucking_balls()

    assert simulator.balls[3].waypoint == pytest.approx(169.0)
    assert simulator.balls[3].suck_count == 9


def test_clear_pending_sucks_preserves_unrelated_suck_counters() -> None:
    simulator = _isolated_simulator()
    _load_isolated(
        simulator,
        colors=[0, 1, 2],
        waypoints=[100.0, 180.0, 260.0],
        contacts=[False, False],
    )
    simulator.balls[0].suck_count = 7
    simulator.balls[2].suck_pending = True
    simulator.balls[2].gap_bonus = 500
    simulator.consecutive_clears = 6

    simulator._clear_pending_sucks(2)

    assert simulator.balls[0].suck_count == 7
    assert not simulator.balls[2].suck_pending
    assert simulator.balls[2].gap_bonus == 0
    assert simulator.consecutive_clears == 0


def test_projectile_moves_eight_pixels_and_tangent_is_not_a_hit() -> None:
    simulator = _isolated_simulator(
        curve=_line_curve(y=100.0),
        shooter=(0.0, 100.0),
    )
    _load_isolated(simulator, colors=[2], waypoints=[44.0])
    projectile = simulator.launch_projectile(0.0, color=1)
    start = projectile.position.copy()

    simulator.tick()

    np.testing.assert_allclose(projectile.position - start, (8.0, 0.0))
    assert projectile in simulator.free_projectiles
    assert not simulator.merging_projectiles

    simulator.tick()

    assert projectile not in simulator.free_projectiles
    assert projectile in simulator.merging_projectiles


def test_just_fired_projectile_collides_before_its_first_move() -> None:
    simulator = _isolated_simulator(
        curve=_line_curve(y=100.0),
        shooter=(100.0, 100.0),
    )
    _load_isolated(simulator, colors=[0], waypoints=[135.0])
    projectile = simulator.launch_projectile(np.pi, color=1)

    simulator.tick()

    assert projectile not in simulator.free_projectiles
    assert projectile in simulator.merging_projectiles
    assert simulator.last_events.hits == 1


def test_free_projectile_collision_advances_overlapping_merge() -> None:
    simulator = _isolated_simulator(
        curve=_line_curve(y=0.0),
        shooter=(100.0, 0.0),
    )
    _load_isolated(simulator, colors=[0], waypoints=[100.0])
    merging = _attach(
        simulator,
        color=1,
        hit_index=0,
        hit_in_front=True,
    )
    free = simulator.launch_projectile(0.0, color=2)

    assert not simulator._try_projectile_collision(free)

    assert merging.hit_percent == pytest.approx(0.05)


def test_merge_redirect_uses_first_update_position_and_is_one_shot() -> None:
    simulator = _isolated_simulator(
        curve=_line_curve(y=0.0),
        shooter=(136.0, 0.0),
    )
    _load_isolated(
        simulator,
        colors=[0, 1, 2],
        waypoints=[100.0, 136.0, 172.0],
        contacts=[False, False],
    )
    projectile = _attach(
        simulator,
        color=3,
        hit_index=2,
        hit_in_front=False,
    )
    original = projectile.position.copy()

    simulator._update_ball_objects()
    np.testing.assert_allclose(projectile.position, original * 0.975)
    simulator._redirect_merge_to_previous(projectile)

    assert projectile.hit_ball_id == simulator.balls[1].id
    assert projectile.have_set_prev_ball
    projectile.position = simulator.ball_position(simulator.balls[0])
    simulator._redirect_merge_to_previous(projectile)
    assert projectile.hit_ball_id == simulator.balls[1].id


def test_find_free_waypoint_scans_from_existing_ball_not_old_projectile() -> None:
    simulator = _isolated_simulator()
    existing_position = np.asarray(
        simulator._point_xy(100.875),
        dtype=np.float32,
    )

    waypoint = simulator._find_free_waypoint_from(
        existing_waypoint=100.875,
        existing_position=existing_position,
        existing_radius=18,
        new_waypoint=60.0,
        new_radius=18,
        in_front=False,
    )

    # 65 remains inside the 36 px radius sum; 64 is the first free integer
    # sample when scanning backward from the existing chain ball.
    assert waypoint == np.float32(64.0)


def test_discontinuous_curve_uses_new_merge_position_semantics() -> None:
    curve = _line_curve(length=400, y=0.0)
    curve.points[150, 0] += 100.0
    simulator = _isolated_simulator(curve=curve, shooter=(136.0, 0.0))
    _load_isolated(simulator, colors=[0], waypoints=[172.0])
    projectile = _attach(
        simulator,
        color=1,
        hit_index=0,
        hit_in_front=True,
    )
    original = projectile.position.copy()

    assert projectile.do_new_merge
    simulator._update_ball_objects()
    np.testing.assert_allclose(projectile.position, original)
    simulator._advance_one_merging_projectile(0)
    np.testing.assert_allclose(
        projectile.position,
        simulator._point_xy(projectile.waypoint),
    )


def test_merge_does_not_push_a_physically_separate_ball() -> None:
    simulator = _isolated_simulator(
        curve=_line_curve(y=0.0),
        shooter=(100.0, 0.0),
    )
    _load_isolated(
        simulator,
        colors=[0, 1],
        waypoints=[100.0, 105.0],
        contacts=[False],
    )
    projectile = _attach(
        simulator,
        color=2,
        hit_index=0,
        hit_in_front=True,
    )
    projectile.hit_percent = np.float32(0.5)
    projectile.position = np.array((500.0, 0.0), dtype=np.float32)
    before = simulator.balls[1].waypoint

    simulator._push_for_merge(projectile, 0)

    assert simulator.balls[1].waypoint == before


def test_merge_of_different_color_cancels_backward_suck() -> None:
    simulator = _isolated_simulator(
        curve=_line_curve(y=0.0),
        shooter=(110.0, 0.0),
    )
    _load_isolated(
        simulator,
        colors=[0, 2],
        waypoints=[100.0, 136.0],
        contacts=[False],
    )
    simulator.balls[1].suck_count = 8
    projectile = _attach(
        simulator,
        color=1,
        hit_index=0,
        hit_in_front=True,
    )
    projectile.hit_percent = np.float32(0.5)

    simulator._push_for_merge(projectile, 0)

    assert simulator.balls[1].suck_count == 0
    assert simulator.balls[1].suck_back


def test_suckback_repositions_merge_attached_to_the_moved_ball() -> None:
    simulator = _isolated_simulator(
        curve=_line_curve(y=0.0),
        shooter=(200.0, 0.0),
    )
    _load_isolated(
        simulator,
        colors=[0, 2],
        waypoints=[100.0, 200.0],
        contacts=[False],
    )
    projectile = _attach(
        simulator,
        color=1,
        hit_index=1,
        hit_in_front=False,
    )
    simulator.balls[1].suck_count = 8

    simulator._update_sucking_balls()

    assert projectile.waypoint == np.float32(163.0)
    np.testing.assert_array_equal(
        projectile.position,
        np.array((163.0, 0.0), dtype=np.float32),
    )
    assert projectile.hit_position is not None
    np.testing.assert_array_equal(
        projectile.hit_position,
        projectile.position,
    )


def test_gap_crossing_is_recorded_and_scores_on_insertion() -> None:
    simulator = _isolated_simulator(
        curve=_line_curve(y=100.0),
        shooter=(181.0, 100.0),
    )
    _load_isolated(
        simulator,
        colors=[0, 1, 2],
        waypoints=[100.0, 300.0, 336.0],
        contacts=[False, True],
    )
    projectile = simulator.launch_projectile(np.pi / 2, color=3)

    assert simulator._check_gap_shot(projectile)
    assert projectile.gap_info == [(simulator.balls[1].id, 200)]

    simulator.attach_projectile(
        projectile,
        hit_index=2,
        hit_in_front=False,
    )
    simulator._finish_merge(0)
    inserted = simulator.balls[2]

    assert inserted.color == 3
    assert inserted.gap_bonus == 270
    assert inserted.num_gaps == 1


def test_gap_curve_proximity_uses_radius_not_diameter_threshold() -> None:
    simulator = _isolated_simulator(curve=_line_curve(y=100.0))
    _load_isolated(
        simulator,
        colors=[0, 1, 2],
        waypoints=[100.0, 300.0, 336.0],
        contacts=[False, True],
    )
    projectile = Projectile(
        id=simulator._new_id(),
        color=3,
        position=np.array((181.0, 127.0), dtype=np.float32),
        velocity=np.zeros(2, dtype=np.float32),
        radius=18,
        just_fired=False,
    )

    # 27 px is inside the Deluxe reconstruction's 2r threshold but outside
    # the pinned Revenge runtime's r threshold.
    assert not simulator._check_gap_shot(projectile)
    assert projectile.gap_info == []

    projectile.position[1] = np.float32(117.0)
    assert simulator._check_gap_shot(projectile)
    assert projectile.gap_info == [(simulator.balls[1].id, 200)]


def test_float32_merge_completes_on_tick_21() -> None:
    simulator = _isolated_simulator()
    _load_isolated(simulator, colors=[0], waypoints=[100.0])
    projectile = _attach(
        simulator,
        color=1,
        hit_index=0,
        hit_in_front=True,
    )

    for _ in range(20):
        simulator.tick()

    assert projectile in simulator.merging_projectiles
    assert projectile.hit_percent == pytest.approx(
        float(np.float32(0.9999996)),
        abs=1e-7,
    )
    assert len(simulator.balls) == 1

    simulator.tick()

    assert projectile not in simulator.merging_projectiles
    assert len(simulator.balls) == 2
    assert all(isinstance(ball, ChainBall) for ball in simulator.balls)


def test_tunnel_on_the_hit_side_blocks_collision() -> None:
    plain = _isolated_simulator(
        curve=_line_curve(y=100.0),
        shooter=(200.0, 100.0),
    )
    tunneled = _isolated_simulator(
        curve=_line_curve(y=100.0, tunnel_waypoints=[118]),
        shooter=(200.0, 100.0),
    )
    for simulator in (plain, tunneled):
        _load_isolated(simulator, colors=[0], waypoints=[100.0])
        simulator.launch_projectile(np.pi, color=1)

    for _ in range(9):
        plain.tick()
        tunneled.tick()

    assert len(plain.merging_projectiles) == 1
    assert not plain.free_projectiles
    assert len(tunneled.free_projectiles) == 1
    assert not tunneled.merging_projectiles


def test_same_seed_replays_pending_colors_and_state() -> None:
    left = RevengeSimulator(
        _line_curve(speed=0.5),
        shooter=(400.0, 300.0),
        seed=2026,
    )
    right = RevengeSimulator(
        _line_curve(speed=0.5),
        shooter=(400.0, 300.0),
        seed=2026,
    )

    assert left.pending_colors == right.pending_colors
    assert len(left.pending_colors) == 10
    for _ in range(50):
        left.tick()
        right.tick()
        assert _snapshot(left) == _snapshot(right)


def test_shooter_palette_includes_free_and_merging_projectiles() -> None:
    simulator = _isolated_simulator(
        curve=_line_curve(y=0.0),
        shooter=(100.0, 0.0),
    )
    _load_isolated(
        simulator,
        colors=[0, 1],
        waypoints=[100.0, 200.0],
        contacts=[False],
    )
    for ball in simulator.balls:
        ball.exploding = True
    simulator.launch_projectile(0.0, color=2)
    merging = simulator.launch_projectile(0.0, color=3)
    simulator.attach_projectile(merging, hit_index=0, hit_in_front=True)

    assert set(simulator._present_colors()) == {2, 3}


def test_color_support_change_clears_qrand_hit_histories() -> None:
    simulator = _isolated_simulator()
    chooser = simulator._color_chooser
    chooser.choose([0, 1])
    chooser.last_hit[:] = (9, 8, 7, 6, 5, 4)
    chooser.previous_hit[:] = (5, 4, 3, 2, 1, 0)
    old_update_count = chooser.update_count

    assert chooser.choose([1]) == 1

    assert chooser.update_count == old_update_count + 1
    assert chooser.last_hit[0] == 0
    assert chooser.last_hit[1] == chooser.update_count
    assert tuple(chooser.previous_hit) == (0, 0, 0, 0, 0, 0)


def test_qrand_zero_needle_cannot_select_absent_colour() -> None:
    simulator = _isolated_simulator()
    chooser = simulator._color_chooser
    # The next MSVC CRT output is exactly zero.  The previous all-slot scan
    # incorrectly selected colour 0 even though only colour 1 had weight.
    chooser.rng = MsvcCRTRandom(20_057)

    assert chooser.choose([1]) == 1
    assert chooser.allowed_support == (1,)
    assert chooser.selected_index == 1
    assert tuple(map(float, chooser.sways)) == (
        0.0,
        0.875,
        0.0,
        0.0,
        0.0,
        0.0,
    )


def test_msvc_crt_rand_matches_retail_state_transition() -> None:
    rng = MsvcCRTRandom(1_674_403_832)

    assert rng.next_u15() == 548
    assert rng.state == 2_183_421_659


def test_popcap_mtrand_matches_retail_tempered_outputs() -> None:
    rng = PopCapMTRandom(5489)

    assert [rng.next_u31() for _ in range(4)] == [
        1_351_727_964,
        581_869_302,
        1_742_863_086,
        1_438_850_937,
    ]
    assert rng.index == 4


def test_pending_colour_consumes_retail_visual_frame_draw_after_choice() -> None:
    class ScriptedMTRandom:
        def __init__(self) -> None:
            self.outputs = iter((71, 8, 123_456))
            self.calls: list[tuple[str, int | None, int]] = []

        def rand_mod(self, modulus: int) -> int:
            raw = next(self.outputs)
            self.calls.append(("mod", modulus, raw))
            return raw % modulus

        def next_u31(self) -> int:
            raw = next(self.outputs)
            self.calls.append(("raw", None, raw))
            return raw

    simulator = _isolated_simulator()
    simulator.balls = [
        ChainBall(
            id=1,
            color=2,
            waypoint=np.float32(0.0),
            radius=18,
        )
    ]
    scripted = ScriptedMTRandom()
    simulator.rng = scripted  # type: ignore[assignment]

    assert simulator._append_pending_color() == 0
    assert simulator.pending_colors == [0]
    assert scripted.calls == [
        ("mod", 100, 71),
        ("mod", 4, 8),
        ("raw", None, 123_456),
    ]


def test_pending_colour_repeat_roll_respects_revenge_max_clump() -> None:
    class ScriptedMTRandom:
        def __init__(self) -> None:
            self.outputs = iter((0, 6, 5, 123_456))
            self.calls: list[tuple[str, int | None, int]] = []

        def rand_mod(self, modulus: int) -> int:
            raw = next(self.outputs)
            self.calls.append(("mod", modulus, raw))
            return raw % modulus

        def next_u31(self) -> int:
            raw = next(self.outputs)
            self.calls.append(("raw", None, raw))
            return raw

    curve = _line_curve()
    curve.parameters = replace(
        curve.parameters,
        ball_repeat_chance=100,
        max_clump_size=2,
    )
    simulator = _isolated_simulator(curve=curve)
    simulator.balls = [
        ChainBall(
            id=index,
            color=2,
            waypoint=np.float32(index * 36.0),
            radius=18,
        )
        for index in (1, 2)
    ]
    scripted = ScriptedMTRandom()
    simulator.rng = scripted  # type: ignore[assignment]

    assert simulator._append_pending_color() == 1
    assert simulator.pending_colors == [1]
    assert scripted.calls == [
        ("mod", 100, 0),
        ("mod", 4, 6),
        ("mod", 4, 5),
        ("raw", None, 123_456),
    ]


def test_qrand_replays_captured_retail_shooter_choice_exactly() -> None:
    simulator = _isolated_simulator()
    simulator.load_shooter_random_state(
        crt_state=1_674_403_832,
        update_count=4,
        selected_index=2,
        weights=(0.25, 0.25, 0.25, 0.25, 0.0, 0.0),
        sways=(
            0.125,
            0.21875,
            0.171875,
            0.0832500010728836,
            0.0,
            0.0,
        ),
        last_hit=(2, 0, 4, 3, 0, 0),
        previous_hit=(0, 0, 1, 0, 0, 0),
    )

    assert simulator._color_chooser.choose([0, 1, 2, 3]) == 0

    chooser = simulator._color_chooser
    assert simulator.crt_rng.state == 2_183_421_659
    assert chooser.update_count == 5
    assert chooser.selected_index == 0
    assert tuple(map(float, chooser.weights)) == (
        0.25,
        0.25,
        0.25,
        0.25,
        0.0,
        0.0,
    )
    assert tuple(map(float, chooser.sways)) == (
        0.1796875,
        0.2734375,
        0.0832500010728836,
        0.1328125,
        0.0,
        0.0,
    )
    assert tuple(map(int, chooser.last_hit)) == (5, 0, 4, 3, 0, 0)
    assert tuple(map(int, chooser.previous_hit)) == (
        2,
        0,
        1,
        0,
        0,
        0,
    )


def test_qrand_restores_c158_uninitialized_sentinel_and_skull_reset() -> None:
    simulator = _isolated_simulator()
    simulator.load_state(
        colors=[0, 1, 2, 3],
        waypoints=[100.0, 136.0, 172.0, 208.0],
        current_color=3,
        next_color=3,
    )
    simulator.load_shooter_random_state(
        crt_state=1_020_014_204,
        update_count=0,
        selected_index=-1,
        weights=(),
        sways=(),
        last_hit=(),
        previous_hit=(),
    )

    chooser = simulator._color_chooser
    assert simulator.crt_rng.state == 1_020_014_204
    assert chooser.update_count == 0
    assert chooser.selected_index == -1
    assert chooser.allowed_support is None
    assert tuple(map(float, chooser.weights)) == (0.0,) * 6
    assert tuple(map(float, chooser.sways)) == (0.0,) * 6
    assert tuple(map(int, chooser.last_hit)) == (0,) * 6
    assert tuple(map(int, chooser.previous_hit)) == (0,) * 6

    simulator._reset_shooter_for_skull_entry()

    assert (simulator.current_color, simulator.next_color) == (0, 2)
    assert simulator.crt_rng.state == 3_527_905_494
    assert chooser.update_count == 2
    assert chooser.selected_index == 2
    assert chooser.allowed_support == (0, 1, 2, 3)
    assert tuple(map(float, chooser.weights)) == (
        0.25,
        0.25,
        0.25,
        0.25,
        0.0,
        0.0,
    )
    assert tuple(map(float, chooser.sways)) == (
        0.0832500010728836,
        0.109375,
        0.109375,
        0.109375,
        0.0,
        0.0,
    )
    assert tuple(map(int, chooser.last_hit)) == (1, 0, 2, 0, 0, 0)
    assert tuple(map(int, chooser.previous_hit)) == (0, 0, 0, 0, 0, 0)


@pytest.mark.parametrize(
    ("update_count", "selected_index"),
    ((1, -1), (0, 0)),
)
def test_qrand_rejects_noncanonical_empty_native_state(
    update_count: int,
    selected_index: int,
) -> None:
    simulator = _isolated_simulator()

    with pytest.raises(ValueError, match="uninitialized state is inconsistent"):
        simulator.load_shooter_random_state(
            crt_state=1_020_014_204,
            update_count=update_count,
            selected_index=selected_index,
            weights=(),
            sways=(),
            last_hit=(),
            previous_hit=(),
        )


def test_qrand_rejects_partially_empty_native_state() -> None:
    simulator = _isolated_simulator()

    with pytest.raises(ValueError, match="state vectors have the wrong length"):
        simulator.load_shooter_random_state(
            crt_state=1_020_014_204,
            update_count=0,
            selected_index=-1,
            weights=(),
            sways=(0.0,) * 6,
            last_hit=(),
            previous_hit=(),
        )


def test_zuma_bar_delays_stopping_generation_for_330_ticks() -> None:
    simulator = RevengeSimulator(
        _line_curve(speed=0.0, zuma_score=100),
        shooter=(400.0, 300.0),
        seed=3,
    )
    simulator.load_state(colors=[0], waypoints=[100.0])
    simulator.score = 100

    # This first UI update observes the newly awarded score and updates the
    # target bar only at its end, matching the original one-tick phase order.
    simulator.tick()
    for _ in range(329):
        simulator.tick()

    assert not simulator.zuma_reached
    assert not simulator.stop_adding
    assert simulator.pending_colors

    simulator.tick()

    assert simulator.zuma_reached
    assert simulator.stop_adding
    assert not simulator.pending_colors


def test_score_target_adds_this_level_quota_to_entry_score() -> None:
    simulator = RevengeSimulator(
        _line_curve(speed=0.0, zuma_score=1_700),
        shooter=(400.0, 300.0),
        seed=3,
    )
    simulator.load_state(
        colors=[0],
        waypoints=[100.0],
        score=7_950,
        score_at_level_start=7_950,
    )

    assert simulator.score_target == 9_650
    assert not simulator.score_achieved


def test_zuma_ui_clear_cannot_win_until_next_tick() -> None:
    simulator = RevengeSimulator(
        _line_curve(speed=0.0, zuma_score=100),
        shooter=(400.0, 300.0),
        seed=3,
    )
    simulator.load_state(
        colors=[0],
        waypoints=[1.0],
        pending_colors=[1],
        score=100,
    )
    simulator.balls[0].exploding = True
    simulator.balls[0].should_remove = True
    simulator.current_bar_size = simulator.config.zuma_bar_width
    simulator.target_bar_size = simulator.config.zuma_bar_width

    simulator.tick()

    assert simulator.zuma_reached
    assert not simulator.balls
    assert not simulator.pending_colors
    assert simulator.outcome is None

    simulator.tick()
    assert simulator.outcome == "win"


def test_nonlethal_curve_uses_two_phase_victory_instead_of_losing() -> None:
    simulator = _isolated_simulator(
        curve=_line_curve(length=300, die_at_end=False),
    )
    _load_isolated(simulator, colors=[0], waypoints=[300.0])

    transition = simulator.tick()

    assert not simulator.balls
    assert transition.win_pending
    assert simulator.win_pending
    assert simulator.outcome is None

    simulator.tick()

    assert not simulator.win_pending
    assert simulator.outcome == "win"


def test_final_removal_locks_input_during_retail_victory_transition() -> None:
    simulator = RevengeSimulator(
        _line_curve(speed=0.0),
        shooter=(400.0, 300.0),
        seed=23,
    )
    simulator.load_state(
        colors=[0],
        waypoints=[100.0],
        current_color=0,
        next_color=1,
    )
    simulator.balls[0].should_remove = True
    simulator.stop_adding = True

    transition = simulator.tick()

    assert transition.win_pending
    assert simulator.win_pending
    assert simulator.outcome is None
    assert simulator.current_color is None
    assert simulator.next_color is None
    assert not simulator.request_fire(0.0)
    assert not simulator.swap_balls()

    outcome = simulator.tick()

    assert not simulator.win_pending
    assert simulator.outcome == "win"
    assert outcome.outcome == "win"
    assert outcome.score_delta == 100


def test_final_free_projectile_exit_clears_chamber_and_starts_win() -> None:
    simulator = RevengeSimulator(
        _line_curve(speed=0.0),
        shooter=(400.0, 300.0),
        seed=29,
    )
    simulator.load_state(
        colors=[],
        waypoints=[],
        current_color=1,
        next_color=1,
        score=120,
    )
    simulator.stop_adding = True
    simulator.free_projectiles.append(
        Projectile(
            id=225,
            color=1,
            position=np.asarray((89.8501, 1.8840), dtype=np.float32),
            velocity=np.asarray((-5.6072, -5.7060), dtype=np.float32),
            radius=18,
            just_fired=False,
        )
    )

    transition = simulator.tick()

    assert not simulator.free_projectiles
    assert transition.balls_removed == 0
    assert transition.win_pending
    assert simulator.win_pending
    assert simulator.current_color is None
    assert simulator.next_color is None

    outcome = simulator.tick()

    assert outcome.outcome == "win"
    assert simulator.outcome == "win"
    assert outcome.score_delta == 100
    assert simulator.score == 220


def test_lethal_curve_uses_two_phase_input_locked_loss_sequence() -> None:
    simulator = RevengeSimulator(
        _line_curve(length=100, speed=0.0),
        shooter=(50.0, 50.0),
        seed=17,
    )
    simulator.load_state(
        colors=[0, 1, 2],
        waypoints=[28.0, 64.0, 100.0],
        contacts=[True, False],
        pending_colors=[3],
        current_color=0,
        next_color=1,
        score=80,
        score_at_level_start=80,
    )
    update_counts_before = [ball.update_count for ball in simulator.balls]
    chooser_updates_before = simulator._color_chooser.update_count

    transition = simulator.tick()

    assert not transition.loss_started
    assert simulator.skull_entry_pending
    assert not simulator.loss_started
    assert simulator.loss_elapsed_ticks == 0
    assert simulator.native_game_time == 0
    assert simulator.outcome is None
    assert len(simulator.balls) == 3
    assert [ball.update_count for ball in simulator.balls] == [
        value + 1 for value in update_counts_before
    ]
    assert simulator._color_chooser.update_count == chooser_updates_before + 2
    assert simulator.current_color is not None
    assert simulator.next_color is not None

    trigger = simulator.tick()

    assert trigger.loss_started
    assert trigger.balls_removed == 1
    assert not simulator.skull_entry_pending
    assert simulator.loss_started
    assert simulator.loss_elapsed_ticks == 1
    assert simulator.native_game_time == 1
    assert simulator.outcome is None
    assert [float(ball.waypoint) for ball in simulator.balls] == [28.0, 64.0]
    assert [ball.suck_count for ball in simulator.balls] == [1, 1]
    assert [ball.update_count for ball in simulator.balls] == [1, 1]
    assert simulator.pending_colors == [3]
    assert simulator.score == 80

    # Loss locking is independent of the current animation state.
    simulator.gun_state = GunState.NORMAL
    simulator.gun_state_percent = np.float32(1.0)
    simulator.current_color = 0
    simulator.next_color = 1
    assert not simulator.request_fire(0.0)
    assert not simulator.swap_balls()

    simulator.tick()

    assert simulator.loss_elapsed_ticks == 2
    assert [float(ball.waypoint) for ball in simulator.balls] == [28.0, 64.0]
    assert [ball.suck_count for ball in simulator.balls] == [2, 2]
    assert [ball.update_count for ball in simulator.balls] == [1, 1]


def test_loss_suction_matches_jungle2_pc_removal_timing() -> None:
    simulator = RevengeSimulator(
        _line_curve(length=3_757, speed=0.125, zuma_score=1_700),
        shooter=(400.0, 300.0),
        seed=29,
    )
    survivor_waypoints = [
        31.875 + 36.0 * index
        for index in range(98)
    ]
    simulator.load_state(
        colors=[index % 4 for index in range(99)],
        waypoints=[*survivor_waypoints, 3_757.0],
        contacts=[True] * 97 + [False],
        pending_colors=[2],
        current_color=0,
        next_color=1,
        score=8_120,
        score_at_level_start=7_950,
    )
    simulator.advance_speed = np.float32(0.125)
    simulator.first_chain_end = int(survivor_waypoints[-1])

    simulator.tick()

    assert simulator.tick_count == 1
    assert simulator.skull_entry_pending
    np.testing.assert_array_equal(
        [ball.waypoint for ball in simulator.balls[:-1]],
        np.asarray(survivor_waypoints, dtype=np.float32) + np.float32(0.125),
    )
    assert simulator.balls[-1].waypoint == np.float32(3_757.0)

    removals: dict[int, int] = {}
    trigger = simulator.tick()
    removals[simulator.tick_count] = trigger.balls_removed
    assert trigger.loss_started
    assert len(simulator.balls) == 98

    while simulator.outcome is None:
        events = simulator.tick()
        if events.balls_removed:
            removals[simulator.tick_count] = events.balls_removed
        assert all(ball.update_count == 1 for ball in simulator.balls)

    # PC oracle mapping: synthetic u11190 -> simulator tick 0.
    assert simulator.tick_count == 177
    assert simulator.loss_elapsed_ticks == 176
    assert removals[2] == 1
    assert removals[48] == 1
    assert removals[177] == 1
    assert sum(removals.values()) == 99
    assert not simulator.balls
    assert simulator.pending_colors == [2]
    assert simulator.score == 8_120
    assert simulator.outcome == "loss"


def test_empty_chain_waits_for_in_flight_projectile_before_win() -> None:
    simulator = _isolated_simulator(
        curve=_line_curve(y=300.0),
        shooter=(400.0, 300.0),
    )
    _load_isolated(simulator, colors=[], waypoints=[])
    simulator.launch_projectile(0.0, color=1)

    simulator.tick()

    assert simulator.outcome is None
    assert simulator.free_projectiles

    for _ in range(100):
        simulator.tick()
        if simulator.outcome is not None:
            break

    assert not simulator.free_projectiles
    assert simulator.outcome == "win"


def test_entrance_recycling_deletes_projectile_attached_to_removed_ball() -> None:
    simulator = _isolated_simulator()
    _load_isolated(simulator, colors=[0], waypoints=[-4.0])
    simulator.has_reached_cruising_speed = True
    projectile = _attach(
        simulator,
        color=1,
        hit_index=0,
        hit_in_front=True,
    )

    assert projectile in simulator.merging_projectiles
    simulator._remove_front_for_rollout()

    assert not simulator.balls
    assert not simulator.merging_projectiles


def test_state_signature_covers_rng_and_hidden_future_state() -> None:
    simulator = _isolated_simulator(
        curve=_line_curve(y=0.0),
        shooter=(100.0, 0.0),
    )
    _load_isolated(simulator, colors=[0], waypoints=[100.0])
    projectile = _attach(
        simulator,
        color=1,
        hit_index=0,
        hit_in_front=True,
    )
    baseline = simulator.state_signature()

    simulator.feed_plan_exhausted = True
    assert simulator.state_signature() != baseline
    simulator.feed_plan_exhausted = False

    simulator.first_chain_end = 101
    assert simulator.state_signature() != baseline
    simulator.first_chain_end = 0

    simulator.current_acceleration = np.float32(0.25)
    assert simulator.state_signature() != baseline
    simulator.current_acceleration = np.float32(0.0)

    simulator.rng.random()
    rng_signature = simulator.state_signature()
    assert rng_signature != baseline

    projectile.target_position = projectile.target_position.copy()
    projectile.target_position[0] += 1.0
    assert simulator.state_signature() != rng_signature


def test_load_state_clears_prior_episode_control_state() -> None:
    simulator = _isolated_simulator()
    simulator.tick_count = 99
    simulator.stop_adding = True
    simulator.feed_plan_exhausted = True
    simulator.has_reached_rollout = True
    simulator.has_reached_cruising_speed = True
    simulator.consecutive_clears = 8
    simulator._color_chooser.choose([0, 1])

    simulator.load_state(colors=[0], waypoints=[100.0])

    assert simulator.tick_count == 0
    assert not simulator.stop_adding
    assert not simulator.feed_plan_exhausted
    assert not simulator.has_reached_rollout
    assert not simulator.has_reached_cruising_speed
    assert simulator.first_chain_end == 0
    assert simulator.consecutive_clears == 0
    assert simulator._color_chooser.update_count > 0
