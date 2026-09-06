from __future__ import annotations

from dataclasses import replace

import pytest

from zuma_rl.pc_memory_trajectory import (
    TrajectoryCurvePowerupState,
    TrajectoryEntity,
    TrajectoryFrame,
)
from zuma_rl.pc_powerup_trigger import (
    PcPowerupTriggerError,
    verify_powerup_trigger_frames,
)


def _state(
    *,
    native_time: int,
    triggered: bool,
    reverse_ticks: int,
) -> TrajectoryCurvePowerupState:
    cooldowns = [-1000] * 14
    counters = [0] * 14
    active_colors = [0] * 6
    active_colors[3] = 0 if triggered else 1
    if triggered:
        cooldowns[3] = 1002
        counters[3] = 1
    return TrajectoryCurvePowerupState(
        curve_index=0,
        last_any_spawn_time=800,
        last_spawn_times=(-1000,) * 14,
        cooldown_times=tuple(cooldowns),
        spawn_counts=(0,) * 14,
        field_124_by_type=tuple(counters),
        active_color_counts=tuple(active_colors),
        reverse_speed=1.0 if triggered else 0.5,
        slow_ticks=0,
        reverse_ticks=reverse_ticks,
        last_powerup_waypoint=0,
        powerup_triggered=triggered,
    )


def _ball(
    ball_id: int,
    color_id: int,
    distance: float,
    *,
    exploding: bool,
    explode_frame: int = 0,
    lifetime: int = 0,
    reverse: bool = False,
) -> TrajectoryEntity:
    return TrajectoryEntity(
        ball_id=ball_id,
        color_id=color_id,
        object_kind="ball",
        zone="curve:0:list:05c",
        index=ball_id,
        curve_distance=distance,
        position_x=distance,
        position_y=20.0,
        radius=18.0,
        exploding=exploding,
        explode_frame=explode_frame,
        backwards_count=0,
        backwards_speed=1.0 if distance >= 3_600.0 else 0.0,
        combo_count=0,
        combo_score=0,
        powerup_previous_type=14,
        powerup_primary_type=3 if reverse else 14,
        powerup_secondary_type=14,
        powerup_previous_ticks=0,
        powerup_lifetime_ticks=lifetime,
        powerup_transition_ticks=0,
        powerup_visual_scale=1.0,
        powerup_visual_step=0.0,
        powerup_visual_index=-1,
    )


def _frame(
    update: int,
    *,
    score: int,
    reverse_ticks: int,
    triggered: bool,
    include_exploding: bool,
) -> TrajectoryFrame:
    delta = max(update - 101, 0)
    reverse_reference_distance = {
        100: 3_601.0,
        101: 3_602.0,
        102: 3_603.0,
        103: 3_602.0,
        104: 3_601.0,
    }[update]
    entities: list[TrajectoryEntity] = [
        _ball(
            176,
            1,
            reverse_reference_distance,
            exploding=False,
        )
    ]
    if include_exploding:
        lifetime = 446 if update < 102 else 445
        entities.extend(
            (
                _ball(28, 3, 3_207.0 - delta, exploding=update >= 102),
                _ball(
                    29,
                    3,
                    3_171.0 - delta,
                    exploding=update >= 102,
                    lifetime=lifetime,
                    reverse=True,
                ),
                _ball(30, 3, 3_099.0 - delta, exploding=update >= 102),
            )
        )
        if update >= 102:
            entities.append(
                _ball(
                    128,
                    3,
                    3_134.0 - max(update - 102, 0),
                    exploding=True,
                    explode_frame=max(update - 101, 1),
                )
            )
    return TrajectoryFrame(
        update=update,
        score=score,
        displayed_score=score,
        score_target=10_000,
        current_ball_id=119,
        current_color_id=3,
        next_ball_id=122,
        next_color_id=1,
        list_counts=((0, 0x5C, len(entities)),),
        entities=tuple(entities),
        native_game_time=1_000 + update - 100,
        curve_powerups=(
            _state(
                native_time=1_000 + update - 100,
                triggered=triggered,
                reverse_ticks=reverse_ticks,
            ),
        ),
    )


def _frames() -> tuple[TrajectoryFrame, ...]:
    return (
        _frame(
            100,
            score=8_120,
            reverse_ticks=0,
            triggered=False,
            include_exploding=True,
        ),
        _frame(
            101,
            score=8_120,
            reverse_ticks=0,
            triggered=False,
            include_exploding=True,
        ),
        _frame(
            102,
            score=8_160,
            reverse_ticks=300,
            triggered=True,
            include_exploding=True,
        ),
        _frame(
            103,
            score=8_160,
            reverse_ticks=299,
            triggered=True,
            include_exploding=True,
        ),
        _frame(
            104,
            score=8_160,
            reverse_ticks=298,
            triggered=True,
            include_exploding=False,
        ),
    )


def test_trigger_verifier_closes_reverse_timing_and_bookkeeping() -> None:
    report = verify_powerup_trigger_frames(
        _frames(),
        curve_index=0,
        trigger_ball_id=29,
        trigger_color_id=3,
        powerup_type=3,
        expected_newly_exploding_ids=(28, 29, 30, 128),
        expected_score_delta=40,
        movement_reference_ball_id=176,
        expected_trigger_update=102,
    )

    assert report["status"] == "PASS"
    assert report["trigger"]["native_game_time"] == 1002
    assert report["trigger"]["target_lifetime_before"] == 446
    assert report["trigger"]["target_lifetime_after"] == 445
    assert report["reverse"]["ticks_at_trigger"] == 300
    assert report["reverse"]["ticks_next_update"] == 299
    assert report["reverse"]["movement"]["sample_phase"] == (
        "two_complete_post_trigger_intervals"
    )
    assert report["reverse"]["movement"]["trigger_distance"] == 3603.0
    assert report["reverse"]["movement"]["post_next_distance"] == 3601.0


def test_trigger_verifier_rejects_missing_second_post_trigger_reverse_step() -> None:
    frames = list(_frames())
    entities = tuple(
        replace(entity, curve_distance=3_602.0)
        if entity.ball_id == 176
        else entity
        for entity in frames[4].entities
    )
    frames[4] = replace(frames[4], entities=entities)

    with pytest.raises(
        PcPowerupTriggerError,
        match="powerup_trigger_post_trigger_reverse_movement_mismatch",
    ):
        verify_powerup_trigger_frames(
            tuple(frames),
            curve_index=0,
            trigger_ball_id=29,
            trigger_color_id=3,
            powerup_type=3,
            expected_newly_exploding_ids=(28, 29, 30, 128),
            expected_score_delta=40,
            movement_reference_ball_id=176,
            expected_trigger_update=102,
        )


def test_trigger_verifier_rejects_off_by_one_reverse_start() -> None:
    frames = list(_frames())
    state = frames[2].curve_powerups[0]
    frames[2] = replace(
        frames[2],
        curve_powerups=(replace(state, reverse_ticks=299),),
    )

    with pytest.raises(
        PcPowerupTriggerError,
        match="powerup_trigger_reverse_initial_state_mismatch",
    ):
        verify_powerup_trigger_frames(
            tuple(frames),
            curve_index=0,
            trigger_ball_id=29,
            trigger_color_id=3,
            powerup_type=3,
            expected_newly_exploding_ids=(28, 29, 30, 128),
            expected_score_delta=40,
            movement_reference_ball_id=176,
            expected_trigger_update=102,
        )


def test_trigger_verifier_rejects_late_powerup_explosion_sample() -> None:
    frames = list(_frames())
    frames[2] = replace(
        frames[2],
        entities=tuple(
            replace(entity, explode_frame=2)
            if entity.ball_id == 29
            else entity
            for entity in frames[2].entities
        ),
    )

    with pytest.raises(
        PcPowerupTriggerError,
        match="powerup_trigger_target_transition_invalid",
    ):
        verify_powerup_trigger_frames(
            tuple(frames),
            curve_index=0,
            trigger_ball_id=29,
            trigger_color_id=3,
            powerup_type=3,
            expected_newly_exploding_ids=(28, 29, 30, 128),
            expected_score_delta=40,
            movement_reference_ball_id=176,
            expected_trigger_update=102,
        )


def _bomb_state(
    *,
    triggered: bool,
) -> TrajectoryCurvePowerupState:
    cooldowns = [-1000] * 14
    counters = [0] * 14
    active_colors = [0] * 6
    active_colors[3] = 0 if triggered else 1
    if triggered:
        cooldowns[0] = 1002
        counters[0] = 1
    return TrajectoryCurvePowerupState(
        curve_index=0,
        last_any_spawn_time=800,
        last_spawn_times=(-1000,) * 14,
        cooldown_times=tuple(cooldowns),
        spawn_counts=(0,) * 14,
        field_124_by_type=tuple(counters),
        active_color_counts=tuple(active_colors),
        reverse_speed=1.0,
        slow_ticks=0,
        reverse_ticks=0,
        last_powerup_waypoint=0,
        powerup_triggered=triggered,
    )


def _bomb_frames() -> tuple[TrajectoryFrame, ...]:
    result: list[TrajectoryFrame] = []
    for frame in _frames():
        triggered = frame.update >= 102
        entities = [
            replace(entity, powerup_primary_type=0)
            if entity.ball_id == 29
            else entity
            for entity in frame.entities
        ]
        target_x = 3_171.0 - max(frame.update - 101, 0)
        if frame.update < 104:
            entities.append(
                _ball(
                    200,
                    2,
                    target_x + 144.0,
                    exploding=triggered,
                    explode_frame=max(frame.update - 102, 0),
                )
            )
        entities.append(
            _ball(
                201,
                2,
                target_x + 148.0,
                exploding=False,
            )
        )
        result.append(
            replace(
                frame,
                score=8_170 if triggered else 8_120,
                displayed_score=8_170 if triggered else 8_120,
                list_counts=((0, 0x5C, len(entities)),),
                entities=tuple(entities),
                curve_powerups=(
                    _bomb_state(triggered=triggered),
                ),
            )
        )
    return tuple(result)


def test_trigger_verifier_uses_strict_bomb_geometry() -> None:
    report = verify_powerup_trigger_frames(
        _bomb_frames(),
        curve_index=0,
        trigger_ball_id=29,
        trigger_color_id=3,
        powerup_type=0,
        expected_newly_exploding_ids=(28, 29, 30, 128, 200),
        expected_direct_match_ids=(28, 29, 30, 128),
        expected_score_delta=50,
        expected_trigger_update=102,
    )

    geometry = report["bomb_geometry"]
    assert geometry["retail_radius_for_18px_balls"] == 148
    assert geometry["farthest_selected"]["ball_id"] == 200
    assert geometry["farthest_selected"]["distance"] == 144.0
    assert geometry["nearest_rejected"]["ball_id"] == 201
    assert geometry["nearest_rejected"]["distance"] == 148.0


def test_trigger_verifier_accepts_exact_bound_concurrent_score() -> None:
    frames = tuple(
        replace(
            frame,
            score=frame.score + 500,
            displayed_score=frame.displayed_score + 500,
        )
        if frame.update >= 102
        else frame
        for frame in _bomb_frames()
    )

    report = verify_powerup_trigger_frames(
        frames,
        curve_index=0,
        trigger_ball_id=29,
        trigger_color_id=3,
        powerup_type=0,
        expected_newly_exploding_ids=(28, 29, 30, 128, 200),
        expected_direct_match_ids=(28, 29, 30, 128),
        expected_score_delta=550,
        expected_concurrent_score_delta=500,
        expected_trigger_update=102,
    )

    assert report["trigger"]["score_delta"] == 550
    assert report["trigger"]["base_explosion_score_delta"] == 50
    assert report["trigger"]["concurrent_score_delta"] == 500


def test_trigger_verifier_rejects_inexact_bound_concurrent_score() -> None:
    frames = tuple(
        replace(
            frame,
            score=frame.score + 499,
            displayed_score=frame.displayed_score + 499,
        )
        if frame.update >= 102
        else frame
        for frame in _bomb_frames()
    )

    with pytest.raises(
        PcPowerupTriggerError,
        match="powerup_trigger_score_delta_mismatch",
    ):
        verify_powerup_trigger_frames(
            frames,
            curve_index=0,
            trigger_ball_id=29,
            trigger_color_id=3,
            powerup_type=0,
            expected_newly_exploding_ids=(28, 29, 30, 128, 200),
            expected_direct_match_ids=(28, 29, 30, 128),
            expected_score_delta=549,
            expected_concurrent_score_delta=500,
            expected_trigger_update=102,
        )


def test_non_timer_powerup_allows_existing_effect_timers_to_decay() -> None:
    frames = []
    for frame in _bomb_frames():
        elapsed = frame.update - 100
        state = frame.curve_powerups[0]
        frames.append(
            replace(
                frame,
                curve_powerups=(
                    replace(
                        state,
                        reverse_ticks=102 - elapsed,
                        slow_ticks=902 - elapsed,
                    ),
                ),
            )
        )

    report = verify_powerup_trigger_frames(
        tuple(frames),
        curve_index=0,
        trigger_ball_id=29,
        trigger_color_id=3,
        powerup_type=0,
        expected_newly_exploding_ids=(28, 29, 30, 128, 200),
        expected_direct_match_ids=(28, 29, 30, 128),
        expected_score_delta=50,
        expected_trigger_update=102,
    )

    assert report["status"] == "PASS"


def _slow_state(
    *,
    triggered: bool,
    slow_ticks: int,
) -> TrajectoryCurvePowerupState:
    cooldowns = [-1000] * 14
    counters = [0] * 14
    active_colors = [0] * 6
    active_colors[3] = 0 if triggered else 1
    if triggered:
        cooldowns[1] = 1002
        counters[1] = 1
    return TrajectoryCurvePowerupState(
        curve_index=0,
        last_any_spawn_time=800,
        last_spawn_times=(-1000,) * 14,
        cooldown_times=tuple(cooldowns),
        spawn_counts=(0,) * 14,
        field_124_by_type=tuple(counters),
        active_color_counts=tuple(active_colors),
        reverse_speed=1.0,
        slow_ticks=slow_ticks,
        reverse_ticks=0,
        last_powerup_waypoint=0,
        powerup_triggered=triggered,
    )


def _slow_frames() -> tuple[TrajectoryFrame, ...]:
    result: list[TrajectoryFrame] = []
    for frame in _frames():
        triggered = frame.update >= 102
        slow_ticks = 0 if not triggered else 800 - (frame.update - 102)
        entities = tuple(
            replace(entity, powerup_primary_type=1)
            if entity.ball_id == 29
            else entity
            for entity in frame.entities
        )
        result.append(
            replace(
                frame,
                curve_powerups=(
                    _slow_state(
                        triggered=triggered,
                        slow_ticks=slow_ticks,
                    ),
                ),
                entities=entities,
            )
        )
    return tuple(result)


def test_trigger_verifier_closes_slow_countdown_timing() -> None:
    report = verify_powerup_trigger_frames(
        _slow_frames(),
        curve_index=0,
        trigger_ball_id=29,
        trigger_color_id=3,
        powerup_type=1,
        expected_newly_exploding_ids=(28, 29, 30, 128),
        expected_score_delta=40,
        expected_trigger_update=102,
    )

    assert report["slow"]["ticks_at_trigger"] == 800
    assert report["slow"]["ticks_next_update"] == 799
    assert report["reverse"] is None
