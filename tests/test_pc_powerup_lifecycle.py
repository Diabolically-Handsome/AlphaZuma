from __future__ import annotations

from dataclasses import replace

import pytest

from zuma_rl.pc_memory_trajectory import (
    TrajectoryCurvePowerupState,
    TrajectoryEntity,
    TrajectoryFrame,
)
from zuma_rl.pc_powerup_lifecycle import (
    PcPowerupLifecycleError,
    verify_powerup_lifecycle_frames,
)


def _curve_state(
    *,
    cooldown: int,
    active_color_1: int,
) -> TrajectoryCurvePowerupState:
    cooldowns = [-1000] * 14
    cooldowns[3] = cooldown
    last_spawns = [-1000] * 14
    last_spawns[3] = 500
    spawn_counts = [0] * 14
    spawn_counts[3] = 1
    active_colors = [0] * 6
    active_colors[1] = active_color_1
    return TrajectoryCurvePowerupState(
        curve_index=0,
        last_any_spawn_time=800,
        last_spawn_times=tuple(last_spawns),
        cooldown_times=tuple(cooldowns),
        spawn_counts=tuple(spawn_counts),
        field_124_by_type=(0,) * 14,
        active_color_counts=tuple(active_colors),
    )


def _frame(
    update: int,
    *,
    native_time: int,
    previous_type: int,
    primary_type: int,
    previous_ticks: int,
    lifetime_ticks: int,
    transition_ticks: int,
    cooldown: int,
    active_color_1: int,
) -> TrajectoryFrame:
    visual_step = 0.04
    entity = TrajectoryEntity(
        ball_id=65,
        color_id=1,
        object_kind="ball",
        zone="curve:0:list:05c",
        index=0,
        curve_distance=100.0,
        position_x=20.0,
        position_y=30.0,
        powerup_previous_type=previous_type,
        powerup_primary_type=primary_type,
        powerup_secondary_type=14,
        powerup_previous_ticks=previous_ticks,
        powerup_lifetime_ticks=lifetime_ticks,
        powerup_transition_ticks=transition_ticks,
        powerup_visual_scale=1.0 + transition_ticks * visual_step,
        powerup_visual_step=visual_step,
        powerup_visual_index=-1,
    )
    return TrajectoryFrame(
        update=update,
        score=100,
        displayed_score=100,
        score_target=1000,
        current_ball_id=90,
        current_color_id=0,
        next_ball_id=91,
        next_color_id=2,
        list_counts=((0, 0x5C, 1),),
        entities=(entity,),
        native_game_time=native_time,
        curve_powerups=(
            _curve_state(
                cooldown=cooldown,
                active_color_1=active_color_1,
            ),
        ),
    )


def _lifecycle_frames() -> tuple[TrajectoryFrame, ...]:
    frames: list[TrajectoryFrame] = []
    for update, lifetime in ((10, 3), (11, 2), (12, 1)):
        frames.append(
            _frame(
                update,
                native_time=1000 + update - 10,
                previous_type=14,
                primary_type=3,
                previous_ticks=0,
                lifetime_ticks=lifetime,
                transition_ticks=0,
                cooldown=-1000,
                active_color_1=1,
            )
        )
    expiration_update = 13
    for update in range(expiration_update, 19):
        delta = update - expiration_update
        frames.append(
            _frame(
                update,
                native_time=1000 + update - 10,
                previous_type=3 if delta < 4 else 14,
                primary_type=3 if delta < 3 else 14,
                previous_ticks=max(4 - delta, 0),
                lifetime_ticks=0,
                transition_ticks=max(3 - delta, 0),
                cooldown=1003,
                active_color_1=0,
            )
        )
    return tuple(frames)


def test_lifecycle_verifier_closes_expiry_fade_and_cleanup() -> None:
    report = verify_powerup_lifecycle_frames(
        _lifecycle_frames(),
        ball_id=65,
        curve_index=0,
        color_id=1,
        powerup_type=3,
        expected_expiration_update=13,
        transition_ticks=3,
        previous_retention_ticks=5,
    )

    assert report["status"] == "PASS"
    assert report["countdown"]["initial_lifetime_ticks"] == 3
    assert report["expiration"]["native_game_time"] == 1003
    assert report["expiration"]["sampled_previous_ticks"] == 4
    assert report["primary_clear_update"] == 16
    assert report["previous_marker_clear_update"] == 17
    assert report["expiration"]["active_color_count_before"] == 1
    assert report["expiration"]["active_color_count_after"] == 0


def test_lifecycle_verifier_rejects_one_bad_fade_tick() -> None:
    frames = list(_lifecycle_frames())
    target = frames[5].entities[0]
    frames[5] = replace(
        frames[5],
        entities=(
            replace(target, powerup_transition_ticks=2),
        ),
    )

    with pytest.raises(
        PcPowerupLifecycleError,
        match="powerup_lifecycle_fade_or_cleanup_mismatch:15",
    ):
        verify_powerup_lifecycle_frames(
            tuple(frames),
            ball_id=65,
            curve_index=0,
            color_id=1,
            powerup_type=3,
            expected_expiration_update=13,
            transition_ticks=3,
            previous_retention_ticks=5,
        )
