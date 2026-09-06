from __future__ import annotations

from dataclasses import replace

import pytest

from zuma_rl.pc_clear_sequence import (
    PcClearSequenceError,
    verify_clear_input_lock_frames,
    verify_clear_sequence_frames,
)
from zuma_rl.pc_memory_trajectory import (
    TrajectoryEntity,
    TrajectoryFrame,
    TrajectoryQRandState,
)


def _curve_ball(
    ball_id: int,
    color_id: int,
    *,
    zone: str,
    index: int,
    distance: float,
) -> TrajectoryEntity:
    return TrajectoryEntity(
        ball_id=ball_id,
        color_id=color_id,
        object_kind="ball",
        zone=zone,
        index=index,
        curve_distance=distance,
        position_x=distance,
        position_y=100.0,
        radius=18.0,
        should_remove=True,
        powerup_previous_type=14,
        powerup_primary_type=14,
        powerup_secondary_type=14,
    )


def _shooter(ball_id: int, color_id: int, zone: str) -> TrajectoryEntity:
    return TrajectoryEntity(
        ball_id=ball_id,
        color_id=color_id,
        object_kind="bullet",
        zone=zone,
        index=0,
        curve_distance=0.0,
        position_x=400.0,
        position_y=300.0,
        radius=18.0,
        fired=False,
    )


def _frames() -> tuple[TrajectoryFrame, ...]:
    qrand = TrajectoryQRandState(
        update_count=4,
        selected_index=1,
        weights=(0.25, 0.25, 0.25, 0.25, 0.0, 0.0),
        sways=(0.1, 0.2, 0.3, 0.4, 0.0, 0.0),
        last_hit=(1, 2, 3, 4, 0, 0),
        previous_hit=(0, 1, 0, 1, 0, 0),
    )
    result: list[TrajectoryFrame] = []
    for update in range(10, 20):
        initial = update == 10
        if initial:
            curve_entities = (
                _curve_ball(
                    1,
                    0,
                    zone="curve:0:list:05c",
                    index=0,
                    distance=40.0,
                ),
                _curve_ball(
                    2,
                    1,
                    zone="curve:0:list:05c",
                    index=1,
                    distance=76.0,
                ),
                _curve_ball(
                    3,
                    2,
                    zone="curve:0:list:068",
                    index=0,
                    distance=1.0,
                ),
            )
            chamber = (
                _shooter(90, 0, "shooter_current"),
                _shooter(91, 1, "shooter_next"),
            )
            current = (90, 0, 91, 1)
        else:
            curve_entities = ()
            chamber = (
                _shooter(100, 2, "shooter_current"),
                _shooter(101, 2, "shooter_next"),
            )
            current = (100, 2, 101, 2)
        score = 50 if update < 12 else (150 if update < 17 else 250)
        result.append(
            TrajectoryFrame(
                update=update,
                score=score,
                displayed_score=40,
                score_target=100,
                current_ball_id=current[0],
                current_color_id=current[1],
                next_ball_id=current[2],
                next_color_id=current[3],
                list_counts=(
                    (0, 0x50, 0),
                    (0, 0x5C, 2 if initial else 0),
                    (0, 0x68, 1 if initial else 0),
                ),
                entities=(*curve_entities, *chamber),
                qrand=qrand,
                thread_crt_rand_state=123,
                native_game_time=100 + update - 10,
                board_runtime_flag_157=update < 12,
                board_runtime_i32_f54=0,
            )
        )
    return tuple(result)


def test_clear_sequence_verifier_closes_two_update_contract() -> None:
    report = verify_clear_sequence_frames(
        _frames(),
        expected_initial_active_count=2,
        expected_clear_update=11,
        expected_formal_transition_update=12,
        expected_delayed_award_count=2,
        minimum_stable_tail_ticks=2,
    )

    assert report["status"] == "PASS"
    assert report["transition"]["updates_from_empty_to_formal"] == 1
    assert report["transition"]["repopulated"] is False
    assert report["diagnostic_score_tail"]["transferable_to_reward_model"] is False
    assert report["diagnostic_score_tail"]["award_count"] == 2


def test_clear_sequence_verifier_rejects_same_update_formal_flag() -> None:
    frames = list(_frames())
    frames[1] = replace(frames[1], board_runtime_flag_157=False)

    with pytest.raises(
        PcClearSequenceError,
        match="clear_sequence_formal_transition_phase_mismatch",
    ):
        verify_clear_sequence_frames(tuple(frames))


def test_clear_input_lock_requires_identical_gameplay_projection() -> None:
    frames = _frames()
    report = verify_clear_input_lock_frames(
        frames,
        frames[:5],
        click_down_update=11,
        click_up_update=12,
    )

    assert report["status"] == "PASS"
    assert report["accepted"] is False
    assert report["fired_projectiles"] == 0


def test_clear_input_lock_rejects_changed_chamber() -> None:
    baseline = _frames()
    clicked = list(baseline[:5])
    clicked[1] = replace(clicked[1], current_ball_id=999)

    with pytest.raises(
        PcClearSequenceError,
        match="clear_input_lock_state_changed:11",
    ):
        verify_clear_input_lock_frames(
            baseline,
            tuple(clicked),
            click_down_update=11,
            click_up_update=12,
        )
