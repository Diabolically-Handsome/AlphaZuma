from __future__ import annotations

from dataclasses import replace

import pytest

from zuma_rl.pc_memory_trajectory import (
    TrajectoryCurvePlanState,
    TrajectoryEntity,
    TrajectoryFrame,
)
from zuma_rl.pc_natural_win import (
    PcNaturalWinError,
    _curve_entities,
    derive_natural_win_feature_proofs,
    verify_natural_win_frames,
)


def _curve_ball(ball_id: int, index: int) -> TrajectoryEntity:
    return TrajectoryEntity(
        ball_id=ball_id,
        color_id=ball_id % 4,
        object_kind="ball",
        zone="curve:0:list:05c",
        index=index,
        curve_distance=100.0 + index * 36.0,
        position_x=200.0 + index * 36.0,
        position_y=300.0,
        radius=18.0,
        should_remove=False,
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
        position_y=500.0,
        radius=18.0,
        fired=False,
    )


def _frames() -> tuple[TrajectoryFrame, ...]:
    result: list[TrajectoryFrame] = []
    for update in range(100, 110):
        if update <= 104:
            chain_count = 3
        elif update == 105:
            chain_count = 2
        elif update == 106:
            chain_count = 1
        else:
            chain_count = 0
        balls = tuple(
            _curve_ball(ball_id, index)
            for index, ball_id in enumerate(
                range(10, 10 + chain_count)
            )
        )
        planned_count = 0
        exhausted = True
        terminal = update >= 107
        if update < 102:
            score = 90
        elif update < 108:
            score = 110
        else:
            score = 210
        result.append(
            TrajectoryFrame(
                update=update,
                score=score,
                displayed_score=min(score, 100),
                score_target=100,
                current_ball_id=None if terminal else 90,
                current_color_id=None if terminal else 1,
                next_ball_id=None if terminal else 91,
                next_color_id=None if terminal else 2,
                list_counts=(
                    (0, 0x50, 0),
                    (0, 0x5C, chain_count),
                    (0, 0x68, 0),
                ),
                entities=(
                    *balls,
                    *(
                        ()
                        if terminal
                        else (
                            _shooter(90, 1, "shooter_current"),
                            _shooter(91, 2, "shooter_next"),
                        )
                    ),
                ),
                native_game_time=500 + update - 100,
                board_runtime_flag_157=False,
                board_runtime_i32_f54=0,
                active_board_version=2,
                board_mode_flag_1064=False,
                curve_plan_exhausted=exhausted,
                post_zuma_timer_remaining=0,
                post_zuma_ramp_404=0.0,
                post_zuma_ramp_408=0.0,
                curve_plans=(
                    TrajectoryCurvePlanState(
                        curve_index=0,
                        begin_address=0x1000,
                        end_address=0x1000 + planned_count * 0x14,
                        capacity_address=0x1100,
                        planned_count=planned_count,
                        capacity_count=12,
                        add_plan_enabled=False,
                    ),
                ),
            )
        )
    return tuple(result)


def test_natural_win_closes_target_feed_and_formal_transition() -> None:
    report = verify_natural_win_frames(
        _frames(),
        expected_score_cross_update=102,
        expected_empty_update=107,
        expected_formal_transition_update=108,
    )

    assert report["status"] == "PASS"
    assert report["classification"] == (
        "frame-contract-requires-provenance-binding"
    )
    assert report["feed_exhaustion"]["repopulated"] is False
    assert report["feed_exhaustion"][
        "transition_observed_in_window"
    ] is False
    assert report["transition"]["updates_from_empty_to_formal"] == 1
    assert report["transition"]["loss_counter"] == 0


def test_natural_win_derives_both_gate_features() -> None:
    proofs = derive_natural_win_feature_proofs(_frames())

    assert [proof["feature"] for proof in proofs] == [
        "natural_win",
        "zuma_transition",
    ]
    assert proofs[0]["sequence"]["status"] == "PASS"
    assert proofs[0]["sequence"]["transition"][
        "terminal_chamber_cleared"
    ] is True


def test_natural_win_counts_insertion_staging_bullets() -> None:
    frame = _frames()[0]
    staging = _shooter(92, 3, "curve:0:list:050")
    frame = replace(
        frame,
        list_counts=tuple(
            (curve, offset, 1 if offset == 0x50 else count)
            for curve, offset, count in frame.list_counts
        ),
        entities=(*frame.entities, staging),
    )

    assert _curve_entities(
        frame,
        curve_index=0,
        container_offset=0x50,
    ) == (staging,)


def test_natural_win_requires_below_target_context() -> None:
    frames = tuple(replace(frame, score=110) for frame in _frames())

    with pytest.raises(
        PcNaturalWinError,
        match="natural_win_below_target_context_missing",
    ):
        verify_natural_win_frames(frames)


def test_natural_win_rejects_plan_repopulation() -> None:
    frames = list(_frames())
    frames[5] = replace(
        frames[5],
        curve_plans=(
            replace(
                frames[5].curve_plans[0],
                planned_count=1,
                add_plan_enabled=True,
            ),
        ),
    )

    with pytest.raises(
        PcNaturalWinError,
        match="natural_win_plan_repopulated:105",
    ):
        verify_natural_win_frames(tuple(frames))


def test_natural_win_rejects_loss_counter_transition() -> None:
    frames = list(_frames())
    frames[8] = replace(frames[8], board_runtime_i32_f54=-1)

    with pytest.raises(
        PcNaturalWinError,
        match="natural_win_loss_counter_changed",
    ):
        verify_natural_win_frames(tuple(frames))


def test_natural_win_detector_ignores_ordinary_gameplay_window() -> None:
    frames = tuple(
        replace(frame, board_runtime_flag_157=True)
        for frame in _frames()
    )

    assert derive_natural_win_feature_proofs(frames) == ()


def test_natural_win_rejects_missing_terminal_award() -> None:
    frames = list(_frames())
    frames[8] = replace(frames[8], score=110)

    with pytest.raises(
        PcNaturalWinError,
        match="natural_win_terminal_award_mismatch",
    ):
        verify_natural_win_frames(tuple(frames))


def test_natural_win_rejects_uncleared_terminal_chamber() -> None:
    frames = list(_frames())
    frames[7] = replace(
        frames[7],
        current_ball_id=90,
        current_color_id=1,
        next_ball_id=91,
        next_color_id=2,
        entities=(
            _shooter(90, 1, "shooter_current"),
            _shooter(91, 2, "shooter_next"),
        ),
    )

    with pytest.raises(
        PcNaturalWinError,
        match="natural_win_terminal_chamber_not_cleared:107",
    ):
        verify_natural_win_frames(tuple(frames))
