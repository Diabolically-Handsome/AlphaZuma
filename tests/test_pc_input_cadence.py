from __future__ import annotations

from dataclasses import replace

import pytest

from zuma_rl.pc_input_cadence import (
    PcClickCadenceError,
    verify_click_cadence_frames,
)
from zuma_rl.pc_memory_trajectory import TrajectoryEntity, TrajectoryFrame


def _bullet(
    ball_id: int,
    color_id: int,
    zone: str,
    *,
    fired: bool,
) -> TrajectoryEntity:
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
        velocity_x=0.0,
        velocity_y=-8.0,
        fired=fired,
    )


def _frames() -> tuple[TrajectoryFrame, ...]:
    result: list[TrajectoryFrame] = []
    for update in range(36):
        if update < 7:
            current_id, current_color = 10, 0
            next_id, next_color = 11, 1
            firing = 2 <= update < 7
        elif update < 28:
            current_id, current_color = 11, 1
            next_id, next_color = 12, 2
            firing = 23 <= update < 28
        else:
            current_id, current_color = 12, 2
            next_id, next_color = 13, 3
            firing = False

        entities = [
            _bullet(
                current_id,
                current_color,
                "shooter_current",
                fired=firing,
            ),
            _bullet(
                next_id,
                next_color,
                "shooter_next",
                fired=False,
            ),
        ]
        if 7 <= update < 10:
            entities.append(_bullet(10, 0, "fired", fired=False))
        if 28 <= update < 31:
            entities.append(_bullet(11, 1, "fired", fired=False))
        result.append(
            TrajectoryFrame(
                update=update,
                score=8_120,
                displayed_score=8_120,
                score_target=9_650,
                current_ball_id=current_id,
                current_color_id=current_color,
                next_ball_id=next_id,
                next_color_id=next_color,
                list_counts=(),
                entities=tuple(entities),
                native_game_time=1_000 + update,
            )
        )
    return tuple(result)


def _downs() -> tuple[int, ...]:
    return tuple(range(1, 35, 3))


def test_click_verifier_closes_accept_release_and_ignore_timing() -> None:
    report = verify_click_cadence_frames(
        _frames(),
        down_updates=_downs(),
        expected_accepted_down_updates=(1, 22),
        expected_free_projectile_ticks=3,
    )

    assert report["status"] == "PASS"
    assert report["input"]["accepted_count"] == 2
    assert report["input"]["rejected_count"] == 10
    assert report["cadence"]["accepted_intervals"] == [21]
    assert report["cadence"]["release_updates"] == [7, 28]
    assert [
        shot["projectile_ball_id"] for shot in report["shots"]
    ] == [10, 11]


def test_click_verifier_rejects_missing_fired_state_transition() -> None:
    frames = list(_frames())
    entities = tuple(
        replace(entity, fired=False)
        if entity.zone == "shooter_current"
        else entity
        for entity in frames[2].entities
    )
    frames[2] = replace(frames[2], entities=entities)

    with pytest.raises(
        PcClickCadenceError,
        match="click_cadence_accepted_transition_mismatch",
    ):
        verify_click_cadence_frames(
            tuple(frames),
            down_updates=_downs(),
            expected_accepted_down_updates=(1, 22),
            expected_free_projectile_ticks=3,
        )


def test_click_verifier_rejects_extra_accept_during_reload() -> None:
    frames = list(_frames())
    entities = tuple(
        replace(entity, fired=False)
        if entity.zone == "shooter_current"
        else entity
        for entity in frames[4].entities
    )
    frames[4] = replace(frames[4], entities=entities)

    with pytest.raises(
        PcClickCadenceError,
        match="click_cadence_accepted_transition_mismatch",
    ):
        verify_click_cadence_frames(
            tuple(frames),
            down_updates=_downs(),
            expected_accepted_down_updates=(1, 22),
            expected_free_projectile_ticks=3,
        )


def test_click_verifier_rejects_wrong_free_projectile_lifetime() -> None:
    frames = list(_frames())
    frames[9] = replace(
        frames[9],
        entities=tuple(
            entity
            for entity in frames[9].entities
            if not (entity.zone == "fired" and entity.ball_id == 10)
        ),
    )

    with pytest.raises(
        PcClickCadenceError,
        match="click_cadence_projectile_lifetime_mismatch:10",
    ):
        verify_click_cadence_frames(
            tuple(frames),
            down_updates=_downs(),
            expected_accepted_down_updates=(1, 22),
            expected_free_projectile_ticks=3,
        )
