from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from zuma_rl import pc_loss_sequence
from zuma_rl.pc_loss_sequence import (
    PcLossSequenceError,
    verify_loss_sequence_frames,
)
from zuma_rl.pc_memory_trajectory import TrajectoryEntity, TrajectoryFrame


def _ball(
    ball_id: int,
    index: int,
    distance: float,
    suck_count: int,
) -> TrajectoryEntity:
    return TrajectoryEntity(
        ball_id=ball_id,
        color_id=ball_id % 4,
        object_kind="ball",
        zone="curve:0:list:05c",
        index=index,
        curve_distance=distance,
        position_x=distance,
        position_y=100.0,
        radius=18.0,
        contact_next=False,
        exploding=False,
        explode_frame=0,
        should_remove=False,
        update_count=5,
        suck_count=suck_count,
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
    chain_states = (
        ((1, 16.0, 0), (2, 19.0, 0)),
        ((1, 17.0, 0), (2, 20.0, 0)),
        ((1, 17.0, 1),),
        ((1, 17.0, 2),),
        ((1, 17.0, 3),),
        ((1, 17.0, 4),),
        ((1, 18.0, 5),),
        ((1, 19.0, 6),),
        ((1, 20.0, 0),),
        (),
    )
    result: list[TrajectoryFrame] = []
    for update, state in enumerate(chain_states):
        before_reset = update == 0
        current_id, current_color = (
            (10, 0) if before_reset else (12, 3)
        )
        next_id, next_color = (
            (11, 1) if before_reset else (13, 3)
        )
        entities = [
            *(
                _ball(ball_id, index, distance, suck_count)
                for index, (ball_id, distance, suck_count) in enumerate(state)
            ),
            _shooter(current_id, current_color, "shooter_current"),
            _shooter(next_id, next_color, "shooter_next"),
        ]
        result.append(
            TrajectoryFrame(
                update=update,
                score=80,
                displayed_score=70,
                score_target=100,
                current_ball_id=current_id,
                current_color_id=current_color,
                next_ball_id=next_id,
                next_color_id=next_color,
                list_counts=(
                    (0, 0x5C, len(state)),
                    (0, 0x68, 1),
                ),
                entities=tuple(entities),
                native_game_time=100 if update == 0 else update - 1,
                board_runtime_flag_157=update < 2,
                board_runtime_i32_f54=0 if update < 2 else -(update - 1),
            )
        )
    return tuple(result)


def test_loss_sequence_verifier_closes_full_suction_contract() -> None:
    report = verify_loss_sequence_frames(
        _frames(),
        curve_index=0,
        trigger_ball_id=2,
        decoded_curve_end=20,
        expected_trigger_update=2,
        expected_chain_empty_update=9,
        expected_pending_count=1,
        expected_pretrigger_advance=1.0,
    )

    assert report["status"] == "PASS"
    assert report["trigger"]["native_time_reset_update"] == 1
    assert report["suction"]["ticks_from_trigger_to_empty"] == 7
    assert report["suction"]["removal_updates"] == {
        "2": [2],
        "9": [1],
    }


def test_loss_sequence_verifier_rejects_stationary_pretrigger_terminal() -> None:
    frames = list(_frames())
    frames[0] = replace(
        frames[0],
        entities=tuple(
            replace(entity, curve_distance=20.0)
            if entity.object_kind == "ball" and entity.ball_id == 2
            else entity
            for entity in frames[0].entities
        ),
    )

    with pytest.raises(
        PcLossSequenceError,
        match="loss_sequence_trigger_ball_transition_invalid",
    ):
        verify_loss_sequence_frames(
            tuple(frames),
            curve_index=0,
            trigger_ball_id=2,
            decoded_curve_end=20,
            expected_trigger_update=2,
            expected_chain_empty_update=9,
            expected_pending_count=1,
            expected_pretrigger_advance=1.0,
        )


def test_loss_sequence_verifier_rejects_wrong_suction_counter() -> None:
    frames = list(_frames())
    entities = tuple(
        replace(entity, suck_count=3)
        if entity.object_kind == "ball"
        else entity
        for entity in frames[3].entities
    )
    frames[3] = replace(frames[3], entities=entities)

    with pytest.raises(
        PcLossSequenceError,
        match="loss_sequence_suction_step_mismatch:3:1",
    ):
        verify_loss_sequence_frames(
            tuple(frames),
            curve_index=0,
            trigger_ball_id=2,
            decoded_curve_end=20,
            expected_trigger_update=2,
            expected_chain_empty_update=9,
            expected_pending_count=1,
            expected_pretrigger_advance=1.0,
        )


def test_natural_terminal_pretrigger_step_reaches_curve_end_exactly() -> None:
    reset = pc_loss_sequence._validate_natural_terminal_pretrigger_step(
        pretrigger_distance=3756.875,
        advance_speed=pc_loss_sequence.np.float32(0.125),
        decoded_curve_end=3757,
    )

    assert reset == 3757.0


def test_natural_terminal_pretrigger_step_rejects_wrong_advance() -> None:
    with pytest.raises(
        PcLossSequenceError,
        match="loss_simulator_terminal_pretrigger_step_mismatch",
    ):
        pc_loss_sequence._validate_natural_terminal_pretrigger_step(
            pretrigger_distance=3756.875,
            advance_speed=pc_loss_sequence.np.float32(0.25),
            decoded_curve_end=3757,
        )


def test_loss_v2_report_is_recomputed_from_bound_native_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trajectory_path = tmp_path / "trajectory" / "index.json"
    trajectory_path.parent.mkdir()
    trajectory_path.write_bytes(b"immutable-source\n")
    initialization = {
        "schema": pc_loss_sequence.LOSS_SOURCE_INITIALIZATION_SCHEMA,
        "version": pc_loss_sequence.LOSS_SOURCE_INITIALIZATION_VERSION,
        "status": "PASS",
        "classification": (
            pc_loss_sequence.LOSS_SOURCE_INITIALIZATION_CLASSIFICATION
        ),
        "source_frame_update": 0,
        "active_chain_entity_count": 2,
        "pending_chain_entity_count": 1,
        "terminal_ball_id": 2,
        "all_initial_entities_copied_from_one_native_frame": True,
        "terminal_geometry_excluded_only_from_curve_fit": True,
        "terminal_pretrigger_step_verified_against_installed_curve": True,
        "process_memory_writes": 0,
        "source_artifact_modified": False,
    }
    base_report = {
        "schema": pc_loss_sequence.LOSS_SIMULATOR_DIFF_SCHEMA,
        "version": pc_loss_sequence.LOSS_SIMULATOR_DIFF_VERSION,
        "status": "PASS",
        "start_update": 0,
        "end_update": 9,
        "ticks_compared": 9,
        "maximum_waypoint_error": 0.0,
        "reset_update": 1,
        "trigger_update": 2,
        "empty_update": 9,
        "removal_updates": {"2": [2], "9": [1]},
        "initial_chain_count": 2,
        "pending_ball_count": 1,
        "score": 80,
        "shooter_colors_after_reset": [3, 3],
        "post_empty_pc_ticks_excluded": 0,
        "scenario": {
            "level_id": "Jungle2",
            "hard": False,
            "curve_index": 0,
            "decoded_curve_end": 20,
            "source_bound_terminal_ball_id": 2,
            "source_bound_terminal_geometry_excluded_from_curve_fit": True,
            "source_terminal_pretrigger_distance": 19.0,
            "source_terminal_reset_distance": 20.0,
            "terminal_pretrigger_advance_exact": True,
            "inferred_advance_speed": 1.0,
        },
        "source_state_initialization": initialization,
    }
    oracle = {
        "schema": pc_loss_sequence.LOSS_SEQUENCE_SCHEMA,
        "status": "PASS",
        "trigger": {"update": 2},
        "suction": {
            "first_chain_empty_update": 9,
            "ticks_from_trigger_to_empty": 7,
        },
    }
    monkeypatch.setattr(
        pc_loss_sequence,
        "load_memory_trajectory",
        lambda _path: _frames(),
    )
    monkeypatch.setattr(
        pc_loss_sequence,
        "verify_loss_sequence_frames",
        lambda *args, **kwargs: deepcopy(oracle),
    )
    monkeypatch.setattr(
        pc_loss_sequence,
        "compare_loss_sequence_with_simulator",
        lambda *args, **kwargs: deepcopy(base_report),
    )
    report = deepcopy(base_report)
    report["trajectory"] = {
        "artifact": "trajectory/index.json",
        "artifact_sha256": pc_loss_sequence._sha256_path(trajectory_path),
        "captured_start_update": 0,
        "captured_end_update": 9,
        "selected_start_update": 0,
        "selected_end_update": 9,
    }
    report["oracle"] = {
        "schema": oracle["schema"],
        "status": oracle["status"],
        "trigger_update": 2,
        "empty_update": 9,
        "ticks_from_trigger_to_empty": 7,
    }

    assert (
        pc_loss_sequence.validate_loss_simulator_diff_report(
            report,
            evidence_root=tmp_path,
            original_root=tmp_path,
        )
        is None
    )

    report["score"] = 81
    assert (
        pc_loss_sequence.validate_loss_simulator_diff_report(
            report,
            evidence_root=tmp_path,
            original_root=tmp_path,
        )
        == "loss simulator diff differs from live recomputation"
    )
