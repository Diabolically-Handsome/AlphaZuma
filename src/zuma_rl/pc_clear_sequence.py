"""Strictly verify the retail two-update empty-chain victory transition."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .pc_memory_trajectory import (
    ACTIVE_CHAIN_LIST_OFFSET,
    TrajectoryEntity,
    TrajectoryFrame,
    load_memory_trajectory,
)
from .revenge_core import PowerupType, RevengeSimulator


CLEAR_SEQUENCE_SCHEMA = "zuma-rl.pc-clear-sequence-verification"
CLEAR_SEQUENCE_VERSION = 1
CLEAR_SIMULATOR_SCHEMA = "zuma-rl.pc-clear-simulator-transition"
CLEAR_SIMULATOR_VERSION = 1
PENDING_CHAIN_LIST_OFFSET = 0x68
INSERTING_CHAIN_LIST_OFFSET = 0x50


class PcClearSequenceError(ValueError):
    """A trajectory does not satisfy the retail victory-transition contract."""


def _fail(code: str) -> None:
    raise PcClearSequenceError(code)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _curve_entities(
    frame: TrajectoryFrame,
    *,
    curve_index: int,
    container_offset: int,
) -> tuple[TrajectoryEntity, ...]:
    zone = f"curve:{curve_index}:list:{container_offset:03x}"
    result = tuple(
        sorted(
            (
                entity
                for entity in frame.entities
                if entity.zone == zone and entity.object_kind == "ball"
            ),
            key=lambda entity: entity.index,
        )
    )
    if tuple(entity.index for entity in result) != tuple(range(len(result))):
        _fail(f"clear_sequence_curve_indexes_invalid:{frame.update}")
    if len({entity.ball_id for entity in result}) != len(result):
        _fail(f"clear_sequence_curve_ids_duplicate:{frame.update}")
    return result


def _active_chain(
    frame: TrajectoryFrame,
    curve_index: int,
) -> tuple[TrajectoryEntity, ...]:
    return _curve_entities(
        frame,
        curve_index=curve_index,
        container_offset=ACTIVE_CHAIN_LIST_OFFSET,
    )


def _pending_chain(
    frame: TrajectoryFrame,
    curve_index: int,
) -> tuple[TrajectoryEntity, ...]:
    return _curve_entities(
        frame,
        curve_index=curve_index,
        container_offset=PENDING_CHAIN_LIST_OFFSET,
    )


def _fired(frame: TrajectoryFrame) -> tuple[TrajectoryEntity, ...]:
    return tuple(entity for entity in frame.entities if entity.zone == "fired")


def _validate_frame_sequence(frames: Sequence[TrajectoryFrame]) -> None:
    if len(frames) < 3:
        _fail("clear_sequence_too_short")
    for before, after in zip(frames, frames[1:]):
        if after.update != before.update + 1:
            _fail(
                "clear_sequence_updates_not_consecutive:"
                f"{before.update}->{after.update}"
            )
        if (
            before.native_game_time is None
            or after.native_game_time is None
            or after.native_game_time != before.native_game_time + 1
        ):
            _fail(
                "clear_sequence_native_time_mismatch:"
                f"{before.update}->{after.update}"
            )
    if any(
        frame.board_runtime_flag_157 is None
        or frame.board_runtime_i32_f54 is None
        for frame in frames
    ):
        _fail("clear_sequence_board_runtime_fields_missing")


def _powerups_are_none(entities: Sequence[TrajectoryEntity]) -> bool:
    none_type = int(PowerupType.NONE)
    return all(
        (
            entity.powerup_previous_type,
            entity.powerup_primary_type,
            entity.powerup_secondary_type,
        )
        == (none_type, none_type, none_type)
        for entity in entities
    )


def verify_clear_sequence_frames(
    frames: Sequence[TrajectoryFrame],
    *,
    curve_index: int = 0,
    expected_initial_active_count: int | None = None,
    expected_initial_pending_count: int = 1,
    expected_clear_update: int | None = None,
    expected_formal_transition_update: int | None = None,
    expected_delayed_award_count: int | None = None,
    expected_award_points: int = 100,
    expected_award_period: int = 5,
    minimum_stable_tail_ticks: int = 0,
) -> dict[str, Any]:
    """Verify the transition and quarantine intervention-tainted score awards."""

    _validate_frame_sequence(frames)
    if (
        curve_index < 0
        or expected_initial_pending_count < 0
        or expected_award_points <= 0
        or expected_award_period <= 0
        or minimum_stable_tail_ticks < 0
    ):
        raise ValueError("clear-sequence expectations are invalid")

    initial = frames[0]
    initial_active = _active_chain(initial, curve_index)
    initial_pending = _pending_chain(initial, curve_index)
    if not initial_active:
        _fail("clear_sequence_initial_active_chain_missing")
    if (
        expected_initial_active_count is not None
        and len(initial_active) != expected_initial_active_count
    ):
        _fail("clear_sequence_initial_active_count_mismatch")
    if len(initial_pending) != expected_initial_pending_count:
        _fail("clear_sequence_initial_pending_count_mismatch")
    injected_entities = (*initial_active, *initial_pending)
    if any(entity.should_remove is not True for entity in injected_entities):
        _fail("clear_sequence_remove_flags_not_set")
    if not _powerups_are_none(injected_entities):
        _fail("clear_sequence_initial_powerup_contamination")
    if _fired(initial):
        _fail("clear_sequence_initial_fired_projectile_present")

    clear_index: int | None = None
    for index, frame in enumerate(frames[1:], start=1):
        if (
            not _active_chain(frame, curve_index)
            and not _pending_chain(frame, curve_index)
            and frame.list_count(curve_index, INSERTING_CHAIN_LIST_OFFSET) == 0
        ):
            clear_index = index
            break
    if clear_index is None:
        _fail("clear_sequence_never_empty")
    clear = frames[clear_index]
    if clear_index != 1:
        _fail("clear_sequence_not_cleared_in_one_update")
    if expected_clear_update is not None and clear.update != expected_clear_update:
        _fail("clear_sequence_clear_update_mismatch")

    formal_index: int | None = None
    for index in range(clear_index, len(frames)):
        if frames[index].board_runtime_flag_157 is False:
            formal_index = index
            break
    if formal_index is None:
        _fail("clear_sequence_formal_transition_missing")
    formal = frames[formal_index]
    if formal_index != clear_index + 1:
        _fail("clear_sequence_formal_transition_phase_mismatch")
    if (
        expected_formal_transition_update is not None
        and formal.update != expected_formal_transition_update
    ):
        _fail("clear_sequence_formal_transition_update_mismatch")

    if any(
        frame.board_runtime_flag_157 is not True
        for frame in frames[:formal_index]
    ) or any(
        frame.board_runtime_flag_157 is not False
        for frame in frames[formal_index:]
    ):
        _fail("clear_sequence_runtime_flag_phase_mismatch")
    if any(frame.board_runtime_i32_f54 != 0 for frame in frames):
        _fail("clear_sequence_runtime_counter_changed")

    stable_current = (
        clear.current_ball_id,
        clear.current_color_id,
        clear.next_ball_id,
        clear.next_color_id,
    )
    if any(value is None for value in stable_current):
        _fail("clear_sequence_postclear_chamber_missing")
    for frame in frames[clear_index:]:
        if (
            _active_chain(frame, curve_index)
            or _pending_chain(frame, curve_index)
            or frame.list_count(curve_index, INSERTING_CHAIN_LIST_OFFSET) != 0
            or _fired(frame)
        ):
            _fail(f"clear_sequence_terminal_state_repopulated:{frame.update}")
        chamber = (
            frame.current_ball_id,
            frame.current_color_id,
            frame.next_ball_id,
            frame.next_color_id,
        )
        if chamber != stable_current:
            _fail(f"clear_sequence_postclear_chamber_changed:{frame.update}")

    if clear.score != initial.score:
        _fail("clear_sequence_score_changed_on_empty_update")
    if any(frame.displayed_score != initial.displayed_score for frame in frames):
        _fail("clear_sequence_displayed_score_changed")
    if any(frame.score_target != initial.score_target for frame in frames):
        _fail("clear_sequence_score_target_changed")
    if initial.qrand is None or initial.thread_crt_rand_state is None:
        _fail("clear_sequence_initial_shooter_rng_missing")
    if any(frame.qrand != initial.qrand for frame in frames):
        _fail("clear_sequence_qrand_changed")
    if any(
        frame.thread_crt_rand_state != initial.thread_crt_rand_state
        for frame in frames
    ):
        _fail("clear_sequence_thread_crt_rng_changed")

    score_changes: list[dict[str, int]] = []
    for before, after in zip(frames, frames[1:]):
        if after.score == before.score:
            continue
        delta = after.score - before.score
        if delta != expected_award_points:
            _fail(
                "clear_sequence_delayed_award_delta_mismatch:"
                f"{after.update}:{delta}"
            )
        score_changes.append({"update": after.update, "delta": delta})
    if not score_changes:
        _fail("clear_sequence_delayed_awards_missing")
    if score_changes[0]["update"] != formal.update:
        _fail("clear_sequence_first_award_phase_mismatch")
    for before, after in zip(score_changes, score_changes[1:]):
        if after["update"] - before["update"] != expected_award_period:
            _fail("clear_sequence_delayed_award_period_mismatch")
    if (
        expected_delayed_award_count is not None
        and len(score_changes) != expected_delayed_award_count
    ):
        _fail("clear_sequence_delayed_award_count_mismatch")
    stable_tail_ticks = frames[-1].update - score_changes[-1]["update"]
    if stable_tail_ticks < minimum_stable_tail_ticks:
        _fail("clear_sequence_stable_tail_too_short")

    return {
        "schema": CLEAR_SEQUENCE_SCHEMA,
        "version": CLEAR_SEQUENCE_VERSION,
        "status": "PASS",
        "classification": "diagnostic-not-unmodified-pc-evidence",
        "curve_index": curve_index,
        "captured_updates": {
            "start": initial.update,
            "end": frames[-1].update,
            "count": len(frames),
        },
        "initial": {
            "active_ball_count": len(initial_active),
            "pending_ball_count": len(initial_pending),
            "score": initial.score,
            "displayed_score": initial.displayed_score,
            "score_target": initial.score_target,
            "native_game_time": initial.native_game_time,
            "powerup_free": True,
        },
        "transition": {
            "empty_update": clear.update,
            "formal_transition_update": formal.update,
            "updates_from_empty_to_formal": formal.update - clear.update,
            "empty_runtime_flag": clear.board_runtime_flag_157,
            "formal_runtime_flag": formal.board_runtime_flag_157,
            "postclear_chamber": {
                "current_ball_id": stable_current[0],
                "current_color_id": stable_current[1],
                "next_ball_id": stable_current[2],
                "next_color_id": stable_current[3],
            },
            "repopulated": False,
            "fired_projectiles": 0,
        },
        "diagnostic_score_tail": {
            "transferable_to_reward_model": False,
            "reason": (
                "simultaneously forcing every curve object through the "
                "should-remove path creates a non-natural retail award queue"
            ),
            "award_points": expected_award_points,
            "award_period_updates": expected_award_period,
            "award_count": len(score_changes),
            "score_delta": frames[-1].score - initial.score,
            "first_award_update": score_changes[0]["update"],
            "last_award_update": score_changes[-1]["update"],
            "stable_tail_ticks": stable_tail_ticks,
            "changes": score_changes,
        },
    }


def _input_lock_projection(frame: TrajectoryFrame) -> tuple[Any, ...]:
    return (
        frame.update,
        frame.score,
        frame.displayed_score,
        frame.score_target,
        frame.current_ball_id,
        frame.current_color_id,
        frame.next_ball_id,
        frame.next_color_id,
        frame.list_counts,
        frame.entities,
        frame.qrand,
        frame.thread_crt_rand_state,
        frame.native_game_time,
        frame.board_runtime_flag_157,
        frame.board_runtime_i32_f54,
    )


def verify_clear_input_lock_frames(
    baseline_frames: Sequence[TrajectoryFrame],
    click_frames: Sequence[TrajectoryFrame],
    *,
    click_down_update: int,
    click_up_update: int,
) -> dict[str, Any]:
    """Prove that a transition-frame click leaves gameplay state unchanged."""

    if click_up_update <= click_down_update:
        raise ValueError("click release must follow click press")
    baseline = {frame.update: frame for frame in baseline_frames}
    click = {frame.update: frame for frame in click_frames}
    required = range(click_down_update, click_frames[-1].update + 1)
    if any(update not in baseline or update not in click for update in required):
        _fail("clear_input_lock_comparison_range_missing")
    mismatches = [
        update
        for update in required
        if _input_lock_projection(baseline[update])
        != _input_lock_projection(click[update])
    ]
    if mismatches:
        _fail(f"clear_input_lock_state_changed:{mismatches[0]}")
    if any(_fired(click[update]) for update in required):
        _fail("clear_input_lock_projectile_created")
    return {
        "status": "PASS",
        "classification": "diagnostic-dmo-input-comparison",
        "click": {
            "down_update": click_down_update,
            "up_update": click_up_update,
        },
        "compared_updates": {
            "start": click_down_update,
            "end": click_frames[-1].update,
            "count": click_frames[-1].update - click_down_update + 1,
        },
        "gameplay_projection_mismatches": 0,
        "fired_projectiles": 0,
        "accepted": False,
    }


def _load_mutation(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, ValueError) as error:
        raise PcClearSequenceError("clear_mutation_json_invalid") from error
    if not isinstance(value, dict):
        _fail("clear_mutation_json_root_invalid")
    return value


def _verify_mutation(
    mutation: Mapping[str, Any],
    initial: TrajectoryFrame,
    *,
    curve_index: int,
) -> dict[str, Any]:
    if (
        mutation.get("schema") != "zuma-rl.pc-synthesized-clear-state"
        or mutation.get("version") != 1
        or mutation.get("classification")
        != "diagnostic-not-unmodified-pc-evidence"
        or mutation.get("persistent_files_modified") is not False
        or mutation.get("curve_index") != curve_index
    ):
        _fail("clear_mutation_contract_invalid")
    rows = mutation.get("mutations")
    if not isinstance(rows, list):
        _fail("clear_mutation_rows_invalid")
    active = _active_chain(initial, curve_index)
    pending = _pending_chain(initial, curve_index)
    active_ids = {entity.ball_id for entity in active}
    pending_ids = {entity.ball_id for entity in pending}
    active_flags = {
        row.get("ball_id")
        for row in rows
        if (
            isinstance(row, dict)
            and row.get("label") == "active_ball_should_remove"
            and row.get("field_offset") == 0xC0
            and row.get("before_hex") == "00"
            and row.get("after_hex") == "01"
        )
    }
    pending_flags = {
        row.get("ball_id")
        for row in rows
        if (
            isinstance(row, dict)
            and row.get("label") == "pending_ball_should_remove"
            and row.get("field_offset") == 0xC0
            and row.get("before_hex") == "00"
            and row.get("after_hex") == "01"
        )
    }
    distance_rows = [
        row
        for row in rows
        if (
            isinstance(row, dict)
            and row.get("label") == "entrance_ball_curve_distance"
            and row.get("field_offset") == 0x1C
        )
    ]
    if (
        active_flags != active_ids
        or pending_flags != pending_ids
        or len(distance_rows) != 1
        or distance_rows[0].get("ball_id") != active[0].ball_id
        or distance_rows[0].get("after_value") != active[0].curve_distance
        or mutation.get("active_ball_count") != len(active)
        or mutation.get("pending_ball_count") != len(pending)
        or len(rows) != len(active) + len(pending) + 1
    ):
        _fail("clear_mutation_rows_do_not_bind_initial_state")
    return {
        "schema": mutation["schema"],
        "version": mutation["version"],
        "classification": mutation["classification"],
        "mutation_count": len(rows),
        "active_remove_flags": len(active_flags),
        "pending_remove_flags": len(pending_flags),
        "distance_guard": {
            "ball_id": distance_rows[0]["ball_id"],
            "before": distance_rows[0].get("before_value"),
            "after": distance_rows[0].get("after_value"),
        },
        "persistent_files_modified": False,
    }


def compare_clear_transition_with_simulator(
    frames: Sequence[TrajectoryFrame],
    *,
    original_root: Path,
    level_id: str = "Jungle2",
    hard: bool = False,
    curve_index: int = 0,
) -> dict[str, Any]:
    """Check the transferable two-tick/input-lock contract in the simulator."""

    initial = frames[0]
    active = _active_chain(initial, curve_index)
    pending = _pending_chain(initial, curve_index)
    if not active or not pending or initial.native_game_time is None:
        _fail("clear_simulator_initial_state_invalid")
    simulator = RevengeSimulator.from_installed(
        level_id,
        root=original_root,
        hard=hard,
        curve_index=curve_index,
        seed=0,
    )
    all_entities = (*pending, *active)
    simulator.load_state(
        colors=[entity.color_id for entity in all_entities],
        waypoints=[entity.curve_distance for entity in all_entities],
        contacts=[False] * max(0, len(all_entities) - 1),
        current_color=initial.current_color_id,
        next_color=initial.next_color_id,
        score=initial.score,
        score_at_level_start=(
            initial.score_target
            - int(getattr(simulator.parameters, "zuma_score", 0))
        ),
    )
    simulator.stop_adding = True
    simulator.native_game_time = initial.native_game_time
    for ball, entity in zip(simulator.balls, all_entities, strict=True):
        ball.id = entity.ball_id
        ball.should_remove = True

    transition = simulator.tick()
    transition_input = {
        "fire_accepted": simulator.request_fire(0.0),
        "swap_accepted": simulator.swap_balls(),
    }
    if (
        simulator.balls
        or simulator.pending_colors
        or not simulator.win_pending
        or simulator.outcome is not None
        or not transition.win_pending
        or transition_input != {
            "fire_accepted": False,
            "swap_accepted": False,
        }
        or simulator.native_game_time != frames[1].native_game_time
    ):
        _fail("clear_simulator_transition_mismatch")
    formal = simulator.tick()
    if (
        simulator.win_pending
        or simulator.outcome != "win"
        or formal.outcome != "win"
        or simulator.native_game_time != frames[2].native_game_time
    ):
        _fail("clear_simulator_formal_outcome_mismatch")
    return {
        "schema": CLEAR_SIMULATOR_SCHEMA,
        "version": CLEAR_SIMULATOR_VERSION,
        "status": "PASS",
        "scope": (
            "transferable empty-chain transition only; intervention-tainted "
            "retail score objects are deliberately excluded"
        ),
        "level_id": level_id,
        "curve_index": curve_index,
        "objects_marked_for_removal": len(all_entities),
        "transition": {
            "win_pending": True,
            "native_outcome": None,
            **transition_input,
        },
        "formal": {
            "ticks_after_transition": 1,
            "native_outcome": simulator.outcome,
        },
    }


def verify_clear_sequence(
    trajectory_path: Path,
    mutation_path: Path,
    *,
    original_root: Path,
    click_trajectory_path: Path | None = None,
    click_dmo_path: Path | None = None,
    click_provenance_path: Path | None = None,
    click_down_update: int | None = None,
    click_up_update: int | None = None,
    curve_index: int = 0,
    expected_initial_active_count: int | None = None,
    expected_initial_pending_count: int = 1,
    expected_clear_update: int | None = None,
    expected_formal_transition_update: int | None = None,
    expected_delayed_award_count: int | None = None,
    expected_award_points: int = 100,
    expected_award_period: int = 5,
    minimum_stable_tail_ticks: int = 0,
    level_id: str = "Jungle2",
    hard: bool = False,
) -> dict[str, Any]:
    """Load bound artifacts and return one auditable victory report."""

    frames = load_memory_trajectory(trajectory_path)
    report = verify_clear_sequence_frames(
        frames,
        curve_index=curve_index,
        expected_initial_active_count=expected_initial_active_count,
        expected_initial_pending_count=expected_initial_pending_count,
        expected_clear_update=expected_clear_update,
        expected_formal_transition_update=expected_formal_transition_update,
        expected_delayed_award_count=expected_delayed_award_count,
        expected_award_points=expected_award_points,
        expected_award_period=expected_award_period,
        minimum_stable_tail_ticks=minimum_stable_tail_ticks,
    )
    mutation = _load_mutation(mutation_path)
    report["intervention"] = {
        **_verify_mutation(
            mutation,
            frames[0],
            curve_index=curve_index,
        ),
        "artifact": str(mutation_path),
        "artifact_sha256": _sha256_path(mutation_path),
    }
    report["trajectory"] = {
        "artifact": str(trajectory_path),
        "artifact_sha256": _sha256_path(trajectory_path),
    }
    report["simulator"] = compare_clear_transition_with_simulator(
        frames,
        original_root=original_root,
        level_id=level_id,
        hard=hard,
        curve_index=curve_index,
    )

    click_values = (
        click_trajectory_path,
        click_dmo_path,
        click_provenance_path,
        click_down_update,
        click_up_update,
    )
    if any(value is not None for value in click_values):
        if any(value is None for value in click_values):
            raise ValueError("all click-comparison inputs must be supplied")
        assert click_trajectory_path is not None
        assert click_dmo_path is not None
        assert click_provenance_path is not None
        assert click_down_update is not None
        assert click_up_update is not None
        click_frames = load_memory_trajectory(click_trajectory_path)
        input_report = verify_clear_input_lock_frames(
            frames,
            click_frames,
            click_down_update=click_down_update,
            click_up_update=click_up_update,
        )
        provenance = _load_mutation(click_provenance_path)
        if (
            provenance.get("schema") != "zuma-rl.diagnostic-dmo-click-burst"
            or provenance.get("version") != 1
            or provenance.get("classification")
            != "diagnostic-not-unmodified-pc-evidence"
        ):
            _fail("clear_input_lock_provenance_invalid")
        source = provenance.get("source")
        output = provenance.get("output")
        transformation = provenance.get("transformation")
        if (
            not isinstance(source, dict)
            or not isinstance(output, dict)
            or not isinstance(transformation, dict)
            or output.get("sha256") != _sha256_path(click_dmo_path)[7:]
            or transformation.get("pairs")
            != [
                {
                    "down_update": click_down_update,
                    "up_update": click_up_update,
                }
            ]
        ):
            _fail("clear_input_lock_provenance_not_bound")
        input_report["artifacts"] = {
            "trajectory": {
                "artifact": str(click_trajectory_path),
                "artifact_sha256": _sha256_path(click_trajectory_path),
            },
            "dmo": {
                "artifact": str(click_dmo_path),
                "artifact_sha256": _sha256_path(click_dmo_path),
                "source_sha256": f"sha256:{source.get('sha256')}",
            },
            "provenance": {
                "artifact": str(click_provenance_path),
                "artifact_sha256": _sha256_path(click_provenance_path),
            },
        }
        report["input_lock"] = input_report
    return report


__all__ = [
    "CLEAR_SEQUENCE_SCHEMA",
    "CLEAR_SEQUENCE_VERSION",
    "PcClearSequenceError",
    "compare_clear_transition_with_simulator",
    "verify_clear_input_lock_frames",
    "verify_clear_sequence",
    "verify_clear_sequence_frames",
]
