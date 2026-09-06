"""Strict verification of the retail skull-entry loss sequence."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .pc_memory_trajectory import (
    ACTIVE_CHAIN_LIST_OFFSET,
    TrajectoryEntity,
    TrajectoryFrame,
    load_memory_trajectory,
)
from .popcap_dmo import PopCapDemo
from .revenge_core import RevengeSimulator


LOSS_SEQUENCE_SCHEMA = "zuma-rl.pc-loss-sequence-verification"
LOSS_SEQUENCE_VERSION = 1
LOSS_SIMULATOR_DIFF_SCHEMA = "zuma-rl.pc-loss-simulator-diff"
LOSS_SIMULATOR_DIFF_VERSION = 2
LOSS_SOURCE_INITIALIZATION_SCHEMA = (
    "zuma-rl.pc-loss-source-state-initialization"
)
LOSS_SOURCE_INITIALIZATION_VERSION = 1
LOSS_SOURCE_INITIALIZATION_CLASSIFICATION = (
    "exact_immutable_native_frame_reconstruction"
)
PENDING_CHAIN_LIST_OFFSET = 0x68


class PcLossSequenceError(ValueError):
    """A trajectory does not exhibit the required retail loss sequence."""


def _fail(code: str) -> None:
    raise PcLossSequenceError(code)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _active_chain(
    frame: TrajectoryFrame,
    curve_index: int,
) -> tuple[TrajectoryEntity, ...]:
    zone = f"curve:{curve_index}:list:{ACTIVE_CHAIN_LIST_OFFSET:03x}"
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
        _fail(f"loss_sequence_chain_indexes_invalid:{frame.update}")
    if len({entity.ball_id for entity in result}) != len(result):
        _fail(f"loss_sequence_duplicate_ball_id:{frame.update}")
    if any(
        entity.suck_count is None or entity.update_count is None
        for entity in result
    ):
        _fail(f"loss_sequence_raw_ball_fields_missing:{frame.update}")
    return result


def _by_id(
    chain: Sequence[TrajectoryEntity],
) -> Mapping[int, TrajectoryEntity]:
    return {entity.ball_id: entity for entity in chain}


def _pending_chain(
    frame: TrajectoryFrame,
    curve_index: int,
) -> tuple[TrajectoryEntity, ...]:
    zone = f"curve:{curve_index}:list:{PENDING_CHAIN_LIST_OFFSET:03x}"
    return tuple(
        sorted(
            (
                entity
                for entity in frame.entities
                if entity.zone == zone and entity.object_kind == "ball"
            ),
            key=lambda entity: entity.index,
        )
    )


def _infer_source_advance_speed(
    before: Sequence[TrajectoryEntity],
    after: Sequence[TrajectoryEntity],
) -> np.float32:
    """Infer the modal float32 chain advance from shared source identities."""

    before_by_id = _by_id(before)
    deltas = [
        np.float32(
            entity.curve_distance
            - before_by_id[entity.ball_id].curve_distance
        )
        for entity in after
        if entity.ball_id in before_by_id
    ]
    if not deltas:
        _fail("loss_simulator_advance_speed_unavailable")
    values, counts = np.unique(
        np.asarray(deltas, dtype=np.float32),
        return_counts=True,
    )
    return np.float32(values[int(np.argmax(counts))])


def _validate_natural_terminal_pretrigger_step(
    *,
    pretrigger_distance: float,
    advance_speed: np.float32,
    decoded_curve_end: int,
) -> float:
    """Require the natural terminal ball to advance exactly into the skull."""

    reset_distance = np.float32(
        np.float32(pretrigger_distance) + np.float32(advance_speed)
    )
    if float(reset_distance) != float(decoded_curve_end):
        _fail("loss_simulator_terminal_pretrigger_step_mismatch")
    return float(reset_distance)


def _build_loss_simulator(
    frames: Sequence[TrajectoryFrame],
    *,
    original_root: Path,
    hard: bool,
    level_id: str,
    curve_index: int,
) -> tuple[RevengeSimulator, dict[str, Any]]:
    """Build the diagnostic loss midstate without fitting its injected ball."""

    first = frames[0]
    active = _active_chain(first, curve_index)
    pending = _pending_chain(first, curve_index)
    if not active or len(frames) < 2:
        _fail("loss_simulator_initial_chain_missing")
    simulator = RevengeSimulator.from_installed(
        level_id,
        root=original_root,
        hard=hard,
        curve_index=curve_index,
        seed=0,
    )
    terminal = active[-1]
    following_active = _active_chain(frames[1], curve_index)
    inferred_advance_speed = _infer_source_advance_speed(
        active,
        following_active,
    )
    terminal_reset_distance = _validate_natural_terminal_pretrigger_step(
        pretrigger_distance=terminal.curve_distance,
        advance_speed=inferred_advance_speed,
        decoded_curve_end=int(simulator.curve.end_waypoint),
    )

    squared_errors: list[float] = []
    maximum_error = 0.0
    for entity in active[:-1]:
        point = simulator.curve.point_at_waypoint(entity.curve_distance)
        error = math.hypot(
            float(point[0]) - entity.position_x,
            float(point[1]) - entity.position_y,
        )
        squared_errors.append(error * error)
        maximum_error = max(maximum_error, error)
    if maximum_error > 0.001:
        _fail(
            "loss_simulator_curve_geometry_not_exact:"
            f"{maximum_error:.9f}"
        )
    rms_error = math.sqrt(
        sum(squared_errors) / max(1, len(squared_errors))
    )

    contacts = [
        (
            bool(entity.contact_next)
            if entity.contact_next is not None
            else (
                active[index + 1].curve_distance
                - entity.curve_distance
                <= 36.001
            )
        )
        for index, entity in enumerate(active[:-1])
    ]
    score_at_level_start = first.score_target - int(
        getattr(simulator.parameters, "zuma_score", 0)
    )
    if not 0 <= score_at_level_start <= first.score:
        _fail("loss_simulator_score_target_inconsistent")
    simulator.load_state(
        [entity.color_id for entity in active],
        [entity.curve_distance for entity in active],
        contacts,
        pending_colors=[entity.color_id for entity in pending],
        current_color=first.current_color_id,
        next_color=first.next_color_id,
        score=first.score,
        score_at_level_start=score_at_level_start,
    )
    if (
        first.qrand is None
        or first.thread_crt_rand_state is None
        or first.global_mtrand is None
        or first.native_game_time is None
    ):
        _fail("loss_simulator_initial_rng_or_timer_missing")
    simulator.load_shooter_random_state(
        crt_state=first.thread_crt_rand_state,
        update_count=first.qrand.update_count,
        selected_index=first.qrand.selected_index,
        weights=first.qrand.weights,
        sways=first.qrand.sways,
        last_hit=first.qrand.last_hit,
        previous_hit=first.qrand.previous_hit,
    )
    simulator.load_mtrand_state(
        words=first.global_mtrand.words,
        index=first.global_mtrand.index,
    )

    copied_fields = (
        "contact_next",
        "exploding",
        "explode_frame",
        "should_remove",
        "update_count",
        "suck_count",
        "backwards_count",
        "combo_count",
        "combo_score",
        "powerup_previous_type",
        "powerup_primary_type",
        "powerup_secondary_type",
        "powerup_previous_ticks",
        "powerup_lifetime_ticks",
        "powerup_transition_ticks",
        "powerup_visual_index",
    )
    float_fields = (
        "backwards_speed",
        "powerup_visual_scale",
        "powerup_visual_step",
    )
    for ball, entity in zip(simulator.balls, active, strict=True):
        ball.id = entity.ball_id
        for field_name in copied_fields:
            value = getattr(entity, field_name)
            if value is not None:
                setattr(ball, field_name, value)
        for field_name in float_fields:
            value = getattr(entity, field_name)
            if value is not None:
                setattr(ball, field_name, np.float32(value))

    simulator._next_id = max(
        entity.ball_id for entity in first.entities
    ) + 1
    simulator.num_balls_created = len(active) + len(pending)
    simulator.tick_count = first.update
    simulator.native_game_time = first.native_game_time
    simulator.advance_speed = inferred_advance_speed
    simulator.first_chain_end = simulator._first_chain_end()
    simulator.has_reached_rollout = True
    simulator.has_reached_cruising_speed = True
    simulator._have_sets = any(ball.exploding for ball in simulator.balls)

    powerup_state = first.curve_powerup_state(curve_index)
    if powerup_state is not None:
        simulator.powerup_last_any_spawn_time = (
            powerup_state.last_any_spawn_time
        )
        simulator.powerup_last_spawn_times = list(
            powerup_state.last_spawn_times
        )
        simulator.powerup_cooldown_times = list(
            powerup_state.cooldown_times
        )
        simulator.powerup_spawn_counts = list(powerup_state.spawn_counts)
        simulator.powerup_field_124_by_type = list(
            powerup_state.field_124_by_type
        )
        simulator.active_powerup_color_counts = list(
            powerup_state.active_color_counts
        )
        simulator.slow_count = powerup_state.slow_ticks
        simulator.backward_count = powerup_state.reverse_ticks
        simulator.powerup_triggered = powerup_state.powerup_triggered

    return simulator, {
        "level_id": level_id,
        "hard": hard,
        "curve_index": curve_index,
        "curve_fit_maximum_error_px": maximum_error,
        "curve_fit_rms_error_px": rms_error,
        "decoded_curve_end": int(simulator.curve.end_waypoint),
        "source_bound_terminal_ball_id": terminal.ball_id,
        "source_bound_terminal_geometry_excluded_from_curve_fit": True,
        "source_terminal_pretrigger_distance": terminal.curve_distance,
        "source_terminal_reset_distance": terminal_reset_distance,
        "terminal_pretrigger_advance_exact": True,
        "inferred_advance_speed": float(simulator.advance_speed),
        "score_at_level_start": score_at_level_start,
        "shooter_rng_state_restored": True,
        "global_mtrand_state_restored": True,
    }


def verify_loss_sequence_frames(
    frames: Sequence[TrajectoryFrame],
    *,
    curve_index: int,
    trigger_ball_id: int,
    decoded_curve_end: int,
    expected_trigger_update: int | None = None,
    expected_chain_empty_update: int | None = None,
    expected_pending_count: int = 1,
    expected_pretrigger_advance: float = 0.125,
) -> dict[str, Any]:
    """Verify native loss locking, suction motion, and complete cleanup."""

    if (
        len(frames) < 5
        or curve_index < 0
        or trigger_ball_id < 0
        or decoded_curve_end < 1
        or expected_pending_count < 0
        or not math.isfinite(expected_pretrigger_advance)
        or expected_pretrigger_advance < 0.0
    ):
        raise ValueError("loss-sequence arguments are invalid")
    for before, after in zip(frames, frames[1:]):
        if after.update != before.update + 1:
            _fail("loss_sequence_frames_not_contiguous")
    if any(
        frame.board_runtime_flag_157 is None
        or frame.board_runtime_i32_f54 is None
        or frame.native_game_time is None
        for frame in frames
    ):
        _fail("loss_sequence_board_runtime_fields_missing")

    trigger_candidates = [
        index
        for index, (before, after) in enumerate(
            zip(frames, frames[1:]),
            start=1,
        )
        if (
            before.board_runtime_flag_157 is True
            and after.board_runtime_flag_157 is False
            and before.board_runtime_i32_f54 == 0
            and after.board_runtime_i32_f54 == -1
        )
    ]
    if len(trigger_candidates) != 1:
        _fail("loss_sequence_trigger_transition_not_unique")
    trigger_index = trigger_candidates[0]
    if trigger_index < 2:
        _fail("loss_sequence_pretrigger_context_not_covered")
    trigger_frame = frames[trigger_index]
    before_frame = frames[trigger_index - 1]
    initial_frame = frames[trigger_index - 2]
    if (
        expected_trigger_update is not None
        and trigger_frame.update != expected_trigger_update
    ):
        _fail("loss_sequence_trigger_update_mismatch")
    if (
        initial_frame.native_game_time is None
        or initial_frame.native_game_time <= 0
        or before_frame.native_game_time != 0
    ):
        _fail("loss_sequence_native_time_reset_mismatch")
    for offset, frame in enumerate(frames[trigger_index - 1 :]):
        if frame.native_game_time != offset:
            _fail(
                f"loss_sequence_native_time_phase_mismatch:{frame.update}"
            )

    chains = tuple(_active_chain(frame, curve_index) for frame in frames)
    initial_chain = chains[trigger_index - 2]
    before_chain = chains[trigger_index - 1]
    trigger_chain = chains[trigger_index]
    initial_by_id = _by_id(initial_chain)
    before_by_id = _by_id(before_chain)
    trigger_by_id = _by_id(trigger_chain)
    target_initial = initial_by_id.get(trigger_ball_id)
    target_before = before_by_id.get(trigger_ball_id)
    if (
        target_initial is None
        or target_before is None
        or trigger_ball_id in trigger_by_id
        or target_initial is not initial_chain[-1]
        or target_before is not before_chain[-1]
        or target_before.curve_distance != float(decoded_curve_end)
        or not math.isclose(
            target_before.curve_distance - target_initial.curve_distance,
            expected_pretrigger_advance,
            rel_tol=0.0,
            abs_tol=1e-7,
        )
        or target_initial.suck_count != 0
        or target_before.suck_count != 0
    ):
        _fail("loss_sequence_trigger_ball_transition_invalid")

    initial_survivors = tuple(
        entity for entity in initial_chain if entity.ball_id != trigger_ball_id
    )
    before_survivors = tuple(
        entity for entity in before_chain if entity.ball_id != trigger_ball_id
    )
    if (
        tuple(entity.ball_id for entity in initial_survivors)
        != tuple(entity.ball_id for entity in before_survivors)
        or tuple(entity.ball_id for entity in before_survivors)
        != tuple(entity.ball_id for entity in trigger_chain)
    ):
        _fail("loss_sequence_pretrigger_topology_invalid")
    for initial, before, triggered in zip(
        initial_survivors,
        before_survivors,
        trigger_chain,
        strict=True,
    ):
        if (
            not math.isclose(
                before.curve_distance - initial.curve_distance,
                expected_pretrigger_advance,
                rel_tol=0.0,
                abs_tol=1e-7,
            )
            or triggered.curve_distance != before.curve_distance
            or initial.suck_count != 0
            or before.suck_count != 0
            or triggered.suck_count != 1
            or triggered.update_count != before.update_count
        ):
            _fail(
                "loss_sequence_initial_suction_state_mismatch:"
                f"{triggered.ball_id}"
            )

    for frame in frames[:trigger_index]:
        if (
            frame.board_runtime_flag_157 is not True
            or frame.board_runtime_i32_f54 != 0
        ):
            _fail("loss_sequence_pretrigger_board_state_invalid")
    for offset, frame in enumerate(frames[trigger_index:], start=1):
        if (
            frame.board_runtime_flag_157 is not False
            or frame.board_runtime_i32_f54 != -offset
        ):
            _fail(f"loss_sequence_board_countdown_mismatch:{frame.update}")

    stable_score = frames[0].score
    stable_displayed_score = frames[0].displayed_score
    initial_current = (
        initial_frame.current_ball_id,
        initial_frame.current_color_id,
    )
    initial_next = (
        initial_frame.next_ball_id,
        initial_frame.next_color_id,
    )
    stable_current = (
        before_frame.current_ball_id,
        before_frame.current_color_id,
    )
    stable_next = (
        before_frame.next_ball_id,
        before_frame.next_color_id,
    )
    if (
        initial_current[0] is None
        or initial_next[0] is None
        or stable_current[0] is None
        or stable_next[0] is None
        or (initial_current, initial_next) == (stable_current, stable_next)
    ):
        _fail("loss_sequence_shooter_chamber_missing")
    for index, frame in enumerate(frames):
        if (
            frame.score != stable_score
            or frame.displayed_score != stable_displayed_score
            or frame.list_count(curve_index, PENDING_CHAIN_LIST_OFFSET)
            != expected_pending_count
            or any(entity.zone == "fired" for entity in frame.entities)
        ):
            _fail(f"loss_sequence_locked_state_mismatch:{frame.update}")
        chamber = (
            (frame.current_ball_id, frame.current_color_id),
            (frame.next_ball_id, frame.next_color_id),
        )
        expected_chamber = (
            (initial_current, initial_next)
            if index < trigger_index - 1
            else (stable_current, stable_next)
        )
        if chamber != expected_chamber:
            _fail(f"loss_sequence_chamber_phase_mismatch:{frame.update}")
        current = [
            entity
            for entity in frame.entities
            if (
                entity.zone == "shooter_current"
                and entity.ball_id == frame.current_ball_id
            )
        ]
        if len(current) != 1 or current[0].fired is not False:
            _fail(f"loss_sequence_shooter_not_locked:{frame.update}")

    removals: dict[int, tuple[int, ...]] = {
        trigger_frame.update: (trigger_ball_id,)
    }
    for before, after, before_chain_step, after_chain_step in zip(
        frames[trigger_index:-1],
        frames[trigger_index + 1 :],
        chains[trigger_index:-1],
        chains[trigger_index + 1 :],
        strict=True,
    ):
        before_by_id_step = _by_id(before_chain_step)
        after_by_id_step = _by_id(after_chain_step)
        marked = tuple(
            entity.ball_id
            for entity in before_chain_step
            if (
                entity.suck_count == 0
                and entity.curve_distance >= float(decoded_curve_end)
            )
        )
        expected_ids = tuple(
            entity.ball_id
            for entity in before_chain_step
            if entity.ball_id not in marked
        )
        actual_ids = tuple(entity.ball_id for entity in after_chain_step)
        if actual_ids != expected_ids:
            _fail(f"loss_sequence_front_removal_mismatch:{after.update}")
        if marked:
            if marked != tuple(
                entity.ball_id
                for entity in before_chain_step[-len(marked) :]
            ):
                _fail(
                    f"loss_sequence_nonfront_ball_marked:{before.update}"
                )
            removals[after.update] = marked

        for ball_id in expected_ids:
            previous = before_by_id_step[ball_id]
            current = after_by_id_step[ball_id]
            speed = previous.suck_count >> 2
            expected_distance = previous.curve_distance + speed
            expected_suck_count = (
                0
                if expected_distance >= float(decoded_curve_end)
                else previous.suck_count + 1
            )
            if (
                current.curve_distance != expected_distance
                or current.suck_count != expected_suck_count
                or current.update_count != previous.update_count
            ):
                _fail(
                    "loss_sequence_suction_step_mismatch:"
                    f"{after.update}:{ball_id}"
                )

    empty_updates = [
        frame.update
        for frame, chain in zip(frames, chains, strict=True)
        if not chain
    ]
    if not empty_updates:
        _fail("loss_sequence_empty_chain_not_covered")
    empty_update = empty_updates[0]
    if (
        expected_chain_empty_update is not None
        and empty_update != expected_chain_empty_update
    ):
        _fail("loss_sequence_empty_update_mismatch")
    if empty_updates != list(range(empty_update, frames[-1].update + 1)):
        _fail("loss_sequence_chain_reappeared")

    removed_ids = tuple(
        ball_id
        for update in sorted(removals)
        for ball_id in removals[update]
    )
    initial_ids = tuple(entity.ball_id for entity in before_chain)
    if len(removed_ids) != len(initial_ids) or set(removed_ids) != set(
        initial_ids
    ):
        _fail("loss_sequence_removal_coverage_mismatch")

    return {
        "schema": LOSS_SEQUENCE_SCHEMA,
        "version": LOSS_SEQUENCE_VERSION,
        "status": "PASS",
        "trajectory": {
            "start_update": frames[0].update,
            "end_update": frames[-1].update,
            "tick_count": len(frames),
        },
        "trigger": {
            "update": trigger_frame.update,
            "native_time_reset_update": before_frame.update,
            "native_time_before_reset": initial_frame.native_game_time,
            "native_time_at_trigger": trigger_frame.native_game_time,
            "ball_id": trigger_ball_id,
            "decoded_curve_end": decoded_curve_end,
            "chain_count_before": len(before_chain),
            "chain_count_after": len(trigger_chain),
            "runtime_flag_157_before": (
                before_frame.board_runtime_flag_157
            ),
            "runtime_flag_157_after": (
                trigger_frame.board_runtime_flag_157
            ),
            "runtime_i32_f54_before": (
                before_frame.board_runtime_i32_f54
            ),
            "runtime_i32_f54_after": (
                trigger_frame.board_runtime_i32_f54
            ),
            "remaining_ball_initial_suck_count": 1,
        },
        "suction": {
            "speed_formula": "previous_suck_count >> 2",
            "counter_increment_per_tick": 1,
            "first_chain_empty_update": empty_update,
            "ticks_from_trigger_to_empty": (
                empty_update - trigger_frame.update
            ),
            "initial_chain_count": len(before_chain),
            "removal_updates": {
                str(update): list(ball_ids)
                for update, ball_ids in sorted(removals.items())
            },
        },
        "locked_state": {
            "pending_ball_count": expected_pending_count,
            "chamber_reset_update": before_frame.update,
            "current_ball_id_before_reset": initial_current[0],
            "next_ball_id_before_reset": initial_next[0],
            "current_ball_id": stable_current[0],
            "next_ball_id": stable_next[0],
            "score": stable_score,
            "displayed_score": stable_displayed_score,
            "fired_bullet_count": 0,
        },
        "post_empty_ticks_covered": frames[-1].update - empty_update,
    }


def verify_loss_sequence(
    trajectory_path: Path,
    dmo_path: Path,
    **kwargs: Any,
) -> dict[str, Any]:
    """Load and verify loss artifacts while retaining provenance."""

    trajectory_path = trajectory_path.resolve()
    dmo_path = dmo_path.resolve()
    frames = load_memory_trajectory(trajectory_path)
    report = verify_loss_sequence_frames(frames, **kwargs)
    trigger_update = report["trigger"]["update"]
    demo = PopCapDemo.read(dmo_path)
    ignored_downs = [
        command.update
        for command in demo.commands
        if (
            trigger_update <= command.update <= frames[-1].update
            and command.kind == "mouse_button"
            and command.payload.get("button") == 1
            and command.payload.get("down") is True
        )
    ]
    if not ignored_downs:
        _fail("loss_sequence_no_post_trigger_click_covered")
    report["trajectory"] = {
        **report["trajectory"],
        "artifact": str(trajectory_path),
        "artifact_sha256": _sha256_path(trajectory_path),
    }
    report["dmo"] = {
        "artifact": str(dmo_path),
        "artifact_bytes": demo.artifact_bytes,
        "artifact_sha256": demo.artifact_sha256,
        "ignored_left_down_updates": ignored_downs,
        "classification": "diagnostic-not-unmodified-pc-evidence",
    }
    return report


def compare_loss_sequence_with_simulator(
    frames: Sequence[TrajectoryFrame],
    *,
    original_root: Path,
    hard: bool,
    level_id: str | None,
    curve_index: int,
) -> dict[str, Any]:
    """Transplant a PC pre-loss midstate and require an exact core replay."""

    if len(frames) < 3 or curve_index < 0:
        raise ValueError("loss simulator diff arguments are invalid")
    if level_id is None:
        raise ValueError(
            "loss simulator diff requires an explicit installed level"
        )
    simulator, scenario = _build_loss_simulator(
        frames,
        original_root=original_root,
        hard=hard,
        level_id=level_id,
        curve_index=curve_index,
    )
    first = frames[0]

    first_chain = _active_chain(first, curve_index)
    if (
        simulator.tick_count != first.update
        or len(simulator.balls) != len(first_chain)
        or tuple(ball.id for ball in simulator.balls)
        != tuple(entity.ball_id for entity in first_chain)
    ):
        _fail("loss_simulator_initial_transplant_mismatch")
    first_pending = first.list_count(
        curve_index,
        PENDING_CHAIN_LIST_OFFSET,
    )
    if (
        len(simulator.pending_colors) != first_pending
        or simulator.score != first.score
        or simulator.current_color != first.current_color_id
        or simulator.next_color != first.next_color_id
    ):
        _fail("loss_simulator_initial_locked_state_mismatch")

    maximum_waypoint_error = 0.0
    compared_ticks = 0
    reset_update: int | None = None
    trigger_update: int | None = None
    empty_update: int | None = None
    removal_updates: dict[str, list[int]] = {}
    previous_ids = tuple(ball.id for ball in simulator.balls)

    for frame in frames[1:]:
        events = simulator.tick()
        compared_ticks += 1
        if simulator.tick_count != frame.update:
            _fail(f"loss_simulator_update_phase_mismatch:{frame.update}")

        chain = _active_chain(frame, curve_index)
        simulator_ids = tuple(ball.id for ball in simulator.balls)
        pc_ids = tuple(entity.ball_id for entity in chain)
        if simulator_ids != pc_ids:
            _fail(f"loss_simulator_chain_identity_mismatch:{frame.update}")
        if len(simulator.balls) != len(chain):
            _fail(f"loss_simulator_chain_count_mismatch:{frame.update}")

        removed_ids = [
            ball_id for ball_id in previous_ids if ball_id not in simulator_ids
        ]
        expected_removed = len(previous_ids) - len(pc_ids)
        if (
            events.balls_removed != expected_removed
            or len(removed_ids) != expected_removed
        ):
            _fail(f"loss_simulator_removal_event_mismatch:{frame.update}")
        if removed_ids:
            removal_updates[str(frame.update)] = removed_ids
        previous_ids = simulator_ids

        for ball, entity in zip(simulator.balls, chain, strict=True):
            error = abs(float(ball.waypoint) - entity.curve_distance)
            maximum_waypoint_error = max(maximum_waypoint_error, error)
            if (
                error != 0.0
                or ball.color != entity.color_id
                or (
                    entity.contact_next is not None
                    and ball.contact_next != entity.contact_next
                )
                or (
                    entity.suck_count is not None
                    and ball.suck_count != entity.suck_count
                )
                or (
                    entity.update_count is not None
                    and ball.update_count != entity.update_count
                )
            ):
                _fail(
                    "loss_simulator_ball_state_mismatch:"
                    f"{frame.update}:{entity.ball_id}"
                )

        if (
            len(simulator.pending_colors)
            != frame.list_count(curve_index, PENDING_CHAIN_LIST_OFFSET)
            or simulator.score != frame.score
            or simulator.current_color != frame.current_color_id
            or simulator.next_color != frame.next_color_id
            or simulator.native_game_time != frame.native_game_time
        ):
            _fail(f"loss_simulator_locked_state_mismatch:{frame.update}")

        if frame.board_runtime_flag_157 is True:
            if frame.board_runtime_i32_f54 != 0:
                _fail(f"loss_simulator_pc_board_phase_invalid:{frame.update}")
            if frame.native_game_time == 0:
                reset_update = frame.update
                if (
                    not simulator.skull_entry_pending
                    or simulator.loss_started
                    or events.loss_started
                ):
                    _fail(
                        "loss_simulator_transition_state_mismatch:"
                        f"{frame.update}"
                    )
        elif frame.board_runtime_flag_157 is False:
            expected_elapsed = -int(frame.board_runtime_i32_f54 or 0)
            if (
                expected_elapsed < 1
                or not simulator.loss_started
                or simulator.skull_entry_pending
                or simulator.loss_elapsed_ticks != expected_elapsed
            ):
                _fail(
                    f"loss_simulator_suction_phase_mismatch:{frame.update}"
                )
            if trigger_update is None:
                trigger_update = frame.update
                if not events.loss_started:
                    _fail("loss_simulator_trigger_event_missing")
            elif events.loss_started:
                _fail("loss_simulator_trigger_event_repeated")
        else:
            _fail(f"loss_simulator_pc_board_state_missing:{frame.update}")

        if not chain:
            empty_update = frame.update
            if simulator.outcome != "loss":
                _fail("loss_simulator_terminal_outcome_missing")
            break
        if simulator.outcome is not None:
            _fail(f"loss_simulator_terminal_outcome_early:{frame.update}")

    if (
        reset_update is None
        or trigger_update is None
        or empty_update is None
    ):
        _fail("loss_simulator_required_phase_not_covered")

    return {
        "schema": LOSS_SIMULATOR_DIFF_SCHEMA,
        "version": LOSS_SIMULATOR_DIFF_VERSION,
        "status": "PASS",
        "start_update": first.update,
        "end_update": empty_update,
        "ticks_compared": compared_ticks,
        "maximum_waypoint_error": maximum_waypoint_error,
        "reset_update": reset_update,
        "trigger_update": trigger_update,
        "empty_update": empty_update,
        "removal_updates": removal_updates,
        "initial_chain_count": len(first_chain),
        "pending_ball_count": first_pending,
        "score": first.score,
        "shooter_colors_after_reset": [
            frames[1].current_color_id,
            frames[1].next_color_id,
        ],
        "post_empty_pc_ticks_excluded": frames[-1].update - empty_update,
        "scenario": dict(scenario),
        "source_state_initialization": {
            "schema": LOSS_SOURCE_INITIALIZATION_SCHEMA,
            "version": LOSS_SOURCE_INITIALIZATION_VERSION,
            "status": "PASS",
            "classification": LOSS_SOURCE_INITIALIZATION_CLASSIFICATION,
            "source_frame_update": first.update,
            "active_chain_entity_count": len(first_chain),
            "pending_chain_entity_count": first_pending,
            "terminal_ball_id": scenario["source_bound_terminal_ball_id"],
            "all_initial_entities_copied_from_one_native_frame": True,
            "terminal_geometry_excluded_only_from_curve_fit": True,
            "terminal_pretrigger_step_verified_against_installed_curve": True,
            "process_memory_writes": 0,
            "source_artifact_modified": False,
        },
    }


def validate_loss_simulator_diff_report(
    report: Mapping[str, Any],
    *,
    evidence_root: str | Path,
    original_root: str | Path,
) -> str | None:
    """Recompute a version-2 loss differential from its immutable source.

    The native terminal ball is an ordinary member of the selected source
    frame.  Version 2 makes that source-bound initialization explicit and
    verifies the complete report again instead of granting a generic
    exception to the synthetic-state guard.
    """

    try:
        if (
            report.get("schema") != LOSS_SIMULATOR_DIFF_SCHEMA
            or report.get("version") != LOSS_SIMULATOR_DIFF_VERSION
            or report.get("status") != "PASS"
        ):
            return "loss simulator diff schema, version, or status is invalid"
        initialization = report.get("source_state_initialization")
        scenario = report.get("scenario")
        trajectory = report.get("trajectory")
        oracle = report.get("oracle")
        if (
            not isinstance(initialization, Mapping)
            or initialization.get("schema")
            != LOSS_SOURCE_INITIALIZATION_SCHEMA
            or initialization.get("version")
            != LOSS_SOURCE_INITIALIZATION_VERSION
            or initialization.get("status") != "PASS"
            or initialization.get("classification")
            != LOSS_SOURCE_INITIALIZATION_CLASSIFICATION
            or initialization.get(
                "all_initial_entities_copied_from_one_native_frame"
            )
            is not True
            or initialization.get(
                "terminal_geometry_excluded_only_from_curve_fit"
            )
            is not True
            or initialization.get(
                "terminal_pretrigger_step_verified_against_installed_curve"
            )
            is not True
            or initialization.get("process_memory_writes") != 0
            or initialization.get("source_artifact_modified") is not False
            or not isinstance(scenario, Mapping)
            or scenario.get(
                "source_bound_terminal_geometry_excluded_from_curve_fit"
            )
            is not True
            or scenario.get("terminal_pretrigger_advance_exact") is not True
            or not isinstance(trajectory, Mapping)
            or not isinstance(oracle, Mapping)
        ):
            return "loss source-state initialization contract is invalid"

        artifact = trajectory.get("artifact")
        artifact_sha256 = trajectory.get("artifact_sha256")
        if not isinstance(artifact, str) or not artifact:
            return "loss source trajectory path is invalid"
        relative = Path(artifact)
        if relative.is_absolute() or "\\" in artifact or ".." in relative.parts:
            return "loss source trajectory is not evidence-root relative"
        root = Path(evidence_root).resolve(strict=True)
        trajectory_path = (root / relative).resolve(strict=True)
        try:
            trajectory_path.relative_to(root)
        except ValueError:
            return "loss source trajectory escapes the evidence root"
        if _sha256_path(trajectory_path) != artifact_sha256:
            return "loss source trajectory SHA-256 differs"

        captured_frames = load_memory_trajectory(trajectory_path)
        captured_start = trajectory.get("captured_start_update")
        captured_end = trajectory.get("captured_end_update")
        selected_start = trajectory.get("selected_start_update")
        selected_end = trajectory.get("selected_end_update")
        integer_boundaries = (
            captured_start,
            captured_end,
            selected_start,
            selected_end,
        )
        if (
            not captured_frames
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in integer_boundaries
            )
            or captured_frames[0].update != captured_start
            or captured_frames[-1].update != captured_end
            or selected_start < captured_start
            or selected_end > captured_end
            or selected_end <= selected_start
            or report.get("start_update") != selected_start
            or report.get("end_update") != selected_end
        ):
            return "loss source trajectory boundaries are invalid"
        selected = tuple(
            frame
            for frame in captured_frames
            if selected_start <= frame.update <= selected_end
        )
        if (
            len(selected) != selected_end - selected_start + 1
            or selected[0].update != selected_start
            or selected[-1].update != selected_end
            or any(
                after.update != before.update + 1
                for before, after in zip(selected, selected[1:])
            )
        ):
            return "loss source trajectory selection is not contiguous"

        trigger_ball_id = scenario.get("source_bound_terminal_ball_id")
        decoded_curve_end = scenario.get("decoded_curve_end")
        curve_index = scenario.get("curve_index")
        expected_advance = scenario.get("inferred_advance_speed")
        expected_trigger = report.get("trigger_update")
        expected_empty = report.get("empty_update")
        expected_pending = report.get("pending_ball_count")
        if (
            any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in (
                    trigger_ball_id,
                    decoded_curve_end,
                    curve_index,
                    expected_trigger,
                    expected_empty,
                    expected_pending,
                )
            )
            or not isinstance(expected_advance, (int, float))
            or isinstance(expected_advance, bool)
            or not math.isfinite(float(expected_advance))
            or expected_advance <= 0.0
            or initialization.get("source_frame_update") != selected_start
            or initialization.get("terminal_ball_id") != trigger_ball_id
        ):
            return "loss source-state parameters are invalid"

        live_oracle = verify_loss_sequence_frames(
            selected,
            curve_index=curve_index,
            trigger_ball_id=trigger_ball_id,
            decoded_curve_end=decoded_curve_end,
            expected_trigger_update=expected_trigger,
            expected_chain_empty_update=expected_empty,
            expected_pending_count=expected_pending,
            expected_pretrigger_advance=float(expected_advance),
        )
        live_report = compare_loss_sequence_with_simulator(
            selected,
            original_root=Path(original_root).resolve(strict=True),
            hard=scenario.get("hard") is True,
            level_id=scenario.get("level_id"),
            curve_index=curve_index,
        )
        live_report["trajectory"] = {
            "artifact": relative.as_posix(),
            "artifact_sha256": artifact_sha256,
            "captured_start_update": captured_frames[0].update,
            "captured_end_update": captured_frames[-1].update,
            "selected_start_update": selected[0].update,
            "selected_end_update": selected[-1].update,
        }
        live_report["oracle"] = {
            "schema": live_oracle["schema"],
            "status": live_oracle["status"],
            "trigger_update": live_oracle["trigger"]["update"],
            "empty_update": live_oracle["suction"][
                "first_chain_empty_update"
            ],
            "ticks_from_trigger_to_empty": live_oracle["suction"][
                "ticks_from_trigger_to_empty"
            ],
        }
        if dict(report) != live_report:
            return "loss simulator diff differs from live recomputation"
    except (KeyError, OSError, PcLossSequenceError, TypeError, ValueError):
        return "loss simulator diff could not be recomputed"
    return None


__all__ = [
    "LOSS_SIMULATOR_DIFF_SCHEMA",
    "LOSS_SIMULATOR_DIFF_VERSION",
    "LOSS_SOURCE_INITIALIZATION_CLASSIFICATION",
    "LOSS_SOURCE_INITIALIZATION_SCHEMA",
    "LOSS_SOURCE_INITIALIZATION_VERSION",
    "LOSS_SEQUENCE_SCHEMA",
    "LOSS_SEQUENCE_VERSION",
    "PcLossSequenceError",
    "compare_loss_sequence_with_simulator",
    "validate_loss_simulator_diff_report",
    "verify_loss_sequence",
    "verify_loss_sequence_frames",
]
