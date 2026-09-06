"""Replay retail DMO input from a frozen PC midstate and diff every tick."""

from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path
import struct
from typing import Any, Mapping, Sequence

import numpy as np

from zuma_rl.pc_memory_trajectory import (
    ACTIVE_CHAIN_LIST_OFFSET,
    BULLET_CURVE_POINT_COUNT,
    INSERTION_STAGING_LIST_OFFSET,
    PcMemoryTrajectoryError,
    RETAIL_DEBUG_FILL_I32,
    TrajectoryEntity,
    TrajectoryFrame,
    TrajectoryFruitState,
    TrajectoryMTRandState,
    TrajectoryQRandState,
)
from zuma_rl.pc_merge_diff import (
    _active,
    _entities_in_list,
    _identify_level_curve,
    _infer_speed,
)
from zuma_rl.popcap_dmo import PopCapDemo
from zuma_rl.revenge_core import (
    PopCapMTRandom,
    PowerupSpawnCalibration,
    Projectile,
    RevengeSimulator,
)


GAMEPLAY_DIFF_SCHEMA = "zuma-rl.pc-gameplay-simulator-diff"
GAMEPLAY_DIFF_VERSION = 5

EXTERNAL_MTRAND_POLICY = (
    "conditional_external_mtrand_reconciliation_v2"
)


def _observed_chain_movement_delta(
    first: TrajectoryFrame,
    second: TrajectoryFrame,
) -> np.float32:
    """Return the modal signed waypoint delta between adjacent PC frames."""

    before = {entity.ball_id: entity for entity in _active(first)}
    deltas = [
        np.float32(
            entity.curve_distance
            - before[entity.ball_id].curve_distance
        )
        for entity in _active(second)
        if entity.ball_id in before
    ]
    if not deltas:
        raise PcMemoryTrajectoryError("gameplay_diff_movement_unavailable")
    values, counts = np.unique(
        np.asarray(deltas, dtype=np.float32),
        return_counts=True,
    )
    movement = values[int(np.argmax(counts))]
    if not bool(np.isfinite(movement)):
        raise PcMemoryTrajectoryError("gameplay_diff_movement_invalid")
    return np.float32(movement)


def _normalized_qrand_vectors(
    state: TrajectoryQRandState,
    *,
    size: int,
) -> tuple[
    tuple[float, ...],
    tuple[float, ...],
    tuple[int, ...],
    tuple[int, ...],
]:
    """Map retail's unallocated QRand vectors to its zero-filled state."""

    vectors = (
        state.weights,
        state.sways,
        state.last_hit,
        state.previous_hit,
    )
    lengths = tuple(len(values) for values in vectors)
    if lengths == (0, 0, 0, 0):
        if state.update_count != 0 or state.selected_index != -1:
            raise PcMemoryTrajectoryError(
                "gameplay_diff_uninitialized_qrand_state_invalid"
            )
        return (
            (0.0,) * size,
            (0.0,) * size,
            (0,) * size,
            (0,) * size,
        )
    if lengths != (size, size, size, size):
        raise PcMemoryTrajectoryError(
            "gameplay_diff_qrand_vector_length_mismatch"
        )
    return vectors


def _restore_shooter_random_state(
    simulator: RevengeSimulator,
    *,
    state: TrajectoryQRandState,
    crt_state: int,
) -> None:
    weights, sways, last_hit, previous_hit = (
        _normalized_qrand_vectors(
            state,
            size=simulator._color_chooser.size,
        )
    )
    if not state.weights:
        chooser = simulator._color_chooser
        if (
            chooser.update_count != 0
            or chooser.selected_index != -1
            or chooser.allowed_support is not None
            or any(float(value) != 0.0 for value in chooser.weights)
            or any(float(value) != 0.0 for value in chooser.sways)
            or any(int(value) != 0 for value in chooser.last_hit)
            or any(int(value) != 0 for value in chooser.previous_hit)
        ):
            raise PcMemoryTrajectoryError(
                "gameplay_diff_qrand_constructor_state_mismatch"
            )
        simulator.crt_rng.state = crt_state
        return
    simulator.load_shooter_random_state(
        crt_state=crt_state,
        update_count=state.update_count,
        selected_index=state.selected_index,
        weights=weights,
        sways=sways,
        last_hit=last_hit,
        previous_hit=previous_hit,
    )


def _restore_global_mtrand_state(
    simulator: RevengeSimulator,
    *,
    state: TrajectoryMTRandState,
) -> None:
    """Restore the read-only retail global gameplay RNG snapshot exactly."""

    simulator.load_mtrand_state(words=state.words, index=state.index)


def _mtrand_state_sha256(
    words: Sequence[int],
    index: int,
) -> str:
    """Content-address one complete 624-word retail MT state."""

    if (
        len(words) != 624
        or isinstance(index, bool)
        or not isinstance(index, int)
        or not 0 <= index <= 624
        or any(
            isinstance(word, bool)
            or not isinstance(word, (int, np.integer))
            or not 0 <= int(word) <= 0xFFFFFFFF
            for word in words
        )
    ):
        raise PcMemoryTrajectoryError(
            "gameplay_diff_mtrand_state_invalid"
        )
    payload = struct.pack(
        "<625I",
        *(int(word) for word in words),
        index,
    )
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _rng_state_sha256(rng: PopCapMTRandom) -> str:
    return _mtrand_state_sha256(rng.words, rng.index)


def _trajectory_mtrand_state_sha256(
    state: TrajectoryMTRandState,
) -> str:
    return _mtrand_state_sha256(state.words, state.index)


def _mtrand_reconciliation_contract(
    rows: Sequence[Mapping[str, Any]],
    *,
    maximum_draws_per_tick: int,
) -> tuple[str, dict[str, Any]]:
    """Describe what occurred, independently of whether search was enabled."""

    reconciliations = list(rows)
    reconciled = bool(reconciliations)
    return (
        EXTERNAL_MTRAND_POLICY if reconciled else "exact_no_reconciliation",
        {
            "enabled": reconciled,
            "classification": (
                "unclassified_until_bound_call_trace_proof"
                if reconciled
                else "no_reconciliation"
            ),
            "state_binding": "sha256_of_624_words_plus_index_le_u32",
            "maximum_draws_per_tick": maximum_draws_per_tick,
            "reconciled_tick_count": len(reconciliations),
            "total_reconciled_draw_count": sum(
                int(row["draw_count"]) for row in reconciliations
            ),
            "rows": reconciliations,
        },
    )


def _reconcile_external_mtrand_draws(
    rng: PopCapMTRandom,
    *,
    observed: TrajectoryMTRandState,
    maximum_draws: int,
) -> int | None:
    """Advance to an observed state, restoring the input state on failure.

    This helper says only whether the observed state is a short forward suffix
    of the simulator stream.  The caller is responsible for proving that all
    gameplay state already matches before classifying the skipped draws as
    rendering-only consumers.
    """

    if (
        isinstance(maximum_draws, bool)
        or not isinstance(maximum_draws, int)
        or maximum_draws <= 0
    ):
        raise ValueError("maximum_draws must be a positive integer")
    initial_words, initial_index = rng.state
    for draw_count in range(maximum_draws + 1):
        if rng.index == observed.index and rng.words == observed.words:
            return draw_count
        if draw_count < maximum_draws:
            rng.next_u31()
    rng.load_state(initial_words, initial_index)
    return None


def _pc_projectiles(
    frame: TrajectoryFrame,
    *,
    staging: bool,
) -> tuple[TrajectoryEntity, ...]:
    if staging:
        return _entities_in_list(frame, INSERTION_STAGING_LIST_OFFSET)
    return tuple(
        sorted(
            (entity for entity in frame.entities if entity.zone == "fired"),
            key=lambda entity: entity.index,
        )
    )


def _restore_initial_free_projectiles(
    simulator: RevengeSimulator,
    rows: Sequence[TrajectoryEntity],
) -> None:
    """Restore raw-memory-bound free bullets without partial mutation.

    Retail keeps one gap-scan waypoint per possible CurveMgr slot.  Unused
    slots can still contain the debug-heap fill ``0xBAADF00D``; that value is
    semantically equivalent to the simulator's absent/default-zero latch and
    must never be interpreted as a real waypoint.
    """

    if simulator.curve_count > BULLET_CURVE_POINT_COUNT:
        raise PcMemoryTrajectoryError(
            "gameplay_diff_initial_projectile_curve_count_unsupported"
        )
    prepared: list[Projectile] = []
    occupied_ids = {ball.id for ball in simulator.balls}
    for row in rows:
        if row.object_kind != "bullet" or row.zone != "fired":
            raise PcMemoryTrajectoryError(
                "gameplay_diff_initial_projectile_kind_invalid"
            )
        required = (
            row.radius,
            row.velocity_x,
            row.velocity_y,
            row.merge_progress,
            row.merge_speed,
            row.fired,
            row.gap_list_sentinel_address,
            row.gap_entry_count,
            row.curve_points,
        )
        if any(value is None for value in required):
            raise PcMemoryTrajectoryError(
                "gameplay_diff_initial_projectile_state_incomplete"
            )
        assert row.radius is not None
        assert row.velocity_x is not None
        assert row.velocity_y is not None
        assert row.merge_progress is not None
        assert row.merge_speed is not None
        assert row.fired is not None
        assert row.gap_list_sentinel_address is not None
        assert row.gap_entry_count is not None
        assert row.curve_points is not None
        numeric_values = (
            row.position_x,
            row.position_y,
            row.curve_distance,
            row.radius,
            row.velocity_x,
            row.velocity_y,
            row.merge_progress,
            row.merge_speed,
        )
        if not all(math.isfinite(value) for value in numeric_values):
            raise PcMemoryTrajectoryError(
                "gameplay_diff_initial_projectile_numeric_state_invalid"
            )
        if (
            row.radius != float(simulator.config.ball_radius)
            or row.merge_progress < 0.0
            or row.merge_progress >= 1.0
            or row.merge_speed < 0.0
            or row.gap_list_sentinel_address <= 0
            or row.gap_entry_count != len(row.gap_entries)
            or len(row.curve_points) != BULLET_CURVE_POINT_COUNT
            or not 0 <= row.color_id < simulator.active_num_colors
        ):
            raise PcMemoryTrajectoryError(
                "gameplay_diff_initial_projectile_state_invalid"
            )
        if row.ball_id in occupied_ids:
            raise PcMemoryTrajectoryError(
                "gameplay_diff_initial_projectile_identity_collision"
            )
        occupied_ids.add(row.ball_id)

        curve_points: dict[int, int] = {}
        for index, waypoint in enumerate(row.curve_points):
            if index >= simulator.curve_count:
                if waypoint not in {0, RETAIL_DEBUG_FILL_I32}:
                    raise PcMemoryTrajectoryError(
                        "gameplay_diff_initial_projectile_curve_slot_unavailable"
                    )
                continue
            if waypoint != RETAIL_DEBUG_FILL_I32:
                curve_points[index] = waypoint
        if any(
            curve_index >= simulator.curve_count
            for curve_index, _, _ in row.gap_entries
        ):
            raise PcMemoryTrajectoryError(
                "gameplay_diff_initial_projectile_gap_curve_unavailable"
            )
        active_curve_point = curve_points.get(
            simulator.active_curve_index,
            0,
        )
        prepared.append(
            Projectile(
                id=row.ball_id,
                color=row.color_id,
                position=np.asarray(
                    (row.position_x, row.position_y),
                    dtype=np.float32,
                ),
                velocity=np.asarray(
                    (row.velocity_x, row.velocity_y),
                    dtype=np.float32,
                ),
                radius=simulator.config.ball_radius,
                just_fired=row.fired,
                hit_percent=np.float32(row.merge_progress),
                merge_speed=np.float32(row.merge_speed),
                waypoint=np.float32(row.curve_distance),
                curve_point=active_curve_point,
                curve_points=curve_points,
                gap_info=[
                    (boundary_ball_id, gap_distance)
                    for _, gap_distance, boundary_ball_id in row.gap_entries
                ],
            )
        )

    # Do not expose a partially restored list if any later row was invalid.
    simulator.free_projectiles = prepared


def _float32_bits(value: float | np.floating[Any]) -> bytes:
    return struct.pack("<f", float(np.float32(value)))


def _restore_fruit_state(
    simulator: RevengeSimulator,
    state: TrajectoryFruitState,
) -> None:
    calibration = simulator.fruit_calibration
    if calibration is None:
        raise PcMemoryTrajectoryError(
            "gameplay_diff_fruit_state_without_calibration"
        )
    if not state.active:
        # Retail initialization uses -1 while the natural expiry/collection
        # clear paths write 0.  Presence is governed solely by the pointer at
        # Board+0x0B4, so both inactive selector residues are legitimate.
        if state.selected_point_index not in {-1, 0} or state.collecting:
            raise PcMemoryTrajectoryError(
                "gameplay_diff_initial_inactive_fruit_state_invalid"
            )
        active_point_index: int | None = None
    else:
        active_point_index = state.selected_point_index
        if not 0 <= active_point_index < len(calibration.points):
            raise PcMemoryTrajectoryError(
                "gameplay_diff_initial_fruit_point_index_invalid"
            )
        if state.collecting:
            # The Board image does not contain the subordinate PopAnim frame
            # needed to restore a mid-collection boundary without inference.
            raise PcMemoryTrajectoryError(
                "gameplay_diff_initial_fruit_collection_phase_unavailable"
            )
    simulator.fruit_active_point_index = active_point_index
    simulator.fruit_collecting = state.collecting
    simulator.fruit_expiry_time = state.expiry_time
    simulator.fruit_velocity = np.float32(state.velocity)
    simulator.fruit_max_velocity = np.float32(state.max_velocity)
    simulator.fruit_acceleration = np.float32(state.acceleration)
    simulator.fruit_vertical_offset = np.float32(state.vertical_offset)
    simulator.fruit_lower_bound = np.float32(state.lower_bound)
    simulator.fruit_upper_bound = np.float32(state.upper_bound)
    simulator.fruit_glow_alpha = state.glow_alpha
    simulator.fruit_glow_step = state.glow_step
    simulator.fruit_alpha = state.alpha
    simulator.fruit_cell_index = state.cell_index
    simulator.fruit_collection_ticks_remaining = 0


def _fruit_state_mismatches(
    simulator: RevengeSimulator,
    state: TrajectoryFruitState | None,
) -> dict[str, int]:
    if state is None:
        return {}
    calibration = simulator.fruit_calibration
    if calibration is None:
        return {"fruit_calibration": 1}
    simulator_active = simulator.fruit_active_point_index is not None
    mismatches: dict[str, int] = {}

    def check(name: str, observed: Any, expected: Any) -> None:
        if observed != expected:
            mismatches[name] = 1

    check("fruit_active", simulator_active, state.active)
    if simulator_active and state.active:
        check(
            "fruit_selected_point_index",
            simulator.fruit_active_point_index,
            state.selected_point_index,
        )
    check("fruit_collecting", simulator.fruit_collecting, state.collecting)
    for name, observed, expected in (
        ("fruit_velocity", simulator.fruit_velocity, state.velocity),
        ("fruit_max_velocity", simulator.fruit_max_velocity, state.max_velocity),
        ("fruit_acceleration", simulator.fruit_acceleration, state.acceleration),
        ("fruit_vertical_offset", simulator.fruit_vertical_offset, state.vertical_offset),
        ("fruit_lower_bound", simulator.fruit_lower_bound, state.lower_bound),
        ("fruit_upper_bound", simulator.fruit_upper_bound, state.upper_bound),
    ):
        if _float32_bits(observed) != _float32_bits(expected):
            mismatches[name] = 1
    for name, observed, expected in (
        ("fruit_glow_alpha", simulator.fruit_glow_alpha, state.glow_alpha),
        ("fruit_glow_step", simulator.fruit_glow_step, state.glow_step),
        ("fruit_alpha", simulator.fruit_alpha, state.alpha),
        ("fruit_expiry_time", simulator.fruit_expiry_time, state.expiry_time),
        ("fruit_cell_index", simulator.fruit_cell_index, state.cell_index),
    ):
        check(name, observed, expected)
    return mismatches


def _transplant_midstate(
    frames: Sequence[TrajectoryFrame],
    *,
    original_root: Path,
    hard: bool,
    level_id: str | None,
    curve_index: int | None,
) -> tuple[RevengeSimulator, Mapping[str, Any]]:
    if len(frames) < 2:
        raise PcMemoryTrajectoryError("gameplay_diff_window_too_short")
    first = frames[0]
    active = _active(first)
    initial_free_projectiles = _pc_projectiles(first, staging=False)
    initial_merging_projectiles = _pc_projectiles(first, staging=True)
    if not active or initial_merging_projectiles:
        raise PcMemoryTrajectoryError(
            "gameplay_diff_initial_state_not_quiescent"
        )
    (
        identified_level,
        identified_curve,
        curve_fit_maximum,
        curve_fit_rms,
    ) = _identify_level_curve(
        active,
        original_root=original_root,
        hard=hard,
        level_id=level_id,
        curve_index=curve_index,
    )
    powerup_calibration = (
        PowerupSpawnCalibration.jungle2_retail_v1()
        if identified_level.casefold() == "jungle2"
        else None
    )
    simulator = RevengeSimulator.from_installed(
        identified_level,
        root=original_root,
        hard=hard,
        curve_index=identified_curve,
        seed=0,
        powerup_calibration=powerup_calibration,
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
    pending = _entities_in_list(first, 0x68)
    score_at_level_start = first.score_target - int(
        getattr(simulator.parameters, "zuma_score", 0)
    )
    if not 0 <= score_at_level_start <= first.score:
        raise PcMemoryTrajectoryError(
            "gameplay_diff_score_target_inconsistent"
        )
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
    global_mtrand_available = first.global_mtrand is not None
    if any(
        (frame.global_mtrand is not None) != global_mtrand_available
        for frame in frames
    ):
        raise PcMemoryTrajectoryError(
            "gameplay_diff_global_mtrand_state_incomplete"
        )
    if first.global_mtrand is not None:
        _restore_global_mtrand_state(
            simulator,
            state=first.global_mtrand,
        )
    if (first.qrand is None) != (
        first.thread_crt_rand_state is None
    ):
        raise PcMemoryTrajectoryError(
            "gameplay_diff_initial_shooter_rng_incomplete"
        )
    if (
        first.qrand is not None
        and first.thread_crt_rand_state is not None
    ):
        _restore_shooter_random_state(
            simulator,
            state=first.qrand,
            crt_state=first.thread_crt_rand_state,
        )
    if first.native_game_time is not None:
        simulator.native_game_time = first.native_game_time
    fruit_runtime_derived_from_initial_boundary = False
    fruit_runtime_state_restored = False
    if simulator.fruit_calibration is not None:
        if first.native_game_time is None:
            raise PcMemoryTrajectoryError(
                "gameplay_diff_fruit_runtime_requires_native_time"
            )
        if first.fruit_state is not None:
            _restore_fruit_state(simulator, first.fruit_state)
            fruit_runtime_state_restored = True
        else:
            if (
                first.native_game_time
                > simulator.fruit_calibration.frequency_ticks
            ):
                raise PcMemoryTrajectoryError(
                    "gameplay_diff_fruit_runtime_unavailable_after_first_eligible_tick"
                )
            # Legacy evidence does not expose fruit fields.  Before the first
            # eligible scheduler tick, constructor state is still exact.
            fruit_runtime_derived_from_initial_boundary = True
    native_powerup_state = first.curve_powerup_state(identified_curve)
    if native_powerup_state is not None:
        simulator.powerup_last_any_spawn_time = (
            native_powerup_state.last_any_spawn_time
        )
        simulator.powerup_last_spawn_times = list(
            native_powerup_state.last_spawn_times
        )
        simulator.powerup_cooldown_times = list(
            native_powerup_state.cooldown_times
        )
        simulator.powerup_spawn_counts = list(
            native_powerup_state.spawn_counts
        )
        simulator.powerup_field_124_by_type = list(
            native_powerup_state.field_124_by_type
        )
        simulator.active_powerup_color_counts = list(
            native_powerup_state.active_color_counts
        )
        simulator.powerup_triggered = (
            native_powerup_state.powerup_triggered
        )
        simulator.slow_count = native_powerup_state.slow_ticks
        simulator.backward_count = native_powerup_state.reverse_ticks
        simulator.last_powerup_waypoint = (
            native_powerup_state.last_powerup_waypoint
        )
    for ball, entity in zip(simulator.balls, active, strict=True):
        ball.id = entity.ball_id
        if entity.contact_next is not None:
            ball.contact_next = entity.contact_next
        if entity.exploding is not None:
            ball.exploding = entity.exploding
        if entity.explode_frame is not None:
            ball.explode_frame = entity.explode_frame
        if entity.should_remove is not None:
            ball.should_remove = entity.should_remove
        if entity.update_count is not None:
            ball.update_count = entity.update_count
        if entity.suck_count is not None:
            ball.suck_count = entity.suck_count
        if entity.backwards_count is not None:
            ball.backwards_count = entity.backwards_count
        if entity.backwards_speed is not None:
            ball.backwards_speed = np.float32(entity.backwards_speed)
        if entity.combo_count is not None:
            ball.combo_count = entity.combo_count
        if entity.combo_score is not None:
            ball.combo_score = entity.combo_score
        for field in (
            "powerup_previous_type",
            "powerup_primary_type",
            "powerup_secondary_type",
            "powerup_previous_ticks",
            "powerup_lifetime_ticks",
            "powerup_transition_ticks",
            "powerup_visual_index",
        ):
            value = getattr(entity, field)
            if value is not None:
                setattr(ball, field, value)
        for field in (
            "powerup_visual_scale",
            "powerup_visual_step",
        ):
            value = getattr(entity, field)
            if value is not None:
                setattr(ball, field, np.float32(value))
    _restore_initial_free_projectiles(
        simulator,
        initial_free_projectiles,
    )
    simulator._next_id = max(
        entity.ball_id for entity in first.entities
    ) + 1
    simulator.num_balls_created = len(active) + len(pending)
    simulator.tick_count = (
        first.update
        if first.board_update_count is None
        else first.board_update_count
    )
    native_curve_runtime = first.curve_runtime_state(identified_curve)
    if first.curve_runtime_states and native_curve_runtime is None:
        raise PcMemoryTrajectoryError(
            "gameplay_diff_curve_runtime_state_missing"
        )
    native_curve_plan = first.curve_plan_state(identified_curve)
    if first.curve_plans and native_curve_plan is None:
        raise PcMemoryTrajectoryError(
            "gameplay_diff_curve_plan_state_missing"
        )
    if native_curve_plan is None:
        simulator.feed_plan_exhausted = False
        curve_plan_state_restored = False
    else:
        simulator.feed_plan_exhausted = (
            native_curve_plan.planned_count == 0
            and not native_curve_plan.add_plan_enabled
        )
        curve_plan_state_restored = True
    if native_curve_runtime is None:
        # Legacy trajectories did not expose the Curve latches.  Retain their
        # old diagnostic behavior, but provenance marks that it was inferred.
        observed_next_movement_speed = _infer_speed(first, frames[1])
        simulator.advance_speed = observed_next_movement_speed
        simulator.first_chain_end = int(active[-1].curve_distance)
        simulator.has_reached_rollout = True
        simulator.has_reached_cruising_speed = True
        curve_runtime_state_restored = False
    else:
        observed_next_movement_speed = _observed_chain_movement_delta(
            first,
            frames[1],
        )
        simulator.advance_speed = np.float32(
            native_curve_runtime.advance_speed
        )
        simulator.first_chain_end = native_curve_runtime.first_chain_end
        simulator.stop_adding = native_curve_runtime.stop_adding
        simulator.stop_time = native_curve_runtime.stop_time
        simulator.first_ball_moved_backwards = (
            native_curve_runtime.first_ball_moved_backwards
        )
        simulator.has_reached_rollout = (
            native_curve_runtime.has_reached_rollout
        )
        simulator.has_reached_cruising_speed = (
            native_curve_runtime.has_reached_cruising_speed
        )
        curve_runtime_state_restored = True
    simulator._have_sets = any(ball.exploding for ball in simulator.balls)
    simulator._store_active_curve()
    return simulator, {
        "level_id": identified_level,
        "hard": hard,
        "curve_index": identified_curve,
        "curve_fit_maximum_error_px": curve_fit_maximum,
        "curve_fit_rms_error_px": curve_fit_rms,
        "inferred_advance_speed": float(simulator.advance_speed),
        "observed_next_movement_speed": float(
            observed_next_movement_speed
        ),
        "curve_runtime_state_restored": curve_runtime_state_restored,
        "curve_plan_state_restored": curve_plan_state_restored,
        "curve_feed_plan_exhausted_restored": (
            simulator.feed_plan_exhausted
        ),
        "score_at_level_start": score_at_level_start,
        "shooter_rng_state_restored": first.qrand is not None,
        "global_mtrand_state_restored": global_mtrand_available,
        "native_game_time_restored": first.native_game_time is not None,
        "powerup_manager_state_restored": (
            native_powerup_state is not None
        ),
        "fruit_runtime_derived_from_initial_boundary": (
            fruit_runtime_derived_from_initial_boundary
        ),
        "fruit_runtime_state_restored": fruit_runtime_state_restored,
        "initial_free_projectile_count": len(initial_free_projectiles),
        "initial_free_projectile_state_restored": True,
        "initial_merging_projectile_count": 0,
        "fruit_spawn_calibration": (
            None
            if simulator.fruit_calibration is None
            else {
                "frequency_ticks": (
                    simulator.fruit_calibration.frequency_ticks
                ),
                "lifetime_ticks": simulator.fruit_calibration.lifetime_ticks,
                "points": [
                    list(point) for point in simulator.fruit_calibration.points
                ],
                "unlock_percentages": [
                    list(row)
                    for row in simulator.fruit_calibration.unlock_percentages
                ],
                "fruit_type": simulator.fruit_calibration.fruit_type,
                "logical_width": simulator.fruit_calibration.logical_width,
                "logical_height": simulator.fruit_calibration.logical_height,
                "sheet_columns": simulator.fruit_calibration.sheet_columns,
                "sheet_rows": simulator.fruit_calibration.sheet_rows,
                "collection_animation_frames": (
                    simulator.fruit_calibration.collection_animation_frames
                ),
                "collection_animation_fps": (
                    simulator.fruit_calibration.collection_animation_fps
                ),
                "provenance": simulator.fruit_calibration.provenance,
            }
        ),
        "powerup_spawn_calibration": (
            None
            if powerup_calibration is None
            else {
                "chance_denominator": (
                    powerup_calibration.chance_denominator
                ),
                "initial_delay_ticks": (
                    powerup_calibration.initial_delay_ticks
                ),
                "spawn_delay_ticks": (
                    powerup_calibration.spawn_delay_ticks
                ),
                "cooldown_ticks": powerup_calibration.cooldown_ticks,
                "unique_color": powerup_calibration.unique_color,
                "supported_types": list(
                    powerup_calibration.supported_types
                ),
                "provenance": powerup_calibration.provenance,
            }
        ),
    }


def _map_ordered_identities(
    *,
    simulator_ids: Sequence[int],
    simulator_colors: Sequence[int],
    pc_entities: Sequence[TrajectoryEntity],
    identity_map: dict[int, int],
    update: int,
    zone: str,
    mappings: list[dict[str, Any]],
) -> None:
    if len(simulator_ids) != len(pc_entities):
        return
    mapped_pc_ids = set(identity_map.values())
    for simulator_id, simulator_color, entity in zip(
        simulator_ids,
        simulator_colors,
        pc_entities,
        strict=True,
    ):
        if simulator_id in identity_map or entity.ball_id in mapped_pc_ids:
            continue
        if simulator_color != entity.color_id:
            continue
        identity_map[simulator_id] = entity.ball_id
        mapped_pc_ids.add(entity.ball_id)
        mappings.append(
            {
                "framework_update": update,
                "zone": zone,
                "simulator_id": simulator_id,
                "pc_id": entity.ball_id,
                "source": "midstate_allocator_lineage",
            }
        )


def _projectile_errors(
    simulator_rows: Sequence[Projectile],
    pc_rows: Sequence[TrajectoryEntity],
    *,
    identity_map: dict[int, int],
) -> tuple[bool, float, float, float]:
    if len(simulator_rows) != len(pc_rows):
        return False, math.inf, math.inf, math.inf
    position_error = 0.0
    waypoint_error = 0.0
    progress_error = 0.0
    matched = True
    for simulator_row, pc_row in zip(
        simulator_rows,
        pc_rows,
        strict=True,
    ):
        matched = matched and (
            identity_map.get(simulator_row.id) == pc_row.ball_id
            and simulator_row.color == pc_row.color_id
        )
        position_error = max(
            position_error,
            math.hypot(
                float(simulator_row.position[0]) - pc_row.position_x,
                float(simulator_row.position[1]) - pc_row.position_y,
            ),
        )
        waypoint_error = max(
            waypoint_error,
            abs(float(simulator_row.waypoint) - pc_row.curve_distance),
        )
        if pc_row.merge_progress is not None:
            progress_error = max(
                progress_error,
                abs(
                    float(simulator_row.hit_percent)
                    - pc_row.merge_progress
                ),
            )
    return matched, position_error, waypoint_error, progress_error


def _projectile_latent_mismatches(
    simulator_rows: Sequence[Projectile],
    pc_rows: Sequence[TrajectoryEntity],
    *,
    float_tolerance: float,
    curve_count: int,
) -> dict[str, int]:
    mismatches: dict[str, int] = {}
    if len(simulator_rows) != len(pc_rows):
        return {"count": 1}
    for simulator_row, pc_row in zip(
        simulator_rows,
        pc_rows,
        strict=True,
    ):
        checks = {
            "radius": (
                pc_row.radius is not None
                and float(simulator_row.radius) != pc_row.radius
            ),
            "velocity": (
                pc_row.velocity_x is not None
                and pc_row.velocity_y is not None
                and math.hypot(
                    float(simulator_row.velocity[0]) - pc_row.velocity_x,
                    float(simulator_row.velocity[1]) - pc_row.velocity_y,
                )
                > float_tolerance
            ),
            "merge_speed": (
                pc_row.merge_speed is not None
                and abs(
                    float(simulator_row.merge_speed)
                    - pc_row.merge_speed
                )
                > float_tolerance
            ),
            "just_fired": (
                pc_row.fired is not None
                and simulator_row.just_fired != pc_row.fired
            ),
            "gap_info": (
                pc_row.gap_entry_count is not None
                and tuple(simulator_row.gap_info)
                != tuple(
                    (boundary_ball_id, gap_distance)
                    for _, gap_distance, boundary_ball_id
                    in pc_row.gap_entries
                )
            ),
        }
        if pc_row.curve_points is not None:
            if len(pc_row.curve_points) < curve_count:
                checks["curve_points"] = True
            else:
                expected_curve_points = tuple(
                    0 if value == RETAIL_DEBUG_FILL_I32 else value
                    for value in pc_row.curve_points[:curve_count]
                )
                simulator_curve_points = tuple(
                    simulator_row.curve_points.get(
                        index,
                        (
                            simulator_row.curve_point
                            if curve_count == 1 and index == 0
                            else 0
                        ),
                    )
                    for index in range(curve_count)
                )
                checks["curve_points"] = (
                    simulator_curve_points != expected_curve_points
                )
        for field, differs in checks.items():
            if differs:
                mismatches[field] = mismatches.get(field, 0) + 1
    return mismatches


def _curve_state_mismatches(
    simulator: RevengeSimulator,
    frame: TrajectoryFrame,
    *,
    curve_index: int,
) -> dict[str, int]:
    mismatches: dict[str, int] = {}

    def check(name: str, observed: Any, expected: Any) -> None:
        if observed != expected:
            mismatches[name] = 1

    if frame.native_game_time is not None:
        check(
            "native_game_time",
            simulator.native_game_time,
            frame.native_game_time,
        )
    runtime = frame.curve_runtime_state(curve_index)
    if runtime is not None:
        check(
            "advance_speed",
            float(np.float32(simulator.advance_speed)),
            runtime.advance_speed,
        )
        check(
            "first_chain_end",
            simulator.first_chain_end,
            runtime.first_chain_end,
        )
        check("stop_adding", simulator.stop_adding, runtime.stop_adding)
        check("stop_time", simulator.stop_time, runtime.stop_time)
        check(
            "first_ball_moved_backwards",
            simulator.first_ball_moved_backwards,
            runtime.first_ball_moved_backwards,
        )
        check(
            "has_reached_cruising_speed",
            simulator.has_reached_cruising_speed,
            runtime.has_reached_cruising_speed,
        )
        check(
            "has_reached_rollout",
            simulator.has_reached_rollout,
            runtime.has_reached_rollout,
        )
    plan = frame.curve_plan_state(curve_index)
    if plan is not None:
        check(
            "feed_plan_exhausted",
            simulator.feed_plan_exhausted,
            plan.planned_count == 0 and not plan.add_plan_enabled,
        )
    powerup = frame.curve_powerup_state(curve_index)
    if powerup is not None:
        pairs = (
            (
                "powerup_last_any_spawn_time",
                simulator.powerup_last_any_spawn_time,
                powerup.last_any_spawn_time,
            ),
            (
                "powerup_last_spawn_times",
                tuple(simulator.powerup_last_spawn_times),
                powerup.last_spawn_times,
            ),
            (
                "powerup_cooldown_times",
                tuple(simulator.powerup_cooldown_times),
                powerup.cooldown_times,
            ),
            (
                "powerup_spawn_counts",
                tuple(simulator.powerup_spawn_counts),
                powerup.spawn_counts,
            ),
            (
                "powerup_field_124_by_type",
                tuple(simulator.powerup_field_124_by_type),
                powerup.field_124_by_type,
            ),
            (
                "active_powerup_color_counts",
                tuple(simulator.active_powerup_color_counts),
                powerup.active_color_counts,
            ),
            (
                "reverse_speed",
                float(simulator.config.reverse_speed),
                powerup.reverse_speed,
            ),
            ("slow_ticks", simulator.slow_count, powerup.slow_ticks),
            (
                "reverse_ticks",
                simulator.backward_count,
                powerup.reverse_ticks,
            ),
            (
                "last_powerup_waypoint",
                simulator.last_powerup_waypoint,
                powerup.last_powerup_waypoint,
            ),
            (
                "powerup_triggered",
                simulator.powerup_triggered,
                powerup.powerup_triggered,
            ),
        )
        for name, observed, expected in pairs:
            check(name, observed, expected)
    return mismatches


def compare_gameplay_trajectory(
    frames: Sequence[TrajectoryFrame],
    *,
    demo: PopCapDemo,
    original_root: Path,
    hard: bool = False,
    level_id: str | None = None,
    curve_index: int | None = None,
    synchronize_shooter: bool = True,
    reconcile_external_visual_mtrand: bool = False,
    maximum_external_visual_mtrand_draws_per_tick: int = 64,
    waypoint_tolerance: float = 1e-4,
    position_tolerance: float = 1e-3,
    progress_tolerance: float = 1e-6,
) -> Mapping[str, Any]:
    """Run a DMO-driven midstate differential replay."""

    if not frames:
        raise PcMemoryTrajectoryError("gameplay_diff_frames_empty")
    if (
        isinstance(maximum_external_visual_mtrand_draws_per_tick, bool)
        or not isinstance(
            maximum_external_visual_mtrand_draws_per_tick,
            int,
        )
        or maximum_external_visual_mtrand_draws_per_tick <= 0
    ):
        raise PcMemoryTrajectoryError(
            "gameplay_diff_external_mtrand_draw_limit_invalid"
        )
    simulator, provenance = _transplant_midstate(
        frames,
        original_root=original_root.resolve(),
        hard=hard,
        level_id=level_id,
        curve_index=curve_index,
    )
    commands: dict[int, list[Any]] = {}
    for command in demo.commands:
        if frames[0].update < command.update <= frames[-1].update:
            commands.setdefault(command.update, []).append(command)

    identity_map = {
        entity.ball_id: entity.ball_id
        for entity in (
            *_active(frames[0]),
            *_pc_projectiles(frames[0], staging=False),
        )
    }
    identity_mappings: list[dict[str, Any]] = []
    input_results: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    failure_reasons: list[str] = []
    shooter_mismatch_updates: list[int] = []
    mtrand_reconciliations: list[dict[str, Any]] = []

    for previous, frame in zip(frames, frames[1:]):
        pc_active = _active(frame)
        pc_pending = _entities_in_list(frame, 0x68)
        previous_pc_pending = _entities_in_list(previous, 0x68)
        leading_mtrand_reconciliation: dict[str, Any] | None = None
        pending_generation_baseline = (
            copy.deepcopy(simulator)
            if (
                reconcile_external_visual_mtrand
                and not previous_pc_pending
                and pc_pending
            )
            else None
        )
        events = simulator.tick()
        if (
            pending_generation_baseline is not None
            and tuple(simulator.pending_colors)
            != tuple(entity.color_id for entity in pc_pending)
            and frame.global_mtrand is not None
        ):
            candidate_matches: list[
                tuple[
                    RevengeSimulator,
                    Any,
                    dict[str, Any],
                ]
            ] = []
            for leading_draws in range(
                1,
                maximum_external_visual_mtrand_draws_per_tick + 1,
            ):
                candidate = copy.deepcopy(pending_generation_baseline)
                before_index = candidate.rng.index
                before_state_sha256 = _rng_state_sha256(candidate.rng)
                for _ in range(leading_draws):
                    candidate.rng.next_u31()
                after_leading_index = candidate.rng.index
                after_leading_state_sha256 = _rng_state_sha256(
                    candidate.rng
                )
                candidate_events = candidate.tick()
                candidate_state = candidate.rng.state
                post_gameplay_state_sha256 = _rng_state_sha256(
                    candidate.rng
                )
                trailing_draws = _reconcile_external_mtrand_draws(
                    candidate.rng,
                    observed=frame.global_mtrand,
                    maximum_draws=(
                        maximum_external_visual_mtrand_draws_per_tick
                    ),
                )
                candidate.rng.load_state(*candidate_state)
                if (
                    trailing_draws is None
                    or tuple(candidate.pending_colors)
                    != tuple(
                        entity.color_id for entity in pc_pending
                    )
                    or tuple(ball.color for ball in candidate.balls)
                    != tuple(entity.color_id for entity in pc_active)
                    or tuple(
                        projectile.color
                        for projectile in candidate.free_projectiles
                    )
                    != tuple(
                        entity.color_id
                        for entity in _pc_projectiles(
                            frame,
                            staging=False,
                        )
                    )
                    or tuple(
                        projectile.color
                        for projectile in candidate.merging_projectiles
                    )
                    != tuple(
                        entity.color_id
                        for entity in _pc_projectiles(
                            frame,
                            staging=True,
                        )
                    )
                    or candidate.score != frame.score
                    or candidate_events.score_delta
                    != frame.score - previous.score
                    or _fruit_state_mismatches(
                        candidate,
                        frame.fruit_state,
                    )
                ):
                    continue
                candidate_matches.append((
                    candidate,
                    candidate_events,
                    {
                    "framework_update": frame.update,
                    "phase": "before_gameplay_tick",
                    "draw_count": leading_draws,
                    "simulator_index_before": before_index,
                    "simulator_index_after": after_leading_index,
                    "simulator_state_sha256_before": (
                        before_state_sha256
                    ),
                    "simulator_state_sha256_after": (
                        after_leading_state_sha256
                    ),
                    "post_gameplay_state_sha256": (
                        post_gameplay_state_sha256
                    ),
                    "pc_final_index": frame.global_mtrand.index,
                    "pc_final_state_sha256": (
                        _trajectory_mtrand_state_sha256(
                            frame.global_mtrand
                        )
                    ),
                    "pc_final_state_forward_reachable": True,
                    "post_gameplay_draw_count": trailing_draws,
                    "source_transition_selected_uniquely_by_full_gameplay_state": (
                        True
                    ),
                    },
                ))
            if len(candidate_matches) == 1:
                (
                    simulator,
                    events,
                    leading_mtrand_reconciliation,
                ) = candidate_matches[0]
        _map_ordered_identities(
            simulator_ids=[ball.id for ball in simulator.balls],
            simulator_colors=[ball.color for ball in simulator.balls],
            pc_entities=pc_active,
            identity_map=identity_map,
            update=frame.update,
            zone="active_chain",
            mappings=identity_mappings,
        )
        pc_fired = _pc_projectiles(frame, staging=False)
        _map_ordered_identities(
            simulator_ids=[
                projectile.id for projectile in simulator.free_projectiles
            ],
            simulator_colors=[
                projectile.color
                for projectile in simulator.free_projectiles
            ],
            pc_entities=pc_fired,
            identity_map=identity_map,
            update=frame.update,
            zone="fired",
            mappings=identity_mappings,
        )
        pc_staging = _pc_projectiles(frame, staging=True)
        _map_ordered_identities(
            simulator_ids=[
                projectile.id
                for projectile in simulator.merging_projectiles
            ],
            simulator_colors=[
                projectile.color
                for projectile in simulator.merging_projectiles
            ],
            pc_entities=pc_staging,
            identity_map=identity_map,
            update=frame.update,
            zone="staging",
            mappings=identity_mappings,
        )

        active_count_match = len(simulator.balls) == len(pc_active)
        active_identity_match = (
            active_count_match
            and [
                identity_map.get(ball.id) for ball in simulator.balls
            ]
            == [entity.ball_id for entity in pc_active]
        )
        active_color_match = (
            active_count_match
            and [ball.color for ball in simulator.balls]
            == [entity.color_id for entity in pc_active]
        )
        pending_colors_pc = tuple(
            entity.color_id for entity in pc_pending
        )
        pending_colors_simulator = tuple(simulator.pending_colors)
        pending_colors_match = (
            pending_colors_simulator == pending_colors_pc
        )
        waypoint_errors = (
            [
                abs(float(ball.waypoint) - entity.curve_distance)
                for ball, entity in zip(
                    simulator.balls,
                    pc_active,
                    strict=True,
                )
            ]
            if active_count_match
            else []
        )
        position_errors = (
            [
                math.hypot(
                    float(simulator.ball_position(ball)[0])
                    - entity.position_x,
                    float(simulator.ball_position(ball)[1])
                    - entity.position_y,
                )
                for ball, entity in zip(
                    simulator.balls,
                    pc_active,
                    strict=True,
                )
            ]
            if active_count_match
            else []
        )
        maximum_waypoint_error = max(
            waypoint_errors,
            default=math.inf if not active_count_match else 0.0,
        )
        maximum_position_error = max(
            position_errors,
            default=math.inf if not active_count_match else 0.0,
        )

        latent_mismatches: dict[str, int] = {}
        if active_count_match:
            latent_fields = (
                "contact_next",
                "exploding",
                "explode_frame",
                "should_remove",
                "update_count",
                "suck_count",
                "backwards_count",
                "backwards_speed",
                "combo_count",
                "combo_score",
                "powerup_previous_type",
                "powerup_primary_type",
                "powerup_secondary_type",
                "powerup_previous_ticks",
                "powerup_lifetime_ticks",
                "powerup_transition_ticks",
                "powerup_visual_scale",
                "powerup_visual_step",
                "powerup_visual_index",
            )
            for ball, entity in zip(
                simulator.balls,
                pc_active,
                strict=True,
            ):
                for field in latent_fields:
                    expected = getattr(entity, field)
                    if (
                        expected is not None
                        and getattr(ball, field) != expected
                    ):
                        latent_mismatches[field] = (
                            latent_mismatches.get(field, 0) + 1
                        )

        (
            fired_identity_match,
            fired_position_error,
            fired_waypoint_error,
            fired_progress_error,
        ) = _projectile_errors(
            simulator.free_projectiles,
            pc_fired,
            identity_map=identity_map,
        )
        (
            staging_identity_match,
            staging_position_error,
            staging_waypoint_error,
            staging_progress_error,
        ) = _projectile_errors(
            simulator.merging_projectiles,
            pc_staging,
            identity_map=identity_map,
        )
        fired_latent_mismatches = _projectile_latent_mismatches(
            simulator.free_projectiles,
            pc_fired,
            float_tolerance=position_tolerance,
            curve_count=simulator.curve_count,
        )
        staging_latent_mismatches = _projectile_latent_mismatches(
            simulator.merging_projectiles,
            pc_staging,
            float_tolerance=position_tolerance,
            curve_count=simulator.curve_count,
        )
        curve_state_mismatches = _curve_state_mismatches(
            simulator,
            frame,
            curve_index=int(provenance["curve_index"]),
        )
        fruit_state_mismatches = _fruit_state_mismatches(
            simulator,
            frame.fruit_state,
        )
        shooter_match = (
            simulator.current_color,
            simulator.next_color,
        ) == (
            frame.current_color_id,
            frame.next_color_id,
        )
        if not shooter_match:
            shooter_mismatch_updates.append(frame.update)
        if (frame.qrand is None) != (
            frame.thread_crt_rand_state is None
        ):
            raise PcMemoryTrajectoryError(
                "gameplay_diff_shooter_rng_incomplete"
            )
        exact_shooter_state = frame.qrand is not None
        exact_global_mtrand_state = frame.global_mtrand is not None
        qrand_match: bool | None = None
        crt_rand_match: bool | None = None
        global_mtrand_match: bool | None = None
        if frame.qrand is not None:
            chooser = simulator._color_chooser
            weights, sways, last_hit, previous_hit = (
                _normalized_qrand_vectors(
                    frame.qrand,
                    size=chooser.size,
                )
            )
            qrand_match = (
                chooser.update_count == frame.qrand.update_count
                and chooser.selected_index
                == frame.qrand.selected_index
                and tuple(map(float, chooser.weights))
                == weights
                and tuple(map(float, chooser.sways))
                == sways
                and tuple(map(int, chooser.last_hit))
                == last_hit
                and tuple(map(int, chooser.previous_hit))
                == previous_hit
            )
            crt_rand_match = (
                simulator.crt_rng.state
                == frame.thread_crt_rand_state
            )
        if frame.global_mtrand is not None:
            global_mtrand_match = (
                simulator.rng.index == frame.global_mtrand.index
                and simulator.rng.words == frame.global_mtrand.words
            )

        global_mtrand_match_before_reconciliation = global_mtrand_match
        global_mtrand_index_before_reconciliation = (
            simulator.rng.index if exact_global_mtrand_state else None
        )
        global_mtrand_state_before_reconciliation = (
            _rng_state_sha256(simulator.rng)
            if exact_global_mtrand_state
            else None
        )
        reconciled_mtrand_draws = 0
        gameplay_state_matches_before_global_rng = (
            active_identity_match
            and active_color_match
            and maximum_waypoint_error <= waypoint_tolerance
            and maximum_position_error <= position_tolerance
            and not latent_mismatches
            and not fired_latent_mismatches
            and not staging_latent_mismatches
            and not curve_state_mismatches
            and not fruit_state_mismatches
            and pending_colors_match
            and fired_identity_match
            and fired_position_error <= position_tolerance
            and fired_waypoint_error <= waypoint_tolerance
            and fired_progress_error <= progress_tolerance
            and staging_identity_match
            and staging_position_error <= position_tolerance
            and staging_waypoint_error <= waypoint_tolerance
            and staging_progress_error <= progress_tolerance
            and simulator.score == frame.score
            and events.score_delta == frame.score - previous.score
            and (
                not exact_shooter_state
                or (
                    shooter_match
                    and qrand_match is True
                    and crt_rand_match is True
                )
            )
        )
        if (
            reconcile_external_visual_mtrand
            and exact_global_mtrand_state
            and global_mtrand_match is False
            and gameplay_state_matches_before_global_rng
            and frame.global_mtrand is not None
        ):
            reconciled = _reconcile_external_mtrand_draws(
                simulator.rng,
                observed=frame.global_mtrand,
                maximum_draws=(
                    maximum_external_visual_mtrand_draws_per_tick
                ),
            )
            if reconciled is not None and reconciled > 0:
                reconciled_mtrand_draws = reconciled
                global_mtrand_match = True
                mtrand_reconciliations.append(
                    {
                        "framework_update": frame.update,
                        "phase": "after_gameplay_state_match",
                        "draw_count": reconciled,
                        "simulator_index_before": (
                            global_mtrand_index_before_reconciliation
                        ),
                        "simulator_index_after": simulator.rng.index,
                        "simulator_state_sha256_before": (
                            global_mtrand_state_before_reconciliation
                        ),
                        "simulator_state_sha256_after": (
                            _rng_state_sha256(simulator.rng)
                        ),
                        "pc_index": frame.global_mtrand.index,
                        "pc_state_sha256": (
                            _trajectory_mtrand_state_sha256(
                                frame.global_mtrand
                            )
                        ),
                        "full_state_reached_exactly": True,
                        "gameplay_state_matched_before_reconciliation": (
                            True
                        ),
                    }
                )
        if (
            leading_mtrand_reconciliation is not None
            and gameplay_state_matches_before_global_rng
        ):
            mtrand_reconciliations.append(
                leading_mtrand_reconciliation
            )

        score_delta_pc = frame.score - previous.score
        row_pass = (
            active_identity_match
            and active_color_match
            and maximum_waypoint_error <= waypoint_tolerance
            and maximum_position_error <= position_tolerance
            and not latent_mismatches
            and not fired_latent_mismatches
            and not staging_latent_mismatches
            and not curve_state_mismatches
            and not fruit_state_mismatches
            and pending_colors_match
            and fired_identity_match
            and fired_position_error <= position_tolerance
            and fired_waypoint_error <= waypoint_tolerance
            and fired_progress_error <= progress_tolerance
            and staging_identity_match
            and staging_position_error <= position_tolerance
            and staging_waypoint_error <= waypoint_tolerance
            and staging_progress_error <= progress_tolerance
            and simulator.score == frame.score
            and events.score_delta == score_delta_pc
            and (
                not exact_shooter_state
                or (
                    shooter_match
                    and qrand_match is True
                    and crt_rand_match is True
                )
            )
            and (
                not exact_global_mtrand_state
                or global_mtrand_match is True
            )
        )
        if not row_pass:
            failure_reasons.append(f"tick_mismatch:{frame.update}")
        rows.append(
            {
                "framework_update": frame.update,
                "status": "PASS" if row_pass else "FAIL",
                "active_count_pc": len(pc_active),
                "active_count_simulator": len(simulator.balls),
                "active_identity_match": active_identity_match,
                "active_color_match": active_color_match,
                "pending_colors_pc": list(pending_colors_pc),
                "pending_colors_simulator": list(
                    pending_colors_simulator
                ),
                "pending_colors_match": pending_colors_match,
                "maximum_waypoint_error": maximum_waypoint_error,
                "maximum_position_error_px": maximum_position_error,
                "latent_mismatches": latent_mismatches,
                "fired_latent_mismatches": fired_latent_mismatches,
                "staging_latent_mismatches": (
                    staging_latent_mismatches
                ),
                "curve_state_mismatches": curve_state_mismatches,
                "fruit_state_mismatches": fruit_state_mismatches,
                "fired_count_pc": len(pc_fired),
                "fired_count_simulator": len(
                    simulator.free_projectiles
                ),
                "fired_identity_match": fired_identity_match,
                "fired_position_error_px": fired_position_error,
                "fired_waypoint_error": fired_waypoint_error,
                "fired_progress_error": fired_progress_error,
                "staging_count_pc": len(pc_staging),
                "staging_count_simulator": len(
                    simulator.merging_projectiles
                ),
                "staging_identity_match": staging_identity_match,
                "staging_position_error_px": staging_position_error,
                "staging_progress_error": staging_progress_error,
                "score_pc": frame.score,
                "score_simulator": simulator.score,
                "score_delta_pc": score_delta_pc,
                "score_delta_simulator": events.score_delta,
                "shooter_observed_match": shooter_match,
                "current_ball_id_pc": frame.current_ball_id,
                "next_ball_id_pc": frame.next_ball_id,
                "current_color_id_pc": frame.current_color_id,
                "current_color_id_simulator": simulator.current_color,
                "next_color_id_pc": frame.next_color_id,
                "next_color_id_simulator": simulator.next_color,
                "qrand_observed_match": qrand_match,
                "thread_crt_rand_observed_match": crt_rand_match,
                "global_mtrand_observed_match": global_mtrand_match,
                "global_mtrand_observed_match_before_reconciliation": (
                    global_mtrand_match_before_reconciliation
                ),
                "global_mtrand_reconciled_draws": (
                    reconciled_mtrand_draws
                ),
                "global_mtrand_leading_reconciled_draws": (
                    0
                    if leading_mtrand_reconciliation is None
                    else leading_mtrand_reconciliation["draw_count"]
                ),
                "gameplay_state_matched_before_global_rng": (
                    gameplay_state_matches_before_global_rng
                ),
                "global_mtrand_index_pc": (
                    frame.global_mtrand.index
                    if frame.global_mtrand is not None
                    else None
                ),
                "global_mtrand_index_simulator": (
                    simulator.rng.index
                    if exact_global_mtrand_state
                    else None
                ),
                "global_mtrand_index_simulator_before_reconciliation": (
                    global_mtrand_index_before_reconciliation
                ),
                "qrand_update_count_pc": (
                    frame.qrand.update_count
                    if frame.qrand is not None
                    else None
                ),
                "qrand_update_count_simulator": (
                    simulator._color_chooser.update_count
                    if exact_shooter_state
                    else None
                ),
                "qrand_selected_index_pc": (
                    frame.qrand.selected_index
                    if frame.qrand is not None
                    else None
                ),
                "qrand_selected_index_simulator": (
                    simulator._color_chooser.selected_index
                    if exact_shooter_state
                    else None
                ),
                "thread_crt_rand_state_pc": (
                    frame.thread_crt_rand_state
                ),
                "thread_crt_rand_state_simulator": (
                    simulator.crt_rng.state
                    if exact_shooter_state
                    else None
                ),
                "events": {
                    "fired": events.fired,
                    "hits": events.hits,
                    "inserted": events.inserted,
                    "matches": events.matches,
                    "balls_exploded": events.balls_exploded,
                    "balls_removed": events.balls_removed,
                    "score_delta": events.score_delta,
                    "powerups_spawned": events.powerups_spawned,
                    "powerups_triggered": events.powerups_triggered,
                    "fruit_chance_draws": events.fruit_chance_draws,
                    "fruits_spawned": events.fruits_spawned,
                    "fruits_expired": events.fruits_expired,
                    "fruits_collected": events.fruits_collected,
                },
            }
        )

        if synchronize_shooter:
            simulator.current_color = frame.current_color_id
            simulator.next_color = frame.next_color_id
        for command in commands.get(frame.update, ()):
            payload = command.payload
            accepted: bool | None = None
            if command.kind == "mouse_move":
                simulator.set_aim(
                    math.atan2(
                        int(payload["y"]) - float(simulator.shooter[1]),
                        int(payload["x"]) - float(simulator.shooter[0]),
                    )
                )
            elif command.kind == "mouse_button" and payload.get("down"):
                if payload.get("button") == 1:
                    accepted = simulator.request_fire(
                        float(simulator.aim_angle)
                    )
                else:
                    accepted = simulator.swap_balls()
            input_results.append(
                {
                    "framework_update": frame.update,
                    "sequence": command.sequence,
                    "kind": command.kind,
                    "accepted": accepted,
                }
            )

    mtrand_policy, mtrand_reconciliation = (
        _mtrand_reconciliation_contract(
            mtrand_reconciliations,
            maximum_draws_per_tick=(
                maximum_external_visual_mtrand_draws_per_tick
            ),
        )
    )
    return {
        "schema": GAMEPLAY_DIFF_SCHEMA,
        "version": GAMEPLAY_DIFF_VERSION,
        "status": "PASS" if not failure_reasons else "FAIL",
        "start_update": frames[0].update,
        "end_update": frames[-1].update,
        "compared_tick_count": len(rows),
        **provenance,
        "synchronize_shooter": synchronize_shooter,
        "global_mtrand_policy": mtrand_policy,
        "global_mtrand_reconciliation": mtrand_reconciliation,
        "shooter_mismatch_updates": shooter_mismatch_updates,
        "identity_mappings": identity_mappings,
        "input_results": input_results,
        "waypoint_tolerance": waypoint_tolerance,
        "position_tolerance_px": position_tolerance,
        "progress_tolerance": progress_tolerance,
        "failure_reasons": failure_reasons,
        "ticks": rows,
    }


__all__ = [
    "GAMEPLAY_DIFF_SCHEMA",
    "compare_gameplay_trajectory",
]
