"""Verify a mutation-free retail Zuma victory from consecutive PC frames.

The frame contract deliberately separates gameplay semantics from source
provenance.  :func:`verify_natural_win_frames` proves the native state
transition, while the PC mechanism audit binds those frames to a verified
retail executable, DMO, video, and an explicit no-mutation attestation.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .pc_memory_trajectory import (
    ACTIVE_BOARD_VERSION,
    ACTIVE_CHAIN_LIST_OFFSET,
    TrajectoryEntity,
    TrajectoryFrame,
)
from .revenge_core import Projectile, RevengeSimulator


NATURAL_WIN_SCHEMA = "zuma-rl.pc-natural-win-sequence"
NATURAL_WIN_VERSION = 1
NATURAL_WIN_SIMULATOR_DIFF_SCHEMA = (
    "zuma-rl.pc-natural-win-simulator-diff"
)
NATURAL_WIN_SIMULATOR_DIFF_VERSION = 2
PENDING_CHAIN_LIST_OFFSET = 0x68
INSERTING_CHAIN_LIST_OFFSET = 0x50


class PcNaturalWinError(ValueError):
    """A native trajectory does not satisfy the natural-victory contract."""


def _fail(code: str) -> None:
    raise PcNaturalWinError(code)


def _declared_curve_count(
    frame: TrajectoryFrame,
    *,
    curve_index: int,
    container_offset: int,
) -> int:
    matches = [
        count
        for candidate_curve, candidate_offset, count in frame.list_counts
        if (
            candidate_curve == curve_index
            and candidate_offset == container_offset
        )
    ]
    if len(matches) != 1:
        _fail(
            "natural_win_curve_list_count_missing_or_duplicate:"
            f"{frame.update}:{curve_index}:{container_offset:03x}"
        )
    return matches[0]


def _curve_entities(
    frame: TrajectoryFrame,
    *,
    curve_index: int,
    container_offset: int,
) -> tuple[TrajectoryEntity, ...]:
    zone = f"curve:{curve_index}:list:{container_offset:03x}"
    expected_kind = (
        "bullet"
        if container_offset == INSERTING_CHAIN_LIST_OFFSET
        else "ball"
    )
    entities = tuple(
        sorted(
            (
                entity
                for entity in frame.entities
                if entity.zone == zone
                and entity.object_kind == expected_kind
            ),
            key=lambda entity: entity.index,
        )
    )
    if tuple(entity.index for entity in entities) != tuple(
        range(len(entities))
    ):
        _fail(
            "natural_win_curve_entity_indexes_invalid:"
            f"{frame.update}:{curve_index}:{container_offset:03x}"
        )
    if len({entity.ball_id for entity in entities}) != len(entities):
        _fail(
            "natural_win_curve_entity_ids_duplicate:"
            f"{frame.update}:{curve_index}:{container_offset:03x}"
        )
    declared = _declared_curve_count(
        frame,
        curve_index=curve_index,
        container_offset=container_offset,
    )
    if len(entities) != declared:
        _fail(
            "natural_win_curve_entity_count_mismatch:"
            f"{frame.update}:{curve_index}:{container_offset:03x}"
        )
    return entities


def _fired(frame: TrajectoryFrame) -> tuple[TrajectoryEntity, ...]:
    entities = tuple(
        entity
        for entity in frame.entities
        if entity.zone == "fired" and entity.object_kind == "bullet"
    )
    if len({entity.ball_id for entity in entities}) != len(entities):
        _fail(f"natural_win_fired_ids_duplicate:{frame.update}")
    return entities


def _gameplay_count(
    frame: TrajectoryFrame,
    *,
    curve_indices: Sequence[int],
) -> int:
    total = len(_fired(frame))
    for curve_index in curve_indices:
        for offset in (
            INSERTING_CHAIN_LIST_OFFSET,
            ACTIVE_CHAIN_LIST_OFFSET,
            PENDING_CHAIN_LIST_OFFSET,
        ):
            total += len(
                _curve_entities(
                    frame,
                    curve_index=curve_index,
                    container_offset=offset,
                )
            )
    return total


def _plan_counts(
    frame: TrajectoryFrame,
    *,
    curve_indices: Sequence[int],
) -> tuple[int, ...]:
    states = {state.curve_index: state for state in frame.curve_plans}
    if (
        len(states) != len(frame.curve_plans)
        or tuple(sorted(states)) != tuple(curve_indices)
    ):
        _fail(f"natural_win_curve_plan_set_invalid:{frame.update}")
    return tuple(states[index].planned_count for index in curve_indices)


def _plan_enabled(
    frame: TrajectoryFrame,
    *,
    curve_indices: Sequence[int],
) -> tuple[bool, ...]:
    states = {state.curve_index: state for state in frame.curve_plans}
    if (
        len(states) != len(frame.curve_plans)
        or tuple(sorted(states)) != tuple(curve_indices)
    ):
        _fail(f"natural_win_curve_plan_set_invalid:{frame.update}")
    return tuple(states[index].add_plan_enabled for index in curve_indices)


def _validate_frame_contract(
    frames: Sequence[TrajectoryFrame],
    *,
    curve_indices: Sequence[int],
) -> None:
    if len(frames) < 5:
        _fail("natural_win_sequence_too_short")
    for before, after in zip(frames, frames[1:]):
        if after.update != before.update + 1:
            _fail(
                "natural_win_updates_not_consecutive:"
                f"{before.update}->{after.update}"
            )
        if (
            before.native_game_time is None
            or after.native_game_time is None
            or after.native_game_time != before.native_game_time + 1
        ):
            _fail(
                "natural_win_native_time_not_consecutive:"
                f"{before.update}->{after.update}"
            )
    for frame in frames:
        if frame.active_board_version != ACTIVE_BOARD_VERSION:
            _fail(f"natural_win_active_board_v2_required:{frame.update}")
        if frame.board_mode_flag_1064 is not False:
            _fail(f"natural_win_normal_mode_required:{frame.update}")
        if (
            frame.curve_plan_exhausted is None
            or frame.board_runtime_flag_157 is None
            or frame.board_runtime_i32_f54 is None
        ):
            _fail(f"natural_win_runtime_fields_missing:{frame.update}")
        _plan_counts(frame, curve_indices=curve_indices)
        _gameplay_count(frame, curve_indices=curve_indices)


def verify_natural_win_frames(
    frames: Sequence[TrajectoryFrame],
    *,
    curve_count: int = 1,
    minimum_terminal_tail_ticks: int = 1,
    expected_terminal_award_points: int = 100,
    expected_score_cross_update: int | None = None,
    expected_plan_exhaustion_update: int | None = None,
    expected_empty_update: int | None = None,
    expected_formal_transition_update: int | None = None,
) -> dict[str, Any]:
    """Prove the complete normal-mode target, drain, and victory sequence."""

    if (
        curve_count < 1
        or minimum_terminal_tail_ticks < 0
        or expected_terminal_award_points <= 0
    ):
        raise ValueError("natural-win expectations are invalid")
    curve_indices = tuple(range(curve_count))
    _validate_frame_contract(frames, curve_indices=curve_indices)

    target = frames[0].score_target
    if target <= 0 or any(frame.score_target != target for frame in frames):
        _fail("natural_win_score_target_invalid_or_changed")
    if any(frame.score < 0 or frame.displayed_score < 0 for frame in frames):
        _fail("natural_win_negative_score")
    if any(
        after.score < before.score
        for before, after in zip(frames, frames[1:])
    ):
        _fail("natural_win_score_decreased")
    if frames[0].score >= target:
        _fail("natural_win_below_target_context_missing")

    score_crossings = [
        index
        for index, (before, after) in enumerate(
            zip(frames, frames[1:]),
            start=1,
        )
        if before.score < target <= after.score
    ]
    if len(score_crossings) != 1:
        _fail("natural_win_score_crossing_not_unique")
    score_cross_index = score_crossings[0]
    score_cross = frames[score_cross_index]
    if (
        expected_score_cross_update is not None
        and score_cross.update != expected_score_cross_update
    ):
        _fail("natural_win_score_cross_update_mismatch")

    exhaustion_transitions = [
        index
        for index, (before, after) in enumerate(
            zip(frames, frames[1:]),
            start=1,
        )
        if (
            before.curve_plan_exhausted is False
            and after.curve_plan_exhausted is True
        )
    ]
    if len(exhaustion_transitions) > 1:
        _fail("natural_win_plan_exhaustion_transition_not_unique")
    if exhaustion_transitions:
        exhaustion_index = exhaustion_transitions[0]
        exhaustion = frames[exhaustion_index]
        if any(
            frame.curve_plan_exhausted is not False
            for frame in frames[:exhaustion_index]
        ) or any(
            frame.curve_plan_exhausted is not True
            for frame in frames[exhaustion_index:]
        ):
            _fail("natural_win_plan_exhaustion_phase_invalid")
        if not any(
            any(_plan_counts(frame, curve_indices=curve_indices))
            or any(_plan_enabled(frame, curve_indices=curve_indices))
            for frame in frames[:exhaustion_index]
        ):
            _fail("natural_win_live_feed_context_missing")
        exhaustion_transition_observed = True
        exhaustion_transition_update: int | None = exhaustion.update
        pre_exhaustion = frames[exhaustion_index - 1]
        exhausted_frames = frames[exhaustion_index:]
    else:
        # Retail Jungle2 exposes the global curve-plan flag as already closed
        # on the first readable Board frame.  A bounded exact-step trajectory
        # therefore cannot be required to contain a synthetic False -> True
        # transition.  It must instead prove the stronger invariant that the
        # feed is closed and every per-curve plan remains empty throughout.
        if any(frame.curve_plan_exhausted is not True for frame in frames):
            _fail("natural_win_plan_exhaustion_missing")
        if expected_plan_exhaustion_update is not None:
            _fail("natural_win_plan_exhaustion_update_mismatch")
        exhaustion_index = 0
        exhaustion = frames[0]
        exhaustion_transition_observed = False
        exhaustion_transition_update = None
        pre_exhaustion = None
        exhausted_frames = frames
    for frame in exhausted_frames:
        if any(_plan_counts(frame, curve_indices=curve_indices)) or any(
            _plan_enabled(frame, curve_indices=curve_indices)
        ):
            _fail(f"natural_win_plan_repopulated:{frame.update}")

    runtime_transitions = [
        index
        for index, (before, after) in enumerate(
            zip(frames, frames[1:]),
            start=1,
        )
        if (
            before.board_runtime_flag_157 is True
            and after.board_runtime_flag_157 is False
        )
    ]
    if runtime_transitions:
        _fail("natural_win_runtime_flag_changed")
    if any(frame.board_runtime_flag_157 is not False for frame in frames):
        _fail("natural_win_runtime_flag_retail_invariant_missing")
    if any(frame.board_runtime_i32_f54 != 0 for frame in frames):
        _fail("natural_win_loss_counter_changed")

    empty_transitions = [
        index
        for index, (before, after) in enumerate(
            zip(frames, frames[1:]),
            start=1,
        )
        if (
            _gameplay_count(before, curve_indices=curve_indices) > 0
            and _gameplay_count(after, curve_indices=curve_indices) == 0
        )
    ]
    if len(empty_transitions) != 1:
        _fail("natural_win_empty_transition_not_unique")
    empty_index = empty_transitions[0]
    empty = frames[empty_index]
    if (
        expected_empty_update is not None
        and empty.update != expected_empty_update
    ):
        _fail("natural_win_empty_update_mismatch")
    if _gameplay_count(empty, curve_indices=curve_indices) != 0:
        _fail("natural_win_empty_tick_not_empty")
    for frame in frames[empty_index:]:
        if _gameplay_count(frame, curve_indices=curve_indices) != 0:
            _fail(f"natural_win_terminal_state_repopulated:{frame.update}")
    if (
        score_cross_index > empty_index
        or exhaustion_index > empty_index
        or empty.score < target
    ):
        _fail("natural_win_target_or_exhaustion_after_empty")

    formal_index = empty_index + 1
    if formal_index >= len(frames):
        _fail("natural_win_formal_transition_missing")
    formal = frames[formal_index]
    if (
        expected_formal_transition_update is not None
        and formal.update != expected_formal_transition_update
    ):
        _fail("natural_win_formal_transition_update_mismatch")
    terminal_award = formal.score - empty.score
    if terminal_award != expected_terminal_award_points:
        _fail("natural_win_terminal_award_mismatch")

    preterminal = frames[empty_index - 1]
    preterminal_chamber = (
        preterminal.current_ball_id,
        preterminal.current_color_id,
        preterminal.next_ball_id,
        preterminal.next_color_id,
    )
    if any(value is None for value in preterminal_chamber):
        _fail("natural_win_preterminal_chamber_missing")
    for frame in frames[empty_index:]:
        chamber = (
            frame.current_ball_id,
            frame.current_color_id,
            frame.next_ball_id,
            frame.next_color_id,
        )
        shooter_entities = tuple(
            entity
            for entity in frame.entities
            if entity.zone in {"shooter_current", "shooter_next"}
        )
        if chamber != (None, None, None, None) or shooter_entities:
            _fail(f"natural_win_terminal_chamber_not_cleared:{frame.update}")

    stable_tail_ticks = frames[-1].update - formal.update
    if stable_tail_ticks < minimum_terminal_tail_ticks:
        _fail("natural_win_terminal_tail_too_short")

    return {
        "schema": NATURAL_WIN_SCHEMA,
        "version": NATURAL_WIN_VERSION,
        "status": "PASS",
        "classification": "frame-contract-requires-provenance-binding",
        "curve_count": curve_count,
        "captured_updates": {
            "start": frames[0].update,
            "end": frames[-1].update,
            "count": len(frames),
        },
        "score": {
            "initial": frames[0].score,
            "target": target,
            "cross_update": score_cross.update,
            "score_at_cross": score_cross.score,
            "score_at_empty": empty.score,
            "score_at_formal_transition": formal.score,
            "final": frames[-1].score,
            "monotonic": True,
        },
        "feed_exhaustion": {
            "transition_observed_in_window": (
                exhaustion_transition_observed
            ),
            "last_live_update": (
                None if pre_exhaustion is None else pre_exhaustion.update
            ),
            "transition_update": exhaustion_transition_update,
            "first_observed_exhausted_update": exhaustion.update,
            "planned_counts_before": (
                None
                if pre_exhaustion is None
                else list(
                    _plan_counts(
                        pre_exhaustion,
                        curve_indices=curve_indices,
                    )
                )
            ),
            "add_plan_enabled_before": (
                None
                if pre_exhaustion is None
                else list(
                    _plan_enabled(
                        pre_exhaustion,
                        curve_indices=curve_indices,
                    )
                )
            ),
            "planned_counts_after": list(
                _plan_counts(exhaustion, curve_indices=curve_indices)
            ),
            "add_plan_enabled_after": list(
                _plan_enabled(exhaustion, curve_indices=curve_indices)
            ),
            "repopulated": False,
        },
        "transition": {
            "last_nonempty_update": frames[empty_index - 1].update,
            "empty_update": empty.update,
            "formal_transition_update": formal.update,
            "updates_from_empty_to_formal": formal.update - empty.update,
            "runtime_flag_at_empty": empty.board_runtime_flag_157,
            "runtime_flag_at_formal": formal.board_runtime_flag_157,
            "loss_counter": 0,
            "terminal_award_points": terminal_award,
            "postformal_stable_ticks": stable_tail_ticks,
            "preterminal_chamber": {
                "current_ball_id": preterminal_chamber[0],
                "current_color_id": preterminal_chamber[1],
                "next_ball_id": preterminal_chamber[2],
                "next_color_id": preterminal_chamber[3],
            },
            "terminal_chamber_cleared": True,
            "repopulated": False,
        },
    }


def derive_natural_win_feature_proofs(
    frames: Sequence[TrajectoryFrame],
    *,
    curve_count: int = 1,
) -> tuple[Mapping[str, Any], ...]:
    """Return conservative Gate feature proofs when all win signals exist."""

    if len(frames) < 2:
        return ()
    try:
        report = verify_natural_win_frames(
            frames,
            curve_count=curve_count,
        )
    except PcNaturalWinError:
        return ()
    score = report["score"]
    feed = report["feed_exhaustion"]
    transition = report["transition"]
    return (
        {
            "feature": "natural_win",
            "status": "PASS",
            "score_target": score["target"],
            "score_cross_update": score["cross_update"],
            "feed_exhaustion_update": feed["transition_update"],
            "feed_closed_at_capture_start": (
                not feed["transition_observed_in_window"]
            ),
            "empty_update": transition["empty_update"],
            "formal_transition_update": (
                transition["formal_transition_update"]
            ),
            "proof": (
                "normal-mode retail score crosses its target, the native "
                "feed is closed without repopulation, every gameplay "
                "entity drains, the loss counter remains zero, and the "
                "native terminal award begins on the following update"
            ),
            "sequence": report,
        },
        {
            "feature": "zuma_transition",
            "status": "PASS",
            "feed_exhaustion_update": feed["transition_update"],
            "empty_update": transition["empty_update"],
            "formal_transition_update": (
                transition["formal_transition_update"]
            ),
            "updates_from_empty_to_formal": (
                transition["updates_from_empty_to_formal"]
            ),
            "proof": (
                "the retail Board reaches its first fully empty update and "
                "begins the native terminal score-award sequence exactly "
                "one update later without feed or entity repopulation"
            ),
            "sequence_schema": report["schema"],
            "sequence_version": report["version"],
        },
    )


def compare_natural_win_with_simulator(
    frames: Sequence[TrajectoryFrame],
    *,
    original_root: Path,
    level_id: str,
    hard: bool = False,
    curve_index: int = 0,
) -> dict[str, Any]:
    """Transplant the final natural PC drain and require an exact two-tick win."""

    if curve_index != 0:
        raise ValueError("natural-win simulator diff currently requires curve 0")
    oracle = verify_natural_win_frames(frames, curve_count=1)
    empty_update = int(oracle["transition"]["empty_update"])
    formal_update = int(
        oracle["transition"]["formal_transition_update"]
    )
    by_update = {frame.update: frame for frame in frames}
    before = by_update[empty_update - 1]
    empty = by_update[empty_update]
    formal = by_update[formal_update]

    active = _curve_entities(
        before,
        curve_index=curve_index,
        container_offset=ACTIVE_CHAIN_LIST_OFFSET,
    )
    pending = _curve_entities(
        before,
        curve_index=curve_index,
        container_offset=PENDING_CHAIN_LIST_OFFSET,
    )
    inserting = _curve_entities(
        before,
        curve_index=curve_index,
        container_offset=INSERTING_CHAIN_LIST_OFFSET,
    )
    fired = _fired(before)
    final_chain_drain = bool(active) and not fired and all(
        entity.should_remove is True for entity in active
    )
    final_projectile_exit = not active and len(fired) == 1
    if (
        pending
        or inserting
        or final_chain_drain == final_projectile_exit
    ):
        _fail("natural_win_simulator_final_drain_not_transplantable")
    if (
        before.current_color_id is None
        or before.next_color_id is None
        or before.native_game_time is None
        or before.qrand is None
        or before.thread_crt_rand_state is None
        or before.global_mtrand is None
    ):
        _fail("natural_win_simulator_initial_runtime_state_missing")

    simulator = RevengeSimulator.from_installed(
        level_id,
        root=original_root,
        hard=hard,
        curve_index=curve_index,
        seed=0,
    )
    score_at_level_start = before.score_target - int(
        getattr(simulator.parameters, "zuma_score", 0)
    )
    if not 0 <= score_at_level_start <= before.score:
        _fail("natural_win_simulator_score_target_inconsistent")
    contacts = [
        (
            bool(entity.contact_next)
            if entity.contact_next is not None
            else (
                active[index + 1].curve_distance
                - entity.curve_distance
                <= (
                    float(entity.radius or simulator.config.ball_radius)
                    + float(
                        active[index + 1].radius
                        or simulator.config.ball_radius
                    )
                    + 1e-3
                )
            )
        )
        for index, entity in enumerate(active[:-1])
    ]
    simulator.load_state(
        colors=[entity.color_id for entity in active],
        waypoints=[entity.curve_distance for entity in active],
        contacts=contacts,
        current_color=before.current_color_id,
        next_color=before.next_color_id,
        score=before.score,
        score_at_level_start=score_at_level_start,
    )
    simulator.load_shooter_random_state(
        crt_state=before.thread_crt_rand_state,
        update_count=before.qrand.update_count,
        selected_index=before.qrand.selected_index,
        weights=before.qrand.weights,
        sways=before.qrand.sways,
        last_hit=before.qrand.last_hit,
        previous_hit=before.qrand.previous_hit,
    )
    simulator.load_mtrand_state(
        words=before.global_mtrand.words,
        index=before.global_mtrand.index,
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
    maximum_waypoint_error = 0.0
    for ball, entity in zip(simulator.balls, active, strict=True):
        ball.id = entity.ball_id
        error = abs(float(ball.waypoint) - entity.curve_distance)
        maximum_waypoint_error = max(maximum_waypoint_error, error)
        if not math.isfinite(error) or error != 0.0:
            _fail("natural_win_simulator_initial_waypoint_mismatch")
        for field_name in copied_fields:
            value = getattr(entity, field_name)
            if value is not None:
                setattr(ball, field_name, value)
        for field_name in float_fields:
            value = getattr(entity, field_name)
            if value is not None:
                setattr(ball, field_name, np.float32(value))

    if final_projectile_exit:
        source_projectile = fired[0]
        projectile_values = (
            source_projectile.position_x,
            source_projectile.position_y,
            source_projectile.velocity_x,
            source_projectile.velocity_y,
            source_projectile.radius,
            source_projectile.merge_speed,
        )
        if (
            source_projectile.object_kind != "bullet"
            or source_projectile.fired is not False
            or any(
                value is None or not math.isfinite(value)
                for value in projectile_values
            )
            or source_projectile.curve_points is None
            or source_projectile.gap_entry_count
            != len(source_projectile.gap_entries)
        ):
            _fail("natural_win_simulator_final_projectile_invalid")
        curve_points = {
            index: value
            for index, value in enumerate(
                source_projectile.curve_points[: simulator.curve_count]
            )
            if value >= 0
        }
        simulator.free_projectiles.append(
            Projectile(
                id=source_projectile.ball_id,
                color=source_projectile.color_id,
                position=np.asarray(
                    (
                        source_projectile.position_x,
                        source_projectile.position_y,
                    ),
                    dtype=np.float32,
                ),
                velocity=np.asarray(
                    (
                        source_projectile.velocity_x,
                        source_projectile.velocity_y,
                    ),
                    dtype=np.float32,
                ),
                radius=int(source_projectile.radius),
                just_fired=False,
                hit_percent=np.float32(
                    source_projectile.merge_progress or 0.0
                ),
                merge_speed=np.float32(source_projectile.merge_speed),
                waypoint=np.float32(source_projectile.curve_distance),
                curve_point=curve_points.get(0, 0),
                curve_points=curve_points,
                gap_info=[
                    (boundary_ball_id, gap_distance)
                    for _, gap_distance, boundary_ball_id
                    in source_projectile.gap_entries
                ],
            )
        )

    simulator._next_id = max(
        entity.ball_id for entity in before.entities
    ) + 1
    simulator.tick_count = before.update
    simulator.native_game_time = before.native_game_time
    simulator.stop_adding = True
    simulator.zuma_reached = True
    simulator.current_bar_size = simulator.config.zuma_bar_width
    simulator.target_bar_size = simulator.config.zuma_bar_width
    simulator.has_reached_rollout = True
    simulator.has_reached_cruising_speed = True
    simulator.first_chain_end = simulator._first_chain_end()
    simulator._have_sets = any(ball.exploding for ball in simulator.balls)
    powerup = before.curve_powerup_state(curve_index)
    if powerup is not None:
        simulator.powerup_last_any_spawn_time = powerup.last_any_spawn_time
        simulator.powerup_last_spawn_times = list(powerup.last_spawn_times)
        simulator.powerup_cooldown_times = list(powerup.cooldown_times)
        simulator.powerup_spawn_counts = list(powerup.spawn_counts)
        simulator.powerup_field_124_by_type = list(
            powerup.field_124_by_type
        )
        simulator.active_powerup_color_counts = list(
            powerup.active_color_counts
        )
        simulator.slow_count = powerup.slow_ticks
        simulator.backward_count = powerup.reverse_ticks
        simulator.powerup_triggered = powerup.powerup_triggered

    empty_events = simulator.tick()
    initial_gameplay_entity_count = len(active) + len(fired)
    if (
        simulator.tick_count != empty.update
        or simulator.native_game_time != empty.native_game_time
        or simulator.balls
        or simulator.pending_colors
        or simulator.merging_projectiles
        or simulator.free_projectiles
        or not simulator.win_pending
        or simulator.outcome is not None
        or empty_events.balls_removed != len(active)
        or empty_events.win_pending is not True
        or simulator.score != empty.score
        or simulator.current_color != empty.current_color_id
        or simulator.next_color != empty.next_color_id
    ):
        _fail("natural_win_simulator_empty_tick_mismatch")

    formal_events = simulator.tick()
    if (
        simulator.tick_count != formal.update
        or simulator.native_game_time != formal.native_game_time
        or simulator.win_pending
        or simulator.outcome != "win"
        or formal_events.outcome != "win"
        or simulator.score != formal.score
        or simulator.current_color != formal.current_color_id
        or simulator.next_color != formal.next_color_id
    ):
        _fail("natural_win_simulator_formal_tick_mismatch")

    return {
        "schema": NATURAL_WIN_SIMULATOR_DIFF_SCHEMA,
        "version": NATURAL_WIN_SIMULATOR_DIFF_VERSION,
        "status": "PASS",
        "start_update": before.update,
        "empty_update": empty.update,
        "formal_transition_update": formal.update,
        "end_update": formal.update,
        "ticks_compared": 2,
        "terminal_object_kind": (
            "chain" if final_chain_drain else "free_projectile"
        ),
        "initial_chain_count": len(active),
        "initial_free_projectile_count": len(fired),
        "initial_gameplay_entity_count": initial_gameplay_entity_count,
        "gameplay_entities_removed": initial_gameplay_entity_count,
        "balls_removed": empty_events.balls_removed,
        "maximum_waypoint_error": maximum_waypoint_error,
        "score_target_achieved": before.score >= before.score_target,
        "plan_exhausted": before.curve_plan_exhausted is True,
        "stop_adding_transplanted": True,
        "pc_empty_state_matched": True,
        "pc_formal_state_matched": True,
        "pc_terminal_chamber_cleared": True,
        "scenario": {
            "level_id": level_id,
            "hard": hard,
            "curve_index": curve_index,
            "score_at_level_start": score_at_level_start,
            "shooter_rng_state_restored": True,
            "global_mtrand_state_restored": True,
        },
        "oracle": {
            "schema": oracle["schema"],
            "version": oracle["version"],
            "status": oracle["status"],
            "score_target": oracle["score"]["target"],
            "score_cross_update": oracle["score"]["cross_update"],
            "feed_exhaustion_update": (
                oracle["feed_exhaustion"]["transition_update"]
            ),
            "feed_closed_at_capture_start": (
                not oracle["feed_exhaustion"][
                    "transition_observed_in_window"
                ]
            ),
            "empty_update": empty.update,
            "formal_transition_update": formal.update,
        },
    }


__all__ = [
    "NATURAL_WIN_SCHEMA",
    "NATURAL_WIN_SIMULATOR_DIFF_SCHEMA",
    "NATURAL_WIN_SIMULATOR_DIFF_VERSION",
    "NATURAL_WIN_VERSION",
    "PcNaturalWinError",
    "compare_natural_win_with_simulator",
    "derive_natural_win_feature_proofs",
    "verify_natural_win_frames",
]
