"""Strict retail power-up trigger verification from full trajectories."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .pc_memory_trajectory import (
    ACTIVE_CHAIN_LIST_OFFSET,
    POWERUP_TYPE_COUNT,
    TrajectoryCurvePowerupState,
    TrajectoryEntity,
    TrajectoryFrame,
    load_memory_trajectory,
)


POWERUP_TRIGGER_SCHEMA = "zuma-rl.pc-powerup-trigger-verification"
POWERUP_TRIGGER_VERSION = 1
REVERSE_POWERUP_TYPE = 3
PROXIMITY_BOMB_POWERUP_TYPE = 0
SLOW_POWERUP_TYPE = 1


class PcPowerupTriggerError(ValueError):
    """A trajectory does not exhibit the required retail trigger."""


def _fail(code: str) -> None:
    raise PcPowerupTriggerError(code)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _curve_state(
    frame: TrajectoryFrame,
    curve_index: int,
) -> TrajectoryCurvePowerupState:
    state = frame.curve_powerup_state(curve_index)
    if (
        state is None
        or len(state.last_spawn_times) != POWERUP_TYPE_COUNT
        or len(state.cooldown_times) != POWERUP_TYPE_COUNT
        or len(state.spawn_counts) != POWERUP_TYPE_COUNT
        or len(state.field_124_by_type) != POWERUP_TYPE_COUNT
        or len(state.active_color_counts) != 6
    ):
        _fail("powerup_trigger_curve_state_missing")
    return state


def _active_chain(
    frame: TrajectoryFrame,
    curve_index: int,
) -> Mapping[int, TrajectoryEntity]:
    zone = f"curve:{curve_index}:list:{ACTIVE_CHAIN_LIST_OFFSET:03x}"
    result: dict[int, TrajectoryEntity] = {}
    for entity in frame.entities:
        if entity.object_kind != "ball" or entity.zone != zone:
            continue
        if entity.ball_id in result:
            _fail("powerup_trigger_duplicate_chain_ball_id")
        result[entity.ball_id] = entity
    return result


def _single_changed_index(
    before: Sequence[int],
    after: Sequence[int],
    expected_index: int,
    code: str,
) -> None:
    changed = [
        index
        for index, (left, right) in enumerate(
            zip(before, after, strict=True)
        )
        if left != right
    ]
    if changed != [expected_index]:
        _fail(code)


def _powerup_signature(entity: TrajectoryEntity) -> tuple[Any, ...]:
    fields = (
        entity.powerup_previous_type,
        entity.powerup_primary_type,
        entity.powerup_secondary_type,
        entity.powerup_previous_ticks,
        entity.powerup_lifetime_ticks,
        entity.powerup_transition_ticks,
        entity.powerup_visual_scale,
        entity.powerup_visual_step,
        entity.powerup_visual_index,
    )
    if any(value is None for value in fields):
        _fail("powerup_trigger_target_raw_fields_missing")
    return fields


def _require_chain_ball(
    chain: Mapping[int, TrajectoryEntity],
    ball_id: int,
    code: str,
) -> TrajectoryEntity:
    entity = chain.get(ball_id)
    if entity is None:
        _fail(code)
    return entity


def verify_powerup_trigger_frames(
    frames: Sequence[TrajectoryFrame],
    *,
    curve_index: int,
    trigger_ball_id: int,
    trigger_color_id: int,
    powerup_type: int,
    expected_newly_exploding_ids: Sequence[int],
    expected_score_delta: int,
    expected_concurrent_score_delta: int = 0,
    expected_direct_match_ids: Sequence[int] | None = None,
    movement_reference_ball_id: int | None = None,
    expected_trigger_update: int | None = None,
    expected_trigger_count: int = 1,
    reverse_ticks: int = 300,
    reverse_speed: float = 1.0,
    slow_ticks: int = 800,
    bomb_collision_pad: int = 56,
    require_explosion_removal: bool = True,
) -> dict[str, Any]:
    """Verify one trigger, including any independently bound concurrent score."""

    if (
        curve_index < 0
        or trigger_ball_id < 0
        or not 0 <= trigger_color_id < 6
        or not 0 <= powerup_type < POWERUP_TYPE_COUNT
    ):
        raise ValueError("power-up trigger identifiers are invalid")
    if expected_trigger_count < 1 or expected_score_delta < 0:
        raise ValueError("expected trigger count or score delta is invalid")
    if (
        isinstance(expected_concurrent_score_delta, bool)
        or not isinstance(expected_concurrent_score_delta, int)
        or expected_concurrent_score_delta < 0
    ):
        raise ValueError(
            "expected concurrent score delta must be a nonnegative integer"
        )
    if reverse_ticks < 1 or slow_ticks < 1:
        raise ValueError("power-up countdown durations must be positive")
    if bomb_collision_pad < 0:
        raise ValueError("bomb_collision_pad must be nonnegative")
    if not math.isfinite(reverse_speed) or reverse_speed <= 0.0:
        raise ValueError("reverse_speed must be finite and positive")
    expected_exploding = tuple(sorted(set(expected_newly_exploding_ids)))
    if (
        not expected_exploding
        or len(expected_exploding) != len(expected_newly_exploding_ids)
        or any(ball_id < 0 for ball_id in expected_exploding)
        or trigger_ball_id not in expected_exploding
    ):
        raise ValueError("expected newly exploding IDs are invalid")
    direct_match_ids = (
        ()
        if expected_direct_match_ids is None
        else tuple(sorted(set(expected_direct_match_ids)))
    )
    if (
        len(direct_match_ids)
        != len(expected_direct_match_ids or ())
        or any(ball_id < 0 for ball_id in direct_match_ids)
        or not set(direct_match_ids).issubset(expected_exploding)
    ):
        raise ValueError("expected direct match IDs are invalid")
    if len(frames) < 4:
        _fail("powerup_trigger_trajectory_too_short")

    states = tuple(_curve_state(frame, curve_index) for frame in frames)
    chains = tuple(_active_chain(frame, curve_index) for frame in frames)
    if any(frame.native_game_time is None for frame in frames):
        _fail("powerup_trigger_native_time_missing")
    for before, after in zip(frames, frames[1:]):
        if (
            after.update != before.update + 1
            or after.native_game_time != before.native_game_time + 1
        ):
            _fail("powerup_trigger_time_not_contiguous")

    trigger_candidates: list[int] = []
    for index, (before, after) in enumerate(zip(states, states[1:])):
        delta = (
            after.field_124_by_type[powerup_type]
            - before.field_124_by_type[powerup_type]
        )
        if delta != 0:
            trigger_candidates.append(index + 1)
    if len(trigger_candidates) != 1:
        _fail("powerup_trigger_transition_not_unique")

    trigger_index = trigger_candidates[0]
    if trigger_index < 1 or trigger_index + 1 >= len(frames):
        _fail("powerup_trigger_context_not_covered")
    before_frame = frames[trigger_index - 1]
    trigger_frame = frames[trigger_index]
    next_frame = frames[trigger_index + 1]
    before_state = states[trigger_index - 1]
    trigger_state = states[trigger_index]
    next_state = states[trigger_index + 1]
    before_chain = chains[trigger_index - 1]
    trigger_chain = chains[trigger_index]
    next_chain = chains[trigger_index + 1]

    if (
        expected_trigger_update is not None
        and trigger_frame.update != expected_trigger_update
    ):
        _fail("powerup_trigger_update_mismatch")
    if (
        trigger_state.field_124_by_type[powerup_type]
        != before_state.field_124_by_type[powerup_type]
        + expected_trigger_count
    ):
        _fail("powerup_trigger_counter_delta_mismatch")
    _single_changed_index(
        before_state.field_124_by_type,
        trigger_state.field_124_by_type,
        powerup_type,
        "powerup_trigger_counter_mutation_invalid",
    )
    if (
        before_state.powerup_triggered
        or not trigger_state.powerup_triggered
        or trigger_state.cooldown_times[powerup_type]
        != trigger_frame.native_game_time
    ):
        _fail("powerup_trigger_flag_or_cooldown_mismatch")
    _single_changed_index(
        before_state.cooldown_times,
        trigger_state.cooldown_times,
        powerup_type,
        "powerup_trigger_cooldown_mutation_invalid",
    )
    if (
        trigger_state.active_color_counts[trigger_color_id]
        != before_state.active_color_counts[trigger_color_id]
        - expected_trigger_count
        or trigger_state.active_color_counts[trigger_color_id] < 0
    ):
        _fail("powerup_trigger_active_color_delta_mismatch")
    _single_changed_index(
        before_state.active_color_counts,
        trigger_state.active_color_counts,
        trigger_color_id,
        "powerup_trigger_active_color_mutation_invalid",
    )
    if (
        trigger_state.last_any_spawn_time
        != before_state.last_any_spawn_time
        or trigger_state.last_spawn_times
        != before_state.last_spawn_times
        or trigger_state.spawn_counts != before_state.spawn_counts
    ):
        _fail("powerup_trigger_manager_side_effect")

    before_target = _require_chain_ball(
        before_chain,
        trigger_ball_id,
        "powerup_trigger_target_missing_before",
    )
    trigger_target = _require_chain_ball(
        trigger_chain,
        trigger_ball_id,
        "powerup_trigger_target_missing_at_trigger",
    )
    if (
        before_target.color_id != trigger_color_id
        or trigger_target.color_id != trigger_color_id
        or before_target.exploding is not False
        or trigger_target.exploding is not True
        # The active-chain traversal may encounter the triggering ball before
        # or after the match commits within the same retail Board update.  A
        # coherent post-update sample therefore observes the first explosion
        # frame as either zero or one; later values mean the transition was
        # not captured at its first tick.
        or trigger_target.explode_frame not in {0, 1}
        or before_target.powerup_primary_type != powerup_type
    ):
        _fail("powerup_trigger_target_transition_invalid")
    before_signature = _powerup_signature(before_target)
    trigger_signature = _powerup_signature(trigger_target)
    if (
        trigger_signature[:4] != before_signature[:4]
        or trigger_signature[4] != before_signature[4] - 1
        or trigger_signature[5:] != before_signature[5:]
    ):
        _fail("powerup_trigger_target_powerup_fields_mutated")

    newly_exploding = tuple(
        sorted(
            ball_id
            for ball_id, entity in trigger_chain.items()
            if (
                entity.exploding is True
                and (
                    ball_id not in before_chain
                    or before_chain[ball_id].exploding is not True
                )
            )
        )
    )
    if newly_exploding != expected_exploding:
        _fail("powerup_trigger_exploding_set_mismatch")
    if any(
        trigger_chain[ball_id].combo_count != 0
        for ball_id in newly_exploding
    ):
        _fail("powerup_trigger_expected_base_score_combo_mismatch")
    score_delta = trigger_frame.score - before_frame.score
    base_explosion_score_delta = 10 * len(newly_exploding)
    if (
        score_delta != expected_score_delta
        or score_delta
        != base_explosion_score_delta + expected_concurrent_score_delta
    ):
        _fail("powerup_trigger_score_delta_mismatch")

    bomb_geometry: dict[str, Any] | None = None
    if powerup_type == PROXIMITY_BOMB_POWERUP_TYPE:
        if not direct_match_ids or trigger_target.radius is None:
            _fail("powerup_trigger_bomb_geometry_contract_missing")
        spatial_ids: list[int] = []
        included_distances: list[tuple[int, float, float]] = []
        excluded_distances: list[tuple[int, float, float]] = []
        for ball_id, entity in trigger_chain.items():
            if (
                ball_id in before_chain
                and before_chain[ball_id].exploding is True
            ):
                continue
            if entity.radius is None:
                _fail("powerup_trigger_bomb_radius_missing")
            distance = math.hypot(
                entity.position_x - trigger_target.position_x,
                entity.position_y - trigger_target.position_y,
            )
            threshold = (
                math.floor(trigger_target.radius + 0.5)
                + 2 * bomb_collision_pad
                + math.floor(entity.radius + 0.5)
            )
            record = (ball_id, distance, float(threshold))
            if distance * distance < threshold * threshold:
                spatial_ids.append(ball_id)
                included_distances.append(record)
            else:
                excluded_distances.append(record)
        predicted = tuple(sorted(set(direct_match_ids) | set(spatial_ids)))
        if predicted != newly_exploding:
            _fail("powerup_trigger_bomb_spatial_set_mismatch")
        farthest_inside = max(
            included_distances,
            key=lambda item: item[1],
        )
        nearest_outside = min(
            excluded_distances,
            key=lambda item: item[1],
            default=None,
        )
        bomb_geometry = {
            "collision_pad_per_ball": bomb_collision_pad,
            "retail_radius_for_18px_balls": (
                18 + 2 * bomb_collision_pad + 18
            ),
            "direct_match_ball_ids": list(direct_match_ids),
            "spatially_selected_ball_ids": sorted(spatial_ids),
            "farthest_selected": {
                "ball_id": farthest_inside[0],
                "distance": farthest_inside[1],
                "threshold": farthest_inside[2],
            },
            "nearest_rejected": (
                None
                if nearest_outside is None
                else {
                    "ball_id": nearest_outside[0],
                    "distance": nearest_outside[1],
                    "threshold": nearest_outside[2],
                }
            ),
            "strict_less_than_threshold": True,
        }

    movement: dict[str, Any] | None = None
    if movement_reference_ball_id is not None:
        if trigger_index + 2 >= len(frames):
            _fail("powerup_trigger_post_trigger_movement_not_covered")
        post_next_chain = chains[trigger_index + 2]
        trigger_reference = _require_chain_ball(
            trigger_chain,
            movement_reference_ball_id,
            "powerup_trigger_movement_reference_missing_at_trigger",
        )
        next_reference = _require_chain_ball(
            next_chain,
            movement_reference_ball_id,
            "powerup_trigger_movement_reference_missing_after_trigger",
        )
        post_next_reference = _require_chain_ball(
            post_next_chain,
            movement_reference_ball_id,
            "powerup_trigger_movement_reference_missing_two_after_trigger",
        )
        if (
            trigger_reference.exploding is not False
            or next_reference.exploding is not False
            or post_next_reference.exploding is not False
            or not math.isclose(
                next_reference.curve_distance,
                trigger_reference.curve_distance - reverse_speed,
                rel_tol=0.0,
                abs_tol=1e-5,
            )
            or not math.isclose(
                post_next_reference.curve_distance,
                next_reference.curve_distance - reverse_speed,
                rel_tol=0.0,
                abs_tol=1e-5,
            )
            or next_reference.backwards_count != 0
            or not math.isclose(
                float(next_reference.backwards_speed),
                reverse_speed,
                rel_tol=0.0,
                abs_tol=1e-7,
            )
        ):
            _fail("powerup_trigger_post_trigger_reverse_movement_mismatch")
        movement = {
            "reference_ball_id": movement_reference_ball_id,
            "sample_phase": "two_complete_post_trigger_intervals",
            "trigger_distance": trigger_reference.curve_distance,
            "next_distance": next_reference.curve_distance,
            "post_next_distance": post_next_reference.curve_distance,
            "distance_per_tick": reverse_speed,
        }

    if powerup_type == REVERSE_POWERUP_TYPE:
        if (
            before_state.reverse_ticks != 0
            or trigger_state.reverse_ticks != reverse_ticks
            or next_state.reverse_ticks != reverse_ticks - 1
            or not math.isclose(
                trigger_state.reverse_speed,
                reverse_speed,
                rel_tol=0.0,
                abs_tol=1e-7,
            )
        ):
            _fail("powerup_trigger_reverse_initial_state_mismatch")
        for offset, state in enumerate(states[trigger_index:]):
            if state.reverse_ticks != max(reverse_ticks - offset, 0):
                _fail(
                    "powerup_trigger_reverse_countdown_mismatch:"
                    f"{frames[trigger_index + offset].update}"
                )
        for before_timer, after_timer in zip(
            states[trigger_index - 1 : trigger_index + 2],
            states[trigger_index : trigger_index + 2],
        ):
            if after_timer.slow_ticks != max(before_timer.slow_ticks - 1, 0):
                _fail("powerup_trigger_reverse_changed_slow_countdown")
    else:
        for before_timer, after_timer in zip(
            states[trigger_index - 1 : trigger_index + 2],
            states[trigger_index : trigger_index + 2],
        ):
            if after_timer.reverse_ticks != max(
                before_timer.reverse_ticks - 1,
                0,
            ):
                _fail("powerup_trigger_nonreverse_changed_reverse_countdown")

    if powerup_type == SLOW_POWERUP_TYPE:
        if (
            before_state.slow_ticks >= 1_000
            or trigger_state.slow_ticks != slow_ticks
            or next_state.slow_ticks != slow_ticks - 1
        ):
            _fail("powerup_trigger_slow_initial_state_mismatch")
        for offset, state in enumerate(states[trigger_index:]):
            if state.slow_ticks != max(slow_ticks - offset, 0):
                _fail(
                    "powerup_trigger_slow_countdown_mismatch:"
                    f"{frames[trigger_index + offset].update}"
                )
    else:
        for before_timer, after_timer in zip(
            states[trigger_index - 1 : trigger_index + 2],
            states[trigger_index : trigger_index + 2],
        ):
            if after_timer.slow_ticks != max(before_timer.slow_ticks - 1, 0):
                _fail("powerup_trigger_nonslow_changed_slow_countdown")

    stable_counter = trigger_state.field_124_by_type[powerup_type]
    stable_cooldown = trigger_state.cooldown_times[powerup_type]
    stable_active_colors = trigger_state.active_color_counts
    stable_spawn_counts = trigger_state.spawn_counts
    stable_last_spawns = trigger_state.last_spawn_times
    stable_last_any = trigger_state.last_any_spawn_time
    for frame, state in zip(
        frames[trigger_index:],
        states[trigger_index:],
        strict=True,
    ):
        if (
            not state.powerup_triggered
            or state.field_124_by_type[powerup_type] != stable_counter
            or state.cooldown_times[powerup_type] != stable_cooldown
            or state.active_color_counts != stable_active_colors
            or state.spawn_counts != stable_spawn_counts
            or state.last_spawn_times != stable_last_spawns
            or state.last_any_spawn_time != stable_last_any
            or frame.score != trigger_frame.score
        ):
            _fail(
                "powerup_trigger_post_state_not_stable:"
                f"{frame.update}"
            )

    for chain in chains[trigger_index + 1 :]:
        target = chain.get(trigger_ball_id)
        if target is None:
            break
        if (
            target.exploding is not True
            or _powerup_signature(target) != trigger_signature
        ):
            _fail("powerup_trigger_target_fields_not_retained")

    removals: dict[str, int] = {}
    for ball_id in newly_exploding:
        last_seen = max(
            frame.update
            for frame, chain in zip(frames, chains, strict=True)
            if ball_id in chain
        )
        if require_explosion_removal and ball_id in chains[-1]:
            _fail("powerup_trigger_explosion_removal_not_covered")
        removals[str(ball_id)] = last_seen

    return {
        "schema": POWERUP_TRIGGER_SCHEMA,
        "version": POWERUP_TRIGGER_VERSION,
        "status": "PASS",
        "trajectory": {
            "start_update": frames[0].update,
            "end_update": frames[-1].update,
            "tick_count": len(frames),
        },
        "target": {
            "curve_index": curve_index,
            "ball_id": trigger_ball_id,
            "color_id": trigger_color_id,
            "powerup_type": powerup_type,
        },
        "trigger": {
            "update": trigger_frame.update,
            "native_game_time": trigger_frame.native_game_time,
            "newly_exploding_ball_ids": list(newly_exploding),
            "score_before": before_frame.score,
            "score_after": trigger_frame.score,
            "score_delta": score_delta,
            "base_explosion_score_delta": base_explosion_score_delta,
            "concurrent_score_delta": expected_concurrent_score_delta,
            "counter_before": (
                before_state.field_124_by_type[powerup_type]
            ),
            "counter_after": stable_counter,
            "cooldown_before": before_state.cooldown_times[powerup_type],
            "cooldown_after": stable_cooldown,
            "active_color_count_before": (
                before_state.active_color_counts[trigger_color_id]
            ),
            "active_color_count_after": (
                trigger_state.active_color_counts[trigger_color_id]
            ),
            "target_lifetime_before": (
                before_target.powerup_lifetime_ticks
            ),
            "target_lifetime_after": (
                trigger_target.powerup_lifetime_ticks
            ),
            "target_explode_frame_at_trigger": (
                trigger_target.explode_frame
            ),
        },
        "reverse": (
            {
                "ticks_at_trigger": trigger_state.reverse_ticks,
                "ticks_next_update": next_state.reverse_ticks,
                "speed": trigger_state.reverse_speed,
                "movement": movement,
            }
            if powerup_type == REVERSE_POWERUP_TYPE
            else None
        ),
        "slow": (
            {
                "ticks_at_trigger": trigger_state.slow_ticks,
                "ticks_next_update": next_state.slow_ticks,
            }
            if powerup_type == SLOW_POWERUP_TYPE
            else None
        ),
        "bomb_geometry": bomb_geometry,
        "last_seen_update_by_exploding_ball": removals,
        "trigger_flag_persists_through_update": frames[-1].update,
    }


def verify_powerup_trigger(
    trajectory_path: Path,
    **kwargs: Any,
) -> dict[str, Any]:
    """Load and verify one trigger while retaining artifact provenance."""

    trajectory_path = trajectory_path.resolve()
    report = verify_powerup_trigger_frames(
        load_memory_trajectory(trajectory_path),
        **kwargs,
    )
    report["trajectory"] = {
        **report["trajectory"],
        "artifact": str(trajectory_path),
        "artifact_sha256": _sha256_path(trajectory_path),
    }
    return report


__all__ = [
    "POWERUP_TRIGGER_SCHEMA",
    "POWERUP_TRIGGER_VERSION",
    "PcPowerupTriggerError",
    "verify_powerup_trigger",
    "verify_powerup_trigger_frames",
]
