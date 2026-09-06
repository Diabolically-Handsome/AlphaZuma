"""Strict retail power-up lifetime verification from full memory trajectories."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import struct
from typing import Any, Sequence

from .pc_memory_trajectory import (
    POWERUP_NONE_TYPE,
    POWERUP_TYPE_COUNT,
    TrajectoryCurvePowerupState,
    TrajectoryEntity,
    TrajectoryFrame,
    load_memory_trajectory,
)


POWERUP_LIFECYCLE_SCHEMA = "zuma-rl.pc-powerup-lifecycle-verification"
POWERUP_LIFECYCLE_VERSION = 1


class PcPowerupLifecycleError(ValueError):
    """A full trajectory does not exhibit the required retail lifecycle."""


def _fail(code: str) -> None:
    raise PcPowerupLifecycleError(code)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _target_ball(
    frame: TrajectoryFrame,
    *,
    ball_id: int,
    curve_index: int,
    color_id: int,
) -> TrajectoryEntity:
    entity = frame.entities_by_id.get(ball_id)
    if (
        entity is None
        or entity.object_kind != "ball"
        or entity.zone != f"curve:{curve_index}:list:05c"
        or entity.color_id != color_id
    ):
        _fail("powerup_lifecycle_target_ball_missing")
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
        _fail("powerup_lifecycle_raw_fields_missing")
    return entity


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
        _fail("powerup_lifecycle_curve_state_missing")
    return state


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


def verify_powerup_lifecycle_frames(
    frames: Sequence[TrajectoryFrame],
    *,
    ball_id: int,
    curve_index: int,
    color_id: int,
    powerup_type: int,
    expected_expiration_update: int | None = None,
    transition_ticks: int = 100,
    previous_retention_ticks: int = 150,
    visual_step: float = 0.04,
) -> dict[str, Any]:
    """Verify countdown, expiry bookkeeping, fade, and residual cleanup."""

    if ball_id < 0 or curve_index < 0 or not 0 <= color_id < 6:
        raise ValueError("ball, curve, or color identifier is invalid")
    if not 0 <= powerup_type < POWERUP_TYPE_COUNT:
        raise ValueError("powerup_type must be in [0, 13]")
    if transition_ticks < 1 or previous_retention_ticks < 2:
        raise ValueError("lifecycle durations are invalid")
    if not math.isfinite(visual_step) or visual_step <= 0.0:
        raise ValueError("visual_step must be finite and positive")

    if len(frames) < previous_retention_ticks + 1:
        _fail("powerup_lifecycle_trajectory_too_short")
    balls = tuple(
        _target_ball(
            frame,
            ball_id=ball_id,
            curve_index=curve_index,
            color_id=color_id,
        )
        for frame in frames
    )
    states = tuple(
        _curve_state(frame, curve_index)
        for frame in frames
    )
    if any(frame.native_game_time is None for frame in frames):
        _fail("powerup_lifecycle_native_time_missing")
    for before, after in zip(frames, frames[1:]):
        if (
            after.update != before.update + 1
            or after.native_game_time != before.native_game_time + 1
        ):
            _fail("powerup_lifecycle_time_not_contiguous")

    expiration_candidates = [
        index
        for index, (before, after) in enumerate(
            zip(balls, balls[1:])
        )
        if before.powerup_lifetime_ticks == 1
        and after.powerup_lifetime_ticks == 0
        and before.powerup_primary_type == powerup_type
        and after.powerup_previous_type == powerup_type
        and after.powerup_transition_ticks == transition_ticks
    ]
    if len(expiration_candidates) != 1:
        _fail("powerup_lifecycle_expiration_not_unique")
    before_expiration_index = expiration_candidates[0]
    expiration_index = before_expiration_index + 1
    expiration_frame = frames[expiration_index]
    expiration_ball = balls[expiration_index]
    if (
        expected_expiration_update is not None
        and expiration_frame.update != expected_expiration_update
    ):
        _fail("powerup_lifecycle_expiration_update_mismatch")

    countdown_start = before_expiration_index
    while (
        countdown_start > 0
        and balls[countdown_start - 1].powerup_lifetime_ticks
        == balls[countdown_start].powerup_lifetime_ticks + 1
        and balls[countdown_start - 1].powerup_primary_type
        == powerup_type
    ):
        countdown_start -= 1
    initial_lifetime = balls[countdown_start].powerup_lifetime_ticks
    if initial_lifetime is None or initial_lifetime < 1:
        _fail("powerup_lifecycle_countdown_invalid")
    for offset, ball in enumerate(
        balls[countdown_start:expiration_index]
    ):
        if (
            ball.powerup_lifetime_ticks != initial_lifetime - offset
            or ball.powerup_previous_ticks != 0
            or ball.powerup_previous_type != POWERUP_NONE_TYPE
            or ball.powerup_primary_type != powerup_type
            or ball.powerup_secondary_type != POWERUP_NONE_TYPE
            or ball.powerup_transition_ticks != 0
        ):
            _fail("powerup_lifecycle_countdown_invalid")

    if (
        expiration_ball.powerup_previous_ticks
        != previous_retention_ticks - 1
        or expiration_ball.powerup_previous_type != powerup_type
        or expiration_ball.powerup_primary_type != powerup_type
        or expiration_ball.powerup_secondary_type != POWERUP_NONE_TYPE
        or expiration_ball.powerup_lifetime_ticks != 0
        or expiration_ball.powerup_visual_index != -1
        or not math.isclose(
            float(expiration_ball.powerup_visual_scale),
            1.0 + transition_ticks * visual_step,
            rel_tol=0.0,
            abs_tol=1e-5,
        )
    ):
        _fail("powerup_lifecycle_expiration_state_invalid")

    required_end_update = (
        expiration_frame.update + previous_retention_ticks - 1
    )
    if frames[-1].update < required_end_update:
        _fail("powerup_lifecycle_cleanup_not_covered")
    primary_clear_update = expiration_frame.update + transition_ticks
    previous_clear_update = required_end_update
    for frame, ball in zip(
        frames[expiration_index:],
        balls[expiration_index:],
        strict=True,
    ):
        delta = frame.update - expiration_frame.update
        expected_transition = max(transition_ticks - delta, 0)
        expected_previous = max(
            previous_retention_ticks - 1 - delta,
            0,
        )
        expected_primary = (
            powerup_type
            if frame.update < primary_clear_update
            else POWERUP_NONE_TYPE
        )
        expected_previous_type = (
            powerup_type
            if frame.update < previous_clear_update
            else POWERUP_NONE_TYPE
        )
        if (
            ball.powerup_lifetime_ticks != 0
            or ball.powerup_transition_ticks != expected_transition
            or ball.powerup_previous_ticks != expected_previous
            or ball.powerup_primary_type != expected_primary
            or ball.powerup_previous_type != expected_previous_type
            or ball.powerup_secondary_type != POWERUP_NONE_TYPE
            or ball.powerup_visual_index != -1
            or not math.isclose(
                float(ball.powerup_visual_step),
                visual_step,
                rel_tol=0.0,
                abs_tol=1e-7,
            )
            or not math.isclose(
                float(ball.powerup_visual_scale),
                1.0 + expected_transition * visual_step,
                rel_tol=0.0,
                abs_tol=1e-4,
            )
        ):
            _fail(
                "powerup_lifecycle_fade_or_cleanup_mismatch:"
                f"{frame.update}"
            )

    before_state = states[before_expiration_index]
    expiration_state = states[expiration_index]
    before_native_time = frames[
        before_expiration_index
    ].native_game_time
    expiration_native_time = expiration_frame.native_game_time
    if (
        before_native_time is None
        or expiration_native_time is None
        or expiration_state.cooldown_times[powerup_type]
        != expiration_native_time
        or before_state.cooldown_times[powerup_type]
        == expiration_state.cooldown_times[powerup_type]
        or expiration_state.active_color_counts[color_id]
        != before_state.active_color_counts[color_id] - 1
        or expiration_state.last_any_spawn_time
        != before_state.last_any_spawn_time
        or expiration_state.last_spawn_times
        != before_state.last_spawn_times
        or expiration_state.spawn_counts != before_state.spawn_counts
        or expiration_state.field_124_by_type
        != before_state.field_124_by_type
    ):
        _fail("powerup_lifecycle_manager_expiration_mismatch")
    _single_changed_index(
        before_state.cooldown_times,
        expiration_state.cooldown_times,
        powerup_type,
        "powerup_lifecycle_manager_cooldown_mutation_invalid",
    )
    _single_changed_index(
        before_state.active_color_counts,
        expiration_state.active_color_counts,
        color_id,
        "powerup_lifecycle_manager_color_mutation_invalid",
    )
    cooldown_time = expiration_state.cooldown_times[powerup_type]
    active_color_count = expiration_state.active_color_counts[color_id]
    for frame, state in zip(
        frames[expiration_index:],
        states[expiration_index:],
        strict=True,
    ):
        if (
            state.cooldown_times[powerup_type] != cooldown_time
            or state.active_color_counts[color_id]
            != active_color_count
        ):
            _fail(
                "powerup_lifecycle_manager_state_not_stable:"
                f"{frame.update}"
            )

    return {
        "schema": POWERUP_LIFECYCLE_SCHEMA,
        "version": POWERUP_LIFECYCLE_VERSION,
        "status": "PASS",
        "trajectory": {
            "start_update": frames[0].update,
            "end_update": frames[-1].update,
            "tick_count": len(frames),
        },
        "target": {
            "curve_index": curve_index,
            "ball_id": ball_id,
            "color_id": color_id,
            "powerup_type": powerup_type,
        },
        "countdown": {
            "start_update": frames[countdown_start].update,
            "initial_lifetime_ticks": initial_lifetime,
            "last_positive_update": frames[
                before_expiration_index
            ].update,
        },
        "expiration": {
            "update": expiration_frame.update,
            "native_game_time": expiration_native_time,
            "transition_ticks": transition_ticks,
            "sampled_previous_ticks": previous_retention_ticks - 1,
            "cooldown_before": before_state.cooldown_times[powerup_type],
            "cooldown_after": cooldown_time,
            "active_color_count_before": (
                before_state.active_color_counts[color_id]
            ),
            "active_color_count_after": active_color_count,
        },
        "primary_clear_update": primary_clear_update,
        "previous_marker_clear_update": previous_clear_update,
        "visual": {
            "step": visual_step,
            "expiration_scale": (
                expiration_ball.powerup_visual_scale
            ),
            "terminal_scale": balls[
                expiration_index + transition_ticks
            ].powerup_visual_scale,
        },
        "manager": {
            "last_any_spawn_time": (
                expiration_state.last_any_spawn_time
            ),
            "last_type_spawn_time": (
                expiration_state.last_spawn_times[powerup_type]
            ),
            "spawn_count": (
                expiration_state.spawn_counts[powerup_type]
            ),
            "cooldown_time": cooldown_time,
        },
    }


def verify_powerup_lifecycle(
    trajectory_path: Path,
    *,
    ball_id: int,
    curve_index: int,
    color_id: int,
    powerup_type: int,
    expected_expiration_update: int | None = None,
    transition_ticks: int = 100,
    previous_retention_ticks: int = 150,
    visual_step: float = 0.04,
) -> dict[str, Any]:
    """Load and verify a lifecycle while retaining artifact provenance."""

    trajectory_path = trajectory_path.resolve()
    report = verify_powerup_lifecycle_frames(
        load_memory_trajectory(trajectory_path),
        ball_id=ball_id,
        curve_index=curve_index,
        color_id=color_id,
        powerup_type=powerup_type,
        expected_expiration_update=expected_expiration_update,
        transition_ticks=transition_ticks,
        previous_retention_ticks=previous_retention_ticks,
        visual_step=visual_step,
    )
    report["trajectory"] = {
        **report["trajectory"],
        "artifact": str(trajectory_path),
        "artifact_sha256": _sha256_path(trajectory_path),
    }
    return report


def compare_powerup_lifecycle_with_simulator(
    frames: Sequence[TrajectoryFrame],
    simulator: Any,
    *,
    ball_id: int,
    curve_index: int,
    color_id: int,
    powerup_type: int,
) -> dict[str, Any]:
    """Replay captured raw fields through ``RevengeSimulator`` exactly."""

    if len(frames) < 2:
        raise ValueError("at least two frames are required")
    captured_balls = tuple(
        _target_ball(
            frame,
            ball_id=ball_id,
            curve_index=curve_index,
            color_id=color_id,
        )
        for frame in frames
    )
    captured_states = tuple(
        _curve_state(frame, curve_index)
        for frame in frames
    )
    first = captured_balls[0]
    first_state = captured_states[0]
    simulator.load_state(
        colors=[color_id],
        waypoints=[float(first.curve_distance)],
        pending_colors=[],
    )
    simulator.stop_adding = True
    simulator.pending_colors.clear()
    simulated_ball = simulator.balls[0]
    for name in (
        "powerup_previous_type",
        "powerup_primary_type",
        "powerup_secondary_type",
        "powerup_previous_ticks",
        "powerup_lifetime_ticks",
        "powerup_transition_ticks",
        "powerup_visual_scale",
        "powerup_visual_step",
        "powerup_visual_index",
    ):
        setattr(simulated_ball, name, getattr(first, name))
    simulator.native_game_time = frames[0].native_game_time
    simulator.powerup_last_any_spawn_time = (
        first_state.last_any_spawn_time
    )
    simulator.powerup_last_spawn_times = list(
        first_state.last_spawn_times
    )
    simulator.powerup_cooldown_times = list(
        first_state.cooldown_times
    )
    simulator.powerup_spawn_counts = list(first_state.spawn_counts)
    simulator.powerup_field_124_by_type = list(
        first_state.field_124_by_type
    )
    simulator.active_powerup_color_counts = list(
        first_state.active_color_counts
    )

    def float32_bits(value: float | None) -> bytes | None:
        return (
            None
            if value is None
            else struct.pack("<f", float(value))
        )

    def ball_signature(entity: Any) -> tuple[Any, ...]:
        return (
            entity.powerup_previous_type,
            entity.powerup_primary_type,
            entity.powerup_secondary_type,
            entity.powerup_previous_ticks,
            entity.powerup_lifetime_ticks,
            entity.powerup_transition_ticks,
            float32_bits(entity.powerup_visual_scale),
            float32_bits(entity.powerup_visual_step),
            entity.powerup_visual_index,
        )

    if ball_signature(simulated_ball) != ball_signature(first):
        _fail("powerup_lifecycle_simulator_initial_ball_mismatch")
    for expected_frame, expected_ball, expected_state in zip(
        frames[1:],
        captured_balls[1:],
        captured_states[1:],
        strict=True,
    ):
        simulator.native_game_time += 1
        simulator._update_ball_objects()
        if (
            simulator.native_game_time
            != expected_frame.native_game_time
            or ball_signature(simulated_ball)
            != ball_signature(expected_ball)
            or simulator.powerup_last_any_spawn_time
            != expected_state.last_any_spawn_time
            or tuple(simulator.powerup_last_spawn_times)
            != expected_state.last_spawn_times
            or tuple(simulator.powerup_cooldown_times)
            != expected_state.cooldown_times
            or tuple(simulator.powerup_spawn_counts)
            != expected_state.spawn_counts
            or tuple(simulator.powerup_field_124_by_type)
            != expected_state.field_124_by_type
            or tuple(simulator.active_powerup_color_counts)
            != expected_state.active_color_counts
        ):
            _fail(
                "powerup_lifecycle_simulator_transition_mismatch:"
                f"{expected_frame.update}"
            )
    return {
        "schema": "zuma-rl.pc-powerup-lifecycle-core-diff",
        "version": 1,
        "status": "PASS",
        "start_update": frames[0].update,
        "end_update": frames[-1].update,
        "transition_count": len(frames) - 1,
        "target": {
            "curve_index": curve_index,
            "ball_id": ball_id,
            "color_id": color_id,
            "powerup_type": powerup_type,
        },
        "exact_fields": [
            "previous_type",
            "primary_type",
            "secondary_type",
            "previous_ticks",
            "lifetime_ticks",
            "transition_ticks",
            "visual_scale_float32_bits",
            "visual_step_float32_bits",
            "visual_index",
            "native_game_time",
            "curve_powerup_manager_arrays",
        ],
    }


def compare_powerup_spawn_with_simulator(
    before: TrajectoryFrame,
    after: TrajectoryFrame,
    simulator: Any,
    *,
    curve_index: int,
    expected_ball_id: int,
    expected_powerup_type: int,
) -> dict[str, Any]:
    """Replay one full-object PC spawn through the calibrated simulator."""

    zone = f"curve:{curve_index}:list:05c"
    before_balls = tuple(
        sorted(
            (
                entity
                for entity in before.entities
                if entity.zone == zone
            ),
            key=lambda entity: entity.index,
        )
    )
    after_balls = tuple(
        sorted(
            (
                entity
                for entity in after.entities
                if entity.zone == zone
            ),
            key=lambda entity: entity.index,
        )
    )
    if (
        after.update != before.update + 1
        or before.native_game_time is None
        or after.native_game_time != before.native_game_time + 1
        or not before_balls
        or tuple(
            (ball.ball_id, ball.color_id)
            for ball in before_balls
        )
        != tuple(
            (ball.ball_id, ball.color_id)
            for ball in after_balls
        )
        or before.global_mtrand is None
        or after.global_mtrand is None
    ):
        _fail("powerup_spawn_core_diff_frame_contract_invalid")
    before_state = _curve_state(before, curve_index)
    after_state = _curve_state(after, curve_index)
    simulator.load_state(
        colors=[ball.color_id for ball in before_balls],
        waypoints=[
            float(ball.curve_distance) for ball in before_balls
        ],
        pending_colors=[],
    )
    simulator.stop_adding = True
    simulator.pending_colors.clear()
    for simulated, captured in zip(
        simulator.balls,
        before_balls,
        strict=True,
    ):
        simulated.id = captured.ball_id
        for name in (
            "powerup_previous_type",
            "powerup_primary_type",
            "powerup_secondary_type",
            "powerup_previous_ticks",
            "powerup_lifetime_ticks",
            "powerup_transition_ticks",
            "powerup_visual_scale",
            "powerup_visual_step",
            "powerup_visual_index",
        ):
            value = getattr(captured, name)
            if value is None:
                _fail("powerup_spawn_core_diff_raw_field_missing")
            setattr(simulated, name, value)
    simulator._next_id = max(ball.ball_id for ball in before_balls) + 1
    simulator.native_game_time = after.native_game_time
    simulator.powerup_last_any_spawn_time = (
        before_state.last_any_spawn_time
    )
    simulator.powerup_last_spawn_times = list(
        before_state.last_spawn_times
    )
    simulator.powerup_cooldown_times = list(
        before_state.cooldown_times
    )
    simulator.powerup_spawn_counts = list(before_state.spawn_counts)
    simulator.powerup_field_124_by_type = list(
        before_state.field_124_by_type
    )
    simulator.active_powerup_color_counts = list(
        before_state.active_color_counts
    )
    simulator.rng.load_state(
        before.global_mtrand.words,
        before.global_mtrand.index,
    )
    selected = simulator._maybe_spawn_powerup()
    if (
        selected is None
        or selected.id != expected_ball_id
        or selected.powerup_secondary_type != expected_powerup_type
        or simulator.rng.state
        != (
            after.global_mtrand.words,
            after.global_mtrand.index,
        )
    ):
        _fail("powerup_spawn_core_diff_selection_mismatch")

    def float32_bits(value: float | None) -> bytes | None:
        return (
            None
            if value is None
            else struct.pack("<f", float(value))
        )

    def signature(entity: Any) -> tuple[Any, ...]:
        return (
            entity.id if hasattr(entity, "id") else entity.ball_id,
            entity.color if hasattr(entity, "color") else entity.color_id,
            entity.powerup_previous_type,
            entity.powerup_primary_type,
            entity.powerup_secondary_type,
            entity.powerup_previous_ticks,
            entity.powerup_lifetime_ticks,
            entity.powerup_transition_ticks,
            float32_bits(entity.powerup_visual_scale),
            float32_bits(entity.powerup_visual_step),
            entity.powerup_visual_index,
        )

    if (
        tuple(signature(ball) for ball in simulator.balls)
        != tuple(signature(ball) for ball in after_balls)
        or simulator.powerup_last_any_spawn_time
        != after_state.last_any_spawn_time
        or tuple(simulator.powerup_last_spawn_times)
        != after_state.last_spawn_times
        or tuple(simulator.powerup_cooldown_times)
        != after_state.cooldown_times
        or tuple(simulator.powerup_spawn_counts)
        != after_state.spawn_counts
        or tuple(simulator.powerup_field_124_by_type)
        != after_state.field_124_by_type
        or tuple(simulator.active_powerup_color_counts)
        != after_state.active_color_counts
        or simulator.last_events.powerups_spawned != 1
    ):
        _fail("powerup_spawn_core_diff_state_mismatch")
    return {
        "schema": "zuma-rl.pc-powerup-spawn-core-diff",
        "version": 1,
        "status": "PASS",
        "from_update": before.update,
        "to_update": after.update,
        "chain_ball_count": len(before_balls),
        "selected_ball_id": selected.id,
        "selected_color_id": selected.color,
        "selected_powerup_type": selected.powerup_secondary_type,
        "mtrand_index_before": before.global_mtrand.index,
        "mtrand_index_after": after.global_mtrand.index,
        "exact_ball_and_manager_state": True,
    }


__all__ = [
    "POWERUP_LIFECYCLE_SCHEMA",
    "POWERUP_LIFECYCLE_VERSION",
    "PcPowerupLifecycleError",
    "compare_powerup_lifecycle_with_simulator",
    "compare_powerup_spawn_with_simulator",
    "verify_powerup_lifecycle",
    "verify_powerup_lifecycle_frames",
]
