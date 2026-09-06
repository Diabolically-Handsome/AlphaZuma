"""Compare one source-observed retail tunnel-collision transition exactly."""

from __future__ import annotations

from dataclasses import asdict, replace
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .pc_gameplay_diff import (
    _active,
    _curve_state_mismatches,
    _fruit_state_mismatches,
    _pc_projectiles,
    _projectile_errors,
    _projectile_latent_mismatches,
    _transplant_midstate,
)
from .pc_mechanism_audit import derive_native_mechanism_features
from .pc_memory_trajectory import TrajectoryEntity, TrajectoryFrame
from .revenge_core import Projectile, RevengeSimulator


TUNNEL_COLLISION_SIMULATOR_DIFF_SCHEMA = (
    "zuma-rl.pc-tunnel-collision-simulator-diff"
)
TUNNEL_COLLISION_SIMULATOR_DIFF_VERSION = 1


class PcTunnelCollisionError(ValueError):
    """The selected native transition cannot certify tunnel collision."""


def _fail(code: str) -> None:
    raise PcTunnelCollisionError(code)


def _restore_free_projectile(
    simulator: RevengeSimulator,
    entity: TrajectoryEntity,
) -> Projectile:
    values = (
        entity.position_x,
        entity.position_y,
        entity.velocity_x,
        entity.velocity_y,
        entity.radius,
        entity.merge_speed,
    )
    if (
        entity.object_kind != "bullet"
        or entity.zone != "fired"
        or entity.fired is not False
        or any(value is None or not math.isfinite(value) for value in values)
        or entity.radius <= 0
        or int(entity.radius) != entity.radius
        or entity.curve_points is None
        or entity.gap_entry_count != len(entity.gap_entries)
    ):
        _fail("tunnel_collision_initial_projectile_invalid")
    curve_points = {
        index: value
        for index, value in enumerate(
            entity.curve_points[: simulator.curve_count]
        )
        if value >= 0
    }
    projectile = Projectile(
        id=entity.ball_id,
        color=entity.color_id,
        position=np.asarray(
            (entity.position_x, entity.position_y),
            dtype=np.float32,
        ),
        velocity=np.asarray(
            (entity.velocity_x, entity.velocity_y),
            dtype=np.float32,
        ),
        radius=int(entity.radius),
        just_fired=False,
        hit_percent=np.float32(entity.merge_progress or 0.0),
        merge_speed=np.float32(entity.merge_speed),
        waypoint=np.float32(entity.curve_distance),
        curve_point=curve_points.get(0, 0),
        curve_points=curve_points,
        gap_info=[
            (boundary_ball_id, gap_distance)
            for _, gap_distance, boundary_ball_id in entity.gap_entries
        ],
    )
    simulator.free_projectiles = [projectile]
    return projectile


def _active_latent_mismatches(
    simulator: RevengeSimulator,
    source: Sequence[TrajectoryEntity],
) -> dict[str, int]:
    if len(simulator.balls) != len(source):
        return {"active_count": 1}
    mismatches: dict[str, int] = {}
    fields = (
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
    for ball, entity in zip(simulator.balls, source, strict=True):
        for field_name in fields:
            expected = getattr(entity, field_name)
            if expected is not None and getattr(ball, field_name) != expected:
                mismatches[field_name] = mismatches.get(field_name, 0) + 1
    return mismatches


def _rng_state_matches(
    simulator: RevengeSimulator,
    frame: TrajectoryFrame,
) -> Mapping[str, bool]:
    if (
        frame.qrand is None
        or frame.thread_crt_rand_state is None
        or frame.global_mtrand is None
    ):
        _fail("tunnel_collision_rng_state_missing")
    chooser = simulator._color_chooser
    qrand_match = (
        chooser.update_count == frame.qrand.update_count
        and chooser.selected_index == frame.qrand.selected_index
        and tuple(float(value) for value in chooser.weights)
        == frame.qrand.weights
        and tuple(float(value) for value in chooser.sways)
        == frame.qrand.sways
        and tuple(int(value) for value in chooser.last_hit)
        == frame.qrand.last_hit
        and tuple(int(value) for value in chooser.previous_hit)
        == frame.qrand.previous_hit
    )
    return {
        "qrand": qrand_match,
        "thread_crt_rand": (
            simulator.crt_rng.state == frame.thread_crt_rand_state
        ),
        "global_mtrand": (
            simulator.rng.index == frame.global_mtrand.index
            and simulator.rng.words == frame.global_mtrand.words
        ),
    }


def compare_tunnel_collision_transition(
    frames: Sequence[TrajectoryFrame],
    *,
    original_root: str | Path,
    level_id: str = "Jungle2",
    hard: bool = False,
    curve_index: int = 0,
    position_tolerance: float = 1e-5,
    waypoint_tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Restore the first native frame and compare the next retail update."""

    if len(frames) != 2:
        _fail("tunnel_collision_requires_two_frames")
    before, after = frames
    if after.update != before.update + 1:
        _fail("tunnel_collision_updates_not_consecutive")
    if position_tolerance < 0.0 or waypoint_tolerance < 0.0:
        raise ValueError("tolerances must be non-negative")

    features, proofs = derive_native_mechanism_features(
        frames,
        original_root=Path(original_root).resolve(strict=True),
        level_id=level_id,
        hard=hard,
        curve_index=curve_index,
    )
    if features != ("tunnel_collision",) or len(proofs) != 1:
        _fail("tunnel_collision_source_proof_missing_or_ambiguous")
    proof = proofs[0]
    if (
        proof.get("from_update") != before.update
        or proof.get("framework_update") != after.update
    ):
        _fail("tunnel_collision_source_proof_window_mismatch")

    fired_before = _pc_projectiles(before, staging=False)
    fired_after = _pc_projectiles(after, staging=False)
    staging_before = _pc_projectiles(before, staging=True)
    staging_after = _pc_projectiles(after, staging=True)
    if (
        len(fired_before) != 1
        or len(fired_after) != 1
        or fired_before[0].ball_id != fired_after[0].ball_id
        or staging_before
        or staging_after
    ):
        _fail("tunnel_collision_projectile_transition_invalid")

    base_before = replace(
        before,
        entities=tuple(
            entity for entity in before.entities if entity.zone != "fired"
        ),
    )
    simulator, provenance = _transplant_midstate(
        (base_before, after),
        original_root=Path(original_root).resolve(strict=True),
        hard=hard,
        level_id=level_id,
        curve_index=curve_index,
    )
    _restore_free_projectile(simulator, fired_before[0])
    simulator._next_id = max(entity.ball_id for entity in before.entities) + 1
    events = simulator.tick()

    active_after = _active(after)
    active_count_match = len(simulator.balls) == len(active_after)
    active_identity_match = active_count_match and [
        ball.id for ball in simulator.balls
    ] == [entity.ball_id for entity in active_after]
    active_color_match = active_count_match and [
        ball.color for ball in simulator.balls
    ] == [entity.color_id for entity in active_after]
    active_waypoint_error = max(
        (
            abs(float(ball.waypoint) - entity.curve_distance)
            for ball, entity in zip(
                simulator.balls,
                active_after,
                strict=active_count_match,
            )
        ),
        default=math.inf if not active_count_match else 0.0,
    )
    active_position_error = max(
        (
            math.hypot(
                float(simulator.ball_position(ball)[0]) - entity.position_x,
                float(simulator.ball_position(ball)[1]) - entity.position_y,
            )
            for ball, entity in zip(
                simulator.balls,
                active_after,
                strict=active_count_match,
            )
        ),
        default=math.inf if not active_count_match else 0.0,
    )
    identity_map = {
        entity.ball_id: entity.ball_id for entity in before.entities
    }
    (
        projectile_identity_match,
        projectile_position_error,
        projectile_waypoint_error,
        projectile_progress_error,
    ) = _projectile_errors(
        simulator.free_projectiles,
        fired_after,
        identity_map=identity_map,
    )
    projectile_latent_mismatches = _projectile_latent_mismatches(
        simulator.free_projectiles,
        fired_after,
        float_tolerance=position_tolerance,
    )
    active_latent_mismatches = _active_latent_mismatches(
        simulator,
        active_after,
    )
    curve_state_mismatches = _curve_state_mismatches(
        simulator,
        after,
        curve_index=curve_index,
    )
    fruit_state_mismatches = _fruit_state_mismatches(
        simulator,
        after.fruit_state,
    )
    pending_colors_source = tuple(
        entity.color_id
        for entity in after.entities
        if entity.zone == f"curve:{curve_index}:list:068"
    )
    pending_colors_match = (
        tuple(simulator.pending_colors) == pending_colors_source
    )
    shooter_match = (
        simulator.current_color,
        simulator.next_color,
    ) == (after.current_color_id, after.next_color_id)
    rng_matches = _rng_state_matches(simulator, after)
    event_row = asdict(events)
    collision_suppressed = (
        len(simulator.free_projectiles) == 1
        and not simulator.merging_projectiles
        and event_row["hits"] == 0
        and event_row["inserted"] == 0
    )
    exact_state_match = (
        active_identity_match
        and active_color_match
        and active_waypoint_error <= waypoint_tolerance
        and active_position_error <= position_tolerance
        and not active_latent_mismatches
        and projectile_identity_match
        and projectile_position_error <= position_tolerance
        and projectile_waypoint_error <= waypoint_tolerance
        and projectile_progress_error <= waypoint_tolerance
        and not projectile_latent_mismatches
        and pending_colors_match
        and not curve_state_mismatches
        and not fruit_state_mismatches
        and shooter_match
        and all(rng_matches.values())
        and simulator.score == after.score
        and events.score_delta == after.score - before.score
    )
    passed = collision_suppressed and exact_state_match
    failure_reasons: list[str] = []
    if not collision_suppressed:
        failure_reasons.append("tunnel_collision_not_suppressed")
    if not exact_state_match:
        failure_reasons.append("tunnel_collision_state_mismatch")

    return {
        "schema": TUNNEL_COLLISION_SIMULATOR_DIFF_SCHEMA,
        "version": TUNNEL_COLLISION_SIMULATOR_DIFF_VERSION,
        "status": "PASS" if passed else "FAIL",
        "failure_reasons": failure_reasons,
        "level_id": level_id,
        "hard": hard,
        "profile_mode": "tutorials_completed",
        "curve_index": curve_index,
        "start_update": before.update,
        "end_update": after.update,
        "compared_tick_count": 1,
        "source_authorized_features": list(features),
        "source_feature_proofs": list(proofs),
        "source_projectile": {
            "ball_id": fired_before[0].ball_id,
            "color_id": fired_before[0].color_id,
            "position_before": [
                fired_before[0].position_x,
                fired_before[0].position_y,
            ],
            "position_after": [
                fired_after[0].position_x,
                fired_after[0].position_y,
            ],
            "velocity": [
                fired_before[0].velocity_x,
                fired_before[0].velocity_y,
            ],
            "remained_free": True,
            "staging_count_before": len(staging_before),
            "staging_count_after": len(staging_after),
        },
        "simulator_transition": {
            "projectile_count_after": len(simulator.free_projectiles),
            "staging_count_after": len(simulator.merging_projectiles),
            "projectile_identity_match": projectile_identity_match,
            "projectile_position_error_px": projectile_position_error,
            "projectile_waypoint_error": projectile_waypoint_error,
            "projectile_progress_error": projectile_progress_error,
            "projectile_latent_mismatches": projectile_latent_mismatches,
            "active_identity_match": active_identity_match,
            "active_color_match": active_color_match,
            "active_waypoint_error": active_waypoint_error,
            "active_position_error_px": active_position_error,
            "active_latent_mismatches": active_latent_mismatches,
            "pending_colors_match": pending_colors_match,
            "curve_state_mismatches": curve_state_mismatches,
            "fruit_state_mismatches": fruit_state_mismatches,
            "shooter_state_match": shooter_match,
            "rng_state_matches": dict(rng_matches),
            "score_source": after.score,
            "score_simulator": simulator.score,
            "events": event_row,
            "collision_suppressed": collision_suppressed,
            "exact_native_state_match": exact_state_match,
        },
        "scenario": dict(provenance),
    }


__all__ = [
    "PcTunnelCollisionError",
    "TUNNEL_COLLISION_SIMULATOR_DIFF_SCHEMA",
    "TUNNEL_COLLISION_SIMULATOR_DIFF_VERSION",
    "compare_tunnel_collision_transition",
]
