"""Transplant one retail staged merge into the simulator and diff each tick."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from zuma_rl.original_data import OriginalDataError, OriginalGameCatalog
from zuma_rl.pc_memory_trajectory import (
    ACTIVE_CHAIN_LIST_OFFSET,
    INSERTION_STAGING_LIST_OFFSET,
    PcMemoryTrajectoryError,
    TrajectoryEntity,
    TrajectoryFrame,
)
from zuma_rl.revenge_core import Projectile, RevengeSimulator


MERGE_DIFF_SCHEMA = "zuma-rl.pc-merge-simulator-diff"
MERGE_DIFF_VERSION = 1
CURVE_FIT_TOLERANCE_PX = 1e-3


def _zone_suffix(offset: int) -> str:
    return f"list:{offset:03x}"


def _entities_in_list(
    frame: TrajectoryFrame,
    offset: int,
) -> tuple[TrajectoryEntity, ...]:
    suffix = _zone_suffix(offset)
    return tuple(
        sorted(
            (
                entity
                for entity in frame.entities
                if entity.zone.endswith(suffix)
            ),
            key=lambda entity: entity.index,
        )
    )


def _staging(frame: TrajectoryFrame) -> tuple[TrajectoryEntity, ...]:
    return _entities_in_list(frame, INSERTION_STAGING_LIST_OFFSET)


def _active(frame: TrajectoryFrame) -> tuple[TrajectoryEntity, ...]:
    return _entities_in_list(frame, ACTIVE_CHAIN_LIST_OFFSET)


def _infer_speed(
    first: TrajectoryFrame,
    second: TrajectoryFrame,
) -> np.float32:
    before = {entity.ball_id: entity for entity in _active(first)}
    deltas = [
        np.float32(entity.curve_distance - before[entity.ball_id].curve_distance)
        for entity in _active(second)
        if entity.ball_id in before
    ]
    if not deltas:
        raise PcMemoryTrajectoryError("merge_diff_speed_unavailable")
    values, counts = np.unique(np.asarray(deltas, dtype=np.float32), return_counts=True)
    speed = values[int(np.argmax(counts))]
    if not bool(np.isfinite(speed)) or speed < np.float32(0.0):
        raise PcMemoryTrajectoryError("merge_diff_speed_invalid")
    return np.float32(speed)


def _identify_level_curve(
    active: Sequence[TrajectoryEntity],
    *,
    original_root: Path,
    hard: bool,
    level_id: str | None,
    curve_index: int | None,
) -> tuple[str, int, float, float]:
    """Bind a captured PC chain to one installed curve by exact geometry."""

    catalog = OriginalGameCatalog(original_root)
    candidates: list[tuple[float, float, str, int]] = []
    definitions = (
        (catalog.level(level_id),)
        if level_id is not None
        else tuple(catalog.levels.values())
    )
    maximum_waypoint = max(entity.curve_distance for entity in active)
    for definition in definitions:
        try:
            loaded = catalog.load_level(definition.id, hard=hard)
        except (OSError, ValueError, OriginalDataError):
            continue
        indices = (
            (curve_index,)
            if curve_index is not None
            else tuple(range(len(loaded.curves)))
        )
        for candidate_index in indices:
            if not 0 <= candidate_index < len(loaded.curves):
                continue
            curve = loaded.curves[candidate_index]
            if maximum_waypoint > curve.end_waypoint:
                continue
            squared_errors: list[float] = []
            maximum_error = 0.0
            for entity in active:
                position = curve.point_at_waypoint(entity.curve_distance)
                error = math.hypot(
                    float(position[0]) - entity.position_x,
                    float(position[1]) - entity.position_y,
                )
                squared_errors.append(error * error)
                maximum_error = max(maximum_error, error)
            rms_error = math.sqrt(
                sum(squared_errors) / len(squared_errors)
            )
            candidates.append(
                (
                    maximum_error,
                    rms_error,
                    loaded.definition.id,
                    candidate_index,
                )
            )
    if not candidates:
        raise PcMemoryTrajectoryError("merge_diff_curve_candidate_missing")
    candidates.sort()
    best_maximum, best_rms, best_level, best_curve = candidates[0]
    if best_maximum > CURVE_FIT_TOLERANCE_PX:
        raise PcMemoryTrajectoryError(
            "merge_diff_curve_geometry_not_exact:"
            f"{best_level}:{best_curve}:{best_maximum:.9f}"
        )
    if (
        level_id is None
        and len(candidates) > 1
        and candidates[1][0] <= CURVE_FIT_TOLERANCE_PX
        and (
            candidates[1][2],
            candidates[1][3],
        )
        != (best_level, best_curve)
    ):
        raise PcMemoryTrajectoryError("merge_diff_curve_geometry_ambiguous")
    return best_level, best_curve, best_maximum, best_rms


def _build_simulator(
    frames: Sequence[TrajectoryFrame],
    *,
    original_root: Path,
    hard: bool,
    level_id: str | None,
    curve_index: int | None,
) -> tuple[
    RevengeSimulator,
    int,
    int,
    bool,
    str,
    int,
    float,
    float,
]:
    if len(frames) < 2:
        raise PcMemoryTrajectoryError("merge_diff_window_too_short")
    first = frames[0]
    active = _active(first)
    staging = _staging(first)
    if len(active) < 1 or len(staging) != 1:
        raise PcMemoryTrajectoryError("merge_diff_initial_topology_invalid")
    projectile_row = staging[0]
    if (
        projectile_row.object_kind != "bullet"
        or projectile_row.velocity_x is None
        or projectile_row.velocity_y is None
        or projectile_row.merge_progress is None
        or projectile_row.merge_speed is None
    ):
        raise PcMemoryTrajectoryError("merge_diff_projectile_fields_missing")

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
    simulator = RevengeSimulator.from_installed(
        identified_level,
        root=original_root,
        hard=hard,
        curve_index=identified_curve,
        seed=0,
    )
    contacts = [
        (
            active[index + 1].curve_distance
            - active[index].curve_distance
        )
        <= 36.001
        for index in range(len(active) - 1)
    ]
    pending = _entities_in_list(first, 0x68)
    score_at_level_start = first.score_target - int(
        getattr(simulator.parameters, "zuma_score", 0)
    )
    if not 0 <= score_at_level_start <= first.score:
        raise PcMemoryTrajectoryError(
            "merge_diff_score_target_inconsistent"
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
    for ball, entity in zip(simulator.balls, active, strict=True):
        ball.id = entity.ball_id
    simulator._next_id = max(
        entity.ball_id for entity in first.entities
    ) + 1
    simulator.num_balls_created = len(active) + len(pending)
    simulator.tick_count = first.update
    simulator.advance_speed = _infer_speed(first, frames[1])
    simulator.first_chain_end = int(active[-1].curve_distance)
    simulator.has_reached_rollout = True
    simulator.has_reached_cruising_speed = True

    projectile_position = np.array(
        (projectile_row.position_x, projectile_row.position_y),
        dtype=np.float32,
    )
    distances = [
        float(
            np.linalg.norm(
                simulator.ball_position(ball) - projectile_position
            )
        )
        for ball in simulator.balls
    ]
    hit_index = int(np.argmin(np.asarray(distances)))
    hit_ball = simulator.balls[hit_index]
    if distances[hit_index] >= hit_ball.radius + simulator.config.ball_radius:
        raise PcMemoryTrajectoryError("merge_diff_hit_ball_not_colliding")
    perpendicular = simulator._perpendicular_xy(hit_ball.waypoint)
    offset = np.subtract(
        projectile_position,
        simulator.ball_position(hit_ball),
        dtype=np.float32,
    )
    cross_z = np.float32(
        np.float32(offset[0] * perpendicular[1])
        - np.float32(offset[1] * perpendicular[0])
    )
    hit_in_front = bool(cross_z < np.float32(0.0))

    target_position = np.asarray(
        simulator._point_xy(projectile_row.curve_distance),
        dtype=np.float32,
    )
    progress = np.float32(projectile_row.merge_progress)
    if not np.float32(0.0) <= progress < np.float32(1.0):
        raise PcMemoryTrajectoryError("merge_diff_progress_invalid")
    hit_position = np.asarray(
        np.divide(
            np.subtract(
                projectile_position,
                np.multiply(progress, target_position, dtype=np.float32),
                dtype=np.float32,
            ),
            np.float32(1.0) - progress,
            dtype=np.float32,
        ),
        dtype=np.float32,
    )
    projectile = Projectile(
        id=projectile_row.ball_id,
        color=projectile_row.color_id,
        position=projectile_position,
        velocity=np.array(
            (projectile_row.velocity_x, projectile_row.velocity_y),
            dtype=np.float32,
        ),
        radius=simulator.config.ball_radius,
        just_fired=False,
        hit_ball_id=hit_ball.id,
        hit_in_front=hit_in_front,
        hit_percent=progress,
        merge_speed=np.float32(projectile_row.merge_speed),
        hit_position=hit_position,
        target_position=target_position,
        waypoint=np.float32(projectile_row.curve_distance),
        have_set_prev_ball=False,
        do_new_merge=simulator._has_discontinuity(
            int(hit_ball.waypoint) - 80,
            160,
        ),
    )
    simulator.merging_projectiles = [projectile]
    return (
        simulator,
        hit_index,
        hit_ball.id,
        hit_in_front,
        identified_level,
        identified_curve,
        curve_fit_maximum,
        curve_fit_rms,
    )


def compare_staged_merge_trajectory(
    frames: Sequence[TrajectoryFrame],
    *,
    original_root: Path,
    hard: bool = False,
    level_id: str | None = None,
    curve_index: int | None = None,
    waypoint_tolerance: float = 1e-4,
    position_tolerance: float = 1e-3,
    progress_tolerance: float = 1e-6,
) -> Mapping[str, Any]:
    """Compare a PC staging-to-insertion window against a transplanted sim."""

    if not frames:
        raise PcMemoryTrajectoryError("merge_diff_frames_empty")
    start_index = next(
        (
            index
            for index, frame in enumerate(frames)
            if len(_staging(frame)) == 1
        ),
        None,
    )
    if start_index is None or start_index + 1 >= len(frames):
        raise PcMemoryTrajectoryError("merge_diff_staging_window_missing")
    window = tuple(frames[start_index:])
    (
        simulator,
        hit_index,
        hit_ball_id,
        hit_in_front,
        identified_level,
        identified_curve,
        curve_fit_maximum,
        curve_fit_rms,
    ) = _build_simulator(
        window,
        original_root=original_root.resolve(),
        hard=hard,
        level_id=level_id,
        curve_index=curve_index,
    )

    rows: list[dict[str, Any]] = []
    failure_reasons: list[str] = []
    insertion_update: int | None = None
    identity_map = {
        entity.ball_id: entity.ball_id for entity in _active(window[0])
    }
    allocated_identity_mappings: list[dict[str, Any]] = []
    for frame in window[1:]:
        tick_events = simulator.tick()
        pc_active = _active(frame)
        sim_ids = [ball.id for ball in simulator.balls]
        pc_ids = [entity.ball_id for entity in pc_active]
        unknown_simulator_ids = [
            ball_id for ball_id in sim_ids if ball_id not in identity_map
        ]
        mapped_pc_ids = set(identity_map.values())
        unknown_pc_ids = [
            ball_id for ball_id in pc_ids if ball_id not in mapped_pc_ids
        ]
        if (
            tick_events.inserted == 1
            and len(unknown_simulator_ids) == 1
            and len(unknown_pc_ids) == 1
        ):
            simulator_id = unknown_simulator_ids[0]
            pc_id = unknown_pc_ids[0]
            identity_map[simulator_id] = pc_id
            allocated_identity_mappings.append(
                {
                    "framework_update": frame.update,
                    "simulator_id": simulator_id,
                    "pc_id": pc_id,
                    "source": "midstate_allocator_lineage",
                }
            )
        mapped_sim_ids = [
            identity_map.get(ball_id) for ball_id in sim_ids
        ]
        identity_match = mapped_sim_ids == pc_ids
        color_match = [ball.color for ball in simulator.balls] == [
            entity.color_id for entity in pc_active
        ]
        waypoint_errors = (
            [
                abs(float(ball.waypoint) - entity.curve_distance)
                for ball, entity in zip(
                    simulator.balls,
                    pc_active,
                    strict=True,
                )
            ]
            if len(simulator.balls) == len(pc_active)
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
            if len(simulator.balls) == len(pc_active)
            else []
        )
        maximum_waypoint_error = max(waypoint_errors, default=math.inf)
        maximum_position_error = max(position_errors, default=math.inf)

        pc_staging = _staging(frame)
        staging_identity_match = (
            len(pc_staging) == len(simulator.merging_projectiles)
        )
        staging_position_error: float | None = None
        staging_progress_error: float | None = None
        staging_waypoint_error: float | None = None
        if len(pc_staging) == len(simulator.merging_projectiles) == 1:
            pc_projectile = pc_staging[0]
            sim_projectile = simulator.merging_projectiles[0]
            staging_identity_match = (
                sim_projectile.id == pc_projectile.ball_id
                and sim_projectile.color == pc_projectile.color_id
            )
            staging_position_error = math.hypot(
                float(sim_projectile.position[0])
                - pc_projectile.position_x,
                float(sim_projectile.position[1])
                - pc_projectile.position_y,
            )
            staging_progress_error = abs(
                float(sim_projectile.hit_percent)
                - float(pc_projectile.merge_progress)
            )
            staging_waypoint_error = abs(
                float(sim_projectile.waypoint)
                - pc_projectile.curve_distance
            )
        if tick_events.inserted:
            insertion_update = frame.update

        row_pass = (
            identity_match
            and color_match
            and maximum_waypoint_error <= waypoint_tolerance
            and maximum_position_error <= position_tolerance
            and staging_identity_match
            and (
                staging_position_error is None
                or staging_position_error <= position_tolerance
            )
            and (
                staging_progress_error is None
                or staging_progress_error <= progress_tolerance
            )
            and (
                staging_waypoint_error is None
                or staging_waypoint_error <= waypoint_tolerance
            )
            and simulator.score == frame.score
        )
        if not row_pass:
            failure_reasons.append(f"tick_mismatch:{frame.update}")
        rows.append(
            {
                "framework_update": frame.update,
                "status": "PASS" if row_pass else "FAIL",
                "active_identity_match": identity_match,
                "active_color_match": color_match,
                "active_count_pc": len(pc_active),
                "active_count_simulator": len(simulator.balls),
                "maximum_waypoint_error": maximum_waypoint_error,
                "maximum_position_error_px": maximum_position_error,
                "staging_identity_match": staging_identity_match,
                "staging_position_error_px": staging_position_error,
                "staging_progress_error": staging_progress_error,
                "staging_waypoint_error": staging_waypoint_error,
                "score_pc": frame.score,
                "score_simulator": simulator.score,
                "events": {
                    "hits": tick_events.hits,
                    "inserted": tick_events.inserted,
                    "matches": tick_events.matches,
                    "balls_exploded": tick_events.balls_exploded,
                    "balls_removed": tick_events.balls_removed,
                    "score_delta": tick_events.score_delta,
                },
            }
        )

    pc_insertion_update = next(
        (
            frame.update
            for previous, frame in zip(window, window[1:])
            if len(_staging(previous)) == 1
            and len(_staging(frame)) == 0
            and len(_active(frame)) == len(_active(previous)) + 1
        ),
        None,
    )
    if insertion_update != pc_insertion_update:
        failure_reasons.append("insertion_update_mismatch")
    return {
        "schema": MERGE_DIFF_SCHEMA,
        "version": MERGE_DIFF_VERSION,
        "status": "PASS" if not failure_reasons else "FAIL",
        "start_update": window[0].update,
        "end_update": window[-1].update,
        "compared_tick_count": len(rows),
        "level_id": identified_level,
        "hard": hard,
        "curve_index": identified_curve,
        "curve_fit_maximum_error_px": curve_fit_maximum,
        "curve_fit_rms_error_px": curve_fit_rms,
        "allocated_identity_mappings": allocated_identity_mappings,
        "initial_hit_index": hit_index,
        "initial_hit_ball_id": hit_ball_id,
        "initial_hit_in_front": hit_in_front,
        "inferred_advance_speed": float(
            _infer_speed(window[0], window[1])
        ),
        "pc_insertion_update": pc_insertion_update,
        "simulator_insertion_update": insertion_update,
        "waypoint_tolerance": waypoint_tolerance,
        "position_tolerance_px": position_tolerance,
        "progress_tolerance": progress_tolerance,
        "failure_reasons": failure_reasons,
        "ticks": rows,
    }


__all__ = [
    "MERGE_DIFF_SCHEMA",
    "compare_staged_merge_trajectory",
]
