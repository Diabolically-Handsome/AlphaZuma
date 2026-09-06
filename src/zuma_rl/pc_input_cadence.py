"""Strict retail continuous-click and firing-cadence verification."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from .pc_memory_trajectory import (
    TrajectoryEntity,
    TrajectoryFrame,
    load_memory_trajectory,
)
from .popcap_dmo import PopCapDemo


CLICK_CADENCE_SCHEMA = "zuma-rl.pc-click-cadence-verification"
CLICK_CADENCE_VERSION = 1


class PcClickCadenceError(ValueError):
    """A trajectory does not exhibit the required retail click cadence."""


def _fail(code: str) -> None:
    raise PcClickCadenceError(code)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _updates(
    values: Sequence[int],
    *,
    name: str,
    minimum: int,
    maximum: int,
) -> tuple[int, ...]:
    result = tuple(values)
    if (
        not result
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not minimum <= value <= maximum
            for value in result
        )
        or tuple(sorted(set(result))) != result
    ):
        raise ValueError(
            f"{name} must be sorted, unique updates inside the trajectory"
        )
    return result


def _entity_in_zone(
    frame: TrajectoryFrame,
    *,
    zone: str,
    ball_id: int,
    code: str,
) -> TrajectoryEntity:
    matches = [
        entity
        for entity in frame.entities
        if entity.zone == zone and entity.ball_id == ball_id
    ]
    if len(matches) != 1:
        _fail(f"{code}:{frame.update}")
    return matches[0]


def _current_entity(frame: TrajectoryFrame) -> TrajectoryEntity:
    if frame.current_ball_id is None:
        _fail(f"click_cadence_current_ball_missing:{frame.update}")
    entity = _entity_in_zone(
        frame,
        zone="shooter_current",
        ball_id=frame.current_ball_id,
        code="click_cadence_current_entity_missing",
    )
    if entity.object_kind != "bullet" or entity.fired is None:
        _fail(f"click_cadence_current_fired_state_missing:{frame.update}")
    return entity


def _free_entities(
    frame: TrajectoryFrame,
) -> Mapping[int, TrajectoryEntity]:
    result: dict[int, TrajectoryEntity] = {}
    for entity in frame.entities:
        if entity.zone != "fired":
            continue
        if (
            entity.object_kind != "bullet"
            or entity.fired is None
            or entity.ball_id in result
        ):
            _fail(f"click_cadence_free_entity_invalid:{frame.update}")
        result[entity.ball_id] = entity
    return result


def verify_click_cadence_frames(
    frames: Sequence[TrajectoryFrame],
    *,
    down_updates: Sequence[int],
    expected_accepted_down_updates: Sequence[int],
    release_delay_ticks: int = 6,
    expected_cycle_ticks: int = 21,
    expected_free_projectile_ticks: int = 10,
    require_stable_score: bool = True,
) -> dict[str, Any]:
    """Verify accepted and ignored left-clicks against full retail frames."""

    if len(frames) < 2:
        _fail("click_cadence_trajectory_too_short")
    if (
        isinstance(release_delay_ticks, bool)
        or not isinstance(release_delay_ticks, int)
        or release_delay_ticks < 1
        or isinstance(expected_cycle_ticks, bool)
        or not isinstance(expected_cycle_ticks, int)
        or expected_cycle_ticks < 1
        or isinstance(expected_free_projectile_ticks, bool)
        or not isinstance(expected_free_projectile_ticks, int)
        or expected_free_projectile_ticks < 1
    ):
        raise ValueError("click-cadence durations must be positive integers")

    for before, after in zip(frames, frames[1:]):
        if after.update != before.update + 1:
            _fail("click_cadence_frames_not_contiguous")
        if (
            before.native_game_time is not None
            and after.native_game_time is not None
            and after.native_game_time != before.native_game_time + 1
        ):
            _fail("click_cadence_native_time_not_contiguous")

    start_update = frames[0].update
    end_update = frames[-1].update
    downs = _updates(
        down_updates,
        name="down_updates",
        minimum=start_update,
        maximum=end_update - 1,
    )
    accepted = _updates(
        expected_accepted_down_updates,
        name="expected_accepted_down_updates",
        minimum=start_update,
        maximum=end_update - 1,
    )
    if not set(accepted).issubset(downs):
        raise ValueError("accepted updates must be left-button down updates")
    expected_releases = tuple(
        update + release_delay_ticks for update in accepted
    )
    if expected_releases[-1] + expected_free_projectile_ticks > end_update:
        _fail("click_cadence_post_release_context_not_covered")

    by_update = {frame.update: frame for frame in frames}
    current_entities = {
        frame.update: _current_entity(frame) for frame in frames
    }
    free_by_update = {
        frame.update: _free_entities(frame) for frame in frames
    }
    if free_by_update[start_update]:
        _fail("click_cadence_preexisting_projectile")

    observed_accepts: list[int] = []
    for before, after in zip(frames, frames[1:]):
        before_entity = current_entities[before.update]
        after_entity = current_entities[after.update]
        if (
            before.current_ball_id == after.current_ball_id
            and before_entity.fired is False
            and after_entity.fired is True
        ):
            observed_accepts.append(before.update)
    if tuple(observed_accepts) != accepted:
        _fail("click_cadence_accepted_transition_mismatch")

    births_by_update: dict[int, tuple[int, ...]] = {}
    for before, after in zip(frames, frames[1:]):
        births = tuple(
            sorted(
                set(free_by_update[after.update])
                - set(free_by_update[before.update])
            )
        )
        if births:
            births_by_update[after.update] = births
    if tuple(births_by_update) != expected_releases:
        _fail("click_cadence_release_updates_mismatch")
    if any(len(ids) != 1 for ids in births_by_update.values()):
        _fail("click_cadence_release_not_singleton")

    intervals = tuple(
        right - left for left, right in zip(accepted, accepted[1:])
    )
    if any(interval != expected_cycle_ticks for interval in intervals):
        _fail("click_cadence_cycle_interval_mismatch")

    shots: list[dict[str, Any]] = []
    for accepted_update, release_update in zip(
        accepted,
        expected_releases,
        strict=True,
    ):
        accepted_frame = by_update[accepted_update]
        accepted_entity = current_entities[accepted_update]
        next_entity = current_entities[accepted_update + 1]
        if (
            accepted_entity.fired is not False
            or next_entity.fired is not True
            or next_entity.ball_id != accepted_entity.ball_id
        ):
            _fail(
                "click_cadence_fire_state_transition_mismatch:"
                f"{accepted_update}"
            )
        for update in range(accepted_update + 1, release_update):
            entity = current_entities[update]
            if (
                entity.ball_id != accepted_entity.ball_id
                or entity.fired is not True
            ):
                _fail(f"click_cadence_firing_window_mismatch:{update}")

        release_frame = by_update[release_update]
        released_id = births_by_update[release_update][0]
        released_entity = _entity_in_zone(
            release_frame,
            zone="fired",
            ball_id=released_id,
            code="click_cadence_released_entity_missing",
        )
        if (
            released_id != accepted_entity.ball_id
            or released_entity.fired is not False
            or accepted_frame.next_ball_id is None
            or release_frame.current_ball_id
            != accepted_frame.next_ball_id
            or current_entities[release_update].fired is not False
        ):
            _fail(
                "click_cadence_release_identity_mismatch:"
                f"{release_update}"
            )

        free_updates = tuple(
            frame.update
            for frame in frames
            if released_id in free_by_update[frame.update]
        )
        expected_free_updates = tuple(
            range(
                release_update,
                release_update + expected_free_projectile_ticks,
            )
        )
        if free_updates != expected_free_updates:
            _fail(
                "click_cadence_projectile_lifetime_mismatch:"
                f"{released_id}"
            )
        shots.append(
            {
                "accepted_down_update": accepted_update,
                "release_update": release_update,
                "projectile_ball_id": released_id,
                "projectile_color_id": released_entity.color_id,
                "current_ball_id_before_release": (
                    accepted_entity.ball_id
                ),
                "current_ball_id_after_release": (
                    release_frame.current_ball_id
                ),
                "next_ball_id_after_release": release_frame.next_ball_id,
                "free_projectile_first_update": free_updates[0],
                "free_projectile_last_update": free_updates[-1],
                "free_projectile_tick_count": len(free_updates),
            }
        )

    rejected = tuple(update for update in downs if update not in accepted)
    if require_stable_score:
        scores = {frame.score for frame in frames}
        if len(scores) != 1:
            _fail("click_cadence_score_not_stable")

    return {
        "schema": CLICK_CADENCE_SCHEMA,
        "version": CLICK_CADENCE_VERSION,
        "status": "PASS",
        "trajectory": {
            "start_update": start_update,
            "end_update": end_update,
            "tick_count": len(frames),
        },
        "input": {
            "left_down_updates": list(downs),
            "accepted_down_updates": list(accepted),
            "rejected_down_updates": list(rejected),
            "down_count": len(downs),
            "accepted_count": len(accepted),
            "rejected_count": len(rejected),
        },
        "cadence": {
            "release_delay_ticks": release_delay_ticks,
            "expected_cycle_ticks": expected_cycle_ticks,
            "accepted_intervals": list(intervals),
            "release_updates": list(expected_releases),
            "free_projectile_ticks": expected_free_projectile_ticks,
        },
        "shots": shots,
        "score": {
            "stable_required": require_stable_score,
            "first": frames[0].score,
            "last": frames[-1].score,
        },
    }


def _left_button_edges(
    demo: PopCapDemo,
    *,
    start_update: int,
    end_update: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    state = False
    state_before_start = False
    captured_start_state = False
    downs: list[int] = []
    ups: list[int] = []
    for command in demo.commands:
        payload = command.payload
        if (
            command.kind != "mouse_button"
            or payload.get("button") != 1
        ):
            continue
        down = payload.get("down")
        if not isinstance(down, bool):
            _fail("click_cadence_dmo_button_state_invalid")
        if command.update >= start_update and not captured_start_state:
            state_before_start = state
            captured_start_state = True
        if command.update > end_update:
            break
        if down == state:
            _fail(
                "click_cadence_dmo_duplicate_button_edge:"
                f"{command.update}"
            )
        state = down
        if start_update <= command.update <= end_update:
            (downs if down else ups).append(command.update)

    if state_before_start or state:
        _fail("click_cadence_dmo_button_not_released_at_boundary")
    return tuple(downs), tuple(ups)


def verify_click_cadence(
    trajectory_path: Path,
    dmo_path: Path,
    *,
    expected_accepted_down_updates: Sequence[int],
    release_delay_ticks: int = 6,
    expected_cycle_ticks: int = 21,
    expected_free_projectile_ticks: int = 10,
    require_stable_score: bool = True,
) -> dict[str, Any]:
    """Load retail artifacts and verify continuous-click behavior."""

    trajectory_path = trajectory_path.resolve()
    dmo_path = dmo_path.resolve()
    frames = load_memory_trajectory(trajectory_path)
    demo = PopCapDemo.read(dmo_path)
    if demo.length_updates < frames[-1].update:
        _fail("click_cadence_dmo_does_not_cover_trajectory")
    downs, ups = _left_button_edges(
        demo,
        start_update=frames[0].update,
        end_update=frames[-1].update,
    )
    report = verify_click_cadence_frames(
        frames,
        down_updates=downs,
        expected_accepted_down_updates=expected_accepted_down_updates,
        release_delay_ticks=release_delay_ticks,
        expected_cycle_ticks=expected_cycle_ticks,
        expected_free_projectile_ticks=expected_free_projectile_ticks,
        require_stable_score=require_stable_score,
    )
    report["trajectory"] = {
        **report["trajectory"],
        "artifact": str(trajectory_path),
        "artifact_sha256": _sha256_path(trajectory_path),
    }
    report["dmo"] = {
        "artifact": str(dmo_path),
        "artifact_bytes": demo.artifact_bytes,
        "artifact_sha256": demo.artifact_sha256,
        "random_seed": demo.random_seed,
        "length_updates": demo.length_updates,
        "left_up_updates": list(ups),
        "classification": "diagnostic-not-unmodified-pc-evidence",
    }
    return report


__all__ = [
    "CLICK_CADENCE_SCHEMA",
    "CLICK_CADENCE_VERSION",
    "PcClickCadenceError",
    "verify_click_cadence",
    "verify_click_cadence_frames",
]
