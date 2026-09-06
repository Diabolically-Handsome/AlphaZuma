"""Replay format and deterministic trajectory-fingerprint tests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pytest

from zuma_rl.original_data import CurveParameters
from zuma_rl.replay import (
    FINGERPRINT_SCOPE,
    REPLAY_SCHEMA,
    REPLAY_VERSION,
    ReplayAction,
    ReplayMismatchError,
    ReplayPlan,
    ReplayResult,
    ReplayValidationError,
    environment_fingerprint,
    run_replay,
    verify_replay,
)
from zuma_rl.revenge_core import RevengePhysicsConfig, RevengeSimulator


@dataclass(slots=True)
class SyntheticCurve:
    """A tiny, one-pixel-sampled straight curve requiring no game install."""

    points: np.ndarray
    in_tunnel: np.ndarray
    parameters: CurveParameters
    die_at_end: bool = True

    @property
    def end_waypoint(self) -> int:
        return len(self.points) - 1

    def point_at_waypoint(
        self,
        waypoint: float | Iterable[float],
        *,
        loop_at_end: bool = False,
    ) -> np.ndarray:
        values = np.asarray(waypoint, dtype=np.float64)
        if loop_at_end:
            values = np.mod(values, len(self.points))
        clipped = np.clip(values, 0.0, float(self.end_waypoint))
        lower = np.floor(clipped).astype(np.int64)
        upper = np.minimum(lower + 1, self.end_waypoint)
        fraction = clipped - lower
        return self.points[lower] + fraction[..., np.newaxis] * (
            self.points[upper] - self.points[lower]
        )

    def perpendicular_at_waypoint(
        self,
        waypoint: float | Iterable[float],
    ) -> np.ndarray:
        values = np.asarray(waypoint)
        return np.broadcast_to(
            np.array((0.0, -1.0), dtype=np.float64),
            values.shape + (2,),
        ).copy()

    def is_in_tunnel_at_waypoint(
        self,
        waypoint: float | Iterable[float],
    ) -> np.ndarray | np.bool_:
        values = np.asarray(waypoint, dtype=np.float64)
        indices = np.trunc(values).astype(np.int64)
        result = np.zeros(values.shape, dtype=np.bool_)
        result = np.where(indices < 0, True, result)
        valid = (indices >= 0) & (indices <= self.end_waypoint)
        return np.where(
            valid,
            self.in_tunnel[np.clip(indices, 0, self.end_waypoint)],
            result,
        )


def _curve() -> SyntheticCurve:
    x = np.arange(1_001, dtype=np.float64)
    points = np.column_stack((x, np.full_like(x, 100.0)))
    return SyntheticCurve(
        points=points,
        in_tunnel=np.zeros(len(points), dtype=np.bool_),
        parameters=CurveParameters(
            start_distance_percent=65,
            num_balls=0,
            ball_repeat_chance=45,
            max_single=2,
            colors=4,
            speed=0.0,
            slow_distance=200,
            acceleration_rate=0.0,
            max_speed=100.0,
            zuma_score=10_000,
            skull_rotation_degrees=75,
            zuma_back_distance=300,
            zuma_slow_duration=1_100,
            slow_factor=4.0,
            max_clump_size=6,
            powerup_records=(),
            powerup_chance=600,
        ),
    )


def _factory(level: str, seed: int) -> RevengeSimulator:
    assert level == "synthetic-line-v1"
    simulator = RevengeSimulator(
        _curve(),
        shooter=(200.0, 300.0),
        seed=seed,
    )
    simulator.load_state(
        colors=[0, 1, 2],
        waypoints=[300.0, 336.0, 372.0],
        contacts=[True, True],
        current_color=0,
        next_color=1,
    )
    simulator.stop_adding = True
    simulator.pending_colors.clear()
    simulator.advance_speed = 0.0
    return simulator


def _plan() -> ReplayPlan:
    return ReplayPlan(
        level="synthetic-line-v1",
        seed=1729,
        total_ticks=12,
        actions=(
            ReplayAction(tick=1, kind="aim", angle=-np.pi / 2.0),
            ReplayAction(tick=2, kind="fire"),
            ReplayAction(tick=10, kind="wait"),
        ),
    )


def test_replay_json_round_trip_with_optional_tick_events(
    tmp_path: Path,
) -> None:
    result = run_replay(
        _plan(),
        simulator_factory=_factory,
        record_tick_events=True,
    )
    payload = json.loads(result.to_json())
    assert payload["schema"] == REPLAY_SCHEMA
    assert payload["version"] == REPLAY_VERSION
    assert payload["fingerprint_scope"] == FINGERPRINT_SCOPE
    assert payload["level"] == "synthetic-line-v1"
    assert payload["profile_mode"] == "tutorials_completed"
    assert len(payload["tick_events"]) == result.ticks_executed
    assert payload["final_fingerprint"].startswith("sha256:")

    decoded = ReplayResult.from_json(result.to_json())
    assert decoded == result

    path = tmp_path / "trace.json"
    result.write_json(path)
    assert ReplayResult.read_json(path) == result

    compact = run_replay(_plan(), simulator_factory=_factory)
    assert "tick_events" not in compact.to_dict()
    assert ReplayResult.from_json(compact.to_json(indent=None)) == compact


def test_same_seed_and_actions_replay_to_same_trajectory() -> None:
    first = run_replay(
        _plan(),
        simulator_factory=_factory,
        record_tick_events=True,
    )
    second = run_replay(
        _plan(),
        simulator_factory=_factory,
        record_tick_events=True,
    )
    assert first.final_fingerprint == second.final_fingerprint
    assert first.tick_records == second.tick_records
    assert verify_replay(first, simulator_factory=_factory) == first


@pytest.mark.parametrize(
    "variant",
    ("shooter", "physics", "curve", "curve_count"),
)
def test_replay_fingerprint_binds_static_environment_definition(
    variant: str,
) -> None:
    plan = ReplayPlan(
        level="synthetic-line-v1",
        seed=5,
        total_ticks=1,
        actions=(ReplayAction(tick=1, kind="wait"),),
    )
    baseline_simulator = _factory(plan.level, plan.seed)
    baseline_environment = environment_fingerprint(baseline_simulator)
    expected = run_replay(plan, simulator_factory=_factory)

    def wrong_factory(level: str, seed: int) -> RevengeSimulator:
        assert level == "synthetic-line-v1"
        curve = _curve()
        curves = None
        shooter = (200.0, 300.0)
        config = None
        if variant == "shooter":
            shooter = (201.0, 300.0)
        elif variant == "physics":
            config = RevengePhysicsConfig(left_miss_margin=81.0)
        elif variant == "curve":
            curve.points[:, 1] += 1.0
        else:
            curves = (curve, _curve())
        simulator = RevengeSimulator(
            curve,
            shooter=shooter,
            curves=curves,
            seed=seed,
            config=config,
        )
        simulator.load_state(
            colors=[0, 1, 2],
            waypoints=[300.0, 336.0, 372.0],
            contacts=[True, True],
            current_color=0,
            next_color=1,
        )
        simulator.stop_adding = True
        simulator.pending_colors.clear()
        simulator.advance_speed = np.float32(0.0)
        return simulator

    wrong_simulator = wrong_factory(plan.level, plan.seed)
    assert environment_fingerprint(wrong_simulator) != baseline_environment
    actual = run_replay(plan, simulator_factory=wrong_factory)
    assert actual.final_fingerprint != expected.final_fingerprint
    with pytest.raises(ReplayMismatchError, match="fingerprint"):
        verify_replay(expected, simulator_factory=wrong_factory)


def test_verify_replay_rejects_tampered_compact_tick_count() -> None:
    plan = ReplayPlan(
        level="synthetic-line-v1",
        seed=5,
        total_ticks=1,
        actions=(ReplayAction(tick=1, kind="wait"),),
    )
    result = run_replay(plan, simulator_factory=_factory)
    tampered = ReplayResult(
        plan=result.plan,
        ticks_executed=0,
        final_fingerprint=result.final_fingerprint,
        tick_records=None,
    )

    with pytest.raises(ReplayMismatchError, match="tick count"):
        verify_replay(tampered, simulator_factory=_factory)


def test_action_change_changes_state_trajectory_fingerprint() -> None:
    wait = ReplayPlan(
        level="synthetic-line-v1",
        seed=5,
        total_ticks=3,
        actions=(ReplayAction(tick=1, kind="wait"),),
    )
    swap = ReplayPlan(
        level="synthetic-line-v1",
        seed=5,
        total_ticks=3,
        actions=(ReplayAction(tick=1, kind="swap"),),
    )
    wait_result = run_replay(wait, simulator_factory=_factory)
    swap_result = run_replay(swap, simulator_factory=_factory)
    assert wait_result.final_fingerprint != swap_result.final_fingerprint


def test_fire_can_apply_cursor_angle_in_the_same_native_tick() -> None:
    inherited_angle = ReplayPlan(
        level="synthetic-line-v1",
        seed=5,
        total_ticks=1,
        actions=(ReplayAction(tick=1, kind="fire"),),
    )
    same_tick_angle = ReplayPlan(
        level="synthetic-line-v1",
        seed=5,
        total_ticks=1,
        actions=(ReplayAction(tick=1, kind="fire", angle=np.pi / 3.0),),
    )

    inherited = run_replay(inherited_angle, simulator_factory=_factory)
    simultaneous = run_replay(same_tick_angle, simulator_factory=_factory)

    assert inherited.final_fingerprint != simultaneous.final_fingerprint
    assert ReplayPlan.from_dict(same_tick_angle.to_dict()) == same_tick_angle

    simulator = _factory("synthetic-line-v1", 5)
    from zuma_rl.replay import _apply_action

    _apply_action(
        simulator,
        ReplayAction(tick=1, kind="aim", angle=np.pi / 3.0),
    )
    assert isinstance(simulator.aim_angle, np.float32)


@pytest.mark.parametrize(
    "actions",
    [
        (
            ReplayAction(tick=2, kind="wait"),
            ReplayAction(tick=1, kind="wait"),
        ),
        (
            ReplayAction(tick=1, kind="wait"),
            ReplayAction(tick=1, kind="swap"),
        ),
    ],
)
def test_replay_rejects_out_of_order_or_duplicate_actions(
    actions: tuple[ReplayAction, ...],
) -> None:
    with pytest.raises(ReplayValidationError, match="strictly ordered"):
        ReplayPlan(
            level="synthetic-line-v1",
            seed=0,
            total_ticks=2,
            actions=actions,
        )


def test_replay_rejects_illegal_actions_and_ranges() -> None:
    with pytest.raises(ReplayValidationError, match="one of"):
        ReplayAction(tick=1, kind="teleport")  # type: ignore[arg-type]
    with pytest.raises(ReplayValidationError, match="finite"):
        ReplayAction(tick=1, kind="aim", angle=float("nan"))
    with pytest.raises(ReplayValidationError, match="only valid"):
        ReplayAction(tick=1, kind="wait", angle=0.0)
    with pytest.raises(ReplayValidationError, match="at least one"):
        ReplayAction(tick=0, kind="wait")
    with pytest.raises(ReplayValidationError, match="exceeds"):
        ReplayPlan(
            level="synthetic-line-v1",
            seed=0,
            total_ticks=1,
            actions=(ReplayAction(tick=2, kind="wait"),),
        )
    with pytest.raises(ReplayValidationError, match="profile_mode"):
        ReplayPlan(
            level="synthetic-line-v1",
            seed=0,
            total_ticks=1,
            profile_mode="fresh_profile",
        )


def test_replay_rejects_unknown_schema_and_malformed_json() -> None:
    result = run_replay(_plan(), simulator_factory=_factory)
    payload = result.to_dict()
    payload["schema"] = "some-other-schema"
    with pytest.raises(ReplayValidationError, match="schema"):
        ReplayResult.from_dict(payload)
    with pytest.raises(ReplayValidationError, match="invalid replay JSON"):
        ReplayResult.from_json("{")
