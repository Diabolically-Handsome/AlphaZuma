"""Deterministic native-tick replays for the fidelity-first simulator.

This module is deliberately small and independent of the Gym wrapper.  Its
JSON format is intended to become a bridge for future differential checks
against captured PC traces; it is not itself evidence that the Python
simulator has been calibrated against such a trace.

Ticks in a replay are one-based.  An action scheduled for tick ``N`` is
applied immediately before the simulator advances from tick ``N - 1`` to
tick ``N``.  Omitted ticks are implicit waits.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np

from zuma_rl.revenge_core import (
    SUPPORTED_PROFILE_MODE,
    RevengeSimulator,
    TickEvents,
)

REPLAY_SCHEMA = "zuma-rl.revenge-replay"
REPLAY_VERSION = 4
FINGERPRINT_ALGORITHM = "sha256"
FINGERPRINT_SCOPE = "environment-and-state-trajectory-v4"

ActionKind = Literal["aim", "fire", "swap", "wait"]
_ACTION_KINDS = frozenset(("aim", "fire", "swap", "wait"))
_FINGERPRINT_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


class ReplayValidationError(ValueError):
    """Raised when a replay object or JSON document is malformed."""


class ReplayMismatchError(AssertionError):
    """Raised when replaying a stored record produces a different trace."""


class SimulatorFactory(Protocol):
    """Create a fresh tick-zero simulator for ``level`` and ``seed``."""

    def __call__(self, level: str, seed: int) -> RevengeSimulator: ...


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ReplayValidationError(f"{name} must be an integer")
    return int(value)


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ReplayValidationError(f"{name} must be a boolean")
    return bool(value)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReplayValidationError(f"{name} must be a JSON object")
    if not all(isinstance(key, str) for key in value):
        raise ReplayValidationError(f"{name} keys must be strings")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ReplayValidationError(f"{name} must be a JSON array")
    return value


def _check_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] = frozenset(),
    name: str,
) -> None:
    missing = required.difference(value)
    if missing:
        joined = ", ".join(sorted(missing))
        raise ReplayValidationError(f"{name} is missing: {joined}")
    unknown = set(value).difference(required, optional)
    if unknown:
        joined = ", ".join(sorted(unknown))
        raise ReplayValidationError(f"{name} has unknown fields: {joined}")


def _validate_fingerprint(value: Any, name: str) -> str:
    if not isinstance(value, str) or _FINGERPRINT_PATTERN.fullmatch(value) is None:
        raise ReplayValidationError(
            f"{name} must have the form 'sha256:' plus 64 lowercase hex digits"
        )
    return value


@dataclass(frozen=True, slots=True)
class ReplayAction:
    """One action applied immediately before a one-based native tick."""

    tick: int
    kind: ActionKind
    angle: float | None = None

    def __post_init__(self) -> None:
        tick = _integer(self.tick, "action tick")
        if tick < 1:
            raise ReplayValidationError("action tick must be at least one")
        object.__setattr__(self, "tick", tick)

        if not isinstance(self.kind, str) or self.kind not in _ACTION_KINDS:
            raise ReplayValidationError(
                "action kind must be one of: aim, fire, swap, wait"
            )
        if self.kind == "aim" and self.angle is None:
            raise ReplayValidationError(
                "aim actions require a finite numeric angle"
            )
        if self.angle is not None:
            if (
                self.kind == "wait"
                or isinstance(self.angle, bool)
                or not isinstance(self.angle, Real)
                or not math.isfinite(float(self.angle))
            ):
                raise ReplayValidationError(
                    "angle must be finite and is only valid for aim, fire, "
                    "or swap actions"
                )
            # Fire/swap may carry the cursor angle observed in the same native
            # input tick.  Requiring a separate aim action one tick earlier
            # would introduce an artificial phase error in golden replays.
            object.__setattr__(self, "angle", float(self.angle))

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"tick": self.tick, "kind": self.kind}
        if self.angle is not None:
            result["angle"] = self.angle
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReplayAction":
        data = _mapping(value, "action")
        _check_keys(
            data,
            required={"tick", "kind"},
            optional={"angle"},
            name="action",
        )
        kind = data["kind"]
        if not isinstance(kind, str):
            raise ReplayValidationError("action kind must be a string")
        return cls(
            tick=data["tick"],
            kind=kind,  # type: ignore[arg-type]
            angle=data.get("angle"),
        )


@dataclass(frozen=True, slots=True)
class ReplayPlan:
    """Replay inputs, independent of any recorded result."""

    level: str
    seed: int
    total_ticks: int
    actions: tuple[ReplayAction, ...] = ()
    profile_mode: str = SUPPORTED_PROFILE_MODE

    def __post_init__(self) -> None:
        if not isinstance(self.level, str) or not self.level.strip():
            raise ReplayValidationError("level must be a non-empty string")
        seed = _integer(self.seed, "seed")
        if not 0 <= seed <= (2**64 - 1):
            raise ReplayValidationError("seed must fit in an unsigned 64-bit integer")
        total_ticks = _integer(self.total_ticks, "total_ticks")
        if total_ticks < 0:
            raise ReplayValidationError("total_ticks must not be negative")
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "total_ticks", total_ticks)
        if self.profile_mode != SUPPORTED_PROFILE_MODE:
            raise ReplayValidationError(
                "unsupported replay profile_mode; only "
                f"{SUPPORTED_PROFILE_MODE!r} is currently implemented"
            )

        actions = tuple(self.actions)
        if any(not isinstance(action, ReplayAction) for action in actions):
            raise ReplayValidationError(
                "actions must contain only ReplayAction objects"
            )
        previous_tick = 0
        for action in actions:
            if action.tick <= previous_tick:
                raise ReplayValidationError(
                    "actions must be strictly ordered with no duplicate ticks"
                )
            if action.tick > total_ticks:
                raise ReplayValidationError(
                    f"action tick {action.tick} exceeds total_ticks={total_ticks}"
                )
            previous_tick = action.tick
        object.__setattr__(self, "actions", actions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "seed": self.seed,
            "total_ticks": self.total_ticks,
            "profile_mode": self.profile_mode,
            "actions": [action.to_dict() for action in self.actions],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReplayPlan":
        data = _mapping(value, "replay plan")
        _check_keys(
            data,
            required={
                "level",
                "seed",
                "total_ticks",
                "profile_mode",
                "actions",
            },
            name="replay plan",
        )
        actions = tuple(
            ReplayAction.from_dict(_mapping(item, "action"))
            for item in _sequence(data["actions"], "actions")
        )
        return cls(
            level=data["level"],
            seed=data["seed"],
            total_ticks=data["total_ticks"],
            actions=actions,
            profile_mode=data["profile_mode"],
        )


@dataclass(frozen=True, slots=True)
class ReplayEvents:
    """JSON-safe copy of :class:`~zuma_rl.revenge_core.TickEvents`."""

    fired: int = 0
    hits: int = 0
    inserted: int = 0
    matches: int = 0
    balls_exploded: int = 0
    balls_removed: int = 0
    score_delta: int = 0
    powerups_spawned: int = 0
    powerups_triggered: int = 0
    fruit_chance_draws: int = 0
    fruits_spawned: int = 0
    fruits_expired: int = 0
    fruits_collected: int = 0
    zuma_triggered: bool = False
    win_pending: bool = False
    loss_started: bool = False
    outcome: str | None = None

    def __post_init__(self) -> None:
        counter_names = (
            "fired",
            "hits",
            "inserted",
            "matches",
            "balls_exploded",
            "balls_removed",
            "powerups_spawned",
            "powerups_triggered",
            "fruit_chance_draws",
            "fruits_spawned",
            "fruits_expired",
            "fruits_collected",
        )
        for name in counter_names:
            value = _integer(getattr(self, name), f"events.{name}")
            if value < 0:
                raise ReplayValidationError(f"events.{name} must not be negative")
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "score_delta",
            _integer(self.score_delta, "events.score_delta"),
        )
        object.__setattr__(
            self,
            "zuma_triggered",
            _boolean(self.zuma_triggered, "events.zuma_triggered"),
        )
        object.__setattr__(
            self,
            "win_pending",
            _boolean(self.win_pending, "events.win_pending"),
        )
        object.__setattr__(
            self,
            "loss_started",
            _boolean(self.loss_started, "events.loss_started"),
        )
        if self.outcome is not None and not isinstance(self.outcome, str):
            raise ReplayValidationError("events.outcome must be a string or null")

    @classmethod
    def from_tick_events(cls, events: TickEvents) -> "ReplayEvents":
        return cls(
            fired=events.fired,
            hits=events.hits,
            inserted=events.inserted,
            matches=events.matches,
            balls_exploded=events.balls_exploded,
            balls_removed=events.balls_removed,
            score_delta=events.score_delta,
            powerups_spawned=events.powerups_spawned,
            powerups_triggered=events.powerups_triggered,
            fruit_chance_draws=events.fruit_chance_draws,
            fruits_spawned=events.fruits_spawned,
            fruits_expired=events.fruits_expired,
            fruits_collected=events.fruits_collected,
            zuma_triggered=events.zuma_triggered,
            win_pending=events.win_pending,
            loss_started=events.loss_started,
            outcome=events.outcome,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "fired": self.fired,
            "hits": self.hits,
            "inserted": self.inserted,
            "matches": self.matches,
            "balls_exploded": self.balls_exploded,
            "balls_removed": self.balls_removed,
            "score_delta": self.score_delta,
            "powerups_spawned": self.powerups_spawned,
            "powerups_triggered": self.powerups_triggered,
            "fruit_chance_draws": self.fruit_chance_draws,
            "fruits_spawned": self.fruits_spawned,
            "fruits_expired": self.fruits_expired,
            "fruits_collected": self.fruits_collected,
            "zuma_triggered": self.zuma_triggered,
            "win_pending": self.win_pending,
            "loss_started": self.loss_started,
            "outcome": self.outcome,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReplayEvents":
        data = _mapping(value, "events")
        fields = {
            "fired",
            "hits",
            "inserted",
            "matches",
            "balls_exploded",
            "balls_removed",
            "score_delta",
            "powerups_spawned",
            "powerups_triggered",
            "fruit_chance_draws",
            "fruits_spawned",
            "fruits_expired",
            "fruits_collected",
            "zuma_triggered",
            "win_pending",
            "loss_started",
            "outcome",
        }
        _check_keys(data, required=fields, name="events")
        return cls(**{name: data[name] for name in fields})


@dataclass(frozen=True, slots=True)
class ReplayTickRecord:
    """Optional diagnostic data for one executed native tick."""

    tick: int
    action: ReplayAction | None
    action_accepted: bool | None
    events: ReplayEvents
    state_fingerprint: str

    def __post_init__(self) -> None:
        tick = _integer(self.tick, "tick record tick")
        if tick < 1:
            raise ReplayValidationError("tick record tick must be at least one")
        object.__setattr__(self, "tick", tick)
        if self.action is not None:
            if not isinstance(self.action, ReplayAction):
                raise ReplayValidationError(
                    "tick record action must be a ReplayAction or null"
                )
            if self.action.tick != tick:
                raise ReplayValidationError(
                    "tick record action tick must match its record tick"
                )
            if self.action_accepted is None:
                raise ReplayValidationError(
                    "an explicit action requires action_accepted"
                )
        elif self.action_accepted is not None:
            raise ReplayValidationError(
                "an implicit wait must have null action_accepted"
            )
        if self.action_accepted is not None:
            object.__setattr__(
                self,
                "action_accepted",
                _boolean(self.action_accepted, "action_accepted"),
            )
        if not isinstance(self.events, ReplayEvents):
            raise ReplayValidationError("events must be a ReplayEvents object")
        object.__setattr__(
            self,
            "state_fingerprint",
            _validate_fingerprint(
                self.state_fingerprint,
                "state_fingerprint",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tick": self.tick,
            "action": None if self.action is None else self.action.to_dict(),
            "action_accepted": self.action_accepted,
            "events": self.events.to_dict(),
            "state_fingerprint": self.state_fingerprint,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReplayTickRecord":
        data = _mapping(value, "tick record")
        _check_keys(
            data,
            required={
                "tick",
                "action",
                "action_accepted",
                "events",
                "state_fingerprint",
            },
            name="tick record",
        )
        raw_action = data["action"]
        action = (
            None
            if raw_action is None
            else ReplayAction.from_dict(_mapping(raw_action, "tick record action"))
        )
        return cls(
            tick=data["tick"],
            action=action,
            action_accepted=data["action_accepted"],
            events=ReplayEvents.from_dict(_mapping(data["events"], "events")),
            state_fingerprint=data["state_fingerprint"],
        )


@dataclass(frozen=True, slots=True)
class ReplayResult:
    """A completed replay and its cumulative state-trajectory fingerprint."""

    plan: ReplayPlan
    ticks_executed: int
    final_fingerprint: str
    tick_records: tuple[ReplayTickRecord, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.plan, ReplayPlan):
            raise ReplayValidationError("plan must be a ReplayPlan")
        ticks_executed = _integer(self.ticks_executed, "ticks_executed")
        if not 0 <= ticks_executed <= self.plan.total_ticks:
            raise ReplayValidationError(
                "ticks_executed must be between zero and total_ticks"
            )
        object.__setattr__(self, "ticks_executed", ticks_executed)
        object.__setattr__(
            self,
            "final_fingerprint",
            _validate_fingerprint(
                self.final_fingerprint,
                "final_fingerprint",
            ),
        )
        if self.tick_records is None:
            return

        records = tuple(self.tick_records)
        if len(records) != ticks_executed:
            raise ReplayValidationError(
                "tick_records must contain exactly one record per executed tick"
            )
        actions_by_tick = {action.tick: action for action in self.plan.actions}
        for expected_tick, record in enumerate(records, start=1):
            if not isinstance(record, ReplayTickRecord):
                raise ReplayValidationError(
                    "tick_records must contain ReplayTickRecord objects"
                )
            if record.tick != expected_tick:
                raise ReplayValidationError(
                    "tick_records must be contiguous and start at tick one"
                )
            if record.action != actions_by_tick.get(expected_tick):
                raise ReplayValidationError(
                    f"tick record action disagrees with plan at tick {expected_tick}"
                )
        object.__setattr__(self, "tick_records", records)

    def to_dict(self) -> dict[str, Any]:
        value = {
            "schema": REPLAY_SCHEMA,
            "version": REPLAY_VERSION,
            **self.plan.to_dict(),
            "ticks_executed": self.ticks_executed,
            "fingerprint_algorithm": FINGERPRINT_ALGORITHM,
            "fingerprint_scope": FINGERPRINT_SCOPE,
            "final_fingerprint": self.final_fingerprint,
        }
        if self.tick_records is not None:
            value["tick_events"] = [
                record.to_dict() for record in self.tick_records
            ]
        return value

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            indent=indent,
            sort_keys=True,
            separators=(",", ":") if indent is None else None,
        )

    def write_json(self, path: str | Path, *, indent: int | None = 2) -> None:
        Path(path).write_text(
            self.to_json(indent=indent) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReplayResult":
        data = _mapping(value, "replay document")
        required = {
            "schema",
            "version",
            "level",
            "seed",
            "total_ticks",
            "profile_mode",
            "actions",
            "ticks_executed",
            "fingerprint_algorithm",
            "fingerprint_scope",
            "final_fingerprint",
        }
        _check_keys(
            data,
            required=required,
            optional={"tick_events"},
            name="replay document",
        )
        if data["schema"] != REPLAY_SCHEMA:
            raise ReplayValidationError(
                f"unsupported replay schema: {data['schema']!r}"
            )
        version = _integer(data["version"], "version")
        if version != REPLAY_VERSION:
            raise ReplayValidationError(
                f"unsupported replay version: {version}"
            )
        if data["fingerprint_algorithm"] != FINGERPRINT_ALGORITHM:
            raise ReplayValidationError("unsupported fingerprint algorithm")
        if data["fingerprint_scope"] != FINGERPRINT_SCOPE:
            raise ReplayValidationError("unsupported fingerprint scope")

        plan = ReplayPlan.from_dict(
            {
                "level": data["level"],
                "seed": data["seed"],
                "total_ticks": data["total_ticks"],
                "profile_mode": data["profile_mode"],
                "actions": data["actions"],
            }
        )
        records: tuple[ReplayTickRecord, ...] | None = None
        if "tick_events" in data:
            records = tuple(
                ReplayTickRecord.from_dict(_mapping(item, "tick record"))
                for item in _sequence(data["tick_events"], "tick_events")
            )
        return cls(
            plan=plan,
            ticks_executed=data["ticks_executed"],
            final_fingerprint=data["final_fingerprint"],
            tick_records=records,
        )

    @classmethod
    def from_json(cls, text: str) -> "ReplayResult":
        try:
            value = json.loads(text)
        except (json.JSONDecodeError, TypeError) as error:
            raise ReplayValidationError(f"invalid replay JSON: {error}") from error
        return cls.from_dict(_mapping(value, "replay document"))

    @classmethod
    def read_json(cls, path: str | Path) -> "ReplayResult":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


def _canonical_value(value: Any) -> Any:
    """Convert state values to a cross-run, JSON-canonical representation."""

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number):
            raise ReplayValidationError(
                "simulator state contains a non-finite floating-point value"
            )
        # Hex retains the exact binary value and avoids locale/decimal
        # formatting differences between machines.
        return {"$float": number.hex()}
    if isinstance(value, Enum):
        return _canonical_value(value.value)
    if isinstance(value, np.ndarray):
        return _canonical_value(value.tolist())
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ReplayValidationError(
                "simulator state mappings must have string keys"
            )
        return {
            key: _canonical_value(item)
            for key, item in sorted(value.items())
        }
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    raise ReplayValidationError(
        f"unsupported simulator state value: {type(value).__name__}"
    )


def _array_descriptor(value: Any) -> dict[str, Any]:
    array = np.ascontiguousarray(np.asarray(value))
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def _field_mapping(value: Any, name: str) -> dict[str, Any]:
    if not is_dataclass(value):
        raise ReplayValidationError(
            f"simulator {name} must be a dataclass for stable replay hashing"
        )
    return {
        field.name: getattr(value, field.name)
        for field in fields(value)
    }


def _environment_bytes(simulator: RevengeSimulator) -> bytes:
    """Canonical static definition that determines future state transitions."""

    curve_definitions: list[dict[str, Any]] = []
    for curve_index, state in enumerate(simulator.curve_states):
        curve = state.curve
        points = state.curve_points
        if points is None:
            waypoints = np.arange(
                int(curve.end_waypoint) + 1,
                dtype=np.float32,
            )
            points = np.asarray(
                curve.point_at_waypoint(waypoints),
                dtype=np.float32,
            )
        tunnels = state.curve_tunnels
        if tunnels is None:
            waypoints = np.arange(
                int(curve.end_waypoint) + 1,
                dtype=np.float32,
            )
            tunnels = np.asarray(
                curve.is_in_tunnel_at_waypoint(waypoints),
                dtype=np.bool_,
            )

        optional_arrays = {}
        for attribute in ("point_flags", "priorities", "absolute_anchors"):
            value = getattr(curve, attribute, None)
            optional_arrays[attribute] = (
                None if value is None else _array_descriptor(value)
            )
        curve_flags = {
            attribute: getattr(curve, attribute, None)
            for attribute in (
                "version",
                "linear",
                "draw_curve",
                "draw_tunnels",
                "destroy_all",
                "draw_pit",
                "die_at_end",
                "edit_type",
            )
        }
        curve_definitions.append(
            {
                "index": curve_index,
                "end_waypoint": int(curve.end_waypoint),
                "entrance_cutoff": int(state.entrance_cutoff),
                "flags": curve_flags,
                "parameters": _field_mapping(
                    curve.parameters,
                    "curve parameters",
                ),
                "points": _array_descriptor(points),
                "in_tunnel": _array_descriptor(tunnels),
                **optional_arrays,
            }
        )
    definition = {
        "profile_mode": simulator.profile_mode,
        "physics_config": _field_mapping(
            simulator.config,
            "physics config",
        ),
        "shooter": _array_descriptor(
            simulator.shooter_positions[simulator.initial_shooter_index]
        ),
        "curves": curve_definitions,
        "powerup_calibration": (
            None
            if simulator.powerup_calibration is None
            else _field_mapping(
                simulator.powerup_calibration,
                "power-up calibration",
            )
        ),
        "fruit_calibration": (
            None
            if simulator.fruit_calibration is None
            else _field_mapping(
                simulator.fruit_calibration,
                "fruit calibration",
            )
        ),
    }
    if simulator.shooter_position_count > 1:
        definition["shooter_positions"] = [
            _array_descriptor(position)
            for position in simulator.shooter_positions
        ]
        definition["initial_shooter_index"] = simulator.initial_shooter_index
    canonical = _canonical_value(definition)
    return json.dumps(
        canonical,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def environment_fingerprint(simulator: RevengeSimulator) -> str:
    """SHA-256 fingerprint of static curve, shooter, profile, and physics."""

    return _digest(_environment_bytes(simulator))


def _state_bytes(
    simulator: RevengeSimulator,
    static_fingerprint: str,
) -> bytes:
    state = {
        "environment_fingerprint": static_fingerprint,
        "state_signature": simulator.state_signature(),
    }
    canonical = _canonical_value(state)
    return json.dumps(
        canonical,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _digest(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def state_fingerprint(simulator: RevengeSimulator) -> str:
    """Return a standalone SHA-256 fingerprint of the current replay state."""

    static_fingerprint = environment_fingerprint(simulator)
    return _digest(_state_bytes(simulator, static_fingerprint))


def _add_trajectory_state(
    hasher: Any,
    simulator: RevengeSimulator,
    static_fingerprint: str,
) -> None:
    payload = _state_bytes(simulator, static_fingerprint)
    hasher.update(len(payload).to_bytes(8, byteorder="big", signed=False))
    hasher.update(payload)


def _default_simulator_factory(level: str, seed: int) -> RevengeSimulator:
    return RevengeSimulator.from_installed(
        level_id=level,
        seed=seed,
        profile_mode=SUPPORTED_PROFILE_MODE,
    )


def _apply_action(
    simulator: RevengeSimulator,
    action: ReplayAction | None,
) -> bool | None:
    if action is None:
        return None
    if action.kind == "aim":
        # ReplayAction already guarantees a finite angle.
        simulator.set_aim(float(action.angle))
        return True
    if action.kind == "fire":
        if action.angle is not None:
            simulator.set_aim(float(action.angle))
        return bool(simulator.request_fire(float(simulator.aim_angle)))
    if action.kind == "swap":
        if action.angle is not None:
            simulator.set_aim(float(action.angle))
        return bool(simulator.swap_balls())
    if action.kind == "wait":
        return True
    raise ReplayValidationError(f"unsupported replay action: {action.kind!r}")


def run_replay(
    plan: ReplayPlan,
    *,
    simulator_factory: SimulatorFactory | None = None,
    record_tick_events: bool = False,
) -> ReplayResult:
    """Execute ``plan`` and fingerprint its initial and post-tick states.

    ``simulator_factory`` makes CI replays independent of an original-game
    installation.  It must return a fresh simulator at tick zero, initialized
    with the requested seed.  With no factory, ``level`` is loaded read-only
    through :meth:`RevengeSimulator.from_installed`.
    """

    if not isinstance(plan, ReplayPlan):
        raise TypeError("plan must be a ReplayPlan")
    factory: Callable[[str, int], RevengeSimulator] = (
        _default_simulator_factory
        if simulator_factory is None
        else simulator_factory
    )
    simulator = factory(plan.level, plan.seed)
    if not isinstance(simulator, RevengeSimulator):
        raise TypeError("simulator_factory must return a RevengeSimulator")
    if simulator.tick_count != 0:
        raise ReplayValidationError(
            "simulator_factory must return a simulator at tick zero"
        )
    if simulator.profile_mode != plan.profile_mode:
        raise ReplayValidationError(
            "simulator profile_mode does not match the replay plan"
        )

    trajectory = hashlib.sha256()
    static_fingerprint = environment_fingerprint(simulator)
    trajectory.update(
        (
            f"{REPLAY_SCHEMA}\0{FINGERPRINT_SCOPE}\0"
            f"{static_fingerprint}\0"
        ).encode("ascii")
    )
    _add_trajectory_state(trajectory, simulator, static_fingerprint)

    actions_by_tick = {action.tick: action for action in plan.actions}
    records: list[ReplayTickRecord] | None = [] if record_tick_events else None
    ticks_executed = 0
    for tick in range(1, plan.total_ticks + 1):
        if simulator.outcome is not None:
            break
        action = actions_by_tick.get(tick)
        accepted = _apply_action(simulator, action)
        before = simulator.tick_count
        events = simulator.tick(1)
        if simulator.tick_count != before + 1:
            raise ReplayValidationError(
                "simulator did not advance exactly one native tick"
            )
        ticks_executed += 1
        _add_trajectory_state(trajectory, simulator, static_fingerprint)
        if records is not None:
            records.append(
                ReplayTickRecord(
                    tick=tick,
                    action=action,
                    action_accepted=accepted,
                    events=ReplayEvents.from_tick_events(events),
                    state_fingerprint=_digest(
                        _state_bytes(simulator, static_fingerprint)
                    ),
                )
            )

    return ReplayResult(
        plan=plan,
        ticks_executed=ticks_executed,
        final_fingerprint=f"sha256:{trajectory.hexdigest()}",
        tick_records=None if records is None else tuple(records),
    )


def verify_replay(
    expected: ReplayResult,
    *,
    simulator_factory: SimulatorFactory | None = None,
) -> ReplayResult:
    """Replay a stored result and raise with the first available divergence."""

    actual = run_replay(
        expected.plan,
        simulator_factory=simulator_factory,
        record_tick_events=expected.tick_records is not None,
    )
    if actual.ticks_executed != expected.ticks_executed:
        raise ReplayMismatchError(
            "executed tick count differs: expected "
            f"{expected.ticks_executed}, got {actual.ticks_executed}"
        )
    if actual.final_fingerprint != expected.final_fingerprint:
        if actual.tick_records is not None and expected.tick_records is not None:
            for actual_tick, expected_tick in zip(
                actual.tick_records,
                expected.tick_records,
                strict=False,
            ):
                if (
                    actual_tick.state_fingerprint
                    != expected_tick.state_fingerprint
                ):
                    raise ReplayMismatchError(
                        "state trajectory first differs at native tick "
                        f"{actual_tick.tick}: expected "
                        f"{expected_tick.state_fingerprint}, got "
                        f"{actual_tick.state_fingerprint}"
                    )
        raise ReplayMismatchError(
            "state trajectory fingerprint differs: expected "
            f"{expected.final_fingerprint}, got {actual.final_fingerprint}"
        )
    if actual.tick_records != expected.tick_records:
        raise ReplayMismatchError(
            "state fingerprints match, but recorded tick events differ"
        )
    return actual


__all__ = [
    "FINGERPRINT_ALGORITHM",
    "FINGERPRINT_SCOPE",
    "REPLAY_SCHEMA",
    "REPLAY_VERSION",
    "ReplayAction",
    "ReplayEvents",
    "ReplayMismatchError",
    "ReplayPlan",
    "ReplayResult",
    "ReplayTickRecord",
    "ReplayValidationError",
    "SimulatorFactory",
    "environment_fingerprint",
    "run_replay",
    "state_fingerprint",
    "verify_replay",
]
