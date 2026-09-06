"""Strict tick-by-tick comparison of PC evidence and simulator snapshots.

The PC trace is deliberately partial: only ``observed`` and ``inferred``
measurements are evidence.  A required channel that cannot be compared makes
the result ``INCOMPARABLE``; a comparable value outside its frozen tolerance
makes the result ``FAIL``.

Simulator events are represented by their kind because PC event payloads can
contain tracker-specific identities which have no stable simulator analogue.
"""

from __future__ import annotations

import math
from collections import Counter, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from zuma_rl.pc_golden import (
    EVENT_KINDS,
    MEASUREMENT_CHANNELS,
    SUPPORTED_COVERAGE_CHANNELS,
    ComparisonResult,
    ComparisonStatus,
    CoverageRange,
    CoverageStatus,
    GoldenEvent,
    MeasurementStatus,
    PcGoldenArtifactError,
    PcGoldenManifest,
    PcGoldenTrace,
    PcGoldenValidationError,
    validate_measurement_value,
)
from zuma_rl.revenge_core import RevengeSimulator
from zuma_rl.verify_pc_golden import verify_pc_golden_case

_EXACT_CHANNELS = frozenset(
    {
        "score",
        "current_color",
        "next_color",
        "gun_state",
        "outcome",
        "zuma_reached",
        "chain_count",
        "projectile_count",
    }
)
_CENTER_CHANNELS = frozenset({"chain_centers", "projectile_centers"})
_WAYPOINT_CHANNEL = "chain_waypoints"
_EVENT_CHANNEL = "events"
_MEASUREMENT_CHANNELS = MEASUREMENT_CHANNELS
_SUPPORTED_CHANNELS = SUPPORTED_COVERAGE_CHANNELS
_AVAILABLE_STATUSES = frozenset(
    {MeasurementStatus.OBSERVED, MeasurementStatus.INFERRED}
)
_POST_UPDATE_SAMPLE_PHASE = "post_update_presented"
_TOPOLOGY_EVENT_KINDS = EVENT_KINDS.intersection(
    {"inserted", "match_started", "balls_exploded", "balls_removed"}
)
_MISSING = object()
_REASON_LIMIT = 32


def _strict_tick(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise PcGoldenValidationError(f"{name} must be an integer")
    tick = int(value)
    if tick < 0:
        raise PcGoldenValidationError(f"{name} must be non-negative")
    return tick


def _event_kind(value: Any, name: str) -> str:
    if isinstance(value, str):
        kind = value
    elif isinstance(value, GoldenEvent):
        kind = value.kind
    elif isinstance(value, Mapping) and "kind" in value:
        kind = value["kind"]
    else:
        kind = getattr(value, "kind", None)
    if not isinstance(kind, str) or not kind.strip():
        raise PcGoldenValidationError(
            f"{name} must be a non-empty event kind or expose one"
        )
    return kind


def _freeze_value(value: Any, name: str) -> Any:
    """Detach a JSON-like value and reject ambiguous/non-finite state."""

    if isinstance(value, np.ndarray):
        value = value.tolist()
    elif isinstance(value, np.generic):
        value = value.item()
    elif isinstance(value, Enum):
        value = value.value

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PcGoldenValidationError(f"{name} must be finite")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise PcGoldenValidationError(f"{name} keys must be strings")
        return MappingProxyType(
            {
                key: _freeze_value(item, f"{name}.{key}")
                for key, item in value.items()
            }
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(
            _freeze_value(item, f"{name}[{index}]")
            for index, item in enumerate(value)
        )
    raise PcGoldenValidationError(
        f"{name} contains unsupported type {type(value).__name__}"
    )


@dataclass(frozen=True, slots=True)
class SimulatorTick:
    """One simulator snapshot aligned to a native 100 Hz PC trace tick.

    ``source_tick`` is copied from ``RevengeSimulator.tick_count`` by
    :func:`snapshot_revenge_simulator`; it must equal the trace-facing
    ``tick``.  ``sample_phase`` is explicit so a pre-update state cannot be
    relabelled as the v3 post-update sample.

    ``events`` may contain event-kind strings, :class:`GoldenEvent` objects,
    mappings with a ``kind`` field, or other objects exposing a string
    ``kind`` attribute.  They are normalized to event-kind strings.
    """

    tick: int
    source_tick: int
    sample_phase: str
    events: tuple[str, ...] = ()
    measurements: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        tick = _strict_tick(self.tick, "simulator tick")
        source_tick = _strict_tick(
            self.source_tick,
            "simulator source_tick",
        )
        if source_tick != tick:
            raise PcGoldenValidationError(
                "simulator source_tick must equal its aligned tick"
            )
        if self.sample_phase != _POST_UPDATE_SAMPLE_PHASE:
            raise PcGoldenValidationError(
                "simulator sample_phase must be "
                f"{_POST_UPDATE_SAMPLE_PHASE!r}"
            )
        object.__setattr__(self, "tick", tick)
        object.__setattr__(self, "source_tick", source_tick)
        if isinstance(self.events, (str, bytes)):
            raise PcGoldenValidationError(
                "simulator events must be a sequence, not a string"
            )
        try:
            raw_events = tuple(self.events)
        except TypeError as error:
            raise PcGoldenValidationError(
                "simulator events must be a sequence"
            ) from error
        object.__setattr__(
            self,
            "events",
            tuple(
                _event_kind(event, f"simulator events[{index}]")
                for index, event in enumerate(raw_events)
            ),
        )

        if not isinstance(self.measurements, Mapping):
            raise PcGoldenValidationError(
                "simulator measurements must be a mapping"
            )
        normalized: dict[str, Any] = {}
        for channel, value in sorted(self.measurements.items()):
            if not isinstance(channel, str) or not channel.strip():
                raise PcGoldenValidationError(
                    "simulator measurement channels must be non-empty strings"
                )
            normalized[channel] = _freeze_value(
                value,
                f"simulator measurement {channel!r}",
            )
        object.__setattr__(
            self,
            "measurements",
            MappingProxyType(normalized),
        )


class _InvalidChannelValue(ValueError):
    pass


@dataclass(slots=True)
class _ReasonCollector:
    total: int = 0
    reasons: list[str] = field(default_factory=list)

    def add(self, reason: str) -> None:
        self.total += 1
        if len(self.reasons) < _REASON_LIMIT:
            self.reasons.append(reason)

    def rendered(self, category: str) -> tuple[str, ...]:
        result = list(self.reasons)
        omitted = self.total - len(self.reasons)
        if omitted:
            result.append(f"{omitted} additional {category} omitted")
        return tuple(result)


@dataclass(slots=True)
class _DriftTracker:
    """Track net error growth over exact 100-tick stable-topology windows."""

    threshold: float
    history: dict[
        str,
        deque[tuple[int, tuple[float, ...]]],
    ] = field(
        default_factory=dict
    )
    maximum: float = 0.0

    def reset(self, channel: str) -> None:
        self.history.pop(channel, None)

    def observe(
        self,
        channel: str,
        tick: int,
        errors: tuple[float, ...],
    ) -> int:
        records = self.history.setdefault(channel, deque())
        if records:
            previous_tick, previous_errors = records[-1]
            if (
                tick != previous_tick + 1
                or len(previous_errors) != len(errors)
            ):
                records.clear()
        records.append((tick, errors))
        while records and tick - records[0][0] > 100:
            records.popleft()
        if not records or tick - records[0][0] != 100:
            return 0
        _, prior_errors = records[0]
        violations = 0
        for previous_error, current_error in zip(
            prior_errors,
            errors,
            strict=True,
        ):
            net_drift = max(0.0, current_error - previous_error)
            self.maximum = max(self.maximum, net_drift)
            violations += int(net_drift > self.threshold)
        return violations


def _finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise PcGoldenValidationError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise PcGoldenValidationError(
            f"{name} must be finite and non-negative"
        )
    return result


def _validate_simulator_ticks(
    simulator_ticks: Iterable[SimulatorTick],
    tick_end: int,
    sample_phase: str,
) -> tuple[SimulatorTick, ...]:
    if isinstance(simulator_ticks, (str, bytes)):
        raise PcGoldenValidationError(
            "simulator_ticks must be an iterable of SimulatorTick objects"
        )
    try:
        records = tuple(simulator_ticks)
    except TypeError as error:
        raise PcGoldenValidationError(
            "simulator_ticks must be iterable"
        ) from error
    expected_length = tick_end + 1
    if len(records) != expected_length:
        raise PcGoldenValidationError(
            "simulator ticks must cover tick 0 through tick_end exactly; "
            f"expected {expected_length} records, got {len(records)}"
        )
    for expected_tick, record in enumerate(records):
        if not isinstance(record, SimulatorTick):
            raise PcGoldenValidationError(
                "simulator_ticks must contain only SimulatorTick objects"
            )
        if record.tick != expected_tick:
            raise PcGoldenValidationError(
                "simulator ticks must be contiguous and start at zero; "
                f"expected {expected_tick}, got {record.tick}"
            )
        if record.source_tick != expected_tick:
            raise PcGoldenValidationError(
                "simulator source ticks must match aligned trace ticks; "
                f"expected {expected_tick}, got {record.source_tick}"
            )
        if record.sample_phase != sample_phase:
            raise PcGoldenValidationError(
                "simulator sample phase does not match the manifest clock"
            )
    return records


def _coverage_at(
    ranges: Sequence[CoverageRange],
    tick: int,
) -> CoverageRange | None:
    for item in ranges:
        if item.start_tick <= tick <= item.end_tick:
            return item
    return None


def _numeric_sequence(value: Any, name: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise _InvalidChannelValue(f"{name} must be a sequence")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise _InvalidChannelValue(f"{name} must contain only numbers")
        number = float(item)
        if not math.isfinite(number):
            raise _InvalidChannelValue(f"{name} must contain finite numbers")
        result.append(number)
    return tuple(result)


def _centers(value: Any, name: str) -> tuple[tuple[float, float], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise _InvalidChannelValue(f"{name} must be a sequence")
    result: list[tuple[float, float]] = []
    for index, item in enumerate(value):
        point = _numeric_sequence(item, f"{name}[{index}]")
        if len(point) != 2:
            raise _InvalidChannelValue(
                f"{name}[{index}] must contain exactly two coordinates"
            )
        result.append((point[0], point[1]))
    return tuple(result)


def _record_unavailable(
    *,
    required: bool,
    failures: _ReasonCollector,
    metrics: dict[str, int | float],
    reason: str,
) -> None:
    if required:
        failures.add(reason)
        metrics["incomparable_count"] += 1
    else:
        metrics["optional_values_skipped"] += 1


def _is_subsequence(
    expected: Sequence[str],
    actual: Sequence[str],
) -> bool:
    expected_index = 0
    for kind in actual:
        if expected_index < len(expected) and kind == expected[expected_index]:
            expected_index += 1
    return expected_index == len(expected)


def _event_tick_counts(
    expected: Sequence[GoldenEvent],
    actual: Sequence[str],
    *,
    complete: bool,
) -> tuple[int, int, int, bool]:
    """Return matches, unmatched PC/simulator counts, and order mismatch.

    Events carrying a sequence number retain their relative order.  Events
    without one are an unordered multiset.  On partial coverage, extra
    simulator events are not negative evidence.
    """

    expected_counter = Counter(event.kind for event in expected)
    actual_counter = Counter(actual)
    matches = sum(
        min(count, actual_counter[kind])
        for kind, count in expected_counter.items()
    )
    unmatched_expected = len(expected) - matches
    unmatched_actual = (
        len(actual) - matches
        if complete
        else 0
    )
    known_order = tuple(
        event.kind
        for event in sorted(
            (
                event
                for event in expected
                if event.sequence is not None
            ),
            key=lambda event: int(event.sequence),
        )
    )
    order_mismatch = bool(known_order) and not _is_subsequence(
        known_order,
        actual,
    )
    return (
        matches,
        unmatched_expected,
        unmatched_actual,
        order_mismatch,
    )


def _compare_events(
    *,
    ranges: Sequence[CoverageRange],
    trace: PcGoldenTrace,
    simulator_ticks: Sequence[SimulatorTick],
    tolerance: int,
    failures: _ReasonCollector,
    metrics: dict[str, int | float],
) -> None:
    if tolerance != 0:
        raise PcGoldenValidationError(
            "PC golden event comparison requires zero tick tolerance"
        )

    for pc_record, simulator_record in zip(
        trace.records,
        simulator_ticks,
        strict=True,
    ):
        coverage = _coverage_at(ranges, pc_record.tick)
        if coverage is None or coverage.status is CoverageStatus.ABSENT:
            continue
        expected = pc_record.events
        actual = simulator_record.events
        (
            matches,
            unmatched_expected,
            unmatched_actual,
            order_mismatch,
        ) = _event_tick_counts(
            expected,
            actual,
            complete=coverage.status is CoverageStatus.COMPLETE,
        )
        metrics["events_pc"] += len(expected)
        metrics["events_simulator"] += len(actual)
        metrics["events_matched"] += matches
        event_mismatches = unmatched_expected + unmatched_actual
        if event_mismatches:
            failures.add(
                "event kinds differ after one-to-one matching at tick "
                f"{pc_record.tick}: {unmatched_expected} PC and "
                f"{unmatched_actual} simulator events remain unmatched"
            )
            metrics["mismatch_count"] += event_mismatches
        if order_mismatch:
            failures.add(
                "known PC event sequence differs from simulator order at "
                f"tick {pc_record.tick}"
            )
            metrics["mismatch_count"] += 1


def _compare_pc_trace(
    manifest: PcGoldenManifest,
    trace: PcGoldenTrace,
    simulator_ticks: Iterable[SimulatorTick],
) -> ComparisonResult:
    """Internal comparator for an already accepted PC golden case."""

    # These calls intentionally precede all differential work.  Artifact,
    # environment, clock, frame, and evidence-readiness failures must not be
    # obscured by a downstream state mismatch.
    trace.validate_against_manifest(manifest)
    readiness = manifest.coverage_readiness()

    simulator_records = _validate_simulator_ticks(
        simulator_ticks,
        manifest.clock.tick_end,
        manifest.clock.sample_phase,
    )
    if readiness.status is not ComparisonStatus.PASS:
        return readiness

    contract = manifest.comparison_contract
    event_tolerance = _strict_tick(
        contract.event_tick_tolerance,
        "comparison_contract.event_tick_tolerance",
    )
    center_tolerance = _finite_nonnegative(
        contract.center_l2_tolerance_px,
        "comparison_contract.center_l2_tolerance_px",
    )
    waypoint_tolerance = _finite_nonnegative(
        contract.waypoint_abs_tolerance,
        "comparison_contract.waypoint_abs_tolerance",
    )
    drift_tolerance = _finite_nonnegative(
        contract.max_drift_per_100_ticks,
        "comparison_contract.max_drift_per_100_ticks",
    )
    drift = _DriftTracker(drift_tolerance)

    metrics: dict[str, int | float] = {
        "ticks_compared": len(simulator_records),
        "measurement_comparisons": 0,
        "scalar_comparisons": 0,
        "center_points_compared": 0,
        "waypoints_compared": 0,
        "events_pc": 0,
        "events_simulator": 0,
        "events_matched": 0,
        "max_event_tick_error": 0,
        "max_center_l2_error_px": 0.0,
        "max_chain_center_l2_error_px": 0.0,
        "max_projectile_center_l2_error_px": 0.0,
        "max_waypoint_abs_error": 0.0,
        "max_drift_per_100_ticks": 0.0,
        "mismatch_count": 0,
        "incomparable_count": 0,
        "optional_values_skipped": 0,
        "unsupported_optional_ranges": 0,
        "projectile_center_ranges_skipped": 0,
        "topology_segment_resets": 0,
    }
    incomparable = _ReasonCollector()
    mismatches = _ReasonCollector()

    ranges_by_channel: dict[str, list[CoverageRange]] = {}
    for item in manifest.coverage:
        ranges_by_channel.setdefault(item.channel, []).append(item)
        if item.channel == "projectile_centers":
            metrics["projectile_center_ranges_skipped"] += 1
            if item.required:
                incomparable.add(
                    "required channel 'projectile_centers' has no stable "
                    "cross-runtime identity/order contract"
                )
                metrics["incomparable_count"] += 1
            else:
                metrics["optional_values_skipped"] += 1
            continue
        if item.channel in _SUPPORTED_CHANNELS:
            continue
        if item.required:
            incomparable.add(
                f"required channel {item.channel!r} is not supported "
                "by the differential comparator"
            )
            metrics["incomparable_count"] += 1
        else:
            metrics["unsupported_optional_ranges"] += 1

    topology_ticks = {
        pc_record.tick
        for pc_record, simulator_record in zip(
            trace.records,
            simulator_records,
            strict=True,
        )
        if (
            any(
                event.kind in _TOPOLOGY_EVENT_KINDS
                for event in pc_record.events
            )
            or any(
                kind in _TOPOLOGY_EVENT_KINDS
                for kind in simulator_record.events
            )
        )
    }

    for coverage in manifest.coverage:
        channel = coverage.channel
        if channel not in _MEASUREMENT_CHANNELS:
            continue
        if channel == "projectile_centers":
            drift.reset(channel)
            continue
        if coverage.status is CoverageStatus.ABSENT:
            drift.reset(channel)
            continue
        for tick in range(coverage.start_tick, coverage.end_tick + 1):
            if (
                tick in topology_ticks
                and channel in {"chain_centers", "chain_waypoints"}
            ):
                drift.reset(channel)
                metrics["topology_segment_resets"] += 1
            pc_record = trace.records[tick]
            simulator_record = simulator_records[tick]
            measurement = pc_record.measurements.get(channel)
            must_be_available = (
                coverage.required
                or coverage.status is CoverageStatus.COMPLETE
            )
            if measurement is None:
                drift.reset(channel)
                _record_unavailable(
                    required=must_be_available,
                    failures=incomparable,
                    metrics=metrics,
                    reason=(
                        f"channel {channel!r} is missing from "
                        f"PC tick {tick}"
                    ),
                )
                continue
            if measurement.status not in _AVAILABLE_STATUSES:
                drift.reset(channel)
                _record_unavailable(
                    required=must_be_available,
                    failures=incomparable,
                    metrics=metrics,
                    reason=(
                        f"channel {channel!r} is unavailable at "
                        f"PC tick {tick} ({measurement.status.value})"
                    ),
                )
                continue
            actual_raw = simulator_record.measurements.get(channel, _MISSING)
            if actual_raw is _MISSING:
                drift.reset(channel)
                _record_unavailable(
                    required=must_be_available,
                    failures=incomparable,
                    metrics=metrics,
                    reason=(
                        f"channel {channel!r} is missing from "
                        f"simulator tick {tick}"
                    ),
                )
                continue

            try:
                validate_measurement_value(channel, measurement.value)
                validate_measurement_value(channel, actual_raw)
                if channel in _EXACT_CHANNELS:
                    metrics["measurement_comparisons"] += 1
                    metrics["scalar_comparisons"] += 1
                    if measurement.value != actual_raw:
                        mismatches.add(
                            f"exact channel {channel!r} differs at tick {tick}"
                        )
                        metrics["mismatch_count"] += 1
                    continue

                if channel in _CENTER_CHANNELS:
                    expected_centers = _centers(
                        measurement.value,
                        f"PC {channel}",
                    )
                    actual_centers = _centers(
                        actual_raw,
                        f"simulator {channel}",
                    )
                    metrics["measurement_comparisons"] += 1
                    if len(expected_centers) != len(actual_centers):
                        drift.reset(channel)
                        mismatches.add(
                            f"ordered channel {channel!r} has a different "
                            f"length at tick {tick}"
                        )
                        metrics["mismatch_count"] += 1
                    local_maximum = 0.0
                    above_tolerance = 0
                    center_errors: list[float] = []
                    for expected, actual in zip(
                        expected_centers,
                        actual_centers,
                        strict=False,
                    ):
                        error = math.hypot(
                            expected[0] - actual[0],
                            expected[1] - actual[1],
                        )
                        center_errors.append(error)
                        local_maximum = max(local_maximum, error)
                        metrics["center_points_compared"] += 1
                        above_tolerance += int(error > center_tolerance)
                    metrics["max_center_l2_error_px"] = max(
                        metrics["max_center_l2_error_px"],
                        local_maximum,
                    )
                    channel_metric = (
                        "max_chain_center_l2_error_px"
                        if channel == "chain_centers"
                        else "max_projectile_center_l2_error_px"
                    )
                    metrics[channel_metric] = max(
                        metrics[channel_metric],
                        local_maximum,
                    )
                    if above_tolerance:
                        mismatches.add(
                            f"ordered channel {channel!r} exceeds its L2 "
                            f"tolerance at tick {tick}"
                        )
                        metrics["mismatch_count"] += above_tolerance
                    if (
                        channel == "chain_centers"
                        and len(expected_centers) == len(actual_centers)
                    ):
                        drift_violations = drift.observe(
                            channel,
                            tick,
                            tuple(center_errors),
                        )
                        if drift_violations:
                            mismatches.add(
                                "ordered channel 'chain_centers' exceeds "
                                "the per-100-tick drift tolerance at "
                                f"tick {tick}"
                            )
                            metrics["mismatch_count"] += drift_violations
                        metrics["max_drift_per_100_ticks"] = max(
                            metrics["max_drift_per_100_ticks"],
                            drift.maximum,
                        )
                    continue

                expected_waypoints = _numeric_sequence(
                    measurement.value,
                    "PC chain_waypoints",
                )
                actual_waypoints = _numeric_sequence(
                    actual_raw,
                    "simulator chain_waypoints",
                )
                metrics["measurement_comparisons"] += 1
                if len(expected_waypoints) != len(actual_waypoints):
                    drift.reset(channel)
                    mismatches.add(
                        "ordered channel 'chain_waypoints' has a different "
                        f"length at tick {tick}"
                    )
                    metrics["mismatch_count"] += 1
                local_maximum = 0.0
                above_tolerance = 0
                waypoint_errors: list[float] = []
                for expected, actual in zip(
                    expected_waypoints,
                    actual_waypoints,
                    strict=False,
                ):
                    error = abs(expected - actual)
                    waypoint_errors.append(error)
                    local_maximum = max(local_maximum, error)
                    metrics["waypoints_compared"] += 1
                    above_tolerance += int(error > waypoint_tolerance)
                metrics["max_waypoint_abs_error"] = max(
                    metrics["max_waypoint_abs_error"],
                    local_maximum,
                )
                if above_tolerance:
                    mismatches.add(
                        "ordered channel 'chain_waypoints' exceeds its "
                        f"absolute tolerance at tick {tick}"
                    )
                    metrics["mismatch_count"] += above_tolerance
                if len(expected_waypoints) == len(actual_waypoints):
                    drift_violations = drift.observe(
                        channel,
                        tick,
                        tuple(waypoint_errors),
                    )
                    if drift_violations:
                        mismatches.add(
                            "ordered channel 'chain_waypoints' exceeds "
                            "the per-100-tick drift tolerance at "
                            f"tick {tick}"
                        )
                        metrics["mismatch_count"] += drift_violations
                    metrics["max_drift_per_100_ticks"] = max(
                        metrics["max_drift_per_100_ticks"],
                        drift.maximum,
                    )
            except (PcGoldenValidationError, _InvalidChannelValue):
                drift.reset(channel)
                _record_unavailable(
                    required=must_be_available,
                    failures=incomparable,
                    metrics=metrics,
                    reason=(
                        f"channel {channel!r} has an invalid value "
                        f"at tick {tick}"
                    ),
                )

    event_ranges = ranges_by_channel.get(_EVENT_CHANNEL)
    if event_ranges:
        _compare_events(
            ranges=event_ranges,
            trace=trace,
            simulator_ticks=simulator_records,
            tolerance=event_tolerance,
            failures=mismatches,
            metrics=metrics,
        )

    if incomparable.total:
        reasons = list(incomparable.rendered("incomparability reasons"))
        if mismatches.total:
            reasons.extend(mismatches.rendered("mismatch reasons"))
        return ComparisonResult(
            status=ComparisonStatus.INCOMPARABLE,
            reasons=tuple(reasons),
            metrics=metrics,
        )
    if mismatches.total:
        return ComparisonResult(
            status=ComparisonStatus.FAIL,
            reasons=mismatches.rendered("mismatch reasons"),
            metrics=metrics,
        )
    return ComparisonResult(
        status=ComparisonStatus.PASS,
        reasons=(),
        metrics=metrics,
    )


def compare_pc_trace(
    manifest: PcGoldenManifest,
    trace: PcGoldenTrace,
    simulator_ticks: Iterable[SimulatorTick],
) -> ComparisonResult:
    """Reject unverified in-memory comparison.

    This compatibility symbol remains importable for older callers, but it is
    fail-closed.  Formal comparison must begin with
    :func:`compare_pc_golden_case`, which executes the read-only verifier.
    """

    del manifest, trace, simulator_ticks
    raise PcGoldenValidationError(
        "unverified compare_pc_trace use is disabled; "
        "use compare_pc_golden_case with a manifest path"
    )


def compare_pc_golden_case(
    manifest_path: str | Path,
    simulator_ticks: Iterable[SimulatorTick],
    *,
    case_root: str | Path | None = None,
    original_root: str | Path | None = None,
) -> ComparisonResult:
    """Verify an immutable case, then run the internal differential."""

    verification = verify_pc_golden_case(
        manifest_path,
        case_root=case_root,
        original_root=original_root,
    )
    if verification.status is not ComparisonStatus.PASS:
        return ComparisonResult(
            status=verification.status,
            reasons=verification.reasons,
            metrics={
                "verification_status": verification.status.value,
                "verification_check_count": len(verification.checks),
            },
        )

    manifest_source = Path(manifest_path)
    root = manifest_source.parent if case_root is None else Path(case_root)
    try:
        manifest = PcGoldenManifest.read_json(manifest_source)
        verified = manifest.verify_artifacts(root)
        trace = PcGoldenTrace.read_ndjson(
            verified[manifest.trace_artifact]
        )
        trace.validate_against_manifest(manifest)
    except (
        OSError,
        UnicodeError,
        PcGoldenArtifactError,
        PcGoldenValidationError,
    ) as error:
        return ComparisonResult(
            status=ComparisonStatus.FAIL,
            reasons=(
                "verified PC golden case changed while preparing comparison",
            ),
            metrics={"error_type": type(error).__name__},
        )
    return _compare_pc_trace(manifest, trace, simulator_ticks)


def snapshot_revenge_simulator(
    simulator: RevengeSimulator,
    *,
    events: Sequence[Any],
) -> SimulatorTick:
    """Bind a post-update snapshot to ``RevengeSimulator.tick_count``.

    Chain values preserve ``simulator.balls`` order.  Projectile values use
    free-projectile order followed by merging-projectile order; this mirrors
    the simulator's two public collections and gives the trace producer a
    deterministic ordering contract.
    """

    if not isinstance(simulator, RevengeSimulator):
        raise TypeError("simulator must be a RevengeSimulator")
    if (
        hasattr(simulator, "_curve_states")
        and simulator.curve_count != 1
    ):
        raise ValueError(
            "the current PC golden snapshot schema is single-curve; "
            "capture and verify a curve-indexed oracle before comparing a "
            "multi-curve Board"
        )
    projectiles = (
        tuple(simulator.free_projectiles)
        + tuple(simulator.merging_projectiles)
    )
    gun_state = simulator.gun_state
    gun_state_value = (
        gun_state.value if isinstance(gun_state, Enum) else str(gun_state)
    )
    measurements = {
        "score": int(simulator.score),
        "current_color": (
            None
            if simulator.current_color is None
            else int(simulator.current_color)
        ),
        "next_color": (
            None if simulator.next_color is None else int(simulator.next_color)
        ),
        "gun_state": gun_state_value,
        "outcome": simulator.outcome,
        "zuma_reached": bool(simulator.zuma_reached),
        "chain_count": len(simulator.balls),
        "projectile_count": len(projectiles),
        "chain_centers": [
            [float(coordinate) for coordinate in simulator.ball_position(ball)]
            for ball in simulator.balls
        ],
        "projectile_centers": [
            [float(coordinate) for coordinate in projectile.position]
            for projectile in projectiles
        ],
        "chain_waypoints": [
            float(ball.waypoint) for ball in simulator.balls
        ],
    }
    source_tick = _strict_tick(
        simulator.tick_count,
        "RevengeSimulator.tick_count",
    )
    return SimulatorTick(
        tick=source_tick,
        source_tick=source_tick,
        sample_phase=_POST_UPDATE_SAMPLE_PHASE,
        events=tuple(events),
        measurements=measurements,
    )


__all__ = [
    "SimulatorTick",
    "compare_pc_golden_case",
    "snapshot_revenge_simulator",
]
