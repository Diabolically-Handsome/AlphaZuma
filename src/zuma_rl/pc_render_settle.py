"""Fail-closed binding from DXGI presents to settled retail draw events.

The DXGI collector records a frame and then samples the retail framework
update counter.  A framework update can advance between those two operations,
so that observation is deliberately retained as raw evidence rather than
silently corrected.  This module derives a second, explicitly provenance-bound
map from the frame's presentation QPC and the high-rate draw-event observer.

The only accepted derivation is the independently calibrated 400 microsecond
settle rule.  Recomputing the map requires every source artifact; consumers do
not trust producer-authored update labels.
"""

from __future__ import annotations

from bisect import bisect_right
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from zuma_rl.pc_golden import (
    PcGoldenValidationError,
    _check_keys,
    _json_loads_strict,
    _mapping,
    _sequence,
    _sha256,
    _strict_int,
)
from zuma_rl.pc_protocol_evidence import (
    PcDxgiCaptureMetadata,
    PcFrameworkUpdateMap,
)


RENDER_SETTLED_UPDATE_SCHEMA = "zuma-rl.pc-render-settled-update-map"
RENDER_SETTLED_UPDATE_VERSION = 1
RENDER_SETTLE_METHOD = (
    "latest_observed_draw_before_present_minus_fixed_delay"
)
RENDER_SETTLE_DELAY_NS = 400_000
MINIMUM_MEAN_POLL_HZ = 4_000.0
MAXIMUM_P99_POLL_INTERVAL_NS = 300_000

_POLL_SCHEMA = "zuma-rl.pc-framework-state-poll-diagnostic"
_CALIBRATION_PREREGISTRATION_SCHEMA = (
    "zuma-rl.pc-render-transport-preregistration"
)
_CALIBRATION_EXECUTION_BINDING_SCHEMA = (
    "zuma-rl.pc-render-transport-execution-binding"
)
_CALIBRATION_REPORT_SCHEMA = "zuma-rl.pc-render-transport-holdout-report"
_MAX_RECORDS = 100_000
_FRAME_CSV_FIELDS = (
    "sequence",
    "present_ticks",
    "qpc_frequency",
    "host_perf_counter_ns",
    "accumulated_frames",
    "raw_offset",
    "raw_bytes",
    "frame_sha256",
)


def _canonical_json_file(value: Mapping[str, Any]) -> str:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )


def _sha256_bytes(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _read_bytes(path: str | Path, name: str) -> bytes:
    try:
        return Path(path).read_bytes()
    except OSError as error:
        raise PcGoldenValidationError(f"{name} could not be read") from error


def _read_json_object(
    path: str | Path,
    name: str,
    *,
    require_canonical: bool = False,
) -> tuple[Mapping[str, Any], bytes, str]:
    payload = _read_bytes(path, name)
    try:
        text = payload.decode("ascii")
    except UnicodeError as error:
        raise PcGoldenValidationError(
            f"{name} could not be read as ASCII"
        ) from error
    value = _mapping(_json_loads_strict(text, name), name)
    if require_canonical and text != _canonical_json_file(value):
        raise PcGoldenValidationError(f"{name} JSON is not canonical")
    return value, payload, _sha256_bytes(payload)


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PcGoldenValidationError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise PcGoldenValidationError(f"{name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class RenderSettledUpdateRecord:
    """One frame's source observation and independently settled assignment."""

    sequence: int
    present_ticks: int
    present_perf_counter_ns: int
    source_update_before: int
    source_update_after: int
    assigned_draw_count: int
    assigned_framework_update: int
    draw_event_sample_after_perf_counter_ns: int
    settle_age_ns: int

    def __post_init__(self) -> None:
        for name in (
            "sequence",
            "present_ticks",
            "present_perf_counter_ns",
            "source_update_before",
            "source_update_after",
            "assigned_draw_count",
            "assigned_framework_update",
            "draw_event_sample_after_perf_counter_ns",
            "settle_age_ns",
        ):
            object.__setattr__(
                self,
                name,
                _strict_int(
                    getattr(self, name),
                    f"render-settled update {name}",
                    minimum=0,
                ),
            )
        if (
            self.present_ticks == 0
            or self.present_perf_counter_ns == 0
            or self.draw_event_sample_after_perf_counter_ns == 0
        ):
            raise PcGoldenValidationError(
                "render-settled update timestamps must be positive"
            )
        if self.source_update_after < self.source_update_before:
            raise PcGoldenValidationError(
                "render-settled source updates must be non-decreasing"
            )
        if self.settle_age_ns != (
            self.present_perf_counter_ns
            - self.draw_event_sample_after_perf_counter_ns
        ):
            raise PcGoldenValidationError(
                "render-settled update age differs from its timestamps"
            )

    def to_dict(self) -> dict[str, int]:
        return {
            "sequence": self.sequence,
            "present_ticks": self.present_ticks,
            "present_perf_counter_ns": self.present_perf_counter_ns,
            "source_update_before": self.source_update_before,
            "source_update_after": self.source_update_after,
            "assigned_draw_count": self.assigned_draw_count,
            "assigned_framework_update": self.assigned_framework_update,
            "draw_event_sample_after_perf_counter_ns": (
                self.draw_event_sample_after_perf_counter_ns
            ),
            "settle_age_ns": self.settle_age_ns,
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> "RenderSettledUpdateRecord":
        data = _mapping(value, "render-settled update record")
        fields = {
            "sequence",
            "present_ticks",
            "present_perf_counter_ns",
            "source_update_before",
            "source_update_after",
            "assigned_draw_count",
            "assigned_framework_update",
            "draw_event_sample_after_perf_counter_ns",
            "settle_age_ns",
        }
        _check_keys(data, required=fields, name="render-settled update record")
        return cls(**{name: data[name] for name in fields})


@dataclass(frozen=True, slots=True)
class PcRenderSettledUpdateMap:
    """Canonical derived map whose complete source set is SHA-256 bound."""

    capture_metadata_sha256: str
    frames_csv_sha256: str
    framework_update_map_sha256: str
    framework_state_sidecar_sha256: str
    framework_poll_sha256: str
    calibration_preregistration_sha256: str
    calibration_execution_binding_sha256: str
    calibration_holdout_report_sha256: str
    process_id: int
    process_creation_filetime_100ns: int
    executable_sha256: str
    records: tuple[RenderSettledUpdateRecord, ...]
    render_settle_delay_ns: int = RENDER_SETTLE_DELAY_NS
    minimum_mean_poll_hz: float = MINIMUM_MEAN_POLL_HZ
    maximum_p99_poll_interval_ns: int = MAXIMUM_P99_POLL_INTERVAL_NS

    def __post_init__(self) -> None:
        for name in (
            "capture_metadata_sha256",
            "frames_csv_sha256",
            "framework_update_map_sha256",
            "framework_state_sidecar_sha256",
            "framework_poll_sha256",
            "calibration_preregistration_sha256",
            "calibration_execution_binding_sha256",
            "calibration_holdout_report_sha256",
            "executable_sha256",
        ):
            object.__setattr__(
                self,
                name,
                _sha256(getattr(self, name), f"render-settled {name}"),
            )
        for name in ("process_id", "process_creation_filetime_100ns"):
            object.__setattr__(
                self,
                name,
                _strict_int(
                    getattr(self, name),
                    f"render-settled {name}",
                    minimum=1,
                ),
            )
        delay = _strict_int(
            self.render_settle_delay_ns,
            "render-settled delay",
            minimum=1,
        )
        if delay != RENDER_SETTLE_DELAY_NS:
            raise PcGoldenValidationError(
                "render-settled delay differs from the calibrated contract"
            )
        object.__setattr__(self, "render_settle_delay_ns", delay)
        minimum_hz = _finite_number(
            self.minimum_mean_poll_hz,
            "render-settled minimum poll rate",
        )
        if minimum_hz != MINIMUM_MEAN_POLL_HZ:
            raise PcGoldenValidationError(
                "render-settled minimum poll rate differs from calibration"
            )
        object.__setattr__(self, "minimum_mean_poll_hz", minimum_hz)
        maximum_p99 = _strict_int(
            self.maximum_p99_poll_interval_ns,
            "render-settled maximum p99 poll interval",
            minimum=1,
        )
        if maximum_p99 != MAXIMUM_P99_POLL_INTERVAL_NS:
            raise PcGoldenValidationError(
                "render-settled p99 threshold differs from calibration"
            )
        object.__setattr__(
            self,
            "maximum_p99_poll_interval_ns",
            maximum_p99,
        )
        records = tuple(self.records)
        if not records or len(records) > _MAX_RECORDS:
            raise PcGoldenValidationError(
                "render-settled map record count is invalid"
            )
        if any(
            not isinstance(row, RenderSettledUpdateRecord) for row in records
        ):
            raise PcGoldenValidationError(
                "render-settled map contains an invalid record"
            )
        previous_present = -1
        previous_assigned_update = -1
        previous_draw = -1
        for expected_sequence, row in enumerate(records):
            if row.sequence != expected_sequence:
                raise PcGoldenValidationError(
                    "render-settled sequences must be contiguous from zero"
                )
            if row.present_ticks <= previous_present:
                raise PcGoldenValidationError(
                    "render-settled presents must be strictly increasing"
                )
            if row.assigned_framework_update < previous_assigned_update:
                raise PcGoldenValidationError(
                    "render-settled updates must be non-decreasing"
                )
            if row.assigned_draw_count < previous_draw:
                raise PcGoldenValidationError(
                    "render-settled draw counts must be non-decreasing"
                )
            if row.settle_age_ns < delay:
                raise PcGoldenValidationError(
                    "render-settled record violates the fixed delay"
                )
            previous_present = row.present_ticks
            previous_assigned_update = row.assigned_framework_update
            previous_draw = row.assigned_draw_count
        object.__setattr__(self, "records", records)

    @property
    def process_instance(self) -> tuple[int, int]:
        return self.process_id, self.process_creation_filetime_100ns

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": RENDER_SETTLED_UPDATE_SCHEMA,
            "version": RENDER_SETTLED_UPDATE_VERSION,
            "method": RENDER_SETTLE_METHOD,
            "render_settle_delay_ns": self.render_settle_delay_ns,
            "minimum_mean_poll_hz": self.minimum_mean_poll_hz,
            "maximum_p99_poll_interval_ns": (
                self.maximum_p99_poll_interval_ns
            ),
            "capture_metadata_sha256": self.capture_metadata_sha256,
            "frames_csv_sha256": self.frames_csv_sha256,
            "framework_update_map_sha256": (
                self.framework_update_map_sha256
            ),
            "framework_state_sidecar_sha256": (
                self.framework_state_sidecar_sha256
            ),
            "framework_poll_sha256": self.framework_poll_sha256,
            "calibration_preregistration_sha256": (
                self.calibration_preregistration_sha256
            ),
            "calibration_execution_binding_sha256": (
                self.calibration_execution_binding_sha256
            ),
            "calibration_holdout_report_sha256": (
                self.calibration_holdout_report_sha256
            ),
            "process_id": self.process_id,
            "process_creation_filetime_100ns": (
                self.process_creation_filetime_100ns
            ),
            "executable_sha256": self.executable_sha256,
            "records": [row.to_dict() for row in self.records],
        }

    def to_json(self) -> str:
        return _canonical_json_file(self.to_dict())

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> "PcRenderSettledUpdateMap":
        data = _mapping(value, "render-settled update map")
        fields = {
            "schema",
            "version",
            "method",
            "render_settle_delay_ns",
            "minimum_mean_poll_hz",
            "maximum_p99_poll_interval_ns",
            "capture_metadata_sha256",
            "frames_csv_sha256",
            "framework_update_map_sha256",
            "framework_state_sidecar_sha256",
            "framework_poll_sha256",
            "calibration_preregistration_sha256",
            "calibration_execution_binding_sha256",
            "calibration_holdout_report_sha256",
            "process_id",
            "process_creation_filetime_100ns",
            "executable_sha256",
            "records",
        }
        _check_keys(data, required=fields, name="render-settled update map")
        if (
            data["schema"] != RENDER_SETTLED_UPDATE_SCHEMA
            or _strict_int(data["version"], "render-settled map version")
            != RENDER_SETTLED_UPDATE_VERSION
            or data["method"] != RENDER_SETTLE_METHOD
        ):
            raise PcGoldenValidationError(
                "unsupported render-settled update map schema/version/method"
            )
        return cls(
            capture_metadata_sha256=data["capture_metadata_sha256"],
            frames_csv_sha256=data["frames_csv_sha256"],
            framework_update_map_sha256=data[
                "framework_update_map_sha256"
            ],
            framework_state_sidecar_sha256=data[
                "framework_state_sidecar_sha256"
            ],
            framework_poll_sha256=data["framework_poll_sha256"],
            calibration_preregistration_sha256=data[
                "calibration_preregistration_sha256"
            ],
            calibration_execution_binding_sha256=data[
                "calibration_execution_binding_sha256"
            ],
            calibration_holdout_report_sha256=data[
                "calibration_holdout_report_sha256"
            ],
            process_id=data["process_id"],
            process_creation_filetime_100ns=data[
                "process_creation_filetime_100ns"
            ],
            executable_sha256=data["executable_sha256"],
            records=tuple(
                RenderSettledUpdateRecord.from_dict(
                    _mapping(row, "render-settled update record")
                )
                for row in _sequence(data["records"], "render-settled records")
            ),
            render_settle_delay_ns=data["render_settle_delay_ns"],
            minimum_mean_poll_hz=data["minimum_mean_poll_hz"],
            maximum_p99_poll_interval_ns=data[
                "maximum_p99_poll_interval_ns"
            ],
        )

    @classmethod
    def from_json(cls, text: str) -> "PcRenderSettledUpdateMap":
        if not isinstance(text, str):
            raise TypeError("render-settled update map text must be a string")
        result = cls.from_dict(
            _mapping(
                _json_loads_strict(text, "render-settled update map"),
                "render-settled update map",
            )
        )
        if text != result.to_json():
            raise PcGoldenValidationError(
                "render-settled update map JSON is not canonical"
            )
        return result

    @classmethod
    def read(cls, path: str | Path) -> "PcRenderSettledUpdateMap":
        try:
            text = Path(path).read_text(encoding="ascii")
        except (OSError, UnicodeError) as error:
            raise PcGoldenValidationError(
                "render-settled update map could not be read as ASCII"
            ) from error
        return cls.from_json(text)


@dataclass(frozen=True, slots=True)
class _DrawEvent:
    draw_count: int
    update_count: int
    sample_after_perf_counter_ns: int


def _validate_calibration(
    preregistration_path: str | Path,
    execution_binding_path: str | Path,
    holdout_report_path: str | Path,
) -> tuple[str, str, str]:
    prereg, _, prereg_sha = _read_json_object(
        preregistration_path,
        "render-settle calibration preregistration",
    )
    binding, _, binding_sha = _read_json_object(
        execution_binding_path,
        "render-settle calibration execution binding",
    )
    report, _, report_sha = _read_json_object(
        holdout_report_path,
        "render-settle calibration holdout report",
    )
    comparison = _mapping(
        prereg.get("comparison"),
        "render-settle calibration comparison",
    )
    acceptance = _mapping(
        prereg.get("acceptance"),
        "render-settle calibration acceptance",
    )
    if (
        prereg.get("schema") != _CALIBRATION_PREREGISTRATION_SCHEMA
        or prereg.get("version") != 1
        or prereg.get("status") != "FROZEN_BEFORE_HOLDOUT_COLLECTION"
        or comparison.get("render_settle_delay_ns")
        != RENDER_SETTLE_DELAY_NS
        or comparison.get("draw_event_sample")
        != "first_present_after_fixed_settle_delay"
        or acceptance.get("minimum_mean_poll_hz_each_run")
        != MINIMUM_MEAN_POLL_HZ
        or acceptance.get("maximum_p99_poll_interval_ns_each_run")
        != MAXIMUM_P99_POLL_INTERVAL_NS
        or acceptance.get("maximum_skipped_draw_increments_each_run") != 0
        or acceptance.get("require_disjoint_physical_cores") is not True
    ):
        raise PcGoldenValidationError(
            "render-settle calibration preregistration is incompatible"
        )
    parent = _mapping(
        binding.get("parent"),
        "render-settle calibration execution parent",
    )
    if (
        binding.get("schema") != _CALIBRATION_EXECUTION_BINDING_SCHEMA
        or binding.get("version") != 1
        or binding.get("status")
        != "FROZEN_RETRY_AFTER_INFRASTRUCTURE_INVALIDATION"
        or parent.get("sha256") != prereg_sha
    ):
        raise PcGoldenValidationError(
            "render-settle calibration execution binding is incompatible"
        )
    report_prereg = _mapping(
        report.get("preregistration"),
        "render-settle calibration report preregistration",
    )
    chain = _sequence(
        report_prereg.get("chain"),
        "render-settle calibration report chain",
    )
    chain_hashes = {
        row.get("sha256")
        for row in chain
        if isinstance(row, Mapping)
    }
    report_comparison = _mapping(
        report.get("comparison"),
        "render-settle calibration report comparison",
    )
    criteria = _sequence(
        report.get("criteria"),
        "render-settle calibration criteria",
    )
    if (
        report.get("schema") != _CALIBRATION_REPORT_SCHEMA
        or report.get("version") != 1
        or report.get("classification")
        != "independent-diagnostic-not-pc-golden"
        or report.get("gate_effect") != "none"
        or report.get("status") != "PASS"
        or report.get("reasons") != []
        or report_prereg.get("sha256") != binding_sha
        or prereg_sha not in chain_hashes
        or binding_sha not in chain_hashes
        or report_comparison.get("render_settle_delay_ns")
        != RENDER_SETTLE_DELAY_NS
        or not criteria
        or any(
            not isinstance(row, Mapping) or row.get("status") != "PASS"
            for row in criteria
        )
    ):
        raise PcGoldenValidationError(
            "render-settle calibration holdout did not pass its frozen plan"
        )
    return prereg_sha, binding_sha, report_sha


def _read_frame_csv(
    path: str | Path,
    *,
    metadata: PcDxgiCaptureMetadata,
    update_map: PcFrameworkUpdateMap,
) -> str:
    payload = _read_bytes(path, "render-settle frame CSV")
    digest = _sha256_bytes(payload)
    if digest != metadata.frames_csv_sha256:
        raise PcGoldenValidationError(
            "render-settle frame CSV differs from DXGI metadata"
        )
    try:
        text = payload.decode("ascii")
        reader = csv.DictReader(text.splitlines())
        if tuple(reader.fieldnames or ()) != _FRAME_CSV_FIELDS:
            raise PcGoldenValidationError(
                "render-settle frame CSV header differs from v2"
            )
        rows = list(reader)
    except (UnicodeError, csv.Error) as error:
        raise PcGoldenValidationError(
            "render-settle frame CSV could not be parsed"
        ) from error
    if len(rows) != metadata.frame_count or len(rows) != len(update_map.records):
        raise PcGoldenValidationError(
            "render-settle frame source lengths differ"
        )
    frame_bytes = metadata.width * metadata.height * 4
    for sequence, (row, update) in enumerate(
        zip(rows, update_map.records, strict=True)
    ):
        try:
            parsed = {
                name: int(row[name])
                for name in _FRAME_CSV_FIELDS[:-1]
            }
        except (KeyError, TypeError, ValueError) as error:
            raise PcGoldenValidationError(
                "render-settle frame CSV contains an invalid integer"
            ) from error
        frame_sha = row.get("frame_sha256")
        _sha256(frame_sha, "render-settle frame hash")
        if (
            parsed["sequence"] != sequence
            or parsed["present_ticks"] != update.present_ticks
            or parsed["qpc_frequency"] != metadata.qpc_frequency
            or parsed["host_perf_counter_ns"]
            != update.host_perf_counter_ns
            or parsed["accumulated_frames"] != 1
            or parsed["raw_offset"] != sequence * frame_bytes
            or parsed["raw_bytes"] != frame_bytes
        ):
            raise PcGoldenValidationError(
                "render-settle frame CSV identity differs from source map"
            )
    return digest


def _read_poll_events(
    path: str | Path,
    *,
    metadata: PcDxgiCaptureMetadata,
    metadata_sha256: str,
    framework_update_sha256: str,
    framework_state_sha256: str,
) -> tuple[tuple[_DrawEvent, ...], str]:
    poll, _, poll_sha = _read_json_object(
        path,
        "render-settle framework poll",
        require_canonical=True,
    )
    identity = _mapping(
        poll.get("process_identity"),
        "render-settle poll identity",
    )
    observer = _mapping(
        poll.get("observer"),
        "render-settle poll observer",
    )
    binding = _mapping(
        poll.get("capture_binding"),
        "render-settle poll capture binding",
    )
    intervals = _mapping(
        observer.get("sample_completion_interval_ns"),
        "render-settle poll intervals",
    )
    topology = _mapping(
        observer.get("affinity_topology"),
        "render-settle poll topology",
    )
    mean_hz = _finite_number(
        observer.get("mean_polls_per_second"),
        "render-settle observed poll rate",
    )
    p99 = _strict_int(
        intervals.get("p99"),
        "render-settle observed p99 interval",
        minimum=1,
    )
    poll_start = _strict_int(
        observer.get("poll_start_perf_counter_ns"),
        "render-settle poll start",
        minimum=1,
    )
    poll_end = _strict_int(
        observer.get("poll_end_perf_counter_ns"),
        "render-settle poll end",
        minimum=1,
    )
    if (
        poll.get("schema") != _POLL_SCHEMA
        or poll.get("version") != 1
        or poll.get("status") != "complete"
        or poll.get("skipped_draw_increment_count") != 0
        or identity.get("process_id") != metadata.process_id
        or identity.get("process_creation_filetime_100ns")
        != metadata.process_creation_filetime_100ns
        or identity.get("executable_sha256") != metadata.executable_sha256
        or binding.get("capture_metadata_sha256") != metadata_sha256
        or binding.get("framework_update_map_sha256")
        != framework_update_sha256
        or binding.get("framework_state_sidecar_sha256")
        != framework_state_sha256
        or binding.get("capture_start_perf_counter_ns")
        != metadata.capture_start_perf_counter_ns
        or binding.get("capture_end_perf_counter_ns")
        != metadata.capture_end_perf_counter_ns
        or binding.get("capture_interval_fully_observed") is not True
        or mean_hz < MINIMUM_MEAN_POLL_HZ
        or p99 > MAXIMUM_P99_POLL_INTERVAL_NS
        or topology.get("physical_cores_disjoint") is not True
        or topology.get("observer_physical_core_logical_mask")
        == topology.get("game_physical_core_logical_mask")
        or not (
            poll_start
            <= metadata.capture_start_perf_counter_ns
            < metadata.capture_end_perf_counter_ns
            <= poll_end
        )
    ):
        raise PcGoldenValidationError(
            "render-settle framework poll violates the calibrated contract"
        )
    event_values = _sequence(
        poll.get("draw_events"),
        "render-settle draw events",
    )
    declared_count = _strict_int(
        poll.get("draw_event_count"),
        "render-settle draw event count",
        minimum=1,
    )
    if not event_values or len(event_values) != declared_count:
        raise PcGoldenValidationError(
            "render-settle draw event count differs"
        )
    events: list[_DrawEvent] = []
    previous_draw: int | None = None
    previous_update = -1
    previous_time = -1
    for event_index, value in enumerate(event_values):
        row = _mapping(value, "render-settle draw event")
        draw = _strict_int(
            row.get("draw_count"),
            "render-settle draw count",
            minimum=0,
        )
        update = _strict_int(
            row.get("update_count"),
            "render-settle draw update",
            minimum=0,
        )
        timestamp = _strict_int(
            row.get("sample_after_perf_counter_ns"),
            "render-settle draw timestamp",
            minimum=1,
        )
        if (
            row.get("event_index") != event_index
            or row.get("draw_count_delta") != 1
            or (previous_draw is not None and draw != previous_draw + 1)
            or update < previous_update
            or timestamp <= previous_time
            or not poll_start <= timestamp <= poll_end
        ):
            raise PcGoldenValidationError(
                "render-settle draw event sequence is invalid"
            )
        events.append(_DrawEvent(draw, update, timestamp))
        previous_draw = draw
        previous_update = update
        previous_time = timestamp
    return tuple(events), poll_sha


def recompute_render_settled_update_map(
    *,
    capture_metadata_path: str | Path,
    frames_csv_path: str | Path,
    framework_update_map_path: str | Path,
    framework_state_sidecar_path: str | Path,
    framework_poll_path: str | Path,
    calibration_preregistration_path: str | Path,
    calibration_execution_binding_path: str | Path,
    calibration_holdout_report_path: str | Path,
) -> PcRenderSettledUpdateMap:
    """Recompute a settled map from a complete, immutable source set."""

    metadata = PcDxgiCaptureMetadata.read(capture_metadata_path)
    metadata_sha = metadata.sha256
    framework_updates = PcFrameworkUpdateMap.read(framework_update_map_path)
    framework_update_payload = _read_bytes(
        framework_update_map_path,
        "render-settle framework update map",
    )
    framework_update_sha = _sha256_bytes(framework_update_payload)
    framework_state_sha = _sha256_bytes(
        _read_bytes(
            framework_state_sidecar_path,
            "render-settle framework state sidecar",
        )
    )
    if (
        framework_updates.capture_metadata_sha256 != metadata_sha
        or framework_updates.process_instance != metadata.process_instance
        or framework_updates.executable_sha256 != metadata.executable_sha256
        or len(framework_updates.records) != metadata.frame_count
    ):
        raise PcGoldenValidationError(
            "render-settle source update identity differs from DXGI metadata"
        )
    frames_csv_sha = _read_frame_csv(
        frames_csv_path,
        metadata=metadata,
        update_map=framework_updates,
    )
    calibration_hashes = _validate_calibration(
        calibration_preregistration_path,
        calibration_execution_binding_path,
        calibration_holdout_report_path,
    )
    events, poll_sha = _read_poll_events(
        framework_poll_path,
        metadata=metadata,
        metadata_sha256=metadata_sha,
        framework_update_sha256=framework_update_sha,
        framework_state_sha256=framework_state_sha,
    )
    timestamps = [row.sample_after_perf_counter_ns for row in events]
    records: list[RenderSettledUpdateRecord] = []
    for source in framework_updates.records:
        present_ns = (
            source.present_ticks * 1_000_000_000
            + metadata.qpc_frequency // 2
        ) // metadata.qpc_frequency
        event_index = (
            bisect_right(
                timestamps,
                present_ns - RENDER_SETTLE_DELAY_NS,
            )
            - 1
        )
        if event_index < 0:
            raise PcGoldenValidationError(
                "render-settle frame precedes every eligible draw event"
            )
        event = events[event_index]
        records.append(
            RenderSettledUpdateRecord(
                sequence=source.sequence,
                present_ticks=source.present_ticks,
                present_perf_counter_ns=present_ns,
                source_update_before=source.update_before,
                source_update_after=source.update_after,
                assigned_draw_count=event.draw_count,
                assigned_framework_update=event.update_count,
                draw_event_sample_after_perf_counter_ns=(
                    event.sample_after_perf_counter_ns
                ),
                settle_age_ns=(
                    present_ns - event.sample_after_perf_counter_ns
                ),
            )
        )
    return PcRenderSettledUpdateMap(
        capture_metadata_sha256=metadata_sha,
        frames_csv_sha256=frames_csv_sha,
        framework_update_map_sha256=framework_update_sha,
        framework_state_sidecar_sha256=framework_state_sha,
        framework_poll_sha256=poll_sha,
        calibration_preregistration_sha256=calibration_hashes[0],
        calibration_execution_binding_sha256=calibration_hashes[1],
        calibration_holdout_report_sha256=calibration_hashes[2],
        process_id=metadata.process_id,
        process_creation_filetime_100ns=(
            metadata.process_creation_filetime_100ns
        ),
        executable_sha256=metadata.executable_sha256,
        records=tuple(records),
    )


def verify_render_settled_update_map(
    expected: PcRenderSettledUpdateMap,
    **source_paths: str | Path,
) -> PcRenderSettledUpdateMap:
    """Recompute and byte-compare a producer-authored settled map."""

    observed = recompute_render_settled_update_map(**source_paths)
    if observed.to_json() != expected.to_json():
        raise PcGoldenValidationError(
            "render-settled update map differs from source recomputation"
        )
    return observed


__all__ = [
    "MAXIMUM_P99_POLL_INTERVAL_NS",
    "MINIMUM_MEAN_POLL_HZ",
    "PcRenderSettledUpdateMap",
    "RENDER_SETTLED_UPDATE_SCHEMA",
    "RENDER_SETTLED_UPDATE_VERSION",
    "RENDER_SETTLE_DELAY_NS",
    "RENDER_SETTLE_METHOD",
    "RenderSettledUpdateRecord",
    "recompute_render_settled_update_map",
    "verify_render_settled_update_map",
]
