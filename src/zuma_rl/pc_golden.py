"""Strict, partial-observation PC golden-capture infrastructure.

This module deliberately does not compare a capture with
``RevengeSimulator`` yet.  It establishes two independent, content-addressed
formats first:

``zuma-rl.pc-golden-manifest`` v4/v5
    The immutable game/capture environment, artifact inventory, clock and
    coordinate calibration, evidence coverage, frozen comparison limits, and
    fail-closed save/replay evidence contracts.  Version 5 adds a separate
    exact-step external-frame protocol without relabeling continuous DXGI
    metadata.  The reader retains explicit v3 compatibility so old captures
    remain inspectable but cannot satisfy current protocol gates.

``zuma-rl.pc-golden-trace`` v2
    Canonical newline-delimited JSON with one baseline record at tick zero and
    one contiguous record for every subsequent native 100 Hz tick.

PC video is partial evidence, not a complete simulator state.  Missing values
therefore use explicit :class:`MeasurementStatus` values and are never filled
with zero or silently treated as matching.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from zuma_rl.revenge_core import SUPPORTED_PROFILE_MODE

MANIFEST_SCHEMA = "zuma-rl.pc-golden-manifest"
LEGACY_MANIFEST_VERSION = 3
MANIFEST_VERSION = 4
EXACT_STEP_MANIFEST_VERSION = 5
TRACE_SCHEMA = "zuma-rl.pc-golden-trace"
TRACE_VERSION = 2
CAPTURE_CONTRACT_SCHEMA = "zuma-rl.pc-golden-capture-contract"
CAPTURE_CONTRACT_VERSION = 2
EVIDENCE_SET_SCHEMA = "zuma-rl.pc-golden-evidence-set"
EVIDENCE_SET_VERSION = 1
EXACT_STEP_CAPTURE_CONTRACT_SCHEMA = (
    "zuma-rl.pc-golden-exact-step-capture-contract"
)
EXACT_STEP_CAPTURE_CONTRACT_VERSION = 1
DMO_FORMAT = "popcap_dmo_v2"
DMO_FILE_ID = 0x42BEEF78
DMO_VERSION = 2

MEASUREMENT_CHANNELS = frozenset(
    {
        "score",
        "current_color",
        "next_color",
        "gun_state",
        "outcome",
        "zuma_reached",
        "chain_count",
        "projectile_count",
        "chain_centers",
        "projectile_centers",
        "chain_waypoints",
    }
)
STRUCTURAL_TRACE_CHANNELS = frozenset(
    {"events", "input_events", "frames"}
)
SUPPORTED_COVERAGE_CHANNELS = (
    MEASUREMENT_CHANNELS | STRUCTURAL_TRACE_CHANNELS
)
EVENT_KINDS = frozenset(
    {
        "shot_fired",
        "hit",
        "inserted",
        "match_started",
        "balls_exploded",
        "balls_removed",
        "zuma_triggered",
        "outcome",
    }
)
TRUSTED_PC_EVIDENCE_SOURCES = frozenset(
    {
        "pc_memory_probe",
        "video_ocr",
        "video_tracker",
        "frame_annotation",
        "internal_screenshot",
    }
)

_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_LOGICAL_WIDTH = 800
_LOGICAL_HEIGHT = 600
_SAMPLE_PHASE = "post_update_presented"
_JSON_MAX_DEPTH = 128
_MAX_CALIBRATION_RMS_ERROR_PX = 0.25
_MAX_CALIBRATION_ERROR_PX = 0.5


class PcGoldenValidationError(ValueError):
    """A manifest, trace, or public value violates the current schema."""


class PcGoldenArtifactError(RuntimeError):
    """An artifact is missing, escapes its root, or fails size/hash checks."""


class MeasurementStatus(str, Enum):
    """Why a trace channel does or does not contain a value."""

    OBSERVED = "observed"
    INFERRED = "inferred"
    OCCLUDED = "occluded"
    NOT_CAPTURED = "not_captured"
    NOT_APPLICABLE = "not_applicable"


class CoverageStatus(str, Enum):
    """Completeness of one named channel over a declared tick interval."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    ABSENT = "absent"


class ComparisonStatus(str, Enum):
    """Top-level outcome used by future differential comparison."""

    PASS = "PASS"
    FAIL = "FAIL"
    INCOMPARABLE = "INCOMPARABLE"


def _strict_int(value: Any, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise PcGoldenValidationError(f"{name} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise PcGoldenValidationError(f"{name} must be at least {minimum}")
    return result


def _strict_bool(value: Any, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise PcGoldenValidationError(f"{name} must be a boolean")
    return bool(value)


def _finite_float(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise PcGoldenValidationError(f"{name} must be numeric")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise PcGoldenValidationError(f"{name} must be finite") from error
    if not math.isfinite(result):
        raise PcGoldenValidationError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise PcGoldenValidationError(f"{name} must be at least {minimum}")
    return result


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PcGoldenValidationError(f"{name} must be a non-empty string")
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PcGoldenValidationError(f"{name} must be a JSON object")
    if any(not isinstance(key, str) for key in value):
        raise PcGoldenValidationError(f"{name} keys must be strings")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PcGoldenValidationError(f"{name} must be a JSON array")
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
        raise PcGoldenValidationError(
            f"{name} is missing fields: {', '.join(sorted(missing))}"
        )
    unknown = set(value).difference(required, optional)
    if unknown:
        raise PcGoldenValidationError(
            f"{name} has unknown fields: {', '.join(sorted(unknown))}"
        )


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise PcGoldenValidationError(
            f"{name} must be 'sha256:' plus 64 lowercase hex digits"
        )
    return value


def _enum_value(value: Any, enum_type: type[Enum], name: str) -> Enum:
    if isinstance(value, enum_type):
        return value
    if not isinstance(value, str):
        raise PcGoldenValidationError(f"{name} must be a string")
    try:
        return enum_type(value)
    except ValueError as error:
        allowed = ", ".join(member.value for member in enum_type)
        raise PcGoldenValidationError(
            f"{name} must be one of: {allowed}"
        ) from error


def _json_value(
    value: Any,
    name: str,
    *,
    allow_none: bool = True,
    _depth: int = 0,
) -> Any:
    """Return a detached JSON value while rejecting non-finite numbers."""

    if _depth > _JSON_MAX_DEPTH:
        raise PcGoldenValidationError(
            f"{name} exceeds the maximum JSON nesting depth"
        )
    if isinstance(value, np.ndarray):
        value = value.tolist()
    elif isinstance(value, np.generic):
        value = value.item()

    if value is None:
        if not allow_none:
            raise PcGoldenValidationError(f"{name} must not be null")
        return None
    if isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PcGoldenValidationError(f"{name} contains NaN or infinity")
        return float(value)
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise PcGoldenValidationError(f"{name} keys must be strings")
        return {
            key: _json_value(
                item,
                f"{name}.{key}",
                _depth=_depth + 1,
            )
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [
            _json_value(
                item,
                f"{name}[{index}]",
                _depth=_depth + 1,
            )
            for index, item in enumerate(value)
        ]
    raise PcGoldenValidationError(
        f"{name} contains unsupported type {type(value).__name__}"
    )


def _freeze_json_value(value: Any) -> Any:
    """Recursively freeze an already validated detached JSON value."""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                key: _freeze_json_value(item)
                for key, item in value.items()
            }
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(_freeze_json_value(item) for item in value)
    return value


def _plain_json_value(value: Any) -> Any:
    """Return ordinary detached dict/list containers for serialization."""

    if isinstance(value, Mapping):
        return {
            key: _plain_json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_plain_json_value(item) for item in value]
    return value


def _frozen_json_mapping(value: Any, name: str) -> Mapping[str, Any]:
    detached = _json_value(_mapping(value, name), name)
    frozen = _freeze_json_value(detached)
    if not isinstance(frozen, Mapping):  # Defensive: _mapping guaranteed this.
        raise AssertionError("frozen JSON mapping lost its object type")
    return frozen


def _canonical_json_text(value: Any) -> str:
    detached = _json_value(value, "canonical value")
    return json.dumps(
        detached,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_sha256(value: Any) -> str:
    """Hash the package's deterministic UTF-8 canonical JSON representation."""

    payload = _canonical_json_text(value).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise PcGoldenValidationError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_nonfinite_constant(value: str) -> None:
    raise PcGoldenValidationError(f"JSON contains forbidden constant {value}")


def _json_loads_strict(text: str, name: str) -> Any:
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except PcGoldenValidationError:
        raise
    except RecursionError as error:
        raise PcGoldenValidationError(
            f"{name} JSON nesting is too deep"
        ) from error
    except (TypeError, ValueError) as error:
        raise PcGoldenValidationError(f"invalid {name} JSON: {error}") from error


def _digest_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            hasher.update(chunk)
    return f"sha256:{hasher.hexdigest()}"


@dataclass(frozen=True, slots=True)
class Measurement:
    """A value plus explicit observability/provenance semantics."""

    status: MeasurementStatus
    value: Any = None
    source: str | None = None
    uncertainty: float | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        status = _enum_value(
            self.status,
            MeasurementStatus,
            "measurement.status",
        )
        object.__setattr__(self, "status", status)

        has_value = status in {
            MeasurementStatus.OBSERVED,
            MeasurementStatus.INFERRED,
        }
        if has_value:
            detached_value = _json_value(
                self.value,
                "measurement.value",
                allow_none=True,
            )
            object.__setattr__(
                self,
                "value",
                _freeze_json_value(detached_value),
            )
            source = _nonempty_string(
                self.source,
                "measurement.source",
            )
            object.__setattr__(self, "source", source)
            if self.uncertainty is not None:
                object.__setattr__(
                    self,
                    "uncertainty",
                    _finite_float(
                        self.uncertainty,
                        "measurement.uncertainty",
                        minimum=0.0,
                    ),
                )
            if status is MeasurementStatus.INFERRED:
                object.__setattr__(
                    self,
                    "reason",
                    _nonempty_string(
                        self.reason,
                        "inferred measurement.reason",
                    ),
                )
            elif self.reason is not None:
                object.__setattr__(
                    self,
                    "reason",
                    _nonempty_string(
                        self.reason,
                        "measurement.reason",
                    ),
                )
            return

        if self.value is not None:
            raise PcGoldenValidationError(
                f"{status.value} measurements must not contain a value"
            )
        if self.uncertainty is not None:
            raise PcGoldenValidationError(
                f"{status.value} measurements must not contain uncertainty"
            )
        object.__setattr__(
            self,
            "reason",
            _nonempty_string(
                self.reason,
                f"{status.value} measurement.reason",
            ),
        )
        if self.source is not None:
            object.__setattr__(
                self,
                "source",
                _nonempty_string(
                    self.source,
                    "measurement.source",
                ),
            )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"status": self.status.value}
        if self.status in {
            MeasurementStatus.OBSERVED,
            MeasurementStatus.INFERRED,
        }:
            result["value"] = _plain_json_value(self.value)
            result["source"] = self.source
            if self.uncertainty is not None:
                result["uncertainty"] = self.uncertainty
        elif self.source is not None:
            result["source"] = self.source
        if self.reason is not None:
            result["reason"] = self.reason
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Measurement":
        data = _mapping(value, "measurement")
        _check_keys(
            data,
            required={"status"},
            optional={"value", "source", "uncertainty", "reason"},
            name="measurement",
        )
        status = _enum_value(
            data["status"],
            MeasurementStatus,
            "measurement.status",
        )
        if (
            status
            in {MeasurementStatus.OBSERVED, MeasurementStatus.INFERRED}
            and "value" not in data
        ):
            raise PcGoldenValidationError(
                f"{status.value} measurements must contain a value field"
            )
        if status in {
            MeasurementStatus.OCCLUDED,
            MeasurementStatus.NOT_CAPTURED,
            MeasurementStatus.NOT_APPLICABLE,
        }:
            forbidden = {"value", "uncertainty"}.intersection(data)
            if forbidden:
                labels = [
                    "a value" if name == "value" else "uncertainty"
                    for name in sorted(forbidden)
                ]
                raise PcGoldenValidationError(
                    f"{status.value} measurements must not contain "
                    + " or ".join(labels)
                )
        return cls(
            status=status,
            value=data.get("value"),
            source=data.get("source"),
            uncertainty=data.get("uncertainty"),
            reason=data.get("reason"),
        )


def validate_measurement_value(channel: str, value: Any) -> None:
    """Validate the typed value contract for a known v3 trace channel."""

    if channel in {"score", "chain_count", "projectile_count"}:
        _strict_int(value, f"measurement {channel!r}", minimum=0)
        return
    if channel in {"current_color", "next_color"}:
        if value is not None:
            _strict_int(value, f"measurement {channel!r}", minimum=0)
        return
    if channel == "gun_state":
        if value not in {"normal", "firing", "reloading"}:
            raise PcGoldenValidationError(
                "measurement 'gun_state' must be normal, firing, or "
                "reloading"
            )
        return
    if channel == "outcome":
        if value not in {None, "win", "loss"}:
            raise PcGoldenValidationError(
                "measurement 'outcome' must be null, win, or loss"
            )
        return
    if channel == "zuma_reached":
        _strict_bool(value, "measurement 'zuma_reached'")
        return
    if channel in {"chain_centers", "projectile_centers"}:
        points = _sequence(value, f"measurement {channel!r}")
        for index, point in enumerate(points):
            coordinates = _sequence(
                point,
                f"measurement {channel!r}[{index}]",
            )
            if len(coordinates) != 2:
                raise PcGoldenValidationError(
                    f"measurement {channel!r}[{index}] must have two "
                    "coordinates"
                )
            for axis, coordinate in enumerate(coordinates):
                _finite_float(
                    coordinate,
                    f"measurement {channel!r}[{index}][{axis}]",
                )
        return
    if channel == "chain_waypoints":
        values = _sequence(value, "measurement 'chain_waypoints'")
        for index, waypoint in enumerate(values):
            _finite_float(
                waypoint,
                f"measurement 'chain_waypoints'[{index}]",
            )


@dataclass(frozen=True, slots=True)
class ArtifactSpec:
    """One content-addressed file stored relative to a case root."""

    path: str
    sha256: str
    bytes: int

    def __post_init__(self) -> None:
        path = _nonempty_string(self.path, "artifact.path")
        pure = PurePosixPath(path)
        if (
            pure.is_absolute()
            or ".." in pure.parts
            or "\\" in path
            or str(pure) != path
        ):
            raise PcGoldenValidationError(
                "artifact.path must be a normalized relative POSIX path"
            )
        object.__setattr__(self, "path", path)
        object.__setattr__(
            self,
            "sha256",
            _sha256(self.sha256, "artifact.sha256"),
        )
        object.__setattr__(
            self,
            "bytes",
            _strict_int(self.bytes, "artifact.bytes", minimum=0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "bytes": self.bytes,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ArtifactSpec":
        data = _mapping(value, "artifact")
        _check_keys(
            data,
            required={"path", "sha256", "bytes"},
            name="artifact",
        )
        return cls(
            path=data["path"],
            sha256=data["sha256"],
            bytes=data["bytes"],
        )

    def verify(self, root: str | Path) -> Path:
        base = Path(root).resolve()
        target = (base / self.path).resolve()
        if not target.is_relative_to(base):
            raise PcGoldenArtifactError(
                f"artifact escapes case root: {self.path}"
            )
        if not target.is_file():
            raise PcGoldenArtifactError(f"artifact is missing: {self.path}")
        actual_size = target.stat().st_size
        if actual_size != self.bytes:
            raise PcGoldenArtifactError(
                f"artifact size differs for {self.path}: "
                f"expected {self.bytes}, got {actual_size}"
            )
        actual_digest = _digest_file(target)
        if actual_digest != self.sha256:
            raise PcGoldenArtifactError(
                f"artifact SHA-256 differs for {self.path}: "
                f"expected {self.sha256}, got {actual_digest}"
            )
        return target


@dataclass(frozen=True, slots=True)
class Scenario:
    """The exact original-game mode represented by a capture."""

    level_id: str
    hard: bool
    curve_index: int
    gun_index: int
    mode: str
    profile_mode: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "level_id",
            _nonempty_string(self.level_id, "scenario.level_id"),
        )
        object.__setattr__(
            self,
            "hard",
            _strict_bool(self.hard, "scenario.hard"),
        )
        object.__setattr__(
            self,
            "curve_index",
            _strict_int(
                self.curve_index,
                "scenario.curve_index",
                minimum=0,
            ),
        )
        object.__setattr__(
            self,
            "gun_index",
            _strict_int(self.gun_index, "scenario.gun_index", minimum=0),
        )
        object.__setattr__(
            self,
            "mode",
            _nonempty_string(self.mode, "scenario.mode"),
        )
        object.__setattr__(
            self,
            "profile_mode",
            _nonempty_string(
                self.profile_mode,
                "scenario.profile_mode",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "level_id": self.level_id,
            "hard": self.hard,
            "curve_index": self.curve_index,
            "gun_index": self.gun_index,
            "mode": self.mode,
            "profile_mode": self.profile_mode,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Scenario":
        data = _mapping(value, "scenario")
        fields = {
            "level_id",
            "hard",
            "curve_index",
            "gun_index",
            "mode",
            "profile_mode",
        }
        _check_keys(data, required=fields, name="scenario")
        return cls(**{name: data[name] for name in fields})


@dataclass(frozen=True, slots=True)
class PcEnvironment:
    """Static PC state whose canonical hash identifies comparable captures."""

    executable_sha256: str
    main_pak_sha256: str
    levels_xml_sha256: str
    curve_sha256: Mapping[str, str]
    pre_capture_save_sha256: str
    profile_mode: str
    renderer_api: str
    renderer_mode: str
    ball_radius_branch: int
    os_build: str
    gpu: str
    driver: str
    game_settings: Mapping[str, Any] = field(default_factory=dict)
    dynamic_difficulty_state: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "executable_sha256",
            "main_pak_sha256",
            "levels_xml_sha256",
            "pre_capture_save_sha256",
        ):
            object.__setattr__(
                self,
                name,
                _sha256(getattr(self, name), f"pc_environment.{name}"),
            )

        curves = _mapping(self.curve_sha256, "pc_environment.curve_sha256")
        if not curves:
            raise PcGoldenValidationError(
                "pc_environment.curve_sha256 must not be empty"
            )
        normalized_curves: dict[str, str] = {}
        for name, digest in sorted(curves.items()):
            logical_name = _nonempty_string(name, "curve logical name")
            normalized_curves[logical_name] = _sha256(
                digest,
                f"curve_sha256[{logical_name!r}]",
            )
        object.__setattr__(
            self,
            "curve_sha256",
            MappingProxyType(normalized_curves),
        )

        for name in (
            "profile_mode",
            "renderer_api",
            "renderer_mode",
            "os_build",
            "gpu",
            "driver",
        ):
            object.__setattr__(
                self,
                name,
                _nonempty_string(
                    getattr(self, name),
                    f"pc_environment.{name}",
                ),
            )
        radius = _strict_int(
            self.ball_radius_branch,
            "pc_environment.ball_radius_branch",
        )
        if radius not in {17, 18}:
            raise PcGoldenValidationError(
                "ball_radius_branch must be the native 17 or 18 branch"
            )
        object.__setattr__(self, "ball_radius_branch", radius)
        object.__setattr__(
            self,
            "game_settings",
            _frozen_json_mapping(
                self.game_settings,
                "pc_environment.game_settings",
            ),
        )
        object.__setattr__(
            self,
            "dynamic_difficulty_state",
            _frozen_json_mapping(
                self.dynamic_difficulty_state,
                "pc_environment.dynamic_difficulty_state",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "executable_sha256": self.executable_sha256,
            "main_pak_sha256": self.main_pak_sha256,
            "levels_xml_sha256": self.levels_xml_sha256,
            "curve_sha256": dict(self.curve_sha256),
            "pre_capture_save_sha256": self.pre_capture_save_sha256,
            "profile_mode": self.profile_mode,
            "renderer_api": self.renderer_api,
            "renderer_mode": self.renderer_mode,
            "ball_radius_branch": self.ball_radius_branch,
            "os_build": self.os_build,
            "gpu": self.gpu,
            "driver": self.driver,
            "game_settings": _plain_json_value(self.game_settings),
            "dynamic_difficulty_state": _plain_json_value(
                self.dynamic_difficulty_state
            ),
        }

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PcEnvironment":
        data = _mapping(value, "pc_environment")
        fields = {
            "executable_sha256",
            "main_pak_sha256",
            "levels_xml_sha256",
            "curve_sha256",
            "pre_capture_save_sha256",
            "profile_mode",
            "renderer_api",
            "renderer_mode",
            "ball_radius_branch",
            "os_build",
            "gpu",
            "driver",
            "game_settings",
            "dynamic_difficulty_state",
        }
        _check_keys(data, required=fields, name="pc_environment")
        return cls(**{name: data[name] for name in fields})


@dataclass(frozen=True, slots=True)
class InputTimeline:
    """Immutable PopCap DMO v2 input evidence for one capture."""

    artifact: str
    format: str
    file_id: int
    dmo_version: int
    product_version: str
    random_seed: int
    length_updates: int
    native_tick_offset: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "artifact",
            _nonempty_string(
                self.artifact,
                "input_timeline.artifact",
            ),
        )
        if self.format != DMO_FORMAT:
            raise PcGoldenValidationError(
                f"input_timeline.format must be {DMO_FORMAT!r}"
            )
        file_id = _strict_int(
            self.file_id,
            "input_timeline.file_id",
            minimum=0,
        )
        if file_id != DMO_FILE_ID:
            raise PcGoldenValidationError(
                "input_timeline.file_id must be the PopCap DMO "
                f"identifier 0x{DMO_FILE_ID:08X}"
            )
        object.__setattr__(self, "file_id", file_id)
        version = _strict_int(
            self.dmo_version,
            "input_timeline.dmo_version",
            minimum=0,
        )
        if version != DMO_VERSION:
            raise PcGoldenValidationError(
                f"input_timeline.dmo_version must be {DMO_VERSION}"
            )
        object.__setattr__(self, "dmo_version", version)
        object.__setattr__(
            self,
            "product_version",
            _nonempty_string(
                self.product_version,
                "input_timeline.product_version",
            ),
        )
        random_seed = _strict_int(
            self.random_seed,
            "input_timeline.random_seed",
            minimum=0,
        )
        if random_seed > 0xFFFFFFFF:
            raise PcGoldenValidationError(
                "input_timeline.random_seed must fit in an unsigned 32-bit "
                "integer"
            )
        object.__setattr__(self, "random_seed", random_seed)
        object.__setattr__(
            self,
            "length_updates",
            _strict_int(
                self.length_updates,
                "input_timeline.length_updates",
                minimum=0,
            ),
        )
        object.__setattr__(
            self,
            "native_tick_offset",
            _strict_int(
                self.native_tick_offset,
                "input_timeline.native_tick_offset",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact": self.artifact,
            "format": self.format,
            "file_id": self.file_id,
            "dmo_version": self.dmo_version,
            "product_version": self.product_version,
            "random_seed": self.random_seed,
            "length_updates": self.length_updates,
            "native_tick_offset": self.native_tick_offset,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "InputTimeline":
        data = _mapping(value, "input_timeline")
        fields = {
            "artifact",
            "format",
            "file_id",
            "dmo_version",
            "product_version",
            "random_seed",
            "length_updates",
            "native_tick_offset",
        }
        _check_keys(data, required=fields, name="input_timeline")
        return cls(**{name: data[name] for name in fields})


def _ratio(value: Any, name: str) -> tuple[int, int]:
    sequence = _sequence(value, name)
    if len(sequence) != 2:
        raise PcGoldenValidationError(f"{name} must contain [numerator, denominator]")
    numerator = _strict_int(sequence[0], f"{name}[0]", minimum=1)
    denominator = _strict_int(sequence[1], f"{name}[1]", minimum=1)
    return numerator, denominator


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    """Decode/presentation metadata for the immutable raw video artifact."""

    artifact: str
    width: int
    height: int
    codec: str
    pixel_format: str
    time_base: tuple[int, int]
    nominal_fps: tuple[int, int]
    frame_count: int
    first_pts: int
    last_pts: int
    cfr: bool
    dropped_frames: int
    duplicate_frames: int
    timing_mode: str = "integer_cfr_grid"

    def __post_init__(self) -> None:
        for name in ("artifact", "codec", "pixel_format"):
            object.__setattr__(
                self,
                name,
                _nonempty_string(getattr(self, name), f"video.{name}"),
            )
        object.__setattr__(
            self,
            "width",
            _strict_int(self.width, "video.width", minimum=1),
        )
        object.__setattr__(
            self,
            "height",
            _strict_int(self.height, "video.height", minimum=1),
        )
        object.__setattr__(
            self,
            "time_base",
            _ratio(self.time_base, "video.time_base"),
        )
        object.__setattr__(
            self,
            "nominal_fps",
            _ratio(self.nominal_fps, "video.nominal_fps"),
        )
        object.__setattr__(
            self,
            "frame_count",
            _strict_int(self.frame_count, "video.frame_count", minimum=1),
        )
        first_pts = _strict_int(self.first_pts, "video.first_pts")
        last_pts = _strict_int(self.last_pts, "video.last_pts")
        if last_pts < first_pts:
            raise PcGoldenValidationError(
                "video.last_pts must not precede video.first_pts"
            )
        object.__setattr__(self, "first_pts", first_pts)
        object.__setattr__(self, "last_pts", last_pts)
        object.__setattr__(self, "cfr", _strict_bool(self.cfr, "video.cfr"))
        object.__setattr__(
            self,
            "dropped_frames",
            _strict_int(
                self.dropped_frames,
                "video.dropped_frames",
                minimum=0,
            ),
        )
        object.__setattr__(
            self,
            "duplicate_frames",
            _strict_int(
                self.duplicate_frames,
                "video.duplicate_frames",
                minimum=0,
            ),
        )
        if self.timing_mode not in {
            "integer_cfr_grid",
            "qpc_vfr_pts_table",
        }:
            raise PcGoldenValidationError(
                "video.timing_mode must be 'integer_cfr_grid' or "
                "'qpc_vfr_pts_table'"
            )
        if (
            self.timing_mode == "qpc_vfr_pts_table"
            and (
                self.dropped_frames != 0
                or self.duplicate_frames != 0
            )
        ):
            raise PcGoldenValidationError(
                "QPC VFR video must report source acquisition loss through "
                "raw provenance, not CFR dropped/duplicate slots"
            )
        frame_period_numerator = (
            self.time_base[1] * self.nominal_fps[1]
        )
        frame_period_denominator = (
            self.time_base[0] * self.nominal_fps[0]
        )
        if (
            self.timing_mode == "integer_cfr_grid"
            and frame_period_numerator % frame_period_denominator != 0
        ):
            raise PcGoldenValidationError(
                "video time_base must represent the nominal CFR frame "
                "period as an integer PTS step"
            )
        pts_step = (
            frame_period_numerator // frame_period_denominator
            if frame_period_numerator % frame_period_denominator == 0
            else None
        )
        if (
            self.timing_mode == "integer_cfr_grid"
            and self.cfr
            and self.dropped_frames == 0
            and self.duplicate_frames == 0
        ):
            assert pts_step is not None
            expected_last_pts = (
                self.first_pts + (self.frame_count - 1) * pts_step
            )
            if self.last_pts != expected_last_pts:
                raise PcGoldenValidationError(
                    "video CFR PTS range does not match time_base, "
                    "nominal_fps, and frame_count"
                )

    @property
    def cfr_pts_step(self) -> int:
        if self.timing_mode != "integer_cfr_grid":
            raise PcGoldenValidationError(
                "QPC VFR video has no nominal integer CFR PTS step"
            )
        return (
            self.time_base[1] * self.nominal_fps[1]
        ) // (self.time_base[0] * self.nominal_fps[0])

    def to_dict(self) -> dict[str, Any]:
        result = {
            "artifact": self.artifact,
            "width": self.width,
            "height": self.height,
            "codec": self.codec,
            "pixel_format": self.pixel_format,
            "time_base": list(self.time_base),
            "nominal_fps": list(self.nominal_fps),
            "frame_count": self.frame_count,
            "first_pts": self.first_pts,
            "last_pts": self.last_pts,
            "cfr": self.cfr,
            "dropped_frames": self.dropped_frames,
            "duplicate_frames": self.duplicate_frames,
        }
        if self.timing_mode != "integer_cfr_grid":
            result["timing_mode"] = self.timing_mode
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "VideoMetadata":
        data = _mapping(value, "video")
        fields = {
            "artifact",
            "width",
            "height",
            "codec",
            "pixel_format",
            "time_base",
            "nominal_fps",
            "frame_count",
            "first_pts",
            "last_pts",
            "cfr",
            "dropped_frames",
            "duplicate_frames",
        }
        _check_keys(
            data,
            required=fields,
            optional={"timing_mode"},
            name="video",
        )
        return cls(
            **{name: data[name] for name in fields},
            timing_mode=data.get("timing_mode", "integer_cfr_grid"),
        )


@dataclass(frozen=True, slots=True)
class TickClock:
    """Explicit native-tick and video-PTS alignment contract."""

    logic_hz: tuple[int, int]
    tick_start: int
    tick_end: int
    tick0_video_pts: int
    sample_phase: str
    mapping_kind: str
    tick_map_artifact: str
    uncertainty_ticks: float

    def __post_init__(self) -> None:
        logic_hz = _ratio(self.logic_hz, "clock.logic_hz")
        if logic_hz != (100, 1):
            raise PcGoldenValidationError(
                "PC golden v3 requires the native 100/1 Hz logic clock"
            )
        object.__setattr__(self, "logic_hz", logic_hz)
        tick_start = _strict_int(
            self.tick_start,
            "clock.tick_start",
            minimum=0,
        )
        if tick_start != 0:
            raise PcGoldenValidationError("clock.tick_start must be zero")
        tick_end = _strict_int(
            self.tick_end,
            "clock.tick_end",
            minimum=0,
        )
        object.__setattr__(self, "tick_start", tick_start)
        object.__setattr__(self, "tick_end", tick_end)
        object.__setattr__(
            self,
            "tick0_video_pts",
            _strict_int(self.tick0_video_pts, "clock.tick0_video_pts"),
        )
        if self.sample_phase != _SAMPLE_PHASE:
            raise PcGoldenValidationError(
                f"clock.sample_phase must be {_SAMPLE_PHASE!r}"
            )
        object.__setattr__(
            self,
            "mapping_kind",
            _nonempty_string(
                self.mapping_kind,
                "clock.mapping_kind",
            ),
        )
        object.__setattr__(
            self,
            "tick_map_artifact",
            _nonempty_string(
                self.tick_map_artifact,
                "clock.tick_map_artifact",
            ),
        )
        object.__setattr__(
            self,
            "uncertainty_ticks",
            _finite_float(
                self.uncertainty_ticks,
                "clock.uncertainty_ticks",
                minimum=0.0,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "logic_hz": list(self.logic_hz),
            "tick_start": self.tick_start,
            "tick_end": self.tick_end,
            "tick0_video_pts": self.tick0_video_pts,
            "sample_phase": self.sample_phase,
            "mapping_kind": self.mapping_kind,
            "tick_map_artifact": self.tick_map_artifact,
            "uncertainty_ticks": self.uncertainty_ticks,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TickClock":
        data = _mapping(value, "clock")
        fields = {
            "logic_hz",
            "tick_start",
            "tick_end",
            "tick0_video_pts",
            "sample_phase",
            "mapping_kind",
            "tick_map_artifact",
            "uncertainty_ticks",
        }
        _check_keys(data, required=fields, name="clock")
        return cls(**{name: data[name] for name in fields})


@dataclass(frozen=True, slots=True)
class CoordinateCalibration:
    """A finite invertible 3×3 raw-pixel to 800×600 logical transform."""

    raw_width: int
    raw_height: int
    logical_width: int
    logical_height: int
    transform_kind: str
    pixel_center_convention: str
    logical_from_raw: tuple[tuple[float, float, float], ...]
    rms_error_px: float
    max_error_px: float
    calibration_artifact: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "raw_width",
            _strict_int(
                self.raw_width,
                "coordinates.raw_width",
                minimum=1,
            ),
        )
        object.__setattr__(
            self,
            "raw_height",
            _strict_int(
                self.raw_height,
                "coordinates.raw_height",
                minimum=1,
            ),
        )
        logical_width = _strict_int(
            self.logical_width,
            "coordinates.logical_width",
            minimum=1,
        )
        logical_height = _strict_int(
            self.logical_height,
            "coordinates.logical_height",
            minimum=1,
        )
        if (logical_width, logical_height) != (
            _LOGICAL_WIDTH,
            _LOGICAL_HEIGHT,
        ):
            raise PcGoldenValidationError(
                "coordinates logical space must be the original 800x600 canvas"
            )
        object.__setattr__(self, "logical_width", logical_width)
        object.__setattr__(self, "logical_height", logical_height)

        if self.transform_kind not in {"axis_aligned_affine", "homography"}:
            raise PcGoldenValidationError(
                "coordinates.transform_kind must be axis_aligned_affine "
                "or homography"
            )
        if self.pixel_center_convention not in {
            "center_at_integer",
            "center_at_half",
        }:
            raise PcGoldenValidationError(
                "coordinates.pixel_center_convention must be "
                "center_at_integer or center_at_half"
            )

        try:
            matrix = np.asarray(
                self.logical_from_raw,
                dtype=np.float64,
            )
        except (TypeError, ValueError) as error:
            raise PcGoldenValidationError(
                "coordinates.logical_from_raw must be a finite 3x3 matrix"
            ) from error
        if matrix.shape != (3, 3) or np.any(~np.isfinite(matrix)):
            raise PcGoldenValidationError(
                "coordinates.logical_from_raw must be a finite 3x3 matrix"
            )
        if abs(float(np.linalg.det(matrix))) <= 1e-15:
            raise PcGoldenValidationError(
                "coordinates.logical_from_raw must be invertible"
            )
        if self.transform_kind == "axis_aligned_affine":
            expected_zero = np.array(
                (
                    matrix[0, 1],
                    matrix[1, 0],
                    matrix[2, 0],
                    matrix[2, 1],
                )
            )
            if (
                np.any(np.abs(expected_zero) > 1e-12)
                or abs(float(matrix[2, 2]) - 1.0) > 1e-12
                or matrix[0, 0] <= 0.0
                or matrix[1, 1] <= 0.0
            ):
                raise PcGoldenValidationError(
                    "axis_aligned_affine must contain positive x/y scales, "
                    "translation, and final row [0,0,1]"
                )
        object.__setattr__(
            self,
            "logical_from_raw",
            tuple(tuple(float(item) for item in row) for row in matrix),
        )

        rms = _finite_float(
            self.rms_error_px,
            "coordinates.rms_error_px",
            minimum=0.0,
        )
        maximum = _finite_float(
            self.max_error_px,
            "coordinates.max_error_px",
            minimum=0.0,
        )
        if rms > maximum:
            raise PcGoldenValidationError(
                "coordinates.rms_error_px must not exceed max_error_px"
            )
        if rms > _MAX_CALIBRATION_RMS_ERROR_PX:
            raise PcGoldenValidationError(
                "coordinates.rms_error_px exceeds the 0.25 px fidelity "
                "ceiling"
            )
        if maximum > _MAX_CALIBRATION_ERROR_PX:
            raise PcGoldenValidationError(
                "coordinates.max_error_px exceeds the 0.5 px fidelity "
                "ceiling"
            )
        object.__setattr__(self, "rms_error_px", rms)
        object.__setattr__(self, "max_error_px", maximum)
        object.__setattr__(
            self,
            "calibration_artifact",
            _nonempty_string(
                self.calibration_artifact,
                "coordinates.calibration_artifact",
            ),
        )

    @property
    def matrix(self) -> NDArray[np.float64]:
        return np.asarray(self.logical_from_raw, dtype=np.float64)

    def raw_to_logical(self, points: Any) -> NDArray[np.float64]:
        try:
            values = np.asarray(points, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise PcGoldenValidationError(
                "raw points must be a finite numeric array"
            ) from error
        if values.shape == () or values.shape[-1] != 2:
            raise PcGoldenValidationError(
                "raw points must have a final dimension of size two"
            )
        if np.any(~np.isfinite(values)):
            raise PcGoldenValidationError("raw points must be finite")
        flat = values.reshape(-1, 2)
        homogeneous = np.column_stack(
            (flat, np.ones(len(flat), dtype=np.float64))
        )
        mapped = homogeneous @ self.matrix.T
        denominator = mapped[:, 2]
        if np.any(~np.isfinite(denominator)) or np.any(
            np.abs(denominator) <= 1e-15
        ):
            raise PcGoldenValidationError(
                "coordinate transform maps a point to infinity"
            )
        logical = mapped[:, :2] / denominator[:, np.newaxis]
        if np.any(~np.isfinite(logical)):
            raise PcGoldenValidationError(
                "coordinate transform produced non-finite coordinates"
            )
        return logical.reshape(values.shape)

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_width": self.raw_width,
            "raw_height": self.raw_height,
            "logical_width": self.logical_width,
            "logical_height": self.logical_height,
            "transform_kind": self.transform_kind,
            "pixel_center_convention": self.pixel_center_convention,
            "logical_from_raw": [
                list(row) for row in self.logical_from_raw
            ],
            "rms_error_px": self.rms_error_px,
            "max_error_px": self.max_error_px,
            "calibration_artifact": self.calibration_artifact,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CoordinateCalibration":
        data = _mapping(value, "coordinates")
        fields = {
            "raw_width",
            "raw_height",
            "logical_width",
            "logical_height",
            "transform_kind",
            "pixel_center_convention",
            "logical_from_raw",
            "rms_error_px",
            "max_error_px",
            "calibration_artifact",
        }
        _check_keys(data, required=fields, name="coordinates")
        return cls(**{name: data[name] for name in fields})


@dataclass(frozen=True, slots=True)
class CoverageRange:
    """Evidence coverage for a named trace channel and inclusive tick range."""

    channel: str
    start_tick: int
    end_tick: int
    status: CoverageStatus
    required: bool
    reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "channel",
            _nonempty_string(self.channel, "coverage.channel"),
        )
        start = _strict_int(
            self.start_tick,
            "coverage.start_tick",
            minimum=0,
        )
        end = _strict_int(
            self.end_tick,
            "coverage.end_tick",
            minimum=0,
        )
        if end < start:
            raise PcGoldenValidationError(
                "coverage.end_tick must not precede start_tick"
            )
        object.__setattr__(self, "start_tick", start)
        object.__setattr__(self, "end_tick", end)
        status = _enum_value(
            self.status,
            CoverageStatus,
            "coverage.status",
        )
        object.__setattr__(self, "status", status)
        object.__setattr__(
            self,
            "required",
            _strict_bool(self.required, "coverage.required"),
        )
        if status is CoverageStatus.COMPLETE:
            if self.reason is not None:
                object.__setattr__(
                    self,
                    "reason",
                    _nonempty_string(self.reason, "coverage.reason"),
                )
        else:
            object.__setattr__(
                self,
                "reason",
                _nonempty_string(
                    self.reason,
                    f"{status.value} coverage.reason",
                ),
            )

    def to_dict(self) -> dict[str, Any]:
        result = {
            "channel": self.channel,
            "start_tick": self.start_tick,
            "end_tick": self.end_tick,
            "status": self.status.value,
            "required": self.required,
        }
        if self.reason is not None:
            result["reason"] = self.reason
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CoverageRange":
        data = _mapping(value, "coverage")
        _check_keys(
            data,
            required={
                "channel",
                "start_tick",
                "end_tick",
                "status",
                "required",
            },
            optional={"reason"},
            name="coverage",
        )
        return cls(
            channel=data["channel"],
            start_tick=data["start_tick"],
            end_tick=data["end_tick"],
            status=data["status"],
            required=data["required"],
            reason=data.get("reason"),
        )


@dataclass(frozen=True, slots=True)
class ComparisonContract:
    """Frozen v3 tolerances with project-level fidelity ceilings."""

    event_tick_tolerance: int
    center_l2_tolerance_px: float
    waypoint_abs_tolerance: float
    max_drift_per_100_ticks: float

    def __post_init__(self) -> None:
        event_tolerance = _strict_int(
            self.event_tick_tolerance,
            "comparison_contract.event_tick_tolerance",
            minimum=0,
        )
        if event_tolerance != 0:
            raise PcGoldenValidationError(
                "PC golden discrete events require zero tick tolerance"
            )
        object.__setattr__(
            self,
            "event_tick_tolerance",
            event_tolerance,
        )
        for name in (
            "center_l2_tolerance_px",
            "waypoint_abs_tolerance",
            "max_drift_per_100_ticks",
        ):
            object.__setattr__(
                self,
                name,
                _finite_float(
                    getattr(self, name),
                    f"comparison_contract.{name}",
                    minimum=0.0,
                ),
            )
        limits = {
            "center_l2_tolerance_px": 0.5,
            "waypoint_abs_tolerance": 0.05,
            "max_drift_per_100_ticks": 0.05,
        }
        for name, maximum in limits.items():
            if getattr(self, name) > maximum:
                raise PcGoldenValidationError(
                    f"comparison_contract.{name} must not exceed {maximum}"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_tick_tolerance": self.event_tick_tolerance,
            "center_l2_tolerance_px": self.center_l2_tolerance_px,
            "waypoint_abs_tolerance": self.waypoint_abs_tolerance,
            "max_drift_per_100_ticks": self.max_drift_per_100_ticks,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ComparisonContract":
        data = _mapping(value, "comparison_contract")
        fields = {
            "event_tick_tolerance",
            "center_l2_tolerance_px",
            "waypoint_abs_tolerance",
            "max_drift_per_100_ticks",
        }
        _check_keys(data, required=fields, name="comparison_contract")
        return cls(**{name: data[name] for name in fields})


@dataclass(frozen=True, slots=True)
class SaveRunContract:
    """State snapshots and native process identity for one replay run."""

    run_id: str
    start_snapshot_artifact: str
    end_snapshot_artifact: str
    process_id: int
    process_creation_filetime_100ns: int

    def __post_init__(self) -> None:
        run_id = _nonempty_string(self.run_id, "save run_id")
        if re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", run_id) is None:
            raise PcGoldenValidationError(
                "save run_id must use lowercase ASCII letters, digits, "
                "underscores, or hyphens"
            )
        object.__setattr__(self, "run_id", run_id)
        for name in (
            "start_snapshot_artifact",
            "end_snapshot_artifact",
        ):
            object.__setattr__(
                self,
                name,
                _nonempty_string(getattr(self, name), f"save run {name}"),
            )
        if self.start_snapshot_artifact == self.end_snapshot_artifact:
            raise PcGoldenValidationError(
                "save run start and end snapshots must be distinct artifacts"
            )
        object.__setattr__(
            self,
            "process_id",
            _strict_int(self.process_id, "save run process_id", minimum=1),
        )
        object.__setattr__(
            self,
            "process_creation_filetime_100ns",
            _strict_int(
                self.process_creation_filetime_100ns,
                "save run process_creation_filetime_100ns",
                minimum=1,
            ),
        )

    @property
    def process_instance(self) -> tuple[int, int]:
        return self.process_id, self.process_creation_filetime_100ns

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "start_snapshot_artifact": self.start_snapshot_artifact,
            "end_snapshot_artifact": self.end_snapshot_artifact,
            "process_id": self.process_id,
            "process_creation_filetime_100ns": (
                self.process_creation_filetime_100ns
            ),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SaveRunContract":
        data = _mapping(value, "save run")
        fields = {
            "run_id",
            "start_snapshot_artifact",
            "end_snapshot_artifact",
            "process_id",
            "process_creation_filetime_100ns",
        }
        _check_keys(data, required=fields, name="save run")
        return cls(**{name: data[name] for name in fields})


SAVE_VOLATILE_REGISTRY_ROLES = (
    "launcher_last_game_pid",
    "launcher_registration_execution_data",
)


@dataclass(frozen=True, slots=True)
class SaveTransactionContract:
    """Raw artifacts needed to recompute a complete save transaction."""

    journal_artifact: str
    process_timeline_artifact: str
    pre_snapshot_artifact: str
    restored_snapshot_artifact: str
    runs: tuple[SaveRunContract, ...]
    volatile_registry_roles: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "journal_artifact",
            "process_timeline_artifact",
            "pre_snapshot_artifact",
            "restored_snapshot_artifact",
        ):
            object.__setattr__(
                self,
                name,
                _nonempty_string(
                    getattr(self, name),
                    f"save_transaction.{name}",
                ),
            )
        if self.pre_snapshot_artifact == self.restored_snapshot_artifact:
            raise PcGoldenValidationError(
                "pre and restored snapshots must be distinct artifacts"
            )
        runs = tuple(self.runs)
        if len(runs) < 2:
            raise PcGoldenValidationError(
                "save transaction requires at least two replay runs"
            )
        if any(not isinstance(run, SaveRunContract) for run in runs):
            raise PcGoldenValidationError(
                "save transaction runs must contain SaveRunContract objects"
            )
        run_ids = [run.run_id for run in runs]
        if len(set(run_ids)) != len(run_ids):
            raise PcGoldenValidationError(
                "save transaction run_id values must be unique"
            )
        process_instances = [run.process_instance for run in runs]
        if len(set(process_instances)) != len(process_instances):
            raise PcGoldenValidationError(
                "save transaction runs must use independent process instances"
            )
        snapshot_artifacts = [
            self.pre_snapshot_artifact,
            self.restored_snapshot_artifact,
            *(
                artifact
                for run in runs
                for artifact in (
                    run.start_snapshot_artifact,
                    run.end_snapshot_artifact,
                )
            ),
        ]
        if len(set(snapshot_artifacts)) != len(snapshot_artifacts):
            raise PcGoldenValidationError(
                "save transaction phases must use distinct snapshot artifacts"
            )
        control_artifacts = {
            self.journal_artifact,
            self.process_timeline_artifact,
        }
        if control_artifacts.intersection(snapshot_artifacts):
            raise PcGoldenValidationError(
                "save transaction control and snapshot artifacts must differ"
            )
        if len(control_artifacts) != 2:
            raise PcGoldenValidationError(
                "save journal and process timeline artifacts must differ"
            )
        volatile_registry_roles = tuple(self.volatile_registry_roles)
        if volatile_registry_roles != SAVE_VOLATILE_REGISTRY_ROLES:
            raise PcGoldenValidationError(
                "save transaction must declare the exact frozen Zuma "
                "launcher volatile-registry roles"
            )
        object.__setattr__(self, "runs", runs)
        object.__setattr__(
            self,
            "volatile_registry_roles",
            volatile_registry_roles,
        )

    @property
    def referenced_artifacts(self) -> frozenset[str]:
        return frozenset(
            {
                self.journal_artifact,
                self.process_timeline_artifact,
                self.pre_snapshot_artifact,
                self.restored_snapshot_artifact,
                *(
                    artifact
                    for run in self.runs
                    for artifact in (
                        run.start_snapshot_artifact,
                        run.end_snapshot_artifact,
                    )
                ),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "journal_artifact": self.journal_artifact,
            "process_timeline_artifact": self.process_timeline_artifact,
            "pre_snapshot_artifact": self.pre_snapshot_artifact,
            "restored_snapshot_artifact": self.restored_snapshot_artifact,
            "runs": [run.to_dict() for run in self.runs],
            "volatile_registry_roles": list(
                self.volatile_registry_roles
            ),
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> "SaveTransactionContract":
        data = _mapping(value, "save_transaction")
        fields = {
            "journal_artifact",
            "process_timeline_artifact",
            "pre_snapshot_artifact",
            "restored_snapshot_artifact",
            "runs",
            "volatile_registry_roles",
        }
        _check_keys(data, required=fields, name="save_transaction")
        return cls(
            journal_artifact=data["journal_artifact"],
            process_timeline_artifact=data["process_timeline_artifact"],
            pre_snapshot_artifact=data["pre_snapshot_artifact"],
            restored_snapshot_artifact=data["restored_snapshot_artifact"],
            runs=tuple(
                SaveRunContract.from_dict(_mapping(item, "save run"))
                for item in _sequence(data["runs"], "save_transaction.runs")
            ),
            volatile_registry_roles=tuple(
                _nonempty_string(
                    item,
                    "save_transaction.volatile_registry_roles item",
                )
                for item in _sequence(
                    data["volatile_registry_roles"],
                    "save_transaction.volatile_registry_roles",
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class RenderSettledReplayEvidence:
    """Run-local sources and shared calibration for settled frame labels."""

    update_map_artifact: str
    frames_csv_artifact: str
    framework_poll_artifact: str
    framework_state_artifact: str
    calibration_preregistration_artifact: str
    calibration_execution_binding_artifact: str
    calibration_holdout_report_artifact: str

    def __post_init__(self) -> None:
        for name in (
            "update_map_artifact",
            "frames_csv_artifact",
            "framework_poll_artifact",
            "framework_state_artifact",
            "calibration_preregistration_artifact",
            "calibration_execution_binding_artifact",
            "calibration_holdout_report_artifact",
        ):
            object.__setattr__(
                self,
                name,
                _nonempty_string(
                    getattr(self, name),
                    f"render-settled replay {name}",
                ),
            )

    @property
    def referenced_artifacts(self) -> frozenset[str]:
        return frozenset(
            {
                self.update_map_artifact,
                self.frames_csv_artifact,
                self.framework_poll_artifact,
                self.framework_state_artifact,
                self.calibration_preregistration_artifact,
                self.calibration_execution_binding_artifact,
                self.calibration_holdout_report_artifact,
            }
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "update_map_artifact": self.update_map_artifact,
            "frames_csv_artifact": self.frames_csv_artifact,
            "framework_poll_artifact": self.framework_poll_artifact,
            "framework_state_artifact": self.framework_state_artifact,
            "calibration_preregistration_artifact": (
                self.calibration_preregistration_artifact
            ),
            "calibration_execution_binding_artifact": (
                self.calibration_execution_binding_artifact
            ),
            "calibration_holdout_report_artifact": (
                self.calibration_holdout_report_artifact
            ),
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> "RenderSettledReplayEvidence":
        data = _mapping(value, "render-settled replay evidence")
        fields = {
            "update_map_artifact",
            "frames_csv_artifact",
            "framework_poll_artifact",
            "framework_state_artifact",
            "calibration_preregistration_artifact",
            "calibration_execution_binding_artifact",
            "calibration_holdout_report_artifact",
        }
        _check_keys(
            data,
            required=fields,
            name="render-settled replay evidence",
        )
        return cls(**{name: data[name] for name in fields})


@dataclass(frozen=True, slots=True)
class ReplayRunContract:
    """Video/clock/trace evidence for one independent native replay."""

    run_id: str
    trace_artifact: str
    video: VideoMetadata
    clock: TickClock
    capture_metadata_artifact: str
    framework_update_artifact: str
    render_settled_evidence: RenderSettledReplayEvidence | None = None

    def __post_init__(self) -> None:
        run_id = _nonempty_string(self.run_id, "replay run_id")
        if re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", run_id) is None:
            raise PcGoldenValidationError(
                "replay run_id must use lowercase ASCII letters, digits, "
                "underscores, or hyphens"
            )
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(
            self,
            "trace_artifact",
            _nonempty_string(
                self.trace_artifact,
                "replay run trace_artifact",
            ),
        )
        object.__setattr__(
            self,
            "capture_metadata_artifact",
            _nonempty_string(
                self.capture_metadata_artifact,
                "replay run capture_metadata_artifact",
            ),
        )
        object.__setattr__(
            self,
            "framework_update_artifact",
            _nonempty_string(
                self.framework_update_artifact,
                "replay run framework_update_artifact",
            ),
        )
        if not isinstance(self.video, VideoMetadata):
            raise PcGoldenValidationError(
                "replay run video must be VideoMetadata"
            )
        if not isinstance(self.clock, TickClock):
            raise PcGoldenValidationError(
                "replay run clock must be TickClock"
            )
        if (
            self.render_settled_evidence is not None
            and not isinstance(
                self.render_settled_evidence,
                RenderSettledReplayEvidence,
            )
        ):
            raise PcGoldenValidationError(
                "replay run render_settled_evidence must be a "
                "RenderSettledReplayEvidence or null"
            )
        if not (
            self.video.first_pts
            <= self.clock.tick0_video_pts
            <= self.video.last_pts
        ):
            raise PcGoldenValidationError(
                "replay run tick-zero PTS is outside its video"
            )

    @property
    def referenced_artifacts(self) -> frozenset[str]:
        result = {
            self.trace_artifact,
            self.video.artifact,
            self.clock.tick_map_artifact,
            self.capture_metadata_artifact,
            self.framework_update_artifact,
        }
        if self.render_settled_evidence is not None:
            result.update(
                self.render_settled_evidence.referenced_artifacts
            )
        return frozenset(result)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "run_id": self.run_id,
            "trace_artifact": self.trace_artifact,
            "video": self.video.to_dict(),
            "clock": self.clock.to_dict(),
            "capture_metadata_artifact": self.capture_metadata_artifact,
            "framework_update_artifact": self.framework_update_artifact,
        }
        if self.render_settled_evidence is not None:
            result["render_settled_evidence"] = (
                self.render_settled_evidence.to_dict()
            )
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReplayRunContract":
        data = _mapping(value, "replay run")
        fields = {
            "run_id",
            "trace_artifact",
            "video",
            "clock",
            "capture_metadata_artifact",
            "framework_update_artifact",
        }
        _check_keys(
            data,
            required=fields,
            optional={"render_settled_evidence"},
            name="replay run",
        )
        return cls(
            run_id=data["run_id"],
            trace_artifact=data["trace_artifact"],
            video=VideoMetadata.from_dict(
                _mapping(data["video"], "replay run video")
            ),
            clock=TickClock.from_dict(
                _mapping(data["clock"], "replay run clock")
            ),
            capture_metadata_artifact=data["capture_metadata_artifact"],
            framework_update_artifact=data["framework_update_artifact"],
            render_settled_evidence=(
                RenderSettledReplayEvidence.from_dict(
                    _mapping(
                        data["render_settled_evidence"],
                        "render-settled replay evidence",
                    )
                )
                if "render_settled_evidence" in data
                else None
            ),
        )


@dataclass(frozen=True, slots=True)
class ReplayPixelComparisonContract:
    """A narrow, explicit exception for the native client's bottom raster edge.

    The retail client can leave a handful of process-local pixels in its final
    client row while the complete gameplay viewport above that row is exact.
    This contract deliberately supports only that observed one-row boundary;
    it cannot be used to mask HUD or gameplay-area differences.
    """

    mode: str
    excluded_bottom_rows: int
    maximum_excluded_edge_mismatches: int

    def __post_init__(self) -> None:
        mode = _nonempty_string(self.mode, "replay pixel comparison mode")
        if mode != "exact_except_bounded_bottom_raster_edge":
            raise PcGoldenValidationError(
                "replay pixel comparison mode is unsupported"
            )
        excluded_bottom_rows = _strict_int(
            self.excluded_bottom_rows,
            "replay pixel comparison excluded_bottom_rows",
            minimum=1,
        )
        if excluded_bottom_rows != 1:
            raise PcGoldenValidationError(
                "replay pixel comparison may exclude exactly one bottom row"
            )
        maximum_mismatches = _strict_int(
            self.maximum_excluded_edge_mismatches,
            "replay pixel comparison maximum_excluded_edge_mismatches",
            minimum=1,
        )
        object.__setattr__(self, "mode", mode)
        object.__setattr__(
            self,
            "excluded_bottom_rows",
            excluded_bottom_rows,
        )
        object.__setattr__(
            self,
            "maximum_excluded_edge_mismatches",
            maximum_mismatches,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "excluded_bottom_rows": self.excluded_bottom_rows,
            "maximum_excluded_edge_mismatches": (
                self.maximum_excluded_edge_mismatches
            ),
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> "ReplayPixelComparisonContract":
        data = _mapping(value, "replay pixel comparison")
        fields = {
            "mode",
            "excluded_bottom_rows",
            "maximum_excluded_edge_mismatches",
        }
        _check_keys(data, required=fields, name="replay pixel comparison")
        return cls(
            mode=data["mode"],
            excluded_bottom_rows=data["excluded_bottom_rows"],
            maximum_excluded_edge_mismatches=(
                data["maximum_excluded_edge_mismatches"]
            ),
        )


@dataclass(frozen=True, slots=True)
class ReplayDeterminismContract:
    """At least two complete, independently launched native replays."""

    runs: tuple[ReplayRunContract, ...]
    pixel_comparison: ReplayPixelComparisonContract | None = None

    def __post_init__(self) -> None:
        runs = tuple(self.runs)
        if len(runs) < 2:
            raise PcGoldenValidationError(
                "replay determinism requires at least two runs"
            )
        if any(not isinstance(run, ReplayRunContract) for run in runs):
            raise PcGoldenValidationError(
                "replay determinism runs must contain ReplayRunContract "
                "objects"
            )
        run_ids = [run.run_id for run in runs]
        if len(set(run_ids)) != len(run_ids):
            raise PcGoldenValidationError(
                "replay determinism run_id values must be unique"
            )
        for field_name, values in (
            ("trace", [run.trace_artifact for run in runs]),
            ("video", [run.video.artifact for run in runs]),
            ("tick-map", [run.clock.tick_map_artifact for run in runs]),
            (
                "capture metadata",
                [run.capture_metadata_artifact for run in runs],
            ),
            (
                "framework update",
                [run.framework_update_artifact for run in runs],
            ),
        ):
            if len(set(values)) != len(values):
                raise PcGoldenValidationError(
                    f"replay runs must use distinct {field_name} artifacts"
                )
        settled = [run.render_settled_evidence for run in runs]
        if any(value is not None for value in settled):
            if not all(value is not None for value in settled):
                raise PcGoldenValidationError(
                    "all replay runs must declare render-settled evidence"
                )
            settled_rows = tuple(
                value for value in settled if value is not None
            )
            for field_name, values in (
                (
                    "render-settled update-map",
                    [row.update_map_artifact for row in settled_rows],
                ),
                (
                    "render-settled frame CSV",
                    [row.frames_csv_artifact for row in settled_rows],
                ),
                (
                    "framework poll",
                    [row.framework_poll_artifact for row in settled_rows],
                ),
                (
                    "framework state",
                    [row.framework_state_artifact for row in settled_rows],
                ),
            ):
                if len(set(values)) != len(values):
                    raise PcGoldenValidationError(
                        "replay runs must use distinct "
                        f"{field_name} artifacts"
                    )
            for field_name, values in (
                (
                    "calibration preregistration",
                    [
                        row.calibration_preregistration_artifact
                        for row in settled_rows
                    ],
                ),
                (
                    "calibration execution binding",
                    [
                        row.calibration_execution_binding_artifact
                        for row in settled_rows
                    ],
                ),
                (
                    "calibration holdout report",
                    [
                        row.calibration_holdout_report_artifact
                        for row in settled_rows
                    ],
                ),
            ):
                if len(set(values)) != 1:
                    raise PcGoldenValidationError(
                        "replay runs must share one render-settle "
                        f"{field_name} artifact"
                    )
        tick_ends = {run.clock.tick_end for run in runs}
        if len(tick_ends) != 1:
            raise PcGoldenValidationError(
                "replay runs must cover the same final native tick"
            )
        logic_rates = {run.clock.logic_hz for run in runs}
        if len(logic_rates) != 1:
            raise PcGoldenValidationError(
                "replay runs must use the same native logic rate"
            )
        sample_phases = {run.clock.sample_phase for run in runs}
        if len(sample_phases) != 1:
            raise PcGoldenValidationError(
                "replay runs must use the same sample phase"
            )
        pixel_comparison = self.pixel_comparison
        if pixel_comparison is not None:
            if not isinstance(
                pixel_comparison,
                ReplayPixelComparisonContract,
            ):
                raise PcGoldenValidationError(
                    "replay pixel comparison must be a "
                    "ReplayPixelComparisonContract or null"
                )
            widths = {run.video.width for run in runs}
            heights = {run.video.height for run in runs}
            if len(widths) != 1 or len(heights) != 1:
                raise PcGoldenValidationError(
                    "replay pixel comparison requires common video geometry"
                )
            width = next(iter(widths))
            height = next(iter(heights))
            if pixel_comparison.excluded_bottom_rows >= height:
                raise PcGoldenValidationError(
                    "replay pixel comparison excludes the complete viewport"
                )
            edge_pixels = width * pixel_comparison.excluded_bottom_rows
            if (
                pixel_comparison.maximum_excluded_edge_mismatches
                > edge_pixels
            ):
                raise PcGoldenValidationError(
                    "replay pixel mismatch budget exceeds the excluded edge"
                )
        object.__setattr__(self, "runs", runs)
        object.__setattr__(self, "pixel_comparison", pixel_comparison)

    @property
    def referenced_artifacts(self) -> frozenset[str]:
        return frozenset(
            artifact
            for run in self.runs
            for artifact in run.referenced_artifacts
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "runs": [run.to_dict() for run in self.runs]
        }
        if self.pixel_comparison is not None:
            result["pixel_comparison"] = self.pixel_comparison.to_dict()
        return result

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> "ReplayDeterminismContract":
        data = _mapping(value, "replay_determinism")
        _check_keys(
            data,
            required={"runs"},
            optional={"pixel_comparison"},
            name="replay_determinism",
        )
        return cls(
            runs=tuple(
                ReplayRunContract.from_dict(_mapping(item, "replay run"))
                for item in _sequence(
                    data["runs"],
                    "replay_determinism.runs",
                )
            ),
            pixel_comparison=(
                ReplayPixelComparisonContract.from_dict(
                    _mapping(
                        data["pixel_comparison"],
                        "replay pixel comparison",
                    )
                )
                if "pixel_comparison" in data
                else None
            ),
        )


@dataclass(frozen=True, slots=True)
class ExactStepRunContract:
    """One independently launched exact-step retail replay source."""

    run_id: str
    selected_attempt: int
    attempts_artifact: str
    memory_probe_artifact: str
    trajectory_index_artifact: str
    post_snapshot_artifact: str
    trace_artifact: str
    video: VideoMetadata
    clock: TickClock
    process_id: int
    process_creation_filetime_100ns: int

    def __post_init__(self) -> None:
        run_id = _nonempty_string(self.run_id, "exact-step run_id")
        if re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", run_id) is None:
            raise PcGoldenValidationError(
                "exact-step run_id must use lowercase ASCII letters, "
                "digits, underscores, or hyphens"
            )
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(
            self,
            "selected_attempt",
            _strict_int(
                self.selected_attempt,
                "exact-step selected_attempt",
                minimum=1,
            ),
        )
        artifact_fields = (
            "attempts_artifact",
            "memory_probe_artifact",
            "trajectory_index_artifact",
            "post_snapshot_artifact",
            "trace_artifact",
        )
        for name in artifact_fields:
            object.__setattr__(
                self,
                name,
                _nonempty_string(
                    getattr(self, name),
                    f"exact-step run {name}",
                ),
            )
        if len({getattr(self, name) for name in artifact_fields}) != len(
            artifact_fields
        ):
            raise PcGoldenValidationError(
                "exact-step run control artifacts must be distinct"
            )
        if not isinstance(self.video, VideoMetadata):
            raise PcGoldenValidationError(
                "exact-step run video must be VideoMetadata"
            )
        if not isinstance(self.clock, TickClock):
            raise PcGoldenValidationError(
                "exact-step run clock must be TickClock"
            )
        if self.video.artifact in {
            getattr(self, name) for name in artifact_fields
        } or self.clock.tick_map_artifact in {
            getattr(self, name) for name in artifact_fields
        } | {self.video.artifact}:
            raise PcGoldenValidationError(
                "exact-step run evidence artifacts must be distinct"
            )
        object.__setattr__(
            self,
            "process_id",
            _strict_int(
                self.process_id,
                "exact-step run process_id",
                minimum=1,
            ),
        )
        object.__setattr__(
            self,
            "process_creation_filetime_100ns",
            _strict_int(
                self.process_creation_filetime_100ns,
                "exact-step run process_creation_filetime_100ns",
                minimum=1,
            ),
        )
        if not (
            self.video.first_pts
            <= self.clock.tick0_video_pts
            <= self.video.last_pts
        ):
            raise PcGoldenValidationError(
                "exact-step run tick-zero PTS is outside its video"
            )

    @property
    def process_instance(self) -> tuple[int, int]:
        return self.process_id, self.process_creation_filetime_100ns

    @property
    def referenced_artifacts(self) -> frozenset[str]:
        return frozenset(
            {
                self.attempts_artifact,
                self.memory_probe_artifact,
                self.trajectory_index_artifact,
                self.post_snapshot_artifact,
                self.trace_artifact,
                self.video.artifact,
                self.clock.tick_map_artifact,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "selected_attempt": self.selected_attempt,
            "attempts_artifact": self.attempts_artifact,
            "memory_probe_artifact": self.memory_probe_artifact,
            "trajectory_index_artifact": self.trajectory_index_artifact,
            "post_snapshot_artifact": self.post_snapshot_artifact,
            "trace_artifact": self.trace_artifact,
            "video": self.video.to_dict(),
            "clock": self.clock.to_dict(),
            "process_id": self.process_id,
            "process_creation_filetime_100ns": (
                self.process_creation_filetime_100ns
            ),
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> "ExactStepRunContract":
        data = _mapping(value, "exact-step run")
        fields = {
            "run_id",
            "selected_attempt",
            "attempts_artifact",
            "memory_probe_artifact",
            "trajectory_index_artifact",
            "post_snapshot_artifact",
            "trace_artifact",
            "video",
            "clock",
            "process_id",
            "process_creation_filetime_100ns",
        }
        _check_keys(data, required=fields, name="exact-step run")
        return cls(
            run_id=data["run_id"],
            selected_attempt=data["selected_attempt"],
            attempts_artifact=data["attempts_artifact"],
            memory_probe_artifact=data["memory_probe_artifact"],
            trajectory_index_artifact=data["trajectory_index_artifact"],
            post_snapshot_artifact=data["post_snapshot_artifact"],
            trace_artifact=data["trace_artifact"],
            video=VideoMetadata.from_dict(
                _mapping(data["video"], "exact-step run video")
            ),
            clock=TickClock.from_dict(
                _mapping(data["clock"], "exact-step run clock")
            ),
            process_id=data["process_id"],
            process_creation_filetime_100ns=data[
                "process_creation_filetime_100ns"
            ],
        )


@dataclass(frozen=True, slots=True)
class ExactStepReplayContract:
    """Formal two-run exact-step transport and restoration contract."""

    preregistration_artifact: str
    execution_binding_artifact: str
    comparison_artifact: str
    pre_snapshot_artifact: str
    host_pre_snapshot_artifact: str
    runs: tuple[ExactStepRunContract, ...]
    freeze_update: int
    source_start_update: int
    source_end_update: int
    warmup_tick_count: int
    maximum_startup_attempts: int
    sample_phase: str
    pixel_comparison: str

    def __post_init__(self) -> None:
        for name in (
            "preregistration_artifact",
            "execution_binding_artifact",
            "comparison_artifact",
            "pre_snapshot_artifact",
            "host_pre_snapshot_artifact",
        ):
            object.__setattr__(
                self,
                name,
                _nonempty_string(
                    getattr(self, name),
                    f"exact_step_replay.{name}",
                ),
            )
        control_artifacts = {
            self.preregistration_artifact,
            self.execution_binding_artifact,
            self.comparison_artifact,
            self.pre_snapshot_artifact,
            self.host_pre_snapshot_artifact,
        }
        if len(control_artifacts) != 5:
            raise PcGoldenValidationError(
                "exact-step global control artifacts must be distinct"
            )
        runs = tuple(self.runs)
        if len(runs) < 2 or any(
            not isinstance(run, ExactStepRunContract) for run in runs
        ):
            raise PcGoldenValidationError(
                "exact-step replay requires at least two run contracts"
            )
        if len({run.run_id for run in runs}) != len(runs):
            raise PcGoldenValidationError(
                "exact-step replay run_id values must be unique"
            )
        if len({run.process_instance for run in runs}) != len(runs):
            raise PcGoldenValidationError(
                "exact-step replay runs must use independent process "
                "instances"
            )
        run_artifacts = [
            artifact
            for run in runs
            for artifact in run.referenced_artifacts
        ]
        if (
            len(set(run_artifacts)) != len(run_artifacts)
            or control_artifacts.intersection(run_artifacts)
        ):
            raise PcGoldenValidationError(
                "exact-step replay runs must use distinct evidence artifacts"
            )
        freeze = _strict_int(
            self.freeze_update,
            "exact-step freeze_update",
            minimum=0,
        )
        start = _strict_int(
            self.source_start_update,
            "exact-step source_start_update",
            minimum=0,
        )
        end = _strict_int(
            self.source_end_update,
            "exact-step source_end_update",
            minimum=0,
        )
        warmup = _strict_int(
            self.warmup_tick_count,
            "exact-step warmup_tick_count",
            minimum=1,
        )
        if not freeze < start <= end or warmup != start - freeze:
            raise PcGoldenValidationError(
                "exact-step source update range or warmup count is invalid"
            )
        tick_count = end - start + 1
        maximum_attempts = _strict_int(
            self.maximum_startup_attempts,
            "exact-step maximum_startup_attempts",
            minimum=1,
        )
        if maximum_attempts > 3:
            raise PcGoldenValidationError(
                "exact-step startup attempt budget must not exceed three"
            )
        if any(run.selected_attempt > maximum_attempts for run in runs):
            raise PcGoldenValidationError(
                "exact-step selected attempt exceeds the frozen budget"
            )
        if self.sample_phase != "frozen_post_replay_update_barrier":
            raise PcGoldenValidationError(
                "exact-step sample_phase is unsupported"
            )
        if self.pixel_comparison != "full_800x600_bgra_exact":
            raise PcGoldenValidationError(
                "exact-step pixel comparison must be full BGRA exact"
            )
        for run in runs:
            if (
                (run.video.width, run.video.height) != (800, 600)
                or run.video.codec != "ffv1"
                or run.video.pixel_format != "bgra"
                or run.video.time_base != (1, 100)
                or run.video.nominal_fps != (100, 1)
                or run.video.first_pts != 0
                or run.video.last_pts != tick_count - 1
                or run.video.frame_count != tick_count
                or not run.video.cfr
                or run.video.dropped_frames != 0
                or run.video.duplicate_frames != 0
                or run.video.timing_mode != "integer_cfr_grid"
                or run.clock.tick_end != tick_count - 1
                or run.clock.tick0_video_pts != 0
                or run.clock.mapping_kind != "per_tick_pts_table"
                or run.clock.uncertainty_ticks != 0.0
            ):
                raise PcGoldenValidationError(
                    "exact-step replay video/clock does not cover the exact "
                    "native update range"
                )
        object.__setattr__(self, "runs", runs)
        object.__setattr__(self, "freeze_update", freeze)
        object.__setattr__(self, "source_start_update", start)
        object.__setattr__(self, "source_end_update", end)
        object.__setattr__(self, "warmup_tick_count", warmup)
        object.__setattr__(
            self,
            "maximum_startup_attempts",
            maximum_attempts,
        )

    @property
    def referenced_artifacts(self) -> frozenset[str]:
        return frozenset(
            {
                self.preregistration_artifact,
                self.execution_binding_artifact,
                self.comparison_artifact,
                self.pre_snapshot_artifact,
                self.host_pre_snapshot_artifact,
                *(
                    artifact
                    for run in self.runs
                    for artifact in run.referenced_artifacts
                ),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "preregistration_artifact": self.preregistration_artifact,
            "execution_binding_artifact": self.execution_binding_artifact,
            "comparison_artifact": self.comparison_artifact,
            "pre_snapshot_artifact": self.pre_snapshot_artifact,
            "host_pre_snapshot_artifact": self.host_pre_snapshot_artifact,
            "runs": [run.to_dict() for run in self.runs],
            "freeze_update": self.freeze_update,
            "source_start_update": self.source_start_update,
            "source_end_update": self.source_end_update,
            "warmup_tick_count": self.warmup_tick_count,
            "maximum_startup_attempts": self.maximum_startup_attempts,
            "sample_phase": self.sample_phase,
            "pixel_comparison": self.pixel_comparison,
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> "ExactStepReplayContract":
        data = _mapping(value, "exact_step_replay")
        fields = {
            "preregistration_artifact",
            "execution_binding_artifact",
            "comparison_artifact",
            "pre_snapshot_artifact",
            "host_pre_snapshot_artifact",
            "runs",
            "freeze_update",
            "source_start_update",
            "source_end_update",
            "warmup_tick_count",
            "maximum_startup_attempts",
            "sample_phase",
            "pixel_comparison",
        }
        _check_keys(data, required=fields, name="exact_step_replay")
        return cls(
            preregistration_artifact=data["preregistration_artifact"],
            execution_binding_artifact=data["execution_binding_artifact"],
            comparison_artifact=data["comparison_artifact"],
            pre_snapshot_artifact=data["pre_snapshot_artifact"],
            host_pre_snapshot_artifact=data["host_pre_snapshot_artifact"],
            runs=tuple(
                ExactStepRunContract.from_dict(
                    _mapping(item, "exact-step run")
                )
                for item in _sequence(data["runs"], "exact_step_replay.runs")
            ),
            freeze_update=data["freeze_update"],
            source_start_update=data["source_start_update"],
            source_end_update=data["source_end_update"],
            warmup_tick_count=data["warmup_tick_count"],
            maximum_startup_attempts=data["maximum_startup_attempts"],
            sample_phase=data["sample_phase"],
            pixel_comparison=data["pixel_comparison"],
        )


def _legacy_capture_contract_fingerprint(
    *,
    case_id: str,
    scenario: Scenario,
    pc_environment_fingerprint: str,
    artifacts: Mapping[str, ArtifactSpec],
    input_timeline: InputTimeline,
    trace_artifact: str,
    video: VideoMetadata,
    clock: TickClock,
    coordinates: CoordinateCalibration,
    coverage: Sequence[CoverageRange],
    comparison_contract: ComparisonContract,
) -> str:
    """Recompute the exact v3/v1 contract for legacy read compatibility."""

    raw_artifacts = _mapping(artifacts, "artifacts")
    evidence_artifacts = {
        _nonempty_string(name, "artifact name"): artifact.to_dict()
        for name, artifact in sorted(raw_artifacts.items())
        if name != trace_artifact
    }
    ordered_coverage = sorted(
        tuple(coverage),
        key=lambda item: (
            item.channel,
            item.start_tick,
            item.end_tick,
        ),
    )
    payload = {
        "schema": CAPTURE_CONTRACT_SCHEMA,
        "version": 1,
        "case_id": _nonempty_string(case_id, "case_id"),
        "scenario": scenario.to_dict(),
        "pc_environment_fingerprint": _sha256(
            pc_environment_fingerprint,
            "pc_environment_fingerprint",
        ),
        "evidence_artifacts": evidence_artifacts,
        "input_timeline": input_timeline.to_dict(),
        "trace_artifact": _nonempty_string(
            trace_artifact,
            "trace_artifact",
        ),
        "video": video.to_dict(),
        "clock": clock.to_dict(),
        "coordinates": coordinates.to_dict(),
        "coverage": [item.to_dict() for item in ordered_coverage],
        "comparison_contract": comparison_contract.to_dict(),
    }
    return canonical_sha256(payload)


def capture_contract_fingerprint(
    *,
    case_id: str,
    scenario: Scenario,
    pc_environment_fingerprint: str,
    artifacts: Mapping[str, ArtifactSpec],
    input_timeline: InputTimeline,
    trace_artifact: str,
    video: VideoMetadata,
    clock: TickClock,
    coordinates: CoordinateCalibration,
    coverage: Sequence[CoverageRange],
    comparison_contract: ComparisonContract,
    excluded_artifacts: Sequence[str] | None = None,
    save_transaction: SaveTransactionContract | None = None,
    replay_determinism: ReplayDeterminismContract | None = None,
) -> str:
    """Hash every non-trace fact that determines capture interpretation.

    Each canonical trace embeds this fingerprint and therefore must be
    excluded from the artifact-spec portion of the hash.  The closed,
    validated exclusion list is itself hashed.  All non-trace artifacts,
    including save snapshots, process evidence and replay videos, remain
    covered.
    """

    normalized_case_id = _nonempty_string(case_id, "case_id")
    if not isinstance(scenario, Scenario):
        raise PcGoldenValidationError("scenario must be a Scenario")
    environment_fingerprint = _sha256(
        pc_environment_fingerprint,
        "pc_environment_fingerprint",
    )
    if not isinstance(input_timeline, InputTimeline):
        raise PcGoldenValidationError(
            "input_timeline must be an InputTimeline"
        )
    normalized_trace_artifact = _nonempty_string(
        trace_artifact,
        "trace_artifact",
    )
    if not isinstance(video, VideoMetadata):
        raise PcGoldenValidationError("video must be VideoMetadata")
    if not isinstance(clock, TickClock):
        raise PcGoldenValidationError("clock must be a TickClock")
    if not isinstance(coordinates, CoordinateCalibration):
        raise PcGoldenValidationError(
            "coordinates must be CoordinateCalibration"
        )
    if not isinstance(comparison_contract, ComparisonContract):
        raise PcGoldenValidationError(
            "comparison_contract must be ComparisonContract"
        )
    if save_transaction is not None and not isinstance(
        save_transaction,
        SaveTransactionContract,
    ):
        raise PcGoldenValidationError(
            "save_transaction must be a SaveTransactionContract or null"
        )
    if replay_determinism is not None and not isinstance(
        replay_determinism,
        ReplayDeterminismContract,
    ):
        raise PcGoldenValidationError(
            "replay_determinism must be a ReplayDeterminismContract or null"
        )

    expected_exclusions = {normalized_trace_artifact}
    if replay_determinism is not None:
        expected_exclusions.update(
            run.trace_artifact for run in replay_determinism.runs
        )
    if excluded_artifacts is None:
        normalized_exclusions = tuple(sorted(expected_exclusions))
    else:
        exclusion_items = _sequence(
            excluded_artifacts,
            "capture_contract_excluded_artifacts",
        )
        normalized_exclusions = tuple(
            sorted(
                _nonempty_string(
                    item,
                    "capture_contract_excluded_artifact",
                )
                for item in exclusion_items
            )
        )
        if len(set(normalized_exclusions)) != len(normalized_exclusions):
            raise PcGoldenValidationError(
                "capture contract exclusion list must not contain duplicates"
            )
        if set(normalized_exclusions) != expected_exclusions:
            raise PcGoldenValidationError(
                "capture contract exclusion list must contain exactly the "
                "referenced canonical trace artifacts"
            )

    raw_artifacts = _mapping(artifacts, "artifacts")
    artifact_names: set[str] = set()
    evidence_artifacts: dict[str, dict[str, Any]] = {}
    for name, artifact in sorted(raw_artifacts.items()):
        artifact_name = _nonempty_string(name, "artifact name")
        if not isinstance(artifact, ArtifactSpec):
            raise PcGoldenValidationError(
                f"artifacts[{artifact_name!r}] must be an ArtifactSpec"
            )
        artifact_names.add(artifact_name)
        if artifact_name not in normalized_exclusions:
            evidence_artifacts[artifact_name] = artifact.to_dict()
    required_evidence = {
        input_timeline.artifact,
        video.artifact,
        clock.tick_map_artifact,
        coordinates.calibration_artifact,
    }
    contract_references: set[str] = set()
    if save_transaction is not None:
        contract_references.update(save_transaction.referenced_artifacts)
    if replay_determinism is not None:
        contract_references.update(replay_determinism.referenced_artifacts)
    missing_contract_references = contract_references.difference(
        artifact_names,
        normalized_exclusions,
    )
    if missing_contract_references:
        raise PcGoldenValidationError(
            "capture contract references unknown protocol artifacts: "
            + ", ".join(sorted(missing_contract_references))
        )
    missing_evidence = required_evidence.difference(evidence_artifacts)
    if missing_evidence:
        raise PcGoldenValidationError(
            "capture contract references unknown non-trace artifacts: "
            + ", ".join(sorted(missing_evidence))
        )

    coverage_items = tuple(coverage)
    if any(not isinstance(item, CoverageRange) for item in coverage_items):
        raise PcGoldenValidationError(
            "coverage must contain only CoverageRange objects"
        )
    ordered_coverage = sorted(
        coverage_items,
        key=lambda item: (
            item.channel,
            item.start_tick,
            item.end_tick,
        ),
    )
    payload = {
        "schema": CAPTURE_CONTRACT_SCHEMA,
        "version": CAPTURE_CONTRACT_VERSION,
        "case_id": normalized_case_id,
        "scenario": scenario.to_dict(),
        "pc_environment_fingerprint": environment_fingerprint,
        "evidence_artifacts": evidence_artifacts,
        "excluded_artifacts": list(normalized_exclusions),
        "input_timeline": input_timeline.to_dict(),
        "trace_artifact": normalized_trace_artifact,
        "video": video.to_dict(),
        "clock": clock.to_dict(),
        "coordinates": coordinates.to_dict(),
        "coverage": [item.to_dict() for item in ordered_coverage],
        "comparison_contract": comparison_contract.to_dict(),
        "save_transaction": (
            save_transaction.to_dict()
            if save_transaction is not None
            else None
        ),
        "replay_determinism": (
            replay_determinism.to_dict()
            if replay_determinism is not None
            else None
        ),
    }
    return canonical_sha256(payload)


def exact_step_capture_contract_fingerprint(
    *,
    case_id: str,
    scenario: Scenario,
    pc_environment_fingerprint: str,
    artifacts: Mapping[str, ArtifactSpec],
    input_timeline: InputTimeline,
    trace_artifact: str,
    video: VideoMetadata,
    clock: TickClock,
    coordinates: CoordinateCalibration,
    coverage: Sequence[CoverageRange],
    comparison_contract: ComparisonContract,
    exact_step_replay: ExactStepReplayContract,
    excluded_artifacts: Sequence[str] | None = None,
) -> str:
    """Hash the independent v5 exact-step capture interpretation.

    This has a separate schema/domain from the v4 continuous-DXGI contract so
    an exact-step package can never be mistaken for raw cadence evidence.
    Canonical traces remain the only excluded artifacts because they embed the
    resulting fingerprint.
    """

    normalized_case_id = _nonempty_string(case_id, "case_id")
    if not isinstance(scenario, Scenario):
        raise PcGoldenValidationError("scenario must be a Scenario")
    environment_fingerprint = _sha256(
        pc_environment_fingerprint,
        "pc_environment_fingerprint",
    )
    if not isinstance(input_timeline, InputTimeline):
        raise PcGoldenValidationError(
            "input_timeline must be an InputTimeline"
        )
    normalized_trace_artifact = _nonempty_string(
        trace_artifact,
        "trace_artifact",
    )
    if not isinstance(video, VideoMetadata):
        raise PcGoldenValidationError("video must be VideoMetadata")
    if not isinstance(clock, TickClock):
        raise PcGoldenValidationError("clock must be a TickClock")
    if not isinstance(coordinates, CoordinateCalibration):
        raise PcGoldenValidationError(
            "coordinates must be CoordinateCalibration"
        )
    if not isinstance(comparison_contract, ComparisonContract):
        raise PcGoldenValidationError(
            "comparison_contract must be ComparisonContract"
        )
    if not isinstance(exact_step_replay, ExactStepReplayContract):
        raise PcGoldenValidationError(
            "exact_step_replay must be an ExactStepReplayContract"
        )

    expected_exclusions = {
        normalized_trace_artifact,
        *(run.trace_artifact for run in exact_step_replay.runs),
    }
    if excluded_artifacts is None:
        normalized_exclusions = tuple(sorted(expected_exclusions))
    else:
        normalized_exclusions = tuple(
            sorted(
                _nonempty_string(
                    item,
                    "capture_contract_excluded_artifact",
                )
                for item in _sequence(
                    excluded_artifacts,
                    "capture_contract_excluded_artifacts",
                )
            )
        )
        if len(set(normalized_exclusions)) != len(normalized_exclusions):
            raise PcGoldenValidationError(
                "capture contract exclusion list must not contain duplicates"
            )
        if set(normalized_exclusions) != expected_exclusions:
            raise PcGoldenValidationError(
                "exact-step capture exclusions must contain exactly the "
                "canonical trace artifacts"
            )

    raw_artifacts = _mapping(artifacts, "artifacts")
    artifact_names: set[str] = set()
    evidence_artifacts: dict[str, dict[str, Any]] = {}
    for name, artifact in sorted(raw_artifacts.items()):
        artifact_name = _nonempty_string(name, "artifact name")
        if not isinstance(artifact, ArtifactSpec):
            raise PcGoldenValidationError(
                f"artifacts[{artifact_name!r}] must be an ArtifactSpec"
            )
        artifact_names.add(artifact_name)
        if artifact_name not in normalized_exclusions:
            evidence_artifacts[artifact_name] = artifact.to_dict()
    required_evidence = {
        input_timeline.artifact,
        video.artifact,
        clock.tick_map_artifact,
        coordinates.calibration_artifact,
    }
    missing_contract_references = (
        exact_step_replay.referenced_artifacts.difference(
            artifact_names,
            normalized_exclusions,
        )
    )
    if missing_contract_references:
        raise PcGoldenValidationError(
            "exact-step contract references unknown protocol artifacts: "
            + ", ".join(sorted(missing_contract_references))
        )
    missing_evidence = required_evidence.difference(evidence_artifacts)
    if missing_evidence:
        raise PcGoldenValidationError(
            "exact-step contract references unknown non-trace artifacts: "
            + ", ".join(sorted(missing_evidence))
        )
    coverage_items = tuple(coverage)
    if any(not isinstance(item, CoverageRange) for item in coverage_items):
        raise PcGoldenValidationError(
            "coverage must contain only CoverageRange objects"
        )
    ordered_coverage = sorted(
        coverage_items,
        key=lambda item: (
            item.channel,
            item.start_tick,
            item.end_tick,
        ),
    )
    return canonical_sha256(
        {
            "schema": EXACT_STEP_CAPTURE_CONTRACT_SCHEMA,
            "version": EXACT_STEP_CAPTURE_CONTRACT_VERSION,
            "case_id": normalized_case_id,
            "scenario": scenario.to_dict(),
            "pc_environment_fingerprint": environment_fingerprint,
            "evidence_artifacts": evidence_artifacts,
            "excluded_artifacts": list(normalized_exclusions),
            "input_timeline": input_timeline.to_dict(),
            "trace_artifact": normalized_trace_artifact,
            "video": video.to_dict(),
            "clock": clock.to_dict(),
            "coordinates": coordinates.to_dict(),
            "coverage": [item.to_dict() for item in ordered_coverage],
            "comparison_contract": comparison_contract.to_dict(),
            "exact_step_replay": exact_step_replay.to_dict(),
        }
    )


def evidence_set_fingerprint(
    *,
    capture_contract_fingerprint_value: str,
    artifacts: Mapping[str, ArtifactSpec],
) -> str:
    """Hash the final artifact inventory, including all canonical traces."""

    contract_fingerprint = _sha256(
        capture_contract_fingerprint_value,
        "capture_contract_fingerprint",
    )
    raw_artifacts = _mapping(artifacts, "artifacts")
    normalized_artifacts: dict[str, dict[str, Any]] = {}
    for name, artifact in sorted(raw_artifacts.items()):
        artifact_name = _nonempty_string(name, "artifact name")
        if not isinstance(artifact, ArtifactSpec):
            raise PcGoldenValidationError(
                f"artifacts[{artifact_name!r}] must be an ArtifactSpec"
            )
        normalized_artifacts[artifact_name] = artifact.to_dict()
    if not normalized_artifacts:
        raise PcGoldenValidationError(
            "evidence set artifact inventory must not be empty"
        )
    return canonical_sha256(
        {
            "schema": EVIDENCE_SET_SCHEMA,
            "version": EVIDENCE_SET_VERSION,
            "capture_contract_fingerprint": contract_fingerprint,
            "artifacts": normalized_artifacts,
        }
    )


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    """Basic result envelope used by readiness checks and future diff code."""

    status: ComparisonStatus
    reasons: tuple[str, ...] = ()
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "status",
            _enum_value(
                self.status,
                ComparisonStatus,
                "comparison.status",
            ),
        )
        reason_items = _sequence(self.reasons, "comparison.reasons")
        reasons = tuple(
            _nonempty_string(reason, "comparison reason")
            for reason in reason_items
        )
        if self.status is not ComparisonStatus.PASS and not reasons:
            raise PcGoldenValidationError(
                "FAIL and INCOMPARABLE results require at least one reason"
            )
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(
            self,
            "metrics",
            _frozen_json_mapping(self.metrics, "comparison.metrics"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "reasons": list(self.reasons),
            "metrics": _plain_json_value(self.metrics),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ComparisonResult":
        data = _mapping(value, "comparison result")
        _check_keys(
            data,
            required={"status", "reasons", "metrics"},
            name="comparison result",
        )
        return cls(
            status=data["status"],
            reasons=tuple(_sequence(data["reasons"], "comparison.reasons")),
            metrics=data["metrics"],
        )


@dataclass(frozen=True, slots=True)
class PcGoldenManifest:
    """Validated v4/v5 manifest, with read-only v3 compatibility."""

    case_id: str
    scenario: Scenario
    pc_environment: PcEnvironment
    pc_environment_fingerprint: str
    artifacts: Mapping[str, ArtifactSpec]
    input_timeline: InputTimeline
    trace_artifact: str
    video: VideoMetadata
    clock: TickClock
    coordinates: CoordinateCalibration
    coverage: tuple[CoverageRange, ...]
    comparison_contract: ComparisonContract
    producer: Mapping[str, Any] = field(default_factory=dict)
    manifest_version: int = MANIFEST_VERSION
    save_transaction: SaveTransactionContract | None = None
    replay_determinism: ReplayDeterminismContract | None = None
    exact_step_replay: ExactStepReplayContract | None = None
    capture_contract_excluded_artifacts: tuple[str, ...] = field(init=False)
    capture_contract_fingerprint: str = field(init=False)
    evidence_set_fingerprint: str | None = field(init=False)

    def __post_init__(self) -> None:
        version = _strict_int(
            self.manifest_version,
            "manifest.version",
            minimum=1,
        )
        if version not in {
            LEGACY_MANIFEST_VERSION,
            MANIFEST_VERSION,
            EXACT_STEP_MANIFEST_VERSION,
        }:
            raise PcGoldenValidationError(
                f"unsupported manifest version: {version}"
            )
        object.__setattr__(self, "manifest_version", version)
        object.__setattr__(
            self,
            "case_id",
            _nonempty_string(self.case_id, "case_id"),
        )
        if not isinstance(self.scenario, Scenario):
            raise PcGoldenValidationError("scenario must be a Scenario")
        if not isinstance(self.pc_environment, PcEnvironment):
            raise PcGoldenValidationError(
                "pc_environment must be a PcEnvironment"
            )
        fingerprint = _sha256(
            self.pc_environment_fingerprint,
            "pc_environment_fingerprint",
        )
        if fingerprint != self.pc_environment.fingerprint:
            raise PcGoldenValidationError(
                "pc_environment_fingerprint does not match canonical "
                "pc_environment content"
            )
        object.__setattr__(
            self,
            "pc_environment_fingerprint",
            fingerprint,
        )
        if self.scenario.profile_mode != self.pc_environment.profile_mode:
            raise PcGoldenValidationError(
                "scenario and pc_environment profile_mode differ"
            )

        raw_artifacts = _mapping(self.artifacts, "artifacts")
        if not raw_artifacts:
            raise PcGoldenValidationError("artifacts must not be empty")
        artifacts: dict[str, ArtifactSpec] = {}
        for name, artifact in sorted(raw_artifacts.items()):
            artifact_name = _nonempty_string(name, "artifact name")
            if not isinstance(artifact, ArtifactSpec):
                raise PcGoldenValidationError(
                    f"artifacts[{artifact_name!r}] must be an ArtifactSpec"
                )
            artifacts[artifact_name] = artifact
        artifact_paths = [artifact.path for artifact in artifacts.values()]
        if len(set(artifact_paths)) != len(artifact_paths):
            raise PcGoldenValidationError(
                "artifacts must not reuse the same relative path"
            )
        object.__setattr__(
            self,
            "artifacts",
            MappingProxyType(artifacts),
        )
        trace_artifact = _nonempty_string(
            self.trace_artifact,
            "trace_artifact",
        )
        object.__setattr__(self, "trace_artifact", trace_artifact)

        if not isinstance(self.video, VideoMetadata):
            raise PcGoldenValidationError("video must be VideoMetadata")
        if not isinstance(self.input_timeline, InputTimeline):
            raise PcGoldenValidationError(
                "input_timeline must be an InputTimeline"
            )
        if not isinstance(self.clock, TickClock):
            raise PcGoldenValidationError("clock must be TickClock")
        if not isinstance(self.coordinates, CoordinateCalibration):
            raise PcGoldenValidationError(
                "coordinates must be CoordinateCalibration"
            )
        if not isinstance(self.comparison_contract, ComparisonContract):
            raise PcGoldenValidationError(
                "comparison_contract must be ComparisonContract"
            )
        if self.manifest_version == LEGACY_MANIFEST_VERSION:
            if (
                self.save_transaction is not None
                or self.replay_determinism is not None
                or self.exact_step_replay is not None
            ):
                raise PcGoldenValidationError(
                    "legacy v3 manifests cannot declare protocol evidence"
                )
        elif self.manifest_version == MANIFEST_VERSION:
            if self.exact_step_replay is not None:
                raise PcGoldenValidationError(
                    "v4 manifests cannot declare v5 exact-step evidence"
                )
            if self.save_transaction is not None and not isinstance(
                self.save_transaction,
                SaveTransactionContract,
            ):
                raise PcGoldenValidationError(
                    "save_transaction must be a SaveTransactionContract "
                    "or null"
                )
            if self.replay_determinism is not None and not isinstance(
                self.replay_determinism,
                ReplayDeterminismContract,
            ):
                raise PcGoldenValidationError(
                    "replay_determinism must be a "
                    "ReplayDeterminismContract or null"
                )
            if (
                (self.save_transaction is None)
                != (self.replay_determinism is None)
            ):
                raise PcGoldenValidationError(
                    "v4 save_transaction and replay_determinism must be "
                    "declared together"
                )
            if (
                self.save_transaction is not None
                and self.replay_determinism is not None
            ):
                save_run_ids = tuple(
                    run.run_id for run in self.save_transaction.runs
                )
                replay_run_ids = tuple(
                    run.run_id for run in self.replay_determinism.runs
                )
                if replay_run_ids != save_run_ids:
                    raise PcGoldenValidationError(
                        "save and replay contracts must use the same ordered "
                        "run_id values"
                    )
                primary = self.replay_determinism.runs[0]
                if (
                    primary.trace_artifact != trace_artifact
                    or primary.video != self.video
                    or primary.clock != self.clock
                ):
                    raise PcGoldenValidationError(
                        "the first replay run must be the manifest primary "
                        "trace/video/clock"
                    )
        else:
            if (
                self.save_transaction is not None
                or self.replay_determinism is not None
            ):
                raise PcGoldenValidationError(
                    "v5 exact-step manifests cannot relabel v4 continuous "
                    "DXGI evidence"
                )
            if not isinstance(
                self.exact_step_replay,
                ExactStepReplayContract,
            ):
                raise PcGoldenValidationError(
                    "v5 manifests require an ExactStepReplayContract"
                )
            primary = self.exact_step_replay.runs[0]
            if (
                primary.trace_artifact != trace_artifact
                or primary.video != self.video
                or primary.clock != self.clock
            ):
                raise PcGoldenValidationError(
                    "the first exact-step run must be the manifest primary "
                    "trace/video/clock"
                )

        references = {
            trace_artifact,
            self.input_timeline.artifact,
            self.video.artifact,
            self.clock.tick_map_artifact,
            self.coordinates.calibration_artifact,
        }
        if self.save_transaction is not None:
            references.update(self.save_transaction.referenced_artifacts)
        if self.replay_determinism is not None:
            references.update(self.replay_determinism.referenced_artifacts)
        if self.exact_step_replay is not None:
            references.update(self.exact_step_replay.referenced_artifacts)
        missing_references = references.difference(artifacts)
        if missing_references:
            raise PcGoldenValidationError(
                "manifest references unknown artifacts: "
                + ", ".join(sorted(missing_references))
            )
        if (
            self.coordinates.raw_width,
            self.coordinates.raw_height,
        ) != (self.video.width, self.video.height):
            raise PcGoldenValidationError(
                "coordinate raw dimensions must match video dimensions"
            )
        if not (
            self.video.first_pts
            <= self.clock.tick0_video_pts
            <= self.video.last_pts
        ):
            raise PcGoldenValidationError(
                "clock.tick0_video_pts is outside the video time range"
            )
        if self.replay_determinism is not None:
            for run in self.replay_determinism.runs:
                if (
                    run.video.width,
                    run.video.height,
                ) != (
                    self.coordinates.raw_width,
                    self.coordinates.raw_height,
                ):
                    raise PcGoldenValidationError(
                        "every replay video must use the calibrated raw "
                        "dimensions"
                    )
                if run.clock.tick_end != self.clock.tick_end:
                    raise PcGoldenValidationError(
                        "every replay clock must end at the primary tick"
                    )
        if self.exact_step_replay is not None:
            for run in self.exact_step_replay.runs:
                if (
                    run.video.width,
                    run.video.height,
                ) != (
                    self.coordinates.raw_width,
                    self.coordinates.raw_height,
                ):
                    raise PcGoldenValidationError(
                        "every exact-step video must use the calibrated raw "
                        "dimensions"
                    )
                if run.clock.tick_end != self.clock.tick_end:
                    raise PcGoldenValidationError(
                        "every exact-step clock must end at the primary tick"
                    )

        coverage = tuple(self.coverage)
        if not coverage:
            raise PcGoldenValidationError("coverage must not be empty")
        if any(not isinstance(item, CoverageRange) for item in coverage):
            raise PcGoldenValidationError(
                "coverage must contain only CoverageRange objects"
            )
        if not any(item.required for item in coverage):
            raise PcGoldenValidationError(
                "coverage must declare at least one required range"
            )
        ordered = tuple(
            sorted(
                coverage,
                key=lambda item: (
                    item.channel,
                    item.start_tick,
                    item.end_tick,
                ),
            )
        )
        previous_by_channel: dict[str, CoverageRange] = {}
        for item in ordered:
            if item.end_tick > self.clock.tick_end:
                raise PcGoldenValidationError(
                    f"coverage for {item.channel!r} exceeds clock.tick_end"
                )
            previous = previous_by_channel.get(item.channel)
            if previous is None:
                if item.start_tick != 0:
                    raise PcGoldenValidationError(
                        f"coverage for {item.channel!r} must start at tick zero"
                    )
            else:
                if item.start_tick <= previous.end_tick:
                    raise PcGoldenValidationError(
                        f"coverage ranges overlap for channel {item.channel!r}"
                    )
                if item.start_tick != previous.end_tick + 1:
                    raise PcGoldenValidationError(
                        f"coverage ranges contain a gap for channel "
                        f"{item.channel!r}"
                    )
            previous_by_channel[item.channel] = item
        for channel, final_range in previous_by_channel.items():
            if final_range.end_tick != self.clock.tick_end:
                raise PcGoldenValidationError(
                    f"coverage for {channel!r} must end at clock.tick_end"
                )
        object.__setattr__(self, "coverage", ordered)
        object.__setattr__(
            self,
            "producer",
            _frozen_json_mapping(self.producer, "producer"),
        )
        excluded_artifacts = {self.trace_artifact}
        if self.replay_determinism is not None:
            excluded_artifacts.update(
                run.trace_artifact
                for run in self.replay_determinism.runs
            )
        if self.exact_step_replay is not None:
            excluded_artifacts.update(
                run.trace_artifact for run in self.exact_step_replay.runs
            )
        normalized_exclusions = tuple(sorted(excluded_artifacts))
        object.__setattr__(
            self,
            "capture_contract_excluded_artifacts",
            normalized_exclusions,
        )
        if self.manifest_version == LEGACY_MANIFEST_VERSION:
            contract_fingerprint = _legacy_capture_contract_fingerprint(
                case_id=self.case_id,
                scenario=self.scenario,
                pc_environment_fingerprint=(
                    self.pc_environment_fingerprint
                ),
                artifacts=self.artifacts,
                input_timeline=self.input_timeline,
                trace_artifact=self.trace_artifact,
                video=self.video,
                clock=self.clock,
                coordinates=self.coordinates,
                coverage=self.coverage,
                comparison_contract=self.comparison_contract,
            )
        elif self.manifest_version == MANIFEST_VERSION:
            contract_fingerprint = capture_contract_fingerprint(
                case_id=self.case_id,
                scenario=self.scenario,
                pc_environment_fingerprint=(
                    self.pc_environment_fingerprint
                ),
                artifacts=self.artifacts,
                input_timeline=self.input_timeline,
                trace_artifact=self.trace_artifact,
                video=self.video,
                clock=self.clock,
                coordinates=self.coordinates,
                coverage=self.coverage,
                comparison_contract=self.comparison_contract,
                excluded_artifacts=normalized_exclusions,
                save_transaction=self.save_transaction,
                replay_determinism=self.replay_determinism,
            )
        else:
            assert self.exact_step_replay is not None
            contract_fingerprint = exact_step_capture_contract_fingerprint(
                case_id=self.case_id,
                scenario=self.scenario,
                pc_environment_fingerprint=(
                    self.pc_environment_fingerprint
                ),
                artifacts=self.artifacts,
                input_timeline=self.input_timeline,
                trace_artifact=self.trace_artifact,
                video=self.video,
                clock=self.clock,
                coordinates=self.coordinates,
                coverage=self.coverage,
                comparison_contract=self.comparison_contract,
                exact_step_replay=self.exact_step_replay,
                excluded_artifacts=normalized_exclusions,
            )
        object.__setattr__(
            self,
            "capture_contract_fingerprint",
            contract_fingerprint,
        )
        object.__setattr__(
            self,
            "evidence_set_fingerprint",
            (
                evidence_set_fingerprint(
                    capture_contract_fingerprint_value=contract_fingerprint,
                    artifacts=self.artifacts,
                )
                if self.manifest_version
                in {MANIFEST_VERSION, EXACT_STEP_MANIFEST_VERSION}
                else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": MANIFEST_SCHEMA,
            "version": self.manifest_version,
            "case_id": self.case_id,
            "scenario": self.scenario.to_dict(),
            "pc_environment": self.pc_environment.to_dict(),
            "pc_environment_fingerprint": self.pc_environment_fingerprint,
            "artifacts": {
                name: artifact.to_dict()
                for name, artifact in self.artifacts.items()
            },
            "input_timeline": self.input_timeline.to_dict(),
            "trace_artifact": self.trace_artifact,
            "video": self.video.to_dict(),
            "clock": self.clock.to_dict(),
            "coordinates": self.coordinates.to_dict(),
            "coverage": [item.to_dict() for item in self.coverage],
            "comparison_contract": self.comparison_contract.to_dict(),
            "capture_contract_fingerprint": (
                self.capture_contract_fingerprint
            ),
            "producer": _plain_json_value(self.producer),
        }
        if self.manifest_version == MANIFEST_VERSION:
            payload.update(
                {
                    "save_transaction": (
                        self.save_transaction.to_dict()
                        if self.save_transaction is not None
                        else None
                    ),
                    "replay_determinism": (
                        self.replay_determinism.to_dict()
                        if self.replay_determinism is not None
                        else None
                    ),
                    "capture_contract_excluded_artifacts": list(
                        self.capture_contract_excluded_artifacts
                    ),
                    "evidence_set_fingerprint": (
                        self.evidence_set_fingerprint
                    ),
                }
            )
        elif self.manifest_version == EXACT_STEP_MANIFEST_VERSION:
            assert self.exact_step_replay is not None
            payload.update(
                {
                    "exact_step_replay": self.exact_step_replay.to_dict(),
                    "capture_contract_excluded_artifacts": list(
                        self.capture_contract_excluded_artifacts
                    ),
                    "evidence_set_fingerprint": (
                        self.evidence_set_fingerprint
                    ),
                }
            )
        return payload

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=indent,
            separators=(",", ":") if indent is None else None,
        )

    def write_json(
        self,
        path: str | Path,
        *,
        indent: int | None = 2,
    ) -> None:
        Path(path).write_text(
            self.to_json(indent=indent) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PcGoldenManifest":
        data = _mapping(value, "PC golden manifest")
        if "schema" not in data or "version" not in data:
            missing = {"schema", "version"}.difference(data)
            raise PcGoldenValidationError(
                "PC golden manifest is missing fields: "
                + ", ".join(sorted(missing))
            )
        if data["schema"] != MANIFEST_SCHEMA:
            raise PcGoldenValidationError(
                f"unsupported manifest schema: {data['schema']!r}"
            )
        version = _strict_int(data["version"], "manifest.version")
        if version not in {
            LEGACY_MANIFEST_VERSION,
            MANIFEST_VERSION,
            EXACT_STEP_MANIFEST_VERSION,
        }:
            raise PcGoldenValidationError(
                f"unsupported manifest version: {version}"
            )
        required = {
            "schema",
            "version",
            "case_id",
            "scenario",
            "pc_environment",
            "pc_environment_fingerprint",
            "artifacts",
            "input_timeline",
            "trace_artifact",
            "video",
            "clock",
            "coordinates",
            "coverage",
            "comparison_contract",
            "capture_contract_fingerprint",
            "producer",
        }
        if version == MANIFEST_VERSION:
            required.update(
                {
                    "save_transaction",
                    "replay_determinism",
                    "capture_contract_excluded_artifacts",
                    "evidence_set_fingerprint",
                }
            )
        elif version == EXACT_STEP_MANIFEST_VERSION:
            required.update(
                {
                    "exact_step_replay",
                    "capture_contract_excluded_artifacts",
                    "evidence_set_fingerprint",
                }
            )
        _check_keys(data, required=required, name="PC golden manifest")
        artifacts_data = _mapping(data["artifacts"], "artifacts")
        artifacts = {
            name: ArtifactSpec.from_dict(
                _mapping(item, f"artifacts[{name!r}]")
            )
            for name, item in artifacts_data.items()
        }
        coverage = tuple(
            CoverageRange.from_dict(_mapping(item, "coverage item"))
            for item in _sequence(data["coverage"], "coverage")
        )
        save_transaction: SaveTransactionContract | None = None
        replay_determinism: ReplayDeterminismContract | None = None
        exact_step_replay: ExactStepReplayContract | None = None
        if version == MANIFEST_VERSION:
            if data["save_transaction"] is not None:
                save_transaction = SaveTransactionContract.from_dict(
                    _mapping(
                        data["save_transaction"],
                        "save_transaction",
                    )
                )
            if data["replay_determinism"] is not None:
                replay_determinism = ReplayDeterminismContract.from_dict(
                    _mapping(
                        data["replay_determinism"],
                        "replay_determinism",
                    )
                )
        elif version == EXACT_STEP_MANIFEST_VERSION:
            exact_step_replay = ExactStepReplayContract.from_dict(
                _mapping(
                    data["exact_step_replay"],
                    "exact_step_replay",
                )
            )
        manifest = cls(
            case_id=data["case_id"],
            scenario=Scenario.from_dict(
                _mapping(data["scenario"], "scenario")
            ),
            pc_environment=PcEnvironment.from_dict(
                _mapping(data["pc_environment"], "pc_environment")
            ),
            pc_environment_fingerprint=data[
                "pc_environment_fingerprint"
            ],
            artifacts=artifacts,
            input_timeline=InputTimeline.from_dict(
                _mapping(data["input_timeline"], "input_timeline")
            ),
            trace_artifact=data["trace_artifact"],
            video=VideoMetadata.from_dict(
                _mapping(data["video"], "video")
            ),
            clock=TickClock.from_dict(_mapping(data["clock"], "clock")),
            coordinates=CoordinateCalibration.from_dict(
                _mapping(data["coordinates"], "coordinates")
            ),
            coverage=coverage,
            comparison_contract=ComparisonContract.from_dict(
                _mapping(
                    data["comparison_contract"],
                    "comparison_contract",
                )
            ),
            producer=data["producer"],
            manifest_version=version,
            save_transaction=save_transaction,
            replay_determinism=replay_determinism,
            exact_step_replay=exact_step_replay,
        )
        declared_contract_fingerprint = _sha256(
            data["capture_contract_fingerprint"],
            "capture_contract_fingerprint",
        )
        if (
            declared_contract_fingerprint
            != manifest.capture_contract_fingerprint
        ):
            raise PcGoldenValidationError(
                "capture_contract_fingerprint does not match canonical "
                "capture interpretation"
            )
        if version in {MANIFEST_VERSION, EXACT_STEP_MANIFEST_VERSION}:
            declared_exclusions = tuple(
                _nonempty_string(
                    item,
                    "capture_contract_excluded_artifact",
                )
                for item in _sequence(
                    data["capture_contract_excluded_artifacts"],
                    "capture_contract_excluded_artifacts",
                )
            )
            if (
                declared_exclusions
                != manifest.capture_contract_excluded_artifacts
            ):
                raise PcGoldenValidationError(
                    "capture_contract_excluded_artifacts does not match the "
                    "closed canonical trace set"
                )
            declared_evidence_fingerprint = _sha256(
                data["evidence_set_fingerprint"],
                "evidence_set_fingerprint",
            )
            if (
                declared_evidence_fingerprint
                != manifest.evidence_set_fingerprint
            ):
                raise PcGoldenValidationError(
                    "evidence_set_fingerprint does not match the final "
                    "artifact inventory"
                )
        return manifest

    @classmethod
    def from_json(cls, text: str) -> "PcGoldenManifest":
        return cls.from_dict(
            _mapping(
                _json_loads_strict(text, "manifest"),
                "PC golden manifest",
            )
        )

    @classmethod
    def read_json(cls, path: str | Path) -> "PcGoldenManifest":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    def verify_artifacts(self, root: str | Path) -> Mapping[str, Path]:
        verified = {
            name: artifact.verify(root)
            for name, artifact in self.artifacts.items()
        }
        return MappingProxyType(verified)

    def coverage_readiness(self) -> ComparisonResult:
        reasons: list[str] = []
        if self.scenario.profile_mode != SUPPORTED_PROFILE_MODE:
            reasons.append(
                "profile mode is not supported by the current simulator: "
                f"{self.scenario.profile_mode!r}"
            )
        if self.scenario.mode != "adventure":
            reasons.append(
                "scenario mode is not supported by the current simulator: "
                f"{self.scenario.mode!r}"
            )
        if self.scenario.curve_index != 0:
            reasons.append(
                "only curve_index 0 is currently mapped to the original "
                "adventure-level contract"
            )
        if self.scenario.gun_index != 0:
            reasons.append(
                "only gun_index 0 is currently mapped to the original "
                "adventure-level contract"
            )
        if self.clock.uncertainty_ticks != 0.0:
            reasons.append(
                "clock uncertainty is non-zero; exact event ticks are unavailable"
            )
        if self.video.dropped_frames != 0:
            reasons.append("video declares dropped frames")
        if self.video.duplicate_frames != 0:
            reasons.append("video declares duplicate frames")
        for item in self.coverage:
            if (
                item.required
                and item.channel not in SUPPORTED_COVERAGE_CHANNELS
            ):
                reasons.append(
                    f"required channel {item.channel!r} is unknown to "
                    f"PC golden v{self.manifest_version}"
                )
            if item.required and item.status is not CoverageStatus.COMPLETE:
                reasons.append(
                    f"required channel {item.channel!r} is "
                    f"{item.status.value} on ticks "
                    f"{item.start_tick}-{item.end_tick}"
                )
        if reasons:
            return ComparisonResult(
                status=ComparisonStatus.INCOMPARABLE,
                reasons=tuple(reasons),
                metrics={},
            )
        return ComparisonResult(
            status=ComparisonStatus.PASS,
            reasons=(),
            metrics={},
        )


@dataclass(frozen=True, slots=True)
class FrameRef:
    """One presentation-order frame associated with a native tick."""

    frame_index: int
    pts: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "frame_index",
            _strict_int(self.frame_index, "frame_index", minimum=0),
        )
        object.__setattr__(self, "pts", _strict_int(self.pts, "frame pts"))

    def to_dict(self) -> dict[str, Any]:
        return {"frame_index": self.frame_index, "pts": self.pts}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FrameRef":
        data = _mapping(value, "frame reference")
        _check_keys(
            data,
            required={"frame_index", "pts"},
            name="frame reference",
        )
        return cls(frame_index=data["frame_index"], pts=data["pts"])


@dataclass(frozen=True, slots=True)
class GoldenInput:
    """One raw input edge/sample, ordered within its containing tick."""

    sequence: int
    kind: str
    pts: int
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sequence",
            _strict_int(self.sequence, "input.sequence", minimum=0),
        )
        object.__setattr__(
            self,
            "kind",
            _nonempty_string(self.kind, "input.kind"),
        )
        object.__setattr__(self, "pts", _strict_int(self.pts, "input.pts"))
        object.__setattr__(
            self,
            "payload",
            _frozen_json_mapping(self.payload, "input.payload"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "kind": self.kind,
            "pts": self.pts,
            "payload": _plain_json_value(self.payload),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GoldenInput":
        data = _mapping(value, "input")
        _check_keys(
            data,
            required={"sequence", "kind", "pts", "payload"},
            name="input",
        )
        return cls(
            sequence=data["sequence"],
            kind=data["kind"],
            pts=data["pts"],
            payload=data["payload"],
        )


@dataclass(frozen=True, slots=True)
class GoldenEvent:
    """One event annotation; sequence is optional when intra-tick order is unknown."""

    kind: str
    subjects: tuple[str, ...]
    payload: Mapping[str, Any]
    source: str
    sequence: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "kind",
            _nonempty_string(self.kind, "event.kind"),
        )
        if self.kind not in EVENT_KINDS:
            raise PcGoldenValidationError(
                f"event.kind is unknown to PC golden v3: {self.kind!r}"
            )
        subject_items = _sequence(self.subjects, "event.subjects")
        subjects = tuple(
            _nonempty_string(subject, "event subject")
            for subject in subject_items
        )
        if len(set(subjects)) != len(subjects):
            raise PcGoldenValidationError(
                "event.subjects must not contain duplicates"
            )
        object.__setattr__(self, "subjects", subjects)
        object.__setattr__(
            self,
            "payload",
            _frozen_json_mapping(self.payload, "event.payload"),
        )
        object.__setattr__(
            self,
            "source",
            _nonempty_string(self.source, "event.source"),
        )
        if self.sequence is not None:
            object.__setattr__(
                self,
                "sequence",
                _strict_int(
                    self.sequence,
                    "event.sequence",
                    minimum=0,
                ),
            )

    def to_dict(self) -> dict[str, Any]:
        result = {
            "kind": self.kind,
            "subjects": list(self.subjects),
            "payload": _plain_json_value(self.payload),
            "source": self.source,
        }
        if self.sequence is not None:
            result["sequence"] = self.sequence
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GoldenEvent":
        data = _mapping(value, "event")
        _check_keys(
            data,
            required={"kind", "subjects", "payload", "source"},
            optional={"sequence"},
            name="event",
        )
        return cls(
            kind=data["kind"],
            subjects=tuple(_sequence(data["subjects"], "event.subjects")),
            payload=data["payload"],
            source=data["source"],
            sequence=data.get("sequence"),
        )


@dataclass(frozen=True, slots=True)
class GoldenTick:
    """One baseline/post-update record in a PC golden trace."""

    tick: int
    frames: tuple[FrameRef, ...]
    inputs: tuple[GoldenInput, ...]
    events: tuple[GoldenEvent, ...]
    measurements: Mapping[str, Measurement]

    def __post_init__(self) -> None:
        tick = _strict_int(self.tick, "trace tick", minimum=0)
        object.__setattr__(self, "tick", tick)
        frames = tuple(self.frames)
        if any(not isinstance(item, FrameRef) for item in frames):
            raise PcGoldenValidationError(
                "frames must contain only FrameRef objects"
            )
        for previous, current in zip(frames, frames[1:]):
            if (
                current.frame_index <= previous.frame_index
                or current.pts < previous.pts
            ):
                raise PcGoldenValidationError(
                    "frames within a tick must have increasing indices and "
                    "non-decreasing PTS"
                )
        object.__setattr__(self, "frames", frames)

        inputs = tuple(self.inputs)
        if any(not isinstance(item, GoldenInput) for item in inputs):
            raise PcGoldenValidationError(
                "inputs must contain only GoldenInput objects"
            )
        for expected_sequence, item in enumerate(inputs):
            if item.sequence != expected_sequence:
                raise PcGoldenValidationError(
                    "input sequences must be contiguous and start at zero "
                    "within each tick"
                )
            if expected_sequence > 0 and item.pts < inputs[
                expected_sequence - 1
            ].pts:
                raise PcGoldenValidationError(
                    "input PTS must be non-decreasing within a tick"
                )
        object.__setattr__(self, "inputs", inputs)

        events = tuple(self.events)
        if any(not isinstance(item, GoldenEvent) for item in events):
            raise PcGoldenValidationError(
                "events must contain only GoldenEvent objects"
            )
        known_sequences = [
            item.sequence for item in events if item.sequence is not None
        ]
        if known_sequences != sorted(set(known_sequences)):
            raise PcGoldenValidationError(
                "known event sequences must be unique and increasing"
            )
        object.__setattr__(self, "events", events)

        if tick == 0 and (inputs or events):
            raise PcGoldenValidationError(
                "tick zero is a pre-input baseline and cannot contain "
                "transition inputs or events"
            )

        raw_measurements = _mapping(self.measurements, "measurements")
        measurements: dict[str, Measurement] = {}
        for channel, measurement in sorted(raw_measurements.items()):
            channel_name = _nonempty_string(channel, "measurement channel")
            if channel_name not in MEASUREMENT_CHANNELS:
                raise PcGoldenValidationError(
                    f"measurement channel is unknown to PC golden v3: "
                    f"{channel_name!r}"
                )
            if not isinstance(measurement, Measurement):
                raise PcGoldenValidationError(
                    f"measurements[{channel_name!r}] must be a Measurement"
                )
            if measurement.status in {
                MeasurementStatus.OBSERVED,
                MeasurementStatus.INFERRED,
            }:
                validate_measurement_value(
                    channel_name,
                    measurement.value,
                )
            measurements[channel_name] = measurement
        for count_channel, sequence_channels in (
            ("chain_count", ("chain_centers", "chain_waypoints")),
            ("projectile_count", ("projectile_centers",)),
        ):
            count_measurement = measurements.get(count_channel)
            if (
                count_measurement is None
                or count_measurement.status
                not in {
                    MeasurementStatus.OBSERVED,
                    MeasurementStatus.INFERRED,
                }
            ):
                continue
            for sequence_channel in sequence_channels:
                sequence_measurement = measurements.get(sequence_channel)
                if (
                    sequence_measurement is not None
                    and sequence_measurement.status
                    in {
                        MeasurementStatus.OBSERVED,
                        MeasurementStatus.INFERRED,
                    }
                    and len(sequence_measurement.value)
                    != count_measurement.value
                ):
                    raise PcGoldenValidationError(
                        f"measurement {count_channel!r} does not match "
                        f"the length of {sequence_channel!r}"
                    )
        object.__setattr__(
            self,
            "measurements",
            MappingProxyType(measurements),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tick": self.tick,
            "frames": [item.to_dict() for item in self.frames],
            "inputs": [item.to_dict() for item in self.inputs],
            "events": [item.to_dict() for item in self.events],
            "measurements": {
                channel: measurement.to_dict()
                for channel, measurement in self.measurements.items()
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GoldenTick":
        data = _mapping(value, "trace tick")
        _check_keys(
            data,
            required={"tick", "frames", "inputs", "events", "measurements"},
            name="trace tick",
        )
        measurements_data = _mapping(
            data["measurements"],
            "measurements",
        )
        return cls(
            tick=data["tick"],
            frames=tuple(
                FrameRef.from_dict(_mapping(item, "frame reference"))
                for item in _sequence(data["frames"], "frames")
            ),
            inputs=tuple(
                GoldenInput.from_dict(_mapping(item, "input"))
                for item in _sequence(data["inputs"], "inputs")
            ),
            events=tuple(
                GoldenEvent.from_dict(_mapping(item, "event"))
                for item in _sequence(data["events"], "events")
            ),
            measurements={
                channel: Measurement.from_dict(
                    _mapping(item, f"measurement {channel!r}")
                )
                for channel, item in measurements_data.items()
            },
        )


@dataclass(frozen=True, slots=True)
class PcGoldenTrace:
    """Contiguous tick-zero-based v2 NDJSON trace."""

    case_id: str
    pc_environment_fingerprint: str
    capture_contract_fingerprint: str
    records: tuple[GoldenTick, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "case_id",
            _nonempty_string(self.case_id, "trace case_id"),
        )
        object.__setattr__(
            self,
            "pc_environment_fingerprint",
            _sha256(
                self.pc_environment_fingerprint,
                "trace pc_environment_fingerprint",
            ),
        )
        object.__setattr__(
            self,
            "capture_contract_fingerprint",
            _sha256(
                self.capture_contract_fingerprint,
                "trace capture_contract_fingerprint",
            ),
        )
        records = tuple(self.records)
        if not records:
            raise PcGoldenValidationError(
                "trace must contain at least the tick-zero baseline"
            )
        for expected_tick, record in enumerate(records):
            if not isinstance(record, GoldenTick):
                raise PcGoldenValidationError(
                    "trace records must contain only GoldenTick objects"
                )
            if record.tick != expected_tick:
                raise PcGoldenValidationError(
                    "trace ticks must be contiguous and start at zero; "
                    f"expected {expected_tick}, got {record.tick}"
                )
        object.__setattr__(self, "records", records)

    @property
    def tick_end(self) -> int:
        return len(self.records) - 1

    def _header_dict(self) -> dict[str, Any]:
        return {
            "schema": TRACE_SCHEMA,
            "version": TRACE_VERSION,
            "case_id": self.case_id,
            "pc_environment_fingerprint": self.pc_environment_fingerprint,
            "capture_contract_fingerprint": (
                self.capture_contract_fingerprint
            ),
        }

    def to_ndjson(self) -> str:
        lines = [_canonical_json_text(self._header_dict())]
        lines.extend(
            _canonical_json_text(record.to_dict())
            for record in self.records
        )
        return "\n".join(lines) + "\n"

    @property
    def canonical_content_sha256(self) -> str:
        return (
            f"sha256:{hashlib.sha256(self.to_ndjson().encode('utf-8')).hexdigest()}"
        )

    def write_ndjson(self, path: str | Path) -> None:
        Path(path).write_text(self.to_ndjson(), encoding="utf-8")

    @classmethod
    def from_ndjson(cls, text: str) -> "PcGoldenTrace":
        raw_lines = text.splitlines()
        if not raw_lines:
            raise PcGoldenValidationError("trace NDJSON is empty")
        if any(not line.strip() for line in raw_lines):
            raise PcGoldenValidationError(
                "trace NDJSON must not contain blank lines"
            )
        header = _mapping(
            _json_loads_strict(raw_lines[0], "trace header"),
            "trace header",
        )
        _check_keys(
            header,
            required={
                "schema",
                "version",
                "case_id",
                "pc_environment_fingerprint",
                "capture_contract_fingerprint",
            },
            name="trace header",
        )
        if header["schema"] != TRACE_SCHEMA:
            raise PcGoldenValidationError(
                f"unsupported trace schema: {header['schema']!r}"
            )
        version = _strict_int(header["version"], "trace.version")
        if version != TRACE_VERSION:
            raise PcGoldenValidationError(
                f"unsupported trace version: {version}"
            )
        records = tuple(
            GoldenTick.from_dict(
                _mapping(
                    _json_loads_strict(line, f"trace line {line_number}"),
                    f"trace line {line_number}",
                )
            )
            for line_number, line in enumerate(raw_lines[1:], start=2)
        )
        trace = cls(
            case_id=header["case_id"],
            pc_environment_fingerprint=header[
                "pc_environment_fingerprint"
            ],
            capture_contract_fingerprint=header[
                "capture_contract_fingerprint"
            ],
            records=records,
        )
        if text != trace.to_ndjson():
            raise PcGoldenValidationError(
                "trace NDJSON is not in canonical v2 form"
            )
        return trace

    @classmethod
    def read_ndjson(cls, path: str | Path) -> "PcGoldenTrace":
        return cls.from_ndjson(Path(path).read_text(encoding="utf-8"))

    def validate_against_manifest(
        self,
        manifest: PcGoldenManifest,
    ) -> None:
        self.validate_against_run(
            manifest,
            trace_artifact=manifest.trace_artifact,
            video=manifest.video,
            clock=manifest.clock,
        )

    def validate_against_run(
        self,
        manifest: PcGoldenManifest,
        *,
        trace_artifact: str,
        video: VideoMetadata,
        clock: TickClock,
    ) -> None:
        """Validate one primary or replay trace against its run contract."""

        if not isinstance(manifest, PcGoldenManifest):
            raise TypeError("manifest must be a PcGoldenManifest")
        normalized_trace_artifact = _nonempty_string(
            trace_artifact,
            "trace_artifact",
        )
        if not isinstance(video, VideoMetadata):
            raise TypeError("video must be VideoMetadata")
        if not isinstance(clock, TickClock):
            raise TypeError("clock must be TickClock")
        if normalized_trace_artifact not in manifest.artifacts:
            raise PcGoldenValidationError(
                "trace artifact is not present in the manifest"
            )
        if self.case_id != manifest.case_id:
            raise PcGoldenValidationError(
                "trace case_id does not match manifest"
            )
        if (
            self.pc_environment_fingerprint
            != manifest.pc_environment_fingerprint
        ):
            raise PcGoldenValidationError(
                "trace PC environment fingerprint does not match manifest"
            )
        if (
            self.capture_contract_fingerprint
            != manifest.capture_contract_fingerprint
        ):
            raise PcGoldenValidationError(
                "trace capture-contract fingerprint does not match manifest"
            )
        if self.tick_end != clock.tick_end:
            raise PcGoldenValidationError(
                "trace final tick does not match clock.tick_end"
            )
        trace_spec = manifest.artifacts[normalized_trace_artifact]
        payload = self.to_ndjson().encode("utf-8")
        actual_bytes = len(payload)
        if actual_bytes != trace_spec.bytes:
            raise PcGoldenValidationError(
                "trace artifact byte count does not match canonical trace: "
                f"expected {trace_spec.bytes}, got {actual_bytes}"
            )
        actual_sha256 = (
            f"sha256:{hashlib.sha256(payload).hexdigest()}"
        )
        if actual_sha256 != trace_spec.sha256:
            raise PcGoldenValidationError(
                "trace artifact SHA-256 does not match canonical trace: "
                f"expected {trace_spec.sha256}, got {actual_sha256}"
            )

        last_frame_index: int | None = None
        last_frame_pts: int | None = None
        last_input_pts: int | None = None
        for record in self.records:
            for frame in record.frames:
                if not 0 <= frame.frame_index < video.frame_count:
                    raise PcGoldenValidationError(
                        f"frame index {frame.frame_index} is outside video"
                    )
                if not (
                    video.first_pts
                    <= frame.pts
                    <= video.last_pts
                ):
                    raise PcGoldenValidationError(
                        f"frame PTS {frame.pts} is outside video"
                    )
                if (
                    last_frame_index is not None
                    and frame.frame_index <= last_frame_index
                ):
                    raise PcGoldenValidationError(
                        "frame indices must increase across trace ticks"
                    )
                if last_frame_pts is not None and frame.pts < last_frame_pts:
                    raise PcGoldenValidationError(
                        "frame PTS must be non-decreasing across trace ticks"
                    )
                last_frame_index = frame.frame_index
                last_frame_pts = frame.pts
            for item in record.inputs:
                if not (
                    video.first_pts
                    <= item.pts
                    <= video.last_pts
                ):
                    raise PcGoldenValidationError(
                        f"input PTS {item.pts} is outside video time range"
                    )
                if last_input_pts is not None and item.pts < last_input_pts:
                    raise PcGoldenValidationError(
                        "input PTS must be non-decreasing across trace ticks"
                    )
                last_input_pts = item.pts


def pc_golden_native_source_fingerprint(
    manifest: PcGoldenManifest,
) -> str:
    """Identify immutable native capture inputs, not mutable packaging.

    Evidence-set fingerprints intentionally change when provenance or
    annotations are enriched.  Independence must instead bind the input DMO
    and every deterministic replay's raw video, DXGI metadata, and
    framework-update map.
    """

    artifacts = manifest.artifacts
    input_artifact = artifacts[manifest.input_timeline.artifact]
    exact_step_replay = getattr(manifest, "exact_step_replay", None)
    if exact_step_replay is not None:
        replay_sources = sorted(
            (
                artifacts[run.attempts_artifact].sha256,
                artifacts[run.memory_probe_artifact].sha256,
                artifacts[run.trajectory_index_artifact].sha256,
                run.process_id,
                run.process_creation_filetime_100ns,
            )
            for run in exact_step_replay.runs
        )
        return canonical_sha256(
            {
                "schema": "zuma-rl.pc-golden-native-source",
                "version": 3,
                "transport": "formal_exact_step_external_bgra",
                "input_dmo_sha256": input_artifact.sha256,
                "replay_sources": [
                    {
                        "attempts_sha256": attempts_sha256,
                        "memory_probe_sha256": probe_sha256,
                        "trajectory_index_sha256": index_sha256,
                        "process_id": process_id,
                        "process_creation_filetime_100ns": creation_time,
                    }
                    for (
                        attempts_sha256,
                        probe_sha256,
                        index_sha256,
                        process_id,
                        creation_time,
                    ) in replay_sources
                ],
            }
        )
    replay_runs = manifest.replay_determinism.runs
    if not any(
        getattr(run, "render_settled_evidence", None) is not None
        for run in replay_runs
    ):
        replay_sources = sorted(
            (
                artifacts[run.video.artifact].sha256,
                artifacts[run.capture_metadata_artifact].sha256,
                artifacts[run.framework_update_artifact].sha256,
            )
            for run in replay_runs
        )
        return canonical_sha256(
            {
                "schema": "zuma-rl.pc-golden-native-source",
                "version": 1,
                "input_dmo_sha256": input_artifact.sha256,
                "replay_sources": [
                    {
                        "video_sha256": video_sha256,
                        "capture_metadata_sha256": capture_metadata_sha256,
                        "framework_update_sha256": framework_update_sha256,
                    }
                    for (
                        video_sha256,
                        capture_metadata_sha256,
                        framework_update_sha256,
                    ) in replay_sources
                ],
            }
        )
    replay_sources_v2 = []
    for run in replay_runs:
        settled = getattr(run, "render_settled_evidence", None)
        assert settled is not None
        replay_sources_v2.append(
            {
                "video_sha256": artifacts[run.video.artifact].sha256,
                "capture_metadata_sha256": artifacts[
                    run.capture_metadata_artifact
                ].sha256,
                "framework_update_sha256": artifacts[
                    run.framework_update_artifact
                ].sha256,
                "render_settled_update_sha256": artifacts[
                    settled.update_map_artifact
                ].sha256,
                "frames_csv_sha256": artifacts[
                    settled.frames_csv_artifact
                ].sha256,
                "framework_poll_sha256": artifacts[
                    settled.framework_poll_artifact
                ].sha256,
                "framework_state_sha256": artifacts[
                    settled.framework_state_artifact
                ].sha256,
                "calibration_preregistration_sha256": artifacts[
                    settled.calibration_preregistration_artifact
                ].sha256,
                "calibration_execution_binding_sha256": artifacts[
                    settled.calibration_execution_binding_artifact
                ].sha256,
                "calibration_holdout_report_sha256": artifacts[
                    settled.calibration_holdout_report_artifact
                ].sha256,
            }
        )
    return canonical_sha256(
        {
            "schema": "zuma-rl.pc-golden-native-source",
            "version": 2,
            "input_dmo_sha256": input_artifact.sha256,
            "replay_sources": sorted(
                replay_sources_v2,
                key=lambda row: (
                    row["video_sha256"],
                    row["capture_metadata_sha256"],
                    row["framework_update_sha256"],
                ),
            ),
        }
    )


__all__ = [
    "CAPTURE_CONTRACT_SCHEMA",
    "CAPTURE_CONTRACT_VERSION",
    "DMO_FILE_ID",
    "DMO_FORMAT",
    "DMO_VERSION",
    "EVIDENCE_SET_SCHEMA",
    "EVIDENCE_SET_VERSION",
    "EXACT_STEP_CAPTURE_CONTRACT_SCHEMA",
    "EXACT_STEP_CAPTURE_CONTRACT_VERSION",
    "EXACT_STEP_MANIFEST_VERSION",
    "EVENT_KINDS",
    "LEGACY_MANIFEST_VERSION",
    "MANIFEST_SCHEMA",
    "MANIFEST_VERSION",
    "MEASUREMENT_CHANNELS",
    "STRUCTURAL_TRACE_CHANNELS",
    "SUPPORTED_COVERAGE_CHANNELS",
    "TRACE_SCHEMA",
    "TRACE_VERSION",
    "TRUSTED_PC_EVIDENCE_SOURCES",
    "ArtifactSpec",
    "ComparisonContract",
    "ComparisonResult",
    "ComparisonStatus",
    "CoordinateCalibration",
    "CoverageRange",
    "CoverageStatus",
    "ExactStepReplayContract",
    "ExactStepRunContract",
    "FrameRef",
    "GoldenEvent",
    "GoldenInput",
    "GoldenTick",
    "InputTimeline",
    "Measurement",
    "MeasurementStatus",
    "PcEnvironment",
    "PcGoldenArtifactError",
    "PcGoldenManifest",
    "PcGoldenTrace",
    "PcGoldenValidationError",
    "ReplayDeterminismContract",
    "ReplayPixelComparisonContract",
    "ReplayRunContract",
    "RenderSettledReplayEvidence",
    "SAVE_VOLATILE_REGISTRY_ROLES",
    "SaveRunContract",
    "SaveTransactionContract",
    "Scenario",
    "TickClock",
    "VideoMetadata",
    "canonical_sha256",
    "capture_contract_fingerprint",
    "evidence_set_fingerprint",
    "exact_step_capture_contract_fingerprint",
    "pc_golden_native_source_fingerprint",
    "validate_measurement_value",
]
