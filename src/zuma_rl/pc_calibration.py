"""Strict, read-only coordinate-calibration evidence for PC golden cases.

The v1 sidecar contains raw/video pixel control points, their corresponding
800x600 logical-canvas points, and the declared result of an axis-aligned
least-squares fit::

    logical_x = scale_x * raw_x + translate_x
    logical_y = scale_y * raw_y + translate_y

Parsing is deliberately canonical and closed-world.  The module never writes
files, and its public errors do not include the input path or control-point
values.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from zuma_rl.pc_golden import (
    PcGoldenManifest,
    PcGoldenValidationError,
)

CALIBRATION_SCHEMA = "zuma-rl.pc-coordinate-calibration"
CALIBRATION_VERSION = 1
CALIBRATION_NUMERIC_TOLERANCE = 1e-9

_MIN_CONTROL_POINTS = 9
_MIN_DISTINCT_PER_AXIS = 3
_MAX_CONTROL_POINTS = 4096
_MAX_SIDECAR_BYTES = 4 * 1024 * 1024
_MAX_DESIGN_CONDITION = 1e12
_MAX_CANVAS_DIMENSION = 65536
_TRANSFORM_KIND = "axis_aligned_affine"
_PIXEL_CENTER_CONVENTIONS = frozenset(
    {"center_at_integer", "center_at_half"}
)


def _strict_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PcGoldenValidationError(f"{name} must be a JSON object")
    if any(not isinstance(key, str) for key in value):
        raise PcGoldenValidationError(f"{name} keys must be strings")
    return value


def _strict_sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PcGoldenValidationError(f"{name} must be a JSON array")
    return value


def _closed_fields(
    value: Mapping[str, Any],
    *,
    required: frozenset[str],
    name: str,
) -> None:
    if required.difference(value):
        raise PcGoldenValidationError(f"{name} is missing required fields")
    if set(value).difference(required):
        raise PcGoldenValidationError(f"{name} has unknown fields")


def _strict_int(
    value: Any,
    name: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise PcGoldenValidationError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise PcGoldenValidationError(f"{name} is outside the allowed range")
    if maximum is not None and result > maximum:
        raise PcGoldenValidationError(f"{name} is outside the allowed range")
    return result


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise PcGoldenValidationError(f"{name} must be numeric")
    try:
        result = float(value)
    except (OverflowError, ValueError):
        raise PcGoldenValidationError(f"{name} must be finite") from None
    if not math.isfinite(result):
        raise PcGoldenValidationError(f"{name} must be finite")
    # Do not preserve negative zero in canonical evidence.
    return 0.0 if result == 0.0 else result


def _json_number(value: float) -> int | float:
    """Use one deterministic JSON spelling for integral finite floats."""

    if value == 0.0:
        return 0
    if value.is_integer() and abs(value) <= (1 << 53):
        return int(value)
    return value


def _reject_constant(_: str) -> None:
    raise ValueError("non-finite JSON number")


def _object_without_duplicates(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object field")
        result[key] = value
    return result


def _loads_json(text: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_object_without_duplicates,
        )
    except (json.JSONDecodeError, UnicodeError, ValueError, RecursionError):
        raise PcGoldenValidationError(
            "calibration sidecar is not valid strict JSON"
        ) from None
    return _strict_mapping(value, "calibration sidecar")


def _canonical_json(value: Mapping[str, Any]) -> str:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
    except (TypeError, ValueError, UnicodeError, RecursionError):
        raise PcGoldenValidationError(
            "calibration sidecar cannot be represented canonically"
        ) from None


def _point_pair(value: Any, name: str) -> tuple[float, float]:
    coordinates = _strict_sequence(value, name)
    if len(coordinates) != 2:
        raise PcGoldenValidationError(
            f"{name} must contain exactly two coordinates"
        )
    return (
        _finite_number(coordinates[0], f"{name}[0]"),
        _finite_number(coordinates[1], f"{name}[1]"),
    )


def _coordinate_bounds(
    *,
    width: int,
    height: int,
    convention: str,
) -> tuple[float, float, float, float]:
    if convention == "center_at_integer":
        return 0.0, float(width - 1), 0.0, float(height - 1)
    return 0.5, float(width) - 0.5, 0.5, float(height) - 0.5


def _inside_bounds(
    point: tuple[float, float],
    bounds: tuple[float, float, float, float],
) -> bool:
    x_min, x_max, y_min, y_max = bounds
    return x_min <= point[0] <= x_max and y_min <= point[1] <= y_max


@dataclass(frozen=True, slots=True)
class CalibrationControlPoint:
    """One raw-pixel/logical-canvas correspondence."""

    raw: tuple[float, float]
    logical: tuple[float, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw", _point_pair(self.raw, "point.raw"))
        object.__setattr__(
            self,
            "logical",
            _point_pair(self.logical, "point.logical"),
        )

    def to_dict(self) -> dict[str, list[int | float]]:
        return {
            "raw": [_json_number(item) for item in self.raw],
            "logical": [_json_number(item) for item in self.logical],
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> "CalibrationControlPoint":
        data = _strict_mapping(value, "control point")
        _closed_fields(
            data,
            required=frozenset({"raw", "logical"}),
            name="control point",
        )
        return cls(raw=data["raw"], logical=data["logical"])


@dataclass(frozen=True, slots=True)
class CalibrationFit:
    """Recomputed axis-aligned affine fit and point residuals."""

    logical_from_raw: tuple[tuple[float, float, float], ...]
    rms_error_px: float
    max_error_px: float

    @property
    def matrix(self) -> NDArray[np.float64]:
        return np.asarray(self.logical_from_raw, dtype=np.float64)


def _least_squares_fit(
    points: tuple[CalibrationControlPoint, ...],
) -> CalibrationFit:
    raw = np.asarray([point.raw for point in points], dtype=np.float64)
    logical = np.asarray(
        [point.logical for point in points],
        dtype=np.float64,
    )
    ones = np.ones(len(points), dtype=np.float64)
    x_design = np.column_stack((raw[:, 0], ones))
    y_design = np.column_stack((raw[:, 1], ones))
    try:
        x_coefficients, _, x_rank, x_singular = np.linalg.lstsq(
            x_design,
            logical[:, 0],
            rcond=None,
        )
        y_coefficients, _, y_rank, y_singular = np.linalg.lstsq(
            y_design,
            logical[:, 1],
            rcond=None,
        )
    except np.linalg.LinAlgError:
        raise PcGoldenValidationError(
            "calibration control-point geometry is degenerate"
        ) from None

    if x_rank != 2 or y_rank != 2:
        raise PcGoldenValidationError(
            "calibration control-point geometry is degenerate"
        )
    if (
        len(x_singular) != 2
        or len(y_singular) != 2
        or x_singular[-1] <= 0.0
        or y_singular[-1] <= 0.0
        or x_singular[0] / x_singular[-1] > _MAX_DESIGN_CONDITION
        or y_singular[0] / y_singular[-1] > _MAX_DESIGN_CONDITION
    ):
        raise PcGoldenValidationError(
            "calibration control-point geometry is ill-conditioned"
        )

    scale_x, translate_x = (float(item) for item in x_coefficients)
    scale_y, translate_y = (float(item) for item in y_coefficients)
    if (
        not all(
            math.isfinite(item)
            for item in (
                scale_x,
                translate_x,
                scale_y,
                translate_y,
            )
        )
        or scale_x <= 0.0
        or scale_y <= 0.0
    ):
        raise PcGoldenValidationError(
            "calibration fit must have finite positive axis scales"
        )

    matrix = np.asarray(
        (
            (scale_x, 0.0, translate_x),
            (0.0, scale_y, translate_y),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    predicted = np.column_stack(
        (
            scale_x * raw[:, 0] + translate_x,
            scale_y * raw[:, 1] + translate_y,
        )
    )
    errors = np.linalg.norm(predicted - logical, axis=1)
    rms = float(np.sqrt(np.mean(np.square(errors))))
    maximum = float(np.max(errors))
    if (
        np.any(~np.isfinite(matrix))
        or np.any(~np.isfinite(errors))
        or not math.isfinite(rms)
        or not math.isfinite(maximum)
    ):
        raise PcGoldenValidationError(
            "calibration fit produced non-finite results"
        )
    return CalibrationFit(
        logical_from_raw=tuple(
            tuple(float(item) for item in row) for row in matrix
        ),
        rms_error_px=rms,
        max_error_px=maximum,
    )


def _declared_matrix(value: Any) -> tuple[tuple[float, float, float], ...]:
    rows = _strict_sequence(value, "logical_from_raw")
    if len(rows) != 3:
        raise PcGoldenValidationError(
            "logical_from_raw must be a 3x3 matrix"
        )
    matrix_rows: list[tuple[float, float, float]] = []
    for row in rows:
        items = _strict_sequence(row, "logical_from_raw row")
        if len(items) != 3:
            raise PcGoldenValidationError(
                "logical_from_raw must be a 3x3 matrix"
            )
        matrix_rows.append(
            tuple(
                _finite_number(item, "logical_from_raw element")
                for item in items
            )
        )
    matrix = np.asarray(matrix_rows, dtype=np.float64)
    if (
        matrix[0, 1] != 0.0
        or matrix[1, 0] != 0.0
        or matrix[2, 0] != 0.0
        or matrix[2, 1] != 0.0
        or matrix[2, 2] != 1.0
        or matrix[0, 0] <= 0.0
        or matrix[1, 1] <= 0.0
    ):
        raise PcGoldenValidationError(
            "logical_from_raw is not an axis-aligned affine matrix"
        )
    return tuple(matrix_rows)


@dataclass(frozen=True, slots=True)
class PcCalibrationSidecar:
    """Canonical v1 calibration evidence and its declared fit."""

    raw_width: int
    raw_height: int
    logical_width: int
    logical_height: int
    transform_kind: str
    pixel_center_convention: str
    control_points: tuple[CalibrationControlPoint, ...]
    logical_from_raw: tuple[tuple[float, float, float], ...]
    rms_error_px: float
    max_error_px: float

    def __post_init__(self) -> None:
        raw_width = _strict_int(
            self.raw_width,
            "raw_width",
            minimum=1,
            maximum=_MAX_CANVAS_DIMENSION,
        )
        raw_height = _strict_int(
            self.raw_height,
            "raw_height",
            minimum=1,
            maximum=_MAX_CANVAS_DIMENSION,
        )
        logical_width = _strict_int(
            self.logical_width,
            "logical_width",
            minimum=1,
        )
        logical_height = _strict_int(
            self.logical_height,
            "logical_height",
            minimum=1,
        )
        if (logical_width, logical_height) != (800, 600):
            raise PcGoldenValidationError(
                "logical canvas must be the original 800x600 space"
            )
        object.__setattr__(self, "raw_width", raw_width)
        object.__setattr__(self, "raw_height", raw_height)
        object.__setattr__(self, "logical_width", logical_width)
        object.__setattr__(self, "logical_height", logical_height)

        if (
            not isinstance(self.transform_kind, str)
            or self.transform_kind != _TRANSFORM_KIND
        ):
            raise PcGoldenValidationError(
                "transform_kind must be axis_aligned_affine"
            )
        if (
            not isinstance(self.pixel_center_convention, str)
            or self.pixel_center_convention
            not in _PIXEL_CENTER_CONVENTIONS
        ):
            raise PcGoldenValidationError(
                "pixel_center_convention is unsupported"
            )

        points = tuple(
            _strict_sequence(self.control_points, "control_points")
        )
        if len(points) < _MIN_CONTROL_POINTS:
            raise PcGoldenValidationError(
                "calibration requires at least nine control points"
            )
        if len(points) > _MAX_CONTROL_POINTS:
            raise PcGoldenValidationError(
                "calibration has too many control points"
            )
        if any(not isinstance(point, CalibrationControlPoint) for point in points):
            raise PcGoldenValidationError(
                "control_points must contain calibration control points"
            )
        if tuple(sorted(points, key=lambda point: point.raw)) != points:
            raise PcGoldenValidationError(
                "control_points must be ordered canonically by raw coordinates"
            )
        raw_points = tuple(point.raw for point in points)
        logical_points = tuple(point.logical for point in points)
        if len(set(raw_points)) != len(raw_points):
            raise PcGoldenValidationError(
                "calibration contains duplicate raw control points"
            )
        if len(set(logical_points)) != len(logical_points):
            raise PcGoldenValidationError(
                "calibration contains duplicate logical control points"
            )
        if (
            len({point[0] for point in raw_points})
            < _MIN_DISTINCT_PER_AXIS
            or len({point[1] for point in raw_points})
            < _MIN_DISTINCT_PER_AXIS
            or len({point[0] for point in logical_points})
            < _MIN_DISTINCT_PER_AXIS
            or len({point[1] for point in logical_points})
            < _MIN_DISTINCT_PER_AXIS
        ):
            raise PcGoldenValidationError(
                "calibration control-point geometry is degenerate"
            )

        raw_bounds = _coordinate_bounds(
            width=raw_width,
            height=raw_height,
            convention=self.pixel_center_convention,
        )
        logical_bounds = _coordinate_bounds(
            width=logical_width,
            height=logical_height,
            convention=self.pixel_center_convention,
        )
        if any(
            not _inside_bounds(point.raw, raw_bounds)
            or not _inside_bounds(point.logical, logical_bounds)
            for point in points
        ):
            raise PcGoldenValidationError(
                "calibration control point lies outside its declared canvas"
            )
        object.__setattr__(self, "control_points", points)

        matrix = _declared_matrix(self.logical_from_raw)
        object.__setattr__(self, "logical_from_raw", matrix)
        rms = _finite_number(self.rms_error_px, "rms_error_px")
        maximum = _finite_number(self.max_error_px, "max_error_px")
        if rms < 0.0 or maximum < 0.0 or rms > maximum:
            raise PcGoldenValidationError(
                "declared calibration errors are inconsistent"
            )
        object.__setattr__(self, "rms_error_px", rms)
        object.__setattr__(self, "max_error_px", maximum)

        self._validate_declared_fit(self.recompute())

    @property
    def matrix(self) -> NDArray[np.float64]:
        return np.asarray(self.logical_from_raw, dtype=np.float64)

    def recompute(self) -> CalibrationFit:
        """Recompute the fit only from the immutable control points."""

        return _least_squares_fit(self.control_points)

    def _validate_declared_fit(self, fit: CalibrationFit) -> None:
        if not np.allclose(
            self.matrix,
            fit.matrix,
            rtol=0.0,
            atol=CALIBRATION_NUMERIC_TOLERANCE,
        ):
            raise PcGoldenValidationError(
                "declared calibration matrix does not match control points"
            )
        if not math.isclose(
            self.rms_error_px,
            fit.rms_error_px,
            rel_tol=0.0,
            abs_tol=CALIBRATION_NUMERIC_TOLERANCE,
        ):
            raise PcGoldenValidationError(
                "declared RMS error does not match control points"
            )
        if not math.isclose(
            self.max_error_px,
            fit.max_error_px,
            rel_tol=0.0,
            abs_tol=CALIBRATION_NUMERIC_TOLERANCE,
        ):
            raise PcGoldenValidationError(
                "declared maximum error does not match control points"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CALIBRATION_SCHEMA,
            "version": CALIBRATION_VERSION,
            "raw_width": self.raw_width,
            "raw_height": self.raw_height,
            "logical_width": self.logical_width,
            "logical_height": self.logical_height,
            "transform_kind": self.transform_kind,
            "pixel_center_convention": self.pixel_center_convention,
            "control_points": [
                point.to_dict() for point in self.control_points
            ],
            "logical_from_raw": [
                [_json_number(item) for item in row]
                for row in self.logical_from_raw
            ],
            "rms_error_px": _json_number(self.rms_error_px),
            "max_error_px": _json_number(self.max_error_px),
        }

    def to_json(self) -> str:
        """Return the one accepted UTF-8 JSON representation."""

        return _canonical_json(self.to_dict())

    @property
    def canonical_content_sha256(self) -> str:
        return (
            "sha256:"
            + hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PcCalibrationSidecar":
        data = _strict_mapping(value, "calibration sidecar")
        fields = frozenset(
            {
                "schema",
                "version",
                "raw_width",
                "raw_height",
                "logical_width",
                "logical_height",
                "transform_kind",
                "pixel_center_convention",
                "control_points",
                "logical_from_raw",
                "rms_error_px",
                "max_error_px",
            }
        )
        _closed_fields(data, required=fields, name="calibration sidecar")
        if data["schema"] != CALIBRATION_SCHEMA:
            raise PcGoldenValidationError(
                "unsupported calibration sidecar schema"
            )
        version = _strict_int(data["version"], "version", minimum=1)
        if version != CALIBRATION_VERSION:
            raise PcGoldenValidationError(
                "unsupported calibration sidecar version"
            )
        point_values = _strict_sequence(
            data["control_points"],
            "control_points",
        )
        points = tuple(
            CalibrationControlPoint.from_dict(
                _strict_mapping(item, "control point")
            )
            for item in point_values
        )
        return cls(
            raw_width=data["raw_width"],
            raw_height=data["raw_height"],
            logical_width=data["logical_width"],
            logical_height=data["logical_height"],
            transform_kind=data["transform_kind"],
            pixel_center_convention=data["pixel_center_convention"],
            control_points=points,
            logical_from_raw=data["logical_from_raw"],
            rms_error_px=data["rms_error_px"],
            max_error_px=data["max_error_px"],
        )

    @classmethod
    def from_json(cls, text: str) -> "PcCalibrationSidecar":
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        try:
            encoded_size = len(text.encode("utf-8", errors="strict"))
        except UnicodeError:
            raise PcGoldenValidationError(
                "calibration sidecar is not valid UTF-8"
            ) from None
        if encoded_size > _MAX_SIDECAR_BYTES:
            raise PcGoldenValidationError(
                "calibration sidecar exceeds the parser size limit"
            )
        sidecar = cls.from_dict(_loads_json(text))
        if text != sidecar.to_json():
            raise PcGoldenValidationError(
                "calibration sidecar is not in canonical v1 JSON form"
            )
        return sidecar

    @classmethod
    def read(cls, path: str | Path) -> "PcCalibrationSidecar":
        """Read a sidecar without modifying it or exposing its path on error."""

        try:
            payload = Path(path).read_bytes()
        except (OSError, TypeError, ValueError):
            raise PcGoldenValidationError(
                "calibration sidecar could not be read"
            ) from None
        if len(payload) > _MAX_SIDECAR_BYTES:
            raise PcGoldenValidationError(
                "calibration sidecar exceeds the parser size limit"
            )
        try:
            text = payload.decode("utf-8", errors="strict")
        except UnicodeError:
            raise PcGoldenValidationError(
                "calibration sidecar is not valid UTF-8"
            ) from None
        return cls.from_json(text)

    def validate_against(
        self,
        manifest: PcGoldenManifest,
    ) -> CalibrationFit:
        """Recompute and validate every declaration against a v3 manifest."""

        if not isinstance(manifest, PcGoldenManifest):
            raise TypeError("manifest must be a PcGoldenManifest")
        fit = self.recompute()
        self._validate_declared_fit(fit)
        coordinates = manifest.coordinates
        if (
            self.raw_width != coordinates.raw_width
            or self.raw_height != coordinates.raw_height
            or self.logical_width != coordinates.logical_width
            or self.logical_height != coordinates.logical_height
        ):
            raise PcGoldenValidationError(
                "calibration dimensions do not match manifest coordinates"
            )
        if (
            self.transform_kind != coordinates.transform_kind
            or self.pixel_center_convention
            != coordinates.pixel_center_convention
        ):
            raise PcGoldenValidationError(
                "calibration conventions do not match manifest coordinates"
            )
        if not np.allclose(
            fit.matrix,
            coordinates.matrix,
            rtol=0.0,
            atol=CALIBRATION_NUMERIC_TOLERANCE,
        ):
            raise PcGoldenValidationError(
                "recomputed calibration matrix does not match manifest"
            )
        if (
            not math.isclose(
                fit.rms_error_px,
                coordinates.rms_error_px,
                rel_tol=0.0,
                abs_tol=CALIBRATION_NUMERIC_TOLERANCE,
            )
            or not math.isclose(
                fit.max_error_px,
                coordinates.max_error_px,
                rel_tol=0.0,
                abs_tol=CALIBRATION_NUMERIC_TOLERANCE,
            )
        ):
            raise PcGoldenValidationError(
                "recomputed calibration errors do not match manifest"
            )

        artifact = manifest.artifacts.get(
            coordinates.calibration_artifact
        )
        if artifact is None:
            raise PcGoldenValidationError(
                "manifest does not reference a calibration artifact"
            )
        payload = self.to_json().encode("utf-8")
        if (
            artifact.bytes != len(payload)
            or artifact.sha256 != self.canonical_content_sha256
        ):
            raise PcGoldenValidationError(
                "calibration artifact identity does not match manifest"
            )
        return fit


__all__ = [
    "CALIBRATION_NUMERIC_TOLERANCE",
    "CALIBRATION_SCHEMA",
    "CALIBRATION_VERSION",
    "CalibrationControlPoint",
    "CalibrationFit",
    "PcCalibrationSidecar",
]
