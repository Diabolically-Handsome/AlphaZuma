"""Independent validation of retail Zuma memory-probe artifacts.

The diagnostic collector stores both semantic JSON and fixed-size byte dumps
from the original 32-bit retail process.  This module treats the byte dumps as
the source of truth and recomputes every gameplay field used by the shot
transition and pixel-provenance checks.

The supported layout is deliberately narrow: the frozen retail executable
identified by the surrounding PC-golden case, one active Board, its shooter,
its curve manager, retail Ball objects, and retail Bullet objects.  Any layout
or artifact mismatch fails closed.
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import struct
from typing import Any, Mapping, Sequence


MEMORY_PROBE_SCHEMA = "zuma-rl.pc-memory-int32-probe"
MEMORY_PROBE_VERSION = 1
SCORE_BINDING_SCHEMA = "zuma-rl.pc-memory-score-binding"
SCORE_BINDING_LEGACY_VERSION = 1
SCORE_BINDING_INDEPENDENT_VERSION = 2
INDEPENDENT_SCORE_BINDING_MODE = (
    "independent_active_board_fields_read_only"
)
INDEPENDENT_SCORE_BINDING_ACCEPTANCE_RULE = (
    "all_int32_pairs_without_outcome_filtering"
)
FORMAL_FULL_STATE_CLASSIFICATION = (
    "formal_pc_full_state_exact_step_source"
)
ACTIVE_BOARD_SCHEMA = "zuma-rl.pc-active-board-probe"
ACTIVE_BOARD_LEGACY_VERSION = 1
ACTIVE_BOARD_VERSION = 2
ACTIVE_BOARD_SUPPORTED_VERSIONS = frozenset(
    {ACTIVE_BOARD_LEGACY_VERSION, ACTIVE_BOARD_VERSION}
)

G_CURVE_PLAN_EXHAUSTED_ADDRESS = 0x009E8252
BOARD_VTABLE = 0x0096356C
BOARD_EMBEDDED_VTABLE_OFFSET = 0x88
BOARD_EMBEDDED_VTABLE = 0x0096368C
BOARD_CURVE_MANAGER_OFFSET = 0x9C
BOARD_SCORE_OFFSET = 0x104
BOARD_SCORE_TARGET_OFFSET = 0x108
BOARD_PRIMARY_CHILD_OFFSET = 0x68C
BOARD_FIRED_BULLET_LIST_OFFSET = 0x698
BOARD_DISPLAYED_SCORE_OFFSET = 0xEFC
BOARD_MODE_FLAG_1064_OFFSET = 0x1064
BOARD_DUMP_SIZE = 0x1100

SHOOTER_VTABLE = 0x009670E0
SHOOTER_OBJECT_SIZE = 0x32C
SHOOTER_BULLET_POINTER_OFFSETS = (0x130, 0x134)

BULLET_VTABLE = 0x009636E4
BULLET_OBJECT_SIZE = 0x18C
BULLET_GAP_SENTINEL_POINTER_OFFSET = 0x174
BULLET_GAP_COUNT_OFFSET = 0x178
BULLET_CURVE_POINTS_OFFSET = 0x17C
BULLET_CURVE_POINT_COUNT = 4
RETAIL_DEBUG_FILL_I32 = -1163005939  # Signed 0xBAADF00D.
BULLET_GAP_NODE_SIZE = 0x14
MAX_BULLET_GAP_ENTRIES = 64

CURVE_MANAGER_VTABLE = 0x0096A204
CURVE_MANAGER_LEGACY_DUMP_SIZE = 0x400
CURVE_MANAGER_DUMP_SIZE = 0x40C
CURVE_MANAGER_CURVE_ARRAY_OFFSET = 0x16C
CURVE_MANAGER_CURVE_COUNT_OFFSET = 0x360
CURVE_MANAGER_POST_ZUMA_TIMER_OFFSET = 0x400
CURVE_MANAGER_POST_ZUMA_RAMP_404_OFFSET = 0x404
CURVE_MANAGER_POST_ZUMA_RAMP_408_OFFSET = 0x408
CURVE_VTABLE = 0x009641F0
CURVE_DUMP_SIZE = 0x200
CURVE_PLANNED_VECTOR_BEGIN_OFFSET = 0x34
CURVE_PLANNED_VECTOR_END_OFFSET = 0x38
CURVE_PLANNED_VECTOR_CAPACITY_OFFSET = 0x3C
CURVE_PLANNED_ITEM_SIZE = 0x14
CURVE_ADD_PLAN_ENABLED_OFFSET = 0x1A3
CURVE_INTRUSIVE_LIST_OFFSETS = (0x50, 0x5C, 0x68)
INSERTION_STAGING_LIST_OFFSET = 0x50
ACTIVE_CHAIN_LIST_OFFSET = 0x5C

BALL_VTABLE = 0x00960160
BALL_OBJECT_SIZE = 0x134
MAX_CURVES = 16
MAX_CURVE_PLANNED_ITEMS = 4096
MAX_CURVE_LIST_ITEMS = 2048
MAX_FIRED_BULLETS = 64
FREEZE_STATE_BYTES = 168
FREEZE_STATE_MULTIPLIER_OFFSET = 60
FROZEN_UPDATE_MULTIPLIER = 0.0877914951989026
FORMAL_FULL_STATE_UPDATE_MULTIPLIER = 0.00770734662925894
SAMPLE_RADIUS = 14.5
MINIMUM_SATURATION = 0.35
MINIMUM_VALUE = 0.20
HUE_KERNEL_WIDTH = 0.09
ROLLOUT_FOREGROUND_OCCLUSION_X = 80.0
ROLLOUT_FOREGROUND_OCCLUSION_Y = 100.0
PROTOTYPE_HUES = {
    0: 0.610,
    1: 0.145,
    2: 0.000,
    3: 0.330,
    4: 0.800,
}

_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
MEMORY_TRANSITION_CONTRACT_SCHEMA = (
    "zuma-rl.pc-memory-transition-contract"
)
MEMORY_TRANSITION_CONTRACT_VERSION = 1
MEMORY_TRANSITION_CONTRACT_ARTIFACT = "memory_transition_contract"
MEMORY_CURVE_GEOMETRY_SCHEMA = "zuma-rl.pc-memory-curve-geometry-binding"
MEMORY_CURVE_GEOMETRY_VERSION = 1
MEMORY_CURVE_GEOMETRY_ARTIFACT = "memory_curve_geometry_binding"
MEMORY_CURVE_GEOMETRY_TOLERANCE_PX = 1e-3


class PcMemoryEvidenceError(ValueError):
    """A probe or one of its raw artifacts is inconsistent."""


class PcMemoryImageDecoderUnavailable(RuntimeError):
    """Pillow is unavailable for the lossless BMP pixel cross-check."""


@dataclass(frozen=True, slots=True)
class PcMemoryTransitionContract:
    """Manifest-key references and frozen tolerances for one shot proof."""

    case_id: str
    before_probe_artifact: str
    after_probe_artifact: str
    before_pixel_validation_artifact: str
    after_pixel_validation_artifact: str
    transition_validation_artifact: str
    video_binding_validation_artifact: str
    expected_before_update: int
    expected_after_update: int
    expected_score: int
    expected_chain_distance_delta: float
    distance_tolerance: float
    minimum_chain_count: int
    minimum_visible: int
    maximum_mismatches: int
    minimum_visible_after_fired_bullets: int
    maximum_fired_bullet_mismatches: int
    video_binding_excluded_bottom_rows: int
    maximum_excluded_edge_mismatches: int

    def __post_init__(self) -> None:
        for name in (
            "case_id",
            "before_probe_artifact",
            "after_probe_artifact",
            "before_pixel_validation_artifact",
            "after_pixel_validation_artifact",
            "transition_validation_artifact",
            "video_binding_validation_artifact",
        ):
            value = getattr(self, name)
            _require(
                isinstance(value, str) and bool(value),
                f"memory_transition_contract_{name}_invalid",
            )
        references = tuple(self.referenced_artifacts)
        _require(
            len(references) == 6,
            "memory_transition_contract_artifacts_not_distinct",
        )
        for name in (
            "expected_before_update",
            "expected_after_update",
            "expected_score",
            "minimum_chain_count",
            "minimum_visible",
            "maximum_mismatches",
            "minimum_visible_after_fired_bullets",
            "maximum_fired_bullet_mismatches",
            "video_binding_excluded_bottom_rows",
            "maximum_excluded_edge_mismatches",
        ):
            value = getattr(self, name)
            _require(
                isinstance(value, int)
                and not isinstance(value, bool)
                and value >= 0,
                f"memory_transition_contract_{name}_invalid",
            )
        _require(
            self.expected_after_update > self.expected_before_update,
            "memory_transition_contract_update_order_invalid",
        )
        _require(
            self.minimum_chain_count >= 1 and self.minimum_visible >= 1,
            "memory_transition_contract_minimum_invalid",
        )
        _require(
            self.video_binding_excluded_bottom_rows <= 4
            and self.maximum_excluded_edge_mismatches
            <= 800 * self.video_binding_excluded_bottom_rows,
            "memory_transition_contract_video_edge_budget_invalid",
        )
        for name in (
            "expected_chain_distance_delta",
            "distance_tolerance",
        ):
            value = getattr(self, name)
            _require(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value)),
                f"memory_transition_contract_{name}_invalid",
            )
            object.__setattr__(self, name, float(value))
        _require(
            self.distance_tolerance >= 0.0,
            "memory_transition_contract_tolerance_invalid",
        )

    @property
    def referenced_artifacts(self) -> frozenset[str]:
        return frozenset(
            {
                self.before_probe_artifact,
                self.after_probe_artifact,
                self.before_pixel_validation_artifact,
                self.after_pixel_validation_artifact,
                self.transition_validation_artifact,
                self.video_binding_validation_artifact,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": MEMORY_TRANSITION_CONTRACT_SCHEMA,
            "version": MEMORY_TRANSITION_CONTRACT_VERSION,
            "case_id": self.case_id,
            "before_probe_artifact": self.before_probe_artifact,
            "after_probe_artifact": self.after_probe_artifact,
            "before_pixel_validation_artifact": (
                self.before_pixel_validation_artifact
            ),
            "after_pixel_validation_artifact": (
                self.after_pixel_validation_artifact
            ),
            "transition_validation_artifact": (
                self.transition_validation_artifact
            ),
            "video_binding_validation_artifact": (
                self.video_binding_validation_artifact
            ),
            "expected_before_update": self.expected_before_update,
            "expected_after_update": self.expected_after_update,
            "expected_score": self.expected_score,
            "expected_chain_distance_delta": (
                self.expected_chain_distance_delta
            ),
            "distance_tolerance": self.distance_tolerance,
            "minimum_chain_count": self.minimum_chain_count,
            "minimum_visible": self.minimum_visible,
            "maximum_mismatches": self.maximum_mismatches,
            "minimum_visible_after_fired_bullets": (
                self.minimum_visible_after_fired_bullets
            ),
            "maximum_fired_bullet_mismatches": (
                self.maximum_fired_bullet_mismatches
            ),
            "video_binding_excluded_bottom_rows": (
                self.video_binding_excluded_bottom_rows
            ),
            "maximum_excluded_edge_mismatches": (
                self.maximum_excluded_edge_mismatches
            ),
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> "PcMemoryTransitionContract":
        _require(isinstance(value, Mapping), "memory_transition_contract_invalid")
        fields = {
            "case_id",
            "before_probe_artifact",
            "after_probe_artifact",
            "before_pixel_validation_artifact",
            "after_pixel_validation_artifact",
            "transition_validation_artifact",
            "video_binding_validation_artifact",
            "expected_before_update",
            "expected_after_update",
            "expected_score",
            "expected_chain_distance_delta",
            "distance_tolerance",
            "minimum_chain_count",
            "minimum_visible",
            "maximum_mismatches",
            "minimum_visible_after_fired_bullets",
            "maximum_fired_bullet_mismatches",
            "video_binding_excluded_bottom_rows",
            "maximum_excluded_edge_mismatches",
        }
        _require(
            value.get("schema") == MEMORY_TRANSITION_CONTRACT_SCHEMA
            and value.get("version")
            == MEMORY_TRANSITION_CONTRACT_VERSION,
            "memory_transition_contract_schema_invalid",
        )
        _require(
            set(value) == fields | {"schema", "version"},
            "memory_transition_contract_fields_invalid",
        )
        return cls(**{name: value[name] for name in fields})

    @classmethod
    def read(cls, path: Path) -> "PcMemoryTransitionContract":
        return cls.from_dict(_load_probe(path))


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise PcMemoryEvidenceError(reason)


def build_memory_curve_geometry_binding(
    *,
    case_id: str,
    level_id: str,
    hard: bool,
    curve_index: int,
    curve: Any,
    phases: Sequence[
        tuple[str, int, Sequence[Mapping[str, Any]]]
    ],
    position_tolerance_px: float = MEMORY_CURVE_GEOMETRY_TOLERANCE_PX,
) -> Mapping[str, Any]:
    """Bind decoded retail chain centers to the claimed installed curve."""

    _require(
        isinstance(case_id, str) and bool(case_id),
        "memory_curve_binding_case_id_invalid",
    )
    _require(
        isinstance(level_id, str) and bool(level_id),
        "memory_curve_binding_level_id_invalid",
    )
    _require(
        isinstance(hard, bool),
        "memory_curve_binding_hard_invalid",
    )
    _require(
        isinstance(curve_index, int)
        and not isinstance(curve_index, bool)
        and curve_index >= 0,
        "memory_curve_binding_curve_index_invalid",
    )
    _require(
        isinstance(position_tolerance_px, (int, float))
        and not isinstance(position_tolerance_px, bool)
        and math.isfinite(float(position_tolerance_px))
        and position_tolerance_px > 0,
        "memory_curve_binding_tolerance_invalid",
    )
    _require(bool(phases), "memory_curve_binding_phases_empty")

    phase_rows: list[dict[str, Any]] = []
    seen_phases: set[str] = set()
    for phase, update, records in phases:
        _require(
            isinstance(phase, str)
            and phase in {"before", "after"}
            and phase not in seen_phases,
            "memory_curve_binding_phase_invalid",
        )
        seen_phases.add(phase)
        _require(
            isinstance(update, int)
            and not isinstance(update, bool)
            and update >= 0,
            "memory_curve_binding_update_invalid",
        )
        _require(
            isinstance(records, Sequence) and bool(records),
            "memory_curve_binding_chain_empty",
        )
        squared_errors: list[float] = []
        maximum_error = 0.0
        for record in records:
            _require(
                isinstance(record, Mapping)
                and record.get("object_kind") == "ball",
                "memory_curve_binding_record_invalid",
            )
            ball = record.get("ball")
            _require(
                isinstance(ball, Mapping),
                "memory_curve_binding_ball_invalid",
            )
            values = (
                ball.get("curve_distance"),
                ball.get("position_x"),
                ball.get("position_y"),
            )
            _require(
                all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    for value in values
                ),
                "memory_curve_binding_ball_invalid",
            )
            point = curve.point_at_waypoint(float(values[0]))
            error = math.hypot(
                float(point[0]) - float(values[1]),
                float(point[1]) - float(values[2]),
            )
            squared_errors.append(error * error)
            maximum_error = max(maximum_error, error)
        rms_error = math.sqrt(
            sum(squared_errors) / len(squared_errors)
        )
        _require(
            maximum_error <= float(position_tolerance_px),
            "memory_curve_geometry_mismatch:"
            f"{phase}:{maximum_error:.9f}",
        )
        phase_rows.append(
            {
                "phase": phase,
                "framework_update": update,
                "sample_count": len(records),
                "maximum_position_error_px": maximum_error,
                "rms_position_error_px": rms_error,
            }
        )
    return {
        "schema": MEMORY_CURVE_GEOMETRY_SCHEMA,
        "version": MEMORY_CURVE_GEOMETRY_VERSION,
        "status": "PASS",
        "case_id": case_id,
        "scenario": {
            "level_id": level_id,
            "hard": hard,
            "curve_index": curve_index,
        },
        "position_tolerance_px": float(position_tolerance_px),
        "phases": phase_rows,
    }


def sha256_bytes(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _u32(payload: bytes, offset: int) -> int:
    return struct.unpack_from("<I", payload, offset)[0]


def _i32(payload: bytes, offset: int) -> int:
    return struct.unpack_from("<i", payload, offset)[0]


def decode_ball(
    payload: bytes,
    *,
    expected_vtable: int = BALL_VTABLE,
) -> dict[str, Any]:
    _require(len(payload) >= BALL_OBJECT_SIZE, "ball_payload_too_short")
    _require(_u32(payload, 0) == expected_vtable, "ball_vtable_mismatch")
    color_id = _i32(payload, 0x14)
    _require(0 <= color_id <= 5, "ball_color_invalid")
    curve_distance = struct.unpack_from("<f", payload, 0x1C)[0]
    orientation = struct.unpack_from("<f", payload, 0x20)[0]
    previous_orientation = struct.unpack_from("<f", payload, 0x24)[0]
    angular_step = struct.unpack_from("<f", payload, 0x28)[0]
    position_x, position_y = struct.unpack_from("<ff", payload, 0x2C)
    scale = struct.unpack_from("<f", payload, 0x34)[0]
    radius = struct.unpack_from("<f", payload, 0x38)[0]
    _require(
        all(
            math.isfinite(value)
            for value in (
                curve_distance,
                orientation,
                previous_orientation,
                angular_step,
                position_x,
                position_y,
                scale,
                radius,
            )
        ),
        "ball_float_nonfinite",
    )
    return {
        "ball_id": _u32(payload, 0x10),
        "color_id": color_id,
        "curve_distance": curve_distance,
        "orientation_radians": orientation,
        "previous_orientation_radians": previous_orientation,
        "angular_step_radians": angular_step,
        "position_x": position_x,
        "position_y": position_y,
        "scale": scale,
        "radius": radius,
        "powerup_previous_type": _i32(payload, 0xC8),
        "powerup_primary_type": _i32(payload, 0x11C),
        "powerup_secondary_type": _i32(payload, 0x120),
        "flags_b4_c2_hex": payload[0xB4:0xC3].hex(),
    }


def decode_bullet_subclass(payload: bytes) -> dict[str, Any]:
    _require(
        len(payload) == BULLET_OBJECT_SIZE,
        "bullet_payload_size_invalid",
    )
    _require(_u32(payload, 0) == BULLET_VTABLE, "bullet_vtable_mismatch")
    floats = {
        "velocity_x": struct.unpack_from("<f", payload, 0x138)[0],
        "velocity_y": struct.unpack_from("<f", payload, 0x13C)[0],
        "field_148_float": struct.unpack_from("<f", payload, 0x148)[0],
        "field_14c_float": struct.unpack_from("<f", payload, 0x14C)[0],
        "field_158_float": struct.unpack_from("<f", payload, 0x158)[0],
        "field_15c_float": struct.unpack_from("<f", payload, 0x15C)[0],
        "heading_radians": struct.unpack_from("<f", payload, 0x160)[0],
    }
    _require(
        all(math.isfinite(value) for value in floats.values()),
        "bullet_float_nonfinite",
    )
    owner = _u32(payload, 0x130)
    return {
        "owner_shooter_address": owner,
        "owner_shooter_address_hex": f"0x{owner:08x}",
        "field_134_pointer": _u32(payload, 0x134),
        **floats,
        "field_164_i32": _i32(payload, 0x164),
        "fired": bool(payload[0x16A]),
        "flags_16a_16c_hex": payload[0x16A:0x16D].hex(),
        "field_174_pointer": _u32(payload, 0x174),
        "field_178_i32": _i32(payload, 0x178),
    }


def decode_freeze_state(
    payload: bytes,
    *,
    multiplier_address: int,
) -> dict[str, Any]:
    _require(
        len(payload) == FREEZE_STATE_BYTES,
        "freeze_state_payload_size_invalid",
    )
    offset = FREEZE_STATE_MULTIPLIER_OFFSET

    def i32(relative: int) -> int:
        return struct.unpack_from("<i", payload, offset + relative)[0]

    multiplier = struct.unpack_from("<d", payload, offset)[0]
    _require(math.isfinite(multiplier), "freeze_multiplier_nonfinite")
    return {
        "multiplier_address": multiplier_address,
        "non_draw_count": i32(-60),
        "frame_time_ms": i32(-56),
        "sleep_count": i32(-20),
        "draw_count": i32(-16),
        "update_count": i32(-12),
        "update_app_state": i32(-8),
        "update_app_depth": i32(-4),
        "update_multiplier": multiplier,
        "paused": bool(payload[offset + 8]),
        "fast_forward_target": i32(12),
        "loading_thread_started": bool(payload[offset + 105]),
        "loading_thread_completed": bool(payload[offset + 106]),
        "loaded": bool(payload[offset + 107]),
    }


def _load_probe(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="ascii"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                PcMemoryEvidenceError(f"nonfinite_json_number:{value}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PcMemoryEvidenceError("memory_probe_json_invalid") from error
    _require(isinstance(value, dict), "memory_probe_root_invalid")
    return value


def _sha256(value: Any, reason: str) -> str:
    _require(
        isinstance(value, str) and _SHA256_PATTERN.fullmatch(value) is not None,
        reason,
    )
    return value


def _artifact_path(
    probe_path: Path,
    value: Any,
    *,
    allowed_paths: frozenset[Path] | None,
) -> Path:
    _require(isinstance(value, str) and bool(value), "artifact_name_invalid")
    pure = PurePosixPath(value)
    _require(
        not pure.is_absolute()
        and ".." not in pure.parts
        and "\\" not in value
        and str(pure) == value,
        "artifact_name_noncanonical",
    )
    base = probe_path.parent.resolve()
    target = (base / value).resolve()
    _require(target.is_relative_to(base), "artifact_escapes_probe_root")
    if allowed_paths is not None:
        _require(target in allowed_paths, "artifact_not_declared_by_manifest")
    return target


def _read_artifact(
    probe_path: Path,
    row: Mapping[str, Any],
    *,
    expected_size: int | None,
    maximum_size: int,
    allowed_paths: frozenset[Path] | None,
) -> tuple[Path, bytes]:
    path = _artifact_path(
        probe_path,
        row["artifact"],
        allowed_paths=allowed_paths,
    )
    try:
        size = path.stat().st_size
    except OSError as error:
        raise PcMemoryEvidenceError("probe_artifact_missing") from error
    _require(size <= maximum_size, "probe_artifact_too_large")
    if expected_size is not None:
        _require(size == expected_size, "probe_artifact_size_mismatch")
    if "artifact_bytes" in row:
        _require(row["artifact_bytes"] == size, "declared_artifact_size_mismatch")
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise PcMemoryEvidenceError("probe_artifact_unreadable") from error
    digest = sha256_bytes(payload)
    if "artifact_sha256" in row:
        _require(
            _sha256(row["artifact_sha256"], "artifact_sha256_invalid")
            == digest,
            "probe_artifact_sha256_mismatch",
        )
    return path, payload


def _require_semantics(
    declared: Any,
    decoded: Mapping[str, Any],
    *,
    reason: str,
) -> None:
    _require(isinstance(declared, Mapping), reason)
    for key, value in decoded.items():
        _require(key in declared and declared[key] == value, f"{reason}:{key}")


def _validate_payload_records(
    payload: bytes,
    row: Mapping[str, Any],
    *,
    object_size: int,
    expected_vtable: int,
    bullet: bool,
    maximum_count: int,
) -> list[dict[str, Any]]:
    records = row["records"]
    _require(isinstance(records, list), "payload_records_invalid")
    _require(len(records) <= maximum_count, "payload_record_count_too_large")
    if "payload_count" in row:
        _require(row["payload_count"] == len(records), "payload_count_mismatch")
    if "declared_count" in row:
        _require(row["declared_count"] == len(records), "declared_count_mismatch")
    if "traversed_count" in row:
        _require(row["traversed_count"] == len(records), "traversed_count_mismatch")
    _require(
        len(payload) == len(records) * object_size,
        "payload_blob_size_mismatch",
    )
    decoded_rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        _require(isinstance(record, Mapping), "payload_record_invalid")
        offset = index * object_size
        _require(record.get("index", index) == index, "record_index_mismatch")
        _require(
            record["artifact_offset"] == offset,
            "record_artifact_offset_mismatch",
        )
        _require(
            record["artifact_bytes"] == object_size,
            "record_artifact_size_mismatch",
        )
        raw = payload[offset : offset + object_size]
        _require(
            _u32(raw, 0) == expected_vtable,
            "record_vtable_mismatch",
        )
        if "payload_sha256" in record:
            _require(
                _sha256(
                    record["payload_sha256"],
                    "record_payload_sha256_invalid",
                )
                == sha256_bytes(raw),
                "record_payload_sha256_mismatch",
            )
        ball = decode_ball(raw, expected_vtable=expected_vtable)
        _require_semantics(
            record["ball"],
            ball,
            reason="record_ball_semantics_mismatch",
        )
        decoded: dict[str, Any] = {"ball": ball}
        if bullet:
            subclass = decode_bullet_subclass(raw)
            _require_semantics(
                record["subclass_fields"],
                subclass,
                reason="record_bullet_semantics_mismatch",
            )
            decoded["subclass_fields"] = subclass
        decoded_rows.append(decoded)
    return decoded_rows


def _validate_curve_payload_records(
    payload: bytes,
    row: Mapping[str, Any],
    *,
    allow_bullets: bool,
    maximum_count: int,
) -> list[dict[str, Any]]:
    """Validate a variable-width curve-list blob without trusting its JSON."""

    records = row["records"]
    _require(isinstance(records, list), "payload_records_invalid")
    _require(len(records) <= maximum_count, "payload_record_count_too_large")
    if "payload_count" in row:
        _require(
            row["payload_count"] == len(records),
            "payload_count_mismatch",
        )
    if "declared_count" in row:
        _require(
            row["declared_count"] == len(records),
            "declared_count_mismatch",
        )
    if "traversed_count" in row:
        _require(
            row["traversed_count"] == len(records),
            "traversed_count_mismatch",
        )

    decoded_rows: list[dict[str, Any]] = []
    frequencies: dict[str, int] = {}
    offset = 0
    for index, record in enumerate(records):
        _require(isinstance(record, Mapping), "payload_record_invalid")
        _require(
            record.get("index", index) == index,
            "record_index_mismatch",
        )
        _require(offset + 4 <= len(payload), "payload_blob_size_mismatch")
        vtable = _u32(payload, offset)
        if vtable == BALL_VTABLE:
            object_size = BALL_OBJECT_SIZE
            object_kind = "ball"
        elif allow_bullets and vtable == BULLET_VTABLE:
            object_size = BULLET_OBJECT_SIZE
            object_kind = "bullet"
        else:
            _require(False, "record_vtable_mismatch")
        _require(
            offset + object_size <= len(payload),
            "payload_blob_size_mismatch",
        )
        _require(
            record["artifact_offset"] == offset,
            "record_artifact_offset_mismatch",
        )
        _require(
            record["artifact_bytes"] == object_size,
            "record_artifact_size_mismatch",
        )
        if "payload_kind" in record:
            _require(
                record["payload_kind"] == object_kind,
                "record_payload_kind_mismatch",
            )
        raw = payload[offset : offset + object_size]
        if "payload_sha256" in record:
            _require(
                _sha256(
                    record["payload_sha256"],
                    "record_payload_sha256_invalid",
                )
                == sha256_bytes(raw),
                "record_payload_sha256_mismatch",
            )
        ball = decode_ball(raw, expected_vtable=vtable)
        _require_semantics(
            record["ball"],
            ball,
            reason="record_ball_semantics_mismatch",
        )
        decoded: dict[str, Any] = {
            "object_kind": object_kind,
            "ball": ball,
        }
        if object_kind == "bullet":
            subclass = decode_bullet_subclass(raw)
            _require_semantics(
                record["subclass_fields"],
                subclass,
                reason="record_bullet_semantics_mismatch",
            )
            decoded["subclass_fields"] = subclass
        decoded_rows.append(decoded)
        key = f"0x{vtable:08x}"
        frequencies[key] = frequencies.get(key, 0) + 1
        offset += object_size
    _require(offset == len(payload), "payload_blob_size_mismatch")
    if "payload_vtable_frequencies" in row:
        _require(
            row["payload_vtable_frequencies"] == frequencies,
            "payload_vtable_frequencies_mismatch",
        )
    return decoded_rows


def _validate_fired_gap_records(
    gap_payload: bytes,
    fired_payload: bytes,
    fired_row: Mapping[str, Any],
    decoded_fired: Sequence[dict[str, Any]],
) -> None:
    records = fired_row["records"]
    _require(
        isinstance(records, list) and len(records) == len(decoded_fired),
        "fired_gap_records_invalid",
    )
    cursor = 0
    for index, (record, decoded) in enumerate(
        zip(records, decoded_fired, strict=True)
    ):
        _require(isinstance(record, Mapping), "fired_gap_record_invalid")
        bullet_offset = index * BULLET_OBJECT_SIZE
        bullet = fired_payload[
            bullet_offset : bullet_offset + BULLET_OBJECT_SIZE
        ]
        sentinel = _u32(bullet, BULLET_GAP_SENTINEL_POINTER_OFFSET)
        count = _u32(bullet, BULLET_GAP_COUNT_OFFSET)
        curve_points = list(
            struct.unpack_from(
                f"<{BULLET_CURVE_POINT_COUNT}i",
                bullet,
                BULLET_CURVE_POINTS_OFFSET,
            )
        )
        _require(
            sentinel != 0
            and count <= MAX_BULLET_GAP_ENTRIES
            and all(
                value >= 0 or value == RETAIL_DEBUG_FILL_I32
                for value in curve_points
            ),
            "fired_gap_raw_state_invalid",
        )
        subclass = record.get("subclass_fields")
        _require(isinstance(subclass, Mapping), "fired_gap_subclass_invalid")
        _require(
            subclass.get("gap_list_sentinel_address") == sentinel
            and subclass.get("gap_entry_count") == count
            and subclass.get("curve_points") == curve_points,
            "fired_gap_declared_state_mismatch",
        )
        entries = subclass.get("gap_entries")
        _require(
            isinstance(entries, list) and len(entries) == count,
            "fired_gap_entry_count_mismatch",
        )
        gap_offset = record.get("gap_artifact_offset")
        gap_bytes = record.get("gap_artifact_bytes")
        _require(
            isinstance(gap_offset, int)
            and not isinstance(gap_offset, bool)
            and gap_offset == cursor
            and isinstance(gap_bytes, int)
            and not isinstance(gap_bytes, bool)
            and gap_bytes == count * BULLET_GAP_NODE_SIZE,
            "fired_gap_record_layout_invalid",
        )
        end = cursor + gap_bytes
        _require(end <= len(gap_payload), "fired_gap_record_bounds_invalid")
        record_payload = gap_payload[cursor:end]
        _require(
            _sha256(
                record.get("gap_payload_sha256"),
                "fired_gap_payload_sha256_invalid",
            )
            == sha256_bytes(record_payload),
            "fired_gap_payload_sha256_mismatch",
        )
        node_addresses: list[int] = []
        for entry_index, entry in enumerate(entries):
            _require(isinstance(entry, Mapping), "fired_gap_entry_invalid")
            node_address = entry.get("node_address")
            _require(
                isinstance(node_address, int)
                and not isinstance(node_address, bool)
                and node_address > 0
                and entry.get("index", entry_index) == entry_index,
                "fired_gap_node_address_invalid",
            )
            node_addresses.append(node_address)
        semantic_entries: list[dict[str, int]] = []
        boundary_ids: set[int] = set()
        for entry_index, entry in enumerate(entries):
            node_offset = entry_index * BULLET_GAP_NODE_SIZE
            (
                next_node,
                previous_node,
                curve_index,
                gap_distance,
                boundary_ball_id,
            ) = struct.unpack_from("<IIiii", record_payload, node_offset)
            expected_previous = (
                sentinel
                if entry_index == 0
                else node_addresses[entry_index - 1]
            )
            expected_next = (
                sentinel
                if entry_index + 1 == count
                else node_addresses[entry_index + 1]
            )
            _require(
                previous_node == expected_previous
                and next_node == expected_next
                and entry.get("previous_node_address") == previous_node
                and entry.get("next_node_address") == next_node
                and entry.get("curve_index") == curve_index
                and entry.get("gap_distance") == gap_distance
                and entry.get("boundary_ball_id") == boundary_ball_id
                and 0 <= curve_index < BULLET_CURVE_POINT_COUNT
                and curve_points[curve_index] >= 0
                and gap_distance > 0
                and boundary_ball_id > 0
                and boundary_ball_id not in boundary_ids,
                "fired_gap_entry_semantics_mismatch",
            )
            boundary_ids.add(boundary_ball_id)
            semantic_entries.append(
                {
                    "curve_index": curve_index,
                    "gap_distance": gap_distance,
                    "boundary_ball_id": boundary_ball_id,
                }
            )
        decoded_subclass = decoded["subclass_fields"]
        decoded_subclass.update(
            {
                "gap_list_sentinel_address": sentinel,
                "gap_entry_count": count,
                "curve_points": curve_points,
                "gap_entries": semantic_entries,
            }
        )
        cursor = end
    _require(cursor == len(gap_payload), "fired_gap_artifact_size_mismatch")


def validate_probe_score_binding(
    probe: Mapping[str, Any],
    *,
    observed_score: int,
    observed_displayed_score: int,
    required: bool = False,
) -> Mapping[str, Any] | None:
    """Validate the score-selection contract independently of scan output.

    Version 1 is retained for already-published evidence and is intentionally
    limited to a stable frame where the accumulated and rolling display scores
    agree.  Version 2 records the two native Board fields independently and
    proves that every int32 pair is admissible, so a rolling display lag cannot
    become an outcome-dependent sample filter.
    """

    binding = probe.get("score_binding")
    if binding is None:
        _require(not required, "score_binding_missing")
        return None
    _require(isinstance(binding, Mapping), "score_binding_invalid")

    version = binding.get("version")
    common_fields = {
        "schema",
        "version",
        "mode",
        "score",
        "displayed_score",
        "scan_value",
    }
    if version == SCORE_BINDING_LEGACY_VERSION:
        _require(set(binding) == common_fields, "score_binding_fields_invalid")
        _require(
            binding.get("schema") == SCORE_BINDING_SCHEMA
            and binding.get("mode")
            in {"active_board_read_only", "declared"},
            "score_binding_schema_invalid",
        )
    elif version == SCORE_BINDING_INDEPENDENT_VERSION:
        _require(
            set(binding)
            == common_fields
            | {
                "score_offset",
                "displayed_score_offset",
                "acceptance_rule",
            },
            "score_binding_fields_invalid",
        )
        _require(
            binding.get("schema") == SCORE_BINDING_SCHEMA
            and binding.get("mode") == INDEPENDENT_SCORE_BINDING_MODE
            and binding.get("score_offset") == BOARD_SCORE_OFFSET
            and binding.get("displayed_score_offset")
            == BOARD_DISPLAYED_SCORE_OFFSET
            and binding.get("acceptance_rule")
            == INDEPENDENT_SCORE_BINDING_ACCEPTANCE_RULE
            and probe.get("evidence_classification")
            == FORMAL_FULL_STATE_CLASSIFICATION,
            "score_binding_independent_contract_invalid",
        )
    else:
        _require(False, "score_binding_version_invalid")

    score = binding.get("score")
    displayed_score = binding.get("displayed_score")
    scan_value = binding.get("scan_value")
    _require(
        all(
            isinstance(value, int)
            and not isinstance(value, bool)
            and -(2**31) <= value < 2**31
            for value in (score, displayed_score, scan_value)
        ),
        "score_binding_value_invalid",
    )
    _require(
        score == observed_score
        and displayed_score == observed_displayed_score,
        "score_binding_observation_mismatch",
    )
    _require(
        scan_value == score and probe.get("value") == score,
        "score_binding_scan_mismatch",
    )
    if version == SCORE_BINDING_LEGACY_VERSION:
        _require(score == displayed_score, "board_score_display_disagree")

    board = probe.get("active_board")
    _require(isinstance(board, Mapping), "score_binding_active_board_missing")
    _require(
        board.get("score_offset") == BOARD_SCORE_OFFSET
        and board.get("displayed_score_offset")
        == BOARD_DISPLAYED_SCORE_OFFSET
        and board.get("score") == score
        and board.get("displayed_score") == displayed_score,
        "score_binding_active_board_mismatch",
    )
    return binding


def validate_probe_payloads(
    probe_path: Path,
    *,
    expected_runtime_sha256: str | None = None,
    expected_dmo_sha256: str | None = None,
    expected_update: int | None = None,
    expected_score: int | None = None,
    allowed_paths: frozenset[Path] | None = None,
    require_freeze_state: bool = False,
) -> dict[str, Any]:
    """Recompute the gameplay semantics in one frozen memory probe."""

    probe_path = probe_path.resolve()
    probe = _load_probe(probe_path)
    _require(probe.get("schema") == MEMORY_PROBE_SCHEMA, "probe_schema_invalid")
    _require(
        probe.get("version") == MEMORY_PROBE_VERSION,
        "probe_version_invalid",
    )
    runtime_sha256 = _sha256(
        probe.get("runtime_executable_sha256"),
        "runtime_sha256_invalid",
    )
    dmo_sha256 = _sha256(probe.get("dmo_sha256"), "dmo_sha256_invalid")
    if expected_runtime_sha256 is not None:
        _require(
            runtime_sha256 == expected_runtime_sha256,
            "runtime_sha256_mismatch",
        )
    if expected_dmo_sha256 is not None:
        _require(dmo_sha256 == expected_dmo_sha256, "dmo_sha256_mismatch")
    framework_update = probe.get("framework_update")
    _require(
        isinstance(framework_update, int)
        and not isinstance(framework_update, bool)
        and framework_update >= 0,
        "framework_update_invalid",
    )
    if expected_update is not None:
        _require(framework_update == expected_update, "framework_update_mismatch")

    freeze_row = probe.get("freeze_state")
    if require_freeze_state:
        _require(isinstance(freeze_row, Mapping), "freeze_state_missing")
    decoded_freeze: dict[str, Any] | None = None
    freeze_path: Path | None = None
    if isinstance(freeze_row, Mapping):
        _require(
            freeze_row.get("schema") == "zuma-rl.pc-replay-freeze-state"
            and freeze_row.get("version") == 1,
            "freeze_state_schema_invalid",
        )
        _require(
            freeze_row.get("multiplier_offset")
            == FREEZE_STATE_MULTIPLIER_OFFSET,
            "freeze_state_multiplier_offset_invalid",
        )
        multiplier_address = freeze_row.get("multiplier_address")
        _require(
            isinstance(multiplier_address, int)
            and not isinstance(multiplier_address, bool)
            and multiplier_address > FREEZE_STATE_MULTIPLIER_OFFSET,
            "freeze_state_multiplier_address_invalid",
        )
        freeze_path, freeze_payload = _read_artifact(
            probe_path,
            freeze_row,
            expected_size=FREEZE_STATE_BYTES,
            maximum_size=FREEZE_STATE_BYTES,
            allowed_paths=allowed_paths,
        )
        decoded_freeze = decode_freeze_state(
            freeze_payload,
            multiplier_address=multiplier_address,
        )
        _require_semantics(
            freeze_row,
            decoded_freeze,
            reason="freeze_state_semantics_mismatch",
        )
        _require(
            decoded_freeze["update_count"] == framework_update,
            "freeze_state_update_mismatch",
        )
        _require(
            decoded_freeze["frame_time_ms"] == 10,
            "freeze_state_frame_time_mismatch",
        )
        expected_freeze_multiplier = (
            FORMAL_FULL_STATE_UPDATE_MULTIPLIER
            if probe.get("evidence_classification")
            == FORMAL_FULL_STATE_CLASSIFICATION
            else FROZEN_UPDATE_MULTIPLIER
        )
        _require(
            decoded_freeze["update_multiplier"]
            == expected_freeze_multiplier
            and decoded_freeze["update_multiplier"] < 0.1,
            "freeze_state_multiplier_not_frozen",
        )

    board_row = probe.get("active_board")
    _require(isinstance(board_row, Mapping), "active_board_missing")
    _require(
        board_row.get("schema") == ACTIVE_BOARD_SCHEMA,
        "active_board_schema_invalid",
    )
    board_version = board_row.get("version")
    _require(
        isinstance(board_version, int)
        and not isinstance(board_version, bool)
        and board_version in ACTIVE_BOARD_SUPPORTED_VERSIONS,
        "active_board_version_invalid",
    )
    used_paths: set[Path] = set()
    if freeze_path is not None:
        used_paths.add(freeze_path)
    board_path, board = _read_artifact(
        probe_path,
        board_row,
        expected_size=BOARD_DUMP_SIZE,
        maximum_size=BOARD_DUMP_SIZE,
        allowed_paths=allowed_paths,
    )
    used_paths.add(board_path)
    _require(_u32(board, 0) == BOARD_VTABLE, "board_vtable_mismatch")
    _require(
        _u32(board, BOARD_EMBEDDED_VTABLE_OFFSET) == BOARD_EMBEDDED_VTABLE,
        "board_embedded_vtable_mismatch",
    )
    _require(
        board_row["board_vtable"] == BOARD_VTABLE,
        "declared_board_vtable_mismatch",
    )
    _require(
        board_row["embedded_vtable_offset"] == BOARD_EMBEDDED_VTABLE_OFFSET
        and board_row["embedded_vtable"] == BOARD_EMBEDDED_VTABLE,
        "declared_board_embedded_vtable_mismatch",
    )
    score = _i32(board, BOARD_SCORE_OFFSET)
    score_target = _i32(board, BOARD_SCORE_TARGET_OFFSET)
    displayed_score = _i32(board, BOARD_DISPLAYED_SCORE_OFFSET)
    _require(
        board_row["score_offset"] == BOARD_SCORE_OFFSET
        and board_row["score"] == score,
        "board_score_semantics_mismatch",
    )
    _require(
        board_row["score_target_offset"] == BOARD_SCORE_TARGET_OFFSET
        and board_row["score_target"] == score_target,
        "board_score_target_semantics_mismatch",
    )
    _require(
        board_row["displayed_score_offset"] == BOARD_DISPLAYED_SCORE_OFFSET
        and board_row["displayed_score"] == displayed_score,
        "board_displayed_score_semantics_mismatch",
    )
    score_binding = validate_probe_score_binding(
        probe,
        observed_score=score,
        observed_displayed_score=displayed_score,
        required=(
            probe.get("evidence_classification")
            == FORMAL_FULL_STATE_CLASSIFICATION
        ),
    )
    if score_binding is None:
        _require(score == displayed_score, "board_score_display_disagree")
    if expected_score is not None:
        _require(score == expected_score, "board_expected_score_mismatch")
    _require(probe.get("value") == score, "probe_scan_value_score_mismatch")

    board_mode_flag_1064: bool | None = None
    curve_plan_exhausted: bool | None = None
    if board_version == ACTIVE_BOARD_VERSION:
        mode_raw = board[BOARD_MODE_FLAG_1064_OFFSET]
        _require(mode_raw in {0, 1}, "board_mode_flag_1064_raw_invalid")
        declared_mode = board_row.get("mode_flag_1064")
        _require(
            board_row.get("mode_flag_1064_offset")
            == BOARD_MODE_FLAG_1064_OFFSET
            and isinstance(declared_mode, bool)
            and declared_mode is bool(mode_raw),
            "board_mode_flag_1064_semantics_mismatch",
        )
        board_mode_flag_1064 = bool(mode_raw)

        exhausted_row = board_row.get("curve_plan_exhausted")
        _require(
            isinstance(exhausted_row, Mapping),
            "curve_plan_exhausted_missing",
        )
        _require(
            exhausted_row.get("address")
            == G_CURVE_PLAN_EXHAUSTED_ADDRESS,
            "curve_plan_exhausted_address_mismatch",
        )
        exhausted_path, exhausted_payload = _read_artifact(
            probe_path,
            exhausted_row,
            expected_size=1,
            maximum_size=1,
            allowed_paths=allowed_paths,
        )
        used_paths.add(exhausted_path)
        exhausted_raw = exhausted_payload[0]
        _require(
            exhausted_raw in {0, 1},
            "curve_plan_exhausted_raw_invalid",
        )
        declared_exhausted = exhausted_row.get("value")
        _require(
            isinstance(declared_exhausted, bool)
            and declared_exhausted is bool(exhausted_raw),
            "curve_plan_exhausted_semantics_mismatch",
        )
        curve_plan_exhausted = bool(exhausted_raw)

    shooter_row = board_row.get("primary_child")
    _require(isinstance(shooter_row, Mapping), "shooter_missing")
    _require(
        shooter_row["board_offset"] == BOARD_PRIMARY_CHILD_OFFSET,
        "shooter_board_offset_mismatch",
    )
    _require(
        _u32(board, BOARD_PRIMARY_CHILD_OFFSET) == shooter_row["address"],
        "shooter_board_pointer_mismatch",
    )
    shooter_path, shooter = _read_artifact(
        probe_path,
        shooter_row,
        expected_size=SHOOTER_OBJECT_SIZE,
        maximum_size=SHOOTER_OBJECT_SIZE,
        allowed_paths=allowed_paths,
    )
    used_paths.add(shooter_path)
    _require(_u32(shooter, 0) == SHOOTER_VTABLE, "shooter_vtable_mismatch")
    shooter_records = shooter_row.get("bullets")
    _require(isinstance(shooter_records, list), "shooter_bullets_invalid")
    decoded_shooter: dict[int, dict[str, Any]] = {}
    for pointer_offset in SHOOTER_BULLET_POINTER_OFFSETS:
        matches = [
            row
            for row in shooter_records
            if isinstance(row, Mapping)
            and row.get("shooter_pointer_offset") == pointer_offset
        ]
        _require(len(matches) == 1, "shooter_bullet_pointer_record_missing")
        record = matches[0]
        address = _u32(shooter, pointer_offset)
        _require(address != 0 and record["address"] == address, "shooter_bullet_pointer_mismatch")
        bullet_path, bullet_payload = _read_artifact(
            probe_path,
            record,
            expected_size=BULLET_OBJECT_SIZE,
            maximum_size=BULLET_OBJECT_SIZE,
            allowed_paths=allowed_paths,
        )
        used_paths.add(bullet_path)
        ball = decode_ball(
            bullet_payload,
            expected_vtable=BULLET_VTABLE,
        )
        subclass = decode_bullet_subclass(bullet_payload)
        _require_semantics(
            record["ball"],
            ball,
            reason="shooter_ball_semantics_mismatch",
        )
        _require_semantics(
            record["subclass_fields"],
            subclass,
            reason="shooter_bullet_semantics_mismatch",
        )
        decoded_shooter[pointer_offset] = {
            "ball": ball,
            "subclass_fields": subclass,
        }

    fired_row = board_row.get("fired_bullets")
    _require(isinstance(fired_row, Mapping), "fired_bullets_missing")
    _require(
        fired_row["board_container_offset"]
        == BOARD_FIRED_BULLET_LIST_OFFSET,
        "fired_bullet_container_offset_mismatch",
    )
    _require(
        fired_row["sentinel_address"]
        == _u32(board, BOARD_FIRED_BULLET_LIST_OFFSET + 4),
        "fired_bullet_sentinel_mismatch",
    )
    fired_count = _u32(board, BOARD_FIRED_BULLET_LIST_OFFSET + 8)
    _require(
        fired_count <= MAX_FIRED_BULLETS
        and fired_row["declared_count"] == fired_count
        and fired_row["traversed_count"] == fired_count,
        "fired_bullet_count_mismatch",
    )
    fired_path, fired_payload = _read_artifact(
        probe_path,
        fired_row,
        expected_size=fired_count * BULLET_OBJECT_SIZE,
        maximum_size=MAX_FIRED_BULLETS * BULLET_OBJECT_SIZE,
        allowed_paths=allowed_paths,
    )
    used_paths.add(fired_path)
    decoded_fired = _validate_payload_records(
        fired_payload,
        fired_row,
        object_size=BULLET_OBJECT_SIZE,
        expected_vtable=BULLET_VTABLE,
        bullet=True,
        maximum_count=MAX_FIRED_BULLETS,
    )
    if fired_row.get("gap_artifact") is not None:
        declared_gap_sizes = [
            record.get("gap_artifact_bytes")
            if isinstance(record, Mapping)
            else None
            for record in fired_row["records"]
        ]
        _require(
            all(
                isinstance(value, int)
                and not isinstance(value, bool)
                and value >= 0
                for value in declared_gap_sizes
            ),
            "fired_gap_record_size_invalid",
        )
        gap_path, gap_payload = _read_artifact(
            probe_path,
            {
                "artifact": fired_row.get("gap_artifact"),
                "artifact_bytes": fired_row.get("gap_artifact_bytes"),
                "artifact_sha256": fired_row.get("gap_artifact_sha256"),
            },
            expected_size=sum(declared_gap_sizes),
            maximum_size=(
                MAX_FIRED_BULLETS
                * MAX_BULLET_GAP_ENTRIES
                * BULLET_GAP_NODE_SIZE
            ),
            allowed_paths=allowed_paths,
        )
        used_paths.add(gap_path)
        _validate_fired_gap_records(
            gap_payload,
            fired_payload,
            fired_row,
            decoded_fired,
        )
    else:
        _require(
            all(
                isinstance(record, Mapping)
                and "gap_artifact_offset" not in record
                and (
                    not isinstance(record.get("subclass_fields"), Mapping)
                    or "gap_entries" not in record["subclass_fields"]
                )
                for record in fired_row["records"]
            ),
            "fired_gap_artifact_missing",
        )

    manager_row = board_row.get("curve_manager")
    _require(isinstance(manager_row, Mapping), "curve_manager_missing")
    _require(
        manager_row["board_offset"] == BOARD_CURVE_MANAGER_OFFSET,
        "curve_manager_board_offset_mismatch",
    )
    _require(
        manager_row["address"] == _u32(board, BOARD_CURVE_MANAGER_OFFSET),
        "curve_manager_board_pointer_mismatch",
    )
    manager_dump_size = (
        CURVE_MANAGER_DUMP_SIZE
        if board_version == ACTIVE_BOARD_VERSION
        else CURVE_MANAGER_LEGACY_DUMP_SIZE
    )
    manager_path, manager = _read_artifact(
        probe_path,
        manager_row,
        expected_size=manager_dump_size,
        maximum_size=CURVE_MANAGER_DUMP_SIZE,
        allowed_paths=allowed_paths,
    )
    used_paths.add(manager_path)
    _require(
        _u32(manager, 0) == CURVE_MANAGER_VTABLE,
        "curve_manager_vtable_mismatch",
    )
    curve_count = _i32(manager, CURVE_MANAGER_CURVE_COUNT_OFFSET)
    _require(0 < curve_count <= MAX_CURVES, "curve_count_invalid")
    _require(curve_count == 1, "memory_evidence_requires_one_curve")
    _require(
        manager_row["curve_count_offset"]
        == CURVE_MANAGER_CURVE_COUNT_OFFSET
        and manager_row["curve_array_offset"]
        == CURVE_MANAGER_CURVE_ARRAY_OFFSET
        and manager_row["curve_count"] == curve_count,
        "curve_manager_semantics_mismatch",
    )
    post_zuma_diagnostics: dict[str, int | float] | None = None
    if board_version == ACTIVE_BOARD_VERSION:
        diagnostics_row = manager_row.get("post_zuma_diagnostics")
        _require(
            isinstance(diagnostics_row, Mapping),
            "curve_manager_post_zuma_diagnostics_missing",
        )
        post_zuma_timer = _i32(
            manager,
            CURVE_MANAGER_POST_ZUMA_TIMER_OFFSET,
        )
        post_zuma_ramp_404 = struct.unpack_from(
            "<f",
            manager,
            CURVE_MANAGER_POST_ZUMA_RAMP_404_OFFSET,
        )[0]
        post_zuma_ramp_408 = struct.unpack_from(
            "<f",
            manager,
            CURVE_MANAGER_POST_ZUMA_RAMP_408_OFFSET,
        )[0]
        _require(
            math.isfinite(post_zuma_ramp_404)
            and math.isfinite(post_zuma_ramp_408),
            "curve_manager_post_zuma_diagnostics_raw_invalid",
        )
        post_zuma_diagnostics = {
            "timer_remaining_offset": (
                CURVE_MANAGER_POST_ZUMA_TIMER_OFFSET
            ),
            "timer_remaining": post_zuma_timer,
            "ramp_404_offset": CURVE_MANAGER_POST_ZUMA_RAMP_404_OFFSET,
            "ramp_404": post_zuma_ramp_404,
            "ramp_408_offset": CURVE_MANAGER_POST_ZUMA_RAMP_408_OFFSET,
            "ramp_408": post_zuma_ramp_408,
        }
        _require_semantics(
            diagnostics_row,
            post_zuma_diagnostics,
            reason="curve_manager_post_zuma_diagnostics_semantics_mismatch",
        )
    curves = manager_row.get("curves")
    _require(
        isinstance(curves, list) and len(curves) == curve_count,
        "curve_records_count_mismatch",
    )
    decoded_lists: dict[int, list[dict[str, Any]]] = {}
    curve_plans: list[dict[str, Any]] = []
    for curve_index, curve_row in enumerate(curves):
        _require(
            isinstance(curve_row, Mapping)
            and curve_row.get("index") == curve_index,
            "curve_record_invalid",
        )
        curve_address = _u32(
            manager,
            CURVE_MANAGER_CURVE_ARRAY_OFFSET + curve_index * 4,
        )
        _require(
            curve_row["address"] == curve_address,
            "curve_pointer_mismatch",
        )
        curve_path, curve_payload = _read_artifact(
            probe_path,
            curve_row,
            expected_size=CURVE_DUMP_SIZE,
            maximum_size=CURVE_DUMP_SIZE,
            allowed_paths=allowed_paths,
        )
        used_paths.add(curve_path)
        _require(
            _u32(curve_payload, 0) == CURVE_VTABLE,
            "curve_vtable_mismatch",
        )
        if board_version == ACTIVE_BOARD_VERSION:
            plan_row = curve_row.get("planned_balls")
            _require(
                isinstance(plan_row, Mapping),
                "curve_plan_missing",
            )
            begin = _u32(
                curve_payload,
                CURVE_PLANNED_VECTOR_BEGIN_OFFSET,
            )
            end = _u32(
                curve_payload,
                CURVE_PLANNED_VECTOR_END_OFFSET,
            )
            capacity = _u32(
                curve_payload,
                CURVE_PLANNED_VECTOR_CAPACITY_OFFSET,
            )
            if begin == end == capacity == 0:
                planned_count = 0
                planned_capacity_count = 0
                planned_bytes = 0
            else:
                _require(
                    begin != 0
                    and end != 0
                    and capacity != 0
                    and begin <= end <= capacity,
                    "curve_plan_vector_bounds_invalid",
                )
                planned_bytes = end - begin
                capacity_bytes = capacity - begin
                _require(
                    planned_bytes % CURVE_PLANNED_ITEM_SIZE == 0
                    and capacity_bytes % CURVE_PLANNED_ITEM_SIZE == 0,
                    "curve_plan_vector_alignment_invalid",
                )
                planned_count = (
                    planned_bytes // CURVE_PLANNED_ITEM_SIZE
                )
                planned_capacity_count = (
                    capacity_bytes // CURVE_PLANNED_ITEM_SIZE
                )
                _require(
                    planned_count <= MAX_CURVE_PLANNED_ITEMS
                    and planned_capacity_count <= MAX_CURVE_PLANNED_ITEMS,
                    "curve_plan_vector_count_invalid",
                )
            add_plan_raw = curve_payload[
                CURVE_ADD_PLAN_ENABLED_OFFSET
            ]
            _require(
                add_plan_raw in {0, 1},
                "curve_add_plan_enabled_raw_invalid",
            )
            plan_path, _ = _read_artifact(
                probe_path,
                plan_row,
                expected_size=planned_bytes,
                maximum_size=(
                    MAX_CURVE_PLANNED_ITEMS
                    * CURVE_PLANNED_ITEM_SIZE
                ),
                allowed_paths=allowed_paths,
            )
            used_paths.add(plan_path)
            plan_semantics = {
                "vector_begin_offset": (
                    CURVE_PLANNED_VECTOR_BEGIN_OFFSET
                ),
                "vector_end_offset": CURVE_PLANNED_VECTOR_END_OFFSET,
                "vector_capacity_offset": (
                    CURVE_PLANNED_VECTOR_CAPACITY_OFFSET
                ),
                "begin_address": begin,
                "end_address": end,
                "capacity_address": capacity,
                "item_size": CURVE_PLANNED_ITEM_SIZE,
                "count": planned_count,
                "capacity_count": planned_capacity_count,
                "add_plan_enabled_offset": (
                    CURVE_ADD_PLAN_ENABLED_OFFSET
                ),
                "add_plan_enabled": bool(add_plan_raw),
            }
            _require_semantics(
                plan_row,
                plan_semantics,
                reason="curve_plan_semantics_mismatch",
            )
            _require(
                isinstance(plan_row.get("add_plan_enabled"), bool),
                "curve_plan_add_enabled_type_invalid",
            )
            curve_plans.append(
                {
                    "curve_index": curve_index,
                    **plan_semantics,
                }
            )
        list_rows = curve_row.get("intrusive_lists")
        _require(isinstance(list_rows, list), "curve_lists_invalid")
        for container_offset in CURVE_INTRUSIVE_LIST_OFFSETS:
            matches = [
                row
                for row in list_rows
                if isinstance(row, Mapping)
                and row.get("container_offset") == container_offset
            ]
            _require(len(matches) == 1, "curve_list_record_missing")
            list_row = matches[0]
            _require(
                list_row["sentinel_address"]
                == _u32(curve_payload, container_offset + 4),
                "curve_list_sentinel_mismatch",
            )
            declared_count = _u32(curve_payload, container_offset + 8)
            _require(
                declared_count <= MAX_CURVE_LIST_ITEMS
                and list_row["declared_count"] == declared_count
                and list_row["traversed_count"] == declared_count,
                "curve_list_count_mismatch",
            )
            records = list_row.get("records")
            _require(
                isinstance(records, list) and len(records) == declared_count,
                "curve_list_records_count_mismatch",
            )
            _require(
                all(
                    isinstance(record, Mapping)
                    and bool(record.get("payload_address"))
                    for record in records
                ),
                "curve_list_null_payload_unsupported",
            )
            list_path, list_payload = _read_artifact(
                probe_path,
                list_row,
                expected_size=None,
                maximum_size=MAX_CURVE_LIST_ITEMS * BULLET_OBJECT_SIZE,
                allowed_paths=allowed_paths,
            )
            used_paths.add(list_path)
            decoded_lists[container_offset] = _validate_curve_payload_records(
                list_payload,
                list_row,
                allow_bullets=(
                    container_offset
                    == INSERTION_STAGING_LIST_OFFSET
                ),
                maximum_count=MAX_CURVE_LIST_ITEMS,
            )

    frame_row = probe.get("frozen_frame")
    _require(isinstance(frame_row, Mapping), "frozen_frame_missing")
    frame_path = _artifact_path(
        probe_path,
        frame_row["artifact"],
        allowed_paths=allowed_paths,
    )
    try:
        frame_size = frame_path.stat().st_size
    except OSError as error:
        raise PcMemoryEvidenceError("frozen_frame_missing") from error
    _require(0 < frame_size <= 800 * 600 * 4 + 4096, "frozen_frame_size_invalid")
    _require(
        _sha256(frame_row["sha256"], "frozen_frame_sha256_invalid")
        == sha256_path(frame_path),
        "frozen_frame_sha256_mismatch",
    )
    used_paths.add(frame_path)

    return {
        "schema": "zuma-rl.pc-memory-payload-validation",
        "version": 1,
        "status": "PASS",
        "probe_sha256": sha256_path(probe_path),
        "runtime_executable_sha256": runtime_sha256,
        "dmo_sha256": dmo_sha256,
        "framework_update": framework_update,
        "active_board_version": board_version,
        "freeze_state": decoded_freeze,
        "score": score,
        "displayed_score": displayed_score,
        "score_target": score_target,
        "board_mode_flag_1064": board_mode_flag_1064,
        "curve_plan_exhausted": curve_plan_exhausted,
        "post_zuma_diagnostics": post_zuma_diagnostics,
        "curve_plans": tuple(curve_plans),
        "active_chain_count": len(
            decoded_lists[ACTIVE_CHAIN_LIST_OFFSET]
        ),
        "fired_bullet_count": len(decoded_fired),
        "shooter_current": decoded_shooter[
            SHOOTER_BULLET_POINTER_OFFSETS[0]
        ],
        "shooter_next": decoded_shooter[
            SHOOTER_BULLET_POINTER_OFFSETS[1]
        ],
        "active_chain": decoded_lists[ACTIVE_CHAIN_LIST_OFFSET],
        "fired_bullets": decoded_fired,
        "frozen_frame_path": frame_path,
        "used_artifact_paths": frozenset(used_paths),
    }


def _circular_distance(left: float, right: float) -> float:
    distance = abs(left - right)
    return min(distance, 1.0 - distance)


def _classify_patch(
    pixels: Any,
    *,
    width: int,
    height: int,
    center_x: float,
    center_y: float,
) -> tuple[int, int, dict[int, float]]:
    scores = {color_id: 0.0 for color_id in PROTOTYPE_HUES}
    used_pixels = 0
    left = max(0, math.floor(center_x - SAMPLE_RADIUS))
    right = min(width - 1, math.ceil(center_x + SAMPLE_RADIUS))
    top = max(0, math.floor(center_y - SAMPLE_RADIUS))
    bottom = min(height - 1, math.ceil(center_y + SAMPLE_RADIUS))
    radius_squared = SAMPLE_RADIUS * SAMPLE_RADIUS
    for y in range(top, bottom + 1):
        for x in range(left, right + 1):
            if (x - center_x) ** 2 + (y - center_y) ** 2 > radius_squared:
                continue
            red, green, blue = pixels[x, y]
            hue, saturation, value = colorsys.rgb_to_hsv(
                red / 255.0,
                green / 255.0,
                blue / 255.0,
            )
            if saturation < MINIMUM_SATURATION or value < MINIMUM_VALUE:
                continue
            used_pixels += 1
            weight = saturation * value
            for color_id, prototype in PROTOTYPE_HUES.items():
                distance = _circular_distance(hue, prototype)
                scores[color_id] += weight * math.exp(
                    -((distance / HUE_KERNEL_WIDTH) ** 2)
                )
    _require(used_pixels > 0, "ball_patch_has_no_admissible_pixels")
    predicted = max(scores, key=scores.__getitem__)
    return predicted, used_pixels, scores


def _classify_ball(
    ball: Mapping[str, Any],
    *,
    index: int,
    pixels: Any,
    width: int,
    height: int,
    allow_rollout_occlusion: bool,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    ball_id = ball["ball_id"]
    color_id = ball["color_id"]
    position_x = ball["position_x"]
    position_y = ball["position_y"]
    reason: str | None = None
    if color_id not in PROTOTYPE_HUES:
        reason = "unsupported_color_id"
    elif (
        allow_rollout_occlusion
        and position_x < ROLLOUT_FOREGROUND_OCCLUSION_X
        and position_y < ROLLOUT_FOREGROUND_OCCLUSION_Y
    ):
        reason = "rollout_entrance_foreground_occlusion"
    elif (
        position_x < SAMPLE_RADIUS
        or position_x > width - 1 - SAMPLE_RADIUS
        or position_y < SAMPLE_RADIUS
        or position_y > height - 1 - SAMPLE_RADIUS
    ):
        reason = "sample_disk_outside_frame"
    identity = {
        "index": index,
        "ball_id": ball_id,
        "color_id": color_id,
        "position_x": position_x,
        "position_y": position_y,
    }
    if reason is not None:
        return None, {**identity, "reason": reason}
    predicted, used_pixels, scores = _classify_patch(
        pixels,
        width=width,
        height=height,
        center_x=position_x,
        center_y=position_y,
    )
    expected_score = scores[color_id]
    competing_score = max(
        value
        for candidate, value in scores.items()
        if candidate != color_id
    )
    return (
        {
            **identity,
            "predicted_color_id": predicted,
            "match": predicted == color_id,
            "used_pixels": used_pixels,
            "expected_score": expected_score,
            "strongest_competing_score": competing_score,
            "score_margin": expected_score - competing_score,
        },
        None,
    )


def _active_chain_row(probe: Mapping[str, Any]) -> Mapping[str, Any]:
    curves = probe["active_board"]["curve_manager"]["curves"]
    _require(len(curves) == 1, "pixel_validation_requires_one_curve")
    matches = [
        row
        for row in curves[0]["intrusive_lists"]
        if row["container_offset"] == ACTIVE_CHAIN_LIST_OFFSET
    ]
    _require(len(matches) == 1, "active_chain_list_missing")
    return matches[0]


def validate_probe_pixels(
    probe_path: Path,
    *,
    minimum_visible: int,
    maximum_mismatches: int,
    minimum_visible_fired_bullets: int,
    maximum_fired_bullet_mismatches: int,
    expected_runtime_sha256: str | None = None,
    expected_dmo_sha256: str | None = None,
    expected_update: int | None = None,
    expected_score: int | None = None,
    allowed_paths: frozenset[Path] | None = None,
    require_freeze_state: bool = True,
    expected_dimensions: tuple[int, int] = (800, 600),
) -> dict[str, Any]:
    """Recompute the memory-to-BMP color agreement for one probe."""

    _require(minimum_visible >= 1, "minimum_visible_invalid")
    _require(maximum_mismatches >= 0, "maximum_mismatches_invalid")
    _require(
        minimum_visible_fired_bullets >= 0,
        "minimum_visible_fired_bullets_invalid",
    )
    _require(
        maximum_fired_bullet_mismatches >= 0,
        "maximum_fired_bullet_mismatches_invalid",
    )
    raw = validate_probe_payloads(
        probe_path,
        expected_runtime_sha256=expected_runtime_sha256,
        expected_dmo_sha256=expected_dmo_sha256,
        expected_update=expected_update,
        expected_score=expected_score,
        allowed_paths=allowed_paths,
        require_freeze_state=require_freeze_state,
    )
    probe = _load_probe(probe_path)
    try:
        from PIL import Image
    except ImportError as error:
        raise PcMemoryImageDecoderUnavailable(
            "Pillow is required for frozen BMP verification"
        ) from error
    try:
        with Image.open(raw["frozen_frame_path"]) as source:
            _require(source.format == "BMP", "frozen_frame_format_invalid")
            image = source.convert("RGB")
    except PcMemoryEvidenceError:
        raise
    except Exception as error:
        raise PcMemoryEvidenceError("frozen_frame_decode_failed") from error
    width, height = image.size
    _require(
        (width, height) == expected_dimensions,
        "frozen_frame_dimensions_mismatch",
    )
    pixels = image.load()

    active_results: list[dict[str, Any]] = []
    active_excluded: list[dict[str, Any]] = []
    for index, record in enumerate(raw["active_chain"]):
        result, exclusion = _classify_ball(
            record["ball"],
            index=index,
            pixels=pixels,
            width=width,
            height=height,
            allow_rollout_occlusion=True,
        )
        if result is not None:
            active_results.append(result)
        if exclusion is not None:
            active_excluded.append(exclusion)
    fired_results: list[dict[str, Any]] = []
    fired_excluded: list[dict[str, Any]] = []
    for index, record in enumerate(raw["fired_bullets"]):
        result, exclusion = _classify_ball(
            record["ball"],
            index=index,
            pixels=pixels,
            width=width,
            height=height,
            allow_rollout_occlusion=False,
        )
        if result is not None:
            fired_results.append(result)
        if exclusion is not None:
            fired_excluded.append(exclusion)
    active_mismatches = [
        row for row in active_results if not row["match"]
    ]
    fired_mismatches = [
        row for row in fired_results if not row["match"]
    ]
    _require(
        len(active_results) >= minimum_visible,
        "too_few_visible_balls",
    )
    _require(
        len(active_mismatches) <= maximum_mismatches,
        "ball_pixel_mismatch_limit_exceeded",
    )
    _require(
        len(fired_results) >= minimum_visible_fired_bullets,
        "too_few_visible_fired_bullets",
    )
    _require(
        len(fired_mismatches) <= maximum_fired_bullet_mismatches,
        "fired_bullet_pixel_mismatch_limit_exceeded",
    )
    chain_row = _active_chain_row(probe)
    fired_row = probe["active_board"]["fired_bullets"]
    frame_row = probe["frozen_frame"]
    color_ids = sorted({row["color_id"] for row in active_results})
    freeze = raw["freeze_state"]
    return {
        "schema": "zuma-rl.pc-ball-pixel-validation",
        "version": 4,
        "status": "PASS",
        "memory_probe": {
            "artifact": probe_path.name,
            "sha256": raw["probe_sha256"],
            "framework_update": raw["framework_update"],
        },
        "ball_payload": {
            "artifact": chain_row["artifact"],
            "sha256": chain_row["artifact_sha256"],
            "total_count": len(raw["active_chain"]),
            "retail_vtable_hex": f"0x{BALL_VTABLE:08x}",
            "retail_object_bytes": BALL_OBJECT_SIZE,
        },
        "frozen_frame": {
            "artifact": frame_row["artifact"],
            "sha256": frame_row["sha256"],
            "width": width,
            "height": height,
        },
        "raw_payload_validation": {
            "schema": raw["schema"],
            "version": raw["version"],
            "status": raw["status"],
            "probe_sha256": raw["probe_sha256"],
            "active_chain_count": raw["active_chain_count"],
            "fired_bullet_count": raw["fired_bullet_count"],
            "freeze_update_count": (
                freeze["update_count"] if freeze is not None else None
            ),
            "freeze_update_multiplier": (
                freeze["update_multiplier"] if freeze is not None else None
            ),
        },
        "method": {
            "active_chain_list_offset_hex": (
                f"0x{ACTIVE_CHAIN_LIST_OFFSET:03x}"
            ),
            "sample_radius": SAMPLE_RADIUS,
            "minimum_saturation": MINIMUM_SATURATION,
            "minimum_value": MINIMUM_VALUE,
            "hue_kernel_width": HUE_KERNEL_WIDTH,
            "rollout_foreground_occlusion_x": (
                ROLLOUT_FOREGROUND_OCCLUSION_X
            ),
            "rollout_foreground_occlusion_y": (
                ROLLOUT_FOREGROUND_OCCLUSION_Y
            ),
            "prototype_hues": {
                str(key): value
                for key, value in sorted(PROTOTYPE_HUES.items())
            },
        },
        "visible_count": len(active_results),
        "excluded_count": len(active_excluded),
        "match_count": len(active_results) - len(active_mismatches),
        "mismatch_count": len(active_mismatches),
        "accuracy": (
            (len(active_results) - len(active_mismatches))
            / len(active_results)
        ),
        "per_color": {
            str(color_id): {
                "visible": sum(
                    row["color_id"] == color_id
                    for row in active_results
                ),
                "matches": sum(
                    row["color_id"] == color_id and row["match"]
                    for row in active_results
                ),
            }
            for color_id in color_ids
        },
        "fired_bullets": {
            "payload": {
                "artifact": fired_row["artifact"],
                "sha256": fired_row["artifact_sha256"],
                "total_count": len(raw["fired_bullets"]),
                "retail_vtable_hex": f"0x{BULLET_VTABLE:08x}",
                "retail_object_bytes": BULLET_OBJECT_SIZE,
                "board_container_offset_hex": (
                    fired_row["board_container_offset_hex"]
                ),
            },
            "visible_count": len(fired_results),
            "excluded_count": len(fired_excluded),
            "match_count": len(fired_results) - len(fired_mismatches),
            "mismatch_count": len(fired_mismatches),
            "accuracy": (
                (len(fired_results) - len(fired_mismatches))
                / len(fired_results)
                if fired_results
                else 1.0
            ),
            "excluded": fired_excluded,
            "results": fired_results,
        },
        "excluded": active_excluded,
        "results": active_results,
    }


def canonical_report_bytes(report: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            report,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")


def read_canonical_report(path: Path) -> dict[str, Any]:
    report = _load_probe(path)
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise PcMemoryEvidenceError("canonical_report_unreadable") from error
    _require(
        payload == canonical_report_bytes(report),
        "canonical_report_encoding_mismatch",
    )
    return report


def _ball_identity(record: Mapping[str, Any]) -> dict[str, int]:
    ball = record["ball"]
    return {
        "ball_id": int(ball["ball_id"]),
        "color_id": int(ball["color_id"]),
    }


def _load_pixel_report(path: Path) -> dict[str, Any]:
    report = _load_probe(path)
    _require(
        report.get("schema") == "zuma-rl.pc-ball-pixel-validation"
        and report.get("version") == 4
        and report.get("status") == "PASS",
        "pixel_report_schema_invalid",
    )
    return report


def validate_shot_transition(
    before_probe_path: Path,
    after_probe_path: Path,
    before_pixel_path: Path,
    after_pixel_path: Path,
    *,
    expected_before_update: int,
    expected_after_update: int,
    expected_score: int,
    expected_chain_distance_delta: float,
    distance_tolerance: float,
    minimum_chain_count: int,
    minimum_visible: int,
    maximum_mismatches: int,
    minimum_visible_after_fired_bullets: int,
    maximum_fired_bullet_mismatches: int,
    expected_runtime_sha256: str | None = None,
    expected_dmo_sha256: str | None = None,
    allowed_paths: frozenset[Path] | None = None,
) -> dict[str, Any]:
    """Independently prove one no-hit retail shot transition."""

    before_probe_path = before_probe_path.resolve()
    after_probe_path = after_probe_path.resolve()
    before_pixel_path = before_pixel_path.resolve()
    after_pixel_path = after_pixel_path.resolve()
    if allowed_paths is not None:
        _require(
            before_probe_path in allowed_paths
            and after_probe_path in allowed_paths
            and before_pixel_path in allowed_paths
            and after_pixel_path in allowed_paths,
            "transition_artifact_not_declared_by_manifest",
        )
    before_raw = validate_probe_payloads(
        before_probe_path,
        expected_runtime_sha256=expected_runtime_sha256,
        expected_dmo_sha256=expected_dmo_sha256,
        expected_update=expected_before_update,
        expected_score=expected_score,
        allowed_paths=allowed_paths,
        require_freeze_state=True,
    )
    after_raw = validate_probe_payloads(
        after_probe_path,
        expected_runtime_sha256=before_raw["runtime_executable_sha256"],
        expected_dmo_sha256=before_raw["dmo_sha256"],
        expected_update=expected_after_update,
        expected_score=expected_score,
        allowed_paths=allowed_paths,
        require_freeze_state=True,
    )
    recomputed_before_pixels = validate_probe_pixels(
        before_probe_path,
        minimum_visible=minimum_visible,
        maximum_mismatches=maximum_mismatches,
        minimum_visible_fired_bullets=0,
        maximum_fired_bullet_mismatches=(
            maximum_fired_bullet_mismatches
        ),
        expected_runtime_sha256=before_raw["runtime_executable_sha256"],
        expected_dmo_sha256=before_raw["dmo_sha256"],
        expected_update=expected_before_update,
        expected_score=expected_score,
        allowed_paths=allowed_paths,
        require_freeze_state=True,
    )
    recomputed_after_pixels = validate_probe_pixels(
        after_probe_path,
        minimum_visible=minimum_visible,
        maximum_mismatches=maximum_mismatches,
        minimum_visible_fired_bullets=(
            minimum_visible_after_fired_bullets
        ),
        maximum_fired_bullet_mismatches=(
            maximum_fired_bullet_mismatches
        ),
        expected_runtime_sha256=before_raw["runtime_executable_sha256"],
        expected_dmo_sha256=before_raw["dmo_sha256"],
        expected_update=expected_after_update,
        expected_score=expected_score,
        allowed_paths=allowed_paths,
        require_freeze_state=True,
    )
    stored_before_pixels = _load_pixel_report(before_pixel_path)
    stored_after_pixels = _load_pixel_report(after_pixel_path)
    _require(
        stored_before_pixels == recomputed_before_pixels,
        "before_pixel_report_not_canonical",
    )
    _require(
        stored_after_pixels == recomputed_after_pixels,
        "after_pixel_report_not_canonical",
    )
    _require(
        len(before_raw["fired_bullets"]) == 0
        and recomputed_before_pixels["fired_bullets"]["visible_count"] == 0,
        "before_fired_list_not_empty",
    )
    _require(
        len(after_raw["fired_bullets"]) == 1
        and recomputed_after_pixels["fired_bullets"]["visible_count"] == 1
        and recomputed_after_pixels["fired_bullets"]["match_count"] == 1,
        "after_fired_bullet_evidence_invalid",
    )
    before_chain = before_raw["active_chain"]
    after_chain = after_raw["active_chain"]
    _require(
        len(before_chain) == len(after_chain)
        and len(before_chain) >= minimum_chain_count,
        "active_chain_count_invalid",
    )
    _require(
        [_ball_identity(row) for row in before_chain]
        == [_ball_identity(row) for row in after_chain],
        "active_chain_identity_or_color_changed",
    )
    distance_deltas = [
        after_row["ball"]["curve_distance"]
        - before_row["ball"]["curve_distance"]
        for before_row, after_row in zip(before_chain, after_chain)
    ]
    _require(
        all(
            math.isfinite(value)
            and abs(value - expected_chain_distance_delta)
            <= distance_tolerance
            for value in distance_deltas
        ),
        "active_chain_distance_delta_mismatch",
    )
    before_current = before_raw["shooter_current"]
    before_next = before_raw["shooter_next"]
    after_current = after_raw["shooter_current"]
    after_next = after_raw["shooter_next"]
    transferred = after_raw["fired_bullets"][0]
    _require(
        _ball_identity(after_current) == _ball_identity(before_next),
        "shooter_next_was_not_promoted",
    )
    _require(
        _ball_identity(transferred) == _ball_identity(before_current),
        "shooter_current_was_not_transferred",
    )
    _require(
        _ball_identity(after_next)
        not in (
            _ball_identity(before_current),
            _ball_identity(before_next),
        ),
        "shooter_next_was_not_replenished",
    )
    fired_pixel = recomputed_after_pixels["fired_bullets"]["results"][0]
    _require(
        {
            "ball_id": fired_pixel["ball_id"],
            "color_id": fired_pixel["color_id"],
        }
        == _ball_identity(transferred),
        "fired_bullet_pixel_identity_mismatch",
    )
    velocity_x = transferred["subclass_fields"]["velocity_x"]
    velocity_y = transferred["subclass_fields"]["velocity_y"]
    speed = math.hypot(velocity_x, velocity_y)
    travel_x = (
        transferred["ball"]["position_x"]
        - before_current["ball"]["position_x"]
    )
    travel_y = (
        transferred["ball"]["position_y"]
        - before_current["ball"]["position_y"]
    )
    travel_distance = math.hypot(travel_x, travel_y)
    _require(
        math.isfinite(speed)
        and speed > 0.0
        and math.isfinite(travel_distance)
        and travel_distance > 0.0,
        "fired_bullet_motion_invalid",
    )
    return {
        "schema": "zuma-rl.pc-memory-transition-validation",
        "version": 2,
        "status": "PASS",
        "inputs": {
            "before_memory_probe": {
                "artifact": before_probe_path.name,
                "sha256": before_raw["probe_sha256"],
            },
            "after_memory_probe": {
                "artifact": after_probe_path.name,
                "sha256": after_raw["probe_sha256"],
            },
            "before_pixel_validation": {
                "artifact": before_pixel_path.name,
                "sha256": sha256_path(before_pixel_path),
                "active_visible_count": recomputed_before_pixels[
                    "visible_count"
                ],
                "active_match_count": recomputed_before_pixels[
                    "match_count"
                ],
                "fired_visible_count": recomputed_before_pixels[
                    "fired_bullets"
                ]["visible_count"],
                "fired_match_count": recomputed_before_pixels[
                    "fired_bullets"
                ]["match_count"],
            },
            "after_pixel_validation": {
                "artifact": after_pixel_path.name,
                "sha256": sha256_path(after_pixel_path),
                "active_visible_count": recomputed_after_pixels[
                    "visible_count"
                ],
                "active_match_count": recomputed_after_pixels[
                    "match_count"
                ],
                "fired_visible_count": recomputed_after_pixels[
                    "fired_bullets"
                ]["visible_count"],
                "fired_match_count": recomputed_after_pixels[
                    "fired_bullets"
                ]["match_count"],
            },
            "runtime_executable_sha256": before_raw[
                "runtime_executable_sha256"
            ],
            "dmo_sha256": before_raw["dmo_sha256"],
            "raw_payloads_recomputed": True,
        },
        "updates": {
            "before": expected_before_update,
            "after": expected_after_update,
            "delta": expected_after_update - expected_before_update,
        },
        "score": {
            "before": before_raw["score"],
            "after": after_raw["score"],
            "displayed_before": before_raw["displayed_score"],
            "displayed_after": after_raw["displayed_score"],
            "target": before_raw["score_target"],
        },
        "active_chain": {
            "retail_list_offset_hex": (
                f"0x{ACTIVE_CHAIN_LIST_OFFSET:03x}"
            ),
            "count": len(before_chain),
            "ordered_identities_and_colors_equal": True,
            "expected_curve_distance_delta": (
                expected_chain_distance_delta
            ),
            "distance_tolerance": distance_tolerance,
            "minimum_curve_distance_delta": min(distance_deltas),
            "maximum_curve_distance_delta": max(distance_deltas),
            "mean_curve_distance_delta": (
                sum(distance_deltas) / len(distance_deltas)
            ),
        },
        "shot_transition": {
            "shooter_current_pointer_offset_hex": "0x130",
            "shooter_next_pointer_offset_hex": "0x134",
            "board_fired_list_offset_hex": (
                f"0x{BOARD_FIRED_BULLET_LIST_OFFSET:03x}"
            ),
            "before_current": _ball_identity(before_current),
            "before_next": _ball_identity(before_next),
            "after_current": _ball_identity(after_current),
            "after_next": _ball_identity(after_next),
            "transferred_fired_bullet": {
                **_ball_identity(transferred),
                "position_x": transferred["ball"]["position_x"],
                "position_y": transferred["ball"]["position_y"],
                "velocity_x": velocity_x,
                "velocity_y": velocity_y,
                "speed": speed,
                "travel_from_before_current": travel_distance,
                "transient_flag_16a_after_transfer": transferred[
                    "subclass_fields"
                ]["fired"],
            },
        },
    }


__all__ = [
    "ACTIVE_CHAIN_LIST_OFFSET",
    "BALL_OBJECT_SIZE",
    "BALL_VTABLE",
    "BOARD_FIRED_BULLET_LIST_OFFSET",
    "BULLET_OBJECT_SIZE",
    "BULLET_VTABLE",
    "FROZEN_UPDATE_MULTIPLIER",
    "FORMAL_FULL_STATE_CLASSIFICATION",
    "FORMAL_FULL_STATE_UPDATE_MULTIPLIER",
    "INDEPENDENT_SCORE_BINDING_ACCEPTANCE_RULE",
    "INDEPENDENT_SCORE_BINDING_MODE",
    "MEMORY_CURVE_GEOMETRY_ARTIFACT",
    "MEMORY_CURVE_GEOMETRY_SCHEMA",
    "MEMORY_CURVE_GEOMETRY_TOLERANCE_PX",
    "MEMORY_CURVE_GEOMETRY_VERSION",
    "MEMORY_PROBE_SCHEMA",
    "SCORE_BINDING_INDEPENDENT_VERSION",
    "SCORE_BINDING_LEGACY_VERSION",
    "SCORE_BINDING_SCHEMA",
    "MEMORY_TRANSITION_CONTRACT_ARTIFACT",
    "MEMORY_TRANSITION_CONTRACT_SCHEMA",
    "MEMORY_TRANSITION_CONTRACT_VERSION",
    "PcMemoryEvidenceError",
    "PcMemoryImageDecoderUnavailable",
    "PcMemoryTransitionContract",
    "build_memory_curve_geometry_binding",
    "canonical_report_bytes",
    "decode_ball",
    "decode_bullet_subclass",
    "decode_freeze_state",
    "sha256_bytes",
    "sha256_path",
    "read_canonical_report",
    "validate_probe_payloads",
    "validate_probe_pixels",
    "validate_probe_score_binding",
    "validate_shot_transition",
]
