"""Machine-derived PC mechanism coverage bound to one verified native source.

The fidelity suite must not turn handwritten feature labels into evidence.  A
mechanism audit therefore starts from a retail memory trajectory, verifies its
raw payloads, binds the trajectory's first frozen frame to the corresponding
PC Golden video tick, and derives only mechanisms that are unambiguous in the
recomputed trajectory events.

The audit intentionally does not replace PC Golden verification.  The fidelity
gate accepts a mechanism authorization only when the referenced native-source
fingerprint also belongs to a PC Golden case that independently passes the full
verifier in the same gate run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np

from zuma_rl.actor_audit import (
    AUDIT_SCHEMA,
    AUDIT_VERSION,
    POLICY_ID,
)
from zuma_rl.pc_evidence import PcTickMap
from zuma_rl.pc_exact_step_evidence import (
    ExactStepLoadedRun,
    PcExactStepEvidenceError,
    decode_top_down_bgra_bmp,
    load_formal_exact_step_run,
)
from zuma_rl.pc_golden import (
    ExactStepRunContract,
    PcGoldenArtifactError,
    PcGoldenManifest,
    PcGoldenValidationError,
    canonical_sha256,
    pc_golden_native_source_fingerprint,
)
from zuma_rl.pc_full_state_evidence import (
    FullStateLoadedRun,
    PcFullStateEvidenceError,
    load_formal_full_state_run,
)
from zuma_rl.pc_memory_evidence import (
    PcMemoryEvidenceError,
    PcMemoryImageDecoderUnavailable,
    read_canonical_report,
    validate_probe_score_binding,
    validate_probe_payloads,
)
from zuma_rl.pc_memory_trajectory import (
    ACTIVE_CHAIN_LIST_OFFSET,
    INSERTION_STAGING_LIST_OFFSET,
    POWERUP_NONE_TYPE,
    POWERUP_TYPE_COUNT,
    TRAJECTORY_SCHEMA,
    TRAJECTORY_VERSION,
    PcMemoryTrajectoryError,
    TrajectoryCurvePowerupState,
    TrajectoryEntity,
    TrajectoryFrame,
    TrajectoryMTRandState,
    TrajectoryQRandState,
    derive_trajectory_events,
    load_legacy_memory_trajectory_v1,
    load_memory_trajectory,
)
from zuma_rl.pc_loss_sequence import (
    PcLossSequenceError,
    verify_loss_sequence_frames,
)
from zuma_rl.pc_natural_win import derive_natural_win_feature_proofs
from zuma_rl.pc_powerup_trigger import (
    PROXIMITY_BOMB_POWERUP_TYPE,
    REVERSE_POWERUP_TYPE,
    SLOW_POWERUP_TYPE,
    PcPowerupTriggerError,
    verify_powerup_trigger_frames,
)
from zuma_rl.pc_source import (
    FULL_STATE_SOURCE_KIND,
    FULL_STATE_SOURCE_VERSION,
    SOURCE_BOUND_FULL_STATE_SOURCE_KIND,
    SOURCE_BOUND_FULL_STATE_SOURCE_VERSION,
    SOURCE_SCHEMA as PC_SOURCE_SCHEMA,
    PcSourceValidationError,
    verify_pc_source_manifest,
)
from zuma_rl.pc_video import (
    PcVideoError,
    canonical_rgb24_frame_sha256,
    decode_pc_video_rgb24_at_pts,
)
from zuma_rl.original_data import (
    OriginalCurve,
    OriginalDataError,
    OriginalGameCatalog,
)
from zuma_rl.revenge_core import (
    SUPPORTED_PROFILE_MODE,
    FruitSpawnCalibration,
    MsvcCRTRandom,
    PopCapMTRandom,
    PowerupSpawnCalibration,
    PowerupType,
    _BalancedColorChooser,
)

AUDIT_TYPE = "pc_mechanism_coverage"
HISTORICAL_AUDIT_VERSION = 1
HISTORICAL_POLICY_ID = "original-transfer-jungle2-v1"
PLAN_SCHEMA = "zuma-rl.pc-mechanism-audit-plan"
PLAN_VERSION = 1
MAX_MECHANISM_AUDIT_CASES = 64
VIDEO_BINDING_EXCLUDED_BOTTOM_ROWS = 4
CURVE_FIT_TOLERANCE_PX = 1e-3
INSERTION_DIRECTION_CROSS_EPSILON_PX = 1e-3
POWERUP_TRANSITION_TICKS = 100
POWERUP_REVERSE_TICKS = 300
POWERUP_REVERSE_SPEED = 1.0
POWERUP_SLOW_TICKS = 800
POWERUP_BOMB_COLLISION_PAD = 56
POWERUP_VISUAL_SCALE = 5.0
POWERUP_VISUAL_STEP = float(
    np.divide(np.float32(4.0), np.float32(POWERUP_TRANSITION_TICKS))
)
POWERUP_VISUAL_INDEX_BY_TYPE = {
    int(PowerupType.PROXIMITY_BOMB): 3,
    int(PowerupType.SLOW): 2,
    int(PowerupType.REVERSE): 4,
}
FRUIT_LIFETIME_TICKS = 1_000
FRUIT_INITIAL_GLOW_STEP = 12
FRUIT_POWERUP_COLLISION_RADIUS = 108.0
POWERUP_FEATURE_BY_TYPE = {
    PROXIMITY_BOMB_POWERUP_TYPE: "powerup_proximity_bomb",
    SLOW_POWERUP_TYPE: "powerup_slow",
    REVERSE_POWERUP_TYPE: "powerup_reverse",
}

_CASE_ID_PATTERN = re.compile(r"[a-z0-9]+(?:[a-z0-9_-]*[a-z0-9])?")
_SUPPORTED_DERIVED_FEATURES = frozenset(
    {
        "back_insertion",
        "front_insertion",
        "fruit_expiry",
        "fruit_collection_animation",
        "fruit_collection_score",
        "fruit_powerup_collision",
        "fruit_projectile_collision",
        "fruit_scheduler_spawn",
        "fruit_visual_oscillator",
        "gap_shot",
        "match3",
        "match4",
        "natural_loss",
        "natural_win",
        "powerup_proximity_bomb",
        "powerup_reverse",
        "powerup_slow",
        "powerup_spawn",
        "projectile_collision",
        "rng_pending",
        "rng_rejection",
        "rollback_chain",
        "shot_release",
        "swap",
        "tunnel_collision",
        "zuma_transition",
    }
)


class PcMechanismAuditError(ValueError):
    """A mechanism-audit input or its retail provenance is invalid."""


def _fail(code: str) -> None:
    raise PcMechanismAuditError(code)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _reports_semantically_equivalent(
    submitted: Mapping[str, Any],
    live: Mapping[str, Any],
) -> bool:
    """Compare a current report or one immutable pre-v2 mechanism audit.

    The v1-to-v2 audit migration changed only the schema version, policy label,
    and fingerprint.  Verify the submitted historical fingerprint before
    normalizing exactly those metadata fields; all retail-derived content must
    still equal the current live recomputation.
    """

    if submitted == live:
        return True
    scope = submitted.get("scope")
    if (
        submitted.get("version") != HISTORICAL_AUDIT_VERSION
        or not isinstance(scope, Mapping)
        or scope.get("policy") != HISTORICAL_POLICY_ID
    ):
        return False
    submitted_fingerprint = submitted.get("audit_fingerprint")
    fingerprint_payload = dict(submitted)
    fingerprint_payload.pop("audit_fingerprint", None)
    if (
        not isinstance(submitted_fingerprint, str)
        or submitted_fingerprint != canonical_sha256(fingerprint_payload)
    ):
        return False

    normalized = dict(submitted)
    normalized["version"] = live.get("version")
    normalized_scope = dict(scope)
    live_scope = live.get("scope")
    if not isinstance(live_scope, Mapping):
        return False
    normalized_scope["policy"] = live_scope.get("policy")
    normalized["scope"] = normalized_scope
    normalized["audit_fingerprint"] = live.get("audit_fingerprint")
    return normalized == live


def _strict_int(value: Any, code: str, *, minimum: int = 0) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
    ):
        _fail(code)
    return value


def _strict_json(path: Path) -> Mapping[str, Any]:
    def reject_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _fail("mechanism_audit_json_duplicate_key")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        _fail(f"mechanism_audit_json_nonfinite:{value}")

    try:
        payload = path.read_text(encoding="utf-8")
        value = json.loads(
            payload,
            object_pairs_hook=reject_pairs,
            parse_constant=reject_constant,
        )
    except PcMechanismAuditError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PcMechanismAuditError(
            "mechanism_audit_json_invalid"
        ) from error
    if not isinstance(value, Mapping):
        _fail("mechanism_audit_json_root_invalid")
    return value


def _relative_file(
    root: Path,
    value: Any,
    *,
    code: str,
) -> tuple[str, Path]:
    if not isinstance(value, str) or not value:
        _fail(code)
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or not relative.parts
        or "\\" in value
        or any(part in {"", ".", ".."} for part in relative.parts)
        or relative.as_posix() != value
    ):
        _fail(code)
    candidate = root.joinpath(*relative.parts)
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise PcMechanismAuditError(code) from error
    if not resolved.is_file():
        _fail(code)
    return value, resolved


def _child_artifact(
    parent: Path,
    value: Any,
    *,
    code: str,
) -> Path:
    if not isinstance(value, str) or not value:
        _fail(code)
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or not relative.parts
        or "\\" in value
        or any(part in {"", ".", ".."} for part in relative.parts)
        or relative.as_posix() != value
    ):
        _fail(code)
    root = parent.resolve()
    candidate = root.joinpath(*relative.parts)
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise PcMechanismAuditError(code) from error
    if not resolved.is_file():
        _fail(code)
    return resolved


def _case_spec(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail("mechanism_audit_case_invalid")
    if set(value) != {
        "id",
        "manifest_path",
        "memory_probe_path",
        "window",
    }:
        _fail("mechanism_audit_case_fields_invalid")
    case_id = value.get("id")
    if (
        not isinstance(case_id, str)
        or _CASE_ID_PATTERN.fullmatch(case_id) is None
    ):
        _fail("mechanism_audit_case_id_invalid")
    manifest_path = value.get("manifest_path")
    memory_probe_path = value.get("memory_probe_path")
    if not isinstance(manifest_path, str) or not isinstance(
        memory_probe_path, str
    ):
        _fail("mechanism_audit_case_path_invalid")
    window = value.get("window")
    if not isinstance(window, Mapping) or set(window) != {
        "start_update",
        "end_update",
    }:
        _fail("mechanism_audit_case_window_invalid")
    start = _strict_int(
        window.get("start_update"),
        "mechanism_audit_case_start_update_invalid",
    )
    end = _strict_int(
        window.get("end_update"),
        "mechanism_audit_case_end_update_invalid",
    )
    if end <= start:
        _fail("mechanism_audit_case_window_invalid")
    return {
        "id": case_id,
        "manifest_path": manifest_path,
        "memory_probe_path": memory_probe_path,
        "window": {
            "start_update": start,
            "end_update": end,
        },
    }


def read_mechanism_audit_plan(path: str | Path) -> tuple[dict[str, Any], ...]:
    """Read a strict plan that selects evidence windows but declares no features."""

    plan = _strict_json(Path(path))
    if (
        plan.get("schema") != PLAN_SCHEMA
        or plan.get("version") != PLAN_VERSION
        or set(plan) != {"schema", "version", "cases"}
    ):
        _fail("mechanism_audit_plan_contract_invalid")
    cases = plan.get("cases")
    if (
        isinstance(cases, (str, bytes))
        or not isinstance(cases, Sequence)
        or not cases
        or len(cases) > MAX_MECHANISM_AUDIT_CASES
    ):
        _fail("mechanism_audit_plan_cases_invalid")
    normalized = tuple(_case_spec(item) for item in cases)
    ids = [item["id"] for item in normalized]
    if len(ids) != len(set(ids)):
        _fail("mechanism_audit_case_id_duplicate")
    path_pairs = [
        (item["manifest_path"], item["memory_probe_path"])
        for item in normalized
    ]
    if len(path_pairs) != len(set(path_pairs)):
        _fail("mechanism_audit_case_source_duplicate")
    return normalized


def _trajectory_from_probe(
    probe: Mapping[str, Any],
    *,
    probe_path: Path,
) -> tuple[Path, tuple[TrajectoryFrame, ...], Mapping[str, Any]]:
    row = probe.get("trajectory")
    if not isinstance(row, Mapping):
        _fail("mechanism_audit_trajectory_binding_missing")
    required = {
        "artifact",
        "artifact_sha256",
        "end_update",
        "schema",
        "start_update",
        "tick_count",
        "version",
    }
    allowed = required | {"freeze_update"}
    if not required.issubset(row) or not set(row).issubset(allowed):
        _fail("mechanism_audit_trajectory_binding_fields_invalid")
    if "freeze_update" in row:
        freeze_update = row.get("freeze_update")
        if (
            isinstance(freeze_update, bool)
            or not isinstance(freeze_update, int)
            or freeze_update != row.get("start_update")
        ):
            _fail("mechanism_audit_trajectory_freeze_mismatch")
    if (
        row.get("schema") != TRAJECTORY_SCHEMA
        or row.get("version") != TRAJECTORY_VERSION
    ):
        _fail("mechanism_audit_trajectory_binding_schema_invalid")
    trajectory_path = _child_artifact(
        probe_path.parent,
        row.get("artifact"),
        code="mechanism_audit_trajectory_path_invalid",
    )
    if row.get("artifact_sha256") != _sha256_path(trajectory_path):
        _fail("mechanism_audit_trajectory_sha256_mismatch")
    frames = load_memory_trajectory(trajectory_path)
    if (
        row.get("start_update") != frames[0].update
        or row.get("end_update") != frames[-1].update
        or row.get("tick_count") != len(frames)
        or len(frames) != frames[-1].update - frames[0].update + 1
    ):
        _fail("mechanism_audit_trajectory_summary_mismatch")
    return trajectory_path, frames, row


def _require_natural_probe(
    probe: Mapping[str, Any],
) -> Mapping[str, Any]:
    if "diagnostic_mutation" not in probe:
        _fail("mechanism_audit_mutation_attestation_missing")
    if probe.get("diagnostic_mutation") is not None:
        _fail("mechanism_audit_diagnostic_mutation_forbidden")
    repaint = probe.get("repaint")
    if (
        not isinstance(repaint, Mapping)
        or repaint.get("schema") != "zuma-rl.pc-window-repaint-handshake"
        or repaint.get("version") != 1
        or repaint.get("effectful_command_guard_satisfied") is not True
        or repaint.get("geometry_restored") is not True
        or repaint.get("foreground_activation_verified") is not True
    ):
        _fail("mechanism_audit_repaint_provenance_invalid")
    frame = probe.get("frozen_frame")
    if not isinstance(frame, Mapping) or frame.get("status") != "PASS":
        _fail("mechanism_audit_frozen_frame_provenance_invalid")
    board = probe.get("active_board")
    if not isinstance(board, Mapping):
        _fail("mechanism_audit_score_binding_invalid")
    try:
        score_binding = validate_probe_score_binding(
            probe,
            observed_score=_strict_int(
                board.get("score"),
                "mechanism_audit_score_binding_invalid",
            ),
            observed_displayed_score=_strict_int(
                board.get("displayed_score"),
                "mechanism_audit_score_binding_invalid",
            ),
            required=True,
        )
    except PcMemoryEvidenceError:
        _fail("mechanism_audit_score_binding_invalid")
    if score_binding is None:
        _fail("mechanism_audit_score_binding_invalid")
    return score_binding


def _bind_probe_frame_to_video(
    *,
    manifest: PcGoldenManifest,
    manifest_root: Path,
    probe_validation: Mapping[str, Any],
) -> dict[str, Any]:
    update = _strict_int(
        probe_validation.get("framework_update"),
        "mechanism_audit_probe_update_invalid",
    )
    native_tick = update + manifest.input_timeline.native_tick_offset
    if not 0 <= native_tick <= manifest.clock.tick_end:
        _fail("mechanism_audit_probe_outside_pc_window")

    tick_map_spec = manifest.artifacts[manifest.clock.tick_map_artifact]
    tick_map_path = tick_map_spec.verify(manifest_root)
    tick_map = PcTickMap.read(tick_map_path)
    if native_tick >= len(tick_map.records):
        _fail("mechanism_audit_tick_map_too_short")
    pts = tick_map.records[native_tick].pts

    video_spec = manifest.artifacts[manifest.video.artifact]
    video_path = manifest_root.joinpath(
        *PurePosixPath(video_spec.path).parts
    ).resolve()
    try:
        video_path.relative_to(manifest_root.resolve())
    except ValueError as error:
        raise PcMechanismAuditError(
            "mechanism_audit_video_path_escape"
        ) from error
    video_payload = decode_pc_video_rgb24_at_pts(
        video_path,
        metadata=manifest.video,
        artifact=video_spec,
        requested_pts=(pts,),
    )[pts]
    if len(video_payload) != 800 * 600 * 3:
        _fail("mechanism_audit_video_frame_size_invalid")
    video_rgb = np.frombuffer(video_payload, dtype=np.uint8).reshape(
        (600, 800, 3)
    )

    try:
        from PIL import Image
    except ImportError as error:
        raise PcMemoryImageDecoderUnavailable(
            "Pillow is required for mechanism memory/video binding"
        ) from error
    frame_path = probe_validation.get("frozen_frame_path")
    if not isinstance(frame_path, Path):
        _fail("mechanism_audit_probe_frame_path_invalid")
    try:
        with Image.open(frame_path) as source:
            if source.format != "BMP" or source.size != (800, 600):
                _fail("mechanism_audit_probe_frame_invalid")
            probe_rgb = np.asarray(source.convert("RGB"))
    except PcMechanismAuditError:
        raise
    except Exception as error:
        raise PcMechanismAuditError(
            "mechanism_audit_probe_frame_decode_failed"
        ) from error
    difference = np.any(probe_rgb != video_rgb, axis=2)
    interior_height = 600 - VIDEO_BINDING_EXCLUDED_BOTTOM_ROWS
    interior_mismatches = int(difference[:interior_height].sum())
    edge_mismatches = int(difference[interior_height:].sum())
    if interior_mismatches:
        _fail("mechanism_audit_probe_video_gameplay_pixels_differ")
    if edge_mismatches > 800 * VIDEO_BINDING_EXCLUDED_BOTTOM_ROWS:
        _fail("mechanism_audit_probe_video_edge_budget_exceeded")

    probe_payload = np.ascontiguousarray(probe_rgb).tobytes()
    interior_payload = np.ascontiguousarray(
        probe_rgb[:interior_height]
    ).tobytes()
    return {
        "status": "PASS",
        "framework_update": update,
        "native_tick": native_tick,
        "video_pts": pts,
        "interior_height": interior_height,
        "interior_mismatch_count": interior_mismatches,
        "excluded_bottom_rows": VIDEO_BINDING_EXCLUDED_BOTTOM_ROWS,
        "excluded_edge_mismatch_count": edge_mismatches,
        "probe_rgb24_sha256": (
            f"sha256:{hashlib.sha256(probe_payload).hexdigest()}"
        ),
        "video_rgb24_sha256": (
            f"sha256:{hashlib.sha256(video_payload).hexdigest()}"
        ),
        "interior_rgb24_sha256": (
            f"sha256:{hashlib.sha256(interior_payload).hexdigest()}"
        ),
    }


def _formal_mapping(value: Any, code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(code)
    return value


def _formal_sequence(value: Any, code: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _fail(code)
    return value


def _formal_float(value: Any, code: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        _fail(code)
    return float(value)


def _formal_entity(
    value: Any,
    *,
    zone: str,
    index: int,
) -> TrajectoryEntity | None:
    if value is None:
        return None
    row = _formal_mapping(value, "mechanism_audit_exact_step_entity_invalid")
    kind = row.get("kind")
    if kind not in {"ball", "bullet"}:
        _fail("mechanism_audit_exact_step_entity_invalid")
    ball_id = _strict_int(
        row.get("ball_id"),
        "mechanism_audit_exact_step_entity_invalid",
    )
    color_id = _strict_int(
        row.get("color_id"),
        "mechanism_audit_exact_step_entity_invalid",
    )
    if ball_id > 0xFFFFFFFF or color_id > 5:
        _fail("mechanism_audit_exact_step_entity_invalid")
    powerup_types = tuple(
        _strict_int(
            row.get(name),
            "mechanism_audit_exact_step_entity_invalid",
        )
        for name in (
            "powerup_previous_type",
            "powerup_primary_type",
            "powerup_secondary_type",
        )
    )
    if any(value > POWERUP_NONE_TYPE for value in powerup_types):
        _fail("mechanism_audit_exact_step_entity_invalid")
    flags_hex = row.get("flags_b4_c2_hex")
    try:
        flags = bytes.fromhex(flags_hex)
    except (TypeError, ValueError) as error:
        raise PcMechanismAuditError(
            "mechanism_audit_exact_step_entity_invalid"
        ) from error
    if len(flags) != 15:
        _fail("mechanism_audit_exact_step_entity_invalid")
    radius = _formal_float(
        row.get("radius"),
        "mechanism_audit_exact_step_entity_invalid",
    )
    if radius <= 0.0:
        _fail("mechanism_audit_exact_step_entity_invalid")
    return TrajectoryEntity(
        ball_id=ball_id,
        color_id=color_id,
        object_kind=kind,
        zone=zone,
        index=index,
        curve_distance=_formal_float(
            row.get("curve_distance"),
            "mechanism_audit_exact_step_entity_invalid",
        ),
        position_x=_formal_float(
            row.get("position_x"),
            "mechanism_audit_exact_step_entity_invalid",
        ),
        position_y=_formal_float(
            row.get("position_y"),
            "mechanism_audit_exact_step_entity_invalid",
        ),
        radius=radius,
        # Formal exact-step rows contain shooter and curve-container
        # entities only; the separately counted fired-bullet container is
        # deliberately not projected.  A Bullet found in any represented
        # container is therefore known to have completed the native
        # fired->container transition (or never to have been fired).  Keep
        # that derivable false value instead of degrading it to ``None`` so
        # chamber-only mechanisms such as a pure swap remain provable.
        fired=False if kind == "bullet" else None,
        contact_next=bool(flags[0]),
        exploding=bool(flags[6]),
        explode_frame=int.from_bytes(flags[8:12], "little", signed=True),
        should_remove=bool(flags[12]),
        powerup_previous_type=powerup_types[0],
        powerup_primary_type=powerup_types[1],
        powerup_secondary_type=powerup_types[2],
    )


def _formal_identity(entity: TrajectoryEntity | None) -> tuple[Any, ...] | None:
    if entity is None:
        return None
    return (
        entity.ball_id,
        entity.color_id,
        entity.object_kind,
        entity.powerup_previous_type,
        entity.powerup_primary_type,
        entity.powerup_secondary_type,
    )


def _rng_identity(value: Any) -> tuple[Any, ...] | None:
    if value is None:
        return None
    return (
        value.ball_id,
        value.color_id,
        value.kind,
        value.powerup_previous_type,
        value.powerup_primary_type,
        value.powerup_secondary_type,
    )


def _formal_exact_step_frames(
    source: ExactStepLoadedRun,
) -> tuple[TrajectoryFrame, ...]:
    """Project a fully validated exact-step source into mechanism frames."""

    if len(source.tick_rows) != len(source.frames):
        _fail("mechanism_audit_exact_step_frame_count_mismatch")
    result: list[TrajectoryFrame] = []
    for raw, rng_frame in zip(source.tick_rows, source.frames, strict=True):
        row = _formal_mapping(
            raw,
            "mechanism_audit_exact_step_tick_invalid",
        )
        update = _strict_int(
            row.get("framework_update"),
            "mechanism_audit_exact_step_tick_invalid",
        )
        if update != rng_frame.update:
            _fail("mechanism_audit_exact_step_tick_invalid")
        current = _formal_entity(
            row.get("current_ball"),
            zone="shooter_current",
            index=0,
        )
        following = _formal_entity(
            row.get("next_ball"),
            zone="shooter_next",
            index=0,
        )
        entities: list[TrajectoryEntity] = [
            entity for entity in (current, following) if entity is not None
        ]
        list_counts: list[tuple[int, int, int]] = []
        observed_lists: dict[
            tuple[int, int], tuple[TrajectoryEntity, ...]
        ] = {}
        seen_lists: set[tuple[int, int]] = set()
        for value in _formal_sequence(
            row.get("curve_lists"),
            "mechanism_audit_exact_step_curve_lists_invalid",
        ):
            curve = _formal_mapping(
                value,
                "mechanism_audit_exact_step_curve_list_invalid",
            )
            curve_index = _strict_int(
                curve.get("curve_index"),
                "mechanism_audit_exact_step_curve_list_invalid",
            )
            offset = _strict_int(
                curve.get("container_offset"),
                "mechanism_audit_exact_step_curve_list_invalid",
            )
            key = (curve_index, offset)
            if offset not in {0x50, 0x5C, 0x68} or key in seen_lists:
                _fail("mechanism_audit_exact_step_curve_list_invalid")
            seen_lists.add(key)
            raw_entities = _formal_sequence(
                curve.get("entities"),
                "mechanism_audit_exact_step_curve_list_invalid",
            )
            declared_count = _strict_int(
                curve.get("declared_count"),
                "mechanism_audit_exact_step_curve_list_invalid",
            )
            traversed_count = _strict_int(
                curve.get("traversed_count"),
                "mechanism_audit_exact_step_curve_list_invalid",
            )
            if declared_count != traversed_count or declared_count != len(
                raw_entities
            ):
                _fail("mechanism_audit_exact_step_curve_list_invalid")
            zone = f"curve:{curve_index}:list:{offset:03x}"
            decoded = tuple(
                entity
                for entity in (
                    _formal_entity(value, zone=zone, index=index)
                    for index, value in enumerate(raw_entities)
                )
                if entity is not None
            )
            if len(decoded) != declared_count:
                _fail("mechanism_audit_exact_step_curve_list_invalid")
            entities.extend(decoded)
            list_counts.append((curve_index, offset, declared_count))
            observed_lists[key] = decoded
        if len({entity.ball_id for entity in entities}) != len(entities):
            _fail("mechanism_audit_exact_step_entity_identity_duplicate")

        qrand_vectors = dict(rng_frame.qrand_vectors)
        if set(qrand_vectors) != {
            "weights",
            "sways",
            "last_hit",
            "previous_hit",
        }:
            _fail("mechanism_audit_exact_step_rng_projection_mismatch")

        frame = TrajectoryFrame(
            update=update,
            score=_strict_int(
                row.get("score"),
                "mechanism_audit_exact_step_tick_invalid",
            ),
            displayed_score=_strict_int(
                row.get("displayed_score"),
                "mechanism_audit_exact_step_tick_invalid",
            ),
            score_target=_strict_int(
                row.get("score_target"),
                "mechanism_audit_exact_step_tick_invalid",
            ),
            current_ball_id=None if current is None else current.ball_id,
            current_color_id=None if current is None else current.color_id,
            next_ball_id=None if following is None else following.ball_id,
            next_color_id=None if following is None else following.color_id,
            list_counts=tuple(sorted(list_counts)),
            entities=tuple(entities),
            qrand=TrajectoryQRandState(
                update_count=rng_frame.qrand_update_count,
                selected_index=rng_frame.qrand_selected_index,
                weights=tuple(float(value) for value in qrand_vectors["weights"]),
                sways=tuple(float(value) for value in qrand_vectors["sways"]),
                last_hit=tuple(int(value) for value in qrand_vectors["last_hit"]),
                previous_hit=tuple(
                    int(value) for value in qrand_vectors["previous_hit"]
                ),
            ),
            thread_crt_rand_state=rng_frame.thread_crt_rand_state,
            global_mtrand=TrajectoryMTRandState(
                index=rng_frame.mtrand_index,
                words=tuple(rng_frame.mtrand_words),
            ),
        )
        if (
            frame.score != rng_frame.score
            or frame.displayed_score != rng_frame.displayed_score
            or frame.score_target != rng_frame.score_target
            or _formal_identity(current) != _rng_identity(rng_frame.current_ball)
            or _formal_identity(following) != _rng_identity(rng_frame.next_ball)
        ):
            _fail("mechanism_audit_exact_step_rng_projection_mismatch")
        expected_keys = {
            (item.curve_index, item.container_offset)
            for item in rng_frame.curve_lists
        }
        if set(observed_lists) != expected_keys:
            _fail("mechanism_audit_exact_step_rng_projection_mismatch")
        for rng_list in rng_frame.curve_lists:
            observed = observed_lists[
                (rng_list.curve_index, rng_list.container_offset)
            ]
            if tuple(_formal_identity(item) for item in observed) != tuple(
                _rng_identity(item) for item in rng_list.entities
            ):
                _fail("mechanism_audit_exact_step_rng_projection_mismatch")
        result.append(frame)
    return tuple(result)


def _load_formal_exact_step_mechanism_source(
    *,
    manifest: PcGoldenManifest,
    manifest_root: Path,
    probe_path: Path,
) -> tuple[
    ExactStepRunContract,
    ExactStepLoadedRun,
    tuple[TrajectoryFrame, ...],
    Mapping[str, Path],
]:
    contract = manifest.exact_step_replay
    if contract is None:
        _fail("mechanism_audit_exact_step_contract_missing")
    verified_artifacts = manifest.verify_artifacts(manifest_root)
    matching_runs = [
        run
        for run in contract.runs
        if verified_artifacts[run.memory_probe_artifact].resolve()
        == probe_path.resolve()
    ]
    if len(matching_runs) != 1:
        _fail("mechanism_audit_exact_step_probe_run_binding_invalid")
    run = matching_runs[0]
    loaded = load_formal_exact_step_run(
        run_id=run.run_id,
        selected_attempt=run.selected_attempt,
        attempts_path=verified_artifacts[run.attempts_artifact],
        probe_path=verified_artifacts[run.memory_probe_artifact],
        index_path=verified_artifacts[run.trajectory_index_artifact],
        expected_process_id=run.process_id,
        expected_process_creation_filetime_100ns=(
            run.process_creation_filetime_100ns
        ),
        expected_runtime_sha256=manifest.pc_environment.executable_sha256,
        expected_dmo_sha256=(
            manifest.artifacts[manifest.input_timeline.artifact].sha256
        ),
        expected_freeze_update=contract.freeze_update,
        expected_start_update=contract.source_start_update,
        expected_end_update=contract.source_end_update,
        expected_warmup_tick_count=contract.warmup_tick_count,
        maximum_startup_attempts=contract.maximum_startup_attempts,
        declared_artifact_paths=tuple(verified_artifacts.values()),
    )
    return run, loaded, _formal_exact_step_frames(loaded), verified_artifacts


def _pc_source_binding(
    evidence_root: Path,
    value: Any,
    *,
    code: str,
) -> tuple[str, Path, str]:
    row = _formal_mapping(value, code)
    if set(row) != {"path", "sha256"}:
        _fail(code)
    relative, path = _relative_file(
        evidence_root,
        row.get("path"),
        code=code,
    )
    digest = row.get("sha256")
    if not isinstance(digest, str) or digest != _sha256_path(path):
        _fail(code)
    return relative, path, digest


def _load_pc_source_mechanism_source(
    *,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    evidence_root: Path,
    original_root: Path,
    probe_path: Path,
) -> tuple[
    Mapping[str, Any],
    ExactStepLoadedRun | FullStateLoadedRun,
    tuple[TrajectoryFrame, ...],
    Path,
    str,
]:
    """Revalidate and project one independently authentic retail source."""

    try:
        source_report = verify_pc_source_manifest(
            manifest_path,
            evidence_root=evidence_root,
            original_root=original_root,
        )
    except PcSourceValidationError as error:
        raise PcMechanismAuditError(str(error)) from error

    run = _formal_mapping(
        manifest.get("run"),
        "mechanism_audit_pc_source_run_invalid",
    )
    window = _formal_mapping(
        manifest.get("window"),
        "mechanism_audit_pc_source_window_invalid",
    )
    _, attempts_path, _ = _pc_source_binding(
        evidence_root,
        run.get("attempts"),
        code="mechanism_audit_pc_source_attempts_invalid",
    )
    _, bound_probe_path, _ = _pc_source_binding(
        evidence_root,
        run.get("memory_probe"),
        code="mechanism_audit_pc_source_probe_invalid",
    )
    trajectory_relative, index_path, index_sha256 = _pc_source_binding(
        evidence_root,
        run.get("trajectory_index"),
        code="mechanism_audit_pc_source_trajectory_invalid",
    )
    if bound_probe_path.resolve() != probe_path.resolve():
        _fail("mechanism_audit_pc_source_probe_binding_invalid")
    _, _, runtime_sha256 = _pc_source_binding(
        evidence_root,
        manifest.get("runtime_payload"),
        code="mechanism_audit_pc_source_runtime_invalid",
    )
    _, _, dmo_sha256 = _pc_source_binding(
        evidence_root,
        manifest.get("dmo"),
        code="mechanism_audit_pc_source_dmo_invalid",
    )
    try:
        full_state = (
            manifest.get("version"),
            run.get("source_kind"),
        ) in {
            (FULL_STATE_SOURCE_VERSION, FULL_STATE_SOURCE_KIND),
            (
                SOURCE_BOUND_FULL_STATE_SOURCE_VERSION,
                SOURCE_BOUND_FULL_STATE_SOURCE_KIND,
            ),
        }
        loader = (
            load_formal_full_state_run
            if full_state
            else load_formal_exact_step_run
        )
        loaded = loader(
            run_id=str(run.get("run_id")),
            selected_attempt=_strict_int(
                run.get("selected_attempt"),
                "mechanism_audit_pc_source_run_invalid",
            ),
            attempts_path=attempts_path,
            probe_path=bound_probe_path,
            index_path=index_path,
            expected_process_id=_strict_int(
                run.get("process_id"),
                "mechanism_audit_pc_source_run_invalid",
            ),
            expected_process_creation_filetime_100ns=_strict_int(
                run.get("process_creation_filetime_100ns"),
                "mechanism_audit_pc_source_run_invalid",
            ),
            expected_runtime_sha256=runtime_sha256,
            expected_dmo_sha256=dmo_sha256,
            expected_freeze_update=_strict_int(
                window.get("freeze_update"),
                "mechanism_audit_pc_source_window_invalid",
            ),
            expected_start_update=_strict_int(
                window.get("start_update"),
                "mechanism_audit_pc_source_window_invalid",
            ),
            expected_end_update=_strict_int(
                window.get("end_update"),
                "mechanism_audit_pc_source_window_invalid",
            ),
            expected_warmup_tick_count=_strict_int(
                window.get("warmup_tick_count"),
                "mechanism_audit_pc_source_window_invalid",
            ),
            maximum_startup_attempts=_strict_int(
                run.get("maximum_startup_attempts"),
                "mechanism_audit_pc_source_run_invalid",
            ),
        )
    except (PcExactStepEvidenceError, PcFullStateEvidenceError) as error:
        raise PcMechanismAuditError(str(error)) from error
    return (
        source_report,
        loaded,
        (
            loaded.frames
            if isinstance(loaded, FullStateLoadedRun)
            else _formal_exact_step_frames(loaded)
        ),
        index_path,
        index_sha256,
    )


def _bind_pc_source_frame(
    *,
    source: ExactStepLoadedRun | FullStateLoadedRun,
    evidence_root: Path,
    update: int,
) -> dict[str, Any]:
    matches = [frame for frame in source.visual_frames if frame.update == update]
    if len(matches) != 1:
        _fail("mechanism_audit_pc_source_visual_update_invalid")
    frame = matches[0]
    try:
        relative = PurePosixPath(
            *frame.path.resolve().relative_to(evidence_root).parts
        ).as_posix()
    except ValueError as error:
        raise PcMechanismAuditError(
            "mechanism_audit_pc_source_visual_path_invalid"
        ) from error
    return {
        "status": "PASS",
        "transport": (
            "single_retail_process_full_state_exact_step_source"
            if isinstance(source, FullStateLoadedRun)
            else "single_retail_process_exact_step_source"
        ),
        "framework_update": update,
        "artifact": relative,
        "bmp_sha256": frame.bmp_sha256,
        "rgb24_sha256": frame.rgb24_sha256,
        "canonical_rgb24_sha256": frame.pc_video_rgb24_sha256,
    }


def _bind_exact_step_frame_to_video(
    *,
    manifest: PcGoldenManifest,
    manifest_root: Path,
    run: ExactStepRunContract,
    source: ExactStepLoadedRun,
    verified_artifacts: Mapping[str, Path],
    update: int,
) -> dict[str, Any]:
    contract = manifest.exact_step_replay
    if contract is None:
        _fail("mechanism_audit_exact_step_contract_missing")
    native_tick = update - contract.source_start_update
    if (
        native_tick != update + manifest.input_timeline.native_tick_offset
        or not 0 <= native_tick < len(source.visual_frames)
        or native_tick > run.clock.tick_end
    ):
        _fail("mechanism_audit_exact_step_update_out_of_bounds")
    tick_map = PcTickMap.read(
        verified_artifacts[run.clock.tick_map_artifact]
    )
    if native_tick >= len(tick_map.records):
        _fail("mechanism_audit_tick_map_too_short")
    pts = tick_map.records[native_tick].pts
    video_payload = decode_pc_video_rgb24_at_pts(
        verified_artifacts[run.video.artifact],
        metadata=run.video,
        artifact=manifest.artifacts[run.video.artifact],
        requested_pts=(pts,),
    )[pts]
    if len(video_payload) != 800 * 600 * 3:
        _fail("mechanism_audit_video_frame_size_invalid")
    visual = source.visual_frames[native_tick]
    if visual.update != update:
        _fail("mechanism_audit_exact_step_visual_update_mismatch")
    bgra = decode_top_down_bgra_bmp(visual.path)
    source_rgb = np.ascontiguousarray(bgra[:, :, (2, 1, 0)])
    if canonical_rgb24_frame_sha256(source_rgb) != visual.pc_video_rgb24_sha256:
        _fail("mechanism_audit_exact_step_visual_hash_mismatch")
    video_rgb = np.frombuffer(video_payload, dtype=np.uint8).reshape(
        (600, 800, 3)
    )
    mismatch_count = int(np.any(source_rgb != video_rgb, axis=2).sum())
    if mismatch_count:
        _fail("mechanism_audit_exact_step_video_pixels_differ")
    source_payload = source_rgb.tobytes()
    return {
        "status": "PASS",
        "transport": "formal_exact_step_external_lossless",
        "run_id": run.run_id,
        "framework_update": update,
        "native_tick": native_tick,
        "video_pts": pts,
        "full_frame_mismatch_count": mismatch_count,
        "probe_rgb24_sha256": (
            f"sha256:{hashlib.sha256(source_payload).hexdigest()}"
        ),
        "video_rgb24_sha256": (
            f"sha256:{hashlib.sha256(video_payload).hexdigest()}"
        ),
        "canonical_rgb24_sha256": visual.pc_video_rgb24_sha256,
    }


def _positive_score_delta(
    events_by_update: Mapping[int, tuple[Mapping[str, Any], ...]],
    update: int,
) -> int | None:
    rows = [
        row
        for row in events_by_update.get(update, ())
        if row.get("kind") == "score_change"
    ]
    if len(rows) != 1:
        return None
    before = rows[0].get("score_before")
    after = rows[0].get("score_after")
    if (
        isinstance(before, bool)
        or not isinstance(before, int)
        or isinstance(after, bool)
        or not isinstance(after, int)
        or after <= before
    ):
        return None
    return after - before


def _curve_entities(
    frame: TrajectoryFrame,
    *,
    curve_index: int,
    container_offset: int,
) -> tuple[TrajectoryEntity, ...]:
    zone = f"curve:{curve_index}:list:{container_offset:03x}"
    rows = tuple(
        sorted(
            (
                entity
                for entity in frame.entities
                if entity.zone == zone
            ),
            key=lambda entity: entity.index,
        )
    )
    if tuple(entity.index for entity in rows) != tuple(range(len(rows))):
        return ()
    if len({entity.ball_id for entity in rows}) != len(rows):
        return ()
    return rows


def _original_curve(
    *,
    original_root: Path,
    level_id: str,
    hard: bool,
    curve_index: int,
) -> OriginalCurve:
    try:
        catalog = OriginalGameCatalog(original_root)
        loaded = catalog.load_level(
            level_id,
            hard=hard,
        )
    except (OSError, ValueError, OriginalDataError) as error:
        raise PcMechanismAuditError(
            "mechanism_audit_original_curve_unavailable"
        ) from error
    if not 0 <= curve_index < len(loaded.curves):
        _fail("mechanism_audit_original_curve_index_invalid")
    return loaded.curves[curve_index]


def _curve_fit(
    entities: Sequence[TrajectoryEntity],
    curve: OriginalCurve,
) -> tuple[float, float] | None:
    if not entities:
        return None
    squared_errors: list[float] = []
    maximum_error = 0.0
    for entity in entities:
        point = np.asarray(
            curve.point_at_waypoint(entity.curve_distance),
            dtype=np.float32,
        )
        if point.shape != (2,) or not bool(np.all(np.isfinite(point))):
            return None
        error = math.hypot(
            float(point[0]) - entity.position_x,
            float(point[1]) - entity.position_y,
        )
        if not math.isfinite(error):
            return None
        squared_errors.append(error * error)
        maximum_error = max(maximum_error, error)
    if maximum_error > CURVE_FIT_TOLERANCE_PX:
        return None
    return (
        maximum_error,
        math.sqrt(sum(squared_errors) / len(squared_errors)),
    )


def _positive_radius(entity: TrajectoryEntity) -> float | None:
    radius = entity.radius
    if radius is None or not math.isfinite(radius) or radius <= 0.0:
        return None
    return radius


def _curve_powerup_state(
    frame: TrajectoryFrame,
    *,
    curve_index: int,
) -> TrajectoryCurvePowerupState | None:
    state = frame.curve_powerup_state(curve_index)
    if (
        state is None
        or len(state.last_spawn_times) != POWERUP_TYPE_COUNT
        or len(state.cooldown_times) != POWERUP_TYPE_COUNT
        or len(state.spawn_counts) != POWERUP_TYPE_COUNT
        or len(state.field_124_by_type) != POWERUP_TYPE_COUNT
        or len(state.active_color_counts) != 6
    ):
        return None
    return state


def _mtrand_outputs_between(
    before: TrajectoryMTRandState | None,
    after: TrajectoryMTRandState | None,
    *,
    maximum_draws: int,
) -> tuple[int, ...] | None:
    if before is None or after is None or maximum_draws < 0:
        return None
    try:
        random = PopCapMTRandom(1)
        random.load_state(before.words, before.index)
    except (TypeError, ValueError):
        return None
    outputs: list[int] = []
    for draw_count in range(maximum_draws + 1):
        if random.index == after.index and random.words == after.words:
            return tuple(outputs)
        if draw_count == maximum_draws:
            break
        outputs.append(random.next_u31())
    return None


def _installed_fruit_scheduler_context(
    *,
    original_root: Path,
    level_id: str,
    hard: bool,
    curve_index: int,
) -> tuple[tuple[OriginalCurve, ...], FruitSpawnCalibration] | None:
    """Load only the installed metadata used by the retail fruit selector."""

    try:
        catalog = OriginalGameCatalog(original_root)
        loaded = catalog.load_level(
            level_id,
            hard=hard,
        )
        definition = loaded.definition
        frequency_text = definition.attributes.get("tfreq")
        frequency_ticks = int(frequency_text) if frequency_text is not None else 0
        points = tuple(
            (float(point[0]), float(point[1]))
            for point in definition.treasure_points
        )
        unlocks = tuple(
            tuple(int(value) for value in row)
            for row in definition.treasure_point_distances
        )
        assets = (
            catalog.fruit_assets(definition.fruit_type)
            if definition.fruit_type
            else None
        )
    except (IndexError, OriginalDataError, TypeError, ValueError):
        return None
    # This proof currently targets the one-curve Jungle2 Board.  Refuse to
    # project a partial multi-curve eligibility list into a transfer claim.
    if (
        curve_index != 0
        or len(loaded.curves) != 1
        or not points
        or len(points) != len(unlocks)
        or any(len(row) != len(loaded.curves) for row in unlocks)
        or frequency_ticks < 1
        or not definition.fruit_type
        or assets is None
    ):
        return None
    try:
        calibration = FruitSpawnCalibration(
            frequency_ticks=frequency_ticks,
            lifetime_ticks=FRUIT_LIFETIME_TICKS,
            points=points,
            unlock_percentages=unlocks,
            fruit_type=assets.fruit_type,
            logical_width=assets.logical_width,
            logical_height=assets.logical_height,
            sheet_columns=assets.sheet_columns,
            sheet_rows=assets.sheet_rows,
            collection_animation_frames=(
                assets.collection_animation_frames
            ),
            collection_animation_fps=assets.collection_animation_fps,
            provenance=(
                "installed levels.xml tfreq/TreasurePoint metadata and "
                f"{assets.provenance}; retail static control flow "
                "0x00417A00-0x00417FCF, 0x00418C90-0x00419166, "
                "0x0041C120-0x0041C527, 0x004573A0-0x0045746F, "
                "and 0x004B5A50-0x004B5CD9"
            ),
        )
    except ValueError:
        return None
    return loaded.curves, calibration


def _derive_fruit_scheduler_spawn_proofs(
    frames: Sequence[TrajectoryFrame],
    *,
    original_root: Path | None,
    level_id: str | None,
    hard: bool,
    curve_index: int,
) -> tuple[Mapping[str, Any], ...]:
    """Prove a natural retail fruit spawn and its two gameplay MT draws."""

    if original_root is None or level_id is None:
        return ()
    context = _installed_fruit_scheduler_context(
        original_root=original_root,
        level_id=level_id,
        hard=hard,
        curve_index=curve_index,
    )
    if context is None:
        return ()
    curves, calibration = context

    def same_f32(left: float, right: float) -> bool:
        return np.float32(left).tobytes() == np.float32(right).tobytes()

    proofs: list[Mapping[str, Any]] = []
    for before, after in zip(frames, frames[1:]):
        source_before = before.fruit_state
        source_after = after.fruit_state
        if (
            source_before is None
            or source_after is None
            or before.native_game_time is None
            or after.native_game_time != before.native_game_time + 1
            or source_before.active
            or not source_after.active
            or source_before.collecting
            or source_after.collecting
            or not 0 <= source_after.selected_point_index < len(calibration.points)
            or (
                before.score_target > 0
                and before.score >= before.score_target
            )
            or (
                after.native_game_time - source_before.expiry_time
                <= calibration.frequency_ticks
            )
            or before.qrand != after.qrand
            or before.thread_crt_rand_state
            != after.thread_crt_rand_state
        ):
            continue

        outputs = _mtrand_outputs_between(
            before.global_mtrand,
            after.global_mtrand,
            maximum_draws=64,
        )
        if (
            outputs is None
            or len(outputs) < 2
            or outputs[0] % calibration.frequency_ticks != 0
        ):
            continue

        progress_rows: list[dict[str, Any]] = []
        eligible: list[int] = []
        valid_progress = True
        for current_curve_index, curve in enumerate(curves):
            chain = _curve_entities(
                before,
                curve_index=current_curve_index,
                container_offset=ACTIVE_CHAIN_LIST_OFFSET,
            )
            point_count = int(curve.end_waypoint) + 1
            if not chain or point_count <= 0:
                progress = 0
                front_ball_id: int | None = None
                front_waypoint: float | None = None
            else:
                front = chain[-1]
                if not math.isfinite(front.curve_distance):
                    valid_progress = False
                    break
                front_ball_id = front.ball_id
                front_waypoint = float(np.float32(front.curve_distance))
                progress = math.trunc(
                    front_waypoint * 100.0 / point_count
                )
            progress_rows.append(
                {
                    "curve_index": current_curve_index,
                    "front_ball_id": front_ball_id,
                    "front_waypoint": front_waypoint,
                    "point_count": point_count,
                    "progress_percent": progress,
                }
            )
            for point_index, thresholds in enumerate(
                calibration.unlock_percentages
            ):
                threshold = thresholds[current_curve_index]
                if threshold > 0 and progress >= threshold:
                    eligible.append(point_index)
        if not valid_progress or not eligible:
            continue
        selected = eligible[outputs[1] % len(eligible)]
        maximum_float = float(np.finfo(np.float32).max)
        if (
            source_after.selected_point_index != selected
            or source_after.expiry_time
            != after.native_game_time + calibration.lifetime_ticks
            or not same_f32(source_after.velocity, 0.25)
            or not same_f32(source_after.max_velocity, 0.25)
            or not same_f32(source_after.acceleration, -0.01)
            or not same_f32(source_after.vertical_offset, 0.0)
            or not same_f32(source_after.lower_bound, maximum_float)
            or not same_f32(source_after.upper_bound, maximum_float)
            or source_after.glow_alpha != 0
            or source_after.glow_step != FRUIT_INITIAL_GLOW_STEP
            or source_after.alpha != 255
        ):
            continue
        proofs.append(
            {
                "feature": "fruit_scheduler_spawn",
                "status": "PASS",
                "framework_update": after.update,
                "native_game_time": after.native_game_time,
                "selected_point_index": selected,
                "eligible_point_indices": eligible,
                "curve_progress": progress_rows,
                "frequency_ticks": calibration.frequency_ticks,
                "lifetime_ticks": calibration.lifetime_ticks,
                "expiry_time": source_after.expiry_time,
                "mtrand_outputs": list(outputs),
                "mtrand_total_draw_count": len(outputs),
                "mtrand_scheduler_outputs": list(outputs[:2]),
                "mtrand_scheduler_draw_count": 2,
                "mtrand_trailing_draw_count": len(outputs) - 2,
                "mtrand_state_exact": True,
                "active_pointer_transition": "zero_to_nonzero",
                "fruit_reset_state_exact": True,
                "calibration_provenance": calibration.provenance,
                "proof": (
                    "the inactive retail Board advances the exact global MT "
                    "prefix through the tfreq chance and eligible-point "
                    "choice, then commits the selected TreasurePoint pointer "
                    "and complete 1000-tick fruit reset state"
                ),
            }
        )
    return tuple(proofs)


def _derive_fruit_expiry_proofs(
    frames: Sequence[TrajectoryFrame],
) -> tuple[Mapping[str, Any], ...]:
    """Prove the exact non-collection fruit lifetime boundary in retail."""

    proofs: list[Mapping[str, Any]] = []
    for before, after in zip(frames, frames[1:]):
        source_before = before.fruit_state
        source_after = after.fruit_state
        if (
            source_before is None
            or source_after is None
            or before.native_game_time is None
            or after.native_game_time != before.native_game_time + 1
            or not source_before.active
            or source_after.active
            or source_before.collecting
            or source_after.collecting
            or source_before.expiry_time != after.native_game_time
            or source_after.expiry_time != source_before.expiry_time
            or source_after.selected_point_index != 0
            or before.score != after.score
        ):
            continue
        proofs.append(
            {
                "feature": "fruit_expiry",
                "status": "PASS",
                "start_update": before.update,
                "framework_update": after.update,
                "native_game_time_before": before.native_game_time,
                "native_game_time_after": after.native_game_time,
                "expiry_time": source_before.expiry_time,
                "selected_point_index_before": (
                    source_before.selected_point_index
                ),
                "selected_point_index_after": (
                    source_after.selected_point_index
                ),
                "active_pointer_transition": "nonzero_to_zero",
                "collecting_before": False,
                "collecting_after": False,
                "score_delta": 0,
                "proof": (
                    "the uncollected retail TreasurePoint pointer clears "
                    "exactly when native game time reaches the retained "
                    "expiry timestamp, without a collection-score change"
                ),
            }
        )
    return tuple(proofs)


def _fruit_projectile_collision_candidate(
    before: TrajectoryFrame,
    after: TrajectoryFrame,
    *,
    calibration: FruitSpawnCalibration,
) -> Mapping[str, Any] | None:
    """Identify one disappearing fired bullet at the exact fruit overlap."""

    state = after.fruit_state
    if state is None or not 0 <= state.selected_point_index < len(
        calibration.points
    ):
        return None
    point = calibration.points[state.selected_point_index]
    center_x = np.float32(point[0]) + np.float32(
        calibration.logical_width // 2
    )
    center_y = (
        np.float32(point[1])
        + np.float32(calibration.logical_height // 2)
        + np.float32(state.vertical_offset)
    )
    after_identities = {
        entity.native_identity for entity in after.entities
    }
    candidates: list[dict[str, Any]] = []
    for projectile in before.entities:
        if (
            projectile.zone != "fired"
            or projectile.object_kind != "bullet"
            or projectile.native_object_address is None
            or projectile.native_identity in after_identities
            or projectile.radius is None
            or projectile.velocity_x is None
            or projectile.velocity_y is None
            or projectile.fired is None
            or not math.isfinite(projectile.radius)
            or projectile.radius <= 0.0
        ):
            continue
        # Retail clears Bullet+0x16A on its first half-step.  A bullet already
        # clear at the previous boundary advances once before this collision
        # check; a newly fired bullet is tested at its current position.
        if projectile.fired:
            collision_x = np.float32(projectile.position_x)
            collision_y = np.float32(projectile.position_y)
        else:
            collision_x = np.float32(projectile.position_x) + np.float32(
                projectile.velocity_x
            )
            collision_y = np.float32(projectile.position_y) + np.float32(
                projectile.velocity_y
            )
        delta_x = np.float32(collision_x - center_x)
        delta_y = np.float32(collision_y - center_y)
        distance_squared = np.float32(
            np.float32(delta_x * delta_x)
            + np.float32(delta_y * delta_y)
        )
        collision_radius = np.float32(projectile.radius) + np.float32(
            calibration.logical_height // 2
        )
        radius_squared = np.float32(
            collision_radius * collision_radius
        )
        if distance_squared > radius_squared:
            continue
        candidates.append(
            {
                "projectile_ball_id": projectile.ball_id,
                "projectile_color_id": projectile.color_id,
                "projectile_was_newly_fired": projectile.fired,
                "projectile_radius": projectile.radius,
                "collision_position": [
                    float(collision_x),
                    float(collision_y),
                ],
                "fruit_center": [float(center_x), float(center_y)],
                "distance_squared": float(distance_squared),
                "collision_radius_squared": float(radius_squared),
                "strict_overlap_or_tangent": True,
            }
        )
    return candidates[0] if len(candidates) == 1 else None


def _fruit_collection_animation_proof(
    frames: Sequence[TrajectoryFrame],
    *,
    collection_index: int,
    calibration: FruitSpawnCalibration,
) -> Mapping[str, Any] | None:
    """Verify every Board field until the installed PAM stops naturally."""

    collection_ticks = calibration.collection_ticks(100)
    clear_index = collection_index + collection_ticks
    if clear_index >= len(frames):
        return None
    first = frames[collection_index]
    first_state = first.fruit_state
    if (
        first_state is None
        or not first_state.active
        or not first_state.collecting
        or first.board_update_count is None
        or first.native_game_time is None
    ):
        return None
    pointer = first_state.active_point_pointer
    selected = first_state.selected_point_index
    cell_count = calibration.sheet_columns * calibration.sheet_rows

    def same_f32(left: float, right: float) -> bool:
        return np.float32(left).tobytes() == np.float32(right).tobytes()

    fixed_float_fields = (
        "velocity",
        "max_velocity",
        "acceleration",
        "vertical_offset",
        "lower_bound",
        "upper_bound",
    )
    for offset in range(1, collection_ticks + 1):
        previous = frames[collection_index + offset - 1]
        current = frames[collection_index + offset]
        previous_state = previous.fruit_state
        current_state = current.fruit_state
        if (
            previous_state is None
            or current_state is None
            or previous.board_update_count is None
            or current.board_update_count
            != previous.board_update_count + 1
            or previous.native_game_time is None
            or current.native_game_time
            != previous.native_game_time + 1
            or current.update != previous.update + 1
            or not previous_state.collecting
            or not current_state.collecting
            or any(
                not same_f32(
                    getattr(previous_state, field),
                    getattr(current_state, field),
                )
                for field in fixed_float_fields
            )
            or current_state.expiry_time != previous_state.expiry_time
        ):
            return None

        expected_alpha = max(0, previous_state.alpha - 8)
        expected_glow_alpha = (
            previous_state.glow_alpha + previous_state.glow_step
        )
        expected_glow_step = previous_state.glow_step
        if expected_glow_step > 0 and expected_glow_alpha >= 255:
            expected_glow_alpha = 255
            expected_glow_step = -expected_glow_step
        elif expected_glow_step < 0 and expected_glow_alpha <= 0:
            expected_glow_alpha = 0
            expected_glow_step = 0
        expected_cell = previous_state.cell_index
        if current.board_update_count % 3 == 0:
            expected_cell = (expected_cell + 1) % cell_count
        if (
            current_state.alpha != expected_alpha
            or current_state.glow_alpha != expected_glow_alpha
            or current_state.glow_step != expected_glow_step
            or current_state.cell_index != expected_cell
        ):
            return None

        final = offset == collection_ticks
        if final:
            if (
                current_state.active
                or current_state.active_point_pointer != 0
                or current_state.selected_point_index != 0
            ):
                return None
        elif (
            not current_state.active
            or current_state.active_point_pointer != pointer
            or current_state.selected_point_index != selected
        ):
            return None

    final = frames[clear_index]
    return {
        "feature": "fruit_collection_animation",
        "status": "PASS",
        "start_update": first.update,
        "framework_update": first.update,
        "clear_update": final.update,
        "start_native_game_time": first.native_game_time,
        "clear_native_game_time": final.native_game_time,
        "collection_ticks": collection_ticks,
        "tick_hz": 100,
        "selected_point_index": selected,
        "active_pointer_clear_offset": collection_ticks,
        "alpha_step": -8,
        "glow_recurrence_exact": True,
        "cell_recurrence_exact": True,
        "float32_state_retention_exact": True,
        "installed_animation_frames": (
            calibration.collection_animation_frames
        ),
        "installed_animation_fps": (
            calibration.collection_animation_fps
        ),
        "calibration_provenance": calibration.provenance,
        "proof": (
            "every native Board collection field follows the exact alpha, "
            "glow, cell, and retained-float recurrence until the installed "
            "PAM stop tick clears the TreasurePoint pointer"
        ),
    }


def _derive_fruit_collection_proofs(
    frames: Sequence[TrajectoryFrame],
    *,
    original_root: Path | None,
    level_id: str | None,
    hard: bool,
    curve_index: int,
    powerup_effect_proofs: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    """Derive natural projectile/bomb collection, score, and PAM proofs."""

    if original_root is None or level_id is None:
        return ()
    context = _installed_fruit_scheduler_context(
        original_root=original_root,
        level_id=level_id,
        hard=hard,
        curve_index=curve_index,
    )
    if context is None:
        return ()
    curves, calibration = context
    if not 0 <= curve_index < len(curves):
        return ()
    try:
        zuma_score = int(curves[curve_index].parameters.zuma_score)
    except (AttributeError, TypeError, ValueError):
        return ()
    bomb_proofs = {
        proof.get("framework_update"): proof
        for proof in powerup_effect_proofs
        if (
            proof.get("feature") == "powerup_proximity_bomb"
            and proof.get("status") == "PASS"
            and proof.get("powerup_type")
            == PROXIMITY_BOMB_POWERUP_TYPE
        )
    }
    proofs: dict[str, Mapping[str, Any]] = {}
    for collection_index, (before, after) in enumerate(
        zip(frames, frames[1:]),
        start=1,
    ):
        source_before = before.fruit_state
        source_after = after.fruit_state
        if (
            source_before is None
            or source_after is None
            or before.native_game_time is None
            or after.native_game_time != before.native_game_time + 1
            or before.board_update_count is None
            or after.board_update_count != before.board_update_count + 1
            or not source_before.active
            or source_before.collecting
            or not source_after.active
            or not source_after.collecting
            or source_after.active_point_pointer
            != source_before.active_point_pointer
            or source_after.selected_point_index
            != source_before.selected_point_index
            or source_after.expiry_time != source_before.expiry_time
            or source_after.glow_step != FRUIT_INITIAL_GLOW_STEP
            or not 0
            <= source_after.selected_point_index
            < len(calibration.points)
        ):
            continue
        score_at_level_start = before.score_target - zuma_score
        if not 0 <= score_at_level_start <= before.score:
            continue
        expected_points = max(
            500,
            ((before.score - score_at_level_start) // 600) * 100,
        )
        observed_score_delta = after.score - before.score
        projectile = _fruit_projectile_collision_candidate(
            before,
            after,
            calibration=calibration,
        )
        classified = projectile is not None
        if projectile is not None:
            proofs.setdefault(
                "fruit_projectile_collision",
                {
                    "feature": "fruit_projectile_collision",
                    "status": "PASS",
                    "start_update": before.update,
                    "framework_update": after.update,
                    "native_game_time": after.native_game_time,
                    "selected_point_index": (
                        source_after.selected_point_index
                    ),
                    "active_pointer_transition": (
                        "same_nonzero_to_collecting"
                    ),
                    "collision_cause": "free_projectile",
                    **projectile,
                    "calibration_provenance": calibration.provenance,
                    "proof": (
                        "one uniquely identified retail fired bullet reaches "
                        "the bobbed fruit centre/radius boundary, disappears "
                        "without entering a curve list, and atomically starts "
                        "the collection state"
                    ),
                },
            )
            before_powerups = before.curve_powerup_state(curve_index)
            after_powerups = after.curve_powerup_state(curve_index)
            newly_exploding = any(
                previous.exploding is False and current.exploding is True
                for identity, previous in before.entities_by_native_identity.items()
                if (
                    (current := after.entities_by_native_identity.get(identity))
                    is not None
                )
            )
            if (
                observed_score_delta == expected_points
                and before_powerups == after_powerups
                and not newly_exploding
            ):
                proofs.setdefault(
                    "fruit_collection_score",
                    {
                        "feature": "fruit_collection_score",
                        "status": "PASS",
                        "start_update": before.update,
                        "framework_update": after.update,
                        "score_before": before.score,
                        "score_after": after.score,
                        "score_delta": expected_points,
                        "score_at_level_start": score_at_level_start,
                        "zuma_score": zuma_score,
                        "tier_divisor": 600,
                        "tier_multiplier": 100,
                        "minimum_points": 500,
                        "isolated_projectile_collection": True,
                        "proof": (
                            "the isolated retail collection score equals "
                            "max(500, floor((score-level_start)/600)*100) "
                            "with no simultaneous explosion or manager change"
                        ),
                    },
                )

        bomb = bomb_proofs.get(after.update)
        if bomb is not None and projectile is None:
            trigger_ball_id = bomb.get("trigger_ball_id")
            trigger_ball = next(
                (
                    entity
                    for entity in _curve_entities(
                        after,
                        curve_index=curve_index,
                        container_offset=ACTIVE_CHAIN_LIST_OFFSET,
                    )
                    if entity.ball_id == trigger_ball_id
                ),
                None,
            )
            if trigger_ball is not None:
                point = calibration.points[
                    source_after.selected_point_index
                ]
                delta_x = np.float32(point[0]) - np.float32(
                    trigger_ball.position_x
                )
                delta_y = np.float32(point[1]) - np.float32(
                    trigger_ball.position_y
                )
                distance_squared = (
                    float(delta_x) * float(delta_x)
                    + float(delta_y) * float(delta_y)
                )
                radius = float(np.float32(
                    FRUIT_POWERUP_COLLISION_RADIUS
                ))
                if distance_squared < radius * radius:
                    classified = True
                    proofs.setdefault(
                        "fruit_powerup_collision",
                        {
                            "feature": "fruit_powerup_collision",
                            "status": "PASS",
                            "start_update": before.update,
                            "framework_update": after.update,
                            "native_game_time": after.native_game_time,
                            "selected_point_index": (
                                source_after.selected_point_index
                            ),
                            "active_pointer_transition": (
                                "same_nonzero_to_collecting"
                            ),
                            "collision_cause": "proximity_bomb",
                            "powerup_type": (
                                PROXIMITY_BOMB_POWERUP_TYPE
                            ),
                            "trigger_ball_id": trigger_ball.ball_id,
                            "trigger_ball_position": [
                                trigger_ball.position_x,
                                trigger_ball.position_y,
                            ],
                            "fruit_treasure_point": [
                                float(point[0]),
                                float(point[1]),
                            ],
                            "distance_squared": distance_squared,
                            "strict_radius_squared": radius * radius,
                            "collision_radius": radius,
                            "linked_powerup_manager_transition_exact": True,
                            "observed_total_score_delta": (
                                observed_score_delta
                            ),
                            "expected_fruit_points": expected_points,
                            "calibration_provenance": (
                                calibration.provenance
                            ),
                            "proof": (
                                "the exact retail type-0 trigger and native "
                                "TreasurePoint enter collection in one tick "
                                "inside the strict 108-pixel static radius"
                            ),
                        },
                    )

        if classified and "fruit_collection_animation" not in proofs:
            animation = _fruit_collection_animation_proof(
                frames,
                collection_index=collection_index,
                calibration=calibration,
            )
            if animation is not None:
                proofs["fruit_collection_animation"] = animation
    return tuple(proofs[name] for name in sorted(proofs))


def _powerup_weights(
    curve: OriginalCurve,
    calibration: PowerupSpawnCalibration,
) -> tuple[int, ...] | None:
    records = curve.parameters.powerup_records
    if len(records) > POWERUP_NONE_TYPE:
        return None
    supported = set(calibration.supported_types)
    weights = [0] * POWERUP_NONE_TYPE
    for powerup_type, record in enumerate(records):
        if len(record) != 2:
            return None
        weight = record[0]
        if (
            isinstance(weight, bool)
            or not isinstance(weight, int)
            or weight < 0
            or (weight > 0 and powerup_type not in supported)
        ):
            return None
        if powerup_type in supported:
            weights[powerup_type] = weight
    if sum(weights) <= 0:
        return None
    return tuple(weights)


def _single_changed_index(
    before: Sequence[int],
    after: Sequence[int],
) -> int | None:
    if len(before) != len(after):
        return None
    changed = [
        index
        for index, (left, right) in enumerate(
            zip(before, after, strict=True)
        )
        if left != right
    ]
    return changed[0] if len(changed) == 1 else None


def _derive_powerup_spawn_proofs(
    frames: Sequence[TrajectoryFrame],
    *,
    original_root: Path | None,
    level_id: str | None,
    hard: bool,
    curve_index: int,
) -> tuple[Mapping[str, Any], ...]:
    if (
        original_root is None
        or level_id != "Jungle2"
        or hard
    ):
        return ()
    has_spawn_transition = False
    for before, after in zip(frames, frames[1:]):
        before_state = _curve_powerup_state(
            before,
            curve_index=curve_index,
        )
        after_state = _curve_powerup_state(
            after,
            curve_index=curve_index,
        )
        if (
            before_state is not None
            and after_state is not None
            and before_state.spawn_counts != after_state.spawn_counts
        ):
            has_spawn_transition = True
            break
    if not has_spawn_transition:
        return ()
    curve = _original_curve(
        original_root=original_root,
        level_id=level_id,
        hard=hard,
        curve_index=curve_index,
    )
    calibration = PowerupSpawnCalibration.jungle2_retail_v1()
    weights = _powerup_weights(curve, calibration)
    active_num_colors = curve.parameters.colors
    if (
        weights is None
        or not 1 <= active_num_colors <= 6
    ):
        return ()
    total_weight = sum(weights)
    active_zone = (
        f"curve:{curve_index}:list:{ACTIVE_CHAIN_LIST_OFFSET:03x}"
    )
    proofs: list[Mapping[str, Any]] = []
    for before, after in zip(frames, frames[1:]):
        before_state = _curve_powerup_state(
            before,
            curve_index=curve_index,
        )
        after_state = _curve_powerup_state(
            after,
            curve_index=curve_index,
        )
        if (
            before_state is None
            or after_state is None
            or before.native_game_time is None
            or after.native_game_time != before.native_game_time + 1
            or before.qrand is None
            or after.qrand != before.qrand
            or before.thread_crt_rand_state is None
            or after.thread_crt_rand_state
            != before.thread_crt_rand_state
            or before.score != after.score
            or before.displayed_score != after.displayed_score
            or before.score_target != after.score_target
            or before.current_ball_id != after.current_ball_id
            or before.current_color_id != after.current_color_id
            or before.next_ball_id != after.next_ball_id
            or before.next_color_id != after.next_color_id
            or before.list_counts != after.list_counts
            or set(before.entities_by_id) != set(after.entities_by_id)
        ):
            continue
        selected_type = _single_changed_index(
            before_state.spawn_counts,
            after_state.spawn_counts,
        )
        if (
            selected_type is None
            or selected_type not in calibration.supported_types
            or after_state.spawn_counts[selected_type]
            != before_state.spawn_counts[selected_type] + 1
            or after_state.last_any_spawn_time
            != after.native_game_time
        ):
            continue

        expected_last_spawns = list(before_state.last_spawn_times)
        expected_last_spawns[selected_type] = after.native_game_time
        expected_spawn_counts = list(before_state.spawn_counts)
        expected_spawn_counts[selected_type] += 1
        if (
            after_state.last_spawn_times
            != tuple(expected_last_spawns)
            or after_state.spawn_counts
            != tuple(expected_spawn_counts)
            or after_state.cooldown_times
            != before_state.cooldown_times
            or after_state.field_124_by_type
            != before_state.field_124_by_type
            or after_state.reverse_speed != before_state.reverse_speed
            or after_state.slow_ticks != before_state.slow_ticks
            or after_state.reverse_ticks != before_state.reverse_ticks
            or after_state.last_powerup_waypoint
            != before_state.last_powerup_waypoint
            or after_state.powerup_triggered
            != before_state.powerup_triggered
        ):
            continue
        if (
            after.native_game_time < calibration.initial_delay_ticks
            or (
                calibration.spawn_delay_ticks > 0
                and after.native_game_time
                - before_state.last_any_spawn_time
                < calibration.spawn_delay_ticks
            )
            or (
                calibration.unique_color
                and all(
                    before_state.active_color_counts[color] > 0
                    for color in range(active_num_colors)
                )
            )
        ):
            continue

        outputs = _mtrand_outputs_between(
            before.global_mtrand,
            after.global_mtrand,
            maximum_draws=4,
        )
        if (
            outputs is None
            or len(outputs) != 4
            or outputs[0] % calibration.chance_denominator != 0
        ):
            continue
        weighted_roll = outputs[1] % total_weight
        cumulative = 0
        predicted_type: int | None = None
        for powerup_type, weight in enumerate(weights):
            cumulative += weight
            if weighted_roll < cumulative:
                predicted_type = powerup_type
                break
        if (
            predicted_type != selected_type
            or (
                after.native_game_time
                - before_state.last_spawn_times[selected_type]
                < calibration.cooldown_ticks
            )
            or (
                before_state.cooldown_times[selected_type] > 0
                and after.native_game_time
                - before_state.cooldown_times[selected_type]
                < calibration.cooldown_ticks
            )
        ):
            continue

        eligible_colors = [
            color
            for color in range(active_num_colors)
            if (
                not calibration.unique_color
                or before_state.active_color_counts[color] == 0
            )
        ]
        if not eligible_colors:
            continue
        selected_color = eligible_colors[
            outputs[2] % len(eligible_colors)
        ]
        expected_active_colors = list(
            before_state.active_color_counts
        )
        expected_active_colors[selected_color] += 1
        if (
            after_state.active_color_counts
            != tuple(expected_active_colors)
        ):
            continue

        before_chain = _curve_entities(
            before,
            curve_index=curve_index,
            container_offset=ACTIVE_CHAIN_LIST_OFFSET,
        )
        after_chain = _curve_entities(
            after,
            curve_index=curve_index,
            container_offset=ACTIVE_CHAIN_LIST_OFFSET,
        )
        if (
            not before_chain
            or tuple(entity.ball_id for entity in before_chain)
            != tuple(entity.ball_id for entity in after_chain)
        ):
            continue
        candidates = [
            entity
            for entity in before_chain
            if (
                entity.color_id == selected_color
                and entity.object_kind == "ball"
                and entity.exploding is False
                and entity.powerup_primary_type == POWERUP_NONE_TYPE
                and entity.powerup_secondary_type == POWERUP_NONE_TYPE
            )
        ]
        if not candidates:
            continue
        selected_ball = candidates[outputs[3] % len(candidates)]
        after_ball = after.entities_by_id.get(selected_ball.ball_id)
        if (
            after_ball is None
            or selected_ball.zone != active_zone
            or after_ball.zone != active_zone
            or after_ball.index != selected_ball.index
            or after_ball.color_id != selected_color
            or after_ball.object_kind != "ball"
            or after_ball.exploding is not False
            or after_ball.powerup_previous_type != POWERUP_NONE_TYPE
            or after_ball.powerup_primary_type != POWERUP_NONE_TYPE
            or after_ball.powerup_secondary_type != selected_type
            or after_ball.powerup_previous_ticks != 0
            or after_ball.powerup_lifetime_ticks != 0
            or after_ball.powerup_transition_ticks
            != POWERUP_TRANSITION_TICKS
            or after_ball.powerup_visual_index
            != POWERUP_VISUAL_INDEX_BY_TYPE[selected_type]
            or after_ball.powerup_visual_scale is None
            or not math.isclose(
                after_ball.powerup_visual_scale,
                POWERUP_VISUAL_SCALE,
                rel_tol=0.0,
                abs_tol=1e-7,
            )
            or after_ball.powerup_visual_step is None
            or not math.isclose(
                after_ball.powerup_visual_step,
                POWERUP_VISUAL_STEP,
                rel_tol=0.0,
                abs_tol=1e-7,
            )
        ):
            continue
        new_transitions = [
            right.ball_id
            for left, right in zip(
                before_chain,
                after_chain,
                strict=True,
            )
            if (
                left.powerup_secondary_type == POWERUP_NONE_TYPE
                and right.powerup_secondary_type
                != POWERUP_NONE_TYPE
                and right.powerup_transition_ticks
                == POWERUP_TRANSITION_TICKS
            )
        ]
        if new_transitions != [selected_ball.ball_id]:
            continue
        proofs.append(
            {
                "feature": "powerup_spawn",
                "status": "PASS",
                "framework_update": after.update,
                "native_game_time": after.native_game_time,
                "curve_index": curve_index,
                "powerup_type": selected_type,
                "selected_color_id": selected_color,
                "selected_ball_id": selected_ball.ball_id,
                "selected_ball_active_index": selected_ball.index,
                "candidate_count": len(candidates),
                "eligible_colors": eligible_colors,
                "chance_denominator": calibration.chance_denominator,
                "powerup_weights": list(weights),
                "mtrand_outputs": list(outputs),
                "mtrand_draw_count": len(outputs),
                "mtrand_state_exact": True,
                "manager_transition_exact": True,
                "ball_transition_exact": True,
                "calibration_provenance": calibration.provenance,
                "proof": (
                    "the exact retail MT state advances four draws through "
                    "chance, weighted type, eligible colour, and active-list "
                    "candidate selection, landing on the sole observed "
                    "manager and ball transition"
                ),
            }
        )
    return tuple(proofs)


def _direct_match_run(
    chain: Sequence[TrajectoryEntity],
    *,
    trigger_ball_id: int,
) -> tuple[int, ...]:
    identifiers = [entity.ball_id for entity in chain]
    if trigger_ball_id not in identifiers:
        return ()
    seed = identifiers.index(trigger_ball_id)
    color = chain[seed].color_id
    left = seed
    while (
        left > 0
        and chain[left - 1].contact_next is True
        and chain[left - 1].color_id == color
    ):
        left -= 1
    right = seed
    while (
        right + 1 < len(chain)
        and chain[right].contact_next is True
        and chain[right + 1].color_id == color
    ):
        right += 1
    return tuple(
        chain[index].ball_id for index in range(left, right + 1)
    )


def _reverse_movement_reference(
    trigger: TrajectoryFrame,
    after: TrajectoryFrame,
    post_after: TrajectoryFrame,
    *,
    curve_index: int,
) -> int | None:
    chains = [
        {
            entity.ball_id: entity
            for entity in _curve_entities(
                frame,
                curve_index=curve_index,
                container_offset=ACTIVE_CHAIN_LIST_OFFSET,
            )
        }
        for frame in (trigger, after, post_after)
    ]
    candidates: list[int] = []
    for ball_id in sorted(set(chains[0]) & set(chains[1]) & set(chains[2])):
        first, second, third = (
            chain[ball_id] for chain in chains
        )
        if (
            first.exploding is False
            and second.exploding is False
            and third.exploding is False
            and second.backwards_count == 0
            and second.backwards_speed is not None
            and math.isclose(
                second.backwards_speed,
                POWERUP_REVERSE_SPEED,
                rel_tol=0.0,
                abs_tol=1e-7,
            )
            and math.isclose(
                second.curve_distance,
                first.curve_distance - POWERUP_REVERSE_SPEED,
                rel_tol=0.0,
                abs_tol=1e-5,
            )
            and math.isclose(
                third.curve_distance,
                second.curve_distance - POWERUP_REVERSE_SPEED,
                rel_tol=0.0,
                abs_tol=1e-5,
            )
        ):
            candidates.append(ball_id)
    return candidates[0] if candidates else None


def _slow_movement_proof(
    pre_before: TrajectoryFrame,
    before: TrajectoryFrame,
    trigger: TrajectoryFrame,
    after: TrajectoryFrame,
    *,
    curve_index: int,
) -> Mapping[str, Any] | None:
    chains = [
        {
            entity.ball_id: entity
            for entity in _curve_entities(
                frame,
                curve_index=curve_index,
                container_offset=ACTIVE_CHAIN_LIST_OFFSET,
            )
        }
        for frame in (pre_before, before, trigger, after)
    ]
    for ball_id in sorted(
        set(chains[0]) & set(chains[1]) & set(chains[2]) & set(chains[3])
    ):
        entities = [chain[ball_id] for chain in chains]
        if any(entity.exploding is not False for entity in entities):
            continue
        if any(
            entity.suck_count != 0 or entity.backwards_count != 0
            for entity in entities
        ):
            continue
        movement_before = (
            entities[1].curve_distance - entities[0].curve_distance
        )
        movement_at_trigger = (
            entities[2].curve_distance - entities[1].curve_distance
        )
        movement_after = (
            entities[3].curve_distance - entities[2].curve_distance
        )
        if (
            movement_before > 1e-6
            and movement_at_trigger >= 0.0
            and movement_after >= 0.0
            and movement_at_trigger < movement_before
            and movement_after < movement_before
        ):
            return {
                "reference_ball_id": ball_id,
                "movement_before_trigger": movement_before,
                "movement_at_trigger": movement_at_trigger,
                "movement_after_trigger": movement_after,
                "strictly_slower": True,
            }
    return None


def _simultaneous_fruit_score_binding(
    before: TrajectoryFrame,
    after: TrajectoryFrame,
    *,
    original_root: Path | None,
    level_id: str | None,
    hard: bool,
    curve_index: int,
) -> Mapping[str, Any] | None:
    """Bind an exact concurrent fruit score without deciding collision cause."""

    if original_root is None or level_id is None:
        return None
    source_before = before.fruit_state
    source_after = after.fruit_state
    if (
        source_before is None
        or source_after is None
        or before.native_game_time is None
        or after.native_game_time != before.native_game_time + 1
        or before.board_update_count is None
        or after.board_update_count != before.board_update_count + 1
        or not source_before.active
        or source_before.collecting
        or not source_after.active
        or not source_after.collecting
        or source_after.active_point_pointer
        != source_before.active_point_pointer
        or source_after.selected_point_index
        != source_before.selected_point_index
        or source_after.expiry_time != source_before.expiry_time
        or source_after.glow_step != FRUIT_INITIAL_GLOW_STEP
    ):
        return None
    context = _installed_fruit_scheduler_context(
        original_root=original_root,
        level_id=level_id,
        hard=hard,
        curve_index=curve_index,
    )
    if context is None:
        return None
    curves, calibration = context
    if (
        not 0 <= curve_index < len(curves)
        or not 0
        <= source_after.selected_point_index
        < len(calibration.points)
    ):
        return None
    try:
        zuma_score = int(curves[curve_index].parameters.zuma_score)
    except (AttributeError, TypeError, ValueError):
        return None
    score_at_level_start = before.score_target - zuma_score
    if not 0 <= score_at_level_start <= before.score:
        return None
    expected_points = max(
        500,
        ((before.score - score_at_level_start) // 600) * 100,
    )
    point = calibration.points[source_after.selected_point_index]
    return {
        "status": "PASS",
        "classification": "independently_bound_concurrent_fruit_score",
        "framework_update": after.update,
        "native_game_time": after.native_game_time,
        "selected_point_index": source_after.selected_point_index,
        "fruit_treasure_point": [float(point[0]), float(point[1])],
        "score_at_level_start": score_at_level_start,
        "score_before": before.score,
        "score_target": before.score_target,
        "zuma_score": zuma_score,
        "tier_divisor": 600,
        "tier_multiplier": 100,
        "minimum_points": 500,
        "expected_score_delta": expected_points,
        "collision_cause_authorized": False,
        "calibration_provenance": calibration.provenance,
        "proof": (
            "the native TreasurePoint enters collection on this exact tick "
            "and the installed level formula independently fixes its score "
            "component; collision cause remains unclassified"
        ),
    }


def _derive_powerup_effect_proofs(
    frames: Sequence[TrajectoryFrame],
    *,
    original_root: Path | None,
    level_id: str | None,
    hard: bool,
    curve_index: int,
) -> tuple[Mapping[str, Any], ...]:
    proofs: dict[str, Mapping[str, Any]] = {}
    for trigger_index in range(2, len(frames) - 2):
        before = frames[trigger_index - 1]
        trigger = frames[trigger_index]
        after = frames[trigger_index + 1]
        before_state = _curve_powerup_state(
            before,
            curve_index=curve_index,
        )
        trigger_state = _curve_powerup_state(
            trigger,
            curve_index=curve_index,
        )
        if before_state is None or trigger_state is None:
            continue
        powerup_type = _single_changed_index(
            before_state.field_124_by_type,
            trigger_state.field_124_by_type,
        )
        if (
            powerup_type not in POWERUP_FEATURE_BY_TYPE
            or trigger_state.field_124_by_type[powerup_type]
            != before_state.field_124_by_type[powerup_type] + 1
        ):
            continue
        before_chain = {
            entity.ball_id: entity
            for entity in _curve_entities(
                before,
                curve_index=curve_index,
                container_offset=ACTIVE_CHAIN_LIST_OFFSET,
            )
        }
        trigger_chain_rows = _curve_entities(
            trigger,
            curve_index=curve_index,
            container_offset=ACTIVE_CHAIN_LIST_OFFSET,
        )
        trigger_chain = {
            entity.ball_id: entity for entity in trigger_chain_rows
        }
        if not before_chain or not trigger_chain:
            continue
        target_ids = [
            ball_id
            for ball_id in sorted(set(before_chain) & set(trigger_chain))
            if (
                before_chain[ball_id].powerup_primary_type
                == powerup_type
                and before_chain[ball_id].exploding is False
                and trigger_chain[ball_id].exploding is True
            )
        ]
        if len(target_ids) != 1:
            continue
        trigger_ball_id = target_ids[0]
        trigger_color_id = trigger_chain[trigger_ball_id].color_id
        newly_exploding = tuple(
            sorted(
                ball_id
                for ball_id, entity in trigger_chain.items()
                if (
                    entity.exploding is True
                    and (
                        ball_id not in before_chain
                        or before_chain[ball_id].exploding is not True
                    )
                )
            )
        )
        if not newly_exploding or trigger_ball_id not in newly_exploding:
            continue
        direct_match_ids: tuple[int, ...] | None = None
        if powerup_type == PROXIMITY_BOMB_POWERUP_TYPE:
            direct_match_ids = _direct_match_run(
                trigger_chain_rows,
                trigger_ball_id=trigger_ball_id,
            )
            if len(direct_match_ids) < 3:
                continue

        movement_reference: int | None = None
        if powerup_type == REVERSE_POWERUP_TYPE:
            movement_reference = _reverse_movement_reference(
                trigger,
                after,
                frames[trigger_index + 2],
                curve_index=curve_index,
            )
            if movement_reference is None:
                continue
        slow_movement: Mapping[str, Any] | None = None
        if powerup_type == SLOW_POWERUP_TYPE:
            slow_movement = _slow_movement_proof(
                frames[trigger_index - 2],
                before,
                trigger,
                after,
                curve_index=curve_index,
            )
            if slow_movement is None:
                continue

        window = tuple(
            frames[trigger_index - 2 : trigger_index + 3]
        )
        concurrent_score_binding = _simultaneous_fruit_score_binding(
            before,
            trigger,
            original_root=original_root,
            level_id=level_id,
            hard=hard,
            curve_index=curve_index,
        )
        concurrent_score_delta = (
            0
            if concurrent_score_binding is None
            else int(concurrent_score_binding["expected_score_delta"])
        )
        try:
            verification = verify_powerup_trigger_frames(
                window,
                curve_index=curve_index,
                trigger_ball_id=trigger_ball_id,
                trigger_color_id=trigger_color_id,
                powerup_type=powerup_type,
                expected_newly_exploding_ids=newly_exploding,
                expected_score_delta=trigger.score - before.score,
                expected_concurrent_score_delta=concurrent_score_delta,
                expected_direct_match_ids=direct_match_ids,
                movement_reference_ball_id=movement_reference,
                expected_trigger_update=trigger.update,
                reverse_ticks=POWERUP_REVERSE_TICKS,
                reverse_speed=POWERUP_REVERSE_SPEED,
                slow_ticks=POWERUP_SLOW_TICKS,
                bomb_collision_pad=POWERUP_BOMB_COLLISION_PAD,
                require_explosion_removal=False,
            )
        except (PcPowerupTriggerError, ValueError):
            continue
        feature = POWERUP_FEATURE_BY_TYPE[powerup_type]
        proofs.setdefault(
            feature,
            {
                "feature": feature,
                "status": "PASS",
                "framework_update": trigger.update,
                "native_game_time": trigger.native_game_time,
                "curve_index": curve_index,
                "powerup_type": powerup_type,
                "trigger_ball_id": trigger_ball_id,
                "trigger_color_id": trigger_color_id,
                "newly_exploding_ball_ids": list(newly_exploding),
                "score_delta": trigger.score - before.score,
                "base_explosion_score_delta": 10 * len(newly_exploding),
                "concurrent_score_binding": concurrent_score_binding,
                "manager_transition_exact": True,
                "effect_verification": verification,
                "slow_physical_movement": slow_movement,
                "proof": (
                    "the native trigger counter, cooldown, active-colour "
                    "bookkeeping, explosion set, score, exact effect timer, "
                    "and physical effect all agree in one exact transition"
                ),
            },
        )
    return tuple(proofs[name] for name in sorted(proofs))


def _shooter_present_colors(
    frame: TrajectoryFrame,
) -> tuple[int, ...] | None:
    colors: list[int] = []
    for entity in frame.entities:
        include = False
        if entity.zone.endswith(
            f"list:{ACTIVE_CHAIN_LIST_OFFSET:03x}"
        ):
            if entity.object_kind != "ball" or entity.exploding is None:
                return None
            include = entity.exploding is False
        elif entity.zone == "fired" or entity.zone.endswith(
            f"list:{INSERTION_STAGING_LIST_OFFSET:03x}"
        ):
            if entity.object_kind != "bullet":
                return None
            include = True
        if include:
            if not 0 <= entity.color_id < 6:
                return None
            colors.append(entity.color_id)
    return tuple(colors) if colors else None


def _qrand_matches(
    chooser: _BalancedColorChooser,
    frame: TrajectoryFrame,
) -> bool:
    state = frame.qrand
    return bool(
        state is not None
        and chooser.update_count == state.update_count
        and chooser.selected_index == state.selected_index
        and tuple(float(value) for value in chooser.weights)
        == state.weights
        and tuple(float(value) for value in chooser.sways)
        == state.sways
        and tuple(int(value) for value in chooser.last_hit)
        == state.last_hit
        and tuple(int(value) for value in chooser.previous_hit)
        == state.previous_hit
    )


def _derive_shooter_rng_proofs(
    frames: Sequence[TrajectoryFrame],
    events_by_update: Mapping[int, tuple[Mapping[str, Any], ...]],
) -> tuple[Mapping[str, Any], ...]:
    proofs: list[Mapping[str, Any]] = []
    for before, after in zip(frames, frames[1:]):
        if (
            before.qrand is None
            or after.qrand is None
            or before.thread_crt_rand_state is None
            or after.thread_crt_rand_state is None
            or after.qrand.update_count
            != before.qrand.update_count + 1
        ):
            continue
        identifiers = (
            before.current_ball_id,
            before.next_ball_id,
            after.current_ball_id,
            after.next_ball_id,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in identifiers
        ):
            continue
        before_current_id = int(before.current_ball_id)
        before_next_id = int(before.next_ball_id)
        after_next_id = int(after.next_ball_id)
        if (
            before_current_id == before_next_id
            or after.current_ball_id != before_next_id
            or after.current_color_id != before.next_color_id
            or after_next_id in before.entities_by_id
        ):
            continue
        before_current = before.entities_by_id.get(before_current_id)
        before_next = before.entities_by_id.get(before_next_id)
        fired = after.entities_by_id.get(before_current_id)
        current = after.entities_by_id.get(before_next_id)
        next_ball = after.entities_by_id.get(after_next_id)
        entities = (
            before_current,
            before_next,
            fired,
            current,
            next_ball,
        )
        if any(entity is None for entity in entities):
            continue
        if (
            before_current.zone != "shooter_current"
            or before_next.zone != "shooter_next"
            or fired.zone != "fired"
            or current.zone != "shooter_current"
            or next_ball.zone != "shooter_next"
            or any(
                entity.object_kind != "bullet"
                for entity in entities
                if entity is not None
            )
            or not isinstance(before_current.fired, bool)
            or before_next.fired is not False
            or fired.fired is not False
            or current.fired is not False
            or next_ball.fired is not False
            or fired.color_id != before.current_color_id
            or current.color_id != before.next_color_id
        ):
            continue

        rows = events_by_update.get(after.update, ())
        matching_spawns = [
            row
            for row in rows
            if row.get("kind") == "projectile_spawn"
            and isinstance(row.get("entity"), Mapping)
            and row["entity"].get("ball_id") == before_current_id
        ]
        matching_transfers = [
            row
            for row in rows
            if row.get("kind") == "entity_zone_change"
            and row.get("ball_id") == before_current_id
            and row.get("from_zone") == "shooter_current"
            and row.get("to_zone") == "fired"
        ]
        matching_chambers = [
            row
            for row in rows
            if row.get("kind") == "shooter_chamber_change"
            and isinstance(row.get("before"), Mapping)
            and isinstance(row.get("after"), Mapping)
            and row["before"].get("current_ball_id")
            == before_current_id
            and row["before"].get("next_ball_id") == before_next_id
            and row["after"].get("current_ball_id") == before_next_id
            and row["after"].get("next_ball_id") == after_next_id
        ]
        if (
            len(matching_spawns) != 1
            or len(matching_transfers) != 1
            or len(matching_chambers) != 1
        ):
            continue

        present_colors = _shooter_present_colors(after)
        if present_colors is None:
            continue
        try:
            crt = MsvcCRTRandom(1)
            crt.state = before.thread_crt_rand_state
            chooser = _BalancedColorChooser(crt, size=6)
            chooser.load_state(
                update_count=before.qrand.update_count,
                selected_index=before.qrand.selected_index,
                weights=before.qrand.weights,
                sways=before.qrand.sways,
                last_hit=before.qrand.last_hit,
                previous_hit=before.qrand.previous_hit,
            )
            selected_color = chooser.choose(present_colors)
        except (TypeError, ValueError):
            continue
        if (
            selected_color is None
            or selected_color != after.next_color_id
            or next_ball.color_id != selected_color
            or crt.state != after.thread_crt_rand_state
            or not _qrand_matches(chooser, after)
        ):
            continue
        proofs.append(
            {
                "feature": "rng_pending",
                "status": "PASS",
                "framework_update": after.update,
                "fired_ball_id": before_current_id,
                "promoted_ball_id": before_next_id,
                "new_next_ball_id": after_next_id,
                "selected_color_id": selected_color,
                "present_color_support": sorted(set(present_colors)),
                "present_color_observation_count": len(present_colors),
                "qrand_update_count_before": before.qrand.update_count,
                "qrand_update_count_after": after.qrand.update_count,
                "qrand_selected_index": after.qrand.selected_index,
                "thread_crt_state_before": before.thread_crt_rand_state,
                "thread_crt_state_after": after.thread_crt_rand_state,
                "qrand_state_exact": True,
                "thread_crt_state_exact": True,
                "proof": (
                    "the exact pre-shot QRand vectors and thread-local CRT "
                    "state replay one colour draw over the native post-shot "
                    "colour support, producing the observed replacement "
                    "chamber identity, colour, and complete RNG state"
                ),
            }
        )
    return tuple(proofs)


def _entity_topology_signature(
    frame: TrajectoryFrame,
) -> tuple[tuple[int, int, str, str, int], ...] | None:
    if len(frame.entities_by_id) != len(frame.entities):
        return None
    return tuple(
        sorted(
            (
                entity.ball_id,
                entity.color_id,
                entity.object_kind,
                entity.zone,
                entity.index,
            )
            for entity in frame.entities
        )
    )


def _powerup_manager_rng_signature(
    frame: TrajectoryFrame,
) -> tuple[tuple[Any, ...], ...] | None:
    rows: list[tuple[Any, ...]] = []
    seen: set[int] = set()
    for state in frame.curve_powerups:
        if (
            state.curve_index in seen
            or _curve_powerup_state(
                frame,
                curve_index=state.curve_index,
            )
            is None
        ):
            return None
        seen.add(state.curve_index)
        rows.append(
            (
                state.curve_index,
                state.last_any_spawn_time,
                state.last_spawn_times,
                state.cooldown_times,
                state.spawn_counts,
                state.field_124_by_type,
                state.active_color_counts,
                state.reverse_speed,
                state.last_powerup_waypoint,
                state.powerup_triggered,
            )
        )
    return tuple(sorted(rows))


def _quiet_global_rng_outputs(
    before: TrajectoryFrame,
    after: TrajectoryFrame,
) -> tuple[int, ...] | None:
    outputs = _mtrand_outputs_between(
        before.global_mtrand,
        after.global_mtrand,
        maximum_draws=2,
    )
    # An unchanged MT state proves that this neighbouring transition consumed
    # no global draws even when ordinary gameplay topology changed (retail
    # commonly moves the previous pending ball into the chain immediately
    # before refilling the pending list).  Non-zero draws need a fully quiet
    # transition before they can be classified as the ambient envelope.
    if outputs == ():
        return outputs
    if (
        outputs is None
        or before.score != after.score
        or before.displayed_score != after.displayed_score
        or before.score_target != after.score_target
        or before.current_ball_id != after.current_ball_id
        or before.current_color_id != after.current_color_id
        or before.next_ball_id != after.next_ball_id
        or before.next_color_id != after.next_color_id
        or before.list_counts != after.list_counts
        or before.qrand != after.qrand
        or before.thread_crt_rand_state
        != after.thread_crt_rand_state
        or _entity_topology_signature(before)
        != _entity_topology_signature(after)
        or _powerup_manager_rng_signature(before)
        != _powerup_manager_rng_signature(after)
    ):
        return None
    return outputs


def _pending_run_length(colors: Sequence[int], color: int) -> int:
    result = 0
    for candidate in colors:
        if candidate != color:
            break
        result += 1
    return result


def _pending_single_count(
    colors: Sequence[int],
    group_limit: int,
) -> int:
    group_count = 0
    previous = -1
    singles = 0
    run_length = 0
    for color in colors:
        if group_count > group_limit:
            break
        if color != previous:
            if run_length == 1:
                singles += 1
            run_length = 1
            group_count += 1
            previous = color
        else:
            run_length += 1
    return singles


def _prechance_powerup_ineligibility(
    frame: TrajectoryFrame,
    *,
    curve_index: int,
    calibration: PowerupSpawnCalibration,
    active_num_colors: int,
) -> str | None:
    state = _curve_powerup_state(
        frame,
        curve_index=curve_index,
    )
    native_time = frame.native_game_time
    if state is None or native_time is None:
        return None
    if native_time < calibration.initial_delay_ticks:
        return "before_initial_delay"
    if (
        calibration.spawn_delay_ticks > 0
        and native_time - state.last_any_spawn_time
        < calibration.spawn_delay_ticks
    ):
        return "within_spawn_delay"
    if (
        calibration.unique_color
        and all(
            state.active_color_counts[color] > 0
            for color in range(active_num_colors)
        )
    ):
        return "all_colours_already_powered"
    return None


def _derive_pending_rng_rejection_proofs(
    frames: Sequence[TrajectoryFrame],
    *,
    original_root: Path | None,
    level_id: str | None,
    hard: bool,
    curve_index: int,
) -> tuple[Mapping[str, Any], ...]:
    if (
        original_root is None
        or level_id != "Jungle2"
        or hard
        or len(frames) < 4
    ):
        return ()
    pending_offset = 0x68
    candidate_indices = [
        index
        for index in range(2, len(frames) - 1)
        if (
            frames[index - 1].list_count(
                curve_index,
                pending_offset,
            )
            == 0
            and frames[index].list_count(
                curve_index,
                pending_offset,
            )
            == 1
        )
    ]
    if not candidate_indices:
        return ()
    curve = _original_curve(
        original_root=original_root,
        level_id=level_id,
        hard=hard,
        curve_index=curve_index,
    )
    parameters = curve.parameters
    active_num_colors = parameters.colors
    if not 2 <= active_num_colors <= 6:
        return ()
    calibration = PowerupSpawnCalibration.jungle2_retail_v1()
    proofs: list[Mapping[str, Any]] = []
    for index in candidate_indices:
        pre_before = frames[index - 2]
        before = frames[index - 1]
        after = frames[index]
        post_after = frames[index + 1]
        ambient_before = _quiet_global_rng_outputs(pre_before, before)
        ambient_after = _quiet_global_rng_outputs(after, post_after)
        if (
            ambient_before is None
            or ambient_after is None
            or len(ambient_before) != len(ambient_after)
            or len(ambient_before) not in (0, 2)
            or before.native_game_time is None
            or after.native_game_time != before.native_game_time + 1
            or before.score != after.score
            or before.displayed_score != after.displayed_score
            or before.score_target != after.score_target
            or before.current_ball_id != after.current_ball_id
            or before.current_color_id != after.current_color_id
            or before.next_ball_id != after.next_ball_id
            or before.next_color_id != after.next_color_id
            or before.qrand is None
            or after.qrand != before.qrand
            or before.thread_crt_rand_state is None
            or after.thread_crt_rand_state
            != before.thread_crt_rand_state
            or _powerup_manager_rng_signature(before)
            != _powerup_manager_rng_signature(after)
        ):
            continue
        ineligibility = _prechance_powerup_ineligibility(
            after,
            curve_index=curve_index,
            calibration=calibration,
            active_num_colors=active_num_colors,
        )
        if ineligibility is None:
            continue

        before_active = _curve_entities(
            before,
            curve_index=curve_index,
            container_offset=ACTIVE_CHAIN_LIST_OFFSET,
        )
        after_active = _curve_entities(
            after,
            curve_index=curve_index,
            container_offset=ACTIVE_CHAIN_LIST_OFFSET,
        )
        before_pending = _curve_entities(
            before,
            curve_index=curve_index,
            container_offset=pending_offset,
        )
        after_pending = _curve_entities(
            after,
            curve_index=curve_index,
            container_offset=pending_offset,
        )
        if (
            not before_active
            or before_pending
            or len(after_pending) != 1
            or tuple(entity.ball_id for entity in before_active)
            != tuple(entity.ball_id for entity in after_active)
            or tuple(entity.color_id for entity in before_active)
            != tuple(entity.color_id for entity in after_active)
        ):
            continue
        new_ball = after_pending[0]
        before_topology = _entity_topology_signature(before)
        after_without_new = tuple(
            row
            for row in (
                _entity_topology_signature(after) or ()
            )
            if row[0] != new_ball.ball_id
        )
        if (
            before_topology is None
            or new_ball.ball_id in before.entities_by_id
            or after_without_new != before_topology
        ):
            continue
        list_keys = {
            (candidate_curve, offset)
            for candidate_curve, offset, _ in (
                *before.list_counts,
                *after.list_counts,
            )
        }
        if any(
            (
                before.list_count(candidate_curve, offset),
                after.list_count(candidate_curve, offset),
            )
            != (
                (0, 1)
                if (
                    candidate_curve == curve_index
                    and offset == pending_offset
                )
                else (
                    before.list_count(candidate_curve, offset),
                    before.list_count(candidate_curve, offset),
                )
            )
            for candidate_curve, offset in list_keys
        ):
            continue

        outputs = _mtrand_outputs_between(
            before.global_mtrand,
            after.global_mtrand,
            maximum_draws=32,
        )
        neighbouring_ambient_draw_count = len(ambient_before)
        if neighbouring_ambient_draw_count == 0:
            if outputs is None or len(outputs) < 4:
                continue
            repeat_output = outputs[0]
            candidate_outputs = outputs[1:-1]
            ball_visual_frame_output = outputs[-1]
            ambient_before_output = None
            ambient_after_output = None
            rng_sequence_layout = (
                "repeat,candidate_rejection_loop,ball_visual_frame"
            )
        else:
            if outputs is None or len(outputs) < 6:
                continue
            repeat_output = outputs[1]
            candidate_outputs = outputs[2:-2]
            ball_visual_frame_output = outputs[-2]
            ambient_before_output = outputs[0]
            ambient_after_output = outputs[-1]
            rng_sequence_layout = (
                "ambient_before,repeat,candidate_rejection_loop,"
                "ball_visual_frame,ambient_after"
            )
        if len(candidate_outputs) < 2:
            continue
        active_colors = tuple(
            entity.color_id for entity in before_active
        )
        previous_color = active_colors[0]
        if not 0 <= previous_color < active_num_colors:
            continue
        current_run = _pending_run_length(
            active_colors,
            previous_color,
        )
        repeat_roll = repeat_output % 100
        repeat_chance = parameters.ball_repeat_chance
        max_clump = parameters.max_clump_size
        max_single = parameters.max_single
        forced_single_repeat = bool(
            max_single < 10
            and _pending_single_count(active_colors, 1) == 1
            and (
                max_single == 0
                or _pending_single_count(active_colors, 10)
                > max_single
            )
        )
        candidate_colors = tuple(
            output % active_num_colors
            for output in candidate_outputs
        )
        if (
            (repeat_roll <= repeat_chance and current_run < max_clump)
            or forced_single_repeat
            or any(
                color != previous_color
                for color in candidate_colors[:-1]
            )
            or candidate_colors[-1] == previous_color
            or new_ball.color_id != candidate_colors[-1]
        ):
            continue
        proofs.append(
            {
                "feature": "rng_rejection",
                "status": "PASS",
                "from_update": before.update,
                "framework_update": after.update,
                "curve_index": curve_index,
                "previous_color_id": previous_color,
                "generated_color_id": new_ball.color_id,
                "generated_ball_id": new_ball.ball_id,
                "repeat_roll_output": repeat_output,
                "repeat_roll_mod_100": repeat_roll,
                "repeat_chance": repeat_chance,
                "current_run_length": current_run,
                "max_clump_size": max_clump,
                "forced_single_repeat": False,
                "rejected_candidate_outputs": list(
                    candidate_outputs[:-1]
                ),
                "rejected_candidate_colors": list(
                    candidate_colors[:-1]
                ),
                "accepted_candidate_output": candidate_outputs[-1],
                "accepted_candidate_color": candidate_colors[-1],
                "rejection_count": len(candidate_outputs) - 1,
                "ball_visual_frame_output": ball_visual_frame_output,
                "ambient_before_output": ambient_before_output,
                "ambient_after_output": ambient_after_output,
                "mtrand_outputs": list(outputs),
                "mtrand_draw_count": len(outputs),
                "mtrand_state_exact": True,
                "neighbouring_ambient_draw_count": (
                    neighbouring_ambient_draw_count
                ),
                "rng_sequence_layout": rng_sequence_layout,
                "powerup_rng_prechance_ineligible": ineligibility,
                "proof": (
                    "neighbouring exact MT states establish an equal zero- "
                    "or independently quiet two-draw ambient envelope; "
                    "power-up RNG is blocked before its chance draw, and "
                    "the remaining exact MT sequence rejects one or more "
                    "copies of the rear colour before creating the sole "
                    "observed pending ball"
                ),
            }
        )
    return tuple(proofs)


def _derive_tunnel_collision_proofs(
    frames: Sequence[TrajectoryFrame],
    events_by_update: Mapping[int, tuple[Mapping[str, Any], ...]],
    *,
    original_root: Path | None,
    level_id: str | None,
    hard: bool,
    curve_index: int,
) -> tuple[Mapping[str, Any], ...]:
    if original_root is None or level_id is None:
        return ()
    candidate_pairs = [
        (before, after)
        for before, after in zip(frames, frames[1:])
        if any(entity.zone == "fired" for entity in before.entities)
        and any(entity.zone == "fired" for entity in after.entities)
    ]
    if not candidate_pairs:
        return ()
    curve = _original_curve(
        original_root=original_root,
        level_id=level_id,
        hard=hard,
        curve_index=curve_index,
    )
    proofs: list[Mapping[str, Any]] = []
    staging_zone = (
        f"curve:{curve_index}:list:{INSERTION_STAGING_LIST_OFFSET:03x}"
    )
    for before, after in candidate_pairs:
        if any(
            entity.zone == staging_zone
            for entity in (*before.entities, *after.entities)
        ):
            continue
        active = _curve_entities(
            before,
            curve_index=curve_index,
            container_offset=ACTIVE_CHAIN_LIST_OFFSET,
        )
        fit = _curve_fit(active, curve)
        if not active or fit is None:
            continue
        after_by_id = after.entities_by_id
        for projectile in (
            entity
            for entity in before.entities
            if entity.zone == "fired"
            and entity.object_kind == "bullet"
        ):
            retained = after_by_id.get(projectile.ball_id)
            projectile_radius = _positive_radius(projectile)
            values = (
                projectile.position_x,
                projectile.position_y,
                projectile.velocity_x,
                projectile.velocity_y,
            )
            if (
                retained is None
                or retained.zone != "fired"
                or retained.object_kind != "bullet"
                or projectile.fired is not False
                or retained.fired is not False
                or projectile_radius is None
                or any(
                    value is None or not math.isfinite(value)
                    for value in values
                )
                or retained.velocity_x != projectile.velocity_x
                or retained.velocity_y != projectile.velocity_y
            ):
                continue
            position = np.asarray(
                (projectile.position_x, projectile.position_y),
                dtype=np.float32,
            )
            velocity = np.asarray(
                (projectile.velocity_x, projectile.velocity_y),
                dtype=np.float32,
            )
            advanced = np.add(position, velocity, dtype=np.float32)
            if (
                not math.isclose(
                    retained.position_x,
                    float(advanced[0]),
                    rel_tol=0.0,
                    abs_tol=1e-5,
                )
                or not math.isclose(
                    retained.position_y,
                    float(advanced[1]),
                    rel_tol=0.0,
                    abs_tol=1e-5,
                )
            ):
                continue

            tunnel_hits: list[dict[str, Any]] = []
            non_tunnel_overlap = False
            ambiguous_overlap = False
            for substep, projectile_position in enumerate(
                (position, advanced)
            ):
                for ball in active:
                    if ball.exploding is True:
                        continue
                    ball_radius = _positive_radius(ball)
                    if (
                        ball.object_kind != "ball"
                        or ball.exploding is None
                        or ball_radius is None
                        or not math.isclose(
                            ball_radius,
                            round(ball_radius),
                            rel_tol=0.0,
                            abs_tol=1e-7,
                        )
                    ):
                        ambiguous_overlap = True
                        break
                    ball_position = np.asarray(
                        (ball.position_x, ball.position_y),
                        dtype=np.float32,
                    )
                    delta = np.subtract(
                        projectile_position,
                        ball_position,
                        dtype=np.float32,
                    )
                    squared_distance = np.add(
                        np.multiply(delta[0], delta[0], dtype=np.float32),
                        np.multiply(delta[1], delta[1], dtype=np.float32),
                        dtype=np.float32,
                    )
                    threshold = np.add(
                        np.float32(projectile_radius),
                        np.float32(ball_radius),
                        dtype=np.float32,
                    )
                    threshold_squared = np.multiply(
                        threshold,
                        threshold,
                        dtype=np.float32,
                    )
                    if squared_distance >= threshold_squared:
                        continue
                    perpendicular = np.asarray(
                        curve.perpendicular_at_waypoint(
                            ball.curve_distance
                        ),
                        dtype=np.float32,
                    )
                    if (
                        perpendicular.shape != (2,)
                        or not bool(np.all(np.isfinite(perpendicular)))
                    ):
                        ambiguous_overlap = True
                        break
                    cross_z = np.subtract(
                        np.multiply(
                            delta[0],
                            perpendicular[1],
                            dtype=np.float32,
                        ),
                        np.multiply(
                            delta[1],
                            perpendicular[0],
                            dtype=np.float32,
                        ),
                        dtype=np.float32,
                    )
                    if (
                        not bool(np.isfinite(cross_z))
                        or abs(float(cross_z))
                        <= INSERTION_DIRECTION_CROSS_EPSILON_PX
                    ):
                        ambiguous_overlap = True
                        break
                    hit_in_front = bool(cross_z < np.float32(0.0))
                    tunnel_waypoint = int(ball.curve_distance) + (
                        int(round(ball_radius))
                        if hit_in_front
                        else -int(round(ball_radius))
                    )
                    in_tunnel = bool(
                        curve.is_in_tunnel_at_waypoint(tunnel_waypoint)
                    )
                    if not in_tunnel:
                        non_tunnel_overlap = True
                        break
                    tunnel_hits.append(
                        {
                            "substep": substep,
                            "hit_ball_id": ball.ball_id,
                            "hit_ball_index": ball.index,
                            "hit_in_front": hit_in_front,
                            "cross_z": float(cross_z),
                            "tunnel_waypoint": tunnel_waypoint,
                            "distance_px": float(
                                np.sqrt(squared_distance)
                            ),
                            "collision_threshold_px": float(threshold),
                        }
                    )
                if ambiguous_overlap or non_tunnel_overlap:
                    break
            if (
                ambiguous_overlap
                or non_tunnel_overlap
                or not tunnel_hits
            ):
                continue
            rows = events_by_update.get(after.update, ())
            if any(
                (
                    row.get("kind") == "projectile_hit_staging"
                    and row.get("ball_id") == projectile.ball_id
                )
                or (
                    row.get("kind") == "entity_removed"
                    and isinstance(row.get("entity"), Mapping)
                    and row["entity"].get("ball_id")
                    == projectile.ball_id
                )
                for row in rows
            ):
                continue
            proofs.append(
                {
                    "feature": "tunnel_collision",
                    "status": "PASS",
                    "from_update": before.update,
                    "framework_update": after.update,
                    "curve_index": curve_index,
                    "projectile_ball_id": projectile.ball_id,
                    "projectile_color_id": projectile.color_id,
                    "projectile_position_before": [
                        float(position[0]),
                        float(position[1]),
                    ],
                    "projectile_position_after": [
                        float(advanced[0]),
                        float(advanced[1]),
                    ],
                    "projectile_velocity": [
                        float(velocity[0]),
                        float(velocity[1]),
                    ],
                    "tunnel_overlaps": tunnel_hits,
                    "non_tunnel_overlap_count": 0,
                    "staging_list_empty": True,
                    "projectile_remained_free": True,
                    "curve_fit_maximum_error_px": fit[0],
                    "curve_fit_rms_error_px": fit[1],
                    "proof": (
                        "the retained free projectile follows its exact "
                        "float32 substep, physically overlaps only installed-"
                        "curve collision sides marked as tunnel, and never "
                        "enters insertion staging"
                    ),
                }
            )
    return tuple(proofs)


def _simulate_gap_checks(
    *,
    curve: OriginalCurve,
    chain: Sequence[TrajectoryEntity],
    projectile_positions: Sequence[np.ndarray],
    projectile_radius: int,
    curve_index: int,
    initial_curve_point: int,
    existing_boundary_ids: set[int],
) -> tuple[
    tuple[tuple[int, int, int], ...],
    int,
    tuple[Mapping[str, Any], ...],
] | None:
    count = int(curve.end_waypoint) + 1
    if count <= 1 or not 0 <= initial_curve_point < count:
        return None
    step = projectile_radius * 2
    if step <= 0:
        return None
    # Static retail proof at 0x0045C500 establishes a 2r sample stride but an
    # r**2 proximity threshold.  CircleShootApp's Deluxe source differs.
    radius_squared = np.multiply(
        np.float32(projectile_radius),
        np.float32(projectile_radius),
        dtype=np.float32,
    )
    curve_point = initial_curve_point
    known_boundaries = set(existing_boundary_ids)
    additions: list[tuple[int, int, int]] = []
    hits: list[Mapping[str, Any]] = []

    def squared_distance(
        left: np.ndarray,
        right: np.ndarray,
    ) -> np.float32:
        delta = np.subtract(left, right, dtype=np.float32)
        return np.add(
            np.multiply(delta[0], delta[0], dtype=np.float32),
            np.multiply(delta[1], delta[1], dtype=np.float32),
            dtype=np.float32,
        )

    for substep, projectile_position in enumerate(projectile_positions):
        if 0 < curve_point < count:
            latched_point = np.asarray(
                curve.point_at_waypoint(float(curve_point)),
                dtype=np.float32,
            )
            if (
                latched_point.shape != (2,)
                or not bool(np.all(np.isfinite(latched_point)))
            ):
                return None
            if (
                squared_distance(latched_point, projectile_position)
                < radius_squared
            ):
                continue
            curve_point = 0

        hit_waypoint: int | None = None
        hit_distance_squared: np.float32 | None = None
        for waypoint in range(1, count, step):
            if bool(curve.is_in_tunnel_at_waypoint(waypoint)):
                continue
            point = np.asarray(
                curve.point_at_waypoint(float(waypoint)),
                dtype=np.float32,
            )
            if point.shape != (2,) or not bool(np.all(np.isfinite(point))):
                return None
            distance_squared = squared_distance(point, projectile_position)
            if distance_squared >= radius_squared:
                continue
            hit_waypoint = waypoint
            hit_distance_squared = distance_squared
            curve_point = waypoint
            break
        if hit_waypoint is None or hit_distance_squared is None:
            continue

        candidate: tuple[int, int, int] | None = None
        for ball_index, ball in enumerate(chain):
            if ball.curve_distance <= hit_waypoint:
                continue
            if ball_index == 0:
                break
            if ball.exploding is True and not any(
                later.exploding is False
                for later in chain[ball_index + 1 :]
            ):
                break
            previous = chain[ball_index - 1]
            gap_distance = int(
                ball.curve_distance - previous.curve_distance
            )
            if gap_distance <= 0:
                break
            candidate = (
                curve_index,
                gap_distance,
                ball.ball_id,
            )
            break
        hit_row: dict[str, Any] = {
            "substep": substep,
            "curve_waypoint": hit_waypoint,
            "distance_px": float(np.sqrt(hit_distance_squared)),
        }
        if candidate is not None:
            hit_row.update(
                {
                    "gap_distance": candidate[1],
                    "boundary_ball_id": candidate[2],
                }
            )
            if candidate[2] not in known_boundaries:
                known_boundaries.add(candidate[2])
                additions.append(candidate)
        hits.append(hit_row)
    return tuple(additions), curve_point, tuple(hits)


def _derive_gap_shot_proofs(
    frames: Sequence[TrajectoryFrame],
    events_by_update: Mapping[int, tuple[Mapping[str, Any], ...]],
    *,
    original_root: Path | None,
    level_id: str | None,
    hard: bool,
    curve_index: int,
) -> tuple[Mapping[str, Any], ...]:
    if original_root is None or level_id is None:
        return ()
    candidates = [
        (before, after)
        for before, after in zip(frames, frames[1:])
        if any(
            entity.zone == "fired"
            and entity.object_kind == "bullet"
            and entity.gap_entry_count is not None
            for entity in before.entities
        )
    ]
    if not candidates:
        return ()
    curve = _original_curve(
        original_root=original_root,
        level_id=level_id,
        hard=hard,
        curve_index=curve_index,
    )
    staging_zone = (
        f"curve:{curve_index}:list:{INSERTION_STAGING_LIST_OFFSET:03x}"
    )
    proofs: list[Mapping[str, Any]] = []
    for before, after in candidates:
        if (
            before.score != after.score
            or before.list_counts != after.list_counts
            or any(
                entity.zone == staging_zone
                for entity in (*before.entities, *after.entities)
            )
        ):
            continue
        chain_before = _curve_entities(
            before,
            curve_index=curve_index,
            container_offset=ACTIVE_CHAIN_LIST_OFFSET,
        )
        chain_after = _curve_entities(
            after,
            curve_index=curve_index,
            container_offset=ACTIVE_CHAIN_LIST_OFFSET,
        )
        fit_before = _curve_fit(chain_before, curve)
        fit_after = _curve_fit(chain_after, curve)
        if (
            not chain_before
            or tuple(ball.ball_id for ball in chain_before)
            != tuple(ball.ball_id for ball in chain_after)
            or fit_before is None
            or fit_after is None
            or any(
                ball.object_kind != "ball"
                or not isinstance(ball.exploding, bool)
                or not math.isfinite(ball.curve_distance)
                for ball in (*chain_before, *chain_after)
            )
        ):
            continue
        after_by_id = after.entities_by_id
        for projectile in (
            entity
            for entity in before.entities
            if entity.zone == "fired"
            and entity.object_kind == "bullet"
        ):
            retained = after_by_id.get(projectile.ball_id)
            if (
                retained is None
                or retained.zone != "fired"
                or retained.object_kind != "bullet"
                or projectile.fired is not False
                or retained.fired is not False
                or projectile.radius is None
                or not math.isclose(
                    projectile.radius,
                    round(projectile.radius),
                    rel_tol=0.0,
                    abs_tol=1e-7,
                )
                or projectile.radius <= 0.0
                or projectile.velocity_x is None
                or projectile.velocity_y is None
                or retained.velocity_x != projectile.velocity_x
                or retained.velocity_y != projectile.velocity_y
                or projectile.gap_entry_count
                != len(projectile.gap_entries)
                or retained.gap_entry_count != len(retained.gap_entries)
                or retained.gap_entry_count
                != projectile.gap_entry_count + 1
                or retained.gap_entries[:-1] != projectile.gap_entries
                or projectile.curve_points is None
                or retained.curve_points is None
                or len(projectile.curve_points) != 4
                or len(retained.curve_points) != 4
                or projectile.gap_list_sentinel_address is None
                or retained.gap_list_sentinel_address
                != projectile.gap_list_sentinel_address
            ):
                continue
            values = (
                projectile.position_x,
                projectile.position_y,
                projectile.velocity_x,
                projectile.velocity_y,
                retained.position_x,
                retained.position_y,
            )
            if any(not math.isfinite(value) for value in values):
                continue
            position = np.asarray(
                (projectile.position_x, projectile.position_y),
                dtype=np.float32,
            )
            velocity = np.asarray(
                (projectile.velocity_x, projectile.velocity_y),
                dtype=np.float32,
            )
            advanced = np.add(position, velocity, dtype=np.float32)
            if (
                not math.isclose(
                    retained.position_x,
                    float(advanced[0]),
                    rel_tol=0.0,
                    abs_tol=1e-5,
                )
                or not math.isclose(
                    retained.position_y,
                    float(advanced[1]),
                    rel_tol=0.0,
                    abs_tol=1e-5,
                )
            ):
                continue
            new_entry = retained.gap_entries[-1]
            if (
                new_entry[0] != curve_index
                or new_entry[2]
                in {entry[2] for entry in projectile.gap_entries}
                or any(
                    projectile.curve_points[index]
                    != retained.curve_points[index]
                    for index in range(4)
                    if index != curve_index
                )
            ):
                continue
            simulation_rows: list[Mapping[str, Any]] = []
            simulation_valid = True
            for chain_role, chain in (
                ("before_chain", chain_before),
                ("after_chain", chain_after),
            ):
                simulation = _simulate_gap_checks(
                    curve=curve,
                    chain=chain,
                    projectile_positions=(position, advanced),
                    projectile_radius=int(round(projectile.radius)),
                    curve_index=curve_index,
                    initial_curve_point=projectile.curve_points[
                        curve_index
                    ],
                    existing_boundary_ids={
                        entry[2] for entry in projectile.gap_entries
                    },
                )
                if (
                    simulation is None
                    or simulation[0] != (new_entry,)
                    or simulation[1]
                    != retained.curve_points[curve_index]
                ):
                    simulation_valid = False
                    break
                simulation_rows.append(
                    {
                        "chain_role": chain_role,
                        "new_entries": [
                            {
                                "curve_index": entry[0],
                                "gap_distance": entry[1],
                                "boundary_ball_id": entry[2],
                            }
                            for entry in simulation[0]
                        ],
                        "final_curve_point": simulation[1],
                        "curve_hits": list(simulation[2]),
                    }
                )
            if not simulation_valid:
                continue
            rows = events_by_update.get(after.update, ())
            if any(
                (
                    row.get("kind") == "projectile_hit_staging"
                    and row.get("ball_id") == projectile.ball_id
                )
                or (
                    row.get("kind") == "entity_removed"
                    and isinstance(row.get("entity"), Mapping)
                    and row["entity"].get("ball_id")
                    == projectile.ball_id
                )
                for row in rows
            ):
                continue
            proofs.append(
                {
                    "feature": "gap_shot",
                    "status": "PASS",
                    "from_update": before.update,
                    "framework_update": after.update,
                    "curve_index": curve_index,
                    "projectile_ball_id": projectile.ball_id,
                    "projectile_radius": int(round(projectile.radius)),
                    "projectile_position_before": [
                        float(position[0]),
                        float(position[1]),
                    ],
                    "projectile_position_after": [
                        float(advanced[0]),
                        float(advanced[1]),
                    ],
                    "new_gap_entry": {
                        "curve_index": new_entry[0],
                        "gap_distance": new_entry[1],
                        "boundary_ball_id": new_entry[2],
                    },
                    "curve_point_before": projectile.curve_points[
                        curve_index
                    ],
                    "curve_point_after": retained.curve_points[curve_index],
                    "retail_layout": {
                        "gap_sentinel_pointer_offset_hex": "0x174",
                        "gap_count_offset_hex": "0x178",
                        "curve_points_offset_hex": "0x17c",
                        "gap_node_fields": [
                            "curve_index",
                            "gap_distance",
                            "boundary_ball_id",
                        ],
                    },
                    "simulations": simulation_rows,
                    "curve_fit_maximum_error_px": max(
                        fit_before[0],
                        fit_after[0],
                    ),
                    "curve_fit_rms_error_px": max(
                        fit_before[1],
                        fit_after[1],
                    ),
                    "projectile_remained_free": True,
                    "proof": (
                        "the raw retail Bullet list grows by exactly one "
                        "source-bound gap node, and both adjacent installed-"
                        "curve chain snapshots independently replay the "
                        "native two-substep curve-sample algorithm to the "
                        "same curve index, gap distance, boundary ball, and "
                        "final curve-point latch"
                    ),
                }
            )
    return tuple(proofs)


def _derive_insertion_direction_proofs(
    frames: Sequence[TrajectoryFrame],
    events: Sequence[Mapping[str, Any]],
    *,
    original_root: Path | None,
    level_id: str | None,
    hard: bool,
    curve_index: int,
    allow_initial_staging_observation: bool = False,
) -> tuple[Mapping[str, Any], ...]:
    if original_root is None or level_id is None:
        return ()
    hit_rows = [
        row for row in events if row.get("kind") == "projectile_hit_staging"
    ]
    if not hit_rows and allow_initial_staging_observation and frames:
        initial_staging = _curve_entities(
            frames[0],
            curve_index=curve_index,
            container_offset=INSERTION_STAGING_LIST_OFFSET,
        )
        if len(initial_staging) == 1:
            hit_rows = [
                {
                    "kind": "projectile_hit_staging",
                    "ball_id": initial_staging[0].ball_id,
                    "update": frames[0].update,
                    "observation_mode": (
                        "formal_exact_step_window_start_staging"
                    ),
                }
            ]
    if not hit_rows:
        return ()
    curve = _original_curve(
        original_root=original_root,
        level_id=level_id,
        hard=hard,
        curve_index=curve_index,
    )
    frames_by_update = {frame.update: frame for frame in frames}
    commits = [
        row for row in events if row.get("kind") == "insertion_commit"
    ]
    proofs: dict[str, Mapping[str, Any]] = {}
    for hit in hit_rows:
        source_ball_id = hit.get("ball_id")
        collision_update = hit.get("update")
        if (
            isinstance(source_ball_id, bool)
            or not isinstance(source_ball_id, int)
            or isinstance(collision_update, bool)
            or not isinstance(collision_update, int)
        ):
            continue
        collision_frame = frames_by_update.get(collision_update)
        if collision_frame is None:
            continue
        staging = _curve_entities(
            collision_frame,
            curve_index=curve_index,
            container_offset=INSERTION_STAGING_LIST_OFFSET,
        )
        if len(staging) != 1 or staging[0].ball_id != source_ball_id:
            continue
        projectile = staging[0]
        if projectile.object_kind != "bullet":
            continue
        projectile_radius = _positive_radius(projectile)
        active = _curve_entities(
            collision_frame,
            curve_index=curve_index,
            container_offset=ACTIVE_CHAIN_LIST_OFFSET,
        )
        fit = _curve_fit(active, curve)
        if projectile_radius is None or fit is None:
            continue

        projectile_position = np.asarray(
            (projectile.position_x, projectile.position_y),
            dtype=np.float32,
        )
        colliding: list[tuple[float, TrajectoryEntity, float]] = []
        for entity in active:
            ball_radius = _positive_radius(entity)
            if ball_radius is None or entity.object_kind != "ball":
                continue
            ball_position = np.asarray(
                (entity.position_x, entity.position_y),
                dtype=np.float32,
            )
            delta = np.subtract(
                projectile_position,
                ball_position,
                dtype=np.float32,
            )
            squared_distance = np.add(
                np.multiply(delta[0], delta[0], dtype=np.float32),
                np.multiply(delta[1], delta[1], dtype=np.float32),
                dtype=np.float32,
            )
            threshold = np.add(
                np.float32(projectile_radius),
                np.float32(ball_radius),
                dtype=np.float32,
            )
            threshold_squared = np.multiply(
                threshold,
                threshold,
                dtype=np.float32,
            )
            if squared_distance < threshold_squared:
                colliding.append(
                    (
                        float(np.sqrt(squared_distance)),
                        entity,
                        float(threshold),
                    )
                )
        if len(colliding) != 1:
            continue
        collision_distance, hit_ball, collision_threshold = colliding[0]
        perpendicular = np.asarray(
            curve.perpendicular_at_waypoint(hit_ball.curve_distance),
            dtype=np.float32,
        )
        if (
            perpendicular.shape != (2,)
            or not bool(np.all(np.isfinite(perpendicular)))
        ):
            continue
        offset = np.subtract(
            projectile_position,
            np.asarray(
                (hit_ball.position_x, hit_ball.position_y),
                dtype=np.float32,
            ),
            dtype=np.float32,
        )
        cross_z = np.subtract(
            np.multiply(offset[0], perpendicular[1], dtype=np.float32),
            np.multiply(offset[1], perpendicular[0], dtype=np.float32),
            dtype=np.float32,
        )
        if (
            not bool(np.isfinite(cross_z))
            or abs(float(cross_z)) <= INSERTION_DIRECTION_CROSS_EPSILON_PX
        ):
            continue
        hit_in_front = bool(cross_z < np.float32(0.0))
        hit_side_waypoint = int(hit_ball.curve_distance) + (
            float(hit_ball.radius)
            if hit_in_front
            else -float(hit_ball.radius)
        )
        if bool(curve.is_in_tunnel_at_waypoint(hit_side_waypoint)):
            continue

        matching_commits = [
            row
            for row in commits
            if isinstance(row.get("source_entity"), Mapping)
            and row["source_entity"].get("ball_id") == source_ball_id
            and isinstance(row.get("update"), int)
            and not isinstance(row.get("update"), bool)
            and row["update"] > collision_update
        ]
        if len(matching_commits) != 1:
            continue
        commit = matching_commits[0]
        insertion_update = int(commit["update"])
        before_frame = frames_by_update.get(insertion_update - 1)
        after_frame = frames_by_update.get(insertion_update)
        inserted = commit.get("inserted_entity")
        source = commit.get("source_entity")
        if (
            before_frame is None
            or after_frame is None
            or not isinstance(inserted, Mapping)
            or not isinstance(source, Mapping)
        ):
            continue
        inserted_ball_id = inserted.get("ball_id")
        if (
            isinstance(inserted_ball_id, bool)
            or not isinstance(inserted_ball_id, int)
            or inserted.get("color_id") != projectile.color_id
            or source.get("color_id") != projectile.color_id
        ):
            continue
        before_staging = _curve_entities(
            before_frame,
            curve_index=curve_index,
            container_offset=INSERTION_STAGING_LIST_OFFSET,
        )
        after_staging = _curve_entities(
            after_frame,
            curve_index=curve_index,
            container_offset=INSERTION_STAGING_LIST_OFFSET,
        )
        before_active = _curve_entities(
            before_frame,
            curve_index=curve_index,
            container_offset=ACTIVE_CHAIN_LIST_OFFSET,
        )
        after_active = _curve_entities(
            after_frame,
            curve_index=curve_index,
            container_offset=ACTIVE_CHAIN_LIST_OFFSET,
        )
        before_ids = [entity.ball_id for entity in before_active]
        after_ids = [entity.ball_id for entity in after_active]
        if (
            len(before_staging) != 1
            or before_staging[0].ball_id != source_ball_id
            or after_staging
            or source_ball_id in after_frame.entities_by_id
            or inserted_ball_id in before_frame.entities_by_id
            or hit_ball.ball_id not in before_ids
            or hit_ball.ball_id not in after_ids
        ):
            continue
        hit_index = before_ids.index(hit_ball.ball_id)
        expected_insertion_index = hit_index + (1 if hit_in_front else 0)
        expected_after_ids = list(before_ids)
        expected_after_ids.insert(expected_insertion_index, inserted_ball_id)
        if after_ids != expected_after_ids:
            continue
        feature = "front_insertion" if hit_in_front else "back_insertion"
        proofs.setdefault(
            feature,
            {
                "feature": feature,
                "status": "PASS",
                "collision_update": collision_update,
                "collision_observation_mode": hit.get(
                    "observation_mode",
                    "fired_to_staging_zone_transition",
                ),
                "insertion_update": insertion_update,
                "source_projectile_ball_id": source_ball_id,
                "inserted_ball_id": inserted_ball_id,
                "hit_ball_id": hit_ball.ball_id,
                "hit_index_before": hit_index,
                "inserted_index_after": expected_insertion_index,
                "cross_z": float(cross_z),
                "cross_epsilon_px": INSERTION_DIRECTION_CROSS_EPSILON_PX,
                "collision_distance_px": collision_distance,
                "collision_threshold_px": collision_threshold,
                "curve_fit_maximum_error_px": fit[0],
                "curve_fit_rms_error_px": fit[1],
                "hit_side_tunnel": False,
                "proof": (
                    "exact installed-curve geometry gives an unambiguous "
                    "collision side, and the later native active-list order "
                    "commits the replacement ball on that same side"
                ),
            },
        )
    return tuple(proofs[name] for name in sorted(proofs))


def _derive_swap_proofs(
    frames: Sequence[TrajectoryFrame],
    events_by_update: Mapping[int, tuple[Mapping[str, Any], ...]],
) -> tuple[Mapping[str, Any], ...]:
    frames_by_update = {frame.update: frame for frame in frames}
    proofs: list[Mapping[str, Any]] = []
    for update, rows in sorted(events_by_update.items()):
        chamber_rows = [
            row for row in rows if row.get("kind") == "shooter_chamber_change"
        ]
        if len(chamber_rows) != 1:
            continue
        before_frame = frames_by_update.get(update - 1)
        after_frame = frames_by_update.get(update)
        if before_frame is None or after_frame is None:
            continue
        before = chamber_rows[0].get("before")
        after = chamber_rows[0].get("after")
        if not isinstance(before, Mapping) or not isinstance(after, Mapping):
            continue
        before_current = before.get("current_ball_id")
        before_next = before.get("next_ball_id")
        after_current = after.get("current_ball_id")
        after_next = after.get("next_ball_id")
        identifiers = (
            before_current,
            before_next,
            after_current,
            after_next,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in identifiers
        ):
            continue
        if (
            before_current == before_next
            or after_current != before_next
            or after_next != before_current
            or after.get("current_color_id") != before.get("next_color_id")
            or after.get("next_color_id") != before.get("current_color_id")
            or before_frame.qrand is None
            or after_frame.qrand != before_frame.qrand
            or before_frame.thread_crt_rand_state is None
            or after_frame.thread_crt_rand_state
            != before_frame.thread_crt_rand_state
            or before_frame.score != after_frame.score
            or before_frame.displayed_score != after_frame.displayed_score
            or before_frame.score_target != after_frame.score_target
            or before_frame.list_counts != after_frame.list_counts
        ):
            continue
        expected_zone_changes = {
            (
                before_current,
                "shooter_current",
                "shooter_next",
            ),
            (
                before_next,
                "shooter_next",
                "shooter_current",
            ),
        }
        zone_changes = {
            (
                row.get("ball_id"),
                row.get("from_zone"),
                row.get("to_zone"),
            )
            for row in rows
            if row.get("kind") == "entity_zone_change"
        }
        if (
            zone_changes != expected_zone_changes
            or len(rows) != 3
            or any(
                row.get("kind")
                not in {"shooter_chamber_change", "entity_zone_change"}
                for row in rows
            )
        ):
            continue
        before_current_entity = before_frame.entities_by_id.get(before_current)
        before_next_entity = before_frame.entities_by_id.get(before_next)
        after_current_entity = after_frame.entities_by_id.get(after_current)
        after_next_entity = after_frame.entities_by_id.get(after_next)
        shooter_entities = (
            before_current_entity,
            before_next_entity,
            after_current_entity,
            after_next_entity,
        )
        if any(entity is None for entity in shooter_entities):
            continue
        if (
            before_current_entity.zone != "shooter_current"
            or before_next_entity.zone != "shooter_next"
            or after_current_entity.zone != "shooter_current"
            or after_next_entity.zone != "shooter_next"
            or any(
                entity.object_kind != "bullet" or entity.fired is not False
                for entity in shooter_entities
                if entity is not None
            )
        ):
            continue
        proofs.append(
            {
                "feature": "swap",
                "status": "PASS",
                "framework_update": update,
                "current_ball_id_before": before_current,
                "next_ball_id_before": before_next,
                "current_ball_id_after": after_current,
                "next_ball_id_after": after_next,
                "qrand_unchanged": True,
                "thread_crt_rand_unchanged": True,
                "score_and_curve_lists_unchanged": True,
                "proof": (
                    "the two retail shooter identities and colours exchange "
                    "current/next zones exactly, with no projectile event, "
                    "RNG draw, score change, or curve-list mutation"
                ),
            }
        )
    return tuple(proofs)


def _derive_fruit_visual_oscillator_proof(
    frames: Sequence[TrajectoryFrame],
) -> Mapping[str, Any] | None:
    """Derive the retail float32 bob/cell oscillator without simulator code."""

    if len(frames) < 501 or any(frame.fruit_state is None for frame in frames):
        return None
    states = [frame.fruit_state for frame in frames]
    if any(state is None or state.collecting for state in states):
        return None
    typed_states = [state for state in states if state is not None]
    if any(frame.native_game_time is None for frame in frames):
        return None
    if any(frame.board_update_count is None for frame in frames):
        return None

    def f32(value: float) -> np.float32:
        return np.float32(value)

    def f32_add(left: float, right: float) -> np.float32:
        return np.float32(f32(left) + f32(right))

    def same_f32(left: float, right: float) -> bool:
        return f32(left).tobytes() == f32(right).tobytes()

    state = typed_states[0]
    velocity = f32(state.velocity)
    max_velocity = f32(state.max_velocity)
    acceleration = f32(state.acceleration)
    vertical_offset = f32(state.vertical_offset)
    lower_bound = f32(state.lower_bound)
    upper_bound = f32(state.upper_bound)
    glow_alpha = state.glow_alpha
    glow_step = state.glow_step
    cell_index = state.cell_index
    cell_count = 60
    maximum_float = float(np.finfo(np.float32).max)
    for previous, frame, observed in zip(
        frames[:-1],
        frames[1:],
        typed_states[1:],
        strict=True,
    ):
        assert previous.native_game_time is not None
        assert frame.native_game_time is not None
        assert previous.board_update_count is not None
        assert frame.board_update_count is not None
        if (
            frame.update != previous.update + 1
            or frame.native_game_time != previous.native_game_time + 1
            or frame.board_update_count
            != previous.board_update_count + 1
        ):
            return None
        if frame.native_game_time == state.expiry_time - 200:
            glow_step *= 4
        if frame.board_update_count % 3 == 0:
            cell_index = (cell_index + 1) % cell_count
        velocity = f32_add(velocity, acceleration)
        if acceleration < f32(0.0) and velocity <= -max_velocity:
            acceleration = f32(-acceleration)
            if abs(float(lower_bound) - maximum_float) <= 1.0:
                lower_bound = vertical_offset
            else:
                vertical_offset = lower_bound
        elif acceleration >= f32(0.0) and velocity > max_velocity:
            acceleration = f32(-acceleration)
            if abs(float(upper_bound) - maximum_float) <= 1.0:
                upper_bound = vertical_offset
            else:
                vertical_offset = upper_bound
        vertical_offset = f32_add(vertical_offset, velocity)
        glow_alpha += glow_step
        if glow_step > 0 and glow_alpha >= 255:
            glow_alpha = 255
            glow_step = -glow_step
        elif glow_step < 0 and glow_alpha <= 0:
            glow_alpha = 0
            glow_step = -glow_step
        if (
            observed.active_point_pointer != state.active_point_pointer
            or observed.selected_point_index != state.selected_point_index
            or observed.collecting is not False
            or not same_f32(observed.velocity, velocity)
            or not same_f32(observed.max_velocity, max_velocity)
            or not same_f32(observed.acceleration, acceleration)
            or not same_f32(observed.vertical_offset, vertical_offset)
            or not same_f32(observed.lower_bound, lower_bound)
            or not same_f32(observed.upper_bound, upper_bound)
            or observed.glow_alpha != glow_alpha
            or observed.glow_step != glow_step
            or observed.alpha != state.alpha
            or observed.expiry_time != state.expiry_time
            or observed.cell_index != cell_index
        ):
            return None
    offsets = [state.vertical_offset for state in typed_states]
    velocities = [state.velocity for state in typed_states]
    cells = {state.cell_index for state in typed_states}
    if (
        min(offsets) >= 0.0
        or max(offsets) <= 0.0
        or min(velocities) >= 0.0
        or max(velocities) <= 0.0
        or cells != set(range(cell_count))
    ):
        return None
    return {
        "feature": "fruit_visual_oscillator",
        "status": "PASS",
        "start_update": frames[0].update,
        "end_update": frames[-1].update,
        "transition_count": len(frames) - 1,
        "float32_bit_exact": True,
        "integer_exact": True,
        "vertical_offset_min": min(offsets),
        "vertical_offset_max": max(offsets),
        "velocity_min": min(velocities),
        "velocity_max": max(velocities),
        "cell_indices_observed": sorted(cells),
        "proof": (
            "every source-bound Board transition follows the independently "
            "recomputed float32 velocity, bound, offset, glow, and 60-cell "
            "oscillator recurrence"
        ),
    }


def _derive_natural_loss_proofs(
    frames: Sequence[TrajectoryFrame],
    *,
    original_root: Path | None,
    level_id: str | None,
    hard: bool,
    curve_index: int,
) -> tuple[Mapping[str, Any], ...]:
    """Derive skull-entry loss only from the exact native suction sequence."""

    if original_root is None or level_id is None or len(frames) < 5:
        return ()
    curve = _original_curve(
        original_root=original_root,
        level_id=level_id,
        hard=hard,
        curve_index=curve_index,
    )
    decoded_curve_end = int(curve.end_waypoint)
    if decoded_curve_end < 1:
        return ()

    proofs: list[Mapping[str, Any]] = []
    for trigger_index in range(2, len(frames)):
        before = frames[trigger_index - 1]
        after = frames[trigger_index]
        if (
            before.board_runtime_flag_157 is not True
            or after.board_runtime_flag_157 is not False
            or before.board_runtime_i32_f54 != 0
            or after.board_runtime_i32_f54 != -1
            or before.list_count(curve_index, 0x68) != 1
        ):
            continue
        before_chain = _curve_entities(
            before,
            curve_index=curve_index,
            container_offset=ACTIVE_CHAIN_LIST_OFFSET,
        )
        after_chain = _curve_entities(
            after,
            curve_index=curve_index,
            container_offset=ACTIVE_CHAIN_LIST_OFFSET,
        )
        if (
            not before_chain
            or any(entity.object_kind != "ball" for entity in before_chain)
            or any(entity.object_kind != "ball" for entity in after_chain)
        ):
            continue
        after_ids = {entity.ball_id for entity in after_chain}
        removed = tuple(
            entity for entity in before_chain if entity.ball_id not in after_ids
        )
        if (
            len(removed) != 1
            or removed[0] is not before_chain[-1]
            or removed[0].curve_distance != float(decoded_curve_end)
        ):
            continue
        empty_index = next(
            (
                index
                for index in range(trigger_index, len(frames))
                if frames[index].list_count(
                    curve_index,
                    ACTIVE_CHAIN_LIST_OFFSET,
                )
                == 0
                and not _curve_entities(
                    frames[index],
                    curve_index=curve_index,
                    container_offset=ACTIVE_CHAIN_LIST_OFFSET,
                )
            ),
            None,
        )
        if empty_index is None:
            continue
        selected = tuple(frames[trigger_index - 2 : empty_index + 1])
        try:
            oracle = verify_loss_sequence_frames(
                selected,
                curve_index=curve_index,
                trigger_ball_id=removed[0].ball_id,
                decoded_curve_end=decoded_curve_end,
                expected_trigger_update=after.update,
                expected_chain_empty_update=frames[empty_index].update,
                expected_pending_count=1,
                expected_pretrigger_advance=0.125,
            )
        except (PcLossSequenceError, ValueError):
            continue
        trigger = oracle["trigger"]
        suction = oracle["suction"]
        proofs.append(
            {
                "feature": "natural_loss",
                "status": "PASS",
                "start_update": selected[0].update,
                "trigger_update": trigger["update"],
                "empty_update": suction["first_chain_empty_update"],
                "end_update": selected[-1].update,
                "trigger_ball_id": trigger["ball_id"],
                "decoded_curve_end": trigger["decoded_curve_end"],
                "initial_chain_count": suction["initial_chain_count"],
                "ticks_from_trigger_to_empty": suction[
                    "ticks_from_trigger_to_empty"
                ],
                "removal_updates": suction["removal_updates"],
                "oracle": oracle,
                "proof": (
                    "the native Board enters its unique skull-loss state, "
                    "locks score and shooter state, then removes every ball "
                    "by the exact retail suck_count recurrence"
                ),
            }
        )
    return tuple(proofs)


def derive_native_mechanism_features(
    frames: Sequence[TrajectoryFrame],
    *,
    original_root: Path | None = None,
    level_id: str | None = None,
    hard: bool = False,
    curve_index: int = 0,
    allow_initial_staging_observation: bool = False,
) -> tuple[tuple[str, ...], tuple[Mapping[str, Any], ...]]:
    """Derive conservative mechanism proofs from recomputed retail frames."""

    if len(frames) < 2:
        _fail("mechanism_audit_window_too_short")
    if any(
        after.update != before.update + 1
        for before, after in zip(frames, frames[1:])
    ):
        _fail("mechanism_audit_window_not_contiguous")
    events = derive_trajectory_events(frames)
    events_by_update_lists: dict[int, list[Mapping[str, Any]]] = defaultdict(
        list
    )
    for row in events:
        update = _strict_int(
            row.get("update"),
            "mechanism_audit_event_update_invalid",
        )
        events_by_update_lists[update].append(row)
    events_by_update = {
        update: tuple(rows)
        for update, rows in events_by_update_lists.items()
    }

    proofs: dict[str, Mapping[str, Any]] = {}
    powerup_effect_proofs = _derive_powerup_effect_proofs(
        frames,
        original_root=original_root,
        level_id=level_id,
        hard=hard,
        curve_index=curve_index,
    )
    fruit_visual_proof = _derive_fruit_visual_oscillator_proof(frames)
    if fruit_visual_proof is not None:
        proofs["fruit_visual_oscillator"] = fruit_visual_proof
    for proof in _derive_fruit_scheduler_spawn_proofs(
        frames,
        original_root=original_root,
        level_id=level_id,
        hard=hard,
        curve_index=curve_index,
    ):
        proofs.setdefault("fruit_scheduler_spawn", proof)
    for proof in _derive_fruit_expiry_proofs(frames):
        proofs.setdefault("fruit_expiry", proof)
    for proof in _derive_insertion_direction_proofs(
        frames,
        events,
        original_root=original_root,
        level_id=level_id,
        hard=hard,
        curve_index=curve_index,
        allow_initial_staging_observation=(
            allow_initial_staging_observation
        ),
    ):
        feature = proof.get("feature")
        if isinstance(feature, str):
            proofs.setdefault(feature, proof)
    for proof in _derive_swap_proofs(frames, events_by_update):
        proofs.setdefault("swap", proof)
    for proof in _derive_shooter_rng_proofs(
        frames,
        events_by_update,
    ):
        proofs.setdefault("rng_pending", proof)
    for proof in _derive_pending_rng_rejection_proofs(
        frames,
        original_root=original_root,
        level_id=level_id,
        hard=hard,
        curve_index=curve_index,
    ):
        proofs.setdefault("rng_rejection", proof)
    for proof in _derive_tunnel_collision_proofs(
        frames,
        events_by_update,
        original_root=original_root,
        level_id=level_id,
        hard=hard,
        curve_index=curve_index,
    ):
        proofs.setdefault("tunnel_collision", proof)
    for proof in _derive_gap_shot_proofs(
        frames,
        events_by_update,
        original_root=original_root,
        level_id=level_id,
        hard=hard,
        curve_index=curve_index,
    ):
        proofs.setdefault("gap_shot", proof)
    for proof in _derive_powerup_spawn_proofs(
        frames,
        original_root=original_root,
        level_id=level_id,
        hard=hard,
        curve_index=curve_index,
    ):
        proofs.setdefault("powerup_spawn", proof)
    for proof in powerup_effect_proofs:
        feature = proof.get("feature")
        if isinstance(feature, str):
            proofs.setdefault(feature, proof)
    for proof in _derive_fruit_collection_proofs(
        frames,
        original_root=original_root,
        level_id=level_id,
        hard=hard,
        curve_index=curve_index,
        powerup_effect_proofs=powerup_effect_proofs,
    ):
        feature = proof.get("feature")
        if isinstance(feature, str):
            proofs.setdefault(feature, proof)
    for proof in _derive_natural_loss_proofs(
        frames,
        original_root=original_root,
        level_id=level_id,
        hard=hard,
        curve_index=curve_index,
    ):
        proofs.setdefault("natural_loss", proof)
    for proof in derive_natural_win_feature_proofs(
        frames,
        curve_count=1,
    ):
        feature = proof.get("feature")
        if isinstance(feature, str):
            proofs.setdefault(feature, proof)

    spawn_update_by_id: dict[int, int] = {}
    for update in sorted(events_by_update):
        rows = events_by_update[update]
        chamber_rows = [
            row for row in rows if row.get("kind") == "shooter_chamber_change"
        ]
        zone_rows = [
            row
            for row in rows
            if row.get("kind") == "entity_zone_change"
            and row.get("from_zone") == "shooter_current"
            and row.get("to_zone") == "fired"
        ]
        for spawn in (
            row for row in rows if row.get("kind") == "projectile_spawn"
        ):
            entity = spawn.get("entity")
            if not isinstance(entity, Mapping):
                continue
            ball_id = entity.get("ball_id")
            color_id = entity.get("color_id")
            if (
                isinstance(ball_id, bool)
                or not isinstance(ball_id, int)
                or isinstance(color_id, bool)
                or not isinstance(color_id, int)
            ):
                continue
            matching_zone = [
                row for row in zone_rows if row.get("ball_id") == ball_id
            ]
            matching_chamber = [
                row
                for row in chamber_rows
                if isinstance(row.get("before"), Mapping)
                and isinstance(row.get("after"), Mapping)
                and row["before"].get("current_ball_id") == ball_id
                and row["after"].get("current_ball_id")
                == row["before"].get("next_ball_id")
            ]
            if len(matching_zone) != 1 or len(matching_chamber) != 1:
                continue
            spawn_update_by_id[ball_id] = update
            proofs.setdefault(
                "shot_release",
                {
                    "feature": "shot_release",
                    "status": "PASS",
                    "framework_update": update,
                    "ball_id": ball_id,
                    "color_id": color_id,
                    "proof": (
                        "same-tick shooter-current to fired transfer, "
                        "projectile spawn, and chamber advance"
                    ),
                },
            )

    for row in events:
        if row.get("kind") != "projectile_hit_staging":
            continue
        ball_id = row.get("ball_id")
        update = row.get("update")
        if (
            isinstance(ball_id, bool)
            or not isinstance(ball_id, int)
            or isinstance(update, bool)
            or not isinstance(update, int)
            or ball_id not in spawn_update_by_id
            or spawn_update_by_id[ball_id] >= update
        ):
            continue
        proofs.setdefault(
            "projectile_collision",
            {
                "feature": "projectile_collision",
                "status": "PASS",
                "spawn_update": spawn_update_by_id[ball_id],
                "collision_update": update,
                "ball_id": ball_id,
                "color_id": row.get("color_id"),
                "from_zone": row.get("from_zone"),
                "to_zone": row.get("to_zone"),
                "proof": (
                    "a previously observed fired projectile entered the "
                    "retail insertion-staging list"
                ),
            },
        )

    explosion_groups: dict[
        int, tuple[Mapping[str, Any], ...]
    ] = {}
    for update, rows in events_by_update.items():
        explosions = tuple(
            row for row in rows if row.get("kind") == "explosion_started"
        )
        ids = [row.get("ball_id") for row in explosions]
        colors = {row.get("color_id") for row in explosions}
        if (
            len(explosions) < 3
            or any(
                isinstance(ball_id, bool) or not isinstance(ball_id, int)
                for ball_id in ids
            )
            or len(ids) != len(set(ids))
            or len(colors) != 1
            or _positive_score_delta(events_by_update, update) is None
        ):
            continue
        explosion_groups[update] = explosions

    for update, explosions in sorted(explosion_groups.items()):
        rows = events_by_update[update]
        insertion_rows = [
            row for row in rows if row.get("kind") == "insertion_commit"
        ]
        rollback_rows = [
            row for row in rows if row.get("kind") == "rollback_stopped"
        ]
        explosion_ids = {int(row["ball_id"]) for row in explosions}
        direct_ids = {
            row.get("inserted_entity", {}).get("ball_id")
            for row in insertion_rows
            if isinstance(row.get("inserted_entity"), Mapping)
        }
        rollback_ids = {
            row.get("ball_id")
            for row in rollback_rows
            if isinstance(row.get("ball_id"), int)
            and not isinstance(row.get("ball_id"), bool)
        }
        coupled = bool(explosion_ids & (direct_ids | rollback_ids))
        if not coupled:
            continue
        count = len(explosions)
        feature = "match3" if count == 3 else "match4" if count == 4 else None
        if feature is not None:
            proofs.setdefault(
                feature,
                {
                    "feature": feature,
                    "status": "PASS",
                    "framework_update": update,
                    "exploding_ball_ids": sorted(explosion_ids),
                    "color_id": explosions[0].get("color_id"),
                    "score_delta": _positive_score_delta(
                        events_by_update, update
                    ),
                    "trigger": (
                        "insertion_commit"
                        if direct_ids & explosion_ids
                        else "rollback_stopped"
                    ),
                    "proof": (
                        "same-color exact-size explosion group coupled to "
                        "an insertion or rollback stop and positive score"
                    ),
                },
            )

    rollback_starts = [
        row for row in events if row.get("kind") == "rollback_started"
    ]
    rollback_stops = [
        row for row in events if row.get("kind") == "rollback_stopped"
    ]
    for start in rollback_starts:
        ball_id = start.get("ball_id")
        start_update = start.get("update")
        if (
            isinstance(ball_id, bool)
            or not isinstance(ball_id, int)
            or isinstance(start_update, bool)
            or not isinstance(start_update, int)
        ):
            continue
        candidates = [
            row
            for row in rollback_stops
            if row.get("ball_id") == ball_id
            and isinstance(row.get("update"), int)
            and not isinstance(row.get("update"), bool)
            and row["update"] > start_update
            and row["update"] in explosion_groups
            and any(
                explosion.get("ball_id") == ball_id
                and isinstance(explosion.get("combo_count"), int)
                and not isinstance(explosion.get("combo_count"), bool)
                and explosion["combo_count"] >= 1
                for explosion in explosion_groups[row["update"]]
            )
        ]
        if not candidates:
            continue
        stop = min(candidates, key=lambda row: int(row["update"]))
        proofs.setdefault(
            "rollback_chain",
            {
                "feature": "rollback_chain",
                "status": "PASS",
                "ball_id": ball_id,
                "color_id": start.get("color_id"),
                "start_update": start_update,
                "stop_update": stop["update"],
                "score_delta": _positive_score_delta(
                    events_by_update, int(stop["update"])
                ),
                "proof": (
                    "the same retail ball starts rollback, stops later, and "
                    "participates in a scored combo explosion"
                ),
            },
        )

    unsupported = set(proofs) - _SUPPORTED_DERIVED_FEATURES
    if unsupported:
        _fail("mechanism_audit_internal_feature_not_allowlisted")
    features = tuple(sorted(proofs))
    return features, tuple(proofs[name] for name in features)


def _audit_case(
    spec: Mapping[str, Any],
    *,
    evidence_root: Path,
    original_root: Path,
    level_id: str,
    hard: bool,
    profile_mode: str,
) -> dict[str, Any]:
    normalized = _case_spec(spec)
    manifest_relative, manifest_path = _relative_file(
        evidence_root,
        normalized["manifest_path"],
        code="mechanism_audit_manifest_path_invalid",
    )
    probe_relative, probe_path = _relative_file(
        evidence_root,
        normalized["memory_probe_path"],
        code="mechanism_audit_probe_path_invalid",
    )
    manifest_payload = _strict_json(manifest_path)
    if manifest_payload.get("schema") == PC_SOURCE_SCHEMA:
        scope = _formal_mapping(
            manifest_payload.get("scope"),
            "mechanism_audit_pc_source_scope_invalid",
        )
        if (
            scope.get("level_id") != level_id
            or scope.get("hard") is not hard
            or scope.get("profile_mode") != profile_mode
        ):
            _fail("mechanism_audit_scenario_scope_mismatch")
        (
            source_report,
            source,
            all_frames,
            trajectory_path,
            trajectory_sha256,
        ) = _load_pc_source_mechanism_source(
            manifest_path=manifest_path,
            manifest=manifest_payload,
            evidence_root=evidence_root,
            original_root=original_root,
            probe_path=probe_path,
        )
        start = normalized["window"]["start_update"]
        end = normalized["window"]["end_update"]
        if start < all_frames[0].update or end > all_frames[-1].update:
            _fail("mechanism_audit_selected_window_out_of_bounds")
        frames = tuple(
            frame for frame in all_frames if start <= frame.update <= end
        )
        if (
            not frames
            or frames[0].update != start
            or frames[-1].update != end
            or len(frames) != end - start + 1
        ):
            _fail("mechanism_audit_selected_window_incomplete")
        features, proofs = derive_native_mechanism_features(
            frames,
            original_root=original_root,
            level_id=level_id,
            hard=hard,
            curve_index=0,
            allow_initial_staging_observation=(
                start == all_frames[0].update
            ),
        )
        if not features:
            _fail("mechanism_audit_no_machine_derived_features")
        runtime = _formal_mapping(
            manifest_payload.get("runtime_payload"),
            "mechanism_audit_pc_source_runtime_invalid",
        )
        dmo = _formal_mapping(
            manifest_payload.get("dmo"),
            "mechanism_audit_pc_source_dmo_invalid",
        )
        return {
            "id": normalized["id"],
            "status": "PASS",
            "source_kind": "pc_source",
            "manifest_path": manifest_relative,
            "manifest_sha256": _sha256_path(manifest_path),
            "memory_probe_path": probe_relative,
            "memory_probe_sha256": _sha256_path(probe_path),
            "case_id": manifest_payload.get("source_id"),
            "source_fingerprint": source_report["source_fingerprint"],
            "runtime_executable_sha256": runtime.get("sha256"),
            "dmo_sha256": dmo.get("sha256"),
            "window": {
                "start_update": start,
                "end_update": end,
                "tick_count": len(frames),
            },
            "trajectory": {
                "artifact": PurePosixPath(
                    *trajectory_path.relative_to(evidence_root).parts
                ).as_posix(),
                "artifact_sha256": trajectory_sha256,
                "start_update": all_frames[0].update,
                "end_update": all_frames[-1].update,
                "tick_count": len(all_frames),
            },
            "video_binding": _bind_pc_source_frame(
                source=source,
                evidence_root=evidence_root,
                update=start,
            ),
            "authorized_features": list(features),
            "feature_proofs": list(proofs),
        }

    manifest = PcGoldenManifest.read_json(manifest_path)
    if (
        manifest.scenario.level_id != level_id
        or manifest.scenario.hard is not hard
        or manifest.pc_environment.profile_mode != profile_mode
    ):
        _fail("mechanism_audit_scenario_scope_mismatch")
    if manifest.scenario.curve_index != 0:
        _fail("mechanism_audit_curve_scope_mismatch")

    dmo_spec = manifest.artifacts[manifest.input_timeline.artifact]
    dmo_spec.verify(manifest_path.parent)
    exact_step_run: ExactStepRunContract | None = None
    exact_step_source: ExactStepLoadedRun | None = None
    exact_step_artifacts: Mapping[str, Path] | None = None
    probe_validation: Mapping[str, Any] | None = None
    if manifest.exact_step_replay is not None:
        (
            exact_step_run,
            exact_step_source,
            all_frames,
            exact_step_artifacts,
        ) = _load_formal_exact_step_mechanism_source(
            manifest=manifest,
            manifest_root=manifest_path.parent,
            probe_path=probe_path,
        )
        trajectory_path = exact_step_artifacts[
            exact_step_run.trajectory_index_artifact
        ]
        trajectory_row: Mapping[str, Any] = {
            "artifact_sha256": manifest.artifacts[
                exact_step_run.trajectory_index_artifact
            ].sha256,
        }
    else:
        probe = read_canonical_report(probe_path)
        score_binding = _require_natural_probe(probe)
        probe_validation = validate_probe_payloads(
            probe_path,
            expected_runtime_sha256=manifest.pc_environment.executable_sha256,
            expected_dmo_sha256=dmo_spec.sha256,
            require_freeze_state=True,
        )
        trajectory_path, all_frames, trajectory_row = _trajectory_from_probe(
            probe,
            probe_path=probe_path,
        )
        if (
            all_frames[0].update != probe_validation["framework_update"]
            or all_frames[0].score != score_binding["score"]
            or all_frames[0].displayed_score
            != score_binding["displayed_score"]
        ):
            _fail("mechanism_audit_probe_trajectory_start_mismatch")

    selected_first_update = -manifest.input_timeline.native_tick_offset
    pc_last_update = selected_first_update + manifest.clock.tick_end
    start = normalized["window"]["start_update"]
    end = normalized["window"]["end_update"]
    if (
        start < all_frames[0].update
        or end > all_frames[-1].update
        or start < selected_first_update
        or end > pc_last_update
    ):
        _fail("mechanism_audit_selected_window_out_of_bounds")
    frames = tuple(
        frame for frame in all_frames if start <= frame.update <= end
    )
    if (
        not frames
        or frames[0].update != start
        or frames[-1].update != end
        or len(frames) != end - start + 1
    ):
        _fail("mechanism_audit_selected_window_incomplete")

    if exact_step_run is not None:
        if exact_step_source is None or exact_step_artifacts is None:
            _fail("mechanism_audit_exact_step_source_missing")
        video_binding = _bind_exact_step_frame_to_video(
            manifest=manifest,
            manifest_root=manifest_path.parent,
            run=exact_step_run,
            source=exact_step_source,
            verified_artifacts=exact_step_artifacts,
            update=start,
        )
    else:
        if probe_validation is None:
            _fail("mechanism_audit_probe_validation_missing")
        video_binding = _bind_probe_frame_to_video(
            manifest=manifest,
            manifest_root=manifest_path.parent,
            probe_validation=probe_validation,
        )
    features, proofs = derive_native_mechanism_features(
        frames,
        original_root=original_root,
        level_id=level_id,
        hard=hard,
        curve_index=manifest.scenario.curve_index,
        allow_initial_staging_observation=(
            exact_step_run is not None and start == all_frames[0].update
        ),
    )
    if not features:
        _fail("mechanism_audit_no_machine_derived_features")
    return {
        "id": normalized["id"],
        "status": "PASS",
        "manifest_path": manifest_relative,
        "manifest_sha256": _sha256_path(manifest_path),
        "memory_probe_path": probe_relative,
        "memory_probe_sha256": _sha256_path(probe_path),
        "case_id": manifest.case_id,
        "source_fingerprint": pc_golden_native_source_fingerprint(manifest),
        "runtime_executable_sha256": (
            manifest.pc_environment.executable_sha256
        ),
        "dmo_sha256": dmo_spec.sha256,
        "window": {
            "start_update": start,
            "end_update": end,
            "tick_count": len(frames),
        },
        "trajectory": {
            "artifact": PurePosixPath(
                *trajectory_path.relative_to(evidence_root).parts
            ).as_posix(),
            "artifact_sha256": trajectory_row["artifact_sha256"],
            "start_update": all_frames[0].update,
            "end_update": all_frames[-1].update,
            "tick_count": len(all_frames),
        },
        "video_binding": video_binding,
        "authorized_features": list(features),
        "feature_proofs": list(proofs),
    }


def audit_pc_mechanism_coverage(
    case_specs: Sequence[Mapping[str, Any]],
    *,
    evidence_root: str | Path,
    original_root: str | Path | None,
    level_id: str = "Jungle2",
    hard: bool = False,
    profile_mode: str = SUPPORTED_PROFILE_MODE,
) -> dict[str, Any]:
    """Recompute source-bound mechanism feature authorizations."""

    try:
        root = Path(evidence_root).resolve(strict=True)
    except OSError as error:
        raise PcMechanismAuditError(
            "mechanism_audit_evidence_root_missing"
        ) from error
    if not root.is_dir():
        _fail("mechanism_audit_evidence_root_invalid")
    if original_root is None:
        _fail("mechanism_audit_original_root_missing")
    try:
        installed_root = Path(original_root).resolve(strict=True)
    except OSError as error:
        raise PcMechanismAuditError(
            "mechanism_audit_original_root_missing"
        ) from error
    if not installed_root.is_dir():
        _fail("mechanism_audit_original_root_invalid")
    if (
        isinstance(case_specs, (str, bytes))
        or not isinstance(case_specs, Sequence)
        or not case_specs
        or len(case_specs) > MAX_MECHANISM_AUDIT_CASES
    ):
        _fail("mechanism_audit_cases_invalid")
    normalized = tuple(_case_spec(item) for item in case_specs)
    ids = [item["id"] for item in normalized]
    if len(ids) != len(set(ids)):
        _fail("mechanism_audit_case_id_duplicate")

    cases: list[dict[str, Any]] = []
    failures: list[str] = []
    for spec in normalized:
        try:
            cases.append(
                _audit_case(
                    spec,
                    evidence_root=root,
                    original_root=installed_root,
                    level_id=level_id,
                    hard=hard,
                    profile_mode=profile_mode,
                )
            )
        except (
            KeyError,
            OSError,
            OverflowError,
            RecursionError,
            TypeError,
            UnicodeError,
            PcGoldenArtifactError,
            PcGoldenValidationError,
            PcExactStepEvidenceError,
            PcMechanismAuditError,
            PcMemoryEvidenceError,
            PcMemoryImageDecoderUnavailable,
            PcMemoryTrajectoryError,
            PcVideoError,
            ValueError,
        ) as error:
            reason = str(error) or type(error).__name__
            failures.append(f"{spec['id']}:{reason}")
            cases.append(
                {
                    "id": spec["id"],
                    "status": "INCOMPARABLE",
                    "manifest_path": spec["manifest_path"],
                    "memory_probe_path": spec["memory_probe_path"],
                    "window": dict(spec["window"]),
                    "authorized_features": [],
                    "feature_proofs": [],
                    "reason": reason,
                }
            )

    source_features: dict[str, set[str]] = defaultdict(set)
    for case in cases:
        source = case.get("source_fingerprint")
        features = case.get("authorized_features")
        if (
            case.get("status") == "PASS"
            and isinstance(source, str)
            and isinstance(features, list)
        ):
            source_features[source].update(
                feature
                for feature in features
                if feature in _SUPPORTED_DERIVED_FEATURES
            )
    authorizations = [
        {
            "source_fingerprint": source,
            "authorized_features": sorted(features),
        }
        for source, features in sorted(source_features.items())
        if features
    ]
    if not authorizations:
        failures.append("no source has a machine-derived mechanism feature")
    status = "PASS" if not failures else "INCOMPARABLE"
    payload: dict[str, Any] = {
        "schema": AUDIT_SCHEMA,
        "version": AUDIT_VERSION,
        "status": status,
        "audit_type": AUDIT_TYPE,
        "scope": {
            "policy": POLICY_ID,
            "environment_id": "ZumaRevenge-v0",
            "level_id": level_id,
            "hard": hard,
            "profile_mode": profile_mode,
            "observation_mode": "actor",
        },
        "cases": cases,
        "checks": [
            {
                "name": "natural_memory_trajectory_provenance",
                "status": "PASS" if not failures else "INCOMPARABLE",
            },
            {
                "name": "pc_video_tick_binding",
                "status": "PASS" if not failures else "INCOMPARABLE",
            },
            {
                "name": "machine_derived_feature_authorization",
                "status": (
                    "PASS" if authorizations and not failures else "INCOMPARABLE"
                ),
            },
        ],
        "failure_reasons": failures,
        "summary": {
            "case_count": len(cases),
            "passing_case_count": sum(
                case.get("status") == "PASS" for case in cases
            ),
            "pc_source_count": len(authorizations),
            "pc_source_fingerprints": [
                row["source_fingerprint"] for row in authorizations
            ],
            "source_feature_authorizations": authorizations,
            "authorized_feature_union": sorted(
                {
                    feature
                    for row in authorizations
                    for feature in row["authorized_features"]
                }
            ),
        },
    }
    payload["audit_fingerprint"] = canonical_sha256(payload)
    return payload


def validate_pc_mechanism_audit_report(
    report: Mapping[str, Any],
    *,
    evidence_root: str | Path,
    original_root: str | Path | None,
    level_id: str,
    hard: bool,
    profile_mode: str,
) -> str | None:
    """Recompute a submitted PASS report and reject any handwritten drift."""

    if report.get("audit_type") != AUDIT_TYPE:
        return "unsupported PC mechanism audit type"
    if report.get("status") != "PASS":
        return "PC mechanism audit report is not PASS"
    cases = report.get("cases")
    if (
        not isinstance(cases, list)
        or not cases
        or len(cases) > MAX_MECHANISM_AUDIT_CASES
        or any(not isinstance(item, Mapping) for item in cases)
    ):
        return "PC mechanism audit case list is missing or invalid"
    specs: list[dict[str, Any]] = []
    try:
        root = Path(evidence_root).resolve(strict=True)
        for case in cases:
            spec = _case_spec(
                {
                    "id": case.get("id"),
                    "manifest_path": case.get("manifest_path"),
                    "memory_probe_path": case.get("memory_probe_path"),
                    "window": {
                        "start_update": (
                            case.get("window", {}).get("start_update")
                            if isinstance(case.get("window"), Mapping)
                            else None
                        ),
                        "end_update": (
                            case.get("window", {}).get("end_update")
                            if isinstance(case.get("window"), Mapping)
                            else None
                        ),
                    },
                }
            )
            _, manifest_path = _relative_file(
                root,
                spec["manifest_path"],
                code="mechanism_audit_manifest_path_invalid",
            )
            _, probe_path = _relative_file(
                root,
                spec["memory_probe_path"],
                code="mechanism_audit_probe_path_invalid",
            )
            if case.get("manifest_sha256") != _sha256_path(manifest_path):
                return "PC mechanism audit manifest SHA-256 differs"
            if case.get("memory_probe_sha256") != _sha256_path(probe_path):
                return "PC mechanism audit probe SHA-256 differs"
            specs.append(spec)
    except (
        OSError,
        PcMechanismAuditError,
        ValueError,
    ):
        return "PC mechanism audit paths or windows are invalid"

    try:
        live = audit_pc_mechanism_coverage(
            specs,
            evidence_root=root,
            original_root=original_root,
            level_id=level_id,
            hard=hard,
            profile_mode=profile_mode,
        )
    except (
        OSError,
        PcMechanismAuditError,
        ValueError,
    ):
        return "live PC mechanism audit could not be recomputed"
    if live.get("status") != "PASS":
        return "live PC mechanism audit is not comparable or did not pass"
    if not _reports_semantically_equivalent(report, live):
        return "PC mechanism audit differs from live retail evidence"
    summary = report.get("summary")
    if (
        not isinstance(summary, Mapping)
        or not summary.get("source_feature_authorizations")
        or summary.get("pc_source_count", 0) < 1
    ):
        return "PC mechanism audit authorizations are missing"
    return None


def validate_simulator_source_feature_proofs(
    report: Mapping[str, Any],
    *,
    evidence_root: str | Path,
    original_root: str | Path | None,
    level_id: str,
    hard: bool,
) -> str | None:
    """Recompute optional mechanism proofs embedded in a simulator diff."""

    raw_features = report.get("source_authorized_features")
    raw_proofs = report.get("source_feature_proofs")
    if raw_features is None and raw_proofs is None:
        return None
    if (
        isinstance(raw_features, (str, bytes))
        or not isinstance(raw_features, Sequence)
        or isinstance(raw_proofs, (str, bytes))
        or not isinstance(raw_proofs, Sequence)
        or any(not isinstance(item, str) for item in raw_features)
        or any(not isinstance(item, Mapping) for item in raw_proofs)
    ):
        return "simulator source feature proofs are malformed"
    features = tuple(raw_features)
    if (
        features != tuple(sorted(set(features)))
        or not features
        or not set(features).issubset(_SUPPORTED_DERIVED_FEATURES)
    ):
        return "simulator source feature list is malformed"
    trajectory = report.get("trajectory")
    if not isinstance(trajectory, Mapping):
        return "simulator source trajectory binding is missing"
    source_version = trajectory.get("source_version")
    if isinstance(source_version, bool) or source_version not in {1, 2}:
        return "simulator source trajectory version is invalid"
    start = trajectory.get("selected_start_update")
    end = trajectory.get("selected_end_update")
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(end, bool)
        or not isinstance(end, int)
        or end <= start
        or report.get("start_update") != start
        or report.get("end_update") != end
    ):
        return "simulator source trajectory window is invalid"
    proof_start = trajectory.get("source_proof_start_update", start)
    proof_end = trajectory.get("source_proof_end_update", end)
    if (
        isinstance(proof_start, bool)
        or not isinstance(proof_start, int)
        or isinstance(proof_end, bool)
        or not isinstance(proof_end, int)
        or proof_start < 0
        or proof_end <= proof_start
        or proof_start > start
        or proof_end < end
    ):
        return "simulator source proof window is invalid"
    captured_start = trajectory.get("captured_start_update")
    captured_end = trajectory.get("captured_end_update")
    if (captured_start is None) != (captured_end is None):
        return "simulator source capture window is malformed"
    if captured_start is not None:
        if (
            isinstance(captured_start, bool)
            or not isinstance(captured_start, int)
            or isinstance(captured_end, bool)
            or not isinstance(captured_end, int)
            or captured_start < 0
            or captured_end <= captured_start
            or captured_start > proof_start
            or captured_end < proof_end
        ):
            return "simulator source capture window is malformed"
    if original_root is None:
        return "original_root is required for simulator source proofs"
    try:
        root = Path(evidence_root).resolve(strict=True)
        _, index_path = _relative_file(
            root,
            trajectory.get("artifact"),
            code="simulator_source_trajectory_path_invalid",
        )
        if trajectory.get("artifact_sha256") != _sha256_path(index_path):
            return "simulator source trajectory SHA-256 differs"
        loader = (
            load_legacy_memory_trajectory_v1
            if source_version == 1
            else load_memory_trajectory
        )
        frames = loader(
            index_path,
            start_update=proof_start,
            end_update=proof_end,
        )
        live_features, live_proofs = derive_native_mechanism_features(
            frames,
            original_root=Path(original_root).resolve(strict=True),
            level_id=level_id,
            hard=hard,
            curve_index=int(report.get("curve_index", 0)),
            allow_initial_staging_observation=(
                report.get("schema") == "zuma-rl.pc-merge-simulator-diff"
            ),
        )
    except (
        OSError,
        PcMechanismAuditError,
        PcMemoryTrajectoryError,
        TypeError,
        ValueError,
    ):
        return "simulator source feature proofs could not be recomputed"
    if features != live_features or list(raw_proofs) != list(live_proofs):
        return "simulator source feature proofs differ from raw evidence"
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Bind natural retail memory trajectories to PC Golden video ticks "
            "and derive mechanism coverage without handwritten feature labels."
        )
    )
    parser.add_argument("plan", type=Path)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--original-root", required=True, type=Path)
    parser.add_argument("--level", default="Jungle2")
    parser.add_argument("--hard", action="store_true")
    parser.add_argument("--compact", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        help="Exclusively create the recomputed audit report at this path.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cases = read_mechanism_audit_plan(args.plan)
    report = audit_pc_mechanism_coverage(
        cases,
        evidence_root=args.evidence_root,
        original_root=args.original_root,
        level_id=args.level,
        hard=args.hard,
    )
    payload = json.dumps(
        report,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=None if args.compact else 2,
        separators=(",", ":") if args.compact else None,
    )
    if args.output is not None:
        output = args.output.resolve()
        if output.exists() or not output.parent.is_dir():
            _fail("mechanism_audit_output_invalid")
        try:
            with output.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(payload + "\n")
        except OSError as error:
            raise PcMechanismAuditError(
                "mechanism_audit_output_invalid"
            ) from error
    print(payload)
    return 0 if report["status"] == "PASS" else 1


__all__ = [
    "AUDIT_TYPE",
    "MAX_MECHANISM_AUDIT_CASES",
    "PLAN_SCHEMA",
    "PLAN_VERSION",
    "PcMechanismAuditError",
    "audit_pc_mechanism_coverage",
    "build_parser",
    "derive_native_mechanism_features",
    "main",
    "read_mechanism_audit_plan",
    "validate_pc_mechanism_audit_report",
    "validate_simulator_source_feature_proofs",
]


if __name__ == "__main__":
    raise SystemExit(main())
