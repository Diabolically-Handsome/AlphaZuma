"""Strict readers for formal exact-step retail replay evidence.

The exact-step transport deliberately has no continuous-cadence claim.  It
proves a narrower fact: after each requested native 100 Hz update reaches the
frozen replay barrier, a verified external Desktop Duplication snapshot is a
single-valued function of the loaded retail gameplay/RNG/render-driving state.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import struct
from typing import Any, Iterable, Mapping

import numpy as np

from .pc_rng_trajectory import RngTrajectoryFrame, load_rng_trajectory
from .pc_video import canonical_rgb24_frame_sha256
from .pc_external_input import (
    EXTERNAL_INPUT_GUARD_BINDING_SCHEMA,
    EXTERNAL_INPUT_GUARD_BINDING_VERSION,
    PcExternalInputError,
    validate_external_input_guard_receipt,
)


FORMAL_CLASSIFICATION = "formal_pc_golden_exact_step_source"
FORMAL_SNAPSHOT_SEMANTICS = (
    "formal_exact_step_external_lossless_evidence"
)
FORMAL_SAMPLE_PHASE = "frozen_post_replay_update_barrier"
COMPARISON_SCHEMA = "zuma-rl.pc-exact-step-replay-comparison"
COMPARISON_VERSION = 1
EXPECTED_WIDTH = 800
EXPECTED_HEIGHT = 600
NONCLIENT_DRAG_REPAINT_MECHANISM = (
    "nonclient_titlebar_drag_then_exact_origin_restore"
)
SET_WINDOW_POS_REPAINT_MECHANISM = (
    "set_window_pos_temporary_translation_then_exact_origin_restore"
)
_PROCESS_LOCAL_TOPOLOGY_DIGEST = "sha256:" + "0" * 64
_RENDER_FLOAT_FIELDS = (
    "curve_distance",
    "orientation_radians",
    "previous_orientation_radians",
    "angular_step_radians",
    "position_x",
    "position_y",
    "scale",
    "radius",
)


class PcExactStepEvidenceError(ValueError):
    """A formal exact-step source or one of its bound artifacts is invalid."""


@dataclass(frozen=True, slots=True)
class ExactStepFrameEvidence:
    update: int
    path: Path
    bmp_sha256: str
    rgb24_sha256: str
    pc_video_rgb24_sha256: str


@dataclass(frozen=True, slots=True)
class ExactStepLoadedRun:
    run_id: str
    selected_attempt: int
    process_id: int
    process_creation_filetime_100ns: int
    dmo_sha256: str
    runtime_sha256: str
    freeze_update: int
    start_update: int
    end_update: int
    warmup_tick_count: int
    attempts_sha256: str
    probe_sha256: str
    index_sha256: str
    external_input_guard_sha256: str
    campaign_started_perf_counter_ns: int
    attempt_finished_perf_counter_ns: int
    frames: tuple[RngTrajectoryFrame, ...]
    render_states: tuple[tuple[Any, ...], ...]
    visual_frames: tuple[ExactStepFrameEvidence, ...]
    tick_rows: tuple[Mapping[str, Any], ...]


def _fail(code: str) -> None:
    raise PcExactStepEvidenceError(code)


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"exact_step_duplicate_json_key:{key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    _fail(f"exact_step_nonfinite_json_number:{value}")


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")


def read_canonical_json(path: Path) -> Any:
    try:
        payload = path.read_bytes()
        value = json.loads(
            payload.decode("ascii"),
            object_pairs_hook=_pairs,
            parse_constant=_reject_constant,
        )
    except PcExactStepEvidenceError:
        raise
    except (OSError, UnicodeError, ValueError) as error:
        raise PcExactStepEvidenceError(
            "exact_step_json_invalid"
        ) from error
    try:
        canonical = canonical_json_bytes(value)
    except (TypeError, ValueError) as error:
        raise PcExactStepEvidenceError(
            "exact_step_json_invalid"
        ) from error
    if payload != canonical:
        _fail("exact_step_json_not_canonical")
    return value


def _mapping(value: Any, code: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        _fail(code)
    return value


def _integer(
    value: Any,
    code: str,
    *,
    minimum: int = 0,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(code)
    return value


def _digest(value: Any, code: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 71
        or not value.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        _fail(code)
    return value


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise PcExactStepEvidenceError(
            "exact_step_artifact_unreadable"
        ) from error
    return f"sha256:{digest.hexdigest()}"


def _artifact_path(root: Path, member: Any) -> Path:
    if not isinstance(member, str) or not member or "\\" in member or ":" in member:
        _fail("exact_step_artifact_path_invalid")
    pure = PurePosixPath(member)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        _fail("exact_step_artifact_path_invalid")
    path = root.joinpath(*pure.parts)
    try:
        path.resolve().relative_to(root.resolve())
    except (OSError, ValueError) as error:
        raise PcExactStepEvidenceError(
            "exact_step_artifact_path_escape"
        ) from error
    return path


def _validate_bound_artifacts(
    value: Any,
    *,
    root: Path,
    declared_paths: frozenset[Path] | None,
) -> None:
    """Recursively rehash every ``artifact`` receipt in a JSON source."""

    if isinstance(value, dict):
        if "artifact" in value and "artifact_sha256" in value:
            path = _artifact_path(root, value["artifact"])
            try:
                size = path.stat().st_size
            except OSError as error:
                raise PcExactStepEvidenceError(
                    "exact_step_bound_artifact_missing"
                ) from error
            if (
                value.get("artifact_bytes", size) != size
                or _digest(
                    value["artifact_sha256"],
                    "exact_step_bound_artifact_digest_invalid",
                )
                != _sha256_path(path)
            ):
                _fail("exact_step_bound_artifact_mismatch")
            if (
                declared_paths is not None
                and path.resolve() not in declared_paths
            ):
                _fail("exact_step_bound_artifact_not_declared")
        for item in value.values():
            _validate_bound_artifacts(
                item,
                root=root,
                declared_paths=declared_paths,
            )
    elif isinstance(value, list):
        for item in value:
            _validate_bound_artifacts(
                item,
                root=root,
                declared_paths=declared_paths,
            )


def decode_top_down_bgra_bmp(path: Path) -> np.ndarray[Any, np.dtype[np.uint8]]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise PcExactStepEvidenceError(
            "exact_step_bmp_unreadable"
        ) from error
    if len(payload) < 54:
        _fail("exact_step_bmp_header_invalid")
    signature, file_size, _, _, pixel_offset = struct.unpack_from(
        "<2sIHHI", payload, 0
    )
    (
        dib_size,
        width,
        height,
        planes,
        bits_per_pixel,
        compression,
        image_bytes,
        _,
        _,
        _,
        _,
    ) = struct.unpack_from("<IiiHHIIiiII", payload, 14)
    expected_bytes = EXPECTED_WIDTH * EXPECTED_HEIGHT * 4
    if (
        signature != b"BM"
        or file_size != len(payload)
        or pixel_offset != 54
        or dib_size != 40
        or width != EXPECTED_WIDTH
        or height != -EXPECTED_HEIGHT
        or planes != 1
        or bits_per_pixel != 32
        or compression != 0
        or image_bytes != expected_bytes
        or len(payload) != pixel_offset + expected_bytes
    ):
        _fail("exact_step_bmp_layout_invalid")
    return np.frombuffer(payload, dtype=np.uint8, offset=pixel_offset).reshape(
        EXPECTED_HEIGHT,
        EXPECTED_WIDTH,
        4,
    )


def _stable_render_identity(value: Any, *, code: str) -> tuple[Any, ...] | None:
    if value is None:
        return None
    row = _mapping(value, code)
    if row.get("kind") not in {"ball", "bullet"}:
        _fail(code)
    ball_id = _integer(row.get("ball_id"), code)
    color_id = _integer(row.get("color_id"), code)
    if ball_id > 0xFFFFFFFF or color_id > 5:
        _fail(code)
    powerups = tuple(
        _integer(row.get(field), code)
        for field in (
            "powerup_previous_type",
            "powerup_primary_type",
            "powerup_secondary_type",
        )
    )
    if any(value > 14 for value in powerups):
        _fail(code)
    floats: list[float] = []
    for field in _RENDER_FLOAT_FIELDS:
        value = row.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            _fail(code)
        floats.append(float(value))
    float_bits = row.get("render_float32_bits_hex")
    flags = row.get("flags_b4_c2_hex")
    if (
        not isinstance(float_bits, str)
        or len(float_bits) != 64
        or struct.pack("<8f", *floats).hex() != float_bits
        or not isinstance(flags, str)
        or len(flags) != 30
    ):
        _fail(code)
    try:
        bytes.fromhex(float_bits)
        bytes.fromhex(flags)
    except ValueError:
        _fail(code)
    return (
        row["kind"],
        ball_id,
        color_id,
        *powerups,
        float_bits,
        flags,
    )


def _stable_render_tick(row: Mapping[str, Any]) -> tuple[Any, ...]:
    curve_lists = row.get("curve_lists")
    if not isinstance(curve_lists, list) or not curve_lists:
        _fail("exact_step_curve_render_state_invalid")
    stable_lists: list[tuple[Any, ...]] = []
    seen: set[tuple[int, int]] = set()
    for value in curve_lists:
        curve = _mapping(value, "exact_step_curve_render_state_invalid")
        curve_index = _integer(
            curve.get("curve_index"),
            "exact_step_curve_render_state_invalid",
        )
        offset = _integer(
            curve.get("container_offset"),
            "exact_step_curve_render_state_invalid",
        )
        key = (curve_index, offset)
        if offset not in {0x50, 0x5C, 0x68} or key in seen:
            _fail("exact_step_curve_render_state_invalid")
        seen.add(key)
        entities = curve.get("entities")
        if not isinstance(entities, list):
            _fail("exact_step_curve_render_state_invalid")
        if (
            curve.get("declared_count") != len(entities)
            or curve.get("traversed_count") != len(entities)
        ):
            _fail("exact_step_curve_render_state_invalid")
        stable_lists.append(
            (
                curve_index,
                offset,
                tuple(
                    _stable_render_identity(
                        entity,
                        code="exact_step_curve_entity_render_state_invalid",
                    )
                    for entity in entities
                ),
            )
        )
    return (
        _stable_render_identity(
            row.get("current_ball"),
            code="exact_step_current_render_state_invalid",
        ),
        _stable_render_identity(
            row.get("next_ball"),
            code="exact_step_next_render_state_invalid",
        ),
        tuple(sorted(stable_lists)),
    )


def _semantic_rng_frame(frame: RngTrajectoryFrame) -> RngTrajectoryFrame:
    return replace(
        frame,
        curve_lists=tuple(
            replace(
                curve_list,
                topology_sha256=_PROCESS_LOCAL_TOPOLOGY_DIGEST,
            )
            for curve_list in frame.curve_lists
        ),
    )


def _validate_repaint(value: Any, *, process_id: int, window: str) -> None:
    repaint = _mapping(value, "exact_step_repaint_invalid")
    times = tuple(
        _integer(repaint.get(field), "exact_step_repaint_invalid", minimum=1)
        for field in (
            "started_perf_counter_ns",
            "moved_perf_counter_ns",
            "held_perf_counter_ns",
            "restored_perf_counter_ns",
            "finished_perf_counter_ns",
        )
    )
    mechanism = repaint.get("mechanism")
    supported_mechanisms = {
        NONCLIENT_DRAG_REPAINT_MECHANISM,
        SET_WINDOW_POS_REPAINT_MECHANISM,
    }
    if (
        repaint.get("schema") != "zuma-rl.pc-window-repaint-handshake"
        or repaint.get("version") != 1
        or mechanism not in supported_mechanisms
        or repaint.get("restoration_mechanism")
        != "set_window_pos_exact_origin"
        or repaint.get("geometry_restored") is not True
        or repaint.get("process_id") != process_id
        or repaint.get("window_handle_hex") != window
        or repaint.get("outer_rect_after") != repaint.get("outer_rect_before")
        or any(left >= right for left, right in zip(times, times[1:]))
    ):
        _fail("exact_step_repaint_invalid")
    if mechanism == SET_WINDOW_POS_REPAINT_MECHANISM:
        before = repaint.get("outer_rect_before")
        moved = repaint.get("outer_rect_moved")
        requested = repaint.get("requested_temporary_delta")
        actual = repaint.get("actual_temporary_delta")
        if (
            not isinstance(before, list)
            or len(before) != 4
            or any(
                isinstance(item, bool) or not isinstance(item, int)
                for item in before
            )
            or not isinstance(moved, list)
            or len(moved) != 4
            or any(
                isinstance(item, bool) or not isinstance(item, int)
                for item in moved
            )
            or not isinstance(requested, list)
            or requested != actual
            or len(requested) != 2
            or isinstance(requested[0], bool)
            or not isinstance(requested[0], int)
            or requested[1] != 0
            or not 1 <= abs(requested[0]) <= 128
            or moved
            != [
                before[0] + requested[0],
                before[1],
                before[2] + requested[0],
                before[3],
            ]
            or repaint.get("input_transport")
            != "none_window_manager_api_only"
            or repaint.get("synthetic_input_event_count") != 0
            or repaint.get("dwm_flush_count") != 2
        ):
            _fail("exact_step_repaint_invalid")


def _validate_visual_snapshot(
    value: Any,
    *,
    index_root: Path,
    process_id: int,
    process_identity: Mapping[str, Any],
    expected_update: int,
    declared_paths: frozenset[Path] | None,
    require_schema: bool,
) -> ExactStepFrameEvidence:
    snapshot = _mapping(value, "exact_step_visual_snapshot_invalid")
    if require_schema and (
        snapshot.get("schema") != "zuma-rl.pc-step-visual-snapshot"
        or snapshot.get("version") != 1
    ):
        _fail("exact_step_visual_snapshot_invalid")
    path = _artifact_path(index_root, snapshot.get("artifact"))
    if declared_paths is not None and path.resolve() not in declared_paths:
        _fail("exact_step_frame_artifact_not_declared")
    try:
        artifact_bytes = path.stat().st_size
    except OSError as error:
        raise PcExactStepEvidenceError(
            "exact_step_frame_artifact_missing"
        ) from error
    digest = _sha256_path(path)
    capture = _mapping(
        snapshot.get("capture"),
        "exact_step_capture_receipt_invalid",
    )
    if (
        snapshot.get("artifact_bytes") != artifact_bytes
        or snapshot.get("artifact_sha256") != digest
        or capture.get("process_id") != process_id
        or capture.get("width") != EXPECTED_WIDTH
        or capture.get("height") != EXPECTED_HEIGHT
        or capture.get("bytes") != artifact_bytes
        or capture.get("semantics") != FORMAL_SNAPSHOT_SEMANTICS
        or capture.get("window_handle_hex")
        != process_identity.get("window_handle_hex")
        or capture.get("client_region")
        != process_identity.get("window_client_region")
    ):
        _fail("exact_step_capture_receipt_invalid")
    _validate_repaint(
        snapshot.get("repaint"),
        process_id=process_id,
        window=str(process_identity["window_handle_hex"]),
    )
    pixels = decode_top_down_bgra_bmp(path)
    rgb = np.ascontiguousarray(pixels[:, :, (2, 1, 0)])
    rgb_sha256 = (
        "sha256:" + hashlib.sha256(rgb.tobytes(order="C")).hexdigest()
    )
    return ExactStepFrameEvidence(
        update=expected_update,
        path=path.resolve(),
        bmp_sha256=digest,
        rgb24_sha256=rgb_sha256,
        pc_video_rgb24_sha256=canonical_rgb24_frame_sha256(rgb),
    )


def load_formal_exact_step_run(
    *,
    run_id: str,
    selected_attempt: int,
    attempts_path: Path,
    probe_path: Path,
    index_path: Path,
    expected_process_id: int,
    expected_process_creation_filetime_100ns: int,
    expected_runtime_sha256: str,
    expected_dmo_sha256: str,
    expected_freeze_update: int,
    expected_start_update: int,
    expected_end_update: int,
    expected_warmup_tick_count: int,
    maximum_startup_attempts: int,
    declared_artifact_paths: Iterable[Path] | None = None,
) -> ExactStepLoadedRun:
    """Load and independently revalidate one formal source run."""

    declared = (
        frozenset(path.resolve() for path in declared_artifact_paths)
        if declared_artifact_paths is not None
        else None
    )
    attempts_path = attempts_path.resolve()
    probe_path = probe_path.resolve()
    index_path = index_path.resolve()
    attempts = read_canonical_json(attempts_path)
    if (
        not isinstance(attempts, list)
        or not 1 <= len(attempts) <= maximum_startup_attempts
        or selected_attempt != len(attempts)
    ):
        _fail("exact_step_attempt_selection_invalid")
    previous_finish = 0
    for expected_attempt, value in enumerate(attempts, start=1):
        row = _mapping(value, "exact_step_attempt_receipt_invalid")
        started = _integer(
            row.get("started_perf_counter_ns"),
            "exact_step_attempt_receipt_invalid",
            minimum=1,
        )
        finished = _integer(
            row.get("finished_perf_counter_ns"),
            "exact_step_attempt_receipt_invalid",
            minimum=1,
        )
        expected_status = (
            "PASS" if expected_attempt == selected_attempt else "RETRY"
        )
        if (
            row.get("attempt") != expected_attempt
            or row.get("status") != expected_status
            or not previous_finish <= started < finished
        ):
            _fail("exact_step_attempt_receipt_invalid")
        previous_finish = finished
    selected = attempts[-1]
    probe_sha256 = _sha256_path(probe_path)
    if (
        selected.get("process_id") != expected_process_id
        or selected.get("process_creation_filetime_100ns")
        != expected_process_creation_filetime_100ns
        or selected.get("probe_sha256") != probe_sha256
    ):
        _fail("exact_step_selected_attempt_binding_invalid")

    probe = _mapping(
        read_canonical_json(probe_path),
        "exact_step_probe_invalid",
    )
    process_identity = _mapping(
        probe.get("process_identity"),
        "exact_step_process_identity_invalid",
    )
    guard_binding = _mapping(
        probe.get("external_input_guard"),
        "exact_step_external_input_guard_binding_invalid",
    )
    guard_required_fields = {
        "schema",
        "version",
        "status",
        "artifact",
        "artifact_bytes",
        "artifact_sha256",
        "coverage_start_perf_counter_ns",
        "coverage_end_perf_counter_ns",
        "external_event_count",
        "allowed_repaint_event_count",
    }
    if set(guard_binding) != guard_required_fields:
        _fail("exact_step_external_input_guard_binding_invalid")
    guard_path = _artifact_path(
        probe_path.parent,
        guard_binding.get("artifact"),
    ).resolve()
    if declared is not None and guard_path not in declared:
        _fail("exact_step_external_input_guard_not_declared")
    try:
        guard_bytes = guard_path.stat().st_size
    except OSError as error:
        raise PcExactStepEvidenceError(
            "exact_step_external_input_guard_missing"
        ) from error
    guard_sha256 = _sha256_path(guard_path)
    guard_receipt = read_canonical_json(guard_path)
    try:
        validate_external_input_guard_receipt(guard_receipt)
    except PcExternalInputError as error:
        raise PcExactStepEvidenceError(
            "exact_step_external_input_guard_invalid"
        ) from error
    guard_coverage = _mapping(
        guard_receipt.get("coverage"),
        "exact_step_external_input_guard_invalid",
    )
    guard_counts = _mapping(
        guard_receipt.get("event_counts"),
        "exact_step_external_input_guard_invalid",
    )
    guard_start = guard_coverage.get("start_perf_counter_ns")
    guard_end = guard_coverage.get("end_perf_counter_ns")
    if (
        guard_binding.get("schema")
        != EXTERNAL_INPUT_GUARD_BINDING_SCHEMA
        or guard_binding.get("version")
        != EXTERNAL_INPUT_GUARD_BINDING_VERSION
        or guard_binding.get("status") != "PASS"
        or guard_binding.get("artifact_bytes") != guard_bytes
        or guard_binding.get("artifact_sha256") != guard_sha256
        or guard_binding.get("coverage_start_perf_counter_ns")
        != guard_start
        or guard_binding.get("coverage_end_perf_counter_ns") != guard_end
        or guard_binding.get("external_event_count") != 0
        or guard_binding.get("external_event_count")
        != guard_counts.get("external")
        or guard_binding.get("allowed_repaint_event_count")
        != guard_counts.get("allowed_repaint")
        or not isinstance(guard_start, int)
        or isinstance(guard_start, bool)
        or not isinstance(guard_end, int)
        or isinstance(guard_end, bool)
        or not started <= guard_start < guard_end <= finished
        or selected.get("external_input_guard_sha256") != guard_sha256
        or selected.get(
            "external_input_guard_coverage_start_perf_counter_ns"
        )
        != guard_start
        or selected.get(
            "external_input_guard_coverage_end_perf_counter_ns"
        )
        != guard_end
    ):
        _fail("exact_step_external_input_guard_binding_invalid")
    if (
        probe.get("schema") != "zuma-rl.pc-memory-int32-probe"
        or probe.get("version") != 1
        or probe.get("evidence_classification") != FORMAL_CLASSIFICATION
        or probe.get("process_id") != expected_process_id
        or process_identity.get("process_id") != expected_process_id
        or process_identity.get("process_creation_filetime_100ns")
        != expected_process_creation_filetime_100ns
        or process_identity.get("executable_sha256")
        != expected_runtime_sha256
        or probe.get("runtime_executable_sha256")
        != expected_runtime_sha256
        or probe.get("dmo_sha256") != expected_dmo_sha256
        or probe.get("diagnostic_mutation") is not None
        or probe.get("diagnostic_observation") is not None
        or probe.get("diagnostic_process_affinity") is not None
        or probe.get("global_rng_call_trace") is not None
        or probe.get("live_rng_monitor") is not None
        or probe.get("gameplay_mtrand_sync") is not None
        or _mapping(probe.get("repaint"), "exact_step_probe_invalid").get(
            "status"
        )
        != "SUPERSEDED_BY_PER_TICK_REPAINT_HANDSHAKE"
        or _mapping(
            probe.get("frozen_frame"), "exact_step_probe_invalid"
        ).get("status")
        != "SUPERSEDED_BY_RETAINED_EXACT_STEP_FRAMES"
    ):
        _fail("exact_step_probe_invalid")
    trajectory = _mapping(
        probe.get("trajectory"),
        "exact_step_probe_trajectory_binding_invalid",
    )
    if (
        trajectory.get("artifact_sha256") != _sha256_path(index_path)
        or trajectory.get("freeze_update") != expected_freeze_update
        or trajectory.get("start_update") != expected_start_update
        or trajectory.get("end_update") != expected_end_update
        or trajectory.get("warmup_tick_count") != expected_warmup_tick_count
        or trajectory.get("tick_count")
        != expected_end_update - expected_start_update + 1
    ):
        _fail("exact_step_probe_trajectory_binding_invalid")
    expected_index_from_probe = _artifact_path(
        probe_path.parent,
        trajectory.get("artifact"),
    ).resolve()
    if expected_index_from_probe != index_path:
        _fail("exact_step_probe_trajectory_binding_invalid")
    _validate_bound_artifacts(
        probe,
        root=probe_path.parent,
        declared_paths=declared,
    )

    index = _mapping(
        read_canonical_json(index_path),
        "exact_step_index_invalid",
    )
    index_identity = _mapping(
        index.get("process_identity"),
        "exact_step_index_process_identity_invalid",
    )
    visual = _mapping(
        index.get("visual_snapshots"),
        "exact_step_visual_contract_invalid",
    )
    corner = _mapping(
        _mapping(
            index.get("window_transport"),
            "exact_step_window_transport_invalid",
        ).get("square_dwm_corners"),
        "exact_step_window_transport_invalid",
    )
    tick_count = expected_end_update - expected_start_update + 1
    if (
        index.get("schema") != "zuma-rl.pc-rng-trajectory"
        or index.get("version") != 2
        or index.get("evidence_classification") != FORMAL_CLASSIFICATION
        or index.get("sample_phase") != FORMAL_SAMPLE_PHASE
        or index.get("freeze_update") != expected_freeze_update
        or index.get("start_update") != expected_start_update
        or index.get("end_update") != expected_end_update
        or index.get("warmup_tick_count") != expected_warmup_tick_count
        or index.get("tick_count") != tick_count
        or index_identity != process_identity
        or visual.get("enabled") is not True
        or visual.get("frame_count") != tick_count
        or visual.get("sample_phase") != FORMAL_SAMPLE_PHASE
        or visual.get("semantics") != FORMAL_SNAPSHOT_SEMANTICS
        or corner.get("schema") != "zuma-rl.pc-dwm-corner-preference"
        or corner.get("version") != 1
        or corner.get("attribute") != "DWMWA_WINDOW_CORNER_PREFERENCE"
        or corner.get("requested") != "DWMWCP_DONOTROUND"
        or corner.get("observed_value") != 1
        or corner.get("status") != "verified"
        or corner.get("window_handle_hex")
        != process_identity.get("window_handle_hex")
    ):
        _fail("exact_step_index_contract_invalid")
    _validate_bound_artifacts(
        index,
        root=index_path.parent,
        declared_paths=declared,
    )
    frames = load_rng_trajectory(index_path)
    rows_value = index.get("ticks")
    if not isinstance(rows_value, list) or len(rows_value) != tick_count:
        _fail("exact_step_tick_rows_invalid")
    rows = tuple(
        _mapping(value, "exact_step_tick_row_invalid")
        for value in rows_value
    )
    warmup = _mapping(
        visual.get("warmup_frame"),
        "exact_step_warmup_invalid",
    )
    if (
        warmup.get("framework_update") != expected_freeze_update
        or warmup.get("acceptance_role")
        != "discarded_dxgi_surface_warmup"
    ):
        _fail("exact_step_warmup_invalid")
    _validate_visual_snapshot(
        warmup,
        index_root=index_path.parent,
        process_id=expected_process_id,
        process_identity=process_identity,
        expected_update=expected_freeze_update,
        declared_paths=declared,
        require_schema=False,
    )
    visual_frames: list[ExactStepFrameEvidence] = []
    render_states: list[tuple[Any, ...]] = []
    for offset, row in enumerate(rows):
        update = expected_start_update + offset
        if row.get("framework_update") != update:
            _fail("exact_step_tick_row_invalid")
        visual_frames.append(
            _validate_visual_snapshot(
                row.get("visual_snapshot"),
                index_root=index_path.parent,
                process_id=expected_process_id,
                process_identity=process_identity,
                expected_update=update,
                declared_paths=declared,
                require_schema=True,
            )
        )
        render_states.append(_stable_render_tick(row))
    guarded_snapshots = (warmup,) + tuple(
        _mapping(
            row.get("visual_snapshot"),
            "exact_step_visual_snapshot_invalid",
        )
        for row in rows
    )
    repaint_mechanisms: set[str] = set()
    for snapshot in guarded_snapshots:
        repaint = _mapping(
            snapshot.get("repaint"),
            "exact_step_repaint_invalid",
        )
        mechanism = repaint.get("mechanism")
        if not isinstance(mechanism, str):
            _fail("exact_step_repaint_invalid")
        repaint_mechanisms.add(mechanism)
        repaint_start = _integer(
            repaint.get("started_perf_counter_ns"),
            "exact_step_repaint_invalid",
            minimum=1,
        )
        repaint_end = _integer(
            repaint.get("finished_perf_counter_ns"),
            "exact_step_repaint_invalid",
            minimum=1,
        )
        if not guard_start <= repaint_start < repaint_end <= guard_end:
            _fail("exact_step_external_input_guard_coverage_invalid")
    if len(repaint_mechanisms) != 1:
        _fail("exact_step_repaint_mechanism_mixed")
    repaint_mechanism = next(iter(repaint_mechanisms))
    if visual.get("repaint_handshake") != f"{repaint_mechanism}_per_frame":
        _fail("exact_step_visual_repaint_contract_invalid")
    allowed_repaint_count = guard_counts.get("allowed_repaint")
    if (
        repaint_mechanism == SET_WINDOW_POS_REPAINT_MECHANISM
        and allowed_repaint_count != 0
    ) or (
        repaint_mechanism == NONCLIENT_DRAG_REPAINT_MECHANISM
        and (
            isinstance(allowed_repaint_count, bool)
            or not isinstance(allowed_repaint_count, int)
            or allowed_repaint_count <= 0
        )
    ):
        _fail("exact_step_repaint_input_count_invalid")
    return ExactStepLoadedRun(
        run_id=run_id,
        selected_attempt=selected_attempt,
        process_id=expected_process_id,
        process_creation_filetime_100ns=(
            expected_process_creation_filetime_100ns
        ),
        dmo_sha256=expected_dmo_sha256,
        runtime_sha256=expected_runtime_sha256,
        freeze_update=expected_freeze_update,
        start_update=expected_start_update,
        end_update=expected_end_update,
        warmup_tick_count=expected_warmup_tick_count,
        attempts_sha256=_sha256_path(attempts_path),
        probe_sha256=probe_sha256,
        index_sha256=_sha256_path(index_path),
        external_input_guard_sha256=guard_sha256,
        campaign_started_perf_counter_ns=_integer(
            attempts[0].get("started_perf_counter_ns"),
            "exact_step_attempt_receipt_invalid",
            minimum=1,
        ),
        attempt_finished_perf_counter_ns=_integer(
            selected.get("finished_perf_counter_ns"),
            "exact_step_attempt_receipt_invalid",
            minimum=1,
        ),
        frames=frames,
        render_states=tuple(render_states),
        visual_frames=tuple(visual_frames),
        tick_rows=rows,
    )


def compare_formal_exact_step_runs(
    runs: Iterable[ExactStepLoadedRun],
) -> Mapping[str, Any]:
    """Return the canonical deterministic comparison for formal runs."""

    run_rows = tuple(runs)
    if len(run_rows) < 2:
        _fail("exact_step_comparison_requires_two_runs")
    reference = run_rows[0]
    if len({run.run_id for run in run_rows}) != len(run_rows):
        _fail("exact_step_comparison_run_id_duplicate")
    independent = len(
        {
            (run.process_id, run.process_creation_filetime_100ns)
            for run in run_rows
        }
    ) == len(run_rows)
    same_protocol = all(
        (
            run.dmo_sha256,
            run.runtime_sha256,
            run.freeze_update,
            run.start_update,
            run.end_update,
            run.warmup_tick_count,
            len(run.frames),
        )
        == (
            reference.dmo_sha256,
            reference.runtime_sha256,
            reference.freeze_update,
            reference.start_update,
            reference.end_update,
            reference.warmup_tick_count,
            len(reference.frames),
        )
        for run in run_rows[1:]
    )
    reference_memory = tuple(
        _semantic_rng_frame(frame) for frame in reference.frames
    )
    memory_exact = all(
        tuple(_semantic_rng_frame(frame) for frame in run.frames)
        == reference_memory
        for run in run_rows[1:]
    )
    render_exact = all(
        run.render_states == reference.render_states
        for run in run_rows[1:]
    )
    reference_bmp = tuple(
        frame.bmp_sha256 for frame in reference.visual_frames
    )
    reference_rgb = tuple(
        frame.rgb24_sha256 for frame in reference.visual_frames
    )
    bgra_exact = all(
        tuple(frame.bmp_sha256 for frame in run.visual_frames)
        == reference_bmp
        for run in run_rows[1:]
    )
    rgb_exact = all(
        tuple(frame.rgb24_sha256 for frame in run.visual_frames)
        == reference_rgb
        for run in run_rows[1:]
    )
    status = (
        "PASS"
        if independent
        and same_protocol
        and memory_exact
        and render_exact
        and bgra_exact
        and rgb_exact
        else "FAIL"
    )
    return {
        "schema": COMPARISON_SCHEMA,
        "version": COMPARISON_VERSION,
        "status": status,
        "protocol": {
            "same_protocol": same_protocol,
            "freeze_update": reference.freeze_update,
            "start_update": reference.start_update,
            "end_update": reference.end_update,
            "warmup_tick_count": reference.warmup_tick_count,
            "retained_tick_count": len(reference.frames),
            "sample_phase": FORMAL_SAMPLE_PHASE,
        },
        "independence": {
            "independent_process_instances": independent,
            "runs": [
                {
                    "run_id": run.run_id,
                    "selected_attempt": run.selected_attempt,
                    "process_id": run.process_id,
                    "process_creation_filetime_100ns": (
                        run.process_creation_filetime_100ns
                    ),
                    "attempts_sha256": run.attempts_sha256,
                    "memory_probe_sha256": run.probe_sha256,
                    "trajectory_index_sha256": run.index_sha256,
                    "external_input_guard_sha256": (
                        run.external_input_guard_sha256
                    ),
                }
                for run in run_rows
            ],
        },
        "memory": {
            "normalized_gameplay_and_rng_exact": memory_exact,
            "excluded_process_local_fields": [
                "curve_lists[].topology_sha256"
            ],
        },
        "render_state": {
            "exact": render_exact,
            "comparison": (
                "exact_float32_bits_flags_identity_colour_powerups_and_"
                "curve_list_order"
            ),
        },
        "pixels": {
            "full_800x600_bgra_bmp_exact": bgra_exact,
            "full_800x600_rgb24_exact": rgb_exact,
            "per_tick_bmp_sha256": list(reference_bmp),
            "per_tick_rgb24_sha256": list(reference_rgb),
        },
    }


__all__ = [
    "COMPARISON_SCHEMA",
    "COMPARISON_VERSION",
    "EXPECTED_HEIGHT",
    "EXPECTED_WIDTH",
    "FORMAL_CLASSIFICATION",
    "FORMAL_SAMPLE_PHASE",
    "FORMAL_SNAPSHOT_SEMANTICS",
    "ExactStepFrameEvidence",
    "ExactStepLoadedRun",
    "PcExactStepEvidenceError",
    "canonical_json_bytes",
    "compare_formal_exact_step_runs",
    "decode_top_down_bgra_bmp",
    "load_formal_exact_step_run",
    "read_canonical_json",
]
