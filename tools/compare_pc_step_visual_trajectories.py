"""Compare two exact-step retail RNG trajectories and their RGB snapshots.

The result is a diagnostic receipt, not PC Golden evidence.  Both inputs are
fully revalidated, bound to their memory probes, required to come from
different retail processes, and compared at every retained post-update tick.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import struct
import sys
from typing import Any, Iterable, Mapping

import numpy as np

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from zuma_rl.pc_rng_trajectory import (
    RngTrajectoryFrame,
    load_rng_trajectory,
)


SCHEMA = "zuma-rl.pc-step-visual-trajectory-comparison"
VERSION = 2
EXPECTED_WIDTH = 800
EXPECTED_HEIGHT = 600
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


class StepVisualComparisonError(ValueError):
    """An input receipt, trajectory, or frame artifact is invalid."""


@dataclass(frozen=True, slots=True)
class Run:
    index_path: Path
    index_sha256: str
    probe_path: Path
    probe_sha256: str
    process_id: int
    dmo_sha256: str
    runtime_sha256: str
    freeze_update: int
    start_update: int
    end_update: int
    warmup_tick_count: int
    frames: tuple[RngTrajectoryFrame, ...]
    render_states: tuple[tuple[Any, ...], ...]
    snapshots: tuple[Mapping[str, Any], ...]


def _fail(code: str) -> None:
    raise StepVisualComparisonError(code)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise StepVisualComparisonError(
            "step_visual_artifact_unreadable"
        ) from error
    return f"sha256:{digest.hexdigest()}"


def _json_object(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_bytes().decode("ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise StepVisualComparisonError(
            "step_visual_json_invalid"
        ) from error
    if not isinstance(value, dict):
        _fail("step_visual_json_root_invalid")
    return value


def _integer(value: Any, code: str, *, minimum: int = 0) -> int:
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


def _artifact(root: Path, member: Any) -> Path:
    if not isinstance(member, str) or not member or "\\" in member or ":" in member:
        _fail("step_visual_artifact_path_invalid")
    pure = PurePosixPath(member)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        _fail("step_visual_artifact_path_invalid")
    path = root.joinpath(*pure.parts)
    try:
        path.resolve().relative_to(root.resolve())
    except (OSError, ValueError) as error:
        raise StepVisualComparisonError(
            "step_visual_artifact_path_escape"
        ) from error
    return path


def _hex_bytes(value: Any, *, characters: int, code: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != characters
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(code)
    return value


def _stable_render_identity(
    value: Any,
    *,
    code: str,
) -> tuple[Any, ...] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        _fail(code)
    kind = value.get("kind")
    if kind not in {"ball", "bullet"}:
        _fail(code)
    ball_id = _integer(value.get("ball_id"), code)
    color_id = _integer(value.get("color_id"), code)
    if ball_id > 0xFFFFFFFF or color_id > 5:
        _fail(code)
    powerups: list[int] = []
    for field in (
        "powerup_previous_type",
        "powerup_primary_type",
        "powerup_secondary_type",
    ):
        powerup = _integer(value.get(field), code)
        if powerup > 14:
            _fail(code)
        powerups.append(powerup)
    floats: list[float] = []
    for field in _RENDER_FLOAT_FIELDS:
        number = value.get(field)
        if not isinstance(number, float) or not math.isfinite(number):
            _fail(code)
        floats.append(number)
    float_bits = _hex_bytes(
        value.get("render_float32_bits_hex"),
        characters=64,
        code=code,
    )
    if struct.pack("<8f", *floats).hex() != float_bits:
        _fail(code)
    flags = _hex_bytes(
        value.get("flags_b4_c2_hex"),
        characters=30,
        code=code,
    )
    return (
        kind,
        ball_id,
        color_id,
        *powerups,
        float_bits,
        flags,
    )


def _stable_render_tick(row: Mapping[str, Any]) -> tuple[Any, ...]:
    current = _stable_render_identity(
        row.get("current_ball"),
        code="step_visual_current_render_state_invalid",
    )
    following = _stable_render_identity(
        row.get("next_ball"),
        code="step_visual_next_render_state_invalid",
    )
    curve_lists = row.get("curve_lists")
    if not isinstance(curve_lists, list) or not curve_lists:
        _fail("step_visual_curve_render_state_invalid")
    stable_lists: list[tuple[Any, ...]] = []
    seen: set[tuple[int, int]] = set()
    for curve_list in curve_lists:
        if not isinstance(curve_list, dict):
            _fail("step_visual_curve_render_state_invalid")
        curve_index = _integer(
            curve_list.get("curve_index"),
            "step_visual_curve_render_state_invalid",
        )
        container_offset = _integer(
            curve_list.get("container_offset"),
            "step_visual_curve_render_state_invalid",
        )
        key = (curve_index, container_offset)
        if container_offset not in {0x50, 0x5C, 0x68} or key in seen:
            _fail("step_visual_curve_render_state_invalid")
        seen.add(key)
        entities = curve_list.get("entities")
        if not isinstance(entities, list):
            _fail("step_visual_curve_render_state_invalid")
        declared = _integer(
            curve_list.get("declared_count"),
            "step_visual_curve_render_state_invalid",
        )
        traversed = _integer(
            curve_list.get("traversed_count"),
            "step_visual_curve_render_state_invalid",
        )
        if declared != len(entities) or traversed != len(entities):
            _fail("step_visual_curve_render_state_invalid")
        stable_lists.append(
            (
                curve_index,
                container_offset,
                tuple(
                    _stable_render_identity(
                        entity,
                        code="step_visual_curve_entity_render_state_invalid",
                    )
                    for entity in entities
                ),
            )
        )
    return current, following, tuple(sorted(stable_lists))


def _load_run(index_path: Path) -> Run:
    index_path = index_path.resolve()
    frames = load_rng_trajectory(index_path)
    index = _json_object(index_path)
    start = _integer(index.get("start_update"), "step_visual_start_invalid")
    end = _integer(index.get("end_update"), "step_visual_end_invalid")
    freeze = _integer(
        index.get("freeze_update", start),
        "step_visual_freeze_invalid",
    )
    warmup = _integer(
        index.get("warmup_tick_count", start - freeze),
        "step_visual_warmup_invalid",
    )
    if warmup != start - freeze or len(frames) != end - start + 1:
        _fail("step_visual_range_invalid")
    render_contract = index.get("entity_render_state")
    if (
        not isinstance(render_contract, dict)
        or render_contract.get("schema")
        != "zuma-rl.pc-light-ball-render-state"
        or render_contract.get("version") != 1
        or render_contract.get("scope")
        != "shooter_current_next_and_all_curve_intrusive_list_entities"
        or render_contract.get("float_fields") != list(_RENDER_FLOAT_FIELDS)
        or render_contract.get("float_binding")
        != "exact_little_endian_float32_bits"
        or render_contract.get("flag_binding")
        != "exact_ball_base_bytes_b4_through_c2"
    ):
        _fail("step_visual_render_state_contract_invalid")
    visual = index.get("visual_snapshots")
    if (
        not isinstance(visual, dict)
        or visual.get("enabled") is not True
        or visual.get("frame_count") != len(frames)
        or visual.get("semantics")
        != "diagnostic_single_frame_not_formal_evidence"
    ):
        _fail("step_visual_snapshot_contract_invalid")

    probe_path = index_path.parent.parent / "memory-probe.json"
    probe = _json_object(probe_path)
    trajectory = probe.get("trajectory")
    if (
        probe.get("schema") != "zuma-rl.pc-memory-int32-probe"
        or probe.get("version") != 1
        or not isinstance(trajectory, dict)
        or trajectory.get("artifact") != "trajectory/index.json"
        or trajectory.get("artifact_sha256") != _sha256_path(index_path)
        or trajectory.get("freeze_update") != freeze
        or trajectory.get("start_update") != start
        or trajectory.get("end_update") != end
        or trajectory.get("warmup_tick_count") != warmup
        or trajectory.get("tick_count") != len(frames)
        or trajectory.get("visual_snapshot_count") != len(frames)
    ):
        _fail("step_visual_probe_binding_invalid")
    process_id = _integer(
        probe.get("process_id"),
        "step_visual_process_id_invalid",
        minimum=1,
    )
    dmo_sha256 = _digest(
        probe.get("dmo_sha256"),
        "step_visual_dmo_sha256_invalid",
    )
    runtime_sha256 = _digest(
        probe.get("runtime_executable_sha256"),
        "step_visual_runtime_sha256_invalid",
    )

    rows = index.get("ticks")
    if not isinstance(rows, list) or len(rows) != len(frames):
        _fail("step_visual_tick_rows_invalid")
    snapshots: list[Mapping[str, Any]] = []
    render_states: list[tuple[Any, ...]] = []
    for offset, row in enumerate(rows):
        if not isinstance(row, dict):
            _fail("step_visual_tick_invalid")
        snapshot = row.get("visual_snapshot")
        if (
            not isinstance(snapshot, dict)
            or snapshot.get("schema") != "zuma-rl.pc-step-visual-snapshot"
            or snapshot.get("version") != 1
        ):
            _fail("step_visual_tick_snapshot_invalid")
        artifact = _artifact(index_path.parent, snapshot.get("artifact"))
        if (
            snapshot.get("artifact_bytes") != artifact.stat().st_size
            or snapshot.get("artifact_sha256") != _sha256_path(artifact)
        ):
            _fail("step_visual_frame_artifact_mismatch")
        capture = snapshot.get("capture")
        repaint = snapshot.get("repaint")
        if (
            not isinstance(capture, dict)
            or capture.get("process_id") != process_id
            or capture.get("width") != EXPECTED_WIDTH
            or capture.get("height") != EXPECTED_HEIGHT
            or capture.get("semantics")
            != "diagnostic_single_frame_not_formal_evidence"
            or not isinstance(repaint, dict)
            or repaint.get("geometry_restored") is not True
            or frames[offset].update != start + offset
        ):
            _fail("step_visual_capture_receipt_invalid")
        snapshots.append({**snapshot, "resolved_artifact": str(artifact)})
        render_states.append(_stable_render_tick(row))

    return Run(
        index_path=index_path,
        index_sha256=_sha256_path(index_path),
        probe_path=probe_path.resolve(),
        probe_sha256=_sha256_path(probe_path),
        process_id=process_id,
        dmo_sha256=dmo_sha256,
        runtime_sha256=runtime_sha256,
        freeze_update=freeze,
        start_update=start,
        end_update=end,
        warmup_tick_count=warmup,
        frames=frames,
        render_states=tuple(render_states),
        snapshots=tuple(snapshots),
    )


def _decode_bgra_bmp(path: Path) -> np.ndarray[Any, np.dtype[np.uint8]]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise StepVisualComparisonError(
            "step_visual_bmp_unreadable"
        ) from error
    if len(payload) < 54:
        _fail("step_visual_bmp_header_invalid")
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
        _fail("step_visual_bmp_layout_invalid")
    return np.frombuffer(payload, dtype=np.uint8, offset=pixel_offset).reshape(
        EXPECTED_HEIGHT,
        EXPECTED_WIDTH,
        4,
    )


def _semantic_rng_frame(frame: RngTrajectoryFrame) -> RngTrajectoryFrame:
    """Remove only process-address-dependent curve provenance.

    ``topology_sha256`` hashes live list-node, payload, and vtable addresses in
    addition to stable ball identity fields.  Those addresses are useful
    within one process, but cannot be an independent-process gameplay equality
    requirement.  Every loaded gameplay and RNG field is otherwise retained.
    """

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


def _process_local_topology_signature(
    frame: RngTrajectoryFrame,
) -> tuple[tuple[int, int, str], ...]:
    return tuple(
        (
            curve_list.curve_index,
            curve_list.container_offset,
            curve_list.topology_sha256,
        )
        for curve_list in frame.curve_lists
    )


def compare(left_index: Path, right_index: Path) -> Mapping[str, Any]:
    left = _load_run(left_index)
    right = _load_run(right_index)
    same_protocol = (
        left.dmo_sha256 == right.dmo_sha256
        and left.runtime_sha256 == right.runtime_sha256
        and left.freeze_update == right.freeze_update
        and left.start_update == right.start_update
        and left.end_update == right.end_update
        and left.warmup_tick_count == right.warmup_tick_count
        and len(left.frames) == len(right.frames)
    )
    independent_processes = left.process_id != right.process_id
    left_semantic_frames = tuple(
        _semantic_rng_frame(frame) for frame in left.frames
    )
    right_semantic_frames = tuple(
        _semantic_rng_frame(frame) for frame in right.frames
    )
    memory_exact = left_semantic_frames == right_semantic_frames
    first_memory_mismatch = next(
        (
            left_frame.update
            for left_frame, right_frame in zip(
                left_semantic_frames,
                right_semantic_frames,
                strict=True,
            )
            if left_frame != right_frame
        ),
        None,
    )
    render_state_exact = left.render_states == right.render_states
    first_render_state_mismatch = next(
        (
            left_frame.update
            for left_frame, left_state, right_state in zip(
                left.frames,
                left.render_states,
                right.render_states,
                strict=True,
            )
            if left_state != right_state
        ),
        None,
    )
    topology_exact = all(
        _process_local_topology_signature(left_frame)
        == _process_local_topology_signature(right_frame)
        for left_frame, right_frame in zip(
            left.frames,
            right.frames,
            strict=True,
        )
    )
    first_topology_mismatch = next(
        (
            left_frame.update
            for left_frame, right_frame in zip(
                left.frames,
                right.frames,
                strict=True,
            )
            if _process_local_topology_signature(left_frame)
            != _process_local_topology_signature(right_frame)
        ),
        None,
    )

    pixel_rows: list[dict[str, Any]] = []
    if same_protocol:
        for left_frame, left_snapshot, right_snapshot in zip(
            left.frames,
            left.snapshots,
            right.snapshots,
            strict=True,
        ):
            left_pixels = _decode_bgra_bmp(
                Path(str(left_snapshot["resolved_artifact"]))
            )
            right_pixels = _decode_bgra_bmp(
                Path(str(right_snapshot["resolved_artifact"]))
            )
            # Desktop Duplication is BGRA; full viewport RGB excludes only
            # the transport alpha byte and compares every visible channel.
            mismatch = np.any(
                left_pixels[:, :, :3] != right_pixels[:, :, :3],
                axis=2,
            )
            mismatch_count = int(np.count_nonzero(mismatch))
            if mismatch_count:
                coordinates = np.argwhere(mismatch)
                difference_bounds = [
                    int(coordinates[:, 1].min()),
                    int(coordinates[:, 0].min()),
                    int(coordinates[:, 1].max()) + 1,
                    int(coordinates[:, 0].max()) + 1,
                ]
            else:
                difference_bounds = None
            delta = np.abs(
                left_pixels[:, :, :3].astype(np.int16)
                - right_pixels[:, :, :3].astype(np.int16)
            )
            pixel_rows.append(
                {
                    "update": left_frame.update,
                    "left_artifact_sha256": left_snapshot[
                        "artifact_sha256"
                    ],
                    "right_artifact_sha256": right_snapshot[
                        "artifact_sha256"
                    ],
                    "mismatched_rgb_pixel_count": mismatch_count,
                    "full_viewport_rgb_matched": mismatch_count == 0,
                    "difference_bounds_xyxy": difference_bounds,
                    "maximum_absolute_delta_by_bgr_channel": [
                        int(value)
                        for value in np.max(delta, axis=(0, 1)).tolist()
                    ],
                }
            )
    matched_ticks = sum(
        row["full_viewport_rgb_matched"] for row in pixel_rows
    )
    maximum_mismatch = max(
        (row["mismatched_rgb_pixel_count"] for row in pixel_rows),
        default=None,
    )
    full_pixels_exact = (
        same_protocol
        and len(pixel_rows) == len(left.frames)
        and matched_ticks == len(pixel_rows)
    )
    status = (
        "PASS"
        if same_protocol
        and independent_processes
        and memory_exact
        and render_state_exact
        and full_pixels_exact
        else "FAIL"
    )

    def input_row(run: Run) -> dict[str, Any]:
        return {
            "index": str(run.index_path),
            "index_sha256": run.index_sha256,
            "memory_probe": str(run.probe_path),
            "memory_probe_sha256": run.probe_sha256,
            "process_id": run.process_id,
            "dmo_sha256": run.dmo_sha256,
            "runtime_executable_sha256": run.runtime_sha256,
        }

    return {
        "schema": SCHEMA,
        "version": VERSION,
        "classification": "diagnostic-not-pc-golden",
        "status": status,
        "inputs": {"left": input_row(left), "right": input_row(right)},
        "protocol": {
            "same_protocol": same_protocol,
            "freeze_update": left.freeze_update,
            "start_update": left.start_update,
            "end_update": left.end_update,
            "warmup_tick_count": left.warmup_tick_count,
            "retained_tick_count": len(left.frames),
            "sample_phase": "post_update_safe_delete_complete",
        },
        "independence": {
            "distinct_process_ids": independent_processes,
            "left_process_id": left.process_id,
            "right_process_id": right.process_id,
        },
        "memory": {
            "normalized_rng_trajectory_exact": memory_exact,
            "first_mismatch_update": first_memory_mismatch,
            "comparison": (
                "all_loaded_gameplay_and_rng_fields_except_"
                "process_local_curve_topology_sha256"
            ),
            "excluded_process_local_fields": [
                "curve_lists[].topology_sha256"
            ],
            "process_local_topology_digests_exact": topology_exact,
            "first_process_local_topology_mismatch_update": (
                first_topology_mismatch
            ),
        },
        "render_state": {
            "exact": render_state_exact,
            "first_mismatch_update": first_render_state_mismatch,
            "comparison": (
                "exact_float32_bits_for_curve_distance_orientation_"
                "position_scale_radius_plus_ball_flags_identity_colour_"
                "powerups_and_list_order"
            ),
            "excluded_process_local_fields": [
                "address",
                "address_hex",
                "vtable",
                "curve_lists[].sentinel_address",
                "curve_lists[].first_node_address",
                "curve_lists[].last_node_address",
            ],
        },
        "pixels": {
            "comparison": "full_800x600_rgb_from_lossless_bgra_bmp",
            "matched_tick_count": matched_ticks,
            "mismatched_tick_count": len(pixel_rows) - matched_ticks,
            "maximum_mismatched_rgb_pixel_count": maximum_mismatch,
            "full_viewport_rgb_per_tick_matched": full_pixels_exact,
            "ticks": pixel_rows,
        },
    }


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    if not path.is_absolute() or path.exists() or not path.parent.is_dir():
        _fail("step_visual_output_invalid")
    payload = (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-index", required=True, type=Path)
    parser.add_argument("--right-index", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = compare(args.left_index, args.right_index)
        _write_exclusive(args.output.resolve(), report)
    except StepVisualComparisonError as error:
        print(f"step visual comparison error: {error}", file=sys.stderr)
        return 2
    print(args.output.resolve())
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
