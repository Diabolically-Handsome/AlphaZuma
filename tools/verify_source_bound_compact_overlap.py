"""Verify a diagnostic compact trajectory against a formal full-state source."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import struct
import sys
from typing import Any, Iterable, Mapping

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from zuma_rl.pc_memory_evidence import canonical_report_bytes
from zuma_rl.pc_memory_trajectory import load_memory_trajectory
from zuma_rl.pc_rng_trajectory import load_rng_trajectory


REPORT_SCHEMA = "zuma-rl.pc-source-bound-compact-overlap-verification"
REPORT_VERSION = 1
BOARD_DUMP_SIZE = 0x1100
BOARD_COLOR_COUNTS_OFFSET = 0xE4
BOARD_COLOR_COUNT_SLOTS = 6
BALL_VTABLE = 0x00960160
BULLET_VTABLE = 0x009636E4
SHOOTER_BULLET_POINTER_OFFSETS = (0x130, 0x134)
CURVE_LIST_OFFSETS = (0x50, 0x5C, 0x68)
BALL_FIELDS = (
    "ball_id",
    "color_id",
    "curve_distance",
    "orientation_radians",
    "previous_orientation_radians",
    "angular_step_radians",
    "position_x",
    "position_y",
    "scale",
    "radius",
    "powerup_previous_type",
    "powerup_primary_type",
    "powerup_secondary_type",
    "flags_b4_c2_hex",
)
RENDER_FLOAT_FIELDS = (
    "curve_distance",
    "orientation_radians",
    "previous_orientation_radians",
    "angular_step_radians",
    "position_x",
    "position_y",
    "scale",
    "radius",
)


class OverlapVerificationError(ValueError):
    """Raised when an input or source-bound overlap is invalid."""


def _fail(reason: str) -> None:
    raise OverlapVerificationError(reason)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise OverlapVerificationError("artifact_unreadable") from error
    return f"sha256:{digest.hexdigest()}"


def _sha256_bytes(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise OverlapVerificationError("json_unreadable") from error
    if not isinstance(value, Mapping):
        _fail("json_root_invalid")
    return value


def _artifact_path(root: Path, artifact: Any) -> Path:
    if not isinstance(artifact, str) or not artifact:
        _fail("artifact_path_invalid")
    pure = PurePosixPath(artifact)
    if pure.is_absolute() or ".." in pure.parts or "\\" in artifact:
        _fail("artifact_path_escape")
    root = root.resolve()
    path = root.joinpath(*pure.parts).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        _fail("artifact_path_escape")
    return path


def _bound_artifact(
    root: Path,
    row: Mapping[str, Any],
    *,
    expected_size: int | None = None,
) -> bytes:
    path = _artifact_path(root, row.get("artifact"))
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise OverlapVerificationError("artifact_unreadable") from error
    if expected_size is not None and len(payload) != expected_size:
        _fail("artifact_size_mismatch")
    declared_bytes = row.get("artifact_bytes")
    if declared_bytes is not None and declared_bytes != len(payload):
        _fail("artifact_declared_size_mismatch")
    if row.get("artifact_sha256") != _sha256_bytes(payload):
        _fail("artifact_hash_mismatch")
    return payload


def _ball_projection(
    value: Mapping[str, Any],
    *,
    kind: str,
    vtable: int,
    verify_render_bits: bool,
) -> dict[str, Any]:
    try:
        result = {
            "kind": kind,
            "vtable": vtable,
            **{key: value[key] for key in BALL_FIELDS},
        }
    except KeyError as error:
        raise OverlapVerificationError("ball_field_missing") from error
    if (
        (kind == "ball" and vtable != BALL_VTABLE)
        or (kind == "bullet" and vtable != BULLET_VTABLE)
        or kind not in {"ball", "bullet"}
    ):
        _fail("ball_identity_invalid")
    floats = [value[key] for key in RENDER_FLOAT_FIELDS]
    if any(
        isinstance(item, bool)
        or not isinstance(item, (int, float))
        or not math.isfinite(float(item))
        for item in floats
    ):
        _fail("ball_float_invalid")
    if verify_render_bits:
        expected_bits = struct.pack(
            "<8f", *(float(item) for item in floats)
        ).hex()
        if value.get("render_float32_bits_hex") != expected_bits:
            _fail("ball_render_bits_mismatch")
    return result


def _source_tick_projection(
    tick_path: Path,
    tick: Mapping[str, Any],
) -> dict[str, Any]:
    board = tick.get("active_board")
    if not isinstance(board, Mapping):
        _fail("source_active_board_missing")
    board_raw = _bound_artifact(
        tick_path.parent,
        board,
        expected_size=BOARD_DUMP_SIZE,
    )
    counts = list(
        struct.unpack_from(
            f"<{BOARD_COLOR_COUNT_SLOTS}i",
            board_raw,
            BOARD_COLOR_COUNTS_OFFSET,
        )
    )

    child = board.get("primary_child")
    if not isinstance(child, Mapping):
        _fail("source_shooter_missing")
    bullet_rows = child.get("bullets")
    if not isinstance(bullet_rows, list) or len(bullet_rows) != 2:
        _fail("source_shooter_bullets_invalid")
    bullets: list[dict[str, Any]] = []
    for row, expected_offset in zip(
        bullet_rows,
        SHOOTER_BULLET_POINTER_OFFSETS,
        strict=True,
    ):
        if (
            not isinstance(row, Mapping)
            or row.get("shooter_pointer_offset") != expected_offset
            or not isinstance(row.get("ball"), Mapping)
        ):
            _fail("source_shooter_bullet_invalid")
        bullets.append(
            _ball_projection(
                row["ball"],
                kind="bullet",
                vtable=BULLET_VTABLE,
                verify_render_bits=False,
            )
        )

    qrand = board.get("qrand")
    if not isinstance(qrand, Mapping):
        _fail("source_qrand_missing")
    vector_rows = qrand.get("vectors")
    if not isinstance(vector_rows, list):
        _fail("source_qrand_vectors_invalid")
    vectors = {
        row["name"]: row["values"]
        for row in vector_rows
        if isinstance(row, Mapping)
    }
    if set(vectors) != {"weights", "sways", "last_hit", "previous_hit"}:
        _fail("source_qrand_vector_set_invalid")

    manager = board.get("curve_manager")
    if not isinstance(manager, Mapping):
        _fail("source_curve_manager_missing")
    curve_rows = manager.get("curves")
    if not isinstance(curve_rows, list):
        _fail("source_curves_invalid")
    curves: list[dict[str, Any]] = []
    for expected_curve_index, curve in enumerate(curve_rows):
        if (
            not isinstance(curve, Mapping)
            or curve.get("index") != expected_curve_index
        ):
            _fail("source_curve_invalid")
        list_rows = curve.get("intrusive_lists")
        if not isinstance(list_rows, list) or [
            row.get("container_offset")
            for row in list_rows
            if isinstance(row, Mapping)
        ] != list(CURVE_LIST_OFFSETS):
            _fail("source_curve_lists_invalid")
        lists: list[dict[str, Any]] = []
        for list_row in list_rows:
            records = list_row.get("records")
            if not isinstance(records, list):
                _fail("source_curve_records_invalid")
            entities: list[dict[str, Any]] = []
            for expected_index, record in enumerate(records):
                if (
                    not isinstance(record, Mapping)
                    or record.get("index") != expected_index
                    or not isinstance(record.get("ball"), Mapping)
                ):
                    _fail("source_curve_record_invalid")
                entities.append(
                    _ball_projection(
                        record["ball"],
                        kind=record.get("payload_kind"),
                        vtable=record.get("payload_head"),
                        verify_render_bits=False,
                    )
                )
            if (
                list_row.get("declared_count") != len(entities)
                or list_row.get("traversed_count") != len(entities)
            ):
                _fail("source_curve_list_count_mismatch")
            lists.append(
                {
                    "curve_index": expected_curve_index,
                    "container_offset": list_row["container_offset"],
                    "entities": entities,
                }
            )
        curves.append({"index": expected_curve_index, "lists": lists})

    crt = board.get("thread_crt_rand")
    fired = board.get("fired_bullets")
    if not isinstance(crt, Mapping) or not isinstance(fired, Mapping):
        _fail("source_rng_or_fired_state_missing")
    return {
        "score": board.get("score"),
        "displayed_score": board.get("displayed_score"),
        "score_target": board.get("score_target"),
        "board_color_counts": counts,
        "bullets": bullets,
        "qrand": {
            "update_count": qrand.get("update_count"),
            "selected_index": qrand.get("selected_index"),
            "vectors": vectors,
        },
        "crt": crt.get("state"),
        "fired_count": fired.get("traversed_count"),
        "curves": curves,
        "sample_barrier": tick.get("sample_barrier"),
        "sample_phase": tick.get("sample_phase"),
    }


def _diagnostic_tick_projection(tick: Mapping[str, Any]) -> dict[str, Any]:
    current = tick.get("current_ball")
    following = tick.get("next_ball")
    if not isinstance(current, Mapping) or not isinstance(following, Mapping):
        _fail("diagnostic_shooter_bullets_missing")
    curve_rows = tick.get("curve_lists")
    if not isinstance(curve_rows, list):
        _fail("diagnostic_curve_lists_invalid")
    curve_indices = sorted(
        {
            row.get("curve_index")
            for row in curve_rows
            if isinstance(row, Mapping)
        }
    )
    curves: list[dict[str, Any]] = []
    for curve_index in curve_indices:
        lists: list[dict[str, Any]] = []
        selected = [
            row
            for row in curve_rows
            if isinstance(row, Mapping)
            and row.get("curve_index") == curve_index
        ]
        if [row.get("container_offset") for row in selected] != list(
            CURVE_LIST_OFFSETS
        ):
            _fail("diagnostic_curve_list_set_invalid")
        for row in selected:
            entities = row.get("entities")
            if not isinstance(entities, list) or any(
                not isinstance(entity, Mapping) for entity in entities
            ):
                _fail("diagnostic_curve_entities_invalid")
            if (
                row.get("declared_count") != len(entities)
                or row.get("traversed_count") != len(entities)
            ):
                _fail("diagnostic_curve_list_count_mismatch")
            lists.append(
                {
                    "curve_index": curve_index,
                    "container_offset": row["container_offset"],
                    "entities": [
                        _ball_projection(
                            entity,
                            kind=entity.get("kind"),
                            vtable=entity.get("vtable"),
                            verify_render_bits=True,
                        )
                        for entity in entities
                    ],
                }
            )
        curves.append({"index": curve_index, "lists": lists})
    qrand = tick.get("qrand")
    if not isinstance(qrand, Mapping):
        _fail("diagnostic_qrand_missing")
    return {
        "score": tick.get("score"),
        "displayed_score": tick.get("displayed_score"),
        "score_target": tick.get("score_target"),
        "board_color_counts": tick.get("board_color_counts"),
        "bullets": [
            _ball_projection(
                current,
                kind=current.get("kind"),
                vtable=current.get("vtable"),
                verify_render_bits=True,
            ),
            _ball_projection(
                following,
                kind=following.get("kind"),
                vtable=following.get("vtable"),
                verify_render_bits=True,
            ),
        ],
        "qrand": {
            "update_count": qrand.get("update_count"),
            "selected_index": qrand.get("selected_index"),
            "vectors": qrand.get("vectors"),
        },
        "crt": tick.get("thread_crt_rand_state"),
        "fired_count": tick.get("fired_bullet_count"),
        "curves": curves,
        "sample_barrier": tick.get("sample_barrier"),
        "sample_phase": tick.get("sample_phase"),
    }


def _difference_paths(left: Any, right: Any, path: str = "") -> list[str]:
    if type(left) is not type(right):
        return [path or "/"]
    if isinstance(left, Mapping):
        result: list[str] = []
        for key in sorted(set(left) | set(right)):
            child = f"{path}/{key}"
            if key not in left or key not in right:
                result.append(child)
            else:
                result.extend(_difference_paths(left[key], right[key], child))
        return result
    if isinstance(left, list):
        if len(left) != len(right):
            return [f"{path}/length"]
        result = []
        for index, (left_item, right_item) in enumerate(
            zip(left, right, strict=True)
        ):
            result.extend(
                _difference_paths(left_item, right_item, f"{path}/{index}")
            )
        return result
    return [] if left == right else [path or "/"]


def verify_overlap(
    *,
    source_index_path: Path,
    diagnostic_index_path: Path,
    expected_source_sha256: str,
    expected_diagnostic_sha256: str,
) -> dict[str, Any]:
    source_index_path = source_index_path.resolve()
    diagnostic_index_path = diagnostic_index_path.resolve()
    if _sha256_path(source_index_path) != expected_source_sha256:
        _fail("source_index_hash_mismatch")
    if _sha256_path(diagnostic_index_path) != expected_diagnostic_sha256:
        _fail("diagnostic_index_hash_mismatch")

    diagnostic_index = _load_json(diagnostic_index_path)
    start_update = diagnostic_index.get("start_update")
    end_update = diagnostic_index.get("end_update")
    if (
        isinstance(start_update, bool)
        or not isinstance(start_update, int)
        or isinstance(end_update, bool)
        or not isinstance(end_update, int)
        or end_update < start_update
    ):
        _fail("diagnostic_update_range_invalid")
    source_frames = load_memory_trajectory(
        source_index_path,
        start_update=start_update,
        end_update=end_update,
    )
    diagnostic_frames = load_rng_trajectory(diagnostic_index_path)
    diagnostic_ticks = diagnostic_index.get("ticks")
    if (
        not isinstance(diagnostic_ticks, list)
        or len(source_frames) != len(diagnostic_frames)
        or len(source_frames) != len(diagnostic_ticks)
    ):
        _fail("trajectory_tick_count_mismatch")
    source_index = _load_json(source_index_path)
    source_rows = {
        row["framework_update"]: row
        for row in source_index.get("ticks", [])
        if isinstance(row, Mapping)
        and isinstance(row.get("framework_update"), int)
    }

    digests: list[dict[str, Any]] = []
    first_mismatch: dict[str, Any] | None = None
    compact_equal_count = 0
    mtrand_equal_count = 0
    for source_frame, diagnostic_frame, diagnostic_tick in zip(
        source_frames,
        diagnostic_frames,
        diagnostic_ticks,
        strict=True,
    ):
        update = source_frame.update
        if (
            diagnostic_frame.update != update
            or not isinstance(diagnostic_tick, Mapping)
            or diagnostic_tick.get("framework_update") != update
            or update not in source_rows
        ):
            _fail("trajectory_update_alignment_invalid")
        source_tick_path = _artifact_path(
            source_index_path.parent,
            source_rows[update].get("artifact"),
        )
        source_tick = _load_json(source_tick_path)
        source_projection = _source_tick_projection(
            source_tick_path,
            source_tick,
        )
        diagnostic_projection = _diagnostic_tick_projection(diagnostic_tick)
        difference_paths = _difference_paths(
            source_projection,
            diagnostic_projection,
        )
        compact_equal = not difference_paths
        if compact_equal:
            compact_equal_count += 1
        source_mtrand = source_frame.global_mtrand
        mtrand_equal = bool(
            source_mtrand is not None
            and source_mtrand.index == diagnostic_frame.mtrand_index
            and source_mtrand.words == diagnostic_frame.mtrand_words
        )
        if mtrand_equal:
            mtrand_equal_count += 1
        source_digest = _sha256_bytes(
            canonical_report_bytes(source_projection)
        )
        diagnostic_digest = _sha256_bytes(
            canonical_report_bytes(diagnostic_projection)
        )
        digests.append(
            {
                "framework_update": update,
                "compact_projection_sha256": source_digest,
                "mtrand_index": diagnostic_frame.mtrand_index,
            }
        )
        if first_mismatch is None and (not compact_equal or not mtrand_equal):
            first_mismatch = {
                "framework_update": update,
                "compact_difference_paths": difference_paths,
                "source_compact_projection_sha256": source_digest,
                "diagnostic_compact_projection_sha256": diagnostic_digest,
                "mtrand_exact": mtrand_equal,
            }

    tick_count = len(source_frames)
    status = "PASS" if first_mismatch is None else "FAIL"
    digest_root = _sha256_bytes(canonical_report_bytes({"ticks": digests}))
    return {
        "schema": REPORT_SCHEMA,
        "version": REPORT_VERSION,
        "status": status,
        "source": {
            "index": str(source_index_path),
            "index_sha256": expected_source_sha256,
        },
        "diagnostic": {
            "index": str(diagnostic_index_path),
            "index_sha256": expected_diagnostic_sha256,
        },
        "start_update": start_update,
        "end_update": end_update,
        "tick_count": tick_count,
        "compact_projection_exact_tick_count": compact_equal_count,
        "global_mtrand_exact_tick_count": mtrand_equal_count,
        "all_compact_projections_exact": compact_equal_count == tick_count,
        "all_global_mtrand_states_exact": mtrand_equal_count == tick_count,
        "render_float32_bits_recomputed": True,
        "source_full_state_artifacts_revalidated": True,
        "diagnostic_compact_artifacts_revalidated": True,
        "tick_digest_root": digest_root,
        "first_tick_digest": digests[0],
        "last_tick_digest": digests[-1],
        "first_mismatch": first_mismatch,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-index", required=True, type=Path)
    parser.add_argument("--source-index-sha256", required=True)
    parser.add_argument("--diagnostic-index", required=True, type=Path)
    parser.add_argument("--diagnostic-index-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output.resolve()
    if output.exists() or not output.parent.is_dir():
        raise FileExistsError("output path must be new in an existing directory")
    report = verify_overlap(
        source_index_path=args.source_index,
        diagnostic_index_path=args.diagnostic_index,
        expected_source_sha256=args.source_index_sha256,
        expected_diagnostic_sha256=args.diagnostic_index_sha256,
    )
    output.write_bytes(canonical_report_bytes(report))
    print(
        f"status={report['status']} ticks={report['tick_count']} "
        f"compact_exact={report['compact_projection_exact_tick_count']} "
        f"mtrand_exact={report['global_mtrand_exact_tick_count']}"
    )
    print(output)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
