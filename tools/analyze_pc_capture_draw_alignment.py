"""Compare two diagnostic PC captures by update and framework draw count."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
from difflib import SequenceMatcher
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "zuma-rl.pc-capture-draw-alignment-diagnostic"
VERSION = 1
_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")


class DrawAlignmentError(ValueError):
    """Raised when capture evidence is malformed or incompletely bound."""


def _fail(reason: str) -> None:
    raise DrawAlignmentError(reason)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except OSError:
        _fail("draw_alignment_input_read_failed")
    return f"sha256:{digest.hexdigest()}"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        _fail("draw_alignment_json_invalid")
    if not isinstance(value, dict):
        _fail("draw_alignment_json_invalid")
    return value


def _require_int(value: Any, reason: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(reason)
    return value


def _require_sha256(value: Any, reason: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        _fail(reason)
    return value


@dataclass(frozen=True)
class FrameRecord:
    sequence: int
    present_ticks: int
    host_perf_counter_ns: int
    frame_sha256: str
    update_before: int
    update_after: int
    draw_before: int
    draw_after: int


@dataclass(frozen=True)
class RunTimeline:
    root: Path
    metadata_sha256: str
    frames_csv_sha256: str
    update_map_sha256: str
    state_sidecar_sha256: str
    process_id: int
    process_creation_filetime_100ns: int
    executable_sha256: str
    width: int
    height: int
    records: tuple[FrameRecord, ...]


@dataclass(frozen=True)
class VisualSegment:
    frame_sha256: str
    first_sequence: int
    last_sequence: int
    frame_count: int


def _snapshot_counter(
    snapshot: Any,
    field: str,
    reason: str,
) -> int:
    if not isinstance(snapshot, dict):
        _fail(reason)
    value = _require_int(snapshot.get(field), reason)
    if value < 0:
        _fail(reason)
    return value


def _load_run(run_root: Path) -> RunTimeline:
    metadata_path = run_root / "capture" / "metadata.json"
    frames_csv_path = run_root / "capture" / "frames.csv"
    update_map_path = run_root / "framework-updates.json"
    state_path = run_root / "framework-state-diagnostic.json"
    if not run_root.is_dir():
        _fail("draw_alignment_run_root_missing")

    metadata = _read_json(metadata_path)
    update_map = _read_json(update_map_path)
    state = _read_json(state_path)
    metadata_sha256 = _sha256_file(metadata_path)
    frames_csv_sha256 = _sha256_file(frames_csv_path)
    update_map_sha256 = _sha256_file(update_map_path)
    state_sidecar_sha256 = _sha256_file(state_path)

    if (
        metadata.get("schema") != "zuma-rl.dxgi-bgra-capture"
        or metadata.get("status") != "acquisition_complete"
    ):
        _fail("draw_alignment_capture_metadata_invalid")
    if (
        update_map.get("schema") != "zuma-rl.pc-framework-update-map"
        or update_map.get("version") != 1
    ):
        _fail("draw_alignment_update_map_invalid")
    if (
        state.get("schema") != "zuma-rl.pc-framework-state-diagnostic"
        or state.get("version") != 1
        or state.get("diagnostic_only") is not True
    ):
        _fail("draw_alignment_state_sidecar_invalid")
    if (
        state.get("capture_metadata_sha256") != metadata_sha256
        or state.get("framework_update_map_sha256") != update_map_sha256
    ):
        _fail("draw_alignment_state_binding_mismatch")
    if metadata.get("frames_csv_sha256") != frames_csv_sha256:
        _fail("draw_alignment_frames_csv_binding_mismatch")

    frame_count = _require_int(
        metadata.get("frame_count"),
        "draw_alignment_capture_metadata_invalid",
    )
    width = _require_int(
        metadata.get("width"), "draw_alignment_capture_metadata_invalid"
    )
    height = _require_int(
        metadata.get("height"), "draw_alignment_capture_metadata_invalid"
    )
    if frame_count <= 0 or width <= 0 or height <= 0:
        _fail("draw_alignment_capture_metadata_invalid")

    target = metadata.get("target_identity")
    if not isinstance(target, dict):
        _fail("draw_alignment_target_identity_invalid")
    process_id = _require_int(
        target.get("process_id"), "draw_alignment_target_identity_invalid"
    )
    creation = _require_int(
        target.get("process_creation_filetime_100ns"),
        "draw_alignment_target_identity_invalid",
    )
    executable_sha256 = _require_sha256(
        target.get("executable_sha256"),
        "draw_alignment_target_identity_invalid",
    )
    if (
        state.get("process_id") != process_id
        or state.get("process_creation_filetime_100ns") != creation
        or state.get("executable_sha256") != executable_sha256
    ):
        _fail("draw_alignment_process_binding_mismatch")

    update_records = update_map.get("records")
    state_records = state.get("records")
    if not isinstance(update_records, list) or not isinstance(
        state_records, list
    ):
        _fail("draw_alignment_timeline_invalid")
    try:
        with frames_csv_path.open("r", encoding="ascii", newline="") as stream:
            frame_rows = list(csv.DictReader(stream))
    except (OSError, UnicodeError, csv.Error):
        _fail("draw_alignment_frames_csv_invalid")
    if (
        len(frame_rows) != frame_count
        or len(update_records) != frame_count
        or len(state_records) != frame_count
    ):
        _fail("draw_alignment_timeline_length_mismatch")

    records: list[FrameRecord] = []
    prior_draw_before = -1
    prior_draw_after = -1
    for sequence, (frame, update, sample) in enumerate(
        zip(frame_rows, update_records, state_records, strict=True)
    ):
        if not isinstance(update, dict) or not isinstance(sample, dict):
            _fail("draw_alignment_timeline_invalid")
        try:
            frame_sequence = int(frame["sequence"])
            present_ticks = int(frame["present_ticks"])
            host_ns = int(frame["host_perf_counter_ns"])
        except (KeyError, TypeError, ValueError):
            _fail("draw_alignment_frames_csv_invalid")
        if (
            frame_sequence != sequence
            or update.get("sequence") != sequence
            or sample.get("sequence") != sequence
            or update.get("present_ticks") != present_ticks
            or sample.get("present_ticks") != present_ticks
            or update.get("host_perf_counter_ns") != host_ns
            or sample.get("host_perf_counter_ns") != host_ns
        ):
            _fail("draw_alignment_sequence_binding_mismatch")
        frame_sha256 = _require_sha256(
            frame.get("frame_sha256"), "draw_alignment_frame_hash_invalid"
        )
        update_before = _require_int(
            update.get("update_before"), "draw_alignment_update_map_invalid"
        )
        update_after = _require_int(
            update.get("update_after"), "draw_alignment_update_map_invalid"
        )
        state_before = _snapshot_counter(
            sample.get("before"),
            "update_count",
            "draw_alignment_state_sidecar_invalid",
        )
        state_after = _snapshot_counter(
            sample.get("after"),
            "update_count",
            "draw_alignment_state_sidecar_invalid",
        )
        draw_before = _snapshot_counter(
            sample.get("before"),
            "draw_count",
            "draw_alignment_state_sidecar_invalid",
        )
        draw_after = _snapshot_counter(
            sample.get("after"),
            "draw_count",
            "draw_alignment_state_sidecar_invalid",
        )
        if (
            update_before != state_before
            or update_after != state_after
            or update_after < update_before
            or draw_after < draw_before
            or draw_before < prior_draw_before
            or draw_after < prior_draw_after
        ):
            _fail("draw_alignment_counter_invariant_failed")
        prior_draw_before = draw_before
        prior_draw_after = draw_after
        records.append(
            FrameRecord(
                sequence=sequence,
                present_ticks=present_ticks,
                host_perf_counter_ns=host_ns,
                frame_sha256=frame_sha256,
                update_before=update_before,
                update_after=update_after,
                draw_before=draw_before,
                draw_after=draw_after,
            )
        )

    return RunTimeline(
        root=run_root.resolve(),
        metadata_sha256=metadata_sha256,
        frames_csv_sha256=frames_csv_sha256,
        update_map_sha256=update_map_sha256,
        state_sidecar_sha256=state_sidecar_sha256,
        process_id=process_id,
        process_creation_filetime_100ns=creation,
        executable_sha256=executable_sha256,
        width=width,
        height=height,
        records=tuple(records),
    )


def _collapse(records: Sequence[FrameRecord]) -> tuple[VisualSegment, ...]:
    collapsed: list[VisualSegment] = []
    for record in records:
        if collapsed and collapsed[-1].frame_sha256 == record.frame_sha256:
            prior = collapsed[-1]
            collapsed[-1] = VisualSegment(
                frame_sha256=prior.frame_sha256,
                first_sequence=prior.first_sequence,
                last_sequence=record.sequence,
                frame_count=prior.frame_count + 1,
            )
            continue
        collapsed.append(
            VisualSegment(
                frame_sha256=record.frame_sha256,
                first_sequence=record.sequence,
                last_sequence=record.sequence,
                frame_count=1,
            )
        )
    return tuple(collapsed)


def _histogram(counter: Counter[int]) -> list[list[int]]:
    return [[value, counter[value]] for value in sorted(counter)]


def _mode(counter: Counter[int]) -> int:
    if not counter:
        _fail("draw_alignment_no_matched_visuals")
    return min(counter, key=lambda value: (-counter[value], abs(value), value))


def _ordered_visual_alignment(
    left: RunTimeline,
    right: RunTimeline,
) -> tuple[dict[str, Any], int]:
    left_segments = _collapse(left.records)
    right_segments = _collapse(right.records)
    matcher = SequenceMatcher(
        None,
        [row.frame_sha256 for row in left_segments],
        [row.frame_sha256 for row in right_segments],
        autojunk=False,
    )
    draw_deltas: Counter[int] = Counter()
    update_deltas: Counter[int] = Counter()
    matched_rows: list[dict[str, int]] = []
    for block in matcher.get_matching_blocks():
        for index in range(block.size):
            left_segment = left_segments[block.a + index]
            right_segment = right_segments[block.b + index]
            left_frame = left.records[left_segment.first_sequence]
            right_frame = right.records[right_segment.first_sequence]
            draw_delta = left_frame.draw_after - right_frame.draw_after
            update_delta = left_frame.update_after - right_frame.update_after
            draw_deltas[draw_delta] += 1
            update_deltas[update_delta] += 1
            matched_rows.append(
                {
                    "left_collapsed_index": block.a + index,
                    "right_collapsed_index": block.b + index,
                    "left_first_sequence": left_segment.first_sequence,
                    "right_first_sequence": right_segment.first_sequence,
                    "draw_after_delta": draw_delta,
                    "update_after_delta": update_delta,
                }
            )
    modal_draw_delta = _mode(draw_deltas)
    opcodes = matcher.get_opcodes()
    non_equal = [
        {
            "tag": tag,
            "left_collapsed_range": [left_start, left_end],
            "right_collapsed_range": [right_start, right_end],
        }
        for tag, left_start, left_end, right_start, right_end in opcodes
        if tag != "equal"
    ]
    return (
        {
            "left_frame_count": len(left.records),
            "right_frame_count": len(right.records),
            "left_collapsed_visual_count": len(left_segments),
            "right_collapsed_visual_count": len(right_segments),
            "matched_collapsed_visual_count": len(matched_rows),
            "sequence_matcher_ratio": matcher.ratio(),
            "sample_phase": "first_present_of_matched_visual_after_capture",
            "counter_phase": "framework_state_after_dxgi_grab",
            "draw_after_delta_histogram": _histogram(draw_deltas),
            "update_after_delta_histogram": _histogram(update_deltas),
            "modal_draw_after_delta": modal_draw_delta,
            "modal_draw_after_delta_support": draw_deltas[modal_draw_delta],
            "modal_draw_after_delta_fraction": (
                draw_deltas[modal_draw_delta] / len(matched_rows)
            ),
            "non_modal_matched_rows": [
                row
                for row in matched_rows
                if row["draw_after_delta"] != modal_draw_delta
            ],
            "non_equal_opcode_count": len(non_equal),
            "non_equal_opcodes": non_equal,
        },
        modal_draw_delta,
    )


def _candidate_map(
    records: Sequence[FrameRecord],
    *,
    counter: str,
) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for row in records:
        if counter == "draw":
            values = {row.draw_before, row.draw_after}
        elif counter == "stable_update":
            if row.update_before != row.update_after:
                continue
            values = {row.update_before}
        else:
            raise AssertionError(counter)
        for value in values:
            entry = result.setdefault(
                value,
                {"hashes": set(), "sequences": [], "updates": set()},
            )
            entry["hashes"].add(row.frame_sha256)
            entry["sequences"].append(row.sequence)
            entry["updates"].update(
                {row.update_before, row.update_after}
            )
    return result


def _longest_contiguous(values: Sequence[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "range": None}
    best_start = values[0]
    best_end = values[0]
    start = values[0]
    prior = values[0]
    for value in values[1:]:
        if value != prior + 1:
            if prior - start > best_end - best_start:
                best_start, best_end = start, prior
            start = value
        prior = value
    if prior - start > best_end - best_start:
        best_start, best_end = start, prior
    return {
        "count": best_end - best_start + 1,
        "range": [best_start, best_end],
    }


def _compare_update_baseline(
    left: RunTimeline,
    right: RunTimeline,
) -> dict[str, Any]:
    left_map = _candidate_map(left.records, counter="stable_update")
    right_map = _candidate_map(right.records, counter="stable_update")
    common = sorted(set(left_map).intersection(right_map))
    if not common:
        _fail("draw_alignment_no_common_stable_updates")
    exact: list[int] = []
    failures: list[dict[str, Any]] = []
    for update in common:
        left_entry = left_map[update]
        right_entry = right_map[update]
        if left_entry["hashes"].intersection(right_entry["hashes"]):
            exact.append(update)
            continue
        failures.append(
            {
                "update": update,
                "left_sequences": left_entry["sequences"],
                "right_sequences": right_entry["sequences"],
                "left_hashes": sorted(left_entry["hashes"]),
                "right_hashes": sorted(right_entry["hashes"]),
            }
        )
    return {
        "comparison": "full_frame_sha256_at_same_stable_update",
        "common_stable_update_count": len(common),
        "first_common_stable_update": common[0],
        "last_common_stable_update": common[-1],
        "exact_hash_overlap_count": len(exact),
        "exact_hash_overlap_fraction": len(exact) / len(common),
        "mismatch_count": len(failures),
        "longest_contiguous_exact_range": _longest_contiguous(exact),
        "failures": failures,
    }


def _compare_draw_candidates(
    left: RunTimeline,
    right: RunTimeline,
    modal_draw_delta: int,
) -> dict[str, Any]:
    left_map = _candidate_map(left.records, counter="draw")
    right_map = _candidate_map(right.records, counter="draw")
    pairs = sorted(
        (left_draw, left_draw - modal_draw_delta)
        for left_draw in left_map
        if left_draw - modal_draw_delta in right_map
    )
    if not pairs:
        _fail("draw_alignment_no_common_draw_candidates")
    exact_count = 0
    failures: list[dict[str, Any]] = []
    for left_draw, right_draw in pairs:
        left_entry = left_map[left_draw]
        right_entry = right_map[right_draw]
        shared = left_entry["hashes"].intersection(right_entry["hashes"])
        if shared:
            exact_count += 1
            continue
        failures.append(
            {
                "left_draw_count": left_draw,
                "right_draw_count": right_draw,
                "left_updates": sorted(left_entry["updates"]),
                "right_updates": sorted(right_entry["updates"]),
                "left_sequences": left_entry["sequences"],
                "right_sequences": right_entry["sequences"],
                "left_hashes": sorted(left_entry["hashes"]),
                "right_hashes": sorted(right_entry["hashes"]),
            }
        )
    return {
        "comparison": (
            "full_frame_sha256_overlap_for_union_of_"
            "framework_state_before_after_draw_candidates"
        ),
        "draw_delta_source": (
            "mode_of_after_draw_count_delta_at_first_present_"
            "of_sequence-matched_collapsed_visuals"
        ),
        "modal_draw_delta": modal_draw_delta,
        "paired_draw_count": len(pairs),
        "first_left_draw_count": pairs[0][0],
        "last_left_draw_count": pairs[-1][0],
        "first_right_draw_count": pairs[0][1],
        "last_right_draw_count": pairs[-1][1],
        "exact_hash_overlap_count": exact_count,
        "exact_hash_overlap_fraction": exact_count / len(pairs),
        "unresolved_pair_count": len(failures),
        "unresolved_pairs": failures,
    }


def _input_row(run: RunTimeline) -> dict[str, Any]:
    return {
        "run_root": str(run.root),
        "capture_metadata_sha256": run.metadata_sha256,
        "frames_csv_sha256": run.frames_csv_sha256,
        "framework_update_map_sha256": run.update_map_sha256,
        "framework_state_sidecar_sha256": run.state_sidecar_sha256,
        "process_id": run.process_id,
        "process_creation_filetime_100ns": (
            run.process_creation_filetime_100ns
        ),
        "executable_sha256": run.executable_sha256,
        "frame_count": len(run.records),
        "width": run.width,
        "height": run.height,
    }


def analyze(session_root: Path) -> dict[str, Any]:
    session_root = session_root.resolve()
    left = _load_run(session_root / "run-r1")
    right = _load_run(session_root / "run-r2")
    if (left.width, left.height) != (right.width, right.height):
        _fail("draw_alignment_capture_geometry_mismatch")
    if left.executable_sha256 != right.executable_sha256:
        _fail("draw_alignment_executable_mismatch")

    ordered, modal_draw_delta = _ordered_visual_alignment(left, right)
    update_baseline = _compare_update_baseline(left, right)
    draw_alignment = _compare_draw_candidates(
        left, right, modal_draw_delta
    )
    comparable_denominator = (
        update_baseline["common_stable_update_count"]
        == draw_alignment["paired_draw_count"]
    )
    exact_improvement = (
        draw_alignment["exact_hash_overlap_count"]
        - update_baseline["exact_hash_overlap_count"]
        if comparable_denominator
        else None
    )
    unresolved = draw_alignment["unresolved_pair_count"]
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "classification": "diagnostic-only-not-pc-golden",
        "gate_effect": "none",
        "status": (
            "OBSERVED_DRAW_ALIGNMENT_EXACT"
            if unresolved == 0
            else "OBSERVED_DRAW_ALIGNMENT_INCOMPLETE"
        ),
        "session_root": str(session_root),
        "inputs": {
            "plan_sha256": (
                _sha256_file(session_root / "plan.json")
                if (session_root / "plan.json").is_file()
                else None
            ),
            "left": _input_row(left),
            "right": _input_row(right),
        },
        "independence": {
            "distinct_process_ids": left.process_id != right.process_id,
            "distinct_process_creation_times": (
                left.process_creation_filetime_100ns
                != right.process_creation_filetime_100ns
            ),
            "same_executable_sha256": True,
        },
        "ordered_present_visuals": ordered,
        "same_stable_update_baseline": update_baseline,
        "normalized_draw_candidate_alignment": draw_alignment,
        "comparison": {
            "denominators_equal": comparable_denominator,
            "exact_hash_overlap_improvement_count": exact_improvement,
            "observation": (
                "draw_count_normalization_is_exact_for_all_paired_draws"
                if unresolved == 0
                else (
                    "draw_count_normalization_improves_exact_hash_overlap_"
                    "but_per-present_sampling_leaves_unresolved_boundaries"
                )
            ),
        },
    }


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> str:
    if not path.is_absolute() or path.exists() or not path.parent.is_dir():
        _fail("draw_alignment_output_invalid")
    payload = (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")
    try:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError:
        _fail("draw_alignment_output_write_failed")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = analyze(args.session_root)
        digest = _write_exclusive(args.output.resolve(), report)
    except DrawAlignmentError as error:
        print(f"draw alignment error: {error}", file=sys.stderr)
        return 2
    print(f"{args.output.resolve()}\n{digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
