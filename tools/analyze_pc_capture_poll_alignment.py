"""Diagnose masked visual alignment with high-rate framework draw events."""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter
import csv
from dataclasses import dataclass
from difflib import SequenceMatcher
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

if __package__:
    from .analyze_pc_capture_draw_alignment import (
        FrameRecord,
        RunTimeline,
        _load_run,
        _longest_contiguous,
        _sha256_file,
    )
    from .analyze_pc_capture_visual_alignment import (
        _derive_anchor_mask,
        _mask_sha256,
        _masked_digest,
    )
else:
    from analyze_pc_capture_draw_alignment import (
        FrameRecord,
        RunTimeline,
        _load_run,
        _longest_contiguous,
        _sha256_file,
    )
    from analyze_pc_capture_visual_alignment import (
        _derive_anchor_mask,
        _mask_sha256,
        _masked_digest,
    )


SCHEMA = "zuma-rl.pc-capture-poll-alignment-diagnostic"
VERSION = 1


class PollAlignmentError(ValueError):
    """Raised when diagnostic evidence is malformed or incompletely bound."""


def _fail(reason: str) -> None:
    raise PollAlignmentError(reason)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        _fail("poll_alignment_json_invalid")
    if not isinstance(value, dict):
        _fail("poll_alignment_json_invalid")
    return value


@dataclass(frozen=True)
class PollDrawEvent:
    draw_count: int
    update_count: int
    sample_after_perf_counter_ns: int


@dataclass(frozen=True)
class MaskedRun:
    timeline: RunTimeline
    qpc_frequency: int
    capture_start_perf_counter_ns: int
    capture_end_perf_counter_ns: int
    raw_sha256: str
    masked_digests: tuple[str, ...]
    draw_events: tuple[PollDrawEvent, ...]
    poll_sidecar_sha256: str
    poll_summary: Mapping[str, Any]


@dataclass(frozen=True)
class CollapsedVisual:
    digest: str
    first_sequence: int
    last_sequence: int


def _stable_sequences(timeline: RunTimeline) -> dict[int, tuple[int, ...]]:
    values: dict[int, list[int]] = {}
    for row in timeline.records:
        if row.update_before == row.update_after:
            values.setdefault(row.update_before, []).append(row.sequence)
    return {key: tuple(rows) for key, rows in values.items()}


def _load_raw(
    timeline: RunTimeline,
) -> tuple[np.memmap, dict[str, Any], str]:
    metadata_path = timeline.root / "capture" / "metadata.json"
    metadata = _read_json(metadata_path)
    raw_path = timeline.root / "capture" / "frames.bgra.raw"
    expected_raw_sha = metadata.get("raw_sha256")
    if not isinstance(expected_raw_sha, str):
        _fail("poll_alignment_raw_binding_invalid")
    expected_bytes = (
        len(timeline.records) * timeline.height * timeline.width * 4
    )
    try:
        if raw_path.stat().st_size != expected_bytes:
            _fail("poll_alignment_raw_binding_invalid")
    except OSError:
        _fail("poll_alignment_raw_binding_invalid")
    observed_raw_sha = _sha256_file(raw_path)
    if observed_raw_sha != expected_raw_sha:
        _fail("poll_alignment_raw_binding_invalid")
    raw = np.memmap(
        raw_path,
        dtype=np.uint8,
        mode="r",
        shape=(
            len(timeline.records),
            timeline.height,
            timeline.width,
            4,
        ),
    )
    return raw, metadata, observed_raw_sha


def _load_poll(
    timeline: RunTimeline,
    *,
    metadata: Mapping[str, Any],
) -> tuple[tuple[PollDrawEvent, ...], str, Mapping[str, Any]]:
    path = timeline.root / "framework-state-poll-diagnostic.json"
    poll = _read_json(path)
    poll_sha = _sha256_file(path)
    identity = poll.get("process_identity")
    observer = poll.get("observer")
    binding = poll.get("capture_binding")
    topology = (
        observer.get("affinity_topology")
        if isinstance(observer, dict)
        else None
    )
    if not all(
        isinstance(value, dict)
        for value in (identity, observer, binding, topology)
    ):
        _fail("poll_alignment_poll_sidecar_invalid")
    assert isinstance(identity, dict)
    assert isinstance(observer, dict)
    assert isinstance(binding, dict)
    assert isinstance(topology, dict)
    target = metadata.get("target_identity")
    if not isinstance(target, dict):
        _fail("poll_alignment_poll_sidecar_invalid")
    if (
        poll.get("schema")
        != "zuma-rl.pc-framework-state-poll-diagnostic"
        or poll.get("version") != 1
        or poll.get("classification")
        != "diagnostic-only-not-pc-golden"
        or poll.get("gate_effect") != "none"
        or poll.get("status") != "complete"
        or poll.get("skipped_draw_increment_count") != 0
        or identity.get("process_id") != timeline.process_id
        or identity.get("process_creation_filetime_100ns")
        != timeline.process_creation_filetime_100ns
        or identity.get("executable_sha256")
        != timeline.executable_sha256
        or target.get("process_id") != timeline.process_id
        or binding.get("capture_metadata_sha256")
        != timeline.metadata_sha256
        or binding.get("framework_update_map_sha256")
        != timeline.update_map_sha256
        or binding.get("framework_state_sidecar_sha256")
        != timeline.state_sidecar_sha256
        or binding.get("capture_interval_fully_observed") is not True
        or topology.get("physical_cores_disjoint") is not True
        or topology.get("observer_physical_core_logical_mask")
        == topology.get("game_physical_core_logical_mask")
    ):
        _fail("poll_alignment_poll_sidecar_invalid")
    events_value = poll.get("draw_events")
    if not isinstance(events_value, list) or not events_value:
        _fail("poll_alignment_poll_sidecar_invalid")
    events: list[PollDrawEvent] = []
    prior_draw = -1
    prior_time = -1
    for event_index, row in enumerate(events_value):
        if not isinstance(row, dict):
            _fail("poll_alignment_poll_sidecar_invalid")
        draw = row.get("draw_count")
        update = row.get("update_count")
        timestamp = row.get("sample_after_perf_counter_ns")
        if (
            row.get("event_index") != event_index
            or row.get("draw_count_delta") != 1
            or isinstance(draw, bool)
            or not isinstance(draw, int)
            or isinstance(update, bool)
            or not isinstance(update, int)
            or isinstance(timestamp, bool)
            or not isinstance(timestamp, int)
            or draw != prior_draw + 1 and prior_draw >= 0
            or timestamp <= prior_time
        ):
            _fail("poll_alignment_poll_sidecar_invalid")
        events.append(
            PollDrawEvent(
                draw_count=draw,
                update_count=update,
                sample_after_perf_counter_ns=timestamp,
            )
        )
        prior_draw = draw
        prior_time = timestamp
    mean_hz = observer.get("mean_polls_per_second")
    p99 = observer.get("sample_completion_interval_ns", {})
    if not isinstance(p99, dict):
        _fail("poll_alignment_poll_sidecar_invalid")
    summary = {
        "artifact": str(path.resolve()),
        "sha256": poll_sha,
        "arm_at_framework_update": observer.get(
            "arm_at_framework_update"
        ),
        "target_poll_hz": observer.get("target_poll_hz"),
        "mean_polls_per_second": mean_hz,
        "p99_sample_completion_interval_ns": p99.get("p99"),
        "draw_event_count": len(events),
        "skipped_draw_increment_count": 0,
        "affinity_topology": topology,
    }
    return tuple(events), poll_sha, summary


def _masked_run(
    timeline: RunTimeline,
    raw: np.memmap,
    metadata: Mapping[str, Any],
    raw_sha256: str,
    mask: np.ndarray,
) -> MaskedRun:
    qpc_frequency = metadata.get("qpc_frequency")
    capture_start = metadata.get("capture_start_perf_counter_ns")
    capture_end = metadata.get("capture_end_perf_counter_ns")
    if (
        isinstance(qpc_frequency, bool)
        or not isinstance(qpc_frequency, int)
        or qpc_frequency <= 0
        or isinstance(capture_start, bool)
        or not isinstance(capture_start, int)
        or isinstance(capture_end, bool)
        or not isinstance(capture_end, int)
        or capture_end <= capture_start
    ):
        _fail("poll_alignment_capture_clock_invalid")
    events, poll_sha, summary = _load_poll(timeline, metadata=metadata)
    return MaskedRun(
        timeline=timeline,
        qpc_frequency=qpc_frequency,
        capture_start_perf_counter_ns=capture_start,
        capture_end_perf_counter_ns=capture_end,
        raw_sha256=raw_sha256,
        masked_digests=tuple(
            _masked_digest(raw[row.sequence], mask)
            for row in timeline.records
        ),
        draw_events=events,
        poll_sidecar_sha256=poll_sha,
        poll_summary=summary,
    )


def _collapse(digests: Sequence[str]) -> tuple[CollapsedVisual, ...]:
    result: list[CollapsedVisual] = []
    for sequence, digest in enumerate(digests):
        if result and result[-1].digest == digest:
            prior = result[-1]
            result[-1] = CollapsedVisual(
                digest=digest,
                first_sequence=prior.first_sequence,
                last_sequence=sequence,
            )
        else:
            result.append(
                CollapsedVisual(
                    digest=digest,
                    first_sequence=sequence,
                    last_sequence=sequence,
                )
            )
    return tuple(result)


def _stable_update_baseline(
    left: MaskedRun,
    right: MaskedRun,
) -> dict[str, Any]:
    maps: list[dict[int, dict[str, Any]]] = []
    for run in (left, right):
        values: dict[int, dict[str, Any]] = {}
        for row, digest in zip(
            run.timeline.records, run.masked_digests, strict=True
        ):
            if row.update_before != row.update_after:
                continue
            entry = values.setdefault(
                row.update_before, {"hashes": set(), "sequences": []}
            )
            entry["hashes"].add(digest)
            entry["sequences"].append(row.sequence)
        maps.append(values)
    common = sorted(set(maps[0]).intersection(maps[1]))
    exact: list[int] = []
    failures: list[dict[str, Any]] = []
    for update in common:
        if maps[0][update]["hashes"].intersection(
            maps[1][update]["hashes"]
        ):
            exact.append(update)
        else:
            failures.append(
                {
                    "update": update,
                    "left_sequences": maps[0][update]["sequences"],
                    "right_sequences": maps[1][update]["sequences"],
                }
            )
    return {
        "comparison": "masked_bgr_sha256_at_same_stable_update",
        "common_stable_update_count": len(common),
        "exact_hash_overlap_count": len(exact),
        "mismatch_count": len(failures),
        "longest_contiguous_exact_range": _longest_contiguous(exact),
        "failures": failures,
    }


def _present_ns(row: FrameRecord, qpc_frequency: int) -> int:
    return (
        row.present_ticks * 1_000_000_000 + qpc_frequency // 2
    ) // qpc_frequency


def _assigned_draws(run: MaskedRun, delay_ns: int) -> tuple[int, ...]:
    timestamps = [
        row.sample_after_perf_counter_ns for row in run.draw_events
    ]
    draws: list[int] = []
    for frame in run.timeline.records:
        presentation = _present_ns(frame, run.qpc_frequency)
        index = bisect_right(timestamps, presentation - delay_ns) - 1
        if index < 0:
            _fail("poll_alignment_frame_precedes_draw_observation")
        draws.append(run.draw_events[index].draw_count)
    return tuple(draws)


def _mode(counter: Counter[int]) -> int:
    if not counter:
        _fail("poll_alignment_no_matched_visuals")
    return min(counter, key=lambda value: (-counter[value], abs(value), value))


def analyze_delay(
    *,
    delay_ns: int,
    left_records: Sequence[FrameRecord],
    right_records: Sequence[FrameRecord],
    left_digests: Sequence[str],
    right_digests: Sequence[str],
    left_draws: Sequence[int],
    right_draws: Sequence[int],
    matcher: SequenceMatcher,
    left_collapsed: Sequence[CollapsedVisual],
    right_collapsed: Sequence[CollapsedVisual],
) -> dict[str, Any]:
    deltas: Counter[int] = Counter()
    matched_count = 0
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            left_sequence = left_collapsed[
                block.a + offset
            ].first_sequence
            right_sequence = right_collapsed[
                block.b + offset
            ].first_sequence
            deltas[
                left_draws[left_sequence] - right_draws[right_sequence]
            ] += 1
            matched_count += 1
    modal_delta = _mode(deltas)
    maps: list[dict[int, dict[str, Any]]] = []
    for records, digests, draws in (
        (left_records, left_digests, left_draws),
        (right_records, right_digests, right_draws),
    ):
        values: dict[int, dict[str, Any]] = {}
        for record, digest, draw in zip(
            records, digests, draws, strict=True
        ):
            entry = values.setdefault(
                draw,
                {"hashes": set(), "sequences": [], "updates": set()},
            )
            entry["hashes"].add(digest)
            entry["sequences"].append(record.sequence)
            entry["updates"].update(
                {record.update_before, record.update_after}
            )
        maps.append(values)
    pairs = sorted(
        (draw, draw - modal_delta)
        for draw in maps[0]
        if draw - modal_delta in maps[1]
    )
    exact_count = 0
    unresolved: list[dict[str, Any]] = []
    for left_draw, right_draw in pairs:
        if maps[0][left_draw]["hashes"].intersection(
            maps[1][right_draw]["hashes"]
        ):
            exact_count += 1
        else:
            unresolved.append(
                {
                    "left_draw_count": left_draw,
                    "right_draw_count": right_draw,
                    "left_sequences": maps[0][left_draw]["sequences"],
                    "right_sequences": maps[1][right_draw]["sequences"],
                    "left_updates": sorted(maps[0][left_draw]["updates"]),
                    "right_updates": sorted(maps[1][right_draw]["updates"]),
                }
            )
    return {
        "render_settle_delay_ns": delay_ns,
        "modal_draw_delta": modal_delta,
        "modal_draw_delta_support": deltas[modal_delta],
        "matched_collapsed_visual_count": matched_count,
        "paired_draw_count": len(pairs),
        "exact_hash_overlap_count": exact_count,
        "unresolved_pair_count": len(unresolved),
        "unresolved_pairs": unresolved,
    }


def _input_row(run: MaskedRun) -> dict[str, Any]:
    timeline = run.timeline
    return {
        "run_root": str(timeline.root),
        "capture_metadata_sha256": timeline.metadata_sha256,
        "frames_csv_sha256": timeline.frames_csv_sha256,
        "raw_sha256": run.raw_sha256,
        "framework_update_map_sha256": timeline.update_map_sha256,
        "framework_state_sidecar_sha256": timeline.state_sidecar_sha256,
        "framework_poll": run.poll_summary,
        "process_id": timeline.process_id,
        "process_creation_filetime_100ns": (
            timeline.process_creation_filetime_100ns
        ),
        "executable_sha256": timeline.executable_sha256,
        "frame_count": len(timeline.records),
    }


def analyze(
    session_root: Path,
    *,
    mask_anchor_update: int,
    delays_ns: Sequence[int],
) -> dict[str, Any]:
    if mask_anchor_update < 0:
        _fail("poll_alignment_mask_anchor_invalid")
    delays = sorted(set(delays_ns))
    if (
        not delays
        or len(delays) > 101
        or delays[0] < 0
        or delays[-1] > 5_000_000
    ):
        _fail("poll_alignment_delay_sweep_invalid")
    session_root = session_root.resolve()
    left_timeline = _load_run(session_root / "run-r1")
    right_timeline = _load_run(session_root / "run-r2")
    if (
        left_timeline.width,
        left_timeline.height,
        left_timeline.executable_sha256,
    ) != (
        right_timeline.width,
        right_timeline.height,
        right_timeline.executable_sha256,
    ):
        _fail("poll_alignment_run_identity_mismatch")
    left_raw, left_metadata, left_raw_sha = _load_raw(left_timeline)
    right_raw, right_metadata, right_raw_sha = _load_raw(right_timeline)
    left_stable = _stable_sequences(left_timeline)
    right_stable = _stable_sequences(right_timeline)
    try:
        mask, pixel_count, left_sequence, right_sequence = (
            _derive_anchor_mask(
                left_raw,
                right_raw,
                left_stable.get(mask_anchor_update, ()),
                right_stable.get(mask_anchor_update, ()),
            )
        )
    except ValueError:
        _fail("poll_alignment_mask_derivation_failed")
    left = _masked_run(
        left_timeline,
        left_raw,
        left_metadata,
        left_raw_sha,
        mask,
    )
    right = _masked_run(
        right_timeline,
        right_raw,
        right_metadata,
        right_raw_sha,
        mask,
    )
    left_collapsed = _collapse(left.masked_digests)
    right_collapsed = _collapse(right.masked_digests)
    matcher = SequenceMatcher(
        None,
        [row.digest for row in left_collapsed],
        [row.digest for row in right_collapsed],
        autojunk=False,
    )
    delay_rows: list[dict[str, Any]] = []
    for delay in delays:
        delay_rows.append(
            analyze_delay(
                delay_ns=delay,
                left_records=left.timeline.records,
                right_records=right.timeline.records,
                left_digests=left.masked_digests,
                right_digests=right.masked_digests,
                left_draws=_assigned_draws(left, delay),
                right_draws=_assigned_draws(right, delay),
                matcher=matcher,
                left_collapsed=left_collapsed,
                right_collapsed=right_collapsed,
            )
        )
    maximum_exact = max(row["exact_hash_overlap_count"] for row in delay_rows)
    minimum_unresolved = min(
        row["unresolved_pair_count"]
        for row in delay_rows
        if row["exact_hash_overlap_count"] == maximum_exact
    )
    best_rows = [
        row
        for row in delay_rows
        if row["exact_hash_overlap_count"] == maximum_exact
        and row["unresolved_pair_count"] == minimum_unresolved
    ]
    ys, xs = np.nonzero(mask)
    plan_path = session_root / "plan.json"
    collection_path = session_root / "collection.json"
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "classification": (
            "diagnostic-post-hoc-delay-sweep-not-pc-golden"
        ),
        "gate_effect": "none",
        "status": (
            "OBSERVED_POLL_ALIGNMENT_EXACT"
            if minimum_unresolved == 0
            else "OBSERVED_POLL_ALIGNMENT_INCOMPLETE"
        ),
        "session_root": str(session_root),
        "inputs": {
            "plan_sha256": (
                _sha256_file(plan_path) if plan_path.is_file() else None
            ),
            "collection_sha256": (
                _sha256_file(collection_path)
                if collection_path.is_file()
                else None
            ),
            "left": _input_row(left),
            "right": _input_row(right),
        },
        "mask": {
            "derivation": "minimum_same_update_difference_at_anchor",
            "anchor_update": mask_anchor_update,
            "left_sequence": left_sequence,
            "right_sequence": right_sequence,
            "pixel_count": pixel_count,
            "mask_sha256": _mask_sha256(mask),
            "coordinates_xy": [
                [int(x), int(y)] for y, x in zip(ys, xs, strict=True)
            ],
        },
        "ordered_present_visuals": {
            "left_collapsed_visual_count": len(left_collapsed),
            "right_collapsed_visual_count": len(right_collapsed),
            "matched_collapsed_visual_count": sum(
                block.size for block in matcher.get_matching_blocks()
            ),
            "sequence_matcher_ratio": matcher.ratio(),
        },
        "same_stable_update_baseline": _stable_update_baseline(
            left, right
        ),
        "delay_sweep": {
            "selection_status": (
                "post_hoc_same_session_sweep_not_preregistered"
            ),
            "candidate_count": len(delay_rows),
            "candidates": delay_rows,
            "best_observed": {
                "maximum_exact_hash_overlap_count": maximum_exact,
                "minimum_unresolved_pair_count": minimum_unresolved,
                "delays_ns": [
                    row["render_settle_delay_ns"] for row in best_rows
                ],
                "representative": best_rows[0],
            },
        },
    }


def _write_exclusive(path: Path, report: Mapping[str, Any]) -> str:
    if not path.is_absolute() or path.exists() or not path.parent.is_dir():
        _fail("poll_alignment_output_invalid")
    payload = (
        json.dumps(
            report,
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
        _fail("poll_alignment_output_write_failed")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-root", required=True, type=Path)
    parser.add_argument("--mask-anchor-update", required=True, type=int)
    parser.add_argument("--delay-start-ns", default=0, type=int)
    parser.add_argument("--delay-stop-ns", default=1_000_000, type=int)
    parser.add_argument("--delay-step-ns", default=50_000, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if (
            args.delay_start_ns < 0
            or args.delay_stop_ns < args.delay_start_ns
            or args.delay_step_ns <= 0
        ):
            _fail("poll_alignment_delay_sweep_invalid")
        delays = list(
            range(
                args.delay_start_ns,
                args.delay_stop_ns + 1,
                args.delay_step_ns,
            )
        )
        if not delays or delays[-1] != args.delay_stop_ns:
            _fail("poll_alignment_delay_sweep_invalid")
        report = analyze(
            args.session_root,
            mask_anchor_update=args.mask_anchor_update,
            delays_ns=delays,
        )
        digest = _write_exclusive(args.output.resolve(), report)
    except PollAlignmentError as error:
        print(f"poll alignment error: {error}", file=sys.stderr)
        return 2
    print(f"{args.output.resolve()}\n{digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
