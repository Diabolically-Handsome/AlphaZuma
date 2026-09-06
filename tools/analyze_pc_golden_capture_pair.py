"""Compare stable native-update frame candidates from two PC Golden runs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class StableFrame:
    sequence: int
    update: int
    frame_sha256: str


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} is not a JSON object")
    return value


def _stable_frames(run_root: Path) -> tuple[StableFrame, ...]:
    update_map = _read_json(run_root / "framework-updates.json")
    records = update_map.get("records")
    if not isinstance(records, list):
        raise ValueError("framework update records are unavailable")

    with (run_root / "capture" / "frames.csv").open(
        "r",
        encoding="ascii",
        newline="",
    ) as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(records):
        raise ValueError("capture and framework-update lengths differ")

    stable: list[StableFrame] = []
    for expected_sequence, (row, record) in enumerate(
        zip(rows, records, strict=True)
    ):
        if (
            not isinstance(record, dict)
            or int(row["sequence"]) != expected_sequence
            or record.get("sequence") != expected_sequence
        ):
            raise ValueError("capture sequence identity mismatch")
        update_before = record.get("update_before")
        update_after = record.get("update_after")
        if (
            isinstance(update_before, int)
            and not isinstance(update_before, bool)
            and update_before == update_after
        ):
            stable.append(
                StableFrame(
                    sequence=expected_sequence,
                    update=update_before,
                    frame_sha256=row["frame_sha256"],
                )
            )
    return tuple(stable)


def _by_update(
    frames: tuple[StableFrame, ...],
) -> dict[int, tuple[StableFrame, ...]]:
    grouped: dict[int, list[StableFrame]] = {}
    for frame in frames:
        grouped.setdefault(frame.update, []).append(frame)
    return {update: tuple(items) for update, items in grouped.items()}


def _contiguous_ranges(values: Iterable[int]) -> tuple[tuple[int, int], ...]:
    ordered = sorted(set(values))
    if not ordered:
        return ()
    ranges: list[tuple[int, int]] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value != previous + 1:
            ranges.append((start, previous))
            start = value
        previous = value
    ranges.append((start, previous))
    return tuple(ranges)


def analyze(session_root: Path) -> dict[str, Any]:
    collection = _read_json(session_root / "collection.json")
    if collection.get("status") != "complete":
        raise ValueError("collection is not complete")

    left = _stable_frames(session_root / "run-r1")
    right = _stable_frames(session_root / "run-r2")
    left_updates = _by_update(left)
    right_updates = _by_update(right)
    common_updates = sorted(set(left_updates) & set(right_updates))

    matching_updates: list[int] = []
    matching_pairs = 0
    first_mismatch: dict[str, Any] | None = None
    for update in common_updates:
        left_hashes = {
            frame.frame_sha256: frame.sequence
            for frame in left_updates[update]
        }
        right_hashes = {
            frame.frame_sha256: frame.sequence
            for frame in right_updates[update]
        }
        shared = sorted(set(left_hashes) & set(right_hashes))
        if shared:
            matching_updates.append(update)
            matching_pairs += len(shared)
        elif first_mismatch is None:
            first_mismatch = {
                "update": update,
                "r1_sequences": [
                    frame.sequence for frame in left_updates[update]
                ],
                "r2_sequences": [
                    frame.sequence for frame in right_updates[update]
                ],
            }

    matching_ranges = _contiguous_ranges(matching_updates)
    longest_matching_range = max(
        matching_ranges,
        key=lambda item: item[1] - item[0] + 1,
        default=None,
    )
    return {
        "schema": "zuma-rl.pc-golden-capture-pair-analysis",
        "version": 1,
        "session_root": str(session_root),
        "r1": {
            "stable_frames": len(left),
            "stable_updates": len(left_updates),
            "first_update": min(left_updates),
            "last_update": max(left_updates),
        },
        "r2": {
            "stable_frames": len(right),
            "stable_updates": len(right_updates),
            "first_update": min(right_updates),
            "last_update": max(right_updates),
        },
        "common_stable_updates": len(common_updates),
        "common_stable_update_range": [
            min(common_updates),
            max(common_updates),
        ],
        "updates_with_exact_frame_match": len(matching_updates),
        "exact_matching_frame_hashes": matching_pairs,
        "contiguous_exact_match_ranges": [
            {
                "first_update": start,
                "last_update": end,
                "tick_count": end - start + 1,
            }
            for start, end in matching_ranges
        ],
        "longest_contiguous_exact_match_range": (
            {
                "first_update": longest_matching_range[0],
                "last_update": longest_matching_range[1],
                "tick_count": (
                    longest_matching_range[1]
                    - longest_matching_range[0]
                    + 1
                ),
            }
            if longest_matching_range is not None
            else None
        ),
        "all_common_updates_have_exact_frame_match": (
            len(matching_updates) == len(common_updates)
        ),
        "first_common_update_without_exact_match": first_mismatch,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare exact frame hashes among stable native-update samples."
        )
    )
    parser.add_argument("session_root", type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = analyze(args.session_root.resolve())
    print(
        json.dumps(
            result,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
