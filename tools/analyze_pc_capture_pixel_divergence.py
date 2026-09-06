"""Measure the smallest same-update pixel divergence in two raw captures."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def _load_run(
    run_root: Path,
) -> tuple[np.memmap, dict[int, tuple[int, ...]], int, int]:
    metadata = _read_json(run_root / "capture" / "metadata.json")
    width = int(metadata["width"])
    height = int(metadata["height"])
    frame_count = int(metadata["frame_count"])
    update_map = _read_json(run_root / "framework-updates.json")
    records = update_map["records"]
    with (run_root / "capture" / "frames.csv").open(
        "r", encoding="ascii", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != frame_count or len(records) != frame_count:
        raise ValueError("capture timeline length mismatch")
    stable: dict[int, list[int]] = {}
    for sequence, (row, record) in enumerate(zip(rows, records, strict=True)):
        if int(row["sequence"]) != sequence or record["sequence"] != sequence:
            raise ValueError("capture sequence mismatch")
        if record["update_before"] == record["update_after"]:
            stable.setdefault(int(record["update_before"]), []).append(sequence)
    raw = np.memmap(
        run_root / "capture" / "frames.bgra.raw",
        dtype=np.uint8,
        mode="r",
        shape=(frame_count, height, width, 4),
    )
    return raw, {key: tuple(value) for key, value in stable.items()}, width, height


def _difference(
    left: np.ndarray,
    right: np.ndarray,
) -> tuple[
    int,
    float,
    int,
    list[int] | None,
    list[list[int]],
    list[list[int]],
    list[list[int]],
]:
    channel_difference = np.abs(
        left.astype(np.int16) - right.astype(np.int16)
    )
    mask = np.any(channel_difference != 0, axis=2)
    count = int(mask.sum())
    if count:
        ys, xs = np.nonzero(mask)
        bounds: list[int] | None = [
            int(xs.min()),
            int(ys.min()),
            int(xs.max()) + 1,
            int(ys.max()) + 1,
        ]
    else:
        bounds = None
    row_counts = mask.sum(axis=1)
    busiest_rows = [
        [int(index), int(row_counts[index])]
        for index in np.argsort(row_counts)[-8:][::-1]
        if row_counts[index]
    ]
    column_counts = mask.sum(axis=0)
    busiest_columns = [
        [int(index), int(column_counts[index])]
        for index in np.argsort(column_counts)[-8:][::-1]
        if column_counts[index]
    ]
    samples = (
        [
            [
                int(x),
                int(y),
                [int(value) for value in left[y, x]],
                [int(value) for value in right[y, x]],
            ]
            for y, x in zip(ys[:24], xs[:24], strict=True)
        ]
        if count
        else []
    )
    return (
        count,
        float(channel_difference.mean()),
        int(channel_difference.max()),
        bounds,
        busiest_rows,
        busiest_columns,
        samples,
    )


def _mask_sha256(mask: np.ndarray) -> str:
    packed = np.packbits(mask.reshape(-1), bitorder="little")
    return f"sha256:{hashlib.sha256(packed.tobytes()).hexdigest()}"


def _mask_bounds(mask: np.ndarray) -> list[int] | None:
    if not np.any(mask):
        return None
    ys, xs = np.nonzero(mask)
    return [
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    ]


def analyze(
    session_root: Path,
    *,
    updates: Iterable[int] | None,
    image_output: Path | None = None,
    summary_only: bool = False,
) -> dict[str, Any]:
    left, left_updates, width, height = _load_run(session_root / "run-r1")
    right, right_updates, right_width, right_height = _load_run(
        session_root / "run-r2"
    )
    if (right_width, right_height) != (width, height):
        raise ValueError("capture dimensions differ")
    selected_updates = (
        sorted(set(left_updates).intersection(right_updates))
        if updates is None
        else list(updates)
    )
    if not selected_updates:
        raise ValueError("no updates selected")
    rows: list[dict[str, Any]] = []
    first_images: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    divergent_masks: list[np.ndarray] = []
    mismatch_counts: list[int] = []
    maximum_channel_deltas: list[int] = []
    per_channel_maximum = np.zeros(4, dtype=np.int16)
    mask_hash_counts: Counter[str] = Counter()
    signed_delta_hash_counts: Counter[str] = Counter()
    comparable_updates: list[int] = []
    exact_updates: list[int] = []
    for update in selected_updates:
        candidates: list[tuple[int, int, tuple[Any, ...]]] = []
        for left_sequence in left_updates.get(update, ()):
            for right_sequence in right_updates.get(update, ()):
                difference = _difference(
                    left[left_sequence],
                    right[right_sequence],
                )
                candidates.append(
                    (left_sequence, right_sequence, difference)
                )
        if not candidates:
            rows.append({"update": update, "status": "NO_COMMON_STABLE_FRAME"})
            continue
        candidates.sort(key=lambda item: (item[2][0], item[0], item[1]))
        left_sequence, right_sequence, difference = candidates[0]
        (
            count,
            mean_delta,
            maximum_delta,
            bounds,
            busiest_rows,
            busiest_columns,
            samples,
        ) = difference
        left_frame = np.asarray(left[left_sequence])
        right_frame = np.asarray(right[right_sequence])
        signed_delta = left_frame.astype(np.int16) - right_frame.astype(
            np.int16
        )
        mask = np.any(signed_delta != 0, axis=2)
        mask_sha256 = _mask_sha256(mask)
        signed_delta_sha256 = (
            "sha256:"
            + hashlib.sha256(signed_delta.tobytes()).hexdigest()
        )
        comparable_updates.append(update)
        mismatch_counts.append(count)
        maximum_channel_deltas.append(maximum_delta)
        per_channel_maximum = np.maximum(
            per_channel_maximum,
            np.abs(signed_delta).max(axis=(0, 1)),
        )
        mask_hash_counts[mask_sha256] += 1
        signed_delta_hash_counts[signed_delta_sha256] += 1
        if count:
            divergent_masks.append(mask.copy())
        else:
            exact_updates.append(update)
        rows.append(
            {
                "update": update,
                "status": "EXACT" if count == 0 else "DIFFERENT",
                "r1_sequence": left_sequence,
                "r2_sequence": right_sequence,
                "mismatched_pixel_count": count,
                "mismatched_pixel_fraction": count / (width * height),
                "mean_absolute_bgra_delta": mean_delta,
                "maximum_channel_delta": maximum_delta,
                "maximum_absolute_delta_by_bgra_channel": [
                    int(value)
                    for value in np.abs(signed_delta).max(axis=(0, 1))
                ],
                "difference_bounds_xyxy": bounds,
                "mismatch_mask_sha256": mask_sha256,
                "signed_delta_sha256": signed_delta_sha256,
                "busiest_rows": busiest_rows,
                "busiest_columns": busiest_columns,
                "mismatch_coordinate_samples_xy": samples,
            }
        )
        if first_images is None:
            l = left_frame.copy()
            r = right_frame.copy()
            first_images = (l, r, mask)

    if image_output is not None and first_images is not None:
        if image_output.exists() or not image_output.parent.is_dir():
            raise ValueError("image output must be new in an existing directory")
        left_image, right_image, mask = first_images
        rgb = left_image[:, :, [2, 1, 0]].copy()
        rgb[mask] = np.array([255, 0, 255], dtype=np.uint8)
        side_by_side = np.concatenate(
            (
                left_image[:, :, [2, 1, 0]],
                right_image[:, :, [2, 1, 0]],
                rgb,
            ),
            axis=1,
        )
        Image.fromarray(side_by_side, mode="RGB").save(image_output)

    union_mask = np.zeros((height, width), dtype=bool)
    intersection_mask = np.zeros((height, width), dtype=bool)
    if divergent_masks:
        union_mask = np.logical_or.reduce(divergent_masks)
        intersection_mask = np.logical_and.reduce(divergent_masks)
    union_coordinates = [
        [int(x), int(y)]
        for y, x in zip(*np.nonzero(union_mask), strict=True)
    ]
    intersection_coordinates = [
        [int(x), int(y)]
        for y, x in zip(*np.nonzero(intersection_mask), strict=True)
    ]
    mismatch_histogram = Counter(mismatch_counts)
    baseline_mismatch_count = (
        mismatch_histogram.most_common(1)[0][0]
        if mismatch_histogram
        else None
    )
    outlier_updates = [
        {
            "update": row["update"],
            "r1_sequence": row["r1_sequence"],
            "r2_sequence": row["r2_sequence"],
            "mismatched_pixel_count": row["mismatched_pixel_count"],
            "maximum_channel_delta": row["maximum_channel_delta"],
            "difference_bounds_xyxy": row["difference_bounds_xyxy"],
            "mismatch_mask_sha256": row["mismatch_mask_sha256"],
        }
        for row in rows
        if row.get("mismatched_pixel_count") != baseline_mismatch_count
    ]
    aggregate = {
        "requested_update_count": len(selected_updates),
        "comparable_update_count": len(comparable_updates),
        "first_comparable_update": (
            min(comparable_updates) if comparable_updates else None
        ),
        "last_comparable_update": (
            max(comparable_updates) if comparable_updates else None
        ),
        "exact_update_count": len(exact_updates),
        "divergent_update_count": len(divergent_masks),
        "mismatched_pixel_count_minimum": (
            min(mismatch_counts) if mismatch_counts else None
        ),
        "mismatched_pixel_count_maximum": (
            max(mismatch_counts) if mismatch_counts else None
        ),
        "mismatched_pixel_count_histogram": [
            [count, frequency]
            for count, frequency in sorted(mismatch_histogram.items())
        ],
        "modal_mismatched_pixel_count": baseline_mismatch_count,
        "modal_mismatch_update_count": (
            mismatch_histogram[baseline_mismatch_count]
            if baseline_mismatch_count is not None
            else 0
        ),
        "nonmodal_mismatch_updates": outlier_updates,
        "maximum_channel_delta_minimum": (
            min(maximum_channel_deltas)
            if maximum_channel_deltas
            else None
        ),
        "maximum_channel_delta_maximum": (
            max(maximum_channel_deltas)
            if maximum_channel_deltas
            else None
        ),
        "maximum_absolute_delta_by_bgra_channel": [
            int(value) for value in per_channel_maximum
        ],
        "unique_mismatch_mask_count": len(mask_hash_counts),
        "mismatch_mask_histogram": [
            [digest, frequency]
            for digest, frequency in mask_hash_counts.most_common(16)
        ],
        "all_selected_masks_identical": len(mask_hash_counts) == 1,
        "unique_signed_delta_count": len(signed_delta_hash_counts),
        "mismatch_union_pixel_count": int(union_mask.sum()),
        "mismatch_union_bounds_xyxy": _mask_bounds(union_mask),
        "mismatch_union_mask_sha256": _mask_sha256(union_mask),
        "mismatch_union_coordinates_xy": (
            union_coordinates if len(union_coordinates) <= 256 else None
        ),
        "mismatch_intersection_pixel_count": int(
            intersection_mask.sum()
        ),
        "mismatch_intersection_bounds_xyxy": _mask_bounds(
            intersection_mask
        ),
        "mismatch_intersection_mask_sha256": _mask_sha256(
            intersection_mask
        ),
        "mismatch_intersection_coordinates_xy": (
            intersection_coordinates
            if len(intersection_coordinates) <= 256
            else None
        ),
    }
    return {
        "schema": "zuma-rl.pc-capture-pixel-divergence",
        "version": 2,
        "session_root": str(session_root.resolve()),
        "width": width,
        "height": height,
        "aggregate": aggregate,
        "updates": [] if summary_only else rows,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_root", type=Path)
    update_group = parser.add_mutually_exclusive_group(required=True)
    update_group.add_argument("--update", action="append", type=int)
    update_group.add_argument("--all-common", action="store_true")
    parser.add_argument("--image-output", type=Path)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args(argv)
    report = analyze(
        args.session_root.resolve(),
        updates=None if args.all_common else args.update,
        image_output=(
            None if args.image_output is None else args.image_output.resolve()
        ),
        summary_only=args.summary_only,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
