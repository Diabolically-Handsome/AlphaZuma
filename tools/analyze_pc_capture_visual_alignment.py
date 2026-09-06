"""Measure exact cross-run visual alignment near each framework update."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


_DIGEST_DOMAIN = b"zuma-rl.masked-bgr-frame.v1\0"


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


def _mask_sha256(mask: np.ndarray) -> str:
    packed = np.packbits(mask.reshape(-1), bitorder="little")
    return f"sha256:{hashlib.sha256(packed.tobytes()).hexdigest()}"


def _derive_anchor_mask(
    left: np.memmap,
    right: np.memmap,
    left_sequences: tuple[int, ...],
    right_sequences: tuple[int, ...],
) -> tuple[np.ndarray, int, int, int]:
    candidates: list[tuple[int, int, int, np.ndarray]] = []
    for left_sequence in left_sequences:
        for right_sequence in right_sequences:
            mask = np.any(
                left[left_sequence] != right[right_sequence], axis=2
            )
            candidates.append(
                (
                    int(mask.sum()),
                    left_sequence,
                    right_sequence,
                    mask,
                )
            )
    if not candidates:
        raise ValueError("mask anchor has no common stable frames")
    candidates.sort(key=lambda row: row[:3])
    count, left_sequence, right_sequence, mask = candidates[0]
    if count <= 0 or count > 256:
        raise ValueError("mask anchor is not a narrow pixel mask")
    alpha_delta = np.abs(
        left[left_sequence, :, :, 3].astype(np.int16)
        - right[right_sequence, :, :, 3].astype(np.int16)
    )
    if np.any(alpha_delta):
        raise ValueError("mask anchor contains alpha divergence")
    return mask.copy(), count, left_sequence, right_sequence


def _masked_digest(frame: np.ndarray, mask: np.ndarray) -> str:
    bgr = np.asarray(frame[:, :, :3]).copy()
    bgr[mask] = 0
    digest = hashlib.sha256(_DIGEST_DOMAIN)
    digest.update(np.asarray(bgr.shape, dtype="<u4").tobytes())
    digest.update(bgr.tobytes())
    return f"sha256:{digest.hexdigest()}"


def _digest_timeline(
    raw: np.memmap,
    stable: dict[int, tuple[int, ...]],
    mask: np.ndarray,
) -> dict[int, tuple[tuple[int, str], ...]]:
    return {
        update: tuple(
            (sequence, _masked_digest(raw[sequence], mask))
            for sequence in sequences
        )
        for update, sequences in stable.items()
    }


def _collapsed_digest_sequence(
    raw: np.memmap,
    mask: np.ndarray,
) -> list[dict[str, Any]]:
    collapsed: list[dict[str, Any]] = []
    for sequence in range(raw.shape[0]):
        digest = _masked_digest(raw[sequence], mask)
        if collapsed and collapsed[-1]["masked_bgr_sha256"] == digest:
            collapsed[-1]["last_sequence"] = sequence
            collapsed[-1]["frame_count"] += 1
            continue
        collapsed.append(
            {
                "masked_bgr_sha256": digest,
                "first_sequence": sequence,
                "last_sequence": sequence,
                "frame_count": 1,
            }
        )
    return collapsed


def _compare_collapsed_sequences(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
) -> dict[str, Any]:
    left_digests = [row["masked_bgr_sha256"] for row in left]
    right_digests = [row["masked_bgr_sha256"] for row in right]
    matcher = SequenceMatcher(
        None,
        left_digests,
        right_digests,
        autojunk=False,
    )
    opcodes = matcher.get_opcodes()
    non_equal = [
        {
            "tag": tag,
            "r1_collapsed_range": [left_start, left_end],
            "r2_collapsed_range": [right_start, right_end],
            "r1_frame_range": (
                [
                    left[left_start]["first_sequence"],
                    left[left_end - 1]["last_sequence"],
                ]
                if left_start < left_end
                else None
            ),
            "r2_frame_range": (
                [
                    right[right_start]["first_sequence"],
                    right[right_end - 1]["last_sequence"],
                ]
                if right_start < right_end
                else None
            ),
        }
        for tag, left_start, left_end, right_start, right_end in opcodes
        if tag != "equal"
    ]
    matched = sum(
        left_end - left_start
        for tag, left_start, left_end, unused_start, unused_end in opcodes
        if tag == "equal"
    )
    return {
        "r1_frame_count": sum(row["frame_count"] for row in left),
        "r2_frame_count": sum(row["frame_count"] for row in right),
        "r1_collapsed_visual_count": len(left),
        "r2_collapsed_visual_count": len(right),
        "collapsed_sequences_exactly_equal": left_digests == right_digests,
        "matched_collapsed_visual_count": matched,
        "sequence_matcher_ratio": matcher.ratio(),
        "non_equal_opcode_count": len(non_equal),
        "non_equal_opcodes": non_equal[:32],
    }


def _align_direction(
    source: dict[int, tuple[tuple[int, str], ...]],
    target: dict[int, tuple[tuple[int, str], ...]],
    updates: list[int],
    radius: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    unmatched: list[int] = []
    offsets: Counter[int] = Counter()
    selected_target_updates: list[int] = []
    for update in updates:
        candidates: list[tuple[int, int, int, int, str]] = []
        for source_sequence, source_digest in source.get(update, ()):
            for offset in range(-radius, radius + 1):
                target_update = update + offset
                for target_sequence, target_digest in target.get(
                    target_update, ()
                ):
                    if source_digest == target_digest:
                        candidates.append(
                            (
                                abs(offset),
                                offset,
                                source_sequence,
                                target_sequence,
                                source_digest,
                            )
                        )
        if not candidates:
            unmatched.append(update)
            continue
        candidates.sort()
        _, offset, source_sequence, target_sequence, digest = candidates[0]
        target_update = update + offset
        offsets[offset] += 1
        selected_target_updates.append(target_update)
        rows.append(
            {
                "source_update": update,
                "target_update": target_update,
                "update_offset": offset,
                "source_sequence": source_sequence,
                "target_sequence": target_sequence,
                "masked_bgr_sha256": digest,
            }
        )
    monotonic_violations = sum(
        right < left
        for left, right in zip(
            selected_target_updates,
            selected_target_updates[1:],
            strict=False,
        )
    )
    return {
        "requested_update_count": len(updates),
        "exactly_aligned_update_count": len(rows),
        "unmatched_update_count": len(unmatched),
        "unmatched_updates": unmatched,
        "offset_histogram": [
            [offset, count] for offset, count in sorted(offsets.items())
        ],
        "selected_target_update_monotonic_violations": monotonic_violations,
        "nonzero_offset_rows": [
            row for row in rows if row["update_offset"] != 0
        ],
    }


def analyze(
    session_root: Path,
    *,
    mask_anchor_update: int,
    neighbor_radius: int,
    mask_session_root: Path | None = None,
) -> dict[str, Any]:
    if mask_anchor_update < 0:
        raise ValueError("mask anchor update must be nonnegative")
    if neighbor_radius < 0 or neighbor_radius > 8:
        raise ValueError("neighbor radius must be in [0, 8]")
    left, left_stable, width, height = _load_run(session_root / "run-r1")
    right, right_stable, right_width, right_height = _load_run(
        session_root / "run-r2"
    )
    if (right_width, right_height) != (width, height):
        raise ValueError("capture dimensions differ")
    mask_left = left
    mask_right = right
    mask_left_stable = left_stable
    mask_right_stable = right_stable
    resolved_mask_root = session_root.resolve()
    if mask_session_root is not None:
        resolved_mask_root = mask_session_root.resolve()
        (
            mask_left,
            mask_left_stable,
            mask_width,
            mask_height,
        ) = _load_run(resolved_mask_root / "run-r1")
        (
            mask_right,
            mask_right_stable,
            mask_right_width,
            mask_right_height,
        ) = _load_run(resolved_mask_root / "run-r2")
        if (
            mask_width,
            mask_height,
            mask_right_width,
            mask_right_height,
        ) != (width, height, width, height):
            raise ValueError("mask-source capture dimensions differ")
    mask, mask_count, left_sequence, right_sequence = _derive_anchor_mask(
        mask_left,
        mask_right,
        mask_left_stable.get(mask_anchor_update, ()),
        mask_right_stable.get(mask_anchor_update, ()),
    )
    left_timeline = _digest_timeline(left, left_stable, mask)
    right_timeline = _digest_timeline(right, right_stable, mask)
    left_collapsed = _collapsed_digest_sequence(left, mask)
    right_collapsed = _collapsed_digest_sequence(right, mask)
    common_updates = sorted(set(left_timeline).intersection(right_timeline))
    if not common_updates:
        raise ValueError("captures have no common stable updates")
    ys, xs = np.nonzero(mask)
    coordinates = [
        [int(x), int(y)]
        for y, x in zip(ys, xs, strict=True)
    ]
    return {
        "schema": "zuma-rl.pc-capture-visual-alignment-diagnostic",
        "version": 1,
        "session_root": str(session_root.resolve()),
        "width": width,
        "height": height,
        "common_stable_update_count": len(common_updates),
        "first_common_stable_update": common_updates[0],
        "last_common_stable_update": common_updates[-1],
        "neighbor_radius": neighbor_radius,
        "mask": {
            "derivation": (
                "minimum_same_update_difference_at_anchor"
                if mask_session_root is None
                else "minimum_same_update_difference_at_external_anchor"
            ),
            "source_session_root": str(resolved_mask_root),
            "anchor_update": mask_anchor_update,
            "r1_sequence": left_sequence,
            "r2_sequence": right_sequence,
            "pixel_count": mask_count,
            "mask_sha256": _mask_sha256(mask),
            "coordinates_xy": coordinates,
        },
        "ordered_present_sequence": _compare_collapsed_sequences(
            left_collapsed,
            right_collapsed,
        ),
        "r1_to_r2": _align_direction(
            left_timeline,
            right_timeline,
            common_updates,
            neighbor_radius,
        ),
        "r2_to_r1": _align_direction(
            right_timeline,
            left_timeline,
            common_updates,
            neighbor_radius,
        ),
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_root", type=Path)
    parser.add_argument("--mask-anchor-update", required=True, type=int)
    parser.add_argument(
        "--mask-session-root",
        type=Path,
        help=(
            "Derive the narrow compositor mask from another same-geometry "
            "two-run session instead of from the session under analysis."
        ),
    )
    parser.add_argument("--neighbor-radius", default=2, type=int)
    args = parser.parse_args(argv)
    report = analyze(
        args.session_root.resolve(),
        mask_anchor_update=args.mask_anchor_update,
        neighbor_radius=args.neighbor_radius,
        mask_session_root=(
            None
            if args.mask_session_root is None
            else args.mask_session_root.resolve()
        ),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
