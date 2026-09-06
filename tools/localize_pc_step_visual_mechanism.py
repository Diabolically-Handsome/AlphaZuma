"""Localize a validated exact-step visual pair against frozen mask evidence.

This is a post-hoc mechanism diagnostic.  It recomputes the frozen v2 pair
comparison, verifies the independently calibrated C65 pixel mask, measures
whether the per-tick mismatch coordinates are stable, and compares one target
tick with declared reference images.  It never grants PC Golden credit.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import numpy as np
from PIL import Image

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tools.compare_pc_step_visual_trajectories import (
    EXPECTED_HEIGHT,
    EXPECTED_WIDTH,
    _decode_bgra_bmp,
    _load_run,
    compare,
)


SCHEMA = "zuma-rl.pc-step-visual-mechanism-localization"
VERSION = 1


class MechanismLocalizationError(ValueError):
    """A bound comparison, mask, frame, or output is invalid."""


def _fail(code: str) -> None:
    raise MechanismLocalizationError(code)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise MechanismLocalizationError(
            "mechanism_localization_artifact_unreadable"
        ) from error
    return f"sha256:{digest.hexdigest()}"


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")


def _json_object(path: Path) -> Mapping[str, Any]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload.decode("ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MechanismLocalizationError(
            "mechanism_localization_json_invalid"
        ) from error
    if not isinstance(value, dict):
        _fail("mechanism_localization_json_root_invalid")
    return value


def _mask_sha256(mask: np.ndarray[Any, np.dtype[np.bool_]]) -> str:
    packed = np.packbits(mask.reshape(-1), bitorder="little")
    return f"sha256:{hashlib.sha256(packed.tobytes()).hexdigest()}"


def _coordinates(mask: np.ndarray[Any, np.dtype[np.bool_]]) -> set[tuple[int, int]]:
    ys, xs = np.nonzero(mask)
    return {(int(x), int(y)) for y, x in zip(ys, xs, strict=True)}


def _mask_from_coordinates(value: Any) -> np.ndarray[Any, np.dtype[np.bool_]]:
    if not isinstance(value, list):
        _fail("mechanism_localization_mask_coordinates_invalid")
    mask = np.zeros((EXPECTED_HEIGHT, EXPECTED_WIDTH), dtype=np.bool_)
    for coordinate in value:
        if (
            not isinstance(coordinate, list)
            or len(coordinate) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) for item in coordinate)
        ):
            _fail("mechanism_localization_mask_coordinates_invalid")
        x, y = coordinate
        if not 0 <= x < EXPECTED_WIDTH or not 0 <= y < EXPECTED_HEIGHT:
            _fail("mechanism_localization_mask_coordinates_invalid")
        if mask[y, x]:
            _fail("mechanism_localization_mask_coordinate_duplicate")
        mask[y, x] = True
    return mask


def _rgb_image(path: Path) -> np.ndarray[Any, np.dtype[np.uint8]]:
    try:
        with Image.open(path) as source:
            image = np.asarray(source.convert("RGB"), dtype=np.uint8)
    except (OSError, ValueError) as error:
        raise MechanismLocalizationError(
            "mechanism_localization_reference_image_invalid"
        ) from error
    if image.shape != (EXPECTED_HEIGHT, EXPECTED_WIDTH, 3):
        _fail("mechanism_localization_reference_shape_invalid")
    return image


def _snapshot_rgb(snapshot: Mapping[str, Any]) -> np.ndarray[Any, np.dtype[np.uint8]]:
    bgra = _decode_bgra_bmp(Path(str(snapshot["resolved_artifact"])))
    return bgra[:, :, (2, 1, 0)]


def _difference(
    left: np.ndarray[Any, np.dtype[np.uint8]],
    right: np.ndarray[Any, np.dtype[np.uint8]],
) -> tuple[np.ndarray[Any, np.dtype[np.bool_]], dict[str, Any]]:
    if left.shape != right.shape or left.shape != (
        EXPECTED_HEIGHT,
        EXPECTED_WIDTH,
        3,
    ):
        _fail("mechanism_localization_frame_shape_mismatch")
    mismatch = np.any(left != right, axis=2)
    count = int(np.count_nonzero(mismatch))
    if count:
        ys, xs = np.nonzero(mismatch)
        bounds = [
            int(xs.min()),
            int(ys.min()),
            int(xs.max()) + 1,
            int(ys.max()) + 1,
        ]
    else:
        bounds = None
    delta = np.abs(left.astype(np.int16) - right.astype(np.int16))
    return mismatch, {
        "mismatched_rgb_pixel_count": count,
        "difference_bounds_xyxy": bounds,
        "maximum_absolute_delta_by_rgb_channel": [
            int(item) for item in np.max(delta, axis=(0, 1)).tolist()
        ],
        "mismatch_mask_sha256": _mask_sha256(mismatch),
    }


def _dominant_rgb(image: np.ndarray[Any, np.dtype[np.uint8]]) -> dict[str, Any]:
    rows = image.reshape(-1, 3)
    counts = Counter(map(tuple, rows.tolist()))
    colour, count = counts.most_common(1)[0]
    return {
        "rgb": [int(item) for item in colour],
        "pixel_count": count,
        "fraction": count / (EXPECTED_WIDTH * EXPECTED_HEIGHT),
    }


def _reference_row(
    *,
    run_image: np.ndarray[Any, np.dtype[np.uint8]],
    reference_image: np.ndarray[Any, np.dtype[np.uint8]],
    frozen_mask: np.ndarray[Any, np.dtype[np.bool_]],
    observed_mask: np.ndarray[Any, np.dtype[np.bool_]],
) -> dict[str, Any]:
    mismatch, summary = _difference(run_image, reference_image)
    return {
        **summary,
        "outside_frozen_mask_pixel_count": int(
            np.count_nonzero(mismatch & ~frozen_mask)
        ),
        "outside_observed_stable_mask_pixel_count": int(
            np.count_nonzero(mismatch & ~observed_mask)
        ),
    }


def localize(
    *,
    left_index: Path,
    right_index: Path,
    comparison_path: Path,
    frozen_mask_preregistration: Path,
    target_update: int,
    references: Mapping[str, Path],
) -> Mapping[str, Any]:
    left = _load_run(left_index)
    right = _load_run(right_index)
    if (
        left.start_update != right.start_update
        or left.end_update != right.end_update
        or not left.start_update <= target_update <= left.end_update
    ):
        _fail("mechanism_localization_update_range_invalid")

    declared_comparison = _json_object(comparison_path)
    recomputed_comparison = compare(left_index, right_index)
    if declared_comparison != recomputed_comparison:
        _fail("mechanism_localization_comparison_recompute_mismatch")
    if (
        declared_comparison.get("schema")
        != "zuma-rl.pc-step-visual-trajectory-comparison"
        or declared_comparison.get("version") != 2
    ):
        _fail("mechanism_localization_comparison_schema_invalid")

    preregistration = _json_object(frozen_mask_preregistration)
    mask_record = preregistration.get("comparison", {}).get("mask")
    if (
        preregistration.get("schema")
        != "zuma-rl.pc-render-transport-preregistration"
        or preregistration.get("version") != 1
        or not isinstance(mask_record, dict)
        or mask_record.get("width") != EXPECTED_WIDTH
        or mask_record.get("height") != EXPECTED_HEIGHT
    ):
        _fail("mechanism_localization_frozen_mask_invalid")
    frozen_mask = _mask_from_coordinates(mask_record.get("coordinates_xy"))
    if (
        mask_record.get("pixel_count") != int(np.count_nonzero(frozen_mask))
        or mask_record.get("mask_sha256") != _mask_sha256(frozen_mask)
    ):
        _fail("mechanism_localization_frozen_mask_binding_invalid")

    mismatch_masks: list[np.ndarray[Any, np.dtype[np.bool_]]] = []
    tick_rows: list[dict[str, Any]] = []
    left_images: list[np.ndarray[Any, np.dtype[np.uint8]]] = []
    right_images: list[np.ndarray[Any, np.dtype[np.uint8]]] = []
    for offset, update in enumerate(range(left.start_update, left.end_update + 1)):
        left_image = _snapshot_rgb(left.snapshots[offset])
        right_image = _snapshot_rgb(right.snapshots[offset])
        mismatch, summary = _difference(left_image, right_image)
        left_images.append(left_image)
        right_images.append(right_image)
        mismatch_masks.append(mismatch)
        tick_rows.append({"update": update, **summary})

    post_initial_masks = mismatch_masks[1:]
    stable_after_initial = bool(post_initial_masks) and all(
        np.array_equal(post_initial_masks[0], candidate)
        for candidate in post_initial_masks[1:]
    )
    observed_mask = (
        post_initial_masks[0]
        if stable_after_initial
        else np.logical_or.reduce(post_initial_masks)
    )
    frozen_coordinates = _coordinates(frozen_mask)
    observed_coordinates = _coordinates(observed_mask)
    extra_coordinates = sorted(observed_coordinates - frozen_coordinates)
    missing_coordinates = sorted(frozen_coordinates - observed_coordinates)

    target_offset = target_update - left.start_update
    reference_rows: dict[str, Any] = {}
    for label, path in sorted(references.items()):
        reference = _rgb_image(path)
        reference_rows[label] = {
            "artifact": str(path.resolve()),
            "artifact_sha256": _sha256_path(path),
            "left": _reference_row(
                run_image=left_images[target_offset],
                reference_image=reference,
                frozen_mask=frozen_mask,
                observed_mask=observed_mask,
            ),
            "right": _reference_row(
                run_image=right_images[target_offset],
                reference_image=reference,
                frozen_mask=frozen_mask,
                observed_mask=observed_mask,
            ),
        }

    return {
        "schema": SCHEMA,
        "version": VERSION,
        "classification": "posthoc-mechanism-localization-not-pc-golden",
        "status": "PASS",
        "gate_effect": "none",
        "inputs": {
            "left_index": {
                "artifact": str(left_index.resolve()),
                "sha256": _sha256_path(left_index),
            },
            "right_index": {
                "artifact": str(right_index.resolve()),
                "sha256": _sha256_path(right_index),
            },
            "comparison": {
                "artifact": str(comparison_path.resolve()),
                "sha256": _sha256_path(comparison_path),
                "recomputed_exactly": True,
            },
            "frozen_mask_preregistration": {
                "artifact": str(frozen_mask_preregistration.resolve()),
                "sha256": _sha256_path(frozen_mask_preregistration),
            },
        },
        "protocol": {
            "start_update": left.start_update,
            "end_update": left.end_update,
            "tick_count": len(left.frames),
            "target_update": target_update,
        },
        "memory_and_render_state": {
            "normalized_rng_trajectory_exact": declared_comparison["memory"][
                "normalized_rng_trajectory_exact"
            ],
            "render_state_exact": declared_comparison["render_state"]["exact"],
            "distinct_process_ids": declared_comparison["independence"][
                "distinct_process_ids"
            ],
        },
        "initial_capture": {
            "update": left.start_update,
            "left_dominant_rgb": _dominant_rgb(left_images[0]),
            "right_dominant_rgb": _dominant_rgb(right_images[0]),
            "pair_difference": tick_rows[0],
            "interpretation": "cold_capture_diagnostic_only",
        },
        "post_initial_pair": {
            "update_range": [left.start_update + 1, left.end_update],
            "tick_count": len(post_initial_masks),
            "mismatch_coordinate_set_exactly_stable": stable_after_initial,
            "observed_mask_pixel_count": int(np.count_nonzero(observed_mask)),
            "observed_mask_sha256": _mask_sha256(observed_mask),
            "frozen_mask_pixel_count": int(np.count_nonzero(frozen_mask)),
            "frozen_mask_sha256": _mask_sha256(frozen_mask),
            "observed_is_frozen_mask_superset": not missing_coordinates,
            "extra_coordinates_xy": [list(item) for item in extra_coordinates],
            "missing_coordinates_xy": [list(item) for item in missing_coordinates],
            "ticks": tick_rows[1:],
        },
        "target_reference_comparisons": reference_rows,
        "non_authorizations": [
            "pc_golden",
            "mask_expansion",
            "c78_reclassification",
            "fidelity_gate_open",
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-index", required=True, type=Path)
    parser.add_argument("--right-index", required=True, type=Path)
    parser.add_argument("--comparison", required=True, type=Path)
    parser.add_argument(
        "--frozen-mask-preregistration",
        required=True,
        type=Path,
    )
    parser.add_argument("--target-update", required=True, type=int)
    parser.add_argument(
        "--reference",
        action="append",
        default=[],
        metavar="LABEL=PATH",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser


def _references(values: list[str]) -> Mapping[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        label, separator, member = value.partition("=")
        if (
            separator != "="
            or not label
            or label in result
            or not member
            or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in label)
        ):
            _fail("mechanism_localization_reference_argument_invalid")
        result[label] = Path(member).resolve()
    if not result:
        _fail("mechanism_localization_references_empty")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        output = args.output.resolve()
        if output.exists() or not output.is_absolute() or not output.parent.is_dir():
            _fail("mechanism_localization_output_invalid")
        report = localize(
            left_index=args.left_index.resolve(),
            right_index=args.right_index.resolve(),
            comparison_path=args.comparison.resolve(),
            frozen_mask_preregistration=(
                args.frozen_mask_preregistration.resolve()
            ),
            target_update=args.target_update,
            references=_references(args.reference),
        )
        payload = _canonical_bytes(report)
        with output.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except (MechanismLocalizationError, OSError) as error:
        print(f"mechanism localization error: {error}", file=sys.stderr)
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
