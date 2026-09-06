"""Cross-check retail Ball memory state against the frozen gameplay pixels.

The memory probe records each fixed-size retail ``Ball`` object from the
active curve list.  This validator samples an interior disk around every
unoccluded center, classifies saturated pixels by hue, and requires the
rendered color to agree with the Ball ``color_id``.  It is deliberately a
measurement-provenance check, not a computer-vision source of game state.
"""

from __future__ import annotations

import argparse
import colorsys
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
from typing import Any, Iterable

from PIL import Image

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from zuma_rl.pc_memory_evidence import (
    ACTIVE_CHAIN_LIST_OFFSET,
    BALL_OBJECT_SIZE,
    BALL_VTABLE,
    BULLET_OBJECT_SIZE,
    BULLET_VTABLE,
    PcMemoryEvidenceError,
    validate_probe_pixels,
)

SAMPLE_RADIUS = 14.5
MINIMUM_SATURATION = 0.35
MINIMUM_VALUE = 0.20
HUE_KERNEL_WIDTH = 0.09
ROLLOUT_FOREGROUND_OCCLUSION_X = 80.0
ROLLOUT_FOREGROUND_OCCLUSION_Y = 100.0
PROTOTYPE_HUES = {
    0: 0.610,  # blue
    1: 0.145,  # yellow
    2: 0.000,  # red
    3: 0.330,  # green
    4: 0.800,  # purple
}


class PixelValidationError(RuntimeError):
    """Raised when the memory/pixel cross-check is not admissible."""


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _circular_distance(left: float, right: float) -> float:
    distance = abs(left - right)
    return min(distance, 1.0 - distance)


def _classify_patch(
    pixels: Any,
    *,
    width: int,
    height: int,
    center_x: float,
    center_y: float,
) -> tuple[int, int, dict[int, float]]:
    scores = {color_id: 0.0 for color_id in PROTOTYPE_HUES}
    used_pixels = 0
    left = max(0, math.floor(center_x - SAMPLE_RADIUS))
    right = min(width - 1, math.ceil(center_x + SAMPLE_RADIUS))
    top = max(0, math.floor(center_y - SAMPLE_RADIUS))
    bottom = min(height - 1, math.ceil(center_y + SAMPLE_RADIUS))
    radius_squared = SAMPLE_RADIUS * SAMPLE_RADIUS
    for y in range(top, bottom + 1):
        for x in range(left, right + 1):
            if (x - center_x) ** 2 + (y - center_y) ** 2 > radius_squared:
                continue
            red, green, blue = pixels[x, y]
            hue, saturation, value = colorsys.rgb_to_hsv(
                red / 255.0,
                green / 255.0,
                blue / 255.0,
            )
            if (
                saturation < MINIMUM_SATURATION
                or value < MINIMUM_VALUE
            ):
                continue
            used_pixels += 1
            weight = saturation * value
            for color_id, prototype in PROTOTYPE_HUES.items():
                distance = _circular_distance(hue, prototype)
                scores[color_id] += weight * math.exp(
                    -((distance / HUE_KERNEL_WIDTH) ** 2)
                )
    if used_pixels == 0:
        raise PixelValidationError("ball_patch_has_no_admissible_pixels")
    predicted = max(scores, key=scores.__getitem__)
    return predicted, used_pixels, scores


def _active_chain(probe: dict[str, Any]) -> dict[str, Any]:
    curves = probe["active_board"]["curve_manager"]["curves"]
    if len(curves) != 1:
        raise PixelValidationError("validator_requires_one_curve")
    matches = [
        row
        for row in curves[0]["intrusive_lists"]
        if row["container_offset"] == ACTIVE_CHAIN_LIST_OFFSET
    ]
    if len(matches) != 1:
        raise PixelValidationError("active_chain_list_missing")
    return matches[0]


def _classify_record(
    payload: bytes,
    *,
    index: int,
    source_record: dict[str, Any],
    object_size: int,
    expected_vtable: int,
    pixels: Any,
    width: int,
    height: int,
    allow_rollout_occlusion: bool,
    record_kind: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    offset = source_record["artifact_offset"]
    if (
        source_record["artifact_bytes"] != object_size
        or offset != index * object_size
        or struct.unpack_from("<I", payload, offset)[0] != expected_vtable
    ):
        raise PixelValidationError(f"{record_kind}_record_layout_invalid")
    ball_id = struct.unpack_from("<I", payload, offset + 0x10)[0]
    color_id = struct.unpack_from("<i", payload, offset + 0x14)[0]
    position_x, position_y = struct.unpack_from(
        "<ff", payload, offset + 0x2C
    )
    reason: str | None = None
    if color_id not in PROTOTYPE_HUES:
        reason = "unsupported_color_id"
    elif (
        allow_rollout_occlusion
        and position_x < ROLLOUT_FOREGROUND_OCCLUSION_X
        and position_y < ROLLOUT_FOREGROUND_OCCLUSION_Y
    ):
        reason = "rollout_entrance_foreground_occlusion"
    elif (
        position_x < SAMPLE_RADIUS
        or position_x > width - 1 - SAMPLE_RADIUS
        or position_y < SAMPLE_RADIUS
        or position_y > height - 1 - SAMPLE_RADIUS
    ):
        reason = "sample_disk_outside_frame"
    identity = {
        "index": index,
        "ball_id": ball_id,
        "color_id": color_id,
        "position_x": position_x,
        "position_y": position_y,
    }
    if reason is not None:
        return None, {**identity, "reason": reason}
    predicted, used_pixels, scores = _classify_patch(
        pixels,
        width=width,
        height=height,
        center_x=position_x,
        center_y=position_y,
    )
    expected_score = scores[color_id]
    competing_score = max(
        score
        for candidate, score in scores.items()
        if candidate != color_id
    )
    return (
        {
            **identity,
            "predicted_color_id": predicted,
            "match": predicted == color_id,
            "used_pixels": used_pixels,
            "expected_score": expected_score,
            "strongest_competing_score": competing_score,
            "score_margin": expected_score - competing_score,
        },
        None,
    )


def validate(
    probe_path: Path,
    *,
    minimum_visible: int,
    maximum_mismatches: int,
    minimum_visible_fired_bullets: int = 0,
    maximum_fired_bullet_mismatches: int = 0,
    verify_raw_payloads: bool = True,
) -> dict[str, Any]:
    if verify_raw_payloads:
        try:
            return validate_probe_pixels(
                probe_path,
                minimum_visible=minimum_visible,
                maximum_mismatches=maximum_mismatches,
                minimum_visible_fired_bullets=(
                    minimum_visible_fired_bullets
                ),
                maximum_fired_bullet_mismatches=(
                    maximum_fired_bullet_mismatches
                ),
                require_freeze_state=True,
            )
        except PcMemoryEvidenceError as error:
            raise PixelValidationError(str(error)) from error
    probe = json.loads(probe_path.read_text(encoding="ascii"))
    if probe.get("schema") != "zuma-rl.pc-memory-int32-probe":
        raise PixelValidationError("memory_probe_schema_invalid")
    raw_validation = None
    chain = _active_chain(probe)
    payload_path = probe_path.parent / chain["artifact"]
    screenshot_path = probe_path.parent / probe["frozen_frame"]["artifact"]
    payload = payload_path.read_bytes()
    records = chain["records"]
    if (
        chain["payload_count"] != len(records)
        or len(payload) != len(records) * BALL_OBJECT_SIZE
    ):
        raise PixelValidationError("ball_payload_artifact_size_mismatch")

    with Image.open(screenshot_path) as source:
        image = source.convert("RGB")
    width, height = image.size
    pixels = image.load()
    results: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for index, source_record in enumerate(records):
        result, exclusion = _classify_record(
            payload,
            index=index,
            source_record=source_record,
            object_size=BALL_OBJECT_SIZE,
            expected_vtable=BALL_VTABLE,
            pixels=pixels,
            width=width,
            height=height,
            allow_rollout_occlusion=True,
            record_kind="ball",
        )
        if exclusion is not None:
            excluded.append(exclusion)
        elif result is not None:
            results.append(result)

    fired = probe["active_board"].get("fired_bullets")
    if not isinstance(fired, dict):
        raise PixelValidationError("fired_bullet_list_missing")
    fired_records = fired["records"]
    fired_payload_path = probe_path.parent / fired["artifact"]
    fired_payload = fired_payload_path.read_bytes()
    if (
        fired["declared_count"] != len(fired_records)
        or fired["traversed_count"] != len(fired_records)
        or len(fired_payload) != len(fired_records) * BULLET_OBJECT_SIZE
    ):
        raise PixelValidationError("fired_bullet_payload_artifact_size_mismatch")
    fired_results: list[dict[str, Any]] = []
    fired_excluded: list[dict[str, Any]] = []
    for index, source_record in enumerate(fired_records):
        result, exclusion = _classify_record(
            fired_payload,
            index=index,
            source_record=source_record,
            object_size=BULLET_OBJECT_SIZE,
            expected_vtable=BULLET_VTABLE,
            pixels=pixels,
            width=width,
            height=height,
            allow_rollout_occlusion=False,
            record_kind="fired_bullet",
        )
        if exclusion is not None:
            fired_excluded.append(exclusion)
        elif result is not None:
            fired_results.append(result)

    mismatches = [row for row in results if not row["match"]]
    fired_mismatches = [
        row for row in fired_results if not row["match"]
    ]
    if len(results) < minimum_visible:
        raise PixelValidationError(
            f"too_few_visible_balls:{len(results)}<{minimum_visible}"
        )
    if len(mismatches) > maximum_mismatches:
        raise PixelValidationError(
            "ball_pixel_mismatch_limit_exceeded:"
            f"{len(mismatches)}>{maximum_mismatches}"
        )
    if len(fired_results) < minimum_visible_fired_bullets:
        raise PixelValidationError(
            "too_few_visible_fired_bullets:"
            f"{len(fired_results)}<{minimum_visible_fired_bullets}"
        )
    if len(fired_mismatches) > maximum_fired_bullet_mismatches:
        raise PixelValidationError(
            "fired_bullet_pixel_mismatch_limit_exceeded:"
            f"{len(fired_mismatches)}>{maximum_fired_bullet_mismatches}"
        )
    color_ids = sorted({row["color_id"] for row in results})
    return {
        "schema": "zuma-rl.pc-ball-pixel-validation",
        "version": 3 if raw_validation is not None else 2,
        "status": "PASS",
        "memory_probe": {
            "artifact": probe_path.name,
            "sha256": _sha256_path(probe_path),
            "framework_update": probe["framework_update"],
        },
        "ball_payload": {
            "artifact": chain["artifact"],
            "sha256": _sha256_path(payload_path),
            "total_count": len(records),
            "retail_vtable_hex": f"0x{BALL_VTABLE:08x}",
            "retail_object_bytes": BALL_OBJECT_SIZE,
        },
        "frozen_frame": {
            "artifact": screenshot_path.name,
            "sha256": _sha256_path(screenshot_path),
            "width": width,
            "height": height,
        },
        "raw_payload_validation": (
            {
                "schema": raw_validation["schema"],
                "version": raw_validation["version"],
                "status": raw_validation["status"],
                "probe_sha256": raw_validation["probe_sha256"],
                "active_chain_count": raw_validation[
                    "active_chain_count"
                ],
                "fired_bullet_count": raw_validation[
                    "fired_bullet_count"
                ],
            }
            if raw_validation is not None
            else None
        ),
        "method": {
            "active_chain_list_offset_hex": (
                f"0x{ACTIVE_CHAIN_LIST_OFFSET:03x}"
            ),
            "sample_radius": SAMPLE_RADIUS,
            "minimum_saturation": MINIMUM_SATURATION,
            "minimum_value": MINIMUM_VALUE,
            "hue_kernel_width": HUE_KERNEL_WIDTH,
            "rollout_foreground_occlusion_x": (
                ROLLOUT_FOREGROUND_OCCLUSION_X
            ),
            "rollout_foreground_occlusion_y": (
                ROLLOUT_FOREGROUND_OCCLUSION_Y
            ),
            "prototype_hues": {
                str(key): value
                for key, value in sorted(PROTOTYPE_HUES.items())
            },
        },
        "visible_count": len(results),
        "excluded_count": len(excluded),
        "match_count": len(results) - len(mismatches),
        "mismatch_count": len(mismatches),
        "accuracy": (
            (len(results) - len(mismatches)) / len(results)
            if results
            else 0.0
        ),
        "per_color": {
            str(color_id): {
                "visible": sum(
                    row["color_id"] == color_id for row in results
                ),
                "matches": sum(
                    row["color_id"] == color_id and row["match"]
                    for row in results
                ),
            }
            for color_id in color_ids
        },
        "fired_bullets": {
            "payload": {
                "artifact": fired["artifact"],
                "sha256": _sha256_path(fired_payload_path),
                "total_count": len(fired_records),
                "retail_vtable_hex": f"0x{BULLET_VTABLE:08x}",
                "retail_object_bytes": BULLET_OBJECT_SIZE,
                "board_container_offset_hex": (
                    fired["board_container_offset_hex"]
                ),
            },
            "visible_count": len(fired_results),
            "excluded_count": len(fired_excluded),
            "match_count": len(fired_results) - len(fired_mismatches),
            "mismatch_count": len(fired_mismatches),
            "accuracy": (
                (len(fired_results) - len(fired_mismatches))
                / len(fired_results)
                if fired_results
                else 1.0
            ),
            "excluded": fired_excluded,
            "results": fired_results,
        },
        "excluded": excluded,
        "results": results,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("memory_probe", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--minimum-visible", type=int, default=50)
    parser.add_argument("--maximum-mismatches", type=int, default=0)
    parser.add_argument("--minimum-visible-fired-bullets", type=int, default=0)
    parser.add_argument(
        "--maximum-fired-bullet-mismatches",
        type=int,
        default=0,
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if (
        args.minimum_visible < 1
        or args.maximum_mismatches < 0
        or args.minimum_visible_fired_bullets < 0
        or args.maximum_fired_bullet_mismatches < 0
    ):
        raise SystemExit("invalid validation threshold")
    output = args.output.resolve()
    if output.exists() or not output.parent.is_dir():
        raise SystemExit("output path must be new and its parent must exist")
    report = validate(
        args.memory_probe.resolve(),
        minimum_visible=args.minimum_visible,
        maximum_mismatches=args.maximum_mismatches,
        minimum_visible_fired_bullets=(
            args.minimum_visible_fired_bullets
        ),
        maximum_fired_bullet_mismatches=(
            args.maximum_fired_bullet_mismatches
        ),
    )
    output.write_text(
        json.dumps(
            report,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="ascii",
    )
    print(
        f"status={report['status']} visible={report['visible_count']} "
        f"matches={report['match_count']} "
        f"excluded={report['excluded_count']} "
        f"fired_visible={report['fired_bullets']['visible_count']} "
        f"fired_matches={report['fired_bullets']['match_count']}"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
