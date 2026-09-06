"""Validate a retail Zuma shot transition from two frozen memory probes.

This comparator binds the before/after memory snapshots to their independent
pixel-validation reports, then checks the state transition expected around one
recorded shot:

* the active curve keeps the same ordered Ball identities and colors;
* every active Ball advances by the expected curve distance;
* the shooter's current Bullet moves into the Board fired-Bullet list;
* the shooter's next Bullet is promoted to current; and
* score state remains unchanged over the short transition.

The output is a compact, hash-bound evidence report suitable for inclusion in a
PC golden case.  It does not claim that every Zuma mechanic is modelled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from zuma_rl.pc_memory_evidence import (
    ACTIVE_CHAIN_LIST_OFFSET,
    BOARD_FIRED_BULLET_LIST_OFFSET,
    MEMORY_PROBE_SCHEMA,
    validate_probe_payloads,
    validate_shot_transition,
)


PIXEL_VALIDATION_SCHEMA = "zuma-rl.pc-ball-pixel-validation"
SHOOTER_CURRENT_BULLET_OFFSET = 0x130
SHOOTER_NEXT_BULLET_OFFSET = 0x134


class TransitionValidationError(RuntimeError):
    """Raised when the two probes do not prove the requested transition."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise TransitionValidationError(reason)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise TransitionValidationError(f"json_root_invalid:{path.name}")
    return value


def _active_chain(probe: dict[str, Any]) -> dict[str, Any]:
    curves = probe["active_board"]["curve_manager"]["curves"]
    _require(len(curves) == 1, "transition_requires_one_curve")
    matches = [
        row
        for row in curves[0]["intrusive_lists"]
        if row["container_offset"] == ACTIVE_CHAIN_LIST_OFFSET
    ]
    _require(len(matches) == 1, "active_chain_list_missing")
    return matches[0]


def _shooter_bullet(
    probe: dict[str, Any],
    pointer_offset: int,
) -> dict[str, Any]:
    matches = [
        row
        for row in probe["active_board"]["primary_child"]["bullets"]
        if row["shooter_pointer_offset"] == pointer_offset
    ]
    _require(
        len(matches) == 1,
        f"shooter_bullet_pointer_missing:0x{pointer_offset:03x}",
    )
    return matches[0]


def _identity(record: dict[str, Any]) -> dict[str, int]:
    ball = record["ball"]
    return {
        "ball_id": int(ball["ball_id"]),
        "color_id": int(ball["color_id"]),
    }


def _validate_pixel_report(
    report: dict[str, Any],
    *,
    report_path: Path,
    probe_path: Path,
    probe: dict[str, Any],
) -> dict[str, Any]:
    _require(
        report.get("schema") == PIXEL_VALIDATION_SCHEMA,
        f"pixel_validation_schema_invalid:{report_path.name}",
    )
    _require(
        report.get("version", 0) >= 3,
        f"pixel_validation_version_too_old:{report_path.name}",
    )
    _require(
        report.get("status") == "PASS",
        f"pixel_validation_not_pass:{report_path.name}",
    )
    _require(
        report["memory_probe"]["sha256"] == _sha256_path(probe_path),
        f"pixel_validation_probe_hash_mismatch:{report_path.name}",
    )
    _require(
        report["memory_probe"]["framework_update"]
        == probe["framework_update"],
        f"pixel_validation_update_mismatch:{report_path.name}",
    )
    _require(
        report["mismatch_count"] == 0,
        f"active_ball_pixel_mismatch:{report_path.name}",
    )
    fired = report["fired_bullets"]
    _require(
        fired["mismatch_count"] == 0,
        f"fired_bullet_pixel_mismatch:{report_path.name}",
    )
    raw_validation = report.get("raw_payload_validation")
    _require(
        isinstance(raw_validation, dict)
        and raw_validation.get("status") == "PASS"
        and raw_validation.get("probe_sha256") == _sha256_path(probe_path),
        f"pixel_validation_raw_payload_binding_invalid:{report_path.name}",
    )
    return {
        "artifact": report_path.name,
        "sha256": _sha256_path(report_path),
        "active_visible_count": report["visible_count"],
        "active_match_count": report["match_count"],
        "fired_visible_count": fired["visible_count"],
        "fired_match_count": fired["match_count"],
    }


def validate_transition(
    before_probe_path: Path,
    after_probe_path: Path,
    before_pixel_path: Path,
    after_pixel_path: Path,
    *,
    expected_before_update: int,
    expected_after_update: int,
    expected_score: int,
    expected_chain_distance_delta: float,
    distance_tolerance: float,
    minimum_chain_count: int,
) -> dict[str, Any]:
    before = _load_json(before_probe_path)
    after = _load_json(after_probe_path)
    before_pixel = _load_json(before_pixel_path)
    after_pixel = _load_json(after_pixel_path)
    for label, probe in (("before", before), ("after", after)):
        _require(
            probe.get("schema") == MEMORY_PROBE_SCHEMA,
            f"{label}_memory_probe_schema_invalid",
        )
    before_raw = validate_probe_payloads(
        before_probe_path,
        expected_update=expected_before_update,
        expected_score=expected_score,
        require_freeze_state=True,
    )
    after_raw = validate_probe_payloads(
        after_probe_path,
        expected_runtime_sha256=before_raw[
            "runtime_executable_sha256"
        ],
        expected_dmo_sha256=before_raw["dmo_sha256"],
        expected_update=expected_after_update,
        expected_score=expected_score,
        require_freeze_state=True,
    )
    _require(
        before["framework_update"] == expected_before_update,
        "before_update_mismatch",
    )
    _require(
        after["framework_update"] == expected_after_update,
        "after_update_mismatch",
    )
    _require(
        expected_after_update > expected_before_update,
        "expected_update_order_invalid",
    )
    _require(
        before["runtime_executable_sha256"]
        == after["runtime_executable_sha256"],
        "runtime_executable_identity_changed",
    )
    _require(
        before["dmo_sha256"] == after["dmo_sha256"],
        "dmo_identity_changed",
    )

    before_board = before["active_board"]
    after_board = after["active_board"]
    for label, raw in (("before", before_raw), ("after", after_raw)):
        _require(raw["score"] == expected_score, f"{label}_score_mismatch")
        _require(
            raw["displayed_score"] == expected_score,
            f"{label}_displayed_score_mismatch",
        )
    _require(
        before_raw["score_target"] == after_raw["score_target"],
        "score_target_changed",
    )

    before_records = before_raw["active_chain"]
    after_records = after_raw["active_chain"]
    _require(
        len(before_records) == len(after_records),
        "active_chain_count_changed",
    )
    _require(
        len(before_records) >= minimum_chain_count,
        "active_chain_too_short",
    )
    before_identities = [_identity(row) for row in before_records]
    after_identities = [_identity(row) for row in after_records]
    _require(
        before_identities == after_identities,
        "active_chain_identity_or_color_changed",
    )
    distance_deltas = [
        float(after_row["ball"]["curve_distance"])
        - float(before_row["ball"]["curve_distance"])
        for before_row, after_row in zip(before_records, after_records)
    ]
    _require(
        all(math.isfinite(value) for value in distance_deltas),
        "active_chain_distance_delta_non_finite",
    )
    _require(
        all(
            abs(value - expected_chain_distance_delta)
            <= distance_tolerance
            for value in distance_deltas
        ),
        "active_chain_distance_delta_mismatch",
    )

    before_current = before_raw["shooter_current"]
    before_next = before_raw["shooter_next"]
    after_current = after_raw["shooter_current"]
    after_next = after_raw["shooter_next"]
    _require(
        _identity(after_current) == _identity(before_next),
        "shooter_next_was_not_promoted",
    )

    before_fired = before_raw["fired_bullets"]
    after_fired = after_raw["fired_bullets"]
    _require(len(before_fired) == 0, "before_fired_list_not_empty")
    _require(len(after_fired) == 1, "after_fired_list_not_singleton")
    transferred = after_fired[0]
    _require(
        _identity(transferred) == _identity(before_current),
        "shooter_current_was_not_transferred",
    )
    _require(
        _identity(after_next)
        not in (_identity(before_current), _identity(before_next)),
        "shooter_next_was_not_replenished",
    )
    velocity_x = float(transferred["subclass_fields"]["velocity_x"])
    velocity_y = float(transferred["subclass_fields"]["velocity_y"])
    speed = math.hypot(velocity_x, velocity_y)
    _require(math.isfinite(speed) and speed > 0.0, "fired_bullet_velocity_invalid")
    travel_x = (
        float(transferred["ball"]["position_x"])
        - float(before_current["ball"]["position_x"])
    )
    travel_y = (
        float(transferred["ball"]["position_y"])
        - float(before_current["ball"]["position_y"])
    )
    travel_distance = math.hypot(travel_x, travel_y)
    _require(
        math.isfinite(travel_distance) and travel_distance > 0.0,
        "fired_bullet_did_not_move",
    )

    before_pixel_evidence = _validate_pixel_report(
        before_pixel,
        report_path=before_pixel_path,
        probe_path=before_probe_path,
        probe=before,
    )
    after_pixel_evidence = _validate_pixel_report(
        after_pixel,
        report_path=after_pixel_path,
        probe_path=after_probe_path,
        probe=after,
    )
    _require(
        before_pixel_evidence["fired_visible_count"] == 0,
        "before_pixel_report_has_fired_bullet",
    )
    _require(
        after_pixel_evidence["fired_visible_count"] == 1
        and after_pixel_evidence["fired_match_count"] == 1,
        "after_fired_bullet_pixel_evidence_missing",
    )
    fired_pixel_rows = after_pixel["fired_bullets"]["results"]
    _require(
        len(fired_pixel_rows) == 1
        and {
            "ball_id": fired_pixel_rows[0]["ball_id"],
            "color_id": fired_pixel_rows[0]["color_id"],
        }
        == _identity(transferred),
        "fired_bullet_pixel_identity_mismatch",
    )

    return {
        "schema": "zuma-rl.pc-memory-transition-validation",
        "version": 1,
        "status": "PASS",
        "inputs": {
            "before_memory_probe": {
                "artifact": before_probe_path.name,
                "sha256": _sha256_path(before_probe_path),
            },
            "after_memory_probe": {
                "artifact": after_probe_path.name,
                "sha256": _sha256_path(after_probe_path),
            },
            "before_pixel_validation": before_pixel_evidence,
            "after_pixel_validation": after_pixel_evidence,
            "runtime_executable_sha256": before[
                "runtime_executable_sha256"
            ],
            "dmo_sha256": before["dmo_sha256"],
        },
        "updates": {
            "before": expected_before_update,
            "after": expected_after_update,
            "delta": expected_after_update - expected_before_update,
        },
        "score": {
            "before": before_raw["score"],
            "after": after_raw["score"],
            "displayed_before": before_raw["displayed_score"],
            "displayed_after": after_raw["displayed_score"],
            "target": before_raw["score_target"],
        },
        "active_chain": {
            "retail_list_offset_hex": (
                f"0x{ACTIVE_CHAIN_LIST_OFFSET:03x}"
            ),
            "count": len(before_records),
            "ordered_identities_and_colors_equal": True,
            "expected_curve_distance_delta": (
                expected_chain_distance_delta
            ),
            "distance_tolerance": distance_tolerance,
            "minimum_curve_distance_delta": min(distance_deltas),
            "maximum_curve_distance_delta": max(distance_deltas),
            "mean_curve_distance_delta": (
                sum(distance_deltas) / len(distance_deltas)
            ),
        },
        "shot_transition": {
            "shooter_current_pointer_offset_hex": (
                f"0x{SHOOTER_CURRENT_BULLET_OFFSET:03x}"
            ),
            "shooter_next_pointer_offset_hex": (
                f"0x{SHOOTER_NEXT_BULLET_OFFSET:03x}"
            ),
            "board_fired_list_offset_hex": (
                f"0x{BOARD_FIRED_BULLET_LIST_OFFSET:03x}"
            ),
            "before_current": _identity(before_current),
            "before_next": _identity(before_next),
            "after_current": _identity(after_current),
            "after_next": _identity(after_next),
            "transferred_fired_bullet": {
                **_identity(transferred),
                "position_x": transferred["ball"]["position_x"],
                "position_y": transferred["ball"]["position_y"],
                "velocity_x": velocity_x,
                "velocity_y": velocity_y,
                "speed": speed,
                "travel_from_before_current": travel_distance,
                "transient_flag_16a_after_transfer": transferred[
                    "subclass_fields"
                ]["fired"],
            },
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-probe", required=True, type=Path)
    parser.add_argument("--after-probe", required=True, type=Path)
    parser.add_argument("--before-pixel", required=True, type=Path)
    parser.add_argument("--after-pixel", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-before-update", required=True, type=int)
    parser.add_argument("--expected-after-update", required=True, type=int)
    parser.add_argument("--expected-score", required=True, type=int)
    parser.add_argument(
        "--expected-chain-distance-delta",
        required=True,
        type=float,
    )
    parser.add_argument("--distance-tolerance", type=float, default=1e-6)
    parser.add_argument("--minimum-chain-count", type=int, default=1)
    parser.add_argument("--minimum-visible", type=int, default=90)
    parser.add_argument("--maximum-mismatches", type=int, default=0)
    parser.add_argument(
        "--minimum-visible-after-fired-bullets",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--maximum-fired-bullet-mismatches",
        type=int,
        default=0,
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if (
        args.expected_before_update < 0
        or args.expected_after_update < 0
        or args.expected_score < 0
        or not math.isfinite(args.expected_chain_distance_delta)
        or not math.isfinite(args.distance_tolerance)
        or args.distance_tolerance < 0.0
        or args.minimum_chain_count < 1
        or args.minimum_visible < 1
        or args.maximum_mismatches < 0
        or args.minimum_visible_after_fired_bullets < 0
        or args.maximum_fired_bullet_mismatches < 0
    ):
        raise SystemExit("invalid transition-validation threshold")
    output = args.output.resolve()
    if output.exists() or not output.parent.is_dir():
        raise SystemExit("output path must be new and its parent must exist")
    report = validate_shot_transition(
        args.before_probe.resolve(),
        args.after_probe.resolve(),
        args.before_pixel.resolve(),
        args.after_pixel.resolve(),
        expected_before_update=args.expected_before_update,
        expected_after_update=args.expected_after_update,
        expected_score=args.expected_score,
        expected_chain_distance_delta=(
            args.expected_chain_distance_delta
        ),
        distance_tolerance=args.distance_tolerance,
        minimum_chain_count=args.minimum_chain_count,
        minimum_visible=args.minimum_visible,
        maximum_mismatches=args.maximum_mismatches,
        minimum_visible_after_fired_bullets=(
            args.minimum_visible_after_fired_bullets
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
    shot = report["shot_transition"]["transferred_fired_bullet"]
    print(
        f"status={report['status']} "
        f"updates={report['updates']['before']}->{report['updates']['after']} "
        f"chain={report['active_chain']['count']} "
        f"fired_id={shot['ball_id']} fired_color={shot['color_id']}"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
