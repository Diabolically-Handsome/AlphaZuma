"""Fail-closed audit of actor features against verified retail pixels.

The state actor is useful for fast curriculum training only if every value it
receives can later be reconstructed from the original game's pixels, a short
pixel history, static level geometry, or an explicit agent-side API channel.
This module never trusts a suite feature label.  It re-runs the actor contract
and every referenced PC Golden verifier, then builds a per-feature proof
matrix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from zuma_rl.actor_audit import (
    AUDIT_SCHEMA,
    AUDIT_VERSION,
    POLICY_ID,
    audit_actor_observation,
)
from zuma_rl.pc_golden import (
    ComparisonStatus,
    PcGoldenManifest,
    PcGoldenValidationError,
    canonical_sha256,
    pc_golden_native_source_fingerprint,
)
from zuma_rl.revenge_core import SUPPORTED_PROFILE_MODE
from zuma_rl.verify_pc_golden import verify_pc_golden_case

AUDIT_TYPE = "actor_visual_derivability"
MAX_VISUAL_AUDIT_CASES = 64


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _positive_finite(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def _rows(value: Any) -> tuple[Mapping[str, Any], ...] | None:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return None
    result = tuple(value)
    if any(not isinstance(item, Mapping) for item in result):
        return None
    return result


def _integer_values(value: Any) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return ()
    result: list[int] = []
    for item in value:
        normalized = _nonnegative_int(item)
        if normalized is None:
            return ()
        result.append(normalized)
    return tuple(result)


def _verification_capabilities(verification: Any) -> frozenset[str]:
    if (
        verification is None
        or verification.status is not ComparisonStatus.PASS
    ):
        return frozenset()
    summary = getattr(verification, "summary", None)
    if not isinstance(summary, Mapping):
        return frozenset()
    semantic = summary.get("semantic_limits")
    if not isinstance(semantic, Mapping):
        return frozenset()

    capabilities: set[str] = set()
    replay = summary.get("replay_determinism")
    if (
        semantic.get("deterministic_replay_attested") is True
        and isinstance(replay, Mapping)
        and replay.get("machine_verifiable_evidence") is True
        and replay.get("full_viewport_rgb24_per_tick_matched") is True
        and replay.get("normalized_trace_matched") is True
        and replay.get("dmo_binding_matched") is True
        and (_nonnegative_int(replay.get("run_count")) or 0) >= 2
        and (_nonnegative_int(replay.get("native_tick_count")) or 0) > 0
    ):
        capabilities.add("full_viewport_tick_history")

    measurement = summary.get("measurement_provenance")
    if (
        semantic.get("memory_measurement_provenance_attested") is not True
        or not isinstance(measurement, Mapping)
        or measurement.get("machine_verifiable_evidence") is not True
        or measurement.get("gameplay_viewport_pixels_matched") is not True
    ):
        return frozenset(capabilities)

    active_visible = _nonnegative_int(
        measurement.get("active_visible_count")
    )
    active_matches = _nonnegative_int(
        measurement.get("active_pixel_match_count")
    )
    if (
        active_visible is not None
        and active_visible > 0
        and active_matches == active_visible
    ):
        capabilities.add("active_ball_presence_position")
        for color_id in _integer_values(
            measurement.get("active_color_ids")
        ):
            capabilities.add(f"active_ball_color_{color_id}")

    fired_visible = _nonnegative_int(
        measurement.get("fired_visible_count")
    )
    fired_matches = _nonnegative_int(
        measurement.get("fired_pixel_match_count")
    )
    if (
        fired_visible is not None
        and fired_visible > 0
        and fired_matches == fired_visible
        and _nonnegative_int(measurement.get("fired_ball_id")) is not None
        and _nonnegative_int(measurement.get("fired_color_id")) is not None
        and _positive_finite(measurement.get("fired_speed"))
    ):
        capabilities.add("projectile_presence_position")
        for color_id in _integer_values(
            measurement.get("fired_color_ids")
        ):
            capabilities.add(f"projectile_color_{color_id}")

    curve = measurement.get("curve_geometry_binding")
    if (
        isinstance(curve, Mapping)
        and curve.get("resolved_against_original_data") is True
        and _rows(curve.get("phases"))
    ):
        capabilities.add("curve_geometry_pc_binding")
    return frozenset(capabilities)


def _feature_requirements(
    section: str,
    feature: str,
    visibility: str,
) -> tuple[str, ...]:
    requirements: set[str]
    if section == "balls":
        if feature in {"present", "x", "y"}:
            requirements = {"active_ball_presence_position"}
        elif feature == "exploding":
            requirements = {
                "active_ball_presence_position",
                "exploding_ball_visual",
            }
        elif feature == "in_tunnel":
            requirements = {"actor_tunnel_filter_contract"}
        elif feature in {"waypoint", "contact_next_visible"}:
            requirements = {
                "active_ball_presence_position",
                "curve_geometry_pc_binding",
            }
        elif feature.startswith("color_"):
            requirements = {
                f"active_ball_color_{feature.removeprefix('color_')}"
            }
        elif feature.startswith("powerup_"):
            requirements = {
                f"active_ball_powerup_{feature.removeprefix('powerup_')}"
            }
        elif feature.startswith("curve_"):
            requirements = {"curve_geometry_pc_binding"}
        else:
            requirements = {f"unmapped:{section}:{feature}:{visibility}"}
    elif section == "projectiles":
        if feature in {"present", "x", "y"}:
            requirements = {"projectile_presence_position"}
        elif feature in {"velocity_x", "velocity_y"}:
            requirements = {
                "projectile_presence_position",
                "projectile_motion_history",
            }
        elif feature in {"merging", "merge_progress"}:
            requirements = {"projectile_merge_history"}
        elif feature == "waypoint":
            requirements = {
                "projectile_presence_position",
                "curve_geometry_pc_binding",
            }
        elif feature.startswith("color_"):
            requirements = {
                f"projectile_color_{feature.removeprefix('color_')}"
            }
        elif feature.startswith("curve_"):
            requirements = {"curve_geometry_pc_binding"}
        else:
            requirements = {f"unmapped:{section}:{feature}:{visibility}"}
    elif section == "globals":
        if visibility == "privileged_zero":
            requirements = {"actor_privileged_zero_contract"}
        elif visibility == "agent_clock":
            requirements = {"agent_clock_channel"}
        elif visibility == "terminal_api":
            requirements = {"terminal_api_channel"}
        elif feature in {"aim_sin", "aim_cos"}:
            requirements = {"shooter_orientation_visual"}
        elif feature in {"current_missing", "next_missing"} or feature.startswith(
            ("current_color_", "next_color_")
        ):
            requirements = {"shooter_chamber_visual"}
        elif feature in {"bar_current", "bar_target"}:
            requirements = {"zuma_bar_visual"}
        elif feature in {"score_progress", "score_tanh"}:
            requirements = {"score_hud_visual"}
        elif feature in {"gun_normal", "gun_firing", "gun_reloading"}:
            requirements = {"gun_animation_visual"}
        elif feature == "gun_state_percent":
            requirements = {"gun_animation_history"}
        elif feature in {"zuma_reached", "stop_adding"}:
            requirements = {"board_phase_history"}
        elif feature in {"visible_ball_fraction", "known_ball_fraction"}:
            requirements = {"active_ball_presence_position"}
        elif feature == "projectile_fraction":
            requirements = {"projectile_presence_position"}
        else:
            requirements = {f"unmapped:{section}:{feature}:{visibility}"}
    else:
        requirements = {f"unmapped:{section}:{feature}:{visibility}"}
    return tuple(sorted(requirements))


def _safe_manifest_inputs(
    manifest_paths: Sequence[str | Path],
    *,
    evidence_root: str | Path,
) -> tuple[tuple[str, Path], ...]:
    root = Path(evidence_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("evidence_root must be a directory")
    if len(manifest_paths) > MAX_VISUAL_AUDIT_CASES:
        raise ValueError("too many PC Golden manifests for one visual audit")
    result: list[tuple[str, Path]] = []
    for value in manifest_paths:
        path = Path(value).resolve(strict=True)
        if not path.is_file():
            raise ValueError("visual-audit manifest path must be a file")
        try:
            relative = path.relative_to(root)
        except ValueError as error:
            raise ValueError(
                "visual-audit manifest escapes evidence_root"
            ) from error
        result.append((PurePosixPath(*relative.parts).as_posix(), path))
    result.sort(key=lambda item: item[0])
    if len({relative for relative, _ in result}) != len(result):
        raise ValueError("visual-audit manifest paths must be unique")
    return tuple(result)


def audit_actor_visual_derivability(
    manifest_paths: Sequence[str | Path],
    *,
    evidence_root: str | Path,
    original_root: str | Path | None,
    level_id: str = "Jungle2",
    hard: bool = False,
    profile_mode: str = SUPPORTED_PROFILE_MODE,
) -> dict[str, Any]:
    """Build a deterministic per-feature proof matrix."""

    inputs = _safe_manifest_inputs(
        manifest_paths,
        evidence_root=evidence_root,
    )
    actor = audit_actor_observation(
        root=original_root,
        level_id=level_id,
        hard=hard,
        profile_mode=profile_mode,
    )
    actor_pass = actor.get("status") == "PASS"
    actor_contract = actor.get("feature_contract")
    if not isinstance(actor_contract, list):
        actor_contract = []

    local_capabilities: set[str] = set()
    if actor_pass:
        local_capabilities.update(
            {
                "actor_privileged_zero_contract",
                "actor_tunnel_filter_contract",
                "agent_clock_channel",
                "terminal_api_channel",
            }
        )

    cases: list[dict[str, Any]] = []
    case_capabilities: dict[str, frozenset[str]] = {}
    case_failures = 0
    for relative, path in inputs:
        manifest_sha256 = _sha256_path(path)
        try:
            manifest = PcGoldenManifest.read_json(path)
            source_fingerprint = pc_golden_native_source_fingerprint(manifest)
            verification = verify_pc_golden_case(
                path,
                case_root=path.parent,
                original_root=original_root,
            )
        except (
            KeyError,
            OSError,
            RecursionError,
            UnicodeError,
            PcGoldenValidationError,
            ValueError,
        ):
            case_failures += 1
            cases.append(
                {
                    "manifest_path": relative,
                    "manifest_sha256": manifest_sha256,
                    "case_id": None,
                    "source_fingerprint": None,
                    "verification_status": "INCOMPARABLE",
                    "capabilities": [],
                }
            )
            continue

        capabilities = _verification_capabilities(verification)
        verification_status = verification.status.value
        if verification.status is not ComparisonStatus.PASS:
            case_failures += 1
        case_capabilities[source_fingerprint] = capabilities
        cases.append(
            {
                "manifest_path": relative,
                "manifest_sha256": manifest_sha256,
                "case_id": manifest.case_id,
                "source_fingerprint": source_fingerprint,
                "verification_status": verification_status,
                "capabilities": sorted(capabilities),
            }
        )

    available = set(local_capabilities)
    for capabilities in case_capabilities.values():
        available.update(capabilities)

    feature_proofs: list[dict[str, Any]] = []
    missing_capabilities: set[str] = set()
    for raw_row in actor_contract:
        if not isinstance(raw_row, Mapping):
            continue
        section = raw_row.get("section")
        feature = raw_row.get("feature")
        visibility = raw_row.get("visibility")
        if not all(
            isinstance(value, str)
            for value in (section, feature, visibility)
        ):
            continue
        required = _feature_requirements(section, feature, visibility)
        missing = tuple(
            capability
            for capability in required
            if capability not in available
        )
        missing_capabilities.update(missing)
        supporting_sources = sorted(
            source
            for source, capabilities in case_capabilities.items()
            if any(capability in capabilities for capability in required)
        )
        feature_proofs.append(
            {
                "section": section,
                "feature": feature,
                "visibility": visibility,
                "required_capabilities": list(required),
                "status": "PASS" if not missing else "MISSING",
                "missing_capabilities": list(missing),
                "supporting_source_fingerprints": supporting_sources,
            }
        )

    missing_feature_count = sum(
        row["status"] != "PASS" for row in feature_proofs
    )
    reasons: list[str] = []
    if not actor_pass:
        reasons.append(
            "live actor observation contract is not comparable or did not pass"
        )
    if not inputs:
        reasons.append("no PC Golden manifests were supplied")
    if case_failures:
        reasons.append(
            f"{case_failures} PC Golden case(s) did not pass live verification"
        )
    if missing_feature_count:
        reasons.append(
            f"{missing_feature_count} actor feature(s) lack derivation proof"
        )
    status = "PASS" if not reasons else "INCOMPARABLE"
    checks = [
        {
            "name": "live_actor_contract",
            "status": "PASS" if actor_pass else "INCOMPARABLE",
        },
        {
            "name": "pc_golden_verification",
            "status": (
                "PASS"
                if inputs and case_failures == 0
                else "INCOMPARABLE"
            ),
        },
        {
            "name": "feature_derivation_coverage",
            "status": "PASS" if missing_feature_count == 0 else "INCOMPARABLE",
        },
    ]
    source_fingerprints = sorted(case_capabilities)
    payload: dict[str, Any] = {
        "schema": AUDIT_SCHEMA,
        "version": AUDIT_VERSION,
        "status": status,
        "audit_type": AUDIT_TYPE,
        "scope": {
            "policy": POLICY_ID,
            "environment_id": "ZumaRevenge-v0",
            "level_id": level_id,
            "hard": hard,
            "profile_mode": profile_mode,
            "observation_mode": "actor",
        },
        "actor_audit_fingerprint": actor.get("audit_fingerprint"),
        "implementation_fingerprint": actor.get(
            "implementation_fingerprint"
        ),
        "environment_fingerprint": actor.get("environment_fingerprint"),
        "cases": cases,
        "checks": checks,
        "feature_proofs": feature_proofs,
        "failure_reasons": reasons,
        "summary": {
            "feature_count": len(feature_proofs),
            "derivable_feature_count": (
                len(feature_proofs) - missing_feature_count
            ),
            "missing_feature_count": missing_feature_count,
            "missing_capabilities": sorted(missing_capabilities),
            "pc_manifest_count": len(inputs),
            "pc_source_count": len(source_fingerprints),
            "pc_source_fingerprints": source_fingerprints,
            "available_capabilities": sorted(available),
        },
    }
    payload["audit_fingerprint"] = canonical_sha256(payload)
    return payload


def _resolve_report_manifest(
    value: Any,
    *,
    evidence_root: Path,
) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        return None
    candidate = evidence_root.joinpath(*relative.parts)
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(evidence_root)
    except (OSError, ValueError):
        return None
    return resolved if resolved.is_file() else None


def validate_actor_visual_audit_report(
    report: Mapping[str, Any],
    *,
    evidence_root: str | Path,
    original_root: str | Path | None,
    level_id: str,
    hard: bool,
    profile_mode: str,
) -> str | None:
    """Recompute a submitted PASS report and return a rejection reason."""

    if report.get("audit_type") != AUDIT_TYPE:
        return "unsupported actor visual audit type"
    if report.get("status") != "PASS":
        return "actor visual audit report is not PASS"
    root = Path(evidence_root)
    try:
        root = root.resolve(strict=True)
    except OSError:
        return "visual audit evidence root does not exist"
    if not root.is_dir():
        return "visual audit evidence root is not a directory"

    cases = report.get("cases")
    if (
        not isinstance(cases, list)
        or not cases
        or len(cases) > MAX_VISUAL_AUDIT_CASES
        or any(not isinstance(item, Mapping) for item in cases)
    ):
        return "visual audit case list is missing or invalid"
    manifest_paths: list[Path] = []
    seen: set[Path] = set()
    for item in cases:
        path = _resolve_report_manifest(
            item.get("manifest_path"),
            evidence_root=root,
        )
        if path is None:
            return "visual audit manifest path is invalid or escapes its root"
        if path in seen:
            return "visual audit manifest paths are duplicated"
        seen.add(path)
        if item.get("manifest_sha256") != _sha256_path(path):
            return "visual audit manifest SHA-256 differs"
        manifest_paths.append(path)

    live = audit_actor_visual_derivability(
        manifest_paths,
        evidence_root=root,
        original_root=original_root,
        level_id=level_id,
        hard=hard,
        profile_mode=profile_mode,
    )
    if live.get("status") != "PASS":
        return "live actor visual audit is not comparable or did not pass"
    if report.get("audit_fingerprint") != live.get("audit_fingerprint"):
        return "actor visual audit fingerprint differs from live evidence"
    summary = report.get("summary")
    if (
        not isinstance(summary, Mapping)
        or summary.get("missing_feature_count") != 0
        or summary.get("pc_source_count", 0) < 1
    ):
        return "actor visual audit summary does not prove derivability"
    checks = report.get("checks")
    if (
        not isinstance(checks, list)
        or not checks
        or any(
            not isinstance(item, Mapping) or item.get("status") != "PASS"
            for item in checks
        )
    ):
        return "actor visual audit checks are missing or not all PASS"
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute whether every state-actor feature can be recovered "
            "from verified retail pixels, short history, geometry, or an "
            "explicit agent-side channel."
        )
    )
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--original-root", required=True, type=Path)
    parser.add_argument("--level", default="Jungle2")
    parser.add_argument("--hard", action="store_true")
    parser.add_argument("--compact", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = audit_actor_visual_derivability(
        args.manifests,
        evidence_root=args.evidence_root,
        original_root=args.original_root,
        level_id=args.level,
        hard=args.hard,
    )
    print(
        json.dumps(
            report,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=None if args.compact else 2,
            separators=(",", ":") if args.compact else None,
        )
    )
    return 0 if report["status"] == "PASS" else 1


__all__ = [
    "AUDIT_TYPE",
    "MAX_VISUAL_AUDIT_CASES",
    "audit_actor_visual_derivability",
    "build_parser",
    "main",
    "validate_actor_visual_audit_report",
]


if __name__ == "__main__":
    raise SystemExit(main())
