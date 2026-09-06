"""Recomputable audit of installed Jungle2 fruit assets and calibration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from zuma_rl.original_data import OriginalGameCatalog
from zuma_rl.revenge_core import RevengeSimulator, SUPPORTED_PROFILE_MODE


AUDIT_SCHEMA = "zuma-rl.fidelity-audit"
AUDIT_VERSION = 1
AUDIT_TYPE = "original_asset_fidelity"
POLICY_ID = "original-transfer-jungle2-v3"


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def audit_original_assets(
    *,
    original_root: str | Path,
    level_id: str = "Jungle2",
    hard: bool = False,
    profile_mode: str = SUPPORTED_PROFILE_MODE,
) -> dict[str, Any]:
    root = Path(original_root).resolve(strict=True)
    catalog = OriginalGameCatalog(root)
    level = catalog.level(level_id)
    if not level.fruit_type:
        raise ValueError("level has no Zone fruit assignment")
    assets = catalog.fruit_assets(level.fruit_type)
    simulator = RevengeSimulator.from_installed(
        level_id,
        root=root,
        hard=hard,
        curve_index=0,
        seed=0,
    )
    calibration = simulator.fruit_calibration
    if calibration is None:
        raise ValueError("simulator did not install fruit calibration")
    expected = {
        "fruit_type": assets.fruit_type,
        "logical_width": assets.logical_width,
        "logical_height": assets.logical_height,
        "sheet_columns": assets.sheet_columns,
        "sheet_rows": assets.sheet_rows,
        "collection_animation_frames": assets.collection_animation_frames,
        "collection_animation_fps": assets.collection_animation_fps,
    }
    observed = {
        "fruit_type": calibration.fruit_type,
        "logical_width": calibration.logical_width,
        "logical_height": calibration.logical_height,
        "sheet_columns": calibration.sheet_columns,
        "sheet_rows": calibration.sheet_rows,
        "collection_animation_frames": (
            calibration.collection_animation_frames
        ),
        "collection_animation_fps": calibration.collection_animation_fps,
    }
    if observed != expected:
        raise ValueError("simulator fruit calibration differs from installed assets")
    payload: dict[str, Any] = {
        "schema": AUDIT_SCHEMA,
        "version": AUDIT_VERSION,
        "status": "PASS",
        "audit_type": AUDIT_TYPE,
        "scope": {
            "policy": POLICY_ID,
            "environment_id": "ZumaRevenge-v0",
            "level_id": level_id,
            "hard": hard,
            "profile_mode": profile_mode,
            "observation_mode": "actor",
        },
        "installed_source": {
            "main_pak_sha256": _sha256_path(root / "main.pak"),
            "original_executable_sha256": _sha256_path(
                root / "ZumasRevenge.exe"
            ),
            "zone_number": level.zone_number,
            "zone_fruit_type": level.fruit_type,
        },
        "fruit_assets": {
            **expected,
            "image_resource_id": assets.image_resource_id,
            "image_path": assets.image_path,
            "high_resolution_gif_sha256": (
                assets.high_resolution_gif_sha256
            ),
            "low_resolution_gif_sha256": assets.low_resolution_gif_sha256,
            "collection_animation_sha256": (
                assets.collection_animation_sha256
            ),
            "collection_ticks_at_100hz": calibration.collection_ticks(100),
            "provenance": assets.provenance,
        },
        "simulator_calibration": observed,
        "checks": [
            {"name": "zone_to_fruit_mapping", "status": "PASS"},
            {"name": "dual_resolution_sheet_geometry", "status": "PASS"},
            {"name": "complete_version5_pam_parse", "status": "PASS"},
            {"name": "simulator_asset_binding", "status": "PASS"},
        ],
        "failure_reasons": [],
        "summary": {
            "fruit_asset_geometry_verified": True,
            "collection_animation_metadata_verified": True,
        },
    }
    payload["audit_fingerprint"] = _canonical_sha256(payload)
    return payload


def validate_original_asset_audit_report(
    report: Mapping[str, Any],
    *,
    original_root: str | Path | None,
    level_id: str,
    hard: bool,
    profile_mode: str,
) -> str | None:
    if original_root is None:
        return "original_root is required for original asset audit"
    try:
        live = audit_original_assets(
            original_root=original_root,
            level_id=level_id,
            hard=hard,
            profile_mode=profile_mode,
        )
    except (OSError, ValueError):
        return "original asset audit could not be recomputed"
    if dict(report) != live:
        return "original asset audit differs from installed retail data"
    return None


__all__ = [
    "AUDIT_TYPE",
    "POLICY_ID",
    "audit_original_assets",
    "validate_original_asset_audit_report",
]
