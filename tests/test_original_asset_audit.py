"""Tests for installed fruit-asset fidelity auditing."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from zuma_rl import original_asset_audit


ORIGINAL_ROOT = Path("D:/SteamLibrary/steamapps/common/Zuma's Revenge")


def test_validator_recomputes_instead_of_trusting_summary(monkeypatch) -> None:
    live = {"schema": "test", "status": "PASS", "value": 1}
    monkeypatch.setattr(
        original_asset_audit,
        "audit_original_assets",
        lambda **kwargs: deepcopy(live),
    )
    assert (
        original_asset_audit.validate_original_asset_audit_report(
            live,
            original_root=Path("game"),
            level_id="Jungle2",
            hard=False,
            profile_mode="tutorials_completed",
        )
        is None
    )
    tampered = dict(live, value=2)
    assert "differs" in original_asset_audit.validate_original_asset_audit_report(
        tampered,
        original_root=Path("game"),
        level_id="Jungle2",
        hard=False,
        profile_mode="tutorials_completed",
    )


@pytest.mark.skipif(
    not (ORIGINAL_ROOT / "main.pak").is_file(),
    reason="installed retail assets are unavailable",
)
def test_installed_jungle2_fruit_assets_pass_live_recomputation() -> None:
    report = original_asset_audit.audit_original_assets(
        original_root=ORIGINAL_ROOT,
    )

    assert report["status"] == "PASS"
    assert report["installed_source"]["zone_fruit_type"] == "pineapple"
    assert report["fruit_assets"]["sheet_columns"] == 10
    assert report["fruit_assets"]["sheet_rows"] == 6
    assert report["fruit_assets"]["collection_animation_frames"] > 0
    assert (
        original_asset_audit.validate_original_asset_audit_report(
            report,
            original_root=ORIGINAL_ROOT,
            level_id="Jungle2",
            hard=False,
            profile_mode="tutorials_completed",
        )
        is None
    )
