from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import (
    build_alphazuma_55_polar_intent_wide_successor_intent_v2 as builder,
)
from tools.build_alphazuma_55_eval_contract import _sha256


def test_v2_transfers_identical_seed_matrices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    formal = {
        "final_blind": {"base_seed": 2_500_000_000, "last_seed": 2_500_000_439},
        "continuous_campaign_challenge": {
            "base_seed": 2_600_000_000,
            "last_seed": 2_600_000_219,
        },
    }
    superseded = tmp_path / "old.json"
    superseded.write_text(
        json.dumps(
            {
                "schema": (
                    "zuma-rl.alphazuma-55-polar-intent-wide-successor-intent"
                ),
                "status": "FROZEN_BEFORE_ENGINEERING_RESULT",
                "implementation_boundary": {
                    "this_intent_does_not_authorize_policy_inference": True
                },
                "outputs": {
                    "root": str(tmp_path / "old-root"),
                    "independent_audit_receipt": str(tmp_path / "old-audit.json"),
                },
                "formal_campaign_if_promoted": formal,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        builder.legacy,
        "build",
        lambda **_: {
            "formal_campaign_if_promoted": formal,
            "implementation_boundary": {},
        },
    )

    value = builder.build(
        source_master_path=tmp_path,
        validation_plan_path=tmp_path,
        sibling_wide_intent_path=tmp_path,
        superseded_intent_path=superseded,
        expected_superseded_sha256=_sha256(superseded),
        output_root=tmp_path / "new-root",
        independent_audit_receipt=tmp_path / "new-audit.json",
        campaign_id="v2",
    )

    assert value["seed_reservation_transfer"][
        "final_blind_range_unchanged"
    ] is True
    assert value["supersedes"]["superseded_intent_formal_seed_consumption"] == "NONE"
