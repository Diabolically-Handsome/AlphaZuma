from __future__ import annotations

import json
from pathlib import Path

from tools.build_alphazuma_55_eval_contract import _sha256
from tools.build_alphazuma_55_successor_master import RANKING, build_master


def test_successor_master_freezes_separate_seed_ranges_and_gate(tmp_path: Path) -> None:
    source_path = tmp_path / "source.json"
    plan_path = tmp_path / "plan.json"
    source = {
        "schema": "zuma-rl.alphazuma-55-weekend-master-preregistration",
        "version": 1,
        "campaign_id": "source",
        "deadline_utc": "2026-08-17T15:00:00Z",
        "deadline_local": "2026-08-17 11:00 America/Toronto",
        "scope": {
            "included_levels_in_adventure_order": [
                f"level-{index:02d}" for index in range(55)
            ]
        },
    }
    source_path.write_text(json.dumps(source), encoding="utf-8")
    source_path = source_path.resolve()
    plan = {
        "schema": "zuma-rl.alphazuma-55-polar-dagger-validation-plan",
        "version": 1,
        "status": "FROZEN_DURING_TARGET_TRAINING",
        "master_preregistration": {
            "path": str(source_path),
            "sha256": _sha256(source_path),
        },
        "decision_rule": {"ranking": RANKING},
        "matrix": {"levels": 55, "models": 4, "expected_attempts": 220},
        "authority_boundary": {"formal_seed_consumption": False},
        "original_root": str(tmp_path / "original"),
        "outputs": {"audit_receipt": str(tmp_path / "engineering-audit.json")},
    }
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    plan_path = plan_path.resolve()
    master = build_master(
        source_master_path=source_path,
        expected_source_sha256=_sha256(source_path),
        engineering_plan_path=plan_path,
        expected_plan_sha256=_sha256(plan_path),
        campaign_id="successor",
        output_root=tmp_path / "run",
        independent_audit_receipt=tmp_path / "independent-audit.json",
        minimum_cleared_levels=35,
        minimum_wins=35,
    )
    assert master["status"] == "FROZEN_BEFORE_SUCCESSOR_FINAL_BLIND"
    assert master["seed_registry"]["final_blind"]["first"] == 2_100_000_000
    assert master["seed_registry"]["continuous_campaign_challenge"]["last"] == 2_200_000_219
    assert master["successor"]["promotion_gate"]["minimum_cleared_levels"] == 35
    assert master["successor"]["engineering_validation"]["ranking"] == RANKING
    assert master["power_policy"] == {
        "rtx_5090_limit_watts": 550,
        "rtx_5080_limit_watts": 250,
        "does_not_authorize_early_power_restore": True,
    }
    assert master["execution"]["devices"] == ["cuda:0", "cuda:0"]
