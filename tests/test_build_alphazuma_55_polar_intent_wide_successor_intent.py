from __future__ import annotations

import json
from pathlib import Path

from tools import (
    build_alphazuma_55_polar_intent_wide_successor_intent as builder,
)
from tools.build_alphazuma_55_eval_contract import _sha256


def _write(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_intent_freezes_gate_and_disjoint_formal_ranges(tmp_path: Path) -> None:
    levels = [f"level-{index:02d}" for index in range(55)]
    source = tmp_path / "source.json"
    _write(
        source,
        {
            "schema": "zuma-rl.alphazuma-55-weekend-master-preregistration",
            "scope": {"included_levels_in_adventure_order": levels},
            "seed_registry": {
                "training": {"first": 1_500_000_000, "last": 1_549_999_999},
                "final_blind": {"first": 1_700_000_000, "last": 1_700_000_439},
            },
        },
    )
    validation = tmp_path / "validation.json"
    audit = tmp_path / "audit.json"
    _write(
        validation,
        {
            "schema": "zuma-rl.alphazuma-55-polar-intent-wide-validation-plan",
            "status": "FROZEN_DURING_TARGET_TRAINING",
            "master_preregistration": {"sha256": _sha256(source)},
            "matrix": {
                "levels": 55,
                "models": 4,
                "expected_attempts": 220,
                "paired_models_share_identical_task_seeds": True,
            },
            "decision_rule": {"ranking": ["wins_desc", "level_coverage_desc"]},
            "outputs": {"audit_receipt": str(audit)},
            "authority_boundary": {
                "formal_seed_consumption": False,
                "s99081535_successor_candidate_authority": False,
            },
        },
    )
    sibling = tmp_path / "wide-intent.json"
    _write(
        sibling,
        {
            "schema": "zuma-rl.alphazuma-55-polar-wide-successor-intent",
            "status": "FROZEN_BEFORE_ENGINEERING_RESULT",
            "source_campaign_master": {"sha256": _sha256(source)},
            "formal_campaign_if_promoted": {
                "final_blind": {
                    "base_seed": 2_300_000_000,
                    "last_seed": 2_300_000_439,
                },
                "continuous_campaign_challenge": {
                    "base_seed": 2_400_000_000,
                    "last_seed": 2_400_000_219,
                },
            },
        },
    )

    intent = builder.build(
        source_master_path=source,
        validation_plan_path=validation,
        sibling_wide_intent_path=sibling,
        output_root=tmp_path / "successor",
        independent_audit_receipt=tmp_path / "independent.json",
        campaign_id="intent-wide-successor",
    )

    assert intent["promotion_gate"]["selected_model_must_be_one_of"] == [
        "polar-intent-wide-epoch-10",
        "polar-intent-wide-epoch-20",
        "polar-intent-wide-epoch-30",
    ]
    assert intent["promotion_gate"]["minimum_cleared_levels"] == 35
    assert intent["formal_campaign_if_promoted"]["final_blind"] == {
        "base_seed": 2_500_000_000,
        "last_seed": 2_500_000_439,
        "attempts_per_level": 8,
        "expected_attempts": 440,
        "gate": {
            "cleared_levels": 55,
            "minimum_win_per_level": 1,
            "minimum_total_wins": 220,
        },
    }
    assert intent["formal_campaign_if_promoted"][
        "continuous_campaign_challenge"
    ]["last_seed"] == 2_600_000_219
    assert intent["authority_boundary"]["formal_seed_consumption"] is False
