from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import audit_alphazuma_55_configured_successor_result as independent
from tools import build_alphazuma_55_configured_successor_eval_contract as contracts
from tools.build_alphazuma_55_configured_successor_master import (
    SUPPORTED_RANKING,
    build_master,
)
from tools.build_alphazuma_55_eval_contract import _sha256
from tools.run_alphazuma_55_configured_successor_certification import (
    _engineering_decision,
    _load_master,
)
from tools import run_alphazuma_55_configured_successor_certification as controller


IDS = [
    "polar-source-final",
    "polar-intent-wide-epoch-10",
    "polar-intent-wide-epoch-20",
    "polar-intent-wide-epoch-30",
]


def _write(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path.resolve()


def _configured_fixture(tmp_path: Path) -> tuple[Path, dict]:
    original = tmp_path / "original"
    original.mkdir()
    levels = [f"level-{index:02d}" for index in range(55)]
    source_path = tmp_path / "source.json"
    source = {
        "schema": "zuma-rl.alphazuma-55-weekend-master-preregistration",
        "version": 1,
        "campaign_id": "source",
        "deadline_utc": "2026-08-17T15:00:00Z",
        "deadline_local": "2026-08-17 11:00 America/Toronto",
        "scope": {"included_levels_in_adventure_order": levels},
        "seed_registry": {
            "selection": {"first": 1_600_000_000, "last": 1_600_000_999},
            "final_blind": {"first": 1_700_000_000, "last": 1_700_000_999},
            "continuous": {"first": 1_800_000_000, "last": 1_800_000_999},
        },
    }
    _write(source_path, source)
    source_reference = {"path": str(source_path.resolve()), "sha256": _sha256(source_path)}

    plan_path = tmp_path / "plan.json"
    audit_path = tmp_path / "engineering-audit.json"
    plan = {
        "schema": "zuma-rl.alphazuma-55-polar-intent-wide-validation-plan",
        "version": 1,
        "status": "FROZEN_DURING_TARGET_TRAINING",
        "master_preregistration": source_reference,
        "target": {"expected_failure": str(tmp_path / "training-failure.json")},
        "decision_rule": {"ranking": SUPPORTED_RANKING},
        "matrix": {"levels": 55, "models": 4, "expected_attempts": 220},
        "authority_boundary": {"formal_seed_consumption": False},
        "original_root": str(original.resolve()),
        "outputs": {"audit_receipt": str(audit_path.resolve())},
    }
    _write(plan_path, plan)
    plan_reference = {"path": str(plan_path.resolve()), "sha256": _sha256(plan_path)}

    sibling_path = tmp_path / "sibling.json"
    sibling = {
        "formal_campaign_if_promoted": {
            "final_blind": {"base_seed": 2_300_000_000, "last_seed": 2_300_000_439},
            "continuous": {"base_seed": 2_400_000_000, "last_seed": 2_400_000_219},
        }
    }
    _write(sibling_path, sibling)
    intent_path = tmp_path / "intent.json"
    intent = {
        "schema": "zuma-rl.alphazuma-55-polar-intent-wide-successor-intent",
        "version": 1,
        "status": "FROZEN_BEFORE_ENGINEERING_RESULT",
        "campaign_id": "configured-successor",
        "source_campaign_master": source_reference,
        "sibling_reserved_seed_intent": {
            "path": str(sibling_path.resolve()),
            "sha256": _sha256(sibling_path),
        },
        "engineering_validation": {
            "plan": plan_reference,
            "audit_receipt_expected": str(audit_path.resolve()),
            "candidate_order": IDS,
            "ranking": SUPPORTED_RANKING,
            "paired_attempts": 220,
            "formal_authority": False,
        },
        "promotion_gate": {
            "selected_model_must_be_one_of": IDS[1:],
            "minimum_cleared_levels": 35,
            "minimum_wins": 35,
            "evaluated_before_formal_contract_or_seed_consumption": True,
        },
        "formal_campaign_if_promoted": {
            "single_selected_policy_only": True,
            "scope_levels": levels,
            "final_blind": {
                "base_seed": 2_500_000_000,
                "last_seed": 2_500_000_439,
                "attempts_per_level": 8,
                "expected_attempts": 440,
                "gate": {
                    "cleared_levels": 55,
                    "minimum_win_per_level": 1,
                    "minimum_total_wins": 220,
                },
            },
            "continuous_campaign_challenge": {
                "base_seed": 2_600_000_000,
                "last_seed": 2_600_000_219,
                "campaigns": 4,
                "levels_per_campaign": 55,
                "expected_attempts": 220,
                "gate": {"minimum_complete_55_level_campaigns": 1},
            },
            "execution": {
                "shard_count": 2,
                "devices": ["cuda:0", "cuda:1"],
                "parallel_envs_per_shard": 12,
            },
            "all_seeds_frozen_before_policy_inference": True,
        },
        "outputs": {
            "root": str((tmp_path / "run").resolve()),
            "independent_audit_receipt": str((tmp_path / "independent.json").resolve()),
        },
        "implementation_boundary": {
            "formal_master_controller_and_auditor_must_be_hash_frozen_before_inference": True,
            "this_intent_does_not_authorize_policy_inference": True,
        },
        "authority_boundary": {
            "current_campaign_candidate_authority": False,
            "formal_seed_consumption": False,
        },
    }
    _write(intent_path, intent)
    return intent_path.resolve(), intent


def test_configured_master_transfers_frozen_intent_without_seed_drift(
    tmp_path: Path,
) -> None:
    intent_path, _ = _configured_fixture(tmp_path)
    master = build_master(
        successor_intent_path=intent_path,
        expected_intent_sha256=_sha256(intent_path),
    )
    assert master["seed_registry"]["final_blind"] == {
        "first": 2_500_000_000,
        "last": 2_500_000_439,
        "matrix": "55 levels x 8 attempts for one frozen successor policy",
        "embargo_until_promotion_decision_is_frozen": True,
    }
    assert master["seed_registry"]["continuous_campaign_challenge"]["first"] == 2_600_000_000
    assert master["execution"]["devices"] == ["cuda:0", "cuda:1"]
    assert master["successor"]["promotion_gate"]["selected_model_must_be_one_of"] == IDS[1:]
    assert (
        master["successor"]["engineering_validation"]["manifest_plan_binding_key"]
        == "polar_intent_wide_validation_plan"
    )

    master_path = _write(tmp_path / "master.json", master)
    loaded = _load_master(master_path)
    assert loaded["final_seed"] == 2_500_000_000
    assert loaded["continuous_seed"] == 2_600_000_000
    assert loaded["promotable_model_ids"] == IDS[1:]


def test_configured_master_rejects_formal_seed_overlap(tmp_path: Path) -> None:
    intent_path, intent = _configured_fixture(tmp_path)
    intent["formal_campaign_if_promoted"]["final_blind"].update(
        {"base_seed": 2_300_000_000, "last_seed": 2_300_000_439}
    )
    _write(intent_path, intent)
    with pytest.raises(ValueError, match="overlap"):
        build_master(
            successor_intent_path=intent_path,
            expected_intent_sha256=_sha256(intent_path),
        )


def _models() -> list[dict]:
    return [
        {
            "id": model_id,
            "path": f"/{model_id}.zip",
            "sha256": f"hash-{index}",
            "training_steps": index,
            "bytes": [100, 80, 60, 40][index],
        }
        for index, model_id in enumerate(IDS)
    ]


def _engineering_rows(win_counts: list[int]) -> list[dict]:
    rows = []
    for model_index, (model, wins) in enumerate(
        zip(_models(), win_counts, strict=True)
    ):
        for level_index in range(55):
            won = level_index < wins
            rows.append(
                {
                    "model_id": model["id"],
                    "level_id": f"level-{level_index:02d}",
                    "outcome": "win" if won else "loss",
                    "ticks": 1000 + level_index,
                    "score": 100 if won else 0,
                }
            )
    return rows


def test_configured_ranking_uses_smaller_model_tie_break() -> None:
    models = _models()
    rows = _engineering_rows([40, 40, 40, 40])
    decision = _engineering_decision(
        models=models,
        rows=rows,
        minimum_levels=35,
        minimum_wins=35,
        promotable_model_ids=IDS[1:],
    )
    assert decision["selected_policy"]["id"] == IDS[3]
    assert decision["promotion_gate"]["passed"] is True
    assert decision["ranking"] == independent._rank(models, rows)


def test_configured_gate_does_not_promote_the_source_baseline() -> None:
    decision = _engineering_decision(
        models=_models(),
        rows=_engineering_rows([55, 40, 39, 38]),
        minimum_levels=35,
        minimum_wins=35,
        promotable_model_ids=IDS[1:],
    )
    assert decision["selected_policy"]["id"] == IDS[0]
    assert decision["status"] == "NO_PROMOTION"
    assert decision["successor_formal_seed_consumption_authorized"] is False


def test_configured_contract_uses_master_topology(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    master_path = _write(
        tmp_path / "master.json",
        {
            "schema": "zuma-rl.alphazuma-55-configured-successor-master-preregistration",
            "version": 1,
            "status": "FROZEN_BEFORE_SUCCESSOR_FINAL_BLIND",
            "seed_registry": {
                "final_blind": {"first": 2_500_000_000, "last": 2_500_000_439}
            },
            "execution": {
                "shard_count": 2,
                "devices": ["cuda:0", "cuda:1"],
                "parallel_envs_per_shard": 12,
            },
        },
    )
    original = tmp_path / "original"
    original.mkdir()
    evaluator = tmp_path / "evaluator.py"
    evaluator.write_text("# evaluator\n", encoding="utf-8")
    model = tmp_path / "model.zip"
    model.write_bytes(b"model")

    def fake_build_contracts(**kwargs: object) -> tuple[dict, dict]:
        return (
            {"models": kwargs["models"]},
            {
                "attempts_per_level": 8,
                "total_attempts": 440,
                "seed_plan": {
                    "base_seed": 2_500_000_000,
                    "last_seed": 2_500_000_439,
                },
                "execution": {
                    "shard_count": 2,
                    "devices": ["cuda:0", "cuda:1"],
                    "parallel_envs_per_shard": 12,
                },
            },
        )

    monkeypatch.setattr(contracts.legacy, "build_contracts", fake_build_contracts)
    _, prereg = contracts.build_successor_contracts(
        master_path=master_path,
        original_root=original,
        evaluator_path=evaluator,
        stage="final_blind",
        base_seed=2_500_000_000,
        attempts_per_level=8,
        output_root=tmp_path / "output",
        models=[
            {
                "id": "candidate",
                "training_steps": 1,
                "path": str(model),
                "sha256": _sha256(model),
                "bytes": model.stat().st_size,
            }
        ],
        parallel_envs_per_shard=12,
    )
    assert prereg["execution"]["devices"] == ["cuda:0", "cuda:1"]
    assert prereg["seed_plan"]["all_seeds_frozen_before_policy_inference"] is True


def test_no_promotion_controller_leaves_both_formal_roots_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    intent_path, _ = _configured_fixture(tmp_path)
    master = build_master(
        successor_intent_path=intent_path,
        expected_intent_sha256=_sha256(intent_path),
    )
    master_path = _write(tmp_path / "master.json", master)
    models = _models()
    rows = _engineering_rows([55, 40, 39, 38])
    engineering_prereg = _write(tmp_path / "engineering-prereg.json", {})
    engineering_manifest = _write(
        tmp_path / "engineering-models.json",
        {
            "models": models,
            "polar_intent_wide_validation_plan": master["successor"][
                "engineering_validation"
            ]["plan"],
        },
    )
    engineering_matrix = _write(
        tmp_path / "engineering-matrix.json", {"attempts": rows}
    )
    engineering_audit = Path(
        master["successor"]["engineering_validation"]["audit_receipt"]
    )
    engineering_receipt = {
        "schema": "zuma-rl.alphazuma-55-training-validation-audit",
        "status": "PASS",
        "formal_selection_authority": False,
        "artifacts": {
            "master": master["successor"]["source_campaign_master"],
            "preregistration": {
                "path": str(engineering_prereg),
                "sha256": _sha256(engineering_prereg),
            },
            "models_manifest": {
                "path": str(engineering_manifest),
                "sha256": _sha256(engineering_manifest),
            },
            "matrix_shard": {
                "path": str(engineering_matrix),
                "sha256": _sha256(engineering_matrix),
            },
            "auditor": master["implementation"]["engineering_auditor"],
        },
        "model_summaries": [],
        "matrix": {"attempts": 220},
    }
    _write(engineering_audit, engineering_receipt)
    monkeypatch.setattr(
        controller,
        "_load_engineering_evidence",
        lambda _: (models, rows, engineering_receipt),
    )
    assert controller.run(
        master_path,
        Path(master["original_root"]),
        poll_seconds=0.001,
    ) == 0
    root = Path(master["outputs"]["root"])
    status = json.loads(
        (root / "controller" / "controller_status.json").read_text(
            encoding="utf-8"
        )
    )
    assert status["status"] == "COMPLETE_NO_PROMOTION"
    assert status["formal_seed_consumption"] == "NONE"
    assert not (root / "final-blind").exists()
    assert not (root / "continuous").exists()
    monkeypatch.setattr(
        independent,
        "audit_engineering",
        lambda **_: {"model_summaries": [], "matrix": {"attempts": 220}},
    )
    audit = independent.audit(master_path)
    assert audit["status"] == "PASS"
    assert audit["controller_result"] == "COMPLETE_NO_PROMOTION"
    assert audit["formal_seed_consumption"] == "NONE"
