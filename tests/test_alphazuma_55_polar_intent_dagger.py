from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools import build_alphazuma_55_polar_intent_dagger_plan as planner
from tools import (
    build_alphazuma_55_polar_intent_dagger_validation_plan as validation_planner,
)
from tools import (
    build_alphazuma_55_polar_intent_dagger_successor_intent as successor_intent,
)
from tools import distill_alphazuma_55_polar_intent_dagger_v1 as trainer
from tools import materialize_alphazuma_55_polar_intent_dagger as materializer
from tools import (
    materialize_alphazuma_55_polar_intent_dagger_validation as validation_binding,
)
from tools import (
    build_alphazuma_55_configured_successor_master_v2 as successor_master_v2,
)
from tools import (
    run_alphazuma_55_configured_successor_certification_v2 as successor_controller_v2,
)
from tools import (
    audit_alphazuma_55_configured_successor_result_v2 as successor_auditor_v2,
)
from tools import (
    build_alphazuma_55_configured_successor_master_v3 as successor_master_v3,
)
from tools import (
    run_alphazuma_55_configured_successor_certification_v3 as successor_controller_v3,
)


def test_schedule_is_frozen_without_redundant_teacher_only_round() -> None:
    assert trainer._validate_probability_schedule((0.75, 0.25, 0.0)) == (
        0.75,
        0.25,
        0.0,
    )
    with pytest.raises(ValueError, match="exactly"):
        trainer._validate_probability_schedule((1.0, 0.5, 0.0))
    with pytest.raises(ValueError, match="exactly"):
        trainer._validate_probability_schedule((0.75, 0.0))


def test_raw_intent_label_survives_while_teacher_execution_stays_legal() -> None:
    exact_mask = np.ones(184, dtype=np.bool_)
    exact_mask[1] = False
    raw, effective, training_mask, forced = trainer.intent._intent_label(
        np.asarray((1, 42), dtype=np.int64), exact_mask
    )
    actions, execute_teacher = trainer.dagger._mix_actions(
        teacher_actions=np.asarray((effective,), dtype=np.int64),
        student_actions=np.asarray(((0, 42),), dtype=np.int64),
        active=np.asarray((True,), dtype=np.bool_),
        teacher_probability=1.0,
        rng=np.random.default_rng(7),
    )
    np.testing.assert_array_equal(raw, (1, 42))
    np.testing.assert_array_equal(effective, (0, 42))
    np.testing.assert_array_equal(actions, ((0, 42),))
    np.testing.assert_array_equal(execute_teacher, (True,))
    assert forced is True
    assert bool(training_mask[1]) is True
    assert bool(exact_mask[1]) is False


def test_run_adapter_restores_frozen_base_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prereg_path = tmp_path / "prereg.json"
    prereg_path.write_text("{}\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    prereg = {"run": {"run_dir": str(run_dir)}}
    monkeypatch.setattr(trainer, "_validate_preregistration", lambda _: prereg)
    original_validate = trainer.dagger._validate_preregistration
    original_collector = trainer.dagger._collect_dagger_round
    original_schedule = trainer.dagger._validate_probability_schedule
    original_script_path = trainer.dagger.SCRIPT_PATH

    def fake_dagger_run(*, preregistration_path: Path, original_root: Path):
        assert trainer.dagger._collect_dagger_round is (
            trainer._collect_intent_dagger_round
        )
        assert trainer.dagger._validate_probability_schedule is (
            trainer._validate_probability_schedule
        )
        assert trainer.dagger.SCRIPT_PATH == trainer.SCRIPT_PATH
        assert preregistration_path == prereg_path.resolve()
        assert original_root == tmp_path
        run_dir.mkdir()
        return {
            "status": "COMPLETE",
            "formal_seed_consumption": False,
            "formal_candidate_authority": False,
            "final_model": {
                "path": str(run_dir / "final_model.zip"),
                "sha256": "sha256:fixture",
            },
        }

    monkeypatch.setattr(trainer.dagger, "run", fake_dagger_run)
    result = trainer.run(
        preregistration_path=prereg_path,
        original_root=tmp_path,
    )
    assert result["schema"] == (
        "zuma-rl.alphazuma-55-polar-intent-dagger-completion"
    )
    assert result["training_label_semantics"] == "raw_teacher_intent"
    assert result["state_distribution_semantics"] == (
        "dagger_teacher_student_mixture"
    )
    receipt = json.loads((run_dir / "completion.json").read_text())
    assert receipt == result
    assert trainer.dagger._validate_preregistration is original_validate
    assert trainer.dagger._collect_dagger_round is original_collector
    assert trainer.dagger._validate_probability_schedule is original_schedule
    assert trainer.dagger.SCRIPT_PATH == original_script_path


def test_planner_freezes_isolated_seed_and_semantic_contract(
    tmp_path: Path,
) -> None:
    output = tmp_path / "plan.json"
    plan = planner.build(
        project_root=planner.SCRIPT_PATH.parents[1],
        output=output,
    )
    run = plan["run"]
    assert run["teacher_execution_probabilities"] == [0.75, 0.25, 0.0]
    assert run["training_seed_base"] == 1_542_004_000
    assert run["training_seed_last_consumed"] == 1_542_004_164
    assert run["raw_teacher_intent_labels"] is True
    assert run["student_state_collection"] is True
    assert run["exact_action_masks_for_online_policy_inference"] is True
    frozen = plan["frozen_engineering_validation"]
    assert frozen["base_seed"] == 1_550_001_800
    assert frozen["last_seed"] == 1_550_001_854
    assert plan["authority_boundary"]["formal_selection_seed_consumption"] is False
    assert materializer.validate_plan_static(
        output, trainer.legacy._sha256(output)
    )["campaign_id"] == plan["campaign_id"]


def test_materializer_rejects_hash_mismatch(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="bytes differ"):
        materializer._bound_path(
            {"path": str(artifact), "sha256": "sha256:wrong"},
            "fixture",
        )


def test_validation_plan_is_frozen_before_training_preregistration(
    tmp_path: Path,
) -> None:
    value = validation_planner.build(
        project_root=validation_planner.SCRIPT_PATH.parents[1]
    )
    assert value["status"] == "FROZEN_BEFORE_TARGET_PREREGISTRATION"
    assert value["matrix"] == {
        "registry": "training_validation",
        "base_seed": 1_550_001_800,
        "last_seed": 1_550_001_854,
        "levels": 55,
        "models": 4,
        "expected_attempts": 220,
        "max_ticks": 30000,
        "paired_models_share_identical_task_seeds": True,
    }
    assert value["execution"]["device"] == "cuda:1"
    assert value["authority_boundary"]["formal_seed_consumption"] is False
    output = tmp_path / "validation-plan.json"
    output.write_text(json.dumps(value) + "\n", encoding="utf-8")
    assert validation_binding.validate_plan_static(
        output, trainer.legacy._sha256(output)
    )["training_plan"]["sha256"] == value["training_plan"]["sha256"]


def test_successor_intent_reserves_fresh_formal_ranges(tmp_path: Path) -> None:
    root = successor_intent.SCRIPT_PATH.parents[1]
    value = successor_intent.build(
        source_master_path=(
            root
            / "diagnostics/alphazuma-55-weekend-"
            "s81081401-preregistration-v1.json"
        ),
        validation_plan_path=(
            root
            / "diagnostics/alphazuma-55-polar-intent-dagger-validation-"
            "s99081550-plan-v1.json"
        ),
        formal_registry_path=(
            root
            / "diagnostics/alphazuma-55-weekend-formal-seed-registry-"
            "s99081546-v1.json"
        ),
        sibling_intent_path=(
            root
            / "diagnostics/alphazuma-55-polar-intent-wide-successor-"
            "s99081545-intent-v1.json"
        ),
        output_root=tmp_path / "successor",
        independent_audit_receipt=tmp_path / "audit.json",
        campaign_id="fixture",
    )
    formal = value["formal_campaign_if_promoted"]
    assert formal["final_blind"]["base_seed"] == 2_700_000_000
    assert formal["final_blind"]["last_seed"] == 2_700_000_439
    assert formal["continuous_campaign_challenge"]["base_seed"] == 2_800_000_000
    assert formal["continuous_campaign_challenge"]["last_seed"] == 2_800_000_219
    assert value["promotion_gate"]["selected_model_must_be_one_of"] == [
        "polar-intent-dagger-round-00",
        "polar-intent-dagger-round-01",
        "polar-intent-dagger-round-02",
    ]
    assert value["implementation_boundary"][
        "this_intent_does_not_authorize_policy_inference"
    ] is True


def _tied_engineering_fixture() -> tuple[list[dict], list[dict]]:
    models = [
        {"id": "round-00", "bytes": 2_000},
        {"id": "round-01", "bytes": 1_000},
        {"id": "round-02", "bytes": 500},
        {"id": "baseline", "bytes": 100},
    ]
    rows = [
        {
            "model_id": model["id"],
            "level_id": f"level-{level_index:02d}",
            "outcome": "win",
            "ticks": 1_000,
            "score": 2_000,
        }
        for model in models
        for level_index in range(55)
    ]
    return models, rows


def test_v2_ranking_uses_earlier_round_before_model_size() -> None:
    models, rows = _tied_engineering_fixture()
    decision = successor_controller_v2._engineering_decision(
        models=models,
        rows=rows,
        minimum_levels=35,
        minimum_wins=35,
        promotable_model_ids={"round-00", "round-01", "round-02"},
        ranking_rule=successor_controller_v2.EXPECTED_RANKING,
    )
    assert decision["selected_policy"]["id"] == "round-00"
    assert [row["model"]["id"] for row in decision["ranking"]] == [
        "round-00",
        "round-01",
        "round-02",
        "baseline",
    ]
    assert decision["successor_formal_seed_consumption_authorized"] is True
    audited = successor_auditor_v2._rank(models, rows)
    assert [row["model"]["id"] for row in audited] == [
        "round-00",
        "round-01",
        "round-02",
        "baseline",
    ]


def test_v2_master_builds_and_loads_frozen_intent(tmp_path: Path) -> None:
    root = successor_master_v2.PROJECT_ROOT
    intent_path = (
        root
        / "diagnostics/alphazuma-55-polar-intent-dagger-successor-"
        "s99081551-intent-v1.json"
    )
    intent_hash = successor_master_v2.base._sha256(intent_path)
    master = successor_master_v2.build_master(
        successor_intent_path=intent_path,
        expected_intent_sha256=intent_hash,
    )
    assert master["compatibility_adapter"]["policy_inference_performed"] is False
    assert master["successor"]["engineering_validation"][
        "ranking_tie_break_semantics"
    ] == "earlier_manifest_round_first"
    master_path = tmp_path / "master.json"
    master_path.write_text(
        json.dumps(master, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    contract = successor_controller_v2._load_master(master_path)
    assert contract["ranking"] == successor_controller_v2.EXPECTED_RANKING
    assert contract["engineering_plan"]["status"] == (
        "FROZEN_BEFORE_TARGET_PREREGISTRATION"
    )


def test_v3_forwards_poll_seconds_to_base_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def fake_run(*, master_path: Path, original_root: Path, poll_seconds: float):
        captured.update(
            {
                "master_path": master_path,
                "original_root": original_root,
                "poll_seconds": poll_seconds,
            }
        )
        return 0

    monkeypatch.setattr(successor_controller_v3.base, "run", fake_run)
    master_path = tmp_path / "master.json"
    original_root = tmp_path / "retail"
    assert successor_controller_v3.run(
        master_path=master_path,
        original_root=original_root,
        poll_seconds=17.5,
    ) == 0
    assert captured == {
        "master_path": master_path,
        "original_root": original_root,
        "poll_seconds": 17.5,
    }


def test_v3_master_builds_and_loads_frozen_intent(tmp_path: Path) -> None:
    root = successor_master_v3.PROJECT_ROOT
    intent_path = (
        root
        / "diagnostics/alphazuma-55-polar-intent-dagger-successor-"
        "s99081551-intent-v1.json"
    )
    intent_hash = successor_master_v3.v2.base._sha256(intent_path)
    master = successor_master_v3.build_master(
        successor_intent_path=intent_path,
        expected_intent_sha256=intent_hash,
    )
    correction = master["preflight_correction"]
    assert correction["formal_seed_consumption_before_correction"] is False
    assert correction["policy_inference_before_correction"] is False
    assert correction["correction"] == "forward poll_seconds to base.run"
    master_path = tmp_path / "master-v3.json"
    master_path.write_text(
        json.dumps(master, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    contract = successor_controller_v3._load_master(master_path)
    assert contract["master"] == master
    assert contract["ranking"] == successor_controller_v3.EXPECTED_RANKING
