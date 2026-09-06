from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import build_alphazuma_55_polar_intent_autonomy_plan as planner
from tools import (
    build_alphazuma_55_polar_intent_autonomy_validation_plan as validation_planner,
)
from tools import (
    build_alphazuma_55_polar_intent_autonomy_successor_intent as successor_intent,
)
from tools import (
    build_alphazuma_55_configured_successor_master_autonomy_v1 as successor_master,
)
from tools import (
    run_alphazuma_55_configured_successor_certification_autonomy_v1 as successor_controller,
)
from tools import (
    audit_alphazuma_55_configured_successor_result_autonomy_v1 as successor_auditor,
)
from tools import audit_alphazuma_55_single_policy_goal_v3 as goal_auditor_v3
from tools import distill_alphazuma_55_polar_intent_autonomy_v1 as trainer
from tools import materialize_alphazuma_55_polar_intent_autonomy as materializer
from tools import (
    materialize_alphazuma_55_polar_intent_autonomy_validation as validation_binding,
)


def test_schedule_is_exactly_four_student_only_rounds() -> None:
    assert trainer._validate_probability_schedule((0.0, 0.0, 0.0, 0.0)) == (
        0.0,
        0.0,
        0.0,
        0.0,
    )
    with pytest.raises(ValueError, match="exactly"):
        trainer._validate_probability_schedule((0.25, 0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="exactly"):
        trainer._validate_probability_schedule((0.0, 0.0, 0.0))


def test_run_adapter_restores_base_runtime(
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
            trainer.intent_dagger._collect_intent_dagger_round
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
        "zuma-rl.alphazuma-55-polar-intent-autonomy-completion"
    )
    assert result["state_distribution_semantics"] == "student_only_dagger"
    assert result["teacher_execution_probabilities"] == [0.0] * 4
    receipt = json.loads((run_dir / "completion.json").read_text())
    assert receipt == result
    assert trainer.dagger._validate_preregistration is original_validate
    assert trainer.dagger._collect_dagger_round is original_collector
    assert trainer.dagger._validate_probability_schedule is original_schedule
    assert trainer.dagger.SCRIPT_PATH == original_script_path


def test_plan_freezes_isolated_student_state_contract(tmp_path: Path) -> None:
    output = tmp_path / "plan.json"
    plan = planner.build(
        project_root=planner.SCRIPT_PATH.parents[1],
        output=output,
    )
    run = plan["run"]
    assert run["teacher_execution_probabilities"] == [0.0] * 4
    assert run["training_seed_base"] == 1_542_005_000
    assert run["training_seed_last_consumed"] == 1_542_005_219
    assert run["continuation_changes_only_state_coverage"] is True
    assert run["source_is_frozen_raw_intent_dagger_final_model"] is True
    validation = plan["frozen_engineering_validation"]
    assert validation["base_seed"] == 1_550_002_000
    assert validation["last_seed"] == 1_550_002_054
    assert validation["expected_attempts"] == 275
    assert plan["authority_boundary"][
        "formal_selection_seed_consumption"
    ] is False
    plan_hash = trainer.legacy._sha256(output)
    assert materializer.validate_plan_static(output, plan_hash)[
        "campaign_id"
    ] == plan["campaign_id"]


def test_materializer_rejects_hash_mismatch(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="bytes differ"):
        materializer._bound_path(
            {"path": str(artifact), "sha256": "sha256:wrong"},
            "fixture",
        )


def test_validation_plan_freezes_five_model_matrix() -> None:
    output = (
        validation_planner.SCRIPT_PATH.parents[1]
        / "diagnostics/alphazuma-55-polar-intent-autonomy-validation-"
        "s99081602-plan-v1.json"
    )
    value = json.loads(output.read_text(encoding="utf-8-sig"))
    assert value["status"] == "FROZEN_BEFORE_TARGET_PREREGISTRATION"
    assert value["matrix"] == {
        "registry": "training_validation",
        "base_seed": 1_550_002_000,
        "last_seed": 1_550_002_054,
        "levels": 55,
        "models": 5,
        "expected_attempts": 275,
        "max_ticks": 30000,
        "paired_models_share_identical_task_seeds": True,
    }
    assert value["execution"]["device"] == "cuda:1"
    assert value["authority_boundary"]["formal_seed_consumption"] is False
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
            / "diagnostics/alphazuma-55-polar-intent-autonomy-validation-"
            "s99081602-plan-v1.json"
        ),
        formal_registry_path=(
            root
            / "diagnostics/alphazuma-55-weekend-formal-seed-registry-"
            "s99081552-v2.json"
        ),
        predecessor_intent_path=(
            root
            / "diagnostics/alphazuma-55-polar-intent-dagger-successor-"
            "s99081551-intent-v1.json"
        ),
        output_root=tmp_path / "successor",
        independent_audit_receipt=tmp_path / "audit.json",
        campaign_id="fixture",
    )
    formal = value["formal_campaign_if_promoted"]
    assert formal["final_blind"]["base_seed"] == 2_900_000_000
    assert formal["final_blind"]["last_seed"] == 2_900_000_439
    assert formal["continuous_campaign_challenge"]["base_seed"] == 3_000_000_000
    assert formal["continuous_campaign_challenge"]["last_seed"] == 3_000_000_219
    assert value["promotion_gate"]["selected_model_must_be_one_of"] == [
        "polar-intent-autonomy-round-00",
        "polar-intent-autonomy-round-01",
        "polar-intent-autonomy-round-02",
        "polar-intent-autonomy-round-03",
    ]
    assert value["engineering_validation"]["paired_attempts"] == 275
    assert value["implementation_boundary"][
        "this_intent_does_not_authorize_policy_inference"
    ] is True


def test_five_candidate_legacy_length_adapter_is_narrow() -> None:
    assert successor_controller._legacy_candidate_len(
        successor_controller.EXPECTED_CANDIDATES
    ) == 4
    assert successor_controller._legacy_candidate_len(
        set(successor_controller.EXPECTED_CANDIDATES)
    ) == 4
    assert successor_controller._legacy_candidate_len(["a"] * 5) == 5
    assert successor_controller._legacy_candidate_len(list(range(55))) == 55
    assert successor_auditor._legacy_candidate_len(
        successor_auditor.EXPECTED_CANDIDATES
    ) == 4


def test_autonomy_successor_master_preserves_actual_five_candidates() -> None:
    root = successor_master.PROJECT_ROOT
    master_path = (
        root
        / "diagnostics/alphazuma-55-polar-intent-autonomy-successor-"
        "s99081603-preregistration-v1.json"
    )
    master = json.loads(master_path.read_text(encoding="utf-8-sig"))
    engineering = master["successor"]["engineering_validation"]
    assert engineering["eligible_model_ids"] == (
        successor_controller.EXPECTED_CANDIDATES
    )
    assert engineering["matrix_models_actual"] == 5
    assert engineering["matrix_attempts_actual"] == 275
    assert master["seed_registry"]["final_blind"]["first"] == 2_900_000_000
    assert master["seed_registry"]["continuous_campaign_challenge"][
        "first"
    ] == 3_000_000_000
    assert master["compatibility_adapter"]["policy_inference_performed"] is False
    contract = successor_controller._load_master(master_path)
    assert contract["expected_model_ids"] == (
        successor_controller.EXPECTED_CANDIDATES
    )
    assert contract["engineering_plan"]["matrix"]["models"] == 5


def test_autonomy_controller_forwards_poll_seconds(
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

    monkeypatch.setattr(successor_controller.base, "run", fake_run)
    monkeypatch.setattr(
        successor_controller,
        "_load_master",
        lambda path: {"fixture": str(path)},
    )
    master_path = tmp_path / "master.json"
    original_root = tmp_path / "retail"
    assert successor_controller.run(
        master_path=master_path,
        original_root=original_root,
        poll_seconds=17.5,
    ) == 0
    assert captured == {
        "master_path": master_path,
        "original_root": original_root,
        "poll_seconds": 17.5,
    }


def test_goal_v3_registers_autonomy_independent_recomputer() -> None:
    recomputers = goal_auditor_v3._production_recomputers()
    row = recomputers["configured_successor_autonomy_v1"]
    assert row["path"] == successor_auditor.SCRIPT_PATH
    assert row["call"] is successor_auditor.audit
    assert "configured_successor_v3" in recomputers
