from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import (
    build_alphazuma_55_polar_intent_replay_anchor_plan as planner,
)
from tools import (
    distill_alphazuma_55_polar_intent_replay_anchor_v1 as trainer,
)
from tools import (
    materialize_alphazuma_55_polar_intent_replay_anchor as materializer,
)
from tools import (
    build_alphazuma_55_polar_intent_replay_anchor_validation_plan as validation_planner,
)
from tools import (
    materialize_alphazuma_55_polar_intent_replay_anchor_validation as validation_materializer,
)
from tools import (
    build_alphazuma_55_polar_intent_replay_anchor_successor_intent as successor_intent,
)
from tools import (
    build_alphazuma_55_configured_successor_master_replay_anchor_v1 as successor_master,
)
from tools import (
    run_alphazuma_55_configured_successor_certification_replay_anchor_v1 as successor_controller,
)
from tools import (
    audit_alphazuma_55_configured_successor_result_replay_anchor_v1 as successor_auditor,
)
from tools import audit_alphazuma_55_single_policy_goal_v4 as goal_auditor_v4


def test_schedule_and_anchor_count_are_frozen() -> None:
    assert trainer.EXPECTED_ANCHOR_PASSES == 2
    assert trainer._validate_probability_schedule((0.5, 0.1, 0.0)) == (
        0.5,
        0.1,
        0.0,
    )
    with pytest.raises(ValueError, match="exactly"):
        trainer._validate_probability_schedule((0.5, 0.0, 0.0))


def test_plan_freezes_replay_anchor_and_fresh_seeds(tmp_path: Path) -> None:
    output = tmp_path / "plan.json"
    plan = planner.build(
        project_root=planner.SCRIPT_PATH.parents[1],
        output=output,
    )
    run = plan["run"]
    assert run["anchor_teacher_passes"] == 2
    assert run["teacher_execution_probabilities"] == [0.5, 0.1, 0.0]
    assert run["anchor_seed_base"] == 1_542_006_000
    assert run["anchor_seed_last_consumed"] == 1_542_006_109
    assert run["student_seed_base"] == 1_542_006_110
    assert run["student_seed_last_consumed"] == 1_542_006_274
    assert run["teacher_anchor_replayed_every_round"] is True
    validation = plan["frozen_engineering_validation"]
    assert validation["base_seed"] == 1_550_002_200
    assert validation["last_seed"] == 1_550_002_254
    assert validation["expected_attempts"] == 275
    assert plan["authority_boundary"][
        "formal_selection_seed_consumption"
    ] is False
    plan_hash = trainer.legacy._sha256(output)
    validated = materializer.validate_plan_static(output, plan_hash)
    assert validated["campaign_id"] == plan["campaign_id"]


def test_materializer_rejects_hash_mismatch(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="bytes differ"):
        materializer._bound_path(
            {"path": str(artifact), "sha256": "sha256:wrong"},
            "fixture",
        )


def test_validation_plan_freezes_five_candidates_on_cuda0(
    tmp_path: Path,
) -> None:
    path = (
        validation_planner.SCRIPT_PATH.parents[1]
        / "diagnostics/alphazuma-55-polar-intent-replay-anchor-"
        "validation-s99081608-plan-v1.json"
    )
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    assert value["matrix"] == {
        "registry": "training_validation",
        "base_seed": 1_550_002_200,
        "last_seed": 1_550_002_254,
        "levels": 55,
        "models": 5,
        "expected_attempts": 275,
        "max_ticks": 30000,
        "paired_models_share_identical_task_seeds": True,
    }
    assert value["execution"]["device"] == "cuda:0"
    validated = validation_materializer.validate_plan_static(
        path, trainer.legacy._sha256(path)
    )
    assert validated["authority_boundary"]["formal_seed_consumption"] is False


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
            / "diagnostics/alphazuma-55-polar-intent-replay-anchor-"
            "validation-s99081608-plan-v1.json"
        ),
        formal_registry_path=(
            root
            / "diagnostics/alphazuma-55-weekend-formal-seed-registry-"
            "s99081604-v3.json"
        ),
        predecessor_intent_path=(
            root
            / "diagnostics/alphazuma-55-polar-intent-autonomy-successor-"
            "s99081603-intent-v1.json"
        ),
        output_root=tmp_path / "successor",
        independent_audit_receipt=tmp_path / "audit.json",
        campaign_id="fixture",
    )
    formal = value["formal_campaign_if_promoted"]
    assert formal["final_blind"]["base_seed"] == 3_100_000_000
    assert formal["final_blind"]["last_seed"] == 3_100_000_439
    assert formal["continuous_campaign_challenge"][
        "base_seed"
    ] == 3_200_000_000
    assert formal["continuous_campaign_challenge"][
        "last_seed"
    ] == 3_200_000_219
    assert value["promotion_gate"]["selected_model_must_be_one_of"] == [
        "polar-intent-replay-anchor-round-00",
        "polar-intent-replay-anchor-round-01",
        "polar-intent-replay-anchor-round-02",
    ]
    assert value["implementation_boundary"][
        "this_intent_does_not_authorize_policy_inference"
    ] is True


def test_replay_anchor_successor_master_preserves_five_candidates() -> None:
    root = successor_master.PROJECT_ROOT
    master_path = (
        root
        / "diagnostics/alphazuma-55-polar-intent-replay-anchor-successor-"
        "s99081609-preregistration-v1.json"
    )
    master = json.loads(master_path.read_text(encoding="utf-8-sig"))
    engineering = master["successor"]["engineering_validation"]
    assert engineering["eligible_model_ids"] == (
        successor_controller.EXPECTED_CANDIDATES
    )
    assert engineering["matrix_models_actual"] == 5
    assert engineering["matrix_attempts_actual"] == 275
    assert master["seed_registry"]["final_blind"]["first"] == 3_100_000_000
    assert master["seed_registry"]["continuous_campaign_challenge"][
        "first"
    ] == 3_200_000_000
    assert master["compatibility_adapter"]["policy_inference_performed"] is False


def test_replay_anchor_successor_controller_loads_frozen_master(
) -> None:
    root = successor_master.PROJECT_ROOT
    master_path = (
        root
        / "diagnostics/alphazuma-55-polar-intent-replay-anchor-successor-"
        "s99081609-preregistration-v1.json"
    )
    contract = successor_controller._load_master(master_path)
    assert contract["expected_model_ids"] == (
        successor_controller.EXPECTED_CANDIDATES
    )
    assert contract["engineering_plan"]["matrix"]["models"] == 5
    assert contract["final_seed"] == 3_100_000_000
    assert contract["continuous_seed"] == 3_200_000_000


def test_replay_anchor_five_candidate_adapter_is_narrow() -> None:
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


def test_replay_anchor_controller_forwards_poll_seconds(
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
        poll_seconds=11.5,
    ) == 0
    assert captured == {
        "master_path": master_path,
        "original_root": original_root,
        "poll_seconds": 11.5,
    }


def test_goal_v4_registers_replay_anchor_recomputer() -> None:
    recomputers = goal_auditor_v4._production_recomputers()
    row = recomputers["configured_successor_replay_anchor_v1"]
    assert row["path"] == successor_auditor.SCRIPT_PATH
    assert row["call"] is successor_auditor.audit
    assert "configured_successor_autonomy_v1" in recomputers
    assert "configured_successor_v3" in recomputers
