from __future__ import annotations

import json
from pathlib import Path

from tools import (
    build_alphazuma_55_polar_intent_wide_validation_plan as builder,
)
from tools.build_alphazuma_55_eval_contract import _sha256


def _write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_intent_validation_freezes_source_and_three_checkpoints(
    tmp_path: Path,
) -> None:
    master = tmp_path / "master.json"
    _write(
        master,
        {"schema": "zuma-rl.alphazuma-55-weekend-master-preregistration"},
    )
    original_root = tmp_path / "original"
    original_root.mkdir()
    source_run = tmp_path / "source-run"
    source_model = source_run / "final_model.zip"
    source_model.parent.mkdir()
    source_model.write_bytes(b"source-model")
    source_completion = source_run / "completion.json"
    _write(
        source_completion,
        {
            "schema": "zuma-rl.alphazuma-55-polar-distillation-completion",
            "status": "COMPLETE",
            "policy_architecture": "entity_polar",
            "formal_seed_consumption": False,
            "final_model": {
                "path": str(source_model),
                "sha256": _sha256(source_model),
                "num_timesteps": 123,
            },
        },
    )
    source_prereg = tmp_path / "source-prereg.json"
    _write(
        source_prereg,
        {
            "schema": "zuma-rl.alphazuma-55-polar-distillation-preregistration",
            "run": {"run_dir": str(source_run)},
        },
    )
    intent_run = tmp_path / "intent-run"
    intent_prereg = tmp_path / "intent-prereg.json"
    _write(
        intent_prereg,
        {
            "schema": (
                "zuma-rl.alphazuma-55-polar-intent-wide-"
                "distillation-preregistration"
            ),
            "status": "FROZEN_BEFORE_TRAINING",
            "master_preregistration": {"sha256": _sha256(master)},
            "run": {"run_dir": str(intent_run)},
            "frozen_engineering_validation": {
                "registry": "training_validation",
                "base_seed": 1_550_001_600,
                "last_seed": 1_550_001_654,
                "checkpoint_epochs": [10, 20, 30],
                "comparison_model_id": "polar-source-final",
                "ranking": ["wins_desc", "coverage_desc"],
            },
            "authority_boundary": {
                "current_campaign_candidate_authority": False,
                "s99081535_successor_candidate_authority": False,
            },
        },
    )
    launch_plan = tmp_path / "launch.json"
    launch_status_root = tmp_path / "launch-status"
    _write(
        launch_plan,
        {
            "schema": "zuma-rl.alphazuma-55-polar-intent-wide-launch-plan",
            "status": "FROZEN_BEFORE_WAIT",
            "intent_preregistration": {"sha256": _sha256(intent_prereg)},
            "target": {"run_dir": str(intent_run)},
            "outputs": {"status_root": str(launch_status_root)},
            "authority_boundary": {"formal_seed_consumption": False},
        },
    )

    plan = builder.build(
        master_path=master,
        training_preregistration_path=intent_prereg,
        launch_plan_path=launch_plan,
        source_training_preregistration_path=source_prereg,
        source_completion_path=source_completion,
        original_root=original_root,
        output_root=tmp_path / "evaluation",
        manifest_output=tmp_path / "models.json",
        evaluation_preregistration_output=tmp_path / "evaluation.json",
        audit_output=tmp_path / "audit.json",
        prior_wide_audit_path=tmp_path / "wide-audit.json",
        status_root=tmp_path / "validation-status",
        device="cuda:1",
        parallel_envs=24,
    )

    assert plan["matrix"] == {
        "registry": "training_validation",
        "base_seed": 1_550_001_600,
        "last_seed": 1_550_001_654,
        "levels": 55,
        "models": 4,
        "expected_attempts": 220,
        "max_ticks": 30_000,
        "paired_models_share_identical_task_seeds": True,
    }
    assert plan["target"]["checkpoint_epochs"] == [10, 20, 30]
    assert plan["source_baseline"]["sha256"] == _sha256(source_model)
    assert plan["launch_controller"]["status"] == str(
        (launch_status_root / "watcher_status.json").resolve()
    )
    assert plan["authority_boundary"]["formal_seed_consumption"] is False
