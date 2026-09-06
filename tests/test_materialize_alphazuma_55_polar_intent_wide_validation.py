from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import materialize_alphazuma_55_polar_intent_wide_validation as target
from tools.build_alphazuma_55_eval_contract import _sha256


def _write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _ref(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    master = tmp_path / "master.json"
    _write(master, {"schema": "master"})
    original_root = tmp_path / "original"
    original_root.mkdir()
    source_model = tmp_path / "source.zip"
    source_model.write_bytes(b"source")
    source_prereg = tmp_path / "source-prereg.json"
    source_completion = tmp_path / "source-completion.json"
    launch = tmp_path / "launch.json"
    for path in (source_prereg, source_completion, launch):
        _write(path, {"placeholder": True})

    run_dir = tmp_path / "intent-run"
    run_dir.mkdir()
    checkpoints: list[dict[str, object]] = []
    for epoch in (10, 20, 30):
        model = run_dir / f"epoch_{epoch:02d}_model.zip"
        model.write_bytes(f"model-{epoch}".encode())
        checkpoints.append(
            {"epoch": epoch, "path": str(model), "sha256": _sha256(model)}
        )
    completion = run_dir / "completion.json"
    _write(
        completion,
        {
            "schema": (
                "zuma-rl.alphazuma-55-polar-intent-wide-"
                "distillation-completion"
            ),
            "status": "COMPLETE",
            "policy_architecture": "entity_polar_intent_wide",
            "training_label_semantics": "raw_teacher_intent",
            "environment_execution_semantics": "exact_mask_effective_action",
            "online_policy_inference_semantics": "exact_action_masks",
            "learned_features_dim": 1024,
            "policy_parameter_count": 963_866,
            "formal_seed_consumption": False,
            "formal_candidate_authority": False,
            "optimization": {"checkpoints": checkpoints},
            "final_model": {"num_timesteps": 999},
        },
    )
    training = tmp_path / "training.json"
    _write(
        training,
        {
            "schema": (
                "zuma-rl.alphazuma-55-polar-intent-wide-"
                "distillation-preregistration"
            ),
            "frozen_engineering_validation": {
                "checkpoint_epochs": [10, 20, 30]
            },
        },
    )
    implementation: dict[str, dict[str, str]] = {}
    for name in ("planner", "watcher", "contract_builder", "evaluator", "independent_auditor"):
        path = tmp_path / f"{name}.py"
        path.write_text(f"# {name}\n", encoding="utf-8")
        implementation[name] = _ref(path)
    implementation["materializer"] = _ref(target.SCRIPT_PATH)
    plan = tmp_path / "plan.json"
    _write(
        plan,
        {
            "schema": "zuma-rl.alphazuma-55-polar-intent-wide-validation-plan",
            "version": 1,
            "status": "FROZEN_DURING_TARGET_TRAINING",
            "master_preregistration": _ref(master),
            "training_preregistration": _ref(training),
            "launch_plan": _ref(launch),
            "source_baseline": {
                "id": "polar-source-final",
                "training_steps": 123,
                **_ref(source_model),
                "training_preregistration": _ref(source_prereg),
                "completion": _ref(source_completion),
            },
            "target": {
                "expected_completion": str(completion),
                "checkpoint_paths": [
                    str(run_dir / f"epoch_{epoch:02d}_model.zip")
                    for epoch in (10, 20, 30)
                ],
            },
            "matrix": {
                "base_seed": 1_550_001_600,
                "last_seed": 1_550_001_654,
                "levels": 55,
                "models": 4,
                "expected_attempts": 220,
                "max_ticks": 30_000,
                "paired_models_share_identical_task_seeds": True,
            },
            "execution": {"device": "cuda:1", "parallel_envs": 24},
            "original_root": str(original_root),
            "outputs": {
                "models_manifest": str(tmp_path / "models.json"),
                "evaluation_preregistration": str(tmp_path / "evaluation.json"),
                "run_root": str(tmp_path / "evaluation-run"),
            },
            "implementation": implementation,
            "authority_boundary": {
                "formal_seed_consumption": False,
                "current_campaign_candidate_authority": False,
                "s99081535_successor_candidate_authority": False,
                "power_restore_authority": False,
            },
        },
    )
    return plan, completion


def test_materialize_binds_raw_intent_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, _ = _fixture(tmp_path)

    def fake_contracts(**kwargs: object) -> tuple[dict[str, object], dict[str, object]]:
        models = kwargs["models"]
        return (
            {"models": models},
            {
                "environment": {"base_config": {"max_ticks": 12_000}},
                "diagnostic_contract": {"max_ticks": 12_000},
                "seed_plan": {
                    "base_seed": 1_550_001_600,
                    "last_seed": 1_550_001_654,
                },
            },
        )

    monkeypatch.setattr(
        target.contracts, "build_training_validation_contracts", fake_contracts
    )
    manifest_path, preregistration_path = target.materialize(
        plan, _sha256(plan)
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    preregistration = json.loads(
        preregistration_path.read_text(encoding="utf-8")
    )

    assert [row["id"] for row in manifest["models"]] == [
        "polar-source-final",
        "polar-intent-wide-epoch-10",
        "polar-intent-wide-epoch-20",
        "polar-intent-wide-epoch-30",
    ]
    assert preregistration["environment"]["base_config"]["max_ticks"] == 30_000
    assert "polar_intent_wide_validation_plan" in preregistration


def test_materialize_rejects_effective_action_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, completion = _fixture(tmp_path)
    value = json.loads(completion.read_text(encoding="utf-8"))
    value["training_label_semantics"] = "effective_executed_action"
    _write(completion, value)

    with pytest.raises(ValueError, match="intent-wide completion is invalid"):
        target.materialize(plan, _sha256(plan))
