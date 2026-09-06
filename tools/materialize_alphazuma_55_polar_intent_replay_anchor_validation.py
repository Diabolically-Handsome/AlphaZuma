"""Bind replay-anchor candidates into paired full55 validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools import build_alphazuma_55_training_validation as contracts
from tools import distill_alphazuma_55_polar_intent_replay_anchor_v1 as trainer
from tools import (
    materialize_alphazuma_55_polar_intent_replay_anchor as training_materializer,
)
from tools.build_alphazuma_55_eval_contract import _write_json_exclusive


SCRIPT_PATH = Path(__file__).resolve()
EXPECTED_ROUNDS = (0, 1, 2)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _bound_path(reference: Any, name: str) -> Path:
    _require(isinstance(reference, dict), f"{name} must be an object")
    path = Path(str(reference.get("path", ""))).resolve(strict=True)
    _require(
        reference.get("sha256") == contracts._sha256(path),
        f"{name} hash mismatch",
    )
    return path


def validate_plan_static(plan_path: Path, expected_sha256: str) -> dict[str, Any]:
    plan_path = plan_path.resolve(strict=True)
    _require(contracts._sha256(plan_path) == expected_sha256, "plan hash mismatch")
    plan = _read(plan_path)
    _require(
        plan.get("schema")
        == "zuma-rl.alphazuma-55-polar-intent-replay-anchor-validation-plan"
        and plan.get("version") == 1
        and plan.get("status")
        == "FROZEN_BEFORE_TARGET_TRAINING_COMPLETION",
        "unexpected replay-anchor validation plan",
    )
    _bound_path(plan["master_preregistration"], "master preregistration")
    training_plan_path = _bound_path(plan["training_plan"], "training plan")
    training_materializer.validate_plan_static(
        training_plan_path, plan["training_plan"]["sha256"]
    )
    expected_training_prereg = Path(
        str(plan["training_preregistration"]["expected_path"])
    ).resolve()
    training_plan = _read(training_plan_path)
    _require(
        expected_training_prereg
        == Path(training_plan["outputs"]["preregistration"]).resolve(),
        "validation plan expects another training preregistration",
    )
    _bound_path(
        plan["comparison_unanchored_dagger"]["preregistration"],
        "comparison DAgger preregistration",
    )
    for name, reference in plan["implementation"].items():
        implementation = _bound_path(reference, f"implementation {name}")
        if name == "materializer":
            _require(
                implementation == SCRIPT_PATH,
                "plan binds another replay-anchor validation materializer",
            )
    matrix = plan["matrix"]
    _require(
        int(matrix["levels"]) == 55
        and int(matrix["models"]) == 5
        and int(matrix["expected_attempts"]) == 275
        and int(matrix["last_seed"]) == int(matrix["base_seed"]) + 54
        and int(matrix["max_ticks"]) == 30000
        and matrix["paired_models_share_identical_task_seeds"] is True,
        "invalid replay-anchor validation matrix",
    )
    boundary = plan["authority_boundary"]
    _require(
        boundary["formal_seed_consumption"] is False
        and boundary["current_campaign_candidate_authority"] is False
        and boundary["power_restore_authority"] is False,
        "replay-anchor validation claims formal authority",
    )
    return plan


def materialize(plan_path: Path, expected_sha256: str) -> tuple[Path, Path]:
    plan_path = plan_path.resolve(strict=True)
    plan = validate_plan_static(plan_path, expected_sha256)
    training_path = Path(
        str(plan["training_preregistration"]["expected_path"])
    ).resolve(strict=True)
    training = trainer._validate_preregistration(training_path)
    _require(
        training["materialization_plan"]["sha256"]
        == plan["training_plan"]["sha256"],
        "training preregistration binds another replay-anchor plan",
    )
    completion_path = Path(plan["target"]["expected_completion"]).resolve(
        strict=True
    )
    completion = _read(completion_path)
    _require(
        completion.get("schema")
        == "zuma-rl.alphazuma-55-polar-intent-replay-anchor-completion"
        and completion.get("status") == "COMPLETE"
        and completion.get("policy_architecture")
        == "entity_polar_intent_wide_replay_anchor"
        and completion.get("training_label_semantics")
        == "raw_teacher_intent"
        and completion.get("state_distribution_semantics")
        == "clean_teacher_replay_plus_dagger_student_states"
        and completion.get("online_policy_inference_semantics")
        == "exact_action_masks"
        and int(completion.get("learned_features_dim", 0)) == 1024
        and completion.get("formal_seed_consumption") is False
        and completion.get("formal_candidate_authority") is False,
        "replay-anchor completion is invalid",
    )

    source_completion_path = Path(
        training["source"]["completion_path"]
    ).resolve(strict=True)
    source_completion = _read(source_completion_path)
    source_model = Path(training["source"]["model_path"]).resolve(strict=True)
    source_digest = contracts._sha256(source_model)
    _require(
        source_digest == training["source"]["model_sha256"]
        and source_completion["final_model"]["sha256"] == source_digest,
        "intent-wide source model changed before validation",
    )
    comparison = plan["comparison_unanchored_dagger"]
    comparison_completion_path = Path(
        comparison["expected_completion"]
    ).resolve(strict=True)
    comparison_completion = _read(comparison_completion_path)
    _require(
        comparison_completion.get("schema")
        == "zuma-rl.alphazuma-55-polar-intent-dagger-completion"
        and comparison_completion.get("status") == "COMPLETE"
        and comparison_completion.get("formal_seed_consumption") is False
        and comparison_completion.get("formal_candidate_authority") is False,
        "unanchored DAgger comparison completion is invalid",
    )
    comparison_model = Path(
        comparison_completion["final_model"]["path"]
    ).resolve(strict=True)
    comparison_digest = contracts._sha256(comparison_model)
    _require(
        comparison_completion["final_model"]["sha256"]
        == comparison_digest,
        "unanchored DAgger comparison model changed",
    )
    models = [
        {
            "id": "polar-intent-replay-anchor-source-wide-final",
            "training_steps": int(
                source_completion["final_model"]["num_timesteps"]
            ),
            "path": str(source_model),
            "sha256": source_digest,
        },
        {
            "id": "polar-intent-replay-anchor-source-dagger-final",
            "training_steps": int(
                comparison_completion["final_model"]["num_timesteps"]
            ),
            "path": str(comparison_model),
            "sha256": comparison_digest,
        },
    ]
    round_by_index = {
        int(row["round_index"]): row for row in completion["student_rounds"]
    }
    expected_paths = tuple(
        Path(value).resolve() for value in plan["target"]["round_model_paths"]
    )
    cumulative_steps = int(source_completion["final_model"]["num_timesteps"])
    cumulative_steps += sum(
        int(episode["steps"])
        for collection in completion["anchor_collections"]
        for episode in collection["episodes"]
    )
    for round_index, expected_path in zip(
        EXPECTED_ROUNDS, expected_paths, strict=True
    ):
        _require(round_index in round_by_index, "completion misses a round")
        row = round_by_index[round_index]
        path = Path(str(row["model"]["path"])).resolve(strict=True)
        _require(path == expected_path, f"round {round_index} path differs")
        digest = contracts._sha256(path)
        _require(row["model"]["sha256"] == digest, "round model hash mismatch")
        cumulative_steps += sum(
            int(episode["steps"]) for episode in row["collection"]["episodes"]
        )
        models.append(
            {
                "id": f"polar-intent-replay-anchor-round-{round_index:02d}",
                "training_steps": cumulative_steps,
                "path": str(path),
                "sha256": digest,
            }
        )

    outputs = plan["outputs"]
    manifest_path = Path(outputs["models_manifest"]).resolve()
    preregistration_path = Path(outputs["evaluation_preregistration"]).resolve()
    run_root = Path(outputs["run_root"]).resolve()
    for path in (manifest_path, preregistration_path, run_root):
        _require(not path.exists(), f"validation output already exists: {path}")
    master_path = Path(plan["master_preregistration"]["path"]).resolve(
        strict=True
    )
    evaluator_path = Path(plan["implementation"]["evaluator"]["path"]).resolve(
        strict=True
    )
    manifest, preregistration = contracts.build_training_validation_contracts(
        master_path=master_path,
        original_root=Path(plan["original_root"]).resolve(strict=True),
        evaluator_path=evaluator_path,
        output_root=run_root,
        models=models,
        base_seed=int(plan["matrix"]["base_seed"]),
        device=str(plan["execution"]["device"]),
        parallel_envs=int(plan["execution"]["parallel_envs"]),
    )
    preregistration["environment"]["base_config"]["max_ticks"] = int(
        plan["matrix"]["max_ticks"]
    )
    preregistration["diagnostic_contract"]["max_ticks"] = int(
        plan["matrix"]["max_ticks"]
    )
    plan_reference = {"path": str(plan_path), "sha256": expected_sha256}
    completion_reference = {
        "path": str(completion_path),
        "sha256": contracts._sha256(completion_path),
    }
    comparison_reference = {
        "path": str(comparison_completion_path),
        "sha256": contracts._sha256(comparison_completion_path),
    }
    training_reference = {
        "path": str(training_path),
        "sha256": contracts._sha256(training_path),
    }
    for value in (manifest, preregistration):
        value["polar_intent_replay_anchor_validation_plan"] = plan_reference
        value["training_preregistration"] = training_reference
        value["target_completion"] = completion_reference
        value["comparison_unanchored_dagger_completion"] = (
            comparison_reference
        )
    _require(
        int(preregistration["seed_plan"]["base_seed"])
        == int(plan["matrix"]["base_seed"])
        and int(preregistration["seed_plan"]["last_seed"])
        == int(plan["matrix"]["last_seed"]),
        "materialized seed matrix differs from replay-anchor plan",
    )
    _write_json_exclusive(manifest_path, manifest)
    _write_json_exclusive(preregistration_path, preregistration)
    return manifest_path, preregistration_path
