"""Bind frozen wide-polar checkpoints into paired full55 validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools import build_alphazuma_55_training_validation as contracts
from tools.build_alphazuma_55_eval_contract import _write_json_exclusive


SCRIPT_PATH = Path(__file__).resolve()
EXPECTED_EPOCHS = (10, 30, 60)


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
        == "zuma-rl.alphazuma-55-polar-wide-validation-plan"
        and plan.get("version") == 1
        and plan.get("status") == "FROZEN_DURING_TARGET_TRAINING",
        "unexpected wide polar validation plan",
    )
    _bound_path(plan["master_preregistration"], "master preregistration")
    training_path = _bound_path(
        plan["training_preregistration"], "training preregistration"
    )
    training = _read(training_path)
    _require(
        training.get("schema")
        == "zuma-rl.alphazuma-55-polar-wide-distillation-preregistration"
        and tuple(
            int(value)
            for value in training["frozen_engineering_validation"][
                "checkpoint_epochs"
            ]
        )
        == EXPECTED_EPOCHS,
        "plan does not bind the frozen wide training contract",
    )
    _bound_path(plan["launch_plan"], "launch plan")
    _bound_path(plan["source_baseline"], "source baseline model")
    _bound_path(
        plan["source_baseline"]["training_preregistration"],
        "source training preregistration",
    )
    _bound_path(plan["source_baseline"]["completion"], "source completion")
    for name, reference in plan["implementation"].items():
        implementation = _bound_path(reference, f"implementation {name}")
        if name == "materializer":
            _require(implementation == SCRIPT_PATH, "plan binds another materializer")
    matrix = plan["matrix"]
    _require(
        int(matrix["levels"]) == 55
        and int(matrix["models"]) == 4
        and int(matrix["expected_attempts"]) == 220
        and int(matrix["last_seed"]) == int(matrix["base_seed"]) + 54
        and int(matrix["max_ticks"]) == 30_000
        and matrix["paired_models_share_identical_task_seeds"] is True,
        "invalid wide polar validation matrix",
    )
    boundary = plan["authority_boundary"]
    _require(
        boundary["formal_seed_consumption"] is False
        and boundary["current_campaign_candidate_authority"] is False
        and boundary["s99081535_successor_candidate_authority"] is False
        and boundary["power_restore_authority"] is False,
        "plan claims frozen-campaign authority",
    )
    return plan


def materialize(plan_path: Path, expected_sha256: str) -> tuple[Path, Path]:
    plan_path = plan_path.resolve(strict=True)
    plan = validate_plan_static(plan_path, expected_sha256)
    completion_path = Path(plan["target"]["expected_completion"]).resolve(
        strict=True
    )
    completion = _read(completion_path)
    _require(
        completion.get("schema")
        == "zuma-rl.alphazuma-55-polar-wide-distillation-completion"
        and completion.get("status") == "COMPLETE"
        and completion.get("policy_architecture") == "entity_polar_wide"
        and int(completion.get("learned_features_dim", 0)) == 1024
        and int(completion.get("policy_parameter_count", 0)) == 963_866
        and completion.get("formal_seed_consumption") is False
        and completion.get("formal_candidate_authority") is False,
        "wide polar completion is invalid",
    )
    checkpoint_by_epoch = {
        int(row["epoch"]): row for row in completion["optimization"]["checkpoints"]
    }
    _require(
        all(epoch in checkpoint_by_epoch for epoch in EXPECTED_EPOCHS),
        "wide completion misses a frozen checkpoint",
    )
    baseline = plan["source_baseline"]
    models = [
        {
            "id": baseline["id"],
            "training_steps": int(baseline["training_steps"]),
            "path": str(_bound_path(baseline, "source baseline model")),
            "sha256": baseline["sha256"],
        }
    ]
    expected_paths = tuple(
        Path(value).resolve() for value in plan["target"]["checkpoint_paths"]
    )
    for epoch, expected_path in zip(EXPECTED_EPOCHS, expected_paths, strict=True):
        row = checkpoint_by_epoch[epoch]
        path = Path(str(row["path"])).resolve(strict=True)
        _require(path == expected_path, f"wide epoch {epoch} path differs")
        digest = contracts._sha256(path)
        _require(row["sha256"] == digest, f"wide epoch {epoch} hash mismatch")
        models.append(
            {
                "id": f"polar-wide-epoch-{epoch:02d}",
                "training_steps": int(completion["final_model"]["num_timesteps"]),
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
    manifest["polar_wide_validation_plan"] = plan_reference
    manifest["target_completion"] = completion_reference
    preregistration["polar_wide_validation_plan"] = plan_reference
    preregistration["target_completion"] = completion_reference
    _require(
        int(preregistration["seed_plan"]["base_seed"])
        == int(plan["matrix"]["base_seed"])
        and int(preregistration["seed_plan"]["last_seed"])
        == int(plan["matrix"]["last_seed"]),
        "materialized seed matrix differs from plan",
    )
    _write_json_exclusive(manifest_path, manifest)
    _write_json_exclusive(preregistration_path, preregistration)
    return manifest_path, preregistration_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--expected-plan-sha256", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_path, preregistration_path = materialize(
        args.plan.expanduser().resolve(strict=True),
        str(args.expected_plan_sha256),
    )
    print(
        json.dumps(
            {
                "status": "FROZEN_AFTER_TARGET_MODEL",
                "manifest": {
                    "path": str(manifest_path),
                    "sha256": contracts._sha256(manifest_path),
                },
                "preregistration": {
                    "path": str(preregistration_path),
                    "sha256": contracts._sha256(preregistration_path),
                },
                "models": 4,
                "expected_attempts": 220,
                "formal_seed_consumption": False,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
