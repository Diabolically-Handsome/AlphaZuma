"""Bind reset-aim checkpoints into their frozen fresh validation."""

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
EXPECTED_EPOCHS = (4, 8, 12, 16, 20, 24)


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
    _require(reference.get("sha256") == contracts._sha256(path), f"{name} hash mismatch")
    return path


def validate_plan_static(plan_path: Path, expected_sha256: str) -> dict[str, Any]:
    _require(
        contracts._sha256(plan_path) == expected_sha256,
        "reset-aim validation plan hash mismatch",
    )
    plan = _read(plan_path)
    _require(
        plan.get("schema")
        == "zuma-rl.alphazuma-55-settled-v3-reset-aim-validation-plan"
        and plan.get("version") == 1
        and plan.get("status") == "FROZEN_BEFORE_TARGET_MODEL",
        "unexpected reset-aim validation plan",
    )
    _bound_path(plan["master_preregistration"], "master preregistration")
    distillation_path = _bound_path(
        plan["distillation_preregistration"], "distillation preregistration"
    )
    distillation = _read(distillation_path)
    _require(
        distillation.get("run", {}).get("aim_head_reset", {}).get("enabled") is True,
        "plan does not bind a reset-aim distillation",
    )
    _bound_path(plan["models"]["baseline"], "baseline model")
    _bound_path(plan["models"]["baseline_migration"], "baseline migration")
    for name, reference in plan["implementation"].items():
        implementation = _bound_path(reference, f"implementation {name}")
        if name == "materializer":
            _require(implementation == SCRIPT_PATH, "plan binds another materializer")
    matrix = plan["matrix"]
    _require(
        int(matrix["levels"]) == 55
        and int(matrix["last_seed"]) == int(matrix["base_seed"]) + 54
        and int(matrix["max_ticks"]) == 30_000
        and matrix["paired_models_share_identical_task_seeds"] is True,
        "invalid reset-aim validation matrix",
    )
    _require(
        tuple(int(row["epoch"]) for row in plan["models"]["target"]["expected_checkpoints"])
        == EXPECTED_EPOCHS,
        "reset-aim checkpoint schedule differs",
    )
    _require(
        plan["authority_boundary"]["formal_seed_consumption"] is False,
        "plan claims formal seed authority",
    )
    return plan


def materialize(plan_path: Path, expected_sha256: str) -> tuple[Path, Path]:
    plan = validate_plan_static(plan_path, expected_sha256)
    target = plan["models"]["target"]
    completion_path = Path(str(target["expected_completion_path"])).resolve(strict=True)
    completion = _read(completion_path)
    reset = completion.get("aim_head_reset", {})
    _require(
        completion.get("schema")
        == "zuma-rl.alphazuma-55-settled-v3-distillation-completion"
        and completion.get("status") == "COMPLETE"
        and reset.get("status")
        == "APPLIED_BEFORE_OPTIMIZER_REPLACEMENT_AND_COLLECTION"
        and float(reset.get("aim_weight_l2_after", -1.0)) == 0.0
        and float(reset.get("aim_bias_l2_after", -1.0)) == 0.0
        and reset.get("verb_head_unchanged") is True,
        "reset-aim distillation is not complete or reset receipt is invalid",
    )
    checkpoint_by_epoch = {
        int(row["epoch"]): row for row in completion["optimization"]["checkpoints"]
    }
    _require(
        tuple(sorted(checkpoint_by_epoch)) == EXPECTED_EPOCHS,
        "completion checkpoint set differs from plan",
    )
    baseline = plan["models"]["baseline"]
    models = [
        {
            "id": str(baseline["id"]),
            "training_steps": int(baseline["training_steps"]),
            "path": str(Path(str(baseline["path"])).resolve(strict=True)),
            "sha256": str(baseline["sha256"]),
        }
    ]
    for expected in target["expected_checkpoints"]:
        epoch = int(expected["epoch"])
        row = checkpoint_by_epoch[epoch]
        path = Path(str(row["path"])).resolve(strict=True)
        _require(
            path == Path(str(expected["expected_path"])).resolve(),
            f"epoch {epoch} path differs from plan",
        )
        digest = contracts._sha256(path)
        _require(row["sha256"] == digest, f"epoch {epoch} hash mismatch")
        models.append(
            {
                "id": str(expected["id"]),
                "training_steps": int(completion["final_model"]["num_timesteps"]),
                "path": str(path),
                "sha256": digest,
            }
        )
    final = completion["final_model"]
    final_path = Path(str(final["path"])).resolve(strict=True)
    _require(
        final_path == Path(str(target["expected_final"]["expected_path"])).resolve(),
        "final model path differs from plan",
    )
    final_hash = contracts._sha256(final_path)
    _require(final["sha256"] == final_hash, "final model hash mismatch")
    models.append(
        {
            "id": str(target["expected_final"]["id"]),
            "training_steps": int(final["num_timesteps"]),
            "path": str(final_path),
            "sha256": final_hash,
        }
    )

    outputs = plan["outputs"]
    manifest_path = Path(str(outputs["models_manifest"])).resolve()
    preregistration_path = Path(str(outputs["preregistration"])).resolve()
    run_root = Path(str(outputs["run_root"])).resolve()
    for path in (manifest_path, preregistration_path, run_root):
        _require(not path.exists(), f"validation output already exists: {path}")
    master_path = Path(str(plan["master_preregistration"]["path"])).resolve(strict=True)
    evaluator_path = Path(str(plan["implementation"]["evaluator"]["path"])).resolve(
        strict=True
    )
    manifest, preregistration = contracts.build_training_validation_contracts(
        master_path=master_path,
        original_root=Path(str(plan["original_root"])).resolve(strict=True),
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
    manifest["settled_v3_reset_aim_validation_plan"] = plan_reference
    manifest["target_completion"] = completion_reference
    preregistration["settled_v3_reset_aim_validation_plan"] = plan_reference
    preregistration["target_completion"] = completion_reference
    _require(
        preregistration["seed_plan"]["base_seed"] == plan["matrix"]["base_seed"]
        and preregistration["seed_plan"]["last_seed"] == plan["matrix"]["last_seed"],
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
    plan_path = args.plan.expanduser().resolve(strict=True)
    manifest_path, preregistration_path = materialize(
        plan_path, str(args.expected_plan_sha256)
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
                "models": 8,
                "max_ticks": 30_000,
                "formal_seed_consumption": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
