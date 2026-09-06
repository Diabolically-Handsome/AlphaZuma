"""Bind a completed balanced-distillation model into its frozen validation plan."""

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
    _require(contracts._sha256(plan_path) == expected_sha256, "validation plan hash mismatch")
    plan = _read(plan_path)
    _require(
        plan.get("schema")
        == "zuma-rl.alphazuma-55-balanced-distillation-validation-plan"
        and plan.get("version") == 1
        and plan.get("status") == "FROZEN_BEFORE_TARGET_MODEL",
        "unexpected balanced validation plan",
    )
    _bound_path(plan["master_preregistration"], "master preregistration")
    _bound_path(plan["balanced_distillation_preregistration"], "distillation preregistration")
    _bound_path(plan["models"]["baseline"], "baseline model")
    _bound_path(plan["models"]["baseline_completion"], "baseline completion")
    for name, reference in plan["implementation"].items():
        implementation_path = _bound_path(reference, f"implementation {name}")
        if name == "materializer":
            _require(implementation_path == SCRIPT_PATH, "plan binds another materializer")
    matrix = plan["matrix"]
    _require(int(matrix["levels"]) == 55, "balanced validation must contain 55 levels")
    _require(
        int(matrix["last_seed"]) == int(matrix["base_seed"]) + 54,
        "balanced validation seed range is not one paired 55-level matrix",
    )
    _require(matrix["paired_models_share_identical_task_seeds"] is True, "matrix is not paired")
    _require(plan["authority_boundary"]["formal_seed_consumption"] is False, "plan claims formal seed authority")
    return plan


def materialize(plan_path: Path, expected_sha256: str) -> tuple[Path, Path]:
    plan = validate_plan_static(plan_path, expected_sha256)
    target_completion_path = Path(
        str(plan["models"]["target"]["expected_completion_path"])
    ).resolve(strict=True)
    target_completion = _read(target_completion_path)
    _require(target_completion.get("status") == "COMPLETE", "target distillation is not complete")
    final = target_completion["final_model"]
    target_path = Path(str(final["path"])).resolve(strict=True)
    expected_target = Path(str(plan["models"]["target"]["expected_path"])).resolve()
    _require(target_path == expected_target, "target model path differs from plan")
    target_sha256 = contracts._sha256(target_path)
    _require(final["sha256"] == target_sha256, "target completion model hash mismatch")

    outputs = plan["outputs"]
    manifest_path = Path(str(outputs["models_manifest"])).resolve()
    preregistration_path = Path(str(outputs["preregistration"])).resolve()
    run_root = Path(str(outputs["run_root"])).resolve()
    for path in (manifest_path, preregistration_path, run_root):
        _require(not path.exists(), f"balanced validation output already exists: {path}")

    baseline = plan["models"]["baseline"]
    models = [
        {
            "id": str(baseline["id"]),
            "training_steps": int(baseline["training_steps"]),
            "path": str(Path(str(baseline["path"])).resolve(strict=True)),
            "sha256": str(baseline["sha256"]),
        },
        {
            "id": str(plan["models"]["target"]["id"]),
            "training_steps": int(final["num_timesteps"]),
            "path": str(target_path),
            "sha256": target_sha256,
        },
    ]
    master_path = Path(str(plan["master_preregistration"]["path"])).resolve(strict=True)
    evaluator_path = Path(str(plan["implementation"]["evaluator"]["path"])).resolve(strict=True)
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
    plan_reference = {"path": str(plan_path), "sha256": expected_sha256}
    target_reference = {
        "path": str(target_completion_path),
        "sha256": contracts._sha256(target_completion_path),
    }
    manifest["balanced_validation_plan"] = plan_reference
    manifest["target_completion"] = target_reference
    preregistration["balanced_validation_plan"] = plan_reference
    preregistration["target_completion"] = target_reference
    _require(
        preregistration["seed_plan"]["base_seed"] == plan["matrix"]["base_seed"]
        and preregistration["seed_plan"]["last_seed"] == plan["matrix"]["last_seed"],
        "materialized seed matrix differs from plan",
    )
    _require(preregistration["evaluator"]["sha256"] == plan["implementation"]["evaluator"]["sha256"], "materialized evaluator differs from plan")
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
                "formal_seed_consumption": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
