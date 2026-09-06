"""Bind the frozen DAgger source, round-one model, and final model for fresh validation."""

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
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def _bound_path(reference: Any, name: str) -> Path:
    _require(isinstance(reference, dict), f"{name} must be an object")
    path = Path(str(reference.get("path", ""))).resolve(strict=True)
    _require(reference.get("sha256") == contracts._sha256(path), f"{name} hash mismatch")
    return path


def validate_plan_static(plan_path: Path, expected_sha256: str) -> dict[str, Any]:
    _require(contracts._sha256(plan_path) == expected_sha256, "DAgger validation plan hash mismatch")
    plan = _read(plan_path)
    _require(
        plan.get("schema") == "zuma-rl.alphazuma-55-dagger-validation-plan"
        and plan.get("version") == 1
        and plan.get("status") == "FROZEN_BEFORE_DAGGER_TRAINING",
        "unexpected DAgger validation plan",
    )
    _bound_path(plan["master_preregistration"], "master preregistration")
    dagger_prereg_path = _bound_path(plan["dagger_preregistration"], "DAgger preregistration")
    dagger_prereg = _read(dagger_prereg_path)
    _require(
        dagger_prereg.get("schema") == "zuma-rl.alphazuma-55-dagger-preregistration"
        and dagger_prereg.get("status") == "FROZEN_BEFORE_TRAINING"
        and int(dagger_prereg.get("run", {}).get("rounds", -1)) == 2,
        "plan does not bind a frozen two-round DAgger run",
    )
    _bound_path(plan["models"]["source"], "source model")
    for name, reference in plan["implementation"].items():
        path = _bound_path(reference, f"implementation {name}")
        if name == "materializer":
            _require(path == SCRIPT_PATH, "plan binds another materializer")
    matrix = plan["matrix"]
    _require(
        int(matrix["levels"]) == 55
        and int(matrix["last_seed"]) == int(matrix["base_seed"]) + 54
        and int(matrix["max_ticks"]) == 30_000
        and matrix["paired_models_share_identical_task_seeds"] is True,
        "invalid DAgger validation matrix",
    )
    _require(plan["authority_boundary"]["formal_seed_consumption"] is False, "plan claims formal seed authority")
    return plan


def materialize(plan_path: Path, expected_sha256: str) -> tuple[Path, Path]:
    plan = validate_plan_static(plan_path, expected_sha256)
    completion_path = Path(str(plan["models"]["target"]["expected_completion_path"])).resolve(strict=True)
    completion = _read(completion_path)
    _require(
        completion.get("schema") == "zuma-rl.alphazuma-55-dagger-completion"
        and completion.get("status") == "COMPLETE"
        and completion.get("formal_seed_consumption") is False,
        "DAgger completion is not clean",
    )
    dagger_audit_path = Path(str(plan["outputs"]["dagger_training_audit"])).resolve(strict=True)
    dagger_audit = _read(dagger_audit_path)
    _require(
        dagger_audit.get("schema") == "zuma-rl.alphazuma-55-dagger-independent-audit"
        and dagger_audit.get("status") == "PASS"
        and dagger_audit.get("formal_selection_authority") is False,
        "DAgger training lacks a passing independent audit",
    )
    dagger_prereg_path = Path(str(plan["dagger_preregistration"]["path"])).resolve(strict=True)
    _require(
        dagger_audit["artifacts"]["preregistration"]["sha256"]
        == contracts._sha256(dagger_prereg_path)
        and dagger_audit["artifacts"]["completion"]["sha256"]
        == contracts._sha256(completion_path),
        "DAgger audit does not bind the frozen training artifacts",
    )
    collections = completion.get("collections")
    optimizations = completion.get("optimizations")
    _require(
        isinstance(collections, list)
        and len(collections) == 2
        and isinstance(optimizations, list)
        and len(optimizations) == 2,
        "DAgger completion does not contain exactly two rounds",
    )
    _require(
        sum(int(row["capacity_overflows"]) for row in collections) == 0,
        "DAgger collection reported observation overflow",
    )

    source = plan["models"]["source"]
    source_path = Path(str(source["path"])).resolve(strict=True)
    models = [
        {
            "id": str(source["id"]),
            "training_steps": int(source["training_steps"]),
            "path": str(source_path),
            "sha256": str(source["sha256"]),
        }
    ]
    round_one = optimizations[0]["round_model"]
    round_one_path = Path(str(round_one["path"])).resolve(strict=True)
    expected_round_one = Path(str(plan["models"]["target"]["expected_round_one"]["expected_path"])).resolve()
    _require(round_one_path == expected_round_one, "round-one model path differs from plan")
    round_one_hash = contracts._sha256(round_one_path)
    _require(round_one["sha256"] == round_one_hash, "round-one model hash mismatch")
    models.append(
        {
            "id": str(plan["models"]["target"]["expected_round_one"]["id"]),
            "training_steps": int(round_one["num_timesteps"]),
            "path": str(round_one_path),
            "sha256": round_one_hash,
        }
    )
    final = completion["final_model"]
    final_path = Path(str(final["path"])).resolve(strict=True)
    expected_final = Path(str(plan["models"]["target"]["expected_final"]["expected_path"])).resolve()
    _require(final_path == expected_final, "final model path differs from plan")
    final_hash = contracts._sha256(final_path)
    _require(final["sha256"] == final_hash, "final model hash mismatch")
    models.append(
        {
            "id": str(plan["models"]["target"]["expected_final"]["id"]),
            "training_steps": int(final["num_timesteps"]),
            "path": str(final_path),
            "sha256": final_hash,
        }
    )

    outputs = plan["outputs"]
    manifest_path = Path(str(outputs["models_manifest"])).resolve()
    prereg_path = Path(str(outputs["preregistration"])).resolve()
    run_root = Path(str(outputs["run_root"])).resolve()
    for path in (manifest_path, prereg_path, run_root):
        _require(not path.exists(), f"validation output already exists: {path}")
    master_path = Path(str(plan["master_preregistration"]["path"])).resolve(strict=True)
    evaluator_path = Path(str(plan["implementation"]["evaluator"]["path"])).resolve(strict=True)
    manifest, prereg = contracts.build_training_validation_contracts(
        master_path=master_path,
        original_root=Path(str(plan["original_root"])).resolve(strict=True),
        evaluator_path=evaluator_path,
        output_root=run_root,
        models=models,
        base_seed=int(plan["matrix"]["base_seed"]),
        device=str(plan["execution"]["device"]),
        parallel_envs=int(plan["execution"]["parallel_envs"]),
    )
    prereg["environment"]["base_config"]["max_ticks"] = int(plan["matrix"]["max_ticks"])
    prereg["diagnostic_contract"]["max_ticks"] = int(plan["matrix"]["max_ticks"])
    plan_ref = {"path": str(plan_path), "sha256": expected_sha256}
    completion_ref = {"path": str(completion_path), "sha256": contracts._sha256(completion_path)}
    dagger_audit_ref = {
        "path": str(dagger_audit_path),
        "sha256": contracts._sha256(dagger_audit_path),
    }
    manifest["dagger_validation_plan"] = plan_ref
    manifest["target_completion"] = completion_ref
    manifest["dagger_training_audit"] = dagger_audit_ref
    prereg["dagger_validation_plan"] = plan_ref
    prereg["target_completion"] = completion_ref
    prereg["dagger_training_audit"] = dagger_audit_ref
    _require(
        prereg["seed_plan"]["base_seed"] == plan["matrix"]["base_seed"]
        and prereg["seed_plan"]["last_seed"] == plan["matrix"]["last_seed"],
        "materialized seed matrix differs from plan",
    )
    _write_json_exclusive(manifest_path, manifest)
    _write_json_exclusive(prereg_path, prereg)
    return manifest_path, prereg_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--expected-plan-sha256", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan_path = args.plan.expanduser().resolve(strict=True)
    manifest_path, prereg_path = materialize(plan_path, str(args.expected_plan_sha256))
    print(json.dumps({
        "status": "FROZEN_AFTER_DAGGER_TRAINING",
        "manifest": {"path": str(manifest_path), "sha256": contracts._sha256(manifest_path)},
        "preregistration": {"path": str(prereg_path), "sha256": contracts._sha256(prereg_path)},
        "models": 3,
        "formal_seed_consumption": False,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
