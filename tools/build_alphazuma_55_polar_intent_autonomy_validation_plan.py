"""Freeze paired full55 validation before autonomy preregistration exists."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools.build_alphazuma_55_eval_contract import (
    _read_json,
    _sha256,
    _write_json_exclusive,
)
from tools.materialize_alphazuma_55_polar_intent_autonomy import (
    validate_plan_static as validate_training_plan,
)


SCRIPT_PATH = Path(__file__).resolve()
MATERIALIZER = SCRIPT_PATH.with_name(
    "materialize_alphazuma_55_polar_intent_autonomy_validation.py"
)
WATCHER = SCRIPT_PATH.with_name(
    "watch_alphazuma_55_polar_intent_autonomy_validation.py"
)
CONTRACT_BUILDER = SCRIPT_PATH.with_name(
    "build_alphazuma_55_training_validation.py"
)
EVALUATOR = SCRIPT_PATH.with_name("evaluate_zero_shot_multilevel_v2.py")
AUDITOR = SCRIPT_PATH.with_name(
    "audit_alphazuma_55_training_validation.py"
)
EXPECTED_MODELS = (
    "source-final",
    "autonomy-round-00",
    "autonomy-round-01",
    "autonomy-round-02",
    "autonomy-round-03",
)


def _artifact(path: Path) -> dict[str, str]:
    path = path.resolve(strict=True)
    return {"path": str(path), "sha256": _sha256(path)}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def build(*, project_root: Path) -> dict[str, Any]:
    root = project_root.resolve(strict=True)
    master_path = (
        root
        / "diagnostics/alphazuma-55-weekend-s81081401-preregistration-v1.json"
    )
    training_plan_path = (
        root
        / "diagnostics/alphazuma-55-polar-intent-autonomy-"
        "s99081601-plan-v1.json"
    )
    training_plan_sha = _sha256(training_plan_path)
    training_plan = validate_training_plan(
        training_plan_path, training_plan_sha
    )
    master = _read_json(master_path)
    _require(
        master.get("schema")
        == "zuma-rl.alphazuma-55-weekend-master-preregistration",
        "unexpected autonomy validation master",
    )
    frozen = training_plan["frozen_engineering_validation"]
    _require(
        frozen["registry"] == "training_validation"
        and int(frozen["last_seed"]) == int(frozen["base_seed"]) + 54
        and tuple(frozen["models"]) == EXPECTED_MODELS,
        "autonomy frozen validation changed",
    )
    target_run = Path(str(training_plan["outputs"]["run_dir"]))
    original_root = Path(str(training_plan["original_root"])).resolve(
        strict=True
    )
    output_root = Path(
        "/mnt/d/ZumaTraining/alphazuma-55-polar-intent-autonomy-"
        "validation-s99081602-v1"
    )
    manifest_output = (
        root
        / "diagnostics/alphazuma-55-polar-intent-autonomy-validation-"
        "s99081602-models-v1.json"
    )
    evaluation_output = (
        root
        / "diagnostics/alphazuma-55-polar-intent-autonomy-validation-"
        "s99081602-preregistration-v1.json"
    )
    audit_output = (
        root
        / "diagnostics/alphazuma-55-polar-intent-autonomy-validation-"
        "s99081602-audit-v1.json"
    )
    status_root = Path(
        "/mnt/d/ZumaTraining/alphazuma-55-polar-intent-autonomy-"
        "validation-s99081602-watch-v1"
    )
    for path in (
        output_root,
        manifest_output,
        evaluation_output,
        audit_output,
        status_root,
    ):
        _require(not path.exists(), f"validation output exists: {path}")
    return {
        "schema": (
            "zuma-rl.alphazuma-55-polar-intent-autonomy-validation-plan"
        ),
        "version": 1,
        "status": "FROZEN_BEFORE_TARGET_PREREGISTRATION",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "master_preregistration": _artifact(master_path),
        "training_plan": _artifact(training_plan_path),
        "training_preregistration": {
            "expected_path": str(
                Path(training_plan["outputs"]["preregistration"])
            ),
            "binding_rule": (
                "must validate and bind this exact training_plan sha256"
            ),
        },
        "training_launch_controller": {
            "status": str(
                Path(training_plan["outputs"]["status_root"])
                / "controller_status.json"
            ),
            "required_terminal_status": "COMPLETE",
        },
        "target": {
            "run_dir": str(target_run),
            "expected_completion": str(target_run / "completion.json"),
            "expected_failure": str(target_run / "failure.json"),
            "round_indices": [0, 1, 2, 3],
            "round_model_paths": [
                str(target_run / f"round_{index:02d}/final_model.zip")
                for index in (0, 1, 2, 3)
            ],
        },
        "matrix": {
            "registry": frozen["registry"],
            "base_seed": int(frozen["base_seed"]),
            "last_seed": int(frozen["last_seed"]),
            "levels": 55,
            "models": 5,
            "expected_attempts": 275,
            "max_ticks": 30000,
            "paired_models_share_identical_task_seeds": True,
        },
        "execution": {
            "device": "cuda:1",
            "parallel_envs": 24,
            "shard_count": 1,
        },
        "original_root": str(original_root),
        "wait_for_engineering_slot": {
            "prior_polar_audit": str(
                root
                / "diagnostics/alphazuma-55-polar-intent-dagger-"
                "validation-s99081550-audit-v1.json"
            ),
            "required_status": "PASS",
            "reason": "serialize all 24-env engineering matrices",
        },
        "outputs": {
            "run_root": str(output_root),
            "models_manifest": str(manifest_output),
            "evaluation_preregistration": str(evaluation_output),
            "audit_receipt": str(audit_output),
            "status_root": str(status_root),
        },
        "implementation": {
            "planner": _artifact(SCRIPT_PATH),
            "materializer": _artifact(MATERIALIZER),
            "watcher": _artifact(WATCHER),
            "contract_builder": _artifact(CONTRACT_BUILDER),
            "evaluator": _artifact(EVALUATOR),
            "independent_auditor": _artifact(AUDITOR),
        },
        "decision_rule": {
            "ranking": list(frozen["ranking"]),
            "minimum_wins": 35,
            "minimum_cleared_levels": 35,
            "negative_results_retained": True,
            "current_campaign_candidate_promotion": False,
        },
        "authority_boundary": {
            "classification": "engineering_validation_only",
            "current_campaign_candidate_authority": False,
            "predecessor_successor_candidate_authority": False,
            "formal_seed_consumption": False,
            "power_restore_authority": False,
            "training_recipe_change_authority": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite validation plan: {output}")
    value = build(project_root=args.project_root.expanduser())
    _write_json_exclusive(output, value)
    print(
        json.dumps(
            {
                "status": value["status"],
                "output": {"path": str(output), "sha256": _sha256(output)},
                "matrix": value["matrix"],
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
