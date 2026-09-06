"""Freeze post-training validation for the isolated full55 polar policy."""

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


SCRIPT_PATH = Path(__file__).resolve()
MATERIALIZER = SCRIPT_PATH.with_name(
    "materialize_alphazuma_55_polar_validation.py"
)
WATCHER = SCRIPT_PATH.with_name("watch_alphazuma_55_polar_validation.py")
CONTRACT_BUILDER = SCRIPT_PATH.with_name(
    "build_alphazuma_55_training_validation.py"
)
EVALUATOR = SCRIPT_PATH.with_name("evaluate_zero_shot_multilevel_v2.py")
AUDITOR = SCRIPT_PATH.with_name("audit_alphazuma_55_training_validation.py")


def _artifact(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "sha256": _sha256(path)}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def build(
    *,
    master_path: Path,
    training_preregistration_path: Path,
    baseline_model_path: Path,
    baseline_migration_path: Path,
    original_root: Path,
    output_root: Path,
    manifest_output: Path,
    evaluation_preregistration_output: Path,
    audit_output: Path,
    reset_audit_path: Path,
    status_root: Path,
    device: str,
    parallel_envs: int,
) -> dict[str, Any]:
    master = _read_json(master_path)
    _require(
        master.get("schema")
        == "zuma-rl.alphazuma-55-weekend-master-preregistration",
        "unexpected master preregistration",
    )
    training = _read_json(training_preregistration_path)
    _require(
        training.get("schema")
        == "zuma-rl.alphazuma-55-polar-distillation-preregistration"
        and training.get("status") == "FROZEN_BEFORE_TRAINING",
        "unexpected polar training preregistration",
    )
    _require(
        training["master_preregistration"]["sha256"] == _sha256(master_path),
        "polar training binds another master",
    )
    frozen = training["frozen_engineering_validation"]
    _require(
        frozen["registry"] == "training_validation"
        and int(frozen["last_seed"]) == int(frozen["base_seed"]) + 54
        and tuple(int(value) for value in frozen["checkpoint_epochs"])
        == (10, 30, 60, 90),
        "polar frozen validation changed",
    )
    _require(
        training["authority_boundary"]["current_campaign_candidate_authority"]
        is False,
        "polar training claims current-campaign authority",
    )
    run_dir = Path(str(training["run"]["run_dir"])).resolve()
    _require(
        not (run_dir / "completion.json").exists(),
        "validation plan must be frozen before target completion",
    )
    baseline_model_path = baseline_model_path.resolve(strict=True)
    baseline_migration_path = baseline_migration_path.resolve(strict=True)
    migration = _read_json(baseline_migration_path)
    migrated = migration.get("migrated_model", {})
    baseline_hash = _sha256(baseline_model_path)
    _require(
        migration.get("status") == "PASS"
        and Path(str(migrated.get("path", ""))).resolve() == baseline_model_path
        and migrated.get("sha256") == baseline_hash
        and tuple(migrated.get("observation_shape", ())) == (22_833,)
        and tuple(migrated.get("action_nvec", ())) == (4, 180),
        "baseline migration does not bind a full55-v1 model",
    )
    for path in (
        output_root,
        manifest_output,
        evaluation_preregistration_output,
        audit_output,
        status_root,
    ):
        _require(not path.exists(), f"polar validation output exists: {path}")
    _require(device in {"cuda:0", "cuda:1"}, "invalid validation device")
    _require(parallel_envs > 0, "parallel_envs must be positive")
    return {
        "schema": "zuma-rl.alphazuma-55-polar-validation-plan",
        "version": 1,
        "status": "FROZEN_DURING_TARGET_TRAINING",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "master_preregistration": _artifact(master_path),
        "training_preregistration": _artifact(training_preregistration_path),
        "target": {
            "run_dir": str(run_dir),
            "expected_completion": str(run_dir / "completion.json"),
            "expected_failure": str(run_dir / "failure.json"),
            "checkpoint_epochs": [10, 30, 60, 90],
            "checkpoint_paths": [
                str(run_dir / f"epoch_{epoch:02d}_model.zip")
                for epoch in (10, 30, 60, 90)
            ],
        },
        "baseline": {
            "id": "v11-full55-v1",
            "training_steps": int(migrated["num_timesteps"]),
            "path": str(baseline_model_path),
            "sha256": baseline_hash,
            "migration": _artifact(baseline_migration_path),
        },
        "matrix": {
            "registry": frozen["registry"],
            "base_seed": int(frozen["base_seed"]),
            "last_seed": int(frozen["last_seed"]),
            "levels": 55,
            "models": 5,
            "expected_attempts": 275,
            "max_ticks": 30_000,
            "paired_models_share_identical_task_seeds": True,
        },
        "execution": {
            "device": device,
            "parallel_envs": parallel_envs,
            "shard_count": 1,
        },
        "original_root": str(original_root.resolve(strict=True)),
        "wait_for_engineering_slot": {
            "reset_aim_audit": str(reset_audit_path.resolve()),
            "required_status": "PASS",
            "reason": "avoid concurrent 24-env engineering matrices",
        },
        "outputs": {
            "run_root": str(output_root.resolve()),
            "models_manifest": str(manifest_output.resolve()),
            "evaluation_preregistration": str(
                evaluation_preregistration_output.resolve()
            ),
            "audit_receipt": str(audit_output.resolve()),
            "status_root": str(status_root.resolve()),
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
            "negative_results_retained": True,
            "current_campaign_candidate_promotion": False,
        },
        "authority_boundary": {
            "classification": "engineering_validation_only",
            "current_campaign_candidate_authority": False,
            "formal_seed_consumption": False,
            "training_recipe_change_authority": False,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-preregistration", required=True, type=Path)
    parser.add_argument("--training-preregistration", required=True, type=Path)
    parser.add_argument("--baseline-model", required=True, type=Path)
    parser.add_argument("--baseline-migration", required=True, type=Path)
    parser.add_argument("--original-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--models-manifest", required=True, type=Path)
    parser.add_argument("--evaluation-preregistration", required=True, type=Path)
    parser.add_argument("--audit-output", required=True, type=Path)
    parser.add_argument("--reset-audit", required=True, type=Path)
    parser.add_argument("--status-root", required=True, type=Path)
    parser.add_argument("--device", choices=("cuda:0", "cuda:1"), default="cuda:0")
    parser.add_argument("--parallel-envs", type=int, default=24)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite validation plan: {output}")
    value = build(
        master_path=args.master_preregistration.expanduser().resolve(strict=True),
        training_preregistration_path=args.training_preregistration.expanduser().resolve(
            strict=True
        ),
        baseline_model_path=args.baseline_model.expanduser(),
        baseline_migration_path=args.baseline_migration.expanduser(),
        original_root=args.original_root.expanduser(),
        output_root=args.output_root.expanduser(),
        manifest_output=args.models_manifest.expanduser(),
        evaluation_preregistration_output=args.evaluation_preregistration.expanduser(),
        audit_output=args.audit_output.expanduser(),
        reset_audit_path=args.reset_audit.expanduser(),
        status_root=args.status_root.expanduser(),
        device=args.device,
        parallel_envs=args.parallel_envs,
    )
    _write_json_exclusive(output, value)
    print(
        json.dumps(
            {
                "status": value["status"],
                "output": str(output),
                "sha256": _sha256(output),
                "matrix": value["matrix"],
                "formal_seed_consumption": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
