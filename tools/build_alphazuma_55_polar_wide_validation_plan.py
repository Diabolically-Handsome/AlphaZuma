"""Freeze paired full55 validation for the isolated wide polar policy."""

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
    "materialize_alphazuma_55_polar_wide_validation.py"
)
WATCHER = SCRIPT_PATH.with_name(
    "watch_alphazuma_55_polar_wide_validation.py"
)
CONTRACT_BUILDER = SCRIPT_PATH.with_name(
    "build_alphazuma_55_training_validation.py"
)
EVALUATOR = SCRIPT_PATH.with_name("evaluate_zero_shot_multilevel_v2.py")
AUDITOR = SCRIPT_PATH.with_name("audit_alphazuma_55_training_validation.py")
EXPECTED_EPOCHS = (10, 30, 60)


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
    launch_plan_path: Path,
    source_training_preregistration_path: Path,
    source_completion_path: Path,
    original_root: Path,
    output_root: Path,
    manifest_output: Path,
    evaluation_preregistration_output: Path,
    audit_output: Path,
    prior_dagger_audit_path: Path,
    status_root: Path,
    device: str,
    parallel_envs: int,
) -> dict[str, Any]:
    master_path = master_path.resolve(strict=True)
    training_preregistration_path = training_preregistration_path.resolve(
        strict=True
    )
    launch_plan_path = launch_plan_path.resolve(strict=True)
    source_training_preregistration_path = (
        source_training_preregistration_path.resolve(strict=True)
    )
    source_completion_path = source_completion_path.resolve(strict=True)
    master = _read_json(master_path)
    _require(
        master.get("schema")
        == "zuma-rl.alphazuma-55-weekend-master-preregistration",
        "unexpected master preregistration",
    )
    training = _read_json(training_preregistration_path)
    _require(
        training.get("schema")
        == "zuma-rl.alphazuma-55-polar-wide-distillation-preregistration"
        and training.get("status") == "FROZEN_BEFORE_TRAINING",
        "unexpected wide polar training preregistration",
    )
    _require(
        training["master_preregistration"]["sha256"] == _sha256(master_path),
        "wide training binds another master",
    )
    frozen = training["frozen_engineering_validation"]
    _require(
        frozen["registry"] == "training_validation"
        and int(frozen["last_seed"]) == int(frozen["base_seed"]) + 54
        and tuple(int(value) for value in frozen["checkpoint_epochs"])
        == EXPECTED_EPOCHS
        and frozen["comparison_model_id"] == "polar-source-final",
        "wide frozen validation changed",
    )
    _require(
        training["authority_boundary"]["current_campaign_candidate_authority"]
        is False
        and training["authority_boundary"][
            "s99081535_successor_candidate_authority"
        ]
        is False,
        "wide training claims frozen-campaign authority",
    )
    run_dir = Path(str(training["run"]["run_dir"])).resolve()
    _require(
        not (run_dir / "completion.json").exists(),
        "validation plan must be frozen before wide completion",
    )

    launch = _read_json(launch_plan_path)
    _require(
        launch.get("schema") == "zuma-rl.alphazuma-55-polar-wide-launch-plan"
        and launch.get("status") == "FROZEN_BEFORE_WAIT",
        "unexpected wide launch plan",
    )
    _require(
        launch["wide_preregistration"]["sha256"]
        == _sha256(training_preregistration_path)
        and Path(str(launch["target"]["run_dir"])).resolve() == run_dir
        and launch["authority_boundary"]["formal_seed_consumption"] is False,
        "wide launch plan does not bind this training run",
    )

    source_training = _read_json(source_training_preregistration_path)
    source_completion = _read_json(source_completion_path)
    _require(
        source_training.get("schema")
        == "zuma-rl.alphazuma-55-polar-distillation-preregistration",
        "unexpected source polar preregistration",
    )
    _require(
        Path(str(source_training["run"]["run_dir"]) + "/completion.json").resolve()
        == source_completion_path,
        "source completion is outside the source run",
    )
    source_model = Path(str(source_completion["final_model"]["path"])).resolve(
        strict=True
    )
    source_digest = _sha256(source_model)
    _require(
        source_completion.get("schema")
        == "zuma-rl.alphazuma-55-polar-distillation-completion"
        and source_completion.get("status") == "COMPLETE"
        and source_completion.get("policy_architecture") == "entity_polar"
        and source_completion.get("formal_seed_consumption") is False
        and source_completion["final_model"]["sha256"] == source_digest,
        "source polar completion is invalid",
    )
    for path in (
        output_root,
        manifest_output,
        evaluation_preregistration_output,
        audit_output,
        status_root,
    ):
        _require(not path.exists(), f"wide validation output exists: {path}")
    _require(device in {"cuda:0", "cuda:1"}, "invalid validation device")
    _require(parallel_envs > 0, "parallel_envs must be positive")
    launch_status_root = Path(str(launch["outputs"]["status_root"])).resolve()
    return {
        "schema": "zuma-rl.alphazuma-55-polar-wide-validation-plan",
        "version": 1,
        "status": "FROZEN_DURING_TARGET_TRAINING",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "master_preregistration": _artifact(master_path),
        "training_preregistration": _artifact(training_preregistration_path),
        "launch_plan": _artifact(launch_plan_path),
        "launch_controller": {
            "status": str(launch_status_root / "watcher_status.json"),
            "required_terminal_status": "COMPLETE",
        },
        "target": {
            "run_dir": str(run_dir),
            "expected_completion": str(run_dir / "completion.json"),
            "expected_failure": str(run_dir / "failure.json"),
            "checkpoint_epochs": list(EXPECTED_EPOCHS),
            "checkpoint_paths": [
                str(run_dir / f"epoch_{epoch:02d}_model.zip")
                for epoch in EXPECTED_EPOCHS
            ],
        },
        "source_baseline": {
            "id": "polar-source-final",
            "training_steps": int(
                source_completion["final_model"]["num_timesteps"]
            ),
            "path": str(source_model),
            "sha256": source_digest,
            "training_preregistration": _artifact(
                source_training_preregistration_path
            ),
            "completion": _artifact(source_completion_path),
        },
        "matrix": {
            "registry": frozen["registry"],
            "base_seed": int(frozen["base_seed"]),
            "last_seed": int(frozen["last_seed"]),
            "levels": 55,
            "models": 4,
            "expected_attempts": 220,
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
            "prior_dagger_audit": str(prior_dagger_audit_path.resolve()),
            "required_status": "PASS",
            "reason": "serialize the 24-env engineering matrices",
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
            "s99081535_successor_candidate_promotion": False,
        },
        "authority_boundary": {
            "classification": "isolated_adaptive_engineering_validation_only",
            "current_campaign_candidate_authority": False,
            "s99081535_successor_candidate_authority": False,
            "formal_seed_consumption": False,
            "training_recipe_change_authority": False,
            "power_restore_authority": False,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-preregistration", required=True, type=Path)
    parser.add_argument("--training-preregistration", required=True, type=Path)
    parser.add_argument("--launch-plan", required=True, type=Path)
    parser.add_argument("--source-training-preregistration", required=True, type=Path)
    parser.add_argument("--source-completion", required=True, type=Path)
    parser.add_argument("--original-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--models-manifest", required=True, type=Path)
    parser.add_argument("--evaluation-preregistration", required=True, type=Path)
    parser.add_argument("--audit-output", required=True, type=Path)
    parser.add_argument("--prior-dagger-audit", required=True, type=Path)
    parser.add_argument("--status-root", required=True, type=Path)
    parser.add_argument("--device", choices=("cuda:0", "cuda:1"), default="cuda:1")
    parser.add_argument("--parallel-envs", type=int, default=24)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite wide validation plan: {output}")
    value = build(
        master_path=args.master_preregistration.expanduser(),
        training_preregistration_path=args.training_preregistration.expanduser(),
        launch_plan_path=args.launch_plan.expanduser(),
        source_training_preregistration_path=(
            args.source_training_preregistration.expanduser()
        ),
        source_completion_path=args.source_completion.expanduser(),
        original_root=args.original_root.expanduser(),
        output_root=args.output_root.expanduser(),
        manifest_output=args.models_manifest.expanduser(),
        evaluation_preregistration_output=(
            args.evaluation_preregistration.expanduser()
        ),
        audit_output=args.audit_output.expanduser(),
        prior_dagger_audit_path=args.prior_dagger_audit.expanduser(),
        status_root=args.status_root.expanduser(),
        device=args.device,
        parallel_envs=args.parallel_envs,
    )
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
