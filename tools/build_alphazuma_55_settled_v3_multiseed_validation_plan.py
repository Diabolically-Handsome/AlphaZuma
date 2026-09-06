"""Freeze fresh validation before the multiseed target exists."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools import build_alphazuma_55_settled_v3_validation_plan as base


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
MATERIALIZER = SCRIPT_PATH.with_name(
    "materialize_alphazuma_55_settled_v3_multiseed_validation.py"
)
EVALUATOR = SCRIPT_PATH.with_name("evaluate_zero_shot_multilevel_v2.py")
CONTRACT_BUILDER = SCRIPT_PATH.with_name(
    "build_alphazuma_55_training_validation.py"
)
AUDITOR = SCRIPT_PATH.with_name("audit_alphazuma_55_training_validation.py")
PRIOR_VALIDATIONS = (
    PROJECT_ROOT
    / "diagnostics/alphazuma-55-distilled-first-checkpoint-validation-s92081501-preregistration-v1.json",
    PROJECT_ROOT
    / "diagnostics/alphazuma-55-distilled-milestone-validation-s92081502-plan-v1.json",
    PROJECT_ROOT
    / "diagnostics/alphazuma-55-balanced-distillation-validation-s99081511-plan-v1.json",
    PROJECT_ROOT
    / "diagnostics/alphazuma-55-settled-v3-validation-s99081518-plan-v1.json",
    PROJECT_ROOT
    / "diagnostics/alphazuma-55-settled-v3-high-fidelity-validation-s99081520-plan-v1.json",
    PROJECT_ROOT
    / "diagnostics/alphazuma-55-settled-v3-reset-aim-validation-s99081522-plan-v1.json",
    PROJECT_ROOT
    / "diagnostics/alphazuma-55-settled-v3-reset-aim-v2-validation-s99081524-plan-v1.json",
)
CHECKPOINT_EPOCHS = (4, 8, 12, 16)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-preregistration", required=True, type=Path)
    parser.add_argument("--distillation-preregistration", required=True, type=Path)
    parser.add_argument("--baseline-model", required=True, type=Path)
    parser.add_argument("--baseline-migration", required=True, type=Path)
    parser.add_argument("--target-run-dir", required=True, type=Path)
    parser.add_argument("--original-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--models-manifest", required=True, type=Path)
    parser.add_argument("--preregistration-output", required=True, type=Path)
    parser.add_argument("--audit-output", required=True, type=Path)
    parser.add_argument("--base-seed", required=True, type=int)
    parser.add_argument("--device", choices=("cuda:0", "cuda:1"), default="cuda:1")
    parser.add_argument("--parallel-envs", type=int, default=24)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    master_path = args.master_preregistration.expanduser().resolve(strict=True)
    distillation_path = args.distillation_preregistration.expanduser().resolve(
        strict=True
    )
    baseline_model = args.baseline_model.expanduser().resolve(strict=True)
    baseline_migration = args.baseline_migration.expanduser().resolve(strict=True)
    target_run = args.target_run_dir.expanduser().resolve()
    original_root = args.original_root.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve()
    models_manifest = args.models_manifest.expanduser().resolve()
    preregistration_output = args.preregistration_output.expanduser().resolve()
    audit_output = args.audit_output.expanduser().resolve()
    master = base._read(master_path)
    distillation = base._read(distillation_path)
    migration = base._read(baseline_migration)
    migrated = migration.get("migrated_model", {})
    if (
        master.get("schema")
        != "zuma-rl.alphazuma-55-weekend-master-preregistration"
        or master.get("version") != 1
    ):
        raise ValueError("unexpected master preregistration")
    if (
        distillation.get("schema")
        != "zuma-rl.alphazuma-55-settled-v3-multiseed-preregistration"
        or distillation.get("status") != "FROZEN_BEFORE_TRAINING"
        or int(distillation.get("run", {}).get("rounds", -1)) != 4
    ):
        raise ValueError("multiseed distillation is not frozen")
    if Path(str(distillation["run"]["run_dir"])).resolve() != target_run:
        raise ValueError("validation target differs from multiseed run")
    if (
        migration.get("status") != "PASS"
        or migrated.get("sha256") != base._sha256(baseline_model)
        or Path(str(migrated.get("path", ""))).resolve() != baseline_model
    ):
        raise ValueError("baseline migration does not bind the baseline")
    expected_paths = [
        target_run / f"epoch_{epoch:02d}_model.zip" for epoch in CHECKPOINT_EPOCHS
    ] + [target_run / "final_model.zip", target_run / "completion.json"]
    if target_run.exists() or any(path.exists() for path in expected_paths):
        raise FileExistsError("validation must freeze before multiseed training")
    if any(
        path.exists()
        for path in (output, output_root, models_manifest, preregistration_output, audit_output)
    ):
        raise FileExistsError("multiseed validation output already exists")
    if args.device not in {"cuda:0", "cuda:1"} or not 1 <= args.parallel_envs <= 55:
        raise ValueError("invalid multiseed validation execution")
    last_seed = args.base_seed + 54
    registry = master["seed_registry"]["training_validation"]
    if not int(registry["first"]) <= args.base_seed <= last_seed <= int(
        registry["last"]
    ):
        raise ValueError("multiseed validation escaped its seed registry")
    prior_ranges = [
        base._range_from_prior(base._read(path.resolve(strict=True)))
        for path in PRIOR_VALIDATIONS
    ]
    if any(base._overlap(args.base_seed, last_seed, value) for value in prior_ranges):
        raise ValueError("multiseed validation overlaps an earlier matrix")
    value = {
        "schema": "zuma-rl.alphazuma-55-settled-v3-multiseed-validation-plan",
        "version": 1,
        "status": "FROZEN_BEFORE_TARGET_MODEL",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_id": master["campaign_id"],
        "purpose": (
            "Paired fresh-seed checkpoint comparison of the four-round "
            "multiseed continuation against the unchanged V11 baseline."
        ),
        "master_preregistration": base._artifact(master_path),
        "distillation_preregistration": base._artifact(distillation_path),
        "original_root": str(original_root),
        "models": {
            "baseline": {
                "id": "v11-full55-v1",
                "training_steps": int(migrated["num_timesteps"]),
                **base._artifact(baseline_model),
            },
            "baseline_migration": base._artifact(baseline_migration),
            "target": {
                "run_dir": str(target_run),
                "expected_completion_path": str(target_run / "completion.json"),
                "expected_checkpoints": [
                    {
                        "id": f"settled-v3-multiseed-epoch-{epoch:02d}",
                        "epoch": epoch,
                        "expected_path": str(target_run / f"epoch_{epoch:02d}_model.zip"),
                    }
                    for epoch in CHECKPOINT_EPOCHS
                ],
                "expected_final": {
                    "id": "settled-v3-multiseed-final",
                    "expected_path": str(target_run / "final_model.zip"),
                },
                "all_paths_must_be_absent_at_freeze": True,
                "hashes_frozen_after_completion": True,
            },
        },
        "matrix": {
            "levels": 55,
            "attempts_per_level_per_model": 1,
            "paired_models_share_identical_task_seeds": True,
            "seed_registry": "training_validation",
            "base_seed": args.base_seed,
            "last_seed": last_seed,
            "max_ticks": 30_000,
            "prior_reserved_or_consumed_ranges": [
                list(value) for value in prior_ranges
            ],
        },
        "execution": {
            "device": args.device,
            "parallel_envs": args.parallel_envs,
            "shard_count": 1,
        },
        "implementation": {
            "planner": base._artifact(SCRIPT_PATH),
            "materializer": base._artifact(MATERIALIZER),
            "contract_builder": base._artifact(CONTRACT_BUILDER),
            "evaluator": base._artifact(EVALUATOR),
            "independent_auditor": base._artifact(AUDITOR),
        },
        "outputs": {
            "run_root": str(output_root),
            "models_manifest": str(models_manifest),
            "preregistration": str(preregistration_output),
            "audit_receipt": str(audit_output),
        },
        "decision_rule": {
            "ranking": [
                "cleared_levels_descending",
                "wins_descending",
                "median_winning_ticks_ascending",
                "earlier_epoch_then_final",
            ],
            "evidence_scope": "engineering_checkpoint_choice_only",
            "formal_candidate_registration_authorized": False,
            "formal_selection_seed_consumption": False,
            "negative_result_must_be_retained": True,
        },
        "authority_boundary": {
            "formal_seed_consumption": False,
            "formal_selection_authority": False,
            "training_recipe_change_authority": False,
        },
    }
    base._write_exclusive(output, value)
    print(
        json.dumps(
            {
                "status": value["status"],
                "output": str(output),
                "sha256": base._sha256(output),
                "seed_range": [args.base_seed, last_seed],
                "target_absent": True,
                "models_after_materialization": 6,
                "formal_seed_consumption": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
