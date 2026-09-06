"""Freeze a fresh source/round-one/final DAgger validation before training starts."""

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
MATERIALIZER = SCRIPT_PATH.with_name("materialize_alphazuma_55_dagger_validation.py")
EVALUATOR = SCRIPT_PATH.with_name("evaluate_zero_shot_multilevel_v2.py")
CONTRACT_BUILDER = SCRIPT_PATH.with_name("build_alphazuma_55_training_validation.py")
AUDITOR = SCRIPT_PATH.with_name("audit_alphazuma_55_training_validation.py")
DAGGER_AUDITOR = SCRIPT_PATH.with_name("audit_alphazuma_55_dagger.py")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-preregistration", required=True, type=Path)
    parser.add_argument("--dagger-preregistration", required=True, type=Path)
    parser.add_argument("--original-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--models-manifest", required=True, type=Path)
    parser.add_argument("--preregistration-output", required=True, type=Path)
    parser.add_argument("--audit-output", required=True, type=Path)
    parser.add_argument("--dagger-audit-output", required=True, type=Path)
    parser.add_argument("--base-seed", required=True, type=int)
    parser.add_argument("--device", choices=("cuda:0", "cuda:1"), default="cuda:1")
    parser.add_argument("--parallel-envs", type=int, default=24)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    master_path = args.master_preregistration.expanduser().resolve(strict=True)
    dagger_path = args.dagger_preregistration.expanduser().resolve(strict=True)
    original_root = args.original_root.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve()
    models_manifest = args.models_manifest.expanduser().resolve()
    preregistration_output = args.preregistration_output.expanduser().resolve()
    audit_output = args.audit_output.expanduser().resolve()
    dagger_audit_output = args.dagger_audit_output.expanduser().resolve()
    master = base._read(master_path)
    dagger = base._read(dagger_path)
    if (
        master.get("schema") != "zuma-rl.alphazuma-55-weekend-master-preregistration"
        or master.get("version") != 1
    ):
        raise ValueError("unexpected master preregistration")
    if (
        dagger.get("schema") != "zuma-rl.alphazuma-55-dagger-preregistration"
        or dagger.get("status") != "FROZEN_BEFORE_TRAINING"
        or int(dagger.get("run", {}).get("rounds", -1)) != 2
        or dagger.get("run", {}).get("optimize_after_each_round") is not True
    ):
        raise ValueError("DAgger run is not frozen as two true update rounds")
    run_dir = Path(str(dagger["run"]["run_dir"])).resolve()
    expected_paths = [
        run_dir / "round_01/model_after_round.zip",
        run_dir / "final_model.zip",
        run_dir / "completion.json",
    ]
    if run_dir.exists() or any(path.exists() for path in expected_paths):
        raise FileExistsError("DAgger validation must freeze before training")
    if any(
        path.exists()
        for path in (
            output,
            output_root,
            models_manifest,
            preregistration_output,
            audit_output,
            dagger_audit_output,
        )
    ):
        raise FileExistsError("DAgger validation output already exists")
    if not 1 <= args.parallel_envs <= 55:
        raise ValueError("invalid validation parallel width")

    reserved = dagger["reserved_validation"]
    last_seed = args.base_seed + 54
    if (
        int(reserved["base_seed"]) != args.base_seed
        or int(reserved["last_seed"]) != last_seed
        or int(reserved["levels"]) != 55
    ):
        raise ValueError("validation seeds differ from the pre-outcome reservation")
    registry = master["seed_registry"]["training_validation"]
    if not int(registry["first"]) <= args.base_seed <= last_seed <= int(registry["last"]):
        raise ValueError("DAgger validation escaped its seed registry")
    source = dagger["student_source_model"]
    source_path = Path(str(source["path"])).resolve(strict=True)
    if source["sha256"] != base._sha256(source_path):
        raise ValueError("DAgger source model hash mismatch")

    value = {
        "schema": "zuma-rl.alphazuma-55-dagger-validation-plan",
        "version": 1,
        "status": "FROZEN_BEFORE_DAGGER_TRAINING",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_id": master["campaign_id"],
        "purpose": (
            "Paired fresh-seed comparison of the frozen DAgger source, the "
            "post-round-one policy, and the final post-round-two policy."
        ),
        "master_preregistration": base._artifact(master_path),
        "dagger_preregistration": base._artifact(dagger_path),
        "original_root": str(original_root),
        "models": {
            "source": {
                "id": "dagger-source",
                "training_steps": int(source["training_steps"]),
                **base._artifact(source_path),
            },
            "target": {
                "run_dir": str(run_dir),
                "expected_completion_path": str(run_dir / "completion.json"),
                "expected_round_one": {
                    "id": "dagger-round-01",
                    "expected_path": str(run_dir / "round_01/model_after_round.zip"),
                },
                "expected_final": {
                    "id": "dagger-final",
                    "expected_path": str(run_dir / "final_model.zip"),
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
            "dagger_training_auditor": base._artifact(DAGGER_AUDITOR),
        },
        "outputs": {
            "run_root": str(output_root),
            "models_manifest": str(models_manifest),
            "preregistration": str(preregistration_output),
            "audit_receipt": str(audit_output),
            "dagger_training_audit": str(dagger_audit_output),
        },
        "decision_rule": {
            "ranking": [
                "cleared_levels_descending",
                "wins_descending",
                "median_winning_ticks_ascending",
                "model_sha256_ascending",
            ],
            "all_three_models_reported": True,
            "negative_result_must_be_retained": True,
            "formal_candidate_registration_authorized": False,
        },
        "authority_boundary": {
            "formal_seed_consumption": False,
            "formal_selection_authority": False,
            "training_recipe_change_authority": False,
        },
    }
    base._write_exclusive(output, value)
    print(json.dumps({
        "status": value["status"],
        "output": str(output),
        "sha256": base._sha256(output),
        "seed_range": [args.base_seed, last_seed],
        "target_absent": True,
        "models_after_materialization": 3,
        "formal_seed_consumption": False,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
