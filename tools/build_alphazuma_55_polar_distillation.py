"""Freeze an isolated full55 polar distillation engineering run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from zuma_rl.alphazuma_55 import INCLUDED_LEVELS


POLICY_ID = "curve-aware-settled-strategic-masked-v3"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "sha256": _sha256(resolved)}


def _validate_probe(path: Path) -> None:
    result = _read_json(path)
    summary = result.get("summary", {})
    if (
        result.get("status") != "COMPLETE"
        or result.get("formal_seed_consumption") is not False
        or result.get("teacher", {}).get("policy_id") != POLICY_ID
        or int(summary.get("attempts", -1)) != 55
        or int(summary.get("wins", -1)) != 55
        or int(summary.get("level_coverage", -1)) != 55
        or int(summary.get("losses", -1)) != 0
        or int(summary.get("truncations", -1)) != 0
        or int(summary.get("capacity_overflows", -1)) != 0
    ):
        raise ValueError("teacher evidence is not a clean masked 55/55 result")


def build(
    *,
    master_path: Path,
    teacher_result_path: Path,
    run_dir: Path,
    device: str,
    training_seed_base: int,
    model_seed: int,
    validation_seed_base: int,
) -> dict[str, Any]:
    master = _read_json(master_path)
    if (
        master.get("schema")
        != "zuma-rl.alphazuma-55-weekend-master-preregistration"
        or master.get("version") != 1
    ):
        raise ValueError("unexpected master preregistration")
    _validate_probe(teacher_result_path)
    if run_dir.exists():
        raise FileExistsError(f"polar run already exists: {run_dir}")
    training_seed_last = training_seed_base + len(INCLUDED_LEVELS) - 1
    training = master["seed_registry"]["training"]
    if not (
        int(training["first"])
        <= training_seed_base
        <= training_seed_last
        <= int(training["last"])
    ):
        raise ValueError("polar seeds escaped the training registry")
    validation_seed_last = validation_seed_base + len(INCLUDED_LEVELS) - 1
    validation = master["seed_registry"]["training_validation"]
    if not (
        int(validation["first"])
        <= validation_seed_base
        <= validation_seed_last
        <= int(validation["last"])
    ):
        raise ValueError("polar validation seeds escaped the registry")

    root = Path(__file__).resolve().parents[1]
    trainer = root / "tools/distill_alphazuma_55_polar_v1.py"
    implementation_paths = {
        "legacy_distillation_runtime": root / "tools/distill_alphazuma_55.py",
        "settled_v3_distillation_runtime": root
        / "tools/distill_alphazuma_55_settled_v3.py",
        "behavior_clone_batch_primitive": root
        / "tools/pretrain_maskable_ppo_teacher.py",
        "settled_teacher_v3": root
        / "src/zuma_rl/revenge_settled_strategic_teacher_v3.py",
        "settled_teacher_v1": root
        / "src/zuma_rl/revenge_settled_strategic_teacher.py",
        "strategic_teacher": root / "src/zuma_rl/revenge_strategic_teacher.py",
        "geometric_teacher": root / "src/zuma_rl/revenge_teacher.py",
        "environment": root / "src/zuma_rl/revenge_env.py",
        "core": root / "src/zuma_rl/revenge_core.py",
        "features": root / "src/zuma_rl/revenge_features.py",
        "scope": root / "src/zuma_rl/alphazuma_55.py",
    }
    return {
        "schema": "zuma-rl.alphazuma-55-polar-distillation-preregistration",
        "version": 1,
        "status": "FROZEN_BEFORE_TRAINING",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_id": master["campaign_id"],
        "objective": (
            "Train a fresh full55 actor-observable polar policy from the clean "
            "55/55 settled-V3 teacher while restricting aim loss to fire actions."
        ),
        "hypotheses": {
            "alternative": (
                "a direct polar basis plus fire-only aim supervision avoids the "
                "generic-head bottleneck and all-action wait collapse"
            ),
            "null": (
                "the polar policy does not retain all four verbs or does not "
                "generalize beyond the consumed teacher trajectories"
            ),
        },
        "master_preregistration": _artifact(master_path),
        "trainer": _artifact(trainer),
        "implementation": {
            name: _artifact(path) for name, path in implementation_paths.items()
        },
        "teacher_evidence": {
            "exact_mask_fresh_55": _artifact(teacher_result_path),
            "classification": "engineering_teacher_not_formal_blind",
        },
        "prior_evidence": {
            "single_level_polar_capacity": _artifact(
                root
                / "diagnostics/teacher-offline-polar90-s20260873-outcome-v1.json"
            ),
            "single_level_fresh_horizon": _artifact(
                root
                / "diagnostics/teacher-polar-fullhorizon-s20275873-outcome-v1.json"
            ),
            "reset_aim_validation": {
                "path": str(
                    root
                    / "diagnostics/alphazuma-55-settled-v3-reset-aim-v2-validation-s99081524-preregistration-v1.json"
                ),
                "classification": "running_engineering_matrix_not_input_data",
            },
        },
        "levels": list(INCLUDED_LEVELS),
        "run": {
            "id": "alphazuma55-polar-distillation-5090",
            "run_dir": str(run_dir),
            "device": device,
            "policy_architecture": "entity_polar",
            "model_seed": model_seed,
            "training_seed_base": training_seed_base,
            "training_seed_last_consumed": training_seed_last,
            "rounds": 1,
            "parallel_envs": 24,
            "max_ticks": 30_000,
            "reservoir_capacity_by_verb": {
                "wait": 256,
                "fire": 256,
                "swap": 192,
                "hop": 192,
            },
            "sample_stride_by_verb": {
                "wait": 4,
                "fire": 1,
                "swap": 1,
                "hop": 1,
            },
            "target_batch_fraction_by_verb": {
                "wait": 0.45,
                "fire": 0.35,
                "swap": 0.15,
                "hop": 0.05,
            },
            "batch_size": 256,
            "capacity_eval_batch_size": 128,
            "epochs_per_round": 90,
            "checkpoint_interval_epochs": 10,
            "learning_rate": 0.0001,
            "verb_loss_weight": 2.0,
            "max_grad_norm": 0.5,
            "aim_loss_scope": "fire_only",
            "aim_loss_mode": "categorical",
            "aim_smoothing_sigma": 1.5,
            "aim_distance_weight": 0.0,
            "aim_head_initialization": "scaled_identity",
            "aim_head_identity_scale": 5.0,
            "teacher_execution_probability": 1.0,
            "exact_action_masks_before_every_decision": True,
            "masked_verb_fallback": "wait_with_teacher_aim",
            "verb_balanced_optimization": True,
            "final_runtime_teacher_calls_forbidden": True,
        },
        "frozen_engineering_validation": {
            "registry": "training_validation",
            "base_seed": validation_seed_base,
            "last_seed": validation_seed_last,
            "level_order": list(INCLUDED_LEVELS),
            "attempts_per_level_per_model": 1,
            "checkpoint_epochs": [10, 30, 60, 90],
            "baseline_model_id": "v11-full55-v1",
            "ranking": [
                "wins_desc",
                "level_coverage_desc",
                "total_score_desc",
                "median_winning_ticks_asc",
                "earlier_checkpoint_first",
            ],
            "promotion_rule": (
                "engineering evidence only; no current-campaign candidate "
                "promotion regardless of result"
            ),
        },
        "authority_boundary": {
            "classification": "isolated_engineering_training_only",
            "current_campaign_candidate_authority": False,
            "formal_selection_seed_consumption": False,
            "formal_final_blind_seed_consumption": False,
            "continuous_campaign_seed_consumption": False,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-preregistration", required=True, type=Path)
    parser.add_argument("--teacher-result", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--device", choices=("cuda:0", "cuda:1"), default="cuda:0")
    parser.add_argument("--training-seed-base", required=True, type=int)
    parser.add_argument("--model-seed", required=True, type=int)
    parser.add_argument("--validation-seed-base", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite preregistration: {output}")
    result = build(
        master_path=args.master_preregistration.expanduser().resolve(strict=True),
        teacher_result_path=args.teacher_result.expanduser().resolve(strict=True),
        run_dir=args.run_dir.expanduser().resolve(),
        device=args.device,
        training_seed_base=args.training_seed_base,
        model_seed=args.model_seed,
        validation_seed_base=args.validation_seed_base,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(output),
                "sha256": _sha256(output),
                "run": result["run"],
                "validation": result["frozen_engineering_validation"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
