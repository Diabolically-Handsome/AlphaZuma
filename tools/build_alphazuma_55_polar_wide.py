"""Freeze a wider, multi-pass full55 polar distillation experiment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tools import build_alphazuma_55_polar_distillation as base
from zuma_rl.alphazuma_55 import INCLUDED_LEVELS


SCRIPT_PATH = Path(__file__).resolve()
TRAINING_PASSES = 3
FEATURES_DIM = 1024


def build(
    *, master_path: Path, teacher_result_path: Path, run_dir: Path,
    device: str, training_seed_base: int, model_seed: int,
    validation_seed_base: int
) -> dict[str, Any]:
    master_path = master_path.resolve(strict=True)
    teacher_result_path = teacher_result_path.resolve(strict=True)
    master = base._read_json(master_path)
    if (
        master.get("schema")
        != "zuma-rl.alphazuma-55-weekend-master-preregistration"
        or master.get("version") != 1
    ):
        raise ValueError("unexpected master preregistration")
    base._validate_probe(teacher_result_path)
    if run_dir.exists():
        raise FileExistsError(f"wide polar run already exists: {run_dir}")
    training_seed_last = (
        int(training_seed_base) + TRAINING_PASSES * len(INCLUDED_LEVELS) - 1
    )
    training = master["seed_registry"]["training"]
    if not (
        int(training["first"])
        <= int(training_seed_base)
        <= training_seed_last
        <= int(training["last"])
    ):
        raise ValueError("wide polar seeds escaped the training registry")
    validation_seed_last = int(validation_seed_base) + len(INCLUDED_LEVELS) - 1
    validation = master["seed_registry"]["training_validation"]
    if not (
        int(validation["first"])
        <= int(validation_seed_base)
        <= validation_seed_last
        <= int(validation["last"])
    ):
        raise ValueError("wide polar validation seeds escaped the registry")
    root = SCRIPT_PATH.parents[1]
    trainer = root / "tools/distill_alphazuma_55_polar_wide_v1.py"
    implementation_paths = {
        "base_polar_builder": root / "tools/build_alphazuma_55_polar_distillation.py",
        "base_polar_runtime": root / "tools/distill_alphazuma_55_polar_v1.py",
        "settled_v3_distillation_runtime": root
        / "tools/distill_alphazuma_55_settled_v3.py",
        "legacy_distillation_runtime": root / "tools/distill_alphazuma_55.py",
        "behavior_clone_batch_primitive": root
        / "tools/pretrain_maskable_ppo_teacher.py",
        "settled_teacher_v3": root
        / "src/zuma_rl/revenge_settled_strategic_teacher_v3.py",
        "environment": root / "src/zuma_rl/revenge_env.py",
        "core": root / "src/zuma_rl/revenge_core.py",
        "features": root / "src/zuma_rl/revenge_features.py",
        "scope": root / "src/zuma_rl/alphazuma_55.py",
    }
    return {
        "schema": "zuma-rl.alphazuma-55-polar-wide-distillation-preregistration",
        "version": 1,
        "status": "FROZEN_BEFORE_TRAINING",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_id": master["campaign_id"],
        "objective": (
            "Test whether a 1024-dimensional polar student trained on three "
            "independent full55 teacher passes reduces the compression and "
            "single-trajectory generalization bottleneck."
        ),
        "decision_evidence": {
            "source_polar_parameters": 324122,
            "source_polar_consumed_teacher_passes": 1,
            "source_polar_verb_accuracy": 0.828107,
            "source_polar_fire_aim_within_three_accuracy": 0.646703,
            "dagger_round_00_parameters": 324122,
            "dagger_round_00_fire_aim_within_three_accuracy": 0.586304,
            "dagger_round_01_partial_online_result": {
                "levels_observed": 24,
                "wins": 3,
                "losses": 21,
                "teacher_execution_probability": 0.5,
                "classification": "adaptive engineering evidence before this preregistration",
            },
        },
        "hypotheses": {
            "alternative": (
                "more learned polar capacity plus three independent teacher "
                "streams improves unseen-seed online coverage without changing "
                "the actor interface"
            ),
            "null": (
                "the failure is sequential distribution shift rather than "
                "capacity or trajectory diversity, so the wider clone does not "
                "improve online coverage"
            ),
        },
        "master_preregistration": base._artifact(master_path),
        "builder": base._artifact(SCRIPT_PATH),
        "trainer": base._artifact(trainer),
        "implementation": {
            name: base._artifact(path) for name, path in implementation_paths.items()
        },
        "teacher_evidence": {
            "exact_mask_fresh_55": base._artifact(teacher_result_path),
            "classification": "engineering_teacher_not_formal_blind",
        },
        "levels": list(INCLUDED_LEVELS),
        "run": {
            "id": "alphazuma55-polar-wide-5090",
            "run_dir": str(run_dir.resolve()),
            "device": device,
            "policy_architecture": "entity_polar_wide",
            "learned_features_dim": FEATURES_DIM,
            "model_seed": int(model_seed),
            "training_seed_base": int(training_seed_base),
            "training_seed_last_consumed": training_seed_last,
            "teacher_passes": TRAINING_PASSES,
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
            "batch_size": 512,
            "capacity_eval_batch_size": 128,
            "epochs_per_round": 60,
            "checkpoint_interval_epochs": 10,
            "learning_rate": 0.00005,
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
            "base_seed": int(validation_seed_base),
            "last_seed": validation_seed_last,
            "level_order": list(INCLUDED_LEVELS),
            "attempts_per_level_per_model": 1,
            "checkpoint_epochs": [10, 30, 60],
            "comparison_model_id": "polar-source-final",
            "ranking": [
                "wins_desc",
                "level_coverage_desc",
                "total_score_desc",
                "median_winning_ticks_asc",
                "smaller_model_first_as_tie_break",
            ],
            "promotion_rule": (
                "engineering evidence only; any later formal successor campaign "
                "requires a separately frozen seed range"
            ),
        },
        "authority_boundary": {
            "classification": "isolated_adaptive_engineering_training_only",
            "current_campaign_candidate_authority": False,
            "s99081535_successor_candidate_authority": False,
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
        raise FileExistsError(f"refusing to overwrite wide polar preregistration: {output}")
    value = build(
        master_path=args.master_preregistration.expanduser(),
        teacher_result_path=args.teacher_result.expanduser(),
        run_dir=args.run_dir.expanduser(),
        device=args.device,
        training_seed_base=args.training_seed_base,
        model_seed=args.model_seed,
        validation_seed_base=args.validation_seed_base,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {
                "status": value["status"],
                "output": {"path": str(output), "sha256": base._sha256(output)},
                "run": value["run"],
                "validation": value["frozen_engineering_validation"],
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
