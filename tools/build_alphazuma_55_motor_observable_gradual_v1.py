"""Freeze the gradual motor-observable DAgger engineering route."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tools import distill_alphazuma_55 as legacy
from tools import distill_alphazuma_55_polar_dagger_v1 as dagger
from zuma_rl.alphazuma_55 import INCLUDED_LEVELS


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
CAMPAIGN_ID = "alphazuma-55-motor-observable-gradual-s99081628-v1"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "diagnostics/alphazuma-55-motor-observable-gradual-"
    "s99081628-preregistration-v1.json"
)


def _reference(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "sha256": legacy._sha256(resolved)}


def build(*, output: Path) -> dict[str, Any]:
    master_path = (
        PROJECT_ROOT
        / "diagnostics/alphazuma-55-weekend-s81081401-preregistration-v1.json"
    ).resolve(strict=True)
    source_completion_path = Path(
        "/mnt/d/ZumaTraining/alphazuma-55-motor-observable-replay-"
        "s99081616-v3/completion.json"
    ).resolve(strict=True)
    source_completion = legacy._read_json(source_completion_path)
    source_model_path = Path(
        str(source_completion.get("final_model", {}).get("path", ""))
    ).resolve(strict=True)
    teacher_evidence_path = Path(
        "/mnt/d/ZumaTraining/alphazuma-55-weekend-s81081401-v1/engineering/"
        "masked-teacher-v3-fresh-probe-s99081516-v1/result.json"
    ).resolve(strict=True)
    screen_audit_path = (
        PROJECT_ROOT
        / "diagnostics/alphazuma-55-motor-observable-postprocess-"
        "s99081617-screen-audit-v2.json"
    )
    if not screen_audit_path.exists():
        screen_audit_path = Path(
            "/mnt/d/ZumaTraining/alphazuma-55-motor-observable-postprocess-"
            "s99081617-v2/screen-audit.json"
        )
    screen_audit_path = screen_audit_path.resolve(strict=True)
    trainer_path = (
        PROJECT_ROOT
        / "tools/distill_alphazuma_55_motor_observable_gradual_v1.py"
    ).resolve(strict=True)
    run_dir = Path(
        "/mnt/d/ZumaTraining/alphazuma-55-motor-observable-gradual-"
        "s99081628-v1"
    )

    dependencies = {
        "base_dagger_trainer": PROJECT_ROOT
        / "tools/distill_alphazuma_55_polar_dagger_v1.py",
        "raw_intent_collector": PROJECT_ROOT
        / "tools/distill_alphazuma_55_polar_intent_dagger_v1.py",
        "motor_environment_adapter": PROJECT_ROOT
        / "tools/distill_alphazuma_55_motor_observable_replay_v2.py",
        "optimizer": PROJECT_ROOT / "tools/distill_alphazuma_55_settled_v3.py",
        "feature_extractor": PROJECT_ROOT / "src/zuma_rl/revenge_features.py",
        "observable_motor_wrapper": PROJECT_ROOT
        / "src/zuma_rl/observable_human_speedrun.py",
        "teacher": PROJECT_ROOT / "src/zuma_rl/revenge_teacher.py",
    }

    value = {
        "schema": "zuma-rl.alphazuma-55-motor-observable-gradual-preregistration",
        "version": 1,
        "status": "FROZEN_BEFORE_TRAINING",
        "campaign_id": CAMPAIGN_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "objective": (
            "Repair the demonstrated motor-observable closed-loop distribution "
            "gap without changing the 55-level environment, human actuator, "
            "formal gates, or runtime single-policy constraint."
        ),
        "hypothesis": {
            "observed": (
                "The V3 consumed-data loss and action metrics improved through "
                "the last epoch, while the frozen 156-attempt screen produced "
                "only nine wins and every win was on Jungle1."
            ),
            "mechanism": (
                "The prior teacher handoff 0.5 -> 0.1 -> 0.0 moved too quickly "
                "into mostly unrecoverable student states before the classifier "
                "had fitted the teacher motor commands."
            ),
            "intervention": (
                "Continue the exact frozen V3 model with a 1.0 -> 0.9 -> 0.7 "
                "-> 0.5 teacher-execution schedule and 12 optimization epochs "
                "per round."
            ),
            "unchanged": [
                "included 55 levels",
                "actor-observable simulator state",
                "elite-human-v1 actuator limits",
                "win-time-score-v1 reward",
                "raw teacher intent labels",
                "exact runtime action masks",
                "final runtime teacher calls forbidden",
                "one frozen policy file",
                "formal selection and final-blind seed embargoes",
            ],
        },
        "master_preregistration": _reference(master_path),
        "teacher_evidence": _reference(teacher_evidence_path),
        "predecessor_screen_audit": _reference(screen_audit_path),
        "source": {
            "completion": _reference(source_completion_path),
            "model": _reference(source_model_path),
            "selection_rule": "exact frozen V3 final_model before this route",
        },
        "trainer": _reference(trainer_path),
        "builder": _reference(SCRIPT_PATH),
        "implementation": {
            name: _reference(path) for name, path in dependencies.items()
        },
        "levels": list(INCLUDED_LEVELS),
        "run": {
            "id": "alphazuma55-motor-observable-gradual-5080",
            "run_dir": str(run_dir),
            "device": "cuda:1",
            "policy_architecture": (
                "entity_polar_intent_motor_observable_gradual"
            ),
            "motor_observation_profile": "motor-observable-v1",
            "learned_features_dim": 1024,
            "model_seed": 1_544_002_800,
            "training_seed_base": 1_544_000_000,
            "training_seed_last_consumed": 1_544_000_219,
            "rounds": 4,
            "teacher_execution_probabilities": [1.0, 0.9, 0.7, 0.5],
            "parallel_envs": 24,
            "max_ticks": 30000,
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
            "epochs_per_round": 12,
            "checkpoint_interval_epochs": 3,
            "learning_rate": 1.0e-5,
            "verb_loss_weight": 2.0,
            "max_grad_norm": 0.5,
            "aim_loss_scope": "fire_only",
            "aim_loss_mode": "circular_smoothed",
            "aim_smoothing_sigma": 1.5,
            "aim_distance_weight": 0.5,
            "source_is_frozen_motor_v3_final_model": True,
            "aggregate_dataset_across_rounds": True,
            "optimize_after_each_round": True,
            "raw_teacher_intent_labels": True,
            "training_mask_relaxation": "unmask_only_the_teacher_target_verb",
            "exact_action_masks_for_environment_execution": True,
            "exact_action_masks_for_online_policy_inference": True,
            "label_aim_must_remain_exact_mask_legal": True,
            "final_runtime_teacher_calls_forbidden": True,
        },
        "seed_registry": {
            "classification": "engineering_training_only",
            "range": [1_544_000_000, 1_544_000_219],
            "formal_selection_seed_consumption": "NONE",
            "formal_final_blind_seed_consumption": "NONE",
            "continuous_campaign_seed_consumption": "NONE",
        },
        "authority_boundary": {
            "current_campaign_candidate_authority": False,
            "formal_selection_seed_consumption": False,
            "formal_final_blind_seed_consumption": False,
            "continuous_campaign_seed_consumption": False,
            "power_restore_authority": False,
        },
        "promotion_boundary": (
            "This route remains engineering-only until a separately frozen "
            "screen and full55 gate promote a model hash."
        ),
        "outputs": {
            "run_dir": str(run_dir),
            "completion": str(run_dir / "completion.json"),
            "failure": str(run_dir / "failure.json"),
        },
    }
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"preregistration already exists: {output}")
    dagger._write_atomic(output, value)
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    value = build(output=args.output)
    print(
        f"{value['status']} {value['campaign_id']} "
        f"{legacy._sha256(args.output.resolve())}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
