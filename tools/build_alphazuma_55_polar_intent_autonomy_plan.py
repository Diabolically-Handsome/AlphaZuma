"""Freeze a student-only raw-intent continuation before source termination."""

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

from tools import distill_alphazuma_55 as legacy
from zuma_rl.alphazuma_55 import INCLUDED_LEVELS


SCRIPT_PATH = Path(__file__).resolve()
SCHEDULE = (0.0, 0.0, 0.0, 0.0)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _reference(path: Path) -> dict[str, str]:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "sha256": legacy._sha256(resolved)}


def _write_new(path: Path, value: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists():
        temporary.unlink()
        raise FileExistsError(f"plan already exists: {path}")
    os.replace(temporary, path)


def build(*, project_root: Path, output: Path) -> dict[str, Any]:
    root = project_root.resolve(strict=True)
    master = (
        root
        / "diagnostics/alphazuma-55-weekend-s81081401-preregistration-v1.json"
    )
    source_prereg = (
        root
        / "diagnostics/alphazuma-55-polar-intent-dagger-"
        "s99081549-preregistration-v1.json"
    )
    source_value = json.loads(source_prereg.read_text(encoding="utf-8-sig"))
    source_run = Path(str(source_value["run"]["run_dir"]))
    run_dir = Path(
        "/mnt/d/ZumaTraining/alphazuma-55-polar-intent-autonomy-"
        "s99081601-v1"
    )
    plan = {
        "schema": "zuma-rl.alphazuma-55-polar-intent-autonomy-plan",
        "version": 1,
        "status": "FROZEN_BEFORE_SOURCE_TERMINAL_RESULT_INSPECTION",
        "created_utc": _utc_now(),
        "campaign_id": "alphazuma-55-polar-intent-autonomy-s99081601-v1",
        "objective": (
            "Continue the frozen raw-intent DAgger final policy through four "
            "student-only collection rounds so teacher labels cover states "
            "caused by the policy's own compounding full55 errors."
        ),
        "decision_evidence_available_before_freeze": {
            "source_round_00": {
                "teacher_execution_probability": 0.75,
                "wins": 8,
                "attempts": 55,
                "retained_samples": 36120,
                "classification": "training collection evidence only",
            },
            "source_round_01": {
                "teacher_execution_probability": 0.25,
                "wins": 2,
                "attempts": 55,
                "retained_samples": 23659,
                "classification": "training collection evidence only",
            },
            "source_round_02_partial_before_terminal_result": {
                "teacher_execution_probability": 0.0,
                "wins": 0,
                "completed_attempts": 24,
                "expected_attempts": 55,
                "retained_samples": 8697,
                "observed_updated_utc": "2026-08-16T00:26:35.913400+00:00",
                "classification": (
                    "pre-terminal training collection evidence only"
                ),
            },
            "interpretation": (
                "offline label capacity improved while closed-loop autonomy "
                "remained weak; the isolated continuation tests covariate "
                "shift without changing representation or optimizer recipe"
            ),
        },
        "hypotheses": {
            "alternative": (
                "repeated raw-intent supervision on student-visited states "
                "improves independent full55 wins and coverage"
            ),
            "null": (
                "representation, objective, or teacher limitations dominate, "
                "so additional student-state coverage does not recover wins"
            ),
        },
        "master_preregistration": _reference(master),
        "source": {
            "preregistration": _reference(source_prereg),
            "expected_completion": str(source_run / "completion.json"),
            "expected_failure": str(source_run / "failure.json"),
            "selection_rule": (
                "exact frozen source final_model only; no checkpoint or "
                "metric-conditioned source selection"
            ),
        },
        "teacher_evidence": source_value["teacher_evidence"],
        "levels": list(INCLUDED_LEVELS),
        "original_root": "/mnt/d/SteamLibrary/steamapps/common/Zuma's Revenge",
        "run": {
            "id": "alphazuma55-polar-intent-autonomy-5080",
            "run_dir": str(run_dir),
            "device": "cuda:1",
            "policy_architecture": (
                "entity_polar_intent_wide_dagger_autonomy"
            ),
            "learned_features_dim": 1024,
            "model_seed": 1542005500,
            "training_seed_base": 1542005000,
            "training_seed_last_consumed": 1542005219,
            "rounds": 4,
            "teacher_execution_probabilities": list(SCHEDULE),
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
            "epochs_per_round": 18,
            "checkpoint_interval_epochs": 6,
            "learning_rate": 0.00002,
            "verb_loss_weight": 2.0,
            "max_grad_norm": 0.5,
            "aim_loss_scope": "fire_only",
            "aim_loss_mode": "categorical",
            "aim_smoothing_sigma": 1.5,
            "aim_distance_weight": 0.0,
            "exact_action_masks_before_every_decision": True,
            "masked_verb_fallback": "wait_with_teacher_aim",
            "verb_balanced_optimization": True,
            "aggregate_dataset_across_rounds": True,
            "optimize_after_each_round": True,
            "source_is_frozen_raw_intent_dagger_final_model": True,
            "student_state_collection": True,
            "raw_teacher_intent_labels": True,
            "training_mask_relaxation": (
                "unmask_only_the_teacher_target_verb"
            ),
            "exact_action_masks_for_environment_execution": True,
            "exact_action_masks_for_online_policy_inference": True,
            "invalid_intent_execution": "unchanged_wait_fallback",
            "label_aim_must_remain_exact_mask_legal": True,
            "final_runtime_teacher_calls_forbidden": True,
            "continuation_changes_only_state_coverage": True,
        },
        "frozen_engineering_validation": {
            "registry": "training_validation",
            "base_seed": 1550002000,
            "last_seed": 1550002054,
            "level_order": list(INCLUDED_LEVELS),
            "attempts_per_level_per_model": 1,
            "models": [
                "source-final",
                "autonomy-round-00",
                "autonomy-round-01",
                "autonomy-round-02",
                "autonomy-round-03",
            ],
            "expected_attempts": 275,
            "ranking": [
                "wins_desc",
                "level_coverage_desc",
                "total_score_desc",
                "median_winning_ticks_asc",
                "earlier_round_first",
            ],
            "promotion_rule": (
                "engineering evidence only; minimum 35 wins and 35-level "
                "coverage, followed by a separately frozen successor"
            ),
        },
        "implementation": {
            "builder": _reference(SCRIPT_PATH),
            "trainer": _reference(
                root
                / "tools/distill_alphazuma_55_polar_intent_autonomy_v1.py"
            ),
            "materializer": _reference(
                root
                / "tools/materialize_alphazuma_55_polar_intent_autonomy.py"
            ),
            "watcher": _reference(
                root / "tools/watch_alphazuma_55_polar_intent_autonomy.py"
            ),
        },
        "implementation_dependencies": {
            "legacy_distillation_runtime": _reference(
                root / "tools/distill_alphazuma_55.py"
            ),
            "settled_v3_distillation_runtime": _reference(
                root / "tools/distill_alphazuma_55_settled_v3.py"
            ),
            "polar_dagger_runtime": _reference(
                root / "tools/distill_alphazuma_55_polar_dagger_v1.py"
            ),
            "raw_intent_dagger_runtime": _reference(
                root
                / "tools/distill_alphazuma_55_polar_intent_dagger_v1.py"
            ),
            "raw_intent_runtime": _reference(
                root / "tools/distill_alphazuma_55_polar_intent_wide_v1.py"
            ),
            "wide_runtime": _reference(
                root / "tools/distill_alphazuma_55_polar_wide_v1.py"
            ),
            "behavior_clone_batch_primitive": _reference(
                root / "tools/pretrain_maskable_ppo_teacher.py"
            ),
            "settled_teacher_v3": _reference(
                root / "src/zuma_rl/revenge_settled_strategic_teacher_v3.py"
            ),
            "environment": _reference(root / "src/zuma_rl/revenge_env.py"),
            "core": _reference(root / "src/zuma_rl/revenge_core.py"),
            "features": _reference(root / "src/zuma_rl/revenge_features.py"),
            "scope": _reference(root / "src/zuma_rl/alphazuma_55.py"),
        },
        "outputs": {
            "run_dir": str(run_dir),
            "preregistration": str(
                root
                / "diagnostics/alphazuma-55-polar-intent-autonomy-"
                "s99081601-preregistration-v1.json"
            ),
            "status_root": (
                "/mnt/d/ZumaTraining/alphazuma-55-polar-intent-autonomy-"
                "s99081601-watch-v1"
            ),
        },
        "authority_boundary": {
            "classification": (
                "isolated_student_state_engineering_training_only"
            ),
            "current_campaign_candidate_authority": False,
            "predecessor_successor_candidate_authority": False,
            "formal_selection_seed_consumption": False,
            "formal_final_blind_seed_consumption": False,
            "continuous_campaign_seed_consumption": False,
        },
    }
    _write_new(output, plan)
    return plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = build(project_root=args.project_root, output=args.output)
    print(json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
