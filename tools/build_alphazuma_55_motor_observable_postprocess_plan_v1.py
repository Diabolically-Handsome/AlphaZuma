"""Freeze the motor-observable checkpoint screen and full55 gate."""

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

from tools import run_alphazuma_55_motor_observable_postprocess_v1 as controller


SCRIPT_PATH = Path(__file__).resolve()
CAMPAIGN_ID = "alphazuma-55-motor-observable-postprocess-s99081615-v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _reference(path: Path) -> dict[str, str]:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "sha256": controller._sha256(resolved)}


def _write_new(path: Path, value: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists():
        temporary.unlink()
        raise FileExistsError(f"postprocess plan already exists: {path}")
    os.replace(temporary, path)


def build(*, project_root: Path, output: Path) -> dict[str, Any]:
    root = project_root.resolve(strict=True)
    master = (
        root
        / "diagnostics/alphazuma-55-weekend-s81081401-preregistration-v1.json"
    )
    route = (
        root
        / "diagnostics/alphazuma-55-motor-observable-replay-"
        "s99081614-preregistration-v2.json"
    )
    template = (
        root
        / "diagnostics/alphazuma-55-polar-intent-wide-validation-"
        "s99081544-preregistration-v1.json"
    )
    template_value = controller._read(template)
    levels_by_id = {
        str(row["id"]): dict(row) for row in template_value["levels"]
    }
    screen_levels = [
        levels_by_id[level_id] for level_id in controller.SCREEN_LEVEL_IDS
    ]
    route_value = controller._read(route)
    run_dir = Path(str(route_value["run"]["run_dir"]))
    output_root = Path(f"/mnt/d/ZumaTraining/{CAMPAIGN_ID}")
    plan = {
        "schema": controller.PLAN_SCHEMA,
        "version": 1,
        "status": "FROZEN_DURING_TRAINING_BEFORE_CHECKPOINT_INFERENCE",
        "created_utc": _utc_now(),
        "campaign_id": CAMPAIGN_ID,
        "objective": (
            "Select early motor-observable checkpoints without training-result "
            "peeking, then require at least 35 wins and 35-level coverage on "
            "a fresh paired full55 engineering matrix before formal promotion."
        ),
        "hypotheses": {
            "alternative": (
                "at least one early actor-visible motor checkpoint closes the "
                "autonomous control gap and clears the 35/55 engineering gate"
            ),
            "null": (
                "motor-state observability does not yield a promotable full55 "
                "single policy under the frozen training recipe"
            ),
        },
        "master_preregistration": _reference(master),
        "training_route": _reference(route),
        "environment_template": _reference(template),
        "expected_training_completion": str(run_dir / "completion.json"),
        "expected_training_failure": str(run_dir / "failure.json"),
        "original_root": "/mnt/d/SteamLibrary/steamapps/common/Zuma's Revenge",
        "screen_levels": screen_levels,
        "full55_levels": [dict(row) for row in template_value["levels"]],
        "candidate_inventory": {
            "baseline": "frozen base epoch-10 model",
            "motor_checkpoints": (
                "all 3 rounds x all 4 per-epoch checkpoints in chronological "
                "order; final_model aliases are excluded"
            ),
            "expected_motor_checkpoints": 12,
            "expected_total_candidates": 13,
            "checkpoint_inclusion_is_metric_independent": True,
        },
        "screen": {
            "registry": "training_validation",
            "base_seed": 1_550_002_400,
            "last_seed": 1_550_002_411,
            "expected_levels": 12,
            "expected_candidates": 13,
            "expected_attempts": 156,
            "attempts_per_level_per_model": 1,
            "ranking": [
                "wins_desc",
                "cleared_levels_desc",
                "total_score_desc",
                "median_winning_ticks_asc",
                "pre_frozen_candidate_order",
            ],
            "select_top": 5,
        },
        "full55_gate": {
            "registry": "training_validation",
            "base_seed": 1_550_002_500,
            "last_seed": 1_550_002_554,
            "expected_levels": 55,
            "maximum_candidates": 5,
            "maximum_expected_attempts": 275,
            "attempts_per_level_per_model": 1,
            "minimum_wins": 35,
            "minimum_cleared_levels": 35,
            "baseline_is_not_promotable": True,
            "formal_promotion_requires_separate_frozen_successor": True,
        },
        "execution": {
            "devices": ["cuda:0", "cuda:1"],
            "parallel_envs_per_shard": 24,
            "paired_models_share_identical_task_seeds": True,
            "capacity_overflow_semantics": "explicit_fail_closed_loss",
        },
        "deadline_utc": "2026-08-17T14:45:00+00:00",
        "implementation": {
            "builder": _reference(SCRIPT_PATH),
            "controller": _reference(
                root
                / "tools/run_alphazuma_55_motor_observable_postprocess_v1.py"
            ),
            "evaluator": _reference(
                root / "tools/evaluate_alphazuma_55_motor_observable.py"
            ),
            "auditor": _reference(
                root
                / "tools/audit_alphazuma_55_motor_observable_postprocess_v1.py"
            ),
            "capacity_safe_evaluator_dependency": _reference(
                root / "tools/evaluate_zero_shot_multilevel_v2.py"
            ),
            "base_evaluator_dependency": _reference(
                root / "tools/evaluate_zero_shot_multilevel.py"
            ),
            "motor_observation_wrapper": _reference(
                root / "src/zuma_rl/observable_human_speedrun.py"
            ),
        },
        "outputs": {
            "output_root": str(output_root),
            "independent_audit_receipt": str(
                root
                / "diagnostics/alphazuma-55-motor-observable-postprocess-"
                "s99081615-independent-audit-v1.json"
            ),
        },
        "authority_boundary": {
            "classification": "engineering_training_validation_only",
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
