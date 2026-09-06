"""Freeze a fresh engineering probe of the privileged lookahead teacher."""

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

from tools import build_alphazuma_55_geometric_teacher_probe as legacy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()
MASTER = PROJECT_ROOT / "diagnostics/alphazuma-55-weekend-s81081401-preregistration-v1.json"
SOURCE = PROJECT_ROOT / "diagnostics/alphazuma-55-engineering-baseline-s82081402-preregistration-v1.json"
EVALUATOR = PROJECT_ROOT / "tools/evaluate_alphazuma_55_lookahead_teacher.py"
TEACHER = PROJECT_ROOT / "src/zuma_rl/revenge_lookahead_teacher.py"
STRATEGIC_TEACHER = PROJECT_ROOT / "src/zuma_rl/revenge_strategic_teacher.py"
CAPACITY_WRAPPER = PROJECT_ROOT / "tools/evaluate_zero_shot_multilevel_v2.py"
SEED_FIRST = 1_400_800_000


def build(*, output_root: Path, workers: int, max_candidates: int, horizon_ticks: int) -> dict[str, Any]:
    if not 1 <= workers <= 24:
        raise ValueError("workers must be in [1, 24]")
    if max_candidates < 1 or horizon_ticks < 1:
        raise ValueError("lookahead settings must be positive")
    master = legacy._read(MASTER.resolve(strict=True))
    source = legacy._read(SOURCE.resolve(strict=True))
    levels = source.get("levels")
    if not isinstance(levels, list) or len(levels) != 55:
        raise ValueError("source evaluation contract does not contain 55 levels")
    seed_last = SEED_FIRST + len(levels) - 1
    registry = master["seed_registry"]["engineering_and_calibration"]
    if not (int(registry["first"]) <= SEED_FIRST <= seed_last <= int(registry["last"])):
        raise ValueError("lookahead probe escaped engineering seeds")
    result_path = output_root / "result.json"
    return {
        "schema": "zuma-rl.alphazuma-55-lookahead-teacher-probe-preregistration",
        "version": 1,
        "status": "FROZEN_BEFORE_ENGINEERING_PROBE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_id": master["campaign_id"],
        "objective": "Screen a training-only simulator-lookahead teacher for new winning demonstrations under the canonical human-input wrapper.",
        "master_preregistration": legacy._artifact(MASTER),
        "source_evaluation_contract": legacy._artifact(SOURCE),
        "levels": levels,
        "tasks": [{"level_id": level["id"], "seed": SEED_FIRST + index} for index, level in enumerate(levels)],
        "seed_range": [SEED_FIRST, seed_last],
        "execution": {
            "device": "cpu",
            "workers": workers,
            "attempts_per_level": 1,
            "max_candidates": max_candidates,
            "horizon_ticks": horizon_ticks,
            "settle_ticks_after_fire": 32,
            "human_input_wrapper": "elite-human-v1",
            "capacity_overflow": "fail_closed_affected_attempt_only",
        },
        "implementation": {
            "builder": legacy._artifact(SCRIPT_PATH),
            "evaluator": legacy._artifact(EVALUATOR),
            "teacher": legacy._artifact(TEACHER),
            "strategic_teacher": legacy._artifact(STRATEGIC_TEACHER),
            "capacity_wrapper": legacy._artifact(CAPACITY_WRAPPER),
        },
        "outputs": {"root": str(output_root.resolve()), "result": str(result_path.resolve())},
        "authority_boundary": {
            "formal_seed_consumption": False,
            "training_recipe_change_authorized": False,
            "formal_candidate_registration_authorized": False,
            "runtime_policy_valid": False,
            "winning_trajectories_may_support_a_separately_preregistered_distillation_route": True,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--horizon-ticks", type=int, default=240)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"lookahead output root exists: {output_root}")
    value = build(output_root=output_root, workers=args.workers, max_candidates=args.max_candidates, horizon_ticks=args.horizon_ticks)
    legacy._write_exclusive(output, value)
    print(json.dumps({"status": "FROZEN", "output": str(output), "sha256": legacy._sha256(output), "levels": len(value["levels"]), "seed_range": value["seed_range"], "execution": value["execution"], "runtime_policy_valid": False}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
