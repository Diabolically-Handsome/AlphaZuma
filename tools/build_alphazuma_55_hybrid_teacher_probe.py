"""Freeze a fresh-seed probe of a paired-evidence hybrid teacher map."""

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
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tools import build_alphazuma_55_geometric_teacher_probe as legacy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()
MASTER = PROJECT_ROOT / "diagnostics/alphazuma-55-weekend-s81081401-preregistration-v1.json"
SOURCE = PROJECT_ROOT / "diagnostics/alphazuma-55-engineering-baseline-s82081402-preregistration-v1.json"
EVALUATOR = PROJECT_ROOT / "tools/evaluate_alphazuma_55_hybrid_teacher.py"
GEOMETRIC_TEACHER = PROJECT_ROOT / "src/zuma_rl/revenge_teacher.py"
STRATEGIC_TEACHER = PROJECT_ROOT / "src/zuma_rl/revenge_strategic_teacher.py"
CAPACITY_WRAPPER = PROJECT_ROOT / "tools/evaluate_zero_shot_multilevel_v2.py"
GEOMETRIC_RESULT = Path(
    "/mnt/d/ZumaTraining/alphazuma-55-weekend-s81081401-v1/engineering/"
    "geometric-teacher-probe-s99081506-v1/result.json"
)
STRATEGIC_RESULT = Path(
    "/mnt/d/ZumaTraining/alphazuma-55-weekend-s81081401-v1/engineering/"
    "strategic-teacher-probe-s99081507-v1/result.json"
)
SEED_FIRST = 1_400_600_000
GEOMETRIC_POLICY = "geometric-actor-observable-v1"
STRATEGIC_POLICY = "strategic-actor-observable-v1"


def _outcome_rank(row: dict[str, Any]) -> int:
    if row.get("outcome") == "win":
        return 3
    if bool(row.get("time_limit_truncated", False)):
        return 2
    if row.get("outcome") == "loss":
        return 1
    return 0


def _select_teacher(
    geometric: dict[str, Any], strategic: dict[str, Any]
) -> tuple[str, str]:
    geometric_win = geometric.get("outcome") == "win"
    strategic_win = strategic.get("outcome") == "win"
    if geometric_win and strategic_win:
        if int(strategic["ticks"]) < int(geometric["ticks"]):
            return STRATEGIC_POLICY, "both_win_faster_strategic"
        return GEOMETRIC_POLICY, "both_win_faster_geometric"
    if geometric_win:
        return GEOMETRIC_POLICY, "only_geometric_win"
    if strategic_win:
        return STRATEGIC_POLICY, "only_strategic_win"
    geometric_rank = _outcome_rank(geometric)
    strategic_rank = _outcome_rank(strategic)
    if strategic_rank > geometric_rank:
        return STRATEGIC_POLICY, "higher_nonwin_outcome_rank_strategic"
    if geometric_rank > strategic_rank:
        return GEOMETRIC_POLICY, "higher_nonwin_outcome_rank_geometric"
    geometric_score = int(geometric["score"])
    strategic_score = int(strategic["score"])
    if strategic_score > geometric_score:
        return STRATEGIC_POLICY, "higher_nonwin_score_strategic"
    if geometric_score > strategic_score:
        return GEOMETRIC_POLICY, "higher_nonwin_score_geometric"
    if int(strategic["ticks"]) > int(geometric["ticks"]):
        return STRATEGIC_POLICY, "longer_nonwin_survival_strategic"
    if int(geometric["ticks"]) > int(strategic["ticks"]):
        return GEOMETRIC_POLICY, "longer_nonwin_survival_geometric"
    return GEOMETRIC_POLICY, "deterministic_geometric_tie"


def _source_metrics(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "outcome": row.get("outcome"),
        "time_limit_truncated": bool(row.get("time_limit_truncated", False)),
        "ticks": int(row["ticks"]),
        "score": int(row["score"]),
        "trajectory_sha256": str(row["trajectory_sha256"]),
    }


def _attempt_map(result: dict[str, Any], name: str) -> dict[str, dict[str, Any]]:
    if result.get("status") != "COMPLETE":
        raise ValueError(f"{name} result is not COMPLETE")
    attempts = result.get("attempts")
    if not isinstance(attempts, list) or len(attempts) != 55:
        raise ValueError(f"{name} result must contain 55 attempts")
    mapped = {str(row["level_id"]): row for row in attempts}
    if len(mapped) != 55:
        raise ValueError(f"{name} result level IDs are not unique")
    return mapped


def build(*, output_root: Path, workers: int) -> dict[str, Any]:
    if workers < 1 or workers > 24:
        raise ValueError("workers must be in [1, 24]")
    master = legacy._read(MASTER.resolve(strict=True))
    source = legacy._read(SOURCE.resolve(strict=True))
    geometric_result = legacy._read(GEOMETRIC_RESULT.resolve(strict=True))
    strategic_result = legacy._read(STRATEGIC_RESULT.resolve(strict=True))
    geometric = _attempt_map(geometric_result, "geometric")
    strategic = _attempt_map(strategic_result, "strategic")
    levels = source.get("levels")
    if not isinstance(levels, list) or len(levels) != 55:
        raise ValueError("source evaluation contract does not contain 55 levels")
    level_ids = [str(level["id"]) for level in levels]
    if set(level_ids) != set(geometric) or set(level_ids) != set(strategic):
        raise ValueError("paired teacher evidence differs from the 55-level inventory")
    engineering = master["seed_registry"]["engineering_and_calibration"]
    seed_last = SEED_FIRST + len(levels) - 1
    if not (
        int(engineering["first"])
        <= SEED_FIRST
        <= seed_last
        <= int(engineering["last"])
    ):
        raise ValueError("hybrid probe seeds escaped engineering namespace")

    policy_map: list[dict[str, Any]] = []
    for level_id in level_ids:
        policy_id, reason = _select_teacher(
            geometric[level_id], strategic[level_id]
        )
        policy_map.append(
            {
                "level_id": level_id,
                "teacher_policy_id": policy_id,
                "selection_reason": reason,
                "paired_source_metrics": {
                    "geometric": _source_metrics(geometric[level_id]),
                    "strategic": _source_metrics(strategic[level_id]),
                },
            }
        )
    result_path = output_root / "result.json"
    return {
        "schema": "zuma-rl.alphazuma-55-hybrid-teacher-probe-preregistration",
        "version": 1,
        "status": "FROZEN_BEFORE_FRESH_ENGINEERING_PROBE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_id": master["campaign_id"],
        "objective": (
            "Test on fresh engineering seeds whether a deterministic per-level map "
            "selected only from the consumed paired teacher screen generalizes."
        ),
        "master_preregistration": legacy._artifact(MASTER),
        "source_evaluation_contract": legacy._artifact(SOURCE),
        "selection_evidence": {
            "geometric_result": legacy._artifact(GEOMETRIC_RESULT),
            "strategic_result": legacy._artifact(STRATEGIC_RESULT),
            "fresh_seed_relative_to_mapping": True,
            "selection_conditioned_policy_development": True,
            "formal_evidence": False,
            "rule": (
                "wins outrank nonwins; among two wins lower ticks wins; among "
                "nonwins compare outcome rank, score, survival ticks, then geometric"
            ),
        },
        "levels": levels,
        "teacher_policy_map": policy_map,
        "tasks": [
            {
                "level_id": row["level_id"],
                "seed": SEED_FIRST + index,
                "teacher_policy_id": row["teacher_policy_id"],
                "selection_reason": row["selection_reason"],
            }
            for index, row in enumerate(policy_map)
        ],
        "seed_range": [SEED_FIRST, seed_last],
        "execution": {
            "device": "cpu",
            "workers": workers,
            "attempts_per_level": 1,
            "human_input_wrapper": "elite-human-v1",
            "capacity_overflow": "fail_closed_affected_attempt_only",
        },
        "implementation": {
            "builder": legacy._artifact(SCRIPT_PATH),
            "evaluator": legacy._artifact(EVALUATOR),
            "geometric_teacher": legacy._artifact(GEOMETRIC_TEACHER),
            "strategic_teacher": legacy._artifact(STRATEGIC_TEACHER),
            "capacity_wrapper": legacy._artifact(CAPACITY_WRAPPER),
        },
        "outputs": {
            "root": str(output_root.resolve()),
            "result": str(result_path.resolve()),
        },
        "authority_boundary": {
            "formal_seed_consumption": False,
            "training_recipe_change_authorized": False,
            "formal_candidate_registration_authorized": False,
            "positive_holdout_rows_may_support_a_separate_distillation_route": True,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"hybrid teacher output root exists: {output_root}")
    value = build(output_root=output_root, workers=args.workers)
    legacy._write_exclusive(output, value)
    counts: dict[str, int] = {}
    for row in value["teacher_policy_map"]:
        policy_id = str(row["teacher_policy_id"])
        counts[policy_id] = counts.get(policy_id, 0) + 1
    print(
        json.dumps(
            {
                "status": "FROZEN_FRESH_ENGINEERING",
                "output": str(output),
                "sha256": legacy._sha256(output),
                "levels": len(value["levels"]),
                "seed_range": value["seed_range"],
                "workers": value["execution"]["workers"],
                "policy_counts": counts,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
