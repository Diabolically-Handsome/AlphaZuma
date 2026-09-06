"""Freeze the fresh long-horizon probe for curve-aware teacher V3."""

from __future__ import annotations

import argparse
import copy
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
EVALUATOR = PROJECT_ROOT / "tools/evaluate_alphazuma_55_settled_teacher_v3.py"
SETTLED_V3 = PROJECT_ROOT / "src/zuma_rl/revenge_settled_strategic_teacher_v3.py"
SETTLED_V1 = PROJECT_ROOT / "src/zuma_rl/revenge_settled_strategic_teacher.py"
STRATEGIC = PROJECT_ROOT / "src/zuma_rl/revenge_strategic_teacher.py"
CAPACITY_WRAPPER = PROJECT_ROOT / "tools/evaluate_zero_shot_multilevel_v2.py"
V2_FRESH_RESULT = Path(
    "/mnt/d/ZumaTraining/alphazuma-55-weekend-s81081401-v1/engineering/"
    "settled-teacher-v2-fresh-probe-s99081513-v1/result.json"
)
SEED_FIRST = 1_400_940_000
POLICY_ID = "curve-aware-settled-strategic-actor-observable-v3"


def build(*, output_root: Path, workers: int) -> dict[str, Any]:
    if not 1 <= workers <= 24:
        raise ValueError("workers must be in [1, 24]")
    master = legacy._read(MASTER.resolve(strict=True))
    source = legacy._read(SOURCE.resolve(strict=True))
    levels = source.get("levels")
    if not isinstance(levels, list) or len(levels) != 55:
        raise ValueError("source evaluation contract does not contain 55 levels")
    seed_last = SEED_FIRST + len(levels) - 1
    engineering = master["seed_registry"]["engineering_and_calibration"]
    if not (
        int(engineering["first"])
        <= SEED_FIRST
        <= seed_last
        <= int(engineering["last"])
    ):
        raise ValueError("V3 probe seeds escaped engineering namespace")
    environment = copy.deepcopy(source["environment"])
    environment["base_config"]["max_ticks"] = 30_000
    return {
        "schema": "zuma-rl.alphazuma-55-settled-teacher-v3-probe-preregistration",
        "version": 1,
        "status": "FROZEN_BEFORE_FRESH_ENGINEERING_PROBE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_id": master["campaign_id"],
        "objective": (
            "Measure fresh 55-level coverage of the curve-aware swap gate at "
            "the frozen 300-second engineering horizon."
        ),
        "master_preregistration": legacy._artifact(MASTER),
        "source_evaluation_contract": legacy._artifact(SOURCE),
        "environment": environment,
        "teacher": {
            "policy_id": POLICY_ID,
            "aim_tolerance_bins": 2,
            "fire_alignment_gate": True,
            "single_curve_swap_alignment_gate": True,
            "multi_curve_immediate_swap": True,
            "runtime_state": "none",
            "actor_observation_only": True,
            "canonical_human_input_unchanged": True,
        },
        "development_disclosure": {
            "selection_conditioned_engineering": True,
            "v2_fresh_result": legacy._artifact(V2_FRESH_RESULT),
            "v3_calibration_seeds_already_consumed": [
                1400700100,
                1400900009,
                1400900046,
                1400920044,
            ],
            "fresh_seed_relative_to_frozen_v3_teacher_bytes": True,
            "formal_evidence": False,
        },
        "levels": levels,
        "tasks": [
            {
                "level_id": str(level["id"]),
                "seed": SEED_FIRST + index,
                "teacher_policy_id": POLICY_ID,
            }
            for index, level in enumerate(levels)
        ],
        "seed_range": [SEED_FIRST, seed_last],
        "execution": {
            "device": "cpu",
            "workers": workers,
            "attempts_per_level": 1,
            "max_ticks": 30_000,
            "human_input_wrapper": "elite-human-v1",
            "capacity_overflow": "fail_closed_affected_attempt_only",
        },
        "implementation": {
            "builder": legacy._artifact(SCRIPT_PATH),
            "evaluator": legacy._artifact(EVALUATOR),
            "settled_teacher_v3": legacy._artifact(SETTLED_V3),
            "settled_teacher_v1": legacy._artifact(SETTLED_V1),
            "strategic_teacher": legacy._artifact(STRATEGIC),
            "capacity_wrapper": legacy._artifact(CAPACITY_WRAPPER),
        },
        "outputs": {
            "root": str(output_root.resolve()),
            "result": str((output_root / "result.json").resolve()),
        },
        "authority_boundary": {
            "formal_seed_consumption": False,
            "training_recipe_change_authorized": False,
            "formal_candidate_registration_authorized": False,
            "positive_rows_may_support_separate_preregistered_distillation": True,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"settled teacher V3 output root exists: {output_root}")
    value = build(output_root=output_root, workers=args.workers)
    legacy._write_exclusive(output, value)
    print(
        json.dumps(
            {
                "status": "FROZEN_FRESH_ENGINEERING",
                "output": str(output),
                "sha256": legacy._sha256(output),
                "levels": len(value["levels"]),
                "seed_range": value["seed_range"],
                "workers": value["execution"]["workers"],
                "max_ticks": value["execution"]["max_ticks"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
