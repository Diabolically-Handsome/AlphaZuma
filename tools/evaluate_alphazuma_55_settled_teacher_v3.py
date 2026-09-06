"""Evaluate the frozen curve-aware V3 teacher on fresh 55-level seeds."""

from __future__ import annotations

import copy
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tools import evaluate_alphazuma_55_geometric_teacher as geometric
from tools import evaluate_alphazuma_55_settled_teacher_v2 as legacy
from zuma_rl.revenge_settled_strategic_teacher_v3 import (
    CurveAwareSettledStrategicRevengeTeacher,
)


SCRIPT_PATH = Path(__file__).resolve()
POLICY_ID = "curve-aware-settled-strategic-actor-observable-v3"


def _validate_preregistration(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    prereg = geometric._read_json(path)
    if (
        prereg.get("schema")
        != "zuma-rl.alphazuma-55-settled-teacher-v3-probe-preregistration"
        or prereg.get("version") != 1
        or prereg.get("status") != "FROZEN_BEFORE_FRESH_ENGINEERING_PROBE"
    ):
        raise ValueError("unexpected settled-teacher-v3 preregistration")
    evaluator = geometric._bound_path(
        prereg["implementation"]["evaluator"], "evaluator"
    )
    if evaluator != SCRIPT_PATH:
        raise ValueError("preregistration binds another V3 evaluator")
    for key, name in (
        ("settled_teacher_v3", "settled teacher V3"),
        ("settled_teacher_v1", "settled teacher V1"),
        ("strategic_teacher", "strategic teacher"),
        ("capacity_wrapper", "capacity wrapper"),
    ):
        geometric._bound_path(prereg["implementation"][key], name)
    source_path = geometric._bound_path(
        prereg["source_evaluation_contract"], "source contract"
    )
    source = geometric._read_json(source_path)
    levels = prereg.get("levels")
    tasks = prereg.get("tasks")
    if not isinstance(levels, list) or len(levels) != 55:
        raise ValueError("V3 probe must contain exactly 55 levels")
    if not isinstance(tasks, list) or len(tasks) != 55:
        raise ValueError("V3 probe must contain exactly 55 tasks")
    ids = [str(level["id"]) for level in levels]
    if ids != [str(level["id"]) for level in source["levels"]]:
        raise ValueError("V3 probe level inventory differs from source")
    if len(set(ids)) != 55 or [str(task["level_id"]) for task in tasks] != ids:
        raise ValueError("V3 probe task matrix is invalid")
    if any(str(task["teacher_policy_id"]) != POLICY_ID for task in tasks):
        raise ValueError("V3 probe task references another policy")
    seeds = [int(task["seed"]) for task in tasks]
    seed_range = prereg["seed_range"]
    if seeds != list(range(int(seed_range[0]), int(seed_range[1]) + 1)):
        raise ValueError("V3 probe seeds are not contiguous")
    if not (1_400_000_000 <= seeds[0] <= seeds[-1] <= 1_400_999_999):
        raise ValueError("V3 probe escaped the engineering namespace")
    if int(prereg["teacher"]["aim_tolerance_bins"]) != 2:
        raise ValueError("V3 tolerance differs from frozen teacher")
    environment = prereg.get("environment")
    expected = copy.deepcopy(source["environment"])
    expected["base_config"]["max_ticks"] = 30_000
    if environment != expected:
        raise ValueError("V3 environment differs beyond the frozen horizon")
    workers = int(prereg["execution"]["workers"])
    if not 1 <= workers <= 24:
        raise ValueError("V3 workers are outside [1, 24]")
    source = copy.deepcopy(source)
    source["environment"] = environment
    return prereg, source


def _run_task(payload: dict[str, Any]) -> dict[str, Any]:
    geometric.ActorObservableRevengeTeacher = (
        CurveAwareSettledStrategicRevengeTeacher
    )
    row = legacy.legacy._BASE_RUN_TASK(payload)
    row["teacher_policy_id"] = POLICY_ID
    row["aim_tolerance_bins"] = 2
    return row


def main(argv: list[str] | None = None) -> int:
    legacy.SCRIPT_PATH = SCRIPT_PATH
    legacy.POLICY_ID = POLICY_ID
    legacy._validate_preregistration = _validate_preregistration
    legacy._run_task = _run_task
    return legacy.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())

