"""Evaluate a frozen per-level hybrid of the two observation-only teachers."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import multiprocessing as mp
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tools import evaluate_alphazuma_55_geometric_teacher as legacy
from zuma_rl.revenge_strategic_teacher import (
    StrategicActorObservableRevengeTeacher,
)
from zuma_rl.revenge_teacher import ActorObservableRevengeTeacher


SCRIPT_PATH = Path(__file__).resolve()
_BASE_RUN_TASK = legacy._run_task
_POLICIES = {
    "geometric-actor-observable-v1": ActorObservableRevengeTeacher,
    "strategic-actor-observable-v1": StrategicActorObservableRevengeTeacher,
}


def _validate_preregistration(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    prereg = legacy._read_json(path)
    if (
        prereg.get("schema")
        != "zuma-rl.alphazuma-55-hybrid-teacher-probe-preregistration"
        or prereg.get("version") != 1
        or prereg.get("status") != "FROZEN_BEFORE_FRESH_ENGINEERING_PROBE"
    ):
        raise ValueError("unexpected hybrid-teacher preregistration")
    evaluator = legacy._bound_path(
        prereg["implementation"]["evaluator"], "evaluator"
    )
    if evaluator != SCRIPT_PATH:
        raise ValueError("preregistration binds another evaluator")
    legacy._bound_path(
        prereg["implementation"]["geometric_teacher"], "geometric teacher"
    )
    legacy._bound_path(
        prereg["implementation"]["strategic_teacher"], "strategic teacher"
    )
    legacy._bound_path(
        prereg["implementation"]["capacity_wrapper"], "capacity wrapper"
    )
    source_path = legacy._bound_path(
        prereg["source_evaluation_contract"], "source contract"
    )
    source = legacy._read_json(source_path)
    legacy._bound_path(
        prereg["selection_evidence"]["geometric_result"], "geometric result"
    )
    legacy._bound_path(
        prereg["selection_evidence"]["strategic_result"], "strategic result"
    )

    levels = prereg.get("levels")
    tasks = prereg.get("tasks")
    policy_map = prereg.get("teacher_policy_map")
    if not isinstance(levels, list) or len(levels) != 55:
        raise ValueError("hybrid probe must contain exactly 55 levels")
    if not isinstance(tasks, list) or len(tasks) != 55:
        raise ValueError("hybrid probe must contain exactly 55 tasks")
    if not isinstance(policy_map, list) or len(policy_map) != 55:
        raise ValueError("hybrid probe must contain exactly 55 policy-map rows")
    ids = [str(level["id"]) for level in levels]
    source_ids = [str(level["id"]) for level in source["levels"]]
    if ids != source_ids or len(set(ids)) != 55:
        raise ValueError("hybrid probe level inventory differs from source contract")
    task_ids = [str(task["level_id"]) for task in tasks]
    map_ids = [str(row["level_id"]) for row in policy_map]
    if task_ids != ids or map_ids != ids:
        raise ValueError("hybrid task or policy-map order differs from level inventory")
    policies = [str(task["teacher_policy_id"]) for task in tasks]
    if any(policy not in _POLICIES for policy in policies):
        raise ValueError("hybrid task references an unknown teacher policy")
    if policies != [str(row["teacher_policy_id"]) for row in policy_map]:
        raise ValueError("hybrid tasks do not match the frozen teacher policy map")
    if len({str(task["selection_reason"]) for task in tasks}) < 1:
        raise ValueError("hybrid tasks are missing selection reasons")
    if [str(task["selection_reason"]) for task in tasks] != [
        str(row["selection_reason"]) for row in policy_map
    ]:
        raise ValueError("hybrid task reasons do not match the frozen map")

    seeds = [int(task["seed"]) for task in tasks]
    seed_range = prereg["seed_range"]
    if seeds != list(range(int(seed_range[0]), int(seed_range[1]) + 1)):
        raise ValueError("hybrid probe seeds are not the frozen contiguous range")
    if not (1_400_000_000 <= seeds[0] <= seeds[-1] <= 1_400_999_999):
        raise ValueError("hybrid probe escaped the engineering seed namespace")
    if prereg["selection_evidence"].get("fresh_seed_relative_to_mapping") is not True:
        raise ValueError("hybrid probe does not declare a fresh mapping holdout")
    if int(prereg["execution"]["workers"]) < 1:
        raise ValueError("worker count must be positive")
    return prereg, source


def _run_task(payload: dict[str, Any]) -> dict[str, Any]:
    policy_id = str(payload["teacher_policy_id"])
    legacy.ActorObservableRevengeTeacher = _POLICIES[policy_id]
    row = _BASE_RUN_TASK(payload)
    row["teacher_policy_id"] = policy_id
    row["selection_reason"] = str(payload["selection_reason"])
    return row


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--original-root", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prereg_path = args.preregistration.expanduser().resolve(strict=True)
    prereg, source = _validate_preregistration(prereg_path)
    original_root = args.original_root.expanduser().resolve(strict=True)
    output_root = Path(str(prereg["outputs"]["root"])).resolve()
    result_path = Path(str(prereg["outputs"]["result"])).resolve()
    status_path = output_root / "status.json"
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "VALID",
                    "preregistration": {
                        "path": str(prereg_path),
                        "sha256": legacy._sha256(prereg_path),
                    },
                    "levels": 55,
                    "workers": int(prereg["execution"]["workers"]),
                    "seed_range": prereg["seed_range"],
                    "formal_seed_consumption": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if output_root.exists() or result_path.exists():
        raise FileExistsError("hybrid teacher probe output already exists")
    output_root.mkdir(parents=True)
    started = datetime.now(timezone.utc)
    legacy._write_atomic(
        status_path,
        {
            "schema": "zuma-rl.alphazuma-55-hybrid-teacher-probe-status",
            "version": 1,
            "status": "RUNNING",
            "started_utc": started.isoformat(),
            "completed_attempts": 0,
            "expected_attempts": 55,
        },
    )
    level_map = {str(level["id"]): level for level in prereg["levels"]}
    task_payloads = [
        {
            "task_index": index,
            "level_id": task["level_id"],
            "display_name": level_map[str(task["level_id"])]["display_name"],
            "curve_count": level_map[str(task["level_id"])]["curve_count"],
            "seed": task["seed"],
            "teacher_policy_id": task["teacher_policy_id"],
            "selection_reason": task["selection_reason"],
            "original_root": str(original_root),
            "environment": source["environment"],
        }
        for index, task in enumerate(prereg["tasks"])
    ]
    rows: list[dict[str, Any]] = []
    try:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=int(prereg["execution"]["workers"]),
            mp_context=context,
        ) as executor:
            futures = [executor.submit(_run_task, payload) for payload in task_payloads]
            for future in as_completed(futures):
                rows.append(future.result())
                legacy._write_atomic(
                    status_path,
                    {
                        "schema": "zuma-rl.alphazuma-55-hybrid-teacher-probe-status",
                        "version": 1,
                        "status": "RUNNING",
                        "started_utc": started.isoformat(),
                        "updated_utc": datetime.now(timezone.utc).isoformat(),
                        "completed_attempts": len(rows),
                        "expected_attempts": 55,
                    },
                )
    except BaseException as error:
        legacy._write_atomic(
            status_path,
            {
                "schema": "zuma-rl.alphazuma-55-hybrid-teacher-probe-status",
                "version": 1,
                "status": "ERROR",
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "completed_attempts": len(rows),
                "expected_attempts": 55,
                "error": {"type": type(error).__name__, "message": str(error)},
            },
        )
        raise
    rows.sort(key=lambda row: int(row["task_index"]))
    wins = [row for row in rows if row["outcome"] == "win"]
    payload = {
        "schema": "zuma-rl.alphazuma-55-hybrid-teacher-probe-result",
        "version": 1,
        "status": "COMPLETE",
        "classification": "fresh_engineering_holdout_not_selection_or_final_blind",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration": {
            "path": str(prereg_path),
            "sha256": legacy._sha256(prereg_path),
        },
        "source_evaluation_contract": prereg["source_evaluation_contract"],
        "selection_evidence": prereg["selection_evidence"],
        "summary": {
            "attempts": len(rows),
            "wins": len(wins),
            "level_coverage": len({row["level_id"] for row in wins}),
            "losses": sum(row["outcome"] == "loss" for row in rows),
            "truncations": sum(row["time_limit_truncated"] for row in rows),
            "capacity_overflows": sum(
                row["observation_capacity_overflow"] for row in rows
            ),
            "winning_levels": [row["level_id"] for row in wins],
        },
        "attempts": rows,
        "formal_seed_consumption": False,
    }
    legacy._write_exclusive(result_path, payload)
    legacy._write_atomic(
        status_path,
        {
            "schema": "zuma-rl.alphazuma-55-hybrid-teacher-probe-status",
            "version": 1,
            "status": "COMPLETE",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "completed_attempts": 55,
            "expected_attempts": 55,
            "result": {
                "path": str(result_path),
                "sha256": legacy._sha256(result_path),
            },
        },
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"result_sha256={legacy._sha256(result_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
