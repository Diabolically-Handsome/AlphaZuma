"""Evaluate the frozen aim-settled strategic teacher on 55 fresh seeds."""

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
from zuma_rl.revenge_settled_strategic_teacher import (
    AimSettledStrategicRevengeTeacher,
)


SCRIPT_PATH = Path(__file__).resolve()
POLICY_ID = "aim-settled-strategic-actor-observable-v1"
_BASE_RUN_TASK = legacy._run_task


def _validate_preregistration(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    prereg = legacy._read_json(path)
    if (
        prereg.get("schema")
        != "zuma-rl.alphazuma-55-settled-teacher-probe-preregistration"
        or prereg.get("version") != 1
        or prereg.get("status") != "FROZEN_BEFORE_FRESH_ENGINEERING_PROBE"
    ):
        raise ValueError("unexpected settled-teacher preregistration")
    evaluator = legacy._bound_path(
        prereg["implementation"]["evaluator"], "evaluator"
    )
    if evaluator != SCRIPT_PATH:
        raise ValueError("preregistration binds another settled evaluator")
    legacy._bound_path(
        prereg["implementation"]["settled_teacher"], "settled teacher"
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

    levels = prereg.get("levels")
    tasks = prereg.get("tasks")
    if not isinstance(levels, list) or len(levels) != 55:
        raise ValueError("settled probe must contain exactly 55 levels")
    if not isinstance(tasks, list) or len(tasks) != 55:
        raise ValueError("settled probe must contain exactly 55 tasks")
    ids = [str(level["id"]) for level in levels]
    if ids != [str(level["id"]) for level in source["levels"]]:
        raise ValueError("settled probe level inventory differs from source")
    if len(set(ids)) != 55 or [str(task["level_id"]) for task in tasks] != ids:
        raise ValueError("settled probe task matrix is invalid")
    if any(str(task["teacher_policy_id"]) != POLICY_ID for task in tasks):
        raise ValueError("settled probe task references another policy")
    seeds = [int(task["seed"]) for task in tasks]
    seed_range = prereg["seed_range"]
    if seeds != list(range(int(seed_range[0]), int(seed_range[1]) + 1)):
        raise ValueError("settled probe seeds are not contiguous")
    if not (1_400_000_000 <= seeds[0] <= seeds[-1] <= 1_400_999_999):
        raise ValueError("settled probe escaped the engineering namespace")
    if int(prereg["teacher"]["aim_tolerance_bins"]) != 2:
        raise ValueError("settled probe tolerance differs from frozen teacher")
    workers = int(prereg["execution"]["workers"])
    if not 1 <= workers <= 24:
        raise ValueError("settled probe workers are outside [1, 24]")
    return prereg, source


def _run_task(payload: dict[str, Any]) -> dict[str, Any]:
    legacy.ActorObservableRevengeTeacher = AimSettledStrategicRevengeTeacher
    row = _BASE_RUN_TASK(payload)
    row["teacher_policy_id"] = POLICY_ID
    row["aim_tolerance_bins"] = 2
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
        raise FileExistsError("settled teacher probe output already exists")
    output_root.mkdir(parents=True)
    started = datetime.now(timezone.utc)
    legacy._write_atomic(
        status_path,
        {
            "schema": "zuma-rl.alphazuma-55-settled-teacher-probe-status",
            "version": 1,
            "status": "RUNNING",
            "started_utc": started.isoformat(),
            "completed_attempts": 0,
            "expected_attempts": 55,
        },
    )
    level_map = {str(level["id"]): level for level in prereg["levels"]}
    payloads = [
        {
            "task_index": index,
            "level_id": task["level_id"],
            "display_name": level_map[str(task["level_id"])]["display_name"],
            "curve_count": level_map[str(task["level_id"])]["curve_count"],
            "seed": task["seed"],
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
            futures = [executor.submit(_run_task, payload) for payload in payloads]
            for future in as_completed(futures):
                rows.append(future.result())
                legacy._write_atomic(
                    status_path,
                    {
                        "schema": "zuma-rl.alphazuma-55-settled-teacher-probe-status",
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
                "schema": "zuma-rl.alphazuma-55-settled-teacher-probe-status",
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
    result = {
        "schema": "zuma-rl.alphazuma-55-settled-teacher-probe-result",
        "version": 1,
        "status": "COMPLETE",
        "classification": "fresh_engineering_holdout_not_selection_or_final_blind",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration": {
            "path": str(prereg_path),
            "sha256": legacy._sha256(prereg_path),
        },
        "source_evaluation_contract": prereg["source_evaluation_contract"],
        "teacher": prereg["teacher"],
        "development_disclosure": prereg["development_disclosure"],
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
    legacy._write_exclusive(result_path, result)
    legacy._write_atomic(
        status_path,
        {
            "schema": "zuma-rl.alphazuma-55-settled-teacher-probe-status",
            "version": 1,
            "status": "COMPLETE",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "completed_attempts": 55,
            "expected_attempts": 55,
            "result": {"path": str(result_path), "sha256": legacy._sha256(result_path)},
        },
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"result_sha256={legacy._sha256(result_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

