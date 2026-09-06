"""Evaluate the privileged simulator-lookahead teacher on frozen tasks."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
from typing import Any, Mapping

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import numpy as np

from tools.evaluate_alphazuma_55_geometric_teacher import (
    _bound_path,
    _read_json,
    _sha256,
    _write_atomic,
    _write_exclusive,
)
from tools.evaluate_zero_shot_multilevel import _build_configs
from tools.evaluate_zero_shot_multilevel_v2 import ObservationCapacityFailClosed
from zuma_rl.human_speedrun import HumanSpeedrunWrapper
from zuma_rl.revenge_env import RevengeEnv, RevengeEnvConfig
from zuma_rl.revenge_lookahead_teacher import (
    PrivilegedLookaheadRevengeTeacher,
)


SCRIPT_PATH = Path(__file__).resolve()


def _validate_preregistration(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    prereg = _read_json(path)
    if (
        prereg.get("schema")
        != "zuma-rl.alphazuma-55-lookahead-teacher-probe-preregistration"
        or prereg.get("version") != 1
        or prereg.get("status") != "FROZEN_BEFORE_ENGINEERING_PROBE"
    ):
        raise ValueError("unexpected lookahead-teacher preregistration")
    evaluator = _bound_path(prereg["implementation"]["evaluator"], "evaluator")
    if evaluator != SCRIPT_PATH:
        raise ValueError("preregistration binds another evaluator")
    _bound_path(prereg["implementation"]["teacher"], "lookahead teacher")
    _bound_path(prereg["implementation"]["strategic_teacher"], "strategic teacher")
    _bound_path(prereg["implementation"]["capacity_wrapper"], "capacity wrapper")
    source_path = _bound_path(prereg["source_evaluation_contract"], "source contract")
    source = _read_json(source_path)
    levels = prereg.get("levels")
    tasks = prereg.get("tasks")
    if not isinstance(levels, list) or len(levels) != 55:
        raise ValueError("lookahead probe must contain exactly 55 levels")
    if not isinstance(tasks, list) or len(tasks) != 55:
        raise ValueError("lookahead probe must contain exactly 55 tasks")
    ids = [str(level["id"]) for level in levels]
    if ids != [str(level["id"]) for level in source["levels"]] or len(set(ids)) != 55:
        raise ValueError("lookahead level inventory differs from source contract")
    seeds = [int(task["seed"]) for task in tasks]
    if [str(task["level_id"]) for task in tasks] != ids or len(set(seeds)) != 55:
        raise ValueError("lookahead task matrix is invalid")
    if seeds != list(range(int(prereg["seed_range"][0]), int(prereg["seed_range"][1]) + 1)):
        raise ValueError("lookahead seeds are not the frozen contiguous range")
    if not (1_400_000_000 <= seeds[0] <= seeds[-1] <= 1_400_999_999):
        raise ValueError("lookahead probe escaped engineering seeds")
    execution = prereg["execution"]
    if not 1 <= int(execution["workers"]) <= 24:
        raise ValueError("lookahead worker count is invalid")
    if int(execution["max_candidates"]) < 1 or int(execution["horizon_ticks"]) < 1:
        raise ValueError("lookahead planning settings are invalid")
    if prereg["authority_boundary"]["runtime_policy_valid"] is not False:
        raise ValueError("privileged teacher cannot be declared a runtime policy")
    return prereg, source


def _run_task(payload: dict[str, Any]) -> dict[str, Any]:
    environment = payload["environment"]
    base_config, input_config, reward_config = _build_configs(
        {"environment": environment}
    )
    if not isinstance(base_config, RevengeEnvConfig):
        raise RuntimeError("source contract did not build RevengeEnvConfig")
    base = RevengeEnv(
        config=base_config,
        level_id=str(payload["level_id"]),
        root=Path(str(payload["original_root"])),
        hard=False,
        curve_index=0,
        profile_mode=str(environment["profile_mode"]),
    )
    env = HumanSpeedrunWrapper(
        ObservationCapacityFailClosed(base),
        input_config=input_config,
        reward_config=reward_config,
    )
    teacher = PrivilegedLookaheadRevengeTeacher(
        env,
        max_candidates=int(payload["max_candidates"]),
        horizon_ticks=int(payload["horizon_ticks"]),
        settle_ticks_after_fire=int(payload["settle_ticks_after_fire"]),
    )
    trajectory = hashlib.sha256()
    desired_counts = {"wait": 0, "fire": 0, "swap": 0, "relocate": 0}
    names = ("wait", "fire", "swap", "relocate")
    reward_total = 0.0
    decisions = 0
    info: dict[str, Any] = {}
    try:
        observation, _ = env.reset(seed=int(payload["seed"]))
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action = teacher.act(observation)
            verb = int(action[0])
            desired_counts[names[verb]] += 1
            trajectory.update(np.asarray(action, dtype=np.int64).tobytes())
            observation, reward, terminated, truncated, info = env.step(action)
            reward_total += float(reward)
            decisions += 1
    finally:
        env.close()
    return {
        "task_index": int(payload["task_index"]),
        "level_id": str(payload["level_id"]),
        "display_name": str(payload["display_name"]),
        "curve_count": int(payload["curve_count"]),
        "seed": int(payload["seed"]),
        "outcome": info.get("outcome"),
        "time_limit_truncated": bool(info.get("TimeLimit.truncated", False)),
        "ticks": int(info["ticks"]),
        "seconds": float(info["ticks"]) / 100.0,
        "score": int(info["score"]),
        "reward": reward_total,
        "decisions": decisions,
        "desired_action_counts": desired_counts,
        "executed_action_counts": dict(info.get("executed_action_counts", {})),
        "observation_capacity_overflow": bool(info.get("observation_capacity_overflow", False)),
        "visible_balls_at_failure": info.get("visible_balls_at_failure"),
        "capacity_limit_balls": info.get("capacity_limit_balls"),
        "trajectory_sha256": "sha256:" + trajectory.hexdigest(),
        "teacher_policy_id": PrivilegedLookaheadRevengeTeacher.policy_id,
        "lookahead_metrics": teacher.metrics(),
    }


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
        print(json.dumps({"status": "VALID", "preregistration_sha256": _sha256(prereg_path), "levels": 55, "seed_range": prereg["seed_range"], "formal_seed_consumption": False, "runtime_policy_valid": False}, ensure_ascii=False, indent=2))
        return 0
    if output_root.exists() or result_path.exists():
        raise FileExistsError("lookahead teacher probe output already exists")
    output_root.mkdir(parents=True)
    started = datetime.now(timezone.utc)
    _write_atomic(status_path, {"schema": "zuma-rl.alphazuma-55-lookahead-teacher-probe-status", "version": 1, "status": "RUNNING", "started_utc": started.isoformat(), "completed_attempts": 0, "expected_attempts": 55})
    level_map = {str(level["id"]): level for level in prereg["levels"]}
    execution = prereg["execution"]
    payloads = [
        {
            "task_index": index,
            "level_id": task["level_id"],
            "display_name": level_map[str(task["level_id"])]["display_name"],
            "curve_count": level_map[str(task["level_id"])]["curve_count"],
            "seed": task["seed"],
            "original_root": str(original_root),
            "environment": source["environment"],
            "max_candidates": execution["max_candidates"],
            "horizon_ticks": execution["horizon_ticks"],
            "settle_ticks_after_fire": execution["settle_ticks_after_fire"],
        }
        for index, task in enumerate(prereg["tasks"])
    ]
    rows: list[dict[str, Any]] = []
    try:
        with ProcessPoolExecutor(max_workers=int(execution["workers"]), mp_context=mp.get_context("spawn")) as executor:
            futures = [executor.submit(_run_task, payload) for payload in payloads]
            for future in as_completed(futures):
                rows.append(future.result())
                _write_atomic(status_path, {"schema": "zuma-rl.alphazuma-55-lookahead-teacher-probe-status", "version": 1, "status": "RUNNING", "started_utc": started.isoformat(), "updated_utc": datetime.now(timezone.utc).isoformat(), "completed_attempts": len(rows), "expected_attempts": 55})
    except BaseException as error:
        _write_atomic(status_path, {"schema": "zuma-rl.alphazuma-55-lookahead-teacher-probe-status", "version": 1, "status": "ERROR", "updated_utc": datetime.now(timezone.utc).isoformat(), "completed_attempts": len(rows), "expected_attempts": 55, "error": {"type": type(error).__name__, "message": str(error)}})
        raise
    rows.sort(key=lambda row: int(row["task_index"]))
    wins = [row for row in rows if row["outcome"] == "win"]
    result = {
        "schema": "zuma-rl.alphazuma-55-lookahead-teacher-probe-result",
        "version": 1,
        "status": "COMPLETE",
        "classification": "privileged_engineering_teacher_not_runtime_policy",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration": {"path": str(prereg_path), "sha256": _sha256(prereg_path)},
        "summary": {
            "attempts": len(rows),
            "wins": len(wins),
            "level_coverage": len({row["level_id"] for row in wins}),
            "losses": sum(row["outcome"] == "loss" for row in rows),
            "truncations": sum(row["time_limit_truncated"] for row in rows),
            "capacity_overflows": sum(row["observation_capacity_overflow"] for row in rows),
            "winning_levels": [row["level_id"] for row in wins],
            "planning_decisions": sum(row["lookahead_metrics"]["planning_decisions"] for row in rows),
            "candidate_simulations": sum(row["lookahead_metrics"]["candidate_simulations"] for row in rows),
        },
        "attempts": rows,
        "formal_seed_consumption": False,
        "runtime_policy_valid": False,
    }
    _write_exclusive(result_path, result)
    _write_atomic(status_path, {"schema": "zuma-rl.alphazuma-55-lookahead-teacher-probe-status", "version": 1, "status": "COMPLETE", "completed_utc": datetime.now(timezone.utc).isoformat(), "completed_attempts": 55, "expected_attempts": 55, "result": {"path": str(result_path), "sha256": _sha256(result_path)}})
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"result_sha256={_sha256(result_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
