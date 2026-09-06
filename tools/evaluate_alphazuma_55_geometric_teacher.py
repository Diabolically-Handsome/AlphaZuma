"""Evaluate the observation-only geometric teacher on the frozen 55 levels."""

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

from tools.evaluate_zero_shot_multilevel import _build_configs
from tools.evaluate_zero_shot_multilevel_v2 import ObservationCapacityFailClosed
from zuma_rl.human_speedrun import HumanSpeedrunWrapper
from zuma_rl.revenge_env import RevengeEnv, RevengeEnvConfig
from zuma_rl.revenge_teacher import ActorObservableRevengeTeacher


SCRIPT_PATH = Path(__file__).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _write_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _bound_path(reference: Any, name: str) -> Path:
    if not isinstance(reference, dict):
        raise ValueError(f"{name} must be an object")
    path = Path(str(reference.get("path", ""))).resolve(strict=True)
    if reference.get("sha256") != _sha256(path):
        raise ValueError(f"{name} hash mismatch")
    return path


def _validate_preregistration(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    prereg = _read_json(path)
    if (
        prereg.get("schema")
        != "zuma-rl.alphazuma-55-geometric-teacher-probe-preregistration"
        or prereg.get("version") != 1
        or prereg.get("status") != "FROZEN_BEFORE_ENGINEERING_PROBE"
    ):
        raise ValueError("unexpected geometric-teacher preregistration")
    evaluator = _bound_path(prereg["implementation"]["evaluator"], "evaluator")
    if evaluator != SCRIPT_PATH:
        raise ValueError("preregistration binds another evaluator")
    _bound_path(prereg["implementation"]["teacher"], "teacher")
    _bound_path(prereg["implementation"]["capacity_wrapper"], "capacity wrapper")
    source_path = _bound_path(prereg["source_evaluation_contract"], "source contract")
    source = _read_json(source_path)
    levels = prereg.get("levels")
    tasks = prereg.get("tasks")
    if not isinstance(levels, list) or len(levels) != 55:
        raise ValueError("teacher probe must contain exactly 55 levels")
    if not isinstance(tasks, list) or len(tasks) != 55:
        raise ValueError("teacher probe must contain exactly 55 tasks")
    ids = [str(level["id"]) for level in levels]
    source_ids = [str(level["id"]) for level in source["levels"]]
    if ids != source_ids or len(set(ids)) != 55:
        raise ValueError("teacher probe level inventory differs from source contract")
    task_ids = [str(task["level_id"]) for task in tasks]
    seeds = [int(task["seed"]) for task in tasks]
    if task_ids != ids or len(set(seeds)) != 55:
        raise ValueError("teacher probe task matrix is invalid")
    seed_range = prereg["seed_range"]
    if seeds != list(range(int(seed_range[0]), int(seed_range[1]) + 1)):
        raise ValueError("teacher probe seeds are not the frozen contiguous range")
    if not (1_400_000_000 <= seeds[0] <= seeds[-1] <= 1_400_999_999):
        raise ValueError("teacher probe escaped the engineering seed namespace")
    if int(prereg["execution"]["workers"]) < 1:
        raise ValueError("worker count must be positive")
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
    teacher = ActorObservableRevengeTeacher.from_env(base)
    env = HumanSpeedrunWrapper(
        ObservationCapacityFailClosed(base),
        input_config=input_config,
        reward_config=reward_config,
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
        "outcome": str(info["outcome"]),
        "time_limit_truncated": bool(info.get("TimeLimit.truncated", False)),
        "ticks": int(info["ticks"]),
        "seconds": float(info["ticks"]) / 100.0,
        "score": int(info["score"]),
        "reward": reward_total,
        "decisions": decisions,
        "desired_action_counts": desired_counts,
        "executed_action_counts": dict(info.get("executed_action_counts", {})),
        "observation_capacity_overflow": bool(
            info.get("observation_capacity_overflow", False)
        ),
        "visible_balls_at_failure": info.get("visible_balls_at_failure"),
        "capacity_limit_balls": info.get("capacity_limit_balls"),
        "trajectory_sha256": "sha256:" + trajectory.hexdigest(),
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
        print(
            json.dumps(
                {
                    "status": "VALID",
                    "preregistration": {
                        "path": str(prereg_path),
                        "sha256": _sha256(prereg_path),
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
        raise FileExistsError("teacher probe output already exists")
    output_root.mkdir(parents=True)
    started = datetime.now(timezone.utc)
    _write_atomic(
        status_path,
        {
            "schema": "zuma-rl.alphazuma-55-geometric-teacher-probe-status",
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
                _write_atomic(
                    status_path,
                    {
                        "schema": "zuma-rl.alphazuma-55-geometric-teacher-probe-status",
                        "version": 1,
                        "status": "RUNNING",
                        "started_utc": started.isoformat(),
                        "updated_utc": datetime.now(timezone.utc).isoformat(),
                        "completed_attempts": len(rows),
                        "expected_attempts": 55,
                    },
                )
    except BaseException as error:
        _write_atomic(
            status_path,
            {
                "schema": "zuma-rl.alphazuma-55-geometric-teacher-probe-status",
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
        "schema": "zuma-rl.alphazuma-55-geometric-teacher-probe-result",
        "version": 1,
        "status": "COMPLETE",
        "classification": "engineering_only_not_selection_or_final_blind",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration": {"path": str(prereg_path), "sha256": _sha256(prereg_path)},
        "source_evaluation_contract": prereg["source_evaluation_contract"],
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
    _write_exclusive(result_path, payload)
    _write_atomic(
        status_path,
        {
            "schema": "zuma-rl.alphazuma-55-geometric-teacher-probe-status",
            "version": 1,
            "status": "COMPLETE",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "completed_attempts": 55,
            "expected_attempts": 55,
            "result": {"path": str(result_path), "sha256": _sha256(result_path)},
        },
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"result_sha256={_sha256(result_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
