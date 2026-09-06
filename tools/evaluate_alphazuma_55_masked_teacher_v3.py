"""Evaluate V3 after applying the exact MaskablePPO action-mask contract."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import numpy as np

from sb3_contrib.common.maskable.utils import get_action_masks
from tools import evaluate_alphazuma_55_geometric_teacher as geometric
from tools import evaluate_alphazuma_55_settled_teacher_v3 as legacy
from tools.evaluate_zero_shot_multilevel import _build_configs
from tools.evaluate_zero_shot_multilevel_v2 import ObservationCapacityFailClosed
from zuma_rl.human_speedrun import HumanSpeedrunWrapper
from zuma_rl.revenge_env import RevengeEnv, RevengeEnvConfig
from zuma_rl.revenge_settled_strategic_teacher_v3 import (
    CurveAwareSettledStrategicRevengeTeacher,
)


SCRIPT_PATH = Path(__file__).resolve()
POLICY_ID = "curve-aware-settled-strategic-masked-v3"


def _validate_preregistration(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    prereg = geometric._read_json(path)
    if (
        prereg.get("schema")
        != "zuma-rl.alphazuma-55-masked-teacher-v3-probe-preregistration"
        or prereg.get("version") != 1
        or prereg.get("status") != "FROZEN_BEFORE_FRESH_ENGINEERING_PROBE"
    ):
        raise ValueError("unexpected masked-teacher-v3 preregistration")
    evaluator = geometric._bound_path(
        prereg["implementation"]["evaluator"], "evaluator"
    )
    if evaluator != SCRIPT_PATH:
        raise ValueError("preregistration binds another masked evaluator")
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
        raise ValueError("masked V3 probe must contain 55 levels")
    if not isinstance(tasks, list) or len(tasks) != 55:
        raise ValueError("masked V3 probe must contain 55 tasks")
    ids = [str(level["id"]) for level in levels]
    if ids != [str(level["id"]) for level in source["levels"]]:
        raise ValueError("masked V3 inventory differs from source")
    if [str(task["level_id"]) for task in tasks] != ids or len(set(ids)) != 55:
        raise ValueError("masked V3 task matrix is invalid")
    if any(str(task["teacher_policy_id"]) != POLICY_ID for task in tasks):
        raise ValueError("masked V3 task references another policy")
    seeds = [int(task["seed"]) for task in tasks]
    seed_range = prereg["seed_range"]
    if seeds != list(range(int(seed_range[0]), int(seed_range[1]) + 1)):
        raise ValueError("masked V3 seeds are not contiguous")
    if not (1_400_000_000 <= seeds[0] <= seeds[-1] <= 1_400_999_999):
        raise ValueError("masked V3 escaped engineering namespace")
    environment = prereg.get("environment")
    expected = copy.deepcopy(source["environment"])
    expected["base_config"]["max_ticks"] = 30_000
    if environment != expected:
        raise ValueError("masked V3 environment differs beyond the horizon")
    workers = int(prereg["execution"]["workers"])
    if not 1 <= workers <= 24:
        raise ValueError("masked V3 workers are outside [1, 24]")
    source = copy.deepcopy(source)
    source["environment"] = environment
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
    teacher = CurveAwareSettledStrategicRevengeTeacher.from_env(base)
    env = HumanSpeedrunWrapper(
        ObservationCapacityFailClosed(base),
        input_config=input_config,
        reward_config=reward_config,
    )
    trajectory = hashlib.sha256()
    names = ("wait", "fire", "swap", "relocate")
    desired_counts = {name: 0 for name in names}
    raw_counts = {name: 0 for name in names}
    forced_waits = 0
    reward_total = 0.0
    decisions = 0
    info: dict[str, Any] = {}
    try:
        observation, _ = env.reset(seed=int(payload["seed"]))
        terminated = False
        truncated = False
        while not (terminated or truncated):
            raw = np.asarray(teacher.act(observation), dtype=np.int64)
            raw_counts[names[int(raw[0])]] += 1
            mask = np.asarray(get_action_masks(env), dtype=np.bool_)
            action = raw.copy()
            if not bool(mask[int(raw[0])]):
                action[0] = 0
                forced_waits += 1
            desired_counts[names[int(action[0])]] += 1
            trajectory.update(action.tobytes())
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
        "raw_teacher_action_counts": raw_counts,
        "desired_action_counts": desired_counts,
        "mask_forced_waits": forced_waits,
        "executed_action_counts": dict(info.get("executed_action_counts", {})),
        "observation_capacity_overflow": bool(
            info.get("observation_capacity_overflow", False)
        ),
        "visible_balls_at_failure": info.get("visible_balls_at_failure"),
        "capacity_limit_balls": info.get("capacity_limit_balls"),
        "trajectory_sha256": "sha256:" + trajectory.hexdigest(),
        "teacher_policy_id": POLICY_ID,
        "aim_tolerance_bins": 2,
    }


def main(argv: list[str] | None = None) -> int:
    legacy.legacy.SCRIPT_PATH = SCRIPT_PATH
    legacy.legacy.POLICY_ID = POLICY_ID
    legacy.legacy._validate_preregistration = _validate_preregistration
    legacy.legacy._run_task = _run_task
    return legacy.legacy.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())

