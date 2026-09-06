"""Balanced-fire distillation for the AlphaZuma 55 single policy."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
from typing import Any, Sequence

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import numpy as np

from tools import distill_alphazuma_55 as legacy
from zuma_rl.alphazuma_55 import INCLUDED_LEVELS
from zuma_rl.human_speedrun import EliteHumanInputConfig, WinFirstRewardConfig
from zuma_rl.revenge_strategic_teacher import (
    StrategicActorObservableRevengeTeacher,
)
from zuma_rl.revenge_teacher import ActorObservableRevengeTeacher


SCRIPT_PATH = Path(__file__).resolve()
GEOMETRIC_POLICY = "geometric-actor-observable-v1"
STRATEGIC_POLICY = "strategic-actor-observable-v1"
_OBSERVATION_TEACHERS = {
    GEOMETRIC_POLICY: ActorObservableRevengeTeacher,
    STRATEGIC_POLICY: StrategicActorObservableRevengeTeacher,
}


def _validate_preregistration(path: Path) -> dict[str, Any]:
    prereg = legacy._read_json(path)
    if (
        prereg.get("schema")
        != "zuma-rl.alphazuma-55-balanced-distillation-preregistration"
        or prereg.get("version") != 1
        or prereg.get("status") != "FROZEN_BEFORE_TRAINING"
    ):
        raise ValueError("unexpected balanced-distillation preregistration")
    trainer = prereg.get("trainer", {})
    if Path(str(trainer.get("path", ""))).resolve() != SCRIPT_PATH:
        raise ValueError("preregistration binds another balanced distiller")
    if trainer.get("sha256") != legacy._sha256(SCRIPT_PATH):
        raise ValueError("balanced distiller bytes differ from preregistration")
    for name, artifact in prereg.get("implementation", {}).items():
        artifact_path = Path(str(artifact["path"])).resolve(strict=True)
        if artifact.get("sha256") != legacy._sha256(artifact_path):
            raise ValueError(f"implementation bytes differ: {name}")
    source = prereg["student_source_model"]
    source_path = Path(str(source["path"])).resolve(strict=True)
    if source["sha256"] != legacy._sha256(source_path):
        raise ValueError("student source model bytes differ")
    teacher_map_path = Path(str(prereg["teacher_map"]["path"])).resolve(
        strict=True
    )
    if prereg["teacher_map"]["sha256"] != legacy._sha256(teacher_map_path):
        raise ValueError("balanced teacher map bytes differ")
    teacher_map = legacy._read_json(teacher_map_path)
    if (
        teacher_map.get("schema")
        != "zuma-rl.alphazuma-55-balanced-teacher-map"
        or teacher_map.get("status") != "FROZEN"
    ):
        raise ValueError("balanced teacher map is not frozen")
    frozen_levels = [row["level_id"] for row in teacher_map["levels"]]
    if frozen_levels != list(INCLUDED_LEVELS):
        raise ValueError("balanced teacher map level order differs from campaign scope")
    for row in teacher_map["levels"]:
        model = row["teacher_model"]
        model_path = Path(str(model["path"])).resolve(strict=True)
        if model["sha256"] != legacy._sha256(model_path):
            raise ValueError(f"teacher model bytes differ: {model['id']}")
        if row.get("observation_teacher_policy_id") not in _OBSERVATION_TEACHERS:
            raise ValueError(
                f"unknown observation teacher for {row['level_id']}"
            )
    run = prereg["run"]
    run_dir = Path(str(run["run_dir"])).resolve()
    if run_dir.exists():
        raise FileExistsError(f"refusing to reuse balanced distillation run: {run_dir}")
    positive = (
        "rounds",
        "samples_per_level_per_round",
        "fire_samples_per_level_per_round",
        "sample_stride",
        "batch_size",
        "epochs_per_round",
        "parallel_envs",
        "max_ticks",
    )
    if any(int(run[name]) < 1 for name in positive):
        raise ValueError("balanced distillation integer hyperparameters must be positive")
    if not 0.0 <= float(run["teacher_execution_probability_after_round_zero"]) <= 1.0:
        raise ValueError("teacher execution probability is outside [0, 1]")
    if float(run["learning_rate"]) <= 0.0:
        raise ValueError("learning rate must be positive")
    return prereg


def _collect_round_balanced(
    *,
    original_root: Path,
    teacher_rows: Sequence[dict[str, Any]],
    teacher_models: dict[str, Any],
    student: Any,
    device: str,
    round_index: int,
    seed_base: int,
    parallel_envs: int,
    max_ticks: int,
    samples_per_level: int,
    fire_samples_per_level: int,
    sample_stride: int,
    teacher_execution_probability: float,
    model_seed: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray]], dict[str, Any]]:
    del device
    from sb3_contrib.common.maskable.utils import get_action_masks
    from stable_baselines3.common.vec_env import SubprocVecEnv

    input_config = EliteHumanInputConfig(
        profile_id="elite-human-v1",
        reaction_delay_ticks=12,
        max_aim_speed_degrees_per_second=1080.0,
        max_aim_acceleration_degrees_per_second_squared=18000.0,
        min_button_interval_ticks=5,
    )
    reward_config = WinFirstRewardConfig(
        profile_id="win-time-score-v1",
        win_reward=10.0,
        failure_reward=-10.0,
        time_penalty_per_native_tick=-0.0001,
        score_progress_reward_cap=0.01,
    )
    rng = np.random.default_rng(model_seed + round_index * 1_000_003)
    aggregate: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    episode_rows: list[dict[str, Any]] = []
    label_counts: Counter[str] = Counter()
    execution_counts: Counter[str] = Counter()
    started = time.perf_counter()

    for batch_start in range(0, len(teacher_rows), parallel_envs):
        batch = list(teacher_rows[batch_start : batch_start + parallel_envs])
        factories = [
            legacy._make_env_factory(
                original_root=original_root,
                level_id=str(row["level_id"]),
                max_ticks=max_ticks,
                input_config=input_config,
                reward_config=reward_config,
            )
            for row in batch
        ]
        vector = SubprocVecEnv(factories, start_method="forkserver")
        try:
            first_seed = seed_base + round_index * len(teacher_rows) + batch_start
            vector.seed(first_seed)
            observations = vector.reset()
            specifications = vector.env_method("teacher_spec")
            observation_teachers = [
                _OBSERVATION_TEACHERS[
                    str(row["observation_teacher_policy_id"])
                ](legacy._teacher_spec(value))
                for row, value in zip(batch, specifications, strict=True)
            ]
            finished = np.zeros(len(batch), dtype=np.bool_)
            step_counts = np.zeros(len(batch), dtype=np.int64)
            fire_candidate_counts = np.zeros(len(batch), dtype=np.int64)
            nonfire_candidate_counts = np.zeros(len(batch), dtype=np.int64)
            fire_reservoirs: list[
                list[tuple[np.ndarray, np.ndarray, np.ndarray]]
            ] = [[] for _ in batch]
            nonfire_reservoirs: list[
                list[tuple[np.ndarray, np.ndarray, np.ndarray]]
            ] = [[] for _ in batch]
            last_infos: list[dict[str, Any]] = [{} for _ in batch]
            for _ in range(max_ticks + 1):
                masks = np.asarray(get_action_masks(vector), dtype=np.bool_)
                labels = np.zeros((len(batch), 2), dtype=np.int64)
                teacher_groups: dict[str, list[int]] = {}
                for position, row in enumerate(batch):
                    if not finished[position]:
                        teacher_groups.setdefault(
                            str(row["teacher_model"]["id"]), []
                        ).append(position)
                for model_id, positions in teacher_groups.items():
                    predicted, _ = teacher_models[model_id].predict(
                        observations[positions],
                        deterministic=True,
                        action_masks=masks[positions],
                    )
                    labels[positions] = np.asarray(
                        predicted, dtype=np.int64
                    ).reshape(len(positions), 2)

                for position, row in enumerate(batch):
                    if finished[position]:
                        continue
                    observation_action = np.asarray(
                        observation_teachers[position].act(observations[position]),
                        dtype=np.int64,
                    )
                    observation_verb = int(observation_action[0])
                    if not bool(masks[position, observation_verb]):
                        observation_action = np.asarray(
                            (0, int(observation_action[1])), dtype=np.int64
                        )
                    if (
                        int(row["engineering_wins"]) == 0
                        or int(observation_action[0]) == 3
                    ):
                        labels[position] = observation_action
                        label_counts[
                            str(row["observation_teacher_policy_id"])
                        ] += 1
                    else:
                        label_counts["specialist"] += 1
                    verb = int(labels[position, 0])
                    if not bool(masks[position, verb]):
                        raise RuntimeError(
                            f"teacher emitted masked verb {verb} for {row['level_id']}"
                        )
                    sample = (
                        np.asarray(observations[position], dtype=np.float32).copy(),
                        labels[position].copy(),
                        masks[position].copy(),
                    )
                    if verb == 1:
                        fire_candidate_counts[position] += 1
                        legacy._reservoir_add(
                            fire_reservoirs[position],
                            seen=int(fire_candidate_counts[position]),
                            capacity=fire_samples_per_level,
                            row=sample,
                            rng=rng,
                        )
                    elif step_counts[position] % sample_stride == 0 or verb in {2, 3}:
                        nonfire_candidate_counts[position] += 1
                        legacy._reservoir_add(
                            nonfire_reservoirs[position],
                            seen=int(nonfire_candidate_counts[position]),
                            capacity=samples_per_level,
                            row=sample,
                            rng=rng,
                        )

                if round_index == 0:
                    actions = labels.copy()
                    execution_counts["teacher"] += int(np.sum(~finished))
                else:
                    student_actions, _ = student.predict(
                        observations,
                        deterministic=True,
                        action_masks=masks,
                    )
                    student_actions = np.asarray(
                        student_actions, dtype=np.int64
                    ).reshape(len(batch), 2)
                    execute_teacher = (
                        rng.random(len(batch)) < teacher_execution_probability
                    )
                    execute_teacher |= finished
                    actions = np.where(
                        execute_teacher[:, None], labels, student_actions
                    )
                    execution_counts["teacher"] += int(
                        np.sum(execute_teacher & ~finished)
                    )
                    execution_counts["student"] += int(
                        np.sum(~execute_teacher & ~finished)
                    )

                observations, _, dones, infos = vector.step(actions)
                step_counts[~finished] += 1
                for position, (done, info) in enumerate(
                    zip(dones, infos, strict=True)
                ):
                    if finished[position]:
                        continue
                    last_infos[position] = dict(info)
                    if bool(done):
                        finished[position] = True
                if bool(np.all(finished)):
                    break
            if not bool(np.all(finished)):
                missing = [
                    batch[index]["level_id"]
                    for index in range(len(batch))
                    if not finished[index]
                ]
                raise RuntimeError(
                    f"balanced distillation episodes exceeded guard: {missing}"
                )
            for position, row in enumerate(batch):
                aggregate.extend(nonfire_reservoirs[position])
                aggregate.extend(fire_reservoirs[position])
                info = last_infos[position]
                episode_rows.append(
                    {
                        "level_id": row["level_id"],
                        "teacher_model_id": row["teacher_model"]["id"],
                        "teacher_proven": int(row["engineering_wins"]) > 0,
                        "observation_teacher_policy_id": row[
                            "observation_teacher_policy_id"
                        ],
                        "seed": first_seed + position,
                        "steps": int(step_counts[position]),
                        "outcome": info.get("outcome"),
                        "score": int(info.get("score", 0)),
                        "fire_candidate_samples": int(
                            fire_candidate_counts[position]
                        ),
                        "nonfire_candidate_samples": int(
                            nonfire_candidate_counts[position]
                        ),
                        "retained_fire_samples": len(
                            fire_reservoirs[position]
                        ),
                        "retained_nonfire_samples": len(
                            nonfire_reservoirs[position]
                        ),
                        "retained_samples": len(fire_reservoirs[position])
                        + len(nonfire_reservoirs[position]),
                    }
                )
        finally:
            vector.close()

    return aggregate, {
        "round_index": round_index,
        "wall_seconds": time.perf_counter() - started,
        "episodes": episode_rows,
        "wins": sum(row["outcome"] == "win" for row in episode_rows),
        "retained_samples": len(aggregate),
        "retained_fire_samples": sum(
            int(row["retained_fire_samples"]) for row in episode_rows
        ),
        "retained_nonfire_samples": sum(
            int(row["retained_nonfire_samples"]) for row in episode_rows
        ),
        "label_counts": dict(label_counts),
        "execution_counts": dict(execution_counts),
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
    original_root = args.original_root.expanduser().resolve(strict=True)
    prereg = _validate_preregistration(prereg_path)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "VALID",
                    "preregistration": str(prereg_path),
                    "sha256": legacy._sha256(prereg_path),
                    "run": prereg["run"],
                    "formal_seed_consumption": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    fire_samples = int(prereg["run"]["fire_samples_per_level_per_round"])

    def collect_bound(**kwargs: Any):
        return _collect_round_balanced(
            **kwargs, fire_samples_per_level=fire_samples
        )

    legacy._collect_round = collect_bound
    legacy.__file__ = str(SCRIPT_PATH)
    result = legacy.run_distillation(
        prereg_path=prereg_path,
        prereg=prereg,
        original_root=original_root,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "final_model": result["final_model"],
                "round_wins": [
                    row["collection"]["wins"] for row in result["rounds"]
                ],
                "retained_fire_samples": [
                    row["collection"]["retained_fire_samples"]
                    for row in result["rounds"]
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
