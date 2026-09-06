"""Distill per-level specialist teachers into one AlphaZuma 55 policy."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Sequence

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import numpy as np

from tools.pretrain_maskable_ppo_teacher import _train_batch
from zuma_rl.alphazuma_55 import INCLUDED_LEVELS, full55_environment_config
from zuma_rl.human_speedrun import (
    EliteHumanInputConfig,
    HumanSpeedrunWrapper,
    WinFirstRewardConfig,
)
from zuma_rl.revenge_core import SUPPORTED_PROFILE_MODE
from zuma_rl.revenge_env import RevengeEnv
from zuma_rl.revenge_teacher import (
    ActorObservableRevengeTeacher,
    RevengeTeacherSpec,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class TeacherSpecHumanSpeedrunWrapper(HumanSpeedrunWrapper):
    """Expose only the immutable actor-observable teacher specification."""

    def teacher_spec(self) -> dict[str, Any]:
        return asdict(RevengeTeacherSpec.from_env(self.env))


def _make_env_factory(
    *,
    original_root: Path,
    level_id: str,
    max_ticks: int,
    input_config: EliteHumanInputConfig,
    reward_config: WinFirstRewardConfig,
) -> Any:
    def make_env() -> TeacherSpecHumanSpeedrunWrapper:
        base = RevengeEnv(
            config=full55_environment_config(max_ticks=max_ticks),
            level_id=level_id,
            root=original_root,
            hard=False,
            curve_index=0,
            profile_mode=SUPPORTED_PROFILE_MODE,
        )
        return TeacherSpecHumanSpeedrunWrapper(
            base,
            input_config=input_config,
            reward_config=reward_config,
        )

    return make_env


def _teacher_spec(value: dict[str, Any]) -> RevengeTeacherSpec:
    value = dict(value)
    value["ball_feature_names"] = tuple(value["ball_feature_names"])
    value["global_feature_names"] = tuple(value["global_feature_names"])
    value["shooter_positions"] = tuple(
        tuple(float(coordinate) for coordinate in position)
        for position in value["shooter_positions"]
    )
    return RevengeTeacherSpec(**value)


def _validate_preregistration(path: Path) -> dict[str, Any]:
    prereg = _read_json(path)
    if prereg.get("schema") != "zuma-rl.alphazuma-55-distillation-preregistration":
        raise ValueError("unexpected distillation preregistration schema")
    if prereg.get("version") != 1 or prereg.get("status") != "FROZEN_BEFORE_TRAINING":
        raise ValueError("distillation preregistration is not frozen")
    trainer = prereg.get("trainer", {})
    if Path(str(trainer.get("path", ""))).resolve() != Path(__file__).resolve():
        raise ValueError("preregistration binds another distiller")
    if trainer.get("sha256") != _sha256(Path(__file__).resolve()):
        raise ValueError("distiller bytes differ from preregistration")
    for name, artifact in prereg.get("implementation", {}).items():
        artifact_path = Path(str(artifact["path"])).resolve(strict=True)
        if artifact.get("sha256") != _sha256(artifact_path):
            raise ValueError(f"implementation bytes differ: {name}")
    source = prereg["student_source_model"]
    source_path = Path(str(source["path"])).resolve(strict=True)
    if source["sha256"] != _sha256(source_path):
        raise ValueError("student source model bytes differ")
    teacher_map_path = Path(str(prereg["teacher_map"]["path"])).resolve(strict=True)
    if prereg["teacher_map"]["sha256"] != _sha256(teacher_map_path):
        raise ValueError("teacher map bytes differ")
    teacher_map = _read_json(teacher_map_path)
    if teacher_map.get("status") != "FROZEN":
        raise ValueError("teacher map is not frozen")
    frozen_levels = [row["level_id"] for row in teacher_map["levels"]]
    if frozen_levels != list(INCLUDED_LEVELS):
        raise ValueError("teacher map level order differs from campaign scope")
    for row in teacher_map["levels"]:
        model = row["teacher_model"]
        model_path = Path(str(model["path"])).resolve(strict=True)
        if model["sha256"] != _sha256(model_path):
            raise ValueError(f"teacher model bytes differ: {model['id']}")
    run = prereg["run"]
    run_dir = Path(str(run["run_dir"])).resolve()
    if run_dir.exists():
        raise FileExistsError(f"refusing to reuse distillation run: {run_dir}")
    positive = (
        "rounds",
        "samples_per_level_per_round",
        "sample_stride",
        "batch_size",
        "epochs_per_round",
        "parallel_envs",
        "max_ticks",
    )
    if any(int(run[name]) < 1 for name in positive):
        raise ValueError("distillation integer hyperparameters must be positive")
    if not 0.0 <= float(run["teacher_execution_probability_after_round_zero"]) <= 1.0:
        raise ValueError("teacher execution probability is outside [0, 1]")
    if float(run["learning_rate"]) <= 0.0:
        raise ValueError("learning rate must be positive")
    return prereg


def _reservoir_add(
    reservoir: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    seen: int,
    capacity: int,
    row: tuple[np.ndarray, np.ndarray, np.ndarray],
    rng: np.random.Generator,
) -> None:
    if len(reservoir) < capacity:
        reservoir.append(row)
        return
    replacement = int(rng.integers(0, seen))
    if replacement < capacity:
        reservoir[replacement] = row


def _collect_round(
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
    sample_stride: int,
    teacher_execution_probability: float,
    model_seed: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray]], dict[str, Any]]:
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
            _make_env_factory(
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
            geometric = [
                ActorObservableRevengeTeacher(_teacher_spec(value))
                for value in specifications
            ]
            finished = np.zeros(len(batch), dtype=np.bool_)
            step_counts = np.zeros(len(batch), dtype=np.int64)
            candidate_counts = np.zeros(len(batch), dtype=np.int64)
            reservoirs: list[
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
                            str(row["teacher_model"]["id"]),
                            [],
                        ).append(position)
                for model_id, positions in teacher_groups.items():
                    predicted, _ = teacher_models[model_id].predict(
                        observations[positions],
                        deterministic=True,
                        action_masks=masks[positions],
                    )
                    labels[positions] = np.asarray(predicted, dtype=np.int64).reshape(
                        len(positions),
                        2,
                    )

                for position, row in enumerate(batch):
                    if finished[position]:
                        continue
                    geometric_action = np.asarray(
                        geometric[position].act(observations[position]),
                        dtype=np.int64,
                    )
                    geometric_verb = int(geometric_action[0])
                    if not bool(masks[position, geometric_verb]):
                        # The geometric teacher sees the actor observation but
                        # not the wrapper's reaction/button cooldown state.
                        # A temporarily blocked request therefore becomes wait,
                        # exactly as the canonical input wrapper would enforce.
                        geometric_action = np.asarray(
                            (0, int(geometric_action[1])),
                            dtype=np.int64,
                        )
                    if (
                        int(row["engineering_wins"]) == 0
                        or int(geometric_action[0]) == 3
                    ):
                        labels[position] = geometric_action
                        label_counts["geometric"] += 1
                    else:
                        label_counts["specialist"] += 1
                    verb = int(labels[position, 0])
                    if not bool(masks[position, verb]):
                        raise RuntimeError(
                            f"teacher emitted masked verb {verb} for {row['level_id']}"
                        )
                    include = (
                        step_counts[position] % sample_stride == 0
                        or verb in {2, 3}
                    )
                    if include:
                        candidate_counts[position] += 1
                        _reservoir_add(
                            reservoirs[position],
                            seen=int(candidate_counts[position]),
                            capacity=samples_per_level,
                            row=(
                                np.asarray(
                                    observations[position],
                                    dtype=np.float32,
                                ).copy(),
                                labels[position].copy(),
                                masks[position].copy(),
                            ),
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
                        student_actions,
                        dtype=np.int64,
                    ).reshape(len(batch), 2)
                    execute_teacher = rng.random(len(batch)) < teacher_execution_probability
                    execute_teacher |= finished
                    actions = np.where(
                        execute_teacher[:, None],
                        labels,
                        student_actions,
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
                raise RuntimeError(f"distillation episodes exceeded guard: {missing}")
            for position, row in enumerate(batch):
                aggregate.extend(reservoirs[position])
                info = last_infos[position]
                episode_rows.append(
                    {
                        "level_id": row["level_id"],
                        "teacher_model_id": row["teacher_model"]["id"],
                        "teacher_proven": int(row["engineering_wins"]) > 0,
                        "seed": first_seed + position,
                        "steps": int(step_counts[position]),
                        "outcome": info.get("outcome"),
                        "score": int(info.get("score", 0)),
                        "candidate_samples": int(candidate_counts[position]),
                        "retained_samples": len(reservoirs[position]),
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
        "label_counts": dict(label_counts),
        "execution_counts": dict(execution_counts),
    }


def _optimize(
    *,
    model: Any,
    dataset: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    epochs: int,
    batch_size: int,
    max_grad_norm: float,
    verb_loss_weight: float,
    model_seed: int,
    round_index: int,
) -> dict[str, Any]:
    if not dataset:
        raise RuntimeError("distillation dataset is empty")
    rng = np.random.default_rng(model_seed + 7_919 * (round_index + 1))
    rows: list[dict[str, Any]] = []
    for epoch in range(epochs):
        permutation = rng.permutation(len(dataset))
        for batch_index, start in enumerate(range(0, len(dataset), batch_size)):
            indices = permutation[start : start + batch_size]
            batch = [dataset[int(index)] for index in indices]
            metrics = _train_batch(
                model=model,
                observations=[row[0] for row in batch],
                actions=[row[1] for row in batch],
                masks=[row[2] for row in batch],
                max_grad_norm=max_grad_norm,
                verb_loss_weight=verb_loss_weight,
                aim_loss_scope="fire_only",
                aim_loss_mode="circular_smoothed",
                aim_smoothing_sigma=1.5,
                aim_distance_weight=0.5,
            )
            metrics = {
                key: (
                    None
                    if isinstance(value, float) and not math.isfinite(value)
                    else value
                )
                for key, value in metrics.items()
            }
            metrics.update(
                {
                    "round_index": round_index,
                    "epoch": epoch,
                    "batch_index": batch_index,
                }
            )
            rows.append(metrics)
    return {
        "updates": len(rows),
        "samples": len(dataset),
        "first_batch": rows[0],
        "last_batch": rows[-1],
        "mean_loss": float(np.mean([row["loss"] for row in rows])),
        "mean_verb_accuracy": float(
            np.mean([row["verb_accuracy"] for row in rows])
        ),
    }


def run_distillation(
    *,
    prereg_path: Path,
    prereg: dict[str, Any],
    original_root: Path,
) -> dict[str, Any]:
    import torch
    from sb3_contrib import MaskablePPO

    torch.set_num_threads(1)
    run = prereg["run"]
    device = str(run["device"])
    run_dir = Path(str(run["run_dir"])).resolve()
    run_dir.mkdir(parents=True)
    started = time.perf_counter()
    teacher_map_path = Path(str(prereg["teacher_map"]["path"])).resolve(strict=True)
    teacher_map = _read_json(teacher_map_path)
    teacher_rows = list(teacher_map["levels"])
    source_path = Path(str(prereg["student_source_model"]["path"])).resolve(
        strict=True
    )
    config_receipt = {
        "schema": "zuma-rl.alphazuma-55-distillation-config",
        "version": 1,
        "status": "STARTED",
        "started_utc": _utc_now(),
        "preregistration": {
            "path": str(prereg_path),
            "sha256": _sha256(prereg_path),
        },
        "trainer_sha256": _sha256(Path(__file__).resolve()),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": device,
            "cuda_device": torch.cuda.get_device_name(torch.device(device)),
        },
        "run": run,
    }
    _write_json_atomic(run_dir / "config.json", config_receipt)

    student: Any | None = None
    teacher_models: dict[str, Any] = {}
    try:
        student = MaskablePPO.load(source_path, device=device)
        if tuple(int(value) for value in student.observation_space.shape) != (22833,):
            raise ValueError("student observation space is not full55-v1")
        if tuple(int(value) for value in student.action_space.nvec) != (4, 180):
            raise ValueError("student action space is not full55-v1")
        optimizer_class = student.policy.optimizer_class
        optimizer_kwargs = dict(student.policy.optimizer_kwargs)
        student.policy.optimizer = optimizer_class(
            student.policy.parameters(),
            lr=float(run["learning_rate"]),
            **optimizer_kwargs,
        )

        unique_models = {
            row["teacher_model"]["id"]: row["teacher_model"]
            for row in teacher_rows
        }
        for model_id, spec in unique_models.items():
            teacher_models[model_id] = MaskablePPO.load(
                spec["path"],
                device=device,
            )

        aggregate: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        rounds: list[dict[str, Any]] = []
        for round_index in range(int(run["rounds"])):
            collected, collection = _collect_round(
                original_root=original_root,
                teacher_rows=teacher_rows,
                teacher_models=teacher_models,
                student=student,
                device=device,
                round_index=round_index,
                seed_base=int(run["training_seed_base"]),
                parallel_envs=int(run["parallel_envs"]),
                max_ticks=int(run["max_ticks"]),
                samples_per_level=int(run["samples_per_level_per_round"]),
                sample_stride=int(run["sample_stride"]),
                teacher_execution_probability=(
                    1.0
                    if round_index == 0
                    else float(
                        run[
                            "teacher_execution_probability_after_round_zero"
                        ]
                    )
                ),
                model_seed=int(run["model_seed"]),
            )
            aggregate.extend(collected)
            optimization = _optimize(
                model=student,
                dataset=aggregate,
                epochs=int(run["epochs_per_round"]),
                batch_size=int(run["batch_size"]),
                max_grad_norm=float(run["max_grad_norm"]),
                verb_loss_weight=float(run["verb_loss_weight"]),
                model_seed=int(run["model_seed"]),
                round_index=round_index,
            )
            row = {
                "collection": collection,
                "aggregate_samples": len(aggregate),
                "optimization": optimization,
            }
            rounds.append(row)
            _write_json_atomic(
                run_dir / "training_status.json",
                {
                    "schema": "zuma-rl.alphazuma-55-distillation-status",
                    "version": 1,
                    "status": "RUNNING",
                    "updated_utc": _utc_now(),
                    "completed_rounds": round_index + 1,
                    "expected_rounds": int(run["rounds"]),
                    "aggregate_samples": len(aggregate),
                    "rounds": rounds,
                    "wall_seconds": time.perf_counter() - started,
                },
            )

        student.num_timesteps = int(student.num_timesteps) + sum(
            sum(int(episode["steps"]) for episode in row["collection"]["episodes"])
            for row in rounds
        )
        final_path = run_dir / "final_model.zip"
        student.save(final_path)
        completion = {
            "schema": "zuma-rl.alphazuma-55-distillation-completion",
            "version": 1,
            "status": "COMPLETE",
            "completed_utc": _utc_now(),
            "wall_seconds": time.perf_counter() - started,
            "source_model": prereg["student_source_model"],
            "teacher_map": prereg["teacher_map"],
            "rounds": rounds,
            "final_model": {
                "path": str(final_path),
                "sha256": _sha256(final_path),
                "bytes": final_path.stat().st_size,
                "num_timesteps": int(student.num_timesteps),
            },
        }
        _write_json_atomic(run_dir / "completion.json", completion)
        return completion
    except BaseException as error:
        failure = {
            "schema": "zuma-rl.alphazuma-55-distillation-failure",
            "version": 1,
            "status": "FAILED",
            "failed_utc": _utc_now(),
            "wall_seconds": time.perf_counter() - started,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        _write_json_atomic(run_dir / "failure.json", failure)
        raise
    finally:
        teacher_models.clear()
        student = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


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
                    "sha256": _sha256(prereg_path),
                    "levels": len(INCLUDED_LEVELS),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    completion = run_distillation(
        prereg_path=prereg_path,
        prereg=prereg,
        original_root=original_root,
    )
    print(json.dumps(completion, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
