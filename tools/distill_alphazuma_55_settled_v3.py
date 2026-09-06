"""Distill the mask-aware settled V3 teacher into one AlphaZuma 55 CNN."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Callable, Sequence

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import numpy as np

from tools import distill_alphazuma_55 as legacy
from tools.pretrain_maskable_ppo_teacher import _train_batch
from zuma_rl.alphazuma_55 import INCLUDED_LEVELS
from zuma_rl.human_speedrun import EliteHumanInputConfig, WinFirstRewardConfig
from zuma_rl.revenge_settled_strategic_teacher_v3 import (
    CurveAwareSettledStrategicRevengeTeacher,
)


SCRIPT_PATH = Path(__file__).resolve()
POLICY_ID = "curve-aware-settled-strategic-masked-v3"
VERB_NAMES = ("wait", "fire", "swap", "hop")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_probe_result(
    artifact: dict[str, Any],
    *,
    expected_policy_id: str,
) -> dict[str, Any]:
    path = Path(str(artifact["path"])).resolve(strict=True)
    if artifact.get("sha256") != legacy._sha256(path):
        raise ValueError(f"teacher probe bytes differ: {path}")
    result = legacy._read_json(path)
    summary = result.get("summary", {})
    if (
        result.get("status") != "COMPLETE"
        or result.get("formal_seed_consumption") is not False
        or result.get("teacher", {}).get("policy_id") != expected_policy_id
        or int(summary.get("attempts", -1)) != 55
        or int(summary.get("wins", -1)) != 55
        or int(summary.get("level_coverage", -1)) != 55
        or int(summary.get("losses", -1)) != 0
        or int(summary.get("truncations", -1)) != 0
        or int(summary.get("capacity_overflows", -1)) != 0
    ):
        raise ValueError(f"teacher probe is not a clean 55/55 result: {path}")
    return result


def _validate_preregistration(path: Path) -> dict[str, Any]:
    prereg = legacy._read_json(path)
    if (
        prereg.get("schema")
        != "zuma-rl.alphazuma-55-settled-v3-distillation-preregistration"
        or prereg.get("version") != 1
        or prereg.get("status") != "FROZEN_BEFORE_TRAINING"
    ):
        raise ValueError("unexpected settled-V3 distillation preregistration")
    trainer = prereg.get("trainer", {})
    if Path(str(trainer.get("path", ""))).resolve() != SCRIPT_PATH:
        raise ValueError("preregistration binds another settled-V3 distiller")
    if trainer.get("sha256") != legacy._sha256(SCRIPT_PATH):
        raise ValueError("settled-V3 distiller bytes differ from preregistration")
    master = prereg.get("master_preregistration", {})
    master_path = Path(str(master.get("path", ""))).resolve(strict=True)
    if master.get("sha256") != legacy._sha256(master_path):
        raise ValueError("master preregistration bytes differ")
    for name, artifact in prereg.get("implementation", {}).items():
        artifact_path = Path(str(artifact["path"])).resolve(strict=True)
        if artifact.get("sha256") != legacy._sha256(artifact_path):
            raise ValueError(f"implementation bytes differ: {name}")
    source = prereg["student_source_model"]
    source_path = Path(str(source["path"])).resolve(strict=True)
    if source.get("sha256") != legacy._sha256(source_path):
        raise ValueError("student source model bytes differ")
    migration_artifact = prereg["source_migration_receipt"]
    migration_path = Path(str(migration_artifact["path"])).resolve(strict=True)
    if migration_artifact.get("sha256") != legacy._sha256(migration_path):
        raise ValueError("source migration receipt bytes differ")
    migration = legacy._read_json(migration_path)
    migrated = migration.get("migrated_model", {})
    if (
        migration.get("status") != "PASS"
        or migrated.get("sha256") != source.get("sha256")
        or Path(str(migrated.get("path", ""))).resolve() != source_path
        or int(migrated.get("num_timesteps", -1))
        != int(source.get("training_steps", -2))
    ):
        raise ValueError("source migration receipt does not bind the student")
    evidence = prereg["teacher_evidence"]
    _validate_probe_result(
        evidence["unmasked_fresh_55"],
        expected_policy_id="curve-aware-settled-strategic-actor-observable-v3",
    )
    _validate_probe_result(
        evidence["exact_mask_fresh_55"],
        expected_policy_id=POLICY_ID,
    )
    levels = prereg.get("levels")
    if levels != list(INCLUDED_LEVELS):
        raise ValueError("settled-V3 distillation scope differs from AlphaZuma 55")
    run = prereg["run"]
    run_dir = Path(str(run["run_dir"])).resolve()
    if run_dir.exists():
        raise FileExistsError(f"refusing to reuse settled-V3 run: {run_dir}")
    positive = (
        "rounds",
        "parallel_envs",
        "max_ticks",
        "batch_size",
        "epochs_per_round",
        "checkpoint_interval_epochs",
    )
    if any(int(run[name]) < 1 for name in positive):
        raise ValueError("settled-V3 integer hyperparameters must be positive")
    if int(run["rounds"]) != 1:
        raise ValueError("this frozen settled-V3 distiller supports one round")
    if not 1 <= int(run["parallel_envs"]) <= 24:
        raise ValueError("parallel_envs must be in [1, 24]")
    if int(run["max_ticks"]) != 30_000:
        raise ValueError("settled-V3 training must retain the 300-second horizon")
    capacities = run.get("reservoir_capacity_by_verb", {})
    strides = run.get("sample_stride_by_verb", {})
    fractions = run.get("target_batch_fraction_by_verb", {})
    if tuple(capacities) != VERB_NAMES or tuple(strides) != VERB_NAMES:
        raise ValueError("verb reservoir keys differ from the actor interface")
    if tuple(fractions) != VERB_NAMES:
        raise ValueError("verb batch-fraction keys differ from the actor interface")
    if any(int(capacities[name]) < 1 for name in VERB_NAMES):
        raise ValueError("verb reservoir capacities must be positive")
    if any(int(strides[name]) < 1 for name in VERB_NAMES):
        raise ValueError("verb sample strides must be positive")
    if any(float(fractions[name]) < 0.0 for name in VERB_NAMES):
        raise ValueError("verb batch fractions cannot be negative")
    if not math.isclose(
        sum(float(fractions[name]) for name in VERB_NAMES),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("verb batch fractions must sum to one")
    if run.get("aim_loss_scope") != "all_actions":
        raise ValueError("settled waits require all-actions aim supervision")
    if run.get("aim_loss_mode") != "circular_smoothed":
        raise ValueError("unexpected settled-V3 aim loss")
    if float(run["learning_rate"]) <= 0.0:
        raise ValueError("learning rate must be positive")
    seeds = [int(run["training_seed_base"]) + index for index in range(55)]
    if int(run["training_seed_last_consumed"]) != seeds[-1]:
        raise ValueError("training seed receipt is inconsistent")
    registry = legacy._read_json(master_path)["seed_registry"]["training"]
    if not int(registry["first"]) <= seeds[0] <= seeds[-1] <= int(
        registry["last"]
    ):
        raise ValueError("settled-V3 seeds escaped the training registry")
    return prereg


def _effective_label(
    raw_action: np.ndarray,
    action_mask: np.ndarray,
) -> tuple[np.ndarray, bool]:
    action = np.asarray(raw_action, dtype=np.int64).reshape(2).copy()
    mask = np.asarray(action_mask, dtype=np.bool_).reshape(-1)
    verb = int(action[0])
    aim = int(action[1])
    if not 0 <= verb < len(VERB_NAMES) or not 0 <= aim < 180:
        raise ValueError(f"teacher action is outside the actor interface: {action}")
    if len(mask) != len(VERB_NAMES) + 180:
        raise ValueError(f"unexpected flattened action-mask size: {len(mask)}")
    if not bool(mask[len(VERB_NAMES) + aim]):
        raise RuntimeError(f"teacher emitted masked aim bin {aim}")
    forced_wait = not bool(mask[verb])
    if forced_wait:
        if not bool(mask[0]):
            raise RuntimeError("wait fallback is unexpectedly masked")
        action[0] = 0
    return action, forced_wait


def _reservoir_consider(
    reservoir: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    seen: int,
    capacity: int,
    observation: np.ndarray,
    action: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
) -> None:
    if len(reservoir) < capacity:
        index = len(reservoir)
    else:
        index = int(rng.integers(0, seen))
        if index >= capacity:
            return
    row = (
        np.asarray(observation, dtype=np.float32).copy(),
        np.asarray(action, dtype=np.int64).copy(),
        np.asarray(mask, dtype=np.bool_).copy(),
    )
    if index == len(reservoir):
        reservoir.append(row)
    else:
        reservoir[index] = row


def _collect_round(
    *,
    original_root: Path,
    level_ids: Sequence[str],
    round_index: int,
    seed_base: int,
    parallel_envs: int,
    max_ticks: int,
    capacities: dict[str, int],
    strides: dict[str, int],
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
    episodes: list[dict[str, Any]] = []
    total_raw: Counter[str] = Counter()
    total_effective: Counter[str] = Counter()
    forced_waits = 0
    started = time.perf_counter()

    for batch_start in range(0, len(level_ids), parallel_envs):
        batch = list(level_ids[batch_start : batch_start + parallel_envs])
        factories = [
            legacy._make_env_factory(
                original_root=original_root,
                level_id=level_id,
                max_ticks=max_ticks,
                input_config=input_config,
                reward_config=reward_config,
            )
            for level_id in batch
        ]
        vector = SubprocVecEnv(factories, start_method="forkserver")
        try:
            first_seed = seed_base + round_index * len(level_ids) + batch_start
            vector.seed(first_seed)
            observations = vector.reset()
            specifications = vector.env_method("teacher_spec")
            teachers = [
                CurveAwareSettledStrategicRevengeTeacher(
                    legacy._teacher_spec(value)
                )
                for value in specifications
            ]
            finished = np.zeros(len(batch), dtype=np.bool_)
            steps = np.zeros(len(batch), dtype=np.int64)
            raw_counts = [Counter() for _ in batch]
            effective_counts = [Counter() for _ in batch]
            candidate_counts = [Counter() for _ in batch]
            retained: list[
                dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]]
            ] = [{name: [] for name in VERB_NAMES} for _ in batch]
            per_episode_forced_waits = np.zeros(len(batch), dtype=np.int64)
            last_infos: list[dict[str, Any]] = [{} for _ in batch]

            for _ in range(max_ticks + 1):
                masks = np.asarray(get_action_masks(vector), dtype=np.bool_)
                actions = np.zeros((len(batch), 2), dtype=np.int64)
                active = ~finished
                for position in range(len(batch)):
                    if finished[position]:
                        continue
                    raw = np.asarray(
                        teachers[position].act(observations[position]),
                        dtype=np.int64,
                    )
                    raw_name = VERB_NAMES[int(raw[0])]
                    raw_counts[position][raw_name] += 1
                    total_raw[raw_name] += 1
                    action, forced = _effective_label(raw, masks[position])
                    actions[position] = action
                    name = VERB_NAMES[int(action[0])]
                    effective_counts[position][name] += 1
                    total_effective[name] += 1
                    if forced:
                        per_episode_forced_waits[position] += 1
                        forced_waits += 1
                    stride = int(strides[name])
                    if int(steps[position]) % stride != 0:
                        continue
                    candidate_counts[position][name] += 1
                    _reservoir_consider(
                        retained[position][name],
                        seen=int(candidate_counts[position][name]),
                        capacity=int(capacities[name]),
                        observation=observations[position],
                        action=action,
                        mask=masks[position],
                        rng=rng,
                    )

                observations, _, dones, infos = vector.step(actions)
                steps[active] += 1
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
                    batch[index]
                    for index in range(len(batch))
                    if not finished[index]
                ]
                raise RuntimeError(
                    f"settled-V3 collection exceeded the episode guard: {missing}"
                )
            for position, level_id in enumerate(batch):
                retained_counts: dict[str, int] = {}
                for name in VERB_NAMES:
                    aggregate.extend(retained[position][name])
                    retained_counts[name] = len(retained[position][name])
                info = last_infos[position]
                episodes.append(
                    {
                        "level_id": level_id,
                        "seed": first_seed + position,
                        "steps": int(steps[position]),
                        "outcome": info.get("outcome"),
                        "time_limit_truncated": bool(
                            info.get("TimeLimit.truncated", False)
                        ),
                        "score": int(info.get("score", 0)),
                        "raw_action_counts": dict(raw_counts[position]),
                        "effective_action_counts": dict(
                            effective_counts[position]
                        ),
                        "mask_forced_waits": int(
                            per_episode_forced_waits[position]
                        ),
                        "candidate_samples_by_verb": dict(
                            candidate_counts[position]
                        ),
                        "retained_samples_by_verb": retained_counts,
                        "retained_samples": sum(retained_counts.values()),
                        "observation_capacity_overflow": bool(
                            info.get("observation_capacity_overflow", False)
                        ),
                    }
                )
        finally:
            vector.close()

    retained_by_verb = Counter(
        VERB_NAMES[int(row[1][0])] for row in aggregate
    )
    return aggregate, {
        "round_index": round_index,
        "wall_seconds": time.perf_counter() - started,
        "episodes": episodes,
        "wins": sum(row["outcome"] == "win" for row in episodes),
        "losses": sum(row["outcome"] == "loss" for row in episodes),
        "truncations": sum(row["time_limit_truncated"] for row in episodes),
        "capacity_overflows": sum(
            row["observation_capacity_overflow"] for row in episodes
        ),
        "retained_samples": len(aggregate),
        "retained_samples_by_verb": dict(retained_by_verb),
        "raw_action_counts": dict(total_raw),
        "effective_action_counts": dict(total_effective),
        "mask_forced_waits": forced_waits,
        "teacher_policy_id": POLICY_ID,
    }


def _verb_balanced_indices(
    *,
    dataset: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    batch_size: int,
    fractions: dict[str, float],
    rng: np.random.Generator,
    groups: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    if groups is None:
        groups = {
            name: np.asarray(
                [
                    index
                    for index, row in enumerate(dataset)
                    if int(row[1][0]) == verb
                ],
                dtype=np.int64,
            )
            for verb, name in enumerate(VERB_NAMES)
        }
    present = [name for name in VERB_NAMES if len(groups[name]) > 0]
    if not present:
        raise RuntimeError("distillation dataset has no action labels")
    total_fraction = sum(float(fractions[name]) for name in present)
    probabilities = np.asarray(
        [float(fractions[name]) / total_fraction for name in present],
        dtype=np.float64,
    )
    expected = probabilities * batch_size
    counts = np.floor(expected).astype(np.int64)
    for index in np.argsort(-(expected - counts))[: batch_size - int(counts.sum())]:
        counts[int(index)] += 1
    chosen: list[np.ndarray] = []
    for name, count in zip(present, counts, strict=True):
        if int(count) > 0:
            chosen.append(
                rng.choice(groups[name], size=int(count), replace=True)
            )
    result = np.concatenate(chosen)
    rng.shuffle(result)
    return result


def _finite_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (
            None
            if isinstance(value, float) and not math.isfinite(value)
            else value
        )
        for key, value in metrics.items()
    }


def _optimize(
    *,
    model: Any,
    dataset: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    run: dict[str, Any],
    run_dir: Path,
    status_callback: Callable[[int, list[dict[str, Any]], list[dict[str, Any]]], None],
) -> dict[str, Any]:
    if not dataset:
        raise RuntimeError("settled-V3 distillation dataset is empty")
    rng = np.random.default_rng(int(run["model_seed"]) + 7_919)
    batch_size = int(run["batch_size"])
    updates_per_epoch = math.ceil(len(dataset) / batch_size)
    groups = {
        name: np.asarray(
            [
                index
                for index, row in enumerate(dataset)
                if int(row[1][0]) == verb
            ],
            dtype=np.int64,
        )
        for verb, name in enumerate(VERB_NAMES)
    }
    epoch_rows: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    for epoch in range(int(run["epochs_per_round"])):
        batches: list[dict[str, Any]] = []
        for batch_index in range(updates_per_epoch):
            indices = _verb_balanced_indices(
                dataset=dataset,
                batch_size=batch_size,
                fractions={
                    name: float(run["target_batch_fraction_by_verb"][name])
                    for name in VERB_NAMES
                },
                rng=rng,
                groups=groups,
            )
            batch = [dataset[int(index)] for index in indices]
            metrics = _train_batch(
                model=model,
                observations=[row[0] for row in batch],
                actions=[row[1] for row in batch],
                masks=[row[2] for row in batch],
                max_grad_norm=float(run["max_grad_norm"]),
                verb_loss_weight=float(run["verb_loss_weight"]),
                aim_loss_scope=str(run["aim_loss_scope"]),
                aim_loss_mode=str(run["aim_loss_mode"]),
                aim_smoothing_sigma=float(run["aim_smoothing_sigma"]),
                aim_distance_weight=float(run["aim_distance_weight"]),
            )
            metrics = _finite_metrics(metrics)
            metrics["batch_index"] = batch_index
            batches.append(metrics)
        epoch_row = {
            "epoch": epoch + 1,
            "updates": len(batches),
            "mean_loss": float(np.mean([row["loss"] for row in batches])),
            "mean_verb_accuracy": float(
                np.mean([row["verb_accuracy"] for row in batches])
            ),
            "mean_aim_exact_accuracy": float(
                np.mean([row["aim_exact_accuracy"] for row in batches])
            ),
            "mean_aim_within_one_accuracy": float(
                np.mean([row["aim_within_one_accuracy"] for row in batches])
            ),
            "last_batch": batches[-1],
        }
        epoch_rows.append(epoch_row)
        if (epoch + 1) % int(run["checkpoint_interval_epochs"]) == 0:
            checkpoint = run_dir / f"epoch_{epoch + 1:02d}_model.zip"
            model.save(checkpoint)
            checkpoints.append(
                {
                    "epoch": epoch + 1,
                    "path": str(checkpoint),
                    "sha256": legacy._sha256(checkpoint),
                    "bytes": checkpoint.stat().st_size,
                }
            )
        status_callback(epoch + 1, epoch_rows, checkpoints)
    return {
        "updates": sum(int(row["updates"]) for row in epoch_rows),
        "samples": len(dataset),
        "updates_per_epoch": updates_per_epoch,
        "epochs": epoch_rows,
        "checkpoints": checkpoints,
        "sampling": "fixed_target_fraction_with_replacement_per_batch",
        "aim_loss_scope": run["aim_loss_scope"],
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
    source_path = Path(str(prereg["student_source_model"]["path"])).resolve(
        strict=True
    )
    legacy._write_json_atomic(
        run_dir / "config.json",
        {
            "schema": "zuma-rl.alphazuma-55-settled-v3-distillation-config",
            "version": 1,
            "status": "STARTED",
            "started_utc": _utc_now(),
            "preregistration": {
                "path": str(prereg_path),
                "sha256": legacy._sha256(prereg_path),
            },
            "trainer_sha256": legacy._sha256(SCRIPT_PATH),
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "torch": torch.__version__,
                "device": device,
                "cuda_device": torch.cuda.get_device_name(torch.device(device)),
            },
            "run": run,
        },
    )

    student: Any | None = None
    try:
        student = MaskablePPO.load(source_path, device=device)
        if tuple(int(value) for value in student.observation_space.shape) != (
            22_833,
        ):
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

        dataset, collection = _collect_round(
            original_root=original_root,
            level_ids=list(INCLUDED_LEVELS),
            round_index=0,
            seed_base=int(run["training_seed_base"]),
            parallel_envs=int(run["parallel_envs"]),
            max_ticks=int(run["max_ticks"]),
            capacities={
                name: int(run["reservoir_capacity_by_verb"][name])
                for name in VERB_NAMES
            },
            strides={
                name: int(run["sample_stride_by_verb"][name])
                for name in VERB_NAMES
            },
            model_seed=int(run["model_seed"]),
        )

        def write_status(
            completed_epochs: int,
            epoch_rows: list[dict[str, Any]],
            checkpoints: list[dict[str, Any]],
        ) -> None:
            legacy._write_json_atomic(
                run_dir / "training_status.json",
                {
                    "schema": "zuma-rl.alphazuma-55-settled-v3-distillation-status",
                    "version": 1,
                    "status": "RUNNING",
                    "stage": "OPTIMIZING",
                    "updated_utc": _utc_now(),
                    "collection": collection,
                    "aggregate_samples": len(dataset),
                    "completed_epochs": completed_epochs,
                    "expected_epochs": int(run["epochs_per_round"]),
                    "epochs": epoch_rows,
                    "checkpoints": checkpoints,
                    "wall_seconds": time.perf_counter() - started,
                },
            )

        write_status(0, [], [])
        optimization = _optimize(
            model=student,
            dataset=dataset,
            run=run,
            run_dir=run_dir,
            status_callback=write_status,
        )
        student.num_timesteps = int(student.num_timesteps) + sum(
            int(row["steps"]) for row in collection["episodes"]
        )
        final_path = run_dir / "final_model.zip"
        student.save(final_path)
        completion = {
            "schema": "zuma-rl.alphazuma-55-settled-v3-distillation-completion",
            "version": 1,
            "status": "COMPLETE",
            "completed_utc": _utc_now(),
            "wall_seconds": time.perf_counter() - started,
            "source_model": prereg["student_source_model"],
            "teacher_policy_id": POLICY_ID,
            "collection": collection,
            "optimization": optimization,
            "final_model": {
                "path": str(final_path),
                "sha256": legacy._sha256(final_path),
                "bytes": final_path.stat().st_size,
                "num_timesteps": int(student.num_timesteps),
            },
            "formal_seed_consumption": False,
        }
        legacy._write_json_atomic(run_dir / "completion.json", completion)
        return completion
    except BaseException as error:
        legacy._write_json_atomic(
            run_dir / "failure.json",
            {
                "schema": "zuma-rl.alphazuma-55-settled-v3-distillation-failure",
                "version": 1,
                "status": "FAILED",
                "failed_utc": _utc_now(),
                "wall_seconds": time.perf_counter() - started,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise
    finally:
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
                    "sha256": legacy._sha256(prereg_path),
                    "levels": len(INCLUDED_LEVELS),
                    "run": prereg["run"],
                    "formal_seed_consumption": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    result = run_distillation(
        prereg_path=prereg_path,
        prereg=prereg,
        original_root=original_root,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "teacher_wins": result["collection"]["wins"],
                "retained_samples": result["collection"]["retained_samples"],
                "final_model": result["final_model"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
