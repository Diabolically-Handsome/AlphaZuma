"""Continue a full55 polar policy with frozen, mask-aware DAgger rounds."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
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
from tools import distill_alphazuma_55_polar_v1 as polar
from tools import distill_alphazuma_55_settled_v3 as settled
from zuma_rl.alphazuma_55 import INCLUDED_LEVELS, full55_environment_config
from zuma_rl.human_speedrun import EliteHumanInputConfig, WinFirstRewardConfig
from zuma_rl.revenge_core import SUPPORTED_PROFILE_MODE
from zuma_rl.revenge_env import RevengeEnv
from zuma_rl.revenge_settled_strategic_teacher_v3 import (
    CurveAwareSettledStrategicRevengeTeacher,
)


SCRIPT_PATH = Path(__file__).resolve()
POLICY_ID = "curve-aware-settled-strategic-masked-v3"
EXPECTED_OBSERVATION_SHAPE = (22_833,)
EXPECTED_ACTION_NVEC = (4, 180)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _write_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _validate_probability_schedule(values: Sequence[Any]) -> tuple[float, ...]:
    schedule = tuple(float(value) for value in values)
    if not schedule or schedule[0] != 1.0:
        raise ValueError("DAgger schedule must begin with a teacher-only round")
    if any(not 0.0 <= value <= 1.0 for value in schedule):
        raise ValueError("DAgger execution probabilities must be in [0, 1]")
    if any(right > left for left, right in zip(schedule, schedule[1:])):
        raise ValueError("DAgger teacher probability must be non-increasing")
    if schedule[-1] != 0.0:
        raise ValueError("DAgger schedule must finish with a student-only round")
    return schedule


def _mix_actions(
    *,
    teacher_actions: np.ndarray,
    student_actions: np.ndarray,
    active: np.ndarray,
    teacher_probability: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    teacher = np.asarray(teacher_actions, dtype=np.int64)
    student = np.asarray(student_actions, dtype=np.int64)
    active_array = np.asarray(active, dtype=np.bool_)
    if teacher.shape != student.shape or teacher.ndim != 2 or teacher.shape[1] != 2:
        raise ValueError("teacher and student actions must be matching Nx2 arrays")
    if active_array.shape != (teacher.shape[0],):
        raise ValueError("active mask shape does not match the action batch")
    probability = float(teacher_probability)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("teacher probability is outside [0, 1]")
    if probability == 1.0:
        execute_teacher = np.ones(teacher.shape[0], dtype=np.bool_)
    elif probability == 0.0:
        execute_teacher = np.zeros(teacher.shape[0], dtype=np.bool_)
    else:
        execute_teacher = rng.random(teacher.shape[0]) < probability
    # Completed vector slots auto-reset under SB3.  Keep those ignored slots on
    # the teacher/wait side so they cannot contaminate student-execution counts.
    execute_teacher |= ~active_array
    actions = np.where(execute_teacher[:, None], teacher, student)
    return actions, execute_teacher


def _validate_preregistration(path: Path) -> dict[str, Any]:
    prereg = _read(path)
    if (
        prereg.get("schema")
        != "zuma-rl.alphazuma-55-polar-dagger-preregistration"
        or prereg.get("version") != 1
        or prereg.get("status") != "FROZEN_BEFORE_TRAINING"
    ):
        raise ValueError("unexpected polar DAgger preregistration")
    trainer = prereg.get("trainer", {})
    if Path(str(trainer.get("path", ""))).resolve() != SCRIPT_PATH:
        raise ValueError("preregistration binds another polar DAgger trainer")
    if trainer.get("sha256") != legacy._sha256(SCRIPT_PATH):
        raise ValueError("polar DAgger trainer bytes differ")
    master = prereg.get("master_preregistration", {})
    master_path = Path(str(master.get("path", ""))).resolve(strict=True)
    if master.get("sha256") != legacy._sha256(master_path):
        raise ValueError("master preregistration bytes differ")
    master_value = _read(master_path)
    if (
        master_value.get("schema")
        != "zuma-rl.alphazuma-55-weekend-master-preregistration"
    ):
        raise ValueError("unexpected master preregistration")
    plan = prereg.get("materialization_plan", {})
    plan_path = Path(str(plan.get("path", ""))).resolve(strict=True)
    if plan.get("sha256") != legacy._sha256(plan_path):
        raise ValueError("materialization plan bytes differ")
    for name, artifact in prereg.get("implementation", {}).items():
        artifact_path = Path(str(artifact["path"])).resolve(strict=True)
        if artifact.get("sha256") != legacy._sha256(artifact_path):
            raise ValueError(f"implementation bytes differ: {name}")
    settled._validate_probe_result(
        prereg.get("teacher_evidence", {}).get("exact_mask_fresh_55", {}),
        expected_policy_id=POLICY_ID,
    )
    source = prereg.get("source", {})
    completion_path = Path(str(source.get("completion_path", ""))).resolve(
        strict=True
    )
    if source.get("completion_sha256") != legacy._sha256(completion_path):
        raise ValueError("source completion bytes differ")
    completion = _read(completion_path)
    if (
        completion.get("status") != "COMPLETE"
        or completion.get("policy_architecture") != "entity_polar"
        or completion.get("formal_seed_consumption") is not False
    ):
        raise ValueError("source polar completion is not eligible")
    source_model = Path(str(source.get("model_path", ""))).resolve(strict=True)
    if source.get("model_sha256") != legacy._sha256(source_model):
        raise ValueError("source polar model bytes differ")
    completion_model = completion.get("final_model", {})
    if (
        Path(str(completion_model.get("path", ""))).resolve() != source_model
        or completion_model.get("sha256") != source.get("model_sha256")
    ):
        raise ValueError("source model is not the completion's frozen final model")
    if tuple(prereg.get("levels", ())) != tuple(INCLUDED_LEVELS):
        raise ValueError("polar DAgger level order changed")
    run = prereg.get("run", {})
    schedule = _validate_probability_schedule(
        run.get("teacher_execution_probabilities", ())
    )
    if (
        run.get("policy_architecture") != "entity_polar"
        or int(run.get("rounds", -1)) != len(schedule)
        or run.get("aim_loss_scope") != "fire_only"
        or run.get("aim_loss_mode") != "categorical"
        or run.get("exact_action_masks_before_every_decision") is not True
        or run.get("aggregate_dataset_across_rounds") is not True
    ):
        raise ValueError("polar DAgger semantic contract changed")
    first = int(run["training_seed_base"])
    last = int(run["training_seed_last_consumed"])
    if last != first + len(INCLUDED_LEVELS) * len(schedule) - 1:
        raise ValueError("polar DAgger seed interval does not match its rounds")
    registry = master_value["seed_registry"]["training"]
    if not int(registry["first"]) <= first <= last <= int(registry["last"]):
        raise ValueError("polar DAgger training seeds escaped the registry")
    boundary = prereg.get("authority_boundary", {})
    if any(
        boundary.get(name) is not False
        for name in (
            "current_campaign_candidate_authority",
            "formal_selection_seed_consumption",
            "formal_final_blind_seed_consumption",
            "continuous_campaign_seed_consumption",
        )
    ):
        raise ValueError("polar DAgger authority boundary changed")
    return prereg


def _prototype_environment(*, original_root: Path, max_ticks: int) -> RevengeEnv:
    return RevengeEnv(
        config=full55_environment_config(max_ticks=max_ticks),
        level_id=INCLUDED_LEVELS[0],
        root=original_root,
        hard=False,
        curve_index=0,
        profile_mode=SUPPORTED_PROFILE_MODE,
    )


def _validate_student_action(action: np.ndarray, mask: np.ndarray) -> None:
    verb = int(action[0])
    aim = int(action[1])
    flat_mask = np.asarray(mask, dtype=np.bool_).reshape(-1)
    if not 0 <= verb < 4 or not 0 <= aim < 180:
        raise RuntimeError(f"student action escaped full55 interface: {action}")
    if len(flat_mask) != 184:
        raise RuntimeError("student action mask is not full55-v1")
    if not bool(flat_mask[verb]) or not bool(flat_mask[4 + aim]):
        raise RuntimeError(f"student emitted masked action: {action}")


def _collect_dagger_round(
    *,
    original_root: Path,
    level_ids: Sequence[str],
    student: Any,
    round_index: int,
    seed_base: int,
    parallel_envs: int,
    max_ticks: int,
    capacities: dict[str, int],
    strides: dict[str, int],
    teacher_execution_probability: float,
    model_seed: int,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
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
    total_execution: Counter[str] = Counter()
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
            execution_counts = [Counter() for _ in batch]
            candidate_counts = [Counter() for _ in batch]
            retained: list[
                dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]]
            ] = [{name: [] for name in settled.VERB_NAMES} for _ in batch]
            per_episode_forced_waits = np.zeros(len(batch), dtype=np.int64)
            last_infos: list[dict[str, Any]] = [{} for _ in batch]

            for _ in range(max_ticks + 1):
                masks = np.asarray(get_action_masks(vector), dtype=np.bool_)
                teacher_actions = np.zeros((len(batch), 2), dtype=np.int64)
                active = ~finished
                for position in range(len(batch)):
                    if finished[position]:
                        continue
                    raw = np.asarray(
                        teachers[position].act(observations[position]),
                        dtype=np.int64,
                    )
                    raw_name = settled.VERB_NAMES[int(raw[0])]
                    raw_counts[position][raw_name] += 1
                    total_raw[raw_name] += 1
                    label, forced = settled._effective_label(raw, masks[position])
                    teacher_actions[position] = label
                    name = settled.VERB_NAMES[int(label[0])]
                    effective_counts[position][name] += 1
                    total_effective[name] += 1
                    if forced:
                        per_episode_forced_waits[position] += 1
                        forced_waits += 1
                    stride = int(strides[name])
                    if int(steps[position]) % stride != 0:
                        continue
                    candidate_counts[position][name] += 1
                    settled._reservoir_consider(
                        retained[position][name],
                        seen=int(candidate_counts[position][name]),
                        capacity=int(capacities[name]),
                        observation=observations[position],
                        action=label,
                        mask=masks[position],
                        rng=rng,
                    )

                if teacher_execution_probability == 1.0:
                    student_actions = teacher_actions.copy()
                else:
                    predicted, _ = student.predict(
                        observations,
                        deterministic=True,
                        action_masks=masks,
                    )
                    student_actions = np.asarray(predicted, dtype=np.int64).reshape(
                        len(batch), 2
                    )
                    for position in np.flatnonzero(active):
                        _validate_student_action(
                            student_actions[int(position)], masks[int(position)]
                        )
                actions, execute_teacher = _mix_actions(
                    teacher_actions=teacher_actions,
                    student_actions=student_actions,
                    active=active,
                    teacher_probability=teacher_execution_probability,
                    rng=rng,
                )
                for position in np.flatnonzero(active):
                    source = "teacher" if execute_teacher[int(position)] else "student"
                    execution_counts[int(position)][source] += 1
                    total_execution[source] += 1

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
                raise RuntimeError(f"polar DAgger episodes exceeded guard: {missing}")
            for position, level_id in enumerate(batch):
                retained_counts: dict[str, int] = {}
                for name in settled.VERB_NAMES:
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
                        "execution_counts": dict(execution_counts[position]),
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
        if progress_callback is not None:
            progress_callback(
                {
                    "round_index": round_index,
                    "completed_levels": len(episodes),
                    "expected_levels": len(level_ids),
                    "retained_samples": len(aggregate),
                    "wins": sum(row["outcome"] == "win" for row in episodes),
                    "losses": sum(row["outcome"] == "loss" for row in episodes),
                    "truncations": sum(
                        row["time_limit_truncated"] for row in episodes
                    ),
                    "wall_seconds": time.perf_counter() - started,
                }
            )

    retained_by_verb = Counter(
        settled.VERB_NAMES[int(row[1][0])] for row in aggregate
    )
    return aggregate, {
        "round_index": round_index,
        "teacher_execution_probability": teacher_execution_probability,
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
        "execution_counts": dict(total_execution),
        "mask_forced_waits": forced_waits,
        "teacher_policy_id": POLICY_ID,
    }


def run(*, preregistration_path: Path, original_root: Path) -> dict[str, Any]:
    preregistration_path = preregistration_path.resolve(strict=True)
    original_root = original_root.resolve(strict=True)
    prereg = _validate_preregistration(preregistration_path)
    run_config = prereg["run"]
    run_dir = Path(str(run_config["run_dir"])).resolve()
    if run_dir.exists():
        raise FileExistsError(f"polar DAgger run already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    started = time.perf_counter()
    _write_atomic(
        run_dir / "config.json",
        {
            "schema": "zuma-rl.alphazuma-55-polar-dagger-config",
            "version": 1,
            "status": "FROZEN",
            "preregistration": {
                "path": str(preregistration_path),
                "sha256": legacy._sha256(preregistration_path),
            },
            "trainer": {
                "path": str(SCRIPT_PATH),
                "sha256": legacy._sha256(SCRIPT_PATH),
            },
            "source": prereg["source"],
            "run": run_config,
        },
    )

    prototype: RevengeEnv | None = None
    try:
        from sb3_contrib import MaskablePPO
        import torch

        prototype = _prototype_environment(
            original_root=original_root,
            max_ticks=int(run_config["max_ticks"]),
        )
        source_model = Path(str(prereg["source"]["model_path"]))
        model = MaskablePPO.load(
            source_model,
            env=prototype,
            device=str(run_config["device"]),
        )
        if tuple(int(value) for value in model.observation_space.shape) != (
            EXPECTED_OBSERVATION_SHAPE
        ):
            raise ValueError("source polar observation space is not full55-v1")
        if tuple(int(value) for value in model.action_space.nvec) != (
            EXPECTED_ACTION_NVEC
        ):
            raise ValueError("source polar action space is not full55-v1")
        for group in model.policy.optimizer.param_groups:
            group["lr"] = float(run_config["learning_rate"])

        schedule = _validate_probability_schedule(
            run_config["teacher_execution_probabilities"]
        )
        aggregate_dataset: list[
            tuple[np.ndarray, np.ndarray, np.ndarray]
        ] = []
        completed_rounds: list[dict[str, Any]] = []

        def write_status(stage: str, **extra: Any) -> None:
            _write_atomic(
                run_dir / "training_status.json",
                {
                    "schema": "zuma-rl.alphazuma-55-polar-dagger-status",
                    "version": 1,
                    "status": "RUNNING",
                    "stage": stage,
                    "updated_utc": _utc_now(),
                    "completed_rounds": len(completed_rounds),
                    "expected_rounds": len(schedule),
                    "aggregate_samples": len(aggregate_dataset),
                    "rounds": completed_rounds,
                    "wall_seconds": time.perf_counter() - started,
                    **extra,
                },
            )

        for round_index, teacher_probability in enumerate(schedule):
            write_status(
                "COLLECTING",
                current_round=round_index,
                teacher_execution_probability=teacher_probability,
            )

            def collection_progress(progress: dict[str, Any]) -> None:
                write_status(
                    "COLLECTING",
                    current_round=round_index,
                    teacher_execution_probability=teacher_probability,
                    collection_progress=progress,
                )

            collected, collection = _collect_dagger_round(
                original_root=original_root,
                level_ids=INCLUDED_LEVELS,
                student=model,
                round_index=round_index,
                seed_base=int(run_config["training_seed_base"]),
                parallel_envs=int(run_config["parallel_envs"]),
                max_ticks=int(run_config["max_ticks"]),
                capacities={
                    name: int(run_config["reservoir_capacity_by_verb"][name])
                    for name in settled.VERB_NAMES
                },
                strides={
                    name: int(run_config["sample_stride_by_verb"][name])
                    for name in settled.VERB_NAMES
                },
                teacher_execution_probability=teacher_probability,
                model_seed=int(run_config["model_seed"]),
                progress_callback=collection_progress,
            )
            aggregate_dataset.extend(collected)
            round_dir = run_dir / f"round_{round_index:02d}"
            round_dir.mkdir()

            def optimization_progress(
                completed_epochs: int,
                epoch_rows: list[dict[str, Any]],
                checkpoints: list[dict[str, Any]],
            ) -> None:
                write_status(
                    "OPTIMIZING",
                    current_round=round_index,
                    teacher_execution_probability=teacher_probability,
                    collection=collection,
                    completed_epochs_in_round=completed_epochs,
                    expected_epochs_in_round=int(
                        run_config["epochs_per_round"]
                    ),
                    epochs=epoch_rows,
                    checkpoints=checkpoints,
                )

            optimization_run = dict(run_config)
            optimization_run["model_seed"] = (
                int(run_config["model_seed"]) + round_index * 1_000_003
            )
            optimization = settled._optimize(
                model=model,
                dataset=aggregate_dataset,
                run=optimization_run,
                run_dir=round_dir,
                status_callback=optimization_progress,
            )
            capacity = polar._evaluate_full55_dataset(
                model=model,
                dataset=aggregate_dataset,
                batch_size=int(run_config["capacity_eval_batch_size"]),
            )
            round_model = round_dir / "final_model.zip"
            model.save(round_model)
            completed_rounds.append(
                {
                    "round_index": round_index,
                    "teacher_execution_probability": teacher_probability,
                    "collection": collection,
                    "aggregate_samples": len(aggregate_dataset),
                    "optimization": optimization,
                    "capacity_metrics_on_consumed_training_data": capacity,
                    "model": {
                        "path": str(round_model),
                        "sha256": legacy._sha256(round_model),
                        "bytes": round_model.stat().st_size,
                    },
                }
            )
            write_status("ROUND_COMPLETE", current_round=round_index)

        final_model = run_dir / "final_model.zip"
        model.num_timesteps = sum(
            int(episode["steps"])
            for row in completed_rounds
            for episode in row["collection"]["episodes"]
        )
        model.save(final_model)
        completion = {
            "schema": "zuma-rl.alphazuma-55-polar-dagger-completion",
            "version": 1,
            "status": "COMPLETE",
            "completed_utc": _utc_now(),
            "wall_seconds": time.perf_counter() - started,
            "policy_architecture": "entity_polar",
            "policy_parameter_count": sum(
                parameter.numel() for parameter in model.policy.parameters()
            ),
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "torch": torch.__version__,
                "device": str(run_config["device"]),
                "cuda_device": torch.cuda.get_device_name(
                    torch.device(str(run_config["device"]))
                ),
            },
            "teacher_policy_id": POLICY_ID,
            "source": prereg["source"],
            "rounds": completed_rounds,
            "aggregate_samples": len(aggregate_dataset),
            "final_capacity_metrics_on_consumed_training_data": (
                completed_rounds[-1][
                    "capacity_metrics_on_consumed_training_data"
                ]
            ),
            "final_model": {
                "path": str(final_model),
                "sha256": legacy._sha256(final_model),
                "bytes": final_model.stat().st_size,
                "num_timesteps": int(model.num_timesteps),
            },
            "formal_seed_consumption": False,
            "formal_candidate_authority": False,
        }
        _write_atomic(run_dir / "completion.json", completion)
        return completion
    except BaseException as error:
        _write_atomic(
            run_dir / "failure.json",
            {
                "schema": "zuma-rl.alphazuma-55-polar-dagger-failure",
                "version": 1,
                "status": "FAILED",
                "failed_utc": _utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
                "formal_seed_consumption": False,
            },
        )
        raise
    finally:
        if prototype is not None:
            prototype.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--original-root", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(
        preregistration_path=args.preregistration.expanduser(),
        original_root=args.original_root.expanduser(),
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "final_model": result["final_model"],
                "capacity_metrics": result[
                    "final_capacity_metrics_on_consumed_training_data"
                ],
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
