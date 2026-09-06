"""Run frozen full-55 DAgger correction from a selected neural student."""

from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
import math
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

from tools import distill_alphazuma_55_settled_v3 as base
from zuma_rl.human_speedrun import EliteHumanInputConfig, WinFirstRewardConfig
from zuma_rl.revenge_settled_strategic_teacher_v3 import (
    CurveAwareSettledStrategicRevengeTeacher,
)


SCRIPT_PATH = Path(__file__).resolve()
POLICY_ID = base.POLICY_ID
VERB_NAMES = base.VERB_NAMES
INCLUDED_LEVELS = base.INCLUDED_LEVELS


def _is_exact_masked_action_legal(action: np.ndarray, mask: np.ndarray) -> bool:
    """Return whether a factorized ``(verb, aim)`` action is exactly legal."""

    candidate = np.asarray(action, dtype=np.int64).reshape(-1)
    exact_mask = np.asarray(mask, dtype=np.bool_).reshape(-1)
    if candidate.size != 2 or exact_mask.size != len(VERB_NAMES) + 180:
        return False
    verb, aim = (int(candidate[0]), int(candidate[1]))
    return bool(
        0 <= verb < len(VERB_NAMES)
        and 0 <= aim < 180
        and exact_mask[verb]
        and exact_mask[len(VERB_NAMES) + aim]
    )


def _validate_preregistration(path: Path) -> dict[str, Any]:
    prereg = base.legacy._read_json(path)
    if (
        prereg.get("schema") != "zuma-rl.alphazuma-55-dagger-preregistration"
        or prereg.get("version") != 1
        or prereg.get("status") != "FROZEN_BEFORE_TRAINING"
    ):
        raise ValueError("unexpected DAgger preregistration")
    trainer = prereg["trainer"]
    if (
        Path(str(trainer["path"])).resolve() != SCRIPT_PATH
        or trainer["sha256"] != base.legacy._sha256(SCRIPT_PATH)
    ):
        raise ValueError("DAgger trainer bytes differ")
    for name, artifact in prereg["implementation"].items():
        artifact_path = Path(str(artifact["path"])).resolve(strict=True)
        if artifact["sha256"] != base.legacy._sha256(artifact_path):
            raise ValueError(f"implementation bytes differ: {name}")
    master_reference = prereg["master_preregistration"]
    master_path = Path(str(master_reference["path"])).resolve(strict=True)
    if master_reference["sha256"] != base.legacy._sha256(master_path):
        raise ValueError("master preregistration bytes differ")
    source = prereg["student_source_model"]
    source_path = Path(str(source["path"])).resolve(strict=True)
    if source["sha256"] != base.legacy._sha256(source_path):
        raise ValueError("DAgger source model bytes differ")
    selection = prereg["source_selection"]
    selection_path = Path(str(selection["path"])).resolve(strict=True)
    if selection["sha256"] != base.legacy._sha256(selection_path):
        raise ValueError("DAgger source-selection receipt bytes differ")
    selection_value = base.legacy._read_json(selection_path)
    if (
        selection_value.get("status") != "SELECTED"
        or selection_value.get("selected_model", {}).get("sha256")
        != source["sha256"]
    ):
        raise ValueError("source-selection receipt does not bind the student")
    for name, policy_id in (
        ("unmasked_fresh_55", "curve-aware-settled-strategic-actor-observable-v3"),
        ("exact_mask_fresh_55", POLICY_ID),
    ):
        base._validate_probe_result(
            prereg["teacher_evidence"][name], expected_policy_id=policy_id
        )
    if prereg["levels"] != list(INCLUDED_LEVELS):
        raise ValueError("DAgger level scope differs from AlphaZuma 55")

    run = prereg["run"]
    run_dir = Path(str(run["run_dir"])).resolve()
    if run_dir.exists():
        raise FileExistsError(f"refusing to reuse DAgger run: {run_dir}")
    rounds = int(run["rounds"])
    probabilities = [float(value) for value in run["teacher_execution_probabilities"]]
    if (
        rounds != 2
        or len(probabilities) != rounds
        or any(not 0.0 <= value <= 1.0 for value in probabilities)
        or int(run["parallel_envs"]) != 24
        or int(run["max_ticks"]) != 30_000
        or int(run["epochs_per_round"]) != 12
        or int(run["checkpoint_interval_epochs"]) != 4
        or run.get("dataset_aggregation")
        != "student_visited_states_labeled_by_exact_masked_teacher"
        or run.get("optimize_after_each_round") is not True
        or run.get("aim_loss_scope") != "all_actions"
        or run.get("aim_loss_mode") != "circular_smoothed"
    ):
        raise ValueError("unexpected DAgger recipe")
    if not math.isclose(
        sum(float(value) for value in run["target_batch_fraction_by_verb"].values()),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("DAgger batch fractions do not sum to one")
    first_seed = int(run["training_seed_base"])
    last_seed = first_seed + rounds * len(INCLUDED_LEVELS) - 1
    if int(run["training_seed_last_consumed"]) != last_seed:
        raise ValueError("DAgger seed receipt is inconsistent")
    registry = base.legacy._read_json(master_path)["seed_registry"]["training"]
    if not int(registry["first"]) <= first_seed <= last_seed <= int(registry["last"]):
        raise ValueError("DAgger seeds escaped the training registry")
    return prereg


def _collect_dagger_round(
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
    teacher_execution_probability: float,
    student: Any,
) -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray]], dict[str, Any]]:
    """Collect teacher labels on a teacher/student mixture state distribution."""

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
    total_teacher_raw: Counter[str] = Counter()
    total_teacher_effective: Counter[str] = Counter()
    total_executed: Counter[str] = Counter()
    teacher_executions = 0
    student_executions = 0
    exact_agreements = 0
    verb_agreements = 0
    comparisons = 0
    forced_waits = 0
    started = time.perf_counter()

    for batch_start in range(0, len(level_ids), parallel_envs):
        batch = list(level_ids[batch_start : batch_start + parallel_envs])
        factories = [
            base.legacy._make_env_factory(
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
                    base.legacy._teacher_spec(value)
                )
                for value in specifications
            ]
            finished = np.zeros(len(batch), dtype=np.bool_)
            steps = np.zeros(len(batch), dtype=np.int64)
            retained: list[
                dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]]
            ] = [{name: [] for name in VERB_NAMES} for _ in batch]
            candidates = [Counter() for _ in batch]
            episode_forced_waits = np.zeros(len(batch), dtype=np.int64)
            episode_teacher = np.zeros(len(batch), dtype=np.int64)
            episode_student = np.zeros(len(batch), dtype=np.int64)
            episode_agreements = np.zeros(len(batch), dtype=np.int64)
            episode_verb_agreements = np.zeros(len(batch), dtype=np.int64)
            episode_comparisons = np.zeros(len(batch), dtype=np.int64)
            last_infos: list[dict[str, Any]] = [{} for _ in batch]

            for _ in range(max_ticks + 1):
                masks = np.asarray(get_action_masks(vector), dtype=np.bool_)
                predicted, _ = student.predict(
                    observations,
                    deterministic=True,
                    action_masks=masks,
                )
                predicted = np.asarray(predicted, dtype=np.int64).reshape(len(batch), 2)
                actions = np.zeros((len(batch), 2), dtype=np.int64)
                active = ~finished
                for position in range(len(batch)):
                    if finished[position]:
                        continue
                    raw = np.asarray(
                        teachers[position].act(observations[position]), dtype=np.int64
                    )
                    total_teacher_raw[VERB_NAMES[int(raw[0])]] += 1
                    label, forced = base._effective_label(raw, masks[position])
                    label_name = VERB_NAMES[int(label[0])]
                    total_teacher_effective[label_name] += 1
                    if forced:
                        forced_waits += 1
                        episode_forced_waits[position] += 1
                    student_action = np.asarray(predicted[position], dtype=np.int64)
                    student_verb = int(student_action[0])
                    student_aim = int(student_action[1])
                    if not _is_exact_masked_action_legal(
                        student_action, masks[position]
                    ):
                        raise RuntimeError("student emitted an action outside its exact mask")

                    comparisons += 1
                    episode_comparisons[position] += 1
                    exact = bool(np.array_equal(label, student_action))
                    verb_equal = int(label[0]) == student_verb
                    exact_agreements += int(exact)
                    verb_agreements += int(verb_equal)
                    episode_agreements[position] += int(exact)
                    episode_verb_agreements[position] += int(verb_equal)

                    stride = int(strides[label_name])
                    if int(steps[position]) % stride == 0:
                        candidates[position][label_name] += 1
                        base._reservoir_consider(
                            retained[position][label_name],
                            seen=int(candidates[position][label_name]),
                            capacity=int(capacities[label_name]),
                            observation=observations[position],
                            action=label,
                            mask=masks[position],
                            rng=rng,
                        )
                    use_teacher = bool(rng.random() < teacher_execution_probability)
                    if use_teacher:
                        action = label
                        teacher_executions += 1
                        episode_teacher[position] += 1
                    else:
                        action = student_action
                        student_executions += 1
                        episode_student[position] += 1
                    actions[position] = action
                    total_executed[VERB_NAMES[int(action[0])]] += 1

                observations, _, dones, infos = vector.step(actions)
                steps[active] += 1
                for position, (done, info) in enumerate(zip(dones, infos, strict=True)):
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
                raise RuntimeError(f"DAgger collection exceeded the guard: {missing}")
            for position, level_id in enumerate(batch):
                retained_counts: dict[str, int] = {}
                for name in VERB_NAMES:
                    aggregate.extend(retained[position][name])
                    retained_counts[name] = len(retained[position][name])
                info = last_infos[position]
                count = int(episode_comparisons[position])
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
                        "teacher_executions": int(episode_teacher[position]),
                        "student_executions": int(episode_student[position]),
                        "exact_action_agreement": (
                            float(episode_agreements[position]) / count if count else None
                        ),
                        "verb_agreement": (
                            float(episode_verb_agreements[position]) / count
                            if count
                            else None
                        ),
                        "mask_forced_waits": int(episode_forced_waits[position]),
                        "retained_samples_by_verb": retained_counts,
                        "retained_samples": sum(retained_counts.values()),
                        "observation_capacity_overflow": bool(
                            info.get("observation_capacity_overflow", False)
                        ),
                    }
                )
        finally:
            vector.close()

    retained_by_verb = Counter(VERB_NAMES[int(row[1][0])] for row in aggregate)
    return aggregate, {
        "round_index": round_index,
        "wall_seconds": time.perf_counter() - started,
        "teacher_execution_probability": float(teacher_execution_probability),
        "episodes": episodes,
        "wins": sum(row["outcome"] == "win" for row in episodes),
        "losses": sum(row["outcome"] == "loss" for row in episodes),
        "truncations": sum(row["time_limit_truncated"] for row in episodes),
        "capacity_overflows": sum(
            row["observation_capacity_overflow"] for row in episodes
        ),
        "retained_samples": len(aggregate),
        "retained_samples_by_verb": dict(retained_by_verb),
        "teacher_raw_action_counts": dict(total_teacher_raw),
        "teacher_effective_action_counts": dict(total_teacher_effective),
        "executed_action_counts": dict(total_executed),
        "teacher_executions": teacher_executions,
        "student_executions": student_executions,
        "exact_action_agreement": exact_agreements / comparisons,
        "verb_agreement": verb_agreements / comparisons,
        "mask_forced_waits": forced_waits,
        "teacher_policy_id": POLICY_ID,
    }


def run_dagger(
    *, prereg_path: Path, prereg: dict[str, Any], original_root: Path
) -> dict[str, Any]:
    import torch
    from sb3_contrib import MaskablePPO

    torch.set_num_threads(1)
    run = prereg["run"]
    torch.manual_seed(int(run["model_seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(run["model_seed"]))
    device = str(run["device"])
    run_dir = Path(str(run["run_dir"])).resolve()
    run_dir.mkdir(parents=True)
    started = time.perf_counter()
    source_path = Path(str(prereg["student_source_model"]["path"])).resolve(
        strict=True
    )
    base.legacy._write_json_atomic(
        run_dir / "config.json",
        {
            "schema": "zuma-rl.alphazuma-55-dagger-config",
            "version": 1,
            "status": "STARTED",
            "started_utc": base._utc_now(),
            "preregistration": {
                "path": str(prereg_path),
                "sha256": base.legacy._sha256(prereg_path),
            },
            "trainer_sha256": base.legacy._sha256(SCRIPT_PATH),
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
        if tuple(int(value) for value in student.observation_space.shape) != (22_833,):
            raise ValueError("DAgger student observation space is not full55-v1")
        if tuple(int(value) for value in student.action_space.nvec) != (4, 180):
            raise ValueError("DAgger student action space is not full55-v1")
        optimizer_class = student.policy.optimizer_class
        optimizer_kwargs = dict(student.policy.optimizer_kwargs)
        student.policy.optimizer = optimizer_class(
            student.policy.parameters(),
            lr=float(run["learning_rate"]),
            **optimizer_kwargs,
        )
        aggregate: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        collections: list[dict[str, Any]] = []
        optimizations: list[dict[str, Any]] = []
        probabilities = [float(value) for value in run["teacher_execution_probabilities"]]
        for round_index, probability in enumerate(probabilities):
            dataset, receipt = _collect_dagger_round(
                original_root=original_root,
                level_ids=list(INCLUDED_LEVELS),
                round_index=round_index,
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
                teacher_execution_probability=probability,
                student=student,
            )
            aggregate.extend(dataset)
            collections.append(receipt)
            collection_steps = sum(
                int(episode["steps"]) for episode in receipt["episodes"]
            )
            student.num_timesteps = int(student.num_timesteps) + collection_steps
            base.legacy._write_json_atomic(
                run_dir / "training_status.json",
                {
                    "schema": "zuma-rl.alphazuma-55-dagger-status",
                    "version": 1,
                    "status": "RUNNING",
                    "stage": "ROUND_COLLECTION_COMPLETE",
                    "updated_utc": base._utc_now(),
                    "completed_collections": round_index + 1,
                    "completed_optimizations": round_index,
                    "expected_rounds": len(probabilities),
                    "aggregate_samples": len(aggregate),
                    "student_num_timesteps": int(student.num_timesteps),
                    "collections": collections,
                    "optimizations": optimizations,
                    "wall_seconds": time.perf_counter() - started,
                },
            )
            round_dir = run_dir / f"round_{round_index + 1:02d}"
            round_dir.mkdir()

            def write_status(
                completed_epochs: int,
                epoch_rows: list[dict[str, Any]],
                checkpoints: list[dict[str, Any]],
            ) -> None:
                base.legacy._write_json_atomic(
                    run_dir / "training_status.json",
                    {
                        "schema": "zuma-rl.alphazuma-55-dagger-status",
                        "version": 1,
                        "status": "RUNNING",
                        "stage": "OPTIMIZING_ROUND",
                        "updated_utc": base._utc_now(),
                        "active_round": round_index + 1,
                        "completed_collections": round_index + 1,
                        "completed_optimizations": round_index,
                        "expected_rounds": len(probabilities),
                        "aggregate_samples": len(aggregate),
                        "student_num_timesteps": int(student.num_timesteps),
                        "collections": collections,
                        "optimizations": optimizations,
                        "completed_epochs_in_active_round": completed_epochs,
                        "expected_epochs_in_active_round": int(
                            run["epochs_per_round"]
                        ),
                        "active_round_epochs": epoch_rows,
                        "active_round_checkpoints": checkpoints,
                        "wall_seconds": time.perf_counter() - started,
                    },
                )

            write_status(0, [], [])
            optimization = base._optimize(
                model=student,
                dataset=aggregate,
                run=run,
                run_dir=round_dir,
                status_callback=write_status,
            )
            round_model = round_dir / "model_after_round.zip"
            student.save(round_model)
            optimization = {
                "round_index": round_index,
                "aggregate_samples": len(aggregate),
                "teacher_execution_probability": probability,
                **optimization,
                "round_model": {
                    "path": str(round_model),
                    "sha256": base.legacy._sha256(round_model),
                    "bytes": round_model.stat().st_size,
                    "num_timesteps": int(student.num_timesteps),
                },
            }
            optimizations.append(optimization)
            base.legacy._write_json_atomic(
                run_dir / "training_status.json",
                {
                    "schema": "zuma-rl.alphazuma-55-dagger-status",
                    "version": 1,
                    "status": "RUNNING",
                    "stage": "ROUND_COMPLETE",
                    "updated_utc": base._utc_now(),
                    "completed_collections": round_index + 1,
                    "completed_optimizations": round_index + 1,
                    "expected_rounds": len(probabilities),
                    "aggregate_samples": len(aggregate),
                    "student_num_timesteps": int(student.num_timesteps),
                    "collections": collections,
                    "optimizations": optimizations,
                    "wall_seconds": time.perf_counter() - started,
                },
            )
        final_path = run_dir / "final_model.zip"
        student.save(final_path)
        completion = {
            "schema": "zuma-rl.alphazuma-55-dagger-completion",
            "version": 1,
            "status": "COMPLETE",
            "completed_utc": base._utc_now(),
            "wall_seconds": time.perf_counter() - started,
            "source_model": prereg["student_source_model"],
            "source_selection": prereg["source_selection"],
            "teacher_policy_id": POLICY_ID,
            "collections": collections,
            "aggregate_collection": {
                "rounds": len(collections),
                "episodes": sum(len(row["episodes"]) for row in collections),
                "wins": sum(int(row["wins"]) for row in collections),
                "losses": sum(int(row["losses"]) for row in collections),
                "truncations": sum(int(row["truncations"]) for row in collections),
                "capacity_overflows": sum(
                    int(row["capacity_overflows"]) for row in collections
                ),
                "retained_samples": len(aggregate),
            },
            "optimizations": optimizations,
            "final_model": {
                "path": str(final_path),
                "sha256": base.legacy._sha256(final_path),
                "bytes": final_path.stat().st_size,
                "num_timesteps": int(student.num_timesteps),
            },
            "formal_seed_consumption": False,
        }
        base.legacy._write_json_atomic(run_dir / "completion.json", completion)
        return completion
    except BaseException as error:
        base.legacy._write_json_atomic(
            run_dir / "failure.json",
            {
                "schema": "zuma-rl.alphazuma-55-dagger-failure",
                "version": 1,
                "status": "FAILED",
                "failed_utc": base._utc_now(),
                "wall_seconds": time.perf_counter() - started,
                "error_type": type(error).__name__,
                "error": str(error),
                "formal_seed_consumption": False,
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
                    "sha256": base.legacy._sha256(prereg_path),
                    "run": prereg["run"],
                    "formal_seed_consumption": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    result = run_dagger(
        prereg_path=prereg_path, prereg=prereg, original_root=original_root
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "aggregate_collection": result["aggregate_collection"],
                "final_model": result["final_model"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
