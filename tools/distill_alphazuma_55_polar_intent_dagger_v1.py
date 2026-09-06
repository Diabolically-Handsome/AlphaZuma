"""Fine-tune the raw-intent wide policy on frozen DAgger state mixtures."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable, Sequence

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import numpy as np

from tools import distill_alphazuma_55 as legacy
from tools import distill_alphazuma_55_polar_dagger_v1 as dagger
from tools import distill_alphazuma_55_polar_intent_wide_v1 as intent
from tools import distill_alphazuma_55_settled_v3 as settled
from zuma_rl.alphazuma_55 import INCLUDED_LEVELS


SCRIPT_PATH = Path(__file__).resolve()
EXPECTED_SCHEDULE = (0.75, 0.25, 0.0)
EXPECTED_FEATURES_DIM = 1024
EXPECTED_EPOCHS_PER_ROUND = 18


def _validate_probability_schedule(values: Sequence[Any]) -> tuple[float, ...]:
    schedule = tuple(float(value) for value in values)
    if schedule != EXPECTED_SCHEDULE:
        raise ValueError(
            "raw-intent DAgger schedule must remain exactly "
            f"{EXPECTED_SCHEDULE}"
        )
    return schedule


def _bound(reference: Any, label: str) -> Path:
    if not isinstance(reference, dict):
        raise ValueError(f"{label} reference must be an object")
    path = Path(str(reference.get("path", ""))).resolve(strict=True)
    if reference.get("sha256") != legacy._sha256(path):
        raise ValueError(f"{label} bytes differ: {path}")
    return path


def _validate_preregistration(path: Path) -> dict[str, Any]:
    prereg = dagger._read(path)
    if (
        prereg.get("schema")
        != "zuma-rl.alphazuma-55-polar-intent-dagger-preregistration"
        or prereg.get("version") != 1
        or prereg.get("status") != "FROZEN_BEFORE_TRAINING"
    ):
        raise ValueError("unexpected raw-intent DAgger preregistration")
    trainer = prereg.get("trainer", {})
    if (
        Path(str(trainer.get("path", ""))).resolve() != SCRIPT_PATH
        or trainer.get("sha256") != legacy._sha256(SCRIPT_PATH)
    ):
        raise ValueError("raw-intent DAgger trainer binding differs")
    master_path = _bound(prereg.get("master_preregistration"), "master")
    master = dagger._read(master_path)
    if (
        master.get("schema")
        != "zuma-rl.alphazuma-55-weekend-master-preregistration"
    ):
        raise ValueError("unexpected raw-intent DAgger master")
    _bound(prereg.get("materialization_plan"), "materialization plan")
    for name, artifact in prereg.get("implementation", {}).items():
        _bound(artifact, f"implementation {name}")
    evidence = prereg.get("teacher_evidence", {}).get(
        "exact_mask_fresh_55", {}
    )
    evidence_path = _bound(evidence, "teacher evidence")
    intent.registration._validate_probe(evidence_path)

    source = prereg.get("source", {})
    completion_path = Path(str(source.get("completion_path", ""))).resolve(
        strict=True
    )
    if source.get("completion_sha256") != legacy._sha256(completion_path):
        raise ValueError("raw-intent source completion bytes differ")
    completion = dagger._read(completion_path)
    if not (
        completion.get("schema")
        == (
            "zuma-rl.alphazuma-55-polar-intent-wide-"
            "distillation-completion"
        )
        and completion.get("status") == "COMPLETE"
        and completion.get("policy_architecture")
        == "entity_polar_intent_wide"
        and completion.get("training_label_semantics")
        == "raw_teacher_intent"
        and completion.get("online_policy_inference_semantics")
        == "exact_action_masks"
        and completion.get("formal_seed_consumption") is False
    ):
        raise ValueError("raw-intent source completion is not eligible")
    source_model = Path(str(source.get("model_path", ""))).resolve(strict=True)
    if source.get("model_sha256") != legacy._sha256(source_model):
        raise ValueError("raw-intent source model bytes differ")
    final = completion.get("final_model", {})
    if (
        Path(str(final.get("path", ""))).resolve() != source_model
        or final.get("sha256") != source.get("model_sha256")
    ):
        raise ValueError("source is not the frozen raw-intent final model")
    if tuple(prereg.get("levels", ())) != tuple(INCLUDED_LEVELS):
        raise ValueError("raw-intent DAgger level order changed")

    run = prereg.get("run", {})
    schedule = _validate_probability_schedule(
        run.get("teacher_execution_probabilities", ())
    )
    if not (
        run.get("policy_architecture")
        == "entity_polar_intent_wide_dagger"
        and int(run.get("learned_features_dim", -1))
        == EXPECTED_FEATURES_DIM
        and int(run.get("rounds", -1)) == len(schedule)
        and int(run.get("epochs_per_round", -1))
        == EXPECTED_EPOCHS_PER_ROUND
        and run.get("source_is_frozen_raw_intent_final_model") is True
        and run.get("student_state_collection") is True
        and run.get("raw_teacher_intent_labels") is True
        and run.get("training_mask_relaxation")
        == "unmask_only_the_teacher_target_verb"
        and run.get("exact_action_masks_for_environment_execution") is True
        and run.get("exact_action_masks_for_online_policy_inference") is True
        and run.get("label_aim_must_remain_exact_mask_legal") is True
        and run.get("aggregate_dataset_across_rounds") is True
        and run.get("optimize_after_each_round") is True
        and run.get("aim_loss_scope") == "fire_only"
        and run.get("aim_loss_mode") == "categorical"
    ):
        raise ValueError("raw-intent DAgger semantic contract changed")
    first = int(run["training_seed_base"])
    last = int(run["training_seed_last_consumed"])
    if last != first + len(INCLUDED_LEVELS) * len(schedule) - 1:
        raise ValueError("raw-intent DAgger seed interval changed")
    registry = master["seed_registry"]["training"]
    if not int(registry["first"]) <= first <= last <= int(registry["last"]):
        raise ValueError("raw-intent DAgger seeds escaped training registry")
    boundary = prereg.get("authority_boundary", {})
    for name in (
        "current_campaign_candidate_authority",
        "s99081545_successor_candidate_authority",
        "formal_selection_seed_consumption",
        "formal_final_blind_seed_consumption",
        "continuous_campaign_seed_consumption",
    ):
        if boundary.get(name) is not False:
            raise ValueError(f"raw-intent DAgger authority changed: {name}")
    return prereg


def _collect_intent_dagger_round(
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
    """Collect raw teacher intent on teacher/student-mixed state trajectories."""

    from sb3_contrib.common.maskable.utils import get_action_masks
    from stable_baselines3.common.vec_env import SubprocVecEnv

    input_config = dagger.EliteHumanInputConfig(
        profile_id="elite-human-v1",
        reaction_delay_ticks=12,
        max_aim_speed_degrees_per_second=1080.0,
        max_aim_acceleration_degrees_per_second_squared=18000.0,
        min_button_interval_ticks=5,
    )
    reward_config = dagger.WinFirstRewardConfig(
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
    relaxations = 0
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
                dagger.CurveAwareSettledStrategicRevengeTeacher(
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
            episode_relaxations = np.zeros(len(batch), dtype=np.int64)
            last_infos: list[dict[str, Any]] = [{} for _ in batch]

            for _ in range(max_ticks + 1):
                masks = np.asarray(get_action_masks(vector), dtype=np.bool_)
                teacher_effective = np.zeros((len(batch), 2), dtype=np.int64)
                active = ~finished
                for position in range(len(batch)):
                    if finished[position]:
                        continue
                    teacher_raw = np.asarray(
                        teachers[position].act(observations[position]),
                        dtype=np.int64,
                    )
                    raw, effective, training_mask, forced = intent._intent_label(
                        teacher_raw, masks[position]
                    )
                    teacher_effective[position] = effective
                    raw_name = settled.VERB_NAMES[int(raw[0])]
                    effective_name = settled.VERB_NAMES[int(effective[0])]
                    raw_counts[position][raw_name] += 1
                    effective_counts[position][effective_name] += 1
                    total_raw[raw_name] += 1
                    total_effective[effective_name] += 1
                    if forced:
                        episode_relaxations[position] += 1
                        relaxations += 1
                    stride = int(strides[raw_name])
                    if int(steps[position]) % stride != 0:
                        continue
                    candidate_counts[position][raw_name] += 1
                    settled._reservoir_consider(
                        retained[position][raw_name],
                        seen=int(candidate_counts[position][raw_name]),
                        capacity=int(capacities[raw_name]),
                        observation=observations[position],
                        action=raw,
                        mask=training_mask,
                        rng=rng,
                    )

                if teacher_execution_probability == 1.0:
                    student_actions = teacher_effective.copy()
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
                        dagger._validate_student_action(
                            student_actions[int(position)], masks[int(position)]
                        )
                actions, execute_teacher = dagger._mix_actions(
                    teacher_actions=teacher_effective,
                    student_actions=student_actions,
                    active=active,
                    teacher_probability=teacher_execution_probability,
                    rng=rng,
                )
                for position in np.flatnonzero(active):
                    source = (
                        "teacher_effective"
                        if execute_teacher[int(position)]
                        else "student_exact_masked"
                    )
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
                raise RuntimeError(
                    f"raw-intent DAgger episodes exceeded guard: {missing}"
                )
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
                        "raw_intent_action_counts": dict(raw_counts[position]),
                        "effective_teacher_action_counts": dict(
                            effective_counts[position]
                        ),
                        "environment_execution_source_counts": dict(
                            execution_counts[position]
                        ),
                        "intent_mask_relaxations": int(
                            episode_relaxations[position]
                        ),
                        "candidate_samples_by_raw_verb": dict(
                            candidate_counts[position]
                        ),
                        "retained_samples_by_raw_verb": retained_counts,
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
        "raw_intent_action_counts": dict(total_raw),
        "effective_teacher_action_counts": dict(total_effective),
        "environment_execution_source_counts": dict(total_execution),
        "intent_mask_relaxations": relaxations,
        "training_label_semantics": "raw_teacher_intent",
        "state_distribution_semantics": "dagger_teacher_student_mixture",
        "environment_execution_semantics": (
            "exact_mask_teacher_effective_or_exact_mask_student"
        ),
        "teacher_policy_id": dagger.POLICY_ID,
    }


def run(*, preregistration_path: Path, original_root: Path) -> dict[str, Any]:
    preregistration_path = preregistration_path.resolve(strict=True)
    prereg = _validate_preregistration(preregistration_path)
    original_validate = dagger._validate_preregistration
    original_collector = dagger._collect_dagger_round
    original_schedule = dagger._validate_probability_schedule
    original_script_path = dagger.SCRIPT_PATH
    try:
        dagger._validate_preregistration = lambda _: prereg
        dagger._collect_dagger_round = _collect_intent_dagger_round
        dagger._validate_probability_schedule = _validate_probability_schedule
        dagger.SCRIPT_PATH = SCRIPT_PATH
        completion = dagger.run(
            preregistration_path=preregistration_path,
            original_root=original_root,
        )
    except BaseException:
        run_dir = Path(str(prereg["run"]["run_dir"])).resolve()
        failure_path = run_dir / "failure.json"
        if failure_path.exists():
            failure = dagger._read(failure_path)
            failure["schema"] = (
                "zuma-rl.alphazuma-55-polar-intent-dagger-failure"
            )
            failure["training_label_semantics"] = "raw_teacher_intent"
            failure["state_distribution_semantics"] = (
                "dagger_teacher_student_mixture"
            )
            dagger._write_atomic(failure_path, failure)
        raise
    finally:
        dagger._validate_preregistration = original_validate
        dagger._collect_dagger_round = original_collector
        dagger._validate_probability_schedule = original_schedule
        dagger.SCRIPT_PATH = original_script_path
    completion.update(
        {
            "schema": "zuma-rl.alphazuma-55-polar-intent-dagger-completion",
            "policy_architecture": "entity_polar_intent_wide_dagger",
            "learned_features_dim": EXPECTED_FEATURES_DIM,
            "training_label_semantics": "raw_teacher_intent",
            "training_mask_relaxation": "unmask_only_the_teacher_target_verb",
            "state_distribution_semantics": "dagger_teacher_student_mixture",
            "teacher_execution_probabilities": list(EXPECTED_SCHEDULE),
            "environment_execution_semantics": (
                "exact_mask_teacher_effective_or_exact_mask_student"
            ),
            "online_policy_inference_semantics": "exact_action_masks",
        }
    )
    completion_path = Path(str(prereg["run"]["run_dir"])) / "completion.json"
    dagger._write_atomic(completion_path, completion)
    return completion


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
        from sb3_contrib import MaskablePPO

        prototype = dagger._prototype_environment(
            original_root=original_root,
            max_ticks=int(prereg["run"]["max_ticks"]),
        )
        try:
            model = MaskablePPO.load(
                Path(prereg["source"]["model_path"]),
                env=prototype,
                device=str(prereg["run"]["device"]),
            )
            learned = int(model.policy.features_extractor.learned_features_dim)
            if learned != EXPECTED_FEATURES_DIM:
                raise ValueError("raw-intent source feature width changed")
            print(
                json.dumps(
                    {
                        "status": "VALID",
                        "preregistration": {
                            "path": str(prereg_path),
                            "sha256": legacy._sha256(prereg_path),
                        },
                        "policy_architecture": (
                            "entity_polar_intent_wide_dagger"
                        ),
                        "learned_features_dim": learned,
                        "teacher_execution_probabilities": list(
                            EXPECTED_SCHEDULE
                        ),
                        "training_label_semantics": "raw_teacher_intent",
                        "state_distribution_semantics": (
                            "dagger_teacher_student_mixture"
                        ),
                        "formal_seed_consumption": False,
                    },
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                )
            )
        finally:
            prototype.close()
        return 0
    completion = run(
        preregistration_path=prereg_path,
        original_root=original_root,
    )
    print(
        json.dumps(
            {
                "status": completion["status"],
                "model": completion["final_model"],
                "training_label_semantics": completion[
                    "training_label_semantics"
                ],
                "state_distribution_semantics": completion[
                    "state_distribution_semantics"
                ],
                "formal_seed_consumption": False,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
