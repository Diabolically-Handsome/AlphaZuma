"""Train a wide polar policy from persistent raw teacher intent labels."""

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

from tools import build_alphazuma_55_polar_distillation as registration
from tools import distill_alphazuma_55_polar_v1 as base
from tools import distill_alphazuma_55_polar_wide_v1 as wide
from tools import distill_alphazuma_55_settled_v3 as settled
from zuma_rl.alphazuma_55 import INCLUDED_LEVELS


SCRIPT_PATH = Path(__file__).resolve()
EXPECTED_FEATURES_DIM = 1024
EXPECTED_PASSES = 3
EXPECTED_EPOCHS = 30


def _validate_preregistration(path: Path) -> dict[str, Any]:
    prereg = base.legacy._read_json(path)
    if (
        prereg.get("schema")
        != "zuma-rl.alphazuma-55-polar-intent-wide-distillation-preregistration"
        or prereg.get("version") != 1
        or prereg.get("status") != "FROZEN_BEFORE_TRAINING"
    ):
        raise ValueError("unexpected intent-wide preregistration")
    trainer = prereg.get("trainer", {})
    if (
        Path(str(trainer.get("path", ""))).resolve() != SCRIPT_PATH
        or trainer.get("sha256") != base.legacy._sha256(SCRIPT_PATH)
    ):
        raise ValueError("intent-wide trainer binding differs")
    builder = prereg.get("builder", {})
    builder_path = Path(str(builder.get("path", ""))).resolve(strict=True)
    if builder.get("sha256") != base.legacy._sha256(builder_path):
        raise ValueError("intent-wide builder binding differs")
    master = prereg.get("master_preregistration", {})
    master_path = Path(str(master.get("path", ""))).resolve(strict=True)
    if master.get("sha256") != base.legacy._sha256(master_path):
        raise ValueError("intent-wide master bytes differ")
    master_value = base.legacy._read_json(master_path)
    if (
        master_value.get("schema")
        != "zuma-rl.alphazuma-55-weekend-master-preregistration"
    ):
        raise ValueError("unexpected intent-wide master")
    for name, artifact in prereg.get("implementation", {}).items():
        artifact_path = Path(str(artifact.get("path", ""))).resolve(strict=True)
        if artifact.get("sha256") != base.legacy._sha256(artifact_path):
            raise ValueError(f"intent-wide implementation changed: {name}")
    evidence = prereg["teacher_evidence"]["exact_mask_fresh_55"]
    evidence_path = Path(str(evidence["path"])).resolve(strict=True)
    if evidence["sha256"] != base.legacy._sha256(evidence_path):
        raise ValueError("intent-wide teacher evidence changed")
    registration._validate_probe(evidence_path)
    if tuple(prereg.get("levels", ())) != tuple(INCLUDED_LEVELS):
        raise ValueError("intent-wide level order changed")
    run = prereg.get("run", {})
    if not (
        run.get("policy_architecture") == "entity_polar_intent_wide"
        and int(run.get("learned_features_dim", -1)) == EXPECTED_FEATURES_DIM
        and int(run.get("teacher_passes", -1)) == EXPECTED_PASSES
        and int(run.get("epochs_per_round", -1)) == EXPECTED_EPOCHS
        and run.get("raw_teacher_intent_labels") is True
        and run.get("training_mask_relaxation")
        == "unmask_only_the_teacher_target_verb"
        and run.get("exact_action_masks_for_environment_execution") is True
        and run.get("exact_action_masks_for_online_policy_inference") is True
        and run.get("label_aim_must_remain_exact_mask_legal") is True
        and run.get("aim_loss_scope") == "fire_only"
        and run.get("aim_loss_mode") == "categorical"
    ):
        raise ValueError("intent-wide semantic contract changed")
    first = int(run["training_seed_base"])
    last = int(run["training_seed_last_consumed"])
    if last != first + EXPECTED_PASSES * len(INCLUDED_LEVELS) - 1:
        raise ValueError("intent-wide training seed interval changed")
    registry = master_value["seed_registry"]["training"]
    if not int(registry["first"]) <= first <= last <= int(registry["last"]):
        raise ValueError("intent-wide training seeds escaped the registry")
    boundary = prereg.get("authority_boundary", {})
    for name in (
        "current_campaign_candidate_authority",
        "s99081535_successor_candidate_authority",
        "formal_selection_seed_consumption",
        "formal_final_blind_seed_consumption",
        "continuous_campaign_seed_consumption",
    ):
        if boundary.get(name) is not False:
            raise ValueError(f"intent-wide authority changed: {name}")
    return prereg


def _intent_label(
    raw_action: np.ndarray, exact_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    raw = np.asarray(raw_action, dtype=np.int64).reshape(2).copy()
    mask = np.asarray(exact_mask, dtype=np.bool_).reshape(-1)
    effective, forced = settled._effective_label(raw, mask)
    training_mask = mask.copy()
    training_mask[int(raw[0])] = True
    if not bool(training_mask[4 + int(raw[1])]):
        raise RuntimeError("raw teacher aim is masked")
    return raw, effective, training_mask, forced


def _collect_intent_round(
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

    input_config = settled.EliteHumanInputConfig(
        profile_id="elite-human-v1",
        reaction_delay_ticks=12,
        max_aim_speed_degrees_per_second=1080.0,
        max_aim_acceleration_degrees_per_second_squared=18000.0,
        min_button_interval_ticks=5,
    )
    reward_config = settled.WinFirstRewardConfig(
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
            settled.legacy._make_env_factory(
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
                settled.CurveAwareSettledStrategicRevengeTeacher(
                    settled.legacy._teacher_spec(value)
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
            ] = [{name: [] for name in settled.VERB_NAMES} for _ in batch]
            per_episode_forced = np.zeros(len(batch), dtype=np.int64)
            last_infos: list[dict[str, Any]] = [{} for _ in batch]
            for _ in range(max_ticks + 1):
                masks = np.asarray(get_action_masks(vector), dtype=np.bool_)
                actions = np.zeros((len(batch), 2), dtype=np.int64)
                active = ~finished
                for position in range(len(batch)):
                    if finished[position]:
                        continue
                    teacher_raw = np.asarray(
                        teachers[position].act(observations[position]),
                        dtype=np.int64,
                    )
                    raw, effective, training_mask, forced = _intent_label(
                        teacher_raw, masks[position]
                    )
                    raw_name = settled.VERB_NAMES[int(raw[0])]
                    effective_name = settled.VERB_NAMES[int(effective[0])]
                    raw_counts[position][raw_name] += 1
                    effective_counts[position][effective_name] += 1
                    total_raw[raw_name] += 1
                    total_effective[effective_name] += 1
                    actions[position] = effective
                    if forced:
                        per_episode_forced[position] += 1
                        forced_waits += 1
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
                    f"intent-wide collection exceeded episode guard: {missing}"
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
                        "effective_execution_action_counts": dict(
                            effective_counts[position]
                        ),
                        "intent_mask_relaxations": int(
                            per_episode_forced[position]
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
    retained_by_verb = Counter(
        settled.VERB_NAMES[int(row[1][0])] for row in aggregate
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
        "raw_intent_action_counts": dict(total_raw),
        "effective_execution_action_counts": dict(total_effective),
        "intent_mask_relaxations": forced_waits,
        "training_label_semantics": "raw_teacher_intent",
        "environment_execution_semantics": "exact_mask_effective_action",
        "teacher_policy_id": wide.base.POLICY_ID,
    }


def run(*, preregistration_path: Path, original_root: Path) -> dict[str, Any]:
    preregistration_path = preregistration_path.resolve(strict=True)
    prereg = _validate_preregistration(preregistration_path)
    original_validate = wide._validate_preregistration
    original_collector = settled._collect_round
    original_script_path = wide.SCRIPT_PATH
    try:
        wide._validate_preregistration = lambda _: prereg
        settled._collect_round = _collect_intent_round
        wide.SCRIPT_PATH = SCRIPT_PATH
        completion = wide.run(
            preregistration_path=preregistration_path,
            original_root=original_root,
        )
    except BaseException:
        run_dir = Path(str(prereg["run"]["run_dir"])).resolve()
        failure_path = run_dir / "failure.json"
        if failure_path.exists():
            failure = base.legacy._read_json(failure_path)
            failure["schema"] = (
                "zuma-rl.alphazuma-55-polar-intent-wide-"
                "distillation-failure"
            )
            failure["training_label_semantics"] = "raw_teacher_intent"
            base.legacy._write_json_atomic(failure_path, failure)
        raise
    finally:
        wide._validate_preregistration = original_validate
        settled._collect_round = original_collector
        wide.SCRIPT_PATH = original_script_path
    completion.update(
        {
            "schema": (
                "zuma-rl.alphazuma-55-polar-intent-wide-"
                "distillation-completion"
            ),
            "policy_architecture": "entity_polar_intent_wide",
            "training_label_semantics": "raw_teacher_intent",
            "training_mask_relaxation": "unmask_only_the_teacher_target_verb",
            "environment_execution_semantics": "exact_mask_effective_action",
            "online_policy_inference_semantics": "exact_action_masks",
        }
    )
    completion_path = Path(str(prereg["run"]["run_dir"])) / "completion.json"
    base.legacy._write_json_atomic(completion_path, completion)
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
        prototype = wide.base._prototype_environment(
            original_root=original_root,
            max_ticks=int(prereg["run"]["max_ticks"]),
        )
        try:
            model = wide._build_model(prototype, prereg["run"])
            print(
                json.dumps(
                    {
                        "status": "VALID",
                        "preregistration": {
                            "path": str(prereg_path),
                            "sha256": base.legacy._sha256(prereg_path),
                        },
                        "policy_architecture": "entity_polar_intent_wide",
                        "learned_features_dim": int(
                            model.policy.features_extractor.learned_features_dim
                        ),
                        "policy_parameter_count": sum(
                            parameter.numel()
                            for parameter in model.policy.parameters()
                        ),
                        "observation_shape": list(model.observation_space.shape),
                        "action_nvec": [
                            int(value) for value in model.action_space.nvec
                        ],
                        "training_label_semantics": "raw_teacher_intent",
                        "online_action_masks": "exact",
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
                "completion": {
                    "path": str(Path(prereg["run"]["run_dir"]) / "completion.json"),
                    "sha256": base.legacy._sha256(
                        Path(prereg["run"]["run_dir"]) / "completion.json"
                    ),
                },
                "model": completion["final_model"],
                "training_label_semantics": completion[
                    "training_label_semantics"
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
