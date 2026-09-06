"""Run the preregistered AlphaZuma V1 human-speedrun blind matrix."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import statistics
import struct
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from zuma_rl.human_speedrun import (
    EliteHumanInputConfig,
    HumanSpeedrunWrapper,
    WinFirstRewardConfig,
)
from zuma_rl.revenge_env import RevengeEnv, RevengeEnvConfig


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _require_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _load_and_validate_preregistration(path: Path) -> dict[str, Any]:
    prereg = _require_mapping(
        json.loads(path.read_text(encoding="utf-8")),
        "preregistration",
    )
    if prereg.get("schema") != (
        "zuma-rl.human-speedrun-blind-evaluation-preregistration"
    ):
        raise ValueError("unexpected preregistration schema")
    if prereg.get("version") != 1:
        raise ValueError("unsupported preregistration version")
    if prereg.get("status") != "FROZEN_BEFORE_EVALUATION":
        raise ValueError("preregistration is not frozen for evaluation")

    seeds = prereg.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("preregistration seeds must be a non-empty list")
    parsed_seeds = [
        _require_int(value, f"seeds[{index}]", minimum=1)
        for index, value in enumerate(seeds)
    ]
    if len(set(parsed_seeds)) != len(parsed_seeds):
        raise ValueError("preregistration seeds contain duplicates")
    if parsed_seeds != list(range(parsed_seeds[0], parsed_seeds[0] + len(seeds))):
        raise ValueError("this evaluator requires a contiguous ordered seed list")

    models = prereg.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("preregistration models must be a non-empty list")
    model_ids: set[str] = set()
    for index, raw_model in enumerate(models):
        model = _require_mapping(raw_model, f"models[{index}]")
        model_id = model.get("id")
        if not isinstance(model_id, str) or not model_id:
            raise ValueError(f"models[{index}].id must be a non-empty string")
        if model_id in model_ids:
            raise ValueError("preregistration model ids contain duplicates")
        model_ids.add(model_id)
        _require_int(model.get("training_steps"), "model.training_steps")
        model_path = Path(str(model.get("path", "")))
        if not model_path.is_file():
            raise ValueError(f"model file does not exist: {model_path}")
        expected_sha = model.get("sha256")
        if not isinstance(expected_sha, str) or not expected_sha.startswith(
            "sha256:"
        ):
            raise ValueError(f"models[{index}].sha256 is invalid")
        actual_sha = _sha256(model_path)
        if actual_sha != expected_sha:
            raise ValueError(
                f"model hash mismatch for {model_id}: {actual_sha} != {expected_sha}"
            )

    disclosure = _require_mapping(
        prereg.get("selection_disclosure"),
        "selection_disclosure",
    )
    expected_attempts = len(seeds) * len(models)
    if disclosure.get("total_preregistered_attempts") != expected_attempts:
        raise ValueError("preregistered attempt count does not match matrix size")

    execution = _require_mapping(prereg.get("execution"), "execution")
    parallel_envs = _require_int(
        execution.get("parallel_envs"),
        "execution.parallel_envs",
        minimum=1,
    )
    if len(seeds) % parallel_envs:
        raise ValueError("seed count must be divisible by parallel_envs")

    evaluator = _require_mapping(prereg.get("evaluator"), "evaluator")
    evaluator_path = Path(str(evaluator.get("path", "")))
    if evaluator_path.resolve() != Path(__file__).resolve():
        raise ValueError("preregistration binds a different evaluator path")
    expected_evaluator_sha = evaluator.get("sha256")
    if not isinstance(expected_evaluator_sha, str):
        raise ValueError("evaluator hash was not bound before execution")
    actual_evaluator_sha = _sha256(Path(__file__).resolve())
    if actual_evaluator_sha != expected_evaluator_sha:
        raise ValueError(
            "evaluator source hash differs from the preregistered value"
        )
    return prereg


def _build_configs(
    prereg: dict[str, Any],
) -> tuple[RevengeEnvConfig, EliteHumanInputConfig, WinFirstRewardConfig]:
    environment = _require_mapping(prereg["environment"], "environment")
    input_profile = _require_mapping(
        environment["input_profile"],
        "environment.input_profile",
    )
    reward_profile = _require_mapping(
        environment["reward_profile"],
        "environment.reward_profile",
    )
    base_config = RevengeEnvConfig(
        aim_bins=180,
        action_mode="factorized",
        frame_skip=_require_int(environment["frame_skip"], "frame_skip", minimum=1),
        max_ticks=_require_int(environment["max_ticks"], "max_ticks", minimum=1),
        max_balls=_require_int(environment["max_balls"], "max_balls", minimum=160),
        max_projectiles=32,
        max_curves=2,
        score_reward_scale=0.01,
        step_penalty=-0.0001,
        win_reward=10.0,
        loss_reward=-10.0,
    )
    input_config = EliteHumanInputConfig(
        profile_id=str(input_profile["id"]),
        reaction_delay_ticks=_require_int(
            input_profile["reaction_delay_ticks"],
            "reaction_delay_ticks",
        ),
        max_aim_speed_degrees_per_second=float(
            input_profile["max_aim_speed_degrees_per_second"]
        ),
        max_aim_acceleration_degrees_per_second_squared=float(
            input_profile[
                "max_aim_acceleration_degrees_per_second_squared"
            ]
        ),
        min_button_interval_ticks=_require_int(
            input_profile["min_button_interval_ticks"],
            "min_button_interval_ticks",
            minimum=1,
        ),
    )
    reward_config = WinFirstRewardConfig(
        profile_id=str(reward_profile["id"]),
        win_reward=float(reward_profile["win_reward"]),
        failure_reward=float(reward_profile["failure_reward"]),
        time_penalty_per_native_tick=float(
            reward_profile["time_penalty_per_native_tick"]
        ),
        score_progress_reward_cap=float(
            reward_profile["score_progress_reward_cap"]
        ),
    )
    return base_config, input_config, reward_config


def _make_env_factory(
    *,
    original_root: Path,
    level: str,
    profile_mode: str,
    base_config: RevengeEnvConfig,
    input_config: EliteHumanInputConfig,
    reward_config: WinFirstRewardConfig,
) -> Any:
    def make_env() -> HumanSpeedrunWrapper:
        base = RevengeEnv(
            config=base_config,
            level_id=level,
            root=original_root,
            hard=False,
            curve_index=0,
            profile_mode=profile_mode,
        )
        return HumanSpeedrunWrapper(
            base,
            input_config=input_config,
            reward_config=reward_config,
        )

    return make_env


@dataclass(slots=True)
class EpisodeTracker:
    seed: int
    digest: Any = field(default_factory=hashlib.sha256)
    reward: float = 0.0
    decisions: int = 0
    desired_verbs: Counter[int] = field(default_factory=Counter)
    executed_verbs: Counter[int] = field(default_factory=Counter)

    def update(
        self,
        action: np.ndarray,
        reward: float,
        info: dict[str, Any],
    ) -> None:
        desired_verb = int(action[0])
        desired_aim = int(action[1])
        human = _require_mapping(info.get("human_speedrun"), "human_speedrun")
        executed_verb = int(human["executed_verb"])
        executed_aim = int(human["executed_aim_bin"])
        outcome_value = info.get("outcome")
        outcome_code = 1 if outcome_value == "win" else (-1 if outcome_value == "loss" else 0)
        truncated = bool(info.get("TimeLimit.truncated", False))
        self.digest.update(
            struct.pack(
                "<iidqqiiii?",
                desired_verb,
                desired_aim,
                float(reward),
                int(info["ticks"]),
                int(info["score"]),
                executed_verb,
                executed_aim,
                outcome_code,
                int(info.get("score_delta", 0)),
                truncated,
            )
        )
        self.reward += float(reward)
        self.decisions += 1
        self.desired_verbs[desired_verb] += 1
        self.executed_verbs[executed_verb] += 1

    def finish(
        self,
        *,
        info: dict[str, Any],
        model: dict[str, Any],
        threshold_ticks: int,
    ) -> dict[str, Any]:
        outcome = info.get("outcome")
        ticks = int(info["ticks"])
        return {
            "model_id": model["id"],
            "training_steps": int(model["training_steps"]),
            "model_sha256": model["sha256"],
            "seed": self.seed,
            "outcome": outcome,
            "native_outcome": info.get("native_outcome"),
            "time_limit_truncated": bool(
                info.get("TimeLimit.truncated", False)
            ),
            "ticks": ticks,
            "seconds": ticks / 100.0,
            "score": int(info["score"]),
            "reward": self.reward,
            "decisions": self.decisions,
            "desired_action_counts": {
                "wait": self.desired_verbs[0],
                "fire": self.desired_verbs[1],
                "swap": self.desired_verbs[2],
            },
            "executed_action_counts": {
                "wait": self.executed_verbs[0],
                "fire": self.executed_verbs[1],
                "swap": self.executed_verbs[2],
            },
            "trajectory_sha256": "sha256:" + self.digest.hexdigest(),
            "qualifies_sub_12": outcome == "win" and ticks < threshold_ticks,
        }


def _run_batch(
    *,
    vec_env: Any,
    model: Any,
    model_spec: dict[str, Any],
    seeds: Sequence[int],
    threshold_ticks: int,
    stop_after_seed: int | None = None,
) -> tuple[list[dict[str, Any]], int]:
    from sb3_contrib.common.maskable.utils import get_action_masks

    if list(seeds) != list(range(seeds[0], seeds[0] + len(seeds))):
        raise ValueError("batch seeds must be contiguous")
    vec_env.seed(int(seeds[0]))
    observations = vec_env.reset()
    trackers = [EpisodeTracker(seed=int(seed)) for seed in seeds]
    results: list[dict[str, Any] | None] = [None] * len(seeds)
    target_index = (
        list(seeds).index(stop_after_seed)
        if stop_after_seed is not None
        else None
    )
    vector_transitions = 0
    max_steps = int(vec_env.get_attr("config", indices=[0])[0].max_ticks) + 1
    for _ in range(max_steps):
        masks = get_action_masks(vec_env)
        actions, _ = model.predict(
            observations,
            deterministic=True,
            action_masks=masks,
        )
        actions_array = np.asarray(actions, dtype=np.int64).reshape(len(seeds), 2)
        observations, rewards, dones, infos = vec_env.step(actions_array)
        vector_transitions += len(seeds)
        for index, (reward, done, info) in enumerate(
            zip(rewards, dones, infos, strict=True)
        ):
            if results[index] is not None:
                continue
            trackers[index].update(actions_array[index], float(reward), info)
            if bool(done):
                results[index] = trackers[index].finish(
                    info=info,
                    model=model_spec,
                    threshold_ticks=threshold_ticks,
                )
        if target_index is not None and results[target_index] is not None:
            return [results[target_index]], vector_transitions  # type: ignore[list-item]
        if target_index is None and all(result is not None for result in results):
            return [result for result in results if result is not None], vector_transitions
    raise RuntimeError("one or more episodes exceeded the configured max tick guard")


def _model_summary(
    model: dict[str, Any],
    attempts: Sequence[dict[str, Any]],
    *,
    wall_seconds: float,
    vector_transitions: int,
) -> dict[str, Any]:
    wins = [row for row in attempts if row["outcome"] == "win"]
    ticks = [int(row["ticks"]) for row in wins]
    qualifying = [row for row in attempts if row["qualifies_sub_12"]]
    return {
        "model_id": model["id"],
        "training_steps": model["training_steps"],
        "model_sha256": model["sha256"],
        "attempts": len(attempts),
        "wins": len(wins),
        "losses": sum(row["outcome"] == "loss" for row in attempts),
        "truncations": sum(row["time_limit_truncated"] for row in attempts),
        "qualifying_sub_12": len(qualifying),
        "best_ticks": min(ticks) if ticks else None,
        "best_seconds": min(ticks) / 100.0 if ticks else None,
        "median_win_ticks": statistics.median(ticks) if ticks else None,
        "mean_win_ticks": statistics.fmean(ticks) if ticks else None,
        "wall_seconds": wall_seconds,
        "vector_transitions": vector_transitions,
        "effective_transitions_per_second": (
            vector_transitions / wall_seconds if wall_seconds > 0.0 else None
        ),
    }


def _rank_candidates(attempts: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (row for row in attempts if row["qualifies_sub_12"]),
        key=lambda row: (
            int(row["ticks"]),
            -int(row["score"]),
            int(row["training_steps"]),
            int(row["seed"]),
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prereg_path = args.preregistration.expanduser().resolve(strict=True)
    original_root = args.original_root.expanduser().resolve(strict=True)
    prereg = _load_and_validate_preregistration(prereg_path)
    prereg_sha = _sha256(prereg_path)
    evaluator_sha = _sha256(Path(__file__).resolve())
    base_config, input_config, reward_config = _build_configs(prereg)

    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "VALID",
                    "preregistration_sha256": prereg_sha,
                    "evaluator_sha256": evaluator_sha,
                    "models": len(prereg["models"]),
                    "seeds": len(prereg["seeds"]),
                    "attempts": len(prereg["models"]) * len(prereg["seeds"]),
                },
                indent=2,
            )
        )
        return 0

    try:
        import torch
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.vec_env import SubprocVecEnv
    except ImportError as error:
        raise SystemExit("training dependencies are required for evaluation") from error

    torch.set_num_threads(1)
    execution = prereg["execution"]
    output_directory = Path(execution["output_directory"])
    if output_directory.exists():
        raise SystemExit(
            f"blind evaluation output directory already exists: {output_directory}"
        )
    output_directory.mkdir(parents=True)
    output_path = output_directory / "blind_evaluation.json"

    parallel_envs = int(execution["parallel_envs"])
    seeds = [int(seed) for seed in prereg["seeds"]]
    seed_batches = [
        seeds[index : index + parallel_envs]
        for index in range(0, len(seeds), parallel_envs)
    ]
    environment = prereg["environment"]
    factory = _make_env_factory(
        original_root=original_root,
        level=str(environment["level"]),
        profile_mode=str(environment["profile_mode"]),
        base_config=base_config,
        input_config=input_config,
        reward_config=reward_config,
    )
    vec_env = SubprocVecEnv(
        [factory for _ in range(parallel_envs)],
        start_method="forkserver",
    )

    result: dict[str, Any] = {
        "schema": "zuma-rl.human-speedrun-blind-evaluation",
        "version": 1,
        "status": "RUNNING",
        "started_utc": _utc_now(),
        "completed_utc": None,
        "preregistration": {
            "path": str(prereg_path),
            "sha256": prereg_sha,
        },
        "evaluator": {
            "path": str(Path(__file__).resolve()),
            "sha256": evaluator_sha,
        },
        "device": str(execution["device"]),
        "parallel_envs": parallel_envs,
        "human_record_ticks": int(
            prereg["human_record"]["strict_record_threshold_ticks"]
        ),
        "preregistered_attempts": int(
            prereg["selection_disclosure"]["total_preregistered_attempts"]
        ),
        "completed_attempts": 0,
        "model_summaries": [],
        "attempts": [],
        "qualifying_candidates": [],
        "selected_candidate": None,
        "replays": [],
        "decision": None,
    }
    _write_json_atomic(output_path, result)

    threshold_ticks = int(result["human_record_ticks"])
    total_start = time.perf_counter()
    selected_model: Any | None = None
    try:
        for model_index, model_spec in enumerate(prereg["models"], start=1):
            model_start = time.perf_counter()
            model = MaskablePPO.load(
                model_spec["path"],
                device=str(execution["device"]),
            )
            model_attempts: list[dict[str, Any]] = []
            model_transitions = 0
            for batch in seed_batches:
                batch_attempts, transitions = _run_batch(
                    vec_env=vec_env,
                    model=model,
                    model_spec=model_spec,
                    seeds=batch,
                    threshold_ticks=threshold_ticks,
                )
                model_attempts.extend(batch_attempts)
                model_transitions += transitions
            wall_seconds = time.perf_counter() - model_start
            result["attempts"].extend(model_attempts)
            result["model_summaries"].append(
                _model_summary(
                    model_spec,
                    model_attempts,
                    wall_seconds=wall_seconds,
                    vector_transitions=model_transitions,
                )
            )
            result["completed_attempts"] = len(result["attempts"])
            result["qualifying_candidates"] = _rank_candidates(
                result["attempts"]
            )
            _write_json_atomic(output_path, result)
            summary = result["model_summaries"][-1]
            print(
                f"model={model_index}/{len(prereg['models'])} "
                f"id={model_spec['id']} "
                f"wins={summary['wins']}/{summary['attempts']} "
                f"best={summary['best_seconds']}s "
                f"sub12={summary['qualifying_sub_12']} "
                f"wall={wall_seconds:.1f}s",
                flush=True,
            )
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        candidates = _rank_candidates(result["attempts"])
        result["qualifying_candidates"] = candidates
        if candidates:
            selected = candidates[0]
            result["selected_candidate"] = selected
            selected_spec = next(
                model
                for model in prereg["models"]
                if model["id"] == selected["model_id"]
            )
            selected_model = MaskablePPO.load(
                selected_spec["path"],
                device=str(execution["device"]),
            )
            selected_seed = int(selected["seed"])
            selected_batch = next(
                batch for batch in seed_batches if selected_seed in batch
            )
            for repetition in range(
                int(prereg["replay_protocol"]["independent_repetitions"])
            ):
                replay_rows, _ = _run_batch(
                    vec_env=vec_env,
                    model=selected_model,
                    model_spec=selected_spec,
                    seeds=selected_batch,
                    threshold_ticks=threshold_ticks,
                    stop_after_seed=selected_seed,
                )
                replay = replay_rows[0]
                replay["repetition"] = repetition + 1
                replay["identical_to_blind_attempt"] = all(
                    replay[key] == selected[key]
                    for key in (
                        "outcome",
                        "ticks",
                        "score",
                        "trajectory_sha256",
                    )
                )
                result["replays"].append(replay)
                _write_json_atomic(output_path, result)
            replay_pass = all(
                row["identical_to_blind_attempt"] for row in result["replays"]
            )
            result["decision"] = {
                "status": "PASS" if replay_pass else "FAIL",
                "reason": (
                    "sub_12_candidate_replayed_identically_three_times"
                    if replay_pass
                    else "selected_candidate_replay_identity_failed"
                ),
            }
        else:
            result["decision"] = {
                "status": "FAIL",
                "reason": "no_preregistered_attempt_finished_below_1200_ticks",
            }

        result["status"] = "COMPLETE"
        result["completed_utc"] = _utc_now()
        result["wall_seconds"] = time.perf_counter() - total_start
        _write_json_atomic(output_path, result)
        print(json.dumps(result["decision"], ensure_ascii=False), flush=True)
    finally:
        if selected_model is not None:
            del selected_model
        vec_env.close()

    return 0 if result["decision"]["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
