"""Train a frozen V1 policy across a preregistered multilevel curriculum.

The trainer is deliberately receipt-driven.  All levels, seeds, hyperparameters,
deadlines, and output directories come from a frozen preregistration.  A dynamic
environment selects a level at each reset while preserving the original
single-frog observation/action contract and assigning disjoint deterministic
episode seeds.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
import copy
from dataclasses import asdict
from datetime import datetime, timezone
from functools import partial
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from zuma_rl.human_speedrun import (
    EliteHumanInputConfig,
    HumanSpeedrunWrapper,
    WinFirstRewardConfig,
)
from zuma_rl.original_data import OriginalGameCatalog
from zuma_rl.revenge_core import SUPPORTED_PROFILE_MODE
from zuma_rl.revenge_env import RevengeEnv, RevengeEnvConfig


FloatObservation = NDArray[np.float32]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("UTC timestamp must include an offset")
    return parsed.astimezone(timezone.utc)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_positive(value: Any, name: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return parsed


def _normalize_weights(
    levels: Sequence[str],
    weights: Mapping[str, float],
) -> tuple[dict[str, float], NDArray[np.float64]]:
    expected = set(levels)
    if set(weights) != expected:
        missing = sorted(expected - set(weights))
        extra = sorted(set(weights) - expected)
        raise ValueError(f"level weights differ; missing={missing}, extra={extra}")
    parsed: dict[str, float] = {}
    for level in levels:
        value = float(weights[level])
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"invalid weight for {level}: {value}")
        parsed[level] = value
    total = sum(parsed.values())
    if total <= 0.0:
        raise ValueError("at least one level weight must be positive")
    probabilities = np.asarray(
        [parsed[level] / total for level in levels],
        dtype=np.float64,
    )
    return parsed, probabilities


def _configs(
    prereg: dict[str, Any],
) -> tuple[RevengeEnvConfig, EliteHumanInputConfig, WinFirstRewardConfig]:
    environment = _mapping(prereg.get("environment"), "environment")
    base = _mapping(environment.get("base_config"), "environment.base_config")
    input_profile = _mapping(
        environment.get("input_profile"),
        "environment.input_profile",
    )
    reward_profile = _mapping(
        environment.get("reward_profile"),
        "environment.reward_profile",
    )
    return (
        RevengeEnvConfig(**base),
        EliteHumanInputConfig(
            profile_id=str(input_profile["profile_id"]),
            reaction_delay_ticks=int(input_profile["reaction_delay_ticks"]),
            max_aim_speed_degrees_per_second=float(
                input_profile["max_aim_speed_degrees_per_second"]
            ),
            max_aim_acceleration_degrees_per_second_squared=float(
                input_profile[
                    "max_aim_acceleration_degrees_per_second_squared"
                ]
            ),
            min_button_interval_ticks=int(
                input_profile["min_button_interval_ticks"]
            ),
        ),
        WinFirstRewardConfig(
            profile_id=str(reward_profile["profile_id"]),
            win_reward=float(reward_profile["win_reward"]),
            failure_reward=float(reward_profile["failure_reward"]),
            time_penalty_per_native_tick=float(
                reward_profile["time_penalty_per_native_tick"]
            ),
            score_progress_reward_cap=float(
                reward_profile["score_progress_reward_cap"]
            ),
        ),
    )


def _make_human_env(
    *,
    original_root: Path,
    level_id: str,
    base_config: RevengeEnvConfig,
    input_config: EliteHumanInputConfig,
    reward_config: WinFirstRewardConfig,
) -> HumanSpeedrunWrapper:
    base = RevengeEnv(
        config=base_config,
        level_id=level_id,
        root=original_root,
        hard=False,
        curve_index=0,
        profile_mode=SUPPORTED_PROFILE_MODE,
    )
    return HumanSpeedrunWrapper(
        base,
        input_config=input_config,
        reward_config=reward_config,
    )


class MultiLevelHumanSpeedrunEnv(gym.Env[FloatObservation, Any]):
    """Select one preregistered compatible level at each episode reset."""

    metadata = RevengeEnv.metadata

    def __init__(
        self,
        *,
        original_root: Path,
        levels: Sequence[str],
        weights: Mapping[str, float],
        base_config: RevengeEnvConfig,
        input_config: EliteHumanInputConfig,
        reward_config: WinFirstRewardConfig,
        selector_seed: int,
        episode_seed_base: int,
        episode_seed_stride: int,
        worker_rank: int,
    ) -> None:
        super().__init__()
        if not levels or len(set(levels)) != len(levels):
            raise ValueError("levels must be non-empty and unique")
        if worker_rank < 0:
            raise ValueError("worker_rank cannot be negative")
        if episode_seed_stride < 1:
            raise ValueError("episode_seed_stride must be positive")
        self.original_root = original_root
        self.levels = tuple(str(level) for level in levels)
        self.base_config = base_config
        self.input_config = input_config
        self.reward_config = reward_config
        self.selector_seed = int(selector_seed)
        self.episode_seed_base = int(episode_seed_base)
        self.episode_seed_stride = int(episode_seed_stride)
        self.worker_rank = int(worker_rank)
        self._weights, self._probabilities = _normalize_weights(
            self.levels,
            weights,
        )
        self._selector = np.random.default_rng(self.selector_seed)
        self._episode_index = 0
        self._active_level = self.levels[0]
        self._active_episode_seed: int | None = None
        self._active = _make_human_env(
            original_root=self.original_root,
            level_id=self._active_level,
            base_config=self.base_config,
            input_config=self.input_config,
            reward_config=self.reward_config,
        )
        self.observation_space = self._active.observation_space
        self.action_space = self._active.action_space

    def _replace_active(self, level_id: str) -> None:
        if level_id == self._active_level:
            return
        self._active.close()
        replacement = _make_human_env(
            original_root=self.original_root,
            level_id=level_id,
            base_config=self.base_config,
            input_config=self.input_config,
            reward_config=self.reward_config,
        )
        if replacement.observation_space != self.observation_space:
            replacement.close()
            raise RuntimeError(f"observation space changed for {level_id}")
        if replacement.action_space != self.action_space:
            replacement.close()
            raise RuntimeError(f"action space changed for {level_id}")
        self._active = replacement
        self._active_level = level_id

    def set_level_weights(self, weights: Mapping[str, float]) -> dict[str, float]:
        self._weights, self._probabilities = _normalize_weights(
            self.levels,
            weights,
        )
        return dict(self._weights)

    def get_curriculum_state(self) -> dict[str, Any]:
        return {
            "worker_rank": self.worker_rank,
            "episode_index": self._episode_index,
            "active_level": self._active_level,
            "active_episode_seed": self._active_episode_seed,
            "weights": dict(self._weights),
        }

    def action_masks(self) -> NDArray[np.bool_]:
        return self._active.action_masks()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[FloatObservation, dict[str, Any]]:
        if seed is not None:
            # SB3 supplies a deterministic per-worker seed on the first reset.
            # Fold it into the preregistered selector seed without using it for
            # simulation RNG, whose namespace is independently reserved below.
            self._selector = np.random.default_rng(
                np.random.SeedSequence([self.selector_seed, int(seed)])
            )
        selected_index = int(
            self._selector.choice(len(self.levels), p=self._probabilities)
        )
        level_id = self.levels[selected_index]
        self._replace_active(level_id)
        if self._episode_index >= self.episode_seed_stride:
            raise RuntimeError("worker exhausted its preregistered seed namespace")
        episode_seed = (
            self.episode_seed_base
            + self.worker_rank * self.episode_seed_stride
            + self._episode_index
        )
        self._active_episode_seed = episode_seed
        episode_index = self._episode_index
        self._episode_index += 1
        observation, info = self._active.reset(seed=episode_seed, options=options)
        result = dict(info)
        result.update(
            {
                "training_level_id": level_id,
                "training_episode_seed": episode_seed,
                "training_episode_index": episode_index,
                "training_worker_rank": self.worker_rank,
            }
        )
        return observation, result

    def step(
        self,
        action: Any,
    ) -> tuple[FloatObservation, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self._active.step(action)
        result = dict(info)
        result.update(
            {
                "training_level_id": self._active_level,
                "training_episode_seed": self._active_episode_seed,
                "training_episode_index": self._episode_index - 1,
                "training_worker_rank": self.worker_rank,
            }
        )
        return observation, reward, terminated, truncated, result

    def render(self) -> Any:
        return self._active.render()

    def close(self) -> None:
        self._active.close()


def _make_worker(
    *,
    original_root: Path,
    levels: Sequence[str],
    weights: Mapping[str, float],
    base_config: RevengeEnvConfig,
    input_config: EliteHumanInputConfig,
    reward_config: WinFirstRewardConfig,
    selector_seed: int,
    episode_seed_base: int,
    episode_seed_stride: int,
    worker_rank: int,
) -> gym.Env:
    from stable_baselines3.common.monitor import Monitor

    environment = MultiLevelHumanSpeedrunEnv(
        original_root=original_root,
        levels=levels,
        weights=weights,
        base_config=base_config,
        input_config=input_config,
        reward_config=reward_config,
        selector_seed=selector_seed + worker_rank,
        episode_seed_base=episode_seed_base,
        episode_seed_stride=episode_seed_stride,
        worker_rank=worker_rank,
    )
    return Monitor(environment)


def _runtime() -> dict[str, Any]:
    import torch

    packages: dict[str, str] = {}
    for name in ("zuma-rl", "stable-baselines3", "sb3-contrib", "torch"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "unknown"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_devices": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
    }


def _validate_preregistration(path: Path) -> dict[str, Any]:
    prereg = _read_json(path)
    if prereg.get("schema") != "zuma-rl.overnight-multilevel-preregistration":
        raise ValueError("unexpected preregistration schema")
    if prereg.get("version") != 1:
        raise ValueError("unsupported preregistration version")
    if prereg.get("status") != "FROZEN_BEFORE_TRAINING":
        raise ValueError("preregistration is not frozen before training")
    trainer = _mapping(prereg.get("trainer"), "trainer")
    if Path(str(trainer.get("path", ""))).resolve() != Path(__file__).resolve():
        raise ValueError("preregistration binds another trainer path")
    if trainer.get("sha256") != _sha256(Path(__file__).resolve()):
        raise ValueError("trainer bytes differ from preregistration")
    implementation_raw = prereg.get("implementation")
    if implementation_raw is not None:
        implementation = _mapping(implementation_raw, "implementation")
        for name, raw in implementation.items():
            artifact = _mapping(raw, f"implementation.{name}")
            artifact_path = Path(str(artifact.get("path", ""))).resolve(strict=True)
            if artifact.get("sha256") != _sha256(artifact_path):
                raise ValueError(f"implementation bytes differ: {name}")
    initial_model = _mapping(prereg.get("initial_model"), "initial_model")
    model_path = Path(str(initial_model.get("path", ""))).resolve(strict=True)
    if initial_model.get("sha256") != _sha256(model_path):
        raise ValueError("initial model bytes differ from preregistration")
    levels = prereg.get("levels")
    if not isinstance(levels, list) or not levels:
        raise ValueError("levels must be a non-empty list")
    ids = [str(_mapping(row, "level").get("id", "")) for row in levels]
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("level ids are empty or duplicated")
    runs = prereg.get("runs")
    if not isinstance(runs, list) or len(runs) < 1:
        raise ValueError("runs must be a non-empty list")
    run_ids: set[str] = set()
    run_directories: set[str] = set()
    episode_ranges: list[tuple[int, int, str]] = []
    for index, raw in enumerate(runs):
        run = _mapping(raw, f"runs[{index}]")
        run_id = str(run.get("id", ""))
        if not run_id or run_id in run_ids:
            raise ValueError("run ids are empty or duplicated")
        run_ids.add(run_id)
        run_directory = str(Path(str(run.get("run_dir", ""))).resolve())
        if run_directory in run_directories:
            raise ValueError("run directories are duplicated")
        run_directories.add(run_directory)
        num_envs = _positive_int(run.get("num_envs"), "run.num_envs")
        rollout_steps = _positive_int(run.get("rollout_steps"), "rollout_steps")
        batch_size = _positive_int(run.get("batch_size"), "batch_size")
        if batch_size > num_envs * rollout_steps:
            raise ValueError("batch size exceeds rollout size")
        for name in (
            "total_steps",
            "ppo_epochs",
            "checkpoint_every",
            "episode_seed_base",
            "episode_seed_stride",
        ):
            _positive_int(run.get(name), f"run.{name}")
        _finite_positive(run.get("learning_rate"), "run.learning_rate")
        entropy = float(run.get("entropy_coef"))
        if not math.isfinite(entropy) or entropy < 0.0:
            raise ValueError("entropy coefficient must be finite and non-negative")
        weights = _mapping(run.get("initial_weights"), "run.initial_weights")
        _normalize_weights(ids, weights)
        resume_curriculum_raw = run.get("resume_curriculum")
        if resume_curriculum_raw is not None:
            resume_curriculum = _mapping(
                resume_curriculum_raw,
                "run.resume_curriculum",
            )
            _normalize_weights(
                ids,
                _mapping(resume_curriculum.get("weights"), "resume_curriculum.weights"),
            )
        resume_raw = run.get("resume_model")
        if resume_raw is not None:
            resume = _mapping(resume_raw, "run.resume_model")
            resume_path = Path(str(resume.get("path", ""))).resolve(strict=True)
            if resume.get("sha256") != _sha256(resume_path):
                raise ValueError("resume model bytes differ from preregistration")
            _positive_int(resume.get("timesteps"), "run.resume_model.timesteps")
        base = int(run["episode_seed_base"])
        stride = int(run["episode_seed_stride"])
        episode_ranges.append((base, base + num_envs * stride - 1, run_id))
    for left_index, left in enumerate(episode_ranges):
        for right in episode_ranges[left_index + 1 :]:
            if max(left[0], right[0]) <= min(left[1], right[1]):
                raise ValueError(
                    f"training seed namespaces overlap: {left[2]} and {right[2]}"
                )
    schedule = _mapping(prereg.get("schedule"), "schedule")
    stop = _parse_utc(str(schedule.get("training_stop_utc", "")))
    end = _parse_utc(str(schedule.get("goal_end_utc", "")))
    if stop >= end:
        raise ValueError("training stop must precede goal end")
    return prereg


def _validate_catalog(
    *,
    prereg: dict[str, Any],
    original_root: Path,
) -> None:
    catalog = OriginalGameCatalog(original_root)
    by_id = {definition.id.casefold(): definition for definition in catalog.levels.values()}
    for frozen in prereg["levels"]:
        definition = by_id.get(str(frozen["id"]).casefold())
        if definition is None:
            raise ValueError(f"level is absent from installed assets: {frozen['id']}")
        loaded = catalog.load_level(definition.id, hard=False)
        actual = {
            "display_name": definition.display_name,
            "curve_count": len(loaded.curves),
            "curve_names": list(definition.curve_names),
        }
        for field, value in actual.items():
            if frozen[field] != value:
                raise ValueError(f"installed metadata changed for {frozen['id']}: {field}")


def _prevalidate_spaces(
    *,
    prereg: dict[str, Any],
    original_root: Path,
    base_config: RevengeEnvConfig,
    input_config: EliteHumanInputConfig,
    reward_config: WinFirstRewardConfig,
) -> dict[str, Any]:
    from sb3_contrib import MaskablePPO

    model_path = Path(str(prereg["initial_model"]["path"])).resolve(strict=True)
    source = MaskablePPO.load(model_path, device="cpu")
    source_observation_shape = list(source.observation_space.shape)
    source_action_nvec = source.action_space.nvec.tolist()
    resume_models: list[dict[str, Any]] = []
    for run in prereg["runs"]:
        resume_raw = run.get("resume_model")
        if resume_raw is None:
            continue
        resume = _mapping(resume_raw, "run.resume_model")
        resume_path = Path(str(resume["path"])).resolve(strict=True)
        resumed = MaskablePPO.load(resume_path, device="cpu")
        try:
            if resumed.observation_space != source.observation_space:
                raise ValueError("resume model observation space differs from baseline")
            if resumed.action_space != source.action_space:
                raise ValueError("resume model action space differs from baseline")
            if int(resumed.num_timesteps) != int(resume["timesteps"]):
                raise ValueError("resume model timestep differs from preregistration")
            resume_models.append(
                {
                    "run_id": str(run["id"]),
                    "path": str(resume_path),
                    "sha256": _sha256(resume_path),
                    "timesteps": int(resumed.num_timesteps),
                    "optimizer_state_entries": len(
                        resumed.policy.optimizer.state_dict().get("state", {})
                    ),
                }
            )
        finally:
            del resumed
    contracts: list[dict[str, Any]] = []
    try:
        for frozen in prereg["levels"]:
            environment = _make_human_env(
                original_root=original_root,
                level_id=str(frozen["id"]),
                base_config=base_config,
                input_config=input_config,
                reward_config=reward_config,
            )
            try:
                if environment.observation_space != source.observation_space:
                    raise ValueError(
                        f"model observation-space mismatch for {frozen['id']}"
                    )
                if environment.action_space != source.action_space:
                    raise ValueError(
                        f"model action-space mismatch for {frozen['id']}"
                    )
                contracts.append(environment.contract())
            finally:
                environment.close()
    finally:
        del source
    if any(contract != contracts[0] for contract in contracts[1:]):
        raise ValueError("human-speedrun contract differs between levels")
    return {
        "levels_verified": len(contracts),
        "observation_shape": source_observation_shape,
        "action_nvec": source_action_nvec,
        "contract": contracts[0],
        "resume_models": resume_models,
    }


def _adaptive_weights(
    *,
    levels: Sequence[str],
    initial: Mapping[str, float],
    fallback: Mapping[str, float] | None = None,
    recent: Mapping[str, deque[int]],
    anchor_level: str,
    minimum_weight: float,
    maximum_weight: float,
    minimum_episodes: int,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for level in levels:
        outcomes = recent[level]
        base = float(initial[level])
        if len(outcomes) < minimum_episodes:
            value = float(fallback[level]) if fallback is not None else base
        else:
            wins = sum(outcomes)
            # Beta(1, 3) is a conservative 25% prior.  Failed levels rise in
            # sampling probability while successful levels retain a floor.
            smoothed_win_rate = (1.0 + wins) / (4.0 + len(outcomes))
            difficulty = 1.0 + 3.0 * (1.0 - smoothed_win_rate)
            value = 0.35 * base + 0.65 * difficulty
        if level.casefold() == anchor_level.casefold():
            value = max(value, 2.5)
        result[level] = min(maximum_weight, max(minimum_weight, value))
    return result


class OvernightTrainingCallback:  # Runtime subclassed after SB3 import.
    pass


def _build_callback_class() -> type[Any]:
    from stable_baselines3.common.callbacks import BaseCallback

    class _OvernightTrainingCallback(BaseCallback):
        def __init__(
            self,
            *,
            run_dir: Path,
            run_spec: dict[str, Any],
            levels: Sequence[str],
            stop_at: datetime,
        ) -> None:
            super().__init__(verbose=0)
            self.run_dir = run_dir
            self.run_spec = run_spec
            self.levels = tuple(levels)
            self.stop_at = stop_at
            resume = run_spec.get("resume_model")
            self.source_timesteps = (
                int(_mapping(resume, "run.resume_model")["timesteps"])
                if resume is not None
                else 0
            )
            self.started_monotonic = time.monotonic()
            self.started_utc = _utc_now()
            self.episodes = defaultdict(int)
            self.wins = defaultdict(int)
            self.losses = defaultdict(int)
            self.truncations = defaultdict(int)
            adaptive = _mapping(run_spec.get("adaptive"), "run.adaptive")
            self.adaptive_enabled = bool(adaptive.get("enabled"))
            self.adapt_every = int(adaptive.get("update_every_steps", 1))
            self.next_adapt = (
                self.source_timesteps // self.adapt_every + 1
            ) * self.adapt_every
            self.recent_window = int(adaptive.get("rolling_episodes", 64))
            self.minimum_episodes = int(adaptive.get("minimum_episodes", 4))
            self.minimum_weight = float(adaptive.get("minimum_weight", 1.0))
            self.maximum_weight = float(adaptive.get("maximum_weight", 4.0))
            self.anchor_level = str(adaptive.get("anchor_level", "Jungle2"))
            self.initial_weights = {
                key: float(value)
                for key, value in run_spec["initial_weights"].items()
            }
            resume_curriculum_raw = run_spec.get("resume_curriculum")
            self.current_weights = (
                {
                    key: float(value)
                    for key, value in _mapping(
                        _mapping(
                            resume_curriculum_raw,
                            "run.resume_curriculum",
                        ).get("weights"),
                        "resume_curriculum.weights",
                    ).items()
                }
                if resume_curriculum_raw is not None
                else dict(self.initial_weights)
            )
            self.fallback_weights = dict(self.current_weights)
            self.recent = {
                level: deque(maxlen=self.recent_window) for level in self.levels
            }
            self.history_path = run_dir / "curriculum_history.jsonl"
            self.status_path = run_dir / "training_status.json"
            self.last_status_monotonic = 0.0
            self.stop_reason: str | None = None

        def _status(self) -> dict[str, Any]:
            elapsed = time.monotonic() - self.started_monotonic
            steps_this_run = max(0, int(self.num_timesteps) - self.source_timesteps)
            per_level = {}
            for level in self.levels:
                episodes = self.episodes[level]
                per_level[level] = {
                    "episodes": episodes,
                    "wins": self.wins[level],
                    "losses": self.losses[level],
                    "truncations": self.truncations[level],
                    "observed_win_rate": (
                        self.wins[level] / episodes if episodes else None
                    ),
                    "recent_episodes": len(self.recent[level]),
                    "recent_win_rate": (
                        sum(self.recent[level]) / len(self.recent[level])
                        if self.recent[level]
                        else None
                    ),
                    "current_weight": self.current_weights[level],
                }
            return {
                "schema": "zuma-rl.overnight-multilevel-training-status",
                "version": 1,
                "status": "STOPPING" if self.stop_reason else "RUNNING",
                "run_id": self.run_spec["id"],
                "started_utc": self.started_utc,
                "updated_utc": _utc_now(),
                "stop_at_utc": self.stop_at.isoformat(),
                "stop_reason": self.stop_reason,
                "timesteps": int(self.num_timesteps),
                "source_timesteps": self.source_timesteps,
                "steps_this_run": steps_this_run,
                "elapsed_seconds": elapsed,
                "steps_per_second": steps_this_run / elapsed if elapsed else 0.0,
                "episodes": sum(self.episodes.values()),
                "wins": sum(self.wins.values()),
                "losses": sum(self.losses.values()),
                "truncations": sum(self.truncations.values()),
                "weights": dict(self.current_weights),
                "per_level": per_level,
            }

        def _publish(self, *, force: bool = False) -> None:
            now = time.monotonic()
            if not force and now - self.last_status_monotonic < 30.0:
                return
            _write_json_atomic(self.status_path, self._status())
            self.last_status_monotonic = now

        def _record_history(self, reason: str) -> None:
            row = {
                "utc": _utc_now(),
                "reason": reason,
                "timesteps": int(self.num_timesteps),
                "weights": dict(self.current_weights),
                "episodes": dict(self.episodes),
                "wins": dict(self.wins),
            }
            with self.history_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False))
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())

        def _on_training_start(self) -> None:
            self._record_history("training_start")
            self._publish(force=True)

        def _on_step(self) -> bool:
            infos = self.locals.get("infos", ())
            dones = self.locals.get("dones", ())
            for info, done in zip(infos, dones, strict=True):
                if not bool(done):
                    continue
                level = str(info.get("training_level_id", ""))
                if level not in self.recent:
                    raise RuntimeError(f"terminal info has unknown level: {level!r}")
                win = info.get("outcome") == "win"
                truncated = bool(info.get("TimeLimit.truncated", False))
                self.episodes[level] += 1
                self.wins[level] += int(win)
                self.truncations[level] += int(truncated)
                self.losses[level] += int(not win and not truncated)
                self.recent[level].append(int(win))

            if self.adaptive_enabled and self.num_timesteps >= self.next_adapt:
                self.current_weights = _adaptive_weights(
                    levels=self.levels,
                    initial=self.initial_weights,
                    fallback=self.fallback_weights,
                    recent=self.recent,
                    anchor_level=self.anchor_level,
                    minimum_weight=self.minimum_weight,
                    maximum_weight=self.maximum_weight,
                    minimum_episodes=self.minimum_episodes,
                )
                self.training_env.env_method(
                    "set_level_weights",
                    dict(self.current_weights),
                )
                self._record_history("adaptive_update")
                while self.next_adapt <= self.num_timesteps:
                    self.next_adapt += self.adapt_every

            if datetime.now(timezone.utc) >= self.stop_at:
                self.stop_reason = "PREREGISTERED_WALL_CLOCK_DEADLINE"
                self._publish(force=True)
                return False
            self._publish()
            return True

        def _on_training_end(self) -> None:
            if self.stop_reason is None:
                self.stop_reason = "REQUESTED_TIMESTEPS_REACHED"
            self._record_history("training_end")
            self._publish(force=True)

    return _OvernightTrainingCallback


def _run_training(
    *,
    prereg_path: Path,
    prereg: dict[str, Any],
    run_spec: dict[str, Any],
    original_root: Path,
    validation: dict[str, Any],
) -> dict[str, Any]:
    import torch
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv

    def state_equal(left: Any, right: Any) -> bool:
        if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
            return torch.equal(left.detach().cpu(), right.detach().cpu())
        if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
            return np.array_equal(left, right)
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            return set(left) == set(right) and all(
                state_equal(left[key], right[key]) for key in left
            )
        if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
            return len(left) == len(right) and all(
                state_equal(a, b) for a, b in zip(left, right, strict=True)
            )
        return bool(left == right)

    torch.set_num_threads(1)
    levels = [str(row["id"]) for row in prereg["levels"]]
    base_config, input_config, reward_config = _configs(prereg)
    run_dir = Path(str(run_spec["run_dir"])).resolve()
    if run_dir.exists():
        raise FileExistsError(f"refusing to reuse run directory: {run_dir}")
    run_dir.mkdir(parents=True)
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir()

    factories = [
        partial(
            _make_worker,
            original_root=original_root,
            levels=levels,
            weights=(
                _mapping(
                    _mapping(
                        run_spec["resume_curriculum"],
                        "run.resume_curriculum",
                    ).get("weights"),
                    "resume_curriculum.weights",
                )
                if run_spec.get("resume_curriculum") is not None
                else run_spec["initial_weights"]
            ),
            base_config=base_config,
            input_config=input_config,
            reward_config=reward_config,
            selector_seed=int(run_spec["seed"]),
            episode_seed_base=int(run_spec["episode_seed_base"]),
            episode_seed_stride=int(run_spec["episode_seed_stride"]),
            worker_rank=rank,
        )
        for rank in range(int(run_spec["num_envs"]))
    ]
    train_env = SubprocVecEnv(factories, start_method="forkserver")
    started_utc = _utc_now()
    started_monotonic = time.monotonic()
    initial_path = Path(str(prereg["initial_model"]["path"])).resolve(strict=True)
    resume_raw = run_spec.get("resume_model")
    resume_spec = (
        _mapping(resume_raw, "run.resume_model")
        if resume_raw is not None
        else None
    )
    source_path = (
        Path(str(resume_spec["path"])).resolve(strict=True)
        if resume_spec is not None
        else initial_path
    )
    source_timesteps = int(resume_spec["timesteps"]) if resume_spec else 0
    receipt = {
        "schema": "zuma-rl.overnight-multilevel-training-config",
        "version": 1,
        "status": "STARTED",
        "started_utc": started_utc,
        "preregistration": {
            "path": str(prereg_path),
            "sha256": _sha256(prereg_path),
        },
        "trainer_sha256": _sha256(Path(__file__).resolve()),
        "run": run_spec,
        "initial_model": prereg["initial_model"],
        "training_source_model": {
            "path": str(source_path),
            "sha256": _sha256(source_path),
            "timesteps": source_timesteps,
            "mode": "resume" if resume_spec is not None else "fresh",
        },
        "levels": prereg["levels"],
        "environment": prereg["environment"],
        "prevalidation": validation,
        "runtime": _runtime(),
    }
    _write_json_atomic(run_dir / "config.json", receipt)

    callback_class = _build_callback_class()
    stop_at = _parse_utc(prereg["schedule"]["training_stop_utc"])
    telemetry_callback = callback_class(
        run_dir=run_dir,
        run_spec=run_spec,
        levels=levels,
        stop_at=stop_at,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(
            1,
            int(run_spec["checkpoint_every"]) // int(run_spec["num_envs"]),
        ),
        save_path=str(checkpoints),
        name_prefix=f"{run_spec['id']}",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    completion: dict[str, Any]
    try:
        source = MaskablePPO.load(source_path, device="cpu")
        if resume_spec is None:
            model = MaskablePPO(
                "MlpPolicy",
                train_env,
                learning_rate=float(run_spec["learning_rate"]),
                n_steps=int(run_spec["rollout_steps"]),
                batch_size=int(run_spec["batch_size"]),
                n_epochs=int(run_spec["ppo_epochs"]),
                gamma=0.995,
                gae_lambda=0.95,
                ent_coef=float(run_spec["entropy_coef"]),
                policy_kwargs=copy.deepcopy(source.policy_kwargs),
                tensorboard_log=str(run_dir / "tensorboard"),
                seed=int(run_spec["seed"]),
                device=str(run_spec["device"]),
                verbose=1,
            )
            source_state = source.policy.state_dict()
            model.policy.load_state_dict(source_state, strict=True)
            optimizer_state_restored = False
        else:
            if int(source.num_timesteps) != source_timesteps:
                raise RuntimeError(
                    "resume checkpoint timestep differs from preregistration"
                )
            model = MaskablePPO.load(
                source_path,
                env=train_env,
                device=str(run_spec["device"]),
                tensorboard_log=str(run_dir / "tensorboard"),
            )
            expected_hyperparameters = {
                "n_steps": int(run_spec["rollout_steps"]),
                "batch_size": int(run_spec["batch_size"]),
                "n_epochs": int(run_spec["ppo_epochs"]),
                "gamma": 0.995,
                "gae_lambda": 0.95,
                "ent_coef": float(run_spec["entropy_coef"]),
            }
            actual_hyperparameters = {
                key: getattr(model, key) for key in expected_hyperparameters
            }
            if actual_hyperparameters != expected_hyperparameters:
                raise RuntimeError(
                    "resume checkpoint hyperparameters differ from preregistration"
                )
            if float(model.lr_schedule(1.0)) != float(run_spec["learning_rate"]):
                raise RuntimeError(
                    "resume checkpoint learning rate differs from preregistration"
                )
            if not state_equal(
                source.policy.optimizer.state_dict(),
                model.policy.optimizer.state_dict(),
            ):
                raise RuntimeError("optimizer state did not restore bitwise")
            source_state = source.policy.state_dict()
            optimizer_state_restored = True
            model.verbose = 1
            model.seed = int(run_spec["seed"])
            model.set_random_seed(int(run_spec["seed"]))
        copied_exactly = all(
            torch.equal(source_state[name].detach().cpu(), value.detach().cpu())
            for name, value in model.policy.state_dict().items()
        )
        if not copied_exactly:
            raise RuntimeError("initial policy parameters did not copy bitwise")
        _write_json_atomic(
            run_dir / "initialization.json",
            {
                "schema": "zuma-rl.overnight-multilevel-initialization",
                "version": 1,
                "source_model_sha256": _sha256(initial_path),
                "training_source_model_sha256": _sha256(source_path),
                "source_timesteps": source_timesteps,
                "policy_parameters": sum(
                    parameter.numel() for parameter in model.policy.parameters()
                ),
                "policy_copied_bitwise": True,
                "optimizer_state_reset": resume_spec is None,
                "optimizer_state_restored_bitwise": optimizer_state_restored,
                "optimizer_state_entries": len(
                    model.policy.optimizer.state_dict().get("state", {})
                ),
            },
        )
        model.learn(
            total_timesteps=int(run_spec["total_steps"]),
            callback=CallbackList([checkpoint_callback, telemetry_callback]),
            reset_num_timesteps=resume_spec is None,
            progress_bar=False,
        )
        final_path = run_dir / "final_model.zip"
        model.save(final_path)
        completion = {
            "schema": "zuma-rl.overnight-multilevel-completion",
            "version": 1,
            "status": "COMPLETE",
            "completed_utc": _utc_now(),
            "elapsed_seconds": time.monotonic() - started_monotonic,
            "stop_reason": telemetry_callback.stop_reason,
            "requested_steps": int(run_spec["total_steps"]),
            "source_steps": source_timesteps,
            "actual_steps": int(model.num_timesteps),
            "actual_steps_this_run": int(model.num_timesteps) - source_timesteps,
            "final_model": {
                "path": str(final_path),
                "sha256": _sha256(final_path),
                "bytes": final_path.stat().st_size,
            },
            "training_status": _read_json(run_dir / "training_status.json"),
        }
        _write_json_atomic(run_dir / "completion.json", completion)
    except BaseException as error:
        failure = {
            "schema": "zuma-rl.overnight-multilevel-failure",
            "version": 1,
            "status": "FAILED",
            "failed_utc": _utc_now(),
            "elapsed_seconds": time.monotonic() - started_monotonic,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        _write_json_atomic(run_dir / "failure.json", failure)
        raise
    finally:
        train_env.close()
    return completion


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prereg_path = args.preregistration.expanduser().resolve(strict=True)
    original_root = args.original_root.expanduser().resolve(strict=True)
    prereg = _validate_preregistration(prereg_path)
    _validate_catalog(prereg=prereg, original_root=original_root)
    base_config, input_config, reward_config = _configs(prereg)
    validation = _prevalidate_spaces(
        prereg=prereg,
        original_root=original_root,
        base_config=base_config,
        input_config=input_config,
        reward_config=reward_config,
    )
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "VALID",
                    "preregistration_sha256": _sha256(prereg_path),
                    "trainer_sha256": _sha256(Path(__file__).resolve()),
                    **validation,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if not args.run_id:
        raise SystemExit("--run-id is required unless --validate-only is used")
    matches = [run for run in prereg["runs"] if run["id"] == args.run_id]
    if len(matches) != 1:
        raise SystemExit(f"unknown or duplicate run id: {args.run_id}")
    completion = _run_training(
        prereg_path=prereg_path,
        prereg=prereg,
        run_spec=matches[0],
        original_root=original_root,
        validation=validation,
    )
    print(json.dumps(completion, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
