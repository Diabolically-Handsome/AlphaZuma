"""Run the frozen multilevel trainer with a shared, cached retail catalog.

The PPO algorithm, receipt validation, callback, and checkpoint semantics stay
in ``train_overnight_multilevel.py``.  This campaign-specific launcher replaces
only the worker environment factory so one worker parses ``main.pak`` once and
reuses a small LRU of level environments across episode resets.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from tools import train_overnight_multilevel as frozen_trainer
from zuma_rl.human_speedrun import (
    EliteHumanInputConfig,
    HumanSpeedrunWrapper,
    WinFirstRewardConfig,
)
from zuma_rl.original_data import LoadedOriginalLevel, OriginalGameCatalog
from zuma_rl.revenge_core import SUPPORTED_PROFILE_MODE
from zuma_rl.revenge_env import FloatObservation, RevengeEnv, RevengeEnvConfig


class CachedOriginalGameCatalog(OriginalGameCatalog):
    """Cache immutable decoded level assets inside one training worker."""

    def __init__(self, root: str | Path, *, maximum_levels: int = 55):
        super().__init__(root)
        if maximum_levels < 1:
            raise ValueError("maximum_levels must be positive")
        self.maximum_levels = int(maximum_levels)
        self._loaded_levels: OrderedDict[
            tuple[str, bool], LoadedOriginalLevel
        ] = OrderedDict()

    def load_level(
        self,
        level_id: str,
        *,
        hard: bool = False,
    ) -> LoadedOriginalLevel:
        key = (str(level_id).casefold(), bool(hard))
        cached = self._loaded_levels.get(key)
        if cached is not None:
            self._loaded_levels.move_to_end(key)
            return cached
        loaded = super().load_level(level_id, hard=hard)
        self._loaded_levels[key] = loaded
        if len(self._loaded_levels) > self.maximum_levels:
            self._loaded_levels.popitem(last=False)
        return loaded

    @property
    def loaded_level_count(self) -> int:
        return len(self._loaded_levels)


def _make_human_env(
    *,
    catalog: CachedOriginalGameCatalog,
    level_id: str,
    base_config: RevengeEnvConfig,
    input_config: EliteHumanInputConfig,
    reward_config: WinFirstRewardConfig,
) -> HumanSpeedrunWrapper:
    base = RevengeEnv(
        config=base_config,
        level_id=level_id,
        catalog=catalog,
        hard=False,
        curve_index=0,
        profile_mode=SUPPORTED_PROFILE_MODE,
    )
    return HumanSpeedrunWrapper(
        base,
        input_config=input_config,
        reward_config=reward_config,
    )


class SharedCatalogMultiLevelEnv(gym.Env[FloatObservation, Any]):
    """Select frozen levels while sharing decoded retail assets per worker."""

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
        environment_cache_size: int = 4,
    ) -> None:
        super().__init__()
        if not levels or len(set(levels)) != len(levels):
            raise ValueError("levels must be non-empty and unique")
        if worker_rank < 0:
            raise ValueError("worker_rank cannot be negative")
        if episode_seed_stride < 1:
            raise ValueError("episode_seed_stride must be positive")
        if environment_cache_size < 1:
            raise ValueError("environment_cache_size must be positive")

        self.original_root = Path(original_root)
        self.levels = tuple(str(level) for level in levels)
        self.base_config = base_config
        self.input_config = input_config
        self.reward_config = reward_config
        self.selector_seed = int(selector_seed)
        self.episode_seed_base = int(episode_seed_base)
        self.episode_seed_stride = int(episode_seed_stride)
        self.worker_rank = int(worker_rank)
        self.environment_cache_size = int(environment_cache_size)
        self._weights, self._probabilities = frozen_trainer._normalize_weights(
            self.levels,
            weights,
        )
        self._selector = np.random.default_rng(self.selector_seed)
        self._episode_index = 0
        self._active_level = self.levels[0]
        self._active_episode_seed: int | None = None
        self._catalog = CachedOriginalGameCatalog(
            self.original_root,
            maximum_levels=len(self.levels),
        )
        self._environments: OrderedDict[str, HumanSpeedrunWrapper] = (
            OrderedDict()
        )
        self._active = self._create_environment(self._active_level)
        self._environments[self._active_level] = self._active
        self.observation_space = self._active.observation_space
        self.action_space = self._active.action_space

    def _create_environment(self, level_id: str) -> HumanSpeedrunWrapper:
        return _make_human_env(
            catalog=self._catalog,
            level_id=level_id,
            base_config=self.base_config,
            input_config=self.input_config,
            reward_config=self.reward_config,
        )

    def _activate(self, level_id: str) -> None:
        if level_id == self._active_level:
            self._environments.move_to_end(level_id)
            return
        replacement = self._environments.get(level_id)
        if replacement is None:
            replacement = self._create_environment(level_id)
            if replacement.observation_space != self.observation_space:
                replacement.close()
                raise RuntimeError(f"observation space changed for {level_id}")
            if replacement.action_space != self.action_space:
                replacement.close()
                raise RuntimeError(f"action space changed for {level_id}")
            self._environments[level_id] = replacement
        self._environments.move_to_end(level_id)
        self._active = replacement
        self._active_level = level_id
        while len(self._environments) > self.environment_cache_size:
            evicted_level, evicted = self._environments.popitem(last=False)
            if evicted_level == self._active_level:
                raise RuntimeError("active environment selected for eviction")
            evicted.close()

    def set_level_weights(self, weights: Mapping[str, float]) -> dict[str, float]:
        self._weights, self._probabilities = frozen_trainer._normalize_weights(
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
            "cached_environments": list(self._environments),
            "decoded_levels": self._catalog.loaded_level_count,
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
            self._selector = np.random.default_rng(
                np.random.SeedSequence([self.selector_seed, int(seed)])
            )
        selected_index = int(
            self._selector.choice(len(self.levels), p=self._probabilities)
        )
        level_id = self.levels[selected_index]
        self._activate(level_id)
        if self._episode_index >= self.episode_seed_stride:
            raise RuntimeError("worker exhausted its preregistered seed namespace")
        episode_seed = (
            self.episode_seed_base
            + self.worker_rank * self.episode_seed_stride
            + self._episode_index
        )
        episode_index = self._episode_index
        self._episode_index += 1
        self._active_episode_seed = episode_seed
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
        while self._environments:
            _, environment = self._environments.popitem(last=False)
            environment.close()


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

    environment = SharedCatalogMultiLevelEnv(
        original_root=original_root,
        levels=levels,
        weights=weights,
        base_config=base_config,
        input_config=input_config,
        reward_config=reward_config,
        selector_seed=selector_seed,
        episode_seed_base=episode_seed_base,
        episode_seed_stride=episode_seed_stride,
        worker_rank=worker_rank,
    )
    return Monitor(environment)


def main(argv: list[str] | None = None) -> int:
    frozen_trainer._make_worker = _make_worker
    return frozen_trainer.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
