"""Measure raw environment throughput without rendering."""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable

import gymnasium as gym
import numpy as np

from zuma_rl.config import ZumaConfig
from zuma_rl.env import ZumaEnv


def _factory(config: ZumaConfig) -> Callable[[], ZumaEnv]:
    def make() -> ZumaEnv:
        return ZumaEnv(config=config)

    return make


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--warmup-seconds", type=float, default=2.0)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--vector-mode",
        choices=("sync", "async"),
        default="sync",
        help="async uses worker processes and is useful for CPU scaling tests.",
    )
    parser.add_argument("--aim-bins", type=int, default=180)
    parser.add_argument("--initial-balls", type=int, default=24)
    return parser.parse_args()


def _make_vector_env(
    config: ZumaConfig,
    count: int,
    mode: str,
) -> gym.vector.VectorEnv:
    constructors = [_factory(config) for _ in range(count)]
    vector_type = (
        gym.vector.AsyncVectorEnv if mode == "async" else gym.vector.SyncVectorEnv
    )
    return vector_type(constructors)


def _run_for(
    env: gym.vector.VectorEnv,
    *,
    duration: float,
    rng: np.random.Generator,
    action_count: int,
    num_envs: int,
    count_episodes: bool,
) -> tuple[int, int]:
    steps = 0
    episodes = 0
    deadline = time.perf_counter() + duration
    while time.perf_counter() < deadline:
        actions = rng.integers(
            0,
            action_count,
            size=num_envs,
            dtype=np.int64,
        )
        _, _, terminated, truncated, _ = env.step(actions)
        steps += num_envs
        if count_episodes:
            episodes += int(np.count_nonzero(terminated | truncated))
    return steps, episodes


def main() -> None:
    args = parse_args()
    if args.seconds <= 0 or args.warmup_seconds < 0:
        raise SystemExit("seconds must be positive and warmup-seconds non-negative")
    if args.num_envs < 1:
        raise SystemExit("num-envs must be positive")

    config = ZumaConfig(
        aim_bins=args.aim_bins,
        initial_balls=args.initial_balls,
    )
    env = _make_vector_env(config, args.num_envs, args.vector_mode)
    rng = np.random.default_rng(args.seed)
    env.reset(seed=args.seed)

    if args.warmup_seconds:
        _run_for(
            env,
            duration=args.warmup_seconds,
            rng=rng,
            action_count=config.action_count,
            num_envs=args.num_envs,
            count_episodes=False,
        )

    started = time.perf_counter()
    steps, episodes = _run_for(
        env,
        duration=args.seconds,
        rng=rng,
        action_count=config.action_count,
        num_envs=args.num_envs,
        count_episodes=True,
    )
    elapsed = time.perf_counter() - started
    env.close()

    steps_per_second = steps / elapsed
    completed_per_day = episodes / elapsed * 86_400.0
    print(f"vector_mode       : {args.vector_mode}")
    print(f"parallel_envs     : {args.num_envs}")
    print(f"elapsed_seconds   : {elapsed:.3f}")
    print(f"environment_steps : {steps:,}")
    print(f"steps_per_second  : {steps_per_second:,.0f}")
    print(f"completed_episodes: {episodes:,}")
    print(f"episodes_per_day  : {completed_per_day:,.0f}")
    if episodes == 0:
        conservative = steps_per_second * 86_400.0 / config.max_steps
        print(f"max_step_estimate : {conservative:,.0f} episodes/day")


if __name__ == "__main__":
    main()

