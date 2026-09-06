"""Benchmark the fidelity-first original-data environment without rendering."""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from pathlib import Path

import gymnasium as gym
import numpy as np

from zuma_rl.revenge_env import RevengeEnv, RevengeEnvConfig
from zuma_rl.revenge_core import SUPPORTED_PROFILE_MODE


def _factory(
    config: RevengeEnvConfig,
    level_id: str,
    root: Path | None,
) -> Callable[[], RevengeEnv]:
    def make() -> RevengeEnv:
        return RevengeEnv(config=config, level_id=level_id, root=root)

    return make


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--warmup-seconds", type=float, default=2.0)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--level", default="Jungle1")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--aim-bins", type=int, default=180)
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--max-ticks", type=int, default=180_000)
    parser.add_argument(
        "--vector-mode",
        choices=("sync", "async"),
        default="sync",
    )
    parser.add_argument(
        "--policy",
        choices=("random", "wait"),
        default="random",
        help="wait isolates chain simulation; random also stresses projectiles.",
    )
    return parser.parse_args()


def _make_vector_env(
    config: RevengeEnvConfig,
    *,
    count: int,
    mode: str,
    level_id: str,
    root: Path | None,
) -> gym.vector.VectorEnv:
    constructors = [
        _factory(config, level_id, root)
        for _ in range(count)
    ]
    vector_type = (
        gym.vector.AsyncVectorEnv
        if mode == "async"
        else gym.vector.SyncVectorEnv
    )
    return vector_type(constructors)


def _ticks_from_info(info: dict[str, object], fallback: int) -> int:
    values = info.get("ticks_advanced")
    if values is None:
        return fallback
    return int(np.asarray(values, dtype=np.int64).sum())


def _run_for(
    env: gym.vector.VectorEnv,
    *,
    duration: float,
    rng: np.random.Generator,
    action_space: gym.Space,
    num_envs: int,
    frame_skip: int,
    policy: str,
    count_episodes: bool,
) -> tuple[int, int, int]:
    decisions = 0
    native_ticks = 0
    episodes = 0
    deadline = time.perf_counter() + duration
    while time.perf_counter() < deadline:
        if policy == "random":
            if isinstance(action_space, gym.spaces.MultiDiscrete):
                actions = np.column_stack(
                    (
                        rng.integers(
                            0,
                            3,
                            size=num_envs,
                            dtype=np.int64,
                        ),
                        rng.integers(
                            0,
                            int(action_space.nvec[1]),
                            size=num_envs,
                            dtype=np.int64,
                        ),
                    )
                )
            else:
                actions = rng.integers(
                    0,
                    int(action_space.n),
                    size=num_envs,
                    dtype=np.int64,
                )
        else:
            if isinstance(action_space, gym.spaces.MultiDiscrete):
                actions = np.zeros((num_envs, 2), dtype=np.int64)
            else:
                actions = np.zeros(num_envs, dtype=np.int64)
        _, _, terminated, truncated, info = env.step(actions)
        decisions += num_envs
        native_ticks += _ticks_from_info(
            info,
            fallback=num_envs * frame_skip,
        )
        if count_episodes:
            episodes += int(np.count_nonzero(terminated | truncated))
    return decisions, native_ticks, episodes


def main() -> None:
    args = parse_args()
    if args.seconds <= 0.0 or args.warmup_seconds < 0.0:
        raise SystemExit("seconds must be positive and warmup non-negative")
    if args.num_envs < 1:
        raise SystemExit("num-envs must be positive")

    config = RevengeEnvConfig(
        aim_bins=args.aim_bins,
        frame_skip=args.frame_skip,
        max_ticks=args.max_ticks,
    )
    env = _make_vector_env(
        config,
        count=args.num_envs,
        mode=args.vector_mode,
        level_id=args.level,
        root=args.root,
    )
    rng = np.random.default_rng(args.seed)
    env.reset(seed=args.seed)
    if args.warmup_seconds:
        _run_for(
            env,
            duration=args.warmup_seconds,
            rng=rng,
            action_space=env.single_action_space,
            num_envs=args.num_envs,
            frame_skip=config.frame_skip,
            policy=args.policy,
            count_episodes=False,
        )

    started = time.perf_counter()
    decisions, native_ticks, episodes = _run_for(
        env,
        duration=args.seconds,
        rng=rng,
        action_space=env.single_action_space,
        num_envs=args.num_envs,
        frame_skip=config.frame_skip,
        policy=args.policy,
        count_episodes=True,
    )
    elapsed = time.perf_counter() - started
    env.close()

    decisions_per_second = decisions / elapsed
    ticks_per_second = native_ticks / elapsed
    episodes_per_day = episodes / elapsed * 86_400.0
    decisions_per_day = decisions_per_second * 86_400.0
    print(f"environment        : ZumaRevenge-v0 / {args.level}")
    print(f"profile_mode       : {SUPPORTED_PROFILE_MODE}")
    print("simulation_device  : CPU (GPU is used by the policy network)")
    print(f"policy             : {args.policy}")
    print(f"vector_mode        : {args.vector_mode}")
    print(f"parallel_envs      : {args.num_envs}")
    print(f"frame_skip         : {config.frame_skip}")
    print(f"elapsed_seconds    : {elapsed:.3f}")
    print(f"decisions          : {decisions:,}")
    print(f"native_100hz_ticks : {native_ticks:,}")
    print(f"decisions_per_sec  : {decisions_per_second:,.0f}")
    print(f"decisions_per_day  : {decisions_per_day:,.0f}")
    print(f"native_ticks_per_sec: {ticks_per_second:,.0f}")
    print(f"completed_episodes : {episodes:,}")
    print(f"episodes_per_day   : {episodes_per_day:,.0f}")
    if episodes == 0:
        upper_bound = ticks_per_second * 86_400.0 / config.max_ticks
        print(f"max_tick_estimate  : {upper_bound:,.0f} episodes/day")
    else:
        print(
            "episode_rate_note   : policy-duration dependent; short random "
            "failures are not full trained-agent games"
        )


if __name__ == "__main__":
    main()
