"""Run short random or greedy episodes without a learned model."""

from __future__ import annotations

import argparse

import numpy as np

from zuma_rl.config import ZumaConfig
from zuma_rl.env import ZumaEnv
from zuma_rl.policies import greedy_action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument(
        "--policy",
        choices=("random", "greedy"),
        default="greedy",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--print-every",
        type=int,
        default=25,
        help="Print the ANSI state every N decisions; use 0 to disable.",
    )
    parser.add_argument("--max-steps", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ZumaConfig(max_steps=args.max_steps)
    env = ZumaEnv(config=config, render_mode="ansi")
    rng = np.random.default_rng(args.seed)
    wins = 0

    for episode in range(args.episodes):
        _, _ = env.reset(seed=args.seed + episode)
        terminated = truncated = False
        total_reward = 0.0
        while not (terminated or truncated):
            if args.policy == "greedy":
                action = greedy_action(env.sim)
            else:
                action = int(rng.integers(config.action_count))
            _, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if args.print_every and env.sim.steps % args.print_every == 0:
                print(env.render())
        wins += int(info["outcome"] == "win")
        print(
            f"episode={episode + 1} outcome={info['outcome']} "
            f"steps={info['steps']} score={info['score']} "
            f"reward={total_reward:.3f}"
        )

    print(f"wins={wins}/{args.episodes}")
    env.close()


if __name__ == "__main__":
    main()

