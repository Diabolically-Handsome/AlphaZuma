"""Benchmark the actor-observable geometric teacher on full episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from zuma_rl.revenge_core import SUPPORTED_PROFILE_MODE
from zuma_rl.revenge_env import RevengeEnv, RevengeEnvConfig
from zuma_rl.revenge_teacher import ActorObservableRevengeTeacher


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", required=True, type=Path)
    parser.add_argument("--level", default="Jungle2")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20271840)
    parser.add_argument("--aim-bins", type=int, default=180)
    parser.add_argument("--max-ticks", type=int, default=12000)
    parser.add_argument("--max-balls", type=int, default=768)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.episodes < 1:
        raise SystemExit("episodes must be positive")
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")
    config = RevengeEnvConfig(
        aim_bins=args.aim_bins,
        frame_skip=1,
        max_ticks=args.max_ticks,
        max_balls=args.max_balls,
        score_reward_scale=0.01,
        step_penalty=-0.0001,
        win_reward=10.0,
        loss_reward=-10.0,
    )
    env = RevengeEnv(
        config=config,
        level_id=args.level,
        root=args.original_root.resolve(),
        profile_mode=SUPPORTED_PROFILE_MODE,
    )
    teacher = ActorObservableRevengeTeacher.from_env(env)
    rows: list[dict[str, Any]] = []
    try:
        for episode_index in range(args.episodes):
            observation, _ = env.reset(
                seed=args.seed if episode_index == 0 else None
            )
            reward_total = 0.0
            action_counts = {"wait": 0, "fire": 0, "swap": 0}
            terminated = False
            truncated = False
            info: dict[str, Any] = {}
            while not (terminated or truncated):
                action = teacher.act(observation)
                verb = int(action[0])
                action_counts[("wait", "fire", "swap")[verb]] += 1
                observation, reward, terminated, truncated, info = env.step(
                    action
                )
                reward_total += float(reward)
            rows.append(
                {
                    "episode_index": episode_index,
                    "initial_seed": args.seed if episode_index == 0 else None,
                    "reward": reward_total,
                    "score": int(info["score"]),
                    "ticks": int(info["ticks"]),
                    "outcome": info["outcome"],
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "action_counts": action_counts,
                }
            )
    finally:
        env.close()

    wins = sum(row["outcome"] == "win" for row in rows)
    payload = {
        "schema": "zuma-rl.actor-observable-teacher-benchmark",
        "version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(
            timespec="milliseconds"
        ).replace("+00:00", "Z"),
        "status": "PASS" if wins == args.episodes else "FAIL",
        "teacher_contract": {
            "dynamic_inputs": "RevengeEnv actor observation only",
            "static_inputs": [
                "feature layout",
                "logical canvas size",
                "fixed shooter position",
                "collision radius",
            ],
            "privileged_dynamic_state": False,
        },
        "environment": {
            "level": args.level,
            "profile_mode": SUPPORTED_PROFILE_MODE,
            "aim_bins": args.aim_bins,
            "max_ticks": args.max_ticks,
            "max_balls": args.max_balls,
            "frame_skip": 1,
        },
        "evaluation": {
            "seed": args.seed,
            "seed_protocol": "seed_first_reset_then_continuous_resets",
            "episode_count": args.episodes,
        },
        "summary": {
            "wins": wins,
            "losses": sum(row["outcome"] == "loss" for row in rows),
            "truncations": sum(row["truncated"] for row in rows),
            "mean_reward": sum(row["reward"] for row in rows) / len(rows),
            "mean_score": sum(row["score"] for row in rows) / len(rows),
            "mean_ticks": sum(row["ticks"] for row in rows) / len(rows),
        },
        "episodes": rows,
        "source_hashes": {
            "revenge_teacher.py": _sha256(
                Path(__file__).resolve().parents[1]
                / "src"
                / "zuma_rl"
                / "revenge_teacher.py"
            ),
            "revenge_env.py": _sha256(
                Path(__file__).resolve().parents[1]
                / "src"
                / "zuma_rl"
                / "revenge_env.py"
            ),
        },
        "non_authorizations": [
            "large_scale_training",
            "fidelity_gate_opening",
            "original_transfer_certification",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"report_sha256={_sha256(args.output)}")


if __name__ == "__main__":
    main()
