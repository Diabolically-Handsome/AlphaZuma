"""Fine-tune an existing state policy under the human-speedrun v1 contract.

This is intentionally separate from ``zuma_rl.train``.  The existing transfer
recipe and PC Golden environment remain immutable; this launcher creates a new
experimental comparison track with human-limited control and win-first rewards.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

from zuma_rl.human_speedrun import (
    EliteHumanInputConfig,
    HumanSpeedrunWrapper,
    WinFirstRewardConfig,
)
from zuma_rl.revenge_core import SUPPORTED_PROFILE_MODE
from zuma_rl.revenge_env import RevengeEnv, RevengeEnvConfig


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-model", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--original-root", type=Path, default=None)
    parser.add_argument("--level", default="Jungle2")
    parser.add_argument("--seed", type=int, default=20_261_000)
    parser.add_argument("--total-steps", type=int, default=98_304)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--ppo-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--checkpoint-every", type=int, default=16_384)
    parser.add_argument("--eval-every", type=int, default=16_384)
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--eval-num-envs", type=int, default=4)
    parser.add_argument("--max-ticks", type=int, default=12_000)
    parser.add_argument("--max-balls", type=int, default=768)

    parser.add_argument("--reaction-delay-ticks", type=int, default=12)
    parser.add_argument(
        "--max-aim-speed-degrees-per-second",
        type=float,
        default=1_080.0,
    )
    parser.add_argument(
        "--max-aim-acceleration-degrees-per-second-squared",
        type=float,
        default=18_000.0,
    )
    parser.add_argument("--min-button-interval-ticks", type=int, default=5)

    parser.add_argument("--win-reward", type=float, default=10.0)
    parser.add_argument("--failure-reward", type=float, default=-10.0)
    parser.add_argument(
        "--time-penalty-per-native-tick",
        type=float,
        default=-0.0001,
    )
    parser.add_argument(
        "--score-progress-reward-cap",
        type=float,
        default=0.01,
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not args.initial_model.is_file():
        raise SystemExit(f"initial model does not exist: {args.initial_model}")
    if args.run_dir.exists():
        raise SystemExit(f"run directory already exists: {args.run_dir}")
    if args.original_root is not None and not args.original_root.is_dir():
        raise SystemExit(f"original root does not exist: {args.original_root}")
    integer_positive = (
        "total_steps",
        "num_envs",
        "rollout_steps",
        "batch_size",
        "ppo_epochs",
        "checkpoint_every",
        "max_ticks",
        "max_balls",
    )
    for name in integer_positive:
        if int(getattr(args, name)) < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    rollout_size = args.num_envs * args.rollout_steps
    if args.batch_size > rollout_size:
        raise SystemExit("--batch-size cannot exceed the vector rollout size")
    if args.eval_episodes < 0 or args.eval_num_envs < 1:
        raise SystemExit("evaluation counts are invalid")
    if args.eval_episodes and args.eval_num_envs > args.eval_episodes:
        raise SystemExit("--eval-num-envs cannot exceed --eval-episodes")
    if args.eval_episodes and args.eval_every < 1:
        raise SystemExit("--eval-every must be positive when evaluation is enabled")
    floats_positive = ("learning_rate",)
    for name in floats_positive:
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if not math.isfinite(args.entropy_coef) or args.entropy_coef < 0.0:
        raise SystemExit("--entropy-coef must be finite and non-negative")


def _configs(
    args: argparse.Namespace,
) -> tuple[RevengeEnvConfig, EliteHumanInputConfig, WinFirstRewardConfig]:
    base = RevengeEnvConfig(
        aim_bins=180,
        action_mode="factorized",
        frame_skip=1,
        max_ticks=args.max_ticks,
        max_balls=args.max_balls,
        max_projectiles=32,
        max_curves=2,
        # Retained only as a diagnostic side channel in wrapper info.
        score_reward_scale=0.01,
        step_penalty=-0.0001,
        win_reward=10.0,
        loss_reward=-10.0,
    )
    human = EliteHumanInputConfig(
        reaction_delay_ticks=args.reaction_delay_ticks,
        max_aim_speed_degrees_per_second=(
            args.max_aim_speed_degrees_per_second
        ),
        max_aim_acceleration_degrees_per_second_squared=(
            args.max_aim_acceleration_degrees_per_second_squared
        ),
        min_button_interval_ticks=args.min_button_interval_ticks,
    )
    reward = WinFirstRewardConfig(
        win_reward=args.win_reward,
        failure_reward=args.failure_reward,
        time_penalty_per_native_tick=args.time_penalty_per_native_tick,
        score_progress_reward_cap=args.score_progress_reward_cap,
    )
    return base, human, reward


def _environment_kwargs(
    args: argparse.Namespace,
    base_config: RevengeEnvConfig,
) -> dict[str, Any]:
    return {
        "config": base_config,
        "level_id": args.level,
        "root": args.original_root,
        "hard": False,
        "curve_index": 0,
        "profile_mode": SUPPORTED_PROFILE_MODE,
    }


def _runtime() -> dict[str, Any]:
    import importlib.metadata
    import torch

    packages = {}
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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.initial_model = args.initial_model.expanduser().resolve()
    args.run_dir = args.run_dir.expanduser().resolve()
    if args.original_root is not None:
        args.original_root = args.original_root.expanduser().resolve()
    _validate_args(args)
    base_config, human_config, reward_config = _configs(args)
    env_kwargs = _environment_kwargs(args, base_config)

    # Construct once in the parent process so asset/config failures happen
    # before a run directory or worker pool is created.
    probe_base = RevengeEnv(**env_kwargs)
    probe = HumanSpeedrunWrapper(
        probe_base,
        input_config=human_config,
        reward_config=reward_config,
    )
    contract = probe.contract()
    observation_shape = tuple(probe.observation_space.shape)
    action_nvec = probe.action_space.nvec.tolist()
    probe.close()

    try:
        import torch
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
        from stable_baselines3.common.callbacks import (
            CallbackList,
            CheckpointCallback,
        )
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    except ImportError as error:
        raise SystemExit(
            "training dependencies are missing; install zuma-rl[train]"
        ) from error

    args.run_dir.mkdir(parents=True)
    checkpoints = args.run_dir / "checkpoints"
    evaluations = args.run_dir / "evaluations"
    checkpoints.mkdir()
    if args.eval_episodes:
        evaluations.mkdir()

    input_model_sha256 = _sha256(args.initial_model)
    config_payload = {
        "schema": "zuma-rl.human-speedrun-training",
        "version": 1,
        "status": "EXPERIMENTAL",
        "created_at": _utc_now(),
        "scope": (
            "state-policy human-comparison calibration; not a replacement for "
            "the frozen PC Golden or transfer gate"
        ),
        "initial_model": {
            "path": str(args.initial_model),
            "sha256": input_model_sha256,
        },
        "run_dir": str(args.run_dir),
        "level": args.level,
        "seed": args.seed,
        "contract": contract,
        "base_environment": asdict(base_config),
        "observation_shape": observation_shape,
        "action_nvec": action_nvec,
        "training": {
            "total_steps": args.total_steps,
            "num_envs": args.num_envs,
            "device": args.device,
            "rollout_steps": args.rollout_steps,
            "batch_size": args.batch_size,
            "ppo_epochs": args.ppo_epochs,
            "learning_rate": args.learning_rate,
            "entropy_coef": args.entropy_coef,
            "checkpoint_every": args.checkpoint_every,
            "eval_every": args.eval_every,
            "eval_episodes": args.eval_episodes,
            "eval_num_envs": args.eval_num_envs,
            "optimizer_state_reset": True,
            "teacher_anchor": False,
        },
        "runtime": _runtime(),
    }
    (args.run_dir / "config.json").write_text(
        json.dumps(config_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    wrapper_class = partial(
        HumanSpeedrunWrapper,
        input_config=human_config,
        reward_config=reward_config,
    )
    vector_class = DummyVecEnv if args.num_envs == 1 else SubprocVecEnv
    train_env = make_vec_env(
        RevengeEnv,
        n_envs=args.num_envs,
        seed=args.seed,
        env_kwargs=env_kwargs,
        wrapper_class=wrapper_class,
        vec_env_cls=vector_class,
    )
    eval_env = None
    if args.eval_episodes:
        eval_vector_class = (
            DummyVecEnv if args.eval_num_envs == 1 else SubprocVecEnv
        )
        eval_env = make_vec_env(
            RevengeEnv,
            n_envs=args.eval_num_envs,
            seed=args.seed + 10_000,
            env_kwargs=env_kwargs,
            wrapper_class=wrapper_class,
            vec_env_cls=eval_vector_class,
        )

    try:
        source = MaskablePPO.load(args.initial_model, device="cpu")
        if tuple(source.observation_space.shape) != observation_shape:
            raise RuntimeError(
                "initial model observation shape does not match the humanized "
                "environment"
            )
        source_nvec = source.action_space.nvec.tolist()
        if source_nvec != action_nvec:
            raise RuntimeError(
                "initial model action space does not match the humanized "
                "environment"
            )

        model = MaskablePPO(
            "MlpPolicy",
            train_env,
            learning_rate=args.learning_rate,
            n_steps=args.rollout_steps,
            batch_size=args.batch_size,
            n_epochs=args.ppo_epochs,
            gamma=0.995,
            gae_lambda=0.95,
            ent_coef=args.entropy_coef,
            policy_kwargs=copy.deepcopy(source.policy_kwargs),
            tensorboard_log=str(args.run_dir / "tensorboard"),
            seed=args.seed,
            device=args.device,
            verbose=1,
        )
        source_state = source.policy.state_dict()
        model.policy.load_state_dict(source_state, strict=True)
        copied_exactly = all(
            torch.equal(
                source_state[name].detach().cpu(),
                value.detach().cpu(),
            )
            for name, value in model.policy.state_dict().items()
        )
        if not copied_exactly:
            raise RuntimeError("initial policy parameters did not copy bitwise")
        parameter_count = sum(
            parameter.numel() for parameter in model.policy.parameters()
        )
        (args.run_dir / "initialization.json").write_text(
            json.dumps(
                {
                    "schema": "zuma-rl.human-speedrun-initialization",
                    "version": 1,
                    "source_model_sha256": input_model_sha256,
                    "policy_parameters": parameter_count,
                    "policy_copied_bitwise": True,
                    "optimizer_state_reset": True,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        callbacks: list[Any] = [
            CheckpointCallback(
                save_freq=max(1, args.checkpoint_every // args.num_envs),
                save_path=str(checkpoints),
                name_prefix="zuma_human_speedrun_v1",
                save_replay_buffer=False,
                save_vecnormalize=False,
            )
        ]
        if eval_env is not None:
            callbacks.append(
                MaskableEvalCallback(
                    eval_env,
                    best_model_save_path=str(args.run_dir / "best"),
                    log_path=str(evaluations),
                    eval_freq=max(1, args.eval_every // args.num_envs),
                    n_eval_episodes=args.eval_episodes,
                    deterministic=True,
                )
            )
        model.learn(
            total_timesteps=args.total_steps,
            callback=CallbackList(callbacks),
            reset_num_timesteps=True,
            progress_bar=False,
        )
        final_model = args.run_dir / "final_model.zip"
        model.save(final_model)
        completion = {
            "schema": "zuma-rl.human-speedrun-completion",
            "version": 1,
            "status": "COMPLETE",
            "completed_at": _utc_now(),
            "requested_steps": args.total_steps,
            "actual_steps": int(model.num_timesteps),
            "final_model": {
                "path": str(final_model),
                "sha256": _sha256(final_model),
                "bytes": final_model.stat().st_size,
            },
            "policy_parameters": parameter_count,
            "contract": contract,
        }
        (args.run_dir / "completion.json").write_text(
            json.dumps(completion, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    finally:
        train_env.close()
        if eval_env is not None:
            eval_env.close()

    print(json.dumps(completion, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

