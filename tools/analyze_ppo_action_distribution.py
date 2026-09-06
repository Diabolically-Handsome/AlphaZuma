"""Audit PPO verb probabilities and behavior on the Revenge environment.

This diagnostic distinguishes a genuinely low-entropy policy collapse from a
deterministic argmax collapse.  The latter can look like an all-wait policy
even when the learned verb distribution remains close to uniform.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO

from zuma_rl.revenge_core import GunState, SUPPORTED_PROFILE_MODE
from zuma_rl.revenge_env import RevengeEnv, RevengeEnvConfig


VERBS = ("wait", "fire", "swap")


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _legal_verbs(env: RevengeEnv) -> tuple[bool, bool, bool]:
    sim = env.sim
    live = not sim.win_pending and not sim.loss_started
    fire = (
        live
        and sim.gun_state is GunState.NORMAL
        and sim.current_color is not None
    )
    swap = (
        fire
        and sim.next_color is not None
        and sim.current_color != sim.next_color
    )
    return True, bool(fire), bool(swap)


def _policy_step(
    model: PPO | MaskablePPO,
    observation: np.ndarray,
    *,
    deterministic: bool,
    action_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    observation_tensor, _ = model.policy.obs_to_tensor(observation)
    with torch.no_grad():
        if action_mask is None:
            distribution = model.policy.get_distribution(observation_tensor)
        else:
            distribution = model.policy.get_distribution(
                observation_tensor,
                action_masks=action_mask,
            )
        categoricals = getattr(distribution, "distribution", None)
        if categoricals is None:
            categoricals = getattr(distribution, "distributions", None)
        if not isinstance(categoricals, list) or len(categoricals) != 2:
            raise RuntimeError("expected factorized [verb, aim] distribution")
        verb_probs = categoricals[0].probs[0]
        action_tensor = distribution.get_actions(deterministic=deterministic)
        verb_entropy = categoricals[0].entropy()[0]
        aim_entropy = categoricals[1].entropy()[0]
    action = action_tensor.detach().cpu().numpy()[0].astype(np.int64)
    probabilities = verb_probs.detach().cpu().numpy().astype(np.float64)
    return (
        action,
        probabilities,
        float(verb_entropy.detach().cpu()),
        float(aim_entropy.detach().cpu()),
    )


def _evaluate_mode(
    model: PPO | MaskablePPO,
    env: RevengeEnv,
    *,
    deterministic: bool,
    episodes: int,
    seed: int,
    seed_protocol: str,
    use_masking: bool,
) -> dict[str, Any]:
    mode_seed = seed + (0 if deterministic else 1_000_000)
    model.set_random_seed(mode_seed)
    torch.manual_seed(mode_seed)
    np.random.seed(mode_seed)

    episode_rows: list[dict[str, Any]] = []
    action_counts: Counter[str] = Counter()
    argmax_counts: Counter[str] = Counter()
    accepted_counts: Counter[str] = Counter()
    legal_counts: Counter[str] = Counter()
    conditional_probability_sums = {
        "all": np.zeros(3, dtype=np.float64),
        "fire_legal": np.zeros(3, dtype=np.float64),
        "fire_illegal": np.zeros(3, dtype=np.float64),
        "swap_legal": np.zeros(3, dtype=np.float64),
        "swap_illegal": np.zeros(3, dtype=np.float64),
    }
    conditional_counts: Counter[str] = Counter()
    probability_sums = np.zeros(3, dtype=np.float64)
    verb_entropy_sum = 0.0
    aim_entropy_sum = 0.0
    argmax_margin_sum = 0.0
    total_actions = 0

    for episode_index in range(episodes):
        if seed_protocol == "continuous_after_first":
            episode_seed = seed if episode_index == 0 else None
        elif seed_protocol == "increment_each_episode":
            episode_seed = seed + episode_index
        else:
            raise RuntimeError(f"unsupported seed protocol: {seed_protocol}")
        observation, _ = env.reset(seed=episode_seed)
        terminated = False
        truncated = False
        reward_total = 0.0
        episode_actions: Counter[str] = Counter()
        episode_accepted: Counter[str] = Counter()
        episode_start = total_actions

        while not (terminated or truncated):
            legal = _legal_verbs(env)
            action, probabilities, verb_entropy, aim_entropy = _policy_step(
                model,
                observation,
                deterministic=deterministic,
                action_mask=(env.action_masks() if use_masking else None),
            )
            verb = int(action[0])
            if not 0 <= verb < len(VERBS):
                raise RuntimeError(f"invalid verb sampled: {verb}")
            verb_name = VERBS[verb]
            argmax_verb = int(np.argmax(probabilities))
            sorted_probabilities = np.sort(probabilities)
            argmax_margin = float(
                sorted_probabilities[-1] - sorted_probabilities[-2]
            )

            probability_sums += probabilities
            conditional_probability_sums["all"] += probabilities
            conditional_counts["all"] += 1
            for key, condition in (
                ("fire_legal", legal[1]),
                ("fire_illegal", not legal[1]),
                ("swap_legal", legal[2]),
                ("swap_illegal", not legal[2]),
            ):
                if condition:
                    conditional_probability_sums[key] += probabilities
                    conditional_counts[key] += 1
            verb_entropy_sum += verb_entropy
            aim_entropy_sum += aim_entropy
            argmax_margin_sum += argmax_margin
            action_counts[verb_name] += 1
            episode_actions[verb_name] += 1
            argmax_counts[VERBS[argmax_verb]] += 1
            if legal[1]:
                legal_counts["fire"] += 1
            if legal[2]:
                legal_counts["swap"] += 1

            observation, reward, terminated, truncated, info = env.step(action)
            reward_total += float(reward)
            accepted_key = f"{verb_name}_{bool(info['action_accepted'])}"
            accepted_counts[accepted_key] += 1
            episode_accepted[accepted_key] += 1
            total_actions += 1

        episode_rows.append(
            {
                "accepted": dict(sorted(episode_accepted.items())),
                "actions": dict(sorted(episode_actions.items())),
                "episode_index": episode_index,
                "length": total_actions - episode_start,
                "outcome": info["outcome"],
                "reward": reward_total,
                "score": int(info["score"]),
                "seed": episode_seed,
            }
        )

    if total_actions < 1:
        raise RuntimeError("evaluation produced no actions")

    conditional_means: dict[str, Any] = {}
    for key in sorted(conditional_probability_sums):
        count = int(conditional_counts[key])
        conditional_means[key] = {
            "count": count,
            "mean_verb_probabilities": (
                (conditional_probability_sums[key] / count).tolist()
                if count
                else None
            ),
        }

    rewards = [row["reward"] for row in episode_rows]
    scores = [row["score"] for row in episode_rows]
    return {
        "accepted": dict(sorted(accepted_counts.items())),
        "action_counts": {name: int(action_counts[name]) for name in VERBS},
        "action_fractions": {
            name: action_counts[name] / total_actions for name in VERBS
        },
        "argmax_counts": {name: int(argmax_counts[name]) for name in VERBS},
        "argmax_fractions": {
            name: argmax_counts[name] / total_actions for name in VERBS
        },
        "deterministic": deterministic,
        "episodes": episode_rows,
        "legal_state_fractions": {
            "fire": legal_counts["fire"] / total_actions,
            "swap": legal_counts["swap"] / total_actions,
        },
        "mean_aim_entropy": aim_entropy_sum / total_actions,
        "mean_argmax_margin": argmax_margin_sum / total_actions,
        "mean_reward": math.fsum(rewards) / len(rewards),
        "mean_score": math.fsum(scores) / len(scores),
        "mean_verb_entropy": verb_entropy_sum / total_actions,
        "mean_verb_probabilities": (probability_sums / total_actions).tolist(),
        "masking": use_masking,
        "probabilities_by_legality": conditional_means,
        "seed": seed,
        "seed_protocol": seed_protocol,
        "total_actions": total_actions,
    }


def _parse_model(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError("model must be LABEL=PATH")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"model does not exist: {path}")
    return label, path


def _reconstruct_initial_model(
    path: Path,
    env: RevengeEnv,
    *,
    device: str,
) -> tuple[PPO | MaskablePPO, dict[str, Any]]:
    source = path.expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    training = payload["training"]
    environment = payload["environment"]
    expected = {
        "action_mode": "flat" if training["flat_actions"] else "factorized",
        "aim_bins": int(training["aim_bins"]),
        "level": str(training["level"]),
        "max_balls": int(training["max_balls"]),
        "max_ticks": int(training["max_ticks"]),
    }
    actual = {
        "action_mode": env.config.action_mode,
        "aim_bins": env.config.aim_bins,
        "level": environment["level"],
        "max_balls": env.config.max_balls,
        "max_ticks": env.config.max_ticks,
    }
    if actual != expected:
        raise RuntimeError(
            f"run config does not match evaluation environment: {expected!r}"
        )
    algorithm = str(payload.get("algorithm", "ppo"))
    model_class: type[PPO] | type[MaskablePPO]
    if algorithm == "maskable_ppo":
        model_class = MaskablePPO
    elif algorithm == "ppo":
        model_class = PPO
    else:
        raise RuntimeError(f"unsupported algorithm in run config: {algorithm}")
    model = model_class(
        "MlpPolicy",
        env,
        learning_rate=float(training["learning_rate"]),
        n_steps=int(training["rollout_steps"]),
        batch_size=int(training["batch_size"]),
        gamma=0.995,
        gae_lambda=0.95,
        ent_coef=float(training["entropy_coef"]),
        policy_kwargs={"net_arch": [256, 256]},
        seed=int(training["seed"]),
        device=device,
        verbose=0,
    )
    return model, {
        "algorithm": algorithm,
        "config_path": str(source),
        "config_sha256": _sha256(source),
        "training_seed": int(training["seed"]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", default=[], type=_parse_model)
    parser.add_argument(
        "--mask-model",
        action="append",
        default=[],
        type=_parse_model,
        help="Evaluate a MaskablePPO checkpoint as LABEL=PATH.",
    )
    parser.add_argument(
        "--reconstruct-initial-from-config",
        type=Path,
        help=(
            "Reconstruct the untrained PPO initialization from a saved run "
            "config and evaluate it before the supplied checkpoints."
        ),
    )
    parser.add_argument("--original-root", required=True, type=Path)
    parser.add_argument("--level", default="Jungle2")
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20270805)
    parser.add_argument(
        "--seed-protocol",
        choices=("continuous_after_first", "increment_each_episode"),
        default="continuous_after_first",
        help=(
            "SB3 boundary evaluation seeds the first reset and lets later "
            "episodes continue the simulator RNG stream."
        ),
    )
    parser.add_argument(
        "--evaluation-mode",
        choices=("both", "deterministic", "stochastic"),
        default="both",
    )
    parser.add_argument("--aim-bins", type=int, default=180)
    parser.add_argument("--max-ticks", type=int, default=12_000)
    parser.add_argument("--max-balls", type=int, default=768)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.episodes < 1:
        raise SystemExit("episodes must be positive")
    if (
        not args.model
        and not args.mask_model
        and args.reconstruct_initial_from_config is None
    ):
        raise SystemExit(
            "at least one --model or --reconstruct-initial-from-config is required"
        )

    config = RevengeEnvConfig(
        aim_bins=args.aim_bins,
        action_mode="factorized",
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
        hard=False,
        curve_index=0,
        profile_mode=SUPPORTED_PROFILE_MODE,
    )
    deterministic_modes = {
        "both": (True, False),
        "deterministic": (True,),
        "stochastic": (False,),
    }[args.evaluation_mode]
    try:
        models: list[dict[str, Any]] = []
        if args.reconstruct_initial_from_config is not None:
            model, reconstruction = _reconstruct_initial_model(
                args.reconstruct_initial_from_config,
                env,
                device=args.device,
            )
            models.append(
                {
                    "label": "reconstructed_initial",
                    "model_num_timesteps": int(model.num_timesteps),
                    "path": None,
                    "reconstruction": reconstruction,
                    "sha256": None,
                    "modes": [
                        _evaluate_mode(
                            model,
                            env,
                            deterministic=deterministic,
                            episodes=args.episodes,
                            seed=args.seed,
                            seed_protocol=args.seed_protocol,
                            use_masking=(
                                reconstruction["algorithm"] == "maskable_ppo"
                            ),
                        )
                        for deterministic in deterministic_modes
                    ],
                }
            )
        for label, path in args.model:
            model = PPO.load(path, device=args.device)
            models.append(
                {
                    "label": label,
                    "model_num_timesteps": int(model.num_timesteps),
                    "path": str(path),
                    "sha256": _sha256(path),
                    "modes": [
                        _evaluate_mode(
                            model,
                            env,
                            deterministic=deterministic,
                            episodes=args.episodes,
                            seed=args.seed,
                            seed_protocol=args.seed_protocol,
                            use_masking=False,
                        )
                        for deterministic in deterministic_modes
                    ],
                }
            )
        for label, path in args.mask_model:
            model = MaskablePPO.load(path, device=args.device)
            models.append(
                {
                    "label": label,
                    "model_num_timesteps": int(model.num_timesteps),
                    "path": str(path),
                    "sha256": _sha256(path),
                    "modes": [
                        _evaluate_mode(
                            model,
                            env,
                            deterministic=deterministic,
                            episodes=args.episodes,
                            seed=args.seed,
                            seed_protocol=args.seed_protocol,
                            use_masking=True,
                        )
                        for deterministic in deterministic_modes
                    ],
                }
            )
    finally:
        env.close()

    report = {
        "environment": {
            "action_mode": "factorized",
            "aim_bins": args.aim_bins,
            "level": args.level,
            "max_balls": args.max_balls,
            "max_ticks": args.max_ticks,
            "original_root": str(args.original_root.resolve()),
            "profile_mode": SUPPORTED_PROFILE_MODE,
        },
        "evaluation": {
            "episodes_per_mode": args.episodes,
            "mode": args.evaluation_mode,
            "seed": args.seed,
            "seed_protocol": args.seed_protocol,
        },
        "models": models,
        "schema": "zuma-rl.ppo-action-distribution-audit",
        "version": 1,
    }
    payload = _canonical_bytes(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    args.output.write_bytes(payload)
    print(json.dumps(report, ensure_ascii=True, sort_keys=True, indent=2))
    print("report_sha256=" + _sha256(args.output))


if __name__ == "__main__":
    main()
