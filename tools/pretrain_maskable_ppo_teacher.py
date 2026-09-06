"""Behavior-clone the actor-observable teacher into a MaskablePPO policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from zuma_rl.revenge_core import SUPPORTED_PROFILE_MODE
from zuma_rl.revenge_env import RevengeEnv, RevengeEnvConfig
from zuma_rl.revenge_features import (
    initialize_polar_aim_head,
    revenge_entity_policy_kwargs,
    revenge_polar_policy_kwargs,
    revenge_relational_policy_kwargs,
)
from zuma_rl.revenge_teacher import ActorObservableRevengeTeacher


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", required=True, type=Path)
    parser.add_argument("--level", default="Jungle2")
    parser.add_argument("--training-episodes", type=int, default=32)
    parser.add_argument("--validation-episodes", type=int, default=8)
    parser.add_argument("--training-seed", type=int, default=20271850)
    parser.add_argument("--validation-seed", type=int, default=20272850)
    parser.add_argument("--model-seed", type=int, default=20260850)
    parser.add_argument(
        "--policy-architecture",
        choices=(
            "flat_mlp",
            "entity_attention",
            "entity_relational",
            "entity_polar",
        ),
        default="flat_mlp",
    )
    parser.add_argument("--aim-bins", type=int, default=180)
    parser.add_argument("--max-ticks", type=int, default=12000)
    parser.add_argument("--max-balls", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--wait-sample-stride", type=int, default=16)
    parser.add_argument("--swap-sample-repeat", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--verb-loss-weight", type=float, default=2.0)
    parser.add_argument(
        "--aim-loss-scope",
        choices=("fire_only", "all_actions"),
        default="fire_only",
        help=(
            "Apply aim supervision only where aim affects the environment, "
            "or reproduce the legacy all-action objective."
        ),
    )
    parser.add_argument(
        "--aim-loss-mode",
        choices=(
            "categorical",
            "categorical_distance",
            "circular_smoothed",
        ),
        default="categorical",
    )
    parser.add_argument("--aim-smoothing-sigma", type=float, default=2.0)
    parser.add_argument("--aim-distance-weight", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--dagger-rounds", type=int, default=0)
    parser.add_argument("--dagger-episodes-per-round", type=int, default=1)
    parser.add_argument("--dagger-epochs", type=int, default=3)
    parser.add_argument("--dagger-seed", type=int, default=20273850)
    parser.add_argument("--dagger-max-steps", type=int, default=4000)
    parser.add_argument(
        "--dagger-teacher-action-probability",
        type=float,
        default=0.25,
        help=(
            "Probability of executing the teacher action during DAgger data "
            "collection; every visited state is still labeled by the teacher."
        ),
    )
    parser.add_argument("--final-refinement-epochs", type=int, default=0)
    parser.add_argument(
        "--final-refinement-verb-loss-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--save-dataset",
        action="store_true",
        help="Save the aggregated actor-observation dataset as compressed NPZ.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def _environment(
    args: argparse.Namespace,
    root: Path,
) -> RevengeEnv:
    return RevengeEnv(
        config=RevengeEnvConfig(
            aim_bins=args.aim_bins,
            frame_skip=1,
            max_ticks=args.max_ticks,
            max_balls=args.max_balls,
            score_reward_scale=0.01,
            step_penalty=-0.0001,
            win_reward=10.0,
            loss_reward=-10.0,
        ),
        level_id=args.level,
        root=root,
        profile_mode=SUPPORTED_PROFILE_MODE,
    )


def _batch_metrics(
    predicted: np.ndarray,
    actions: np.ndarray,
    aim_bins: int,
) -> dict[str, Any]:
    verb_match = predicted[:, 0] == actions[:, 0]
    aim_distance = np.abs(predicted[:, 1] - actions[:, 1])
    aim_distance = np.minimum(aim_distance, aim_bins - aim_distance)
    fire = actions[:, 0] == 1
    semantic_match = verb_match & (~fire | (aim_distance == 0))
    fire_count = int(np.sum(fire))
    return {
        "exact_accuracy": float(np.mean(np.all(predicted == actions, axis=1))),
        "semantic_accuracy": float(np.mean(semantic_match)),
        "verb_accuracy": float(np.mean(verb_match)),
        "aim_exact_accuracy": float(np.mean(aim_distance == 0)),
        "aim_within_one_accuracy": float(np.mean(aim_distance <= 1)),
        "fire_aim_sample_count": fire_count,
        "fire_aim_exact_accuracy": (
            float(np.mean(aim_distance[fire] == 0)) if fire_count else math.nan
        ),
        "fire_aim_within_one_accuracy": (
            float(np.mean(aim_distance[fire] <= 1)) if fire_count else math.nan
        ),
    }


def _train_batch(
    *,
    model: Any,
    observations: list[np.ndarray],
    actions: list[np.ndarray],
    masks: list[np.ndarray],
    max_grad_norm: float,
    verb_loss_weight: float,
    aim_loss_scope: str,
    aim_loss_mode: str = "categorical",
    aim_smoothing_sigma: float = 2.0,
    aim_distance_weight: float = 1.0,
) -> dict[str, Any]:
    import torch

    observation_array = np.asarray(observations, dtype=np.float32)
    action_array = np.asarray(actions, dtype=np.int64)
    mask_array = np.asarray(masks, dtype=np.bool_)
    observation_tensor, _ = model.policy.obs_to_tensor(observation_array)
    action_tensor = torch.as_tensor(
        action_array,
        dtype=torch.long,
        device=model.device,
    )
    mask_tensor = torch.as_tensor(
        mask_array,
        dtype=torch.bool,
        device=model.device,
    )
    model.policy.set_training_mode(True)
    distribution = model.policy.get_distribution(
        observation_tensor,
        action_masks=mask_tensor,
    )
    components = distribution.distributions
    if len(components) != 2:
        raise RuntimeError("expected factorized verb and aim distributions")
    verb_loss = -components[0].log_prob(action_tensor[:, 0]).mean()
    if aim_loss_mode == "categorical":
        aim_nll = -components[1].log_prob(action_tensor[:, 1])
    elif aim_loss_mode in {"categorical_distance", "circular_smoothed"}:
        aim_logits = components[1].logits
        aim_bins = int(aim_logits.shape[1])
        bin_indices = torch.arange(
            aim_bins,
            device=model.device,
            dtype=torch.long,
        ).unsqueeze(0)
        distances = torch.abs(bin_indices - action_tensor[:, 1].unsqueeze(1))
        distances = torch.minimum(distances, aim_bins - distances)
        distance_values = distances.to(dtype=aim_logits.dtype)
        if aim_loss_mode == "categorical_distance":
            aim_nll = -components[1].log_prob(action_tensor[:, 1])
        else:
            target_weights = torch.exp(
                -0.5 * (distance_values / aim_smoothing_sigma) ** 2
            )
            target_weights = target_weights / target_weights.sum(
                dim=1,
                keepdim=True,
            )
            aim_nll = -(
                target_weights * torch.log_softmax(aim_logits, dim=1)
            ).sum(dim=1)
        aim_nll = aim_nll + aim_distance_weight * (
            torch.softmax(aim_logits, dim=1)
            * (distance_values / max(1.0, aim_bins / 2.0))
        ).sum(dim=1)
    else:
        raise ValueError(f"unsupported aim loss mode: {aim_loss_mode!r}")
    if aim_loss_scope == "fire_only":
        fire_rows = action_tensor[:, 0] == 1
        aim_loss = (
            aim_nll[fire_rows].mean()
            if bool(fire_rows.any())
            else aim_nll.sum() * 0.0
        )
        aim_loss_sample_count = int(fire_rows.sum().detach().cpu())
    elif aim_loss_scope == "all_actions":
        aim_loss = aim_nll.mean()
        aim_loss_sample_count = len(actions)
    else:
        raise ValueError(f"unsupported aim loss scope: {aim_loss_scope!r}")
    loss = verb_loss_weight * verb_loss + aim_loss
    entropy = components[0].entropy() + components[1].entropy()
    model.policy.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.policy.parameters(),
        max_grad_norm,
    )
    model.policy.optimizer.step()
    model.policy.set_training_mode(False)
    with torch.no_grad():
        distribution = model.policy.get_distribution(
            observation_tensor,
            action_masks=mask_array,
        )
        predicted = distribution.get_actions(deterministic=True)
    metrics = _batch_metrics(
        predicted.detach().cpu().numpy(),
        action_array,
        model.action_space.nvec[1],
    )
    metrics.update(
        {
            "loss": float(loss.detach().cpu()),
            "verb_loss": float(verb_loss.detach().cpu()),
            "aim_loss": float(aim_loss.detach().cpu()),
            "entropy": (
                float(entropy.mean().detach().cpu())
                if entropy is not None
                else math.nan
            ),
            "gradient_norm": float(
                torch.as_tensor(gradient_norm).detach().cpu()
            ),
            "sample_count": len(actions),
            "aim_loss_sample_count": aim_loss_sample_count,
            "aim_loss_scope": aim_loss_scope,
            "aim_loss_mode": aim_loss_mode,
            "aim_smoothing_sigma": aim_smoothing_sigma,
            "aim_distance_weight": aim_distance_weight,
        }
    )
    return metrics


def _evaluate_dataset(
    *,
    model: Any,
    observations: list[np.ndarray],
    actions: list[np.ndarray],
    masks: list[np.ndarray],
    batch_size: int,
) -> dict[str, Any]:
    import torch

    exact = 0
    verb = 0
    aim_exact = 0
    aim_within_one = 0
    semantic_exact = 0
    fire_aim_exact = 0
    fire_aim_within_one = 0
    fire_count = 0
    fire_aim_distances: list[int] = []
    fire_expected_aims: set[int] = set()
    fire_predicted_aims: set[int] = set()
    verb_confusion = np.zeros((3, 3), dtype=np.int64)
    aim_bins = int(model.action_space.nvec[1])
    model.policy.set_training_mode(False)
    with torch.no_grad():
        for start in range(0, len(actions), batch_size):
            stop = min(len(actions), start + batch_size)
            observation_array = np.asarray(
                observations[start:stop],
                dtype=np.float32,
            )
            action_array = np.asarray(actions[start:stop], dtype=np.int64)
            mask_array = np.asarray(masks[start:stop], dtype=np.bool_)
            observation_tensor, _ = model.policy.obs_to_tensor(
                observation_array
            )
            distribution = model.policy.get_distribution(
                observation_tensor,
                action_masks=mask_array,
            )
            predicted = distribution.get_actions(deterministic=True)
            predicted_array = predicted.detach().cpu().numpy()
            exact += int(np.sum(np.all(predicted_array == action_array, axis=1)))
            verb += int(np.sum(predicted_array[:, 0] == action_array[:, 0]))
            for expected, actual in zip(
                action_array[:, 0],
                predicted_array[:, 0],
                strict=True,
            ):
                verb_confusion[int(expected), int(actual)] += 1
            aim_distance = np.abs(
                predicted_array[:, 1] - action_array[:, 1]
            )
            aim_distance = np.minimum(aim_distance, aim_bins - aim_distance)
            aim_exact += int(np.sum(aim_distance == 0))
            aim_within_one += int(np.sum(aim_distance <= 1))
            fire = action_array[:, 0] == 1
            semantic_exact += int(
                np.sum(
                    (predicted_array[:, 0] == action_array[:, 0])
                    & (~fire | (aim_distance == 0))
                )
            )
            fire_count += int(np.sum(fire))
            fire_aim_exact += int(np.sum((aim_distance == 0) & fire))
            fire_aim_within_one += int(np.sum((aim_distance <= 1) & fire))
            fire_aim_distances.extend(
                int(value) for value in aim_distance[fire].tolist()
            )
            fire_expected_aims.update(
                int(value) for value in action_array[fire, 1].tolist()
            )
            fire_predicted_aims.update(
                int(value) for value in predicted_array[fire, 1].tolist()
            )
    count = len(actions)
    fire_distance_array = np.asarray(fire_aim_distances, dtype=np.float64)
    return {
        "sample_count": count,
        "exact_accuracy": exact / count,
        "semantic_accuracy": semantic_exact / count,
        "verb_accuracy": verb / count,
        "aim_exact_accuracy": aim_exact / count,
        "aim_within_one_accuracy": aim_within_one / count,
        "fire_aim_sample_count": fire_count,
        "fire_aim_exact_accuracy": (
            fire_aim_exact / fire_count if fire_count else None
        ),
        "fire_aim_within_one_accuracy": (
            fire_aim_within_one / fire_count if fire_count else None
        ),
        "fire_aim_within_three_accuracy": (
            float(np.mean(fire_distance_array <= 3)) if fire_count else None
        ),
        "fire_aim_within_five_accuracy": (
            float(np.mean(fire_distance_array <= 5)) if fire_count else None
        ),
        "fire_aim_within_ten_accuracy": (
            float(np.mean(fire_distance_array <= 10)) if fire_count else None
        ),
        "fire_aim_mean_circular_bin_error": (
            float(np.mean(fire_distance_array)) if fire_count else None
        ),
        "fire_aim_median_circular_bin_error": (
            float(np.median(fire_distance_array)) if fire_count else None
        ),
        "fire_aim_p90_circular_bin_error": (
            float(np.quantile(fire_distance_array, 0.9))
            if fire_count
            else None
        ),
        "fire_expected_unique_aim_bins": len(fire_expected_aims),
        "fire_predicted_unique_aim_bins": len(fire_predicted_aims),
        "verb_confusion_expected_rows_predicted_columns": (
            verb_confusion.tolist()
        ),
    }


def _collect_and_train(
    *,
    args: argparse.Namespace,
    env: RevengeEnv,
    model: Any,
) -> dict[str, Any]:
    teacher = ActorObservableRevengeTeacher.from_env(env)
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    batch_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    dagger_round_rows: list[dict[str, Any]] = []
    sampled_legal = 0
    sampled_forced_wait = 0
    raw_verb_counts = [0, 0, 0]
    effective_verb_counts = [0, 0, 0]
    forced_wait_counter = 0
    shuffle_rng = np.random.default_rng(args.model_seed + 1)
    mixture_rng = np.random.default_rng(args.model_seed + 2)

    def append_labeled_state(
        observation: np.ndarray,
        teacher_action: np.ndarray,
        action_mask: np.ndarray,
    ) -> bool:
        nonlocal forced_wait_counter, sampled_forced_wait, sampled_legal

        fire_legal = bool(action_mask[1])
        include = fire_legal
        if fire_legal:
            sampled_legal += 1
        else:
            forced_wait_counter += 1
            include = forced_wait_counter % args.wait_sample_stride == 0
            if include:
                sampled_forced_wait += 1
        if not include:
            return False

        verb = int(teacher_action[0])
        raw_verb_counts[verb] += 1
        repeats = args.swap_sample_repeat if verb == 2 else 1
        for _ in range(repeats):
            observations.append(
                np.asarray(observation, dtype=np.float32).copy()
            )
            actions.append(np.asarray(teacher_action, dtype=np.int64).copy())
            masks.append(np.asarray(action_mask, dtype=np.bool_).copy())
            effective_verb_counts[verb] += 1
        return True

    def train_epochs(
        epoch_count: int,
        *,
        phase: str,
        round_index: int | None,
        verb_loss_weight: float,
    ) -> None:
        if not actions:
            raise RuntimeError("behavior cloning produced no training samples")
        for local_epoch in range(epoch_count):
            permutation = shuffle_rng.permutation(len(actions))
            for batch_index, start in enumerate(
                range(0, len(actions), args.batch_size)
            ):
                selected = permutation[start : start + args.batch_size]
                row = _train_batch(
                    model=model,
                    observations=[observations[index] for index in selected],
                    actions=[actions[index] for index in selected],
                    masks=[masks[index] for index in selected],
                    max_grad_norm=args.max_grad_norm,
                    verb_loss_weight=verb_loss_weight,
                    aim_loss_scope=args.aim_loss_scope,
                    aim_loss_mode=args.aim_loss_mode,
                    aim_smoothing_sigma=args.aim_smoothing_sigma,
                    aim_distance_weight=args.aim_distance_weight,
                )
                row.update(
                    {
                        "phase": phase,
                        "round_index": round_index,
                        "epoch": local_epoch,
                        "batch_index": batch_index,
                        "dataset_sample_count": len(actions),
                    }
                )
                batch_rows.append(row)

    for episode_index in range(args.training_episodes):
        observation, _ = env.reset(
            seed=args.training_seed if episode_index == 0 else None
        )
        terminated = False
        truncated = False
        reward_total = 0.0
        info: dict[str, Any] = {}
        while not (terminated or truncated):
            action = np.asarray(teacher.act(observation), dtype=np.int64)
            action_mask = env.action_masks()
            append_labeled_state(observation, action, action_mask)
            observation, reward, terminated, truncated, info = env.step(action)
            reward_total += float(reward)
        episode_rows.append(
            {
                "episode_index": episode_index,
                "reward": reward_total,
                "score": int(info["score"]),
                "ticks": int(info["ticks"]),
                "outcome": info["outcome"],
            }
        )
    train_epochs(
        args.epochs,
        phase="teacher_behavior_cloning",
        round_index=None,
        verb_loss_weight=args.verb_loss_weight,
    )

    dagger_episode_index = 0
    for round_index in range(args.dagger_rounds):
        round_episodes: list[dict[str, Any]] = []
        round_samples_before = len(actions)
        for episode_in_round in range(args.dagger_episodes_per_round):
            observation, _ = env.reset(
                seed=args.dagger_seed if dagger_episode_index == 0 else None
            )
            dagger_episode_index += 1
            terminated = False
            truncated = False
            reward_total = 0.0
            info: dict[str, Any] = {}
            steps = 0
            exact_agreements = 0
            verb_agreements = 0
            semantic_agreements = 0
            teacher_verb_counts = np.zeros(3, dtype=np.int64)
            policy_verb_counts = np.zeros(3, dtype=np.int64)
            verb_confusion = np.zeros((3, 3), dtype=np.int64)
            teacher_executions = 0
            policy_executions = 0
            labeled_states = 0
            while (
                not (terminated or truncated)
                and steps < args.dagger_max_steps
            ):
                action_mask = env.action_masks()
                teacher_action = np.asarray(
                    teacher.act(observation),
                    dtype=np.int64,
                )
                if append_labeled_state(
                    observation,
                    teacher_action,
                    action_mask,
                ):
                    labeled_states += 1
                policy_action, _ = model.predict(
                    observation,
                    deterministic=True,
                    action_masks=action_mask,
                )
                policy_action = np.asarray(policy_action, dtype=np.int64)
                exact_agreements += int(
                    np.array_equal(policy_action, teacher_action)
                )
                verb_agreements += int(
                    policy_action[0] == teacher_action[0]
                )
                teacher_verb = int(teacher_action[0])
                policy_verb = int(policy_action[0])
                teacher_verb_counts[teacher_verb] += 1
                policy_verb_counts[policy_verb] += 1
                verb_confusion[teacher_verb, policy_verb] += 1
                semantic_agreements += int(
                    teacher_verb == policy_verb
                    and (
                        teacher_verb != 1
                        or teacher_action[1] == policy_action[1]
                    )
                )
                execute_teacher = bool(
                    mixture_rng.random()
                    < args.dagger_teacher_action_probability
                )
                if execute_teacher:
                    executed_action = teacher_action
                    teacher_executions += 1
                else:
                    executed_action = policy_action
                    policy_executions += 1
                observation, reward, terminated, truncated, info = env.step(
                    executed_action
                )
                reward_total += float(reward)
                steps += 1
            round_episodes.append(
                {
                    "episode_in_round": episode_in_round,
                    "reward": reward_total,
                    "score": int(info.get("score", 0)),
                    "ticks": int(info.get("ticks", steps)),
                    "outcome": info.get("outcome", "not_started"),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "hit_step_cap": bool(
                        steps >= args.dagger_max_steps
                        and not (terminated or truncated)
                    ),
                    "steps": steps,
                    "labeled_states": labeled_states,
                    "exact_action_agreement": exact_agreements / steps,
                    "verb_agreement": verb_agreements / steps,
                    "semantic_action_agreement": (
                        semantic_agreements / steps
                    ),
                    "teacher_verb_counts": teacher_verb_counts.tolist(),
                    "policy_verb_counts": policy_verb_counts.tolist(),
                    "verb_confusion_expected_rows_predicted_columns": (
                        verb_confusion.tolist()
                    ),
                    "verb_agreement_by_teacher_verb": [
                        (
                            float(verb_confusion[index, index])
                            / int(teacher_verb_counts[index])
                            if teacher_verb_counts[index]
                            else None
                        )
                        for index in range(3)
                    ],
                    "teacher_executions": teacher_executions,
                    "policy_executions": policy_executions,
                }
            )
        round_samples_after = len(actions)
        train_epochs(
            args.dagger_epochs,
            phase="dagger_aggregate",
            round_index=round_index,
            verb_loss_weight=args.verb_loss_weight,
        )
        dagger_round_rows.append(
            {
                "round_index": round_index,
                "episodes": round_episodes,
                "effective_samples_added": (
                    round_samples_after - round_samples_before
                ),
                "dataset_sample_count_after_collection": round_samples_after,
                "retraining_epochs": args.dagger_epochs,
            }
        )
    if args.final_refinement_epochs:
        train_epochs(
            args.final_refinement_epochs,
            phase="final_aim_refinement",
            round_index=None,
            verb_loss_weight=args.final_refinement_verb_loss_weight,
        )

    if args.save_dataset:
        np.savez_compressed(
            args.output_dir / "teacher_dataset.npz",
            observations=np.asarray(observations, dtype=np.float32),
            actions=np.asarray(actions, dtype=np.int64),
            action_masks=np.asarray(masks, dtype=np.bool_),
        )
    dataset_metrics = _evaluate_dataset(
        model=model,
        observations=observations,
        actions=actions,
        masks=masks,
        batch_size=args.batch_size,
    )
    return {
        "episodes": episode_rows,
        "dagger": {
            "enabled": bool(args.dagger_rounds),
            "rounds": dagger_round_rows,
            "round_count": args.dagger_rounds,
            "episodes_per_round": args.dagger_episodes_per_round,
            "teacher_action_probability": (
                args.dagger_teacher_action_probability
            ),
            "max_steps_per_episode": args.dagger_max_steps,
        },
        "sampled_legal_states": sampled_legal,
        "sampled_forced_wait_states": sampled_forced_wait,
        "raw_verb_counts": raw_verb_counts,
        "effective_verb_counts": effective_verb_counts,
        "swap_sample_repeat": args.swap_sample_repeat,
        "unique_sample_count": len(actions),
        "optimization_sample_count": sum(
            row["sample_count"] for row in batch_rows
        ),
        "initial_epochs": args.epochs,
        "dagger_epochs_per_round": args.dagger_epochs,
        "final_refinement_epochs": args.final_refinement_epochs,
        "final_refinement_verb_loss_weight": (
            args.final_refinement_verb_loss_weight
        ),
        "update_count": len(batch_rows),
        "first_batch": batch_rows[0],
        "last_batch": batch_rows[-1],
        "mean_loss": float(np.mean([row["loss"] for row in batch_rows])),
        "mean_exact_accuracy": float(
            np.mean([row["exact_accuracy"] for row in batch_rows])
        ),
        "final_dataset_metrics": dataset_metrics,
        "teacher_wins": sum(row["outcome"] == "win" for row in episode_rows),
    }


def _evaluate_model(
    *,
    args: argparse.Namespace,
    env: RevengeEnv,
    model: Any,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for episode_index in range(args.validation_episodes):
        observation, _ = env.reset(
            seed=args.validation_seed if episode_index == 0 else None
        )
        terminated = False
        truncated = False
        reward_total = 0.0
        info: dict[str, Any] = {}
        action_counts = {"wait": 0, "fire": 0, "swap": 0}
        while not (terminated or truncated):
            action, _ = model.predict(
                observation,
                deterministic=True,
                action_masks=env.action_masks(),
            )
            action = np.asarray(action, dtype=np.int64)
            action_counts[("wait", "fire", "swap")[int(action[0])]] += 1
            observation, reward, terminated, truncated, info = env.step(action)
            reward_total += float(reward)
        rows.append(
            {
                "episode_index": episode_index,
                "reward": reward_total,
                "score": int(info["score"]),
                "ticks": int(info["ticks"]),
                "outcome": info["outcome"],
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "action_counts": action_counts,
            }
        )
    return {
        "episodes": rows,
        "wins": sum(row["outcome"] == "win" for row in rows),
        "losses": sum(row["outcome"] == "loss" for row in rows),
        "truncations": sum(row["truncated"] for row in rows),
        "mean_reward": sum(row["reward"] for row in rows) / len(rows),
        "mean_score": sum(row["score"] for row in rows) / len(rows),
        "mean_ticks": sum(row["ticks"] for row in rows) / len(rows),
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.training_episodes < 1 or args.validation_episodes < 1:
        raise SystemExit("training and validation episodes must be positive")
    if args.batch_size < 2:
        raise SystemExit("batch-size must be at least 2")
    if args.epochs < 1:
        raise SystemExit("epochs must be positive")
    if args.wait_sample_stride < 1:
        raise SystemExit("wait-sample-stride must be positive")
    if args.swap_sample_repeat < 1:
        raise SystemExit("swap-sample-repeat must be positive")
    if args.dagger_rounds < 0:
        raise SystemExit("dagger-rounds must be non-negative")
    if args.dagger_rounds > 0 and args.dagger_episodes_per_round < 1:
        raise SystemExit("dagger-episodes-per-round must be positive")
    if args.dagger_rounds > 0 and args.dagger_epochs < 1:
        raise SystemExit("dagger-epochs must be positive")
    if args.dagger_rounds > 0 and args.dagger_max_steps < 1:
        raise SystemExit("dagger-max-steps must be positive")
    if args.final_refinement_epochs < 0:
        raise SystemExit("final-refinement-epochs must be non-negative")
    if (
        not math.isfinite(args.final_refinement_verb_loss_weight)
        or args.final_refinement_verb_loss_weight < 0.0
    ):
        raise SystemExit(
            "final-refinement-verb-loss-weight must be non-negative"
        )
    if (
        not math.isfinite(args.dagger_teacher_action_probability)
        or not 0.0 <= args.dagger_teacher_action_probability <= 1.0
    ):
        raise SystemExit(
            "dagger-teacher-action-probability must be between 0 and 1"
        )
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0.0:
        raise SystemExit("learning-rate must be positive")
    if not math.isfinite(args.verb_loss_weight) or args.verb_loss_weight <= 0.0:
        raise SystemExit("verb-loss-weight must be positive")
    if (
        not math.isfinite(args.aim_smoothing_sigma)
        or args.aim_smoothing_sigma <= 0.0
    ):
        raise SystemExit("aim-smoothing-sigma must be positive")
    if (
        not math.isfinite(args.aim_distance_weight)
        or args.aim_distance_weight < 0.0
    ):
        raise SystemExit("aim-distance-weight must be non-negative")
    if not math.isfinite(args.max_grad_norm) or args.max_grad_norm <= 0.0:
        raise SystemExit("max-grad-norm must be positive")
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise SystemExit(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    root = args.original_root.resolve()

    from sb3_contrib import MaskablePPO

    training_env = _environment(args, root)
    validation_env = _environment(args, root)
    if args.policy_architecture == "entity_attention":
        policy_kwargs = revenge_entity_policy_kwargs(training_env)
    elif args.policy_architecture == "entity_relational":
        policy_kwargs = revenge_relational_policy_kwargs(training_env)
    elif args.policy_architecture == "entity_polar":
        policy_kwargs = revenge_polar_policy_kwargs(training_env)
    else:
        policy_kwargs = {"net_arch": [256, 256]}
    model = MaskablePPO(
        "MlpPolicy",
        training_env,
        learning_rate=args.learning_rate,
        n_steps=512,
        batch_size=1024,
        gamma=0.995,
        gae_lambda=0.95,
        ent_coef=0.0,
        policy_kwargs=policy_kwargs,
        seed=args.model_seed,
        device=args.device,
        verbose=0,
    )
    if args.policy_architecture == "entity_polar":
        initialize_polar_aim_head(model)
    try:
        training = _collect_and_train(
            args=args,
            env=training_env,
            model=model,
        )
        model.save(output_dir / "cloned_model")
        validation = _evaluate_model(
            args=args,
            env=validation_env,
            model=model,
        )
    finally:
        training_env.close()
        validation_env.close()

    config = vars(args).copy()
    config["original_root"] = str(root)
    config["output_dir"] = str(output_dir)
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    status = "PASS" if validation["wins"] == args.validation_episodes else "FAIL"
    artifacts = {
        "config": _sha256(output_dir / "config.json"),
        "cloned_model": _sha256(output_dir / "cloned_model.zip"),
    }
    dataset_path = output_dir / "teacher_dataset.npz"
    if dataset_path.exists():
        artifacts["teacher_dataset"] = _sha256(dataset_path)
    report = {
        "schema": "zuma-rl.maskable-ppo-teacher-pretraining",
        "version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(
            timespec="milliseconds"
        ).replace("+00:00", "Z"),
        "status": status,
        "algorithm": "maskable_ppo_actor_behavior_cloning",
        "policy_architecture": args.policy_architecture,
        "policy_parameter_count": sum(
            parameter.numel() for parameter in model.policy.parameters()
        ),
        "teacher_dynamic_input": "packed_actor_observation_only",
        "training": training,
        "validation": validation,
        "artifacts": artifacts,
        "source_hashes": {
            "tool": _sha256(Path(__file__).resolve()),
            "teacher": _sha256(
                Path(__file__).resolve().parents[1]
                / "src"
                / "zuma_rl"
                / "revenge_teacher.py"
            ),
        },
        "non_authorizations": [
            "large_scale_training",
            "rl_recipe_promotion",
            "fidelity_gate_opening",
            "original_transfer_certification",
        ],
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    completion = {
        "schema": "zuma-rl.maskable-ppo-teacher-pretraining-completion",
        "version": 1,
        "status": "COMPLETE",
        "report_sha256": _sha256(output_dir / "report.json"),
        "model_sha256": _sha256(output_dir / "cloned_model.zip"),
    }
    (output_dir / "completion.json").write_text(
        json.dumps(completion, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"report_sha256={_sha256(output_dir / 'report.json')}")
    print(f"completion_sha256={_sha256(output_dir / 'completion.json')}")


if __name__ == "__main__":
    main()
