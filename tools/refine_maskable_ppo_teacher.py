"""Refine a cloned MaskablePPO policy on a frozen teacher dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from pretrain_maskable_ppo_teacher import (
    _environment,
    _evaluate_dataset,
    _evaluate_model,
    _train_batch,
)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", required=True, type=Path)
    parser.add_argument("--level", default="Jungle2")
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model", type=Path)
    model_group.add_argument(
        "--initialize-policy-architecture",
        choices=("entity_polar",),
    )
    parser.add_argument("--model-seed", type=int, default=20260850)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--validation-episodes", type=int, default=2)
    parser.add_argument("--validation-seed", type=int, default=20272850)
    parser.add_argument("--required-wins", type=int, default=1)
    parser.add_argument("--shuffle-seed", type=int, default=20274850)
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--verb-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--aim-loss-scope",
        choices=("fire_only", "all_actions"),
        default="fire_only",
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
    parser.add_argument("--aim-bins", type=int, default=180)
    parser.add_argument("--max-ticks", type=int, default=12000)
    parser.add_argument("--max-balls", type=int, default=768)
    parser.add_argument("--device", default="auto")
    return parser


def _load_dataset(
    *,
    path: Path,
    env: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"observations", "actions", "action_masks"}
        if set(archive.files) != required:
            raise ValueError(
                f"dataset fields must be exactly {sorted(required)!r}"
            )
        observations = np.asarray(archive["observations"], dtype=np.float32)
        actions = np.asarray(archive["actions"], dtype=np.int64)
        masks = np.asarray(archive["action_masks"], dtype=np.bool_)

    expected_observation = int(env.observation_space.shape[0])
    action_sizes = np.asarray(env.action_space.nvec, dtype=np.int64)
    expected_mask = int(np.sum(action_sizes))
    if observations.ndim != 2 or observations.shape[1] != expected_observation:
        raise ValueError("dataset observation shape does not match environment")
    if actions.shape != (len(observations), len(action_sizes)):
        raise ValueError("dataset action shape does not match environment")
    if masks.shape != (len(observations), expected_mask):
        raise ValueError("dataset action-mask shape does not match environment")
    if len(observations) < 1:
        raise ValueError("dataset must contain at least one sample")
    if not np.isfinite(observations).all():
        raise ValueError("dataset observations must be finite")
    if np.any(actions < 0) or np.any(actions >= action_sizes):
        raise ValueError("dataset contains an out-of-range action")
    rows = np.arange(len(actions))
    offset = 0
    for component, size in enumerate(action_sizes):
        selected = masks[rows, offset + actions[:, component]]
        if not bool(np.all(selected)):
            raise ValueError("dataset contains an action rejected by its mask")
        offset += int(size)
    return observations, actions, masks


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = np.asarray([row[key] for row in rows], dtype=np.float64)
    return float(np.nanmean(values))


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.validation_episodes < 0:
        raise SystemExit("validation-episodes must be non-negative")
    if not 0 <= args.required_wins <= args.validation_episodes:
        raise SystemExit("required-wins must be within validation episode count")
    if args.epochs < 0:
        raise SystemExit("epochs must be non-negative")
    if args.epochs == 0 and args.validation_episodes == 0:
        raise SystemExit("at least one of refinement or validation is required")
    if args.batch_size < 2:
        raise SystemExit("batch-size must be at least 2")
    for name in ("learning_rate", "max_grad_norm"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0.0:
            raise SystemExit(f"{name.replace('_', '-')} must be positive")
    if (
        not math.isfinite(args.verb_loss_weight)
        or args.verb_loss_weight < 0.0
    ):
        raise SystemExit("verb-loss-weight must be non-negative")
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

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise SystemExit(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    root = args.original_root.resolve()
    model_path = args.model.resolve() if args.model is not None else None
    dataset_path = args.dataset.resolve()

    from sb3_contrib import MaskablePPO

    training_env = _environment(args, root)
    validation_env = _environment(args, root)
    try:
        if model_path is not None:
            model = MaskablePPO.load(
                model_path,
                env=training_env,
                device=args.device,
            )
        else:
            from zuma_rl.revenge_features import (
                initialize_polar_aim_head,
                revenge_polar_policy_kwargs,
            )

            policy_kwargs = revenge_polar_policy_kwargs(training_env)
            model = MaskablePPO(
                "MlpPolicy",
                training_env,
                learning_rate=args.learning_rate,
                n_steps=512,
                batch_size=512,
                gamma=0.995,
                gae_lambda=0.95,
                ent_coef=0.0,
                policy_kwargs=policy_kwargs,
                seed=args.model_seed,
                device=args.device,
                verbose=0,
            )
            initialize_polar_aim_head(model)
        observations, actions, masks = _load_dataset(
            path=dataset_path,
            env=training_env,
        )
        if model.observation_space != training_env.observation_space:
            raise ValueError("model observation space does not match environment")
        if model.action_space != training_env.action_space:
            raise ValueError("model action space does not match environment")
        for group in model.policy.optimizer.param_groups:
            group["lr"] = args.learning_rate

        before = _evaluate_dataset(
            model=model,
            observations=observations,
            actions=actions,
            masks=masks,
            batch_size=args.batch_size,
        )
        rng = np.random.default_rng(args.shuffle_seed)
        epoch_rows: list[dict[str, Any]] = []
        total_updates = 0
        for epoch in range(args.epochs):
            permutation = rng.permutation(len(actions))
            batch_rows: list[dict[str, Any]] = []
            for start in range(0, len(actions), args.batch_size):
                selected = permutation[start : start + args.batch_size]
                row = _train_batch(
                    model=model,
                    observations=observations[selected],
                    actions=actions[selected],
                    masks=masks[selected],
                    max_grad_norm=args.max_grad_norm,
                    verb_loss_weight=args.verb_loss_weight,
                    aim_loss_scope=args.aim_loss_scope,
                    aim_loss_mode=args.aim_loss_mode,
                    aim_smoothing_sigma=args.aim_smoothing_sigma,
                    aim_distance_weight=args.aim_distance_weight,
                )
                batch_rows.append(row)
                total_updates += 1
            epoch_rows.append(
                {
                    "epoch": epoch,
                    "mean_loss": _mean(batch_rows, "loss"),
                    "mean_verb_loss": _mean(batch_rows, "verb_loss"),
                    "mean_aim_loss": _mean(batch_rows, "aim_loss"),
                    "mean_verb_accuracy": _mean(
                        batch_rows,
                        "verb_accuracy",
                    ),
                    "mean_fire_aim_exact_accuracy": _mean(
                        batch_rows,
                        "fire_aim_exact_accuracy",
                    ),
                    "mean_gradient_norm": _mean(
                        batch_rows,
                        "gradient_norm",
                    ),
                    "update_count": len(batch_rows),
                }
            )
        after = _evaluate_dataset(
            model=model,
            observations=observations,
            actions=actions,
            masks=masks,
            batch_size=args.batch_size,
        )
        model.save(output_dir / "refined_model")
        if args.validation_episodes:
            validation = _evaluate_model(
                args=args,
                env=validation_env,
                model=model,
            )
        else:
            validation = {
                "skipped": True,
                "reason": "offline_candidate_screen",
            }
    finally:
        training_env.close()
        validation_env.close()

    config = vars(args).copy()
    resolved_paths = {
        "original_root": root,
        "model": model_path,
        "dataset": dataset_path,
        "output_dir": output_dir,
    }
    for key, value in resolved_paths.items():
        config[key] = str(value) if value is not None else None
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    status = (
        "OFFLINE_COMPLETE"
        if args.validation_episodes == 0
        else (
            "PASS"
            if validation["wins"] >= args.required_wins
            else "FAIL"
        )
    )
    report = {
        "schema": "zuma-rl.maskable-ppo-teacher-offline-refinement",
        "version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(
            timespec="milliseconds"
        ).replace("+00:00", "Z"),
        "status": status,
        "algorithm": "frozen_dataset_maskable_ppo_policy_refinement",
        "policy_architecture": (
            args.initialize_policy_architecture or "loaded_model"
        ),
        "policy_parameter_count": sum(
            parameter.numel() for parameter in model.policy.parameters()
        ),
        "input_artifacts": {
            "model": _sha256(model_path) if model_path is not None else None,
            "model_initialization": args.initialize_policy_architecture,
            "model_seed": args.model_seed,
            "dataset": _sha256(dataset_path),
        },
        "dataset": {
            "sample_count": len(actions),
            "before": before,
            "after": after,
        },
        "optimization": {
            "epochs": args.epochs,
            "update_count": total_updates,
            "first_epoch": epoch_rows[0] if epoch_rows else None,
            "last_epoch": epoch_rows[-1] if epoch_rows else None,
        },
        "validation": validation,
        "artifacts": {
            "config": _sha256(output_dir / "config.json"),
            "refined_model": _sha256(output_dir / "refined_model.zip"),
        },
        "source_hashes": {
            "tool": _sha256(Path(__file__).resolve()),
            "training_primitives": _sha256(
                Path(__file__).resolve().parent
                / "pretrain_maskable_ppo_teacher.py"
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
        "schema": "zuma-rl.maskable-ppo-teacher-offline-refinement-completion",
        "version": 1,
        "status": "COMPLETE",
        "report_sha256": _sha256(output_dir / "report.json"),
        "model_sha256": _sha256(output_dir / "refined_model.zip"),
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
