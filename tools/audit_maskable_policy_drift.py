"""Audit masked factorized-policy drift on a frozen observation dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _quantiles(values: np.ndarray) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {
            "mean": None,
            "max": None,
            "p50": None,
            "p90": None,
            "p99": None,
        }
    return {
        "mean": float(np.mean(array)),
        "max": float(np.max(array)),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "p99": float(np.quantile(array, 0.99)),
    }


def summarize_factor(
    *,
    forward_kl: np.ndarray,
    total_variation: np.ndarray,
    reference_actions: np.ndarray,
    candidate_actions: np.ndarray,
    reference_margins: np.ndarray,
    candidate_margins: np.ndarray,
) -> dict[str, Any]:
    """Summarize one categorical factor from precomputed per-row values."""

    arrays = [
        np.asarray(forward_kl),
        np.asarray(total_variation),
        np.asarray(reference_actions),
        np.asarray(candidate_actions),
        np.asarray(reference_margins),
        np.asarray(candidate_margins),
    ]
    if any(array.ndim != 1 for array in arrays):
        raise ValueError("factor audit inputs must be one-dimensional")
    count = int(arrays[0].size)
    if any(array.size != count for array in arrays):
        raise ValueError("factor audit inputs must have equal lengths")
    if not all(np.isfinite(array).all() for array in arrays[:2] + arrays[4:]):
        raise ValueError("factor audit inputs must be finite")
    if np.any(arrays[0] < -1e-7) or np.any(arrays[1] < -1e-7):
        raise ValueError("KL and total variation must be non-negative")

    flips = arrays[2] != arrays[3]
    flip_count = int(np.sum(flips))
    reference_flip_margins = arrays[4][flips]
    candidate_flip_margins = arrays[5][flips]
    return {
        "sample_count": count,
        "forward_kl": _quantiles(np.maximum(arrays[0], 0.0)),
        "total_variation": _quantiles(np.maximum(arrays[1], 0.0)),
        "deterministic_argmax_flip_count": flip_count,
        "deterministic_argmax_flip_rate": (
            float(flip_count / count) if count else None
        ),
        "reference_top1_probability_margin": _quantiles(arrays[4]),
        "candidate_top1_probability_margin": _quantiles(arrays[5]),
        "flip_reference_margin": _quantiles(reference_flip_margins),
        "flip_candidate_margin": _quantiles(candidate_flip_margins),
        "flips_with_reference_margin_at_most_1e_3": int(
            np.sum(reference_flip_margins <= 1e-3)
        ),
        "flips_with_reference_margin_at_most_1e_2": int(
            np.sum(reference_flip_margins <= 1e-2)
        ),
        "flips_with_reference_margin_at_most_5e_2": int(
            np.sum(reference_flip_margins <= 5e-2)
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-model", required=True, type=Path)
    parser.add_argument("--candidate-model", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    return parser


def _validate_dataset(
    observations: np.ndarray,
    actions: np.ndarray,
    masks: np.ndarray,
) -> None:
    if observations.ndim != 2 or observations.dtype != np.float32:
        raise ValueError("observations must be a float32 matrix")
    if actions.ndim != 2 or actions.shape[1] != 2:
        raise ValueError("actions must have shape (N, 2)")
    if masks.ndim != 2 or masks.dtype != np.bool_:
        raise ValueError("action_masks must be a boolean matrix")
    if not (len(observations) == len(actions) == len(masks)):
        raise ValueError("dataset arrays must have equal row counts")
    if len(observations) == 0:
        raise ValueError("dataset must not be empty")


def _margin(probabilities: Any) -> Any:
    import torch

    if probabilities.shape[1] < 2:
        return torch.ones(
            probabilities.shape[0],
            dtype=probabilities.dtype,
            device=probabilities.device,
        )
    top_two = torch.topk(probabilities, k=2, dim=1).values
    return top_two[:, 0] - top_two[:, 1]


def main() -> int:
    args = _parser().parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    import torch
    from sb3_contrib import MaskablePPO

    with np.load(args.dataset, allow_pickle=False) as dataset:
        observations = np.asarray(dataset["observations"])
        actions = np.asarray(dataset["actions"], dtype=np.int64)
        masks = np.asarray(dataset["action_masks"])
    _validate_dataset(observations, actions, masks)

    reference = MaskablePPO.load(args.reference_model, device=args.device)
    candidate = MaskablePPO.load(args.candidate_model, device=args.device)
    if not np.array_equal(reference.action_space.nvec, candidate.action_space.nvec):
        raise ValueError("model action spaces differ")
    if list(reference.action_space.nvec) != [3, 180]:
        raise ValueError("expected factorized [3, 180] action space")
    expected_mask_width = int(np.sum(reference.action_space.nvec))
    if masks.shape[1] != expected_mask_width:
        raise ValueError("dataset action-mask width does not match models")

    reference.policy.set_training_mode(False)
    candidate.policy.set_training_mode(False)
    factor_rows: list[dict[str, list[np.ndarray]]] = [
        {
            "forward_kl": [],
            "total_variation": [],
            "reference_actions": [],
            "candidate_actions": [],
            "reference_margins": [],
            "candidate_margins": [],
        }
        for _ in range(2)
    ]
    reference_joint_actions: list[np.ndarray] = []
    candidate_joint_actions: list[np.ndarray] = []
    reference_values: list[np.ndarray] = []
    candidate_values: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(observations), args.batch_size):
            stop = min(len(observations), start + args.batch_size)
            observation_batch = observations[start:stop]
            mask_batch = masks[start:stop]
            reference_observations, _ = reference.policy.obs_to_tensor(
                observation_batch
            )
            candidate_observations, _ = candidate.policy.obs_to_tensor(
                observation_batch
            )
            reference_distribution = reference.policy.get_distribution(
                reference_observations,
                action_masks=mask_batch,
            )
            candidate_distribution = candidate.policy.get_distribution(
                candidate_observations,
                action_masks=mask_batch,
            )
            reference_parts = reference_distribution.distributions
            candidate_parts = candidate_distribution.distributions
            if len(reference_parts) != 2 or len(candidate_parts) != 2:
                raise RuntimeError("expected two categorical action factors")

            reference_actions_batch = []
            candidate_actions_batch = []
            for index, (reference_part, candidate_part) in enumerate(
                zip(reference_parts, candidate_parts, strict=True)
            ):
                reference_prob = reference_part.probs
                candidate_prob = candidate_part.probs
                if reference_prob.shape != candidate_prob.shape:
                    raise RuntimeError("categorical factor shapes differ")
                terms = reference_prob * (
                    reference_part.logits - candidate_part.logits
                )
                terms = terms.where(reference_prob > 0, terms.new_zeros(()))
                forward_kl = terms.sum(dim=1).clamp_min(0.0)
                total_variation = 0.5 * torch.abs(
                    reference_prob - candidate_prob
                ).sum(dim=1)
                reference_action = torch.argmax(reference_prob, dim=1)
                candidate_action = torch.argmax(candidate_prob, dim=1)

                row = factor_rows[index]
                row["forward_kl"].append(forward_kl.cpu().numpy())
                row["total_variation"].append(
                    total_variation.cpu().numpy()
                )
                row["reference_actions"].append(
                    reference_action.cpu().numpy()
                )
                row["candidate_actions"].append(
                    candidate_action.cpu().numpy()
                )
                row["reference_margins"].append(
                    _margin(reference_prob).cpu().numpy()
                )
                row["candidate_margins"].append(
                    _margin(candidate_prob).cpu().numpy()
                )
                reference_actions_batch.append(reference_action)
                candidate_actions_batch.append(candidate_action)

            reference_joint_actions.append(
                torch.stack(reference_actions_batch, dim=1).cpu().numpy()
            )
            candidate_joint_actions.append(
                torch.stack(candidate_actions_batch, dim=1).cpu().numpy()
            )
            reference_values.append(
                reference.policy.predict_values(reference_observations)
                .flatten()
                .cpu()
                .numpy()
            )
            candidate_values.append(
                candidate.policy.predict_values(candidate_observations)
                .flatten()
                .cpu()
                .numpy()
            )

    factor_arrays = [
        {key: np.concatenate(values) for key, values in row.items()}
        for row in factor_rows
    ]
    factor_summaries = [summarize_factor(**row) for row in factor_arrays]
    reference_actions_array = np.concatenate(reference_joint_actions)
    candidate_actions_array = np.concatenate(candidate_joint_actions)
    joint_flips = np.any(
        reference_actions_array != candidate_actions_array,
        axis=1,
    )
    semantic_flips = (
        reference_actions_array[:, 0] != candidate_actions_array[:, 0]
    ) | (
        (reference_actions_array[:, 0] == 1)
        & (
            reference_actions_array[:, 1]
            != candidate_actions_array[:, 1]
        )
    )
    expected_fire = actions[:, 0] == 1
    reference_fire = reference_actions_array[:, 0] == 1
    aim_fire_expected = summarize_factor(
        **{key: value[expected_fire] for key, value in factor_arrays[1].items()}
    )
    aim_fire_reference = summarize_factor(
        **{key: value[reference_fire] for key, value in factor_arrays[1].items()}
    )

    reference_value_array = np.concatenate(reference_values).astype(np.float64)
    candidate_value_array = np.concatenate(candidate_values).astype(np.float64)
    value_delta = candidate_value_array - reference_value_array
    value_correlation = None
    if np.std(reference_value_array) > 0 and np.std(candidate_value_array) > 0:
        value_correlation = float(
            np.corrcoef(reference_value_array, candidate_value_array)[0, 1]
        )

    reference_verb_match = reference_actions_array[:, 0] == actions[:, 0]
    candidate_verb_match = candidate_actions_array[:, 0] == actions[:, 0]
    aim_distance_reference = np.abs(
        reference_actions_array[:, 1] - actions[:, 1]
    )
    aim_distance_candidate = np.abs(
        candidate_actions_array[:, 1] - actions[:, 1]
    )
    aim_distance_reference = np.minimum(
        aim_distance_reference,
        180 - aim_distance_reference,
    )
    aim_distance_candidate = np.minimum(
        aim_distance_candidate,
        180 - aim_distance_candidate,
    )

    report = {
        "schema": "zuma-rl.maskable-policy-drift-audit",
        "version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "status": "PASS",
        "classification": "frozen_dataset_post_hoc_diagnostic_only",
        "inputs": {
            "reference_model": {
                "path": str(args.reference_model),
                "sha256": _sha256(args.reference_model),
            },
            "candidate_model": {
                "path": str(args.candidate_model),
                "sha256": _sha256(args.candidate_model),
            },
            "dataset": {
                "path": str(args.dataset),
                "sha256": _sha256(args.dataset),
                "sample_count": len(observations),
            },
        },
        "action_factors": {
            "verb_all_rows": factor_summaries[0],
            "aim_all_rows": factor_summaries[1],
            "aim_expected_fire_rows": aim_fire_expected,
            "aim_reference_fire_rows": aim_fire_reference,
        },
        "joint_deterministic_actions": {
            "exact_flip_count": int(np.sum(joint_flips)),
            "exact_flip_rate": float(np.mean(joint_flips)),
            "semantic_flip_count": int(np.sum(semantic_flips)),
            "semantic_flip_rate": float(np.mean(semantic_flips)),
        },
        "teacher_label_metrics": {
            "reference_verb_accuracy": float(np.mean(reference_verb_match)),
            "candidate_verb_accuracy": float(np.mean(candidate_verb_match)),
            "reference_fire_aim_exact_accuracy": float(
                np.mean(aim_distance_reference[expected_fire] == 0)
            ),
            "candidate_fire_aim_exact_accuracy": float(
                np.mean(aim_distance_candidate[expected_fire] == 0)
            ),
        },
        "value_function": {
            "reference": _quantiles(reference_value_array),
            "candidate": _quantiles(candidate_value_array),
            "signed_delta": _quantiles(value_delta),
            "absolute_delta": _quantiles(np.abs(value_delta)),
            "pearson_correlation": value_correlation,
        },
        "tool": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
            "batch_size": args.batch_size,
            "device": args.device,
        },
        "non_authorizations": [
            "certification_evidence",
            "fresh_holdout_claim",
            "recipe_promotion",
            "large_scale_training",
        ],
    }
    if not math.isfinite(
        factor_summaries[0]["forward_kl"]["mean"]
        + factor_summaries[1]["forward_kl"]["mean"]
    ):
        raise RuntimeError("non-finite factor KL summary")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"report_sha256={_sha256(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
