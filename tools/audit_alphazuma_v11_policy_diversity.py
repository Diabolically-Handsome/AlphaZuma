"""Audit deterministic policy diversity on a frozen, non-formal dataset."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ACTION_NVEC = (3, 180)
FIRE_ACTION = 1


def sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def action_signature(actions: np.ndarray) -> str:
    canonical = np.ascontiguousarray(np.asarray(actions, dtype="<i8"))
    return "sha256:" + hashlib.sha256(canonical.tobytes()).hexdigest()


def validate_actions(
    actions: np.ndarray,
    masks: np.ndarray,
    *,
    nvec: tuple[int, int] = ACTION_NVEC,
) -> None:
    actions = np.asarray(actions)
    masks = np.asarray(masks)
    if actions.ndim != 2 or actions.shape[1] != len(nvec):
        raise ValueError("actions must have shape (N, 2)")
    if masks.ndim != 2 or masks.dtype != np.bool_:
        raise ValueError("action masks must be a boolean matrix")
    if len(actions) != len(masks):
        raise ValueError("actions and masks must have equal row counts")
    if masks.shape[1] != sum(nvec):
        raise ValueError("action-mask width does not match action factors")
    if not np.issubdtype(actions.dtype, np.integer):
        raise ValueError("actions must be integers")

    offsets = np.cumsum((0,) + nvec[:-1])
    rows = np.arange(len(actions))
    for factor, (width, offset) in enumerate(zip(nvec, offsets, strict=True)):
        values = actions[:, factor]
        if np.any(values < 0) or np.any(values >= width):
            raise ValueError(f"action factor {factor} is out of range")
        if not np.all(masks[rows, offset + values]):
            raise ValueError(f"action factor {factor} violates its mask")


def summarize_action_pair(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    left = np.asarray(left, dtype=np.int64)
    right = np.asarray(right, dtype=np.int64)
    if left.ndim != 2 or left.shape[1] != 2 or left.shape != right.shape:
        raise ValueError("paired actions must have equal shape (N, 2)")
    if len(left) == 0:
        raise ValueError("paired actions must not be empty")

    verb_diff = left[:, 0] != right[:, 0]
    aim_diff = left[:, 1] != right[:, 1]
    exact_diff = verb_diff | aim_diff
    semantic_diff = verb_diff | ((left[:, 0] == FIRE_ACTION) & aim_diff)
    both_fire = (left[:, 0] == FIRE_ACTION) & (right[:, 0] == FIRE_ACTION)
    circular_aim_distance = np.abs(left[:, 1] - right[:, 1])
    circular_aim_distance = np.minimum(
        circular_aim_distance,
        ACTION_NVEC[1] - circular_aim_distance,
    )

    def count_rate(values: np.ndarray) -> dict[str, int | float]:
        return {
            "count": int(np.sum(values)),
            "rate": float(np.mean(values)),
        }

    both_fire_distances = circular_aim_distance[both_fire]
    return {
        "sample_count": int(len(left)),
        "exact_action_difference": count_rate(exact_diff),
        "semantic_action_difference": count_rate(semantic_diff),
        "verb_difference": count_rate(verb_diff),
        "aim_index_difference_all_rows": count_rate(aim_diff),
        "both_fire_rows": int(np.sum(both_fire)),
        "both_fire_circular_aim_distance": {
            "mean": (
                float(np.mean(both_fire_distances))
                if both_fire_distances.size
                else None
            ),
            "median": (
                float(np.median(both_fire_distances))
                if both_fire_distances.size
                else None
            ),
            "max": (
                int(np.max(both_fire_distances))
                if both_fire_distances.size
                else None
            ),
        },
    }


def _require_hash(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} hash mismatch: expected {expected}, got {actual}")
    return actual


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
    validate_actions(actions, masks)


def _predict_actions(
    model_path: Path,
    observations: np.ndarray,
    masks: np.ndarray,
    *,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    from sb3_contrib import MaskablePPO

    model = MaskablePPO.load(model_path, device=device)
    if tuple(int(value) for value in model.action_space.nvec) != ACTION_NVEC:
        raise ValueError(f"unexpected action space for {model_path}")
    if tuple(int(value) for value in model.observation_space.shape) != (
        observations.shape[1],
    ):
        raise ValueError(f"unexpected observation space for {model_path}")

    batches: list[np.ndarray] = []
    for start in range(0, len(observations), batch_size):
        stop = min(len(observations), start + batch_size)
        predicted, _ = model.predict(
            observations[start:stop],
            deterministic=True,
            action_masks=masks[start:stop],
        )
        batches.append(np.asarray(predicted, dtype=np.int64))
    actions = np.concatenate(batches, axis=0)
    metadata = {
        "internal_timesteps": int(model.num_timesteps),
        "observation_shape": list(model.observation_space.shape),
        "action_nvec": [int(value) for value in model.action_space.nvec],
    }
    del model
    gc.collect()
    validate_actions(actions, masks)
    return actions, metadata


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-model", required=True, type=Path)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--dataset-sha256", required=True)
    parser.add_argument("--checkpoint-audit", required=True, type=Path)
    parser.add_argument("--checkpoint-audit-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--expected-candidates", type=int, default=7)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.expected_candidates <= 1:
        raise ValueError("--expected-candidates must exceed one")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    reference_hash = _require_hash(
        args.reference_model, args.reference_sha256, "reference model"
    )
    dataset_hash = _require_hash(args.dataset, args.dataset_sha256, "dataset")
    audit_hash = _require_hash(
        args.checkpoint_audit,
        args.checkpoint_audit_sha256,
        "checkpoint audit",
    )

    checkpoint_audit = json.loads(args.checkpoint_audit.read_text(encoding="utf-8"))
    if checkpoint_audit.get("status") != "PASS":
        raise ValueError("checkpoint audit did not pass")
    checkpoints = checkpoint_audit.get("checkpoints")
    if not isinstance(checkpoints, list) or len(checkpoints) != args.expected_candidates:
        raise ValueError("checkpoint audit has an unexpected candidate count")
    run_ids = [row.get("run") for row in checkpoints]
    if any(not isinstance(run_id, str) or not run_id for run_id in run_ids):
        raise ValueError("checkpoint audit contains an invalid run id")
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("checkpoint audit contains duplicate run ids")

    with np.load(args.dataset, allow_pickle=False) as dataset:
        observations = np.asarray(dataset["observations"])
        teacher_actions = np.asarray(dataset["actions"], dtype=np.int64)
        masks = np.asarray(dataset["action_masks"])
    _validate_dataset(observations, teacher_actions, masks)

    reference_actions, reference_metadata = _predict_actions(
        args.reference_model,
        observations,
        masks,
        batch_size=args.batch_size,
        device=args.device,
    )
    if not np.array_equal(reference_actions, teacher_actions):
        mismatch_count = int(
            np.sum(np.any(reference_actions != teacher_actions, axis=1))
        )
        raise ValueError(
            f"stored collector actions differ from reference on {mismatch_count} rows"
        )

    candidate_actions: dict[str, np.ndarray] = {}
    candidate_rows: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        if checkpoint.get("status") != "PASS":
            raise ValueError(f"checkpoint did not pass: {checkpoint.get('run')}")
        path = Path(checkpoint["path"])
        expected_hash = checkpoint["sha256"]
        actual_hash = _require_hash(path, expected_hash, checkpoint["run"])
        actions, metadata = _predict_actions(
            path,
            observations,
            masks,
            batch_size=args.batch_size,
            device=args.device,
        )
        candidate_actions[checkpoint["run"]] = actions
        candidate_rows.append(
            {
                "run": checkpoint["run"],
                "path": str(path),
                "sha256": actual_hash,
                "filename_timesteps": checkpoint.get("filename_timesteps"),
                "model": metadata,
                "action_signature": action_signature(actions),
                "versus_reference": summarize_action_pair(
                    reference_actions, actions
                ),
            }
        )

    pairwise: list[dict[str, Any]] = []
    for left_index, left_id in enumerate(run_ids):
        for right_id in run_ids[left_index + 1 :]:
            pairwise.append(
                {
                    "left": left_id,
                    "right": right_id,
                    **summarize_action_pair(
                        candidate_actions[left_id], candidate_actions[right_id]
                    ),
                }
            )

    exact_rates = np.asarray(
        [row["exact_action_difference"]["rate"] for row in pairwise]
    )
    semantic_rates = np.asarray(
        [row["semantic_action_difference"]["rate"] for row in pairwise]
    )
    signatures = [row["action_signature"] for row in candidate_rows]
    report = {
        "schema": "zuma-rl.alphazuma-v11-policy-diversity-audit",
        "version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "status": "PASS",
        "classification": "frozen_dataset_post_hoc_training_health_diagnostic_only",
        "inputs": {
            "reference_model": {
                "path": str(args.reference_model),
                "sha256": reference_hash,
                "model": reference_metadata,
                "action_signature": action_signature(reference_actions),
                "stored_collector_actions_exact_match": True,
            },
            "dataset": {
                "path": str(args.dataset),
                "sha256": dataset_hash,
                "sample_count": int(len(observations)),
                "observation_shape": list(observations.shape),
                "action_shape": list(teacher_actions.shape),
                "mask_shape": list(masks.shape),
            },
            "checkpoint_audit": {
                "path": str(args.checkpoint_audit),
                "sha256": audit_hash,
            },
        },
        "candidates": candidate_rows,
        "pairwise_candidate_comparisons": pairwise,
        "summary": {
            "candidate_count": len(candidate_rows),
            "pair_count": len(pairwise),
            "unique_candidate_action_signatures": len(set(signatures)),
            "all_candidate_action_signatures_unique": len(set(signatures))
            == len(signatures),
            "exact_action_difference_rate": {
                "min": float(np.min(exact_rates)),
                "median": float(np.median(exact_rates)),
                "max": float(np.max(exact_rates)),
            },
            "semantic_action_difference_rate": {
                "min": float(np.min(semantic_rates)),
                "median": float(np.median(semantic_rates)),
                "max": float(np.max(semantic_rates)),
            },
            "identical_candidate_pairs": int(np.sum(exact_rates == 0.0)),
        },
        "interpretation": {
            "behavioral_diversity_measured": True,
            "performance_improvement_claimed": False,
            "reason": (
                "Policy disagreement on a frozen diagnostic dataset establishes "
                "behavioral diversity, not game-playing quality."
            ),
        },
        "tool": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
            "batch_size": args.batch_size,
            "device": args.device,
        },
        "formal_seed_consumption": "none",
        "formal_gate_effect": "none",
        "non_authorizations": [
            "candidate_selection",
            "certification_evidence",
            "fresh_holdout_claim",
            "recipe_change",
            "promotion",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    print(f"report_sha256={sha256_file(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
