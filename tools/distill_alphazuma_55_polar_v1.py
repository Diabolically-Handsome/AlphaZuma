"""Distill the settled V3 teacher into a fresh full55 polar policy."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import platform
from pathlib import Path
import sys
import time
from typing import Any

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import numpy as np

from tools import distill_alphazuma_55 as legacy
from tools import distill_alphazuma_55_settled_v3 as settled
from zuma_rl.alphazuma_55 import INCLUDED_LEVELS, full55_environment_config
from zuma_rl.revenge_core import SUPPORTED_PROFILE_MODE
from zuma_rl.revenge_env import RevengeEnv
from zuma_rl.revenge_features import (
    initialize_polar_aim_head,
    revenge_polar_policy_kwargs,
)


SCRIPT_PATH = Path(__file__).resolve()
POLICY_ID = "curve-aware-settled-strategic-masked-v3"
EXPECTED_OBSERVATION_SHAPE = (22_833,)
EXPECTED_ACTION_NVEC = (4, 180)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_preregistration(path: Path) -> dict[str, Any]:
    prereg = legacy._read_json(path)
    if (
        prereg.get("schema")
        != "zuma-rl.alphazuma-55-polar-distillation-preregistration"
        or prereg.get("version") != 1
        or prereg.get("status") != "FROZEN_BEFORE_TRAINING"
    ):
        raise ValueError("unexpected polar distillation preregistration")
    trainer = prereg.get("trainer", {})
    if Path(str(trainer.get("path", ""))).resolve() != SCRIPT_PATH:
        raise ValueError("preregistration binds another polar distiller")
    if trainer.get("sha256") != legacy._sha256(SCRIPT_PATH):
        raise ValueError("polar distiller bytes differ from preregistration")
    master = prereg.get("master_preregistration", {})
    master_path = Path(str(master.get("path", ""))).resolve(strict=True)
    if master.get("sha256") != legacy._sha256(master_path):
        raise ValueError("master preregistration bytes differ")
    master_value = legacy._read_json(master_path)
    if (
        master_value.get("schema")
        != "zuma-rl.alphazuma-55-weekend-master-preregistration"
    ):
        raise ValueError("unexpected master preregistration")
    for name, artifact in prereg.get("implementation", {}).items():
        artifact_path = Path(str(artifact["path"])).resolve(strict=True)
        if artifact.get("sha256") != legacy._sha256(artifact_path):
            raise ValueError(f"implementation bytes differ: {name}")
    evidence = prereg.get("teacher_evidence", {}).get("exact_mask_fresh_55", {})
    settled._validate_probe_result(evidence, expected_policy_id=POLICY_ID)
    if tuple(prereg.get("levels", ())) != tuple(INCLUDED_LEVELS):
        raise ValueError("polar distillation level order changed")
    run = prereg.get("run", {})
    if (
        run.get("policy_architecture") != "entity_polar"
        or run.get("aim_head_initialization") != "scaled_identity"
        or run.get("aim_loss_scope") != "fire_only"
        or run.get("aim_loss_mode") != "categorical"
        or run.get("teacher_execution_probability") != 1.0
        or run.get("exact_action_masks_before_every_decision") is not True
    ):
        raise ValueError("polar semantic contract changed")
    training = master_value["seed_registry"]["training"]
    first = int(run["training_seed_base"])
    last = int(run["training_seed_last_consumed"])
    if last != first + len(INCLUDED_LEVELS) - 1:
        raise ValueError("polar training seed interval is not one paired full55 pass")
    if not int(training["first"]) <= first <= last <= int(training["last"]):
        raise ValueError("polar training seeds escaped the registry")
    boundary = prereg.get("authority_boundary", {})
    if any(
        boundary.get(name) is not False
        for name in (
            "current_campaign_candidate_authority",
            "formal_selection_seed_consumption",
            "formal_final_blind_seed_consumption",
            "continuous_campaign_seed_consumption",
        )
    ):
        raise ValueError("polar engineering authority boundary changed")
    return prereg


def _prototype_environment(*, original_root: Path, max_ticks: int) -> RevengeEnv:
    return RevengeEnv(
        config=full55_environment_config(max_ticks=max_ticks),
        level_id=INCLUDED_LEVELS[0],
        root=original_root,
        hard=False,
        curve_index=0,
        profile_mode=SUPPORTED_PROFILE_MODE,
    )


def _evaluate_full55_dataset(
    *,
    model: Any,
    dataset: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    batch_size: int,
) -> dict[str, Any]:
    """Evaluate the frozen training set while supporting the fourth hop verb."""

    import torch

    action_nvec = tuple(int(value) for value in model.action_space.nvec)
    if action_nvec != EXPECTED_ACTION_NVEC:
        raise ValueError("dataset evaluation requires the full55 action interface")
    confusion = np.zeros((action_nvec[0], action_nvec[0]), dtype=np.int64)
    exact = verb_exact = fire_count = fire_aim_exact = fire_within_three = 0
    fire_distances: list[int] = []
    predicted_fire_bins: set[int] = set()
    expected_fire_bins: set[int] = set()
    model.policy.set_training_mode(False)
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            batch = dataset[start : start + batch_size]
            observations = np.asarray([row[0] for row in batch], dtype=np.float32)
            actions = np.asarray([row[1] for row in batch], dtype=np.int64)
            masks = np.asarray([row[2] for row in batch], dtype=np.bool_)
            observation_tensor, _ = model.policy.obs_to_tensor(observations)
            distribution = model.policy.get_distribution(
                observation_tensor,
                action_masks=masks,
            )
            predicted = distribution.get_actions(deterministic=True)
            predicted_array = predicted.detach().cpu().numpy()
            exact += int(np.sum(np.all(predicted_array == actions, axis=1)))
            verb_exact += int(np.sum(predicted_array[:, 0] == actions[:, 0]))
            for expected, actual in zip(
                actions[:, 0], predicted_array[:, 0], strict=True
            ):
                confusion[int(expected), int(actual)] += 1
            distance = np.abs(predicted_array[:, 1] - actions[:, 1])
            distance = np.minimum(distance, action_nvec[1] - distance)
            fire = actions[:, 0] == 1
            fire_count += int(np.sum(fire))
            fire_aim_exact += int(np.sum(fire & (distance == 0)))
            fire_within_three += int(np.sum(fire & (distance <= 3)))
            fire_distances.extend(int(value) for value in distance[fire].tolist())
            expected_fire_bins.update(int(value) for value in actions[fire, 1])
            predicted_fire_bins.update(
                int(value) for value in predicted_array[fire, 1]
            )
    count = len(dataset)
    distance_array = np.asarray(fire_distances, dtype=np.float64)
    return {
        "sample_count": count,
        "exact_action_accuracy": exact / count,
        "verb_accuracy": verb_exact / count,
        "verb_confusion_expected_rows_predicted_columns": confusion.tolist(),
        "fire_aim_sample_count": fire_count,
        "fire_aim_exact_accuracy": fire_aim_exact / fire_count,
        "fire_aim_within_three_accuracy": fire_within_three / fire_count,
        "fire_aim_mean_circular_bin_error": float(np.mean(distance_array)),
        "fire_aim_median_circular_bin_error": float(np.median(distance_array)),
        "fire_aim_p90_circular_bin_error": float(
            np.quantile(distance_array, 0.9)
        ),
        "fire_expected_unique_aim_bins": len(expected_fire_bins),
        "fire_predicted_unique_aim_bins": len(predicted_fire_bins),
    }


def run(*, preregistration_path: Path, original_root: Path) -> dict[str, Any]:
    preregistration_path = preregistration_path.resolve(strict=True)
    original_root = original_root.resolve(strict=True)
    prereg = _validate_preregistration(preregistration_path)
    run_config = prereg["run"]
    run_dir = Path(str(run_config["run_dir"])).resolve()
    if run_dir.exists():
        raise FileExistsError(f"polar distillation run already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    started = time.perf_counter()
    legacy._write_json_atomic(
        run_dir / "config.json",
        {
            "schema": "zuma-rl.alphazuma-55-polar-distillation-config",
            "version": 1,
            "status": "FROZEN",
            "preregistration": {
                "path": str(preregistration_path),
                "sha256": legacy._sha256(preregistration_path),
            },
            "trainer": {
                "path": str(SCRIPT_PATH),
                "sha256": legacy._sha256(SCRIPT_PATH),
            },
            "run": run_config,
        },
    )

    prototype: RevengeEnv | None = None
    model: Any | None = None
    try:
        from sb3_contrib import MaskablePPO
        import torch

        device = str(run_config["device"])
        prototype = _prototype_environment(
            original_root=original_root,
            max_ticks=int(run_config["max_ticks"]),
        )
        policy_kwargs = revenge_polar_policy_kwargs(prototype)
        model = MaskablePPO(
            "MlpPolicy",
            prototype,
            learning_rate=float(run_config["learning_rate"]),
            n_steps=512,
            batch_size=int(run_config["batch_size"]),
            gamma=0.995,
            gae_lambda=0.95,
            ent_coef=0.0,
            policy_kwargs=policy_kwargs,
            seed=int(run_config["model_seed"]),
            device=device,
            verbose=0,
        )
        initialize_polar_aim_head(
            model,
            scale=float(run_config["aim_head_identity_scale"]),
        )
        if tuple(int(value) for value in model.observation_space.shape) != (
            EXPECTED_OBSERVATION_SHAPE
        ):
            raise ValueError("polar model observation space is not full55-v1")
        if tuple(int(value) for value in model.action_space.nvec) != (
            EXPECTED_ACTION_NVEC
        ):
            raise ValueError("polar model action space is not full55-v1")

        dataset, collection = settled._collect_round(
            original_root=original_root,
            level_ids=list(INCLUDED_LEVELS),
            round_index=0,
            seed_base=int(run_config["training_seed_base"]),
            parallel_envs=int(run_config["parallel_envs"]),
            max_ticks=int(run_config["max_ticks"]),
            capacities={
                name: int(run_config["reservoir_capacity_by_verb"][name])
                for name in settled.VERB_NAMES
            },
            strides={
                name: int(run_config["sample_stride_by_verb"][name])
                for name in settled.VERB_NAMES
            },
            model_seed=int(run_config["model_seed"]),
        )

        def write_status(
            completed_epochs: int,
            epoch_rows: list[dict[str, Any]],
            checkpoints: list[dict[str, Any]],
        ) -> None:
            legacy._write_json_atomic(
                run_dir / "training_status.json",
                {
                    "schema": "zuma-rl.alphazuma-55-polar-distillation-status",
                    "version": 1,
                    "status": "RUNNING",
                    "stage": "OPTIMIZING",
                    "updated_utc": _utc_now(),
                    "collection": collection,
                    "aggregate_samples": len(dataset),
                    "completed_epochs": completed_epochs,
                    "expected_epochs": int(run_config["epochs_per_round"]),
                    "epochs": epoch_rows,
                    "checkpoints": checkpoints,
                    "wall_seconds": time.perf_counter() - started,
                },
            )

        write_status(0, [], [])
        optimization = settled._optimize(
            model=model,
            dataset=dataset,
            run=run_config,
            run_dir=run_dir,
            status_callback=write_status,
        )
        capacity_metrics = _evaluate_full55_dataset(
            model=model,
            dataset=dataset,
            batch_size=int(run_config["capacity_eval_batch_size"]),
        )
        model.num_timesteps = sum(
            int(row["steps"]) for row in collection["episodes"]
        )
        final_path = run_dir / "final_model.zip"
        model.save(final_path)
        completion = {
            "schema": "zuma-rl.alphazuma-55-polar-distillation-completion",
            "version": 1,
            "status": "COMPLETE",
            "completed_utc": _utc_now(),
            "wall_seconds": time.perf_counter() - started,
            "policy_architecture": "entity_polar",
            "policy_parameter_count": sum(
                parameter.numel() for parameter in model.policy.parameters()
            ),
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "torch": torch.__version__,
                "device": device,
                "cuda_device": torch.cuda.get_device_name(torch.device(device)),
            },
            "teacher_policy_id": POLICY_ID,
            "collection": collection,
            "optimization": optimization,
            "capacity_metrics_on_consumed_training_data": capacity_metrics,
            "final_model": {
                "path": str(final_path),
                "sha256": legacy._sha256(final_path),
                "bytes": final_path.stat().st_size,
                "num_timesteps": int(model.num_timesteps),
            },
            "formal_seed_consumption": False,
            "formal_candidate_authority": False,
        }
        legacy._write_json_atomic(run_dir / "completion.json", completion)
        return completion
    except BaseException as error:
        legacy._write_json_atomic(
            run_dir / "failure.json",
            {
                "schema": "zuma-rl.alphazuma-55-polar-distillation-failure",
                "version": 1,
                "status": "FAILED",
                "failed_utc": _utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
                "formal_seed_consumption": False,
            },
        )
        raise
    finally:
        if prototype is not None:
            prototype.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--original-root", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(
        preregistration_path=args.preregistration.expanduser(),
        original_root=args.original_root.expanduser(),
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "final_model": result["final_model"],
                "capacity_metrics": result[
                    "capacity_metrics_on_consumed_training_data"
                ],
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
