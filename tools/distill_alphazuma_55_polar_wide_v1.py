"""Train a wider polar policy from three independent full55 teacher passes."""

from __future__ import annotations

import argparse
from collections import Counter
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

from tools import build_alphazuma_55_polar_distillation as registration
from tools import distill_alphazuma_55_polar_v1 as base
from tools import distill_alphazuma_55_settled_v3 as settled
from zuma_rl.alphazuma_55 import INCLUDED_LEVELS
from zuma_rl.revenge_features import (
    initialize_polar_aim_head,
    revenge_polar_policy_kwargs,
)


SCRIPT_PATH = Path(__file__).resolve()
EXPECTED_OBSERVATION_SHAPE = (22_833,)
EXPECTED_ACTION_NVEC = (4, 180)
EXPECTED_FEATURES_DIM = 1024
EXPECTED_PASSES = 3


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_preregistration(path: Path) -> dict[str, Any]:
    prereg = base.legacy._read_json(path)
    if (
        prereg.get("schema")
        != "zuma-rl.alphazuma-55-polar-wide-distillation-preregistration"
        or prereg.get("version") != 1
        or prereg.get("status") != "FROZEN_BEFORE_TRAINING"
    ):
        raise ValueError("unexpected wide polar preregistration")
    trainer = prereg.get("trainer", {})
    if (
        Path(str(trainer.get("path", ""))).resolve() != SCRIPT_PATH
        or trainer.get("sha256") != base.legacy._sha256(SCRIPT_PATH)
    ):
        raise ValueError("wide polar trainer binding differs")
    builder = prereg.get("builder", {})
    builder_path = Path(str(builder.get("path", ""))).resolve(strict=True)
    if builder.get("sha256") != base.legacy._sha256(builder_path):
        raise ValueError("wide polar builder binding differs")
    master = prereg.get("master_preregistration", {})
    master_path = Path(str(master.get("path", ""))).resolve(strict=True)
    if master.get("sha256") != base.legacy._sha256(master_path):
        raise ValueError("wide polar master bytes differ")
    master_value = base.legacy._read_json(master_path)
    if (
        master_value.get("schema")
        != "zuma-rl.alphazuma-55-weekend-master-preregistration"
    ):
        raise ValueError("unexpected wide polar master")
    for name, artifact in prereg.get("implementation", {}).items():
        artifact_path = Path(str(artifact.get("path", ""))).resolve(strict=True)
        if artifact.get("sha256") != base.legacy._sha256(artifact_path):
            raise ValueError(f"wide polar implementation changed: {name}")
    teacher_path = Path(
        str(prereg["teacher_evidence"]["exact_mask_fresh_55"]["path"])
    ).resolve(strict=True)
    if (
        prereg["teacher_evidence"]["exact_mask_fresh_55"]["sha256"]
        != base.legacy._sha256(teacher_path)
    ):
        raise ValueError("wide polar teacher evidence changed")
    registration._validate_probe(teacher_path)
    if tuple(prereg.get("levels", ())) != tuple(INCLUDED_LEVELS):
        raise ValueError("wide polar level order changed")
    run = prereg.get("run", {})
    if not (
        run.get("policy_architecture") == "entity_polar_wide"
        and int(run.get("learned_features_dim", -1)) == EXPECTED_FEATURES_DIM
        and int(run.get("teacher_passes", -1)) == EXPECTED_PASSES
        and run.get("aim_head_initialization") == "scaled_identity"
        and run.get("aim_loss_scope") == "fire_only"
        and run.get("aim_loss_mode") == "categorical"
        and run.get("teacher_execution_probability") == 1.0
        and run.get("exact_action_masks_before_every_decision") is True
    ):
        raise ValueError("wide polar semantic contract changed")
    first = int(run["training_seed_base"])
    last = int(run["training_seed_last_consumed"])
    if last != first + EXPECTED_PASSES * len(INCLUDED_LEVELS) - 1:
        raise ValueError("wide polar training seed interval changed")
    training = master_value["seed_registry"]["training"]
    if not int(training["first"]) <= first <= last <= int(training["last"]):
        raise ValueError("wide polar training seeds escaped the registry")
    boundary = prereg.get("authority_boundary", {})
    for name in (
        "current_campaign_candidate_authority",
        "s99081535_successor_candidate_authority",
        "formal_selection_seed_consumption",
        "formal_final_blind_seed_consumption",
        "continuous_campaign_seed_consumption",
    ):
        if boundary.get(name) is not False:
            raise ValueError(f"wide polar authority changed: {name}")
    return prereg


def _wide_policy_kwargs(prototype: Any, features_dim: int) -> dict[str, Any]:
    if int(features_dim) != EXPECTED_FEATURES_DIM:
        raise ValueError("wide polar features dimension differs")
    kwargs = revenge_polar_policy_kwargs(prototype)
    extractor = dict(kwargs["features_extractor_kwargs"])
    extractor["features_dim"] = int(features_dim)
    kwargs["features_extractor_kwargs"] = extractor
    return kwargs


def _collection_summary(collections: list[dict[str, Any]]) -> dict[str, Any]:
    verbs: Counter[str] = Counter()
    samples = wins = losses = truncations = steps = 0
    for collection in collections:
        samples += int(collection["retained_samples"])
        wins += int(collection["wins"])
        losses += int(collection["losses"])
        truncations += int(collection["truncations"])
        verbs.update(collection["retained_samples_by_verb"])
        steps += sum(int(row["steps"]) for row in collection["episodes"])
    return {
        "teacher_passes_completed": len(collections),
        "episodes": len(collections) * len(INCLUDED_LEVELS),
        "wins": wins,
        "losses": losses,
        "truncations": truncations,
        "retained_samples": samples,
        "retained_samples_by_verb": dict(verbs),
        "teacher_steps": steps,
    }


def _build_model(prototype: Any, config: dict[str, Any]) -> Any:
    from sb3_contrib import MaskablePPO

    model = MaskablePPO(
        "MlpPolicy",
        prototype,
        learning_rate=float(config["learning_rate"]),
        n_steps=512,
        batch_size=int(config["batch_size"]),
        gamma=0.995,
        gae_lambda=0.95,
        ent_coef=0.0,
        policy_kwargs=_wide_policy_kwargs(
            prototype, int(config["learned_features_dim"])
        ),
        seed=int(config["model_seed"]),
        device=str(config["device"]),
        verbose=0,
    )
    initialize_polar_aim_head(
        model, scale=float(config["aim_head_identity_scale"])
    )
    if tuple(int(value) for value in model.observation_space.shape) != (
        EXPECTED_OBSERVATION_SHAPE
    ):
        raise ValueError("wide polar observation space is not full55-v1")
    if tuple(int(value) for value in model.action_space.nvec) != (
        EXPECTED_ACTION_NVEC
    ):
        raise ValueError("wide polar action space is not full55-v1")
    learned = int(model.policy.features_extractor.learned_features_dim)
    if learned != EXPECTED_FEATURES_DIM:
        raise ValueError("wide polar extractor capacity differs")
    return model


def run(*, preregistration_path: Path, original_root: Path) -> dict[str, Any]:
    preregistration_path = preregistration_path.resolve(strict=True)
    original_root = original_root.resolve(strict=True)
    prereg = _validate_preregistration(preregistration_path)
    config = prereg["run"]
    run_dir = Path(str(config["run_dir"])).resolve()
    if run_dir.exists():
        raise FileExistsError(f"wide polar run already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    started = time.perf_counter()
    base.legacy._write_json_atomic(
        run_dir / "config.json",
        {
            "schema": "zuma-rl.alphazuma-55-polar-wide-distillation-config",
            "version": 1,
            "status": "FROZEN",
            "preregistration": {
                "path": str(preregistration_path),
                "sha256": base.legacy._sha256(preregistration_path),
            },
            "trainer": {
                "path": str(SCRIPT_PATH),
                "sha256": base.legacy._sha256(SCRIPT_PATH),
            },
            "run": config,
        },
    )
    prototype: Any | None = None
    model: Any | None = None
    try:
        import torch

        device = str(config["device"])
        prototype = base._prototype_environment(
            original_root=original_root,
            max_ticks=int(config["max_ticks"]),
        )
        model = _build_model(prototype, config)
        learned = int(model.policy.features_extractor.learned_features_dim)

        dataset: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        collections: list[dict[str, Any]] = []
        for pass_index in range(EXPECTED_PASSES):
            base.legacy._write_json_atomic(
                run_dir / "training_status.json",
                {
                    "schema": "zuma-rl.alphazuma-55-polar-wide-distillation-status",
                    "version": 1,
                    "status": "RUNNING",
                    "stage": "COLLECTING",
                    "updated_utc": _utc_now(),
                    "current_teacher_pass": pass_index,
                    "expected_teacher_passes": EXPECTED_PASSES,
                    "completed_collections": collections,
                    "aggregate": _collection_summary(collections),
                    "wall_seconds": time.perf_counter() - started,
                },
            )
            pass_dataset, collection = settled._collect_round(
                original_root=original_root,
                level_ids=list(INCLUDED_LEVELS),
                round_index=pass_index,
                seed_base=int(config["training_seed_base"]),
                parallel_envs=int(config["parallel_envs"]),
                max_ticks=int(config["max_ticks"]),
                capacities={
                    name: int(config["reservoir_capacity_by_verb"][name])
                    for name in settled.VERB_NAMES
                },
                strides={
                    name: int(config["sample_stride_by_verb"][name])
                    for name in settled.VERB_NAMES
                },
                model_seed=int(config["model_seed"]),
            )
            dataset.extend(pass_dataset)
            collections.append(collection)

        def write_status(
            completed_epochs: int,
            epoch_rows: list[dict[str, Any]],
            checkpoints: list[dict[str, Any]],
        ) -> None:
            base.legacy._write_json_atomic(
                run_dir / "training_status.json",
                {
                    "schema": "zuma-rl.alphazuma-55-polar-wide-distillation-status",
                    "version": 1,
                    "status": "RUNNING",
                    "stage": "OPTIMIZING",
                    "updated_utc": _utc_now(),
                    "collections": collections,
                    "aggregate": _collection_summary(collections),
                    "aggregate_samples": len(dataset),
                    "completed_epochs": completed_epochs,
                    "expected_epochs": int(config["epochs_per_round"]),
                    "epochs": epoch_rows,
                    "checkpoints": checkpoints,
                    "wall_seconds": time.perf_counter() - started,
                },
            )

        write_status(0, [], [])
        optimization = settled._optimize(
            model=model,
            dataset=dataset,
            run=config,
            run_dir=run_dir,
            status_callback=write_status,
        )
        capacity = base._evaluate_full55_dataset(
            model=model,
            dataset=dataset,
            batch_size=int(config["capacity_eval_batch_size"]),
        )
        collection_summary = _collection_summary(collections)
        model.num_timesteps = int(collection_summary["teacher_steps"])
        final_path = run_dir / "final_model.zip"
        model.save(final_path)
        completion = {
            "schema": "zuma-rl.alphazuma-55-polar-wide-distillation-completion",
            "version": 1,
            "status": "COMPLETE",
            "completed_utc": _utc_now(),
            "wall_seconds": time.perf_counter() - started,
            "policy_architecture": "entity_polar_wide",
            "learned_features_dim": learned,
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
            "teacher_policy_id": base.POLICY_ID,
            "collections": collections,
            "collection_summary": collection_summary,
            "aggregate_samples": len(dataset),
            "optimization": optimization,
            "capacity_metrics_on_consumed_training_data": capacity,
            "final_model": {
                "path": str(final_path),
                "sha256": base.legacy._sha256(final_path),
                "bytes": final_path.stat().st_size,
                "num_timesteps": int(model.num_timesteps),
            },
            "formal_seed_consumption": False,
            "formal_candidate_authority": False,
        }
        base.legacy._write_json_atomic(run_dir / "completion.json", completion)
        return completion
    except BaseException as error:
        base.legacy._write_json_atomic(
            run_dir / "failure.json",
            {
                "schema": "zuma-rl.alphazuma-55-polar-wide-distillation-failure",
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
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.validate_only:
        preregistration_path = args.preregistration.expanduser().resolve(
            strict=True
        )
        original_root = args.original_root.expanduser().resolve(strict=True)
        prereg = _validate_preregistration(preregistration_path)
        prototype = base._prototype_environment(
            original_root=original_root,
            max_ticks=int(prereg["run"]["max_ticks"]),
        )
        try:
            model = _build_model(prototype, prereg["run"])
            print(
                json.dumps(
                    {
                        "status": "VALID",
                        "preregistration": {
                            "path": str(preregistration_path),
                            "sha256": base.legacy._sha256(preregistration_path),
                        },
                        "policy_architecture": "entity_polar_wide",
                        "learned_features_dim": int(
                            model.policy.features_extractor.learned_features_dim
                        ),
                        "policy_parameter_count": sum(
                            parameter.numel()
                            for parameter in model.policy.parameters()
                        ),
                        "observation_shape": list(
                            model.observation_space.shape
                        ),
                        "action_nvec": [
                            int(value) for value in model.action_space.nvec
                        ],
                        "formal_seed_consumption": False,
                    },
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                )
            )
            return 0
        finally:
            prototype.close()
    result = run(
        preregistration_path=args.preregistration.expanduser(),
        original_root=args.original_root.expanduser(),
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "final_model": result["final_model"],
                "policy_parameter_count": result["policy_parameter_count"],
                "aggregate_samples": result["aggregate_samples"],
                "capacity": result["capacity_metrics_on_consumed_training_data"],
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
