"""Continue a reset-aim V3 clone on four fresh teacher-seed rounds."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tools import distill_alphazuma_55_settled_v3 as legacy


SCRIPT_PATH = Path(__file__).resolve()


def _validate_preregistration(path: Path) -> dict[str, Any]:
    prereg = legacy.legacy._read_json(path)
    if (
        prereg.get("schema")
        != "zuma-rl.alphazuma-55-settled-v3-multiseed-preregistration"
        or prereg.get("version") != 1
        or prereg.get("status") != "FROZEN_BEFORE_TRAINING"
    ):
        raise ValueError("unexpected multiseed preregistration")
    trainer = prereg["trainer"]
    if (
        Path(str(trainer["path"])).resolve() != SCRIPT_PATH
        or trainer["sha256"] != legacy.legacy._sha256(SCRIPT_PATH)
    ):
        raise ValueError("multiseed trainer bytes differ")
    for name, artifact in prereg["implementation"].items():
        artifact_path = Path(str(artifact["path"])).resolve(strict=True)
        if artifact["sha256"] != legacy.legacy._sha256(artifact_path):
            raise ValueError(f"implementation bytes differ: {name}")
    master_reference = prereg["master_preregistration"]
    master_path = Path(str(master_reference["path"])).resolve(strict=True)
    if master_reference["sha256"] != legacy.legacy._sha256(master_path):
        raise ValueError("master preregistration bytes differ")
    source = prereg["student_source_model"]
    source_path = Path(str(source["path"])).resolve(strict=True)
    if source["sha256"] != legacy.legacy._sha256(source_path):
        raise ValueError("multiseed source model bytes differ")
    completion_reference = prereg["source_completion"]
    completion_path = Path(str(completion_reference["path"])).resolve(strict=True)
    if completion_reference["sha256"] != legacy.legacy._sha256(completion_path):
        raise ValueError("source completion bytes differ")
    completion = legacy.legacy._read_json(completion_path)
    if (
        completion.get("status") != "COMPLETE"
        or completion.get("final_model", {}).get("sha256") != source["sha256"]
        or Path(str(completion["final_model"]["path"])).resolve() != source_path
        or completion.get("aim_head_reset", {}).get("status")
        != "APPLIED_BEFORE_OPTIMIZER_REPLACEMENT_AND_COLLECTION"
    ):
        raise ValueError("source completion does not bind a reset-aim model")
    parent_reference = prereg["continuation_lineage"]["parent_preregistration"]
    parent_path = Path(str(parent_reference["path"])).resolve(strict=True)
    if parent_reference["sha256"] != legacy.legacy._sha256(parent_path):
        raise ValueError("parent preregistration bytes differ")
    for name, policy_id in (
        ("unmasked_fresh_55", "curve-aware-settled-strategic-actor-observable-v3"),
        ("exact_mask_fresh_55", legacy.POLICY_ID),
    ):
        legacy._validate_probe_result(
            prereg["teacher_evidence"][name], expected_policy_id=policy_id
        )
    if prereg["levels"] != list(legacy.INCLUDED_LEVELS):
        raise ValueError("multiseed level scope differs from AlphaZuma 55")
    run = prereg["run"]
    run_dir = Path(str(run["run_dir"])).resolve()
    if run_dir.exists():
        raise FileExistsError(f"refusing to reuse multiseed run: {run_dir}")
    if (
        int(run["rounds"]) != 4
        or int(run["parallel_envs"]) != 24
        or int(run["max_ticks"]) != 30_000
        or int(run["epochs_per_round"]) != 16
        or int(run["checkpoint_interval_epochs"]) != 4
        or run.get("dataset_aggregation")
        != "four_fresh_teacher_rounds_before_joint_optimization"
        or run.get("teacher_execution_probability") != 1.0
        or run.get("aim_loss_scope") != "all_actions"
        or run.get("aim_loss_mode") != "circular_smoothed"
    ):
        raise ValueError("unexpected multiseed training recipe")
    if not math.isclose(
        sum(float(value) for value in run["target_batch_fraction_by_verb"].values()),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("multiseed batch fractions do not sum to one")
    first_seed = int(run["training_seed_base"])
    last_seed = first_seed + int(run["rounds"]) * 55 - 1
    if int(run["training_seed_last_consumed"]) != last_seed:
        raise ValueError("multiseed training seed receipt is inconsistent")
    registry = legacy.legacy._read_json(master_path)["seed_registry"]["training"]
    if not int(registry["first"]) <= first_seed <= last_seed <= int(registry["last"]):
        raise ValueError("multiseed seeds escaped the training registry")
    return prereg


def run_distillation(
    *, prereg_path: Path, prereg: dict[str, Any], original_root: Path
) -> dict[str, Any]:
    import torch
    from sb3_contrib import MaskablePPO

    torch.set_num_threads(1)
    run = prereg["run"]
    device = str(run["device"])
    run_dir = Path(str(run["run_dir"])).resolve()
    run_dir.mkdir(parents=True)
    started = time.perf_counter()
    source_path = Path(str(prereg["student_source_model"]["path"])).resolve(
        strict=True
    )
    legacy.legacy._write_json_atomic(
        run_dir / "config.json",
        {
            "schema": "zuma-rl.alphazuma-55-settled-v3-multiseed-config",
            "version": 1,
            "status": "STARTED",
            "started_utc": legacy._utc_now(),
            "preregistration": {
                "path": str(prereg_path),
                "sha256": legacy.legacy._sha256(prereg_path),
            },
            "trainer_sha256": legacy.legacy._sha256(SCRIPT_PATH),
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "torch": torch.__version__,
                "device": device,
                "cuda_device": torch.cuda.get_device_name(torch.device(device)),
            },
            "run": run,
        },
    )
    student: Any | None = None
    try:
        student = MaskablePPO.load(source_path, device=device)
        if tuple(int(value) for value in student.observation_space.shape) != (22_833,):
            raise ValueError("student observation space is not full55-v1")
        if tuple(int(value) for value in student.action_space.nvec) != (4, 180):
            raise ValueError("student action space is not full55-v1")
        optimizer_class = student.policy.optimizer_class
        optimizer_kwargs = dict(student.policy.optimizer_kwargs)
        student.policy.optimizer = optimizer_class(
            student.policy.parameters(),
            lr=float(run["learning_rate"]),
            **optimizer_kwargs,
        )
        aggregate: list[tuple[Any, Any, Any]] = []
        collections: list[dict[str, Any]] = []
        for round_index in range(int(run["rounds"])):
            dataset, receipt = legacy._collect_round(
                original_root=original_root,
                level_ids=list(legacy.INCLUDED_LEVELS),
                round_index=round_index,
                seed_base=int(run["training_seed_base"]),
                parallel_envs=int(run["parallel_envs"]),
                max_ticks=int(run["max_ticks"]),
                capacities={
                    name: int(run["reservoir_capacity_by_verb"][name])
                    for name in legacy.VERB_NAMES
                },
                strides={
                    name: int(run["sample_stride_by_verb"][name])
                    for name in legacy.VERB_NAMES
                },
                model_seed=int(run["model_seed"]),
            )
            aggregate.extend(dataset)
            collections.append(receipt)
            legacy.legacy._write_json_atomic(
                run_dir / "training_status.json",
                {
                    "schema": "zuma-rl.alphazuma-55-settled-v3-multiseed-status",
                    "version": 1,
                    "status": "RUNNING",
                    "stage": "COLLECTING" if round_index + 1 < int(run["rounds"]) else "COLLECTION_COMPLETE",
                    "updated_utc": legacy._utc_now(),
                    "completed_rounds": round_index + 1,
                    "expected_rounds": int(run["rounds"]),
                    "aggregate_samples": len(aggregate),
                    "collections": collections,
                    "wall_seconds": time.perf_counter() - started,
                },
            )

        def write_status(
            completed_epochs: int,
            epoch_rows: list[dict[str, Any]],
            checkpoints: list[dict[str, Any]],
        ) -> None:
            legacy.legacy._write_json_atomic(
                run_dir / "training_status.json",
                {
                    "schema": "zuma-rl.alphazuma-55-settled-v3-multiseed-status",
                    "version": 1,
                    "status": "RUNNING",
                    "stage": "OPTIMIZING",
                    "updated_utc": legacy._utc_now(),
                    "completed_rounds": int(run["rounds"]),
                    "expected_rounds": int(run["rounds"]),
                    "aggregate_samples": len(aggregate),
                    "collections": collections,
                    "completed_epochs": completed_epochs,
                    "expected_epochs": int(run["epochs_per_round"]),
                    "epochs": epoch_rows,
                    "checkpoints": checkpoints,
                    "wall_seconds": time.perf_counter() - started,
                },
            )

        write_status(0, [], [])
        optimization = legacy._optimize(
            model=student,
            dataset=aggregate,
            run=run,
            run_dir=run_dir,
            status_callback=write_status,
        )
        added_steps = sum(
            int(episode["steps"])
            for receipt in collections
            for episode in receipt["episodes"]
        )
        student.num_timesteps = int(student.num_timesteps) + added_steps
        final_path = run_dir / "final_model.zip"
        student.save(final_path)
        completion = {
            "schema": "zuma-rl.alphazuma-55-settled-v3-multiseed-completion",
            "version": 1,
            "status": "COMPLETE",
            "completed_utc": legacy._utc_now(),
            "wall_seconds": time.perf_counter() - started,
            "source_model": prereg["student_source_model"],
            "teacher_policy_id": legacy.POLICY_ID,
            "collections": collections,
            "aggregate_collection": {
                "rounds": len(collections),
                "episodes": sum(len(row["episodes"]) for row in collections),
                "wins": sum(int(row["wins"]) for row in collections),
                "losses": sum(int(row["losses"]) for row in collections),
                "truncations": sum(int(row["truncations"]) for row in collections),
                "capacity_overflows": sum(
                    int(row["capacity_overflows"]) for row in collections
                ),
                "retained_samples": len(aggregate),
            },
            "optimization": optimization,
            "final_model": {
                "path": str(final_path),
                "sha256": legacy.legacy._sha256(final_path),
                "bytes": final_path.stat().st_size,
                "num_timesteps": int(student.num_timesteps),
            },
            "formal_seed_consumption": False,
        }
        legacy.legacy._write_json_atomic(run_dir / "completion.json", completion)
        return completion
    except BaseException as error:
        legacy.legacy._write_json_atomic(
            run_dir / "failure.json",
            {
                "schema": "zuma-rl.alphazuma-55-settled-v3-multiseed-failure",
                "version": 1,
                "status": "FAILED",
                "failed_utc": legacy._utc_now(),
                "wall_seconds": time.perf_counter() - started,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise
    finally:
        student = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--original-root", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prereg_path = args.preregistration.expanduser().resolve(strict=True)
    original_root = args.original_root.expanduser().resolve(strict=True)
    prereg = _validate_preregistration(prereg_path)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "VALID",
                    "preregistration": str(prereg_path),
                    "sha256": legacy.legacy._sha256(prereg_path),
                    "run": prereg["run"],
                    "formal_seed_consumption": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    result = run_distillation(
        prereg_path=prereg_path, prereg=prereg, original_root=original_root
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "aggregate_collection": result["aggregate_collection"],
                "final_model": result["final_model"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
