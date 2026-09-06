"""Train raw-intent DAgger while replaying frozen clean teacher states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, Sequence

import numpy as np
import torch

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tools import distill_alphazuma_55 as legacy
from tools import distill_alphazuma_55_polar_dagger_v1 as dagger
from tools import distill_alphazuma_55_polar_intent_dagger_v1 as intent_dagger
from tools import distill_alphazuma_55_polar_intent_wide_v1 as intent
from zuma_rl.alphazuma_55 import INCLUDED_LEVELS


SCRIPT_PATH = Path(__file__).resolve()
EXPECTED_SCHEDULE = (0.5, 0.1, 0.0)
EXPECTED_ANCHOR_PASSES = 2
EXPECTED_FEATURES_DIM = 1024
EXPECTED_EPOCHS_PER_ROUND = 12


def _validate_probability_schedule(values: Sequence[Any]) -> tuple[float, ...]:
    schedule = tuple(float(value) for value in values)
    if schedule != EXPECTED_SCHEDULE:
        raise ValueError(
            "replay-anchor schedule must remain exactly "
            f"{EXPECTED_SCHEDULE}"
        )
    return schedule


def _bound(reference: Any, label: str) -> Path:
    if not isinstance(reference, dict):
        raise ValueError(f"{label} reference must be an object")
    path = Path(str(reference.get("path", ""))).resolve(strict=True)
    if reference.get("sha256") != legacy._sha256(path):
        raise ValueError(f"{label} bytes differ: {path}")
    return path


def _validate_preregistration(path: Path) -> dict[str, Any]:
    prereg = dagger._read(path)
    if not (
        prereg.get("schema")
        == "zuma-rl.alphazuma-55-polar-intent-replay-anchor-preregistration"
        and prereg.get("version") == 1
        and prereg.get("status") == "FROZEN_BEFORE_TRAINING"
    ):
        raise ValueError("unexpected replay-anchor preregistration")
    trainer = prereg.get("trainer", {})
    if (
        Path(str(trainer.get("path", ""))).resolve() != SCRIPT_PATH
        or trainer.get("sha256") != legacy._sha256(SCRIPT_PATH)
    ):
        raise ValueError("replay-anchor trainer binding differs")
    master_path = _bound(prereg.get("master_preregistration"), "master")
    master = dagger._read(master_path)
    if (
        master.get("schema")
        != "zuma-rl.alphazuma-55-weekend-master-preregistration"
    ):
        raise ValueError("unexpected replay-anchor master")
    _bound(prereg.get("materialization_plan"), "materialization plan")
    for name, artifact in prereg.get("implementation", {}).items():
        _bound(artifact, f"implementation {name}")
    evidence_path = _bound(
        prereg.get("teacher_evidence", {}).get("exact_mask_fresh_55", {}),
        "teacher evidence",
    )
    intent.registration._validate_probe(evidence_path)

    source = prereg.get("source", {})
    completion_path = Path(str(source.get("completion_path", ""))).resolve(
        strict=True
    )
    if source.get("completion_sha256") != legacy._sha256(completion_path):
        raise ValueError("source intent-wide completion bytes differ")
    completion = dagger._read(completion_path)
    if not (
        completion.get("schema")
        == "zuma-rl.alphazuma-55-polar-intent-wide-distillation-completion"
        and completion.get("status") == "COMPLETE"
        and completion.get("policy_architecture")
        == "entity_polar_intent_wide"
        and completion.get("training_label_semantics")
        == "raw_teacher_intent"
        and completion.get("online_policy_inference_semantics")
        == "exact_action_masks"
        and completion.get("formal_seed_consumption") is False
        and completion.get("formal_candidate_authority") is False
    ):
        raise ValueError("source intent-wide completion is not eligible")
    source_model = Path(str(source.get("model_path", ""))).resolve(strict=True)
    if source.get("model_sha256") != legacy._sha256(source_model):
        raise ValueError("source intent-wide model bytes differ")
    final = completion.get("final_model", {})
    if (
        Path(str(final.get("path", ""))).resolve() != source_model
        or final.get("sha256") != source.get("model_sha256")
    ):
        raise ValueError("source is not the frozen intent-wide final model")
    if tuple(prereg.get("levels", ())) != tuple(INCLUDED_LEVELS):
        raise ValueError("replay-anchor level order changed")

    run = prereg.get("run", {})
    schedule = _validate_probability_schedule(
        run.get("teacher_execution_probabilities", ())
    )
    if not (
        run.get("policy_architecture")
        == "entity_polar_intent_wide_replay_anchor"
        and int(run.get("learned_features_dim", -1))
        == EXPECTED_FEATURES_DIM
        and int(run.get("anchor_teacher_passes", -1))
        == EXPECTED_ANCHOR_PASSES
        and int(run.get("student_rounds", -1)) == len(schedule)
        and int(run.get("epochs_per_round", -1))
        == EXPECTED_EPOCHS_PER_ROUND
        and run.get("source_is_frozen_raw_intent_wide_final_model") is True
        and run.get("teacher_anchor_replayed_every_round") is True
        and run.get("aggregate_student_states_across_rounds") is True
        and run.get("raw_teacher_intent_labels") is True
        and run.get("training_mask_relaxation")
        == "unmask_only_the_teacher_target_verb"
        and run.get("exact_action_masks_for_environment_execution") is True
        and run.get("exact_action_masks_for_online_policy_inference") is True
        and run.get("label_aim_must_remain_exact_mask_legal") is True
        and run.get("final_runtime_teacher_calls_forbidden") is True
    ):
        raise ValueError("replay-anchor semantic contract changed")
    anchor_first = int(run["anchor_seed_base"])
    anchor_last = int(run["anchor_seed_last_consumed"])
    student_first = int(run["student_seed_base"])
    student_last = int(run["student_seed_last_consumed"])
    if anchor_last != (
        anchor_first + len(INCLUDED_LEVELS) * EXPECTED_ANCHOR_PASSES - 1
    ):
        raise ValueError("replay-anchor teacher seed interval changed")
    if student_first != anchor_last + 1:
        raise ValueError("replay-anchor seed intervals are not contiguous")
    if student_last != student_first + len(INCLUDED_LEVELS) * len(schedule) - 1:
        raise ValueError("replay-anchor student seed interval changed")
    registry = master["seed_registry"]["training"]
    if not (
        int(registry["first"])
        <= anchor_first
        <= student_last
        <= int(registry["last"])
    ):
        raise ValueError("replay-anchor seeds escaped registry")
    boundary = prereg.get("authority_boundary", {})
    for name in (
        "current_campaign_candidate_authority",
        "formal_selection_seed_consumption",
        "formal_final_blind_seed_consumption",
        "continuous_campaign_seed_consumption",
    ):
        if boundary.get(name) is not False:
            raise ValueError(f"replay-anchor authority changed: {name}")
    return prereg


def run(*, preregistration_path: Path, original_root: Path) -> dict[str, Any]:
    from sb3_contrib import MaskablePPO

    preregistration_path = preregistration_path.resolve(strict=True)
    prereg = _validate_preregistration(preregistration_path)
    run_config = prereg["run"]
    run_dir = Path(str(run_config["run_dir"])).resolve()
    if run_dir.exists():
        raise FileExistsError(f"replay-anchor run directory exists: {run_dir}")
    run_dir.mkdir(parents=True)
    started = time.perf_counter()
    dagger._write_atomic(
        run_dir / "config.json",
        {
            "schema": "zuma-rl.alphazuma-55-polar-intent-replay-anchor-config",
            "version": 1,
            "preregistration": {
                "path": str(preregistration_path),
                "sha256": legacy._sha256(preregistration_path),
            },
            "run": run_config,
            "formal_seed_consumption": False,
            "formal_candidate_authority": False,
        },
    )

    anchor_dataset: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    anchor_collections: list[dict[str, Any]] = []
    student_rounds: list[dict[str, Any]] = []

    def write_status(stage: str, **extra: Any) -> None:
        dagger._write_atomic(
            run_dir / "training_status.json",
            {
                "schema": (
                    "zuma-rl.alphazuma-55-polar-intent-replay-anchor-status"
                ),
                "version": 1,
                "status": "RUNNING",
                "stage": stage,
                "updated_utc": dagger._utc_now(),
                "completed_anchor_passes": len(anchor_collections),
                "expected_anchor_passes": EXPECTED_ANCHOR_PASSES,
                "anchor_samples": len(anchor_dataset),
                "completed_student_rounds": len(student_rounds),
                "expected_student_rounds": len(EXPECTED_SCHEDULE),
                "student_rounds": student_rounds,
                "wall_seconds": time.perf_counter() - started,
                **extra,
            },
        )

    prototype = dagger._prototype_environment(
        original_root=original_root,
        max_ticks=int(run_config["max_ticks"]),
    )
    try:
        model = MaskablePPO.load(
            Path(str(prereg["source"]["model_path"])),
            env=prototype,
            device=str(run_config["device"]),
        )
        learned = int(model.policy.features_extractor.learned_features_dim)
        if learned != EXPECTED_FEATURES_DIM:
            raise ValueError("replay-anchor source feature width changed")
        for group in model.policy.optimizer.param_groups:
            group["lr"] = float(run_config["learning_rate"])

        for anchor_index in range(EXPECTED_ANCHOR_PASSES):
            write_status("COLLECTING_TEACHER_ANCHOR", anchor_pass=anchor_index)
            collected, collection = intent._collect_intent_round(
                original_root=original_root,
                level_ids=INCLUDED_LEVELS,
                round_index=anchor_index,
                seed_base=int(run_config["anchor_seed_base"]),
                parallel_envs=int(run_config["parallel_envs"]),
                max_ticks=int(run_config["max_ticks"]),
                capacities={
                    name: int(run_config["reservoir_capacity_by_verb"][name])
                    for name in dagger.settled.VERB_NAMES
                },
                strides={
                    name: int(run_config["sample_stride_by_verb"][name])
                    for name in dagger.settled.VERB_NAMES
                },
                model_seed=int(run_config["model_seed"]),
            )
            if collection.get("wins") != len(INCLUDED_LEVELS):
                raise RuntimeError(
                    "teacher anchor failed a level: "
                    f"pass={anchor_index} wins={collection.get('wins')}"
                )
            anchor_dataset.extend(collected)
            anchor_collections.append(collection)
            write_status("TEACHER_ANCHOR_PASS_COMPLETE")

        aggregate_dataset = list(anchor_dataset)
        schedule = _validate_probability_schedule(
            run_config["teacher_execution_probabilities"]
        )
        for round_index, teacher_probability in enumerate(schedule):
            write_status(
                "COLLECTING_STUDENT_STATES",
                current_round=round_index,
                teacher_execution_probability=teacher_probability,
            )

            def collection_progress(progress: dict[str, Any]) -> None:
                write_status(
                    "COLLECTING_STUDENT_STATES",
                    current_round=round_index,
                    teacher_execution_probability=teacher_probability,
                    collection_progress=progress,
                )

            collected, collection = intent_dagger._collect_intent_dagger_round(
                original_root=original_root,
                level_ids=INCLUDED_LEVELS,
                student=model,
                round_index=round_index,
                seed_base=int(run_config["student_seed_base"]),
                parallel_envs=int(run_config["parallel_envs"]),
                max_ticks=int(run_config["max_ticks"]),
                capacities={
                    name: int(run_config["reservoir_capacity_by_verb"][name])
                    for name in dagger.settled.VERB_NAMES
                },
                strides={
                    name: int(run_config["sample_stride_by_verb"][name])
                    for name in dagger.settled.VERB_NAMES
                },
                teacher_execution_probability=teacher_probability,
                model_seed=int(run_config["model_seed"]),
                progress_callback=collection_progress,
            )
            aggregate_dataset.extend(collected)
            round_dir = run_dir / f"round_{round_index:02d}"
            round_dir.mkdir()

            def optimization_progress(
                completed_epochs: int,
                epoch_rows: list[dict[str, Any]],
                checkpoints: list[dict[str, Any]],
            ) -> None:
                write_status(
                    "OPTIMIZING_REPLAY_ANCHORED",
                    current_round=round_index,
                    teacher_execution_probability=teacher_probability,
                    collection=collection,
                    aggregate_samples=len(aggregate_dataset),
                    anchor_fraction=(
                        len(anchor_dataset) / len(aggregate_dataset)
                    ),
                    completed_epochs_in_round=completed_epochs,
                    expected_epochs_in_round=int(
                        run_config["epochs_per_round"]
                    ),
                    epochs=epoch_rows,
                    checkpoints=checkpoints,
                )

            optimization_run = dict(run_config)
            optimization_run["model_seed"] = (
                int(run_config["model_seed"]) + round_index * 1_000_003
            )
            optimization = dagger.settled._optimize(
                model=model,
                dataset=aggregate_dataset,
                run=optimization_run,
                run_dir=round_dir,
                status_callback=optimization_progress,
            )
            capacity = dagger.polar._evaluate_full55_dataset(
                model=model,
                dataset=aggregate_dataset,
                batch_size=int(run_config["capacity_eval_batch_size"]),
            )
            round_model = round_dir / "final_model.zip"
            model.save(round_model)
            student_rounds.append(
                {
                    "round_index": round_index,
                    "teacher_execution_probability": teacher_probability,
                    "collection": collection,
                    "anchor_samples": len(anchor_dataset),
                    "aggregate_samples": len(aggregate_dataset),
                    "anchor_fraction": (
                        len(anchor_dataset) / len(aggregate_dataset)
                    ),
                    "optimization": optimization,
                    "capacity_metrics_on_consumed_training_data": capacity,
                    "model": {
                        "path": str(round_model),
                        "sha256": legacy._sha256(round_model),
                        "bytes": round_model.stat().st_size,
                    },
                }
            )
            write_status("STUDENT_ROUND_COMPLETE", current_round=round_index)

        final_model = run_dir / "final_model.zip"
        model.num_timesteps = sum(
            int(episode["steps"])
            for collection in anchor_collections
            for episode in collection["episodes"]
        ) + sum(
            int(episode["steps"])
            for row in student_rounds
            for episode in row["collection"]["episodes"]
        )
        model.save(final_model)
        completion = {
            "schema": (
                "zuma-rl.alphazuma-55-polar-intent-replay-anchor-completion"
            ),
            "version": 1,
            "status": "COMPLETE",
            "completed_utc": dagger._utc_now(),
            "wall_seconds": time.perf_counter() - started,
            "policy_architecture": (
                "entity_polar_intent_wide_replay_anchor"
            ),
            "policy_parameter_count": sum(
                parameter.numel() for parameter in model.policy.parameters()
            ),
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "torch": torch.__version__,
                "device": str(run_config["device"]),
                "cuda_device": torch.cuda.get_device_name(
                    torch.device(str(run_config["device"]))
                ),
            },
            "teacher_policy_id": dagger.POLICY_ID,
            "source": prereg["source"],
            "anchor_collections": anchor_collections,
            "anchor_samples": len(anchor_dataset),
            "student_rounds": student_rounds,
            "aggregate_samples": (
                student_rounds[-1]["aggregate_samples"]
            ),
            "final_capacity_metrics_on_consumed_training_data": (
                student_rounds[-1][
                    "capacity_metrics_on_consumed_training_data"
                ]
            ),
            "final_model": {
                "path": str(final_model),
                "sha256": legacy._sha256(final_model),
                "bytes": final_model.stat().st_size,
                "num_timesteps": int(model.num_timesteps),
            },
            "learned_features_dim": EXPECTED_FEATURES_DIM,
            "training_label_semantics": "raw_teacher_intent",
            "state_distribution_semantics": (
                "clean_teacher_replay_plus_dagger_student_states"
            ),
            "teacher_execution_probabilities": list(EXPECTED_SCHEDULE),
            "online_policy_inference_semantics": "exact_action_masks",
            "final_runtime_teacher_calls_forbidden": True,
            "formal_seed_consumption": False,
            "formal_candidate_authority": False,
        }
        dagger._write_atomic(run_dir / "completion.json", completion)
        return completion
    except BaseException as error:
        failure = {
            "schema": (
                "zuma-rl.alphazuma-55-polar-intent-replay-anchor-failure"
            ),
            "version": 1,
            "status": "ERROR",
            "failed_utc": dagger._utc_now(),
            "wall_seconds": time.perf_counter() - started,
            "error_type": type(error).__name__,
            "error": str(error),
            "formal_seed_consumption": False,
            "formal_candidate_authority": False,
        }
        dagger._write_atomic(run_dir / "failure.json", failure)
        raise
    finally:
        prototype.close()


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
                    "preregistration": {
                        "path": str(prereg_path),
                        "sha256": legacy._sha256(prereg_path),
                    },
                    "anchor_teacher_passes": EXPECTED_ANCHOR_PASSES,
                    "teacher_execution_probabilities": list(
                        EXPECTED_SCHEDULE
                    ),
                    "formal_seed_consumption": False,
                },
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        )
        return 0
    completion = run(
        preregistration_path=prereg_path,
        original_root=original_root,
    )
    print(
        json.dumps(
            {
                "status": completion["status"],
                "model": completion["final_model"],
                "state_distribution_semantics": completion[
                    "state_distribution_semantics"
                ],
                "formal_seed_consumption": False,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
