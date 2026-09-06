"""Replay-anchored DAgger with actor-visible elite-human motor state."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, Sequence

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import numpy as np

from tools import distill_alphazuma_55 as legacy
from tools import distill_alphazuma_55_polar_dagger_v1 as dagger
from tools import distill_alphazuma_55_polar_intent_dagger_v1 as intent_dagger
from tools import distill_alphazuma_55_polar_intent_wide_v1 as intent
from tools import distill_alphazuma_55_polar_v1 as polar
from tools import distill_alphazuma_55_polar_wide_v1 as wide
from tools import distill_alphazuma_55_settled_v3 as settled
from zuma_rl.alphazuma_55 import INCLUDED_LEVELS, full55_environment_config
from zuma_rl.human_speedrun import EliteHumanInputConfig, WinFirstRewardConfig
from zuma_rl.motor_observable_migration import (
    transplant_appended_global_policy,
)
from zuma_rl.observable_human_speedrun import (
    MOTOR_OBSERVATION_PROFILE,
    ObservableHumanSpeedrunWrapper,
)
from zuma_rl.revenge_env import RevengeEnv
from zuma_rl.revenge_teacher import RevengeTeacherSpec


SCRIPT_PATH = Path(__file__).resolve()
EXPECTED_FEATURES_DIM = 1024
EXPECTED_ANCHOR_PASSES = 3
EXPECTED_SCHEDULE = (0.5, 0.1, 0.0)
EXPECTED_EPOCHS = 4


class MotorTeacherSpecWrapper(ObservableHumanSpeedrunWrapper):
    """Expose only immutable teacher geometry alongside motor state."""

    def teacher_spec(self) -> dict[str, Any]:
        # The strategic teacher ignores the appended motor names, but its
        # shape contract must cover the full actor observation collected for
        # imitation.  All original game globals retain their exact indices.
        return asdict(RevengeTeacherSpec.from_env(self))


def _make_motor_env_factory(
    *,
    original_root: Path,
    level_id: str,
    max_ticks: int,
    input_config: EliteHumanInputConfig,
    reward_config: WinFirstRewardConfig,
) -> Any:
    def make_env() -> MotorTeacherSpecWrapper:
        base = RevengeEnv(
            config=full55_environment_config(max_ticks=max_ticks),
            level_id=level_id,
            root=original_root,
            hard=False,
            curve_index=0,
            profile_mode="tutorials_completed",
        )
        return MotorTeacherSpecWrapper(
            base,
            input_config=input_config,
            reward_config=reward_config,
        )

    return make_env


def _prototype(
    *, original_root: Path, max_ticks: int
) -> ObservableHumanSpeedrunWrapper:
    base = RevengeEnv(
        config=full55_environment_config(max_ticks=max_ticks),
        level_id=INCLUDED_LEVELS[0],
        root=original_root,
        hard=False,
        curve_index=0,
        profile_mode="tutorials_completed",
    )
    return ObservableHumanSpeedrunWrapper(
        base,
        input_config=EliteHumanInputConfig(
            profile_id="elite-human-v1",
            reaction_delay_ticks=12,
            max_aim_speed_degrees_per_second=1080.0,
            max_aim_acceleration_degrees_per_second_squared=18000.0,
            min_button_interval_ticks=5,
        ),
        reward_config=WinFirstRewardConfig(
            profile_id="win-time-score-v1",
            win_reward=10.0,
            failure_reward=-10.0,
            time_penalty_per_native_tick=-0.0001,
            score_progress_reward_cap=0.01,
        ),
    )


def _build_motor_model(
    prototype: ObservableHumanSpeedrunWrapper,
    config: dict[str, Any],
) -> Any:
    """Build the wide polar policy without rejecting the appended tail."""

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
        policy_kwargs=wide._wide_policy_kwargs(
            prototype, int(config["learned_features_dim"])
        ),
        seed=int(config["model_seed"]),
        device=str(config["device"]),
        verbose=0,
    )
    wide.initialize_polar_aim_head(
        model, scale=float(config["aim_head_identity_scale"])
    )
    expected_shape = (
        prototype.base_observation_size
        + len(prototype.motor_feature_names),
    )
    if tuple(int(value) for value in model.observation_space.shape) != (
        expected_shape
    ):
        raise ValueError("motor-observable observation shape differs")
    if tuple(int(value) for value in model.action_space.nvec) != (
        wide.EXPECTED_ACTION_NVEC
    ):
        raise ValueError("motor-observable action space differs")
    learned = int(model.policy.features_extractor.learned_features_dim)
    if learned != EXPECTED_FEATURES_DIM:
        raise ValueError("motor-observable extractor capacity differs")
    return model


def _validate_schedule(values: Sequence[Any]) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if result != EXPECTED_SCHEDULE:
        raise ValueError("motor-observable DAgger schedule changed")
    return result


def _validate_preregistration(path: Path) -> dict[str, Any]:
    prereg = legacy._read_json(path)
    if (
        prereg.get("schema")
        != "zuma-rl.alphazuma-55-motor-observable-replay-preregistration"
        or prereg.get("version") != 1
        or prereg.get("status") != "FROZEN_BEFORE_TRAINING"
    ):
        raise ValueError("unexpected motor-observable preregistration")
    trainer = prereg.get("trainer", {})
    if (
        Path(str(trainer.get("path", ""))).resolve() != SCRIPT_PATH
        or trainer.get("sha256") != legacy._sha256(SCRIPT_PATH)
    ):
        raise ValueError("motor-observable trainer binding differs")
    builder = prereg.get("builder", {})
    builder_path = Path(str(builder.get("path", ""))).resolve(strict=True)
    if builder.get("sha256") != legacy._sha256(builder_path):
        raise ValueError("motor-observable builder binding differs")
    master = prereg.get("master_preregistration", {})
    master_path = Path(str(master.get("path", ""))).resolve(strict=True)
    if master.get("sha256") != legacy._sha256(master_path):
        raise ValueError("motor-observable master bytes differ")
    incident = prereg.get("predecessor_startup_incident", {})
    incident_path = Path(str(incident.get("path", ""))).resolve(strict=True)
    if incident.get("sha256") != legacy._sha256(incident_path):
        raise ValueError("motor-observable predecessor incident differs")
    incident_value = legacy._read_json(incident_path)
    if not (
        incident_value.get("status")
        == "STARTUP_ABORTED_BEFORE_CONFIG_NO_SEEDS_CONSUMED"
        and incident_value.get("seed_boundary", {}).get(
            "training_seed_consumption"
        )
        == "NONE"
    ):
        raise ValueError("motor-observable predecessor consumed training seeds")
    for name, artifact in prereg.get("implementation", {}).items():
        artifact_path = Path(str(artifact.get("path", ""))).resolve(strict=True)
        if artifact.get("sha256") != legacy._sha256(artifact_path):
            raise ValueError(f"motor-observable implementation changed: {name}")
    source = prereg.get("source", {})
    source_path = Path(str(source.get("model_path", ""))).resolve(strict=True)
    if source.get("model_sha256") != legacy._sha256(source_path):
        raise ValueError("motor-observable source model bytes differ")
    audit_path = Path(str(source.get("selection_audit_path", ""))).resolve(
        strict=True
    )
    if source.get("selection_audit_sha256") != legacy._sha256(audit_path):
        raise ValueError("motor-observable source audit bytes differ")
    audit = legacy._read_json(audit_path)
    if audit.get("status") != "PASS":
        raise ValueError("motor-observable source audit is not PASS")
    if tuple(prereg.get("levels", ())) != tuple(INCLUDED_LEVELS):
        raise ValueError("motor-observable level order changed")

    run = prereg.get("run", {})
    if not (
        run.get("policy_architecture")
        == "entity_polar_intent_motor_observable_replay"
        and run.get("motor_observation_profile")
        == MOTOR_OBSERVATION_PROFILE
        and int(run.get("learned_features_dim", -1))
        == EXPECTED_FEATURES_DIM
        and int(run.get("anchor_teacher_passes", -1))
        == EXPECTED_ANCHOR_PASSES
        and int(run.get("student_rounds", -1)) == len(EXPECTED_SCHEDULE)
        and int(run.get("epochs_per_round", -1)) == EXPECTED_EPOCHS
        and int(run.get("checkpoint_interval_epochs", -1)) == 1
        and run.get("teacher_anchor_replayed_every_round") is True
        and run.get("aggregate_student_states_across_rounds") is True
        and run.get("raw_teacher_intent_labels") is True
        and run.get("aim_loss_scope") == "fire_only"
        and run.get("aim_loss_mode") == "circular_smoothed"
        and float(run.get("aim_smoothing_sigma", -1.0)) == 1.5
        and float(run.get("aim_distance_weight", -1.0)) == 0.5
        and run.get("final_runtime_teacher_calls_forbidden") is True
    ):
        raise ValueError("motor-observable semantic contract changed")
    _validate_schedule(run.get("teacher_execution_probabilities", ()))
    anchor_first = int(run["anchor_seed_base"])
    anchor_last = int(run["anchor_seed_last_consumed"])
    student_first = int(run["student_seed_base"])
    student_last = int(run["student_seed_last_consumed"])
    if anchor_last != (
        anchor_first + EXPECTED_ANCHOR_PASSES * len(INCLUDED_LEVELS) - 1
    ):
        raise ValueError("motor-observable anchor seed interval changed")
    if student_first != anchor_last + 1:
        raise ValueError("motor-observable seed intervals are not contiguous")
    if student_last != (
        student_first + len(EXPECTED_SCHEDULE) * len(INCLUDED_LEVELS) - 1
    ):
        raise ValueError("motor-observable student seed interval changed")
    master_value = legacy._read_json(master_path)
    training = master_value["seed_registry"]["training"]
    if not (
        int(training["first"])
        <= anchor_first
        <= student_last
        <= int(training["last"])
    ):
        raise ValueError("motor-observable seeds escaped training registry")
    boundary = prereg.get("authority_boundary", {})
    for name in (
        "current_campaign_candidate_authority",
        "formal_selection_seed_consumption",
        "formal_final_blind_seed_consumption",
        "continuous_campaign_seed_consumption",
    ):
        if boundary.get(name) is not False:
            raise ValueError(f"motor-observable authority changed: {name}")
    return prereg


def _verify_initial_equivalence(
    *, source: Any, target: Any, prototype: ObservableHumanSpeedrunWrapper
) -> dict[str, Any]:
    import torch

    observation, _ = prototype.reset(seed=1543000501)
    base = observation[: prototype.base_observation_size]
    mask = prototype.action_masks()
    source_observation, _ = source.policy.obs_to_tensor(base[None, :])
    target_observation, _ = target.policy.obs_to_tensor(observation[None, :])
    mask_batch = np.asarray(mask, dtype=np.bool_)[None, :]
    with torch.no_grad():
        source_distribution = source.policy.get_distribution(
            source_observation, action_masks=mask_batch
        )
        target_distribution = target.policy.get_distribution(
            target_observation, action_masks=mask_batch
        )
    source_logits = [
        component.logits.detach().cpu()
        for component in source_distribution.distributions
    ]
    target_logits = [
        component.logits.detach().cpu()
        for component in target_distribution.distributions
    ]
    if len(source_logits) != len(target_logits):
        raise RuntimeError("motor-observable action component count changed")
    exact = all(
        torch.equal(left, right)
        for left, right in zip(source_logits, target_logits, strict=True)
    )
    component_rows: list[dict[str, Any]] = []
    tolerance = 1.0e-5
    for index, (left, right) in enumerate(
        zip(source_logits, target_logits, strict=True)
    ):
        finite = torch.isfinite(left) & torch.isfinite(right)
        special_values_equal = torch.equal(
            torch.isposinf(left), torch.isposinf(right)
        ) and torch.equal(torch.isneginf(left), torch.isneginf(right))
        max_abs = (
            float(torch.max(torch.abs(left[finite] - right[finite])).item())
            if bool(finite.any())
            else 0.0
        )
        component_rows.append(
            {
                "component": index,
                "max_abs_logit_difference": max_abs,
                "within_absolute_tolerance": (
                    special_values_equal and max_abs <= tolerance
                ),
                "deterministic_argmax_exact": bool(
                    torch.equal(torch.argmax(left, dim=-1), torch.argmax(right, dim=-1))
                ),
            }
        )
    equivalent = all(
        row["within_absolute_tolerance"]
        and row["deterministic_argmax_exact"]
        for row in component_rows
    )
    if not equivalent:
        raise RuntimeError(
            "motor-observable initial policy exceeds source-equivalence "
            f"tolerance: {component_rows}"
        )
    return {
        "schema": "zuma-rl.motor-observable-initial-equivalence",
        "version": 1,
        "status": "PASS",
        "source_observation_size": int(base.shape[0]),
        "target_observation_size": int(observation.shape[0]),
        "action_component_logits_bit_exact": exact,
        "absolute_logit_tolerance": tolerance,
        "deterministic_component_argmax_exact": True,
        "components": component_rows,
    }


def run(*, preregistration_path: Path, original_root: Path) -> dict[str, Any]:
    import torch
    from sb3_contrib import MaskablePPO

    torch.set_num_threads(1)
    preregistration_path = preregistration_path.resolve(strict=True)
    prereg = _validate_preregistration(preregistration_path)
    config = prereg["run"]
    run_dir = Path(str(config["run_dir"])).resolve()
    if run_dir.exists():
        raise FileExistsError(f"motor-observable run directory exists: {run_dir}")
    run_dir.mkdir(parents=True)
    started = time.perf_counter()
    dagger._write_atomic(
        run_dir / "config.json",
        {
            "schema": "zuma-rl.alphazuma-55-motor-observable-replay-config",
            "version": 1,
            "preregistration": {
                "path": str(preregistration_path),
                "sha256": legacy._sha256(preregistration_path),
            },
            "run": config,
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
                "schema": "zuma-rl.alphazuma-55-motor-observable-replay-status",
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

    prototype = _prototype(
        original_root=original_root,
        max_ticks=int(config["max_ticks"]),
    )
    original_factory = legacy._make_env_factory
    try:
        source = MaskablePPO.load(
            Path(str(prereg["source"]["model_path"])),
            device=str(config["device"]),
        )
        target = _build_motor_model(prototype, config)
        migration = transplant_appended_global_policy(
            source_policy=source.policy,
            target_policy=target.policy,
            source_global_feature_size=int(
                prototype.revenge_env.global_feature_size
            ),
            target_global_feature_size=int(prototype.global_feature_size),
        )
        equivalence = _verify_initial_equivalence(
            source=source,
            target=target,
            prototype=prototype,
        )
        migration.update(
            {
                "source_model": prereg["source"],
                "target_policy_parameter_count": sum(
                    parameter.numel()
                    for parameter in target.policy.parameters()
                ),
                "equivalence": equivalence,
            }
        )
        dagger._write_atomic(run_dir / "migration.json", migration)
        model = target
        del source
        legacy._make_env_factory = _make_motor_env_factory

        for anchor_index in range(EXPECTED_ANCHOR_PASSES):
            write_status("COLLECTING_TEACHER_ANCHOR", anchor_pass=anchor_index)
            collected, collection = intent._collect_intent_round(
                original_root=original_root,
                level_ids=INCLUDED_LEVELS,
                round_index=anchor_index,
                seed_base=int(config["anchor_seed_base"]),
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
            if int(collection.get("wins", -1)) != len(INCLUDED_LEVELS):
                raise RuntimeError(
                    "motor-observable teacher anchor failed a level: "
                    f"pass={anchor_index} wins={collection.get('wins')}"
                )
            anchor_dataset.extend(collected)
            anchor_collections.append(collection)
            write_status("TEACHER_ANCHOR_PASS_COMPLETE")

        aggregate_dataset = list(anchor_dataset)
        for round_index, teacher_probability in enumerate(EXPECTED_SCHEDULE):
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

            collected, collection = (
                intent_dagger._collect_intent_dagger_round(
                    original_root=original_root,
                    level_ids=INCLUDED_LEVELS,
                    student=model,
                    round_index=round_index,
                    seed_base=int(config["student_seed_base"]),
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
                    teacher_execution_probability=teacher_probability,
                    model_seed=int(config["model_seed"]),
                    progress_callback=collection_progress,
                )
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
                    anchor_fraction=len(anchor_dataset) / len(aggregate_dataset),
                    completed_epochs_in_round=completed_epochs,
                    expected_epochs_in_round=int(config["epochs_per_round"]),
                    epochs=epoch_rows,
                    checkpoints=checkpoints,
                )

            optimization_run = dict(config)
            optimization_run["model_seed"] = (
                int(config["model_seed"]) + round_index * 1_000_003
            )
            optimization = settled._optimize(
                model=model,
                dataset=aggregate_dataset,
                run=optimization_run,
                run_dir=round_dir,
                status_callback=optimization_progress,
            )
            capacity = polar._evaluate_full55_dataset(
                model=model,
                dataset=aggregate_dataset,
                batch_size=int(config["capacity_eval_batch_size"]),
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
                    "anchor_fraction": len(anchor_dataset)
                    / len(aggregate_dataset),
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
            "schema": "zuma-rl.alphazuma-55-motor-observable-replay-completion",
            "version": 1,
            "status": "COMPLETE",
            "completed_utc": dagger._utc_now(),
            "wall_seconds": time.perf_counter() - started,
            "policy_architecture": (
                "entity_polar_intent_motor_observable_replay"
            ),
            "motor_observation_profile": MOTOR_OBSERVATION_PROFILE,
            "policy_parameter_count": sum(
                parameter.numel() for parameter in model.policy.parameters()
            ),
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "torch": torch.__version__,
                "device": str(config["device"]),
                "cuda_device": torch.cuda.get_device_name(
                    torch.device(str(config["device"]))
                ),
            },
            "source": prereg["source"],
            "migration": migration,
            "anchor_collections": anchor_collections,
            "anchor_samples": len(anchor_dataset),
            "student_rounds": student_rounds,
            "aggregate_samples": student_rounds[-1]["aggregate_samples"],
            "final_capacity_metrics_on_consumed_training_data": (
                student_rounds[-1]["capacity_metrics_on_consumed_training_data"]
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
            "online_policy_inference_semantics": (
                "exact_action_masks_plus_actor_visible_motor_state"
            ),
            "actuator_dynamics_unchanged": True,
            "final_runtime_teacher_calls_forbidden": True,
            "formal_seed_consumption": False,
            "formal_candidate_authority": False,
        }
        dagger._write_atomic(run_dir / "completion.json", completion)
        return completion
    except BaseException as error:
        failure = {
            "schema": "zuma-rl.alphazuma-55-motor-observable-replay-failure",
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
        legacy._make_env_factory = original_factory
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
        prototype = _prototype(
            original_root=original_root,
            max_ticks=int(prereg["run"]["max_ticks"]),
        )
        try:
            model = _build_motor_model(prototype, prereg["run"])
            print(
                json.dumps(
                    {
                        "status": "VALID",
                        "preregistration": {
                            "path": str(prereg_path),
                            "sha256": legacy._sha256(prereg_path),
                        },
                        "observation_shape": list(model.observation_space.shape),
                        "motor_feature_count": len(
                            prototype.motor_feature_names
                        ),
                        "policy_parameter_count": sum(
                            parameter.numel()
                            for parameter in model.policy.parameters()
                        ),
                        "formal_seed_consumption": False,
                    },
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                )
            )
        finally:
            prototype.close()
        return 0
    completion = run(
        preregistration_path=prereg_path,
        original_root=original_root,
    )
    print(
        json.dumps(
            {
                "status": completion["status"],
                "final_model": completion["final_model"],
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
