"""Continue motor-observable DAgger with a gradual teacher handoff.

This engineering-only continuation starts from the frozen V3 motor policy.
It collects four new full-55 state distributions while reducing teacher
execution slowly (1.0, 0.9, 0.7, 0.5).  Runtime inference remains a single
student policy with the exact human-input action masks and no teacher calls.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tools import distill_alphazuma_55 as legacy
from tools import distill_alphazuma_55_motor_observable_replay_v2 as motor
from tools import distill_alphazuma_55_polar_dagger_v1 as dagger
from tools import distill_alphazuma_55_polar_intent_dagger_v1 as intent_dagger
from zuma_rl.alphazuma_55 import INCLUDED_LEVELS
from zuma_rl.observable_human_speedrun import MOTOR_OBSERVATION_PROFILE


SCRIPT_PATH = Path(__file__).resolve()
EXPECTED_SCHEDULE = (1.0, 0.9, 0.7, 0.5)
EXPECTED_FEATURES_DIM = 1024
EXPECTED_EPOCHS_PER_ROUND = 12
EXPECTED_OBSERVATION_SHAPE = (22904,)
EXPECTED_SEED_BASE = 1_544_000_000


def _bound(reference: Any, label: str) -> Path:
    if not isinstance(reference, dict):
        raise ValueError(f"{label} reference must be an object")
    path = Path(str(reference.get("path", ""))).resolve(strict=True)
    if reference.get("sha256") != legacy._sha256(path):
        raise ValueError(f"{label} bytes differ: {path}")
    return path


def _validate_schedule(values: Sequence[Any]) -> tuple[float, ...]:
    schedule = tuple(float(value) for value in values)
    if schedule != EXPECTED_SCHEDULE:
        raise ValueError(
            "gradual motor DAgger schedule must remain exactly "
            f"{EXPECTED_SCHEDULE}"
        )
    return schedule


def _validate_preregistration(path: Path) -> dict[str, Any]:
    prereg = legacy._read_json(path)
    if not (
        prereg.get("schema")
        == "zuma-rl.alphazuma-55-motor-observable-gradual-preregistration"
        and prereg.get("version") == 1
        and prereg.get("status") == "FROZEN_BEFORE_TRAINING"
    ):
        raise ValueError("unexpected gradual motor preregistration")

    trainer = prereg.get("trainer", {})
    if (
        Path(str(trainer.get("path", ""))).resolve() != SCRIPT_PATH
        or trainer.get("sha256") != legacy._sha256(SCRIPT_PATH)
    ):
        raise ValueError("gradual motor trainer binding differs")
    _bound(prereg.get("builder"), "builder")
    master_path = _bound(prereg.get("master_preregistration"), "master")
    master = legacy._read_json(master_path)
    if master.get("schema") != "zuma-rl.alphazuma-55-weekend-master-preregistration":
        raise ValueError("unexpected AlphaZuma 55 master")
    for name, artifact in prereg.get("implementation", {}).items():
        _bound(artifact, f"implementation {name}")

    evidence_path = _bound(prereg.get("teacher_evidence"), "teacher evidence")
    evidence = legacy._read_json(evidence_path)
    summary = evidence.get("summary", {})
    if not (
        evidence.get("status") == "COMPLETE"
        and int(summary.get("attempts", -1)) == len(INCLUDED_LEVELS)
        and int(summary.get("wins", -1)) == len(INCLUDED_LEVELS)
        and int(summary.get("level_coverage", -1)) == len(INCLUDED_LEVELS)
        and int(summary.get("losses", -1)) == 0
        and int(summary.get("truncations", -1)) == 0
    ):
        raise ValueError("settled-V3 teacher evidence is not 55/55")

    screen_path = _bound(prereg.get("predecessor_screen_audit"), "screen audit")
    screen = legacy._read_json(screen_path)
    if not (
        screen.get("status") == "PASS"
        and int(screen.get("attempts", -1)) == 156
        and sum(int(row.get("wins", 0)) for row in screen.get("model_summaries", []))
        == 9
    ):
        raise ValueError("predecessor screen evidence changed")

    source = prereg.get("source", {})
    completion_path = _bound(source.get("completion"), "source completion")
    completion = legacy._read_json(completion_path)
    model_path = _bound(source.get("model"), "source model")
    final = completion.get("final_model", {})
    if not (
        completion.get("schema")
        == "zuma-rl.alphazuma-55-motor-observable-replay-completion"
        and completion.get("status") == "COMPLETE"
        and completion.get("route_version") == "v3_retry_audited"
        and completion.get("motor_observation_profile")
        == MOTOR_OBSERVATION_PROFILE
        and completion.get("final_runtime_teacher_calls_forbidden") is True
        and completion.get("formal_seed_consumption") is False
        and completion.get("formal_candidate_authority") is False
        and Path(str(final.get("path", ""))).resolve() == model_path
        and final.get("sha256") == source["model"]["sha256"]
    ):
        raise ValueError("source V3 motor completion is not eligible")
    if tuple(prereg.get("levels", ())) != tuple(INCLUDED_LEVELS):
        raise ValueError("gradual motor level order changed")

    run = prereg.get("run", {})
    schedule = _validate_schedule(run.get("teacher_execution_probabilities", ()))
    expected_fractions = {"wait": 0.45, "fire": 0.35, "swap": 0.15, "hop": 0.05}
    expected_capacities = {"wait": 256, "fire": 256, "swap": 192, "hop": 192}
    expected_strides = {"wait": 4, "fire": 1, "swap": 1, "hop": 1}
    if not (
        run.get("policy_architecture")
        == "entity_polar_intent_motor_observable_gradual"
        and run.get("motor_observation_profile") == MOTOR_OBSERVATION_PROFILE
        and int(run.get("learned_features_dim", -1)) == EXPECTED_FEATURES_DIM
        and int(run.get("rounds", -1)) == len(schedule)
        and int(run.get("epochs_per_round", -1)) == EXPECTED_EPOCHS_PER_ROUND
        and int(run.get("checkpoint_interval_epochs", -1)) == 3
        and int(run.get("parallel_envs", -1)) == 24
        and int(run.get("max_ticks", -1)) == 30000
        and int(run.get("batch_size", -1)) == 512
        and int(run.get("capacity_eval_batch_size", -1)) == 128
        and float(run.get("learning_rate", -1.0)) == 1.0e-5
        and float(run.get("verb_loss_weight", -1.0)) == 2.0
        and float(run.get("max_grad_norm", -1.0)) == 0.5
        and run.get("aim_loss_scope") == "fire_only"
        and run.get("aim_loss_mode") == "circular_smoothed"
        and float(run.get("aim_smoothing_sigma", -1.0)) == 1.5
        and float(run.get("aim_distance_weight", -1.0)) == 0.5
        and run.get("reservoir_capacity_by_verb") == expected_capacities
        and run.get("sample_stride_by_verb") == expected_strides
        and run.get("target_batch_fraction_by_verb") == expected_fractions
        and run.get("source_is_frozen_motor_v3_final_model") is True
        and run.get("aggregate_dataset_across_rounds") is True
        and run.get("optimize_after_each_round") is True
        and run.get("raw_teacher_intent_labels") is True
        and run.get("training_mask_relaxation")
        == "unmask_only_the_teacher_target_verb"
        and run.get("exact_action_masks_for_environment_execution") is True
        and run.get("exact_action_masks_for_online_policy_inference") is True
        and run.get("label_aim_must_remain_exact_mask_legal") is True
        and run.get("final_runtime_teacher_calls_forbidden") is True
    ):
        raise ValueError("gradual motor semantic contract changed")

    first = int(run.get("training_seed_base", -1))
    last = int(run.get("training_seed_last_consumed", -1))
    if not (
        first == EXPECTED_SEED_BASE
        and last == first + len(schedule) * len(INCLUDED_LEVELS) - 1
    ):
        raise ValueError("gradual motor seed interval changed")
    registry = master["seed_registry"]["training"]
    if not int(registry["first"]) <= first <= last <= int(registry["last"]):
        raise ValueError("gradual motor seeds escaped training registry")

    boundary = prereg.get("authority_boundary", {})
    for name in (
        "current_campaign_candidate_authority",
        "formal_selection_seed_consumption",
        "formal_final_blind_seed_consumption",
        "continuous_campaign_seed_consumption",
        "power_restore_authority",
    ):
        if boundary.get(name) is not False:
            raise ValueError(f"authority boundary changed: {name}")
    return prereg


def run(*, preregistration_path: Path, original_root: Path) -> dict[str, Any]:
    preregistration_path = preregistration_path.resolve(strict=True)
    prereg = _validate_preregistration(preregistration_path)
    run_dir = Path(str(prereg["run"]["run_dir"])).resolve()

    original_validate = dagger._validate_preregistration
    original_collector = dagger._collect_dagger_round
    original_schedule = dagger._validate_probability_schedule
    original_script_path = dagger.SCRIPT_PATH
    original_shape = dagger.EXPECTED_OBSERVATION_SHAPE
    original_prototype = dagger._prototype_environment
    original_factory = legacy._make_env_factory
    try:
        dagger._validate_preregistration = lambda _: prereg
        dagger._collect_dagger_round = intent_dagger._collect_intent_dagger_round
        dagger._validate_probability_schedule = _validate_schedule
        dagger.SCRIPT_PATH = SCRIPT_PATH
        dagger.EXPECTED_OBSERVATION_SHAPE = EXPECTED_OBSERVATION_SHAPE
        dagger._prototype_environment = motor._prototype
        legacy._make_env_factory = motor._make_motor_env_factory
        completion = dagger.run(
            preregistration_path=preregistration_path,
            original_root=original_root,
        )
    except BaseException:
        failure_path = run_dir / "failure.json"
        if failure_path.exists():
            failure = legacy._read_json(failure_path)
            failure.update(
                {
                    "schema": "zuma-rl.alphazuma-55-motor-observable-gradual-failure",
                    "motor_observation_profile": MOTOR_OBSERVATION_PROFILE,
                    "formal_seed_consumption": False,
                    "formal_candidate_authority": False,
                }
            )
            dagger._write_atomic(failure_path, failure)
        raise
    finally:
        dagger._validate_preregistration = original_validate
        dagger._collect_dagger_round = original_collector
        dagger._validate_probability_schedule = original_schedule
        dagger.SCRIPT_PATH = original_script_path
        dagger.EXPECTED_OBSERVATION_SHAPE = original_shape
        dagger._prototype_environment = original_prototype
        legacy._make_env_factory = original_factory

    completion.update(
        {
            "schema": "zuma-rl.alphazuma-55-motor-observable-gradual-completion",
            "policy_architecture": "entity_polar_intent_motor_observable_gradual",
            "motor_observation_profile": MOTOR_OBSERVATION_PROFILE,
            "learned_features_dim": EXPECTED_FEATURES_DIM,
            "training_label_semantics": "raw_teacher_intent",
            "state_distribution_semantics": "gradual_teacher_student_dagger",
            "teacher_execution_probabilities": list(EXPECTED_SCHEDULE),
            "environment_execution_semantics": (
                "exact_mask_teacher_effective_or_exact_mask_student"
            ),
            "online_policy_inference_semantics": (
                "exact_action_masks_plus_actor_visible_motor_state"
            ),
            "actuator_dynamics_unchanged": True,
            "final_runtime_teacher_calls_forbidden": True,
            "formal_seed_consumption": False,
            "formal_candidate_authority": False,
        }
    )
    dagger._write_atomic(run_dir / "completion.json", completion)
    return completion


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
        from sb3_contrib import MaskablePPO

        prototype = motor._prototype(
            original_root=original_root,
            max_ticks=int(prereg["run"]["max_ticks"]),
        )
        try:
            model = MaskablePPO.load(
                Path(prereg["source"]["model"]["path"]),
                env=prototype,
                device=str(prereg["run"]["device"]),
            )
            learned = int(model.policy.features_extractor.learned_features_dim)
            if learned != EXPECTED_FEATURES_DIM:
                raise ValueError("gradual motor source feature width changed")
            if tuple(int(value) for value in model.observation_space.shape) != (
                EXPECTED_OBSERVATION_SHAPE
            ):
                raise ValueError("gradual motor observation shape changed")
            print(
                json.dumps(
                    {
                        "status": "VALID",
                        "preregistration": {
                            "path": str(prereg_path),
                            "sha256": legacy._sha256(prereg_path),
                        },
                        "observation_shape": list(model.observation_space.shape),
                        "teacher_execution_probabilities": list(EXPECTED_SCHEDULE),
                        "epochs_per_round": EXPECTED_EPOCHS_PER_ROUND,
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
