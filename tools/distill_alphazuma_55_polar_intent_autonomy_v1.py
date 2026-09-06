"""Continue raw-intent DAgger on four frozen student-only state rounds."""

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
from tools import distill_alphazuma_55_polar_dagger_v1 as dagger
from tools import distill_alphazuma_55_polar_intent_dagger_v1 as intent_dagger
from tools import distill_alphazuma_55_polar_intent_wide_v1 as intent
from zuma_rl.alphazuma_55 import INCLUDED_LEVELS


SCRIPT_PATH = Path(__file__).resolve()
EXPECTED_SCHEDULE = (0.0, 0.0, 0.0, 0.0)
EXPECTED_FEATURES_DIM = 1024
EXPECTED_EPOCHS_PER_ROUND = 18


def _validate_probability_schedule(values: Sequence[Any]) -> tuple[float, ...]:
    schedule = tuple(float(value) for value in values)
    if schedule != EXPECTED_SCHEDULE:
        raise ValueError(
            "autonomy continuation schedule must remain exactly "
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
        == "zuma-rl.alphazuma-55-polar-intent-autonomy-preregistration"
        and prereg.get("version") == 1
        and prereg.get("status") == "FROZEN_BEFORE_TRAINING"
    ):
        raise ValueError("unexpected autonomy continuation preregistration")
    trainer = prereg.get("trainer", {})
    if (
        Path(str(trainer.get("path", ""))).resolve() != SCRIPT_PATH
        or trainer.get("sha256") != legacy._sha256(SCRIPT_PATH)
    ):
        raise ValueError("autonomy continuation trainer binding differs")
    master_path = _bound(prereg.get("master_preregistration"), "master")
    master = dagger._read(master_path)
    if (
        master.get("schema")
        != "zuma-rl.alphazuma-55-weekend-master-preregistration"
    ):
        raise ValueError("unexpected autonomy continuation master")
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
        raise ValueError("source DAgger completion bytes differ")
    completion = dagger._read(completion_path)
    if not (
        completion.get("schema")
        == "zuma-rl.alphazuma-55-polar-intent-dagger-completion"
        and completion.get("status") == "COMPLETE"
        and completion.get("policy_architecture")
        == "entity_polar_intent_wide_dagger"
        and completion.get("training_label_semantics")
        == "raw_teacher_intent"
        and completion.get("online_policy_inference_semantics")
        == "exact_action_masks"
        and completion.get("formal_seed_consumption") is False
        and completion.get("formal_candidate_authority") is False
    ):
        raise ValueError("source DAgger completion is not eligible")
    source_model = Path(str(source.get("model_path", ""))).resolve(strict=True)
    if source.get("model_sha256") != legacy._sha256(source_model):
        raise ValueError("source DAgger model bytes differ")
    final = completion.get("final_model", {})
    if (
        Path(str(final.get("path", ""))).resolve() != source_model
        or final.get("sha256") != source.get("model_sha256")
    ):
        raise ValueError("source is not the frozen DAgger final model")
    if tuple(prereg.get("levels", ())) != tuple(INCLUDED_LEVELS):
        raise ValueError("autonomy continuation level order changed")

    run = prereg.get("run", {})
    schedule = _validate_probability_schedule(
        run.get("teacher_execution_probabilities", ())
    )
    if not (
        run.get("policy_architecture")
        == "entity_polar_intent_wide_dagger_autonomy"
        and int(run.get("learned_features_dim", -1))
        == EXPECTED_FEATURES_DIM
        and int(run.get("rounds", -1)) == len(schedule)
        and int(run.get("epochs_per_round", -1))
        == EXPECTED_EPOCHS_PER_ROUND
        and run.get("source_is_frozen_raw_intent_dagger_final_model") is True
        and run.get("student_state_collection") is True
        and run.get("raw_teacher_intent_labels") is True
        and run.get("training_mask_relaxation")
        == "unmask_only_the_teacher_target_verb"
        and run.get("exact_action_masks_for_environment_execution") is True
        and run.get("exact_action_masks_for_online_policy_inference") is True
        and run.get("label_aim_must_remain_exact_mask_legal") is True
        and run.get("aggregate_dataset_across_rounds") is True
        and run.get("optimize_after_each_round") is True
        and run.get("aim_loss_scope") == "fire_only"
        and run.get("aim_loss_mode") == "categorical"
        and run.get("continuation_changes_only_state_coverage") is True
    ):
        raise ValueError("autonomy continuation semantic contract changed")
    first = int(run["training_seed_base"])
    last = int(run["training_seed_last_consumed"])
    if last != first + len(INCLUDED_LEVELS) * len(schedule) - 1:
        raise ValueError("autonomy continuation seed interval changed")
    registry = master["seed_registry"]["training"]
    if not int(registry["first"]) <= first <= last <= int(registry["last"]):
        raise ValueError("autonomy continuation seeds escaped registry")
    boundary = prereg.get("authority_boundary", {})
    for name in (
        "current_campaign_candidate_authority",
        "predecessor_successor_candidate_authority",
        "formal_selection_seed_consumption",
        "formal_final_blind_seed_consumption",
        "continuous_campaign_seed_consumption",
    ):
        if boundary.get(name) is not False:
            raise ValueError(f"autonomy continuation authority changed: {name}")
    return prereg


def run(*, preregistration_path: Path, original_root: Path) -> dict[str, Any]:
    preregistration_path = preregistration_path.resolve(strict=True)
    prereg = _validate_preregistration(preregistration_path)
    original_validate = dagger._validate_preregistration
    original_collector = dagger._collect_dagger_round
    original_schedule = dagger._validate_probability_schedule
    original_script_path = dagger.SCRIPT_PATH
    try:
        dagger._validate_preregistration = lambda _: prereg
        dagger._collect_dagger_round = (
            intent_dagger._collect_intent_dagger_round
        )
        dagger._validate_probability_schedule = _validate_probability_schedule
        dagger.SCRIPT_PATH = SCRIPT_PATH
        completion = dagger.run(
            preregistration_path=preregistration_path,
            original_root=original_root,
        )
    except BaseException:
        run_dir = Path(str(prereg["run"]["run_dir"])).resolve()
        failure_path = run_dir / "failure.json"
        if failure_path.exists():
            failure = dagger._read(failure_path)
            failure["schema"] = (
                "zuma-rl.alphazuma-55-polar-intent-autonomy-failure"
            )
            failure["training_label_semantics"] = "raw_teacher_intent"
            failure["state_distribution_semantics"] = "student_only_dagger"
            dagger._write_atomic(failure_path, failure)
        raise
    finally:
        dagger._validate_preregistration = original_validate
        dagger._collect_dagger_round = original_collector
        dagger._validate_probability_schedule = original_schedule
        dagger.SCRIPT_PATH = original_script_path
    completion.update(
        {
            "schema": "zuma-rl.alphazuma-55-polar-intent-autonomy-completion",
            "policy_architecture": (
                "entity_polar_intent_wide_dagger_autonomy"
            ),
            "learned_features_dim": EXPECTED_FEATURES_DIM,
            "training_label_semantics": "raw_teacher_intent",
            "training_mask_relaxation": "unmask_only_the_teacher_target_verb",
            "state_distribution_semantics": "student_only_dagger",
            "teacher_execution_probabilities": list(EXPECTED_SCHEDULE),
            "environment_execution_semantics": "exact_mask_student_only",
            "online_policy_inference_semantics": "exact_action_masks",
        }
    )
    completion_path = Path(str(prereg["run"]["run_dir"])) / "completion.json"
    dagger._write_atomic(completion_path, completion)
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

        prototype = dagger._prototype_environment(
            original_root=original_root,
            max_ticks=int(prereg["run"]["max_ticks"]),
        )
        try:
            model = MaskablePPO.load(
                Path(prereg["source"]["model_path"]),
                env=prototype,
                device=str(prereg["run"]["device"]),
            )
            learned = int(model.policy.features_extractor.learned_features_dim)
            if learned != EXPECTED_FEATURES_DIM:
                raise ValueError("autonomy source feature width changed")
            print(
                json.dumps(
                    {
                        "status": "VALID",
                        "preregistration": {
                            "path": str(prereg_path),
                            "sha256": legacy._sha256(prereg_path),
                        },
                        "policy_architecture": (
                            "entity_polar_intent_wide_dagger_autonomy"
                        ),
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
