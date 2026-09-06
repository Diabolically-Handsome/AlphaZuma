"""Freeze the action-head-only gradual motor DAgger ablation."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tools import distill_alphazuma_55 as legacy
from tools import distill_alphazuma_55_polar_dagger_v1 as dagger


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
CAMPAIGN_ID = "alphazuma-55-motor-observable-gradual-headonly-s99081642-v1"
BASE_PREREGISTRATION = (
    PROJECT_ROOT
    / "diagnostics/alphazuma-55-motor-observable-gradual-"
    "s99081629-preregistration-v1.json"
)
BASE_PREREGISTRATION_SHA256 = (
    "sha256:11dd10f8537dfa12deb94c41ad3be3998a4204efec93a53fa97b8a04f21d0076"
)
ALLAIM_PREREGISTRATION = (
    PROJECT_ROOT
    / "diagnostics/alphazuma-55-motor-observable-gradual-allaim-"
    "s99081633-preregistration-v1.json"
)
ALLAIM_PREREGISTRATION_SHA256 = (
    "sha256:502c0ab5566a66fa33ff33df80b10b101c5ea1b75dc4e21ff07f5996965730e1"
)
SEED_OVERLAP_INCIDENT = (
    PROJECT_ROOT
    / "diagnostics/alphazuma-55-training-seed-overlap-"
    "1546000000-incident-v1.json"
)
SEED_OVERLAP_INCIDENT_SHA256 = (
    "sha256:e957e188226f1b2b6f402e68c1da40a18d2c6c85c505973db885d7759bbc2bb5"
)
PREDECESSOR_GATE_AUDIT = (
    PROJECT_ROOT
    / "diagnostics/alphazuma-55-motor-observable-gradual-postprocess-"
    "s99081630-timeout-recovery-audit-v1.json"
)
PREDECESSOR_GATE_AUDIT_SHA256 = (
    "sha256:e2f73963847962ca7de39e1bcf3d7ad6eed604145caf57e78d77d32223818ff6"
)
TRAINER = (
    PROJECT_ROOT
    / "tools/distill_alphazuma_55_motor_observable_gradual_headonly_v1.py"
)
PREDECESSOR_TRAINER = (
    PROJECT_ROOT / "tools/distill_alphazuma_55_motor_observable_gradual_v2.py"
)
RUN_DIR = Path(
    "/mnt/d/ZumaTraining/"
    "alphazuma-55-motor-observable-gradual-headonly-s99081642-v1"
)
TRAINING_SEED_FIRST = 1_546_000_000
TRAINING_SEED_LAST = TRAINING_SEED_FIRST + 4 * 55 - 1
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "diagnostics/alphazuma-55-motor-observable-gradual-headonly-"
    "s99081642-preregistration-v1.json"
)


def _reference(path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    actual = legacy._sha256(resolved)
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError(f"frozen bytes differ: {resolved}")
    return {"path": str(resolved), "sha256": actual}


def _validate_reused_training_range(base: dict[str, Any]) -> None:
    master_path = Path(str(base["master_preregistration"]["path"]))
    master = legacy._read_json(master_path.resolve(strict=True))
    training = master["seed_registry"]["training"]
    if not (
        int(training["first"])
        <= TRAINING_SEED_FIRST
        <= TRAINING_SEED_LAST
        <= int(training["last"])
    ):
        raise ValueError("head-only seeds escaped the training registry")
    incident = legacy._read_json(SEED_OVERLAP_INCIDENT.resolve(strict=True))
    if not (
        incident.get("status") == "CONFIRMED_PRE_INFERENCE"
        and incident.get("classification", {}).get("scope")
        == "engineering_training_seed_reuse"
        and incident.get("classification", {}).get(
            "formal_seed_consumption_detected"
        )
        is False
    ):
        raise ValueError("training-seed reuse incident contract changed")


def build(*, output: Path) -> dict[str, Any]:
    base_ref = _reference(BASE_PREREGISTRATION, BASE_PREREGISTRATION_SHA256)
    allaim_ref = _reference(
        ALLAIM_PREREGISTRATION, ALLAIM_PREREGISTRATION_SHA256
    )
    incident_ref = _reference(
        SEED_OVERLAP_INCIDENT, SEED_OVERLAP_INCIDENT_SHA256
    )
    gate_ref = _reference(
        PREDECESSOR_GATE_AUDIT, PREDECESSOR_GATE_AUDIT_SHA256
    )
    trainer_ref = _reference(TRAINER)
    predecessor_trainer_ref = _reference(PREDECESSOR_TRAINER)
    value = copy.deepcopy(legacy._read_json(BASE_PREREGISTRATION))
    _validate_reused_training_range(value)
    if RUN_DIR.exists():
        raise FileExistsError(f"head-only run already exists: {RUN_DIR}")

    value["campaign_id"] = CAMPAIGN_ID
    value["created_utc"] = datetime.now(timezone.utc).isoformat()
    value["trainer"] = trainer_ref
    value["builder"] = _reference(SCRIPT_PATH)
    value.pop("recovery_lineage", None)
    value["predecessor_gate_audit"] = gate_ref
    value["experimental_lineage"] = {
        "fire_only_predecessor_preregistration": base_ref,
        "fire_only_predecessor_trainer": predecessor_trainer_ref,
        "parallel_all_actions_route": allaim_ref,
        "training_seed_overlap_incident": incident_ref,
        "single_substantive_recipe_change": (
            "optimize only action_net.weight and action_net.bias instead of "
            "the complete shared policy"
        ),
        "optimizer_rebuild_is_required_by_parameter_scope": True,
        "optimizer_class_changed": False,
        "optimizer_state_reset_before_first_update": True,
        "environment_changed": False,
        "teacher_changed": False,
        "labels_changed": False,
        "student_architecture_changed": False,
        "dagger_schedule_changed": False,
        "loss_function_changed": False,
        "formal_seed_authority_changed": False,
    }
    value["hypothesis"] = {
        "observed": (
            "In the independently recomputed fire-only Gate, the frozen "
            "source cleared 5 of 55 levels while every trained checkpoint "
            "cleared at most 2 of 55 and later rounds reached 0 of 55."
        ),
        "mechanism": (
            "full-policy imitation gradients overwrite source representations "
            "that remain useful for closed-loop play"
        ),
        "intervention": (
            "retain the frozen source representation and train only the "
            "factorized 184-logit action head under the original fire-only "
            "gradual DAgger recipe"
        ),
        "falsifier": (
            "head-only checkpoints fail to improve independently seeded "
            "closed-loop screen and full55 Gate coverage"
        ),
    }
    value["implementation"]["head_only_optimizer_adapter"] = trainer_ref
    run = value["run"]
    run.update(
        {
            "id": "alphazuma55-motor-observable-gradual-headonly-v1-5090",
            "run_dir": str(RUN_DIR),
            "device": "cuda:0",
            "model_seed": 1_546_004_200,
            "training_seed_base": TRAINING_SEED_FIRST,
            "training_seed_last_consumed": TRAINING_SEED_LAST,
            "aim_loss_scope": "fire_only",
            "route_variant": "action_head_only_fire_aim_v1",
            "optimizer_parameter_scope": "action_net_only",
            "trainable_parameter_names": [
                "action_net.weight",
                "action_net.bias"
            ],
            "frozen_backbone": True,
            "optimizer_state_reset_before_first_update": True,
            "frozen_parameter_invariant_checked_after_each_round": True,
        }
    )
    value["seed_registry"] = {
        "classification": (
            "engineering_training_seed_reuse_for_registered_ablation"
        ),
        "range": [TRAINING_SEED_FIRST, TRAINING_SEED_LAST],
        "fresh_training_randomness_claim": False,
        "reuse_disclosure": incident_ref,
        "formal_selection_seed_consumption": "NONE",
        "formal_final_blind_seed_consumption": "NONE",
        "continuous_campaign_seed_consumption": "NONE",
    }
    value["outputs"] = {
        "run_dir": str(RUN_DIR),
        "completion": str(RUN_DIR / "completion.json"),
        "failure": str(RUN_DIR / "failure.json"),
    }

    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"preregistration already exists: {output}")
    dagger._write_atomic(output, value)
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    value = build(output=args.output)
    print(
        f"{value['status']} {value['campaign_id']} "
        f"{legacy._sha256(args.output.resolve())}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
