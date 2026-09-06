"""Freeze the head-only plus all-actions-aim factorial DAgger route."""

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
CAMPAIGN_ID = (
    "alphazuma-55-motor-observable-gradual-headonly-allaim-s99081646-v1"
)
ALLAIM_PREREGISTRATION = PROJECT_ROOT / (
    "diagnostics/alphazuma-55-motor-observable-gradual-allaim-"
    "s99081633-preregistration-v1.json"
)
ALLAIM_PREREGISTRATION_SHA256 = (
    "sha256:502c0ab5566a66fa33ff33df80b10b101c5ea1b75dc4e21ff07f5996965730e1"
)
HEADONLY_PREREGISTRATION = PROJECT_ROOT / (
    "diagnostics/alphazuma-55-motor-observable-gradual-headonly-"
    "s99081642-preregistration-v1.json"
)
HEADONLY_PREREGISTRATION_SHA256 = (
    "sha256:18b12280221b59588a65349812303ab58c115e859fdfb8e2eb674d222cd2ea72"
)
HEADONLY_PREREG_AUDIT = PROJECT_ROOT / (
    "diagnostics/alphazuma-55-motor-observable-gradual-headonly-"
    "s99081642-independent-prereg-audit-v1.json"
)
HEADONLY_PREREG_AUDIT_SHA256 = (
    "sha256:9efd8a02d81df0cce8cd9f339d3f16faf1012b8ccca358194bdeafef5bd5be09"
)
TRAINER = PROJECT_ROOT / (
    "tools/distill_alphazuma_55_motor_observable_gradual_"
    "headonly_allaim_v1.py"
)
RUN_DIR = Path(
    "/mnt/d/ZumaTraining/"
    "alphazuma-55-motor-observable-gradual-headonly-allaim-s99081646-v1"
)
TRAINING_SEED_FIRST = 1_546_001_000
TRAINING_SEED_LAST = TRAINING_SEED_FIRST + 4 * 55 - 1
DEFAULT_OUTPUT = PROJECT_ROOT / (
    "diagnostics/alphazuma-55-motor-observable-gradual-headonly-allaim-"
    "s99081646-preregistration-v1.json"
)


def _reference(path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    actual = legacy._sha256(resolved)
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError(f"frozen bytes differ: {resolved}")
    return {"path": str(resolved), "sha256": actual}


def _validate_fresh_training_range(base: dict[str, Any]) -> None:
    master_path = Path(str(base["master_preregistration"]["path"]))
    master = legacy._read_json(master_path.resolve(strict=True))
    training = master["seed_registry"]["training"]
    if not (
        int(training["first"])
        <= TRAINING_SEED_FIRST
        <= TRAINING_SEED_LAST
        <= int(training["last"])
        and TRAINING_SEED_FIRST > 1_546_000_219
    ):
        raise ValueError("factorial seeds are not a fresh registered subrange")


def build(*, output: Path) -> dict[str, Any]:
    allaim_ref = _reference(
        ALLAIM_PREREGISTRATION, ALLAIM_PREREGISTRATION_SHA256
    )
    headonly_ref = _reference(
        HEADONLY_PREREGISTRATION, HEADONLY_PREREGISTRATION_SHA256
    )
    headonly_audit_ref = _reference(
        HEADONLY_PREREG_AUDIT, HEADONLY_PREREG_AUDIT_SHA256
    )
    trainer_ref = _reference(TRAINER)
    value = copy.deepcopy(legacy._read_json(ALLAIM_PREREGISTRATION))
    _validate_fresh_training_range(value)
    if RUN_DIR.exists():
        raise FileExistsError(f"factorial run already exists: {RUN_DIR}")

    value["campaign_id"] = CAMPAIGN_ID
    value["created_utc"] = datetime.now(timezone.utc).isoformat()
    value["trainer"] = trainer_ref
    value["builder"] = _reference(SCRIPT_PATH)
    value["experimental_lineage"] = {
        "all_actions_parent": allaim_ref,
        "head_only_parent": headonly_ref,
        "head_only_parent_independent_prereg_audit": headonly_audit_ref,
        "factorial_cell": {
            "frozen_backbone": True,
            "all_action_aim_supervision": True,
        },
        "relative_to_all_actions_parent": (
            "train action_net.weight and action_net.bias only"
        ),
        "relative_to_head_only_parent": (
            "supervise aim on every retained raw teacher intent"
        ),
        "environment_changed": False,
        "teacher_changed": False,
        "student_architecture_changed": False,
        "dagger_schedule_changed": False,
        "formal_seed_authority_changed": False,
    }
    value["hypothesis"] = {
        "mechanism": (
            "all-action aim supervision aligns the delayed actuator while "
            "head-only optimization prevents representation erasure"
        ),
        "falsifier": (
            "independently seeded screen and full55 Gate fail to improve "
            "closed-loop coverage"
        ),
    }
    value["implementation"]["headonly_allaim_factorial_adapter"] = trainer_ref
    run = value["run"]
    run.update(
        {
            "id": "alphazuma55-headonly-allaim-v1-5080",
            "run_dir": str(RUN_DIR),
            "device": "cuda:1",
            "model_seed": 1_546_006_200,
            "training_seed_base": TRAINING_SEED_FIRST,
            "training_seed_last_consumed": TRAINING_SEED_LAST,
            "aim_loss_scope": "all_actions",
            "route_variant": "action_head_only_all_actions_aim_v1",
            "all_action_aim_supervision": True,
            "aim_loss_sample_semantics": (
                "raw_teacher_intent_on_every_retained_action"
            ),
            "optimizer_parameter_scope": "action_net_only",
            "trainable_parameter_names": [
                "action_net.weight",
                "action_net.bias",
            ],
            "frozen_backbone": True,
            "optimizer_state_reset_before_first_update": True,
            "frozen_parameter_invariant_checked_after_each_round": True,
        }
    )
    value["seed_registry"] = {
        "classification": "fresh_engineering_training_subrange",
        "range": [TRAINING_SEED_FIRST, TRAINING_SEED_LAST],
        "subrange_proof": {
            "master_training_range": [1_500_000_000, 1_549_999_999],
            "previous_current_route_last": 1_546_000_219,
            "new_range": [TRAINING_SEED_FIRST, TRAINING_SEED_LAST],
            "strictly_after_current_routes": True,
        },
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
