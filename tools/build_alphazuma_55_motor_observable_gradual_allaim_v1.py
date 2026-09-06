"""Freeze the all-actions aim-supervision motor DAgger contingency."""

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
CAMPAIGN_ID = "alphazuma-55-motor-observable-gradual-allaim-s99081633-v1"
BASE_PREREGISTRATION = (
    PROJECT_ROOT
    / "diagnostics/alphazuma-55-motor-observable-gradual-"
    "s99081629-preregistration-v1.json"
)
BASE_PREREGISTRATION_SHA256 = (
    "sha256:11dd10f8537dfa12deb94c41ad3be3998a4204efec93a53fa97b8a04f21d0076"
)
HARD_RESCUE_PREREGISTRATION = (
    PROJECT_ROOT
    / "diagnostics/alphazuma-55-distilled-hard-rescue-"
    "s99081622-preregistration-v1.json"
)
HARD_RESCUE_PREREGISTRATION_SHA256 = (
    "sha256:e2ed54003a254a3c4212525301803e4fbf394d95114dfd9e3b706b431b75631c"
)
TRAINER = (
    PROJECT_ROOT
    / "tools/distill_alphazuma_55_motor_observable_gradual_allaim_v1.py"
)
PREDECESSOR_TRAINER = (
    PROJECT_ROOT / "tools/distill_alphazuma_55_motor_observable_gradual_v2.py"
)
RUN_DIR = Path(
    "/mnt/d/ZumaTraining/"
    "alphazuma-55-motor-observable-gradual-allaim-s99081633-v1"
)
TRAINING_SEED_FIRST = 1_546_000_000
TRAINING_SEED_LAST = TRAINING_SEED_FIRST + 4 * 55 - 1
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "diagnostics/alphazuma-55-motor-observable-gradual-allaim-"
    "s99081633-preregistration-v1.json"
)


def _reference(path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    actual = legacy._sha256(resolved)
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError(f"frozen bytes differ: {resolved}")
    return {"path": str(resolved), "sha256": actual}


def _validate_seed_subrange(base: dict[str, Any]) -> dict[str, Any]:
    master_path = Path(str(base["master_preregistration"]["path"]))
    master = legacy._read_json(master_path.resolve(strict=True))
    training = master["seed_registry"]["training"]
    if not (
        int(training["first"])
        <= TRAINING_SEED_FIRST
        <= TRAINING_SEED_LAST
        <= int(training["last"])
    ):
        raise ValueError("all-action aim seeds escaped the training registry")

    hard = legacy._read_json(HARD_RESCUE_PREREGISTRATION.resolve(strict=True))
    hard_runs = hard["runs"]
    if not isinstance(hard_runs, list) or len(hard_runs) != 1:
        raise ValueError("hard-rescue preregistration must contain one run")
    hard_run = hard_runs[0]
    hard_last = int(hard_run["episode_seed_last"])
    if hard_last + 1 != TRAINING_SEED_FIRST:
        raise ValueError("all-action aim range is not adjacent to hard-rescue")
    if TRAINING_SEED_LAST >= int(training["last"]):
        raise ValueError("all-action aim range leaves no registry headroom")
    return {
        "master_training_range": [int(training["first"]), int(training["last"])],
        "previous_reserved_last": hard_last,
        "new_range": [TRAINING_SEED_FIRST, TRAINING_SEED_LAST],
        "strictly_after_previous_reserved_range": True,
    }


def build(*, output: Path) -> dict[str, Any]:
    base_ref = _reference(BASE_PREREGISTRATION, BASE_PREREGISTRATION_SHA256)
    hard_ref = _reference(
        HARD_RESCUE_PREREGISTRATION,
        HARD_RESCUE_PREREGISTRATION_SHA256,
    )
    trainer_ref = _reference(TRAINER)
    predecessor_trainer_ref = _reference(PREDECESSOR_TRAINER)
    value = copy.deepcopy(legacy._read_json(BASE_PREREGISTRATION))
    subrange_proof = _validate_seed_subrange(value)
    if RUN_DIR.exists():
        raise FileExistsError(f"all-action aim run already exists: {RUN_DIR}")

    value["campaign_id"] = CAMPAIGN_ID
    value["created_utc"] = datetime.now(timezone.utc).isoformat()
    value["trainer"] = trainer_ref
    value["builder"] = _reference(SCRIPT_PATH)
    value.pop("recovery_lineage", None)
    value["experimental_lineage"] = {
        "fire_only_predecessor_preregistration": base_ref,
        "fire_only_predecessor_trainer": predecessor_trainer_ref,
        "reserved_range_predecessor": hard_ref,
        "single_recipe_change": "aim_loss_scope fire_only -> all_actions",
        "environment_changed": False,
        "teacher_changed": False,
        "student_architecture_changed": False,
        "dagger_schedule_changed": False,
        "optimizer_changed": False,
        "formal_seed_authority_changed": False,
    }
    value["hypothesis"] = {
        "mechanism": (
            "every wait/fire/swap/hop action enqueues an aim target before "
            "the elite-human actuator advances its delayed aim state"
        ),
        "predecessor_mismatch": (
            "fire-only aim loss leaves most queued wait-tick aim targets "
            "unsupervised"
        ),
        "expected_effect": (
            "reduce closed-loop motor drift without teacher calls at runtime"
        ),
    }
    value["implementation"][
        "all_action_aim_supervision_adapter"
    ] = trainer_ref
    run = value["run"]
    run.update(
        {
            "id": "alphazuma55-motor-observable-gradual-allaim-v1-5080",
            "run_dir": str(RUN_DIR),
            "model_seed": 1_546_002_800,
            "training_seed_base": TRAINING_SEED_FIRST,
            "training_seed_last_consumed": TRAINING_SEED_LAST,
            "aim_loss_scope": "all_actions",
            "route_variant": "all_actions_aim_supervision_v1",
            "all_action_aim_supervision": True,
            "aim_loss_sample_semantics": (
                "raw_teacher_intent_on_every_retained_action"
            ),
        }
    )
    value["seed_registry"] = {
        "classification": "engineering_training_only",
        "range": [TRAINING_SEED_FIRST, TRAINING_SEED_LAST],
        "subrange_proof": subrange_proof,
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
