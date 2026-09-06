"""Freeze the compatibility-only successor to gradual motor DAgger V1."""

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
CAMPAIGN_ID = "alphazuma-55-motor-observable-gradual-s99081629-v1"
BASE_PREREGISTRATION = (
    PROJECT_ROOT
    / "diagnostics/alphazuma-55-motor-observable-gradual-"
    "s99081628-preregistration-v1.json"
)
BASE_PREREGISTRATION_SHA256 = (
    "sha256:e9747af431410abc5c47d048e87f6bf874cd6bf6023f04b94adbd628ca220ea0"
)
RUNTIME_INCIDENT = (
    PROJECT_ROOT
    / "diagnostics/alphazuma-55-motor-observable-gradual-"
    "s99081628-runtime-incident-v1.json"
)
RUNTIME_INCIDENT_SHA256 = (
    "sha256:9371f830fc36baf2d21c95a2e56c4eb9d9094a5da227acc9f36b12265e961b8e"
)
TRAINER = (
    PROJECT_ROOT / "tools/distill_alphazuma_55_motor_observable_gradual_v2.py"
)
TRAINER_SHA256 = (
    "sha256:7a7da4c374c72c3dafdb96b184a4a57e43c1dd08d224a81a72e04bde193577d2"
)
PREDECESSOR_TRAINER = (
    PROJECT_ROOT / "tools/distill_alphazuma_55_motor_observable_gradual_v1.py"
)
PREDECESSOR_TRAINER_SHA256 = (
    "sha256:b01572103c9ad4441a534b5970674c799c0f1f54cf75506aa4dbb3e58c34f64a"
)
RUN_DIR = Path(
    "/mnt/d/ZumaTraining/alphazuma-55-motor-observable-gradual-s99081629-v1"
)
FAILED_RUN_DIR = Path(
    "/mnt/d/ZumaTraining/alphazuma-55-motor-observable-gradual-s99081628-v1"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "diagnostics/alphazuma-55-motor-observable-gradual-"
    "s99081629-preregistration-v1.json"
)


def _reference(path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    actual = legacy._sha256(resolved)
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError(f"frozen bytes differ: {resolved}")
    return {"path": str(resolved), "sha256": actual}


def _verify_no_seed_consumption() -> None:
    incident = legacy._read_json(RUNTIME_INCIDENT.resolve(strict=True))
    if not (
        incident.get("status") == "FAILED_BEFORE_MODEL_LOAD_NO_SEEDS_CONSUMED"
        and incident.get("seed_boundary", {}).get(
            "engineering_training_seed_consumption"
        )
        == "NONE"
        and incident.get("evidence", {}).get("model_load_started") is False
        and incident.get("evidence", {}).get("environment_collection_started")
        is False
    ):
        raise ValueError("V1 incident does not authorize seed reuse")
    if (FAILED_RUN_DIR / "training_status.json").exists():
        raise ValueError("failed V1 unexpectedly has a training status")
    if list(FAILED_RUN_DIR.glob("checkpoint*.zip")):
        raise ValueError("failed V1 unexpectedly has a checkpoint")
    failure = legacy._read_json((FAILED_RUN_DIR / "failure.json").resolve(strict=True))
    if not (
        failure.get("status") == "FAILED"
        and failure.get("error_type") == "KeyError"
        and failure.get("error") == "'model_path'"
        and failure.get("formal_seed_consumption") is False
    ):
        raise ValueError("failed V1 evidence changed")


def build(*, output: Path) -> dict[str, Any]:
    base_ref = _reference(BASE_PREREGISTRATION, BASE_PREREGISTRATION_SHA256)
    incident_ref = _reference(RUNTIME_INCIDENT, RUNTIME_INCIDENT_SHA256)
    trainer_ref = _reference(TRAINER, TRAINER_SHA256)
    predecessor_trainer_ref = _reference(
        PREDECESSOR_TRAINER, PREDECESSOR_TRAINER_SHA256
    )
    _verify_no_seed_consumption()
    if RUN_DIR.exists():
        raise FileExistsError(f"successor run already exists: {RUN_DIR}")

    value = copy.deepcopy(legacy._read_json(BASE_PREREGISTRATION))
    value["campaign_id"] = CAMPAIGN_ID
    value["created_utc"] = datetime.now(timezone.utc).isoformat()
    value["trainer"] = trainer_ref
    value["builder"] = _reference(SCRIPT_PATH)
    value["recovery_lineage"] = {
        "predecessor_preregistration": base_ref,
        "predecessor_runtime_incident": incident_ref,
        "classification": "compatibility_only_successor_before_any_seed_use",
        "allowed_change": "add source.model_path alias in memory after validation",
        "training_recipe_changed": False,
        "seed_interval_changed": False,
    }
    value["implementation"][
        "compatibility_adapter_predecessor"
    ] = predecessor_trainer_ref
    value["run"]["id"] = "alphazuma55-motor-observable-gradual-v2-5080"
    value["run"]["run_dir"] = str(RUN_DIR)
    value["seed_registry"]["recovery_reuse"] = {
        "range": list(value["seed_registry"]["range"]),
        "authorization": incident_ref,
        "reason": "V1 failed before model load and environment collection",
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
