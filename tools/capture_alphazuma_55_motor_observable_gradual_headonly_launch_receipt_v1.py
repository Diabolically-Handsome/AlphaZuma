"""Capture the immutable launch receipt for the head-only DAgger ablation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools import (
    capture_alphazuma_55_motor_observable_gradual_allaim_launch_receipt_v1
    as common,
)


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
CAMPAIGN_ID = "alphazuma-55-motor-observable-gradual-headonly-s99081642-v1"
PREREGISTRATION = (
    PROJECT_ROOT
    / "diagnostics/alphazuma-55-motor-observable-gradual-headonly-"
    "s99081642-preregistration-v1.json"
)
PREREGISTRATION_SHA256 = (
    "sha256:18b12280221b59588a65349812303ab58c115e859fdfb8e2eb674d222cd2ea72"
)
PREREGISTRATION_AUDIT = (
    PROJECT_ROOT
    / "diagnostics/alphazuma-55-motor-observable-gradual-headonly-"
    "s99081642-independent-prereg-audit-v1.json"
)
PREREGISTRATION_AUDIT_SHA256 = (
    "sha256:9efd8a02d81df0cce8cd9f339d3f16faf1012b8ccca358194bdeafef5bd5be09"
)
TRAINER = (
    PROJECT_ROOT
    / "tools/distill_alphazuma_55_motor_observable_gradual_headonly_v1.py"
)
TRAINER_SHA256 = (
    "sha256:ceb6e87d388f9ab6d03ba2c1f0194da4c7dd2d62344d70bc6a6f1c1ab30744ba"
)
PREREGISTRATION_BUILDER = (
    PROJECT_ROOT
    / "tools/build_alphazuma_55_motor_observable_gradual_headonly_v1.py"
)
PREREGISTRATION_BUILDER_SHA256 = (
    "sha256:728d38dd8ec9dda5f42778b9e4bcef00e75f3aef15937e124badd2f198521c1b"
)
LAUNCH_SCRIPT = (
    PROJECT_ROOT
    / "tools/launch_alphazuma_55_motor_observable_gradual_headonly_v1.sh"
)
RUN_DIR = Path(
    "/mnt/d/ZumaTraining/"
    "alphazuma-55-motor-observable-gradual-headonly-s99081642-v1"
)
ORIGINAL_ROOT = Path("/mnt/d/SteamLibrary/steamapps/common/Zuma's Revenge")
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "diagnostics/alphazuma-55-motor-observable-gradual-headonly-"
    "s99081642-launch-receipt-v1.json"
)
EXPECTED_SCHEDULE = (1.0, 0.9, 0.7, 0.5)


def _validate_runtime() -> dict[str, Any]:
    completed = subprocess.run(
        [
            sys.executable,
            str(TRAINER),
            "--preregistration",
            str(PREREGISTRATION),
            "--original-root",
            str(ORIGINAL_ROOT),
            "--validate-only",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    value = json.loads(completed.stdout)
    if not (
        value.get("status") == "VALID"
        and value.get("observation_shape") == [22904]
        and tuple(
            float(item)
            for item in value.get("teacher_execution_probabilities", ())
        )
        == EXPECTED_SCHEDULE
        and int(value.get("epochs_per_round", -1)) == 12
        and value.get("formal_seed_consumption") is False
    ):
        raise ValueError("head-only runtime validation differs")
    return value


def capture(
    *,
    windows_launcher_pid: int,
    wsl_trainer_pid: int,
    stdout_log: Path,
    stderr_log: Path,
    output: Path,
) -> dict[str, Any]:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"launch receipt already exists: {output}")
    preregistration_ref = common._reference(
        PREREGISTRATION, PREREGISTRATION_SHA256
    )
    trainer_ref = common._reference(TRAINER, TRAINER_SHA256)
    builder_ref = common._reference(
        PREREGISTRATION_BUILDER, PREREGISTRATION_BUILDER_SHA256
    )
    audit_ref = common._reference(
        PREREGISTRATION_AUDIT, PREREGISTRATION_AUDIT_SHA256
    )
    audit = common._read(PREREGISTRATION_AUDIT)
    if not (
        audit.get("status") == "PASS"
        and audit.get("formal_seed_consumption") == "NONE"
        and audit.get("formal_candidate_authority") is False
    ):
        raise ValueError("head-only preregistration audit is not PASS")
    if not common._windows_process_alive(windows_launcher_pid):
        raise ValueError("Windows launcher is not alive")
    if not common._wsl_process_alive(wsl_trainer_pid, TRAINER.name):
        raise ValueError("WSL trainer is not alive or has another command")

    validation = _validate_runtime()
    config_path = (RUN_DIR / "config.json").resolve(strict=True)
    status_path = (RUN_DIR / "training_status.json").resolve(strict=True)
    config = common._read(config_path)
    status = common._read(status_path)
    run = config.get("run", {})
    completed_rounds = int(status.get("completed_rounds", -1))
    current_round = int(status.get("current_round", completed_rounds))
    if not (
        config.get("schema") == "zuma-rl.alphazuma-55-polar-dagger-config"
        and config.get("status") == "FROZEN"
        and config.get("preregistration") == preregistration_ref
        and config.get("trainer") == trainer_ref
        and run.get("run_dir") == str(RUN_DIR)
        and run.get("aim_loss_scope") == "fire_only"
        and run.get("route_variant") == "action_head_only_fire_aim_v1"
        and run.get("optimizer_parameter_scope") == "action_net_only"
        and run.get("frozen_backbone") is True
        and status.get("schema") == "zuma-rl.alphazuma-55-polar-dagger-status"
        and status.get("status") == "RUNNING"
        and status.get("stage") == "COLLECTING"
        and completed_rounds == 0
        and current_round == 0
        and int(status.get("expected_rounds", -1)) == 4
    ):
        raise ValueError("initial head-only training artifacts differ")

    available_memory, swap_used = common._memory_snapshot()
    root_usage = shutil.disk_usage(Path("/"))
    d_usage = shutil.disk_usage(Path("/mnt/d"))
    receipt = {
        "schema": "zuma-rl.alphazuma-55-motor-observable-gradual-launch-receipt",
        "version": 1,
        "status": "LAUNCHED_AND_COLLECTING",
        "observed_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_id": CAMPAIGN_ID,
        "bindings": {
            "preregistration": preregistration_ref,
            "trainer": trainer_ref,
            "builder": builder_ref,
            "preregistration_independent_audit": audit_ref,
            "launch_script": common._reference(LAUNCH_SCRIPT),
            "launch_receipt_builder": common._reference(SCRIPT_PATH),
        },
        "validation": {
            "status": "VALID",
            "observation_shape": validation["observation_shape"],
            "teacher_execution_probabilities": list(EXPECTED_SCHEDULE),
            "epochs_per_round": 12,
            "formal_seed_consumption": False,
        },
        "processes": {
            "windows_launcher_pid": windows_launcher_pid,
            "wsl_trainer_pid": wsl_trainer_pid,
            "windows_launcher_alive": True,
            "wsl_trainer_alive": True,
        },
        "logs": {
            "stdout": common._log_snapshot(stdout_log),
            "stderr": common._log_snapshot(stderr_log),
        },
        "initial_artifacts": {
            "config": common._reference(config_path),
            "training_status": {
                "path": str(status_path),
                "sha256_at_snapshot": common._sha256(status_path),
                "stage": "COLLECTING",
                "current_round": 0,
                "teacher_execution_probability": 1.0,
                "completed_rounds": 0,
                "expected_rounds": 4,
            },
        },
        "resource_snapshot": {
            "wsl_available_memory_bytes": available_memory,
            "wsl_swap_used_bytes": swap_used,
            "wsl_root_available_bytes": root_usage.free,
            "d_drive_available_bytes": d_usage.free,
            "gpus": common._gpu_snapshot(),
        },
        "authority_boundary": {
            "engineering_training_route_only": True,
            "training_seed_reuse_disclosed": True,
            "formal_selection_seed_consumption": "NONE",
            "formal_final_blind_seed_consumption": "NONE",
            "continuous_campaign_seed_consumption": "NONE",
            "formal_candidate_authority": False,
            "power_restore_authority": False,
        },
    }
    common._write_new(output, receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows-launcher-pid", required=True, type=int)
    parser.add_argument("--wsl-trainer-pid", required=True, type=int)
    parser.add_argument("--stdout-log", required=True, type=Path)
    parser.add_argument("--stderr-log", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    receipt = capture(
        windows_launcher_pid=args.windows_launcher_pid,
        wsl_trainer_pid=args.wsl_trainer_pid,
        stdout_log=args.stdout_log,
        stderr_log=args.stderr_log,
        output=args.output,
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "campaign_id": receipt["campaign_id"],
                "receipt": {
                    "path": str(args.output.resolve()),
                    "sha256": common._sha256(args.output.resolve(strict=True)),
                },
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
