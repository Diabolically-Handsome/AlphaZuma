"""Capture an immutable launch receipt for the all-actions motor DAgger route."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
CAMPAIGN_ID = "alphazuma-55-motor-observable-gradual-allaim-s99081633-v1"
PREREGISTRATION = (
    PROJECT_ROOT
    / "diagnostics/alphazuma-55-motor-observable-gradual-allaim-"
    "s99081633-preregistration-v1.json"
)
PREREGISTRATION_SHA256 = (
    "sha256:502c0ab5566a66fa33ff33df80b10b101c5ea1b75dc4e21ff07f5996965730e1"
)
PREREGISTRATION_AUDIT = (
    PROJECT_ROOT
    / "diagnostics/alphazuma-55-motor-observable-gradual-allaim-"
    "s99081633-independent-prereg-audit-v1.json"
)
PREREGISTRATION_AUDIT_SHA256 = (
    "sha256:61be913b30601f2e529cc8112dd18aab8b434d82a7364a794e1ff756c5d99b6d"
)
TRAINER = (
    PROJECT_ROOT
    / "tools/distill_alphazuma_55_motor_observable_gradual_allaim_v1.py"
)
TRAINER_SHA256 = (
    "sha256:4e55d406a65be0a7ea1d0e4302b736cf1b3fb1dd48e85e4f5d04633f21cf1b3a"
)
PREREGISTRATION_BUILDER = (
    PROJECT_ROOT
    / "tools/build_alphazuma_55_motor_observable_gradual_allaim_v1.py"
)
PREREGISTRATION_BUILDER_SHA256 = (
    "sha256:37254db4f8e68966fa5b5dc96d94e4cb368ac80d78ce5939e63cf6825e9a8663"
)
RUN_DIR = Path(
    "/mnt/d/ZumaTraining/alphazuma-55-motor-observable-gradual-allaim-"
    "s99081633-v1"
)
ORIGINAL_ROOT = Path("/mnt/d/SteamLibrary/steamapps/common/Zuma's Revenge")
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "diagnostics/alphazuma-55-motor-observable-gradual-allaim-"
    "s99081633-launch-receipt-v1.json"
)
EXPECTED_SCHEDULE = (1.0, 0.9, 0.7, 0.5)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _reference(path: Path, expected_sha256: str | None = None) -> dict[str, str]:
    resolved = path.resolve(strict=True)
    actual = _sha256(resolved)
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError(f"frozen bytes differ: {resolved}")
    return {"path": str(resolved), "sha256": actual}


def _wsl_process_alive(pid: int, expected_fragment: str) -> bool:
    if pid <= 0:
        return False
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ")
    except OSError:
        return False
    return expected_fragment.encode("utf-8") in command


def _windows_process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    command = (
        f"if (Get-Process -Id {pid} -ErrorAction SilentlyContinue) "
        "{ exit 0 } else { exit 1 }"
    )
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            command,
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return completed.returncode == 0


def _windows_path(path: Path) -> str:
    completed = subprocess.run(
        ["wslpath", "-w", str(path.resolve(strict=True))],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _log_snapshot(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": _windows_path(resolved),
        "bytes_at_snapshot": resolved.stat().st_size,
        "sha256_at_snapshot": _sha256(resolved),
    }


def _memory_snapshot() -> tuple[int, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, raw = line.split(":", 1)
        values[key] = int(raw.strip().split()[0]) * 1024
    return (
        values["MemAvailable"],
        values["SwapTotal"] - values["SwapFree"],
    )


def _gpu_snapshot() -> list[dict[str, Any]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,power.draw,power.limit,utilization.gpu,"
            "memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 8:
            raise ValueError("unexpected nvidia-smi CSV")
        rows.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "power_draw_watts": float(parts[2]),
                "power_limit_watts": float(parts[3]),
                "utilization_percent": int(parts[4]),
                "memory_used_mib": int(parts[5]),
                "memory_total_mib": int(parts[6]),
                "temperature_c": int(parts[7]),
            }
        )
    return rows


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
        raise ValueError("all-action runtime validation differs")
    return value


def _write_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=False,
            allow_nan=False,
        )
        + "\n"
    )
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


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
    preregistration_ref = _reference(PREREGISTRATION, PREREGISTRATION_SHA256)
    trainer_ref = _reference(TRAINER, TRAINER_SHA256)
    builder_ref = _reference(
        PREREGISTRATION_BUILDER, PREREGISTRATION_BUILDER_SHA256
    )
    audit_ref = _reference(
        PREREGISTRATION_AUDIT, PREREGISTRATION_AUDIT_SHA256
    )
    audit = _read(PREREGISTRATION_AUDIT)
    if audit.get("status") != "PASS":
        raise ValueError("all-action preregistration audit is not PASS")
    if not _windows_process_alive(windows_launcher_pid):
        raise ValueError("Windows launcher is not alive")
    if not _wsl_process_alive(wsl_trainer_pid, TRAINER.name):
        raise ValueError("WSL trainer is not alive or has another command")

    validation = _validate_runtime()
    config_path = (RUN_DIR / "config.json").resolve(strict=True)
    status_path = (RUN_DIR / "training_status.json").resolve(strict=True)
    config = _read(config_path)
    status = _read(status_path)
    run = config.get("run", {})
    completed_rounds = int(status.get("completed_rounds", -1))
    current_round = int(status.get("current_round", completed_rounds))
    if not (
        config.get("schema") == "zuma-rl.alphazuma-55-polar-dagger-config"
        and config.get("status") == "FROZEN"
        and config.get("preregistration") == preregistration_ref
        and config.get("trainer") == trainer_ref
        and run.get("run_dir") == str(RUN_DIR)
        and run.get("aim_loss_scope") == "all_actions"
        and run.get("route_variant") == "all_actions_aim_supervision_v1"
        and run.get("all_action_aim_supervision") is True
        and status.get("schema") == "zuma-rl.alphazuma-55-polar-dagger-status"
        and status.get("status") == "RUNNING"
        and status.get("stage") == "COLLECTING"
        and completed_rounds == 0
        and current_round == 0
        and int(status.get("expected_rounds", -1)) == 4
    ):
        raise ValueError("initial all-action training artifacts differ")

    available_memory, swap_used = _memory_snapshot()
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
            "launch_receipt_builder": _reference(SCRIPT_PATH),
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
            "stdout": _log_snapshot(stdout_log),
            "stderr": _log_snapshot(stderr_log),
        },
        "initial_artifacts": {
            "config": _reference(config_path),
            "training_status": {
                "path": str(status_path),
                "sha256_at_snapshot": _sha256(status_path),
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
            "gpus": _gpu_snapshot(),
        },
        "authority_boundary": {
            "engineering_training_route_only": True,
            "formal_selection_seed_consumption": "NONE",
            "formal_final_blind_seed_consumption": "NONE",
            "continuous_campaign_seed_consumption": "NONE",
            "formal_candidate_authority": False,
            "power_restore_authority": False,
        },
    }
    _write_new(output, receipt)
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
                    "sha256": _sha256(args.output.resolve(strict=True)),
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
