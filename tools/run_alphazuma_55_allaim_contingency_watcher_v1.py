"""Launch the audited all-actions DAgger contingency after a clean no-promotion."""

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
import time
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
CURRENT_ROOT = Path(
    "/mnt/d/ZumaTraining/alphazuma-55-motor-observable-gradual-postprocess-"
    "s99081630-v1"
)
CURRENT_STATUS = CURRENT_ROOT / "controller_status.json"
CURRENT_REPORT = CURRENT_ROOT / "final_report.json"
CURRENT_AUDIT = (
    PROJECT_ROOT
    / "diagnostics/alphazuma-55-motor-observable-gradual-postprocess-"
    "s99081630-independent-audit-v1.json"
)
CURRENT_TRAINER_FRAGMENT = "distill_alphazuma_55_motor_observable_gradual_v2.py"
CURRENT_POSTPROCESS_FRAGMENT = (
    "run_alphazuma_55_motor_observable_gradual_postprocess_v1.py"
)
ALLAIM_CAMPAIGN = "alphazuma-55-motor-observable-gradual-allaim-s99081633-v1"
ALLAIM_RUN_DIR = Path(f"/mnt/d/ZumaTraining/{ALLAIM_CAMPAIGN}")
ALLAIM_POSTPROCESS_CAMPAIGN = (
    "alphazuma-55-motor-observable-gradual-allaim-postprocess-s99081634-v1"
)
ALLAIM_POSTPROCESS_ROOT = Path(
    f"/mnt/d/ZumaTraining/{ALLAIM_POSTPROCESS_CAMPAIGN}"
)
WATCH_ROOT = Path(
    "/mnt/d/ZumaTraining/alphazuma-55-allaim-contingency-watcher-s99081635-v1"
)
WATCH_STATUS = WATCH_ROOT / "controller_status.json"
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
TRAINER_LAUNCH_SCRIPT = (
    PROJECT_ROOT / "tools/launch_alphazuma_55_motor_observable_gradual_allaim_v1.sh"
)
TRAINER_LAUNCH_SCRIPT_SHA256 = (
    "sha256:f263631f2de837ebd8cf04e39b33134cd8b99c8d5ae8d610c57f8db970c5d489"
)
CAPTURE = (
    PROJECT_ROOT
    / "tools/capture_alphazuma_55_motor_observable_gradual_allaim_"
    "launch_receipt_v1.py"
)
CAPTURE_SHA256 = (
    "sha256:c80a13cfdbaf91cd923fbf1cbee7786ae429a3b77b01dae8d25c4e2703d5fd69"
)
LAUNCH_RECEIPT = (
    PROJECT_ROOT
    / "diagnostics/alphazuma-55-motor-observable-gradual-allaim-"
    "s99081633-launch-receipt-v1.json"
)
PLAN_BUILDER = (
    PROJECT_ROOT
    / "tools/build_alphazuma_55_motor_observable_gradual_allaim_"
    "postprocess_plan_v1.py"
)
PLAN_BUILDER_SHA256 = (
    "sha256:66b11d8fc1961fdfd63982c4d912f394f1eee2966f5107e26ad87722ba5fe7f2"
)
POSTPROCESS_CONTROLLER = (
    PROJECT_ROOT
    / "tools/run_alphazuma_55_motor_observable_gradual_allaim_"
    "postprocess_v1.py"
)
POSTPROCESS_CONTROLLER_SHA256 = (
    "sha256:fbb8346b062f58c586614e1fcab39014a3d63e98cc7e82fe534a72cbc2c57aec"
)
POSTPROCESS_LAUNCH_SCRIPT = (
    PROJECT_ROOT
    / "tools/launch_alphazuma_55_motor_observable_gradual_allaim_"
    "postprocess_v1.sh"
)
POSTPROCESS_LAUNCH_SCRIPT_SHA256 = (
    "sha256:30092ce474489bbe4571211c7ed8af80f2674f36892abc976a53cbbe7c08e617"
)
POSTPROCESS_PLAN = (
    PROJECT_ROOT
    / "diagnostics/alphazuma-55-motor-observable-gradual-allaim-"
    "postprocess-s99081634-plan-v1.json"
)
TRAINER_STDOUT = Path(
    "/mnt/d/ZumaTraining/alphazuma-55-motor-observable-gradual-allaim-"
    "s99081633-launcher-01.stdout.log"
)
TRAINER_STDERR = Path(
    "/mnt/d/ZumaTraining/alphazuma-55-motor-observable-gradual-allaim-"
    "s99081633-launcher-01.stderr.log"
)
POSTPROCESS_STDOUT = Path(
    "/mnt/d/ZumaTraining/alphazuma-55-motor-observable-gradual-allaim-"
    "postprocess-s99081634-launcher-01.stdout.log"
)
POSTPROCESS_STDERR = Path(
    "/mnt/d/ZumaTraining/alphazuma-55-motor-observable-gradual-allaim-"
    "postprocess-s99081634-launcher-01.stderr.log"
)
MINIMUM_AVAILABLE_MEMORY = 35 * 1024**3
MINIMUM_ROOT_FREE = 12 * 1024**3


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _write_status(phase: str, **extra: Any) -> None:
    WATCH_ROOT.mkdir(parents=True, exist_ok=True)
    value = {
        "schema": "zuma-rl.alphazuma-55-allaim-contingency-watcher-status",
        "version": 1,
        "status": "RUNNING",
        "phase": phase,
        "updated_utc": _utc_now(),
        "formal_seed_consumption": "NONE",
        "power_restore_authority": False,
        "error": None,
        **extra,
    }
    temporary = WATCH_STATUS.with_name(f".{WATCH_STATUS.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, WATCH_STATUS)


def _processes(fragment: str) -> list[int]:
    rows: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ")
        except OSError:
            continue
        if fragment.encode("utf-8") in command:
            rows.append(int(entry.name))
    return sorted(rows)


def _memory() -> tuple[int, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, raw = line.split(":", 1)
        values[key] = int(raw.strip().split()[0]) * 1024
    return values["MemAvailable"], values["SwapTotal"] - values["SwapFree"]


def _gpu_limits() -> list[float]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=power.limit",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return [float(line.strip()) for line in completed.stdout.splitlines()]


def _assert_hashes() -> None:
    expected = {
        PREREGISTRATION: PREREGISTRATION_SHA256,
        PREREGISTRATION_AUDIT: PREREGISTRATION_AUDIT_SHA256,
        TRAINER: TRAINER_SHA256,
        TRAINER_LAUNCH_SCRIPT: TRAINER_LAUNCH_SCRIPT_SHA256,
        CAPTURE: CAPTURE_SHA256,
        PLAN_BUILDER: PLAN_BUILDER_SHA256,
        POSTPROCESS_CONTROLLER: POSTPROCESS_CONTROLLER_SHA256,
        POSTPROCESS_LAUNCH_SCRIPT: POSTPROCESS_LAUNCH_SCRIPT_SHA256,
    }
    for path, digest in expected.items():
        if _sha256(path.resolve(strict=True)) != digest:
            raise ValueError(f"frozen all-action implementation differs: {path}")
    audit = _read(PREREGISTRATION_AUDIT)
    if not (
        audit.get("status") == "PASS"
        and audit.get("formal_seed_consumption") == "NONE"
        and audit.get("formal_candidate_authority") is False
    ):
        raise ValueError("all-action preregistration audit is ineligible")


def _powershell_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _start_wsl(
    *, script: Path, arguments: list[str], stdout: Path, stderr: Path
) -> int:
    if stdout.exists() or stderr.exists():
        raise FileExistsError("launcher log already exists")
    wsl_script = str(script.resolve(strict=True))
    windows_stdout = subprocess.run(
        ["wslpath", "-w", str(stdout)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    windows_stderr = subprocess.run(
        ["wslpath", "-w", str(stderr)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    values = ["-e", "bash", wsl_script, *arguments]
    array = ",".join(_powershell_literal(value) for value in values)
    command = (
        f"$launchArgs=@({array}); "
        "$process=Start-Process -FilePath 'wsl.exe' "
        "-ArgumentList $launchArgs "
        f"-RedirectStandardOutput {_powershell_literal(windows_stdout)} "
        f"-RedirectStandardError {_powershell_literal(windows_stderr)} "
        "-WindowStyle Hidden -PassThru; "
        "[Console]::WriteLine($process.Id)"
    )
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            command,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(completed.stdout.strip().splitlines()[-1])


def _wait_for_single_process(fragment: str, timeout_seconds: float) -> int:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        rows = _processes(fragment)
        if len(rows) == 1:
            return rows[0]
        if len(rows) > 1:
            raise RuntimeError(f"multiple processes match {fragment}: {rows}")
        time.sleep(2.0)
    raise TimeoutError(f"process did not appear: {fragment}")


def _run_checked(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, capture_output=True, text=True)


def _wait_for_current_result(poll_seconds: float) -> bool:
    while True:
        if CURRENT_STATUS.exists():
            status = _read(CURRENT_STATUS)
            if status.get("status") == "COMPLETE":
                if not CURRENT_REPORT.exists() or not CURRENT_AUDIT.exists():
                    raise RuntimeError("current postprocess completed without evidence")
                report = _read(CURRENT_REPORT)
                audit = _read(CURRENT_AUDIT)
                gate_passed = bool(status.get("gate_passed"))
                expected_result = (
                    "COMPLETE_GATE_PASS"
                    if gate_passed
                    else "COMPLETE_NO_PROMOTION"
                )
                if not (
                    report.get("status") == expected_result
                    and report.get("promotion_gate", {}).get("passed")
                    is gate_passed
                    and audit.get("status") == "PASS"
                    and audit.get("gate_passed") is gate_passed
                    and audit.get("controller_result") == expected_result
                    and audit.get("formal_seed_consumption") == "NONE"
                ):
                    raise ValueError("current independent engineering result differs")
                return gate_passed
            if status.get("status") == "ERROR":
                raise RuntimeError("current postprocess reported ERROR")
        _write_status("WAITING_FOR_CURRENT_ENGINEERING_RESULT")
        time.sleep(poll_seconds)


def _wait_for_resource_release(poll_seconds: float) -> dict[str, Any]:
    while True:
        available, swap_used = _memory()
        root_free = shutil.disk_usage(Path("/")).free
        d_free = shutil.disk_usage(Path("/mnt/d")).free
        current_processes = _processes(CURRENT_TRAINER_FRAGMENT) + _processes(
            CURRENT_POSTPROCESS_FRAGMENT
        )
        limits = _gpu_limits()
        snapshot = {
            "available_memory_bytes": available,
            "swap_used_bytes": swap_used,
            "root_free_bytes": root_free,
            "d_free_bytes": d_free,
            "current_processes": current_processes,
            "gpu_power_limits_watts": limits,
        }
        if root_free < MINIMUM_ROOT_FREE:
            raise RuntimeError("WSL root free space fell below 12 GiB")
        if limits != [550.0, 250.0]:
            raise RuntimeError("GPU power limits differ from 550W/250W")
        if (
            available >= MINIMUM_AVAILABLE_MEMORY
            and swap_used == 0
            and not current_processes
        ):
            return snapshot
        _write_status("WAITING_FOR_RESOURCE_RELEASE", resources=snapshot)
        time.sleep(poll_seconds)


def run(*, poll_seconds: float) -> int:
    _assert_hashes()
    gate_passed = _wait_for_current_result(poll_seconds)
    if gate_passed:
        _write_status(
            "NOT_TRIGGERED_CURRENT_GATE_PASS",
            status="COMPLETE",
            trigger="CURRENT_GRADUAL_ROUTE_PROMOTED",
        )
        return 0

    resources = _wait_for_resource_release(poll_seconds)
    forbidden = [
        ALLAIM_RUN_DIR,
        ALLAIM_POSTPROCESS_ROOT,
        LAUNCH_RECEIPT,
        POSTPROCESS_PLAN,
        TRAINER_STDOUT,
        TRAINER_STDERR,
        POSTPROCESS_STDOUT,
        POSTPROCESS_STDERR,
    ]
    existing = [str(path) for path in forbidden if path.exists()]
    if existing:
        raise FileExistsError(f"all-action fresh-launch boundary differs: {existing}")

    _run_checked(
        [
            sys.executable,
            str(TRAINER),
            "--preregistration",
            str(PREREGISTRATION),
            "--original-root",
            "/mnt/d/SteamLibrary/steamapps/common/Zuma's Revenge",
            "--validate-only",
        ]
    )
    windows_trainer_pid = _start_wsl(
        script=TRAINER_LAUNCH_SCRIPT,
        arguments=[],
        stdout=TRAINER_STDOUT,
        stderr=TRAINER_STDERR,
    )
    wsl_trainer_pid = _wait_for_single_process(TRAINER.name, 120.0)
    deadline = time.monotonic() + 180.0
    while time.monotonic() < deadline and not (
        (ALLAIM_RUN_DIR / "config.json").exists()
        and (ALLAIM_RUN_DIR / "training_status.json").exists()
    ):
        time.sleep(2.0)
    if not (ALLAIM_RUN_DIR / "training_status.json").exists():
        raise TimeoutError("all-action initial artifacts did not appear")

    _run_checked(
        [
            sys.executable,
            str(CAPTURE),
            "--windows-launcher-pid",
            str(windows_trainer_pid),
            "--wsl-trainer-pid",
            str(wsl_trainer_pid),
            "--stdout-log",
            str(TRAINER_STDOUT),
            "--stderr-log",
            str(TRAINER_STDERR),
            "--output",
            str(LAUNCH_RECEIPT),
        ]
    )
    _run_checked([sys.executable, str(PLAN_BUILDER), "--output", str(POSTPROCESS_PLAN)])
    plan_sha256 = _sha256(POSTPROCESS_PLAN.resolve(strict=True))
    _run_checked(
        [
            sys.executable,
            str(POSTPROCESS_CONTROLLER),
            "--plan",
            str(POSTPROCESS_PLAN),
            "--expected-plan-sha256",
            plan_sha256,
            "--validate-only",
        ]
    )
    windows_postprocess_pid = _start_wsl(
        script=POSTPROCESS_LAUNCH_SCRIPT,
        arguments=[plan_sha256],
        stdout=POSTPROCESS_STDOUT,
        stderr=POSTPROCESS_STDERR,
    )
    wsl_postprocess_pid = _wait_for_single_process(
        POSTPROCESS_CONTROLLER.name, 120.0
    )
    _write_status(
        "ALLAIM_TRAINING_AND_POSTPROCESS_LAUNCHED",
        status="COMPLETE",
        trigger="CURRENT_GRADUAL_ROUTE_COMPLETE_NO_PROMOTION",
        resources_at_release=resources,
        training={
            "windows_launcher_pid": windows_trainer_pid,
            "wsl_pid": wsl_trainer_pid,
            "launch_receipt": {
                "path": str(LAUNCH_RECEIPT),
                "sha256": _sha256(LAUNCH_RECEIPT),
            },
        },
        postprocess={
            "windows_launcher_pid": windows_postprocess_pid,
            "wsl_pid": wsl_postprocess_pid,
            "plan": {"path": str(POSTPROCESS_PLAN), "sha256": plan_sha256},
        },
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_only:
        _assert_hashes()
        print(
            json.dumps(
                {
                    "status": "VALID",
                    "current_engineering_status_exists": CURRENT_STATUS.exists(),
                    "allaim_run_dir_exists": ALLAIM_RUN_DIR.exists(),
                    "allaim_postprocess_root_exists": (
                        ALLAIM_POSTPROCESS_ROOT.exists()
                    ),
                    "formal_seed_consumption": "NONE",
                    "minimum_available_memory_bytes": (
                        MINIMUM_AVAILABLE_MEMORY
                    ),
                    "gpu_power_limits_watts": [550.0, 250.0],
                },
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        )
        return 0
    try:
        return run(poll_seconds=max(5.0, float(args.poll_seconds)))
    except BaseException as error:
        _write_status(
            "ERROR",
            status="ERROR",
            error_type=type(error).__name__,
            error=str(error),
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
