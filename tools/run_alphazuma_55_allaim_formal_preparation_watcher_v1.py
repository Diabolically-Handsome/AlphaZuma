"""Freeze and launch the all-actions formal successor while engineering runs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
CONTINGENCY_STATUS = Path(
    "/mnt/d/ZumaTraining/alphazuma-55-allaim-contingency-watcher-"
    "s99081635-v1/controller_status.json"
)
ENGINEERING_PLAN = (
    PROJECT_ROOT
    / "diagnostics/alphazuma-55-motor-observable-gradual-allaim-"
    "postprocess-s99081634-plan-v1.json"
)
MASTER = (
    PROJECT_ROOT
    / "diagnostics/alphazuma-55-motor-observable-gradual-allaim-"
    "successor-s99081636-preregistration-v1.json"
)
REGISTRY = (
    PROJECT_ROOT
    / "diagnostics/alphazuma-55-weekend-formal-seed-registry-"
    "s99081637-v7.json"
)
MASTER_BUILDER = (
    PROJECT_ROOT
    / "tools/build_alphazuma_55_motor_observable_gradual_allaim_"
    "successor_master_v1.py"
)
MASTER_BUILDER_SHA256 = (
    "sha256:9af157a675a1c42bad8ce42d144c77bdfee6874eb365101a717be66e95536dfb"
)
REGISTRY_BUILDER = PROJECT_ROOT / "tools/build_alphazuma_55_formal_seed_registry_v7.py"
REGISTRY_BUILDER_SHA256 = (
    "sha256:1cc92c7cd852cc173c5b67e2340072e5fc7bc4c210cffe63585b08c1ab84f7f5"
)
CONTROLLER = (
    PROJECT_ROOT
    / "tools/run_alphazuma_55_motor_observable_gradual_allaim_"
    "successor_v1.py"
)
CONTROLLER_SHA256 = (
    "sha256:82af3e08941ead744ae5193e7a40086f1016d395f13a4586bfcbcccf33395fc4"
)
AUDITOR = (
    PROJECT_ROOT
    / "tools/audit_alphazuma_55_motor_observable_gradual_allaim_"
    "successor_v1.py"
)
AUDITOR_SHA256 = (
    "sha256:146aec3051ec7eee55294ec3134198c3be02b492c04c3ff402f1afe84d00f166"
)
LAUNCH_SCRIPT = (
    PROJECT_ROOT
    / "tools/launch_alphazuma_55_motor_observable_gradual_allaim_"
    "successor_v1.sh"
)
LAUNCH_SCRIPT_SHA256 = (
    "sha256:ba4e51d6f52dc6da1ef1d954d3e6b4bb2f6326045aa36be9bee4e3cdfeaa5caa"
)
SUCCESSOR_ROOT = Path(
    "/mnt/d/ZumaTraining/alphazuma-55-motor-observable-gradual-allaim-"
    "successor-s99081636-v1"
)
WATCH_ROOT = Path(
    "/mnt/d/ZumaTraining/alphazuma-55-allaim-formal-preparation-watcher-"
    "s99081638-v1"
)
WATCH_STATUS = WATCH_ROOT / "controller_status.json"
STDOUT_LOG = Path(
    "/mnt/d/ZumaTraining/alphazuma-55-motor-observable-gradual-allaim-"
    "successor-s99081636-launcher-01.stdout.log"
)
STDERR_LOG = Path(
    "/mnt/d/ZumaTraining/alphazuma-55-motor-observable-gradual-allaim-"
    "successor-s99081636-launcher-01.stderr.log"
)


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
        "schema": "zuma-rl.alphazuma-55-allaim-formal-preparation-status",
        "version": 1,
        "status": "RUNNING",
        "phase": phase,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
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


def _assert_hashes() -> None:
    for path, expected in {
        MASTER_BUILDER: MASTER_BUILDER_SHA256,
        REGISTRY_BUILDER: REGISTRY_BUILDER_SHA256,
        CONTROLLER: CONTROLLER_SHA256,
        AUDITOR: AUDITOR_SHA256,
        LAUNCH_SCRIPT: LAUNCH_SCRIPT_SHA256,
    }.items():
        if _sha256(path.resolve(strict=True)) != expected:
            raise ValueError(f"all-action formal implementation differs: {path}")


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, capture_output=True, text=True)


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


def _powershell_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _start_successor(expected_master_sha256: str) -> int:
    if STDOUT_LOG.exists() or STDERR_LOG.exists():
        raise FileExistsError("all-action successor launcher logs exist")
    windows_stdout = _run(
        ["wslpath", "-w", str(STDOUT_LOG)]
    ).stdout.strip()
    windows_stderr = _run(
        ["wslpath", "-w", str(STDERR_LOG)]
    ).stdout.strip()
    values = [
        "-e",
        "bash",
        str(LAUNCH_SCRIPT.resolve(strict=True)),
        expected_master_sha256,
    ]
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
    completed = _run(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            command,
        ]
    )
    return int(completed.stdout.strip().splitlines()[-1])


def _wait_for_contingency(poll_seconds: float) -> bool:
    while True:
        if CONTINGENCY_STATUS.exists():
            status = _read(CONTINGENCY_STATUS)
            if status.get("status") == "ERROR":
                raise RuntimeError("all-action contingency watcher reported ERROR")
            if status.get("status") == "COMPLETE":
                phase = str(status.get("phase"))
                if phase == "NOT_TRIGGERED_CURRENT_GATE_PASS":
                    return False
                if phase == "ALLAIM_TRAINING_AND_POSTPROCESS_LAUNCHED":
                    return True
                raise ValueError("unexpected terminal contingency phase")
        _write_status("WAITING_FOR_ALLAIM_CONTINGENCY_DECISION")
        time.sleep(poll_seconds)


def run(*, poll_seconds: float) -> int:
    _assert_hashes()
    if not _wait_for_contingency(poll_seconds):
        _write_status(
            "NOT_NEEDED_CURRENT_GATE_PASS",
            status="COMPLETE",
        )
        return 0
    while not ENGINEERING_PLAN.exists():
        _write_status("WAITING_FOR_ALLAIM_ENGINEERING_PLAN")
        time.sleep(poll_seconds)
    if any(path.exists() for path in (MASTER, REGISTRY, SUCCESSOR_ROOT, STDOUT_LOG, STDERR_LOG)):
        raise FileExistsError("all-action formal fresh boundary differs")
    plan_sha256 = _sha256(ENGINEERING_PLAN.resolve(strict=True))
    _run(
        [
            sys.executable,
            str(MASTER_BUILDER),
            "--expected-engineering-plan-sha256",
            plan_sha256,
            "--output",
            str(MASTER),
        ]
    )
    master_sha256 = _sha256(MASTER.resolve(strict=True))
    _run(
        [
            sys.executable,
            str(REGISTRY_BUILDER),
            "--expected-master-sha256",
            master_sha256,
            "--output",
            str(REGISTRY),
        ]
    )
    registry_sha256 = _sha256(REGISTRY.resolve(strict=True))
    validation = _run(
        [
            sys.executable,
            str(CONTROLLER),
            "--master",
            str(MASTER),
            "--expected-master-sha256",
            master_sha256,
            "--original-root",
            "/mnt/d/SteamLibrary/steamapps/common/Zuma's Revenge",
            "--validate-only",
        ]
    )
    validated = json.loads(validation.stdout)
    if not (
        validated.get("status") == "VALID"
        and validated.get("formal_seed_consumption") == "NONE"
        and int(validated.get("final_blind_attempts", -1)) == 440
        and int(validated.get("continuous_attempts", -1)) == 220
    ):
        raise ValueError("all-action formal successor validation differs")
    windows_pid = _start_successor(master_sha256)
    deadline = time.monotonic() + 120.0
    wsl_pids: list[int] = []
    while time.monotonic() < deadline:
        wsl_pids = _processes(CONTROLLER.name)
        if len(wsl_pids) == 1:
            break
        if len(wsl_pids) > 1:
            raise RuntimeError(f"multiple all-action successor processes: {wsl_pids}")
        time.sleep(2.0)
    if len(wsl_pids) != 1:
        raise TimeoutError("all-action successor process did not appear")
    _write_status(
        "ALLAIM_FORMAL_SUCCESSOR_LAUNCHED_AND_EMBARGOED",
        status="COMPLETE",
        engineering_plan={"path": str(ENGINEERING_PLAN), "sha256": plan_sha256},
        master={"path": str(MASTER), "sha256": master_sha256},
        registry={"path": str(REGISTRY), "sha256": registry_sha256},
        processes={"windows_launcher_pid": windows_pid, "wsl_pid": wsl_pids[0]},
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
                    "master_exists": MASTER.exists(),
                    "registry_exists": REGISTRY.exists(),
                    "formal_seed_consumption": "NONE",
                    "final_seed_range": [3_700_000_000, 3_700_000_439],
                    "continuous_seed_range": [3_800_000_000, 3_800_000_219],
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
