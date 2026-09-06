"""Watch a successor controller and run its bound independent auditor."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools import run_alphazuma_55_successor_certification as controller


SCRIPT_PATH = Path(__file__).resolve()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _write_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def run(master_path: Path, expected_master_sha256: str, controller_pid: int, poll_seconds: float) -> int:
    master_path = master_path.resolve(strict=True)
    if controller._sha256(master_path) != expected_master_sha256:
        raise ValueError("successor master hash differs")
    contract = controller._load_master(master_path)
    implementation = contract["master"]["implementation"]
    watcher_reference = implementation.get("watcher")
    if not isinstance(watcher_reference, dict):
        raise ValueError("successor master does not bind the watcher")
    if (
        Path(str(watcher_reference.get("path", ""))).resolve(strict=True)
        != SCRIPT_PATH
        or watcher_reference.get("sha256") != controller._sha256(SCRIPT_PATH)
    ):
        raise ValueError("successor watcher binding differs")
    status_root = Path(str(contract["master"]["outputs"]["watcher"])).resolve()
    status_root.mkdir(parents=True, exist_ok=False)
    status_path = status_root / "watcher_status.json"
    controller_status_path = contract["outputs"]["controller"] / "controller_status.json"
    audit_receipt = contract["outputs"]["independent_audit_receipt"]
    state: dict[str, Any] = {
        "schema": "zuma-rl.alphazuma-55-successor-watcher-status",
        "version": 1,
        "status": "RUNNING",
        "phase": "WAITING_FOR_CONTROLLER",
        "started_utc": _utc_now(),
        "updated_utc": _utc_now(),
        "master_preregistration": {
            "path": str(master_path),
            "sha256": expected_master_sha256,
        },
        "controller_pid": controller_pid,
        "power_restore_authority": False,
        "error": None,
    }
    _write_atomic(status_path, state)
    try:
        terminal = {"COMPLETE", "COMPLETE_NO_PROMOTION", "FAILED"}
        while True:
            controller_status = (
                _read(controller_status_path) if controller_status_path.exists() else None
            )
            state["updated_utc"] = _utc_now()
            state["controller"] = controller_status
            _write_atomic(status_path, state)
            if controller_status and controller_status.get("status") in terminal:
                break
            if not _pid_alive(controller_pid):
                raise RuntimeError("successor controller exited without a terminal receipt")
            time.sleep(poll_seconds)
        if controller_status is None or controller_status.get("status") == "FAILED":
            raise RuntimeError("successor controller reported FAILED")
        auditor = Path(str(implementation["independent_auditor"]["path"])).resolve(
            strict=True
        )
        state["phase"] = "INDEPENDENT_AUDIT"
        state["updated_utc"] = _utc_now()
        _write_atomic(status_path, state)
        if audit_receipt.exists():
            receipt = _read(audit_receipt)
            if receipt.get("status") != "PASS":
                raise RuntimeError("existing successor audit receipt is not PASS")
        else:
            command = [
                sys.executable,
                str(auditor),
                "--master-preregistration",
                str(master_path),
                "--receipt",
                str(audit_receipt),
            ]
            environment = os.environ.copy()
            environment.update({"TMPDIR": "/tmp", "TEMP": "/tmp"})
            with (status_root / "auditor.stdout.log").open(
                "x", encoding="utf-8", newline="\n"
            ) as stdout, (status_root / "auditor.stderr.log").open(
                "x", encoding="utf-8", newline="\n"
            ) as stderr:
                result = subprocess.run(
                    command,
                    cwd=controller.PROJECT_ROOT,
                    env=environment,
                    stdout=stdout,
                    stderr=stderr,
                    text=True,
                    check=False,
                )
            if result.returncode:
                raise RuntimeError(
                    f"successor independent auditor exited with {result.returncode}"
                )
            receipt = _read(audit_receipt)
            if receipt.get("status") != "PASS":
                raise RuntimeError("successor independent audit is not PASS")
        state.update(
            {
                "status": "COMPLETE",
                "phase": "COMPLETE",
                "updated_utc": _utc_now(),
                "controller_result": controller_status["status"],
                "independent_audit": {
                    "path": str(audit_receipt),
                    "sha256": controller._sha256(audit_receipt),
                    "status": receipt["status"],
                    "controller_result": receipt["controller_result"],
                },
                "power_restore_authority": False,
            }
        )
        _write_atomic(status_path, state)
        return 0
    except BaseException as error:
        state.update(
            {
                "status": "FAILED",
                "phase": "FAILED",
                "updated_utc": _utc_now(),
                "error": {"type": type(error).__name__, "message": str(error)},
                "power_restore_authority": False,
            }
        )
        _write_atomic(status_path, state)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-preregistration", required=True, type=Path)
    parser.add_argument("--expected-master-sha256", required=True)
    parser.add_argument("--controller-pid", required=True, type=int)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(
        args.master_preregistration.expanduser(),
        str(args.expected_master_sha256),
        args.controller_pid,
        args.poll_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
