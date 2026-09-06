"""Wait for one frozen AlphaZuma 55 controller, then run its independent audit."""

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
from typing import Any, Mapping


SCRIPT_PATH = Path(__file__).resolve()


class WatchError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise WatchError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def _artifact(value: Any, name: str) -> Path:
    _require(isinstance(value, dict), f"{name} reference is not an object")
    path = Path(str(value.get("path", ""))).expanduser().resolve(strict=True)
    _require(value.get("sha256") == _sha256(path), f"{name} hash differs")
    return path


def _utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    _require(parsed.tzinfo is not None, "deadline lacks timezone")
    return parsed.astimezone(timezone.utc)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _write_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _validate_plan(path: Path, expected_sha256: str) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    _require(_sha256(path) == expected_sha256, "watch plan hash differs")
    plan = _read(path)
    _require(
        plan.get("schema") == "zuma-rl.alphazuma-55-independent-audit-watcher-plan"
        and plan.get("version") == 1
        and plan.get("status") == "FROZEN_BEFORE_AUDIT",
        "watch plan identity differs",
    )
    _require(_artifact(plan.get("watcher"), "watcher") == SCRIPT_PATH, "watcher path differs")
    _artifact(plan.get("preregistration"), "preregistration")
    _artifact(plan.get("auditor"), "auditor")
    controller_status = Path(str(plan["controller_status"])).expanduser().resolve()
    receipt = Path(str(plan["independent_audit_receipt"])).expanduser().resolve()
    status_root = Path(str(plan["status_root"])).expanduser().resolve()
    _require(controller_status != receipt, "controller status and receipt collide")
    _require(status_root not in {controller_status.parent, receipt.parent}, "status root collides with formal outputs")
    _require(float(plan["poll_seconds"]) > 0, "poll interval is not positive")
    _utc(str(plan["deadline_utc"]))
    return plan


def _next_log_paths(root: Path) -> tuple[Path, Path]:
    for attempt in range(1, 1000):
        stdout = root / f"independent-audit-attempt-{attempt:02d}.stdout.log"
        stderr = root / f"independent-audit-attempt-{attempt:02d}.stderr.log"
        if not stdout.exists() and not stderr.exists():
            return stdout, stderr
    raise WatchError("independent-audit log namespace is exhausted")


def run(plan_path: Path, expected_sha256: str, *, validate_only: bool = False) -> int:
    plan_path = plan_path.expanduser().resolve(strict=True)
    plan = _validate_plan(plan_path, expected_sha256)
    if validate_only:
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "plan": {"path": str(plan_path), "sha256": _sha256(plan_path)},
                    "controller_status": str(plan["controller_status"]),
                    "independent_audit_receipt": str(plan["independent_audit_receipt"]),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    status_root = Path(str(plan["status_root"])).expanduser().resolve()
    status_root.mkdir(parents=True, exist_ok=True)
    status_path = status_root / "watcher_status.json"
    controller_path = Path(str(plan["controller_status"])).expanduser().resolve()
    receipt_path = Path(str(plan["independent_audit_receipt"])).expanduser().resolve()
    preregistration = _artifact(plan.get("preregistration"), "preregistration")
    auditor = _artifact(plan.get("auditor"), "auditor")
    deadline = _utc(str(plan["deadline_utc"]))
    poll_seconds = float(plan["poll_seconds"])
    state: dict[str, Any] = {
        "schema": "zuma-rl.alphazuma-55-independent-audit-watcher-status",
        "version": 1,
        "status": "RUNNING",
        "phase": "WAITING_FOR_CONTROLLER",
        "started_utc": _now().isoformat(),
        "updated_utc": _now().isoformat(),
        "plan": {"path": str(plan_path), "sha256": _sha256(plan_path)},
        "controller_status": str(controller_path),
        "independent_audit_receipt": str(receipt_path),
        "deadline_utc": deadline.isoformat(),
        "error": None,
    }
    _write_atomic(status_path, state)
    try:
        while True:
            now = _now()
            if controller_path.exists():
                controller = _read(controller_path)
                state.update(
                    {
                        "updated_utc": now.isoformat(),
                        "observed_controller_status": controller.get("status"),
                        "observed_controller_phase": controller.get("phase"),
                        "observed_controller_updated_utc": controller.get("updated_utc"),
                    }
                )
                if controller.get("status") == "COMPLETE" and controller.get("phase") == "COMPLETE":
                    state["phase"] = "RUNNING_INDEPENDENT_AUDIT"
                    _write_atomic(status_path, state)
                    stdout_path, stderr_path = _next_log_paths(status_root)
                    environment = os.environ.copy()
                    environment.update({"TMPDIR": "/tmp", "TEMP": "/tmp", "PYTHONUNBUFFERED": "1"})
                    with stdout_path.open("x", encoding="utf-8", newline="\n") as stdout, stderr_path.open(
                        "x", encoding="utf-8", newline="\n"
                    ) as stderr:
                        completed = subprocess.run(
                            [
                                sys.executable,
                                str(auditor),
                                "--preregistration",
                                str(preregistration),
                                "--receipt",
                                str(receipt_path),
                            ],
                            cwd=SCRIPT_PATH.parents[1],
                            env=environment,
                            stdout=stdout,
                            stderr=stderr,
                            text=True,
                        )
                    _require(completed.returncode == 0, f"independent auditor exited {completed.returncode}")
                    receipt = _read(receipt_path)
                    _require(receipt.get("status") == "PASS", "independent audit receipt is not PASS")
                    state.update(
                        {
                            "status": "COMPLETE",
                            "phase": "COMPLETE",
                            "updated_utc": _now().isoformat(),
                            "audit_stdout": str(stdout_path),
                            "audit_stderr": str(stderr_path),
                            "receipt": {"path": str(receipt_path), "sha256": _sha256(receipt_path)},
                        }
                    )
                    _write_atomic(status_path, state)
                    return 0
                if controller.get("status") in {"FAILED", "ERROR"} or controller.get("phase") in {"FAILED", "ERROR"}:
                    raise WatchError(f"controller terminated as {controller.get('status')}/{controller.get('phase')}")
                _write_atomic(status_path, state)
            if now >= deadline:
                raise TimeoutError("controller did not complete before the frozen goal deadline")
            remaining = max(0.0, (deadline - _now()).total_seconds())
            time.sleep(min(poll_seconds, remaining) if remaining else 0.1)
    except BaseException as error:
        state.update(
            {
                "status": "FAILED",
                "phase": "FAILED",
                "updated_utc": _now().isoformat(),
                "error": {"type": type(error).__name__, "message": str(error)},
            }
        )
        _write_atomic(status_path, state)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args.plan, str(args.expected_plan_sha256), validate_only=bool(args.validate_only))


if __name__ == "__main__":
    raise SystemExit(main())
