"""Safely replace a waiting AlphaZuma 55 deployment after a terminal milestone."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence


SCRIPT_PATH = Path(__file__).resolve()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON document {path}: {error}") from error
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _write_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError:
        raise FileExistsError(f"refusing to overwrite immutable switch evidence: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _bound_path(reference: Any, name: str) -> Path:
    _require(isinstance(reference, dict), f"{name} must be an object")
    path = Path(str(reference.get("path", ""))).expanduser().resolve(strict=True)
    _require(reference.get("sha256") == _sha256(path), f"{name} hash mismatch")
    return path


def _process_inventory() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        command = raw.replace(b"\0", b" ").decode("utf-8", errors="replace").strip()
        if command:
            rows.append({"pid": int(entry.name), "command": command})
    return rows


def _old_processes(
    inventory: Sequence[dict[str, Any]],
    *,
    deployer_name: str,
    deployment_name: str,
    controller_name: str,
    postprocess_name: str,
) -> dict[str, list[dict[str, Any]]]:
    return {
        "deployer": [
            dict(row)
            for row in inventory
            if deployer_name in str(row["command"])
            and deployment_name in str(row["command"])
        ],
        "controller": [
            dict(row)
            for row in inventory
            if controller_name in str(row["command"])
            and postprocess_name in str(row["command"])
        ],
    }


def _gpu_limits() -> dict[str, float]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,power.limit",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result: dict[str, float] = {}
    for line in completed.stdout.splitlines():
        index, limit = [part.strip() for part in line.split(",")]
        result[index] = float(limit)
    return result


def _active_power_plan() -> str:
    raw = subprocess.run(
        ["powercfg.exe", "/getactivescheme"],
        check=True,
        capture_output=True,
    ).stdout
    return raw.decode("ascii", errors="ignore").strip()


def _validate_preregistration(path: Path) -> dict[str, Any]:
    prereg = _read_json(path)
    _require(
        prereg.get("schema") == "zuma-rl.alphazuma-55-waiting-deployment-switch"
        and prereg.get("version") == 1
        and prereg.get("status") == "FROZEN_BEFORE_SWITCH",
        "unexpected switch preregistration",
    )
    implementation = prereg.get("implementation")
    _require(isinstance(implementation, dict), "implementation must be an object")
    orchestrator = _bound_path(implementation["orchestrator"], "switch orchestrator")
    _require(orchestrator == SCRIPT_PATH, "switch preregistration binds another orchestrator")
    for name in ("withdrawal_builder", "deployment_builder", "deployer"):
        _bound_path(implementation[name], name)

    expansion_path = _bound_path(prereg["capacity_expansion"], "capacity expansion")
    current_path = _bound_path(prereg["current_deployment"], "current deployment")
    expansion = _read_json(expansion_path)
    current = _read_json(current_path)
    _require(
        expansion.get("schema")
        == "zuma-rl.alphazuma-55-capacity-expansion-preregistration"
        and expansion.get("version") == 2
        and expansion.get("status") == "FROZEN_BEFORE_HARD_FRONTIER_TRAINING",
        "switch requires the frozen six-route expansion",
    )
    _require(
        current.get("schema")
        == "zuma-rl.alphazuma-55-postprocess-deployment-preregistration"
        and current.get("status") == "FROZEN_BEFORE_FORMAL_SEEDS",
        "unexpected current deployment",
    )
    current_reference = expansion["postprocess_supersession"]["current_deployment"]
    _require(
        Path(str(current_reference["path"])).resolve(strict=True) == current_path
        and current_reference["sha256"] == _sha256(current_path),
        "capacity expansion binds another current deployment",
    )
    outputs = prereg["outputs"]
    supersession = expansion["postprocess_supersession"]
    _require(
        Path(str(outputs["new_deployment_preregistration"])).resolve()
        == Path(str(supersession["new_deployment_preregistration"])).resolve(),
        "switch binds another new deployment output",
    )
    _require(
        Path(str(outputs["withdrawal_receipt"])).resolve()
        != Path(str(outputs["new_deployment_preregistration"])).resolve(),
        "switch output paths are duplicated",
    )
    milestone = prereg["milestone_controller"]
    _require(
        Path(str(milestone["path"])).resolve()
        == Path(
            str(
                expansion["safe_switch_gate"][
                    "wait_for_milestone_controller_to_reach_terminal_complete_or_error"
                ]
            )
        ).resolve(),
        "switch binds another milestone controller",
    )
    _require(float(prereg["poll_seconds"]) > 0.0, "poll interval must be positive")
    _require(
        float(prereg["process_policy"]["graceful_exit_timeout_seconds"]) > 0.0,
        "graceful exit timeout must be positive",
    )
    return prereg


def _prestop_evidence(
    *,
    prereg: dict[str, Any],
    inventory: Sequence[dict[str, Any]],
    gpu_limits: dict[str, float],
    active_power_plan: str,
) -> dict[str, Any]:
    expansion_path = Path(str(prereg["capacity_expansion"]["path"])).resolve(strict=True)
    current_path = Path(str(prereg["current_deployment"]["path"])).resolve(strict=True)
    expansion = _read_json(expansion_path)
    current = _read_json(current_path)
    milestone_path = Path(str(prereg["milestone_controller"]["path"])).resolve(strict=True)
    milestone = _read_json(milestone_path)
    _require(milestone.get("status") in {"COMPLETE", "ERROR"}, "milestone is not terminal")

    controller_status_path = (
        Path(str(current["formal_outputs"]["controller"])).resolve(strict=True)
        / "controller_status.json"
    )
    deployment_status_path = (
        Path(str(current["deployment_output_root"])).resolve(strict=True)
        / "deployment_status.json"
    )
    controller_status = _read_json(controller_status_path)
    _require(
        controller_status.get("phase") == "WAITING_FOR_TRAINING"
        and controller_status.get("error") is None,
        "old controller is not cleanly waiting",
    )
    old_formal_absent = not any(
        Path(str(value)).exists()
        for name, value in current["formal_outputs"].items()
        if name != "controller"
    )
    _require(old_formal_absent, "old formal roots are not absent")

    supersession = expansion["postprocess_supersession"]
    new_paths = [
        supersession["new_postprocess_preregistration"],
        supersession["new_deployment_preregistration"],
        supersession["new_independent_audit_receipt"],
        supersession["new_deployment_output_root"],
        *supersession["new_formal_outputs"].values(),
        prereg["outputs"]["withdrawal_receipt"],
    ]
    _require(
        not any(Path(str(value)).exists() for value in new_paths),
        "one or more six-route outputs already exist",
    )

    implementation = current["implementation"]
    matches = _old_processes(
        inventory,
        deployer_name=Path(str(implementation["deployer"]["path"])).name,
        deployment_name=current_path.name,
        controller_name=Path(str(implementation["controller"]["path"])).name,
        postprocess_name=Path(str(current["postprocess_preregistration"])).name,
    )
    _require(len(matches["controller"]) == 1, "expected exactly one old controller process")
    _require(len(matches["deployer"]) == 1, "expected exactly one old deployer process")

    expected_limits = prereg["power_state"]["training_gpu_limits_watts"]
    _require(
        abs(float(gpu_limits.get("0", -1.0)) - float(expected_limits["0"])) < 0.6
        and abs(float(gpu_limits.get("1", -1.0)) - float(expected_limits["1"])) < 0.6,
        "training GPU limits changed before switch",
    )
    expected_plan = str(prereg["power_state"]["windows_training_power_plan_guid"])
    _require(expected_plan.casefold() in active_power_plan.casefold(), "power plan changed before switch")
    return {
        "schema": "zuma-rl.alphazuma-55-waiting-deployment-prestop-evidence",
        "version": 1,
        "status": "PASS",
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "milestone": {
            "path": str(milestone_path),
            "sha256": _sha256(milestone_path),
            "status": milestone["status"],
        },
        "old_controller_status": {
            "path": str(controller_status_path),
            "sha256": _sha256(controller_status_path),
            "phase": controller_status["phase"],
        },
        "old_deployment_status": {
            "path": str(deployment_status_path),
            "sha256": _sha256(deployment_status_path),
        },
        "formal_boundary": {
            "old_formal_roots_absent": old_formal_absent,
            "new_six_route_outputs_absent": True,
            "formal_seed_consumption": False,
        },
        "processes": matches,
        "power_state": {
            "gpu_limits_watts": gpu_limits,
            "active_windows_power_plan": active_power_plan,
        },
    }


def _run_logged(name: str, command: Sequence[str], output_root: Path) -> None:
    stdout_path = output_root / f"{name}.stdout.log"
    stderr_path = output_root / f"{name}.stderr.log"
    _require(not stdout_path.exists() and not stderr_path.exists(), f"{name} logs already exist")
    environment = os.environ.copy()
    environment.update({"TMPDIR": "/tmp", "TEMP": "/tmp", "PYTHONUNBUFFERED": "1"})
    with stdout_path.open("x", encoding="utf-8", newline="\n") as stdout, stderr_path.open(
        "x", encoding="utf-8", newline="\n"
    ) as stderr:
        completed = subprocess.run(
            list(command),
            cwd=SCRIPT_PATH.parents[1],
            env=environment,
            stdout=stdout,
            stderr=stderr,
            text=True,
        )
    if completed.returncode:
        raise RuntimeError(f"{name} failed with exit code {completed.returncode}; stderr={stderr_path}")


class SwitchController:
    def __init__(
        self,
        preregistration_path: Path,
        preregistration: dict[str, Any],
        *,
        inventory_fn: Callable[[], list[dict[str, Any]]] = _process_inventory,
        kill_fn: Callable[[int, int], None] = os.kill,
    ) -> None:
        self.preregistration_path = preregistration_path
        self.preregistration = preregistration
        self.inventory_fn = inventory_fn
        self.kill_fn = kill_fn
        self.output_root = Path(str(preregistration["outputs"]["switch_output_root"])).resolve()
        self.status_path = self.output_root / "controller_status.json"
        self.started_utc = datetime.now(timezone.utc)

    def publish(self, phase: str, **extra: Any) -> None:
        _write_atomic(
            self.status_path,
            {
                "schema": "zuma-rl.alphazuma-55-waiting-deployment-switch-status",
                "version": 1,
                "status": "RUNNING",
                "phase": phase,
                "started_utc": self.started_utc.isoformat(),
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "preregistration": {
                    "path": str(self.preregistration_path),
                    "sha256": _sha256(self.preregistration_path),
                },
                **extra,
            },
        )

    def _matches(self) -> dict[str, list[dict[str, Any]]]:
        current_path = Path(str(self.preregistration["current_deployment"]["path"])).resolve(strict=True)
        current = _read_json(current_path)
        implementation = current["implementation"]
        return _old_processes(
            self.inventory_fn(),
            deployer_name=Path(str(implementation["deployer"]["path"])).name,
            deployment_name=current_path.name,
            controller_name=Path(str(implementation["controller"]["path"])).name,
            postprocess_name=Path(str(current["postprocess_preregistration"])).name,
        )

    def _stop_old_processes(self, evidence: dict[str, Any]) -> None:
        controller_pid = int(evidence["processes"]["controller"][0]["pid"])
        self.publish("STOPPING_OLD_CONTROLLER", controller_pid=controller_pid)
        self.kill_fn(controller_pid, signal.SIGTERM)
        timeout = float(
            self.preregistration["process_policy"]["graceful_exit_timeout_seconds"]
        )
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            matches = self._matches()
            if not matches["controller"] and not matches["deployer"]:
                return
            self.publish("WAITING_FOR_OLD_PROCESSES_TO_EXIT", matches=matches)
            time.sleep(min(2.0, float(self.preregistration["poll_seconds"])))
        matches = self._matches()
        if matches["controller"]:
            raise RuntimeError("old controller remained alive after SIGTERM")
        if len(matches["deployer"]) == 1:
            deployer_pid = int(matches["deployer"][0]["pid"])
            self.publish("STOPPING_OLD_DEPLOYER", deployer_pid=deployer_pid)
            self.kill_fn(deployer_pid, signal.SIGTERM)
            second_deadline = time.monotonic() + min(timeout, 30.0)
            while time.monotonic() < second_deadline:
                if not self._matches()["deployer"]:
                    return
                time.sleep(1.0)
        raise RuntimeError("old deployment processes did not exit cleanly")

    def execute(self) -> None:
        milestone_path = Path(str(self.preregistration["milestone_controller"]["path"])).resolve(strict=True)
        while True:
            milestone = _read_json(milestone_path)
            if milestone.get("status") in {"COMPLETE", "ERROR"}:
                break
            _require(
                milestone.get("status") == "WAITING_FOR_TARGET",
                f"unexpected milestone status: {milestone.get('status')}",
            )
            self.publish(
                "WAITING_FOR_MILESTONE",
                milestone_status=milestone.get("status"),
                milestone_updated_utc=milestone.get("updated_utc"),
                route=milestone.get("route"),
            )
            time.sleep(float(self.preregistration["poll_seconds"]))

        evidence = _prestop_evidence(
            prereg=self.preregistration,
            inventory=self.inventory_fn(),
            gpu_limits=_gpu_limits(),
            active_power_plan=_active_power_plan(),
        )
        evidence_path = self.output_root / "prestop_evidence.json"
        _write_exclusive(evidence_path, evidence)
        self._stop_old_processes(evidence)

        implementation = self.preregistration["implementation"]
        outputs = self.preregistration["outputs"]
        expansion = self.preregistration["capacity_expansion"]
        current = self.preregistration["current_deployment"]
        withdrawal_path = Path(str(outputs["withdrawal_receipt"])).resolve()
        new_deployment_path = Path(str(outputs["new_deployment_preregistration"])).resolve()

        self.publish("FREEZING_WITHDRAWAL")
        _run_logged(
            "freeze-withdrawal",
            [
                sys.executable,
                str(_bound_path(implementation["withdrawal_builder"], "withdrawal builder")),
                "--expansion-preregistration",
                str(_bound_path(expansion, "capacity expansion")),
                "--expected-expansion-sha256",
                str(expansion["sha256"]),
                "--current-deployment",
                str(_bound_path(current, "current deployment")),
                "--expected-current-deployment-sha256",
                str(current["sha256"]),
                "--output",
                str(withdrawal_path),
            ],
            self.output_root,
        )

        self.publish("FREEZING_NEW_DEPLOYMENT")
        _run_logged(
            "build-deployment",
            [
                sys.executable,
                str(_bound_path(implementation["deployment_builder"], "deployment builder")),
                "--expansion-preregistration",
                str(_bound_path(expansion, "capacity expansion")),
                "--expected-expansion-sha256",
                str(expansion["sha256"]),
                "--withdrawal-receipt",
                str(withdrawal_path),
                "--current-deployment",
                str(_bound_path(current, "current deployment")),
                "--output",
                str(new_deployment_path),
            ],
            self.output_root,
        )

        deployer = _bound_path(implementation["deployer"], "deployer")
        self.publish("VALIDATING_NEW_DEPLOYMENT")
        _run_logged(
            "validate-deployment",
            [sys.executable, str(deployer), "--preregistration", str(new_deployment_path), "--validate-only"],
            self.output_root,
        )
        handoff_path = self.output_root / "handoff.json"
        _write_exclusive(
            handoff_path,
            {
                "schema": "zuma-rl.alphazuma-55-waiting-deployment-switch-handoff",
                "version": 1,
                "status": "READY_TO_EXEC",
                "recorded_utc": datetime.now(timezone.utc).isoformat(),
                "prestop_evidence": {"path": str(evidence_path), "sha256": _sha256(evidence_path)},
                "withdrawal_receipt": {"path": str(withdrawal_path), "sha256": _sha256(withdrawal_path)},
                "new_deployment": {"path": str(new_deployment_path), "sha256": _sha256(new_deployment_path)},
            },
        )
        self.publish(
            "EXECING_NEW_DEPLOYER",
            handoff={"path": str(handoff_path), "sha256": _sha256(handoff_path)},
        )
        os.execv(
            sys.executable,
            [sys.executable, str(deployer), "--preregistration", str(new_deployment_path)],
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path = args.preregistration.expanduser().resolve(strict=True)
    prereg = _validate_preregistration(path)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "VALID",
                    "preregistration": {"path": str(path), "sha256": _sha256(path)},
                    "milestone_controller": prereg["milestone_controller"],
                    "current_deployment": prereg["current_deployment"],
                    "capacity_expansion": prereg["capacity_expansion"],
                    "outputs": prereg["outputs"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    output_root = Path(str(prereg["outputs"]["switch_output_root"])).resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to reuse switch output root: {output_root}")
    output_root.mkdir(parents=True)
    controller = SwitchController(path, prereg)
    controller.publish("STARTING")
    try:
        controller.execute()
    except BaseException as error:
        failure_path = output_root / "failure.json"
        _write_exclusive(
            failure_path,
            {
                "schema": "zuma-rl.alphazuma-55-waiting-deployment-switch-failure",
                "version": 1,
                "status": "ERROR",
                "failed_utc": datetime.now(timezone.utc).isoformat(),
                "error": {"type": type(error).__name__, "message": str(error)},
            },
        )
        _write_atomic(
            controller.status_path,
            {
                "schema": "zuma-rl.alphazuma-55-waiting-deployment-switch-status",
                "version": 1,
                "status": "ERROR",
                "phase": "ERROR",
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "failure": {"path": str(failure_path), "sha256": _sha256(failure_path)},
            },
        )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
