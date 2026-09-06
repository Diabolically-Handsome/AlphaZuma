"""Wait for the distilled route, then deploy and finalize AlphaZuma 55."""

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
from typing import Any, Mapping, Sequence


SCRIPT_PATH = Path(__file__).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
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


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _artifact(reference: Mapping[str, Any], name: str) -> Path:
    path = Path(str(reference.get("path", ""))).expanduser().resolve(strict=True)
    if reference.get("sha256") != _sha256(path):
        raise ValueError(f"{name} hash differs")
    return path


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"UTC timestamp lacks timezone: {value}")
    return parsed.astimezone(timezone.utc)


def _windows_path(path: Path) -> str:
    completed = subprocess.run(
        ["wslpath", "-w", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


class Deployment:
    def __init__(self, preregistration_path: Path, preregistration: dict[str, Any]):
        self.preregistration_path = preregistration_path
        self.preregistration = preregistration
        self.output_root = Path(str(preregistration["deployment_output_root"])).resolve()
        self.status_path = self.output_root / "deployment_status.json"
        self.started_utc = datetime.now(timezone.utc)

    def publish(self, stage: str, **extra: Any) -> None:
        value = {
            "schema": "zuma-rl.alphazuma-55-postprocess-deployment-status",
            "version": 1,
            "status": "RUNNING",
            "stage": stage,
            "started_utc": self.started_utc.isoformat(),
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": (datetime.now(timezone.utc) - self.started_utc).total_seconds(),
            **extra,
        }
        _write_atomic(self.status_path, value)

    def run_logged(
        self,
        name: str,
        command: Sequence[str],
        *,
        stage: str,
        watched_receipt: Path | None = None,
    ) -> None:
        stdout_path = self.output_root / f"{name}.stdout.log"
        stderr_path = self.output_root / f"{name}.stderr.log"
        for path in (stdout_path, stderr_path):
            if path.exists():
                raise FileExistsError(f"refusing to overwrite deployment log: {path}")
        environment = os.environ.copy()
        environment.update(
            {
                "TMPDIR": "/tmp",
                "TEMP": "/tmp",
                "PYTHONUNBUFFERED": "1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            }
        )
        with stdout_path.open("x", encoding="utf-8", newline="\n") as stdout, stderr_path.open(
            "x", encoding="utf-8", newline="\n"
        ) as stderr:
            process = subprocess.Popen(
                list(command),
                cwd=Path(__file__).resolve().parents[1],
                env=environment,
                stdout=stdout,
                stderr=stderr,
                text=True,
            )
            while process.poll() is None:
                watched = None
                if watched_receipt is not None and watched_receipt.exists():
                    try:
                        watched = _read_json(watched_receipt)
                    except (OSError, json.JSONDecodeError, ValueError):
                        watched = {"status": "TRANSIENT_UNREADABLE"}
                self.publish(
                    stage,
                    child_pid=process.pid,
                    watched_receipt={
                        "path": str(watched_receipt) if watched_receipt is not None else None,
                        "value": watched,
                    },
                )
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    pass
        if process.returncode:
            raise RuntimeError(
                f"{name} failed with exit code {process.returncode}; stderr={stderr_path}"
            )

    def request_restore(self) -> dict[str, Any]:
        restore = _mapping(self.preregistration["restore"], "restore")
        gpu_script = _artifact(_mapping(restore["gpu_guard"], "gpu guard"), "gpu guard")
        plan_script = _artifact(_mapping(restore["power_plan_guard"], "power plan guard"), "power plan guard")
        for script in (gpu_script, plan_script):
            subprocess.run(
                [
                    "powershell.exe",
                    "-NoLogo",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    _windows_path(script),
                    "-Mode",
                    "RequestRestore",
                ],
                check=True,
            )
        gpu_receipt = Path(str(restore["gpu_receipt"])).resolve()
        plan_receipt = Path(str(restore["power_plan_receipt"])).resolve()
        accepted_gpu = {"RESTORED_ON_REQUEST", "RESTORED_AT_DEADLINE"}
        accepted_plan = {"RESTORED_ON_REQUEST", "RESTORED_AT_DEADLINE"}
        deadline = time.monotonic() + 120.0
        gpu_value = None
        plan_value = None
        while time.monotonic() < deadline:
            if gpu_receipt.exists():
                gpu_value = _read_json(gpu_receipt)
            if plan_receipt.exists():
                plan_value = _read_json(plan_receipt)
            if (
                gpu_value is not None
                and gpu_value.get("status") in accepted_gpu
                and plan_value is not None
                and plan_value.get("status") in accepted_plan
            ):
                break
            time.sleep(5)
        if gpu_value is None or gpu_value.get("status") not in accepted_gpu:
            raise RuntimeError("GPU power guard did not confirm restoration")
        if plan_value is None or plan_value.get("status") not in accepted_plan:
            raise RuntimeError("Windows power-plan guard did not confirm restoration")
        active_bytes = subprocess.run(
            ["powercfg.exe", "/getactivescheme"],
            check=True,
            capture_output=True,
        ).stdout
        # Chinese Windows emits this legacy CLI output in the active OEM code
        # page.  The GUID is ASCII, so ignore non-ASCII bytes instead of asking
        # the WSL UTF-8 locale to decode the localized description.
        active = active_bytes.decode("ascii", errors="ignore")
        if str(restore["balanced_guid"]).casefold() not in active.casefold():
            raise RuntimeError("Windows power plan is not Balanced after restoration")
        return {
            "gpu_guard": {"path": str(gpu_receipt), "sha256": _sha256(gpu_receipt), "value": gpu_value},
            "power_plan_guard": {"path": str(plan_receipt), "sha256": _sha256(plan_receipt), "value": plan_value},
            "powercfg_getactivescheme": active.strip(),
        }

    def execute(self) -> dict[str, Any]:
        prereg = self.preregistration
        expected_route = _mapping(prereg["expected_distilled_route"], "expected distilled route")
        route_path = Path(str(expected_route["path"])).resolve()
        wait_deadline = _parse_utc(str(prereg["route_wait_deadline_utc"]))
        while not route_path.exists():
            self.publish("WAITING_FOR_DISTILLED_ROUTE", expected_path=str(route_path))
            if datetime.now(timezone.utc) > wait_deadline:
                raise TimeoutError("distilled PPO route did not freeze before its deadline")
            time.sleep(float(prereg["poll_seconds"]))
        route = _read_json(route_path)
        runs = route.get("runs")
        source_completion_path = Path(str(expected_route["source_completion_path"])).resolve(strict=True)
        source_completion = _read_json(source_completion_path)
        source_model = _mapping(source_completion.get("final_model"), "distillation final model")
        route_initial_model = _mapping(route.get("initial_model"), "distilled route initial model")
        if (
            route.get("schema") != "zuma-rl.overnight-multilevel-preregistration"
            or route.get("status") != "FROZEN_BEFORE_TRAINING"
            or not isinstance(runs, list)
            or len(runs) != 1
            or runs[0].get("id") != expected_route["route_id"]
            or int(runs[0].get("episode_seed_base", -1)) != int(expected_route["episode_seed_first"])
            or int(runs[0].get("episode_seed_last", -1)) != int(expected_route["episode_seed_last"])
            or source_completion.get("status") != "COMPLETE"
            or route_initial_model.get("path") != source_model.get("path")
            or route_initial_model.get("sha256") != source_model.get("sha256")
            or source_model.get("sha256")
            != _sha256(Path(str(source_model.get("path", ""))).resolve(strict=True))
        ):
            raise ValueError("distilled route differs from the frozen deployment boundary")

        implementation = _mapping(prereg["implementation"], "implementation")
        builder = _artifact(_mapping(implementation["builder"], "builder"), "builder")
        controller = _artifact(_mapping(implementation["controller"], "controller"), "controller")
        auditor = _artifact(_mapping(implementation["auditor"], "auditor"), "auditor")
        builder_output = Path(str(prereg["postprocess_preregistration"])).resolve()
        command = [
            sys.executable,
            str(builder),
            "--master-preregistration",
            str(_artifact(_mapping(prereg["master_preregistration"], "master"), "master")),
            "--original-root",
            str(Path(str(prereg["original_root"])).resolve(strict=True)),
        ]
        for migration in prereg["migrations"]:
            command.extend(["--migration", str(migration)])
        for route_reference in prereg["fixed_routes"]:
            command.extend(
                ["--training-route", str(_artifact(_mapping(route_reference, "fixed route"), "fixed route"))]
            )
        command.extend(["--training-route", str(route_path)])
        outputs = _mapping(prereg["formal_outputs"], "formal outputs")
        command.extend(
            [
                "--controller-root",
                str(outputs["controller"]),
                "--selection-root",
                str(outputs["selection"]),
                "--final-blind-root",
                str(outputs["final_blind"]),
                "--continuous-root",
                str(outputs["continuous"]),
                "--output",
                str(builder_output),
            ]
        )
        self.run_logged("freeze-postprocess", command, stage="FREEZING_POSTPROCESS")
        self.run_logged(
            "validate-postprocess",
            [
                sys.executable,
                str(controller),
                "--preregistration",
                str(builder_output),
                "--original-root",
                str(Path(str(prereg["original_root"])).resolve(strict=True)),
                "--validate-only",
            ],
            stage="VALIDATING_POSTPROCESS",
        )
        controller_status = Path(str(outputs["controller"])).resolve() / "controller_status.json"
        self.run_logged(
            "run-postprocess",
            [
                sys.executable,
                str(controller),
                "--preregistration",
                str(builder_output),
                "--original-root",
                str(Path(str(prereg["original_root"])).resolve(strict=True)),
            ],
            stage="RUNNING_POSTPROCESS",
            watched_receipt=controller_status,
        )
        audit_receipt = Path(str(prereg["independent_audit_receipt"])).resolve()
        self.run_logged(
            "independent-audit",
            [
                sys.executable,
                str(auditor),
                "--preregistration",
                str(builder_output),
                "--receipt",
                str(audit_receipt),
            ],
            stage="INDEPENDENT_AUDIT",
            watched_receipt=audit_receipt,
        )
        audit = _read_json(audit_receipt)
        if audit.get("status") != "PASS":
            raise RuntimeError("independent audit is not PASS")
        self.publish("RESTORING_POWER")
        restoration = self.request_restore()
        return {
            "schema": "zuma-rl.alphazuma-55-postprocess-deployment-completion",
            "version": 1,
            "status": "COMPLETE",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "distilled_route": {"path": str(route_path), "sha256": _sha256(route_path)},
            "postprocess_preregistration": {"path": str(builder_output), "sha256": _sha256(builder_output)},
            "controller_status": {"path": str(controller_status), "sha256": _sha256(controller_status)},
            "independent_audit": {"path": str(audit_receipt), "sha256": _sha256(audit_receipt)},
            "restoration": restoration,
        }


def _validate_preregistration(path: Path) -> dict[str, Any]:
    prereg = _read_json(path)
    if (
        prereg.get("schema")
        != "zuma-rl.alphazuma-55-postprocess-deployment-preregistration"
        or prereg.get("version") != 1
        or prereg.get("status") != "FROZEN_BEFORE_FORMAL_SEEDS"
    ):
        raise ValueError("unexpected deployment preregistration")
    implementation = _mapping(prereg["implementation"], "implementation")
    if _artifact(_mapping(implementation["deployer"], "deployer"), "deployer") != SCRIPT_PATH.resolve(strict=True):
        raise ValueError("deployment script path differs")
    for name, reference in implementation.items():
        _artifact(_mapping(reference, f"implementation {name}"), f"implementation {name}")
    _artifact(_mapping(prereg["teacher_pipeline_preregistration"], "teacher pipeline"), "teacher pipeline")
    _artifact(_mapping(prereg["supersedes"], "supersedes"), "withdrawal receipt")
    _artifact(
        _mapping(prereg["parallelism_decision"], "parallelism decision"),
        "parallelism decision",
    )
    return prereg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
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
                    "expected_distilled_route": prereg["expected_distilled_route"],
                    "fixed_routes": len(prereg["fixed_routes"]),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    output_root = Path(str(prereg["deployment_output_root"])).resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to reuse deployment output root: {output_root}")
    output_root.mkdir(parents=True)
    deployment = Deployment(path, prereg)
    deployment.publish("STARTING")
    try:
        completion = deployment.execute()
        completion_path = output_root / "completion.json"
        _write_atomic(completion_path, completion)
        _write_atomic(
            deployment.status_path,
            {
                "schema": "zuma-rl.alphazuma-55-postprocess-deployment-status",
                "version": 1,
                "status": "COMPLETE",
                "stage": "COMPLETE",
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "completion": {"path": str(completion_path), "sha256": _sha256(completion_path)},
            },
        )
        return 0
    except BaseException as error:
        failure = {
            "schema": "zuma-rl.alphazuma-55-postprocess-deployment-failure",
            "version": 1,
            "status": "FAILED",
            "failed_utc": datetime.now(timezone.utc).isoformat(),
            "error": {"type": type(error).__name__, "message": str(error)},
        }
        failure_path = output_root / "failure.json"
        _write_atomic(failure_path, failure)
        _write_atomic(
            deployment.status_path,
            {
                "schema": "zuma-rl.alphazuma-55-postprocess-deployment-status",
                "version": 1,
                "status": "FAILED",
                "stage": "FAILED",
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "failure": {"path": str(failure_path), "sha256": _sha256(failure_path)},
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
