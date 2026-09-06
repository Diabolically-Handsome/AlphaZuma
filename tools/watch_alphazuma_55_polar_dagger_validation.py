"""Run frozen polar DAgger validation after training and the prior matrix."""

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

from tools.build_alphazuma_55_eval_contract import _sha256
from tools.materialize_alphazuma_55_polar_dagger_validation import (
    materialize,
    validate_plan_static,
)


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
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _pid_alive(pid: int) -> bool:
    return pid > 0 and Path(f"/proc/{pid}").exists()


def _matrix_snapshot(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    value = _read(path)
    return {
        "status": value.get("status"),
        "completed_attempts": value.get("completed_attempts"),
        "expected_attempts": value.get("expected_attempts"),
        "wall_seconds": value.get("runtime", {}).get("wall_seconds"),
        "error": value.get("error"),
    }


def run(
    *,
    plan_path: Path,
    expected_plan_sha256: str,
    trainer_pid: int,
    poll_seconds: float,
) -> dict[str, Any]:
    plan_path = plan_path.resolve(strict=True)
    plan = validate_plan_static(plan_path, expected_plan_sha256)
    watcher = plan["implementation"]["watcher"]
    if (
        Path(str(watcher["path"])).resolve() != SCRIPT_PATH
        or watcher["sha256"] != _sha256(SCRIPT_PATH)
    ):
        raise ValueError("plan binds another DAgger validation watcher")
    status_root = Path(plan["outputs"]["status_root"]).resolve()
    if status_root.exists():
        raise FileExistsError(f"DAgger validation status root exists: {status_root}")
    status_root.mkdir(parents=True)
    status_path = status_root / "controller_status.json"
    failure_path = status_root / "failure.json"
    started = time.perf_counter()

    def status(stage: str, **extra: Any) -> None:
        _write_atomic(
            status_path,
            {
                "schema": "zuma-rl.alphazuma-55-polar-dagger-validation-watcher",
                "version": 1,
                "status": "RUNNING",
                "stage": stage,
                "updated_utc": _utc_now(),
                "wall_seconds": time.perf_counter() - started,
                "controller": {
                    "path": str(SCRIPT_PATH),
                    "sha256": _sha256(SCRIPT_PATH),
                    "pid": os.getpid(),
                },
                "plan": {"path": str(plan_path), "sha256": expected_plan_sha256},
                "trainer_pid": trainer_pid,
                "formal_seed_consumption": False,
                "current_campaign_candidate_authority": False,
                **extra,
            },
        )

    completion_path = Path(plan["target"]["expected_completion"])
    target_failure_path = Path(plan["target"]["expected_failure"])
    try:
        while not completion_path.exists():
            if target_failure_path.exists():
                raise RuntimeError(
                    "polar DAgger training failed: "
                    + json.dumps(_read(target_failure_path), ensure_ascii=False)
                )
            if not _pid_alive(trainer_pid):
                raise RuntimeError(
                    "polar DAgger trainer exited without a terminal receipt"
                )
            status("WAITING_FOR_TRAINING")
            time.sleep(poll_seconds)

        prior_audit_path = Path(
            plan["wait_for_engineering_slot"]["prior_polar_audit"]
        )
        while not prior_audit_path.exists():
            status(
                "WAITING_FOR_ENGINEERING_SLOT",
                target_completion={
                    "path": str(completion_path),
                    "sha256": _sha256(completion_path),
                },
                waiting_for=str(prior_audit_path),
            )
            time.sleep(poll_seconds)
        prior_audit = _read(prior_audit_path)
        if prior_audit.get("status") != "PASS":
            raise RuntimeError("prior polar engineering audit did not PASS")

        status("MATERIALIZING_VALIDATION")
        manifest_path, preregistration_path = materialize(
            plan_path, expected_plan_sha256
        )
        evaluator_path = Path(plan["implementation"]["evaluator"]["path"])
        output_root = Path(plan["outputs"]["run_root"])
        matrix_path = output_root / "matrix-shard-00-of-01.json"
        evaluator_command = [
            sys.executable,
            str(evaluator_path),
            "--preregistration",
            str(preregistration_path),
            "--original-root",
            str(plan["original_root"]),
            "--shard-index",
            "0",
            "--models-manifest",
            str(manifest_path),
        ]
        environment = dict(os.environ)
        environment.update({"TMPDIR": "/tmp", "TEMP": "/tmp"})
        with (status_root / "evaluator.stdout.log").open(
            "x", encoding="utf-8"
        ) as stdout, (status_root / "evaluator.stderr.log").open(
            "x", encoding="utf-8"
        ) as stderr:
            process = subprocess.Popen(
                evaluator_command,
                stdout=stdout,
                stderr=stderr,
                env=environment,
            )
            while process.poll() is None:
                status(
                    "EVALUATING",
                    evaluator={"pid": process.pid, "command": evaluator_command},
                    matrix=_matrix_snapshot(matrix_path),
                )
                time.sleep(poll_seconds)
            if process.returncode != 0:
                raise RuntimeError(
                    f"DAgger evaluator exited with code {process.returncode}"
                )
        matrix = _matrix_snapshot(matrix_path)
        if (
            matrix is None
            or matrix["status"] != "COMPLETE"
            or int(matrix["completed_attempts"]) != 220
            or int(matrix["expected_attempts"]) != 220
        ):
            raise RuntimeError("DAgger validation matrix is not complete")

        auditor_path = Path(plan["implementation"]["independent_auditor"]["path"])
        audit_output = Path(plan["outputs"]["audit_receipt"])
        audit_command = [
            sys.executable,
            str(auditor_path),
            "--master-preregistration",
            str(plan["master_preregistration"]["path"]),
            "--preregistration",
            str(preregistration_path),
            "--models-manifest",
            str(manifest_path),
            "--matrix-shard",
            str(matrix_path),
            "--output",
            str(audit_output),
        ]
        status("AUDITING", matrix=matrix, audit_command=audit_command)
        with (status_root / "auditor.stdout.log").open(
            "x", encoding="utf-8"
        ) as stdout, (status_root / "auditor.stderr.log").open(
            "x", encoding="utf-8"
        ) as stderr:
            result = subprocess.run(
                audit_command,
                stdout=stdout,
                stderr=stderr,
                env=environment,
                check=False,
            )
        if result.returncode != 0:
            raise RuntimeError(f"DAgger auditor exited with code {result.returncode}")
        audit = _read(audit_output)
        if audit.get("status") != "PASS":
            raise RuntimeError("DAgger independent audit did not PASS")
        final = {
            "schema": "zuma-rl.alphazuma-55-polar-dagger-validation-watcher",
            "version": 1,
            "status": "COMPLETE",
            "stage": "COMPLETE",
            "completed_utc": _utc_now(),
            "wall_seconds": time.perf_counter() - started,
            "plan": {"path": str(plan_path), "sha256": expected_plan_sha256},
            "target_completion": {
                "path": str(completion_path),
                "sha256": _sha256(completion_path),
            },
            "matrix": {"path": str(matrix_path), **matrix},
            "audit": {
                "path": str(audit_output),
                "sha256": _sha256(audit_output),
                "status": audit["status"],
            },
            "formal_seed_consumption": False,
            "current_campaign_candidate_authority": False,
        }
        _write_atomic(status_path, final)
        return final
    except BaseException as error:
        failure = {
            "schema": "zuma-rl.alphazuma-55-polar-dagger-validation-watcher-failure",
            "version": 1,
            "status": "ERROR",
            "stage": "ERROR",
            "failed_utc": _utc_now(),
            "wall_seconds": time.perf_counter() - started,
            "error_type": type(error).__name__,
            "error": str(error),
            "plan": {"path": str(plan_path), "sha256": expected_plan_sha256},
            "formal_seed_consumption": False,
        }
        _write_atomic(failure_path, failure)
        _write_atomic(status_path, failure)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--trainer-pid", required=True, type=int)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.poll_seconds <= 0:
        raise SystemExit("poll-seconds must be positive")
    result = run(
        plan_path=args.plan.expanduser(),
        expected_plan_sha256=str(args.expected_plan_sha256),
        trainer_pid=args.trainer_pid,
        poll_seconds=args.poll_seconds,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
