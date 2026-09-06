"""Wait for raw-intent DAgger, then run its frozen autonomy continuation."""

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

from tools import distill_alphazuma_55 as legacy
from tools.materialize_alphazuma_55_polar_intent_autonomy import (
    materialize,
    validate_plan_static,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
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


def _training_snapshot(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "training_status.json"
    if not path.exists():
        return None
    value = _read(path)
    result = {
        "path": str(path),
        "status": value.get("status"),
        "stage": value.get("stage"),
        "updated_utc": value.get("updated_utc"),
        "completed_rounds": value.get("completed_rounds"),
        "expected_rounds": value.get("expected_rounds"),
        "aggregate_samples": value.get("aggregate_samples"),
        "current_round": value.get("current_round"),
        "teacher_execution_probability": value.get(
            "teacher_execution_probability"
        ),
        "completed_epochs_in_round": value.get("completed_epochs_in_round"),
        "expected_epochs_in_round": value.get("expected_epochs_in_round"),
        "wall_seconds": value.get("wall_seconds"),
    }
    if isinstance(value.get("collection_progress"), dict):
        result["collection_progress"] = value["collection_progress"]
    return result


def run(
    *,
    plan_path: Path,
    expected_plan_sha256: str,
    source_controller_pid: int,
    poll_seconds: float,
) -> dict[str, Any]:
    plan_path = plan_path.resolve(strict=True)
    plan = validate_plan_static(plan_path, expected_plan_sha256)
    watcher = plan["implementation"]["watcher"]
    if (
        Path(str(watcher["path"])).resolve() != SCRIPT_PATH
        or watcher["sha256"] != legacy._sha256(SCRIPT_PATH)
    ):
        raise ValueError("plan binds another autonomy continuation watcher")
    status_root = Path(str(plan["outputs"]["status_root"])).resolve()
    if status_root.exists():
        raise FileExistsError(f"autonomy watcher root exists: {status_root}")
    status_root.mkdir(parents=True)
    status_path = status_root / "controller_status.json"
    failure_path = status_root / "failure.json"
    started = time.perf_counter()

    def status(stage: str, **extra: Any) -> None:
        _write_atomic(
            status_path,
            {
                "schema": (
                    "zuma-rl.alphazuma-55-polar-intent-autonomy-watcher"
                ),
                "version": 1,
                "status": "RUNNING",
                "stage": stage,
                "updated_utc": _utc_now(),
                "wall_seconds": time.perf_counter() - started,
                "controller": {
                    "path": str(SCRIPT_PATH),
                    "sha256": legacy._sha256(SCRIPT_PATH),
                    "pid": os.getpid(),
                },
                "plan": {
                    "path": str(plan_path),
                    "sha256": expected_plan_sha256,
                },
                "source_controller_pid": source_controller_pid,
                "training_label_semantics": "raw_teacher_intent",
                "state_distribution_semantics": "student_only_dagger",
                "formal_seed_consumption": False,
                "current_campaign_candidate_authority": False,
                **extra,
            },
        )

    source = plan["source"]
    completion_path = Path(str(source["expected_completion"]))
    source_failure_path = Path(str(source["expected_failure"]))
    try:
        while not completion_path.exists():
            if source_failure_path.exists():
                raise RuntimeError(
                    "source DAgger training failed: "
                    + json.dumps(_read(source_failure_path), ensure_ascii=False)
                )
            if not _pid_alive(source_controller_pid):
                raise RuntimeError(
                    "source DAgger controller exited without a terminal receipt"
                )
            status("WAITING_FOR_SOURCE_DAGGER")
            time.sleep(poll_seconds)

        status(
            "MATERIALIZING",
            source_completion={
                "path": str(completion_path),
                "sha256": legacy._sha256(completion_path),
            },
        )
        preregistration = materialize(plan_path, expected_plan_sha256)
        trainer = Path(str(plan["implementation"]["trainer"]["path"]))
        run_dir = Path(str(plan["outputs"]["run_dir"]))
        command = [
            sys.executable,
            str(trainer),
            "--preregistration",
            str(preregistration),
            "--original-root",
            str(plan["original_root"]),
        ]
        environment = dict(os.environ)
        environment.update(
            {
                "TMPDIR": "/tmp",
                "TEMP": "/tmp",
                "PYTHONPATH": str(PROJECT_ROOT / "src"),
                "PYTHONUNBUFFERED": "1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            }
        )
        with (status_root / "trainer.stdout.log").open(
            "x", encoding="utf-8", newline="\n"
        ) as stdout, (status_root / "trainer.stderr.log").open(
            "x", encoding="utf-8", newline="\n"
        ) as stderr:
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdout=stdout,
                stderr=stderr,
                env=environment,
                text=True,
            )
            while process.poll() is None:
                status(
                    "TRAINING",
                    trainer={"pid": process.pid, "command": command},
                    training=_training_snapshot(run_dir),
                )
                time.sleep(poll_seconds)
            if process.returncode != 0:
                failure = run_dir / "failure.json"
                detail = _read(failure) if failure.exists() else None
                raise RuntimeError(
                    "autonomy trainer exited with code "
                    f"{process.returncode}: "
                    + json.dumps(detail, ensure_ascii=False)
                )
        completion = run_dir / "completion.json"
        if not completion.exists():
            raise RuntimeError("autonomy trainer omitted completion receipt")
        result = _read(completion)
        if not (
            result.get("schema")
            == "zuma-rl.alphazuma-55-polar-intent-autonomy-completion"
            and result.get("status") == "COMPLETE"
            and result.get("policy_architecture")
            == "entity_polar_intent_wide_dagger_autonomy"
            and result.get("training_label_semantics")
            == "raw_teacher_intent"
            and result.get("state_distribution_semantics")
            == "student_only_dagger"
            and result.get("formal_seed_consumption") is False
            and result.get("formal_candidate_authority") is False
        ):
            raise RuntimeError("autonomy completion is not eligible")
        final = {
            "schema": "zuma-rl.alphazuma-55-polar-intent-autonomy-watcher",
            "version": 1,
            "status": "COMPLETE",
            "stage": "COMPLETE",
            "completed_utc": _utc_now(),
            "wall_seconds": time.perf_counter() - started,
            "plan": {"path": str(plan_path), "sha256": expected_plan_sha256},
            "preregistration": {
                "path": str(preregistration),
                "sha256": legacy._sha256(preregistration),
            },
            "completion": {
                "path": str(completion),
                "sha256": legacy._sha256(completion),
            },
            "final_model": result["final_model"],
            "training_label_semantics": "raw_teacher_intent",
            "state_distribution_semantics": "student_only_dagger",
            "formal_seed_consumption": False,
            "current_campaign_candidate_authority": False,
        }
        _write_atomic(status_path, final)
        return final
    except BaseException as error:
        failure = {
            "schema": (
                "zuma-rl.alphazuma-55-polar-intent-autonomy-watcher-failure"
            ),
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
    parser.add_argument("--source-controller-pid", required=True, type=int)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.poll_seconds <= 0:
        raise SystemExit("poll-seconds must be positive")
    result = run(
        plan_path=args.plan.expanduser(),
        expected_plan_sha256=str(args.expected_plan_sha256),
        source_controller_pid=args.source_controller_pid,
        poll_seconds=args.poll_seconds,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
