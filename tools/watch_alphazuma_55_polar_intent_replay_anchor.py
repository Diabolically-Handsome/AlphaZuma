"""Materialize and supervise replay-anchored raw-intent DAgger."""

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
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tools import distill_alphazuma_55 as legacy
from tools.materialize_alphazuma_55_polar_intent_replay_anchor import (
    materialize,
    validate_plan_static,
)


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _write_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _training_snapshot(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "training_status.json"
    if not path.exists():
        return None
    value = _read(path)
    result = {
        "path": str(path),
        "sha256": legacy._sha256(path),
        "status": value.get("status"),
        "stage": value.get("stage"),
        "updated_utc": value.get("updated_utc"),
        "completed_anchor_passes": value.get("completed_anchor_passes"),
        "expected_anchor_passes": value.get("expected_anchor_passes"),
        "anchor_samples": value.get("anchor_samples"),
        "completed_student_rounds": value.get("completed_student_rounds"),
        "expected_student_rounds": value.get("expected_student_rounds"),
        "current_round": value.get("current_round"),
        "teacher_execution_probability": value.get(
            "teacher_execution_probability"
        ),
        "completed_epochs_in_round": value.get("completed_epochs_in_round"),
        "expected_epochs_in_round": value.get("expected_epochs_in_round"),
        "aggregate_samples": value.get("aggregate_samples"),
        "anchor_fraction": value.get("anchor_fraction"),
        "wall_seconds": value.get("wall_seconds"),
    }
    if isinstance(value.get("collection_progress"), dict):
        result["collection_progress"] = value["collection_progress"]
    return result


def run(
    *,
    plan_path: Path,
    expected_plan_sha256: str,
    poll_seconds: float,
) -> dict[str, Any]:
    plan_path = plan_path.resolve(strict=True)
    plan = validate_plan_static(plan_path, expected_plan_sha256)
    watcher = plan["implementation"]["watcher"]
    if (
        Path(str(watcher["path"])).resolve() != SCRIPT_PATH
        or watcher["sha256"] != legacy._sha256(SCRIPT_PATH)
    ):
        raise ValueError("plan binds another replay-anchor watcher")
    status_root = Path(str(plan["outputs"]["status_root"])).resolve()
    if status_root.exists():
        raise FileExistsError(f"replay-anchor watcher root exists: {status_root}")
    status_root.mkdir(parents=True)
    status_path = status_root / "controller_status.json"
    failure_path = status_root / "failure.json"
    started = time.perf_counter()

    def status(stage: str, **extra: Any) -> None:
        _write_atomic(
            status_path,
            {
                "schema": (
                    "zuma-rl.alphazuma-55-polar-intent-"
                    "replay-anchor-watcher"
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
                "state_distribution_semantics": (
                    "clean_teacher_replay_plus_dagger_student_states"
                ),
                "formal_seed_consumption": False,
                "current_campaign_candidate_authority": False,
                **extra,
            },
        )

    try:
        status("MATERIALIZING")
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
                    "replay-anchor trainer exited with code "
                    f"{process.returncode}: "
                    + json.dumps(detail, ensure_ascii=False)
                )
        completion = run_dir / "completion.json"
        if not completion.exists():
            raise RuntimeError("replay-anchor trainer omitted completion")
        result = _read(completion)
        if not (
            result.get("schema")
            == "zuma-rl.alphazuma-55-polar-intent-replay-anchor-completion"
            and result.get("status") == "COMPLETE"
            and result.get("policy_architecture")
            == "entity_polar_intent_wide_replay_anchor"
            and result.get("formal_seed_consumption") is False
            and result.get("formal_candidate_authority") is False
        ):
            raise RuntimeError("replay-anchor completion is not eligible")
        final = {
            "schema": (
                "zuma-rl.alphazuma-55-polar-intent-replay-anchor-watcher"
            ),
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
            "state_distribution_semantics": (
                "clean_teacher_replay_plus_dagger_student_states"
            ),
            "formal_seed_consumption": False,
            "current_campaign_candidate_authority": False,
        }
        _write_atomic(status_path, final)
        return final
    except BaseException as error:
        failure = {
            "schema": (
                "zuma-rl.alphazuma-55-polar-intent-"
                "replay-anchor-watcher-failure"
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args(argv)
    if args.poll_seconds <= 0:
        raise SystemExit("poll-seconds must be positive")
    result = run(
        plan_path=args.plan.expanduser(),
        expected_plan_sha256=str(args.expected_plan_sha256),
        poll_seconds=args.poll_seconds,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
