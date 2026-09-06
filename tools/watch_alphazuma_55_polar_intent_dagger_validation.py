"""Wait for raw-intent DAgger terminal state, then run frozen validation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools.build_alphazuma_55_eval_contract import _sha256
from tools import materialize_alphazuma_55_polar_intent_dagger_validation as binding
from tools import watch_alphazuma_55_polar_dagger_validation as legacy


SCRIPT_PATH = Path(__file__).resolve()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_wait_failure(
    *, plan: dict[str, Any], plan_path: Path, plan_sha: str, error: BaseException
) -> None:
    status_root = Path(plan["outputs"]["status_root"]).resolve()
    status_root.mkdir(parents=True, exist_ok=True)
    value = {
        "schema": (
            "zuma-rl.alphazuma-55-polar-intent-dagger-"
            "validation-watcher-failure"
        ),
        "version": 1,
        "status": "ERROR",
        "stage": "WAITING_FOR_TRAINING_ERROR",
        "failed_utc": _utc_now(),
        "error_type": type(error).__name__,
        "error": str(error),
        "plan": {"path": str(plan_path), "sha256": plan_sha},
        "formal_seed_consumption": False,
    }
    legacy._write_atomic(status_root / "failure.json", value)
    legacy._write_atomic(status_root / "controller_status.json", value)


def run(
    *, plan_path: Path, expected_plan_sha256: str, poll_seconds: float
) -> dict[str, Any]:
    plan_path = plan_path.resolve(strict=True)
    plan = binding.validate_plan_static(plan_path, expected_plan_sha256)
    watcher = plan["implementation"]["watcher"]
    if (
        Path(str(watcher["path"])).resolve() != SCRIPT_PATH
        or watcher["sha256"] != _sha256(SCRIPT_PATH)
    ):
        raise ValueError("plan binds another raw-intent DAgger validation watcher")
    launch_status_path = Path(plan["training_launch_controller"]["status"])
    try:
        while True:
            if launch_status_path.exists():
                launch_status = legacy._read(launch_status_path)
                terminal = str(launch_status.get("status", ""))
                if terminal == "COMPLETE":
                    break
                if terminal in {"FAILED", "ERROR"}:
                    raise RuntimeError(
                        "raw-intent DAgger launch controller failed: "
                        + json.dumps(launch_status, ensure_ascii=False)
                    )
            time.sleep(poll_seconds)
    except BaseException as error:
        _write_wait_failure(
            plan=plan,
            plan_path=plan_path,
            plan_sha=expected_plan_sha256,
            error=error,
        )
        raise

    original_script = legacy.SCRIPT_PATH
    original_validate = legacy.validate_plan_static
    original_materialize = legacy.materialize
    try:
        legacy.SCRIPT_PATH = SCRIPT_PATH
        legacy.validate_plan_static = binding.validate_plan_static
        legacy.materialize = binding.materialize
        result = legacy.run(
            plan_path=plan_path,
            expected_plan_sha256=expected_plan_sha256,
            trainer_pid=-1,
            poll_seconds=poll_seconds,
        )
    finally:
        legacy.SCRIPT_PATH = original_script
        legacy.validate_plan_static = original_validate
        legacy.materialize = original_materialize
    result.update(
        {
            "schema": (
                "zuma-rl.alphazuma-55-polar-intent-dagger-"
                "validation-watcher"
            ),
            "training_label_semantics": "raw_teacher_intent",
            "state_distribution_semantics": "dagger_teacher_student_mixture",
        }
    )
    status_path = Path(plan["outputs"]["status_root"]) / "controller_status.json"
    legacy._write_atomic(status_path, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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
