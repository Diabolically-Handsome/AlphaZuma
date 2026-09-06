"""Wait for the intent trainer wrapper to exit, then run frozen validation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Any

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools.build_alphazuma_55_eval_contract import _sha256
from tools.materialize_alphazuma_55_polar_intent_wide_validation import (
    validate_plan_static,
)
from tools import watch_alphazuma_55_polar_intent_wide_validation as legacy


SCRIPT_PATH = Path(__file__).resolve()


def run(
    *, plan_path: Path, expected_plan_sha256: str, poll_seconds: float
) -> dict[str, Any]:
    plan_path = plan_path.resolve(strict=True)
    plan = validate_plan_static(plan_path, expected_plan_sha256)
    watcher = plan["implementation"]["watcher"]
    if (
        Path(str(watcher["path"])).resolve() != SCRIPT_PATH
        or watcher["sha256"] != _sha256(SCRIPT_PATH)
    ):
        raise ValueError("plan binds another V2 watcher")
    status_root = Path(plan["outputs"]["status_root"]).resolve()
    if status_root.exists():
        raise FileExistsError(
            f"intent-wide V2 validation status root exists: {status_root}"
        )
    launch_status_path = Path(plan["launch_controller"]["status"])
    while True:
        if launch_status_path.exists():
            launch_status = legacy._read(launch_status_path)
            if launch_status.get("status") in {"COMPLETE", "FAILED"}:
                break
        time.sleep(poll_seconds)

    original_script_path = legacy.SCRIPT_PATH
    try:
        # The legacy body is otherwise correct.  Delaying entry until the parent
        # launch controller is terminal closes the transient completion-schema
        # window created while the intent wrapper rewrites wide.run's receipt.
        legacy.SCRIPT_PATH = SCRIPT_PATH
        return legacy.run(
            plan_path=plan_path,
            expected_plan_sha256=expected_plan_sha256,
            poll_seconds=poll_seconds,
        )
    finally:
        legacy.SCRIPT_PATH = original_script_path


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
    print(legacy.json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
