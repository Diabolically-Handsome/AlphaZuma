"""Run the retry-audited motor-observable checkpoint screen and full55 gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tools import run_alphazuma_55_motor_observable_postprocess_v1 as base


SCRIPT_PATH = Path(__file__).resolve()
PLAN_SCHEMA = base.PLAN_SCHEMA
SCREEN_LEVEL_IDS = base.SCREEN_LEVEL_IDS
EXPECTED_MOTOR_CHECKPOINTS = base.EXPECTED_MOTOR_CHECKPOINTS
EXPECTED_SCREEN_CANDIDATES = base.EXPECTED_SCREEN_CANDIDATES
MAXIMUM_GATE_CANDIDATES = base.MAXIMUM_GATE_CANDIDATES

_utc_now = base._utc_now
_sha256 = base._sha256
_read = base._read
_write_atomic = base._write_atomic
_write_new = base._write_new
_bound = base._bound
_parse_utc = base._parse_utc
_shard_bounds = base._shard_bounds
_candidate_inventory = base._candidate_inventory
_manifest = base._manifest
_evaluation_preregistration = base._evaluation_preregistration
_rank_key = base._rank_key
audit_matrix = base.audit_matrix


def _recovery_processes(
    process_finder: Any, fragment: str
) -> list[int]:
    if fragment == "distill_alphazuma_55_motor_observable_replay_v2.py":
        fragment = "distill_alphazuma_55_motor_observable_replay_v3.py"
    return list(process_finder(fragment))


def validate_plan(
    path: Path, expected_sha256: str | None = None
) -> dict[str, Any]:
    original_script_path = base.SCRIPT_PATH
    try:
        base.SCRIPT_PATH = SCRIPT_PATH
        plan = base.validate_plan(path, expected_sha256)
    finally:
        base.SCRIPT_PATH = original_script_path
    route = _read(_bound(plan["training_route"], "training route"))
    incident = plan.get("predecessor_postprocess_incident", {})
    if not (
        int(route.get("version", -1)) == 2
        and route.get("campaign_id")
        == "alphazuma-55-motor-observable-replay-s99081616-v3"
        and route.get("post_hoc_disclosure", {}).get(
            "formal_gate_unchanged"
        )
        is True
        and incident.get("classification")
        == "V1_WAITING_CONTROLLER_ABORTED_WITHOUT_MATRIX_INFERENCE"
        and incident.get("formal_seed_consumption") == "NONE"
    ):
        raise ValueError("retry-audited postprocess recovery contract changed")
    return plan


def run(*, plan_path: Path, expected_plan_sha256: str, poll_seconds: float) -> int:
    plan_path = plan_path.resolve(strict=True)
    plan = validate_plan(plan_path, expected_plan_sha256)
    original_validate = base.validate_plan
    original_processes = base._processes_containing

    def recovery_processes(fragment: str) -> list[int]:
        return _recovery_processes(original_processes, fragment)

    try:
        base.validate_plan = lambda _path, _hash=None: plan
        base._processes_containing = recovery_processes
        return base.run(
            plan_path=plan_path,
            expected_plan_sha256=expected_plan_sha256,
            poll_seconds=poll_seconds,
        )
    finally:
        base.validate_plan = original_validate
        base._processes_containing = original_processes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    plan_path = args.plan.expanduser().resolve(strict=True)
    if args.validate_only:
        plan = validate_plan(plan_path, str(args.expected_plan_sha256))
        print(
            json.dumps(
                {
                    "status": "VALID",
                    "plan": {
                        "path": str(plan_path),
                        "sha256": _sha256(plan_path),
                    },
                    "training_route": plan["training_route"],
                    "screen_candidates": int(
                        plan["screen"]["expected_candidates"]
                    ),
                    "screen_attempts": int(plan["screen"]["expected_attempts"]),
                    "maximum_full55_attempts": int(
                        plan["full55_gate"]["maximum_expected_attempts"]
                    ),
                    "formal_seed_consumption": False,
                },
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        )
        return 0
    return run(
        plan_path=plan_path,
        expected_plan_sha256=str(args.expected_plan_sha256),
        poll_seconds=max(1.0, float(args.poll_seconds)),
    )


if __name__ == "__main__":
    raise SystemExit(main())
