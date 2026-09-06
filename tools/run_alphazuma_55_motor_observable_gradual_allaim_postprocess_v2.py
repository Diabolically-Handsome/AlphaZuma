"""Run all-actions postprocessing with canonical timeout support."""

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

from tools import (
    audit_alphazuma_55_motor_observable_timeout_recovery_v1 as timeout_reader,
)
from tools import (
    run_alphazuma_55_motor_observable_gradual_allaim_postprocess_v1 as legacy,
)


SCRIPT_PATH = Path(__file__).resolve()
MAXIMUM_GATE_CANDIDATES = legacy.base.MAXIMUM_GATE_CANDIDATES
audit_matrix = timeout_reader.audit_matrix_timeout_compatible
_read = legacy.base._read
_sha256 = legacy.base._sha256
_utc_now = legacy.base._utc_now
_write_new = legacy.base._write_new


def validate_plan(
    path: Path, expected_sha256: str | None = None
) -> dict[str, Any]:
    original_script = legacy.SCRIPT_PATH
    try:
        legacy.SCRIPT_PATH = SCRIPT_PATH
        plan = legacy.validate_plan(path, expected_sha256)
    finally:
        legacy.SCRIPT_PATH = original_script
    correction = plan.get("timeout_reader_correction", {})
    auditor_ref = plan.get("implementation", {}).get(
        "timeout_compatible_matrix_auditor", {}
    )
    auditor_path = Path(str(auditor_ref.get("path", ""))).resolve(strict=True)
    if not (
        correction.get("classification")
        == "PRE_EVALUATION_READER_CONTRACT_CORRECTION"
        and correction.get("inference_already_consumed") is False
        and correction.get("seed_ranges_changed") is False
        and correction.get("promotion_gate_changed") is False
        and correction.get("canonical_timeout_tuple")
        == {
            "outcome": None,
            "time_limit_truncated": True,
            "ticks_equals_max_ticks": True,
        }
        and auditor_path == timeout_reader.SCRIPT_PATH.resolve(strict=True)
        and auditor_ref.get("sha256") == _sha256(auditor_path)
    ):
        raise ValueError("all-actions timeout-reader correction differs")
    return plan


def run(*, plan_path: Path, expected_plan_sha256: str, poll_seconds: float) -> int:
    plan_path = plan_path.resolve(strict=True)
    plan = validate_plan(plan_path, expected_plan_sha256)
    original_script = legacy.SCRIPT_PATH
    original_validate = legacy.validate_plan
    original_audit = legacy.base.base.audit_matrix
    try:
        legacy.SCRIPT_PATH = SCRIPT_PATH
        legacy.validate_plan = lambda _path, _hash=None: plan
        legacy.base.base.audit_matrix = audit_matrix
        return legacy.run(
            plan_path=plan_path,
            expected_plan_sha256=expected_plan_sha256,
            poll_seconds=poll_seconds,
        )
    finally:
        legacy.base.base.audit_matrix = original_audit
        legacy.validate_plan = original_validate
        legacy.SCRIPT_PATH = original_script


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    plan_path = args.plan.expanduser().resolve(strict=True)
    plan = validate_plan(plan_path, str(args.expected_plan_sha256))
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "VALID",
                    "plan": {
                        "path": str(plan_path),
                        "sha256": _sha256(plan_path),
                    },
                    "screen_candidates": int(
                        plan["screen"]["expected_candidates"]
                    ),
                    "screen_attempts": int(plan["screen"]["expected_attempts"]),
                    "maximum_full55_attempts": int(
                        plan["full55_gate"]["maximum_expected_attempts"]
                    ),
                    "timeout_reader": "canonical_v1",
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
