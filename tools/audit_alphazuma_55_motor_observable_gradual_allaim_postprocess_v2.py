"""Independently audit all-actions postprocessing with timeout support."""

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

from tools import audit_alphazuma_55_motor_observable_postprocess_v1 as base
from tools import (
    run_alphazuma_55_motor_observable_gradual_allaim_postprocess_v2 as controller,
)


SCRIPT_PATH = Path(__file__).resolve()


def audit(*, plan_path: Path, expected_plan_sha256: str) -> dict[str, Any]:
    original_controller = base.controller
    original_script = base.SCRIPT_PATH
    try:
        base.controller = controller
        base.SCRIPT_PATH = SCRIPT_PATH
        receipt = base.audit(
            plan_path=plan_path,
            expected_plan_sha256=expected_plan_sha256,
        )
    finally:
        base.SCRIPT_PATH = original_script
        base.controller = original_controller
    receipt["timeout_reader"] = {
        "classification": "canonical_timeout_tuple_supported",
        "auditor": {
            "path": str(SCRIPT_PATH),
            "sha256": controller._sha256(SCRIPT_PATH),
        },
    }
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    receipt = audit(
        plan_path=args.plan.expanduser(),
        expected_plan_sha256=str(args.expected_plan_sha256),
    )
    output = args.output.expanduser().resolve()
    base._write_new(output, receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
