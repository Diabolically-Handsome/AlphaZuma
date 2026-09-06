"""Independently audit the head-only successor result."""

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

from tools import audit_alphazuma_55_motor_observable_successor_v1 as base
from tools import (
    run_alphazuma_55_motor_observable_gradual_headonly_successor_v1
    as controller,
)


SCRIPT_PATH = Path(__file__).resolve()


def audit(
    *,
    master_path: Path,
    expected_master_sha256: str,
    controller_report: Path,
    controller_status: Path,
) -> dict[str, Any]:
    original_controller = base.controller
    original_script = base.SCRIPT_PATH
    try:
        base.controller = controller
        base.SCRIPT_PATH = SCRIPT_PATH
        return base.audit(
            master_path=master_path,
            expected_master_sha256=expected_master_sha256,
            controller_report=controller_report,
            controller_status=controller_status,
        )
    finally:
        base.SCRIPT_PATH = original_script
        base.controller = original_controller


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", required=True, type=Path)
    parser.add_argument("--expected-master-sha256", required=True)
    parser.add_argument("--controller-report", required=True, type=Path)
    parser.add_argument("--controller-status", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    receipt = audit(
        master_path=args.master.expanduser(),
        expected_master_sha256=str(args.expected_master_sha256),
        controller_report=args.controller_report.expanduser(),
        controller_status=args.controller_status.expanduser(),
    )
    output = args.output.expanduser().resolve()
    controller._write_new(output, receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
