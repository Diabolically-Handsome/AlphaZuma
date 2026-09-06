"""Audit the configured V3 result with the frozen earlier-round tie break."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools import audit_alphazuma_55_configured_successor_result_v2 as v2


SCRIPT_PATH = Path(__file__).resolve()


def audit(master_path: Path) -> dict[str, Any]:
    original_script = v2.SCRIPT_PATH
    try:
        v2.SCRIPT_PATH = SCRIPT_PATH
        result = v2.audit(master_path)
    finally:
        v2.SCRIPT_PATH = original_script
    result["preflight_controller_version"] = 3
    return result


def main(argv: list[str] | None = None) -> int:
    args = v2.base.build_parser().parse_args(argv)
    receipt_path = args.receipt.expanduser().resolve()
    if receipt_path.exists():
        raise FileExistsError(f"refusing to overwrite audit receipt: {receipt_path}")
    result = audit(args.master_preregistration.expanduser())
    v2.base._write_exclusive(receipt_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
