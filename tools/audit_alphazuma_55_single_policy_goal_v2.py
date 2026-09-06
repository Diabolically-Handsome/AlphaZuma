"""Audit the five-route AlphaZuma 55 single-policy goal."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Mapping

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools import audit_alphazuma_55_configured_successor_result_v3 as dagger_v3
from tools import audit_alphazuma_55_single_policy_goal as base


SCRIPT_PATH = Path(__file__).resolve()
GoalAuditError = base.GoalAuditError
_sha256 = base._sha256
_read = base._read
_require = base._require
_artifact = base._artifact
_reference = base._reference
_utc = base._utc
_write_exclusive = base._write_exclusive


def _production_recomputers() -> dict[str, dict[str, Any]]:
    recomputers = base._production_recomputers()
    recomputers["configured_successor_v3"] = {
        "path": dagger_v3.SCRIPT_PATH,
        "call": dagger_v3.audit,
    }
    return recomputers


def audit(
    plan_path: Path,
    *,
    now=None,
    recomputers: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    original_script = base.SCRIPT_PATH
    try:
        base.SCRIPT_PATH = SCRIPT_PATH
        return base.audit(
            plan_path,
            now=now,
            recomputers=recomputers or _production_recomputers(),
        )
    finally:
        base.SCRIPT_PATH = original_script


def main(argv: list[str] | None = None) -> int:
    args = base.build_parser().parse_args(argv)
    receipt_path = args.receipt.expanduser().resolve()
    result = audit(args.plan)
    if args.status_only:
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    if receipt_path.exists():
        existing = _read(receipt_path)
        _require(existing.get("status") == "PASS", "existing goal receipt is not PASS")
        _require(result.get("status") == "PASS", "current evidence no longer passes")
        _require(
            existing.get("plan") == result.get("plan")
            and existing.get("selected_campaign_id")
            == result.get("selected_campaign_id")
            and existing.get("selected_single_policy")
            == result.get("selected_single_policy"),
            "existing goal receipt differs from current evidence",
        )
        print(json.dumps(existing, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    if result["status"] != "PASS":
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 2
    _write_exclusive(receipt_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
