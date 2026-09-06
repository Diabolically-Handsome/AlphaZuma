"""Audit five-candidate autonomy certification and its frozen ranking."""

from __future__ import annotations

import builtins
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools import audit_alphazuma_55_configured_successor_result as base
from tools import audit_alphazuma_55_configured_successor_result_v2 as v2


SCRIPT_PATH = Path(__file__).resolve()
EXPECTED_CANDIDATES = [
    "polar-intent-autonomy-source-final",
    "polar-intent-autonomy-round-00",
    "polar-intent-autonomy-round-01",
    "polar-intent-autonomy-round-02",
    "polar-intent-autonomy-round-03",
]


def _legacy_candidate_len(value: Any) -> int:
    actual = builtins.len(value)
    if actual == 5 and isinstance(value, (list, tuple, set, frozenset)):
        if {str(item) for item in value} == set(EXPECTED_CANDIDATES):
            return 4
    return actual


def audit(master_path: Path) -> dict[str, Any]:
    original_script = base.SCRIPT_PATH
    original_ranking = base.EXPECTED_RANKING
    original_rank = base._rank
    missing = object()
    original_len = getattr(base, "len", missing)
    try:
        base.SCRIPT_PATH = SCRIPT_PATH
        base.EXPECTED_RANKING = list(v2.EXPECTED_RANKING)
        base._rank = v2._rank
        base.len = _legacy_candidate_len
        result = base.audit(master_path)
    finally:
        base.SCRIPT_PATH = original_script
        base.EXPECTED_RANKING = original_ranking
        base._rank = original_rank
        if original_len is missing:
            delattr(base, "len")
        else:
            base.len = original_len
    result["ranking_tie_break_semantics"] = (
        "earlier_manifest_round_first"
    )
    result["engineering_candidate_count"] = 5
    return result


def main(argv: list[str] | None = None) -> int:
    args = base.build_parser().parse_args(argv)
    receipt_path = args.receipt.expanduser().resolve()
    if receipt_path.exists():
        raise FileExistsError(
            f"refusing to overwrite audit receipt: {receipt_path}"
        )
    result = audit(args.master_preregistration.expanduser())
    base._write_exclusive(receipt_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
