"""Audit configured certification with the frozen earlier-round tie break."""

from __future__ import annotations

import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Mapping

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools import audit_alphazuma_55_configured_successor_result as base


SCRIPT_PATH = Path(__file__).resolve()
EXPECTED_RANKING = [
    "wins_desc",
    "level_coverage_desc",
    "total_score_desc",
    "median_winning_ticks_asc",
    "earlier_round_first",
]


def _rank(
    models: list[dict[str, Any]], rows: Iterable[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    order = {str(model["id"]): index for index, model in enumerate(models)}
    grouped: dict[str, list[Mapping[str, Any]]] = {
        model_id: [] for model_id in order
    }
    for row in rows:
        model_id = str(row["model_id"])
        base._require(model_id in grouped, f"unknown engineering model: {model_id}")
        grouped[model_id].append(row)
    ranked: list[dict[str, Any]] = []
    for model in models:
        model_id = str(model["id"])
        model_rows = grouped[model_id]
        base._require(
            len(model_rows) == base.LEVEL_COUNT,
            f"{model_id} lacks 55 attempts",
        )
        wins = [row for row in model_rows if row.get("outcome") == "win"]
        ticks = [int(row["ticks"]) for row in wins]
        metrics = {
            "wins": len(wins),
            "cleared_levels": len(
                {str(row["level_id"]) for row in wins}
            ),
            "total_score": sum(int(row["score"]) for row in model_rows),
            "median_winning_ticks": (
                statistics.median(ticks) if ticks else None
            ),
            "model_bytes": base._model_bytes(model),
        }
        ranked.append(
            {
                "model": dict(model),
                "metrics": metrics,
                "_key": (
                    -metrics["wins"],
                    -metrics["cleared_levels"],
                    -metrics["total_score"],
                    (
                        float(metrics["median_winning_ticks"])
                        if metrics["median_winning_ticks"] is not None
                        else math.inf
                    ),
                    order[model_id],
                ),
            }
        )
    ranked.sort(key=lambda row: row["_key"])
    for index, row in enumerate(ranked, start=1):
        row["rank"] = index
        del row["_key"]
    return ranked


def audit(master_path: Path) -> dict[str, Any]:
    original_script = base.SCRIPT_PATH
    original_ranking = base.EXPECTED_RANKING
    original_rank = base._rank
    try:
        base.SCRIPT_PATH = SCRIPT_PATH
        base.EXPECTED_RANKING = list(EXPECTED_RANKING)
        base._rank = _rank
        result = base.audit(master_path)
    finally:
        base.SCRIPT_PATH = original_script
        base.EXPECTED_RANKING = original_ranking
        base._rank = original_rank
    result["ranking_tie_break_semantics"] = "earlier_manifest_round_first"
    return result


def main(argv: list[str] | None = None) -> int:
    args = base.build_parser().parse_args(argv)
    receipt_path = args.receipt.expanduser().resolve()
    if receipt_path.exists():
        raise FileExistsError(f"refusing to overwrite audit receipt: {receipt_path}")
    result = audit(args.master_preregistration.expanduser())
    base._write_exclusive(receipt_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
