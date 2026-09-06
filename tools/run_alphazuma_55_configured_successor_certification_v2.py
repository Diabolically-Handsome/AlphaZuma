"""Run configured certification with the frozen earlier-round tie break."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Mapping

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools import run_alphazuma_55_configured_successor_certification as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()
EXPECTED_RANKING = [
    "wins_desc",
    "level_coverage_desc",
    "total_score_desc",
    "median_winning_ticks_asc",
    "earlier_round_first",
]
_BASE_LOAD_MASTER = base._load_master
_sha256 = base._sha256


def _artifact(path: Path) -> dict[str, str]:
    path = path.resolve(strict=True)
    return {"path": str(path), "sha256": _sha256(path)}


def _load_master(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    actual_master = base._read(path)
    base._require(
        actual_master.get("compatibility_adapter", {}).get(
            "engineering_plan_lifecycle_normalization"
        )
        == (
            "FROZEN_BEFORE_TARGET_PREREGISTRATION -> "
            "FROZEN_DURING_TARGET_TRAINING"
        ),
        "configured successor V2 compatibility declaration is absent",
    )
    implementation = actual_master.get("implementation", {})
    expected_actual = {
        "master_builder": PROJECT_ROOT
        / "tools/build_alphazuma_55_configured_successor_master_v2.py",
        "controller": SCRIPT_PATH,
        "watcher": PROJECT_ROOT
        / "tools/watch_alphazuma_55_configured_successor_certification_v2.py",
        "independent_auditor": PROJECT_ROOT
        / "tools/audit_alphazuma_55_configured_successor_result_v2.py",
    }
    for name, expected in expected_actual.items():
        reference = implementation.get(name, {})
        resolved = Path(str(reference.get("path", ""))).resolve(strict=True)
        base._require(
            resolved == expected.resolve(strict=True)
            and reference.get("sha256") == _sha256(resolved),
            f"configured successor V2 {name} binding differs",
        )
    plan_path = Path(
        actual_master["successor"]["engineering_validation"]["plan"]["path"]
    ).resolve(strict=True)
    actual_plan = base._read(plan_path)
    base._require(
        actual_plan.get("status") == "FROZEN_BEFORE_TARGET_PREREGISTRATION",
        "unexpected V2 engineering plan lifecycle",
    )
    original_read = base._read
    original_script = base.SCRIPT_PATH
    original_ranking = base.SUPPORTED_RANKING

    def adapted_read(candidate: Path) -> dict[str, Any]:
        resolved = Path(candidate).resolve(strict=True)
        value = original_read(resolved)
        if resolved == path:
            value = copy.deepcopy(value)
            translated = value["implementation"]
            translated["master_builder"] = _artifact(
                PROJECT_ROOT
                / "tools/build_alphazuma_55_configured_successor_master.py"
            )
            translated["watcher"] = _artifact(
                PROJECT_ROOT
                / "tools/watch_alphazuma_55_configured_successor_certification.py"
            )
            translated["independent_auditor"] = _artifact(
                PROJECT_ROOT
                / "tools/audit_alphazuma_55_configured_successor_result.py"
            )
        elif resolved == plan_path:
            value = copy.deepcopy(value)
            value["status"] = "FROZEN_DURING_TARGET_TRAINING"
        return value

    try:
        base._read = adapted_read
        base.SCRIPT_PATH = SCRIPT_PATH
        base.SUPPORTED_RANKING = list(EXPECTED_RANKING)
        contract = _BASE_LOAD_MASTER(path)
    finally:
        base._read = original_read
        base.SCRIPT_PATH = original_script
        base.SUPPORTED_RANKING = original_ranking
    contract["master"] = actual_master
    contract["engineering_plan"] = actual_plan
    contract["ranking"] = list(EXPECTED_RANKING)
    return contract


def _engineering_decision(
    *,
    models: list[dict[str, Any]],
    rows: Iterable[Mapping[str, Any]],
    minimum_levels: int,
    minimum_wins: int,
    promotable_model_ids: Iterable[str],
    ranking_rule: list[str] | None = None,
) -> dict[str, Any]:
    ranking_rule = list(ranking_rule or EXPECTED_RANKING)
    base._require(
        ranking_rule == EXPECTED_RANKING,
        "unsupported V2 engineering ranking",
    )
    order = {str(model["id"]): index for index, model in enumerate(models)}
    rows_by_model: dict[str, list[Mapping[str, Any]]] = {
        model_id: [] for model_id in order
    }
    for row in rows:
        model_id = str(row["model_id"])
        base._require(
            model_id in rows_by_model,
            f"unknown engineering model: {model_id}",
        )
        rows_by_model[model_id].append(row)
    ranking: list[dict[str, Any]] = []
    for model in models:
        model_id = str(model["id"])
        model_rows = rows_by_model[model_id]
        base._require(
            len(model_rows) == base.LEVEL_COUNT,
            f"{model_id} lacks 55 attempts",
        )
        wins = [row for row in model_rows if row.get("outcome") == "win"]
        win_ticks = [int(row["ticks"]) for row in wins]
        metrics = {
            "wins": len(wins),
            "cleared_levels": len(
                {str(row["level_id"]) for row in wins}
            ),
            "total_score": sum(int(row["score"]) for row in model_rows),
            "median_winning_ticks": (
                statistics.median(win_ticks) if win_ticks else None
            ),
            "model_bytes": base._model_bytes(model),
        }
        median_key = (
            float(metrics["median_winning_ticks"])
            if metrics["median_winning_ticks"] is not None
            else math.inf
        )
        ranking.append(
            {
                "model": dict(model),
                "metrics": metrics,
                "_key": (
                    -metrics["wins"],
                    -metrics["cleared_levels"],
                    -metrics["total_score"],
                    median_key,
                    order[model_id],
                ),
            }
        )
    ranking.sort(key=lambda row: row["_key"])
    for index, row in enumerate(ranking, start=1):
        row["rank"] = index
        del row["_key"]
    selected = ranking[0]
    promotable = set(str(value) for value in promotable_model_ids)
    passed = (
        str(selected["model"]["id"]) in promotable
        and int(selected["metrics"]["cleared_levels"]) >= minimum_levels
        and int(selected["metrics"]["wins"]) >= minimum_wins
    )
    return {
        "schema": "zuma-rl.alphazuma-55-configured-successor-promotion-decision",
        "version": 1,
        "status": "PROMOTED" if passed else "NO_PROMOTION",
        "completed_utc": base._utc_now(),
        "ranking_rule": ranking_rule,
        "ranking": ranking,
        "selected_policy": selected["model"],
        "promotion_gate": {
            "actual": selected["metrics"],
            "required": {
                "selected_model_must_be_one_of": sorted(promotable),
                "minimum_cleared_levels": minimum_levels,
                "minimum_wins": minimum_wins,
            },
            "passed": passed,
        },
        "successor_formal_seed_consumption_authorized": passed,
    }


def run(*, master_path: Path, original_root: Path) -> dict[str, Any]:
    original_load = base._load_master
    original_decision = base._engineering_decision
    original_ranking = base.SUPPORTED_RANKING
    try:
        base._load_master = _load_master
        base._engineering_decision = _engineering_decision
        base.SUPPORTED_RANKING = list(EXPECTED_RANKING)
        return base.run(master_path=master_path, original_root=original_root)
    finally:
        base._load_master = original_load
        base._engineering_decision = original_decision
        base.SUPPORTED_RANKING = original_ranking


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-preregistration", required=True, type=Path)
    parser.add_argument("--original-root", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(
        master_path=args.master_preregistration.expanduser(),
        original_root=args.original_root.expanduser(),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
