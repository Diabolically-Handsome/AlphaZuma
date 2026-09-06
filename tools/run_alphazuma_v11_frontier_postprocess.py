"""Postprocess the AlphaZuma V1.1 frontier training portfolio.

The controller waits for every preregistered formal route, freezes a candidate
manifest containing each route's midpoint and final policy, performs isolated
candidate selection, and then runs one paired blind matrix against the
published V1 champion.  Calibration and failed/pre-OOM routes are never
eligible candidates.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import re
import statistics
import time
from typing import Any, Iterable

try:
    from run_overnight_v11_postprocess import (
        EVALUATOR,
        SUMMARIZER,
        _evaluate_and_summarize,
        _evaluation_preregistration,
        _level,
        _model_summary,
        _parse_utc,
        _read_json,
        _replace_json,
        _sha256,
        _utc_now,
        _write_new_json,
        _write_new_text,
    )
except ModuleNotFoundError:  # Support importing this controller as tools.<module> in tests.
    from tools.run_overnight_v11_postprocess import (
        EVALUATOR,
        SUMMARIZER,
        _evaluate_and_summarize,
        _evaluation_preregistration,
        _level,
        _model_summary,
        _parse_utc,
        _read_json,
        _replace_json,
        _sha256,
        _utc_now,
        _write_new_json,
        _write_new_text,
    )


SCRIPT_PATH = Path(__file__).resolve()


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _validate_master(master: dict[str, Any], master_path: Path) -> None:
    if master.get("schema") != "zuma-rl.alphazuma-v11-frontier-postprocess-preregistration":
        raise ValueError("unexpected postprocess preregistration schema")
    if master.get("version") != 1 or master.get("status") != "FROZEN_BEFORE_SELECTION":
        raise ValueError("postprocess preregistration is not a frozen version-1 contract")
    controller = _mapping(master.get("controller"), "controller")
    if Path(str(controller.get("path", ""))).resolve(strict=True) != SCRIPT_PATH:
        raise ValueError("controller path differs from preregistration")
    if controller.get("sha256") != _sha256(SCRIPT_PATH):
        raise ValueError("controller bytes differ from preregistration")
    for name, path in (("evaluator", EVALUATOR), ("summarizer", SUMMARIZER)):
        receipt = _mapping(master.get(name), name)
        if Path(str(receipt.get("path", ""))).resolve(strict=True) != path.resolve(strict=True):
            raise ValueError(f"{name} path differs from preregistration")
        if receipt.get("sha256") != _sha256(path):
            raise ValueError(f"{name} bytes differ from preregistration")
    baseline = _mapping(master.get("baseline_model"), "baseline_model")
    baseline_path = Path(str(baseline.get("path", ""))).resolve(strict=True)
    if baseline.get("sha256") != _sha256(baseline_path):
        raise ValueError("baseline model bytes differ from preregistration")
    runs = master.get("runs")
    if not isinstance(runs, list) or len(runs) != 7:
        raise ValueError("the frozen frontier portfolio must contain seven formal routes")
    run_ids: set[str] = set()
    run_dirs: set[Path] = set()
    for index, raw in enumerate(runs):
        run = _mapping(raw, f"runs[{index}]")
        run_id = str(run.get("id", ""))
        if not run_id or run_id in run_ids:
            raise ValueError("formal route ids are empty or duplicated")
        run_ids.add(run_id)
        run_dir = Path(str(run.get("run_dir", ""))).resolve()
        if run_dir in run_dirs:
            raise ValueError("formal route directories are duplicated")
        run_dirs.add(run_dir)
        prereg_path = Path(str(run.get("preregistration_path", ""))).resolve(strict=True)
        if run.get("preregistration_sha256") != _sha256(prereg_path):
            raise ValueError(f"training preregistration bytes changed: {run_id}")
        if int(run.get("source_counter_timesteps", -1)) < 0:
            raise ValueError(f"invalid source counter: {run_id}")
        if int(run.get("source_effective_training_steps", -1)) < 0:
            raise ValueError(f"invalid effective source steps: {run_id}")
    selection = _mapping(master.get("selection"), "selection")
    final_blind = _mapping(master.get("final_blind"), "final_blind")
    if int(selection.get("attempts_per_level", 0)) != 4:
        raise ValueError("selection contract must use four attempts per level")
    if int(final_blind.get("attempts_per_level", 0)) != 16:
        raise ValueError("final blind contract must use sixteen attempts per level")
    selection_levels = [str(value).casefold() for value in selection.get("level_ids", [])]
    if "jungle2" not in selection_levels:
        raise ValueError("selection matrix must include the Jungle2 retention anchor")
    zero_win = [str(value).casefold() for value in master.get("zero_win_level_ids", [])]
    if len(zero_win) != 3 or not set(zero_win).issubset(selection_levels):
        raise ValueError("all three frozen zero-win levels must be in selection")
    all_level_ids = [str(level["id"]).casefold() for level in master.get("levels", [])]
    if len(all_level_ids) != 17 or len(set(all_level_ids)) != 17:
        raise ValueError("final blind inventory must contain seventeen unique levels")
    seed_ranges = [
        (
            int(selection["seed_base"]),
            int(selection["seed_base"])
            + len(selection_levels) * int(selection["attempts_per_level"])
            - 1,
            "selection",
        ),
        (
            int(final_blind["seed_base"]),
            int(final_blind["seed_base"])
            + len(all_level_ids) * int(final_blind["attempts_per_level"])
            - 1,
            "final_blind",
        ),
    ]
    if seed_ranges[0][1] >= seed_ranges[1][0]:
        raise ValueError("selection and final blind seed namespaces overlap")
    if Path(master_path).resolve(strict=True) != master_path:
        raise ValueError("master preregistration path is not canonical")


def _checkpoint_steps(path: Path) -> int:
    match = re.search(r"_(\d+)_steps\.zip$", path.name)
    if match is None:
        raise ValueError(f"cannot parse checkpoint timestep: {path}")
    return int(match.group(1))


def _verify_training_receipts(run: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    completion_path = run_dir / "completion.json"
    completion = _read_json(completion_path)
    if completion.get("status") != "COMPLETE":
        raise RuntimeError(f"formal route did not complete: {run['id']}")
    config_path = run_dir / "config.json"
    config = _read_json(config_path)
    frozen_prereg = _mapping(config.get("preregistration"), "config.preregistration")
    if frozen_prereg.get("sha256") != run["preregistration_sha256"]:
        raise RuntimeError(f"training config is bound to the wrong preregistration: {run['id']}")
    if str(_mapping(config.get("run"), "config.run").get("id")) != str(run["training_run_id"]):
        raise RuntimeError(f"training config has the wrong run id: {run['id']}")
    source_counter = int(run["source_counter_timesteps"])
    if int(completion["source_steps"]) != source_counter:
        raise RuntimeError(f"completion source counter differs from contract: {run['id']}")
    if int(completion["actual_steps_this_run"]) <= 0:
        raise RuntimeError(f"formal route produced no training steps: {run['id']}")
    if int(completion["actual_steps"]) - source_counter != int(completion["actual_steps_this_run"]):
        raise RuntimeError(f"completion timestep accounting is inconsistent: {run['id']}")
    final = Path(str(_mapping(completion.get("final_model"), "final_model")["path"])).resolve(
        strict=True
    )
    if completion["final_model"]["sha256"] != _sha256(final):
        raise RuntimeError(f"formal final model hash changed: {run['id']}")
    return {
        "completion_path": completion_path,
        "completion": completion,
        "config_path": config_path,
        "config": config,
        "final_path": final,
    }


def _candidate_models(master: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    baseline = dict(master["baseline_model"])
    baseline["path"] = str(Path(str(baseline["path"])).resolve(strict=True))
    candidates = [baseline]
    receipts: list[dict[str, Any]] = []
    for run in master["runs"]:
        run_dir = Path(str(run["run_dir"])).resolve(strict=True)
        verified = _verify_training_receipts(run, run_dir)
        completion = verified["completion"]
        source_counter = int(run["source_counter_timesteps"])
        source_effective = int(run["source_effective_training_steps"])
        final_counter = int(completion["actual_steps"])
        midpoint_target = source_counter + int(completion["actual_steps_this_run"]) / 2.0
        checkpoints = [
            path
            for path in (run_dir / "checkpoints").glob("*_steps.zip")
            if source_counter < _checkpoint_steps(path) <= final_counter
        ]
        if not checkpoints:
            raise RuntimeError(f"formal route has no eligible checkpoint: {run['id']}")
        midpoint = min(
            checkpoints,
            key=lambda path: (abs(_checkpoint_steps(path) - midpoint_target), _checkpoint_steps(path)),
        )
        midpoint_counter = _checkpoint_steps(midpoint)
        stages = (
            ("mid", midpoint, midpoint_counter),
            ("final", verified["final_path"], final_counter),
        )
        for stage, path, counter in stages:
            steps_this_route = counter - source_counter
            candidates.append(
                {
                    "id": f"{run['id']}-{stage}-{counter}",
                    "training_steps": source_effective + steps_this_route,
                    "frontier_steps": steps_this_route,
                    "path": str(path),
                    "sha256": _sha256(path),
                    "source": f"{run['id']}:{stage}",
                    "route_family": run["family"],
                    "device": run["device"],
                }
            )
        receipts.append(
            {
                "route_id": run["id"],
                "training_run_id": run["training_run_id"],
                "family": run["family"],
                "device": run["device"],
                "preregistration": {
                    "path": run["preregistration_path"],
                    "sha256": run["preregistration_sha256"],
                },
                "config": {
                    "path": str(verified["config_path"]),
                    "sha256": _sha256(verified["config_path"]),
                },
                "completion": {
                    "path": str(verified["completion_path"]),
                    "sha256": _sha256(verified["completion_path"]),
                },
                "source_counter_timesteps": source_counter,
                "source_effective_training_steps": source_effective,
                "actual_counter_timesteps": final_counter,
                "actual_steps_this_route": int(completion["actual_steps_this_run"]),
                "midpoint_target_counter": midpoint_target,
                "midpoint_checkpoint": str(midpoint),
                "midpoint_counter_timesteps": midpoint_counter,
                "final_model": str(verified["final_path"]),
            }
        )
    expected = 1 + 2 * len(master["runs"])
    if len(candidates) != expected:
        raise RuntimeError(f"frozen candidate rule expected {expected} models, got {len(candidates)}")
    ids = [str(candidate["id"]) for candidate in candidates]
    if len(set(ids)) != len(ids):
        raise RuntimeError("candidate ids are duplicated")
    return candidates, receipts


def _median_best_win_ticks(
    summary: dict[str, Any], excluded: set[str] | None = None
) -> float | None:
    excluded = excluded or set()
    values = [
        int(level["best_attempt"]["ticks"])
        for level in summary["level_summaries"]
        if str(level["level_id"]).casefold() not in excluded and level["best_attempt"] is not None
    ]
    return statistics.median(values) if values else None


def _selection_decision(
    master: dict[str, Any], aggregate: dict[str, Any], candidates: list[dict[str, Any]]
) -> dict[str, Any]:
    zero_ids = {str(value).casefold() for value in master["zero_win_level_ids"]}
    anchor_id = str(master["selection"]["retention_anchor"]).casefold()
    rows: list[dict[str, Any]] = []
    for manifest_index, candidate in enumerate(candidates):
        summary = _model_summary(aggregate, str(candidate["id"]))
        anchor = _level(summary, anchor_id)
        targets = [
            level
            for level in summary["level_summaries"]
            if str(level["level_id"]).casefold() != anchor_id
        ]
        zero = [
            level
            for level in targets
            if str(level["level_id"]).casefold() in zero_ids
        ]
        median_best_win_ticks = _median_best_win_ticks(summary, {anchor_id})
        metrics = {
            "anchor_wins": int(anchor["wins"]),
            "anchor_attempts": int(anchor["attempts"]),
            "anchor_retained": int(anchor["wins"])
            >= math.ceil(0.75 * int(anchor["attempts"])),
            "zero_win_levels_cleared": sum(int(level["wins"]) > 0 for level in zero),
            "zero_win_wins": sum(int(level["wins"]) for level in zero),
            "selection_target_levels_cleared": sum(
                int(level["wins"]) > 0 for level in targets
            ),
            "selection_target_wins": sum(int(level["wins"]) for level in targets),
            "median_best_win_ticks": median_best_win_ticks,
        }
        # A candidate with no target win ranks below every finite winning time.
        # Keep the persisted rank key finite so strict JSON remains valid.
        speed_rank = (
            -float(median_best_win_ticks)
            if median_best_win_ticks is not None
            else -1_000_000_000.0
        )
        rank_key = (
            int(metrics["anchor_retained"]),
            metrics["zero_win_levels_cleared"],
            metrics["zero_win_wins"],
            metrics["selection_target_levels_cleared"],
            metrics["selection_target_wins"],
            speed_rank,
            -manifest_index,
        )
        rows.append(
            {
                "manifest_index": manifest_index,
                "model": candidate,
                "metrics": metrics,
                "rank_key": list(rank_key),
                "_rank_key": rank_key,
            }
        )
    ranked = sorted(rows, key=lambda row: row["_rank_key"], reverse=True)
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
        del row["_rank_key"]
    return {
        "schema": "zuma-rl.alphazuma-v11-frontier-selection-decision",
        "version": 1,
        "status": "COMPLETE",
        "completed_utc": _utc_now(),
        "aggregate_receipts": {
            "created_utc": aggregate["created_utc"],
            "preregistration": aggregate["preregistration"],
            "models_manifest": aggregate["models_manifest"],
            "validation": aggregate["validation"],
        },
        "frozen_ranking": master["selection"]["ranking"],
        "selected_model": ranked[0]["model"],
        "ranking": ranked,
    }


def _subset_metrics(summary: dict[str, Any], included_ids: Iterable[str]) -> dict[str, Any]:
    included = {str(value).casefold() for value in included_ids}
    levels = [
        level
        for level in summary["level_summaries"]
        if str(level["level_id"]).casefold() in included
    ]
    if len(levels) != len(included):
        raise ValueError("blind aggregate is missing one or more frozen levels")
    attempts = sum(int(level["attempts"]) for level in levels)
    wins = sum(int(level["wins"]) for level in levels)
    return {
        "levels": len(levels),
        "levels_cleared": sum(int(level["wins"]) > 0 for level in levels),
        "attempts": attempts,
        "wins": wins,
        "losses": sum(int(level["losses"]) for level in levels),
        "truncations": sum(int(level["truncations"]) for level in levels),
        "win_rate": wins / attempts if attempts else 0.0,
        "median_best_win_ticks": _median_best_win_ticks(
            {"level_summaries": levels}
        ),
        "level_summaries": levels,
    }


def _paired_counts(
    *,
    output_root: Path,
    shard_count: int,
    baseline_id: str,
    selected_id: str,
    included_ids: Iterable[str],
) -> dict[str, Any]:
    included = {str(value).casefold() for value in included_ids}
    paired: dict[tuple[str, int], dict[str, bool]] = defaultdict(dict)
    for index in range(shard_count):
        shard = _read_json(output_root / f"matrix-shard-{index:02d}-of-{shard_count:02d}.json")
        for row in shard["attempts"]:
            if str(row["level_id"]).casefold() not in included:
                continue
            paired[(str(row["level_id"]).casefold(), int(row["attempt_index"]))][
                str(row["model_id"])
            ] = row["outcome"] == "win"
    expected_models = {baseline_id, selected_id}
    if not paired or any(set(value) != expected_models for value in paired.values()):
        raise ValueError("paired blind matrix is incomplete")
    selected_only = sum(value[selected_id] and not value[baseline_id] for value in paired.values())
    baseline_only = sum(value[baseline_id] and not value[selected_id] for value in paired.values())
    both_win = sum(value[baseline_id] and value[selected_id] for value in paired.values())
    both_fail = sum(not value[baseline_id] and not value[selected_id] for value in paired.values())
    discordant = selected_only + baseline_only
    if discordant:
        lower = min(selected_only, baseline_only)
        tail = sum(math.comb(discordant, index) for index in range(lower + 1))
        p_value = min(1.0, 2.0 * tail / (2**discordant))
    else:
        p_value = 1.0
    return {
        "pairs": len(paired),
        "both_win": both_win,
        "selected_only_win": selected_only,
        "baseline_only_win": baseline_only,
        "both_fail": both_fail,
        "mcnemar_exact_two_sided_p": p_value,
    }


def _performance_report(
    *,
    master: dict[str, Any],
    selection: dict[str, Any],
    aggregate: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    baseline_id = "baseline-v1"
    selected_id = "selected-v11"
    selected_summary = _model_summary(aggregate, selected_id)
    baseline_summary = _model_summary(aggregate, baseline_id)
    anchor_id = str(master["selection"]["retention_anchor"])
    target_ids = [
        str(level["id"])
        for level in master["levels"]
        if str(level["id"]).casefold() != anchor_id.casefold()
    ]
    zero_ids = [str(value) for value in master["zero_win_level_ids"]]
    selected_target = _subset_metrics(selected_summary, target_ids)
    baseline_target = _subset_metrics(baseline_summary, target_ids)
    selected_zero = _subset_metrics(selected_summary, zero_ids)
    baseline_zero = _subset_metrics(baseline_summary, zero_ids)
    selected_anchor = _level(selected_summary, anchor_id)
    baseline_anchor = _level(baseline_summary, anchor_id)
    gates = master["promotion_gates"]
    checks = {
        "jungle2_retention": {
            "actual": int(selected_anchor["wins"]),
            "required": int(gates["jungle2_wins_at_least"]),
            "comparison": ">=",
        },
        "target_breadth": {
            "actual": selected_target["levels_cleared"],
            "required": int(gates["target_levels_cleared_at_least"]),
            "comparison": ">=",
        },
        "target_wins": {
            "actual": selected_target["wins"],
            "required": int(gates["target_wins_at_least"]),
            "comparison": ">=",
        },
        "zero_win_breadth": {
            "actual": selected_zero["levels_cleared"],
            "required": int(gates["zero_win_levels_cleared_at_least"]),
            "comparison": ">=",
        },
        "zero_win_wins": {
            "actual": selected_zero["wins"],
            "required": int(gates["zero_win_total_wins_at_least"]),
            "comparison": ">=",
        },
        "target_wins_noninferior_to_same_seed_baseline": {
            "actual": selected_target["wins"],
            "required": baseline_target["wins"],
            "comparison": ">=",
        },
        "target_breadth_noninferior_to_same_seed_baseline": {
            "actual": selected_target["levels_cleared"],
            "required": baseline_target["levels_cleared"],
            "comparison": ">=",
        },
    }
    for check in checks.values():
        check["passed"] = check["actual"] >= check["required"]
    stretch = master["stretch_goals"]
    stretch_checks = {
        "target_wins": {
            "actual": selected_target["wins"],
            "required": int(stretch["target_wins_at_least"]),
        },
        "zero_win_breadth": {
            "actual": selected_zero["levels_cleared"],
            "required": int(stretch["zero_win_levels_cleared_at_least"]),
        },
        "zero_win_wins": {
            "actual": selected_zero["wins"],
            "required": int(stretch["zero_win_total_wins_at_least"]),
        },
    }
    for check in stretch_checks.values():
        check["passed"] = check["actual"] >= check["required"]
    paired_target = _paired_counts(
        output_root=output_root,
        shard_count=2,
        baseline_id=baseline_id,
        selected_id=selected_id,
        included_ids=target_ids,
    )
    paired_zero = _paired_counts(
        output_root=output_root,
        shard_count=2,
        baseline_id=baseline_id,
        selected_id=selected_id,
        included_ids=zero_ids,
    )
    validation = aggregate.get("validation", {})
    scientific_validity = bool(validation) and all(bool(value) for value in validation.values())
    return {
        "schema": "zuma-rl.alphazuma-v11-frontier-final-report",
        "version": 1,
        "status": "COMPLETE",
        "completed_utc": _utc_now(),
        "goal_end_utc": master["schedule"]["goal_end_utc"],
        "within_goal_window": datetime.now(timezone.utc)
        <= _parse_utc(master["schedule"]["goal_end_utc"]),
        "scientifically_valid": scientific_validity,
        "v11_promotion_passed": scientific_validity
        and all(bool(check["passed"]) for check in checks.values()),
        "stretch_goals_passed": scientific_validity
        and all(bool(check["passed"]) for check in stretch_checks.values()),
        "selected_source_model": selection["selected_model"],
        "selection_decision": selection,
        "blind_validation": validation,
        "promotion_gates": checks,
        "stretch_goals": stretch_checks,
        "target_blind": {
            "level_ids": target_ids,
            "selected": selected_target,
            "baseline": baseline_target,
            "delta_wins": selected_target["wins"] - baseline_target["wins"],
            "delta_levels_cleared": selected_target["levels_cleared"]
            - baseline_target["levels_cleared"],
            "paired_comparison": paired_target,
        },
        "zero_win_blind": {
            "level_ids": zero_ids,
            "selected": selected_zero,
            "baseline": baseline_zero,
            "delta_wins": selected_zero["wins"] - baseline_zero["wins"],
            "delta_levels_cleared": selected_zero["levels_cleared"]
            - baseline_zero["levels_cleared"],
            "paired_comparison": paired_zero,
        },
        "jungle2_blind": {
            "selected": selected_anchor,
            "baseline": baseline_anchor,
            "delta_wins": int(selected_anchor["wins"]) - int(baseline_anchor["wins"]),
        },
    }


def _report_markdown(report: dict[str, Any]) -> str:
    target = report["target_blind"]
    zero = report["zero_win_blind"]
    anchor = report["jungle2_blind"]
    lines = [
        "# AlphaZuma V1.1 frontier result",
        "",
        f"- Completed: {report['completed_utc']}",
        f"- Within 10:30 local goal window: {report['within_goal_window']}",
        f"- Scientific validity: {report['scientifically_valid']}",
        f"- V1.1 promotion gate: {'PASS' if report['v11_promotion_passed'] else 'FAIL'}",
        f"- Stretch goals: {'PASS' if report['stretch_goals_passed'] else 'NOT YET'}",
        f"- Selected candidate: `{report['selected_source_model']['id']}`",
        "",
        "## Paired blind comparison",
        "",
        "| Model | Target levels | Target wins | Zero-win levels | Zero-win wins | Jungle2 wins |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| V1.1 selected | {target['selected']['levels_cleared']}/16 | "
            f"{target['selected']['wins']}/256 | {zero['selected']['levels_cleared']}/3 | "
            f"{zero['selected']['wins']}/48 | {anchor['selected']['wins']}/16 |"
        ),
        (
            f"| V1 baseline | {target['baseline']['levels_cleared']}/16 | "
            f"{target['baseline']['wins']}/256 | {zero['baseline']['levels_cleared']}/3 | "
            f"{zero['baseline']['wins']}/48 | {anchor['baseline']['wins']}/16 |"
        ),
        "",
        "## Frozen promotion gates",
        "",
        "| Gate | Actual | Required | Result |",
        "|---|---:|---:|---|",
    ]
    for name, check in report["promotion_gates"].items():
        lines.append(
            f"| {name} | {check['actual']} | {check['comparison']} {check['required']} | "
            f"{'PASS' if check['passed'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## Stretch goals (reported separately)",
            "",
            "| Goal | Actual | Required | Result |",
            "|---|---:|---:|---|",
        ]
    )
    for name, check in report["stretch_goals"].items():
        lines.append(
            f"| {name} | {check['actual']} | >= {check['required']} | "
            f"{'PASS' if check['passed'] else 'NOT YET'} |"
        )
    return "\n".join(lines) + "\n"


def _levels(master: dict[str, Any], ids: Iterable[str]) -> list[dict[str, Any]]:
    by_id = {str(level["id"]).casefold(): level for level in master["levels"]}
    result = []
    for level_id in ids:
        key = str(level_id).casefold()
        if key not in by_id:
            raise ValueError(f"frozen level is absent from inventory: {level_id}")
        result.append(by_id[key])
    return result


def _status_snapshot(master: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    complete = True
    snapshots: dict[str, Any] = {}
    for run in master["runs"]:
        run_dir = Path(str(run["run_dir"])).resolve()
        failure = run_dir / "failure.json"
        if failure.exists():
            raise RuntimeError(f"formal route failed: {failure}")
        completion = run_dir / "completion.json"
        training_status = run_dir / "training_status.json"
        snapshots[str(run["id"])] = (
            _read_json(completion)
            if completion.exists()
            else _read_json(training_status)
            if training_status.exists()
            else {"status": "STARTING"}
        )
        complete = complete and completion.exists()
    return complete, snapshots


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-preregistration", type=Path, required=True)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    master_path = args.master_preregistration.expanduser().resolve(strict=True)
    original_root = args.original_root.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve()
    master = _read_json(master_path)
    _validate_master(master, master_path)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "VALID",
                    "master_preregistration": str(master_path),
                    "master_sha256": _sha256(master_path),
                    "formal_routes": len(master["runs"]),
                    "candidate_count": 1 + 2 * len(master["runs"]),
                    "selection_attempts": len(master["selection"]["level_ids"])
                    * int(master["selection"]["attempts_per_level"]),
                    "final_blind_attempts_per_model": len(master["levels"])
                    * int(master["final_blind"]["attempts_per_level"]),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if output_root.exists():
        raise FileExistsError(f"refusing to reuse postprocess output root: {output_root}")
    output_root.mkdir(parents=True)
    status_path = output_root / "controller_status.json"
    state: dict[str, Any] = {
        "schema": "zuma-rl.alphazuma-v11-frontier-controller-status",
        "version": 1,
        "status": "RUNNING",
        "phase": "WAITING_FOR_TRAINING",
        "started_utc": _utc_now(),
        "updated_utc": _utc_now(),
        "master_preregistration": {"path": str(master_path), "sha256": _sha256(master_path)},
        "training": {},
        "error": None,
    }
    _replace_json(status_path, state)
    try:
        latest_allowed = _parse_utc(master["schedule"]["training_stop_utc"]) + timedelta(
            minutes=20
        )
        while True:
            complete, snapshots = _status_snapshot(master)
            state["training"] = snapshots
            state["updated_utc"] = _utc_now()
            _replace_json(status_path, state)
            if complete:
                break
            if datetime.now(timezone.utc) > latest_allowed:
                raise TimeoutError("formal routes did not finalize within twenty minutes of stop time")
            time.sleep(args.poll_seconds)

        state["phase"] = "FREEZING_CANDIDATES"
        state["updated_utc"] = _utc_now()
        _replace_json(status_path, state)
        candidates, training_receipts = _candidate_models(master)
        selection_manifest = output_root / "selection-models-manifest.json"
        _write_new_json(
            selection_manifest,
            {
                "schema": "zuma-rl.zero-shot-models-manifest",
                "version": 1,
                "status": "FROZEN",
                "frozen_utc": _utc_now(),
                "purpose": "V1.1 frontier candidate selection on isolated frozen seeds",
                "candidate_rule": master["selection"]["candidate_rule"],
                "models": candidates,
                "training_receipts": training_receipts,
            },
        )
        selection_root = output_root / "selection-evaluation"
        selection_prereg = output_root / "selection-preregistration.json"
        selection_levels = _levels(master, master["selection"]["level_ids"])
        _write_new_json(
            selection_prereg,
            _evaluation_preregistration(
                baseline_model=candidates[0],
                levels=selection_levels,
                attempts_per_level=int(master["selection"]["attempts_per_level"]),
                base_seed=int(master["selection"]["seed_base"]),
                output_root=selection_root,
                shard_count=2,
                devices=["cuda:0", "cuda:1"],
                environment=master["environment"],
                purpose="V1.1 frontier candidate selection; not final blind evidence",
            ),
        )
        state["phase"] = "SELECTION_EVALUATION"
        state["updated_utc"] = _utc_now()
        _replace_json(status_path, state)
        selection_aggregate = _evaluate_and_summarize(
            phase="selection",
            preregistration=selection_prereg,
            manifest=selection_manifest,
            original_root=original_root,
            output_root=selection_root,
            shard_count=2,
            purpose="smoke",
        )
        selection = _selection_decision(master, selection_aggregate, candidates)
        selection_path = output_root / "selection-decision.json"
        _write_new_json(selection_path, selection)

        baseline = dict(candidates[0])
        selected = dict(selection["selected_model"])
        selected["selected_source_id"] = selected["id"]
        selected["id"] = "selected-v11"
        final_manifest = output_root / "final-blind-models-manifest.json"
        _write_new_json(
            final_manifest,
            {
                "schema": "zuma-rl.zero-shot-models-manifest",
                "version": 1,
                "status": "FROZEN",
                "frozen_utc": _utc_now(),
                "purpose": "paired final blind comparison against published V1 champion",
                "models": [baseline, selected],
                "selection_decision": {
                    "path": str(selection_path),
                    "sha256": _sha256(selection_path),
                },
                "selection_aggregate": {
                    "path": str(selection_root / "aggregate.json"),
                    "sha256": _sha256(selection_root / "aggregate.json"),
                },
            },
        )
        final_root = output_root / "final-blind-evaluation"
        final_prereg = output_root / "final-blind-preregistration.json"
        _write_new_json(
            final_prereg,
            _evaluation_preregistration(
                baseline_model=baseline,
                levels=master["levels"],
                attempts_per_level=int(master["final_blind"]["attempts_per_level"]),
                base_seed=int(master["final_blind"]["seed_base"]),
                output_root=final_root,
                shard_count=2,
                devices=["cuda:0", "cuda:1"],
                environment=master["environment"],
                purpose="paired final blind V1.1 promotion evaluation",
            ),
        )
        state["phase"] = "FINAL_BLIND_EVALUATION"
        state["updated_utc"] = _utc_now()
        _replace_json(status_path, state)
        final_aggregate = _evaluate_and_summarize(
            phase="final-blind",
            preregistration=final_prereg,
            manifest=final_manifest,
            original_root=original_root,
            output_root=final_root,
            shard_count=2,
            purpose="morning-target",
        )

        state["phase"] = "FINAL_REPORT"
        state["updated_utc"] = _utc_now()
        _replace_json(status_path, state)
        report = _performance_report(
            master=master,
            selection=selection,
            aggregate=final_aggregate,
            output_root=final_root,
        )
        report_path = output_root / "final_report.json"
        markdown_path = output_root / "FINAL_REPORT.md"
        _write_new_json(report_path, report)
        _write_new_text(markdown_path, _report_markdown(report))
        state.update(
            {
                "status": "COMPLETE",
                "phase": "COMPLETE",
                "updated_utc": _utc_now(),
                "final_report": {"path": str(report_path), "sha256": _sha256(report_path)},
                "summary": {"path": str(markdown_path), "sha256": _sha256(markdown_path)},
            }
        )
        _replace_json(status_path, state)
    except BaseException as error:
        state.update(
            {
                "status": "FAILED",
                "phase": "FAILED",
                "updated_utc": _utc_now(),
                "error": {"type": type(error).__name__, "message": str(error)},
            }
        )
        _replace_json(status_path, state)
        raise


if __name__ == "__main__":
    main()
