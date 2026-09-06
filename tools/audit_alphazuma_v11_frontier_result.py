"""Independently audit a completed AlphaZuma V1.1 frontier campaign.

The postprocess controller and its summarizer already validate their own inputs.
This auditor deliberately starts again from the frozen manifests, preregistrations,
and raw matrix shards.  It recomputes the candidate ranking, paired blind metrics,
and promotion gates without writing to the campaign output directory.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable


class AuditError(RuntimeError):
    """Raised when a frozen campaign invariant does not hold."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    _require(parsed.tzinfo is not None, f"timestamp has no timezone: {value}")
    return parsed.astimezone(timezone.utc)


def _models(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    models = manifest.get("models")
    _require(isinstance(models, list) and models, "manifest models must be a non-empty list")
    ids = [str(model.get("id", "")) for model in models]
    _require(all(ids) and len(ids) == len(set(ids)), "manifest model ids are empty or duplicated")
    return models


def _summary_by_id(aggregate: dict[str, Any]) -> dict[str, dict[str, Any]]:
    summaries = aggregate.get("models")
    _require(isinstance(summaries, list) and summaries, "aggregate models are absent")
    result = {str(summary["model"]["id"]): summary for summary in summaries}
    _require(len(result) == len(summaries), "aggregate model ids are duplicated")
    return result


def _level(summary: dict[str, Any], level_id: str) -> dict[str, Any]:
    matches = [
        value
        for value in summary["level_summaries"]
        if str(value["level_id"]).casefold() == level_id.casefold()
    ]
    _require(len(matches) == 1, f"expected one level summary for {level_id}")
    return matches[0]


def _audit_evaluation(
    *,
    root: Path,
    preregistration_path: Path,
    manifest_path: Path,
    aggregate_path: Path,
    expected_level_ids: list[str],
    expected_attempts_per_level: int,
    expected_seed_base: int,
    expected_model_ids: list[str],
) -> dict[str, Any]:
    preregistration = _read_json(preregistration_path)
    manifest = _read_json(manifest_path)
    aggregate = _read_json(aggregate_path)
    _require(
        preregistration.get("schema") == "zuma-rl.zero-shot-multilevel-preregistration"
        and preregistration.get("status") == "FROZEN_BEFORE_EVALUATION",
        f"evaluation preregistration is not frozen: {preregistration_path}",
    )
    _require(
        manifest.get("schema") == "zuma-rl.zero-shot-models-manifest"
        and manifest.get("status") == "FROZEN",
        f"models manifest is not frozen: {manifest_path}",
    )
    _require(
        aggregate.get("schema") == "zuma-rl.paired-multimodel-evaluation-aggregate"
        and aggregate.get("status") == "COMPLETE",
        f"evaluation aggregate is incomplete: {aggregate_path}",
    )
    levels = preregistration.get("levels")
    _require(isinstance(levels, list), "evaluation level inventory is absent")
    level_ids = [str(level["id"]) for level in levels]
    _require(level_ids == expected_level_ids, "evaluation level inventory differs from frozen contract")
    attempts_per_level = int(preregistration.get("attempts_per_level", -1))
    _require(
        attempts_per_level == expected_attempts_per_level,
        "evaluation attempts per level differ from frozen contract",
    )
    seed_plan = preregistration.get("seed_plan", {})
    expected_last_seed = expected_seed_base + len(level_ids) * attempts_per_level - 1
    _require(int(seed_plan.get("base_seed", -1)) == expected_seed_base, "wrong seed base")
    _require(int(seed_plan.get("last_seed", -1)) == expected_last_seed, "wrong last seed")
    _require(
        bool(seed_plan.get("all_seeds_frozen_before_policy_inference")),
        "seed plan is not marked frozen before inference",
    )

    preregistration_hash = _sha256(preregistration_path)
    manifest_hash = _sha256(manifest_path)
    _require(
        aggregate.get("preregistration", {}).get("sha256") == preregistration_hash,
        "aggregate preregistration hash differs",
    )
    _require(
        aggregate.get("models_manifest", {}).get("sha256") == manifest_hash,
        "aggregate manifest hash differs",
    )
    validation = aggregate.get("validation")
    _require(
        isinstance(validation, dict) and validation and all(bool(value) for value in validation.values()),
        "aggregate validation is not entirely true",
    )

    models = _models(manifest)
    model_ids = [str(model["id"]) for model in models]
    _require(model_ids == expected_model_ids, "evaluation model order differs from frozen contract")
    model_by_id = {str(model["id"]): model for model in models}
    for model in models:
        model_path = Path(str(model["path"])).resolve(strict=True)
        _require(model.get("sha256") == _sha256(model_path), f"model bytes changed: {model['id']}")

    execution = preregistration.get("execution", {})
    shard_count = int(execution.get("shard_count", -1))
    _require(shard_count == 2, "formal evaluation must contain exactly two shards")
    evaluator_hash = str(preregistration.get("evaluator", {}).get("sha256", ""))
    shard_receipts = aggregate.get("shards")
    _require(isinstance(shard_receipts, list) and len(shard_receipts) == shard_count, "wrong shard receipt count")
    receipt_by_path = {
        Path(str(receipt["path"])).resolve(strict=True): receipt for receipt in shard_receipts
    }
    all_rows: list[dict[str, Any]] = []
    shard_hashes = []
    for index in range(shard_count):
        shard_path = (root / f"matrix-shard-{index:02d}-of-{shard_count:02d}.json").resolve(
            strict=True
        )
        receipt = receipt_by_path.get(shard_path)
        _require(receipt is not None, f"aggregate lacks shard receipt: {shard_path}")
        shard_hash = _sha256(shard_path)
        _require(receipt.get("sha256") == shard_hash, f"shard hash differs: {shard_path}")
        shard = _read_json(shard_path)
        _require(
            shard.get("schema") == "zuma-rl.zero-shot-multimodel-shard"
            and shard.get("status") == "COMPLETE"
            and shard.get("error") is None,
            f"shard is incomplete: {shard_path}",
        )
        _require(
            int(shard.get("shard_index", -1)) == index
            and int(shard.get("shard_count", -1)) == shard_count,
            f"shard identity differs: {shard_path}",
        )
        _require(
            shard.get("preregistration", {}).get("sha256") == preregistration_hash,
            f"shard preregistration hash differs: {shard_path}",
        )
        _require(
            shard.get("models_manifest", {}).get("sha256") == manifest_hash,
            f"shard manifest hash differs: {shard_path}",
        )
        _require(shard.get("evaluator_sha256") == evaluator_hash, "shard evaluator hash differs")
        rows = shard.get("attempts")
        _require(
            isinstance(rows, list) and len(rows) == int(shard.get("expected_attempts", -1)),
            f"shard attempt count differs: {shard_path}",
        )
        all_rows.extend(rows)
        shard_hashes.append(shard_hash)

    expected: dict[tuple[str, str, int], int] = {}
    for model in models:
        for level_index, level_id in enumerate(level_ids):
            for attempt_index in range(attempts_per_level):
                expected[(str(model["id"]), level_id, attempt_index)] = (
                    expected_seed_base + level_index * attempts_per_level + attempt_index
                )
    actual: dict[tuple[str, str, int], int] = {}
    rows_by_model_level: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        key = (str(row["model_id"]), str(row["level_id"]), int(row["attempt_index"]))
        _require(key not in actual, f"duplicate attempt key: {key}")
        actual[key] = int(row["seed"])
        model = model_by_id.get(key[0])
        _require(model is not None and key[1] in level_ids, f"unknown attempt identity: {key}")
        _require(row.get("model_sha256") == model.get("sha256"), f"attempt model hash differs: {key}")
        _require(
            int(row.get("training_steps", -1)) == int(model.get("training_steps", -2)),
            f"attempt training steps differ: {key}",
        )
        outcome = row.get("outcome")
        _require(outcome in ("win", "loss", None), f"unknown outcome: {key}")
        _require(
            bool(row.get("time_limit_truncated")) == (outcome is None),
            f"attempt truncation and outcome differ: {key}",
        )
        rows_by_model_level[(key[0], key[1])].append(row)
    _require(actual == expected, "raw attempt matrix or seed matrix differs from frozen contract")

    summaries = _summary_by_id(aggregate)
    _require(list(summaries) == model_ids, "aggregate model order differs from manifest")
    for model_id in model_ids:
        summary = summaries[model_id]
        _require(summary.get("model") == model_by_id[model_id], f"aggregate model receipt differs: {model_id}")
        model_rows = [row for row in all_rows if str(row["model_id"]) == model_id]
        wins = sum(row.get("outcome") == "win" for row in model_rows)
        losses = sum(row.get("outcome") == "loss" for row in model_rows)
        truncations = sum(bool(row.get("time_limit_truncated")) for row in model_rows)
        _require(int(summary.get("attempts", -1)) == len(model_rows), f"model attempts differ: {model_id}")
        _require(int(summary.get("wins", -1)) == wins, f"model wins differ: {model_id}")
        _require(int(summary.get("losses", -1)) == losses, f"model losses differ: {model_id}")
        _require(int(summary.get("truncations", -1)) == truncations, f"model truncations differ: {model_id}")
        level_summaries = summary.get("level_summaries")
        _require(
            isinstance(level_summaries, list)
            and [str(value["level_id"]) for value in level_summaries] == level_ids,
            f"level summary order differs: {model_id}",
        )
        for level_summary in level_summaries:
            level_id = str(level_summary["level_id"])
            rows = rows_by_model_level[(model_id, level_id)]
            level_wins = [row for row in rows if row.get("outcome") == "win"]
            level_losses = sum(row.get("outcome") == "loss" for row in rows)
            level_truncations = sum(bool(row.get("time_limit_truncated")) for row in rows)
            best = (
                min(level_wins, key=lambda row: (int(row["ticks"]), -int(row["score"])))
                if level_wins
                else None
            )
            _require(int(level_summary.get("attempts", -1)) == len(rows), "level attempts differ")
            _require(int(level_summary.get("wins", -1)) == len(level_wins), "level wins differ")
            _require(int(level_summary.get("losses", -1)) == level_losses, "level losses differ")
            _require(
                int(level_summary.get("truncations", -1)) == level_truncations,
                "level truncations differ",
            )
            _require(level_summary.get("best_attempt") == best, "level best attempt differs")

    return {
        "preregistration": preregistration,
        "manifest": manifest,
        "aggregate": aggregate,
        "summaries": summaries,
        "rows": all_rows,
        "preregistration_sha256": preregistration_hash,
        "manifest_sha256": manifest_hash,
        "aggregate_sha256": _sha256(aggregate_path),
        "shard_sha256": shard_hashes,
        "attempts": len(all_rows),
        "seed_range": [expected_seed_base, expected_last_seed],
    }


def _selection_ranking(
    master: dict[str, Any],
    candidates: list[dict[str, Any]],
    summaries: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    zero_ids = {str(value).casefold() for value in master["zero_win_level_ids"]}
    anchor_id = str(master["selection"]["retention_anchor"]).casefold()
    rows = []
    for manifest_index, candidate in enumerate(candidates):
        summary = summaries[str(candidate["id"])]
        anchor = _level(summary, anchor_id)
        targets = [
            level
            for level in summary["level_summaries"]
            if str(level["level_id"]).casefold() != anchor_id
        ]
        zero = [
            level for level in targets if str(level["level_id"]).casefold() in zero_ids
        ]
        win_ticks = [
            int(level["best_attempt"]["ticks"])
            for level in targets
            if level.get("best_attempt") is not None
        ]
        median_ticks = statistics.median(win_ticks) if win_ticks else None
        metrics = {
            "anchor_wins": int(anchor["wins"]),
            "anchor_attempts": int(anchor["attempts"]),
            "anchor_retained": int(anchor["wins"]) >= math.ceil(0.75 * int(anchor["attempts"])),
            "zero_win_levels_cleared": sum(int(level["wins"]) > 0 for level in zero),
            "zero_win_wins": sum(int(level["wins"]) for level in zero),
            "selection_target_levels_cleared": sum(int(level["wins"]) > 0 for level in targets),
            "selection_target_wins": sum(int(level["wins"]) for level in targets),
            "median_best_win_ticks": median_ticks,
        }
        speed_rank = -float(median_ticks) if median_ticks is not None else -1_000_000_000.0
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
    return ranked


def _median_best_win_ticks(levels: Iterable[dict[str, Any]]) -> float | None:
    values = [
        int(level["best_attempt"]["ticks"])
        for level in levels
        if level.get("best_attempt") is not None
    ]
    return statistics.median(values) if values else None


def _subset_metrics(summary: dict[str, Any], included_ids: Iterable[str]) -> dict[str, Any]:
    included = {str(value).casefold() for value in included_ids}
    levels = [
        level
        for level in summary["level_summaries"]
        if str(level["level_id"]).casefold() in included
    ]
    _require(len(levels) == len(included), "blind aggregate omits a frozen level")
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
        "median_best_win_ticks": _median_best_win_ticks(levels),
        "level_summaries": levels,
    }


def _paired_counts(
    rows: list[dict[str, Any]],
    *,
    included_ids: Iterable[str],
    baseline_id: str,
    selected_id: str,
) -> dict[str, Any]:
    included = {str(value).casefold() for value in included_ids}
    paired: dict[tuple[str, int], dict[str, bool]] = defaultdict(dict)
    for row in rows:
        level_id = str(row["level_id"]).casefold()
        if level_id in included:
            paired[(level_id, int(row["attempt_index"]))][str(row["model_id"])] = (
                row.get("outcome") == "win"
            )
    expected_models = {baseline_id, selected_id}
    _require(paired and all(set(value) == expected_models for value in paired.values()), "paired matrix is incomplete")
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


def _audit_report(
    *,
    master: dict[str, Any],
    selection: dict[str, Any],
    final: dict[str, Any],
    report: dict[str, Any],
) -> dict[str, Any]:
    baseline_id = "baseline-v1"
    selected_id = "selected-v11"
    summaries = final["summaries"]
    _require(set(summaries) == {baseline_id, selected_id}, "final aggregate has wrong models")
    selected_summary = summaries[selected_id]
    baseline_summary = summaries[baseline_id]
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
    stretch_contract = master["stretch_goals"]
    stretch = {
        "target_wins": {
            "actual": selected_target["wins"],
            "required": int(stretch_contract["target_wins_at_least"]),
        },
        "zero_win_breadth": {
            "actual": selected_zero["levels_cleared"],
            "required": int(stretch_contract["zero_win_levels_cleared_at_least"]),
        },
        "zero_win_wins": {
            "actual": selected_zero["wins"],
            "required": int(stretch_contract["zero_win_total_wins_at_least"]),
        },
    }
    for check in stretch.values():
        check["passed"] = check["actual"] >= check["required"]

    paired_target = _paired_counts(
        final["rows"], included_ids=target_ids, baseline_id=baseline_id, selected_id=selected_id
    )
    paired_zero = _paired_counts(
        final["rows"], included_ids=zero_ids, baseline_id=baseline_id, selected_id=selected_id
    )
    expected_target = {
        "level_ids": target_ids,
        "selected": selected_target,
        "baseline": baseline_target,
        "delta_wins": selected_target["wins"] - baseline_target["wins"],
        "delta_levels_cleared": selected_target["levels_cleared"]
        - baseline_target["levels_cleared"],
        "paired_comparison": paired_target,
    }
    expected_zero = {
        "level_ids": zero_ids,
        "selected": selected_zero,
        "baseline": baseline_zero,
        "delta_wins": selected_zero["wins"] - baseline_zero["wins"],
        "delta_levels_cleared": selected_zero["levels_cleared"]
        - baseline_zero["levels_cleared"],
        "paired_comparison": paired_zero,
    }
    expected_anchor = {
        "selected": selected_anchor,
        "baseline": baseline_anchor,
        "delta_wins": int(selected_anchor["wins"]) - int(baseline_anchor["wins"]),
    }
    scientific_validity = bool(final["aggregate"].get("validation")) and all(
        bool(value) for value in final["aggregate"]["validation"].values()
    )
    promotion_passed = scientific_validity and all(check["passed"] for check in checks.values())
    stretch_passed = scientific_validity and all(check["passed"] for check in stretch.values())
    _require(
        report.get("schema") == "zuma-rl.alphazuma-v11-frontier-final-report"
        and report.get("status") == "COMPLETE",
        "final report is incomplete",
    )
    _require(report.get("selected_source_model") == selection["selected_model"], "selected source differs")
    _require(report.get("selection_decision") == selection, "embedded selection decision differs")
    _require(report.get("blind_validation") == final["aggregate"]["validation"], "blind validation differs")
    _require(report.get("promotion_gates") == checks, "promotion gate arithmetic differs")
    _require(report.get("stretch_goals") == stretch, "stretch-goal arithmetic differs")
    _require(report.get("target_blind") == expected_target, "target blind metrics differ")
    _require(report.get("zero_win_blind") == expected_zero, "zero-win blind metrics differ")
    _require(report.get("jungle2_blind") == expected_anchor, "Jungle2 blind metrics differ")
    _require(bool(report.get("scientifically_valid")) == scientific_validity, "scientific validity differs")
    _require(bool(report.get("v11_promotion_passed")) == promotion_passed, "promotion decision differs")
    _require(bool(report.get("stretch_goals_passed")) == stretch_passed, "stretch decision differs")
    goal_end = _parse_utc(str(master["schedule"]["goal_end_utc"]))
    completed = _parse_utc(str(report["completed_utc"]))
    _require(report.get("goal_end_utc") == master["schedule"]["goal_end_utc"], "goal end differs")
    _require(bool(report.get("within_goal_window")) == (completed <= goal_end), "goal-window flag differs")
    return {
        "scientifically_valid": scientific_validity,
        "v11_promotion_passed": promotion_passed,
        "stretch_goals_passed": stretch_passed,
        "within_goal_window": completed <= goal_end,
        "promotion_gates": checks,
        "target_blind": {
            "selected_wins": selected_target["wins"],
            "baseline_wins": baseline_target["wins"],
            "selected_levels_cleared": selected_target["levels_cleared"],
            "baseline_levels_cleared": baseline_target["levels_cleared"],
        },
        "zero_win_blind": {
            "selected_wins": selected_zero["wins"],
            "baseline_wins": baseline_zero["wins"],
            "selected_levels_cleared": selected_zero["levels_cleared"],
            "baseline_levels_cleared": baseline_zero["levels_cleared"],
        },
        "jungle2_selected_wins": int(selected_anchor["wins"]),
    }


def audit_campaign(master_path: Path, output_root: Path) -> dict[str, Any]:
    master_path = master_path.expanduser().resolve(strict=True)
    output_root = output_root.expanduser().resolve(strict=True)
    master = _read_json(master_path)
    _require(
        master.get("schema") == "zuma-rl.alphazuma-v11-frontier-postprocess-preregistration"
        and master.get("status") == "FROZEN_BEFORE_SELECTION",
        "master preregistration is not frozen",
    )
    runs = master.get("runs")
    levels = master.get("levels")
    _require(isinstance(runs, list) and len(runs) == 7, "master must contain seven routes")
    _require(isinstance(levels, list) and len(levels) == 17, "master must contain seventeen levels")
    level_ids = [str(level["id"]) for level in levels]
    _require(len(set(value.casefold() for value in level_ids)) == 17, "level ids are duplicated")
    selection_contract = master["selection"]
    final_contract = master["final_blind"]
    selection_ids = [str(value) for value in selection_contract["level_ids"]]
    selection_range = (
        int(selection_contract["seed_base"]),
        int(selection_contract["seed_base"])
        + len(selection_ids) * int(selection_contract["attempts_per_level"])
        - 1,
    )
    final_range = (
        int(final_contract["seed_base"]),
        int(final_contract["seed_base"])
        + len(level_ids) * int(final_contract["attempts_per_level"])
        - 1,
    )
    _require(selection_range[1] < final_range[0], "selection and final seed ranges overlap")

    controller_path = output_root / "controller_status.json"
    controller = _read_json(controller_path)
    _require(
        controller.get("status") == "COMPLETE"
        and controller.get("phase") == "COMPLETE"
        and controller.get("error") is None,
        "postprocess controller is not complete",
    )
    master_hash = _sha256(master_path)
    _require(
        controller.get("master_preregistration", {}).get("sha256") == master_hash,
        "controller master hash differs",
    )

    selection_manifest_path = output_root / "selection-models-manifest.json"
    selection_preregistration_path = output_root / "selection-preregistration.json"
    selection_root = output_root / "selection-evaluation"
    selection_aggregate_path = selection_root / "aggregate.json"
    selection_manifest = _read_json(selection_manifest_path)
    selection_candidates = _models(selection_manifest)
    _require(len(selection_candidates) == 15, "selection manifest must contain fifteen candidates")
    baseline = selection_candidates[0]
    _require(baseline == master["baseline_model"], "selection baseline differs from master")
    receipts = selection_manifest.get("training_receipts")
    _require(isinstance(receipts, list) and len(receipts) == 7, "selection training receipts differ")
    route_ids = [str(run["id"]) for run in runs]
    _require([str(receipt["route_id"]) for receipt in receipts] == route_ids, "training receipt route order differs")
    for receipt in receipts:
        for field in ("preregistration", "config", "completion"):
            artifact = receipt[field]
            artifact_path = Path(str(artifact["path"])).resolve(strict=True)
            _require(artifact.get("sha256") == _sha256(artifact_path), f"training receipt changed: {field}")
        route_id = str(receipt["route_id"])
        sources = [str(candidate.get("source", "")) for candidate in selection_candidates]
        _require(sources.count(f"{route_id}:mid") == 1, f"route midpoint candidate absent: {route_id}")
        _require(sources.count(f"{route_id}:final") == 1, f"route final candidate absent: {route_id}")
    selection_model_ids = [str(candidate["id"]) for candidate in selection_candidates]
    selection_eval = _audit_evaluation(
        root=selection_root,
        preregistration_path=selection_preregistration_path,
        manifest_path=selection_manifest_path,
        aggregate_path=selection_aggregate_path,
        expected_level_ids=selection_ids,
        expected_attempts_per_level=int(selection_contract["attempts_per_level"]),
        expected_seed_base=int(selection_contract["seed_base"]),
        expected_model_ids=selection_model_ids,
    )

    selection_decision_path = output_root / "selection-decision.json"
    selection_decision = _read_json(selection_decision_path)
    _require(
        selection_decision.get("schema") == "zuma-rl.alphazuma-v11-frontier-selection-decision"
        and selection_decision.get("status") == "COMPLETE",
        "selection decision is incomplete",
    )
    expected_ranking = _selection_ranking(master, selection_candidates, selection_eval["summaries"])
    _require(selection_decision.get("ranking") == expected_ranking, "candidate ranking differs")
    _require(
        selection_decision.get("selected_model") == expected_ranking[0]["model"],
        "selected candidate differs from frozen ranking",
    )
    _require(
        selection_decision.get("frozen_ranking") == selection_contract["ranking"],
        "selection ranking contract differs",
    )
    aggregate_receipts = selection_decision.get("aggregate_receipts", {})
    _require(
        aggregate_receipts.get("preregistration", {}).get("sha256")
        == selection_eval["preregistration_sha256"],
        "selection decision preregistration hash differs",
    )
    _require(
        aggregate_receipts.get("models_manifest", {}).get("sha256")
        == selection_eval["manifest_sha256"],
        "selection decision manifest hash differs",
    )
    _require(
        aggregate_receipts.get("validation") == selection_eval["aggregate"]["validation"],
        "selection decision validation differs",
    )

    final_manifest_path = output_root / "final-blind-models-manifest.json"
    final_preregistration_path = output_root / "final-blind-preregistration.json"
    final_root = output_root / "final-blind-evaluation"
    final_aggregate_path = final_root / "aggregate.json"
    final_manifest = _read_json(final_manifest_path)
    final_models = _models(final_manifest)
    _require([str(model["id"]) for model in final_models] == ["baseline-v1", "selected-v11"], "final model ids differ")
    _require(final_models[0] == baseline, "final baseline differs from selection baseline")
    selected_source = selection_decision["selected_model"]
    selected_final = final_models[1]
    _require(selected_final.get("selected_source_id") == selected_source["id"], "selected source id differs")
    for key, value in selected_source.items():
        if key != "id":
            _require(selected_final.get(key) == value, f"selected final model differs at {key}")
    _require(
        final_manifest.get("selection_decision", {}).get("sha256") == _sha256(selection_decision_path),
        "final manifest selection decision hash differs",
    )
    _require(
        final_manifest.get("selection_aggregate", {}).get("sha256")
        == selection_eval["aggregate_sha256"],
        "final manifest selection aggregate hash differs",
    )
    final_eval = _audit_evaluation(
        root=final_root,
        preregistration_path=final_preregistration_path,
        manifest_path=final_manifest_path,
        aggregate_path=final_aggregate_path,
        expected_level_ids=level_ids,
        expected_attempts_per_level=int(final_contract["attempts_per_level"]),
        expected_seed_base=int(final_contract["seed_base"]),
        expected_model_ids=["baseline-v1", "selected-v11"],
    )

    final_report_path = output_root / "final_report.json"
    markdown_path = output_root / "FINAL_REPORT.md"
    final_report = _read_json(final_report_path)
    _require(
        controller.get("final_report", {}).get("sha256") == _sha256(final_report_path),
        "controller final report hash differs",
    )
    _require(
        controller.get("summary", {}).get("sha256") == _sha256(markdown_path),
        "controller markdown hash differs",
    )
    performance = _audit_report(
        master=master,
        selection=selection_decision,
        final=final_eval,
        report=final_report,
    )
    return {
        "schema": "zuma-rl.alphazuma-v11-frontier-independent-audit",
        "version": 1,
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "master_preregistration": {"path": str(master_path), "sha256": master_hash},
        "output_root": str(output_root),
        "invariants_verified": [
            "controller_completion_and_receipts",
            "seven_route_fifteen_candidate_manifest",
            "model_and_training_receipt_hashes",
            "selection_attempt_and_seed_matrix",
            "frozen_candidate_ranking",
            "selection_to_final_model_binding",
            "final_blind_attempt_and_paired_seed_matrix",
            "raw_result_to_aggregate_arithmetic",
            "paired_mcnemar_counts",
            "seven_frozen_promotion_gates",
            "goal_window_arithmetic",
        ],
        "selection": {
            "candidates": len(selection_candidates),
            "attempt_records": selection_eval["attempts"],
            "seed_range": selection_eval["seed_range"],
            "selected_model": selection_decision["selected_model"],
            "aggregate_sha256": selection_eval["aggregate_sha256"],
            "shard_sha256": selection_eval["shard_sha256"],
        },
        "final_blind": {
            "models": 2,
            "attempt_records": final_eval["attempts"],
            "seed_range": final_eval["seed_range"],
            "aggregate_sha256": final_eval["aggregate_sha256"],
            "shard_sha256": final_eval["shard_sha256"],
        },
        "performance": performance,
        "artifacts": {
            "controller_status": {"path": str(controller_path), "sha256": _sha256(controller_path)},
            "selection_decision": {
                "path": str(selection_decision_path),
                "sha256": _sha256(selection_decision_path),
            },
            "final_report": {"path": str(final_report_path), "sha256": _sha256(final_report_path)},
            "markdown": {"path": str(markdown_path), "sha256": _sha256(markdown_path)},
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-preregistration", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = audit_campaign(args.master_preregistration, args.output_root)
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.receipt is not None:
        receipt = args.receipt.expanduser().resolve()
        receipt.parent.mkdir(parents=True, exist_ok=True)
        with receipt.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
