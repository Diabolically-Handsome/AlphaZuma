"""Validate and aggregate a paired multimodel multilevel evaluation."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
from typing import Any, Mapping


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_atomic(path: Path, data: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _region(level_id: str) -> str:
    folded = level_id.casefold()
    for prefix in ("jungle", "village", "city", "coast", "grotto", "volcano"):
        if folded.startswith(prefix):
            return prefix
    return "other"


def _level_summary(
    *,
    frozen_index: int,
    level: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    wins = [row for row in rows if row["outcome"] == "win"]
    win_ticks = [int(row["ticks"]) for row in wins]
    best = min(wins, key=lambda row: (int(row["ticks"]), -int(row["score"]))) if wins else None
    return {
        "frozen_index": frozen_index,
        "level_id": level["id"],
        "display_name": level["display_name"],
        "region": _region(str(level["id"])),
        "curve_count": int(level["curve_count"]),
        "attempts": len(rows),
        "wins": len(wins),
        "losses": sum(row["outcome"] == "loss" for row in rows),
        "truncations": sum(bool(row["time_limit_truncated"]) for row in rows),
        "win_rate": len(wins) / len(rows) if rows else 0.0,
        "best_ticks": min(win_ticks) if win_ticks else None,
        "best_seconds": min(win_ticks) / 100.0 if win_ticks else None,
        "median_win_ticks": statistics.median(win_ticks) if win_ticks else None,
        "median_win_seconds": statistics.median(win_ticks) / 100.0 if win_ticks else None,
        "best_attempt": best,
        "classification": "CLEARED" if wins else "NOT_CLEARED",
    }


def _aggregate_model(
    *,
    model: dict[str, Any],
    levels: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    attempts_per_level: int,
) -> dict[str, Any]:
    by_level: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_level[str(row["level_id"])].append(row)
    summaries = [
        _level_summary(
            frozen_index=index,
            level=level,
            rows=by_level[str(level["id"])],
        )
        for index, level in enumerate(levels)
    ]
    wins = [row for row in rows if row["outcome"] == "win"]
    region_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        region_rows[_region(str(row["level_id"]))].append(row)
    regions = {}
    for region, values in sorted(region_rows.items()):
        region_levels = [summary for summary in summaries if summary["region"] == region]
        region_wins = sum(value["outcome"] == "win" for value in values)
        regions[region] = {
            "levels": len(region_levels),
            "levels_cleared": sum(summary["wins"] > 0 for summary in region_levels),
            "attempts": len(values),
            "wins": region_wins,
            "win_rate": region_wins / len(values),
        }
    return {
        "model": model,
        "attempts": len(rows),
        "wins": len(wins),
        "losses": sum(row["outcome"] == "loss" for row in rows),
        "truncations": sum(bool(row["time_limit_truncated"]) for row in rows),
        "win_rate": len(wins) / len(rows) if rows else 0.0,
        "levels": len(levels),
        "levels_cleared": sum(summary["wins"] > 0 for summary in summaries),
        "levels_reliably_cleared": sum(
            summary["wins"] >= math.ceil(0.75 * attempts_per_level)
            for summary in summaries
        ),
        "perfect_levels": sum(summary["wins"] == attempts_per_level for summary in summaries),
        "median_win_ticks": statistics.median(int(row["ticks"]) for row in wins) if wins else None,
        "median_win_seconds": (
            statistics.median(int(row["ticks"]) for row in wins) / 100.0
            if wins
            else None
        ),
        "best_attempt": (
            min(wins, key=lambda row: (int(row["ticks"]), -int(row["score"])))
            if wins
            else None
        ),
        "regions": regions,
        "curve_counts": {
            str(curves): {
                "levels": sum(summary["curve_count"] == curves for summary in summaries),
                "levels_cleared": sum(
                    summary["curve_count"] == curves and summary["wins"] > 0
                    for summary in summaries
                ),
                "attempts": sum(
                    summary["attempts"] for summary in summaries if summary["curve_count"] == curves
                ),
                "wins": sum(
                    summary["wins"] for summary in summaries if summary["curve_count"] == curves
                ),
            }
            for curves in sorted({int(summary["curve_count"]) for summary in summaries})
        },
        "level_summaries": summaries,
    }


def _selection_key(summary: dict[str, Any], manifest_index: int) -> tuple[Any, ...]:
    levels = summary["level_summaries"]
    anchor = next(
        (level for level in levels if str(level["level_id"]).casefold() == "jungle2"),
        None,
    )
    if anchor is None:
        raise ValueError("selection evaluation must include Jungle2")
    targets = [level for level in levels if str(level["level_id"]).casefold() != "jungle2"]
    target_wins = sum(level["wins"] for level in targets)
    village_wins = sum(level["wins"] for level in targets if level["region"] == "village")
    two_curve_wins = sum(level["wins"] for level in targets if level["curve_count"] == 2)
    win_ticks = [
        int(level["best_attempt"]["ticks"])
        for level in targets
        if level["best_attempt"] is not None
    ]
    return (
        int(anchor["wins"] >= math.ceil(0.75 * anchor["attempts"])),
        sum(level["wins"] > 0 for level in targets),
        target_wins,
        village_wins,
        two_curve_wins,
        -statistics.median(win_ticks) if win_ticks else float("-inf"),
        -manifest_index,
    )


def _markdown(aggregate: dict[str, Any]) -> str:
    lines = [
        "# AlphaZuma paired multimodel evaluation",
        "",
        f"- Status: {aggregate['status']}",
        f"- Purpose: {aggregate['purpose']}",
        f"- Models: {len(aggregate['models'])}",
        f"- Levels: {aggregate['levels']}",
        f"- Attempts per level/model: {aggregate['attempts_per_level']}",
        "",
        "| Model | Levels cleared | Wins | Win rate | Jungle2 | Village wins | Two-curve wins |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in aggregate["models"]:
        level_summaries = model["level_summaries"]
        anchor = next(
            (row for row in level_summaries if str(row["level_id"]).casefold() == "jungle2"),
            None,
        )
        village_wins = sum(row["wins"] for row in level_summaries if row["region"] == "village")
        two_curve_wins = sum(row["wins"] for row in level_summaries if row["curve_count"] == 2)
        anchor_text = f"{anchor['wins']}/{anchor['attempts']}" if anchor else "-"
        lines.append(
            f"| {model['model']['id']} | {model['levels_cleared']}/{model['levels']} | "
            f"{model['wins']}/{model['attempts']} | {100.0 * model['win_rate']:.1f}% | "
            f"{anchor_text} | {village_wins} | {two_curve_wins} |"
        )
    if aggregate.get("selection"):
        selection = aggregate["selection"]
        lines.extend(
            [
                "",
                "## Frozen selection decision",
                "",
                f"Selected: `{selection['selected_model_id']}`",
                "",
                "The decision follows the preregistered lexicographic ranking; these are selection results, not morning blind claims.",
            ]
        )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--models-manifest", type=Path, required=True)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument(
        "--purpose",
        choices=("selection", "morning-target", "morning-anchor", "smoke"),
        required=True,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    prereg_path = args.preregistration.expanduser().resolve(strict=True)
    manifest_path = args.models_manifest.expanduser().resolve(strict=True)
    run_directory = args.run_directory.expanduser().resolve(strict=True)
    prereg = _read_json(prereg_path)
    manifest = _read_json(manifest_path)
    if prereg.get("schema") != "zuma-rl.zero-shot-multilevel-preregistration":
        raise ValueError("unexpected preregistration schema")
    if prereg.get("status") != "FROZEN_BEFORE_EVALUATION":
        raise ValueError("evaluation preregistration is not frozen")
    if manifest.get("schema") != "zuma-rl.zero-shot-models-manifest":
        raise ValueError("unexpected models manifest schema")
    if manifest.get("status") != "FROZEN":
        raise ValueError("models manifest is not frozen")
    models = manifest.get("models")
    levels = prereg.get("levels")
    if not isinstance(models, list) or not models or not isinstance(levels, list) or not levels:
        raise ValueError("models and levels must be non-empty lists")
    for model in models:
        path = Path(str(model["path"])).resolve(strict=True)
        if model["sha256"] != _sha256(path):
            raise ValueError(f"model bytes changed: {model['id']}")

    shard_count = int(prereg["execution"]["shard_count"])
    shard_paths = [
        run_directory / f"matrix-shard-{index:02d}-of-{shard_count:02d}.json"
        for index in range(shard_count)
    ]
    shards = [_read_json(path) for path in shard_paths]
    prereg_sha = _sha256(prereg_path)
    manifest_sha = _sha256(manifest_path)
    evaluator_sha = str(prereg["evaluator"]["sha256"])
    all_rows: list[dict[str, Any]] = []
    shard_receipts = []
    for index, (path, shard) in enumerate(zip(shard_paths, shards, strict=True)):
        if shard.get("schema") != "zuma-rl.zero-shot-multimodel-shard":
            raise ValueError(f"unexpected shard schema: {path}")
        if shard.get("status") != "COMPLETE" or shard.get("error") is not None:
            raise ValueError(f"shard is incomplete or errored: {path}")
        if int(shard.get("shard_index", -1)) != index or int(shard.get("shard_count", -1)) != shard_count:
            raise ValueError(f"shard identity mismatch: {path}")
        if shard.get("preregistration", {}).get("sha256") != prereg_sha:
            raise ValueError(f"shard preregistration mismatch: {path}")
        if shard.get("models_manifest", {}).get("sha256") != manifest_sha:
            raise ValueError(f"shard models manifest mismatch: {path}")
        if shard.get("evaluator_sha256") != evaluator_sha:
            raise ValueError(f"shard evaluator mismatch: {path}")
        rows = shard.get("attempts")
        if not isinstance(rows, list) or len(rows) != int(shard["expected_attempts"]):
            raise ValueError(f"shard attempt count mismatch: {path}")
        all_rows.extend(rows)
        shard_receipts.append(
            {
                "path": str(path),
                "sha256": _sha256(path),
                "device": shard["device"],
                "attempts": len(rows),
                "wall_seconds": float(shard["runtime"]["wall_seconds"]),
            }
        )

    attempts_per_level = int(prereg["attempts_per_level"])
    base_seed = int(prereg["seed_plan"]["base_seed"])
    expected: dict[tuple[str, str, int], int] = {}
    for model in models:
        for level_index, level in enumerate(levels):
            for attempt_index in range(attempts_per_level):
                expected[(str(model["id"]), str(level["id"]), attempt_index)] = (
                    base_seed + level_index * attempts_per_level + attempt_index
                )
    actual: dict[tuple[str, str, int], int] = {}
    model_by_id = {str(model["id"]): model for model in models}
    level_by_id = {str(level["id"]): level for level in levels}
    for row in all_rows:
        key = (str(row["model_id"]), str(row["level_id"]), int(row["attempt_index"]))
        if key in actual:
            raise ValueError(f"duplicate attempt key: {key}")
        actual[key] = int(row["seed"])
        model = model_by_id.get(key[0])
        level = level_by_id.get(key[1])
        if model is None or level is None:
            raise ValueError(f"unknown model or level in attempt: {key}")
        if row["model_sha256"] != model["sha256"]:
            raise ValueError(f"attempt model hash mismatch: {key}")
        if int(row["training_steps"]) != int(model["training_steps"]):
            raise ValueError(f"attempt training step mismatch: {key}")
        if bool(row["time_limit_truncated"]) != bool(row["outcome"] is None):
            raise ValueError(f"attempt truncation/outcome mismatch: {key}")
    if actual != expected:
        missing = sorted(set(expected) - set(actual))[:5]
        extra = sorted(set(actual) - set(expected))[:5]
        wrong = [key for key in expected.keys() & actual.keys() if expected[key] != actual[key]][:5]
        raise ValueError(f"attempt matrix differs; missing={missing}, extra={extra}, wrong={wrong}")

    aggregates = []
    for model in models:
        model_rows = [row for row in all_rows if row["model_id"] == model["id"]]
        aggregates.append(
            _aggregate_model(
                model=model,
                levels=levels,
                rows=model_rows,
                attempts_per_level=attempts_per_level,
            )
        )
    selection = None
    if args.purpose == "selection":
        ranked = sorted(
            enumerate(aggregates),
            key=lambda item: _selection_key(item[1], item[0]),
            reverse=True,
        )
        selection = {
            "selected_model_id": ranked[0][1]["model"]["id"],
            "selected_model": ranked[0][1]["model"],
            "ranking": [
                {
                    "rank": rank,
                    "model_id": summary["model"]["id"],
                    "key": list(_selection_key(summary, index)),
                }
                for rank, (index, summary) in enumerate(ranked, start=1)
            ],
        }

    aggregate = {
        "schema": "zuma-rl.paired-multimodel-evaluation-aggregate",
        "version": 1,
        "status": "COMPLETE",
        "created_utc": datetime.now().astimezone().isoformat(),
        "purpose": args.purpose,
        "validation": {
            "all_shards_complete": True,
            "attempt_matrix_exact": True,
            "paired_seed_matrix_exact": True,
            "model_identity_exact": True,
            "evaluator_identity_exact": True,
        },
        "preregistration": {"path": str(prereg_path), "sha256": prereg_sha},
        "models_manifest": {"path": str(manifest_path), "sha256": manifest_sha},
        "shards": shard_receipts,
        "levels": len(levels),
        "attempts_per_level": attempts_per_level,
        "models": aggregates,
        "selection": selection,
    }
    aggregate_path = run_directory / "aggregate.json"
    summary_path = run_directory / "SUMMARY.md"
    _write_atomic(
        aggregate_path,
        (json.dumps(aggregate, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"),
    )
    _write_atomic(summary_path, _markdown(aggregate).encode("utf-8"))
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "aggregate": str(aggregate_path),
                "aggregate_sha256": _sha256(aggregate_path),
                "summary": str(summary_path),
                "summary_sha256": _sha256(summary_path),
                "selected_model_id": selection["selected_model_id"] if selection else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
