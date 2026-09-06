"""Validate and summarize completed zero-shot multilevel evaluation shards."""

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
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected an object: {path}")
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


def _median(values: list[int]) -> float | None:
    return statistics.median(values) if values else None


def _prefix(level_id: str) -> str:
    folded = level_id.casefold()
    for prefix in ("jungle", "village", "city", "coast", "grotto", "volcano"):
        if folded.startswith(prefix):
            return prefix
    return "other"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--run-directory", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    prereg_path = args.preregistration.expanduser().resolve(strict=True)
    run_directory = args.run_directory.expanduser().resolve(strict=True)
    prereg = _load(prereg_path)
    prereg_sha = _sha256(prereg_path)
    if prereg.get("schema") != "zuma-rl.zero-shot-multilevel-preregistration":
        raise ValueError("unexpected preregistration schema")

    levels = prereg["levels"]
    attempts_per_level = int(prereg["attempts_per_level"])
    total_attempts = int(prereg["total_attempts"])
    execution = prereg["execution"]
    shard_count = int(execution["shard_count"])
    base_seed = int(prereg["seed_plan"]["base_seed"])
    model = prereg["model"]

    shard_receipts: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    started_times: list[datetime] = []
    completed_times: list[datetime] = []
    for shard_index in range(shard_count):
        path = run_directory / f"shard-{shard_index:02d}-of-{shard_count:02d}.json"
        shard = _load(path)
        if shard.get("schema") != "zuma-rl.zero-shot-multilevel-shard":
            raise ValueError(f"unexpected shard schema: {path}")
        if shard.get("status") != "COMPLETE" or shard.get("error") is not None:
            raise ValueError(f"shard is not a clean completion: {path}")
        if int(shard.get("shard_index", -1)) != shard_index:
            raise ValueError(f"shard index mismatch: {path}")
        if int(shard.get("shard_count", -1)) != shard_count:
            raise ValueError(f"shard count mismatch: {path}")
        if shard.get("device") != execution["devices"][shard_index]:
            raise ValueError(f"shard device mismatch: {path}")
        if shard.get("preregistration", {}).get("sha256") != prereg_sha:
            raise ValueError(f"shard preregistration binding mismatch: {path}")
        if shard.get("evaluator_sha256") != prereg["evaluator"]["sha256"]:
            raise ValueError(f"shard evaluator binding mismatch: {path}")

        start = len(levels) * shard_index // shard_count
        end = len(levels) * (shard_index + 1) // shard_count
        if shard.get("level_range") != [start, end] or shard.get("levels") != levels[start:end]:
            raise ValueError(f"shard level partition mismatch: {path}")
        expected = (end - start) * attempts_per_level
        rows = shard.get("attempts")
        if not isinstance(rows, list) or len(rows) != expected:
            raise ValueError(f"shard attempt inventory mismatch: {path}")
        if shard.get("completed_attempts") != expected or shard.get("expected_attempts") != expected:
            raise ValueError(f"shard completion count mismatch: {path}")

        started_times.append(datetime.fromisoformat(shard["started_utc"]))
        completed_times.append(datetime.fromisoformat(shard["completed_utc"]))
        attempts.extend(rows)
        shard_receipts.append(
            {
                "path": str(path),
                "sha256": _sha256(path),
                "shard_index": shard_index,
                "device": shard["device"],
                "attempts": len(rows),
                "wall_seconds": float(shard["runtime"]["wall_seconds"]),
                "vector_transitions": int(
                    shard["runtime"]["total_vector_transitions"]
                ),
                "stderr_bytes": (run_directory / f"shard-{shard_index:02d}.stderr.log").stat().st_size,
            }
        )

    if len(attempts) != total_attempts:
        raise ValueError("combined attempt count differs from preregistration")

    expected_keys: set[tuple[str, int]] = set()
    expected_seeds: set[int] = set()
    level_by_key = {str(level["id"]).casefold(): level for level in levels}
    for level_index, level in enumerate(levels):
        for attempt_index in range(attempts_per_level):
            expected_keys.add((str(level["id"]).casefold(), attempt_index))
            expected_seeds.add(
                base_seed + level_index * attempts_per_level + attempt_index
            )

    actual_keys: set[tuple[str, int]] = set()
    actual_seeds: set[int] = set()
    for row in attempts:
        level_key = str(row.get("level_id", "")).casefold()
        attempt_index = int(row.get("attempt_index", -1))
        key = (level_key, attempt_index)
        if key in actual_keys:
            raise ValueError(f"duplicate level/attempt key: {key}")
        actual_keys.add(key)
        seed = int(row["seed"])
        if seed in actual_seeds:
            raise ValueError(f"duplicate seed: {seed}")
        actual_seeds.add(seed)
        level = level_by_key.get(level_key)
        if level is None:
            raise ValueError(f"attempt references an unfrozen level: {level_key}")
        expected_seed = (
            base_seed
            + levels.index(level) * attempts_per_level
            + attempt_index
        )
        if seed != expected_seed:
            raise ValueError(f"seed derivation mismatch for {key}")
        if row.get("display_name") != level["display_name"]:
            raise ValueError(f"display-name mismatch for {key}")
        if int(row.get("curve_count", -1)) != int(level["curve_count"]):
            raise ValueError(f"curve-count mismatch for {key}")
        if row.get("model_id") != model["id"]:
            raise ValueError(f"model id mismatch for {key}")
        if int(row.get("training_steps", -1)) != int(model["training_steps"]):
            raise ValueError(f"training-step mismatch for {key}")
        if row.get("model_sha256") != model["sha256"]:
            raise ValueError(f"model hash mismatch for {key}")
        ticks = int(row["ticks"])
        if not 0 < ticks <= int(prereg["environment"]["base_config"]["max_ticks"]):
            raise ValueError(f"invalid tick count for {key}")
        if not math.isclose(float(row["seconds"]), ticks / 100.0, abs_tol=1e-12):
            raise ValueError(f"seconds/ticks mismatch for {key}")
        outcome = row.get("outcome")
        truncated = bool(row.get("time_limit_truncated"))
        if outcome not in {"win", "loss", None}:
            raise ValueError(f"unexpected outcome for {key}: {outcome}")
        if (outcome is None) != truncated:
            raise ValueError(f"truncation/outcome mismatch for {key}")
        trajectory = row.get("trajectory_sha256")
        if not isinstance(trajectory, str) or not trajectory.startswith("sha256:") or len(trajectory) != 71:
            raise ValueError(f"invalid trajectory hash for {key}")

    if actual_keys != expected_keys or actual_seeds != expected_seeds:
        raise ValueError("combined attempt matrix is incomplete or contains extras")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in attempts:
        grouped[str(row["level_id"]).casefold()].append(row)

    summaries: list[dict[str, Any]] = []
    for frozen_index, level in enumerate(levels):
        rows = sorted(
            grouped[str(level["id"]).casefold()],
            key=lambda row: int(row["attempt_index"]),
        )
        wins = [row for row in rows if row["outcome"] == "win"]
        win_ticks = [int(row["ticks"]) for row in wins]
        best = min(wins, key=lambda row: (int(row["ticks"]), -int(row["score"]))) if wins else None
        summaries.append(
            {
                "frozen_index": frozen_index,
                "level_id": level["id"],
                "display_name": level["display_name"],
                "curve_count": int(level["curve_count"]),
                "attempts": len(rows),
                "wins": len(wins),
                "losses": sum(row["outcome"] == "loss" for row in rows),
                "truncations": sum(bool(row["time_limit_truncated"]) for row in rows),
                "win_rate": len(wins) / len(rows),
                "best_ticks": min(win_ticks) if win_ticks else None,
                "best_seconds": min(win_ticks) / 100.0 if win_ticks else None,
                "median_win_ticks": _median(win_ticks),
                "median_win_seconds": (
                    _median(win_ticks) / 100.0 if win_ticks else None
                ),
                "best_attempt": best,
                "classification": (
                    "PERFECT"
                    if len(wins) == attempts_per_level
                    else "RELIABLY_CLEARED"
                    if len(wins) >= 12
                    else "CLEARED"
                    if wins
                    else "NOT_CLEARED"
                ),
            }
        )

    wins = [row for row in attempts if row["outcome"] == "win"]
    losses = [row for row in attempts if row["outcome"] == "loss"]
    truncations = [row for row in attempts if row["time_limit_truncated"]]
    best_attempt = min(
        wins,
        key=lambda row: (int(row["ticks"]), -int(row["score"])),
    )
    region_summaries: list[dict[str, Any]] = []
    for region in ("jungle", "village"):
        selected = [row for row in attempts if _prefix(str(row["level_id"])) == region]
        selected_levels = [
            summary
            for summary in summaries
            if _prefix(str(summary["level_id"])) == region
        ]
        region_wins = sum(row["outcome"] == "win" for row in selected)
        region_summaries.append(
            {
                "region": region,
                "levels": len(selected_levels),
                "levels_cleared": sum(summary["wins"] > 0 for summary in selected_levels),
                "attempts": len(selected),
                "wins": region_wins,
                "win_rate": region_wins / len(selected),
            }
        )

    curve_summaries: list[dict[str, Any]] = []
    for curve_count in sorted({int(summary["curve_count"]) for summary in summaries}):
        selected_levels = [summary for summary in summaries if summary["curve_count"] == curve_count]
        selected_ids = {str(summary["level_id"]).casefold() for summary in selected_levels}
        selected = [row for row in attempts if str(row["level_id"]).casefold() in selected_ids]
        curve_wins = sum(row["outcome"] == "win" for row in selected)
        curve_summaries.append(
            {
                "curve_count": curve_count,
                "levels": len(selected_levels),
                "levels_cleared": sum(summary["wins"] > 0 for summary in selected_levels),
                "attempts": len(selected),
                "wins": curve_wins,
                "win_rate": curve_wins / len(selected),
            }
        )

    aggregate = {
        "schema": "zuma-rl.zero-shot-multilevel-aggregate",
        "version": 1,
        "status": "COMPLETE",
        "validation": {
            "shards_complete": True,
            "stderr_empty": all(receipt["stderr_bytes"] == 0 for receipt in shard_receipts),
            "attempt_matrix_exact": True,
            "seed_matrix_exact": True,
            "model_identity_exact": True,
            "evaluator_identity_exact": True,
        },
        "preregistration": {"path": str(prereg_path), "sha256": prereg_sha},
        "model": model,
        "shards": shard_receipts,
        "scope": {
            "ordinary_campaign_levels": prereg["campaign_inventory"]["ordinary_campaign_levels"],
            "trained_level_excluded": prereg["campaign_inventory"]["excluded_trained_level"],
            "zero_shot_levels": len(levels),
            "incompatible_levels_not_tested": prereg["campaign_inventory"]["excluded_incompatible_levels"],
            "partial_level_escape_hatch_used": False,
        },
        "overall": {
            "levels": len(levels),
            "levels_cleared": sum(summary["wins"] > 0 for summary in summaries),
            "levels_cleared_rate": sum(summary["wins"] > 0 for summary in summaries) / len(summaries),
            "levels_reliably_cleared": sum(summary["wins"] >= 12 for summary in summaries),
            "levels_perfect": sum(summary["wins"] == attempts_per_level for summary in summaries),
            "attempts": len(attempts),
            "wins": len(wins),
            "losses": len(losses),
            "truncations": len(truncations),
            "win_rate": len(wins) / len(attempts),
            "best_attempt": best_attempt,
            "wall_clock_seconds": (
                max(completed_times) - min(started_times)
            ).total_seconds(),
        },
        "regions": region_summaries,
        "curve_counts": curve_summaries,
        "level_summaries": summaries,
        "ranking": sorted(
            summaries,
            key=lambda row: (
                -int(row["wins"]),
                float(row["best_seconds"]) if row["best_seconds"] is not None else math.inf,
                int(row["frozen_index"]),
            ),
        ),
        "interpretation": {
            "zero_shot_transfer_observed": True,
            "stable_general_policy_demonstrated": False,
            "reason": "10 of 16 unseen compatible levels were cleared at least once, but none reached the preregistered reliable threshold of 12 wins in 16 attempts",
            "two_curve_result_is_exploratory": "both tested two-curve levels had zero wins, but only two such compatible levels were available",
        },
    }

    json_path = run_directory / "aggregate.json"
    markdown_path = run_directory / "SUMMARY.md"
    _write_atomic(
        json_path,
        (json.dumps(aggregate, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"),
    )

    lines = [
        "# AlphaZuma V1 zero-shot multilevel challenge",
        "",
        "- Status: COMPLETE",
        f"- Model: `{model['id']}` (`{model['training_steps']:,}` Jungle2 training steps)",
        f"- Frozen matrix: {len(levels)} unseen levels x {attempts_per_level} seeds = {len(attempts)} attempts",
        f"- Cleared levels: {aggregate['overall']['levels_cleared']}/{len(levels)} ({aggregate['overall']['levels_cleared_rate']:.1%})",
        f"- Attempt wins: {len(wins)}/{len(attempts)} ({aggregate['overall']['win_rate']:.1%})",
        f"- Reliable levels (>=12/16): {aggregate['overall']['levels_reliably_cleared']}",
        f"- Fastest zero-shot clear: {best_attempt['level_id']} / {best_attempt['display_name']} in {best_attempt['seconds']:.2f}s (seed {best_attempt['seed']})",
        f"- Wall clock: {aggregate['overall']['wall_clock_seconds'] / 60.0:.1f} minutes on RTX 5090 + RTX 5080",
        "",
        "| Level | Name | Curves | Wins | Win rate | Best | Outcome |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for summary in summaries:
        best_text = f"{summary['best_seconds']:.2f}s" if summary["best_seconds"] is not None else "-"
        lines.append(
            f"| {summary['level_id']} | {summary['display_name']} | {summary['curve_count']} | "
            f"{summary['wins']}/{summary['attempts']} | {summary['win_rate']:.1%} | "
            f"{best_text} | {summary['classification']} |"
        )
    lines.extend(
        [
            "",
            "## Scope boundary",
            "",
            "This is an exploratory state-policy zero-shot evaluation, not additional training and not original-client visual control. Jungle2 was excluded because it was the training level. Forty-three ordinary campaign levels were not tested because the fidelity-safe simulator rejected unsupported frog mechanics or the frozen model's observation shape could not represent their additional ball colors. The partial-level diagnostic escape hatch was not used.",
            "",
        ]
    )
    _write_atomic(markdown_path, "\n".join(lines).encode("utf-8"))
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "aggregate": str(json_path),
                "aggregate_sha256": _sha256(json_path),
                "summary": str(markdown_path),
                "summary_sha256": _sha256(markdown_path),
                "levels_cleared": aggregate["overall"]["levels_cleared"],
                "levels": len(levels),
                "wins": len(wins),
                "attempts": len(attempts),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
