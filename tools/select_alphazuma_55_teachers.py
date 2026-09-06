"""Freeze the strongest engineering-baseline teacher for each of 55 levels."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _rank(rows: list[dict[str, Any]]) -> tuple[Any, ...]:
    wins = [row for row in rows if row.get("outcome") == "win"]
    scores = [int(row.get("score", 0)) for row in rows]
    win_ticks = [int(row["ticks"]) for row in wins]
    return (
        len(wins),
        max(scores, default=0),
        -statistics.median(win_ticks) if win_ticks else float("-inf"),
    )


def select_teachers(
    *,
    prereg_path: Path,
    manifest_path: Path,
    shard_paths: list[Path],
) -> dict[str, Any]:
    prereg = _read_json(prereg_path)
    manifest = _read_json(manifest_path)
    if prereg.get("stage") != "engineering":
        raise ValueError("teacher selection requires an engineering matrix")
    if manifest.get("status") != "FROZEN":
        raise ValueError("models manifest is not frozen")
    model_specs = {model["id"]: model for model in manifest["models"]}
    if not model_specs:
        raise ValueError("models manifest is empty")

    attempts: list[dict[str, Any]] = []
    shard_receipts: list[dict[str, Any]] = []
    seen_shards: set[int] = set()
    for path in shard_paths:
        shard = _read_json(path)
        if shard.get("status") != "COMPLETE" or shard.get("error") is not None:
            raise ValueError(f"evaluation shard is not complete: {path}")
        if shard.get("preregistration", {}).get("sha256") != _sha256(prereg_path):
            raise ValueError("shard preregistration hash differs")
        if shard.get("models_manifest", {}).get("sha256") != _sha256(manifest_path):
            raise ValueError("shard models-manifest hash differs")
        shard_index = int(shard["shard_index"])
        if shard_index in seen_shards:
            raise ValueError("duplicate shard index")
        seen_shards.add(shard_index)
        if int(shard["completed_attempts"]) != int(shard["expected_attempts"]):
            raise ValueError("shard attempt count is incomplete")
        attempts.extend(shard["attempts"])
        shard_receipts.append(
            {"path": str(path), "sha256": _sha256(path), "index": shard_index}
        )
    expected_shards = int(prereg["execution"]["shard_count"])
    if seen_shards != set(range(expected_shards)):
        raise ValueError("evaluation shard set is incomplete")

    expected_rows = (
        len(prereg["levels"])
        * int(prereg["attempts_per_level"])
        * len(model_specs)
    )
    if len(attempts) != expected_rows:
        raise ValueError("combined attempt count differs from frozen matrix")
    keys = [
        (row["model_id"], row["level_id"], int(row["attempt_index"]))
        for row in attempts
    ]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate model/level/attempt rows")

    selected: list[dict[str, Any]] = []
    model_selection_counts = {model_id: 0 for model_id in model_specs}
    proven_count = 0
    for level in prereg["levels"]:
        level_id = str(level["id"])
        candidates: list[dict[str, Any]] = []
        for model_id, model in model_specs.items():
            rows = [
                row
                for row in attempts
                if row["level_id"].casefold() == level_id.casefold()
                and row["model_id"] == model_id
            ]
            if len(rows) != int(prereg["attempts_per_level"]):
                raise ValueError(f"incomplete candidate rows for {model_id}/{level_id}")
            candidates.append(
                {
                    "model_id": model_id,
                    "model": model,
                    "rows": rows,
                    "rank": _rank(rows),
                }
            )
        winner = min(
            candidates,
            key=lambda candidate: (
                tuple(-value for value in candidate["rank"]),
                candidate["model"]["sha256"],
            ),
        )
        rows = winner["rows"]
        wins = [row for row in rows if row["outcome"] == "win"]
        proven = bool(wins)
        proven_count += int(proven)
        model_selection_counts[winner["model_id"]] += 1
        selected.append(
            {
                "level_id": level_id,
                "display_name": level["display_name"],
                "teacher_model": winner["model"],
                "selection_basis": "engineering_win" if proven else "best_engineering_score",
                "engineering_wins": len(wins),
                "engineering_attempts": len(rows),
                "best_ticks": min((int(row["ticks"]) for row in wins), default=None),
                "best_score": max(int(row.get("score", 0)) for row in rows),
                "geometric_teacher_hop_override_allowed": True,
            }
        )

    return {
        "schema": "zuma-rl.alphazuma-55-teacher-map",
        "version": 1,
        "status": "FROZEN",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_id": prereg["campaign_id"],
        "source_stage": "engineering",
        "preregistration": {
            "path": str(prereg_path),
            "sha256": _sha256(prereg_path),
        },
        "models_manifest": {
            "path": str(manifest_path),
            "sha256": _sha256(manifest_path),
        },
        "shards": sorted(shard_receipts, key=lambda item: item["index"]),
        "selection_rule": [
            "engineering wins descending",
            "best score descending",
            "median winning ticks ascending",
            "model sha256 ascending",
        ],
        "teacher_proven_level_count": proven_count,
        "teacher_unproven_level_count": len(selected) - proven_count,
        "model_selection_counts": model_selection_counts,
        "levels": selected,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--models-manifest", required=True, type=Path)
    parser.add_argument("--shard", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite teacher map: {output}")
    result = select_teachers(
        prereg_path=args.preregistration.expanduser().resolve(strict=True),
        manifest_path=args.models_manifest.expanduser().resolve(strict=True),
        shard_paths=[path.expanduser().resolve(strict=True) for path in args.shard],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(output),
                "sha256": _sha256(output),
                "teacher_proven_level_count": result[
                    "teacher_proven_level_count"
                ],
                "model_selection_counts": result["model_selection_counts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
