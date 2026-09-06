"""Freeze five-route training evidence for an AlphaZuma 55 rescue curriculum."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any


DUAL_FIXED_LEVELS = {
    "Jungle9",
    "village6",
    "city4",
    "city10",
    "Coast10",
    "grotto10",
    "volcano9",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _write_json_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError:
        raise FileExistsError(f"refusing to overwrite frontier analysis: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _rescue_weight(
    *,
    level_id: str,
    episodes: int,
    wins: int,
    truncations: int,
) -> float:
    if wins > 0:
        return 1.0 if level_id == "Jungle2" else 0.5
    if episodes >= 60:
        value = 3.5
    elif episodes >= 40:
        value = 3.0
    else:
        value = 2.5
    if level_id in DUAL_FIXED_LEVELS:
        value += 0.75
    if episodes > 0 and truncations / episodes >= 0.75:
        value += 0.5
    return min(value, 4.75)


def analyze(
    *,
    expansion_path: Path,
    expected_expansion_sha256: str,
    route_paths: list[Path],
    source_route_id: str,
    target_checkpoint_steps: int,
) -> dict[str, Any]:
    _require(_sha256(expansion_path) == expected_expansion_sha256, "expansion hash mismatch")
    expansion = _read_json(expansion_path)
    _require(
        expansion.get("schema")
        == "zuma-rl.alphazuma-55-capacity-expansion-preregistration"
        and expansion.get("status") == "FROZEN_BEFORE_REPLICATE_TRAINING",
        "unexpected capacity expansion",
    )
    _require(len(route_paths) == 5, "frontier analysis requires exactly five routes")

    level_order: list[str] | None = None
    aggregate: dict[str, dict[str, Any]] = {}
    route_receipts: list[dict[str, Any]] = []
    source_route: dict[str, Any] | None = None
    for route_path in route_paths:
        route = _read_json(route_path)
        _require(
            route.get("schema") == "zuma-rl.overnight-multilevel-preregistration"
            and route.get("status") == "FROZEN_BEFORE_TRAINING",
            f"route is not frozen: {route_path}",
        )
        runs = route.get("runs")
        _require(isinstance(runs, list) and len(runs) == 1, "each route must have one run")
        run = runs[0]
        if str(run["id"]) == source_route_id:
            source_route = route
        ids = [str(row["id"]) for row in route["levels"]]
        if level_order is None:
            level_order = ids
        _require(ids == level_order and len(ids) == 55, "route level order differs")
        run_dir = Path(str(run["run_dir"])).resolve(strict=True)
        failure_path = run_dir / "failure.json"
        status_path = run_dir / "training_status.json"
        _require(not failure_path.exists(), f"route has failure receipt: {run['id']}")
        status = _read_json(status_path.resolve(strict=True))
        _require(status.get("status") == "RUNNING", f"route is not running: {run['id']}")
        per_level = status.get("per_level")
        _require(isinstance(per_level, dict) and set(per_level) == set(ids), "invalid per-level status")
        route_receipts.append(
            {
                "run_id": run["id"],
                "preregistration": {"path": str(route_path), "sha256": _sha256(route_path)},
                "training_status_snapshot": {
                    "path": str(status_path),
                    "sha256": _sha256(status_path),
                    "updated_utc": status["updated_utc"],
                    "steps_this_run": int(status["steps_this_run"]),
                    "steps_per_second": float(status["steps_per_second"]),
                },
            }
        )
        for level_id in ids:
            value = per_level[level_id]
            row = aggregate.setdefault(
                level_id,
                {
                    "level_id": level_id,
                    "episodes": 0,
                    "wins": 0,
                    "losses": 0,
                    "truncations": 0,
                    "winning_routes": 0,
                },
            )
            row["episodes"] += int(value["episodes"])
            row["wins"] += int(value["wins"])
            row["losses"] += int(value["losses"])
            row["truncations"] += int(value["truncations"])
            row["winning_routes"] += int(value["wins"]) > 0

    _require(source_route is not None, "source route id was not found")
    assert level_order is not None
    rows = [aggregate[level_id] for level_id in level_order]
    for row in rows:
        row["rescue_initial_weight"] = _rescue_weight(
            level_id=row["level_id"],
            episodes=int(row["episodes"]),
            wins=int(row["wins"]),
            truncations=int(row["truncations"]),
        )
    zero_win = [row for row in rows if int(row["wins"]) == 0]
    any_win = [row for row in rows if int(row["wins"]) > 0]
    source_run = source_route["runs"][0]
    source_run_dir = Path(str(source_run["run_dir"])).resolve(strict=True)
    checkpoint_name = f"{source_run['id']}_{target_checkpoint_steps}_steps.zip"
    target_path = source_run_dir / "checkpoints" / checkpoint_name
    return {
        "schema": "zuma-rl.alphazuma-55-curriculum-frontier-analysis",
        "version": 1,
        "status": "FROZEN_BEFORE_SOURCE_CHECKPOINT",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_id": expansion["campaign_id"],
        "capacity_expansion": {"path": str(expansion_path), "sha256": expected_expansion_sha256},
        "route_status_snapshots": route_receipts,
        "summary": {
            "levels": len(rows),
            "levels_with_any_training_win": len(any_win),
            "zero_win_levels": len(zero_win),
            "total_episodes": sum(int(row["episodes"]) for row in rows),
            "zero_win_level_episodes": sum(int(row["episodes"]) for row in zero_win),
        },
        "per_level": rows,
        "rescue_initial_weights": {
            row["level_id"]: row["rescue_initial_weight"] for row in rows
        },
        "source_checkpoint": {
            "source_route_id": source_route_id,
            "source_route_preregistration": {
                "path": str(next(path for path in route_paths if _read_json(path)["runs"][0]["id"] == source_route_id)),
                "sha256": _sha256(next(path for path in route_paths if _read_json(path)["runs"][0]["id"] == source_route_id)),
            },
            "training_steps": target_checkpoint_steps,
            "expected_path": str(target_path),
            "must_be_absent_at_freeze": not target_path.exists(),
            "sha256_frozen_only_after_file_exists": True,
        },
        "rescue_contract": {
            "single_policy": True,
            "solved_level_replay_floor_weight": 0.5,
            "Jungle2_anchor_weight": 1.0,
            "zero_win_weight_range": [2.5, 4.75],
            "dual_fixed_bonus": 0.75,
            "truncation_dominant_bonus": 0.5,
            "reward_environment_input_and_actor_interfaces_unchanged": True,
        },
        "authority_boundary": {
            "formal_selection_authority": False,
            "formal_seed_consumption": False,
            "may_freeze_one_sixth_route_after_exact_checkpoint_exists": True,
            "may_not_change_other_routes": True,
            "may_not_change_gate_or_ranking": True,
        },
        "builder": {"path": str(Path(__file__).resolve()), "sha256": _sha256(Path(__file__).resolve())},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expansion-preregistration", required=True, type=Path)
    parser.add_argument("--expected-expansion-sha256", required=True)
    parser.add_argument("--training-route", required=True, action="append", type=Path)
    parser.add_argument("--source-route-id", required=True)
    parser.add_argument("--target-checkpoint-steps", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frontier analysis: {output}")
    value = analyze(
        expansion_path=args.expansion_preregistration.expanduser().resolve(strict=True),
        expected_expansion_sha256=args.expected_expansion_sha256,
        route_paths=[path.expanduser().resolve(strict=True) for path in args.training_route],
        source_route_id=args.source_route_id,
        target_checkpoint_steps=args.target_checkpoint_steps,
    )
    _write_json_exclusive(output, value)
    print(
        json.dumps(
            {
                "status": value["status"],
                "output": str(output),
                "sha256": _sha256(output),
                "summary": value["summary"],
                "source_checkpoint": value["source_checkpoint"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
