"""Freeze one engineering-only distilled hard-frontier PPO rescue route.

The builder snapshots current per-level training evidence, binds an immutable
checkpoint, and emits a normal ``overnight-multilevel`` preregistration.  It
does not grant formal-selection authority or consume any evaluation seed.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools import build_alphazuma_55_hard_frontier_route as base


DUAL_FIXED_LEVELS = {
    "Jungle9",
    "village6",
    "city4",
    "city10",
    "Coast10",
    "grotto10",
    "volcano9",
}


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _snapshot_json(path: Path) -> tuple[dict[str, Any], str]:
    payload = path.read_bytes()
    value = json.loads(payload.decode("utf-8-sig"))
    base._require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value, _sha256_bytes(payload)


def _route_range(route: dict[str, Any]) -> tuple[str, int, int]:
    run = route["runs"][0]
    return str(run["id"]), int(run["episode_seed_base"]), int(run["episode_seed_last"])


def _rescue_weight(
    *,
    level_id: str,
    source: dict[str, Any],
    aggregate: dict[str, int],
) -> float:
    source_wins = int(source["wins"])
    source_recent = float(source["recent_win_rate"])
    aggregate_wins = int(aggregate["wins"])
    aggregate_episodes = int(aggregate["episodes"])
    aggregate_truncations = int(aggregate["truncations"])
    source_episodes = int(source["episodes"])
    source_truncations = int(source["truncations"])

    # Keep a materially larger rehearsal floor than the first rescue route.
    # Fragile solved levels get extra replay, while mastered levels still
    # remain present in every rollout distribution.
    if source_wins > 0:
        return 1.0 if source_recent >= 0.25 else 1.5

    # A level solved by a sibling PPO route is a plausible transfer frontier,
    # not a completely cold task for the distilled checkpoint.
    if aggregate_wins > 0:
        value = 2.75
    else:
        value = 4.0 if aggregate_episodes >= 64 else 3.5

    if level_id in DUAL_FIXED_LEVELS:
        value += 0.75
    source_truncation_rate = (
        source_truncations / source_episodes if source_episodes else 0.0
    )
    aggregate_truncation_rate = (
        aggregate_truncations / aggregate_episodes if aggregate_episodes else 0.0
    )
    if max(source_truncation_rate, aggregate_truncation_rate) >= 0.65:
        value += 0.75
    return min(value, 6.5)


def _validate_route(route: dict[str, Any], path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    base._require(
        route.get("schema") == "zuma-rl.overnight-multilevel-preregistration"
        and route.get("status") == "FROZEN_BEFORE_TRAINING",
        f"route is not frozen: {path}",
    )
    runs = route.get("runs")
    base._require(isinstance(runs, list) and len(runs) == 1, "each route must have one run")
    run = runs[0]
    level_ids = [str(row["id"]) for row in route["levels"]]
    base._require(len(level_ids) == 55 and len(set(level_ids)) == 55, "route must contain 55 unique levels")
    return run, {level_id: row for level_id, row in zip(level_ids, route["levels"], strict=True)}


def build(
    *,
    master_path: Path,
    route_paths: list[Path],
    source_route_path: Path,
    checkpoint_path: Path,
    checkpoint_steps: int,
    route_id: str,
    run_dir: Path,
    device: str,
    model_seed: int,
    episode_seed_base: int,
    num_envs: int,
    analysis_output: Path,
    route_output: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    base._require(len(route_paths) >= 6, "rescue analysis requires all active PPO routes")
    base._require(source_route_path in route_paths, "source route is absent from route inventory")
    base._require(device in {"cuda:0", "cuda:1"} and num_envs > 0, "invalid execution topology")
    base._require(checkpoint_steps > 0, "checkpoint steps must be positive")
    base._require(not run_dir.exists(), "rescue run directory already exists")
    base._require(not analysis_output.exists(), "analysis output already exists")
    base._require(not route_output.exists(), "route preregistration already exists")

    master, master_sha256 = _snapshot_json(master_path)
    base._require(
        master.get("schema")
        == "zuma-rl.alphazuma-55-weekend-master-preregistration"
        and isinstance(master.get("seed_registry"), dict)
        and isinstance(master.get("schedule"), dict),
        "unexpected campaign master",
    )
    checkpoint_path = checkpoint_path.resolve(strict=True)
    checkpoint_sha256 = base._sha256(checkpoint_path)
    match = base.CHECKPOINT_RE.search(checkpoint_path.name)
    base._require(
        match is not None and int(match.group(1)) == checkpoint_steps,
        "checkpoint filename and step counter differ",
    )

    route_receipts: list[dict[str, Any]] = []
    route_values: dict[Path, dict[str, Any]] = {}
    level_order: list[str] | None = None
    aggregate: dict[str, dict[str, int]] = {}
    existing_ranges: list[tuple[str, int, int]] = []
    source_status: dict[str, Any] | None = None
    source_route: dict[str, Any] | None = None

    for route_path in route_paths:
        route_path = route_path.resolve(strict=True)
        route, route_sha256 = _snapshot_json(route_path)
        run, _ = _validate_route(route, route_path)
        ids = [str(row["id"]) for row in route["levels"]]
        if level_order is None:
            level_order = ids
        base._require(ids == level_order, "route level order differs")
        existing_ranges.append(_route_range(route))
        run_root = Path(str(run["run_dir"])).resolve(strict=True)
        failure = run_root / "failure.json"
        base._require(not failure.exists(), f"route has failure receipt: {run['id']}")
        status_path = (run_root / "training_status.json").resolve(strict=True)
        status, status_sha256 = _snapshot_json(status_path)
        base._require(status.get("status") == "RUNNING", f"route is not running: {run['id']}")
        per_level = status.get("per_level")
        base._require(isinstance(per_level, dict) and list(per_level) == ids, "per-level status differs")
        route_receipts.append(
            {
                "run_id": run["id"],
                "preregistration": {"path": str(route_path), "sha256": route_sha256},
                "training_status_snapshot": {
                    "path": str(status_path),
                    "sha256": status_sha256,
                    "updated_utc": status["updated_utc"],
                    "timesteps": int(status["timesteps"]),
                    "steps_this_run": int(status["steps_this_run"]),
                    "steps_per_second": float(status["steps_per_second"]),
                },
            }
        )
        for level_id in ids:
            value = per_level[level_id]
            row = aggregate.setdefault(
                level_id,
                {"episodes": 0, "wins": 0, "losses": 0, "truncations": 0, "winning_routes": 0},
            )
            for key in ("episodes", "wins", "losses", "truncations"):
                row[key] += int(value[key])
            row["winning_routes"] += int(int(value["wins"]) > 0)
        if route_path == source_route_path.resolve(strict=True):
            source_route = route
            source_status = status
        route_values[route_path] = route

    base._require(source_route is not None and source_status is not None, "source route was not found")
    assert level_order is not None
    source_run = source_route["runs"][0]
    expected_checkpoint = (
        Path(str(source_run["run_dir"])).resolve(strict=True)
        / "checkpoints"
        / f"{source_run['id']}_{checkpoint_steps}_steps.zip"
    )
    base._require(checkpoint_path == expected_checkpoint, "checkpoint is outside the source route")

    candidate_last = episode_seed_base + num_envs * 250_000 - 1
    registry = master["seed_registry"]["training"]
    base._require(
        int(registry["first"]) <= episode_seed_base <= candidate_last <= int(registry["last"]),
        "rescue seed namespace is outside the master training registry",
    )
    base._require_disjoint((route_id, episode_seed_base, candidate_last), existing_ranges)

    source_per_level = source_status["per_level"]
    weights = {
        level_id: _rescue_weight(
            level_id=level_id,
            source=source_per_level[level_id],
            aggregate=aggregate[level_id],
        )
        for level_id in level_order
    }
    rows = []
    for level_id in level_order:
        row = {
            "level_id": level_id,
            **aggregate[level_id],
            "source_episodes": int(source_per_level[level_id]["episodes"]),
            "source_wins": int(source_per_level[level_id]["wins"]),
            "source_recent_win_rate": float(source_per_level[level_id]["recent_win_rate"]),
            "rescue_initial_weight": weights[level_id],
        }
        rows.append(row)

    now = datetime.now(timezone.utc).isoformat()
    analysis = {
        "schema": "zuma-rl.alphazuma-55-distilled-hard-frontier-analysis",
        "version": 1,
        "status": "FROZEN_AFTER_SOURCE_CHECKPOINT_BEFORE_RESCUE_TRAINING",
        "created_utc": now,
        "campaign_id": master["campaign_id"],
        "master_preregistration": {"path": str(master_path), "sha256": master_sha256},
        "route_status_snapshots": route_receipts,
        "summary": {
            "levels": 55,
            "routes": len(route_receipts),
            "source_levels_with_any_win": sum(int(row["source_wins"]) > 0 for row in rows),
            "union_levels_with_any_win": sum(int(row["wins"]) > 0 for row in rows),
            "source_zero_win_levels": sum(int(row["source_wins"]) == 0 for row in rows),
        },
        "per_level": rows,
        "rescue_initial_weights": weights,
        "source_checkpoint": {
            "source_route_id": source_run["id"],
            "source_route_preregistration": {
                "path": str(source_route_path),
                "sha256": base._sha256(source_route_path),
            },
            "training_steps": checkpoint_steps,
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha256,
        },
        "rescue_contract": {
            "single_policy": True,
            "environment_network_reward_and_ppo_hyperparameters_unchanged": True,
            "solved_level_replay_floor_weight": 1.0,
            "fragile_solved_level_weight": 1.5,
            "sibling_solved_source_unsolved_weight": 2.75,
            "cold_frontier_weight_range": [3.5, 6.5],
            "adaptive_weight_ceiling_unchanged": float(source_run["adaptive"]["maximum_weight"]),
        },
        "authority_boundary": {
            "classification": "engineering_training_only",
            "formal_selection_authority": False,
            "formal_seed_consumption": False,
            "may_train_one_additional_distilled_rescue_route": True,
            "promotion_requires_a_separate_frozen_successor_contract": True,
            "may_not_change_existing_routes_or_gates": True,
        },
        "builder": {"path": str(Path(__file__).resolve()), "sha256": base._sha256(Path(__file__).resolve())},
    }
    base._write_json_exclusive(analysis_output, analysis)
    analysis_sha256 = base._sha256(analysis_output)

    result = copy.deepcopy(source_route)
    result["created_utc"] = now
    result["route_family"] = "alphazuma-55-single-policy-distilled-hard-frontier-rescue"
    result["initial_model"] = {
        "id": f"{source_run['id']}-checkpoint-{checkpoint_steps}",
        "training_steps": int(source_route["initial_model"]["training_steps"]) + checkpoint_steps,
        "path": str(checkpoint_path),
        "sha256": checkpoint_sha256,
    }
    result["implementation"]["distilled_hard_frontier_analysis"] = {
        "path": str(analysis_output),
        "sha256": analysis_sha256,
    }
    result["implementation"]["distilled_hard_frontier_builder"] = {
        "path": str(Path(__file__).resolve()),
        "sha256": base._sha256(Path(__file__).resolve()),
    }
    result["runs"] = [
        {
            "id": route_id,
            "description": (
                "Engineering-only hard-frontier rescue initialized from frozen "
                f"{source_run['id']} checkpoint {checkpoint_steps}."
            ),
            "run_dir": str(run_dir),
            "device": device,
            "seed": model_seed,
            "episode_seed_base": episode_seed_base,
            "episode_seed_stride": 250_000,
            "episode_seed_last": candidate_last,
            "total_steps": int(source_run["total_steps"]),
            "num_envs": num_envs,
            "rollout_steps": int(source_run["rollout_steps"]),
            "batch_size": int(source_run["batch_size"]),
            "ppo_epochs": int(source_run["ppo_epochs"]),
            "learning_rate": float(source_run["learning_rate"]),
            "entropy_coef": float(source_run["entropy_coef"]),
            "checkpoint_every": int(source_run["checkpoint_every"]),
            "initial_weights": weights,
            "adaptive": copy.deepcopy(source_run["adaptive"]),
        }
    ]
    result["route_seed_namespace"] = {
        "first": episode_seed_base,
        "last": candidate_last,
        "registry": "training",
    }
    result["evidence_anchor"] = {
        "analysis": {"path": str(analysis_output), "sha256": analysis_sha256},
        "source_route": {"path": str(source_route_path), "sha256": base._sha256(source_route_path)},
        "source_checkpoint": {"path": str(checkpoint_path), "sha256": checkpoint_sha256},
        "source_checkpoint_steps": checkpoint_steps,
        "formal_seed_consumption_before_freeze": False,
        "existing_formal_candidate_set_unchanged": True,
    }
    base._require(result["environment"] == source_route["environment"], "environment changed")
    base._require(result["levels"] == source_route["levels"], "level inventory changed")
    base._require(result["trainer"] == source_route["trainer"], "trainer changed")
    base._write_json_exclusive(route_output, result)
    return analysis, result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", required=True, type=Path)
    parser.add_argument("--training-route", required=True, action="append", type=Path)
    parser.add_argument("--source-route", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint-steps", required=True, type=int)
    parser.add_argument("--route-id", required=True)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--device", choices=("cuda:0", "cuda:1"), required=True)
    parser.add_argument("--model-seed", required=True, type=int)
    parser.add_argument("--episode-seed-base", required=True, type=int)
    parser.add_argument("--num-envs", required=True, type=int)
    parser.add_argument("--analysis-output", required=True, type=Path)
    parser.add_argument("--route-output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    analysis_output = args.analysis_output.expanduser().resolve()
    route_output = args.route_output.expanduser().resolve()
    analysis, route = build(
        master_path=args.master.expanduser().resolve(strict=True),
        route_paths=[path.expanduser().resolve(strict=True) for path in args.training_route],
        source_route_path=args.source_route.expanduser().resolve(strict=True),
        checkpoint_path=args.checkpoint.expanduser().resolve(strict=True),
        checkpoint_steps=args.checkpoint_steps,
        route_id=args.route_id,
        run_dir=args.run_dir.expanduser().resolve(),
        device=args.device,
        model_seed=args.model_seed,
        episode_seed_base=args.episode_seed_base,
        num_envs=args.num_envs,
        analysis_output=analysis_output,
        route_output=route_output,
    )
    print(
        json.dumps(
            {
                "status": "FROZEN_BEFORE_RESCUE_TRAINING",
                "analysis": {"path": str(analysis_output), "sha256": base._sha256(analysis_output)},
                "route": {"path": str(route_output), "sha256": base._sha256(route_output)},
                "summary": analysis["summary"],
                "run": route["runs"][0],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
