"""Freeze one AlphaZuma 55 PPO route for the shared-catalog launcher."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from zuma_rl.alphazuma_55 import (
    DUAL_POSITION_LEVELS,
    INCLUDED_LEVELS,
    full55_environment_config,
)
from zuma_rl.original_data import OriginalGameCatalog


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


def _write_json_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _level_weight(level_id: str, *, mode: str) -> float:
    key = level_id.casefold()
    if mode == "balanced":
        value = 1.0
    elif mode == "frontier":
        zone_weights = {
            "jungle": 0.60,
            "village": 0.80,
            "city": 1.00,
            "coast": 1.25,
            "grotto": 1.55,
            "volcano": 1.85,
        }
        value = next(
            weight
            for prefix, weight in zone_weights.items()
            if key.startswith(prefix)
        )
    else:
        raise ValueError(f"unsupported curriculum mode: {mode}")
    if key in {item.casefold() for item in DUAL_POSITION_LEVELS}:
        value += 1.75
    if key in {
        "jungle9",
        "village6",
        "city4",
        "city10",
        "coast10",
        "grotto10",
        "volcano9",
    }:
        value += 0.50
    return value


def build_route(
    *,
    master_path: Path,
    calibration_path: Path,
    original_root: Path,
    source_model: Path,
    source_id: str,
    source_training_steps: int,
    migration_receipt: Path,
    route_id: str,
    run_dir: Path,
    device: str,
    model_seed: int,
    episode_seed_base: int,
    num_envs: int,
    curriculum_mode: str,
    learning_rate: float,
    entropy_coef: float,
    ppo_epochs: int,
) -> dict[str, Any]:
    master = _read_json(master_path)
    if (
        master.get("schema")
        != "zuma-rl.alphazuma-55-weekend-master-preregistration"
        or master.get("version") != 1
    ):
        raise ValueError("unexpected master preregistration")
    calibration = _read_json(calibration_path)
    if calibration.get("status") != "PASS":
        raise ValueError("lilypad calibration is not PASS")
    source_receipt = _read_json(migration_receipt)
    source_receipt_schema = str(source_receipt.get("schema", ""))
    if source_receipt_schema == "zuma-rl.alphazuma-55-model-migration":
        receipt_status = source_receipt.get("status") == "PASS"
        receipt_model = source_receipt.get("migrated_model", {})
    elif (
        source_receipt_schema
        == "zuma-rl.alphazuma-55-distillation-completion"
    ):
        receipt_status = source_receipt.get("status") == "COMPLETE"
        receipt_model = source_receipt.get("final_model", {})
    else:
        raise ValueError("unsupported source provenance receipt")
    if (
        not receipt_status
        or receipt_model.get("sha256") != _sha256(source_model)
    ):
        raise ValueError("source receipt does not bind the source model")
    if source_training_steps < 0:
        raise ValueError("source training steps cannot be negative")
    if num_envs < 1 or ppo_epochs < 1:
        raise ValueError("environment and epoch counts must be positive")
    if learning_rate <= 0.0 or entropy_coef < 0.0:
        raise ValueError("invalid optimizer hyperparameters")
    if not route_id or run_dir.exists():
        raise ValueError("route id is empty or run directory already exists")

    training_registry = master["seed_registry"]["training"]
    episode_seed_stride = 250_000
    last_episode_seed = episode_seed_base + num_envs * episode_seed_stride - 1
    if not (
        int(training_registry["first"]) <= episode_seed_base
        and last_episode_seed <= int(training_registry["last"])
    ):
        raise ValueError("route seed namespace exceeds the training registry")

    frozen_scope = master["scope"]["included_levels_in_adventure_order"]
    if frozen_scope != list(INCLUDED_LEVELS):
        raise ValueError("source level scope differs from master preregistration")
    catalog = OriginalGameCatalog(original_root)
    levels: list[dict[str, Any]] = []
    for level_id in INCLUDED_LEVELS:
        definition = catalog.level(level_id)
        loaded = catalog.load_level(level_id, hard=False)
        levels.append(
            {
                "id": definition.id,
                "display_name": definition.display_name,
                "curve_count": len(loaded.curves),
                "curve_names": list(definition.curve_names),
            }
        )

    project_root = Path(__file__).resolve().parents[1]
    frozen_trainer = project_root / "tools" / "train_overnight_multilevel.py"
    launcher = project_root / "tools" / "train_alphazuma_55.py"
    source_paths = {
        "shared_catalog_launcher": launcher,
        "alphazuma_55_scope": project_root / "src" / "zuma_rl" / "alphazuma_55.py",
        "revenge_core": project_root / "src" / "zuma_rl" / "revenge_core.py",
        "revenge_environment": project_root / "src" / "zuma_rl" / "revenge_env.py",
        "revenge_features": project_root / "src" / "zuma_rl" / "revenge_features.py",
        "human_speedrun_wrapper": project_root / "src" / "zuma_rl" / "human_speedrun.py",
        "source_provenance_receipt": migration_receipt,
        "lilypad_calibration": calibration_path,
    }
    implementation = {
        name: {"path": str(path), "sha256": _sha256(path)}
        for name, path in source_paths.items()
    }

    config = full55_environment_config(max_ticks=12_000)
    initial_weights = {
        level_id: _level_weight(level_id, mode=curriculum_mode)
        for level_id in INCLUDED_LEVELS
    }
    created = datetime.now(timezone.utc).isoformat()
    return {
        "schema": "zuma-rl.overnight-multilevel-preregistration",
        "version": 1,
        "status": "FROZEN_BEFORE_TRAINING",
        "created_utc": created,
        "campaign_id": master["campaign_id"],
        "route_family": "alphazuma-55-single-policy",
        "master_preregistration": {
            "path": str(master_path),
            "sha256": _sha256(master_path),
        },
        "trainer": {
            "path": str(frozen_trainer),
            "sha256": _sha256(frozen_trainer),
        },
        "implementation": implementation,
        "initial_model": {
            "id": source_id,
            "training_steps": source_training_steps,
            "path": str(source_model),
            "sha256": _sha256(source_model),
        },
        "levels": levels,
        "environment": {
            "profile_mode": "tutorials_completed",
            "hard": False,
            "curve_index": 0,
            "base_config": {
                "aim_bins": config.aim_bins,
                "action_mode": config.action_mode,
                "frame_skip": config.frame_skip,
                "max_ticks": config.max_ticks,
                "max_balls": config.max_balls,
                "max_projectiles": config.max_projectiles,
                "max_curves": config.max_curves,
                "score_reward_scale": config.score_reward_scale,
                "step_penalty": config.step_penalty,
                "win_reward": config.win_reward,
                "loss_reward": config.loss_reward,
                "privileged_debug": config.privileged_debug,
                "render_width": config.render_width,
                "render_height": config.render_height,
                "actor_interface": config.actor_interface,
            },
            "input_profile": {
                "profile_id": "elite-human-v1",
                "reaction_delay_ticks": 12,
                "max_aim_speed_degrees_per_second": 1080.0,
                "max_aim_acceleration_degrees_per_second_squared": 18000.0,
                "min_button_interval_ticks": 5,
            },
            "reward_profile": {
                "profile_id": "win-time-score-v1",
                "win_reward": 10.0,
                "failure_reward": -10.0,
                "time_penalty_per_native_tick": -0.0001,
                "score_progress_reward_cap": 0.01,
            },
        },
        "runs": [
            {
                "id": route_id,
                "description": (
                    f"AlphaZuma 55 {curriculum_mode} adaptive route initialized "
                    f"from {source_id}."
                ),
                "run_dir": str(run_dir),
                "device": device,
                "seed": model_seed,
                "episode_seed_base": episode_seed_base,
                "episode_seed_stride": episode_seed_stride,
                "episode_seed_last": last_episode_seed,
                "total_steps": 1_000_000_000,
                "num_envs": num_envs,
                "rollout_steps": 256,
                "batch_size": 512,
                "ppo_epochs": ppo_epochs,
                "learning_rate": learning_rate,
                "entropy_coef": entropy_coef,
                "checkpoint_every": 1_048_576,
                "initial_weights": initial_weights,
                "adaptive": {
                    "enabled": True,
                    "update_every_steps": 262_144,
                    "rolling_episodes": 64,
                    "minimum_episodes": 4,
                    "minimum_weight": 0.50,
                    "maximum_weight": 8.00,
                    "anchor_level": "Jungle2",
                },
            }
        ],
        "schedule": {
            "training_stop_utc": master["schedule"]["formal_training_stop_utc"],
            "goal_end_utc": master["deadline_utc"],
        },
        "route_seed_namespace": {
            "first": episode_seed_base,
            "last": last_episode_seed,
            "registry": "training",
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-preregistration", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--original-root", required=True, type=Path)
    parser.add_argument("--source-model", required=True, type=Path)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--source-training-steps", required=True, type=int)
    parser.add_argument("--migration-receipt", required=True, type=Path)
    parser.add_argument("--route-id", required=True)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--device", choices=("cuda:0", "cuda:1"), required=True)
    parser.add_argument("--model-seed", required=True, type=int)
    parser.add_argument("--episode-seed-base", required=True, type=int)
    parser.add_argument("--num-envs", required=True, type=int)
    parser.add_argument(
        "--curriculum-mode",
        choices=("balanced", "frontier"),
        required=True,
    )
    parser.add_argument("--learning-rate", required=True, type=float)
    parser.add_argument("--entropy-coef", required=True, type=float)
    parser.add_argument("--ppo-epochs", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen route: {output}")
    route = build_route(
        master_path=args.master_preregistration.expanduser().resolve(strict=True),
        calibration_path=args.calibration.expanduser().resolve(strict=True),
        original_root=args.original_root.expanduser().resolve(strict=True),
        source_model=args.source_model.expanduser().resolve(strict=True),
        source_id=args.source_id,
        source_training_steps=args.source_training_steps,
        migration_receipt=args.migration_receipt.expanduser().resolve(strict=True),
        route_id=args.route_id,
        run_dir=args.run_dir.expanduser().resolve(),
        device=args.device,
        model_seed=args.model_seed,
        episode_seed_base=args.episode_seed_base,
        num_envs=args.num_envs,
        curriculum_mode=args.curriculum_mode,
        learning_rate=args.learning_rate,
        entropy_coef=args.entropy_coef,
        ppo_epochs=args.ppo_epochs,
    )
    _write_json_exclusive(output, route)
    print(
        json.dumps(
            {
                "status": "FROZEN",
                "output": str(output),
                "sha256": _sha256(output),
                "route_id": route["runs"][0]["id"],
                "levels": len(route["levels"]),
                "seed_namespace": route["route_seed_namespace"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
