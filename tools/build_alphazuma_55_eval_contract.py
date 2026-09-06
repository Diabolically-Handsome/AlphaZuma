"""Freeze a hash-bound AlphaZuma 55 evaluation matrix and model manifest."""

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

from zuma_rl.alphazuma_55 import INCLUDED_LEVELS, full55_environment_config
from zuma_rl.original_data import OriginalGameCatalog


STAGES = {
    "engineering": {"base_seed": 1_400_200_000, "attempts": 1},
    "selection": {"base_seed": 1_600_000_000, "attempts": 4},
    "final_blind": {"base_seed": 1_700_000_000, "attempts": 8},
    "continuous": {"base_seed": 1_800_000_000, "attempts": 4},
}


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
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError:
        raise FileExistsError(f"refusing to overwrite frozen contract: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _model_spec(value: str) -> dict[str, Any]:
    try:
        model_id, steps_text, path_text = value.split("=", 2)
        training_steps = int(steps_text)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "model must be ID=TRAINING_STEPS=PATH"
        ) from error
    if not model_id or training_steps < 0 or not path_text:
        raise argparse.ArgumentTypeError("model spec contains an empty value")
    path = Path(path_text).expanduser().resolve(strict=True)
    return {
        "id": model_id,
        "training_steps": training_steps,
        "path": str(path),
        "sha256": _sha256(path),
    }


def build_contracts(
    *,
    master_path: Path,
    original_root: Path,
    evaluator_path: Path,
    stage: str,
    output_root: Path,
    models: list[dict[str, Any]],
    parallel_envs_per_shard: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    master = _read_json(master_path)
    if (
        master.get("schema")
        != "zuma-rl.alphazuma-55-weekend-master-preregistration"
        or master.get("version") != 1
    ):
        raise ValueError("unexpected master preregistration")
    if stage not in STAGES:
        raise ValueError(f"unsupported stage: {stage}")
    if not models or len({model["id"] for model in models}) != len(models):
        raise ValueError("model ids must be non-empty and unique")
    if parallel_envs_per_shard < 1:
        raise ValueError("parallel env count must be positive")

    frozen_scope = master["scope"]["included_levels_in_adventure_order"]
    if list(INCLUDED_LEVELS) != frozen_scope:
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

    stage_spec = STAGES[stage]
    attempts = int(stage_spec["attempts"])
    total_attempts = len(levels) * attempts
    base_seed = int(stage_spec["base_seed"])
    expected_registry = {
        "engineering": "engineering_and_calibration",
        "selection": "selection",
        "final_blind": "final_blind",
        "continuous": "continuous_campaign_challenge",
    }[stage]
    registry = master["seed_registry"][expected_registry]
    if not (
        int(registry["first"]) <= base_seed
        and base_seed + total_attempts - 1 <= int(registry["last"])
    ):
        raise ValueError("evaluation matrix exceeds its frozen seed registry")

    created = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema": "zuma-rl.zero-shot-models-manifest",
        "version": 1,
        "status": "FROZEN",
        "created_utc": created,
        "campaign_id": master["campaign_id"],
        "stage": stage,
        "master_preregistration": {
            "path": str(master_path),
            "sha256": _sha256(master_path),
        },
        "models": models,
    }

    # Engineering teacher assignment must cycle failures quickly, matching the
    # historical training/evaluation horizon.  Formal selection and final blind
    # retain the long capability ceiling so a slow valid clear is not hidden.
    max_ticks = 12_000 if stage == "engineering" else 180_000
    base_config = full55_environment_config(max_ticks=max_ticks)
    preregistration = {
        "schema": "zuma-rl.zero-shot-multilevel-preregistration",
        "version": 1,
        "status": "FROZEN_BEFORE_EVALUATION",
        "created_utc": created,
        "campaign_id": master["campaign_id"],
        "stage": stage,
        "master_preregistration": {
            "path": str(master_path),
            "sha256": _sha256(master_path),
        },
        "model": models[0],
        "levels": levels,
        "attempts_per_level": attempts,
        "total_attempts": total_attempts,
        "seed_plan": {
            "registry": expected_registry,
            "base_seed": base_seed,
            "last_seed": base_seed + total_attempts - 1,
        },
        "environment": {
            "profile_mode": "tutorials_completed",
            "base_config": {
                "aim_bins": base_config.aim_bins,
                "action_mode": base_config.action_mode,
                "frame_skip": base_config.frame_skip,
                "max_ticks": base_config.max_ticks,
                "max_balls": base_config.max_balls,
                "max_projectiles": base_config.max_projectiles,
                "max_curves": base_config.max_curves,
                "score_reward_scale": base_config.score_reward_scale,
                "step_penalty": base_config.step_penalty,
                "win_reward": base_config.win_reward,
                "loss_reward": base_config.loss_reward,
                "privileged_debug": base_config.privileged_debug,
                "render_width": base_config.render_width,
                "render_height": base_config.render_height,
                "actor_interface": base_config.actor_interface,
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
        "execution": {
            "output_root": str(output_root),
            "shard_count": 2,
            "devices": ["cuda:0", "cuda:1"],
            "parallel_envs_per_shard": parallel_envs_per_shard,
        },
        "evaluator": {
            "path": str(evaluator_path),
            "sha256": _sha256(evaluator_path),
        },
    }
    return manifest, preregistration


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-preregistration", required=True, type=Path)
    parser.add_argument("--original-root", required=True, type=Path)
    parser.add_argument(
        "--evaluator",
        type=Path,
        default=Path(__file__).resolve().with_name(
            "evaluate_zero_shot_multilevel.py"
        ),
    )
    parser.add_argument("--stage", choices=tuple(STAGES), required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--manifest-output", required=True, type=Path)
    parser.add_argument("--preregistration-output", required=True, type=Path)
    parser.add_argument("--model", action="append", type=_model_spec, required=True)
    parser.add_argument("--parallel-envs-per-shard", type=int, default=24)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_path = args.manifest_output.expanduser().resolve()
    prereg_path = args.preregistration_output.expanduser().resolve()
    for path in (manifest_path, prereg_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite frozen contract: {path}")
    manifest, preregistration = build_contracts(
        master_path=args.master_preregistration.expanduser().resolve(strict=True),
        original_root=args.original_root.expanduser().resolve(strict=True),
        evaluator_path=args.evaluator.expanduser().resolve(strict=True),
        stage=args.stage,
        output_root=args.output_root.expanduser().resolve(),
        models=args.model,
        parallel_envs_per_shard=args.parallel_envs_per_shard,
    )
    _write_json_exclusive(manifest_path, manifest)
    _write_json_exclusive(prereg_path, preregistration)
    print(
        json.dumps(
            {
                "status": "FROZEN",
                "manifest": str(manifest_path),
                "manifest_sha256": _sha256(manifest_path),
                "preregistration": str(prereg_path),
                "preregistration_sha256": _sha256(prereg_path),
                "models": len(manifest["models"]),
                "levels": len(preregistration["levels"]),
                "attempts": preregistration["total_attempts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
