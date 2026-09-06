"""Evaluate one frozen policy zero-shot across preregistered campaign levels."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any, Sequence

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import numpy as np

from tools.evaluate_human_speedrun_blind import EpisodeTracker
from zuma_rl.human_speedrun import (
    EliteHumanInputConfig,
    HumanSpeedrunWrapper,
    WinFirstRewardConfig,
)
from zuma_rl.original_data import OriginalGameCatalog
from zuma_rl.revenge_env import RevengeEnv, RevengeEnvConfig


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _write_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _load_preregistration(path: Path) -> dict[str, Any]:
    prereg = _mapping(json.loads(path.read_text(encoding="utf-8")), "root")
    if prereg.get("schema") != "zuma-rl.zero-shot-multilevel-preregistration":
        raise ValueError("unexpected preregistration schema")
    if prereg.get("version") != 1 or prereg.get("status") != "FROZEN_BEFORE_EVALUATION":
        raise ValueError("preregistration is not a frozen version-1 contract")

    evaluator = _mapping(prereg.get("evaluator"), "evaluator")
    if evaluator.get("sha256") != _sha256(Path(__file__).resolve()):
        raise ValueError("evaluator bytes differ from the preregistered hash")
    model = _mapping(prereg.get("model"), "model")
    model_path = Path(str(model.get("path", ""))).expanduser().resolve(strict=True)
    if model.get("sha256") != _sha256(model_path):
        raise ValueError("model bytes differ from the preregistered hash")

    levels = prereg.get("levels")
    if not isinstance(levels, list) or not levels:
        raise ValueError("levels must be a non-empty array")
    ids = [str(row.get("id", "")) for row in levels if isinstance(row, dict)]
    if len(ids) != len(levels) or len(set(value.casefold() for value in ids)) != len(ids):
        raise ValueError("levels contain invalid or duplicate ids")

    attempts_per_level = int(prereg.get("attempts_per_level", 0))
    if attempts_per_level < 1:
        raise ValueError("attempts_per_level must be positive")
    expected_attempts = len(levels) * attempts_per_level
    if prereg.get("total_attempts") != expected_attempts:
        raise ValueError("total_attempts does not match the frozen matrix")

    execution = _mapping(prereg.get("execution"), "execution")
    devices = execution.get("devices")
    shard_count = int(execution.get("shard_count", 0))
    if not isinstance(devices, list) or shard_count != len(devices) or shard_count < 1:
        raise ValueError("devices and shard_count disagree")
    if int(execution.get("parallel_envs_per_shard", 0)) < 1:
        raise ValueError("parallel_envs_per_shard must be positive")
    return prereg


def _model_specs(
    prereg: dict[str, Any],
    manifest_path: Path | None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if manifest_path is None:
        return [dict(prereg["model"])], None
    manifest = _load_models_manifest(manifest_path)
    return [dict(model) for model in manifest["models"]], {
        "path": str(manifest_path),
        "sha256": _sha256(manifest_path),
    }


def _build_configs(
    prereg: dict[str, Any],
) -> tuple[RevengeEnvConfig, EliteHumanInputConfig, WinFirstRewardConfig]:
    environment = _mapping(prereg.get("environment"), "environment")
    base = _mapping(environment.get("base_config"), "environment.base_config")
    input_profile = _mapping(environment.get("input_profile"), "environment.input_profile")
    reward_profile = _mapping(environment.get("reward_profile"), "environment.reward_profile")
    base_config = RevengeEnvConfig(**base)
    input_config = EliteHumanInputConfig(
        profile_id=str(input_profile["profile_id"]),
        reaction_delay_ticks=int(input_profile["reaction_delay_ticks"]),
        max_aim_speed_degrees_per_second=float(
            input_profile["max_aim_speed_degrees_per_second"]
        ),
        max_aim_acceleration_degrees_per_second_squared=float(
            input_profile["max_aim_acceleration_degrees_per_second_squared"]
        ),
        min_button_interval_ticks=int(input_profile["min_button_interval_ticks"]),
    )
    reward_config = WinFirstRewardConfig(
        profile_id=str(reward_profile["profile_id"]),
        win_reward=float(reward_profile["win_reward"]),
        failure_reward=float(reward_profile["failure_reward"]),
        time_penalty_per_native_tick=float(
            reward_profile["time_penalty_per_native_tick"]
        ),
        score_progress_reward_cap=float(reward_profile["score_progress_reward_cap"]),
    )
    return base_config, input_config, reward_config


def _make_env_factory(
    *,
    original_root: Path,
    level_id: str,
    profile_mode: str,
    base_config: RevengeEnvConfig,
    input_config: EliteHumanInputConfig,
    reward_config: WinFirstRewardConfig,
) -> Any:
    def make_env() -> HumanSpeedrunWrapper:
        base = RevengeEnv(
            config=base_config,
            level_id=level_id,
            root=original_root,
            hard=False,
            curve_index=0,
            profile_mode=profile_mode,
        )
        return HumanSpeedrunWrapper(
            base,
            input_config=input_config,
            reward_config=reward_config,
        )

    return make_env


def _level_summary(level: dict[str, Any], rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    wins = [row for row in rows if row["outcome"] == "win"]
    win_ticks = [int(row["ticks"]) for row in wins]
    return {
        "level_id": level["id"],
        "display_name": level["display_name"],
        "curve_count": int(level["curve_count"]),
        "attempts": len(rows),
        "wins": len(wins),
        "losses": sum(row["outcome"] == "loss" for row in rows),
        "truncations": sum(row["time_limit_truncated"] for row in rows),
        "win_rate": len(wins) / len(rows) if rows else 0.0,
        "best_ticks": min(win_ticks) if win_ticks else None,
        "best_seconds": min(win_ticks) / 100.0 if win_ticks else None,
        "median_win_ticks": statistics.median(win_ticks) if win_ticks else None,
        "median_win_seconds": (
            statistics.median(win_ticks) / 100.0 if win_ticks else None
        ),
        "best_score": max((int(row["score"]) for row in rows), default=None),
        "status": "CLEARED" if wins else "NOT_CLEARED",
    }


def _campaign_inventory(original_root: Path) -> list[dict[str, Any]]:
    catalog = OriginalGameCatalog(original_root)
    prefixes = ("jungle", "village", "city", "coast", "grotto", "volcano")
    result: list[dict[str, Any]] = []
    for definition in catalog.levels.values():
        key = definition.id.casefold()
        if not any(
            key == f"{prefix}{index}"
            for prefix in prefixes
            for index in range(1, 11)
        ):
            continue
        loaded = catalog.load_level(definition.id, hard=False)
        result.append(
            {
                "id": definition.id,
                "display_name": definition.display_name,
                "curve_count": len(loaded.curves),
                "curve_names": list(definition.curve_names),
            }
        )
    return result


def _validate_environment_inventory(
    *,
    prereg: dict[str, Any],
    original_root: Path,
) -> dict[str, Any]:
    expected = prereg["levels"]
    discovered = _campaign_inventory(original_root)
    discovered_by_id = {row["id"].casefold(): row for row in discovered}
    for level in expected:
        actual = discovered_by_id.get(str(level["id"]).casefold())
        if actual is None:
            raise ValueError(f"frozen level is absent from installed assets: {level['id']}")
        for key in ("display_name", "curve_count", "curve_names"):
            if actual[key] != level[key]:
                raise ValueError(f"installed metadata changed for {level['id']}: {key}")
    return {"campaign_levels_discovered": len(discovered), "frozen_levels_verified": len(expected)}


def _validate_model_spaces(
    *,
    prereg: dict[str, Any],
    original_root: Path,
    model_specs: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    from sb3_contrib import MaskablePPO

    specs = list(model_specs or [prereg["model"]])
    base_config, input_config, reward_config = _build_configs(prereg)
    verified: list[dict[str, Any]] = []
    for model_spec in specs:
        model_path = Path(str(model_spec["path"])).resolve(strict=True)
        model = MaskablePPO.load(model_path, device="cpu")
        model_levels: list[str] = []
        for level in prereg["levels"]:
            env = _make_env_factory(
                original_root=original_root,
                level_id=str(level["id"]),
                profile_mode=str(prereg["environment"]["profile_mode"]),
                base_config=base_config,
                input_config=input_config,
                reward_config=reward_config,
            )()
            try:
                if env.observation_space != model.observation_space:
                    raise ValueError(f"model observation-space mismatch for {level['id']}")
                if env.action_space != model.action_space:
                    raise ValueError(f"model action-space mismatch for {level['id']}")
                model_levels.append(str(level["id"]))
            finally:
                env.close()
        del model
        gc.collect()
        verified.append(
            {
                "model_id": str(model_spec["id"]),
                "levels_verified": len(model_levels),
            }
        )
    return {
        "models_verified": len(verified),
        "model_compatible_levels_verified_per_model": len(prereg["levels"]),
        "model_space_validation": verified,
    }


def _shard_bounds(count: int, index: int, shard_count: int) -> tuple[int, int]:
    return count * index // shard_count, count * (index + 1) // shard_count


def _run_batch(
    *,
    tasks: Sequence[dict[str, Any]],
    model: Any,
    model_spec: dict[str, Any],
    original_root: Path,
    profile_mode: str,
    base_config: RevengeEnvConfig,
    input_config: EliteHumanInputConfig,
    reward_config: WinFirstRewardConfig,
) -> tuple[list[dict[str, Any]], int, float]:
    from sb3_contrib.common.maskable.utils import get_action_masks
    from stable_baselines3.common.vec_env import SubprocVecEnv

    seeds = [int(task["seed"]) for task in tasks]
    if seeds != list(range(seeds[0], seeds[0] + len(seeds))):
        raise ValueError("batch task seeds must be contiguous")
    factories = [
        _make_env_factory(
            original_root=original_root,
            level_id=str(task["level_id"]),
            profile_mode=profile_mode,
            base_config=base_config,
            input_config=input_config,
            reward_config=reward_config,
        )
        for task in tasks
    ]
    vec_env = SubprocVecEnv(factories, start_method="forkserver")
    started = time.perf_counter()
    try:
        vec_env.seed(seeds[0])
        observations = vec_env.reset()
        trackers = [EpisodeTracker(seed=seed) for seed in seeds]
        results: list[dict[str, Any] | None] = [None] * len(tasks)
        transitions = 0
        for _ in range(int(base_config.max_ticks) + 1):
            masks = get_action_masks(vec_env)
            actions, _ = model.predict(
                observations,
                deterministic=True,
                action_masks=masks,
            )
            action_array = np.asarray(actions, dtype=np.int64).reshape(len(tasks), 2)
            observations, rewards, dones, infos = vec_env.step(action_array)
            transitions += len(tasks)
            for position, (reward, done, info) in enumerate(
                zip(rewards, dones, infos, strict=True)
            ):
                if results[position] is not None:
                    continue
                trackers[position].update(action_array[position], float(reward), info)
                if bool(done):
                    row = trackers[position].finish(
                        info=info,
                        model=model_spec,
                        threshold_ticks=0,
                    )
                    row.pop("qualifies_sub_12", None)
                    row["level_id"] = tasks[position]["level_id"]
                    row["display_name"] = tasks[position]["display_name"]
                    row["curve_count"] = tasks[position]["curve_count"]
                    row["attempt_index"] = tasks[position]["attempt_index"]
                    results[position] = row
            if all(result is not None for result in results):
                return (
                    [result for result in results if result is not None],
                    transitions,
                    time.perf_counter() - started,
                )
        raise RuntimeError("one or more episodes exceeded the max-tick guard")
    finally:
        vec_env.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path)
    parser.add_argument("--original-root", type=Path)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--compatibility-model", type=Path)
    parser.add_argument(
        "--models-manifest",
        type=Path,
        help=(
            "optional frozen JSON manifest of one or more model specs; when "
            "present, every model is evaluated on the same preregistered tasks"
        ),
    )
    return parser


def _load_models_manifest(path: Path) -> dict[str, Any]:
    manifest = _mapping(json.loads(path.read_text(encoding="utf-8")), "models manifest")
    if manifest.get("schema") != "zuma-rl.zero-shot-models-manifest":
        raise ValueError("unexpected models manifest schema")
    if manifest.get("version") != 1 or manifest.get("status") != "FROZEN":
        raise ValueError("models manifest is not a frozen version-1 contract")
    models = manifest.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("models manifest must contain models")
    ids: set[str] = set()
    for index, raw in enumerate(models):
        model = _mapping(raw, f"models[{index}]")
        model_id = str(model.get("id", ""))
        if not model_id or model_id in ids:
            raise ValueError("model ids are empty or duplicated")
        ids.add(model_id)
        model_path = Path(str(model.get("path", ""))).expanduser().resolve(strict=True)
        if model.get("sha256") != _sha256(model_path):
            raise ValueError(f"model bytes differ from manifest: {model_id}")
        if int(model.get("training_steps", -1)) < 0:
            raise ValueError(f"invalid training steps for model: {model_id}")
    return manifest


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    original_root = (
        args.original_root.expanduser().resolve(strict=True)
        if args.original_root is not None
        else OriginalGameCatalog().root
    )
    if args.inventory_only:
        inventory = _campaign_inventory(original_root)
        if args.compatibility_model is not None:
            from sb3_contrib import MaskablePPO

            model = MaskablePPO.load(
                args.compatibility_model.expanduser().resolve(strict=True),
                device="cpu",
            )
            base_config = RevengeEnvConfig(
                aim_bins=180,
                action_mode="factorized",
                frame_skip=1,
                max_ticks=12000,
                max_balls=768,
                max_projectiles=32,
                max_curves=2,
                score_reward_scale=0.01,
                step_penalty=-0.0001,
                win_reward=10.0,
                loss_reward=-10.0,
            )
            compatible: list[dict[str, Any]] = []
            incompatible: list[dict[str, Any]] = []
            for level in inventory:
                try:
                    env = RevengeEnv(
                        config=base_config,
                        level_id=level["id"],
                        root=original_root,
                        hard=False,
                        curve_index=0,
                        profile_mode="tutorials_completed",
                    )
                except Exception as error:
                    incompatible.append(
                        {
                            **level,
                            "error_type": type(error).__name__,
                            "error": str(error),
                        }
                    )
                    continue
                try:
                    if env.observation_space != model.observation_space:
                        raise RuntimeError(
                            f"observation-space mismatch for {level['id']}: "
                            f"{env.observation_space} != {model.observation_space}"
                        )
                    if env.action_space != model.action_space:
                        raise RuntimeError(
                            f"action-space mismatch for {level['id']}: "
                            f"{env.action_space} != {model.action_space}"
                        )
                    compatible.append(level)
                except Exception as error:
                    incompatible.append(
                        {
                            **level,
                            "error_type": type(error).__name__,
                            "error": str(error),
                        }
                    )
                finally:
                    env.close()
            print(
                json.dumps(
                    {
                        "status": "COMPATIBILITY_INVENTORY",
                        "campaign_levels": len(inventory),
                        "compatible_count": len(compatible),
                        "incompatible_count": len(incompatible),
                        "compatible": compatible,
                        "incompatible": incompatible,
                        "model_sha256": _sha256(
                            args.compatibility_model.expanduser().resolve(strict=True)
                        ),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        print(json.dumps(inventory, ensure_ascii=False, indent=2))
        return 0
    if args.preregistration is None:
        raise SystemExit("--preregistration is required unless --inventory-only is used")
    prereg_path = args.preregistration.expanduser().resolve(strict=True)
    prereg = _load_preregistration(prereg_path)
    manifest_path = (
        args.models_manifest.expanduser().resolve(strict=True)
        if args.models_manifest is not None
        else None
    )
    model_specs, manifest_receipt = _model_specs(prereg, manifest_path)
    inventory = _validate_environment_inventory(prereg=prereg, original_root=original_root)
    if args.validate_only:
        compatibility = _validate_model_spaces(
            prereg=prereg,
            original_root=original_root,
            model_specs=model_specs,
        )
        print(
            json.dumps(
                {
                    "status": "VALID",
                    "preregistration_sha256": _sha256(prereg_path),
                    "evaluator_sha256": _sha256(Path(__file__).resolve()),
                    "model_sha256": (
                        prereg["model"]["sha256"]
                        if manifest_receipt is None
                        else None
                    ),
                    "models_manifest": manifest_receipt,
                    "levels": len(prereg["levels"]),
                    "attempts": prereg["total_attempts"],
                    **inventory,
                    **compatibility,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    execution = prereg["execution"]
    shard_count = int(execution["shard_count"])
    if args.shard_index is None or not 0 <= args.shard_index < shard_count:
        raise SystemExit(f"--shard-index must be in [0, {shard_count})")
    shard_index = int(args.shard_index)
    device = str(execution["devices"][shard_index])

    levels = prereg["levels"]
    start, end = _shard_bounds(len(levels), shard_index, shard_count)
    shard_levels = levels[start:end]
    attempts_per_level = int(prereg["attempts_per_level"])
    seed_base = int(prereg["seed_plan"]["base_seed"])
    tasks: list[dict[str, Any]] = []
    for global_level_index in range(start, end):
        level = levels[global_level_index]
        for attempt_index in range(attempts_per_level):
            global_task_index = global_level_index * attempts_per_level + attempt_index
            tasks.append(
                {
                    "level_id": level["id"],
                    "display_name": level["display_name"],
                    "curve_count": level["curve_count"],
                    "attempt_index": attempt_index,
                    "seed": seed_base + global_task_index,
                }
            )

    output_root = Path(str(execution["output_root"])).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_name = (
        f"shard-{shard_index:02d}-of-{shard_count:02d}.json"
        if manifest_receipt is None
        else f"matrix-shard-{shard_index:02d}-of-{shard_count:02d}.json"
    )
    output_path = output_root / output_name
    if output_path.exists():
        raise SystemExit(f"refusing to overwrite existing shard: {output_path}")

    base_config, input_config, reward_config = _build_configs(prereg)
    result: dict[str, Any] = {
        "schema": (
            "zuma-rl.zero-shot-multilevel-shard"
            if manifest_receipt is None
            else "zuma-rl.zero-shot-multimodel-shard"
        ),
        "version": 1,
        "status": "RUNNING",
        "started_utc": _utc_now(),
        "completed_utc": None,
        "preregistration": {"path": str(prereg_path), "sha256": _sha256(prereg_path)},
        "evaluator_sha256": _sha256(Path(__file__).resolve()),
        "models_manifest": manifest_receipt,
        "models": model_specs,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "device": device,
        "level_range": [start, end],
        "levels": shard_levels,
        "expected_attempts": len(tasks) * len(model_specs),
        "completed_attempts": 0,
        "attempts": [],
        "level_summaries": [],
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "parallel_envs": int(execution["parallel_envs_per_shard"]),
            "total_vector_transitions": 0,
            "wall_seconds": 0.0,
        },
        "error": None,
    }
    _write_json_atomic(output_path, result)

    started = time.perf_counter()
    model: Any | None = None
    try:
        import torch
        from sb3_contrib import MaskablePPO

        torch.set_num_threads(1)
        parallel_envs = int(execution["parallel_envs_per_shard"])
        total_expected = len(tasks) * len(model_specs)
        global_batch_index = 0
        for model_index, model_spec in enumerate(model_specs):
            model = MaskablePPO.load(model_spec["path"], device=device)
            try:
                for offset in range(0, len(tasks), parallel_envs):
                    global_batch_index += 1
                    batch = tasks[offset : offset + parallel_envs]
                    rows, transitions, batch_wall = _run_batch(
                        tasks=batch,
                        model=model,
                        model_spec=model_spec,
                        original_root=original_root,
                        profile_mode=str(prereg["environment"]["profile_mode"]),
                        base_config=base_config,
                        input_config=input_config,
                        reward_config=reward_config,
                    )
                    result["attempts"].extend(rows)
                    result["completed_attempts"] = len(result["attempts"])
                    result["runtime"]["total_vector_transitions"] += transitions
                    result["runtime"]["wall_seconds"] = time.perf_counter() - started
                    summaries: list[dict[str, Any]] = []
                    for summary_model in model_specs:
                        model_rows = [
                            row
                            for row in result["attempts"]
                            if row["model_id"] == summary_model["id"]
                        ]
                        for level in shard_levels:
                            level_rows = [
                                row
                                for row in model_rows
                                if row["level_id"] == level["id"]
                            ]
                            if level_rows:
                                summary = _level_summary(level, level_rows)
                                summary["model_id"] = summary_model["id"]
                                summaries.append(summary)
                    result["level_summaries"] = summaries
                    _write_json_atomic(output_path, result)
                    batch_wins = sum(row["outcome"] == "win" for row in rows)
                    print(
                        f"shard={shard_index}/{shard_count} "
                        f"model={model_spec['id']} ({model_index + 1}/{len(model_specs)}) "
                        f"batch={global_batch_index} "
                        f"completed={result['completed_attempts']}/{total_expected} "
                        f"batch_wins={batch_wins}/{len(rows)} "
                        f"transitions={transitions} wall={batch_wall:.1f}s",
                        flush=True,
                    )
            finally:
                del model
                model = None
                gc.collect()

        result["status"] = "COMPLETE"
        result["completed_utc"] = _utc_now()
        result["runtime"]["wall_seconds"] = time.perf_counter() - started
        _write_json_atomic(output_path, result)
        cleared_by_model = {
            model_spec["id"]: sum(
                summary["wins"] > 0
                for summary in result["level_summaries"]
                if summary.get("model_id") == model_spec["id"]
            )
            for model_spec in model_specs
        }
        print(
            json.dumps(
                {
                    "status": "COMPLETE",
                    "shard": shard_index,
                    "levels_cleared_by_model": cleared_by_model,
                    "levels_total": len(shard_levels),
                    "attempts": len(result["attempts"]),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 0
    except BaseException as error:
        result["status"] = "ERROR"
        result["completed_utc"] = _utc_now()
        result["runtime"]["wall_seconds"] = time.perf_counter() - started
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        _write_json_atomic(output_path, result)
        raise
    finally:
        if model is not None:
            del model
            gc.collect()


if __name__ == "__main__":
    raise SystemExit(main())
