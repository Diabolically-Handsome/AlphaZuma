#!/usr/bin/env python3
"""Audit the frozen 55-level actor interface without consuming blind seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from zuma_rl.alphazuma_55 import (
    DUAL_POSITION_LEVELS,
    EXCLUDED_MOVING_FROG_LEVELS,
    INCLUDED_LEVELS,
    full55_environment_config,
)
from zuma_rl.full55_migration import sha256_path
from zuma_rl.original_data import OriginalGameCatalog
from zuma_rl.replay import state_fingerprint
from zuma_rl.revenge_env import RevengeEnv

SCHEMA = "zuma-rl.alphazuma-55-environment-audit"
VERSION = 1
SEED_BASE = 1_400_100_000
STEPS_PER_LEVEL = 32


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-root", required=True, type=Path)
    parser.add_argument("--master-preregistration", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def _feed(digest: Any, label: str, payload: bytes) -> None:
    encoded = label.encode("ascii")
    digest.update(struct.pack("<I", len(encoded)))
    digest.update(encoded)
    digest.update(struct.pack("<Q", len(payload)))
    digest.update(payload)


def _inactive_color_channels_are_zero(env: RevengeEnv, observation: np.ndarray) -> bool:
    if env.num_colors >= env.color_feature_count:
        return True
    ball_slice = env.observation_layout["balls"]
    balls = observation[ball_slice].reshape(
        env.config.max_balls,
        env.ball_feature_size,
    )
    ball_start = env.ball_feature_names.index("color_0")
    projectile_slice = env.observation_layout["projectiles"]
    projectiles = observation[projectile_slice].reshape(
        env.config.max_projectiles,
        env.projectile_feature_size,
    )
    projectile_start = env.projectile_feature_names.index("color_0")
    global_slice = env.observation_layout["globals"]
    globals_ = observation[global_slice]
    current_start = env.global_feature_names.index("current_color_0")
    next_start = env.global_feature_names.index("next_color_0")
    inactive = slice(env.num_colors, env.color_feature_count)
    return bool(
        np.all(balls[:, ball_start : ball_start + env.color_feature_count][:, inactive] == 0)
        and np.all(
            projectiles[
                :,
                projectile_start : projectile_start + env.color_feature_count,
            ][:, inactive]
            == 0
        )
        and np.all(
            globals_[
                current_start : current_start + env.color_feature_count
            ][inactive]
            == 0
        )
        and np.all(
            globals_[next_start : next_start + env.color_feature_count][inactive]
            == 0
        )
    )


def _trajectory(env: RevengeEnv, *, seed: int) -> dict[str, Any]:
    digest = hashlib.sha256()
    observation, info = env.reset(seed=seed)
    hop_accepts = 0
    all_inactive_zero = True
    for tick in range(STEPS_PER_LEVEL):
        mask = env.action_masks()
        all_inactive_zero = all_inactive_zero and _inactive_color_channels_are_zero(
            env,
            observation,
        )
        _feed(digest, "observation", observation.astype("<f4", copy=False).tobytes())
        _feed(digest, "mask", mask.astype(np.uint8).tobytes())
        _feed(digest, "state", state_fingerprint(env.sim).encode("ascii"))
        verb = 0
        if hop_accepts == 0 and bool(mask[3]):
            verb = 3
        elif tick % 17 == 5 and bool(mask[1]):
            verb = 1
        elif tick % 23 == 7 and bool(mask[2]):
            verb = 2
        action = env.encode_action(verb, (seed + tick * 47) % 180)
        observation, reward, terminated, truncated, info = env.step(action)
        if verb == 3 and info.get("action_accepted") is True:
            hop_accepts += 1
        _feed(digest, "reward", struct.pack("<d", float(reward)))
        _feed(digest, "done", bytes((int(terminated), int(truncated))))
        if terminated or truncated:
            raise RuntimeError("engineering trajectory terminated unexpectedly")
    return {
        "trajectory_sha256": "sha256:" + digest.hexdigest(),
        "inactive_color_channels_zero": all_inactive_zero,
        "hop_accepts": hop_accepts,
        "final_state_sha256": state_fingerprint(env.sim),
        "initial_info": {
            key: info.get(key)
            for key in (
                "actor_interface",
                "active_num_colors",
                "color_feature_count",
                "curve_count",
                "frog_position_count",
            )
        },
    }


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    args = _parser().parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    preregistration = json.loads(
        args.master_preregistration.read_text(encoding="utf-8")
    )
    frozen_levels = tuple(
        preregistration["scope"]["included_levels_in_adventure_order"]
    )
    if frozen_levels != INCLUDED_LEVELS:
        raise ValueError("code and preregistered included-level order differ")
    if tuple(preregistration["scope"]["excluded_moving_frog_levels"]) != (
        EXCLUDED_MOVING_FROG_LEVELS
    ):
        raise ValueError("code and preregistered moving-frog exclusions differ")
    if tuple(preregistration["scope"]["included_dual_position_levels"]) != (
        DUAL_POSITION_LEVELS
    ):
        raise ValueError("code and preregistered dual-position levels differ")

    catalog = OriginalGameCatalog(args.original_root)
    config = full55_environment_config()
    reference_observation_space = None
    reference_action_space = None
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for index, level_id in enumerate(INCLUDED_LEVELS):
        env = RevengeEnv(
            config,
            level_id=level_id,
            catalog=catalog,
            seed=SEED_BASE + index,
        )
        try:
            if reference_observation_space is None:
                reference_observation_space = env.observation_space
                reference_action_space = env.action_space
            if env.observation_space != reference_observation_space:
                failures.append(f"{level_id}:observation_space")
            if env.action_space != reference_action_space:
                failures.append(f"{level_id}:action_space")
            first = _trajectory(env, seed=SEED_BASE + index)
            second = _trajectory(env, seed=SEED_BASE + index)
            deterministic = first == second
            expected_positions = 2 if level_id in DUAL_POSITION_LEVELS else 1
            if env.sim.shooter_position_count != expected_positions:
                failures.append(f"{level_id}:frog_position_count")
            if not deterministic:
                failures.append(f"{level_id}:determinism")
            if not first["inactive_color_channels_zero"]:
                failures.append(f"{level_id}:inactive_color_channels")
            if expected_positions == 2 and first["hop_accepts"] < 1:
                failures.append(f"{level_id}:hop_never_accepted")
            if expected_positions == 1 and first["hop_accepts"] != 0:
                failures.append(f"{level_id}:single_position_hop_accepted")
            rows.append(
                {
                    "index": index,
                    "level_id": level_id,
                    "display_name": catalog.levels[level_id.casefold()].display_name,
                    "active_num_colors": env.num_colors,
                    "curve_count": env.sim.curve_count,
                    "frog_position_count": env.sim.shooter_position_count,
                    "deterministic": deterministic,
                    **first,
                }
            )
            print(
                f"{index + 1:02d}/55 {level_id} colors={env.num_colors} "
                f"curves={env.sim.curve_count} pads={env.sim.shooter_position_count}",
                flush=True,
            )
        finally:
            env.close()

    rejected: list[dict[str, Any]] = []
    for level_id in (*EXCLUDED_MOVING_FROG_LEVELS, "boss1"):
        try:
            RevengeEnv(config, level_id=level_id, catalog=catalog).close()
        except NotImplementedError as error:
            rejected.append({"level_id": level_id, "reason": str(error)})
        else:
            failures.append(f"{level_id}:unsupported_level_was_accepted")

    report = {
        "schema": SCHEMA,
        "version": VERSION,
        "status": "PASS" if not failures else "FAIL",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "master_preregistration": {
            "path": str(args.master_preregistration.resolve(strict=True)),
            "sha256": sha256_path(args.master_preregistration),
        },
        "audit_tool": {
            "path": str(Path(__file__).resolve(strict=True)),
            "sha256": sha256_path(Path(__file__).resolve(strict=True)),
        },
        "environment_sources": {
            relative: sha256_path(Path(__file__).resolve().parents[1] / relative)
            for relative in (
                "src/zuma_rl/alphazuma_55.py",
                "src/zuma_rl/revenge_core.py",
                "src/zuma_rl/revenge_env.py",
                "src/zuma_rl/revenge_features.py",
                "src/zuma_rl/revenge_teacher.py",
                "src/zuma_rl/replay.py",
            )
        },
        "level_count": len(rows),
        "observation_shape": list(reference_observation_space.shape),
        "action_nvec": reference_action_space.nvec.tolist(),
        "seed_base": SEED_BASE,
        "steps_per_level": STEPS_PER_LEVEL,
        "rows": rows,
        "unsupported_rejections": rejected,
        "failures": failures,
    }
    _write_json_atomic(args.output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "level_count": report["level_count"],
                "observation_shape": report["observation_shape"],
                "action_nvec": report["action_nvec"],
                "failures": failures,
                "output": str(args.output.resolve(strict=True)),
                "output_sha256": sha256_path(args.output),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
