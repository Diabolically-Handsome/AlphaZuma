"""Recomputable contract for the exact state-policy training environment."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import struct
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from zuma_rl.replay import environment_fingerprint, state_fingerprint
from zuma_rl.revenge_core import SUPPORTED_PROFILE_MODE
from zuma_rl.revenge_env import RevengeEnv, RevengeEnvConfig


CONTRACT_SCHEMA = "zuma-rl.training-environment-contract"
CONTRACT_VERSION = 2
POLICY_ID = "jungle2-transfer-bootstrap-v3"
KAT_SEEDS = (0, 1, 20_260_950)
KAT_TICKS_PER_SEED = 128

# These are deliberately explicit runtime dependency closures.  The previous
# contract hashed every Python module in the package, so changes to independent
# PC evidence tooling invalidated the training environment and model migration.
# Keep evidence/verifier code out of these groups: it is recomputed by its own
# gate and does not participate in simulator transitions or policy updates.
ENVIRONMENT_SOURCE_PATHS = (
    "src/zuma_rl/original_data.py",
    "src/zuma_rl/revenge_core.py",
    "src/zuma_rl/revenge_env.py",
)
POLICY_SOURCE_PATHS = (
    "src/zuma_rl/revenge_features.py",
    "src/zuma_rl/teacher_anchor.py",
    "src/zuma_rl/train.py",
)

DEPENDENCIES = (
    "gymnasium",
    "numpy",
    "sb3-contrib",
    "stable-baselines3",
    "torch",
)


def training_environment_config() -> RevengeEnvConfig:
    """Return the immutable v3 bootstrap environment configuration."""

    return RevengeEnvConfig(
        aim_bins=180,
        action_mode="factorized",
        frame_skip=1,
        max_ticks=12_000,
        max_balls=768,
    )


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _source_hashes(
    project_root: Path,
    relative_paths: tuple[str, ...],
) -> dict[str, str]:
    return {
        relative: _sha256_path(project_root.joinpath(*relative.split("/")))
        for relative in relative_paths
    }


def _implementation_sources(project_root: Path) -> dict[str, dict[str, str]]:
    """Hash only code that can affect transitions or policy optimization."""

    return {
        "environment_runtime": _source_hashes(
            project_root,
            ENVIRONMENT_SOURCE_PATHS,
        ),
        "policy_runtime": _source_hashes(project_root, POLICY_SOURCE_PATHS),
    }


def _feed(hasher: Any, label: str, payload: bytes) -> None:
    encoded_label = label.encode("ascii")
    hasher.update(struct.pack("<I", len(encoded_label)))
    hasher.update(encoded_label)
    hasher.update(struct.pack("<Q", len(payload)))
    hasher.update(payload)


def _kat_row(
    *,
    original_root: Path,
    seed: int,
    level_id: str,
    hard: bool,
    profile_mode: str,
) -> dict[str, Any]:
    env = RevengeEnv(
        training_environment_config(),
        level_id=level_id,
        root=original_root,
        hard=hard,
        curve_index=0,
        profile_mode=profile_mode,
        seed=seed,
    )
    digest = hashlib.sha256()
    try:
        observation, _ = env.reset(seed=seed)
        terminal_tick: int | None = None
        for tick in range(KAT_TICKS_PER_SEED):
            mask = np.asarray(env.action_masks(), dtype=np.bool_)
            _feed(digest, "observation", np.asarray(observation, dtype="<f4").tobytes())
            _feed(digest, "action_mask", mask.astype(np.uint8).tobytes())
            _feed(
                digest,
                "state_fingerprint",
                state_fingerprint(env.sim).encode("ascii"),
            )
            verb = 0
            if tick % 37 == 0 and bool(mask[1]):
                verb = 1
            elif tick % 53 == 0 and bool(mask[2]):
                verb = 2
            aim_bin = (seed + tick * 47) % env.config.aim_bins
            action = env.encode_action(verb, aim_bin)
            _feed(
                digest,
                "action",
                np.asarray(action, dtype="<i8").reshape(-1).tobytes(),
            )
            observation, reward, terminated, truncated, _ = env.step(action)
            _feed(digest, "reward", struct.pack("<d", float(reward)))
            _feed(
                digest,
                "done",
                bytes((int(bool(terminated)), int(bool(truncated)))),
            )
            if terminated or truncated:
                terminal_tick = tick + 1
                break
        completed = terminal_tick or KAT_TICKS_PER_SEED
        _feed(
            digest,
            "final_state_fingerprint",
            state_fingerprint(env.sim).encode("ascii"),
        )
        return {
            "seed": seed,
            "requested_ticks": KAT_TICKS_PER_SEED,
            "completed_ticks": completed,
            "terminal_tick": terminal_tick,
            "trajectory_sha256": "sha256:" + digest.hexdigest(),
        }
    finally:
        env.close()


def build_training_environment_contract(
    *,
    original_root: str | Path,
    level_id: str = "Jungle2",
    hard: bool = False,
    profile_mode: str = SUPPORTED_PROFILE_MODE,
) -> dict[str, Any]:
    """Build the exact environment and actor-interface identity used by v3."""

    root = Path(original_root).resolve(strict=True)
    project_root = Path(__file__).resolve().parents[2]
    config = training_environment_config()
    env = RevengeEnv(
        config,
        level_id=level_id,
        root=root,
        hard=hard,
        curve_index=0,
        profile_mode=profile_mode,
        seed=0,
    )
    try:
        layout = {
            name: {
                "start": int(section.start or 0),
                "stop": int(section.stop or env.observation_size),
            }
            for name, section in env.observation_layout.items()
        }
        action_nvec = [int(value) for value in env.action_space.nvec]
        static_fingerprint = environment_fingerprint(env.sim)
        fruit_calibration = (
            None
            if env.sim.fruit_calibration is None
            else asdict(env.sim.fruit_calibration)
        )
        actor_interface = {
            "observation_shape": [int(value) for value in env.observation_space.shape],
            "observation_dtype": str(env.observation_space.dtype),
            "observation_low": float(np.min(env.observation_space.low)),
            "observation_high": float(np.max(env.observation_space.high)),
            "layout": layout,
            "ball_feature_names": list(env.ball_feature_names),
            "projectile_feature_names": list(env.projectile_feature_names),
            "global_feature_names": list(env.global_feature_names),
            "fruit_global_indices": {
                name: env.global_feature_names.index(name)
                for name in (
                    "fruit_present",
                    "fruit_collectable",
                    "fruit_x",
                    "fruit_y",
                    "fruit_collecting",
                )
            },
            "current_color_start": env.global_feature_names.index(
                "current_color_0"
            ),
            "next_color_start": env.global_feature_names.index("next_color_0"),
            "action_space": {
                "kind": "MultiDiscrete",
                "nvec": action_nvec,
                "mask_length": int(sum(action_nvec)),
                "mask_semantics": "actor_observable_retail_rejection_mask",
            },
        }
    finally:
        env.close()

    source_hashes = _implementation_sources(project_root)
    dependency_versions = {
        name: importlib.metadata.version(name) for name in DEPENDENCIES
    }
    kat_rows = [
        _kat_row(
            original_root=root,
            seed=seed,
            level_id=level_id,
            hard=hard,
            profile_mode=profile_mode,
        )
        for seed in KAT_SEEDS
    ]
    payload: dict[str, Any] = {
        "schema": CONTRACT_SCHEMA,
        "version": CONTRACT_VERSION,
        "status": "PASS",
        "policy": POLICY_ID,
        "scope": {
            "environment_id": "ZumaRevenge-v0",
            "level_id": level_id,
            "hard": hard,
            "curve_index": 0,
            "profile_mode": profile_mode,
            "observation_mode": "actor",
        },
        "configuration": asdict(config),
        "installed_original": {
            "main_pak_sha256": _sha256_path(root / "main.pak"),
            "executable_sha256": _sha256_path(root / "ZumasRevenge.exe"),
        },
        "implementation_source_sha256": source_hashes,
        "dependency_versions": dependency_versions,
        "simulator": {
            "environment_fingerprint": static_fingerprint,
            "fruit_calibration": fruit_calibration,
        },
        "actor_interface": actor_interface,
        "deterministic_kat": {
            "encoding": "tagged-little-endian-v1",
            "rows": kat_rows,
            "aggregate_sha256": _canonical_sha256({"rows": kat_rows}),
        },
        "failure_reasons": [],
    }
    # Normalize tuples and NumPy-compatible scalar containers through the
    # exact JSON representation that is persisted.  This makes a freshly
    # built contract dictionary equal to the same contract read from disk.
    normalized = json.loads(
        json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    normalized["contract_fingerprint"] = _canonical_sha256(normalized)
    return normalized


def validate_training_environment_contract(
    report: Mapping[str, Any],
    *,
    original_root: str | Path | None,
) -> str | None:
    """Recompute the full contract instead of trusting its summary fields."""

    if original_root is None:
        return "original_root is required for the training environment contract"
    try:
        live = build_training_environment_contract(original_root=original_root)
    except (OSError, RuntimeError, ValueError, importlib.metadata.PackageNotFoundError):
        return "training environment contract could not be recomputed"
    if dict(report) != live:
        return "training environment contract differs from the live environment"
    return None


__all__ = [
    "CONTRACT_SCHEMA",
    "CONTRACT_VERSION",
    "ENVIRONMENT_SOURCE_PATHS",
    "KAT_SEEDS",
    "POLICY_ID",
    "POLICY_SOURCE_PATHS",
    "build_training_environment_contract",
    "training_environment_config",
    "validate_training_environment_contract",
]
