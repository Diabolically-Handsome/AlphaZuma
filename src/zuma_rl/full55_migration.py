"""Function-anchored migration from the frozen V1 interface to full55-v1."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from zuma_rl.alphazuma_55 import (
    full55_environment_config,
    legacy_migration_environment_config,
)
from zuma_rl.revenge_core import PowerupType, SUPPORTED_PROFILE_MODE
from zuma_rl.revenge_env import RevengeEnv
from zuma_rl.revenge_features import revenge_polar_policy_kwargs

MIGRATION_SCHEMA = "zuma-rl.alphazuma-55-model-migration"
MIGRATION_VERSION = 1
PROOF_LEVEL = "Jungle2"
PROOF_SEED = 1_400_000_000
PROOF_STEPS = 64
MAX_LOG_PROBABILITY_ERROR = 1e-5
MAX_VALUE_ERROR = 1e-5


class Full55MigrationError(ValueError):
    """The source model or a function-preservation invariant is invalid."""


def sha256_path(path: str | Path) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _legacy_feature_names(source_model: Any) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    extractor = source_model.policy.features_extractor
    colors = int(getattr(extractor, "num_colors", 0))
    curves = int(getattr(extractor, "curve_feature_count", 0))
    if colors != 4 or curves != 2:
        raise Full55MigrationError(
            "source model is not the frozen four-colour, two-curve interface"
        )
    ball_names = (
        RevengeEnv._BALL_BASE_FEATURES
        + tuple(f"color_{index}" for index in range(colors))
        + tuple(f"powerup_{index}" for index in range(int(PowerupType.NONE)))
        + tuple(f"curve_{index}" for index in range(curves))
    )
    projectile_names = (
        RevengeEnv._PROJECTILE_BASE_FEATURES
        + tuple(f"color_{index}" for index in range(colors))
        + tuple(f"curve_{index}" for index in range(curves))
    )
    global_names = (
        RevengeEnv._GLOBAL_BASE_FEATURES
        + tuple(f"current_color_{index}" for index in range(colors))
        + tuple(f"next_color_{index}" for index in range(colors))
    )
    expected = (
        int(extractor.ball_feature_size),
        int(extractor.projectile_feature_size),
        int(extractor.global_feature_size),
    )
    actual = (len(ball_names), len(projectile_names), len(global_names))
    if actual != expected:
        raise Full55MigrationError(
            f"source feature names do not match extractor layout: {actual} != {expected}"
        )
    return ball_names, projectile_names, global_names


def _column_mapping(
    source_names: Sequence[str],
    target_names: Sequence[str],
) -> dict[int, int]:
    if len(set(source_names)) != len(source_names):
        raise Full55MigrationError("source feature names are not unique")
    target_index = {name: index for index, name in enumerate(target_names)}
    missing = [name for name in source_names if name not in target_index]
    if missing:
        raise Full55MigrationError(f"target feature layout lost columns: {missing}")
    return {
        old_index: target_index[name]
        for old_index, name in enumerate(source_names)
    }


def _expand_linear_columns(
    source_tensor: Any,
    target_tensor: Any,
    mapping: Mapping[int, int],
    *,
    source_raw_width: int | None = None,
    target_raw_width: int | None = None,
) -> Any:
    import torch

    if source_tensor.ndim != 2 or target_tensor.ndim != 2:
        raise Full55MigrationError("expanded tensors must be matrices")
    if source_tensor.shape[0] != target_tensor.shape[0]:
        raise Full55MigrationError("expanded tensors changed output width")
    expanded = torch.zeros_like(target_tensor)
    for old_index, new_index in mapping.items():
        expanded[:, new_index] = source_tensor[:, old_index]
    if source_raw_width is not None or target_raw_width is not None:
        if source_raw_width is None or target_raw_width is None:
            raise Full55MigrationError("relation-column widths are incomplete")
        relation_count = source_tensor.shape[1] - source_raw_width
        if relation_count < 1 or target_tensor.shape[1] - target_raw_width != relation_count:
            raise Full55MigrationError("relation-column count changed")
        expanded[:, target_raw_width:] = source_tensor[:, source_raw_width:]
    return expanded


def expanded_policy_state(
    source_model: Any,
    target_model: Any,
    target_env: RevengeEnv,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return a strict target state with all old functions embedded by name."""

    import torch

    if tuple(int(value) for value in source_model.action_space.nvec) != (3, 180):
        raise Full55MigrationError("source action space is not [3, 180]")
    if tuple(int(value) for value in target_model.action_space.nvec) != (4, 180):
        raise Full55MigrationError("target action space is not [4, 180]")
    old_ball, old_projectile, old_global = _legacy_feature_names(source_model)
    ball_mapping = _column_mapping(old_ball, target_env.ball_feature_names)
    projectile_mapping = _column_mapping(
        old_projectile,
        target_env.projectile_feature_names,
    )
    global_mapping = _column_mapping(old_global, target_env.global_feature_names)

    source_state = source_model.policy.state_dict()
    target_state = target_model.policy.state_dict()
    source_only = set(source_state) - set(target_state)
    if source_only:
        raise Full55MigrationError(
            f"target policy lost parameters: {sorted(source_only)}"
        )

    transformed: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for name, target_value in target_state.items():
        target_tensor = target_value.detach().cpu()
        if name not in source_state:
            if not name.endswith("dynamic_shooter_mix"):
                raise Full55MigrationError(f"unexpected target-only parameter: {name}")
            if bool(torch.count_nonzero(target_tensor)):
                raise Full55MigrationError("dynamic shooter anchor is not zero")
            transformed[name] = target_tensor.clone()
            rows.append(
                {
                    "parameter": name,
                    "operation": "zero_initialize_dynamic_shooter_mix",
                    "target_shape": list(target_tensor.shape),
                }
            )
            continue

        source_tensor = source_state[name].detach().cpu()
        if tuple(source_tensor.shape) == tuple(target_tensor.shape):
            transformed[name] = source_tensor.clone()
            continue

        if name.endswith("ball_encoder.0.weight"):
            transformed[name] = _expand_linear_columns(
                source_tensor,
                target_tensor,
                ball_mapping,
                source_raw_width=len(old_ball),
                target_raw_width=len(target_env.ball_feature_names),
            )
            operation = "map_ball_columns_and_relations"
        elif name.endswith("projectile_encoder.0.weight"):
            transformed[name] = _expand_linear_columns(
                source_tensor,
                target_tensor,
                projectile_mapping,
            )
            operation = "map_projectile_columns"
        elif name.endswith(("ball_query.0.weight", "global_encoder.0.weight")):
            transformed[name] = _expand_linear_columns(
                source_tensor,
                target_tensor,
                global_mapping,
            )
            operation = "map_global_columns"
        elif name == "action_net.weight":
            if source_tensor.shape != (183, target_tensor.shape[1]) or target_tensor.shape[0] != 184:
                raise Full55MigrationError("action weight has an unexpected shape")
            expanded = torch.zeros_like(target_tensor)
            expanded[:3] = source_tensor[:3]
            expanded[4:] = source_tensor[3:]
            transformed[name] = expanded
            operation = "copy_three_verbs_zero_hop_shift_aim_rows"
        elif name == "action_net.bias":
            if source_tensor.shape != (183,) or target_tensor.shape != (184,):
                raise Full55MigrationError("action bias has an unexpected shape")
            expanded = torch.zeros_like(target_tensor)
            expanded[:3] = source_tensor[:3]
            expanded[4:] = source_tensor[3:]
            transformed[name] = expanded
            operation = "copy_three_verbs_zero_hop_shift_aim_rows"
        else:
            raise Full55MigrationError(
                f"unexpected parameter shape change for {name}: "
                f"{tuple(source_tensor.shape)} -> {tuple(target_tensor.shape)}"
            )
        rows.append(
            {
                "parameter": name,
                "operation": operation,
                "source_shape": list(source_tensor.shape),
                "target_shape": list(target_tensor.shape),
            }
        )
    return transformed, rows


def _make_target_model(source_model: Any, env: RevengeEnv, *, seed: int) -> Any:
    from sb3_contrib import MaskablePPO

    clip_range_vf = (
        None
        if source_model.clip_range_vf is None
        else float(source_model.clip_range_vf(1.0))
    )
    return MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=float(source_model.lr_schedule(1.0)),
        n_steps=int(source_model.n_steps),
        batch_size=min(int(source_model.batch_size), int(source_model.n_steps)),
        n_epochs=int(source_model.n_epochs),
        gamma=float(source_model.gamma),
        gae_lambda=float(source_model.gae_lambda),
        clip_range=float(source_model.clip_range(1.0)),
        clip_range_vf=clip_range_vf,
        normalize_advantage=bool(source_model.normalize_advantage),
        ent_coef=float(source_model.ent_coef),
        vf_coef=float(source_model.vf_coef),
        max_grad_norm=float(source_model.max_grad_norm),
        policy_kwargs=revenge_polar_policy_kwargs(env),
        seed=seed,
        device="cpu",
        verbose=0,
    )


def _collect_paired_proof(
    legacy_env: RevengeEnv,
    target_env: RevengeEnv,
    *,
    seed: int,
    steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    legacy_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    legacy_masks: list[np.ndarray] = []
    target_masks: list[np.ndarray] = []
    legacy_observation, _ = legacy_env.reset(seed=seed)
    target_observation, _ = target_env.reset(seed=seed)
    for tick in range(steps):
        if legacy_env.sim.state_signature() != target_env.sim.state_signature():
            raise Full55MigrationError(f"paired simulator states differ at tick {tick}")
        old_mask = legacy_env.action_masks()
        new_mask = target_env.action_masks()
        if not np.array_equal(old_mask[:3], new_mask[:3]):
            raise Full55MigrationError("legacy verb masks differ")
        if bool(new_mask[3]):
            raise Full55MigrationError("hop is not masked on the proof level")
        if not np.array_equal(old_mask[3:], new_mask[4:]):
            raise Full55MigrationError("aim masks differ")
        legacy_rows.append(legacy_observation.copy())
        target_rows.append(target_observation.copy())
        legacy_masks.append(old_mask.copy())
        target_masks.append(new_mask.copy())

        verb = 0
        if tick % 17 == 0 and bool(old_mask[1]):
            verb = 1
        elif tick % 29 == 0 and bool(old_mask[2]):
            verb = 2
        aim_bin = (seed + 47 * tick) % 180
        legacy_action = legacy_env.encode_action(verb, aim_bin)
        target_action = target_env.encode_action(verb, aim_bin)
        legacy_step = legacy_env.step(legacy_action)
        target_step = target_env.step(target_action)
        if legacy_step[1:4] != target_step[1:4]:
            raise Full55MigrationError(
                f"paired rewards or terminal flags differ at tick {tick}"
            )
        legacy_observation = legacy_step[0]
        target_observation = target_step[0]
        if legacy_step[2] or legacy_step[3]:
            break
    if len(legacy_rows) != steps:
        raise Full55MigrationError("proof episode terminated before the frozen batch")
    return (
        np.stack(legacy_rows),
        np.stack(target_rows),
        np.stack(legacy_masks),
        np.stack(target_masks),
    )


def prove_function_preservation(
    source_model: Any,
    target_model: Any,
    legacy_env: RevengeEnv,
    target_env: RevengeEnv,
    *,
    seed: int = PROOF_SEED,
    steps: int = PROOF_STEPS,
) -> dict[str, Any]:
    import torch

    old_obs, new_obs, old_masks, new_masks = _collect_paired_proof(
        legacy_env,
        target_env,
        seed=seed,
        steps=steps,
    )
    old_tensor = torch.as_tensor(old_obs, dtype=torch.float32)
    new_tensor = torch.as_tensor(new_obs, dtype=torch.float32)
    source_model.policy.set_training_mode(False)
    target_model.policy.set_training_mode(False)
    with torch.no_grad():
        source_distribution = source_model.policy.get_distribution(
            old_tensor,
            action_masks=old_masks,
        ).distributions
        target_distribution = target_model.policy.get_distribution(
            new_tensor,
            action_masks=new_masks,
        ).distributions
        if len(source_distribution) != 2 or len(target_distribution) != 2:
            raise Full55MigrationError("factorized distributions are unavailable")
        verb_error = float(
            torch.max(
                torch.abs(
                    source_distribution[0].logits
                    - target_distribution[0].logits[:, :3]
                )
            ).item()
        )
        aim_error = float(
            torch.max(
                torch.abs(
                    source_distribution[1].logits
                    - target_distribution[1].logits
                )
            ).item()
        )
        source_values = source_model.policy.predict_values(old_tensor)
        target_values = target_model.policy.predict_values(new_tensor)
        value_error = float(
            torch.max(torch.abs(source_values - target_values)).item()
        )
    source_actions, _ = source_model.predict(
        old_obs,
        deterministic=True,
        action_masks=old_masks,
    )
    target_actions, _ = target_model.predict(
        new_obs,
        deterministic=True,
        action_masks=new_masks,
    )
    actions_equal = bool(np.array_equal(source_actions, target_actions))
    mix_values = {
        name: float(value.detach().cpu().item())
        for name, value in target_model.policy.state_dict().items()
        if name.endswith("dynamic_shooter_mix")
    }
    max_log_error = max(verb_error, aim_error)
    status = (
        "PASS"
        if max_log_error <= MAX_LOG_PROBABILITY_ERROR
        and value_error <= MAX_VALUE_ERROR
        and actions_equal
        and mix_values
        and all(value == 0.0 for value in mix_values.values())
        else "FAIL"
    )
    return {
        "status": status,
        "level_id": PROOF_LEVEL,
        "seed": seed,
        "batch_size": steps,
        "max_verb_log_probability_error": verb_error,
        "max_aim_log_probability_error": aim_error,
        "max_value_error": value_error,
        "deterministic_actions_equal": actions_equal,
        "dynamic_shooter_mix": mix_values,
        "thresholds": {
            "max_log_probability_error": MAX_LOG_PROBABILITY_ERROR,
            "max_value_error": MAX_VALUE_ERROR,
        },
    }


def migrate_model_to_full55(
    *,
    source_model_path: str | Path,
    target_model_path: str | Path,
    original_root: str | Path,
    source_id: str,
    seed: int = PROOF_SEED,
) -> dict[str, Any]:
    """Migrate, prove, save, and return a hash-bound report."""

    from sb3_contrib import MaskablePPO

    source_path = Path(source_model_path).resolve(strict=True)
    target_path = Path(target_model_path).resolve(strict=False)
    root = Path(original_root).resolve(strict=True)
    if target_path.exists():
        raise Full55MigrationError(f"target model already exists: {target_path}")
    target_path.parent.mkdir(parents=True, exist_ok=True)

    legacy_env = RevengeEnv(
        legacy_migration_environment_config(),
        level_id=PROOF_LEVEL,
        root=root,
        profile_mode=SUPPORTED_PROFILE_MODE,
        seed=seed,
    )
    target_env = RevengeEnv(
        full55_environment_config(),
        level_id=PROOF_LEVEL,
        root=root,
        profile_mode=SUPPORTED_PROFILE_MODE,
        seed=seed,
    )
    try:
        source_model = MaskablePPO.load(source_path, device="cpu")
        target_model = _make_target_model(source_model, target_env, seed=seed)
        transformed, transformations = expanded_policy_state(
            source_model,
            target_model,
            target_env,
        )
        target_model.policy.load_state_dict(transformed, strict=True)
        proof = prove_function_preservation(
            source_model,
            target_model,
            legacy_env,
            target_env,
            seed=seed,
        )
        if proof["status"] != "PASS":
            raise Full55MigrationError(f"function-preservation proof failed: {proof}")
        target_model.num_timesteps = int(source_model.num_timesteps)
        target_model.save(target_path)
    finally:
        legacy_env.close()
        target_env.close()

    if not target_path.is_file():
        possible = Path(str(target_path) + ".zip")
        if possible.is_file():
            target_path = possible
        else:
            raise Full55MigrationError("target model was not written")
    reloaded = MaskablePPO.load(target_path, device="cpu")
    if tuple(int(value) for value in reloaded.action_space.nvec) != (4, 180):
        raise Full55MigrationError("saved target action space changed on reload")
    if tuple(int(value) for value in reloaded.observation_space.shape) != (22833,):
        raise Full55MigrationError("saved target observation space changed on reload")
    return {
        "schema": MIGRATION_SCHEMA,
        "version": MIGRATION_VERSION,
        "status": "PASS",
        "source_model": {
            "id": source_id,
            "path": str(source_path),
            "sha256": sha256_path(source_path),
            "observation_shape": list(source_model.observation_space.shape),
            "action_nvec": source_model.action_space.nvec.tolist(),
        },
        "migrated_model": {
            "path": str(target_path),
            "sha256": sha256_path(target_path),
            "observation_shape": list(reloaded.observation_space.shape),
            "action_nvec": reloaded.action_space.nvec.tolist(),
            "num_timesteps": int(reloaded.num_timesteps),
        },
        "proof": proof,
        "transformations": transformations,
    }
