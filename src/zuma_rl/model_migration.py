"""Function-preserving migration of the pre-fruit actor model."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np

from zuma_rl.environment_contract import (
    build_training_environment_contract,
    training_environment_config,
)
from zuma_rl.revenge_core import SUPPORTED_PROFILE_MODE
from zuma_rl.revenge_env import RevengeEnv


MIGRATION_SCHEMA = "zuma-rl.fruit-actor-model-migration"
MIGRATION_VERSION = 2
POLICY_ID = "jungle2-transfer-bootstrap-v3"
PROOF_SEED = 20_260_951
PROOF_BATCH_SIZE = 64
MAX_OUTPUT_ERROR = 1e-6

FRUIT_GLOBAL_NAMES = (
    "fruit_present",
    "fruit_collectable",
    "fruit_x",
    "fruit_y",
    "fruit_collecting",
)


class ModelMigrationError(ValueError):
    """The source, target, or semantic migration contract is invalid."""


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


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve(strict=True).relative_to(root.resolve(strict=True)).as_posix()
    except (OSError, ValueError) as error:
        raise ModelMigrationError("model artifact is outside the evidence root") from error


def _resolve(root: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ModelMigrationError(f"{name} must be a relative POSIX path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise ModelMigrationError(f"{name} escapes the evidence root")
    try:
        resolved_root = root.resolve(strict=True)
        path = resolved_root.joinpath(*relative.parts).resolve(strict=True)
        path.relative_to(resolved_root)
    except (OSError, ValueError) as error:
        raise ModelMigrationError(f"{name} cannot be resolved") from error
    if not path.is_file():
        raise ModelMigrationError(f"{name} is not a file")
    return path


def _new_environment(
    *,
    original_root: Path,
    seed: int = PROOF_SEED,
) -> RevengeEnv:
    return RevengeEnv(
        training_environment_config(),
        level_id="Jungle2",
        root=original_root,
        hard=False,
        curve_index=0,
        profile_mode=SUPPORTED_PROFILE_MODE,
        seed=seed,
    )


def _global_column_mapping(old_extractor: Any, env: RevengeEnv) -> dict[int, int]:
    old_size = int(old_extractor.global_feature_size)
    old_current = int(old_extractor.global_current_color_start)
    old_next = int(old_extractor.global_next_color_start)
    colors = int(old_extractor.num_colors)
    new_names = tuple(env.global_feature_names)
    new_current = new_names.index("current_color_0")
    new_next = new_names.index("next_color_0")
    if (
        colors != env.num_colors
        or old_current != len(env._GLOBAL_BASE_FEATURES) - len(FRUIT_GLOBAL_NAMES)
        or old_next != old_current + colors
        or old_next + colors != old_size
        or new_current != old_current + len(FRUIT_GLOBAL_NAMES)
        or new_next != new_current + colors
        or new_next + colors != env.global_feature_size
    ):
        raise ModelMigrationError("source and target global layouts are incompatible")
    mapping = {index: index for index in range(old_current)}
    mapping.update(
        {old_current + index: new_current + index for index in range(colors)}
    )
    mapping.update(
        {old_next + index: new_next + index for index in range(colors)}
    )
    if set(mapping) != set(range(old_size)):
        raise ModelMigrationError("global-column mapping is incomplete")
    return mapping


def _expanded_state_dict(
    source_model: Any,
    target_model: Any,
    *,
    env: RevengeEnv,
) -> tuple[dict[str, Any], list[dict[str, Any]], tuple[int, ...]]:
    import torch

    source_state = source_model.policy.state_dict()
    target_state = target_model.policy.state_dict()
    if set(source_state) != set(target_state):
        raise ModelMigrationError("source and target policy parameter sets differ")
    mapping = _global_column_mapping(
        source_model.policy.features_extractor,
        env,
    )
    fruit_indices = tuple(
        env.global_feature_names.index(name) for name in FRUIT_GLOBAL_NAMES
    )
    transformed: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    expandable_suffixes = (
        "features_extractor.ball_query.0.weight",
        "features_extractor.global_encoder.0.weight",
    )
    for name in sorted(source_state):
        source_tensor = source_state[name].detach().cpu()
        target_tensor = target_state[name].detach().cpu()
        if tuple(source_tensor.shape) == tuple(target_tensor.shape):
            transformed[name] = source_tensor.clone()
            continue
        if not name.endswith(expandable_suffixes):
            raise ModelMigrationError(
                f"unexpected parameter shape change for {name}"
            )
        if (
            source_tensor.ndim != 2
            or target_tensor.ndim != 2
            or source_tensor.shape[0] != target_tensor.shape[0]
            or source_tensor.shape[1] != len(mapping)
            or target_tensor.shape[1] != env.global_feature_size
        ):
            raise ModelMigrationError(f"invalid expandable tensor {name}")
        expanded = torch.zeros_like(target_tensor)
        for old_index, new_index in mapping.items():
            expanded[:, new_index] = source_tensor[:, old_index]
        if any(bool(torch.count_nonzero(expanded[:, index])) for index in fruit_indices):
            raise ModelMigrationError("new fruit columns are not zero initialized")
        transformed[name] = expanded
        rows.append(
            {
                "parameter": name,
                "source_shape": list(source_tensor.shape),
                "target_shape": list(target_tensor.shape),
                "copied_source_column_count": len(mapping),
                "zero_initialized_target_columns": list(fruit_indices),
            }
        )
    if not rows or len(rows) % 2 != 0:
        raise ModelMigrationError(
            "global encoder matrix expansion set is invalid: "
            + ", ".join(row["parameter"] for row in rows)
        )
    return transformed, rows, fruit_indices


def _old_observations(
    current: np.ndarray,
    *,
    env: RevengeEnv,
    old_global_size: int,
) -> np.ndarray:
    globals_slice = env.observation_layout["globals"]
    globals_start = int(globals_slice.start or 0)
    old_current_start = env.global_feature_names.index("fruit_present")
    new_current_start = env.global_feature_names.index("current_color_0")
    old_globals = np.concatenate(
        (
            current[:, globals_start : globals_start + old_current_start],
            current[:, globals_start + new_current_start :],
        ),
        axis=1,
    )
    if old_globals.shape[1] != old_global_size:
        raise ModelMigrationError("old observation projection has the wrong width")
    return np.concatenate((current[:, :globals_start], old_globals), axis=1)


def _policy_outputs(model: Any, observations: np.ndarray) -> tuple[Any, Any, Any]:
    import torch

    tensor = torch.as_tensor(observations, dtype=torch.float32, device=model.device)
    with torch.no_grad():
        features = model.policy.extract_features(tensor)
        distribution = model.policy.get_distribution(tensor)
        categorical = getattr(distribution, "distributions", None)
        if not isinstance(categorical, Sequence):
            raise ModelMigrationError("policy does not expose factorized logits")
        logits = torch.cat([item.logits for item in categorical], dim=1)
        values = model.policy.predict_values(tensor)
    return (
        features.detach().cpu(),
        logits.detach().cpu(),
        values.detach().cpu(),
    )


def _functional_proof(
    source_model: Any,
    target_model: Any,
    *,
    env: RevengeEnv,
) -> dict[str, Any]:
    import torch

    generator = np.random.default_rng(PROOF_SEED)
    current = generator.uniform(
        -1.0,
        1.0,
        size=(PROOF_BATCH_SIZE, env.observation_size),
    ).astype(np.float32)
    globals_slice = env.observation_layout["globals"]
    globals_start = int(globals_slice.start or 0)
    fruit_indices = [
        env.global_feature_names.index(name) for name in FRUIT_GLOBAL_NAMES
    ]
    current[:, [globals_start + index for index in fruit_indices]] = (
        generator.uniform(-1.0, 1.0, size=(PROOF_BATCH_SIZE, len(fruit_indices)))
        .astype(np.float32)
    )
    old_size = int(source_model.observation_space.shape[0])
    old_global_size = int(
        source_model.policy.features_extractor.global_feature_size
    )
    source_observations = _old_observations(
        current,
        env=env,
        old_global_size=old_global_size,
    )
    if source_observations.shape != (PROOF_BATCH_SIZE, old_size):
        raise ModelMigrationError("source proof observations have the wrong shape")
    source_features, source_logits, source_values = _policy_outputs(
        source_model,
        source_observations,
    )
    target_features, target_logits, target_values = _policy_outputs(
        target_model,
        current,
    )
    feature_error = float(torch.max(torch.abs(source_features - target_features)))
    logit_error = float(torch.max(torch.abs(source_logits - target_logits)))
    value_error = float(torch.max(torch.abs(source_values - target_values)))
    source_argmax = torch.argmax(source_logits, dim=1)
    target_argmax = torch.argmax(target_logits, dim=1)
    exact_argmax = bool(torch.equal(source_argmax, target_argmax))
    passed = (
        feature_error <= MAX_OUTPUT_ERROR
        and logit_error <= MAX_OUTPUT_ERROR
        and value_error <= MAX_OUTPUT_ERROR
        and exact_argmax
    )
    return {
        "seed": PROOF_SEED,
        "batch_size": PROOF_BATCH_SIZE,
        "source_observation_shape": list(source_observations.shape),
        "target_observation_shape": list(current.shape),
        "source_observations_sha256": "sha256:"
        + hashlib.sha256(source_observations.tobytes()).hexdigest(),
        "target_observations_sha256": "sha256:"
        + hashlib.sha256(current.tobytes()).hexdigest(),
        "maximum_feature_error": feature_error,
        "maximum_logit_error": logit_error,
        "maximum_value_error": value_error,
        "maximum_allowed_error": MAX_OUTPUT_ERROR,
        "factorized_argmax_exact": exact_argmax,
        "status": "PASS" if passed else "FAIL",
    }


def _current_environment_smoke(model: Any, *, env: RevengeEnv) -> dict[str, Any]:
    digest = hashlib.sha256()
    observation, _ = env.reset(seed=PROOF_SEED)
    completed = 0
    finite = True
    for tick in range(64):
        mask = env.action_masks()
        action, _ = model.predict(
            observation,
            deterministic=True,
            action_masks=mask,
        )
        if not env.action_space.contains(action):
            raise ModelMigrationError("migrated model emitted an invalid action")
        observation, reward, terminated, truncated, _ = env.step(action)
        finite = finite and bool(np.isfinite(observation).all()) and bool(
            np.isfinite(reward)
        )
        digest.update(np.asarray(action, dtype="<i8").reshape(-1).tobytes())
        digest.update(np.asarray(observation, dtype="<f4").tobytes())
        completed = tick + 1
        if terminated or truncated:
            break
    return {
        "seed": PROOF_SEED,
        "requested_ticks": 64,
        "completed_ticks": completed,
        "finite_outputs": finite,
        "trajectory_sha256": "sha256:" + digest.hexdigest(),
        "status": "PASS" if finite and completed > 0 else "FAIL",
    }


def _load_models(source: Path, target: Path) -> tuple[Any, Any]:
    import torch
    from sb3_contrib import MaskablePPO
    from zuma_rl.revenge_features import revenge_polar_policy_kwargs

    torch.set_num_threads(1)
    source_model = MaskablePPO.load(source, device="cpu")
    target_model = MaskablePPO.load(target, device="cpu")
    return source_model, target_model


def audit_model_migration(
    *,
    evidence_root: str | Path,
    original_root: str | Path,
    source_model: str | Path,
    migrated_model: str | Path,
    environment_contract: str | Path,
) -> dict[str, Any]:
    """Independently reload both models and recompute migration semantics."""

    root = Path(evidence_root).resolve(strict=True)
    original = Path(original_root).resolve(strict=True)
    source_path = Path(source_model).resolve(strict=True)
    target_path = Path(migrated_model).resolve(strict=True)
    contract_path = Path(environment_contract).resolve(strict=True)
    source_relative = _relative(root, source_path)
    target_relative = _relative(root, target_path)
    contract_relative = _relative(root, contract_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    live_contract = build_training_environment_contract(original_root=original)
    if contract != live_contract:
        raise ModelMigrationError("bound environment contract is not current")
    source_loaded, target_loaded = _load_models(source_path, target_path)
    env = _new_environment(original_root=original)
    try:
        source_extractor = source_loaded.policy.features_extractor
        target_extractor = target_loaded.policy.features_extractor
        mapping = _global_column_mapping(source_extractor, env)
        transformed, transform_rows, fruit_indices = _expanded_state_dict(
            source_loaded,
            target_loaded,
            env=env,
        )
        target_state = target_loaded.policy.state_dict()
        transformed_exact = all(
            bool(target_state[name].detach().cpu().equal(tensor))
            for name, tensor in transformed.items()
        )
        fruit_columns_zero = True
        fruit_columns_trainable = True
        named_parameters = dict(target_loaded.policy.named_parameters())
        for row in transform_rows:
            parameter_name = row["parameter"]
            parameter = named_parameters.get(parameter_name)
            if parameter is None:
                for alias_prefix in ("pi_", "vf_"):
                    if parameter_name.startswith(
                        alias_prefix + "features_extractor."
                    ):
                        parameter = named_parameters.get(
                            parameter_name[len(alias_prefix) :]
                        )
                        break
            if parameter is None:
                raise ModelMigrationError(
                    f"expanded parameter alias cannot be resolved: {parameter_name}"
                )
            fruit_columns_zero = fruit_columns_zero and all(
                not bool(parameter[:, index].detach().count_nonzero())
                for index in fruit_indices
            )
            fruit_columns_trainable = (
                fruit_columns_trainable and bool(parameter.requires_grad)
            )
        functional = _functional_proof(
            source_loaded,
            target_loaded,
            env=env,
        )
        smoke = _current_environment_smoke(target_loaded, env=env)
        target_kwargs = target_loaded.policy_kwargs.get(
            "features_extractor_kwargs",
            {},
        )
        interface_pass = (
            list(target_loaded.observation_space.shape) == [env.observation_size]
            and [int(value) for value in target_loaded.action_space.nvec] == [3, 180]
            and int(target_extractor.global_feature_size) == env.global_feature_size
            and int(target_extractor.global_current_color_start)
            == env.global_feature_names.index("current_color_0")
            and int(target_extractor.global_next_color_start)
            == env.global_feature_names.index("next_color_0")
            and int(target_kwargs.get("global_feature_size", -1))
            == env.global_feature_size
        )
    finally:
        env.close()
    passed = (
        transformed_exact
        and fruit_columns_zero
        and fruit_columns_trainable
        and interface_pass
        and functional["status"] == "PASS"
        and smoke["status"] == "PASS"
    )
    payload: dict[str, Any] = {
        "schema": MIGRATION_SCHEMA,
        "version": MIGRATION_VERSION,
        "status": "PASS" if passed else "FAIL",
        "policy": POLICY_ID,
        "source_model": {
            "path": source_relative,
            "sha256": _sha256_path(source_path),
            "observation_size": int(source_loaded.observation_space.shape[0]),
            "global_feature_size": int(source_extractor.global_feature_size),
        },
        "migrated_model": {
            "path": target_relative,
            "sha256": _sha256_path(target_path),
            "observation_size": int(target_loaded.observation_space.shape[0]),
            "global_feature_size": int(target_extractor.global_feature_size),
        },
        "environment_contract": {
            "path": contract_relative,
            "sha256": _sha256_path(contract_path),
            "contract_fingerprint": contract.get("contract_fingerprint"),
        },
        "column_mapping": {
            "source_to_target": [
                {"source": old, "target": new}
                for old, new in sorted(mapping.items())
            ],
            "fruit_target_columns": list(fruit_indices),
        },
        "parameter_transforms": transform_rows,
        "checks": {
            "transformed_state_dict_exact": transformed_exact,
            "fruit_columns_zero_initialized": fruit_columns_zero,
            "fruit_columns_trainable": fruit_columns_trainable,
            "current_actor_interface_compatible": interface_pass,
        },
        "functional_preservation": functional,
        "current_environment_smoke": smoke,
        "failure_reasons": [] if passed else ["migration semantic check failed"],
    }
    payload["migration_fingerprint"] = _canonical_sha256(payload)
    return payload


def migrate_model_for_fruit_actor(
    *,
    evidence_root: str | Path,
    original_root: str | Path,
    source_model: str | Path,
    output_model: str | Path,
    environment_contract: str | Path,
) -> dict[str, Any]:
    """Create the expanded model, then reload it through the independent audit."""

    import torch
    from sb3_contrib import MaskablePPO

    source_path = Path(source_model).resolve(strict=True)
    output_path = Path(output_model).resolve(strict=False)
    if output_path.exists() or not output_path.parent.is_dir():
        raise ModelMigrationError("output model must be absent with an existing parent")
    torch.set_num_threads(1)
    source_loaded = MaskablePPO.load(source_path, device="cpu")
    env = _new_environment(original_root=Path(original_root).resolve(strict=True))
    try:
        target = MaskablePPO(
            "MlpPolicy",
            env,
            learning_rate=0.000005,
            n_steps=512,
            batch_size=512,
            n_epochs=1,
            gamma=0.995,
            gae_lambda=0.95,
            ent_coef=0.0,
            policy_kwargs=revenge_polar_policy_kwargs(env),
            seed=20_260_950,
            device="cpu",
            verbose=0,
        )
        transformed, _, _ = _expanded_state_dict(source_loaded, target, env=env)
        target.policy.load_state_dict(transformed, strict=True)
        target.num_timesteps = 0
        target.save(output_path)
    finally:
        env.close()
    if not output_path.is_file():
        raise ModelMigrationError("model save did not create the requested output")
    return audit_model_migration(
        evidence_root=evidence_root,
        original_root=original_root,
        source_model=source_path,
        migrated_model=output_path,
        environment_contract=environment_contract,
    )


def validate_model_migration_report(
    report: Mapping[str, Any],
    *,
    evidence_root: str | Path,
    original_root: str | Path | None,
) -> str | None:
    """Fail closed unless the report equals a fresh model-level recomputation."""

    if original_root is None:
        return "original_root is required for model migration verification"
    try:
        root = Path(evidence_root).resolve(strict=True)
        source = _resolve(root, report.get("source_model", {}).get("path"), "source model")
        target = _resolve(root, report.get("migrated_model", {}).get("path"), "migrated model")
        contract = _resolve(
            root,
            report.get("environment_contract", {}).get("path"),
            "environment contract",
        )
        live = audit_model_migration(
            evidence_root=root,
            original_root=original_root,
            source_model=source,
            migrated_model=target,
            environment_contract=contract,
        )
    except (
        ImportError,
        KeyError,
        ModelMigrationError,
        OSError,
        RuntimeError,
        ValueError,
    ):
        return "model migration could not be independently verified"
    if dict(report) != live:
        return "model migration report differs from live model semantics"
    return None


__all__ = [
    "MIGRATION_SCHEMA",
    "MIGRATION_VERSION",
    "ModelMigrationError",
    "audit_model_migration",
    "migrate_model_for_fruit_actor",
    "validate_model_migration_report",
]
