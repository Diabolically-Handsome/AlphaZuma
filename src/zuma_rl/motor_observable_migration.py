"""Exact policy transplant for the appended motor-observation interface."""

from __future__ import annotations

from typing import Any

import torch


_FEATURE_EXTRACTOR_PREFIXES = (
    "features_extractor",
    "pi_features_extractor",
    "vf_features_extractor",
)
_EXPANDED_GLOBAL_WEIGHT_SUFFIXES = (
    "ball_query.0.weight",
    "global_encoder.0.weight",
)


def _expanded_global_weight_keys(state: dict[str, torch.Tensor]) -> set[str]:
    """Return every serialized extractor replica that consumes globals."""

    keys = {
        f"{prefix}.{suffix}"
        for prefix in _FEATURE_EXTRACTOR_PREFIXES
        for suffix in _EXPANDED_GLOBAL_WEIGHT_SUFFIXES
        if f"{prefix}.{suffix}" in state
    }
    shared = {
        f"features_extractor.{suffix}"
        for suffix in _EXPANDED_GLOBAL_WEIGHT_SUFFIXES
    }
    if not shared.issubset(keys):
        raise ValueError("policy does not expose the shared global projections")
    for prefix in _FEATURE_EXTRACTOR_PREFIXES:
        present = {
            f"{prefix}.{suffix}" in state
            for suffix in _EXPANDED_GLOBAL_WEIGHT_SUFFIXES
        }
        if len(present) != 1:
            raise ValueError(
                f"policy serializes only part of extractor replica {prefix}"
            )
    return keys


def transplant_appended_global_policy(
    *,
    source_policy: Any,
    target_policy: Any,
    source_global_feature_size: int,
    target_global_feature_size: int,
) -> dict[str, Any]:
    """Copy a polar policy while zero-initializing appended global columns.

    ``ObservableHumanSpeedrunWrapper`` changes only the width of the global
    feature vector.  The first two global projections in every serialized
    extractor replica are therefore the only tensors allowed to differ in
    shape.  Their original columns are copied exactly and every new motor-state
    column starts at zero, proving that the migrated policy initially
    reproduces the source policy before fine-tuning.
    """

    source_width = int(source_global_feature_size)
    target_width = int(target_global_feature_size)
    if source_width < 1 or target_width <= source_width:
        raise ValueError("motor-observable migration requires appended globals")
    source_state = source_policy.state_dict()
    target_state = target_policy.state_dict()
    if set(source_state) != set(target_state):
        missing = sorted(set(source_state) - set(target_state))
        added = sorted(set(target_state) - set(source_state))
        raise ValueError(
            f"policy parameter inventory changed: missing={missing} added={added}"
        )
    expected_expanded_keys = _expanded_global_weight_keys(source_state)

    migrated: dict[str, torch.Tensor] = {}
    expanded: list[dict[str, Any]] = []
    unexpected_mismatches: list[dict[str, Any]] = []
    copied_parameters = 0
    for key, target_tensor in target_state.items():
        source_tensor = source_state[key]
        if tuple(source_tensor.shape) == tuple(target_tensor.shape):
            migrated[key] = source_tensor.detach().to(
                device=target_tensor.device,
                dtype=target_tensor.dtype,
            )
            copied_parameters += int(source_tensor.numel())
            continue
        if key not in expected_expanded_keys:
            unexpected_mismatches.append(
                {
                    "key": key,
                    "source_shape": list(source_tensor.shape),
                    "target_shape": list(target_tensor.shape),
                }
            )
            continue
        if not (
            source_tensor.ndim == 2
            and target_tensor.ndim == 2
            and int(source_tensor.shape[0]) == int(target_tensor.shape[0])
            and int(source_tensor.shape[1]) == source_width
            and int(target_tensor.shape[1]) == target_width
        ):
            unexpected_mismatches.append(
                {
                    "key": key,
                    "source_shape": list(source_tensor.shape),
                    "target_shape": list(target_tensor.shape),
                }
            )
            continue
        value = torch.zeros_like(target_tensor)
        value[:, :source_width].copy_(
            source_tensor.detach().to(
                device=value.device,
                dtype=value.dtype,
            )
        )
        migrated[key] = value
        copied_parameters += int(source_tensor.numel())
        expanded.append(
            {
                "key": key,
                "source_shape": list(source_tensor.shape),
                "target_shape": list(target_tensor.shape),
                "appended_columns_zero": True,
            }
        )
    if unexpected_mismatches:
        raise ValueError(
            f"unexpected motor-observable tensor mismatch: {unexpected_mismatches}"
        )
    expanded_keys = {row["key"] for row in expanded}
    if expanded_keys != expected_expanded_keys:
        raise ValueError(
            "motor-observable migration did not expand every global "
            f"projections: {sorted(expanded_keys)}"
        )
    target_policy.load_state_dict(migrated, strict=True)

    verified = target_policy.state_dict()
    for row in expanded:
        tensor = verified[row["key"]]
        if not torch.equal(
            tensor[:, :source_width].detach().cpu(),
            source_state[row["key"]].detach().cpu(),
        ):
            raise RuntimeError("motor-observable source columns changed")
        if bool(torch.count_nonzero(tensor[:, source_width:]).item()):
            raise RuntimeError("motor-observable appended columns are not zero")
    return {
        "schema": "zuma-rl.motor-observable-policy-migration",
        "version": 1,
        "status": "PASS",
        "source_global_feature_size": source_width,
        "target_global_feature_size": target_width,
        "appended_global_features": target_width - source_width,
        "copied_source_parameters": copied_parameters,
        "expanded_tensors": expanded,
        "initial_policy_ignores_appended_features": True,
    }
