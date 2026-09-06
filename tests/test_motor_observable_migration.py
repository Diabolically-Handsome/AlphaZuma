from __future__ import annotations

import pytest
import torch
from torch import nn

from zuma_rl.motor_observable_migration import (
    transplant_appended_global_policy,
)


class _Extractor(nn.Module):
    def __init__(self, globals_width: int) -> None:
        super().__init__()
        self.ball_query = nn.Sequential(nn.Linear(globals_width, 4), nn.Tanh())
        self.global_encoder = nn.Sequential(
            nn.Linear(globals_width, 4), nn.ReLU(), nn.Linear(4, 4)
        )
        self.shared = nn.Linear(4, 3)


class _Policy(nn.Module):
    def __init__(self, globals_width: int, *, output_width: int = 2) -> None:
        super().__init__()
        self.features_extractor = _Extractor(globals_width)
        self.action_net = nn.Linear(3, output_width)


def test_transplant_copies_policy_and_zeros_appended_columns() -> None:
    torch.manual_seed(7)
    source = _Policy(5)
    target = _Policy(8)
    receipt = transplant_appended_global_policy(
        source_policy=source,
        target_policy=target,
        source_global_feature_size=5,
        target_global_feature_size=8,
    )
    source_state = source.state_dict()
    target_state = target.state_dict()
    for key, value in source_state.items():
        if key in {
            "features_extractor.ball_query.0.weight",
            "features_extractor.global_encoder.0.weight",
        }:
            assert torch.equal(target_state[key][:, :5], value)
            assert torch.count_nonzero(target_state[key][:, 5:]) == 0
        else:
            assert torch.equal(target_state[key], value)
    assert receipt["appended_global_features"] == 3
    assert receipt["initial_policy_ignores_appended_features"] is True


def test_transplant_rejects_unexpected_shape_change() -> None:
    source = _Policy(5, output_width=2)
    target = _Policy(8, output_width=3)
    with pytest.raises(ValueError, match="unexpected motor-observable tensor"):
        transplant_appended_global_policy(
            source_policy=source,
            target_policy=target,
            source_global_feature_size=5,
            target_global_feature_size=8,
        )
