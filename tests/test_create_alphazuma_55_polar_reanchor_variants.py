from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from tools.create_alphazuma_55_polar_reanchor_variants import apply_transform


def _model() -> SimpleNamespace:
    action_net = torch.nn.Linear(5, 7)
    with torch.no_grad():
        action_net.weight.copy_(torch.arange(35, dtype=torch.float32).reshape(7, 5))
        action_net.bias.copy_(torch.arange(7, dtype=torch.float32))
    extractor = SimpleNamespace(learned_features_dim=2, polar_aim_bins=3)
    policy = SimpleNamespace(action_net=action_net, features_extractor=extractor)
    return SimpleNamespace(
        policy=policy,
        action_space=SimpleNamespace(nvec=np.asarray((4, 3), dtype=np.int64)),
    )


def test_reset_identity_changes_only_aim_rows() -> None:
    model = _model()
    verb_weight = model.policy.action_net.weight[:4].detach().clone()
    verb_bias = model.policy.action_net.bias[:4].detach().clone()

    receipt = apply_transform(model, mode="reset_identity", scale=5.0)

    assert torch.equal(model.policy.action_net.weight[:4], verb_weight)
    assert torch.equal(model.policy.action_net.bias[:4], verb_bias)
    assert torch.equal(model.policy.action_net.bias[4:], torch.zeros(3))
    expected = torch.zeros((3, 5))
    expected[torch.arange(3), 2 + torch.arange(3)] = 5.0
    assert torch.equal(model.policy.action_net.weight[4:], expected)
    assert receipt["verb_rows_bitwise_unchanged"] is True


def test_additive_identity_preserves_every_other_parameter() -> None:
    model = _model()
    before_weight = model.policy.action_net.weight.detach().clone()
    before_bias = model.policy.action_net.bias.detach().clone()

    apply_transform(model, mode="additive_identity", scale=2.0)

    expected = before_weight.clone()
    expected[4 + torch.arange(3), 2 + torch.arange(3)] += 2.0
    assert torch.equal(model.policy.action_net.weight, expected)
    assert torch.equal(model.policy.action_net.bias, before_bias)
