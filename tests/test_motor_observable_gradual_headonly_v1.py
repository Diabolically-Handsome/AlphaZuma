from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tools import (
    build_alphazuma_55_motor_observable_gradual_headonly_v1 as builder,
)
from tools import (
    distill_alphazuma_55_motor_observable_gradual_headonly_v1 as headonly,
)


class _DummyPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(3, 4)
        self.action_net = nn.Linear(4, 184)
        self.value_net = nn.Linear(4, 1)
        self.optimizer_class = torch.optim.Adam
        self.optimizer_kwargs = {"eps": 1e-5}


def _model() -> SimpleNamespace:
    return SimpleNamespace(policy=_DummyPolicy())


def test_head_only_optimizer_excludes_every_backbone_parameter() -> None:
    headonly._MODEL_STATES.clear()
    model = _model()
    named = dict(model.policy.named_parameters())
    frozen_before = {
        name: value.detach().clone()
        for name, value in named.items()
        if name not in headonly.TRAINABLE_PARAMETER_NAMES
    }
    action_before = named["action_net.weight"].detach().clone()

    state = headonly._configure_head_only_optimizer(
        model=model, run={"learning_rate": 1e-2}
    )
    trainable_names = {
        name for name, value in named.items() if value.requires_grad
    }
    optimizer_ids = {
        id(parameter)
        for group in model.policy.optimizer.param_groups
        for parameter in group["params"]
    }
    assert trainable_names == set(headonly.TRAINABLE_PARAMETER_NAMES)
    assert optimizer_ids == {
        id(named[name]) for name in headonly.TRAINABLE_PARAMETER_NAMES
    }

    features = model.policy.backbone(torch.ones(2, 3))
    loss = model.policy.action_net(features).square().mean()
    model.policy.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    model.policy.optimizer.step()
    headonly._verify_frozen_parameters(model=model, state=state)

    assert not torch.equal(
        named["action_net.weight"].detach(), action_before
    )
    for name, expected in frozen_before.items():
        assert torch.equal(named[name].detach(), expected)


def test_head_only_invariant_detects_frozen_parameter_mutation() -> None:
    headonly._MODEL_STATES.clear()
    model = _model()
    state = headonly._configure_head_only_optimizer(
        model=model, run={"learning_rate": 1e-2}
    )
    with torch.no_grad():
        model.policy.backbone.bias.add_(1.0)
    with pytest.raises(
        RuntimeError, match="frozen head-only parameter changed"
    ):
        headonly._verify_frozen_parameters(model=model, state=state)


def test_builder_discloses_training_seed_reuse_and_formal_embargo(
    tmp_path, monkeypatch,
) -> None:
    monkeypatch.setattr(builder, "RUN_DIR", tmp_path / "run")
    value = builder.build(output=tmp_path / "preregistration.json")
    assert value["run"]["optimizer_parameter_scope"] == "action_net_only"
    assert value["run"]["aim_loss_scope"] == "fire_only"
    assert value["run"]["frozen_backbone"] is True
    assert value["seed_registry"]["fresh_training_randomness_claim"] is False
    assert value["seed_registry"]["formal_selection_seed_consumption"] == (
        "NONE"
    )
    assert value["authority_boundary"][
        "formal_final_blind_seed_consumption"
    ] is False
