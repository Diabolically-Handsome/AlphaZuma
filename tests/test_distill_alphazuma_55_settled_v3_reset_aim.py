from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools.distill_alphazuma_55_settled_v3_reset_aim import _reset_aim_head
from tools import distill_alphazuma_55_settled_v3_reset_aim_v2 as recovered


def _dummy_model(*, nvec: tuple[int, int] = (4, 180)) -> SimpleNamespace:
    layer = torch.nn.Linear(11, sum(nvec))
    with torch.no_grad():
        layer.weight.copy_(
            torch.arange(layer.weight.numel(), dtype=torch.float32).reshape_as(
                layer.weight
            )
            / 1000.0
        )
        layer.bias.copy_(torch.arange(layer.bias.numel(), dtype=torch.float32))
    return SimpleNamespace(
        action_space=SimpleNamespace(nvec=np.asarray(nvec, dtype=np.int64)),
        policy=SimpleNamespace(action_net=layer),
    )


def test_reset_aim_head_zeros_only_aim_rows():
    model = _dummy_model()
    verb_weight = model.policy.action_net.weight[:4].detach().clone()
    verb_bias = model.policy.action_net.bias[:4].detach().clone()

    receipt = _reset_aim_head(model)

    assert receipt["status"] == "APPLIED_BEFORE_OPTIMIZER_REPLACEMENT_AND_COLLECTION"
    assert receipt["aim_weight_l2_before"] > 0.0
    assert receipt["aim_bias_l2_before"] > 0.0
    assert receipt["aim_weight_l2_after"] == 0.0
    assert receipt["aim_bias_l2_after"] == 0.0
    torch.testing.assert_close(model.policy.action_net.weight[:4], verb_weight)
    torch.testing.assert_close(model.policy.action_net.bias[:4], verb_bias)
    assert torch.count_nonzero(model.policy.action_net.weight[4:]).item() == 0
    assert torch.count_nonzero(model.policy.action_net.bias[4:]).item() == 0


def test_reset_aim_head_rejects_another_action_interface():
    with pytest.raises(ValueError, match="full55-v1 action interface"):
        _reset_aim_head(_dummy_model(nvec=(4, 179)))


def test_recovered_loader_restores_an_inherited_classmethod(monkeypatch):
    sentinel = object()

    class Parent:
        @classmethod
        def load(cls, path):
            del cls, path
            return sentinel

    class Child(Parent):
        pass

    monkeypatch.setattr(
        recovered,
        "_reset_aim_head",
        lambda model: {"model_is_sentinel": model is sentinel},
    )
    assert "load" not in Child.__dict__

    receipts, restore = recovered._install_reset_loader(Child)
    try:
        assert Child.load("unused") is sentinel
        assert receipts == [{"model_is_sentinel": True}]
        assert "load" in Child.__dict__
    finally:
        restore()

    assert "load" not in Child.__dict__
    assert Child.load("unused") is sentinel
