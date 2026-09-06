from __future__ import annotations

from tools import build_alphazuma_55_capacity_safe_deployment as subject


def test_build_rebinds_only_capacity_safe_execution(monkeypatch):
    original = {
        "implementation": {
            "deployer": {"path": "old-deployer", "sha256": "old"},
            "builder": {"path": "old-builder", "sha256": "old"},
            "controller": {"path": "old-controller", "sha256": "old"},
            "auditor": {"path": "frozen-auditor", "sha256": "frozen"},
        },
        "fixed_routes": [{}, {}, {}, {}, {}],
        "formal_outputs": {
            "controller": "/controller",
            "selection": "/selection",
            "final_blind": "/final",
            "continuous": "/continuous",
        },
    }
    monkeypatch.setattr(
        subject.legacy,
        "build_deployment",
        lambda **kwargs: original,
    )

    value = subject.build_deployment(unused=True)

    assert value["implementation"]["deployer"] == subject._artifact(subject.DEPLOYER)
    assert value["implementation"]["builder"] == subject._artifact(subject.POSTPROCESS_BUILDER)
    assert value["implementation"]["controller"] == subject._artifact(subject.CONTROLLER)
    assert value["implementation"]["auditor"] == {
        "path": "frozen-auditor",
        "sha256": "frozen",
    }
    remediation = value["evaluation_capacity_remediation"]
    assert remediation["recovery_outcome"] == subject._artifact(subject.RECOVERY_OUTCOME)
    assert remediation["promotion_scope"] == {
        "capacity_safe_postprocess_authorized": True,
        "training_recipe_change_authorized": False,
        "formal_seed_consumption_before_supersession": False,
    }
