from __future__ import annotations

import copy
from pathlib import Path

from tools import build_alphazuma_55_capacity_safe_switch as subject


def test_build_rebinds_capacity_safe_handoff(monkeypatch, tmp_path):
    old_switch = (
        subject.PROJECT_ROOT
        / "diagnostics/alphazuma-55-postprocess-switch-s98081501-preregistration-v1.json"
    )
    launch = (
        subject.PROJECT_ROOT
        / "diagnostics/alphazuma-55-postprocess-switch-s98081501-launch-receipt-v1.json"
    )
    switch_root = tmp_path / "new-switch-root"
    frozen = subject.legacy._validate_preregistration(old_switch)
    fixture = copy.deepcopy(frozen)
    fixture["outputs"]["withdrawal_receipt"] = str(tmp_path / "withdrawal.json")
    fixture["outputs"]["new_deployment_preregistration"] = str(
        tmp_path / "deployment.json"
    )
    monkeypatch.setattr(
        subject.legacy,
        "_validate_preregistration",
        lambda path: copy.deepcopy(fixture),
    )

    value = subject.build(
        old_switch_path=old_switch.resolve(strict=True),
        old_launch_receipt_path=launch.resolve(strict=True),
        switch_output_root=switch_root,
    )

    assert value["implementation"]["deployment_builder"] == subject._artifact(
        subject.DEPLOYMENT_BUILDER
    )
    assert value["implementation"]["deployer"] == subject._artifact(subject.DEPLOYER)
    assert Path(value["outputs"]["switch_output_root"]) == switch_root.resolve()
    assert value["evaluation_capacity_remediation"]["training_semantics_changed"] is False
    assert value["evaluation_capacity_remediation"][
        "selection_final_and_continuous_seeds_changed"
    ] is False
