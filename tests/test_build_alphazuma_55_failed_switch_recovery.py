from __future__ import annotations

import copy
from pathlib import Path

from tools import build_alphazuma_55_failed_switch_recovery as subject


def test_recovery_changes_only_metadata_and_switch_root(monkeypatch, tmp_path: Path) -> None:
    failed_path = tmp_path / "failed-switch.json"
    failed_path.write_text("{}", encoding="utf-8")
    failed_root = tmp_path / "failed-root"
    failed_root.mkdir()
    status_path = failed_root / "controller_status.json"
    failure_path = failed_root / "failure.json"
    failure_path.write_text("failure", encoding="utf-8")
    status_path.write_text("status", encoding="utf-8")
    milestone_path = tmp_path / "milestone.json"
    milestone_path.write_text("milestone", encoding="utf-8")
    frozen = {
        "created_utc": "old",
        "objective": "old objective",
        "outputs": {
            "switch_output_root": str(failed_root),
            "withdrawal_receipt": str(tmp_path / "withdrawal.json"),
            "new_deployment_preregistration": str(tmp_path / "deployment.json"),
        },
        "milestone_controller": {"path": str(milestone_path)},
        "selection_rules": {"frozen": True},
        "formal_semantics": {"registered_training_routes": 6},
    }
    failure = {
        "status": "ERROR",
        "error": {"message": "unexpected milestone status: CONTRACTS_FROZEN"},
    }
    status = {
        "status": "ERROR",
        "phase": "ERROR",
        "failure": {
            "path": str(failure_path),
            "sha256": "sha256:failure",
        },
    }

    monkeypatch.setattr(subject.legacy, "_sha256", lambda path: {
        failed_path: "sha256:failed",
        failure_path: "sha256:failure",
        status_path: "sha256:status",
        milestone_path: "sha256:milestone",
        subject.SCRIPT_PATH: "sha256:builder",
    }[Path(path)])
    monkeypatch.setattr(
        subject.legacy, "_validate_preregistration", lambda path: copy.deepcopy(frozen)
    )
    monkeypatch.setattr(
        subject.legacy,
        "_read_json",
        lambda path: {
            status_path: status,
            failure_path: failure,
            milestone_path: {"status": "COMPLETE"},
        }[Path(path)],
    )
    monkeypatch.setattr(
        subject,
        "_assert_clean_formal_boundary",
        lambda prereg: {
            "formal_controller_phase": "WAITING_FOR_TRAINING",
            "formal_seed_consumption": False,
            "prospective_six_route_outputs_absent": True,
        },
    )

    new_root = tmp_path / "recovery-root"
    result = subject.build(
        failed_switch_path=failed_path,
        expected_failed_switch_sha256="sha256:failed",
        switch_output_root=new_root,
    )

    assert result["outputs"]["switch_output_root"] == str(new_root.resolve())
    assert result["selection_rules"] == frozen["selection_rules"]
    assert result["formal_semantics"] == frozen["formal_semantics"]
    recovery = result["recovers_failed_switch"]
    assert recovery["training_semantics_changed"] is False
    assert recovery["candidate_rules_changed"] is False
    assert recovery["gate_changed"] is False
    assert recovery["formal_seeds_changed_or_consumed"] is False
