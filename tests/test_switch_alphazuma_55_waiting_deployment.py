import json
from pathlib import Path

import pytest

from tools import switch_alphazuma_55_waiting_deployment as switch


def _write(path: Path, value: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return switch._sha256(path)


def _reference(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": switch._sha256(path)}


def test_old_processes_require_script_and_contract_names() -> None:
    inventory = [
        {"pid": 10, "command": "python deploy.py --preregistration old-deploy.json"},
        {"pid": 11, "command": "python controller.py --preregistration old-post.json"},
        {"pid": 12, "command": "python controller.py --preregistration other-post.json"},
    ]

    result = switch._old_processes(
        inventory,
        deployer_name="deploy.py",
        deployment_name="old-deploy.json",
        controller_name="controller.py",
        postprocess_name="old-post.json",
    )

    assert [row["pid"] for row in result["deployer"]] == [10]
    assert [row["pid"] for row in result["controller"]] == [11]


def _switch_fixture(tmp_path: Path) -> tuple[Path, dict, Path, Path, Path]:
    dummy_withdrawal = tmp_path / "withdrawal.py"
    dummy_deployment = tmp_path / "deployment.py"
    dummy_deployer = tmp_path / "deployer.py"
    for path in (dummy_withdrawal, dummy_deployment, dummy_deployer):
        path.write_text("# fixture\n", encoding="utf-8")

    current_path = tmp_path / "old-deploy.json"
    current_hash = _write(
        current_path,
        {
            "schema": "zuma-rl.alphazuma-55-postprocess-deployment-preregistration",
            "status": "FROZEN_BEFORE_FORMAL_SEEDS",
            "postprocess_preregistration": str(tmp_path / "old-post.json"),
            "formal_outputs": {
                "controller": str(tmp_path / "old-controller"),
                "selection": str(tmp_path / "old-selection"),
                "final_blind": str(tmp_path / "old-final"),
                "continuous": str(tmp_path / "old-continuous"),
            },
            "deployment_output_root": str(tmp_path / "old-deployment-root"),
            "implementation": {
                "deployer": {"path": str(dummy_deployer)},
                "controller": {"path": str(tmp_path / "controller.py")},
            },
        },
    )
    milestone_path = tmp_path / "milestone.json"
    _write(milestone_path, {"status": "COMPLETE"})
    new_deployment_path = tmp_path / "new-deploy.json"
    expansion_path = tmp_path / "expansion.json"
    expansion_hash = _write(
        expansion_path,
        {
            "schema": "zuma-rl.alphazuma-55-capacity-expansion-preregistration",
            "version": 2,
            "status": "FROZEN_BEFORE_HARD_FRONTIER_TRAINING",
            "postprocess_supersession": {
                "current_deployment": {"path": str(current_path), "sha256": current_hash},
                "new_postprocess_preregistration": str(tmp_path / "new-post.json"),
                "new_deployment_preregistration": str(new_deployment_path),
                "new_independent_audit_receipt": str(tmp_path / "new-audit.json"),
                "new_deployment_output_root": str(tmp_path / "new-deployment-root"),
                "new_formal_outputs": {
                    "controller": str(tmp_path / "new-controller"),
                    "selection": str(tmp_path / "new-selection"),
                    "final_blind": str(tmp_path / "new-final"),
                    "continuous": str(tmp_path / "new-continuous"),
                },
            },
            "safe_switch_gate": {
                "wait_for_milestone_controller_to_reach_terminal_complete_or_error": str(milestone_path)
            },
        },
    )
    prereg = {
        "schema": "zuma-rl.alphazuma-55-waiting-deployment-switch",
        "version": 1,
        "status": "FROZEN_BEFORE_SWITCH",
        "implementation": {
            "orchestrator": _reference(Path(switch.__file__).resolve()),
            "withdrawal_builder": _reference(dummy_withdrawal),
            "deployment_builder": _reference(dummy_deployment),
            "deployer": _reference(dummy_deployer),
        },
        "capacity_expansion": {"path": str(expansion_path), "sha256": expansion_hash},
        "current_deployment": {"path": str(current_path), "sha256": current_hash},
        "milestone_controller": {"path": str(milestone_path)},
        "outputs": {
            "withdrawal_receipt": str(tmp_path / "withdrawal.json"),
            "new_deployment_preregistration": str(new_deployment_path),
            "switch_output_root": str(tmp_path / "switch-root"),
        },
        "poll_seconds": 1,
        "process_policy": {"graceful_exit_timeout_seconds": 10},
        "power_state": {
            "training_gpu_limits_watts": {"0": 550, "1": 250},
            "windows_training_power_plan_guid": "training-guid",
        },
    }
    prereg_path = tmp_path / "switch.json"
    _write(prereg_path, prereg)
    return prereg_path, prereg, current_path, expansion_path, milestone_path


def test_validate_preregistration_accepts_bound_version2_fixture(tmp_path: Path) -> None:
    prereg_path, prereg, _, _, _ = _switch_fixture(tmp_path)

    result = switch._validate_preregistration(prereg_path)

    assert result == prereg


def test_validate_preregistration_rejects_other_deployment_output(tmp_path: Path) -> None:
    prereg_path, prereg, _, _, _ = _switch_fixture(tmp_path)
    prereg["outputs"]["new_deployment_preregistration"] = str(tmp_path / "wrong.json")
    _write(tmp_path / "wrong-switch.json", prereg)

    with pytest.raises(ValueError, match="another new deployment"):
        switch._validate_preregistration(tmp_path / "wrong-switch.json")


def test_prestop_evidence_requires_terminal_clean_boundary(tmp_path: Path) -> None:
    _, prereg, current_path, _, milestone_path = _switch_fixture(tmp_path)
    current = json.loads(current_path.read_text(encoding="utf-8"))
    _write(
        Path(current["formal_outputs"]["controller"]) / "controller_status.json",
        {"phase": "WAITING_FOR_TRAINING", "error": None},
    )
    _write(
        Path(current["deployment_output_root"]) / "deployment_status.json",
        {"status": "RUNNING", "stage": "RUNNING_POSTPROCESS"},
    )
    milestone_path.write_text(json.dumps({"status": "COMPLETE"}), encoding="utf-8")
    inventory = [
        {
            "pid": 20,
            "command": f"python deployer.py --preregistration {current_path.name}",
        },
        {
            "pid": 21,
            "command": "python controller.py --preregistration old-post.json",
        },
    ]

    result = switch._prestop_evidence(
        prereg=prereg,
        inventory=inventory,
        gpu_limits={"0": 550.0, "1": 250.0},
        active_power_plan="Power Scheme training-guid",
    )

    assert result["status"] == "PASS"
    assert result["formal_boundary"]["formal_seed_consumption"] is False
    assert result["processes"]["controller"][0]["pid"] == 21
