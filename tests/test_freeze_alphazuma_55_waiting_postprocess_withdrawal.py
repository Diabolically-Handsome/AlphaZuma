import json
from pathlib import Path

from tools import freeze_alphazuma_55_waiting_postprocess_withdrawal as withdrawal


def _write(path: Path, value: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return withdrawal._sha256(path)


def test_old_processes_bind_both_script_and_contract_name() -> None:
    inventory = [
        {
            "pid": 10,
            "command": "python deploy_alphazuma_55_postprocess_parallel_v2.py --preregistration old-deploy.json",
        },
        {
            "pid": 11,
            "command": "python run_alphazuma_55_postprocess_parallel_v2.py --preregistration old-post.json",
        },
        {
            "pid": 12,
            "command": "python run_alphazuma_55_postprocess_parallel_v2.py --preregistration other-post.json",
        },
    ]

    value = withdrawal._old_processes(
        inventory,
        deployment_name="old-deploy.json",
        postprocess_name="old-post.json",
    )

    assert [row["pid"] for row in value["deployer"]] == [10]
    assert [row["pid"] for row in value["controller"]] == [11]


def test_old_processes_ignore_unrelated_python() -> None:
    value = withdrawal._old_processes(
        [{"pid": 20, "command": "python train_alphazuma_55.py --run-id model"}],
        deployment_name="old-deploy.json",
        postprocess_name="old-post.json",
    )

    assert value == {"deployer": [], "controller": []}


def test_freeze_withdrawal_accepts_version2_six_route_expansion(tmp_path: Path) -> None:
    training_guid = "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c"
    master_path = tmp_path / "master.json"
    master_hash = _write(
        master_path,
        {"hardware_contract": {"windows_training_power_plan": f"GUID {training_guid}"}},
    )
    controller_root = tmp_path / "old-controller"
    deployment_root = tmp_path / "old-deployment"
    _write(
        controller_root / "controller_status.json",
        {"phase": "WAITING_FOR_TRAINING", "error": None},
    )
    _write(
        deployment_root / "deployment_status.json",
        {"status": "RUNNING", "stage": "RUNNING_POSTPROCESS"},
    )
    current_path = tmp_path / "current.json"
    current_hash = _write(
        current_path,
        {
            "schema": "zuma-rl.alphazuma-55-postprocess-deployment-preregistration",
            "status": "FROZEN_BEFORE_FORMAL_SEEDS",
            "master_preregistration": {"path": str(master_path), "sha256": master_hash},
            "postprocess_preregistration": str(tmp_path / "old-postprocess.json"),
            "deployment_output_root": str(deployment_root),
            "formal_outputs": {
                "controller": str(controller_root),
                "selection": str(tmp_path / "old-selection"),
                "final_blind": str(tmp_path / "old-final"),
                "continuous": str(tmp_path / "old-continuous"),
            },
        },
    )
    milestone_path = tmp_path / "milestone.json"
    _write(milestone_path, {"status": "COMPLETE"})
    expansion_path = tmp_path / "expansion-v2.json"
    expansion_hash = _write(
        expansion_path,
        {
            "schema": "zuma-rl.alphazuma-55-capacity-expansion-preregistration",
            "version": 2,
            "status": "FROZEN_BEFORE_HARD_FRONTIER_TRAINING",
            "postprocess_supersession": {
                "current_deployment": {"path": str(current_path), "sha256": current_hash},
                "registered_route_count": 6,
                "new_postprocess_preregistration": str(tmp_path / "new-post.json"),
                "new_deployment_preregistration": str(tmp_path / "new-deploy.json"),
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

    result = withdrawal.freeze_withdrawal(
        expansion_path=expansion_path,
        expected_expansion_sha256=expansion_hash,
        current_deployment_path=current_path,
        expected_current_deployment_sha256=current_hash,
        process_inventory=[],
        gpu_limits={"0": 550.0, "1": 250.0},
        active_power_plan=f"Power Scheme GUID: {training_guid}",
    )

    assert result["status"] == "PASS"
    assert result["preconditions"]["new_expansion_artifacts_absent"] is True
    assert "6-route" in result["reason"]
