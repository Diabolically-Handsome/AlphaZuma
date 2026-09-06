import json
from pathlib import Path

import pytest

from tools import build_alphazuma_55_capacity_expansion_deployment as builder


def _write(path: Path, value: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return builder._sha256(path)


def _route(path: Path, route_id: str, first: int, last: int) -> dict:
    value = {
        "schema": "zuma-rl.overnight-multilevel-preregistration",
        "status": "FROZEN_BEFORE_TRAINING",
        "runs": [
            {
                "id": route_id,
                "episode_seed_base": first,
                "episode_seed_last": last,
            }
        ],
    }
    return {"path": str(path), "sha256": _write(path, value)}


def test_validate_withdrawal_rejects_formal_seed_consumption() -> None:
    withdrawal = {
        "schema": "zuma-rl.alphazuma-55-waiting-postprocess-withdrawal",
        "version": 1,
        "status": "PASS",
        "capacity_expansion": {"sha256": "expansion"},
        "withdrawn_deployment": {"sha256": "deployment"},
        "preconditions": {
            "old_controller_phase": "WAITING_FOR_TRAINING",
            "old_formal_roots_absent": True,
            "formal_seed_consumption": True,
        },
        "processes": {
            "old_controller_absent": True,
            "old_deployer_absent": True,
        },
        "milestone_controller": {"status": "COMPLETE"},
        "power_state": {"restoration_requested": False},
    }

    with pytest.raises(ValueError, match="formal boundary"):
        builder._validate_withdrawal(
            withdrawal,
            expansion_sha256="expansion",
            current_deployment_sha256="deployment",
        )


def test_build_deployment_adds_only_registered_replicate(tmp_path: Path) -> None:
    routes = [
        _route(tmp_path / f"route-{index}.json", f"route-{index}", 1000 + index * 100, 1099 + index * 100)
        for index in range(4)
    ]
    milestone_path = tmp_path / "milestone.json"
    milestone_hash = _write(milestone_path, {"status": "COMPLETE"})
    old_outputs = {
        "controller": str(tmp_path / "old-controller"),
        "selection": str(tmp_path / "old-selection"),
        "final_blind": str(tmp_path / "old-final"),
        "continuous": str(tmp_path / "old-continuous"),
    }
    current_path = tmp_path / "current.json"
    current = {
        "schema": "zuma-rl.alphazuma-55-postprocess-deployment-preregistration",
        "version": 1,
        "status": "FROZEN_BEFORE_FORMAL_SEEDS",
        "campaign_id": "campaign",
        "created_utc": "old",
        "fixed_routes": routes[:3],
        "expected_distilled_route": {
            "route_id": "distilled",
            "episode_seed_first": 2000,
            "episode_seed_last": 2099,
        },
        "formal_outputs": old_outputs,
        "parallelism_decision": {"sha256": "parallel"},
        "migrations": ["one", "two", "three"],
        "implementation": {"deployer": {"sha256": "code"}},
        "restore": {"balanced_guid": "balanced"},
        "supersedes": {"sha256": "old"},
        "postprocess_preregistration": str(tmp_path / "old-post.json"),
        "independent_audit_receipt": str(tmp_path / "old-audit.json"),
        "deployment_output_root": str(tmp_path / "old-deployment"),
    }
    current_hash = _write(current_path, current)
    expansion_path = tmp_path / "expansion.json"
    expansion = {
        "schema": "zuma-rl.alphazuma-55-capacity-expansion-preregistration",
        "version": 1,
        "status": "FROZEN_BEFORE_REPLICATE_TRAINING",
        "postprocess_supersession": {
            "current_deployment": {"path": str(current_path), "sha256": current_hash},
            "new_postprocess_preregistration": str(tmp_path / "new-post.json"),
            "new_deployment_preregistration": str(tmp_path / "new-deployment.json"),
            "new_independent_audit_receipt": str(tmp_path / "new-audit.json"),
            "new_deployment_output_root": str(tmp_path / "new-deployment-root"),
            "new_formal_outputs": {
                "controller": str(tmp_path / "new-controller"),
                "selection": str(tmp_path / "new-selection"),
                "final_blind": str(tmp_path / "new-final"),
                "continuous": str(tmp_path / "new-continuous"),
            },
        },
        "replicate_route": routes[3],
        "safe_switch_gate": {
            "wait_for_milestone_controller_to_reach_terminal_complete_or_error": str(milestone_path)
        },
    }
    expansion_hash = _write(expansion_path, expansion)
    withdrawal_path = tmp_path / "withdrawal.json"
    withdrawal = {
        "schema": "zuma-rl.alphazuma-55-waiting-postprocess-withdrawal",
        "version": 1,
        "status": "PASS",
        "capacity_expansion": {"sha256": expansion_hash},
        "withdrawal_builder": {
            "path": str(Path(builder.__file__).resolve()),
            "sha256": builder._sha256(Path(builder.__file__).resolve()),
        },
        "withdrawn_deployment": {"sha256": current_hash},
        "preconditions": {
            "old_controller_phase": "WAITING_FOR_TRAINING",
            "old_formal_roots_absent": True,
            "formal_seed_consumption": False,
        },
        "processes": {
            "old_controller_absent": True,
            "old_deployer_absent": True,
        },
        "milestone_controller": {
            "path": str(milestone_path),
            "status": "COMPLETE",
            "sha256": milestone_hash,
        },
        "power_state": {"restoration_requested": False},
    }
    _write(withdrawal_path, withdrawal)

    result = builder.build_deployment(
        expansion_path=expansion_path,
        expected_expansion_sha256=expansion_hash,
        withdrawal_path=withdrawal_path,
        current_deployment_path=current_path,
        process_inventory=[],
    )

    assert len(result["fixed_routes"]) == 4
    assert result["fixed_routes"][-1] == routes[3]
    assert result["expected_distilled_route"] == current["expected_distilled_route"]
    assert result["migrations"] == current["migrations"]
    assert result["formal_outputs"] == expansion["postprocess_supersession"]["new_formal_outputs"]


def test_expanded_route_references_adds_two_version2_routes(tmp_path: Path) -> None:
    routes = [
        _route(
            tmp_path / f"route-{index}.json",
            f"route-{index}",
            1000 + index * 100,
            1099 + index * 100,
        )
        for index in range(6)
    ]
    registry = []
    for index, reference in enumerate(routes):
        registry.append(
            {
                **reference,
                "run_id": f"route-{index}",
                "episode_seed_first": 1000 + index * 100,
                "episode_seed_last": 1099 + index * 100,
            }
        )
    expansion = {
        "version": 2,
        "route_registry": registry,
        "postprocess_supersession": {"registered_route_count": 6},
    }

    result = builder._expanded_route_references(
        expansion,
        current_fixed_routes=routes[:3],
        expected_distilled_route_id="route-3",
    )

    assert result == routes[4:]


def test_build_deployment_supports_six_route_version2(tmp_path: Path) -> None:
    routes = [
        _route(
            tmp_path / f"route-{index}.json",
            f"route-{index}",
            1000 + index * 100,
            1099 + index * 100,
        )
        for index in range(6)
    ]
    milestone_path = tmp_path / "milestone.json"
    milestone_hash = _write(milestone_path, {"status": "COMPLETE"})
    current_path = tmp_path / "current.json"
    current = {
        "schema": "zuma-rl.alphazuma-55-postprocess-deployment-preregistration",
        "version": 1,
        "status": "FROZEN_BEFORE_FORMAL_SEEDS",
        "campaign_id": "campaign",
        "created_utc": "old",
        "fixed_routes": routes[:3],
        "expected_distilled_route": {
            "route_id": "route-3",
            "episode_seed_first": 1300,
            "episode_seed_last": 1399,
        },
        "formal_outputs": {
            "controller": str(tmp_path / "old-controller"),
            "selection": str(tmp_path / "old-selection"),
            "final_blind": str(tmp_path / "old-final"),
            "continuous": str(tmp_path / "old-continuous"),
        },
        "parallelism_decision": {"sha256": "parallel"},
        "migrations": ["one", "two", "three"],
        "implementation": {"deployer": {"sha256": "code"}},
        "restore": {"balanced_guid": "balanced"},
        "postprocess_preregistration": str(tmp_path / "old-post.json"),
        "independent_audit_receipt": str(tmp_path / "old-audit.json"),
        "deployment_output_root": str(tmp_path / "old-deployment"),
    }
    current_hash = _write(current_path, current)
    registry = []
    for index, reference in enumerate(routes):
        registry.append(
            {
                **reference,
                "run_id": f"route-{index}",
                "episode_seed_first": 1000 + index * 100,
                "episode_seed_last": 1099 + index * 100,
            }
        )
    expansion_path = tmp_path / "expansion-v2.json"
    expansion = {
        "schema": "zuma-rl.alphazuma-55-capacity-expansion-preregistration",
        "version": 2,
        "status": "FROZEN_BEFORE_HARD_FRONTIER_TRAINING",
        "route_registry": registry,
        "postprocess_supersession": {
            "current_deployment": {"path": str(current_path), "sha256": current_hash},
            "registered_route_count": 6,
            "new_postprocess_preregistration": str(tmp_path / "new-post.json"),
            "new_deployment_preregistration": str(tmp_path / "new-deployment.json"),
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
    }
    expansion_hash = _write(expansion_path, expansion)
    withdrawal_path = tmp_path / "withdrawal.json"
    withdrawal = {
        "schema": "zuma-rl.alphazuma-55-waiting-postprocess-withdrawal",
        "version": 1,
        "status": "PASS",
        "capacity_expansion": {"sha256": expansion_hash},
        "withdrawal_builder": {
            "path": str(Path(builder.__file__).resolve()),
            "sha256": builder._sha256(Path(builder.__file__).resolve()),
        },
        "withdrawn_deployment": {"sha256": current_hash},
        "preconditions": {
            "old_controller_phase": "WAITING_FOR_TRAINING",
            "old_formal_roots_absent": True,
            "formal_seed_consumption": False,
        },
        "processes": {
            "old_controller_absent": True,
            "old_deployer_absent": True,
        },
        "milestone_controller": {
            "path": str(milestone_path),
            "status": "COMPLETE",
            "sha256": milestone_hash,
        },
        "power_state": {"restoration_requested": False},
    }
    _write(withdrawal_path, withdrawal)

    result = builder.build_deployment(
        expansion_path=expansion_path,
        expected_expansion_sha256=expansion_hash,
        withdrawal_path=withdrawal_path,
        current_deployment_path=current_path,
        process_inventory=[],
    )

    assert result["fixed_routes"] == routes[:3] + routes[4:]
    assert result["expected_distilled_route"] == current["expected_distilled_route"]
    assert len(result["fixed_routes"]) + 1 == 6
