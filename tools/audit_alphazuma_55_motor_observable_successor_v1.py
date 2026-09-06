"""Independently audit one motor-observable successor certification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tools import run_alphazuma_55_motor_observable_successor_v1 as controller


SCRIPT_PATH = Path(__file__).resolve()


def _reference(reference: Mapping[str, Any], label: str) -> Path:
    path = Path(str(reference.get("path", ""))).resolve(strict=True)
    controller._require(
        reference.get("sha256") == controller._sha256(path),
        f"{label} hash differs",
    )
    return path


def _validate_contract(
    *,
    contract: Mapping[str, Any],
    report: Mapping[str, Any],
    stage: str,
    expected_attempts: int,
    expected_seed: int,
    selected: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    section = report["final_blind"] if stage == "final_blind" else report["continuous"]
    manifest_path = _reference(section["manifest"], f"{stage} manifest")
    prereg_path = _reference(
        section["preregistration"], f"{stage} preregistration"
    )
    aggregate_path = _reference(section["aggregate"], f"{stage} aggregate")
    manifest = controller._read(manifest_path)
    prereg = controller._read(prereg_path)
    expected_model = dict(selected)
    controller._require(
        manifest.get("schema") == "zuma-rl.zero-shot-models-manifest"
        and manifest.get("status") == "FROZEN"
        and manifest.get("single_policy_only") is True
        and manifest.get("runtime_model_switching_forbidden") is True
        and manifest.get("models") == [expected_model],
        f"{stage} single-policy manifest changed",
    )
    master_ref = {
        "path": str(contract["path"]),
        "sha256": controller._sha256(Path(contract["path"])),
    }
    controller._require(
        prereg.get("schema") == "zuma-rl.zero-shot-multilevel-preregistration"
        and prereg.get("status") == "FROZEN_BEFORE_EVALUATION"
        and prereg.get("master_preregistration") == master_ref
        and prereg.get("model") == expected_model
        and [str(row["id"]) for row in prereg.get("levels", [])]
        == contract["levels"]
        and int(prereg.get("attempts_per_level", -1)) == expected_attempts
        and int(prereg.get("total_attempts", -1))
        == controller.LEVEL_COUNT * expected_attempts
        and int(prereg.get("seed_plan", {}).get("base_seed", -1))
        == expected_seed
        and int(prereg.get("seed_plan", {}).get("last_seed", -1))
        == expected_seed
        + controller.LEVEL_COUNT * expected_attempts
        - 1
        and prereg.get("seed_plan", {}).get("all_seeds_frozen_before_policy_inference")
        is True
        and prereg.get("authority_boundary", {}).get("single_frozen_policy")
        is True
        and prereg.get("authority_boundary", {}).get("runtime_model_switching")
        is False,
        f"{stage} preregistration changed",
    )
    evaluator = Path(str(prereg["evaluator"]["path"])).resolve(strict=True)
    controller._require(
        evaluator == contract["implementation"]["evaluator"]
        and prereg["evaluator"]["sha256"] == controller._sha256(evaluator),
        f"{stage} evaluator binding changed",
    )
    strict = controller._strict_audit_evaluation(
        root=aggregate_path.parent,
        preregistration_path=prereg_path,
        manifest_path=manifest_path,
        aggregate_path=aggregate_path,
        expected_level_ids=contract["levels"],
        expected_attempts_per_level=expected_attempts,
        expected_seed_base=expected_seed,
        expected_model_ids=[str(selected["id"])],
    )
    aggregate = controller._read(aggregate_path)
    controller._require(
        aggregate.get("status") == "COMPLETE",
        f"{stage} aggregate is incomplete",
    )
    return aggregate, strict


def audit(
    *,
    master_path: Path,
    expected_master_sha256: str,
    controller_report: Path,
    controller_status: Path,
) -> dict[str, Any]:
    master_path = master_path.resolve(strict=True)
    controller._require(
        controller._sha256(master_path) == expected_master_sha256,
        "motor successor master hash differs",
    )
    contract = controller._load_master(master_path)
    report_path = controller_report.resolve(strict=True)
    status_path = controller_status.resolve(strict=True)
    report = controller._read(report_path)
    status = controller._read(status_path)
    controller._require(
        status.get("status") == "RUNNING"
        and status.get("phase") == "RUNNING_INDEPENDENT_AUDIT"
        and status.get("final_report", {}).get("sha256")
        == controller._sha256(report_path),
        "controller was not frozen at independent-audit boundary",
    )
    decision_path = _reference(report["promotion_decision"], "promotion decision")
    decision = controller._read(decision_path)
    engineering_result = controller._engineering_result(contract)
    controller._require(
        bool(decision.get("promotion_gate_passed"))
        is bool(engineering_result["passed"])
        and decision.get("selected_policy") == engineering_result["model"],
        "promotion decision cannot be reproduced",
    )

    if not engineering_result["passed"]:
        controller._require(
            report.get("status") == "COMPLETE_NO_PROMOTION"
            and report.get("formal_seed_consumption") == "NONE"
            and report.get("goal_gates_passed") is False
            and not contract["outputs"]["final_blind"].exists()
            and not contract["outputs"]["continuous"].exists(),
            "negative promotion result consumed formal inference",
        )
        return {
            "schema": "zuma-rl.alphazuma-55-motor-successor-independent-audit",
            "version": 1,
            "status": "PASS",
            "audited_utc": controller._utc_now(),
            "controller_result": "COMPLETE_NO_PROMOTION",
            "engineering_gate_passed": False,
            "single_policy_capability_gate": False,
            "continuous_full_game_gate": False,
            "goal_gates_passed": False,
            "formal_seed_consumption": "NONE",
            "artifacts": {
                "master": {
                    "path": str(master_path),
                    "sha256": expected_master_sha256,
                },
                "controller_report": {
                    "path": str(report_path),
                    "sha256": controller._sha256(report_path),
                },
                "promotion_decision": {
                    "path": str(decision_path),
                    "sha256": controller._sha256(decision_path),
                },
                "auditor": {
                    "path": str(SCRIPT_PATH),
                    "sha256": controller._sha256(SCRIPT_PATH),
                },
            },
        }

    selected = dict(decision["selected_policy"])
    model_path = Path(str(selected["path"])).resolve(strict=True)
    controller._require(
        selected.get("sha256") == controller._sha256(model_path)
        and selected.get("observation_profile") == "motor-observable-v1"
        and bool(selected.get("promotable")) is True,
        "selected formal policy bytes or profile differ",
    )
    final_aggregate, final_strict = _validate_contract(
        contract=contract,
        report=report,
        stage="final_blind",
        expected_attempts=controller.FINAL_ATTEMPTS,
        expected_seed=controller.FINAL_SEED_BASE,
        selected=selected,
    )
    continuous_aggregate, continuous_strict = _validate_contract(
        contract=contract,
        report=report,
        stage="continuous",
        expected_attempts=controller.CONTINUOUS_CAMPAIGNS,
        expected_seed=controller.CONTINUOUS_SEED_BASE,
        selected=selected,
    )
    final_summary = next(
        row
        for row in final_aggregate["models"]
        if str(row["model_id"]) == str(selected["id"])
    )
    capability_actual, capability_passed = (
        controller.successor_helpers._capability(final_summary)
    )
    campaigns = controller.successor_helpers._continuous_campaigns(
        controller.successor_helpers._all_rows(contract["outputs"]["continuous"]),
        level_ids=contract["levels"],
        model_id=str(selected["id"]),
    )
    goal_passed = capability_passed and bool(campaigns["passed"])
    expected_status = (
        "COMPLETE_FORMAL_PASS" if goal_passed else "COMPLETE_FORMAL_FAIL"
    )
    controller._require(
        report.get("status") == expected_status
        and report.get("selected_single_policy") == selected
        and report.get("final_blind", {}).get("summary") == final_summary
        and report.get("continuous_campaigns") == campaigns
        and report.get("success_gates", {})
        .get("single_policy_capability_gate", {})
        .get("actual")
        == capability_actual
        and bool(
            report.get("success_gates", {})
            .get("single_policy_capability_gate", {})
            .get("passed")
        )
        is capability_passed
        and bool(
            report.get("success_gates", {})
            .get("continuous_full_game_gate", {})
            .get("passed")
        )
        is bool(campaigns["passed"])
        and bool(report.get("goal_gates_passed")) is goal_passed,
        "reported formal gates cannot be independently reproduced",
    )
    registry_path = _reference(
        report["formal_seed_registry"], "reported formal seed registry"
    )
    controller._require(
        registry_path == contract["registry"]["path"],
        "reported formal seed registry differs",
    )
    return {
        "schema": "zuma-rl.alphazuma-55-motor-successor-independent-audit",
        "version": 1,
        "status": "PASS",
        "audited_utc": controller._utc_now(),
        "controller_result": expected_status,
        "engineering_gate_passed": True,
        "single_policy_capability_gate": capability_passed,
        "continuous_full_game_gate": bool(campaigns["passed"]),
        "goal_gates_passed": goal_passed,
        "selected_single_policy": selected,
        "recomputed": {
            "single_policy_capability": capability_actual,
            "continuous_campaigns": campaigns,
            "final_strict_audit": final_strict,
            "continuous_strict_audit": continuous_strict,
            "continuous_aggregate_status": continuous_aggregate["status"],
        },
        "formal_seed_consumption": report["formal_seed_consumption"],
        "artifacts": {
            "master": {
                "path": str(master_path),
                "sha256": expected_master_sha256,
            },
            "controller_report": {
                "path": str(report_path),
                "sha256": controller._sha256(report_path),
            },
            "promotion_decision": {
                "path": str(decision_path),
                "sha256": controller._sha256(decision_path),
            },
            "formal_seed_registry": {
                "path": str(registry_path),
                "sha256": controller._sha256(registry_path),
            },
            "auditor": {
                "path": str(SCRIPT_PATH),
                "sha256": controller._sha256(SCRIPT_PATH),
            },
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", required=True, type=Path)
    parser.add_argument("--expected-master-sha256", required=True)
    parser.add_argument("--controller-report", required=True, type=Path)
    parser.add_argument("--controller-status", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    receipt = audit(
        master_path=args.master.expanduser(),
        expected_master_sha256=str(args.expected_master_sha256),
        controller_report=args.controller_report.expanduser(),
        controller_status=args.controller_status.expanduser(),
    )
    output = args.output.expanduser().resolve()
    controller._write_new(output, receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
