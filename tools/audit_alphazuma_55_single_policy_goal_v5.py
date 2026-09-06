"""Audit the eight-route AlphaZuma 55 single-policy goal.

V5 adds the motor-observable successor.  That successor intentionally emits a
different independent-audit schema from the seven legacy routes, so this
adapter normalizes only its already independently reproduced result into the
common goal-proof shape.  It has no policy-inference authority.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools import audit_alphazuma_55_motor_observable_successor_v1 as motor
from tools import audit_alphazuma_55_single_policy_goal_v4 as v4


SCRIPT_PATH = Path(__file__).resolve()
base = v4.base
GoalAuditError = base.GoalAuditError
_sha256 = base._sha256
_read = base._read
_require = base._require
_artifact = base._artifact
_reference = base._reference
_utc = base._utc
_write_exclusive = base._write_exclusive
_LEGACY_CAMPAIGN_RESULT = base._campaign_result
MOTOR_KIND = "motor_observable_successor_v1"


def _recompute_motor_receipt(master_path: Path) -> dict[str, Any]:
    """Re-run the frozen motor auditor from immutable completed artifacts.

    The motor controller correctly invokes its auditor while its status is at
    RUNNING_INDEPENDENT_AUDIT, then atomically advances that same status file
    to COMPLETE.  For later reproducibility we verify the completed status and
    its receipt hash, reconstruct that former boundary in a temporary file,
    and pass it to the frozen auditor.  No production artifact is changed.
    """

    master_path = master_path.expanduser().resolve(strict=True)
    master = _read(master_path)
    outputs = master.get("outputs")
    _require(isinstance(outputs, dict), "motor successor outputs are absent")
    controller_root = Path(str(outputs.get("controller", ""))).resolve(strict=True)
    report_path = (controller_root / "final_report.json").resolve(strict=True)
    status_path = (controller_root / "controller_status.json").resolve(strict=True)
    receipt_path = Path(
        str(outputs.get("independent_audit_receipt", ""))
    ).resolve(strict=True)

    status = _read(status_path)
    allowed = {
        "COMPLETE_NO_PROMOTION": "COMPLETE_NO_PROMOTION",
        "COMPLETE_FORMAL_PASS": "COMPLETE",
        "COMPLETE_FORMAL_FAIL": "COMPLETE",
    }
    current_status = str(status.get("status", ""))
    _require(current_status in allowed, "motor successor controller is not terminal")
    _require(
        status.get("phase") == allowed[current_status],
        "motor successor terminal phase differs",
    )
    independent = status.get("independent_audit")
    _require(isinstance(independent, dict), "motor independent-audit anchor is absent")
    anchored_receipt = Path(str(independent.get("path", ""))).resolve(strict=True)
    _require(
        anchored_receipt == receipt_path
        and independent.get("sha256") == _sha256(receipt_path),
        "motor independent-audit receipt is not anchored by controller status",
    )
    final_report = status.get("final_report")
    _require(isinstance(final_report, dict), "motor final-report anchor is absent")
    _require(
        Path(str(final_report.get("path", ""))).resolve(strict=True) == report_path
        and final_report.get("sha256") == _sha256(report_path),
        "motor final report is not anchored by controller status",
    )

    boundary = deepcopy(status)
    boundary.update(
        {
            "status": "RUNNING",
            "phase": "RUNNING_INDEPENDENT_AUDIT",
            "final_report": {
                "path": str(report_path),
                "sha256": _sha256(report_path),
            },
        }
    )
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix="alphazuma-motor-audit-boundary-",
            suffix=".json",
            delete=False,
        ) as stream:
            json.dump(boundary, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            temporary_path = Path(stream.name)
        return motor.audit(
            master_path=master_path,
            expected_master_sha256=_sha256(master_path),
            controller_report=report_path,
            controller_status=temporary_path,
        )
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _production_recomputers() -> dict[str, dict[str, Any]]:
    recomputers = v4._production_recomputers()
    recomputers[MOTOR_KIND] = {
        "path": motor.SCRIPT_PATH,
        "call": _recompute_motor_receipt,
    }
    return recomputers


def _motor_campaign_result(
    campaign: dict[str, Any],
    *,
    deadline,
    now,
    recomputers: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    campaign_id = str(campaign["id"])
    kind = str(campaign["kind"])
    _require(kind == MOTOR_KIND, "motor campaign kind differs")
    master_path = _artifact(
        campaign.get("campaign_master"), f"{campaign_id} campaign master"
    )
    audit_input = _artifact(
        campaign.get("audit_input"), f"{campaign_id} audit input"
    )
    _require(master_path == audit_input, "motor campaign audit input differs from master")
    auditor_path = _artifact(
        campaign.get("independent_auditor"), f"{campaign_id} independent auditor"
    )
    _require(kind in recomputers, f"unsupported campaign kind: {kind}")
    recomputer = recomputers[kind]
    expected_auditor = Path(str(recomputer["path"])).resolve(strict=True)
    _require(auditor_path == expected_auditor, f"{campaign_id} binds another auditor")
    callback = recomputer.get("call")
    _require(callable(callback), f"{campaign_id} recomputer is not callable")

    receipt_path = Path(str(campaign["independent_receipt"])).expanduser().resolve()
    if not receipt_path.exists():
        return {
            "id": campaign_id,
            "kind": kind,
            "status": "MISSING_AT_DEADLINE" if now >= deadline else "PENDING",
            "independent_receipt": str(receipt_path),
            "eligible": False,
        }
    receipt = _read(receipt_path)
    recomputed = callback(audit_input)
    _require(isinstance(recomputed, dict), f"{campaign_id} auditor returned no object")
    _require(
        base._without_audit_time(receipt) == base._without_audit_time(recomputed),
        f"{campaign_id} receipt cannot be reproduced",
    )
    _require(receipt.get("status") == "PASS", f"{campaign_id} audit is not PASS")
    receipt_reference = _reference(receipt_path)
    controller_result = str(receipt.get("controller_result", ""))

    if controller_result == "COMPLETE_NO_PROMOTION":
        _require(
            receipt.get("engineering_gate_passed") is False
            and receipt.get("goal_gates_passed") is False
            and receipt.get("formal_seed_consumption") == "NONE",
            f"{campaign_id} no-promotion boundary differs",
        )
        return {
            "id": campaign_id,
            "kind": kind,
            "status": "TERMINAL_NO_PROMOTION",
            "independent_receipt": receipt_reference,
            "promotion_gate": {"passed": False, "motor_observable": True},
            "eligible": False,
        }

    _require(
        controller_result in {"COMPLETE_FORMAL_PASS", "COMPLETE_FORMAL_FAIL"},
        f"{campaign_id} controller is incomplete",
    )
    _require(
        receipt.get("engineering_gate_passed") is True
        and receipt.get("formal_seed_consumption") != "NONE",
        f"{campaign_id} formal inference lacks a passing engineering gate",
    )
    capability_pass = receipt.get("single_policy_capability_gate") is True
    continuous_pass = receipt.get("continuous_full_game_gate") is True
    goal_pass = receipt.get("goal_gates_passed") is True
    _require(
        goal_pass is (capability_pass and continuous_pass),
        f"{campaign_id} goal-gate conjunction differs",
    )
    artifacts = receipt.get("artifacts")
    _require(isinstance(artifacts, dict), f"{campaign_id} artifacts are absent")
    report_path = _artifact(
        artifacts.get("controller_report"), f"{campaign_id} controller report"
    )
    report = _read(report_path)
    _require(
        report.get("status") == controller_result,
        f"{campaign_id} controller report status differs",
    )

    if controller_result == "COMPLETE_FORMAL_FAIL":
        _require(not goal_pass, f"{campaign_id} formal fail claims goal success")
        return {
            "id": campaign_id,
            "kind": kind,
            "status": "TERMINAL_GATE_FAIL",
            "independent_receipt": receipt_reference,
            "final_report": _reference(report_path),
            "final_blind_attempts": base.FINAL_ATTEMPTS,
            "continuous_attempts": base.CONTINUOUS_ATTEMPTS,
            "single_policy_capability_gate": capability_pass,
            "continuous_full_game_gate": continuous_pass,
            "eligible": False,
        }

    _require(goal_pass, f"{campaign_id} formal pass lacks both goal gates")
    selected = receipt.get("selected_single_policy")
    _require(
        report.get("selected_single_policy") == selected,
        f"{campaign_id} selected policy differs between artifacts",
    )
    policy = base._verify_policy(selected)
    gates = report.get("success_gates")
    _require(isinstance(gates, dict), f"{campaign_id} report gates are absent")
    capability = gates.get("single_policy_capability_gate", {}).get("actual")
    _require(isinstance(capability, dict), f"{campaign_id} capability metrics are absent")
    _require(
        int(capability.get("levels_cleared", -1)) == base.LEVEL_COUNT
        and int(capability.get("minimum_wins_per_level", -1)) >= 1
        and int(capability.get("total_wins", -1)) >= 220
        and int(capability.get("attempts", -1)) == base.FINAL_ATTEMPTS,
        f"{campaign_id} capability metrics do not prove the 55-level goal",
    )
    continuous = report.get("continuous_campaigns")
    _require(isinstance(continuous, dict), f"{campaign_id} continuous evidence is absent")
    campaigns = continuous.get("campaigns")
    _require(
        isinstance(campaigns, list)
        and len(campaigns) == 4
        and any(row.get("cleared_all_55") is True for row in campaigns),
        f"{campaign_id} has no complete continuous 55-level campaign",
    )
    recomputed_section = receipt.get("recomputed")
    _require(isinstance(recomputed_section, dict), f"{campaign_id} recomputation is absent")
    _require(
        recomputed_section.get("single_policy_capability") == capability
        and recomputed_section.get("continuous_campaigns") == continuous,
        f"{campaign_id} report differs from independent recomputation",
    )
    return {
        "id": campaign_id,
        "kind": kind,
        "status": "ELIGIBLE",
        "independent_receipt": receipt_reference,
        "final_report": _reference(report_path),
        "selected_single_policy": policy,
        "capability": capability,
        "continuous_campaigns_cleared": int(continuous.get("campaigns_cleared", 0)),
        "final_blind_attempts": base.FINAL_ATTEMPTS,
        "continuous_attempts": base.CONTINUOUS_ATTEMPTS,
        "single_policy_capability_gate": True,
        "continuous_full_game_gate": True,
        "eligible": True,
    }


def _campaign_result(
    campaign: dict[str, Any],
    *,
    deadline,
    now,
    recomputers: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if str(campaign.get("kind", "")) == MOTOR_KIND:
        return _motor_campaign_result(
            campaign, deadline=deadline, now=now, recomputers=recomputers
        )
    return _LEGACY_CAMPAIGN_RESULT(
        campaign, deadline=deadline, now=now, recomputers=recomputers
    )


def audit(
    plan_path: Path,
    *,
    now=None,
    recomputers: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    original_script = v4.SCRIPT_PATH
    original_campaign_result = base._campaign_result
    try:
        v4.SCRIPT_PATH = SCRIPT_PATH
        base._campaign_result = _campaign_result
        return v4.audit(
            plan_path,
            now=now,
            recomputers=recomputers or _production_recomputers(),
        )
    finally:
        base._campaign_result = original_campaign_result
        v4.SCRIPT_PATH = original_script


def main(argv: list[str] | None = None) -> int:
    args = base.build_parser().parse_args(argv)
    receipt_path = args.receipt.expanduser().resolve()
    result = audit(args.plan)
    if args.status_only:
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    if receipt_path.exists():
        existing = _read(receipt_path)
        _require(existing.get("status") == "PASS", "existing goal receipt is not PASS")
        _require(result.get("status") == "PASS", "current evidence no longer passes")
        _require(
            existing.get("plan") == result.get("plan")
            and existing.get("selected_campaign_id")
            == result.get("selected_campaign_id")
            and existing.get("selected_single_policy")
            == result.get("selected_single_policy"),
            "existing goal receipt differs from current evidence",
        )
        print(json.dumps(existing, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    if result["status"] != "PASS":
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 2
    _write_exclusive(receipt_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
