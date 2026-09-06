"""Freeze an exact-semantics recovery for a failed waiting deployment switch."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tools import switch_alphazuma_55_waiting_deployment as legacy


SCRIPT_PATH = Path(__file__).resolve()


def _artifact(path: Path) -> dict[str, str]:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "sha256": legacy._sha256(resolved)}


def _assert_clean_formal_boundary(prereg: dict[str, Any]) -> dict[str, Any]:
    current_path = Path(str(prereg["current_deployment"]["path"])).resolve(
        strict=True
    )
    current = legacy._read_json(current_path)
    controller_status_path = (
        Path(str(current["formal_outputs"]["controller"])).resolve(strict=True)
        / "controller_status.json"
    )
    controller_status = legacy._read_json(controller_status_path)
    legacy._require(
        controller_status.get("status") == "RUNNING"
        and controller_status.get("phase") == "WAITING_FOR_TRAINING"
        and controller_status.get("error") is None,
        "formal controller is not cleanly waiting",
    )
    for name in ("selection", "final_blind", "continuous"):
        legacy._require(
            not Path(str(current["formal_outputs"][name])).exists(),
            f"formal {name} root already exists",
        )

    expansion_path = Path(str(prereg["capacity_expansion"]["path"])).resolve(
        strict=True
    )
    expansion = legacy._read_json(expansion_path)
    supersession = expansion["postprocess_supersession"]
    prospective_paths = [
        prereg["outputs"]["withdrawal_receipt"],
        prereg["outputs"]["new_deployment_preregistration"],
        supersession["new_postprocess_preregistration"],
        supersession["new_independent_audit_receipt"],
        supersession["new_deployment_output_root"],
        *supersession["new_formal_outputs"].values(),
    ]
    legacy._require(
        not any(Path(str(value)).exists() for value in prospective_paths),
        "one or more prospective six-route outputs already exist",
    )
    return {
        "formal_controller_status": _artifact(controller_status_path),
        "formal_controller_phase": controller_status["phase"],
        "formal_seed_consumption": False,
        "prospective_six_route_outputs_absent": True,
    }


def build(
    *,
    failed_switch_path: Path,
    expected_failed_switch_sha256: str,
    switch_output_root: Path,
) -> dict[str, Any]:
    failed_switch_path = failed_switch_path.resolve(strict=True)
    legacy._require(
        legacy._sha256(failed_switch_path) == expected_failed_switch_sha256,
        "failed switch preregistration hash mismatch",
    )
    failed = legacy._validate_preregistration(failed_switch_path)
    failed_root = Path(str(failed["outputs"]["switch_output_root"])).resolve(
        strict=True
    )
    status_path = failed_root / "controller_status.json"
    failure_path = failed_root / "failure.json"
    status = legacy._read_json(status_path)
    failure = legacy._read_json(failure_path)
    legacy._require(
        status.get("status") == "ERROR" and status.get("phase") == "ERROR",
        "switch recovery requires an ERROR controller status",
    )
    legacy._require(
        failure.get("status") == "ERROR"
        and failure.get("error", {}).get("message")
        == "unexpected milestone status: CONTRACTS_FROZEN",
        "switch recovery does not recognize the preserved failure",
    )
    status_failure = status.get("failure", {})
    legacy._require(
        Path(str(status_failure.get("path", ""))).resolve(strict=True)
        == failure_path
        and status_failure.get("sha256") == legacy._sha256(failure_path),
        "controller status does not bind the preserved failure",
    )
    for forbidden in ("prestop_evidence.json", "handoff.json"):
        legacy._require(
            not (failed_root / forbidden).exists(),
            f"failed switch already materialized {forbidden}",
        )

    milestone_path = Path(
        str(failed["milestone_controller"]["path"])
    ).resolve(strict=True)
    milestone = legacy._read_json(milestone_path)
    legacy._require(
        milestone.get("status") in {"COMPLETE", "ERROR"},
        "milestone is not terminal at recovery freeze time",
    )
    boundary = _assert_clean_formal_boundary(failed)
    legacy._require(
        not switch_output_root.exists(), "recovery switch output root already exists"
    )

    value = copy.deepcopy(failed)
    value["created_utc"] = datetime.now(timezone.utc).isoformat()
    value["objective"] = (
        "Recover the exact six-route capacity-safe deployment switch after its "
        "waiting-state parser exited before prestop or formal seed consumption."
    )
    value["outputs"]["switch_output_root"] = str(switch_output_root.resolve())
    value["recovers_failed_switch"] = {
        "preregistration": _artifact(failed_switch_path),
        "controller_status": _artifact(status_path),
        "failure": _artifact(failure_path),
        "milestone_at_recovery_freeze": {
            **_artifact(milestone_path),
            "status": milestone["status"],
        },
        "recovery_builder": _artifact(SCRIPT_PATH),
        "old_failure_evidence_preserved": True,
        "old_switch_restarted": False,
        "training_semantics_changed": False,
        "candidate_rules_changed": False,
        "gate_changed": False,
        "formal_seeds_changed_or_consumed": False,
        **boundary,
    }
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failed-switch-preregistration", required=True, type=Path)
    parser.add_argument("--expected-failed-switch-sha256", required=True)
    parser.add_argument("--switch-output-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen recovery: {output}")
    value = build(
        failed_switch_path=args.failed_switch_preregistration.expanduser(),
        expected_failed_switch_sha256=args.expected_failed_switch_sha256,
        switch_output_root=args.switch_output_root.expanduser(),
    )
    legacy._write_exclusive(output, value)
    print(
        json.dumps(
            {
                "status": "FROZEN_FAILED_SWITCH_RECOVERY",
                "output": str(output),
                "sha256": legacy._sha256(output),
                "switch_output_root": value["outputs"]["switch_output_root"],
                "formal_seed_consumption": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
