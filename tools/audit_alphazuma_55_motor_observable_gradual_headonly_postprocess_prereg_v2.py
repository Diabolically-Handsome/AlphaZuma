"""Independently audit the corrected head-only postprocess plan."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tools import (
    run_alphazuma_55_motor_observable_gradual_headonly_postprocess_v1
    as controller,
)


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
EXPECTED_CAMPAIGN = (
    "alphazuma-55-motor-observable-gradual-headonly-postprocess-s99081643-v2"
)
EXPECTED_RECOVERY_RECEIPT = PROJECT_ROOT / (
    "diagnostics/alphazuma-55-motor-observable-gradual-postprocess-"
    "s99081630-timeout-recovery-audit-v1.json"
)
EXPECTED_RECOVERY_SHA256 = (
    "sha256:e2f73963847962ca7de39e1bcf3d7ad6eed604145caf57e78d77d32223818ff6"
)
EXPECTED_SEED_INCIDENT_SHA256 = (
    "sha256:e957e188226f1b2b6f402e68c1da40a18d2c6c85c505973db885d7759bbc2bb5"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def audit(*, plan_path: Path, expected_plan_sha256: str) -> dict[str, Any]:
    plan_path = plan_path.resolve(strict=True)
    if _sha256(plan_path) != expected_plan_sha256:
        raise ValueError("postprocess plan hash differs")
    plan = controller.validate_plan(plan_path, expected_plan_sha256)
    recovery_path = EXPECTED_RECOVERY_RECEIPT.resolve(strict=True)
    recovery = _read(recovery_path)
    if not (
        _sha256(recovery_path) == EXPECTED_RECOVERY_SHA256
        and recovery.get("status") == "PASS"
        and recovery.get("controller_result") == "COMPLETE_NO_PROMOTION"
        and recovery.get("gate_passed") is False
        and recovery.get("formal_seed_consumption") == "NONE"
    ):
        raise ValueError("predecessor timeout incident audit differs")

    implementation_receipts: dict[str, dict[str, str]] = {}
    for name, reference in plan.get("implementation", {}).items():
        path = Path(str(reference.get("path", ""))).resolve(strict=True)
        actual = _sha256(path)
        if reference.get("sha256") != actual:
            raise ValueError(f"implementation hash differs: {name}")
        implementation_receipts[name] = {"path": str(path), "sha256": actual}

    screen = plan["screen"]
    gate = plan["full55_gate"]
    output_root = Path(str(plan["outputs"]["output_root"]))
    final_audit = Path(str(plan["outputs"]["independent_audit_receipt"]))
    prereg_audit = Path(
        str(plan["pre_inference_independent_audit_receipt"])
    )
    boundary = plan.get("authority_boundary", {})
    correction = plan.get("audit_path_correction", {})
    seed_incident = plan.get("training_seed_reuse_disclosure", {}).get(
        "incident", {}
    )
    seed_incident_path = Path(str(seed_incident.get("path", ""))).resolve(
        strict=True
    )
    if not (
        plan.get("status")
        == "FROZEN_DURING_TRAINING_BEFORE_CHECKPOINT_INFERENCE"
        and plan.get("campaign_id") == EXPECTED_CAMPAIGN
        and len(plan.get("screen_levels", [])) == 12
        and len(plan.get("full55_levels", [])) == 55
        and int(screen.get("base_seed", -1)) == 1_550_003_000
        and int(screen.get("last_seed", -1)) == 1_550_003_011
        and int(screen.get("expected_candidates", -1)) == 17
        and int(screen.get("expected_attempts", -1)) == 204
        and int(gate.get("base_seed", -1)) == 1_550_003_100
        and int(gate.get("last_seed", -1)) == 1_550_003_154
        and int(gate.get("maximum_candidates", -1)) == 5
        and int(gate.get("maximum_expected_attempts", -1)) == 275
        and int(gate.get("minimum_wins", -1)) == 35
        and int(gate.get("minimum_cleared_levels", -1)) == 35
        and boundary.get("classification")
        == "engineering_training_validation_only"
        and boundary.get("formal_selection_seed_consumption") is False
        and boundary.get("formal_final_blind_seed_consumption") is False
        and boundary.get("continuous_campaign_seed_consumption") is False
        and plan.get("fresh_seed_disclosure", {}).get(
            "formal_seed_consumption"
        )
        == "NONE"
        and plan.get("timeout_reader_correction", {}).get(
            "ranking_classification"
        )
        == "non_win"
        and correction.get("classification")
        == "PRE_INFERENCE_AUDIT_PATH_METADATA_CORRECTION"
        and correction.get("inference_already_consumed") is False
        and correction.get("seeds_changed") is False
        and correction.get("candidate_rule_changed") is False
        and correction.get("promotion_gate_changed") is False
        and seed_incident.get("sha256") == EXPECTED_SEED_INCIDENT_SHA256
        and _sha256(seed_incident_path) == EXPECTED_SEED_INCIDENT_SHA256
        and prereg_audit != final_audit
        and not prereg_audit.exists()
        and not final_audit.exists()
        and not output_root.exists()
    ):
        raise ValueError("head-only postprocess preregistration differs")

    return {
        "schema": "zuma-rl.alphazuma-55-headonly-postprocess-prereg-audit",
        "version": 2,
        "status": "PASS",
        "audited_utc": datetime.now(timezone.utc).isoformat(),
        "plan": {"path": str(plan_path), "sha256": _sha256(plan_path)},
        "auditor": {"path": str(SCRIPT_PATH), "sha256": _sha256(SCRIPT_PATH)},
        "predecessor_timeout_recovery": {
            "path": str(recovery_path),
            "sha256": _sha256(recovery_path),
            "result": "COMPLETE_NO_PROMOTION",
        },
        "implementation": implementation_receipts,
        "matrix": {
            "screen_candidates": 17,
            "screen_attempts": 204,
            "screen_seed_range": [1_550_003_000, 1_550_003_011],
            "maximum_gate_candidates": 5,
            "maximum_gate_attempts": 275,
            "gate_seed_range": [1_550_003_100, 1_550_003_154],
            "minimum_wins": 35,
            "minimum_cleared_levels": 35,
        },
        "training_seed_reuse_disclosed": True,
        "timeout_semantics": "canonical_max_tick_timeout_is_non_win",
        "output_root_absent_before_audit": True,
        "formal_seed_consumption": "NONE",
        "formal_candidate_authority": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    receipt = audit(
        plan_path=args.plan,
        expected_plan_sha256=str(args.expected_plan_sha256),
    )
    output = args.output.resolve()
    _write_new(output, receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "receipt": {
                    "path": str(output),
                    "sha256": _sha256(output.resolve(strict=True)),
                },
                "formal_seed_consumption": "NONE",
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
