"""Freeze the capacity-safe replacement for a waiting deployment switch."""

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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()
DEPLOYMENT_BUILDER = PROJECT_ROOT / "tools/build_alphazuma_55_capacity_safe_deployment.py"
DEPLOYER = PROJECT_ROOT / "tools/deploy_alphazuma_55_postprocess_parallel_v3.py"
REMEDIATION = PROJECT_ROOT / "diagnostics/alphazuma-55-evaluation-capacity-remediation-s99081503-preregistration-v1.json"
RECOVERY_OUTCOME = PROJECT_ROOT / "diagnostics/alphazuma-55-horizon-probe-recovery-s99081502-outcome-v1.json"
VALIDATED_PREVIEW = PROJECT_ROOT / "diagnostics/alphazuma-55-postprocess-capacity-safe-preview-s99081504-preregistration-v1.json"


def _artifact(path: Path) -> dict[str, str]:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "sha256": legacy._sha256(resolved)}


def build(
    *,
    old_switch_path: Path,
    old_launch_receipt_path: Path,
    switch_output_root: Path,
) -> dict[str, Any]:
    old_switch = legacy._validate_preregistration(old_switch_path)
    old_launch = legacy._read_json(old_launch_receipt_path)
    outcome = legacy._read_json(RECOVERY_OUTCOME.resolve(strict=True))
    legacy._require(
        old_launch.get("status") == "PASS"
        and old_launch.get("preregistration", {}).get("sha256")
        == legacy._sha256(old_switch_path),
        "old switch launch receipt does not bind the old switch",
    )
    legacy._require(
        outcome.get("status") == "PASS"
        and outcome.get("promotion", {}).get("capacity_safe_postprocess_authorized")
        is True
        and outcome.get("promotion", {}).get("training_recipe_change_authorized")
        is False,
        "capacity recovery outcome does not authorize switch supersession",
    )
    preview = legacy._read_json(VALIDATED_PREVIEW.resolve(strict=True))
    legacy._require(
        preview.get("schema") == "zuma-rl.alphazuma-55-postprocess-preregistration"
        and len(preview.get("training_routes", [])) == 6
        and preview.get("capacity_fail_closed_contract", {}).get(
            "exact_capacity_overflow_outcome"
        )
        == "loss",
        "capacity-safe six-route preview is invalid",
    )
    outputs = old_switch["outputs"]
    legacy._require(
        not switch_output_root.exists(),
        "capacity-safe switch output root already exists",
    )
    legacy._require(
        not Path(str(outputs["withdrawal_receipt"])).exists()
        and not Path(str(outputs["new_deployment_preregistration"])).exists(),
        "old switch has already materialized a handoff artifact",
    )

    value = copy.deepcopy(old_switch)
    value["created_utc"] = datetime.now(timezone.utc).isoformat()
    value["objective"] = (
        "After the frozen milestone becomes terminal, replace the waiting "
        "four-route deployment with the six-route capacity-safe deployment "
        "before any formal seed is consumed."
    )
    value["implementation"]["deployment_builder"] = _artifact(DEPLOYMENT_BUILDER)
    value["implementation"]["deployer"] = _artifact(DEPLOYER)
    value["outputs"]["switch_output_root"] = str(switch_output_root.resolve())
    value["capacity_safe_switch_builder"] = _artifact(SCRIPT_PATH)
    value["supersedes_waiting_switch"] = {
        "preregistration": _artifact(old_switch_path),
        "launch_receipt": _artifact(old_launch_receipt_path),
        "reason": "capacity overflow recovery audit passed before formal seed consumption",
        "old_switch_may_not_materialize_handoff": True,
    }
    value["evaluation_capacity_remediation"] = {
        "preregistration": _artifact(REMEDIATION),
        "recovery_outcome": _artifact(RECOVERY_OUTCOME),
        "validated_six_route_preview": _artifact(VALIDATED_PREVIEW),
        "training_semantics_changed": False,
        "selection_final_and_continuous_seeds_changed": False,
    }
    value["execution_sequence"] = [
        "wait for terminal milestone without changing training",
        "freeze immutable prestop evidence",
        "terminate only the exact waiting V2 controller and allow its deployer to exit",
        "freeze the V2 withdrawal receipt",
        "freeze and validate the six-route capacity-safe deployment preregistration",
        "exec-replace this controller with the capacity-safe deployer",
        "new deployer freezes and validates the capacity-safe postprocess then waits for the original training deadline",
    ]
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-switch-preregistration", required=True, type=Path)
    parser.add_argument("--old-launch-receipt", required=True, type=Path)
    parser.add_argument("--switch-output-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen switch: {output}")
    value = build(
        old_switch_path=args.old_switch_preregistration.expanduser().resolve(strict=True),
        old_launch_receipt_path=args.old_launch_receipt.expanduser().resolve(strict=True),
        switch_output_root=args.switch_output_root.expanduser().resolve(),
    )
    legacy._write_exclusive(output, value)
    print(
        json.dumps(
            {
                "status": "FROZEN_CAPACITY_SAFE_SWITCH",
                "output": str(output),
                "sha256": legacy._sha256(output),
                "switch_output_root": value["outputs"]["switch_output_root"],
                "deployment_builder": value["implementation"]["deployment_builder"],
                "deployer": value["implementation"]["deployer"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
