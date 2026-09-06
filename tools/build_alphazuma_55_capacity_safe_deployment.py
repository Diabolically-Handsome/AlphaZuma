"""Freeze a capacity-safe six-route deployment after clean withdrawal."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tools import build_alphazuma_55_capacity_expansion_deployment as legacy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()
DEPLOYER = PROJECT_ROOT / "tools/deploy_alphazuma_55_postprocess_parallel_v3.py"
POSTPROCESS_BUILDER = PROJECT_ROOT / "tools/build_alphazuma_55_postprocess_parallel_v3.py"
CONTROLLER = PROJECT_ROOT / "tools/run_alphazuma_55_postprocess_parallel_v3.py"
EVALUATOR = PROJECT_ROOT / "tools/evaluate_zero_shot_multilevel_v2.py"
REMEDIATION = PROJECT_ROOT / "diagnostics/alphazuma-55-evaluation-capacity-remediation-s99081503-preregistration-v1.json"
RECOVERY_OUTCOME = PROJECT_ROOT / "diagnostics/alphazuma-55-horizon-probe-recovery-s99081502-outcome-v1.json"


def _artifact(path: Path) -> dict[str, str]:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "sha256": legacy._sha256(resolved)}


def _validated_remediation() -> dict[str, Any]:
    remediation = legacy._read_json(REMEDIATION.resolve(strict=True))
    outcome = legacy._read_json(RECOVERY_OUTCOME.resolve(strict=True))
    legacy._require(
        remediation.get("schema")
        == "zuma-rl.alphazuma-55-evaluation-capacity-remediation-preregistration"
        and remediation.get("status")
        == "FROZEN_DURING_RECOVERY_BEFORE_RECOVERY_RESULT",
        "unexpected capacity remediation preregistration",
    )
    legacy._require(
        outcome.get("schema")
        == "zuma-rl.alphazuma-55-horizon-probe-recovery-outcome"
        and outcome.get("status") == "PASS",
        "capacity recovery outcome is not PASS",
    )
    promotion = outcome.get("promotion")
    legacy._require(
        isinstance(promotion, dict)
        and promotion.get("capacity_safe_postprocess_authorized") is True
        and promotion.get("training_recipe_change_authorized") is False,
        "capacity recovery outcome does not authorize the narrow promotion",
    )
    capacity = outcome.get("capacity_recovery")
    legacy._require(
        isinstance(capacity, dict)
        and int(capacity.get("overflow_attempts", 0)) >= 1
        and capacity.get("all_overflows_are_explicit_losses") is True,
        "capacity recovery evidence is incomplete",
    )
    return {
        "remediation_preregistration": _artifact(REMEDIATION),
        "recovery_outcome": _artifact(RECOVERY_OUTCOME),
        "evaluator": _artifact(EVALUATOR),
        "promotion_scope": {
            "capacity_safe_postprocess_authorized": True,
            "training_recipe_change_authorized": False,
            "formal_seed_consumption_before_supersession": False,
        },
    }


def build_deployment(**kwargs: Any) -> dict[str, Any]:
    value = legacy.build_deployment(**kwargs)
    implementation = value["implementation"]
    implementation["deployer"] = _artifact(DEPLOYER)
    implementation["builder"] = _artifact(POSTPROCESS_BUILDER)
    implementation["controller"] = _artifact(CONTROLLER)
    value["capacity_expansion_deployment_builder"] = _artifact(SCRIPT_PATH)
    value["evaluation_capacity_remediation"] = _validated_remediation()
    value["created_utc"] = datetime.now(timezone.utc).isoformat()
    return value


def main(argv: list[str] | None = None) -> int:
    args = legacy.build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen deployment: {output}")
    value = build_deployment(
        expansion_path=args.expansion_preregistration.expanduser().resolve(strict=True),
        expected_expansion_sha256=args.expected_expansion_sha256,
        withdrawal_path=args.withdrawal_receipt.expanduser().resolve(strict=True),
        current_deployment_path=args.current_deployment.expanduser().resolve(strict=True),
    )
    legacy._write_json_exclusive(output, value)
    print(
        json.dumps(
            {
                "status": "FROZEN_CAPACITY_SAFE",
                "output": str(output),
                "sha256": legacy._sha256(output),
                "fixed_routes": len(value["fixed_routes"]),
                "registered_routes": len(value["fixed_routes"]) + 1,
                "implementation": value["implementation"],
                "formal_outputs": value["formal_outputs"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
