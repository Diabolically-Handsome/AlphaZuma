"""Freeze the conditional formal master for the head-only ablation route."""

from __future__ import annotations

import argparse
import copy
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
    build_alphazuma_55_motor_observable_gradual_allaim_successor_master_v1
    as legacy,
)
from tools import (
    run_alphazuma_55_motor_observable_gradual_headonly_successor_v1
    as controller,
)


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
CAMPAIGN_ID = controller.EXPECTED_CAMPAIGN
ENGINEERING_PLAN = PROJECT_ROOT / (
    "diagnostics/alphazuma-55-motor-observable-gradual-headonly-"
    "postprocess-s99081643-plan-v2.json"
)
ACTIVE_REGISTRY = PROJECT_ROOT / (
    "diagnostics/alphazuma-55-weekend-formal-seed-registry-s99081645-v9.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / (
    "diagnostics/alphazuma-55-motor-observable-gradual-headonly-"
    "successor-s99081644-preregistration-v1.json"
)


def _reference(path: Path) -> dict[str, str]:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "sha256": controller._sha256(resolved)}


def _write_new(path: Path, value: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def build(*, expected_plan_sha256: str, output: Path) -> dict[str, Any]:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"head-only successor master exists: {output}")
    captured: dict[str, Any] = {}
    originals = {
        "controller": legacy.controller,
        "CAMPAIGN_ID": legacy.CAMPAIGN_ID,
        "ENGINEERING_PLAN": legacy.ENGINEERING_PLAN,
        "ACTIVE_REGISTRY": legacy.ACTIVE_REGISTRY,
        "_write_new": legacy._write_new,
    }
    try:
        legacy.controller = controller
        legacy.CAMPAIGN_ID = CAMPAIGN_ID
        legacy.ENGINEERING_PLAN = ENGINEERING_PLAN
        legacy.ACTIVE_REGISTRY = ACTIVE_REGISTRY
        legacy._write_new = (
            lambda _path, value: captured.setdefault(
                "master", copy.deepcopy(value)
            )
        )
        legacy.build(
            expected_plan_sha256=expected_plan_sha256,
            output=output,
        )
    finally:
        for name, value in originals.items():
            setattr(legacy, name, value)

    master = captured["master"]
    output_root = Path(f"/mnt/d/ZumaTraining/{CAMPAIGN_ID}")
    master["campaign_id"] = CAMPAIGN_ID
    master["objective"]["target"] = (
        "Certify one frozen-backbone, action-head-adapted policy on all 55 "
        "included levels and four continuous no-retry campaigns."
    )
    master["formal_seed_registry"]["active_registry_path"] = str(
        ACTIVE_REGISTRY
    )
    master["implementation"].update(
        {
            "master_builder": _reference(SCRIPT_PATH),
            "controller": _reference(controller.SCRIPT_PATH),
            "independent_auditor": _reference(
                PROJECT_ROOT
                / "tools/audit_alphazuma_55_motor_observable_gradual_"
                "headonly_successor_v1.py"
            ),
            "engineering_controller": _reference(
                controller.engineering.SCRIPT_PATH
            ),
            "engineering_auditor": _reference(
                controller.engineering_auditor.SCRIPT_PATH
            ),
        }
    )
    master["outputs"].update(
        {
            "root": str(output_root),
            "controller": str(output_root / "controller"),
            "final_blind": str(output_root / "final-blind"),
            "continuous": str(output_root / "continuous"),
            "independent_audit_receipt": str(
                PROJECT_ROOT
                / "diagnostics/alphazuma-55-motor-observable-gradual-"
                "headonly-successor-s99081644-independent-audit-v1.json"
            ),
        }
    )
    master["route_contract"] = {
        "causal_ablation": "freeze_source_representation_train_action_net_only",
        "optimizer_parameter_scope": "action_net_only",
        "required_training_completion_frozen_parameters_unchanged": True,
        "engineering_gate_unchanged": True,
        "formal_gates_unchanged": True,
    }
    master["reader_lineage"] = {
        "engineering_timeout_reader": "canonical_v2",
        "formal_matrix_reader_already_supported_null_timeout": True,
        "formal_gate_changed": False,
        "formal_seed_ranges_are_route_unique": True,
    }
    _write_new(output, master)
    return master


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-engineering-plan-sha256", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    value = build(
        expected_plan_sha256=str(args.expected_engineering_plan_sha256),
        output=args.output,
    )
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
