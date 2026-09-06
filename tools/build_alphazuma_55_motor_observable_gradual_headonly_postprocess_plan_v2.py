"""Correct head-only postprocess audit paths before any inference."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
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
CAMPAIGN_ID = (
    "alphazuma-55-motor-observable-gradual-headonly-postprocess-s99081643-v2"
)
BASE_PLAN = (
    PROJECT_ROOT
    / "diagnostics/alphazuma-55-motor-observable-gradual-headonly-"
    "postprocess-s99081643-plan-v1.json"
)
BASE_PLAN_SHA256 = (
    "sha256:24ecc5dcfef99070f58bee31ad319730b1899e9d11979945a5ed19c5d0e4ae51"
)
INCIDENT = (
    PROJECT_ROOT
    / "diagnostics/alphazuma-55-motor-observable-gradual-headonly-"
    "postprocess-s99081643-audit-path-incident-v1.json"
)
INCIDENT_SHA256 = (
    "sha256:d55cf7e553dd691cdb53f4a32a134350c560b99568f96e881044fa5002d5d93a"
)
PREREG_AUDITOR = (
    PROJECT_ROOT
    / "tools/audit_alphazuma_55_motor_observable_gradual_"
    "headonly_postprocess_prereg_v2.py"
)
OUTPUT_ROOT = Path(f"/mnt/d/ZumaTraining/{CAMPAIGN_ID}")
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "diagnostics/alphazuma-55-motor-observable-gradual-headonly-"
    "postprocess-s99081643-plan-v2.json"
)
PREREG_AUDIT_OUTPUT = (
    PROJECT_ROOT
    / "diagnostics/alphazuma-55-motor-observable-gradual-headonly-"
    "postprocess-s99081643-independent-prereg-audit-v2.json"
)
FINAL_AUDIT_OUTPUT = (
    PROJECT_ROOT
    / "diagnostics/alphazuma-55-motor-observable-gradual-headonly-"
    "postprocess-s99081643-independent-audit-v2.json"
)


def _reference(path: Path, expected_sha256: str | None = None) -> dict[str, str]:
    resolved = path.resolve(strict=True)
    actual = controller._sha256(resolved)
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError(f"frozen bytes differ: {resolved}")
    return {"path": str(resolved), "sha256": actual}


def _write_new(path: Path, value: dict[str, Any]) -> None:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    ) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def build(*, output: Path) -> dict[str, Any]:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"postprocess plan already exists: {output}")
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"postprocess output already exists: {OUTPUT_ROOT}")
    base_ref = _reference(BASE_PLAN, BASE_PLAN_SHA256)
    incident_ref = _reference(INCIDENT, INCIDENT_SHA256)
    plan = copy.deepcopy(controller._read(BASE_PLAN))
    plan.update(
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "campaign_id": CAMPAIGN_ID,
            "supersedes": base_ref,
            "pre_inference_independent_audit_receipt": str(
                PREREG_AUDIT_OUTPUT
            ),
        }
    )
    plan["implementation"].update(
        {
            "builder": _reference(SCRIPT_PATH),
            "preregistration_auditor": _reference(PREREG_AUDITOR),
        }
    )
    plan["outputs"] = {
        "output_root": str(OUTPUT_ROOT),
        "independent_audit_receipt": str(FINAL_AUDIT_OUTPUT),
    }
    plan["audit_path_correction"] = {
        "classification": "PRE_INFERENCE_AUDIT_PATH_METADATA_CORRECTION",
        "incident": incident_ref,
        "inference_already_consumed": False,
        "seeds_changed": False,
        "models_changed": False,
        "candidate_rule_changed": False,
        "promotion_gate_changed": False,
        "timeout_semantics_changed": False,
        "formal_seed_consumption": "NONE",
    }
    _write_new(output, plan)
    return plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    plan = build(output=args.output)
    print(
        json.dumps(
            {
                "status": plan["status"],
                "campaign_id": plan["campaign_id"],
                "plan": {
                    "path": str(args.output.resolve()),
                    "sha256": controller._sha256(
                        args.output.resolve(strict=True)
                    ),
                },
                "audit_paths_are_distinct": (
                    plan["pre_inference_independent_audit_receipt"]
                    != plan["outputs"]["independent_audit_receipt"]
                ),
                "formal_seed_consumption": False,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
