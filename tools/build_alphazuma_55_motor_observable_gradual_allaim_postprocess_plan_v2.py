"""Freeze timeout-compatible postprocessing for all-actions motor DAgger."""

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
    audit_alphazuma_55_motor_observable_timeout_recovery_v1 as timeout_reader,
)
from tools import (
    build_alphazuma_55_motor_observable_gradual_allaim_postprocess_plan_v1 as legacy,
)
from tools import (
    run_alphazuma_55_motor_observable_gradual_allaim_postprocess_v2 as controller,
)


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
CAMPAIGN_ID = (
    "alphazuma-55-motor-observable-gradual-allaim-postprocess-s99081634-v2"
)
OUTPUT_ROOT = Path(f"/mnt/d/ZumaTraining/{CAMPAIGN_ID}")
DEFAULT_OUTPUT = PROJECT_ROOT / (
    "diagnostics/alphazuma-55-motor-observable-gradual-allaim-"
    "postprocess-s99081634-plan-v2.json"
)
AUDITOR = PROJECT_ROOT / (
    "tools/audit_alphazuma_55_motor_observable_gradual_allaim_postprocess_v2.py"
)
AUDIT_OUTPUT = PROJECT_ROOT / (
    "diagnostics/alphazuma-55-motor-observable-gradual-allaim-"
    "postprocess-s99081634-independent-audit-v2.json"
)


def _reference(path: Path) -> dict[str, str]:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "sha256": controller._sha256(resolved)}


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
    captured: dict[str, Any] = {}
    original_write = legacy.controller.base._write_new
    try:
        legacy.controller.base._write_new = (
            lambda _path, value: captured.setdefault("plan", copy.deepcopy(value))
        )
        legacy.build(output=output)
    finally:
        legacy.controller.base._write_new = original_write
    plan = captured["plan"]
    plan.update(
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "campaign_id": CAMPAIGN_ID,
            "objective": (
                "Screen the frozen all-actions checkpoints and apply the "
                "unchanged 35-win/35-level Gate while accepting only the "
                "evaluator's canonical max-tick timeout representation."
            ),
        }
    )
    plan["implementation"].update(
        {
            "builder": _reference(SCRIPT_PATH),
            "controller": _reference(controller.SCRIPT_PATH),
            "auditor": _reference(AUDITOR),
            "timeout_compatible_matrix_auditor": _reference(
                timeout_reader.SCRIPT_PATH
            ),
        }
    )
    plan["outputs"] = {
        "output_root": str(OUTPUT_ROOT),
        "independent_audit_receipt": str(AUDIT_OUTPUT),
    }
    plan["timeout_reader_correction"] = {
        "classification": "PRE_EVALUATION_READER_CONTRACT_CORRECTION",
        "incident": (
            "The predecessor reader rejected evaluator rows with outcome=null "
            "and time_limit_truncated=true at max_ticks."
        ),
        "canonical_timeout_tuple": {
            "outcome": None,
            "time_limit_truncated": True,
            "ticks_equals_max_ticks": True,
        },
        "ranking_classification": "non_win",
        "inference_already_consumed": False,
        "seed_ranges_changed": False,
        "promotion_gate_changed": False,
        "candidate_rule_changed": False,
        "evaluation_environment_changed": False,
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
                    "sha256": controller._sha256(args.output.resolve(strict=True)),
                },
                "screen_seed_range": [
                    plan["screen"]["base_seed"],
                    plan["screen"]["last_seed"],
                ],
                "gate_seed_range": [
                    plan["full55_gate"]["base_seed"],
                    plan["full55_gate"]["last_seed"],
                ],
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
