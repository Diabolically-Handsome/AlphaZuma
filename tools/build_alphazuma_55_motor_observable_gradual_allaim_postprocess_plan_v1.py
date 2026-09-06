"""Freeze postprocessing for the all-actions motor DAgger contingency."""

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

from tools import (
    run_alphazuma_55_motor_observable_gradual_allaim_postprocess_v1 as controller,
)


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
CAMPAIGN_ID = (
    "alphazuma-55-motor-observable-gradual-allaim-postprocess-s99081634-v1"
)
TEMPLATE_PLAN = (
    PROJECT_ROOT
    / "diagnostics/alphazuma-55-motor-observable-gradual-postprocess-"
    "s99081630-plan-v1.json"
)
TEMPLATE_PLAN_SHA256 = (
    "sha256:e8e8f1b7bf4c9a73e353d6c278523e005790565753d42ad293ce1c47be2cbe67"
)
TRAINING_ROUTE = (
    PROJECT_ROOT
    / "diagnostics/alphazuma-55-motor-observable-gradual-allaim-"
    "s99081633-preregistration-v1.json"
)
TRAINING_ROUTE_SHA256 = (
    "sha256:502c0ab5566a66fa33ff33df80b10b101c5ea1b75dc4e21ff07f5996965730e1"
)
TRAINING_ROUTE_AUDIT = (
    PROJECT_ROOT
    / "diagnostics/alphazuma-55-motor-observable-gradual-allaim-"
    "s99081633-independent-prereg-audit-v1.json"
)
TRAINING_ROUTE_AUDIT_SHA256 = (
    "sha256:61be913b30601f2e529cc8112dd18aab8b434d82a7364a794e1ff756c5d99b6d"
)
TRAINING_LAUNCH_RECEIPT = (
    PROJECT_ROOT
    / "diagnostics/alphazuma-55-motor-observable-gradual-allaim-"
    "s99081633-launch-receipt-v1.json"
)
OUTPUT_ROOT = Path(f"/mnt/d/ZumaTraining/{CAMPAIGN_ID}")
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "diagnostics/alphazuma-55-motor-observable-gradual-allaim-postprocess-"
    "s99081634-plan-v1.json"
)


def _reference(path: Path, expected_sha256: str | None = None) -> dict[str, str]:
    resolved = path.resolve(strict=True)
    actual = controller.base._sha256(resolved)
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError(f"frozen bytes differ: {resolved}")
    return {"path": str(resolved), "sha256": actual}


def _assert_fresh_seed_bases(output: Path) -> None:
    needles = {
        str(controller.EXPECTED_SCREEN_SEED),
        str(controller.EXPECTED_GATE_SEED),
    }
    for path in (PROJECT_ROOT / "diagnostics").glob("*.json"):
        if path.resolve() == output.resolve():
            continue
        text = path.read_text(encoding="utf-8-sig")
        collided = sorted(value for value in needles if value in text)
        if collided:
            raise ValueError(f"training-validation seed base already frozen: {path}")


def build(*, output: Path) -> dict[str, Any]:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"postprocess plan already exists: {output}")
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"postprocess output already exists: {OUTPUT_ROOT}")
    template_ref = _reference(TEMPLATE_PLAN, TEMPLATE_PLAN_SHA256)
    route_ref = _reference(TRAINING_ROUTE, TRAINING_ROUTE_SHA256)
    audit_ref = _reference(TRAINING_ROUTE_AUDIT, TRAINING_ROUTE_AUDIT_SHA256)
    launch_ref = _reference(TRAINING_LAUNCH_RECEIPT)
    _assert_fresh_seed_bases(output)
    route = controller.base._read(TRAINING_ROUTE)
    run_dir = Path(str(route["run"]["run_dir"]))

    plan = copy.deepcopy(controller.base._read(TEMPLATE_PLAN))
    plan.update(
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "campaign_id": CAMPAIGN_ID,
            "objective": (
                "Freeze the V3 source baseline plus all sixteen all-actions "
                "aim-supervised DAgger checkpoints before checkpoint inference, "
                "screen on unused training-validation seeds, and retain the "
                "unchanged 35-win and 35-level engineering promotion gate."
            ),
            "training_route": route_ref,
            "training_route_independent_prereg_audit": audit_ref,
            "training_route_launch_receipt": launch_ref,
            "expected_training_completion": str(run_dir / "completion.json"),
            "expected_training_failure": str(run_dir / "failure.json"),
            "deadline_utc": "2026-08-17T13:00:00+00:00",
            "template_lineage": template_ref,
            "candidate_inventory_rule": {
                "source_baseline_included": True,
                "source_baseline_promotable": False,
                "round_indices": [0, 1, 2, 3],
                "checkpoint_epochs": [3, 6, 9, 12],
                "final_model_duplicate_excluded": True,
                "expected_candidates": controller.base.EXPECTED_SCREEN_CANDIDATES,
                "frozen_before_checkpoint_inference": True,
            },
        }
    )
    plan["hypotheses"] = {
        "primary": (
            "supervising queued aim targets on every actor tick reduces "
            "closed-loop motor drift"
        ),
        "negative_result_is_valid": True,
    }
    plan["screen"].update(
        {
            "base_seed": controller.EXPECTED_SCREEN_SEED,
            "last_seed": controller.EXPECTED_SCREEN_SEED
            + len(plan["screen_levels"])
            - 1,
            "expected_candidates": controller.base.EXPECTED_SCREEN_CANDIDATES,
            "expected_attempts": controller.base.EXPECTED_SCREEN_CANDIDATES
            * len(plan["screen_levels"]),
            "select_top": controller.base.MAXIMUM_GATE_CANDIDATES,
        }
    )
    plan["full55_gate"].update(
        {
            "base_seed": controller.EXPECTED_GATE_SEED,
            "last_seed": controller.EXPECTED_GATE_SEED
            + len(plan["full55_levels"])
            - 1,
            "maximum_candidates": controller.base.MAXIMUM_GATE_CANDIDATES,
            "maximum_expected_attempts": controller.base.MAXIMUM_GATE_CANDIDATES
            * len(plan["full55_levels"]),
        }
    )
    plan["fresh_seed_disclosure"] = {
        "classification": "unused_training_validation_ranges",
        "screen": [plan["screen"]["base_seed"], plan["screen"]["last_seed"]],
        "full55_gate": [
            plan["full55_gate"]["base_seed"],
            plan["full55_gate"]["last_seed"],
        ],
        "frozen_before_training_completion": True,
        "formal_seed_consumption": "NONE",
    }
    plan["implementation"].update(
        {
            "builder": _reference(SCRIPT_PATH),
            "controller": _reference(
                PROJECT_ROOT
                / "tools/run_alphazuma_55_motor_observable_gradual_"
                "allaim_postprocess_v1.py"
            ),
            "auditor": _reference(
                PROJECT_ROOT
                / "tools/audit_alphazuma_55_motor_observable_gradual_"
                "allaim_postprocess_v1.py"
            ),
            "gradual_training_runtime": route["trainer"],
        }
    )
    plan["outputs"] = {
        "output_root": str(OUTPUT_ROOT),
        "independent_audit_receipt": str(
            PROJECT_ROOT
            / "diagnostics/alphazuma-55-motor-observable-gradual-allaim-"
            "postprocess-s99081634-independent-audit-v1.json"
        ),
    }
    controller.base._write_new(output, plan)
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
                    "sha256": controller.base._sha256(args.output.resolve()),
                },
                "screen_attempts": plan["screen"]["expected_attempts"],
                "maximum_full55_attempts": plan["full55_gate"][
                    "maximum_expected_attempts"
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
