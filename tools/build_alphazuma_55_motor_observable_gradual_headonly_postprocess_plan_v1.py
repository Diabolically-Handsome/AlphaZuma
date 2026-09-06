"""Freeze timeout-compatible postprocessing for head-only motor DAgger."""

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
    run_alphazuma_55_motor_observable_gradual_headonly_postprocess_v1
    as controller,
)


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
CAMPAIGN_ID = (
    "alphazuma-55-motor-observable-gradual-headonly-postprocess-s99081643-v1"
)
TEMPLATE_PLAN = (
    PROJECT_ROOT
    / "diagnostics/alphazuma-55-motor-observable-gradual-allaim-"
    "postprocess-s99081634-plan-v2.json"
)
TEMPLATE_PLAN_SHA256 = (
    "sha256:88d7c794cc4285db7e769e01ea14ca4f742293cee73050ad1143be1777328c01"
)
TRAINING_ROUTE = (
    PROJECT_ROOT
    / "diagnostics/alphazuma-55-motor-observable-gradual-headonly-"
    "s99081642-preregistration-v1.json"
)
TRAINING_ROUTE_SHA256 = (
    "sha256:18b12280221b59588a65349812303ab58c115e859fdfb8e2eb674d222cd2ea72"
)
TRAINING_ROUTE_AUDIT = (
    PROJECT_ROOT
    / "diagnostics/alphazuma-55-motor-observable-gradual-headonly-"
    "s99081642-independent-prereg-audit-v1.json"
)
TRAINING_ROUTE_AUDIT_SHA256 = (
    "sha256:9efd8a02d81df0cce8cd9f339d3f16faf1012b8ccca358194bdeafef5bd5be09"
)
TRAINING_LAUNCH_RECEIPT = (
    PROJECT_ROOT
    / "diagnostics/alphazuma-55-motor-observable-gradual-headonly-"
    "s99081642-launch-receipt-v1.json"
)
TRAINING_LAUNCH_RECEIPT_SHA256 = (
    "sha256:9a8c5f6b822bad3835f8f997ea9760dfa393d0821a6fb6f819778cdccea632d9"
)
SEED_OVERLAP_INCIDENT = (
    PROJECT_ROOT
    / "diagnostics/alphazuma-55-training-seed-overlap-"
    "1546000000-incident-v1.json"
)
SEED_OVERLAP_INCIDENT_SHA256 = (
    "sha256:e957e188226f1b2b6f402e68c1da40a18d2c6c85c505973db885d7759bbc2bb5"
)
AUDITOR = (
    PROJECT_ROOT
    / "tools/audit_alphazuma_55_motor_observable_gradual_"
    "headonly_postprocess_v1.py"
)
OUTPUT_ROOT = Path(f"/mnt/d/ZumaTraining/{CAMPAIGN_ID}")
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "diagnostics/alphazuma-55-motor-observable-gradual-headonly-"
    "postprocess-s99081643-plan-v1.json"
)
AUDIT_OUTPUT = (
    PROJECT_ROOT
    / "diagnostics/alphazuma-55-motor-observable-gradual-headonly-"
    "postprocess-s99081643-independent-prereg-audit-v1.json"
)


def _reference(path: Path, expected_sha256: str | None = None) -> dict[str, str]:
    resolved = path.resolve(strict=True)
    actual = controller._sha256(resolved)
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
            raise ValueError(
                f"training-validation seed base already frozen: {path}"
            )


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
    _assert_fresh_seed_bases(output)
    template_ref = _reference(TEMPLATE_PLAN, TEMPLATE_PLAN_SHA256)
    route_ref = _reference(TRAINING_ROUTE, TRAINING_ROUTE_SHA256)
    audit_ref = _reference(
        TRAINING_ROUTE_AUDIT, TRAINING_ROUTE_AUDIT_SHA256
    )
    launch_ref = _reference(
        TRAINING_LAUNCH_RECEIPT, TRAINING_LAUNCH_RECEIPT_SHA256
    )
    incident_ref = _reference(
        SEED_OVERLAP_INCIDENT, SEED_OVERLAP_INCIDENT_SHA256
    )
    route = controller._read(TRAINING_ROUTE)
    run_dir = Path(str(route["run"]["run_dir"]))
    plan = copy.deepcopy(controller._read(TEMPLATE_PLAN))
    plan.update(
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "campaign_id": CAMPAIGN_ID,
            "objective": (
                "Screen the frozen action-head-only checkpoints and apply "
                "the unchanged 35-win/35-level engineering Gate with the "
                "canonical max-tick timeout reader."
            ),
            "hypotheses": {
                "primary": (
                    "freezing the source representation prevents full-policy "
                    "imitation updates from erasing closed-loop competence"
                ),
                "negative_result_is_valid": True,
            },
            "training_route": route_ref,
            "training_route_independent_prereg_audit": audit_ref,
            "training_route_launch_receipt": launch_ref,
            "expected_training_completion": str(run_dir / "completion.json"),
            "expected_training_failure": str(run_dir / "failure.json"),
            "deadline_utc": "2026-08-17T13:00:00+00:00",
            "template_lineage": template_ref,
        }
    )
    plan["candidate_inventory"] = {
        "baseline": "frozen V3 source model",
        "motor_checkpoints": (
            "all 4 rounds x all 4 per-round checkpoints in chronological "
            "order; final_model aliases are excluded"
        ),
        "expected_motor_checkpoints": 16,
        "expected_total_candidates": 17,
        "checkpoint_inclusion_is_metric_independent": True,
    }
    plan["candidate_inventory_rule"] = {
        "source_baseline_included": True,
        "source_baseline_promotable": False,
        "round_indices": [0, 1, 2, 3],
        "checkpoint_epochs": [3, 6, 9, 12],
        "final_model_duplicate_excluded": True,
        "expected_candidates": 17,
        "frozen_before_checkpoint_inference": True,
    }
    plan["screen"].update(
        {
            "base_seed": controller.EXPECTED_SCREEN_SEED,
            "last_seed": controller.EXPECTED_SCREEN_SEED
            + len(plan["screen_levels"])
            - 1,
            "expected_candidates": 17,
            "expected_attempts": 17 * len(plan["screen_levels"]),
            "select_top": controller.MAXIMUM_GATE_CANDIDATES,
        }
    )
    plan["full55_gate"].update(
        {
            "base_seed": controller.EXPECTED_GATE_SEED,
            "last_seed": controller.EXPECTED_GATE_SEED
            + len(plan["full55_levels"])
            - 1,
            "maximum_candidates": controller.MAXIMUM_GATE_CANDIDATES,
            "maximum_expected_attempts": (
                controller.MAXIMUM_GATE_CANDIDATES
                * len(plan["full55_levels"])
            ),
        }
    )
    plan["fresh_seed_disclosure"] = {
        "classification": "unused_training_validation_ranges",
        "screen": [
            plan["screen"]["base_seed"],
            plan["screen"]["last_seed"],
        ],
        "full55_gate": [
            plan["full55_gate"]["base_seed"],
            plan["full55_gate"]["last_seed"],
        ],
        "frozen_before_training_completion": True,
        "formal_seed_consumption": "NONE",
    }
    plan["training_seed_reuse_disclosure"] = {
        "incident": incident_ref,
        "scope": "engineering_training_only",
        "screen_and_gate_ranges_are_distinct": True,
        "formal_seed_overlap": False,
    }
    plan["implementation"].update(
        {
            "builder": _reference(SCRIPT_PATH),
            "controller": _reference(controller.SCRIPT_PATH),
            "auditor": _reference(AUDITOR),
            "gradual_training_runtime": route["trainer"],
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
            "The historical reader rejected evaluator rows with outcome=null "
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
                    "sha256": controller._sha256(
                        args.output.resolve(strict=True)
                    ),
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
