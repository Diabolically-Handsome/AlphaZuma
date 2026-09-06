"""Freeze the conditional formal master for the all-actions motor route."""

from __future__ import annotations

import argparse
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
    run_alphazuma_55_motor_observable_gradual_allaim_successor_v1
    as controller,
)
from zuma_rl.alphazuma_55 import (
    DUAL_POSITION_LEVELS,
    EXCLUDED_MOVING_FROG_LEVELS,
    INCLUDED_LEVELS,
)


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
CAMPAIGN_ID = (
    "alphazuma-55-motor-observable-gradual-allaim-successor-s99081636-v1"
)
ENGINEERING_PLAN = (
    PROJECT_ROOT
    / "diagnostics/alphazuma-55-motor-observable-gradual-allaim-"
    "postprocess-s99081634-plan-v1.json"
)
REGISTRY_V6 = (
    PROJECT_ROOT
    / "diagnostics/alphazuma-55-weekend-formal-seed-registry-"
    "s99081632-v6.json"
)
REGISTRY_V6_SHA256 = (
    "sha256:9c079b5de166623b112864547e4e6f040c29680593dedbb53bde5a5da04e7db3"
)
ACTIVE_REGISTRY = (
    PROJECT_ROOT
    / "diagnostics/alphazuma-55-weekend-formal-seed-registry-"
    "s99081637-v7.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "diagnostics/alphazuma-55-motor-observable-gradual-allaim-"
    "successor-s99081636-preregistration-v1.json"
)


def _reference(path: Path, expected_sha256: str | None = None) -> dict[str, str]:
    resolved = path.resolve(strict=True)
    actual = controller._sha256(resolved)
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError(f"frozen bytes differ: {resolved}")
    return {"path": str(resolved), "sha256": actual}


def _write_new(path: Path, value: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists():
        temporary.unlink()
        raise FileExistsError(f"all-action successor master exists: {path}")
    os.replace(temporary, path)


def build(*, expected_plan_sha256: str, output: Path) -> dict[str, Any]:
    if output.resolve().exists():
        raise FileExistsError(f"all-action successor master exists: {output}")
    plan_ref = _reference(ENGINEERING_PLAN, expected_plan_sha256)
    plan = controller.engineering.validate_plan(
        ENGINEERING_PLAN, expected_plan_sha256
    )
    if plan.get("campaign_id") != controller.EXPECTED_ENGINEERING_CAMPAIGN:
        raise ValueError("all-action engineering plan campaign differs")
    template = (
        PROJECT_ROOT
        / "diagnostics/alphazuma-55-polar-intent-wide-validation-"
        "s99081544-preregistration-v1.json"
    )
    template_value = controller._read(template)
    if [str(row["id"]) for row in template_value["levels"]] != list(
        INCLUDED_LEVELS
    ):
        raise ValueError("environment template level order changed")
    registry_ref = _reference(REGISTRY_V6, REGISTRY_V6_SHA256)
    output_root = Path(f"/mnt/d/ZumaTraining/{CAMPAIGN_ID}")
    master = {
        "schema": controller.MASTER_SCHEMA,
        "version": 1,
        "status": "FROZEN_BEFORE_ENGINEERING_RESULT",
        "campaign_id": CAMPAIGN_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "deadline_utc": "2026-08-17T15:00:00Z",
        "deadline_local": "2026-08-17 11:00 America/Toronto",
        "objective": {
            "one_frozen_policy_file": True,
            "runtime_model_switching_forbidden": True,
            "per_level_weight_routing_forbidden": True,
            "simulator_state_policy": True,
            "visual_frontend_in_scope": False,
            "original_client_world_record_claim": False,
            "target": (
                "Certify one all-actions motor policy on all 55 included "
                "levels and four continuous no-retry campaigns."
            ),
        },
        "scope": {
            "retail_ordinary_adventure_level_count": 60,
            "included_level_count": len(INCLUDED_LEVELS),
            "included_levels": list(INCLUDED_LEVELS),
            "excluded_moving_frog_levels": list(EXCLUDED_MOVING_FROG_LEVELS),
            "included_dual_position_levels": list(DUAL_POSITION_LEVELS),
            "bosses_excluded": True,
        },
        "original_root": str(plan["original_root"]),
        "environment_template": _reference(template),
        "engineering": {
            "plan": plan_ref,
            "independent_audit": str(
                Path(plan["outputs"]["independent_audit_receipt"]).resolve()
            ),
            "promotion_gate": {
                "minimum_wins": 35,
                "minimum_cleared_levels": 35,
                "must_be_promotable_motor_checkpoint": True,
                "evaluated_before_formal_seed_consumption": True,
                "failure_action": (
                    "write COMPLETE_NO_PROMOTION and leave both formal ranges "
                    "unconsumed"
                ),
            },
        },
        "seed_registry": {
            "final_blind": {
                "first": controller.FINAL_SEED_BASE,
                "last": (
                    controller.FINAL_SEED_BASE
                    + len(INCLUDED_LEVELS) * controller.FINAL_ATTEMPTS
                    - 1
                ),
                "matrix": "55 levels x 8 attempts for one frozen policy",
                "embargo_until_engineering_promotion_is_frozen": True,
            },
            "continuous_campaign_challenge": {
                "first": controller.CONTINUOUS_SEED_BASE,
                "last": (
                    controller.CONTINUOUS_SEED_BASE
                    + len(INCLUDED_LEVELS) * controller.CONTINUOUS_CAMPAIGNS
                    - 1
                ),
                "matrix": "4 campaigns x 55 levels for one frozen policy",
                "embargo_until_policy_hash_is_frozen": True,
            },
        },
        "formal_seed_registry": {
            "predecessor": registry_ref,
            "active_registry_path": str(ACTIVE_REGISTRY),
            "active_registry_must_bind_this_master_before_controller_launch": True,
        },
        "formal_gates": {
            "single_policy_capability_gate": {
                "levels_cleared": 55,
                "minimum_wins_per_level": 1,
                "total_wins": 220,
                "attempts": 440,
            },
            "continuous_full_game_gate": {
                "campaigns_cleared": 1,
                "campaigns_attempted": 4,
                "no_retry": True,
                "no_model_change": True,
            },
        },
        "execution": {
            "shard_count": 2,
            "devices": ["cuda:0", "cuda:1"],
            "parallel_envs_per_shard": 24,
            "final_blind_attempts_per_level": controller.FINAL_ATTEMPTS,
            "continuous_campaigns": controller.CONTINUOUS_CAMPAIGNS,
        },
        "power_policy": {
            "rtx_5090_limit_watts": 550,
            "rtx_5080_limit_watts": 250,
            "does_not_authorize_early_power_restore": True,
        },
        "implementation": {
            "master_builder": _reference(SCRIPT_PATH),
            "controller": _reference(
                PROJECT_ROOT
                / "tools/run_alphazuma_55_motor_observable_gradual_allaim_"
                "successor_v1.py"
            ),
            "independent_auditor": _reference(
                PROJECT_ROOT
                / "tools/audit_alphazuma_55_motor_observable_gradual_allaim_"
                "successor_v1.py"
            ),
            "evaluator": _reference(
                PROJECT_ROOT / "tools/evaluate_alphazuma_55_motor_observable.py"
            ),
            "evaluation_helper": _reference(
                PROJECT_ROOT / "tools/run_overnight_v11_postprocess.py"
            ),
            "matrix_auditor": _reference(
                PROJECT_ROOT / "tools/audit_alphazuma_v11_frontier_result.py"
            ),
            "summarizer": _reference(
                PROJECT_ROOT / "tools/summarize_multimodel_evaluation.py"
            ),
            "engineering_controller": _reference(
                PROJECT_ROOT
                / "tools/run_alphazuma_55_motor_observable_gradual_allaim_"
                "postprocess_v1.py"
            ),
            "engineering_auditor": _reference(
                PROJECT_ROOT
                / "tools/audit_alphazuma_55_motor_observable_gradual_allaim_"
                "postprocess_v1.py"
            ),
        },
        "outputs": {
            "root": str(output_root),
            "controller": str(output_root / "controller"),
            "final_blind": str(output_root / "final-blind"),
            "continuous": str(output_root / "continuous"),
            "independent_audit_receipt": str(
                PROJECT_ROOT
                / "diagnostics/alphazuma-55-motor-observable-gradual-allaim-"
                "successor-s99081636-independent-audit-v1.json"
            ),
        },
        "authority_boundary": {
            "engineering_result_unseen_at_freeze": True,
            "formal_seeds_embargoed_until_promotion_decision": True,
            "negative_results_retained": True,
            "power_restore_authority": False,
        },
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
        output=args.output.expanduser(),
    )
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
