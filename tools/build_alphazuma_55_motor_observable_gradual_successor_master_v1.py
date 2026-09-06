"""Freeze the conditional formal master for the gradual motor route."""

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

from tools import run_alphazuma_55_motor_observable_gradual_successor_v1 as controller
from zuma_rl.alphazuma_55 import (
    DUAL_POSITION_LEVELS,
    EXCLUDED_MOVING_FROG_LEVELS,
    INCLUDED_LEVELS,
)


SCRIPT_PATH = Path(__file__).resolve()
CAMPAIGN_ID = "alphazuma-55-motor-observable-gradual-successor-s99081631-v1"
EXPECTED_ENGINEERING_PLAN_SHA256 = (
    "sha256:e8e8f1b7bf4c9a73e353d6c278523e005790565753d42ad293ce1c47be2cbe67"
)
EXPECTED_REGISTRY_V5_SHA256 = (
    "sha256:2f6996cd46778455e1069936da7a72b041882265a2c533b0db7faeddc63c5680"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _reference(path: Path) -> dict[str, str]:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "sha256": controller._sha256(resolved)}


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
        raise FileExistsError(f"gradual successor master already exists: {path}")
    os.replace(temporary, path)


def build(*, project_root: Path, output: Path) -> dict[str, Any]:
    root = project_root.resolve(strict=True)
    engineering_plan = (
        root
        / "diagnostics/alphazuma-55-motor-observable-gradual-postprocess-"
        "s99081630-plan-v1.json"
    )
    if controller._sha256(engineering_plan) != EXPECTED_ENGINEERING_PLAN_SHA256:
        raise ValueError("gradual engineering plan bytes changed")
    plan = controller.engineering.validate_plan(
        engineering_plan, EXPECTED_ENGINEERING_PLAN_SHA256
    )
    template = (
        root
        / "diagnostics/alphazuma-55-polar-intent-wide-validation-"
        "s99081544-preregistration-v1.json"
    )
    template_value = controller._read(template)
    if [str(row["id"]) for row in template_value["levels"]] != list(
        INCLUDED_LEVELS
    ):
        raise ValueError("environment template level order changed")
    registry_v5 = (
        root
        / "diagnostics/alphazuma-55-weekend-formal-seed-registry-"
        "s99081619-v5.json"
    )
    if controller._sha256(registry_v5) != EXPECTED_REGISTRY_V5_SHA256:
        raise ValueError("formal seed registry V5 bytes changed")
    active_registry = (
        root
        / "diagnostics/alphazuma-55-weekend-formal-seed-registry-"
        "s99081632-v6.json"
    )
    output_root = Path(f"/mnt/d/ZumaTraining/{CAMPAIGN_ID}")
    master = {
        "schema": controller.MASTER_SCHEMA,
        "version": 1,
        "status": "FROZEN_BEFORE_ENGINEERING_RESULT",
        "campaign_id": CAMPAIGN_ID,
        "created_utc": _utc_now(),
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
                "Certify one gradual-motor policy on all 55 included levels "
                "and four continuous no-retry campaigns."
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
            "plan": _reference(engineering_plan),
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
            "predecessor": _reference(registry_v5),
            "active_registry_path": str(active_registry),
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
                root
                / "tools/run_alphazuma_55_motor_observable_gradual_successor_v1.py"
            ),
            "independent_auditor": _reference(
                root
                / "tools/audit_alphazuma_55_motor_observable_gradual_successor_v1.py"
            ),
            "evaluator": _reference(
                root / "tools/evaluate_alphazuma_55_motor_observable.py"
            ),
            "evaluation_helper": _reference(
                root / "tools/run_overnight_v11_postprocess.py"
            ),
            "matrix_auditor": _reference(
                root / "tools/audit_alphazuma_v11_frontier_result.py"
            ),
            "summarizer": _reference(
                root / "tools/summarize_multimodel_evaluation.py"
            ),
            "engineering_controller": _reference(
                root
                / "tools/run_alphazuma_55_motor_observable_gradual_postprocess_v1.py"
            ),
            "engineering_auditor": _reference(
                root
                / "tools/audit_alphazuma_55_motor_observable_gradual_postprocess_v1.py"
            ),
        },
        "outputs": {
            "root": str(output_root),
            "controller": str(output_root / "controller"),
            "final_blind": str(output_root / "final-blind"),
            "continuous": str(output_root / "continuous"),
            "independent_audit_receipt": str(
                root
                / "diagnostics/alphazuma-55-motor-observable-gradual-successor-"
                "s99081631-independent-audit-v1.json"
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
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    value = build(project_root=args.project_root, output=args.output)
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
