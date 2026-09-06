"""Freeze a separate polar/DAgger AlphaZuma 55 successor certification."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools.build_alphazuma_55_eval_contract import (
    _read_json,
    _sha256,
    _write_json_exclusive,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()
EXPECTED_IDS = [
    "polar-source-final",
    "polar-dagger-round-00",
    "polar-dagger-round-01",
    "polar-dagger-round-02",
]
RANKING = [
    "wins_desc",
    "level_coverage_desc",
    "total_score_desc",
    "median_winning_ticks_asc",
    "earlier_round_first",
]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _reference(path: Path) -> dict[str, str]:
    path = path.resolve(strict=True)
    return {"path": str(path), "sha256": _sha256(path)}


def build_master(
    *,
    source_master_path: Path,
    expected_source_sha256: str,
    engineering_plan_path: Path,
    expected_plan_sha256: str,
    campaign_id: str,
    output_root: Path,
    independent_audit_receipt: Path,
    minimum_cleared_levels: int,
    minimum_wins: int,
) -> dict[str, Any]:
    source_master_path = source_master_path.resolve(strict=True)
    engineering_plan_path = engineering_plan_path.resolve(strict=True)
    _require(
        _sha256(source_master_path) == expected_source_sha256,
        "source master hash differs",
    )
    _require(
        _sha256(engineering_plan_path) == expected_plan_sha256,
        "engineering plan hash differs",
    )
    source = _read_json(source_master_path)
    plan = _read_json(engineering_plan_path)
    _require(
        source.get("schema")
        == "zuma-rl.alphazuma-55-weekend-master-preregistration"
        and source.get("version") == 1,
        "unexpected source master",
    )
    levels = list(source.get("scope", {}).get("included_levels_in_adventure_order", []))
    _require(len(levels) == 55 and len(set(levels)) == 55, "source scope differs")
    _require(
        plan.get("schema")
        == "zuma-rl.alphazuma-55-polar-dagger-validation-plan"
        and plan.get("version") == 1
        and plan.get("status") == "FROZEN_DURING_TARGET_TRAINING",
        "unexpected engineering validation plan",
    )
    _require(
        plan.get("master_preregistration")
        == {"path": str(source_master_path), "sha256": expected_source_sha256},
        "engineering plan does not bind the source master",
    )
    _require(
        plan.get("decision_rule", {}).get("ranking") == RANKING,
        "engineering plan ranking differs",
    )
    _require(
        int(plan.get("matrix", {}).get("levels", -1)) == 55
        and int(plan.get("matrix", {}).get("models", -1)) == 4
        and int(plan.get("matrix", {}).get("expected_attempts", -1)) == 220
        and plan.get("authority_boundary", {}).get("formal_seed_consumption")
        is False,
        "engineering matrix or authority differs",
    )
    _require(campaign_id.strip() == campaign_id and campaign_id, "campaign id is invalid")
    minimum_cleared_levels = int(minimum_cleared_levels)
    minimum_wins = int(minimum_wins)
    _require(
        1 <= minimum_cleared_levels <= 55 and 1 <= minimum_wins <= 55,
        "promotion threshold is outside [1, 55]",
    )
    output_root = output_root.resolve()
    independent_audit_receipt = independent_audit_receipt.resolve()
    _require(not output_root.exists(), "successor output root already exists")
    _require(
        not independent_audit_receipt.exists(),
        "successor independent audit receipt already exists",
    )
    implementation_paths = {
        "master_builder": SCRIPT_PATH,
        "controller": PROJECT_ROOT
        / "tools"
        / "run_alphazuma_55_successor_certification.py",
        "watcher": PROJECT_ROOT
        / "tools"
        / "watch_alphazuma_55_successor_certification.py",
        "independent_auditor": PROJECT_ROOT
        / "tools"
        / "audit_alphazuma_55_successor_result.py",
        "evaluation_builder": PROJECT_ROOT
        / "tools"
        / "build_alphazuma_55_successor_eval_contract.py",
        "evaluator": PROJECT_ROOT / "tools" / "evaluate_zero_shot_multilevel_v2.py",
        "evaluation_helper": PROJECT_ROOT / "tools" / "run_overnight_v11_postprocess.py",
        "matrix_auditor": PROJECT_ROOT
        / "tools"
        / "audit_alphazuma_v11_frontier_result.py",
        "engineering_auditor": PROJECT_ROOT
        / "tools"
        / "audit_alphazuma_55_training_validation.py",
        "summarizer": PROJECT_ROOT / "tools" / "summarize_multimodel_evaluation.py",
    }
    implementation = {
        name: _reference(path) for name, path in implementation_paths.items()
    }
    created = datetime.now(timezone.utc).isoformat()
    return {
        "schema": "zuma-rl.alphazuma-55-weekend-master-preregistration",
        "version": 1,
        "status": "FROZEN_BEFORE_SUCCESSOR_FINAL_BLIND",
        "campaign_id": campaign_id,
        "created_utc": created,
        "deadline_utc": source["deadline_utc"],
        "deadline_local": source.get("deadline_local"),
        "objective": {
            "one_frozen_policy_file": True,
            "runtime_model_switching_forbidden": True,
            "per_level_weight_routing_forbidden": True,
            "simulator_state_policy": True,
            "visual_frontend_in_scope": False,
            "original_client_world_record_claim": False,
            "target": (
                "Certify one frozen polar/DAgger policy on all 55 included "
                "levels and four continuous no-retry campaigns."
            ),
        },
        "scope": source["scope"],
        "original_root": plan["original_root"],
        "seed_registry": {
            "final_blind": {
                "first": 2_100_000_000,
                "last": 2_100_000_439,
                "matrix": "55 levels x 8 attempts for one frozen successor policy",
                "embargo_until_promotion_decision_is_frozen": True,
            },
            "continuous_campaign_challenge": {
                "first": 2_200_000_000,
                "last": 2_200_000_219,
                "matrix": "4 frozen campaigns x 55 levels, one attempt per level",
                "embargo_until_successor_policy_hash_is_frozen": True,
            },
            "overlap_with_source_campaign_formal_seeds": False,
        },
        "successor": {
            "source_campaign_master": _reference(source_master_path),
            "engineering_validation": {
                "plan": _reference(engineering_plan_path),
                "audit_receipt": str(Path(plan["outputs"]["audit_receipt"]).resolve()),
                "eligible_model_ids": EXPECTED_IDS,
                "ranking": RANKING,
                "one_paired_attempt_per_level": True,
                "current_campaign_candidate_authority": False,
            },
            "promotion_gate": {
                "minimum_cleared_levels": minimum_cleared_levels,
                "minimum_wins": minimum_wins,
                "evaluated_before_successor_seed_consumption": True,
                "failure_action": (
                    "write COMPLETE_NO_PROMOTION and leave both successor formal "
                    "seed ranges unconsumed"
                ),
                "rationale": (
                    "The one-shot engineering signal must exceed the 50 percent "
                    "final-blind total-win requirement before spending 660 formal "
                    "attempts; the eight-attempt matrix remains the authority for "
                    "55-level coverage."
                ),
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
            "authority_boundary": {
                "separate_from_source_campaign": True,
                "does_not_modify_source_candidate_inventory": True,
                "does_not_consume_source_selection_final_or_continuous_seeds": True,
                "negative_results_retained": True,
            },
        },
        "execution": {
            "shard_count": 2,
            "devices": ["cuda:0", "cuda:0"],
            "parallel_envs_per_shard": 12,
            "final_blind_attempts_per_level": 8,
            "continuous_campaigns": 4,
            "resource_isolation": (
                "Both successor shards use the 32 GiB RTX 5090 so the nearly "
                "full RTX 5080 can continue its frozen training routes."
            ),
        },
        "power_policy": {
            "rtx_5090_limit_watts": 550,
            "rtx_5080_limit_watts": 250,
            "does_not_authorize_early_power_restore": True,
        },
        "implementation": implementation,
        "outputs": {
            "root": str(output_root),
            "controller": str(output_root / "controller"),
            "watcher": str(output_root / "watcher"),
            "final_blind": str(output_root / "final-blind"),
            "continuous": str(output_root / "continuous"),
            "independent_audit_receipt": str(independent_audit_receipt),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-master", required=True, type=Path)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--engineering-plan", required=True, type=Path)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--independent-audit-receipt", required=True, type=Path)
    parser.add_argument("--minimum-cleared-levels", type=int, default=35)
    parser.add_argument("--minimum-wins", type=int, default=35)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite successor master: {output}")
    master = build_master(
        source_master_path=args.source_master.expanduser(),
        expected_source_sha256=str(args.expected_source_sha256),
        engineering_plan_path=args.engineering_plan.expanduser(),
        expected_plan_sha256=str(args.expected_plan_sha256),
        campaign_id=str(args.campaign_id),
        output_root=args.output_root.expanduser(),
        independent_audit_receipt=args.independent_audit_receipt.expanduser(),
        minimum_cleared_levels=args.minimum_cleared_levels,
        minimum_wins=args.minimum_wins,
    )
    _write_json_exclusive(output, master)
    print(
        json.dumps(
            {
                "status": master["status"],
                "output": {"path": str(output), "sha256": _sha256(output)},
                "promotion_gate": master["successor"]["promotion_gate"],
                "final_blind_seed_range": [2_100_000_000, 2_100_000_439],
                "continuous_seed_range": [2_200_000_000, 2_200_000_219],
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
