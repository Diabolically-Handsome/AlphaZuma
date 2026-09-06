"""Freeze a configured AlphaZuma 55 successor certification master.

The no-inference intent is the authority for candidates, promotion threshold,
formal seeds, topology, and output roots.  This builder only materializes and
hash-binds those already frozen choices; it never evaluates a policy.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Iterable

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
LEVEL_COUNT = 55
PLAN_BINDING_KEYS = {
    "zuma-rl.alphazuma-55-polar-wide-validation-plan": "polar_wide_validation_plan",
    "zuma-rl.alphazuma-55-polar-intent-wide-validation-plan": (
        "polar_intent_wide_validation_plan"
    ),
}
SUPPORTED_RANKING = [
    "wins_desc",
    "level_coverage_desc",
    "total_score_desc",
    "median_winning_ticks_asc",
    "smaller_model_first_as_tie_break",
]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _reference(path: Path) -> dict[str, str]:
    path = path.resolve(strict=True)
    return {"path": str(path), "sha256": _sha256(path)}


def _ranges(value: Any) -> Iterable[tuple[int, int]]:
    if isinstance(value, dict):
        if "first" in value and "last" in value:
            try:
                yield int(value["first"]), int(value["last"])
            except (TypeError, ValueError):
                pass
        if "base_seed" in value and "last_seed" in value:
            try:
                yield int(value["base_seed"]), int(value["last_seed"])
            except (TypeError, ValueError):
                pass
        for nested in value.values():
            yield from _ranges(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _ranges(nested)


def _overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return max(left[0], right[0]) <= min(left[1], right[1])


def build_master(
    *,
    successor_intent_path: Path,
    expected_intent_sha256: str,
) -> dict[str, Any]:
    successor_intent_path = successor_intent_path.resolve(strict=True)
    _require(
        _sha256(successor_intent_path) == expected_intent_sha256,
        "successor intent hash differs",
    )
    intent = _read_json(successor_intent_path)
    _require(
        str(intent.get("schema", "")).endswith("-successor-intent")
        and intent.get("version") == 1
        and intent.get("status") == "FROZEN_BEFORE_ENGINEERING_RESULT",
        "unexpected successor intent",
    )
    boundary = intent.get("implementation_boundary", {})
    _require(
        boundary.get("this_intent_does_not_authorize_policy_inference") is True
        and boundary.get(
            "formal_master_controller_and_auditor_must_be_hash_frozen_before_inference"
        )
        is True,
        "successor intent authority boundary differs",
    )
    authority = intent.get("authority_boundary", {})
    _require(
        authority.get("formal_seed_consumption") is False
        and authority.get("current_campaign_candidate_authority") is False,
        "successor intent already claims formal authority",
    )

    source_reference = intent.get("source_campaign_master")
    _require(isinstance(source_reference, dict), "source master reference is absent")
    source_path = Path(str(source_reference.get("path", ""))).resolve(strict=True)
    _require(
        source_reference.get("sha256") == _sha256(source_path),
        "source master hash differs",
    )
    source = _read_json(source_path)
    _require(
        source.get("schema")
        == "zuma-rl.alphazuma-55-weekend-master-preregistration"
        and source.get("version") == 1,
        "unexpected source campaign master",
    )
    levels = list(source.get("scope", {}).get("included_levels_in_adventure_order", []))
    _require(
        len(levels) == LEVEL_COUNT and len(set(levels)) == LEVEL_COUNT,
        "source campaign scope is not exactly 55 unique levels",
    )

    engineering = intent.get("engineering_validation")
    _require(isinstance(engineering, dict), "engineering validation intent is absent")
    plan_reference = engineering.get("plan")
    _require(isinstance(plan_reference, dict), "engineering plan reference is absent")
    plan_path = Path(str(plan_reference.get("path", ""))).resolve(strict=True)
    _require(
        plan_reference.get("sha256") == _sha256(plan_path),
        "engineering plan hash differs",
    )
    plan = _read_json(plan_path)
    plan_schema = str(plan.get("schema", ""))
    _require(
        plan_schema in PLAN_BINDING_KEYS
        and plan.get("version") == 1
        and plan.get("status") == "FROZEN_DURING_TARGET_TRAINING",
        "unsupported engineering validation plan",
    )
    _require(
        plan.get("master_preregistration") == _reference(source_path),
        "engineering plan does not bind the source master",
    )
    audit_receipt = Path(str(engineering.get("audit_receipt_expected", ""))).resolve()
    _require(
        audit_receipt == Path(str(plan["outputs"]["audit_receipt"])).resolve(),
        "engineering audit receipt differs from its plan",
    )
    candidate_order = [str(value) for value in engineering.get("candidate_order", [])]
    ranking = list(engineering.get("ranking", []))
    _require(
        len(candidate_order) == 4
        and len(set(candidate_order)) == 4
        and ranking == SUPPORTED_RANKING,
        "engineering candidates or ranking differ",
    )
    _require(
        plan.get("decision_rule", {}).get("ranking") == ranking
        and int(plan.get("matrix", {}).get("levels", -1)) == LEVEL_COUNT
        and int(plan.get("matrix", {}).get("models", -1)) == len(candidate_order)
        and int(plan.get("matrix", {}).get("expected_attempts", -1))
        == LEVEL_COUNT * len(candidate_order)
        and plan.get("authority_boundary", {}).get("formal_seed_consumption")
        is False,
        "engineering plan matrix, ranking, or authority differs",
    )

    gate = intent.get("promotion_gate")
    _require(isinstance(gate, dict), "promotion gate is absent")
    promotable_ids = [str(value) for value in gate.get("selected_model_must_be_one_of", [])]
    minimum_levels = int(gate.get("minimum_cleared_levels", -1))
    minimum_wins = int(gate.get("minimum_wins", -1))
    _require(
        promotable_ids
        and len(set(promotable_ids)) == len(promotable_ids)
        and set(promotable_ids).issubset(set(candidate_order))
        and 1 <= minimum_levels <= LEVEL_COUNT
        and 1 <= minimum_wins <= LEVEL_COUNT
        and gate.get("evaluated_before_formal_contract_or_seed_consumption") is True,
        "promotion gate is invalid",
    )

    formal = intent.get("formal_campaign_if_promoted")
    _require(isinstance(formal, dict), "formal campaign intent is absent")
    _require(
        formal.get("single_selected_policy_only") is True
        and formal.get("all_seeds_frozen_before_policy_inference") is True
        and formal.get("scope_levels") == levels,
        "formal campaign scope or single-policy boundary differs",
    )
    final = formal.get("final_blind", {})
    continuous = formal.get("continuous_campaign_challenge", {})
    final_range = (int(final.get("base_seed", -1)), int(final.get("last_seed", -1)))
    continuous_range = (
        int(continuous.get("base_seed", -1)),
        int(continuous.get("last_seed", -1)),
    )
    _require(
        int(final.get("attempts_per_level", -1)) == 8
        and int(final.get("expected_attempts", -1)) == 440
        and final_range[1] == final_range[0] + 439
        and int(continuous.get("campaigns", -1)) == 4
        and int(continuous.get("levels_per_campaign", -1)) == LEVEL_COUNT
        and int(continuous.get("expected_attempts", -1)) == 220
        and continuous_range[1] == continuous_range[0] + 219
        and not _overlap(final_range, continuous_range),
        "formal seed matrices differ",
    )
    _require(
        final.get("gate")
        == {
            "cleared_levels": 55,
            "minimum_win_per_level": 1,
            "minimum_total_wins": 220,
        }
        and continuous.get("gate")
        == {"minimum_complete_55_level_campaigns": 1},
        "formal success gates differ",
    )

    sibling_reference = intent.get("sibling_reserved_seed_intent")
    sibling_ranges: list[tuple[int, int]] = []
    if sibling_reference is not None:
        _require(isinstance(sibling_reference, dict), "sibling intent reference is invalid")
        sibling_path = Path(str(sibling_reference.get("path", ""))).resolve(strict=True)
        _require(
            sibling_reference.get("sha256") == _sha256(sibling_path),
            "sibling seed intent hash differs",
        )
        sibling_ranges = list(_ranges(_read_json(sibling_path).get("formal_campaign_if_promoted")))
    protected_ranges = list(_ranges(source.get("seed_registry", {}))) + sibling_ranges
    for reserved in (final_range, continuous_range):
        _require(
            all(not _overlap(reserved, protected) for protected in protected_ranges),
            "configured successor formal seeds overlap a protected campaign",
        )

    execution = formal.get("execution", {})
    devices = list(execution.get("devices", []))
    shard_count = int(execution.get("shard_count", -1))
    parallel_envs = int(execution.get("parallel_envs_per_shard", -1))
    _require(
        shard_count == 2
        and len(devices) == shard_count
        and all(device in {"cuda:0", "cuda:1"} for device in devices)
        and 1 <= parallel_envs <= 64,
        "formal execution topology is invalid",
    )

    outputs = intent.get("outputs")
    _require(isinstance(outputs, dict), "successor outputs are absent")
    output_root = Path(str(outputs.get("root", ""))).resolve()
    independent_audit = Path(
        str(outputs.get("independent_audit_receipt", ""))
    ).resolve()
    _require(not output_root.exists(), "configured successor output root already exists")
    _require(
        not independent_audit.exists(),
        "configured successor independent audit receipt already exists",
    )

    implementation_paths = {
        "master_builder": SCRIPT_PATH,
        "controller": PROJECT_ROOT
        / "tools"
        / "run_alphazuma_55_configured_successor_certification.py",
        "watcher": PROJECT_ROOT
        / "tools"
        / "watch_alphazuma_55_configured_successor_certification.py",
        "independent_auditor": PROJECT_ROOT
        / "tools"
        / "audit_alphazuma_55_configured_successor_result.py",
        "evaluation_builder": PROJECT_ROOT
        / "tools"
        / "build_alphazuma_55_configured_successor_eval_contract.py",
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
        "schema": "zuma-rl.alphazuma-55-configured-successor-master-preregistration",
        "version": 1,
        "status": "FROZEN_BEFORE_SUCCESSOR_FINAL_BLIND",
        "campaign_id": str(intent["campaign_id"]),
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
                "Certify one frozen configured policy on all 55 included levels "
                "and four continuous no-retry campaigns."
            ),
        },
        "scope": source["scope"],
        "original_root": str(Path(str(plan["original_root"])).resolve(strict=True)),
        "seed_registry": {
            "final_blind": {
                "first": final_range[0],
                "last": final_range[1],
                "matrix": "55 levels x 8 attempts for one frozen successor policy",
                "embargo_until_promotion_decision_is_frozen": True,
            },
            "continuous_campaign_challenge": {
                "first": continuous_range[0],
                "last": continuous_range[1],
                "matrix": "4 frozen campaigns x 55 levels, one attempt per level",
                "embargo_until_successor_policy_hash_is_frozen": True,
            },
            "overlap_with_source_or_sibling_formal_seeds": False,
        },
        "successor": {
            "intent": _reference(successor_intent_path),
            "source_campaign_master": _reference(source_path),
            "engineering_validation": {
                "plan": _reference(plan_path),
                "plan_schema": plan_schema,
                "manifest_plan_binding_key": PLAN_BINDING_KEYS[plan_schema],
                "audit_receipt": str(audit_receipt),
                "eligible_model_ids": candidate_order,
                "ranking": ranking,
                "one_paired_attempt_per_level": True,
                "current_campaign_candidate_authority": False,
            },
            "promotion_gate": {
                "selected_model_must_be_one_of": promotable_ids,
                "minimum_cleared_levels": minimum_levels,
                "minimum_wins": minimum_wins,
                "evaluated_before_successor_seed_consumption": True,
                "failure_action": (
                    "write COMPLETE_NO_PROMOTION and leave both configured "
                    "successor formal seed ranges unconsumed"
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
            "shard_count": shard_count,
            "devices": devices,
            "parallel_envs_per_shard": parallel_envs,
            "final_blind_attempts_per_level": 8,
            "continuous_campaigns": 4,
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
            "independent_audit_receipt": str(independent_audit),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--successor-intent", required=True, type=Path)
    parser.add_argument("--expected-intent-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite configured successor master: {output}"
        )
    master = build_master(
        successor_intent_path=args.successor_intent.expanduser(),
        expected_intent_sha256=str(args.expected_intent_sha256),
    )
    _write_json_exclusive(output, master)
    print(
        json.dumps(
            {
                "status": master["status"],
                "output": {"path": str(output), "sha256": _sha256(output)},
                "promotion_gate": master["successor"]["promotion_gate"],
                "final_blind_seed_range": [
                    master["seed_registry"]["final_blind"]["first"],
                    master["seed_registry"]["final_blind"]["last"],
                ],
                "continuous_seed_range": [
                    master["seed_registry"]["continuous_campaign_challenge"]["first"],
                    master["seed_registry"]["continuous_campaign_challenge"]["last"],
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
