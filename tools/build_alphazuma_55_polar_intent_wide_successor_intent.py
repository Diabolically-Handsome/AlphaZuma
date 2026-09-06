"""Freeze the no-inference intent for a possible intent-wide successor."""

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


SCRIPT_PATH = Path(__file__).resolve()
FINAL_BASE = 2_500_000_000
FINAL_LAST = FINAL_BASE + 55 * 8 - 1
CONTINUOUS_BASE = 2_600_000_000
CONTINUOUS_LAST = CONTINUOUS_BASE + 55 * 4 - 1
EXPECTED_CANDIDATES = (
    "polar-source-final",
    "polar-intent-wide-epoch-10",
    "polar-intent-wide-epoch-20",
    "polar-intent-wide-epoch-30",
)
INTENT_CANDIDATES = EXPECTED_CANDIDATES[1:]


def _artifact(path: Path) -> dict[str, str]:
    path = path.resolve(strict=True)
    return {"path": str(path), "sha256": _sha256(path)}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _disjoint(first: tuple[int, int], second: tuple[int, int]) -> bool:
    return first[1] < second[0] or second[1] < first[0]


def _proposed_ranges() -> tuple[tuple[int, int], tuple[int, int]]:
    return ((FINAL_BASE, FINAL_LAST), (CONTINUOUS_BASE, CONTINUOUS_LAST))


def _assert_disjoint_from_source(source: dict[str, Any]) -> None:
    for name, registry in source.get("seed_registry", {}).items():
        if not isinstance(registry, dict):
            continue
        first = registry.get("first")
        last = registry.get("last")
        if first is None or last is None:
            continue
        source_range = (int(first), int(last))
        for new_range in _proposed_ranges():
            _require(
                _disjoint(new_range, source_range),
                f"proposed successor seeds overlap source registry {name}",
            )


def _assert_disjoint_from_sibling(sibling: dict[str, Any]) -> None:
    _require(
        sibling.get("schema")
        == "zuma-rl.alphazuma-55-polar-wide-successor-intent"
        and sibling.get("status") == "FROZEN_BEFORE_ENGINEERING_RESULT",
        "unexpected sibling wide successor intent",
    )
    formal = sibling["formal_campaign_if_promoted"]
    sibling_ranges = (
        (
            int(formal["final_blind"]["base_seed"]),
            int(formal["final_blind"]["last_seed"]),
        ),
        (
            int(formal["continuous_campaign_challenge"]["base_seed"]),
            int(formal["continuous_campaign_challenge"]["last_seed"]),
        ),
    )
    for new_range in _proposed_ranges():
        for sibling_range in sibling_ranges:
            _require(
                _disjoint(new_range, sibling_range),
                "proposed successor seeds overlap sibling wide intent",
            )


def build(
    *,
    source_master_path: Path,
    validation_plan_path: Path,
    sibling_wide_intent_path: Path,
    output_root: Path,
    independent_audit_receipt: Path,
    campaign_id: str,
) -> dict[str, Any]:
    source_master_path = source_master_path.resolve(strict=True)
    validation_plan_path = validation_plan_path.resolve(strict=True)
    sibling_wide_intent_path = sibling_wide_intent_path.resolve(strict=True)
    source = _read_json(source_master_path)
    _require(
        source.get("schema")
        == "zuma-rl.alphazuma-55-weekend-master-preregistration",
        "unexpected source master",
    )
    levels = tuple(source["scope"]["included_levels_in_adventure_order"])
    _require(
        len(levels) == 55 and len(set(levels)) == 55,
        "source scope is not 55 levels",
    )
    validation = _read_json(validation_plan_path)
    _require(
        validation.get("schema")
        == "zuma-rl.alphazuma-55-polar-intent-wide-validation-plan"
        and validation.get("status") == "FROZEN_DURING_TARGET_TRAINING",
        "unexpected intent-wide validation plan",
    )
    _require(
        validation["master_preregistration"]["sha256"]
        == _sha256(source_master_path),
        "intent-wide validation binds another source master",
    )
    matrix = validation["matrix"]
    _require(
        int(matrix["levels"]) == 55
        and int(matrix["models"]) == 4
        and int(matrix["expected_attempts"]) == 220
        and matrix["paired_models_share_identical_task_seeds"] is True,
        "intent-wide engineering matrix differs",
    )
    _require(
        validation["authority_boundary"]["formal_seed_consumption"] is False
        and validation["authority_boundary"][
            "s99081535_successor_candidate_authority"
        ]
        is False,
        "intent-wide engineering plan claims formal authority",
    )
    sibling = _read_json(sibling_wide_intent_path)
    _require(
        sibling["source_campaign_master"]["sha256"]
        == _sha256(source_master_path),
        "sibling wide intent binds another source master",
    )
    _assert_disjoint_from_source(source)
    _assert_disjoint_from_sibling(sibling)
    output_root = output_root.resolve()
    _require(not output_root.exists(), f"successor output root exists: {output_root}")
    _require(bool(campaign_id.strip()), "campaign id is empty")
    return {
        "schema": "zuma-rl.alphazuma-55-polar-intent-wide-successor-intent",
        "version": 1,
        "status": "FROZEN_BEFORE_ENGINEERING_RESULT",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_id": campaign_id,
        "objective": (
            "Authorize a separate formal full55 and continuous campaign only if "
            "a raw-intent wide checkpoint beats the source model and clears the "
            "predeclared engineering threshold."
        ),
        "source_campaign_master": _artifact(source_master_path),
        "sibling_reserved_seed_intent": _artifact(sibling_wide_intent_path),
        "engineering_validation": {
            "plan": _artifact(validation_plan_path),
            "audit_receipt_expected": str(
                Path(validation["outputs"]["audit_receipt"]).resolve()
            ),
            "candidate_order": list(EXPECTED_CANDIDATES),
            "ranking": list(validation["decision_rule"]["ranking"]),
            "paired_attempts": 220,
            "formal_authority": False,
        },
        "promotion_gate": {
            "selected_model_must_be_one_of": list(INTENT_CANDIDATES),
            "minimum_cleared_levels": 35,
            "minimum_wins": 35,
            "training_label_semantics": "raw_teacher_intent",
            "online_inference_semantics": "exact_action_masks",
            "evaluated_on_frozen_engineering_matrix": True,
            "evaluated_before_formal_contract_or_seed_consumption": True,
            "negative_result_action": "COMPLETE_NO_PROMOTION",
        },
        "formal_campaign_if_promoted": {
            "single_selected_policy_only": True,
            "scope_levels": list(levels),
            "final_blind": {
                "base_seed": FINAL_BASE,
                "last_seed": FINAL_LAST,
                "attempts_per_level": 8,
                "expected_attempts": 440,
                "gate": {
                    "cleared_levels": 55,
                    "minimum_win_per_level": 1,
                    "minimum_total_wins": 220,
                },
            },
            "continuous_campaign_challenge": {
                "base_seed": CONTINUOUS_BASE,
                "last_seed": CONTINUOUS_LAST,
                "campaigns": 4,
                "levels_per_campaign": 55,
                "expected_attempts": 220,
                "gate": {"minimum_complete_55_level_campaigns": 1},
            },
            "execution": {
                "shard_count": 2,
                "devices": ["cuda:0", "cuda:1"],
                "parallel_envs_per_shard": 12,
            },
            "all_seeds_frozen_before_policy_inference": True,
        },
        "outputs": {
            "root": str(output_root),
            "independent_audit_receipt": str(independent_audit_receipt.resolve()),
        },
        "implementation_boundary": {
            "intent_builder": _artifact(SCRIPT_PATH),
            "formal_master_controller_and_auditor_must_be_hash_frozen_before_inference": True,
            "this_intent_does_not_authorize_policy_inference": True,
        },
        "authority_boundary": {
            "current_campaign_candidate_authority": False,
            "s99081535_successor_candidate_authority": False,
            "formal_seed_consumption": False,
            "power_restore_authority": False,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-master", required=True, type=Path)
    parser.add_argument("--intent-validation-plan", required=True, type=Path)
    parser.add_argument("--sibling-wide-intent", required=True, type=Path)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--independent-audit-receipt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite successor intent: {output}")
    value = build(
        source_master_path=args.source_master.expanduser(),
        validation_plan_path=args.intent_validation_plan.expanduser(),
        sibling_wide_intent_path=args.sibling_wide_intent.expanduser(),
        output_root=args.output_root.expanduser(),
        independent_audit_receipt=args.independent_audit_receipt.expanduser(),
        campaign_id=str(args.campaign_id),
    )
    _write_json_exclusive(output, value)
    print(
        json.dumps(
            {
                "status": value["status"],
                "output": {"path": str(output), "sha256": _sha256(output)},
                "promotion_gate": value["promotion_gate"],
                "formal_seed_ranges": {
                    "final_blind": [FINAL_BASE, FINAL_LAST],
                    "continuous": [CONTINUOUS_BASE, CONTINUOUS_LAST],
                },
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
