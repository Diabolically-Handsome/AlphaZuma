"""Freeze no-inference formal intent for an autonomy-trained successor."""

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
from tools.materialize_alphazuma_55_polar_intent_autonomy_validation import (
    validate_plan_static,
)


SCRIPT_PATH = Path(__file__).resolve()
FINAL_BASE = 2_900_000_000
FINAL_LAST = FINAL_BASE + 55 * 8 - 1
CONTINUOUS_BASE = 3_000_000_000
CONTINUOUS_LAST = CONTINUOUS_BASE + 55 * 4 - 1
EXPECTED_CANDIDATES = (
    "polar-intent-autonomy-source-final",
    "polar-intent-autonomy-round-00",
    "polar-intent-autonomy-round-01",
    "polar-intent-autonomy-round-02",
    "polar-intent-autonomy-round-03",
)
PROMOTABLE_CANDIDATES = EXPECTED_CANDIDATES[1:]


def _artifact(path: Path) -> dict[str, str]:
    path = path.resolve(strict=True)
    return {"path": str(path), "sha256": _sha256(path)}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _ranges(value: Any) -> Iterable[tuple[int, int]]:
    if isinstance(value, dict):
        if "first" in value and "last" in value:
            yield int(value["first"]), int(value["last"])
        if "base_seed" in value and "last_seed" in value:
            yield int(value["base_seed"]), int(value["last_seed"])
        for nested in value.values():
            yield from _ranges(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _ranges(nested)


def _disjoint(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[1] < right[0] or right[1] < left[0]


def build(
    *,
    source_master_path: Path,
    validation_plan_path: Path,
    formal_registry_path: Path,
    predecessor_intent_path: Path,
    output_root: Path,
    independent_audit_receipt: Path,
    campaign_id: str,
) -> dict[str, Any]:
    source_master_path = source_master_path.resolve(strict=True)
    validation_plan_path = validation_plan_path.resolve(strict=True)
    formal_registry_path = formal_registry_path.resolve(strict=True)
    predecessor_intent_path = predecessor_intent_path.resolve(strict=True)
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
    validation = validate_plan_static(
        validation_plan_path, _sha256(validation_plan_path)
    )
    _require(
        validation["master_preregistration"]["sha256"]
        == _sha256(source_master_path),
        "validation plan binds another source master",
    )
    matrix = validation["matrix"]
    _require(
        int(matrix["levels"]) == 55
        and int(matrix["models"]) == 5
        and int(matrix["expected_attempts"]) == 275
        and matrix["paired_models_share_identical_task_seeds"] is True,
        "autonomy engineering matrix differs",
    )
    _require(
        validation["authority_boundary"]["formal_seed_consumption"] is False
        and validation["authority_boundary"][
            "predecessor_successor_candidate_authority"
        ]
        is False,
        "engineering plan claims formal authority",
    )
    registry = _read_json(formal_registry_path)
    _require(
        registry.get("schema")
        == "zuma-rl.alphazuma-55-weekend-formal-seed-registry"
        and registry.get("status")
        == "FROZEN_BEFORE_ANY_SUCCESSOR_FORMAL_INFERENCE"
        and registry.get("global_checks", {}).get(
            "ranges_strictly_non_overlapping"
        )
        is True,
        "unexpected prior formal seed registry",
    )
    predecessor = _read_json(predecessor_intent_path)
    _require(
        predecessor.get("schema")
        == "zuma-rl.alphazuma-55-polar-intent-dagger-successor-intent"
        and predecessor.get("status") == "FROZEN_BEFORE_ENGINEERING_RESULT",
        "unexpected predecessor successor intent",
    )
    proposed = (
        (FINAL_BASE, FINAL_LAST),
        (CONTINUOUS_BASE, CONTINUOUS_LAST),
    )
    protected = list(_ranges(registry.get("campaigns", [])))
    protected.extend(_ranges(source.get("seed_registry", {})))
    protected.extend(
        _ranges(predecessor.get("formal_campaign_if_promoted", {}))
    )
    for reserved in proposed:
        _require(
            all(_disjoint(reserved, prior) for prior in protected),
            "proposed autonomy formal seeds overlap prior ranges",
        )
        _require(
            0 <= reserved[0] <= reserved[1] <= 0xFFFFFFFF,
            "proposed autonomy formal seeds exceed uint32",
        )
    _require(_disjoint(*proposed), "new formal ranges overlap each other")
    output_root = output_root.resolve()
    independent_audit_receipt = independent_audit_receipt.resolve()
    _require(not output_root.exists(), "successor output root already exists")
    _require(
        not independent_audit_receipt.exists(),
        "successor independent audit receipt already exists",
    )
    _require(bool(campaign_id.strip()), "campaign id is empty")
    ranking = list(validation["decision_rule"]["ranking"])
    return {
        "schema": (
            "zuma-rl.alphazuma-55-polar-intent-autonomy-successor-intent"
        ),
        "version": 1,
        "status": "FROZEN_BEFORE_ENGINEERING_RESULT",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_id": campaign_id,
        "objective": (
            "Authorize formal full55 and continuous evaluation only if an "
            "autonomy-trained raw-intent round clears the frozen engineering "
            "gate."
        ),
        "source_campaign_master": _artifact(source_master_path),
        "formal_seed_registry_snapshot": _artifact(formal_registry_path),
        "sibling_reserved_seed_intent": _artifact(predecessor_intent_path),
        "engineering_validation": {
            "plan": _artifact(validation_plan_path),
            "audit_receipt_expected": str(
                Path(validation["outputs"]["audit_receipt"]).resolve()
            ),
            "candidate_order": list(EXPECTED_CANDIDATES),
            "ranking": ranking,
            "paired_attempts": 275,
            "formal_authority": False,
        },
        "promotion_gate": {
            "selected_model_must_be_one_of": list(PROMOTABLE_CANDIDATES),
            "minimum_cleared_levels": 35,
            "minimum_wins": 35,
            "training_label_semantics": "raw_teacher_intent",
            "state_distribution_semantics": "student_only_dagger",
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
            "independent_audit_receipt": str(independent_audit_receipt),
        },
        "implementation_boundary": {
            "intent_builder": _artifact(SCRIPT_PATH),
            "formal_master_controller_and_auditor_must_be_hash_frozen_before_inference": True,
            "this_intent_does_not_authorize_policy_inference": True,
        },
        "authority_boundary": {
            "current_campaign_candidate_authority": False,
            "predecessor_successor_candidate_authority": False,
            "formal_seed_consumption": False,
            "power_restore_authority": False,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-master", required=True, type=Path)
    parser.add_argument("--validation-plan", required=True, type=Path)
    parser.add_argument("--formal-seed-registry", required=True, type=Path)
    parser.add_argument("--predecessor-intent", required=True, type=Path)
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
        validation_plan_path=args.validation_plan.expanduser(),
        formal_registry_path=args.formal_seed_registry.expanduser(),
        predecessor_intent_path=args.predecessor_intent.expanduser(),
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
