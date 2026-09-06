"""Freeze configured certification for raw-intent DAgger engineering plans."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools import build_alphazuma_55_configured_successor_master as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()
PLAN_SCHEMA = "zuma-rl.alphazuma-55-polar-intent-dagger-validation-plan"
PLAN_BINDING_KEY = "polar_intent_dagger_validation_plan"
EXPECTED_RANKING = [
    "wins_desc",
    "level_coverage_desc",
    "total_score_desc",
    "median_winning_ticks_asc",
    "earlier_round_first",
]


def _artifact(path: Path) -> dict[str, str]:
    path = path.resolve(strict=True)
    return {"path": str(path), "sha256": base._sha256(path)}


def build_master(
    *, successor_intent_path: Path, expected_intent_sha256: str
) -> dict[str, Any]:
    successor_intent_path = successor_intent_path.resolve(strict=True)
    base._require(
        base._sha256(successor_intent_path) == expected_intent_sha256,
        "V2 successor intent hash differs",
    )
    intent = base._read_json(successor_intent_path)
    base._require(
        intent.get("schema")
        == "zuma-rl.alphazuma-55-polar-intent-dagger-successor-intent"
        and intent.get("status") == "FROZEN_BEFORE_ENGINEERING_RESULT",
        "unexpected raw-intent DAgger successor intent",
    )
    registry_reference = intent.get("formal_seed_registry_snapshot", {})
    registry_path = Path(str(registry_reference.get("path", ""))).resolve(
        strict=True
    )
    base._require(
        registry_reference.get("sha256") == base._sha256(registry_path),
        "formal seed registry snapshot changed",
    )
    registry = base._read_json(registry_path)
    base._require(
        registry.get("global_checks", {}).get(
            "ranges_strictly_non_overlapping"
        )
        is True,
        "prior formal seed registry is not clean",
    )
    plan_path = Path(intent["engineering_validation"]["plan"]["path"]).resolve(
        strict=True
    )
    plan = base._read_json(plan_path)
    base._require(
        plan.get("schema") == PLAN_SCHEMA
        and plan.get("status") == "FROZEN_BEFORE_TARGET_PREREGISTRATION"
        and plan.get("decision_rule", {}).get("ranking") == EXPECTED_RANKING,
        "unexpected raw-intent DAgger engineering plan",
    )
    original_script = base.SCRIPT_PATH
    original_keys = base.PLAN_BINDING_KEYS
    original_ranking = base.SUPPORTED_RANKING
    original_read = base._read_json

    def adapted_read(path: Path) -> dict[str, Any]:
        resolved = Path(path).resolve(strict=True)
        value = original_read(resolved)
        if resolved == plan_path:
            value = copy.deepcopy(value)
            value["status"] = "FROZEN_DURING_TARGET_TRAINING"
        return value

    try:
        base.SCRIPT_PATH = SCRIPT_PATH
        base.PLAN_BINDING_KEYS = {
            **original_keys,
            PLAN_SCHEMA: PLAN_BINDING_KEY,
        }
        base.SUPPORTED_RANKING = list(EXPECTED_RANKING)
        base._read_json = adapted_read
        master = base.build_master(
            successor_intent_path=successor_intent_path,
            expected_intent_sha256=expected_intent_sha256,
        )
    finally:
        base.SCRIPT_PATH = original_script
        base.PLAN_BINDING_KEYS = original_keys
        base.SUPPORTED_RANKING = original_ranking
        base._read_json = original_read
    master["implementation"].update(
        {
            "master_builder": _artifact(SCRIPT_PATH),
            "controller": _artifact(
                PROJECT_ROOT
                / "tools/run_alphazuma_55_configured_successor_certification_v2.py"
            ),
            "watcher": _artifact(
                PROJECT_ROOT
                / "tools/watch_alphazuma_55_configured_successor_certification_v2.py"
            ),
            "independent_auditor": _artifact(
                PROJECT_ROOT
                / "tools/audit_alphazuma_55_configured_successor_result_v2.py"
            ),
            "compatibility_base_master_builder": _artifact(
                PROJECT_ROOT
                / "tools/build_alphazuma_55_configured_successor_master.py"
            ),
            "compatibility_base_controller": _artifact(
                PROJECT_ROOT
                / "tools/run_alphazuma_55_configured_successor_certification.py"
            ),
            "compatibility_base_watcher": _artifact(
                PROJECT_ROOT
                / "tools/watch_alphazuma_55_configured_successor_certification.py"
            ),
            "compatibility_base_auditor": _artifact(
                PROJECT_ROOT
                / "tools/audit_alphazuma_55_configured_successor_result.py"
            ),
        }
    )
    master["successor"]["engineering_validation"].update(
        {
            "plan_lifecycle_status_actual": (
                "FROZEN_BEFORE_TARGET_PREREGISTRATION"
            ),
            "ranking_tie_break_semantics": "earlier_manifest_round_first",
        }
    )
    master["formal_seed_registry_snapshot"] = _artifact(registry_path)
    master["compatibility_adapter"] = {
        "engineering_plan_lifecycle_normalization": (
            "FROZEN_BEFORE_TARGET_PREREGISTRATION -> "
            "FROZEN_DURING_TARGET_TRAINING"
        ),
        "normalization_is_validation_only": True,
        "plan_bytes_changed": False,
        "candidate_order_changed": False,
        "ranking_changed": False,
        "seed_matrix_changed": False,
        "policy_inference_performed": False,
    }
    return master


def main(argv: list[str] | None = None) -> int:
    args = base.build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite V2 successor master: {output}")
    master = build_master(
        successor_intent_path=args.successor_intent.expanduser(),
        expected_intent_sha256=str(args.expected_intent_sha256),
    )
    base._write_json_exclusive(output, master)
    print(
        json.dumps(
            {
                "status": master["status"],
                "output": {
                    "path": str(output),
                    "sha256": base._sha256(output),
                },
                "promotion_gate": master["successor"]["promotion_gate"],
                "ranking": master["successor"]["engineering_validation"][
                    "ranking"
                ],
                "final_blind_seed_range": [
                    master["seed_registry"]["final_blind"]["first"],
                    master["seed_registry"]["final_blind"]["last"],
                ],
                "continuous_seed_range": [
                    master["seed_registry"][
                        "continuous_campaign_challenge"
                    ]["first"],
                    master["seed_registry"][
                        "continuous_campaign_challenge"
                    ]["last"],
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
