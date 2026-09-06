"""Freeze five-candidate configured certification for autonomy DAgger."""

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
from tools.materialize_alphazuma_55_polar_intent_autonomy_validation import (
    validate_plan_static,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()
PLAN_SCHEMA = (
    "zuma-rl.alphazuma-55-polar-intent-autonomy-validation-plan"
)
PLAN_BINDING_KEY = "polar_intent_autonomy_validation_plan"
EXPECTED_RANKING = [
    "wins_desc",
    "level_coverage_desc",
    "total_score_desc",
    "median_winning_ticks_asc",
    "earlier_round_first",
]
EXPECTED_CANDIDATES = [
    "polar-intent-autonomy-source-final",
    "polar-intent-autonomy-round-00",
    "polar-intent-autonomy-round-01",
    "polar-intent-autonomy-round-02",
    "polar-intent-autonomy-round-03",
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
        "autonomy successor intent hash differs",
    )
    intent = base._read_json(successor_intent_path)
    base._require(
        intent.get("schema")
        == "zuma-rl.alphazuma-55-polar-intent-autonomy-successor-intent"
        and intent.get("status") == "FROZEN_BEFORE_ENGINEERING_RESULT",
        "unexpected autonomy successor intent",
    )
    actual_candidates = list(
        intent["engineering_validation"]["candidate_order"]
    )
    base._require(
        actual_candidates == EXPECTED_CANDIDATES,
        "autonomy candidate order differs",
    )
    plan_path = Path(
        intent["engineering_validation"]["plan"]["path"]
    ).resolve(strict=True)
    plan_sha = base._sha256(plan_path)
    plan = validate_plan_static(plan_path, plan_sha)
    base._require(
        plan.get("decision_rule", {}).get("ranking") == EXPECTED_RANKING
        and int(plan["matrix"]["models"]) == 5
        and int(plan["matrix"]["expected_attempts"]) == 275,
        "autonomy engineering plan differs",
    )
    registry_reference = intent["formal_seed_registry_snapshot"]
    registry_path = Path(registry_reference["path"]).resolve(strict=True)
    base._require(
        registry_reference["sha256"] == base._sha256(registry_path),
        "autonomy formal registry snapshot changed",
    )
    registry = base._read_json(registry_path)
    base._require(
        registry.get("global_checks", {}).get(
            "ranges_strictly_non_overlapping"
        )
        is True,
        "autonomy prior formal registry is not clean",
    )

    original_script = base.SCRIPT_PATH
    original_keys = base.PLAN_BINDING_KEYS
    original_ranking = base.SUPPORTED_RANKING
    original_read = base._read_json

    def adapted_read(path: Path) -> dict[str, Any]:
        resolved = Path(path).resolve(strict=True)
        value = original_read(resolved)
        if resolved == successor_intent_path:
            value = copy.deepcopy(value)
            value["engineering_validation"]["candidate_order"] = list(
                EXPECTED_CANDIDATES[1:]
            )
        elif resolved == plan_path:
            value = copy.deepcopy(value)
            value["status"] = "FROZEN_DURING_TARGET_TRAINING"
            value["matrix"]["models"] = 4
            value["matrix"]["expected_attempts"] = 220
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
                / "tools/run_alphazuma_55_configured_successor_"
                "certification_autonomy_v1.py"
            ),
            "watcher": _artifact(
                PROJECT_ROOT
                / "tools/watch_alphazuma_55_configured_successor_"
                "certification_autonomy_v1.py"
            ),
            "independent_auditor": _artifact(
                PROJECT_ROOT
                / "tools/audit_alphazuma_55_configured_successor_"
                "result_autonomy_v1.py"
            ),
            "compatibility_base_master_builder": _artifact(
                PROJECT_ROOT
                / "tools/build_alphazuma_55_configured_successor_master.py"
            ),
            "compatibility_base_controller": _artifact(
                PROJECT_ROOT
                / "tools/run_alphazuma_55_configured_successor_"
                "certification.py"
            ),
            "compatibility_base_watcher": _artifact(
                PROJECT_ROOT
                / "tools/watch_alphazuma_55_configured_successor_"
                "certification.py"
            ),
            "compatibility_base_auditor": _artifact(
                PROJECT_ROOT
                / "tools/audit_alphazuma_55_configured_successor_result.py"
            ),
        }
    )
    engineering = master["successor"]["engineering_validation"]
    engineering.update(
        {
            "plan": _artifact(plan_path),
            "plan_schema": PLAN_SCHEMA,
            "manifest_plan_binding_key": PLAN_BINDING_KEY,
            "eligible_model_ids": list(EXPECTED_CANDIDATES),
            "ranking": list(EXPECTED_RANKING),
            "plan_lifecycle_status_actual": (
                "FROZEN_BEFORE_TARGET_PREREGISTRATION"
            ),
            "ranking_tie_break_semantics": (
                "earlier_manifest_round_first"
            ),
            "matrix_models_actual": 5,
            "matrix_attempts_actual": 275,
        }
    )
    master["formal_seed_registry_snapshot"] = _artifact(registry_path)
    master["compatibility_adapter"] = {
        "engineering_plan_lifecycle_normalization": (
            "FROZEN_BEFORE_TARGET_PREREGISTRATION -> "
            "FROZEN_DURING_TARGET_TRAINING"
        ),
        "base_builder_validation_projection": (
            "temporarily omit the non-promotable source baseline and project "
            "5x55 to 4x55 solely while exercising the legacy static builder"
        ),
        "actual_master_inventory_restored": list(EXPECTED_CANDIDATES),
        "actual_engineering_matrix_attempts": 275,
        "normalization_is_validation_only": True,
        "plan_bytes_changed": False,
        "candidate_order_changed_in_actual_master": False,
        "ranking_changed": False,
        "seed_matrix_changed": False,
        "policy_inference_performed": False,
    }
    return master


def main(argv: list[str] | None = None) -> int:
    args = base.build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite autonomy successor master: {output}"
        )
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
                "candidate_order": master["successor"][
                    "engineering_validation"
                ]["eligible_model_ids"],
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
