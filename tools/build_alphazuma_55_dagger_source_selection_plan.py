"""Freeze the polar-probe ranking and downstream DAgger recipe before candidate results."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MASTER = PROJECT_ROOT / "diagnostics/alphazuma-55-weekend-s81081401-preregistration-v1.json"
DEFAULT_POLAR_PLAN = PROJECT_ROOT / "diagnostics/alphazuma-55-polar-reanchor-probe-s99081527-plan-v1.json"
DEFAULT_POLAR_PREREG = PROJECT_ROOT / "diagnostics/alphazuma-55-polar-reanchor-probe-s99081527-preregistration-v1.json"
DEFAULT_POLAR_MANIFEST = PROJECT_ROOT / "diagnostics/alphazuma-55-polar-reanchor-probe-s99081527-models-v1.json"
DEFAULT_POLAR_MATRIX = Path("/mnt/d/ZumaTraining/alphazuma-55-polar-reanchor-probe-s99081527-v1/matrix-shard-00-of-01.json")
DEFAULT_OUTPUT = PROJECT_ROOT / "diagnostics/alphazuma-55-dagger-source-selection-s99081528-plan-v1.json"


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def _write_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-preregistration", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--polar-plan", type=Path, default=DEFAULT_POLAR_PLAN)
    parser.add_argument("--polar-preregistration", type=Path, default=DEFAULT_POLAR_PREREG)
    parser.add_argument("--polar-manifest", type=Path, default=DEFAULT_POLAR_MANIFEST)
    parser.add_argument("--polar-matrix", type=Path, default=DEFAULT_POLAR_MATRIX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen plan: {output}")

    master_path = args.master_preregistration.expanduser().resolve(strict=True)
    polar_plan_path = args.polar_plan.expanduser().resolve(strict=True)
    polar_prereg_path = args.polar_preregistration.expanduser().resolve(strict=True)
    polar_manifest_path = args.polar_manifest.expanduser().resolve(strict=True)
    polar_matrix_path = args.polar_matrix.expanduser().resolve(strict=True)
    master = _read_json(master_path)
    polar_plan = _read_json(polar_plan_path)
    polar_prereg = _read_json(polar_prereg_path)
    polar_manifest = _read_json(polar_manifest_path)
    polar_matrix = _read_json(polar_matrix_path)

    if polar_plan.get("status") != "FROZEN_BEFORE_VARIANT_CREATION":
        raise ValueError("polar plan is not the frozen pre-variant contract")
    if polar_prereg.get("status") != "FROZEN_BEFORE_EVALUATION":
        raise ValueError("polar evaluation preregistration is not frozen")
    if polar_manifest.get("status") != "FROZEN":
        raise ValueError("polar model manifest is not frozen")
    models = polar_manifest.get("models")
    if not isinstance(models, list) or len(models) != 4:
        raise ValueError("expected source plus exactly three polar variants")
    model_ids = [str(row["id"]) for row in models]
    if len(set(model_ids)) != len(model_ids):
        raise ValueError("polar manifest contains duplicate model ids")
    source_id = str(polar_plan["source_model"]["id"])
    candidate_ids = set(model_ids) - {source_id}
    if len(candidate_ids) != 3:
        raise ValueError("polar source/candidate boundary is malformed")

    attempts = polar_matrix.get("attempts")
    if not isinstance(attempts, list):
        raise ValueError("polar matrix attempts are malformed")
    candidate_attempts = [
        row for row in attempts if str(row.get("model_id")) in candidate_ids
    ]
    unknown_attempts = [
        row for row in attempts if str(row.get("model_id")) not in set(model_ids)
    ]
    if unknown_attempts:
        raise ValueError("polar matrix contains an unknown model")
    if candidate_attempts:
        raise ValueError(
            "refusing late freeze: at least one polar candidate outcome already exists"
        )

    training_registry = master["seed_registry"]["training"]
    training_seed_base = 1_542_000_700
    training_seed_last = training_seed_base + 2 * 55 - 1
    if not (
        int(training_registry["first"])
        <= training_seed_base
        <= training_seed_last
        <= int(training_registry["last"])
    ):
        raise ValueError("DAgger seeds escaped the training registry")
    validation_registry = master["seed_registry"]["training_validation"]
    validation_seed_base = 1_550_000_800
    validation_seed_last = validation_seed_base + 55 - 1
    if not (
        int(validation_registry["first"])
        <= validation_seed_base
        <= validation_seed_last
        <= int(validation_registry["last"])
    ):
        raise ValueError("DAgger validation seeds escaped their registry")

    plan = {
        "schema": "zuma-rl.alphazuma-55-dagger-source-selection-plan",
        "version": 1,
        "status": "FROZEN_BEFORE_POLAR_CANDIDATE_OUTCOMES",
        "created_utc": _utc_now(),
        "campaign_id": "alphazuma-55-weekend-s81081401-v1",
        "objective": (
            "Choose the polar source for a student-visited exact-teacher DAgger "
            "repair without adapting the choice rule to candidate outcomes."
        ),
        "master_preregistration": {
            "path": str(master_path),
            "sha256": _sha256(master_path),
        },
        "polar_probe": {
            "plan": {"path": str(polar_plan_path), "sha256": _sha256(polar_plan_path)},
            "preregistration": {"path": str(polar_prereg_path), "sha256": _sha256(polar_prereg_path)},
            "models_manifest": {"path": str(polar_manifest_path), "sha256": _sha256(polar_manifest_path)},
            "expected_matrix_path": str(polar_matrix_path),
            "expected_models_in_order": [
                {"id": str(row["id"]), "sha256": str(row["sha256"])}
                for row in models
            ],
            "expected_levels_in_order": [
                str(row["id"]) for row in polar_prereg["levels"]
            ],
            "expected_attempts": len(models) * len(polar_prereg["levels"]),
        },
        "freeze_boundary_evidence": {
            "matrix_sha256_at_freeze": _sha256(polar_matrix_path),
            "matrix_status_at_freeze": polar_matrix.get("status"),
            "completed_attempts_at_freeze": len(attempts),
            "completed_model_ids_at_freeze": sorted(
                {str(row.get("model_id")) for row in attempts}
            ),
            "source_attempts_completed_at_freeze": sum(
                str(row.get("model_id")) == source_id for row in attempts
            ),
            "candidate_attempts_completed_at_freeze": 0,
            "candidate_outcomes_observed_before_rule_freeze": False,
        },
        "ranking": {
            "scope": "all_12_paired_probe_attempts_per_model",
            "ordered_keys": [
                {"metric": "wins", "direction": "descending"},
                {"metric": "total_score", "direction": "descending"},
                {
                    "metric": "median_winning_ticks",
                    "direction": "ascending",
                    "no_wins_value": "positive_infinity",
                },
                {"metric": "model_sha256", "direction": "ascending"},
            ],
            "selection_count": 1,
            "all_models_eligible": True,
            "no_manual_override": True,
        },
        "dagger_recipe": {
            "rounds": 2,
            "parallel_envs": 24,
            "max_ticks": 30000,
            "teacher_execution_probabilities": [0.25, 0.10],
            "training_seed_base": training_seed_base,
            "training_seed_last_consumed": training_seed_last,
            "reservoir_capacity_by_verb": {"wait": 128, "fire": 192, "swap": 128, "hop": 96},
            "sample_stride_by_verb": {"wait": 4, "fire": 1, "swap": 1, "hop": 1},
            "target_batch_fraction_by_verb": {"wait": 0.60, "fire": 0.30, "swap": 0.08, "hop": 0.02},
            "batch_size": 512,
            "epochs_per_round": 12,
            "checkpoint_interval_epochs": 4,
            "learning_rate": 2e-5,
            "verb_loss_weight": 1.0,
            "max_grad_norm": 1.0,
            "aim_loss_scope": "all_actions",
            "aim_loss_mode": "circular_smoothed",
            "aim_smoothing_sigma": 1.5,
            "aim_distance_weight": 0.5,
            "device": "cuda:0",
            "model_seed": 991529,
            "dataset_aggregation": "student_visited_states_labeled_by_exact_masked_teacher",
            "optimize_after_each_round": True,
            "run_dir": "/mnt/d/ZumaTraining/alphazuma-55-dagger-s99081529-v1",
        },
        "reserved_validation": {
            "base_seed": validation_seed_base,
            "last_seed": validation_seed_last,
            "attempts_per_level": 1,
            "levels": 55,
            "formal_seed_consumption": False,
        },
        "planned_outputs": {
            "independent_probe_audit": str(PROJECT_ROOT / "diagnostics/alphazuma-55-polar-reanchor-probe-s99081527-independent-audit-v1.json"),
            "source_selection_receipt": str(PROJECT_ROOT / "diagnostics/alphazuma-55-dagger-s99081529-source-selection-v1.json"),
            "dagger_preregistration": str(PROJECT_ROOT / "diagnostics/alphazuma-55-dagger-s99081529-preregistration-v1.json"),
        },
        "authority_boundary": {
            "classification": "posthoc_engineering_only",
            "formal_candidate_registration_authorized": False,
            "formal_selection_seed_consumption": False,
            "formal_final_blind_seed_consumption": False,
            "continuous_campaign_seed_consumption": False,
        },
    }
    _write_exclusive(output, plan)
    print(json.dumps({
        "status": plan["status"],
        "path": str(output),
        "sha256": _sha256(output),
        "freeze_boundary_evidence": plan["freeze_boundary_evidence"],
        "ranking": plan["ranking"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
