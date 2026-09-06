"""Select the frozen polar winner and materialize its exact two-round DAgger run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEACHER_PREREG = PROJECT_ROOT / "diagnostics/alphazuma-55-settled-v3-multiseed-s99081525-preregistration-v1.json"
DEFAULT_TRAINER = PROJECT_ROOT / "tools/distill_alphazuma_55_dagger.py"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _bound_path(reference: Any, name: str) -> Path:
    _require(isinstance(reference, dict), f"{name} reference must be an object")
    path = Path(str(reference.get("path", ""))).expanduser().resolve(strict=True)
    _require(reference.get("sha256") == _sha256(path), f"{name} hash mismatch")
    return path


def _write_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _summaries(
    models: list[dict[str, Any]], rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for model in models:
        model_id = str(model["id"])
        model_rows = [row for row in rows if str(row["model_id"]) == model_id]
        wins = [row for row in model_rows if row["outcome"] == "win"]
        result.append(
            {
                "model_id": model_id,
                "model_sha256": str(model["sha256"]),
                "attempts": len(model_rows),
                "wins": len(wins),
                "losses": sum(row["outcome"] == "loss" for row in model_rows),
                "truncations": sum(bool(row["time_limit_truncated"]) for row in model_rows),
                "total_score": sum(int(row["score"]) for row in model_rows),
                "median_winning_ticks": (
                    statistics.median(int(row["ticks"]) for row in wins)
                    if wins
                    else None
                ),
            }
        )
    return result


def _ranking_key(row: dict[str, Any]) -> tuple[Any, ...]:
    median = row["median_winning_ticks"]
    return (
        -int(row["wins"]),
        -int(row["total_score"]),
        float(median) if median is not None else math.inf,
        str(row["model_sha256"]),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-plan", required=True, type=Path)
    parser.add_argument("--expected-selection-plan-sha256", required=True)
    parser.add_argument("--audit-receipt", required=True, type=Path)
    parser.add_argument("--teacher-preregistration", type=Path, default=DEFAULT_TEACHER_PREREG)
    parser.add_argument("--trainer", type=Path, default=DEFAULT_TRAINER)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    selection_plan_path = args.selection_plan.expanduser().resolve(strict=True)
    if _sha256(selection_plan_path) != args.expected_selection_plan_sha256:
        raise ValueError("selection-plan hash differs from the expected frozen hash")
    audit_path = args.audit_receipt.expanduser().resolve(strict=True)
    teacher_prereg_path = args.teacher_preregistration.expanduser().resolve(strict=True)
    trainer_path = args.trainer.expanduser().resolve(strict=True)
    plan = _read_json(selection_plan_path)
    audit = _read_json(audit_path)
    teacher_prereg = _read_json(teacher_prereg_path)
    _require(
        plan.get("schema") == "zuma-rl.alphazuma-55-dagger-source-selection-plan"
        and plan.get("version") == 1
        and plan.get("status") == "FROZEN_BEFORE_POLAR_CANDIDATE_OUTCOMES",
        "unexpected source-selection plan",
    )
    _require(
        plan["freeze_boundary_evidence"]["candidate_attempts_completed_at_freeze"] == 0
        and plan["freeze_boundary_evidence"]["candidate_outcomes_observed_before_rule_freeze"] is False,
        "source-selection rule was not frozen before candidate outcomes",
    )
    _require(
        audit.get("schema") == "zuma-rl.alphazuma-55-polar-reanchor-probe-independent-audit"
        and audit.get("status") == "PASS"
        and audit.get("formal_selection_authority") is False,
        "polar probe lacks a passing independent audit",
    )

    master_path = _bound_path(plan["master_preregistration"], "master")
    polar_plan_path = _bound_path(plan["polar_probe"]["plan"], "polar plan")
    polar_prereg_path = _bound_path(plan["polar_probe"]["preregistration"], "polar preregistration")
    manifest_path = _bound_path(plan["polar_probe"]["models_manifest"], "polar manifest")
    matrix_path = Path(str(plan["polar_probe"]["expected_matrix_path"])).resolve(strict=True)
    _require(
        audit["artifacts"]["master"]["sha256"] == _sha256(master_path)
        and audit["artifacts"]["polar_plan"]["sha256"] == _sha256(polar_plan_path)
        and audit["artifacts"]["preregistration"]["sha256"] == _sha256(polar_prereg_path)
        and audit["artifacts"]["models_manifest"]["sha256"] == _sha256(manifest_path)
        and audit["artifacts"]["matrix"]["sha256"] == _sha256(matrix_path),
        "independent audit does not bind the expected final artifacts",
    )
    auditor_path = _bound_path(audit["artifacts"]["auditor"], "polar auditor")

    manifest = _read_json(manifest_path)
    matrix = _read_json(matrix_path)
    models = manifest.get("models")
    rows = matrix.get("attempts")
    _require(isinstance(models, list) and isinstance(rows, list), "polar artifacts malformed")
    expected_models = plan["polar_probe"]["expected_models_in_order"]
    _require(
        [
            {"id": str(row["id"]), "sha256": str(row["sha256"])}
            for row in models
        ]
        == expected_models,
        "final polar models differ from the pre-outcome plan",
    )
    _require(
        matrix.get("status") == "COMPLETE"
        and matrix.get("error") is None
        and len(rows) == int(plan["polar_probe"]["expected_attempts"]),
        "polar matrix is incomplete",
    )

    summaries = _summaries(models, rows)
    audited_by_id = {str(row["model_id"]): row for row in audit["model_summaries"]}
    for row in summaries:
        audited = audited_by_id.get(str(row["model_id"]))
        _require(audited is not None, f"audit omitted model {row['model_id']}")
        for key in (
            "model_sha256",
            "attempts",
            "wins",
            "losses",
            "truncations",
            "total_score",
            "median_winning_ticks",
        ):
            _require(row[key] == audited[key], f"audit summary mismatch: {row['model_id']} {key}")
    ranked = sorted(summaries, key=_ranking_key)
    ranked = [{"rank": index + 1, **row} for index, row in enumerate(ranked)]
    selected_summary = ranked[0]
    selected_model = next(
        dict(row) for row in models if row["id"] == selected_summary["model_id"]
    )
    selected_path = Path(str(selected_model["path"])).resolve(strict=True)
    _require(selected_model["sha256"] == _sha256(selected_path), "selected model hash mismatch")

    selection_output = Path(str(plan["planned_outputs"]["source_selection_receipt"])).resolve()
    prereg_output = Path(str(plan["planned_outputs"]["dagger_preregistration"])).resolve()
    for output in (selection_output, prereg_output):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite frozen artifact: {output}")
    selection = {
        "schema": "zuma-rl.alphazuma-55-dagger-source-selection",
        "version": 1,
        "status": "SELECTED",
        "selected_utc": _utc_now(),
        "selection_plan": {"path": str(selection_plan_path), "sha256": _sha256(selection_plan_path)},
        "independent_probe_audit": {"path": str(audit_path), "sha256": _sha256(audit_path)},
        "final_matrix": {"path": str(matrix_path), "sha256": _sha256(matrix_path)},
        "ranking_rule": plan["ranking"],
        "ranking": ranked,
        "selected_model": selected_model,
        "formal_seed_consumption": False,
        "formal_candidate_registration_authorized": False,
    }
    _write_exclusive(selection_output, selection)

    recipe = dict(plan["dagger_recipe"])
    _require(recipe.pop("optimize_after_each_round") is True, "DAgger must optimize after each round")
    run = {
        "id": "alphazuma55-dagger-5090",
        **recipe,
        "exact_action_masks_before_every_decision": True,
        "masked_verb_fallback": "wait_with_teacher_aim",
        "verb_balanced_optimization": True,
        "final_runtime_teacher_calls_forbidden": True,
        "optimize_after_each_round": True,
    }
    implementation_names = (
        "legacy_distillation_runtime",
        "behavior_clone_batch_primitive",
        "settled_teacher_v3",
        "environment",
        "core",
        "features",
        "scope",
        "base_settled_v3_distiller",
    )
    source_implementation = teacher_prereg["implementation"]
    implementation = {
        name: dict(source_implementation[name]) for name in implementation_names
    }
    implementation.update(
        {
            "source_selection_plan_builder": {
                "path": str(PROJECT_ROOT / "tools/build_alphazuma_55_dagger_source_selection_plan.py"),
                "sha256": _sha256(PROJECT_ROOT / "tools/build_alphazuma_55_dagger_source_selection_plan.py"),
            },
            "polar_independent_auditor": {"path": str(auditor_path), "sha256": _sha256(auditor_path)},
            "dagger_materializer": {"path": str(Path(__file__).resolve()), "sha256": _sha256(Path(__file__).resolve())},
            "dagger_independent_auditor": {
                "path": str(PROJECT_ROOT / "tools/audit_alphazuma_55_dagger.py"),
                "sha256": _sha256(PROJECT_ROOT / "tools/audit_alphazuma_55_dagger.py"),
            },
        }
    )
    prereg = {
        "schema": "zuma-rl.alphazuma-55-dagger-preregistration",
        "version": 1,
        "status": "FROZEN_BEFORE_TRAINING",
        "created_utc": _utc_now(),
        "campaign_id": "alphazuma-55-weekend-s81081401-v1",
        "objective": "Repair the best frozen polar student on its own visited states using exact teacher labels.",
        "master_preregistration": {"path": str(master_path), "sha256": _sha256(master_path)},
        "source_selection_plan": {"path": str(selection_plan_path), "sha256": _sha256(selection_plan_path)},
        "student_source_model": selected_model,
        "source_selection": {"path": str(selection_output), "sha256": _sha256(selection_output)},
        "teacher_evidence": teacher_prereg["teacher_evidence"],
        "trainer": {"path": str(trainer_path), "sha256": _sha256(trainer_path)},
        "implementation": implementation,
        "levels": list(_read_json(master_path)["scope"]["included_levels_in_adventure_order"]),
        "run": run,
        "reserved_validation": plan["reserved_validation"],
        "authority_boundary": plan["authority_boundary"],
        "formal_seed_consumption": False,
    }
    _write_exclusive(prereg_output, prereg)
    print(json.dumps({
        "status": prereg["status"],
        "selection": {"path": str(selection_output), "sha256": _sha256(selection_output), "selected_model": selected_model, "ranking": ranked},
        "preregistration": {"path": str(prereg_output), "sha256": _sha256(prereg_output)},
        "formal_seed_consumption": False,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
