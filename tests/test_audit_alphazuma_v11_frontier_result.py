from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import pytest

from tools.audit_alphazuma_v11_frontier_result import AuditError, audit_campaign
from tools.run_alphazuma_v11_frontier_postprocess import (
    _performance_report,
    _report_markdown,
    _selection_decision,
)
from tools.summarize_multimodel_evaluation import _aggregate_model


LEVEL_IDS = [
    "Jungle1",
    "Jungle2",
    "Jungle3",
    "Jungle4",
    "jungle6",
    "Jungle7",
    "Jungle8",
    "Jungle9",
    "Jungle10",
    "village1",
    "village2",
    "village4",
    "village5",
    "village6",
    "village7",
    "village8",
    "village10",
]
SELECTION_IDS = [
    "Jungle2",
    "Jungle7",
    "Jungle9",
    "village4",
    "village5",
    "village6",
    "village8",
    "village10",
]
ZERO_IDS = ["Jungle7", "village6", "village10"]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _model(path: Path, model_id: str, index: int, source: str) -> dict[str, Any]:
    path.write_bytes(f"model-{model_id}".encode())
    return {
        "id": model_id,
        "training_steps": 1000 + index,
        "frontier_steps": index,
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "source": source,
    }


def _levels(ids: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "id": level_id,
            "display_name": level_id,
            "curve_count": 2 if level_id in {"Jungle9", "village6"} else 1,
            "curve_names": [level_id],
        }
        for level_id in ids
    ]


def _write_evaluation(
    *,
    root: Path,
    preregistration_path: Path,
    manifest_path: Path,
    models: list[dict[str, Any]],
    levels: list[dict[str, Any]],
    attempts_per_level: int,
    base_seed: int,
    purpose: str,
    wins_for: Callable[[str, str], int],
) -> dict[str, Any]:
    preregistration = {
        "schema": "zuma-rl.zero-shot-multilevel-preregistration",
        "version": 1,
        "status": "FROZEN_BEFORE_EVALUATION",
        "frozen_utc": "2026-08-14T00:00:00+00:00",
        "purpose": purpose,
        "model": models[0],
        "levels": levels,
        "attempts_per_level": attempts_per_level,
        "total_attempts": len(levels) * attempts_per_level,
        "seed_plan": {
            "base_seed": base_seed,
            "last_seed": base_seed + len(levels) * attempts_per_level - 1,
            "all_seeds_frozen_before_policy_inference": True,
        },
        "environment": {},
        "execution": {
            "output_root": str(root.resolve()),
            "shard_count": 2,
            "devices": ["cuda:0", "cuda:1"],
            "parallel_envs_per_shard": 24,
        },
        "reporting": {"all_attempts_must_be_reported": True},
        "evaluator": {"path": "/synthetic/evaluator.py", "sha256": "sha256:evaluator"},
    }
    _write_json(preregistration_path, preregistration)
    prereg_hash = _sha256(preregistration_path)
    manifest_hash = _sha256(manifest_path)
    shards: list[list[dict[str, Any]]] = [[], []]
    all_rows = []
    for model_index, model in enumerate(models):
        for level_index, level in enumerate(levels):
            wins = wins_for(str(model["id"]), str(level["id"]))
            for attempt_index in range(attempts_per_level):
                outcome = "win" if attempt_index < wins else "loss"
                row = {
                    "model_id": model["id"],
                    "model_sha256": model["sha256"],
                    "training_steps": model["training_steps"],
                    "level_id": level["id"],
                    "attempt_index": attempt_index,
                    "seed": base_seed + level_index * attempts_per_level + attempt_index,
                    "outcome": outcome,
                    "time_limit_truncated": False,
                    "ticks": 1000 + model_index * 10 + level_index * 100 + attempt_index,
                    "score": 10000 - attempt_index,
                }
                shards[(level_index * attempts_per_level + attempt_index) % 2].append(row)
                all_rows.append(row)
    shard_receipts = []
    root.mkdir(parents=True)
    for index, rows in enumerate(shards):
        shard_path = root / f"matrix-shard-{index:02d}-of-02.json"
        shard = {
            "schema": "zuma-rl.zero-shot-multimodel-shard",
            "version": 1,
            "status": "COMPLETE",
            "error": None,
            "shard_index": index,
            "shard_count": 2,
            "device": f"cuda:{index}",
            "preregistration": {"path": str(preregistration_path), "sha256": prereg_hash},
            "models_manifest": {"path": str(manifest_path), "sha256": manifest_hash},
            "evaluator_sha256": "sha256:evaluator",
            "expected_attempts": len(rows),
            "attempts": rows,
            "runtime": {"wall_seconds": 1.0},
        }
        _write_json(shard_path, shard)
        shard_receipts.append(
            {
                "path": str(shard_path.resolve()),
                "sha256": _sha256(shard_path),
                "device": f"cuda:{index}",
                "attempts": len(rows),
                "wall_seconds": 1.0,
            }
        )
    summaries = []
    for model in models:
        summaries.append(
            _aggregate_model(
                model=model,
                levels=levels,
                rows=[row for row in all_rows if row["model_id"] == model["id"]],
                attempts_per_level=attempts_per_level,
            )
        )
    aggregate = {
        "schema": "zuma-rl.paired-multimodel-evaluation-aggregate",
        "version": 1,
        "status": "COMPLETE",
        "created_utc": "2026-08-14T00:10:00+00:00",
        "purpose": purpose,
        "validation": {
            "all_shards_complete": True,
            "attempt_matrix_exact": True,
            "paired_seed_matrix_exact": True,
            "model_identity_exact": True,
            "evaluator_identity_exact": True,
        },
        "preregistration": {"path": str(preregistration_path), "sha256": prereg_hash},
        "models_manifest": {"path": str(manifest_path), "sha256": manifest_hash},
        "shards": shard_receipts,
        "levels": len(levels),
        "attempts_per_level": attempts_per_level,
        "models": summaries,
        "selection": None,
    }
    _write_json(root / "aggregate.json", aggregate)
    return aggregate


def _campaign(tmp_path: Path) -> tuple[Path, Path]:
    output_root = tmp_path / "campaign"
    output_root.mkdir()
    models_root = tmp_path / "models"
    models_root.mkdir()
    baseline = _model(models_root / "baseline.zip", "baseline-v1", 0, "published_v1_champion")
    baseline["frontier_steps"] = 0
    candidates = [baseline]
    run_ids = [f"route-{index}" for index in range(7)]
    for route_index, route_id in enumerate(run_ids):
        candidates.append(
            _model(
                models_root / f"{route_id}-mid.zip",
                f"{route_id}-mid-{100 + route_index}",
                1 + route_index * 2,
                f"{route_id}:mid",
            )
        )
        candidates.append(
            _model(
                models_root / f"{route_id}-final.zip",
                f"{route_id}-final-{200 + route_index}",
                2 + route_index * 2,
                f"{route_id}:final",
            )
        )

    receipt_root = tmp_path / "training-receipts"
    training_receipts = []
    for route_id in run_ids:
        artifacts = {}
        for field in ("preregistration", "config", "completion"):
            path = receipt_root / f"{route_id}-{field}.json"
            _write_json(path, {"route": route_id, "kind": field})
            artifacts[field] = {"path": str(path.resolve()), "sha256": _sha256(path)}
        training_receipts.append(
            {
                "route_id": route_id,
                **artifacts,
                "midpoint_checkpoint": "synthetic",
                "final_model": "synthetic",
            }
        )

    levels = _levels(LEVEL_IDS)
    master = {
        "schema": "zuma-rl.alphazuma-v11-frontier-postprocess-preregistration",
        "version": 1,
        "status": "FROZEN_BEFORE_SELECTION",
        "baseline_model": baseline,
        "runs": [{"id": route_id} for route_id in run_ids],
        "levels": levels,
        "zero_win_level_ids": ZERO_IDS,
        "selection": {
            "level_ids": SELECTION_IDS,
            "attempts_per_level": 4,
            "seed_base": 910000000,
            "retention_anchor": "Jungle2",
            "ranking": [
                "anchor retained",
                "zero-win breadth",
                "zero-win wins",
                "target breadth",
                "target wins",
                "speed",
            ],
        },
        "final_blind": {"attempts_per_level": 16, "seed_base": 1010000000},
        "promotion_gates": {
            "jungle2_wins_at_least": 14,
            "target_levels_cleared_at_least": 14,
            "target_wins_at_least": 168,
            "zero_win_levels_cleared_at_least": 1,
            "zero_win_total_wins_at_least": 4,
        },
        "stretch_goals": {
            "target_wins_at_least": 176,
            "zero_win_levels_cleared_at_least": 3,
            "zero_win_total_wins_at_least": 12,
        },
        "schedule": {"goal_end_utc": "2099-01-01T00:00:00+00:00"},
    }
    master_path = tmp_path / "master.json"
    _write_json(master_path, master)

    selection_manifest_path = output_root / "selection-models-manifest.json"
    _write_json(
        selection_manifest_path,
        {
            "schema": "zuma-rl.zero-shot-models-manifest",
            "version": 1,
            "status": "FROZEN",
            "models": candidates,
            "training_receipts": training_receipts,
        },
    )
    selected_id = "route-0-final-200"

    def selection_wins(model_id: str, level_id: str) -> int:
        if level_id == "Jungle2":
            return 4
        if model_id == selected_id:
            return {
                "Jungle7": 2,
                "Jungle9": 3,
                "village4": 3,
                "village5": 1,
                "village6": 0,
                "village8": 1,
                "village10": 0,
            }.get(level_id, 0)
        return 1 if level_id == "Jungle9" else 0

    selection_root = output_root / "selection-evaluation"
    selection_aggregate = _write_evaluation(
        root=selection_root,
        preregistration_path=output_root / "selection-preregistration.json",
        manifest_path=selection_manifest_path,
        models=candidates,
        levels=_levels(SELECTION_IDS),
        attempts_per_level=4,
        base_seed=910000000,
        purpose="smoke",
        wins_for=selection_wins,
    )
    selection_decision = _selection_decision(master, selection_aggregate, candidates)
    selection_decision_path = output_root / "selection-decision.json"
    _write_json(selection_decision_path, selection_decision)
    assert selection_decision["selected_model"]["id"] == selected_id

    selected_final = dict(selection_decision["selected_model"])
    selected_final["selected_source_id"] = selected_final["id"]
    selected_final["id"] = "selected-v11"
    final_models = [baseline, selected_final]
    final_manifest_path = output_root / "final-blind-models-manifest.json"
    _write_json(
        final_manifest_path,
        {
            "schema": "zuma-rl.zero-shot-models-manifest",
            "version": 1,
            "status": "FROZEN",
            "models": final_models,
            "selection_decision": {
                "path": str(selection_decision_path.resolve()),
                "sha256": _sha256(selection_decision_path),
            },
            "selection_aggregate": {
                "path": str((selection_root / "aggregate.json").resolve()),
                "sha256": _sha256(selection_root / "aggregate.json"),
            },
        },
    )

    def final_wins(model_id: str, level_id: str) -> int:
        if level_id == "Jungle2":
            return 16 if model_id == "baseline-v1" else 14
        if level_id in ZERO_IDS:
            return 4 if model_id == "selected-v11" and level_id == "Jungle7" else 0
        return 12 if model_id == "baseline-v1" else 14

    final_root = output_root / "final-blind-evaluation"
    final_aggregate = _write_evaluation(
        root=final_root,
        preregistration_path=output_root / "final-blind-preregistration.json",
        manifest_path=final_manifest_path,
        models=final_models,
        levels=levels,
        attempts_per_level=16,
        base_seed=1010000000,
        purpose="morning-target",
        wins_for=final_wins,
    )
    report = _performance_report(
        master=master,
        selection=selection_decision,
        aggregate=final_aggregate,
        output_root=final_root,
    )
    final_report_path = output_root / "final_report.json"
    markdown_path = output_root / "FINAL_REPORT.md"
    _write_json(final_report_path, report)
    markdown_path.write_text(_report_markdown(report), encoding="utf-8")
    controller = {
        "schema": "zuma-rl.alphazuma-v11-frontier-controller-status",
        "version": 1,
        "status": "COMPLETE",
        "phase": "COMPLETE",
        "error": None,
        "master_preregistration": {"path": str(master_path.resolve()), "sha256": _sha256(master_path)},
        "final_report": {"path": str(final_report_path.resolve()), "sha256": _sha256(final_report_path)},
        "summary": {"path": str(markdown_path.resolve()), "sha256": _sha256(markdown_path)},
    }
    _write_json(output_root / "controller_status.json", controller)
    return master_path, output_root


def test_independent_audit_closes_complete_campaign(tmp_path: Path) -> None:
    master_path, output_root = _campaign(tmp_path)

    report = audit_campaign(master_path, output_root)

    assert report["status"] == "PASS"
    assert report["selection"]["candidates"] == 15
    assert report["selection"]["attempt_records"] == 15 * 8 * 4
    assert report["final_blind"]["attempt_records"] == 2 * 17 * 16
    assert report["performance"]["v11_promotion_passed"] is True


def test_independent_audit_rejects_seed_tamper_even_if_shard_receipt_is_rehashed(
    tmp_path: Path,
) -> None:
    master_path, output_root = _campaign(tmp_path)
    shard_path = output_root / "final-blind-evaluation" / "matrix-shard-00-of-02.json"
    shard = json.loads(shard_path.read_text(encoding="utf-8"))
    shard["attempts"][0]["seed"] += 1
    _write_json(shard_path, shard)
    aggregate_path = output_root / "final-blind-evaluation" / "aggregate.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    receipt = next(value for value in aggregate["shards"] if Path(value["path"]) == shard_path.resolve())
    receipt["sha256"] = _sha256(shard_path)
    _write_json(aggregate_path, aggregate)

    with pytest.raises(AuditError, match="attempt matrix or seed matrix"):
        audit_campaign(master_path, output_root)


def test_independent_audit_rejects_rehashed_gate_tamper(tmp_path: Path) -> None:
    master_path, output_root = _campaign(tmp_path)
    report_path = output_root / "final_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["promotion_gates"]["target_wins"]["actual"] -= 1
    _write_json(report_path, report)
    controller_path = output_root / "controller_status.json"
    controller = json.loads(controller_path.read_text(encoding="utf-8"))
    controller["final_report"]["sha256"] = _sha256(report_path)
    _write_json(controller_path, controller)

    with pytest.raises(AuditError, match="promotion gate arithmetic"):
        audit_campaign(master_path, output_root)
