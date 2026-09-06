from __future__ import annotations

import json
from pathlib import Path

from tools import (
    audit_alphazuma_55_motor_observable_postprocess_v1 as auditor,
)
from tools import (
    build_alphazuma_55_motor_observable_postprocess_plan_v1 as builder,
)
from tools import run_alphazuma_55_motor_observable_postprocess_v1 as controller


def _write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def test_plan_freezes_screen_gate_and_nonformal_seeds(tmp_path: Path) -> None:
    output = tmp_path / "plan.json"
    value = builder.build(
        project_root=builder.SCRIPT_PATH.parents[1],
        output=output,
    )
    assert [row["id"] for row in value["screen_levels"]] == list(
        controller.SCREEN_LEVEL_IDS
    )
    assert value["screen"]["expected_attempts"] == 156
    assert value["full55_gate"]["maximum_expected_attempts"] == 275
    assert value["full55_gate"]["minimum_wins"] == 35
    assert value["full55_gate"]["minimum_cleared_levels"] == 35
    assert value["authority_boundary"][
        "formal_selection_seed_consumption"
    ] is False
    validated = controller.validate_plan(output, controller._sha256(output))
    assert validated["campaign_id"] == builder.CAMPAIGN_ID


def test_candidate_inventory_keeps_all_twelve_early_checkpoints(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.zip"
    source.write_bytes(b"source")
    route_path = tmp_path / "route.json"
    route = {
        "source": {
            "model_path": str(source),
            "model_sha256": controller._sha256(source),
        }
    }
    _write(route_path, route)
    rounds: list[dict[str, object]] = []
    for round_index in range(3):
        checkpoints: list[dict[str, object]] = []
        for epoch in range(1, 5):
            model = tmp_path / f"r{round_index}-e{epoch}.zip"
            model.write_bytes(f"{round_index}-{epoch}".encode())
            checkpoints.append(
                {
                    "epoch": epoch,
                    "path": str(model),
                    "sha256": controller._sha256(model),
                }
            )
        rounds.append(
            {
                "round_index": round_index,
                "collection": {"episodes": [{"steps": round_index + 10}]},
                "optimization": {"checkpoints": checkpoints},
            }
        )
    completion_path = tmp_path / "completion.json"
    _write(
        completion_path,
        {
            "schema": "zuma-rl.alphazuma-55-motor-observable-replay-completion",
            "status": "COMPLETE",
            "motor_observation_profile": "motor-observable-v1",
            "formal_seed_consumption": False,
            "formal_candidate_authority": False,
            "anchor_collections": [{"episodes": [{"steps": 7}]}],
            "student_rounds": rounds,
        },
    )
    candidates, receipt = controller._candidate_inventory(
        plan={
            "training_route": {
                "path": str(route_path),
                "sha256": controller._sha256(route_path),
            },
            "expected_training_completion": str(completion_path),
        }
    )
    assert len(candidates) == 13
    assert candidates[0]["promotable"] is False
    assert [row["id"] for row in candidates[1:]] == [
        f"motor-r{round_index:02d}-e{epoch:02d}"
        for round_index in range(3)
        for epoch in range(1, 5)
    ]
    assert receipt["sha256"] == controller._sha256(completion_path)


def test_matrix_audit_recomputes_paired_ranking(tmp_path: Path) -> None:
    stage = tmp_path / "matrix"
    stage.mkdir()
    evaluator = controller.SCRIPT_PATH.parents[0] / (
        "evaluate_alphazuma_55_motor_observable.py"
    )
    models = [
        {
            "id": "base",
            "sha256": "sha256:base",
            "observation_profile": "human-speedrun-v1",
            "promotable": False,
            "candidate_order": 0,
        },
        {
            "id": "motor",
            "sha256": "sha256:motor",
            "observation_profile": "motor-observable-v1",
            "promotable": True,
            "candidate_order": 1,
        },
    ]
    levels = [
        {"id": "Jungle1", "display_name": "A", "curve_count": 1},
        {"id": "Jungle9", "display_name": "B", "curve_count": 2},
    ]
    prereg_path = tmp_path / "prereg.json"
    manifest_path = tmp_path / "manifest.json"
    _write(
        prereg_path,
        {
            "evaluator": {
                "path": str(evaluator),
                "sha256": controller._sha256(evaluator),
            },
            "levels": levels,
            "seed_plan": {"base_seed": 100},
            "execution": {"output_root": str(stage), "shard_count": 2},
        },
    )
    _write(manifest_path, {"models": models})
    for shard_index, level in enumerate(levels):
        attempts = []
        for model in models:
            win = model["id"] == "motor" or level["id"] == "Jungle1"
            attempts.append(
                {
                    "model_id": model["id"],
                    "model_sha256": model["sha256"],
                    "level_id": level["id"],
                    "seed": 100 + shard_index,
                    "outcome": "win" if win else "loss",
                    "ticks": 500 + shard_index,
                    "score": 1000 if win else 10,
                    "observation_capacity_overflow": False,
                }
            )
        _write(
            stage / f"matrix-shard-{shard_index:02d}-of-02.json",
            {
                "status": "COMPLETE",
                "error": None,
                "completed_attempts": 2,
                "expected_attempts": 2,
                "preregistration": {
                    "sha256": controller._sha256(prereg_path)
                },
                "models_manifest": {
                    "sha256": controller._sha256(manifest_path)
                },
                "evaluator_sha256": controller._sha256(evaluator),
                "attempts": attempts,
            },
        )
    result = controller.audit_matrix(
        prereg_path=prereg_path,
        manifest_path=manifest_path,
    )
    assert result["status"] == "PASS"
    assert result["attempts"] == 4
    assert result["ranking"][0]["model_id"] == "motor"
    assert result["ranking"][0]["wins"] == 2


def _synthetic_matrix(
    *,
    root: Path,
    prereg_path: Path,
    manifest_path: Path,
    models: list[dict[str, object]],
    levels: list[dict[str, object]],
    base_seed: int,
    wins_by_model: dict[str, int],
) -> None:
    evaluator = controller.SCRIPT_PATH.parent / (
        "evaluate_alphazuma_55_motor_observable.py"
    )
    _write(
        prereg_path,
        {
            "evaluator": {
                "path": str(evaluator),
                "sha256": controller._sha256(evaluator),
            },
            "levels": levels,
            "seed_plan": {"base_seed": base_seed},
            "execution": {"output_root": str(root), "shard_count": 2},
        },
    )
    _write(manifest_path, {"models": models})
    root.mkdir(parents=True)
    for shard_index in range(2):
        start, end = controller._shard_bounds(len(levels), shard_index, 2)
        attempts: list[dict[str, object]] = []
        for level_index in range(start, end):
            level = levels[level_index]
            for model in models:
                win = level_index < wins_by_model[str(model["id"])]
                attempts.append(
                    {
                        "model_id": model["id"],
                        "model_sha256": model["sha256"],
                        "level_id": level["id"],
                        "seed": base_seed + level_index,
                        "outcome": "win" if win else "loss",
                        "ticks": 1000 + level_index,
                        "score": 2000 if win else 20,
                        "observation_capacity_overflow": False,
                    }
                )
        _write(
            root / f"matrix-shard-{shard_index:02d}-of-02.json",
            {
                "status": "COMPLETE",
                "error": None,
                "completed_attempts": len(attempts),
                "expected_attempts": len(attempts),
                "preregistration": {
                    "sha256": controller._sha256(prereg_path)
                },
                "models_manifest": {
                    "sha256": controller._sha256(manifest_path)
                },
                "evaluator_sha256": controller._sha256(evaluator),
                "attempts": attempts,
            },
        )


def test_independent_auditor_accepts_only_recomputed_gate(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan.json"
    plan = builder.build(
        project_root=builder.SCRIPT_PATH.parents[1],
        output=plan_path,
    )
    output_root = tmp_path / "postprocess"
    plan["outputs"]["output_root"] = str(output_root)
    plan["outputs"]["independent_audit_receipt"] = str(
        tmp_path / "audit.json"
    )
    _write(plan_path, plan)
    plan_hash = controller._sha256(plan_path)

    screen_models = [
        {
            "id": "base" if index == 0 else f"motor-{index:02d}",
            "sha256": f"sha256:model-{index}",
            "observation_profile": (
                "human-speedrun-v1" if index == 0 else "motor-observable-v1"
            ),
            "promotable": index != 0,
            "candidate_order": index,
        }
        for index in range(13)
    ]
    screen_prereg = output_root / "screen-preregistration.json"
    screen_manifest = output_root / "screen-models-manifest.json"
    _synthetic_matrix(
        root=output_root / "screen",
        prereg_path=screen_prereg,
        manifest_path=screen_manifest,
        models=screen_models,
        levels=plan["screen_levels"],
        base_seed=plan["screen"]["base_seed"],
        wins_by_model={
            str(model["id"]): max(0, 12 - index)
            for index, model in enumerate(screen_models)
        },
    )
    screen_audit = controller.audit_matrix(
        prereg_path=screen_prereg,
        manifest_path=screen_manifest,
    )
    selected_ids = [row["model_id"] for row in screen_audit["ranking"][:5]]
    _write(
        output_root / "screen-audit.json",
        {**screen_audit, "selected_for_full55_gate": selected_ids},
    )

    gate_models = [
        next(row for row in screen_models if row["id"] == model_id)
        for model_id in selected_ids
    ]
    gate_prereg = output_root / "full55-preregistration.json"
    gate_manifest = output_root / "full55-models-manifest.json"
    _synthetic_matrix(
        root=output_root / "full55",
        prereg_path=gate_prereg,
        manifest_path=gate_manifest,
        models=gate_models,
        levels=plan["full55_levels"],
        base_seed=plan["full55_gate"]["base_seed"],
        wins_by_model={
            str(model["id"]): (55 if model["id"] == "base" else 41 - index)
            for index, model in enumerate(gate_models)
        },
    )
    gate_audit = controller.audit_matrix(
        prereg_path=gate_prereg,
        manifest_path=gate_manifest,
    )
    winner = next(row for row in gate_audit["ranking"] if row["promotable"])
    final_report = {
        "schema": "zuma-rl.alphazuma-55-motor-observable-postprocess-result",
        "status": "COMPLETE_GATE_PASS",
        "formal_seed_consumption": "NONE",
        "formal_candidate_authority": True,
        "promotion_gate": {
            "passed": True,
            "winner": winner,
        },
    }
    final_path = output_root / "final_report.json"
    _write(final_path, final_report)
    _write(
        output_root / "preaudit-controller-status.json",
        {
            "status": "RUNNING",
            "phase": "RUNNING_INDEPENDENT_AUDIT",
            "final_report": {
                "sha256": controller._sha256(final_path)
            },
        },
    )
    receipt = auditor.audit(
        plan_path=plan_path,
        expected_plan_sha256=plan_hash,
    )
    assert receipt["status"] == "PASS"
    assert receipt["gate_passed"] is True
    assert receipt["winner"]["model_id"] == winner["model_id"]
