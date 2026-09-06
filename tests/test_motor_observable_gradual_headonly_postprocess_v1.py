from __future__ import annotations

import json
from pathlib import Path

from tools import (
    build_alphazuma_55_motor_observable_gradual_headonly_postprocess_plan_v1
    as builder,
)
from tools import (
    run_alphazuma_55_motor_observable_gradual_headonly_postprocess_v1
    as controller,
)


def _write(path: Path, value: dict[str, object]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def test_headonly_candidate_inventory_is_one_plus_sixteen(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.zip"
    source.write_bytes(b"source")
    route_path = tmp_path / "route.json"
    _write(
        route_path,
        {
            "source": {
                "model": {
                    "path": str(source),
                    "sha256": controller._sha256(source),
                }
            }
        },
    )
    schedule = (1.0, 0.9, 0.7, 0.5)
    checkpoint_epochs = (3, 6, 9, 12)
    rounds: list[dict[str, object]] = []
    for round_index, teacher_probability in enumerate(schedule):
        checkpoints: list[dict[str, object]] = []
        for epoch in checkpoint_epochs:
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
                "teacher_execution_probability": teacher_probability,
                "collection": {
                    "episodes": [
                        {"steps": 10 + level_index}
                        for level_index in range(55)
                    ]
                },
                "optimization": {
                    "aim_loss_scope": "fire_only",
                    "optimizer_parameter_scope": "action_net_only",
                    "frozen_parameters_unchanged": True,
                    "checkpoints": checkpoints,
                },
            }
        )
    completion_path = tmp_path / "completion.json"
    _write(
        completion_path,
        {
            "schema": (
                "zuma-rl.alphazuma-55-motor-observable-gradual-completion"
            ),
            "status": "COMPLETE",
            "motor_observation_profile": "motor-observable-v1",
            "teacher_execution_probabilities": list(schedule),
            "state_distribution_semantics": "gradual_teacher_student_dagger",
            "final_runtime_teacher_calls_forbidden": True,
            "formal_seed_consumption": False,
            "formal_candidate_authority": False,
            "route_variant": "action_head_only_fire_aim_v1",
            "optimizer_parameter_scope": "action_net_only",
            "frozen_backbone": True,
            "frozen_parameters_unchanged": True,
            "rounds": rounds,
        },
    )
    candidates, evidence = controller._candidate_inventory(
        plan={
            "training_route": {
                "path": str(route_path),
                "sha256": controller._sha256(route_path),
            },
            "expected_training_completion": str(completion_path),
        }
    )
    assert len(candidates) == 17
    assert candidates[0]["id"] == "motor-v3-source-final"
    assert candidates[0]["promotable"] is False
    assert [row["id"] for row in candidates[1:]] == [
        f"headonly-gradual-r{round_index:02d}-e{epoch:02d}"
        for round_index in range(4)
        for epoch in checkpoint_epochs
    ]
    assert evidence["sha256"] == controller._sha256(completion_path)


def test_builder_freezes_distinct_unused_engineering_gate_ranges(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(builder, "OUTPUT_ROOT", tmp_path / "output")
    monkeypatch.setattr(builder, "_assert_fresh_seed_bases", lambda _path: None)
    value = builder.build(output=tmp_path / "plan.json")
    assert value["screen"]["base_seed"] == 1_550_003_000
    assert value["screen"]["last_seed"] == 1_550_003_011
    assert value["screen"]["expected_attempts"] == 204
    assert value["full55_gate"]["base_seed"] == 1_550_003_100
    assert value["full55_gate"]["last_seed"] == 1_550_003_154
    assert value["full55_gate"]["maximum_expected_attempts"] == 275
    assert value["full55_gate"]["minimum_wins"] == 35
    assert value["full55_gate"]["minimum_cleared_levels"] == 35
    assert value["training_seed_reuse_disclosure"][
        "formal_seed_overlap"
    ] is False
