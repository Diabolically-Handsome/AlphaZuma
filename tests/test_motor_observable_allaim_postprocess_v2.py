from __future__ import annotations

import json
from pathlib import Path

from tools import (
    audit_alphazuma_55_motor_observable_timeout_recovery_v1 as timeout_reader,
)
from tools import (
    run_alphazuma_55_motor_observable_gradual_allaim_postprocess_v2 as controller,
)


def test_v2_controller_uses_timeout_compatible_matrix_reader() -> None:
    assert controller.audit_matrix is timeout_reader.audit_matrix_timeout_compatible


def test_timeout_correction_does_not_change_gate_constants() -> None:
    assert controller.MAXIMUM_GATE_CANDIDATES == 5
    assert controller.legacy.EXPECTED_SCREEN_SEED == 1_550_002_800
    assert controller.legacy.EXPECTED_GATE_SEED == 1_550_002_900


def _write(path: Path, value: dict[str, object]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def test_all_actions_inventory_realizes_one_plus_sixteen_candidates(
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

    rounds: list[dict[str, object]] = []
    schedule = (1.0, 0.9, 0.7, 0.5)
    checkpoint_epochs = (3, 6, 9, 12)
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
                    "aim_loss_scope": "all_actions",
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
            "route_variant": "all_actions_aim_supervision_v1",
            "aim_loss_scope": "all_actions",
            "all_action_aim_supervision": True,
            "rounds": rounds,
        },
    )

    candidates, receipt = controller.legacy._candidate_inventory(
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
        f"allaim-gradual-r{round_index:02d}-e{epoch:02d}"
        for round_index in range(4)
        for epoch in checkpoint_epochs
    ]
    assert receipt["sha256"] == controller._sha256(completion_path)
