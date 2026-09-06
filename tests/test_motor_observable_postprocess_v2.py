from __future__ import annotations

from pathlib import Path

from tools import (
    build_alphazuma_55_motor_observable_postprocess_plan_v2 as builder,
)
from tools import run_alphazuma_55_motor_observable_postprocess_v2 as controller


def test_recovery_plan_binds_v3_and_preserves_matrix_gate(
    tmp_path: Path,
) -> None:
    output = tmp_path / "plan.json"
    value = builder.build(
        project_root=builder.SCRIPT_PATH.parents[1], output=output
    )
    assert value["campaign_id"] == builder.CAMPAIGN_ID
    assert value["screen"]["expected_attempts"] == 156
    assert value["full55_gate"]["maximum_expected_attempts"] == 275
    assert value["full55_gate"]["minimum_wins"] == 35
    assert value["full55_gate"]["minimum_cleared_levels"] == 35
    assert value["predecessor_postprocess_incident"][
        "matrix_artifacts_present"
    ] is False
    assert value["seed_reuse_disclosure"]["formal_gate_unchanged"] is True
    validated = controller.validate_plan(output, controller._sha256(output))
    assert validated["training_route"] == value["training_route"]


def test_process_fragment_is_redirected_to_v3() -> None:
    observed: list[str] = []

    def fake_processes(fragment: str) -> list[int]:
        observed.append(fragment)
        return [123]

    result = controller._recovery_processes(
        fake_processes,
        "distill_alphazuma_55_motor_observable_replay_v2.py",
    )
    assert result == [123]
    assert observed == [
        "distill_alphazuma_55_motor_observable_replay_v3.py"
    ]
