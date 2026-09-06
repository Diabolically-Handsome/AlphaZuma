from __future__ import annotations

import pytest

from tools.snapshot_alphazuma_v11_frontier import (
    classify_preselection_entries,
    summarize_evaluation_shards,
    summarize_training_liveness,
)


def _shard(
    index: int,
    *,
    status: str,
    completed: int,
    expected: int,
    wall: float,
) -> dict:
    return {
        "shard_index": index,
        "status": status,
        "completed_attempts": completed,
        "expected_attempts": expected,
        "runtime": {"wall_seconds": wall},
        "error": None,
    }


def test_running_shards_compute_slowest_eta() -> None:
    result = summarize_evaluation_shards(
        [
            _shard(0, status="RUNNING", completed=40, expected=100, wall=20.0),
            _shard(1, status="RUNNING", completed=25, expected=100, wall=20.0),
        ]
    )

    assert result["status"] == "RUNNING"
    assert result["completed_attempts"] == 65
    assert result["expected_attempts"] == 200
    assert result["eta_seconds"] == 60.0


def test_complete_shards_have_zero_eta() -> None:
    result = summarize_evaluation_shards(
        [
            _shard(0, status="COMPLETE", completed=64, expected=64, wall=30.0),
            _shard(1, status="COMPLETE", completed=72, expected=72, wall=35.0),
        ]
    )

    assert result["status"] == "COMPLETE"
    assert result["eta_seconds"] == 0.0
    assert result["progress_fraction"] == 1.0


def test_error_status_dominates() -> None:
    first = _shard(0, status="ERROR", completed=20, expected=100, wall=10.0)
    first["error"] = {"type": "RuntimeError", "message": "boom"}
    result = summarize_evaluation_shards(
        [first, _shard(1, status="RUNNING", completed=30, expected=100, wall=10.0)]
    )

    assert result["status"] == "ERROR"


@pytest.mark.parametrize(
    "completed,expected,wall",
    [(-1, 10, 1.0), (11, 10, 1.0), (0, 0, 1.0), (0, 10, -1.0)],
)
def test_invalid_shard_numbers_are_rejected(
    completed: int, expected: int, wall: float
) -> None:
    with pytest.raises(ValueError):
        summarize_evaluation_shards(
            [
                _shard(
                    0,
                    status="RUNNING",
                    completed=completed,
                    expected=expected,
                    wall=wall,
                )
            ]
        )


def test_duplicate_shard_index_is_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate shard"):
        summarize_evaluation_shards(
            [
                _shard(0, status="RUNNING", completed=1, expected=10, wall=1.0),
                _shard(0, status="RUNNING", completed=1, expected=10, wall=1.0),
            ]
        )


def test_completed_route_age_does_not_drive_running_staleness() -> None:
    rows = [
        {
            "status": "COMPLETE",
            "completion_exists": True,
            "failure_exists": False,
            "steps_this_run": 100,
            "age_seconds": 600.0,
        },
        {
            "status": "RUNNING",
            "completion_exists": False,
            "failure_exists": False,
            "steps_this_run": 50,
            "age_seconds": 12.0,
        },
    ]

    result = summarize_training_liveness(rows)

    assert result["max_status_age_seconds"] == 600.0
    assert result["max_running_status_age_seconds"] == 12.0
    assert result["completed_routes"] == 1
    assert result["running_routes"] == 1
    assert result["unexpected_incomplete_routes"] == 0


def test_nonrunning_route_without_terminal_receipt_is_unexpected() -> None:
    result = summarize_training_liveness(
        [
            {
                "status": "STOPPED",
                "completion_exists": False,
                "failure_exists": False,
                "steps_this_run": 10,
                "age_seconds": 1.0,
            }
        ]
    )

    assert result["unexpected_incomplete_routes"] == 1
    assert result["max_running_status_age_seconds"] == 0.0


def test_preselection_inventory_allows_only_status_and_atomic_temp() -> None:
    result = classify_preselection_entries(
        ["controller_status.json", ".controller_status.json.2103.tmp"]
    )

    assert result["embargo_boundary_intact"] is True
    assert result["unexpected_entries"] == []
    assert result["temporary_atomic_write_entries"] == [
        ".controller_status.json.2103.tmp"
    ]


def test_preselection_inventory_rejects_early_evaluation_artifact() -> None:
    result = classify_preselection_entries(
        ["controller_status.json", "selection-preregistration.json"]
    )

    assert result["embargo_boundary_intact"] is False
    assert result["unexpected_entries"] == ["selection-preregistration.json"]
