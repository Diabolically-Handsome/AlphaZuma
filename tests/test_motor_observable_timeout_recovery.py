from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import (
    audit_alphazuma_55_motor_observable_timeout_recovery_v1 as recovery,
)


def _write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path, *, truncated: bool, ticks: int) -> tuple[Path, Path]:
    output = tmp_path / "matrix"
    output.mkdir()
    evaluator = tmp_path / "evaluator.py"
    evaluator.write_text("# fixture\n", encoding="utf-8")
    prereg = tmp_path / "prereg.json"
    manifest = tmp_path / "manifest.json"
    model = {
        "id": "model",
        "sha256": "sha256:model",
        "observation_profile": "motor-observable-v1",
        "promotable": True,
        "candidate_order": 1,
    }
    level = {"id": "Jungle1"}
    _write(
        prereg,
        {
            "evaluator": {
                "path": str(evaluator),
                "sha256": recovery._sha256(evaluator),
            },
            "levels": [level],
            "seed_plan": {"base_seed": 100},
            "environment": {"base_config": {"max_ticks": 30000}},
            "execution": {"output_root": str(output), "shard_count": 1},
        },
    )
    _write(manifest, {"models": [model]})
    row = {
        "model_id": "model",
        "model_sha256": "sha256:model",
        "level_id": "Jungle1",
        "seed": 100,
        "attempt_index": 0,
        "outcome": None,
        "time_limit_truncated": truncated,
        "ticks": ticks,
        "score": 42,
        "observation_capacity_overflow": False,
    }
    _write(
        output / "matrix-shard-00-of-01.json",
        {
            "status": "COMPLETE",
            "error": None,
            "shard_index": 0,
            "shard_count": 1,
            "completed_attempts": 1,
            "expected_attempts": 1,
            "preregistration": {"sha256": recovery._sha256(prereg)},
            "models_manifest": {"sha256": recovery._sha256(manifest)},
            "evaluator_sha256": recovery._sha256(evaluator),
            "models": [model],
            "levels": [level],
            "attempts": [row],
        },
    )
    return prereg, manifest


def test_canonical_timeout_is_a_non_win(tmp_path: Path) -> None:
    prereg, manifest = _fixture(tmp_path, truncated=True, ticks=30000)
    result = recovery.audit_matrix_timeout_compatible(
        prereg_path=prereg,
        manifest_path=manifest,
    )
    summary = result["model_summaries"][0]
    assert summary["wins"] == 0
    assert summary["losses"] == 0
    assert summary["truncations"] == 1
    assert summary["non_wins"] == 1


@pytest.mark.parametrize(
    ("truncated", "ticks"),
    [(False, 30000), (True, 29999)],
)
def test_noncanonical_null_outcome_is_rejected(
    tmp_path: Path, truncated: bool, ticks: int
) -> None:
    prereg, manifest = _fixture(
        tmp_path,
        truncated=truncated,
        ticks=ticks,
    )
    with pytest.raises(ValueError, match="noncanonical outcome"):
        recovery.audit_matrix_timeout_compatible(
            prereg_path=prereg,
            manifest_path=manifest,
        )
