from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import build_alphazuma_55_polar_intent_wide_launch_plan as builder
from tools.build_alphazuma_55_eval_contract import _sha256


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(tmp_path: Path) -> dict[str, Path]:
    original_root = tmp_path / "original"
    original_root.mkdir()
    trainer = tmp_path / "trainer.py"
    trainer.write_text("# frozen trainer\n", encoding="utf-8")
    intent_path = tmp_path / "intent.json"
    intent_run = tmp_path / "intent-run"
    _write_json(
        intent_path,
        {
            "schema": (
                "zuma-rl.alphazuma-55-polar-intent-wide-"
                "distillation-preregistration"
            ),
            "status": "FROZEN_BEFORE_TRAINING",
            "run": {"run_dir": str(intent_run)},
            "trainer": {"path": str(trainer), "sha256": _sha256(trainer)},
        },
    )
    source_path = tmp_path / "effective-wide-launch.json"
    source_run = tmp_path / "effective-wide-run"
    _write_json(
        source_path,
        {
            "schema": "zuma-rl.alphazuma-55-polar-wide-launch-plan",
            "status": "FROZEN_BEFORE_WAIT",
            "target": {"run_dir": str(source_run)},
        },
    )
    return {
        "original_root": original_root,
        "intent_path": intent_path,
        "intent_run": intent_run,
        "source_path": source_path,
        "source_run": source_run,
        "status_root": tmp_path / "watch",
    }


def test_launch_plan_waits_for_source_receipt_and_watcher_exit(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    plan = builder.build(
        intent_preregistration=paths["intent_path"],
        expected_intent_sha256=_sha256(paths["intent_path"]),
        source_launch_plan=paths["source_path"],
        expected_source_plan_sha256=_sha256(paths["source_path"]),
        source_watcher_pid=12345,
        original_root=paths["original_root"],
        status_root=paths["status_root"],
    )

    assert plan["source_terminal"] == {
        "completion": str((paths["source_run"] / "completion.json").resolve()),
        "failure": str((paths["source_run"] / "failure.json").resolve()),
        "watcher_pid_at_registration": 12345,
        "launch_after_either_terminal_receipt_and_watcher_exit": True,
    }
    assert plan["authority_boundary"] == {
        "formal_seed_consumption": False,
        "current_campaign_candidate_authority": False,
        "s99081535_successor_candidate_authority": False,
        "power_restore_authority": False,
    }


def test_launch_plan_rejects_hash_drift_and_existing_outputs(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    common = {
        "intent_preregistration": paths["intent_path"],
        "source_launch_plan": paths["source_path"],
        "expected_source_plan_sha256": _sha256(paths["source_path"]),
        "source_watcher_pid": 12345,
        "original_root": paths["original_root"],
        "status_root": paths["status_root"],
    }

    with pytest.raises(ValueError, match="intent-wide preregistration hash differs"):
        builder.build(expected_intent_sha256="sha256:wrong", **common)

    paths["intent_run"].mkdir()
    with pytest.raises(FileExistsError, match="intent-wide delayed output exists"):
        builder.build(
            expected_intent_sha256=_sha256(paths["intent_path"]), **common
        )
