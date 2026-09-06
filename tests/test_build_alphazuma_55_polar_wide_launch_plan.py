from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import build_alphazuma_55_polar_wide_launch_plan as builder
from tools.build_alphazuma_55_eval_contract import _sha256


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_launch_plan_waits_for_terminal_receipt_and_pid_exit(tmp_path: Path) -> None:
    original_root = tmp_path / "original"
    original_root.mkdir()
    trainer = tmp_path / "trainer.py"
    trainer.write_text("# frozen trainer\n", encoding="utf-8")
    wide_path = tmp_path / "wide.json"
    dagger_path = tmp_path / "dagger.json"
    wide_run = tmp_path / "wide-run"
    dagger_run = tmp_path / "dagger-run"
    _write_json(
        wide_path,
        {
            "schema": (
                "zuma-rl.alphazuma-55-polar-wide-distillation-preregistration"
            ),
            "status": "FROZEN_BEFORE_TRAINING",
            "run": {"run_dir": str(wide_run)},
            "trainer": {"path": str(trainer), "sha256": _sha256(trainer)},
        },
    )
    _write_json(
        dagger_path,
        {
            "schema": "zuma-rl.alphazuma-55-polar-dagger-preregistration",
            "status": "FROZEN_BEFORE_TRAINING",
            "run": {"run_dir": str(dagger_run)},
        },
    )

    plan = builder.build(
        wide_preregistration=wide_path,
        expected_wide_sha256=_sha256(wide_path),
        dagger_preregistration=dagger_path,
        expected_dagger_sha256=_sha256(dagger_path),
        source_trainer_pid=12345,
        original_root=original_root,
        status_root=tmp_path / "watch",
    )

    assert plan["source_terminal"] == {
        "completion": str((dagger_run / "completion.json").resolve()),
        "failure": str((dagger_run / "failure.json").resolve()),
        "trainer_pid_at_registration": 12345,
        "launch_after_either_terminal_receipt_and_process_exit": True,
    }
    assert plan["authority_boundary"] == {
        "formal_seed_consumption": False,
        "current_campaign_candidate_authority": False,
        "s99081535_successor_candidate_authority": False,
        "power_restore_authority": False,
    }


def test_launch_plan_rejects_hash_drift_and_existing_outputs(tmp_path: Path) -> None:
    original_root = tmp_path / "original"
    original_root.mkdir()
    trainer = tmp_path / "trainer.py"
    trainer.write_text("# frozen trainer\n", encoding="utf-8")
    wide_path = tmp_path / "wide.json"
    dagger_path = tmp_path / "dagger.json"
    wide_run = tmp_path / "wide-run"
    _write_json(
        wide_path,
        {
            "schema": (
                "zuma-rl.alphazuma-55-polar-wide-distillation-preregistration"
            ),
            "status": "FROZEN_BEFORE_TRAINING",
            "run": {"run_dir": str(wide_run)},
            "trainer": {"path": str(trainer), "sha256": _sha256(trainer)},
        },
    )
    _write_json(
        dagger_path,
        {
            "schema": "zuma-rl.alphazuma-55-polar-dagger-preregistration",
            "status": "FROZEN_BEFORE_TRAINING",
            "run": {"run_dir": str(tmp_path / "dagger-run")},
        },
    )
    common = {
        "wide_preregistration": wide_path,
        "dagger_preregistration": dagger_path,
        "expected_dagger_sha256": _sha256(dagger_path),
        "source_trainer_pid": 12345,
        "original_root": original_root,
        "status_root": tmp_path / "watch",
    }

    with pytest.raises(ValueError, match="wide preregistration hash differs"):
        builder.build(expected_wide_sha256="sha256:wrong", **common)

    wide_run.mkdir()
    with pytest.raises(FileExistsError, match="delayed wide output already exists"):
        builder.build(expected_wide_sha256=_sha256(wide_path), **common)
