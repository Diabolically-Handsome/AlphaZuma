from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from tools import build_alphazuma_55_polar_intent_wide_validation_plan_v2 as builder
from tools import watch_alphazuma_55_polar_intent_wide_validation_v2 as watcher
from tools.build_alphazuma_55_eval_contract import _sha256


def test_watcher_enters_validation_only_after_launch_is_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text("{}", encoding="utf-8")
    launch_status = tmp_path / "launch.json"
    launch_status.write_text(json.dumps({"status": "COMPLETE"}), encoding="utf-8")
    plan = {
        "implementation": {
            "watcher": {
                "path": str(watcher.SCRIPT_PATH),
                "sha256": _sha256(watcher.SCRIPT_PATH),
            }
        },
        "outputs": {"status_root": str(tmp_path / "status")},
        "launch_controller": {"status": str(launch_status)},
    }
    monkeypatch.setattr(watcher, "validate_plan_static", lambda *_: plan)
    original = watcher.legacy.SCRIPT_PATH
    observed: list[Path] = []

    def fake_run(**_: object) -> dict[str, object]:
        observed.append(watcher.legacy.SCRIPT_PATH)
        return {"status": "COMPLETE"}

    monkeypatch.setattr(watcher.legacy, "run", fake_run)
    result = watcher.run(
        plan_path=plan_path,
        expected_plan_sha256="sha256:test",
        poll_seconds=0.01,
    )

    assert result == {"status": "COMPLETE"}
    assert observed == [watcher.SCRIPT_PATH]
    assert watcher.legacy.SCRIPT_PATH == original


def test_v2_builder_records_exact_supersession(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    supersedes = tmp_path / "superseded.json"
    supersedes.write_text("{}\n", encoding="utf-8")
    expected = _sha256(supersedes)
    monkeypatch.setattr(
        builder.legacy,
        "build",
        lambda **_: {
            "status": "FROZEN_DURING_TARGET_TRAINING",
            "implementation": {},
        },
    )
    args = argparse.Namespace(
        supersedes_plan=supersedes,
        expected_supersedes_sha256=expected,
        master_preregistration=tmp_path,
        training_preregistration=tmp_path,
        launch_plan=tmp_path,
        source_training_preregistration=tmp_path,
        source_completion=tmp_path,
        original_root=tmp_path,
        output_root=tmp_path / "output",
        models_manifest=tmp_path / "models",
        evaluation_preregistration=tmp_path / "evaluation",
        audit_output=tmp_path / "audit",
        prior_wide_audit=tmp_path / "wide-audit",
        status_root=tmp_path / "status",
        device="cuda:1",
        parallel_envs=24,
    )

    value = builder.build(args)

    assert value["supersedes"]["sha256"] == expected
    assert value["race_fix"][
        "wait_for_launch_controller_terminal_before_completion_read"
    ] is True
    assert value["race_fix"]["seed_matrix_changed"] is False
