"""Tests for the guarded memory follow-up task launcher."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import run_pc_memory_followup_task
from zuma_rl.pc_memory_followup import PcMemoryFollowupReport


def _report(status: str) -> PcMemoryFollowupReport:
    return PcMemoryFollowupReport(
        status="PASS",
        tasks=(
            {
                "id": "c2",
                "status": status,
                "collector_arguments": ["--plan", "plan.json"],
            },
        ),
    )


def test_show_is_read_only_and_prints_verified_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        run_pc_memory_followup_task,
        "verify_pc_memory_followup_plan",
        lambda *args, **kwargs: _report(
            "waiting_for_pc_golden_collection"
        ),
    )

    result = run_pc_memory_followup_task.main(
        [
            "show",
            str(tmp_path / "plan.json"),
            "--evidence-root",
            str(tmp_path),
            "--task",
            "c2",
        ]
    )

    assert result == 0
    row = json.loads(capsys.readouterr().out)
    assert row["id"] == "c2"
    assert row["status"] == "waiting_for_pc_golden_collection"


def test_collect_refuses_before_pc_golden_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        run_pc_memory_followup_task,
        "verify_pc_memory_followup_plan",
        lambda *args, **kwargs: _report(
            "waiting_for_pc_golden_collection"
        ),
    )
    with pytest.raises(
        SystemExit,
        match="requires a completed PC Golden source session",
    ):
        run_pc_memory_followup_task.main(
            [
                "collect",
                str(tmp_path / "plan.json"),
                "--evidence-root",
                str(tmp_path),
                "--task",
                "c2",
            ]
        )
