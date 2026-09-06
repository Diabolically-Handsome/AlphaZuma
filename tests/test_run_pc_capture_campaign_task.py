"""Tests for materializing a verified campaign task into collector argv."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import run_pc_capture_campaign_task


def _options() -> dict:
    return {
        "dmo": "inputs/input.dmo",
        "prestate_dir": "prestate",
        "direct_runtime_executable": "runtime/popcapgame1.exe",
        "crt_rand_seed": 123,
        "board_seed": 456,
        "global_rng_seed": 123,
        "thread_crt_rng_seed": 123,
        "duration_seconds": 5.5,
        "frame_budget_fps": 240,
        "wait_until_framework_update": 6980,
        "window_repaint_update": 6970,
        "attach_at_update": 0,
        "detach_at_update": 6800,
        "reattach_at_update": 7800,
        "allow_pre_stream_commands": True,
    }


def test_prepare_arguments_are_materialized_without_a_shell(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = tmp_path / "evidence"
    (evidence / "inputs").mkdir(parents=True)
    (evidence / "prestate").mkdir()
    (evidence / "runtime").mkdir()
    (evidence / "inputs" / "input.dmo").write_bytes(b"dmo")
    (evidence / "runtime" / "popcapgame1.exe").write_bytes(b"runtime")
    original = tmp_path / "original"
    original.mkdir()
    (original / "ZumasRevenge.exe").write_bytes(b"launcher")
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    monkeypatch.setattr(
        run_pc_capture_campaign_task,
        "verify_pc_capture_campaign",
        lambda *args, **kwargs: SimpleNamespace(
            status="PASS",
            reasons=(),
            tasks=(
                {
                    "id": "c2",
                    "status": "ready_for_collection",
                    "collector_prepare_options": _options(),
                },
            ),
        ),
    )
    args = Namespace(
        campaign=tmp_path / "campaign.json",
        task="c2",
        evidence_root=evidence,
        original_root=original,
        session_root=sessions / "c2-session",
        session_nonce="1" * 32,
        maximum_hits=10_000,
        trace_timeout_seconds=900.0,
        device_index=0,
        output_index=0,
    )

    result = run_pc_capture_campaign_task._prepare_arguments(args)

    assert result[0] == "prepare"
    assert result[result.index("--session-nonce") + 1] == "1" * 32
    assert result[result.index("--dmo") + 1] == str(
        (evidence / "inputs" / "input.dmo").resolve()
    )
    assert result[result.index("--changedir") + 1] == str(original.resolve())
    assert "--allow-pre-stream-commands" in result
    assert "--startup-trace-handoff" in result
    assert result[result.index("--maximum-hits") + 1] == "10000"

    late_options = _options()
    late_options["attach_at_update"] = 230
    late_options["allow_pre_stream_commands"] = False
    monkeypatch.setattr(
        run_pc_capture_campaign_task,
        "verify_pc_capture_campaign",
        lambda *args, **kwargs: SimpleNamespace(
            status="PASS",
            reasons=(),
            tasks=(
                {
                    "id": "c2",
                    "status": "ready_for_collection",
                    "collector_prepare_options": late_options,
                },
            ),
        ),
    )
    args.session_root = sessions / "c2-late-session"
    late_result = run_pc_capture_campaign_task._prepare_arguments(args)
    assert "--allow-pre-stream-commands" not in late_result
    assert "--startup-trace-handoff" not in late_result
    assert "--allow-attach-stabilization" in late_result
    assert "--allow-pre-attach-file-write-debt" in late_result
    assert "--allow-font-cache-manifest-completion-debt" in late_result


def test_nonready_task_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        run_pc_capture_campaign_task,
        "verify_pc_capture_campaign",
        lambda *args, **kwargs: SimpleNamespace(
            status="PASS",
            reasons=(),
            tasks=(
                {
                    "id": "c7",
                    "status": "requires_recording",
                    "blocker": "new DMO required",
                },
            ),
        ),
    )
    args = Namespace(
        campaign=tmp_path / "campaign.json",
        task="c7",
        evidence_root=tmp_path,
        original_root=tmp_path,
        session_root=tmp_path / "session",
        session_nonce="1" * 32,
        maximum_hits=10_000,
        trace_timeout_seconds=900.0,
        device_index=0,
        output_index=0,
    )

    with pytest.raises(ValueError, match="not ready"):
        run_pc_capture_campaign_task._prepare_arguments(args)
