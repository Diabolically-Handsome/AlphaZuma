"""Tests for content-addressed post-video memory follow-up planning."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from zuma_rl import pc_memory_followup


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(
    tmp_path: Path,
    *,
    slowdown_update: int = 240,
) -> tuple[Path, Path, Path]:
    root = tmp_path / "evidence"
    root.mkdir()
    session = root / "session"
    (session / "inputs").mkdir(parents=True)
    (session / "protocol").mkdir()
    (session / "safety").mkdir()
    dmo = session / "inputs" / "input.dmo"
    dmo.write_bytes(b"dmo")
    (session / "protocol" / "pre-template.json").write_text(
        "{}",
        encoding="ascii",
    )
    (session / "safety" / "host-pre.json").write_text(
        "{}",
        encoding="ascii",
    )
    collection_plan = {
        "schema": pc_memory_followup.COLLECTION_PLAN_SCHEMA,
        "version": pc_memory_followup.COLLECTION_PLAN_VERSION,
        "session_nonce": "nonce",
        "dmo": {
            "artifact": "inputs/input.dmo",
            "bytes": len(dmo.read_bytes()),
            "sha256": _digest(dmo),
            "random_seed": 123,
            "length_updates": 1000,
        },
        "trace": {
            "detach_at_update": 120,
            "reattach_at_update": 800,
        },
        "capture": {
            "window_repaint_update": 200,
            "window_repaint_guard": {
                "previous_effectful_command_update": 150,
                "next_effectful_command_update": 900,
            },
        },
    }
    session_plan = session / "plan.json"
    session_plan.write_text(
        json.dumps(collection_plan, sort_keys=True),
        encoding="utf-8",
    )
    plan = {
        "schema": pc_memory_followup.PLAN_SCHEMA,
        "version": pc_memory_followup.PLAN_VERSION,
        "tasks": [
            {
                "id": "memory",
                "session_path": "session",
                "session_plan_sha256": _digest(session_plan),
                "output_path": "memory-output",
                "slowdown_update": slowdown_update,
                "probe_update": 250,
                "trajectory_end_update": 750,
                "maximum_attempts": 3,
            }
        ],
    }
    followup = tmp_path / "followup.json"
    followup.write_text(
        json.dumps(plan, sort_keys=True),
        encoding="utf-8",
    )
    return followup, root, session


def _demo() -> SimpleNamespace:
    return SimpleNamespace(
        random_seed=123,
        length_updates=1000,
        commands=(
            SimpleNamespace(update=150, kind="mouse_button"),
            SimpleNamespace(update=500, kind="idle"),
            SimpleNamespace(update=900, kind="mouse_move"),
        ),
    )


def test_prepared_followup_waits_for_pc_golden_without_losing_args(
    tmp_path: Path,
    monkeypatch,
) -> None:
    plan, root, _ = _fixture(tmp_path)
    monkeypatch.setattr(
        pc_memory_followup.PopCapDemo,
        "read",
        lambda path: _demo(),
    )

    report = pc_memory_followup.verify_pc_memory_followup_plan(
        plan,
        evidence_root=root,
    )

    assert report.status == "PASS"
    assert report.summary["waiting_for_pc_golden_count"] == 1
    row = report.tasks[0]
    assert row["status"] == "waiting_for_pc_golden_collection"
    assert row["trajectory_tick_count"] == 501
    assert "--discover-score" in row["collector_arguments"]
    assert "--int32-value" not in row["collector_arguments"]


def test_completed_pc_golden_unlocks_memory_collection(
    tmp_path: Path,
    monkeypatch,
) -> None:
    plan, root, session = _fixture(tmp_path)
    monkeypatch.setattr(
        pc_memory_followup.PopCapDemo,
        "read",
        lambda path: _demo(),
    )
    (session / "collection.json").write_text(
        json.dumps(
            {
                "schema": pc_memory_followup.COLLECTION_RESULT_SCHEMA,
                "version": pc_memory_followup.COLLECTION_RESULT_VERSION,
                "status": "complete",
                "session_nonce": "nonce",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    report = pc_memory_followup.verify_pc_memory_followup_plan(
        plan,
        evidence_root=root,
    )

    assert report.status == "PASS"
    assert report.summary["ready_for_memory_collection_count"] == 1
    assert report.tasks[0]["status"] == "ready_for_memory_collection"


def test_session_plan_tampering_fails_closed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    plan, root, session = _fixture(tmp_path)
    monkeypatch.setattr(
        pc_memory_followup.PopCapDemo,
        "read",
        lambda path: _demo(),
    )
    (session / "plan.json").write_text("{}", encoding="utf-8")

    report = pc_memory_followup.verify_pc_memory_followup_plan(
        plan,
        evidence_root=root,
    )

    assert report.status == "FAIL"
    assert "SHA-256 differs" in report.reasons[0]


def test_repaint_headroom_is_enforced(
    tmp_path: Path,
    monkeypatch,
) -> None:
    plan, root, _ = _fixture(tmp_path, slowdown_update=220)
    monkeypatch.setattr(
        pc_memory_followup.PopCapDemo,
        "read",
        lambda path: _demo(),
    )

    report = pc_memory_followup.verify_pc_memory_followup_plan(
        plan,
        evidence_root=root,
    )

    assert report.status == "FAIL"
    assert "repaint-handshake headroom" in report.reasons[0]


def test_source_bound_blackout_can_cross_effectful_command(
    tmp_path: Path,
    monkeypatch,
) -> None:
    plan, root, session = _fixture(tmp_path)
    collection_plan_path = session / "plan.json"
    collection_plan = json.loads(
        collection_plan_path.read_text(encoding="utf-8")
    )
    collection_plan["trace"].update(
        {
            "detach_at_update": 170,
            "reattach_at_update": 950,
            "startup_trace_handoff": True,
            "source_bound_board_anchor": {
                "framework_update": 150,
            },
            "source_bound_board_precall_global_restore": {
                "framework_update": 150,
                "suspend_other_threads_until_board_call": True,
            },
        }
    )
    collection_plan_path.write_text(
        json.dumps(collection_plan, sort_keys=True),
        encoding="utf-8",
    )
    followup = json.loads(plan.read_text(encoding="utf-8"))
    followup["tasks"][0]["session_plan_sha256"] = _digest(
        collection_plan_path
    )
    followup["tasks"][0]["trajectory_end_update"] = 920
    plan.write_text(json.dumps(followup, sort_keys=True), encoding="utf-8")
    monkeypatch.setattr(
        pc_memory_followup.PopCapDemo,
        "read",
        lambda path: _demo(),
    )

    report = pc_memory_followup.verify_pc_memory_followup_plan(
        plan,
        evidence_root=root,
    )

    assert report.status == "PASS"
    assert report.tasks[0]["timing_mode"] == "source_bound_blackout"


def test_source_bound_blackout_requires_suspended_anchor_restore(
    tmp_path: Path,
    monkeypatch,
) -> None:
    plan, root, session = _fixture(tmp_path)
    collection_plan_path = session / "plan.json"
    collection_plan = json.loads(
        collection_plan_path.read_text(encoding="utf-8")
    )
    collection_plan["trace"].update(
        {
            "detach_at_update": 170,
            "reattach_at_update": 950,
            "startup_trace_handoff": True,
            "source_bound_board_anchor": {
                "framework_update": 150,
            },
            "source_bound_board_precall_global_restore": {
                "framework_update": 150,
                "suspend_other_threads_until_board_call": False,
            },
        }
    )
    collection_plan_path.write_text(
        json.dumps(collection_plan, sort_keys=True),
        encoding="utf-8",
    )
    followup = json.loads(plan.read_text(encoding="utf-8"))
    followup["tasks"][0]["session_plan_sha256"] = _digest(
        collection_plan_path
    )
    followup["tasks"][0]["trajectory_end_update"] = 920
    plan.write_text(json.dumps(followup, sort_keys=True), encoding="utf-8")
    monkeypatch.setattr(
        pc_memory_followup.PopCapDemo,
        "read",
        lambda path: _demo(),
    )

    report = pc_memory_followup.verify_pc_memory_followup_plan(
        plan,
        evidence_root=root,
    )

    assert report.status == "FAIL"
    assert "timing order is invalid" in report.reasons[0]


def test_source_bound_post_input_blackout_uses_later_idle_corridor(
    tmp_path: Path,
    monkeypatch,
) -> None:
    plan, root, session = _fixture(tmp_path)
    collection_plan_path = session / "plan.json"
    collection_plan = json.loads(
        collection_plan_path.read_text(encoding="utf-8")
    )
    collection_plan["trace"].update(
        {
            "detach_at_update": 170,
            "reattach_at_update": 950,
            "startup_trace_handoff": True,
            "source_bound_board_anchor": {
                "framework_update": 150,
            },
            "source_bound_board_precall_global_restore": {
                "framework_update": 150,
                "suspend_other_threads_until_board_call": True,
            },
        }
    )
    collection_plan["capture"]["window_repaint_guard"][
        "previous_effectful_command_update"
    ] = 180
    collection_plan_path.write_text(
        json.dumps(collection_plan, sort_keys=True),
        encoding="utf-8",
    )
    followup = json.loads(plan.read_text(encoding="utf-8"))
    followup["tasks"][0]["session_plan_sha256"] = _digest(
        collection_plan_path
    )
    followup["tasks"][0]["trajectory_end_update"] = 920
    plan.write_text(json.dumps(followup, sort_keys=True), encoding="utf-8")
    monkeypatch.setattr(
        pc_memory_followup.PopCapDemo,
        "read",
        lambda path: SimpleNamespace(
            random_seed=123,
            length_updates=1000,
            commands=(
                SimpleNamespace(update=150, kind="mouse_button"),
                SimpleNamespace(update=180, kind="mouse_button"),
                SimpleNamespace(update=500, kind="idle"),
                SimpleNamespace(update=900, kind="mouse_move"),
            ),
        ),
    )

    report = pc_memory_followup.verify_pc_memory_followup_plan(
        plan,
        evidence_root=root,
    )

    assert report.status == "PASS"
    assert (
        report.tasks[0]["timing_mode"]
        == "source_bound_post_input_blackout"
    )


def test_followup_plan_cannot_contain_feature_labels(
    tmp_path: Path,
    monkeypatch,
) -> None:
    plan, root, _ = _fixture(tmp_path)
    value = json.loads(plan.read_text(encoding="utf-8"))
    value["tasks"][0]["features"] = ["projectile_collision"]
    plan.write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setattr(
        pc_memory_followup.PopCapDemo,
        "read",
        lambda path: _demo(),
    )

    report = pc_memory_followup.verify_pc_memory_followup_plan(
        plan,
        evidence_root=root,
    )

    assert report.status == "FAIL"
    assert "fields are invalid" in report.reasons[0]
