"""Tests for immutable, effect-free PC capture campaign planning."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from zuma_rl import pc_capture_campaign


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _campaign_fixture(
    tmp_path: Path,
    *,
    classification: str = "certifying_candidate",
    previous_effectful_update: int = 100,
    source_prestate: bool = False,
) -> tuple[Path, Path]:
    root = tmp_path / "evidence"
    root.mkdir()
    prestate = root / "prestate"
    prestate.mkdir()
    for name in ("users.dat", "user2.dat", "adv_in_game2.sav"):
        (prestate / name).write_bytes(name.encode("ascii"))
    alternate_prestate = root / "alternate-prestate"
    alternate_prestate.mkdir()
    for name in ("users.dat", "user2.dat", "adv_in_game2.sav"):
        (alternate_prestate / name).write_bytes(
            b"alternate-" + name.encode("ascii")
        )
    runtime = root / "runtime.exe"
    runtime.write_bytes(b"runtime")
    dmo = root / "input.dmo"
    dmo.write_bytes(b"dmo")
    exclusion = (
        "focus-deadlocked diagnostic"
        if classification == "diagnostic_only"
        else None
    )
    source = {
        "id": "source",
        "path": "input.dmo",
        "sha256": _digest(dmo),
        "random_seed": 123,
        "length_updates": 1000,
        "classification": classification,
        "exclusion_reason": exclusion,
    }
    if source_prestate:
        source["prestate_path"] = "alternate-prestate"
    campaign = {
        "schema": pc_capture_campaign.CAMPAIGN_SCHEMA,
        "version": pc_capture_campaign.CAMPAIGN_VERSION,
        "policy": "original-transfer-jungle2-v1",
        "prestate_path": "prestate",
        "direct_runtime_path": "runtime.exe",
        "direct_runtime_sha256": _digest(runtime),
        "sources": [source],
        "tasks": [
            {
                "id": "capture",
                "status": "ready_for_collection",
                "candidate_features": [
                    "long_horizon_drift",
                    "shot_release",
                ],
                "source_id": "source",
                "window": {
                    "attach_at_update": 0,
                    "detach_at_update": 120,
                    "repaint_update": 200,
                    "wait_until_update": 210,
                    "reattach_at_update": 800,
                    "previous_effectful_update": previous_effectful_update,
                    "next_effectful_update": 900,
                    "duration_seconds": 5.5,
                    "frame_budget_fps": 240,
                    "minimum_required_ticks": 500,
                },
                "blocker": None,
            },
            {
                "id": "pending",
                "status": "requires_recording",
                "candidate_features": ["natural_loss"],
                "source_id": None,
                "window": None,
                "blocker": "a natural loss DMO is required",
            },
        ],
    }
    path = tmp_path / "campaign.json"
    path.write_text(
        json.dumps(campaign, sort_keys=True),
        encoding="utf-8",
    )
    return path, root


def _fake_demo() -> SimpleNamespace:
    return SimpleNamespace(
        random_seed=123,
        length_updates=1000,
        commands=(
            SimpleNamespace(update=100, kind="mouse_button"),
            SimpleNamespace(update=150, kind="idle"),
            SimpleNamespace(update=700, kind="idle"),
            SimpleNamespace(update=900, kind="mouse_move"),
        ),
    )


def test_valid_campaign_proves_hashes_and_idle_window(
    tmp_path: Path,
    monkeypatch,
) -> None:
    campaign, root = _campaign_fixture(tmp_path)
    monkeypatch.setattr(
        pc_capture_campaign.PopCapDemo,
        "read",
        lambda path: _fake_demo(),
    )

    report = pc_capture_campaign.verify_pc_capture_campaign(
        campaign,
        evidence_root=root,
    )

    assert report.status == "PASS"
    assert report.exit_code == 0
    assert report.summary["ready_for_collection_count"] == 1
    assert report.summary["requires_recording_count"] == 1
    assert report.summary["campaign_complete"] is False
    ready = report.tasks[0]
    assert ready["idle_update_budget"] == 690
    assert ready["collector_prepare_options"]["crt_rand_seed"] == 123
    assert ready["collector_prepare_options"][
        "allow_pre_stream_commands"
    ] is True
    assert ready["collector_prepare_options"]["prestate_dir"] == "prestate"


def test_source_specific_prestate_overrides_campaign_default(
    tmp_path: Path,
    monkeypatch,
) -> None:
    campaign, root = _campaign_fixture(tmp_path, source_prestate=True)
    monkeypatch.setattr(
        pc_capture_campaign.PopCapDemo,
        "read",
        lambda path: _fake_demo(),
    )

    report = pc_capture_campaign.verify_pc_capture_campaign(
        campaign,
        evidence_root=root,
    )

    assert report.status == "PASS"
    assert report.sources[0]["prestate_path"] == "alternate-prestate"
    assert report.tasks[0]["collector_prepare_options"]["prestate_dir"] == (
        "alternate-prestate"
    )


def test_diagnostic_dmo_cannot_back_a_ready_task(
    tmp_path: Path,
    monkeypatch,
) -> None:
    campaign, root = _campaign_fixture(
        tmp_path,
        classification="diagnostic_only",
    )
    monkeypatch.setattr(
        pc_capture_campaign.PopCapDemo,
        "read",
        lambda path: _fake_demo(),
    )

    report = pc_capture_campaign.verify_pc_capture_campaign(
        campaign,
        evidence_root=root,
    )

    assert report.status == "FAIL"
    assert "diagnostic-only DMO" in report.reasons[0]


def test_effectful_anchor_drift_fails_closed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    campaign, root = _campaign_fixture(
        tmp_path,
        previous_effectful_update=99,
    )
    monkeypatch.setattr(
        pc_capture_campaign.PopCapDemo,
        "read",
        lambda path: _fake_demo(),
    )

    report = pc_capture_campaign.verify_pc_capture_campaign(
        campaign,
        evidence_root=root,
    )

    assert report.status == "FAIL"
    assert "anchors differ" in report.reasons[0]


def test_source_hash_tampering_fails_before_planning(
    tmp_path: Path,
    monkeypatch,
) -> None:
    campaign, root = _campaign_fixture(tmp_path)
    (root / "input.dmo").write_bytes(b"tampered")
    monkeypatch.setattr(
        pc_capture_campaign.PopCapDemo,
        "read",
        lambda path: _fake_demo(),
    )

    report = pc_capture_campaign.verify_pc_capture_campaign(
        campaign,
        evidence_root=root,
    )

    assert report.status == "FAIL"
    assert "SHA-256 differs" in report.reasons[0]
