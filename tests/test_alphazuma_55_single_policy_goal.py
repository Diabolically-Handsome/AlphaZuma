from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import zipfile

import pytest

from tools import audit_alphazuma_55_single_policy_goal as goal
from tools import watch_alphazuma_55_single_policy_goal as watcher


def _write(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
    return path.resolve()


def _ref(path: Path) -> dict[str, str]:
    path = path.resolve()
    return {"path": str(path), "sha256": goal._sha256(path)}


def _policy(path: Path) -> dict:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("data", "{}")
        archive.writestr("policy.pth", b"neural-policy")
    return {
        "id": "one-policy",
        "path": str(path.resolve()),
        "sha256": goal._sha256(path),
        "training_steps": 123,
    }


def _eligible_source_receipt(tmp_path: Path) -> dict:
    model = _policy(tmp_path / "policy.zip")
    report_path = tmp_path / "source-final-report.json"
    report = {
        "status": "COMPLETE",
        "selected_single_policy": model,
        "success_gates": {
            "single_policy_capability_gate": {
                "actual": {
                    "levels_cleared": 55,
                    "minimum_wins_per_level": 1,
                    "total_wins": 300,
                    "attempts": 440,
                },
                "passed": True,
            },
            "continuous_full_game_gate": {"passed": True},
        },
        "continuous_campaigns": {
            "campaigns": [
                {
                    "campaign_index": index,
                    "levels_attempted": 55,
                    "wins": 55 if index == 0 else 54,
                    "cleared_all_55": index == 0,
                }
                for index in range(4)
            ],
            "passed": True,
            "campaigns_cleared": 1,
        },
    }
    _write(report_path, report)
    return {
        "schema": "zuma-rl.alphazuma-55-independent-final-audit",
        "version": 1,
        "status": "PASS",
        "final_report": _ref(report_path),
        "selected_single_policy": model,
        "final_blind_attempts": 440,
        "continuous_attempts": 220,
        "single_policy_capability_gate": True,
        "continuous_full_game_gate": True,
    }


def _no_promotion_receipt() -> dict:
    return {
        "schema": "zuma-rl.alphazuma-55-configured-successor-independent-audit",
        "version": 1,
        "status": "PASS",
        "audited_utc": "2026-08-15T00:00:00+00:00",
        "controller_result": "COMPLETE_NO_PROMOTION",
        "formal_seed_consumption": "NONE",
        "promotion_gate": {"passed": False},
    }


def _fixture(tmp_path: Path, *, high_receipt: bool = True) -> tuple[Path, dict]:
    source_master = _write(tmp_path / "source-master.json", {"id": "source"})
    high_master = _write(tmp_path / "high-master.json", {"id": "high"})
    source_input = _write(tmp_path / "source-input.json", {"id": "source-input"})
    high_input = _write(tmp_path / "high-input.json", {"id": "high-input"})
    source_auditor = (tmp_path / "source-auditor.py")
    source_auditor.write_text("# frozen source auditor\n", encoding="utf-8")
    high_auditor = (tmp_path / "high-auditor.py")
    high_auditor.write_text("# frozen configured auditor\n", encoding="utf-8")
    source_receipt = _eligible_source_receipt(tmp_path)
    high_result = _no_promotion_receipt()
    source_receipt_path = _write(tmp_path / "source-receipt.json", source_receipt)
    high_receipt_path = tmp_path / "high-receipt.json"
    if high_receipt:
        _write(high_receipt_path, high_result)

    campaign_rows = [
        {
            "id": "high-priority",
            "kind": "configured_successor",
            "seed_registry_campaign_id": "high-registry",
            "campaign_master": _ref(high_master),
            "audit_input": _ref(high_input),
            "independent_auditor": _ref(high_auditor),
            "independent_receipt": str(high_receipt_path.resolve()),
        },
        {
            "id": "source-fallback",
            "kind": "source",
            "seed_registry_campaign_id": "source-registry",
            "campaign_master": _ref(source_master),
            "audit_input": _ref(source_input),
            "independent_auditor": _ref(source_auditor),
            "independent_receipt": str(source_receipt_path.resolve()),
        },
    ]
    registry_path = tmp_path / "registry.json"
    registry = {
        "schema": "zuma-rl.alphazuma-55-weekend-formal-seed-registry",
        "version": 1,
        "status": "FROZEN_BEFORE_ANY_SUCCESSOR_FORMAL_INFERENCE",
        "campaigns": [
            {
                "id": "high-registry",
                "master": campaign_rows[0]["campaign_master"],
                "ranges": [{"stage": "final", "first": 1000, "last": 1439}],
            },
            {
                "id": "source-registry",
                "master": campaign_rows[1]["campaign_master"],
                "ranges": [{"stage": "final", "first": 2000, "last": 2439}],
            },
        ],
        "global_checks": {
            "ranges_strictly_non_overlapping": True,
            "all_ranges_within_uint32": True,
        },
    }
    _write(registry_path, registry)
    plan = {
        "schema": "zuma-rl.alphazuma-55-single-policy-goal-audit-plan",
        "version": 1,
        "status": "FROZEN_BEFORE_GOAL_COMPLETION",
        "deadline_utc": "2026-08-17T15:00:00Z",
        "requirements": {
            "one_frozen_neural_policy": True,
            "included_levels": 55,
            "final_blind_attempts": 440,
            "minimum_final_blind_total_wins": 220,
            "minimum_win_per_level": 1,
            "continuous_campaigns": 4,
            "continuous_attempts": 220,
            "minimum_complete_continuous_55_level_campaigns": 1,
        },
        "selection": {
            "campaign_priority": [row["id"] for row in campaign_rows],
            "higher_priority_must_be_terminal_before_lower_selection": True,
        },
        "formal_seed_registry": _ref(registry_path),
        "campaigns": campaign_rows,
        "implementation": {"goal_auditor": _ref(goal.SCRIPT_PATH)},
    }
    plan_path = _write(tmp_path / "plan.json", plan)
    recomputers = {
        "configured_successor": {
            "path": high_auditor.resolve(),
            "call": lambda _path: high_result,
        },
        "source": {
            "path": source_auditor.resolve(),
            "call": lambda _path: source_receipt,
        },
    }
    return plan_path, recomputers


def test_no_promotion_pass_cannot_satisfy_goal(tmp_path: Path) -> None:
    plan, recomputers = _fixture(tmp_path)
    result = goal.audit(
        plan,
        now=datetime(2026, 8, 16, tzinfo=timezone.utc),
        recomputers=recomputers,
    )
    assert result["status"] == "PASS"
    assert result["selected_campaign_id"] == "source-fallback"
    assert result["campaigns"][0]["status"] == "TERMINAL_NO_PROMOTION"
    assert result["selected_single_policy"]["neural_archive_verified"] is True


def test_pending_higher_priority_blocks_lower_selection(tmp_path: Path) -> None:
    plan, recomputers = _fixture(tmp_path, high_receipt=False)
    result = goal.audit(
        plan,
        now=datetime(2026, 8, 16, tzinfo=timezone.utc),
        recomputers=recomputers,
    )
    assert result["status"] == "WAITING"
    assert result["selection_blocked_by_pending_higher_priority"] == "high-priority"
    assert result["campaigns"][1]["status"] == "ELIGIBLE"


def test_deadline_releases_missing_higher_priority(tmp_path: Path) -> None:
    plan, recomputers = _fixture(tmp_path, high_receipt=False)
    result = goal.audit(
        plan,
        now=datetime(2026, 8, 17, 15, 0, tzinfo=timezone.utc),
        recomputers=recomputers,
    )
    assert result["status"] == "PASS"
    assert result["campaigns"][0]["status"] == "MISSING_AT_DEADLINE"
    assert result["selected_campaign_id"] == "source-fallback"


def test_tampered_policy_fails_closed(tmp_path: Path) -> None:
    plan, recomputers = _fixture(tmp_path)
    with (tmp_path / "policy.zip").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(goal.GoalAuditError, match="policy hash differs"):
        goal.audit(
            plan,
            now=datetime(2026, 8, 16, tzinfo=timezone.utc),
            recomputers=recomputers,
        )


def test_nonreproducible_campaign_receipt_fails_closed(tmp_path: Path) -> None:
    plan, recomputers = _fixture(tmp_path)
    recomputers["source"]["call"] = lambda _path: {"status": "PASS"}
    with pytest.raises(goal.GoalAuditError, match="receipt cannot be reproduced"):
        goal.audit(
            plan,
            now=datetime(2026, 8, 16, tzinfo=timezone.utc),
            recomputers=recomputers,
        )


def test_restore_evidence_requires_both_original_power_states(tmp_path: Path) -> None:
    gpu = _write(
        tmp_path / "gpu.json",
        {
            "status": "RESTORED_ON_REQUEST",
            "gpu_state": [
                {"index": 0, "power_limit_watts": 600},
                {"index": 1, "power_limit_watts": 400},
            ],
        },
    )
    plan = _write(
        tmp_path / "plan-power.json",
        {
            "status": "RESTORED_ON_REQUEST",
            "active_guid": "381b4222-f694-41f0-9685-ff5bb260df2e",
        },
    )
    evidence = watcher._restore_evidence(gpu, plan, timeout_seconds=0.1)
    assert evidence["verified"] is True
    assert evidence["gpu_limits_watts"] == {"0": 600, "1": 400}


def test_restore_marker_is_idempotent(tmp_path: Path) -> None:
    marker = tmp_path / "restore.txt"
    watcher._write_marker(marker, "first")
    original = marker.read_text(encoding="ascii")
    watcher._write_marker(marker, "second")
    assert marker.read_text(encoding="ascii") == original
    assert "first" in original
