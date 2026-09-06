from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import zipfile

import pytest

from tools import audit_alphazuma_55_single_policy_goal_v5 as goal


def _write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
    return path.resolve()


def _ref(path: Path) -> dict[str, str]:
    path = path.resolve()
    return {"path": str(path), "sha256": goal._sha256(path)}


def _policy(path: Path) -> dict:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("data", "{}")
        archive.writestr("policy.pth", b"motor-policy")
    return {
        "id": "motor-policy",
        "path": str(path.resolve()),
        "sha256": goal._sha256(path),
        "observation_profile": "motor-observable-v1",
        "promotable": True,
    }


def _campaign_fixture(tmp_path: Path, result: str) -> tuple[dict, dict]:
    master = _write(tmp_path / "master.json", {"id": "motor"})
    auditor = tmp_path / "auditor.py"
    auditor.write_text("# frozen auditor\n", encoding="utf-8")
    receipt_path = tmp_path / "receipt.json"
    if result == "COMPLETE_NO_PROMOTION":
        receipt = {
            "status": "PASS",
            "audited_utc": "2026-08-16T00:00:00+00:00",
            "controller_result": result,
            "engineering_gate_passed": False,
            "single_policy_capability_gate": False,
            "continuous_full_game_gate": False,
            "goal_gates_passed": False,
            "formal_seed_consumption": "NONE",
        }
    else:
        model = _policy(tmp_path / "policy.zip")
        capability_pass = result == "COMPLETE_FORMAL_PASS"
        capability = {
            "levels_cleared": 55 if capability_pass else 54,
            "minimum_wins_per_level": 1 if capability_pass else 0,
            "total_wins": 300 if capability_pass else 219,
            "attempts": 440,
        }
        campaigns = {
            "campaigns": [
                {
                    "campaign_index": index,
                    "cleared_all_55": capability_pass and index == 0,
                }
                for index in range(4)
            ],
            "campaigns_cleared": 1 if capability_pass else 0,
            "passed": capability_pass,
        }
        report_path = _write(
            tmp_path / "report.json",
            {
                "status": result,
                "selected_single_policy": model,
                "success_gates": {
                    "single_policy_capability_gate": {
                        "actual": capability,
                        "passed": capability_pass,
                    },
                    "continuous_full_game_gate": {"passed": capability_pass},
                },
                "continuous_campaigns": campaigns,
            },
        )
        receipt = {
            "status": "PASS",
            "audited_utc": "2026-08-16T00:00:00+00:00",
            "controller_result": result,
            "engineering_gate_passed": True,
            "single_policy_capability_gate": capability_pass,
            "continuous_full_game_gate": capability_pass,
            "goal_gates_passed": capability_pass,
            "selected_single_policy": model,
            "recomputed": {
                "single_policy_capability": capability,
                "continuous_campaigns": campaigns,
            },
            "formal_seed_consumption": "FORMAL_RANGES_CONSUMED",
            "artifacts": {"controller_report": _ref(report_path)},
        }
    _write(receipt_path, receipt)
    campaign = {
        "id": "motor",
        "kind": goal.MOTOR_KIND,
        "campaign_master": _ref(master),
        "audit_input": _ref(master),
        "independent_auditor": _ref(auditor),
        "independent_receipt": str(receipt_path.resolve()),
    }
    recomputers = {
        goal.MOTOR_KIND: {
            "path": auditor.resolve(),
            "call": lambda _path: receipt,
        }
    }
    return campaign, recomputers


@pytest.mark.parametrize(
    ("controller_result", "expected"),
    [
        ("COMPLETE_NO_PROMOTION", "TERMINAL_NO_PROMOTION"),
        ("COMPLETE_FORMAL_FAIL", "TERMINAL_GATE_FAIL"),
        ("COMPLETE_FORMAL_PASS", "ELIGIBLE"),
    ],
)
def test_motor_campaign_normalization(
    tmp_path: Path, controller_result: str, expected: str
) -> None:
    campaign, recomputers = _campaign_fixture(tmp_path, controller_result)
    result = goal._motor_campaign_result(
        campaign,
        deadline=datetime(2026, 8, 17, 15, tzinfo=timezone.utc),
        now=datetime(2026, 8, 16, tzinfo=timezone.utc),
        recomputers=recomputers,
    )
    assert result["status"] == expected
    assert result["eligible"] is (expected == "ELIGIBLE")


def test_motor_receipt_must_reproduce(tmp_path: Path) -> None:
    campaign, recomputers = _campaign_fixture(tmp_path, "COMPLETE_NO_PROMOTION")
    recomputers[goal.MOTOR_KIND]["call"] = lambda _path: {"status": "PASS"}
    with pytest.raises(goal.GoalAuditError, match="receipt cannot be reproduced"):
        goal._motor_campaign_result(
            campaign,
            deadline=datetime(2026, 8, 17, 15, tzinfo=timezone.utc),
            now=datetime(2026, 8, 16, tzinfo=timezone.utc),
            recomputers=recomputers,
        )


def test_completed_controller_reconstructs_read_only_audit_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller = tmp_path / "controller"
    controller.mkdir()
    report_path = _write(controller / "final_report.json", {"status": "COMPLETE_NO_PROMOTION"})
    receipt_path = _write(tmp_path / "receipt.json", {"status": "PASS"})
    status_path = controller / "controller_status.json"
    _write(
        status_path,
        {
            "status": "COMPLETE_NO_PROMOTION",
            "phase": "COMPLETE_NO_PROMOTION",
            "final_report": _ref(report_path),
            "independent_audit": _ref(receipt_path),
        },
    )
    master_path = _write(
        tmp_path / "master.json",
        {
            "outputs": {
                "controller": str(controller.resolve()),
                "independent_audit_receipt": str(receipt_path.resolve()),
            }
        },
    )

    def fake_audit(**kwargs):
        boundary = goal._read(Path(kwargs["controller_status"]))
        assert boundary["status"] == "RUNNING"
        assert boundary["phase"] == "RUNNING_INDEPENDENT_AUDIT"
        assert Path(kwargs["controller_status"]) != status_path
        assert goal._read(status_path)["status"] == "COMPLETE_NO_PROMOTION"
        return {"status": "PASS"}

    monkeypatch.setattr(goal.motor, "audit", fake_audit)
    assert goal._recompute_motor_receipt(master_path) == {"status": "PASS"}
    assert goal._read(status_path)["status"] == "COMPLETE_NO_PROMOTION"
