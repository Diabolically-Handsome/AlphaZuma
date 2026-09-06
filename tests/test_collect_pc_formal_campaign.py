from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tools import collect_pc_formal_campaign as campaign
from tools.collect_pc_formal_window import FormalWindowCollectionError


NONCE_STEM = "c158000000000000000000000000000"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _fake_attempt(
    *,
    output_root: Path,
    host_pre_output: Path,
    host_post_output: Path,
    error: str | None,
) -> Path:
    attempt_root = output_root / "attempt-01"
    attempt_root.mkdir(parents=True)
    state_root = "sha256:" + "a" * 64
    _write_json(host_pre_output, {"state_root": state_root})
    _write_json(host_post_output, {"state_root": state_root})
    _write_json(
        attempt_root / "external-input-guard.json",
        {
            "status": "PASS",
            "event_counts": {"external": 0},
            "protocol_errors": [],
            "hooks": {
                "keyboard_unhooked": True,
                "mouse_unhooked": True,
            },
        },
    )
    _write_json(attempt_root / "strict-replay.json", {"status": "PASS"})
    (attempt_root / "strict-replay.log").write_text("PASS", encoding="utf-8")
    if error is not None:
        _write_json(
            output_root / "attempts.json",
            [{"attempt": 1, "status": "RETRY", "error": error}],
        )
        raise FormalWindowCollectionError("source_collection_failed")
    _write_json(
        output_root / "attempts.json",
        [{"attempt": 1, "status": "PASS"}],
    )
    _write_json(attempt_root / "trajectory" / "index.json", {"ticks": []})
    probe = attempt_root / "memory-probe.json"
    _write_json(probe, {"status": "PASS"})
    return probe


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    plan = tmp_path / "plan.json"
    prestate = tmp_path / "prestate.json"
    plan.write_text("{}", encoding="utf-8")
    prestate.write_text("{}", encoding="utf-8")
    return plan, prestate


def _run(
    tmp_path: Path,
    *,
    maximum_attempts: int,
) -> Path:
    plan, prestate = _inputs(tmp_path)
    return campaign.collect_formal_campaign(
        plan_path=plan,
        prestate_path=prestate,
        campaign_root=tmp_path / "campaign",
        maximum_attempts=maximum_attempts,
        nonce_stem=NONCE_STEM,
        freeze_update=11200,
        slowdown_update=11180,
        trajectory_end_update=11830,
    )


def test_campaign_retains_retryable_attempt_and_stops_at_first_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_collect(**kwargs: Any) -> Path:
        calls.append(kwargs)
        return _fake_attempt(
            output_root=kwargs["output_root"],
            host_pre_output=kwargs["host_pre_output"],
            host_post_output=kwargs["host_post_output"],
            error=(campaign.RETRYABLE_ERROR if len(calls) == 1 else None),
        )

    monkeypatch.setattr(campaign, "collect_formal_window", fake_collect)
    selected = _run(tmp_path, maximum_attempts=3)

    assert len(calls) == 2
    assert calls[0]["session_nonce"] == NONCE_STEM + "1"
    assert calls[1]["session_nonce"] == NONCE_STEM + "2"
    assert not (tmp_path / "campaign" / "attempt-03").exists()
    assert selected == (
        tmp_path
        / "campaign"
        / "attempt-02"
        / "source"
        / "attempt-01"
        / "memory-probe.json"
    ).resolve()
    receipt = json.loads(
        (tmp_path / "campaign" / "campaign.json").read_text("ascii")
    )
    assert receipt["status"] == "PASS"
    assert receipt["attempt_count"] == 2
    assert receipt["selected_attempt"] == 2
    assert [row["status"] for row in receipt["attempts"]] == [
        "RETRYABLE_TRANSPORT_MISMATCH",
        "PASS",
    ]


def test_campaign_aborts_on_nonretryable_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fake_collect(**kwargs: Any) -> Path:
        nonlocal calls
        calls += 1
        return _fake_attempt(
            output_root=kwargs["output_root"],
            host_pre_output=kwargs["host_pre_output"],
            host_post_output=kwargs["host_post_output"],
            error="different_failure",
        )

    monkeypatch.setattr(campaign, "collect_formal_window", fake_collect)
    with pytest.raises(
        campaign.FormalCampaignError,
        match="non_retryable_attempt_failure",
    ):
        _run(tmp_path, maximum_attempts=3)

    assert calls == 1
    receipt = json.loads(
        (tmp_path / "campaign" / "campaign.json").read_text("ascii")
    )
    assert receipt["status"] == "FAILED_CLOSED"
    assert receipt["attempt_count"] == 1


def test_campaign_exhaustion_is_finite_and_retained(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fake_collect(**kwargs: Any) -> Path:
        nonlocal calls
        calls += 1
        return _fake_attempt(
            output_root=kwargs["output_root"],
            host_pre_output=kwargs["host_pre_output"],
            host_post_output=kwargs["host_post_output"],
            error=campaign.RETRYABLE_ERROR,
        )

    monkeypatch.setattr(campaign, "collect_formal_window", fake_collect)
    with pytest.raises(campaign.FormalCampaignError, match="campaign_exhausted"):
        _run(tmp_path, maximum_attempts=2)

    assert calls == 2
    receipt = json.loads(
        (tmp_path / "campaign" / "campaign.json").read_text("ascii")
    )
    assert receipt["status"] == "FAILED_CLOSED_EXHAUSTED"
    assert receipt["attempt_count"] == 2
    assert all(
        (tmp_path / "campaign" / f"attempt-{index:02d}").is_dir()
        for index in (1, 2)
    )
