"""Focused anti-selection tests for formal full-state source receipts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from zuma_rl.pc_exact_step_evidence import canonical_json_bytes
from zuma_rl.pc_full_state_evidence import (
    PcFullStateEvidenceError,
    _validate_attempts,
    _validate_optional_fruit_lifecycle_trigger,
)


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_full_state_reader_rejects_retry_selected_after_score_mismatch(
    tmp_path: Path,
) -> None:
    probe = tmp_path / "memory-probe.json"
    probe.write_bytes(b"{}\n")
    attempts = tmp_path / "attempts.json"
    attempts.write_bytes(
        canonical_json_bytes(
            [
                {
                    "attempt": 1,
                    "status": "RETRY",
                    "process_id": 101,
                    "started_perf_counter_ns": 10,
                    "finished_perf_counter_ns": 20,
                    "error_type": "ProbeError",
                    "error": (
                        "discovered_score_not_stable:"
                        "score=7990,displayed=7950"
                    ),
                },
                {
                    "attempt": 2,
                    "status": "PASS",
                    "process_id": 202,
                    "process_creation_filetime_100ns": 303,
                    "started_perf_counter_ns": 30,
                    "finished_perf_counter_ns": 40,
                    "probe_sha256": _digest(probe),
                },
            ]
        )
    )

    with pytest.raises(
        PcFullStateEvidenceError,
        match="full_state_attempt_retry_selection_forbidden",
    ):
        _validate_attempts(
            attempts,
            selected_attempt=2,
            maximum_startup_attempts=3,
            expected_process_id=202,
            expected_process_creation_filetime_100ns=303,
            probe_path=probe,
        )


def test_full_state_reader_accepts_first_attempt_receipt(tmp_path: Path) -> None:
    probe = tmp_path / "memory-probe.json"
    probe.write_bytes(b"{}\n")
    attempts = tmp_path / "attempts.json"
    attempts.write_bytes(
        canonical_json_bytes(
            [
                {
                    "attempt": 1,
                    "status": "PASS",
                    "process_id": 202,
                    "process_creation_filetime_100ns": 303,
                    "started_perf_counter_ns": 10,
                    "finished_perf_counter_ns": 20,
                    "probe_sha256": _digest(probe),
                }
            ]
        )
    )

    rows, selected = _validate_attempts(
        attempts,
        selected_attempt=1,
        maximum_startup_attempts=3,
        expected_process_id=202,
        expected_process_creation_filetime_100ns=303,
        probe_path=probe,
    )

    assert len(rows) == 1
    assert selected["status"] == "PASS"


def _fruit_observation(
    update: int,
    *,
    active: bool,
    native_time: int,
    remaining: int | None,
) -> dict[str, object]:
    pointer = 123 if active else 0
    expiry = native_time + remaining if remaining is not None else 50
    return {
        "framework_update": update,
        "native_game_time": native_time,
        "board_update_count": update + 10,
        "active": active,
        "active_point_pointer": pointer,
        "selected_point_index": 1 if active else 0,
        "collecting": False,
        "expiry_time": expiry,
        "remaining_ticks": remaining,
    }


def _fruit_trigger_probe(tmp_path: Path) -> dict[str, object]:
    config = {
        "monitor_start_update": 100,
        "maximum_framework_update": 1000,
        "remaining_threshold_ticks": 384,
        "slowdown_lead_updates": 16,
        "freeze_lead_updates": 64,
        "trajectory_tick_count": 512,
        "poll_interval_seconds": 0.002,
    }
    observations = [
        _fruit_observation(100, active=False, native_time=500, remaining=None),
        _fruit_observation(200, active=True, native_time=600, remaining=400),
        _fruit_observation(216, active=True, native_time=616, remaining=384),
    ]
    frozen = _fruit_observation(
        280,
        active=True,
        native_time=680,
        remaining=320,
    )
    transcript = {
        "schema": "zuma-rl.pc-formal-fruit-lifecycle-trigger-transcript",
        "version": 1,
        "status": "PASS",
        "selection_rule": (
            "first_observed_active_fruit_lifecycle_at_or_below_"
            "remaining_threshold_no_skips"
        ),
        "process_id": 7,
        "config": config,
        "observations": observations,
        "selection": {
            "first_active_observation_index": 1,
            "threshold_observation_index": 2,
            "threshold_observation": observations[-1],
            "slowdown_update": 232,
            "freeze_update": 280,
            "trajectory_end_update": 791,
            "trajectory_tick_count": 512,
            "frozen_observation": frozen,
        },
    }
    transcript_path = tmp_path / "trigger.json"
    transcript_path.write_bytes(canonical_json_bytes(transcript))
    return {
        "formal_fruit_lifecycle_trigger": {
            "schema": "zuma-rl.pc-formal-fruit-lifecycle-trigger-binding",
            "version": 1,
            "status": "PASS",
            "artifact": transcript_path.name,
            "artifact_sha256": _digest(transcript_path),
            "selection_rule": transcript["selection_rule"],
            "monitor_start_update": 100,
            "threshold_observation_update": 216,
            "freeze_update": 280,
            "trajectory_end_update": 791,
            "trajectory_tick_count": 512,
            "process_id": 7,
            "read_only_observation": True,
            "gameplay_or_rng_process_memory_write_count": 0,
        }
    }


def test_full_state_reader_accepts_first_fruit_lifecycle_trigger(
    tmp_path: Path,
) -> None:
    probe = _fruit_trigger_probe(tmp_path)

    _validate_optional_fruit_lifecycle_trigger(
        probe,
        probe_path=tmp_path / "memory-probe.json",
        expected_process_id=7,
        expected_freeze_update=280,
        expected_end_update=791,
    )


def test_full_state_reader_rejects_skipped_earlier_fruit_threshold(
    tmp_path: Path,
) -> None:
    probe = _fruit_trigger_probe(tmp_path)
    transcript_path = tmp_path / "trigger.json"
    transcript = json.loads(transcript_path.read_text())
    transcript["observations"][1]["remaining_ticks"] = 383
    transcript["observations"][1]["expiry_time"] = 983
    transcript_path.write_bytes(canonical_json_bytes(transcript))
    binding = probe["formal_fruit_lifecycle_trigger"]
    assert isinstance(binding, dict)
    binding["artifact_sha256"] = _digest(transcript_path)

    with pytest.raises(
        PcFullStateEvidenceError,
        match="full_state_fruit_trigger_no_skip_rule_invalid",
    ):
        _validate_optional_fruit_lifecycle_trigger(
            probe,
            probe_path=tmp_path / "memory-probe.json",
            expected_process_id=7,
            expected_freeze_update=280,
            expected_end_update=791,
        )
