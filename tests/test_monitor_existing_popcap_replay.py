from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from tools.monitor_existing_popcap_replay import (
    _evaluation_failures,
    _is_natural_loss,
    _is_natural_win,
    _load_strict_trace_receipt,
    _snapshot_spec,
    _terminal_board_fallback_is_armed,
)


def test_snapshot_spec_is_strict() -> None:
    assert _snapshot_spec("5380:score_continue") == (
        5380,
        "score_continue",
    )
    with pytest.raises(argparse.ArgumentTypeError):
        _snapshot_spec("-1:stage")
    with pytest.raises(argparse.ArgumentTypeError):
        _snapshot_spec("12:Bad Stage")


def test_full_replay_evaluation_passes_complete_run() -> None:
    assert (
        _evaluation_failures(
            exit_code=0,
            timed_out=False,
            max_update=7119,
            expected_update=7119,
            max_command_order=766,
            expected_command_order=766,
            max_bit_position=170490,
            expected_bit_position=170490,
            natural_win={"score": 9780},
            snapshots=({"error": None},) * 6,
            expected_snapshot_count=6,
        )
        == []
    )


def test_full_replay_evaluation_applies_strict_command_order_offset() -> None:
    assert _evaluation_failures(
        exit_code=0,
        timed_out=False,
        max_update=7119,
        expected_update=7119,
        max_command_order=754,
        expected_command_order=766,
        command_order_offset=12,
        max_bit_position=170490,
        expected_bit_position=170490,
        natural_win={"score": 9780},
        snapshots=(),
        expected_snapshot_count=0,
    ) == []


def test_full_replay_evaluation_accepts_observed_natural_loss() -> None:
    assert _evaluation_failures(
        exit_code=0,
        timed_out=False,
        max_update=7119,
        expected_update=7119,
        max_command_order=766,
        expected_command_order=766,
        max_bit_position=170490,
        expected_bit_position=170490,
        natural_win=None,
        natural_loss={"loss_counter": 1},
        expected_outcome="natural_loss",
        snapshots=(),
        expected_snapshot_count=0,
    ) == []


def test_full_replay_evaluation_rejects_missing_natural_loss() -> None:
    failures = _evaluation_failures(
        exit_code=0,
        timed_out=False,
        max_update=7119,
        expected_update=7119,
        max_command_order=766,
        expected_command_order=766,
        max_bit_position=170490,
        expected_bit_position=170490,
        natural_win={"score": 9780},
        natural_loss=None,
        expected_outcome="natural_loss",
        snapshots=(),
        expected_snapshot_count=0,
    )
    assert failures == ["natural_loss_not_observed"]


def test_exact_eof_bit_position_substitutes_for_fleeting_final_order() -> None:
    assert _evaluation_failures(
        exit_code=0,
        timed_out=False,
        max_update=7119,
        expected_update=7119,
        max_command_order=750,
        expected_command_order=766,
        command_order_offset=12,
        max_bit_position=170490,
        expected_bit_position=170490,
        natural_win=None,
        natural_win_required=False,
        snapshots=(),
        expected_snapshot_count=0,
    ) == []


def test_natural_win_accepts_terminal_board_without_shooter_fields() -> None:
    assert _is_natural_win(
        {
            "score": 9690,
            "score_target": 9650,
            "active_ball_count": 0,
            "pending_ball_count": 0,
            "inserting_ball_count": 0,
            "fired_ball_count": 0,
            "runtime_active": False,
        },
        minimum_score=9650,
    )


def test_natural_loss_requires_positive_retail_loss_counter() -> None:
    assert _is_natural_loss({"loss_counter": 1})
    assert not _is_natural_loss({"loss_counter": 0})
    assert not _is_natural_loss({"loss_counter": True})
    assert not _is_natural_loss({})


def test_terminal_board_fallback_is_armed_only_after_win_score() -> None:
    assert not _terminal_board_fallback_is_armed(
        max_score=None,
        minimum_win_score=9650,
    )
    assert not _terminal_board_fallback_is_armed(
        max_score=9640,
        minimum_win_score=9650,
    )
    assert _terminal_board_fallback_is_armed(
        max_score=9650,
        minimum_win_score=9650,
    )


def test_strict_trace_receipt_binds_dmo_and_offset(tmp_path: Path) -> None:
    dmo = tmp_path / "input.dmo"
    dmo.write_bytes(b"immutable-dmo")
    digest = hashlib.sha256(dmo.read_bytes()).hexdigest()
    report = tmp_path / "result.json"
    report.write_text(
        json.dumps(
            {
                "schema": "zuma.popcap_strict_replay.v3",
                "source_dmo": {
                    "path": str(dmo),
                    "sha256": digest,
                },
                "result": {
                    "failure_count": 0,
                    "stopped_at_update": True,
                    "command_order_offset": 2,
                    "command_order_rebase_rows": [4, 7],
                },
            }
        ),
        encoding="utf-8",
    )

    receipt = _load_strict_trace_receipt(report, dmo)

    assert receipt["command_order_offset"] == 2
    assert receipt["command_order_rebase_rows"] == [4, 7]
    assert receipt["source_dmo_sha256"] == "sha256:" + digest

    value = json.loads(report.read_text(encoding="utf-8"))
    value["result"]["command_order_rebase_rows"] = [4]
    report.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="command-order receipt"):
        _load_strict_trace_receipt(report, dmo)


def test_strict_trace_receipt_accepts_preregistered_candidate_envelope(
    tmp_path: Path,
) -> None:
    dmo = tmp_path / "input.dmo"
    dmo.write_bytes(b"immutable-dmo")
    digest = hashlib.sha256(dmo.read_bytes()).hexdigest()
    report = tmp_path / "result.json"
    value = {
        "schema": "zuma.popcap_strict_replay.v3",
        "source_dmo": {
            "path": str(dmo),
            "sha256": digest,
        },
        "options": {
            "allow_blackout_file_write_order_rebase": True,
            "blackout_expected_command_order_offset": 4,
            "blackout_expected_native_timeline_offset": 2,
        },
        "result": {
            "failure_count": 0,
            "stopped_at_update": True,
            "command_order_offset": 4,
            "command_order_rebase_rows": [],
            "blackout_file_write_rebase_candidate_rows": [4, 7, 9, 12, 15],
            "blackout_native_timeline_offset": 2,
        },
    }
    report.write_text(json.dumps(value), encoding="utf-8")

    receipt = _load_strict_trace_receipt(report, dmo)

    assert receipt["command_order_offset"] == 4
    assert receipt["command_order_rebase_rows"] == []
    assert receipt["blackout_file_write_rebase_candidate_rows"] == [
        4,
        7,
        9,
        12,
        15,
    ]
    assert receipt["blackout_native_timeline_offset"] == 2
    assert receipt["blackout_file_write_rebase_mode"] == (
        "candidate_envelope"
    )

    value["options"]["blackout_expected_command_order_offset"] = 3
    report.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="command-order receipt"):
        _load_strict_trace_receipt(report, dmo)


def test_strict_trace_receipt_accepts_preregistered_offset_set_zero_member(
    tmp_path: Path,
) -> None:
    dmo = tmp_path / "input.dmo"
    dmo.write_bytes(b"immutable-dmo")
    digest = hashlib.sha256(dmo.read_bytes()).hexdigest()
    report = tmp_path / "result.json"
    allowed_pairs = [
        {
            "command_order_offset": 0,
            "native_timeline_offset": 0,
        },
        {
            "command_order_offset": 4,
            "native_timeline_offset": 2,
        },
    ]
    value = {
        "schema": "zuma.popcap_strict_replay.v3",
        "source_dmo": {
            "path": str(dmo),
            "sha256": digest,
        },
        "options": {
            "allow_blackout_file_write_order_rebase": True,
            "blackout_allowed_offset_pairs": allowed_pairs,
        },
        "result": {
            "failure_count": 0,
            "stopped_at_update": True,
            "command_order_offset": 0,
            "command_order_rebase_rows": [],
            "blackout_file_write_rebase_candidate_rows": [4, 7],
            "blackout_native_timeline_offset": 0,
        },
    }
    report.write_text(json.dumps(value), encoding="utf-8")

    receipt = _load_strict_trace_receipt(report, dmo)

    assert receipt["blackout_file_write_rebase_mode"] == (
        "candidate_envelope_set"
    )
    assert receipt["blackout_allowed_offset_pairs"] == allowed_pairs
    assert receipt["blackout_selected_offset_pair"] == {
        "command_order_offset": 0,
        "native_timeline_offset": 0,
    }

    value["result"]["blackout_native_timeline_offset"] = 2
    report.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="command-order receipt"):
        _load_strict_trace_receipt(report, dmo)


def test_full_replay_evaluation_rejects_partial_run() -> None:
    failures = _evaluation_failures(
        exit_code=None,
        timed_out=True,
        max_update=7000,
        expected_update=7119,
        max_command_order=740,
        expected_command_order=766,
        max_bit_position=160000,
        expected_bit_position=170490,
        natural_win=None,
        snapshots=({"error": "capture failed"},),
        expected_snapshot_count=6,
    )
    assert set(failures) == {
        "runtime_exit_timeout",
        "runtime_exit_code",
        "demo_length_not_reached",
        "final_command_not_observed",
        "final_bit_position_not_observed",
        "natural_win_not_observed",
        "ui_snapshot_count",
        "ui_snapshot_failure",
    }
