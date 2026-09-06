from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import tools.compare_source_bound_popcap_replay as replay_parity
from tools.compare_source_bound_popcap_replay import (
    EXACT_CALL_FIELDS,
    _first_replay_win_score_call,
    _is_natural_loss_terminal_state,
    _is_natural_win_terminal_state,
    _replay_loss_observation,
    compare_exact_call_suffix,
)


def _call(order: int, *, output: int | None = None) -> dict[str, object]:
    row: dict[str, object] = {
        "order": order,
        "framework_update": 3000 + order,
        "native_game_time": 100 + order,
        "caller": 0x0065B821,
        "output": order if output is None else output,
        "pre_index": order,
        "pre_state_sha256": f"sha256:pre{order}",
        "post_index": order + 1,
        "post_state_sha256": f"sha256:post{order}",
        "thread_crt_rand_state": 1000 + order,
        "thread_crt_snapshot_error": None,
        "score": 9000 + order,
        "score_target": 9650,
    }
    assert all(field in row for field in EXACT_CALL_FIELDS)
    return row


def test_exact_call_suffix_accepts_relative_replay_orders() -> None:
    source = [_call(order) for order in range(5)]
    replay = []
    for replay_order, source_row in enumerate(source[2:]):
        row = dict(source_row)
        row["order"] = replay_order
        replay.append(row)

    assert compare_exact_call_suffix(
        source,
        replay,
        source_start_order=2,
    ) == (3, None)


def test_exact_call_suffix_reports_first_semantic_mismatch() -> None:
    source = [_call(order) for order in range(4)]
    replay = []
    for replay_order, source_row in enumerate(source[1:]):
        row = dict(source_row)
        row["order"] = replay_order
        replay.append(row)
    replay[1]["output"] = 999

    compared, mismatch = compare_exact_call_suffix(
        source,
        replay,
        source_start_order=1,
    )

    assert compared == 1
    assert mismatch is not None
    assert mismatch["source_order"] == 2
    assert mismatch["replay_order"] == 1
    assert mismatch["fields"] == ["output"]


def test_exact_call_suffix_rejects_truncated_replay() -> None:
    source = [_call(order) for order in range(4)]
    replay = [dict(source[2], order=0)]

    compared, mismatch = compare_exact_call_suffix(
        source,
        replay,
        source_start_order=2,
    )

    assert compared == 1
    assert mismatch is not None
    assert mismatch["fields"] == ["missing_replay_suffix"]


def test_source_terminal_state_and_exact_score_call_establish_win() -> None:
    terminal = {
        "score": 9690,
        "score_target": 9650,
        "runtime_active": False,
        "curve_plan_exhausted": True,
        "active_ball_count": 0,
        "pending_ball_count": 0,
        "inserting_ball_count": 0,
        "fired_ball_count": 0,
    }
    assert _is_natural_win_terminal_state(
        terminal,
        minimum_score=9650,
    )
    calls = [_call(0), _call(1), _call(2)]
    calls[0]["score"] = 9640
    calls[1]["score"] = 9660

    assert _first_replay_win_score_call(
        calls,
        source_start_order=307,
        source_suffix_count=3,
        minimum_score=9650,
    ) == {
        "evidence": "exact-source-bound-gameplay-call",
        "replay_order": 1,
        "source_order": 308,
        "framework_update": 3001,
        "native_game_time": 101,
        "score": 9660,
        "score_target": 9650,
    }


def test_source_terminal_state_and_live_counter_establish_loss() -> None:
    terminal = {
        "score": 1200,
        "score_target": 9650,
        "loss_counter": 27,
        "runtime_active": False,
        "active_ball_count": 0,
        "pending_ball_count": 0,
        "inserting_ball_count": 0,
        "fired_ball_count": 0,
    }
    assert _is_natural_loss_terminal_state(terminal)
    observation = {
        "natural_loss": {
            "framework_update": 7021,
            "loss_counter": 1,
            "terminal_board_fallback": False,
        }
    }
    assert _replay_loss_observation(observation) == {
        "evidence": (
            "live-retail-loss-counter-under-exact-source-bound-"
            "gameplay-call-parity"
        ),
        **observation["natural_loss"],
    }


def test_loss_evidence_rejects_fallback_or_nonpositive_counter() -> None:
    assert _replay_loss_observation(
        {
            "natural_loss": {
                "loss_counter": 1,
                "terminal_board_fallback": True,
            }
        }
    ) is None


def test_full_source_bound_parity_accepts_natural_loss(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_calls = [_call(0), _call(1)]
    source_trace = tmp_path / "source-trace.json"
    source_trace.write_text(
        json.dumps({"call_count": 2, "calls": source_calls}),
        encoding="utf-8",
    )
    source_report = tmp_path / "source-report.json"
    source_report.write_text(
        json.dumps(
            {
                "version": 2,
                "expected_outcome": "natural_loss",
                "gameplay_policy": "idle",
                "gameplay": {
                    "status": "PASS",
                    "outcome": "natural_loss",
                    "expected_outcome": "natural_loss",
                    "gameplay_policy": "idle",
                    "action_count": 0,
                    "swap_count": 0,
                    "terminal_observation": {
                        "outcome": "natural_loss",
                        "first_seen_perf_counter_ns": 1,
                        "confirmed_perf_counter_ns": 10_000_000_001,
                    },
                    "final_state": {
                        "score": 1200,
                        "score_target": 9650,
                        "loss_counter": 27,
                        "runtime_active": False,
                        "active_ball_count": 0,
                        "pending_ball_count": 0,
                        "inserting_ball_count": 0,
                        "fired_ball_count": 0,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    source_trace_sha256 = "sha256:" + hashlib.sha256(
        source_trace.read_bytes()
    ).hexdigest()
    source_report_sha256 = "sha256:" + hashlib.sha256(
        source_report.read_bytes()
    ).hexdigest()

    restored_state = 123456789
    restore = {
        "framework_update": 3000,
        "caller": 0x0065B821,
        "bytes_written": 0,
        "changed": False,
        "writeback_verified": True,
        "restored": {
            "source_call_order": 0,
            "source_trace_sha256": source_trace_sha256,
            "source_recording_report_sha256": source_report_sha256,
        },
    }
    crt_restore = {
        **restore,
        "bytes_written": 4,
        "changed": True,
        "restored": {
            "source_call_order": 0,
            "source_sha256": source_trace_sha256,
            "source_recording_report_sha256": source_report_sha256,
            "state": restored_state,
        },
    }
    replay_trace = tmp_path / "replay-trace.json"
    replay_trace.write_text(
        json.dumps(
            {
                "schema": "zuma-rl.pc-gameplay-mtrand-call-trace",
                "version": 1,
                "status": "PASS",
                "failure": None,
                "stop_reason": "process_exit",
                "exited": True,
                "exit_code": 0,
                "hardware_breakpoint_restored": True,
                "hardware_breakpoint_restore_error": None,
                "debugger_detach_error": None,
                "persistent_file_modified": False,
                "process_memory_writes": 4,
                "call_count": 2,
                "calls": source_calls,
                "global_mtrand_call_site_restore": restore,
                "thread_crt_call_site_restore": crt_restore,
            }
        ),
        encoding="utf-8",
    )

    aligned_dmo = tmp_path / "aligned.dmo"
    aligned_dmo.write_bytes(b"aligned")
    raw_dmo = tmp_path / "raw.dmo"
    raw_dmo.write_bytes(b"raw")
    strict_result = tmp_path / "strict.json"
    strict_result.write_text("{}", encoding="ascii")
    strict_receipt = {
        "sha256": "sha256:strict",
        "command_order_offset": 0,
    }
    full_replay = tmp_path / "full-replay.json"
    full_replay.write_text(
        json.dumps(
            {
                "schema": "zuma-rl.existing-popcap-full-replay-monitor",
                "version": 3,
                "status": "PASS",
                "failures": [],
                "exit_observed": True,
                "exit_code": 0,
                "timed_out": False,
                "process_memory_writes": 0,
                "persistent_file_modified": False,
                "minimum_win_score": 0,
                "expected_outcome": "natural_loss",
                "natural_outcome_evidence_deferred": True,
                "natural_outcome_evidence_mode": (
                    "source-bound-exact-gameplay-call-parity"
                ),
                "terminal_board_fallback_count": 0,
                "natural_loss": {
                    "framework_update": 7000,
                    "loss_counter": 1,
                    "terminal_board_fallback": False,
                },
                "dmo": {
                    "path": str(aligned_dmo),
                    "length_updates": 100,
                    "expected_final_bit_position": 1000,
                },
                "strict_trace_receipt": strict_receipt,
                "command_order_offset": 0,
                "max_framework_update": 100,
                "max_buffer_read_bit_position": 1000,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        replay_parity,
        "load_source_bound_thread_crt_restore_state",
        lambda *args, **kwargs: SimpleNamespace(
            state=restored_state,
            source_dmo_path=raw_dmo.resolve(),
            source_dmo_sha256="sha256:" + hashlib.sha256(b"raw").hexdigest(),
        ),
    )
    monkeypatch.setattr(
        replay_parity,
        "_load_strict_trace_receipt",
        lambda *args, **kwargs: strict_receipt,
    )

    result = replay_parity.compare_source_bound_replay(
        source_trace_path=source_trace,
        source_recording_report_path=source_report,
        replay_trace_path=replay_trace,
        strict_trace_result_path=strict_result,
        full_replay_result_path=full_replay,
        source_start_order=0,
        source_framework_update=3000,
        source_caller=0x0065B821,
    )

    assert result["status"] == "PASS"
    assert result["failures"] == []
    assert result["source"]["outcome"] == "natural_loss"
    assert result["replay"]["natural_loss"]["loss_counter"] == 1
    assert _replay_loss_observation(
        {
            "natural_loss": {
                "loss_counter": 0,
                "terminal_board_fallback": False,
            }
        }
    ) is None
