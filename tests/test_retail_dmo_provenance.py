"""Tests for the sole certifying retail DMO transport normalization."""

from __future__ import annotations

import hashlib
import json
import struct

import pytest

from zuma_rl.popcap_dmo import DEMO_FILE_ID, PopCapDemo
from zuma_rl.retail_dmo_provenance import (
    PROVENANCE_CLASSIFICATION,
    RetailDmoProvenanceError,
    build_certifying_provenance,
    canonical_provenance_bytes,
    normalize_duplicate_startup_read,
    validate_certifying_provenance,
)


_RUNTIME_SHA256 = "sha256:" + "2" * 64
_PRESTATE_ROOT = "sha256:" + "4" * 64


class _BitWriter:
    def __init__(self) -> None:
        self.data = bytearray()
        self.bit_position = 0

    def bits(self, value: int, count: int) -> None:
        masked = value & ((1 << count) - 1)
        for index in range(count):
            if self.bit_position % 8 == 0:
                self.data.append(0)
            if masked & (1 << index):
                self.data[self.bit_position // 8] |= (
                    1 << (self.bit_position % 8)
                )
            self.bit_position += 1

    def byte(self, value: int) -> None:
        self.bits(value, 8)

    def i32(self, value: int) -> None:
        for shift in (0, 8, 16, 24):
            self.byte((value >> shift) & 0xFF)


def _command(writer: _BitWriter, *, delta: int, number: int) -> None:
    writer.bits(delta, 4)
    writer.bits(0, 1)
    writer.bits(number, 5)


def _registry_dword_one(writer: _BitWriter) -> None:
    _command(writer, delta=0, number=11)
    writer.bits(1, 1)
    writer.i32(4)
    writer.i32(4)
    for byte in struct.pack("<I", 1):
        writer.byte(byte)


def _raw_recording_dmo(*, duplicate_value: int = 1) -> bytes:
    writer = _BitWriter()
    if duplicate_value == 1:
        _registry_dword_one(writer)
        _registry_dword_one(writer)
    else:
        for _ in range(2):
            _command(writer, delta=0, number=11)
            writer.bits(1, 1)
            writer.i32(4)
            writer.i32(4)
            for byte in struct.pack("<I", duplicate_value):
                writer.byte(byte)
    _command(writer, delta=0, number=18)
    writer.i32(1)
    writer.byte(0xA5)
    _command(writer, delta=1, number=31)

    product = b"1.0.4.9496"
    empty_markers = struct.pack("<i", 0)
    return b"".join(
        (
            struct.pack(
                "<IIIH",
                DEMO_FILE_ID,
                2,
                0x12345678,
                len(product),
            ),
            product,
            struct.pack("<I", len(empty_markers)),
            empty_markers,
            struct.pack("<I", 20),
            bytes(writer.data),
        )
    )


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _recording_report(source: bytes) -> dict[str, object]:
    return {
        "schema": "zuma-rl.retail-autoplay-recording",
        "version": 1,
        "status": "PASS",
        "classification": "candidate-recording-not-pc-golden",
        "session_nonce": "recording-session-001",
        "dmo_bytes": len(source),
        "dmo_sha256": _sha256(source),
        "runtime_executable_sha256": _RUNTIME_SHA256,
        "process_memory_writes": 0,
        "normal_exit": True,
        "process_exit_code": 0,
        "exit_method": "retail_ui",
        "input_transport": "Win32 SendInput",
        "host_restored": True,
        "host_pre_state_root": "sha256:" + "a" * 64,
        "host_restored_state_root": "sha256:" + "a" * 64,
        "applied_pre_state_root": _PRESTATE_ROOT,
        "gameplay": {
            "status": "PASS",
            "outcome": "natural_win",
        },
    }


def _outcome_recording_report(
    source: bytes,
    *,
    outcome: str = "natural_loss",
    loss_receipt: str = "stable",
) -> dict[str, object]:
    policy = "idle" if outcome == "natural_loss" else "autoplay"
    target_write_trace_summary = None
    if outcome == "natural_loss" and loss_receipt == "stable":
        terminal_state = {
            "native_game_time": 357,
            "score": 8000,
            "score_target": 9650,
            "loss_counter": -357,
            "runtime_active": False,
            "curve_plan_exhausted": True,
            "active_ball_count": 0,
            "inserting_ball_count": 0,
            "pending_ball_count": 1,
            "fired_ball_count": 0,
        }
        terminal_observation = {
            "kind": "natural_loss_stable_terminal_v1",
            "outcome": "natural_loss",
            "first_seen_perf_counter_ns": 1,
            "confirmed_perf_counter_ns": 10_000_000_001,
            "snapshot_count": 5,
            "state": {
                key: terminal_state[key]
                for key in (
                    "native_game_time",
                    "score",
                    "score_target",
                    "runtime_active",
                    "curve_plan_exhausted",
                    "active_ball_count",
                    "inserting_ball_count",
                    "pending_ball_count",
                    "fired_ball_count",
                )
            },
        }
    elif outcome == "natural_loss" and loss_receipt == "restart":
        before_restart = {
            "native_game_time": 5205,
            "score": 8000,
            "score_target": 9650,
            "loss_counter": 1,
            "runtime_active": False,
            "curve_plan_exhausted": True,
            "active_ball_count": 0,
            "inserting_ball_count": 0,
            "pending_ball_count": 0,
            "fired_ball_count": 0,
        }
        terminal_state = {
            "native_game_time": 119,
            "score": 8000,
            "score_target": 9650,
            "loss_counter": -119,
            "runtime_active": False,
            "curve_plan_exhausted": False,
            "active_ball_count": 1,
            "inserting_ball_count": 0,
            "pending_ball_count": 9,
            "fired_ball_count": 0,
        }
        terminal_observation = {
            "kind": "natural_loss_restart_transition_v1",
            "outcome": "natural_loss",
            "first_gap_perf_counter_ns": 1,
            "confirmed_perf_counter_ns": 2,
            "board_unavailable_count": 17,
            "native_game_time_reset": 5086,
            "before_restart": before_restart,
            "after_restart": dict(terminal_state),
        }
    elif outcome == "natural_loss" and loss_receipt == "board_replacement":
        before_restart = {
            "native_game_time": 5205,
            "score": 8000,
            "score_target": 9650,
            "loss_counter": 1,
            "runtime_active": False,
            "curve_plan_exhausted": True,
            "active_ball_count": 0,
            "inserting_ball_count": 0,
            "pending_ball_count": 0,
            "fired_ball_count": 0,
        }
        first_post_reset = {
            "native_game_time": 51,
            "score": 8000,
            "score_target": 9650,
            "loss_counter": -51,
            "runtime_active": False,
            "curve_plan_exhausted": True,
            "active_ball_count": 96,
            "inserting_ball_count": 0,
            "pending_ball_count": 1,
            "fired_ball_count": 0,
        }
        zero_target_teardown = {
            "native_game_time": 0,
            "score": 8000,
            "score_target": 0,
            "loss_counter": 0,
            "runtime_active": False,
            "curve_plan_exhausted": True,
            "active_ball_count": 0,
            "inserting_ball_count": 0,
            "pending_ball_count": 4,
            "fired_ball_count": 0,
        }
        terminal_state = {
            "native_game_time": 0,
            "score": 8000,
            "score_target": 9616,
            "loss_counter": 0,
            "runtime_active": False,
            "curve_plan_exhausted": False,
            "active_ball_count": 0,
            "inserting_ball_count": 0,
            "pending_ball_count": 10,
            "fired_ball_count": 0,
        }
        terminal_observation = {
            "kind": "natural_loss_board_replacement_transition_v2",
            "outcome": "natural_loss",
            "observation_mode": "read_live_board(require_shooter=false)",
            "shooter_required": False,
            "first_gap_perf_counter_ns": 1,
            "clock_reset_perf_counter_ns": 2,
            "zero_target_teardown_perf_counter_ns": 3,
            "confirmed_perf_counter_ns": 4,
            "board_unavailable_count": 21,
            "clock_reset_unavailable_count": 17,
            "native_game_time_reset": 5205,
            "before_restart": before_restart,
            "first_post_reset_state": first_post_reset,
            "zero_target_teardown_state": zero_target_teardown,
            "after_restart": dict(terminal_state),
        }
    elif outcome == "natural_loss" and loss_receipt == "board_identity":
        before_restart = {
            "board_address": 0x100000,
            "native_game_time": 5205,
            "score": 8000,
            "score_target": 9650,
            "loss_counter": 1,
            "runtime_active": False,
            "curve_plan_exhausted": True,
            "active_ball_count": 0,
            "inserting_ball_count": 0,
            "pending_ball_count": 0,
            "fired_ball_count": 0,
        }
        first_post_reset = {
            "board_address": 0x100000,
            "native_game_time": 51,
            "score": 8000,
            "score_target": 9650,
            "loss_counter": -51,
            "runtime_active": False,
            "curve_plan_exhausted": True,
            "active_ball_count": 96,
            "inserting_ball_count": 0,
            "pending_ball_count": 1,
            "fired_ball_count": 0,
        }
        first_replacement = {
            "board_address": 0x200000,
            "native_game_time": 0,
            "score": 8000,
            "score_target": 9616,
            "loss_counter": 0,
            "runtime_active": False,
            "curve_plan_exhausted": False,
            "active_ball_count": 0,
            "inserting_ball_count": 0,
            "pending_ball_count": 10,
            "fired_ball_count": 0,
        }
        terminal_state = dict(first_replacement)
        terminal_state["native_game_time"] = 5
        terminal_observation = {
            "kind": "natural_loss_board_identity_replacement_transition_v3",
            "outcome": "natural_loss",
            "observation_mode": "read_live_board(require_shooter=false)",
            "shooter_required": False,
            "first_gap_perf_counter_ns": 1,
            "clock_reset_perf_counter_ns": 2,
            "zero_target_teardown_perf_counter_ns": None,
            "board_identity_replacement_first_seen_perf_counter_ns": 3,
            "confirmed_perf_counter_ns": 4,
            "board_unavailable_count": 21,
            "clock_reset_unavailable_count": 17,
            "native_game_time_reset": 5200,
            "replacement_confirmation_snapshot_count": 6,
            "replacement_native_game_time_advance": 5,
            "before_restart": before_restart,
            "first_post_reset_state": first_post_reset,
            "zero_target_teardown_state": None,
            "first_replacement_state": first_replacement,
            "after_restart": dict(terminal_state),
        }
    elif outcome == "natural_loss" and loss_receipt == "target_write":
        board = 0x257AA220
        trace_sha256 = "sha256:" + "a" * 64
        source_anchor = {
            "order": 0,
            "thread_id": 38472,
            "call_address": 0x0065B81C,
            "caller": 0x0065B821,
            "framework_update": 3135,
            "board_address": board,
            "native_game_time": 109,
            "score": 7950,
            "score_target": 9650,
            "curve_plan_exhausted": False,
            "pre_index": 444,
            "pre_state_sha256": "sha256:" + "1" * 64,
            "output": 1838853123,
            "post_index": 445,
            "post_state_sha256": "sha256:" + "2" * 64,
            "perf_counter_ns": 1_000,
            "thread_crt_rand_state": 3293042926,
        }
        clear_write = {
            "order": 1,
            "write_order": 0,
            "thread_id": 38472,
            "watched_address": board + 0x108,
            "exception_address": 0x00412EF3,
            "post_instruction_eip": 0x00412EF3,
            "framework_update": 11854,
            "board_address": board,
            "native_game_time": 0,
            "score": 7950,
            "score_target": 0,
            "curve_plan_exhausted": True,
            "source_score_target": 9650,
            "previous_score_target": 9650,
            "post_score_target": 0,
            "same_active_board_as_source": True,
            "completion": False,
            "perf_counter_ns": 10_000,
            "registers": {
                "Edi": board,
                "Ebx": 0,
                "Eip": 0x00412EF3,
            },
            "code_window_address": 0x00412EE3,
            "code_window_hex": (
                "65100000899fc80e0000899f08010000899f7c060000899f78060000899fd00e"
            ),
        }
        positive_write = {
            "order": 2,
            "write_order": 1,
            "thread_id": 38472,
            "watched_address": board + 0x108,
            "exception_address": 0x00411B93,
            "post_instruction_eip": 0x00411B93,
            "framework_update": 11854,
            "board_address": board,
            "native_game_time": 0,
            "score": 7950,
            "score_target": 9616,
            "curve_plan_exhausted": False,
            "source_score_target": 9650,
            "previous_score_target": 0,
            "post_score_target": 9616,
            "same_active_board_as_source": True,
            "completion": True,
            "perf_counter_ns": 224_700,
            "registers": {
                "Esi": board,
                "Edx": 9616,
                "Eip": 0x00411B93,
            },
            "code_window_address": 0x00411B83,
            "code_window_hex": (
                "7e108b960401000003d0899608010000eb0ac78608010000000000008bcee88a"
            ),
        }
        terminal_state = {
            "board_address": board,
            "native_game_time": 3300,
            "score": 7950,
            "score_target": 9616,
            "loss_counter": -3300,
            "runtime_active": True,
            "curve_plan_exhausted": False,
            "active_ball_count": 38,
            "inserting_ball_count": 0,
            "pending_ball_count": 0,
            "fired_ball_count": 0,
        }
        terminal_observation = {
            "kind": "natural_loss_same_board_target_write_transition_v4",
            "outcome": "natural_loss",
            "observation_mode": (
                "read-only hardware watchpoint on anchored Board+0x108"
            ),
            "trace_sha256": trace_sha256,
            "trace_breakpoint_kind": (
                "board_global_precall_and_natural_loss_target_write"
            ),
            "trace_classification": (
                "read-only-hardware-breakpoint-preregistered-natural-loss-observation"
            ),
            "trace_call_count": 3,
            "source_anchor": source_anchor,
            "clear_write": clear_write,
            "positive_write": positive_write,
            "target_write_interval_ns": 214_700,
            "process_memory_writes": 0,
            "process_memory_mutation": False,
            "persistent_file_modified": False,
            "hardware_breakpoint_restored": True,
            "debugger_detach_error": None,
        }
        target_write_trace_summary = {
            "breakpoint_kind": (
                "board_global_precall_and_natural_loss_target_write"
            ),
            "status": "PASS",
            "failure": None,
            "call_count": 3,
            "process_memory_writes": 0,
            "process_memory_mutation": False,
            "persistent_file_modified": False,
            "hardware_breakpoint_restored": True,
            "hardware_breakpoint_restore_error": None,
            "debugger_detach_error": None,
            "source_anchor_count": 1,
            "target_write_count": 2,
            "stop_reason": "natural_loss_target_write_complete",
            "output_sha256": trace_sha256,
        }
    elif outcome == "natural_win":
        terminal_state = {
            "native_game_time": 7892,
            "score": 9700,
            "score_target": 9650,
            "loss_counter": 0,
            "runtime_active": False,
            "curve_plan_exhausted": True,
            "active_ball_count": 0,
            "inserting_ball_count": 0,
            "pending_ball_count": 0,
            "fired_ball_count": 0,
        }
        terminal_observation = None
    else:  # pragma: no cover - fixture misuse
        raise ValueError("unsupported outcome fixture")
    report = _recording_report(source)
    report.update(
        {
            "version": 2,
            "expected_outcome": outcome,
            "gameplay_policy": policy,
            "gameplay": {
                "status": "PASS",
                "expected_outcome": outcome,
                "gameplay_policy": policy,
                "outcome": outcome,
                "action_count": 0 if policy == "idle" else 1,
                "swap_count": 0,
                "fruit_shot_count": 0,
                "terminal_observation": terminal_observation,
                "final_state": terminal_state,
            },
        }
    )
    if target_write_trace_summary is not None:
        report["gameplay_mtrand_trace_requested"] = True
        report["mtrand_trace_kind"] = (
            "board_global_precall_and_natural_loss_target_write"
        )
        report["gameplay_mtrand_trace"] = target_write_trace_summary
        report["gameplay"].update(
            {
                "polling_status_before_target_write_receipt": "INCOMPLETE",
                "polling_outcome_before_target_write_receipt": "timeout",
                "polling_terminal_observation_before_target_write_receipt": (
                    None
                ),
            }
        )
    return report


def _collector_plan(output: bytes) -> dict[str, object]:
    demo = PopCapDemo.from_bytes(output)
    return {
        "schema": "zuma-rl.pc-golden-v4-collection-plan",
        "version": 5,
        "session_nonce": "collector-session-001",
        "dmo": {
            "bytes": len(output),
            "sha256": _sha256(output),
            "random_seed": demo.random_seed,
            "length_updates": demo.length_updates,
        },
        "runtime": {
            "runtime_executable_sha256": _RUNTIME_SHA256,
        },
        "prestate": {
            "template_state_root": _PRESTATE_ROOT,
        },
        "trace": {
            "seed_board_before_attach": False,
        },
    }


def _valid_payloads() -> dict[str, bytes]:
    source = _raw_recording_dmo()
    output, _ = normalize_duplicate_startup_read(source)
    recording = _json_bytes(_recording_report(source))
    plan = _json_bytes(_collector_plan(output))
    provenance = build_certifying_provenance(
        source_data=source,
        output_data=output,
        recording_report_data=recording,
        collector_plan_data=plan,
        expected_runtime_sha256=_RUNTIME_SHA256,
    )
    return {
        "source_data": source,
        "output_data": output,
        "recording_report_data": recording,
        "collector_plan_data": plan,
        "provenance_data": canonical_provenance_bytes(provenance),
    }


def test_exact_duplicate_startup_normalization_is_certifying() -> None:
    payloads = _valid_payloads()

    summary = validate_certifying_provenance(
        **payloads,
        expected_runtime_sha256=_RUNTIME_SHA256,
    )

    assert summary["classification"] == PROVENANCE_CLASSIFICATION
    assert summary["removed_startup_command_count"] == 1
    assert summary["gameplay_input_commands_preserved"] is True
    assert summary["recording_process_memory_writes"] == 0
    assert summary["provenance_version"] == 1
    assert summary["recording_outcome"] == "natural_win"
    assert summary["collector_board_seed_mutation"] is False


def test_preregistered_natural_loss_recording_is_certifying() -> None:
    source = _raw_recording_dmo()
    output, _ = normalize_duplicate_startup_read(source)
    recording = _json_bytes(_outcome_recording_report(source))
    plan = _json_bytes(_collector_plan(output))
    provenance = build_certifying_provenance(
        source_data=source,
        output_data=output,
        recording_report_data=recording,
        collector_plan_data=plan,
        expected_runtime_sha256=_RUNTIME_SHA256,
    )

    summary = validate_certifying_provenance(
        source_data=source,
        output_data=output,
        recording_report_data=recording,
        collector_plan_data=plan,
        provenance_data=canonical_provenance_bytes(provenance),
        expected_runtime_sha256=_RUNTIME_SHA256,
    )

    assert provenance["version"] == 2
    assert summary["provenance_version"] == 2
    assert summary["recording_outcome"] == "natural_loss"
    assert summary["recording_natural_outcome"] is True
    assert summary["recording_natural_win"] is False


def test_restart_transition_natural_loss_remains_supported() -> None:
    source = _raw_recording_dmo()
    output, _ = normalize_duplicate_startup_read(source)
    provenance = build_certifying_provenance(
        source_data=source,
        output_data=output,
        recording_report_data=_json_bytes(
            _outcome_recording_report(source, loss_receipt="restart")
        ),
        collector_plan_data=_json_bytes(_collector_plan(output)),
        expected_runtime_sha256=_RUNTIME_SHA256,
    )
    assert provenance["recording_report"]["outcome"] == "natural_loss"


def test_board_only_replacement_natural_loss_is_certifying() -> None:
    source = _raw_recording_dmo()
    output, _ = normalize_duplicate_startup_read(source)
    recording = _json_bytes(
        _outcome_recording_report(
            source,
            loss_receipt="board_replacement",
        )
    )
    plan = _json_bytes(_collector_plan(output))
    provenance = build_certifying_provenance(
        source_data=source,
        output_data=output,
        recording_report_data=recording,
        collector_plan_data=plan,
        expected_runtime_sha256=_RUNTIME_SHA256,
    )
    summary = validate_certifying_provenance(
        source_data=source,
        output_data=output,
        recording_report_data=recording,
        collector_plan_data=plan,
        provenance_data=canonical_provenance_bytes(provenance),
        expected_runtime_sha256=_RUNTIME_SHA256,
    )

    assert summary["recording_outcome"] == "natural_loss"
    assert summary["recording_natural_outcome"] is True


@pytest.mark.parametrize(
    "mutation",
    (
        lambda report: report["gameplay"]["terminal_observation"].update(
            observation_mode="read_live_board(require_shooter=true)"
        ),
        lambda report: report["gameplay"]["terminal_observation"].update(
            shooter_required=True
        ),
        lambda report: report["gameplay"]["terminal_observation"][
            "zero_target_teardown_state"
        ].update(score_target=1),
        lambda report: report["gameplay"]["terminal_observation"].update(
            zero_target_teardown_perf_counter_ns=1
        ),
        lambda report: (
            report["gameplay"]["terminal_observation"]["after_restart"].update(
                curve_plan_exhausted=True
            ),
            report["gameplay"]["final_state"].update(
                curve_plan_exhausted=True
            ),
        ),
    ),
)
def test_board_only_replacement_receipt_fails_closed(
    mutation: object,
) -> None:
    source = _raw_recording_dmo()
    output, _ = normalize_duplicate_startup_read(source)
    report = _outcome_recording_report(
        source,
        loss_receipt="board_replacement",
    )
    mutation(report)

    with pytest.raises(
        RetailDmoProvenanceError,
        match="restart transition does not establish a natural loss",
    ):
        build_certifying_provenance(
            source_data=source,
            output_data=output,
            recording_report_data=_json_bytes(report),
            collector_plan_data=_json_bytes(_collector_plan(output)),
            expected_runtime_sha256=_RUNTIME_SHA256,
        )


def test_board_identity_replacement_receipt_is_certifying() -> None:
    source = _raw_recording_dmo()
    output, _ = normalize_duplicate_startup_read(source)
    recording = _json_bytes(
        _outcome_recording_report(
            source,
            loss_receipt="board_identity",
        )
    )
    plan = _json_bytes(_collector_plan(output))
    provenance = build_certifying_provenance(
        source_data=source,
        output_data=output,
        recording_report_data=recording,
        collector_plan_data=plan,
        expected_runtime_sha256=_RUNTIME_SHA256,
    )

    summary = validate_certifying_provenance(
        source_data=source,
        output_data=output,
        recording_report_data=recording,
        collector_plan_data=plan,
        provenance_data=canonical_provenance_bytes(provenance),
        expected_runtime_sha256=_RUNTIME_SHA256,
    )

    assert summary["recording_outcome"] == "natural_loss"
    assert summary["recording_natural_outcome"] is True


@pytest.mark.parametrize(
    "mutation",
    (
        lambda receipt, final: receipt["before_restart"].update(
            board_address=0x200000
        ),
        lambda receipt, final: receipt["first_replacement_state"].update(
            board_address=0
        ),
        lambda receipt, final: receipt.update(
            replacement_confirmation_snapshot_count=1
        ),
        lambda receipt, final: receipt.update(
            replacement_native_game_time_advance=4
        ),
        lambda receipt, final: receipt["after_restart"].update(
            board_address=0x300000
        ),
        lambda receipt, final: final.update(board_address=0x300000),
        lambda receipt, final: receipt["first_replacement_state"].update(
            curve_plan_exhausted=True
        ),
        lambda receipt, final: receipt.update(
            zero_target_teardown_perf_counter_ns=2
        ),
    ),
)
def test_board_identity_replacement_receipt_fails_closed(
    mutation: object,
) -> None:
    source = _raw_recording_dmo()
    output, _ = normalize_duplicate_startup_read(source)
    report = _outcome_recording_report(
        source,
        loss_receipt="board_identity",
    )
    receipt = report["gameplay"]["terminal_observation"]
    final_state = report["gameplay"]["final_state"]
    mutation(receipt, final_state)

    with pytest.raises(RetailDmoProvenanceError):
        build_certifying_provenance(
            source_data=source,
            output_data=output,
            recording_report_data=_json_bytes(report),
            collector_plan_data=_json_bytes(_collector_plan(output)),
            expected_runtime_sha256=_RUNTIME_SHA256,
        )


def test_same_board_target_write_v4_receipt_is_certifying() -> None:
    source = _raw_recording_dmo()
    output, _ = normalize_duplicate_startup_read(source)
    recording = _json_bytes(
        _outcome_recording_report(
            source,
            loss_receipt="target_write",
        )
    )
    plan = _json_bytes(_collector_plan(output))
    provenance = build_certifying_provenance(
        source_data=source,
        output_data=output,
        recording_report_data=recording,
        collector_plan_data=plan,
        expected_runtime_sha256=_RUNTIME_SHA256,
    )
    summary = validate_certifying_provenance(
        source_data=source,
        output_data=output,
        recording_report_data=recording,
        collector_plan_data=plan,
        provenance_data=canonical_provenance_bytes(provenance),
        expected_runtime_sha256=_RUNTIME_SHA256,
    )
    assert summary["recording_outcome"] == "natural_loss"
    assert summary["recording_natural_outcome"] is True


@pytest.mark.parametrize(
    "mutation",
    (
        lambda report: report["gameplay"]["terminal_observation"][
            "clear_write"
        ].update(post_instruction_eip=0x00412EF4),
        lambda report: report["gameplay"]["terminal_observation"][
            "clear_write"
        ].update(code_window_hex="00" * 32),
        lambda report: report["gameplay"]["terminal_observation"][
            "positive_write"
        ].update(previous_score_target=1),
        lambda report: report["gameplay"]["terminal_observation"][
            "positive_write"
        ].update(native_game_time=1),
        lambda report: report["gameplay"]["terminal_observation"][
            "source_anchor"
        ].update(score=7951),
        lambda report: report["gameplay"]["terminal_observation"].update(
            target_write_interval_ns=5_000_001
        ),
        lambda report: report["gameplay"]["terminal_observation"].update(
            trace_sha256="sha256:" + "b" * 64
        ),
        lambda report: report["gameplay"]["terminal_observation"].update(
            process_memory_writes=1
        ),
        lambda report: report["gameplay"]["final_state"].update(
            board_address=0x257AA224
        ),
        lambda report: report.update(
            mtrand_trace_kind="board_global_precall_and_target_write"
        ),
    ),
)
def test_same_board_target_write_v4_receipt_fails_closed(
    mutation: object,
) -> None:
    source = _raw_recording_dmo()
    output, _ = normalize_duplicate_startup_read(source)
    report = _outcome_recording_report(
        source,
        loss_receipt="target_write",
    )
    mutation(report)
    with pytest.raises(RetailDmoProvenanceError):
        build_certifying_provenance(
            source_data=source,
            output_data=output,
            recording_report_data=_json_bytes(report),
            collector_plan_data=_json_bytes(_collector_plan(output)),
            expected_runtime_sha256=_RUNTIME_SHA256,
        )


def test_preregistered_v2_natural_win_recording_is_certifying() -> None:
    source = _raw_recording_dmo()
    output, _ = normalize_duplicate_startup_read(source)
    recording = _json_bytes(
        _outcome_recording_report(source, outcome="natural_win")
    )
    plan = _json_bytes(_collector_plan(output))
    provenance = build_certifying_provenance(
        source_data=source,
        output_data=output,
        recording_report_data=recording,
        collector_plan_data=plan,
        expected_runtime_sha256=_RUNTIME_SHA256,
    )

    summary = validate_certifying_provenance(
        source_data=source,
        output_data=output,
        recording_report_data=recording,
        collector_plan_data=plan,
        provenance_data=canonical_provenance_bytes(provenance),
        expected_runtime_sha256=_RUNTIME_SHA256,
    )

    assert summary["recording_outcome"] == "natural_win"
    assert summary["recording_natural_win"] is True


def test_zero_target_teardown_is_not_a_certifying_win() -> None:
    source = _raw_recording_dmo()
    output, _ = normalize_duplicate_startup_read(source)
    report = _outcome_recording_report(source, outcome="natural_win")
    report["gameplay"]["final_state"]["score_target"] = 0

    with pytest.raises(RetailDmoProvenanceError, match="at least 1"):
        build_certifying_provenance(
            source_data=source,
            output_data=output,
            recording_report_data=_json_bytes(report),
            collector_plan_data=_json_bytes(_collector_plan(output)),
            expected_runtime_sha256=_RUNTIME_SHA256,
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        (
            lambda report: report.update(
                {"expected_outcome": "natural_win"}
            ),
            "outcome contract",
        ),
        (
            lambda report: report["gameplay"].update(
                {"action_count": 1}
            ),
            "idle recording policy",
        ),
        (
            lambda report: report["gameplay"]["terminal_observation"][
                "state"
            ].update(
                {"score": 9650}
            ),
            "natural loss",
        ),
        (
            lambda report: report["gameplay"]["terminal_observation"].update(
                {"confirmed_perf_counter_ns": 2}
            ),
            "natural loss",
        ),
        (
            lambda report: report["gameplay"]["terminal_observation"].update(
                {"snapshot_count": 0}
            ),
            "at least 5",
        ),
    ),
)
def test_natural_loss_contract_fails_closed(
    mutation: object,
    match: str,
) -> None:
    source = _raw_recording_dmo()
    output, _ = normalize_duplicate_startup_read(source)
    report = _outcome_recording_report(source)
    mutation(report)

    with pytest.raises(RetailDmoProvenanceError, match=match):
        build_certifying_provenance(
            source_data=source,
            output_data=output,
            recording_report_data=_json_bytes(report),
            collector_plan_data=_json_bytes(_collector_plan(output)),
            expected_runtime_sha256=_RUNTIME_SHA256,
        )


def test_normalization_rejects_non_dword_one_duplicate() -> None:
    source = _raw_recording_dmo(duplicate_value=2)

    with pytest.raises(RetailDmoProvenanceError, match="REG_DWORD"):
        normalize_duplicate_startup_read(source)


def test_receipt_rejects_any_playback_dmo_change() -> None:
    payloads = _valid_payloads()
    changed = bytearray(payloads["output_data"])
    changed[-1] ^= 0x01
    payloads["output_data"] = bytes(changed)

    with pytest.raises(RetailDmoProvenanceError):
        validate_certifying_provenance(
            **payloads,
            expected_runtime_sha256=_RUNTIME_SHA256,
        )


def test_controlled_rng_source_recording_is_not_certifying() -> None:
    source = _raw_recording_dmo()
    output, _ = normalize_duplicate_startup_read(source)
    report = _recording_report(source)
    report["controlled_crt_rand_seed"] = 1234

    with pytest.raises(RetailDmoProvenanceError, match="controlled"):
        build_certifying_provenance(
            source_data=source,
            output_data=output,
            recording_report_data=_json_bytes(report),
            collector_plan_data=_json_bytes(_collector_plan(output)),
            expected_runtime_sha256=_RUNTIME_SHA256,
        )


def test_recording_and_collection_prestate_must_match() -> None:
    source = _raw_recording_dmo()
    output, _ = normalize_duplicate_startup_read(source)
    plan = _collector_plan(output)
    plan["prestate"] = {
        "template_state_root": "sha256:" + "9" * 64,
    }

    with pytest.raises(RetailDmoProvenanceError, match="same prestate"):
        build_certifying_provenance(
            source_data=source,
            output_data=output,
            recording_report_data=_json_bytes(_recording_report(source)),
            collector_plan_data=_json_bytes(plan),
            expected_runtime_sha256=_RUNTIME_SHA256,
        )


def test_receipt_unknown_or_retrospective_claim_is_rejected() -> None:
    payloads = _valid_payloads()
    declared = json.loads(payloads["provenance_data"])
    declared["retrospective_attestation"] = True
    payloads["provenance_data"] = _json_bytes(declared)

    with pytest.raises(RetailDmoProvenanceError, match="recomputed facts"):
        validate_certifying_provenance(
            **payloads,
            expected_runtime_sha256=_RUNTIME_SHA256,
        )
