"""Fail-closed provenance for retail-recorded PopCap DMO playback.

The Zuma's Revenge retail recorder emits two bit-identical ``Is3D`` registry
reads before the first Direct3D sync command, while the retail playback path
requests only one.  A raw recording from this build is therefore not directly
replayable: the extra service result shifts the shared bit stream.

This module recognizes exactly one transport normalization.  It removes the
first of those two bit-identical 107-bit startup reads and proves, byte for
byte, that the DMO header and every remaining command bit are unchanged.  No
gameplay input, timing, random seed, service result, or tail edit is allowed.

The normalization alone is not certifying evidence.  Certification also binds
the immutable raw DMO to a successful, non-mutating retail recording report
and binds the normalized output to the collector plan used for the two native
replays.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
from typing import Any, Iterable, Mapping, Sequence

from zuma_rl import popcap_dmo as _dmo
from zuma_rl.popcap_dmo import PopCapDemo, PopCapDemoError


PROVENANCE_SCHEMA = "zuma-rl.retail-dmo-transport-normalization"
PROVENANCE_VERSION = 1
OUTCOME_PROVENANCE_VERSION = 2
PROVENANCE_CLASSIFICATION = (
    "certifying-retail-recording-transport-normalization"
)
RECORDING_SCHEMA = "zuma-rl.retail-autoplay-recording"
RECORDING_VERSION = 1
OUTCOME_RECORDING_VERSION = 2
RECORDING_CLASSIFICATION = "candidate-recording-not-pc-golden"
CERTIFYING_RECORDING_OUTCOMES = frozenset(
    {"natural_loss", "natural_win"}
)
CERTIFYING_GAMEPLAY_POLICIES = frozenset({"autoplay", "idle"})
NATURAL_LOSS_RESTART_RECEIPT_KIND = (
    "natural_loss_restart_transition_v1"
)
NATURAL_LOSS_BOARD_REPLACEMENT_V2_RECEIPT_KIND = (
    "natural_loss_board_replacement_transition_v2"
)
NATURAL_LOSS_BOARD_IDENTITY_REPLACEMENT_RECEIPT_KIND = (
    "natural_loss_board_identity_replacement_transition_v3"
)
NATURAL_LOSS_TARGET_WRITE_V4_RECEIPT_KIND = (
    "natural_loss_same_board_target_write_transition_v4"
)
NATURAL_LOSS_TARGET_WRITE_TRACE_KIND = (
    "board_global_precall_and_natural_loss_target_write"
)
NATURAL_LOSS_TARGET_WRITE_TRACE_CLASSIFICATION = (
    "read-only-hardware-breakpoint-preregistered-natural-loss-observation"
)
NATURAL_LOSS_TARGET_CLEAR_WRITER_ADDRESS = 0x00412EED
NATURAL_LOSS_TARGET_CLEAR_INSTRUCTION_HEX = "899f08010000"
NATURAL_LOSS_TARGET_CLEAR_POST_EIP = 0x00412EF3
NATURAL_LOSS_TARGET_POSITIVE_WRITER_ADDRESS = 0x00411B8D
NATURAL_LOSS_TARGET_POSITIVE_INSTRUCTION_HEX = "899608010000"
NATURAL_LOSS_TARGET_POSITIVE_POST_EIP = 0x00411B93
NATURAL_LOSS_TARGET_WRITE_MAX_INTERVAL_NS = 5_000_000
NATURAL_LOSS_MINIMUM_PRE_RESTART_NATIVE_TIME = 1000
NATURAL_LOSS_MAXIMUM_POST_RESTART_NATIVE_TIME = 1000
NATURAL_LOSS_MINIMUM_NATIVE_TIME_RESET = 1000
NATURAL_LOSS_MINIMUM_REPLACEMENT_SNAPSHOTS = 2
NATURAL_LOSS_MINIMUM_REPLACEMENT_NATIVE_ADVANCE = 5
NATURAL_LOSS_STABLE_TERMINAL_RECEIPT_KIND = (
    "natural_loss_stable_terminal_v1"
)
NATURAL_LOSS_STABLE_TERMINAL_NS = 10_000_000_000
NATURAL_LOSS_STABLE_TERMINAL_MINIMUM_SNAPSHOTS = 5

PLAYBACK_DMO_ARTIFACT = "input.dmo"
RAW_RECORDING_DMO_ARTIFACT = "input.recording.dmo"
RECORDING_REPORT_ARTIFACT = "input.recording.report"
COLLECTOR_PLAN_ARTIFACT = "collector.plan"
PROVENANCE_ARTIFACT = "input.transport.normalization"

MAX_JSON_BYTES = 4 * 1024**2
_SHA256_PREFIX = "sha256:"
_REG_DWORD_ONE_SHA256 = (
    "sha256:" + hashlib.sha256(struct.pack("<I", 1)).hexdigest()
)


class RetailDmoProvenanceError(ValueError):
    """Raised when retail recording provenance is absent or not exact."""


def _sha256(data: bytes) -> str:
    return _SHA256_PREFIX + hashlib.sha256(data).hexdigest()


def _strict_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RetailDmoProvenanceError(f"{name} must be an integer")
    if value < minimum:
        raise RetailDmoProvenanceError(
            f"{name} must be at least {minimum}"
        )
    return value


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RetailDmoProvenanceError(
            f"{name} must be a non-empty string"
        )
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RetailDmoProvenanceError(f"{name} must be a JSON object")
    if any(not isinstance(key, str) for key in value):
        raise RetailDmoProvenanceError(f"{name} keys must be strings")
    return value


def _sha256_string(value: Any, name: str) -> str:
    result = _nonempty(value, name)
    if len(result) != 71 or not result.startswith(_SHA256_PREFIX):
        raise RetailDmoProvenanceError(f"{name} must be a SHA-256 digest")
    try:
        int(result[len(_SHA256_PREFIX) :], 16)
    except ValueError as error:
        raise RetailDmoProvenanceError(
            f"{name} must be a SHA-256 digest"
        ) from error
    return result


def _writer_bytes_match(
    event: Mapping[str, Any],
    *,
    writer_address: int,
    instruction_hex: str,
) -> bool:
    start = event.get("code_window_address")
    payload_hex = event.get("code_window_hex")
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or not isinstance(payload_hex, str)
    ):
        return False
    try:
        payload = bytes.fromhex(payload_hex)
        expected = bytes.fromhex(instruction_hex)
    except ValueError:
        return False
    offset = writer_address - start
    return offset >= 0 and payload[offset : offset + len(expected)] == expected


def _validate_natural_loss_target_write_v4(
    *,
    report: Mapping[str, Any],
    gameplay: Mapping[str, Any],
    final_state: Mapping[str, Any],
    observation: Mapping[str, Any],
) -> None:
    """Validate the independently preregistered same-Board writer receipt."""

    trace = _mapping(
        report.get("gameplay_mtrand_trace"),
        "recording natural-loss target-write trace summary",
    )
    source = _mapping(
        observation.get("source_anchor"),
        "recording natural-loss source anchor",
    )
    clear = _mapping(
        observation.get("clear_write"),
        "recording natural-loss target-clear write",
    )
    positive = _mapping(
        observation.get("positive_write"),
        "recording natural-loss positive-target write",
    )
    clear_registers = _mapping(
        clear.get("registers"),
        "recording natural-loss target-clear registers",
    )
    positive_registers = _mapping(
        positive.get("registers"),
        "recording natural-loss positive-target registers",
    )
    trace_sha256 = _sha256_string(
        observation.get("trace_sha256"),
        "recording natural-loss trace digest",
    )
    source_board = _strict_int(
        source.get("board_address"),
        "recording natural-loss source Board",
        minimum=1,
    )
    source_update = _strict_int(
        source.get("framework_update"),
        "recording natural-loss source update",
    )
    source_perf = _strict_int(
        source.get("perf_counter_ns"),
        "recording natural-loss source time",
        minimum=1,
    )
    source_order = _strict_int(
        source.get("order"),
        "recording natural-loss source order",
    )
    source_thread = _strict_int(
        source.get("thread_id"),
        "recording natural-loss source thread",
        minimum=1,
    )
    clear_update = _strict_int(
        clear.get("framework_update"),
        "recording natural-loss clear update",
    )
    clear_perf = _strict_int(
        clear.get("perf_counter_ns"),
        "recording natural-loss clear time",
        minimum=1,
    )
    clear_order = _strict_int(
        clear.get("order"),
        "recording natural-loss clear order",
    )
    positive_perf = _strict_int(
        positive.get("perf_counter_ns"),
        "recording natural-loss positive time",
        minimum=1,
    )
    positive_order = _strict_int(
        positive.get("order"),
        "recording natural-loss positive order",
    )
    positive_target = _strict_int(
        positive.get("post_score_target"),
        "recording natural-loss positive target",
        minimum=1,
    )
    interval_ns = _strict_int(
        observation.get("target_write_interval_ns"),
        "recording natural-loss target-write interval",
        minimum=1,
    )
    trace_call_count = _strict_int(
        observation.get("trace_call_count"),
        "recording natural-loss trace call count",
        minimum=3,
    )
    watched_address = source_board + 0x108
    final_board = _strict_int(
        final_state.get("board_address"),
        "recording final natural-loss Board",
        minimum=1,
    )
    _sha256_string(
        source.get("pre_state_sha256"),
        "recording natural-loss source pre-state digest",
    )
    _sha256_string(
        source.get("post_state_sha256"),
        "recording natural-loss source post-state digest",
    )

    invalid = (
        observation.get("outcome") != "natural_loss"
        or observation.get("observation_mode")
        != "read-only hardware watchpoint on anchored Board+0x108"
        or observation.get("trace_breakpoint_kind")
        != NATURAL_LOSS_TARGET_WRITE_TRACE_KIND
        or observation.get("trace_classification")
        != NATURAL_LOSS_TARGET_WRITE_TRACE_CLASSIFICATION
        or observation.get("process_memory_writes") != 0
        or observation.get("process_memory_mutation") is not False
        or observation.get("persistent_file_modified") is not False
        or observation.get("hardware_breakpoint_restored") is not True
        or observation.get("debugger_detach_error") is not None
        or report.get("gameplay_mtrand_trace_requested") is not True
        or report.get("mtrand_trace_kind")
        != NATURAL_LOSS_TARGET_WRITE_TRACE_KIND
        or trace.get("breakpoint_kind")
        != NATURAL_LOSS_TARGET_WRITE_TRACE_KIND
        or trace.get("status") != "PASS"
        or trace.get("failure") is not None
        or trace.get("call_count") != trace_call_count
        or trace.get("process_memory_writes") != 0
        or trace.get("process_memory_mutation") is not False
        or trace.get("persistent_file_modified") is not False
        or trace.get("hardware_breakpoint_restored") is not True
        or trace.get("hardware_breakpoint_restore_error") is not None
        or trace.get("debugger_detach_error") is not None
        or trace.get("source_anchor_count") != 1
        or trace.get("target_write_count") != 2
        or trace.get("stop_reason") != "natural_loss_target_write_complete"
        or trace.get("output_sha256") != trace_sha256
        or source.get("call_address") != 0x0065B81C
        or source.get("caller") != 0x0065B821
        or source.get("native_game_time") != 109
        or source.get("score") != 7950
        or source.get("score_target") != 9650
        or source.get("curve_plan_exhausted") is not False
        or clear.get("write_order") != 0
        or clear.get("thread_id") != source_thread
        or clear.get("board_address") != source_board
        or clear.get("watched_address") != watched_address
        or clear.get("same_active_board_as_source") is not True
        or clear.get("source_score_target") != 9650
        or clear.get("previous_score_target") != 9650
        or clear.get("post_score_target") != 0
        or clear.get("score_target") != 0
        or clear.get("native_game_time") != 0
        or clear.get("score") != 7950
        or clear.get("curve_plan_exhausted") is not True
        or clear.get("exception_address")
        != NATURAL_LOSS_TARGET_CLEAR_POST_EIP
        or clear.get("post_instruction_eip")
        != NATURAL_LOSS_TARGET_CLEAR_POST_EIP
        or clear.get("completion") is not False
        or clear_registers.get("Edi") != source_board
        or clear_registers.get("Ebx") != 0
        or clear_registers.get("Eip")
        != NATURAL_LOSS_TARGET_CLEAR_POST_EIP
        or not _writer_bytes_match(
            clear,
            writer_address=NATURAL_LOSS_TARGET_CLEAR_WRITER_ADDRESS,
            instruction_hex=NATURAL_LOSS_TARGET_CLEAR_INSTRUCTION_HEX,
        )
        or positive_target == 9650
        or positive.get("write_order") != 1
        or positive.get("thread_id") != source_thread
        or positive.get("board_address") != source_board
        or positive.get("watched_address") != watched_address
        or positive.get("same_active_board_as_source") is not True
        or positive.get("source_score_target") != 9650
        or positive.get("previous_score_target") != 0
        or positive.get("score_target") != positive_target
        or positive.get("native_game_time") != 0
        or positive.get("score") != 7950
        or positive.get("curve_plan_exhausted") is not False
        or positive.get("exception_address")
        != NATURAL_LOSS_TARGET_POSITIVE_POST_EIP
        or positive.get("post_instruction_eip")
        != NATURAL_LOSS_TARGET_POSITIVE_POST_EIP
        or positive.get("completion") is not True
        or positive.get("framework_update") != clear_update
        or positive_registers.get("Esi") != source_board
        or positive_registers.get("Edx") != positive_target
        or positive_registers.get("Eip")
        != NATURAL_LOSS_TARGET_POSITIVE_POST_EIP
        or not _writer_bytes_match(
            positive,
            writer_address=NATURAL_LOSS_TARGET_POSITIVE_WRITER_ADDRESS,
            instruction_hex=NATURAL_LOSS_TARGET_POSITIVE_INSTRUCTION_HEX,
        )
        or clear_update <= source_update
        or clear_perf <= source_perf
        or not source_order < clear_order < positive_order
        or positive_perf - clear_perf != interval_ns
        or interval_ns > NATURAL_LOSS_TARGET_WRITE_MAX_INTERVAL_NS
        or gameplay.get("polling_status_before_target_write_receipt")
        != "INCOMPLETE"
        or gameplay.get("polling_outcome_before_target_write_receipt")
        != "timeout"
        or gameplay.get(
            "polling_terminal_observation_before_target_write_receipt"
        )
        is not None
        or gameplay.get("fruit_shot_count") != 0
        or final_board != source_board
        or final_state.get("score") != 7950
        or final_state.get("score_target") != positive_target
    )
    if invalid:
        raise RetailDmoProvenanceError(
            "recording target-write transition does not establish a natural loss"
        )


def certifying_recording_outcome(report: Mapping[str, Any]) -> str:
    """Validate and return the natural outcome claimed by a recording.

    Version 1 is retained byte-for-byte for the already frozen natural-win
    corpus.  Version 2 removes the old win-only contradiction: the collector
    must declare its expected outcome and input policy, the observed outcome
    must match that declaration, and the terminal Board state must establish
    the claimed win or loss independently of the label.
    """

    version = report.get("version")
    gameplay = _mapping(report.get("gameplay"), "recording gameplay")
    if gameplay.get("status") != "PASS":
        raise RetailDmoProvenanceError(
            "recording gameplay did not pass"
        )
    outcome = gameplay.get("outcome")
    if version == RECORDING_VERSION:
        if outcome != "natural_win":
            raise RetailDmoProvenanceError(
                "legacy certifying source recording must end in a natural win"
            )
        return "natural_win"
    if version != OUTCOME_RECORDING_VERSION:
        raise RetailDmoProvenanceError(
            "recording report version is not certifying-compatible"
        )

    expected_outcome = report.get("expected_outcome")
    gameplay_policy = report.get("gameplay_policy")
    if (
        expected_outcome not in CERTIFYING_RECORDING_OUTCOMES
        or outcome != expected_outcome
        or gameplay_policy not in CERTIFYING_GAMEPLAY_POLICIES
    ):
        raise RetailDmoProvenanceError(
            "recording outcome contract was not satisfied"
        )
    if gameplay.get("expected_outcome") != expected_outcome:
        raise RetailDmoProvenanceError(
            "recording gameplay outcome contract differs"
        )
    if gameplay.get("gameplay_policy") != gameplay_policy:
        raise RetailDmoProvenanceError(
            "recording gameplay policy differs"
        )

    final_state = _mapping(
        gameplay.get("final_state"),
        "recording terminal Board state",
    )
    score = _strict_int(
        final_state.get("score"),
        "recording terminal score",
    )
    score_target = _strict_int(
        final_state.get("score_target"),
        "recording terminal score target",
        minimum=1,
    )
    terminal_counts = tuple(
        _strict_int(
            final_state.get(name),
            f"recording terminal {name}",
        )
        for name in (
            "active_ball_count",
            "inserting_ball_count",
            "pending_ball_count",
            "fired_ball_count",
        )
    )
    if expected_outcome == "natural_win":
        loss_counter = _strict_int(
            final_state.get("loss_counter"),
            "recording terminal loss counter",
        )
        if (
            final_state.get("runtime_active") is not False
            or any(terminal_counts)
            or score < score_target
            or final_state.get("curve_plan_exhausted") is not True
            or loss_counter != 0
        ):
            raise RetailDmoProvenanceError(
                "recording terminal Board state does not establish a natural win"
            )
    else:
        observation = _mapping(
            gameplay.get("terminal_observation"),
            "recording natural-loss observation",
        )
        if (
            observation.get("kind")
            == NATURAL_LOSS_TARGET_WRITE_V4_RECEIPT_KIND
        ):
            _validate_natural_loss_target_write_v4(
                report=report,
                gameplay=gameplay,
                final_state=final_state,
                observation=observation,
            )
        elif observation.get("kind") in {
            NATURAL_LOSS_RESTART_RECEIPT_KIND,
            NATURAL_LOSS_BOARD_REPLACEMENT_V2_RECEIPT_KIND,
            NATURAL_LOSS_BOARD_IDENTITY_REPLACEMENT_RECEIPT_KIND,
        }:
            is_board_replacement_v2 = (
                observation.get("kind")
                == NATURAL_LOSS_BOARD_REPLACEMENT_V2_RECEIPT_KIND
            )
            is_board_replacement_v3 = (
                observation.get("kind")
                == NATURAL_LOSS_BOARD_IDENTITY_REPLACEMENT_RECEIPT_KIND
            )
            before = _mapping(
                observation.get("before_restart"),
                "recording pre-restart Board state",
            )
            after = _mapping(
                observation.get("after_restart"),
                "recording replacement Board state",
            )
            first_gap = _strict_int(
                observation.get("first_gap_perf_counter_ns"),
                "recording natural-loss first unavailable observation",
                minimum=1,
            )
            confirmed = _strict_int(
                observation.get("confirmed_perf_counter_ns"),
                "recording natural-loss restart confirmation",
                minimum=1,
            )
            gap_count = _strict_int(
                observation.get("board_unavailable_count"),
                "recording natural-loss unavailable count",
                minimum=1,
            )
            before_native = _strict_int(
                before.get("native_game_time"),
                "recording pre-restart native game time",
            )
            after_native = _strict_int(
                after.get("native_game_time"),
                "recording replacement native game time",
            )
            after_target = _strict_int(
                after.get("score_target"),
                "recording replacement score target",
                minimum=1,
            )
            before_score = _strict_int(
                before.get("score"),
                "recording pre-restart score",
            )
            before_target = _strict_int(
                before.get("score_target"),
                "recording pre-restart score target",
                minimum=1,
            )
            after_counts = tuple(
                _strict_int(
                    after.get(name),
                    f"recording replacement {name}",
                )
                for name in (
                    "active_ball_count",
                    "inserting_ball_count",
                    "pending_ball_count",
                    "fired_ball_count",
                )
            )
            native_reset = _strict_int(
                observation.get("native_game_time_reset"),
                "recording natural-loss native game time reset",
            )
            final_matches_after = all(
                final_state.get(name) == after.get(name)
                for name in (
                    "native_game_time",
                    "score",
                    "score_target",
                    "runtime_active",
                    "active_ball_count",
                    "inserting_ball_count",
                    "pending_ball_count",
                    "fired_ball_count",
                )
            )
            board_replacement_v2_invalid = False
            board_replacement_v3_invalid = False
            if is_board_replacement_v2 or is_board_replacement_v3:
                clock_reset_seen = _strict_int(
                    observation.get("clock_reset_perf_counter_ns"),
                    "recording natural-loss clock reset observation",
                    minimum=1,
                )
                clock_reset_gap_count = _strict_int(
                    observation.get("clock_reset_unavailable_count"),
                    "recording natural-loss clock-reset unavailable count",
                    minimum=1,
                )
                first_post_reset = _mapping(
                    observation.get("first_post_reset_state"),
                    "recording first post-reset Board state",
                )
                first_post_native = _strict_int(
                    first_post_reset.get("native_game_time"),
                    "recording first post-reset native game time",
                )
            if is_board_replacement_v2:
                zero_target_seen = _strict_int(
                    observation.get(
                        "zero_target_teardown_perf_counter_ns"
                    ),
                    "recording natural-loss zero-target teardown observation",
                    minimum=1,
                )
                zero_target_state = _mapping(
                    observation.get("zero_target_teardown_state"),
                    "recording zero-target teardown Board state",
                )
                zero_target_native = _strict_int(
                    zero_target_state.get("native_game_time"),
                    "recording zero-target teardown native game time",
                )
                zero_target_value = _strict_int(
                    zero_target_state.get("score_target"),
                    "recording zero-target teardown score target",
                )
                board_replacement_v2_invalid = (
                    observation.get("observation_mode")
                    != "read_live_board(require_shooter=false)"
                    or observation.get("shooter_required") is not False
                    or not (
                        first_gap
                        <= clock_reset_seen
                        <= zero_target_seen
                        <= confirmed
                    )
                    or clock_reset_gap_count > gap_count
                    or first_post_native
                    > NATURAL_LOSS_MAXIMUM_POST_RESTART_NATIVE_TIME
                    or before_native - first_post_native
                    < NATURAL_LOSS_MINIMUM_NATIVE_TIME_RESET
                    or zero_target_native
                    > NATURAL_LOSS_MAXIMUM_POST_RESTART_NATIVE_TIME
                    or zero_target_value != 0
                    or zero_target_state.get("runtime_active") is not False
                    or after.get("curve_plan_exhausted") is not False
                    or final_state.get("curve_plan_exhausted")
                    != after.get("curve_plan_exhausted")
                )
            if is_board_replacement_v3:
                identity_replacement_seen = _strict_int(
                    observation.get(
                        "board_identity_replacement_first_seen_perf_counter_ns"
                    ),
                    "recording Board-identity replacement observation",
                    minimum=1,
                )
                replacement_snapshot_count = _strict_int(
                    observation.get(
                        "replacement_confirmation_snapshot_count"
                    ),
                    "recording replacement confirmation snapshot count",
                    minimum=NATURAL_LOSS_MINIMUM_REPLACEMENT_SNAPSHOTS,
                )
                replacement_native_advance = _strict_int(
                    observation.get("replacement_native_game_time_advance"),
                    "recording replacement native game time advance",
                )
                first_replacement = _mapping(
                    observation.get("first_replacement_state"),
                    "recording first replacement Board state",
                )
                before_address = _strict_int(
                    before.get("board_address"),
                    "recording pre-restart Board address",
                    minimum=1,
                )
                first_replacement_address = _strict_int(
                    first_replacement.get("board_address"),
                    "recording first replacement Board address",
                    minimum=1,
                )
                after_address = _strict_int(
                    after.get("board_address"),
                    "recording confirmed replacement Board address",
                    minimum=1,
                )
                final_address = _strict_int(
                    final_state.get("board_address"),
                    "recording final Board address",
                    minimum=1,
                )
                first_replacement_native = _strict_int(
                    first_replacement.get("native_game_time"),
                    "recording first replacement native game time",
                )
                first_replacement_target = _strict_int(
                    first_replacement.get("score_target"),
                    "recording first replacement score target",
                    minimum=1,
                )
                first_replacement_counts = tuple(
                    _strict_int(
                        first_replacement.get(name),
                        f"recording first replacement {name}",
                    )
                    for name in (
                        "active_ball_count",
                        "inserting_ball_count",
                        "pending_ball_count",
                        "fired_ball_count",
                    )
                )
                zero_target_seen_value = observation.get(
                    "zero_target_teardown_perf_counter_ns"
                )
                zero_target_state_value = observation.get(
                    "zero_target_teardown_state"
                )
                optional_zero_target_invalid = (
                    (zero_target_seen_value is None)
                    != (zero_target_state_value is None)
                )
                if (
                    zero_target_seen_value is not None
                    and zero_target_state_value is not None
                ):
                    zero_target_seen = _strict_int(
                        zero_target_seen_value,
                        "recording optional zero-target observation",
                        minimum=1,
                    )
                    zero_target_state = _mapping(
                        zero_target_state_value,
                        "recording optional zero-target Board state",
                    )
                    zero_target_native = _strict_int(
                        zero_target_state.get("native_game_time"),
                        "recording optional zero-target native game time",
                    )
                    zero_target_value = _strict_int(
                        zero_target_state.get("score_target"),
                        "recording optional zero-target score target",
                    )
                    optional_zero_target_invalid = (
                        optional_zero_target_invalid
                        or not (
                            clock_reset_seen
                            <= zero_target_seen
                            <= identity_replacement_seen
                        )
                        or zero_target_native
                        > NATURAL_LOSS_MAXIMUM_POST_RESTART_NATIVE_TIME
                        or zero_target_value != 0
                        or zero_target_state.get("runtime_active") is not False
                    )
                board_replacement_v3_invalid = (
                    observation.get("observation_mode")
                    != "read_live_board(require_shooter=false)"
                    or observation.get("shooter_required") is not False
                    or not (
                        first_gap
                        <= clock_reset_seen
                        <= identity_replacement_seen
                        <= confirmed
                    )
                    or clock_reset_gap_count > gap_count
                    or first_post_native
                    > NATURAL_LOSS_MAXIMUM_POST_RESTART_NATIVE_TIME
                    or before_native - first_post_native
                    < NATURAL_LOSS_MINIMUM_NATIVE_TIME_RESET
                    or before_address == first_replacement_address
                    or first_replacement_address != after_address
                    or final_address != after_address
                    or first_replacement_target != after_target
                    or first_replacement.get("runtime_active") is not False
                    or first_replacement.get("curve_plan_exhausted") is not False
                    or after.get("curve_plan_exhausted") is not False
                    or final_state.get("curve_plan_exhausted") is not False
                    or not any(first_replacement_counts)
                    or replacement_snapshot_count
                    < NATURAL_LOSS_MINIMUM_REPLACEMENT_SNAPSHOTS
                    or after_native < first_replacement_native
                    or replacement_native_advance
                    != after_native - first_replacement_native
                    or replacement_native_advance
                    < NATURAL_LOSS_MINIMUM_REPLACEMENT_NATIVE_ADVANCE
                    or optional_zero_target_invalid
                )
                final_matches_after = final_matches_after and (
                    final_state.get("board_address")
                    == after.get("board_address")
                    and final_state.get("curve_plan_exhausted")
                    == after.get("curve_plan_exhausted")
                )
            if (
                observation.get("outcome") != "natural_loss"
                or confirmed < first_gap
                or gap_count <= 0
                or before_score >= before_target
                or before_native
                < NATURAL_LOSS_MINIMUM_PRE_RESTART_NATIVE_TIME
                or after_native
                > NATURAL_LOSS_MAXIMUM_POST_RESTART_NATIVE_TIME
                or native_reset != before_native - after_native
                or native_reset < NATURAL_LOSS_MINIMUM_NATIVE_TIME_RESET
                or after_target <= 0
                or after.get("runtime_active") is not False
                or not any(after_counts)
                or not final_matches_after
                or board_replacement_v2_invalid
                or board_replacement_v3_invalid
            ):
                raise RetailDmoProvenanceError(
                    "recording restart transition does not establish a natural loss"
                )
        elif (
            observation.get("kind")
            == NATURAL_LOSS_STABLE_TERMINAL_RECEIPT_KIND
        ):
            observed = _mapping(
                observation.get("state"),
                "recording stable natural-loss Board state",
            )
            first_seen = _strict_int(
                observation.get("first_seen_perf_counter_ns"),
                "recording stable natural-loss first observation",
                minimum=1,
            )
            confirmed = _strict_int(
                observation.get("confirmed_perf_counter_ns"),
                "recording stable natural-loss confirmation",
                minimum=1,
            )
            snapshot_count = _strict_int(
                observation.get("snapshot_count"),
                "recording stable natural-loss snapshot count",
                minimum=NATURAL_LOSS_STABLE_TERMINAL_MINIMUM_SNAPSHOTS,
            )
            observed_native = _strict_int(
                observed.get("native_game_time"),
                "recording stable natural-loss native game time",
                minimum=1,
            )
            observed_score = _strict_int(
                observed.get("score"),
                "recording stable natural-loss score",
            )
            observed_target = _strict_int(
                observed.get("score_target"),
                "recording stable natural-loss score target",
                minimum=1,
            )
            observed_counts = tuple(
                _strict_int(
                    observed.get(name),
                    f"recording stable natural-loss {name}",
                )
                for name in (
                    "active_ball_count",
                    "inserting_ball_count",
                    "pending_ball_count",
                    "fired_ball_count",
                )
            )
            exact_fields = (
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
            if (
                observation.get("outcome") != "natural_loss"
                or confirmed - first_seen
                < NATURAL_LOSS_STABLE_TERMINAL_NS
                or snapshot_count
                < NATURAL_LOSS_STABLE_TERMINAL_MINIMUM_SNAPSHOTS
                or observed_native <= 0
                or observed_score >= observed_target
                or observed.get("runtime_active") is not False
                or observed.get("curve_plan_exhausted") is not True
                or observed_counts != (0, 0, 1, 0)
                or any(
                    final_state.get(name) != observed.get(name)
                    for name in exact_fields
                )
            ):
                raise RetailDmoProvenanceError(
                    "recording stable terminal state does not establish a natural loss"
                )
        else:
            loss_counter = _strict_int(
                final_state.get("loss_counter"),
                "recording terminal loss counter",
            )
            first_seen = _strict_int(
                observation.get("first_seen_perf_counter_ns"),
                "recording natural-loss first observation",
                minimum=1,
            )
            confirmed = _strict_int(
                observation.get("confirmed_perf_counter_ns"),
                "recording natural-loss confirmation",
                minimum=1,
            )
            if (
                final_state.get("runtime_active") is not False
                or any(terminal_counts)
                or score >= score_target
                or loss_counter <= 0
                or observation.get("outcome") != "natural_loss"
                or confirmed - first_seen < 10_000_000_000
            ):
                raise RetailDmoProvenanceError(
                    "recording terminal Board state does not establish a natural loss"
                )
    if gameplay_policy == "idle" and (
        gameplay.get("action_count") != 0
        or gameplay.get("swap_count") != 0
    ):
        raise RetailDmoProvenanceError(
            "idle recording policy emitted gameplay input"
        )
    return str(outcome)


def _reject_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RetailDmoProvenanceError(
                f"duplicate JSON key in provenance input: {key}"
            )
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise RetailDmoProvenanceError(
        f"forbidden non-finite JSON constant: {value}"
    )


def _validate_json_value(
    value: Any,
    name: str,
    *,
    depth: int = 0,
) -> None:
    if depth > 64:
        raise RetailDmoProvenanceError(f"{name} is nested too deeply")
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise RetailDmoProvenanceError(f"{name} must be finite")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise RetailDmoProvenanceError(
                    f"{name} keys must be strings"
                )
            _validate_json_value(
                item,
                f"{name}.{key}",
                depth=depth + 1,
            )
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            _validate_json_value(
                item,
                f"{name}[{index}]",
                depth=depth + 1,
            )
        return
    raise RetailDmoProvenanceError(f"{name} is not JSON-safe")


def load_strict_json_bytes(data: bytes, name: str) -> Mapping[str, Any]:
    """Load a bounded UTF-8 JSON object with duplicate-key rejection."""

    if not isinstance(data, bytes):
        raise TypeError("JSON data must be bytes")
    if len(data) > MAX_JSON_BYTES:
        raise RetailDmoProvenanceError(
            f"{name} exceeds the {MAX_JSON_BYTES}-byte limit"
        )
    try:
        text = data.decode("utf-8")
    except UnicodeError as error:
        raise RetailDmoProvenanceError(
            f"{name} is not valid UTF-8"
        ) from error
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except RetailDmoProvenanceError:
        raise
    except (RecursionError, json.JSONDecodeError) as error:
        raise RetailDmoProvenanceError(f"{name} is not valid JSON") from error
    _validate_json_value(value, name)
    return _mapping(value, name)


def _command_stream_offset(data: bytes) -> int:
    if len(data) < 20:
        raise RetailDmoProvenanceError("source DMO is too short")
    if struct.unpack_from("<I", data, 0)[0] != _dmo.DEMO_FILE_ID:
        raise RetailDmoProvenanceError("source DMO file id is invalid")
    version = struct.unpack_from("<I", data, 4)[0]
    if version not in _dmo.SUPPORTED_DEMO_VERSIONS:
        raise RetailDmoProvenanceError(
            f"source DMO version is unsupported: {version}"
        )
    offset = 12
    product_length = struct.unpack_from("<H", data, offset)[0]
    offset += 2 + product_length
    if offset + 4 > len(data):
        raise RetailDmoProvenanceError("source DMO header is truncated")
    if version >= 2:
        marker_size = struct.unpack_from("<I", data, offset)[0]
        offset += 4 + marker_size
    offset += 4
    if offset >= len(data):
        raise RetailDmoProvenanceError("source DMO has no command stream")
    return offset


def _bit_tuple(data: bytes, start: int, end: int) -> tuple[int, ...]:
    return tuple(
        (data[position // 8] >> (position % 8)) & 1
        for position in range(start, end)
    )


def _pack_lsb_bits(bits: Iterable[int]) -> bytes:
    values = tuple(bits)
    output = bytearray((len(values) + 7) // 8)
    for position, value in enumerate(values):
        if value not in (0, 1):
            raise RetailDmoProvenanceError("DMO bit values must be zero or one")
        if value:
            output[position // 8] |= 1 << (position % 8)
    return bytes(output)


def _scan_commands(
    command_data: bytes,
    length_updates: int,
) -> tuple[list[dict[str, Any]], int]:
    reader = _dmo._BitReader(command_data)
    update = 0
    rows: list[dict[str, Any]] = []
    try:
        while reader.remaining_bits:
            if (
                reader.remaining_bits <= 7
                and reader.remaining_is_zero_padding()
            ):
                break
            start = reader.bit_position
            update += reader.read_bits(4, context="command timing")
            short_form = bool(
                reader.read_bits(1, context="command short-form flag")
            )
            command_number = reader.read_bits(
                1 if short_form else 5,
                context="command number",
            )
            if short_form:
                if command_number == 0:
                    reader.read_bits(6, signed=True, context="mouse delta x")
                    reader.read_bits(6, signed=True, context="mouse delta y")
                    kind = "mouse_move"
                elif command_number == 1:
                    reader.read_bits(1, context="mouse button state")
                    reader.read_bits(
                        3,
                        signed=True,
                        context="mouse button",
                    )
                    kind = "mouse_button"
                else:  # pragma: no cover - one-bit command is exhaustive
                    raise RetailDmoProvenanceError(
                        f"unsupported short DMO command: {command_number}"
                    )
                payload: dict[str, Any] = {}
            else:
                try:
                    kind = _dmo._COMMAND_NAMES[command_number]
                except KeyError as error:
                    raise RetailDmoProvenanceError(
                        f"unsupported long DMO command: {command_number}"
                    ) from error
                payload = _dmo._read_long_command_payload(
                    reader,
                    command_number,
                )
            rows.append(
                {
                    "start": start,
                    "end": reader.bit_position,
                    "update": update,
                    "short_form": short_form,
                    "command_number": command_number,
                    "kind": kind,
                    "payload": payload,
                }
            )
            if update > length_updates:
                raise RetailDmoProvenanceError(
                    "source DMO command exceeds its declared length"
                )
    except PopCapDemoError as error:
        raise RetailDmoProvenanceError(
            "source DMO command stream is malformed"
        ) from error
    return rows, reader.bit_position


def normalize_duplicate_startup_read(
    source_data: bytes,
) -> tuple[bytes, Mapping[str, Any]]:
    """Return the sole accepted playback normalization and its safe facts."""

    if not isinstance(source_data, bytes):
        raise TypeError("source_data must be bytes")
    try:
        source_demo = PopCapDemo.from_bytes(source_data)
    except PopCapDemoError as error:
        raise RetailDmoProvenanceError(
            "raw retail recording DMO is malformed"
        ) from error
    offset = _command_stream_offset(source_data)
    length_updates = struct.unpack_from("<I", source_data, offset - 4)[0]
    command_data = source_data[offset:]
    rows, meaningful_bits = _scan_commands(command_data, length_updates)
    if len(rows) < 3:
        raise RetailDmoProvenanceError(
            "raw recording has fewer than three startup commands"
        )
    first, second, third = rows[:3]
    expected_registry = {
        "update": 0,
        "short_form": False,
        "command_number": 11,
        "kind": "registry_read",
    }
    for index, row in enumerate((first, second)):
        for key, expected in expected_registry.items():
            if row[key] != expected:
                raise RetailDmoProvenanceError(
                    f"startup command {index} is not the expected registry read"
                )
        payload = _mapping(row["payload"], f"startup command {index} payload")
        value = _mapping(
            payload.get("value"),
            f"startup command {index} registry value",
        )
        if (
            payload.get("success") is not True
            or payload.get("value_type") != 4
            or value.get("bytes") != 4
            or value.get("sha256") != _REG_DWORD_ONE_SHA256
        ):
            raise RetailDmoProvenanceError(
                f"startup command {index} is not REG_DWORD(1)"
            )
    if (
        first["start"] != 0
        or first["end"] != 107
        or second["start"] != 107
        or second["end"] != 214
    ):
        raise RetailDmoProvenanceError(
            "duplicate startup registry command spans are not exact"
        )
    first_bits = _bit_tuple(command_data, 0, 107)
    second_bits = _bit_tuple(command_data, 107, 214)
    if first_bits != second_bits:
        raise RetailDmoProvenanceError(
            "startup registry reads are not bit-identical"
        )
    if (
        third["start"] != 214
        or third["update"] != 0
        or third["short_form"] is not False
        or third["command_number"] != 18
        or third["kind"] != "sync"
    ):
        raise RetailDmoProvenanceError(
            "duplicate startup reads are not followed by the expected sync"
        )

    source_bits = _bit_tuple(command_data, 0, meaningful_bits)
    output_bits = source_bits[107:]
    output_data = source_data[:offset] + _pack_lsb_bits(output_bits)
    try:
        output_demo = PopCapDemo.from_bytes(output_data)
    except PopCapDemoError as error:  # pragma: no cover - protected by tests
        raise RetailDmoProvenanceError(
            "normalized playback DMO is malformed"
        ) from error
    if (
        output_demo.version != source_demo.version
        or output_demo.random_seed != source_demo.random_seed
        or output_demo.product_version != source_demo.product_version
        or output_demo.length_updates != source_demo.length_updates
        or output_demo.markers != source_demo.markers
    ):
        raise RetailDmoProvenanceError(
            "normalization changed the DMO header or markers"
        )
    if len(output_demo.commands) != len(source_demo.commands) - 1:
        raise RetailDmoProvenanceError(
            "normalization did not remove exactly one command"
        )
    if len(output_demo.input_commands) != len(source_demo.input_commands):
        raise RetailDmoProvenanceError(
            "normalization changed the gameplay input command count"
        )
    remaining_commands_match = all(
        (
            output_command.update == source_command.update
            and output_command.kind == source_command.kind
            and output_command.payload == source_command.payload
            and output_command.short_form == source_command.short_form
            and output_command.command_number == source_command.command_number
        )
        for output_command, source_command in zip(
            output_demo.commands,
            source_demo.commands[1:],
            strict=True,
        )
    )
    if not remaining_commands_match:
        raise RetailDmoProvenanceError(
            "normalization changed a command other than the duplicate read"
        )

    facts = {
        "kind": "remove_bit_identical_duplicate_startup_registry_read",
        "removed_command_index": 0,
        "duplicate_command_indices": [0, 1],
        "removed_bit_start": 0,
        "removed_bit_end": 107,
        "removed_bits": 107,
        "following_command_index": 2,
        "following_command_kind": "sync",
        "following_command_update": 0,
        "header_and_markers_preserved": True,
        "remaining_command_bits_preserved": True,
        "gameplay_input_commands_preserved": True,
        "random_seed_preserved": True,
        "length_updates_preserved": True,
    }
    return output_data, facts


def _validate_recording_report(
    report: Mapping[str, Any],
    *,
    source_data: bytes,
    expected_runtime_sha256: str,
) -> Mapping[str, Any]:
    if report.get("schema") != RECORDING_SCHEMA:
        raise RetailDmoProvenanceError(
            "recording report schema is not certifying-compatible"
        )
    if report.get("version") not in {
        RECORDING_VERSION,
        OUTCOME_RECORDING_VERSION,
    }:
        raise RetailDmoProvenanceError(
            "recording report version is not certifying-compatible"
        )
    if report.get("status") != "PASS":
        raise RetailDmoProvenanceError("retail recording did not pass")
    if report.get("classification") != RECORDING_CLASSIFICATION:
        raise RetailDmoProvenanceError(
            "retail recording classification is not the raw candidate class"
        )
    if report.get("dmo_bytes") != len(source_data):
        raise RetailDmoProvenanceError(
            "recording report DMO byte count does not bind the raw recording"
        )
    if report.get("dmo_sha256") != _sha256(source_data):
        raise RetailDmoProvenanceError(
            "recording report DMO digest does not bind the raw recording"
        )
    if report.get("runtime_executable_sha256") != expected_runtime_sha256:
        raise RetailDmoProvenanceError(
            "recording runtime differs from the native replay runtime"
        )
    if report.get("process_memory_writes") != 0:
        raise RetailDmoProvenanceError(
            "recording used process-memory writes"
        )
    if report.get("normal_exit") is not True:
        raise RetailDmoProvenanceError(
            "recording did not end through a normal retail exit"
        )
    if report.get("process_exit_code") != 0:
        raise RetailDmoProvenanceError(
            "recording process did not exit successfully"
        )
    if report.get("exit_method") != "retail_ui":
        raise RetailDmoProvenanceError(
            "recording did not use the retail UI exit path"
        )
    if report.get("input_transport") != "Win32 SendInput":
        raise RetailDmoProvenanceError(
            "recording input transport is not the declared Win32 path"
        )
    if report.get("host_restored") is not True:
        raise RetailDmoProvenanceError(
            "recording did not attest exact host restoration"
        )
    host_pre = _nonempty(
        report.get("host_pre_state_root"),
        "recording host_pre_state_root",
    )
    host_restored = _nonempty(
        report.get("host_restored_state_root"),
        "recording host_restored_state_root",
    )
    if host_pre != host_restored:
        raise RetailDmoProvenanceError(
            "recording host state was not restored exactly"
        )
    outcome = certifying_recording_outcome(report)
    for key in (
        "controlled_crt_rand_seed",
        "controlled_global_rng_seed",
        "global_rng_seed_override",
        "rng_override",
        "seed_control",
        "startup_rng",
    ):
        if key in report and report[key] not in (None, False, 0, "", [], {}):
            raise RetailDmoProvenanceError(
                f"recording used forbidden controlled instrumentation: {key}"
            )
    summary = {
        "schema": RECORDING_SCHEMA,
        "version": report["version"],
        "session_nonce": _nonempty(
            report.get("session_nonce"),
            "recording session_nonce",
        ),
        "host_pre_state_root": host_pre,
        "applied_pre_state_root": _nonempty(
            report.get("applied_pre_state_root"),
            "recording applied_pre_state_root",
        ),
        "runtime_executable_sha256": expected_runtime_sha256,
        "process_memory_writes": 0,
        "normal_retail_exit": True,
        "host_restored_exactly": True,
    }
    if report["version"] == RECORDING_VERSION:
        summary["natural_win"] = True
    else:
        summary.update(
            {
                "expected_outcome": report["expected_outcome"],
                "gameplay_policy": report["gameplay_policy"],
                "outcome": outcome,
                "natural_outcome": True,
            }
        )
    return summary


def _validate_collector_plan(
    plan: Mapping[str, Any],
    *,
    output_demo: PopCapDemo,
    output_data: bytes,
    expected_runtime_sha256: str,
    recording_summary: Mapping[str, Any],
) -> Mapping[str, Any]:
    if plan.get("schema") != "zuma-rl.pc-golden-v4-collection-plan":
        raise RetailDmoProvenanceError(
            "collector plan schema is not certifying-compatible"
        )
    if plan.get("version") != 5:
        raise RetailDmoProvenanceError(
            "collector plan version is not certifying-compatible"
        )
    dmo = _mapping(plan.get("dmo"), "collector plan dmo")
    expected_dmo = {
        "bytes": len(output_data),
        "sha256": _sha256(output_data),
        "random_seed": output_demo.random_seed,
        "length_updates": output_demo.length_updates,
    }
    for key, expected in expected_dmo.items():
        if dmo.get(key) != expected:
            raise RetailDmoProvenanceError(
                f"collector plan does not bind normalized DMO field {key}"
            )
    runtime = _mapping(plan.get("runtime"), "collector plan runtime")
    if runtime.get("runtime_executable_sha256") != expected_runtime_sha256:
        raise RetailDmoProvenanceError(
            "collector plan runtime differs from the recording runtime"
        )
    prestate = _mapping(plan.get("prestate"), "collector plan prestate")
    template_root = _nonempty(
        prestate.get("template_state_root"),
        "collector plan prestate template_state_root",
    )
    if template_root != recording_summary["applied_pre_state_root"]:
        raise RetailDmoProvenanceError(
            "recording and replay collection did not use the same prestate"
        )
    trace = _mapping(plan.get("trace"), "collector plan trace")
    if trace.get("seed_board_before_attach") is not False:
        raise RetailDmoProvenanceError(
            "collector plan used a board-seed mutation"
        )
    return {
        "schema": plan["schema"],
        "version": plan["version"],
        "session_nonce": _nonempty(
            plan.get("session_nonce"),
            "collector plan session_nonce",
        ),
        "prestate_template_state_root": template_root,
        "runtime_executable_sha256": expected_runtime_sha256,
        "normalized_dmo_bound": True,
        "board_seed_mutation": False,
    }


def _artifact_descriptor(
    artifact: str,
    data: bytes,
) -> dict[str, Any]:
    return {
        "artifact": artifact,
        "bytes": len(data),
        "sha256": _sha256(data),
    }


def build_certifying_provenance(
    *,
    source_data: bytes,
    output_data: bytes,
    recording_report_data: bytes,
    collector_plan_data: bytes,
    expected_runtime_sha256: str,
) -> Mapping[str, Any]:
    """Build the canonical receipt after independently checking every input."""

    expected_output, transformation = normalize_duplicate_startup_read(
        source_data
    )
    if output_data != expected_output:
        raise RetailDmoProvenanceError(
            "playback DMO is not the exact permitted startup normalization"
        )
    try:
        source_demo = PopCapDemo.from_bytes(source_data)
        output_demo = PopCapDemo.from_bytes(output_data)
    except PopCapDemoError as error:  # pragma: no cover - normalized above
        raise RetailDmoProvenanceError("provenance DMO parsing failed") from error
    recording_report = load_strict_json_bytes(
        recording_report_data,
        "retail recording report",
    )
    recording_summary = _validate_recording_report(
        recording_report,
        source_data=source_data,
        expected_runtime_sha256=expected_runtime_sha256,
    )
    collector_plan = load_strict_json_bytes(
        collector_plan_data,
        "collector plan",
    )
    collector_summary = _validate_collector_plan(
        collector_plan,
        output_demo=output_demo,
        output_data=output_data,
        expected_runtime_sha256=expected_runtime_sha256,
        recording_summary=recording_summary,
    )
    provenance_version = (
        PROVENANCE_VERSION
        if recording_summary["version"] == RECORDING_VERSION
        else OUTCOME_PROVENANCE_VERSION
    )
    return {
        "schema": PROVENANCE_SCHEMA,
        "version": provenance_version,
        "classification": PROVENANCE_CLASSIFICATION,
        "source": {
            **_artifact_descriptor(
                RAW_RECORDING_DMO_ARTIFACT,
                source_data,
            ),
            "dmo_version": source_demo.version,
            "random_seed": source_demo.random_seed,
            "length_updates": source_demo.length_updates,
            "command_count": len(source_demo.commands),
            "input_command_count": len(source_demo.input_commands),
        },
        "recording_report": {
            **_artifact_descriptor(
                RECORDING_REPORT_ARTIFACT,
                recording_report_data,
            ),
            **recording_summary,
        },
        "transformation": dict(transformation),
        "output": {
            **_artifact_descriptor(PLAYBACK_DMO_ARTIFACT, output_data),
            "dmo_version": output_demo.version,
            "random_seed": output_demo.random_seed,
            "length_updates": output_demo.length_updates,
            "command_count": len(output_demo.commands),
            "input_command_count": len(output_demo.input_commands),
        },
        "collector_plan": {
            **_artifact_descriptor(
                COLLECTOR_PLAN_ARTIFACT,
                collector_plan_data,
            ),
            **collector_summary,
        },
    }


def canonical_provenance_bytes(provenance: Mapping[str, Any]) -> bytes:
    """Serialize a receipt deterministically for content-addressed packaging."""

    return (
        json.dumps(
            provenance,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def validate_certifying_provenance(
    *,
    source_data: bytes,
    output_data: bytes,
    recording_report_data: bytes,
    collector_plan_data: bytes,
    provenance_data: bytes,
    expected_runtime_sha256: str,
) -> Mapping[str, Any]:
    """Recompute and require an exact canonical certification receipt."""

    declared = load_strict_json_bytes(
        provenance_data,
        "retail DMO provenance receipt",
    )
    expected = build_certifying_provenance(
        source_data=source_data,
        output_data=output_data,
        recording_report_data=recording_report_data,
        collector_plan_data=collector_plan_data,
        expected_runtime_sha256=expected_runtime_sha256,
    )
    if declared != expected:
        raise RetailDmoProvenanceError(
            "retail DMO provenance receipt differs from recomputed facts"
        )
    recording = _mapping(
        expected.get("recording_report"),
        "retail recording provenance summary",
    )
    outcome = recording.get("outcome")
    if outcome is None and recording.get("natural_win") is True:
        outcome = "natural_win"
    return {
        "classification": PROVENANCE_CLASSIFICATION,
        "provenance_version": expected["version"],
        "source_dmo_sha256": expected["source"]["sha256"],
        "playback_dmo_sha256": expected["output"]["sha256"],
        "random_seed": expected["output"]["random_seed"],
        "source_command_count": expected["source"]["command_count"],
        "playback_command_count": expected["output"]["command_count"],
        "removed_startup_command_count": 1,
        "gameplay_input_commands_preserved": True,
        "recording_process_memory_writes": 0,
        "recording_outcome": outcome,
        "recording_natural_outcome": (
            outcome in CERTIFYING_RECORDING_OUTCOMES
        ),
        "recording_natural_win": outcome == "natural_win",
        "recording_normal_retail_exit": True,
        "recording_host_restored_exactly": True,
        "collector_prestate_matched": True,
        "collector_board_seed_mutation": False,
    }


def read_certifying_provenance(
    *,
    source_path: str | Path,
    output_path: str | Path,
    recording_report_path: str | Path,
    collector_plan_path: str | Path,
    provenance_path: str | Path,
    expected_runtime_sha256: str,
) -> Mapping[str, Any]:
    """Read bounded artifacts and validate one certification receipt."""

    paths = {
        "source_data": Path(source_path),
        "output_data": Path(output_path),
        "recording_report_data": Path(recording_report_path),
        "collector_plan_data": Path(collector_plan_path),
        "provenance_data": Path(provenance_path),
    }
    payloads: dict[str, bytes] = {}
    for name, path in paths.items():
        try:
            size = path.stat().st_size
            if name.endswith("_data") and "dmo" not in name and size > MAX_JSON_BYTES:
                raise RetailDmoProvenanceError(
                    f"{path.name} exceeds the provenance JSON size limit"
                )
            payloads[name] = path.read_bytes()
        except OSError as error:
            raise RetailDmoProvenanceError(
                f"provenance artifact is unreadable: {path.name}"
            ) from error
    return validate_certifying_provenance(
        source_data=payloads["source_data"],
        output_data=payloads["output_data"],
        recording_report_data=payloads["recording_report_data"],
        collector_plan_data=payloads["collector_plan_data"],
        provenance_data=payloads["provenance_data"],
        expected_runtime_sha256=expected_runtime_sha256,
    )


__all__ = [
    "COLLECTOR_PLAN_ARTIFACT",
    "CERTIFYING_GAMEPLAY_POLICIES",
    "CERTIFYING_RECORDING_OUTCOMES",
    "OUTCOME_PROVENANCE_VERSION",
    "OUTCOME_RECORDING_VERSION",
    "PLAYBACK_DMO_ARTIFACT",
    "PROVENANCE_ARTIFACT",
    "PROVENANCE_CLASSIFICATION",
    "PROVENANCE_SCHEMA",
    "PROVENANCE_VERSION",
    "RAW_RECORDING_DMO_ARTIFACT",
    "RECORDING_REPORT_ARTIFACT",
    "RetailDmoProvenanceError",
    "build_certifying_provenance",
    "canonical_provenance_bytes",
    "certifying_recording_outcome",
    "normalize_duplicate_startup_read",
    "read_certifying_provenance",
    "validate_certifying_provenance",
]
