"""Strict reader for one formal retail exact-step full-state trajectory."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np

from .pc_exact_step_evidence import (
    ExactStepFrameEvidence,
    PcExactStepEvidenceError,
    decode_top_down_bgra_bmp,
    read_canonical_json,
)
from .pc_external_input import (
    EXTERNAL_INPUT_GUARD_BINDING_SCHEMA,
    EXTERNAL_INPUT_GUARD_BINDING_VERSION,
    PcExternalInputError,
    validate_external_input_guard_receipt,
)
from .pc_memory_evidence import (
    PcMemoryEvidenceError,
    validate_probe_payloads,
)
from .pc_memory_trajectory import (
    TRAJECTORY_SAMPLE_PHASE,
    TRAJECTORY_SCHEMA,
    TRAJECTORY_VERSION,
    PcMemoryTrajectoryError,
    TrajectoryFrame,
    load_memory_trajectory,
)
from .pc_video import canonical_rgb24_frame_sha256


FORMAL_FULL_STATE_CLASSIFICATION = (
    "formal_pc_full_state_exact_step_source"
)
FORMAL_FULL_STATE_SNAPSHOT_SEMANTICS = (
    "formal_full_state_external_lossless_anchor"
)
FORMAL_FRUIT_TRIGGER_TRANSCRIPT_SCHEMA = (
    "zuma-rl.pc-formal-fruit-lifecycle-trigger-transcript"
)
FORMAL_FRUIT_TRIGGER_BINDING_SCHEMA = (
    "zuma-rl.pc-formal-fruit-lifecycle-trigger-binding"
)
FORMAL_FRUIT_TRIGGER_VERSION = 1


def _recorded_host_path(value: Any) -> Path:
    """Translate an immutable Windows capture path when read under WSL."""

    text = str(value)
    match = re.fullmatch(r"([A-Za-z]):[\\/](.*)", text)
    if match is not None and os.name != "nt":
        drive, suffix = match.groups()
        return Path("/mnt") / drive.lower() / Path(
            suffix.replace("\\", "/")
        )
    return Path(text)
DIAGNOSTIC_SNAPSHOT_SEMANTICS = (
    "diagnostic_single_frame_not_formal_evidence"
)
EXPECTED_WIDTH = 800
EXPECTED_HEIGHT = 600


class PcFullStateEvidenceError(ValueError):
    """A formal full-state source or one of its artifacts is invalid."""


@dataclass(frozen=True, slots=True)
class FullStateLoadedRun:
    run_id: str
    selected_attempt: int
    process_id: int
    process_creation_filetime_100ns: int
    dmo_sha256: str
    runtime_sha256: str
    freeze_update: int
    start_update: int
    end_update: int
    warmup_tick_count: int
    attempts_sha256: str
    probe_sha256: str
    index_sha256: str
    external_input_guard_sha256: str
    campaign_started_perf_counter_ns: int
    attempt_finished_perf_counter_ns: int
    frames: tuple[TrajectoryFrame, ...]
    visual_frames: tuple[ExactStepFrameEvidence, ...]


def _fail(code: str) -> None:
    raise PcFullStateEvidenceError(code)


def _mapping(value: Any, code: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        _fail(code)
    return value


def _integer(value: Any, code: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(code)
    return value


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise PcFullStateEvidenceError(
            "full_state_bound_artifact_unreadable"
        ) from error
    return "sha256:" + digest.hexdigest()


def _artifact_path(root: Path, value: Any, code: str) -> Path:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or Path(value).is_absolute()
    ):
        _fail(code)
    try:
        path = (root / value).resolve(strict=True)
        path.relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as error:
        raise PcFullStateEvidenceError(code) from error
    if not path.is_file():
        _fail(code)
    return path


def _validate_optional_fruit_lifecycle_trigger(
    probe: Mapping[str, Any],
    *,
    probe_path: Path,
    expected_process_id: int,
    expected_freeze_update: int,
    expected_end_update: int,
) -> None:
    """Recompute the optional no-skip, first-lifecycle trigger receipt."""

    value = probe.get("formal_fruit_lifecycle_trigger")
    if value is None:
        return
    binding = _mapping(value, "full_state_fruit_trigger_binding_invalid")
    required_binding = {
        "schema",
        "version",
        "status",
        "artifact",
        "artifact_sha256",
        "selection_rule",
        "monitor_start_update",
        "threshold_observation_update",
        "freeze_update",
        "trajectory_end_update",
        "trajectory_tick_count",
        "process_id",
        "read_only_observation",
        "gameplay_or_rng_process_memory_write_count",
    }
    if set(binding) != required_binding:
        _fail("full_state_fruit_trigger_binding_invalid")
    transcript_path = _artifact_path(
        probe_path.parent,
        binding.get("artifact"),
        "full_state_fruit_trigger_artifact_missing",
    )
    transcript = _mapping(
        read_canonical_json(transcript_path),
        "full_state_fruit_trigger_transcript_invalid",
    )
    rule = (
        "first_observed_active_fruit_lifecycle_at_or_below_"
        "remaining_threshold_no_skips"
    )
    if (
        binding.get("schema") != FORMAL_FRUIT_TRIGGER_BINDING_SCHEMA
        or binding.get("version") != FORMAL_FRUIT_TRIGGER_VERSION
        or binding.get("status") != "PASS"
        or binding.get("artifact_sha256") != _sha256_path(transcript_path)
        or binding.get("selection_rule") != rule
        or binding.get("process_id") != expected_process_id
        or binding.get("freeze_update") != expected_freeze_update
        or binding.get("trajectory_end_update") != expected_end_update
        or binding.get("read_only_observation") is not True
        or binding.get("gameplay_or_rng_process_memory_write_count") != 0
        or transcript.get("schema")
        != FORMAL_FRUIT_TRIGGER_TRANSCRIPT_SCHEMA
        or transcript.get("version") != FORMAL_FRUIT_TRIGGER_VERSION
        or transcript.get("status") != "PASS"
        or transcript.get("selection_rule") != rule
        or transcript.get("process_id") != expected_process_id
    ):
        _fail("full_state_fruit_trigger_binding_invalid")

    config = _mapping(
        transcript.get("config"),
        "full_state_fruit_trigger_config_invalid",
    )
    config_fields = {
        "monitor_start_update",
        "maximum_framework_update",
        "remaining_threshold_ticks",
        "slowdown_lead_updates",
        "freeze_lead_updates",
        "trajectory_tick_count",
        "poll_interval_seconds",
    }
    if set(config) != config_fields:
        _fail("full_state_fruit_trigger_config_invalid")
    integer_fields = config_fields - {"poll_interval_seconds"}
    if any(
        isinstance(config.get(name), bool)
        or not isinstance(config.get(name), int)
        or config[name] < 0
        for name in integer_fields
    ):
        _fail("full_state_fruit_trigger_config_invalid")
    interval = config.get("poll_interval_seconds")
    if (
        isinstance(interval, bool)
        or not isinstance(interval, (int, float))
        or not 0 < float(interval) <= 0.1
        or config["maximum_framework_update"]
        <= config["monitor_start_update"]
        or config["slowdown_lead_updates"] < 4
        or config["freeze_lead_updates"]
        <= config["slowdown_lead_updates"]
        or config["remaining_threshold_ticks"]
        <= config["freeze_lead_updates"] + 32
        or config["trajectory_tick_count"]
        <= config["remaining_threshold_ticks"]
        or binding.get("monitor_start_update")
        != config["monitor_start_update"]
        or binding.get("trajectory_tick_count")
        != config["trajectory_tick_count"]
    ):
        _fail("full_state_fruit_trigger_config_invalid")

    raw_observations = transcript.get("observations")
    if not isinstance(raw_observations, list) or not raw_observations:
        _fail("full_state_fruit_trigger_observations_invalid")
    observation_fields = {
        "framework_update",
        "native_game_time",
        "board_update_count",
        "active",
        "active_point_pointer",
        "selected_point_index",
        "collecting",
        "expiry_time",
        "remaining_ticks",
    }
    observations: list[Mapping[str, Any]] = []
    previous_update = -1
    for raw in raw_observations:
        row = _mapping(raw, "full_state_fruit_trigger_observation_invalid")
        if set(row) != observation_fields:
            _fail("full_state_fruit_trigger_observation_invalid")
        update = row.get("framework_update")
        active = row.get("active")
        pointer = row.get("active_point_pointer")
        selected_point = row.get("selected_point_index")
        collecting = row.get("collecting")
        native_time = row.get("native_game_time")
        board_count = row.get("board_update_count")
        expiry = row.get("expiry_time")
        remaining = row.get("remaining_ticks")
        if (
            isinstance(update, bool)
            or not isinstance(update, int)
            or update <= previous_update
            or isinstance(active, bool) is False
            or isinstance(pointer, bool)
            or not isinstance(pointer, int)
            or pointer < 0
            or isinstance(selected_point, bool)
            or not isinstance(selected_point, int)
            or isinstance(collecting, bool) is False
            or isinstance(native_time, bool)
            or not isinstance(native_time, int)
            or native_time < 0
            or isinstance(board_count, bool)
            or not isinstance(board_count, int)
            or board_count < 0
            or isinstance(expiry, bool)
            or not isinstance(expiry, int)
            or active != (pointer != 0)
            or (not active and selected_point not in {-1, 0})
            or (active and selected_point < 0)
            or (
                active
                and (
                    isinstance(remaining, bool)
                    or not isinstance(remaining, int)
                    or remaining != expiry - native_time
                )
            )
            or (not active and remaining is not None)
        ):
            _fail("full_state_fruit_trigger_observation_invalid")
        previous_update = update
        observations.append(row)
    if (
        observations[0]["framework_update"]
        < config["monitor_start_update"]
        or observations[0]["framework_update"]
        > config["monitor_start_update"] + 2
        or observations[-1]["framework_update"]
        > config["maximum_framework_update"]
    ):
        _fail("full_state_fruit_trigger_monitor_window_invalid")

    selection = _mapping(
        transcript.get("selection"),
        "full_state_fruit_trigger_selection_invalid",
    )
    selection_fields = {
        "first_active_observation_index",
        "threshold_observation_index",
        "threshold_observation",
        "slowdown_update",
        "freeze_update",
        "trajectory_end_update",
        "trajectory_tick_count",
        "frozen_observation",
    }
    if set(selection) != selection_fields:
        _fail("full_state_fruit_trigger_selection_invalid")
    first_index = selection.get("first_active_observation_index")
    threshold_index = selection.get("threshold_observation_index")
    if (
        isinstance(first_index, bool)
        or not isinstance(first_index, int)
        or isinstance(threshold_index, bool)
        or not isinstance(threshold_index, int)
        or not 0 <= first_index <= threshold_index
        or threshold_index != len(observations) - 1
    ):
        _fail("full_state_fruit_trigger_selection_invalid")
    derived_first = next(
        (
            index
            for index, row in enumerate(observations)
            if row["active"]
        ),
        None,
    )
    threshold = config["remaining_threshold_ticks"]
    lifecycle_pointer = observations[first_index]["active_point_pointer"]
    if (
        derived_first != first_index
        or lifecycle_pointer == 0
        or any(row["active"] for row in observations[:first_index])
        or any(
            row["active"] is not True
            or row["active_point_pointer"] != lifecycle_pointer
            or row["collecting"] is not False
            or not isinstance(row["remaining_ticks"], int)
            or row["remaining_ticks"] <= 0
            for row in observations[first_index:]
        )
        or any(
            row["remaining_ticks"] <= threshold
            for row in observations[first_index:threshold_index]
        )
        or observations[threshold_index]["remaining_ticks"] > threshold
        or observations[threshold_index]["remaining_ticks"]
        <= config["freeze_lead_updates"] + 32
        or selection.get("threshold_observation")
        != observations[threshold_index]
    ):
        _fail("full_state_fruit_trigger_no_skip_rule_invalid")

    threshold_update = observations[threshold_index]["framework_update"]
    frozen = _mapping(
        selection.get("frozen_observation"),
        "full_state_fruit_trigger_frozen_observation_invalid",
    )
    freeze_update = threshold_update + config["freeze_lead_updates"]
    end_update = freeze_update + config["trajectory_tick_count"] - 1
    if (
        set(frozen) != observation_fields
        or frozen.get("framework_update") != freeze_update
        or frozen.get("active") is not True
        or frozen.get("active_point_pointer") != lifecycle_pointer
        or frozen.get("collecting") is not False
        or not isinstance(frozen.get("remaining_ticks"), int)
        or frozen["remaining_ticks"] <= 0
        or selection.get("slowdown_update")
        != threshold_update + config["slowdown_lead_updates"]
        or selection.get("freeze_update") != freeze_update
        or selection.get("trajectory_end_update") != end_update
        or selection.get("trajectory_tick_count")
        != config["trajectory_tick_count"]
        or binding.get("threshold_observation_update") != threshold_update
        or binding.get("freeze_update") != freeze_update
        or binding.get("trajectory_end_update") != end_update
        or expected_freeze_update != freeze_update
        or expected_end_update != end_update
    ):
        _fail("full_state_fruit_trigger_selection_invalid")


def _validate_attempts(
    attempts_path: Path,
    *,
    selected_attempt: int,
    maximum_startup_attempts: int,
    expected_process_id: int,
    expected_process_creation_filetime_100ns: int,
    probe_path: Path,
) -> tuple[tuple[Mapping[str, Any], ...], Mapping[str, Any]]:
    attempts_value = read_canonical_json(attempts_path)
    if (
        not isinstance(attempts_value, list)
        or not 1 <= len(attempts_value) <= maximum_startup_attempts
        or selected_attempt != len(attempts_value)
    ):
        _fail("full_state_attempt_selection_invalid")
    # A retry receipt does not currently attest whether failure happened before
    # gameplay state was observed.  Certifying a later attempt would therefore
    # leave room for outcome-dependent selection.  Formal full-state evidence
    # is accepted only from the first actual attempt; infrastructure races must
    # be recorded as a failed source and followed by a newly preregistered run.
    if selected_attempt != 1 or len(attempts_value) != 1:
        _fail("full_state_attempt_retry_selection_forbidden")
    rows: list[Mapping[str, Any]] = []
    previous_finish = 0
    for expected_attempt, value in enumerate(attempts_value, start=1):
        row = _mapping(value, "full_state_attempt_receipt_invalid")
        started = _integer(
            row.get("started_perf_counter_ns"),
            "full_state_attempt_receipt_invalid",
            minimum=1,
        )
        finished = _integer(
            row.get("finished_perf_counter_ns"),
            "full_state_attempt_receipt_invalid",
            minimum=1,
        )
        expected_status = (
            "PASS" if expected_attempt == selected_attempt else "RETRY"
        )
        if (
            row.get("attempt") != expected_attempt
            or row.get("status") != expected_status
            or not previous_finish <= started < finished
        ):
            _fail("full_state_attempt_receipt_invalid")
        previous_finish = finished
        rows.append(row)
    selected = rows[-1]
    if (
        selected.get("process_id") != expected_process_id
        or selected.get("process_creation_filetime_100ns")
        != expected_process_creation_filetime_100ns
        or selected.get("probe_sha256") != _sha256_path(probe_path)
    ):
        _fail("full_state_selected_attempt_binding_invalid")
    return tuple(rows), selected


def _validate_external_input_guard(
    probe: Mapping[str, Any],
    *,
    probe_path: Path,
    selected_attempt: Mapping[str, Any],
    attempt_started: int,
    attempt_finished: int,
) -> tuple[str, int, int]:
    binding = _mapping(
        probe.get("external_input_guard"),
        "full_state_external_input_guard_binding_invalid",
    )
    required = {
        "schema",
        "version",
        "status",
        "artifact",
        "artifact_bytes",
        "artifact_sha256",
        "coverage_start_perf_counter_ns",
        "coverage_end_perf_counter_ns",
        "external_event_count",
        "allowed_repaint_event_count",
    }
    if set(binding) != required:
        _fail("full_state_external_input_guard_binding_invalid")
    guard_path = _artifact_path(
        probe_path.parent,
        binding.get("artifact"),
        "full_state_external_input_guard_missing",
    )
    try:
        guard_size = guard_path.stat().st_size
        receipt = read_canonical_json(guard_path)
        validate_external_input_guard_receipt(receipt)
    except (OSError, PcExactStepEvidenceError, PcExternalInputError) as error:
        raise PcFullStateEvidenceError(
            "full_state_external_input_guard_invalid"
        ) from error
    coverage = _mapping(
        receipt.get("coverage"),
        "full_state_external_input_guard_invalid",
    )
    counts = _mapping(
        receipt.get("event_counts"),
        "full_state_external_input_guard_invalid",
    )
    guard_sha256 = _sha256_path(guard_path)
    guard_start = coverage.get("start_perf_counter_ns")
    guard_end = coverage.get("end_perf_counter_ns")
    allowed_repaint = counts.get("allowed_repaint")
    if (
        binding.get("schema") != EXTERNAL_INPUT_GUARD_BINDING_SCHEMA
        or binding.get("version") != EXTERNAL_INPUT_GUARD_BINDING_VERSION
        or binding.get("status") != "PASS"
        or binding.get("artifact_bytes") != guard_size
        or binding.get("artifact_sha256") != guard_sha256
        or binding.get("coverage_start_perf_counter_ns") != guard_start
        or binding.get("coverage_end_perf_counter_ns") != guard_end
        or binding.get("external_event_count") != 0
        or binding.get("external_event_count") != counts.get("external")
        or binding.get("allowed_repaint_event_count") != allowed_repaint
        or isinstance(allowed_repaint, bool)
        or not isinstance(allowed_repaint, int)
        or allowed_repaint < 0
        or isinstance(guard_start, bool)
        or not isinstance(guard_start, int)
        or isinstance(guard_end, bool)
        or not isinstance(guard_end, int)
        or not attempt_started <= guard_start < guard_end <= attempt_finished
        or selected_attempt.get("external_input_guard_sha256")
        != guard_sha256
        or selected_attempt.get(
            "external_input_guard_coverage_start_perf_counter_ns"
        )
        != guard_start
        or selected_attempt.get(
            "external_input_guard_coverage_end_perf_counter_ns"
        )
        != guard_end
    ):
        _fail("full_state_external_input_guard_binding_invalid")
    return guard_sha256, guard_start, guard_end


def _validate_visual_anchor(
    probe: Mapping[str, Any],
    *,
    probe_path: Path,
    process_identity: Mapping[str, Any],
    expected_process_id: int,
    expected_update: int,
    guard_start: int,
    guard_end: int,
) -> ExactStepFrameEvidence:
    repaint = _mapping(
        probe.get("repaint"),
        "full_state_repaint_receipt_invalid",
    )
    repaint_start = _integer(
        repaint.get("started_perf_counter_ns"),
        "full_state_repaint_receipt_invalid",
        minimum=1,
    )
    repaint_end = _integer(
        repaint.get("finished_perf_counter_ns"),
        "full_state_repaint_receipt_invalid",
        minimum=1,
    )
    if (
        repaint.get("schema") != "zuma-rl.pc-window-repaint-handshake"
        or repaint.get("version") != 1
        or repaint.get("process_id") != expected_process_id
        or repaint.get("window_handle_hex")
        != process_identity.get("window_handle_hex")
        or repaint.get("client_region")
        != process_identity.get("window_client_region")
        or repaint.get("geometry_restored") is not True
        or repaint.get("effectful_command_guard_satisfied") is not True
        or not guard_start <= repaint_start < repaint_end <= guard_end
    ):
        _fail("full_state_repaint_receipt_invalid")

    frozen = _mapping(
        probe.get("frozen_frame"),
        "full_state_visual_anchor_invalid",
    )
    if (
        set(frozen)
        != {"status", "artifact", "sha256", "semantics", "tool_output"}
        or frozen.get("status") != "PASS"
        or frozen.get("semantics")
        != FORMAL_FULL_STATE_SNAPSHOT_SEMANTICS
    ):
        _fail("full_state_visual_anchor_invalid")
    anchor_path = _artifact_path(
        probe_path.parent,
        frozen.get("artifact"),
        "full_state_visual_anchor_invalid",
    )
    bmp_sha256 = _sha256_path(anchor_path)
    if frozen.get("sha256") != bmp_sha256:
        _fail("full_state_visual_anchor_invalid")
    try:
        capture = json.loads(str(frozen.get("tool_output")))
    except json.JSONDecodeError as error:
        raise PcFullStateEvidenceError(
            "full_state_visual_anchor_invalid"
        ) from error
    capture = _mapping(capture, "full_state_visual_anchor_invalid")
    try:
        captured_output = _recorded_host_path(
            capture.get("output")
        ).resolve(strict=True)
    except OSError as error:
        raise PcFullStateEvidenceError(
            "full_state_visual_anchor_invalid"
        ) from error
    if (
        captured_output != anchor_path
        or capture.get("process_id") != expected_process_id
        or capture.get("window_handle_hex")
        != process_identity.get("window_handle_hex")
        or capture.get("client_region")
        != process_identity.get("window_client_region")
        or capture.get("width") != EXPECTED_WIDTH
        or capture.get("height") != EXPECTED_HEIGHT
        or capture.get("bytes") != anchor_path.stat().st_size
        or capture.get("semantics") != DIAGNOSTIC_SNAPSHOT_SEMANTICS
    ):
        _fail("full_state_visual_anchor_invalid")
    try:
        pixels = decode_top_down_bgra_bmp(anchor_path)
    except PcExactStepEvidenceError as error:
        raise PcFullStateEvidenceError(
            "full_state_visual_anchor_invalid"
        ) from error
    rgb = np.ascontiguousarray(pixels[:, :, (2, 1, 0)])
    rgb_sha256 = "sha256:" + hashlib.sha256(
        rgb.tobytes(order="C")
    ).hexdigest()
    return ExactStepFrameEvidence(
        update=expected_update,
        path=anchor_path,
        bmp_sha256=bmp_sha256,
        rgb24_sha256=rgb_sha256,
        pc_video_rgb24_sha256=canonical_rgb24_frame_sha256(rgb),
    )


def load_formal_full_state_run(
    *,
    run_id: str,
    selected_attempt: int,
    attempts_path: Path,
    probe_path: Path,
    index_path: Path,
    expected_process_id: int,
    expected_process_creation_filetime_100ns: int,
    expected_runtime_sha256: str,
    expected_dmo_sha256: str,
    expected_freeze_update: int,
    expected_start_update: int,
    expected_end_update: int,
    expected_warmup_tick_count: int,
    maximum_startup_attempts: int,
) -> FullStateLoadedRun:
    """Load and independently revalidate one formal full-state source run."""

    attempts_path = attempts_path.resolve()
    probe_path = probe_path.resolve()
    index_path = index_path.resolve()
    if (
        expected_start_update != expected_freeze_update
        or expected_warmup_tick_count != 0
        or expected_end_update < expected_start_update
    ):
        _fail("full_state_window_contract_invalid")
    try:
        attempts, selected = _validate_attempts(
            attempts_path,
            selected_attempt=selected_attempt,
            maximum_startup_attempts=maximum_startup_attempts,
            expected_process_id=expected_process_id,
            expected_process_creation_filetime_100ns=(
                expected_process_creation_filetime_100ns
            ),
            probe_path=probe_path,
        )
        probe = _mapping(
            read_canonical_json(probe_path),
            "full_state_probe_invalid",
        )
    except PcExactStepEvidenceError as error:
        raise PcFullStateEvidenceError(str(error)) from error
    process_identity = _mapping(
        probe.get("process_identity"),
        "full_state_process_identity_invalid",
    )
    attempt_started = _integer(
        selected.get("started_perf_counter_ns"),
        "full_state_attempt_receipt_invalid",
        minimum=1,
    )
    attempt_finished = _integer(
        selected.get("finished_perf_counter_ns"),
        "full_state_attempt_receipt_invalid",
        minimum=1,
    )
    guard_sha256, guard_start, guard_end = _validate_external_input_guard(
        probe,
        probe_path=probe_path,
        selected_attempt=selected,
        attempt_started=attempt_started,
        attempt_finished=attempt_finished,
    )
    if (
        probe.get("schema") != "zuma-rl.pc-memory-int32-probe"
        or probe.get("version") != 1
        or probe.get("evidence_classification")
        != FORMAL_FULL_STATE_CLASSIFICATION
        or probe.get("process_id") != expected_process_id
        or process_identity.get("process_id") != expected_process_id
        or process_identity.get("process_creation_filetime_100ns")
        != expected_process_creation_filetime_100ns
        or process_identity.get("executable_sha256")
        != expected_runtime_sha256
        or probe.get("runtime_executable_sha256")
        != expected_runtime_sha256
        or probe.get("dmo_sha256") != expected_dmo_sha256
        or probe.get("framework_update") != expected_freeze_update
        or probe.get("diagnostic_mutation") is not None
        or probe.get("diagnostic_observation") is not None
        or probe.get("diagnostic_process_affinity") is not None
        or probe.get("global_rng_call_trace") is not None
        or probe.get("live_rng_monitor") is not None
        or probe.get("gameplay_mtrand_sync") is not None
    ):
        _fail("full_state_probe_invalid")
    _validate_optional_fruit_lifecycle_trigger(
        probe,
        probe_path=probe_path,
        expected_process_id=expected_process_id,
        expected_freeze_update=expected_freeze_update,
        expected_end_update=expected_end_update,
    )
    visual = _validate_visual_anchor(
        probe,
        probe_path=probe_path,
        process_identity=process_identity,
        expected_process_id=expected_process_id,
        expected_update=expected_freeze_update,
        guard_start=guard_start,
        guard_end=guard_end,
    )

    trajectory = _mapping(
        probe.get("trajectory"),
        "full_state_probe_trajectory_binding_invalid",
    )
    if (
        trajectory.get("schema") != TRAJECTORY_SCHEMA
        or trajectory.get("version") != TRAJECTORY_VERSION
        or trajectory.get("artifact_sha256") != _sha256_path(index_path)
        or trajectory.get("freeze_update") != expected_freeze_update
        or trajectory.get("start_update") != expected_start_update
        or trajectory.get("end_update") != expected_end_update
        or trajectory.get("tick_count")
        != expected_end_update - expected_start_update + 1
    ):
        _fail("full_state_probe_trajectory_binding_invalid")
    expected_index = _artifact_path(
        probe_path.parent,
        trajectory.get("artifact"),
        "full_state_probe_trajectory_binding_invalid",
    )
    if expected_index != index_path:
        _fail("full_state_probe_trajectory_binding_invalid")

    try:
        index = _mapping(
            read_canonical_json(index_path),
            "full_state_index_invalid",
        )
        index_identity = _mapping(
            index.get("process_identity"),
            "full_state_index_process_identity_invalid",
        )
        frames = load_memory_trajectory(index_path)
    except (PcExactStepEvidenceError, PcMemoryTrajectoryError) as error:
        raise PcFullStateEvidenceError(str(error)) from error
    tick_count = expected_end_update - expected_start_update + 1
    if (
        index.get("schema") != TRAJECTORY_SCHEMA
        or index.get("version") != TRAJECTORY_VERSION
        or index.get("evidence_classification")
        != FORMAL_FULL_STATE_CLASSIFICATION
        or index.get("sample_phase") != TRAJECTORY_SAMPLE_PHASE
        or index.get("freeze_update") != expected_freeze_update
        or index.get("start_update") != expected_start_update
        or index.get("end_update") != expected_end_update
        or index.get("tick_count") != tick_count
        or index_identity != process_identity
        or len(frames) != tick_count
        or frames[0].update != expected_start_update
        or frames[-1].update != expected_end_update
    ):
        _fail("full_state_index_contract_invalid")
    try:
        raw_probe = validate_probe_payloads(
            probe_path,
            expected_runtime_sha256=expected_runtime_sha256,
            expected_dmo_sha256=expected_dmo_sha256,
            expected_update=expected_freeze_update,
            expected_score=frames[0].score,
            require_freeze_state=True,
        )
    except PcMemoryEvidenceError as error:
        raise PcFullStateEvidenceError(str(error)) from error
    if raw_probe.get("displayed_score") != frames[0].displayed_score:
        _fail("full_state_probe_trajectory_score_binding_invalid")
    return FullStateLoadedRun(
        run_id=run_id,
        selected_attempt=selected_attempt,
        process_id=expected_process_id,
        process_creation_filetime_100ns=(
            expected_process_creation_filetime_100ns
        ),
        dmo_sha256=expected_dmo_sha256,
        runtime_sha256=expected_runtime_sha256,
        freeze_update=expected_freeze_update,
        start_update=expected_start_update,
        end_update=expected_end_update,
        warmup_tick_count=expected_warmup_tick_count,
        attempts_sha256=_sha256_path(attempts_path),
        probe_sha256=_sha256_path(probe_path),
        index_sha256=_sha256_path(index_path),
        external_input_guard_sha256=guard_sha256,
        campaign_started_perf_counter_ns=_integer(
            attempts[0].get("started_perf_counter_ns"),
            "full_state_attempt_receipt_invalid",
            minimum=1,
        ),
        attempt_finished_perf_counter_ns=attempt_finished,
        frames=frames,
        visual_frames=(visual,),
    )


__all__ = [
    "FORMAL_FULL_STATE_CLASSIFICATION",
    "FORMAL_FULL_STATE_SNAPSHOT_SEMANTICS",
    "FullStateLoadedRun",
    "PcFullStateEvidenceError",
    "load_formal_full_state_run",
]
