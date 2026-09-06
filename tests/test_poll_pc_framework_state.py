from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from tools import poll_pc_framework_state as poller
from tools.capture_dxgi import FrameworkStateSnapshot


_SHA = "sha256:" + "ab" * 32


def _state(update: int, draw: int, **changes: object) -> FrameworkStateSnapshot:
    value = FrameworkStateSnapshot(
        non_draw_count=0,
        frame_time_ms=10,
        is_drawing=False,
        last_draw_was_empty=False,
        has_pending_draw=False,
        pending_updates_acc=0.0,
        update_f_time_acc=0.0,
        last_time_check=100,
        last_time=90,
        last_user_input_tick=80,
        sleep_count=10 + update,
        draw_count=draw,
        update_count=update,
        update_app_state=3,
        update_app_depth=1,
        update_multiplier=1.0,
        paused=False,
        fast_forward_target=0,
        fast_forward_to_marker=False,
        fast_forward_step=False,
        last_draw_tick=draw,
        next_draw_tick=draw + 1,
        step_mode=0,
    )
    return replace(value, **changes)


class _Reader:
    def __init__(self, states: list[FrameworkStateSnapshot]) -> None:
        self.states = states
        self.calls = 0

    def sample_state(self) -> FrameworkStateSnapshot:
        result = self.states[min(self.calls, len(self.states) - 1)]
        self.calls += 1
        return result


class _Clock:
    def __init__(self) -> None:
        self.value = 1_000_000

    def __call__(self) -> int:
        self.value += 100
        return self.value


def test_poll_retains_every_draw_transition_and_timing_bracket() -> None:
    reader = _Reader(
        [
            _state(50, 70),
            _state(51, 70, has_pending_draw=True),
            _state(51, 71),
        ]
    )
    ready: list[tuple[int, int, int]] = []
    result = poller.poll_framework_state(
        reader,
        stop_requested=lambda: reader.calls >= 3,
        on_ready=lambda before, after, state: ready.append(
            (before, after, state.draw_count)
        ),
        clock_ns=_Clock(),
        control_check_interval=1,
    )

    assert ready == [(1_000_100, 1_000_200, 70)]
    assert result.poll_count == 3
    assert len(result.transitions) == 2
    assert len(result.draw_events) == 1
    event = result.draw_events[0]
    assert event.previous_draw_count == 70
    assert event.draw_count == 71
    assert event.draw_count_delta == 1
    assert event.previous_sample_after_perf_counter_ns < event.sample_after_perf_counter_ns


def test_poll_rejects_counter_regression() -> None:
    reader = _Reader([_state(50, 70), _state(49, 70)])
    with pytest.raises(
        poller.FrameworkPollError,
        match="framework_poll_counter_decreased",
    ):
        poller.poll_framework_state(
            reader,
            stop_requested=lambda: False,
            on_ready=lambda *unused: None,
            clock_ns=_Clock(),
            control_check_interval=1,
        )


def test_poll_arms_only_at_exact_stable_update_and_schedules_rate() -> None:
    reader = _Reader(
        [
            _state(49, 69),
            _state(50, 69, has_pending_draw=True),
            _state(50, 70),
            _state(51, 71),
        ]
    )
    result = poller.poll_framework_state(
        reader,
        stop_requested=lambda: reader.calls >= 4,
        on_ready=lambda *unused: None,
        arm_at_framework_update=50,
        target_poll_hz=5_000,
        clock_ns=_Clock(),
        sleep=lambda unused: None,
        control_check_interval=1,
    )

    assert result.ready_state.update_count == 49
    assert result.initial_state.update_count == 50
    assert result.initial_state.has_pending_draw is False
    assert result.prearm_sample_count == 3
    assert result.arm_at_framework_update == 50
    assert result.arm_no_later_than_framework_update == 50
    assert result.armed_at_framework_update == 50
    assert result.target_poll_hz == 5_000.0


def test_poll_arms_at_first_stable_update_inside_frozen_window() -> None:
    reader = _Reader(
        [
            _state(49, 69),
            _state(50, 69, has_pending_draw=True),
            _state(51, 70),
            _state(52, 71),
        ]
    )
    result = poller.poll_framework_state(
        reader,
        stop_requested=lambda: reader.calls >= 4,
        on_ready=lambda *unused: None,
        arm_at_framework_update=50,
        arm_no_later_than_framework_update=52,
        target_poll_hz=5_000,
        clock_ns=_Clock(),
        sleep=lambda unused: None,
        control_check_interval=1,
    )

    assert result.initial_state.update_count == 51
    assert result.arm_at_framework_update == 50
    assert result.arm_no_later_than_framework_update == 52
    assert result.armed_at_framework_update == 51


def test_affinity_topology_rejects_smt_sibling_and_accepts_other_core() -> None:
    with pytest.raises(
        poller.FrameworkPollError,
        match="framework_poll_affinity_physical_core_overlap",
    ):
        poller.validate_disjoint_physical_affinity(
            observer_mask=2,
            game_mask=1,
            core_masks=(3, 12),
        )

    evidence = poller.validate_disjoint_physical_affinity(
        observer_mask=4,
        game_mask=1,
        core_masks=(3, 12),
    )
    assert evidence["physical_cores_disjoint"] is True
    assert evidence["observer_physical_core_logical_mask"] == 12


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")


def _write_json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical(value)
    path.write_bytes(payload)
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def test_build_report_binds_capture_and_marks_skipped_draws(
    tmp_path: Path,
) -> None:
    identity = {
        "process_id": 7,
        "process_creation_filetime_100ns": 77,
        "executable_path": "D:\\runtime.exe",
        "executable_sha256": _SHA,
    }
    ready_path = tmp_path / "ready.json"
    stop_path = tmp_path / "stop.json"
    _write_json(
        ready_path,
        {
            "schema": poller.READY_SCHEMA,
            "version": 1,
            "process_id": 7,
            "process_creation_filetime_100ns": 77,
            "executable_sha256": _SHA,
        },
    )
    _write_json(
        stop_path,
        {
            "schema": poller.STOP_SCHEMA,
            "version": 1,
            "process_id": 7,
            "requested_perf_counter_ns": 900,
        },
    )
    metadata_path = tmp_path / "capture" / "metadata.json"
    metadata_sha = _write_json(
        metadata_path,
        {
            "schema": "zuma-rl.dxgi-bgra-capture",
            "status": "acquisition_complete",
            "capture_start_perf_counter_ns": 300,
            "capture_end_perf_counter_ns": 800,
            "target_identity": {
                "process_id": 7,
                "process_creation_filetime_100ns": 77,
                "executable_sha256": _SHA,
            },
        },
    )
    update_path = tmp_path / "updates.json"
    update_sha = _write_json(
        update_path,
        {
            "schema": "zuma-rl.pc-framework-update-map",
            "version": 1,
            "capture_metadata_sha256": metadata_sha,
            "process_id": 7,
            "process_creation_filetime_100ns": 77,
            "executable_sha256": _SHA,
        },
    )
    state_path = tmp_path / "state.json"
    _write_json(
        state_path,
        {
            "schema": "zuma-rl.pc-framework-state-diagnostic",
            "version": 1,
            "diagnostic_only": True,
            "capture_metadata_sha256": metadata_sha,
            "framework_update_map_sha256": update_sha,
            "process_id": 7,
            "process_creation_filetime_100ns": 77,
            "executable_sha256": _SHA,
        },
    )
    event = poller.DrawEvent(
        event_index=0,
        transition_index=0,
        poll_index=2,
        previous_draw_count=70,
        draw_count=72,
        draw_count_delta=2,
        update_count=51,
        previous_sample_after_perf_counter_ns=400,
        sample_before_perf_counter_ns=401,
        sample_after_perf_counter_ns=402,
    )
    poll = poller.PollResult(
        ready_sample_before_perf_counter_ns=100,
        ready_sample_after_perf_counter_ns=110,
        ready_state=_state(50, 70),
        prearm_sample_count=1,
        arm_at_framework_update=None,
        target_poll_hz=None,
        poll_start_perf_counter_ns=100,
        poll_end_perf_counter_ns=1000,
        poll_count=3,
        initial_sample_before_perf_counter_ns=100,
        initial_sample_after_perf_counter_ns=110,
        initial_state=_state(50, 70),
        final_state=_state(51, 72),
        transitions=(),
        draw_events=(event,),
        sample_latency_ns=(10, 11, 12),
        sample_completion_interval_ns=(20, 21),
    )

    report = poller.build_report(
        identity=identity,
        affinity_mask=4,
        affinity_topology={
            "observer_logical_affinity_mask": 4,
            "observer_physical_core_logical_mask": 12,
            "game_logical_affinity_mask": 1,
            "game_physical_core_logical_mask": 3,
            "physical_cores_disjoint": True,
            "observed_physical_core_count": 2,
        },
        ready_marker_path=ready_path,
        stop_marker_path=stop_path,
        capture_metadata_path=metadata_path,
        update_map_path=update_path,
        state_sidecar_path=state_path,
        poll=poll,
    )

    assert report["classification"] == "diagnostic-only-not-pc-golden"
    assert report["gate_effect"] == "none"
    assert report["status"] == "complete_with_observed_draw_counter_gaps"
    assert report["skipped_draw_increment_count"] == 1
    assert report["capture_binding"]["capture_interval_fully_observed"] is True
