from __future__ import annotations

from pathlib import Path
from dataclasses import replace
import json
import struct
from types import SimpleNamespace

import pytest

from tools import collect_pc_memory_probe as collector
from tools.collect_pc_memory_probe import (
    BALL_OBJECT_SIZE,
    BALL_VTABLE,
    BULLET_OBJECT_SIZE,
    BULLET_VTABLE,
    CURVE_ADD_PLAN_ENABLED_OFFSET,
    CURVE_DUMP_SIZE,
    CURVE_PLANNED_VECTOR_BEGIN_OFFSET,
    MAX_POINTER_REFERENCE_TARGETS,
    ProbeError,
    _bounded_reference_targets,
    _collect_bullet_gap_state,
    _collect_curve_plan,
    _curve_payload_layout,
    _decode_bullet_subclass,
    _parser,
    _replay_state_row,
    _natural_exact_freeze,
    _trajectory_sample_barrier,
    _trajectory_tick_summary,
    _write_canonical_attempts,
    _write_canonical_probe,
    collect_board_trajectory,
)
from zuma_rl.pc_memory_evidence import canonical_report_bytes
from tools.inspect_popcap_replay import ReplayState


def _state(update: int = 100) -> ReplayState:
    return ReplayState(
        multiplier_address=0x12345678,
        non_draw_count=0,
        frame_time_ms=10,
        sleep_count=1,
        draw_count=2,
        update_count=update,
        update_app_state=0,
        update_app_depth=0,
        update_multiplier=0.0877914951989026,
        paused=False,
        fast_forward_target=0,
        fast_forward_to_marker=False,
        fast_forward_step=False,
        step_mode=0,
        loading_thread_started=True,
        loading_thread_completed=True,
        loaded=True,
    )


def test_probe_writer_uses_platform_independent_canonical_bytes(
    tmp_path: Path,
) -> None:
    probe = {"schema": "example", "nested": {"value": 3}}
    path = tmp_path / "memory-probe.json"

    _write_canonical_probe(path, probe)

    assert path.read_bytes() == canonical_report_bytes(probe)
    assert b"\r\n" not in path.read_bytes()


def test_attempt_writer_uses_platform_independent_canonical_bytes(
    tmp_path: Path,
) -> None:
    attempts = [
        {
            "status": "PASS",
            "attempt": 1,
            "process_id": 123,
        }
    ]
    path = tmp_path / "attempts.json"

    _write_canonical_attempts(path, attempts)

    assert path.read_bytes() == (
        b'[{"attempt":1,"process_id":123,"status":"PASS"}]\n'
    )
    assert b"\r\n" not in path.read_bytes()


def test_global_rng_trace_timeout_cannot_be_bound_as_complete(
    tmp_path: Path,
) -> None:
    result = tmp_path / "trace.json"
    result.write_text(
        json.dumps(
            {
                "schema": "zuma-rl.pc-global-mtrand-call-trace",
                "version": 1,
                "status": "PASS",
                "stop_reason": "timeout",
                "start_update": 10,
                "end_update": 20,
                "call_count": 0,
                "calls": [],
            }
        ),
        encoding="ascii",
    )
    stop = tmp_path / "trace.stop"
    log_path = tmp_path / "trace.log"
    log_path.write_bytes(b"")

    class CompletedProcess:
        @staticmethod
        def wait(*, timeout: float) -> int:
            assert timeout == 30
            return 0

    class LogStream:
        closed = False

        def close(self) -> None:
            self.closed = True

    log_stream = LogStream()
    with pytest.raises(
        ProbeError,
        match="global_rng_call_trace_contract_invalid",
    ):
        collector._finish_global_rng_call_trace(
            process=CompletedProcess(),  # type: ignore[arg-type]
            log_stream=log_stream,
            result_path=result,
            stop_path=stop,
            log_path=log_path,
            start_update=10,
            end_update=20,
        )
    assert log_stream.closed


def test_global_rng_trace_launch_uses_declared_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempt_root = tmp_path / "attempt"
    attempt_root.mkdir()
    (attempt_root / "global-rng-call-trace.ready.json").write_text(
        "{}\n",
        encoding="ascii",
    )
    captured: dict[str, object] = {}

    class RunningProcess:
        returncode = None

        @staticmethod
        def poll() -> None:
            return None

    def fake_popen(argv: list[str], **kwargs: object) -> RunningProcess:
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return RunningProcess()

    monkeypatch.setattr(collector.subprocess, "Popen", fake_popen)
    bundle = collector._start_global_rng_call_trace(
        project_root=tmp_path,
        pid=123,
        runtime_executable=tmp_path / "popcapgame1.exe",
        attempt_root=attempt_root,
        start_update=3768,
        end_update=3910,
        timeout_seconds=900.0,
    )
    try:
        argv = captured["argv"]
        assert isinstance(argv, list)
        timeout_index = argv.index("--timeout")
        assert argv[timeout_index + 1] == "900.0"
    finally:
        bundle[1].close()


def test_global_rng_trace_timeout_must_be_positive(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ProbeError,
        match="global_rng_call_trace_timeout_invalid",
    ):
        collector.collect_probe(
            plan_path=tmp_path / "plan.json",
            prestate_path=tmp_path / "prestate.json",
            host_restore_path=tmp_path / "host.json",
            output_root=tmp_path / "out",
            probe_update=2,
            slowdown_update=1,
            int32_value=0,
            maximum_attempts=1,
            global_rng_call_trace_timeout_seconds=0.0,
        )


def test_formal_independent_scores_require_one_attempt(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ProbeError,
        match="formal_independent_score_contract_invalid",
    ):
        collector.collect_probe(
            plan_path=tmp_path / "plan.json",
            prestate_path=tmp_path / "prestate.json",
            host_restore_path=tmp_path / "host.json",
            output_root=tmp_path / "out",
            probe_update=100,
            slowdown_update=90,
            int32_value=None,
            maximum_attempts=2,
            trajectory_end_update=100,
            formal_full_state_evidence=True,
            allow_discovered_display_mismatch=True,
            formal_independent_score_binding=True,
        )


def test_formal_fruit_trigger_contract_is_bounded() -> None:
    normalized = collector._normalize_formal_fruit_lifecycle_trigger(
        {
            "monitor_start_update": 100,
            "maximum_framework_update": 1000,
            "remaining_threshold_ticks": 384,
            "slowdown_lead_updates": 16,
            "freeze_lead_updates": 64,
            "trajectory_tick_count": 512,
            "poll_interval_seconds": 0.002,
        }
    )

    assert normalized["trajectory_tick_count"] == 512
    assert normalized["remaining_threshold_ticks"] == 384


def test_formal_fruit_trigger_rejects_suffix_too_short() -> None:
    with pytest.raises(
        ProbeError,
        match="formal_fruit_trigger_contract_invalid",
    ):
        collector._normalize_formal_fruit_lifecycle_trigger(
            {
                "monitor_start_update": 100,
                "maximum_framework_update": 1000,
                "remaining_threshold_ticks": 384,
                "slowdown_lead_updates": 16,
                "freeze_lead_updates": 64,
                "trajectory_tick_count": 384,
                "poll_interval_seconds": 0.002,
            }
        )


def test_natural_exact_freeze_uses_completed_native_step_barrier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preliminary = replace(
        _state(98),
        update_multiplier=collector.expected_multiplier(6),
    )
    observed_minus_counts: list[int] = []
    current_minus_count = 6

    monkeypatch.setattr(
        collector,
        "freeze_replay",
        lambda **kwargs: (
            preliminary
            if kwargs["freeze_update"] == 98
            else pytest.fail("wrong natural freeze trigger")
        ),
    )
    monkeypatch.setattr(collector, "main_window_for_pid", lambda pid: 77)
    monkeypatch.setattr(collector, "open_process_readonly", lambda pid: 88)
    monkeypatch.setattr(collector, "close_process", lambda handle: None)

    def fake_post(hwnd: int, character: str) -> None:
        nonlocal current_minus_count
        assert hwnd == 77
        assert character == "-"
        current_minus_count += 1
        observed_minus_counts.append(current_minus_count)

    monkeypatch.setattr(collector, "post_char", fake_post)
    monkeypatch.setattr(
        collector,
        "read_replay_state",
        lambda handle, address: replace(
            preliminary,
            update_count=99 if current_minus_count >= 10 else 98,
            update_multiplier=collector.expected_multiplier(
                current_minus_count
            ),
        ),
    )

    def fake_step(
        hwnd: int,
        handle: int,
        address: int,
        target: int,
        *,
        timeout_per_step: float,
    ) -> ReplayState:
        assert (hwnd, handle, address, target) == (
            77,
            88,
            preliminary.multiplier_address,
            100,
        )
        assert timeout_per_step == 5.0
        return replace(
            preliminary,
            update_count=100,
            update_multiplier=collector.expected_multiplier(12),
            fast_forward_target=100,
        )

    monkeypatch.setattr(collector, "step_to", fake_step)

    state = _natural_exact_freeze(
        pid=123,
        slowdown_update=90,
        probe_update=100,
        demo_length=1000,
        timeout=30.0,
    )

    assert observed_minus_counts == [7, 8, 9, 10, 11, 12]
    assert state.update_count == 100
    assert state.fast_forward_target == 100


def test_source_bound_formal_replay_uses_natural_exact_freeze() -> None:
    assert collector._uses_natural_exact_freeze(
        collector.DIRECT_FIXED_SEED_LAUNCH_MODE,
        source_bound_formal_replay=True,
    )
    assert not collector._uses_natural_exact_freeze(
        collector.DIRECT_FIXED_SEED_LAUNCH_MODE,
        source_bound_formal_replay=False,
    )


def test_light_ball_identity_binds_exact_render_driving_float_bits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = bytearray(collector.LIGHT_BALL_IDENTITY_BYTES)
    struct.pack_into("<I", payload, 0x00, BALL_VTABLE)
    struct.pack_into("<I", payload, 0x10, 77)
    struct.pack_into("<i", payload, 0x14, 3)
    render_floats = (
        125.5,
        0.25,
        0.125,
        -0.03125,
        400.5,
        588.25,
        1.0,
        16.0,
    )
    struct.pack_into("<8f", payload, 0x1C, *render_floats)
    payload[0xB4:0xC3] = bytes(range(15))
    struct.pack_into("<i", payload, 0xC8, 14)
    struct.pack_into("<i", payload, 0x11C, 2)
    struct.pack_into("<i", payload, 0x120, 14)
    monkeypatch.setattr(
        collector,
        "read_process_bytes",
        lambda handle, address, size: (
            bytes(payload)
            if (handle, address, size)
            == (8, 0x12340000, collector.LIGHT_BALL_IDENTITY_BYTES)
            else b""
        ),
    )

    identity = collector._read_light_ball_identity(8, 0x12340000)

    assert identity is not None
    assert identity["ball_id"] == 77
    assert identity["position_x"] == 400.5
    assert identity["position_y"] == 588.25
    assert identity["orientation_radians"] == 0.25
    assert identity["render_float32_bits_hex"] == (
        struct.pack("<8f", *render_floats).hex()
    )
    assert identity["flags_b4_c2_hex"] == bytes(range(15)).hex()


def test_trajectory_tick_summary_preserves_all_curve_list_counts() -> None:
    board = {
        "score": 100,
        "displayed_score": 90,
        "score_target": 1250,
        "primary_child": {
            "bullets": [
                {"ball": {"ball_id": 11, "color_id": 2}},
                {"ball": {"ball_id": 12, "color_id": 3}},
            ]
        },
        "fired_bullets": {"traversed_count": 1},
        "curve_manager": {
            "curves": [
                {
                    "index": 0,
                    "intrusive_lists": [
                        {"container_offset": 0x50, "traversed_count": 1},
                        {"container_offset": 0x5C, "traversed_count": 94},
                        {"container_offset": 0x68, "traversed_count": 2},
                    ],
                },
                {
                    "index": 1,
                    "intrusive_lists": [
                        {"container_offset": 0x50, "traversed_count": 0},
                        {"container_offset": 0x5C, "traversed_count": 7},
                        {"container_offset": 0x68, "traversed_count": 0},
                    ],
                },
            ]
        },
    }

    summary = _trajectory_tick_summary(100, board)

    assert summary["chain_ball_count"] == 101
    assert summary["fired_bullet_count"] == 1
    assert summary["current_ball_id"] == 11
    assert summary["next_color_id"] == 3
    assert len(summary["list_counts"]) == 6


def test_probe_parser_keeps_score_checks_separate_from_auxiliary_scan() -> None:
    args = _parser().parse_args(
        [
            "--plan",
            "plan.json",
            "--prestate",
            "pre.json",
            "--host-restore",
            "restore.json",
            "--output-root",
            "out",
            "--int32-value",
            "8120",
            "--displayed-score-value",
            "7950",
            "--int32-scan-value",
            "7950",
        ]
    )

    assert args.int32_value == 8120
    assert args.displayed_score_value == 7950
    assert args.int32_scan_value == 7950


def test_probe_parser_exposes_read_only_score_discovery() -> None:
    args = _parser().parse_args(
        [
            "--plan",
            "plan.json",
            "--prestate",
            "pre.json",
            "--host-restore",
            "restore.json",
            "--output-root",
            "out",
            "--discover-score",
        ]
    )

    assert args.discover_score is True
    assert args.int32_value == 7950
    assert args.displayed_score_value is None
    assert args.int32_scan_value is None


def test_probe_parser_exposes_diagnostic_rolling_score_discovery() -> None:
    args = _parser().parse_args(
        [
            "--plan",
            "plan.json",
            "--prestate",
            "pre.json",
            "--host-restore",
            "restore.json",
            "--output-root",
            "out",
            "--diagnostic-discover-rolling-score",
            "--trajectory-end-update",
            "8000",
            "--skip-repaint-guard",
            "--skip-frozen-snapshot",
        ]
    )

    assert args.discover_score is False
    assert args.diagnostic_discover_rolling_score is True


def test_probe_parser_exposes_exact_trajectory_snapshots() -> None:
    args = _parser().parse_args(
        [
            "--plan",
            "plan.json",
            "--prestate",
            "pre.json",
            "--host-restore",
            "restore.json",
            "--output-root",
            "out",
            "--trajectory-end-update",
            "8000",
            "--trajectory-mode",
            "rng",
            "--skip-repaint-guard",
            "--skip-frozen-snapshot",
            "--snapshot-trajectory-frames",
            "--snapshot-square-dwm-corners",
            "--snapshot-trajectory-warmup-frame",
            "--formal-exact-step-evidence",
        ]
    )

    assert args.snapshot_trajectory_frames is True
    assert args.snapshot_square_dwm_corners is True
    assert args.snapshot_trajectory_warmup_frame is True
    assert args.formal_exact_step_evidence is True


def test_probe_parser_exposes_formal_full_state_source() -> None:
    args = _parser().parse_args(
        [
            "--plan",
            "plan.json",
            "--prestate",
            "pre.json",
            "--host-restore",
            "restore.json",
            "--output-root",
            "out",
            "--trajectory-end-update",
            "8000",
            "--trajectory-mode",
            "full",
            "--formal-full-state-evidence",
        ]
    )

    assert args.formal_full_state_evidence is True
    assert args.formal_exact_step_evidence is False


def test_probe_parser_exposes_diagnostic_source_bound_replay() -> None:
    args = _parser().parse_args(
        [
            "--plan",
            "plan.json",
            "--prestate",
            "pre.json",
            "--host-restore",
            "restore.json",
            "--output-root",
            "out",
            "--trajectory-end-update",
            "8000",
            "--trajectory-mode",
            "rng",
            "--trajectory-step-timeout-seconds",
            "300",
            "--maximum-attempts",
            "1",
            "--diagnostic-source-bound-replay",
        ]
    )

    assert args.diagnostic_source_bound_replay is True
    assert args.trajectory_step_timeout_seconds == 300.0
    assert args.formal_full_state_evidence is False
    assert args.formal_exact_step_evidence is False


def test_main_forwards_diagnostic_source_bound_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_collect_probe(**kwargs: object) -> Path:
        captured.update(kwargs)
        return tmp_path / "out" / "attempt-01" / "memory-probe.json"

    monkeypatch.setattr(collector, "collect_probe", fake_collect_probe)

    exit_code = collector.main(
        [
            "--plan",
            str(tmp_path / "plan.json"),
            "--prestate",
            str(tmp_path / "pre.json"),
            "--host-restore",
            str(tmp_path / "restore.json"),
            "--output-root",
            str(tmp_path / "out"),
            "--probe-update",
            "100",
            "--slowdown-update",
            "90",
            "--trajectory-end-update",
            "110",
            "--trajectory-mode",
            "rng",
            "--trajectory-step-timeout-seconds",
            "300",
            "--maximum-attempts",
            "1",
            "--diagnostic-source-bound-replay",
        ]
    )

    assert exit_code == 0
    assert captured["diagnostic_source_bound_replay"] is True
    assert captured["trajectory_step_timeout_seconds"] == 300.0
    assert captured["formal_exact_step_evidence"] is False
    assert captured["formal_full_state_evidence"] is False


def test_probe_parser_exposes_formal_independent_scores() -> None:
    args = _parser().parse_args(
        [
            "--plan",
            "plan.json",
            "--prestate",
            "pre.json",
            "--host-restore",
            "restore.json",
            "--output-root",
            "out",
            "--formal-discover-independent-scores",
            "--formal-full-state-evidence",
            "--maximum-attempts",
            "1",
            "--trajectory-end-update",
            "8000",
        ]
    )

    assert args.formal_discover_independent_scores is True
    assert args.discover_score is False
    assert args.diagnostic_discover_rolling_score is False


def test_main_forwards_formal_independent_scores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_collect_probe(**kwargs: object) -> Path:
        captured.update(kwargs)
        return tmp_path / "out" / "attempt-01" / "memory-probe.json"

    monkeypatch.setattr(collector, "collect_probe", fake_collect_probe)

    exit_code = collector.main(
        [
            "--plan",
            str(tmp_path / "plan.json"),
            "--prestate",
            str(tmp_path / "pre.json"),
            "--host-restore",
            str(tmp_path / "restore.json"),
            "--output-root",
            str(tmp_path / "out"),
            "--probe-update",
            "100",
            "--slowdown-update",
            "90",
            "--trajectory-end-update",
            "100",
            "--maximum-attempts",
            "1",
            "--formal-full-state-evidence",
            "--formal-discover-independent-scores",
        ]
    )

    assert exit_code == 0
    assert captured["int32_value"] is None
    assert captured["allow_discovered_display_mismatch"] is True
    assert captured["formal_independent_score_binding"] is True


def test_probe_parser_exposes_unrecorded_trajectory_warmup() -> None:
    args = _parser().parse_args(
        [
            "--plan",
            "plan.json",
            "--prestate",
            "pre.json",
            "--host-restore",
            "restore.json",
            "--output-root",
            "out",
            "--probe-update",
            "3152",
            "--trajectory-start-update",
            "3429",
            "--trajectory-end-update",
            "3438",
            "--trajectory-mode",
            "rng",
            "--skip-repaint-guard",
            "--skip-frozen-snapshot",
        ]
    )

    assert args.trajectory_start_update == 3429


def test_probe_parser_exposes_live_rng_monitor() -> None:
    args = _parser().parse_args(
        [
            "--plan",
            "plan.json",
            "--prestate",
            "pre.json",
            "--host-restore",
            "restore.json",
            "--output-root",
            "out",
            "--trajectory-end-update",
            "8000",
            "--skip-repaint-guard",
            "--skip-frozen-snapshot",
            "--live-rng-monitor-interval-seconds",
            "0.002",
        ]
    )

    assert args.live_rng_monitor_interval_seconds == 0.002


def test_probe_parser_exposes_gameplay_mtrand_sync() -> None:
    args = _parser().parse_args(
        [
            "--plan",
            "plan.json",
            "--prestate",
            "pre.json",
            "--host-restore",
            "restore.json",
            "--output-root",
            "out",
            "--trajectory-end-update",
            "8000",
            "--skip-repaint-guard",
            "--skip-frozen-snapshot",
            "--gameplay-mtrand-oracle",
            "oracle.json",
            "--gameplay-mtrand-oracle-seed",
            "85961375",
            "--gameplay-mtrand-oracle-maximum-draws",
            "20000",
        ]
    )

    assert args.gameplay_mtrand_oracle == Path("oracle.json")
    assert args.gameplay_mtrand_oracle_seed == 85961375
    assert args.gameplay_mtrand_oracle_maximum_draws == 20000


def test_probe_parser_exposes_global_mtrand_call_sync() -> None:
    args = _parser().parse_args(
        [
            "--plan",
            "plan.json",
            "--prestate",
            "pre.json",
            "--host-restore",
            "restore.json",
            "--output-root",
            "out",
            "--trajectory-end-update",
            "3500",
            "--skip-repaint-guard",
            "--skip-frozen-snapshot",
            "--global-mtrand-call-oracle",
            "global-calls.json",
            "--global-mtrand-call-oracle-seed",
            "23557968",
            "--global-mtrand-call-start-after-update",
            "1702",
            "--global-mtrand-call-end-at-update",
            "3500",
            "--global-mtrand-call-maximum-draws",
            "20000",
            "--global-mtrand-call-sync-timeout-seconds",
            "600",
            "--initial-global-mtrand-oracle",
            "global-calls.json",
            "--initial-global-mtrand-seed",
            "23557968",
            "--diagnostic-disable-source-bound-board-anchor",
        ]
    )

    assert args.global_mtrand_call_oracle == Path("global-calls.json")
    assert args.global_mtrand_call_oracle_seed == 23557968
    assert args.global_mtrand_call_start_after_update == 1702
    assert args.global_mtrand_call_end_at_update == 3500
    assert args.global_mtrand_call_maximum_draws == 20000
    assert args.global_mtrand_call_sync_timeout_seconds == 600.0
    assert args.initial_global_mtrand_oracle == Path("global-calls.json")
    assert args.initial_global_mtrand_seed == 23557968
    assert args.diagnostic_disable_source_bound_board_anchor is True


def test_probe_parser_exposes_global_rng_call_trace_timeout() -> None:
    args = _parser().parse_args(
        [
            "--plan",
            "plan.json",
            "--prestate",
            "pre.json",
            "--host-restore",
            "restore.json",
            "--output-root",
            "out",
            "--trace-global-rng-calls",
            "--global-rng-call-trace-timeout-seconds",
            "900",
        ]
    )

    assert args.global_rng_call_trace_timeout_seconds == 900.0


def test_probe_parser_exposes_read_only_global_boundary_observer() -> None:
    args = _parser().parse_args(
        [
            "--plan",
            "plan.json",
            "--prestate",
            "pre.json",
            "--host-restore",
            "restore.json",
            "--output-root",
            "out",
            "--global-mtrand-call-oracle",
            "global-calls.json",
            "--global-mtrand-call-oracle-seed",
            "23557968",
            "--global-mtrand-call-start-after-update",
            "1702",
            "--global-mtrand-call-end-at-update",
            "3121",
            "--global-mtrand-call-observe-boundary-only",
            "--global-mtrand-call-observe-thread-runtime",
            "--global-mtrand-call-observe-handoff-thread-runtime",
            "--global-mtrand-call-observe-handoff-rng-state",
            "--global-mtrand-call-handoff-rng-wait-target-words-sha256",
            "sha256:" + ("ab" * 32),
            "--global-mtrand-call-handoff-rng-wait-min-index",
            "315",
            "--global-mtrand-call-handoff-rng-wait-timeout-seconds",
            "0.25",
            "--global-mtrand-call-handoff-rng-wait-poll-interval-seconds",
            "0.001",
        ]
    )

    assert args.global_mtrand_call_observe_boundary_only is True
    assert args.global_mtrand_call_observe_thread_runtime is True
    assert args.global_mtrand_call_observe_handoff_thread_runtime is True
    assert args.global_mtrand_call_observe_handoff_rng_state is True
    assert (
        args.global_mtrand_call_handoff_rng_wait_target_words_sha256
        == "sha256:" + ("ab" * 32)
    )
    assert args.global_mtrand_call_handoff_rng_wait_min_index == 315
    assert args.global_mtrand_call_handoff_rng_wait_timeout_seconds == 0.25
    assert (
        args.global_mtrand_call_handoff_rng_wait_poll_interval_seconds
        == 0.001
    )


def test_probe_parser_exposes_startup_global_mtrand_observation() -> None:
    args = _parser().parse_args(
        [
            "--plan",
            "plan.json",
            "--prestate",
            "pre.json",
            "--host-restore",
            "host.json",
            "--output-root",
            "output",
            "--startup-global-mtrand-observation-oracle",
            "global-calls.json",
            "--startup-global-mtrand-observation-seed",
            "23557968",
            "--startup-global-mtrand-observation-end-at-update",
            "1702",
            "--startup-global-mtrand-observation-maximum-draws",
            "20000",
            "--startup-global-mtrand-hidden-draw-start-after-source-order",
            "308",
            "--startup-global-mtrand-hidden-draw-stop-before-source-order",
            "309",
            "--diagnostic-disable-source-bound-board-anchor",
        ]
    )

    assert args.startup_global_mtrand_observation_oracle == Path(
        "global-calls.json"
    )
    assert args.startup_global_mtrand_observation_seed == 23557968
    assert args.startup_global_mtrand_observation_end_at_update == 1702
    assert args.startup_global_mtrand_observation_maximum_draws == 20000
    assert (
        args.startup_global_mtrand_hidden_draw_start_after_source_order == 308
    )
    assert (
        args.startup_global_mtrand_hidden_draw_stop_before_source_order == 309
    )


def test_main_forwards_startup_global_mtrand_observation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oracle = tmp_path / "global-calls.json"
    oracle.write_text("{}\n", encoding="ascii")
    captured: dict[str, object] = {}

    def fake_collect_probe(**kwargs: object) -> Path:
        captured.update(kwargs)
        return tmp_path / "attempt-01" / "memory-probe.json"

    monkeypatch.setattr(collector, "collect_probe", fake_collect_probe)

    exit_code = collector.main(
        [
            "--plan",
            str(tmp_path / "plan.json"),
            "--prestate",
            str(tmp_path / "pre.json"),
            "--host-restore",
            str(tmp_path / "restore.json"),
            "--output-root",
            str(tmp_path / "out"),
            "--probe-update",
            "3425",
            "--slowdown-update",
            "3415",
            "--trajectory-end-update",
            "3425",
            "--trajectory-mode",
            "rng",
            "--skip-repaint-guard",
            "--skip-frozen-snapshot",
            "--startup-global-mtrand-observation-oracle",
            str(oracle),
            "--startup-global-mtrand-observation-seed",
            "23557968",
            "--startup-global-mtrand-observation-end-at-update",
            "1702",
            "--startup-global-mtrand-observation-maximum-draws",
            "20000",
            "--startup-global-mtrand-hidden-draw-start-after-source-order",
            "308",
            "--startup-global-mtrand-hidden-draw-stop-before-source-order",
            "309",
            "--diagnostic-disable-source-bound-board-anchor",
        ]
    )

    assert exit_code == 0
    assert captured[
        "startup_global_mtrand_observation_oracle"
    ] == oracle.resolve()
    assert captured["startup_global_mtrand_observation_seed"] == 23557968
    assert captured[
        "startup_global_mtrand_observation_end_at_update"
    ] == 1702
    assert captured[
        "startup_global_mtrand_observation_maximum_draws"
    ] == 20000
    assert captured[
        "startup_global_mtrand_hidden_draw_start_after_source_order"
    ] == 308
    assert captured[
        "startup_global_mtrand_hidden_draw_stop_before_source_order"
    ] == 309
    assert captured["diagnostic_disable_source_bound_board_anchor"] is True


def test_main_forwards_bounded_global_mtrand_call_sync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oracle = tmp_path / "global-calls.json"
    oracle.write_text("{}\n", encoding="ascii")
    captured: dict[str, object] = {}

    def fake_collect_probe(**kwargs: object) -> Path:
        captured.update(kwargs)
        return tmp_path / "attempt-01" / "memory-probe.json"

    monkeypatch.setattr(collector, "collect_probe", fake_collect_probe)

    exit_code = collector.main(
        [
            "--plan",
            str(tmp_path / "plan.json"),
            "--prestate",
            str(tmp_path / "pre.json"),
            "--host-restore",
            str(tmp_path / "restore.json"),
            "--output-root",
            str(tmp_path / "out"),
            "--probe-update",
            "3425",
            "--slowdown-update",
            "3415",
            "--trajectory-start-update",
            "3429",
            "--trajectory-end-update",
            "3500",
            "--trajectory-mode",
            "rng",
            "--skip-repaint-guard",
            "--skip-frozen-snapshot",
            "--trace-detach-update",
            "1702",
            "--trace-reattach-update",
            "3501",
            "--global-mtrand-call-oracle",
            str(oracle),
            "--global-mtrand-call-oracle-seed",
            "23557968",
            "--global-mtrand-call-start-after-update",
            "1702",
            "--global-mtrand-call-end-at-update",
            "3500",
            "--initial-global-mtrand-oracle",
            str(oracle),
            "--initial-global-mtrand-seed",
            "23557968",
            "--diagnostic-disable-source-bound-board-anchor",
        ]
    )

    assert exit_code == 0
    assert captured["global_mtrand_call_oracle"] == oracle.resolve()
    assert captured["global_mtrand_call_oracle_seed"] == 23557968
    assert captured["global_mtrand_call_start_after_update"] == 1702
    assert captured["global_mtrand_call_end_at_update"] == 3500
    assert captured["initial_global_mtrand_oracle"] == oracle.resolve()
    assert captured["initial_global_mtrand_seed"] == 23557968
    assert captured["diagnostic_disable_source_bound_board_anchor"] is True


def test_main_forwards_read_only_global_boundary_observer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oracle = tmp_path / "global-calls.json"
    oracle.write_text("{}\n", encoding="ascii")
    captured: dict[str, object] = {}

    def fake_collect_probe(**kwargs: object) -> Path:
        captured.update(kwargs)
        return tmp_path / "attempt-01" / "memory-probe.json"

    monkeypatch.setattr(collector, "collect_probe", fake_collect_probe)

    exit_code = collector.main(
        [
            "--plan",
            str(tmp_path / "plan.json"),
            "--prestate",
            str(tmp_path / "pre.json"),
            "--host-restore",
            str(tmp_path / "restore.json"),
            "--output-root",
            str(tmp_path / "out"),
            "--probe-update",
            "3425",
            "--slowdown-update",
            "3415",
            "--trajectory-end-update",
            "3425",
            "--trajectory-mode",
            "rng",
            "--skip-repaint-guard",
            "--skip-frozen-snapshot",
            "--trace-detach-update",
            "1702",
            "--trace-reattach-update",
            "3501",
            "--global-mtrand-call-oracle",
            str(oracle),
            "--global-mtrand-call-oracle-seed",
            "23557968",
            "--global-mtrand-call-start-after-update",
            "1702",
            "--global-mtrand-call-end-at-update",
            "3121",
            "--global-mtrand-call-observe-boundary-only",
            "--global-mtrand-call-observe-thread-runtime",
            "--global-mtrand-call-observe-handoff-thread-runtime",
            "--global-mtrand-call-observe-handoff-rng-state",
            "--global-mtrand-call-handoff-rng-wait-target-words-sha256",
            "sha256:" + ("ab" * 32),
            "--global-mtrand-call-handoff-rng-wait-min-index",
            "315",
            "--global-mtrand-call-handoff-rng-wait-timeout-seconds",
            "0.25",
            "--global-mtrand-call-handoff-rng-wait-poll-interval-seconds",
            "0.001",
            "--diagnostic-disable-source-bound-board-anchor",
        ]
    )

    assert exit_code == 0
    assert captured["global_mtrand_call_end_at_update"] == 3121
    assert captured["global_mtrand_call_observe_boundary_only"] is True
    assert captured["global_mtrand_call_observe_thread_runtime"] is True
    assert (
        captured["global_mtrand_call_observe_handoff_thread_runtime"]
        is True
    )
    assert (
        captured["global_mtrand_call_observe_handoff_rng_state"]
        is True
    )
    assert captured[
        "global_mtrand_call_handoff_rng_wait_target_words_sha256"
    ] == "sha256:" + ("ab" * 32)
    assert captured["global_mtrand_call_handoff_rng_wait_min_index"] == 315
    assert captured[
        "global_mtrand_call_handoff_rng_wait_timeout_seconds"
    ] == 0.25
    assert captured[
        "global_mtrand_call_handoff_rng_wait_poll_interval_seconds"
    ] == 0.001


def test_global_mtrand_retry_boundary_is_pre_handoff_only() -> None:
    assert collector._attempt_failure_disposition(
        global_mtrand_call_requested=True,
        global_mtrand_handoff_ready_observed=False,
    ) == ("RETRY", "pre_handoff_startup_infrastructure")
    assert collector._attempt_failure_disposition(
        global_mtrand_call_requested=True,
        global_mtrand_handoff_ready_observed=True,
    ) == ("FAIL", "post_handoff_nonretryable")
    assert collector._attempt_failure_disposition(
        global_mtrand_call_requested=False,
        global_mtrand_handoff_ready_observed=False,
    ) == ("RETRY", None)


def test_handoff_rng_state_contract_is_bound_and_zero_write() -> None:
    ready = {
        "handoff_rng_state_sha256": "sha256:" + ("ab" * 32),
        "handoff_rng_words_sha256": "sha256:" + ("cd" * 32),
        "handoff_rng_index": 217,
        "handoff_rng_captured_perf_counter_ns": 123456789,
    }
    observation = {
        "schema": "zuma-rl.pc-global-mtrand-handoff-state",
        "version": 1,
        "classification": "diagnostic-read-only-process-memory-snapshot",
        "address": collector.G_FRAMEWORK_MTRAND_ADDRESS,
        "state_bytes": collector.MTRAND_STATE_BYTES,
        "state_sha256": ready["handoff_rng_state_sha256"],
        "words_sha256": ready["handoff_rng_words_sha256"],
        "index": ready["handoff_rng_index"],
        "captured_perf_counter_ns": ready[
            "handoff_rng_captured_perf_counter_ns"
        ],
        "draw_count": 1465,
        "process_memory_reads": 1,
        "process_memory_read_bytes": collector.MTRAND_STATE_BYTES,
        "process_memory_writes": 0,
        "process_memory_mutation": False,
    }

    assert collector._handoff_rng_state_contract_passes(
        observation,
        ready=ready,
        enabled=True,
    )
    assert collector._handoff_rng_state_contract_passes(
        None,
        ready=ready,
        enabled=False,
    )
    assert not collector._handoff_rng_state_contract_passes(
        {**observation, "address": collector.G_FRAMEWORK_MTRAND_ADDRESS + 4},
        ready=ready,
        enabled=True,
    )
    assert not collector._handoff_rng_state_contract_passes(
        {**observation, "process_memory_writes": 1},
        ready=ready,
        enabled=True,
    )


def test_handoff_rng_wait_contract_accepts_target_and_timeout() -> None:
    target_words = "sha256:" + ("cd" * 32)
    state_sha256 = "sha256:" + ("ab" * 32)
    bundle = {
        "handoff_rng_wait_target_words_sha256": target_words,
        "handoff_rng_wait_min_index": 315,
        "handoff_rng_wait_timeout_seconds": 0.25,
        "handoff_rng_wait_poll_interval_seconds": 0.001,
    }
    handoff = {
        "state_sha256": state_sha256,
        "words_sha256": target_words,
        "index": 315,
        "captured_perf_counter_ns": 120,
        "draw_count": 1563,
    }
    ready = {
        "handoff_rng_readiness_wait_final_state_sha256": state_sha256,
        "handoff_rng_readiness_wait_final_words_sha256": target_words,
        "handoff_rng_readiness_wait_final_index": 315,
        "handoff_rng_readiness_wait_final_captured_perf_counter_ns": 120,
    }
    receipt = {
        "schema": "zuma-rl.pc-global-mtrand-handoff-readiness-wait",
        "version": 1,
        "classification": (
            "diagnostic-read-only-main-thread-suspended-rng-readiness-wait"
        ),
        "status": "TARGET_REACHED",
        "target_words_sha256": target_words,
        "minimum_index": 315,
        "timeout_seconds": 0.25,
        "poll_interval_seconds": 0.001,
        "target_reached": True,
        "main_thread_suspended_during_wait": True,
        "started_perf_counter_ns": 100,
        "finished_perf_counter_ns": 130,
        "elapsed_perf_counter_ns": 30,
        "initial_state_sha256": "sha256:" + ("01" * 32),
        "initial_words_sha256": target_words,
        "initial_index": 299,
        "initial_captured_perf_counter_ns": 110,
        "initial_draw_count": 1547,
        "final_state_sha256": state_sha256,
        "final_words_sha256": target_words,
        "final_index": 315,
        "final_captured_perf_counter_ns": 120,
        "final_draw_count": 1563,
        "poll_count": 1,
        "process_memory_reads": 2,
        "process_memory_read_bytes": 2 * collector.MTRAND_STATE_BYTES,
        "process_memory_writes": 0,
        "process_memory_mutation": False,
    }

    assert collector._handoff_rng_wait_contract_passes(
        receipt,
        ready=ready,
        bundle=bundle,
        handoff_rng_state=handoff,
        enabled=True,
    )

    timeout_handoff = {
        **handoff,
        "index": 314,
        "captured_perf_counter_ns": 110,
        "draw_count": 1562,
    }
    timeout_ready = {
        **ready,
        "handoff_rng_readiness_wait_final_index": 314,
        "handoff_rng_readiness_wait_final_captured_perf_counter_ns": 110,
    }
    timeout_receipt = {
        **receipt,
        "status": "TIMEOUT",
        "target_reached": False,
        "finished_perf_counter_ns": 120,
        "elapsed_perf_counter_ns": 20,
        "initial_state_sha256": state_sha256,
        "initial_index": 314,
        "initial_draw_count": 1562,
        "final_index": 314,
        "final_captured_perf_counter_ns": 110,
        "final_draw_count": 1562,
        "poll_count": 0,
        "process_memory_reads": 1,
        "process_memory_read_bytes": collector.MTRAND_STATE_BYTES,
    }
    assert collector._handoff_rng_wait_contract_passes(
        timeout_receipt,
        ready=timeout_ready,
        bundle=bundle,
        handoff_rng_state=timeout_handoff,
        enabled=True,
    )


def test_global_mtrand_ready_requires_transactional_v2_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakePopen:
        def __init__(self) -> None:
            self.returncode: int | None = None

        def poll(self) -> int | None:
            return self.returncode

    monkeypatch.setattr(collector.subprocess, "Popen", FakePopen)
    ready_path = tmp_path / "ready.json"
    payload = {
        "schema": "zuma-rl.pc-global-mtrand-oracle-synchronizer-ready",
        "version": 2,
        "process_id": 123,
        "main_thread_id": 456,
        "resume_main_thread_on_ready": True,
        "handoff_verified": True,
        "handoff_main_thread_resumed": True,
        "handoff_resume_previous_suspend_count": 1,
        "publication_order": "AFTER_VERIFIED_HANDOFF_RESUME",
        "published_perf_counter_ns": 123456,
        "mode": "observe_boundary_only",
        "observe_boundary_thread_runtime": True,
        "expected_call_count": 1,
    }
    ready_path.write_text(json.dumps(payload), encoding="ascii")
    process = FakePopen()
    trace_process = FakePopen()
    bundle = {
        "process": process,
        "ready_path": ready_path,
        "pid": 123,
        "main_thread_id": 456,
        "observe_boundary_only": True,
        "observe_thread_runtime": True,
    }

    observed = collector._wait_for_global_mtrand_call_sync_ready(
        bundle,
        trace_process=trace_process,
        timeout_seconds=1.0,
    )

    assert observed == payload

    payload["version"] = 1
    ready_path.write_text(json.dumps(payload), encoding="ascii")
    with pytest.raises(
        ProbeError,
        match="global_mtrand_call_sync_ready_contract_failed",
    ):
        collector._wait_for_global_mtrand_call_sync_ready(
            bundle,
            trace_process=trace_process,
            timeout_seconds=1.0,
        )


def test_initial_global_seed_requires_zero_write_suffix() -> None:
    collector._require_zero_write_global_mtrand_suffix(
        {
            "state_correction_count": 0,
            "bytes_written_total": 0,
            "process_memory_mutation": False,
        }
    )
    with pytest.raises(
        ProbeError,
        match="initial_global_mtrand_seed_suffix_not_zero_write",
    ):
        collector._require_zero_write_global_mtrand_suffix(
            {
                "state_correction_count": 1,
                "bytes_written_total": 2500,
                "process_memory_mutation": True,
            }
        )


def test_read_only_global_boundary_receipt_accepts_natural_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakePopen:
        def __init__(self) -> None:
            self.returncode: int | None = None

        def wait(self, timeout: float | None = None) -> int:
            self.returncode = 0
            return 0

    monkeypatch.setattr(collector.subprocess, "Popen", FakePopen)
    runtime = tmp_path / "runtime.exe"
    runtime.write_bytes(b"runtime")
    oracle_path = tmp_path / "oracle.json"
    oracle_path.write_bytes(b"oracle")
    result_path = tmp_path / "global-mtrand-boundary-observation.json"
    ready_path = tmp_path / "global-mtrand-boundary-observation.ready.json"
    ready_path.write_text("{}\n", encoding="ascii")
    log_stream = (
        tmp_path / "global-mtrand-boundary-observation.log"
    ).open("xb")
    semantic_hash = "sha256:" + "1" * 64
    state_hash = "sha256:" + "2" * 64
    post_hash = "sha256:" + "3" * 64
    payload = {
        "schema": "zuma-rl.pc-global-mtrand-oracle-synchronizer",
        "version": 1,
        "status": "PASS",
        "failure": None,
        "classification": (
            "diagnostic-read-only-global-rng-boundary-observation"
        ),
        "mode": "observe_boundary_only",
        "process_id": 123,
        "main_thread_id": 456,
        "runtime_executable": str(runtime.resolve()),
        "runtime_executable_sha256": collector._sha256_path(runtime),
        "oracle": {
            "source_path": str(oracle_path.resolve()),
            "source_sha256": collector._sha256_path(oracle_path),
            "seed": 23557968,
            "start_after_update": 1702,
            "end_at_update": 3121,
            "entry_count": 1,
            "semantic_sha256": semantic_hash,
        },
        "stop_reason": "boundary_observed",
        "expected_hit_count": 1,
        "hit_count": 1,
        "missing_hit_count": 0,
        "semantic_mismatch": None,
        "unexpected_extra_call": None,
        "state_correction_count": 0,
        "bytes_written_total": 0,
        "process_memory_writes": 0,
        "process_memory_mutation": False,
        "process_context_mutation": True,
        "persistent_file_modified": False,
        "hardware_breakpoint_restored": True,
        "hardware_breakpoint_restore_error": None,
        "debugger_detach_error": None,
        "resume_main_thread_on_ready": True,
        "observe_handoff_thread_runtime": False,
        "handoff_main_thread_resumed": True,
        "handoff_resume_previous_suspend_count": 1,
        "ready_receipt_schema": (
            "zuma-rl.pc-global-mtrand-oracle-synchronizer-ready"
        ),
        "ready_receipt_version": 2,
        "ready_receipt_published": True,
        "ready_receipt_publication_order": (
            "AFTER_VERIFIED_HANDOFF_RESUME"
        ),
        "ready_receipt_published_perf_counter_ns": 123456,
        "natural_boundary_source_exact": False,
        "draw_count_reconstruction": {
            "status": "PASS",
            "maximum_draws": 100000,
            "state_count": 2,
        },
        "final_post_verification": {
            "verified": True,
            "return_address_match": True,
            "expected_output_match": False,
            "expected_post_state_match": False,
            "source_exact": False,
            "observed_post_draw_count": 1640,
            "observed_post_state_sha256": post_hash,
        },
        "hits": [
            {
                "source_order": 309,
                "contract_match": True,
                "live_pre_draw_count": 1639,
                "live_pre_state_sha256": state_hash,
                "live_pre_state_match": False,
                "bytes_written": 0,
                "changed": False,
                "state_correction_applied": False,
            }
        ],
        "thread_runtime_observation": {
            "schema": (
                "zuma-rl.pc-global-mtrand-boundary-"
                "thread-runtime-observation"
            ),
            "version": 1,
            "enabled": True,
            "process_memory_reads": 0,
            "process_memory_writes": 0,
            "process_context_reads": 0,
            "process_context_writes": 0,
            "handoff_snapshot_enabled": False,
            "lifecycle_event_count": 1,
            "lifecycle_events": [
                {
                    "event": "create_thread",
                    "thread_id": 456,
                }
            ],
            "boundary_snapshot": {
                "schema": "zuma-rl.pc-thread-runtime-snapshot",
                "version": 1,
                "phase": "global_mtrand_boundary_pre_call",
                "process_id": 123,
                "main_thread_id": 456,
                "thread_count": 1,
                "process_memory_reads": 0,
                "process_memory_writes": 0,
                "process_context_reads": 0,
                "process_context_writes": 0,
                "threads": [
                    {
                        "thread_id": 456,
                        "is_main_thread": True,
                        "accessible": True,
                    }
                ],
            },
        },
    }
    result_path.write_text(json.dumps(payload), encoding="ascii")
    process = FakePopen()
    bundle = {
        "process": process,
        "log_stream": log_stream,
        "result_path": result_path,
        "ready_path": ready_path,
        "pid": 123,
        "main_thread_id": 456,
        "runtime_executable": runtime,
        "oracle_path": oracle_path,
        "seed": 23557968,
        "start_after_update": 1702,
        "end_at_update": 3121,
        "timeout_seconds": 1.0,
        "observe_boundary_only": True,
        "observe_thread_runtime": True,
    }
    ready = {
        "expected_call_count": 1,
        "mode": "observe_boundary_only",
        "oracle_semantic_sha256": semantic_hash,
        "observe_boundary_thread_runtime": True,
        "published_perf_counter_ns": 123456,
    }

    evidence = collector._finish_global_mtrand_call_sync(
        bundle,
        ready=ready,
    )

    assert evidence["status"] == "PASS"
    assert evidence["source_exact"] is False
    assert evidence["first_hit"]["live_pre_draw_count"] == 1639
    assert evidence["process_memory_writes"] == 0
    assert evidence["thread_runtime_observation"]["enabled"] is True
    assert evidence["ready_receipt_version"] == 2
    assert evidence["ready_receipt_publication_order"] == (
        "AFTER_VERIFIED_HANDOFF_RESUME"
    )


def test_probe_parser_exposes_diagnostic_startup_priority_bias() -> None:
    args = _parser().parse_args(
        [
            "--plan",
            "plan.json",
            "--prestate",
            "pre.json",
            "--host-restore",
            "restore.json",
            "--output-root",
            "out",
            "--trajectory-end-update",
            "8000",
            "--skip-repaint-guard",
            "--skip-frozen-snapshot",
            "--diagnostic-startup-priority-bias-until-update",
            "500",
        ]
    )

    assert args.diagnostic_startup_priority_bias_until_update == 500


def test_probe_parser_exposes_source_bound_compact_restore() -> None:
    args = _parser().parse_args(
        [
            "--plan",
            "plan.json",
            "--prestate",
            "pre.json",
            "--host-restore",
            "restore.json",
            "--output-root",
            "out",
            "--diagnostic-compact-state-source-probe",
            "source-memory-probe.json",
            "--diagnostic-compact-state-source-probe-sha256",
            "sha256:" + "a" * 64,
            "--diagnostic-compact-state-source-update",
            "3615",
        ]
    )

    assert args.diagnostic_compact_state_source_probe == Path(
        "source-memory-probe.json"
    )
    assert args.diagnostic_compact_state_source_probe_sha256 == (
        "sha256:" + "a" * 64
    )
    assert args.diagnostic_compact_state_source_update == 3615


def test_probe_parser_exposes_diagnostic_process_affinity() -> None:
    args = _parser().parse_args(
        [
            "--plan",
            "plan.json",
            "--prestate",
            "pre.json",
            "--host-restore",
            "restore.json",
            "--output-root",
            "out",
            "--trajectory-end-update",
            "8000",
            "--skip-repaint-guard",
            "--skip-frozen-snapshot",
            "--diagnostic-process-affinity-mask",
            "0x1",
        ]
    )

    assert args.diagnostic_process_affinity_mask == 1


def test_probe_parser_exposes_exact_step_scan_skip() -> None:
    args = _parser().parse_args(
        [
            "--plan",
            "plan.json",
            "--prestate",
            "pre.json",
            "--host-restore",
            "restore.json",
            "--output-root",
            "out",
            "--discover-score",
            "--skip-int32-scan",
        ]
    )

    assert args.discover_score is True
    assert args.skip_int32_scan is True


def test_runtime_identity_uses_signed_source_not_unpacked_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = tmp_path / "popcapgame1.exe"
    runtime.write_bytes(b"unpacked-runtime")
    signed_source = tmp_path / "ZumasRevenge.exe"
    signed_source.write_bytes(b"signed-retail-wrapper")
    plan = {
        "runtime": {
            "runtime_executable": str(runtime),
            "runtime_source_executable": str(signed_source),
            "runtime_source_sha256": collector._sha256_path(signed_source),
        }
    }

    selected_source = collector._runtime_source_executable_from_plan(plan)
    observed: dict[str, object] = {}

    def fake_identity(
        target: object,
        *,
        process_name: str,
        runtime_source_executable: Path,
    ) -> dict[str, object]:
        observed.update(
            {
                "target": target,
                "process_name": process_name,
                "runtime_source_executable": runtime_source_executable,
            }
        )
        return {
            "process_id": 123,
            "executable_sha256": collector._sha256_path(runtime),
        }

    monkeypatch.setattr(collector, "windows_process_identity", fake_identity)
    identity = collector._capture_runtime_process_identity(
        object(),
        runtime_pid=123,
        runtime_executable=runtime,
        runtime_source_executable=selected_source,
    )

    assert selected_source == signed_source.resolve()
    assert observed["runtime_source_executable"] == signed_source.resolve()
    assert observed["runtime_source_executable"] != runtime.resolve()
    assert observed["process_name"] == "popcapgame1.exe"
    assert identity["executable_sha256"] == collector._sha256_path(runtime)


def test_runtime_identity_rejects_wrong_signed_source_hash(
    tmp_path: Path,
) -> None:
    signed_source = tmp_path / "ZumasRevenge.exe"
    signed_source.write_bytes(b"signed-retail-wrapper")
    plan = {
        "runtime": {
            "runtime_source_executable": str(signed_source),
            "runtime_source_sha256": "sha256:" + "0" * 64,
        }
    }

    with pytest.raises(ProbeError, match="runtime_source_identity_invalid"):
        collector._runtime_source_executable_from_plan(plan)


def test_runtime_identity_accepts_locked_steam_payload_via_signed_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = tmp_path / "popcapgame1.exe"
    signed_source = tmp_path / "ZumasRevenge.exe"
    expected_sha256 = "sha256:" + "2" * 64

    monkeypatch.setattr(
        collector,
        "windows_process_identity",
        lambda *args, **kwargs: {
            "process_id": 321,
            "executable_sha256": expected_sha256,
            "executable_hash_provenance": {
                "method": "embedded_signed_pe",
                "runtime_file_direct_hash_verified": False,
            },
        },
    )
    monkeypatch.setattr(
        collector,
        "_sha256_path",
        lambda path: (_ for _ in ()).throw(PermissionError(path)),
    )

    identity = collector._capture_runtime_process_identity(
        object(),
        runtime_pid=321,
        runtime_executable=runtime,
        runtime_source_executable=signed_source,
        expected_runtime_sha256=expected_sha256,
    )

    assert identity["executable_sha256"] == expected_sha256
    assert identity["executable_hash_provenance"][
        "runtime_file_direct_hash_verified"
    ] is False


def test_trace_args_preserve_certified_replay_boundary_options(
    tmp_path: Path,
) -> None:
    plan = {
        "runtime": {
            "runtime_executable": "runtime.exe",
            "changedir": "game-root",
            "crt_rand_seed": 11,
            "board_seed_call_address": 22,
            "board_seed": 33,
            "global_rng_seed": 44,
            "thread_crt_rng_seed": 55,
            "startup_seed_transport": "debugger_register",
        },
        "trace": {
            "attach_at_update": 230,
            "detach_at_update": 12600,
            "reattach_at_update": 13750,
            "maximum_hits": 10000,
            "trace_timeout_seconds": 1200.0,
            "launch_timeout_seconds": 30.0,
            "allow_pre_stream_commands": False,
            "allow_attach_stabilization": True,
            "allow_pre_attach_file_write_debt": True,
            "allow_font_cache_manifest_completion_debt": True,
            "seed_board_before_attach": True,
            "allow_blackout_file_write_order_rebase": True,
            "startup_priority_bias_until_update": 456,
        },
    }

    args = collector._trace_args(
        project_root=tmp_path,
        plan=plan,
        dmo=tmp_path / "input.dmo",
        result_path=tmp_path / "strict-replay.json",
    )

    for flag in (
        "--allow-attach-stabilization",
        "--allow-pre-attach-file-write-debt",
        "--allow-font-cache-manifest-completion-debt",
        "--seed-board-before-attach",
        "--allow-blackout-file-write-order-rebase",
    ):
        assert flag in args
    assert args[args.index("--startup-priority-bias-until-update") + 1] == (
        "456"
    )
    assert args[args.index("--startup-seed-transport") + 1] == (
        "debugger_register"
    )


def test_trace_args_support_natural_steam_without_seed_injection(
    tmp_path: Path,
) -> None:
    plan = {
        "dmo": {"length_updates": 10536},
        "runtime": {
            "launch_mode": "steam_embedded_runtime",
            "runtime_executable": "runtime.exe",
            "steam_executable": "steam.exe",
            "changedir": None,
            "crt_rand_seed": None,
            "board_seed": None,
            "global_rng_seed": None,
            "thread_crt_rng_seed": None,
            "startup_seed_transport": None,
        },
        "trace": {
            "attach_at_update": 0,
            "detach_at_update": 7300,
            "reattach_at_update": 8500,
            "maximum_hits": 6000,
            "trace_timeout_seconds": 600.0,
            "launch_timeout_seconds": 30.0,
            "allow_pre_stream_commands": True,
        },
    }

    args = collector._trace_args(
        project_root=tmp_path,
        plan=plan,
        dmo=tmp_path / "input.dmo",
        result_path=tmp_path / "natural-steam.json",
    )

    assert Path(args[1]).name == "launch_popcap_replay.py"
    assert args[args.index("--steam-exe") + 1] == "steam.exe"
    assert args[args.index("--minus-count") + 1] == "0"
    assert "--monitor-until-exit" in args
    for forbidden in (
        "--direct-runtime-exe",
        "--changedir",
        "--crt-rand-seed",
        "--board-seed",
        "--global-rng-seed",
        "--thread-crt-rng-seed",
        "--startup-seed-transport",
    ):
        assert forbidden not in args


def test_trace_args_support_direct_natural_seed_without_debugger(
    tmp_path: Path,
) -> None:
    plan = {
        "dmo": {"length_updates": 10536},
        "runtime": {
            "launch_mode": "direct_byte_identical_natural_seed",
            "runtime_executable": "runtime.exe",
            "changedir": "game-root",
            "crt_rand_seed": None,
            "board_seed": None,
            "global_rng_seed": None,
            "thread_crt_rng_seed": None,
            "startup_seed_transport": None,
        },
        "trace": {
            "attach_at_update": 0,
            "detach_at_update": 7300,
            "reattach_at_update": 8500,
            "maximum_hits": 6000,
            "trace_timeout_seconds": 600.0,
            "launch_timeout_seconds": 30.0,
            "allow_pre_stream_commands": True,
        },
    }

    args = collector._trace_args(
        project_root=tmp_path,
        plan=plan,
        dmo=tmp_path / "input.dmo",
        result_path=tmp_path / "direct-natural.json",
    )

    assert Path(args[1]).name == "launch_popcap_replay.py"
    assert args[args.index("--direct-runtime-exe") + 1] == "runtime.exe"
    assert args[args.index("--changedir") + 1] == "game-root"
    assert "--monitor-until-exit" in args
    for forbidden in (
        "--crt-rand-seed",
        "--board-seed",
        "--global-rng-seed",
        "--thread-crt-rng-seed",
        "--startup-seed-transport",
        "--broker-service-blocks",
    ):
        assert forbidden not in args

def test_trace_args_support_natural_seed_strict_command_broker(
    tmp_path: Path,
) -> None:
    plan = {
        "dmo": {
            "length_updates": 10536,
            "diagnostic_successful_file_write_padding_rows": [],
        },
        "runtime": {
            "launch_mode": (
                "direct_byte_identical_natural_seed_strict_command_broker"
            ),
            "runtime_executable": "runtime.exe",
            "changedir": "game-root",
            "crt_rand_seed": None,
            "board_seed_call_address": None,
            "board_seed": None,
            "global_rng_seed": None,
            "thread_crt_rng_seed": None,
            "startup_seed_transport": None,
        },
        "trace": {
            "attach_at_update": 0,
            "detach_at_update": None,
            "reattach_at_update": None,
            "natural_command_broker_stop_after_update": 3151,
            "maximum_hits": 6000,
            "trace_timeout_seconds": 600.0,
            "launch_timeout_seconds": 30.0,
            "allow_pre_stream_commands": True,
            "startup_trace_handoff": True,
            "allow_font_cache_manifest_completion_debt": True,
            "font_cache_manifest_receipt": {"entry_count": 35},
            "seed_board_before_attach": False,
            "allow_source_bound_board_global_correction": False,
            "source_bound_board_anchor": None,
            "source_bound_board_precall_global_restore": None,
        },
    }

    args = collector._trace_args(
        project_root=tmp_path,
        plan=plan,
        dmo=tmp_path / "input.dmo",
        result_path=tmp_path / "strict-natural.json",
    )

    assert Path(args[1]).name == "trace_popcap_demo_commands.py"
    assert "--direct-natural-seed" in args
    assert "--startup-trace-handoff" in args
    assert "--broker-service-blocks" in args
    assert "--allow-font-cache-manifest-completion-debt" in args
    assert args[args.index("--stop-after-update") + 1] == "3151"
    for forbidden in (
        "--crt-rand-seed",
        "--board-seed",
        "--global-rng-seed",
        "--thread-crt-rng-seed",
        "--detach-at-update",
        "--reattach-at-update",
    ):
        assert forbidden not in args

    diagnostic_args = collector._trace_args(
        project_root=tmp_path,
        plan=plan,
        dmo=tmp_path / "input.dmo",
        result_path=tmp_path / "strict-natural-diagnostic.json",
        startup_priority_bias_until_update=500,
        process_affinity_mask=1,
    )
    assert diagnostic_args[
        diagnostic_args.index("--startup-priority-bias-until-update") + 1
    ] == "500"
    assert diagnostic_args[
        diagnostic_args.index("--startup-process-affinity-mask") + 1
    ] == "1"

    fixed_startup_args = collector._trace_args(
        project_root=tmp_path,
        plan=plan,
        dmo=tmp_path / "input.dmo",
        result_path=tmp_path / "strict-fixed-startup.json",
        diagnostic_startup_crt_seed=316978734,
    )
    assert "--direct-natural-seed" not in fixed_startup_args
    assert fixed_startup_args[
        fixed_startup_args.index("--crt-rand-seed") + 1
    ] == "316978734"
    assert fixed_startup_args[
        fixed_startup_args.index("--startup-seed-transport") + 1
    ] == "debugger_register"


def _natural_strict_binding_case(
    tmp_path: Path,
) -> tuple[dict[str, object], dict[str, object], Path, Path, Path]:
    dmo = tmp_path / "input.dmo"
    runtime = tmp_path / "runtime.exe"
    artifact = tmp_path / "strict-replay.json"
    dmo.write_bytes(b"dmo")
    runtime.write_bytes(b"runtime")
    artifact.write_bytes(b"strict trace")
    manifest = {
        "entry_count": 35,
        "manifest_sha256": "sha256:" + "1" * 64,
        "main_pak_sha256": "sha256:" + "2" * 64,
    }
    plan: dict[str, object] = {
        "trace": {
            "natural_command_broker_stop_after_update": 3151,
            "font_cache_manifest_receipt": manifest,
        }
    }
    payload: dict[str, object] = {
        "schema": "zuma.popcap_strict_replay.v3",
        "runtime_process_id": 55,
        "runtime_executable": str(runtime),
        "source_dmo": {
            "path": str(dmo),
            "sha256": collector._sha256_path(dmo).removeprefix("sha256:"),
            "size_bytes": dmo.stat().st_size,
        },
        "options": {
            "direct_runtime": True,
            "direct_natural_seed": True,
            "broker_service_blocks": True,
            "allow_pre_stream_commands": True,
            "startup_trace_handoff": True,
            "allow_font_cache_manifest_completion_debt": True,
            "rng_seed_override_count": 0,
            "attach_at_update": 0,
            "stop_after_update": 3151,
            "stop_after_command_order": None,
            "diagnostic_successful_file_write_padding_rows": [],
        },
        "startup_rng": {
            "mode": "retail_natural_seed_observation",
            "process_id": 55,
            "thread_id": 7,
            "observed_seed": 123456,
            "effective_seed": 123456,
            "seed_source": "retail_eax_before_push_to_srand",
            "register_override": None,
            "rng_process_memory_writes": 0,
            "persistent_file_modified": False,
            "main_thread_suspended_on_detach": True,
            "main_thread_suspend_previous_count": 0,
            "original_instruction_hex": "50",
            "compatibility_layer": "HIGHDPIAWARE",
        },
        "result": {
            "failure_count": 0,
            "boundary_failures": [],
            "broker_failures": [],
            "stopped_at_update": True,
            "stopped_at_command_order": False,
            "last_update": 3151,
            "font_cache_manifest_entry_count": 35,
            "font_cache_manifest_sha256": "1" * 64,
            "font_cache_manifest_main_pak_sha256": "2" * 64,
            "startup_post_bypass_worker_yields": [
                {
                    "process_id": 55,
                    "thread_id": 7,
                    "memory_writes": 0,
                    "failure": None,
                    "probe_status": "boundary",
                    "command_breakpoint_rearmed": True,
                    "resume_count": 1,
                    "temporary_breakpoint_memory_writes": 4,
                    "continuation_commit": {
                        "mechanism": "atomic_startup_worker_payload_boundary"
                    },
                }
            ],
            "direct_font_cache_file_write_observations": [],
            "deferred_file_write_discharges": [],
            "font_cache_manifest_completion_discharges": [],
            "terminal_file_write_payload_handoffs": [],
            "service_file_write_header_claims": [],
            "service_continuation_verifications": [],
            "preloading_failed_file_write_tail_short_header_recoveries": [],
        },
    }
    return payload, plan, artifact, dmo, runtime


def test_natural_strict_trace_binding_accepts_zero_rng_write_receipt(
    tmp_path: Path,
) -> None:
    payload, plan, artifact, dmo, runtime = _natural_strict_binding_case(
        tmp_path
    )

    binding = collector._natural_strict_trace_binding(
        payload,
        artifact_path=artifact,
        plan=plan,
        expected_pid=55,
        expected_dmo=dmo,
        expected_runtime=runtime,
    )

    assert binding["status"] == "PASS"
    assert binding["startup_seed"] == 123456
    assert binding["rng_process_memory_writes"] == 0
    assert binding["transport"][
        "temporary_code_breakpoint_memory_write_count"
    ] == 4
    assert binding["transport"]["replay_state_recovery_count"] == 0
    assert binding["transport"][
        "replay_state_recovery_memory_write_count"
    ] == 0


def test_diagnostic_fixed_startup_binding_discloses_context_mutation(
    tmp_path: Path,
) -> None:
    payload, plan, artifact, dmo, runtime = _natural_strict_binding_case(
        tmp_path
    )
    options = payload["options"]
    assert isinstance(options, dict)
    options.update(
        {
            "direct_natural_seed": False,
            "crt_rand_seed": 316978734,
            "startup_seed_transport": "debugger_register",
            "rng_seed_override_count": None,
        }
    )
    payload["startup_rng"] = {
        "process_id": 55,
        "thread_id": 7,
        "seed": 316978734,
        "register_override": "eax_before_push_to_srand",
        "persistent_file_modified": False,
        "main_thread_suspended_on_detach": True,
        "main_thread_suspend_previous_count": 0,
        "original_instruction_hex": "50",
        "compatibility_layer": "HIGHDPIAWARE",
    }

    binding = collector._diagnostic_fixed_startup_strict_trace_binding(
        payload,
        artifact_path=artifact,
        plan=plan,
        expected_pid=55,
        expected_dmo=dmo,
        expected_runtime=runtime,
        expected_startup_seed=316978734,
    )

    assert binding["status"] == "PASS"
    assert binding["startup_seed"] == 316978734
    assert binding["rng_seed_override_count"] == 1
    assert binding["process_context_mutation"] is True
    assert binding["formal_evidence_eligible"] is False


def test_natural_strict_trace_binding_accepts_exact_tail_short_header_recovery(
    tmp_path: Path,
) -> None:
    payload, plan, artifact, dmo, runtime = _natural_strict_binding_case(
        tmp_path
    )
    result = payload["result"]
    assert isinstance(result, dict)
    result[
        "preloading_failed_file_write_tail_short_header_recoveries"
    ] = [
        {
            "mechanism": "audited_false_short_header_pre_payload_recovery",
            "process_id": 55,
            "thread_id": 7,
            "framework_update": 401,
            "false_header_row_index": 74,
            "corridor_start_index": 45,
            "corridor_end_index": 71,
            "command_order_offset": 4,
            "command_order_rebase_rows": [64, 65, 68, 69],
            "before": {
                "buffer_read_bit_position": 1888,
                "last_demo_update": 401,
                "needs_command": 0,
                "is_short": 1,
                "command_number": 1,
                "command_order": 68,
                "command_bit_position": 1882,
                "demo_loading_complete": 0,
            },
            "after": {
                "buffer_read_bit_position": 1860,
                "last_demo_update": 391,
                "needs_command": 1,
                "is_short": 0,
                "command_number": 16,
                "command_order": 67,
                "command_bit_position": 1849,
                "demo_loading_complete": 0,
            },
            "false_payload_executed": False,
            "write_count": 8,
            "bytes_written": 23,
        }
    ]

    binding = collector._natural_strict_trace_binding(
        payload,
        artifact_path=artifact,
        plan=plan,
        expected_pid=55,
        expected_dmo=dmo,
        expected_runtime=runtime,
    )

    assert binding["version"] == 2
    assert binding["transport"]["replay_state_recovery_count"] == 1
    assert binding["transport"][
        "replay_state_recovery_memory_write_count"
    ] == 8
    assert binding["transport"][
        "replay_state_recovery_memory_write_bytes"
    ] == 23


def test_natural_strict_trace_binding_binds_diagnostic_priority_bias(
    tmp_path: Path,
) -> None:
    payload, plan, artifact, dmo, runtime = _natural_strict_binding_case(
        tmp_path
    )
    payload["options"]["startup_priority_bias_until_update"] = 500
    payload["options"]["startup_process_affinity_mask"] = 1

    binding = collector._natural_strict_trace_binding(
        payload,
        artifact_path=artifact,
        plan=plan,
        expected_pid=55,
        expected_dmo=dmo,
        expected_runtime=runtime,
        expected_startup_priority_bias_until_update=500,
        expected_startup_process_affinity_mask=1,
    )

    assert binding["status"] == "PASS"
    with pytest.raises(ProbeError):
        collector._natural_strict_trace_binding(
            payload,
            artifact_path=artifact,
            plan=plan,
            expected_pid=55,
            expected_dmo=dmo,
            expected_runtime=runtime,
        )


@pytest.mark.parametrize(
    "tamper",
    ("seed_override", "rng_write", "worker_write", "manifest"),
)
def test_natural_strict_trace_binding_rejects_control_tampering(
    tmp_path: Path,
    tamper: str,
) -> None:
    payload, plan, artifact, dmo, runtime = _natural_strict_binding_case(
        tmp_path
    )
    startup_rng = payload["startup_rng"]
    result = payload["result"]
    assert isinstance(startup_rng, dict)
    assert isinstance(result, dict)
    if tamper == "seed_override":
        startup_rng["register_override"] = 9
    elif tamper == "rng_write":
        startup_rng["rng_process_memory_writes"] = 4
    elif tamper == "worker_write":
        result["startup_post_bypass_worker_yields"][0][
            "memory_writes"
        ] = 1
    else:
        result["font_cache_manifest_sha256"] = "3" * 64

    with pytest.raises(ProbeError):
        collector._natural_strict_trace_binding(
            payload,
            artifact_path=artifact,
            plan=plan,
            expected_pid=55,
            expected_dmo=dmo,
            expected_runtime=runtime,
        )


def test_trace_args_override_startup_priority_bias_for_diagnostic(
    tmp_path: Path,
) -> None:
    plan = {
        "runtime": {
            "runtime_executable": "runtime.exe",
            "changedir": "game-root",
            "crt_rand_seed": 11,
        },
        "trace": {
            "attach_at_update": 0,
            "detach_at_update": 3150,
            "reattach_at_update": 10373,
            "maximum_hits": 10000,
            "trace_timeout_seconds": 900.0,
            "launch_timeout_seconds": 30.0,
            "allow_pre_stream_commands": True,
            "startup_trace_handoff": True,
            "startup_priority_bias_until_update": None,
        },
    }

    args = collector._trace_args(
        project_root=tmp_path,
        plan=plan,
        dmo=tmp_path / "input.dmo",
        result_path=tmp_path / "strict-replay.json",
        startup_priority_bias_until_update=500,
        process_affinity_mask=1,
    )

    assert args[
        args.index("--startup-priority-bias-until-update") + 1
    ] == "500"
    assert args[args.index("--startup-process-affinity-mask") + 1] == "1"


def test_trace_args_preserve_source_bound_precall_contract(
    tmp_path: Path,
) -> None:
    plan = {
        "dmo": {
            "diagnostic_successful_file_write_padding_rows": [12],
        },
        "runtime": {
            "runtime_executable": "runtime.exe",
            "changedir": "game-root",
            "crt_rand_seed": 11,
            "board_seed_call_address": 0x0065B828,
            "board_seed": 263072237,
            "global_rng_seed": None,
            "thread_crt_rng_seed": None,
            "startup_seed_transport": "debugger_register",
        },
        "trace": {
            "attach_at_update": 0,
            "detach_at_update": 7000,
            "reattach_at_update": 9750,
            "maximum_hits": 100000,
            "trace_timeout_seconds": 900.0,
            "launch_timeout_seconds": 30.0,
            "allow_pre_stream_commands": True,
            "startup_trace_handoff": True,
            "allow_post_blackout_attach_stabilization": True,
            "allow_blackout_file_write_order_rebase": True,
            "blackout_allowed_offset_pairs": [
                {
                    "command_order_offset": 0,
                    "native_timeline_offset": 0,
                },
                {
                    "command_order_offset": 4,
                    "native_timeline_offset": 2,
                },
            ],
            "allow_source_bound_board_global_correction": False,
            "source_bound_board_anchor": {
                "monitor_path": "natural/rng.ndjson",
                "trace_path": "natural/global-mtrand-calls.json",
                "recording_report_path": "natural/result.json",
                "monitor_framework_update": 3428,
                "global_seed": 85961375,
                "rewind_draws": 4,
                "source_order": 307,
                "framework_update": 3120,
                "caller": 0x0065B821,
            },
            "source_bound_board_precall_global_restore": {
                "call_address": 0x0065B81C,
            },
        },
    }

    args = collector._trace_args(
        project_root=tmp_path,
        plan=plan,
        dmo=tmp_path / "input.dmo",
        result_path=tmp_path / "strict-replay.json",
    )

    assert "--global-rng-seed" not in args
    assert "--thread-crt-rng-seed" not in args
    assert "None" not in args
    assert "--startup-trace-handoff" in args
    assert "--allow-post-blackout-attach-stabilization" in args
    assert "--source-bound-board-precall-global-restore" in args
    assert args[args.index("--source-bound-board-monitor") + 1] == (
        "natural/rng.ndjson"
    )
    assert args[args.index("--source-bound-board-caller") + 1] == (
        "0x65b821"
    )
    assert [
        args[index + 1]
        for index, value in enumerate(args)
        if value == "--blackout-allowed-offset-pair"
    ] == ["0:0", "4:2"]
    assert args[args.index("--successful-file-write-padding-row") + 1] == (
        "12"
    )

    disabled_args = collector._trace_args(
        project_root=tmp_path,
        plan=plan,
        dmo=tmp_path / "input.dmo",
        result_path=tmp_path / "strict-replay-disabled.json",
        disable_source_bound_board_anchor=True,
    )
    for flag in (
        "--source-bound-board-monitor",
        "--source-bound-board-trace",
        "--source-bound-board-precall-global-restore",
    ):
        assert flag not in disabled_args


def test_trace_args_support_finite_source_bound_formal_trace(
    tmp_path: Path,
) -> None:
    plan = {
        "dmo": {
            "diagnostic_successful_file_write_padding_rows": [],
        },
        "runtime": {
            "launch_mode": "direct_byte_identical_fixed_seed",
            "runtime_executable": "runtime.exe",
            "changedir": "game-root",
            "crt_rand_seed": 147023375,
            "board_seed_call_address": 0x0065B828,
            "board_seed": 835125098,
            "global_rng_seed": None,
            "thread_crt_rng_seed": None,
            "startup_seed_transport": "debugger_register",
        },
        "trace": {
            "attach_at_update": 0,
            "detach_at_update": None,
            "reattach_at_update": None,
            "source_bound_command_broker_stop_after_update": 3151,
            "source_bound_dmo_provenance_path": "provenance.json",
            "maximum_hits": 100000,
            "trace_timeout_seconds": 900.0,
            "launch_timeout_seconds": 30.0,
            "allow_pre_stream_commands": True,
            "startup_trace_handoff": True,
            "allow_font_cache_manifest_completion_debt": True,
            "font_cache_manifest_receipt": {"entry_count": 35},
            "seed_board_before_attach": False,
            "allow_source_bound_board_global_correction": False,
            "source_bound_board_anchor": {
                "monitor_path": "natural/rng.ndjson",
                "trace_path": "natural/global-mtrand-calls.json",
                "recording_report_path": "natural/result.json",
                "monitor_framework_update": 3429,
                "global_seed": 23557968,
                "rewind_draws": 6,
                "source_order": 309,
                "framework_update": 3121,
                "caller": 0x0065B821,
            },
            "source_bound_board_precall_global_restore": {
                "call_address": 0x0065B81C,
            },
        },
    }

    args = collector._trace_args(
        project_root=tmp_path,
        plan=plan,
        dmo=tmp_path / "input.dmo",
        result_path=tmp_path / "strict-replay.json",
    )

    assert args[args.index("--stop-after-update") + 1] == "3151"
    assert "--source-bound-board-precall-global-restore" in args
    assert "--detach-at-update" not in args
    assert "--reattach-at-update" not in args
    assert "--allow-blackout-file-write-order-rebase" not in args
    assert "--allow-post-blackout-attach-stabilization" not in args


def test_trace_args_add_gameplay_mtrand_oracle_and_early_receipt(
    tmp_path: Path,
) -> None:
    plan = {
        "runtime": {
            "runtime_executable": "runtime.exe",
            "changedir": "game-root",
            "crt_rand_seed": 11,
        },
        "trace": {
            "attach_at_update": 0,
            "detach_at_update": 7000,
            "reattach_at_update": 7300,
            "maximum_hits": 10000,
            "trace_timeout_seconds": 900.0,
            "launch_timeout_seconds": 30.0,
            "allow_pre_stream_commands": True,
            "startup_trace_handoff": True,
        },
    }
    oracle = tmp_path / "oracle.json"
    result = tmp_path / "attempt-01" / "strict-replay.json"

    args = collector._trace_args(
        project_root=tmp_path,
        plan=plan,
        dmo=tmp_path / "input.dmo",
        result_path=result,
        gameplay_mtrand_oracle=oracle,
        gameplay_mtrand_oracle_seed=85961375,
        gameplay_mtrand_oracle_maximum_draws=20000,
    )

    assert args[args.index("--gameplay-mtrand-oracle") + 1] == str(oracle)
    assert args[args.index("--gameplay-mtrand-oracle-seed") + 1] == (
        "85961375"
    )
    assert args[
        args.index("--gameplay-mtrand-oracle-maximum-draws") + 1
    ] == "20000"
    assert args[args.index("--gameplay-mtrand-receipt-json") + 1] == str(
        result.parent / "gameplay-mtrand-sync.json"
    )


def test_trace_args_add_initial_global_mtrand_seed_and_receipt(
    tmp_path: Path,
) -> None:
    plan = {
        "runtime": {
            "runtime_executable": "runtime.exe",
            "changedir": "game-root",
            "crt_rand_seed": 11,
        },
        "trace": {
            "attach_at_update": 0,
            "detach_at_update": 1702,
            "reattach_at_update": 3501,
            "maximum_hits": 10000,
            "trace_timeout_seconds": 900.0,
            "launch_timeout_seconds": 30.0,
            "allow_pre_stream_commands": True,
            "startup_trace_handoff": True,
        },
    }
    oracle = tmp_path / "global-calls.json"
    result = tmp_path / "attempt-01" / "strict-replay.json"

    args = collector._trace_args(
        project_root=tmp_path,
        plan=plan,
        dmo=tmp_path / "input.dmo",
        result_path=result,
        initial_global_mtrand_oracle=oracle,
        initial_global_mtrand_seed=23557968,
    )

    assert args[args.index("--initial-global-mtrand-oracle") + 1] == str(
        oracle
    )
    assert args[args.index("--initial-global-mtrand-seed") + 1] == (
        "23557968"
    )
    assert args[
        args.index("--initial-global-mtrand-receipt-json") + 1
    ] == str(result.parent / "initial-global-mtrand-seed.json")


def test_trace_args_add_startup_global_mtrand_observation_receipt(
    tmp_path: Path,
) -> None:
    plan = {
        "runtime": {
            "runtime_executable": "runtime.exe",
            "changedir": "assets",
            "crt_rand_seed": 11,
            "startup_seed_transport": "debugger_register",
        },
        "trace": {
            "attach_at_update": 0,
            "detach_at_update": 2000,
            "reattach_at_update": 3000,
            "maximum_hits": 100,
            "trace_timeout_seconds": 10,
            "launch_timeout_seconds": 10,
            "allow_pre_stream_commands": True,
            "startup_trace_handoff": True,
        },
    }
    oracle = tmp_path / "global-calls.json"
    result = tmp_path / "attempt-01" / "strict-replay.json"

    args = collector._trace_args(
        project_root=tmp_path,
        plan=plan,
        dmo=tmp_path / "input.dmo",
        result_path=result,
        startup_global_mtrand_observation_oracle=oracle,
        startup_global_mtrand_observation_seed=23557968,
        startup_global_mtrand_observation_end_at_update=1702,
        startup_global_mtrand_observation_maximum_draws=20000,
        startup_global_mtrand_hidden_draw_start_after_source_order=308,
        startup_global_mtrand_hidden_draw_stop_before_source_order=309,
        disable_source_bound_board_anchor=True,
    )

    assert args[
        args.index("--startup-global-mtrand-observation-oracle") + 1
    ] == str(oracle)
    assert args[
        args.index("--startup-global-mtrand-observation-seed") + 1
    ] == "23557968"
    assert args[
        args.index(
            "--startup-global-mtrand-observation-end-at-update"
        )
        + 1
    ] == "1702"
    assert args[
        args.index(
            "--startup-global-mtrand-observation-maximum-draws"
        )
        + 1
    ] == "20000"
    assert args[
        args.index(
            "--startup-global-mtrand-observation-receipt-json"
        )
        + 1
    ] == str(result.parent / "startup-global-mtrand-observation.json")
    assert args[
        args.index(
            "--startup-global-mtrand-hidden-draw-start-after-source-order"
        )
        + 1
    ] == "308"
    assert args[
        args.index(
            "--startup-global-mtrand-hidden-draw-stop-before-source-order"
        )
        + 1
    ] == "309"


def test_trace_args_suspend_main_thread_for_debugger_handoff(
    tmp_path: Path,
) -> None:
    plan = {
        "runtime": {
            "runtime_executable": "runtime.exe",
            "changedir": "game-root",
            "crt_rand_seed": 11,
        },
        "trace": {
            "attach_at_update": 0,
            "detach_at_update": 1702,
            "reattach_at_update": 3501,
            "maximum_hits": 10000,
            "trace_timeout_seconds": 900.0,
            "launch_timeout_seconds": 30.0,
            "allow_pre_stream_commands": True,
            "startup_trace_handoff": True,
        },
    }

    args = collector._trace_args(
        project_root=tmp_path,
        plan=plan,
        dmo=tmp_path / "input.dmo",
        result_path=tmp_path / "strict-replay.json",
        suspend_main_thread_on_stop=True,
    )

    assert "--suspend-main-thread-on-stop" in args


def test_startup_main_thread_identity_is_bound_to_runtime_pid(
    tmp_path: Path,
) -> None:
    log = tmp_path / "strict-replay.log"
    log.write_text(
        "startup_rng seed=11 transport=debugger_register "
        "breakpoint=0x00401000 pid=123 tid=456\n",
        encoding="utf-8",
    )
    process = SimpleNamespace(poll=lambda: None, returncode=None)

    observed = collector._wait_for_startup_main_thread_id(
        trace_log_path=log,
        trace_process=process,
        expected_process_id=123,
        timeout_seconds=1.0,
    )

    assert observed == 456


def test_gameplay_mtrand_sync_receipt_is_provenance_bound(
    tmp_path: Path,
) -> None:
    oracle = tmp_path / "oracle.json"
    oracle.write_bytes(b"oracle\n")
    dmo = tmp_path / "input.dmo"
    dmo.write_bytes(b"dmo\n")
    receipt_path = tmp_path / "gameplay-mtrand-sync.json"
    payload = {
        "schema": "zuma.popcap_gameplay_mtrand_sync_receipt.v1",
        "status": "PASS",
        "classification": (
            "diagnostic_ephemeral_process_state_synchronization"
        ),
        "runtime_process_id": 123,
        "process_context_mutation": True,
        "persistent_file_modified": False,
        "source_dmo": {"sha256": collector._sha256_path(dmo)},
        "receipt": {
            "status": "PASS",
            "failure": None,
            "oracle_complete": True,
            "hardware_breakpoint_armed": False,
            "hardware_breakpoint_restored": True,
            "hardware_breakpoint_restore_error": None,
            "expected_hit_count": 2,
            "hit_count": 2,
            "register_mutation_count": 2,
            "state_correction_count": 1,
            "bytes_written_total": 2500,
            "oracle": {
                "source_path": str(oracle.resolve()),
                "source_sha256": collector._sha256_path(oracle),
                "entry_count": 2,
                "first_update": 10,
                "last_update": 11,
            },
        },
        "observations": [
            {"order": 0, "framework_update": 10},
            {"order": 1, "framework_update": 11},
        ],
        "pre_blackout_trace_result": {
            "failure_count": 0,
            "stopped_at_update": True,
            "last_update": 12,
        },
    }
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")

    evidence = collector._load_gameplay_mtrand_sync_receipt(
        receipt_path,
        expected_process_id=123,
        expected_oracle=oracle,
        expected_dmo=dmo,
    )

    assert evidence["status"] == "PASS"
    assert evidence["observation_count"] == 2
    assert evidence["artifact_sha256"] == collector._sha256_path(
        receipt_path
    )


def test_gameplay_mtrand_sync_receipt_rejects_unrestored_breakpoint(
    tmp_path: Path,
) -> None:
    oracle = tmp_path / "oracle.json"
    oracle.write_bytes(b"oracle\n")
    dmo = tmp_path / "input.dmo"
    dmo.write_bytes(b"dmo\n")
    receipt_path = tmp_path / "gameplay-mtrand-sync.json"
    payload = {
        "schema": "zuma.popcap_gameplay_mtrand_sync_receipt.v1",
        "status": "PASS",
        "runtime_process_id": 123,
        "process_context_mutation": True,
        "persistent_file_modified": False,
        "source_dmo": {"sha256": collector._sha256_path(dmo)},
        "receipt": {
            "status": "PASS",
            "failure": None,
            "oracle_complete": True,
            "hardware_breakpoint_armed": True,
            "hardware_breakpoint_restored": False,
            "hardware_breakpoint_restore_error": None,
            "expected_hit_count": 1,
            "hit_count": 1,
            "register_mutation_count": 1,
            "state_correction_count": 0,
            "bytes_written_total": 0,
            "oracle": {
                "source_path": str(oracle.resolve()),
                "source_sha256": collector._sha256_path(oracle),
                "entry_count": 1,
                "first_update": 10,
                "last_update": 10,
            },
        },
        "observations": [{"order": 0, "framework_update": 10}],
        "pre_blackout_trace_result": {
            "failure_count": 0,
            "stopped_at_update": True,
            "last_update": 12,
        },
    }
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ProbeError,
        match="gameplay_mtrand_sync_receipt_contract_failed",
    ):
        collector._load_gameplay_mtrand_sync_receipt(
            receipt_path,
            expected_process_id=123,
            expected_oracle=oracle,
            expected_dmo=dmo,
        )


def test_initial_global_mtrand_seed_receipt_is_source_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oracle_path = tmp_path / "global-calls.json"
    oracle_path.write_bytes(b"oracle\n")
    dmo = tmp_path / "input.dmo"
    dmo.write_bytes(b"dmo\n")
    pre_hash = "sha256:" + "1" * 64
    post_hash = "sha256:" + "2" * 64
    semantic_hash = "sha256:" + "3" * 64
    expected = SimpleNamespace(
        source_path=oracle_path.resolve(),
        source_sha256=collector._sha256_path(oracle_path),
        source_process_id=111,
        source_main_thread_id=222,
        runtime_executable_sha256="sha256:" + "4" * 64,
        seed=23557968,
        semantic_sha256=semantic_hash,
        wrapper_address=0x617490,
        call_address=0x4E4A61,
        caller=0x4E4A66,
        output=1030539336,
        pre_index=624,
        pre_state_sha256=pre_hash,
        post_index=1,
        post_state_sha256=post_hash,
    )
    monkeypatch.setattr(
        collector,
        "load_initial_global_mtrand_call_oracle",
        lambda path, *, seed: expected,
    )
    oracle_row = {
        "source_path": str(expected.source_path),
        "source_sha256": expected.source_sha256,
        "source_process_id": expected.source_process_id,
        "source_main_thread_id": expected.source_main_thread_id,
        "runtime_executable_sha256": expected.runtime_executable_sha256,
        "seed": expected.seed,
        "semantic_sha256": expected.semantic_sha256,
        "wrapper_address": expected.wrapper_address,
        "call_address": expected.call_address,
        "caller": expected.caller,
        "output": expected.output,
        "pre_index": expected.pre_index,
        "pre_state_sha256": expected.pre_state_sha256,
        "post_index": expected.post_index,
        "post_state_sha256": expected.post_state_sha256,
    }
    observation = {
        "thread_id": 333,
        "source_path": str(expected.source_path),
        "source_sha256": expected.source_sha256,
        "source_process_id": expected.source_process_id,
        "source_main_thread_id": expected.source_main_thread_id,
        "semantic_sha256": expected.semantic_sha256,
        "seed": expected.seed,
        "call_address": expected.call_address,
        "call_instruction_hex": "e800000000",
        "dispatch_target": 0x401530,
        "wrapper_address": expected.wrapper_address,
        "wrapper_instruction_hex": "bac013a300",
        "caller": expected.caller,
        "wrapper_entry_stack_return_address": expected.caller,
        "expected_output": expected.output,
        "observed_output": expected.output,
        "live_pre_index": 77,
        "live_pre_state_sha256": "sha256:" + "5" * 64,
        "restored_pre_index": expected.pre_index,
        "restored_pre_state_sha256": expected.pre_state_sha256,
        "expected_post_index": expected.post_index,
        "expected_post_state_sha256": expected.post_state_sha256,
        "observed_post_index": expected.post_index,
        "observed_post_state_sha256": expected.post_state_sha256,
        "changed": True,
        "bytes_written": 2500,
        "other_threads_suspended_count": 4,
        "atomic_window_completed": True,
        "wrapper_entry_observed": True,
        "call_target_entered": True,
        "call_return_observed": True,
        "other_threads_resumed_after_call_return": True,
        "process_memory_mutation": True,
        "process_context_mutation": True,
        "persistent_file_modified": False,
    }
    payload = {
        "schema": "zuma.popcap_initial_global_mtrand_seed_receipt.v1",
        "status": "PASS",
        "classification": "diagnostic_source_bound_rng_initialization",
        "runtime_process_id": 123,
        "source_dmo": {
            "path": str(dmo.resolve()),
            "size_bytes": dmo.stat().st_size,
            "sha256": collector._sha256_path(dmo),
        },
        "oracle": oracle_row,
        "state_correction_count": 1,
        "bytes_written_total": 2500,
        "process_memory_mutation": True,
        "process_context_mutation": True,
        "persistent_file_modified": False,
        "observations": [observation],
        "pre_blackout_trace_result": {
            "failure_count": 0,
            "stopped_at_update": True,
            "last_update": 0,
        },
    }
    receipt = tmp_path / "initial-global-mtrand-seed.json"
    receipt.write_text(json.dumps(payload), encoding="utf-8")

    evidence = collector._load_initial_global_mtrand_seed_receipt(
        receipt,
        expected_process_id=123,
        expected_oracle=oracle_path,
        expected_seed=23557968,
        expected_dmo=dmo,
    )

    assert evidence["status"] == "PASS"
    assert evidence["state_correction_count"] == 1
    assert evidence["bytes_written_total"] == 2500
    assert evidence["observation"][
        "other_threads_resumed_after_call_return"
    ] is True

    payload["observations"][0][
        "other_threads_resumed_after_call_return"
    ] = False
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        ProbeError,
        match="initial_global_mtrand_seed_receipt_contract_failed",
    ):
        collector._load_initial_global_mtrand_seed_receipt(
            receipt,
            expected_process_id=123,
            expected_oracle=oracle_path,
            expected_seed=23557968,
            expected_dmo=dmo,
        )


def test_reference_target_expansion_is_bounded_and_deterministic() -> None:
    hits = [
        {
            "image": False,
            "address": 0x900000 + index * 8,
            "candidate_objects": [
                {"object_address": 0x500000 + index * 8}
            ],
        }
        for index in reversed(range(MAX_POINTER_REFERENCE_TARGETS + 40))
    ]
    hits.append(
        {
            "image": True,
            "address": 0x401000,
            "candidate_objects": [{"object_address": 0x402000}],
        }
    )

    selected, total = _bounded_reference_targets(hits)

    assert total == 2 * (MAX_POINTER_REFERENCE_TARGETS + 40)
    assert len(selected) == MAX_POINTER_REFERENCE_TARGETS
    assert selected == tuple(sorted(selected))
    assert selected[0] == 0x500000
    assert 0x401000 not in selected


def test_reference_target_expansion_omits_non_wow64_addresses() -> None:
    hits = [
        {
            "image": False,
            "address": 0x1_0000_1000,
            "candidate_objects": [
                {"object_address": 0x1_0000_2000},
                {"object_address": 0x700000},
            ],
        }
    ]

    selected, total = _bounded_reference_targets(hits)

    assert selected == (0x700000,)
    assert total == 1


def test_replay_state_row_is_lossless() -> None:
    state = _state()

    row = _replay_state_row(state)

    assert row["update_count"] == state.update_count
    assert row["update_multiplier"] == state.update_multiplier
    assert row["loaded"] is True
    assert row["fast_forward_step"] is False


def test_trajectory_sample_barrier_distinguishes_freeze_and_step() -> None:
    frozen = _state(100)
    stepped = replace(frozen, update_count=101, fast_forward_target=101)

    assert (
        _trajectory_sample_barrier(
            frozen,
            update=100,
            start_update=100,
            frozen_state=frozen,
        )
        == "freeze_multiplier_message"
    )
    assert (
        _trajectory_sample_barrier(
            stepped,
            update=101,
            start_update=100,
            frozen_state=frozen,
        )
        == "replay_fast_forward_step"
    )


def test_trajectory_sample_barrier_rejects_unfinished_step() -> None:
    frozen = _state(100)
    unfinished = replace(
        frozen,
        update_count=101,
        fast_forward_target=101,
        fast_forward_step=True,
    )

    with pytest.raises(ProbeError, match="trajectory_replay_state_mismatch"):
        _trajectory_sample_barrier(
            unfinished,
            update=101,
            start_update=100,
            frozen_state=frozen,
        )


def test_curve_payload_layout_accepts_ball_and_insertion_bullet() -> None:
    assert _curve_payload_layout(BALL_VTABLE) == (
        "ball",
        BALL_OBJECT_SIZE,
    )
    assert _curve_payload_layout(BULLET_VTABLE) == (
        "bullet",
        BULLET_OBJECT_SIZE,
    )
    with pytest.raises(
        ProbeError,
        match="curve_list_payload_vtable_mismatch",
    ):
        _curve_payload_layout(0x12345678)


def _gap_bullet() -> bytes:
    payload = bytearray(BULLET_OBJECT_SIZE)
    struct.pack_into("<I", payload, 0, BULLET_VTABLE)
    struct.pack_into("<I", payload, 0x174, 0x1000)
    struct.pack_into("<I", payload, 0x178, 1)
    struct.pack_into("<4i", payload, 0x17C, 109, 0, 0, 0)
    return bytes(payload)


def test_bullet_gap_state_uses_retail_list_and_curve_point_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory = {
        0x1000: struct.pack("<II", 0x2000, 0x2000),
        0x2000: struct.pack("<IIiii", 0x1000, 0x1000, 0, 120, 2),
    }

    monkeypatch.setattr(
        collector,
        "read_process_bytes",
        lambda handle, address, size: memory[address][:size],
    )

    state, raw = _collect_bullet_gap_state(
        1,
        bullet=_gap_bullet(),
        regions=((0x1000, 0x2000),),
    )

    assert state["gap_entry_count"] == 1
    assert state["curve_points"] == [109, 0, 0, 0]
    assert state["gap_entries"][0]["gap_distance"] == 120
    assert raw == memory[0x2000]
    subclass = _decode_bullet_subclass(_gap_bullet())
    assert subclass["gap_list_sentinel_address"] == 0x1000
    assert subclass["curve_points"][0] == 109


def test_bullet_gap_state_rejects_broken_previous_link(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory = {
        0x1000: struct.pack("<II", 0x2000, 0x2000),
        0x2000: struct.pack("<IIiii", 0x1000, 0x2222, 0, 120, 2),
    }
    monkeypatch.setattr(
        collector,
        "read_process_bytes",
        lambda handle, address, size: memory[address][:size],
    )

    with pytest.raises(ProbeError, match="bullet_gap_previous_link_mismatch"):
        _collect_bullet_gap_state(
            1,
            bullet=_gap_bullet(),
            regions=((0x1000, 0x2000),),
        )


def test_curve_plan_freezes_vector_and_add_plan_flag(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    curve = bytearray(CURVE_DUMP_SIZE)
    struct.pack_into(
        "<III",
        curve,
        CURVE_PLANNED_VECTOR_BEGIN_OFFSET,
        0x2000,
        0x2028,
        0x2050,
    )
    curve[CURVE_ADD_PLAN_ENABLED_OFFSET] = 1
    planned = bytes(range(0x28))
    monkeypatch.setattr(
        collector,
        "read_process_bytes",
        lambda handle, address, size: planned[:size],
    )

    state = _collect_curve_plan(
        1,
        curve=bytes(curve),
        curve_index=0,
        regions=((0x2000, 0x100),),
        output_root=tmp_path,
    )

    assert state["count"] == 2
    assert state["capacity_count"] == 4
    assert state["add_plan_enabled"] is True
    assert state["artifact_bytes"] == len(planned)
    assert (tmp_path / state["artifact"]).read_bytes() == planned


def test_curve_plan_rejects_misaligned_vector(tmp_path: Path) -> None:
    curve = bytearray(CURVE_DUMP_SIZE)
    struct.pack_into(
        "<III",
        curve,
        CURVE_PLANNED_VECTOR_BEGIN_OFFSET,
        0x2000,
        0x2001,
        0x2014,
    )

    with pytest.raises(ProbeError, match="curve_plan_vector_alignment_invalid"):
        _collect_curve_plan(
            1,
            curve=bytes(curve),
            curve_index=0,
            regions=((0x2000, 0x100),),
            output_root=tmp_path,
        )


def test_trajectory_rejects_reverse_range_before_windows_access(
    tmp_path: Path,
) -> None:
    with pytest.raises(ProbeError, match="trajectory_update_range_invalid"):
        collect_board_trajectory(
            1,
            frozen_state=_state(),
            end_update=99,
            output_root=tmp_path / "trajectory",
        )

    with pytest.raises(ProbeError, match="trajectory_update_range_invalid"):
        collector.collect_rng_trajectory(
            1,
            frozen_state=_state(),
            record_start_update=99,
            end_update=100,
            output_root=tmp_path / "rng-trajectory",
        )


def test_rng_trajectory_step_timeout_default_is_legacy_five_seconds() -> None:
    assert (
        collector.collect_rng_trajectory.__kwdefaults__[
            "trajectory_step_timeout_seconds"
        ]
        == 5.0
    )


@pytest.mark.parametrize(
    "value",
    [0.0, -1.0, float("nan"), float("inf"), True, 3600.1],
)
def test_rng_trajectory_rejects_invalid_step_timeout_before_windows_access(
    tmp_path: Path,
    value: object,
) -> None:
    with pytest.raises(ProbeError, match="trajectory_step_timeout_invalid"):
        collector.collect_rng_trajectory(
            1,
            frozen_state=_state(),
            end_update=100,
            output_root=tmp_path / "rng-trajectory",
            trajectory_step_timeout_seconds=value,  # type: ignore[arg-type]
        )


def test_rng_trajectory_arms_deferred_trace_after_warmup_step(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frozen = _state(100)
    events: list[str] = []
    step_timeouts: list[tuple[int, float]] = []
    monkeypatch.setattr(collector, "main_window_for_pid", lambda _pid: 7)
    monkeypatch.setattr(collector, "open_process_readonly", lambda _pid: 8)
    monkeypatch.setattr(collector, "close_process", lambda _handle: None)
    monkeypatch.setattr(
        collector,
        "read_replay_state",
        lambda _handle, _address: frozen,
    )

    def fake_step_to(*args: object, **kwargs: object) -> ReplayState:
        target = int(args[3])
        timeout = float(kwargs["timeout_per_step"])
        step_timeouts.append((target, timeout))
        events.append(f"step:{target}")
        return replace(
            frozen,
            update_count=target,
            fast_forward_target=target,
        )

    monkeypatch.setattr(collector, "step_to", fake_step_to)

    def fake_tick(
        _handle: int,
        *,
        update: int,
        thread_crt_state: object,
    ) -> tuple[dict[str, object], bytes]:
        assert update in {101, 102}
        assert thread_crt_state is None
        events.append(f"read:{update}")
        return (
            {
                "current_ball": None,
                "next_ball": None,
                "chain_ball_count": 0,
                "pending_ball_count": 0,
                "inserting_ball_count": 0,
                "fired_bullet_count": 0,
                "qrand": {"update_count": 0},
                "global_mtrand_index": 3,
                "score": 10,
            },
            bytes(collector.MTRAND_STATE_BYTES),
        )

    monkeypatch.setattr(collector, "_read_rng_trajectory_tick", fake_tick)

    collector.collect_rng_trajectory(
        123,
        frozen_state=frozen,
        record_start_update=101,
        end_update=102,
        output_root=tmp_path / "trajectory",
        on_record_start=lambda: events.append("arm"),
        trajectory_step_timeout_seconds=300.0,
    )

    assert events == ["step:101", "arm", "read:101", "step:102", "read:102"]
    assert step_timeouts == [(101, 300.0), (102, 300.0)]


def test_diagnostic_thread_crt_mutator_is_source_bound_and_verified(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "thread-crt.bin"
    source.write_bytes(struct.pack("<I", 0x12345678))
    memory = {0x2000: struct.pack("<I", 0x87654321)}
    closed: list[int] = []
    monkeypatch.setattr(
        collector,
        "kernel32",
        SimpleNamespace(
            OpenProcess=lambda *_args: 99,
            CloseHandle=lambda handle: closed.append(handle),
        ),
    )
    monkeypatch.setattr(
        collector,
        "read_memory",
        lambda _process, address, size: memory[address][:size],
    )
    monkeypatch.setattr(
        collector,
        "write_memory",
        lambda _process, address, payload: memory.__setitem__(
            address,
            bytes(payload),
        ),
    )
    mutate = collector._diagnostic_thread_crt_state_mutator(
        source_path=source,
        source_framework_update=3615,
    )

    receipt = mutate(
        123,
        {
            "thread_crt_rand": {
                "rand_state_address": 0x2000,
                "state": 0x87654321,
            }
        },
        tmp_path,
    )

    assert memory[0x2000] == struct.pack("<I", 0x12345678)
    assert receipt["classification"] == (
        "diagnostic_process_memory_mutation_not_pc_evidence"
    )
    assert receipt["source_framework_update"] == 3615
    assert receipt["live_before_state"] == 0x87654321
    assert receipt["restored_state"] == 0x12345678
    assert receipt["bytes_written"] == 4
    assert receipt["writeback_verified"] is True
    assert closed == [99]


def test_diagnostic_compact_difference_contract_is_exact() -> None:
    source = {
        "board_color_counts": [5, 21, 13, 18, 0, 0],
        "bullets": [{"ball": {"color_id": 1}}, {"ball": {"color_id": 0}}],
        "crt": 514352131,
        "curves": [
            {
                "lists": [
                    {"entities": [{"ball": {"color_id": 2}}]},
                ]
            }
        ],
        "qrand": {
            "selected_index": 0,
            "vectors": {
                "last_hit": [3, 2, 1, 0, 0, 0],
                "previous_hit": [0, 0, 0, 0, 0, 0],
                "sways": [0.1640625, 0.08325, 0.1171875, 0.1640625, 0.0, 0.0],
            },
        },
    }
    target = json.loads(json.dumps(source))
    target["board_color_counts"][2:4] = [12, 19]
    target["bullets"][1]["ball"]["color_id"] = 1
    target["crt"] = 3067223990
    target["curves"][0]["lists"][0]["entities"][0]["ball"][
        "color_id"
    ] = 3
    target["qrand"]["selected_index"] = 1
    target["qrand"]["vectors"]["last_hit"] = [0, 3, 0, 1, 0, 0]
    target["qrand"]["vectors"]["previous_hit"][1] = 2
    target["qrand"]["vectors"]["sways"][2:4] = [0.1640625, 0.1171875]

    paths = collector._diagnostic_difference_paths(source, target)

    assert set(paths) == set(
        collector.DIAGNOSTIC_COMPACT_STATE_EXPECTED_PRE_DIFFERENCES
    )
    assert set(paths).issubset(
        collector.DIAGNOSTIC_COMPACT_STATE_ALLOWED_PRE_DIFFERENCES
    )
    unexpected = json.loads(json.dumps(target))
    unexpected["bullets"][0]["ball"]["color_id"] = 4
    assert set(collector._diagnostic_difference_paths(source, unexpected)) != set(
        collector.DIAGNOSTIC_COMPACT_STATE_EXPECTED_PRE_DIFFERENCES
    )


def test_diagnostic_compact_write_plan_is_fixed_116_bytes(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()

    def artifact(root: Path, name: str, payload: bytes) -> dict[str, object]:
        (root / name).write_bytes(payload)
        return {
            "artifact": name,
            "artifact_bytes": len(payload),
            "artifact_sha256": collector._sha256_bytes(payload),
        }

    source_board_raw = bytearray(collector.BOARD_DUMP_SIZE)
    target_board_raw = bytearray(collector.BOARD_DUMP_SIZE)
    struct.pack_into("<6i", source_board_raw, collector.BOARD_COLOR_COUNTS_OFFSET, 5, 21, 13, 18, 0, 0)
    struct.pack_into("<6i", target_board_raw, collector.BOARD_COLOR_COUNTS_OFFSET, 5, 21, 12, 19, 0, 0)
    source_qrand_raw = bytearray(collector.QRAND_OBJECT_SIZE)
    target_qrand_raw = bytearray(collector.QRAND_OBJECT_SIZE)
    struct.pack_into("<i", target_qrand_raw, 4, 1)
    source_next_raw = bytearray(collector.BULLET_OBJECT_SIZE)
    target_next_raw = bytearray(collector.BULLET_OBJECT_SIZE)
    struct.pack_into("<i", target_next_raw, collector.BALL_COLOR_ID_OFFSET, 1)
    source_insertion_raw = bytearray(collector.BULLET_OBJECT_SIZE)
    target_insertion_raw = bytearray(collector.BULLET_OBJECT_SIZE)
    struct.pack_into("<i", source_insertion_raw, collector.BALL_COLOR_ID_OFFSET, 2)
    struct.pack_into("<i", target_insertion_raw, collector.BALL_COLOR_ID_OFFSET, 3)

    source_vector_payloads = {
        "weights": bytes(24),
        "sways": bytes(range(24)),
        "last_hit": bytes(range(24, 48)),
        "previous_hit": bytes(range(48, 72)),
    }
    target_vector_payloads = {
        "weights": bytes(24),
        "sways": bytes(reversed(range(24))),
        "last_hit": bytes(reversed(range(24, 48))),
        "previous_hit": bytes(reversed(range(48, 72))),
    }

    def board(
        root: Path,
        prefix: str,
        *,
        board_raw: bytes,
        qrand_raw: bytes,
        crt_raw: bytes,
        next_raw: bytes,
        insertion_raw: bytes,
        vector_payloads: dict[str, bytes],
        address_base: int,
    ) -> dict[str, object]:
        vectors = []
        for index, name in enumerate(
            ("weights", "sways", "last_hit", "previous_hit")
        ):
            vectors.append(
                {
                    "name": name,
                    "begin": address_base + 0x300 + index * 0x40,
                    **artifact(
                        root,
                        f"{prefix}-{name}.bin",
                        vector_payloads[name],
                    ),
                }
            )
        return {
            "board_address": address_base,
            **artifact(root, f"{prefix}-board.bin", bytes(board_raw)),
            "thread_crt_rand": {
                "rand_state_address": address_base + 0x2000,
                **artifact(root, f"{prefix}-crt.bin", crt_raw),
            },
            "qrand": {
                "address": address_base + 0x100,
                "vectors": vectors,
                **artifact(root, f"{prefix}-qrand.bin", bytes(qrand_raw)),
            },
            "primary_child": {
                "bullets": [
                    {
                        "shooter_pointer_offset": 0x130,
                        "address": address_base + 0x3000,
                        **artifact(
                            root,
                            f"{prefix}-current.bin",
                            bytes(collector.BULLET_OBJECT_SIZE),
                        ),
                    },
                    {
                        "shooter_pointer_offset": 0x134,
                        "address": address_base + 0x4000,
                        **artifact(root, f"{prefix}-next.bin", bytes(next_raw)),
                    },
                ]
            },
            "curve_manager": {
                "curves": [
                    {
                        "intrusive_lists": [
                            {
                                "records": [
                                    {
                                        "artifact_offset": 0,
                                        "payload_address": address_base + 0x5000,
                                    }
                                ],
                                **artifact(
                                    root,
                                    f"{prefix}-insertion.bin",
                                    bytes(insertion_raw),
                                ),
                            }
                        ]
                    }
                ]
            },
        }

    source_board = board(
        source_root,
        "source",
        board_raw=source_board_raw,
        qrand_raw=source_qrand_raw,
        crt_raw=struct.pack("<I", 0x1EA86403),
        next_raw=source_next_raw,
        insertion_raw=source_insertion_raw,
        vector_payloads=source_vector_payloads,
        address_base=0x100000,
    )
    target_board = board(
        target_root,
        "target",
        board_raw=target_board_raw,
        qrand_raw=target_qrand_raw,
        crt_raw=struct.pack("<I", 0xB6D21FB6),
        next_raw=target_next_raw,
        insertion_raw=target_insertion_raw,
        vector_payloads=target_vector_payloads,
        address_base=0x200000,
    )

    operations = collector._diagnostic_compact_write_operations(
        source={"board": source_board, "artifact_root": source_root},
        target_board=target_board,
        target_artifact_root=target_root,
    )

    assert [row["role"] for row in operations] == [
        "board_color_counts",
        "thread_crt_rand_state",
        "qrand_selected_index",
        "qrand_sways",
        "qrand_last_hit",
        "qrand_previous_hit",
        "shooter_current_color",
        "shooter_next_color",
        "insertion_staging_color",
    ]
    assert sum(len(row["after"]) for row in operations) == 116
    spans = sorted(
        (row["address"], row["address"] + len(row["after"]))
        for row in operations
    )
    assert all(left[1] <= right[0] for left, right in zip(spans, spans[1:]))


def test_square_corners_require_trajectory_snapshots(tmp_path: Path) -> None:
    with pytest.raises(
        ProbeError,
        match="trajectory_square_corners_require_snapshots",
    ):
        collector.collect_rng_trajectory(
            1,
            frozen_state=_state(),
            end_update=100,
            output_root=tmp_path / "rng-trajectory",
            square_dwm_corners=True,
        )

    with pytest.raises(
        ProbeError,
        match="trajectory_repaint_mechanism_invalid",
    ):
        collector.collect_rng_trajectory(
            1,
            frozen_state=_state(),
            end_update=100,
            output_root=tmp_path / "bad-repaint-trajectory",
            snapshot_repaint_mechanism="unknown",
        )

    with pytest.raises(
        ProbeError,
        match="trajectory_warmup_requires_snapshots",
    ):
        collector.collect_rng_trajectory(
            1,
            frozen_state=_state(),
            end_update=100,
            output_root=tmp_path / "warmup-trajectory",
            snapshot_warmup_frame=True,
        )

    with pytest.raises(
        ProbeError,
        match="formal_exact_step_contract_incomplete",
    ):
        collector.collect_rng_trajectory(
            1,
            frozen_state=_state(),
            end_update=100,
            output_root=tmp_path / "formal-trajectory",
            snapshot_trajectory_frames=True,
            snapshot_process_name="popcapgame1.exe",
            square_dwm_corners=True,
            snapshot_warmup_frame=True,
            formal_exact_step_evidence=True,
        )


def test_square_dwm_corners_are_set_and_read_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class FakeFunction:
        argtypes: object = None
        restype: object = None

        def __init__(self, name: str) -> None:
            self.name = name

        def __call__(
            self,
            _window: object,
            _attribute: object,
            value: object,
            _size: object,
        ) -> int:
            calls.append(self.name)
            if self.name == "get":
                pointer = collector.ctypes.cast(
                    value,
                    collector.ctypes.POINTER(collector.ctypes.c_int),
                )
                pointer.contents.value = 1
            return 0

    api = SimpleNamespace(
        DwmSetWindowAttribute=FakeFunction("set"),
        DwmGetWindowAttribute=FakeFunction("get"),
    )
    monkeypatch.setattr(
        collector.ctypes,
        "WinDLL",
        lambda *_args, **_kwargs: api,
        raising=False,
    )

    receipt = collector._set_square_dwm_corners(7)

    assert calls == ["set", "get"]
    assert receipt["observed_value"] == 1
    assert receipt["status"] == "verified"

def test_exact_trajectory_snapshots_require_diagnostic_rng_mode(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ProbeError,
        match="trajectory_snapshots_require_diagnostic_rng_trajectory",
    ):
        collector.collect_probe(
            plan_path=tmp_path / "plan.json",
            prestate_path=tmp_path / "pre.json",
            host_restore_path=tmp_path / "restore.json",
            output_root=tmp_path / "out",
            probe_update=100,
            slowdown_update=90,
            int32_value=None,
            maximum_attempts=1,
            trajectory_end_update=100,
            trajectory_mode="full",
            skip_repaint_guard=True,
            skip_frozen_snapshot=True,
            snapshot_trajectory_frames=True,
        )


def test_diagnostic_source_bound_replay_rejects_weakened_contract(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ProbeError,
        match="diagnostic_source_bound_replay_contract_invalid",
    ):
        collector.collect_probe(
            plan_path=tmp_path / "plan.json",
            prestate_path=tmp_path / "pre.json",
            host_restore_path=tmp_path / "restore.json",
            output_root=tmp_path / "out",
            probe_update=100,
            slowdown_update=90,
            int32_value=7950,
            maximum_attempts=1,
            trajectory_end_update=110,
            trajectory_mode="rng",
            skip_repaint_guard=True,
            diagnostic_source_bound_replay=True,
        )


def test_diagnostic_source_bound_replay_accepts_exact_call_trace_slice(
    tmp_path: Path,
) -> None:
    plan = tmp_path / "plan.json"
    plan.write_text('{"schema":"invalid-after-contract"}\n', encoding="ascii")

    with pytest.raises(ProbeError, match="plan_invalid"):
        collector.collect_probe(
            plan_path=plan,
            prestate_path=tmp_path / "pre.json",
            host_restore_path=tmp_path / "restore.json",
            output_root=tmp_path / "out",
            probe_update=100,
            slowdown_update=90,
            int32_value=7950,
            maximum_attempts=1,
            trajectory_start_update=105,
            trajectory_end_update=110,
            trajectory_mode="rng",
            trace_global_rng_calls=True,
            skip_repaint_guard=True,
            skip_frozen_snapshot=True,
            diagnostic_source_bound_replay=True,
        )


def test_diagnostic_source_bound_trace_slice_requires_both_snapshot_skips(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ProbeError,
        match="diagnostic_source_bound_replay_contract_invalid",
    ):
        collector.collect_probe(
            plan_path=tmp_path / "plan.json",
            prestate_path=tmp_path / "pre.json",
            host_restore_path=tmp_path / "restore.json",
            output_root=tmp_path / "out",
            probe_update=100,
            slowdown_update=90,
            int32_value=7950,
            maximum_attempts=1,
            trajectory_start_update=105,
            trajectory_end_update=110,
            trajectory_mode="rng",
            trace_global_rng_calls=True,
            skip_repaint_guard=True,
            skip_frozen_snapshot=False,
            diagnostic_source_bound_replay=True,
        )


def test_rng_trajectory_binds_exact_snapshot_to_tick(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frozen = _state()
    monkeypatch.setattr(collector, "main_window_for_pid", lambda _pid: 7)
    monkeypatch.setattr(
        collector,
        "_wait_for_capture_window",
        lambda _pid: SimpleNamespace(
            process_id=123,
            window_handle=7,
            client_region=(0, 0, 800, 600),
        ),
    )
    activations: list[int] = []
    monkeypatch.setattr(
        collector,
        "_activate_window",
        activations.append,
    )
    monkeypatch.setattr(
        collector,
        "_window_repaint_handshake",
        lambda _target: {
            "mechanism": "nonclient_titlebar_drag_then_exact_origin_restore",
            "geometry_restored": True,
        },
    )
    corner_calls: list[int] = []

    def fake_square_corners(window_handle: int) -> dict[str, object]:
        corner_calls.append(window_handle)
        return {
            "schema": "zuma-rl.pc-dwm-corner-preference",
            "version": 1,
            "observed_value": 1,
            "status": "verified",
        }

    monkeypatch.setattr(
        collector,
        "_set_square_dwm_corners",
        fake_square_corners,
    )
    monkeypatch.setattr(collector, "open_process_readonly", lambda _pid: 8)
    monkeypatch.setattr(collector, "close_process", lambda _handle: None)
    monkeypatch.setattr(
        collector,
        "read_replay_state",
        lambda _handle, _address: frozen,
    )
    monkeypatch.setattr(
        collector,
        "_read_rng_trajectory_tick",
        lambda _handle, *, update, thread_crt_state: (
            {
                "current_ball": None,
                "next_ball": None,
                "chain_ball_count": 0,
                "pending_ball_count": 0,
                "inserting_ball_count": 0,
                "fired_bullet_count": 0,
                "qrand": {"update_count": update},
                "global_mtrand_index": 3,
                "score": 10,
            },
            bytes(collector.MTRAND_STATE_BYTES),
        ),
    )

    def fake_snapshot(**kwargs: object) -> dict[str, object]:
        output = Path(str(kwargs["output"]))
        output.write_bytes(b"BMdiagnostic")
        return {
            "output": str(output),
            "process_id": 123,
            "window_handle_hex": "0x0000000000000007",
            "client_region": [0, 0, 800, 600],
            "width": 800,
            "height": 600,
            "bytes": output.stat().st_size,
            "semantics": kwargs.get(
                "evidence_semantics",
                "diagnostic_single_frame_not_formal_evidence",
            ),
        }

    monkeypatch.setattr(collector, "snapshot_dxgi_window", fake_snapshot)
    output_root = tmp_path / "trajectory"

    receipt = collector.collect_rng_trajectory(
        123,
        frozen_state=frozen,
        end_update=frozen.update_count,
        output_root=output_root,
        snapshot_trajectory_frames=True,
        snapshot_process_name="popcapgame1.exe",
        square_dwm_corners=True,
        snapshot_warmup_frame=True,
    )

    index = json.loads((output_root / "index.json").read_text("ascii"))
    snapshot = index["ticks"][0]["visual_snapshot"]
    assert receipt["visual_snapshot_count"] == 1
    assert index["visual_snapshots"]["frame_count"] == 1
    assert snapshot["artifact"] == "frames/u00000100.bmp"
    assert snapshot["capture"]["process_id"] == 123
    assert snapshot["repaint"]["geometry_restored"] is True
    assert "output" not in snapshot["capture"]
    assert snapshot["artifact_sha256"].startswith("sha256:")
    warmup = index["visual_snapshots"]["warmup_frame"]
    assert warmup["artifact"] == "frames/warmup-u00000100.bmp"
    assert warmup["acceptance_role"] == "discarded_dxgi_surface_warmup"
    assert activations == [7, 7]
    assert corner_calls == [7]
    assert (
        index["window_transport"]["square_dwm_corners"]["status"]
        == "verified"
    )

    monkeypatch.setattr(
        collector,
        "_window_position_repaint_handshake",
        lambda _target: {
            "mechanism": (
                "set_window_pos_temporary_translation_then_exact_origin_restore"
            ),
            "geometry_restored": True,
            "synthetic_input_event_count": 0,
        },
    )
    setpos_root = tmp_path / "setpos-trajectory"
    collector.collect_rng_trajectory(
        123,
        frozen_state=frozen,
        end_update=frozen.update_count,
        output_root=setpos_root,
        snapshot_trajectory_frames=True,
        snapshot_process_name="popcapgame1.exe",
        square_dwm_corners=True,
        snapshot_warmup_frame=True,
        snapshot_repaint_mechanism=(
            "set_window_pos_temporary_translation_then_exact_origin_restore"
        ),
    )
    setpos_index = json.loads(
        (setpos_root / "index.json").read_text("ascii")
    )
    assert setpos_index["visual_snapshots"]["repaint_handshake"] == (
        "set_window_pos_temporary_translation_then_exact_origin_restore_per_frame"
    )
    assert setpos_index["ticks"][0]["visual_snapshot"]["repaint"][
        "synthetic_input_event_count"
    ] == 0

    formal_root = tmp_path / "formal-trajectory"
    process_identity = {
        "process_id": 123,
        "process_creation_filetime_100ns": 456,
        "executable_bytes": 789,
        "executable_sha256": "sha256:" + "a" * 64,
        "executable_hash_provenance": {
            "method": "embedded_signed_pe",
            "source_sha256": "sha256:" + "b" * 64,
            "payload_sha256": "sha256:" + "a" * 64,
            "runtime_file_direct_hash_verified": True,
        },
        "window_handle_hex": "0x0000000000000007",
        "window_client_region": [0, 0, 800, 600],
    }
    collector.collect_rng_trajectory(
        123,
        frozen_state=frozen,
        end_update=frozen.update_count,
        output_root=formal_root,
        snapshot_trajectory_frames=True,
        snapshot_process_name="popcapgame1.exe",
        square_dwm_corners=True,
        snapshot_warmup_frame=True,
        process_identity=process_identity,
        formal_exact_step_evidence=True,
    )
    formal_index = json.loads(
        (formal_root / "index.json").read_text("ascii")
    )
    assert formal_index["evidence_classification"] == (
        "formal_pc_golden_exact_step_source"
    )
    assert formal_index["process_identity"] == process_identity
    assert formal_index["visual_snapshots"]["semantics"] == (
        "formal_exact_step_external_lossless_evidence"
    )
    assert formal_index["ticks"][0]["visual_snapshot"]["capture"][
        "semantics"
    ] == "formal_exact_step_external_lossless_evidence"


def test_startup_global_mtrand_observation_receipt_is_source_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oracle_path = tmp_path / "global-calls.json"
    oracle_path.write_bytes(b"oracle\n")
    dmo = tmp_path / "input.dmo"
    dmo.write_bytes(b"dmo\n")
    pre_zero = "sha256:" + "a" * 64
    state_one = "sha256:" + "b" * 64
    state_two = "sha256:" + "c" * 64
    entries = (
        SimpleNamespace(
            source_order=0,
            framework_update=100,
            caller=0x401000,
            output=10,
            pre_draw_count=0,
            pre_index=624,
            pre_state_sha256=pre_zero,
            post_draw_count=1,
            post_index=1,
            post_state_sha256=state_one,
        ),
        SimpleNamespace(
            source_order=1,
            framework_update=101,
            caller=0x401000,
            output=20,
            pre_draw_count=1,
            pre_index=1,
            pre_state_sha256=state_one,
            post_draw_count=2,
            post_index=2,
            post_state_sha256=state_two,
        ),
    )
    expected = SimpleNamespace(
        source_path=oracle_path.resolve(),
        source_sha256=collector._sha256_path(oracle_path),
        source_process_id=111,
        source_main_thread_id=222,
        runtime_executable_sha256="sha256:" + "d" * 64,
        seed=23557968,
        wrapper_address=0x617490,
        end_at_update=101,
        maximum_draws=20,
        semantic_sha256="sha256:" + "e" * 64,
        entries=entries,
    )
    monkeypatch.setattr(
        collector,
        "load_startup_global_mtrand_observation_oracle",
        lambda path, *, seed, end_at_update, maximum_draws: expected,
    )
    oracle_row = {
        "source_path": str(expected.source_path),
        "source_sha256": expected.source_sha256,
        "source_process_id": expected.source_process_id,
        "source_main_thread_id": expected.source_main_thread_id,
        "runtime_executable_sha256": expected.runtime_executable_sha256,
        "seed": expected.seed,
        "wrapper_address": expected.wrapper_address,
        "end_at_update": expected.end_at_update,
        "maximum_draws": expected.maximum_draws,
        "entry_count": 2,
        "semantic_sha256": expected.semantic_sha256,
    }
    observations = []
    for order, entry in enumerate(entries):
        caller = entry.caller if order == 0 else 0x402000
        caller_match = order == 0
        observations.append(
            {
                "order": order,
                "caller": caller,
                "framework_update": entry.framework_update,
                "pre_index": entry.pre_index,
                "pre_state_sha256": entry.pre_state_sha256,
                "inferred_output": entry.output,
                "inferred_post_index": entry.post_index,
                "inferred_post_state_sha256": entry.post_state_sha256,
                "expected_source_order": entry.source_order,
                "expected_framework_update": entry.framework_update,
                "expected_caller": entry.caller,
                "expected_output": entry.output,
                "expected_pre_draw_count": entry.pre_draw_count,
                "expected_pre_index": entry.pre_index,
                "expected_pre_state_sha256": entry.pre_state_sha256,
                "expected_post_draw_count": entry.post_draw_count,
                "expected_post_index": entry.post_index,
                "expected_post_state_sha256": entry.post_state_sha256,
                "caller_match": caller_match,
                "framework_update_match": True,
                "pre_state_match": True,
                "transition_match": True,
                "exact_match": caller_match,
                "observed_pre_draw_count": entry.pre_draw_count,
                "observed_post_draw_count": entry.post_draw_count,
                "observed_hidden_draws_before_call": 0,
                "expected_hidden_draws_before_call": 0,
                "pre_draw_count_delta": 0,
                "post_draw_count_delta": 0,
                "process_memory_writes": 0,
                "process_memory_mutation": False,
                "process_context_mutation": True,
                "persistent_file_modified": False,
            }
        )
    comparison = {
        "observed_call_count": 2,
        "expected_call_count": 2,
        "exact_match_count": 1,
        "mismatch_count": 1,
        "first_exact_mismatch_order": 1,
        "first_caller_mismatch_order": 1,
        "first_framework_update_mismatch_order": None,
        "first_pre_state_mismatch_order": None,
        "first_transition_mismatch_order": None,
        "first_draw_count_mismatch_order": None,
    }
    payload = {
        "schema": (
            "zuma.popcap_startup_global_mtrand_observation_receipt.v1"
        ),
        "status": "PASS",
        "classification": (
            "diagnostic_read_only_startup_rng_schedule_observation"
        ),
        "runtime_process_id": 123,
        "process_memory_writes": 0,
        "process_memory_mutation": False,
        "process_context_mutation": True,
        "persistent_file_modified": False,
        "source_dmo": {
            "path": str(dmo.resolve()),
            "size_bytes": dmo.stat().st_size,
            "sha256": collector._sha256_path(dmo),
        },
        "oracle": oracle_row,
        "receipt": {
            "status": "PASS",
            "failure": None,
            "oracle_complete": True,
            "expected_hit_count": 2,
            "hit_count": 2,
            "hardware_breakpoint_restored": True,
            "hardware_breakpoint_restore_error": None,
            "process_memory_writes": 0,
            "process_memory_mutation": False,
            "process_context_mutation": True,
            "persistent_file_modified": False,
        },
        "draw_count_reconstruction": {
            "status": "PASS",
            "failure": None,
            "maximum_draws": 20,
            "unique_state_count": 3,
            "reconstructed_state_count": 3,
        },
        "comparison": comparison,
        "observations": observations,
        "pre_blackout_trace_result": {
            "failure_count": 0,
            "stopped_at_update": True,
            "last_update": 200,
        },
    }
    receipt_path = tmp_path / "startup-global-mtrand-observation.json"
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")

    evidence = collector._load_startup_global_mtrand_observation_receipt(
        receipt_path,
        expected_process_id=123,
        expected_oracle=oracle_path,
        expected_seed=23557968,
        expected_end_at_update=101,
        expected_maximum_draws=20,
        expected_dmo=dmo,
    )

    assert evidence["status"] == "PASS"
    assert evidence["comparison"] == comparison
    assert evidence["first_mismatch_observation"]["order"] == 1
    assert evidence["process_memory_writes"] == 0

    hidden_contract = {
        "status": "PASS",
        "failure": None,
        "start_after_source_order": 0,
        "start_entry_order": 0,
        "start_framework_update": 100,
        "start_post_draw_count": 1,
        "start_post_state_sha256": state_one,
        "stop_before_source_order": 1,
        "stop_entry_order": 1,
        "stop_framework_update": 101,
        "expected_stop_pre_draw_count": 1,
        "expected_stop_pre_state_sha256": state_one,
        "expected_hidden_draw_count": 0,
        "expected_total_index_write_count": 1,
        "start_boundary_observed": True,
        "stop_boundary_observed": True,
        "watch_armed": True,
        "watch_disarmed": True,
        "index_write_count": 1,
        "observed_hidden_write_count": 0,
        "process_memory_writes": 0,
        "process_memory_mutation": False,
        "process_context_mutation": True,
        "persistent_file_modified": False,
    }
    hidden_summary = {
        "expected_hidden_draw_count": 0,
        "observed_hidden_draw_count": 0,
        "missing_hidden_draw_count": 0,
        "deficit_detected": False,
        "index_write_count": 1,
        "expected_total_index_write_count": 1,
        "start_post_draw_count": 1,
        "first_observed_post_draw_count": 1,
        "expected_stop_pre_draw_count": 1,
        "observed_stop_pre_draw_count": 1,
        "last_observed_post_draw_count": 1,
        "observed_draw_span": 0,
        "observation_orders_valid": True,
        "draw_counts_complete": True,
        "draw_counts_contiguous": True,
    }
    hidden_row = {
        "order": 0,
        "receipt_order": 0,
        "phase": "start_boundary_wrapper_draw",
        "framework_update": 100,
        "stack_return": 0x401000,
        "stack_return_hex": "0x00401000",
        "instruction_pointer": 0x40158A,
        "instruction_pointer_hex": "0x0040158a",
        "post_index": 1,
        "post_state_sha256": state_one,
        "observed_post_draw_count": 1,
        "hardware_breakpoint_slot": 1,
        "hardware_breakpoint_access": "write",
        "hardware_breakpoint_length_bytes": 4,
        "process_memory_writes": 0,
        "process_memory_mutation": False,
        "process_context_mutation": True,
        "persistent_file_modified": False,
    }
    payload["receipt"]["hidden_draw_interval"] = hidden_contract
    payload["hidden_draw_interval"] = {
        "status": "PASS",
        "failures": [],
        "contract": hidden_contract,
        "summary": hidden_summary,
        "caller_histogram": [],
        "process_memory_writes": 0,
        "process_memory_mutation": False,
        "process_context_mutation": True,
        "persistent_file_modified": False,
        "observations": [hidden_row],
    }
    hidden_receipt_path = tmp_path / "startup-hidden-draws.json"
    hidden_receipt_path.write_text(json.dumps(payload), encoding="utf-8")

    hidden_evidence = (
        collector._load_startup_global_mtrand_observation_receipt(
            hidden_receipt_path,
            expected_process_id=123,
            expected_oracle=oracle_path,
            expected_seed=23557968,
            expected_end_at_update=101,
            expected_maximum_draws=20,
            expected_dmo=dmo,
            expected_hidden_draw_start_after_source_order=0,
            expected_hidden_draw_stop_before_source_order=1,
        )
    )

    assert hidden_evidence["hidden_draw_interval"]["status"] == "PASS"
    assert hidden_evidence["hidden_draw_interval"]["summary"] == hidden_summary
    assert hidden_evidence["hidden_draw_interval"]["draw_schedule"][0][
        "observed_post_draw_count"
    ] == 1


def test_source_bound_global_natural_state_accepts_zero_write_match() -> None:
    collector._validate_source_bound_global_natural_state(
        precall_row={
            "changed": False,
            "bytes_written": 0,
            "process_memory_mutation": False,
        },
        global_row={
            "bounded_global_correction_applied": False,
            "natural_post_state_match": True,
            "bytes_written": 0,
            "changed": False,
        },
    )


@pytest.mark.parametrize(
    ("row_name", "field", "value", "expected_error"),
    (
        (
            "precall",
            "changed",
            True,
            "source_bound_global_precall_natural_state_mismatch",
        ),
        (
            "precall",
            "bytes_written",
            2500,
            "source_bound_global_precall_natural_state_mismatch",
        ),
        (
            "precall",
            "process_memory_mutation",
            True,
            "source_bound_global_precall_natural_state_mismatch",
        ),
        (
            "global",
            "bounded_global_correction_applied",
            True,
            "source_bound_global_postcall_natural_state_mismatch",
        ),
        (
            "global",
            "natural_post_state_match",
            False,
            "source_bound_global_postcall_natural_state_mismatch",
        ),
        (
            "global",
            "bytes_written",
            2500,
            "source_bound_global_postcall_natural_state_mismatch",
        ),
        (
            "global",
            "changed",
            True,
            "source_bound_global_postcall_natural_state_mismatch",
        ),
    ),
)
def test_source_bound_global_natural_state_rejects_mutation(
    row_name: str,
    field: str,
    value: object,
    expected_error: str,
) -> None:
    precall_row: dict[str, object] = {
        "changed": False,
        "bytes_written": 0,
        "process_memory_mutation": False,
    }
    global_row: dict[str, object] = {
        "bounded_global_correction_applied": False,
        "natural_post_state_match": True,
        "bytes_written": 0,
        "changed": False,
    }
    target = precall_row if row_name == "precall" else global_row
    target[field] = value

    with pytest.raises(collector.ProbeError, match=expected_error):
        collector._validate_source_bound_global_natural_state(
            precall_row=precall_row,
            global_row=global_row,
        )
