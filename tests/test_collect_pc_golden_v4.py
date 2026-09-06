from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import pytest

from tools.collect_pc_golden_v4 import (
    CollectionError,
    BOARD_MTRAND_OFFSET,
    BOARD_SEED_OWNER_VTABLE,
    BOARD_VTABLE,
    BLACKOUT_FILE_WRITE_REBASE_ENVELOPE_MODE,
    BLACKOUT_FILE_WRITE_REBASE_ENVELOPE_SET_MODE,
    CRT_FLS_INDEX_ADDRESS,
    CRT_PTD_RAND_STATE_OFFSET,
    DEFAULT_BOARD_RESEED_CALL,
    DIRECT_FIXED_SEED_LAUNCH_MODE,
    GLOBAL_MTRAND_ADDRESS,
    MTRAND_STATE_BYTES,
    SOURCE_BOUND_GLOBAL_CORRECTION_MAX_DRAWS,
    STEAM_LAUNCH_MODE,
    _activate_startup_window,
    _gameplay_mtrand_trace_args,
    _load_gameplay_mtrand_sync_receipt,
    _mtrand_state_sha256,
    _post_blackout_overdue_idle_reentry_candidates,
    _capture_duration,
    _capture_frame_budget_fps,
    _inter_run_cooldown_seconds,
    _projected_end_state_root,
    _runtime_screen_mode_overlay,
    _session_member,
    _sha256_file,
    _validate_font_cache_trace_receipt,
    _validate_run_timeline,
    _validate_trace_result,
    _validated_capture_settings,
    _wait_for_repaint_guard,
    _wait_inter_run_cooldown,
    _window_position_repaint_handshake,
    _window_repaint_handshake,
    _write_exclusive,
    build_parser,
)
from tools.capture_dxgi import CaptureError, WindowTarget
from zuma_rl.pc_protocol_evidence import (
    PcStateSnapshot,
    SnapshotRegistryNode,
    SnapshotRegistryValue,
)


_DMO_SHA256 = "sha256:" + "12" * 32
_ZUMA_REGISTRY_ROOT = r"HKCU\Software\SteamPopCap\ZumasRevenge"
_LEGACY_REGISTRY_ROOT = r"HKCU\Software\PopCap\ZumasRevenge"


def test_collector_parser_exposes_transactional_runtime_screen_mode() -> None:
    args = build_parser().parse_args(
        [
            "prepare",
            "--session-root",
            r"D:\ZumaGolden\screen-mode-test",
            "--session-nonce",
            "ab" * 16,
            "--runtime-screen-mode",
            "1",
        ]
    )

    assert args.runtime_screen_mode == 1
    assert _runtime_screen_mode_overlay(1) == {
        "root": _ZUMA_REGISTRY_ROOT,
        "subkey": "",
        "value_name": "ScreenMode",
        "value_type": 4,
        "data_hex": "01000000",
        "screen_mode": 1,
        "semantic": "fullscreen_800x600",
    }


def test_collector_parser_exposes_gameplay_mtrand_sync() -> None:
    args = build_parser().parse_args(
        [
            "prepare",
            "--session-root",
            r"D:\ZumaGolden\gameplay-mtrand-test",
            "--session-nonce",
            "ab" * 16,
            "--gameplay-mtrand-oracle",
            "oracle.json",
            "--gameplay-mtrand-oracle-seed",
            "23557968",
            "--gameplay-mtrand-oracle-maximum-draws",
            "20000",
        ]
    )

    assert args.gameplay_mtrand_oracle == Path("oracle.json")
    assert args.gameplay_mtrand_oracle_seed == 23557968
    assert args.gameplay_mtrand_oracle_maximum_draws == 20000


def test_collector_parser_exposes_startup_process_affinity() -> None:
    args = build_parser().parse_args(
        [
            "prepare",
            "--session-root",
            r"D:\ZumaGolden\startup-affinity-test",
            "--session-nonce",
            "ab" * 16,
            "--startup-process-affinity-mask",
            "0x1",
        ]
    )

    assert args.startup_process_affinity_mask == 1


def test_collector_parser_exposes_bounded_framework_poll_arm() -> None:
    args = build_parser().parse_args(
        [
            "prepare",
            "--session-root",
            r"D:\ZumaGolden\bounded-poll-arm-test",
            "--session-nonce",
            "ab" * 16,
            "--framework-state-poll-arm-update",
            "50",
            "--framework-state-poll-arm-deadline-update",
            "75",
        ]
    )

    assert args.framework_state_poll_arm_update == 50
    assert args.framework_state_poll_arm_deadline_update == 75


def test_gameplay_mtrand_trace_args_bind_frozen_oracle_and_run_receipt(
    tmp_path: Path,
) -> None:
    session_root = tmp_path / "session"
    oracle = session_root / "inputs" / "gameplay-mtrand-oracle.json"
    oracle.parent.mkdir(parents=True)
    oracle.write_bytes(b"oracle\n")
    run_root = session_root / "run-r1"
    trace = {
        "gameplay_mtrand_sync": {
            "artifact": "inputs/gameplay-mtrand-oracle.json",
            "seed": 23557968,
            "maximum_draws": 20000,
        }
    }

    args = _gameplay_mtrand_trace_args(
        session_root=session_root,
        run_root=run_root,
        trace=trace,
    )

    assert args[args.index("--gameplay-mtrand-oracle") + 1] == str(
        oracle
    )
    assert args[args.index("--gameplay-mtrand-oracle-seed") + 1] == (
        "23557968"
    )
    assert args[
        args.index("--gameplay-mtrand-oracle-maximum-draws") + 1
    ] == "20000"
    assert args[args.index("--gameplay-mtrand-receipt-json") + 1] == str(
        run_root / "gameplay-mtrand-sync.json"
    )


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
        "source_dmo": {"sha256": _sha256_file(dmo)},
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
            "bytes_written_total": MTRAND_STATE_BYTES,
            "oracle": {
                "source_path": str(oracle.resolve()),
                "source_sha256": _sha256_file(oracle),
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

    evidence = _load_gameplay_mtrand_sync_receipt(
        receipt_path,
        expected_process_id=123,
        expected_oracle=oracle,
        expected_dmo=dmo,
        artifact="run-r1/gameplay-mtrand-sync.json",
    )

    assert evidence["status"] == "PASS"
    assert evidence["observation_count"] == 2
    assert evidence["artifact_sha256"] == _sha256_file(receipt_path)

    payload["receipt"]["hardware_breakpoint_restored"] = False
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        CollectionError,
        match="gameplay_mtrand_sync_receipt_contract_failed",
    ):
        _load_gameplay_mtrand_sync_receipt(
            receipt_path,
            expected_process_id=123,
            expected_oracle=oracle,
            expected_dmo=dmo,
            artifact="run-r1/gameplay-mtrand-sync.json",
        )


def _demo_command(
    *,
    update: int,
    kind: str,
    command_number: int,
    payload: dict[str, object],
    short_form: bool = False,
) -> argparse.Namespace:
    return argparse.Namespace(
        update=update,
        kind=kind,
        command_number=command_number,
        payload=payload,
        short_form=short_form,
    )


def _overdue_idle_candidate() -> dict[str, object]:
    return {
        "corridor_start_index": 10,
        "corridor_end_index": 17,
        "corridor_row_count": 8,
        "corridor_recorded_update": 100,
        "bridge_row_index": 18,
        "bridge_recorded_update": 115,
        "bridge_kind": "idle",
        "bridge_command_number": 31,
        "bridge_short_form": False,
        "following_row_index": 19,
        "following_recorded_update": 130,
        "following_kind": "idle",
        "following_command_number": 31,
        "following_short_form": False,
    }


def _overdue_idle_receipt() -> dict[str, object]:
    return {
        **_overdue_idle_candidate(),
        "process_id": 55,
        "thread_id": 77,
        "framework_update": 117,
        "last_demo_update": 117,
        "corridor_end_bit_position": 1000,
        "bridge_payload_empty": True,
        "bridge_start_bit_position": 1000,
        "bridge_end_bit_position": 1010,
        "following_payload_empty": True,
        "following_start_bit_position": 1010,
        "following_end_bit_position": 1020,
        "bridge_late_updates": 2,
        "following_remaining_updates": 13,
        "command_order_offset": 4,
        "observed_command_order": 14,
        "reentries": 4,
        "prepared_reentry_verified": True,
        "prepared_framework_update": 117,
        "prepared_last_demo_update": 117,
        "prepared_needs_command": 1,
        "prepared_command_order": 14,
        "prepared_command_bit_position": 1000,
        "prepared_buffer_read_bit_position": 1010,
        "memory_write_bytes": 0,
        "process_memory_mutation": False,
        "verification_mode": (
            "post_blackout_successful_file_write_overdue_idle_reentry"
        ),
    }


@pytest.mark.parametrize("value", ("0.1", "6.1", "nan", "inf"))
def test_collector_rejects_capture_durations_the_child_cannot_run(
    value: str,
) -> None:
    with pytest.raises(
        argparse.ArgumentTypeError,
        match="invalid capture duration",
    ):
        _capture_duration(value)


def test_collector_accepts_bounded_long_horizon_capture() -> None:
    assert _capture_duration("5.5") == 5.5


@pytest.mark.parametrize("value", ("-0.1", "180.1", "nan", "inf"))
def test_collector_rejects_unbounded_inter_run_cooldown(value: str) -> None:
    with pytest.raises(
        argparse.ArgumentTypeError,
        match="inter-run cooldown",
    ):
        _inter_run_cooldown_seconds(value)


def test_collector_waits_for_bounded_inter_run_cooldown() -> None:
    now = [10.0]
    delays: list[float] = []
    stopped_checks: list[float] = []

    def delay(seconds: float) -> None:
        delays.append(seconds)
        now[0] += seconds

    _wait_inter_run_cooldown(
        2.5,
        delay=delay,
        clock=lambda: now[0],
        require_game_stopped=lambda: stopped_checks.append(now[0]),
    )

    assert delays == [1.0, 1.0, 0.5]
    assert stopped_checks == [10.0, 11.0, 12.0, 12.5]


def test_prepare_freezes_exact_post_blackout_overdue_idle_candidate() -> None:
    commands = [
        _demo_command(
            update=100,
            kind="file_write",
            command_number=16,
            payload={"success": True},
        )
        for _ in range(8)
    ]
    commands.extend(
        (
            _demo_command(
                update=115,
                kind="idle",
                command_number=31,
                payload={},
            ),
            _demo_command(
                update=130,
                kind="idle",
                command_number=31,
                payload={},
            ),
        )
    )
    demo = argparse.Namespace(commands=tuple(commands))

    candidates = _post_blackout_overdue_idle_reentry_candidates(
        demo,
        reattach_at_update=90,
    )

    expected = _overdue_idle_candidate()
    expected["corridor_start_index"] = 0
    expected["corridor_end_index"] = 7
    expected["bridge_row_index"] = 8
    expected["following_row_index"] = 9
    assert candidates == [expected]

    commands[-1].update = 129
    assert (
        _post_blackout_overdue_idle_reentry_candidates(
            demo,
            reattach_at_update=90,
        )
        == []
    )


@pytest.mark.parametrize("value", ("59", "241", "60.0"))
def test_collector_rejects_capture_frame_budgets_the_child_cannot_run(
    value: str,
) -> None:
    with pytest.raises(
        argparse.ArgumentTypeError,
        match="invalid capture frame budget",
    ):
        _capture_frame_budget_fps(value)


def test_collector_rejects_impossible_raw_capture_budget_during_prepare() -> None:
    with pytest.raises(CollectionError, match="raw_budget_exceeded"):
        _validated_capture_settings(5.0, 240)

    assert _validated_capture_settings(5.0, 180) == (5.0, 180)


def _snapshot(
    *,
    phase: str,
    last_game_pid: int,
    registration_data: int = 99,
) -> PcStateSnapshot:
    return PcStateSnapshot(
        session_nonce="12" * 16,
        phase=phase,
        captured_perf_counter_ns=1,
        files=(),
        registry_nodes=(
            SnapshotRegistryNode(
                root=_ZUMA_REGISTRY_ROOT,
                subkey="",
                exists=True,
                values=(
                    SnapshotRegistryValue(
                        name="LastGamePID",
                        value_type=4,
                        data=last_game_pid.to_bytes(4, "little"),
                    ),
                    SnapshotRegistryValue(
                        name="RegExData",
                        value_type=4,
                        data=registration_data.to_bytes(4, "little"),
                    ),
                ),
            ),
            SnapshotRegistryNode(
                root=_LEGACY_REGISTRY_ROOT,
                subkey="",
                exists=False,
                values=(),
            ),
        ),
    )


def test_exclusive_writer_never_overwrites(tmp_path: Path) -> None:
    output = tmp_path / "artifact.bin"
    _write_exclusive(output, b"first")
    assert output.read_bytes() == b"first"
    assert not (tmp_path / "artifact.bin.part").exists()

    with pytest.raises(CollectionError, match="output_path_invalid"):
        _write_exclusive(output, b"second")
    assert output.read_bytes() == b"first"


def test_steam_projection_requires_launcher_pid_binding() -> None:
    pre = _snapshot(phase="pre", last_game_pid=12)
    end = _snapshot(phase="end", last_game_pid=55)

    assert _projected_end_state_root(
        end,
        pre_snapshot=pre,
        process_id=55,
        launch_mode=STEAM_LAUNCH_MODE,
    ) == end.projected_state_root(
        (
            (_ZUMA_REGISTRY_ROOT, "", "LastGamePID"),
            (_ZUMA_REGISTRY_ROOT, "", "RegExData"),
        )
    )

    with pytest.raises(
        CollectionError,
        match="end_state_process_binding_invalid",
    ):
        _projected_end_state_root(
            end,
            pre_snapshot=pre,
            process_id=56,
            launch_mode=STEAM_LAUNCH_MODE,
        )


def test_direct_projection_requires_launcher_controls_unchanged() -> None:
    pre = _snapshot(phase="pre", last_game_pid=12)
    unchanged = _snapshot(phase="end", last_game_pid=12)

    _projected_end_state_root(
        unchanged,
        pre_snapshot=pre,
        process_id=55,
        launch_mode=DIRECT_FIXED_SEED_LAUNCH_MODE,
    )

    changed = _snapshot(phase="end", last_game_pid=55)
    with pytest.raises(
        CollectionError,
        match="end_state_launcher_controls_changed",
    ):
        _projected_end_state_root(
            changed,
            pre_snapshot=pre,
            process_id=55,
            launch_mode=DIRECT_FIXED_SEED_LAUNCH_MODE,
        )


@pytest.mark.parametrize(
    "relative",
    (
        "../escape",
        "nested/../escape",
        "/absolute",
        r"windows\separator",
        "./dot",
        "",
    ),
)
def test_session_member_rejects_noncanonical_paths(
    tmp_path: Path,
    relative: str,
) -> None:
    with pytest.raises(CollectionError, match="plan_artifact_path_invalid"):
        _session_member(tmp_path, relative)


def test_trace_result_requires_natural_zero_exit_and_clean_boundaries(
    tmp_path: Path,
) -> None:
    path = tmp_path / "strict-replay.json"
    payload = {
        "schema": "zuma.popcap_strict_replay.v3",
        "runtime_process_id": 55,
        "source_dmo": {
            "sha256": _DMO_SHA256.removeprefix("sha256:"),
        },
        "trace_started_perf_counter_ns": 100,
        "trace_finished_perf_counter_ns": 200,
        "result": {
            "exited": True,
            "exit_code": 0,
            "failure_count": 0,
            "boundary_failures": [],
            "broker_failures": [],
        },
    }
    path.write_text(json.dumps(payload), encoding="ascii")

    decoded = _validate_trace_result(
        path,
        expected_pid=55,
        expected_dmo_sha256=_DMO_SHA256,
    )
    assert decoded["trace_finished_perf_counter_ns"] == 200

    payload["result"]["exit_code"] = 1
    path.write_text(json.dumps(payload), encoding="ascii")
    with pytest.raises(CollectionError, match="trace_result_invalid"):
        _validate_trace_result(
            path,
            expected_pid=55,
            expected_dmo_sha256=_DMO_SHA256,
        )


def test_trace_result_accepts_exact_finite_stop_without_runtime_exit(
    tmp_path: Path,
) -> None:
    path = tmp_path / "strict-replay.json"
    payload = {
        "schema": "zuma.popcap_strict_replay.v3",
        "runtime_process_id": 55,
        "source_dmo": {
            "sha256": _DMO_SHA256.removeprefix("sha256:"),
        },
        "options": {
            "stop_after_update": 3151,
            "stop_after_command_order": None,
        },
        "trace_started_perf_counter_ns": 100,
        "trace_finished_perf_counter_ns": 200,
        "result": {
            "exited": False,
            "exit_code": None,
            "failure_count": 0,
            "boundary_failures": [],
            "broker_failures": [],
            "stopped_at_update": True,
            "stopped_at_command_order": False,
            "last_update": 3151,
        },
    }
    path.write_text(json.dumps(payload), encoding="ascii")

    decoded = _validate_trace_result(
        path,
        expected_pid=55,
        expected_dmo_sha256=_DMO_SHA256,
        expected_stop_after_update=3151,
    )

    assert decoded["result"]["last_update"] == 3151
    payload["result"]["last_update"] = 3150
    path.write_text(json.dumps(payload), encoding="ascii")
    with pytest.raises(CollectionError, match="trace_result_invalid"):
        _validate_trace_result(
            path,
            expected_pid=55,
            expected_dmo_sha256=_DMO_SHA256,
            expected_stop_after_update=3151,
        )


def _source_bound_anchor() -> dict[str, object]:
    return {
        "framework_update": 3120,
        "caller": 0x0065B821,
        "expected_output": 263072237,
        "expected_post_index": 402,
        "expected_post_state_sha256": "sha256:" + "11" * 32,
        "monitor_path": "D:/source/monitor.json",
        "monitor_sha256": "sha256:" + "22" * 32,
        "monitor_framework_update": 3428,
        "global_seed": 85961375,
        "rewind_draws": 4,
        "trace_path": "D:/source/trace.json",
        "trace_sha256": "sha256:" + "33" * 32,
        "recording_report_path": "D:/source/report.json",
        "recording_report_sha256": "sha256:" + "44" * 32,
        "source_dmo_path": "D:/source/input.dmo",
        "source_dmo_sha256": "sha256:" + "55" * 32,
        "source_order": 307,
        "thread_crt_state": 1165183229,
    }


def _source_bound_trace_payload(
    *,
    corrected: bool,
    allow_correction: bool,
) -> tuple[dict[str, object], dict[str, object]]:
    anchor = _source_bound_anchor()
    process_id = 55
    thread_id = 7
    owner_address = 0x120000
    live_index = 393 if corrected else anchor["expected_post_index"]
    live_sha256 = (
        "sha256:" + "aa" * 32
        if corrected
        else anchor["expected_post_state_sha256"]
    )
    observed_seed = 2044941490 if corrected else anchor["expected_output"]
    correction_delta = anchor["expected_post_index"] - live_index
    global_bytes = MTRAND_STATE_BYTES if corrected else 0
    process_memory_mutation = corrected
    global_source = {
        "source_path": anchor["monitor_path"],
        "source_sha256": anchor["monitor_sha256"],
        "source_trace_path": anchor["trace_path"],
        "source_trace_sha256": anchor["trace_sha256"],
        "source_recording_report_path": anchor[
            "recording_report_path"
        ],
        "source_recording_report_sha256": anchor[
            "recording_report_sha256"
        ],
        "source_dmo_path": anchor["source_dmo_path"],
        "source_dmo_sha256": anchor["source_dmo_sha256"],
        "source_call_order": anchor["source_order"],
        "source_caller": anchor["caller"],
    }
    crt_source = {
        "source_path": anchor["trace_path"],
        "source_sha256": anchor["trace_sha256"],
        "source_recording_report_path": anchor[
            "recording_report_path"
        ],
        "source_recording_report_sha256": anchor[
            "recording_report_sha256"
        ],
        "source_dmo_path": anchor["source_dmo_path"],
        "source_dmo_sha256": anchor["source_dmo_sha256"],
        "source_call_order": anchor["source_order"],
        "source_caller": anchor["caller"],
        "source_framework_update": anchor["framework_update"],
        "state": anchor["thread_crt_state"],
    }
    receipt_observation = {
        "classification": (
            "source-bound-bounded-global-correction-board-anchor"
            if corrected
            else "source-bound-natural-retail-post-call-board-anchor"
        ),
        "process_id": process_id,
        "thread_id": thread_id,
        "perf_counter_ns": 205,
        "framework_update": anchor["framework_update"],
        "board_seed_call_address": DEFAULT_BOARD_RESEED_CALL,
        "source_call_return_address": anchor["caller"],
        "instruction_bridge_hex": "8d564c8bc88bc2",
        "instruction_bridge_contains_call": False,
        "observed_seed": observed_seed,
        "effective_seed": anchor["expected_output"],
        "observed_seed_matches_source": not corrected,
        "identity_mechanism": (
            "board_rng_owner_vtable_and_source_call_state"
        ),
        "expected_main_thread_id": thread_id,
        "active_board_address": owner_address,
        "rng_object_address": owner_address + BOARD_MTRAND_OFFSET,
        "rng_owner_address": owner_address,
        "rng_owner_vtable": BOARD_SEED_OWNER_VTABLE,
        "active_board_vtable": BOARD_VTABLE,
        "rng_owner_is_active_board": True,
        "global": {
            "state_address": GLOBAL_MTRAND_ADDRESS,
            "live_before_index": live_index,
            "live_before_state_sha256": live_sha256,
            "natural_post_state_match": not corrected,
            "live_post_draw_verified": True,
            "same_state_words": True,
            "global_correction_authorized": allow_correction,
            "bounded_global_correction_applied": corrected,
            "correction_index_delta": correction_delta,
            "maximum_correction_draw_delta": (
                SOURCE_BOUND_GLOBAL_CORRECTION_MAX_DRAWS
            ),
            "changed": corrected,
            "bytes_written": global_bytes,
            "restored_pre_call_source": global_source,
            "source_call_output": anchor["expected_output"],
            "source_call_post_index": anchor["expected_post_index"],
            "source_call_post_state_sha256": anchor[
                "expected_post_state_sha256"
            ],
        },
        "thread_crt": {
            "thread_id": thread_id,
            "ptd_thread_id": thread_id,
            "restored_state": anchor["thread_crt_state"],
            "changed": False,
            "bytes_written": 0,
            "restored_source": crt_source,
        },
        "bytes_written": global_bytes,
        "writeback_verified": True,
        "transactional_rollback_on_failure": True,
        "process_memory_mutation": process_memory_mutation,
        "persistent_file_modified": False,
    }
    board_observation = {
        "process_id": process_id,
        "thread_id": thread_id,
        "perf_counter_ns": 200,
        "framework_update": anchor["framework_update"],
        "call_site_address": DEFAULT_BOARD_RESEED_CALL,
        "rng_object_address": owner_address + BOARD_MTRAND_OFFSET,
        "rng_owner_address": owner_address,
        "rng_owner_vtable": BOARD_SEED_OWNER_VTABLE,
        "active_board_address": owner_address,
        "active_board_vtable": BOARD_VTABLE,
        "observed_seed": observed_seed,
        "effective_seed": anchor["expected_output"],
        "overridden": True,
        "skipped_before_override": False,
        "source_bound_board_anchor_applied": True,
        "source_bound_board_anchor_bytes_written": global_bytes,
        "source_bound_target_update_match": True,
        "source_bound_main_thread_match": True,
        "source_bound_rng_owner_board_match": True,
        "source_bound_active_board_context_match": True,
        "source_bound_observed_seed_match": not corrected,
        "source_bound_attach_stabilization_clear": True,
        "source_bound_live_global_post_index": live_index,
        "source_bound_live_global_post_sha256": live_sha256,
        "source_bound_expected_global_post_index": anchor[
            "expected_post_index"
        ],
        "source_bound_expected_global_post_sha256": anchor[
            "expected_post_state_sha256"
        ],
        "source_bound_global_post_state_match": not corrected,
        "source_bound_live_post_draw_verified": True,
        "source_bound_correction_index_delta": correction_delta,
        "source_bound_bounded_correction_eligible": corrected,
    }
    payload = {
        "schema": "zuma.popcap_strict_replay.v3",
        "runtime_process_id": process_id,
        "source_dmo": {
            "sha256": _DMO_SHA256.removeprefix("sha256:"),
        },
        "trace_started_perf_counter_ns": 100,
        "trace_finished_perf_counter_ns": 300,
        "options": {
            "allow_source_bound_board_global_correction": (
                allow_correction
            ),
            "board_seed_address": DEFAULT_BOARD_RESEED_CALL,
            "board_seed": anchor["expected_output"],
            "global_rng_seed": None,
            "thread_crt_rng_seed": None,
        },
        "result": {
            "exited": True,
            "exit_code": 0,
            "failure_count": 0,
            "boundary_failures": [],
            "broker_failures": [],
        },
        "board_rng": {
            "call_site_address": DEFAULT_BOARD_RESEED_CALL,
            "effective_seed": anchor["expected_output"],
            "global_rng_seed": None,
            "thread_crt_rng_seed": None,
            "original_instruction_hex": "e8",
            "register_override": "ecx_seed_before_mtrand_srand_call",
            "persistent_file_modified": False,
            "observations": [board_observation],
        },
        "source_bound_board_anchor": {
            "mechanism": (
                "exact_post_global_call_board_constructor_seed_and_"
                "main_thread_crt_anchor"
            ),
            "persistent_file_modified": False,
            "process_memory_mutation": process_memory_mutation,
            "observations": [receipt_observation],
        },
    }
    return payload, anchor


@pytest.mark.parametrize("corrected", (False, True))
def test_source_bound_board_receipt_accepts_natural_or_bounded_correction(
    tmp_path: Path,
    corrected: bool,
) -> None:
    payload, anchor = _source_bound_trace_payload(
        corrected=corrected,
        allow_correction=corrected,
    )
    path = tmp_path / "strict-replay.json"
    path.write_text(json.dumps(payload), encoding="ascii")

    _validate_trace_result(
        path,
        expected_pid=55,
        expected_dmo_sha256=_DMO_SHA256,
        expected_board_seed_address=DEFAULT_BOARD_RESEED_CALL,
        expected_board_seed=anchor["expected_output"],
        expected_source_bound_board_anchor=anchor,
        expected_allow_source_bound_board_global_correction=corrected,
    )


@pytest.mark.parametrize(
    ("mutation_path", "mutation_value"),
    (
        (("global", "same_state_words"), False),
        (("global", "live_post_draw_verified"), False),
        (("global", "correction_index_delta"), 33),
    ),
)
def test_source_bound_board_correction_rejects_invalid_global_receipt(
    tmp_path: Path,
    mutation_path: tuple[str, str],
    mutation_value: object,
) -> None:
    payload, anchor = _source_bound_trace_payload(
        corrected=True,
        allow_correction=True,
    )
    mutated = copy.deepcopy(payload)
    receipt_row = mutated["source_bound_board_anchor"]["observations"][0]
    receipt_row[mutation_path[0]][mutation_path[1]] = mutation_value
    path = tmp_path / "strict-replay.json"
    path.write_text(json.dumps(mutated), encoding="ascii")

    with pytest.raises(CollectionError, match="trace_result_invalid"):
        _validate_trace_result(
            path,
            expected_pid=55,
            expected_dmo_sha256=_DMO_SHA256,
            expected_board_seed_address=DEFAULT_BOARD_RESEED_CALL,
            expected_board_seed=anchor["expected_output"],
            expected_source_bound_board_anchor=anchor,
            expected_allow_source_bound_board_global_correction=True,
        )


def test_source_bound_board_correction_requires_explicit_opt_in(
    tmp_path: Path,
) -> None:
    payload, anchor = _source_bound_trace_payload(
        corrected=True,
        allow_correction=False,
    )
    path = tmp_path / "strict-replay.json"
    path.write_text(json.dumps(payload), encoding="ascii")

    with pytest.raises(CollectionError, match="trace_result_invalid"):
        _validate_trace_result(
            path,
            expected_pid=55,
            expected_dmo_sha256=_DMO_SHA256,
            expected_board_seed_address=DEFAULT_BOARD_RESEED_CALL,
            expected_board_seed=anchor["expected_output"],
            expected_source_bound_board_anchor=anchor,
            expected_allow_source_bound_board_global_correction=False,
        )


def _attach_source_bound_precall_receipt(
    payload: dict[str, object],
    anchor: dict[str, object],
) -> dict[str, object]:
    global_state = {
        "seed": anchor["global_seed"],
        "captured_draw_count": 1653,
        "captured_index": 405,
        "captured_state_sha256": "sha256:" + "66" * 32,
        "rewind_draws": anchor["rewind_draws"],
        "draw_count": 1649,
        "index": 401,
        "state_sha256": "sha256:" + "77" * 32,
        "semantic_sha256": "sha256:" + "88" * 32,
    }
    precall = {
        "call_address": 0x0065B81C,
        "return_address": anchor["caller"],
        "target_address": 0x00617490,
        "instruction_hex": "e86fbcfbff",
        "framework_update": anchor["framework_update"],
        "allow_dynamic_before": True,
        "suspend_other_threads_until_board_call": True,
        "source_process_id": 62068,
        "source_native_game_time": 417,
        "global_state": global_state,
    }
    restored = {
        "source_path": anchor["monitor_path"],
        "source_sha256": anchor["monitor_sha256"],
        "source_process_id": precall["source_process_id"],
        "source_framework_update": anchor["monitor_framework_update"],
        "source_native_game_time": precall["source_native_game_time"],
        **global_state,
        "source_kind": "natural_retail_hardware_trace_and_monitor",
        "source_trace_path": anchor["trace_path"],
        "source_trace_sha256": anchor["trace_sha256"],
        "source_recording_report_path": anchor["recording_report_path"],
        "source_recording_report_sha256": anchor[
            "recording_report_sha256"
        ],
        "source_dmo_path": anchor["source_dmo_path"],
        "source_dmo_sha256": anchor["source_dmo_sha256"],
        "source_call_order": anchor["source_order"],
        "source_caller": anchor["caller"],
        "source_caller_hex": f"0x{anchor['caller']:08x}",
    }
    options = payload["options"]
    assert isinstance(options, dict)
    options["source_bound_board_precall_global_restore"] = True
    live_sha256 = "sha256:" + "99" * 32
    precall_receipt = {
        "mechanism": "exact_source_bound_global_rng_precall_state_restore",
        "process_id": 55,
        "thread_id": 7,
        "perf_counter_ns": 190,
        "framework_update": anchor["framework_update"],
        "state_address": GLOBAL_MTRAND_ADDRESS,
        "expected_before": {
            **restored,
            "live_reference_match": False,
        },
        "live_before_index": 391,
        "live_before_state_sha256": live_sha256,
        "dynamic_before_allowed": True,
        "restored": restored,
        "changed": True,
        "bytes_written": MTRAND_STATE_BYTES,
        "transactional_rollback": True,
        "writeback_verified": True,
        "persistent_file_modified": False,
        "call_address": precall["call_address"],
        "return_address": precall["return_address"],
        "target_address": precall["target_address"],
        "instruction_hex": precall["instruction_hex"],
        "expected_main_thread_id": 7,
        "main_thread_match": True,
        "active_board_address": 0x120000,
        "active_board_vtable": BOARD_VTABLE,
        "process_memory_mutation": True,
        "classification": "source-bound-natural-retail-global-precall-restore",
        "other_threads_suspended_count": 5,
        "other_threads_suspended_until_board_call": True,
        "atomic_window_completed": True,
        "board_call_reached": True,
        "board_call_address": DEFAULT_BOARD_RESEED_CALL,
        "board_call_framework_update": anchor["framework_update"],
        "board_call_thread_id": 7,
        "other_threads_resumed_after_board_call": True,
        "atomic_window_finished_perf_counter_ns": 210,
    }
    source_receipt = payload["source_bound_board_anchor"]
    source_receipt["observations"][0][
        "global_precall_restore"
    ] = precall_receipt
    return precall


def test_source_bound_board_precall_receipt_binds_exact_natural_state(
    tmp_path: Path,
) -> None:
    payload, anchor = _source_bound_trace_payload(
        corrected=False,
        allow_correction=False,
    )
    precall = _attach_source_bound_precall_receipt(payload, anchor)
    path = tmp_path / "strict-replay.json"
    path.write_text(json.dumps(payload), encoding="ascii")

    _validate_trace_result(
        path,
        expected_pid=55,
        expected_dmo_sha256=_DMO_SHA256,
        expected_board_seed_address=DEFAULT_BOARD_RESEED_CALL,
        expected_board_seed=anchor["expected_output"],
        expected_source_bound_board_anchor=anchor,
        expected_source_bound_board_precall_global_restore=precall,
        expected_allow_source_bound_board_global_correction=False,
    )


@pytest.mark.parametrize(
    ("section", "field", "value"),
    (
        ("options", "source_bound_board_precall_global_restore", False),
        ("observation", "bytes_written", 0),
        ("restored", "state_sha256", "sha256:" + "aa" * 32),
    ),
)
def test_source_bound_board_precall_rejects_mutated_receipt(
    tmp_path: Path,
    section: str,
    field: str,
    value: object,
) -> None:
    payload, anchor = _source_bound_trace_payload(
        corrected=False,
        allow_correction=False,
    )
    precall = _attach_source_bound_precall_receipt(payload, anchor)
    observation = payload["source_bound_board_anchor"]["observations"][0][
        "global_precall_restore"
    ]
    if section == "options":
        payload["options"][field] = value
    elif section == "observation":
        observation[field] = value
    else:
        observation["restored"][field] = value
    path = tmp_path / "strict-replay.json"
    path.write_text(json.dumps(payload), encoding="ascii")

    with pytest.raises(CollectionError, match="trace_result_invalid"):
        _validate_trace_result(
            path,
            expected_pid=55,
            expected_dmo_sha256=_DMO_SHA256,
            expected_board_seed_address=DEFAULT_BOARD_RESEED_CALL,
            expected_board_seed=anchor["expected_output"],
            expected_source_bound_board_anchor=anchor,
            expected_source_bound_board_precall_global_restore=precall,
            expected_allow_source_bound_board_global_correction=False,
        )


def _diagnostic_padding_trace_result() -> dict[str, object]:
    return {
        "exited": True,
        "exit_code": 0,
        "failure_count": 0,
        "boundary_failures": [],
        "broker_failures": [],
        "command_order_offset": 3,
        "command_order_rebase_rows": [1, 5, 7],
        "diagnostic_successful_file_write_padding_rows": [7],
        "service_consumed_file_write_debt_rows": [4, 6],
        "service_consumed_file_write_surplus_rows": [6],
        "effective_deferred_file_write_debt_rows": [1, 4, 5],
        "pre_attach_file_write_debt_rows": [],
        "deferred_file_write_discharges": [
            {"debt_row_index": row_index}
            for row_index in (1, 4, 5)
        ],
        "service_continuation_verifications": [
            {
                "process_id": 55,
                "thread_id": 77,
                "framework_update": 20,
                "row_index": 2,
                "corridor_start_index": 0,
                "corridor_end_index": 1,
                "reentries": 2,
                "command_order_offset": 1,
                "service_exit_timeline_commit_count": 0,
                "bridge_recorded_update": 18,
                "bridge_following_row_index": 3,
                "bridge_following_recorded_update": 20,
                "bridge_following_native_update": 22,
                "bridge_late_updates": 2,
                "native_timeline_rebase_start_row": 3,
                "native_timeline_offset_before": 0,
                "native_timeline_offset_delta": 2,
                "native_timeline_offset_after": 2,
                "verification_mode": (
                    "successful_file_write_late_idle_bridge"
                ),
            },
            {
                "process_id": 55,
                "thread_id": 77,
                "framework_update": 101,
                "row_index": 8,
                "corridor_start_index": 4,
                "corridor_end_index": 7,
                "reentries": 4,
                "command_order_offset": 3,
                "service_exit_timeline_commit_count": 1,
                "service_consumed_file_write_debt_rows": "4,6",
                "verification_mode": (
                    "successful_file_write_deferred_exit"
                ),
            },
        ],
        "service_exit_timeline_commits": [
            {
                "process_id": 55,
                "thread_id": 77,
                "framework_update": 101,
                "row_index": 8,
                "command_bit_position": 100,
                "buffer_read_bit_position": 110,
                "last_demo_update_before": 104,
                "last_demo_update_after": 101,
                "recorded_update": 100,
                "effective_committed_update": 101,
                "timeline_delta_updates": 0,
                "recorded_timeline_delta_updates": -1,
                "late_commit_clamp_updates": 1,
                "corridor_end_row_index": 7,
                "corridor_end_recorded_update": 90,
                "bytes_written": 4,
            }
        ],
        "service_exit_timeline_suffix_rebases": [
            {
                "process_id": 55,
                "thread_id": 77,
                "framework_update": 116,
                "commit_row_index": 8,
                "suffix_start_row_index": 9,
                "timeline_offset_before": 2,
                "timeline_offset_delta": 1,
                "timeline_offset_after": 3,
                "mechanism": "one_tick_clamped_deferred_exit_suffix",
            }
        ],
        "terminal_registry_write_eof_exits": [
            {
                "process_id": 55,
                "thread_id": 77,
                "framework_update": 120,
                "start_index": 10,
                "end_index": 12,
                "terminal_row_index": 12,
                "terminal_row_start_bit_position": 200,
                "terminal_payload_bit_position": 210,
                "terminal_row_end_bit_position": 211,
                "last_command_bit_position": 199,
                "last_read_bit_position": 210,
                "last_command_order": 9,
                "command_order_offset": 3,
                "reentries": 3,
                "unconsumed_result_bits": 1,
                "recorded_success": True,
                "exit_code": 0,
                "mechanism": (
                    "normal_exit_at_terminal_registry_write_result_bit"
                ),
            }
        ],
        "successful_file_write_debt_barriers": [
            {
                "process_id": 55,
                "thread_id": 77,
                "framework_update": 120,
                "barrier_row_index": 10,
                "barrier_start_bit_position": 150,
                "barrier_read_bit_position": 160,
                "barrier_command_number": 12,
                "discharged_row_count": 3,
                "surplus_rows": [6],
                "mechanism": (
                    "later_non_file_write_prepared_service_barrier"
                ),
            }
        ],
    }


def test_trace_result_binds_provenance_padding_to_closed_debt_registry(
    tmp_path: Path,
) -> None:
    path = tmp_path / "strict-replay.json"
    payload = {
        "schema": "zuma.popcap_strict_replay.v3",
        "runtime_process_id": 55,
        "source_dmo": {
            "sha256": _DMO_SHA256.removeprefix("sha256:"),
        },
        "trace_started_perf_counter_ns": 100,
        "trace_finished_perf_counter_ns": 200,
        "options": {
            "diagnostic_successful_file_write_padding_rows": [7],
        },
        "result": _diagnostic_padding_trace_result(),
    }
    path.write_text(json.dumps(payload), encoding="ascii")

    _validate_trace_result(
        path,
        expected_pid=55,
        expected_dmo_sha256=_DMO_SHA256,
        expected_diagnostic_successful_file_write_padding_rows=[7],
    )

    payload["result"]["deferred_file_write_discharges"][-1][
        "debt_row_index"
    ] = 7
    path.write_text(json.dumps(payload), encoding="ascii")
    with pytest.raises(CollectionError, match="trace_result_invalid"):
        _validate_trace_result(
            path,
            expected_pid=55,
            expected_dmo_sha256=_DMO_SHA256,
            expected_diagnostic_successful_file_write_padding_rows=[7],
        )


def test_trace_result_binds_diagnostic_terminal_and_timeline_receipts(
    tmp_path: Path,
) -> None:
    path = tmp_path / "strict-replay.json"
    result = _diagnostic_padding_trace_result()
    result["terminal_registry_write_eof_exits"][0][
        "terminal_payload_bit_position"
    ] = 209
    payload = {
        "schema": "zuma.popcap_strict_replay.v3",
        "runtime_process_id": 55,
        "source_dmo": {
            "sha256": _DMO_SHA256.removeprefix("sha256:"),
        },
        "trace_started_perf_counter_ns": 100,
        "trace_finished_perf_counter_ns": 200,
        "options": {
            "diagnostic_successful_file_write_padding_rows": [7],
        },
        "result": result,
    }
    path.write_text(json.dumps(payload), encoding="ascii")

    with pytest.raises(CollectionError, match="trace_result_invalid"):
        _validate_trace_result(
            path,
            expected_pid=55,
            expected_dmo_sha256=_DMO_SHA256,
            expected_diagnostic_successful_file_write_padding_rows=[7],
        )


def test_trace_result_accepts_audited_diagnostic_rebase_after_blackout(
    tmp_path: Path,
) -> None:
    path = tmp_path / "strict-replay.json"
    late_result = _diagnostic_padding_trace_result()
    merged_result = json.loads(json.dumps(late_result))
    early_result = {
        "exited": False,
        "stopped_at_update": True,
        "failure_count": 0,
        "command_order_offset": 0,
        "command_order_rebase_rows": [],
        "diagnostic_successful_file_write_padding_rows": [7],
    }
    for name in (
        "service_consumed_file_write_debt_rows",
        "service_consumed_file_write_surplus_rows",
        "effective_deferred_file_write_debt_rows",
        "deferred_file_write_discharges",
        "successful_file_write_debt_barriers",
        "service_continuation_verifications",
        "service_exit_timeline_commits",
        "service_exit_timeline_suffix_rebases",
        "terminal_registry_write_eof_exits",
    ):
        early_result[name] = []
    payload = {
        "schema": "zuma.popcap_strict_replay.v3",
        "runtime_process_id": 55,
        "source_dmo": {
            "sha256": _DMO_SHA256.removeprefix("sha256:"),
        },
        "trace_started_perf_counter_ns": 100,
        "trace_finished_perf_counter_ns": 200,
        "options": {
            "attach_at_update": 0,
            "detach_at_update": 100,
            "reattach_at_update": 200,
            "allow_pre_stream_commands": True,
            "allow_blackout_file_write_order_rebase": False,
            "startup_priority_bias_until_update": 500,
            "diagnostic_successful_file_write_padding_rows": [7],
        },
        "startup_priority_phase": {
            "name": "startup_service_priority_bias",
            "mechanism": "transient_main_thread_priority_bias",
            "process_id": 55,
            "main_thread_id": 77,
            "requested_until_update": 500,
            "first_applied_update": 5,
            "restored_at_update": 506,
            "persistent_process_modification": False,
            "started_perf_counter_ns": 102,
            "restored_perf_counter_ns": 110,
            "finished_perf_counter_ns": 115,
            "threads": [
                {
                    "thread_id": 77,
                    "role": "main",
                    "original_priority": 1,
                    "biased_priority": -2,
                    "restored": True,
                    "restore_status": "restored",
                    "restored_priority": 1,
                }
            ],
        },
        "trace_phases": [
            {
                "name": "pre_capture",
                "started_perf_counter_ns": 105,
                "finished_perf_counter_ns": 120,
                "result": early_result,
            },
            {
                "name": "post_capture",
                "started_perf_counter_ns": 180,
                "finished_perf_counter_ns": 195,
                "result": late_result,
            },
        ],
        "debugger_blackout": {
            "requested_detach_update": 100,
            "detached_after_update": 105,
            "requested_reattach_update": 200,
            "reattach_observed_update": 200,
            "started_perf_counter_ns": 120,
            "finished_perf_counter_ns": 180,
            "process_id": 55,
            "read_only_wait": True,
        },
        "result": merged_result,
    }
    path.write_text(json.dumps(payload), encoding="ascii")

    _validate_trace_result(
        path,
        expected_pid=55,
        expected_dmo_sha256=_DMO_SHA256,
        expected_attach_at_update=0,
        expected_detach_at_update=100,
        expected_reattach_at_update=200,
        expected_allow_pre_stream_commands=True,
        expected_startup_priority_bias_until_update=500,
        expected_blackout_file_write_order_rebase=False,
        expected_blackout_file_write_order_rebase_rows=[],
        expected_diagnostic_successful_file_write_padding_rows=[7],
    )

    early_result["terminal_registry_write_eof_exits"] = [
        late_result["terminal_registry_write_eof_exits"][0]
    ]
    path.write_text(json.dumps(payload), encoding="ascii")
    with pytest.raises(CollectionError, match="trace_result_invalid"):
        _validate_trace_result(
            path,
            expected_pid=55,
            expected_dmo_sha256=_DMO_SHA256,
            expected_attach_at_update=0,
            expected_detach_at_update=100,
            expected_reattach_at_update=200,
            expected_allow_pre_stream_commands=True,
            expected_startup_priority_bias_until_update=500,
            expected_blackout_file_write_order_rebase=False,
            expected_blackout_file_write_order_rebase_rows=[],
            expected_diagnostic_successful_file_write_padding_rows=[7],
        )


def test_trace_result_binds_normal_close_to_exact_terminal_idle(
    tmp_path: Path,
) -> None:
    path = tmp_path / "strict-replay.json"
    terminal = {
        "command_order": 7027,
        "update": 8942,
        "kind": "idle",
        "command_number": 31,
        "short_form": False,
    }
    receipt = {
        "process_id": 55,
        "thread_id": 77,
        "mechanism": "wm_close_after_exact_terminal_command",
        "target_command_order": 7027,
        "target_update": 8942,
        "target_kind": "idle",
        "target_command_number": 31,
        "target_short_form": False,
        "target_start_bit_position": 340992,
        "target_end_bit_position": 341002,
        "observed_command_order": 7027,
        "observed_command_order_offset": 0,
        "observed_update": 8942,
        "observed_command_bit_position": 340992,
        "observed_read_bit_position": 341002,
        "main_window_handle": 1234,
        "requested_perf_counter_ns": 9999,
        "post_message_succeeded": True,
    }
    payload = {
        "schema": "zuma.popcap_strict_replay.v3",
        "runtime_process_id": 55,
        "source_dmo": {
            "sha256": _DMO_SHA256.removeprefix("sha256:"),
        },
        "trace_started_perf_counter_ns": 100,
        "trace_finished_perf_counter_ns": 200,
        "options": {"close_after_terminal_command": True},
        "result": {
            "exited": True,
            "exit_code": 0,
            "failure_count": 0,
            "boundary_failures": [],
            "broker_failures": [],
            "stopped_at_command_order": True,
            "terminal_close_requests": [receipt],
        },
    }
    path.write_text(json.dumps(payload), encoding="ascii")

    _validate_trace_result(
        path,
        expected_pid=55,
        expected_dmo_sha256=_DMO_SHA256,
        expected_close_after_terminal_command=True,
        expected_terminal_close_command=terminal,
    )

    receipt["observed_read_bit_position"] -= 1
    path.write_text(json.dumps(payload), encoding="ascii")
    with pytest.raises(CollectionError, match="trace_result_invalid"):
        _validate_trace_result(
            path,
            expected_pid=55,
            expected_dmo_sha256=_DMO_SHA256,
            expected_close_after_terminal_command=True,
            expected_terminal_close_command=terminal,
        )


def test_trace_result_binds_two_phases_and_debugger_blackout(
    tmp_path: Path,
) -> None:
    path = tmp_path / "strict-replay.json"
    clean_result = {
        "exited": True,
        "exit_code": 0,
        "failure_count": 0,
        "boundary_failures": [],
        "broker_failures": [],
    }
    payload = {
        "schema": "zuma.popcap_strict_replay.v3",
        "runtime_process_id": 55,
        "source_dmo": {
            "sha256": _DMO_SHA256.removeprefix("sha256:"),
        },
        "trace_started_perf_counter_ns": 100,
        "trace_finished_perf_counter_ns": 200,
        "options": {
            "attach_at_update": 0,
            "detach_at_update": 7300,
            "reattach_at_update": 8500,
            "allow_pre_stream_commands": True,
        },
        "trace_phases": [
            {
                "name": "pre_capture",
                "started_perf_counter_ns": 105,
                "finished_perf_counter_ns": 120,
                "result": {
                    "exited": False,
                    "stopped_at_update": True,
                    "failure_count": 0,
                },
            },
            {
                "name": "post_capture",
                "started_perf_counter_ns": 180,
                "finished_perf_counter_ns": 195,
                "result": clean_result,
            },
        ],
        "debugger_blackout": {
            "requested_detach_update": 7300,
            "detached_after_update": 7304,
            "requested_reattach_update": 8500,
            "reattach_observed_update": 8500,
            "started_perf_counter_ns": 120,
            "finished_perf_counter_ns": 180,
            "process_id": 55,
            "read_only_wait": True,
        },
        "result": clean_result,
    }
    path.write_text(json.dumps(payload), encoding="ascii")

    decoded = _validate_trace_result(
        path,
        expected_pid=55,
        expected_dmo_sha256=_DMO_SHA256,
        expected_attach_at_update=0,
        expected_detach_at_update=7300,
        expected_reattach_at_update=8500,
        expected_allow_pre_stream_commands=True,
    )
    assert decoded["debugger_blackout"]["read_only_wait"] is True

    payload["debugger_blackout"]["process_id"] = 56
    path.write_text(json.dumps(payload), encoding="ascii")
    with pytest.raises(CollectionError, match="trace_result_invalid"):
        _validate_trace_result(
            path,
            expected_pid=55,
            expected_dmo_sha256=_DMO_SHA256,
            expected_attach_at_update=0,
            expected_detach_at_update=7300,
            expected_reattach_at_update=8500,
            expected_allow_pre_stream_commands=True,
        )


def test_phased_font_cache_validation_uses_pre_capture_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "strict-replay.json"
    early_result = {
        "exited": False,
        "stopped_at_update": True,
        "failure_count": 0,
        "font_marker": "pre_capture",
        "command_order_offset": 0,
    }
    merged_result = {
        "exited": True,
        "exit_code": 0,
        "failure_count": 0,
        "boundary_failures": [],
        "broker_failures": [],
        "font_marker": "merged",
        "command_order_offset": 4,
    }
    payload = {
        "schema": "zuma.popcap_strict_replay.v3",
        "runtime_process_id": 55,
        "source_dmo": {
            "sha256": _DMO_SHA256.removeprefix("sha256:"),
        },
        "trace_started_perf_counter_ns": 100,
        "trace_finished_perf_counter_ns": 200,
        "options": {
            "attach_at_update": 0,
            "detach_at_update": 7000,
            "reattach_at_update": 9750,
            "allow_pre_stream_commands": True,
            "startup_trace_handoff": False,
        },
        "trace_phases": [
            {
                "name": "pre_capture",
                "started_perf_counter_ns": 105,
                "finished_perf_counter_ns": 120,
                "result": early_result,
            },
            {
                "name": "post_capture",
                "started_perf_counter_ns": 180,
                "finished_perf_counter_ns": 195,
                "result": merged_result,
            },
        ],
        "debugger_blackout": {
            "requested_detach_update": 7000,
            "detached_after_update": 7000,
            "requested_reattach_update": 9750,
            "reattach_observed_update": 9750,
            "started_perf_counter_ns": 120,
            "finished_perf_counter_ns": 180,
            "process_id": 55,
            "read_only_wait": True,
        },
        "result": merged_result,
    }
    path.write_text(json.dumps(payload), encoding="ascii")
    observed_markers: list[str] = []

    def validate_font_cache(
        _payload: object,
        phase_result: dict[str, object],
        **_kwargs: object,
    ) -> None:
        observed_markers.append(str(phase_result.get("font_marker")))

    monkeypatch.setattr(
        "tools.collect_pc_golden_v4._validate_font_cache_trace_receipt",
        validate_font_cache,
    )

    _validate_trace_result(
        path,
        expected_pid=55,
        expected_dmo_sha256=_DMO_SHA256,
        expected_attach_at_update=0,
        expected_detach_at_update=7000,
        expected_reattach_at_update=9750,
        expected_allow_pre_stream_commands=True,
        expected_startup_trace_handoff=False,
        expected_allow_pre_attach_file_write_debt=False,
        expected_allow_font_cache_manifest_completion_debt=False,
    )

    assert observed_markers == ["pre_capture"]


def test_trace_result_binds_overdue_idle_receipt_to_frozen_candidate(
    tmp_path: Path,
) -> None:
    path = tmp_path / "strict-replay.json"
    candidate = _overdue_idle_candidate()
    receipt = _overdue_idle_receipt()
    payload = {
        "schema": "zuma.popcap_strict_replay.v3",
        "runtime_process_id": 55,
        "source_dmo": {
            "sha256": _DMO_SHA256.removeprefix("sha256:"),
        },
        "trace_started_perf_counter_ns": 100,
        "trace_finished_perf_counter_ns": 200,
        "options": {
            "allow_post_blackout_overdue_idle_reentry": True,
        },
        "result": {
            "exited": True,
            "exit_code": 0,
            "failure_count": 0,
            "boundary_failures": [],
            "broker_failures": [],
            "service_exit_overdue_idle_reentries": [receipt],
        },
    }
    path.write_text(json.dumps(payload), encoding="ascii")

    _validate_trace_result(
        path,
        expected_pid=55,
        expected_dmo_sha256=_DMO_SHA256,
        expected_allow_post_blackout_overdue_idle_reentry=True,
        expected_post_blackout_overdue_idle_reentry_candidates=[
            candidate
        ],
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("following_kind", "mouse_position"),
        ("framework_update", 130),
        ("prepared_reentry_verified", False),
        ("prepared_framework_update", 118),
        ("memory_write_bytes", 4),
        ("process_memory_mutation", True),
    ),
)
def test_trace_result_rejects_mutated_overdue_idle_receipt(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    path = tmp_path / "strict-replay.json"
    receipt = _overdue_idle_receipt()
    receipt[field] = value
    if field == "framework_update":
        receipt["last_demo_update"] = value
    payload = {
        "schema": "zuma.popcap_strict_replay.v3",
        "runtime_process_id": 55,
        "source_dmo": {
            "sha256": _DMO_SHA256.removeprefix("sha256:"),
        },
        "trace_started_perf_counter_ns": 100,
        "trace_finished_perf_counter_ns": 200,
        "options": {
            "allow_post_blackout_overdue_idle_reentry": True,
        },
        "result": {
            "exited": True,
            "exit_code": 0,
            "failure_count": 0,
            "boundary_failures": [],
            "broker_failures": [],
            "service_exit_overdue_idle_reentries": [receipt],
        },
    }
    path.write_text(json.dumps(payload), encoding="ascii")

    with pytest.raises(CollectionError, match="trace_result_invalid"):
        _validate_trace_result(
            path,
            expected_pid=55,
            expected_dmo_sha256=_DMO_SHA256,
            expected_allow_post_blackout_overdue_idle_reentry=True,
            expected_post_blackout_overdue_idle_reentry_candidates=[
                _overdue_idle_candidate()
            ],
        )


def test_trace_result_rejects_overdue_idle_receipt_without_opt_in(
    tmp_path: Path,
) -> None:
    path = tmp_path / "strict-replay.json"
    payload = {
        "schema": "zuma.popcap_strict_replay.v3",
        "runtime_process_id": 55,
        "source_dmo": {
            "sha256": _DMO_SHA256.removeprefix("sha256:"),
        },
        "trace_started_perf_counter_ns": 100,
        "trace_finished_perf_counter_ns": 200,
        "options": {
            "allow_post_blackout_overdue_idle_reentry": False,
        },
        "result": {
            "exited": True,
            "exit_code": 0,
            "failure_count": 0,
            "boundary_failures": [],
            "broker_failures": [],
            "service_exit_overdue_idle_reentries": [
                _overdue_idle_receipt()
            ],
        },
    }
    path.write_text(json.dumps(payload), encoding="ascii")

    with pytest.raises(CollectionError, match="trace_result_invalid"):
        _validate_trace_result(
            path,
            expected_pid=55,
            expected_dmo_sha256=_DMO_SHA256,
            expected_allow_post_blackout_overdue_idle_reentry=False,
            expected_post_blackout_overdue_idle_reentry_candidates=[],
        )


def test_trace_result_binds_audited_blackout_order_rebase(
    tmp_path: Path,
) -> None:
    path = tmp_path / "strict-replay.json"
    expected_rows = [
        {
            "row_index": 3,
            "update": 7995,
            "command_number": 16,
            "kind": "file_write",
            "short_form": False,
            "success": True,
        },
        {
            "row_index": 5,
            "update": 8629,
            "command_number": 16,
            "kind": "file_write",
            "short_form": False,
            "success": True,
        },
    ]
    clean_result = {
        "exited": True,
        "exit_code": 0,
        "failure_count": 0,
        "boundary_failures": [],
        "broker_failures": [],
        "first_command_order": 9,
        "last_command_order": 20,
        "command_order_offset": 2,
        "command_order_rebase_rows": [3, 5],
        "attach_stabilization_verified": True,
        "attach_stabilization_skipped_hits": 1,
        "attach_stabilization_start_update": 8700,
        "attach_stabilization_verified_update": 8701,
        "attach_stabilization_verified_command_order": 10,
        "attach_stabilization_observations": [
            {
                "framework_update": 8700,
                "command_order": 9,
                "candidate_command_order_offset": 0,
                "accepted": False,
                "reason": "in-flight command",
            },
            {
                "framework_update": 8701,
                "command_order": 10,
                "candidate_command_order_offset": 2,
                "accepted": True,
                "reason": "exact_complete_main_boundary",
            },
        ],
    }
    payload = {
        "schema": "zuma.popcap_strict_replay.v3",
        "runtime_process_id": 55,
        "source_dmo": {
            "sha256": _DMO_SHA256.removeprefix("sha256:"),
        },
        "trace_started_perf_counter_ns": 100,
        "trace_finished_perf_counter_ns": 200,
        "options": {
            "attach_at_update": 0,
            "detach_at_update": 6808,
            "reattach_at_update": 8700,
            "allow_pre_stream_commands": True,
            "allow_blackout_file_write_order_rebase": True,
            "post_blackout_attach_stabilization": True,
        },
        "trace_phases": [
            {
                "name": "pre_capture",
                "started_perf_counter_ns": 105,
                "finished_perf_counter_ns": 120,
                "result": {
                    "exited": False,
                    "stopped_at_update": True,
                    "failure_count": 0,
                    "command_order_offset": 0,
                    "command_order_rebase_rows": [],
                },
            },
            {
                "name": "post_capture",
                "started_perf_counter_ns": 180,
                "finished_perf_counter_ns": 195,
                "result": clean_result,
            },
        ],
        "debugger_blackout": {
            "requested_detach_update": 6808,
            "detached_after_update": 6808,
            "requested_reattach_update": 8700,
            "reattach_observed_update": 8700,
            "started_perf_counter_ns": 120,
            "finished_perf_counter_ns": 180,
            "process_id": 55,
            "read_only_wait": True,
            "command_order_rebase": {
                "mechanism": (
                    "direct_long_form_file_write_result_consumption"
                ),
                "detached_after_update": 6808,
                "observed_offset": 2,
                "first_live_command_order": 10,
                "first_observed_live_command_order": 9,
                "first_offline_command_order": 12,
                "accounted_row_count": 2,
                "accounted_rows": [
                    {
                        **expected_rows[0],
                        "start_bit_position": 100,
                        "end_bit_position": 111,
                    },
                    {
                        **expected_rows[1],
                        "start_bit_position": 130,
                        "end_bit_position": 141,
                    },
                ],
                "fixed_for_entire_post_capture_phase": True,
            },
        },
        "result": clean_result,
    }
    path.write_text(json.dumps(payload), encoding="ascii")

    _validate_trace_result(
        path,
        expected_pid=55,
        expected_dmo_sha256=_DMO_SHA256,
        expected_attach_at_update=0,
        expected_detach_at_update=6808,
        expected_reattach_at_update=8700,
        expected_allow_pre_stream_commands=True,
        expected_post_blackout_attach_stabilization=True,
        expected_blackout_file_write_order_rebase=True,
        expected_blackout_file_write_order_rebase_rows=expected_rows,
    )

    payload["debugger_blackout"]["command_order_rebase"][
        "observed_offset"
    ] = 1
    path.write_text(json.dumps(payload), encoding="ascii")
    with pytest.raises(CollectionError, match="trace_result_invalid"):
        _validate_trace_result(
            path,
            expected_pid=55,
            expected_dmo_sha256=_DMO_SHA256,
            expected_attach_at_update=0,
            expected_detach_at_update=6808,
            expected_reattach_at_update=8700,
            expected_allow_pre_stream_commands=True,
            expected_post_blackout_attach_stabilization=True,
            expected_blackout_file_write_order_rebase=True,
            expected_blackout_file_write_order_rebase_rows=expected_rows,
        )


def test_trace_result_binds_preregistered_blackout_candidate_envelope(
    tmp_path: Path,
) -> None:
    path = tmp_path / "strict-replay.json"
    expected_rows = [
        {
            "row_index": 3,
            "update": 8217,
            "command_number": 16,
            "kind": "file_write",
            "short_form": False,
            "success": True,
        },
        {
            "row_index": 5,
            "update": 9603,
            "command_number": 16,
            "kind": "file_write",
            "short_form": False,
            "success": True,
        },
    ]
    late_result = {
        "exited": True,
        "exit_code": 0,
        "failure_count": 0,
        "boundary_failures": [],
        "broker_failures": [],
        "first_command_order": 9,
        "last_command_order": 20,
        "command_order_offset": 1,
        "command_order_rebase_rows": [],
        "blackout_file_write_rebase_candidate_rows": [3, 5],
        "blackout_native_timeline_offset": 2,
        "attach_stabilization_verified": True,
        "attach_stabilization_skipped_hits": 1,
        "attach_stabilization_start_update": 9750,
        "attach_stabilization_verified_update": 9752,
        "attach_stabilization_verified_command_order": 10,
        "attach_stabilization_observations": [
            {
                "framework_update": 9750,
                "command_order": 9,
                "candidate_command_order_offset": 0,
                "candidate_native_timeline_offset": 0,
                "accepted": False,
                "reason": "in-flight command",
            },
            {
                "framework_update": 9752,
                "command_order": 10,
                "candidate_command_order_offset": 1,
                "candidate_native_timeline_offset": 2,
                "accepted": True,
                "reason": "exact_complete_main_boundary",
            },
        ],
    }
    early_result = {
        "exited": False,
        "stopped_at_update": True,
        "failure_count": 0,
        "command_order_offset": 0,
        "command_order_rebase_rows": [],
        "blackout_file_write_rebase_candidate_rows": [],
        "blackout_native_timeline_offset": 0,
    }
    payload = {
        "schema": "zuma.popcap_strict_replay.v3",
        "runtime_process_id": 55,
        "source_dmo": {
            "sha256": _DMO_SHA256.removeprefix("sha256:"),
        },
        "trace_started_perf_counter_ns": 100,
        "trace_finished_perf_counter_ns": 200,
        "options": {
            "attach_at_update": 0,
            "detach_at_update": 7000,
            "reattach_at_update": 9750,
            "allow_pre_stream_commands": True,
            "allow_blackout_file_write_order_rebase": True,
            "blackout_expected_command_order_offset": 1,
            "blackout_expected_native_timeline_offset": 2,
            "post_blackout_attach_stabilization": True,
        },
        "trace_phases": [
            {
                "name": "pre_capture",
                "started_perf_counter_ns": 105,
                "finished_perf_counter_ns": 120,
                "result": early_result,
            },
            {
                "name": "post_capture",
                "started_perf_counter_ns": 180,
                "finished_perf_counter_ns": 195,
                "result": late_result,
            },
        ],
        "debugger_blackout": {
            "requested_detach_update": 7000,
            "detached_after_update": 7000,
            "requested_reattach_update": 9750,
            "reattach_observed_update": 9750,
            "started_perf_counter_ns": 120,
            "finished_perf_counter_ns": 180,
            "process_id": 55,
            "read_only_wait": True,
            "command_order_rebase": {
                "mechanism": (
                    "bounded_long_form_file_write_candidate_envelope"
                ),
                "detached_after_update": 7000,
                "observed_offset": 1,
                "observed_native_timeline_offset": 2,
                "first_live_command_order": 10,
                "first_observed_live_command_order": 9,
                "first_offline_command_order": 11,
                "candidate_row_count": 2,
                "candidate_rows": [
                    {
                        **expected_rows[0],
                        "start_bit_position": 100,
                        "end_bit_position": 111,
                    },
                    {
                        **expected_rows[1],
                        "start_bit_position": 130,
                        "end_bit_position": 141,
                    },
                ],
                "exact_missing_row_identity_available": False,
                "verification_mode": (
                    "preregistered_candidate_envelope_and_fixed_suffix"
                ),
                "fixed_for_entire_post_capture_phase": True,
            },
        },
        "result": late_result,
    }

    def validate() -> None:
        path.write_text(json.dumps(payload), encoding="ascii")
        _validate_trace_result(
            path,
            expected_pid=55,
            expected_dmo_sha256=_DMO_SHA256,
            expected_attach_at_update=0,
            expected_detach_at_update=7000,
            expected_reattach_at_update=9750,
            expected_allow_pre_stream_commands=True,
            expected_post_blackout_attach_stabilization=True,
            expected_blackout_file_write_order_rebase=True,
            expected_blackout_file_write_order_rebase_rows=expected_rows,
            expected_blackout_file_write_rebase_mode=(
                BLACKOUT_FILE_WRITE_REBASE_ENVELOPE_MODE
            ),
            expected_blackout_command_order_offset=1,
            expected_blackout_native_timeline_offset=2,
        )

    validate()

    payload["debugger_blackout"]["command_order_rebase"][
        "observed_native_timeline_offset"
    ] = 1
    with pytest.raises(CollectionError, match="trace_result_invalid"):
        validate()
    payload["debugger_blackout"]["command_order_rebase"][
        "observed_native_timeline_offset"
    ] = 2
    payload["debugger_blackout"]["command_order_rebase"][
        "candidate_rows"
    ][0]["row_index"] = 4
    with pytest.raises(CollectionError, match="trace_result_invalid"):
        validate()


def test_trace_result_binds_preregistered_blackout_offset_set_zero_member(
    tmp_path: Path,
) -> None:
    path = tmp_path / "strict-replay.json"
    expected_rows = [
        {
            "row_index": 3,
            "update": 8217,
            "command_number": 16,
            "kind": "file_write",
            "short_form": False,
            "success": True,
        },
        {
            "row_index": 5,
            "update": 9603,
            "command_number": 16,
            "kind": "file_write",
            "short_form": False,
            "success": True,
        },
    ]
    allowed_pairs = [
        {
            "command_order_offset": 0,
            "native_timeline_offset": 0,
        },
        {
            "command_order_offset": 1,
            "native_timeline_offset": 2,
        },
    ]
    late_result = {
        "exited": True,
        "exit_code": 0,
        "failure_count": 0,
        "boundary_failures": [],
        "broker_failures": [],
        "first_command_order": 9,
        "last_command_order": 20,
        "command_order_offset": 0,
        "command_order_rebase_rows": [],
        "blackout_file_write_rebase_candidate_rows": [3, 5],
        "blackout_native_timeline_offset": 0,
        "attach_stabilization_verified": True,
        "attach_stabilization_skipped_hits": 1,
        "attach_stabilization_start_update": 9750,
        "attach_stabilization_verified_update": 9752,
        "attach_stabilization_verified_command_order": 10,
        "attach_stabilization_observations": [
            {
                "framework_update": 9750,
                "command_order": 9,
                "candidate_command_order_offset": 0,
                "candidate_native_timeline_offset": 0,
                "accepted": False,
                "reason": "in-flight command",
            },
            {
                "framework_update": 9752,
                "command_order": 10,
                "candidate_command_order_offset": 0,
                "candidate_native_timeline_offset": 0,
                "accepted": True,
                "reason": "exact_complete_main_boundary",
            },
        ],
    }
    early_result = {
        "exited": False,
        "stopped_at_update": True,
        "failure_count": 0,
        "command_order_offset": 0,
        "command_order_rebase_rows": [],
        "blackout_file_write_rebase_candidate_rows": [],
        "blackout_native_timeline_offset": 0,
    }
    payload = {
        "schema": "zuma.popcap_strict_replay.v3",
        "runtime_process_id": 55,
        "source_dmo": {
            "sha256": _DMO_SHA256.removeprefix("sha256:"),
        },
        "trace_started_perf_counter_ns": 100,
        "trace_finished_perf_counter_ns": 200,
        "options": {
            "attach_at_update": 0,
            "detach_at_update": 7000,
            "reattach_at_update": 9750,
            "allow_pre_stream_commands": True,
            "allow_blackout_file_write_order_rebase": True,
            "blackout_allowed_offset_pairs": copy.deepcopy(
                allowed_pairs
            ),
            "post_blackout_attach_stabilization": True,
        },
        "trace_phases": [
            {
                "name": "pre_capture",
                "started_perf_counter_ns": 105,
                "finished_perf_counter_ns": 120,
                "result": early_result,
            },
            {
                "name": "post_capture",
                "started_perf_counter_ns": 180,
                "finished_perf_counter_ns": 195,
                "result": late_result,
            },
        ],
        "debugger_blackout": {
            "requested_detach_update": 7000,
            "detached_after_update": 7000,
            "requested_reattach_update": 9750,
            "reattach_observed_update": 9750,
            "started_perf_counter_ns": 120,
            "finished_perf_counter_ns": 180,
            "process_id": 55,
            "read_only_wait": True,
            "command_order_rebase": {
                "mechanism": (
                    "bounded_long_form_file_write_candidate_envelope_set"
                ),
                "detached_after_update": 7000,
                "observed_offset": 0,
                "observed_native_timeline_offset": 0,
                "first_live_command_order": 10,
                "first_observed_live_command_order": 9,
                "first_offline_command_order": 10,
                "candidate_row_count": 2,
                "candidate_rows": [
                    {
                        **expected_rows[0],
                        "start_bit_position": 100,
                        "end_bit_position": 111,
                    },
                    {
                        **expected_rows[1],
                        "start_bit_position": 130,
                        "end_bit_position": 141,
                    },
                ],
                "exact_missing_row_identity_available": False,
                "verification_mode": (
                    "preregistered_finite_offset_set_and_fixed_suffix"
                ),
                "fixed_for_entire_post_capture_phase": True,
                "allowed_offset_pairs": copy.deepcopy(allowed_pairs),
                "selected_offset_pair": {
                    "command_order_offset": 0,
                    "native_timeline_offset": 0,
                },
            },
        },
        "result": late_result,
    }

    def validate() -> None:
        path.write_text(json.dumps(payload), encoding="ascii")
        _validate_trace_result(
            path,
            expected_pid=55,
            expected_dmo_sha256=_DMO_SHA256,
            expected_attach_at_update=0,
            expected_detach_at_update=7000,
            expected_reattach_at_update=9750,
            expected_allow_pre_stream_commands=True,
            expected_post_blackout_attach_stabilization=True,
            expected_blackout_file_write_order_rebase=True,
            expected_blackout_file_write_order_rebase_rows=expected_rows,
            expected_blackout_file_write_rebase_mode=(
                BLACKOUT_FILE_WRITE_REBASE_ENVELOPE_SET_MODE
            ),
            expected_blackout_allowed_offset_pairs=allowed_pairs,
        )

    validate()

    rebase = payload["debugger_blackout"]["command_order_rebase"]
    rebase["allowed_offset_pairs"] = list(reversed(allowed_pairs))
    with pytest.raises(CollectionError, match="trace_result_invalid"):
        validate()
    rebase["allowed_offset_pairs"] = copy.deepcopy(allowed_pairs)

    late_result["command_order_offset"] = 2
    late_result["blackout_native_timeline_offset"] = 2
    rebase["observed_offset"] = 2
    rebase["observed_native_timeline_offset"] = 2
    rebase["first_offline_command_order"] = 12
    rebase["selected_offset_pair"] = {
        "command_order_offset": 2,
        "native_timeline_offset": 2,
    }
    late_result["attach_stabilization_observations"][-1][
        "candidate_command_order_offset"
    ] = 2
    late_result["attach_stabilization_observations"][-1][
        "candidate_native_timeline_offset"
    ] = 2
    with pytest.raises(CollectionError, match="trace_result_invalid"):
        validate()


def test_trace_result_binds_post_blackout_stabilization_without_rebase(
    tmp_path: Path,
) -> None:
    path = tmp_path / "strict-replay.json"
    late_result = {
        "exited": True,
        "exit_code": 0,
        "failure_count": 0,
        "boundary_failures": [],
        "broker_failures": [],
        "first_command_order": 100,
        "last_command_order": 110,
        "command_order_offset": 0,
        "command_order_rebase_rows": [],
        "attach_stabilization_verified": True,
        "attach_stabilization_skipped_hits": 1,
        "attach_stabilization_start_update": 7800,
        "attach_stabilization_verified_update": 7801,
        "attach_stabilization_verified_command_order": 101,
        "attach_stabilization_observations": [
            {
                "framework_update": 7800,
                "command_order": 100,
                "candidate_command_order_offset": 0,
                "accepted": False,
                "reason": "in-flight command",
            },
            {
                "framework_update": 7801,
                "command_order": 101,
                "candidate_command_order_offset": 0,
                "accepted": True,
                "reason": "exact_complete_main_boundary",
            },
        ],
    }
    payload = {
        "schema": "zuma.popcap_strict_replay.v3",
        "runtime_process_id": 55,
        "source_dmo": {
            "sha256": _DMO_SHA256.removeprefix("sha256:"),
        },
        "trace_started_perf_counter_ns": 100,
        "trace_finished_perf_counter_ns": 200,
        "options": {
            "attach_at_update": 0,
            "detach_at_update": 1800,
            "reattach_at_update": 7800,
            "allow_pre_stream_commands": True,
            "allow_attach_stabilization": False,
            "allow_blackout_file_write_order_rebase": False,
            "post_blackout_attach_stabilization": True,
        },
        "trace_phases": [
            {
                "name": "pre_capture",
                "started_perf_counter_ns": 105,
                "finished_perf_counter_ns": 120,
                "result": {
                    "exited": False,
                    "stopped_at_update": True,
                    "failure_count": 0,
                    "command_order_offset": 0,
                    "command_order_rebase_rows": [],
                },
            },
            {
                "name": "post_capture",
                "started_perf_counter_ns": 180,
                "finished_perf_counter_ns": 195,
                "result": late_result,
            },
        ],
        "debugger_blackout": {
            "requested_detach_update": 1800,
            "detached_after_update": 1805,
            "requested_reattach_update": 7800,
            "reattach_observed_update": 7800,
            "started_perf_counter_ns": 120,
            "finished_perf_counter_ns": 180,
            "process_id": 55,
            "read_only_wait": True,
        },
        "result": late_result,
    }
    path.write_text(json.dumps(payload), encoding="ascii")

    _validate_trace_result(
        path,
        expected_pid=55,
        expected_dmo_sha256=_DMO_SHA256,
        expected_attach_at_update=0,
        expected_detach_at_update=1800,
        expected_reattach_at_update=7800,
        expected_allow_pre_stream_commands=True,
        expected_allow_attach_stabilization=False,
        expected_post_blackout_attach_stabilization=True,
        expected_blackout_file_write_order_rebase=False,
        expected_blackout_file_write_order_rebase_rows=[],
    )

    late_result["attach_stabilization_observations"][-1][
        "candidate_command_order_offset"
    ] = 1
    path.write_text(json.dumps(payload), encoding="ascii")
    with pytest.raises(CollectionError, match="trace_result_invalid"):
        _validate_trace_result(
            path,
            expected_pid=55,
            expected_dmo_sha256=_DMO_SHA256,
            expected_attach_at_update=0,
            expected_detach_at_update=1800,
            expected_reattach_at_update=7800,
            expected_allow_pre_stream_commands=True,
            expected_allow_attach_stabilization=False,
            expected_post_blackout_attach_stabilization=True,
            expected_blackout_file_write_order_rebase=False,
            expected_blackout_file_write_order_rebase_rows=[],
        )


def test_trace_result_binds_fixed_seed_detach_evidence(
    tmp_path: Path,
) -> None:
    path = tmp_path / "strict-replay.json"
    payload = {
        "schema": "zuma.popcap_strict_replay.v3",
        "runtime_process_id": 55,
        "source_dmo": {
            "sha256": _DMO_SHA256.removeprefix("sha256:"),
        },
        "trace_started_perf_counter_ns": 100,
        "trace_finished_perf_counter_ns": 200,
        "options": {
            "direct_runtime": True,
            "crt_rand_seed": 123,
        },
        "startup_rng": {
            "process_id": 55,
            "seed": 123,
            "register_override": "eax_before_push_to_srand",
            "original_instruction_hex": "50",
            "compatibility_layer": "HIGHDPIAWARE",
            "persistent_file_modified": False,
            "breakpoint_observed_perf_counter_ns": 110,
            "debugger_detached_perf_counter_ns": 120,
            "detach_pending_events_drained": 3,
            "detach_attempts": 2,
        },
        "result": {
            "exited": True,
            "exit_code": 0,
            "failure_count": 0,
            "boundary_failures": [],
            "broker_failures": [],
        },
    }
    path.write_text(json.dumps(payload), encoding="ascii")

    _validate_trace_result(
        path,
        expected_pid=55,
        expected_dmo_sha256=_DMO_SHA256,
        expected_crt_rand_seed=123,
    )

    payload["startup_rng"]["detach_attempts"] = 0
    path.write_text(json.dumps(payload), encoding="ascii")
    with pytest.raises(CollectionError, match="trace_result_invalid"):
        _validate_trace_result(
            path,
            expected_pid=55,
            expected_dmo_sha256=_DMO_SHA256,
            expected_crt_rand_seed=123,
        )


def test_trace_result_binds_gapless_startup_trace_handoff(
    tmp_path: Path,
) -> None:
    path = tmp_path / "strict-replay.json"
    payload = {
        "schema": "zuma.popcap_strict_replay.v3",
        "runtime_process_id": 55,
        "source_dmo": {
            "sha256": _DMO_SHA256.removeprefix("sha256:"),
        },
        "trace_started_perf_counter_ns": 100,
        "trace_finished_perf_counter_ns": 200,
        "options": {
            "attach_at_update": 0,
            "detach_at_update": 7300,
            "reattach_at_update": 8500,
            "allow_pre_stream_commands": True,
            "startup_trace_handoff": True,
            "direct_runtime": True,
            "crt_rand_seed": 123,
            "startup_seed_transport": "debugger_register",
        },
        "startup_rng": {
            "process_id": 55,
            "thread_id": 66,
            "seed": 123,
            "register_override": "eax_before_push_to_srand",
            "original_instruction_hex": "50",
            "compatibility_layer": "HIGHDPIAWARE",
            "persistent_file_modified": False,
            "breakpoint_observed_perf_counter_ns": 110,
            "debugger_detached_perf_counter_ns": 120,
            "detach_pending_events_drained": 0,
            "detach_attempts": 1,
            "main_thread_suspended_on_detach": True,
            "main_thread_suspend_previous_count": 0,
        },
        "trace_phases": [
            {
                "name": "pre_capture",
                "started_perf_counter_ns": 125,
                "finished_perf_counter_ns": 140,
                "result": {
                    "exited": False,
                    "stopped_at_update": True,
                    "failure_count": 0,
                },
            },
            {
                "name": "post_capture",
                "started_perf_counter_ns": 180,
                "finished_perf_counter_ns": 195,
                "result": {
                    "exited": True,
                    "exit_code": 0,
                    "failure_count": 0,
                },
            },
        ],
        "debugger_blackout": {
            "requested_detach_update": 7300,
            "detached_after_update": 7300,
            "requested_reattach_update": 8500,
            "reattach_observed_update": 8500,
            "started_perf_counter_ns": 140,
            "finished_perf_counter_ns": 180,
            "process_id": 55,
            "read_only_wait": True,
        },
        "result": {
            "exited": True,
            "exit_code": 0,
            "failure_count": 0,
            "boundary_failures": [],
            "broker_failures": [],
            "startup_handoff_main_thread_id": 66,
            "startup_handoff_launcher_suspend_previous_count": 0,
            "startup_handoff_tracer_resume_previous_count": 2,
            "startup_handoff_resumed_after_breakpoints": True,
        },
    }
    path.write_text(json.dumps(payload), encoding="ascii")

    _validate_trace_result(
        path,
        expected_pid=55,
        expected_dmo_sha256=_DMO_SHA256,
        expected_attach_at_update=0,
        expected_detach_at_update=7300,
        expected_reattach_at_update=8500,
        expected_allow_pre_stream_commands=True,
        expected_startup_trace_handoff=True,
        expected_crt_rand_seed=123,
        expected_startup_seed_transport="debugger_register",
    )

    payload["options"]["startup_priority_bias_until_update"] = 500
    payload["startup_priority_phase"] = {
        "name": "startup_service_priority_bias",
        "mechanism": "transient_main_thread_priority_bias",
        "process_id": 55,
        "main_thread_id": 66,
        "requested_until_update": 500,
        "first_applied_update": 0,
        "restored_at_update": 500,
        "started_perf_counter_ns": 122,
        "restored_perf_counter_ns": 130,
        "finished_perf_counter_ns": 135,
        "persistent_process_modification": False,
        "threads": [
            {
                "thread_id": 66,
                "role": "main",
                "original_priority": 0,
                "biased_priority": -2,
                "restored": True,
                "restore_status": "restored",
                "restored_priority": 0,
            }
        ],
    }
    payload["result"]["startup_handoff_resumed_perf_counter_ns"] = 124
    path.write_text(json.dumps(payload), encoding="ascii")
    _validate_trace_result(
        path,
        expected_pid=55,
        expected_dmo_sha256=_DMO_SHA256,
        expected_attach_at_update=0,
        expected_detach_at_update=7300,
        expected_reattach_at_update=8500,
        expected_allow_pre_stream_commands=True,
        expected_startup_trace_handoff=True,
        expected_startup_priority_bias_until_update=500,
        expected_crt_rand_seed=123,
        expected_startup_seed_transport="debugger_register",
    )

    payload["result"]["startup_handoff_resumed_perf_counter_ns"] = 121
    path.write_text(json.dumps(payload), encoding="ascii")
    with pytest.raises(CollectionError, match="trace_result_invalid"):
        _validate_trace_result(
            path,
            expected_pid=55,
            expected_dmo_sha256=_DMO_SHA256,
            expected_attach_at_update=0,
            expected_detach_at_update=7300,
            expected_reattach_at_update=8500,
            expected_allow_pre_stream_commands=True,
            expected_startup_trace_handoff=True,
            expected_startup_priority_bias_until_update=500,
            expected_crt_rand_seed=123,
            expected_startup_seed_transport="debugger_register",
        )

    payload["options"].pop("startup_priority_bias_until_update")
    payload.pop("startup_priority_phase")
    payload["result"].pop("startup_handoff_resumed_perf_counter_ns")

    payload["result"][
        "startup_handoff_tracer_resume_previous_count"
    ] = 0
    path.write_text(json.dumps(payload), encoding="ascii")
    with pytest.raises(CollectionError, match="trace_result_invalid"):
        _validate_trace_result(
            path,
            expected_pid=55,
            expected_dmo_sha256=_DMO_SHA256,
            expected_attach_at_update=0,
            expected_detach_at_update=7300,
            expected_reattach_at_update=8500,
            expected_allow_pre_stream_commands=True,
            expected_startup_trace_handoff=True,
            expected_crt_rand_seed=123,
            expected_startup_seed_transport="debugger_register",
        )


def test_font_cache_receipt_accepts_gapless_startup_direct_matches() -> None:
    members = [
        {
            "member": f"cached\\fonts\\600\\font{index:02d}.txt.cfw2",
            "size": 100 + index,
        }
        for index in range(35)
    ]
    startup_rows = [
        {
            "row_index": index,
            "update": 0,
            "command_number": 16,
            "kind": "file_write",
            "short_form": False,
            "success": False,
        }
        for index in range(35)
    ]
    receipt = {
        "entry_count": 35,
        "members": members,
        "startup_file_write_rows": startup_rows,
        "pre_attach_file_write_rows": [],
        "manifest_sha256": "sha256:" + "34" * 32,
        "main_pak_sha256": "sha256:" + "56" * 32,
    }
    payload = {
        "options": {
            "startup_trace_handoff": True,
            "allow_pre_attach_file_write_debt": False,
            "allow_font_cache_manifest_completion_debt": True,
        }
    }
    result = {
        "pre_attach_file_write_debt_rows": [],
        "font_cache_manifest_entry_count": 35,
        "font_cache_manifest_sha256": "34" * 32,
        "font_cache_manifest_main_pak_sha256": "56" * 32,
        "command_order_rebase_rows": [20],
        "command_order_offset": 1,
        "terminal_file_write_payload_handoffs": [],
        "service_file_write_header_claims": [],
        "direct_font_cache_file_write_observations": [
            {
                "row_index": 0,
                "row_update": 0,
                "recorded_success": False,
                "file_write_font_cache_member": members[0]["member"],
                "file_write_argument_size": members[0]["size"],
            }
        ],
        "deferred_file_write_discharges": [
            {
                "debt_row_index": 20,
                "recorded_success": False,
                "pre_attach_debt": False,
                "file_write_font_cache_member": members[1]["member"],
                "file_write_argument_size": members[1]["size"],
            }
        ],
        "font_cache_manifest_completion_discharges": [],
    }

    _validate_font_cache_trace_receipt(
        payload,
        result,
        expected_startup_handoff=True,
        expected_allow_pre_attach_debt=False,
        expected_allow_manifest_completion=True,
        expected_manifest_receipt=receipt,
    )

    result["service_file_write_header_claims"] = [
        {
            "mechanism": "broker_adjacent_current_update_file_write",
            "process_id": 55,
            "thread_id": 66,
            "framework_update": 0,
            "row_index": 2,
            "row_start": 100,
            "row_end": 111,
            "row_update": 0,
            "recorded_success": False,
            "command_order": 1,
            "command_order_offset": 1,
            "prior_brokered_row_index": 1,
            "prior_brokered_read_bit_position": 100,
            "prior_brokered_command_order": 0,
            "payload_completion_verified": True,
            "payload_completion_thread_id": 77,
            "payload_completion_framework_update": 0,
            "payload_completion_command_bit_position": 100,
            "payload_completion_buffer_read_bit_position": 111,
            "payload_completion_command_order": 1,
            "natural_payload_consumer": True,
            "file_write_font_cache_member": members[2]["member"],
            "file_write_argument_size": members[2]["size"],
        }
    ]
    _validate_font_cache_trace_receipt(
        payload,
        result,
        expected_startup_handoff=True,
        expected_allow_pre_attach_debt=False,
        expected_allow_manifest_completion=True,
        expected_manifest_receipt=receipt,
    )
    result["service_file_write_header_claims"][0][
        "payload_completion_verified"
    ] = False
    with pytest.raises(CollectionError, match="trace_result_invalid"):
        _validate_font_cache_trace_receipt(
            payload,
            result,
            expected_startup_handoff=True,
            expected_allow_pre_attach_debt=False,
            expected_allow_manifest_completion=True,
            expected_manifest_receipt=receipt,
        )
    result["service_file_write_header_claims"] = []

    result["deferred_file_write_discharges"][0].update(
        {
            "framework_update": 423,
            "demo_loading_complete": 0,
            "pre_loading_service_continuation": True,
            "service_continuation_verified_update": 411,
        }
    )
    result["service_exit_timeline_commits"] = [
        {"framework_update": 392}
    ]
    result["service_continuation_verifications"] = [
        {
            "framework_update": 411,
            "row_index": 73,
            "corridor_end_index": 71,
            "service_exit_timeline_commit_count": 1,
        }
    ]
    _validate_font_cache_trace_receipt(
        payload,
        result,
        expected_startup_handoff=True,
        expected_allow_pre_attach_debt=False,
        expected_allow_manifest_completion=True,
        expected_manifest_receipt=receipt,
    )

    result["direct_font_cache_file_write_observations"][0][
        "row_index"
    ] = 20
    with pytest.raises(CollectionError, match="trace_result_invalid"):
        _validate_font_cache_trace_receipt(
            payload,
            result,
            expected_startup_handoff=True,
            expected_allow_pre_attach_debt=False,
            expected_allow_manifest_completion=True,
            expected_manifest_receipt=receipt,
        )


def test_trace_result_binds_fixed_seed_iat_stub_evidence(
    tmp_path: Path,
) -> None:
    path = tmp_path / "strict-replay.json"
    payload = {
        "schema": "zuma.popcap_strict_replay.v3",
        "runtime_process_id": 55,
        "source_dmo": {
            "sha256": _DMO_SHA256.removeprefix("sha256:"),
        },
        "trace_started_perf_counter_ns": 100,
        "trace_finished_perf_counter_ns": 200,
        "options": {
            "direct_runtime": True,
            "crt_rand_seed": 123,
            "startup_seed_transport": "iat_stub",
        },
        "startup_rng": {
            "process_id": 55,
            "seed": 123,
            "seed_transport": "temporary_get_tick_count_iat_stub",
            "seed_call_return_address_hex": "0x0068f577",
            "iat_address_hex": "0x0094b1d4",
            "entry_point_address_hex": "0x008d1883",
            "entry_original_hex": "e82e",
            "stub_machine_code": (
                "caller_return_address_filter_then_forward"
            ),
            "stub_machine_code_hex": "90" * 12,
            "loader_gate_perf_counter_ns": 105,
            "iat_patched_perf_counter_ns": 110,
            "restoration_guard_perf_counter_ns": 120,
            "iat_restored_perf_counter_ns": 130,
            "restored_at_framework_update": 1,
            "iat_restored": True,
            "compatibility_layer": "HIGHDPIAWARE",
            "persistent_file_modified": False,
        },
        "result": {
            "exited": True,
            "exit_code": 0,
            "failure_count": 0,
            "boundary_failures": [],
            "broker_failures": [],
        },
    }
    path.write_text(json.dumps(payload), encoding="ascii")

    _validate_trace_result(
        path,
        expected_pid=55,
        expected_dmo_sha256=_DMO_SHA256,
        expected_crt_rand_seed=123,
        expected_startup_seed_transport="iat_stub",
    )

    payload["startup_rng"]["iat_restored"] = False
    path.write_text(json.dumps(payload), encoding="ascii")
    with pytest.raises(CollectionError, match="trace_result_invalid"):
        _validate_trace_result(
            path,
            expected_pid=55,
            expected_dmo_sha256=_DMO_SHA256,
            expected_crt_rand_seed=123,
            expected_startup_seed_transport="iat_stub",
        )


def test_trace_result_binds_transient_startup_priority_bias(
    tmp_path: Path,
) -> None:
    path = tmp_path / "strict-replay.json"
    payload = {
        "schema": "zuma.popcap_strict_replay.v3",
        "runtime_process_id": 55,
        "source_dmo": {
            "sha256": _DMO_SHA256.removeprefix("sha256:"),
        },
        "trace_started_perf_counter_ns": 100,
        "trace_finished_perf_counter_ns": 200,
        "options": {
            "startup_priority_bias_until_update": 400,
        },
        "startup_priority_phase": {
            "name": "startup_service_priority_bias",
            "mechanism": "transient_main_thread_priority_bias",
            "process_id": 55,
            "main_thread_id": 66,
            "requested_until_update": 400,
            "first_applied_update": 0,
            "restored_at_update": 400,
            "started_perf_counter_ns": 120,
            "restored_perf_counter_ns": 130,
            "finished_perf_counter_ns": 140,
            "persistent_process_modification": False,
            "threads": [
                {
                    "thread_id": 66,
                    "role": "main",
                    "original_priority": 0,
                    "biased_priority": -2,
                    "restored": True,
                    "restore_status": "restored",
                    "restored_priority": 0,
                },
            ],
        },
        "result": {
            "exited": True,
            "exit_code": 0,
            "failure_count": 0,
            "boundary_failures": [],
            "broker_failures": [],
        },
    }
    path.write_text(json.dumps(payload), encoding="ascii")

    _validate_trace_result(
        path,
        expected_pid=55,
        expected_dmo_sha256=_DMO_SHA256,
        expected_startup_priority_bias_until_update=400,
    )

    payload["startup_priority_phase"]["threads"][0]["restored"] = False
    path.write_text(json.dumps(payload), encoding="ascii")
    with pytest.raises(CollectionError, match="trace_result_invalid"):
        _validate_trace_result(
            path,
            expected_pid=55,
            expected_dmo_sha256=_DMO_SHA256,
            expected_startup_priority_bias_until_update=400,
        )


def test_trace_result_binds_preinstruction_process_affinity(
    tmp_path: Path,
) -> None:
    path = tmp_path / "strict-replay.json"
    payload = {
        "schema": "zuma.popcap_strict_replay.v3",
        "runtime_process_id": 55,
        "source_dmo": {
            "sha256": _DMO_SHA256.removeprefix("sha256:"),
        },
        "trace_started_perf_counter_ns": 100,
        "trace_finished_perf_counter_ns": 300,
        "options": {
            "direct_runtime": True,
            "startup_trace_handoff": True,
            "startup_process_affinity_mask": 1,
        },
        "startup_rng": {
            "process_id": 55,
            "debugger_detached_perf_counter_ns": 150,
            "process_affinity": {
                "application_stage": (
                    "created_suspended_before_first_resume"
                ),
                "previous_process_mask": 0xFFFF,
                "requested_process_mask": 1,
                "observed_process_mask": 1,
                "system_mask": 0xFFFF,
                "single_logical_processor": True,
                "persistent_host_modification": False,
                "handoff_observed_process_mask": 1,
                "handoff_system_mask": 0xFFFF,
                "handoff_verification_stage": (
                    "after_debugger_detach_before_tracer_handoff"
                ),
                "handoff_verification_perf_counter_ns": 160,
                "handoff_verified": True,
            },
        },
        "result": {
            "exited": True,
            "exit_code": 0,
            "failure_count": 0,
            "boundary_failures": [],
            "broker_failures": [],
        },
    }
    path.write_text(json.dumps(payload), encoding="ascii")

    _validate_trace_result(
        path,
        expected_pid=55,
        expected_dmo_sha256=_DMO_SHA256,
        expected_startup_process_affinity_mask=1,
    )

    payload["startup_rng"]["process_affinity"][
        "handoff_observed_process_mask"
    ] = 2
    path.write_text(json.dumps(payload), encoding="ascii")
    with pytest.raises(CollectionError, match="trace_result_invalid"):
        _validate_trace_result(
            path,
            expected_pid=55,
            expected_dmo_sha256=_DMO_SHA256,
            expected_startup_process_affinity_mask=1,
        )


def test_trace_result_binds_board_seed_evidence(
    tmp_path: Path,
) -> None:
    path = tmp_path / "strict-replay.json"
    payload = {
        "schema": "zuma.popcap_strict_replay.v3",
        "runtime_process_id": 55,
        "source_dmo": {
            "sha256": _DMO_SHA256.removeprefix("sha256:"),
        },
        "trace_started_perf_counter_ns": 100,
        "trace_finished_perf_counter_ns": 200,
        "options": {
            "board_seed_address": 0x0065B828,
            "board_seed": 123,
            "global_rng_seed": 456,
            "thread_crt_rng_seed": 789,
        },
        "board_rng": {
            "call_site_address": 0x0065B828,
            "effective_seed": 123,
            "global_rng_seed": 456,
            "thread_crt_rng_seed": 789,
            "original_instruction_hex": "e8",
            "register_override": "ecx_seed_before_mtrand_srand_call",
            "persistent_file_modified": False,
            "observations": [
                {
                    "process_id": 55,
                    "thread_id": 66,
                    "perf_counter_ns": 150,
                    "framework_update": 561,
                    "call_site_address": 0x0065B828,
                    "rng_object_address": 0x12340000,
                    "observed_seed": 987,
                    "effective_seed": 123,
                    "overridden": True,
                    "global_rng_address": GLOBAL_MTRAND_ADDRESS,
                    "global_rng_seed": 456,
                    "global_rng_state_bytes": MTRAND_STATE_BYTES,
                    "global_rng_state_sha256_before": (
                        "sha256:" + "ab" * 32
                    ),
                    "global_rng_state_sha256_after": (
                        _mtrand_state_sha256(456)
                    ),
                    "global_rng_overridden": True,
                    "fs_selector": 0x53,
                    "fs_segment_base": 0x00380000,
                    "teb_address": 0x00380000,
                    "fls_index_address": CRT_FLS_INDEX_ADDRESS,
                    "fls_index": 6,
                    "fls_data_address": 0x010DD000,
                    "fls_block_index": 4,
                    "fls_entry_index": 6,
                    "fls_block_slot_address": 0x010DD008,
                    "fls_block_address": 0x010DDAF0,
                    "fls_value_address": 0x010DDB0C,
                    "ptd_address": 0x031F1234,
                    "ptd_thread_id": 66,
                    "rand_state_address": (
                        0x031F1234 + CRT_PTD_RAND_STATE_OFFSET
                    ),
                    "thread_crt_rng_state_before": 654321,
                    "thread_crt_rng_state_after": 789,
                    "thread_crt_rng_seed": 789,
                    "thread_crt_rng_overridden": True,
                }
            ],
        },
        "result": {
            "exited": True,
            "exit_code": 0,
            "failure_count": 0,
            "boundary_failures": [],
            "broker_failures": [],
        },
    }
    path.write_text(json.dumps(payload), encoding="ascii")

    _validate_trace_result(
        path,
        expected_pid=55,
        expected_dmo_sha256=_DMO_SHA256,
        expected_board_seed_address=0x0065B828,
        expected_board_seed=123,
        expected_global_rng_seed=456,
        expected_thread_crt_rng_seed=789,
    )

    payload["board_rng"]["observations"][0]["effective_seed"] = 124
    path.write_text(json.dumps(payload), encoding="ascii")
    with pytest.raises(CollectionError, match="trace_result_invalid"):
        _validate_trace_result(
            path,
            expected_pid=55,
            expected_dmo_sha256=_DMO_SHA256,
            expected_board_seed_address=0x0065B828,
            expected_board_seed=123,
            expected_global_rng_seed=456,
            expected_thread_crt_rng_seed=789,
        )

    payload["board_rng"]["observations"][0]["effective_seed"] = 123
    payload["board_rng"]["observations"][0]["ptd_thread_id"] = 67
    path.write_text(json.dumps(payload), encoding="ascii")
    with pytest.raises(CollectionError, match="trace_result_invalid"):
        _validate_trace_result(
            path,
            expected_pid=55,
            expected_dmo_sha256=_DMO_SHA256,
            expected_board_seed_address=0x0065B828,
            expected_board_seed=123,
            expected_global_rng_seed=456,
            expected_thread_crt_rng_seed=789,
        )


def test_window_repaint_handshake_restores_exact_geometry() -> None:
    target = WindowTarget(
        process_id=55,
        window_handle=0x1234,
        client_region=(10, 20, 810, 620),
    )
    rect = [2, 3, 818, 642]
    moves: list[tuple[int, int]] = []
    restorations: list[tuple[int, int, int]] = []
    ticks = iter((100, 200, 250, 300, 400))

    def drag_once(supplied: WindowTarget, delta_x: int) -> None:
        assert supplied == target
        moves.append((supplied.window_handle, delta_x))
        rect[0] += delta_x
        rect[2] += delta_x

    def restore_origin(handle: int, left: int, top: int) -> None:
        restorations.append((handle, left, top))
        width = rect[2] - rect[0]
        height = rect[3] - rect[1]
        rect[:] = [left, top, left + width, top + height]

    evidence = _window_repaint_handshake(
        target,
        get_outer_rect=lambda unused: tuple(rect),
        drag_titlebar_once=drag_once,
        restore_window_origin=restore_origin,
        monotonic_ns=lambda: next(ticks),
        delay=lambda unused: None,
    )

    assert moves == [(0x1234, 32)]
    assert restorations == [(0x1234, 2, 3)]
    assert evidence["geometry_restored"] is True
    assert (
        evidence["mechanism"]
        == "nonclient_titlebar_drag_then_exact_origin_restore"
    )
    assert evidence["restoration_mechanism"] == "set_window_pos_exact_origin"
    assert evidence["outer_rect_before"] == [2, 3, 818, 642]
    assert evidence["requested_temporary_delta"] == [32, 0]
    assert evidence["actual_temporary_delta"] == [32, 0]
    assert evidence["outer_rect_after"] == [2, 3, 818, 642]


def test_window_repaint_handshake_rejects_no_actual_drag() -> None:
    target = WindowTarget(
        process_id=55,
        window_handle=0x1234,
        client_region=(10, 20, 810, 620),
    )
    ticks = iter((100, 200, 250, 300, 400))

    with pytest.raises(
        CollectionError,
        match="window_repaint_drag_no_movement",
    ):
        _window_repaint_handshake(
            target,
            get_outer_rect=lambda unused: (2, 3, 818, 642),
            drag_titlebar_once=lambda unused, delta_x: None,
            restore_window_origin=lambda unused, left, top: None,
            monotonic_ns=lambda: next(ticks),
            delay=lambda unused: None,
        )


def test_window_position_repaint_handshake_restores_without_input() -> None:
    target = WindowTarget(
        process_id=55,
        window_handle=0x1234,
        client_region=(10, 20, 810, 620),
    )
    rect = [2, 3, 818, 642]
    moves: list[tuple[int, int, int]] = []
    flushes: list[bool] = []
    ticks = iter((100, 200, 300, 400, 500))

    def set_origin(handle: int, left: int, top: int) -> None:
        moves.append((handle, left, top))
        width = rect[2] - rect[0]
        height = rect[3] - rect[1]
        rect[:] = [left, top, left + width, top + height]

    evidence = _window_position_repaint_handshake(
        target,
        get_outer_rect=lambda unused: tuple(rect),
        get_virtual_screen_rect=lambda: (0, 0, 1920, 1080),
        set_window_origin=set_origin,
        flush_dwm=lambda: flushes.append(True),
        monotonic_ns=lambda: next(ticks),
        delay=lambda unused: None,
    )

    assert moves == [(0x1234, 34, 3), (0x1234, 2, 3)]
    assert flushes == [True, True]
    assert evidence["mechanism"] == (
        "set_window_pos_temporary_translation_then_exact_origin_restore"
    )
    assert evidence["input_transport"] == "none_window_manager_api_only"
    assert evidence["synthetic_input_event_count"] == 0
    assert evidence["outer_rect_moved"] == [34, 3, 850, 642]
    assert evidence["outer_rect_after"] == [2, 3, 818, 642]
    assert evidence["geometry_restored"] is True


def test_window_position_repaint_handshake_uses_leftward_safe_move() -> None:
    target = WindowTarget(
        process_id=55,
        window_handle=0x1234,
        client_region=(1100, 20, 1900, 620),
    )
    rect = [1092, 3, 1908, 642]

    def set_origin(_handle: int, left: int, top: int) -> None:
        width = rect[2] - rect[0]
        height = rect[3] - rect[1]
        rect[:] = [left, top, left + width, top + height]

    evidence = _window_position_repaint_handshake(
        target,
        get_outer_rect=lambda unused: tuple(rect),
        get_virtual_screen_rect=lambda: (0, 0, 1920, 1080),
        set_window_origin=set_origin,
        flush_dwm=lambda: None,
        monotonic_ns=iter((100, 200, 300, 400, 500)).__next__,
        delay=lambda unused: None,
    )

    assert evidence["requested_temporary_delta"] == [-32, 0]
    assert evidence["geometry_restored"] is True


def test_startup_window_activation_is_verified_and_evidenced() -> None:
    target = WindowTarget(
        process_id=55,
        window_handle=0x1234,
        client_region=(10, 20, 810, 620),
    )
    activated: list[int] = []
    verified: list[tuple[WindowTarget, bool]] = []
    ticks = iter((100, 200))

    def verify(
        supplied: WindowTarget,
        *,
        require_foreground: bool = True,
    ) -> None:
        verified.append((supplied, require_foreground))

    evidence = _activate_startup_window(
        target,
        activate_window=activated.append,
        verify_target=verify,
        monotonic_ns=lambda: next(ticks),
    )

    assert activated == [target.window_handle]
    assert verified == [(target, True)]
    assert evidence["window_target"] == {
        "process_id": 55,
        "window_handle_hex": "0x0000000000001234",
        "client_region": [10, 20, 810, 620],
    }
    assert evidence["activation_started_perf_counter_ns"] == 100
    assert evidence["activation_finished_perf_counter_ns"] == 200
    assert evidence["foreground_activation_verified"] is True
    assert evidence["activation_attempts"] == 1


def test_startup_window_activation_retries_only_foreground_failure() -> None:
    target = WindowTarget(
        process_id=55,
        window_handle=0x1234,
        client_region=(10, 20, 810, 620),
    )
    activated: list[int] = []
    checks = 0

    def verify(
        supplied: WindowTarget,
        *,
        require_foreground: bool = True,
    ) -> None:
        nonlocal checks
        assert supplied == target
        assert require_foreground is True
        checks += 1
        if checks == 1:
            raise CaptureError("target_window_not_foreground")

    evidence = _activate_startup_window(
        target,
        activate_window=activated.append,
        verify_target=verify,
    )

    assert activated == [target.window_handle, target.window_handle]
    assert evidence["activation_attempts"] == 2


def test_startup_window_activation_fails_closed() -> None:
    target = WindowTarget(
        process_id=55,
        window_handle=0x1234,
        client_region=(10, 20, 810, 620),
    )

    def reject(
        unused: WindowTarget,
        *,
        require_foreground: bool = True,
    ) -> None:
        assert require_foreground is True
        raise CaptureError("target_not_foreground")

    with pytest.raises(
        CollectionError,
        match="startup_activation_target_not_foreground",
    ):
        _activate_startup_window(
            target,
            activate_window=lambda unused: None,
            verify_target=reject,
        )


def test_repaint_guard_rebinds_only_same_process_window() -> None:
    initial = WindowTarget(
        process_id=55,
        window_handle=0x1234,
        client_region=(10, 20, 810, 620),
    )
    replacement = WindowTarget(
        process_id=55,
        window_handle=0x5678,
        client_region=(40, 20, 840, 620),
    )
    samples = iter((7606, 7607, 7608))
    monotonic_values = iter((0.0, 0.1, 0.2, 0.3))
    clock_values = iter((100, 200, 300))
    verified: list[tuple[WindowTarget, bool]] = []
    activated: list[int] = []

    class FakeTrace:
        @staticmethod
        def poll() -> None:
            return None

    class FakeReader:
        def __init__(self, process_id: int) -> None:
            assert process_id == 55

        def __enter__(self) -> FakeReader:
            return self

        def __exit__(self, *unused: object) -> None:
            return None

        @staticmethod
        def sample() -> int:
            return next(samples)

    def verify(target: WindowTarget, *, require_foreground: bool = True) -> None:
        verified.append((target, require_foreground))
        if target == initial:
            raise CaptureError("target_window_changed")

    def refresh(process_id: int, timeout: float) -> WindowTarget:
        assert process_id == 55
        assert timeout == 5.0
        return replacement

    evidence = _wait_for_repaint_guard(
        target=initial,
        trace_process=FakeTrace(),  # type: ignore[arg-type]
        trigger_update=7607,
        previous_effectful_command_update=7156,
        next_effectful_command_update=7738,
        guarded_idle_command_count=581,
        update_reader_factory=FakeReader,
        verify_target=verify,
        refresh_target=refresh,
        activate_window=activated.append,
        repaint_handshake=lambda supplied: {
            "process_id": supplied.process_id,
            "window_handle_hex": f"0x{supplied.window_handle:016x}",
        },
        monotonic=lambda: next(monotonic_values),
        monotonic_ns=lambda: next(clock_values),
        delay=lambda unused: None,
    )

    assert activated == [replacement.window_handle]
    assert evidence["wait_timeout_seconds"] == 120.0
    assert evidence["window_rebind_count"] == 1
    assert evidence["capture_window_target"]["window_handle_hex"] == (
        "0x0000000000005678"
    )
    assert evidence["window_rebindings"] == [
        {
            "observed_framework_update": None,
            "observed_perf_counter_ns": 100,
            "previous": {
                "process_id": 55,
                "window_handle_hex": "0x0000000000001234",
                "client_region": [10, 20, 810, 620],
            },
            "replacement": {
                "process_id": 55,
                "window_handle_hex": "0x0000000000005678",
                "client_region": [40, 20, 840, 620],
            },
            "same_process_verified": True,
            "expected_client_size_verified": True,
        }
    ]
    assert (replacement, True) in verified


def test_repaint_guard_rejects_cross_process_rebind() -> None:
    initial = WindowTarget(
        process_id=55,
        window_handle=0x1234,
        client_region=(10, 20, 810, 620),
    )
    wrong_process = WindowTarget(
        process_id=56,
        window_handle=0x5678,
        client_region=(10, 20, 810, 620),
    )

    class FakeTrace:
        @staticmethod
        def poll() -> None:
            return None

    class FakeReader:
        def __init__(self, process_id: int) -> None:
            assert process_id == 55

        def __enter__(self) -> FakeReader:
            return self

        def __exit__(self, *unused: object) -> None:
            return None

    with pytest.raises(
        CollectionError,
        match="runtime_process_identity_changed",
    ):
        _wait_for_repaint_guard(
            target=initial,
            trace_process=FakeTrace(),  # type: ignore[arg-type]
            trigger_update=7607,
            previous_effectful_command_update=7156,
            next_effectful_command_update=7738,
            guarded_idle_command_count=581,
            update_reader_factory=FakeReader,
            verify_target=lambda unused, require_foreground=False: (
                (_ for _ in ()).throw(
                    CaptureError("target_window_changed")
                )
            ),
            refresh_target=lambda process_id, timeout: wrong_process,
            monotonic=lambda: 0.0,
        )


def test_repaint_guard_reports_trace_exit_before_window_rebind() -> None:
    target = WindowTarget(
        process_id=55,
        window_handle=0x1234,
        client_region=(10, 20, 810, 620),
    )
    polls = iter((None, 1))

    class FakeTrace:
        @staticmethod
        def poll() -> int | None:
            return next(polls)

    class FakeReader:
        def __init__(self, process_id: int) -> None:
            assert process_id == 55

        def __enter__(self) -> FakeReader:
            return self

        def __exit__(self, *unused: object) -> None:
            return None

    with pytest.raises(CollectionError, match="strict_replay_failed"):
        _wait_for_repaint_guard(
            target=target,
            trace_process=FakeTrace(),  # type: ignore[arg-type]
            trigger_update=7607,
            previous_effectful_command_update=7156,
            next_effectful_command_update=7738,
            guarded_idle_command_count=581,
            update_reader_factory=FakeReader,
            verify_target=lambda unused, require_foreground=False: (
                (_ for _ in ()).throw(
                    CaptureError("target_window_changed")
                )
            ),
            monotonic=iter((0.0, 0.1)).__next__,
        )


def test_run_timeline_allows_launcher_before_process_observation() -> None:
    _validate_run_timeline(
        start_snapshot_ns=100,
        observed_process_start_ns=200,
        capture_start_ns=300,
        capture_end_ns=400,
        trace_start_ns=150,
        trace_stop_ns=600,
    )


def test_run_timeline_rejects_trace_after_process_observation() -> None:
    with pytest.raises(
        CollectionError,
        match="trace_process_lifetime_mismatch",
    ):
        _validate_run_timeline(
            start_snapshot_ns=100,
            observed_process_start_ns=200,
            capture_start_ns=300,
            capture_end_ns=400,
            trace_start_ns=250,
            trace_stop_ns=600,
        )
