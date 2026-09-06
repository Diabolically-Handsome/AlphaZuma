from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from tools.record_autoplay_zuma import (
    BOARD_GLOBAL_PRECALL_ADDRESS,
    BOARD_GLOBAL_PRECALL_AND_RESET_KIND,
    BOARD_GLOBAL_PRECALL_AND_NATURAL_LOSS_TARGET_WRITE_KIND,
    BOARD_GLOBAL_PRECALL_AND_TARGET_WRITE_KIND,
    BOARD_GLOBAL_PRECALL_KIND,
    BOARD_GLOBAL_PRECALL_RETURN,
    DEFAULT_ADVENTURE_UPDATE,
    DEFAULT_ADVENTURE_POINT,
    DEFAULT_CONTINUE_GAME_UPDATE,
    DEFAULT_CONTINUE_GAME_POINT,
    DEFAULT_TITLE_START_UPDATE,
    DEFAULT_TITLE_START_POINT,
    EXPECTED_RETAIL_RUNTIME_SHA256,
    NATURAL_LOSS_TARGET_WRITE_RECEIPT_KIND,
    _parser,
    _promote_natural_loss_target_write_gameplay,
    _write_board_anchor_monitor,
    _validate_prelaunch_cursor_screen_point,
    _validate_navigation_point,
    _validate_navigation_updates,
)
from tools.trace_popcap_gameplay_mtrand import (
    BOARD_RESET_DIRECT_RETURN_ADDRESSES,
    BOARD_RESET_ENTRY_INSTRUCTION,
    BOARD_RESET_NATIVE_CLOCK_ZERO_INSTRUCTION,
    BOARD_RESET_RETURN_INSTRUCTION,
    NATURAL_LOSS_TARGET_CLEAR_POST_EIP,
    NATURAL_LOSS_TARGET_CLEAR_WRITER_ADDRESS,
    NATURAL_LOSS_TARGET_POSITIVE_POST_EIP,
    NATURAL_LOSS_TARGET_POSITIVE_WRITER_ADDRESS,
    NATURAL_LOSS_TARGET_WRITE_MAX_INTERVAL_NS,
    DEFAULT_BOARD_GLOBAL_PRECALL,
    DEFAULT_BOARD_RESET_ENTRY,
    DEFAULT_GAMEPLAY_RNG_RETURN,
    DEFAULT_GLOBAL_RNG_WRAPPER,
    _default_breakpoint_address,
    _is_board_reset_entry_candidate,
    _is_board_target_write_completion,
    _is_natural_loss_source_anchor_candidate,
    _is_natural_loss_target_clear_event,
    _is_natural_loss_target_write_pair,
    _is_source_board_anchor_candidate,
    _restore_hardware_breakpoint_context,
    _retarget_hardware_breakpoint_context,
)


def test_navigation_defaults_preserve_existing_schedule() -> None:
    args = _parser().parse_args(
        [
            "--output-root",
            "out",
            "--runtime-executable",
            "game.exe",
            "--changedir",
            "assets",
            "--prestate-dir",
            "prestate",
        ]
    )
    assert args.title_start_update == DEFAULT_TITLE_START_UPDATE == 950
    assert args.adventure_update == DEFAULT_ADVENTURE_UPDATE == 1280
    assert args.continue_game_update == DEFAULT_CONTINUE_GAME_UPDATE == 1420
    assert args.title_start_point == DEFAULT_TITLE_START_POINT
    assert args.adventure_point == DEFAULT_ADVENTURE_POINT
    assert args.continue_game_point == DEFAULT_CONTINUE_GAME_POINT


def test_recording_cli_accepts_bomb_fruit_policy() -> None:
    args = _parser().parse_args(
        [
            "--output-root",
            "out",
            "--runtime-executable",
            "game.exe",
            "--changedir",
            "assets",
            "--prestate-dir",
            "prestate",
            "--fruit-policy",
            "bomb",
        ]
    )
    assert args.fruit_policy == "bomb"


def test_board_global_precall_cli_and_default_address_are_frozen() -> None:
    args = _parser().parse_args(
        [
            "--output-root",
            "out",
            "--runtime-executable",
            "game.exe",
            "--changedir",
            "assets",
            "--prestate-dir",
            "prestate",
            "--gameplay-mtrand-trace",
            "--mtrand-trace-kind",
            BOARD_GLOBAL_PRECALL_KIND,
        ]
    )
    assert args.mtrand_trace_kind == BOARD_GLOBAL_PRECALL_KIND
    assert (
        _default_breakpoint_address(BOARD_GLOBAL_PRECALL_KIND)
        == DEFAULT_BOARD_GLOBAL_PRECALL
        == BOARD_GLOBAL_PRECALL_ADDRESS
    )
    assert (
        _default_breakpoint_address("global_wrapper_entry")
        == DEFAULT_GLOBAL_RNG_WRAPPER
    )
    assert (
        _default_breakpoint_address("gameplay_return")
        == DEFAULT_GAMEPLAY_RNG_RETURN
    )


def test_combined_reset_trace_cli_preserves_source_boundary() -> None:
    args = _parser().parse_args(
        [
            "--output-root",
            "out",
            "--runtime-executable",
            "game.exe",
            "--changedir",
            "assets",
            "--prestate-dir",
            "prestate",
            "--gameplay-mtrand-trace",
            "--mtrand-trace-kind",
            BOARD_GLOBAL_PRECALL_AND_RESET_KIND,
        ]
    )
    assert args.mtrand_trace_kind == BOARD_GLOBAL_PRECALL_AND_RESET_KIND
    assert (
        _default_breakpoint_address(BOARD_GLOBAL_PRECALL_AND_RESET_KIND)
        == DEFAULT_BOARD_GLOBAL_PRECALL
    )


def test_combined_target_write_cli_preserves_source_boundary() -> None:
    args = _parser().parse_args(
        [
            "--output-root",
            "out",
            "--runtime-executable",
            "game.exe",
            "--changedir",
            "assets",
            "--prestate-dir",
            "prestate",
            "--gameplay-mtrand-trace",
            "--mtrand-trace-kind",
            BOARD_GLOBAL_PRECALL_AND_TARGET_WRITE_KIND,
        ]
    )
    assert (
        args.mtrand_trace_kind
        == BOARD_GLOBAL_PRECALL_AND_TARGET_WRITE_KIND
    )
    assert (
        _default_breakpoint_address(
            BOARD_GLOBAL_PRECALL_AND_TARGET_WRITE_KIND
        )
        == DEFAULT_BOARD_GLOBAL_PRECALL
    )


def test_natural_loss_target_write_cli_is_an_explicit_source_mode() -> None:
    args = _parser().parse_args(
        [
            "--output-root",
            "out",
            "--runtime-executable",
            "game.exe",
            "--changedir",
            "assets",
            "--prestate-dir",
            "prestate",
            "--gameplay-mtrand-trace",
            "--mtrand-trace-kind",
            BOARD_GLOBAL_PRECALL_AND_NATURAL_LOSS_TARGET_WRITE_KIND,
        ]
    )
    assert args.mtrand_trace_kind == (
        BOARD_GLOBAL_PRECALL_AND_NATURAL_LOSS_TARGET_WRITE_KIND
    )
    assert (
        _default_breakpoint_address(args.mtrand_trace_kind)
        == DEFAULT_BOARD_GLOBAL_PRECALL
    )


def test_board_reset_static_identity_is_frozen() -> None:
    assert DEFAULT_BOARD_RESET_ENTRY == 0x00419A90
    assert len(BOARD_RESET_ENTRY_INSTRUCTION) == 32
    assert hashlib.sha256(BOARD_RESET_ENTRY_INSTRUCTION).hexdigest() == (
        "5a8da3a3f64f779b8260c3dffd24affb179daa0f925b6808e9bce44394ea37a4"
    )
    assert BOARD_RESET_NATIVE_CLOCK_ZERO_INSTRUCTION.hex() == (
        "c786c80e000000000000"
    )
    assert BOARD_RESET_RETURN_INSTRUCTION.hex() == "c20800"


def test_board_reset_entry_requires_anchored_board_and_static_caller() -> None:
    board = {"board_address": 0x25981230}
    return_address = min(BOARD_RESET_DIRECT_RETURN_ADDRESSES)
    assert _is_board_reset_entry_candidate(
        board=board,
        esi=0x25981230,
        return_address=return_address,
        source_board_address=0x25981230,
    )
    assert not _is_board_reset_entry_candidate(
        board=board,
        esi=0x25981234,
        return_address=return_address,
        source_board_address=0x25981230,
    )
    assert not _is_board_reset_entry_candidate(
        board=board,
        esi=0x25981230,
        return_address=0x12345678,
        source_board_address=0x25981230,
    )


def test_board_target_write_completion_requires_positive_changed_target() -> None:
    board = {"board_address": 0x25981230, "score_target": 9616}
    assert _is_board_target_write_completion(
        board=board,
        source_board_address=0x25981230,
        source_score_target=9650,
    )
    assert not _is_board_target_write_completion(
        board=board | {"score_target": 0},
        source_board_address=0x25981230,
        source_score_target=9650,
    )
    assert not _is_board_target_write_completion(
        board=board | {"score_target": 9650},
        source_board_address=0x25981230,
        source_score_target=9650,
    )
    assert not _is_board_target_write_completion(
        board=board | {"board_address": 0x25981234},
        source_board_address=0x25981230,
        source_score_target=9650,
    )


def _natural_loss_target_write_fixture() -> tuple[dict, dict]:
    board = 0x257AA220
    source = {
        "event_kind": "board_global_precall",
        "order": 0,
        "thread_id": 38472,
        "caller": BOARD_GLOBAL_PRECALL_RETURN,
        "call_address": BOARD_GLOBAL_PRECALL_ADDRESS,
        "call_instruction_hex": "e86fbcfbff",
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
        "thread_crt_state": {"thread_id": 38472},
        "thread_crt_snapshot_error": None,
        "source_board_candidate": True,
    }
    clear = {
        "event_kind": "board_score_target_write",
        "order": 1,
        "write_order": 0,
        "thread_id": 38472,
        "watched_address": board + 0x108,
        "exception_address": NATURAL_LOSS_TARGET_CLEAR_POST_EIP,
        "post_instruction_eip": NATURAL_LOSS_TARGET_CLEAR_POST_EIP,
        "code_window_address": NATURAL_LOSS_TARGET_CLEAR_POST_EIP - 16,
        "code_window_hex": (
            "65100000899fc80e0000899f08010000899f7c060000899f78060000899fd00e"
        ),
        "registers": {
            "Edi": board,
            "Ebx": 0,
            "Eip": NATURAL_LOSS_TARGET_CLEAR_POST_EIP,
        },
        "source_score_target": 9650,
        "previous_score_target": 9650,
        "post_score_target": 0,
        "same_active_board_as_source": True,
        "completion": False,
        "perf_counter_ns": 10_000,
        "framework_update": 11854,
        "board_address": board,
        "native_game_time": 0,
        "score": 7950,
        "score_target": 0,
        "curve_plan_exhausted": True,
    }
    positive = {
        "event_kind": "board_score_target_write",
        "order": 2,
        "write_order": 1,
        "thread_id": 38472,
        "watched_address": board + 0x108,
        "exception_address": NATURAL_LOSS_TARGET_POSITIVE_POST_EIP,
        "post_instruction_eip": NATURAL_LOSS_TARGET_POSITIVE_POST_EIP,
        "code_window_address": NATURAL_LOSS_TARGET_POSITIVE_POST_EIP - 16,
        "code_window_hex": (
            "7e108b960401000003d0899608010000eb0ac78608010000000000008bcee88a"
        ),
        "registers": {
            "Esi": board,
            "Edx": 9616,
            "Eip": NATURAL_LOSS_TARGET_POSITIVE_POST_EIP,
        },
        "source_score_target": 9650,
        "previous_score_target": 0,
        "post_score_target": 9616,
        "same_active_board_as_source": True,
        "completion": True,
        "perf_counter_ns": 224_700,
        "framework_update": 11854,
        "board_address": board,
        "native_game_time": 0,
        "score": 7950,
        "score_target": 9616,
        "curve_plan_exhausted": False,
    }
    trace = {
        "schema": "zuma-rl.pc-gameplay-mtrand-call-trace",
        "version": 1,
        "status": "PASS",
        "failure": None,
        "classification": (
            "read-only-hardware-breakpoint-preregistered-natural-loss-observation"
        ),
        "runtime_executable_sha256": EXPECTED_RETAIL_RUNTIME_SHA256,
        "breakpoint_kind": (
            BOARD_GLOBAL_PRECALL_AND_NATURAL_LOSS_TARGET_WRITE_KIND
        ),
        "address": BOARD_GLOBAL_PRECALL_ADDRESS,
        "board_global_precall_instruction_hex": "e86fbcfbff",
        "process_memory_writes": 0,
        "process_memory_mutation": False,
        "persistent_file_modified": False,
        "hardware_breakpoint_restored": True,
        "hardware_breakpoint_restore_error": None,
        "debugger_detach_error": None,
        "source_anchor_count": 1,
        "target_write_count": 2,
        "stop_reason": "natural_loss_target_write_complete",
        "natural_loss_target_clear_writer_address": (
            NATURAL_LOSS_TARGET_CLEAR_WRITER_ADDRESS
        ),
        "natural_loss_target_clear_writer_instruction_hex": (
            "899f08010000"
        ),
        "natural_loss_target_clear_post_eip": (
            NATURAL_LOSS_TARGET_CLEAR_POST_EIP
        ),
        "natural_loss_target_positive_writer_address": (
            NATURAL_LOSS_TARGET_POSITIVE_WRITER_ADDRESS
        ),
        "natural_loss_target_positive_writer_instruction_hex": (
            "899608010000"
        ),
        "natural_loss_target_positive_post_eip": (
            NATURAL_LOSS_TARGET_POSITIVE_POST_EIP
        ),
        "natural_loss_target_write_max_interval_ns": (
            NATURAL_LOSS_TARGET_WRITE_MAX_INTERVAL_NS
        ),
        "call_count": 3,
        "calls": [source, clear, positive],
        "target_write_events": [clear, positive],
        "target_write_completion": positive,
        "natural_loss_target_write_pair": {
            "clear": clear,
            "positive": positive,
            "complete": True,
        },
    }
    gameplay = {
        "status": "INCOMPLETE",
        "outcome": "timeout",
        "expected_outcome": "natural_loss",
        "gameplay_policy": "idle",
        "action_count": 0,
        "swap_count": 0,
        "fruit_shot_count": 0,
        "terminal_observation": None,
        "final_state": {
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
        },
    }
    return trace, gameplay


def test_natural_loss_target_writer_pair_matches_frozen_sequence() -> None:
    trace, _ = _natural_loss_target_write_fixture()
    source = trace["calls"][0]
    events = trace["target_write_events"]
    assert _is_natural_loss_source_anchor_candidate(source)
    assert _is_natural_loss_target_clear_event(
        source=source,
        event=events[0],
    )
    assert _is_natural_loss_target_write_pair(
        source=source,
        events=events,
    )


def test_natural_loss_target_writer_promotes_only_after_exact_trace() -> None:
    trace, gameplay = _natural_loss_target_write_fixture()
    promoted = _promote_natural_loss_target_write_gameplay(
        gameplay,
        trace,
        trace_sha256="sha256:" + "a" * 64,
    )
    assert promoted["status"] == "PASS"
    assert promoted["outcome"] == "natural_loss"
    assert promoted["polling_status_before_target_write_receipt"] == (
        "INCOMPLETE"
    )
    assert promoted["polling_outcome_before_target_write_receipt"] == (
        "timeout"
    )
    assert promoted["terminal_observation"]["kind"] == (
        NATURAL_LOSS_TARGET_WRITE_RECEIPT_KIND
    )
    assert promoted["terminal_observation"]["target_write_interval_ns"] == (
        214_700
    )


@pytest.mark.parametrize(
    "mutation",
    (
        lambda trace, gameplay: trace.update(
            breakpoint_kind=BOARD_GLOBAL_PRECALL_AND_TARGET_WRITE_KIND
        ),
        lambda trace, gameplay: trace["target_write_events"][0].update(
            post_instruction_eip=0x00412EF4
        ),
        lambda trace, gameplay: trace["target_write_events"][0].update(
            code_window_hex="00" * 32
        ),
        lambda trace, gameplay: trace["target_write_events"][1].update(
            previous_score_target=1
        ),
        lambda trace, gameplay: trace["target_write_events"][1].update(
            perf_counter_ns=(
                trace["target_write_events"][0]["perf_counter_ns"]
                + NATURAL_LOSS_TARGET_WRITE_MAX_INTERVAL_NS
                + 1
            )
        ),
        lambda trace, gameplay: gameplay["final_state"].update(
            board_address=0x257AA224
        ),
    ),
)
def test_natural_loss_target_writer_promotion_fails_closed(
    mutation: object,
) -> None:
    trace, gameplay = _natural_loss_target_write_fixture()
    mutation(trace, gameplay)
    with pytest.raises(RuntimeError, match="natural-loss"):
        _promote_natural_loss_target_write_gameplay(
            gameplay,
            trace,
            trace_sha256="sha256:" + "a" * 64,
        )


def test_hardware_context_retarget_and_restore_are_slot_zero_only() -> None:
    context = SimpleNamespace(
        Dr0=10,
        Dr1=11,
        Dr2=12,
        Dr3=13,
        Dr6=14,
        Dr7=0x000F0002,
        EFlags=0,
    )
    original = (10, 11, 12, 13, 14, 0x000F0002)
    _retarget_hardware_breakpoint_context(context, 0x00419A90)
    assert context.Dr0 == 0x00419A90
    assert (context.Dr1, context.Dr2, context.Dr3) == (11, 12, 13)
    assert context.Dr6 == 0
    assert context.Dr7 & 0x1
    assert not context.Dr7 & 0x000F0000
    _restore_hardware_breakpoint_context(context, original)
    assert (
        context.Dr0,
        context.Dr1,
        context.Dr2,
        context.Dr3,
        context.Dr6,
        context.Dr7,
    ) == original


def test_four_byte_write_watchpoint_has_frozen_dr7_encoding() -> None:
    context = SimpleNamespace(
        Dr0=0,
        Dr1=11,
        Dr2=12,
        Dr3=13,
        Dr6=14,
        Dr7=0,
        EFlags=0,
    )
    _retarget_hardware_breakpoint_context(
        context,
        0x25981338,
        access="write4",
    )
    assert context.Dr0 == 0x25981338
    assert context.Dr6 == 0
    assert context.Dr7 & 0x000F0003 == 0x000D0001
    with pytest.raises(ValueError, match="aligned"):
        _retarget_hardware_breakpoint_context(
            context,
            0x25981339,
            access="write4",
        )


def test_board_global_precall_candidate_requires_live_below_target_board() -> None:
    board = {
        "framework_update": 3120,
        "board_address": 0x12340000,
        "native_game_time": 109,
        "score": 7950,
        "score_target": 9650,
    }
    assert _is_source_board_anchor_candidate(board)
    assert not _is_source_board_anchor_candidate(board | {"board_address": 0})
    assert not _is_source_board_anchor_candidate(board | {"score_target": 0})
    assert not _is_source_board_anchor_candidate(board | {"score": 9650})


@pytest.mark.parametrize(
    "trace_kind",
    [
        BOARD_GLOBAL_PRECALL_KIND,
        BOARD_GLOBAL_PRECALL_AND_RESET_KIND,
        BOARD_GLOBAL_PRECALL_AND_TARGET_WRITE_KIND,
        BOARD_GLOBAL_PRECALL_AND_NATURAL_LOSS_TARGET_WRITE_KIND,
    ],
)
def test_board_anchor_monitor_is_exactly_derived_from_one_callsite_row(
    tmp_path,
    trace_kind: str,
) -> None:
    trace = {
        "schema": "zuma-rl.pc-gameplay-mtrand-call-trace",
        "version": 1,
        "status": "PASS",
        "failure": None,
        "breakpoint_kind": trace_kind,
        "address": BOARD_GLOBAL_PRECALL_ADDRESS,
        "process_id": 1234,
        "main_thread_id": 5678,
        "process_memory_writes": 0,
        "process_memory_mutation": False,
        "persistent_file_modified": False,
        "hardware_breakpoint_restored": True,
        "calls": [
            {
                "order": 5,
                "thread_id": 5678,
                "caller": BOARD_GLOBAL_PRECALL_RETURN,
                "framework_update": 3120,
                "board_address": 0x12340000,
                "native_game_time": 109,
                "score": 7950,
                "score_target": 9650,
                "pre_index": 431,
                "pre_state_sha256": "sha256:" + "1" * 64,
                "output": 1161337508,
                "post_index": 432,
                "post_state_sha256": "sha256:" + "2" * 64,
                "perf_counter_ns": 123456789,
                "thread_crt_rand_state": 613886368,
                "thread_crt_state": {
                    "thread_id": 5678,
                    "ptd_thread_id": 5678,
                    "rand_state_address": 0x45670000,
                },
                "thread_crt_snapshot_error": None,
                "call_address": BOARD_GLOBAL_PRECALL_ADDRESS,
                "call_instruction_hex": "e86fbcfbff",
                "source_board_candidate": True,
            }
        ],
    }
    target = tmp_path / "rng.ndjson"
    receipt = _write_board_anchor_monitor(target, trace)
    lines = target.read_text(encoding="ascii").splitlines()
    assert len(lines) == 2
    assert receipt["source_call_order"] == 5
    assert receipt["source_caller"] == BOARD_GLOBAL_PRECALL_RETURN
    assert receipt["source"] == trace_kind
    assert receipt["source_boundary_kind"] == BOARD_GLOBAL_PRECALL_KIND
    assert receipt["process_memory_writes"] == 0
    assert '"global_mtrand_index":431' in lines[1]
    assert '"global_mtrand_sha256":"' + "1" * 64 + '"' in lines[1]


def test_navigation_schedule_accepts_old_stable_replay_timing() -> None:
    assert _validate_navigation_updates(1622, 2422, 3108) == (
        1622,
        2422,
        3108,
    )


@pytest.mark.parametrize(
    "updates",
    [
        (-1, 2, 3),
        (1, 1, 3),
        (1, 3, 3),
        (3, 2, 1),
        (True, 2, 3),
    ],
)
def test_navigation_schedule_rejects_invalid_updates(
    updates: tuple[int, int, int],
) -> None:
    with pytest.raises(ValueError, match="navigation updates"):
        _validate_navigation_updates(*updates)


def test_navigation_cli_accepts_explicit_absolute_schedule() -> None:
    args = _parser().parse_args(
        [
            "--output-root",
            "out",
            "--runtime-executable",
            "game.exe",
            "--changedir",
            "assets",
            "--prestate-dir",
            "prestate",
            "--title-start-update",
            "1622",
            "--adventure-update",
            "2422",
            "--continue-game-update",
            "3108",
            "--title-start-point",
            "400",
            "533",
            "--adventure-point",
            "633",
            "281",
            "--continue-game-point",
            "399",
            "456",
            "--prelaunch-cursor-screen-point",
            "100",
            "100",
        ]
    )
    assert (
        args.title_start_update,
        args.adventure_update,
        args.continue_game_update,
    ) == (1622, 2422, 3108)
    assert args.title_start_point == [400.0, 533.0]
    assert args.adventure_point == [633.0, 281.0]
    assert args.continue_game_point == [399.0, 456.0]
    assert args.prelaunch_cursor_screen_point == [100, 100]


def test_navigation_point_accepts_old_stable_replay_coordinates() -> None:
    assert _validate_navigation_point("title", (400, 533)) == (
        400.0,
        533.0,
    )


@pytest.mark.parametrize(
    "point",
    [
        (-1.0, 10.0),
        (800.0, 10.0),
        (10.0, 600.0),
        (float("inf"), 10.0),
        (10.0, float("nan")),
        (True, 10.0),
    ],
)
def test_navigation_point_rejects_out_of_canvas_values(
    point: tuple[float, float],
) -> None:
    with pytest.raises(ValueError, match="navigation point"):
        _validate_navigation_point("test", point)


def test_prelaunch_cursor_point_accepts_virtual_desktop_coordinates() -> None:
    assert _validate_prelaunch_cursor_screen_point((-100, 100)) == (
        -100,
        100,
    )


@pytest.mark.parametrize(
    "point",
    [
        (True, 100),
        (100, False),
        (2**31, 0),
        (0, -(2**31) - 1),
    ],
)
def test_prelaunch_cursor_point_rejects_non_int32_values(
    point: tuple[int, int],
) -> None:
    with pytest.raises(ValueError, match="prelaunch cursor point"):
        _validate_prelaunch_cursor_screen_point(point)
