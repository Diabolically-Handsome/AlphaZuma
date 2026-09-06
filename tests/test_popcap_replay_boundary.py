"""Tests for strict live/offline PopCap DMO command boundaries."""

from __future__ import annotations

import struct

import pytest

from tools.popcap_replay_boundary import (
    DEFERRED_FILE_WRITE_CALLER,
    MAIN_DEMO_CALLER,
    REGISTRY_WRITE_DEMO_CALLER,
    REGISTRY_WRITE_LIVE_PAYLOAD_COMMAND,
    _attach_stabilization_boundary_failure,
    _audited_service_header_reentry,
    _audited_service_payload_reentry,
    _audited_terminal_registry_write_eof_exit,
    _audited_service_prepared_reentry,
    _audited_service_command_order_rebase,
    _audited_service_exit_late_idle_bridge,
    _audited_service_exit_overdue_idle_reentry,
    _audited_service_exit_overdue_idle_prepared_reentry,
    _audited_service_exit_late_idle_bridge_prepared_reentry,
    _audited_late_idle_native_timeline_rebase,
    _audited_successful_file_write_tail_prefetch,
    _audited_preloading_failed_file_write_tail_prefetch,
    _audited_preloading_failed_file_write_tail_short_header_recovery,
    _audited_service_exit_reentry,
    _audited_service_exit_prepared_reentry,
    _audited_service_exit_partial_header,
    _audited_service_exit_forced_read,
    _audited_service_file_write_header_claim,
    _audited_service_file_write_payload_completion,
    _audited_file_write_command_order_rebase,
    _audited_successful_file_write_corridor_prepared_exit,
    _audited_successful_file_write_deferred_exit,
    _audited_successful_file_write_deferred_exit_committed_boundary_snapshot,
    _audited_successful_file_write_deferred_exit_clamped_suffix_rebase,
    _audited_successful_file_write_deferred_exit_timeline_commit,
    _audited_successful_file_write_debt_barrier,
    _audited_post_blackout_attach_stabilization,
    _audited_post_blackout_file_write_offset_envelope,
    _audited_post_blackout_file_write_offset_envelope_set,
    _normalize_blackout_native_timeline_snapshot,
    _audited_deferred_file_write_result,
    _audited_font_cache_manifest_completion,
    _audited_main_file_read_corridor,
    _audited_terminal_file_write_payload_handoff,
    _board_seed_override_applies,
    _is_bounded_late_preloading_failed_file_write_corridor,
    _is_exact_startup_font_cache_failed_write_partition,
    _is_uniform_successful_file_write_corridor,
    _may_adopt_prepared_service_block,
    _may_broker_terminal_prepared_service_row,
    _may_bypass_stalled_service_block,
    _probe_service_progress,
    _service_block,
    _service_continuation_corridor,
    _service_settle_end,
    _snapshot_boundary_failure,
    _source_bound_board_anchor_applies,
    _source_bound_global_correction_eligible,
    _source_bound_global_post_correction_delta,
    _wait_for_service_boundary,
    _wait_for_startup_post_bypass_worker_boundary,
    _commit_startup_post_bypass_worker_continuation,
    _consume_startup_post_bypass_worker_echo,
)


def test_deferred_file_write_discharges_exact_accounted_result() -> None:
    rows = [
        _row(
            start=100,
            end=111,
            update=395,
            command_number=16,
            payload={"success": False},
        ),
        _row(
            start=111,
            end=122,
            update=396,
            command_number=16,
            payload={"success": False},
        ),
        _row(start=122, end=132, update=456, command_number=9),
        _row(start=132, end=142, update=471, command_number=31),
    ]
    snapshot = _snapshot(
        return_address=DEFERRED_FILE_WRITE_CALLER,
        update=456,
        command_bit_position=122,
        buffer_read_bit_position=132,
        command_number=9,
        command_order=0,
        last_demo_update=456,
        needs_command=1,
        demo_loading_complete=1,
    )

    assert _audited_deferred_file_write_result(
        _index(rows),
        rows,
        snapshot,
        command_order_offset=2,
        accounted_rows=(0, 1),
        discharge_count=0,
    ) == (0, False, None)


def test_deferred_file_write_requires_verified_service_window_pre_loading() -> None:
    rows = [
        _row(
            start=100,
            end=111,
            update=395,
            command_number=16,
            payload={"success": False},
        ),
        _row(start=111, end=121, update=471, command_number=31),
    ]
    snapshot = _snapshot(
        return_address=DEFERRED_FILE_WRITE_CALLER,
        update=423,
        command_bit_position=111,
        buffer_read_bit_position=121,
        command_number=31,
        command_order=0,
        last_demo_update=471,
        needs_command=0,
        demo_loading_complete=0,
    )
    kwargs = {
        "command_order_offset": 1,
        "accounted_rows": (0,),
        "discharge_count": 0,
    }

    _, _, failure = _audited_deferred_file_write_result(
        _index(rows), rows, snapshot, **kwargs
    )
    assert failure is not None
    assert "outside the post-loading boundary" in failure
    assert _audited_deferred_file_write_result(
        _index(rows),
        rows,
        snapshot,
        allow_pre_loading_after_service_continuation=True,
        **kwargs,
    ) == (0, False, None)


def test_deferred_file_write_leaves_due_file_write_to_normal_replay() -> None:
    rows = [
        _row(
            start=100,
            end=111,
            update=395,
            command_number=16,
            payload={"success": False},
        ),
        _row(start=111, end=121, update=456, command_number=9),
        _row(
            start=121,
            end=132,
            update=456,
            command_number=16,
            payload={"success": True},
        ),
    ]
    snapshot = _snapshot(
        return_address=DEFERRED_FILE_WRITE_CALLER,
        update=456,
        command_bit_position=111,
        buffer_read_bit_position=121,
        command_number=9,
        command_order=0,
        last_demo_update=456,
        needs_command=1,
        demo_loading_complete=1,
    )

    assert _audited_deferred_file_write_result(
        _index(rows),
        rows,
        snapshot,
        command_order_offset=1,
        accounted_rows=(0,),
        discharge_count=0,
    ) == (None, None, None)


@pytest.mark.parametrize(
    ("needs_command", "buffer_read_bit_position"),
    ((0, 110), (1, 111)),
)
def test_deferred_file_write_leaves_exact_one_tick_early_native_progress(
    needs_command: int,
    buffer_read_bit_position: int,
) -> None:
    rows = [
        _row(
            start=100,
            end=111,
            update=440,
            command_number=16,
            kind="file_write",
            payload={"success": False},
        )
    ]
    snapshot = _snapshot(
        return_address=DEFERRED_FILE_WRITE_CALLER,
        update=439,
        command_bit_position=100,
        buffer_read_bit_position=buffer_read_bit_position,
        command_number=16,
        command_order=0,
        last_demo_update=440,
        needs_command=needs_command,
        demo_loading_complete=1,
    )

    assert _audited_deferred_file_write_result(
        _index(rows),
        rows,
        snapshot,
        command_order_offset=0,
        accounted_rows=(),
        discharge_count=0,
    ) == (None, None, None)


def test_deferred_file_write_rejects_native_progress_more_than_one_tick_early(
) -> None:
    rows = [
        _row(
            start=100,
            end=111,
            update=440,
            command_number=16,
            kind="file_write",
            payload={"success": False},
        )
    ]
    snapshot = _snapshot(
        return_address=DEFERRED_FILE_WRITE_CALLER,
        update=438,
        command_bit_position=100,
        buffer_read_bit_position=110,
        command_number=16,
        command_order=0,
        last_demo_update=440,
        needs_command=0,
        demo_loading_complete=1,
    )

    _, _, failure = _audited_deferred_file_write_result(
        _index(rows),
        rows,
        snapshot,
        command_order_offset=0,
        accounted_rows=(),
        discharge_count=0,
    )

    assert failure is not None
    assert "not fully consumed" in failure


def test_deferred_file_write_discharges_at_complete_future_prepared_row() -> None:
    rows = [
        _row(
            start=100,
            end=111,
            update=395,
            command_number=16,
            payload={"success": False},
        ),
        _row(start=111, end=121, update=456, command_number=9),
        _row(start=121, end=131, update=471, command_number=31),
    ]
    snapshot = _snapshot(
        return_address=DEFERRED_FILE_WRITE_CALLER,
        update=456,
        command_bit_position=121,
        buffer_read_bit_position=131,
        command_number=31,
        command_order=1,
        last_demo_update=471,
        needs_command=0,
        demo_loading_complete=1,
    )

    assert _audited_deferred_file_write_result(
        _index(rows),
        rows,
        snapshot,
        command_order_offset=1,
        accounted_rows=(0,),
        discharge_count=0,
    ) == (0, False, None)


def test_deferred_file_write_allows_only_receipted_one_tick_overdue_prepared_row(
) -> None:
    rows = [
        _row(
            start=100,
            end=111,
            update=395,
            command_number=16,
            payload={"success": False},
        ),
        _row(start=111, end=121, update=456, command_number=9),
        _row(start=121, end=131, update=471, command_number=31),
    ]
    snapshot = _snapshot(
        return_address=DEFERRED_FILE_WRITE_CALLER,
        update=457,
        command_bit_position=111,
        buffer_read_bit_position=121,
        command_number=9,
        command_order=0,
        last_demo_update=456,
        needs_command=0,
        demo_loading_complete=1,
    )
    kwargs = {
        "command_order_offset": 1,
        "accounted_rows": (0,),
        "discharge_count": 0,
    }

    _, _, failure = _audited_deferred_file_write_result(
        _index(rows), rows, snapshot, **kwargs
    )
    assert failure is not None
    assert "prepared row is overdue" in failure
    assert _audited_deferred_file_write_result(
        _index(rows),
        rows,
        snapshot,
        allow_one_tick_overdue_prepared=True,
        **kwargs,
    ) == (0, False, None)

    clamped_snapshot = dict(snapshot, last_demo_update=457)
    _, _, clamped_without_receipt = _audited_deferred_file_write_result(
        _index(rows), rows, clamped_snapshot, **kwargs
    )
    assert clamped_without_receipt is not None
    assert "current boundary is invalid" in clamped_without_receipt
    assert _audited_deferred_file_write_result(
        _index(rows),
        rows,
        clamped_snapshot,
        allow_one_tick_overdue_prepared=True,
        **kwargs,
    ) == (0, False, None)

    two_ticks_late = dict(snapshot, update=458)
    _, _, two_ticks_late_failure = _audited_deferred_file_write_result(
        _index(rows),
        rows,
        two_ticks_late,
        allow_one_tick_overdue_prepared=True,
        **kwargs,
    )
    assert two_ticks_late_failure is not None
    assert "prepared row is overdue" in two_ticks_late_failure


def test_deferred_file_write_rejects_exhausted_result_debt() -> None:
    rows = [
        _row(
            start=100,
            end=111,
            update=395,
            command_number=16,
            payload={"success": False},
        ),
        _row(start=111, end=121, update=456, command_number=9),
        _row(start=121, end=131, update=471, command_number=31),
    ]
    snapshot = _snapshot(
        return_address=DEFERRED_FILE_WRITE_CALLER,
        update=456,
        command_bit_position=111,
        buffer_read_bit_position=121,
        command_number=9,
        command_order=0,
        last_demo_update=456,
        needs_command=1,
        demo_loading_complete=1,
    )

    _, _, failure = _audited_deferred_file_write_result(
        _index(rows),
        rows,
        snapshot,
        command_order_offset=1,
        accounted_rows=(0,),
        discharge_count=1,
    )

    assert failure is not None
    assert "debt is exhausted" in failure


def test_deferred_file_write_uses_labeled_pre_attach_debt_last() -> None:
    rows = [
        _row(
            start=100,
            end=111,
            update=0,
            command_number=16,
            payload={"success": False},
        ),
        _row(
            start=111,
            end=122,
            update=395,
            command_number=16,
            payload={"success": False},
        ),
        _row(start=122, end=132, update=456, command_number=9),
        _row(start=132, end=142, update=471, command_number=31),
    ]
    snapshot = _snapshot(
        return_address=DEFERRED_FILE_WRITE_CALLER,
        update=456,
        command_bit_position=122,
        buffer_read_bit_position=132,
        command_number=9,
        command_order=1,
        last_demo_update=456,
        needs_command=1,
        demo_loading_complete=1,
    )

    assert _audited_deferred_file_write_result(
        _index(rows),
        rows,
        snapshot,
        command_order_offset=1,
        accounted_rows=(1,),
        discharge_count=1,
        pre_attach_rows=(0,),
        pre_attach_before_update=230,
    ) == (0, False, None)


def test_deferred_file_write_excludes_padding_and_merges_service_rows() -> None:
    rows = [
        _row(
            start=100 + index * 11,
            end=111 + index * 11,
            update=395 + index,
            command_number=16,
            payload={"success": True},
        )
        for index in range(5)
    ]
    rows.append(_row(start=155, end=165, update=456, command_number=9))
    rows.append(_row(start=165, end=175, update=471, command_number=31))
    snapshot = _snapshot(
        return_address=DEFERRED_FILE_WRITE_CALLER,
        update=456,
        command_bit_position=155,
        buffer_read_bit_position=165,
        command_number=9,
        command_order=2,
        last_demo_update=456,
        needs_command=1,
        demo_loading_complete=1,
    )
    kwargs = {
        "command_order_offset": 3,
        "accounted_rows": (0, 2, 4),
        "excluded_accounted_rows": (4,),
        "service_consumed_rows": (1, 3),
    }

    for discharge_count, expected_row in enumerate((0, 1, 2, 3)):
        assert _audited_deferred_file_write_result(
            _index(rows),
            rows,
            snapshot,
            discharge_count=discharge_count,
            **kwargs,
        ) == (expected_row, True, None)

    _, _, exhausted = _audited_deferred_file_write_result(
        _index(rows),
        rows,
        snapshot,
        discharge_count=4,
        **kwargs,
    )
    assert exhausted is not None
    assert "debt is exhausted" in exhausted

    _, _, invalid = _audited_deferred_file_write_result(
        _index(rows),
        rows,
        snapshot,
        command_order_offset=3,
        accounted_rows=(0, 2, 4),
        excluded_accounted_rows=(3,),
        service_consumed_rows=(1,),
        discharge_count=0,
    )
    assert invalid is not None
    assert "excluded rows are not accounted" in invalid


def test_successful_file_write_debt_barrier_retires_only_exact_tail() -> None:
    rows = [
        _row(
            start=100 + index * 11,
            end=111 + index * 11,
            update=395,
            command_number=16,
            kind="file_write",
            payload={"success": True},
        )
        for index in range(8)
    ]
    rows.extend(
        (
            _row(start=188, end=198, update=410, command_number=31),
            _row(start=198, end=208, update=425, command_number=31),
            _row(
                start=208,
                end=219,
                update=456,
                command_number=12,
                kind="registry_write",
                payload={"success": True},
            ),
        )
    )
    snapshot = _snapshot(
        return_address=MAIN_DEMO_CALLER,
        update=456,
        command_bit_position=208,
        buffer_read_bit_position=218,
        command_number=12,
        command_order=5,
        last_demo_update=456,
        needs_command=1,
        demo_loading_complete=1,
    )
    kwargs = {
        "command_order_offset": 5,
        "accounted_rows": (0, 2, 4, 6, 7),
        "excluded_accounted_rows": (7,),
        "service_consumed_rows": (1, 3, 5),
        "diagnostic_padding_rows": (7,),
    }

    assert _audited_successful_file_write_debt_barrier(
        rows,
        snapshot,
        discharged_rows=(0, 1, 2, 3),
        **kwargs,
    ) == ((4, 5, 6), None)

    surplus, failure = _audited_successful_file_write_debt_barrier(
        rows,
        snapshot,
        discharged_rows=(0, 2, 3, 4),
        **kwargs,
    )
    assert surplus == ()
    assert failure is not None
    assert "discharged prefix" in failure


def test_terminal_file_write_payload_handoff_accepts_exact_last_bit() -> None:
    rows = [
        _row(
            start=100,
            end=111,
            update=396,
            command_number=16,
            payload={"success": False},
        )
    ]
    snapshot = _snapshot(
        return_address=DEFERRED_FILE_WRITE_CALLER,
        update=395,
        command_bit_position=100,
        buffer_read_bit_position=110,
        command_number=16,
        command_order=0,
        last_demo_update=395,
        needs_command=1,
        demo_loading_complete=0,
    )

    assert _audited_terminal_file_write_payload_handoff(
        rows,
        snapshot,
        row_index=0,
        command_order_offset=0,
    ) == (False, None)


def test_terminal_file_write_payload_handoff_rejects_nonexact_read() -> None:
    rows = [
        _row(
            start=100,
            end=111,
            update=396,
            command_number=16,
            payload={"success": False},
        )
    ]
    snapshot = _snapshot(
        return_address=DEFERRED_FILE_WRITE_CALLER,
        update=395,
        command_bit_position=100,
        buffer_read_bit_position=109,
        command_number=16,
        command_order=0,
        last_demo_update=395,
        needs_command=1,
        demo_loading_complete=0,
    )

    _, failure = _audited_terminal_file_write_payload_handoff(
        rows,
        snapshot,
        row_index=0,
        command_order_offset=0,
    )

    assert failure is not None
    assert "buffer_read_bit_position mismatch" in failure


def test_service_file_write_claim_accepts_exact_early_header() -> None:
    rows = [
        _row(
            start=100 + index * 11,
            end=111 + index * 11,
            update=10,
            command_number=16,
            payload={"success": False},
        )
        for index in range(6)
    ]
    snapshot = _snapshot(
        return_address=DEFERRED_FILE_WRITE_CALLER,
        needs_command=0,
        command_number=16,
        is_short=0,
        command_order=5,
        command_bit_position=155,
        buffer_read_bit_position=165,
        update=8,
        last_demo_update=8,
    )

    assert _audited_service_file_write_header_claim(
        rows,
        snapshot,
        start_index=0,
        end_index=5,
        previous_read_bit_position=155,
        previous_command_order=4,
        previous_reentry_kind=1,
    ) == (5, None)


def test_service_file_write_payload_completion_accepts_exact_false_result(
) -> None:
    rows = [
        _row(
            start=100,
            end=111,
            update=10,
            command_number=16,
            kind="file_write",
            payload={"success": False},
        )
    ]
    snapshot = _snapshot(
        return_address=MAIN_DEMO_CALLER,
        needs_command=1,
        command_number=16,
        is_short=0,
        command_order=0,
        command_bit_position=100,
        buffer_read_bit_position=111,
        update=10,
        last_demo_update=10,
        demo_loading_complete=0,
    )

    assert _audited_service_file_write_payload_completion(
        rows,
        snapshot,
        row_index=0,
        previous_read_bit_position=110,
        previous_command_bit_position=100,
        previous_command_order=0,
        previous_reentry_kind=13,
    ) == (0, None)


def test_broker_adjacent_current_update_file_write_chain_accepts_exact_race(
) -> None:
    """Reproduce the four audited states from the formal PC-001 race."""

    rows = [
        _row(
            start=100 + index * 11,
            end=111 + index * 11,
            update=(41, 42, 42, 42, 43)[index],
            command_number=16,
            kind="file_write",
            payload={"success": False},
        )
        for index in range(5)
    ]
    rows.extend(
        (
            _row(
                start=155,
                end=165,
                update=58,
                command_number=31,
                kind="idle",
                payload={},
            ),
            _row(
                start=165,
                end=175,
                update=73,
                command_number=31,
                kind="idle",
                payload={},
            ),
        )
    )

    worker_claim = _snapshot(
        return_address=DEFERRED_FILE_WRITE_CALLER,
        needs_command=0,
        command_number=16,
        is_short=0,
        command_order=1,
        command_bit_position=111,
        buffer_read_bit_position=121,
        update=42,
        last_demo_update=42,
        demo_loading_complete=0,
    )
    assert _audited_service_file_write_header_claim(
        rows,
        worker_claim,
        start_index=0,
        end_index=4,
        previous_read_bit_position=111,
        previous_command_order=0,
        previous_reentry_kind=1,
        require_before_recorded_update=False,
    ) == (1, None)

    main_payload_completion = _snapshot(
        return_address=MAIN_DEMO_CALLER,
        needs_command=1,
        command_number=16,
        is_short=0,
        command_order=1,
        command_bit_position=111,
        buffer_read_bit_position=122,
        update=42,
        last_demo_update=42,
        demo_loading_complete=0,
    )
    assert _audited_service_file_write_payload_completion(
        rows,
        main_payload_completion,
        row_index=1,
        previous_read_bit_position=121,
        previous_command_bit_position=111,
        previous_command_order=1,
        previous_reentry_kind=13,
    ) == (1, None)

    next_header = _snapshot(
        return_address=MAIN_DEMO_CALLER,
        needs_command=1,
        command_number=16,
        is_short=0,
        command_order=2,
        command_bit_position=122,
        buffer_read_bit_position=132,
        update=42,
        last_demo_update=42,
        demo_loading_complete=0,
    )
    assert _audited_service_prepared_reentry(
        rows,
        next_header,
        start_index=0,
        end_index=4,
        previous_read_bit_position=122,
        previous_command_order=1,
        previous_command_bit_position=111,
        previous_reentry_kind=1,
        require_before_recorded_update=False,
        allow_bounded_late_preloading_failed_file_write_corridor=True,
    ) == (2, None)

    tail_prefetch = _snapshot(
        return_address=MAIN_DEMO_CALLER,
        needs_command=1,
        command_number=0,
        is_short=0,
        command_order=3,
        command_bit_position=132,
        buffer_read_bit_position=166,
        update=42,
        last_demo_update=42,
        demo_loading_complete=0,
    )
    assert _audited_preloading_failed_file_write_tail_prefetch(
        rows,
        tail_prefetch,
        corridor_start_index=0,
        corridor_end_index=4,
        previous_read_bit_position=132,
        previous_command_bit_position=122,
        previous_command_order=2,
        previous_update=42,
        previous_reentry_kind=2,
        command_order_offset=0,
    ) == (6, None)


@pytest.mark.parametrize(
    ("target", "field", "value", "expected"),
    (
        (
            "snapshot",
            "return_address",
            DEFERRED_FILE_WRITE_CALLER,
            "return_address",
        ),
        (
            "snapshot",
            "buffer_read_bit_position",
            110,
            "buffer_read_bit_position",
        ),
        ("row", "payload", {"success": True}, "ineligible"),
        ("previous", "read", 109, "claimed header state"),
    ),
)
def test_service_file_write_payload_completion_fails_closed(
    target: str,
    field: str,
    value: object,
    expected: str,
) -> None:
    rows = [
        _row(
            start=100,
            end=111,
            update=10,
            command_number=16,
            kind="file_write",
            payload={"success": False},
        )
    ]
    values = {
        "return_address": MAIN_DEMO_CALLER,
        "needs_command": 1,
        "command_number": 16,
        "is_short": 0,
        "command_order": 0,
        "command_bit_position": 100,
        "buffer_read_bit_position": 111,
        "update": 10,
        "last_demo_update": 10,
        "demo_loading_complete": 0,
    }
    previous_read = 110
    if target == "snapshot":
        values[field] = value
    elif target == "row":
        rows[0][field] = value
    else:
        previous_read = int(value)
    snapshot = _snapshot(**values)

    _, failure = _audited_service_file_write_payload_completion(
        rows,
        snapshot,
        row_index=0,
        previous_read_bit_position=previous_read,
        previous_command_bit_position=100,
        previous_command_order=0,
        previous_reentry_kind=13,
    )

    assert failure is not None
    assert expected in failure


def _font_manifest_case():
    manifest = {
        f"cached\\fonts\\600\\font{index:02d}.txt.cfw2": 100 + index
        for index in range(35)
    }
    rows = [
        _row(
            start=100 + index * 11,
            end=111 + index * 11,
            update=0 if index < 8 else 384 + (index - 8) // 2,
            command_number=16,
            kind="file_write",
            payload={"success": False},
        )
        for index in range(35)
    ]
    rows.append(
        _row(
            start=485,
            end=495,
            update=456,
            command_number=9,
            kind="loading_complete",
        )
    )
    snapshot = _snapshot(
        return_address=DEFERRED_FILE_WRITE_CALLER,
        update=456,
        command_bit_position=485,
        buffer_read_bit_position=495,
        command_number=9,
        command_order=34,
        last_demo_update=456,
        needs_command=0,
        demo_loading_complete=1,
    )
    kwargs = {
        "command_order_offset": 1,
        "accounted_rows": (20,),
        "discharge_count": 9,
        "pre_attach_rows": tuple(range(8)),
        "pre_attach_before_update": 230,
        "startup_file_write_rows": tuple(range(35)),
        "manifest": manifest,
        "observed_members": tuple(
            (f"cached\\fonts\\600\\font{index:02d}.txt.cfw2", 100 + index)
            for index in range(9)
        ),
        "current_member": (
            "C:\\ProgramData\\Steam\\ZumasRevenge\\"
            "cached\\fonts\\600\\font34.txt.cfw2"
        ),
        "current_size": 134,
        "completion_count": 0,
        "service_handoff_count": 0,
    }
    return rows, snapshot, kwargs


def test_font_cache_manifest_completion_accepts_exact_one_shot() -> None:
    rows, snapshot, kwargs = _font_manifest_case()

    assert _audited_font_cache_manifest_completion(
        _index(rows), rows, snapshot, **kwargs
    ) == (False, None)


def test_font_cache_manifest_completion_accepts_gapless_startup_receipt() -> None:
    rows, snapshot, kwargs = _font_manifest_case()
    kwargs["discharge_count"] = 1
    kwargs["pre_attach_rows"] = ()
    kwargs["pre_attach_before_update"] = None
    kwargs["direct_match_count"] = 8

    assert _audited_font_cache_manifest_completion(
        _index(rows), rows, snapshot, **kwargs
    ) == (False, None)


def test_font_cache_manifest_completion_accepts_verified_pre_loading_window() -> None:
    rows, snapshot, kwargs = _font_manifest_case()
    kwargs["discharge_count"] = 1
    kwargs["pre_attach_rows"] = ()
    kwargs["pre_attach_before_update"] = None
    kwargs["direct_match_count"] = 8
    kwargs["allow_pre_loading_after_service_continuation"] = True
    snapshot["demo_loading_complete"] = 0

    assert _audited_font_cache_manifest_completion(
        _index(rows), rows, snapshot, **kwargs
    ) == (False, None)


def test_font_cache_manifest_completion_accepts_bounded_next_use() -> None:
    rows, snapshot, kwargs = _font_manifest_case()
    kwargs["completion_count"] = 1
    kwargs["observed_members"] = (
        *kwargs["observed_members"],
        ("cached\\fonts\\600\\font34.txt.cfw2", 134),
    )
    kwargs["current_member"] = (
        "cached\\fonts\\600\\font33.txt.cfw2"
    )
    kwargs["current_size"] = 133

    assert _audited_font_cache_manifest_completion(
        _index(rows), rows, snapshot, **kwargs
    ) == (False, None)


def test_font_cache_manifest_completion_rejects_36th_operation() -> None:
    rows, snapshot, kwargs = _font_manifest_case()
    kwargs["completion_count"] = 18
    kwargs["observed_members"] = tuple(
        (f"cached\\fonts\\600\\font{index:02d}.txt.cfw2", 100 + index)
        for index in range(27)
    )
    kwargs["current_member"] = (
        "cached\\fonts\\600\\font27.txt.cfw2"
    )
    kwargs["current_size"] = 127

    _, failure = _audited_font_cache_manifest_completion(
        _index(rows), rows, snapshot, **kwargs
    )

    assert failure is not None
    assert "cardinality is exhausted" in failure


def test_font_cache_manifest_completion_rejects_wrong_live_size() -> None:
    rows, snapshot, kwargs = _font_manifest_case()
    kwargs["current_size"] = 999

    _, failure = _audited_font_cache_manifest_completion(
        _index(rows), rows, snapshot, **kwargs
    )

    assert failure is not None
    assert "size mismatch" in failure


def test_main_file_read_corridor_requires_repeated_large_payload() -> None:
    payload_sha256 = "sha256:" + "ab" * 32
    rows = [
        _row(
            start=100,
            end=200,
            update=1086,
            command_number=15,
            payload={
                "success": True,
                "data": {"bytes": 2225, "sha256": "sha256:" + "cd" * 32},
            },
        ),
        _row(
            start=200,
            end=300,
            update=1086,
            command_number=15,
            payload={
                "success": True,
                "data": {"bytes": 19384, "sha256": payload_sha256},
            },
        ),
        _row(
            start=300,
            end=311,
            update=1086,
            command_number=12,
            payload={"success": True},
        ),
        _row(
            start=311,
            end=322,
            update=1571,
            command_number=14,
            payload={"exists": True},
        ),
        _row(
            start=322,
            end=422,
            update=1571,
            command_number=15,
            payload={
                "success": True,
                "data": {"bytes": 19384, "sha256": payload_sha256},
            },
        ),
    ]

    receipt = _audited_main_file_read_corridor(
        rows,
        start_index=0,
        end_index=2,
    )

    assert receipt is not None
    assert receipt["large_file_read_row_index"] == 1
    assert receipt["large_file_read_bytes"] == 19384
    assert receipt["matching_large_payload_occurrences"] == 2
    rows[4]["payload"]["data"]["sha256"] = "sha256:" + "ef" * 32
    assert (
        _audited_main_file_read_corridor(
            rows,
            start_index=0,
            end_index=2,
        )
        is None
    )


def test_board_seed_skip_targets_requested_call() -> None:
    assert not _board_seed_override_applies(0, 1)
    assert _board_seed_override_applies(1, 1)
    assert _board_seed_override_applies(2, 1)


@pytest.mark.parametrize("hit_index,skip_count", [(-1, 0), (0, -1)])
def test_board_seed_skip_rejects_negative_values(
    hit_index: int,
    skip_count: int,
) -> None:
    with pytest.raises(ValueError):
        _board_seed_override_applies(hit_index, skip_count)


def test_source_bound_board_anchor_uses_constructor_owner_identity() -> None:
    assert _source_bound_board_anchor_applies(
        framework_update=3120,
        target_framework_update=3120,
        thread_id=66,
        main_thread_id=66,
        rng_owner_vtable=0x009884A4,
        expected_board_vtable=0x009884A4,
        observed_seed=263072237,
        expected_seed=263072237,
        global_post_state_match=True,
        active_board_context_match=True,
        attach_stabilization_collision=False,
    )


@pytest.mark.parametrize(
    "replacement",
    [
        {"framework_update": 3119},
        {"thread_id": 67},
        {"rng_owner_vtable": 0x009884A0},
        {"observed_seed": 263072238},
        {"global_post_state_match": False},
        {"active_board_context_match": False},
        {"attach_stabilization_collision": True},
    ],
)
def test_source_bound_board_anchor_rejects_partial_matches(
    replacement: dict[str, int | bool],
) -> None:
    values: dict[str, int | bool] = {
        "framework_update": 3120,
        "target_framework_update": 3120,
        "thread_id": 66,
        "main_thread_id": 66,
        "rng_owner_vtable": 0x009884A4,
        "expected_board_vtable": 0x009884A4,
        "observed_seed": 263072237,
        "expected_seed": 263072237,
        "global_post_state_match": True,
        "active_board_context_match": True,
        "attach_stabilization_collision": False,
    }
    values.update(replacement)
    assert not _source_bound_board_anchor_applies(**values)


def test_source_bound_board_anchor_accepts_only_prevalidated_correction() -> None:
    values = {
        "framework_update": 3120,
        "target_framework_update": 3120,
        "thread_id": 66,
        "main_thread_id": 66,
        "rng_owner_vtable": 0x009884A4,
        "expected_board_vtable": 0x009884A4,
        "observed_seed": 2044941490,
        "expected_seed": 263072237,
        "global_post_state_match": False,
        "active_board_context_match": True,
        "attach_stabilization_collision": False,
    }
    assert not _source_bound_board_anchor_applies(**values)
    assert _source_bound_board_anchor_applies(
        **values,
        bounded_global_correction=True,
    )


def test_source_bound_global_correction_uses_current_seed_match() -> None:
    assert not _source_bound_global_correction_eligible(
        allow_correction=True,
        live_post_draw_verified=True,
        correction_index_delta=0,
        observed_seed_match=True,
        global_post_state_match=True,
    )
    assert _source_bound_global_correction_eligible(
        allow_correction=True,
        live_post_draw_verified=True,
        correction_index_delta=4,
        observed_seed_match=False,
        global_post_state_match=False,
    )


def test_source_bound_global_correction_requires_same_bounded_state_words() -> None:
    state_words = bytes((index % 251 for index in range(624 * 4)))
    live = state_words + struct.pack("<I", 393)
    expected = state_words + struct.pack("<I", 402)

    assert _source_bound_global_post_correction_delta(live, expected) == 9
    assert (
        _source_bound_global_post_correction_delta(
            live,
            state_words + struct.pack("<I", 426),
        )
        is None
    )
    mutated = bytearray(expected)
    mutated[100] ^= 1
    assert (
        _source_bound_global_post_correction_delta(
            live,
            bytes(mutated),
        )
        is None
    )


def test_stalled_main_service_poll_gets_one_same_thread_retry() -> None:
    snapshot = {
        "return_address": MAIN_DEMO_CALLER,
        "needs_command": 0,
    }

    assert _may_bypass_stalled_service_block(
        snapshot,
        already_bypassed=False,
    )
    assert not _may_bypass_stalled_service_block(
        snapshot,
        already_bypassed=True,
    )


def test_stalled_prepared_orphan_gets_one_explicit_continuation() -> None:
    snapshot = {
        "return_address": MAIN_DEMO_CALLER,
        "needs_command": 1,
    }

    assert not _may_bypass_stalled_service_block(
        snapshot,
        already_bypassed=False,
    )
    assert _may_bypass_stalled_service_block(
        snapshot,
        already_bypassed=False,
        allow_prepared_orphan=True,
    )
    assert not _may_bypass_stalled_service_block(
        snapshot,
        already_bypassed=True,
        allow_prepared_orphan=True,
    )


def test_terminal_prepared_service_row_requires_background_handoff() -> None:
    rows = [
        _row(start=100, end=111, update=42, command_number=16),
        _row(start=111, end=122, update=43, command_number=16),
    ]

    assert _may_broker_terminal_prepared_service_row(
        rows,
        _snapshot(
            needs_command=1,
            command_number=16,
            command_bit_position=111,
            buffer_read_bit_position=121,
        ),
        prepared_row=1,
        corridor_end_index=1,
    )


@pytest.mark.parametrize(
    "prepared_row,overrides",
    (
        (0, {}),
        (1, {"needs_command": 0}),
        (1, {"buffer_read_bit_position": 122}),
        (1, {"return_address": 0x00680FB2}),
    ),
)
def test_terminal_prepared_service_handoff_fails_closed(
    prepared_row: int,
    overrides: dict[str, int],
) -> None:
    rows = [
        _row(start=100, end=111, update=42, command_number=16),
        _row(start=111, end=122, update=43, command_number=16),
    ]
    values = {
        "needs_command": 1,
        "command_number": 16,
        "command_bit_position": 111,
        "buffer_read_bit_position": 121,
    }
    values.update(overrides)

    assert not _may_broker_terminal_prepared_service_row(
        rows,
        _snapshot(**values),
        prepared_row=prepared_row,
        corridor_end_index=1,
    )


@pytest.mark.parametrize(
    "snapshot",
    (
        {
            "return_address": MAIN_DEMO_CALLER,
            "needs_command": 1,
        },
        {
            "return_address": 0x0068126E,
            "needs_command": 0,
        },
    ),
)
def test_stalled_service_bypass_rejects_non_main_or_needed_call(
    snapshot: dict[str, int],
) -> None:
    assert not _may_bypass_stalled_service_block(
        snapshot,
        already_bypassed=False,
    )


@pytest.mark.parametrize(
    (
        "already_bypassed",
        "allow_orphan",
        "orphan_already_used",
        "expected",
    ),
    (
        (True, False, True, True),
        (False, True, False, True),
        (False, True, True, False),
        (False, False, False, False),
    ),
)
def test_prepared_service_adoption_is_known_or_one_shot(
    already_bypassed: bool,
    allow_orphan: bool,
    orphan_already_used: bool,
    expected: bool,
) -> None:
    assert (
        _may_adopt_prepared_service_block(
            already_bypassed=already_bypassed,
            allow_orphan=allow_orphan,
            orphan_already_used=orphan_already_used,
        )
        is expected
    )


def _row(
    *,
    start: int,
    end: int,
    update: int,
    command_number: int,
    short_form: bool = False,
    kind: str | None = None,
    payload: dict[str, object] | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "start": start,
        "end": end,
        "update": update,
        "command_number": command_number,
        "short_form": short_form,
    }
    if kind is not None:
        row["kind"] = kind
    if payload is not None:
        row["payload"] = payload
    return row


def _snapshot(**overrides: int) -> dict[str, int]:
    result = {
        "return_address": MAIN_DEMO_CALLER,
        "needs_command": 0,
        "command_number": 11,
        "command_bit_position": 100,
        "command_order": 0,
        "is_short": 0,
        "last_demo_update": 42,
        "update": 42,
        "buffer_read_bit_position": 120,
    }
    result.update(overrides)
    return result


def _index(rows: list[dict[str, object]]):
    return {
        int(row["start"]): (index, row)
        for index, row in enumerate(rows)
    }


def test_service_block_extends_over_contiguous_service_commands() -> None:
    rows = [
        _row(start=100, end=120, update=42, command_number=11),
        _row(start=120, end=150, update=42, command_number=15),
        _row(start=150, end=170, update=42, command_number=22),
        _row(start=170, end=180, update=42, command_number=21),
        _row(start=180, end=200, update=42, command_number=12),
    ]

    assert _service_block(_index(rows), rows, _snapshot()) == (0, 2, 170)


def test_service_block_releases_a_prepared_main_thread_command() -> None:
    rows = [_row(start=100, end=120, update=42, command_number=11)]

    assert (
        _service_block(
            _index(rows),
            rows,
            _snapshot(needs_command=1),
        )
        is None
    )


def test_service_block_accepts_exact_prepared_retry_when_enabled() -> None:
    rows = [
        _row(start=100, end=140, update=42, command_number=15),
        _row(start=140, end=160, update=42, command_number=11),
    ]

    assert _service_block(
        _index(rows),
        rows,
        _snapshot(
            needs_command=1,
            command_number=15,
            buffer_read_bit_position=110,
        ),
        allow_prepared_command=True,
    ) == (0, 1, 160)


@pytest.mark.parametrize(
    "overrides",
    (
        {"buffer_read_bit_position": 109},
        {"buffer_read_bit_position": 111},
        {"is_short": 1},
    ),
)
def test_service_block_rejects_inexact_prepared_retry(
    overrides: dict[str, int],
) -> None:
    rows = [
        _row(
            start=100,
            end=120,
            update=42,
            command_number=11,
            short_form=bool(overrides.get("is_short", 0)),
        )
    ]

    assert (
        _service_block(
            _index(rows),
            rows,
            _snapshot(needs_command=1, **overrides),
            allow_prepared_command=True,
        )
        is None
    )


@pytest.mark.parametrize(
    ("override", "value"),
    [
        ("return_address", MAIN_DEMO_CALLER + 4),
        ("command_number", 21),
        ("command_bit_position", 999),
        ("last_demo_update", 43),
    ],
)
def test_service_block_rejects_non_racing_or_nonmatching_entries(
    override: str,
    value: int,
) -> None:
    rows = [_row(start=100, end=120, update=42, command_number=11)]

    assert (
        _service_block(
            _index(rows),
            rows,
            _snapshot(**{override: value}),
        )
        is None
    )


def test_service_block_stops_before_a_future_update() -> None:
    rows = [
        _row(start=100, end=120, update=42, command_number=11),
        _row(start=120, end=150, update=43, command_number=15),
        _row(start=150, end=160, update=44, command_number=31),
    ]

    assert _service_block(_index(rows), rows, _snapshot()) == (0, 0, 120)


@pytest.mark.parametrize("command_number", (9, 11, 15, 22, 31))
def test_service_settle_accepts_one_following_service_or_settle_header(
    command_number: int,
) -> None:
    rows = [
        _row(start=100, end=120, update=42, command_number=11),
        _row(
            start=120,
            end=150,
            update=43,
            command_number=command_number,
        ),
    ]

    assert _service_settle_end(rows, 0, 120) == 150


def test_service_settle_stops_before_following_gameplay_command() -> None:
    rows = [
        _row(start=100, end=120, update=42, command_number=11),
        _row(start=120, end=138, update=43, command_number=0),
    ]

    assert _service_settle_end(rows, 0, 120) == 120


def test_service_continuation_corridor_crosses_service_updates_only() -> None:
    rows = [
        _row(start=100, end=111, update=42, command_number=16),
        _row(start=111, end=122, update=43, command_number=16),
        _row(start=122, end=132, update=44, command_number=31),
    ]

    assert _service_continuation_corridor(rows, 0) == (1, 122)


def test_uniform_successful_file_write_corridor_is_exact_and_bounded() -> None:
    rows = [
        _row(
            start=100,
            end=111,
            update=42,
            command_number=16,
            kind="file_write",
            payload={"success": True},
        ),
        _row(
            start=111,
            end=122,
            update=42,
            command_number=16,
            kind="file_write",
            payload={"success": True},
        ),
    ]

    assert _is_uniform_successful_file_write_corridor(rows, 0, 1)
    rows[1]["update"] = 43
    assert not _is_uniform_successful_file_write_corridor(rows, 0, 1)
    rows[1]["update"] = 42
    rows[1]["payload"] = {"success": False}
    assert not _is_uniform_successful_file_write_corridor(rows, 0, 1)


def test_service_payload_reentry_accepts_exact_header_continuation() -> None:
    rows = [
        _row(start=100, end=111, update=42, command_number=16),
        _row(start=111, end=122, update=43, command_number=16),
        _row(start=122, end=133, update=44, command_number=16),
    ]
    snapshot = _snapshot(
        command_bit_position=121,
        buffer_read_bit_position=131,
        command_number=0,
        command_order=2,
        needs_command=0,
    )

    assert _audited_service_payload_reentry(
        rows,
        snapshot,
        start_index=0,
        end_index=2,
        previous_read_bit_position=120,
    ) == (1, None)


def test_service_payload_reentry_accepts_only_exact_registry_write_callback(
) -> None:
    rows = [
        _row(
            start=100,
            end=111,
            update=42,
            command_number=12,
            kind="registry_write",
            payload={"success": True},
        ),
        _row(
            start=111,
            end=122,
            update=42,
            command_number=12,
            kind="registry_write",
            payload={"success": True},
        ),
        _row(
            start=122,
            end=133,
            update=42,
            command_number=12,
            kind="registry_write",
            payload={"success": True},
        ),
    ]
    snapshot = _snapshot(
        return_address=REGISTRY_WRITE_DEMO_CALLER,
        command_bit_position=110,
        buffer_read_bit_position=120,
        command_number=REGISTRY_WRITE_LIVE_PAYLOAD_COMMAND,
        command_order=1,
        needs_command=0,
    )
    kwargs = {
        "start_index": 0,
        "end_index": 2,
        "previous_read_bit_position": 109,
    }

    assert _audited_service_payload_reentry(
        rows, snapshot, **kwargs
    ) == (0, None)

    following_payload_start = dict(
        snapshot,
        buffer_read_bit_position=121,
        needs_command=1,
    )
    assert _audited_service_payload_reentry(
        rows,
        following_payload_start,
        start_index=0,
        end_index=2,
        previous_read_bit_position=120,
        previous_command_bit_position=110,
        previous_command_order=1,
        previous_reentry_kind=1,
    ) == (0, None)

    chained_payload = dict(
        snapshot,
        command_bit_position=121,
        buffer_read_bit_position=132,
        command_order=2,
        needs_command=1,
    )
    assert _audited_service_payload_reentry(
        rows,
        chained_payload,
        start_index=0,
        end_index=2,
        previous_read_bit_position=121,
        previous_command_bit_position=110,
        previous_command_order=1,
        previous_reentry_kind=1,
    ) == (1, None)

    _, main_caller_failure = _audited_service_payload_reentry(
        rows,
        dict(snapshot, return_address=MAIN_DEMO_CALLER),
        **kwargs,
    )
    assert main_caller_failure is not None
    assert "return_address mismatch" in main_caller_failure

    _, ordinary_payload_command_failure = _audited_service_payload_reentry(
        rows,
        dict(snapshot, command_number=0),
        **kwargs,
    )
    assert ordinary_payload_command_failure is not None
    assert "command_number mismatch" in ordinary_payload_command_failure

    rows[0]["kind"] = "file_write"
    _, wrong_kind_failure = _audited_service_payload_reentry(
        rows, snapshot, **kwargs
    )
    assert wrong_kind_failure is not None
    assert "return_address mismatch" in wrong_kind_failure

    rows[0]["kind"] = "registry_write"
    rows[1]["kind"] = "file_write"
    _, wrong_following_kind_failure = _audited_service_payload_reentry(
        rows,
        following_payload_start,
        start_index=0,
        end_index=2,
        previous_read_bit_position=120,
        previous_command_bit_position=110,
        previous_command_order=1,
        previous_reentry_kind=1,
    )
    assert wrong_following_kind_failure is not None
    assert "not at a service row end" in wrong_following_kind_failure


def test_terminal_registry_write_eof_exit_accepts_only_final_result_bit(
) -> None:
    rows = [
        _row(
            start=100 + index * 11,
            end=111 + index * 11,
            update=42,
            command_number=12,
            kind="registry_write",
            payload={"success": True},
        )
        for index in range(3)
    ]
    continuation = {
        "base": 0x100000,
        "thread_id": 456,
        "start_index": 0,
        "end_index": 2,
        "end_bit_position": 133,
        "last_read_bit_position": 132,
        "last_command_bit_position": 121,
        "last_command_order": 1,
        "last_reentry_update": 42,
        "last_reentry_kind": 1,
        "reentries": 3,
        "allow_bounded_late_successful_file_write_corridor": False,
        "allow_bounded_late_preloading_failed_file_write_corridor": False,
    }
    kwargs = {
        "process_id": 123,
        "exited": True,
        "exit_code": 0,
        "command_order_offset": 1,
    }

    receipt, failure = _audited_terminal_registry_write_eof_exit(
        rows, continuation, **kwargs
    )
    assert failure is None
    assert receipt is not None
    assert receipt["terminal_row_index"] == 2
    assert receipt["terminal_payload_bit_position"] == 132
    assert receipt["terminal_row_end_bit_position"] == 133
    assert receipt["unconsumed_result_bits"] == 1
    assert receipt["recorded_success"] is True
    assert receipt["exit_code"] == 0

    for altered, expected in (
        (dict(continuation, last_read_bit_position=133), "boundary"),
        (dict(continuation, reentries=2), "boundary"),
        (dict(continuation, end_index=1), "maximal"),
    ):
        altered_receipt, altered_failure = (
            _audited_terminal_registry_write_eof_exit(
                rows, altered, **kwargs
            )
        )
        assert altered_receipt is None
        assert altered_failure is not None
        assert expected in altered_failure

    nonzero_receipt, nonzero_failure = (
        _audited_terminal_registry_write_eof_exit(
            rows,
            continuation,
            **dict(kwargs, exit_code=1),
        )
    )
    assert nonzero_receipt is None
    assert nonzero_failure is not None
    assert "precondition" in nonzero_failure


def test_service_payload_reentry_accepts_same_command_forced_read() -> None:
    rows = [
        _row(start=100, end=111, update=42, command_number=16),
        _row(start=111, end=122, update=43, command_number=16),
        _row(start=122, end=133, update=44, command_number=16),
    ]
    snapshot = _snapshot(
        command_bit_position=121,
        buffer_read_bit_position=133,
        command_number=0,
        command_order=2,
        needs_command=1,
    )

    assert _audited_service_payload_reentry(
        rows,
        snapshot,
        start_index=0,
        end_index=2,
        previous_read_bit_position=131,
        previous_command_bit_position=121,
        previous_command_order=2,
        previous_reentry_kind=1,
    ) == (1, None)


def test_service_payload_reentry_accepts_entry_after_prepared_row() -> None:
    rows = [
        _row(start=100, end=111, update=42, command_number=16),
        _row(start=111, end=122, update=43, command_number=16),
        _row(start=122, end=133, update=44, command_number=16),
        _row(start=133, end=144, update=45, command_number=16),
    ]
    snapshot = _snapshot(
        command_bit_position=132,
        buffer_read_bit_position=144,
        command_number=0,
        command_order=2,
        needs_command=1,
    )

    assert _audited_service_payload_reentry(
        rows,
        snapshot,
        start_index=0,
        end_index=3,
        previous_read_bit_position=132,
        previous_command_bit_position=122,
        previous_command_order=1,
        previous_reentry_kind=2,
        command_order_offset=1,
    ) == (2, None)


def test_service_prepared_reentry_accepts_exact_future_service_row() -> None:
    rows = [
        _row(start=100, end=111, update=42, command_number=16),
        _row(start=111, end=122, update=43, command_number=16),
        _row(start=122, end=133, update=44, command_number=16),
    ]
    snapshot = _snapshot(
        update=43,
        command_bit_position=122,
        buffer_read_bit_position=132,
        command_number=16,
        command_order=2,
        last_demo_update=43,
        needs_command=1,
    )

    assert _audited_service_prepared_reentry(
        rows,
        snapshot,
        start_index=0,
        end_index=2,
        previous_read_bit_position=122,
        previous_command_order=1,
        previous_command_bit_position=121,
        previous_reentry_kind=1,
    ) == (2, None)


def test_service_prepared_reentry_accepts_same_header_transition() -> None:
    rows = [
        _row(start=100, end=111, update=42, command_number=16),
        _row(start=111, end=122, update=43, command_number=16),
    ]
    snapshot = _snapshot(
        update=42,
        command_bit_position=111,
        buffer_read_bit_position=121,
        command_number=16,
        command_order=1,
        last_demo_update=42,
        needs_command=1,
    )

    assert _audited_service_prepared_reentry(
        rows,
        snapshot,
        start_index=0,
        end_index=1,
        previous_read_bit_position=121,
        previous_command_order=1,
        previous_command_bit_position=111,
        previous_reentry_kind=3,
    ) == (1, None)


def test_service_prepared_reentry_accepts_payload_timing_in_corridor() -> None:
    rows = [
        _row(start=100, end=111, update=42, command_number=16),
        _row(start=111, end=122, update=42, command_number=16),
        _row(start=122, end=133, update=43, command_number=16),
        _row(start=133, end=144, update=44, command_number=16),
    ]
    snapshot = _snapshot(
        update=44,
        command_bit_position=122,
        buffer_read_bit_position=132,
        command_number=16,
        command_order=2,
        last_demo_update=44,
        needs_command=1,
    )

    assert _audited_service_prepared_reentry(
        rows,
        snapshot,
        start_index=0,
        end_index=3,
        previous_read_bit_position=122,
        previous_command_order=1,
        previous_command_bit_position=111,
        previous_reentry_kind=1,
        require_before_recorded_update=False,
    ) == (2, None)


def test_service_prepared_reentry_accepts_bounded_late_successful_file_write_corridor() -> None:
    rows = [
        _row(
            start=100,
            end=111,
            update=42,
            command_number=16,
            kind="file_write",
            payload={"success": True},
        ),
        _row(
            start=111,
            end=122,
            update=42,
            command_number=16,
            kind="file_write",
            payload={"success": True},
        ),
    ]
    snapshot = _snapshot(
        update=44,
        last_demo_update=44,
        demo_loading_complete=1,
        command_bit_position=111,
        buffer_read_bit_position=121,
        command_number=16,
        command_order=1,
        needs_command=1,
    )

    assert _audited_service_prepared_reentry(
        rows,
        snapshot,
        start_index=0,
        end_index=1,
        previous_read_bit_position=111,
        previous_command_order=0,
        previous_command_bit_position=100,
        previous_reentry_kind=1,
        require_before_recorded_update=False,
        allow_bounded_late_successful_file_write_corridor=True,
    ) == (1, None)


def test_service_prepared_reentry_accepts_exact_bounded_late_preloading_failed_file_write_corridor() -> None:
    rows = [
        _row(
            start=100,
            end=111,
            update=42,
            command_number=16,
            kind="file_write",
            payload={"success": False},
        ),
        _row(
            start=111,
            end=122,
            update=43,
            command_number=16,
            kind="file_write",
            payload={"success": False},
        ),
    ]
    snapshot = _snapshot(
        update=44,
        last_demo_update=44,
        demo_loading_complete=0,
        command_bit_position=111,
        buffer_read_bit_position=121,
        command_number=16,
        command_order=1,
        needs_command=1,
    )

    assert _audited_service_prepared_reentry(
        rows,
        snapshot,
        start_index=0,
        end_index=1,
        previous_read_bit_position=111,
        previous_command_order=0,
        previous_command_bit_position=100,
        previous_reentry_kind=1,
        require_before_recorded_update=False,
        allow_bounded_late_preloading_failed_file_write_corridor=True,
    ) == (1, None)


@pytest.mark.parametrize(
    ("snapshot_override", "payload_success"),
    (
        ({"update": 45, "last_demo_update": 45}, False),
        ({"demo_loading_complete": 1}, False),
        ({}, True),
    ),
)
def test_service_prepared_reentry_rejects_unaudited_preloading_failed_file_write_corridor(
    snapshot_override: dict[str, int],
    payload_success: bool,
) -> None:
    rows = [
        _row(
            start=100,
            end=111,
            update=42,
            command_number=16,
            kind="file_write",
            payload={"success": payload_success},
        ),
        _row(
            start=111,
            end=122,
            update=43,
            command_number=16,
            kind="file_write",
            payload={"success": payload_success},
        ),
    ]
    snapshot_values = {
        "update": 44,
        "last_demo_update": 44,
        "demo_loading_complete": 0,
        "command_bit_position": 111,
        "buffer_read_bit_position": 121,
        "command_number": 16,
        "command_order": 1,
        "needs_command": 1,
    }
    snapshot_values.update(snapshot_override)

    _, failure = _audited_service_prepared_reentry(
        rows,
        _snapshot(**snapshot_values),
        start_index=0,
        end_index=1,
        previous_read_bit_position=111,
        previous_command_order=0,
        previous_command_bit_position=100,
        previous_reentry_kind=1,
        require_before_recorded_update=False,
        allow_bounded_late_preloading_failed_file_write_corridor=True,
    )

    assert failure is not None
    assert "outside its audited corridor" in failure


def _startup_font_cache_failed_partition_rows(
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    bit_position = 100
    for _ in range(8):
        rows.append(
            _row(
                start=bit_position,
                end=bit_position + 11,
                update=0,
                command_number=16,
                kind="file_write",
                payload={"success": False},
            )
        )
        bit_position += 11
    rows.append(
        _row(
            start=bit_position,
            end=bit_position + 10,
            update=15,
            command_number=31,
            kind="idle",
            payload={},
        )
    )
    bit_position += 10
    for index in range(27):
        rows.append(
            _row(
                start=bit_position,
                end=bit_position + 11,
                update=384 + index // 3,
                command_number=16,
                kind="file_write",
                payload={"success": False},
            )
        )
        bit_position += 11
    rows.append(
        _row(
            start=bit_position,
            end=bit_position + 10,
            update=459,
            command_number=9,
            kind="loading_complete",
            payload={},
        )
    )
    return rows


def test_startup_font_cache_failed_partition_requires_exact_35_row_split(
) -> None:
    rows = _startup_font_cache_failed_partition_rows()

    assert _is_exact_startup_font_cache_failed_write_partition(
        rows,
        start_index=9,
        end_index=35,
        direct_row_indices=tuple(range(8)),
        manifest_entry_count=35,
    )
    assert _is_bounded_late_preloading_failed_file_write_corridor(
        rows,
        _snapshot(
            update=393,
            last_demo_update=393,
            demo_loading_complete=0,
        ),
        9,
        35,
    )


@pytest.mark.parametrize(
    ("start_index", "end_index", "direct_rows", "manifest_count"),
    (
        (9, 35, tuple(range(7)), 35),
        (9, 34, tuple(range(8)), 35),
        (9, 35, tuple(range(8)), 34),
        (8, 35, tuple(range(8)), 35),
    ),
)
def test_startup_font_cache_failed_partition_rejects_incomplete_or_overlapping_split(
    start_index: int,
    end_index: int,
    direct_rows: tuple[int, ...],
    manifest_count: int,
) -> None:
    assert not _is_exact_startup_font_cache_failed_write_partition(
        _startup_font_cache_failed_partition_rows(),
        start_index=start_index,
        end_index=end_index,
        direct_row_indices=direct_rows,
        manifest_entry_count=manifest_count,
    )


@pytest.mark.parametrize(
    ("snapshot_override", "payload_success"),
    (
        ({"update": 45, "last_demo_update": 45}, True),
        ({"demo_loading_complete": 0}, True),
        ({}, False),
    ),
)
def test_service_prepared_reentry_rejects_unaudited_late_file_write_corridor(
    snapshot_override: dict[str, int],
    payload_success: bool,
) -> None:
    rows = [
        _row(
            start=100,
            end=111,
            update=42,
            command_number=16,
            kind="file_write",
            payload={"success": payload_success},
        ),
        _row(
            start=111,
            end=122,
            update=42,
            command_number=16,
            kind="file_write",
            payload={"success": payload_success},
        ),
    ]
    snapshot_values = {
        "update": 43,
        "last_demo_update": 43,
        "demo_loading_complete": 1,
        "command_bit_position": 111,
        "buffer_read_bit_position": 121,
        "command_number": 16,
        "command_order": 1,
        "needs_command": 1,
    }
    snapshot_values.update(snapshot_override)

    _, failure = _audited_service_prepared_reentry(
        rows,
        _snapshot(**snapshot_values),
        start_index=0,
        end_index=1,
        previous_read_bit_position=111,
        previous_command_order=0,
        previous_command_bit_position=100,
        previous_reentry_kind=1,
        require_before_recorded_update=False,
        allow_bounded_late_successful_file_write_corridor=True,
    )

    assert failure is not None
    assert "outside its audited corridor" in failure


def test_service_header_reentry_accepts_exact_future_service_row() -> None:
    rows = [
        _row(start=100, end=111, update=42, command_number=16),
        _row(start=111, end=122, update=43, command_number=16),
        _row(start=122, end=133, update=44, command_number=16),
        _row(start=133, end=144, update=45, command_number=16),
    ]
    snapshot = _snapshot(
        update=44,
        command_bit_position=133,
        buffer_read_bit_position=143,
        command_number=16,
        command_order=2,
        last_demo_update=44,
        needs_command=0,
    )

    assert _audited_service_header_reentry(
        rows,
        snapshot,
        start_index=0,
        end_index=3,
        previous_read_bit_position=133,
        previous_command_order=1,
        previous_reentry_kind=1,
        command_order_offset=1,
    ) == (3, None)


def test_service_header_reentry_accepts_payload_timing_in_corridor() -> None:
    rows = [
        _row(start=100, end=111, update=42, command_number=16),
        _row(start=111, end=122, update=43, command_number=16),
        _row(start=122, end=133, update=45, command_number=16),
    ]
    snapshot = _snapshot(
        update=44,
        command_bit_position=111,
        buffer_read_bit_position=121,
        command_number=16,
        command_order=1,
        last_demo_update=44,
        needs_command=0,
    )

    assert _audited_service_header_reentry(
        rows,
        snapshot,
        start_index=0,
        end_index=2,
        previous_read_bit_position=111,
        previous_command_order=0,
        previous_reentry_kind=1,
        require_before_recorded_update=False,
    ) == (1, None)


def test_service_header_reentry_accepts_bounded_late_successful_file_write_corridor() -> None:
    rows = [
        _row(
            start=100,
            end=111,
            update=42,
            command_number=16,
            kind="file_write",
            payload={"success": True},
        ),
        _row(
            start=111,
            end=122,
            update=42,
            command_number=16,
            kind="file_write",
            payload={"success": True},
        ),
    ]
    snapshot = _snapshot(
        update=43,
        last_demo_update=43,
        demo_loading_complete=1,
        command_bit_position=111,
        buffer_read_bit_position=121,
        command_number=16,
        command_order=1,
        needs_command=0,
    )

    assert _audited_service_header_reentry(
        rows,
        snapshot,
        start_index=0,
        end_index=1,
        previous_read_bit_position=111,
        previous_command_order=0,
        previous_reentry_kind=1,
        require_before_recorded_update=False,
        allow_bounded_late_successful_file_write_corridor=True,
    ) == (1, None)


def test_service_prepared_reentry_accepts_late_header_transition() -> None:
    rows = [
        _row(start=100, end=111, update=42, command_number=16),
        _row(start=111, end=122, update=43, command_number=16),
        _row(start=122, end=133, update=45, command_number=16),
    ]
    snapshot = _snapshot(
        update=44,
        command_bit_position=111,
        buffer_read_bit_position=121,
        command_number=16,
        command_order=1,
        last_demo_update=44,
        needs_command=1,
    )

    assert _audited_service_prepared_reentry(
        rows,
        snapshot,
        start_index=0,
        end_index=2,
        previous_read_bit_position=121,
        previous_command_order=1,
        previous_command_bit_position=111,
        previous_reentry_kind=3,
        require_before_recorded_update=False,
    ) == (1, None)


def test_service_header_reentry_rejects_nonpayload_predecessor() -> None:
    rows = [
        _row(start=100, end=111, update=42, command_number=16),
        _row(start=111, end=122, update=43, command_number=16),
    ]
    snapshot = _snapshot(
        update=42,
        command_bit_position=111,
        buffer_read_bit_position=121,
        command_number=16,
        command_order=1,
        last_demo_update=42,
        needs_command=0,
    )

    with pytest.raises(ValueError, match="service header re-entry boundary"):
        _audited_service_header_reentry(
            rows,
            snapshot,
            start_index=0,
            end_index=1,
            previous_read_bit_position=111,
            previous_command_order=0,
            previous_reentry_kind=2,
        )


def test_service_prepared_reentry_rejects_reached_update() -> None:
    rows = [
        _row(start=100, end=111, update=42, command_number=16),
        _row(start=111, end=122, update=43, command_number=16),
    ]
    snapshot = _snapshot(
        update=43,
        command_bit_position=111,
        buffer_read_bit_position=121,
        command_number=16,
        command_order=1,
        last_demo_update=43,
        needs_command=1,
    )

    _, failure = _audited_service_prepared_reentry(
        rows,
        snapshot,
        start_index=0,
        end_index=1,
        previous_read_bit_position=111,
        previous_command_order=0,
        previous_command_bit_position=100,
        previous_reentry_kind=1,
    )

    assert failure is not None
    assert "not before its recorded update" in failure


def test_service_command_order_rebase_accounts_exact_consumed_rows() -> None:
    rows = [
        _row(start=100, end=111, update=42, command_number=16),
        _row(start=111, end=122, update=43, command_number=16),
        _row(start=122, end=133, update=44, command_number=16),
        _row(start=133, end=144, update=45, command_number=16),
    ]
    snapshot = _snapshot(
        command_bit_position=133,
        buffer_read_bit_position=143,
        command_number=16,
        command_order=1,
        last_demo_update=45,
        needs_command=1,
    )

    assert _audited_service_command_order_rebase(
        _index(rows),
        rows,
        snapshot,
        corridor_start_index=0,
        corridor_end_index=3,
        command_order_offset=0,
        accounted_rows=(),
    ) == (2, (1, 2), None)


def test_service_command_order_rebase_accounts_trailing_rows_at_exit() -> None:
    rows = [
        _row(start=100, end=111, update=42, command_number=16),
        _row(start=111, end=122, update=43, command_number=16),
        _row(start=122, end=133, update=44, command_number=16),
        _row(start=133, end=143, update=45, command_number=31),
    ]
    snapshot = _snapshot(
        update=44,
        command_bit_position=133,
        buffer_read_bit_position=143,
        command_number=31,
        command_order=1,
        last_demo_update=44,
        needs_command=0,
    )

    assert _audited_service_command_order_rebase(
        _index(rows),
        rows,
        snapshot,
        corridor_start_index=0,
        corridor_end_index=2,
        command_order_offset=0,
        accounted_rows=(),
        require_last_demo_update=False,
        allow_exit_row=True,
    ) == (2, (1, 2), None)


def test_successful_file_write_corridor_accepts_complete_prepared_exit() -> None:
    rows = [
        _row(
            start=100,
            end=111,
            update=42,
            command_number=16,
            kind="file_write",
            payload={"success": True},
        ),
        _row(
            start=111,
            end=122,
            update=42,
            command_number=16,
            kind="file_write",
            payload={"success": True},
        ),
        _row(
            start=122,
            end=140,
            update=42,
            command_number=0,
            short_form=True,
            kind="mouse_move",
        ),
    ]
    snapshot = _snapshot(
        update=44,
        last_demo_update=44,
        demo_loading_complete=1,
        command_bit_position=122,
        buffer_read_bit_position=140,
        command_number=0,
        command_order=1,
        needs_command=1,
        is_short=1,
    )

    assert _audited_successful_file_write_corridor_prepared_exit(
        rows,
        snapshot,
        corridor_start_index=0,
        corridor_end_index=1,
        previous_read_bit_position=122,
        previous_reentry_kind=1,
        command_order_offset=1,
    ) == (2, None)


@pytest.mark.parametrize(
    ("override", "payload_success", "expected"),
    (
        ({"update": 45, "last_demo_update": 45}, True, "bounded spillover"),
        ({"needs_command": 0}, True, "needs_command mismatch"),
        ({}, False, "corridor is ineligible"),
    ),
)
def test_successful_file_write_corridor_prepared_exit_fails_closed(
    override: dict[str, int],
    payload_success: bool,
    expected: str,
) -> None:
    rows = [
        _row(
            start=100,
            end=111,
            update=42,
            command_number=16,
            kind="file_write",
            payload={"success": payload_success},
        ),
        _row(
            start=111,
            end=122,
            update=42,
            command_number=16,
            kind="file_write",
            payload={"success": payload_success},
        ),
        _row(
            start=122,
            end=140,
            update=42,
            command_number=0,
            short_form=True,
            kind="mouse_move",
        ),
    ]
    values = {
        "update": 44,
        "last_demo_update": 44,
        "demo_loading_complete": 1,
        "command_bit_position": 122,
        "buffer_read_bit_position": 140,
        "command_number": 0,
        "command_order": 1,
        "needs_command": 1,
        "is_short": 1,
    }
    values.update(override)

    _, failure = _audited_successful_file_write_corridor_prepared_exit(
        rows,
        _snapshot(**values),
        corridor_start_index=0,
        corridor_end_index=1,
        previous_read_bit_position=122,
        previous_reentry_kind=1,
        command_order_offset=1,
    )

    assert failure is not None
    assert expected in failure


def _successful_file_write_deferred_exit_rows() -> list[dict[str, object]]:
    return [
        _row(
            start=100,
            end=111,
            update=42,
            command_number=16,
            kind="file_write",
            payload={"success": True},
        ),
        _row(
            start=111,
            end=122,
            update=42,
            command_number=16,
            kind="file_write",
            payload={"success": True},
        ),
        _row(
            start=122,
            end=133,
            update=42,
            command_number=16,
            kind="file_write",
            payload={"success": True},
        ),
        _row(
            start=133,
            end=143,
            update=57,
            command_number=31,
            kind="idle",
            payload={},
        ),
    ]


def test_successful_file_write_deferred_exit_accepts_exact_idle() -> None:
    rows = _successful_file_write_deferred_exit_rows()
    snapshot = _snapshot(
        return_address=DEFERRED_FILE_WRITE_CALLER,
        update=50,
        last_demo_update=60,
        demo_loading_complete=1,
        command_bit_position=133,
        buffer_read_bit_position=143,
        command_number=31,
        command_order=2,
        needs_command=0,
    )

    assert _audited_successful_file_write_deferred_exit(
        rows,
        snapshot,
        corridor_start_index=0,
        corridor_end_index=2,
        previous_read_bit_position=133,
        previous_reentry_kind=1,
        command_order_offset=1,
    ) == (3, None)


def test_successful_file_write_deferred_exit_accepts_one_tick_late_at_exact_corridor_bound(
) -> None:
    rows = [
        _row(
            start=100 + 11 * index,
            end=111 + 11 * index,
            update=42,
            command_number=16,
            kind="file_write",
            payload={"success": True},
        )
        for index in range(16)
    ]
    rows.append(
        _row(
            start=276,
            end=286,
            update=57,
            command_number=31,
            kind="idle",
            payload={},
        )
    )
    values = {
        "return_address": DEFERRED_FILE_WRITE_CALLER,
        "update": 58,
        "last_demo_update": 61,
        "demo_loading_complete": 1,
        "command_bit_position": 276,
        "buffer_read_bit_position": 286,
        "command_number": 31,
        "command_order": 12,
        "needs_command": 0,
    }

    assert _audited_successful_file_write_deferred_exit(
        rows,
        _snapshot(**values),
        corridor_start_index=0,
        corridor_end_index=15,
        previous_read_bit_position=276,
        previous_reentry_kind=1,
        command_order_offset=4,
    ) == (16, None)

    values["update"] = 59
    _, failure = _audited_successful_file_write_deferred_exit(
        rows,
        _snapshot(**values),
        corridor_start_index=0,
        corridor_end_index=15,
        previous_read_bit_position=276,
        previous_reentry_kind=1,
        command_order_offset=4,
    )
    assert failure is not None
    assert "bounded timeline" in failure


def test_successful_file_write_deferred_exit_commit_clamps_only_exact_late_tick(
) -> None:
    rows = [
        _row(
            start=100 + 11 * index,
            end=111 + 11 * index,
            update=42,
            command_number=16,
            kind="file_write",
            payload={"success": True},
        )
        for index in range(16)
    ]
    rows.append(
        _row(
            start=276,
            end=286,
            update=57,
            command_number=31,
            kind="idle",
            payload={},
        )
    )
    values = {
        "return_address": DEFERRED_FILE_WRITE_CALLER,
        "update": 58,
        "last_demo_update": 61,
        "demo_loading_complete": 1,
        "command_bit_position": 276,
        "buffer_read_bit_position": 286,
        "command_number": 31,
        "command_order": 12,
        "needs_command": 0,
    }
    kwargs = {
        "corridor_start_index": 0,
        "corridor_end_index": 15,
        "previous_read_bit_position": 276,
        "previous_reentry_kind": 1,
        "command_order_offset": 4,
    }

    assert _audited_successful_file_write_deferred_exit_timeline_commit(
        rows, _snapshot(**values), **kwargs
    ) == (16, 58, None)

    values["update"] = 56
    assert _audited_successful_file_write_deferred_exit_timeline_commit(
        rows, _snapshot(**values), **kwargs
    ) == (16, 57, None)

    values["update"] = 59
    _, committed_update, failure = (
        _audited_successful_file_write_deferred_exit_timeline_commit(
            rows, _snapshot(**values), **kwargs
        )
    )
    assert committed_update is None
    assert failure is not None
    assert "bounded timeline" in failure


def test_successful_file_write_deferred_exit_commit_normalizes_only_exact_receipt(
) -> None:
    rows = _successful_file_write_deferred_exit_rows()
    snapshot = _snapshot(
        return_address=MAIN_DEMO_CALLER,
        update=58,
        last_demo_update=58,
        demo_loading_complete=1,
        command_bit_position=133,
        buffer_read_bit_position=143,
        command_number=31,
        command_order=3,
        needs_command=0,
        is_short=0,
    )
    receipt = {
        "process_id": 123,
        "thread_id": 456,
        "framework_update": 58,
        "row_index": 3,
        "command_bit_position": 133,
        "buffer_read_bit_position": 143,
        "recorded_update": 57,
        "effective_committed_update": 58,
        "last_demo_update_after": 58,
        "late_commit_clamp_updates": 1,
        "recorded_timeline_delta_updates": -1,
        "timeline_delta_updates": 0,
        "corridor_end_row_index": 2,
        "corridor_end_recorded_update": 42,
        "bytes_written": 4,
    }

    normalized, accepted = (
        _audited_successful_file_write_deferred_exit_committed_boundary_snapshot(
            _index(rows),
            snapshot,
            receipt,
            process_id=123,
            thread_id=456,
        )
    )
    assert accepted is True
    assert normalized is not snapshot
    assert normalized["last_demo_update"] == 57
    assert snapshot["last_demo_update"] == 58

    consumed_snapshot = dict(snapshot, needs_command=1)
    consumed_normalized, consumed_accepted = (
        _audited_successful_file_write_deferred_exit_committed_boundary_snapshot(
            _index(rows),
            consumed_snapshot,
            receipt,
            process_id=123,
            thread_id=456,
        )
    )
    assert consumed_accepted is True
    assert consumed_normalized["last_demo_update"] == 57

    altered_receipt = dict(receipt, late_commit_clamp_updates=2)
    unchanged, altered_accepted = (
        _audited_successful_file_write_deferred_exit_committed_boundary_snapshot(
            _index(rows),
            snapshot,
            altered_receipt,
            process_id=123,
            thread_id=456,
        )
    )
    assert altered_accepted is False
    assert unchanged is snapshot


def test_successful_file_write_deferred_exit_clamp_rebases_exact_suffix_once(
) -> None:
    rows = [
        _row(
            start=100,
            end=110,
            update=42,
            command_number=31,
            kind="idle",
            payload={},
        ),
        _row(
            start=110,
            end=120,
            update=57,
            command_number=31,
            kind="idle",
            payload={},
        ),
        _row(
            start=120,
            end=130,
            update=72,
            command_number=31,
            kind="idle",
            payload={},
        ),
    ]
    receipt = {
        "process_id": 123,
        "thread_id": 456,
        "framework_update": 43,
        "row_index": 0,
        "recorded_update": 42,
        "effective_committed_update": 43,
        "last_demo_update_after": 43,
        "late_commit_clamp_updates": 1,
        "recorded_timeline_delta_updates": -1,
        "timeline_delta_updates": 0,
        "bytes_written": 4,
    }
    snapshot = _snapshot(
        return_address=MAIN_DEMO_CALLER,
        update=58,
        last_demo_update=58,
        command_bit_position=110,
        buffer_read_bit_position=120,
        command_number=31,
        command_order=0,
        needs_command=0,
        is_short=0,
    )
    kwargs = {
        "process_id": 123,
        "thread_id": 456,
        "command_order_offset": 1,
    }

    assert _audited_successful_file_write_deferred_exit_clamped_suffix_rebase(
        rows, snapshot, receipt, **kwargs
    ) == (1, 1, None)
    assert [row["update"] for row in rows] == [42, 58, 73]

    _, _, repeated_failure = (
        _audited_successful_file_write_deferred_exit_clamped_suffix_rebase(
            rows, snapshot, receipt, **kwargs
        )
    )
    assert repeated_failure is not None

    fresh_rows = [dict(row) for row in rows]
    fresh_rows[1]["update"] = 57
    fresh_rows[2]["update"] = 72
    _, _, altered_snapshot_failure = (
        _audited_successful_file_write_deferred_exit_clamped_suffix_rebase(
            fresh_rows,
            dict(snapshot, update=59, last_demo_update=59),
            receipt,
            **kwargs,
        )
    )
    assert altered_snapshot_failure is not None
    assert "first boundary" in altered_snapshot_failure


@pytest.mark.parametrize(
    ("target", "field", "value", "expected"),
    (
        ("snapshot", "return_address", MAIN_DEMO_CALLER, "return_address"),
        ("snapshot", "buffer_read_bit_position", 142, "buffer_read_bit_position"),
        ("snapshot", "last_demo_update", 57, "bounded timeline"),
        ("corridor", "payload", {"success": False}, "corridor is ineligible"),
        ("exit", "kind", "mouse_position", "long-idle"),
    ),
)
def test_successful_file_write_deferred_exit_fails_closed(
    target: str,
    field: str,
    value: object,
    expected: str,
) -> None:
    rows = _successful_file_write_deferred_exit_rows()
    values = {
        "return_address": DEFERRED_FILE_WRITE_CALLER,
        "update": 50,
        "last_demo_update": 60,
        "demo_loading_complete": 1,
        "command_bit_position": 133,
        "buffer_read_bit_position": 143,
        "command_number": 31,
        "command_order": 2,
        "needs_command": 0,
    }
    if target == "snapshot":
        values[field] = value
    elif target == "corridor":
        rows[2][field] = value
    else:
        rows[3][field] = value

    _, failure = _audited_successful_file_write_deferred_exit(
        rows,
        _snapshot(**values),
        corridor_start_index=0,
        corridor_end_index=2,
        previous_read_bit_position=133,
        previous_reentry_kind=1,
        command_order_offset=1,
    )

    assert failure is not None
    assert expected in failure


def _late_idle_bridge_rows() -> list[dict[str, object]]:
    return [
        _row(
            start=100,
            end=111,
            update=42,
            command_number=16,
            kind="file_write",
            payload={"success": True},
        ),
        _row(
            start=111,
            end=122,
            update=42,
            command_number=16,
            kind="file_write",
            payload={"success": True},
        ),
        _row(
            start=122,
            end=132,
            update=57,
            command_number=31,
            kind="idle",
            payload={},
        ),
        _row(
            start=132,
            end=142,
            update=59,
            command_number=31,
            kind="idle",
            payload={},
        ),
    ]


def test_service_exit_accepts_exact_saturated_late_idle_bridge() -> None:
    rows = _late_idle_bridge_rows()
    snapshot = _snapshot(
        update=59,
        last_demo_update=59,
        command_bit_position=122,
        buffer_read_bit_position=132,
        command_number=31,
        command_order=1,
        demo_loading_complete=1,
    )

    assert _audited_service_exit_late_idle_bridge(
        rows,
        snapshot,
        corridor_start_index=0,
        corridor_end_index=1,
        previous_read_bit_position=122,
        previous_reentry_kind=1,
        command_order_offset=1,
    ) == (2, None)


@pytest.mark.parametrize(
    ("row_index", "field", "value", "expected"),
    (
        (2, "update", 56, "saturated timeline"),
        (3, "update", 60, "saturated timeline"),
        (3, "kind", "mouse_position", "adjacent long idles"),
        (1, "payload", {"success": False}, "corridor is ineligible"),
    ),
)
def test_service_exit_late_idle_bridge_fails_closed(
    row_index: int,
    field: str,
    value: object,
    expected: str,
) -> None:
    rows = _late_idle_bridge_rows()
    rows[row_index][field] = value
    snapshot = _snapshot(
        update=59,
        last_demo_update=59,
        command_bit_position=122,
        buffer_read_bit_position=132,
        command_number=31,
        command_order=1,
        demo_loading_complete=1,
    )

    _, failure = _audited_service_exit_late_idle_bridge(
        rows,
        snapshot,
        corridor_start_index=0,
        corridor_end_index=1,
        previous_read_bit_position=122,
        previous_reentry_kind=1,
        command_order_offset=1,
    )

    assert failure is not None
    assert expected in failure


def test_service_exit_accepts_exact_overdue_idle_open_interval() -> None:
    rows = _late_idle_bridge_rows()
    rows[3]["update"] = 72
    snapshot = _snapshot(
        update=59,
        last_demo_update=59,
        command_bit_position=122,
        buffer_read_bit_position=132,
        command_number=31,
        command_order=1,
        demo_loading_complete=1,
    )

    assert _audited_service_exit_overdue_idle_reentry(
        rows,
        snapshot,
        corridor_start_index=0,
        corridor_end_index=1,
        previous_read_bit_position=122,
        previous_reentry_kind=1,
        command_order_offset=1,
    ) == (2, None)


def test_service_exit_advances_exact_overdue_idle_prepared_reentry() -> None:
    rows = _late_idle_bridge_rows()
    rows[3]["update"] = 72
    snapshot = _snapshot(
        update=59,
        last_demo_update=59,
        command_bit_position=122,
        buffer_read_bit_position=132,
        command_number=31,
        command_order=1,
        needs_command=1,
        demo_loading_complete=1,
    )

    assert _audited_service_exit_overdue_idle_prepared_reentry(
        rows,
        snapshot,
        corridor_start_index=0,
        corridor_end_index=1,
        previous_read_bit_position=132,
        previous_command_bit_position=122,
        previous_command_order=1,
        previous_update=59,
        previous_reentry_kind=10,
        command_order_offset=1,
    ) == (2, None)


@pytest.mark.parametrize(
    ("snapshot_field", "snapshot_value", "previous_update", "expected"),
    (
        ("needs_command", 0, 59, "requires a pending command"),
        ("buffer_read_bit_position", 133, 59, "first-stage boundary"),
        ("command_order", 2, 59, "first-stage boundary"),
        ("update", 60, 59, "first-stage boundary"),
    ),
)
def test_service_exit_overdue_idle_prepared_reentry_fails_closed(
    snapshot_field: str,
    snapshot_value: int,
    previous_update: int,
    expected: str,
) -> None:
    rows = _late_idle_bridge_rows()
    rows[3]["update"] = 72
    snapshot = _snapshot(
        update=59,
        last_demo_update=59,
        command_bit_position=122,
        buffer_read_bit_position=132,
        command_number=31,
        command_order=1,
        needs_command=1,
        demo_loading_complete=1,
    )
    snapshot[snapshot_field] = snapshot_value

    _, failure = _audited_service_exit_overdue_idle_prepared_reentry(
        rows,
        snapshot,
        corridor_start_index=0,
        corridor_end_index=1,
        previous_read_bit_position=132,
        previous_command_bit_position=122,
        previous_command_order=1,
        previous_update=previous_update,
        previous_reentry_kind=10,
        command_order_offset=1,
    )

    assert failure is not None
    assert expected in failure


@pytest.mark.parametrize(
    ("row_index", "field", "value", "live_update", "expected"),
    (
        (2, "update", 56, 59, "open saturated timeline"),
        (3, "update", 71, 59, "open saturated timeline"),
        (3, "kind", "mouse_position", 59, "adjacent long idles"),
        (2, "payload", {"unexpected": 1}, 59, "adjacent long idles"),
        (1, "payload", {"success": False}, 59, "corridor is ineligible"),
        (3, "update", 72, 57, "open saturated timeline"),
        (3, "update", 72, 72, "open saturated timeline"),
    ),
)
def test_service_exit_overdue_idle_reentry_fails_closed(
    row_index: int,
    field: str,
    value: object,
    live_update: int,
    expected: str,
) -> None:
    rows = _late_idle_bridge_rows()
    rows[3]["update"] = 72
    rows[row_index][field] = value
    snapshot = _snapshot(
        update=live_update,
        last_demo_update=live_update,
        command_bit_position=122,
        buffer_read_bit_position=132,
        command_number=31,
        command_order=1,
        demo_loading_complete=1,
    )

    _, failure = _audited_service_exit_overdue_idle_reentry(
        rows,
        snapshot,
        corridor_start_index=0,
        corridor_end_index=1,
        previous_read_bit_position=122,
        previous_reentry_kind=1,
        command_order_offset=1,
    )

    assert failure is not None
    assert expected in failure


def test_service_exit_advances_verified_late_idle_bridge() -> None:
    rows = _late_idle_bridge_rows()
    snapshot = _snapshot(
        update=59,
        last_demo_update=59,
        command_bit_position=122,
        buffer_read_bit_position=132,
        command_number=31,
        command_order=1,
        needs_command=1,
        demo_loading_complete=1,
    )

    assert _audited_service_exit_late_idle_bridge_prepared_reentry(
        rows,
        snapshot,
        corridor_start_index=0,
        corridor_end_index=1,
        previous_read_bit_position=132,
        previous_command_bit_position=122,
        previous_command_order=1,
        previous_update=59,
        previous_reentry_kind=8,
        command_order_offset=1,
    ) == (2, None)


def test_service_exit_late_idle_bridge_advance_binds_first_stage() -> None:
    rows = _late_idle_bridge_rows()
    snapshot = _snapshot(
        update=59,
        last_demo_update=59,
        command_bit_position=122,
        buffer_read_bit_position=132,
        command_number=31,
        command_order=1,
        needs_command=1,
        demo_loading_complete=1,
    )

    _, failure = _audited_service_exit_late_idle_bridge_prepared_reentry(
        rows,
        snapshot,
        corridor_start_index=0,
        corridor_end_index=1,
        previous_read_bit_position=132,
        previous_command_bit_position=122,
        previous_command_order=1,
        previous_update=58,
        previous_reentry_kind=8,
        command_order_offset=1,
    )

    assert failure is not None
    assert "first-stage boundary" in failure


def test_late_idle_native_timeline_rebase_shifts_only_suffix() -> None:
    rows = _late_idle_bridge_rows()
    rows.append(
        _row(
            start=142,
            end=152,
            update=63,
            command_number=31,
            kind="idle",
            payload={},
        )
    )
    original_updates = [int(row["update"]) for row in rows]

    assert _audited_late_idle_native_timeline_rebase(
        rows,
        corridor_start_index=0,
        corridor_end_index=1,
        bridge_index=2,
        live_bridge_update=59,
    ) == (3, 2, None)
    assert [int(row["update"]) for row in rows] == [
        *original_updates[:3],
        61,
        65,
    ]


@pytest.mark.parametrize(
    ("mutation", "expected"),
    (
        ((1, "payload", {"success": False}), "corridor is ineligible"),
        ((3, "kind", "mouse_position"), "adjacent long idles"),
        ((3, "update", 60), "saturated timeline"),
        ((3, "update", 56), "saturated timeline"),
    ),
)
def test_late_idle_native_timeline_rebase_fails_without_mutation(
    mutation: tuple[int, str, object],
    expected: str,
) -> None:
    rows = _late_idle_bridge_rows()
    row_index, field, value = mutation
    rows[row_index][field] = value
    updates_before = [int(row["update"]) for row in rows]

    _, _, failure = _audited_late_idle_native_timeline_rebase(
        rows,
        corridor_start_index=0,
        corridor_end_index=1,
        bridge_index=2,
        live_bridge_update=59,
    )

    assert failure is not None
    assert expected in failure
    assert [int(row["update"]) for row in rows] == updates_before


def _successful_file_write_tail_prefetch_rows() -> list[dict[str, object]]:
    rows = [
        _row(
            start=100 + 11 * index,
            end=111 + 11 * index,
            update=42,
            command_number=16,
            kind="file_write",
            payload={"success": True},
        )
        for index in range(5)
    ]
    rows.extend(
        [
            _row(
                start=155,
                end=165,
                update=57,
                command_number=31,
                kind="idle",
                payload={},
            ),
            _row(
                start=165,
                end=175,
                update=72,
                command_number=31,
                kind="idle",
                payload={},
            ),
        ]
    )
    return rows


def test_successful_file_write_tail_prefetch_accepts_exact_24_bits() -> None:
    rows = _successful_file_write_tail_prefetch_rows()
    snapshot = _snapshot(
        update=46,
        last_demo_update=46,
        command_bit_position=132,
        buffer_read_bit_position=166,
        command_number=0,
        command_order=3,
        needs_command=1,
    )

    assert _audited_successful_file_write_tail_prefetch(
        rows,
        snapshot,
        corridor_start_index=0,
        corridor_end_index=4,
        previous_read_bit_position=142,
        previous_command_bit_position=132,
        previous_command_order=3,
        previous_update=46,
        previous_reentry_kind=1,
        command_order_offset=0,
    ) == (6, None)


@pytest.mark.parametrize(
    ("target", "field", "value", "expected"),
    (
        ("snapshot", "buffer_read_bit_position", 165, "buffer_read_bit_position"),
        ("snapshot", "update", 47, "update"),
        ("row", "kind", "mouse_position", "inert long idles"),
        ("row", "update", 46, "due idle timeline"),
        ("corridor", "payload", {"success": False}, "corridor is ineligible"),
    ),
)
def test_successful_file_write_tail_prefetch_fails_closed(
    target: str,
    field: str,
    value: object,
    expected: str,
) -> None:
    rows = _successful_file_write_tail_prefetch_rows()
    values = {
        "update": 46,
        "last_demo_update": 46,
        "command_bit_position": 132,
        "buffer_read_bit_position": 166,
        "command_number": 0,
        "command_order": 3,
        "needs_command": 1,
    }
    if target == "snapshot":
        values[field] = value
    elif target == "row":
        rows[5][field] = value
    else:
        rows[4][field] = value
    snapshot = _snapshot(**values)

    _, failure = _audited_successful_file_write_tail_prefetch(
        rows,
        snapshot,
        corridor_start_index=0,
        corridor_end_index=4,
        previous_read_bit_position=142,
        previous_command_bit_position=132,
        previous_command_order=3,
        previous_update=46,
        previous_reentry_kind=1,
        command_order_offset=0,
    )

    assert failure is not None
    assert expected in failure


def _preloading_failed_file_write_tail_prefetch_rows(
) -> list[dict[str, object]]:
    updates = (39, 40, 42, 43, 43)
    rows = [
        _row(
            start=100 + 11 * index,
            end=111 + 11 * index,
            update=updates[index],
            command_number=16,
            kind="file_write",
            payload={"success": False},
        )
        for index in range(5)
    ]
    rows.extend(
        [
            _row(
                start=155,
                end=165,
                update=58,
                command_number=31,
                kind="idle",
                payload={},
            ),
            _row(
                start=165,
                end=175,
                update=73,
                command_number=31,
                kind="idle",
                payload={},
            ),
        ]
    )
    return rows


def test_preloading_failed_file_write_tail_prefetch_accepts_exact_24_bits(
) -> None:
    rows = _preloading_failed_file_write_tail_prefetch_rows()
    snapshot = _snapshot(
        update=41,
        last_demo_update=41,
        command_bit_position=132,
        buffer_read_bit_position=166,
        command_number=0,
        command_order=3,
        needs_command=1,
        demo_loading_complete=0,
    )

    assert _audited_preloading_failed_file_write_tail_prefetch(
        rows,
        snapshot,
        corridor_start_index=0,
        corridor_end_index=4,
        previous_read_bit_position=142,
        previous_command_bit_position=132,
        previous_command_order=3,
        previous_update=41,
        previous_reentry_kind=1,
        command_order_offset=0,
    ) == (6, None)


def test_preloading_failed_file_write_tail_prefetch_accepts_exact_current_header(
) -> None:
    rows = _preloading_failed_file_write_tail_prefetch_rows()
    snapshot = _snapshot(
        update=42,
        last_demo_update=42,
        command_bit_position=132,
        buffer_read_bit_position=166,
        command_number=0,
        command_order=3,
        needs_command=1,
        demo_loading_complete=0,
    )

    assert _audited_preloading_failed_file_write_tail_prefetch(
        rows,
        snapshot,
        corridor_start_index=0,
        corridor_end_index=4,
        previous_read_bit_position=132,
        previous_command_bit_position=122,
        previous_command_order=2,
        previous_update=42,
        previous_reentry_kind=2,
        command_order_offset=0,
    ) == (6, None)


def test_preloading_failed_file_write_tail_prefetch_accepts_exact_rebased_late_payload(
) -> None:
    """Mirror the live formal PC-008 tail after rows 1 and 2 were rebased."""

    rows = _preloading_failed_file_write_tail_prefetch_rows()
    rows[3]["update"] = 42
    snapshot = _snapshot(
        update=42,
        last_demo_update=42,
        command_bit_position=143,
        buffer_read_bit_position=177,
        command_number=0,
        command_order=2,
        needs_command=1,
        demo_loading_complete=0,
    )

    assert _audited_preloading_failed_file_write_tail_prefetch(
        rows,
        snapshot,
        corridor_start_index=0,
        corridor_end_index=4,
        previous_read_bit_position=153,
        previous_command_bit_position=143,
        previous_command_order=2,
        previous_update=42,
        previous_reentry_kind=1,
        command_order_offset=2,
        command_order_rebase_rows=(1, 2),
    ) == (6, None)


def _live_pc008_rebased_late_payload_rows() -> list[dict[str, object]]:
    """Reproduce the immutable formal PC-008 row 45 through 73 corridor."""

    rows = [
        _row(
            start=0,
            end=10,
            update=0,
            command_number=31,
            kind="idle",
            payload={},
        )
        for _ in range(45)
    ]
    updates = (
        379,
        379,
        380,
        380,
        381,
        381,
        382,
        382,
        383,
        383,
        384,
        384,
        384,
        385,
        385,
        386,
        387,
        387,
        387,
        388,
        388,
        389,
        389,
        390,
        390,
        390,
        391,
    )
    rows.extend(
        _row(
            start=1563 + 11 * offset,
            end=1574 + 11 * offset,
            update=update,
            command_number=16,
            kind="file_write",
            payload={"success": False},
        )
        for offset, update in enumerate(updates)
    )
    rows.extend(
        (
            _row(
                start=1860,
                end=1870,
                update=406,
                command_number=31,
                kind="idle",
                payload={},
            ),
            _row(
                start=1870,
                end=1880,
                update=421,
                command_number=31,
                kind="idle",
                payload={},
            ),
            _row(
                start=1880,
                end=1890,
                update=436,
                command_number=31,
                kind="idle",
                payload={},
            ),
            _row(
                start=1890,
                end=1900,
                update=450,
                command_number=9,
                kind="loading_complete",
                payload={},
            ),
        )
    )
    return rows


def test_preloading_failed_file_write_tail_prefetch_accepts_live_pc008_values(
) -> None:
    rows = _live_pc008_rebased_late_payload_rows()
    snapshot = _snapshot(
        update=390,
        last_demo_update=390,
        command_bit_position=1848,
        buffer_read_bit_position=1882,
        command_number=0,
        command_order=67,
        needs_command=1,
        demo_loading_complete=0,
    )

    assert _audited_preloading_failed_file_write_tail_prefetch(
        rows,
        snapshot,
        corridor_start_index=45,
        corridor_end_index=71,
        previous_read_bit_position=1858,
        previous_command_bit_position=1848,
        previous_command_order=67,
        previous_update=390,
        previous_reentry_kind=1,
        command_order_offset=4,
        command_order_rebase_rows=(64, 65, 68, 69),
    ) == (73, None)


def test_preloading_failed_file_write_tail_prefetch_rejects_live_pc008_wrong_registry(
) -> None:
    rows = _live_pc008_rebased_late_payload_rows()
    snapshot = _snapshot(
        update=390,
        last_demo_update=390,
        command_bit_position=1848,
        buffer_read_bit_position=1882,
        command_number=0,
        command_order=67,
        needs_command=1,
        demo_loading_complete=0,
    )

    _, failure = _audited_preloading_failed_file_write_tail_prefetch(
        rows,
        snapshot,
        corridor_start_index=45,
        corridor_end_index=71,
        previous_read_bit_position=1858,
        previous_command_bit_position=1848,
        previous_command_order=67,
        previous_update=390,
        previous_reentry_kind=1,
        command_order_offset=4,
        command_order_rebase_rows=(64, 65, 67, 69),
    )

    assert failure is not None
    assert "changed its exact prepared state" in failure


def test_preloading_failed_file_write_tail_short_header_recovers_live_pc007(
) -> None:
    rows = _live_pc008_rebased_late_payload_rows()
    snapshot = _snapshot(
        update=401,
        last_demo_update=401,
        command_bit_position=1882,
        buffer_read_bit_position=1888,
        command_number=1,
        command_order=68,
        needs_command=0,
        is_short=1,
        demo_loading_complete=0,
    )

    assert (
        _audited_preloading_failed_file_write_tail_short_header_recovery(
            rows,
            snapshot,
            corridor_start_index=45,
            corridor_end_index=71,
            previous_read_bit_position=1882,
            previous_command_bit_position=1848,
            previous_command_order=67,
            previous_update=390,
            previous_reentry_kind=11,
            command_order_offset=4,
            command_order_rebase_rows=(64, 65, 68, 69),
        )
        == (
            74,
            {
                "buffer_read_bit_position": 1860,
                "last_demo_update": 391,
                "needs_command": 1,
                "is_short": 0,
                "command_number": 16,
                "command_order": 67,
                "command_bit_position": 1849,
                "demo_loading_complete": 0,
            },
            None,
        )
    )


@pytest.mark.parametrize(
    ("target", "field", "value", "expected"),
    (
        ("snapshot", "buffer_read_bit_position", 1889, "buffer_read_bit_position"),
        ("snapshot", "needs_command", 1, "needs_command"),
        ("snapshot", "command_number", 0, "command_number"),
        ("snapshot", "update", 402, "update"),
        ("third_idle", "kind", "mouse_position", "suffix mismatch"),
        ("loading_complete", "update", 451, "suffix mismatch"),
    ),
)
def test_preloading_failed_file_write_tail_short_header_recovery_fails_closed(
    target: str,
    field: str,
    value: object,
    expected: str,
) -> None:
    rows = _live_pc008_rebased_late_payload_rows()
    values = {
        "update": 401,
        "last_demo_update": 401,
        "command_bit_position": 1882,
        "buffer_read_bit_position": 1888,
        "command_number": 1,
        "command_order": 68,
        "needs_command": 0,
        "is_short": 1,
        "demo_loading_complete": 0,
    }
    if target == "snapshot":
        values[field] = value
    elif target == "third_idle":
        rows[74][field] = value
    else:
        rows[75][field] = value

    _, recovery, failure = (
        _audited_preloading_failed_file_write_tail_short_header_recovery(
            rows,
            _snapshot(**values),
            corridor_start_index=45,
            corridor_end_index=71,
            previous_read_bit_position=1882,
            previous_command_bit_position=1848,
            previous_command_order=67,
            previous_update=390,
            previous_reentry_kind=11,
            command_order_offset=4,
            command_order_rebase_rows=(64, 65, 68, 69),
        )
    )

    assert recovery is None
    assert failure is not None
    assert expected in failure


def test_preloading_failed_file_write_tail_prefetch_rejects_unbound_late_payload(
) -> None:
    rows = _preloading_failed_file_write_tail_prefetch_rows()
    rows[3]["update"] = 42
    snapshot = _snapshot(
        update=42,
        last_demo_update=42,
        command_bit_position=143,
        buffer_read_bit_position=177,
        command_number=0,
        command_order=2,
        needs_command=1,
        demo_loading_complete=0,
    )

    _, failure = _audited_preloading_failed_file_write_tail_prefetch(
        rows,
        snapshot,
        corridor_start_index=0,
        corridor_end_index=4,
        previous_read_bit_position=153,
        previous_command_bit_position=143,
        previous_command_order=2,
        previous_update=42,
        previous_reentry_kind=1,
        command_order_offset=2,
        command_order_rebase_rows=(0, 1),
    )

    assert failure is not None
    assert "changed its exact prepared state" in failure


@pytest.mark.parametrize(
    ("target", "field", "value", "expected"),
    (
        ("snapshot", "buffer_read_bit_position", 165, "buffer_read_bit_position"),
        ("snapshot", "demo_loading_complete", 1, "demo_loading_complete"),
        ("row", "kind", "mouse_position", "inert long idles"),
        ("row", "update", 57, "pre-loading timeline"),
        ("corridor", "payload", {"success": True}, "corridor is ineligible"),
    ),
)
def test_preloading_failed_file_write_tail_prefetch_fails_closed(
    target: str,
    field: str,
    value: object,
    expected: str,
) -> None:
    rows = _preloading_failed_file_write_tail_prefetch_rows()
    values = {
        "update": 41,
        "last_demo_update": 41,
        "command_bit_position": 132,
        "buffer_read_bit_position": 166,
        "command_number": 0,
        "command_order": 3,
        "needs_command": 1,
        "demo_loading_complete": 0,
    }
    if target == "snapshot":
        values[field] = value
    elif target == "row":
        rows[5][field] = value
    else:
        rows[4][field] = value
    snapshot = _snapshot(**values)

    _, failure = _audited_preloading_failed_file_write_tail_prefetch(
        rows,
        snapshot,
        corridor_start_index=0,
        corridor_end_index=4,
        previous_read_bit_position=142,
        previous_command_bit_position=132,
        previous_command_order=3,
        previous_update=41,
        previous_reentry_kind=1,
        command_order_offset=0,
    )

    assert failure is not None
    assert expected in failure


def test_service_exit_reentry_accepts_early_exact_nonservice_boundary() -> None:
    rows = [
        _row(start=100, end=111, update=42, command_number=16),
        _row(start=111, end=122, update=43, command_number=16),
        _row(start=122, end=132, update=50, command_number=31),
    ]
    snapshot = _snapshot(
        update=49,
        command_bit_position=122,
        buffer_read_bit_position=132,
        command_number=31,
        command_order=1,
        last_demo_update=49,
        needs_command=0,
    )

    assert _audited_service_exit_reentry(
        rows,
        snapshot,
        corridor_end_index=1,
        previous_read_bit_position=122,
        previous_reentry_kind=1,
        command_order_offset=1,
    ) == (2, None)


def test_service_exit_reentry_accepts_bounded_multi_tick_gap() -> None:
    rows = [
        _row(start=100, end=111, update=42, command_number=16),
        _row(start=111, end=122, update=43, command_number=16),
        _row(start=122, end=132, update=50, command_number=31),
    ]
    snapshot = _snapshot(
        update=48,
        command_bit_position=122,
        buffer_read_bit_position=132,
        command_number=31,
        command_order=1,
        last_demo_update=48,
        needs_command=0,
    )

    assert _audited_service_exit_reentry(
        rows,
        snapshot,
        corridor_end_index=1,
        previous_read_bit_position=122,
        previous_reentry_kind=1,
        command_order_offset=1,
    ) == (2, None)


def test_service_exit_reentry_rejects_gap_over_fifteen_updates() -> None:
    rows = [
        _row(start=100, end=111, update=42, command_number=16),
        _row(start=111, end=122, update=43, command_number=16),
        _row(start=122, end=132, update=100, command_number=31),
    ]
    snapshot = _snapshot(
        update=80,
        command_bit_position=122,
        buffer_read_bit_position=132,
        command_number=31,
        command_order=1,
        last_demo_update=80,
        needs_command=0,
    )

    _, failure = _audited_service_exit_reentry(
        rows,
        snapshot,
        corridor_end_index=1,
        previous_read_bit_position=122,
        previous_reentry_kind=1,
        command_order_offset=1,
    )

    assert failure is not None
    assert "bounded post-corridor timeline gap" in failure


def test_service_exit_prepared_reentry_accepts_same_exact_header() -> None:
    rows = [
        _row(start=100, end=111, update=42, command_number=16),
        _row(start=111, end=121, update=50, command_number=31),
    ]
    snapshot = _snapshot(
        update=49,
        command_bit_position=111,
        buffer_read_bit_position=121,
        command_number=31,
        command_order=0,
        last_demo_update=49,
        needs_command=1,
    )

    assert _audited_service_exit_prepared_reentry(
        rows,
        snapshot,
        corridor_end_index=0,
        previous_read_bit_position=121,
        previous_command_bit_position=111,
        previous_command_order=0,
        previous_update=49,
        previous_reentry_kind=4,
        command_order_offset=1,
    ) == (1, None)


def test_service_exit_prepared_reentry_accepts_bounded_multi_tick_gap() -> None:
    rows = [
        _row(start=100, end=111, update=42, command_number=16),
        _row(start=111, end=121, update=50, command_number=31),
    ]
    snapshot = _snapshot(
        update=48,
        command_bit_position=111,
        buffer_read_bit_position=121,
        command_number=31,
        command_order=0,
        last_demo_update=48,
        needs_command=1,
    )

    assert _audited_service_exit_prepared_reentry(
        rows,
        snapshot,
        corridor_end_index=0,
        previous_read_bit_position=121,
        previous_command_bit_position=111,
        previous_command_order=0,
        previous_update=48,
        previous_reentry_kind=4,
        command_order_offset=1,
    ) == (1, None)


def test_service_exit_partial_header_accepts_exact_split_state() -> None:
    rows = [
        _row(start=100, end=111, update=42, command_number=16),
        _row(
            start=111,
            end=121,
            update=50,
            command_number=31,
            kind="idle",
        ),
    ]
    snapshot = _snapshot(
        update=49,
        command_bit_position=110,
        buffer_read_bit_position=116,
        command_number=0,
        command_order=1,
        is_short=1,
        last_demo_update=49,
        needs_command=0,
    )

    assert _audited_service_exit_partial_header(
        rows,
        snapshot,
        corridor_end_index=0,
        previous_read_bit_position=110,
        previous_command_bit_position=100,
        previous_command_order=0,
        previous_reentry_kind=2,
        command_order_offset=0,
    ) == (1, None)


def test_service_exit_forced_read_accepts_fixed_idle_suffix_step() -> None:
    rows = [
        _row(start=100, end=111, update=42, command_number=16),
        _row(
            start=111,
            end=121,
            update=50,
            command_number=31,
            kind="idle",
        ),
        _row(
            start=121,
            end=131,
            update=60,
            command_number=31,
            kind="idle",
        ),
    ]
    snapshot = _snapshot(
        update=49,
        command_bit_position=110,
        buffer_read_bit_position=128,
        command_number=0,
        command_order=1,
        is_short=1,
        last_demo_update=49,
        needs_command=1,
    )

    assert _audited_service_exit_forced_read(
        rows,
        snapshot,
        corridor_end_index=0,
        previous_read_bit_position=116,
        previous_command_bit_position=110,
        previous_command_order=1,
        previous_update=49,
        previous_reentry_kind=5,
        command_order_offset=0,
    ) == (2, None)


def test_service_command_order_rebase_can_bind_a_prepared_row() -> None:
    rows = [
        _row(start=100, end=111, update=42, command_number=16),
        _row(start=111, end=122, update=43, command_number=16),
        _row(start=122, end=133, update=44, command_number=16),
        _row(start=133, end=144, update=45, command_number=16),
    ]
    snapshot = _snapshot(
        command_bit_position=133,
        buffer_read_bit_position=143,
        command_number=16,
        command_order=1,
        last_demo_update=44,
        needs_command=1,
    )

    assert _audited_service_command_order_rebase(
        _index(rows),
        rows,
        snapshot,
        corridor_start_index=0,
        corridor_end_index=3,
        command_order_offset=0,
        accounted_rows=(),
        require_last_demo_update=False,
    ) == (2, (1, 2), None)


def test_service_command_order_rebase_rejects_nonservice_gap() -> None:
    rows = [
        _row(start=100, end=111, update=42, command_number=16),
        _row(start=111, end=122, update=43, command_number=0),
        _row(start=122, end=133, update=44, command_number=16),
    ]
    snapshot = _snapshot(
        command_bit_position=122,
        buffer_read_bit_position=132,
        command_number=16,
        command_order=0,
        last_demo_update=44,
        needs_command=1,
    )

    _, _, failure = _audited_service_command_order_rebase(
        _index(rows),
        rows,
        snapshot,
        corridor_start_index=0,
        corridor_end_index=2,
        command_order_offset=0,
        accounted_rows=(),
    )

    assert failure is not None
    assert "ineligible row" in failure


@pytest.mark.parametrize(
    ("override", "value", "expected"),
    (
        ("command_bit_position", 120, "not an exact long-form header"),
        ("buffer_read_bit_position", 132, "header read mismatch"),
        ("command_order", 1, "command_order"),
        ("needs_command", 1, "forced read"),
    ),
)
def test_service_payload_reentry_fails_closed(
    override: str,
    value: int,
    expected: str,
) -> None:
    rows = [
        _row(start=100, end=111, update=42, command_number=16),
        _row(start=111, end=122, update=43, command_number=16),
        _row(start=122, end=133, update=44, command_number=16),
    ]
    values = {
        "command_bit_position": 121,
        "buffer_read_bit_position": 131,
        "command_number": 0,
        "command_order": 2,
        "needs_command": 0,
    }
    values[override] = value
    snapshot = _snapshot(**values)

    _, failure = _audited_service_payload_reentry(
        rows,
        snapshot,
        start_index=0,
        end_index=2,
        previous_read_bit_position=120,
    )

    assert failure is not None
    assert expected in failure


def test_snapshot_boundary_accepts_an_exact_live_entry() -> None:
    rows = [_row(start=100, end=120, update=42, command_number=11)]

    assert _snapshot_boundary_failure(_index(rows), _snapshot()) is None


def test_attach_stabilization_accepts_only_a_complete_main_boundary() -> None:
    rows = [_row(start=100, end=120, update=42, command_number=11)]

    assert (
        _attach_stabilization_boundary_failure(_index(rows), _snapshot())
        is None
    )


@pytest.mark.parametrize(
    "overrides,expected",
    (
        ({"return_address": 0x00680FB2}, "caller mismatch"),
        ({"needs_command": 1}, "requires a complete command"),
        ({"command_bit_position": 999}, "absent from the offline DMO"),
        ({"buffer_read_bit_position": 110}, "not fully consumed"),
        ({"update": 43}, "framework update mismatch"),
    ),
)
def test_attach_stabilization_rejects_inflight_or_inexact_state(
    overrides: dict[str, int],
    expected: str,
) -> None:
    rows = [_row(start=100, end=120, update=42, command_number=11)]

    failure = _attach_stabilization_boundary_failure(
        _index(rows),
        _snapshot(**overrides),
    )

    assert failure is not None
    assert expected in failure


def test_snapshot_boundary_accepts_one_fixed_command_order_offset() -> None:
    rows = [
        _row(start=80, end=100, update=41, command_number=0),
        _row(start=100, end=120, update=42, command_number=11),
    ]

    assert (
        _snapshot_boundary_failure(
            _index(rows),
            _snapshot(command_order=0),
            command_order_offset=1,
        )
        is None
    )


def test_snapshot_boundary_rejects_a_negative_order_offset() -> None:
    rows = [_row(start=100, end=120, update=42, command_number=11)]

    with pytest.raises(ValueError):
        _snapshot_boundary_failure(
            _index(rows),
            _snapshot(),
            command_order_offset=-1,
        )


def _rebase_rows() -> list[dict[str, object]]:
    return [
        _row(start=80, end=100, update=42, command_number=0),
        _row(
            start=100,
            end=111,
            update=50,
            command_number=16,
            kind="file_write",
            payload={"success": True},
        ),
        _row(start=111, end=130, update=55, command_number=0),
        _row(
            start=130,
            end=141,
            update=60,
            command_number=16,
            kind="file_write",
            payload={"success": False},
        ),
        _row(start=141, end=160, update=70, command_number=11),
        _row(start=160, end=180, update=71, command_number=11),
    ]


def test_file_write_order_rebase_accounts_for_every_missing_entry() -> None:
    rows = _rebase_rows()
    snapshot = _snapshot(
        command_bit_position=141,
        command_order=2,
        last_demo_update=70,
    )

    offset, accounted, failure = (
        _audited_file_write_command_order_rebase(
            _index(rows),
            rows,
            snapshot,
            detached_after_update=42,
        )
    )

    assert (offset, accounted, failure) == (2, (1, 3), None)
    assert (
        _snapshot_boundary_failure(
            _index(rows),
            snapshot,
            command_order_offset=offset,
        )
        is None
    )
    assert (
        _snapshot_boundary_failure(
            _index(rows),
            _snapshot(
                command_bit_position=160,
                command_order=3,
                last_demo_update=71,
            ),
            command_order_offset=offset,
        )
        is None
    )


def test_file_write_order_rebase_fails_if_offset_is_not_fully_explained() -> None:
    rows = _rebase_rows()

    offset, accounted, failure = (
        _audited_file_write_command_order_rebase(
            _index(rows),
            rows,
            _snapshot(
                command_bit_position=141,
                command_order=1,
                last_demo_update=70,
            ),
            detached_after_update=42,
        )
    )

    assert offset == 0
    assert accounted == (1, 3)
    assert failure is not None
    assert "offset=3, eligible=2" in failure


def test_file_write_order_rebase_rejects_nonexact_result_rows() -> None:
    rows = _rebase_rows()
    rows[3]["short_form"] = True

    offset, accounted, failure = (
        _audited_file_write_command_order_rebase(
            _index(rows),
            rows,
            _snapshot(
                command_bit_position=141,
                command_order=2,
                last_demo_update=70,
            ),
            detached_after_update=42,
        )
    )

    assert offset == 0
    assert accounted == (1,)
    assert failure is not None
    assert "offset=2, eligible=1" in failure


def test_post_blackout_stabilization_commits_only_a_complete_main_boundary(
) -> None:
    rows = _rebase_rows()
    snapshot = _snapshot(
        command_bit_position=141,
        buffer_read_bit_position=160,
        command_number=11,
        command_order=2,
        last_demo_update=70,
        update=70,
    )

    assert _audited_post_blackout_attach_stabilization(
        _index(rows),
        rows,
        snapshot,
        detached_after_update=42,
    ) == (2, (1, 3), None)


def test_post_blackout_stabilization_does_not_commit_an_inflight_boundary(
) -> None:
    rows = _rebase_rows()
    snapshot = _snapshot(
        command_bit_position=141,
        buffer_read_bit_position=150,
        command_number=11,
        command_order=2,
        last_demo_update=70,
        update=70,
    )

    offset, accounted, failure = (
        _audited_post_blackout_attach_stabilization(
            _index(rows),
            rows,
            snapshot,
            detached_after_update=42,
        )
    )

    assert offset == 0
    assert accounted == (1, 3)
    assert failure is not None
    assert "not fully consumed" in failure


def test_post_blackout_offset_envelope_binds_candidates_and_fixed_offsets(
) -> None:
    rows = _rebase_rows()
    snapshot = _snapshot(
        command_bit_position=141,
        buffer_read_bit_position=160,
        command_number=11,
        command_order=3,
        last_demo_update=72,
        update=72,
    )

    assert _audited_post_blackout_file_write_offset_envelope(
        _index(rows),
        rows,
        snapshot,
        detached_after_update=42,
        expected_command_order_offset=1,
        expected_native_timeline_offset=2,
    ) == (1, (1, 3), 2, None)


@pytest.mark.parametrize(
    ("field", "value", "failure_fragment"),
    [
        ("command_order", 2, "order=2/1"),
        ("last_demo_update", 71, "last_timeline=1/2"),
        ("update", 73, "update_timeline=3/2"),
    ],
)
def test_post_blackout_offset_envelope_rejects_unregistered_offsets(
    field: str,
    value: int,
    failure_fragment: str,
) -> None:
    rows = _rebase_rows()
    snapshot = _snapshot(
        command_bit_position=141,
        buffer_read_bit_position=160,
        command_number=11,
        command_order=3,
        last_demo_update=72,
        update=72,
    )
    snapshot[field] = value

    offset, candidates, timeline_offset, failure = (
        _audited_post_blackout_file_write_offset_envelope(
            _index(rows),
            rows,
            snapshot,
            detached_after_update=42,
            expected_command_order_offset=1,
            expected_native_timeline_offset=2,
        )
    )

    assert offset == 0
    assert candidates == (1, 3)
    assert timeline_offset == 0
    assert failure is not None
    assert failure_fragment in failure


@pytest.mark.parametrize(
    ("command_order", "last_update", "expected"),
    [
        (4, 70, (0, (1, 3), 0, None)),
        (3, 72, (1, (1, 3), 2, None)),
    ],
)
def test_post_blackout_offset_envelope_set_accepts_only_frozen_members(
    command_order: int,
    last_update: int,
    expected: tuple[int, tuple[int, ...], int, str | None],
) -> None:
    rows = _rebase_rows()
    snapshot = _snapshot(
        command_bit_position=141,
        buffer_read_bit_position=160,
        command_number=11,
        command_order=command_order,
        last_demo_update=last_update,
        update=last_update,
    )

    assert _audited_post_blackout_file_write_offset_envelope_set(
        _index(rows),
        rows,
        snapshot,
        detached_after_update=42,
        allowed_offset_pairs=((0, 0), (1, 2)),
    ) == expected


def test_post_blackout_offset_envelope_set_rejects_unknown_pair() -> None:
    rows = _rebase_rows()
    result = _audited_post_blackout_file_write_offset_envelope_set(
        _index(rows),
        rows,
        _snapshot(
            command_bit_position=141,
            buffer_read_bit_position=160,
            command_number=11,
            command_order=2,
            last_demo_update=72,
            update=72,
        ),
        detached_after_update=42,
        allowed_offset_pairs=((0, 0), (1, 2)),
    )

    assert result[:3] == (0, (1, 3), 0)
    assert result[3] is not None
    assert "allowed=0/0,1/2" in result[3]


def test_blackout_native_timeline_normalization_is_idempotent_and_frozen(
) -> None:
    snapshot = {
        "update": 9766,
        "last_demo_update": 9766,
        "command_order": 920,
    }

    normalized = _normalize_blackout_native_timeline_snapshot(
        snapshot,
        native_timeline_offset=2,
    )

    assert normalized == {
        **snapshot,
        "native_update": 9766,
        "native_last_demo_update": 9766,
        "update": 9764,
        "last_demo_update": 9764,
    }
    assert _normalize_blackout_native_timeline_snapshot(
        normalized,
        native_timeline_offset=2,
    ) == normalized
    with pytest.raises(ValueError, match="normalization changed"):
        _normalize_blackout_native_timeline_snapshot(
            normalized,
            native_timeline_offset=1,
        )


def test_snapshot_boundary_rejects_an_unknown_command_start() -> None:
    rows = [_row(start=100, end=120, update=42, command_number=11)]

    failure = _snapshot_boundary_failure(
        _index(rows),
        _snapshot(command_bit_position=999),
    )

    assert failure == "command start is absent from the offline DMO: 999"


def test_service_boundary_wait_accepts_only_the_exact_target() -> None:
    states = iter(((1583, 0), (1595, 1)))
    now = [0.0]

    observed_read, observed_needs, failure = _wait_for_service_boundary(
        target_end=1595,
        timeout=1.0,
        read_state=lambda: next(states),
        monotonic=lambda: now[0],
        delay=lambda seconds: now.__setitem__(0, now[0] + seconds),
    )

    assert (observed_read, observed_needs, failure) == (1595, 1, None)


def test_service_boundary_wait_fails_closed_on_overshoot() -> None:
    observed_read, observed_needs, failure = _wait_for_service_boundary(
        target_end=1595,
        timeout=1.0,
        read_state=lambda: (1596, 1),
        monotonic=lambda: 0.0,
        delay=lambda seconds: None,
    )

    assert (observed_read, observed_needs) == (1596, 1)
    assert failure == "service broker overshot target: 1596 > 1595"


def test_service_boundary_wait_allows_next_idle_header_handoff() -> None:
    states = iter(((1876, 1), (1880, 1)))
    now = [0.0]

    observed_read, observed_needs, failure = _wait_for_service_boundary(
        target_end=1870,
        settle_end=1880,
        timeout=1.0,
        read_state=lambda: next(states),
        monotonic=lambda: now[0],
        delay=lambda seconds: now.__setitem__(0, now[0] + seconds),
    )

    assert (observed_read, observed_needs, failure) == (1876, 1, None)


def test_service_boundary_wait_rejects_partial_terminal_header() -> None:
    states = iter(((1876, 1), (1880, 1)))
    now = [0.0]

    observed_read, observed_needs, failure = _wait_for_service_boundary(
        target_end=1870,
        settle_end=1880,
        timeout=1.0,
        read_state=lambda: next(states),
        monotonic=lambda: now[0],
        delay=lambda seconds: now.__setitem__(0, now[0] + seconds),
        allow_intermediate_prepared=False,
    )

    assert (observed_read, observed_needs, failure) == (1880, 1, None)


def test_service_boundary_wait_reports_a_stable_timeout() -> None:
    now = [0.0]

    observed_read, observed_needs, failure = _wait_for_service_boundary(
        target_end=1595,
        timeout=0.005,
        read_state=lambda: (1583, 0),
        monotonic=lambda: now[0],
        delay=lambda seconds: now.__setitem__(0, now[0] + seconds),
    )

    assert (observed_read, observed_needs) == (1583, 0)
    assert failure == "service broker timed out: read=1583, target=1595"


def test_service_progress_probe_reports_no_progress() -> None:
    now = [0.0]

    observed_read, observed_needs, status, failure = (
        _probe_service_progress(
            initial_read=1583,
            target_end=1595,
            timeout=0.005,
            read_state=lambda: (1583, 0),
            monotonic=lambda: now[0],
            delay=lambda seconds: now.__setitem__(
                0, now[0] + seconds
            ),
        )
    )

    assert (observed_read, observed_needs) == (1583, 0)
    assert (status, failure) == ("no_progress", None)


def test_service_progress_probe_stops_at_first_progress() -> None:
    states = iter(((1583, 0), (1584, 0)))
    now = [0.0]

    observed_read, observed_needs, status, failure = (
        _probe_service_progress(
            initial_read=1583,
            target_end=1595,
            timeout=1.0,
            read_state=lambda: next(states),
            monotonic=lambda: now[0],
            delay=lambda seconds: now.__setitem__(
                0, now[0] + seconds
            ),
        )
    )

    assert (observed_read, observed_needs) == (1584, 0)
    assert (status, failure) == ("progress", None)


def test_service_progress_probe_accepts_exact_boundary() -> None:
    observed_read, observed_needs, status, failure = (
        _probe_service_progress(
            initial_read=1583,
            target_end=1595,
            timeout=1.0,
            read_state=lambda: (1595, 1),
            monotonic=lambda: 0.0,
            delay=lambda seconds: None,
        )
    )

    assert (observed_read, observed_needs) == (1595, 1)
    assert (status, failure) == ("boundary", None)


def test_service_progress_probe_accepts_partial_idle_header_handoff() -> None:
    observed_read, observed_needs, status, failure = (
        _probe_service_progress(
            initial_read=1583,
            target_end=1870,
            settle_end=1880,
            timeout=1.0,
            read_state=lambda: (1876, 1),
            monotonic=lambda: 0.0,
            delay=lambda seconds: None,
        )
    )

    assert (observed_read, observed_needs) == (1876, 1)
    assert (status, failure) == ("boundary", None)


def test_service_progress_probe_fails_closed_on_regression() -> None:
    observed_read, observed_needs, status, failure = (
        _probe_service_progress(
            initial_read=1583,
            target_end=1595,
            timeout=1.0,
            read_state=lambda: (1582, 0),
            monotonic=lambda: 0.0,
            delay=lambda seconds: None,
        )
    )

    assert (observed_read, observed_needs) == (1582, 0)
    assert status == "failure"
    assert failure == "service broker read position regressed: 1582 < 1583"


def _startup_worker_state(
    *,
    read: int,
    needs: int,
    order: int,
    command_start: int,
    update: int = 384,
) -> dict[str, int]:
    return {
        "buffer_read_bit_position": read,
        "needs_command": needs,
        "command_order": order,
        "command_bit_position": command_start,
        "is_short": 0,
        "command_number": 16,
        "demo_loading_complete": 0,
        "update": update,
    }


def _startup_worker_continuation(
    rows: list[dict[str, object]],
) -> dict[str, int | bool]:
    return {
        "base": 0x12340000,
        "thread_id": 77,
        "start_index": 0,
        "end_index": len(rows) - 1,
        "end_bit_position": int(rows[-1]["end"]),
        "last_read_bit_position": int(rows[0]["start"]) + 10,
        "last_command_bit_position": -1,
        "last_command_order": -1,
        "last_reentry_update": int(rows[0]["update"]),
        "last_reentry_kind": 0,
        "reentries": 0,
        "allow_bounded_late_successful_file_write_corridor": False,
        "allow_bounded_late_preloading_failed_file_write_corridor": True,
    }


def test_startup_post_bypass_worker_waits_for_stable_consumed_target(
) -> None:
    rows = [
        _row(
            start=start,
            end=start + 11,
            update=384 + index,
            command_number=16,
            kind="file_write",
            payload={"success": False},
        )
        for index, start in enumerate((100, 111, 122))
    ]
    states = [
        _startup_worker_state(
            read=110,
            needs=0,
            order=0,
            command_start=100,
        ),
        _startup_worker_state(
            read=122,
            needs=1,
            order=1,
            command_start=111,
        ),
    ]
    calls = [0]
    now = [0.0]

    def read_state() -> dict[str, int]:
        index = min(calls[0], len(states) - 1)
        calls[0] += 1
        return states[index]

    observed, status, failure = (
        _wait_for_startup_post_bypass_worker_boundary(
            rows,
            corridor_start_index=0,
            corridor_end_index=2,
            target_row_index=1,
            initial_read_bit_position=110,
            command_order_offset=0,
            timeout=1.0,
            settle_interval=0.03,
            poll_interval=0.01,
            read_state=read_state,
            monotonic=lambda: now[0],
            delay=lambda seconds: now.__setitem__(
                0, now[0] + seconds
            ),
        )
    )

    assert status == "boundary"
    assert failure is None
    assert observed["buffer_read_bit_position"] == 122
    assert observed["settled_row_index"] == 1
    assert observed["settled_after_payload"] == 1
    assert observed["settled_at_corridor_end"] == 0


def test_startup_post_bypass_worker_closes_window_before_accepting() -> None:
    rows = [
        _row(
            start=100,
            end=111,
            update=384,
            command_number=16,
            kind="file_write",
            payload={"success": False},
        )
    ]
    states = [
        _startup_worker_state(
            read=110,
            needs=0,
            order=0,
            command_start=100,
        ),
        _startup_worker_state(
            read=111,
            needs=1,
            order=0,
            command_start=100,
        ),
    ]
    calls = [0]
    closes = [0]
    now = [0.0]

    def read_state() -> dict[str, int]:
        index = min(calls[0], len(states) - 1)
        calls[0] += 1
        return states[index]

    observed, status, failure = (
        _wait_for_startup_post_bypass_worker_boundary(
            rows,
            corridor_start_index=0,
            corridor_end_index=0,
            target_row_index=0,
            initial_read_bit_position=110,
            command_order_offset=0,
            timeout=1.0,
            settle_interval=0.03,
            poll_interval=0.01,
            read_state=read_state,
            monotonic=lambda: now[0],
            delay=lambda seconds: now.__setitem__(
                0, now[0] + seconds
            ),
            close_window=lambda: closes.__setitem__(0, closes[0] + 1),
        )
    )

    assert status == "boundary"
    assert failure is None
    assert observed["buffer_read_bit_position"] == 111
    assert closes == [1]


def test_startup_post_bypass_worker_accepts_exact_next_update_block() -> None:
    rows = [
        _row(
            start=start,
            end=start + 11,
            update=update,
            command_number=16,
            kind="file_write",
            payload={"success": False},
        )
        for start, update in ((100, 384), (111, 385), (122, 385))
    ]
    states = [
        _startup_worker_state(
            read=110,
            needs=0,
            order=0,
            command_start=100,
        ),
        _startup_worker_state(
            read=133,
            needs=1,
            order=2,
            command_start=122,
        ),
    ]
    calls = [0]
    now = [0.0]

    def read_state() -> dict[str, int]:
        index = min(calls[0], len(states) - 1)
        calls[0] += 1
        return states[index]

    observed, status, failure = (
        _wait_for_startup_post_bypass_worker_boundary(
            rows,
            corridor_start_index=0,
            corridor_end_index=2,
            target_row_index=0,
            alternate_target_row_index=2,
            initial_read_bit_position=110,
            command_order_offset=0,
            timeout=1.0,
            settle_interval=0.03,
            poll_interval=0.01,
            read_state=read_state,
            monotonic=lambda: now[0],
            delay=lambda seconds: now.__setitem__(
                0, now[0] + seconds
            ),
        )
    )

    assert status == "boundary"
    assert failure is None
    assert observed["buffer_read_bit_position"] == 133
    assert observed["settled_row_index"] == 2


def test_startup_post_bypass_worker_accepts_boundary_inside_next_block(
) -> None:
    rows = [
        _row(
            start=start,
            end=start + 11,
            update=update,
            command_number=16,
            kind="file_write",
            payload={"success": False},
        )
        for start, update in ((100, 384), (111, 385), (122, 385))
    ]
    states = [
        _startup_worker_state(
            read=110,
            needs=0,
            order=0,
            command_start=100,
        ),
        _startup_worker_state(
            read=122,
            needs=1,
            order=1,
            command_start=111,
        ),
    ]
    calls = [0]
    now = [0.0]

    def read_state() -> dict[str, int]:
        index = min(calls[0], len(states) - 1)
        calls[0] += 1
        return states[index]

    observed, status, failure = (
        _wait_for_startup_post_bypass_worker_boundary(
            rows,
            corridor_start_index=0,
            corridor_end_index=2,
            target_row_index=0,
            alternate_target_row_index=2,
            initial_read_bit_position=110,
            command_order_offset=0,
            timeout=1.0,
            settle_interval=0.03,
            poll_interval=0.01,
            read_state=read_state,
            monotonic=lambda: now[0],
            delay=lambda seconds: now.__setitem__(
                0, now[0] + seconds
            ),
        )
    )

    assert status == "boundary"
    assert failure is None
    assert observed["buffer_read_bit_position"] == 122
    assert observed["settled_row_index"] == 1


def test_startup_worker_commit_preserves_exact_target_branch() -> None:
    rows = [
        _row(
            start=start,
            end=start + 11,
            update=update,
            command_number=16,
            kind="file_write",
            payload={"success": False},
        )
        for start, update in ((100, 384), (111, 385), (122, 385))
    ]
    observation = {
        **_startup_worker_state(
            read=111,
            needs=1,
            order=0,
            command_start=100,
            update=384,
        ),
        "settled_row_index": 0,
        "settled_after_payload": 1,
        "settled_at_corridor_end": 0,
    }

    committed, receipt, failure = (
        _commit_startup_post_bypass_worker_continuation(
            rows,
            _startup_worker_continuation(rows),
            observation,
            target_row_index=0,
            alternate_target_row_index=2,
            initial_read_bit_position=110,
            command_order_offset=0,
        )
    )

    assert failure is None
    assert committed is not None
    assert receipt is not None
    assert committed["last_read_bit_position"] == 111
    assert committed["last_command_bit_position"] == 100
    assert committed["last_command_order"] == 0
    assert committed["last_reentry_kind"] == 12
    assert committed["reentries"] == 1
    assert receipt["settled_row_index"] == 0

    advanced, echo, echo_failure = (
        _consume_startup_post_bypass_worker_echo(
            rows,
            committed,
            _snapshot(
                base=0x12340000,
                return_address=MAIN_DEMO_CALLER,
                buffer_read_bit_position=111,
                needs_command=1,
                command_order=0,
                command_bit_position=100,
                is_short=0,
                command_number=16,
                demo_loading_complete=0,
                update=384,
                last_demo_update=384,
            ),
            thread_id=77,
            command_order_offset=0,
        )
    )

    assert echo_failure is None
    assert advanced is not None
    assert echo is not None
    assert advanced["last_reentry_kind"] == 0
    assert advanced["last_read_bit_position"] == 111
    assert echo["same_update_service_block_start_index"] == -1
    assert echo["same_update_service_block_end_index"] == -1


def test_startup_worker_commit_brokers_alternate_same_update_suffix() -> None:
    rows = [
        _row(
            start=start,
            end=start + 11,
            update=update,
            command_number=16,
            kind="file_write",
            payload={"success": False},
        )
        for start, update in ((100, 384), (111, 385), (122, 385))
    ]
    observation = {
        **_startup_worker_state(
            read=122,
            needs=1,
            order=1,
            command_start=111,
            update=384,
        ),
        "settled_row_index": 1,
        "settled_after_payload": 1,
        "settled_at_corridor_end": 0,
    }

    committed, receipt, failure = (
        _commit_startup_post_bypass_worker_continuation(
            rows,
            _startup_worker_continuation(rows),
            observation,
            target_row_index=0,
            alternate_target_row_index=2,
            initial_read_bit_position=110,
            command_order_offset=0,
        )
    )

    assert failure is None
    assert committed is not None
    assert receipt is not None
    assert committed["last_read_bit_position"] == 122
    assert committed["last_command_bit_position"] == 111
    assert committed["last_command_order"] == 1
    assert committed["last_reentry_update"] == 384
    assert committed["last_reentry_kind"] == 12
    assert committed["reentries"] == 2
    assert receipt["consumed_row_count"] == 2
    assert receipt["settled_recorded_update"] == 385

    advanced, echo, echo_failure = (
        _consume_startup_post_bypass_worker_echo(
            rows,
            committed,
            _snapshot(
                base=0x12340000,
                return_address=MAIN_DEMO_CALLER,
                buffer_read_bit_position=122,
                needs_command=1,
                command_order=1,
                command_bit_position=111,
                is_short=0,
                command_number=16,
                demo_loading_complete=0,
                update=385,
                last_demo_update=385,
            ),
            thread_id=77,
            command_order_offset=0,
        )
    )

    assert echo_failure is None
    assert advanced is not None
    assert echo is not None
    assert advanced["last_reentry_kind"] == 0
    assert advanced["last_reentry_update"] == 385
    assert echo["same_update_service_block_start_index"] == 2
    assert echo["same_update_service_block_end_index"] == 2
    assert echo[
        "same_update_service_block_target_end_bit_position"
    ] == 133


@pytest.mark.parametrize(
    ("field", "value", "expected_name"),
    [
        ("buffer_read_bit_position", 121, "buffer_read_bit_position"),
        ("command_bit_position", 112, "command_bit_position"),
        ("command_order", 2, "command_order"),
        ("update", 386, "update"),
        ("settled_after_payload", 0, "settled_after_payload"),
    ],
)
def test_startup_worker_commit_rejects_tampered_observation(
    field: str,
    value: int,
    expected_name: str,
) -> None:
    rows = [
        _row(
            start=start,
            end=start + 11,
            update=update,
            command_number=16,
            kind="file_write",
            payload={"success": False},
        )
        for start, update in ((100, 384), (111, 385), (122, 385))
    ]
    observation = {
        **_startup_worker_state(
            read=122,
            needs=1,
            order=1,
            command_start=111,
            update=384,
        ),
        "settled_row_index": 1,
        "settled_after_payload": 1,
        "settled_at_corridor_end": 0,
    }
    observation[field] = value

    committed, receipt, failure = (
        _commit_startup_post_bypass_worker_continuation(
            rows,
            _startup_worker_continuation(rows),
            observation,
            target_row_index=0,
            alternate_target_row_index=2,
            initial_read_bit_position=110,
            command_order_offset=0,
        )
    )

    assert committed is None
    assert receipt is None
    assert failure is not None
    assert expected_name in failure


def test_startup_worker_echo_rejects_changed_atomic_boundary() -> None:
    rows = [
        _row(
            start=start,
            end=start + 11,
            update=update,
            command_number=16,
            kind="file_write",
            payload={"success": False},
        )
        for start, update in ((100, 384), (111, 385), (122, 385))
    ]
    observation = {
        **_startup_worker_state(
            read=122,
            needs=1,
            order=1,
            command_start=111,
            update=384,
        ),
        "settled_row_index": 1,
        "settled_after_payload": 1,
        "settled_at_corridor_end": 0,
    }
    committed, _, failure = (
        _commit_startup_post_bypass_worker_continuation(
            rows,
            _startup_worker_continuation(rows),
            observation,
            target_row_index=0,
            alternate_target_row_index=2,
            initial_read_bit_position=110,
            command_order_offset=0,
        )
    )
    assert failure is None
    assert committed is not None

    advanced, receipt, echo_failure = (
        _consume_startup_post_bypass_worker_echo(
            rows,
            committed,
            _snapshot(
                base=0x12340000,
                return_address=MAIN_DEMO_CALLER,
                buffer_read_bit_position=123,
                needs_command=1,
                command_order=1,
                command_bit_position=111,
                is_short=0,
                command_number=16,
                demo_loading_complete=0,
                update=385,
                last_demo_update=385,
            ),
            thread_id=77,
            command_order_offset=0,
        )
    )

    assert advanced is None
    assert receipt is None
    assert echo_failure == (
        "startup worker echo observation mismatch: "
        "buffer_read_bit_position=123 != 122"
    )


def test_startup_post_bypass_worker_rejects_partial_alternate_block() -> None:
    rows = [
        _row(
            start=start,
            end=start + 11,
            update=update,
            command_number=16,
            kind="file_write",
            payload={"success": False},
        )
        for start, update in ((100, 384), (111, 385), (122, 385))
    ]

    with pytest.raises(
        ValueError,
        match="alternate target is not one exact next-update block",
    ):
        _wait_for_startup_post_bypass_worker_boundary(
            rows,
            corridor_start_index=0,
            corridor_end_index=2,
            target_row_index=0,
            alternate_target_row_index=1,
            initial_read_bit_position=110,
            command_order_offset=0,
            timeout=1.0,
            read_state=lambda: _startup_worker_state(
                read=110,
                needs=0,
                order=0,
                command_start=100,
            ),
            monotonic=lambda: 0.0,
            delay=lambda seconds: None,
        )


def test_startup_post_bypass_worker_rejects_advance_while_closing() -> None:
    rows = [
        _row(
            start=start,
            end=start + 11,
            update=384 + index,
            command_number=16,
            kind="file_write",
            payload={"success": False},
        )
        for index, start in enumerate((100, 111))
    ]
    state = _startup_worker_state(
        read=111,
        needs=1,
        order=0,
        command_start=100,
    )
    calls = [0]
    now = [0.0]

    def read_state() -> dict[str, int]:
        calls[0] += 1
        if calls[0] == 1:
            return _startup_worker_state(
                read=110,
                needs=0,
                order=0,
                command_start=100,
            )
        return dict(state)

    def close_window() -> None:
        state.update(
            _startup_worker_state(
                read=122,
                needs=1,
                order=1,
                command_start=111,
            )
        )

    observed, status, failure = (
        _wait_for_startup_post_bypass_worker_boundary(
            rows,
            corridor_start_index=0,
            corridor_end_index=1,
            target_row_index=0,
            initial_read_bit_position=110,
            command_order_offset=0,
            timeout=1.0,
            settle_interval=0.03,
            poll_interval=0.01,
            read_state=read_state,
            monotonic=lambda: now[0],
            delay=lambda seconds: now.__setitem__(
                0, now[0] + seconds
            ),
            close_window=close_window,
        )
    )

    assert observed["buffer_read_bit_position"] == 122
    assert status == "failure"
    assert failure == (
        "startup post-bypass worker escaped its exact target: 122 > 111"
    )


def test_startup_post_bypass_worker_rejects_a_later_consumed_row() -> None:
    rows = [
        _row(
            start=start,
            end=start + 11,
            update=384 + index,
            command_number=16,
            kind="file_write",
            payload={"success": False},
        )
        for index, start in enumerate((100, 111, 122))
    ]
    states = [
        _startup_worker_state(
            read=110,
            needs=0,
            order=0,
            command_start=100,
        ),
        _startup_worker_state(
            read=133,
            needs=1,
            order=2,
            command_start=122,
        ),
    ]
    calls = [0]
    now = [0.0]

    def read_state() -> dict[str, int]:
        index = min(calls[0], len(states) - 1)
        calls[0] += 1
        return states[index]

    observed, status, failure = (
        _wait_for_startup_post_bypass_worker_boundary(
            rows,
            corridor_start_index=0,
            corridor_end_index=2,
            target_row_index=1,
            initial_read_bit_position=110,
            command_order_offset=0,
            timeout=1.0,
            settle_interval=0.03,
            poll_interval=0.01,
            read_state=read_state,
            monotonic=lambda: now[0],
            delay=lambda seconds: now.__setitem__(
                0, now[0] + seconds
            ),
        )
    )

    assert status == "failure"
    assert failure == (
        "startup post-bypass worker escaped its exact target: 133 > 122"
    )


def test_startup_post_bypass_worker_rejects_an_earlier_consumed_row() -> None:
    rows = [
        _row(
            start=start,
            end=start + 11,
            update=384 + index,
            command_number=16,
            kind="file_write",
            payload={"success": False},
        )
        for index, start in enumerate((100, 111, 122))
    ]
    states = [
        _startup_worker_state(
            read=110,
            needs=0,
            order=0,
            command_start=100,
        ),
        _startup_worker_state(
            read=111,
            needs=1,
            order=0,
            command_start=100,
        ),
    ]
    calls = [0]
    now = [0.0]

    def read_state() -> dict[str, int]:
        index = min(calls[0], len(states) - 1)
        calls[0] += 1
        return states[index]

    observed, status, failure = (
        _wait_for_startup_post_bypass_worker_boundary(
            rows,
            corridor_start_index=0,
            corridor_end_index=2,
            target_row_index=1,
            initial_read_bit_position=110,
            command_order_offset=0,
            timeout=1.0,
            settle_interval=0.03,
            poll_interval=0.01,
            read_state=read_state,
            monotonic=lambda: now[0],
            delay=lambda seconds: now.__setitem__(
                0, now[0] + seconds
            ),
        )
    )

    assert status == "failure"
    assert failure == (
        "startup post-bypass worker did not settle at an exact offline "
        "boundary: read=111, needs=1, order=0"
    )


def test_startup_post_bypass_worker_fails_closed_without_progress() -> None:
    rows = [
        _row(
            start=100,
            end=111,
            update=384,
            command_number=16,
            kind="file_write",
            payload={"success": False},
        )
    ]
    now = [0.0]

    observed, status, failure = (
        _wait_for_startup_post_bypass_worker_boundary(
            rows,
            corridor_start_index=0,
            corridor_end_index=0,
            target_row_index=0,
            initial_read_bit_position=110,
            command_order_offset=0,
            timeout=0.05,
            settle_interval=0.02,
            poll_interval=0.01,
            read_state=lambda: _startup_worker_state(
                read=110,
                needs=0,
                order=0,
                command_start=100,
            ),
            monotonic=lambda: now[0],
            delay=lambda seconds: now.__setitem__(
                0, now[0] + seconds
            ),
        )
    )

    assert observed["buffer_read_bit_position"] == 110
    assert status == "no_progress"
    assert failure == (
        "startup post-bypass worker made no progress: read=110"
    )


def test_startup_post_bypass_worker_rejects_corridor_overshoot() -> None:
    rows = [
        _row(
            start=100,
            end=111,
            update=384,
            command_number=16,
            kind="file_write",
            payload={"success": False},
        )
    ]

    observed, status, failure = (
        _wait_for_startup_post_bypass_worker_boundary(
            rows,
            corridor_start_index=0,
            corridor_end_index=0,
            target_row_index=0,
            initial_read_bit_position=110,
            command_order_offset=0,
            timeout=1.0,
            read_state=lambda: _startup_worker_state(
                read=112,
                needs=0,
                order=1,
                command_start=111,
            ),
            monotonic=lambda: 0.0,
            delay=lambda seconds: None,
        )
    )

    assert observed["buffer_read_bit_position"] == 112
    assert status == "failure"
    assert failure == (
        "startup post-bypass worker escaped its exact target: 112 > 111"
    )


@pytest.mark.parametrize(
    ("override", "value", "expected_key"),
    [
        ("command_order", 1, "command_order"),
        ("command_number", 12, "command_number"),
        ("is_short", 1, "is_short"),
        ("last_demo_update", 43, "last_demo_update"),
    ],
)
def test_snapshot_boundary_reports_the_first_mismatch(
    override: str,
    value: int,
    expected_key: str,
) -> None:
    rows = [_row(start=100, end=120, update=42, command_number=11)]

    failure = _snapshot_boundary_failure(
        _index(rows),
        _snapshot(**{override: value}),
    )

    assert failure is not None
    assert expected_key in failure
