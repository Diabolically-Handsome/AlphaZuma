from __future__ import annotations

import struct

import pytest

from tools.inspect_popcap_replay import (
    decode_replay_state,
    exact_value_offsets,
    primitive_window,
)


def test_exact_value_offsets_finds_all_unaligned_matches() -> None:
    needle = struct.pack("<d", 0.125)
    data = b"x" + needle + b"abc" + needle
    assert exact_value_offsets(data, 0.125) == (1, 12)


def test_exact_value_offsets_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        exact_value_offsets(b"", float("nan"))


def test_primitive_window_reports_relative_offsets_and_values() -> None:
    data = struct.pack("<iiidi", 10, 20, 30, 0.5, 40)
    rows = primitive_window(data, 12, radius=12)
    by_offset = {row["relative_offset"]: row for row in rows}
    assert by_offset[-12]["int32"] == 10
    assert by_offset[-4]["int32"] == 30
    assert by_offset[0]["float64"] == 0.5


def test_primitive_window_rejects_negative_arguments() -> None:
    with pytest.raises(ValueError):
        primitive_window(b"", -1)


def test_decode_replay_state_uses_framework_field_offsets() -> None:
    center = 64
    data = bytearray(192)
    struct.pack_into("<i", data, center - 60, 7)
    struct.pack_into("<i", data, center - 56, 10)
    struct.pack_into("<i", data, center - 20, 100)
    struct.pack_into("<i", data, center - 16, 20)
    struct.pack_into("<i", data, center - 12, 439)
    struct.pack_into("<i", data, center - 8, 3)
    struct.pack_into("<i", data, center - 4, 1)
    struct.pack_into("<d", data, center, 0.125)
    data[center + 8] = 1
    struct.pack_into("<i", data, center + 12, 440)
    data[center + 16] = 1
    data[center + 17] = 1
    struct.pack_into("<i", data, center + 28, 2)
    data[center + 105] = 1
    data[center + 106] = 1
    data[center + 107] = 1

    state = decode_replay_state(bytes(data), center, 0x12345678)
    assert state.multiplier_address == 0x12345678
    assert state.non_draw_count == 7
    assert state.frame_time_ms == 10
    assert state.sleep_count == 100
    assert state.draw_count == 20
    assert state.update_count == 439
    assert state.update_app_state == 3
    assert state.update_app_depth == 1
    assert state.update_multiplier == 0.125
    assert state.paused is True
    assert state.fast_forward_target == 440
    assert state.fast_forward_to_marker is True
    assert state.fast_forward_step is True
    assert state.step_mode == 2
    assert state.loading_thread_started is True
    assert state.loading_thread_completed is True
    assert state.loaded is True


def test_decode_replay_state_rejects_short_window() -> None:
    with pytest.raises(ValueError, match="insufficient"):
        decode_replay_state(bytes(20), 10, 0)
