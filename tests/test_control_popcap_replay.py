from __future__ import annotations

import pytest

from tools import control_popcap_replay
from tools.control_popcap_replay import (
    expected_multiplier,
    wait_for_replay_step,
)
from tools.inspect_popcap_replay import ReplayState


def _state(
    update: int,
    *,
    target: int,
    fast_forward_step: bool,
    update_app_state: int = 1,
    update_app_depth: int = 1,
) -> ReplayState:
    return ReplayState(
        multiplier_address=0x12345678,
        non_draw_count=0,
        frame_time_ms=10,
        sleep_count=1,
        draw_count=2,
        update_count=update,
        update_app_state=update_app_state,
        update_app_depth=update_app_depth,
        update_multiplier=0.0877914951989026,
        paused=False,
        fast_forward_target=target,
        fast_forward_to_marker=False,
        fast_forward_step=fast_forward_step,
        step_mode=0,
        loading_thread_started=True,
        loading_thread_completed=True,
        loaded=True,
    )


def test_expected_multiplier_matches_framework_division() -> None:
    assert expected_multiplier(0) == 1.0
    assert expected_multiplier(8) == 1.0 / (1.5**8)
    assert expected_multiplier(5) == float.fromhex("0x1.0db20a88f4695p-3")


def test_expected_multiplier_rejects_negative_count() -> None:
    with pytest.raises(ValueError):
        expected_multiplier(-1)


def test_five_to_six_minuses_crosses_freeze_threshold() -> None:
    assert expected_multiplier(5) > 0.1
    assert expected_multiplier(6) <= 0.1


def test_replay_step_waits_past_early_update_counter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    states = iter(
        (
            _state(100, target=101, fast_forward_step=True),
            _state(101, target=101, fast_forward_step=True),
            _state(101, target=101, fast_forward_step=False),
            _state(101, target=101, fast_forward_step=False),
        )
    )
    monkeypatch.setattr(
        control_popcap_replay,
        "read_replay_state",
        lambda _handle, _address: next(states),
    )
    monkeypatch.setattr(control_popcap_replay.time, "sleep", lambda _delay: None)

    state = wait_for_replay_step(1, 0x12345678, 101, timeout=1.0)

    assert state.update_count == 101
    assert state.fast_forward_step is False


def test_replay_step_accepts_idle_depth_one_after_barrier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(
        101,
        target=101,
        fast_forward_step=False,
        update_app_state=1,
        update_app_depth=1,
    )
    monkeypatch.setattr(
        control_popcap_replay,
        "read_replay_state",
        lambda _handle, _address: state,
    )
    monkeypatch.setattr(control_popcap_replay.time, "sleep", lambda _delay: None)

    observed = wait_for_replay_step(1, 0x12345678, 101, timeout=1.0)

    assert observed.update_app_state == 1
    assert observed.update_app_depth == 1


def test_replay_step_rejects_overshoot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(102, target=101, fast_forward_step=False)
    monkeypatch.setattr(
        control_popcap_replay,
        "read_replay_state",
        lambda _handle, _address: state,
    )

    with pytest.raises(RuntimeError, match="overshot step target"):
        wait_for_replay_step(1, 0x12345678, 101, timeout=1.0)


def test_replay_step_rejects_invalid_stable_read_count() -> None:
    state = _state(101, target=101, fast_forward_step=False)
    with pytest.raises(ValueError, match="stable_reads"):
        wait_for_replay_step(
            1,
            state.multiplier_address,
            101,
            timeout=1.0,
            stable_reads=0,
        )
