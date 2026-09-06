from __future__ import annotations

from tools.trace_popcap_global_rng_calls import (
    _can_stop_without_pending_single_step,
)


def test_frozen_trace_can_stop_without_waiting_for_another_rng_call() -> None:
    assert _can_stop_without_pending_single_step(
        stop_requested=True,
        stepping_thread=None,
    )


def test_trace_waits_until_an_inflight_single_step_is_completed() -> None:
    assert not _can_stop_without_pending_single_step(
        stop_requested=True,
        stepping_thread=123,
    )
    assert not _can_stop_without_pending_single_step(
        stop_requested=False,
        stepping_thread=None,
    )
