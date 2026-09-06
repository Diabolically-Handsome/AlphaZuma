from __future__ import annotations

import pytest

from tools import pc_external_input_guard as guard_module
from tools.pc_external_input_guard import (
    ExternalInputGuard,
    ExternalInputGuardError,
)


def _external_mouse_move(guard: ExternalInputGuard) -> None:
    guard._record(
        device="mouse",
        action="move",
        flags=0,
        extra_info=0,
        injected_flag=0x01,
    )


def test_precoverage_activity_is_not_formal_but_resets_quiescence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guard = ExternalInputGuard()
    monkeypatch.setattr(guard_module.time, "perf_counter_ns", lambda: 123)
    _external_mouse_move(guard)
    assert guard._precoverage_event_count == 1
    assert guard._precoverage_last_activity_ns == 123
    assert guard._counts["external"] == 0
    assert guard._external_events == []

    guard._coverage_start_ns = 100
    monkeypatch.setattr(guard_module.time, "perf_counter_ns", lambda: 124)
    _external_mouse_move(guard)
    assert guard._counts["external"] == 1
    assert guard._external_events == [
        {
            "perf_counter_ns": 124,
            "device": "mouse",
            "action": "move",
            "injected": False,
        }
    ]


def test_start_runs_kat_and_quiescence_before_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guard = ExternalInputGuard()
    order: list[str] = []

    def fake_thread_main() -> None:
        guard._keyboard_installed = True
        guard._mouse_installed = True
        order.append("hooks")
        guard._ready.set()

    def fake_self_test(label: str) -> dict[str, object]:
        assert label == "start"
        assert guard._coverage_start_ns is None
        order.append("kat")
        return {
            "label": "start",
            "status": "PASS",
            "keyboard_event_count": 2,
            "mouse_event_count": 2,
        }

    def fake_start_coverage() -> int:
        assert guard._coverage_start_ns is None
        order.append("quiescence")
        guard._coverage_start_ns = 456
        return 456

    monkeypatch.setattr(guard_module, "_require_windows", lambda: None)
    monkeypatch.setattr(guard, "_thread_main", fake_thread_main)
    monkeypatch.setattr(guard, "_run_self_test", fake_self_test)
    monkeypatch.setattr(
        guard,
        "_start_coverage_after_quiescence",
        fake_start_coverage,
    )
    assert guard.start() == 456
    assert order == ["hooks", "kat", "quiescence"]


def test_quiescence_timer_restarts_after_precoverage_activity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guard = ExternalInputGuard()
    clock_ns = 0
    sleep_count = 0

    def perf_counter_ns() -> int:
        return clock_ns

    def monotonic() -> float:
        return clock_ns / 1_000_000_000

    def sleep(seconds: float) -> None:
        nonlocal clock_ns, sleep_count
        clock_ns += round(seconds * 1_000_000_000)
        sleep_count += 1
        if sleep_count == 1:
            guard._precoverage_last_activity_ns = clock_ns

    monkeypatch.setattr(guard_module, "START_QUIESCENCE_SECONDS", 0.5)
    monkeypatch.setattr(
        guard_module,
        "START_QUIESCENCE_TIMEOUT_SECONDS",
        5.0,
    )
    monkeypatch.setattr(
        guard_module,
        "START_QUIESCENCE_POLL_SECONDS",
        0.1,
    )
    monkeypatch.setattr(guard_module.time, "perf_counter_ns", perf_counter_ns)
    monkeypatch.setattr(guard_module.time, "monotonic", monotonic)
    monkeypatch.setattr(guard_module.time, "sleep", sleep)

    started = guard._start_coverage_after_quiescence()
    assert guard._precoverage_last_activity_ns == 100_000_000
    assert started >= 600_000_000
    assert guard._coverage_start_ns == started


def test_continuous_precoverage_activity_times_out_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guard = ExternalInputGuard()
    clock_ns = 0

    def perf_counter_ns() -> int:
        return clock_ns

    def monotonic() -> float:
        return clock_ns / 1_000_000_000

    def sleep(seconds: float) -> None:
        nonlocal clock_ns
        clock_ns += round(seconds * 1_000_000_000)
        guard._precoverage_last_activity_ns = clock_ns

    monkeypatch.setattr(guard_module, "START_QUIESCENCE_SECONDS", 0.5)
    monkeypatch.setattr(
        guard_module,
        "START_QUIESCENCE_TIMEOUT_SECONDS",
        1.0,
    )
    monkeypatch.setattr(
        guard_module,
        "START_QUIESCENCE_POLL_SECONDS",
        0.1,
    )
    monkeypatch.setattr(guard_module.time, "perf_counter_ns", perf_counter_ns)
    monkeypatch.setattr(guard_module.time, "monotonic", monotonic)
    monkeypatch.setattr(guard_module.time, "sleep", sleep)

    with pytest.raises(
        ExternalInputGuardError,
        match="start_quiescence_timeout",
    ):
        guard._start_coverage_after_quiescence()
    assert guard._coverage_start_ns is None
