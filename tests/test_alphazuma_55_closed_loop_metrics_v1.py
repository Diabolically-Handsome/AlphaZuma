from __future__ import annotations

import math

import pytest

from tools.alphazuma_55_closed_loop_metrics_v1 import (
    AimIntentErrorAccumulator,
    HonestExecutionAccounting,
    RAW_ENV_COMMIT_TO_RELEASE_TICKS,
    ReleaseAngleConfig,
    ReleaseAngleTracker,
    ReleaseTickRecord,
    TickClassificationAccumulator,
    WRAPPED_STACK_COMMIT_TO_RELEASE_TICKS,
    aim_intent_error,
    classify_tick,
)


def test_dead_student_exposes_old_metric_dilution() -> None:
    """Document the v1 flaw: ~2.5% per-tick versus 100% decision-relevant.

    The 55/55 teacher issues non-wait verbs on only 2.2-2.9% of ticks, so
    the old per-tick disagreement rate scores a student that never acts
    as ~2.5% "disagreement" because both-wait ticks agree vacuously.  The
    decision-relevant denominator scores the same stream at 100%.
    """

    accumulator = TickClassificationAccumulator()
    total_ticks = 1000
    for tick in range(total_ticks):
        teacher_action = (1, 90) if tick % 40 == 0 else (0, 90)
        student_action = (0, 17)
        accumulator.add(teacher_action, student_action)
    summary = accumulator.summary()
    assert summary.total_ticks == 1000
    assert summary.decision_relevant_ticks == 25
    assert summary.disagreement_ticks == 25
    assert summary.per_tick_disagreement_rate == pytest.approx(0.025)
    assert summary.decision_relevant_disagreement_rate == pytest.approx(1.0)
    assert summary.reason_counts == {"student_omits_teacher_verb": 25}


def test_classify_tick_carries_v1_reason_taxonomy() -> None:
    added = classify_tick((0, 10), (2, 10))
    assert added.decision_relevant is True
    assert added.disagreement_reason == "student_adds_unrequested_verb"
    omitted = classify_tick((1, 10), (0, 10))
    assert omitted.decision_relevant is True
    assert omitted.disagreement_reason == "student_omits_teacher_verb"
    mismatched = classify_tick((2, 10), (3, 10))
    assert mismatched.decision_relevant is True
    assert mismatched.disagreement_reason == "verb_mismatch"
    fire_aim = classify_tick((1, 10), (1, 20))
    assert fire_aim.decision_relevant is True
    assert fire_aim.disagreement_reason == "fire_aim_mismatch"
    both_fire_close = classify_tick((1, 10), (1, 11))
    assert both_fire_close.decision_relevant is True
    assert both_fire_close.disagreement is False
    both_wait = classify_tick((0, 10), (0, 170))
    assert both_wait.decision_relevant is False
    assert both_wait.disagreement is False
    assert both_wait.disagreement_reason is None


def test_empty_stream_reports_nan_rates_not_vacuous_agreement() -> None:
    summary = TickClassificationAccumulator().summary()
    assert math.isnan(summary.per_tick_disagreement_rate)
    assert math.isnan(summary.decision_relevant_disagreement_rate)


def test_aim_intent_error_wraps_circularly() -> None:
    assert aim_intent_error(179, 0) == 1
    assert aim_intent_error(0, 179) == 1
    assert aim_intent_error(179, 1) == 2
    assert aim_intent_error(45, 135) == 90
    assert aim_intent_error(90, 90) == 0
    with pytest.raises(ValueError, match="escaped"):
        aim_intent_error(180, 0)


def test_aim_intent_accumulator_reports_mean_p50_p90() -> None:
    accumulator = AimIntentErrorAccumulator()
    for teacher_bin, student_bin in [(179, 0), (10, 10), (0, 170)]:
        accumulator.add(teacher_bin, student_bin)
    summary = accumulator.summary()
    assert summary.count == 3
    assert summary.mean_bins == pytest.approx((1 + 0 + 10) / 3)
    assert summary.p50_bins == pytest.approx(1.0)
    assert summary.p90_bins == pytest.approx(8.2)
    assert summary.max_bins == 10


def test_aim_intent_accumulator_rejects_empty_summary() -> None:
    with pytest.raises(ValueError, match="no aim intent samples"):
        AimIntentErrorAccumulator().summary()


def test_release_attribution_across_wrapped_delay_with_drift() -> None:
    """Two shots through the 18-tick wrapped delay, one drifting off."""

    assert WRAPPED_STACK_COMMIT_TO_RELEASE_TICKS == 18
    tracker = ReleaseAngleTracker()
    resolved_by_tick: dict[int, list] = {}
    for tick in range(70):
        if tick == 10:
            executor: str | None = "student"
        elif tick == 40:
            executor = "teacher"
        else:
            executor = None
        target_bin = 50 if tick < 35 else 90
        if tick < 35:
            aim_bin = 50
        elif tick < 47:
            aim_bin = 90
        else:
            aim_bin = 85  # cursor drifts away after the tick-40 commit
        resolved = tracker.observe(
            ReleaseTickRecord(
                tick=tick,
                executed_verb=1 if executor is not None else 0,
                executed_aim_bin=aim_bin,
                teacher_target_bin=target_bin,
                fire_edge_executor=executor,
            )
        )
        if resolved:
            resolved_by_tick[tick] = resolved
    assert sorted(resolved_by_tick) == [28, 58]
    first = resolved_by_tick[28][0]
    assert first.executor == "student"
    assert first.commit_tick == 10
    assert first.release_tick == 28
    assert first.commit_target_bin == 50
    assert first.release_aim_bin == 50
    assert first.release_angle_error_bins == 0
    assert first.on_target is True
    second = resolved_by_tick[58][0]
    assert second.executor == "teacher"
    assert second.commit_tick == 40
    assert second.release_tick == 58
    assert second.commit_target_bin == 90
    assert second.release_aim_bin == 85
    assert second.release_angle_error_bins == 5
    assert second.on_target is False
    summary = tracker.summary()
    assert summary.shot_count == 2
    assert summary.shots_on_target_rate == pytest.approx(0.5)
    assert summary.mean_release_angle_error_bins == pytest.approx(2.5)
    assert summary.pending_commits == 0
    assert summary.per_executor["student"].shot_count == 1
    assert summary.per_executor["student"].shots_on_target_rate == (
        pytest.approx(1.0)
    )
    assert summary.per_executor["teacher"].shot_count == 1
    assert summary.per_executor["teacher"].shots_on_target_rate == (
        pytest.approx(0.0)
    )
    assert summary.per_executor["teacher"].mean_release_angle_error_bins == (
        pytest.approx(5.0)
    )


def test_release_tracker_raw_env_delay_and_pending_commits() -> None:
    assert RAW_ENV_COMMIT_TO_RELEASE_TICKS == 6
    tracker = ReleaseAngleTracker(
        ReleaseAngleConfig(
            commit_to_release_ticks=RAW_ENV_COMMIT_TO_RELEASE_TICKS,
        )
    )
    for tick in range(10):
        tracker.observe(
            ReleaseTickRecord(
                tick=tick,
                executed_verb=1 if tick in (0, 8) else 0,
                executed_aim_bin=60,
                teacher_target_bin=60,
                fire_edge_executor=(
                    "student" if tick == 0 else "teacher" if tick == 8 else None
                ),
            )
        )
    summary = tracker.summary()
    assert summary.shot_count == 1
    assert summary.shots[0].release_tick == 6
    assert summary.pending_commits == 1  # tick-8 commit never released
    assert math.isnan(
        summary.per_executor["teacher"].shots_on_target_rate
    )


def test_release_tracker_rejects_bad_streams() -> None:
    tracker = ReleaseAngleTracker()
    tracker.observe(
        ReleaseTickRecord(
            tick=0,
            executed_verb=0,
            executed_aim_bin=0,
            teacher_target_bin=0,
        )
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        tracker.observe(
            ReleaseTickRecord(
                tick=0,
                executed_verb=0,
                executed_aim_bin=0,
                teacher_target_bin=0,
            )
        )
    with pytest.raises(ValueError, match="executor"):
        tracker.observe(
            ReleaseTickRecord(
                tick=1,
                executed_verb=1,
                executed_aim_bin=0,
                teacher_target_bin=0,
                fire_edge_executor="oracle",
            )
        )


def test_wait_tick_execution_cannot_dilute_honest_accounting() -> None:
    """Teacher runs 95% of wait ticks but zero decision-relevant ticks."""

    accounting = HonestExecutionAccounting()
    for index in range(100):
        accounting.add(
            executor="teacher" if index < 95 else "student",
            teacher_verb=0,
            student_verb=0,
            phase="teacher_recovery",
        )
    for _ in range(20):
        accounting.add(
            executor="student",
            teacher_verb=1,
            student_verb=1,
            phase="student_nominal",
        )
    summary = accounting.summary()
    assert summary.total_ticks == 120
    assert summary.tick_teacher_fraction == pytest.approx(95 / 120)
    assert summary.decision_relevant_ticks == 20
    assert summary.decision_relevant_teacher_fraction == pytest.approx(0.0)
    assert summary.shot_count == 20
    assert summary.teacher_initiated_shot_fraction == pytest.approx(0.0)
    assert summary.phase_ticks == {
        "teacher_recovery": 100,
        "student_nominal": 20,
    }


def test_teacher_running_the_decisions_cannot_hide_behind_wait_ticks() -> None:
    """Inverse dilution: low tick fraction, full decision-relevant control."""

    accounting = HonestExecutionAccounting()
    for _ in range(970):
        accounting.add(
            executor="student",
            teacher_verb=0,
            student_verb=0,
            phase="student_nominal",
        )
    for _ in range(30):
        accounting.add(
            executor="teacher",
            teacher_verb=1,
            student_verb=0,
            phase="teacher_recovery",
        )
    summary = accounting.summary()
    assert summary.tick_teacher_fraction == pytest.approx(0.03)
    assert summary.decision_relevant_teacher_fraction == pytest.approx(1.0)
    assert summary.teacher_initiated_shot_fraction == pytest.approx(1.0)
    assert summary.teacher_initiated_shots == 30
    assert summary.student_initiated_shots == 0


def test_honest_accounting_empty_denominators_report_nan() -> None:
    accounting = HonestExecutionAccounting()
    accounting.add(
        executor="student",
        teacher_verb=0,
        student_verb=0,
        phase="student_nominal",
    )
    summary = accounting.summary()
    assert math.isnan(summary.decision_relevant_teacher_fraction)
    assert math.isnan(summary.teacher_initiated_shot_fraction)
    with pytest.raises(ValueError, match="executor"):
        accounting.add(
            executor="oracle",
            teacher_verb=0,
            student_verb=0,
            phase="student_nominal",
        )
