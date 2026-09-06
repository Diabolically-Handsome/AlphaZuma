from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from tools import probe_alphazuma_55_recovery_intervention_v4 as probe
from tools.alphazuma_55_closed_loop_metrics_v1 import (
    ReleaseAngleConfig,
    ReleaseAngleTracker,
    ReleaseTickRecord,
)
from tools.alphazuma_55_recovery_intervention_v2 import (
    RecoveryInterventionConfigV2,
)
from tools.alphazuma_55_recovery_intervention_v4 import (
    DecisionGateConfigV4,
    RecoveryInterventionConfigV4,
    RecoveryInterventionControllerV4,
    evaluate_decision_gate,
    handback_probability,
)


def _config(**overrides: Any) -> RecoveryInterventionConfigV4:
    values: dict[str, Any] = {
        "trigger_disagreement_events_k": 3,
        "trigger_window_events_n": 4,
        "handback_agreement_events_m": 2,
        "minimum_teacher_ticks": 5,
        "maximum_teacher_ticks": 2000,
        "handoff_cooldown_ticks": 5,
        "minimum_score_gain": 1000,
        "minimum_chain_reduction": 100,
        "score_stall_trigger_ticks": 100000,
        "chain_growth_trigger": 1000,
        "fire_aim_disagreement_bins": 4,
        "reaction_delay_ticks": 12,
        "fire_animation_ticks": 6,
        "release_margin_ticks": 4,
        "fire_override_enabled": True,
    }
    values.update(overrides)
    return RecoveryInterventionConfigV4(**values)


def _decide(
    controller: RecoveryInterventionControllerV4,
    tick: int,
    teacher: tuple[int, int],
    student: tuple[int, int],
) -> Any:
    return controller.decide(
        tick=tick,
        teacher_action=teacher,
        student_action=student,
        previous_info={"score": 0, "chain_length": 5},
    )


def test_config_contract_and_validation() -> None:
    config = RecoveryInterventionConfigV4()
    assert config.fire_lock_ticks == 12 + 6 + 4
    contract = config.contract()
    assert contract["fire_lock_ticks"] == 22
    assert contract["fire_override_enabled"] is True
    assert contract["handback_agreement_events_m"] == 4
    with pytest.raises(ValueError, match="event window"):
        _config(trigger_disagreement_events_k=5, trigger_window_events_n=4)
    with pytest.raises(TypeError, match="ConfigV4"):
        RecoveryInterventionControllerV4(RecoveryInterventionConfigV2())


def test_handback_probability_is_realistic_at_event_level() -> None:
    """V2 needed 48 clean ticks; V4 needs 4 clean decision-relevant events.

    Under the measured 15-36% recovery-phase disagreement, one V2
    handback window succeeded with probability (1-p)**48 in roughly
    4e-4 down to 1e-8, so the observed 73% teacher execution fraction
    was mathematically forced.  The V4 event-level rule hands back with
    probability 0.75**4 ~= 0.316 per window at 25% event disagreement.
    """

    assert handback_probability(0.25, 4) == pytest.approx(0.31640625)
    assert handback_probability(0.25, 4) > 0.30
    assert handback_probability(0.15, 48) < 5e-4
    assert handback_probability(0.36, 48) < 1e-8
    with pytest.raises(ValueError, match="lie in"):
        handback_probability(1.5, 4)
    with pytest.raises(ValueError, match="positive"):
        handback_probability(0.5, 0)


def test_fire_override_window_protects_release_kinematics() -> None:
    """A teacher fire committed at t releases at t+18 under teacher aim.

    The window opens on the teacher-fire disagreement, the teacher
    controls verb AND aim on every tick, and the fire lock keeps the
    window open through commit + 12 + 6 + 4 = 22 ticks.  The synthetic
    cursor here follows the EXECUTED INTENT STREAM directly, so this
    test verifies the TAIL guarantee only: no post-commit student
    intent enters the executed stream before the release.  The physical
    HEAD limitation -- pre-commit student intents still in the
    reaction-delay pipeline steering the slew-limited cursor -- is
    covered by
    test_fire_override_head_deflection_is_observed_not_hidden.
    """

    controller = RecoveryInterventionControllerV4(_config())
    tracker = ReleaseAngleTracker(
        ReleaseAngleConfig(commit_to_release_ticks=18, tolerance_bins=2)
    )
    teacher_bin = 40
    student_bin = 130
    decisions: dict[int, Any] = {}
    executed_bins: dict[int, int] = {}
    for tick in range(23):
        if tick == 0:
            teacher, student = (1, teacher_bin), (0, student_bin)
        elif tick in (20, 21):
            teacher, student = (2, teacher_bin), (2, teacher_bin)
        else:
            teacher, student = (0, teacher_bin), (0, student_bin)
        decision = _decide(controller, tick, teacher, student)
        decisions[tick] = decision
        executed_bin = (
            teacher_bin if decision.execute_teacher else student_bin
        )
        executed_bins[tick] = executed_bin
        tracker.observe(
            ReleaseTickRecord(
                tick=tick,
                executed_verb=(
                    teacher[0] if decision.execute_teacher else student[0]
                ),
                executed_aim_bin=executed_bin,
                teacher_target_bin=teacher_bin,
                fire_edge_executor="teacher" if tick == 0 else None,
            )
        )

    opened = decisions[0]
    assert opened.window_opened is True
    assert opened.window_kind == "fire_override"
    assert opened.trigger_reason == "teacher_fire_override"
    assert opened.fire_lock_remaining_ticks == 22
    # ALL window ticks execute the teacher, including both-wait ticks.
    assert all(decisions[tick].execute_teacher for tick in range(23))
    assert all(
        decisions[tick].decision_relevant is False
        for tick in range(1, 20)
    )
    # m=2 agreement events are reached at tick 21, but the fire lock
    # forbids closing before commit + 22.
    assert decisions[21].window_closed is False
    assert decisions[21].fire_lock_remaining_ticks == 1
    assert decisions[22].window_closed is True
    assert decisions[22].fire_lock_remaining_ticks == 0
    # No student contamination in the executed intent stream before the
    # release (post-commit protection; the pre-commit pipeline is the
    # head-deflection test's subject).
    assert {executed_bins[tick] for tick in range(19)} == {teacher_bin}
    summary = tracker.summary()
    assert summary.shot_count == 1
    shot = summary.shots[0]
    assert shot.commit_tick == 0
    assert shot.release_tick == 18
    assert shot.release_aim_bin == teacher_bin
    assert shot.release_angle_error_bins == 0
    assert shot.on_target is True
    assert summary.per_executor["teacher"].shot_count == 1
    controller_summary = controller.summary()
    row = controller_summary["completed_interventions"][0]
    assert row["window_kind"] == "fire_override"
    assert row["teacher_fire_edges"] == 1
    assert row["student_fire_vetoes"] == 0
    assert controller_summary["agreement_handbacks"] == 1
    after = _decide(controller, 23, (0, teacher_bin), (0, student_bin))
    assert after.execute_teacher is False
    assert after.phase == "student_post_handoff"


def test_fire_override_head_deflection_is_observed_not_hidden() -> None:
    """Pre-commit in-flight student aim deflects the first override shot.

    The reaction-delay pipeline cannot be retracted: at window-open tick
    t the student intents issued at t-12..t-1 are already in flight and
    drive the slew-limited cursor for ticks t..t+11 -- 12 of the 18
    ticks before release.  Teacher intents reach the cursor only at
    t+12, leaving 6 slew-limited ticks (here modeled at the 5.4
    bins/tick ceiling, i.e. MORE correction than the acceleration-
    limited real cursor achieves) before the ~t+18 release.  The
    override shot therefore releases tens of bins off the teacher
    target.  The contract is honesty, not impossibility: the W1
    ReleaseAngleTracker must attribute the shot to the teacher and
    report it off-target instead of hiding it.
    """

    reaction_delay_ticks = 12
    slew_limit_bins_per_tick = 5.4  # 1080 deg/s at 2 deg/bin, 10 ms ticks
    aim_bins = 180
    controller = RecoveryInterventionControllerV4(_config())
    tracker = ReleaseAngleTracker(
        ReleaseAngleConfig(commit_to_release_ticks=18, tolerance_bins=2)
    )
    teacher_bin = 40
    student_bin = 130
    # The student parked bin 130 long before the window: every intent in
    # the delay pipeline at window open commands 130.
    intent_log: dict[int, int] = {
        tick: student_bin for tick in range(-reaction_delay_ticks, 0)
    }
    cursor = float(student_bin)
    for tick in range(23):
        teacher = (1, teacher_bin) if tick == 0 else (0, teacher_bin)
        student = (0, student_bin)
        decision = _decide(controller, tick, teacher, student)
        assert decision.execute_teacher is True
        # Teacher owns the executed intent stream from the first window
        # tick -- but the CURSOR obeys the intent issued 12 ticks ago.
        intent_log[tick] = teacher_bin
        target = float(intent_log[tick - reaction_delay_ticks])
        delta = target - cursor
        if delta > aim_bins / 2:
            delta -= aim_bins
        elif delta < -aim_bins / 2:
            delta += aim_bins
        delta = max(
            -slew_limit_bins_per_tick, min(slew_limit_bins_per_tick, delta)
        )
        cursor = (cursor + delta) % aim_bins
        tracker.observe(
            ReleaseTickRecord(
                tick=tick,
                executed_verb=teacher[0],
                executed_aim_bin=int(round(cursor)) % aim_bins,
                teacher_target_bin=teacher_bin,
                fire_edge_executor="teacher" if tick == 0 else None,
            )
        )

    summary = tracker.summary()
    assert summary.shot_count == 1
    shot = summary.shots[0]
    assert shot.commit_tick == 0
    assert shot.release_tick == 18
    # Ticks 0..11: cursor held at 130 by in-flight student intents.
    # Ticks 12..18: 7 slew steps of 5.4 bins toward 40 -> 130 - 37.8 =
    # 92.2, rounded to bin 92: 52 bins off the teacher target.
    assert shot.release_aim_bin == 92
    assert shot.release_angle_error_bins == 52
    assert shot.on_target is False
    # Honest attribution: an off-target TEACHER shot, visible in the
    # per-executor breakdown the probe reports.
    assert summary.per_executor["teacher"].shot_count == 1
    assert summary.per_executor["teacher"].shots_on_target_rate == 0.0


def test_student_fires_suppressed_in_window_are_counted() -> None:
    controller = RecoveryInterventionControllerV4(_config())
    opened = _decide(controller, 0, (1, 40), (0, 130))
    assert opened.window_opened is True
    vetoed = _decide(controller, 1, (0, 40), (1, 130))
    assert vetoed.execute_teacher is True
    assert vetoed.student_fire_vetoed is True
    executed_fire = _decide(controller, 2, (1, 40), (1, 40))
    assert executed_fire.student_fire_vetoed is False
    assert controller.student_fire_vetoes == 1
    summary = controller.summary()
    assert summary["student_fire_vetoes"] == 1
    assert summary["teacher_fire_edges"] == 2


def test_event_budget_triggers_on_events_not_ticks() -> None:
    """3 of the last 4 decision-relevant events trigger deep recovery
    even when the events are separated by hundreds of both-wait ticks;
    V2's 48-tick rolling window would have silently expired them.
    """

    controller = RecoveryInterventionControllerV4(
        _config(fire_override_enabled=False)
    )

    first = _decide(controller, 0, (2, 10), (3, 10))
    assert first.execute_teacher is False
    assert first.window_opened is False
    for tick in range(1, 151):
        decision = _decide(controller, tick, (0, 10), (0, 90))
        assert decision.execute_teacher is False
        assert decision.phase == "student_event_pressure"
    _decide(controller, 151, (2, 10), (2, 10))
    for tick in range(152, 301):
        assert _decide(controller, tick, (0, 10), (0, 90)).execute_teacher is False
    third = _decide(controller, 301, (2, 10), (3, 10))
    assert third.window_opened is False
    trigger = _decide(controller, 302, (2, 10), (3, 10))
    assert trigger.window_opened is True
    assert trigger.window_kind == "deep_recovery"
    assert trigger.trigger_reason == "event_budget_k_of_n_disagreements"


def test_deep_recovery_handback_needs_events_and_minimum_ticks() -> None:
    controller = RecoveryInterventionControllerV4(
        _config(fire_override_enabled=False)
    )
    for tick in range(3):
        decision = _decide(controller, tick, (2, 10), (3, 10))
    assert decision.window_opened is True
    assert decision.teacher_ticks_in_window == 1
    _decide(controller, 3, (2, 10), (2, 10))
    streak_met = _decide(controller, 4, (2, 10), (2, 10))
    assert streak_met.window_closed is False  # streak==m, ticks 3 < 5
    idle = _decide(controller, 5, (0, 10), (0, 10))
    assert idle.execute_teacher is True  # both-wait tick is still teacher
    assert idle.decision_relevant is False
    assert idle.window_closed is False  # teacher_ticks 4 < minimum 5
    closing = _decide(controller, 6, (0, 10), (0, 10))
    assert closing.execute_teacher is True
    assert closing.window_closed is True  # minimum met, streak preserved
    summary = controller.summary()
    assert summary["agreement_handbacks"] == 1
    assert summary["recovery_successes"] == 0
    row = summary["completed_interventions"][0]
    assert row["window_kind"] == "deep_recovery"
    assert row["handback_agreement_event_streak"] == 2
    assert row["teacher_fire_edges"] == 0
    assert row["forced_maximum_teacher_ticks"] is False


def test_forced_close_still_waits_for_fire_lock() -> None:
    controller = RecoveryInterventionControllerV4(
        _config(
            trigger_disagreement_events_k=1,
            trigger_window_events_n=1,
            minimum_teacher_ticks=2,
            maximum_teacher_ticks=30,
            handback_agreement_events_m=4,
            fire_override_enabled=False,
        )
    )
    opened = _decide(controller, 0, (2, 10), (3, 10))
    assert opened.window_opened is True
    for tick in range(1, 28):
        assert (
            _decide(controller, tick, (0, 10), (0, 10)).window_closed
            is False
        )
    edge = _decide(controller, 28, (1, 40), (0, 10))
    assert edge.window_closed is False
    assert edge.fire_lock_remaining_ticks == 22
    for tick in range(29, 50):
        decision = _decide(controller, tick, (0, 40), (0, 10))
        assert decision.execute_teacher is True
        assert decision.window_closed is False  # max reached, lock active
    final = _decide(controller, 50, (0, 40), (0, 10))
    assert final.window_closed is True
    assert controller.forced_handoffs == 1
    row = controller.summary()["completed_interventions"][0]
    assert row["forced_maximum_teacher_ticks"] is True


def _gate_kwargs(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "gate": DecisionGateConfigV4(),
        "student_only_wins": 2,
        "student_only_total_score": 10_000,
        "intervention_wins": 6,
        "intervention_total_score": 45_000,
        "paired_score_deltas": [500] * 8 + [-100] * 4,
        "interventions": 9,
        "recovery_successes": 2,
        "tick_teacher_fraction": 0.31,
        "decision_relevant_teacher_fraction": 0.42,
        "teacher_initiated_shot_fraction": 0.5,
    }
    values.update(overrides)
    return values


def test_gate_passes_on_honest_fractions() -> None:
    verdict = evaluate_decision_gate(**_gate_kwargs())
    assert verdict["status"] == "PASS"
    assert all(verdict["checks"].values())
    assert verdict["reported_not_gated"]["tick_teacher_fraction"] == 0.31


def test_gate_fails_v3_style_denominator_hack() -> None:
    """V3 receipt shape: ~2.5% tick fraction, every shot teacher-gated.

    The old gate compared tick_teacher_fraction <= 0.6 and would have
    passed this stream; the V4 gate reads the honest fractions and must
    fail it.
    """

    verdict = evaluate_decision_gate(
        **_gate_kwargs(
            tick_teacher_fraction=0.025,
            decision_relevant_teacher_fraction=0.97,
            teacher_initiated_shot_fraction=1.0,
        )
    )
    assert verdict["status"] == "FAIL"
    checks = verdict["checks"]
    assert checks["maximum_decision_relevant_teacher_fraction"] is False
    assert checks["maximum_teacher_initiated_shot_fraction"] is False
    others = {
        name: value
        for name, value in checks.items()
        if not name.startswith("maximum_")
    }
    assert others and all(others.values())


def test_gate_nan_fractions_fail_closed() -> None:
    verdict = evaluate_decision_gate(
        **_gate_kwargs(
            decision_relevant_teacher_fraction=float("nan"),
            teacher_initiated_shot_fraction=float("nan"),
        )
    )
    assert verdict["status"] == "FAIL"


class _StubVector:
    """Minimal SubprocVecEnv stand-in publishing human_speedrun info."""

    def __init__(self, count: int, episode_ticks: int) -> None:
        self.count = count
        self.episode_ticks = episode_ticks
        self._tick = 0

    def seed(self, value: int) -> None:
        self.seed_value = int(value)

    def reset(self) -> np.ndarray:
        self._tick = 0
        return np.zeros((self.count, 6), dtype=np.float32)

    def step(self, actions: np.ndarray) -> tuple[Any, Any, Any, Any]:
        actions = np.asarray(actions, dtype=np.int64)
        tick = self._tick
        self._tick += 1
        done = tick == self.episode_ticks - 1
        infos = []
        for index in range(self.count):
            verb = int(actions[index][0])
            aim = int(actions[index][1])
            outcome = ("win" if index == 0 else "loss") if done else None
            infos.append(
                {
                    "score": (tick + 1) * (index + 1),
                    "chain_length": 5,
                    "ticks": tick + 1,
                    "ticks_advanced": 1,
                    "outcome": outcome,
                    "native_outcome": outcome,
                    "TimeLimit.truncated": False,
                    "human_speedrun": {
                        "executed_verb": verb,
                        "executed_aim_bin": aim,
                        "desired_verb": verb,
                        "button_enqueued": verb == 1,
                    },
                }
            )
        observations = np.zeros((self.count, 6), dtype=np.float32)
        rewards = np.zeros(self.count, dtype=np.float64)
        dones = np.full(self.count, done, dtype=np.bool_)
        return observations, rewards, dones, infos

    def close(self) -> None:
        pass


class _StubModel:
    """Dead student: always waits while parking a wrong aim bin."""

    def __init__(self, count: int) -> None:
        self.count = count

    def predict(
        self,
        observations: Any,
        deterministic: bool = True,
        action_masks: Any = None,
    ) -> tuple[np.ndarray, None]:
        return np.tile(np.array([0, 10], dtype=np.int64), (self.count, 1)), None


class _StubTeacher:
    """Fires bin 40 every fifteenth decision, waits on bin 40 otherwise."""

    def __init__(self) -> None:
        self.calls = 0

    def act(self, observation: Any) -> tuple[int, int]:
        verb = 1 if self.calls % 15 == 0 else 0
        self.calls += 1
        return (verb, 40)


def test_probe_completion_schema_with_stubbed_env() -> None:
    level_ids = ("Jungle1", "Jungle9")
    config = RecoveryInterventionConfigV4()
    gate = DecisionGateConfigV4()
    modes = []
    for mode in probe.EXPECTED_MODES:
        result = probe._run_mode(
            mode=mode,
            model=_StubModel(len(level_ids)),
            vector=_StubVector(len(level_ids), 60),
            teachers=[_StubTeacher() for _ in level_ids],
            level_ids=level_ids,
            seed_base=probe.DEFAULT_SEED_BASE,
            max_ticks=100,
            controller_config=config,
            masks_provider=lambda vector: np.ones(
                (len(level_ids), 184), dtype=np.bool_
            ),
        )
        modes.append(result)
    decision = probe._decision(gate=gate, level_ids=level_ids, modes=modes)
    completion = probe._build_completion(
        modes=modes,
        decision=decision,
        source={"model_path": "stub://model", "model_sha256": "sha256:stub"},
        runtime={"python": "3.12", "device": "cpu"},
        wall_seconds=0.01,
        controller_config=config,
        gate=gate,
        preregistration=None,
    )
    assert completion["schema"] == (
        "zuma-rl.alphazuma-55-recovery-intervention-v4-probe-completion"
    )
    json.dumps(completion, allow_nan=False)

    by_mode = {row["mode"]: row for row in completion["modes"]}
    assert set(by_mode) == {"student_only", "recovery_intervention"}
    for row in by_mode.values():
        summary = row["summary"]
        for key in (
            "shots_on_target_rate",
            "mean_release_angle_error_bins",
            "aim_intent_error",
            "tick_teacher_fraction",
            "decision_relevant_teacher_fraction",
            "teacher_initiated_shot_fraction",
            "per_tick_disagreement_rate",
            "decision_relevant_disagreement_rate",
            "student_fire_vetoes",
        ):
            assert key in summary
        metrics = row["episodes"][0]["closed_loop_metrics"]
        for key in (
            "tick_classification",
            "aim_intent_error",
            "release_angle",
            "honest_execution",
        ):
            assert key in metrics

    student_only = by_mode["student_only"]["summary"]
    assert student_only["shot_count"] == 0
    assert student_only["shots_on_target_rate"] is None  # NaN -> null
    assert student_only["teacher_execution_steps"] == 0
    assert student_only["aim_intent_error"]["mean_bins"] == pytest.approx(30.0)

    intervention = by_mode["recovery_intervention"]["summary"]
    assert intervention["interventions"] >= 2
    assert intervention["shot_count"] > 0
    assert intervention["shots_on_target_rate"] == pytest.approx(1.0)
    assert intervention["teacher_initiated_shot_fraction"] == pytest.approx(1.0)
    assert intervention["student_fire_vetoes"] == 0
    assert intervention["teacher_fire_edges"] > 0

    assert completion["decision"]["status"] in {"PASS", "FAIL"}
    assert set(completion["decision"]["checks"]) == {
        "intervention_wins_exceed_student_only",
        "intervention_total_score_exceeds_student_only",
        "paired_score_improvements",
        "minimum_interventions",
        "minimum_recovery_successes",
        "maximum_decision_relevant_teacher_fraction",
        "maximum_teacher_initiated_shot_fraction",
    }
    assert completion["decision"]["reported_not_gated"][
        "tick_teacher_fraction"
    ] is not None
    assert completion["formal_seed_consumption"] is False
    assert completion["training_authority"] is False
