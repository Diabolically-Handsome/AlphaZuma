"""Override-window recovery controller with honest event-level accounting.

Lineage.  tools/alphazuma_55_recovery_intervention_v1.py defined the
semantic disagreement taxonomy and a consecutive-streak trigger that was
too sparse to fire usefully.  tools/alphazuma_55_recovery_intervention_v2.py
added rolling-window triggers plus deep recovery, but its handback demanded
stable_agreement_ticks=48 consecutive clean TICKS: under the measured
15-36% recovery-phase disagreement one clean window has probability
(1 - p) ** 48, roughly 4e-4 down to 1e-8 per window, so control almost
never returned and the probe's 73% teacher execution fraction was
mathematically forced rather than observed.
tools/alphazuma_55_recovery_intervention_v3.py kept the V2 machine but
executed the teacher only on semantic-disagreement ticks.  That collapses
the reported fraction to roughly the teacher's non-wait density (2.2-2.9%
of ticks) while the teacher still gates every shot -- a denominator hack --
and its 1-tick fire overrides are motor-invalid: through the elite-human
stack (src/zuma_rl/human_speedrun.py, reaction_delay_ticks=12;
revenge_core.py's FIRING state releases after 6 ticks) a fire intent
issued at tick t releases its projectile near t+18 at whatever angle the
slew-limited cursor holds THEN, which is the student-driven angle on
every surrounding tick.

V4 controller contract:

1.  Override windows.  When the controller decides the teacher acts --
    a deep-recovery trigger or a fire override -- the teacher controls
    BOTH verb and aim on EVERY tick until the window closes.  A window
    containing a teacher fire edge cannot close before
    edge + reaction_delay_ticks + fire_animation_ticks +
    release_margin_ticks (12 + 6 + 4 = 22 by default).  The lock
    protects the release TAIL: after the window opens no student
    intent enters the executed stream, so nothing the student proposes
    post-commit can reach the slew-limited cursor before the projectile
    samples it.  It does NOT retract the HEAD: student aim intents
    issued during the reaction_delay_ticks BEFORE the window opened are
    already in flight and keep steering the cursor for the first
    reaction_delay_ticks window ticks, so the first override shot can
    still release off-target when the cursor was moving under those
    in-flight commands.  Such shots are never hidden: the W1
    ReleaseAngleTracker attributes them to the teacher and reports
    their release_angle_error_bins honestly.  Every window tick is
    teacher execution in the accounting: agreement ticks inside a window
    are never re-attributed to the student.
2.  Event-level triggers and handback.  Triggers and handback advance on
    decision-relevant events (tools/alphazuma_55_closed_loop_metrics_v1
    classify_tick: either side issues a non-wait verb), never on raw
    wait-dominated ticks.  Deep recovery opens when
    trigger_disagreement_events_k of the last trigger_window_events_n
    decision-relevant events disagreed, or on the V2 score-stall /
    chain-growth triggers.  Handback needs handback_agreement_events_m
    consecutive decision-relevant agreements plus minimum window ticks:
    handback_probability(0.25, 4) = 0.75 ** 4 ~= 0.316 per window under
    25% event-level disagreement, instead of V2's
    handback_probability(0.15, 48) ~= 4e-4.
3.  Veto accounting.  A student fire proposed on a window tick whose
    executed teacher verb is not fire increments student_fire_vetoes,
    reported in every decision, window record, and summary -- suppressed
    student shots are never hidden.
4.  Decision gate.  evaluate_decision_gate keeps the V2 win/score checks
    and replaces the diluted tick-fraction check with
    decision_relevant_teacher_fraction <= 0.60 AND
    teacher_initiated_shot_fraction <= 0.60.  tick_teacher_fraction is
    reported for continuity but never gated, and NaN fractions fail
    closed, so neither wait-tick padding nor an empty denominator can
    pass the gate.

This module never calls an environment or a policy.  The controller is a
pure state machine so the intervention contract stays deterministic and
independently testable.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from tools.alphazuma_55_closed_loop_metrics_v1 import (
    FIRE_VERB,
    TickClass,
    WAIT_VERB,
    classify_tick,
)
from tools.alphazuma_55_recovery_intervention_v2 import (
    RecoveryInterventionControllerV2,
)


DEEP_RECOVERY_WINDOW = "deep_recovery"
FIRE_OVERRIDE_WINDOW = "fire_override"


def handback_probability(
    event_disagreement_rate: float,
    consecutive_agreement_events: int,
) -> float:
    """Probability that one handback window is a clean-agreement run.

    V2 required 48 consecutive clean TICKS.  Under the measured 15-36%
    recovery-phase disagreement that is handback_probability(0.15, 48)
    ~= 4.1e-4 down to handback_probability(0.36, 48) ~= 5e-10 per
    window, so handback almost never happened and the reported 73%
    teacher execution fraction was forced by the arithmetic.  V4
    requires handback_agreement_events_m consecutive clean
    decision-relevant EVENTS: under 25% event-level disagreement,
    handback_probability(0.25, 4) = 0.75 ** 4 ~= 0.316 per window.
    """

    rate = float(event_disagreement_rate)
    if not 0.0 <= rate <= 1.0:
        raise ValueError("event_disagreement_rate must lie in [0, 1]")
    events = int(consecutive_agreement_events)
    if events < 1:
        raise ValueError("consecutive_agreement_events must be positive")
    return (1.0 - rate) ** events


@dataclass(frozen=True)
class RecoveryInterventionConfigV4:
    """Frozen thresholds for the V4 override-window controller.

    reaction_delay_ticks, fire_animation_ticks, and release_margin_ticks
    describe the wrapped elite-human stack: a teacher fire edge at tick
    t releases its projectile near t + reaction_delay + fire_animation,
    so a window holding that edge stays teacher-controlled through
    t + fire_lock_ticks, which protects the release from POST-commit
    student aim.  Student aim intents still inside the 12-tick
    reaction-delay pipeline when the window opens CAN deflect the first
    override shot (the settle gate the pure-teacher stack relies on is
    position-only and passes a cursor sweeping through the target);
    the closed-loop release-angle attribution reports such shots as
    off-target teacher shots instead of hiding them.
    """

    trigger_disagreement_events_k: int = 3
    trigger_window_events_n: int = 4
    handback_agreement_events_m: int = 4
    minimum_teacher_ticks: int = 96
    maximum_teacher_ticks: int = 900
    handoff_cooldown_ticks: int = 96
    minimum_score_gain: int = 300
    minimum_chain_reduction: int = 6
    score_stall_trigger_ticks: int = 300
    chain_growth_trigger: int = 4
    fire_aim_disagreement_bins: int = 4
    reaction_delay_ticks: int = 12
    fire_animation_ticks: int = 6
    release_margin_ticks: int = 4
    fire_override_enabled: bool = True

    def __post_init__(self) -> None:
        positive = {
            "trigger_disagreement_events_k": self.trigger_disagreement_events_k,
            "trigger_window_events_n": self.trigger_window_events_n,
            "handback_agreement_events_m": self.handback_agreement_events_m,
            "minimum_teacher_ticks": self.minimum_teacher_ticks,
            "maximum_teacher_ticks": self.maximum_teacher_ticks,
            "handoff_cooldown_ticks": self.handoff_cooldown_ticks,
            "minimum_score_gain": self.minimum_score_gain,
            "minimum_chain_reduction": self.minimum_chain_reduction,
            "score_stall_trigger_ticks": self.score_stall_trigger_ticks,
            "chain_growth_trigger": self.chain_growth_trigger,
            "fire_aim_disagreement_bins": self.fire_aim_disagreement_bins,
            "reaction_delay_ticks": self.reaction_delay_ticks,
            "fire_animation_ticks": self.fire_animation_ticks,
        }
        for name, value in positive.items():
            if int(value) < 1:
                raise ValueError(f"{name} must be positive")
        if int(self.release_margin_ticks) < 0:
            raise ValueError("release_margin_ticks cannot be negative")
        if self.trigger_disagreement_events_k > self.trigger_window_events_n:
            raise ValueError(
                "trigger_disagreement_events_k cannot exceed its event window"
            )
        if self.minimum_teacher_ticks > self.maximum_teacher_ticks:
            raise ValueError(
                "minimum_teacher_ticks cannot exceed maximum_teacher_ticks"
            )
        if self.fire_aim_disagreement_bins > 90:
            raise ValueError("fire aim disagreement threshold cannot exceed 90")

    @property
    def fire_lock_ticks(self) -> int:
        """Ticks a window must stay open after a teacher fire edge."""

        return (
            int(self.reaction_delay_ticks)
            + int(self.fire_animation_ticks)
            + int(self.release_margin_ticks)
        )

    def contract(self) -> dict[str, int | bool]:
        value: dict[str, int | bool] = {
            name: (bool(field) if isinstance(field, bool) else int(field))
            for name, field in asdict(self).items()
        }
        value["fire_lock_ticks"] = self.fire_lock_ticks
        return value


@dataclass(frozen=True)
class InterventionDecisionV4:
    """One deterministic execution decision with window accounting."""

    execute_teacher: bool
    phase: str
    window_kind: str | None
    decision_relevant: bool
    semantic_disagreement: bool
    disagreement_reason: str | None
    trigger_reason: str | None
    window_opened: bool
    window_closed: bool
    window_index: int
    teacher_ticks_in_window: int
    fire_lock_remaining_ticks: int
    student_fire_vetoed: bool
    post_handoff_tick: int


class RecoveryInterventionControllerV4(RecoveryInterventionControllerV2):
    """Student-first control with kinematically valid override windows.

    Reuses the V2 machinery where its semantics survived audit: the
    info-metric bookkeeping (_metric/_update_metrics) and the deep
    recovery boundary (_recovered).  The V2 rolling per-tick trigger,
    tick-streak handback, and per-tick override attribution are all
    replaced -- see the module docstring for why each one failed.
    """

    def __init__(self, config: RecoveryInterventionConfigV4) -> None:
        if not isinstance(config, RecoveryInterventionConfigV4):
            raise TypeError("controller requires RecoveryInterventionConfigV4")
        super().__init__(config)

    def reset(self) -> None:
        super().reset()
        self.window_kind: str | None = None
        self.event_flags: deque[bool] = deque(
            maxlen=self.config.trigger_window_events_n
        )
        self.agreement_event_streak = 0
        self.fire_lock_until: int | None = None
        self.window_fire_edges = 0
        self.window_vetoes = 0
        self.window_decision_relevant_events = 0
        self.student_fire_vetoes = 0
        self.teacher_fire_edges = 0
        self.agreement_handbacks = 0
        self._last_executed_verb = WAIT_VERB

    def _classify(self, teacher_action, student_action) -> TickClass:
        return classify_tick(
            teacher_action,
            student_action,
            fire_aim_threshold=self.config.fire_aim_disagreement_bins,
        )

    def _trigger(self, tick_class: TickClass) -> tuple[str, str] | None:
        """Return (trigger_reason, window_kind) when a window must open.

        Deep recovery outranks the fire override because it opens the
        longer, minimum_teacher_ticks-bounded window; a lone teacher
        fire the student omits or mis-aims opens the shorter bounded
        fire-override window so the shot is taken kinematically intact.
        """

        has_error = any(self.event_flags)
        stalled = self.latest_tick - self.last_progress_tick
        low_chain = (
            self.latest_chain
            if self.low_chain_since_progress is None
            else self.low_chain_since_progress
        )
        if (
            has_error
            and self.latest_chain - low_chain >= self.config.chain_growth_trigger
        ):
            return (
                "decision_relevant_disagreement_with_chain_growth",
                DEEP_RECOVERY_WINDOW,
            )
        if has_error and stalled >= self.config.score_stall_trigger_ticks:
            return (
                "decision_relevant_disagreement_with_score_stall",
                DEEP_RECOVERY_WINDOW,
            )
        if (
            sum(self.event_flags)
            >= self.config.trigger_disagreement_events_k
        ):
            return (
                "event_budget_k_of_n_disagreements",
                DEEP_RECOVERY_WINDOW,
            )
        if (
            self.config.fire_override_enabled
            and tick_class.disagreement
            and tick_class.teacher_verb == FIRE_VERB
        ):
            return ("teacher_fire_override", FIRE_OVERRIDE_WINDOW)
        return None

    def _close_window(self, *, forced: bool) -> None:
        recovered, recovery_reason = self._recovered()
        if forced:
            self.forced_handoffs += 1
        elif recovered:
            self.recovery_successes += 1
        else:
            self.agreement_handbacks += 1
        self.completed_interventions.append(
            {
                "intervention_index": self.intervention_count,
                "window_kind": self.window_kind,
                "teacher_ticks": self.teacher_ticks,
                "start_score": self.intervention_start_score,
                "end_score": self.latest_score,
                "score_gain": self.latest_score - self.intervention_start_score,
                "start_chain_length": self.intervention_start_chain,
                "end_chain_length": self.latest_chain,
                "chain_reduction": (
                    self.intervention_start_chain - self.latest_chain
                ),
                "recovered": bool(recovered and not forced),
                "recovery_reason": None if forced else recovery_reason,
                "forced_maximum_teacher_ticks": forced,
                "decision_relevant_events": self.window_decision_relevant_events,
                "handback_agreement_event_streak": self.agreement_event_streak,
                "teacher_fire_edges": self.window_fire_edges,
                "student_fire_vetoes": self.window_vetoes,
            }
        )
        self.phase = "handoff"
        self.window_kind = None
        self.cooldown_remaining = self.config.handoff_cooldown_ticks
        self.cooldown_elapsed = 0
        self.teacher_ticks = 0
        self.agreement_event_streak = 0
        self.fire_lock_until = None
        self.window_fire_edges = 0
        self.window_vetoes = 0
        self.window_decision_relevant_events = 0
        self.event_flags.clear()

    def _window_tick(
        self,
        tick: int,
        tick_class: TickClass,
        *,
        opened: bool,
        trigger_reason: str | None,
    ) -> InterventionDecisionV4:
        self.teacher_ticks += 1
        executed_verb = tick_class.teacher_verb
        fire_edge = (
            executed_verb == FIRE_VERB
            and self._last_executed_verb != FIRE_VERB
        )
        if fire_edge:
            lock_until = tick + self.config.fire_lock_ticks
            self.fire_lock_until = (
                lock_until
                if self.fire_lock_until is None
                else max(self.fire_lock_until, lock_until)
            )
            self.window_fire_edges += 1
            self.teacher_fire_edges += 1
        self._last_executed_verb = executed_verb
        vetoed = (
            tick_class.student_verb == FIRE_VERB
            and executed_verb != FIRE_VERB
        )
        if vetoed:
            self.window_vetoes += 1
            self.student_fire_vetoes += 1
        if tick_class.decision_relevant:
            self.window_decision_relevant_events += 1
            if tick_class.disagreement:
                self.agreement_event_streak = 0
            else:
                self.agreement_event_streak += 1
        lock_remaining = (
            0
            if self.fire_lock_until is None
            else max(0, self.fire_lock_until - tick)
        )
        minimum_ticks = (
            self.config.minimum_teacher_ticks
            if self.window_kind == DEEP_RECOVERY_WINDOW
            else self.config.fire_lock_ticks
        )
        handback = (
            self.agreement_event_streak
            >= self.config.handback_agreement_events_m
            and self.teacher_ticks >= minimum_ticks
            and lock_remaining == 0
        )
        forced = (
            self.teacher_ticks >= self.config.maximum_teacher_ticks
            and lock_remaining == 0
        )
        ended = bool(forced or handback)
        window_kind = self.window_kind
        window_index = self.intervention_count
        teacher_ticks = self.teacher_ticks
        if ended:
            self._close_window(forced=forced)
        return InterventionDecisionV4(
            execute_teacher=True,
            phase=(
                "teacher_deep_recovery_window"
                if window_kind == DEEP_RECOVERY_WINDOW
                else "teacher_fire_override_window"
            ),
            window_kind=window_kind,
            decision_relevant=tick_class.decision_relevant,
            semantic_disagreement=tick_class.disagreement,
            disagreement_reason=tick_class.disagreement_reason,
            trigger_reason=trigger_reason,
            window_opened=opened,
            window_closed=ended,
            window_index=window_index,
            teacher_ticks_in_window=teacher_ticks,
            fire_lock_remaining_ticks=lock_remaining,
            student_fire_vetoed=vetoed,
            post_handoff_tick=0,
        )

    def _open_window(
        self,
        tick: int,
        tick_class: TickClass,
        *,
        kind: str,
        trigger_reason: str,
    ) -> InterventionDecisionV4:
        self.phase = "teacher"
        self.window_kind = kind
        self.intervention_count += 1
        self.intervention_start_score = self.latest_score
        self.intervention_start_chain = self.latest_chain
        self.teacher_ticks = 0
        self.agreement_event_streak = 0
        self.fire_lock_until = None
        self.window_fire_edges = 0
        self.window_vetoes = 0
        self.window_decision_relevant_events = 0
        self.event_flags.clear()
        return self._window_tick(
            tick,
            tick_class,
            opened=True,
            trigger_reason=trigger_reason,
        )

    def _student_decision(
        self,
        tick_class: TickClass,
        *,
        phase: str,
        post_handoff_tick: int = 0,
    ) -> InterventionDecisionV4:
        self._last_executed_verb = tick_class.student_verb
        return InterventionDecisionV4(
            execute_teacher=False,
            phase=phase,
            window_kind=None,
            decision_relevant=tick_class.decision_relevant,
            semantic_disagreement=tick_class.disagreement,
            disagreement_reason=tick_class.disagreement_reason,
            trigger_reason=None,
            window_opened=False,
            window_closed=False,
            window_index=self.intervention_count,
            teacher_ticks_in_window=0,
            fire_lock_remaining_ticks=0,
            student_fire_vetoed=False,
            post_handoff_tick=post_handoff_tick,
        )

    def decide(
        self,
        *,
        tick: int,
        teacher_action: Sequence[int],
        student_action: Sequence[int],
        previous_info: Mapping[str, Any],
        active: bool = True,
    ) -> InterventionDecisionV4:
        """Choose the executor for the current decision tick."""

        tick = int(tick)
        self._update_metrics(tick, previous_info)
        tick_class = self._classify(teacher_action, student_action)
        if not active:
            return InterventionDecisionV4(
                execute_teacher=False,
                phase="inactive",
                window_kind=None,
                decision_relevant=tick_class.decision_relevant,
                semantic_disagreement=tick_class.disagreement,
                disagreement_reason=tick_class.disagreement_reason,
                trigger_reason=None,
                window_opened=False,
                window_closed=False,
                window_index=self.intervention_count,
                teacher_ticks_in_window=0,
                fire_lock_remaining_ticks=0,
                student_fire_vetoed=False,
                post_handoff_tick=0,
            )

        if self.phase == "teacher":
            return self._window_tick(
                tick,
                tick_class,
                opened=False,
                trigger_reason=None,
            )

        if self.phase == "handoff":
            self.cooldown_elapsed += 1
            self.cooldown_remaining -= 1
            post_handoff_tick = self.cooldown_elapsed
            if self.cooldown_remaining <= 0:
                self.phase = "student"
                self.cooldown_remaining = 0
                self.cooldown_elapsed = 0
            return self._student_decision(
                tick_class,
                phase="student_post_handoff",
                post_handoff_tick=post_handoff_tick,
            )

        if tick_class.decision_relevant:
            self.event_flags.append(tick_class.disagreement)
        trigger = self._trigger(tick_class)
        if trigger is not None:
            trigger_reason, kind = trigger
            return self._open_window(
                tick,
                tick_class,
                kind=kind,
                trigger_reason=trigger_reason,
            )
        return self._student_decision(
            tick_class,
            phase=(
                "student_event_pressure"
                if any(self.event_flags)
                else "student_nominal"
            ),
        )

    def summary(self) -> dict[str, Any]:
        return {
            "config": self.config.contract(),
            "interventions": self.intervention_count,
            "recovery_successes": self.recovery_successes,
            "agreement_handbacks": self.agreement_handbacks,
            "forced_handoffs": self.forced_handoffs,
            "student_fire_vetoes": self.student_fire_vetoes,
            "teacher_fire_edges": self.teacher_fire_edges,
            "completed_interventions": list(self.completed_interventions),
            "terminal_phase": self.phase,
            "terminal_window_kind": self.window_kind,
            "terminal_agreement_event_streak": self.agreement_event_streak,
            "execution_rule": (
                "teacher_controls_verb_and_aim_on_every_window_tick"
            ),
            "handback_rule": (
                "m_consecutive_decision_relevant_agreement_events"
                "_plus_minimum_window_ticks_and_fire_lock_expiry"
            ),
        }


@dataclass(frozen=True)
class DecisionGateConfigV4:
    """Frozen probe acceptance thresholds for the V4 mechanism gate.

    Keeps the V2 win/score checks and replaces the diluted per-tick
    teacher-fraction check with the two honest fractions from
    tools/alphazuma_55_closed_loop_metrics_v1.  tick_teacher_fraction is
    reported for continuity with V1-V3 receipts but is never gated.
    """

    minimum_paired_score_improvements: int = 6
    minimum_interventions: int = 6
    minimum_recovery_successes: int = 1
    maximum_decision_relevant_teacher_fraction: float = 0.60
    maximum_teacher_initiated_shot_fraction: float = 0.60

    def __post_init__(self) -> None:
        for name, value in (
            (
                "minimum_paired_score_improvements",
                self.minimum_paired_score_improvements,
            ),
            ("minimum_interventions", self.minimum_interventions),
            ("minimum_recovery_successes", self.minimum_recovery_successes),
        ):
            if int(value) < 1:
                raise ValueError(f"{name} must be positive")
        for name, value in (
            (
                "maximum_decision_relevant_teacher_fraction",
                self.maximum_decision_relevant_teacher_fraction,
            ),
            (
                "maximum_teacher_initiated_shot_fraction",
                self.maximum_teacher_initiated_shot_fraction,
            ),
        ):
            if not 0.0 < float(value) <= 1.0:
                raise ValueError(f"{name} must lie in (0, 1]")

    def contract(self) -> dict[str, int | float]:
        return {
            name: (int(value) if isinstance(value, int) else float(value))
            for name, value in asdict(self).items()
        }


def evaluate_decision_gate(
    *,
    gate: DecisionGateConfigV4,
    student_only_wins: int,
    student_only_total_score: int,
    intervention_wins: int,
    intervention_total_score: int,
    paired_score_deltas: Sequence[int],
    interventions: int,
    recovery_successes: int,
    tick_teacher_fraction: float,
    decision_relevant_teacher_fraction: float,
    teacher_initiated_shot_fraction: float,
) -> dict[str, Any]:
    """Evaluate the V4 mechanism gate with anti-dilution fractions.

    The two gated fractions must come from honest denominators
    (decision-relevant ticks and initiated shots).  A NaN fraction --
    the tools/alphazuma_55_closed_loop_metrics_v1 signal for an empty
    denominator -- fails its comparison, so a stream with no decisions
    or no shots can never pass vacuously.  A V3-style stream whose
    tick_teacher_fraction looks tiny while the teacher gates every shot
    fails on both honest fractions.
    """

    decision_fraction = float(decision_relevant_teacher_fraction)
    shot_fraction = float(teacher_initiated_shot_fraction)
    checks = {
        "intervention_wins_exceed_student_only": bool(
            int(intervention_wins) > int(student_only_wins)
        ),
        "intervention_total_score_exceeds_student_only": bool(
            int(intervention_total_score) > int(student_only_total_score)
        ),
        "paired_score_improvements": bool(
            sum(1 for delta in paired_score_deltas if int(delta) > 0)
            >= gate.minimum_paired_score_improvements
        ),
        "minimum_interventions": bool(
            int(interventions) >= gate.minimum_interventions
        ),
        "minimum_recovery_successes": bool(
            int(recovery_successes) >= gate.minimum_recovery_successes
        ),
        "maximum_decision_relevant_teacher_fraction": bool(
            decision_fraction
            <= gate.maximum_decision_relevant_teacher_fraction
        ),
        "maximum_teacher_initiated_shot_fraction": bool(
            shot_fraction <= gate.maximum_teacher_initiated_shot_fraction
        ),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "gate": gate.contract(),
        "reported_not_gated": {
            "tick_teacher_fraction": float(tick_teacher_fraction),
        },
    }
