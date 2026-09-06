"""Pure state machine for recovery-focused AlphaZuma DAgger collection.

The student executes by default.  Persistent semantic disagreement is allowed
to develop for a short grace window so collection reaches policy-induced
states.  The teacher then executes a bounded recovery burst and hands control
back to the student for a protected post-recovery window.

This module never calls an environment or a policy.  Keeping the controller
pure makes the intervention contract deterministic and independently testable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class RecoveryInterventionConfig:
    """Frozen thresholds for one recovery-intervention controller."""

    disagreement_grace_ticks: int = 12
    minimum_teacher_ticks: int = 96
    maximum_teacher_ticks: int = 600
    stable_agreement_ticks: int = 24
    handoff_cooldown_ticks: int = 96
    minimum_score_gain: int = 30
    minimum_chain_reduction: int = 2
    score_stall_trigger_ticks: int = 300
    chain_growth_trigger: int = 4
    fire_aim_disagreement_bins: int = 4

    def __post_init__(self) -> None:
        positive = {
            "disagreement_grace_ticks": self.disagreement_grace_ticks,
            "minimum_teacher_ticks": self.minimum_teacher_ticks,
            "maximum_teacher_ticks": self.maximum_teacher_ticks,
            "stable_agreement_ticks": self.stable_agreement_ticks,
            "handoff_cooldown_ticks": self.handoff_cooldown_ticks,
            "minimum_score_gain": self.minimum_score_gain,
            "minimum_chain_reduction": self.minimum_chain_reduction,
            "score_stall_trigger_ticks": self.score_stall_trigger_ticks,
            "chain_growth_trigger": self.chain_growth_trigger,
            "fire_aim_disagreement_bins": self.fire_aim_disagreement_bins,
        }
        for name, value in positive.items():
            if int(value) < 1:
                raise ValueError(f"{name} must be positive")
        if self.minimum_teacher_ticks > self.maximum_teacher_ticks:
            raise ValueError(
                "minimum_teacher_ticks cannot exceed maximum_teacher_ticks"
            )
        if self.fire_aim_disagreement_bins > 90:
            raise ValueError("fire aim disagreement threshold cannot exceed 90")

    def contract(self) -> dict[str, int]:
        return {name: int(value) for name, value in asdict(self).items()}


@dataclass(frozen=True)
class InterventionDecision:
    """One deterministic execution decision and its collection annotation."""

    execute_teacher: bool
    phase: str
    semantic_disagreement: bool
    disagreement_reason: str | None
    trigger_reason: str | None
    intervention_started: bool
    intervention_ended: bool
    intervention_index: int
    teacher_ticks_in_intervention: int
    post_handoff_tick: int


def circular_bin_distance(left: int, right: int, bins: int = 180) -> int:
    """Return the shortest integer distance on a circular action lattice."""

    count = int(bins)
    if count < 2:
        raise ValueError("bins must be at least two")
    a = int(left)
    b = int(right)
    if not 0 <= a < count or not 0 <= b < count:
        raise ValueError("aim bin escaped the circular lattice")
    direct = abs(a - b)
    return min(direct, count - direct)


def semantic_action_disagreement(
    teacher_action: Sequence[int],
    student_action: Sequence[int],
    *,
    fire_aim_threshold: int,
    aim_bins: int = 180,
) -> tuple[bool, str | None]:
    """Compare actions only where their semantics affect the environment."""

    if len(teacher_action) != 2 or len(student_action) != 2:
        raise ValueError("teacher and student actions must each contain two values")
    teacher_verb, teacher_aim = (int(value) for value in teacher_action)
    student_verb, student_aim = (int(value) for value in student_action)
    if not 0 <= teacher_verb < 4 or not 0 <= student_verb < 4:
        raise ValueError("verb escaped the full55 action interface")
    if teacher_verb != student_verb:
        if teacher_verb != 0 and student_verb == 0:
            return True, "student_omits_teacher_verb"
        if teacher_verb == 0 and student_verb != 0:
            return True, "student_adds_unrequested_verb"
        return True, "verb_mismatch"
    if teacher_verb == 1:
        distance = circular_bin_distance(teacher_aim, student_aim, aim_bins)
        if distance >= int(fire_aim_threshold):
            return True, "fire_aim_mismatch"
    return False, None


class RecoveryInterventionController:
    """Student-first controller with bounded teacher recovery bursts."""

    def __init__(self, config: RecoveryInterventionConfig) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.phase = "student"
        self.disagreement_streak = 0
        self.teacher_ticks = 0
        self.agreement_streak = 0
        self.cooldown_remaining = 0
        self.cooldown_elapsed = 0
        self.intervention_count = 0
        self.intervention_start_score = 0
        self.intervention_start_chain = 0
        self.last_score = 0
        self.last_progress_tick = 0
        self.low_chain_since_progress: int | None = None
        self.latest_score = 0
        self.latest_chain = 0
        self.latest_tick = 0
        self.recovery_successes = 0
        self.forced_handoffs = 0
        self.completed_interventions: list[dict[str, Any]] = []

    @staticmethod
    def _metric(info: Mapping[str, Any], name: str, default: int) -> int:
        value = info.get(name, default)
        if isinstance(value, bool):
            return int(value)
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(default)

    def _update_metrics(self, tick: int, info: Mapping[str, Any]) -> None:
        score = self._metric(info, "score", self.latest_score)
        chain = max(0, self._metric(info, "chain_length", self.latest_chain))
        if score > self.last_score:
            self.last_score = score
            self.last_progress_tick = int(tick)
            self.low_chain_since_progress = chain
        elif self.low_chain_since_progress is None:
            self.low_chain_since_progress = chain
        else:
            self.low_chain_since_progress = min(
                self.low_chain_since_progress,
                chain,
            )
        self.latest_score = score
        self.latest_chain = chain
        self.latest_tick = int(tick)

    def _trigger_reason(self, disagreement_reason: str | None) -> str:
        stalled = self.latest_tick - self.last_progress_tick
        low_chain = (
            self.latest_chain
            if self.low_chain_since_progress is None
            else self.low_chain_since_progress
        )
        if self.latest_chain - low_chain >= self.config.chain_growth_trigger:
            return "persistent_disagreement_with_chain_growth"
        if stalled >= self.config.score_stall_trigger_ticks:
            return "persistent_disagreement_with_score_stall"
        return f"persistent_{disagreement_reason or 'semantic_disagreement'}"

    def _recovered(self) -> tuple[bool, str | None]:
        score_gain = self.latest_score - self.intervention_start_score
        chain_reduction = self.intervention_start_chain - self.latest_chain
        if score_gain >= self.config.minimum_score_gain:
            return True, "score_gain"
        if chain_reduction >= self.config.minimum_chain_reduction:
            return True, "chain_reduction"
        return False, None

    def _finish_intervention(
        self,
        *,
        forced: bool,
        recovery_reason: str | None,
    ) -> None:
        if forced:
            self.forced_handoffs += 1
        else:
            self.recovery_successes += 1
        self.completed_interventions.append(
            {
                "intervention_index": self.intervention_count,
                "teacher_ticks": self.teacher_ticks,
                "start_score": self.intervention_start_score,
                "end_score": self.latest_score,
                "score_gain": (
                    self.latest_score - self.intervention_start_score
                ),
                "start_chain_length": self.intervention_start_chain,
                "end_chain_length": self.latest_chain,
                "chain_reduction": (
                    self.intervention_start_chain - self.latest_chain
                ),
                "recovered": not forced,
                "recovery_reason": recovery_reason,
                "forced_maximum_teacher_ticks": forced,
            }
        )
        self.phase = "handoff"
        self.cooldown_remaining = self.config.handoff_cooldown_ticks
        self.cooldown_elapsed = 0
        self.disagreement_streak = 0
        self.teacher_ticks = 0
        self.agreement_streak = 0

    def decide(
        self,
        *,
        tick: int,
        teacher_action: Sequence[int],
        student_action: Sequence[int],
        previous_info: Mapping[str, Any],
        active: bool = True,
    ) -> InterventionDecision:
        """Choose the executor for the current decision tick."""

        self._update_metrics(int(tick), previous_info)
        disagreement, disagreement_reason = semantic_action_disagreement(
            teacher_action,
            student_action,
            fire_aim_threshold=self.config.fire_aim_disagreement_bins,
        )
        if not active:
            return InterventionDecision(
                execute_teacher=False,
                phase="inactive",
                semantic_disagreement=disagreement,
                disagreement_reason=disagreement_reason,
                trigger_reason=None,
                intervention_started=False,
                intervention_ended=False,
                intervention_index=self.intervention_count,
                teacher_ticks_in_intervention=0,
                post_handoff_tick=0,
            )

        if self.phase == "teacher":
            self.teacher_ticks += 1
            if disagreement:
                self.agreement_streak = 0
            else:
                self.agreement_streak += 1
            recovered, recovery_reason = self._recovered()
            minimum_met = (
                self.teacher_ticks >= self.config.minimum_teacher_ticks
            )
            stable = self.agreement_streak >= self.config.stable_agreement_ticks
            forced = self.teacher_ticks >= self.config.maximum_teacher_ticks
            ended = bool(forced or (minimum_met and stable and recovered))
            teacher_ticks = self.teacher_ticks
            intervention_index = self.intervention_count
            if ended:
                self._finish_intervention(
                    forced=forced,
                    recovery_reason=(None if forced else recovery_reason),
                )
            return InterventionDecision(
                execute_teacher=True,
                phase="teacher_recovery",
                semantic_disagreement=disagreement,
                disagreement_reason=disagreement_reason,
                trigger_reason=None,
                intervention_started=False,
                intervention_ended=ended,
                intervention_index=intervention_index,
                teacher_ticks_in_intervention=teacher_ticks,
                post_handoff_tick=0,
            )

        if self.phase == "handoff":
            self.cooldown_elapsed += 1
            self.cooldown_remaining -= 1
            post_handoff_tick = self.cooldown_elapsed
            if self.cooldown_remaining <= 0:
                self.phase = "student"
                self.cooldown_remaining = 0
                self.cooldown_elapsed = 0
            return InterventionDecision(
                execute_teacher=False,
                phase="student_post_handoff",
                semantic_disagreement=disagreement,
                disagreement_reason=disagreement_reason,
                trigger_reason=None,
                intervention_started=False,
                intervention_ended=False,
                intervention_index=self.intervention_count,
                teacher_ticks_in_intervention=0,
                post_handoff_tick=post_handoff_tick,
            )

        if disagreement:
            self.disagreement_streak += 1
        else:
            self.disagreement_streak = 0
        if self.disagreement_streak >= self.config.disagreement_grace_ticks:
            trigger_reason = self._trigger_reason(disagreement_reason)
            self.phase = "teacher"
            self.intervention_count += 1
            self.intervention_start_score = self.latest_score
            self.intervention_start_chain = self.latest_chain
            self.teacher_ticks = 1
            self.agreement_streak = 0 if disagreement else 1
            return InterventionDecision(
                execute_teacher=True,
                phase="teacher_recovery",
                semantic_disagreement=disagreement,
                disagreement_reason=disagreement_reason,
                trigger_reason=trigger_reason,
                intervention_started=True,
                intervention_ended=False,
                intervention_index=self.intervention_count,
                teacher_ticks_in_intervention=1,
                post_handoff_tick=0,
            )
        return InterventionDecision(
            execute_teacher=False,
            phase=(
                "student_disagreement_grace"
                if disagreement
                else "student_nominal"
            ),
            semantic_disagreement=disagreement,
            disagreement_reason=disagreement_reason,
            trigger_reason=None,
            intervention_started=False,
            intervention_ended=False,
            intervention_index=self.intervention_count,
            teacher_ticks_in_intervention=0,
            post_handoff_tick=0,
        )

    def summary(self) -> dict[str, Any]:
        return {
            "config": self.config.contract(),
            "interventions": self.intervention_count,
            "recovery_successes": self.recovery_successes,
            "forced_handoffs": self.forced_handoffs,
            "completed_interventions": list(self.completed_interventions),
            "terminal_phase": self.phase,
            "terminal_disagreement_streak": self.disagreement_streak,
        }
