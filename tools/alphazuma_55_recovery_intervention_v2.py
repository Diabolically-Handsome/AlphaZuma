"""Rolling-error, deep-recovery controller for AlphaZuma DAgger.

V1 proved that consecutive disagreements are too sparse and that a tiny local
score gain is not a sufficient recovery boundary.  V2 therefore accumulates
semantic disagreements in a fixed rolling window and requires a materially
deeper teacher recovery before handing control back to the student.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from tools.alphazuma_55_recovery_intervention_v1 import (
    InterventionDecision,
    semantic_action_disagreement,
)


@dataclass(frozen=True)
class RecoveryInterventionConfigV2:
    disagreement_window_ticks: int = 48
    disagreements_to_trigger: int = 4
    minimum_teacher_ticks: int = 300
    maximum_teacher_ticks: int = 900
    stable_agreement_ticks: int = 48
    handoff_cooldown_ticks: int = 96
    minimum_score_gain: int = 300
    minimum_chain_reduction: int = 6
    score_stall_trigger_ticks: int = 300
    chain_growth_trigger: int = 4
    fire_aim_disagreement_bins: int = 4

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if int(value) < 1:
                raise ValueError(f"{name} must be positive")
        if self.disagreements_to_trigger > self.disagreement_window_ticks:
            raise ValueError(
                "disagreements_to_trigger cannot exceed its rolling window"
            )
        if self.minimum_teacher_ticks > self.maximum_teacher_ticks:
            raise ValueError(
                "minimum_teacher_ticks cannot exceed maximum_teacher_ticks"
            )
        if self.fire_aim_disagreement_bins > 90:
            raise ValueError("fire aim disagreement threshold cannot exceed 90")

    def contract(self) -> dict[str, int]:
        return {name: int(value) for name, value in asdict(self).items()}


class RecoveryInterventionControllerV2:
    """Student-first control with rolling triggers and deep recovery bursts."""

    def __init__(self, config: RecoveryInterventionConfigV2) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.phase = "student"
        self.disagreement_ticks: deque[int] = deque()
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

    def _prune_disagreements(self, tick: int) -> None:
        first_allowed = int(tick) - self.config.disagreement_window_ticks + 1
        while self.disagreement_ticks and self.disagreement_ticks[0] < first_allowed:
            self.disagreement_ticks.popleft()

    def _trigger_reason(self) -> str | None:
        stalled = self.latest_tick - self.last_progress_tick
        low_chain = (
            self.latest_chain
            if self.low_chain_since_progress is None
            else self.low_chain_since_progress
        )
        has_error = bool(self.disagreement_ticks)
        if has_error and self.latest_chain - low_chain >= self.config.chain_growth_trigger:
            return "rolling_disagreement_with_chain_growth"
        if has_error and stalled >= self.config.score_stall_trigger_ticks:
            return "rolling_disagreement_with_score_stall"
        if len(self.disagreement_ticks) >= self.config.disagreements_to_trigger:
            return "rolling_semantic_disagreement_budget"
        return None

    def _recovered(self) -> tuple[bool, str | None]:
        score_gain = self.latest_score - self.intervention_start_score
        chain_reduction = self.intervention_start_chain - self.latest_chain
        if score_gain >= self.config.minimum_score_gain:
            return True, "deep_score_gain"
        if chain_reduction >= self.config.minimum_chain_reduction:
            return True, "deep_chain_reduction"
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
                "score_gain": self.latest_score - self.intervention_start_score,
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
        self.teacher_ticks = 0
        self.agreement_streak = 0
        self.disagreement_ticks.clear()

    def decide(
        self,
        *,
        tick: int,
        teacher_action: Sequence[int],
        student_action: Sequence[int],
        previous_info: Mapping[str, Any],
        active: bool = True,
    ) -> InterventionDecision:
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
                phase="teacher_deep_recovery",
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

        self._prune_disagreements(int(tick))
        if disagreement:
            self.disagreement_ticks.append(int(tick))
        trigger_reason = self._trigger_reason()
        if trigger_reason is not None:
            self.phase = "teacher"
            self.intervention_count += 1
            self.intervention_start_score = self.latest_score
            self.intervention_start_chain = self.latest_chain
            self.teacher_ticks = 1
            self.agreement_streak = 0 if disagreement else 1
            return InterventionDecision(
                execute_teacher=True,
                phase="teacher_deep_recovery",
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
                "student_rolling_disagreement"
                if self.disagreement_ticks
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
            "terminal_rolling_disagreements": len(self.disagreement_ticks),
        }
