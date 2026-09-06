"""Closed-loop imitation metrics for AlphaZuma-55 teacher/student streams.

Lineage: tools/alphazuma_55_recovery_intervention_v1.py established the
semantic disagreement taxonomy (student_adds_unrequested_verb /
student_omits_teacher_verb / verb_mismatch / fire_aim_mismatch) but scored
it per tick: both-wait ticks return vacuous agreement, and because the
55/55 teacher issues non-wait verbs on only 2.2-2.9% of ticks, a student
that never acts measures roughly 2.5% "disagreement".  The same metric
never compares aim on wait ticks, even though the human-speed stack
(src/zuma_rl/human_speedrun.py) slew-limits the cursor and the projectile
release angle is decided by aim intents parked 12-18 ticks before release,
almost all of them on wait ticks.

This module fixes both blind spots with four pure instruments:

1. classify_tick / TickClassificationAccumulator -- keep the v1 reason
   taxonomy (delegating to semantic_action_disagreement) but split ticks
   into decision-relevant (either side issues a non-wait verb) and
   background both-wait ticks, so agreement can no longer be diluted by
   the wait-dominated denominator.
2. aim_intent_error / AimIntentErrorAccumulator -- circular aim-bin
   distance on EVERY tick, wait ticks included: the parked-aim channel
   the old metric ignored.
3. ReleaseAngleTracker -- attribute each projectile release (fire edge +
   configurable commit-to-release delay: 18 ticks through the wrapped
   human-speed stack, 6 on the raw env fire animation) to the executed
   aim at release time and compare it against the teacher target bin at
   the COMMIT tick, with a per-executor breakdown.
4. HonestExecutionAccounting -- teacher-execution fractions with an
   explicit decision-relevant denominator and a teacher-initiated shot
   fraction, so a controller cannot pass an autonomy gate by executing
   wait ticks (denominator dilution).  Empty denominators report NaN,
   which fails any threshold comparison instead of passing vacuously.

This module never calls an environment or a policy and never imports
torch.  Pure python/numpy keeps every metric independently testable.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

import numpy

from tools.alphazuma_55_recovery_intervention_v1 import (
    RecoveryInterventionConfig,
    circular_bin_distance,
    semantic_action_disagreement,
)


AIM_BINS = 180
VERB_COUNT = 4
WAIT_VERB = 0
FIRE_VERB = 1
DEFAULT_FIRE_AIM_DISAGREEMENT_BINS = (
    RecoveryInterventionConfig().fire_aim_disagreement_bins
)
DEFAULT_SHOT_TOLERANCE_BINS = 2
RAW_ENV_COMMIT_TO_RELEASE_TICKS = 6
WRAPPED_STACK_COMMIT_TO_RELEASE_TICKS = 18
_EXECUTORS = ("student", "teacher")


def _fraction(numerator: int, denominator: int) -> float:
    """Honest ratio: an empty denominator yields NaN, never a free pass."""

    if int(denominator) <= 0:
        return float("nan")
    return int(numerator) / int(denominator)


def _validate_verb(name: str, verb: int) -> int:
    value = int(verb)
    if not 0 <= value < VERB_COUNT:
        raise ValueError(f"{name} escaped the full55 action interface")
    return value


def _validate_aim_bin(name: str, aim_bin: int, aim_bins: int) -> int:
    value = int(aim_bin)
    if not 0 <= value < int(aim_bins):
        raise ValueError(f"{name} escaped the circular lattice")
    return value


def _validate_executor(executor: str) -> str:
    if executor not in _EXECUTORS:
        raise ValueError("executor must be 'student' or 'teacher'")
    return executor


@dataclass(frozen=True)
class TickClass:
    """Classification of one paired teacher/student decision tick."""

    decision_relevant: bool
    disagreement: bool
    disagreement_reason: str | None
    teacher_verb: int
    student_verb: int


def classify_tick(
    teacher_action: Sequence[int],
    student_action: Sequence[int],
    *,
    fire_aim_threshold: int = DEFAULT_FIRE_AIM_DISAGREEMENT_BINS,
    aim_bins: int = AIM_BINS,
) -> TickClass:
    """Classify one tick with the v1 taxonomy plus decision relevance.

    A tick is decision-relevant when either side issues a non-wait verb.
    Disagreement and its reason delegate to the frozen v1
    semantic_action_disagreement so the taxonomy stays byte-identical;
    the fix is the extra denominator flag, not a new comparison rule.
    """

    disagreement, reason = semantic_action_disagreement(
        teacher_action,
        student_action,
        fire_aim_threshold=fire_aim_threshold,
        aim_bins=aim_bins,
    )
    teacher_verb = _validate_verb("teacher verb", teacher_action[0])
    student_verb = _validate_verb("student verb", student_action[0])
    return TickClass(
        decision_relevant=(
            teacher_verb != WAIT_VERB or student_verb != WAIT_VERB
        ),
        disagreement=disagreement,
        disagreement_reason=reason,
        teacher_verb=teacher_verb,
        student_verb=student_verb,
    )


@dataclass(frozen=True)
class TickClassificationSummary:
    """Stream totals separating the old diluted rate from the honest one."""

    total_ticks: int
    decision_relevant_ticks: int
    disagreement_ticks: int
    per_tick_disagreement_rate: float
    decision_relevant_disagreement_rate: float
    reason_counts: dict[str, int]


class TickClassificationAccumulator:
    """Accumulate TickClass results over one or more episodes."""

    def __init__(
        self,
        *,
        fire_aim_threshold: int = DEFAULT_FIRE_AIM_DISAGREEMENT_BINS,
        aim_bins: int = AIM_BINS,
    ) -> None:
        self.fire_aim_threshold = int(fire_aim_threshold)
        self.aim_bins = int(aim_bins)
        self.total_ticks = 0
        self.decision_relevant_ticks = 0
        self.disagreement_ticks = 0
        self.reason_counts: Counter[str] = Counter()

    def add(
        self,
        teacher_action: Sequence[int],
        student_action: Sequence[int],
    ) -> TickClass:
        tick_class = classify_tick(
            teacher_action,
            student_action,
            fire_aim_threshold=self.fire_aim_threshold,
            aim_bins=self.aim_bins,
        )
        self.total_ticks += 1
        if tick_class.decision_relevant:
            self.decision_relevant_ticks += 1
        if tick_class.disagreement:
            self.disagreement_ticks += 1
            if tick_class.disagreement_reason is not None:
                self.reason_counts[tick_class.disagreement_reason] += 1
        return tick_class

    def summary(self) -> TickClassificationSummary:
        return TickClassificationSummary(
            total_ticks=self.total_ticks,
            decision_relevant_ticks=self.decision_relevant_ticks,
            disagreement_ticks=self.disagreement_ticks,
            per_tick_disagreement_rate=_fraction(
                self.disagreement_ticks,
                self.total_ticks,
            ),
            decision_relevant_disagreement_rate=_fraction(
                self.disagreement_ticks,
                self.decision_relevant_ticks,
            ),
            reason_counts=dict(self.reason_counts),
        )


def aim_intent_error(
    teacher_aim_bin: int,
    student_aim_bin: int,
    aim_bins: int = AIM_BINS,
) -> int:
    """Circular aim-bin distance, valid on every tick including waits."""

    return circular_bin_distance(teacher_aim_bin, student_aim_bin, aim_bins)


@dataclass(frozen=True)
class AimIntentErrorSummary:
    """Distribution of the wait-inclusive aim intent error in bins."""

    count: int
    mean_bins: float
    p50_bins: float
    p90_bins: float
    max_bins: int


class AimIntentErrorAccumulator:
    """Accumulate per-tick circular aim errors and report the tail.

    This is the wait-tick aim channel the old metric ignored: because the
    slew-limited cursor is parked by wait-tick aim intents 12-18 ticks
    before release, aim supervision must be measured on every tick, not
    only on the ~2.5% of ticks where a verb fires.
    """

    def __init__(self, aim_bins: int = AIM_BINS) -> None:
        self.aim_bins = int(aim_bins)
        self.errors: list[int] = []

    def add(self, teacher_aim_bin: int, student_aim_bin: int) -> int:
        error = aim_intent_error(
            teacher_aim_bin,
            student_aim_bin,
            self.aim_bins,
        )
        self.errors.append(error)
        return error

    def summary(self) -> AimIntentErrorSummary:
        if not self.errors:
            raise ValueError("no aim intent samples accumulated")
        values = numpy.asarray(self.errors, dtype=numpy.float64)
        return AimIntentErrorSummary(
            count=len(self.errors),
            mean_bins=float(values.mean()),
            p50_bins=float(numpy.percentile(values, 50.0)),
            p90_bins=float(numpy.percentile(values, 90.0)),
            max_bins=int(values.max()),
        )


@dataclass(frozen=True)
class ReleaseAngleConfig:
    """Frozen commit-to-release contract for one tracked action stack.

    commit_to_release_ticks defaults to 18 for the wrapped human-speed
    stack (reaction_delay_ticks=12 + 6-tick FIRING animation before the
    projectile samples the live aim angle).  Use
    RAW_ENV_COMMIT_TO_RELEASE_TICKS (6) when tracking the raw env, where
    the fire request itself is the commit edge.
    """

    commit_to_release_ticks: int = WRAPPED_STACK_COMMIT_TO_RELEASE_TICKS
    tolerance_bins: int = DEFAULT_SHOT_TOLERANCE_BINS
    aim_bins: int = AIM_BINS

    def __post_init__(self) -> None:
        if int(self.commit_to_release_ticks) < 1:
            raise ValueError("commit_to_release_ticks must be positive")
        if int(self.tolerance_bins) < 0:
            raise ValueError("tolerance_bins cannot be negative")
        if int(self.aim_bins) < 2:
            raise ValueError("aim_bins must be at least two")


@dataclass(frozen=True)
class ReleaseTickRecord:
    """One executed tick as seen by the release-angle tracker."""

    tick: int
    executed_verb: int
    executed_aim_bin: int
    teacher_target_bin: int
    fire_edge_executor: str | None = None


@dataclass(frozen=True)
class ShotRelease:
    """One attributed shot: commit-time intent versus release-time aim."""

    executor: str
    commit_tick: int
    release_tick: int
    commit_target_bin: int
    release_aim_bin: int
    release_angle_error_bins: int
    on_target: bool


@dataclass(frozen=True)
class ExecutorShotSummary:
    """Shot quality restricted to one fire-edge executor."""

    shot_count: int
    shots_on_target_rate: float
    mean_release_angle_error_bins: float


@dataclass(frozen=True)
class ReleaseAngleSummary:
    """All attributed shots plus quality rates and executor breakdown."""

    shot_count: int
    shots_on_target_rate: float
    mean_release_angle_error_bins: float
    pending_commits: int
    shots: tuple[ShotRelease, ...]
    per_executor: dict[str, ExecutorShotSummary]


@dataclass
class _PendingCommit:
    executor: str
    commit_tick: int
    release_tick: int
    commit_target_bin: int


class ReleaseAngleTracker:
    """Attribute each release to its commit-time teacher target.

    The projectile release angle is the live aim angle AT RELEASE, which
    trails the fire edge by commit_to_release_ticks.  Comparing the
    release-time executed aim against the teacher target at the COMMIT
    tick therefore measures exactly the channel the wrapped stack must
    park via wait-tick aim intents: a cursor that drifts between commit
    and release shows up as release_angle_error_bins > 0 even when the
    fire verb itself agreed.
    """

    def __init__(self, config: ReleaseAngleConfig | None = None) -> None:
        self.config = config if config is not None else ReleaseAngleConfig()
        self._pending: list[_PendingCommit] = []
        self._last_tick: int | None = None
        self.shots: list[ShotRelease] = []

    def observe(self, record: ReleaseTickRecord) -> list[ShotRelease]:
        """Consume one tick; return shots whose release resolved here.

        Ticks must be strictly increasing.  With a contiguous tick stream
        each shot resolves exactly at commit_tick + commit_to_release_ticks;
        on a gapped stream it resolves at the first observed tick at or
        past that point.
        """

        tick = int(record.tick)
        if self._last_tick is not None and tick <= self._last_tick:
            raise ValueError("release tracker requires strictly increasing ticks")
        self._last_tick = tick
        _validate_verb("executed verb", record.executed_verb)
        aim_bin = _validate_aim_bin(
            "executed aim bin",
            record.executed_aim_bin,
            self.config.aim_bins,
        )
        target_bin = _validate_aim_bin(
            "teacher target bin",
            record.teacher_target_bin,
            self.config.aim_bins,
        )
        resolved: list[ShotRelease] = []
        still_pending: list[_PendingCommit] = []
        for pending in self._pending:
            if tick >= pending.release_tick:
                error = circular_bin_distance(
                    aim_bin,
                    pending.commit_target_bin,
                    self.config.aim_bins,
                )
                resolved.append(
                    ShotRelease(
                        executor=pending.executor,
                        commit_tick=pending.commit_tick,
                        release_tick=tick,
                        commit_target_bin=pending.commit_target_bin,
                        release_aim_bin=aim_bin,
                        release_angle_error_bins=error,
                        on_target=error <= self.config.tolerance_bins,
                    )
                )
            else:
                still_pending.append(pending)
        self._pending = still_pending
        self.shots.extend(resolved)
        if record.fire_edge_executor is not None:
            executor = _validate_executor(record.fire_edge_executor)
            self._pending.append(
                _PendingCommit(
                    executor=executor,
                    commit_tick=tick,
                    release_tick=tick + self.config.commit_to_release_ticks,
                    commit_target_bin=target_bin,
                )
            )
        return resolved

    @staticmethod
    def _shot_summary(shots: Sequence[ShotRelease]) -> ExecutorShotSummary:
        on_target = sum(1 for shot in shots if shot.on_target)
        if shots:
            mean_error = float(
                numpy.mean([shot.release_angle_error_bins for shot in shots])
            )
        else:
            mean_error = float("nan")
        return ExecutorShotSummary(
            shot_count=len(shots),
            shots_on_target_rate=_fraction(on_target, len(shots)),
            mean_release_angle_error_bins=mean_error,
        )

    def summary(self) -> ReleaseAngleSummary:
        overall = self._shot_summary(self.shots)
        per_executor = {
            executor: self._shot_summary(
                [shot for shot in self.shots if shot.executor == executor]
            )
            for executor in _EXECUTORS
        }
        return ReleaseAngleSummary(
            shot_count=overall.shot_count,
            shots_on_target_rate=overall.shots_on_target_rate,
            mean_release_angle_error_bins=(
                overall.mean_release_angle_error_bins
            ),
            pending_commits=len(self._pending),
            shots=tuple(self.shots),
            per_executor=per_executor,
        )


@dataclass(frozen=True)
class HonestExecutionSummary:
    """Execution fractions that cannot be gamed by wait-tick padding."""

    total_ticks: int
    teacher_executed_ticks: int
    decision_relevant_ticks: int
    decision_relevant_teacher_ticks: int
    shot_count: int
    teacher_initiated_shots: int
    student_initiated_shots: int
    tick_teacher_fraction: float
    decision_relevant_teacher_fraction: float
    teacher_initiated_shot_fraction: float
    phase_ticks: dict[str, int] = field(default_factory=dict)


class HonestExecutionAccounting:
    """Count who actually executed the ticks that mattered.

    tick_teacher_fraction uses every tick and is reported only for
    context: with ~97% wait ticks it can be driven arbitrarily low (or
    high) by wait-tick assignment alone.  Autonomy gates must read
    decision_relevant_teacher_fraction (denominator restricted to ticks
    where either side issued a non-wait verb) and
    teacher_initiated_shot_fraction.  Both report NaN when their
    denominator is empty, so a stream with no decisions can never pass a
    threshold comparison vacuously.

    A shot initiation is an executed fire verb: the executing side's
    verb on that tick, teacher_verb when the teacher executed, else
    student_verb.  Callers feeding the wrapped stack should pass
    edge-filtered decision ticks (buttons act on rising edges only).
    """

    def __init__(self) -> None:
        self.total_ticks = 0
        self.teacher_executed_ticks = 0
        self.decision_relevant_ticks = 0
        self.decision_relevant_teacher_ticks = 0
        self.teacher_initiated_shots = 0
        self.student_initiated_shots = 0
        self.phase_ticks: Counter[str] = Counter()

    def add(
        self,
        *,
        executor: str,
        teacher_verb: int,
        student_verb: int,
        phase: str,
    ) -> None:
        executor = _validate_executor(executor)
        teacher_verb = _validate_verb("teacher verb", teacher_verb)
        student_verb = _validate_verb("student verb", student_verb)
        self.total_ticks += 1
        self.phase_ticks[str(phase)] += 1
        teacher_executed = executor == "teacher"
        if teacher_executed:
            self.teacher_executed_ticks += 1
        if teacher_verb != WAIT_VERB or student_verb != WAIT_VERB:
            self.decision_relevant_ticks += 1
            if teacher_executed:
                self.decision_relevant_teacher_ticks += 1
        executed_verb = teacher_verb if teacher_executed else student_verb
        if executed_verb == FIRE_VERB:
            if teacher_executed:
                self.teacher_initiated_shots += 1
            else:
                self.student_initiated_shots += 1

    def summary(self) -> HonestExecutionSummary:
        shot_count = self.teacher_initiated_shots + self.student_initiated_shots
        return HonestExecutionSummary(
            total_ticks=self.total_ticks,
            teacher_executed_ticks=self.teacher_executed_ticks,
            decision_relevant_ticks=self.decision_relevant_ticks,
            decision_relevant_teacher_ticks=(
                self.decision_relevant_teacher_ticks
            ),
            shot_count=shot_count,
            teacher_initiated_shots=self.teacher_initiated_shots,
            student_initiated_shots=self.student_initiated_shots,
            tick_teacher_fraction=_fraction(
                self.teacher_executed_ticks,
                self.total_ticks,
            ),
            decision_relevant_teacher_fraction=_fraction(
                self.decision_relevant_teacher_ticks,
                self.decision_relevant_ticks,
            ),
            teacher_initiated_shot_fraction=_fraction(
                self.teacher_initiated_shots,
                shot_count,
            ),
            phase_ticks=dict(self.phase_ticks),
        )
