"""Semantic-override deep recovery for AlphaZuma DAgger.

V2 established that rolling sparse-error triggers and deep recovery can turn
student losses into wins, but it counted every tick in the recovery phase as
teacher execution even when teacher and student were semantically equivalent.
V3 preserves the V2 state machine and recovery boundary while executing the
teacher only on genuine semantic disagreement.  Agreement ticks remain
student-executed policy-induced state exposure.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

from tools.alphazuma_55_recovery_intervention_v1 import InterventionDecision
from tools.alphazuma_55_recovery_intervention_v2 import (
    RecoveryInterventionConfigV2,
    RecoveryInterventionControllerV2,
)


RecoveryInterventionConfigV3 = RecoveryInterventionConfigV2


class RecoveryInterventionControllerV3(RecoveryInterventionControllerV2):
    """Run the V2 recovery state machine with disagreement-only overrides."""

    def decide(
        self,
        *,
        tick: int,
        teacher_action: Sequence[int],
        student_action: Sequence[int],
        previous_info: Mapping[str, Any],
        active: bool = True,
    ) -> InterventionDecision:
        decision = super().decide(
            tick=tick,
            teacher_action=teacher_action,
            student_action=student_action,
            previous_info=previous_info,
            active=active,
        )
        if decision.phase != "teacher_deep_recovery":
            return decision
        if decision.semantic_disagreement:
            return replace(decision, phase="teacher_semantic_override")
        return replace(
            decision,
            execute_teacher=False,
            phase="student_semantic_agreement_during_recovery",
        )

    def summary(self) -> dict[str, Any]:
        value = super().summary()
        value["execution_rule"] = (
            "teacher_only_on_semantic_disagreement_during_v2_recovery_phase"
        )
        return value
