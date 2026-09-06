from __future__ import annotations

from tools.alphazuma_55_recovery_intervention_v3 import (
    RecoveryInterventionConfigV3,
    RecoveryInterventionControllerV3,
)


def _config() -> RecoveryInterventionConfigV3:
    return RecoveryInterventionConfigV3(
        disagreement_window_ticks=5,
        disagreements_to_trigger=2,
        minimum_teacher_ticks=2,
        maximum_teacher_ticks=4,
        stable_agreement_ticks=1,
        handoff_cooldown_ticks=2,
        minimum_score_gain=20,
        minimum_chain_reduction=2,
        score_stall_trigger_ticks=4,
        chain_growth_trigger=2,
        fire_aim_disagreement_bins=4,
    )


def test_trigger_tick_executes_teacher_on_semantic_disagreement() -> None:
    controller = RecoveryInterventionControllerV3(_config())
    controller.decide(
        tick=0,
        teacher_action=(1, 20),
        student_action=(0, 20),
        previous_info={"score": 0, "chain_length": 20},
    )
    trigger = controller.decide(
        tick=1,
        teacher_action=(1, 20),
        student_action=(0, 20),
        previous_info={"score": 0, "chain_length": 20},
    )
    assert trigger.intervention_started is True
    assert trigger.execute_teacher is True
    assert trigger.phase == "teacher_semantic_override"


def test_agreement_tick_remains_student_executed_during_recovery() -> None:
    controller = RecoveryInterventionControllerV3(_config())
    controller.decide(
        tick=0,
        teacher_action=(1, 20),
        student_action=(0, 20),
        previous_info={"score": 0, "chain_length": 20},
    )
    controller.decide(
        tick=1,
        teacher_action=(1, 20),
        student_action=(0, 20),
        previous_info={"score": 0, "chain_length": 20},
    )
    agreement = controller.decide(
        tick=2,
        teacher_action=(1, 20),
        student_action=(1, 20),
        previous_info={"score": 20, "chain_length": 18},
    )
    assert agreement.execute_teacher is False
    assert agreement.phase == "student_semantic_agreement_during_recovery"
    assert agreement.intervention_ended is True
    assert controller.recovery_successes == 1


def test_recovery_disagreement_still_uses_teacher_override() -> None:
    controller = RecoveryInterventionControllerV3(_config())
    controller.decide(
        tick=0,
        teacher_action=(1, 20),
        student_action=(0, 20),
        previous_info={"score": 0, "chain_length": 20},
    )
    controller.decide(
        tick=1,
        teacher_action=(1, 20),
        student_action=(0, 20),
        previous_info={"score": 0, "chain_length": 20},
    )
    disagreement = controller.decide(
        tick=2,
        teacher_action=(1, 20),
        student_action=(0, 20),
        previous_info={"score": 0, "chain_length": 20},
    )
    assert disagreement.execute_teacher is True
    assert disagreement.phase == "teacher_semantic_override"
    assert disagreement.intervention_ended is False


def test_summary_freezes_semantic_override_execution_rule() -> None:
    summary = RecoveryInterventionControllerV3(_config()).summary()
    assert summary["execution_rule"] == (
        "teacher_only_on_semantic_disagreement_during_v2_recovery_phase"
    )
