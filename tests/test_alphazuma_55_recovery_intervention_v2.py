from __future__ import annotations

import pytest

from tools.alphazuma_55_recovery_intervention_v2 import (
    RecoveryInterventionConfigV2,
    RecoveryInterventionControllerV2,
)


def _config(**overrides: int) -> RecoveryInterventionConfigV2:
    values = {
        "disagreement_window_ticks": 5,
        "disagreements_to_trigger": 2,
        "minimum_teacher_ticks": 2,
        "maximum_teacher_ticks": 4,
        "stable_agreement_ticks": 1,
        "handoff_cooldown_ticks": 2,
        "minimum_score_gain": 20,
        "minimum_chain_reduction": 2,
        "score_stall_trigger_ticks": 4,
        "chain_growth_trigger": 2,
        "fire_aim_disagreement_bins": 4,
    }
    values.update(overrides)
    return RecoveryInterventionConfigV2(**values)


def test_config_rejects_trigger_larger_than_window() -> None:
    with pytest.raises(ValueError, match="rolling window"):
        _config(disagreement_window_ticks=2, disagreements_to_trigger=3)


def test_nonconsecutive_errors_trigger_inside_rolling_window() -> None:
    controller = RecoveryInterventionControllerV2(_config())
    first = controller.decide(
        tick=0,
        teacher_action=(1, 20),
        student_action=(0, 20),
        previous_info={"score": 0, "chain_length": 20},
    )
    agreement = controller.decide(
        tick=1,
        teacher_action=(0, 20),
        student_action=(0, 90),
        previous_info={"score": 0, "chain_length": 20},
    )
    trigger = controller.decide(
        tick=2,
        teacher_action=(1, 20),
        student_action=(0, 20),
        previous_info={"score": 0, "chain_length": 20},
    )
    assert first.execute_teacher is False
    assert agreement.execute_teacher is False
    assert trigger.intervention_started is True
    assert trigger.trigger_reason == "rolling_semantic_disagreement_budget"


def test_old_errors_expire_from_rolling_window() -> None:
    controller = RecoveryInterventionControllerV2(_config())
    controller.decide(
        tick=0,
        teacher_action=(1, 20),
        student_action=(0, 20),
        previous_info={"score": 0, "chain_length": 20},
    )
    decision = controller.decide(
        tick=6,
        teacher_action=(1, 20),
        student_action=(0, 20),
        previous_info={"score": 10, "chain_length": 20},
    )
    assert decision.intervention_started is False
    assert decision.phase == "student_rolling_disagreement"


def test_deep_recovery_does_not_handoff_on_tiny_local_gain() -> None:
    controller = RecoveryInterventionControllerV2(_config())
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
    tiny = controller.decide(
        tick=2,
        teacher_action=(1, 20),
        student_action=(1, 20),
        previous_info={"score": 10, "chain_length": 19},
    )
    deep = controller.decide(
        tick=3,
        teacher_action=(1, 20),
        student_action=(1, 20),
        previous_info={"score": 20, "chain_length": 18},
    )
    assert tiny.intervention_ended is False
    assert deep.intervention_ended is True
    assert controller.recovery_successes == 1
