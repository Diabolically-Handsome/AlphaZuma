from __future__ import annotations

import pytest

from tools.alphazuma_55_recovery_intervention_v1 import (
    RecoveryInterventionConfig,
    RecoveryInterventionController,
    circular_bin_distance,
    semantic_action_disagreement,
)
from tools import probe_alphazuma_55_recovery_intervention_v1 as probe


def _config(**overrides: int) -> RecoveryInterventionConfig:
    values = {
        "disagreement_grace_ticks": 2,
        "minimum_teacher_ticks": 2,
        "maximum_teacher_ticks": 4,
        "stable_agreement_ticks": 1,
        "handoff_cooldown_ticks": 2,
        "minimum_score_gain": 10,
        "minimum_chain_reduction": 1,
        "score_stall_trigger_ticks": 3,
        "chain_growth_trigger": 2,
        "fire_aim_disagreement_bins": 4,
    }
    values.update(overrides)
    return RecoveryInterventionConfig(**values)


def test_circular_aim_and_semantic_disagreement_ignore_wait_aim() -> None:
    assert circular_bin_distance(179, 1) == 2
    assert semantic_action_disagreement(
        (0, 90), (0, 5), fire_aim_threshold=4
    ) == (False, None)
    assert semantic_action_disagreement(
        (1, 179), (1, 1), fire_aim_threshold=4
    ) == (False, None)
    assert semantic_action_disagreement(
        (1, 179), (1, 4), fire_aim_threshold=4
    ) == (True, "fire_aim_mismatch")
    assert semantic_action_disagreement(
        (2, 10), (0, 10), fire_aim_threshold=4
    ) == (True, "student_omits_teacher_verb")


def test_configuration_rejects_inverted_teacher_window() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        _config(minimum_teacher_ticks=5, maximum_teacher_ticks=4)


def test_student_reaches_off_policy_state_before_teacher_intervenes() -> None:
    controller = RecoveryInterventionController(_config())
    first = controller.decide(
        tick=0,
        teacher_action=(1, 20),
        student_action=(0, 20),
        previous_info={"score": 0, "chain_length": 20},
    )
    second = controller.decide(
        tick=1,
        teacher_action=(1, 20),
        student_action=(0, 20),
        previous_info={"score": 0, "chain_length": 20},
    )
    assert first.phase == "student_disagreement_grace"
    assert first.execute_teacher is False
    assert second.intervention_started is True
    assert second.execute_teacher is True
    assert controller.intervention_count == 1


def test_successful_recovery_requires_minimum_burst_and_stable_handoff() -> None:
    controller = RecoveryInterventionController(_config())
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
    ended = controller.decide(
        tick=2,
        teacher_action=(1, 20),
        student_action=(1, 20),
        previous_info={"score": 10, "chain_length": 19},
    )
    assert ended.intervention_ended is True
    assert ended.teacher_ticks_in_intervention == 2
    assert controller.recovery_successes == 1
    assert controller.phase == "handoff"
    handoff = controller.decide(
        tick=3,
        teacher_action=(1, 20),
        student_action=(0, 20),
        previous_info={"score": 10, "chain_length": 19},
    )
    assert handoff.phase == "student_post_handoff"
    assert handoff.execute_teacher is False


def test_unrecovered_intervention_forces_handoff_at_frozen_maximum() -> None:
    controller = RecoveryInterventionController(_config())
    decisions = []
    for tick in range(6):
        decisions.append(
            controller.decide(
                tick=tick,
                teacher_action=(1, 20),
                student_action=(0, 20),
                previous_info={"score": 0, "chain_length": 20},
            )
        )
    ended = [row for row in decisions if row.intervention_ended]
    assert len(ended) == 1
    assert ended[0].teacher_ticks_in_intervention == 4
    assert controller.forced_handoffs == 1
    assert controller.completed_interventions[0]["recovered"] is False


def test_inactive_vector_slot_never_executes_teacher() -> None:
    controller = RecoveryInterventionController(_config())
    decision = controller.decide(
        tick=0,
        teacher_action=(1, 20),
        student_action=(0, 20),
        previous_info={},
        active=False,
    )
    assert decision.phase == "inactive"
    assert decision.execute_teacher is False
    assert controller.intervention_count == 0


def test_probe_decision_requires_real_paired_gain_and_bounded_teacher() -> None:
    prereg = {
        "levels": [f"level-{index}" for index in range(6)],
        "acceptance": {
            "intervention_wins_must_exceed_student_only": True,
            "intervention_total_score_must_exceed_student_only": True,
            "minimum_paired_score_improvements": 6,
            "minimum_interventions": 6,
            "minimum_recovery_successes": 1,
            "maximum_teacher_execution_fraction": 0.6,
        },
    }
    baseline_episodes = [
        {
            "level_id": level_id,
            "seed": 10 + index,
            "outcome": "win" if index == 0 else "loss",
            "score": 100,
        }
        for index, level_id in enumerate(prereg["levels"])
    ]
    recovery_episodes = [
        {
            "level_id": level_id,
            "seed": 10 + index,
            "outcome": "win" if index < 2 else "loss",
            "score": 200,
        }
        for index, level_id in enumerate(prereg["levels"])
    ]
    decision = probe._decision(
        prereg,
        [
            {
                "mode": "student_only",
                "episodes": baseline_episodes,
                "summary": {"wins": 1, "total_score": 600},
            },
            {
                "mode": "recovery_intervention",
                "episodes": recovery_episodes,
                "summary": {
                    "wins": 2,
                    "total_score": 1200,
                    "interventions": 6,
                    "recovery_successes": 1,
                    "teacher_execution_fraction": 0.5,
                },
            },
        ],
    )
    assert decision["status"] == "PASS"
    assert all(decision["checks"].values())
    assert decision["training_authorized_by_this_probe"] is False
