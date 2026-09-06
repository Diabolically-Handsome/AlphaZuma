from __future__ import annotations

import numpy as np
import pytest

from tools.distill_alphazuma_55_polar_dagger_v1 import (
    _mix_actions,
    _validate_probability_schedule,
)


def test_probability_schedule_requires_teacher_to_student_transition() -> None:
    assert _validate_probability_schedule((1.0, 0.5, 0.0)) == (
        1.0,
        0.5,
        0.0,
    )
    with pytest.raises(ValueError, match="begin"):
        _validate_probability_schedule((0.5, 0.0))
    with pytest.raises(ValueError, match="non-increasing"):
        _validate_probability_schedule((1.0, 0.25, 0.5, 0.0))
    with pytest.raises(ValueError, match="finish"):
        _validate_probability_schedule((1.0, 0.5))


def test_mix_actions_honors_frozen_schedule_extremes() -> None:
    teacher = np.asarray(((1, 10), (2, 20)), dtype=np.int64)
    student = np.asarray(((0, 30), (3, 40)), dtype=np.int64)
    active = np.asarray((True, True), dtype=np.bool_)
    rng = np.random.default_rng(7)

    actions, teacher_mask = _mix_actions(
        teacher_actions=teacher,
        student_actions=student,
        active=active,
        teacher_probability=1.0,
        rng=rng,
    )
    np.testing.assert_array_equal(actions, teacher)
    np.testing.assert_array_equal(teacher_mask, (True, True))

    actions, teacher_mask = _mix_actions(
        teacher_actions=teacher,
        student_actions=student,
        active=active,
        teacher_probability=0.0,
        rng=rng,
    )
    np.testing.assert_array_equal(actions, student)
    np.testing.assert_array_equal(teacher_mask, (False, False))


def test_mix_actions_keeps_completed_vector_slots_off_student_counts() -> None:
    teacher = np.asarray(((1, 10), (0, 0)), dtype=np.int64)
    student = np.asarray(((2, 30), (3, 40)), dtype=np.int64)
    actions, teacher_mask = _mix_actions(
        teacher_actions=teacher,
        student_actions=student,
        active=np.asarray((True, False), dtype=np.bool_),
        teacher_probability=0.0,
        rng=np.random.default_rng(9),
    )
    np.testing.assert_array_equal(actions, ((2, 30), (0, 0)))
    np.testing.assert_array_equal(teacher_mask, (False, True))


def test_mix_actions_rejects_malformed_batches() -> None:
    with pytest.raises(ValueError, match="Nx2"):
        _mix_actions(
            teacher_actions=np.zeros((2, 2), dtype=np.int64),
            student_actions=np.zeros((2, 3), dtype=np.int64),
            active=np.ones(2, dtype=np.bool_),
            teacher_probability=0.5,
            rng=np.random.default_rng(1),
        )
