from __future__ import annotations

import numpy as np

from zuma_rl.revenge_settled_strategic_teacher import (
    AimSettledStrategicRevengeTeacher,
)
from zuma_rl.revenge_settled_strategic_teacher_v2 import (
    AimSettledStrategicRevengeTeacherV2,
)
from tests.test_revenge_settled_strategic_teacher import _observation, _spec


def test_swap_waits_until_cursor_is_near_next_target(monkeypatch):
    teacher = AimSettledStrategicRevengeTeacherV2(
        _spec(), aim_tolerance_bins=1
    )
    monkeypatch.setattr(
        AimSettledStrategicRevengeTeacher,
        "act",
        lambda self, observation: np.asarray((2, 3), dtype=np.int64),
    )

    assert teacher.act(_observation(aim_bin=9)).tolist() == [0, 3]
    assert teacher.act(_observation(aim_bin=3)).tolist() == [2, 3]


def test_non_swap_action_is_unchanged(monkeypatch):
    teacher = AimSettledStrategicRevengeTeacherV2(
        _spec(), aim_tolerance_bins=1
    )
    monkeypatch.setattr(
        AimSettledStrategicRevengeTeacher,
        "act",
        lambda self, observation: np.asarray((3, 7), dtype=np.int64),
    )

    assert teacher.act(_observation(aim_bin=0)).tolist() == [3, 7]
