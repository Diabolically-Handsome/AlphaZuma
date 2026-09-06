from __future__ import annotations

import numpy as np

from zuma_rl.revenge_settled_strategic_teacher import (
    AimSettledStrategicRevengeTeacher,
)
from zuma_rl.revenge_settled_strategic_teacher_v3 import (
    CurveAwareSettledStrategicRevengeTeacher,
)
from tests.test_revenge_settled_strategic_teacher import _observation, _spec


def _swap_teacher(monkeypatch, active_curves: int):
    teacher = CurveAwareSettledStrategicRevengeTeacher(
        _spec(), aim_tolerance_bins=1
    )
    monkeypatch.setattr(
        AimSettledStrategicRevengeTeacher,
        "act",
        lambda self, observation: np.asarray((2, 3), dtype=np.int64),
    )
    monkeypatch.setattr(
        teacher,
        "_active_curve_count",
        lambda observation: active_curves,
    )
    return teacher


def test_single_curve_swap_waits_for_target(monkeypatch):
    teacher = _swap_teacher(monkeypatch, 1)

    assert teacher.act(_observation(aim_bin=9)).tolist() == [0, 3]


def test_multi_curve_swap_is_immediate(monkeypatch):
    teacher = _swap_teacher(monkeypatch, 2)

    assert teacher.act(_observation(aim_bin=9)).tolist() == [2, 3]
