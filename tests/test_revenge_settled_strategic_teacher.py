from __future__ import annotations

from dataclasses import replace

import numpy as np

from zuma_rl.revenge_settled_strategic_teacher import (
    AimSettledStrategicRevengeTeacher,
)
from zuma_rl.revenge_teacher import RevengeTeacherSpec


def _spec() -> RevengeTeacherSpec:
    return RevengeTeacherSpec(
        aim_bins=12,
        max_balls=1,
        ball_feature_names=(
            "present",
            "x",
            "y",
            "exploding",
            "waypoint",
            "contact_next_visible",
            "in_tunnel",
            "color_0",
        ),
        global_feature_names=(
            "aim_sin",
            "aim_cos",
            "current_missing",
            "next_missing",
            "gun_normal",
            "current_color_0",
            "next_color_0",
        ),
        balls_start=0,
        balls_stop=8,
        globals_start=8,
        globals_stop=15,
        logical_width=100.0,
        logical_height=100.0,
        shooter_x=50.0,
        shooter_y=50.0,
        shooter_positions=((50.0, 50.0),),
        collision_radius=10.0,
        num_colors=1,
    )


def _observation(*, aim_bin: int) -> np.ndarray:
    spec = _spec()
    angle = 2.0 * np.pi * (aim_bin + 0.5) / spec.aim_bins
    values = np.zeros(spec.globals_stop, dtype=np.float32)
    values[:8] = (1.0, 1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 1.0)
    values[8:] = (
        np.sin(angle),
        np.cos(angle),
        0.0,
        1.0,
        1.0,
        1.0,
        0.0,
    )
    return values


def test_fire_waits_until_cursor_is_near_target(monkeypatch):
    teacher = AimSettledStrategicRevengeTeacher(_spec(), aim_tolerance_bins=1)
    monkeypatch.setattr(
        teacher.__class__.__mro__[1],
        "act",
        lambda self, observation: np.asarray((1, 3), dtype=np.int64),
    )

    assert teacher.act(_observation(aim_bin=9)).tolist() == [0, 3]
    assert teacher.act(_observation(aim_bin=2)).tolist() == [1, 3]
    assert teacher.act(_observation(aim_bin=4)).tolist() == [1, 3]


def test_bin_distance_wraps_at_zero():
    teacher = AimSettledStrategicRevengeTeacher(_spec(), aim_tolerance_bins=1)

    assert teacher._bin_distance(0, 11) == 1
    assert teacher._bin_distance(1, 10) == 3


def test_tolerance_must_be_less_than_half_circle():
    spec = _spec()
    try:
        AimSettledStrategicRevengeTeacher(
            replace(spec, aim_bins=10), aim_tolerance_bins=5
        )
    except ValueError as error:
        assert "outside the useful range" in str(error)
    else:
        raise AssertionError("invalid tolerance was accepted")
