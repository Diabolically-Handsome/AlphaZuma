"""Curve-aware aim-settled teacher for the 55-level single-policy scope.

Preparing the next-colour aim before a swap fixes one-curve colour-flip loops,
but on a two-curve board the delay can neglect the more dangerous chain.  V3
therefore keeps V1's immediate swap behaviour whenever the actor observation
contains live balls on more than one curve, while retaining V2's prepared swap
on ordinary one-curve states.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from zuma_rl.revenge_env import FloatObservation, RevengeEnv
from zuma_rl.revenge_settled_strategic_teacher import (
    AimSettledStrategicRevengeTeacher,
)
from zuma_rl.revenge_teacher import RevengeTeacherSpec


class CurveAwareSettledStrategicRevengeTeacher(
    AimSettledStrategicRevengeTeacher
):
    """Prepare swap aim only while the visible state is single-curve."""

    @classmethod
    def from_env(
        cls, env: RevengeEnv
    ) -> "CurveAwareSettledStrategicRevengeTeacher":
        return cls(RevengeTeacherSpec.from_env(env))

    def _active_curve_count(self, observation: FloatObservation) -> int:
        matrix, _ = self._split_observation(observation)
        return len({ball.curve for ball in self._decode_balls(matrix)})

    def act(self, observation: FloatObservation) -> NDArray[np.int64]:
        action = np.asarray(super().act(observation), dtype=np.int64)
        if int(action[0]) != 2 or self._active_curve_count(observation) > 1:
            return action
        observed = self._observed_aim_bin(observation)
        target = int(action[1])
        if self._bin_distance(observed, target) <= self.aim_tolerance_bins:
            return action
        return np.asarray((0, target), dtype=np.int64)

