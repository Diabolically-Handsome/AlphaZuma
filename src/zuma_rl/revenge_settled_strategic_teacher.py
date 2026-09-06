"""Aim-settled strategic teacher for engineering-only AlphaZuma probes.

The canonical human input wrapper delays both aim and button intentions.  A
stateless teacher that requests fire before the delayed cursor reaches its
target therefore launches the queued shot along an unrelated intermediate
angle.  This teacher preserves the strategic target selection and merely
holds fire until the actor-observable aim is close to the selected bin.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from zuma_rl.revenge_env import FloatObservation, RevengeEnv
from zuma_rl.revenge_strategic_teacher import (
    StrategicActorObservableRevengeTeacher,
)
from zuma_rl.revenge_teacher import RevengeTeacherSpec


class AimSettledStrategicRevengeTeacher(
    StrategicActorObservableRevengeTeacher
):
    """Gate fire edges until the visible cursor is near the chosen aim bin."""

    def __init__(
        self,
        spec: RevengeTeacherSpec,
        *,
        aim_tolerance_bins: int = 2,
    ) -> None:
        super().__init__(spec)
        tolerance = int(aim_tolerance_bins)
        if not 0 <= tolerance < spec.aim_bins // 2:
            raise ValueError("aim_tolerance_bins is outside the useful range")
        self.aim_tolerance_bins = tolerance

    @classmethod
    def from_env(cls, env: RevengeEnv) -> "AimSettledStrategicRevengeTeacher":
        return cls(RevengeTeacherSpec.from_env(env))

    def _observed_aim_bin(self, observation: FloatObservation) -> int:
        _, globals_ = self._split_observation(observation)
        angle = math.atan2(
            float(globals_[self._global_index["aim_sin"]]),
            float(globals_[self._global_index["aim_cos"]]),
        ) % math.tau
        return int(math.floor(angle * self.spec.aim_bins / math.tau)) % (
            self.spec.aim_bins
        )

    def _bin_distance(self, first: int, second: int) -> int:
        direct = abs(int(first) - int(second)) % self.spec.aim_bins
        return min(direct, self.spec.aim_bins - direct)

    def act(self, observation: FloatObservation) -> NDArray[np.int64]:
        action = np.asarray(super().act(observation), dtype=np.int64)
        if int(action[0]) != 1:
            return action
        observed = self._observed_aim_bin(observation)
        target = int(action[1])
        if self._bin_distance(observed, target) <= self.aim_tolerance_bins:
            return action
        return np.asarray((0, target), dtype=np.int64)

