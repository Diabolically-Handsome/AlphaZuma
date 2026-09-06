"""Second aim-settled teacher with swap-edge target preparation.

V1 waits for cursor alignment before fire edges.  A swap can expose the next
ball's immediate-clear target just as the delayed cursor starts turning away,
which may cause repeated colour flips instead of the intended swap-and-fire
sequence.  V2 prepares the next-ball target before it queues the swap edge.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from zuma_rl.revenge_env import FloatObservation, RevengeEnv
from zuma_rl.revenge_settled_strategic_teacher import (
    AimSettledStrategicRevengeTeacher,
)
from zuma_rl.revenge_teacher import RevengeTeacherSpec


class AimSettledStrategicRevengeTeacherV2(
    AimSettledStrategicRevengeTeacher
):
    """Gate both fire and swap edges on the selected target aim."""

    @classmethod
    def from_env(
        cls, env: RevengeEnv
    ) -> "AimSettledStrategicRevengeTeacherV2":
        return cls(RevengeTeacherSpec.from_env(env))

    def act(self, observation: FloatObservation) -> NDArray[np.int64]:
        action = np.asarray(super().act(observation), dtype=np.int64)
        if int(action[0]) != 2:
            return action
        observed = self._observed_aim_bin(observation)
        target = int(action[1])
        if self._bin_distance(observed, target) <= self.aim_tolerance_bins:
            return action
        return np.asarray((0, target), dtype=np.int64)

