"""Strategic actor-observable teacher for full-55 engineering probes.

The policy uses the same packed actor observation and immutable geometry as
``ActorObservableRevengeTeacher``.  It adds only observation-derived ranking:
physical contact, immediate clears, rollback chain potential, power-up balls,
and skull-side danger.  It never reads simulator objects after construction.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray

from zuma_rl.revenge_env import FloatObservation, RevengeEnv
from zuma_rl.revenge_teacher import (
    ActorObservableRevengeTeacher,
    RevengeTeacherSpec,
)


@dataclass(frozen=True, slots=True)
class _StrategicCandidate:
    aim_bin: int
    run_length: int
    powerups: int
    chain_bonus: int
    danger: float
    entry_distance: float
    shot_distance: float

    @property
    def immediate_clear(self) -> bool:
        return self.run_length >= 2

    def rank(self) -> tuple[float, ...]:
        return (
            0.0 if self.immediate_clear else 1.0,
            -float(self.powerups),
            -float(self.chain_bonus),
            -float(self.run_length),
            -float(self.danger),
            float(self.entry_distance),
            float(self.shot_distance),
            float(self.aim_bin),
        )


class StrategicActorObservableRevengeTeacher(ActorObservableRevengeTeacher):
    """Prefer immediate, connected, chain-forming, high-value clear shots."""

    def __init__(self, spec: RevengeTeacherSpec):
        super().__init__(spec)
        required = {"waypoint", "contact_next_visible", "in_tunnel"}
        if not required.issubset(self._ball_index):
            raise ValueError("strategic teacher ball feature layout is incomplete")
        self._powerup_indices = tuple(
            index
            for name, index in self._ball_index.items()
            if name.startswith("powerup_")
        )

    @classmethod
    def from_env(cls, env: RevengeEnv) -> "StrategicActorObservableRevengeTeacher":
        return cls(RevengeTeacherSpec.from_env(env))

    def _contact_after(
        self,
        balls: tuple[object, ...],
        matrix: NDArray[np.float32],
        index: int,
    ) -> bool:
        ball = balls[index]
        return bool(
            float(
                matrix[
                    int(getattr(ball, "slot")),
                    self._ball_index["contact_next_visible"],
                ]
            )
            > 0.5
        )

    def _connected_bounds(
        self,
        balls: tuple[object, ...],
        matrix: NDArray[np.float32],
        index: int,
        color: int,
    ) -> tuple[int, int]:
        curve = int(getattr(balls[index], "curve"))
        left = index
        right = index
        while (
            left > 0
            and int(getattr(balls[left - 1], "curve")) == curve
            and int(getattr(balls[left - 1], "color")) == color
            and not bool(getattr(balls[left - 1], "exploding"))
            and self._contact_after(balls, matrix, left - 1)
        ):
            left -= 1
        while (
            right + 1 < len(balls)
            and int(getattr(balls[right + 1], "curve")) == curve
            and int(getattr(balls[right + 1], "color")) == color
            and not bool(getattr(balls[right + 1], "exploding"))
            and self._contact_after(balls, matrix, right)
        ):
            right += 1
        return left, right

    def _neighbour_run_length(
        self,
        balls: tuple[object, ...],
        matrix: NDArray[np.float32],
        index: int,
    ) -> int:
        color = int(getattr(balls[index], "color"))
        left, right = self._connected_bounds(balls, matrix, index, color)
        return right - left + 1

    def _strategic_candidate(
        self,
        balls: tuple[object, ...],
        matrix: NDArray[np.float32],
        color: int,
        shooter: tuple[float, float],
    ) -> _StrategicCandidate | None:
        candidates: list[_StrategicCandidate] = []
        for target in balls:
            if bool(getattr(target, "exploding")) or int(getattr(target, "color")) != color:
                continue
            centre = self._aim_bin(
                float(getattr(target, "x")),
                float(getattr(target, "y")),
                shooter,
            )
            for delta in (-1, 0, 1):
                aim_bin = (centre + delta) % self.spec.aim_bins
                hit = self._first_hit(balls, aim_bin, shooter)
                if hit is None:
                    continue
                hit_index, entry = hit
                first = balls[hit_index]
                if int(getattr(first, "color")) != color:
                    continue
                left, right = self._connected_bounds(
                    balls, matrix, hit_index, color
                )
                run_length = right - left + 1
                powerups = 0
                danger = -1.0
                for position in range(left, right + 1):
                    row = matrix[int(getattr(balls[position], "slot"))]
                    powerups += int(
                        any(float(row[index]) > 0.5 for index in self._powerup_indices)
                    )
                    danger = max(
                        danger,
                        float(row[self._ball_index["waypoint"]]),
                    )
                chain_bonus = 0
                if left > 0 and right + 1 < len(balls):
                    before = balls[left - 1]
                    after = balls[right + 1]
                    if (
                        int(getattr(before, "curve"))
                        == int(getattr(first, "curve"))
                        == int(getattr(after, "curve"))
                        and not bool(getattr(before, "exploding"))
                        and not bool(getattr(after, "exploding"))
                        and int(getattr(before, "color"))
                        == int(getattr(after, "color"))
                    ):
                        chain_bonus = self._neighbour_run_length(
                            balls, matrix, left - 1
                        ) + self._neighbour_run_length(
                            balls, matrix, right + 1
                        )
                candidates.append(
                    _StrategicCandidate(
                        aim_bin=aim_bin,
                        run_length=run_length,
                        powerups=powerups,
                        chain_bonus=chain_bonus,
                        danger=danger,
                        entry_distance=float(entry),
                        shot_distance=math.hypot(
                            float(getattr(target, "x")) - shooter[0],
                            float(getattr(target, "y")) - shooter[1],
                        ),
                    )
                )
        return min(candidates, key=lambda item: item.rank()) if candidates else None

    def _strategic_fallback(
        self,
        balls: tuple[object, ...],
        matrix: NDArray[np.float32],
        shooter: tuple[float, float],
    ) -> _StrategicCandidate | None:
        candidates: list[_StrategicCandidate] = []
        for target in balls:
            if bool(getattr(target, "exploding")):
                continue
            aim_bin = self._aim_bin(
                float(getattr(target, "x")),
                float(getattr(target, "y")),
                shooter,
            )
            hit = self._first_hit(balls, aim_bin, shooter)
            if hit is None:
                continue
            hit_index, entry = hit
            first = balls[hit_index]
            row = matrix[int(getattr(first, "slot"))]
            powerups = int(
                any(float(row[index]) > 0.5 for index in self._powerup_indices)
            )
            candidates.append(
                _StrategicCandidate(
                    aim_bin=aim_bin,
                    run_length=0,
                    powerups=powerups,
                    chain_bonus=0,
                    danger=float(row[self._ball_index["waypoint"]]),
                    entry_distance=float(entry),
                    shot_distance=math.hypot(
                        float(getattr(target, "x")) - shooter[0],
                        float(getattr(target, "y")) - shooter[1],
                    ),
                )
            )
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda item: (
                -item.powerups,
                -item.danger,
                item.entry_distance,
                item.shot_distance,
                item.aim_bin,
            ),
        )

    def act(self, observation: FloatObservation) -> NDArray[np.int64]:
        matrix, globals_ = self._split_observation(observation)
        aim_angle = math.atan2(
            float(globals_[self._global_index["aim_sin"]]),
            float(globals_[self._global_index["aim_cos"]]),
        ) % math.tau
        aim_bin = int(
            math.floor(aim_angle * self.spec.aim_bins / math.tau)
        ) % self.spec.aim_bins
        if float(globals_[self._global_index["gun_normal"]]) <= 0.5:
            return np.asarray((0, aim_bin), dtype=np.int64)
        current = self._decode_color(
            globals_, "current_color", missing_name="current_missing"
        )
        if current is None:
            return np.asarray((0, aim_bin), dtype=np.int64)
        next_color = self._decode_color(
            globals_, "next_color", missing_name="next_missing"
        )
        balls = self._decode_balls(matrix)
        if not balls:
            return np.asarray((0, aim_bin), dtype=np.int64)
        shooter = (self.spec.shooter_x, self.spec.shooter_y)
        if {"shooter_x", "shooter_y"}.issubset(self._global_index):
            shooter = (
                (float(globals_[self._global_index["shooter_x"]]) + 1.0)
                * self.spec.logical_width
                / 2.0,
                (float(globals_[self._global_index["shooter_y"]]) + 1.0)
                * self.spec.logical_height
                / 2.0,
            )
        current_match = self._strategic_candidate(
            balls, matrix, current, shooter
        )
        next_match = (
            self._strategic_candidate(balls, matrix, next_color, shooter)
            if next_color is not None and next_color != current
            else None
        )
        if len(self.spec.shooter_positions) > 1:
            nearest = min(
                range(len(self.spec.shooter_positions)),
                key=lambda index: math.hypot(
                    self.spec.shooter_positions[index][0] - shooter[0],
                    self.spec.shooter_positions[index][1] - shooter[1],
                ),
            )
            other_shooter = self.spec.shooter_positions[
                (nearest + 1) % len(self.spec.shooter_positions)
            ]
            other_match = self._strategic_candidate(
                balls, matrix, current, other_shooter
            )
            if other_match is not None and (
                current_match is None
                or other_match.rank()[:5] < current_match.rank()[:5]
            ):
                return np.asarray((3, aim_bin), dtype=np.int64)
        if next_match is not None and (
            current_match is None
            or (
                next_match.immediate_clear
                and not current_match.immediate_clear
            )
            or (
                next_match.immediate_clear
                and current_match.immediate_clear
                and next_match.rank()[:5] < current_match.rank()[:5]
            )
        ):
            return np.asarray((2, next_match.aim_bin), dtype=np.int64)
        target = current_match or self._strategic_fallback(
            balls, matrix, shooter
        )
        if target is None:
            return np.asarray((0, aim_bin), dtype=np.int64)
        return np.asarray((1, target.aim_bin), dtype=np.int64)
