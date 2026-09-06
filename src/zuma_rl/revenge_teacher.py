"""Actor-observable geometric teacher for the fidelity-first environment.

The teacher is a capacity and behavior-cloning aid, not a transfer policy.  It
receives only the packed actor observation.  Shooter location, logical canvas
size, collision radius, and feature layout are immutable level/interface
constants captured once when the teacher is constructed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from zuma_rl.revenge_env import FloatObservation, RevengeEnv


@dataclass(frozen=True, slots=True)
class RevengeTeacherSpec:
    aim_bins: int
    max_balls: int
    ball_feature_names: tuple[str, ...]
    global_feature_names: tuple[str, ...]
    balls_start: int
    balls_stop: int
    globals_start: int
    globals_stop: int
    logical_width: float
    logical_height: float
    shooter_x: float
    shooter_y: float
    shooter_positions: tuple[tuple[float, float], ...]
    collision_radius: float
    num_colors: int

    @classmethod
    def from_env(cls, env: RevengeEnv) -> "RevengeTeacherSpec":
        if env.config.action_mode != "factorized":
            raise ValueError("the actor teacher requires factorized actions")
        balls = env.observation_layout["balls"]
        globals_ = env.observation_layout["globals"]
        return cls(
            aim_bins=int(env.config.aim_bins),
            max_balls=int(env.config.max_balls),
            ball_feature_names=tuple(env.ball_feature_names),
            global_feature_names=tuple(env.global_feature_names),
            balls_start=int(balls.start or 0),
            balls_stop=int(balls.stop or env.observation_size),
            globals_start=int(globals_.start or 0),
            globals_stop=int(globals_.stop or env.observation_size),
            logical_width=float(env.sim.config.logical_width),
            logical_height=float(env.sim.config.logical_height),
            shooter_x=float(env.sim.shooter[0]),
            shooter_y=float(env.sim.shooter[1]),
            shooter_positions=tuple(
                (float(position[0]), float(position[1]))
                for position in env.sim.shooter_positions
            ),
            # Projectile and chain-ball radii are equal in the supported scope.
            collision_radius=2.0 * float(env.sim.config.ball_radius),
            num_colors=int(env.color_feature_count),
        )


@dataclass(frozen=True, slots=True)
class _VisibleBall:
    slot: int
    x: float
    y: float
    color: int
    curve: int
    exploding: bool


@dataclass(frozen=True, slots=True)
class _AimCandidate:
    aim_bin: int
    run_length: int
    entry_distance: float
    shot_distance: float


class ActorObservableRevengeTeacher:
    """Choose unobstructed same-colour shots from one actor observation."""

    def __init__(self, spec: RevengeTeacherSpec):
        self.spec = spec
        self._ball_index = {
            name: index for index, name in enumerate(spec.ball_feature_names)
        }
        self._global_index = {
            name: index for index, name in enumerate(spec.global_feature_names)
        }
        required_ball = {"present", "x", "y", "exploding"}
        required_global = {
            "aim_sin",
            "aim_cos",
            "current_missing",
            "next_missing",
            "gun_normal",
        }
        if not required_ball.issubset(self._ball_index):
            raise ValueError("teacher ball feature layout is incomplete")
        if not required_global.issubset(self._global_index):
            raise ValueError("teacher global feature layout is incomplete")
        for color in range(spec.num_colors):
            if f"color_{color}" not in self._ball_index:
                raise ValueError("teacher ball color layout is incomplete")
            if f"current_color_{color}" not in self._global_index:
                raise ValueError("teacher current-color layout is incomplete")
            if f"next_color_{color}" not in self._global_index:
                raise ValueError("teacher next-color layout is incomplete")

    @classmethod
    def from_env(cls, env: RevengeEnv) -> "ActorObservableRevengeTeacher":
        return cls(RevengeTeacherSpec.from_env(env))

    def _split_observation(
        self,
        observation: FloatObservation,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        values = np.asarray(observation, dtype=np.float32)
        if values.ndim != 1 or values.size != self.spec.globals_stop:
            raise ValueError("teacher observation shape does not match its spec")
        balls = values[
            self.spec.balls_start : self.spec.balls_stop
        ].reshape(self.spec.max_balls, len(self.spec.ball_feature_names))
        globals_ = values[self.spec.globals_start : self.spec.globals_stop]
        return balls, globals_

    def _decode_color(
        self,
        values: NDArray[np.float32],
        prefix: str,
        *,
        missing_name: str,
    ) -> int | None:
        if float(values[self._global_index[missing_name]]) > 0.5:
            return None
        channels = np.asarray(
            [
                values[self._global_index[f"{prefix}_{color}"]]
                for color in range(self.spec.num_colors)
            ],
            dtype=np.float32,
        )
        if float(np.max(channels)) <= 0.5:
            return None
        return int(np.argmax(channels))

    def _decode_balls(
        self,
        matrix: NDArray[np.float32],
    ) -> tuple[_VisibleBall, ...]:
        result: list[_VisibleBall] = []
        curve_names = tuple(
            name
            for name in self.spec.ball_feature_names
            if name.startswith("curve_")
        )
        for slot, row in enumerate(matrix):
            if float(row[self._ball_index["present"]]) <= 0.5:
                continue
            color_channels = np.asarray(
                [
                    row[self._ball_index[f"color_{color}"]]
                    for color in range(self.spec.num_colors)
                ],
                dtype=np.float32,
            )
            if float(np.max(color_channels)) <= 0.5:
                continue
            curve = 0
            if curve_names:
                curve = int(
                    np.argmax(
                        [row[self._ball_index[name]] for name in curve_names]
                    )
                )
            result.append(
                _VisibleBall(
                    slot=slot,
                    x=(float(row[self._ball_index["x"]]) + 1.0)
                    * self.spec.logical_width
                    / 2.0,
                    y=(float(row[self._ball_index["y"]]) + 1.0)
                    * self.spec.logical_height
                    / 2.0,
                    color=int(np.argmax(color_channels)),
                    curve=curve,
                    exploding=(
                        float(row[self._ball_index["exploding"]]) > 0.5
                    ),
                )
            )
        return tuple(result)

    def _aim_bin(
        self,
        x: float,
        y: float,
        shooter: tuple[float, float],
    ) -> int:
        angle = math.atan2(
            y - shooter[1],
            x - shooter[0],
        ) % math.tau
        return int(math.floor(angle * self.spec.aim_bins / math.tau)) % (
            self.spec.aim_bins
        )

    def _first_hit(
        self,
        balls: tuple[_VisibleBall, ...],
        aim_bin: int,
        shooter: tuple[float, float],
    ) -> tuple[int, float] | None:
        angle = math.tau * (aim_bin + 0.5) / self.spec.aim_bins
        dx = math.cos(angle)
        dy = math.sin(angle)
        radius_sq = self.spec.collision_radius**2
        hits: list[tuple[float, int]] = []
        for index, ball in enumerate(balls):
            if ball.exploding:
                continue
            bx = ball.x - shooter[0]
            by = ball.y - shooter[1]
            projection = bx * dx + by * dy
            if projection <= 0.0:
                continue
            perpendicular_sq = bx * bx + by * by - projection * projection
            if perpendicular_sq > radius_sq:
                continue
            entry = projection - math.sqrt(
                max(0.0, radius_sq - perpendicular_sq)
            )
            hits.append((entry, index))
        if not hits:
            return None
        entry, index = min(hits)
        return index, entry

    @staticmethod
    def _run_length(
        balls: tuple[_VisibleBall, ...],
        index: int,
        color: int,
    ) -> int:
        curve = balls[index].curve
        left = index
        right = index
        while (
            left > 0
            and balls[left - 1].curve == curve
            and balls[left - 1].color == color
            and not balls[left - 1].exploding
        ):
            left -= 1
        while (
            right + 1 < len(balls)
            and balls[right + 1].curve == curve
            and balls[right + 1].color == color
            and not balls[right + 1].exploding
        ):
            right += 1
        return right - left + 1

    def _same_color_candidate(
        self,
        balls: tuple[_VisibleBall, ...],
        color: int,
        shooter: tuple[float, float],
    ) -> _AimCandidate | None:
        candidates: list[_AimCandidate] = []
        for target in balls:
            if target.exploding or target.color != color:
                continue
            centre = self._aim_bin(target.x, target.y, shooter)
            for delta in (-1, 0, 1):
                aim_bin = (centre + delta) % self.spec.aim_bins
                hit = self._first_hit(balls, aim_bin, shooter)
                if hit is None:
                    continue
                hit_index, entry = hit
                first = balls[hit_index]
                if first.color != color:
                    continue
                candidates.append(
                    _AimCandidate(
                        aim_bin=aim_bin,
                        run_length=self._run_length(balls, hit_index, color),
                        entry_distance=entry,
                        shot_distance=math.hypot(
                            target.x - shooter[0],
                            target.y - shooter[1],
                        ),
                    )
                )
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda item: (
                -item.run_length,
                item.entry_distance,
                item.shot_distance,
                item.aim_bin,
            ),
        )

    def _fallback_candidate(
        self,
        balls: tuple[_VisibleBall, ...],
        shooter: tuple[float, float],
    ) -> _AimCandidate | None:
        candidates: list[_AimCandidate] = []
        for target in balls:
            if target.exploding:
                continue
            aim_bin = self._aim_bin(target.x, target.y, shooter)
            hit = self._first_hit(balls, aim_bin, shooter)
            if hit is None:
                continue
            _, entry = hit
            candidates.append(
                _AimCandidate(
                    aim_bin=aim_bin,
                    run_length=0,
                    entry_distance=entry,
                    shot_distance=math.hypot(
                        target.x - shooter[0],
                        target.y - shooter[1],
                    ),
                )
            )
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda item: (
                item.entry_distance,
                item.shot_distance,
                item.aim_bin,
            ),
        )

    def act(self, observation: FloatObservation) -> NDArray[np.int64]:
        """Return one factorized ``[verb, aim_bin]`` action."""

        ball_matrix, globals_ = self._split_observation(observation)
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
            globals_,
            "current_color",
            missing_name="current_missing",
        )
        if current is None:
            return np.asarray((0, aim_bin), dtype=np.int64)
        next_color = self._decode_color(
            globals_,
            "next_color",
            missing_name="next_missing",
        )
        balls = self._decode_balls(ball_matrix)
        if not balls:
            return np.asarray((0, aim_bin), dtype=np.int64)

        shooter = (self.spec.shooter_x, self.spec.shooter_y)
        if {"shooter_x", "shooter_y"}.issubset(self._global_index):
            shooter = (
                (
                    float(globals_[self._global_index["shooter_x"]]) + 1.0
                )
                * self.spec.logical_width
                / 2.0,
                (
                    float(globals_[self._global_index["shooter_y"]]) + 1.0
                )
                * self.spec.logical_height
                / 2.0,
            )

        current_match = self._same_color_candidate(balls, current, shooter)
        next_match = (
            self._same_color_candidate(balls, next_color, shooter)
            if next_color is not None and next_color != current
            else None
        )
        if len(self.spec.shooter_positions) > 1:
            other_index = min(
                range(len(self.spec.shooter_positions)),
                key=lambda index: math.hypot(
                    self.spec.shooter_positions[index][0] - shooter[0],
                    self.spec.shooter_positions[index][1] - shooter[1],
                ),
            )
            other_index = (other_index + 1) % len(self.spec.shooter_positions)
            other_shooter = self.spec.shooter_positions[other_index]
            other_match = self._same_color_candidate(
                balls,
                current,
                other_shooter,
            )
            current_rank = (
                (math.inf, math.inf, math.inf, math.inf)
                if current_match is None
                else (
                    -current_match.run_length,
                    current_match.entry_distance,
                    current_match.shot_distance,
                    current_match.aim_bin,
                )
            )
            other_rank = (
                (math.inf, math.inf, math.inf, math.inf)
                if other_match is None
                else (
                    -other_match.run_length,
                    other_match.entry_distance,
                    other_match.shot_distance,
                    other_match.aim_bin,
                )
            )
            if other_rank < current_rank:
                return np.asarray((3, aim_bin), dtype=np.int64)
        current_run = 0 if current_match is None else current_match.run_length
        if (
            next_match is not None
            and next_match.run_length >= 2
            and next_match.run_length > current_run
        ):
            return np.asarray((2, next_match.aim_bin), dtype=np.int64)

        target = current_match or self._fallback_candidate(balls, shooter)
        if target is None:
            return np.asarray((0, aim_bin), dtype=np.int64)
        return np.asarray((1, target.aim_bin), dtype=np.int64)
