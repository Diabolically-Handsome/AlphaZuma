"""Pure simulation logic for a compact Zuma-like game."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from zuma_rl.config import ZumaConfig

FloatArray = NDArray[np.float64]


def resolve_matches(
    colors: Sequence[int],
    insertion_index: int,
    match_size: int = 3,
) -> tuple[list[int], int, int]:
    """Resolve the run containing an inserted ball and any boundary cascades.

    Args:
        colors: Chain ordered from entrance/tail to skull/head, after insertion.
        insertion_index: Index of the newly inserted ball.
        match_size: Minimum contiguous run that disappears.

    Returns:
        ``(remaining_colors, removed_count, cascade_count)``.  The first
        disappearance counts as cascade 1.
    """

    chain = [int(color) for color in colors]
    if not chain:
        return chain, 0, 0
    if not 0 <= insertion_index < len(chain):
        raise IndexError("insertion_index is outside the chain")
    if match_size < 2:
        raise ValueError("match_size must be at least 2")

    def run_bounds(index: int) -> tuple[int, int]:
        color = chain[index]
        left = index
        right = index + 1
        while left > 0 and chain[left - 1] == color:
            left -= 1
        while right < len(chain) and chain[right] == color:
            right += 1
        return left, right

    left, right = run_bounds(insertion_index)
    if right - left < match_size:
        return chain, 0, 0

    removed = right - left
    cascades = 1
    del chain[left:right]
    boundary = left

    while 0 < boundary < len(chain):
        if chain[boundary - 1] != chain[boundary]:
            break
        run_left = boundary - 1
        run_right = boundary + 1
        color = chain[run_left]
        while run_left > 0 and chain[run_left - 1] == color:
            run_left -= 1
        while run_right < len(chain) and chain[run_right] == color:
            run_right += 1
        run_length = run_right - run_left
        if run_length < match_size:
            break
        removed += run_length
        cascades += 1
        del chain[run_left:run_right]
        boundary = run_left

    return chain, removed, cascades


class SpiralTrack:
    """A two-dimensional spiral indexed by normalized arc length."""

    def __init__(self, config: ZumaConfig):
        theta = np.linspace(
            -0.15 * np.pi,
            -0.15 * np.pi + config.track_turns * 2.0 * np.pi,
            config.track_samples,
            dtype=np.float64,
        )
        radius = np.linspace(
            config.track_start_radius,
            config.track_end_radius,
            config.track_samples,
            dtype=np.float64,
        )
        points = np.column_stack(
            (
                config.shooter_x + radius * np.cos(theta),
                config.shooter_y
                + config.track_vertical_scale * radius * np.sin(theta),
            )
        )
        segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
        self.total_length = float(cumulative[-1])
        self._progress = cumulative / self.total_length
        self._points = points
        self._first_tangent = self._unit(points[1] - points[0])
        self._last_tangent = self._unit(points[-1] - points[-2])

    @staticmethod
    def _unit(vector: FloatArray) -> FloatArray:
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            return np.array([1.0, 0.0], dtype=np.float64)
        return vector / norm

    def points_at(self, progress: Iterable[float] | FloatArray) -> FloatArray:
        """Return ``(x, y)`` coordinates, extrapolating outside both ends."""

        values = np.asarray(progress, dtype=np.float64)
        flat = values.reshape(-1)
        clipped = np.clip(flat, 0.0, 1.0)
        x = np.interp(clipped, self._progress, self._points[:, 0])
        y = np.interp(clipped, self._progress, self._points[:, 1])
        result = np.column_stack((x, y))

        before = flat < 0.0
        if np.any(before):
            result[before] = (
                self._points[0]
                + flat[before, None] * self.total_length * self._first_tangent
            )
        after = flat > 1.0
        if np.any(after):
            result[after] = (
                self._points[-1]
                + (flat[after, None] - 1.0)
                * self.total_length
                * self._last_tangent
            )
        return result.reshape(values.shape + (2,))

    def point_at(self, progress: float) -> FloatArray:
        return self.points_at(np.asarray(progress)).reshape(2)

    def tangent_at(self, progress: float) -> FloatArray:
        """Return the unit tangent in the direction of chain travel."""

        if progress <= 0.0:
            return self._first_tangent.copy()
        if progress >= 1.0:
            return self._last_tangent.copy()
        index = int(np.searchsorted(self._progress, progress, side="right") - 1)
        index = min(max(index, 0), len(self._points) - 2)
        return self._unit(self._points[index + 1] - self._points[index])

    def sample(self, count: int = 256) -> FloatArray:
        return self.points_at(np.linspace(0.0, 1.0, count))


@dataclass(frozen=True, slots=True)
class StepResult:
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, int | float | bool | str | None]


class ZumaSimulator:
    """Stateful game simulator independent of Gymnasium."""

    def __init__(
        self,
        config: ZumaConfig | None = None,
        *,
        rng: np.random.Generator | None = None,
        track: SpiralTrack | None = None,
    ):
        self.config = config or ZumaConfig()
        self.rng = rng or np.random.default_rng()
        self.track = track or SpiralTrack(self.config)

        self.balls: list[int] = []
        self.front_progress = self.config.initial_front_progress
        self.current_color = 0
        self.next_color = 0
        self.steps = 0
        self.score = 0
        self.done = False
        self.outcome: str | None = None
        self.last_removed = 0
        self.last_cascades = 0
        self.reset()

    @property
    def shooter(self) -> FloatArray:
        return np.array(
            [self.config.shooter_x, self.config.shooter_y],
            dtype=np.float64,
        )

    def reset(self) -> None:
        self.balls = self._initial_chain()
        self.front_progress = self.config.initial_front_progress
        self.steps = 0
        self.score = 0
        self.done = False
        self.outcome = None
        self.last_removed = 0
        self.last_cascades = 0
        self.current_color = self._sample_shooter_color()
        self.next_color = self._sample_shooter_color()

    def load_state(
        self,
        balls: Sequence[int],
        *,
        front_progress: float | None = None,
        current_color: int | None = None,
        next_color: int | None = None,
        steps: int = 0,
    ) -> None:
        """Load a validated state for tests, curricula, or handcrafted levels."""

        values = [int(color) for color in balls]
        if any(not 0 <= color < self.config.num_colors for color in values):
            raise ValueError("ball colors must be in [0, num_colors)")
        if len(values) > self.config.max_balls + 1:
            raise ValueError("loaded chain exceeds max_balls + 1")
        self.balls = values
        self.front_progress = (
            self.config.initial_front_progress
            if front_progress is None
            else float(front_progress)
        )
        if current_color is not None:
            self._validate_color(current_color)
            self.current_color = int(current_color)
        if next_color is not None:
            self._validate_color(next_color)
            self.next_color = int(next_color)
        self.steps = int(steps)
        self.score = 0
        self.done = False
        self.outcome = None
        self.last_removed = 0
        self.last_cascades = 0

    def _validate_color(self, color: int) -> None:
        if not 0 <= int(color) < self.config.num_colors:
            raise ValueError("color must be in [0, num_colors)")

    def _initial_chain(self) -> list[int]:
        result: list[int] = []
        for _ in range(self.config.initial_balls):
            candidates = list(range(self.config.num_colors))
            if (
                len(result) >= self.config.match_size - 1
                and len(set(result[-(self.config.match_size - 1) :])) == 1
            ):
                candidates.remove(result[-1])
            result.append(int(self.rng.choice(candidates)))
        return result

    def _sample_shooter_color(self) -> int:
        if self.balls and self.rng.random() < self.config.chain_color_probability:
            return int(self.rng.choice(self.balls))
        return int(self.rng.integers(self.config.num_colors))

    def ball_progresses(self) -> FloatArray:
        count = len(self.balls)
        if count == 0:
            return np.empty(0, dtype=np.float64)
        offsets = np.arange(count - 1, -1, -1, dtype=np.float64)
        return self.front_progress - offsets * self.config.ball_spacing

    def ball_positions(self) -> FloatArray:
        return self.track.points_at(self.ball_progresses())

    def action_to_angle(self, action: int) -> tuple[float, bool]:
        if not 0 <= int(action) < self.config.action_count:
            raise ValueError("action is outside the discrete action range")
        swap = int(action) >= self.config.aim_bins
        aim_bin = int(action) % self.config.aim_bins
        angle = 2.0 * np.pi * (aim_bin + 0.5) / self.config.aim_bins
        return float(angle), swap

    def action_towards(self, point: Sequence[float], *, swap: bool = False) -> int:
        """Return the closest discrete action pointing at a world coordinate."""

        vector = np.asarray(point, dtype=np.float64) - self.shooter
        angle = float(np.arctan2(vector[1], vector[0]) % (2.0 * np.pi))
        aim_bin = int(np.floor(angle * self.config.aim_bins / (2.0 * np.pi)))
        aim_bin = min(aim_bin, self.config.aim_bins - 1)
        return aim_bin + (self.config.aim_bins if swap else 0)

    def _ray_hit(self, angle: float) -> int | None:
        positions = self.ball_positions()
        if len(positions) == 0:
            return None
        direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
        relative = positions - self.shooter
        projection = relative @ direction
        perpendicular_sq = np.einsum("ij,ij->i", relative, relative) - projection**2
        radius_sq = self.config.ball_radius**2
        valid = (projection > 0.0) & (perpendicular_sq <= radius_sq)
        if not np.any(valid):
            return None
        entry = np.full(len(positions), np.inf, dtype=np.float64)
        entry[valid] = projection[valid] - np.sqrt(
            np.maximum(0.0, radius_sq - perpendicular_sq[valid])
        )
        return int(np.argmin(entry))

    def _insertion_index(self, hit_index: int, angle: float) -> int:
        progresses = self.ball_progresses()
        tangent = self.track.tangent_at(float(progresses[hit_index]))
        direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
        approaches_from_tail = float(direction @ tangent) >= 0.0
        return hit_index if approaches_from_tail else hit_index + 1

    def step(self, action: int) -> StepResult:
        if self.done:
            raise RuntimeError("step() called after episode ended; reset first")

        angle, swap = self.action_to_angle(action)
        if swap:
            self.current_color, self.next_color = (
                self.next_color,
                self.current_color,
            )

        self.steps += 1
        self.last_removed = 0
        self.last_cascades = 0
        reward = self.config.reward_step
        fired_color = self.current_color
        hit_index = self._ray_hit(angle)
        insertion_index: int | None = None

        if hit_index is None:
            reward += self.config.reward_miss
        else:
            insertion_index = self._insertion_index(hit_index, angle)
            self.balls.insert(insertion_index, fired_color)
            self.balls, removed, cascades = resolve_matches(
                self.balls,
                insertion_index,
                self.config.match_size,
            )
            self.last_removed = removed
            self.last_cascades = cascades
            if removed:
                self.front_progress -= (
                    removed * self.config.retreat_per_removed_ball
                )
                gained = removed * 10 * cascades
                self.score += gained
                reward += removed * self.config.reward_per_removed_ball
                reward += max(0, cascades - 1) * self.config.reward_cascade_bonus
            else:
                reward += self.config.reward_insert_without_match

        self.current_color = self.next_color
        self.next_color = self._sample_shooter_color()

        terminated = False
        truncated = False
        if not self.balls:
            terminated = True
            self.outcome = "win"
            reward += self.config.reward_win
        else:
            self.front_progress += self.config.chain_speed_per_shot
            if self.front_progress >= 1.0:
                terminated = True
                self.outcome = "skull"
                reward += self.config.reward_loss
            elif len(self.balls) > self.config.max_balls:
                terminated = True
                self.outcome = "overflow"
                reward += self.config.reward_loss

        if not terminated and self.steps >= self.config.max_steps:
            truncated = True
            self.outcome = "time_limit"

        self.done = terminated or truncated
        info: dict[str, int | float | bool | str | None] = {
            "outcome": self.outcome,
            "score": self.score,
            "steps": self.steps,
            "chain_length": len(self.balls),
            "front_progress": self.front_progress,
            "fired_color": fired_color,
            "swapped": swap,
            "hit": hit_index is not None,
            "hit_index": hit_index,
            "insertion_index": insertion_index,
            "removed": self.last_removed,
            "cascades": self.last_cascades,
        }
        return StepResult(
            reward=float(reward),
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

