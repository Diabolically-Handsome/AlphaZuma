"""Gymnasium adapter and optional debug renderer."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from zuma_rl.config import ZumaConfig
from zuma_rl.core import ZumaSimulator

RgbArray = NDArray[np.uint8]


class ZumaEnv(gym.Env[NDArray[np.float32], int]):
    """A fixed-size vector observation and discrete ``aim × swap`` action."""

    metadata = {
        "render_modes": ["rgb_array", "ansi"],
        "render_fps": 12,
    }

    def __init__(
        self,
        config: ZumaConfig | None = None,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.config = config or ZumaConfig()
        if render_mode not in {None, *self.metadata["render_modes"]}:
            raise ValueError(f"unsupported render_mode: {render_mode}")
        self.render_mode = render_mode
        self.sim = ZumaSimulator(self.config)

        self.ball_feature_size = 4 + self.config.num_colors
        self.global_feature_size = 3 + 2 * self.config.num_colors
        observation_size = (
            self.config.max_balls * self.ball_feature_size
            + self.global_feature_size
        )
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(observation_size,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.config.action_count)

    @property
    def observation_size(self) -> int:
        return int(self.observation_space.shape[0])

    def _observation(self) -> NDArray[np.float32]:
        ball_features = np.zeros(
            (self.config.max_balls, self.ball_feature_size),
            dtype=np.float32,
        )
        count = min(len(self.sim.balls), self.config.max_balls)
        if count:
            positions = self.sim.ball_positions()[:count]
            progresses = self.sim.ball_progresses()[:count]
            colors = np.asarray(self.sim.balls[:count], dtype=np.int64)
            ball_features[:count, 0] = 1.0
            ball_features[:count, 1:3] = np.clip(
                positions * 2.0 - 1.0,
                -1.0,
                1.0,
            )
            ball_features[:count, 3] = np.clip(progresses, -1.0, 1.0)
            ball_features[np.arange(count), 4 + colors] = 1.0

        globals_ = np.zeros(self.global_feature_size, dtype=np.float32)
        globals_[0] = np.clip(self.sim.front_progress * 2.0 - 1.0, -1.0, 1.0)
        globals_[1] = min(len(self.sim.balls), self.config.max_balls) / float(
            self.config.max_balls
        )
        globals_[2] = 1.0 - 2.0 * min(
            self.sim.steps / float(self.config.max_steps),
            1.0,
        )
        globals_[3 + self.sim.current_color] = 1.0
        next_start = 3 + self.config.num_colors
        globals_[next_start + self.sim.next_color] = 1.0
        return np.concatenate((ball_features.reshape(-1), globals_))

    def _info(self) -> dict[str, int | float | str | None]:
        return {
            "outcome": self.sim.outcome,
            "score": self.sim.score,
            "steps": self.sim.steps,
            "chain_length": len(self.sim.balls),
            "front_progress": self.sim.front_progress,
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, int | float | str | None]]:
        super().reset(seed=seed)
        self.sim.rng = self.np_random
        self.sim.reset()

        if options and "state" in options:
            state = options["state"]
            if not isinstance(state, dict):
                raise TypeError("options['state'] must be a dictionary")
            self.sim.load_state(**state)

        return self._observation(), self._info()

    def step(
        self,
        action: int | np.integer[Any],
    ) -> tuple[
        NDArray[np.float32],
        float,
        bool,
        bool,
        dict[str, int | float | bool | str | None],
    ]:
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action: {action}")
        result = self.sim.step(int(action))
        return (
            self._observation(),
            result.reward,
            result.terminated,
            result.truncated,
            result.info,
        )

    def render(self) -> RgbArray | str | None:
        if self.render_mode is None:
            return None
        if self.render_mode == "ansi":
            chain = "".join(str(color) for color in self.sim.balls)
            return (
                f"step={self.sim.steps:03d} front={self.sim.front_progress:.3f} "
                f"current={self.sim.current_color} next={self.sim.next_color} "
                f"score={self.sim.score} chain={chain}"
            )
        return self._render_rgb()

    def _render_rgb(self) -> RgbArray:
        size = self.config.render_size
        canvas = np.empty((size, size, 3), dtype=np.uint8)
        canvas[:] = (14, 18, 27)

        track_points = self.sim.track.sample(600)
        for point in track_points:
            self._draw_disk(canvas, point, 2, (71, 78, 91))

        end = self.sim.track.point_at(1.0)
        self._draw_disk(canvas, end, 8, (180, 55, 55))
        self._draw_disk(canvas, self.sim.shooter, 12, (79, 170, 103))
        self._draw_disk(canvas, self.sim.shooter, 5, self._color(self.sim.current_color))

        radius = max(3, int(round(self.config.ball_radius * size)))
        for color, point in zip(self.sim.balls, self.sim.ball_positions(), strict=True):
            self._draw_disk(canvas, point, radius + 1, (8, 10, 15))
            self._draw_disk(canvas, point, radius, self._color(color))
        return canvas

    @staticmethod
    def _color(index: int) -> tuple[int, int, int]:
        palette = (
            (239, 83, 80),
            (66, 165, 245),
            (255, 202, 40),
            (102, 187, 106),
            (171, 71, 188),
            (255, 112, 67),
            (38, 198, 218),
            (236, 64, 122),
        )
        return palette[index % len(palette)]

    @staticmethod
    def _draw_disk(
        canvas: RgbArray,
        point: Sequence[float],
        radius: int,
        color: tuple[int, int, int],
    ) -> None:
        height, width, _ = canvas.shape
        center_x = int(round(float(point[0]) * (width - 1)))
        center_y = int(round(float(point[1]) * (height - 1)))
        x0 = max(0, center_x - radius)
        x1 = min(width, center_x + radius + 1)
        y0 = max(0, center_y - radius)
        y1 = min(height, center_y + radius + 1)
        if x0 >= x1 or y0 >= y1:
            return
        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius**2
        view = canvas[y0:y1, x0:x1]
        view[mask] = color

