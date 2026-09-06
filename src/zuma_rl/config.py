"""Configuration for the simplified Zuma simulator."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ZumaConfig:
    """All deterministic level and reward parameters.

    Progress values use normalized arc length: ``0`` is the track entrance and
    ``1`` is the skull/end of the track.
    """

    num_colors: int = 5
    initial_balls: int = 24
    max_balls: int = 96
    match_size: int = 3

    aim_bins: int = 180
    ball_spacing: float = 0.018
    ball_radius: float = 0.025
    initial_front_progress: float = 0.48
    chain_speed_per_shot: float = 0.0022
    retreat_per_removed_ball: float = 0.012
    max_steps: int = 600

    shooter_x: float = 0.5
    shooter_y: float = 0.5
    chain_color_probability: float = 0.92

    reward_step: float = -0.002
    reward_miss: float = -0.04
    reward_insert_without_match: float = -0.01
    reward_per_removed_ball: float = 0.18
    reward_cascade_bonus: float = 0.30
    reward_win: float = 10.0
    reward_loss: float = -10.0

    render_size: int = 256
    track_turns: float = 1.75
    track_start_radius: float = 0.46
    track_end_radius: float = 0.17
    track_vertical_scale: float = 0.90
    track_samples: int = 1_024

    def __post_init__(self) -> None:
        if not 3 <= self.num_colors <= 8:
            raise ValueError("num_colors must be between 3 and 8")
        if self.initial_balls < 1:
            raise ValueError("initial_balls must be positive")
        if self.max_balls <= self.initial_balls:
            raise ValueError("max_balls must exceed initial_balls")
        if self.match_size < 3:
            raise ValueError("match_size must be at least 3")
        if self.aim_bins < 16:
            raise ValueError("aim_bins must be at least 16")
        if not 0.0 < self.ball_spacing < 0.2:
            raise ValueError("ball_spacing must be in (0, 0.2)")
        if not 0.0 < self.ball_radius < 0.1:
            raise ValueError("ball_radius must be in (0, 0.1)")
        if not 0.0 < self.initial_front_progress < 1.0:
            raise ValueError("initial_front_progress must be in (0, 1)")
        if self.chain_speed_per_shot <= 0.0:
            raise ValueError("chain_speed_per_shot must be positive")
        if self.retreat_per_removed_ball < 0.0:
            raise ValueError("retreat_per_removed_ball cannot be negative")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if not 0.0 <= self.chain_color_probability <= 1.0:
            raise ValueError("chain_color_probability must be in [0, 1]")
        if self.render_size < 64:
            raise ValueError("render_size must be at least 64")
        if self.track_samples < 64:
            raise ValueError("track_samples must be at least 64")

        tail = self.initial_front_progress - (
            self.initial_balls - 1
        ) * self.ball_spacing
        if tail < -0.05:
            raise ValueError(
                "initial chain extends too far before the track entrance; "
                "increase initial_front_progress or reduce initial_balls/spacing"
            )

    @property
    def action_count(self) -> int:
        """Number of combined ``aim × swap`` actions."""

        return self.aim_bins * 2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

