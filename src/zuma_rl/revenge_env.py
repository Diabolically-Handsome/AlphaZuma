"""Gymnasium adapter for the fidelity-first :mod:`zuma_rl.revenge_core`.

The simulator runs at the original 100 Hz.  One environment action selects an
aim bin and a verb, then advances a configurable number of native ticks.  The
default observation is an actor view: balls whose centre is inside a tunnel
are omitted and the remaining visible balls are packed into consecutive
slots.  ``privileged_debug`` is intended for diagnostics only and exposes
those hidden balls plus a few internal timers.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from zuma_rl.revenge_core import (
    GunState,
    PowerupType,
    Projectile,
    RevengePhysicsConfig,
    RevengeSimulator,
    SUPPORTED_PROFILE_MODE,
)

FloatObservation = NDArray[np.float32]
RgbArray = NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class RevengeEnvConfig:
    """Agent-interface, episode-limit, and reward configuration.

    Rewards deliberately retain the native game's score scale.  With the
    defaults, a 10-point ball contributes 10 reward, each decision costs
    ``0.01``, and the terminal result adds a symmetric bonus.
    """

    aim_bins: int = 180
    # Factorising verb and aim avoids representing wait/swap as 180 duplicate
    # categorical actions.  The legacy flat encoding remains available for
    # replay compatibility and controlled comparisons.
    action_mode: Literal["factorized", "flat"] = "factorized"
    # Fidelity default: accept a fresh input on every original 10 ms tick.
    # Larger values are an explicit control abstraction for diagnostics.
    frame_skip: int = 1
    max_ticks: int = 180_000
    max_balls: int = 192
    max_projectiles: int = 32
    # The original catalog currently contains at most two simultaneous curves.
    max_curves: int = 2
    # ``legacy-v1`` preserves the exact frozen V1/V1.1 actor interface.
    # ``full55-v1`` fixes six colour channels, exposes the active lily pad,
    # and adds the fourth hop verb required by dual-position levels.
    actor_interface: Literal["legacy-v1", "full55-v1"] = "legacy-v1"
    score_reward_scale: float = 1.0
    step_penalty: float = -0.01
    win_reward: float = 1_000.0
    loss_reward: float = -1_000.0
    privileged_debug: bool = False
    render_width: int = 400
    render_height: int = 300

    def __post_init__(self) -> None:
        if self.aim_bins < 2:
            raise ValueError("aim_bins must be at least 2")
        if self.action_mode not in {"factorized", "flat"}:
            raise ValueError(
                "action_mode must be 'factorized' or 'flat'"
            )
        if self.frame_skip < 1:
            raise ValueError("frame_skip must be at least 1")
        if self.max_ticks < 1:
            raise ValueError("max_ticks must be at least 1")
        if self.max_balls < 160:
            raise ValueError("max_balls must be at least 160")
        if self.max_projectiles < 1:
            raise ValueError("max_projectiles must be positive")
        if self.max_curves < 1:
            raise ValueError("max_curves must be positive")
        if self.actor_interface not in {"legacy-v1", "full55-v1"}:
            raise ValueError(
                "actor_interface must be 'legacy-v1' or 'full55-v1'"
            )
        if self.render_width < 64 or self.render_height < 48:
            raise ValueError("render dimensions are too small")
        reward_values = (
            self.score_reward_scale,
            self.step_penalty,
            self.win_reward,
            self.loss_reward,
        )
        if not all(math.isfinite(value) for value in reward_values):
            raise ValueError("reward configuration must be finite")
        if self.score_reward_scale < 0.0:
            raise ValueError("score_reward_scale cannot be negative")


class RevengeEnv(gym.Env[FloatObservation, Any]):
    """A fixed-vector Gymnasium environment around :class:`RevengeSimulator`.

    The default factorized action is ``[verb, aim_bin]``.  In legacy flat
    mode, actions are grouped by verb::

        action = verb * aim_bins + aim_bin

    ``verb=0`` only aims/waits, ``verb=1`` requests a shot, ``verb=2``
    requests a current/next-ball swap, and the ``full55-v1`` interface adds
    ``verb=3`` to hop to the other fixed lily pad.  Aim bins refer to their
    centre angle in ``[0, 2π)``.
    """

    metadata = {
        "render_modes": ["ansi", "rgb_array"],
        "render_fps": 100,
    }

    _VERB_NAMES = ("wait", "fire", "swap", "hop")
    _BALL_BASE_FEATURES = (
        "present",
        "x",
        "y",
        "waypoint",
        "contact_next_visible",
        "exploding",
        "in_tunnel",
    )
    _PROJECTILE_BASE_FEATURES = (
        "present",
        "x",
        "y",
        "velocity_x",
        "velocity_y",
        "waypoint",
        "merging",
        "merge_progress",
    )
    _GLOBAL_BASE_FEATURES = (
        "aim_sin",
        "aim_cos",
        "gun_state_percent",
        "current_missing",
        "next_missing",
        "bar_current",
        "bar_target",
        "score_progress",
        "score_tanh",
        "tick_progress",
        "visible_ball_fraction",
        "known_ball_fraction",
        "projectile_fraction",
        "zuma_reached",
        "stop_adding",
        "outcome",
        "advance_speed",
        "slow_timer",
        "backward_timer",
        "pending_fraction",
        "gun_normal",
        "gun_firing",
        "gun_reloading",
        "fruit_present",
        "fruit_collectable",
        "fruit_x",
        "fruit_y",
        "fruit_collecting",
    )
    _FULL55_SHOOTER_FEATURES = (
        "shooter_x",
        "shooter_y",
        "frog_position_0_x",
        "frog_position_0_y",
        "frog_position_1_x",
        "frog_position_1_y",
        "frog_position_0",
        "frog_position_1",
        "frog_dual_position",
    )

    def __init__(
        self,
        config: RevengeEnvConfig | None = None,
        *,
        simulator: RevengeSimulator | None = None,
        level_id: str = "Jungle1",
        root: str | Path | None = None,
        catalog: Any | None = None,
        hard: bool = False,
        curve_index: int = 0,
        physics_config: RevengePhysicsConfig | None = None,
        profile_mode: str = SUPPORTED_PROFILE_MODE,
        seed: int | None = None,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.config = config or RevengeEnvConfig()
        if render_mode not in {None, *self.metadata["render_modes"]}:
            raise ValueError(f"unsupported render_mode: {render_mode!r}")
        self.render_mode = render_mode

        if simulator is not None and physics_config is not None:
            raise ValueError(
                "physics_config cannot be supplied with an existing simulator"
            )
        if simulator is None:
            simulator = RevengeSimulator.from_installed(
                level_id,
                root=root,
                catalog=catalog,
                hard=hard,
                curve_index=curve_index,
                seed=seed,
                config=physics_config,
                profile_mode=profile_mode,
            )
        else:
            if simulator.profile_mode != profile_mode:
                raise ValueError(
                    "existing simulator profile_mode does not match the "
                    "environment request"
                )
            if seed is not None:
                simulator.reset(seed=seed)
        self.sim = simulator
        self.num_colors = self.sim.num_colors
        self.full55_interface = self.config.actor_interface == "full55-v1"
        self.color_feature_count = 6 if self.full55_interface else self.num_colors
        self.verb_count = 4 if self.full55_interface else 3
        if self.num_colors > self.color_feature_count:
            raise RuntimeError(
                "actor observation colour capacity exceeded: "
                f"{self.num_colors} active colours > "
                f"{self.color_feature_count} channels"
            )
        if self.sim.curve_count > self.config.max_curves:
            raise RuntimeError(
                "actor observation capacity exceeded: "
                f"{self.sim.curve_count} curves > "
                f"max_curves={self.config.max_curves}"
            )
        self.curve_feature_names = tuple(
            f"curve_{index}" for index in range(self.config.max_curves)
        )

        self.ball_feature_names = (
            self._BALL_BASE_FEATURES
            + tuple(
                f"color_{index}" for index in range(self.color_feature_count)
            )
            + tuple(
                f"powerup_{index}"
                for index in range(int(PowerupType.NONE))
            )
            + self.curve_feature_names
        )
        self.projectile_feature_names = (
            self._PROJECTILE_BASE_FEATURES
            + tuple(
                f"color_{index}" for index in range(self.color_feature_count)
            )
            + self.curve_feature_names
        )
        self.global_feature_names = (
            self._GLOBAL_BASE_FEATURES
            + (
                self._FULL55_SHOOTER_FEATURES
                if self.full55_interface
                else ()
            )
            + tuple(
                f"current_color_{index}"
                for index in range(self.color_feature_count)
            )
            + tuple(
                f"next_color_{index}"
                for index in range(self.color_feature_count)
            )
        )
        self.ball_feature_size = len(self.ball_feature_names)
        self.projectile_feature_size = len(self.projectile_feature_names)
        self.global_feature_size = len(self.global_feature_names)

        ball_end = self.config.max_balls * self.ball_feature_size
        projectile_end = (
            ball_end
            + self.config.max_projectiles * self.projectile_feature_size
        )
        observation_size = projectile_end + self.global_feature_size
        self.observation_layout: Mapping[str, slice] = {
            "balls": slice(0, ball_end),
            "projectiles": slice(ball_end, projectile_end),
            "globals": slice(projectile_end, observation_size),
        }

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(observation_size,),
            dtype=np.float32,
        )
        if self.config.action_mode == "factorized":
            self.action_space = spaces.MultiDiscrete(
                np.array((self.verb_count, self.config.aim_bins), dtype=np.int64)
            )
        else:
            self.action_space = spaces.Discrete(
                self.config.aim_bins * self.verb_count
            )
        self._episode_done = False
        self._last_info: dict[str, Any] = {}

    @property
    def observation_size(self) -> int:
        return int(self.observation_space.shape[0])

    @property
    def tick_hz(self) -> int:
        return int(self.sim.config.tick_hz)

    def decode_action(self, action: Any) -> tuple[int, int, float]:
        """Return ``(verb, aim_bin, centre_angle_radians)`` for an action."""

        if not self.action_space.contains(action):
            raise ValueError(f"invalid action: {action!r}")
        if self.config.action_mode == "factorized":
            values = np.asarray(action, dtype=np.int64)
            verb, aim_bin = (int(values[0]), int(values[1]))
        else:
            verb, aim_bin = divmod(int(action), self.config.aim_bins)
        angle = math.tau * (aim_bin + 0.5) / self.config.aim_bins
        return verb, aim_bin, angle

    def encode_action(
        self,
        verb: int,
        aim_bin: int,
    ) -> int | NDArray[np.int64]:
        """Encode a verb and aim bin, primarily for policies and replay tools."""

        if not 0 <= int(verb) < self.verb_count:
            allowed = "0 (wait), 1 (fire), or 2 (swap)"
            if self.full55_interface:
                allowed += ", or 3 (hop)"
            raise ValueError(f"verb must be {allowed}")
        if not 0 <= int(aim_bin) < self.config.aim_bins:
            raise ValueError("aim_bin is outside the configured range")
        if self.config.action_mode == "factorized":
            return np.asarray((verb, aim_bin), dtype=np.int64)
        return int(verb) * self.config.aim_bins + int(aim_bin)

    def valid_verb_mask(self) -> NDArray[np.bool_]:
        """Return actions that the retail state machine would accept now.

        Waiting/aiming is always available during a live episode.  Fire and
        swap are masked only when the simulator's corresponding request would
        return ``False``.  Every basis variable used here is already exposed
        in the actor observation: terminal state, gun state, and the current
        and next ball colors.  The mask therefore adds no privileged state.
        """

        live = (
            not self._episode_done
            and not self.sim.win_pending
            and not self.sim.loss_started
        )
        fire = (
            live
            and self.sim.gun_state is GunState.NORMAL
            and not self.sim.hop_in_progress
            and self.sim.current_color is not None
        )
        swap = (
            fire
            and self.sim.next_color is not None
            and self.sim.current_color != self.sim.next_color
        )
        verbs = (live, fire, swap)
        if self.full55_interface:
            verbs += (live and self.sim.can_hop_shooter,)
        return np.asarray(verbs, dtype=np.bool_)

    def action_masks(self) -> NDArray[np.bool_]:
        """Return the mask format consumed by ``MaskablePPO``.

        The factorized MultiDiscrete mask concatenates the three verb entries
        and the aim-bin entries.  Aim remains unrestricted because retail
        accepts cursor motion on every native tick, including during reload.
        The legacy flat action mode repeats each verb validity over its aim
        block.
        """

        verbs = self.valid_verb_mask()
        if self.config.action_mode == "factorized":
            return np.concatenate(
                (verbs, np.ones(self.config.aim_bins, dtype=np.bool_))
            )
        return np.repeat(verbs, self.config.aim_bins)

    def _in_tunnel(self, waypoint: float, curve_index: int = 0) -> bool:
        return self.sim.in_tunnel_at_waypoint(
            float(waypoint),
            curve_index=curve_index,
        )

    def _visible_balls(self) -> list[tuple[int, Any]]:
        curve_balls = list(self.sim.iter_curve_balls())
        if self.config.privileged_debug:
            return curve_balls
        return [
            (curve_index, ball)
            for curve_index, ball in curve_balls
            if not self._in_tunnel(ball.waypoint, curve_index)
        ]

    @staticmethod
    def _clip(value: float) -> np.float32:
        return np.float32(np.clip(value, -1.0, 1.0))

    def _normalised_xy(self, position: Sequence[float]) -> tuple[float, float]:
        width = float(self.sim.config.logical_width)
        height = float(self.sim.config.logical_height)
        return (
            float(np.clip(2.0 * float(position[0]) / width - 1.0, -1.0, 1.0)),
            float(np.clip(2.0 * float(position[1]) / height - 1.0, -1.0, 1.0)),
        )

    def _normalised_waypoint(
        self,
        waypoint: float,
        curve_index: int = 0,
    ) -> float:
        denominator = max(
            1.0,
            float(self.sim.curves[curve_index].end_waypoint),
        )
        return float(np.clip(2.0 * waypoint / denominator - 1.0, -1.0, 1.0))

    def _ball_observation(
        self,
        visible_balls: Sequence[tuple[int, Any]],
    ) -> FloatObservation:
        if len(visible_balls) > self.config.max_balls:
            raise RuntimeError(
                "actor observation capacity exceeded: "
                f"{len(visible_balls)} visible balls > "
                f"max_balls={self.config.max_balls}; increase max_balls "
                "instead of silently truncating game state"
            )
        features = np.zeros(
            (self.config.max_balls, self.ball_feature_size),
            dtype=np.float32,
        )
        active_indices = {
            (curve_index, ball.id): index
            for curve_index, state in enumerate(self.sim.curve_states)
            for index, ball in enumerate(state.balls)
        }
        for slot, (curve_index, ball) in enumerate(visible_balls):
            x, y = self._normalised_xy(
                self.sim.ball_position(ball, curve_index=curve_index)
            )
            features[slot, 0] = 1.0
            features[slot, 1] = x
            features[slot, 2] = y
            features[slot, 3] = self._normalised_waypoint(
                ball.waypoint,
                curve_index,
            )
            next_is_visible_neighbour = (
                slot + 1 < len(visible_balls)
                and visible_balls[slot + 1][0] == curve_index
                and active_indices[
                    (curve_index, visible_balls[slot + 1][1].id)
                ]
                == active_indices[(curve_index, ball.id)] + 1
            )
            features[slot, 4] = float(
                bool(ball.contact_next) and next_is_visible_neighbour
            )
            features[slot, 5] = float(bool(ball.exploding))
            features[slot, 6] = float(
                self._in_tunnel(ball.waypoint, curve_index)
            )
            color = int(ball.color)
            if 0 <= color < self.color_feature_count:
                features[slot, len(self._BALL_BASE_FEATURES) + color] = 1.0
            powerup_type = self.sim.effective_ball_powerup(ball)
            powerup_offset = (
                len(self._BALL_BASE_FEATURES) + self.color_feature_count
            )
            if 0 <= powerup_type < int(PowerupType.NONE):
                features[slot, powerup_offset + powerup_type] = 1.0
            curve_offset = powerup_offset + int(PowerupType.NONE)
            features[slot, curve_offset + curve_index] = 1.0
        return features.reshape(-1)

    def _projectile_observation(self) -> FloatObservation:
        features = np.zeros(
            (self.config.max_projectiles, self.projectile_feature_size),
            dtype=np.float32,
        )
        projectiles: list[tuple[Projectile, bool, int | None]] = [
            (projectile, False, None)
            for projectile in self.sim.free_projectiles
        ]
        projectiles.extend(
            (projectile, True, curve_index)
            for curve_index, projectile in self.sim.iter_merging_projectiles()
        )
        if len(projectiles) > self.config.max_projectiles:
            raise RuntimeError(
                "actor observation capacity exceeded: "
                f"{len(projectiles)} projectiles > "
                f"max_projectiles={self.config.max_projectiles}; increase "
                "max_projectiles instead of silently truncating game state"
            )
        speed_scale = max(
            1.0,
            float(self.sim.config.projectile_speed),
            float(self.sim.config.accuracy_projectile_speed),
        )
        for slot, (projectile, merging, curve_index) in enumerate(
            projectiles
        ):
            x, y = self._normalised_xy(projectile.position)
            velocity = np.asarray(projectile.velocity, dtype=np.float64).reshape(2)
            features[slot, 0] = 1.0
            features[slot, 1] = x
            features[slot, 2] = y
            features[slot, 3] = self._clip(float(velocity[0]) / speed_scale)
            features[slot, 4] = self._clip(float(velocity[1]) / speed_scale)
            features[slot, 5] = self._normalised_waypoint(
                projectile.waypoint,
                0 if curve_index is None else curve_index,
            )
            features[slot, 6] = float(merging)
            features[slot, 7] = float(
                np.clip(float(projectile.hit_percent), 0.0, 1.0)
            )
            color = int(projectile.color)
            if 0 <= color < self.color_feature_count:
                offset = len(self._PROJECTILE_BASE_FEATURES)
                features[slot, offset + color] = 1.0
            if curve_index is not None:
                curve_offset = (
                    len(self._PROJECTILE_BASE_FEATURES)
                    + self.color_feature_count
                )
                features[slot, curve_offset + curve_index] = 1.0
        return features.reshape(-1)

    def _global_observation(
        self,
        *,
        visible_count: int,
        projectile_count: int,
    ) -> FloatObservation:
        values = np.zeros(self.global_feature_size, dtype=np.float32)
        angle = float(self.sim.aim_angle)
        values[0] = math.sin(angle)
        values[1] = math.cos(angle)
        values[2] = np.clip(
            float(
                self.sim.hop_progress
                if self.sim.hop_in_progress
                else self.sim.gun_state_percent
            ),
            0.0,
            1.0,
        )
        values[3] = float(self.sim.current_color is None)
        values[4] = float(self.sim.next_color is None)
        bar_width = max(1.0, float(self.sim.config.zuma_bar_width))
        values[5] = np.clip(self.sim.current_bar_size / bar_width, 0.0, 1.0)
        values[6] = np.clip(self.sim.target_bar_size / bar_width, 0.0, 1.0)
        score_target = max(1.0, float(self.sim.score_target))
        values[7] = np.clip(self.sim.score / score_target, 0.0, 1.0)
        values[8] = math.tanh(float(self.sim.score) / 10_000.0)
        values[9] = np.clip(
            self.sim.tick_count / float(self.config.max_ticks),
            0.0,
            1.0,
        )
        values[10] = min(visible_count, self.config.max_balls) / float(
            self.config.max_balls
        )
        known_count = (
            self.sim.total_ball_count
            if self.config.privileged_debug
            else visible_count
        )
        values[11] = min(known_count, self.config.max_balls) / float(
            self.config.max_balls
        )
        values[12] = min(
            projectile_count,
            self.config.max_projectiles,
        ) / float(self.config.max_projectiles)
        values[13] = float(bool(self.sim.zuma_reached))
        values[14] = float(bool(self.sim.all_curves_stop_adding))
        if self.sim.outcome == "win" or self.sim.win_pending:
            values[15] = 1.0
        elif self.sim.outcome == "loss" or self.sim.loss_started:
            values[15] = -1.0

        if self.config.privileged_debug:
            curve_states = self.sim.curve_states
            speed_ratios = (
                float(state.advance_speed)
                / max(
                    1.0,
                    float(
                        getattr(
                            state.curve.parameters,
                            "max_speed",
                            1.0,
                        )
                    ),
                )
                for state in curve_states
            )
            values[16] = self._clip(max(speed_ratios, default=0.0))
            slow_ratios = (
                state.slow_count
                / max(
                    1.0,
                    float(
                        getattr(
                            state.curve.parameters,
                            "zuma_slow_duration",
                            1,
                        )
                    ),
                )
                for state in curve_states
            )
            values[17] = np.clip(max(slow_ratios, default=0.0), 0.0, 1.0)
            back_ratios = (
                state.backward_count
                / max(
                    1.0,
                    float(
                        getattr(
                            state.curve.parameters,
                            "zuma_back_distance",
                            1,
                        )
                    ),
                )
                for state in curve_states
            )
            values[18] = np.clip(
                max(back_ratios, default=0.0),
                0.0,
                1.0,
            )
            values[19] = min(
                sum(len(state.pending_colors) for state in curve_states),
                self.config.max_balls,
            ) / float(self.config.max_balls)

        # The frozen full55-v1 tensor cannot grow a fourth gun-state feature.
        # All-zero gun one-hot values therefore encode the hopping state, while
        # gun_state_percent above carries its progress.  Single-pad and legacy
        # observations remain byte-for-byte unchanged.
        if not self.sim.hop_in_progress:
            state_index = {
                GunState.NORMAL: 20,
                GunState.FIRING: 21,
                GunState.RELOADING: 22,
            }[self.sim.gun_state]
            values[state_index] = 1.0
        fruit_center = self.sim.fruit_center()
        if fruit_center is not None:
            values[23] = 1.0
            values[24] = float(not self.sim.fruit_collecting)
            fruit_x, fruit_y = self._normalised_xy(fruit_center)
            values[25] = fruit_x
            values[26] = fruit_y
            values[27] = float(bool(self.sim.fruit_collecting))
        color_start = len(self._GLOBAL_BASE_FEATURES)
        if self.full55_interface:
            shooter_x, shooter_y = self._normalised_xy(self.sim.shooter)
            values[color_start] = shooter_x
            values[color_start + 1] = shooter_y
            first_x, first_y = self._normalised_xy(
                self.sim.shooter_positions[0]
            )
            values[color_start + 2] = first_x
            values[color_start + 3] = first_y
            if self.sim.shooter_position_count > 1:
                second_x, second_y = self._normalised_xy(
                    self.sim.shooter_positions[1]
                )
                values[color_start + 4] = second_x
                values[color_start + 5] = second_y
            if self.sim.active_shooter_index < 2:
                values[color_start + 6 + self.sim.active_shooter_index] = 1.0
            values[color_start + 8] = float(
                self.sim.shooter_position_count == 2
            )
            color_start += len(self._FULL55_SHOOTER_FEATURES)
        if self.sim.current_color is not None:
            values[color_start + int(self.sim.current_color)] = 1.0
        next_start = color_start + self.color_feature_count
        if self.sim.next_color is not None:
            values[next_start + int(self.sim.next_color)] = 1.0
        return values

    def _observation(self) -> FloatObservation:
        visible_balls = self._visible_balls()
        projectile_count = (
            len(self.sim.free_projectiles)
            + self.sim.total_merging_projectile_count
        )
        observation = np.concatenate(
            (
                self._ball_observation(visible_balls),
                self._projectile_observation(),
                self._global_observation(
                    visible_count=len(visible_balls),
                    projectile_count=projectile_count,
                ),
            )
        ).astype(np.float32, copy=False)
        return observation

    def _info(
        self,
        *,
        score_delta: int = 0,
        ticks_advanced: int = 0,
        verb: int | None = None,
        aim_bin: int | None = None,
        accepted: bool | None = None,
    ) -> dict[str, Any]:
        visible_count = len(self._visible_balls())
        agent_outcome = (
            "win"
            if self.sim.win_pending
            else (
                "loss"
                if self.sim.loss_started
                else self.sim.outcome
            )
        )
        info: dict[str, Any] = {
            "fidelity_gate": "closed",
            "profile_mode": self.sim.profile_mode,
            "outcome": agent_outcome,
            "native_outcome": self.sim.outcome,
            "win_pending": bool(self.sim.win_pending),
            "skull_entry_pending": bool(self.sim.skull_entry_pending),
            "loss_started": bool(self.sim.loss_started),
            "loss_elapsed_ticks": int(self.sim.loss_elapsed_ticks),
            "score": int(self.sim.score),
            "score_delta": int(score_delta),
            "ticks": int(self.sim.tick_count),
            "ticks_advanced": int(ticks_advanced),
            "curve_count": self.sim.curve_count,
            "active_num_colors": int(self.num_colors),
            "color_feature_count": int(self.color_feature_count),
            "actor_interface": self.config.actor_interface,
            "frog_position_count": int(self.sim.shooter_position_count),
            "active_frog_position": int(self.sim.active_shooter_index),
            "hop_in_progress": bool(self.sim.hop_in_progress),
            "hop_ticks_remaining": int(self.sim.hop_ticks_remaining),
            "hop_target_frog_position": self.sim.hop_target_index,
            "chain_length": self.sim.total_ball_count,
            "chain_lengths": tuple(
                len(state.balls) for state in self.sim.curve_states
            ),
            "visible_balls": visible_count,
            "free_projectiles": len(self.sim.free_projectiles),
            "merging_projectiles": (
                self.sim.total_merging_projectile_count
            ),
            "fruit_present": self.sim.fruit_active_point_index is not None,
            "fruit_collecting": bool(
                self.sim.fruit_active_point_index is not None
                and self.sim.fruit_collecting
            ),
            "fruits_spawned": int(self.sim.fruit_spawn_count),
            "fruits_collected": int(self.sim.fruit_collect_count),
            "zuma_reached": bool(self.sim.zuma_reached),
        }
        if verb is not None:
            info["verb"] = self._VERB_NAMES[verb]
            info["verb_id"] = verb
        if aim_bin is not None:
            info["aim_bin"] = aim_bin
        if accepted is not None:
            info["action_accepted"] = accepted
        return info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[FloatObservation, dict[str, Any]]:
        super().reset(seed=seed)
        self.sim.reset(seed=seed)
        if options is not None and "state" in options:
            state = options["state"]
            if not isinstance(state, Mapping):
                raise TypeError("options['state'] must be a mapping")
            self.sim.load_state(**dict(state))
        self._episode_done = False
        info = self._info()
        self._last_info = info
        return self._observation(), info

    def step(
        self,
        action: Any,
    ) -> tuple[FloatObservation, float, bool, bool, dict[str, Any]]:
        if self._episode_done:
            raise RuntimeError("step() called after episode end; call reset()")
        verb, aim_bin, angle = self.decode_action(action)
        self.sim.set_aim(angle)
        accepted = True
        if verb == 1:
            accepted = self.sim.request_fire(angle)
        elif verb == 2:
            accepted = self.sim.swap_balls()
        elif verb == 3:
            accepted = self.sim.request_hop()

        score_before = int(self.sim.score)
        ticks_before = int(self.sim.tick_count)
        remaining = max(0, self.config.max_ticks - ticks_before)
        for _ in range(min(self.config.frame_skip, remaining)):
            self.sim.tick()
            # Retail has input-locked transition tails for both victory and
            # skull suction.  End the RL episode at the first irreversible
            # boundary instead of charging the policy forced no-op steps.
            if (
                self.sim.outcome is not None
                or self.sim.win_pending
                or self.sim.loss_started
            ):
                break
        ticks_advanced = int(self.sim.tick_count) - ticks_before
        score_delta = int(self.sim.score) - score_before

        effective_win = (
            self.sim.outcome == "win" or self.sim.win_pending
        )
        effective_loss = (
            self.sim.outcome == "loss" or self.sim.loss_started
        )
        terminated = effective_win or effective_loss
        truncated = (
            not terminated and self.sim.tick_count >= self.config.max_ticks
        )
        reward = (
            self.config.score_reward_scale * score_delta
            + self.config.step_penalty
        )
        if effective_win:
            reward += self.config.win_reward
        elif effective_loss:
            reward += self.config.loss_reward

        info = self._info(
            score_delta=score_delta,
            ticks_advanced=ticks_advanced,
            verb=verb,
            aim_bin=aim_bin,
            accepted=accepted,
        )
        if truncated:
            info["TimeLimit.truncated"] = True
        self._episode_done = terminated or truncated
        self._last_info = info
        return (
            self._observation(),
            float(reward),
            terminated,
            truncated,
            info,
        )

    def render(self) -> RgbArray | str | None:
        if self.render_mode is None:
            return None
        if self.render_mode == "ansi":
            return self._render_ansi()
        return self._render_rgb()

    def _render_ansi(self) -> str:
        current = "-" if self.sim.current_color is None else self.sim.current_color
        upcoming = "-" if self.sim.next_color is None else self.sim.next_color
        hidden = sum(
            self._in_tunnel(ball.waypoint, curve_index)
            for curve_index, ball in self.sim.iter_curve_balls()
        )
        fruit = "-"
        fruit_center = self.sim.fruit_center()
        if fruit_center is not None:
            fruit = (
                f"{float(fruit_center[0]):.1f},{float(fruit_center[1]):.1f}"
                + (":collecting" if self.sim.fruit_collecting else "")
            )
        return (
            f"tick={self.sim.tick_count:06d} score={self.sim.score:06d} "
            f"gun={self.sim.gun_state.value} current={current} next={upcoming} "
            f"curves={self.sim.curve_count} "
            f"balls={self.sim.total_ball_count} hidden={hidden} "
            f"shots={len(self.sim.free_projectiles)}/"
            f"{self.sim.total_merging_projectile_count} "
            f"fruit={fruit} "
            f"outcome={self.sim.outcome or '-'}"
        )

    def _logical_to_pixel(
        self,
        point: Sequence[float],
    ) -> tuple[int, int]:
        x = int(
            round(
                float(point[0])
                / float(self.sim.config.logical_width)
                * (self.config.render_width - 1)
            )
        )
        y = int(
            round(
                float(point[1])
                / float(self.sim.config.logical_height)
                * (self.config.render_height - 1)
            )
        )
        return x, y

    def _render_rgb(self) -> RgbArray:
        canvas = np.empty(
            (self.config.render_height, self.config.render_width, 3),
            dtype=np.uint8,
        )
        canvas[:] = (13, 17, 24)

        for curve in self.sim.curves:
            sample_count = min(max(curve.end_waypoint + 1, 2), 2_500)
            waypoints = np.linspace(
                0.0,
                float(curve.end_waypoint),
                sample_count,
            )
            points = np.asarray(
                curve.point_at_waypoint(waypoints),
                dtype=np.float64,
            ).reshape(-1, 2)
            tunnels = np.asarray(
                curve.is_in_tunnel_at_waypoint(waypoints),
                dtype=np.bool_,
            ).reshape(-1)
            for point, tunnel in zip(points, tunnels, strict=True):
                self._draw_disk(
                    canvas,
                    self._logical_to_pixel(point),
                    1,
                    (40, 44, 52) if tunnel else (77, 84, 96),
                )

            endpoint = curve.point_at_waypoint(float(curve.end_waypoint))
            self._draw_disk(
                canvas,
                self._logical_to_pixel(endpoint),
                7,
                (150, 45, 45),
            )

        shooter_pixel = self._logical_to_pixel(self.sim.shooter)
        self._draw_disk(canvas, shooter_pixel, 8, (68, 154, 91))
        aim_tip = self.sim.shooter + np.array(
            (math.cos(self.sim.aim_angle), math.sin(self.sim.aim_angle))
        ) * 34.0
        for fraction in np.linspace(0.0, 1.0, 12):
            point = self.sim.shooter + (aim_tip - self.sim.shooter) * fraction
            self._draw_disk(
                canvas,
                self._logical_to_pixel(point),
                1,
                (188, 220, 197),
            )

        radius = max(
            2,
            int(
                round(
                    self.sim.config.ball_radius
                    / self.sim.config.logical_width
                    * self.config.render_width
                )
            ),
        )
        for curve_index, ball in self.sim.iter_curve_balls():
            position = self.sim.ball_position(
                ball,
                curve_index=curve_index,
            )
            color = self._palette(ball.color)
            if self._in_tunnel(ball.waypoint, curve_index):
                color = tuple(channel // 3 for channel in color)
            if ball.exploding:
                color = (245, 245, 245)
            pixel = self._logical_to_pixel(position)
            self._draw_disk(canvas, pixel, radius + 1, (4, 6, 9))
            self._draw_disk(canvas, pixel, radius, color)

        for projectile in (
            *self.sim.free_projectiles,
            *(
                projectile
                for _, projectile in self.sim.iter_merging_projectiles()
            ),
        ):
            self._draw_disk(
                canvas,
                self._logical_to_pixel(projectile.position),
                max(2, radius - 1),
                self._palette(projectile.color),
            )
        fruit_center = self.sim.fruit_center()
        if fruit_center is not None:
            calibration = self.sim.fruit_calibration
            assert calibration is not None
            fruit_radius = max(
                2,
                int(
                    round(
                        (calibration.logical_height / 2.0)
                        / self.sim.config.logical_width
                        * self.config.render_width
                    )
                ),
            )
            fruit_color = (
                (250, 245, 210)
                if self.sim.fruit_collecting
                else (255, 193, 7)
            )
            self._draw_disk(
                canvas,
                self._logical_to_pixel(fruit_center),
                fruit_radius,
                fruit_color,
            )
        return canvas

    @staticmethod
    def _palette(index: int) -> tuple[int, int, int]:
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
        return palette[int(index) % len(palette)]

    @staticmethod
    def _draw_disk(
        canvas: RgbArray,
        centre: tuple[int, int],
        radius: int,
        color: tuple[int, int, int],
    ) -> None:
        height, width, _ = canvas.shape
        centre_x, centre_y = centre
        x0 = max(0, centre_x - radius)
        x1 = min(width, centre_x + radius + 1)
        y0 = max(0, centre_y - radius)
        y1 = min(height, centre_y + radius + 1)
        if x0 >= x1 or y0 >= y1:
            return
        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask = (xx - centre_x) ** 2 + (yy - centre_y) ** 2 <= radius**2
        canvas[y0:y1, x0:x1][mask] = color


__all__ = ["RevengeEnv", "RevengeEnvConfig"]
