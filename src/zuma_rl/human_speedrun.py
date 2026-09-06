"""Human-limited control and win-first rewards for speedrun training.

The retail-faithful :class:`~zuma_rl.revenge_env.RevengeEnv` deliberately
accepts an exact aim and a button request on every native 10 ms tick.  That is
the right interface for simulator fidelity, but it gives a policy superhuman
motor control.  This module keeps the simulator untouched and wraps that
interface with an explicit, deterministic elite-human actuator contract.

The wrapper also replaces the score-dominant training reward with a bounded
speedrun reward.  A completed win is guaranteed to outrank every failed or
timed-out episode; among wins, elapsed native ticks dominate the bounded score
shaping at one-second resolution.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Deque

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from zuma_rl.revenge_env import FloatObservation, RevengeEnv


@dataclass(frozen=True, slots=True)
class EliteHumanInputConfig:
    """Deterministic upper-envelope human input limits at the native tick rate.

    These values are intentionally exposed as a versioned contract.  Version
    1 is a conservative elite-human starting envelope, pending direct raw-mouse
    telemetry from top PC runs.
    """

    profile_id: str = "elite-human-v1"
    reaction_delay_ticks: int = 12
    max_aim_speed_degrees_per_second: float = 1_080.0
    max_aim_acceleration_degrees_per_second_squared: float = 18_000.0
    min_button_interval_ticks: int = 5

    def __post_init__(self) -> None:
        if not self.profile_id:
            raise ValueError("profile_id cannot be empty")
        if self.reaction_delay_ticks < 0:
            raise ValueError("reaction_delay_ticks cannot be negative")
        if self.min_button_interval_ticks < 1:
            raise ValueError("min_button_interval_ticks must be positive")
        values = (
            self.max_aim_speed_degrees_per_second,
            self.max_aim_acceleration_degrees_per_second_squared,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("aim speed and acceleration limits must be positive")


@dataclass(frozen=True, slots=True)
class WinFirstRewardConfig:
    """Bounded reward whose episode ordering is win, then time, then score."""

    profile_id: str = "win-time-score-v1"
    win_reward: float = 10.0
    failure_reward: float = -10.0
    time_penalty_per_native_tick: float = -0.0001
    # This is the maximum total score-shaping reward in one episode.  Progress
    # is clipped at the board's native Zuma score target.
    score_progress_reward_cap: float = 0.01

    def __post_init__(self) -> None:
        if not self.profile_id:
            raise ValueError("profile_id cannot be empty")
        values = (
            self.win_reward,
            self.failure_reward,
            self.time_penalty_per_native_tick,
            self.score_progress_reward_cap,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("reward values must be finite")
        if self.win_reward <= 0.0:
            raise ValueError("win_reward must be positive")
        if self.failure_reward >= 0.0:
            raise ValueError("failure_reward must be negative")
        if self.time_penalty_per_native_tick >= 0.0:
            raise ValueError("time_penalty_per_native_tick must be negative")
        if self.score_progress_reward_cap < 0.0:
            raise ValueError("score_progress_reward_cap cannot be negative")


@dataclass(frozen=True, slots=True)
class _PendingButton:
    due_tick: int
    sequence: int
    verb: int


def _wrapped_delta(target: float, current: float) -> float:
    """Return the shortest signed angular delta in ``[-pi, pi)``."""

    return (target - current + math.pi) % math.tau - math.pi


class HumanSpeedrunWrapper(gym.Wrapper):
    """Apply elite-human input limits and a bounded win-first reward.

    The observation and action spaces are unchanged, so an existing state
    policy can be loaded and fine-tuned.  Actions become *intentions*: aim and
    button inputs take ``reaction_delay_ticks`` to reach the game; aim then
    follows velocity/acceleration limits, and button edges obey a minimum
    interval.  The underlying simulator still advances exactly one retail tick
    per wrapper step.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        input_config: EliteHumanInputConfig | None = None,
        reward_config: WinFirstRewardConfig | None = None,
    ) -> None:
        base_env = env.unwrapped
        if not isinstance(base_env, RevengeEnv):
            raise TypeError(
                "HumanSpeedrunWrapper requires a RevengeEnv, optionally "
                "inside standard Gymnasium wrappers"
            )
        if base_env.config.frame_skip != 1:
            raise ValueError(
                "HumanSpeedrunWrapper requires frame_skip=1 so limits apply "
                "on every native tick"
            )
        super().__init__(env)
        self.input_config = input_config or EliteHumanInputConfig()
        self.reward_config = reward_config or WinFirstRewardConfig()
        self._validate_reward_ordering()

        self._control_tick = 0
        self._aim_angle = 0.0
        self._aim_velocity = 0.0
        self._delayed_target_angle = 0.0
        self._aim_queue: Deque[tuple[int, float]] = deque()
        self._button_queue: Deque[_PendingButton] = deque()
        self._button_sequence = 0
        self._last_desired_verb = 0
        self._last_button_tick: int | None = None
        self._score_progress = 0.0

    @property
    def revenge_env(self) -> RevengeEnv:
        base_env = self.env.unwrapped
        if not isinstance(base_env, RevengeEnv):  # pragma: no cover - guarded
            raise RuntimeError("wrapped RevengeEnv is no longer available")
        return base_env

    def contract(self) -> dict[str, Any]:
        """Return a serializable description suitable for run receipts."""

        return {
            "schema": "zuma-rl.human-speedrun-contract",
            "version": 1,
            "native_tick_hz": self.revenge_env.tick_hz,
            "input": asdict(self.input_config),
            "reward": asdict(self.reward_config),
            "base_environment_unchanged": True,
            "observation_space_unchanged": True,
            "action_space_unchanged": True,
        }

    def _validate_reward_ordering(self) -> None:
        rewards = self.reward_config
        max_ticks = self.revenge_env.config.max_ticks
        minimum_win = (
            rewards.win_reward
            + rewards.time_penalty_per_native_tick * max_ticks
        )
        maximum_failure = (
            rewards.failure_reward + rewards.score_progress_reward_cap
        )
        if minimum_win <= maximum_failure:
            raise ValueError(
                "reward contract does not guarantee that every win outranks "
                "every failure"
            )

        one_second_time_value = (
            -rewards.time_penalty_per_native_tick * self.revenge_env.tick_hz
        )
        if one_second_time_value < rewards.score_progress_reward_cap:
            raise ValueError(
                "score shaping can outweigh more than one second of elapsed "
                "time"
            )

    def _progress(self, score: int) -> float:
        target = max(1, int(self.revenge_env.sim.score_target))
        return min(1.0, max(0.0, float(score) / float(target)))

    def _reset_control_state(self) -> None:
        self._control_tick = 0
        self._aim_angle = float(self.revenge_env.sim.aim_angle) % math.tau
        self._aim_velocity = 0.0
        self._delayed_target_angle = self._aim_angle
        self._aim_queue.clear()
        self._button_queue.clear()
        self._button_sequence = 0
        self._last_desired_verb = 0
        self._last_button_tick = None
        self._score_progress = self._progress(self.revenge_env.sim.score)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[FloatObservation, dict[str, Any]]:
        observation, info = self.env.reset(seed=seed, options=options)
        self._reset_control_state()
        info = dict(info)
        info["human_speedrun"] = {
            "input_profile": self.input_config.profile_id,
            "reward_profile": self.reward_config.profile_id,
            "pending_buttons": 0,
        }
        return observation, info

    def _queue_intention(self, verb: int, angle: float) -> bool:
        due_tick = self._control_tick + self.input_config.reaction_delay_ticks
        self._aim_queue.append((due_tick, angle))

        enqueued = False
        rising_edge = verb != 0 and verb != self._last_desired_verb
        pending_verbs = {command.verb for command in self._button_queue}
        if rising_edge and verb not in pending_verbs:
            self._button_queue.append(
                _PendingButton(
                    due_tick=due_tick,
                    sequence=self._button_sequence,
                    verb=verb,
                )
            )
            self._button_sequence += 1
            enqueued = True
        self._last_desired_verb = verb
        return enqueued

    def _release_due_aim(self) -> None:
        while self._aim_queue and self._aim_queue[0][0] <= self._control_tick:
            _, self._delayed_target_angle = self._aim_queue.popleft()

    def _advance_aim(self) -> None:
        self._release_due_aim()
        config = self.input_config
        tick_hz = float(self.revenge_env.tick_hz)
        max_speed = math.radians(
            config.max_aim_speed_degrees_per_second
        )
        max_acceleration = math.radians(
            config.max_aim_acceleration_degrees_per_second_squared
        )
        delta = _wrapped_delta(self._delayed_target_angle, self._aim_angle)
        if abs(delta) < 1e-12:
            desired_velocity = 0.0
        else:
            braking_speed = math.sqrt(2.0 * max_acceleration * abs(delta))
            desired_velocity = math.copysign(
                min(max_speed, braking_speed),
                delta,
            )
        max_velocity_change = max_acceleration / tick_hz
        velocity_change = float(
            np.clip(
                desired_velocity - self._aim_velocity,
                -max_velocity_change,
                max_velocity_change,
            )
        )
        self._aim_velocity += velocity_change
        step = self._aim_velocity / tick_hz
        if step == 0.0:
            return
        if math.copysign(1.0, step) != math.copysign(1.0, delta) or abs(step) >= abs(
            delta
        ):
            self._aim_angle = self._delayed_target_angle % math.tau
            self._aim_velocity = 0.0
        else:
            self._aim_angle = (self._aim_angle + step) % math.tau

    def _aim_bin(self) -> int:
        bins = self.revenge_env.config.aim_bins
        return min(
            bins - 1,
            int((self._aim_angle % math.tau) * bins / math.tau),
        )

    def _due_button(self) -> int:
        if not self._button_queue:
            return 0
        command = self._button_queue[0]
        if command.due_tick > self._control_tick:
            return 0
        if (
            self._last_button_tick is not None
            and self._control_tick - self._last_button_tick
            < self.input_config.min_button_interval_ticks
        ):
            return 0
        self._button_queue.popleft()
        self._last_button_tick = self._control_tick
        return command.verb

    def valid_verb_mask(self) -> NDArray[np.bool_]:
        """Mask duplicate queued button edges in addition to retail rejects."""

        mask = self.revenge_env.valid_verb_mask().copy()
        for command in self._button_queue:
            mask[command.verb] = False
        return mask

    def action_masks(self) -> NDArray[np.bool_]:
        verbs = self.valid_verb_mask()
        bins = self.revenge_env.config.aim_bins
        if self.revenge_env.config.action_mode == "factorized":
            return np.concatenate((verbs, np.ones(bins, dtype=np.bool_)))
        return np.repeat(verbs, bins)

    def _speedrun_reward(
        self,
        *,
        info: dict[str, Any],
        terminated: bool,
        truncated: bool,
    ) -> tuple[float, dict[str, float]]:
        rewards = self.reward_config
        progress = self._progress(int(info["score"]))
        progress_delta = progress - self._score_progress
        self._score_progress = progress
        score_reward = rewards.score_progress_reward_cap * progress_delta
        time_reward = (
            rewards.time_penalty_per_native_tick * int(info["ticks_advanced"])
        )
        outcome_reward = 0.0
        if terminated or truncated:
            outcome_reward = (
                rewards.win_reward
                if info.get("outcome") == "win"
                else rewards.failure_reward
            )
        terms = {
            "score_progress": float(score_reward),
            "elapsed_time": float(time_reward),
            "outcome": float(outcome_reward),
        }
        return float(sum(terms.values())), terms

    def step(
        self,
        action: Any,
    ) -> tuple[FloatObservation, float, bool, bool, dict[str, Any]]:
        desired_verb, desired_aim_bin, desired_angle = self.revenge_env.decode_action(
            action
        )
        enqueued = self._queue_intention(desired_verb, desired_angle)
        self._advance_aim()
        executed_verb = self._due_button()
        executed_aim_bin = self._aim_bin()
        executed_action = self.revenge_env.encode_action(
            executed_verb,
            executed_aim_bin,
        )
        observation, native_reward, terminated, truncated, info = self.env.step(
            executed_action
        )
        info = dict(info)
        reward, reward_terms = self._speedrun_reward(
            info=info,
            terminated=terminated,
            truncated=truncated,
        )
        info["native_environment_reward"] = float(native_reward)
        info["reward_terms"] = reward_terms
        info["human_speedrun"] = {
            "input_profile": self.input_config.profile_id,
            "reward_profile": self.reward_config.profile_id,
            "control_tick": self._control_tick,
            "reaction_delay_ticks": self.input_config.reaction_delay_ticks,
            "desired_verb": desired_verb,
            "desired_aim_bin": desired_aim_bin,
            "button_enqueued": enqueued,
            "executed_verb": executed_verb,
            "executed_aim_bin": executed_aim_bin,
            "pending_buttons": len(self._button_queue),
            "aim_angle_degrees": math.degrees(self._aim_angle),
            "aim_velocity_degrees_per_second": math.degrees(
                self._aim_velocity
            ),
        }
        self._control_tick += int(info["ticks_advanced"])
        return observation, reward, terminated, truncated, info
