"""Actor-visible state for the deterministic elite-human input actuator.

``HumanSpeedrunWrapper`` deliberately leaves the base observation unchanged.
That made early feed-forward policies partially observable: the next executed
aim and button edge also depend on private delay queues, aim velocity, and the
previous desired verb.  This module provides an opt-in observation wrapper
that keeps the exact same actuator dynamics while appending a bounded,
fixed-width description of all state needed to predict those dynamics.

The existing wrapper is intentionally not modified because several frozen
AlphaZuma campaigns bind its bytes.  New campaigns can explicitly select this
``motor-observable-v1`` interface instead.
"""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from zuma_rl.human_speedrun import (
    EliteHumanInputConfig,
    HumanSpeedrunWrapper,
    WinFirstRewardConfig,
)
from zuma_rl.revenge_env import FloatObservation


MOTOR_OBSERVATION_PROFILE = "motor-observable-v1"
_BUTTON_QUEUE_SLOTS = 3


class ObservableHumanSpeedrunWrapper(HumanSpeedrunWrapper):
    """Expose the wrapper's own deterministic motor state to the actor.

    The base Revenge observation remains an exact prefix.  Motor features are
    appended to the global slice so the existing entity/polar extractor can
    consume them without changing ball or projectile semantics.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        input_config: EliteHumanInputConfig | None = None,
        reward_config: WinFirstRewardConfig | None = None,
    ) -> None:
        super().__init__(
            env,
            input_config=input_config,
            reward_config=reward_config,
        )
        self.aim_queue_slots = int(self.input_config.reaction_delay_ticks)
        self.motor_feature_names = self._build_motor_feature_names()
        self.base_observation_size = int(self.env.observation_space.shape[0])
        size = self.base_observation_size + len(self.motor_feature_names)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(size,),
            dtype=np.float32,
        )

        base = self.revenge_env
        self.ball_feature_names = tuple(base.ball_feature_names)
        self.projectile_feature_names = tuple(base.projectile_feature_names)
        self.global_feature_names = (
            *tuple(base.global_feature_names),
            *self.motor_feature_names,
        )
        self.ball_feature_size = int(base.ball_feature_size)
        self.projectile_feature_size = int(base.projectile_feature_size)
        self.global_feature_size = len(self.global_feature_names)
        self.color_feature_count = int(base.color_feature_count)
        self.curve_feature_names = tuple(base.curve_feature_names)
        base_layout = base.observation_layout
        global_start = int(base_layout["globals"].start or 0)
        self.observation_layout = {
            "balls": base_layout["balls"],
            "projectiles": base_layout["projectiles"],
            "globals": slice(global_start, size),
        }
        self.observation_size = size

    @property
    def config(self) -> Any:
        return self.revenge_env.config

    @property
    def sim(self) -> Any:
        return self.revenge_env.sim

    def _build_motor_feature_names(self) -> tuple[str, ...]:
        names: list[str] = [
            "motor_aim_velocity",
            "motor_delayed_target_sin",
            "motor_delayed_target_cos",
            "motor_button_cooldown_remaining",
        ]
        for slot in range(self.aim_queue_slots):
            names.extend(
                (
                    f"motor_aim_queue_{slot}_present",
                    f"motor_aim_queue_{slot}_due_fraction",
                    f"motor_aim_queue_{slot}_sin",
                    f"motor_aim_queue_{slot}_cos",
                )
            )
        for slot in range(_BUTTON_QUEUE_SLOTS):
            names.extend(
                (
                    f"motor_button_queue_{slot}_present",
                    f"motor_button_queue_{slot}_due_fraction",
                    f"motor_button_queue_{slot}_fire",
                    f"motor_button_queue_{slot}_swap",
                    f"motor_button_queue_{slot}_hop",
                )
            )
        names.extend(
            f"motor_last_desired_{name}"
            for name in ("wait", "fire", "swap", "hop")
        )
        return tuple(names)

    @staticmethod
    def _unit_interval(value: float) -> float:
        return float(np.clip(value, 0.0, 1.0))

    def _due_fraction(self, due_tick: int) -> float:
        denominator = max(1, int(self.input_config.reaction_delay_ticks))
        return self._unit_interval(
            (int(due_tick) - int(self._control_tick)) / denominator
        )

    def motor_observation(self) -> np.ndarray:
        """Return the complete normalized actuator state at decision time."""

        max_speed = math.radians(
            self.input_config.max_aim_speed_degrees_per_second
        )
        values: list[float] = [
            float(np.clip(self._aim_velocity / max_speed, -1.0, 1.0)),
            math.sin(self._delayed_target_angle),
            math.cos(self._delayed_target_angle),
        ]
        if self._last_button_tick is None:
            cooldown = 0.0
        else:
            elapsed = int(self._control_tick) - int(self._last_button_tick)
            remaining = max(
                0,
                int(self.input_config.min_button_interval_ticks) - elapsed,
            )
            cooldown = remaining / max(
                1, int(self.input_config.min_button_interval_ticks)
            )
        values.append(self._unit_interval(cooldown))

        aim_rows = list(self._aim_queue)
        if len(aim_rows) > self.aim_queue_slots:
            raise RuntimeError(
                "motor-observable aim queue exceeded its reaction-delay bound"
            )
        for slot in range(self.aim_queue_slots):
            if slot < len(aim_rows):
                due_tick, angle = aim_rows[slot]
                values.extend(
                    (
                        1.0,
                        self._due_fraction(due_tick),
                        math.sin(angle),
                        math.cos(angle),
                    )
                )
            else:
                values.extend((0.0, 0.0, 0.0, 0.0))

        button_rows = list(self._button_queue)
        if len(button_rows) > _BUTTON_QUEUE_SLOTS:
            raise RuntimeError(
                "motor-observable button queue exceeded the verb bound"
            )
        for slot in range(_BUTTON_QUEUE_SLOTS):
            if slot < len(button_rows):
                command = button_rows[slot]
                one_hot = [0.0, 0.0, 0.0]
                if not 1 <= int(command.verb) <= 3:
                    raise RuntimeError("motor-observable queued verb is invalid")
                one_hot[int(command.verb) - 1] = 1.0
                values.extend(
                    (
                        1.0,
                        self._due_fraction(command.due_tick),
                        *one_hot,
                    )
                )
            else:
                values.extend((0.0, 0.0, 0.0, 0.0, 0.0))

        last_desired = [0.0, 0.0, 0.0, 0.0]
        if not 0 <= int(self._last_desired_verb) < len(last_desired):
            raise RuntimeError("motor-observable last desired verb is invalid")
        last_desired[int(self._last_desired_verb)] = 1.0
        values.extend(last_desired)
        result = np.asarray(values, dtype=np.float32)
        if result.shape != (len(self.motor_feature_names),):
            raise RuntimeError("motor-observable feature width changed")
        if not bool(np.isfinite(result).all()):
            raise RuntimeError("motor-observable features are not finite")
        if bool(np.any(result < -1.0) or np.any(result > 1.0)):
            raise RuntimeError("motor-observable features escaped [-1, 1]")
        return result

    def _augment(self, observation: FloatObservation) -> FloatObservation:
        base = np.asarray(observation, dtype=np.float32).reshape(-1)
        if base.shape != (self.base_observation_size,):
            raise RuntimeError("motor-observable base observation width changed")
        return np.concatenate((base, self.motor_observation())).astype(
            np.float32,
            copy=False,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[FloatObservation, dict[str, Any]]:
        observation, info = super().reset(seed=seed, options=options)
        info = dict(info)
        info["motor_observation_profile"] = MOTOR_OBSERVATION_PROFILE
        return self._augment(observation), info

    def step(
        self,
        action: Any,
    ) -> tuple[FloatObservation, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = super().step(action)
        info = dict(info)
        info["motor_observation_profile"] = MOTOR_OBSERVATION_PROFILE
        return (
            self._augment(observation),
            reward,
            terminated,
            truncated,
            info,
        )

    def contract(self) -> dict[str, Any]:
        value = super().contract()
        value.update(
            {
                "motor_observation_profile": MOTOR_OBSERVATION_PROFILE,
                "base_observation_size": self.base_observation_size,
                "motor_feature_count": len(self.motor_feature_names),
                "observation_space_unchanged": False,
                "base_observation_is_exact_prefix": True,
                "actuator_dynamics_unchanged": True,
                "motor_state_is_actor_visible": True,
            }
        )
        return value
