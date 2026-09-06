"""Park-and-settle macro actions over the elite-human actuator stack.

Lineage.  :class:`~zuma_rl.human_speedrun.HumanSpeedrunWrapper` (and its
``motor-observable-v1`` subclass in :mod:`zuma_rl.observable_human_speedrun`)
turns per-tick actions into *intentions*: aim and button edges arrive
``reaction_delay_ticks`` late and the cursor is slew/acceleration limited.
The 55/55 settled strategic teacher
(:mod:`zuma_rl.revenge_settled_strategic_teacher_v3`) wins only because it
parks its aim during wait ticks and withholds the fire edge until the
observed cursor is within ``aim_tolerance_bins`` of the target; without that
settle gate it wins 18/55.  A per-tick PPO student must rediscover exactly
that park-then-settle motor program across 12,000-20,000 tick episodes in
which its own non-wait intent is ~2-3% of ticks.

This module promotes the teacher's motor program into the action interface
itself.  A macro action is ``(macro_verb, target_aim_bin)`` with
``macro_verb`` in ``{wait_hold, fire, swap}``.  The wrapper expands each
macro into the underlying per-tick intent stream: it parks the delayed,
slew-limited cursor on the target bin, gates the fire edge on the executed
cursor settling within tolerance, and then keeps emitting the same target
through the reaction delay plus the firing animation so the release angle
(which retail samples live at release, six ticks after the executed fire
request) cannot be corrupted by a later intent.  Decision points occur only
when the gun is ``NORMAL`` and no button intention is pending, which
collapses the decision horizon by roughly the mean macro length (~20-70
native ticks) and removes the actuator gap from the policy's burden.

Discounting semantics: the macro reward is the *undiscounted sum* of the
wrapped environment's per-tick rewards over every native tick the macro
consumed.  A discount factor configured in the learner therefore applies
per macro step, not per native tick; ``gamma = 0.999`` over ~40-tick macros
corresponds to a per-tick discount of roughly ``0.999 ** (1 / 40)``.

Interface notes: the macro action space is ``MultiDiscrete([3, aim_bins])``
regardless of the wrapped environment's verb count, so policies trained
through this wrapper are *not* action-compatible with per-tick policies.
The ``hop`` verb of the ``full55-v1`` interface is deliberately not exposed;
dual-position levels need a future macro revision.  Existing wrappers are
not modified because frozen campaigns bind their bytes; this module only
subclasses/wraps them.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from zuma_rl.human_speedrun import HumanSpeedrunWrapper
from zuma_rl.revenge_core import GunState
from zuma_rl.revenge_env import FloatObservation

PARK_SETTLE_PROFILE = "park-settle-v1"

MACRO_VERB_NAMES = ("wait_hold", "fire", "swap")
MACRO_WAIT_HOLD = 0
MACRO_FIRE = 1
MACRO_SWAP = 2


@dataclass(frozen=True, slots=True)
class ParkSettleActionConfig:
    """Constants of the park-settle macro expansion.

    ``settle_tolerance_bins`` mirrors the settled teacher's
    ``aim_tolerance_bins=2``.  ``settle_timeout_ticks`` bounds the park
    phase; the worst-case park (90 bins away) needs roughly
    ``reaction_delay(12) + 90 bins / 5.4 bins-per-tick`` which is about 29
    ticks, so 36 leaves slack for the acceleration ramp.  After the fire
    edge the target is held for ``reaction_delay + fire_animation_ticks +
    post_release_margin_ticks`` native ticks; the retail release happens at
    ``reaction_delay + ~5-6`` ticks after the edge, so the hold always
    covers it.  The reaction delay itself is intentionally *not* a field:
    it is read live from the wrapped actuator's
    :class:`~zuma_rl.human_speedrun.EliteHumanInputConfig` so the two can
    never disagree.
    """

    profile_id: str = PARK_SETTLE_PROFILE
    settle_tolerance_bins: int = 2
    settle_timeout_ticks: int = 36
    fire_animation_ticks: int = 6
    post_release_margin_ticks: int = 4
    hold_ticks: int = 8
    decision_roll_limit_ticks: int = 64

    def __post_init__(self) -> None:
        if not self.profile_id:
            raise ValueError("profile_id cannot be empty")
        if self.settle_tolerance_bins < 0:
            raise ValueError("settle_tolerance_bins cannot be negative")
        for name in (
            "settle_timeout_ticks",
            "fire_animation_ticks",
            "post_release_margin_ticks",
            "hold_ticks",
            "decision_roll_limit_ticks",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")


def resolve_human_speedrun_wrapper(env: gym.Env) -> HumanSpeedrunWrapper:
    """Return the actuator wrapper inside an arbitrarily nested stack."""

    current: Any = env
    while current is not None:
        if isinstance(current, HumanSpeedrunWrapper):
            return current
        current = getattr(current, "env", None)
    raise TypeError(
        "ParkSettleActionWrapper requires a HumanSpeedrunWrapper (or "
        "subclass) somewhere in the wrapped stack"
    )


class ParkSettleActionWrapper(gym.Wrapper):
    """Expose park-settle macro actions over a human-speedrun stack.

    One ``step`` consumes many native ticks.  Every internal tick emits
    ``(wait, target_aim_bin)`` so the delayed slew-limited cursor drives
    toward the target, except for the single rising-edge tick of a ``fire``
    or ``swap`` macro.  After the macro-specific phase the wrapper keeps
    emitting waits until the next decision point (gun ``NORMAL`` and no
    pending button intention) so that every observation handed to the
    policy is an actionable state.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        config: ParkSettleActionConfig | None = None,
    ) -> None:
        super().__init__(env)
        self.config = config or ParkSettleActionConfig()
        self._human = resolve_human_speedrun_wrapper(env)
        base = self._human.revenge_env
        self.aim_bins = int(base.config.aim_bins)
        if self.config.settle_tolerance_bins >= self.aim_bins // 2:
            raise ValueError(
                "settle_tolerance_bins is outside the useful range for "
                f"{self.aim_bins} aim bins"
            )
        self.reaction_delay_ticks = int(
            self._human.input_config.reaction_delay_ticks
        )
        minimum_roll = self.reaction_delay_ticks + 30
        if self.config.decision_roll_limit_ticks < minimum_roll:
            raise ValueError(
                "decision_roll_limit_ticks must cover one full delayed fire "
                f"cycle (>= {minimum_roll} ticks)"
            )
        self.action_space = spaces.MultiDiscrete(
            np.array((len(MACRO_VERB_NAMES), self.aim_bins), dtype=np.int64)
        )
        self._last_executed_aim_bin = 0
        self._last_pending_buttons = 0

    @property
    def sim(self) -> Any:
        return self._human.revenge_env.sim

    def contract(self) -> dict[str, Any]:
        """Return a serializable description suitable for run receipts."""

        return {
            "schema": "zuma-rl.park-settle-macro-contract",
            "version": 1,
            "macro_verbs": list(MACRO_VERB_NAMES),
            "aim_bins": self.aim_bins,
            "reaction_delay_ticks": self.reaction_delay_ticks,
            "config": asdict(self.config),
            "reward_is_per_tick_sum": True,
            "discount_applies_per_macro_step": True,
            "hop_verb_exposed": False,
            "actuator_contract": self._human.contract(),
        }

    # ------------------------------------------------------------------
    # Internal per-tick execution
    # ------------------------------------------------------------------

    def _bin_distance(self, first: int, second: int) -> int:
        direct = abs(int(first) - int(second)) % self.aim_bins
        return min(direct, self.aim_bins - direct)

    def _at_decision_point(self) -> bool:
        return (
            self.sim.gun_state is GunState.NORMAL
            and self._last_pending_buttons == 0
        )

    def _sim_aim_bin(self) -> int:
        angle = float(self.sim.aim_angle) % math.tau
        return min(self.aim_bins - 1, int(angle * self.aim_bins / math.tau))

    def _decode_macro(self, action: Any) -> tuple[int, int]:
        values = np.asarray(action, dtype=np.int64).reshape(-1)
        if values.shape != (2,):
            raise ValueError(f"invalid macro action: {action!r}")
        verb, target = int(values[0]), int(values[1])
        if not 0 <= verb < len(MACRO_VERB_NAMES):
            raise ValueError(f"macro verb out of range: {verb}")
        if not 0 <= target < self.aim_bins:
            raise ValueError(f"target aim bin out of range: {target}")
        return verb, target

    def action_masks(self) -> NDArray[np.bool_]:
        """Return the MaskablePPO mask for the macro MultiDiscrete space.

        Verb validity is delegated to the actuator's own mask (which itself
        extends the retail state machine's mask), truncated to the three
        macro verbs.  Aim bins remain unrestricted, exactly as in the
        per-tick interface.
        """

        verbs = np.asarray(self._human.valid_verb_mask(), dtype=np.bool_)
        macro_verbs = verbs[: len(MACRO_VERB_NAMES)]
        return np.concatenate(
            (macro_verbs, np.ones(self.aim_bins, dtype=np.bool_))
        )

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[FloatObservation, dict[str, Any]]:
        observation, info = self.env.reset(seed=seed, options=options)
        self._last_executed_aim_bin = self._sim_aim_bin()
        self._last_pending_buttons = 0
        info = dict(info)
        info["park_settle"] = {
            "profile": self.config.profile_id,
            "decision_point": self._at_decision_point(),
        }
        return observation, info

    def step(
        self,
        action: Any,
    ) -> tuple[FloatObservation, float, bool, bool, dict[str, Any]]:
        macro_verb, target = self._decode_macro(action)
        config = self.config
        telemetry: dict[str, Any] = {
            "ticks_consumed": 0,
            "settle_ticks": 0,
            "settled": False,
            "button_executed": False,
            "released": False,
            "release_aim_bin": None,
        }
        total_reward = 0.0
        last: tuple[FloatObservation, bool, bool, dict[str, Any]] | None = None
        done = False

        def tick(verb: int) -> bool:
            """Advance one native tick; return True when the episode ends."""

            nonlocal total_reward, last, done
            gun_before = self.sim.gun_state
            native = self._human.revenge_env.encode_action(verb, target)
            observation, reward, terminated, truncated, info = self.env.step(
                native
            )
            gun_after = self.sim.gun_state
            actuator = info.get("human_speedrun", {})
            self._last_executed_aim_bin = int(
                actuator.get("executed_aim_bin", self._last_executed_aim_bin)
            )
            self._last_pending_buttons = int(
                actuator.get("pending_buttons", 0)
            )
            telemetry["ticks_consumed"] += 1
            if int(actuator.get("executed_verb", 0)) != 0:
                telemetry["button_executed"] = True
            if gun_before is GunState.FIRING and gun_after is GunState.RELOADING:
                # Retail samples the live aim angle at release; the wrapper
                # re-set it this very tick, so the executed bin is the
                # release bin.
                telemetry["released"] = True
                telemetry["release_aim_bin"] = self._last_executed_aim_bin
            total_reward += float(reward)
            last = (observation, terminated, truncated, info)
            done = bool(terminated or truncated)
            return done

        if macro_verb == MACRO_WAIT_HOLD:
            for _ in range(config.hold_ticks):
                if tick(0):
                    break
            telemetry["settled"] = (
                self._bin_distance(self._last_executed_aim_bin, target)
                <= config.settle_tolerance_bins
            )
        elif macro_verb == MACRO_FIRE:
            while (
                not done
                and self._bin_distance(self._last_executed_aim_bin, target)
                > config.settle_tolerance_bins
                and telemetry["settle_ticks"] < config.settle_timeout_ticks
            ):
                telemetry["settle_ticks"] += 1
                tick(0)
            telemetry["settled"] = (
                self._bin_distance(self._last_executed_aim_bin, target)
                <= config.settle_tolerance_bins
            )
            if not done:
                tick(1)
            hold = (
                self.reaction_delay_ticks
                + config.fire_animation_ticks
                + config.post_release_margin_ticks
            )
            for _ in range(hold):
                if done:
                    break
                tick(0)
        elif macro_verb == MACRO_SWAP:
            # Swap needs no settle gate: retail swap ignores the cursor
            # angle.  The edge is still held through the reaction delay so
            # the queued button cannot be dropped, and the cursor keeps
            # parking on the carried target throughout.
            if not done:
                tick(2)
            hold = self.reaction_delay_ticks + config.post_release_margin_ticks
            for _ in range(hold):
                if done:
                    break
                tick(0)
            telemetry["settled"] = (
                self._bin_distance(self._last_executed_aim_bin, target)
                <= config.settle_tolerance_bins
            )
        else:  # pragma: no cover - _decode_macro guards
            raise RuntimeError(f"unhandled macro verb: {macro_verb}")

        rolled = 0
        while not done and not self._at_decision_point():
            if rolled >= config.decision_roll_limit_ticks:
                raise RuntimeError(
                    "park-settle roll to the next decision point exceeded "
                    f"{config.decision_roll_limit_ticks} ticks"
                )
            rolled += 1
            tick(0)

        if last is None:  # pragma: no cover - every macro emits >= 1 tick
            raise RuntimeError("park-settle macro consumed zero ticks")
        observation, terminated, truncated, info = last
        info = dict(info)
        info["park_settle"] = {
            "profile": config.profile_id,
            "macro_verb": MACRO_VERB_NAMES[macro_verb],
            "target_aim_bin": target,
            "ticks_consumed": int(telemetry["ticks_consumed"]),
            "settle_ticks": int(telemetry["settle_ticks"]),
            "settled": bool(telemetry["settled"]),
            "button_executed": bool(telemetry["button_executed"]),
            "released": bool(telemetry["released"]),
            "release_aim_bin": telemetry["release_aim_bin"],
            "decision_point": self._at_decision_point(),
            "terminal": bool(terminated or truncated),
        }
        return observation, float(total_reward), terminated, truncated, info
