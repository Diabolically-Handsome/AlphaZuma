from __future__ import annotations

import numpy as np
import pytest

from zuma_rl.alphazuma_55 import full55_environment_config
from zuma_rl.human_speedrun import HumanSpeedrunWrapper
from zuma_rl.observable_human_speedrun import ObservableHumanSpeedrunWrapper
from zuma_rl.park_settle_action_wrapper import (
    MACRO_VERB_NAMES,
    ParkSettleActionConfig,
    ParkSettleActionWrapper,
    resolve_human_speedrun_wrapper,
)
from zuma_rl.revenge_core import GunState
from zuma_rl.revenge_env import RevengeEnv


def _stack(
    seed: int,
    *,
    max_ticks: int = 4_000,
    observable: bool = False,
    **config: int,
) -> ParkSettleActionWrapper:
    base = RevengeEnv(
        config=full55_environment_config(max_ticks=max_ticks),
        level_id="Jungle1",
        seed=seed,
    )
    wrapper_class = (
        ObservableHumanSpeedrunWrapper if observable else HumanSpeedrunWrapper
    )
    return ParkSettleActionWrapper(
        wrapper_class(base),
        config=ParkSettleActionConfig(**config) if config else None,
    )


def _bin_distance(first: int, second: int, bins: int = 180) -> int:
    direct = abs(int(first) - int(second)) % bins
    return min(direct, bins - direct)


def _macro(verb: int, target: int) -> np.ndarray:
    return np.asarray((verb, target), dtype=np.int64)


def test_fire_macro_releases_within_settle_tolerance() -> None:
    env = _stack(7)
    try:
        _, info = env.reset(seed=13)
        assert info["park_settle"]["decision_point"] is True
        # Cursor starts at bin 0; force a worst-case-adjacent 90 degree park.
        target = 45
        _, reward, terminated, truncated, info = env.step(_macro(1, target))
        macro = info["park_settle"]
        assert not terminated and not truncated
        assert macro["macro_verb"] == "fire"
        assert macro["settled"] is True
        assert macro["released"] is True
        assert macro["button_executed"] is True
        assert _bin_distance(macro["release_aim_bin"], target) <= 2
        # The macro spans the park, the delayed edge, the firing animation,
        # and the roll back to the next decision point.
        assert macro["ticks_consumed"] >= env.reaction_delay_ticks + 6
        assert macro["ticks_consumed"] <= 120
        # The next state handed to the policy is actionable again.
        assert macro["decision_point"] is True
        assert env.sim.gun_state is GunState.NORMAL
        assert isinstance(reward, float)
    finally:
        env.close()


def test_fire_macro_timeout_fires_unsettled() -> None:
    env = _stack(11, settle_timeout_ticks=1)
    try:
        env.reset(seed=17)
        # Prime past the initial gun load (ticks 1-15 are RELOADING) so the
        # delayed fire edge lands while the gun is NORMAL.
        env.step(_macro(0, 0))
        target = 90  # 180 degrees away: unreachable within one settle tick.
        _, _, terminated, truncated, info = env.step(_macro(1, target))
        macro = info["park_settle"]
        assert not terminated and not truncated
        assert macro["settle_ticks"] == 1
        assert macro["settled"] is False
        assert macro["released"] is True
        assert _bin_distance(macro["release_aim_bin"], target) > 2
    finally:
        env.close()


def test_wait_hold_consumes_hold_ticks() -> None:
    env = _stack(19)
    try:
        env.reset(seed=23)
        # The very first macro also rolls through the initial gun load
        # (ticks 1-15 are RELOADING) to reach the next decision point.
        _, _, _, _, info = env.step(_macro(0, 90))
        assert info["park_settle"]["ticks_consumed"] == 16
        ticks_before = int(env.sim.tick_count)
        _, reward, terminated, truncated, info = env.step(_macro(0, 90))
        macro = info["park_settle"]
        assert not terminated and not truncated
        assert macro["macro_verb"] == "wait_hold"
        assert macro["ticks_consumed"] == 8
        assert int(env.sim.tick_count) - ticks_before == 8
        # Macro reward is the plain sum of per-tick rewards: eight ticks of
        # the time penalty, no score events this early in Jungle1.
        assert reward == pytest.approx(8 * -0.0001, abs=1e-9)
    finally:
        env.close()

    env = _stack(19, hold_ticks=5)
    try:
        env.reset(seed=23)
        env.step(_macro(0, 90))
        _, _, _, _, info = env.step(_macro(0, 90))
        assert info["park_settle"]["ticks_consumed"] == 5
    finally:
        env.close()


def test_swap_macro_executes_button_and_returns_to_decision_point() -> None:
    env = _stack(29)
    try:
        env.reset(seed=31)
        _, _, terminated, truncated, info = env.step(_macro(2, 10))
        macro = info["park_settle"]
        assert not terminated and not truncated
        assert macro["macro_verb"] == "swap"
        assert macro["button_executed"] is True
        assert macro["released"] is False
        assert macro["decision_point"] is True
    finally:
        env.close()


def test_action_masks_shape_and_dtype() -> None:
    env = _stack(37)
    try:
        env.reset(seed=41)
        # At tick 0 the gun is NORMAL but not yet loaded, so fire is masked.
        masks = env.action_masks()
        assert masks.shape == (len(MACRO_VERB_NAMES) + 180,)
        assert bool(masks[0]) and not bool(masks[1])
        # After one macro the initial load is complete and fire is valid.
        env.step(_macro(0, 0))
        masks = env.action_masks()
        assert masks.shape == (len(MACRO_VERB_NAMES) + 180,)
        assert masks.dtype == np.bool_
        # Wait and fire are valid at a settled decision point; every aim
        # bin is always available.
        assert bool(masks[0]) and bool(masks[1])
        assert bool(np.all(masks[len(MACRO_VERB_NAMES) :]))
        assert env.action_space.nvec.tolist() == [len(MACRO_VERB_NAMES), 180]
    finally:
        env.close()


def test_macro_returns_immediately_on_time_limit() -> None:
    env = _stack(43, max_ticks=10, hold_ticks=30)
    try:
        env.reset(seed=47)
        _, _, terminated, truncated, info = env.step(_macro(0, 0))
        assert truncated is True
        assert terminated is False
        assert info["park_settle"]["ticks_consumed"] == 10
        assert info["park_settle"]["terminal"] is True
    finally:
        env.close()


def test_wrapper_accepts_observable_stack_and_resolves_actuator() -> None:
    env = _stack(53, observable=True)
    try:
        human = resolve_human_speedrun_wrapper(env)
        assert isinstance(human, ObservableHumanSpeedrunWrapper)
        observation, _ = env.reset(seed=59)
        assert observation.shape == env.observation_space.shape
        _, _, _, _, info = env.step(_macro(1, 30))
        macro = info["park_settle"]
        assert macro["released"] is True
        assert _bin_distance(macro["release_aim_bin"], 30) <= 2
        contract = env.contract()
        assert contract["discount_applies_per_macro_step"] is True
        assert contract["macro_verbs"] == ["wait_hold", "fire", "swap"]
    finally:
        env.close()


def test_requires_a_human_speedrun_stack() -> None:
    base = RevengeEnv(
        config=full55_environment_config(max_ticks=100),
        level_id="Jungle1",
        seed=61,
    )
    try:
        with pytest.raises(TypeError):
            ParkSettleActionWrapper(base)
    finally:
        base.close()
