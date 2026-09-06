from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

from tools import evaluate_zero_shot_multilevel as legacy
from tools import evaluate_zero_shot_multilevel_v2 as evaluator_v2
from zuma_rl.revenge_env import RevengeEnv


class _StubRevengeEnv(RevengeEnv):
    def __init__(self, *, mode: str):
        gym.Env.__init__(self)
        self.mode = mode
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(2)
        self.config = SimpleNamespace(
            max_balls=768,
            score_reward_scale=1.0,
            step_penalty=-0.01,
            loss_reward=-1000.0,
        )
        self._last_info = {}
        self._episode_done = False

    def reset(self, *, seed=None, options=None):
        del seed, options
        self._episode_done = False
        self._last_info = {
            "outcome": None,
            "native_outcome": None,
            "visible_balls": 1,
            "ticks": 0,
            "ticks_advanced": 0,
            "score": 0,
            "score_delta": 0,
        }
        return np.array([0.1, 0.2, 0.3], dtype=np.float32), dict(self._last_info)

    def step(self, action):
        del action
        if self.mode == "other_error":
            raise RuntimeError("unrelated simulator defect")
        if self.mode == "overflow":
            self._last_info = {
                "outcome": None,
                "native_outcome": None,
                "visible_balls": 769,
                "ticks": 17777,
                "ticks_advanced": 1,
                "score": 4321,
                "score_delta": 7,
                "verb": "fire",
                "verb_id": 1,
                "aim_bin": 42,
                "action_accepted": True,
            }
            raise RuntimeError(
                "actor observation capacity exceeded: 769 visible balls > "
                "max_balls=768; increase max_balls instead of silently "
                "truncating game state"
            )
        self._last_info = {
            "outcome": None,
            "native_outcome": None,
            "visible_balls": 2,
            "ticks": 1,
            "ticks_advanced": 1,
            "score": 3,
            "score_delta": 3,
        }
        return np.array([0.2, 0.3, 0.4], dtype=np.float32), 2.99, False, False, dict(self._last_info)


def test_capacity_overflow_becomes_explicit_terminal_failure() -> None:
    env = evaluator_v2.ObservationCapacityFailClosed(
        _StubRevengeEnv(mode="overflow")
    )
    initial, _ = env.reset()
    observation, reward, terminated, truncated, info = env.step(1)
    assert np.array_equal(observation, initial)
    assert reward == pytest.approx(-993.01)
    assert terminated is True
    assert truncated is False
    assert info["outcome"] == "loss"
    assert info["native_outcome"] is None
    assert info["observation_capacity_overflow"] is True
    assert info["capacity_limit_balls"] == 768
    assert info["visible_balls_at_failure"] == 769
    assert info["evaluation_fail_closed"] is True
    assert env.revenge_env._episode_done is True


def test_capacity_wrapper_preserves_normal_steps() -> None:
    env = evaluator_v2.ObservationCapacityFailClosed(_StubRevengeEnv(mode="ok"))
    env.reset()
    observation, reward, terminated, truncated, info = env.step(0)
    assert np.array_equal(observation, np.array([0.2, 0.3, 0.4], dtype=np.float32))
    assert reward == pytest.approx(2.99)
    assert terminated is False
    assert truncated is False
    assert info["visible_balls"] == 2


def test_capacity_wrapper_does_not_hide_other_runtime_errors() -> None:
    env = evaluator_v2.ObservationCapacityFailClosed(
        _StubRevengeEnv(mode="other_error")
    )
    env.reset()
    with pytest.raises(RuntimeError, match="unrelated simulator defect"):
        env.step(0)


def test_capacity_aware_tracker_retains_failure_evidence() -> None:
    tracker = evaluator_v2.CapacityAwareEpisodeTracker(seed=123)
    info = {
        "human_speedrun": {"executed_verb": 1, "executed_aim_bin": 42},
        "outcome": "loss",
        "native_outcome": None,
        "ticks": 17777,
        "score": 4321,
        "score_delta": 7,
        "observation_capacity_overflow": True,
        "failure_reason": "actor_observation_capacity_overflow",
        "capacity_limit_balls": 768,
        "visible_balls_at_failure": 769,
        "capacity_error": "actor observation capacity exceeded: 769",
    }
    tracker.update(np.array([1, 42]), -10.0, info)
    row = tracker.finish(
        info=info,
        model={"id": "m", "training_steps": 1, "sha256": "sha256:model"},
        threshold_ticks=0,
    )
    assert row["outcome"] == "loss"
    assert row["observation_capacity_overflow"] is True
    assert row["capacity_limit_balls"] == 768
    assert row["visible_balls_at_failure"] == 769
    assert row["evaluation_fail_closed"] is True


def test_main_rebinds_legacy_evaluator_identity(monkeypatch) -> None:
    observed = {}

    def fake_main(argv):
        observed["argv"] = argv
        observed["file"] = legacy.__file__
        observed["factory"] = legacy._make_env_factory
        observed["tracker"] = legacy.EpisodeTracker
        return 7

    monkeypatch.setattr(legacy, "main", fake_main)
    assert evaluator_v2.main(["--validate-only"]) == 7
    assert observed["argv"] == ["--validate-only"]
    assert observed["file"] == evaluator_v2.__file__
    assert observed["factory"] is evaluator_v2._make_capacity_safe_env_factory
    assert observed["tracker"] is evaluator_v2.CapacityAwareEpisodeTracker
