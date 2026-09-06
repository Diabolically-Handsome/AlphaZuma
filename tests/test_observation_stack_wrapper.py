"""Tests for the runtime observation stack wrapper.

Covers:
(a) ring-buffer lag correctness against a hand-computed sequence,
    including pre-episode clamping to the earliest frame and buffer
    clearing across resets;
(b) trainer-vs-wrapper consistency: a recorded contiguous tick sequence
    fed through the wrapper yields exactly the stacked vectors the v3
    trainer's gather produces for the same rows;
(c) space widening and attribute forwarding (action_masks/teacher_spec)
    so the wrapper composes with the probes' per-env factory convention.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from tools import distill_alphazuma_55_park_settle_v3 as v3
from zuma_rl.observation_stack_wrapper import (
    DEFAULT_STACK_LAGS,
    ObservationStackWrapper,
    parse_stack_lags,
    stacked_box,
)


class _SequenceEnv(gym.Env):
    """Replays a fixed [ticks, width] frame matrix, one row per tick."""

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(self, frames: np.ndarray) -> None:
        super().__init__()
        self._frames = np.asarray(frames, dtype=np.float32)
        self._cursor = 0
        width = int(self._frames.shape[1])
        self.observation_space = spaces.Box(
            low=-1000.0, high=1000.0, shape=(width,), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete([4, 8])

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._cursor = 0
        return self._frames[0].copy(), {}

    def step(
        self, action: Any
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._cursor += 1
        done = self._cursor >= self._frames.shape[0] - 1
        return self._frames[self._cursor].copy(), 0.0, done, False, {}

    def action_masks(self) -> np.ndarray:
        return np.ones(12, dtype=np.bool_)

    def teacher_spec(self) -> dict[str, Any]:
        return {"marker": 7}


def _frame_matrix(ticks: int, width: int) -> np.ndarray:
    frames = np.arange(ticks, dtype=np.float32).reshape(ticks, 1)
    return frames + np.arange(width, dtype=np.float32) / 100.0


def test_parse_stack_lags_round_trips_strings_and_sequences() -> None:
    assert parse_stack_lags("0,4,8") == (0, 4, 8)
    assert parse_stack_lags(" 0, 4 ,8 ") == (0, 4, 8)
    assert parse_stack_lags((0, 4, 8)) == DEFAULT_STACK_LAGS
    assert parse_stack_lags("0") == (0,)
    with pytest.raises(ValueError, match="integers"):
        parse_stack_lags("0,a")
    with pytest.raises(ValueError, match="first stack lag"):
        parse_stack_lags("2,0")


def test_ring_buffer_matches_hand_computed_lags_and_resets() -> None:
    width = 4
    frames = _frame_matrix(9, width)
    env = ObservationStackWrapper(
        _SequenceEnv(frames), stack_lags=(0, 2, 5)
    )
    assert env.observation_space.shape == (3 * width,)

    observation, _ = env.reset()
    # Tick 0: every lag clamps to the only available frame.
    assert np.array_equal(
        observation, np.concatenate([frames[0]] * 3)
    )
    stacked_by_tick = {0: observation}
    for tick in range(1, 9):
        observation, _, done, _, _ = env.step((0, 0))
        stacked_by_tick[tick] = observation
    assert done

    # Hand-computed: lag 2 and lag 5 clamp to frame 0 until enough
    # history exists, then track exactly t-2 and t-5.
    assert np.array_equal(
        stacked_by_tick[1],
        np.concatenate([frames[1], frames[0], frames[0]]),
    )
    assert np.array_equal(
        stacked_by_tick[4],
        np.concatenate([frames[4], frames[2], frames[0]]),
    )
    assert np.array_equal(
        stacked_by_tick[8],
        np.concatenate([frames[8], frames[6], frames[3]]),
    )

    # Reset clears the buffer: no frames leak across episodes.
    observation, _ = env.reset()
    assert np.array_equal(
        observation, np.concatenate([frames[0]] * 3)
    )
    observation, _, _, _, _ = env.step((0, 0))
    assert np.array_equal(
        observation,
        np.concatenate([frames[1], frames[0], frames[0]]),
    )


def test_wrapper_matches_trainer_gather_on_contiguous_rows() -> None:
    width = 6
    ticks = 14
    frames = _frame_matrix(ticks, width)
    lags = (0, 4, 8)

    gather, stats = v3.build_stack_gather(
        episode_indices=np.zeros(ticks, dtype=np.int64),
        tick_indices=np.arange(ticks, dtype=np.int64),
        stack_lags=lags,
    )
    view = v3.StackedObservationView(frames, gather)
    assert stats["nearest_earlier"] == 0  # contiguous: clamps only
    # The trainer receipts ZERO train/runtime divergence here, which is
    # exactly the precondition for the equality this test verifies.
    assert stats["train_runtime_divergence"]["divergent_reads"] == 0

    env = ObservationStackWrapper(_SequenceEnv(frames), stack_lags=lags)
    observation, _ = env.reset()
    for tick in range(ticks):
        if tick:
            observation, _, _, _, _ = env.step((0, 0))
        expected = view[np.asarray([tick], dtype=np.int64)].reshape(-1)
        assert np.array_equal(observation, expected), tick


def test_wrapper_spaces_and_method_forwarding() -> None:
    frames = _frame_matrix(3, 5)
    inner = _SequenceEnv(frames)
    env = ObservationStackWrapper(inner, stack_lags="0,3")
    assert env.stack_lags == (0, 3)
    assert env.raw_observation_dim == 5
    assert isinstance(env.observation_space, spaces.Box)
    assert env.observation_space.dtype == np.float32
    assert np.array_equal(
        env.observation_space.low,
        np.tile(inner.observation_space.low, 2),
    )
    # Per-env probe plumbing: SubprocVecEnv.env_method resolves methods
    # with gymnasium's get_wrapper_attr, which walks the wrapper stack.
    assert env.get_wrapper_attr("action_masks")().shape == (12,)
    assert env.get_wrapper_attr("teacher_spec")() == {"marker": 7}

    box = stacked_box(inner.observation_space, 3)
    assert box.shape == (15,)
    with pytest.raises(TypeError, match="1-D Box"):
        stacked_box(spaces.MultiDiscrete([2, 2]), 2)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="1-D Box"):
        ObservationStackWrapper(_BadSpaceEnv(), stack_lags=(0, 2))


class _BadSpaceEnv(gym.Env):
    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.MultiDiscrete([3, 3])
        self.action_space = spaces.Discrete(2)
