"""Feature-axis observation stacking for temporal context (runtime side).

A single Revenge frame makes chain velocity unobservable: projectile
flight is 20-60 ticks and the chain keeps moving, so a single-frame
student cannot lead moving insertion points the way the settled teacher
(which derives velocity internally) does.  This wrapper gives a deployed
student the SAME temporal context the park-settle v3 trainer supervises
with: the current observation concatenated along the feature axis with
lagged observations, ``[frame(t - lag) for lag in stack_lags]`` in lag
order, so a model trained by ``tools.distill_alphazuma_55_park_settle_v3``
sees an identical input layout at runtime.

Consistency contract with the trainer's gather fallbacks:

* lag 0 leads the stack (enforced), so the CURRENT raw frame is always
  the leading ``raw_dim`` slice -- probes hand exactly that slice to the
  raw-frame teacher.
* pre-episode ticks (``tick - lag < 0``) reuse the EARLIEST available
  frame of the episode.  The trainer clamps a lagged read that precedes
  an episode's earliest retained row to that earliest retained row, so
  on a contiguous tick stream the wrapper and the trainer's gather emit
  identical stacked vectors (degenerating to duplicating the current
  frame on the very first tick, where nothing earlier exists).
* ``reset`` clears the ring buffer: frames never leak across episodes,
  mirroring the trainer's per-episode gather.
* retention GAPS in a training dataset have no runtime counterpart:
  this wrapper always holds the true ``t - lag`` frame, so wherever the
  trainer fell back to a nearest-earlier retained row (or clamped to an
  episode's mid-stream earliest retained row) the two deliberately
  diverge.  The trainer quantifies and receipts that divergence per lag
  (``train_runtime_divergence`` in its gather stats); equivalence is
  exact precisely when the receipted ``divergent_reads`` is zero.

Vector-env usage follows the existing probe convention: wrap PER ENV
inside the env factory (the callable handed to ``SubprocVecEnv``), never
around the vector env itself.  ``SubprocVecEnv.env_method`` resolves
methods (``action_masks``, ``teacher_spec``) with gymnasium's
``get_wrapper_attr``, which walks the wrapper stack, so the probes'
per-env plumbing works through this wrapper unchanged.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Iterable

import gymnasium as gym
import numpy as np
from gymnasium import spaces


DEFAULT_STACK_LAGS = (0, 4, 8)


def parse_stack_lags(value: str | Iterable[int]) -> tuple[int, ...]:
    """Normalize ``"0,4,8"`` or an int sequence into a validated tuple.

    Rules: non-empty; every lag a non-negative integer; no duplicates;
    the FIRST lag must be 0 so the current frame leads the stack (the
    polar aim basis and the probes' raw-frame teacher both read the
    leading slice).
    """

    if isinstance(value, str):
        pieces = [piece.strip() for piece in value.split(",")]
        pieces = [piece for piece in pieces if piece]
        try:
            lags = tuple(int(piece, 10) for piece in pieces)
        except ValueError as error:
            raise ValueError(
                f"stack lags must be comma-separated integers: {value!r}"
            ) from error
    else:
        lags = tuple(value)
        for lag in lags:
            if isinstance(lag, bool) or int(lag) != lag:
                raise ValueError(f"stack lags must be integers: {lags!r}")
        lags = tuple(int(lag) for lag in lags)
    if not lags:
        raise ValueError("stack lags must be non-empty")
    if any(lag < 0 for lag in lags):
        raise ValueError(f"stack lags must be non-negative: {lags!r}")
    if lags[0] != 0:
        raise ValueError(
            f"the first stack lag must be 0 so the current frame leads "
            f"the stack: {lags!r}"
        )
    if len(set(lags)) != len(lags):
        raise ValueError(f"stack lags must be unique: {lags!r}")
    return lags


def stacked_box(space: spaces.Box, frames: int) -> spaces.Box:
    """The 1-D Box space of ``frames`` feature-concatenated frames."""

    if not isinstance(space, spaces.Box) or len(space.shape) != 1:
        raise TypeError(
            f"observation stacking requires a 1-D Box space: {space!r}"
        )
    frames = int(frames)
    if frames < 1:
        raise ValueError("stacked frame count must be positive")
    return spaces.Box(
        low=np.tile(np.asarray(space.low), frames),
        high=np.tile(np.asarray(space.high), frames),
        dtype=space.dtype,
    )


class ObservationStackWrapper(gym.ObservationWrapper):
    """Emit ``concat(frame(t - lag) for lag in stack_lags)`` every tick.

    A ring buffer keeps the last ``max(stack_lags) + 1`` raw frames.
    Frames are defensively copied on entry so an environment reusing its
    observation buffer cannot corrupt history.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        stack_lags: str | Iterable[int] = DEFAULT_STACK_LAGS,
    ) -> None:
        super().__init__(env)
        self.stack_lags = parse_stack_lags(stack_lags)
        base_space = env.observation_space
        if not isinstance(base_space, spaces.Box) or len(base_space.shape) != 1:
            raise TypeError(
                "ObservationStackWrapper requires a 1-D Box observation "
                f"space, got {base_space!r}"
            )
        self.raw_observation_dim = int(base_space.shape[0])
        self.observation_space = stacked_box(base_space, len(self.stack_lags))
        self._history_depth = max(self.stack_lags) + 1
        self._frames: deque[np.ndarray] = deque(maxlen=self._history_depth)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        self._frames.clear()
        return super().reset(seed=seed, options=options)

    def observation(self, observation: Any) -> np.ndarray:
        frame = np.array(
            observation, dtype=self.observation_space.dtype, copy=True
        ).reshape(-1)
        if frame.shape != (self.raw_observation_dim,):
            raise ValueError(
                f"raw observation width changed: expected "
                f"{self.raw_observation_dim}, got {frame.shape}"
            )
        self._frames.append(frame)
        newest = len(self._frames) - 1
        parts = [
            self._frames[max(newest - lag, 0)] for lag in self.stack_lags
        ]
        return np.concatenate(parts)
