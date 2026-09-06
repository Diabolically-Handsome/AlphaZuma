"""Training-only simulator lookahead teacher for AlphaZuma 55.

The teacher may inspect and clone the live simulator while demonstrations are
collected.  It is therefore privileged engineering machinery, never a valid
runtime policy.  Candidate actions still pass through the frozen human-input
wrapper, so reaction delay, aim acceleration, and button cadence are preserved.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from zuma_rl.human_speedrun import HumanSpeedrunWrapper
from zuma_rl.revenge_env import FloatObservation
from zuma_rl.revenge_strategic_teacher import (
    StrategicActorObservableRevengeTeacher,
)


@dataclass(frozen=True, slots=True)
class LookaheadOutcome:
    aim_bin: int
    fire_executed: bool
    outcome: str | None
    score_gain: int
    ball_reduction: int
    danger_gain: float
    simulated_ticks: int
    static_rank: tuple[float, ...]

    def rank(self) -> tuple[float, ...]:
        return (
            0.0 if self.fire_executed else 1.0,
            0.0 if self.outcome == "win" else 1.0,
            1.0 if self.outcome == "loss" else 0.0,
            0.0 if self.danger_gain > 0.0 else 1.0,
            -float(self.danger_gain),
            0.0 if self.ball_reduction > 0 else 1.0,
            -float(self.ball_reduction),
            0.0 if self.score_gain > 0 else 1.0,
            -float(self.score_gain),
            float(self.simulated_ticks),
            *self.static_rank,
            float(self.aim_bin),
        )


class PrivilegedLookaheadRevengeTeacher:
    """Select fire angles by simulating the canonical human-input outcome."""

    policy_id = "privileged-lookahead-human-input-v1"

    def __init__(
        self,
        env: HumanSpeedrunWrapper,
        *,
        max_candidates: int = 8,
        horizon_ticks: int = 240,
        settle_ticks_after_fire: int = 32,
    ) -> None:
        if max_candidates < 1:
            raise ValueError("max_candidates must be positive")
        if horizon_ticks < 1:
            raise ValueError("horizon_ticks must be positive")
        if settle_ticks_after_fire < 1:
            raise ValueError("settle_ticks_after_fire must be positive")
        self.env = env
        self.strategic = StrategicActorObservableRevengeTeacher.from_env(
            env.revenge_env
        )
        self.max_candidates = int(max_candidates)
        self.horizon_ticks = int(horizon_ticks)
        self.settle_ticks_after_fire = int(settle_ticks_after_fire)
        self.planning_decisions = 0
        self.candidate_simulations = 0
        self.positive_lookaheads = 0
        self._last_aim_bin = 0

    @staticmethod
    def _sim_danger(env: HumanSpeedrunWrapper) -> float:
        sim = env.revenge_env.sim
        values = [
            float(ball.waypoint)
            for state in sim.curve_states
            for ball in state.balls
        ]
        return max(values, default=0.0)

    def _candidate_aims(
        self, observation: FloatObservation
    ) -> list[tuple[int, tuple[float, ...]]]:
        teacher = self.strategic
        matrix, globals_ = teacher._split_observation(observation)
        current = teacher._decode_color(
            globals_, "current_color", missing_name="current_missing"
        )
        if current is None:
            return []
        balls = teacher._decode_balls(matrix)
        if not balls:
            return []
        shooter = (teacher.spec.shooter_x, teacher.spec.shooter_y)
        if {"shooter_x", "shooter_y"}.issubset(teacher._global_index):
            shooter = (
                (float(globals_[teacher._global_index["shooter_x"]]) + 1.0)
                * teacher.spec.logical_width
                / 2.0,
                (float(globals_[teacher._global_index["shooter_y"]]) + 1.0)
                * teacher.spec.logical_height
                / 2.0,
            )

        ranked: dict[int, tuple[float, ...]] = {}
        for target in balls:
            if bool(getattr(target, "exploding")) or int(
                getattr(target, "color")
            ) != int(current):
                continue
            centre = teacher._aim_bin(
                float(getattr(target, "x")),
                float(getattr(target, "y")),
                shooter,
            )
            for delta in (-1, 0, 1):
                aim_bin = (centre + delta) % teacher.spec.aim_bins
                hit = teacher._first_hit(balls, aim_bin, shooter)
                if hit is None:
                    continue
                hit_index, entry = hit
                first = balls[hit_index]
                if int(getattr(first, "color")) != int(current):
                    continue
                left, right = teacher._connected_bounds(
                    balls, matrix, hit_index, int(current)
                )
                run_length = right - left + 1
                danger = max(
                    float(
                        matrix[
                            int(getattr(balls[position], "slot")),
                            teacher._ball_index["waypoint"],
                        ]
                    )
                    for position in range(left, right + 1)
                )
                static_rank = (
                    0.0 if run_length >= 2 else 1.0,
                    -float(run_length),
                    -float(danger),
                    float(entry),
                )
                previous = ranked.get(aim_bin)
                if previous is None or static_rank < previous:
                    ranked[aim_bin] = static_rank

        strategic_action = teacher.act(observation)
        strategic_aim: int | None = None
        if int(strategic_action[0]) == 1:
            strategic_aim = int(strategic_action[1])
            ranked.setdefault(
                strategic_aim,
                (2.0, 0.0, 0.0, 0.0),
            )
        if not ranked:
            fallback = teacher._strategic_fallback(balls, matrix, shooter)
            if fallback is not None:
                ranked[int(fallback.aim_bin)] = (
                    3.0,
                    0.0,
                    -float(fallback.danger),
                    float(fallback.entry_distance),
                )
        selected = sorted(
            ranked.items(), key=lambda item: (*item[1], item[0])
        )[: self.max_candidates]
        if (
            strategic_aim is not None
            and strategic_aim in ranked
            and all(aim_bin != strategic_aim for aim_bin, _ in selected)
        ):
            selected[-1] = (strategic_aim, ranked[strategic_aim])
        return selected

    def _simulate_fire(
        self,
        aim_bin: int,
        static_rank: tuple[float, ...],
    ) -> LookaheadOutcome:
        clone = copy.deepcopy(self.env)
        before_score = int(clone.revenge_env.sim.score)
        before_balls = int(clone.revenge_env.sim.total_ball_count)
        before_danger = self._sim_danger(clone)
        fire_executed = False
        fire_tick: int | None = None
        outcome: str | None = None
        simulated = 0
        try:
            observation, _, terminated, truncated, info = clone.step(
                np.asarray((1, aim_bin), dtype=np.int64)
            )
            del observation
            simulated += int(info.get("ticks_advanced", 1))
            while not (terminated or truncated) and simulated < self.horizon_ticks:
                _, _, terminated, truncated, info = clone.step(
                    np.asarray((0, aim_bin), dtype=np.int64)
                )
                simulated += int(info.get("ticks_advanced", 1))
                human = info.get("human_speedrun", {})
                if int(human.get("executed_verb", 0)) == 1:
                    fire_executed = True
                    fire_tick = int(info["ticks"])
                if fire_executed and fire_tick is not None:
                    sim = clone.revenge_env.sim
                    settled = (
                        int(info["ticks"]) - fire_tick
                        >= self.settle_ticks_after_fire
                        and len(sim.free_projectiles) == 0
                        and int(sim.total_merging_projectile_count) == 0
                    )
                    if settled:
                        break
            outcome = info.get("outcome")
            after_score = int(clone.revenge_env.sim.score)
            after_balls = int(clone.revenge_env.sim.total_ball_count)
            after_danger = self._sim_danger(clone)
            return LookaheadOutcome(
                aim_bin=int(aim_bin),
                fire_executed=fire_executed,
                outcome=(str(outcome) if outcome is not None else None),
                score_gain=after_score - before_score,
                ball_reduction=before_balls - after_balls,
                danger_gain=before_danger - after_danger,
                simulated_ticks=simulated,
                static_rank=tuple(float(value) for value in static_rank),
            )
        finally:
            clone.close()

    @staticmethod
    def _masked_action(
        action: NDArray[np.int64], masks: NDArray[np.bool_], fallback_aim: int
    ) -> NDArray[np.int64]:
        verb = int(action[0])
        if not bool(masks[verb]):
            return np.asarray((0, fallback_aim), dtype=np.int64)
        return np.asarray(action, dtype=np.int64)

    def act(self, observation: FloatObservation) -> NDArray[np.int64]:
        masks = np.asarray(self.env.action_masks(), dtype=np.bool_)[:4]
        strategic_action = self._masked_action(
            self.strategic.act(observation), masks, self._last_aim_bin
        )
        if not bool(masks[1]):
            self._last_aim_bin = int(strategic_action[1])
            return strategic_action

        candidates = self._candidate_aims(observation)
        if not candidates:
            self._last_aim_bin = int(strategic_action[1])
            return strategic_action
        self.planning_decisions += 1
        outcomes = [
            self._simulate_fire(aim_bin, static_rank)
            for aim_bin, static_rank in candidates
        ]
        self.candidate_simulations += len(outcomes)
        best = min(outcomes, key=lambda item: item.rank())
        if int(strategic_action[0]) == 1:
            strategic_outcome = next(
                (
                    outcome
                    for outcome in outcomes
                    if outcome.aim_bin == int(strategic_action[1])
                ),
                None,
            )
            if strategic_outcome is not None and best is not strategic_outcome:
                meaningful_override = (
                    (best.outcome == "win" and strategic_outcome.outcome != "win")
                    or (
                        best.outcome != "loss"
                        and strategic_outcome.outcome == "loss"
                    )
                    or (
                        best.ball_reduction
                        >= strategic_outcome.ball_reduction + 3
                        and best.score_gain >= strategic_outcome.score_gain
                    )
                    or (
                        best.score_gain >= strategic_outcome.score_gain + 1_000
                        and best.danger_gain >= strategic_outcome.danger_gain
                    )
                )
                if not meaningful_override:
                    best = strategic_outcome
        if best.score_gain > 0 or best.outcome == "win":
            self.positive_lookaheads += 1
        if (
            int(strategic_action[0]) in {2, 3}
            and best.score_gain <= 0
            and best.outcome != "win"
            and bool(masks[int(strategic_action[0])])
        ):
            self._last_aim_bin = int(strategic_action[1])
            return strategic_action
        self._last_aim_bin = int(best.aim_bin)
        return np.asarray((1, best.aim_bin), dtype=np.int64)

    def metrics(self) -> dict[str, Any]:
        return {
            "planning_decisions": self.planning_decisions,
            "candidate_simulations": self.candidate_simulations,
            "positive_lookaheads": self.positive_lookaheads,
            "max_candidates": self.max_candidates,
            "horizon_ticks": self.horizon_ticks,
            "settle_ticks_after_fire": self.settle_ticks_after_fire,
        }
