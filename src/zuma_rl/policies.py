"""Small non-learning policies used for smoke tests and demonstrations."""

from __future__ import annotations

import numpy as np

from zuma_rl.core import ZumaSimulator, resolve_matches


def greedy_action(sim: ZumaSimulator) -> int:
    """Choose the best immediate shot among rays passing through a ball.

    This policy deliberately has no long-term planning.  It is useful as a
    mechanics smoke test and as a benchmark baseline, not as the final agent.
    """

    if not sim.balls:
        return 0

    candidates: set[int] = set()
    for point in sim.ball_positions():
        for swap in (False, True):
            center = sim.action_towards(point, swap=swap)
            offset = sim.config.aim_bins if swap else 0
            aim_bin = center % sim.config.aim_bins
            for delta in (-1, 0, 1):
                candidates.add(
                    (aim_bin + delta) % sim.config.aim_bins + offset
                )

    best_action = min(candidates)
    best_value = -np.inf
    for action in candidates:
        angle, swap = sim.action_to_angle(action)
        fired_color = sim.next_color if swap else sim.current_color
        hit_index = sim._ray_hit(angle)
        if hit_index is None:
            value = -2.0
        else:
            insertion_index = sim._insertion_index(hit_index, angle)
            trial = sim.balls.copy()
            trial.insert(insertion_index, fired_color)
            remaining, removed, cascades = resolve_matches(
                trial,
                insertion_index,
                sim.config.match_size,
            )
            value = (
                removed * 10.0
                + max(0, cascades - 1) * 15.0
                - len(remaining) * 0.01
            )
            if not remaining:
                value += 1_000.0
        if value > best_value or (
            value == best_value and action < best_action
        ):
            best_value = value
            best_action = action
    return best_action

