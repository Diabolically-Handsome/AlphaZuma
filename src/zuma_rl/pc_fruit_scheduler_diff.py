"""Feature-scoped retail versus simulator fruit-scheduler differential."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping, Sequence

from zuma_rl.pc_gameplay_diff import (
    _active,
    _fruit_state_mismatches,
    _transplant_midstate,
)
from zuma_rl.pc_mechanism_audit import (
    _mtrand_outputs_between,
    derive_native_mechanism_features,
)
from zuma_rl.pc_memory_trajectory import (
    PcMemoryTrajectoryError,
    TrajectoryFrame,
)
from zuma_rl.revenge_core import PopCapMTRandom, TickEvents


FRUIT_SCHEDULER_DIFF_SCHEMA = (
    "zuma-rl.pc-fruit-scheduler-simulator-diff"
)
FRUIT_SCHEDULER_DIFF_VERSION = 1


def _mtrand_matches(simulator: Any, frame: TrajectoryFrame) -> bool:
    state = frame.global_mtrand
    return bool(
        state is not None
        and simulator.rng.index == state.index
        and simulator.rng.words == state.words
    )


def compare_fruit_scheduler_transition(
    frames: Sequence[TrajectoryFrame],
    *,
    original_root: str | Path,
    level_id: str = "Jungle2",
    hard: bool = False,
    curve_index: int = 0,
) -> Mapping[str, Any]:
    """Replay the exact Board phases through one natural fruit spawn.

    The full tick is retained as an RNG/order cross-check, but unrelated
    chain and scoring fields are explicitly outside this report's authority.
    """

    if len(frames) != 2:
        raise PcMemoryTrajectoryError(
            "fruit_scheduler_diff_requires_one_transition"
        )
    before, after = frames
    if (
        after.update != before.update + 1
        or before.native_game_time is None
        or after.native_game_time != before.native_game_time + 1
        or before.board_update_count is None
        or after.board_update_count != before.board_update_count + 1
        or before.fruit_state is None
        or after.fruit_state is None
        or before.global_mtrand is None
        or after.global_mtrand is None
    ):
        raise PcMemoryTrajectoryError(
            "fruit_scheduler_diff_transition_incomplete"
        )
    simulator, provenance = _transplant_midstate(
        frames,
        original_root=Path(original_root).resolve(strict=True),
        hard=hard,
        level_id=level_id,
        curve_index=curve_index,
    )
    full_tick_simulator = copy.deepcopy(simulator)

    # Mirror RevengeSimulator.tick only through Board's fruit scheduler.  The
    # retail ordering is gun, fruit visual, projectiles, then scheduler.
    simulator.last_events = TickEvents()
    simulator.tick_count += 1
    simulator.native_game_time += 1
    simulator._update_gun()
    simulator._update_fruit_visual()
    simulator._update_free_projectiles()
    simulator._update_fruit_scheduler()
    phase_mismatches = _fruit_state_mismatches(
        simulator,
        after.fruit_state,
    )

    source_outputs = _mtrand_outputs_between(
        before.global_mtrand,
        after.global_mtrand,
        maximum_draws=64,
    )
    scheduler_prefix_exact = False
    if source_outputs is not None and len(source_outputs) >= 2:
        prefix = PopCapMTRandom(1)
        prefix.load_state(
            before.global_mtrand.words,
            before.global_mtrand.index,
        )
        prefix_outputs = (prefix.next_u31(), prefix.next_u31())
        scheduler_prefix_exact = bool(
            prefix_outputs == source_outputs[:2]
            and simulator.rng.state == prefix.state
        )

    full_events = full_tick_simulator.tick()
    full_tick_mismatches = _fruit_state_mismatches(
        full_tick_simulator,
        after.fruit_state,
    )
    full_tick_mtrand_exact = _mtrand_matches(full_tick_simulator, after)

    features, proofs = derive_native_mechanism_features(
        frames,
        original_root=Path(original_root).resolve(strict=True),
        level_id=level_id,
        hard=hard,
        curve_index=curve_index,
    )
    scheduler_proof = next(
        (
            proof
            for proof in proofs
            if proof.get("feature") == "fruit_scheduler_spawn"
        ),
        None,
    )
    failure_reasons: list[str] = []
    if features != ("fruit_scheduler_spawn",) or scheduler_proof is None:
        failure_reasons.append("source_spawn_proof_missing_or_ambiguous")
    if phase_mismatches:
        failure_reasons.append("scheduler_phase_fruit_state_mismatch")
    if (
        simulator.last_events.fruit_chance_draws != 1
        or simulator.last_events.fruits_spawned != 1
    ):
        failure_reasons.append("scheduler_phase_event_mismatch")
    if not scheduler_prefix_exact:
        failure_reasons.append("scheduler_mtrand_prefix_mismatch")
    if full_tick_mismatches:
        failure_reasons.append("full_tick_fruit_state_mismatch")
    if (
        full_events.fruit_chance_draws != 1
        or full_events.fruits_spawned != 1
    ):
        failure_reasons.append("full_tick_scheduler_event_mismatch")
    if not full_tick_mtrand_exact:
        failure_reasons.append("full_tick_mtrand_state_mismatch")

    source_before = before.fruit_state
    source_after = after.fruit_state
    assert source_before is not None
    assert source_after is not None
    return {
        "schema": FRUIT_SCHEDULER_DIFF_SCHEMA,
        "version": FRUIT_SCHEDULER_DIFF_VERSION,
        "status": "PASS" if not failure_reasons else "FAIL",
        "failure_reasons": failure_reasons,
        "level_id": level_id,
        "hard": hard,
        "profile_mode": "tutorials_completed",
        "curve_index": curve_index,
        "start_update": before.update,
        "end_update": after.update,
        "compared_tick_count": 1,
        "fruit_runtime_state_restored": provenance.get(
            "fruit_runtime_state_restored"
        ),
        "transition": {
            "source_before_active": source_before.active,
            "source_after_active": source_after.active,
            "source_selected_point_index": (
                source_after.selected_point_index
            ),
            "simulator_selected_point_index": (
                simulator.fruit_active_point_index
            ),
            "source_expiry_time": source_after.expiry_time,
            "simulator_expiry_time": simulator.fruit_expiry_time,
            "scheduler_phase_fruit_state_mismatches": phase_mismatches,
            "full_tick_fruit_state_mismatches": full_tick_mismatches,
            "scheduler_phase_fruit_chance_draws": (
                simulator.last_events.fruit_chance_draws
            ),
            "scheduler_phase_fruits_spawned": (
                simulator.last_events.fruits_spawned
            ),
            "full_tick_fruit_chance_draws": full_events.fruit_chance_draws,
            "full_tick_fruits_spawned": full_events.fruits_spawned,
        },
        "mtrand": {
            "source_total_draw_count": (
                None if source_outputs is None else len(source_outputs)
            ),
            "source_outputs": (
                None if source_outputs is None else list(source_outputs)
            ),
            "scheduler_draw_count": 2,
            "scheduler_prefix_state_exact": scheduler_prefix_exact,
            "full_tick_final_state_exact": full_tick_mtrand_exact,
            "source_final_index": after.global_mtrand.index,
            "full_tick_final_index": full_tick_simulator.rng.index,
        },
        "out_of_scope_full_tick_differences": {
            "authorization": "none",
            "active_count_source": len(_active(after)),
            "active_count_simulator": len(full_tick_simulator.balls),
            "score_source": after.score,
            "score_simulator": full_tick_simulator.score,
        },
        "source_authorized_features": list(features),
        "source_feature_proofs": list(proofs),
    }


__all__ = [
    "FRUIT_SCHEDULER_DIFF_SCHEMA",
    "FRUIT_SCHEDULER_DIFF_VERSION",
    "compare_fruit_scheduler_transition",
]
