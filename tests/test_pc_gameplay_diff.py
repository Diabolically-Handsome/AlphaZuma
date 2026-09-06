from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import numpy as np

from zuma_rl.pc_gameplay_diff import (
    _fruit_state_mismatches,
    _mtrand_state_sha256,
    _mtrand_reconciliation_contract,
    _normalized_qrand_vectors,
    _observed_chain_movement_delta,
    _restore_fruit_state,
    _restore_global_mtrand_state,
    _restore_initial_free_projectiles,
    _restore_shooter_random_state,
)
from zuma_rl.pc_memory_trajectory import (
    PcMemoryTrajectoryError,
    RETAIL_DEBUG_FILL_I32,
    TrajectoryEntity,
    TrajectoryFruitState,
    TrajectoryMTRandState,
    TrajectoryQRandState,
)
from zuma_rl.revenge_core import FruitSpawnCalibration, PopCapMTRandom


def test_unused_mtrand_reconciliation_search_emits_exact_contract() -> None:
    policy, contract = _mtrand_reconciliation_contract(
        (),
        maximum_draws_per_tick=64,
    )

    assert policy == "exact_no_reconciliation"
    assert contract == {
        "enabled": False,
        "classification": "no_reconciliation",
        "state_binding": "sha256_of_624_words_plus_index_le_u32",
        "maximum_draws_per_tick": 64,
        "reconciled_tick_count": 0,
        "total_reconciled_draw_count": 0,
        "rows": [],
    }


def test_used_mtrand_reconciliation_search_remains_proof_gated() -> None:
    row = {"draw_count": 5, "framework_update": 10}

    policy, contract = _mtrand_reconciliation_contract(
        (row,),
        maximum_draws_per_tick=64,
    )

    assert policy == "conditional_external_mtrand_reconciliation_v2"
    assert contract["enabled"] is True
    assert contract["classification"] == (
        "unclassified_until_bound_call_trace_proof"
    )
    assert contract["reconciled_tick_count"] == 1
    assert contract["total_reconciled_draw_count"] == 5
    assert contract["rows"] == [row]


def _free_projectile_row(**changes: object) -> TrajectoryEntity:
    row = TrajectoryEntity(
        ball_id=40,
        color_id=3,
        object_kind="bullet",
        zone="fired",
        index=0,
        curve_distance=0.0,
        position_x=108.0,
        position_y=117.0,
        radius=18.0,
        velocity_x=-2.5,
        velocity_y=7.5,
        merge_progress=0.0,
        merge_speed=0.025,
        fired=False,
        gap_list_sentinel_address=0x1000,
        gap_entry_count=1,
        curve_points=(397, RETAIL_DEBUG_FILL_I32, RETAIL_DEBUG_FILL_I32, RETAIL_DEBUG_FILL_I32),
        gap_entries=((0, 36, 12),),
    )
    return replace(row, **changes)


def test_observed_chain_movement_delta_preserves_reverse_direction() -> None:
    before = SimpleNamespace(
        entities=(
            _free_projectile_row(
                object_kind="ball",
                zone="curve:0:list:05c",
                curve_distance=100.0,
            ),
        )
    )
    after = SimpleNamespace(
        entities=(
            _free_projectile_row(
                object_kind="ball",
                zone="curve:0:list:05c",
                curve_distance=99.0,
            ),
        )
    )

    assert _observed_chain_movement_delta(before, after) == np.float32(-1.0)


def _projectile_restore_simulator() -> SimpleNamespace:
    return SimpleNamespace(
        curve_count=1,
        active_curve_index=0,
        config=SimpleNamespace(ball_radius=18),
        active_num_colors=6,
        balls=[SimpleNamespace(id=12)],
        free_projectiles=[],
    )


def test_initial_free_projectile_restores_exact_native_semantics() -> None:
    simulator = _projectile_restore_simulator()

    _restore_initial_free_projectiles(  # type: ignore[arg-type]
        simulator,
        (_free_projectile_row(),),
    )

    assert len(simulator.free_projectiles) == 1
    projectile = simulator.free_projectiles[0]
    assert projectile.id == 40
    assert projectile.color == 3
    np.testing.assert_array_equal(projectile.position, (108.0, 117.0))
    np.testing.assert_array_equal(projectile.velocity, (-2.5, 7.5))
    assert projectile.just_fired is False
    assert projectile.hit_percent == np.float32(0.0)
    assert projectile.merge_speed == np.float32(0.025)
    assert projectile.curve_point == 397
    assert projectile.curve_points == {0: 397}
    assert projectile.gap_info == [(12, 36)]


def test_initial_free_projectile_debug_fill_becomes_unset_latch() -> None:
    simulator = _projectile_restore_simulator()
    row = _free_projectile_row(
        gap_entry_count=0,
        gap_entries=(),
        curve_points=(RETAIL_DEBUG_FILL_I32,) * 4,
    )

    _restore_initial_free_projectiles(  # type: ignore[arg-type]
        simulator,
        (row,),
    )

    projectile = simulator.free_projectiles[0]
    assert projectile.curve_point == 0
    assert projectile.curve_points == {}


def test_initial_free_projectile_restore_is_two_pass_fail_closed() -> None:
    simulator = _projectile_restore_simulator()
    original = [object()]
    simulator.free_projectiles = original
    incomplete = _free_projectile_row(ball_id=41, velocity_x=None)

    with pytest.raises(
        PcMemoryTrajectoryError,
        match="gameplay_diff_initial_projectile_state_incomplete",
    ):
        _restore_initial_free_projectiles(  # type: ignore[arg-type]
            simulator,
            (_free_projectile_row(), incomplete),
        )

    assert simulator.free_projectiles is original


def test_initial_free_projectile_rejects_unavailable_curve_slot() -> None:
    simulator = _projectile_restore_simulator()
    row = _free_projectile_row(
        curve_points=(397, 109, RETAIL_DEBUG_FILL_I32, RETAIL_DEBUG_FILL_I32),
    )

    with pytest.raises(
        PcMemoryTrajectoryError,
        match="gameplay_diff_initial_projectile_curve_slot_unavailable",
    ):
        _restore_initial_free_projectiles(  # type: ignore[arg-type]
            simulator,
            (row,),
        )


def test_unallocated_qrand_vectors_map_to_zero_filled_constructor_state() -> None:
    state = TrajectoryQRandState(
        update_count=0,
        selected_index=-1,
        weights=(),
        sways=(),
        last_hit=(),
        previous_hit=(),
    )

    vectors = _normalized_qrand_vectors(state, size=6)

    assert vectors == (
        (0.0,) * 6,
        (0.0,) * 6,
        (0,) * 6,
        (0,) * 6,
    )


def test_partial_or_advanced_unallocated_qrand_state_fails_closed() -> None:
    partial = TrajectoryQRandState(
        update_count=0,
        selected_index=-1,
        weights=(),
        sways=(0.0,) * 6,
        last_hit=(),
        previous_hit=(),
    )
    with pytest.raises(
        PcMemoryTrajectoryError,
        match="gameplay_diff_qrand_vector_length_mismatch",
    ):
        _normalized_qrand_vectors(partial, size=6)

    advanced = TrajectoryQRandState(
        update_count=1,
        selected_index=-1,
        weights=(),
        sways=(),
        last_hit=(),
        previous_hit=(),
    )
    with pytest.raises(
        PcMemoryTrajectoryError,
        match="gameplay_diff_uninitialized_qrand_state_invalid",
    ):
        _normalized_qrand_vectors(advanced, size=6)


def test_unallocated_qrand_restores_crt_without_initialized_loader() -> None:
    class DummyChooser:
        size = 6
        update_count = 0
        selected_index = -1
        allowed_support = None
        weights = np.zeros(6, dtype=np.float32)
        sways = np.zeros(6, dtype=np.float32)
        last_hit = np.zeros(6, dtype=np.int64)
        previous_hit = np.zeros(6, dtype=np.int64)

    class DummySimulator:
        _color_chooser = DummyChooser()
        crt_rng = type("DummyCrt", (), {"state": 0})()

        def load_shooter_random_state(self, **kwargs: object) -> None:
            raise AssertionError("initialized QRand loader must not run")

    simulator = DummySimulator()
    state = TrajectoryQRandState(
        update_count=0,
        selected_index=-1,
        weights=(),
        sways=(),
        last_hit=(),
        previous_hit=(),
    )

    _restore_shooter_random_state(
        simulator,  # type: ignore[arg-type]
        state=state,
        crt_state=123456,
    )

    assert simulator.crt_rng.state == 123456


def test_global_mtrand_snapshot_is_restored_exactly() -> None:
    source = PopCapMTRandom(0x12345678)
    for _ in range(37):
        source.next_u31()
    expected = TrajectoryMTRandState(
        index=source.index,
        words=source.words,
    )

    class DummySimulator:
        rng = PopCapMTRandom(0)

        def load_mtrand_state(
            self,
            *,
            words: tuple[int, ...],
            index: int,
        ) -> None:
            self.rng.load_state(words, index)

    simulator = DummySimulator()
    _restore_global_mtrand_state(
        simulator,  # type: ignore[arg-type]
        state=expected,
    )

    assert simulator.rng.state == source.state
    assert simulator.rng.next_u31() == source.next_u31()


def test_global_mtrand_state_digest_binds_words_and_index() -> None:
    source = PopCapMTRandom(0x12345678)
    initial = _mtrand_state_sha256(source.words, source.index)
    source.next_u31()
    advanced = _mtrand_state_sha256(source.words, source.index)

    assert initial.startswith("sha256:")
    assert len(initial) == 71
    assert advanced != initial

    with pytest.raises(
        PcMemoryTrajectoryError,
        match="gameplay_diff_mtrand_state_invalid",
    ):
        _mtrand_state_sha256(source.words[:-1], source.index)


def test_fruit_board_state_restores_and_compares_float32_exactly() -> None:
    class DummySimulator:
        fruit_calibration = FruitSpawnCalibration(
            frequency_ticks=1_000,
            lifetime_ticks=1_000,
            points=((28.0, 105.0), (716.0, 505.0), (602.0, 366.0)),
            unlock_percentages=((10,), (20,), (30,)),
            provenance="test",
        )

    simulator = DummySimulator()
    state = TrajectoryFruitState(
        active_point_pointer=0,
        selected_point_index=-1,
        collecting=False,
        velocity=float(np.float32(0.119999938)),
        max_velocity=0.25,
        acceleration=float(np.float32(0.01)),
        vertical_offset=float(np.float32(-2.4700036)),
        lower_bound=float(np.float32(-1.8626451e-06)),
        upper_bound=float(np.float32(-4.351139e-06)),
        glow_alpha=0,
        glow_step=0,
        alpha=255,
        expiry_time=0,
        cell_index=21,
    )

    _restore_fruit_state(simulator, state)  # type: ignore[arg-type]

    assert _fruit_state_mismatches(  # type: ignore[arg-type]
        simulator,
        state,
    ) == {}
    simulator.fruit_vertical_offset = np.nextafter(
        simulator.fruit_vertical_offset,
        np.float32(np.inf),
        dtype=np.float32,
    )
    assert _fruit_state_mismatches(  # type: ignore[arg-type]
        simulator,
        state,
    ) == {"fruit_vertical_offset": 1}


def test_fruit_mid_collection_restore_fails_closed() -> None:
    class DummySimulator:
        fruit_calibration = FruitSpawnCalibration(
            frequency_ticks=1_000,
            lifetime_ticks=1_000,
            points=((602.0, 366.0),),
            unlock_percentages=((30,),),
            provenance="test",
        )

    state = TrajectoryFruitState(
        active_point_pointer=0x25000030,
        selected_point_index=0,
        collecting=True,
        velocity=0.25,
        max_velocity=0.25,
        acceleration=-0.01,
        vertical_offset=0.0,
        lower_bound=float(np.finfo(np.float32).max),
        upper_bound=float(np.finfo(np.float32).max),
        glow_alpha=0,
        glow_step=12,
        alpha=255,
        expiry_time=2_000,
        cell_index=0,
    )

    with pytest.raises(
        PcMemoryTrajectoryError,
        match="gameplay_diff_initial_fruit_collection_phase_unavailable",
    ):
        _restore_fruit_state(  # type: ignore[arg-type]
            DummySimulator(),
            state,
        )
