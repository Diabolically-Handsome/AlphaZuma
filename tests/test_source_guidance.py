"""Tests for source-guided Deluxe-to-Revenge mechanism decisions."""

from __future__ import annotations

from pathlib import Path

import pytest

from zuma_rl.source_guidance import (
    GAP_SHOT_INSTRUCTION_BLOCKS,
    PENDING_COLOR_INSTRUCTION_BLOCKS,
    POWERUP_EFFECT_INSTRUCTION_BLOCKS,
    ROLLBACK_INSTRUCTION_BLOCKS,
    TERMINAL_INSTRUCTION_BLOCKS,
    SourceGuidanceError,
    verify_circleshoot_source,
    verify_gap_shot_instruction_blocks,
    verify_python_gap_contract,
    verify_python_pending_color_contract,
    verify_python_rollback_contract,
    verify_python_terminal_contract,
    verify_pending_color_instruction_blocks,
    verify_powerup_effect_instruction_blocks,
    verify_python_powerup_effect_contract,
    verify_retail_rollback_static,
    verify_retail_powerup_effects_static,
    verify_retail_terminal_static,
    verify_retail_gap_shot_static,
    verify_retail_pending_color_static,
    verify_rollback_instruction_blocks,
    verify_terminal_instruction_blocks,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PINNED_RUNTIME = Path(
    r"D:\ZumaGolden\tools\direct-runtime\popcapgame1.exe"
)
CIRCLESHOOT_ROOT = PROJECT_ROOT / "external/CircleShootApp"


def test_gap_instruction_blocks_classify_radius_squared_contract() -> None:
    blocks = {
        name: expected
        for name, (_, expected) in GAP_SHOT_INSTRUCTION_BLOCKS.items()
    }

    proof = verify_gap_shot_instruction_blocks(blocks)

    assert proof["sample_step_expression"] == (
        "2 * converted_projectile_radius"
    )
    assert proof["distance_threshold_expression"] == (
        "converted_projectile_radius ** 2"
    )


def test_gap_instruction_blocks_fail_closed_on_threshold_drift() -> None:
    blocks = {
        name: expected
        for name, (_, expected) in GAP_SHOT_INSTRUCTION_BLOCKS.items()
    }
    sample = bytearray(blocks["sample_point_compare"])
    sample[0] ^= 1
    blocks["sample_point_compare"] = bytes(sample)

    with pytest.raises(
        SourceGuidanceError,
        match="runtime_gap_shot_sample_point_compare_bytes_mismatch",
    ):
        verify_gap_shot_instruction_blocks(blocks)


def test_pending_color_blocks_classify_revenge_rng_contract() -> None:
    blocks = {
        name: expected
        for name, (_, expected) in PENDING_COLOR_INSTRUCTION_BLOCKS.items()
    }

    proof = verify_pending_color_instruction_blocks(blocks)

    assert proof["repeat_condition"] == (
        "repeat_roll <= repeat_chance and current_run < max_clump"
    )
    assert proof["visual_frame_draw_order"] == (
        "after_color_selection_before_list_insert"
    )


def test_pending_color_blocks_fail_closed_on_rng_order_drift() -> None:
    blocks = {
        name: expected
        for name, (_, expected) in PENDING_COLOR_INSTRUCTION_BLOCKS.items()
    }
    commit = bytearray(blocks["color_commit_then_visual_frame"])
    commit[-1] ^= 1
    blocks["color_commit_then_visual_frame"] = bytes(commit)

    with pytest.raises(
        SourceGuidanceError,
        match="runtime_pending_color_color_commit_then_visual_frame_bytes_mismatch",
    ):
        verify_pending_color_instruction_blocks(blocks)


def test_rollback_blocks_classify_scaled_directional_contract() -> None:
    blocks = {
        name: expected
        for name, (_, expected) in ROLLBACK_INSTRUCTION_BLOCKS.items()
    }

    proof = verify_rollback_instruction_blocks(blocks)

    assert proof["suck_count_ramp"] == (
        "arithmetic_shift_right_3_then_multiply_curve_reverse_speed"
    )
    assert proof["normal_rollback_direction_flag"] == 1
    assert proof["gap_contact_seed"]["backwards_count"] == 30
    assert proof["entrance_motion_latch"] == {
        "first_ball_moved_backwards": True,
        "first_ball_moved_backwards_offset": 0x1A0,
        "minimum_stop_ticks": 20,
        "stop_time_offset": 0x17C,
    }


def test_rollback_blocks_fail_closed_on_scale_drift() -> None:
    blocks = {
        name: expected
        for name, (_, expected) in ROLLBACK_INSTRUCTION_BLOCKS.items()
    }
    ramp = bytearray(blocks["suck_direction_and_speed"])
    ramp[-1] ^= 1
    blocks["suck_direction_and_speed"] = bytes(ramp)

    with pytest.raises(
        SourceGuidanceError,
        match="runtime_rollback_suck_direction_and_speed_bytes_mismatch",
    ):
        verify_rollback_instruction_blocks(blocks)


def test_powerup_effect_blocks_classify_target_effect_core() -> None:
    blocks = {
        name: expected
        for name, (_, expected) in POWERUP_EFFECT_INSTRUCTION_BLOCKS.items()
    }

    proof = verify_powerup_effect_instruction_blocks(blocks)

    assert proof["effective_type_priority"] == [
        "primary",
        "live_previous",
        "secondary",
    ]
    assert proof["per_explosion_waypoint"]["write_timing"] == (
        "after_exploding_guard_before_powerup_dispatch"
    )
    assert proof["proximity_bomb"]["physical_collision_pad"] == 56
    assert proof["reverse"]["replacement_ticks"] == 300


def test_powerup_effect_blocks_fail_closed_on_waypoint_drift() -> None:
    blocks = {
        name: expected
        for name, (_, expected) in POWERUP_EFFECT_INSTRUCTION_BLOCKS.items()
    }
    waypoint = bytearray(blocks["explode_guard_and_waypoint"])
    waypoint[-1] ^= 1
    blocks["explode_guard_and_waypoint"] = bytes(waypoint)

    with pytest.raises(
        SourceGuidanceError,
        match=(
            "runtime_powerup_effect_explode_guard_and_waypoint_"
            "bytes_mismatch"
        ),
    ):
        verify_powerup_effect_instruction_blocks(blocks)


def test_terminal_blocks_classify_board_quantifiers() -> None:
    blocks = {
        name: expected
        for name, (_, expected) in TERMINAL_INSTRUCTION_BLOCKS.items()
    }

    proof = verify_terminal_instruction_blocks(blocks)

    assert proof["board_win_quantifier"] == (
        "all_curves_satisfy_curve_win_predicate"
    )
    assert proof["board_loss_quantifier"] == (
        "any_curve_satisfies_curve_loss_predicate"
    )


def test_terminal_blocks_fail_closed_on_win_predicate_drift() -> None:
    blocks = {
        name: expected
        for name, (_, expected) in TERMINAL_INSTRUCTION_BLOCKS.items()
    }
    winning = bytearray(blocks["board_win_inline_loop"])
    winning[0] ^= 1
    blocks["board_win_inline_loop"] = bytes(winning)

    with pytest.raises(
        SourceGuidanceError,
        match="runtime_terminal_board_win_inline_loop_bytes_mismatch",
    ):
        verify_terminal_instruction_blocks(blocks)


def test_python_gap_implementations_follow_retail_contract() -> None:
    simulator = verify_python_gap_contract(
        PROJECT_ROOT / "src/zuma_rl/revenge_core.py",
        function_name="_check_gap_shot",
        implementation="simulator",
    )
    mechanism_audit = verify_python_gap_contract(
        PROJECT_ROOT / "src/zuma_rl/pc_mechanism_audit.py",
        function_name="_simulate_gap_checks",
        implementation="mechanism_audit",
    )

    assert simulator["verdict"] == "matches_retail_revenge"
    assert mechanism_audit["verdict"] == "matches_retail_revenge"


def test_python_pending_color_follows_retail_contract() -> None:
    proof = verify_python_pending_color_contract(
        PROJECT_ROOT / "src/zuma_rl/revenge_core.py"
    )

    assert proof["verdict"] == "matches_retail_revenge"
    assert proof["visual_frame_draw_order"] == (
        "after_color_selection_before_list_insert"
    )


def test_python_rollback_core_follows_retail_contract() -> None:
    proof = verify_python_rollback_contract(
        PROJECT_ROOT / "src/zuma_rl/revenge_core.py"
    )

    assert proof["verdict"] == "matches_retail_revenge_rollback_core"
    assert proof["normal_suck_direction"] == "explicit_backward"
    assert proof["curve_update_order"][0] == "_update_sucking_balls"


def test_python_powerup_effect_core_follows_retail_contract() -> None:
    proof = verify_python_powerup_effect_contract(
        PROJECT_ROOT / "src/zuma_rl/revenge_core.py"
    )

    assert proof["verdict"] == (
        "matches_retail_revenge_powerup_effect_core"
    )
    assert proof["last_waypoint_write"] == (
        "every_new_explosion_before_powerup_dispatch"
    )
    assert proof["config"]["verified_defaults"][
        "proximity_bomb_collision_pad"
    ] == 56


def test_python_terminal_predicates_are_scoped_to_nonboss_levels() -> None:
    proof = verify_python_terminal_contract(
        PROJECT_ROOT / "src/zuma_rl/revenge_core.py"
    )

    assert proof["curve_win_scope"] == "normal_nonboss_levels"
    assert proof["special_or_boss_actor_gate_modeled"] is False
    assert proof["board_terminal_timing"] == "natural_trajectory_bound"


@pytest.mark.skipif(
    not PINNED_RUNTIME.is_file(),
    reason="pinned local retail runtime is unavailable",
)
def test_pinned_retail_runtime_gap_shot_static_proof() -> None:
    proof = verify_retail_gap_shot_static(PINNED_RUNTIME.read_bytes())

    assert proof["semantics"]["verdict"] == (
        "radius_squared_threshold_with_diameter_sample_step"
    )
    assert proof["direct_call_xrefs"] == [0x004180CE]


@pytest.mark.skipif(
    not PINNED_RUNTIME.is_file(),
    reason="pinned local retail runtime is unavailable",
)
def test_pinned_retail_runtime_pending_color_static_proof() -> None:
    proof = verify_retail_pending_color_static(PINNED_RUNTIME.read_bytes())

    assert proof["semantics"]["verdict"] == (
        "max_clump_guarded_rejection_then_visual_frame"
    )
    assert proof["curve_manager_random_mod_vtable"][
        "target_virtual_address"
    ] == 0x004B4B70


@pytest.mark.skipif(
    not PINNED_RUNTIME.is_file(),
    reason="pinned local retail runtime is unavailable",
)
def test_pinned_retail_runtime_rollback_static_proof() -> None:
    proof = verify_retail_rollback_static(PINNED_RUNTIME.read_bytes())

    assert proof["semantics"]["verdict"] == (
        "direction_aware_scaled_suction_and_contact_rollback"
    )
    assert proof["functions"]["update_sucking_balls"][
        "direct_call_xrefs"
    ] == [0x0045DE6C]


@pytest.mark.skipif(
    not PINNED_RUNTIME.is_file(),
    reason="pinned local retail runtime is unavailable",
)
def test_pinned_retail_runtime_powerup_effect_static_proof() -> None:
    proof = verify_retail_powerup_effects_static(PINNED_RUNTIME.read_bytes())

    assert proof["semantics"]["verdict"] == (
        "per_explosion_waypoint_and_target_powerup_effects_recovered"
    )
    assert proof["functions"]["curve_activate_power"][
        "direct_call_xrefs"
    ] == [0x00418807]
    assert proof["call_routes"]["bomb_recursive_explosion"][
        "target_virtual_address"
    ] == 0x00457680
    assert proof["scope"]["spawn_scheduler_recovered"] is False


@pytest.mark.skipif(
    not PINNED_RUNTIME.is_file(),
    reason="pinned local retail runtime is unavailable",
)
def test_pinned_retail_runtime_terminal_static_proof() -> None:
    proof = verify_retail_terminal_static(PINNED_RUNTIME.read_bytes())

    assert proof["semantics"]["verdict"] == (
        "curve_terminal_predicates_and_board_quantifiers_recovered"
    )
    assert proof["functions"]["is_losing"]["direct_call_xrefs"] == [
        0x0041B010
    ]
    assert proof["functions"]["is_winning"]["direct_call_xrefs"] == []


@pytest.mark.skipif(
    not (CIRCLESHOOT_ROOT / ".git").is_dir(),
    reason="optional CircleShootApp reference checkout is unavailable",
)
def test_pinned_circleshoot_snapshot_is_clean_and_detached() -> None:
    proof = verify_circleshoot_source(CIRCLESHOOT_ROOT)

    assert proof["detached_head"] is True
    assert proof["clean_worktree"] is True
    assert proof["gap_shot"]["verdict"] == (
        "diameter_squared_threshold_with_diameter_sample_step"
    )
    assert proof["rollback"]["verdict"] == (
        "shared_topology_without_revenge_direction_or_scale"
    )
    assert proof["powerup_effects"]["verdict"] == (
        "shared_effect_topology_with_deluxe_bomb_pad_45"
    )
    assert proof["powerup_effects"]["last_cleared_waypoint_timing"] == (
        "every_new_clear_before_powerup_dispatch"
    )
    assert proof["terminal"]["verdict"] == (
        "ancestor_predicates_without_revenge_special_actor_gate"
    )
