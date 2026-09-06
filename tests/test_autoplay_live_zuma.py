from __future__ import annotations

import math
import struct

import pytest

import tools.autoplay_live_zuma as autoplay_live_zuma
from tools.autoplay_live_zuma import (
    _activate_window_with_api,
    _forward_waypoint_distance,
    _read_ball,
    _stabilize_window_foreground_with_api,
    LiveBoardUnavailable,
    choose_shot,
    fruit_bomb_observation,
    is_natural_loss_board_identity_replacement_state,
    is_natural_loss_board_replacement_transition,
    is_natural_loss_mature_anchor,
    is_natural_loss_restart_clock_reset,
    is_natural_loss_restart_transition,
    is_natural_loss_state,
    is_natural_win_state,
    logical_viewport,
)


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class _FakeForegroundApi:
    def __init__(self, *, foreground: int, activation: str) -> None:
        self.foreground_handle = foreground
        self.activation = activation
        self.restored: list[int] = []
        self.brought: list[int] = []
        self.requested: list[int] = []
        self.attachments: list[tuple[int, int, bool]] = []
        self.attached: set[int] = set()

    def foreground(self) -> int:
        return self.foreground_handle

    def restore(self, window_handle: int) -> None:
        self.restored.append(window_handle)

    def bring_to_top(self, window_handle: int) -> None:
        self.brought.append(window_handle)

    def request_foreground(self, window_handle: int) -> None:
        self.requested.append(window_handle)
        if self.activation == "ordinary" or (
            self.activation == "fallback" and self.attached
        ):
            self.foreground_handle = window_handle

    def current_thread_id(self) -> int:
        return 100

    def window_thread_id(self, window_handle: int) -> int:
        return 200 if window_handle == 0x1111 else 300

    def attach_thread_input(
        self,
        source_thread_id: int,
        target_thread_id: int,
        attach: bool,
    ) -> bool:
        self.attachments.append(
            (source_thread_id, target_thread_id, attach)
        )
        if attach:
            self.attached.add(target_thread_id)
        else:
            self.attached.discard(target_thread_id)
        return True


class _FlappingForegroundApi(_FakeForegroundApi):
    def __init__(
        self,
        clock: _FakeClock,
        *,
        losses_at: list[float],
    ) -> None:
        super().__init__(foreground=0x2222, activation="ordinary")
        self.clock = clock
        self.losses_at = losses_at
        self.loss_index = 0

    def foreground(self) -> int:
        if (
            self.loss_index < len(self.losses_at)
            and self.clock.now >= self.losses_at[self.loss_index]
        ):
            self.loss_index += 1
            self.foreground_handle = 0x1111
        return self.foreground_handle


def test_foreground_activation_fast_path_is_verified() -> None:
    clock = _FakeClock()
    api = _FakeForegroundApi(foreground=0x2222, activation="never")
    receipt = _activate_window_with_api(
        0x2222,
        api,
        timeout=2.0,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    assert receipt["verified"] is True
    assert receipt["activation_attempts"] == 0
    assert receipt["attach_fallback_used"] is False
    assert not api.requested


def test_foreground_activation_ordinary_path_is_verified() -> None:
    clock = _FakeClock()
    api = _FakeForegroundApi(foreground=0x1111, activation="ordinary")
    receipt = _activate_window_with_api(
        0x2222,
        api,
        timeout=2.0,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    assert receipt["verified"] is True
    assert receipt["activation_attempts"] == 1
    assert receipt["attach_fallback_used"] is False
    assert api.restored == [0x2222]
    assert not api.attachments


def test_foreground_activation_fallback_detaches_every_queue() -> None:
    clock = _FakeClock()
    api = _FakeForegroundApi(foreground=0x1111, activation="fallback")
    receipt = _activate_window_with_api(
        0x2222,
        api,
        timeout=2.0,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    assert receipt["verified"] is True
    assert receipt["attach_fallback_used"] is True
    assert receipt["attached_thread_ids"] == [200, 300]
    assert api.attachments[-2:] == [(100, 300, False), (100, 200, False)]
    assert not api.attached


def test_foreground_activation_timeout_fails_closed_and_detaches() -> None:
    clock = _FakeClock()
    api = _FakeForegroundApi(foreground=0x1111, activation="never")
    with pytest.raises(RuntimeError, match="before SendInput"):
        _activate_window_with_api(
            0x2222,
            api,
            timeout=0.08,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )
    assert api.attachments[-2:] == [(100, 300, False), (100, 200, False)]
    assert not api.attached


def test_foreground_stability_accumulates_continuous_interval() -> None:
    clock = _FakeClock()
    api = _FakeForegroundApi(foreground=0x2222, activation="ordinary")
    receipt = _stabilize_window_foreground_with_api(
        0x2222,
        api,
        continuous_seconds=0.15,
        timeout=2.0,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    assert receipt["verified"] is True
    assert receipt["continuous_observed_seconds"] >= 0.15
    assert receipt["foreground_loss_count"] == 0
    assert receipt["reacquisition_count"] == 0


def test_foreground_stability_resets_after_multiple_losses() -> None:
    clock = _FakeClock()
    api = _FlappingForegroundApi(clock, losses_at=[0.03, 0.08])
    receipt = _stabilize_window_foreground_with_api(
        0x2222,
        api,
        continuous_seconds=0.07,
        timeout=1.0,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    assert receipt["verified"] is True
    assert receipt["foreground_loss_count"] == 2
    assert receipt["reacquisition_count"] == 2
    assert receipt["continuous_observed_seconds"] >= 0.07


def test_foreground_stability_timeout_fails_before_input() -> None:
    clock = _FakeClock()
    api = _FakeForegroundApi(foreground=0x1111, activation="never")
    with pytest.raises(RuntimeError, match="before SendInput"):
        _stabilize_window_foreground_with_api(
            0x2222,
            api,
            continuous_seconds=0.05,
            timeout=0.12,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )


def _record(
    index: int,
    *,
    color: int,
    x: float,
    y: float,
    powerup_primary_type: int = 14,
    curve_distance: float = 1_100.0,
    powerup_lifetime_ticks: int = 2_000,
    powerup_transition_ticks: int = 0,
) -> dict[str, object]:
    return {
        "index": index,
        "ball": {
            "ball_id": 100 + index,
            "color_id": color,
            "position_x": x,
            "position_y": y,
            "powerup_primary_type": powerup_primary_type,
            "curve_distance": curve_distance,
            "powerup_lifetime_ticks": powerup_lifetime_ticks,
            "powerup_transition_ticks": powerup_transition_ticks,
        },
    }


def _state(
    colors: list[int],
    *,
    current: int,
    next_color: int,
) -> dict[str, object]:
    positions = (
        (520.0, 300.0),
        (400.0, 420.0),
        (280.0, 300.0),
        (400.0, 180.0),
    )
    return {
        "board_address": 0x12340000,
        "native_game_time": 100,
        "score": 8000,
        "fired_ball_count": 0,
        "inserting_ball_count": 0,
        "active_records": [
            _record(
                index,
                color=color,
                x=positions[index][0],
                y=positions[index][1],
            )
            for index, color in enumerate(colors)
        ],
        "current": {
            "ball_id": 1,
            "color_id": current,
            "position_x": 400.0,
            "position_y": 300.0,
        },
        "next": {
            "ball_id": 2,
            "color_id": next_color,
            "position_x": 400.0,
            "position_y": 300.0,
        },
    }


def test_logical_viewport_preserves_native_window() -> None:
    viewport = logical_viewport(1520, 591, 800, 600)
    assert viewport.scale == pytest.approx(1.0)
    assert viewport.screen_point(400.0, 300.0) == (1920, 891)


def test_logical_viewport_maps_centered_4_by_3_fullscreen() -> None:
    viewport = logical_viewport(0, 0, 3840, 2160)
    assert viewport.scale == pytest.approx(3.6)
    assert viewport.screen_left == pytest.approx(480.0)
    assert viewport.screen_point(0.0, 0.0) == (480, 0)
    assert viewport.screen_point(400.0, 300.0) == (1920, 1080)


def test_choose_shot_completes_current_pair() -> None:
    recommendation = choose_shot(
        _state([1, 1, 3], current=1, next_color=3)
    )
    assert recommendation["action"] == "fire"
    assert recommendation["reason"] == "complete_match"
    assert recommendation["run_length"] == 2


def test_choose_shot_collect_policy_prioritizes_active_fruit() -> None:
    state = _state([1, 1, 3], current=1, next_color=3)
    state["fruit"] = {
        "active": True,
        "collecting": False,
        "selected_point_index": 1,
        "target_x": 742.0,
        "target_y": 531.5,
    }
    recommendation = choose_shot(state, fruit_policy="collect")
    assert recommendation == {
        "action": "fire",
        "reason": "collect_fruit",
        "target_x": 742.0,
        "target_y": 531.5,
        "selected_point_index": 1,
    }


def test_choose_shot_default_policy_preserves_chain_targeting() -> None:
    state = _state([1, 1, 3], current=1, next_color=3)
    state["fruit"] = {
        "active": True,
        "collecting": False,
        "selected_point_index": 1,
        "target_x": 742.0,
        "target_y": 531.5,
    }
    recommendation = choose_shot(state)
    assert recommendation["reason"] == "complete_match"


def test_choose_shot_bomb_policy_targets_current_qualifying_run() -> None:
    state = _state([1, 1, 3], current=1, next_color=3)
    state["active_records"][0]["ball"]["powerup_primary_type"] = 0
    state["fruit"] = {
        "active": True,
        "collecting": False,
        "selected_point_index": 2,
        "point_x": 500,
        "point_y": 300,
    }
    recommendation = choose_shot(state, fruit_policy="bomb")
    assert recommendation["action"] == "fire"
    assert recommendation["reason"] == "trigger_fruit_proximity_bomb"
    assert recommendation["run_length"] == 2
    assert recommendation["proximity_bomb_chain_index"] == 0
    assert recommendation["proximity_bomb_ball_id"] == 100
    assert recommendation["fruit_distance_squared"] == pytest.approx(400.0)
    assert recommendation["fruit_collision_radius_squared"] == pytest.approx(
        108.0**2
    )


def test_choose_shot_bomb_policy_swaps_for_next_qualifying_run() -> None:
    state = _state([2, 2, 3], current=3, next_color=2)
    state["active_records"][0]["ball"]["powerup_primary_type"] = 0
    state["fruit"] = {
        "active": True,
        "collecting": False,
        "selected_point_index": 2,
        "point_x": 500,
        "point_y": 300,
    }
    recommendation = choose_shot(state, fruit_policy="bomb")
    assert recommendation["action"] == "swap_then_fire"
    assert recommendation["reason"] == "trigger_fruit_proximity_bomb"
    assert recommendation["run_length"] == 2


def test_choose_shot_bomb_policy_enforces_strict_radius_boundary() -> None:
    state = _state([1, 1, 3], current=1, next_color=3)
    state["active_records"][0]["ball"].update(
        {"powerup_primary_type": 0, "curve_distance": 2_908.0}
    )
    state["fruit"] = {
        "active": True,
        "collecting": False,
        "selected_point_index": 2,
        "point_x": 412,
        "point_y": 300,
        "target_x": 438.0,
        "target_y": 326.0,
        "expiry_time": 5_000,
    }
    recommendation = choose_shot(state, fruit_policy="bomb")
    assert recommendation["reason"] == "preserve_fruit_proximity_bomb"
    assert recommendation["reason"] != "trigger_fruit_proximity_bomb"


def test_forward_waypoint_distance_matches_frozen_jungle2_examples() -> None:
    intervals = ((1_053.0, 1_259.0),)
    assert _forward_waypoint_distance(417.0, intervals) == 636.0
    assert _forward_waypoint_distance(800.0, intervals) == 253.0
    assert _forward_waypoint_distance(1_100.0, intervals) == 0.0
    assert math.isinf(_forward_waypoint_distance(1_300.0, intervals))


def _point_one_alignment_state(*, curve_distance: float) -> dict[str, object]:
    state = _state([1, 2, 3], current=1, next_color=4)
    state["active_records"][0]["ball"].update(
        {
            "powerup_primary_type": 0,
            "curve_distance": curve_distance,
            "powerup_lifetime_ticks": 2_000,
            "powerup_transition_ticks": 0,
        }
    )
    state["fruit"] = {
        "active": True,
        "collecting": False,
        "selected_point_index": 1,
        "point_x": 716,
        "point_y": 505,
        "target_x": 742.0,
        "target_y": 531.0,
        "expiry_time": 800,
    }
    return state


def test_choose_shot_bomb_alignment_defers_c179a4_like_cycle() -> None:
    recommendation = choose_shot(
        _point_one_alignment_state(curve_distance=417.0),
        fruit_policy="bomb",
    )
    assert recommendation["action"] == "fire"
    assert recommendation["reason"] == "defer_fruit_for_bomb_alignment"
    assert (recommendation["target_x"], recommendation["target_y"]) == (
        742.0,
        531.0,
    )
    assert recommendation["fruit_remaining_ticks"] == 700
    reachability = recommendation["bomb_reachability"][0]
    assert reachability["required_forward_waypoints"] == 636.0
    assert reachability["forward_waypoint_budget"] == 350.0
    assert reachability["reachable_current_cycle"] is False


@pytest.mark.parametrize("curve_distance", (800.0, 1_100.0))
def test_choose_shot_bomb_alignment_preserves_reachable_cycle(
    curve_distance: float,
) -> None:
    recommendation = choose_shot(
        _point_one_alignment_state(curve_distance=curve_distance),
        fruit_policy="bomb",
    )
    assert recommendation["reason"] == "preserve_fruit_proximity_bomb"


def test_choose_shot_bomb_alignment_defers_after_point_interval() -> None:
    recommendation = choose_shot(
        _point_one_alignment_state(curve_distance=1_300.0),
        fruit_policy="bomb",
    )
    assert recommendation["reason"] == "defer_fruit_for_bomb_alignment"
    reachability = recommendation["bomb_reachability"][0]
    assert math.isinf(reachability["required_forward_waypoints"])
    assert reachability["reachable_current_cycle"] is False


def test_choose_shot_strict_overlap_precedes_unreachable_alignment() -> None:
    state = _point_one_alignment_state(curve_distance=1_300.0)
    state["active_records"][0]["ball"].update(
        {"position_x": 716.0, "position_y": 505.0}
    )
    recommendation = choose_shot(state, fruit_policy="bomb")
    assert recommendation["reason"] == "trigger_fruit_proximity_bomb"


@pytest.mark.parametrize(
    ("active", "collecting"),
    ((False, False), (True, True)),
)
def test_choose_shot_bomb_policy_preserves_bomb_for_ineligible_fruit(
    active: bool,
    collecting: bool,
) -> None:
    state = _state([1, 1, 3], current=1, next_color=3)
    state["active_records"][0]["ball"]["powerup_primary_type"] = 0
    state["fruit"] = {
        "active": active,
        "collecting": collecting,
    }
    recommendation = choose_shot(state, fruit_policy="bomb")
    assert recommendation["action"] == "swap_then_fire"
    assert recommendation["reason"] == "preserve_fruit_proximity_bomb"
    assert recommendation["preservation_strategy"] == (
        "bomb_free_candidate_run"
    )


def test_choose_shot_bomb_policy_preserves_legacy_without_bomb() -> None:
    state = _state([1, 1, 3], current=1, next_color=3)
    state["fruit"] = {
        "active": True,
        "collecting": False,
        "selected_point_index": 2,
        "point_x": 500,
        "point_y": 300,
    }
    recommendation = choose_shot(state, fruit_policy="bomb")
    assert recommendation["action"] == "fire"
    assert recommendation["reason"] == "complete_match"


def test_choose_shot_bomb_policy_uses_safe_same_colour_run() -> None:
    state = _state([1, 2, 1], current=1, next_color=4)
    state["active_records"][0]["ball"]["powerup_primary_type"] = 0
    state["fruit"] = {"active": False, "collecting": False}
    recommendation = choose_shot(state, fruit_policy="bomb")
    assert recommendation["action"] == "fire"
    assert recommendation["reason"] == "preserve_fruit_proximity_bomb"
    assert recommendation["preservation_strategy"] == (
        "bomb_free_candidate_run"
    )
    assert recommendation["target_index"] == 2
    assert recommendation["preserved_bomb_chain_indices"] == (0,)


def test_choose_shot_bomb_policy_uses_safe_next_chamber_run() -> None:
    state = _state([1, 2, 3], current=1, next_color=3)
    state["active_records"][0]["ball"]["powerup_primary_type"] = 0
    state["fruit"] = {"active": False, "collecting": False}
    recommendation = choose_shot(state, fruit_policy="bomb")
    assert recommendation["action"] == "swap_then_fire"
    assert recommendation["reason"] == "preserve_fruit_proximity_bomb"
    assert recommendation["target_index"] == 2


def test_choose_shot_bomb_policy_discards_on_clear_perimeter_ray() -> None:
    state = _state([1, 2, 3], current=1, next_color=4)
    state["active_records"][0]["ball"]["powerup_primary_type"] = 0
    state["fruit"] = {"active": False, "collecting": False}
    recommendation = choose_shot(state, fruit_policy="bomb")
    assert recommendation["action"] == "fire"
    assert recommendation["reason"] == "preserve_fruit_proximity_bomb"
    assert recommendation["preservation_strategy"] == (
        "clear_perimeter_discard"
    )
    assert (recommendation["target_x"], recommendation["target_y"]) == (
        0.0,
        0.0,
    )


def test_choose_shot_bomb_discard_avoids_active_fruit_safety_radius() -> None:
    state = _state([1, 2, 3], current=1, next_color=4)
    state["active_records"][0]["ball"]["powerup_primary_type"] = 0
    state["fruit"] = {
        "active": True,
        "collecting": True,
        "selected_point_index": 0,
        "point_x": 0,
        "point_y": 0,
        "target_x": 200.0,
        "target_y": 150.0,
    }
    recommendation = choose_shot(state, fruit_policy="bomb")
    assert recommendation["preservation_strategy"] == (
        "clear_perimeter_discard"
    )
    assert (recommendation["target_x"], recommendation["target_y"]) == (
        799.0,
        0.0,
    )


def test_fruit_bomb_observation_returns_none_when_both_are_absent() -> None:
    state = _state([1, 2, 3], current=1, next_color=2)
    state["fruit"] = {"active": False, "collecting": False}
    assert fruit_bomb_observation(state) is None


def test_fruit_bomb_observation_captures_fruit_only() -> None:
    state = _state([1, 2, 3], current=1, next_color=2)
    state["fruit"] = {
        "active": True,
        "collecting": False,
        "selected_point_index": 2,
        "point_x": 500,
        "point_y": 300,
        "target_x": 526.0,
        "target_y": 326.0,
        "vertical_offset": 0.0,
        "expiry_time": 800,
    }
    observation = fruit_bomb_observation(state)
    assert observation is not None
    assert observation["active_bomb_count"] == 0
    assert observation["fruit_bomb_temporal_overlap"] is False
    assert observation["minimum_static_point_distance_squared"] is None
    assert observation["fruit"]["expiry_time"] == 800


def test_fruit_bomb_observation_captures_bomb_only() -> None:
    state = _state([1, 2, 3], current=1, next_color=2)
    state["active_records"][0]["ball"]["powerup_primary_type"] = 0
    state["fruit"] = {"active": False, "collecting": False}
    observation = fruit_bomb_observation(state)
    assert observation is not None
    assert observation["active_bomb_count"] == 1
    assert observation["fruit_bomb_temporal_overlap"] is False
    assert observation["bombs"][0]["static_point_distance_squared"] is None
    assert observation["bombs"][0]["curve_distance"] == 1_100.0
    assert observation["bombs"][0]["powerup_lifetime_ticks"] == 2_000
    assert observation["bombs"][0]["powerup_transition_ticks"] == 0


def test_fruit_bomb_observation_captures_exact_overlap_distance() -> None:
    state = _state([1, 2, 3], current=1, next_color=2)
    state["active_records"][0]["ball"]["powerup_primary_type"] = 0
    state["fruit"] = {
        "active": True,
        "collecting": False,
        "selected_point_index": 2,
        "point_x": 500,
        "point_y": 300,
        "target_x": 526.0,
        "target_y": 326.0,
        "vertical_offset": 0.0,
        "expiry_time": 800,
    }
    observation = fruit_bomb_observation(state)
    assert observation is not None
    assert observation["fruit_bomb_temporal_overlap"] is True
    assert observation["minimum_static_point_distance_squared"] == (
        pytest.approx(400.0)
    )
    assert observation["strict_overlap_bomb_count"] == 1
    assert observation["bombs"][0]["strictly_within_108"] is True


def test_fruit_bomb_observation_keeps_exact_radius_outside() -> None:
    state = _state([1, 2, 3], current=1, next_color=2)
    state["active_records"][0]["ball"]["powerup_primary_type"] = 0
    state["fruit"] = {
        "active": True,
        "collecting": False,
        "selected_point_index": 2,
        "point_x": 412,
        "point_y": 300,
        "target_x": 438.0,
        "target_y": 326.0,
        "vertical_offset": 0.0,
        "expiry_time": 800,
    }
    observation = fruit_bomb_observation(state)
    assert observation is not None
    assert observation["minimum_static_point_distance_squared"] == (
        pytest.approx(108.0**2)
    )
    assert observation["strict_overlap_bomb_count"] == 0
    assert observation["bombs"][0]["strictly_within_108"] is False


def test_fruit_bomb_observation_rejects_invalid_bomb_timing() -> None:
    state = _state([1, 2, 3], current=1, next_color=2)
    state["active_records"][0]["ball"].update(
        {
            "powerup_primary_type": 0,
            "powerup_lifetime_ticks": -1,
        }
    )
    state["fruit"] = {"active": False, "collecting": False}
    with pytest.raises(ValueError, match="active bomb state"):
        fruit_bomb_observation(state)


def test_read_ball_decodes_powerup_timing_fields(monkeypatch) -> None:
    raw = bytearray(autoplay_live_zuma.BALL_OBJECT_SIZE)
    struct.pack_into("<I", raw, 0, autoplay_live_zuma.BALL_VTABLE)
    struct.pack_into("<i", raw, 0xF8, 1_234)
    struct.pack_into("<i", raw, 0xFC, 56)
    monkeypatch.setattr(
        autoplay_live_zuma,
        "read_process_bytes",
        lambda handle, address, size: bytes(raw[:size]),
    )
    monkeypatch.setattr(
        autoplay_live_zuma,
        "_decode_ball",
        lambda payload, expected_vtable: {"curve_distance": 417.0},
    )
    decoded = _read_ball(1, 0x2000)
    assert decoded["curve_distance"] == 417.0
    assert decoded["powerup_lifetime_ticks"] == 1_234
    assert decoded["powerup_transition_ticks"] == 56


def test_read_ball_rejects_negative_powerup_timing(monkeypatch) -> None:
    raw = bytearray(autoplay_live_zuma.BALL_OBJECT_SIZE)
    struct.pack_into("<I", raw, 0, autoplay_live_zuma.BALL_VTABLE)
    struct.pack_into("<i", raw, 0xF8, -1)
    struct.pack_into("<i", raw, 0xFC, 0)
    monkeypatch.setattr(
        autoplay_live_zuma,
        "read_process_bytes",
        lambda handle, address, size: bytes(raw[:size]),
    )
    monkeypatch.setattr(
        autoplay_live_zuma,
        "_decode_ball",
        lambda payload, expected_vtable: {"curve_distance": 417.0},
    )
    with pytest.raises(LiveBoardUnavailable, match="timing is invalid"):
        _read_ball(1, 0x2000)


def test_choose_shot_rejects_invalid_fruit_policy() -> None:
    with pytest.raises(ValueError, match="fruit policy"):
        choose_shot(
            _state([1, 1, 3], current=1, next_color=3),
            fruit_policy="guess",
        )


def test_choose_shot_swaps_for_only_immediate_match() -> None:
    recommendation = choose_shot(
        _state([2, 2, 3], current=3, next_color=2)
    )
    assert recommendation["action"] == "swap_then_fire"
    assert recommendation["reason"] == "complete_match"


def test_choose_shot_builds_pair_when_no_match_exists() -> None:
    recommendation = choose_shot(
        _state([1, 2, 3], current=2, next_color=3)
    )
    assert recommendation["action"] == "fire"
    assert recommendation["reason"] == "build_pair"


def test_choose_shot_waits_on_empty_chain() -> None:
    state = _state([], current=1, next_color=2)
    assert choose_shot(state) == {
        "action": "wait",
        "reason": "active_chain_empty",
    }


def test_choose_shot_rejects_off_canvas_target() -> None:
    state = _state([1], current=1, next_color=2)
    state["active_records"][0]["ball"]["position_x"] = -20.0
    assert choose_shot(state) == {
        "action": "wait",
        "reason": "no_unobstructed_same_colour",
    }


def test_terminal_state_requires_target_empty_and_no_loss() -> None:
    state = {
        "runtime_active": False,
        "curve_plan_exhausted": True,
        "loss_counter": 0,
        "score": 9700,
        "score_target": 9650,
        "active_ball_count": 0,
        "inserting_ball_count": 0,
        "pending_ball_count": 0,
        "fired_ball_count": 0,
    }
    assert is_natural_win_state(state) is True
    state["active_ball_count"] = 1
    assert is_natural_win_state(state) is False
    state["active_ball_count"] = 0
    state["loss_counter"] = 1
    assert is_natural_win_state(state) is False
    state["loss_counter"] = 0
    state["score_target"] = 0
    assert is_natural_win_state(state) is False
    state["score_target"] = 9650
    state["curve_plan_exhausted"] = False
    assert is_natural_win_state(state) is False


def test_natural_loss_requires_stable_below_target_game_over_board() -> None:
    state = {
        "runtime_active": False,
        "curve_plan_exhausted": True,
        "loss_counter": -357,
        "native_game_time": 357,
        "score": 8000,
        "score_target": 9650,
        "active_ball_count": 0,
        "inserting_ball_count": 0,
        "pending_ball_count": 1,
        "fired_ball_count": 0,
    }
    assert is_natural_loss_state(state) is True
    state["score"] = 9650
    assert is_natural_loss_state(state) is False
    state["score"] = 8000
    state["pending_ball_count"] = 0
    assert is_natural_loss_state(state) is False
    state["pending_ball_count"] = 1
    state["curve_plan_exhausted"] = False
    assert is_natural_loss_state(state) is False
    state["curve_plan_exhausted"] = True
    state["score_target"] = 0
    assert is_natural_loss_state(state) is False


def _restart_state(
    *,
    native_game_time: int,
    score: int = 8000,
    score_target: int = 9650,
    runtime_active: bool,
    active: int,
    pending: int,
    board_address: int = 0x100000,
) -> dict[str, object]:
    return {
        "board_address": board_address,
        "native_game_time": native_game_time,
        "score": score,
        "score_target": score_target,
        "runtime_active": runtime_active,
        "active_ball_count": active,
        "inserting_ball_count": 0,
        "pending_ball_count": pending,
        "fired_ball_count": 0,
        "curve_plan_exhausted": True,
    }


def test_natural_loss_restart_transition_accepts_retail_signature() -> None:
    before = _restart_state(
        native_game_time=5205,
        runtime_active=False,
        active=0,
        pending=0,
    )
    after = _restart_state(
        native_game_time=119,
        runtime_active=False,
        active=1,
        pending=9,
    )
    assert is_natural_loss_restart_transition(
        before,
        after,
        unavailable_count=17,
    ) is True


@pytest.mark.parametrize(
    ("native_game_time", "score", "score_target", "expected"),
    (
        (6492, 7950, 9650, True),
        (1000, 7950, 9650, True),
        (999, 7950, 9650, False),
        (122, 7950, 9650, False),
        (6492, 9650, 9650, False),
        (6492, 7950, 0, False),
    ),
)
def test_natural_loss_mature_anchor_cannot_be_replaced_by_reset_state(
    native_game_time: int,
    score: int,
    score_target: int,
    expected: bool,
) -> None:
    state = _restart_state(
        native_game_time=native_game_time,
        score=score,
        score_target=score_target,
        runtime_active=False,
        active=0,
        pending=1,
    )
    assert is_natural_loss_mature_anchor(state) is expected


def test_natural_loss_clock_reset_latches_replacement_loading_state() -> None:
    before = _restart_state(
        native_game_time=5205,
        runtime_active=False,
        active=0,
        pending=0,
    )
    loading = _restart_state(
        native_game_time=119,
        runtime_active=False,
        active=0,
        pending=0,
    )
    assert is_natural_loss_restart_clock_reset(
        before,
        loading,
        unavailable_count=17,
    ) is True
    assert is_natural_loss_restart_transition(
        before,
        loading,
        unavailable_count=17,
    ) is False


def test_board_only_loss_transition_requires_stable_new_board_identity() -> None:
    before = _restart_state(
        native_game_time=5205,
        runtime_active=False,
        active=0,
        pending=0,
    )
    first_replacement = _restart_state(
        native_game_time=0,
        score_target=9616,
        runtime_active=False,
        active=0,
        pending=10,
        board_address=0x200000,
    )
    first_replacement["curve_plan_exhausted"] = False
    after = dict(first_replacement)
    after["native_game_time"] = 5
    assert is_natural_loss_board_identity_replacement_state(
        before,
        first_replacement,
        unavailable_count=17,
    ) is True
    assert is_natural_loss_board_replacement_transition(
        before,
        first_replacement,
        after,
        unavailable_count=17,
        replacement_snapshot_count=2,
    ) is True
    assert is_natural_loss_board_replacement_transition(
        before,
        first_replacement,
        after,
        unavailable_count=17,
        replacement_snapshot_count=1,
    ) is False


@pytest.mark.parametrize(
    "mutation",
    (
        lambda state: state.update(curve_plan_exhausted=True),
        lambda state: state.update(runtime_active=True),
        lambda state: state.update(score_target=0),
        lambda state: state.update(pending_ball_count=0),
        lambda state: state.update(board_address=0x100000),
        lambda state: state.update(board_address=0),
    ),
)
def test_board_only_loss_transition_rejects_nonreplacement_state(
    mutation: object,
) -> None:
    before = _restart_state(
        native_game_time=5205,
        runtime_active=False,
        active=0,
        pending=0,
    )
    first_replacement = _restart_state(
        native_game_time=0,
        score_target=9616,
        runtime_active=False,
        active=0,
        pending=10,
        board_address=0x200000,
    )
    first_replacement["curve_plan_exhausted"] = False
    mutation(first_replacement)
    after = dict(first_replacement)
    after["native_game_time"] = 5
    assert is_natural_loss_board_replacement_transition(
        before,
        first_replacement,
        after,
        unavailable_count=17,
        replacement_snapshot_count=2,
    ) is False


@pytest.mark.parametrize(
    ("after_update", "snapshot_count"),
    (
        (lambda state: state.update(native_game_time=4), 2),
        (lambda state: state.update(board_address=0x300000), 2),
        (lambda state: None, 1),
    ),
)
def test_board_only_loss_transition_rejects_unstable_confirmation(
    after_update: object,
    snapshot_count: int,
) -> None:
    before = _restart_state(
        native_game_time=5205,
        runtime_active=False,
        active=0,
        pending=0,
    )
    first_replacement = _restart_state(
        native_game_time=0,
        score_target=9616,
        runtime_active=False,
        active=0,
        pending=10,
        board_address=0x200000,
    )
    first_replacement["curve_plan_exhausted"] = False
    after = dict(first_replacement)
    after["native_game_time"] = 5
    after_update(after)
    assert is_natural_loss_board_replacement_transition(
        before,
        first_replacement,
        after,
        unavailable_count=17,
        replacement_snapshot_count=snapshot_count,
    ) is False


@pytest.mark.parametrize(
    ("mutation", "unavailable_count"),
    [
        (lambda before, after: before.update(score=9650), 1),
        (lambda before, after: before.update(native_game_time=900), 1),
        (lambda before, after: after.update(native_game_time=1200), 1),
        (lambda before, after: after.update(score_target=0), 1),
        (
            lambda before, after: after.update(
                active_ball_count=0,
                pending_ball_count=0,
            ),
            1,
        ),
        (lambda before, after: None, 0),
    ],
)
def test_natural_loss_restart_transition_fails_closed(
    mutation: object,
    unavailable_count: int,
) -> None:
    before = _restart_state(
        native_game_time=5205,
        runtime_active=False,
        active=0,
        pending=0,
    )
    after = _restart_state(
        native_game_time=119,
        runtime_active=False,
        active=1,
        pending=9,
    )
    mutation(before, after)
    assert is_natural_loss_restart_transition(
        before,
        after,
        unavailable_count=unavailable_count,
    ) is False
