"""Drive one retail Zuma board through ordinary mouse input.

The helper reads only the minimum live Board, Shooter, Curve and Ball fields
needed to choose a shot.  It never writes process memory.  Actions are sent
through Win32 ``SendInput`` so a simultaneous retail ``-record`` session can
capture an ordinary DMO command stream.

This is an evidence-acquisition controller, not certifying evidence by itself.
The resulting DMO and any later trajectory still have to pass the normal
provenance, no-mutation, determinism and Fidelity Gate checks.
"""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import struct
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tools.collect_pc_memory_probe import (
    ACTIVE_BOARD_OFFSET,
    BALL_OBJECT_SIZE,
    BALL_VTABLE,
    BOARD_CURVE_MANAGER_OFFSET,
    BOARD_DISPLAYED_SCORE_OFFSET,
    BOARD_DUMP_SIZE,
    BOARD_EMBEDDED_VTABLE,
    BOARD_EMBEDDED_VTABLE_OFFSET,
    BOARD_FIRED_BULLET_LIST_OFFSET,
    BOARD_PRIMARY_CHILD_OFFSET,
    BOARD_SCORE_OFFSET,
    BOARD_SCORE_TARGET_OFFSET,
    BOARD_VTABLE,
    BULLET_OBJECT_SIZE,
    BULLET_VTABLE,
    CURVE_INTRUSIVE_LIST_OFFSETS,
    CURVE_MANAGER_CURVE_ARRAY_OFFSET,
    CURVE_MANAGER_CURVE_COUNT_OFFSET,
    CURVE_MANAGER_DUMP_SIZE,
    CURVE_MANAGER_VTABLE,
    G_CURVE_PLAN_EXHAUSTED_ADDRESS,
    G_SEXY_APP_BASE_ADDRESS,
    MAX_CURVE_LIST_ITEMS,
    MAX_CURVES,
    MAX_FIRED_BULLETS,
    ProbeError,
    SHOOTER_BULLET_POINTER_OFFSETS,
    SHOOTER_VTABLE,
    _decode_ball,
    _i32,
    _u32,
)
from tools.control_popcap_replay import main_window_for_pid
from tools.inspect_popcap_replay import (
    close_process,
    open_process_readonly,
    read_process_bytes,
    read_replay_state,
)
from tools.plan_live_zuma_shot import _candidate_rows, _first_hit
from zuma_rl.pc_memory_trajectory import (
    BOARD_FRUIT_ACTIVE_POINT_POINTER_OFFSET,
    BOARD_FRUIT_COLLECTING_OFFSET,
    BOARD_FRUIT_EXPIRY_TIME_OFFSET,
    BOARD_FRUIT_SELECTED_POINT_OFFSET,
    BOARD_FRUIT_VERTICAL_OFFSET,
    BOARD_NATIVE_GAME_TIME_OFFSET,
    BOARD_RUNTIME_FLAG_157_OFFSET,
    BOARD_RUNTIME_I32_F54_OFFSET,
)


ACTIVE_CHAIN_LIST_OFFSET = 0x5C
INSERTING_CHAIN_LIST_OFFSET = 0x50
PENDING_CHAIN_LIST_OFFSET = 0x68
LOGICAL_WIDTH = 800
LOGICAL_HEIGHT = 600
FRUIT_LOGICAL_WIDTH = 52
FRUIT_LOGICAL_HEIGHT = 52
FRUIT_POWERUP_COLLISION_RADIUS = 108.0
FRUIT_DISCARD_SAFETY_RADIUS = 60.0
PROXIMITY_BOMB_POWERUP_TYPE = 0
FRUIT_BOMB_DISCARD_POINTS = (
    (0.0, 0.0),
    (799.0, 0.0),
    (799.0, 599.0),
    (0.0, 599.0),
    (400.0, 0.0),
    (799.0, 300.0),
    (400.0, 599.0),
    (0.0, 300.0),
)
JUNGLE2_STRICT_FRUIT_WAYPOINT_INTERVALS = {
    0: ((0.0, 160.0), (2147.0, 2218.0)),
    1: ((1053.0, 1259.0),),
    2: ((2908.0, 3106.0),),
}
JUNGLE2_BASE_ADVANCE_WAYPOINTS_PER_TICK = 0.5
FRAMEWORK_MULTIPLIER_OFFSET = 0x4D0
_DPI_AWARENESS_VERIFIED = False


class LiveBoardUnavailable(RuntimeError):
    """The retail process is not currently exposing one supported board."""


@dataclass(frozen=True)
class LogicalViewport:
    screen_left: float
    screen_top: float
    scale: float
    logical_width: int = LOGICAL_WIDTH
    logical_height: int = LOGICAL_HEIGHT

    def screen_point(self, x: float, y: float) -> tuple[int, int]:
        if not (
            math.isfinite(x)
            and math.isfinite(y)
            and 0.0 <= x < self.logical_width
            and 0.0 <= y < self.logical_height
        ):
            raise ValueError("logical mouse point is outside the game canvas")
        return (
            round(self.screen_left + x * self.scale),
            round(self.screen_top + y * self.scale),
        )


def logical_viewport(
    client_left: int,
    client_top: int,
    client_width: int,
    client_height: int,
) -> LogicalViewport:
    """Fit the native 800x600 canvas into a Win32 client rectangle."""

    if client_width <= 0 or client_height <= 0:
        raise ValueError("client dimensions must be positive")
    scale = min(
        client_width / LOGICAL_WIDTH,
        client_height / LOGICAL_HEIGHT,
    )
    viewport_width = LOGICAL_WIDTH * scale
    viewport_height = LOGICAL_HEIGHT * scale
    return LogicalViewport(
        screen_left=client_left + (client_width - viewport_width) / 2.0,
        screen_top=client_top + (client_height - viewport_height) / 2.0,
        scale=scale,
    )


def _read_ball(handle: int, address: int) -> dict[str, Any]:
    if address == 0:
        raise LiveBoardUnavailable("shooter ball pointer is null")
    head = _u32(read_process_bytes(handle, address, 4), 0)
    if head == BALL_VTABLE:
        size = BALL_OBJECT_SIZE
    elif head == BULLET_VTABLE:
        size = BULLET_OBJECT_SIZE
    else:
        raise LiveBoardUnavailable(
            f"unsupported ball vtable: 0x{head:08x}"
        )
    raw = read_process_bytes(handle, address, size)
    decoded = _decode_ball(
        raw,
        expected_vtable=head,
    )
    powerup_lifetime_ticks = _i32(raw, 0xF8)
    powerup_transition_ticks = _i32(raw, 0xFC)
    if powerup_lifetime_ticks < 0 or powerup_transition_ticks < 0:
        raise LiveBoardUnavailable("ball power-up timing is invalid")
    decoded.update(
        {
            "powerup_lifetime_ticks": powerup_lifetime_ticks,
            "powerup_transition_ticks": powerup_transition_ticks,
        }
    )
    return decoded


def _read_curve_list(
    handle: int,
    *,
    curve_address: int,
    container_offset: int,
    include_entities: bool,
) -> tuple[dict[str, Any], ...] | int:
    container = read_process_bytes(
        handle,
        curve_address + container_offset,
        12,
    )
    sentinel = _u32(container, 4)
    declared_count = _u32(container, 8)
    if sentinel == 0 or declared_count > MAX_CURVE_LIST_ITEMS:
        raise LiveBoardUnavailable("curve list header is invalid")
    first_node, last_node = struct.unpack(
        "<II",
        read_process_bytes(handle, sentinel, 8),
    )
    node = first_node
    previous = sentinel
    visited: set[int] = set()
    records: list[dict[str, Any]] = []
    traversed = 0
    while node != sentinel:
        if traversed >= MAX_CURVE_LIST_ITEMS or node in visited:
            raise LiveBoardUnavailable("curve list cycle or limit")
        visited.add(node)
        next_node, linked_previous, payload_address = struct.unpack(
            "<III",
            read_process_bytes(handle, node, 12),
        )
        if linked_previous != previous or payload_address == 0:
            raise LiveBoardUnavailable("curve list link is inconsistent")
        if include_entities:
            records.append(
                {
                    "index": traversed,
                    "payload_address": payload_address,
                    "ball": _read_ball(handle, payload_address),
                }
            )
        traversed += 1
        previous = node
        node = next_node
    if traversed != declared_count or previous != last_node:
        raise LiveBoardUnavailable("curve list count is inconsistent")
    return tuple(records) if include_entities else traversed


def read_live_board(
    pid: int,
    *,
    require_shooter: bool = True,
    include_fruit_target: bool = False,
) -> dict[str, Any]:
    """Read one compact, fail-closed retail board snapshot."""

    handle = open_process_readonly(pid)
    try:
        sexy_app_base = _u32(
            read_process_bytes(handle, G_SEXY_APP_BASE_ADDRESS, 4),
            0,
        )
        if sexy_app_base == 0:
            raise LiveBoardUnavailable("SexyApp pointer is null")
        board_address = _u32(
            read_process_bytes(
                handle,
                sexy_app_base + ACTIVE_BOARD_OFFSET,
                4,
            ),
            0,
        )
        if board_address == 0:
            raise LiveBoardUnavailable("active Board pointer is null")
        board = read_process_bytes(handle, board_address, BOARD_DUMP_SIZE)
        if (
            _u32(board, 0) != BOARD_VTABLE
            or _u32(board, BOARD_EMBEDDED_VTABLE_OFFSET)
            != BOARD_EMBEDDED_VTABLE
        ):
            raise LiveBoardUnavailable("active Board identity mismatch")

        fruit_pointer = _u32(
            board,
            BOARD_FRUIT_ACTIVE_POINT_POINTER_OFFSET,
        )
        fruit_selected_point = _i32(
            board,
            BOARD_FRUIT_SELECTED_POINT_OFFSET,
        )
        fruit_collecting_raw = board[BOARD_FRUIT_COLLECTING_OFFSET]
        fruit_expiry_time = _i32(
            board,
            BOARD_FRUIT_EXPIRY_TIME_OFFSET,
        )
        fruit_vertical_offset = struct.unpack_from(
            "<f",
            board,
            BOARD_FRUIT_VERTICAL_OFFSET,
        )[0]
        if (
            fruit_collecting_raw not in {0, 1}
            or fruit_expiry_time < 0
            or not math.isfinite(fruit_vertical_offset)
            or (fruit_pointer == 0 and fruit_selected_point not in {-1, 0})
            or (fruit_pointer != 0 and fruit_selected_point < 0)
        ):
            raise LiveBoardUnavailable("fruit state is invalid")
        fruit: dict[str, Any] = {
            "active": fruit_pointer != 0,
            "selected_point_index": fruit_selected_point,
            "collecting": bool(fruit_collecting_raw),
            "vertical_offset": fruit_vertical_offset,
            "expiry_time": fruit_expiry_time,
            "target_x": None,
            "target_y": None,
        }
        if fruit_pointer != 0 and include_fruit_target:
            # Retail Board+0x0B4 points to the level TreasurePoint record.
            # Its first two int32 fields are x/y.  Every installed Revenge
            # fruit sheet is 52x52 logical pixels; Board's oscillator adds
            # only the captured vertical offset to the rendered centre.
            point_raw = read_process_bytes(handle, fruit_pointer, 20)
            point_x, point_y = struct.unpack_from("<ii", point_raw)
            target_x = point_x + FRUIT_LOGICAL_WIDTH / 2.0
            target_y = (
                point_y
                + FRUIT_LOGICAL_HEIGHT / 2.0
                + fruit_vertical_offset
            )
            if not (
                -FRUIT_LOGICAL_WIDTH <= point_x <= LOGICAL_WIDTH
                and -FRUIT_LOGICAL_HEIGHT <= point_y <= LOGICAL_HEIGHT
                and 0.0 <= target_x < LOGICAL_WIDTH
                and 0.0 <= target_y < LOGICAL_HEIGHT
            ):
                raise LiveBoardUnavailable("fruit point is outside the canvas")
            fruit.update(
                {
                    "point_x": point_x,
                    "point_y": point_y,
                    "target_x": target_x,
                    "target_y": target_y,
                }
            )

        shooter_address = _u32(board, BOARD_PRIMARY_CHILD_OFFSET)
        shooter = read_process_bytes(
            handle,
            shooter_address,
            max(SHOOTER_BULLET_POINTER_OFFSETS) + 4,
        )
        if _u32(shooter, 0) != SHOOTER_VTABLE:
            raise LiveBoardUnavailable("Shooter identity mismatch")
        chamber: list[dict[str, Any] | None] = []
        for pointer_offset in SHOOTER_BULLET_POINTER_OFFSETS:
            address = _u32(shooter, pointer_offset)
            if require_shooter:
                chamber.append(_read_ball(handle, address))
                continue
            if address == 0:
                chamber.append(None)
                continue
            try:
                chamber.append(_read_ball(handle, address))
            except (
                LiveBoardUnavailable,
                OSError,
                ProbeError,
                RuntimeError,
                ValueError,
            ):
                chamber.append(None)
        current, next_ball = chamber

        manager_address = _u32(board, BOARD_CURVE_MANAGER_OFFSET)
        manager = read_process_bytes(
            handle,
            manager_address,
            CURVE_MANAGER_DUMP_SIZE,
        )
        if _u32(manager, 0) != CURVE_MANAGER_VTABLE:
            raise LiveBoardUnavailable("CurveManager identity mismatch")
        curve_count = _i32(manager, CURVE_MANAGER_CURVE_COUNT_OFFSET)
        if not 0 < curve_count <= MAX_CURVES:
            raise LiveBoardUnavailable("CurveManager count is invalid")
        if curve_count != 1:
            raise LiveBoardUnavailable(
                "autoplayer currently requires exactly one curve"
            )
        curve_address = _u32(
            manager,
            CURVE_MANAGER_CURVE_ARRAY_OFFSET,
        )
        active = _read_curve_list(
            handle,
            curve_address=curve_address,
            container_offset=ACTIVE_CHAIN_LIST_OFFSET,
            include_entities=True,
        )
        assert isinstance(active, tuple)
        inserting = _read_curve_list(
            handle,
            curve_address=curve_address,
            container_offset=INSERTING_CHAIN_LIST_OFFSET,
            include_entities=False,
        )
        pending = _read_curve_list(
            handle,
            curve_address=curve_address,
            container_offset=PENDING_CHAIN_LIST_OFFSET,
            include_entities=False,
        )
        fired_count = _u32(
            board,
            BOARD_FIRED_BULLET_LIST_OFFSET + 8,
        )
        if fired_count > MAX_FIRED_BULLETS:
            raise LiveBoardUnavailable("fired bullet count is invalid")
        plan_exhausted = read_process_bytes(
            handle,
            G_CURVE_PLAN_EXHAUSTED_ADDRESS,
            1,
        )[0]
        runtime_flag = board[BOARD_RUNTIME_FLAG_157_OFFSET]
        if plan_exhausted not in {0, 1} or runtime_flag not in {0, 1}:
            raise LiveBoardUnavailable("board runtime flags are invalid")

        return {
            "process_id": pid,
            "board_address": board_address,
            "native_game_time": _i32(
                board,
                BOARD_NATIVE_GAME_TIME_OFFSET,
            ),
            "score": _i32(board, BOARD_SCORE_OFFSET),
            "displayed_score": _i32(
                board,
                BOARD_DISPLAYED_SCORE_OFFSET,
            ),
            "score_target": _i32(board, BOARD_SCORE_TARGET_OFFSET),
            "runtime_active": bool(runtime_flag),
            "loss_counter": _i32(
                board,
                BOARD_RUNTIME_I32_F54_OFFSET,
            ),
            "curve_plan_exhausted": bool(plan_exhausted),
            "active_records": active,
            "active_ball_count": len(active),
            "inserting_ball_count": int(inserting),
            "pending_ball_count": int(pending),
            "fired_ball_count": fired_count,
            "fruit": fruit,
            "current": current,
            "next": next_ball,
        }
    except (
        OSError,
        ProbeError,
        RuntimeError,
        ValueError,
        struct.error,
    ) as error:
        if isinstance(error, LiveBoardUnavailable):
            raise
        raise LiveBoardUnavailable(str(error)) from error
    finally:
        close_process(handle)


def read_framework_state(pid: int) -> dict[str, Any]:
    """Read stable SexyApp timing fields without scanning or mutation."""

    handle = open_process_readonly(pid)
    try:
        sexy_app_base = _u32(
            read_process_bytes(handle, G_SEXY_APP_BASE_ADDRESS, 4),
            0,
        )
        if sexy_app_base == 0:
            raise LiveBoardUnavailable("SexyApp pointer is null")
        state = read_replay_state(
            handle,
            sexy_app_base + FRAMEWORK_MULTIPLIER_OFFSET,
        )
        if state.frame_time_ms <= 0 or state.update_count < 0:
            raise LiveBoardUnavailable("framework timing fields are invalid")
        return {
            "sexy_app_base": sexy_app_base,
            "framework_update": state.update_count,
            "frame_time_ms": state.frame_time_ms,
            "update_multiplier": state.update_multiplier,
            "paused": state.paused,
            "loaded": state.loaded,
            "loading_thread_started": state.loading_thread_started,
            "loading_thread_completed": state.loading_thread_completed,
        }
    except (OSError, RuntimeError, ValueError, struct.error) as error:
        if isinstance(error, LiveBoardUnavailable):
            raise
        raise LiveBoardUnavailable(str(error)) from error
    finally:
        close_process(handle)


def _recommend_candidate_rows(
    current_rows: Sequence[Mapping[str, Any]],
    next_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the frozen ordinary match/build ordering to candidate rows."""

    current_match = next(
        (row for row in current_rows if int(row["run_length"]) >= 2),
        None,
    )
    next_match = next(
        (row for row in next_rows if int(row["run_length"]) >= 2),
        None,
    )
    if current_match is not None or next_match is not None:
        use_next = (
            next_match is not None
            and (
                current_match is None
                or int(next_match["run_length"])
                > int(current_match["run_length"])
            )
        )
        chosen = next_match if use_next else current_match
        assert chosen is not None
        return {
            "action": "swap_then_fire" if use_next else "fire",
            "reason": "complete_match",
            **chosen,
        }
    if current_rows:
        return {
            "action": "fire",
            "reason": "build_pair",
            **current_rows[0],
        }
    if next_rows:
        return {
            "action": "swap_then_fire",
            "reason": "build_pair",
            **next_rows[0],
        }
    return {"action": "wait", "reason": "no_unobstructed_same_colour"}


def _point_segment_distance_squared(
    point_x: float,
    point_y: float,
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
) -> float:
    dx = end_x - start_x
    dy = end_y - start_y
    length_squared = dx * dx + dy * dy
    if length_squared <= 1e-12:
        return (point_x - start_x) ** 2 + (point_y - start_y) ** 2
    fraction = (
        (point_x - start_x) * dx + (point_y - start_y) * dy
    ) / length_squared
    fraction = max(0.0, min(1.0, fraction))
    closest_x = start_x + fraction * dx
    closest_y = start_y + fraction * dy
    return (point_x - closest_x) ** 2 + (point_y - closest_y) ** 2


def _safe_bomb_discard_point(
    records: Sequence[Mapping[str, Any]],
    *,
    origin_x: float,
    origin_y: float,
    fruit: Any,
) -> tuple[float, float] | None:
    fruit_center: tuple[float, float] | None = None
    if isinstance(fruit, Mapping) and fruit.get("active") is True:
        target_x = fruit.get("target_x")
        target_y = fruit.get("target_y")
        if (
            isinstance(target_x, bool)
            or not isinstance(target_x, (int, float))
            or isinstance(target_y, bool)
            or not isinstance(target_y, (int, float))
            or not math.isfinite(float(target_x))
            or not math.isfinite(float(target_y))
            or not 0.0 <= float(target_x) < LOGICAL_WIDTH
            or not 0.0 <= float(target_y) < LOGICAL_HEIGHT
        ):
            raise ValueError("active fruit target is invalid")
        fruit_center = (float(target_x), float(target_y))

    safety_squared = FRUIT_DISCARD_SAFETY_RADIUS**2
    for target_x, target_y in FRUIT_BOMB_DISCARD_POINTS:
        if (
            _first_hit(
                records,
                origin_x=origin_x,
                origin_y=origin_y,
                target_x=target_x,
                target_y=target_y,
            )
            is not None
        ):
            continue
        if fruit_center is not None and (
            _point_segment_distance_squared(
                fruit_center[0],
                fruit_center[1],
                origin_x,
                origin_y,
                target_x,
                target_y,
            )
            < safety_squared
        ):
            continue
        return target_x, target_y
    return None


def _forward_waypoint_distance(
    waypoint: float,
    intervals: Sequence[tuple[float, float]],
) -> float:
    candidates: list[float] = []
    for start, end in intervals:
        if start <= waypoint <= end:
            return 0.0
        if waypoint < start:
            candidates.append(start - waypoint)
    return min(candidates) if candidates else math.inf


def fruit_bomb_observation(
    state: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Project one read-only fruit/bomb geometry sample for diagnostics."""

    records = state.get("active_records")
    fruit = state.get("fruit")
    current = state.get("current")
    next_ball = state.get("next")
    if (
        not isinstance(records, Sequence)
        or isinstance(records, (str, bytes))
        or not isinstance(fruit, Mapping)
        or not isinstance(current, Mapping)
        or not isinstance(next_ball, Mapping)
    ):
        raise ValueError("fruit/bomb observation state is incomplete")

    fruit_active = fruit.get("active") is True
    point_x: float | None = None
    point_y: float | None = None
    if fruit_active:
        raw_point_x = fruit.get("point_x")
        raw_point_y = fruit.get("point_y")
        if (
            isinstance(raw_point_x, bool)
            or not isinstance(raw_point_x, (int, float))
            or isinstance(raw_point_y, bool)
            or not isinstance(raw_point_y, (int, float))
            or not math.isfinite(float(raw_point_x))
            or not math.isfinite(float(raw_point_y))
        ):
            raise ValueError("active fruit point is invalid")
        point_x = float(raw_point_x)
        point_y = float(raw_point_y)

    radius_squared = FRUIT_POWERUP_COLLISION_RADIUS**2
    bombs: list[dict[str, Any]] = []
    for chain_index, record in enumerate(records):
        if not isinstance(record, Mapping):
            continue
        ball = record.get("ball")
        if (
            not isinstance(ball, Mapping)
            or ball.get("powerup_primary_type")
            != PROXIMITY_BOMB_POWERUP_TYPE
        ):
            continue
        ball_id = ball.get("ball_id")
        color_id = ball.get("color_id")
        ball_x = ball.get("position_x")
        ball_y = ball.get("position_y")
        curve_distance = ball.get("curve_distance")
        powerup_lifetime_ticks = ball.get("powerup_lifetime_ticks")
        powerup_transition_ticks = ball.get("powerup_transition_ticks")
        if (
            isinstance(ball_id, bool)
            or not isinstance(ball_id, int)
            or isinstance(color_id, bool)
            or not isinstance(color_id, int)
            or isinstance(ball_x, bool)
            or not isinstance(ball_x, (int, float))
            or isinstance(ball_y, bool)
            or not isinstance(ball_y, (int, float))
            or not math.isfinite(float(ball_x))
            or not math.isfinite(float(ball_y))
            or isinstance(curve_distance, bool)
            or not isinstance(curve_distance, (int, float))
            or not math.isfinite(float(curve_distance))
            or isinstance(powerup_lifetime_ticks, bool)
            or not isinstance(powerup_lifetime_ticks, int)
            or powerup_lifetime_ticks < 0
            or isinstance(powerup_transition_ticks, bool)
            or not isinstance(powerup_transition_ticks, int)
            or powerup_transition_ticks < 0
        ):
            raise ValueError("active bomb state is invalid")
        distance_squared = (
            None
            if point_x is None or point_y is None
            else (
                (float(ball_x) - point_x) ** 2
                + (float(ball_y) - point_y) ** 2
            )
        )
        bombs.append(
            {
                "chain_index": chain_index,
                "ball_id": ball_id,
                "color_id": color_id,
                "position_x": float(ball_x),
                "position_y": float(ball_y),
                "curve_distance": float(curve_distance),
                "powerup_lifetime_ticks": powerup_lifetime_ticks,
                "powerup_transition_ticks": powerup_transition_ticks,
                "static_point_distance_squared": distance_squared,
                "strictly_within_108": (
                    distance_squared is not None
                    and distance_squared < radius_squared
                ),
            }
        )

    if not fruit_active and not bombs:
        return None
    distances = tuple(
        float(bomb["static_point_distance_squared"])
        for bomb in bombs
        if bomb["static_point_distance_squared"] is not None
    )
    strict_count = sum(
        1 for bomb in bombs if bomb["strictly_within_108"] is True
    )
    return {
        "type": "fruit_bomb_observation",
        "board_address": state.get("board_address"),
        "native_game_time": state.get("native_game_time"),
        "score": state.get("score"),
        "fruit": {
            key: fruit.get(key)
            for key in (
                "active",
                "collecting",
                "selected_point_index",
                "point_x",
                "point_y",
                "target_x",
                "target_y",
                "vertical_offset",
                "expiry_time",
            )
        },
        "active_bomb_count": len(bombs),
        "bombs": bombs,
        "fruit_bomb_temporal_overlap": fruit_active and bool(bombs),
        "minimum_static_point_distance_squared": (
            min(distances) if distances else None
        ),
        "strict_overlap_bomb_count": strict_count,
        "strict_radius_squared": radius_squared,
        "current_color_id": current.get("color_id"),
        "next_color_id": next_ball.get("color_id"),
        "fired_ball_count": state.get("fired_ball_count"),
        "inserting_ball_count": state.get("inserting_ball_count"),
    }


def choose_shot(
    state: Mapping[str, Any],
    *,
    fruit_policy: str = "ignore",
) -> dict[str, Any]:
    """Choose an unobstructed same-colour target using one snapshot."""

    if fruit_policy not in {"ignore", "collect", "bomb"}:
        raise ValueError("fruit policy is invalid")

    records = state.get("active_records")
    current = state.get("current")
    next_ball = state.get("next")
    if (
        not isinstance(records, Sequence)
        or isinstance(records, (str, bytes))
        or not isinstance(current, Mapping)
        or not isinstance(next_ball, Mapping)
    ):
        raise ValueError("live board state is incomplete")
    fruit = state.get("fruit")
    if fruit_policy == "collect" and isinstance(fruit, Mapping):
        if fruit.get("active") is True and fruit.get("collecting") is False:
            target_x = fruit.get("target_x")
            target_y = fruit.get("target_y")
            selected = fruit.get("selected_point_index")
            if (
                isinstance(target_x, bool)
                or not isinstance(target_x, (int, float))
                or isinstance(target_y, bool)
                or not isinstance(target_y, (int, float))
                or isinstance(selected, bool)
                or not isinstance(selected, int)
                or selected < 0
                or not math.isfinite(float(target_x))
                or not math.isfinite(float(target_y))
                or not 0.0 <= float(target_x) < LOGICAL_WIDTH
                or not 0.0 <= float(target_y) < LOGICAL_HEIGHT
            ):
                raise ValueError("active fruit target is invalid")
            return {
                "action": "fire",
                "reason": "collect_fruit",
                "target_x": float(target_x),
                "target_y": float(target_y),
                "selected_point_index": selected,
            }
    if not records:
        return {"action": "wait", "reason": "active_chain_empty"}
    origin_x = float(current["position_x"])
    origin_y = float(current["position_y"])
    def visible(
        rows: Sequence[Mapping[str, Any]],
    ) -> tuple[Mapping[str, Any], ...]:
        return tuple(
            row
            for row in rows
            if (
                0.0 <= float(row["target_x"]) < LOGICAL_WIDTH
                and 0.0 <= float(row["target_y"]) < LOGICAL_HEIGHT
            )
        )

    current_rows = visible(
        _candidate_rows(
            records,
            color_id=int(current["color_id"]),
            origin_x=origin_x,
            origin_y=origin_y,
        )
    )
    next_rows = visible(
        _candidate_rows(
            records,
            color_id=int(next_ball["color_id"]),
            origin_x=origin_x,
            origin_y=origin_y,
        )
    )

    active_bombs: dict[int, dict[str, Any]] = {}
    if fruit_policy == "bomb":
        for chain_index, record in enumerate(records):
            if not isinstance(record, Mapping):
                continue
            ball = record.get("ball")
            if (
                not isinstance(ball, Mapping)
                or ball.get("powerup_primary_type")
                != PROXIMITY_BOMB_POWERUP_TYPE
            ):
                continue
            ball_x = ball.get("position_x")
            ball_y = ball.get("position_y")
            ball_id = ball.get("ball_id")
            curve_distance = ball.get("curve_distance")
            powerup_lifetime_ticks = ball.get("powerup_lifetime_ticks")
            powerup_transition_ticks = ball.get("powerup_transition_ticks")
            if (
                isinstance(ball_x, bool)
                or not isinstance(ball_x, (int, float))
                or isinstance(ball_y, bool)
                or not isinstance(ball_y, (int, float))
                or isinstance(ball_id, bool)
                or not isinstance(ball_id, int)
                or not math.isfinite(float(ball_x))
                or not math.isfinite(float(ball_y))
                or isinstance(curve_distance, bool)
                or not isinstance(curve_distance, (int, float))
                or not math.isfinite(float(curve_distance))
                or isinstance(powerup_lifetime_ticks, bool)
                or not isinstance(powerup_lifetime_ticks, int)
                or powerup_lifetime_ticks < 0
                or isinstance(powerup_transition_ticks, bool)
                or not isinstance(powerup_transition_ticks, int)
                or powerup_transition_ticks < 0
            ):
                continue
            active_bombs[chain_index] = {
                "chain_index": chain_index,
                "ball_id": ball_id,
                "position_x": float(ball_x),
                "position_y": float(ball_y),
                "curve_distance": float(curve_distance),
                "powerup_lifetime_ticks": powerup_lifetime_ticks,
                "powerup_transition_ticks": powerup_transition_ticks,
            }

    if fruit_policy == "bomb" and isinstance(fruit, Mapping):
        if fruit.get("active") is True and fruit.get("collecting") is False:
            point_x = fruit.get("point_x")
            point_y = fruit.get("point_y")
            selected = fruit.get("selected_point_index")
            if (
                isinstance(point_x, bool)
                or not isinstance(point_x, (int, float))
                or isinstance(point_y, bool)
                or not isinstance(point_y, (int, float))
                or isinstance(selected, bool)
                or not isinstance(selected, int)
                or selected < 0
                or not math.isfinite(float(point_x))
                or not math.isfinite(float(point_y))
                or not -FRUIT_LOGICAL_WIDTH <= float(point_x) <= LOGICAL_WIDTH
                or not -FRUIT_LOGICAL_HEIGHT <= float(point_y) <= LOGICAL_HEIGHT
            ):
                raise ValueError("active fruit point is invalid")

            radius_squared = FRUIT_POWERUP_COLLISION_RADIUS**2
            qualifying_bombs: dict[int, dict[str, Any]] = {}
            for chain_index, bomb in active_bombs.items():
                distance_squared = (
                    (float(bomb["position_x"]) - float(point_x)) ** 2
                    + (float(bomb["position_y"]) - float(point_y)) ** 2
                )
                if distance_squared < radius_squared:
                    qualifying_bombs[chain_index] = {
                        "chain_index": chain_index,
                        "ball_id": bomb["ball_id"],
                        "distance_squared": distance_squared,
                    }

            bomb_candidates: list[
                tuple[tuple[int, int, int, int], bool, Mapping[str, Any], dict[str, Any]]
            ] = []
            for use_next, rows in (
                (False, current_rows),
                (True, next_rows),
            ):
                for row_rank, row in enumerate(rows):
                    run_start = int(row["run_start"])
                    run_end = int(row["run_end"])
                    bombs_in_run = tuple(
                        bomb
                        for chain_index, bomb in qualifying_bombs.items()
                        if run_start <= chain_index <= run_end
                    )
                    if not bombs_in_run:
                        continue
                    bomb = min(
                        bombs_in_run,
                        key=lambda value: (
                            float(value["distance_squared"]),
                            int(value["chain_index"]),
                        ),
                    )
                    run_length = int(row["run_length"])
                    selection_key = (
                        int(run_length >= 2),
                        run_length,
                        int(not use_next),
                        -row_rank,
                    )
                    bomb_candidates.append(
                        (selection_key, use_next, row, bomb)
                    )
            if bomb_candidates:
                _, use_next, chosen, bomb = max(
                    bomb_candidates,
                    key=lambda value: value[0],
                )
                return {
                    "action": "swap_then_fire" if use_next else "fire",
                    "reason": "trigger_fruit_proximity_bomb",
                    **chosen,
                    "proximity_bomb_chain_index": bomb["chain_index"],
                    "proximity_bomb_ball_id": bomb["ball_id"],
                    "fruit_selected_point_index": selected,
                    "fruit_point_x": float(point_x),
                    "fruit_point_y": float(point_y),
                    "fruit_distance_squared": bomb["distance_squared"],
                    "fruit_collision_radius_squared": radius_squared,
                }

            if active_bombs:
                expiry_time = fruit.get("expiry_time")
                native_game_time = state.get("native_game_time")
                target_x = fruit.get("target_x")
                target_y = fruit.get("target_y")
                intervals = JUNGLE2_STRICT_FRUIT_WAYPOINT_INTERVALS.get(
                    selected
                )
                if (
                    isinstance(expiry_time, bool)
                    or not isinstance(expiry_time, int)
                    or expiry_time < 0
                    or isinstance(native_game_time, bool)
                    or not isinstance(native_game_time, int)
                    or native_game_time < 0
                    or isinstance(target_x, bool)
                    or not isinstance(target_x, (int, float))
                    or isinstance(target_y, bool)
                    or not isinstance(target_y, (int, float))
                    or not math.isfinite(float(target_x))
                    or not math.isfinite(float(target_y))
                    or not 0.0 <= float(target_x) < LOGICAL_WIDTH
                    or not 0.0 <= float(target_y) < LOGICAL_HEIGHT
                    or intervals is None
                ):
                    raise ValueError("active fruit alignment state is invalid")
                fruit_remaining_ticks = max(
                    0,
                    expiry_time - native_game_time,
                )
                reachability: list[dict[str, Any]] = []
                for chain_index, bomb in active_bombs.items():
                    shared_ticks = min(
                        fruit_remaining_ticks,
                        int(bomb["powerup_lifetime_ticks"]),
                    )
                    required_waypoints = _forward_waypoint_distance(
                        float(bomb["curve_distance"]),
                        intervals,
                    )
                    budget_waypoints = (
                        JUNGLE2_BASE_ADVANCE_WAYPOINTS_PER_TICK
                        * shared_ticks
                    )
                    reachability.append(
                        {
                            "chain_index": chain_index,
                            "ball_id": bomb["ball_id"],
                            "curve_distance": bomb["curve_distance"],
                            "powerup_lifetime_ticks": bomb[
                                "powerup_lifetime_ticks"
                            ],
                            "shared_remaining_ticks": shared_ticks,
                            "required_forward_waypoints": (
                                required_waypoints
                            ),
                            "forward_waypoint_budget": budget_waypoints,
                            "reachable_current_cycle": (
                                required_waypoints <= budget_waypoints
                            ),
                        }
                    )
                if not any(
                    row["reachable_current_cycle"]
                    for row in reachability
                ):
                    return {
                        "action": "fire",
                        "reason": "defer_fruit_for_bomb_alignment",
                        "target_x": float(target_x),
                        "target_y": float(target_y),
                        "fruit_selected_point_index": selected,
                        "fruit_expiry_time": expiry_time,
                        "fruit_remaining_ticks": fruit_remaining_ticks,
                        "strict_waypoint_intervals": intervals,
                        "bomb_reachability": tuple(reachability),
                    }

    if fruit_policy == "bomb" and active_bombs:
        bomb_indices = frozenset(active_bombs)

        def bomb_free(
            rows: Sequence[Mapping[str, Any]],
        ) -> tuple[Mapping[str, Any], ...]:
            return tuple(
                row
                for row in rows
                if not any(
                    int(row["run_start"]) <= chain_index
                    <= int(row["run_end"])
                    for chain_index in bomb_indices
                )
            )

        safe_recommendation = _recommend_candidate_rows(
            bomb_free(current_rows),
            bomb_free(next_rows),
        )
        bomb_ids = tuple(
            int(active_bombs[index]["ball_id"])
            for index in sorted(active_bombs)
        )
        if safe_recommendation["action"] != "wait":
            preserved_reason = safe_recommendation["reason"]
            return {
                **safe_recommendation,
                "reason": "preserve_fruit_proximity_bomb",
                "preservation_strategy": "bomb_free_candidate_run",
                "preserved_action_reason": preserved_reason,
                "preserved_bomb_chain_indices": tuple(sorted(active_bombs)),
                "preserved_bomb_ball_ids": bomb_ids,
            }
        discard = _safe_bomb_discard_point(
            records,
            origin_x=origin_x,
            origin_y=origin_y,
            fruit=fruit,
        )
        if discard is None:
            return {
                "action": "wait",
                "reason": "preserve_fruit_proximity_bomb",
                "preservation_strategy": "wait_for_safe_discard_ray",
                "preserved_bomb_chain_indices": tuple(sorted(active_bombs)),
                "preserved_bomb_ball_ids": bomb_ids,
            }
        return {
            "action": "fire",
            "reason": "preserve_fruit_proximity_bomb",
            "preservation_strategy": "clear_perimeter_discard",
            "target_x": discard[0],
            "target_y": discard[1],
            "fruit_discard_safety_radius": FRUIT_DISCARD_SAFETY_RADIUS,
            "preserved_bomb_chain_indices": tuple(sorted(active_bombs)),
            "preserved_bomb_ball_ids": bomb_ids,
        }

    return _recommend_candidate_rows(current_rows, next_rows)


def is_natural_win_state(state: Mapping[str, Any]) -> bool:
    gameplay_count = sum(
        int(state[key])
        for key in (
            "active_ball_count",
            "inserting_ball_count",
            "pending_ball_count",
            "fired_ball_count",
        )
    )
    return (
        state["runtime_active"] is False
        and int(state["loss_counter"]) == 0
        and gameplay_count == 0
        and int(state["score_target"]) > 0
        and int(state["score"]) >= int(state["score_target"])
        and state.get("curve_plan_exhausted") is True
    )


def is_natural_loss_state(state: Mapping[str, Any]) -> bool:
    """Recognize the stable below-target retail Game Over Board."""

    moving_gameplay_count = sum(
        int(state[key])
        for key in (
            "active_ball_count",
            "inserting_ball_count",
            "fired_ball_count",
        )
    )
    return (
        state["runtime_active"] is False
        and state.get("curve_plan_exhausted") is True
        and moving_gameplay_count == 0
        and int(state["pending_ball_count"]) == 1
        and int(state["native_game_time"]) > 0
        and int(state["score_target"]) > 0
        and int(state["score"]) < int(state["score_target"])
    )


NATURAL_LOSS_RESTART_RECEIPT_KIND = (
    "natural_loss_restart_transition_v1"
)
NATURAL_LOSS_BOARD_REPLACEMENT_V2_RECEIPT_KIND = (
    "natural_loss_board_replacement_transition_v2"
)
NATURAL_LOSS_BOARD_IDENTITY_REPLACEMENT_RECEIPT_KIND = (
    "natural_loss_board_identity_replacement_transition_v3"
)
NATURAL_LOSS_MINIMUM_PRE_RESTART_NATIVE_TIME = 1000
NATURAL_LOSS_MAXIMUM_POST_RESTART_NATIVE_TIME = 1000
NATURAL_LOSS_MINIMUM_NATIVE_TIME_RESET = 1000
NATURAL_LOSS_MINIMUM_REPLACEMENT_SNAPSHOTS = 2
NATURAL_LOSS_MINIMUM_REPLACEMENT_NATIVE_ADVANCE = 5
NATURAL_LOSS_STABLE_TERMINAL_RECEIPT_KIND = (
    "natural_loss_stable_terminal_v1"
)
NATURAL_LOSS_STABLE_TERMINAL_SECONDS = 10.0
NATURAL_LOSS_STABLE_TERMINAL_MINIMUM_SNAPSHOTS = 5


def is_natural_loss_mature_anchor(state: Mapping[str, Any]) -> bool:
    """Return whether a Board may anchor a later below-target clock reset."""

    try:
        native_game_time = int(state["native_game_time"])
        score = int(state["score"])
        score_target = int(state["score_target"])
    except (KeyError, TypeError, ValueError):
        return False
    return (
        native_game_time >= NATURAL_LOSS_MINIMUM_PRE_RESTART_NATIVE_TIME
        and score_target > 0
        and score < score_target
    )


def is_natural_loss_restart_clock_reset(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    unavailable_count: int,
) -> bool:
    """Recognize the immutable portion of a retail Board clock reset."""

    if isinstance(unavailable_count, bool) or unavailable_count <= 0:
        return False
    try:
        before_native = int(before["native_game_time"])
        after_native = int(after["native_game_time"])
        before_score = int(before["score"])
        before_target = int(before["score_target"])
    except (KeyError, TypeError, ValueError):
        return False
    return (
        before_score < before_target
        and before_native >= NATURAL_LOSS_MINIMUM_PRE_RESTART_NATIVE_TIME
        and 0
        <= after_native
        <= NATURAL_LOSS_MAXIMUM_POST_RESTART_NATIVE_TIME
        and before_native - after_native
        >= NATURAL_LOSS_MINIMUM_NATIVE_TIME_RESET
    )


def is_natural_loss_restart_transition(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    unavailable_count: int,
) -> bool:
    """Recognize the retail loss-to-automatic-restart transition.

    Revenge does not leave a lost Board quiescent for ten seconds.  It tears
    the Board down, deducts a life, and creates a replacement Board.  The
    reliable read-only signature is therefore a below-target mature Board,
    at least one unavailable observation during teardown, and a populated
    active replacement Board whose native clock has reset.
    """

    try:
        after_target = int(after["score_target"])
        after_population = sum(
            int(after[key])
            for key in (
                "active_ball_count",
                "inserting_ball_count",
                "pending_ball_count",
                "fired_ball_count",
            )
        )
    except (KeyError, TypeError, ValueError):
        return False
    return (
        is_natural_loss_restart_clock_reset(
            before,
            after,
            unavailable_count=unavailable_count,
        )
        and after_target > 0
        and after_population > 0
    )


def is_natural_loss_board_identity_replacement_state(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    unavailable_count: int,
) -> bool:
    """Recognize one coherent replacement Board with a new object identity."""

    try:
        before_address = before["board_address"]
        after_address = after["board_address"]
        if (
            isinstance(before_address, bool)
            or not isinstance(before_address, int)
            or isinstance(after_address, bool)
            or not isinstance(after_address, int)
        ):
            return False
    except KeyError:
        return False
    return (
        before_address > 0
        and after_address > 0
        and before_address != after_address
        and is_natural_loss_restart_transition(
            before,
            after,
            unavailable_count=unavailable_count,
        )
        and after.get("runtime_active") is False
        and after.get("curve_plan_exhausted") is False
    )


def is_natural_loss_board_replacement_transition(
    before: Mapping[str, Any],
    first_replacement: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    unavailable_count: int,
    replacement_snapshot_count: int,
) -> bool:
    """Recognize a stable source-observed Board-identity replacement.

    Retail destroys the Shooter while the lost Board is torn down, so a
    complete gameplay snapshot is unavailable across the transition.  The
    replacement is distinguished from the same-Board post-loss rollback by a
    positive, different Board pointer.  A second coherent snapshot and native
    clock progress make the identity observation stable rather than transient.
    """

    if (
        isinstance(replacement_snapshot_count, bool)
        or replacement_snapshot_count
        < NATURAL_LOSS_MINIMUM_REPLACEMENT_SNAPSHOTS
    ):
        return False
    if not (
        is_natural_loss_board_identity_replacement_state(
            before,
            first_replacement,
            unavailable_count=unavailable_count,
        )
        and is_natural_loss_board_identity_replacement_state(
            before,
            after,
            unavailable_count=unavailable_count,
        )
    ):
        return False
    try:
        first_native = int(first_replacement["native_game_time"])
        after_native = int(after["native_game_time"])
        first_target = int(first_replacement["score_target"])
        after_target = int(after["score_target"])
    except (KeyError, TypeError, ValueError):
        return False
    return (
        first_replacement["board_address"] == after["board_address"]
        and first_target > 0
        and first_target == after_target
        and after_native >= first_native
        and after_native - first_native
        >= NATURAL_LOSS_MINIMUM_REPLACEMENT_NATIVE_ADVANCE
    )


def _natural_loss_transition_state(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        key: state[key]
        for key in (
            "board_address",
            "native_game_time",
            "score",
            "score_target",
            "runtime_active",
            "curve_plan_exhausted",
            "loss_counter",
            "active_ball_count",
            "inserting_ball_count",
            "pending_ball_count",
            "fired_ball_count",
        )
    }


def _natural_loss_stable_terminal_state(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        key: state[key]
        for key in (
            "native_game_time",
            "score",
            "score_target",
            "runtime_active",
            "curve_plan_exhausted",
            "active_ball_count",
            "inserting_ball_count",
            "pending_ball_count",
            "fired_ball_count",
        )
    }


def _client_geometry(window_handle: int) -> tuple[int, int, int, int]:
    if os.name != "nt":
        raise RuntimeError("Win32 geometry requires Windows")
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.GetClientRect.argtypes = (
        wintypes.HWND,
        ctypes.POINTER(wintypes.RECT),
    )
    user32.GetClientRect.restype = wintypes.BOOL
    user32.ClientToScreen.argtypes = (
        wintypes.HWND,
        ctypes.POINTER(wintypes.POINT),
    )
    user32.ClientToScreen.restype = wintypes.BOOL
    rect = wintypes.RECT()
    origin = wintypes.POINT()
    if not user32.GetClientRect(
        wintypes.HWND(window_handle),
        ctypes.byref(rect),
    ) or not user32.ClientToScreen(
        wintypes.HWND(window_handle),
        ctypes.byref(origin),
    ):
        raise OSError(ctypes.get_last_error(), "client geometry unavailable")
    width = int(rect.right - rect.left)
    height = int(rect.bottom - rect.top)
    if width <= 0 or height <= 0:
        raise RuntimeError("window client area is empty")
    return int(origin.x), int(origin.y), width, height


def _ensure_dpi_awareness() -> None:
    global _DPI_AWARENESS_VERIFIED
    if _DPI_AWARENESS_VERIFIED:
        return
    if os.name != "nt":
        raise RuntimeError("DPI awareness requires Windows")
    shcore = ctypes.WinDLL("shcore", use_last_error=True)
    shcore.SetProcessDpiAwareness.argtypes = (ctypes.c_int,)
    shcore.SetProcessDpiAwareness.restype = ctypes.c_long
    shcore.GetProcessDpiAwareness.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(ctypes.c_int),
    )
    shcore.GetProcessDpiAwareness.restype = ctypes.c_long
    # E_ACCESSDENIED is expected if another imported helper set it first.
    shcore.SetProcessDpiAwareness(2)
    awareness = ctypes.c_int()
    if (
        shcore.GetProcessDpiAwareness(None, ctypes.byref(awareness)) != 0
        or awareness.value != 2
    ):
        raise RuntimeError("per-monitor DPI awareness is unavailable")
    _DPI_AWARENESS_VERIFIED = True


class _Win32ForegroundApi:
    """Narrow Win32 adapter for verified foreground acquisition."""

    def __init__(self) -> None:
        self.user32 = ctypes.WinDLL("user32", use_last_error=True)
        self.kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self.user32.GetForegroundWindow.argtypes = ()
        self.user32.GetForegroundWindow.restype = wintypes.HWND
        self.user32.ShowWindow.argtypes = (wintypes.HWND, ctypes.c_int)
        self.user32.ShowWindow.restype = wintypes.BOOL
        self.user32.BringWindowToTop.argtypes = (wintypes.HWND,)
        self.user32.BringWindowToTop.restype = wintypes.BOOL
        self.user32.SetForegroundWindow.argtypes = (wintypes.HWND,)
        self.user32.SetForegroundWindow.restype = wintypes.BOOL
        self.user32.GetWindowThreadProcessId.argtypes = (
            wintypes.HWND,
            ctypes.POINTER(wintypes.DWORD),
        )
        self.user32.GetWindowThreadProcessId.restype = wintypes.DWORD
        self.user32.AttachThreadInput.argtypes = (
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.BOOL,
        )
        self.user32.AttachThreadInput.restype = wintypes.BOOL
        self.kernel32.GetCurrentThreadId.argtypes = ()
        self.kernel32.GetCurrentThreadId.restype = wintypes.DWORD

    def foreground(self) -> int:
        return int(self.user32.GetForegroundWindow() or 0)

    def restore(self, window_handle: int) -> None:
        self.user32.ShowWindow(wintypes.HWND(window_handle), 9)

    def bring_to_top(self, window_handle: int) -> None:
        self.user32.BringWindowToTop(wintypes.HWND(window_handle))

    def request_foreground(self, window_handle: int) -> None:
        self.user32.SetForegroundWindow(wintypes.HWND(window_handle))

    def current_thread_id(self) -> int:
        return int(self.kernel32.GetCurrentThreadId())

    def window_thread_id(self, window_handle: int) -> int:
        process_id = wintypes.DWORD()
        return int(
            self.user32.GetWindowThreadProcessId(
                wintypes.HWND(window_handle),
                ctypes.byref(process_id),
            )
        )

    def attach_thread_input(
        self,
        source_thread_id: int,
        target_thread_id: int,
        attach: bool,
    ) -> bool:
        return bool(
            self.user32.AttachThreadInput(
                source_thread_id,
                target_thread_id,
                attach,
            )
        )


def _activate_window_with_api(
    window_handle: int,
    api: Any,
    *,
    timeout: float,
    monotonic: Any = time.monotonic,
    sleep: Any = time.sleep,
) -> dict[str, Any]:
    """Acquire and verify foreground ownership before any SendInput."""

    if (
        isinstance(window_handle, bool)
        or not isinstance(window_handle, int)
        or window_handle <= 0
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError("foreground activation target is invalid")
    started = float(monotonic())
    deadline = started + timeout
    foreground_before = int(api.foreground())
    attempts = 0
    fallback_used = False
    attached_thread_ids: list[int] = []

    def receipt(foreground_after: int) -> dict[str, Any]:
        return {
            "target_window_handle_hex": f"0x{window_handle:016x}",
            "foreground_before_hex": f"0x{foreground_before:016x}",
            "foreground_after_hex": f"0x{foreground_after:016x}",
            "verified": foreground_after == window_handle,
            "attach_fallback_used": fallback_used,
            "attached_thread_ids": attached_thread_ids.copy(),
            "activation_attempts": attempts,
            "elapsed_seconds": max(0.0, float(monotonic()) - started),
        }

    if foreground_before == window_handle:
        return receipt(foreground_before)

    api.restore(window_handle)
    current_thread_id = int(api.current_thread_id())
    try:
        while float(monotonic()) < deadline:
            attempts += 1
            api.bring_to_top(window_handle)
            api.request_foreground(window_handle)
            foreground_after = int(api.foreground())
            if foreground_after == window_handle:
                return receipt(foreground_after)

            if not fallback_used and float(monotonic()) - started >= 0.05:
                fallback_used = True
                candidate_thread_ids = (
                    int(api.window_thread_id(foreground_after))
                    if foreground_after > 0
                    else 0,
                    int(api.window_thread_id(window_handle)),
                )
                for thread_id in dict.fromkeys(candidate_thread_ids):
                    if thread_id <= 0 or thread_id == current_thread_id:
                        continue
                    if api.attach_thread_input(
                        current_thread_id,
                        thread_id,
                        True,
                    ):
                        attached_thread_ids.append(thread_id)
                api.bring_to_top(window_handle)
                api.request_foreground(window_handle)
                foreground_after = int(api.foreground())
                if foreground_after == window_handle:
                    return receipt(foreground_after)
            sleep(min(0.01, max(0.0, deadline - float(monotonic()))))
    finally:
        for thread_id in reversed(attached_thread_ids):
            api.attach_thread_input(
                current_thread_id,
                thread_id,
                False,
            )

    foreground_after = int(api.foreground())
    raise RuntimeError(
        "retail window did not become foreground before SendInput: "
        f"target=0x{window_handle:016x}, "
        f"observed=0x{foreground_after:016x}, attempts={attempts}, "
        f"fallback={fallback_used}"
    )


def _stabilize_window_foreground_with_api(
    window_handle: int,
    api: Any,
    *,
    continuous_seconds: float,
    timeout: float,
    monotonic: Any = time.monotonic,
    sleep: Any = time.sleep,
) -> dict[str, Any]:
    """Require one uninterrupted foreground interval after activation."""

    if (
        isinstance(window_handle, bool)
        or not isinstance(window_handle, int)
        or window_handle <= 0
        or not math.isfinite(continuous_seconds)
        or continuous_seconds <= 0
        or not math.isfinite(timeout)
        or timeout <= continuous_seconds
    ):
        raise ValueError("foreground stability contract is invalid")
    started = float(monotonic())
    deadline = started + timeout
    stable_since: float | None = None
    foreground_loss_count = 0
    reacquisition_count = 0

    while float(monotonic()) < deadline:
        now = float(monotonic())
        foreground = int(api.foreground())
        if foreground == window_handle:
            if stable_since is None:
                stable_since = now
            observed = now - stable_since
            if observed >= continuous_seconds:
                return {
                    "continuous_required_seconds": continuous_seconds,
                    "continuous_observed_seconds": observed,
                    "foreground_loss_count": foreground_loss_count,
                    "reacquisition_count": reacquisition_count,
                    "foreground_after_hex": f"0x{foreground:016x}",
                    "verified": True,
                    "elapsed_seconds": now - started,
                }
            sleep(
                min(
                    0.01,
                    continuous_seconds - observed,
                    max(0.0, deadline - now),
                )
            )
            continue

        foreground_loss_count += 1
        stable_since = None
        remaining = deadline - now
        if remaining <= 0:
            break
        reacquisition_count += 1
        _activate_window_with_api(
            window_handle,
            api,
            timeout=remaining,
            monotonic=monotonic,
            sleep=sleep,
        )

    foreground = int(api.foreground())
    raise RuntimeError(
        "retail window foreground was not continuously stable before "
        "SendInput: "
        f"target=0x{window_handle:016x}, "
        f"observed=0x{foreground:016x}, "
        f"required={continuous_seconds}, losses={foreground_loss_count}, "
        f"reacquisitions={reacquisition_count}"
    )


def _activate_window(window_handle: int, *, timeout: float = 2.0) -> dict[str, Any]:
    api = _Win32ForegroundApi()
    started = time.monotonic()
    acquisition = _activate_window_with_api(
        window_handle,
        api,
        timeout=timeout,
    )
    stability = _stabilize_window_foreground_with_api(
        window_handle,
        api,
        continuous_seconds=0.15,
        timeout=timeout,
    )
    return {
        **acquisition,
        "foreground_after_hex": stability["foreground_after_hex"],
        "verified": bool(
            acquisition["verified"] and stability["verified"]
        ),
        "continuous_required_seconds": stability[
            "continuous_required_seconds"
        ],
        "continuous_observed_seconds": stability[
            "continuous_observed_seconds"
        ],
        "foreground_loss_count": stability["foreground_loss_count"],
        "reacquisition_count": stability["reacquisition_count"],
        "total_elapsed_seconds": max(0.0, time.monotonic() - started),
    }


def send_mouse_click(
    window_handle: int,
    *,
    logical_x: float,
    logical_y: float,
    button: str,
    hold_seconds: float = 0.012,
) -> dict[str, Any]:
    """Send one ordinary left or right click at a logical canvas point."""

    if os.name != "nt":
        raise RuntimeError("SendInput requires Windows")
    if button not in {"left", "right"}:
        raise ValueError("button must be left or right")
    _ensure_dpi_awareness()
    left, top, width, height = _client_geometry(window_handle)
    viewport = logical_viewport(left, top, width, height)
    screen_x, screen_y = viewport.screen_point(logical_x, logical_y)

    class MouseInput(ctypes.Structure):
        _fields_ = (
            ("dx", wintypes.LONG),
            ("dy", wintypes.LONG),
            ("mouseData", wintypes.DWORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ctypes.c_size_t),
        )

    class InputUnion(ctypes.Union):
        _fields_ = (("mi", MouseInput),)

    class Input(ctypes.Structure):
        _anonymous_ = ("payload",)
        _fields_ = (
            ("type", wintypes.DWORD),
            ("payload", InputUnion),
        )

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.GetSystemMetrics.argtypes = (ctypes.c_int,)
    user32.GetSystemMetrics.restype = ctypes.c_int
    user32.SendInput.argtypes = (
        wintypes.UINT,
        ctypes.POINTER(Input),
        ctypes.c_int,
    )
    user32.SendInput.restype = wintypes.UINT
    virtual_left = user32.GetSystemMetrics(76)
    virtual_top = user32.GetSystemMetrics(77)
    virtual_width = user32.GetSystemMetrics(78)
    virtual_height = user32.GetSystemMetrics(79)
    if virtual_width <= 1 or virtual_height <= 1:
        raise RuntimeError("virtual desktop geometry is invalid")

    def send(flags: int, x: int = 0, y: int = 0) -> None:
        absolute = bool(flags & 0x8000)
        item = Input(
            type=0,
            payload=InputUnion(
                mi=MouseInput(
                    dx=(
                        round(
                            (x - virtual_left)
                            * 65535
                            / (virtual_width - 1)
                        )
                        if absolute
                        else 0
                    ),
                    dy=(
                        round(
                            (y - virtual_top)
                            * 65535
                            / (virtual_height - 1)
                        )
                        if absolute
                        else 0
                    ),
                    mouseData=0,
                    dwFlags=flags,
                    time=0,
                    dwExtraInfo=0,
                )
            ),
        )
        if user32.SendInput(1, ctypes.byref(item), ctypes.sizeof(Input)) != 1:
            raise OSError(ctypes.get_last_error(), "SendInput failed")

    activation = _activate_window(window_handle)
    # MOVE | MOVE_NOCOALESCE | VIRTUALDESK | ABSOLUTE
    send(0x0001 | 0x2000 | 0x4000 | 0x8000, screen_x, screen_y)
    if button == "left":
        down_flag, up_flag = 0x0002, 0x0004
    else:
        down_flag, up_flag = 0x0008, 0x0010
    send(down_flag)
    try:
        time.sleep(hold_seconds)
    finally:
        send(up_flag)
    return {
        "button": button,
        "logical_x": logical_x,
        "logical_y": logical_y,
        "screen_x": screen_x,
        "screen_y": screen_y,
        "client_region": [left, top, left + width, top + height],
        "viewport": {
            "screen_left": viewport.screen_left,
            "screen_top": viewport.screen_top,
            "scale": viewport.scale,
        },
        "activation": activation,
    }


def autoplay(
    pid: int,
    *,
    output_path: Path,
    maximum_seconds: float,
    shot_interval_seconds: float,
    swap_delay_seconds: float,
    idle_poll_seconds: float,
    expected_outcome: str = "natural_win",
    gameplay_policy: str = "autoplay",
    fruit_policy: str = "ignore",
) -> dict[str, Any]:
    if os.name != "nt":
        raise RuntimeError("live retail autoplay requires Windows")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite: {output_path}")
    if (
        maximum_seconds <= 0
        or shot_interval_seconds <= 0
        or swap_delay_seconds < 0
        or idle_poll_seconds <= 0
        or expected_outcome not in {"natural_loss", "natural_win"}
        or gameplay_policy not in {"autoplay", "idle"}
        or fruit_policy not in {"ignore", "collect", "bomb"}
    ):
        raise ValueError("autoplay timing arguments are invalid")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _ensure_dpi_awareness()
    window_handle = main_window_for_pid(pid)
    _activate_window(window_handle)
    started = time.monotonic()
    next_shot_at = started
    action_count = 0
    fruit_shot_count = 0
    swap_count = 0
    unavailable_count = 0
    final_state: Mapping[str, Any] | None = None
    outcome = "timeout"
    last_stable_state: Mapping[str, Any] | None = None
    last_mature_below_target_state: Mapping[str, Any] | None = None
    gap_before_state: Mapping[str, Any] | None = None
    gap_unavailable_count = 0
    gap_first_perf_counter_ns: int | None = None
    restart_candidate_before: Mapping[str, Any] | None = None
    restart_candidate_unavailable_count = 0
    restart_candidate_first_gap_ns: int | None = None
    restart_candidate_clock_reset_ns: int | None = None
    restart_candidate_first_post_reset_state: dict[str, Any] | None = None
    restart_zero_target_teardown_ns: int | None = None
    restart_zero_target_teardown_state: dict[str, Any] | None = None
    restart_replacement_first_seen_ns: int | None = None
    restart_replacement_first_state: dict[str, Any] | None = None
    restart_replacement_snapshot_count = 0
    restart_replacement_identity_consistent = True
    stable_loss_state: dict[str, Any] | None = None
    stable_loss_since: float | None = None
    stable_loss_first_perf_counter_ns: int | None = None
    stable_loss_snapshot_count = 0
    last_relaxed_board_phase: dict[str, Any] | None = None
    terminal_observation: dict[str, Any] | None = None
    last_fruit_bomb_observation_key: tuple[int, int] | None = None

    with output_path.open("x", encoding="utf-8", newline="\n") as stream:
        header = {
            "schema": "zuma-rl.live-retail-autoplay-log",
            "version": 1,
            "classification": "read-only-memory-assisted-input-control",
            "process_id": pid,
            "window_handle_hex": f"0x{window_handle:016x}",
            "process_memory_access": [
                "PROCESS_QUERY_INFORMATION",
                "PROCESS_VM_READ",
            ],
            "process_memory_writes": 0,
            "input_transport": "Win32 SendInput",
            "expected_outcome": expected_outcome,
            "gameplay_policy": gameplay_policy,
            "fruit_policy": fruit_policy,
            "started_perf_counter_ns": time.perf_counter_ns(),
        }
        stream.write(json.dumps(header, sort_keys=True) + "\n")
        stream.flush()

        while time.monotonic() - started < maximum_seconds:
            try:
                state = read_live_board(
                    pid,
                    include_fruit_target=fruit_policy in {"collect", "bomb"},
                )
            except LiveBoardUnavailable as error:
                unavailable_perf_counter_ns = time.perf_counter_ns()
                if gap_unavailable_count == 0:
                    gap_before_state = last_stable_state
                    gap_first_perf_counter_ns = unavailable_perf_counter_ns
                gap_unavailable_count += 1
                try:
                    terminal_state = read_live_board(
                        pid,
                        require_shooter=False,
                    )
                except LiveBoardUnavailable:
                    terminal_state = None
                if terminal_state is not None:
                    final_state = terminal_state
                    relaxed_board_phase = {
                        key: terminal_state[key]
                        for key in (
                            "board_address",
                            "native_game_time",
                            "score",
                            "displayed_score",
                            "score_target",
                            "runtime_active",
                            "curve_plan_exhausted",
                            "loss_counter",
                            "active_ball_count",
                            "inserting_ball_count",
                            "pending_ball_count",
                            "fired_ball_count",
                        )
                    }
                    if relaxed_board_phase != last_relaxed_board_phase:
                        stream.write(
                            json.dumps(
                                {
                                    "type": "relaxed_board_phase",
                                    "perf_counter_ns": (
                                        unavailable_perf_counter_ns
                                    ),
                                    "natural_win_predicate": (
                                        is_natural_win_state(terminal_state)
                                    ),
                                    "natural_loss_predicate": (
                                        is_natural_loss_state(terminal_state)
                                    ),
                                    "state": relaxed_board_phase,
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        )
                        stream.flush()
                        last_relaxed_board_phase = relaxed_board_phase
                    if (
                        restart_candidate_before is None
                        and is_natural_loss_mature_anchor(terminal_state)
                    ):
                        last_mature_below_target_state = terminal_state
                    if is_natural_win_state(terminal_state):
                        outcome = "natural_win"
                        break
                    if (
                        expected_outcome == "natural_loss"
                        and gameplay_policy == "idle"
                        and is_natural_loss_state(terminal_state)
                    ):
                        observed_stable_state = (
                            _natural_loss_stable_terminal_state(
                                terminal_state
                            )
                        )
                        if observed_stable_state != stable_loss_state:
                            stable_loss_state = observed_stable_state
                            stable_loss_since = time.monotonic()
                            stable_loss_first_perf_counter_ns = (
                                unavailable_perf_counter_ns
                            )
                            stable_loss_snapshot_count = 1
                        else:
                            stable_loss_snapshot_count += 1
                        assert stable_loss_since is not None
                        if (
                            time.monotonic() - stable_loss_since
                            >= NATURAL_LOSS_STABLE_TERMINAL_SECONDS
                            and stable_loss_snapshot_count
                            >= NATURAL_LOSS_STABLE_TERMINAL_MINIMUM_SNAPSHOTS
                        ):
                            outcome = "natural_loss"
                            assert (
                                stable_loss_first_perf_counter_ns
                                is not None
                            )
                            terminal_observation = {
                                "kind": (
                                    NATURAL_LOSS_STABLE_TERMINAL_RECEIPT_KIND
                                ),
                                "outcome": outcome,
                                "first_seen_perf_counter_ns": (
                                    stable_loss_first_perf_counter_ns
                                ),
                                "confirmed_perf_counter_ns": (
                                    time.perf_counter_ns()
                                ),
                                "snapshot_count": (
                                    stable_loss_snapshot_count
                                ),
                                "state": stable_loss_state,
                            }
                            break
                    else:
                        stable_loss_state = None
                        stable_loss_since = None
                        stable_loss_first_perf_counter_ns = None
                        stable_loss_snapshot_count = 0
                    if (
                        expected_outcome == "natural_loss"
                        and gameplay_policy == "idle"
                        and restart_candidate_before is None
                        and last_mature_below_target_state is not None
                        and is_natural_loss_restart_clock_reset(
                            last_mature_below_target_state,
                            terminal_state,
                            unavailable_count=gap_unavailable_count,
                        )
                    ):
                        assert gap_first_perf_counter_ns is not None
                        restart_candidate_before = (
                            last_mature_below_target_state
                        )
                        restart_candidate_unavailable_count = (
                            gap_unavailable_count
                        )
                        restart_candidate_first_gap_ns = (
                            gap_first_perf_counter_ns
                        )
                        restart_candidate_clock_reset_ns = (
                            unavailable_perf_counter_ns
                        )
                        restart_candidate_first_post_reset_state = (
                            _natural_loss_transition_state(terminal_state)
                        )
                        stream.write(
                            json.dumps(
                                {
                                    "type": (
                                        "natural_loss_board_replacement_candidate"
                                    ),
                                    "kind": (
                                        NATURAL_LOSS_BOARD_IDENTITY_REPLACEMENT_RECEIPT_KIND
                                    ),
                                    "perf_counter_ns": (
                                        restart_candidate_clock_reset_ns
                                    ),
                                    "observation_mode": (
                                        "read_live_board(require_shooter=false)"
                                    ),
                                    "board_unavailable_count": (
                                        restart_candidate_unavailable_count
                                    ),
                                    "before_restart": (
                                        _natural_loss_transition_state(
                                            restart_candidate_before
                                        )
                                    ),
                                    "first_post_reset_state": (
                                        restart_candidate_first_post_reset_state
                                    ),
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        )
                        stream.flush()
                    if (
                        restart_candidate_before is not None
                        and restart_zero_target_teardown_ns is None
                        and int(terminal_state["score_target"]) == 0
                        and int(terminal_state["native_game_time"])
                        <= NATURAL_LOSS_MAXIMUM_POST_RESTART_NATIVE_TIME
                    ):
                        restart_zero_target_teardown_ns = (
                            unavailable_perf_counter_ns
                        )
                        restart_zero_target_teardown_state = (
                            _natural_loss_transition_state(terminal_state)
                        )
                        stream.write(
                            json.dumps(
                                {
                                    "type": (
                                        "natural_loss_zero_target_teardown"
                                    ),
                                    "kind": (
                                        NATURAL_LOSS_BOARD_IDENTITY_REPLACEMENT_RECEIPT_KIND
                                    ),
                                    "perf_counter_ns": (
                                        restart_zero_target_teardown_ns
                                    ),
                                    "state": (
                                        restart_zero_target_teardown_state
                                    ),
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        )
                        stream.flush()
                    if (
                        restart_candidate_before is not None
                        and restart_replacement_first_state is None
                        and is_natural_loss_board_identity_replacement_state(
                            restart_candidate_before,
                            terminal_state,
                            unavailable_count=(
                                restart_candidate_unavailable_count
                            ),
                        )
                    ):
                        restart_replacement_first_seen_ns = (
                            unavailable_perf_counter_ns
                        )
                        restart_replacement_first_state = (
                            _natural_loss_transition_state(terminal_state)
                        )
                        restart_replacement_snapshot_count = 1
                        stream.write(
                            json.dumps(
                                {
                                    "type": (
                                        "natural_loss_board_identity_replacement_candidate"
                                    ),
                                    "kind": (
                                        NATURAL_LOSS_BOARD_IDENTITY_REPLACEMENT_RECEIPT_KIND
                                    ),
                                    "perf_counter_ns": (
                                        restart_replacement_first_seen_ns
                                    ),
                                    "state": restart_replacement_first_state,
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        )
                        stream.flush()
                    elif (
                        restart_candidate_before is not None
                        and restart_replacement_first_state is not None
                    ):
                        if (
                            terminal_state["board_address"]
                            != restart_replacement_first_state[
                                "board_address"
                            ]
                        ):
                            restart_replacement_identity_consistent = False
                        elif is_natural_loss_board_identity_replacement_state(
                            restart_candidate_before,
                            terminal_state,
                            unavailable_count=(
                                restart_candidate_unavailable_count
                            ),
                        ):
                            restart_replacement_snapshot_count += 1
                    if (
                        expected_outcome == "natural_loss"
                        and gameplay_policy == "idle"
                        and restart_candidate_before is not None
                        and restart_replacement_first_state is not None
                        and restart_replacement_identity_consistent
                        and is_natural_loss_board_replacement_transition(
                            restart_candidate_before,
                            restart_replacement_first_state,
                            terminal_state,
                            unavailable_count=(
                                restart_candidate_unavailable_count
                            ),
                            replacement_snapshot_count=(
                                restart_replacement_snapshot_count
                            ),
                        )
                    ):
                        outcome = "natural_loss"
                        assert restart_candidate_first_gap_ns is not None
                        assert restart_candidate_clock_reset_ns is not None
                        assert restart_candidate_first_post_reset_state is not None
                        assert restart_replacement_first_seen_ns is not None
                        before_restart = _natural_loss_transition_state(
                            restart_candidate_before
                        )
                        after_restart = _natural_loss_transition_state(
                            terminal_state
                        )
                        terminal_observation = {
                            "kind": (
                                NATURAL_LOSS_BOARD_IDENTITY_REPLACEMENT_RECEIPT_KIND
                            ),
                            "outcome": outcome,
                            "observation_mode": (
                                "read_live_board(require_shooter=false)"
                            ),
                            "shooter_required": False,
                            "first_gap_perf_counter_ns": (
                                restart_candidate_first_gap_ns
                            ),
                            "clock_reset_perf_counter_ns": (
                                restart_candidate_clock_reset_ns
                            ),
                            "zero_target_teardown_perf_counter_ns": (
                                restart_zero_target_teardown_ns
                            ),
                            "board_identity_replacement_first_seen_perf_counter_ns": (
                                restart_replacement_first_seen_ns
                            ),
                            "confirmed_perf_counter_ns": (
                                time.perf_counter_ns()
                            ),
                            "board_unavailable_count": (
                                gap_unavailable_count
                            ),
                            "clock_reset_unavailable_count": (
                                restart_candidate_unavailable_count
                            ),
                            "native_game_time_reset": (
                                int(before_restart["native_game_time"])
                                - int(after_restart["native_game_time"])
                            ),
                            "replacement_confirmation_snapshot_count": (
                                restart_replacement_snapshot_count
                            ),
                            "replacement_native_game_time_advance": (
                                int(after_restart["native_game_time"])
                                - int(
                                    restart_replacement_first_state[
                                        "native_game_time"
                                    ]
                                )
                            ),
                            "before_restart": before_restart,
                            "first_post_reset_state": (
                                restart_candidate_first_post_reset_state
                            ),
                            "zero_target_teardown_state": (
                                restart_zero_target_teardown_state
                            ),
                            "first_replacement_state": (
                                restart_replacement_first_state
                            ),
                            "after_restart": after_restart,
                        }
                        break
                    if (
                        int(terminal_state["native_game_time"])
                        >= int(
                            (gap_before_state or terminal_state)[
                                "native_game_time"
                            ]
                        )
                        and int(terminal_state["score"])
                        < int(terminal_state["score_target"])
                    ):
                        gap_before_state = terminal_state
                unavailable_count += 1
                stream.write(
                    json.dumps(
                        {
                            "type": "board_unavailable",
                            "perf_counter_ns": unavailable_perf_counter_ns,
                            "reason": str(error),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                stream.flush()
                time.sleep(idle_poll_seconds)
                continue
            final_state = state
            if (
                restart_candidate_before is None
                and is_natural_loss_mature_anchor(state)
            ):
                last_mature_below_target_state = state
            if is_natural_win_state(state):
                outcome = "natural_win"
                break
            stable_loss_state = None
            stable_loss_since = None
            stable_loss_first_perf_counter_ns = None
            stable_loss_snapshot_count = 0
            if (
                expected_outcome == "natural_loss"
                and gameplay_policy == "idle"
                and restart_candidate_before is None
                and last_mature_below_target_state is not None
                and is_natural_loss_restart_clock_reset(
                    last_mature_below_target_state,
                    state,
                    unavailable_count=gap_unavailable_count,
                )
            ):
                assert gap_first_perf_counter_ns is not None
                restart_candidate_before = last_mature_below_target_state
                restart_candidate_unavailable_count = (
                    gap_unavailable_count
                )
                restart_candidate_first_gap_ns = (
                    gap_first_perf_counter_ns
                )
                restart_candidate_clock_reset_ns = time.perf_counter_ns()
                restart_candidate_first_post_reset_state = (
                    _natural_loss_transition_state(state)
                )
                stream.write(
                    json.dumps(
                        {
                            "type": (
                                "natural_loss_board_replacement_candidate"
                            ),
                            "kind": (
                                NATURAL_LOSS_BOARD_IDENTITY_REPLACEMENT_RECEIPT_KIND
                            ),
                            "perf_counter_ns": (
                                restart_candidate_clock_reset_ns
                            ),
                            "observation_mode": (
                                "read_live_board(require_shooter=true)"
                            ),
                            "board_unavailable_count": (
                                restart_candidate_unavailable_count
                            ),
                            "before_restart": (
                                _natural_loss_transition_state(
                                    restart_candidate_before
                                )
                            ),
                            "first_post_reset_state": (
                                restart_candidate_first_post_reset_state
                            ),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                stream.flush()
            replacement_state: Mapping[str, Any] | None = None
            replacement_state_perf_counter_ns: int | None = None
            if (
                expected_outcome == "natural_loss"
                and gameplay_policy == "idle"
                and restart_candidate_before is not None
            ):
                try:
                    replacement_state = read_live_board(
                        pid,
                        require_shooter=False,
                    )
                    replacement_state_perf_counter_ns = time.perf_counter_ns()
                except LiveBoardUnavailable:
                    replacement_state = None
            if (
                restart_candidate_before is not None
                and replacement_state is not None
                and restart_replacement_first_state is None
                and is_natural_loss_board_identity_replacement_state(
                    restart_candidate_before,
                    replacement_state,
                    unavailable_count=(
                        restart_candidate_unavailable_count
                    ),
                )
            ):
                assert replacement_state_perf_counter_ns is not None
                restart_replacement_first_seen_ns = (
                    replacement_state_perf_counter_ns
                )
                restart_replacement_first_state = (
                    _natural_loss_transition_state(replacement_state)
                )
                restart_replacement_snapshot_count = 1
                stream.write(
                    json.dumps(
                        {
                            "type": (
                                "natural_loss_board_identity_replacement_candidate"
                            ),
                            "kind": (
                                NATURAL_LOSS_BOARD_IDENTITY_REPLACEMENT_RECEIPT_KIND
                            ),
                            "perf_counter_ns": (
                                restart_replacement_first_seen_ns
                            ),
                            "state": restart_replacement_first_state,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                stream.flush()
            elif (
                restart_candidate_before is not None
                and replacement_state is not None
                and restart_replacement_first_state is not None
            ):
                if (
                    replacement_state["board_address"]
                    != restart_replacement_first_state["board_address"]
                ):
                    restart_replacement_identity_consistent = False
                elif is_natural_loss_board_identity_replacement_state(
                    restart_candidate_before,
                    replacement_state,
                    unavailable_count=(
                        restart_candidate_unavailable_count
                    ),
                ):
                    restart_replacement_snapshot_count += 1
            if (
                expected_outcome == "natural_loss"
                and gameplay_policy == "idle"
                and restart_candidate_before is not None
                and replacement_state is not None
                and restart_replacement_first_state is not None
                and restart_replacement_identity_consistent
                and is_natural_loss_board_replacement_transition(
                    restart_candidate_before,
                    restart_replacement_first_state,
                    replacement_state,
                    unavailable_count=(
                        restart_candidate_unavailable_count
                    ),
                    replacement_snapshot_count=(
                        restart_replacement_snapshot_count
                    ),
                )
            ):
                outcome = "natural_loss"
                assert restart_candidate_first_gap_ns is not None
                assert restart_candidate_clock_reset_ns is not None
                assert restart_candidate_first_post_reset_state is not None
                assert restart_replacement_first_seen_ns is not None
                final_state = replacement_state
                before_restart = _natural_loss_transition_state(
                    restart_candidate_before
                )
                after_restart = _natural_loss_transition_state(
                    replacement_state
                )
                terminal_observation = {
                    "kind": (
                        NATURAL_LOSS_BOARD_IDENTITY_REPLACEMENT_RECEIPT_KIND
                    ),
                    "outcome": outcome,
                    "observation_mode": (
                        "read_live_board(require_shooter=false)"
                    ),
                    "shooter_required": False,
                    "first_gap_perf_counter_ns": (
                        restart_candidate_first_gap_ns
                    ),
                    "clock_reset_perf_counter_ns": (
                        restart_candidate_clock_reset_ns
                    ),
                    "zero_target_teardown_perf_counter_ns": (
                        restart_zero_target_teardown_ns
                    ),
                    "board_identity_replacement_first_seen_perf_counter_ns": (
                        restart_replacement_first_seen_ns
                    ),
                    "confirmed_perf_counter_ns": time.perf_counter_ns(),
                    "board_unavailable_count": (
                        restart_candidate_unavailable_count
                    ),
                    "clock_reset_unavailable_count": (
                        restart_candidate_unavailable_count
                    ),
                    "native_game_time_reset": (
                        int(before_restart["native_game_time"])
                        - int(after_restart["native_game_time"])
                    ),
                    "replacement_confirmation_snapshot_count": (
                        restart_replacement_snapshot_count
                    ),
                    "replacement_native_game_time_advance": (
                        int(after_restart["native_game_time"])
                        - int(
                            restart_replacement_first_state[
                                "native_game_time"
                            ]
                        )
                    ),
                    "before_restart": before_restart,
                    "first_post_reset_state": (
                        restart_candidate_first_post_reset_state
                    ),
                    "zero_target_teardown_state": (
                        restart_zero_target_teardown_state
                    ),
                    "first_replacement_state": (
                        restart_replacement_first_state
                    ),
                    "after_restart": after_restart,
                }
                break
            if (
                restart_candidate_before is not None
                and int(state["native_game_time"])
                > NATURAL_LOSS_MAXIMUM_POST_RESTART_NATIVE_TIME
            ):
                stream.write(
                    json.dumps(
                        {
                            "type": "natural_loss_restart_candidate_expired",
                            "perf_counter_ns": time.perf_counter_ns(),
                            "native_game_time": state["native_game_time"],
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                stream.flush()
                restart_candidate_before = None
                restart_candidate_unavailable_count = 0
                restart_candidate_first_gap_ns = None
                restart_candidate_clock_reset_ns = None
                restart_candidate_first_post_reset_state = None
                restart_zero_target_teardown_ns = None
                restart_zero_target_teardown_state = None
                restart_replacement_first_seen_ns = None
                restart_replacement_first_state = None
                restart_replacement_snapshot_count = 0
                restart_replacement_identity_consistent = True
                last_mature_below_target_state = None
            last_stable_state = state
            gap_before_state = None
            gap_unavailable_count = 0
            gap_first_perf_counter_ns = None
            if fruit_policy == "bomb":
                observation = fruit_bomb_observation(state)
                if observation is not None:
                    observation_key = (
                        int(state["board_address"]),
                        int(state["native_game_time"]),
                    )
                    if observation_key != last_fruit_bomb_observation_key:
                        observation["perf_counter_ns"] = time.perf_counter_ns()
                        stream.write(
                            json.dumps(observation, sort_keys=True) + "\n"
                        )
                        stream.flush()
                        last_fruit_bomb_observation_key = observation_key
            now = time.monotonic()
            if gameplay_policy == "idle":
                time.sleep(idle_poll_seconds)
                continue
            if (
                now < next_shot_at
                or int(state["fired_ball_count"]) > 0
                or int(state["inserting_ball_count"]) > 0
            ):
                time.sleep(idle_poll_seconds)
                continue
            recommendation = choose_shot(
                state,
                fruit_policy=fruit_policy,
            )
            event: dict[str, Any] = {
                "type": "decision",
                "perf_counter_ns": time.perf_counter_ns(),
                "native_game_time": state["native_game_time"],
                "score": state["score"],
                "score_target": state["score_target"],
                "active_ball_count": state["active_ball_count"],
                "pending_ball_count": state["pending_ball_count"],
                "current_ball_id": state["current"]["ball_id"],
                "current_color_id": state["current"]["color_id"],
                "next_ball_id": state["next"]["ball_id"],
                "next_color_id": state["next"]["color_id"],
                "recommendation": recommendation,
            }
            if recommendation["action"] == "wait":
                event["executed"] = False
                stream.write(json.dumps(event, sort_keys=True) + "\n")
                stream.flush()
                next_shot_at = now + shot_interval_seconds
                time.sleep(idle_poll_seconds)
                continue
            if recommendation["action"] == "swap_then_fire":
                event["swap_input"] = send_mouse_click(
                    window_handle,
                    logical_x=LOGICAL_WIDTH / 2,
                    logical_y=LOGICAL_HEIGHT / 2,
                    button="right",
                )
                swap_count += 1
                time.sleep(swap_delay_seconds)
            event["fire_input"] = send_mouse_click(
                window_handle,
                logical_x=float(recommendation["target_x"]),
                logical_y=float(recommendation["target_y"]),
                button="left",
            )
            event["executed"] = True
            action_count += 1
            if recommendation.get("reason") == "collect_fruit":
                fruit_shot_count += 1
            stream.write(json.dumps(event, sort_keys=True) + "\n")
            stream.flush()
            next_shot_at = time.monotonic() + shot_interval_seconds

        footer = {
            "type": "result",
            "status": (
                "PASS" if outcome == expected_outcome else "INCOMPLETE"
            ),
            "expected_outcome": expected_outcome,
            "gameplay_policy": gameplay_policy,
            "fruit_policy": fruit_policy,
            "outcome": outcome,
            "process_id": pid,
            "action_count": action_count,
            "fruit_shot_count": fruit_shot_count,
            "swap_count": swap_count,
            "board_unavailable_count": unavailable_count,
            "elapsed_seconds": time.monotonic() - started,
            "finished_perf_counter_ns": time.perf_counter_ns(),
            "terminal_observation": terminal_observation,
            "final_state": (
                {
                    key: final_state[key]
                    for key in (
                        "board_address",
                        "native_game_time",
                        "score",
                        "displayed_score",
                        "score_target",
                        "runtime_active",
                        "loss_counter",
                        "curve_plan_exhausted",
                        "active_ball_count",
                        "inserting_ball_count",
                        "pending_ball_count",
                        "fired_ball_count",
                    )
                }
                if final_state is not None
                else None
            ),
        }
        stream.write(json.dumps(footer, sort_keys=True) + "\n")
        stream.flush()
    return footer


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-seconds", type=float, default=900.0)
    parser.add_argument("--shot-interval-seconds", type=float, default=0.16)
    parser.add_argument("--swap-delay-seconds", type=float, default=0.04)
    parser.add_argument("--idle-poll-seconds", type=float, default=0.01)
    parser.add_argument(
        "--expected-outcome",
        choices=("natural_loss", "natural_win"),
        default="natural_win",
    )
    parser.add_argument(
        "--gameplay-policy",
        choices=("autoplay", "idle"),
        default="autoplay",
    )
    parser.add_argument(
        "--fruit-policy",
        choices=("ignore", "collect", "bomb"),
        default="ignore",
        help=(
            "Optionally prioritize either an ordinary projectile fruit "
            "shot or a nearby active-chain proximity bomb."
        ),
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = autoplay(
        args.pid,
        output_path=args.output.resolve(),
        maximum_seconds=args.maximum_seconds,
        shot_interval_seconds=args.shot_interval_seconds,
        swap_delay_seconds=args.swap_delay_seconds,
        idle_poll_seconds=args.idle_poll_seconds,
        expected_outcome=args.expected_outcome,
        gameplay_policy=args.gameplay_policy,
        fruit_policy=args.fruit_policy,
    )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
