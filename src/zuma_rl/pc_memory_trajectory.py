"""Validate and analyze frozen per-update retail Zuma memory trajectories."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import struct
from typing import Any, Mapping, Sequence


TRAJECTORY_SCHEMA = "zuma-rl.pc-memory-trajectory"
TRAJECTORY_VERSION = 2
TRAJECTORY_TICK_SCHEMA = "zuma-rl.pc-memory-trajectory-tick"
TRAJECTORY_TICK_VERSION = 2
LEGACY_TRAJECTORY_VERSION = 1
LEGACY_TRAJECTORY_TICK_VERSION = 1
LEGACY_TRAJECTORY_SAMPLE_PHASE = "frozen_post_framework_update"
TRAJECTORY_ANALYSIS_SCHEMA = "zuma-rl.pc-memory-trajectory-analysis"
TRAJECTORY_ANALYSIS_VERSION = 1
TRAJECTORY_SAMPLE_PHASE = "frozen_post_replay_update_barrier"
INITIAL_SAMPLE_BARRIER = "freeze_multiplier_message"
STEPPED_SAMPLE_BARRIER = "replay_fast_forward_step"
# These summaries were added to the already-published v2 index format when
# the active-board probe grew from v1 to v2.  Historical v2 indexes bind the
# complete tick artifact by SHA-256 but legitimately do not repeat these
# fields in their compact index row.
_V2_EXTENDED_INDEX_SUMMARY_FIELDS = frozenset(
    {
        "active_board_version",
        "board_mode_flag_1064",
        "curve_plan_exhausted",
        "post_zuma_timer_remaining",
        "post_zuma_ramp_404",
        "post_zuma_ramp_408",
        "curve_plans",
    }
)
_V2_EXTENDED_INDEX_MARKERS = frozenset(
    {"evidence_classification", "freeze_update", "process_identity"}
)
ACTIVE_BOARD_SCHEMA = "zuma-rl.pc-active-board-probe"
ACTIVE_BOARD_LEGACY_VERSION = 1
ACTIVE_BOARD_VERSION = 2
BOARD_OBJECT_SIZE = 0x1100
BOARD_VTABLE = 0x0096356C
BOARD_EMBEDDED_VTABLE_OFFSET = 0x88
BOARD_EMBEDDED_VTABLE = 0x0096368C
BOARD_SCORE_OFFSET = 0x104
BOARD_SCORE_TARGET_OFFSET = 0x108
BOARD_DISPLAYED_SCORE_OFFSET = 0xEFC
BOARD_MODE_FLAG_1064_OFFSET = 0x1064
G_CURVE_PLAN_EXHAUSTED_ADDRESS = 0x009E8252
ACTIVE_CHAIN_LIST_OFFSET = 0x5C
INSERTION_STAGING_LIST_OFFSET = 0x50
QRAND_OBJECT_SIZE = 0x48
QRAND_BOARD_POINTER_OFFSET = 0x7A8
QRAND_VECTOR_LAYOUT = (
    (0x08, "weights", "float32"),
    (0x18, "sways", "float32"),
    (0x28, "last_hit", "int32"),
    (0x38, "previous_hit", "int32"),
)
QRAND_VECTOR_LENGTH = 6
MTRAND_STATE_WORDS = 624
MTRAND_STATE_BYTES = (MTRAND_STATE_WORDS + 1) * 4
POWERUP_TYPE_COUNT = 14
POWERUP_NONE_TYPE = 14
BALL_OBJECT_SIZE = 0x134
BULLET_OBJECT_SIZE = 0x18C
BULLET_GAP_SENTINEL_POINTER_OFFSET = 0x174
BULLET_GAP_COUNT_OFFSET = 0x178
BULLET_CURVE_POINTS_OFFSET = 0x17C
BULLET_CURVE_POINT_COUNT = 4
RETAIL_DEBUG_FILL_I32 = -1163005939  # Signed 0xBAADF00D.
BULLET_GAP_NODE_SIZE = 0x14
MAX_BULLET_GAP_ENTRIES = 64
CURVE_OBJECT_SIZE = 0x200
CURVE_VTABLE = 0x009641F0
CURVE_PLANNED_VECTOR_BEGIN_OFFSET = 0x34
CURVE_PLANNED_VECTOR_END_OFFSET = 0x38
CURVE_PLANNED_VECTOR_CAPACITY_OFFSET = 0x3C
CURVE_PLANNED_ITEM_SIZE = 0x14
CURVE_ADD_PLAN_ENABLED_OFFSET = 0x1A3
# Native Curve runtime fields confirmed against popcapgame1.exe:
# AdvanceBalls reads/writes +0x190, the persistent first-chain endpoint is
# committed at +0x19C, and AdvanceBackwardBalls writes the entrance stop
# timer at +0x17C and the prior-tick backwards-motion latch at +0x1A0.  The
# three adjacent byte latches at +0x1BE..+0x1C0 are stop-adding,
# cruising-speed reached, and rollout reached respectively.
CURVE_STOP_TIME_OFFSET = 0x17C
CURVE_ADVANCE_SPEED_OFFSET = 0x190
CURVE_FIRST_CHAIN_END_OFFSET = 0x19C
CURVE_FIRST_BALL_MOVED_BACKWARDS_OFFSET = 0x1A0
CURVE_STOP_ADDING_OFFSET = 0x1BE
CURVE_HAS_REACHED_CRUISING_SPEED_OFFSET = 0x1BF
CURVE_HAS_REACHED_ROLLOUT_OFFSET = 0x1C0
MAX_CURVE_PLANNED_ITEMS = 4096
CURVE_MANAGER_OBJECT_SIZE = 0x40C
CURVE_MANAGER_VTABLE = 0x0096A204
CURVE_MANAGER_CURVE_ARRAY_OFFSET = 0x16C
CURVE_MANAGER_CURVE_COUNT_OFFSET = 0x360
CURVE_MANAGER_POST_ZUMA_TIMER_OFFSET = 0x400
CURVE_MANAGER_POST_ZUMA_RAMP_404_OFFSET = 0x404
CURVE_MANAGER_POST_ZUMA_RAMP_408_OFFSET = 0x408
BOARD_NATIVE_GAME_TIME_OFFSET = 0xEC8
BOARD_UPDATE_COUNT_OFFSET = 0x028
BOARD_RUNTIME_FLAG_157_OFFSET = 0x157
BOARD_RUNTIME_I32_F54_OFFSET = 0xF54
BOARD_FRUIT_ACTIVE_POINT_POINTER_OFFSET = 0x0B4
BOARD_FRUIT_LOWER_BOUND_OFFSET = 0x130
BOARD_FRUIT_UPPER_BOUND_OFFSET = 0x134
BOARD_FRUIT_SELECTED_POINT_OFFSET = 0x118
BOARD_FRUIT_COLLECTING_OFFSET = 0x153
BOARD_FRUIT_VELOCITY_OFFSET = 0xE90
BOARD_FRUIT_MAX_VELOCITY_OFFSET = 0xE94
BOARD_FRUIT_ACCELERATION_OFFSET = 0xE98
BOARD_FRUIT_VERTICAL_OFFSET = 0xE9C
BOARD_FRUIT_GLOW_ALPHA_OFFSET = 0xEAC
BOARD_FRUIT_GLOW_STEP_OFFSET = 0xEB0
BOARD_FRUIT_ALPHA_OFFSET = 0xEB4
BOARD_FRUIT_EXPIRY_TIME_OFFSET = 0xEB8
BOARD_FRUIT_CELL_INDEX_OFFSET = 0xEBC


class PcMemoryTrajectoryError(ValueError):
    """A trajectory or one of its declared tick artifacts is invalid."""


@dataclass(frozen=True, slots=True)
class TrajectoryEntity:
    ball_id: int
    color_id: int
    object_kind: str
    zone: str
    index: int
    curve_distance: float
    position_x: float
    position_y: float
    # Retail's 32-bit ``ball_id`` is reused while an older ball carrying the
    # same value can still be alive.  The process-local object address is the
    # only observed identity token that remains unique in those long games.
    # It is intentionally omitted from ``to_dict`` because it is not portable
    # across processes and must never become a simulator-facing contract.
    native_object_address: int | None = None
    radius: float | None = None
    velocity_x: float | None = None
    velocity_y: float | None = None
    merge_progress: float | None = None
    merge_speed: float | None = None
    fired: bool | None = None
    contact_next: bool | None = None
    exploding: bool | None = None
    explode_frame: int | None = None
    should_remove: bool | None = None
    update_count: int | None = None
    suck_count: int | None = None
    backwards_count: int | None = None
    backwards_speed: float | None = None
    combo_count: int | None = None
    combo_score: int | None = None
    powerup_previous_type: int | None = None
    powerup_primary_type: int | None = None
    powerup_secondary_type: int | None = None
    powerup_previous_ticks: int | None = None
    powerup_lifetime_ticks: int | None = None
    powerup_transition_ticks: int | None = None
    powerup_visual_scale: float | None = None
    powerup_visual_step: float | None = None
    powerup_visual_index: int | None = None
    gap_list_sentinel_address: int | None = None
    gap_entry_count: int | None = None
    curve_points: tuple[int, ...] | None = None
    gap_entries: tuple[tuple[int, int, int], ...] = ()

    @property
    def native_identity(self) -> tuple[int, int]:
        """Return a stable in-process identity; zero marks legacy evidence."""

        return (self.ball_id, self.native_object_address or 0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ball_id": self.ball_id,
            "color_id": self.color_id,
            "object_kind": self.object_kind,
            "zone": self.zone,
            "index": self.index,
            "curve_distance": self.curve_distance,
            "position_x": self.position_x,
            "position_y": self.position_y,
            "radius": self.radius,
            "velocity_x": self.velocity_x,
            "velocity_y": self.velocity_y,
            "merge_progress": self.merge_progress,
            "merge_speed": self.merge_speed,
            "fired": self.fired,
            "contact_next": self.contact_next,
            "exploding": self.exploding,
            "explode_frame": self.explode_frame,
            "should_remove": self.should_remove,
            "update_count": self.update_count,
            "suck_count": self.suck_count,
            "backwards_count": self.backwards_count,
            "backwards_speed": self.backwards_speed,
            "combo_count": self.combo_count,
            "combo_score": self.combo_score,
            "powerup_previous_type": self.powerup_previous_type,
            "powerup_primary_type": self.powerup_primary_type,
            "powerup_secondary_type": self.powerup_secondary_type,
            "powerup_previous_ticks": self.powerup_previous_ticks,
            "powerup_lifetime_ticks": self.powerup_lifetime_ticks,
            "powerup_transition_ticks": self.powerup_transition_ticks,
            "powerup_visual_scale": self.powerup_visual_scale,
            "powerup_visual_step": self.powerup_visual_step,
            "powerup_visual_index": self.powerup_visual_index,
            "gap_list_sentinel_address": self.gap_list_sentinel_address,
            "gap_entry_count": self.gap_entry_count,
            "curve_points": (
                None if self.curve_points is None else list(self.curve_points)
            ),
            "gap_entries": [
                {
                    "curve_index": curve_index,
                    "gap_distance": gap_distance,
                    "boundary_ball_id": boundary_ball_id,
                }
                for curve_index, gap_distance, boundary_ball_id
                in self.gap_entries
            ],
        }


@dataclass(frozen=True, slots=True)
class TrajectoryCurvePowerupState:
    curve_index: int
    last_any_spawn_time: int
    last_spawn_times: tuple[int, ...]
    cooldown_times: tuple[int, ...]
    spawn_counts: tuple[int, ...]
    field_124_by_type: tuple[int, ...]
    active_color_counts: tuple[int, ...]
    reverse_speed: float = 0.0
    slow_ticks: int = 0
    reverse_ticks: int = 0
    last_powerup_waypoint: int = 0
    powerup_triggered: bool = False


@dataclass(frozen=True, slots=True)
class TrajectoryCurvePlanState:
    curve_index: int
    begin_address: int
    end_address: int
    capacity_address: int
    planned_count: int
    capacity_count: int
    add_plan_enabled: bool


@dataclass(frozen=True, slots=True)
class TrajectoryCurveRuntimeState:
    curve_index: int
    advance_speed: float
    first_chain_end: int
    stop_adding: bool
    has_reached_cruising_speed: bool
    has_reached_rollout: bool
    stop_time: int = 0
    first_ball_moved_backwards: bool = False


@dataclass(frozen=True, slots=True)
class TrajectoryQRandState:
    update_count: int
    selected_index: int
    weights: tuple[float, ...]
    sways: tuple[float, ...]
    last_hit: tuple[int, ...]
    previous_hit: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "update_count": self.update_count,
            "selected_index": self.selected_index,
            "weights": list(self.weights),
            "sways": list(self.sways),
            "last_hit": list(self.last_hit),
            "previous_hit": list(self.previous_hit),
        }


@dataclass(frozen=True, slots=True)
class TrajectoryMTRandState:
    index: int
    words: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "words": list(self.words),
        }


@dataclass(frozen=True, slots=True)
class TrajectoryFruitState:
    """Fruit fields decoded from one content-addressed retail Board image."""

    # This is a pointer to one 24-byte TreasurePoint record, not an inline
    # ``(x, y)`` pair.  Retail rendering dereferences ``Board+0x0B4`` before
    # reading the point coordinates (0x00417F2F-0x00417F68), and the selector
    # stores an address from the level's TreasurePoint array at 0x004B5C87-
    # 0x004B5C99.  Keeping the pointer as an unsigned token lets an immutable
    # Board image prove presence without pretending process addresses are
    # portable geometry.
    active_point_pointer: int
    selected_point_index: int
    collecting: bool
    velocity: float
    max_velocity: float
    acceleration: float
    vertical_offset: float
    lower_bound: float
    upper_bound: float
    glow_alpha: int
    glow_step: int
    alpha: int
    expiry_time: int
    cell_index: int

    @property
    def active(self) -> bool:
        return self.active_point_pointer != 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "active": self.active,
            "active_point_pointer": self.active_point_pointer,
            "selected_point_index": self.selected_point_index,
            "collecting": self.collecting,
            "velocity": self.velocity,
            "max_velocity": self.max_velocity,
            "acceleration": self.acceleration,
            "vertical_offset": self.vertical_offset,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "glow_alpha": self.glow_alpha,
            "glow_step": self.glow_step,
            "alpha": self.alpha,
            "expiry_time": self.expiry_time,
            "cell_index": self.cell_index,
        }


@dataclass(frozen=True, slots=True)
class TrajectoryFrame:
    update: int
    score: int
    displayed_score: int
    score_target: int
    current_ball_id: int | None
    current_color_id: int | None
    next_ball_id: int | None
    next_color_id: int | None
    list_counts: tuple[tuple[int, int, int], ...]
    entities: tuple[TrajectoryEntity, ...]
    qrand: TrajectoryQRandState | None = None
    thread_crt_rand_state: int | None = None
    global_mtrand: TrajectoryMTRandState | None = None
    native_game_time: int | None = None
    board_update_count: int | None = None
    curve_powerups: tuple[TrajectoryCurvePowerupState, ...] = ()
    board_runtime_flag_157: bool | None = None
    board_runtime_i32_f54: int | None = None
    active_board_version: int | None = None
    board_mode_flag_1064: bool | None = None
    curve_plan_exhausted: bool | None = None
    post_zuma_timer_remaining: int | None = None
    post_zuma_ramp_404: float | None = None
    post_zuma_ramp_408: float | None = None
    curve_plans: tuple[TrajectoryCurvePlanState, ...] = ()
    curve_runtime_states: tuple[TrajectoryCurveRuntimeState, ...] = ()
    fruit_state: TrajectoryFruitState | None = None

    @property
    def entities_by_id(self) -> Mapping[int, TrajectoryEntity]:
        counts: dict[int, int] = {}
        for entity in self.entities:
            counts[entity.ball_id] = counts.get(entity.ball_id, 0) + 1
        # A bare retail ball_id is useful only when it is unambiguous in this
        # frame.  Omitting reused IDs prevents callers from silently binding
        # to whichever duplicate happened to be visited last.
        return {
            entity.ball_id: entity
            for entity in self.entities
            if counts[entity.ball_id] == 1
        }

    @property
    def entities_by_native_identity(
        self,
    ) -> Mapping[tuple[int, int], TrajectoryEntity]:
        return {entity.native_identity: entity for entity in self.entities}

    def list_count(self, curve_index: int, offset: int) -> int:
        for candidate_curve, candidate_offset, count in self.list_counts:
            if (
                candidate_curve == curve_index
                and candidate_offset == offset
            ):
                return count
        return 0

    def curve_powerup_state(
        self,
        curve_index: int,
    ) -> TrajectoryCurvePowerupState | None:
        matches = [
            state
            for state in self.curve_powerups
            if state.curve_index == curve_index
        ]
        if len(matches) > 1:
            _fail("trajectory_duplicate_curve_powerup_state")
        return matches[0] if matches else None

    def curve_plan_state(
        self,
        curve_index: int,
    ) -> TrajectoryCurvePlanState | None:
        matches = [
            state
            for state in self.curve_plans
            if state.curve_index == curve_index
        ]
        if len(matches) > 1:
            _fail("trajectory_duplicate_curve_plan_state")
        return matches[0] if matches else None

    def curve_runtime_state(
        self,
        curve_index: int,
    ) -> TrajectoryCurveRuntimeState | None:
        matches = [
            state
            for state in self.curve_runtime_states
            if state.curve_index == curve_index
        ]
        if len(matches) > 1:
            _fail("trajectory_duplicate_curve_runtime_state")
        return matches[0] if matches else None


def _fail(code: str) -> None:
    raise PcMemoryTrajectoryError(code)


def _reject_constant(value: str) -> None:
    _fail(f"nonfinite_json_number:{value}")


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate_json_key:{key}")
        result[key] = value
    return result


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")


def _read_canonical_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = path.read_bytes()
        value = json.loads(
            payload.decode("ascii"),
            object_pairs_hook=_pairs,
            parse_constant=_reject_constant,
        )
    except PcMemoryTrajectoryError:
        raise
    except (OSError, UnicodeError, ValueError) as error:
        raise PcMemoryTrajectoryError(
            f"trajectory_json_invalid:{path.name}"
        ) from error
    if not isinstance(value, dict):
        _fail(f"trajectory_json_root_invalid:{path.name}")
    if payload != _canonical_bytes(value):
        _fail(f"trajectory_json_not_canonical:{path.name}")
    return value


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _artifact_path(root: Path, member: Any) -> Path:
    if not isinstance(member, str) or not member:
        _fail("trajectory_tick_artifact_path_invalid")
    if "\\" in member or ":" in member:
        _fail("trajectory_tick_artifact_path_invalid")
    pure = PurePosixPath(member)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        _fail("trajectory_tick_artifact_path_invalid")
    path = root.joinpath(*pure.parts)
    try:
        path.resolve().relative_to(root.resolve())
    except (OSError, ValueError) as error:
        raise PcMemoryTrajectoryError(
            "trajectory_tick_artifact_path_escape"
        ) from error
    return path


def _integer(value: Any, code: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(code)
    return value


def _optional_integer(value: Any, code: str) -> int | None:
    if value is None:
        return None
    return _integer(value, code)


def _number(value: Any, code: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        _fail(code)
    return float(value)


def _fruit_state_from_board_raw(board_raw: bytes) -> TrajectoryFruitState:
    """Decode the retail fruit state proven by Board update disassembly."""

    active_point_pointer = struct.unpack_from(
        "<I",
        board_raw,
        BOARD_FRUIT_ACTIVE_POINT_POINTER_OFFSET,
    )[0]
    selected_point_index = struct.unpack_from(
        "<i",
        board_raw,
        BOARD_FRUIT_SELECTED_POINT_OFFSET,
    )[0]
    collecting_raw = board_raw[BOARD_FRUIT_COLLECTING_OFFSET]
    if collecting_raw not in {0, 1}:
        _fail("trajectory_tick_fruit_collecting_invalid")
    float_offsets = (
        BOARD_FRUIT_VELOCITY_OFFSET,
        BOARD_FRUIT_MAX_VELOCITY_OFFSET,
        BOARD_FRUIT_ACCELERATION_OFFSET,
        BOARD_FRUIT_VERTICAL_OFFSET,
        BOARD_FRUIT_LOWER_BOUND_OFFSET,
        BOARD_FRUIT_UPPER_BOUND_OFFSET,
    )
    float_values = tuple(
        struct.unpack_from("<f", board_raw, offset)[0]
        for offset in float_offsets
    )
    if not all(math.isfinite(value) for value in float_values):
        _fail("trajectory_tick_fruit_float_invalid")
    glow_alpha = struct.unpack_from(
        "<i",
        board_raw,
        BOARD_FRUIT_GLOW_ALPHA_OFFSET,
    )[0]
    glow_step = struct.unpack_from(
        "<i",
        board_raw,
        BOARD_FRUIT_GLOW_STEP_OFFSET,
    )[0]
    alpha = struct.unpack_from(
        "<i",
        board_raw,
        BOARD_FRUIT_ALPHA_OFFSET,
    )[0]
    expiry_time = struct.unpack_from(
        "<i",
        board_raw,
        BOARD_FRUIT_EXPIRY_TIME_OFFSET,
    )[0]
    cell_index = struct.unpack_from(
        "<i",
        board_raw,
        BOARD_FRUIT_CELL_INDEX_OFFSET,
    )[0]
    if (
        selected_point_index < -1
        or (
            active_point_pointer == 0
            and selected_point_index not in {-1, 0}
        )
        or (active_point_pointer != 0 and selected_point_index < 0)
        or not 0 <= glow_alpha <= 255
        or not 0 <= alpha <= 255
        or expiry_time < 0
        or cell_index < 0
    ):
        _fail("trajectory_tick_fruit_integer_invalid")
    return TrajectoryFruitState(
        active_point_pointer=active_point_pointer,
        selected_point_index=selected_point_index,
        collecting=bool(collecting_raw),
        velocity=float_values[0],
        max_velocity=float_values[1],
        acceleration=float_values[2],
        vertical_offset=float_values[3],
        lower_bound=float_values[4],
        upper_bound=float_values[5],
        glow_alpha=glow_alpha,
        glow_step=glow_step,
        alpha=alpha,
        expiry_time=expiry_time,
        cell_index=cell_index,
    )


def _optional_powerup_type(
    record: Mapping[str, Any],
    key: str,
) -> int | None:
    value = record.get(key)
    if value is None:
        return None
    decoded = _integer(
        value,
        "trajectory_entity_powerup_type_invalid",
    )
    if decoded > POWERUP_NONE_TYPE:
        _fail("trajectory_entity_powerup_type_invalid")
    return decoded


def _mapping(value: Any, code: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        _fail(code)
    return value


def _curve_powerup_state(
    curve: Mapping[str, Any],
    *,
    curve_index: int,
    tick_root: Path | None,
) -> TrajectoryCurvePowerupState | None:
    if curve.get("artifact") is None:
        return None
    raw = _declared_artifact_bytes(
        curve,
        root=tick_root,
        code="trajectory_tick_curve_artifact",
        expected_size=CURVE_OBJECT_SIZE,
        allow_missing_size=True,
    )

    def signed_array(offset: int, count: int) -> tuple[int, ...]:
        return tuple(
            int(value)
            for value in struct.unpack_from(f"<{count}i", raw, offset)
        )

    last_any_spawn_time = struct.unpack_from("<i", raw, 0x78)[0]
    last_spawn_times = signed_array(0x7C, POWERUP_TYPE_COUNT)
    cooldown_times = signed_array(0xB4, POWERUP_TYPE_COUNT)
    spawn_counts = signed_array(0xEC, POWERUP_TYPE_COUNT)
    field_124_by_type = signed_array(0x124, POWERUP_TYPE_COUNT)
    active_color_counts = signed_array(0x15C, 6)
    reverse_speed = struct.unpack_from("<f", raw, 0x18)[0]
    slow_ticks = struct.unpack_from("<i", raw, 0x184)[0]
    reverse_ticks = struct.unpack_from("<i", raw, 0x188)[0]
    last_powerup_waypoint = struct.unpack_from("<i", raw, 0x1B4)[0]
    powerup_triggered_raw = raw[0x1BC]
    if (
        last_any_spawn_time < 0
        or any(value < 0 for value in spawn_counts)
        or any(value < 0 for value in active_color_counts)
        or not math.isfinite(reverse_speed)
        or reverse_speed < 0.0
        or slow_ticks < 0
        or reverse_ticks < 0
        or powerup_triggered_raw not in {0, 1}
    ):
        _fail("trajectory_tick_curve_powerup_state_invalid")
    return TrajectoryCurvePowerupState(
        curve_index=curve_index,
        last_any_spawn_time=last_any_spawn_time,
        last_spawn_times=last_spawn_times,
        cooldown_times=cooldown_times,
        spawn_counts=spawn_counts,
        field_124_by_type=field_124_by_type,
        active_color_counts=active_color_counts,
        reverse_speed=reverse_speed,
        slow_ticks=slow_ticks,
        reverse_ticks=reverse_ticks,
        last_powerup_waypoint=last_powerup_waypoint,
        powerup_triggered=bool(powerup_triggered_raw),
    )


def _curve_plan_state(
    curve: Mapping[str, Any],
    *,
    curve_index: int,
    tick_root: Path | None,
) -> TrajectoryCurvePlanState:
    raw = _declared_artifact_bytes(
        curve,
        root=tick_root,
        code="trajectory_tick_curve_artifact",
        expected_size=CURVE_OBJECT_SIZE,
        allow_missing_size=True,
    )
    if struct.unpack_from("<I", raw, 0)[0] != CURVE_VTABLE:
        _fail("trajectory_tick_curve_vtable_invalid")
    begin = struct.unpack_from(
        "<I",
        raw,
        CURVE_PLANNED_VECTOR_BEGIN_OFFSET,
    )[0]
    end = struct.unpack_from(
        "<I",
        raw,
        CURVE_PLANNED_VECTOR_END_OFFSET,
    )[0]
    capacity = struct.unpack_from(
        "<I",
        raw,
        CURVE_PLANNED_VECTOR_CAPACITY_OFFSET,
    )[0]
    if begin == end == capacity == 0:
        planned_bytes = 0
        planned_count = 0
        capacity_count = 0
    else:
        if (
            begin == 0
            or end == 0
            or capacity == 0
            or not begin <= end <= capacity
        ):
            _fail("trajectory_tick_curve_plan_bounds_invalid")
        planned_bytes = end - begin
        capacity_bytes = capacity - begin
        if (
            planned_bytes % CURVE_PLANNED_ITEM_SIZE != 0
            or capacity_bytes % CURVE_PLANNED_ITEM_SIZE != 0
        ):
            _fail("trajectory_tick_curve_plan_alignment_invalid")
        planned_count = planned_bytes // CURVE_PLANNED_ITEM_SIZE
        capacity_count = capacity_bytes // CURVE_PLANNED_ITEM_SIZE
        if (
            planned_count > MAX_CURVE_PLANNED_ITEMS
            or capacity_count > MAX_CURVE_PLANNED_ITEMS
        ):
            _fail("trajectory_tick_curve_plan_count_invalid")
    add_plan_raw = raw[CURVE_ADD_PLAN_ENABLED_OFFSET]
    if add_plan_raw not in {0, 1}:
        _fail("trajectory_tick_curve_add_plan_enabled_invalid")

    plan = _mapping(
        curve.get("planned_balls"),
        "trajectory_tick_curve_plan_invalid",
    )
    declared_add_plan = plan.get("add_plan_enabled")
    if (
        plan.get("vector_begin_offset")
        != CURVE_PLANNED_VECTOR_BEGIN_OFFSET
        or plan.get("vector_end_offset")
        != CURVE_PLANNED_VECTOR_END_OFFSET
        or plan.get("vector_capacity_offset")
        != CURVE_PLANNED_VECTOR_CAPACITY_OFFSET
        or plan.get("begin_address") != begin
        or plan.get("end_address") != end
        or plan.get("capacity_address") != capacity
        or plan.get("item_size") != CURVE_PLANNED_ITEM_SIZE
        or plan.get("count") != planned_count
        or plan.get("capacity_count") != capacity_count
        or plan.get("add_plan_enabled_offset")
        != CURVE_ADD_PLAN_ENABLED_OFFSET
        or not isinstance(declared_add_plan, bool)
        or declared_add_plan is not bool(add_plan_raw)
    ):
        _fail("trajectory_tick_curve_plan_semantics_mismatch")
    _declared_artifact_bytes(
        plan,
        root=tick_root,
        code="trajectory_tick_curve_plan_artifact",
        expected_size=planned_bytes,
    )
    return TrajectoryCurvePlanState(
        curve_index=curve_index,
        begin_address=begin,
        end_address=end,
        capacity_address=capacity,
        planned_count=planned_count,
        capacity_count=capacity_count,
        add_plan_enabled=bool(add_plan_raw),
    )


def _curve_runtime_state(
    curve: Mapping[str, Any],
    *,
    curve_index: int,
    tick_root: Path | None,
) -> TrajectoryCurveRuntimeState:
    """Decode behavior-critical latches from the hash-bound Curve object."""

    raw = _declared_artifact_bytes(
        curve,
        root=tick_root,
        code="trajectory_tick_curve_runtime_artifact",
        expected_size=CURVE_OBJECT_SIZE,
        allow_missing_size=True,
    )
    advance_speed = struct.unpack_from(
        "<f",
        raw,
        CURVE_ADVANCE_SPEED_OFFSET,
    )[0]
    first_chain_end = struct.unpack_from(
        "<i",
        raw,
        CURVE_FIRST_CHAIN_END_OFFSET,
    )[0]
    stop_time = struct.unpack_from(
        "<i",
        raw,
        CURVE_STOP_TIME_OFFSET,
    )[0]
    raw_flags = (
        raw[CURVE_FIRST_BALL_MOVED_BACKWARDS_OFFSET],
        raw[CURVE_STOP_ADDING_OFFSET],
        raw[CURVE_HAS_REACHED_CRUISING_SPEED_OFFSET],
        raw[CURVE_HAS_REACHED_ROLLOUT_OFFSET],
    )
    if (
        not math.isfinite(advance_speed)
        or stop_time < 0
        or any(value not in {0, 1} for value in raw_flags)
    ):
        _fail("trajectory_tick_curve_runtime_state_invalid")
    return TrajectoryCurveRuntimeState(
        curve_index=curve_index,
        advance_speed=advance_speed,
        first_chain_end=first_chain_end,
        stop_adding=bool(raw_flags[1]),
        has_reached_cruising_speed=bool(raw_flags[2]),
        has_reached_rollout=bool(raw_flags[3]),
        stop_time=stop_time,
        first_ball_moved_backwards=bool(raw_flags[0]),
    )


def _sequence(value: Any, code: str) -> Sequence[Any]:
    if not isinstance(value, list):
        _fail(code)
    return value


def _declared_artifact_bytes(
    record: Mapping[str, Any],
    *,
    root: Path | None,
    code: str,
    expected_size: int | None = None,
    allow_missing_size: bool = False,
) -> bytes:
    if root is None:
        _fail(f"{code}_root_missing")
    path = _artifact_path(root, record.get("artifact"))
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise PcMemoryTrajectoryError(f"{code}_missing") from error
    declared_size_value = record.get("artifact_bytes")
    declared_size = (
        None
        if declared_size_value is None and allow_missing_size
        else _integer(
            declared_size_value,
            f"{code}_size_invalid",
        )
    )
    if (
        (declared_size is not None and declared_size != len(payload))
        or (
            expected_size is not None
            and len(payload) != expected_size
        )
    ):
        _fail(f"{code}_size_mismatch")
    declared_sha = record.get("artifact_sha256")
    actual_sha = f"sha256:{hashlib.sha256(payload).hexdigest()}"
    if not isinstance(declared_sha, str) or declared_sha != actual_sha:
        _fail(f"{code}_sha256_mismatch")
    return payload


def _qrand_from_board(
    board: Mapping[str, Any],
    *,
    tick_root: Path | None,
) -> TrajectoryQRandState | None:
    value = board.get("qrand")
    if value is None:
        return None
    qrand = _mapping(value, "trajectory_tick_qrand_invalid")
    if (
        qrand.get("schema") != "zuma-rl.pc-qrand-state"
        or qrand.get("version") != 1
        or qrand.get("board_pointer_offset")
        != QRAND_BOARD_POINTER_OFFSET
    ):
        _fail("trajectory_tick_qrand_schema_invalid")
    raw = _declared_artifact_bytes(
        qrand,
        root=tick_root,
        code="trajectory_tick_qrand_artifact",
        expected_size=QRAND_OBJECT_SIZE,
    )
    update_count = _integer(
        qrand.get("update_count"),
        "trajectory_tick_qrand_update_count_invalid",
    )
    selected_value = qrand.get("selected_index")
    if (
        isinstance(selected_value, bool)
        or not isinstance(selected_value, int)
        or not -1 <= selected_value < QRAND_VECTOR_LENGTH
    ):
        _fail("trajectory_tick_qrand_selected_index_invalid")
    selected_index = int(selected_value)
    if struct.unpack_from("<i", raw, 0)[0] != update_count:
        _fail("trajectory_tick_qrand_update_count_mismatch")
    if struct.unpack_from("<i", raw, 4)[0] != selected_index:
        _fail("trajectory_tick_qrand_selected_index_mismatch")

    vector_rows = _sequence(
        qrand.get("vectors"),
        "trajectory_tick_qrand_vectors_invalid",
    )
    if len(vector_rows) != len(QRAND_VECTOR_LAYOUT):
        _fail("trajectory_tick_qrand_vectors_invalid")
    decoded: dict[str, tuple[int | float, ...]] = {}
    zero_header_count = 0
    for item, (offset, name, value_type) in zip(
        vector_rows,
        QRAND_VECTOR_LAYOUT,
        strict=True,
    ):
        vector = _mapping(
            item,
            "trajectory_tick_qrand_vector_invalid",
        )
        if (
            vector.get("name") != name
            or vector.get("value_type") != value_type
            or vector.get("object_offset") != offset
        ):
            _fail("trajectory_tick_qrand_vector_layout_invalid")
        begin = _integer(
            vector.get("begin"),
            "trajectory_tick_qrand_vector_header_invalid",
        )
        end = _integer(
            vector.get("end"),
            "trajectory_tick_qrand_vector_header_invalid",
        )
        capacity = _integer(
            vector.get("capacity"),
            "trajectory_tick_qrand_vector_header_invalid",
        )
        zero_header = begin == end == capacity == 0
        if zero_header:
            if update_count != 0 or selected_index != -1:
                _fail("trajectory_tick_qrand_vector_header_mismatch")
            item_count = _integer(
                vector.get("item_count"),
                "trajectory_tick_qrand_vector_count_invalid",
            )
            capacity_count = _integer(
                vector.get("capacity_count"),
                "trajectory_tick_qrand_vector_count_invalid",
            )
            if item_count != 0 or capacity_count != 0:
                _fail("trajectory_tick_qrand_vector_count_mismatch")
            payload = _declared_artifact_bytes(
                vector,
                root=tick_root,
                code=f"trajectory_tick_qrand_{name}_artifact",
                expected_size=0,
            )
            declared_values = _sequence(
                vector.get("values"),
                "trajectory_tick_qrand_vector_values_invalid",
            )
            if payload or declared_values:
                _fail("trajectory_tick_qrand_vector_value_mismatch")
            decoded[name] = ()
            zero_header_count += 1
            continue
        if (
            struct.unpack_from("<III", raw, offset + 4)
            != (begin, end, capacity)
            or begin == 0
            or end < begin
            or capacity < end
            or begin % 4
            or end % 4
            or capacity % 4
        ):
            _fail("trajectory_tick_qrand_vector_header_mismatch")
        item_count = _integer(
            vector.get("item_count"),
            "trajectory_tick_qrand_vector_count_invalid",
        )
        capacity_count = _integer(
            vector.get("capacity_count"),
            "trajectory_tick_qrand_vector_count_invalid",
        )
        if (
            item_count != QRAND_VECTOR_LENGTH
            or item_count != (end - begin) // 4
            or capacity_count != (capacity - begin) // 4
            or capacity_count < item_count
        ):
            _fail("trajectory_tick_qrand_vector_count_mismatch")
        payload = _declared_artifact_bytes(
            vector,
            root=tick_root,
            code=f"trajectory_tick_qrand_{name}_artifact",
            expected_size=item_count * 4,
        )
        if value_type == "float32":
            raw_values: tuple[int | float, ...] = struct.unpack(
                f"<{item_count}f",
                payload,
            )
            declared_values: tuple[int | float, ...] = tuple(
                _number(
                    item_value,
                    "trajectory_tick_qrand_vector_value_invalid",
                )
                for item_value in _sequence(
                    vector.get("values"),
                    "trajectory_tick_qrand_vector_values_invalid",
                )
            )
        else:
            raw_values = struct.unpack(f"<{item_count}i", payload)
            declared_values = tuple(
                _integer(
                    item_value,
                    "trajectory_tick_qrand_vector_value_invalid",
                )
                for item_value in _sequence(
                    vector.get("values"),
                    "trajectory_tick_qrand_vector_values_invalid",
                )
            )
        if (
            len(declared_values) != item_count
            or declared_values != raw_values
        ):
            _fail("trajectory_tick_qrand_vector_value_mismatch")
        decoded[name] = raw_values

    if zero_header_count not in {0, len(QRAND_VECTOR_LAYOUT)}:
        _fail("trajectory_tick_qrand_vector_header_mismatch")

    return TrajectoryQRandState(
        update_count=update_count,
        selected_index=selected_index,
        weights=tuple(float(value) for value in decoded["weights"]),
        sways=tuple(float(value) for value in decoded["sways"]),
        last_hit=tuple(int(value) for value in decoded["last_hit"]),
        previous_hit=tuple(
            int(value) for value in decoded["previous_hit"]
        ),
    )


def _thread_crt_rand_from_board(
    board: Mapping[str, Any],
    *,
    tick_root: Path | None,
) -> int | None:
    value = board.get("thread_crt_rand")
    if value is None:
        return None
    record = _mapping(
        value,
        "trajectory_tick_thread_crt_rand_invalid",
    )
    if (
        record.get("schema")
        != "zuma-rl.pc-thread-crt-rand-state"
        or record.get("version") != 1
    ):
        _fail("trajectory_tick_thread_crt_rand_schema_invalid")
    state = _integer(
        record.get("state"),
        "trajectory_tick_thread_crt_rand_state_invalid",
    )
    if state > 0xFFFFFFFF:
        _fail("trajectory_tick_thread_crt_rand_state_invalid")
    payload = _declared_artifact_bytes(
        record,
        root=tick_root,
        code="trajectory_tick_thread_crt_rand_artifact",
        expected_size=4,
    )
    if struct.unpack("<I", payload)[0] != state:
        _fail("trajectory_tick_thread_crt_rand_state_mismatch")
    return state


def _mtrand_from_board(
    board: Mapping[str, Any],
    *,
    key: str,
    role: str,
    tick_root: Path | None,
) -> TrajectoryMTRandState | None:
    value = board.get(key)
    if value is None:
        return None
    record = _mapping(
        value,
        f"trajectory_tick_{role}_mtrand_invalid",
    )
    if (
        record.get("schema") != "zuma-rl.pc-mtrand-state"
        or record.get("version") != 1
        or record.get("role") != role
        or record.get("state_word_count") != MTRAND_STATE_WORDS
    ):
        _fail(f"trajectory_tick_{role}_mtrand_schema_invalid")
    index = _integer(
        record.get("index"),
        f"trajectory_tick_{role}_mtrand_index_invalid",
    )
    if index > MTRAND_STATE_WORDS:
        _fail(f"trajectory_tick_{role}_mtrand_index_invalid")
    payload = _declared_artifact_bytes(
        record,
        root=tick_root,
        code=f"trajectory_tick_{role}_mtrand_artifact",
        expected_size=MTRAND_STATE_BYTES,
    )
    values = struct.unpack(f"<{MTRAND_STATE_WORDS + 1}I", payload)
    if values[-1] != index:
        _fail(f"trajectory_tick_{role}_mtrand_index_mismatch")
    return TrajectoryMTRandState(
        index=index,
        words=tuple(int(value) for value in values[:-1]),
    )


def _bullet_curve_points_valid(
    curve_points: tuple[int, ...],
    gap_entries: tuple[tuple[int, int, int], ...],
) -> bool:
    """Accept retail debug fill only for curve slots not used by a gap."""

    if any(
        value < 0 and value != RETAIL_DEBUG_FILL_I32
        for value in curve_points
    ):
        return False
    return all(
        curve_points[curve_index] >= 0
        for curve_index, _, _ in gap_entries
    )


def _entity(
    record: Mapping[str, Any],
    *,
    zone: str,
    index: int,
    default_kind: str,
    raw_payload: bytes | None = None,
    gap_payload: bytes | None = None,
) -> TrajectoryEntity:
    ball = _mapping(record.get("ball"), "trajectory_entity_ball_invalid")
    kind = record.get("payload_kind", default_kind)
    if kind not in {"ball", "bullet"}:
        _fail("trajectory_entity_kind_invalid")
    powerup_previous_type = _optional_powerup_type(
        ball,
        "powerup_previous_type",
    )
    powerup_primary_type = _optional_powerup_type(
        ball,
        "powerup_primary_type",
    )
    powerup_secondary_type = _optional_powerup_type(
        ball,
        "powerup_secondary_type",
    )
    ball_id = _integer(
        ball.get("ball_id"),
        "trajectory_entity_id_invalid",
    )
    raw_addresses = [
        record.get(name)
        for name in ("address", "bullet_address", "payload_address")
        if record.get(name) is not None
    ]
    native_object_address: int | None = None
    if raw_addresses:
        parsed_addresses = tuple(
            _integer(
                value,
                "trajectory_entity_native_address_invalid",
                minimum=1,
            )
            for value in raw_addresses
        )
        if len(set(parsed_addresses)) != 1:
            _fail("trajectory_entity_native_address_mismatch")
        native_object_address = parsed_addresses[0]
    color = _integer(
        ball.get("color_id"),
        "trajectory_entity_color_invalid",
    )
    if color > 5:
        _fail("trajectory_entity_color_invalid")
    curve_distance = _number(
        ball.get("curve_distance"),
        "trajectory_entity_distance_invalid",
    )
    position_x = _number(
        ball.get("position_x"),
        "trajectory_entity_position_invalid",
    )
    position_y = _number(
        ball.get("position_y"),
        "trajectory_entity_position_invalid",
    )
    radius_value = ball.get("radius")
    radius = (
        None
        if radius_value is None
        else _number(
            radius_value,
            "trajectory_entity_radius_invalid",
        )
    )
    if radius is not None and radius <= 0.0:
        _fail("trajectory_entity_radius_invalid")
    subclass_value = record.get("subclass_fields")
    subclass: Mapping[str, Any] | None = None
    fired: bool | None = None
    gap_list_sentinel_address: int | None = None
    gap_entry_count: int | None = None
    curve_points: tuple[int, ...] | None = None
    gap_entries: tuple[tuple[int, int, int], ...] = ()
    gap_entry_records: tuple[Mapping[str, Any], ...] = ()
    gap_entries_declared = False
    if kind == "bullet":
        subclass = _mapping(
            subclass_value,
            "trajectory_entity_bullet_subclass_invalid",
        )
        fired_value = subclass.get("fired")
        if not isinstance(fired_value, bool):
            _fail("trajectory_entity_bullet_fired_invalid")
        fired = fired_value
        sentinel_value = subclass.get("gap_list_sentinel_address")
        if sentinel_value is not None:
            gap_list_sentinel_address = _integer(
                sentinel_value,
                "trajectory_entity_gap_sentinel_invalid",
                minimum=1,
            )
        count_value = subclass.get("gap_entry_count")
        if count_value is not None:
            gap_entry_count = _integer(
                count_value,
                "trajectory_entity_gap_count_invalid",
            )
            if gap_entry_count > MAX_BULLET_GAP_ENTRIES:
                _fail("trajectory_entity_gap_count_invalid")
        points_value = subclass.get("curve_points")
        if points_value is not None:
            points = _sequence(
                points_value,
                "trajectory_entity_curve_points_invalid",
            )
            if len(points) != BULLET_CURVE_POINT_COUNT:
                _fail("trajectory_entity_curve_points_invalid")
            curve_points = tuple(
                _integer(
                    value,
                    "trajectory_entity_curve_points_invalid",
                    minimum=-(2**31),
                )
                for value in points
            )
            if any(value > 2**31 - 1 for value in curve_points):
                _fail("trajectory_entity_curve_points_invalid")
        entries_value = subclass.get("gap_entries")
        if entries_value is not None:
            gap_entries_declared = True
            entries = _sequence(
                entries_value,
                "trajectory_entity_gap_entries_invalid",
            )
            if len(entries) > MAX_BULLET_GAP_ENTRIES:
                _fail("trajectory_entity_gap_entries_invalid")
            parsed_entries: list[tuple[int, int, int]] = []
            parsed_records: list[Mapping[str, Any]] = []
            for entry_index, value in enumerate(entries):
                entry = _mapping(
                    value,
                    "trajectory_entity_gap_entry_invalid",
                )
                if entry.get("index", entry_index) != entry_index:
                    _fail("trajectory_entity_gap_entry_index_invalid")
                curve_index_value = _integer(
                    entry.get("curve_index"),
                    "trajectory_entity_gap_curve_index_invalid",
                )
                if curve_index_value >= BULLET_CURVE_POINT_COUNT:
                    _fail("trajectory_entity_gap_curve_index_invalid")
                parsed_entries.append(
                    (
                        curve_index_value,
                        _integer(
                            entry.get("gap_distance"),
                            "trajectory_entity_gap_distance_invalid",
                            minimum=1,
                        ),
                        _integer(
                            entry.get("boundary_ball_id"),
                            "trajectory_entity_gap_boundary_id_invalid",
                            minimum=1,
                        ),
                    )
                )
                parsed_records.append(entry)
            gap_entries = tuple(parsed_entries)
            gap_entry_records = tuple(parsed_records)
        if (
            curve_points is not None
            and not _bullet_curve_points_valid(curve_points, gap_entries)
        ):
            _fail("trajectory_entity_curve_points_invalid")
        if (
            gap_entry_count is not None
            and gap_entry_count != len(gap_entries)
        ):
            _fail("trajectory_entity_gap_count_mismatch")
        if len({entry[2] for entry in gap_entries}) != len(gap_entries):
            _fail("trajectory_entity_gap_boundary_id_duplicate")
    flags_value = ball.get("flags_b4_c2_hex")
    flags: bytes | None = None
    if flags_value is not None:
        if (
            not isinstance(flags_value, str)
            or len(flags_value) != 30
        ):
            _fail("trajectory_entity_flags_invalid")
        try:
            flags = bytes.fromhex(flags_value)
        except ValueError:
            _fail("trajectory_entity_flags_invalid")
        if (
            len(flags) != 15
            or flags[0] not in {0, 1}
            or flags[6] not in {0, 1}
            or flags[12] not in {0, 1}
        ):
            _fail("trajectory_entity_flags_invalid")

    update_count: int | None = None
    suck_count: int | None = None
    backwards_count: int | None = None
    backwards_speed: float | None = None
    combo_count: int | None = None
    combo_score: int | None = None
    powerup_previous_ticks: int | None = None
    powerup_lifetime_ticks: int | None = None
    powerup_transition_ticks: int | None = None
    powerup_visual_scale: float | None = None
    powerup_visual_step: float | None = None
    powerup_visual_index: int | None = None
    if raw_payload is not None:
        expected_size = (
            BULLET_OBJECT_SIZE if kind == "bullet" else BALL_OBJECT_SIZE
        )
        if len(raw_payload) != expected_size:
            _fail("trajectory_entity_raw_size_invalid")
        declared_sha = record.get("payload_sha256")
        actual_sha = f"sha256:{hashlib.sha256(raw_payload).hexdigest()}"
        if (
            not isinstance(declared_sha, str)
            or declared_sha != actual_sha
        ):
            _fail("trajectory_entity_raw_sha256_mismatch")
        if flags is not None and raw_payload[0xB4:0xC3] != flags:
            _fail("trajectory_entity_raw_flags_mismatch")
        if kind == "bullet":
            raw_fired = raw_payload[0x16A]
            if raw_fired not in {0, 1} or bool(raw_fired) != fired:
                _fail("trajectory_entity_raw_fired_mismatch")
            raw_gap_sentinel = struct.unpack_from(
                "<I",
                raw_payload,
                BULLET_GAP_SENTINEL_POINTER_OFFSET,
            )[0]
            raw_gap_count = struct.unpack_from(
                "<I",
                raw_payload,
                BULLET_GAP_COUNT_OFFSET,
            )[0]
            raw_curve_points = tuple(
                int(value)
                for value in struct.unpack_from(
                    f"<{BULLET_CURVE_POINT_COUNT}i",
                    raw_payload,
                    BULLET_CURVE_POINTS_OFFSET,
                )
            )
            if (
                raw_gap_sentinel == 0
                or raw_gap_count > MAX_BULLET_GAP_ENTRIES
                or not _bullet_curve_points_valid(
                    raw_curve_points,
                    gap_entries,
                )
            ):
                _fail("trajectory_entity_raw_gap_state_invalid")
            if (
                gap_list_sentinel_address is not None
                and gap_list_sentinel_address != raw_gap_sentinel
            ):
                _fail("trajectory_entity_raw_gap_sentinel_mismatch")
            if (
                gap_entry_count is not None
                and gap_entry_count != raw_gap_count
            ):
                _fail("trajectory_entity_raw_gap_count_mismatch")
            if (
                curve_points is not None
                and curve_points != raw_curve_points
            ):
                _fail("trajectory_entity_raw_curve_points_mismatch")
            if (
                "field_174_pointer" in subclass
                and subclass["field_174_pointer"] != raw_gap_sentinel
            ):
                _fail("trajectory_entity_raw_gap_sentinel_mismatch")
            if (
                "field_178_i32" in subclass
                and subclass["field_178_i32"] != raw_gap_count
            ):
                _fail("trajectory_entity_raw_gap_count_mismatch")
            gap_list_sentinel_address = raw_gap_sentinel
            gap_entry_count = raw_gap_count
            curve_points = raw_curve_points
        raw_ball_id = struct.unpack_from("<I", raw_payload, 0x10)[0]
        raw_color = struct.unpack_from("<i", raw_payload, 0x14)[0]
        raw_curve_distance = struct.unpack_from(
            "<f",
            raw_payload,
            0x1C,
        )[0]
        raw_position_x, raw_position_y = struct.unpack_from(
            "<ff",
            raw_payload,
            0x2C,
        )
        raw_radius = struct.unpack_from("<f", raw_payload, 0x38)[0]
        if (
            raw_ball_id != ball_id
            or raw_color != color
            or raw_curve_distance != curve_distance
            or raw_position_x != position_x
            or raw_position_y != position_y
            or (radius is not None and raw_radius != radius)
        ):
            _fail("trajectory_entity_raw_geometry_mismatch")
        if (
            not math.isfinite(raw_curve_distance)
            or not math.isfinite(raw_position_x)
            or not math.isfinite(raw_position_y)
            or not math.isfinite(raw_radius)
            or raw_radius <= 0.0
        ):
            _fail("trajectory_entity_raw_geometry_invalid")
        ball_id = raw_ball_id
        color = raw_color
        curve_distance = raw_curve_distance
        position_x = raw_position_x
        position_y = raw_position_y
        radius = raw_radius
        update_count = struct.unpack_from("<i", raw_payload, 0xD4)[0]
        suck_count = struct.unpack_from("<i", raw_payload, 0xE0)[0]
        backwards_count = struct.unpack_from(
            "<i",
            raw_payload,
            0xE4,
        )[0]
        backwards_speed = struct.unpack_from(
            "<f",
            raw_payload,
            0xE8,
        )[0]
        combo_count = struct.unpack_from("<i", raw_payload, 0xEC)[0]
        combo_score = struct.unpack_from("<i", raw_payload, 0xF0)[0]
        powerup_previous_ticks = struct.unpack_from(
            "<i",
            raw_payload,
            0xC4,
        )[0]
        raw_previous_type = struct.unpack_from(
            "<i",
            raw_payload,
            0xC8,
        )[0]
        powerup_lifetime_ticks = struct.unpack_from(
            "<i",
            raw_payload,
            0xF8,
        )[0]
        powerup_transition_ticks = struct.unpack_from(
            "<i",
            raw_payload,
            0xFC,
        )[0]
        powerup_visual_scale = struct.unpack_from(
            "<f",
            raw_payload,
            0x104,
        )[0]
        powerup_visual_step = struct.unpack_from(
            "<f",
            raw_payload,
            0x108,
        )[0]
        powerup_visual_index = struct.unpack_from(
            "<i",
            raw_payload,
            0x10C,
        )[0]
        raw_primary_type = struct.unpack_from(
            "<i",
            raw_payload,
            0x11C,
        )[0]
        raw_secondary_type = struct.unpack_from(
            "<i",
            raw_payload,
            0x120,
        )[0]
        raw_powerup_types = (
            raw_previous_type,
            raw_primary_type,
            raw_secondary_type,
        )
        declared_powerup_types = (
            powerup_previous_type,
            powerup_primary_type,
            powerup_secondary_type,
        )
        if (
            min(
                update_count,
                suck_count,
                backwards_count,
                combo_count,
                combo_score,
                powerup_previous_ticks,
                powerup_lifetime_ticks,
                powerup_transition_ticks,
            )
            < 0
            or any(
                not 0 <= value <= POWERUP_NONE_TYPE
                for value in raw_powerup_types
            )
            or not math.isfinite(powerup_visual_scale)
            or not math.isfinite(powerup_visual_step)
            or not math.isfinite(backwards_speed)
            or backwards_speed < 0.0
        ):
            _fail("trajectory_entity_runtime_field_invalid")
        if any(
            declared is not None and declared != raw
            for declared, raw in zip(
                declared_powerup_types,
                raw_powerup_types,
                strict=True,
            )
        ):
            _fail("trajectory_entity_raw_powerup_type_mismatch")
        (
            powerup_previous_type,
            powerup_primary_type,
            powerup_secondary_type,
        ) = raw_powerup_types
    if gap_payload is not None:
        if kind != "bullet" or raw_payload is None:
            _fail("trajectory_entity_gap_artifact_without_raw_bullet")
        if gap_entry_count is None or gap_list_sentinel_address is None:
            _fail("trajectory_entity_gap_state_missing")
        expected_gap_bytes = gap_entry_count * BULLET_GAP_NODE_SIZE
        if len(gap_payload) != expected_gap_bytes:
            _fail("trajectory_entity_gap_artifact_size_mismatch")
        declared_gap_sha = record.get("gap_payload_sha256")
        actual_gap_sha = f"sha256:{hashlib.sha256(gap_payload).hexdigest()}"
        if (
            not isinstance(declared_gap_sha, str)
            or declared_gap_sha != actual_gap_sha
        ):
            _fail("trajectory_entity_gap_artifact_sha256_mismatch")
        if not gap_entries_declared or len(gap_entry_records) != gap_entry_count:
            _fail("trajectory_entity_gap_entries_missing")
        node_addresses = tuple(
            _integer(
                entry.get("node_address"),
                "trajectory_entity_gap_node_address_invalid",
                minimum=1,
            )
            for entry in gap_entry_records
        )
        for entry_index, entry in enumerate(gap_entry_records):
            offset = entry_index * BULLET_GAP_NODE_SIZE
            (
                next_node,
                previous_node,
                curve_index_value,
                gap_distance,
                boundary_ball_id,
            ) = struct.unpack_from("<IIiii", gap_payload, offset)
            expected_previous = (
                gap_list_sentinel_address
                if entry_index == 0
                else node_addresses[entry_index - 1]
            )
            expected_next = (
                gap_list_sentinel_address
                if entry_index + 1 == gap_entry_count
                else node_addresses[entry_index + 1]
            )
            if (
                previous_node != expected_previous
                or next_node != expected_next
                or entry.get("previous_node_address") != previous_node
                or entry.get("next_node_address") != next_node
                or gap_entries[entry_index]
                != (curve_index_value, gap_distance, boundary_ball_id)
            ):
                _fail("trajectory_entity_gap_artifact_semantics_mismatch")
    elif gap_entries_declared:
        _fail("trajectory_entity_gap_artifact_missing")
    return TrajectoryEntity(
        ball_id=ball_id,
        color_id=color,
        object_kind=kind,
        zone=zone,
        index=index,
        curve_distance=curve_distance,
        position_x=position_x,
        position_y=position_y,
        native_object_address=native_object_address,
        radius=radius,
        velocity_x=(
            _number(
                subclass.get("velocity_x"),
                "trajectory_entity_velocity_invalid",
            )
            if subclass is not None
            else None
        ),
        velocity_y=(
            _number(
                subclass.get("velocity_y"),
                "trajectory_entity_velocity_invalid",
            )
            if subclass is not None
            else None
        ),
        merge_progress=(
            _number(
                subclass.get("field_158_float"),
                "trajectory_entity_merge_progress_invalid",
            )
            if subclass is not None
            else None
        ),
        merge_speed=(
            _number(
                subclass.get("field_15c_float"),
                "trajectory_entity_merge_speed_invalid",
            )
            if subclass is not None
            else None
        ),
        fired=fired,
        contact_next=(bool(flags[0]) if flags is not None else None),
        exploding=(bool(flags[6]) if flags is not None else None),
        explode_frame=(
            int.from_bytes(flags[8:12], "little", signed=True)
            if flags is not None
            else None
        ),
        should_remove=(
            bool(flags[12]) if flags is not None else None
        ),
        update_count=update_count,
        suck_count=suck_count,
        backwards_count=backwards_count,
        backwards_speed=backwards_speed,
        combo_count=combo_count,
        combo_score=combo_score,
        powerup_previous_type=powerup_previous_type,
        powerup_primary_type=powerup_primary_type,
        powerup_secondary_type=powerup_secondary_type,
        powerup_previous_ticks=powerup_previous_ticks,
        powerup_lifetime_ticks=powerup_lifetime_ticks,
        powerup_transition_ticks=powerup_transition_ticks,
        powerup_visual_scale=powerup_visual_scale,
        powerup_visual_step=powerup_visual_step,
        powerup_visual_index=powerup_visual_index,
        gap_list_sentinel_address=gap_list_sentinel_address,
        gap_entry_count=gap_entry_count,
        curve_points=curve_points,
        gap_entries=gap_entries,
    )


def _frame_from_tick(
    tick: Mapping[str, Any],
    *,
    tick_root: Path | None = None,
    expected_barrier: str,
    expected_version: int = TRAJECTORY_TICK_VERSION,
    expected_sample_phase: str = TRAJECTORY_SAMPLE_PHASE,
) -> TrajectoryFrame:
    if (
        tick.get("schema") != TRAJECTORY_TICK_SCHEMA
        or tick.get("version") != expected_version
        or tick.get("sample_phase") != expected_sample_phase
        or tick.get("sample_barrier") != expected_barrier
    ):
        _fail("trajectory_tick_schema_invalid")
    update = _integer(
        tick.get("framework_update"),
        "trajectory_tick_update_invalid",
    )
    replay = _mapping(
        tick.get("replay_state"),
        "trajectory_tick_replay_state_invalid",
    )
    if (
        replay.get("update_count") != update
        or replay.get("frame_time_ms") != 10
        or not isinstance(replay.get("update_multiplier"), (int, float))
        or float(replay["update_multiplier"]) > 0.1
        or replay.get("fast_forward_to_marker") is not False
        or replay.get("fast_forward_step") is not False
        or not isinstance(replay.get("fast_forward_target"), int)
        or (
            expected_barrier == INITIAL_SAMPLE_BARRIER
            and not 0 <= replay["fast_forward_target"] <= update
        )
        or (
            expected_barrier == STEPPED_SAMPLE_BARRIER
            and replay["fast_forward_target"] != update
        )
    ):
        _fail("trajectory_tick_replay_state_invalid")

    board = _mapping(
        tick.get("active_board"),
        "trajectory_tick_board_invalid",
    )
    score = _integer(
        board.get("score"),
        "trajectory_tick_score_invalid",
    )
    displayed_score = _integer(
        board.get("displayed_score"),
        "trajectory_tick_displayed_score_invalid",
    )
    score_target = _integer(
        board.get("score_target"),
        "trajectory_tick_score_target_invalid",
    )
    declared_board_version = board.get("version")
    active_board_version: int | None = None
    if board.get("schema") is not None or declared_board_version is not None:
        if (
            board.get("schema") != ACTIVE_BOARD_SCHEMA
            or isinstance(declared_board_version, bool)
            or declared_board_version
            not in {ACTIVE_BOARD_LEGACY_VERSION, ACTIVE_BOARD_VERSION}
        ):
            _fail("trajectory_tick_active_board_schema_invalid")
        active_board_version = int(declared_board_version)
    native_game_time: int | None = None
    board_update_count: int | None = None
    board_runtime_flag_157: bool | None = None
    board_runtime_i32_f54: int | None = None
    board_mode_flag_1064: bool | None = None
    fruit_state: TrajectoryFruitState | None = None
    board_raw: bytes | None = None
    if board.get("artifact") is not None:
        board_raw = _declared_artifact_bytes(
            board,
            root=tick_root,
            code="trajectory_tick_board_artifact",
            allow_missing_size=True,
        )
        if len(board_raw) < BOARD_NATIVE_GAME_TIME_OFFSET + 4:
            _fail("trajectory_tick_board_artifact_size_mismatch")
        native_game_time = struct.unpack_from(
            "<i",
            board_raw,
            BOARD_NATIVE_GAME_TIME_OFFSET,
        )[0]
        if native_game_time < 0:
            _fail("trajectory_tick_native_game_time_invalid")
        board_update_count = struct.unpack_from(
            "<i",
            board_raw,
            BOARD_UPDATE_COUNT_OFFSET,
        )[0]
        if board_update_count < 0:
            _fail("trajectory_tick_board_update_count_invalid")
        runtime_flag = board_raw[BOARD_RUNTIME_FLAG_157_OFFSET]
        if runtime_flag not in {0, 1}:
            _fail("trajectory_tick_board_runtime_flag_157_invalid")
        board_runtime_flag_157 = bool(runtime_flag)
        board_runtime_i32_f54 = struct.unpack_from(
            "<i",
            board_raw,
            BOARD_RUNTIME_I32_F54_OFFSET,
        )[0]
        if active_board_version is not None:
            if len(board_raw) != BOARD_OBJECT_SIZE:
                _fail("trajectory_tick_board_artifact_size_mismatch")
            if (
                struct.unpack_from("<I", board_raw, 0)[0] != BOARD_VTABLE
                or struct.unpack_from(
                    "<I",
                    board_raw,
                    BOARD_EMBEDDED_VTABLE_OFFSET,
                )[0]
                != BOARD_EMBEDDED_VTABLE
            ):
                _fail("trajectory_tick_board_vtable_invalid")
            if (
                struct.unpack_from(
                    "<i",
                    board_raw,
                    BOARD_SCORE_OFFSET,
                )[0]
                != score
                or struct.unpack_from(
                    "<i",
                    board_raw,
                    BOARD_SCORE_TARGET_OFFSET,
                )[0]
                != score_target
                or struct.unpack_from(
                    "<i",
                    board_raw,
                    BOARD_DISPLAYED_SCORE_OFFSET,
                )[0]
                != displayed_score
            ):
                _fail("trajectory_tick_board_score_semantics_mismatch")
            if active_board_version == ACTIVE_BOARD_VERSION:
                mode_raw = board_raw[BOARD_MODE_FLAG_1064_OFFSET]
                declared_mode = board.get("mode_flag_1064")
                if (
                    mode_raw not in {0, 1}
                    or board.get("mode_flag_1064_offset")
                    != BOARD_MODE_FLAG_1064_OFFSET
                    or not isinstance(declared_mode, bool)
                    or declared_mode is not bool(mode_raw)
                ):
                    _fail(
                        "trajectory_tick_board_mode_flag_1064_invalid"
                    )
                board_mode_flag_1064 = bool(mode_raw)
                fruit_state = _fruit_state_from_board_raw(board_raw)
    elif active_board_version == ACTIVE_BOARD_VERSION:
        _fail("trajectory_tick_board_artifact_missing")

    curve_plan_exhausted: bool | None = None
    if active_board_version == ACTIVE_BOARD_VERSION:
        exhausted = _mapping(
            board.get("curve_plan_exhausted"),
            "trajectory_tick_curve_plan_exhausted_invalid",
        )
        if (
            exhausted.get("address")
            != G_CURVE_PLAN_EXHAUSTED_ADDRESS
        ):
            _fail("trajectory_tick_curve_plan_exhausted_address_invalid")
        exhausted_raw = _declared_artifact_bytes(
            exhausted,
            root=tick_root,
            code="trajectory_tick_curve_plan_exhausted_artifact",
            expected_size=1,
        )[0]
        declared_exhausted = exhausted.get("value")
        if (
            exhausted_raw not in {0, 1}
            or not isinstance(declared_exhausted, bool)
            or declared_exhausted is not bool(exhausted_raw)
        ):
            _fail("trajectory_tick_curve_plan_exhausted_semantics_invalid")
        curve_plan_exhausted = bool(exhausted_raw)
    shooter = _mapping(
        board.get("primary_child"),
        "trajectory_tick_shooter_invalid",
    )
    chamber = _sequence(
        shooter.get("bullets"),
        "trajectory_tick_chamber_invalid",
    )
    if len(chamber) != 2:
        _fail("trajectory_tick_chamber_invalid")

    entities: list[TrajectoryEntity] = []
    chamber_values: list[tuple[int | None, int | None]] = []
    for index, item in enumerate(chamber):
        row = _mapping(item, "trajectory_tick_chamber_invalid")
        if row.get("address") == 0:
            chamber_values.append((None, None))
            continue
        entity = _entity(
            row,
            zone="shooter_current" if index == 0 else "shooter_next",
            index=0,
            default_kind="bullet",
        )
        entities.append(entity)
        chamber_values.append((entity.ball_id, entity.color_id))

    fired = _mapping(
        board.get("fired_bullets"),
        "trajectory_tick_fired_invalid",
    )
    fired_records = _sequence(
        fired.get("records"),
        "trajectory_tick_fired_invalid",
    )
    if fired.get("traversed_count") != len(fired_records):
        _fail("trajectory_tick_fired_count_mismatch")
    fired_raw_blob: bytes | None = None
    if fired.get("artifact") is not None:
        fired_raw_blob = _declared_artifact_bytes(
            fired,
            root=tick_root,
            code="trajectory_tick_fired_artifact",
            expected_size=len(fired_records) * BULLET_OBJECT_SIZE,
        )
    gap_raw_blob: bytes | None = None
    if fired.get("gap_artifact") is not None:
        gap_size = sum(
            _integer(
                _mapping(
                    item,
                    "trajectory_tick_fired_record_invalid",
                ).get("gap_artifact_bytes"),
                "trajectory_tick_gap_record_size_invalid",
            )
            for item in fired_records
        )
        gap_raw_blob = _declared_artifact_bytes(
            {
                "artifact": fired.get("gap_artifact"),
                "artifact_bytes": fired.get("gap_artifact_bytes"),
                "artifact_sha256": fired.get("gap_artifact_sha256"),
            },
            root=tick_root,
            code="trajectory_tick_gap_artifact",
            expected_size=gap_size,
        )
        if fired_raw_blob is None:
            _fail("trajectory_tick_gap_artifact_without_fired_artifact")
    expected_fired_offset = 0
    expected_gap_offset = 0
    for index, item in enumerate(fired_records):
        record = _mapping(item, "trajectory_tick_fired_record_invalid")
        raw_payload: bytes | None = None
        if fired_raw_blob is not None:
            artifact_offset = _integer(
                record.get("artifact_offset"),
                "trajectory_tick_fired_record_offset_invalid",
            )
            artifact_bytes = _integer(
                record.get("artifact_bytes"),
                "trajectory_tick_fired_record_size_invalid",
                minimum=1,
            )
            if (
                artifact_offset != expected_fired_offset
                or artifact_bytes != BULLET_OBJECT_SIZE
            ):
                _fail("trajectory_tick_fired_record_layout_invalid")
            expected_fired_offset += artifact_bytes
            raw_payload = fired_raw_blob[
                artifact_offset:expected_fired_offset
            ]
        gap_payload: bytes | None = None
        if gap_raw_blob is not None:
            gap_offset = _integer(
                record.get("gap_artifact_offset"),
                "trajectory_tick_gap_record_offset_invalid",
            )
            gap_bytes = _integer(
                record.get("gap_artifact_bytes"),
                "trajectory_tick_gap_record_size_invalid",
            )
            if gap_offset != expected_gap_offset:
                _fail("trajectory_tick_gap_record_layout_invalid")
            expected_gap_offset += gap_bytes
            gap_payload = gap_raw_blob[gap_offset:expected_gap_offset]
        entities.append(
            _entity(
                record,
                zone="fired",
                index=index,
                default_kind="bullet",
                raw_payload=raw_payload,
                gap_payload=gap_payload,
            )
        )
    if (
        fired_raw_blob is not None
        and expected_fired_offset != len(fired_raw_blob)
    ):
        _fail("trajectory_tick_fired_record_layout_invalid")
    if gap_raw_blob is not None and expected_gap_offset != len(gap_raw_blob):
        _fail("trajectory_tick_gap_record_layout_invalid")

    manager = _mapping(
        board.get("curve_manager"),
        "trajectory_tick_curve_manager_invalid",
    )
    curves = _sequence(
        manager.get("curves"),
        "trajectory_tick_curves_invalid",
    )
    post_zuma_timer_remaining: int | None = None
    post_zuma_ramp_404: float | None = None
    post_zuma_ramp_408: float | None = None
    manager_raw: bytes | None = None
    if active_board_version == ACTIVE_BOARD_VERSION:
        manager_raw = _declared_artifact_bytes(
            manager,
            root=tick_root,
            code="trajectory_tick_curve_manager_artifact",
            expected_size=CURVE_MANAGER_OBJECT_SIZE,
            allow_missing_size=True,
        )
        if (
            struct.unpack_from("<I", manager_raw, 0)[0]
            != CURVE_MANAGER_VTABLE
        ):
            _fail("trajectory_tick_curve_manager_vtable_invalid")
        manager_curve_count = struct.unpack_from(
            "<i",
            manager_raw,
            CURVE_MANAGER_CURVE_COUNT_OFFSET,
        )[0]
        if (
            manager_curve_count <= 0
            or manager_curve_count != len(curves)
            or manager.get("curve_count") != manager_curve_count
            or manager.get("curve_count_offset")
            != CURVE_MANAGER_CURVE_COUNT_OFFSET
            or manager.get("curve_array_offset")
            != CURVE_MANAGER_CURVE_ARRAY_OFFSET
        ):
            _fail("trajectory_tick_curve_manager_count_invalid")
        post_zuma_timer_remaining = struct.unpack_from(
            "<i",
            manager_raw,
            CURVE_MANAGER_POST_ZUMA_TIMER_OFFSET,
        )[0]
        post_zuma_ramp_404 = struct.unpack_from(
            "<f",
            manager_raw,
            CURVE_MANAGER_POST_ZUMA_RAMP_404_OFFSET,
        )[0]
        post_zuma_ramp_408 = struct.unpack_from(
            "<f",
            manager_raw,
            CURVE_MANAGER_POST_ZUMA_RAMP_408_OFFSET,
        )[0]
        if not (
            math.isfinite(post_zuma_ramp_404)
            and math.isfinite(post_zuma_ramp_408)
        ):
            _fail("trajectory_tick_post_zuma_diagnostics_invalid")
        diagnostics = _mapping(
            manager.get("post_zuma_diagnostics"),
            "trajectory_tick_post_zuma_diagnostics_invalid",
        )
        if (
            diagnostics.get("timer_remaining_offset")
            != CURVE_MANAGER_POST_ZUMA_TIMER_OFFSET
            or diagnostics.get("timer_remaining")
            != post_zuma_timer_remaining
            or diagnostics.get("ramp_404_offset")
            != CURVE_MANAGER_POST_ZUMA_RAMP_404_OFFSET
            or diagnostics.get("ramp_404") != post_zuma_ramp_404
            or diagnostics.get("ramp_408_offset")
            != CURVE_MANAGER_POST_ZUMA_RAMP_408_OFFSET
            or diagnostics.get("ramp_408") != post_zuma_ramp_408
        ):
            _fail(
                "trajectory_tick_post_zuma_diagnostics_semantics_mismatch"
            )
    list_counts: list[tuple[int, int, int]] = []
    curve_powerups: list[TrajectoryCurvePowerupState] = []
    curve_plans: list[TrajectoryCurvePlanState] = []
    curve_runtime_states: list[TrajectoryCurveRuntimeState] = []
    for curve_position, curve_item in enumerate(curves):
        curve = _mapping(
            curve_item,
            "trajectory_tick_curve_invalid",
        )
        curve_index = _integer(
            curve.get("index"),
            "trajectory_tick_curve_index_invalid",
        )
        if active_board_version == ACTIVE_BOARD_VERSION:
            if curve_index != curve_position or manager_raw is None:
                _fail("trajectory_tick_curve_index_invalid")
            expected_curve_address = struct.unpack_from(
                "<I",
                manager_raw,
                CURVE_MANAGER_CURVE_ARRAY_OFFSET + curve_position * 4,
            )[0]
            if curve.get("address") != expected_curve_address:
                _fail("trajectory_tick_curve_address_invalid")
            curve_plans.append(
                _curve_plan_state(
                    curve,
                    curve_index=curve_index,
                    tick_root=tick_root,
                )
            )
            curve_runtime_states.append(
                _curve_runtime_state(
                    curve,
                    curve_index=curve_index,
                    tick_root=tick_root,
                )
            )
        curve_powerup = _curve_powerup_state(
            curve,
            curve_index=curve_index,
            tick_root=tick_root,
        )
        if curve_powerup is not None:
            curve_powerups.append(curve_powerup)
        lists = _sequence(
            curve.get("intrusive_lists"),
            "trajectory_tick_curve_lists_invalid",
        )
        for list_item in lists:
            intrusive = _mapping(
                list_item,
                "trajectory_tick_curve_list_invalid",
            )
            offset = _integer(
                intrusive.get("container_offset"),
                "trajectory_tick_curve_list_offset_invalid",
            )
            records = _sequence(
                intrusive.get("records"),
                "trajectory_tick_curve_list_records_invalid",
            )
            count = _integer(
                intrusive.get("traversed_count"),
                "trajectory_tick_curve_list_count_invalid",
            )
            if count != len(records):
                _fail("trajectory_tick_curve_list_count_mismatch")
            raw_blob: bytes | None = None
            if intrusive.get("artifact") is not None:
                if tick_root is None:
                    _fail("trajectory_tick_raw_root_missing")
                raw_path = _artifact_path(
                    tick_root,
                    intrusive.get("artifact"),
                )
                try:
                    raw_blob = raw_path.read_bytes()
                except OSError as error:
                    raise PcMemoryTrajectoryError(
                        "trajectory_tick_raw_artifact_missing"
                    ) from error
                declared_sha = intrusive.get("artifact_sha256")
                if (
                    not isinstance(declared_sha, str)
                    or f"sha256:{hashlib.sha256(raw_blob).hexdigest()}"
                    != declared_sha
                ):
                    _fail("trajectory_tick_raw_artifact_sha256_mismatch")
                declared_bytes = intrusive.get("artifact_bytes")
                if (
                    declared_bytes is not None
                    and declared_bytes != len(raw_blob)
                ):
                    _fail("trajectory_tick_raw_artifact_size_mismatch")
            list_counts.append((curve_index, offset, count))
            zone = f"curve:{curve_index}:list:{offset:03x}"
            for index, item in enumerate(records):
                record = _mapping(
                    item,
                    "trajectory_tick_curve_record_invalid",
                )
                raw_payload: bytes | None = None
                if raw_blob is not None:
                    artifact_offset = _integer(
                        record.get("artifact_offset"),
                        "trajectory_tick_raw_record_offset_invalid",
                    )
                    artifact_bytes = _integer(
                        record.get("artifact_bytes"),
                        "trajectory_tick_raw_record_size_invalid",
                        minimum=1,
                    )
                    end_offset = artifact_offset + artifact_bytes
                    if end_offset > len(raw_blob):
                        _fail("trajectory_tick_raw_record_bounds_invalid")
                    raw_payload = raw_blob[artifact_offset:end_offset]
                entities.append(
                    _entity(
                        record,
                        zone=zone,
                        index=index,
                        default_kind="ball",
                        raw_payload=raw_payload,
                    )
                )

    identities = [entity.native_identity for entity in entities]
    if len(set(identities)) != len(identities):
        _fail("trajectory_tick_duplicate_entity_identity")
    ids: dict[int, list[TrajectoryEntity]] = {}
    for entity in entities:
        ids.setdefault(entity.ball_id, []).append(entity)
    if any(
        len(rows) > 1
        and any(row.native_object_address is None for row in rows)
        for rows in ids.values()
    ):
        _fail("trajectory_tick_duplicate_ball_id")
    list_counts.sort()
    qrand = _qrand_from_board(board, tick_root=tick_root)
    thread_crt_rand_state = _thread_crt_rand_from_board(
        board,
        tick_root=tick_root,
    )
    global_mtrand = _mtrand_from_board(
        board,
        key="global_mtrand",
        role="global",
        tick_root=tick_root,
    )
    return TrajectoryFrame(
        update=update,
        score=score,
        displayed_score=displayed_score,
        score_target=score_target,
        current_ball_id=chamber_values[0][0],
        current_color_id=chamber_values[0][1],
        next_ball_id=chamber_values[1][0],
        next_color_id=chamber_values[1][1],
        list_counts=tuple(list_counts),
        entities=tuple(entities),
        qrand=qrand,
        thread_crt_rand_state=thread_crt_rand_state,
        global_mtrand=global_mtrand,
        native_game_time=native_game_time,
        board_update_count=board_update_count,
        curve_powerups=tuple(curve_powerups),
        board_runtime_flag_157=board_runtime_flag_157,
        board_runtime_i32_f54=board_runtime_i32_f54,
        active_board_version=active_board_version,
        board_mode_flag_1064=board_mode_flag_1064,
        curve_plan_exhausted=curve_plan_exhausted,
        post_zuma_timer_remaining=post_zuma_timer_remaining,
        post_zuma_ramp_404=post_zuma_ramp_404,
        post_zuma_ramp_408=post_zuma_ramp_408,
        curve_plans=tuple(curve_plans),
        curve_runtime_states=tuple(curve_runtime_states),
        fruit_state=fruit_state,
    )


def _index_summary(frame: TrajectoryFrame) -> dict[str, Any]:
    result = {
        "framework_update": frame.update,
        "score": frame.score,
        "displayed_score": frame.displayed_score,
        "score_target": frame.score_target,
        "chain_ball_count": sum(
            count
            for _, offset, count in frame.list_counts
            if offset == ACTIVE_CHAIN_LIST_OFFSET
        ),
        "fired_bullet_count": sum(
            entity.zone == "fired" for entity in frame.entities
        ),
        "current_ball_id": frame.current_ball_id,
        "current_color_id": frame.current_color_id,
        "next_ball_id": frame.next_ball_id,
        "next_color_id": frame.next_color_id,
        "list_counts": [
            {
                "curve_index": curve,
                "container_offset": offset,
                "count": count,
            }
            for curve, offset, count in frame.list_counts
        ],
    }
    if frame.qrand is not None:
        result["qrand_update_count"] = frame.qrand.update_count
        result["qrand_selected_index"] = frame.qrand.selected_index
    if frame.thread_crt_rand_state is not None:
        result["thread_crt_rand_state"] = frame.thread_crt_rand_state
    if frame.global_mtrand is not None:
        result["global_mtrand_index"] = frame.global_mtrand.index
    if frame.active_board_version is not None:
        result["active_board_version"] = frame.active_board_version
    if frame.board_mode_flag_1064 is not None:
        result["board_mode_flag_1064"] = frame.board_mode_flag_1064
    if frame.curve_plan_exhausted is not None:
        result["curve_plan_exhausted"] = frame.curve_plan_exhausted
    if frame.post_zuma_timer_remaining is not None:
        result["post_zuma_timer_remaining"] = (
            frame.post_zuma_timer_remaining
        )
    if frame.post_zuma_ramp_404 is not None:
        result["post_zuma_ramp_404"] = frame.post_zuma_ramp_404
    if frame.post_zuma_ramp_408 is not None:
        result["post_zuma_ramp_408"] = frame.post_zuma_ramp_408
    if frame.curve_plans:
        result["curve_plans"] = [
            {
                "curve_index": state.curve_index,
                "begin_address": state.begin_address,
                "end_address": state.end_address,
                "capacity_address": state.capacity_address,
                "planned_count": state.planned_count,
                "capacity_count": state.capacity_count,
                "add_plan_enabled": state.add_plan_enabled,
            }
            for state in frame.curve_plans
        ]
    return result


def load_legacy_memory_trajectory_v1(
    index_path: Path,
    *,
    start_update: int | None = None,
    end_update: int | None = None,
) -> tuple[TrajectoryFrame, ...]:
    """Strictly decode immutable pre-v2 trajectories without upgrading them.

    Version 1 predates the explicit replay-barrier receipt and the additional
    runtime fields in version 2.  This decoder keeps that evidence class
    visible and separate: it verifies the original canonical bytes, hashes,
    core summaries, and raw child artifacts, but never relabels a v1 source as
    v2.  Callers must explicitly opt in and preserve the source version in any
    derived report.
    """

    index_path = index_path.resolve()
    index = _read_canonical_json(index_path)
    if (
        index.get("schema") != TRAJECTORY_SCHEMA
        or index.get("version") != LEGACY_TRAJECTORY_VERSION
        or index.get("sample_phase") != LEGACY_TRAJECTORY_SAMPLE_PHASE
    ):
        _fail("legacy_trajectory_index_schema_invalid")
    start = _integer(
        index.get("start_update"),
        "legacy_trajectory_start_update_invalid",
    )
    end = _integer(
        index.get("end_update"),
        "legacy_trajectory_end_update_invalid",
    )
    count = _integer(
        index.get("tick_count"),
        "legacy_trajectory_tick_count_invalid",
        minimum=1,
    )
    rows = _sequence(
        index.get("ticks"),
        "legacy_trajectory_ticks_invalid",
    )
    if end < start or count != end - start + 1 or len(rows) != count:
        _fail("legacy_trajectory_update_range_invalid")

    row_fields = {
        "artifact",
        "artifact_sha256",
        "chain_ball_count",
        "current_ball_id",
        "current_color_id",
        "displayed_score",
        "fired_bullet_count",
        "framework_update",
        "list_counts",
        "next_ball_id",
        "next_color_id",
        "score",
        "score_target",
    }
    indexed_rows: list[Mapping[str, Any]] = []
    for expected_update, item in zip(range(start, end + 1), rows):
        row = _mapping(item, "legacy_trajectory_index_tick_invalid")
        if set(row) != row_fields or row.get(
            "framework_update"
        ) != expected_update:
            _fail("legacy_trajectory_index_tick_invalid")
        indexed_rows.append(row)

    selected_start = start if start_update is None else _integer(
        start_update,
        "legacy_trajectory_window_start_update_invalid",
    )
    selected_end = end if end_update is None else _integer(
        end_update,
        "legacy_trajectory_window_end_update_invalid",
    )
    if not start <= selected_start <= selected_end <= end:
        _fail("legacy_trajectory_window_range_invalid")

    replay_fields = {
        "draw_count",
        "fast_forward_target",
        "frame_time_ms",
        "loaded",
        "loading_thread_completed",
        "loading_thread_started",
        "multiplier_address",
        "non_draw_count",
        "paused",
        "sleep_count",
        "update_app_depth",
        "update_app_state",
        "update_count",
        "update_multiplier",
    }
    frames: list[TrajectoryFrame] = []
    root = index_path.parent
    first_offset = selected_start - start
    selected_rows = indexed_rows[
        first_offset : first_offset + selected_end - selected_start + 1
    ]
    legacy_barrier = "legacy_frozen_post_framework_update"
    for expected_update, row in zip(
        range(selected_start, selected_end + 1),
        selected_rows,
        strict=True,
    ):
        tick_path = _artifact_path(root, row.get("artifact"))
        expected_sha = row.get("artifact_sha256")
        if (
            not isinstance(expected_sha, str)
            or _sha256_path(tick_path) != expected_sha
        ):
            _fail("legacy_trajectory_tick_artifact_sha256_mismatch")
        original_tick = _read_canonical_json(tick_path)
        original_replay = _mapping(
            original_tick.get("replay_state"),
            "legacy_trajectory_tick_replay_state_invalid",
        )
        if (
            original_tick.get("schema") != TRAJECTORY_TICK_SCHEMA
            or original_tick.get("version")
            != LEGACY_TRAJECTORY_TICK_VERSION
            or original_tick.get("sample_phase")
            != LEGACY_TRAJECTORY_SAMPLE_PHASE
            or "sample_barrier" in original_tick
            or set(original_replay) != replay_fields
        ):
            _fail("legacy_trajectory_tick_transport_invalid")

        # Feed the common payload decoder an explicit representation of the
        # legacy contract.  Only transport metadata absent from v1 is added;
        # all gameplay, entity, and raw-artifact fields remain untouched.
        adapted_tick = dict(original_tick)
        adapted_tick["sample_barrier"] = legacy_barrier
        adapted_replay = dict(original_replay)
        adapted_replay["fast_forward_to_marker"] = False
        adapted_replay["fast_forward_step"] = False
        adapted_tick["replay_state"] = adapted_replay
        frame = _frame_from_tick(
            adapted_tick,
            tick_root=tick_path.parent,
            expected_barrier=legacy_barrier,
            expected_version=LEGACY_TRAJECTORY_TICK_VERSION,
            expected_sample_phase=LEGACY_TRAJECTORY_SAMPLE_PHASE,
        )
        if frame.update != expected_update:
            _fail("legacy_trajectory_tick_update_mismatch")
        summary = _index_summary(frame)
        for key in row_fields - {"artifact", "artifact_sha256"}:
            if row.get(key) != summary.get(key):
                _fail(f"legacy_trajectory_index_summary_mismatch:{key}")
        frames.append(frame)
    return tuple(frames)


def load_memory_trajectory(
    index_path: Path,
    *,
    start_update: int | None = None,
    end_update: int | None = None,
) -> tuple[TrajectoryFrame, ...]:
    """Load and verify all ticks, or one inclusive contiguous tick window.

    Windowed loading still validates the complete compact index topology, then
    hashes and decodes only the selected tick artifacts.  This keeps bounded
    differential checks practical for long retail trajectories without
    weakening verification of any tick that participates in the comparison.
    """

    index_path = index_path.resolve()
    index = _read_canonical_json(index_path)
    if (
        index.get("schema") != TRAJECTORY_SCHEMA
        or index.get("version") != TRAJECTORY_VERSION
        or index.get("sample_phase") != TRAJECTORY_SAMPLE_PHASE
    ):
        _fail("trajectory_index_schema_invalid")
    start = _integer(
        index.get("start_update"),
        "trajectory_start_update_invalid",
    )
    end = _integer(
        index.get("end_update"),
        "trajectory_end_update_invalid",
    )
    count = _integer(
        index.get("tick_count"),
        "trajectory_tick_count_invalid",
        minimum=1,
    )
    rows = _sequence(index.get("ticks"), "trajectory_ticks_invalid")
    if end < start or count != end - start + 1 or len(rows) != count:
        _fail("trajectory_update_range_invalid")

    indexed_rows: list[Mapping[str, Any]] = []
    for expected_update, item in zip(range(start, end + 1), rows):
        row = _mapping(item, "trajectory_index_tick_invalid")
        if row.get("framework_update") != expected_update:
            _fail("trajectory_index_tick_update_invalid")
        indexed_rows.append(row)

    selected_start = start if start_update is None else _integer(
        start_update,
        "trajectory_window_start_update_invalid",
    )
    selected_end = end if end_update is None else _integer(
        end_update,
        "trajectory_window_end_update_invalid",
    )
    if not start <= selected_start <= selected_end <= end:
        _fail("trajectory_window_range_invalid")

    frames: list[TrajectoryFrame] = []
    root = index_path.parent
    requires_extended_index_summary = bool(
        _V2_EXTENDED_INDEX_MARKERS.intersection(index)
    )
    first_offset = selected_start - start
    selected_rows = indexed_rows[
        first_offset : first_offset + selected_end - selected_start + 1
    ]
    for expected_update, row in zip(
        range(selected_start, selected_end + 1),
        selected_rows,
        strict=True,
    ):
        tick_path = _artifact_path(root, row.get("artifact"))
        expected_sha = row.get("artifact_sha256")
        if (
            not isinstance(expected_sha, str)
            or _sha256_path(tick_path) != expected_sha
        ):
            _fail("trajectory_tick_artifact_sha256_mismatch")
        frame = _frame_from_tick(
            _read_canonical_json(tick_path),
            tick_root=tick_path.parent,
            expected_barrier=(
                INITIAL_SAMPLE_BARRIER
                if expected_update == start
                else STEPPED_SAMPLE_BARRIER
            ),
        )
        if frame.update != expected_update:
            _fail("trajectory_tick_update_mismatch")
        summary = _index_summary(frame)
        for key, value in summary.items():
            if (
                key not in row
                and key in _V2_EXTENDED_INDEX_SUMMARY_FIELDS
                and not requires_extended_index_summary
            ):
                continue
            if row.get(key) != value:
                _fail(f"trajectory_index_summary_mismatch:{key}")
        frames.append(frame)
    return tuple(frames)


def _distance(first: TrajectoryEntity, second: TrajectoryEntity) -> float:
    return math.hypot(
        second.position_x - first.position_x,
        second.position_y - first.position_y,
    )


def derive_trajectory_events(
    frames: Sequence[TrajectoryFrame],
) -> tuple[Mapping[str, Any], ...]:
    if not frames:
        _fail("trajectory_frames_empty")
    events: list[Mapping[str, Any]] = []
    for before, after in zip(frames, frames[1:]):
        if after.update != before.update + 1:
            _fail("trajectory_frames_not_contiguous")
        before_entities = before.entities_by_native_identity
        after_entities = after.entities_by_native_identity
        if (
            len(before_entities) != len(before.entities)
            or len(after_entities) != len(after.entities)
        ):
            _fail("trajectory_frame_duplicate_native_identity")

        if (
            before.current_ball_id,
            before.current_color_id,
            before.next_ball_id,
            before.next_color_id,
        ) != (
            after.current_ball_id,
            after.current_color_id,
            after.next_ball_id,
            after.next_color_id,
        ):
            events.append(
                {
                    "update": after.update,
                    "kind": "shooter_chamber_change",
                    "before": {
                        "current_ball_id": before.current_ball_id,
                        "current_color_id": before.current_color_id,
                        "next_ball_id": before.next_ball_id,
                        "next_color_id": before.next_color_id,
                    },
                    "after": {
                        "current_ball_id": after.current_ball_id,
                        "current_color_id": after.current_color_id,
                        "next_ball_id": after.next_ball_id,
                        "next_color_id": after.next_color_id,
                    },
                }
            )

        before_fired = {
            entity.native_identity: entity
            for entity in before.entities
            if entity.zone == "fired"
        }
        after_fired = {
            entity.native_identity: entity
            for entity in after.entities
            if entity.zone == "fired"
        }
        for identity in sorted(after_fired.keys() - before_fired.keys()):
            events.append(
                {
                    "update": after.update,
                    "kind": "projectile_spawn",
                    "entity": after_fired[identity].to_dict(),
                }
            )

        for identity in sorted(
            before_entities.keys() & after_entities.keys()
        ):
            first = before_entities[identity]
            second = after_entities[identity]
            if first.zone == second.zone:
                continue
            kind = "entity_zone_change"
            if (
                first.zone == "fired"
                and second.zone.endswith(
                    f"list:{INSERTION_STAGING_LIST_OFFSET:03x}"
                )
            ):
                kind = "projectile_hit_staging"
            events.append(
                {
                    "update": after.update,
                    "kind": kind,
                    "ball_id": second.ball_id,
                    "color_id": second.color_id,
                    "object_kind": second.object_kind,
                    "from_zone": first.zone,
                    "to_zone": second.zone,
                    "position_x": second.position_x,
                    "position_y": second.position_y,
                    "curve_distance": second.curve_distance,
                }
            )

        removed = [
            before_entities[ball_id]
            for ball_id in sorted(
                before_entities.keys() - after_entities.keys()
            )
        ]
        added = [
            after_entities[ball_id]
            for ball_id in sorted(
                after_entities.keys() - before_entities.keys()
            )
        ]
        paired_removed: set[tuple[int, int]] = set()
        paired_added: set[tuple[int, int]] = set()
        candidates: list[
            tuple[
                float,
                tuple[int, int],
                tuple[int, int],
                TrajectoryEntity,
                TrajectoryEntity,
            ]
        ] = []
        for first in removed:
            if not first.zone.endswith(
                f"list:{INSERTION_STAGING_LIST_OFFSET:03x}"
            ):
                continue
            for second in added:
                if (
                    first.color_id != second.color_id
                    or not second.zone.endswith(
                        f"list:{ACTIVE_CHAIN_LIST_OFFSET:03x}"
                    )
                ):
                    continue
                spatial = _distance(first, second)
                if (
                    spatial <= 4.0
                    and abs(
                        second.curve_distance - first.curve_distance
                    )
                    <= 2.0
                ):
                    candidates.append(
                        (
                            spatial,
                            first.native_identity,
                            second.native_identity,
                            first,
                            second,
                        )
                    )
        for _, first_identity, second_identity, first, second in sorted(
            candidates,
            key=lambda row: (row[0], row[1], row[2]),
        ):
            if (
                first_identity in paired_removed
                or second_identity in paired_added
            ):
                continue
            paired_removed.add(first_identity)
            paired_added.add(second_identity)
            events.append(
                {
                    "update": after.update,
                    "kind": "insertion_commit",
                    "source_entity": first.to_dict(),
                    "inserted_entity": second.to_dict(),
                    "spatial_error_px": _distance(first, second),
                    "curve_distance_error": (
                        second.curve_distance - first.curve_distance
                    ),
                }
            )
        for entity in removed:
            if entity.native_identity not in paired_removed:
                events.append(
                    {
                        "update": after.update,
                        "kind": "entity_removed",
                        "entity": entity.to_dict(),
                    }
                )
        for entity in added:
            if (
                entity.native_identity not in paired_added
                and entity.native_identity not in after_fired
            ):
                events.append(
                    {
                        "update": after.update,
                        "kind": "entity_added",
                        "entity": entity.to_dict(),
                    }
                )
            if entity.exploding is True:
                events.append(
                    {
                        "update": after.update,
                        "kind": "explosion_started",
                        "ball_id": entity.ball_id,
                        "color_id": entity.color_id,
                        "curve_distance": entity.curve_distance,
                        "explode_frame": entity.explode_frame,
                        "combo_count": entity.combo_count,
                        "combo_score": entity.combo_score,
                    }
                )

        for identity in sorted(
            before_entities.keys() & after_entities.keys()
        ):
            previous = before_entities[identity]
            current = after_entities[identity]
            if (
                previous.exploding is False
                and current.exploding is True
            ):
                events.append(
                    {
                        "update": after.update,
                        "kind": "explosion_started",
                        "ball_id": current.ball_id,
                        "color_id": current.color_id,
                        "curve_distance": current.curve_distance,
                        "explode_frame": current.explode_frame,
                        "combo_count": current.combo_count,
                        "combo_score": current.combo_score,
                    }
                )
            if (
                previous.should_remove is False
                and current.should_remove is True
            ):
                events.append(
                    {
                        "update": after.update,
                        "kind": "explosion_removal_armed",
                        "ball_id": current.ball_id,
                        "color_id": current.color_id,
                        "explode_frame": current.explode_frame,
                    }
                )
            if (
                previous.suck_count == 0
                and current.suck_count is not None
                and current.suck_count > 0
            ):
                events.append(
                    {
                        "update": after.update,
                        "kind": "rollback_started",
                        "ball_id": current.ball_id,
                        "color_id": current.color_id,
                        "suck_count": current.suck_count,
                        "combo_count": current.combo_count,
                        "combo_score": current.combo_score,
                    }
                )
            if (
                previous.suck_count is not None
                and previous.suck_count > 0
                and current.suck_count == 0
            ):
                events.append(
                    {
                        "update": after.update,
                        "kind": "rollback_stopped",
                        "ball_id": current.ball_id,
                        "color_id": current.color_id,
                        "combo_count": current.combo_count,
                        "combo_score": current.combo_score,
                    }
                )

        all_lists = sorted(set(before.list_counts) | set(after.list_counts))
        list_keys = sorted(
            {
                (curve, offset)
                for curve, offset, _ in all_lists
            }
        )
        for curve, offset in list_keys:
            before_count = before.list_count(curve, offset)
            after_count = after.list_count(curve, offset)
            if before_count != after_count:
                events.append(
                    {
                        "update": after.update,
                        "kind": "curve_list_count_change",
                        "curve_index": curve,
                        "container_offset": offset,
                        "before": before_count,
                        "after": after_count,
                    }
                )
        if (
            before.score,
            before.displayed_score,
        ) != (
            after.score,
            after.displayed_score,
        ):
            events.append(
                {
                    "update": after.update,
                    "kind": "score_change",
                    "score_before": before.score,
                    "score_after": after.score,
                    "displayed_score_before": before.displayed_score,
                    "displayed_score_after": after.displayed_score,
                }
            )
    events.sort(
        key=lambda row: (
            int(row["update"]),
            str(row["kind"]),
        )
    )
    return tuple(events)


def analyze_memory_trajectory(index_path: Path) -> Mapping[str, Any]:
    frames = load_memory_trajectory(index_path)
    events = derive_trajectory_events(frames)
    counts: dict[str, int] = {}
    for event in events:
        kind = str(event["kind"])
        counts[kind] = counts.get(kind, 0) + 1
    return {
        "schema": TRAJECTORY_ANALYSIS_SCHEMA,
        "version": TRAJECTORY_ANALYSIS_VERSION,
        "status": "PASS",
        "trajectory_index_sha256": _sha256_path(index_path.resolve()),
        "start_update": frames[0].update,
        "end_update": frames[-1].update,
        "tick_count": len(frames),
        "event_count": len(events),
        "event_kind_counts": dict(sorted(counts.items())),
        "events": list(events),
    }


def canonical_analysis_bytes(report: Mapping[str, Any]) -> bytes:
    return _canonical_bytes(report)


__all__ = [
    "ACTIVE_CHAIN_LIST_OFFSET",
    "BALL_OBJECT_SIZE",
    "BOARD_FRUIT_ACCELERATION_OFFSET",
    "BOARD_FRUIT_ALPHA_OFFSET",
    "BOARD_FRUIT_CELL_INDEX_OFFSET",
    "BOARD_FRUIT_COLLECTING_OFFSET",
    "BOARD_FRUIT_EXPIRY_TIME_OFFSET",
    "BOARD_FRUIT_GLOW_ALPHA_OFFSET",
    "BOARD_FRUIT_GLOW_STEP_OFFSET",
    "BOARD_FRUIT_LOWER_BOUND_OFFSET",
    "BOARD_FRUIT_MAX_VELOCITY_OFFSET",
    "BOARD_FRUIT_POSITION_OFFSET",
    "BOARD_FRUIT_SELECTED_POINT_OFFSET",
    "BOARD_FRUIT_UPPER_BOUND_OFFSET",
    "BOARD_FRUIT_VELOCITY_OFFSET",
    "BOARD_FRUIT_VERTICAL_OFFSET",
    "BOARD_NATIVE_GAME_TIME_OFFSET",
    "BOARD_RUNTIME_FLAG_157_OFFSET",
    "BOARD_RUNTIME_I32_F54_OFFSET",
    "CURVE_OBJECT_SIZE",
    "INSERTION_STAGING_LIST_OFFSET",
    "LEGACY_TRAJECTORY_SAMPLE_PHASE",
    "LEGACY_TRAJECTORY_TICK_VERSION",
    "LEGACY_TRAJECTORY_VERSION",
    "POWERUP_NONE_TYPE",
    "POWERUP_TYPE_COUNT",
    "PcMemoryTrajectoryError",
    "TRAJECTORY_ANALYSIS_SCHEMA",
    "TRAJECTORY_SCHEMA",
    "TrajectoryCurvePlanState",
    "TrajectoryCurvePowerupState",
    "TrajectoryCurveRuntimeState",
    "TrajectoryEntity",
    "TrajectoryFrame",
    "TrajectoryFruitState",
    "analyze_memory_trajectory",
    "canonical_analysis_bytes",
    "derive_trajectory_events",
    "load_legacy_memory_trajectory_v1",
    "load_memory_trajectory",
]
