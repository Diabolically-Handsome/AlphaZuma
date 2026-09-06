"""Tick-accurate research core for the original Zuma's Revenge rules.

This module is deliberately separate from :mod:`zuma_rl.core`.  The latter is
the early, fast ``ZumaSimple-v0`` prototype; this module is the fidelity-first
implementation that consumes decoded original ``CURV`` data.

The implementation follows the original 100 Hz ordering and waypoint-index
coordinate system.  Features whose PC behaviour has not yet been calibrated
(notably bosses and ambient visual consumers of the shared global MTRand
stream) are intentionally not hidden behind approximations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

Float32Array = NDArray[np.float32]
SUPPORTED_PROFILE_MODE = "tutorials_completed"

_F32_ZERO = np.float32(0.0)
_F32_ONE = np.float32(1.0)
_F32_HALF = np.float32(0.5)
_F32_FIVE = np.float32(5.0)
_F32_TEN = np.float32(10.0)


def _f32_add(left: float, right: float) -> np.float32:
    """One game-float addition, including its write-back rounding."""

    return np.float32(left) + np.float32(right)


def _f32_sub(left: float, right: float) -> np.float32:
    """One game-float subtraction, including its write-back rounding."""

    return np.float32(left) - np.float32(right)


def _f32_mul(left: float, right: float) -> np.float32:
    """One game-float multiplication, including its write-back rounding."""

    return np.float32(left) * np.float32(right)


def _f32_div(left: float, right: float) -> np.float32:
    """One game-float division, including its write-back rounding."""

    return np.float32(left) / np.float32(right)


def _freeze_signature_value(value: Any) -> Any:
    """Convert nested NumPy/RNG state into an exact hashable representation."""

    if isinstance(value, dict):
        return tuple(
            (key, _freeze_signature_value(item))
            for key, item in sorted(value.items())
        )
    if isinstance(value, np.ndarray):
        return (
            value.dtype.str,
            tuple(value.shape),
            tuple(value.reshape(-1).tolist()),
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_signature_value(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


class WaypointCurve(Protocol):
    """Structural interface shared by original and synthetic test curves."""

    parameters: Any
    end_waypoint: int
    die_at_end: bool

    def point_at_waypoint(
        self,
        waypoint: float | Iterable[float],
        *,
        loop_at_end: bool = False,
    ) -> Float32Array: ...

    def perpendicular_at_waypoint(
        self,
        waypoint: float | Iterable[float],
    ) -> Float32Array: ...

    def is_in_tunnel_at_waypoint(
        self,
        waypoint: float | Iterable[float],
    ) -> NDArray[np.bool_]: ...


@dataclass(frozen=True, slots=True)
class RevengePhysicsConfig:
    """PC physics constants and explicit calibration switches.

    ``projectile_speed=8`` is from the native PC executable.  The readable
    XNA/WP7 port uses 9.6 instead, so that value must not silently replace the
    PC constant.  ``ball_radius=18`` is the accelerated-renderer branch and is
    still a fidelity-gate item until a PC capture confirms which branch the
    user's installation takes.
    """

    tick_hz: int = 100
    logical_width: int = 800
    logical_height: int = 600
    ball_radius: int = 18
    projectile_speed: float = 8.0
    accuracy_projectile_speed: float = 19.0
    merge_speed: float = 0.025
    explosion_ticks: int = 40
    danger_distance: int = 600
    initial_pending_balls: int = 10
    zuma_bar_width: int = 330
    fire_state_step: float = 0.15
    release_threshold: float = 0.9
    reload_state_step: float = 0.07
    # A non-tutorial 29.97 fps retail hop spans six frame intervals from the
    # last stable source pose to the first stable destination pose.
    # 6 / 29.97 = 0.200 s, or 20 native 10 ms Board updates.
    hop_transition_ticks: int = 20
    muzzle_forward_offset: float = 8.0
    muzzle_lateral_offset: float = 2.0
    left_miss_margin: float = 80.0
    right_miss_margin: float = 80.0
    contact_insert_pad: int = 5
    speed_ramp_up: float = 0.005
    speed_ramp_down: float = 0.1
    powerup_lifetime_ticks: int = 2000
    powerup_transition_ticks: int = 100
    bonus_ball_transition_ticks: int = 300
    powerup_previous_retention_ticks: int = 150
    powerup_visual_step: float = 0.04
    proximity_bomb_collision_pad: int = 56
    proximity_bomb_fruit_radius: float = 108.0
    reverse_powerup_ticks: int = 300
    reverse_speed: float = 1.0
    slow_powerup_ticks: int = 800
    slow_powerup_replace_threshold: int = 1000
    loss_initial_suck_count: int = 1
    loss_suction_speed_shift: int = 2

    def __post_init__(self) -> None:
        if self.tick_hz <= 0:
            raise ValueError("tick_hz must be positive")
        if self.ball_radius <= 0:
            raise ValueError("ball_radius must be positive")
        if self.projectile_speed <= 0.0:
            raise ValueError("projectile_speed must be positive")
        if self.merge_speed <= 0.0:
            raise ValueError("merge_speed must be positive")
        if self.zuma_bar_width <= 0:
            raise ValueError("zuma_bar_width must be positive")
        if self.hop_transition_ticks <= 0:
            raise ValueError("hop_transition_ticks must be positive")
        if (
            self.powerup_lifetime_ticks <= 0
            or self.powerup_transition_ticks <= 0
            or self.bonus_ball_transition_ticks <= 0
            or self.powerup_previous_retention_ticks <= 1
            or self.powerup_visual_step <= 0.0
            or self.proximity_bomb_collision_pad < 0
            or not math.isfinite(self.proximity_bomb_fruit_radius)
            or self.proximity_bomb_fruit_radius <= 0.0
            or self.reverse_powerup_ticks <= 0
            or self.reverse_speed <= 0.0
            or self.slow_powerup_ticks <= 0
            or self.slow_powerup_replace_threshold <= 0
            or self.loss_initial_suck_count <= 0
            or self.loss_suction_speed_shift < 0
        ):
            raise ValueError("power-up timing constants must be positive")


class PowerupType(IntEnum):
    """Retail Ball power-up IDs recovered from the PC executable."""

    PROXIMITY_BOMB = 0
    SLOW = 1
    ACCURACY = 2
    REVERSE = 3
    UNMAPPED_4 = 4
    UNMAPPED_5 = 5
    UNMAPPED_6 = 6
    TRI_SHOT = 7
    LIGHTNING = 8
    LASER = 9
    UNMAPPED_10 = 10
    UNMAPPED_11 = 11
    UNMAPPED_12 = 12
    BONUS_BALL = 13
    NONE = 14


_POWERUP_VISUAL_INDEX = {
    int(PowerupType.PROXIMITY_BOMB): 3,
    int(PowerupType.SLOW): 2,
    int(PowerupType.ACCURACY): 0,
    int(PowerupType.REVERSE): 4,
    int(PowerupType.TRI_SHOT): 5,
    int(PowerupType.LIGHTNING): 1,
    int(PowerupType.LASER): 6,
}


@dataclass(frozen=True, slots=True)
class PowerupSpawnCalibration:
    """Explicit PC profile values required to enable power-up generation.

    The chance denominator is runtime/profile dependent and therefore cannot
    be inferred safely from ``CURV.powerup_chance`` alone.  Generation stays
    disabled unless a calibrated profile is supplied.
    """

    chance_denominator: int
    initial_delay_ticks: int
    spawn_delay_ticks: int
    cooldown_ticks: int
    unique_color: bool
    supported_types: tuple[int, ...]
    provenance: str

    def __post_init__(self) -> None:
        if (
            self.chance_denominator < 1
            or self.initial_delay_ticks < 0
            or self.spawn_delay_ticks < 0
            or self.cooldown_ticks < 0
        ):
            raise ValueError("power-up calibration timing is invalid")
        if (
            not self.supported_types
            or len(set(self.supported_types)) != len(self.supported_types)
            or any(
                not 0 <= value < int(PowerupType.NONE)
                for value in self.supported_types
            )
        ):
            raise ValueError("power-up calibration types are invalid")
        if not self.provenance:
            raise ValueError("power-up calibration requires provenance")

    @classmethod
    def jungle2_retail_v1(cls) -> "PowerupSpawnCalibration":
        """Return the barrier-safe Jungle2 profile proven at update 9550."""

        return cls(
            chance_denominator=617,
            initial_delay_ticks=1500,
            spawn_delay_ticks=700,
            cooldown_ticks=1000,
            unique_color=True,
            supported_types=(
                int(PowerupType.PROXIMITY_BOMB),
                int(PowerupType.REVERSE),
            ),
            provenance=(
                "retail Jungle2 u9547-u9552 call trace and "
                "u9547-u9652 full-object trajectory"
            ),
        )


@dataclass(frozen=True, slots=True)
class FruitSpawnCalibration:
    """Retail Board fruit scheduler and level-specific treasure metadata.

    The PC executable uses the level ``tfreq`` value both as the idle-time
    gate and as the MTRand chance denominator.  Treasure points become eligible
    when the rear-most ball on a curve reaches that point's ``distN`` percent.
    These semantics come from the unpacked retail runtime; coordinates and
    thresholds remain sourced from the installed ``levels.xml``.
    """

    frequency_ticks: int
    lifetime_ticks: int
    points: tuple[tuple[float, float], ...]
    unlock_percentages: tuple[tuple[int, ...], ...]
    provenance: str
    fruit_type: str = "unknown"
    logical_width: int = 52
    logical_height: int = 52
    sheet_columns: int = 10
    sheet_rows: int = 6
    collection_animation_frames: int = 26
    collection_animation_fps: float = 99.0

    def __post_init__(self) -> None:
        if self.frequency_ticks < 1 or self.lifetime_ticks < 1:
            raise ValueError("fruit timing must be positive")
        if not self.points or len(self.points) != len(
            self.unlock_percentages
        ):
            raise ValueError("fruit points and unlock rows must align")
        curve_counts = {len(row) for row in self.unlock_percentages}
        if len(curve_counts) != 1 or next(iter(curve_counts), 0) < 1:
            raise ValueError("fruit unlock rows must have one width")
        if any(
            len(point) != 2
            or any(not math.isfinite(float(value)) for value in point)
            for point in self.points
        ):
            raise ValueError("fruit point coordinates must be finite pairs")
        if any(
            int(value) != value or not 0 <= int(value) <= 100
            for row in self.unlock_percentages
            for value in row
        ):
            raise ValueError("fruit unlock percentages must be 0..100")
        if any(not any(value > 0 for value in row) for row in self.unlock_percentages):
            raise ValueError("every fruit point needs an eligible curve")
        if not self.fruit_type.strip():
            raise ValueError("fruit type cannot be empty")
        if (
            isinstance(self.logical_width, bool)
            or isinstance(self.logical_height, bool)
            or int(self.logical_width) != self.logical_width
            or int(self.logical_height) != self.logical_height
            or self.logical_width < 1
            or self.logical_height < 1
            or isinstance(self.sheet_columns, bool)
            or isinstance(self.sheet_rows, bool)
            or int(self.sheet_columns) != self.sheet_columns
            or int(self.sheet_rows) != self.sheet_rows
            or self.sheet_columns < 1
            or self.sheet_rows < 1
        ):
            raise ValueError("fruit dimensions and sheet grid must be positive")
        if (
            isinstance(self.collection_animation_frames, bool)
            or int(self.collection_animation_frames)
            != self.collection_animation_frames
            or self.collection_animation_frames < 1
            or not math.isfinite(float(self.collection_animation_fps))
            or self.collection_animation_fps <= 0.0
        ):
            raise ValueError("fruit collection animation timing is invalid")
        if not self.provenance:
            raise ValueError("fruit calibration requires provenance")

    @property
    def curve_count(self) -> int:
        return len(self.unlock_percentages[0])

    def collection_ticks(self, tick_hz: int) -> int:
        """Native updates needed for the PAM clock to reach its stop frame."""

        return max(
            1,
            math.ceil(
                (self.collection_animation_frames - 1)
                * int(tick_hz)
                / float(self.collection_animation_fps)
            ),
        )


@dataclass(slots=True)
class ChainBall:
    """One active chain ball, ordered entrance/rear to skull/front."""

    id: int
    color: int
    waypoint: np.float32
    radius: int
    contact_next: bool = False
    exploding: bool = False
    explode_frame: int = 0
    should_remove: bool = False
    update_count: int = 0
    suck_count: int = 0
    suck_back: bool = True
    suck_pending: bool = False
    backwards_count: int = 0
    backwards_speed: np.float32 = field(
        default_factory=lambda: np.float32(0.0),
    )
    combo_count: int = 0
    combo_score: int = 0
    gap_bonus: int = 0
    num_gaps: int = 0
    powerup_previous_type: int = int(PowerupType.NONE)
    powerup_primary_type: int = int(PowerupType.NONE)
    powerup_secondary_type: int = int(PowerupType.NONE)
    powerup_previous_ticks: int = 0
    powerup_lifetime_ticks: int = 0
    powerup_transition_ticks: int = 0
    powerup_visual_scale: np.float32 = field(
        default_factory=lambda: np.float32(1.0),
    )
    powerup_visual_step: np.float32 = field(
        default_factory=lambda: np.float32(0.0),
    )
    powerup_visual_index: int = -1

    def __post_init__(self) -> None:
        # Ball.mWayPoint and Ball.mBackwardsSpeed are serialized game floats.
        self.waypoint = np.float32(self.waypoint)
        self.backwards_speed = np.float32(self.backwards_speed)
        self.powerup_visual_scale = np.float32(
            self.powerup_visual_scale
        )
        self.powerup_visual_step = np.float32(
            self.powerup_visual_step
        )
        powerup_types = (
            self.powerup_previous_type,
            self.powerup_primary_type,
            self.powerup_secondary_type,
        )
        if any(
            not 0 <= int(value) <= int(PowerupType.NONE)
            for value in powerup_types
        ):
            raise ValueError("power-up type is outside the retail enum")
        if min(
            self.powerup_previous_ticks,
            self.powerup_lifetime_ticks,
            self.powerup_transition_ticks,
        ) < 0:
            raise ValueError("power-up counters must be non-negative")


@dataclass(slots=True)
class Projectile:
    """A free or merging projectile."""

    id: int
    color: int
    position: Float32Array
    velocity: Float32Array
    radius: int
    just_fired: bool = True
    hit_ball_id: int | None = None
    hit_in_front: bool = False
    hit_percent: np.float32 = field(default_factory=lambda: np.float32(0.0))
    merge_speed: np.float32 = field(
        default_factory=lambda: np.float32(0.025),
    )
    hit_position: Float32Array | None = None
    target_position: Float32Array | None = None
    waypoint: np.float32 = field(default_factory=lambda: np.float32(0.0))
    have_set_prev_ball: bool = False
    do_new_merge: bool = False
    curve_point: int = 0
    # ``None`` while free; set to the owning CurveMgr slot after a hit.
    curve_index: int | None = None
    # A free bullet can cross both original curves in one frame.  The retail
    # curve-point latch is therefore tracked independently for every curve.
    curve_points: dict[int, int] = field(default_factory=dict)
    # Bullet.mGapInfo uses globally unique ball ids, so one list remains valid
    # across every curve owned by the Board.
    gap_info: list[tuple[int, int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Bullet position, velocity, merge state, and inherited waypoint are
        # all ``float`` fields in the original game.
        self.position = np.array(self.position, dtype=np.float32, copy=True)
        self.velocity = np.array(self.velocity, dtype=np.float32, copy=True)
        if self.position.shape != (2,) or self.velocity.shape != (2,):
            raise ValueError("projectile position and velocity must be 2D")
        self.hit_percent = np.float32(self.hit_percent)
        self.merge_speed = np.float32(self.merge_speed)
        self.waypoint = np.float32(self.waypoint)
        if self.hit_position is not None:
            self.hit_position = np.array(
                self.hit_position,
                dtype=np.float32,
                copy=True,
            )
        if self.target_position is not None:
            self.target_position = np.array(
                self.target_position,
                dtype=np.float32,
                copy=True,
            )
        if self.curve_index is not None and self.curve_index < 0:
            raise ValueError("projectile curve_index cannot be negative")
        self.curve_points = {
            int(index): int(waypoint)
            for index, waypoint in self.curve_points.items()
        }
        if any(index < 0 for index in self.curve_points):
            raise ValueError("projectile curve-point index cannot be negative")

    @property
    def merging(self) -> bool:
        return self.hit_ball_id is not None


class GunState(str, Enum):
    NORMAL = "normal"
    FIRING = "firing"
    RELOADING = "reloading"


VICTORY_TRANSITION_AWARD_POINTS = 100


@dataclass(slots=True)
class TickEvents:
    """Small, allocation-friendly event summary for rewards and replays."""

    fired: int = 0
    hits: int = 0
    inserted: int = 0
    matches: int = 0
    balls_exploded: int = 0
    balls_removed: int = 0
    score_delta: int = 0
    powerups_spawned: int = 0
    powerups_triggered: int = 0
    fruit_chance_draws: int = 0
    fruits_spawned: int = 0
    fruits_expired: int = 0
    fruits_collected: int = 0
    zuma_triggered: bool = False
    win_pending: bool = False
    loss_started: bool = False
    outcome: str | None = None


@dataclass(slots=True)
class CurveRuntimeState:
    """Mutable state owned by one retail ``CurveMgr`` curve slot.

    The original Board owns the shooter, score, native clock, and RNG streams,
    while every curve owns its chain, pending queue, movement timers, and
    power-up runtime.  Keeping that split explicit prevents a two-curve level
    from degenerating into two independent games.
    """

    curve: WaypointCurve
    curve_points: Float32Array | None = None
    curve_tunnels: NDArray[np.bool_] | None = None
    entrance_cutoff: int = 0
    balls: list[ChainBall] = field(default_factory=list)
    pending_colors: list[int] = field(default_factory=list)
    merging_projectiles: list[Projectile] = field(default_factory=list)
    stop_adding: bool = False
    feed_plan_exhausted: bool = False
    feed_refill_armed: bool = False
    num_balls_created: int = 0
    advance_speed: np.float32 = field(
        default_factory=lambda: np.float32(0.0),
    )
    current_acceleration: np.float32 = field(
        default_factory=lambda: np.float32(0.0),
    )
    slow_count: int = 0
    backward_count: int = 0
    stop_time: int = 0
    first_ball_moved_backwards: bool = False
    has_reached_rollout: bool = False
    has_reached_cruising_speed: bool = False
    first_chain_end: int = 0
    consecutive_clears: int = 0
    have_sets: bool = False
    powerup_last_any_spawn_time: int = 0
    powerup_last_spawn_times: list[int] = field(
        default_factory=lambda: [-1000] * 14,
    )
    powerup_cooldown_times: list[int] = field(
        default_factory=lambda: [-1000] * 14,
    )
    powerup_spawn_counts: list[int] = field(
        default_factory=lambda: [0] * 14,
    )
    powerup_field_124_by_type: list[int] = field(
        default_factory=lambda: [0] * 14,
    )
    active_powerup_color_counts: list[int] = field(
        default_factory=lambda: [0] * 6,
    )
    powerup_triggered: bool = False
    last_powerup_waypoint: int = 0


class PopCapMTRandom:
    """Exact retail MTRand stream, including its signed-positive output."""

    __slots__ = ("_words", "_index")

    _STATE_WORDS = 624
    _PERIOD_OFFSET = 397
    _MATRIX_A = 0x9908B0DF

    def __init__(self, seed: int):
        self.seed(seed)

    @property
    def index(self) -> int:
        return self._index

    @property
    def words(self) -> tuple[int, ...]:
        return tuple(self._words)

    @property
    def state(self) -> tuple[tuple[int, ...], int]:
        return self.words, self.index

    def seed(self, value: int) -> None:
        if isinstance(value, bool) or not isinstance(
            value,
            (int, np.integer),
        ):
            raise TypeError("MTRand seed must be an integer")
        seed = int(value) & 0xFFFFFFFF
        if seed == 0:
            seed = 4357
        words = [seed]
        for index in range(1, self._STATE_WORDS):
            previous = words[-1]
            words.append(
                (
                    1812433253
                    * (previous ^ (previous >> 30))
                    + index
                )
                & 0xFFFFFFFF
            )
        self._words = words
        self._index = self._STATE_WORDS

    def load_state(
        self,
        words: Sequence[int],
        index: int,
    ) -> None:
        if len(words) != self._STATE_WORDS:
            raise ValueError("MTRand state must contain 624 words")
        if (
            isinstance(index, bool)
            or not isinstance(index, (int, np.integer))
            or not 0 <= int(index) <= self._STATE_WORDS
        ):
            raise ValueError("MTRand index is invalid")
        decoded: list[int] = []
        for value in words:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or not 0 <= int(value) <= 0xFFFFFFFF
            ):
                raise ValueError("MTRand state word is not a uint32")
            decoded.append(int(value))
        self._words = decoded
        self._index = int(index)

    def _twist(self) -> None:
        words = self._words
        for index in range(self._STATE_WORDS):
            combined = (
                (words[index] & 0x80000000)
                | (words[(index + 1) % self._STATE_WORDS] & 0x7FFFFFFF)
            )
            words[index] = (
                words[(index + self._PERIOD_OFFSET) % self._STATE_WORDS]
                ^ (combined >> 1)
                ^ (self._MATRIX_A if combined & 1 else 0)
            ) & 0xFFFFFFFF
        self._index = 0

    def next_u31(self) -> int:
        if self._index >= self._STATE_WORDS:
            self._twist()
        value = self._words[self._index]
        self._index += 1
        value ^= value >> 11
        value ^= (value << 7) & 0x9D2C5680
        value ^= (value << 15) & 0xEFC60000
        value ^= value >> 18
        return value & 0x7FFFFFFF

    def rand_mod(self, modulus: int) -> int:
        if modulus <= 0:
            raise ValueError("modulus must be positive")
        return self.next_u31() % modulus

    def random(self) -> float:
        """Advance once and return a convenient half-open unit interval."""

        return self.next_u31() / float(1 << 31)


class MsvcCRTRandom:
    """Exact 32-bit state transition used by the retail MSVC ``rand``."""

    __slots__ = ("_state",)

    def __init__(self, seed: int):
        self.seed(seed)

    @property
    def state(self) -> int:
        return self._state

    @state.setter
    def state(self, value: int) -> None:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or not 0 <= int(value) <= 0xFFFFFFFF
        ):
            raise ValueError("CRT random state must be a uint32")
        self._state = int(value)

    def seed(self, value: int) -> None:
        if isinstance(value, bool) or not isinstance(
            value,
            (int, np.integer),
        ):
            raise TypeError("CRT random seed must be an integer")
        self._state = int(value) & 0xFFFFFFFF

    def next_u15(self) -> int:
        self._state = (
            self._state * 0x343FD + 0x269EC3
        ) & 0xFFFFFFFF
        return (self._state >> 16) & 0x7FFF

    def random_ratio(self) -> float:
        return self.next_u15() / 32767.0


class _BalancedColorChooser:
    """Retail QRand anti-drought weighting over the six colour slots."""

    _MINIMUM_MULTIPLIER = 0.3330000042915344
    _MAXIMUM_MULTIPLIER = 3.0

    def __init__(self, rng: MsvcCRTRandom, size: int = 6):
        if size < 1:
            raise ValueError("QRand size must be positive")
        self.rng = rng
        self.size = size
        self.update_count = 0
        self.selected_index = -1
        self.weights = np.zeros(size, dtype=np.float32)
        self.sways = np.zeros(size, dtype=np.float32)
        self.last_hit = np.zeros(size, dtype=np.int64)
        self.previous_hit = np.zeros(size, dtype=np.int64)
        self.allowed_support: tuple[int, ...] | None = None

    def choose(self, present: Sequence[int]) -> int | None:
        allowed = sorted(set(int(value) for value in present))
        if not allowed:
            return None
        if allowed[0] < 0 or allowed[-1] >= self.size:
            raise ValueError("colour is outside the QRand palette")
        support = tuple(allowed)
        if support != self.allowed_support:
            # Board.GetGoodBallColor calls QRand.Clear/SetWeights whenever a
            # colour enters or leaves the board.  QRand keeps mUpdateCnt but
            # clears both hit-history arrays.
            self.last_hit.fill(0)
            self.previous_hit.fill(0)
            self.weights.fill(np.float32(0.0))
            self.sways.fill(np.float32(0.0))
            weight = np.float32(1.0 / len(allowed))
            self.weights[list(support)] = weight
            self.allowed_support = support
        self.update_count += 1
        total = np.float32(0.0)
        for color in range(self.size):
            weight = np.float32(self.weights[color])
            if weight == np.float32(0.0):
                self.sways[color] = np.float32(0.0)
                continue

            # The retail x87 code writes these intermediates back to float32
            # at the same points.  Keeping those stores explicit makes the
            # captured QRand vectors byte-identical.
            expected = np.float32(1.0 / float(weight))
            recent = np.float32(
                1.0
                + (
                    (
                        self.update_count
                        - int(self.last_hit[color])
                    )
                    - float(expected)
                )
                / float(expected)
            )
            twice_expected = float(expected) * 2.0
            previous = np.float32(
                1.0
                + (
                    (
                        self.update_count
                        - int(self.previous_hit[color])
                    )
                    - twice_expected
                )
                / twice_expected
            )
            multiplier = (
                float(recent) * 0.75 + float(previous) * 0.25
            )
            multiplier = min(
                max(multiplier, self._MINIMUM_MULTIPLIER),
                self._MAXIMUM_MULTIPLIER,
            )
            sway = np.float32(float(weight) * multiplier)
            self.sways[color] = sway
            total = np.float32(float(total) + float(sway))

        needle = np.float32(self.rng.random_ratio() * float(total))
        # Zero-weight colour slots are not members of the live-board support.
        # Iterating all storage slots with ``needle <= sway`` can nevertheless
        # select one of them when the CRT draw (or a float32 subtraction) is
        # exactly zero.  Use the same ordered weighted scan over the actual
        # support and retain its final member as the round-off fallback.
        selected = support[-1]
        for color in support:
            sway = self.sways[color]
            if needle <= sway:
                selected = color
                break
            needle = np.float32(float(needle) - float(sway))
        self.previous_hit[selected] = self.last_hit[selected]
        self.last_hit[selected] = self.update_count
        self.selected_index = selected
        return selected

    def load_state(
        self,
        *,
        update_count: int,
        selected_index: int,
        weights: Sequence[float],
        sways: Sequence[float],
        last_hit: Sequence[int],
        previous_hit: Sequence[int],
    ) -> None:
        arrays = (weights, sways, last_hit, previous_hit)
        lengths = tuple(len(values) for values in arrays)
        if all(length == 0 for length in lengths):
            # A newly constructed retail QRand owns four empty std::vectors
            # until Board.GetGoodBallColor first calls Clear/SetWeights.  The
            # simulator keeps fixed-size NumPy storage, so restore that exact
            # native sentinel as its factory-equivalent zeroed representation.
            if update_count != 0 or selected_index != -1:
                raise ValueError("QRand uninitialized state is inconsistent")
            self.update_count = 0
            self.selected_index = -1
            self.weights.fill(np.float32(0.0))
            self.sways.fill(np.float32(0.0))
            self.last_hit.fill(0)
            self.previous_hit.fill(0)
            self.allowed_support = None
            return
        if any(len(values) != self.size for values in arrays):
            raise ValueError("QRand state vectors have the wrong length")
        if update_count < 0:
            raise ValueError("QRand update_count must be non-negative")
        if not -1 <= selected_index < self.size:
            raise ValueError("QRand selected_index is outside the palette")
        weight_values = np.asarray(weights, dtype=np.float32)
        sway_values = np.asarray(sways, dtype=np.float32)
        if (
            np.any(~np.isfinite(weight_values))
            or np.any(~np.isfinite(sway_values))
            or np.any(weight_values < np.float32(0.0))
            or np.any(sway_values < np.float32(0.0))
        ):
            raise ValueError("QRand float vectors are invalid")
        last_values = np.asarray(last_hit, dtype=np.int64)
        previous_values = np.asarray(previous_hit, dtype=np.int64)
        if (
            np.any(last_values < 0)
            or np.any(previous_values < 0)
            or np.any(last_values > update_count)
            or np.any(previous_values > update_count)
        ):
            raise ValueError("QRand hit histories are invalid")
        support = tuple(
            int(index)
            for index in np.flatnonzero(
                weight_values > np.float32(0.0)
            )
        )
        if not support:
            raise ValueError("QRand weights have empty support")
        self.update_count = int(update_count)
        self.selected_index = int(selected_index)
        self.weights[:] = weight_values
        self.sways[:] = sway_values
        self.last_hit[:] = last_values
        self.previous_hit[:] = previous_values
        self.allowed_support = support


class RevengeSimulator:
    """Shared-Board, 100 Hz Zuma's Revenge mechanics simulator.

    For compatibility, ``curve``, ``balls``, ``pending_colors``, and the
    movement timers expose curve slot zero whenever no internal curve update
    is active.  ``curve_states`` exposes every slot for multi-curve tooling.
    """

    _CURVE_RUNTIME_BINDINGS = (
        ("curve", "curve"),
        ("_curve_points", "curve_points"),
        ("_curve_tunnels", "curve_tunnels"),
        ("_entrance_cutoff", "entrance_cutoff"),
        ("balls", "balls"),
        ("pending_colors", "pending_colors"),
        ("merging_projectiles", "merging_projectiles"),
        ("stop_adding", "stop_adding"),
        ("feed_plan_exhausted", "feed_plan_exhausted"),
        ("feed_refill_armed", "feed_refill_armed"),
        ("num_balls_created", "num_balls_created"),
        ("advance_speed", "advance_speed"),
        ("current_acceleration", "current_acceleration"),
        ("slow_count", "slow_count"),
        ("backward_count", "backward_count"),
        ("stop_time", "stop_time"),
        ("first_ball_moved_backwards", "first_ball_moved_backwards"),
        ("has_reached_rollout", "has_reached_rollout"),
        ("has_reached_cruising_speed", "has_reached_cruising_speed"),
        ("first_chain_end", "first_chain_end"),
        ("consecutive_clears", "consecutive_clears"),
        ("_have_sets", "have_sets"),
        ("powerup_last_any_spawn_time", "powerup_last_any_spawn_time"),
        ("powerup_last_spawn_times", "powerup_last_spawn_times"),
        ("powerup_cooldown_times", "powerup_cooldown_times"),
        ("powerup_spawn_counts", "powerup_spawn_counts"),
        ("powerup_field_124_by_type", "powerup_field_124_by_type"),
        ("active_powerup_color_counts", "active_powerup_color_counts"),
        ("powerup_triggered", "powerup_triggered"),
        ("last_powerup_waypoint", "last_powerup_waypoint"),
    )

    def __init__(
        self,
        curve: WaypointCurve,
        shooter: tuple[float, float] = (400.0, 300.0),
        *,
        curves: Sequence[WaypointCurve] | None = None,
        shooter_positions: Sequence[tuple[float, float]] | None = None,
        initial_shooter_index: int = 0,
        seed: int | None = None,
        config: RevengePhysicsConfig | None = None,
        powerup_calibration: PowerupSpawnCalibration | None = None,
        fruit_calibration: FruitSpawnCalibration | None = None,
        profile_mode: str = SUPPORTED_PROFILE_MODE,
    ):
        if profile_mode != SUPPORTED_PROFILE_MODE:
            raise NotImplementedError(
                "fresh-profile tutorial scripts are not implemented; "
                f"profile_mode must be {SUPPORTED_PROFILE_MODE!r}"
            )
        self.profile_mode = profile_mode
        curve_sequence = (curve,) if curves is None else tuple(curves)
        if not curve_sequence:
            raise ValueError("at least one curve is required")
        if any(candidate is None for candidate in curve_sequence):
            raise ValueError("curves cannot contain None")
        self.curves = curve_sequence
        raw_shooter_positions = (
            (shooter,) if shooter_positions is None else tuple(shooter_positions)
        )
        if not raw_shooter_positions:
            raise ValueError("at least one shooter position is required")
        decoded_shooter_positions: list[Float32Array] = []
        for position in raw_shooter_positions:
            decoded = np.asarray(position, dtype=np.float32)
            if decoded.shape != (2,):
                raise ValueError("every shooter position must be an (x, y) pair")
            if np.any(~np.isfinite(decoded)):
                raise ValueError("shooter coordinates must be finite")
            decoded_shooter_positions.append(decoded)
        if not 0 <= int(initial_shooter_index) < len(decoded_shooter_positions):
            raise IndexError("initial_shooter_index is outside shooter_positions")
        self.shooter_positions = tuple(decoded_shooter_positions)
        self.initial_shooter_index = int(initial_shooter_index)
        self.active_shooter_index = self.initial_shooter_index
        self.config = config or RevengePhysicsConfig()
        self.powerup_calibration = powerup_calibration
        self.fruit_calibration = fruit_calibration
        if (
            self.fruit_calibration is not None
            and self.fruit_calibration.curve_count != len(self.curves)
        ):
            raise ValueError(
                "fruit calibration curve count does not match the level"
            )
        self._initial_mtrand_seed = (
            int(
                np.random.SeedSequence().generate_state(
                    1,
                    dtype=np.uint32,
                )[0]
            )
            if seed is None
            else int(seed) & 0xFFFFFFFF
        )
        self.rng = PopCapMTRandom(self._initial_mtrand_seed)
        self._initial_seed = seed
        self._initial_crt_seed = (
            int(
                np.random.SeedSequence().generate_state(
                    1,
                    dtype=np.uint32,
                )[0]
            )
            if seed is None
            else int(seed) & 0xFFFFFFFF
        )
        self.crt_rng = MsvcCRTRandom(self._initial_crt_seed)
        self._next_id = 1
        self._curve_states = [
            self._make_curve_runtime(candidate)
            for candidate in self.curves
        ]
        self._active_curve_index = -1
        self._activate_curve(0)
        for index in range(len(self._curve_states)):
            self._activate_curve(index)
            self._entrance_cutoff = self._calculate_entrance_cutoff()
        self._activate_curve(0)
        self._color_chooser = _BalancedColorChooser(
            self.crt_rng,
            max(6, self.num_colors),
        )

        self.balls: list[ChainBall] = []
        self.pending_colors: list[int] = []
        self.free_projectiles: list[Projectile] = []
        self.merging_projectiles: list[Projectile] = []

        self.tick_count = 0
        self.score = 0
        self.score_at_level_start = 0
        self.current_bar_size = 0
        self.target_bar_size = 0
        self.zuma_reached = False
        self.stop_adding = False
        self.feed_plan_exhausted = False
        self.feed_refill_armed = False
        self.num_balls_created = 0
        self.advance_speed = np.float32(0.0)
        self.current_acceleration = np.float32(0.0)
        self.slow_count = 0
        self.backward_count = 0
        self.stop_time = 0
        self.first_ball_moved_backwards = False
        self.has_reached_rollout = False
        self.has_reached_cruising_speed = False
        self.first_chain_end = 0
        self.consecutive_clears = 0
        self._have_sets = False
        self.outcome: str | None = None
        self.win_pending = False
        self.skull_entry_pending = False
        self.loss_started = False
        self.loss_elapsed_ticks = 0
        self.loss_curve_index: int | None = None
        self.native_game_time = 0
        self.powerup_last_any_spawn_time = 0
        self.powerup_last_spawn_times = [-1000] * 14
        self.powerup_cooldown_times = [-1000] * 14
        self.powerup_spawn_counts = [0] * 14
        self.powerup_field_124_by_type = [0] * 14
        self.active_powerup_color_counts = [0] * 6
        self.powerup_triggered = False
        self.last_powerup_waypoint = 0
        self.fruit_active_point_index: int | None = None
        self.fruit_expiry_time = 0
        self.fruit_collecting = False
        self.fruit_spawn_count = 0
        self.fruit_collect_count = 0
        self.fruit_velocity = np.float32(0.25)
        self.fruit_max_velocity = np.float32(0.25)
        self.fruit_acceleration = np.float32(-0.01)
        self.fruit_vertical_offset = np.float32(0.0)
        self.fruit_lower_bound = np.float32(np.finfo(np.float32).max)
        self.fruit_upper_bound = np.float32(np.finfo(np.float32).max)
        self.fruit_glow_alpha = 0
        self.fruit_glow_step = 0
        self.fruit_alpha = 255
        self.fruit_cell_index = 0
        self.fruit_collection_ticks_remaining = 0

        self.gun_state = GunState.NORMAL
        self.gun_state_percent = np.float32(1.0)
        self.hop_ticks_remaining = 0
        self.hop_target_index: int | None = None
        self.aim_angle = np.float32(0.0)
        self.current_color: int | None = None
        self.next_color: int | None = None
        self.last_events = TickEvents()
        self.reset(seed=seed)

    @staticmethod
    def _make_curve_runtime(curve: WaypointCurve) -> CurveRuntimeState:
        points_cache: Float32Array | None = None
        raw_points = getattr(curve, "points", None)
        if raw_points is not None:
            points = np.array(raw_points, dtype=np.float32, copy=True)
            if points.shape == (int(curve.end_waypoint) + 1, 2):
                points_cache = points

        tunnel_cache: NDArray[np.bool_] | None = None
        raw_tunnels = getattr(curve, "in_tunnel", None)
        if raw_tunnels is not None:
            tunnels = np.asarray(raw_tunnels, dtype=np.bool_)
            if tunnels.shape == (int(curve.end_waypoint) + 1,):
                tunnel_cache = np.array(tunnels, dtype=np.bool_, copy=True)
        return CurveRuntimeState(
            curve=curve,
            curve_points=points_cache,
            curve_tunnels=tunnel_cache,
        )

    def _store_active_curve(self) -> None:
        if self._active_curve_index < 0:
            return
        state = self._curve_states[self._active_curve_index]
        for simulator_name, state_name in self._CURVE_RUNTIME_BINDINGS:
            setattr(state, state_name, getattr(self, simulator_name))

    def _activate_curve(self, curve_index: int) -> None:
        index = int(curve_index)
        if not 0 <= index < len(self._curve_states):
            raise IndexError("curve_index is outside this level")
        self._store_active_curve()
        self._active_curve_index = index
        state = self._curve_states[index]
        for simulator_name, state_name in self._CURVE_RUNTIME_BINDINGS:
            setattr(self, simulator_name, getattr(state, state_name))

    @property
    def active_curve_index(self) -> int:
        return self._active_curve_index

    @property
    def curve_count(self) -> int:
        return len(self._curve_states)

    @property
    def curve_states(self) -> tuple[CurveRuntimeState, ...]:
        self._store_active_curve()
        return tuple(self._curve_states)

    def iter_curve_balls(self) -> tuple[tuple[int, ChainBall], ...]:
        self._store_active_curve()
        return tuple(
            (curve_index, ball)
            for curve_index, state in enumerate(self._curve_states)
            for ball in state.balls
        )

    def iter_merging_projectiles(
        self,
    ) -> tuple[tuple[int, Projectile], ...]:
        self._store_active_curve()
        return tuple(
            (curve_index, projectile)
            for curve_index, state in enumerate(self._curve_states)
            for projectile in state.merging_projectiles
        )

    @property
    def total_ball_count(self) -> int:
        self._store_active_curve()
        return sum(len(state.balls) for state in self._curve_states)

    @property
    def total_merging_projectile_count(self) -> int:
        self._store_active_curve()
        return sum(
            len(state.merging_projectiles)
            for state in self._curve_states
        )

    @property
    def all_curves_stop_adding(self) -> bool:
        self._store_active_curve()
        return all(state.stop_adding for state in self._curve_states)

    @property
    def parameters(self) -> Any:
        return self.curve.parameters

    @property
    def num_colors(self) -> int:
        values = tuple(
            int(getattr(candidate.parameters, "colors"))
            for candidate in self.curves
        )
        if any(value < 1 for value in values):
            raise ValueError("curve must define at least one colour")
        return max(values)

    @property
    def active_num_colors(self) -> int:
        value = int(getattr(self.parameters, "colors"))
        if value < 1:
            raise ValueError("curve must define at least one colour")
        return value

    @property
    def score_target(self) -> int:
        return self.score_at_level_start + sum(
            int(getattr(candidate.parameters, "zuma_score", 0))
            for candidate in self.curves
        )

    @property
    def danger_point(self) -> int:
        return min(
            int(self.curve.end_waypoint),
            int(self.curve.end_waypoint) + 1 - self.config.danger_distance,
        )

    @property
    def entrance_cutoff(self) -> int:
        """Hidden entrance cutoff derived by ``WayPointMgr.LoadCurve``."""

        return self._entrance_cutoff

    def _calculate_entrance_cutoff(self) -> int:
        # WayPointMgr starts this value at 15 and only replaces it when the
        # *first* waypoint belongs to an initial, contiguous tunnel run.
        # Consequently a curve without an entrance tunnel uses 15-radius,
        # not 0-radius.
        first_visible = 15
        if self.in_tunnel_at_waypoint(0):
            first_visible = 0
            while first_visible <= int(self.curve.end_waypoint):
                if not self.in_tunnel_at_waypoint(first_visible):
                    break
                first_visible += 1
        return first_visible - self.config.ball_radius

    @property
    def score_achieved(self) -> bool:
        return self.score_target > 0 and self.score >= self.score_target

    @classmethod
    def from_installed(
        cls,
        level_id: str = "Jungle1",
        *,
        root: str | Path | None = None,
        catalog: Any | None = None,
        hard: bool = False,
        curve_index: int = 0,
        gun_index: int = 0,
        allow_partial_level: bool = False,
        seed: int | None = None,
        config: RevengePhysicsConfig | None = None,
        powerup_calibration: PowerupSpawnCalibration | None = None,
        fruit_calibration: FruitSpawnCalibration | None = None,
        profile_mode: str = SUPPORTED_PROFILE_MODE,
    ) -> "RevengeSimulator":
        """Build from the user's read-only original installation.

        Ordinary one- and two-curve levels share one Board, shooter, score,
        clock, and RNG stream.  Unsupported gun, boss, and game-mode
        structures still fail closed so a partial simulation cannot be
        mistaken for a faithful level.  The ``allow_partial_level`` escape
        hatch is only for mechanics diagnostics.
        """

        from zuma_rl.original_data import OriginalGameCatalog

        if profile_mode != SUPPORTED_PROFILE_MODE:
            raise NotImplementedError(
                "fresh-profile tutorial scripts are not fidelity-safe yet; "
                f"use profile_mode={SUPPORTED_PROFILE_MODE!r} with a PC "
                "profile whose hints have already been seen"
            )
        if catalog is None:
            catalog = OriginalGameCatalog(root)
        loaded = catalog.load_level(level_id, hard=hard)
        if not 0 <= curve_index < len(loaded.curves):
            raise IndexError("curve_index is outside this level")
        gun_positions = loaded.definition.gun.positions
        if not gun_positions:
            raise ValueError(f"level {level_id!r} has no decoded gun position")
        if not 0 <= gun_index < len(gun_positions):
            raise IndexError("gun_index is outside this level")

        unsupported: list[str] = []
        if len(gun_positions) > 2:
            unsupported.append(f"{len(gun_positions)} frog positions")
        gun_type = loaded.definition.gun.type.casefold()
        if gun_type != "normal":
            unsupported.append(f"{gun_type!r} moving-frog type")
        level_key = loaded.definition.id.casefold()
        if level_key.startswith(("boss", "debugboss")):
            unsupported.append("boss state machine")
        if loaded.definition.attributes.get("ironfrog", "false").casefold() == "true":
            unsupported.append("Iron Frog mode state machine")
        if unsupported and not allow_partial_level:
            details = ", ".join(unsupported)
            raise NotImplementedError(
                f"level {loaded.definition.id!r} is not fidelity-safe yet "
                f"({details}); pass allow_partial_level=True only for "
                "explicit, non-transferable mechanics diagnostics"
            )
        selected_curves: Sequence[WaypointCurve]
        if len(loaded.curves) > 1 and curve_index != 0:
            if not allow_partial_level:
                raise ValueError(
                    "curve_index cannot select one curve from a shared "
                    "multi-curve Board; use curve_index=0 to simulate all "
                    "curves, or allow_partial_level=True only for an explicit "
                    "non-transferable diagnostic"
                )
            selected_curves = (loaded.curves[curve_index],)
        elif unsupported:
            selected_curves = (loaded.curves[curve_index],)
        else:
            selected_curves = loaded.curves
        selected_curve_indices = (
            (curve_index,)
            if len(selected_curves) == 1 and len(loaded.curves) > 1
            else tuple(range(len(selected_curves)))
        )
        if fruit_calibration is None:
            treasure_points = tuple(
                getattr(loaded.definition, "treasure_points", ())
            )
            raw_distances = tuple(
                getattr(
                    loaded.definition,
                    "treasure_point_distances",
                    (),
                )
            )
            frequency_text = loaded.definition.attributes.get("tfreq")
            if treasure_points and raw_distances and frequency_text is not None:
                if len(treasure_points) != len(raw_distances):
                    raise ValueError(
                        "installed treasure points and distance rows disagree"
                    )
                try:
                    frequency_ticks = int(frequency_text)
                    selected_distances = tuple(
                        tuple(int(row[index]) for index in selected_curve_indices)
                        for row in raw_distances
                    )
                except (IndexError, TypeError, ValueError) as error:
                    raise ValueError(
                        "installed fruit scheduler metadata is invalid"
                    ) from error
                # A partial multi-curve diagnostic can omit every curve named
                # by a treasure point.  Such a point can never be selected in
                # that deliberately incomplete Board, so leave fruit disabled.
                if all(any(value > 0 for value in row) for row in selected_distances):
                    fruit_type = getattr(loaded.definition, "fruit_type", "")
                    if not fruit_type:
                        raise ValueError(
                            "installed level has treasure points but no Zone fruit"
                        )
                    assets = catalog.fruit_assets(fruit_type)
                    fruit_calibration = FruitSpawnCalibration(
                        frequency_ticks=frequency_ticks,
                        lifetime_ticks=1000,
                        points=tuple(
                            (float(point[0]), float(point[1]))
                            for point in treasure_points
                        ),
                        unlock_percentages=selected_distances,
                        fruit_type=assets.fruit_type,
                        logical_width=assets.logical_width,
                        logical_height=assets.logical_height,
                        sheet_columns=assets.sheet_columns,
                        sheet_rows=assets.sheet_rows,
                        collection_animation_frames=(
                            assets.collection_animation_frames
                        ),
                        collection_animation_fps=(
                            assets.collection_animation_fps
                        ),
                        provenance=(
                            "installed levels.xml tfreq/TreasurePoint metadata; "
                            f"{assets.provenance}; "
                            "retail runtime sha256:2181ce2bfbfcb4678bf69a1474e08d3d"
                            "b941311aa768176a88453cc69692af20 static control flow "
                            "0x00417A00-0x00417FCF, 0x0041C120-0x0041C527, "
                            "and 0x004B5A50-0x004B5CD9"
                        ),
                    )
        simulated_shooter_positions = (
            (gun_positions[gun_index],) if unsupported else gun_positions
        )
        return cls(
            selected_curves[0],
            shooter=gun_positions[gun_index],
            curves=selected_curves,
            shooter_positions=simulated_shooter_positions,
            initial_shooter_index=0 if unsupported else gun_index,
            seed=seed,
            config=config,
            powerup_calibration=powerup_calibration,
            fruit_calibration=fruit_calibration,
            profile_mode=profile_mode,
        )

    def _new_id(self) -> int:
        value = self._next_id
        self._next_id += 1
        return value

    def _reset_powerup_runtime(self) -> None:
        self.powerup_last_any_spawn_time = 0
        self.powerup_last_spawn_times = [-1000] * 14
        self.powerup_cooldown_times = [-1000] * 14
        self.powerup_spawn_counts = [0] * 14
        self.powerup_field_124_by_type = [0] * 14
        self.active_powerup_color_counts = [0] * 6
        self.powerup_triggered = False
        self.last_powerup_waypoint = 0

    def _reset_fruit_runtime(self) -> None:
        self.fruit_active_point_index = None
        self.fruit_expiry_time = 0
        self.fruit_collecting = False
        self.fruit_spawn_count = 0
        self.fruit_collect_count = 0
        self.fruit_velocity = np.float32(0.25)
        self.fruit_max_velocity = np.float32(0.25)
        self.fruit_acceleration = np.float32(-0.01)
        self.fruit_vertical_offset = np.float32(0.0)
        self.fruit_lower_bound = np.float32(np.finfo(np.float32).max)
        self.fruit_upper_bound = np.float32(np.finfo(np.float32).max)
        self.fruit_glow_alpha = 0
        self.fruit_glow_step = 0
        self.fruit_alpha = 255
        self.fruit_cell_index = 0
        self.fruit_collection_ticks_remaining = 0

    def _rand_mod(self, modulus: int) -> int:
        return self.rng.rand_mod(modulus)

    def _reset_active_curve_runtime(self) -> None:
        self.balls.clear()
        self.pending_colors.clear()
        self.merging_projectiles.clear()
        self.stop_adding = False
        self.feed_plan_exhausted = False
        self.feed_refill_armed = False
        self.num_balls_created = 0
        self.advance_speed = np.float32(0.0)
        self.current_acceleration = np.float32(0.0)
        self.slow_count = 0
        self.backward_count = 0
        self.stop_time = 0
        self.first_ball_moved_backwards = False
        self.has_reached_rollout = False
        self.has_reached_cruising_speed = False
        self.first_chain_end = 0
        self.consecutive_clears = 0
        self._have_sets = False
        self._reset_powerup_runtime()

        configured_total = int(getattr(self.parameters, "num_balls", 0))
        pending_count = self.config.initial_pending_balls
        if configured_total > 0:
            pending_count = min(configured_total, pending_count)
        for _ in range(pending_count):
            self._append_pending_color()
        if configured_total > 0 and self.num_balls_created >= configured_total:
            self.stop_adding = True
        self.advance_speed = self._roll_in_speed()

    def reset(self, *, seed: int | None = None) -> None:
        if seed is not None:
            self._initial_mtrand_seed = int(seed) & 0xFFFFFFFF
            self.rng.seed(self._initial_mtrand_seed)
            self._initial_seed = seed
            self._initial_crt_seed = int(seed) & 0xFFFFFFFF
            self.crt_rng.seed(self._initial_crt_seed)
        self._color_chooser = _BalancedColorChooser(
            self.crt_rng,
            max(6, self.num_colors),
        )
        self._next_id = 1
        self.free_projectiles.clear()
        self.tick_count = 0
        self.score = 0
        self.score_at_level_start = 0
        self.current_bar_size = 0
        self.target_bar_size = 0
        self.zuma_reached = False
        self.outcome = None
        self.win_pending = False
        self.skull_entry_pending = False
        self.loss_started = False
        self.loss_elapsed_ticks = 0
        self.loss_curve_index = None
        self.native_game_time = 0
        self._reset_fruit_runtime()
        self.gun_state = GunState.NORMAL
        self.gun_state_percent = np.float32(1.0)
        self.hop_ticks_remaining = 0
        self.hop_target_index = None
        self.aim_angle = np.float32(0.0)
        self.current_color = None
        self.next_color = None
        self.last_events = TickEvents()
        self.active_shooter_index = self.initial_shooter_index

        for index in range(self.curve_count):
            self._activate_curve(index)
            self._reset_active_curve_runtime()
        self._activate_curve(0)

    @property
    def shooter(self) -> Float32Array:
        """Current fixed lily-pad position used as the projectile origin."""

        return self.shooter_positions[self.active_shooter_index]

    @property
    def shooter_position_count(self) -> int:
        return len(self.shooter_positions)

    @property
    def hop_in_progress(self) -> bool:
        return self.hop_ticks_remaining > 0

    @property
    def hop_progress(self) -> np.float32:
        """Normalised progress of the input-locked fixed-pad transition."""

        if not self.hop_in_progress:
            return np.float32(1.0)
        completed = self.config.hop_transition_ticks - self.hop_ticks_remaining
        return np.float32(completed / self.config.hop_transition_ticks)

    @property
    def can_hop_shooter(self) -> bool:
        """Whether a retail-style request to use the other pad is accepted."""

        return (
            self.shooter_position_count > 1
            and not self.win_pending
            and not self.loss_started
            and self.outcome is None
            and self.gun_state is GunState.NORMAL
            and not self.hop_in_progress
        )

    def load_state(
        self,
        colors: Sequence[int],
        waypoints: Sequence[float],
        contacts: Sequence[bool] | None = None,
        *,
        pending_colors: Sequence[int] = (),
        current_color: int | None = None,
        next_color: int | None = None,
        score: int = 0,
        score_at_level_start: int = 0,
    ) -> None:
        """Load a deterministic state for replay fixtures and curricula."""

        if len(colors) != len(waypoints):
            raise ValueError("colors and waypoints must have equal length")
        if contacts is not None and len(contacts) not in {
            max(0, len(colors) - 1),
            len(colors),
        }:
            raise ValueError("contacts must have n-1 or n values")
        values = [int(value) for value in colors]
        if any(value < 0 or value >= self.active_num_colors for value in values):
            raise ValueError("ball colour is outside the curve palette")
        positions = [np.float32(value) for value in waypoints]
        if any(not bool(np.isfinite(value)) for value in positions):
            raise ValueError("waypoints must be finite")

        self.rng.seed(self._initial_mtrand_seed)
        self.crt_rng.seed(self._initial_crt_seed)
        self._color_chooser = _BalancedColorChooser(
            self.crt_rng,
            max(6, self.num_colors),
        )
        self._next_id = 1
        self.balls = [
            ChainBall(
                id=self._new_id(),
                color=color,
                waypoint=waypoint,
                radius=self.config.ball_radius,
            )
            for color, waypoint in zip(values, positions, strict=True)
        ]
        if contacts is None:
            for ball in self.balls[:-1]:
                ball.contact_next = True
        else:
            for index, value in enumerate(contacts[: max(0, len(values) - 1)]):
                self.balls[index].contact_next = bool(value)
        if self.balls:
            self.balls[-1].contact_next = False

        pending = [int(value) for value in pending_colors]
        if any(
            value < 0 or value >= self.active_num_colors
            for value in pending
        ):
            raise ValueError("pending colour is outside the curve palette")
        self.pending_colors = pending
        self.free_projectiles.clear()
        self.merging_projectiles.clear()
        self.tick_count = 0
        self.score = int(score)
        self.score_at_level_start = int(score_at_level_start)
        if (
            self.score < 0
            or self.score_at_level_start < 0
            or self.score_at_level_start > self.score
        ):
            raise ValueError(
                "score and score_at_level_start must be ordered "
                "non-negative integers"
            )
        self.current_bar_size = 0
        self.target_bar_size = 0
        self.zuma_reached = False
        self.stop_adding = False
        self.feed_plan_exhausted = False
        self.feed_refill_armed = False
        self.num_balls_created = len(values) + len(pending)
        self.advance_speed = np.float32(0.0)
        self.current_acceleration = np.float32(0.0)
        self.slow_count = 0
        self.backward_count = 0
        self.stop_time = 0
        self.first_ball_moved_backwards = False
        self.has_reached_rollout = False
        self.has_reached_cruising_speed = False
        self.first_chain_end = 0
        self._have_sets = False
        self.outcome = None
        self.win_pending = False
        self.skull_entry_pending = False
        self.loss_started = False
        self.loss_elapsed_ticks = 0
        self.loss_curve_index = None
        self.native_game_time = 0
        self._reset_fruit_runtime()
        self._reset_powerup_runtime()
        self.consecutive_clears = 0
        self.gun_state = GunState.NORMAL
        self.gun_state_percent = np.float32(1.0)
        self.hop_ticks_remaining = 0
        self.hop_target_index = None
        self.aim_angle = np.float32(0.0)
        self.current_color = (
            self._validate_optional_color(current_color)
            if current_color is not None
            else None
        )
        self.next_color = (
            self._validate_optional_color(next_color)
            if next_color is not None
            else None
        )
        configured_total = int(getattr(self.parameters, "num_balls", 0))
        if configured_total > 0 and self.num_balls_created >= configured_total:
            self.stop_adding = True
        self.last_events = TickEvents()
        self._ensure_gun_loaded()

    def load_shooter_random_state(
        self,
        *,
        crt_state: int,
        update_count: int,
        selected_index: int,
        weights: Sequence[float],
        sways: Sequence[float],
        last_hit: Sequence[int],
        previous_hit: Sequence[int],
    ) -> None:
        """Restore the retail thread-CRT and QRand shooter state exactly."""

        self.crt_rng.state = crt_state
        self._color_chooser.load_state(
            update_count=update_count,
            selected_index=selected_index,
            weights=weights,
            sways=sways,
            last_hit=last_hit,
            previous_hit=previous_hit,
        )

    def load_mtrand_state(
        self,
        *,
        words: Sequence[int],
        index: int,
    ) -> None:
        """Restore the retail global MTRand used by game logic."""

        self.rng.load_state(words, index)

    def _validate_optional_color(self, value: int) -> int:
        color = int(value)
        if not 0 <= color < self.num_colors:
            raise ValueError("colour is outside the curve palette")
        return color

    def ball_position(
        self,
        ball: ChainBall,
        *,
        curve_index: int | None = None,
    ) -> Float32Array:
        index = self.active_curve_index if curve_index is None else curve_index
        return np.asarray(
            self._point_xy_for_curve(index, ball.waypoint),
            dtype=np.float32,
        )

    def _point_xy(
        self,
        waypoint: float,
    ) -> tuple[np.float32, np.float32]:
        """Float32 scalar form of ``WayPointMgr.SetWayPoint`` lookup."""

        return self._point_xy_for_curve(self.active_curve_index, waypoint)

    def _point_xy_for_curve(
        self,
        curve_index: int,
        waypoint: float,
    ) -> tuple[np.float32, np.float32]:
        state = self._curve_states[int(curve_index)]
        curve = state.curve
        curve_points = state.curve_points
        waypoint_value = np.float32(waypoint)
        if curve_points is None:
            point = np.asarray(
                curve.point_at_waypoint(float(waypoint_value)),
                dtype=np.float32,
            ).reshape(2)
            return np.float32(point[0]), np.float32(point[1])
        raw_index = int(waypoint_value)
        if raw_index < 0:
            index = 0
            next_index = 1
        elif raw_index >= len(curve_points):
            index = int(curve.end_waypoint)
            next_index = index
        else:
            index = raw_index
            next_index = min(index + 1, int(curve.end_waypoint))
        start_x = np.float32(curve_points[index, 0])
        start_y = np.float32(curve_points[index, 1])
        end_x = np.float32(curve_points[next_index, 0])
        end_y = np.float32(curve_points[next_index, 1])
        delta_x = _f32_sub(end_x, start_x)
        delta_y = _f32_sub(end_y, start_y)
        if abs(delta_x) > _F32_FIVE or abs(delta_y) > _F32_FIVE:
            return start_x, start_y
        fraction = _f32_sub(waypoint_value, np.float32(int(waypoint_value)))
        return (
            _f32_add(start_x, _f32_mul(fraction, delta_x)),
            _f32_add(start_y, _f32_mul(fraction, delta_y)),
        )

    def in_tunnel_at_waypoint(
        self,
        waypoint: float,
        *,
        curve_index: int | None = None,
    ) -> bool:
        """Fast scalar form of the game's asymmetric tunnel lookup."""

        index_value = (
            self.active_curve_index if curve_index is None else int(curve_index)
        )
        state = self._curve_states[index_value]
        curve = state.curve
        waypoint_value = np.float32(waypoint)
        index = int(waypoint_value)
        if index < 0:
            return True
        if index > int(curve.end_waypoint):
            return False
        if state.curve_tunnels is not None:
            return bool(state.curve_tunnels[index])
        return bool(
            np.asarray(
                curve.is_in_tunnel_at_waypoint(float(waypoint_value)),
                dtype=np.bool_,
            ).item(),
        )

    def _perpendicular_xy(
        self,
        waypoint: float,
    ) -> tuple[np.float32, np.float32]:
        waypoint_value = np.float32(waypoint)
        if self._curve_points is None:
            perpendicular = np.asarray(
                self.curve.perpendicular_at_waypoint(float(waypoint_value)),
                dtype=np.float32,
            ).reshape(2)
            return np.float32(perpendicular[0]), np.float32(perpendicular[1])
        index = min(
            max(int(waypoint_value), 0),
            int(self.curve.end_waypoint),
        )
        point_x = np.float32(self._curve_points[index, 0])
        point_y = np.float32(self._curve_points[index, 1])
        use_previous = False
        if index + 1 < len(self._curve_points):
            other_x = np.float32(self._curve_points[index + 1, 0])
            other_y = np.float32(self._curve_points[index + 1, 1])
            if (
                (
                    abs(_f32_sub(point_x, other_x)) > _F32_FIVE
                    or abs(_f32_sub(point_y, other_y)) > _F32_FIVE
                )
                and index > 0
            ):
                use_previous = True
                other_x = np.float32(self._curve_points[index - 1, 0])
                other_y = np.float32(self._curve_points[index - 1, 1])
        else:
            use_previous = True
            other_x = np.float32(self._curve_points[index - 1, 0])
            other_y = np.float32(self._curve_points[index - 1, 1])
        if use_previous:
            perpendicular_x = _f32_sub(point_y, other_y)
            perpendicular_y = _f32_sub(other_x, point_x)
        else:
            perpendicular_x = _f32_sub(other_y, point_y)
            perpendicular_y = _f32_sub(point_x, other_x)
        squared = _f32_add(
            _f32_mul(perpendicular_x, perpendicular_x),
            _f32_mul(perpendicular_y, perpendicular_y),
        )
        length = np.float32(math.sqrt(float(squared)))
        inverse_length = _f32_div(_F32_ONE, length)
        return (
            _f32_mul(perpendicular_x, inverse_length),
            _f32_mul(perpendicular_y, inverse_length),
        )

    def _has_discontinuity(self, start: int, distance: int) -> bool:
        """Mirror ``WayPointMgr.CheckDiscontinuity`` for merge selection."""

        count = int(self.curve.end_waypoint) + 1
        lower = min(max(int(start), 0), count)
        upper = min(max(int(start + distance), 0), count)
        if lower >= upper:
            return False
        if self._curve_points is not None:
            points = self._curve_points[lower:upper]
        else:
            points = np.asarray(
                [self._point_xy(float(index)) for index in range(lower, upper)],
                dtype=np.float32,
            )
        if len(points) < 2:
            return False
        deltas = np.abs(np.diff(points, axis=0))
        manhattan = np.add(deltas[:, 0], deltas[:, 1], dtype=np.float32)
        return bool(np.any(manhattan > _F32_TEN))

    def _roll_in_speed(self) -> np.float32:
        """Initial chain speed, preserving RollBallsIn's mixed precision.

        The original stores every local except the explicitly cast
        ``Math.Sqrt`` expression as ``float``.  Keep that documented double
        square root, then round the final assignment to mAdvanceSpeed.
        """

        base_speed = np.float32(getattr(self.parameters, "speed", 0.5))
        start_percent = int(
            getattr(self.parameters, "start_distance_percent", 40),
        )
        target_distance = np.float32(
            (int(self.curve.end_waypoint) + 1) * start_percent // 100,
        )
        coefficient = _f32_add(_f32_mul(20.0, base_speed), _F32_ONE)
        negative_term = _f32_mul(-20.0, target_distance)
        discriminant = _f32_sub(
            _f32_mul(coefficient, coefficient),
            _f32_mul(4.0, negative_term),
        )
        ticks = int(
            (
                -float(coefficient)
                + math.sqrt(max(0.0, float(discriminant)))
            )
            / 2.0
        )
        return _f32_add(base_speed, _f32_mul(np.float32(ticks), 0.1))

    def _pending_run_length(self, color: int) -> int:
        count = 0
        if self.pending_colors:
            iterator = reversed(self.pending_colors)
        else:
            iterator = (ball.color for ball in self.balls)
        for candidate in iterator:
            if int(candidate) != color:
                break
            count += 1
        return count

    def _num_pending_singles(self, group_limit: int) -> int:
        sequence = list(reversed(self.pending_colors))
        sequence.extend(ball.color for ball in self.balls)
        group_count = 0
        previous = -1
        singles = 0
        run_length = 0
        for color in sequence:
            if group_count > group_limit:
                break
            if color != previous:
                if run_length == 1:
                    singles += 1
                run_length = 1
                group_count += 1
                previous = color
            else:
                run_length += 1
        return singles

    def _append_pending_color(self) -> int:
        if self.pending_colors:
            previous = self.pending_colors[-1]
        elif self.balls:
            previous = self.balls[0].color
        else:
            previous = self._rand_mod(self.active_num_colors)
        if previous >= self.active_num_colors:
            previous = self._rand_mod(self.active_num_colors)

        repeat = int(getattr(self.parameters, "ball_repeat_chance", 40))
        max_clump = int(getattr(self.parameters, "max_clump_size", 10))
        max_single = int(getattr(self.parameters, "max_single", 10))
        current_run = self._pending_run_length(previous)
        roll = self._rand_mod(100)
        # Revenge adds the max-clump guard seen at 0x00458CCC; the Deluxe
        # ancestor repeats solely from the percentage roll.
        if roll <= repeat and current_run < max_clump:
            color = previous
        elif (
            max_single < 10
            and self._num_pending_singles(1) == 1
            and (
                max_single == 0
                or self._num_pending_singles(10) > max_single
            )
        ):
            color = previous
        elif self.active_num_colors == 1:
            color = 0
        else:
            color = self._rand_mod(self.active_num_colors)
            while color == previous:
                color = self._rand_mod(self.active_num_colors)

        # Retail Ball setup immediately randomizes its sprite-frame phase after
        # choosing the colour (0x00458D65 -> 0x004020D0).  The frame itself is
        # rendering-only, but the draw shares the global MTRand and therefore
        # must still be consumed or every later gameplay colour is shifted.
        self.rng.next_u31()
        self.pending_colors.append(color)
        self.num_balls_created += 1
        configured_total = int(getattr(self.parameters, "num_balls", 0))
        if configured_total > 0 and self.num_balls_created >= configured_total:
            self.stop_adding = True
        return color

    def _waypoint_contact(
        self,
        first: ChainBall,
        second: ChainBall,
        pad: int = 0,
    ) -> bool:
        threshold = 2 * (self.config.ball_radius + int(pad))
        return abs(int(first.waypoint) - int(second.waypoint)) < threshold

    @staticmethod
    def _physical_collision(
        first_position: Float32Array,
        first_radius: int,
        second_position: Float32Array,
        second_radius: int,
        pad: int = 0,
    ) -> bool:
        first = np.asarray(first_position, dtype=np.float32)
        second = np.asarray(second_position, dtype=np.float32)
        delta = np.subtract(second, first, dtype=np.float32)
        distance_squared = _f32_add(
            _f32_mul(delta[0], delta[0]),
            _f32_mul(delta[1], delta[1]),
        )
        # Ball.CollidesWithPhysically evaluates the three radius terms and the
        # square as game floats before its strict comparison.
        radius = _f32_add(
            _f32_add(np.float32(second_radius), np.float32(pad * 2)),
            np.float32(first_radius),
        )
        radius_squared = _f32_mul(radius, radius)
        return bool(distance_squared < radius_squared)

    def _add_ball(self) -> None:
        if not self.pending_colors:
            if self.stop_adding or (
                self.feed_plan_exhausted and not self.feed_refill_armed
            ):
                return
            self._append_pending_color()
            self.feed_refill_armed = False
        color = self.pending_colors[0]
        candidate = ChainBall(
            id=self._new_id(),
            color=color,
            waypoint=np.float32(1.0),
            radius=self.config.ball_radius,
        )
        if self.balls:
            rear = self.balls[0]
            if self.advance_speed > rear.radius and rear.waypoint >= _F32_ZERO:
                if _f32_sub(rear.waypoint, self.advance_speed) < _F32_FIVE:
                    candidate.waypoint = _f32_sub(
                        _f32_sub(
                            _f32_sub(
                                rear.waypoint,
                                np.float32(rear.radius),
                            ),
                            np.float32(candidate.radius),
                        ),
                        np.float32(0.001),
                    )
            elif (
                candidate.waypoint > rear.waypoint
                or self._waypoint_contact(candidate, rear)
            ):
                return
        self.balls.insert(0, candidate)
        if len(self.balls) > 1:
            pad = self.config.contact_insert_pad + int(self.advance_speed)
            candidate.contact_next = self._waypoint_contact(
                candidate,
                self.balls[1],
                pad,
            )
        self.pending_colors.pop(0)
        # An exhausted Curve plan does not bootstrap a new queue from an
        # arbitrary empty midstate.  Retail does, however, refill one tick
        # after the last already-live entrance ball is consumed.  Preserve
        # that lineage explicitly so source-bound midstate replay can
        # distinguish those two otherwise identical empty-list snapshots.
        if self.feed_plan_exhausted and not self.pending_colors:
            self.feed_refill_armed = True
        if candidate.waypoint > _F32_ONE:
            self._add_ball()

    def set_aim(self, angle: float) -> None:
        """Set the cursor angle at the original game-float API boundary."""

        if not math.isfinite(angle):
            raise ValueError("angle must be finite")
        self.aim_angle = np.float32(angle)

    def request_fire(self, angle: float) -> bool:
        """Begin the original firing animation; release occurs about 6 ticks later."""

        if (
            self.win_pending
            or self.loss_started
            or self.gun_state is not GunState.NORMAL
            or self.hop_in_progress
            or self.current_color is None
        ):
            return False
        self.set_aim(angle)
        self.gun_state = GunState.FIRING
        self.gun_state_percent = np.float32(0.0)
        return True

    def swap_balls(self) -> bool:
        if (
            self.win_pending
            or self.loss_started
            or self.gun_state is not GunState.NORMAL
            or self.hop_in_progress
            or self.current_color is None
            or self.next_color is None
            or self.current_color == self.next_color
        ):
            return False
        self.current_color, self.next_color = (
            self.next_color,
            self.current_color,
        )
        return True

    def request_hop(self) -> bool:
        """Begin the retail input-locked transition to the other fixed pad."""

        if not self.can_hop_shooter:
            return False
        self.hop_target_index = (
            self.active_shooter_index + 1
        ) % self.shooter_position_count
        self.hop_ticks_remaining = self.config.hop_transition_ticks
        return True

    def _update_hop(self) -> None:
        if not self.hop_in_progress:
            return
        self.hop_ticks_remaining -= 1
        if self.hop_ticks_remaining == 0:
            if self.hop_target_index is None:
                raise RuntimeError("hop target is missing at transition completion")
            self.active_shooter_index = self.hop_target_index
            self.hop_target_index = None

    @staticmethod
    def _direction(angle: float) -> Float32Array:
        angle_value = np.float32(angle)
        return np.array(
            (
                np.float32(math.cos(float(angle_value))),
                np.float32(math.sin(float(angle_value))),
            ),
            dtype=np.float32,
        )

    def _muzzle_position(
        self,
        direction: Float32Array,
        *,
        preflight_updates: int = 0,
        speed: float | None = None,
    ) -> Float32Array:
        direction = np.asarray(direction, dtype=np.float32)
        left = np.array((-direction[1], direction[0]), dtype=np.float32)
        forward = np.float32(self.config.muzzle_forward_offset)
        if preflight_updates:
            forward = _f32_add(
                forward,
                _f32_mul(
                    np.float32(preflight_updates),
                    np.float32(
                        self.config.projectile_speed if speed is None else speed,
                    ),
                ),
            )
        position = np.add(
            self.shooter,
            np.multiply(direction, forward, dtype=np.float32),
            dtype=np.float32,
        )
        return np.add(
            position,
            np.multiply(
                left,
                np.float32(self.config.muzzle_lateral_offset),
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

    def launch_projectile(
        self,
        angle: float,
        *,
        color: int | None = None,
    ) -> Projectile:
        """Immediately inject a free projectile for replay and physics tests."""

        if not math.isfinite(angle):
            raise ValueError("angle must be finite")
        if color is None:
            if self.current_color is not None:
                color = self.current_color
            else:
                choice = self._choose_shooter_color()
                color = 0 if choice is None else choice
        color = self._validate_optional_color(color)
        direction = self._direction(float(np.float32(angle)))
        projectile = Projectile(
            id=self._new_id(),
            color=color,
            # This is a replay/test injection primitive, so its unambiguous
            # origin is the decoded gun centre.  Normal player firing goes
            # through request_fire() and retains the original muzzle offsets
            # and two pre-release in-frog updates.
            position=self.shooter.copy(),
            velocity=np.multiply(
                direction,
                np.float32(self.config.projectile_speed),
                dtype=np.float32,
            ),
            radius=self.config.ball_radius,
            merge_speed=np.float32(self.config.merge_speed),
        )
        self.free_projectiles.append(projectile)
        return projectile

    def _release_gun_projectile(self) -> None:
        if self.current_color is None:
            return
        direction = self._direction(self.aim_angle)
        speed = np.float32(self.config.projectile_speed)
        projectile = Projectile(
            id=self._new_id(),
            color=self.current_color,
            position=self._muzzle_position(
                direction,
                preflight_updates=4,
                speed=speed,
            ),
            velocity=np.multiply(direction, speed, dtype=np.float32),
            radius=self.config.ball_radius,
            merge_speed=np.float32(self.config.merge_speed),
        )
        self.free_projectiles.append(projectile)
        self.current_color = None
        self.gun_state = GunState.NORMAL
        self.last_events.fired += 1

    def _update_gun(self) -> None:
        if self.gun_state is GunState.FIRING:
            self.gun_state_percent = np.float32(
                self.gun_state_percent
                + np.float32(self.config.fire_state_step),
            )
            if self.gun_state_percent >= np.float32(
                self.config.release_threshold,
            ):
                self._release_gun_projectile()
        elif self.gun_state is GunState.RELOADING:
            self.gun_state_percent = np.float32(
                self.gun_state_percent
                + np.float32(self.config.reload_state_step),
            )
            if self.gun_state_percent > np.float32(1.0):
                self.gun_state_percent = np.float32(1.0)
                self.gun_state = GunState.NORMAL

    def _present_colors(self) -> list[int]:
        self._store_active_curve()
        colors = [
            ball.color
            for state in self._curve_states
            for ball in state.balls
            if not ball.exploding
        ]
        colors.extend(projectile.color for projectile in self.free_projectiles)
        colors.extend(
            projectile.color
            for state in self._curve_states
            for projectile in state.merging_projectiles
        )
        return colors

    def _choose_shooter_color(self) -> int | None:
        return self._color_chooser.choose(self._present_colors())

    def _ensure_gun_loaded(self) -> None:
        while self.current_color is None or self.next_color is None:
            color = self._choose_shooter_color()
            if color is None:
                return
            self.current_color = self.next_color
            self.next_color = color
            self.gun_state = GunState.RELOADING
            self.gun_state_percent = np.float32(0.0)

    def _check_gun_colors(self) -> None:
        """Replace chamber colours that disappeared from the live board."""

        if self.skull_entry_pending or self.loss_started:
            return
        present = self._present_colors()
        if not present:
            return
        allowed = set(present)
        if self.current_color is not None and self.current_color not in allowed:
            self.current_color = self._color_chooser.choose(present)
        if self.next_color is not None and self.next_color not in allowed:
            self.next_color = self._color_chooser.choose(present)

    def attach_projectile(
        self,
        projectile: Projectile,
        hit_index: int,
        hit_in_front: bool,
    ) -> None:
        """Attach a projectile to a chain ball, bypassing collision search."""

        if not 0 <= hit_index < len(self.balls):
            raise IndexError("hit_index is outside the ball chain")
        if projectile in self.free_projectiles:
            self.free_projectiles.remove(projectile)
        if projectile not in self.merging_projectiles:
            self.merging_projectiles.append(projectile)
        projectile.curve_index = self.active_curve_index
        ball = self.balls[hit_index]
        projectile.hit_ball_id = ball.id
        projectile.hit_in_front = bool(hit_in_front)
        projectile.hit_percent = np.float32(0.0)
        projectile.hit_position = projectile.position.copy()
        # Bullet fields are zero-initialized before the first CurveMgr.DoMerge.
        # UpdateBalls performs one merge update against that old destination.
        projectile.target_position = np.zeros(2, dtype=np.float32)
        projectile.waypoint = np.float32(ball.waypoint)
        projectile.have_set_prev_ball = False
        projectile.do_new_merge = self._has_discontinuity(
            int(ball.waypoint) - 80,
            160,
        )
        boundary_id: int | None
        if projectile.hit_in_front:
            boundary_id = (
                self.balls[hit_index + 1].id
                if hit_index + 1 < len(self.balls)
                else None
            )
        else:
            boundary_id = ball.id
        if boundary_id is not None:
            projectile.gap_info = [
                item for item in projectile.gap_info if item[0] != boundary_id
            ]

    def _ball_index_by_id(self, ball_id: int | None) -> int | None:
        if ball_id is None:
            return None
        for index, ball in enumerate(self.balls):
            if ball.id == ball_id:
                return index
        return None

    def _reserved_merge_ids(self) -> set[int]:
        return {
            int(projectile.hit_ball_id)
            for projectile in self.merging_projectiles
            if projectile.hit_ball_id is not None
        }

    def _try_projectile_collision(self, projectile: Projectile) -> bool:
        # The original first advances a merging bullet hit by this free bullet,
        # then continues the ordinary chain collision search.
        for merging in list(self.merging_projectiles):
            if not self._physical_collision(
                merging.position,
                merging.radius,
                projectile.position,
                projectile.radius,
            ):
                continue
            self._update_merging_projectile(merging)
            merge_index = next(
                (
                    index
                    for index, candidate in enumerate(self.merging_projectiles)
                    if candidate is merging
                ),
                None,
            )
            if merge_index is not None:
                self._advance_one_merging_projectile(merge_index)
            break

        reserved = self._reserved_merge_ids()
        for index, ball in enumerate(self.balls):
            if ball.exploding or ball.id in reserved:
                continue
            if (
                index > 0
                and self.balls[index - 1].contact_next
                and self.balls[index - 1].id in reserved
            ):
                continue
            if (
                index + 1 < len(self.balls)
                and ball.contact_next
                and self.balls[index + 1].id in reserved
            ):
                continue
            ball_position = self.ball_position(ball)
            if not self._physical_collision(
                ball_position,
                ball.radius,
                projectile.position,
                projectile.radius,
            ):
                continue
            perpendicular = self._perpendicular_xy(ball.waypoint)
            offset = np.subtract(
                projectile.position,
                ball_position,
                dtype=np.float32,
            )
            cross_z = _f32_sub(
                _f32_mul(offset[0], perpendicular[1]),
                _f32_mul(offset[1], perpendicular[0]),
            )
            hit_in_front = cross_z < _F32_ZERO
            tunnel_waypoint = int(ball.waypoint) + (
                ball.radius if hit_in_front else -ball.radius
            )
            in_tunnel = self.in_tunnel_at_waypoint(tunnel_waypoint)
            if in_tunnel:
                continue
            self.attach_projectile(projectile, index, hit_in_front)
            self.last_events.hits += 1
            return True
        return False

    def _projectile_out_of_bounds(self, projectile: Projectile) -> bool:
        x = np.float32(projectile.position[0])
        y = np.float32(projectile.position[1])
        return (
            x < np.float32(-self.config.left_miss_margin)
            or y < _F32_ZERO
            or _f32_sub(x, np.float32(projectile.radius))
            > _f32_add(
                np.float32(self.config.logical_width),
                np.float32(self.config.right_miss_margin),
            )
            or _f32_sub(y, np.float32(projectile.radius))
            > np.float32(self.config.logical_height)
        )

    def _check_gap_shot(self, projectile: Projectile) -> bool:
        """Record each distinct chain gap crossed by a free projectile.

        The pinned Revenge runtime at ``0x0045C500`` uses a ``2r`` waypoint
        stride but an ``r**2`` proximity threshold.  CircleShootApp's Deluxe
        reconstruction uses ``(2r)**2`` here, so that ancestor behavior must
        not be copied into the Revenge simulator.
        """

        radius = np.float32(projectile.radius)
        radius_squared = _f32_mul(radius, radius)
        count = int(self.curve.end_waypoint) + 1
        curve_index = self.active_curve_index
        curve_point = projectile.curve_points.get(
            curve_index,
            projectile.curve_point if self.curve_count == 1 else 0,
        )
        if 0 < curve_point < count:
            point = np.asarray(
                self._point_xy(float(curve_point)),
                dtype=np.float32,
            )
            delta = np.subtract(
                point,
                projectile.position,
                dtype=np.float32,
            )
            distance_squared = _f32_add(
                _f32_mul(delta[0], delta[0]),
                _f32_mul(delta[1], delta[1]),
            )
            if distance_squared < radius_squared:
                return False
            curve_point = 0
            projectile.curve_points[curve_index] = 0
            if self.curve_count == 1:
                projectile.curve_point = 0

        step = projectile.radius * 2
        for waypoint in range(1, count, step):
            if self.in_tunnel_at_waypoint(waypoint):
                continue
            point = np.asarray(self._point_xy(float(waypoint)), dtype=np.float32)
            delta = np.subtract(
                point,
                projectile.position,
                dtype=np.float32,
            )
            distance_squared = _f32_add(
                _f32_mul(delta[0], delta[0]),
                _f32_mul(delta[1], delta[1]),
            )
            if distance_squared >= radius_squared:
                continue
            projectile.curve_points[curve_index] = waypoint
            if self.curve_count == 1:
                projectile.curve_point = waypoint
            for index, ball in enumerate(self.balls):
                if ball.waypoint <= waypoint:
                    continue
                if index == 0:
                    break
                if ball.exploding and not any(
                    not candidate.exploding for candidate in self.balls[index + 1 :]
                ):
                    break
                previous = self.balls[index - 1]
                gap_distance = int(ball.waypoint - previous.waypoint)
                if gap_distance <= 0:
                    break
                if any(ball_id == ball.id for ball_id, _ in projectile.gap_info):
                    return False
                projectile.gap_info.append((ball.id, gap_distance))
                return True
            return False
        return False

    def _update_free_projectiles(self) -> None:
        original_curve = self.active_curve_index
        try:
            index = 0
            while index < len(self.free_projectiles):
                projectile = self.free_projectiles[index]
                removed = False
                for substep in range(2):
                    if projectile.just_fired:
                        projectile.just_fired = False
                    elif substep != 0:
                        projectile.position = np.add(
                            projectile.position,
                            projectile.velocity,
                            dtype=np.float32,
                        )
                    else:
                        # A bullet already present at frame start is only
                        # advanced/collision-tested by the second half-step.
                        continue

                    if self._try_projectile_fruit_collision(projectile):
                        self.free_projectiles.pop(index)
                        removed = True
                        break

                    for curve_index in range(self.curve_count):
                        self._activate_curve(curve_index)
                        if self._try_projectile_collision(projectile):
                            removed = True
                            break
                        self._check_gap_shot(projectile)
                    if removed:
                        break

                    if self._projectile_out_of_bounds(projectile):
                        self.free_projectiles.pop(index)
                        self._store_active_curve()
                        for state in self._curve_states:
                            state.consecutive_clears = 0
                        removed = True
                        break
                if removed:
                    continue
                index += 1
        finally:
            self._activate_curve(original_curve)

    def fruit_center(self) -> Float32Array | None:
        """Return the actor-visible retail fruit centre, including its bob."""

        calibration = self.fruit_calibration
        point_index = self.fruit_active_point_index
        if calibration is None or point_index is None:
            return None
        if not 0 <= point_index < len(calibration.points):
            raise RuntimeError("active fruit point index is outside calibration")
        point = calibration.points[point_index]
        return np.asarray(
            (
                _f32_add(
                    np.float32(point[0]),
                    np.float32(calibration.logical_width // 2),
                ),
                _f32_add(
                    _f32_add(
                        np.float32(point[1]),
                        np.float32(calibration.logical_height // 2),
                    ),
                    self.fruit_vertical_offset,
                ),
            ),
            dtype=np.float32,
        )

    def _collect_fruit(self) -> None:
        calibration = self.fruit_calibration
        if (
            calibration is None
            or self.fruit_active_point_index is None
            or self.fruit_collecting
        ):
            return
        tier = (self.score - self.score_at_level_start) // 600
        points = max(500, tier * 100)
        self.score += points
        self.last_events.score_delta += points
        self.fruit_collecting = True
        self.fruit_glow_step = 12
        self.fruit_collection_ticks_remaining = calibration.collection_ticks(
            self.config.tick_hz
        )
        self.fruit_collect_count += 1
        self.last_events.fruits_collected += 1

    def _try_projectile_fruit_collision(
        self,
        projectile: Projectile,
    ) -> bool:
        calibration = self.fruit_calibration
        if (
            calibration is None
            or self.fruit_active_point_index is None
            or self.fruit_collecting
        ):
            return False
        center = self.fruit_center()
        assert center is not None
        delta = np.subtract(projectile.position, center, dtype=np.float32)
        distance_squared = _f32_add(
            _f32_mul(delta[0], delta[0]),
            _f32_mul(delta[1], delta[1]),
        )
        radius = _f32_add(
            np.float32(projectile.radius),
            np.float32(calibration.logical_height // 2),
        )
        radius_squared = _f32_mul(radius, radius)
        if distance_squared > radius_squared:
            return False
        self._collect_fruit()
        return True

    def _try_proximity_bomb_fruit_collision(
        self,
        trigger_position: Float32Array,
    ) -> bool:
        """Apply retail type-0 power-up collision against TreasurePoint.

        The native type-0 handler at ``0x00418EDE-0x00418F72`` compares the
        raw level TreasurePoint coordinates (not the bobbed actor centre) to
        the bomb origin with the strict 108-pixel radius helper at
        ``0x0040AF00``.  Its two subtractions are stored as game floats before
        the x87 square-and-sum, so float64 is exact for the remaining
        arithmetic over those float32 operands.
        """

        calibration = self.fruit_calibration
        point_index = self.fruit_active_point_index
        if (
            calibration is None
            or point_index is None
            or self.fruit_collecting
            or not 0 <= point_index < len(calibration.points)
        ):
            return False
        point = calibration.points[point_index]
        trigger = np.asarray(trigger_position, dtype=np.float32)
        delta_x = _f32_sub(point[0], trigger[0])
        delta_y = _f32_sub(point[1], trigger[1])
        distance_squared = (
            float(delta_x) * float(delta_x)
            + float(delta_y) * float(delta_y)
        )
        radius = float(np.float32(
            self.config.proximity_bomb_fruit_radius
        ))
        if distance_squared >= radius * radius:
            return False
        self._collect_fruit()
        return True

    def _update_fruit_glow(self, *, collecting: bool) -> None:
        step = int(self.fruit_glow_step)
        value = int(self.fruit_glow_alpha) + step
        self.fruit_glow_alpha = value
        if step > 0 and value >= 255:
            self.fruit_glow_alpha = 255
            self.fruit_glow_step = -step
        elif step < 0 and value <= 0:
            self.fruit_glow_alpha = 0
            self.fruit_glow_step = 0 if collecting else -step

    @staticmethod
    def _fruit_bound_is_unset(value: np.float32) -> bool:
        return abs(float(value) - float(np.finfo(np.float32).max)) <= 1.0

    def _update_fruit_visual(self) -> None:
        """Advance Board fruit animation before projectile collision checks."""

        calibration = self.fruit_calibration
        if calibration is None:
            return
        if self.native_game_time == self.fruit_expiry_time - 200:
            self.fruit_glow_step *= 4
        if self.tick_count % 3 == 0:
            cell_count = calibration.sheet_columns * calibration.sheet_rows
            self.fruit_cell_index = (self.fruit_cell_index + 1) % cell_count

        if self.fruit_collecting:
            self.fruit_alpha = max(0, self.fruit_alpha - 8)
            self._update_fruit_glow(collecting=True)
            if self.fruit_active_point_index is not None:
                self.fruit_collection_ticks_remaining = max(
                    0,
                    self.fruit_collection_ticks_remaining - 1,
                )
                if self.fruit_collection_ticks_remaining == 0:
                    self.fruit_active_point_index = None
            return

        velocity = _f32_add(
            self.fruit_velocity,
            self.fruit_acceleration,
        )
        self.fruit_velocity = velocity
        if (
            self.fruit_acceleration < np.float32(0.0)
            and velocity <= -self.fruit_max_velocity
        ):
            self.fruit_acceleration = np.float32(-self.fruit_acceleration)
            if self._fruit_bound_is_unset(self.fruit_lower_bound):
                self.fruit_lower_bound = self.fruit_vertical_offset
            else:
                self.fruit_vertical_offset = self.fruit_lower_bound
        elif (
            self.fruit_acceleration >= np.float32(0.0)
            and velocity > self.fruit_max_velocity
        ):
            self.fruit_acceleration = np.float32(-self.fruit_acceleration)
            if self._fruit_bound_is_unset(self.fruit_upper_bound):
                self.fruit_upper_bound = self.fruit_vertical_offset
            else:
                self.fruit_vertical_offset = self.fruit_upper_bound
        self.fruit_vertical_offset = _f32_add(
            self.fruit_vertical_offset,
            self.fruit_velocity,
        )
        self._update_fruit_glow(collecting=False)

    def _fruit_curve_progress_percent(self, curve_index: int) -> int:
        """Mirror retail ``CurveMgr`` front-ball percentage truncation.

        The selector passes ``true`` to retail 0x004573A0.  That branch reads
        the active list's tail node directly (0x004573BB-0x004573E4), i.e. the
        ball furthest along an increasing-waypoint curve.  C118's natural
        spawn is a concrete discriminator: its tail is at about 87%, while
        the head is still below 1%.
        """

        self._store_active_curve()
        state = self._curve_states[int(curve_index)]
        if not state.balls:
            return 0
        point_count = int(state.curve.end_waypoint) + 1
        if point_count <= 0:
            return 0
        front_waypoint = float(np.float32(state.balls[-1].waypoint))
        return math.trunc(front_waypoint * 100.0 / point_count)

    def _eligible_fruit_points(self) -> tuple[int, ...]:
        calibration = self.fruit_calibration
        if calibration is None:
            return ()
        eligible: list[int] = []
        for curve_index in range(self.curve_count):
            progress = self._fruit_curve_progress_percent(curve_index)
            for point_index, row in enumerate(
                calibration.unlock_percentages
            ):
                threshold = row[curve_index]
                if threshold > 0 and progress >= threshold:
                    eligible.append(point_index)
        return tuple(eligible)

    def _update_fruit_scheduler(self) -> None:
        """Run the retail Board fruit expiry, chance, and spawn scheduler."""

        calibration = self.fruit_calibration
        if calibration is None:
            return
        if self.fruit_active_point_index is not None:
            if self.native_game_time >= self.fruit_expiry_time:
                self.fruit_active_point_index = None
                self.last_events.fruits_expired += 1
            # Retail returns immediately on the expiry update and does not run
            # another chance test in that same Board tick.
            return
        if self.score_target > 0 and self.score >= self.score_target:
            return
        if (
            self.native_game_time - self.fruit_expiry_time
            <= calibration.frequency_ticks
        ):
            return

        chance = self.rng.next_u31()
        self.last_events.fruit_chance_draws += 1
        if chance % calibration.frequency_ticks != 0:
            return
        eligible = self._eligible_fruit_points()
        if not eligible:
            return
        selected = eligible[self.rng.next_u31() % len(eligible)]
        self.fruit_active_point_index = selected
        self.fruit_expiry_time = (
            self.native_game_time + calibration.lifetime_ticks
        )
        self.fruit_collecting = False
        self.fruit_collection_ticks_remaining = 0
        self.fruit_alpha = 255
        self.fruit_velocity = np.float32(0.25)
        self.fruit_max_velocity = np.float32(0.25)
        self.fruit_acceleration = np.float32(-0.01)
        self.fruit_vertical_offset = np.float32(0.0)
        self.fruit_lower_bound = np.float32(np.finfo(np.float32).max)
        self.fruit_upper_bound = np.float32(np.finfo(np.float32).max)
        self.fruit_glow_alpha = 0
        self.fruit_glow_step = 12
        self.fruit_spawn_count += 1
        self.last_events.fruits_spawned += 1

    def _increment_merge(self, projectile: Projectile) -> None:
        # Two float32 +0.025 updates per tick intentionally finish on tick 21:
        # forty additions produce 0.99999958, not 1.0.
        value = np.float32(
            projectile.hit_percent
            + np.float32(projectile.merge_speed),
        )
        if value > np.float32(1.0):
            value = np.float32(1.0)
        projectile.hit_percent = value

    def _update_merging_projectile(self, projectile: Projectile) -> None:
        """Run ``Bullet.Update`` for a projectile already attached to the chain."""

        self._increment_merge(projectile)
        if projectile.do_new_merge:
            return
        start = (
            projectile.position
            if projectile.hit_position is None
            else projectile.hit_position
        )
        target = (
            np.zeros(2, dtype=np.float32)
            if projectile.target_position is None
            else projectile.target_position
        )
        delta = np.subtract(target, start, dtype=np.float32)
        scaled = np.multiply(
            np.float32(projectile.hit_percent),
            delta,
            dtype=np.float32,
        )
        projectile.position = np.add(start, scaled, dtype=np.float32)

    def _find_free_waypoint(
        self,
        existing: ChainBall,
        projectile: Projectile,
        in_front: bool,
    ) -> np.float32:
        return self._find_free_waypoint_from(
            existing_waypoint=existing.waypoint,
            existing_position=self.ball_position(existing),
            existing_radius=existing.radius,
            new_waypoint=projectile.waypoint,
            new_radius=projectile.radius,
            in_front=in_front,
        )

    def _find_free_waypoint_from(
        self,
        *,
        existing_waypoint: float,
        existing_position: Float32Array,
        existing_radius: int,
        new_waypoint: float,
        new_radius: int,
        in_front: bool,
        pad: int = 0,
    ) -> np.float32:
        direction = 1 if in_front else -1
        # Retail FindFreeWayPoint always scans outward from the existing
        # chain ball.  Starting at the projectile's previous waypoint looks
        # equivalent in most states, but skips a newly free integer sample as
        # the chain advances.  The PC Jungle2 trajectory exposes that boundary
        # at update 7764 (2391 -> 2392).
        waypoint = int(existing_waypoint)
        del new_waypoint
        while 0 <= waypoint <= int(self.curve.end_waypoint):
            candidate = np.asarray(
                self._point_xy(float(waypoint)),
                dtype=np.float32,
            )
            if not self._physical_collision(
                existing_position,
                existing_radius,
                candidate,
                new_radius,
                pad,
            ):
                break
            waypoint += direction
        return np.float32(waypoint)

    def _redirect_merge_to_previous(self, projectile: Projectile) -> None:
        if projectile.have_set_prev_ball:
            return
        hit_index = self._ball_index_by_id(projectile.hit_ball_id)
        if hit_index is None or hit_index == 0:
            return
        previous = self.balls[hit_index - 1]
        if previous.exploding:
            return
        if not self._physical_collision(
            self.ball_position(previous),
            previous.radius,
            projectile.position,
            projectile.radius,
        ):
            return
        projectile.have_set_prev_ball = True
        projectile.hit_ball_id = previous.id
        projectile.hit_in_front = True
        projectile.hit_percent = np.float32(0.0)
        projectile.hit_position = projectile.position.copy()
        projectile.waypoint = np.float32(previous.waypoint)

    def _powerup_transition_duration(self, powerup_type: int) -> int:
        if powerup_type == int(PowerupType.BONUS_BALL):
            return self.config.bonus_ball_transition_ticks
        return self.config.powerup_transition_ticks

    def _set_ball_powerup(
        self,
        ball: ChainBall,
        powerup_type: int,
        *,
        animated: bool,
    ) -> bool:
        none_type = int(PowerupType.NONE)
        if not 0 <= powerup_type <= none_type:
            raise ValueError("power-up type is outside the retail enum")
        current_type = ball.powerup_primary_type
        if current_type == powerup_type:
            return False
        ball.powerup_visual_index = -1
        if powerup_type != none_type:
            ball.powerup_previous_ticks = 0
            ball.powerup_previous_type = none_type
        if animated:
            ball.powerup_secondary_type = powerup_type
            duration = (
                self.config.bonus_ball_transition_ticks
                if (
                    powerup_type == none_type
                    and current_type == int(PowerupType.BONUS_BALL)
                )
                else self.config.powerup_transition_ticks
            )
            ball.powerup_transition_ticks = duration
            ball.powerup_visual_index = _POWERUP_VISUAL_INDEX.get(
                powerup_type,
                -1,
            )
            ball.powerup_visual_scale = np.float32(5.0)
            ball.powerup_visual_step = _f32_div(
                np.float32(4.0),
                np.float32(duration),
            )
        else:
            ball.powerup_secondary_type = none_type
            ball.powerup_primary_type = powerup_type
        if powerup_type != none_type:
            self.active_powerup_color_counts[ball.color] += 1
        return True

    def _expire_ball_powerup(
        self,
        ball: ChainBall,
        powerup_type: int,
    ) -> None:
        if not 0 <= powerup_type < int(PowerupType.NONE):
            raise RuntimeError("active ball has an invalid power-up type")
        active_count = self.active_powerup_color_counts[ball.color]
        if active_count <= 0:
            raise RuntimeError(
                "active power-up color count underflow"
            )
        self.active_powerup_color_counts[ball.color] = active_count - 1
        self.powerup_cooldown_times[powerup_type] = self.native_game_time
        ball.powerup_previous_ticks = (
            self.config.powerup_previous_retention_ticks
        )
        ball.powerup_previous_type = powerup_type
        if not self._set_ball_powerup(
            ball,
            int(PowerupType.NONE),
            animated=True,
        ):
            raise RuntimeError(
                "active power-up expiration produced no transition"
            )

    @staticmethod
    def _effective_ball_powerup(ball: ChainBall) -> int:
        """Return the power-up selected by retail ``CurveMgr.ExplodeBall``."""

        none_type = int(PowerupType.NONE)
        if ball.powerup_primary_type != none_type:
            return ball.powerup_primary_type
        if (
            ball.powerup_previous_ticks > 0
            and ball.powerup_previous_type != none_type
        ):
            return ball.powerup_previous_type
        return ball.powerup_secondary_type

    @staticmethod
    def effective_ball_powerup(ball: ChainBall) -> int:
        """Return the currently visible/effective retail power-up type."""

        if not isinstance(ball, ChainBall):
            raise TypeError("ball must be a ChainBall")
        return RevengeSimulator._effective_ball_powerup(ball)

    def _trigger_ball_powerup(
        self,
        ball: ChainBall,
        powerup_type: int,
    ) -> int:
        """Apply one proven retail power-up after its ball starts exploding."""

        supported = {
            int(PowerupType.PROXIMITY_BOMB),
            int(PowerupType.SLOW),
            int(PowerupType.REVERSE),
        }
        if powerup_type not in supported:
            raise NotImplementedError(
                f"power-up trigger type {powerup_type} is not fidelity-safe"
            )

        active_count = self.active_powerup_color_counts[ball.color]
        self.active_powerup_color_counts[ball.color] = max(
            0,
            active_count - 1,
        )
        self.powerup_field_124_by_type[powerup_type] += 1
        self.last_events.powerups_triggered += 1
        recursively_exploded = 0

        if powerup_type == int(PowerupType.PROXIMITY_BOMB):
            trigger_position = self.ball_position(ball)
            # The native Board handler applies this to every curve.  This
            # simulator is deliberately single-curve, so list order here is
            # the exact one-curve subset of that traversal.
            for candidate in self.balls:
                if candidate.exploding:
                    continue
                if self._physical_collision(
                    trigger_position,
                    ball.radius,
                    self.ball_position(candidate),
                    candidate.radius,
                    self.config.proximity_bomb_collision_pad,
                ):
                    recursively_exploded += self._begin_ball_explosion(
                        candidate
                    )
            self._try_proximity_bomb_fruit_collision(trigger_position)
        elif powerup_type == int(PowerupType.SLOW):
            if self.slow_count < self.config.slow_powerup_replace_threshold:
                self.slow_count = self.config.slow_powerup_ticks
        elif powerup_type == int(PowerupType.REVERSE) and self.balls:
            self.backward_count = self.config.reverse_powerup_ticks

        self.powerup_cooldown_times[powerup_type] = self.native_game_time
        self.powerup_triggered = True
        return recursively_exploded

    def _begin_ball_explosion(self, ball: ChainBall) -> int:
        """Enter explosion once, recursively applying any attached power-up."""

        if ball.exploding:
            return 0
        # Retail CurveMgr::ExplodeBall writes curve+0x1B4 from Ball+0x1C
        # on every newly exploding ball, before dispatching its power-up.
        # Keeping this here (rather than once per match) also preserves the
        # active-list order of recursive proximity-bomb explosions.
        self.last_powerup_waypoint = int(ball.waypoint)
        powerup_type = self._effective_ball_powerup(ball)
        none_type = int(PowerupType.NONE)
        if not 0 <= powerup_type <= none_type:
            raise RuntimeError("active ball has an invalid power-up type")
        if powerup_type != none_type and powerup_type not in {
            int(PowerupType.PROXIMITY_BOMB),
            int(PowerupType.SLOW),
            int(PowerupType.REVERSE),
        }:
            raise NotImplementedError(
                f"power-up trigger type {powerup_type} is not fidelity-safe"
            )

        ball.exploding = True
        ball.explode_frame = 0
        ball.should_remove = False
        self.last_events.balls_exploded += 1
        exploded_count = 1
        if powerup_type != none_type:
            exploded_count += self._trigger_ball_powerup(ball, powerup_type)
        return exploded_count

    def _powerup_weights(self) -> tuple[int, ...]:
        calibration = self.powerup_calibration
        if calibration is None:
            return (0,) * int(PowerupType.NONE)
        weights = [0] * int(PowerupType.NONE)
        supported = set(calibration.supported_types)
        records = tuple(
            getattr(self.parameters, "powerup_records", ())
        )
        if len(records) > int(PowerupType.NONE):
            raise ValueError("CURV power-up record count is invalid")
        for powerup_type, record in enumerate(records):
            if len(record) != 2:
                raise ValueError("CURV power-up record is invalid")
            weight = int(record[0])
            if weight < 0:
                raise ValueError("CURV power-up weight is invalid")
            if weight > 0 and powerup_type not in supported:
                raise NotImplementedError(
                    "calibration does not support positive-weight "
                    f"power-up type {powerup_type}"
                )
            if powerup_type in supported:
                weights[powerup_type] += weight
        return tuple(weights)

    def _maybe_spawn_powerup(self) -> ChainBall | None:
        calibration = self.powerup_calibration
        if calibration is None or not self.balls:
            return None
        if self.native_game_time < calibration.initial_delay_ticks:
            return None
        if (
            calibration.spawn_delay_ticks > 0
            and self.native_game_time - self.powerup_last_any_spawn_time
            < calibration.spawn_delay_ticks
        ):
            return None
        if (
            calibration.unique_color
            and all(
                self.active_powerup_color_counts[color] > 0
                for color in range(self.active_num_colors)
            )
        ):
            return None
        if self._rand_mod(calibration.chance_denominator) != 0:
            return None

        weights = self._powerup_weights()
        total_weight = sum(weights)
        if total_weight <= 0:
            return None
        weighted_roll = self._rand_mod(total_weight)
        cumulative = 0
        selected_type: int | None = None
        for powerup_type, weight in enumerate(weights):
            cumulative += weight
            if weighted_roll < cumulative:
                selected_type = powerup_type
                break
        if selected_type is None:
            raise RuntimeError("weighted power-up roll escaped support")
        if (
            self.native_game_time
            - self.powerup_last_spawn_times[selected_type]
            < calibration.cooldown_ticks
        ):
            return None
        cooldown_time = self.powerup_cooldown_times[selected_type]
        if (
            cooldown_time > 0
            and self.native_game_time - cooldown_time
            < calibration.cooldown_ticks
        ):
            return None

        eligible_colors = [
            color
            for color in range(self.active_num_colors)
            if (
                selected_type == int(PowerupType.BONUS_BALL)
                or not calibration.unique_color
                or self.active_powerup_color_counts[color] == 0
            )
        ]
        if not eligible_colors:
            return None
        selected_color = eligible_colors[
            self._rand_mod(len(eligible_colors))
        ]
        candidates = [
            ball
            for ball in self.balls
            if (
                ball.color == selected_color
                and ball.powerup_primary_type == int(PowerupType.NONE)
                and ball.powerup_secondary_type == int(PowerupType.NONE)
                and not ball.exploding
            )
        ]
        if not candidates:
            return None
        selected_ball = candidates[self._rand_mod(len(candidates))]
        if not self._set_ball_powerup(
            selected_ball,
            selected_type,
            animated=True,
        ):
            raise RuntimeError("selected power-up ball rejected mutation")
        self.powerup_last_any_spawn_time = self.native_game_time
        self.powerup_last_spawn_times[selected_type] = self.native_game_time
        self.powerup_spawn_counts[selected_type] += 1
        self.last_events.powerups_spawned += 1
        return selected_ball

    def _update_ball_powerup(self, ball: ChainBall) -> None:
        none_type = int(PowerupType.NONE)
        if ball.powerup_transition_ticks > 0:
            ball.powerup_visual_scale = np.maximum(
                _F32_ONE,
                _f32_sub(
                    ball.powerup_visual_scale,
                    ball.powerup_visual_step,
                ),
            )
            ball.powerup_transition_ticks -= 1
            if ball.powerup_transition_ticks == 0:
                ball.powerup_primary_type = (
                    ball.powerup_secondary_type
                )
                ball.powerup_secondary_type = none_type
                ball.powerup_visual_index = -1
                if ball.powerup_primary_type != none_type:
                    ball.powerup_lifetime_ticks = (
                        self.config.powerup_lifetime_ticks
                    )

        # Retail removes a clearing ball from the rotation/power-up update
        # path. A power-up triggered during insertion has already consumed
        # this tick's decrement, then keeps its remaining lifetime frozen
        # throughout the explosion animation.
        if not ball.exploding and ball.powerup_lifetime_ticks > 0:
            ball.powerup_lifetime_ticks -= 1
            if ball.powerup_lifetime_ticks == 0:
                powerup_type = ball.powerup_primary_type
                if powerup_type != none_type:
                    self._expire_ball_powerup(
                        ball,
                        powerup_type,
                    )

        if ball.powerup_previous_ticks > 0:
            ball.powerup_previous_ticks -= 1
            if ball.powerup_previous_ticks == 0:
                ball.powerup_previous_type = none_type

    def _update_ball_objects(self) -> None:
        for ball in self.balls:
            ball.update_count += 1
            self._update_ball_powerup(ball)
        for projectile in list(self.merging_projectiles):
            self._update_merging_projectile(projectile)

    def _advance_merging_projectiles(self) -> None:
        index = 0
        while index < len(self.merging_projectiles):
            if self._advance_one_merging_projectile(index):
                index += 1

    def _advance_one_merging_projectile(self, index: int) -> bool:
        projectile = self.merging_projectiles[index]
        self._redirect_merge_to_previous(projectile)
        hit_index = self._ball_index_by_id(projectile.hit_ball_id)
        if hit_index is None:
            self.merging_projectiles.pop(index)
            return False
        hit_ball = self.balls[hit_index]
        target_waypoint = self._find_free_waypoint(
            hit_ball,
            projectile,
            projectile.hit_in_front,
        )
        projectile.waypoint = np.float32(target_waypoint)
        projectile.target_position = np.asarray(
            self._point_xy(target_waypoint),
            dtype=np.float32,
        )
        # SetWayPoint/FindFreeWayPoint move the bullet before Bullet.Update.
        projectile.position = projectile.target_position.copy()
        self._update_merging_projectile(projectile)

        self._push_for_merge(projectile, hit_index)
        if projectile.hit_percent >= np.float32(1.0):
            self._finish_merge(index)
            return False
        return True

    def _push_for_merge(
        self,
        projectile: Projectile,
        hit_index: int,
    ) -> None:
        push_index = hit_index + 1 if projectile.hit_in_front else hit_index
        if not 0 <= push_index < len(self.balls):
            return
        push_ball = self.balls[push_index]
        if not projectile.do_new_merge and not self._physical_collision(
            self.ball_position(push_ball),
            push_ball.radius,
            projectile.position,
            projectile.radius,
        ):
            return
        original_waypoint = np.float32(push_ball.waypoint)
        remaining = _f32_sub(_F32_ONE, projectile.hit_percent)
        pad = int(
            _f32_div(
                _f32_mul(np.float32(-projectile.radius), remaining),
                np.float32(2.0),
            ),
        )
        push_ball.waypoint = self._find_free_waypoint_from(
            existing_waypoint=projectile.waypoint,
            existing_position=projectile.position,
            existing_radius=projectile.radius,
            new_waypoint=push_ball.waypoint,
            new_radius=push_ball.radius,
            in_front=True,
            pad=pad,
        )
        target_separation = _f32_mul(
            _f32_mul(projectile.hit_percent, projectile.hit_percent),
            np.float32(push_ball.radius + projectile.radius),
        )
        if (
            _f32_sub(push_ball.waypoint, projectile.waypoint)
            > target_separation
        ):
            proposed = _f32_add(projectile.waypoint, target_separation)
            if proposed > original_waypoint:
                push_ball.waypoint = np.float32(proposed)
            else:
                push_ball.waypoint = np.float32(original_waypoint)
        self._cancel_conflicting_suckback(projectile, hit_index)

    def _cancel_conflicting_suckback(
        self,
        projectile: Projectile,
        hit_index: int,
    ) -> None:
        """Mirror DoMerge's ordinary-play interruption of backward suction."""

        cancel_index = hit_index + 1 if projectile.hit_in_front else hit_index
        if not 0 <= cancel_index < len(self.balls):
            return
        candidate = self.balls[cancel_index]
        if (
            candidate.suck_back
            and candidate.suck_count > 0
            and candidate.color != projectile.color
        ):
            candidate.suck_count = 0
            # Ball.SetSuckCount(count) also restores the default direction.
            candidate.suck_back = True

    def _update_collision_info(self, index: int, pad: int) -> None:
        if index > 0:
            self.balls[index - 1].contact_next = self._waypoint_contact(
                self.balls[index - 1],
                self.balls[index],
                pad,
            )
        if index + 1 < len(self.balls):
            self.balls[index].contact_next = self._waypoint_contact(
                self.balls[index],
                self.balls[index + 1],
                pad,
            )
        else:
            self.balls[index].contact_next = False

    def _finish_merge(self, projectile_index: int) -> None:
        projectile = self.merging_projectiles[projectile_index]
        hit_index = self._ball_index_by_id(projectile.hit_ball_id)
        if hit_index is None:
            self.merging_projectiles.pop(projectile_index)
            return
        minimum_gap = min(
            (distance for _, distance in projectile.gap_info),
            default=0,
        )
        number_of_gaps = len(projectile.gap_info)
        insertion_index = hit_index + (1 if projectile.hit_in_front else 0)
        ball = ChainBall(
            id=self._new_id(),
            color=projectile.color,
            waypoint=np.float32(projectile.waypoint),
            radius=self.config.ball_radius,
        )
        self.balls.insert(insertion_index, ball)
        self.merging_projectiles.pop(projectile_index)
        if minimum_gap > 0:
            adjusted_gap = max(0, minimum_gap - 64)
            gap_bonus = 500 * (300 - adjusted_gap) // 300
            gap_bonus = gap_bonus // 10 * 10
            gap_bonus = max(10, gap_bonus)
            if number_of_gaps > 1:
                gap_bonus *= number_of_gaps
            ball.gap_bonus = gap_bonus
            ball.num_gaps = number_of_gaps
        self._update_collision_info(
            insertion_index,
            self.config.contact_insert_pad,
        )
        self.last_events.inserted += 1

        previous = self.balls[insertion_index - 1] if insertion_index > 0 else None
        next_ball = (
            self.balls[insertion_index + 1]
            if insertion_index + 1 < len(self.balls)
            else None
        )
        if self._check_set(insertion_index):
            self.consecutive_clears += 1
            return
        if (
            previous is not None
            and not previous.contact_next
            and previous.color == ball.color
            and not previous.exploding
        ):
            ball.suck_pending = True
            ball.suck_count = 1
            return
        if (
            next_ball is not None
            and not ball.contact_next
            and next_ball.color == ball.color
            and not next_ball.exploding
        ):
            ball.suck_pending = True
            if next_ball.suck_count <= 0:
                next_ball.suck_count = 1
            return
        self.consecutive_clears = 0

    def _run_bounds(self, seed_index: int) -> tuple[int, int]:
        color = self.balls[seed_index].color
        left = seed_index
        while (
            left > 0
            and self.balls[left - 1].contact_next
            and self.balls[left - 1].color == color
        ):
            left -= 1
        right = seed_index
        while (
            right + 1 < len(self.balls)
            and self.balls[right].contact_next
            and self.balls[right + 1].color == color
        ):
            right += 1
        return left, right

    def _score_match(
        self,
        removed_count: int,
        combo_count: int,
        gap_bonus: int,
    ) -> int:
        if removed_count == 0:
            return 0
        points = removed_count * 10 + combo_count * 100 + gap_bonus
        # Native PC code reads the old counter and increments it only after a
        # successful direct clear.  The first +100 therefore occurs on clear
        # six, followed by +110 on clear seven.
        if combo_count == 0 and self.consecutive_clears > 4:
            points += 100 + 10 * (self.consecutive_clears - 5)
        self.score += points
        self.last_events.score_delta += points
        return points

    def _check_set(self, seed_index: int) -> bool:
        # Native CurveMgr.Match clears this byte on every invocation, then
        # CurveMgr.ExplodeBall sets it if any direct or recursive power-up fires.
        self.powerup_triggered = False
        if not 0 <= seed_index < len(self.balls):
            return False
        left, right = self._run_bounds(seed_index)
        count = right - left + 1
        if count < 3:
            return False
        seed = self.balls[seed_index]
        combo_count = seed.combo_count
        previous_combo_score = seed.combo_score
        gap_bonus = sum(ball.gap_bonus for ball in self.balls[left : right + 1])
        exploding_before = {
            ball.id for ball in self.balls if ball.exploding
        }
        newly_exploding = 0
        for ball in self.balls[left : right + 1]:
            if ball.suck_pending:
                ball.suck_pending = False
                self.consecutive_clears += 1
            newly_exploding += self._begin_ball_explosion(ball)
            ball.gap_bonus = 0
            ball.num_gaps = 0
        points = self._score_match(newly_exploding, combo_count, gap_bonus)
        combo_score = previous_combo_score + points
        for ball in self.balls:
            if ball.id not in exploding_before and ball.exploding:
                # ActivateBomb queues every recursively cleared neighbour for
                # the same post-scoring combo assignment as the direct run.
                ball.combo_count = combo_count
                ball.combo_score = combo_score
        self.last_events.matches += 1
        return True

    def _clear_pending_sucks(self, end_index: int) -> None:
        """Mirror ``CurveMgr.ClearPendingSucks`` over the affected topology."""

        connected = True
        index = end_index
        while index >= 0:
            ball = self.balls[index]
            if ball.suck_pending:
                ball.suck_pending = False
                self.consecutive_clears = 0
                ball.gap_bonus = 0
                ball.num_gaps = 0
            index -= 1
            if index < 0:
                return
            previous = self.balls[index]
            if not previous.contact_next:
                connected = False
            if not connected and previous.suck_count > 0:
                return

    def _reposition_attached_merge_after_suck(self, moved: ChainBall) -> None:
        """Keep a merging bullet coherent when its hit ball is sucked back."""

        for projectile in self.merging_projectiles:
            if projectile.hit_ball_id != moved.id or projectile.do_new_merge:
                continue
            hit_index = self._ball_index_by_id(projectile.hit_ball_id)
            if hit_index is None:
                continue
            push_index = (
                hit_index + 1 if projectile.hit_in_front else hit_index
            )
            if 0 <= push_index < len(self.balls):
                push_ball = self.balls[push_index]
                if self._physical_collision(
                    self.ball_position(push_ball),
                    push_ball.radius,
                    projectile.position,
                    projectile.radius,
                ):
                    projectile.waypoint = self._find_free_waypoint_from(
                        existing_waypoint=push_ball.waypoint,
                        existing_position=self.ball_position(push_ball),
                        existing_radius=push_ball.radius,
                        new_waypoint=projectile.waypoint,
                        new_radius=projectile.radius,
                        in_front=False,
                    )
                    projectile.position = np.asarray(
                        self._point_xy(projectile.waypoint),
                        dtype=np.float32,
                    )
            # Bullet.UpdateHitPos runs even when GetPushBall returns null.
            projectile.hit_position = projectile.position.copy()

    def _update_sucking_balls(self) -> None:
        anchor_index = 0
        while anchor_index < len(self.balls):
            anchor = self.balls[anchor_index]
            if anchor.suck_count <= 0 or not anchor.suck_back:
                anchor_index += 1
                continue

            old_suck_count = anchor.suck_count
            # Retail UpdateSuckingBalls first performs the integer ``>> 3``
            # ramp and then multiplies it by the same per-curve reverse-speed
            # field used by AdvanceBackwardBalls.  The default is 1.0, which
            # hid this distinction in the original regression matrix.
            speed = _f32_mul(
                np.float32(old_suck_count >> 3),
                np.float32(self.config.reverse_speed),
            )
            moved_last = anchor_index
            index = anchor_index
            while index < len(self.balls):
                moved = self.balls[index]
                moved.suck_count = 0
                moved.waypoint = _f32_sub(moved.waypoint, speed)
                self._reposition_attached_merge_after_suck(moved)
                moved_last = index
                if not moved.contact_next:
                    break
                index += 1
            next_anchor_index = moved_last + 1
            anchor.suck_count = old_suck_count + 1

            previous_index = anchor_index - 1
            if (
                previous_index >= 0
                and self.balls[previous_index].exploding
                and self.balls[previous_index].color == anchor.color
            ):
                while previous_index >= 0:
                    previous_index -= 1
                    if (
                        previous_index >= 0
                        and not self.balls[previous_index].exploding
                    ):
                        break

            if previous_index < 0:
                anchor.suck_count = 0
                anchor_index = next_anchor_index
                continue
            previous = self.balls[previous_index]
            if previous.color != anchor.color:
                anchor.suck_count = 0
                anchor_index = next_anchor_index
                continue

            contact_waypoint = _f32_sub(
                _f32_sub(anchor.waypoint, np.float32(anchor.radius)),
                np.float32(previous.radius),
            )
            if previous.waypoint > contact_waypoint:
                previous.waypoint = np.float32(contact_waypoint)
                previous.contact_next = True
                anchor.suck_count = 0
                matched = self._check_set(anchor_index)
                if not matched:
                    anchor.combo_count = 0
                    anchor.combo_score = 0
                front = self.balls[moved_last]
                if front.backwards_count == 0:
                    front.backwards_count = 30
                    backwards_speed = _f32_mul(
                        np.float32(anchor.combo_count),
                        np.float32(1.5),
                    )
                    if backwards_speed <= _F32_HALF:
                        backwards_speed = _F32_HALF
                    front.backwards_speed = np.float32(backwards_speed)
                self._clear_pending_sucks(moved_last)
            anchor_index = next_anchor_index

    def _target_chain_speed(self) -> np.float32:
        target = np.float32(getattr(self.parameters, "speed", 0.5))
        acceleration_rate = np.float32(
            getattr(self.parameters, "acceleration_rate", 0.0),
        )
        if acceleration_rate != _F32_ZERO:
            self.current_acceleration = _f32_add(
                self.current_acceleration,
                acceleration_rate,
            )
            target = _f32_add(target, self.current_acceleration)
            max_speed = np.float32(
                getattr(self.parameters, "max_speed", 100.0),
            )
            if target > max_speed:
                target = max_speed
        if self.slow_count != 0:
            target = _f32_div(target, np.float32(4.0))
        if not self.balls:
            return np.float32(target)

        # CurveMgr.AdvanceBalls reads mFirstChainEnd from the previous tick.
        # It refreshes that persistent integer only after this tick's contact
        # propagation, so merge/suck topology changes must not affect danger
        # slowdown one tick early.
        first_chain_end = self.first_chain_end
        slow_distance = int(getattr(self.parameters, "slow_distance", 500))
        slow_factor = np.float32(
            getattr(self.parameters, "slow_factor", 4.0),
        )
        if slow_factor < _F32_ONE:
            slow_factor = _F32_ONE
        if (
            bool(getattr(self.curve, "die_at_end", True))
            and self.has_reached_cruising_speed
            and slow_distance > 0
            and first_chain_end >= self.danger_point - slow_distance
        ):
            if first_chain_end < self.danger_point:
                fraction = _f32_div(
                    np.float32(
                        first_chain_end
                        - (self.danger_point - slow_distance),
                    ),
                    np.float32(slow_distance),
                )
                target = _f32_add(
                    _f32_mul(_f32_sub(_F32_ONE, fraction), target),
                    _f32_div(
                        _f32_mul(fraction, target),
                        slow_factor,
                    ),
                )
            else:
                target = _f32_div(target, slow_factor)
        return np.float32(target)

    def _first_chain_end(self) -> int:
        if not self.balls:
            return 0
        for ball in self.balls[:-1]:
            if not ball.contact_next:
                return int(ball.waypoint)
        return int(self.balls[-1].waypoint)

    def _advance_balls(self) -> None:
        if not self.balls:
            return
        target = self._target_chain_speed()
        if self.advance_speed > target:
            ramped = _f32_sub(
                self.advance_speed,
                np.float32(self.config.speed_ramp_down),
            )
            # CurveMgr snaps an ordinary ramp-down to the current target when
            # its final subtraction crosses it.  The cruising latch is set one
            # tick before that crossing in the retail game, so it cannot be
            # used to identify the RollBallsIn transition.  Active slow/reverse
            # effects deliberately retain the native overshoot.
            self.advance_speed = (
                np.float32(target)
                if (
                    self.backward_count <= 0
                    and self.slow_count <= 0
                    and ramped < target
                )
                else np.float32(ramped)
            )
        elif self.advance_speed < target:
            ramped = _f32_add(
                self.advance_speed,
                np.float32(self.config.speed_ramp_up),
            )
            self.advance_speed = (
                np.float32(target) if ramped >= target else np.float32(ramped)
            )
        if not self.first_ball_moved_backwards and self.stop_time == 0:
            self.balls[0].waypoint = _f32_add(
                self.balls[0].waypoint,
                self.advance_speed,
            )

        for index in range(len(self.balls) - 1):
            rear = self.balls[index]
            front = self.balls[index + 1]
            minimum_front = _f32_add(
                _f32_add(rear.waypoint, np.float32(rear.radius)),
                np.float32(front.radius),
            )
            if front.waypoint < minimum_front:
                front.waypoint = np.float32(minimum_front)
                rear.contact_next = True
        self.first_chain_end = self._first_chain_end()
        threshold = _f32_mul(
            _f32_div(
                np.float32(
                    int(
                        getattr(
                            self.parameters,
                            "start_distance_percent",
                            40,
                        ),
                    ),
                ),
                np.float32(100.0),
            ),
            np.float32(self.curve.end_waypoint + 1),
        )
        if self.backward_count <= 0 and self.balls[-1].waypoint >= threshold:
            self.has_reached_rollout = True
        if (
            not self.has_reached_cruising_speed
            and _f32_sub(
                self.advance_speed,
                np.float32(getattr(self.parameters, "speed", 0.5)),
            )
            < np.float32(0.1)
        ):
            self.has_reached_cruising_speed = True

    def _advance_backward_balls(self) -> None:
        self.first_ball_moved_backwards = False
        if not self.balls:
            return
        if self.backward_count > 0:
            front = self.balls[-1]
            front.backwards_speed = np.float32(self.config.reverse_speed)
            front.backwards_count = 1

        moving = False
        speed = np.float32(0.0)
        for index in range(len(self.balls) - 1, -1, -1):
            ball = self.balls[index]
            if ball.backwards_count > 0:
                speed = np.float32(ball.backwards_speed)
                ball.waypoint = _f32_sub(ball.waypoint, speed)
                ball.backwards_count -= 1
                moving = True
            if index == 0:
                break
            rear = self.balls[index - 1]
            if not moving:
                continue
            if rear.contact_next:
                rear.waypoint = _f32_sub(rear.waypoint, speed)
                continue
            target = _f32_sub(
                _f32_sub(ball.waypoint, np.float32(ball.radius)),
                np.float32(rear.radius),
            )
            if rear.waypoint > target:
                speed = _f32_sub(rear.waypoint, target)
                rear.waypoint = np.float32(target)
                rear.contact_next = True
                moving = True
            else:
                moving = False
        if moving:
            self.first_ball_moved_backwards = True
            self.stop_time = max(self.stop_time, 20)

    def _remove_ball_at(self, index: int) -> None:
        ball = self.balls[index]
        previous = self.balls[index - 1] if index > 0 else None
        next_ball = self.balls[index + 1] if index + 1 < len(self.balls) else None
        if (
            next_ball is not None
            and not next_ball.exploding
            and previous is not None
            and not previous.should_remove
            and next_ball.color == previous.color
        ):
            next_ball.suck_count = 10
            # Revenge explicitly selects the backward-suck branch here.  The
            # ancestor has no direction flag, so relying on the dataclass
            # default would lose this transition after state restoration.
            next_ball.suck_back = True
            next_ball.combo_count = ball.combo_count + 1
            next_ball.combo_score = ball.combo_score
        if index == 0:
            self.advance_speed = np.float32(0.0)
            self.stop_time = max(self.stop_time, 40)
        if previous is not None:
            previous.contact_next = False
        self.balls.pop(index)
        self.last_events.balls_removed += 1

    def _update_sets(self) -> None:
        self._have_sets = False
        index = 0
        while index < len(self.balls):
            ball = self.balls[index]
            if ball.exploding:
                self._have_sets = True
            if ball.should_remove:
                self._remove_ball_at(index)
                continue
            if ball.exploding:
                if ball.update_count % 2 == 0:
                    ball.explode_frame += 1
                if ball.explode_frame >= 20:
                    ball.should_remove = True
            index += 1

    def _remove_balls_at_end(self) -> None:
        if bool(getattr(self.curve, "die_at_end", True)):
            return
        while (
            self.balls
            and self.balls[-1].waypoint
            >= np.float32(self.curve.end_waypoint)
        ):
            ball = self.balls.pop()
            self.merging_projectiles = [
                projectile
                for projectile in self.merging_projectiles
                if projectile.hit_ball_id != ball.id
            ]
            self.last_events.balls_removed += 1

    def _remove_front_for_rollout(self) -> None:
        # The entrance cutoff only recycles balls after cruising speed.  It is
        # relevant when a rollback passes the hidden entrance.
        if not self.has_reached_cruising_speed:
            return
        cutoff = self.entrance_cutoff
        while self.balls:
            waypoint = self.balls[0].waypoint
            if (
                (waypoint >= cutoff or not self.stop_adding)
                and waypoint >= 1.0
            ):
                break
            ball = self.balls.pop(0)
            # CurveMgr.RemoveBallsAtFront calls DeleteBullet(ball.GetBullet())
            # before recycling/deleting the entrance-side ball.
            self.merging_projectiles = [
                projectile
                for projectile in self.merging_projectiles
                if projectile.hit_ball_id != ball.id
            ]
            if not ball.exploding and not self.stop_adding:
                self.pending_colors.append(ball.color)
            elif self.stop_adding:
                self.score += 10
                self.last_events.score_delta += 10

    def _reset_shooter_for_skull_entry(self) -> None:
        """Rebuild the two-ball chamber at the retail skull-entry boundary."""

        self.current_color = None
        self.next_color = None
        self.gun_state = GunState.NORMAL
        self.gun_state_percent = np.float32(1.0)
        self.hop_ticks_remaining = 0
        self.hop_target_index = None
        self._ensure_gun_loaded()

    def _arm_skull_entry(self) -> None:
        """Enter the one-tick retail transition before loss suction starts."""

        self.skull_entry_pending = True
        # The PC Board timer resets on the transition frame, one update before
        # its loss flag flips and the endpoint ball is removed.
        self.native_game_time = 0
        self._reset_shooter_for_skull_entry()

    def _remove_loss_front_balls(self) -> int:
        """Delete the marked skull-side suffix without normal gap side effects."""

        end_waypoint = np.float32(self.curve.end_waypoint)
        removed = 0
        while (
            self.balls
            and self.balls[-1].suck_count == 0
            and self.balls[-1].waypoint >= end_waypoint
        ):
            ball = self.balls.pop()
            self.merging_projectiles = [
                projectile
                for projectile in self.merging_projectiles
                if projectile.hit_ball_id != ball.id
            ]
            removed += 1
        self.last_events.balls_removed += removed
        return removed

    def _start_loss_suction(self) -> None:
        """Flip the board into its irreversible, input-locked loss state."""

        self.skull_entry_pending = False
        self.loss_started = True
        self.loss_elapsed_ticks = 1
        self.last_events.loss_started = True
        self._remove_loss_front_balls()
        initial_count = self.config.loss_initial_suck_count
        for ball in self.balls:
            ball.suck_count = initial_count
            ball.suck_back = False
            ball.suck_pending = False
            ball.backwards_count = 0

    def _tick_loss_suction(self) -> None:
        """Advance one frozen-chain skull-suction update from the PC oracle."""

        self.loss_elapsed_ticks += 1
        self._tick_active_loss_suction()

    def _tick_active_loss_suction(self) -> None:
        self._remove_loss_front_balls()
        end_waypoint = np.float32(self.curve.end_waypoint)
        shift = self.config.loss_suction_speed_shift
        for ball in self.balls:
            speed = ball.suck_count >> shift
            ball.waypoint = _f32_add(
                ball.waypoint,
                np.float32(speed),
            )
            ball.suck_count = (
                0
                if ball.waypoint >= end_waypoint
                else ball.suck_count + 1
            )

    def _start_all_curve_loss_suction(self) -> None:
        self.skull_entry_pending = False
        self.loss_started = True
        self.loss_elapsed_ticks = 1
        self.last_events.loss_started = True
        original_curve = self.active_curve_index
        try:
            for curve_index in range(self.curve_count):
                self._activate_curve(curve_index)
                self._remove_loss_front_balls()
                initial_count = self.config.loss_initial_suck_count
                for ball in self.balls:
                    ball.suck_count = initial_count
                    ball.suck_back = False
                    ball.suck_pending = False
                    ball.backwards_count = 0
        finally:
            self._activate_curve(original_curve)

    def _tick_all_curve_loss_suction(self) -> None:
        self.loss_elapsed_ticks += 1
        original_curve = self.active_curve_index
        try:
            for curve_index in range(self.curve_count):
                self._activate_curve(curve_index)
                self._tick_active_loss_suction()
        finally:
            self._activate_curve(original_curve)

    def _tick_curve(self) -> None:
        if self.skull_entry_pending:
            self._start_loss_suction()
            return
        if self.loss_started:
            self._tick_loss_suction()
            return

        front_below_cutoff = (
            not self.balls
            or int(self.balls[-1].waypoint) < self.entrance_cutoff
        )
        if self.stop_time > 0:
            self.stop_time -= 1
            if front_below_cutoff:
                self.stop_time = 0
            if self.stop_time == 0:
                self.advance_speed = np.float32(0.0)
        if self.slow_count > 0:
            self.slow_count -= 1
        if self.backward_count > 0:
            self.backward_count -= 1

        self._add_ball()
        self._update_ball_objects()
        self._advance_merging_projectiles()
        self._update_sucking_balls()
        self._advance_balls()
        self._advance_backward_balls()
        self._remove_front_for_rollout()
        self._remove_balls_at_end()
        self._update_sets()
        self._maybe_spawn_powerup()

    def _tick_all_curves(self) -> None:
        if self.curve_count == 1:
            self._activate_curve(0)
            self._tick_curve()
            return
        if self.skull_entry_pending:
            self._start_all_curve_loss_suction()
            return
        if self.loss_started:
            self._tick_all_curve_loss_suction()
            return

        original_curve = self.active_curve_index
        try:
            # Retail CurveMgr iterates its pointer array from slot zero upward.
            for curve_index in range(self.curve_count):
                self._activate_curve(curve_index)
                self._tick_curve()
        finally:
            self._activate_curve(original_curve)

    def _update_zuma_bar(self) -> None:
        # Original UpdateUI increments current first, tests full-bar, then
        # computes target from this tick's score.  Preserve that one-tick phase.
        if self.current_bar_size < self.target_bar_size:
            self.current_bar_size += 1
        if (
            not self.zuma_reached
            and self.current_bar_size == self.config.zuma_bar_width
            and self.score_achieved
        ):
            self.zuma_reached = True
            original_curve = self.active_curve_index
            try:
                for curve_index in range(self.curve_count):
                    self._activate_curve(curve_index)
                    self.stop_adding = True
                    self.pending_colors.clear()
                    self.backward_count = int(
                        getattr(
                            self.parameters,
                            "zuma_back_distance",
                            300,
                        ),
                    )
                    self.slow_count = int(
                        getattr(
                            self.parameters,
                            "zuma_slow_duration",
                            1100,
                        ),
                    )
            finally:
                self._activate_curve(original_curve)
            self.last_events.zuma_triggered = True

        denominator = self.score_target - self.score_at_level_start
        if denominator > 0:
            remaining = max(0, self.score_target - self.score)
            self.target_bar_size = (
                self.config.zuma_bar_width
                - self.config.zuma_bar_width * remaining // denominator
            )

    def _front_segment_is_sucking(self) -> bool:
        for index in range(len(self.balls) - 1, -1, -1):
            ball = self.balls[index]
            if ball.suck_count > 0:
                return True
            if index == 0 or not self.balls[index - 1].contact_next:
                break
        return False

    def _terminal_empty(self) -> bool:
        self._store_active_curve()
        return (
            self.gun_state is not GunState.FIRING
            and not self.free_projectiles
            and all(
                not state.merging_projectiles
                and not state.balls
                and not state.pending_colors
                for state in self._curve_states
            )
        )

    def _check_outcome(self, *, terminal_empty_at_tick_start: bool) -> None:
        if self.loss_started:
            self._store_active_curve()
            if all(not state.balls for state in self._curve_states):
                self.outcome = "loss"
                self.last_events.outcome = self.outcome
            return
        if self.skull_entry_pending:
            return
        if self.win_pending:
            self.win_pending = False
            self.outcome = "win"
            self.score += VICTORY_TRANSITION_AWARD_POINTS
            self.last_events.score_delta += (
                VICTORY_TRANSITION_AWARD_POINTS
            )
            self.last_events.outcome = self.outcome
            return
        if self._terminal_empty():
            # Retail destroys both chamber Bullet objects on the first empty
            # Board update, before the following update reports formal win.
            self.current_color = None
            self.next_color = None
            if terminal_empty_at_tick_start:
                self.outcome = "win"
            else:
                # Retail keeps its Board runtime flag set for the update that
                # removes the final curve objects, rejects input in that
                # state, and flips to the formal victory state one tick later.
                self.win_pending = True
                self.last_events.win_pending = True
        else:
            self._store_active_curve()
        if self.outcome is not None or self.win_pending:
            if self.outcome is not None:
                self.last_events.outcome = self.outcome
            return
        if (
            self.gun_state is GunState.FIRING
            or self.free_projectiles
            or any(
                state.merging_projectiles for state in self._curve_states
            )
        ):
            return

        original_curve = self.active_curve_index
        try:
            for curve_index in range(self.curve_count):
                self._activate_curve(curve_index)
                if not (
                    self.balls
                    and bool(getattr(self.curve, "die_at_end", True))
                    and self.balls[-1].waypoint >= self.curve.end_waypoint
                    and not self._have_sets
                    and self.backward_count == 0
                    and not self._front_segment_is_sucking()
                ):
                    continue
                self.loss_curve_index = curve_index
                self._arm_skull_entry()
                break
        finally:
            self._activate_curve(original_curve)
        if self.outcome is not None:
            self.last_events.outcome = self.outcome

    def tick(self, ticks: int = 1) -> TickEvents:
        """Advance one or more 10 ms game ticks and return aggregate events."""

        if ticks < 1:
            raise ValueError("ticks must be at least one")
        aggregate = TickEvents()
        for _ in range(ticks):
            if self.outcome is not None:
                break
            self.last_events = TickEvents()
            self.tick_count += 1
            self.native_game_time += 1
            terminal_empty_at_tick_start = self._terminal_empty()
            self._update_hop()
            self._update_gun()
            self._update_fruit_visual()
            self._update_free_projectiles()
            self._update_fruit_scheduler()
            self._tick_all_curves()
            self._check_outcome(
                terminal_empty_at_tick_start=terminal_empty_at_tick_start,
            )
            self._check_gun_colors()
            self._ensure_gun_loaded()
            self._update_zuma_bar()
            aggregate.fired += self.last_events.fired
            aggregate.hits += self.last_events.hits
            aggregate.inserted += self.last_events.inserted
            aggregate.matches += self.last_events.matches
            aggregate.balls_exploded += self.last_events.balls_exploded
            aggregate.balls_removed += self.last_events.balls_removed
            aggregate.score_delta += self.last_events.score_delta
            aggregate.powerups_spawned += (
                self.last_events.powerups_spawned
            )
            aggregate.powerups_triggered += (
                self.last_events.powerups_triggered
            )
            aggregate.fruit_chance_draws += (
                self.last_events.fruit_chance_draws
            )
            aggregate.fruits_spawned += self.last_events.fruits_spawned
            aggregate.fruits_expired += self.last_events.fruits_expired
            aggregate.fruits_collected += self.last_events.fruits_collected
            aggregate.zuma_triggered = (
                aggregate.zuma_triggered or self.last_events.zuma_triggered
            )
            aggregate.win_pending = (
                aggregate.win_pending or self.last_events.win_pending
            )
            aggregate.loss_started = (
                aggregate.loss_started or self.last_events.loss_started
            )
            aggregate.outcome = self.last_events.outcome or aggregate.outcome
        self.last_events = aggregate
        return aggregate

    def state_signature(self) -> tuple[Any, ...]:
        """Hashable deterministic signature used by replay regression tests."""

        self._activate_curve(0)
        return (
            self.profile_mode,
            (
                None
                if self.powerup_calibration is None
                else (
                    self.powerup_calibration.chance_denominator,
                    self.powerup_calibration.initial_delay_ticks,
                    self.powerup_calibration.spawn_delay_ticks,
                    self.powerup_calibration.cooldown_ticks,
                    self.powerup_calibration.unique_color,
                    self.powerup_calibration.supported_types,
                    self.powerup_calibration.provenance,
                )
            ),
            (
                None
                if self.fruit_calibration is None
                else (
                    self.fruit_calibration.frequency_ticks,
                    self.fruit_calibration.lifetime_ticks,
                    self.fruit_calibration.points,
                    self.fruit_calibration.unlock_percentages,
                    self.fruit_calibration.provenance,
                    self.fruit_calibration.fruit_type,
                    self.fruit_calibration.logical_width,
                    self.fruit_calibration.logical_height,
                    self.fruit_calibration.sheet_columns,
                    self.fruit_calibration.sheet_rows,
                    self.fruit_calibration.collection_animation_frames,
                    self.fruit_calibration.collection_animation_fps,
                )
            ),
            self.tick_count,
            self._next_id,
            self.score,
            self.score_at_level_start,
            self.current_bar_size,
            self.target_bar_size,
            self.zuma_reached,
            self.stop_adding,
            self.feed_plan_exhausted,
            self.feed_refill_armed,
            self.num_balls_created,
            self.advance_speed,
            self.current_acceleration,
            self.slow_count,
            self.backward_count,
            self.stop_time,
            self.first_ball_moved_backwards,
            self.has_reached_rollout,
            self.has_reached_cruising_speed,
            self.first_chain_end,
            self.consecutive_clears,
            self._have_sets,
            self.win_pending,
            self.skull_entry_pending,
            self.loss_started,
            self.loss_elapsed_ticks,
            self.loss_curve_index,
            self.native_game_time,
            self.powerup_last_any_spawn_time,
            tuple(self.powerup_last_spawn_times),
            tuple(self.powerup_cooldown_times),
            tuple(self.powerup_spawn_counts),
            tuple(self.powerup_field_124_by_type),
            tuple(self.active_powerup_color_counts),
            self.powerup_triggered,
            self.last_powerup_waypoint,
            self.fruit_active_point_index,
            self.fruit_expiry_time,
            self.fruit_collecting,
            self.fruit_spawn_count,
            self.fruit_collect_count,
            self.fruit_velocity,
            self.fruit_max_velocity,
            self.fruit_acceleration,
            self.fruit_vertical_offset,
            self.fruit_lower_bound,
            self.fruit_upper_bound,
            self.fruit_glow_alpha,
            self.fruit_glow_step,
            self.fruit_alpha,
            self.fruit_cell_index,
            self.fruit_collection_ticks_remaining,
            self.active_shooter_index,
            self.hop_ticks_remaining,
            self.hop_target_index,
            self.gun_state.value,
            float(self.gun_state_percent),
            self.aim_angle,
            self.current_color,
            self.next_color,
            self.rng.state,
            self.crt_rng.state,
            (
                self._color_chooser.update_count,
                self._color_chooser.selected_index,
                self._color_chooser.allowed_support,
                tuple(map(float, self._color_chooser.weights)),
                tuple(map(float, self._color_chooser.sways)),
                tuple(map(int, self._color_chooser.last_hit)),
                tuple(map(int, self._color_chooser.previous_hit)),
            ),
            tuple(self.pending_colors),
            tuple(
                (
                    ball.id,
                    ball.color,
                    ball.waypoint,
                    ball.radius,
                    ball.contact_next,
                    ball.exploding,
                    ball.explode_frame,
                    ball.should_remove,
                    ball.update_count,
                    ball.suck_count,
                    ball.suck_back,
                    ball.suck_pending,
                    ball.backwards_count,
                    ball.backwards_speed,
                    ball.combo_count,
                    ball.combo_score,
                    ball.gap_bonus,
                    ball.num_gaps,
                    ball.powerup_previous_type,
                    ball.powerup_primary_type,
                    ball.powerup_secondary_type,
                    ball.powerup_previous_ticks,
                    ball.powerup_lifetime_ticks,
                    ball.powerup_transition_ticks,
                    ball.powerup_visual_scale,
                    ball.powerup_visual_step,
                    ball.powerup_visual_index,
                )
                for ball in self.balls
            ),
            tuple(
                (
                    projectile.id,
                    projectile.color,
                    tuple(map(float, projectile.position)),
                    tuple(map(float, projectile.velocity)),
                    projectile.radius,
                    projectile.just_fired,
                    projectile.hit_ball_id,
                    projectile.hit_in_front,
                    float(projectile.hit_percent),
                    float(projectile.merge_speed),
                    (
                        None
                        if projectile.hit_position is None
                        else tuple(map(float, projectile.hit_position))
                    ),
                    (
                        None
                        if projectile.target_position is None
                        else tuple(map(float, projectile.target_position))
                    ),
                    projectile.waypoint,
                    projectile.have_set_prev_ball,
                    projectile.do_new_merge,
                    projectile.curve_point,
                    projectile.curve_index,
                    tuple(sorted(projectile.curve_points.items())),
                    tuple(projectile.gap_info),
                )
                for projectile in self.free_projectiles
            ),
            tuple(
                (
                    projectile.id,
                    projectile.color,
                    tuple(map(float, projectile.position)),
                    tuple(map(float, projectile.velocity)),
                    projectile.radius,
                    projectile.just_fired,
                    projectile.hit_ball_id,
                    projectile.hit_in_front,
                    float(projectile.hit_percent),
                    float(projectile.merge_speed),
                    (
                        None
                        if projectile.hit_position is None
                        else tuple(map(float, projectile.hit_position))
                    ),
                    (
                        None
                        if projectile.target_position is None
                        else tuple(map(float, projectile.target_position))
                    ),
                    projectile.waypoint,
                    projectile.have_set_prev_ball,
                    projectile.do_new_merge,
                    projectile.curve_point,
                    projectile.curve_index,
                    tuple(sorted(projectile.curve_points.items())),
                    tuple(projectile.gap_info),
                )
                for projectile in self.merging_projectiles
            ),
            tuple(
                (
                    curve_index,
                    state.entrance_cutoff,
                    state.stop_adding,
                    state.feed_plan_exhausted,
                    state.feed_refill_armed,
                    state.num_balls_created,
                    state.advance_speed,
                    state.current_acceleration,
                    state.slow_count,
                    state.backward_count,
                    state.stop_time,
                    state.first_ball_moved_backwards,
                    state.has_reached_rollout,
                    state.has_reached_cruising_speed,
                    state.first_chain_end,
                    state.consecutive_clears,
                    state.have_sets,
                    state.powerup_last_any_spawn_time,
                    tuple(state.powerup_last_spawn_times),
                    tuple(state.powerup_cooldown_times),
                    tuple(state.powerup_spawn_counts),
                    tuple(state.powerup_field_124_by_type),
                    tuple(state.active_powerup_color_counts),
                    state.powerup_triggered,
                    state.last_powerup_waypoint,
                    tuple(state.pending_colors),
                    tuple(
                        (
                            ball.id,
                            ball.color,
                            ball.waypoint,
                            ball.radius,
                            ball.contact_next,
                            ball.exploding,
                            ball.explode_frame,
                            ball.should_remove,
                            ball.update_count,
                            ball.suck_count,
                            ball.suck_back,
                            ball.suck_pending,
                            ball.backwards_count,
                            ball.backwards_speed,
                            ball.combo_count,
                            ball.combo_score,
                            ball.gap_bonus,
                            ball.num_gaps,
                            ball.powerup_previous_type,
                            ball.powerup_primary_type,
                            ball.powerup_secondary_type,
                            ball.powerup_previous_ticks,
                            ball.powerup_lifetime_ticks,
                            ball.powerup_transition_ticks,
                            ball.powerup_visual_scale,
                            ball.powerup_visual_step,
                            ball.powerup_visual_index,
                        )
                        for ball in state.balls
                    ),
                    tuple(
                        (
                            projectile.id,
                            projectile.color,
                            tuple(map(float, projectile.position)),
                            tuple(map(float, projectile.velocity)),
                            projectile.radius,
                            projectile.just_fired,
                            projectile.hit_ball_id,
                            projectile.hit_in_front,
                            float(projectile.hit_percent),
                            float(projectile.merge_speed),
                            (
                                None
                                if projectile.hit_position is None
                                else tuple(
                                    map(float, projectile.hit_position)
                                )
                            ),
                            (
                                None
                                if projectile.target_position is None
                                else tuple(
                                    map(float, projectile.target_position)
                                )
                            ),
                            projectile.waypoint,
                            projectile.have_set_prev_ball,
                            projectile.do_new_merge,
                            projectile.curve_point,
                            projectile.curve_index,
                            tuple(sorted(projectile.curve_points.items())),
                            tuple(projectile.gap_info),
                        )
                        for projectile in state.merging_projectiles
                    ),
                )
                for curve_index, state in enumerate(
                    self.curve_states[1:],
                    start=1,
                )
            ),
            self.outcome,
        )


__all__ = [
    "ChainBall",
    "CurveRuntimeState",
    "GunState",
    "MsvcCRTRandom",
    "PopCapMTRandom",
    "PowerupSpawnCalibration",
    "PowerupType",
    "Projectile",
    "RevengePhysicsConfig",
    "RevengeSimulator",
    "SUPPORTED_PROFILE_MODE",
    "TickEvents",
]
