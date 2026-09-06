"""Read-only adapters for a legally installed copy of Zuma's Revenge.

The project does not redistribute original game data.  These helpers locate a
local installation and derive level geometry/configuration at runtime.
"""

from __future__ import annotations

import hashlib
import os
import re
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable, Mapping

import numpy as np
from numpy.typing import NDArray

Float32Array = NDArray[np.float32]
Float64Array = NDArray[np.float64]
IntArray = NDArray[np.int8]
ByteArray = NDArray[np.uint8]
BoolArray = NDArray[np.bool_]

_PAK_MAGIC = 0xBAC04AC0
_PAK_XOR = 0xF7
_CURVE_MAGIC = b"CURV"
_CURVE_VERSION = 15
_CURVE_MIN_SUPPORTED_VERSION = 12
_PAM_MAGIC = 0xBAF01954
_PAM_RETAIL_VERSION = 5
_F32_ZERO = np.float32(0.0)
_F32_ONE = np.float32(1.0)
_F32_FIVE = np.float32(5.0)
_INV_SUBPIXEL_MULT = np.float32(1.0) / np.float32(100.0)


class OriginalDataError(RuntimeError):
    """Raised when installed original data is missing or malformed."""


def _sha256_bytes(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


@dataclass(frozen=True, slots=True)
class PakEntry:
    name: str
    size: int
    windows_filetime: int
    offset: int


class PopCapPakArchive:
    """Minimal streaming reader for the XOR-obfuscated PopCap PAK format."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.entries = self._read_catalog()
        self._by_name = {
            self._normalize_name(entry.name): entry for entry in self.entries
        }

    @staticmethod
    def _normalize_name(name: str) -> str:
        return name.replace("/", "\\").casefold()

    @staticmethod
    def _decode(raw: bytes) -> bytes:
        return bytes(value ^ _PAK_XOR for value in raw)

    @classmethod
    def _read_decoded(cls, stream: BinaryIO, count: int) -> bytes:
        raw = stream.read(count)
        if len(raw) != count:
            raise OriginalDataError("unexpected end of PopCap PAK")
        return cls._decode(raw)

    def _read_catalog(self) -> tuple[PakEntry, ...]:
        if not self.path.is_file():
            raise OriginalDataError(f"PAK archive not found: {self.path}")

        records: list[tuple[str, int, int]] = []
        with self.path.open("rb") as stream:
            header = self._read_decoded(stream, 8)
            magic, version = struct.unpack("<II", header)
            if magic != _PAK_MAGIC or version != 0:
                raise OriginalDataError(
                    f"unsupported PopCap PAK header: {magic:#x}, version {version}"
                )

            while True:
                marker = self._read_decoded(stream, 1)[0]
                if marker != 0:
                    break
                name_size = self._read_decoded(stream, 1)[0]
                name = self._read_decoded(stream, name_size).decode("latin-1")
                size = struct.unpack("<I", self._read_decoded(stream, 4))[0]
                timestamp = struct.unpack("<Q", self._read_decoded(stream, 8))[0]
                records.append((name, size, timestamp))
            payload_offset = stream.tell()

        entries: list[PakEntry] = []
        offset = payload_offset
        for name, size, timestamp in records:
            entries.append(
                PakEntry(
                    name=name,
                    size=size,
                    windows_filetime=timestamp,
                    offset=offset,
                )
            )
            offset += size

        archive_size = self.path.stat().st_size
        if offset != archive_size:
            raise OriginalDataError(
                f"PAK catalog describes {offset} bytes, archive has {archive_size}"
            )
        return tuple(entries)

    def has_member(self, name: str) -> bool:
        return self._normalize_name(name) in self._by_name

    def read_member(self, name: str) -> bytes:
        normalized = self._normalize_name(name)
        try:
            entry = self._by_name[normalized]
        except KeyError as error:
            raise OriginalDataError(f"PAK member not found: {name}") from error
        with self.path.open("rb") as stream:
            stream.seek(entry.offset)
            return self._read_decoded(stream, entry.size)


@dataclass(frozen=True, slots=True)
class CurveParameters:
    """Gameplay values serialized in a Revenge ``CURV`` header."""

    start_distance_percent: int
    num_balls: int
    ball_repeat_chance: int
    max_single: int
    colors: int
    speed: float
    slow_distance: int
    acceleration_rate: float
    max_speed: float
    zuma_score: int
    skull_rotation_degrees: int
    zuma_back_distance: int
    zuma_slow_duration: int
    slow_factor: float
    max_clump_size: int
    powerup_records: tuple[tuple[int, int], ...]
    powerup_chance: int


@dataclass(frozen=True, slots=True)
class OriginalCurve:
    source_path: Path | None
    version: int
    linear: bool
    parameters: CurveParameters
    draw_curve: bool
    draw_tunnels: bool
    destroy_all: bool
    draw_pit: bool
    die_at_end: bool
    edit_type: int
    # CurveData.PathPoint and WayPoint store coordinates as C#/C++ ``float``.
    # Keeping these samples in float32 is part of the runtime data semantics.
    points: Float32Array
    point_flags: ByteArray
    in_tunnel: BoolArray
    priorities: ByteArray
    absolute_anchors: BoolArray
    # This is an audit-only derived metric, not a value maintained by the game.
    cumulative_distance: Float64Array

    @property
    def total_length(self) -> float:
        """Geometric polyline length, including discontinuous jumps.

        The game does not use this value to advance balls.  It uses waypoint
        indices, whose ordinary samples are approximately one logical pixel
        apart.  The geometric value remains useful for auditing decoded data.
        """

        return float(self.cumulative_distance[-1])

    @property
    def end_waypoint(self) -> int:
        """Return the last valid waypoint index, as ``WayPointMgr`` does."""

        return len(self.points) - 1

    @property
    def normalized_points(self) -> Float32Array:
        """Float32 coordinates normalized from the 800×600 logical canvas."""

        return self.points / np.array([800.0, 600.0], dtype=np.float32)

    @staticmethod
    def _waypoint_integers(
        waypoint: float | Iterable[float],
    ) -> tuple[Float32Array, NDArray[np.int64]]:
        """Round to the float API boundary, then truncate as an ``int`` cast."""

        values = np.asarray(waypoint, dtype=np.float32)
        if np.any(~np.isfinite(values)):
            raise ValueError("waypoint values must be finite")
        indices = np.asarray(np.trunc(values), dtype=np.int64)
        return values, indices

    def point_at_waypoint(
        self,
        waypoint: float | Iterable[float],
        *,
        loop_at_end: bool = False,
    ) -> Float32Array:
        """Reproduce ``WayPointMgr.SetWayPoint`` position lookup.

        Fractional indices interpolate adjacent samples.  A coordinate jump
        greater than five pixels on either axis is a path discontinuity, so
        the game snaps to the lower-index sample instead of interpolating.
        Negative indices extrapolate by their fractional component from the
        first segment, while indices beyond the end clamp unless looping is
        requested.  Input conversion, subtraction, multiplication, and
        addition explicitly use float32, matching the game's ``float`` API
        and fields rather than doing the expression in Python float64.
        """

        values, raw_indices = self._waypoint_integers(waypoint)
        count = len(self.points)
        negative = raw_indices < 0
        indices = np.where(negative, 0, raw_indices)
        next_indices = np.where(negative, 1, raw_indices + 1)

        past_end = raw_indices >= count
        if loop_at_end:
            indices = np.where(past_end, raw_indices % count, indices)
            next_indices = np.where(
                past_end,
                (raw_indices + 1) % count,
                next_indices,
            )
        else:
            indices = np.where(past_end, self.end_waypoint, indices)
            next_indices = np.where(
                past_end,
                self.end_waypoint,
                next_indices,
            )

        next_indices = np.minimum(next_indices, self.end_waypoint)
        start = self.points[indices]
        end = self.points[next_indices]
        delta = np.subtract(end, start, dtype=np.float32)
        discontinuous = np.any(np.abs(delta) > _F32_FIVE, axis=-1)
        fraction = np.subtract(
            values,
            np.trunc(values),
            dtype=np.float32,
        )
        scaled_delta = np.multiply(
            fraction[..., np.newaxis],
            delta,
            dtype=np.float32,
        )
        interpolated = np.add(scaled_delta, start, dtype=np.float32)
        return np.where(discontinuous[..., np.newaxis], start, interpolated)

    def perpendicular_at_waypoint(
        self,
        waypoint: float | Iterable[float],
    ) -> Float32Array:
        """Return the float32 normalized perpendicular used by ``WayPointMgr``."""

        _, raw_indices = self._waypoint_integers(waypoint)
        indices = np.clip(raw_indices, 0, self.end_waypoint)
        flat_indices = np.ravel(indices)
        result = np.empty((flat_indices.size, 2), dtype=np.float32)

        for output_index, point_index_value in enumerate(flat_indices):
            point_index = int(point_index_value)
            point = self.points[point_index]
            use_previous_orientation = False
            if point_index + 1 < len(self.points):
                other = self.points[point_index + 1]
                if (
                    np.any(
                        np.abs(
                            np.subtract(point, other, dtype=np.float32),
                        )
                        > _F32_FIVE
                    )
                    and point_index > 0
                ):
                    use_previous_orientation = True
                    other = self.points[point_index - 1]
            else:
                other = self.points[point_index - 1]
                use_previous_orientation = True

            if use_previous_orientation:
                perpendicular = np.array(
                    (point[1] - other[1], other[0] - point[0]),
                    dtype=np.float32,
                )
            else:
                perpendicular = np.array(
                    (other[1] - point[1], point[0] - other[0]),
                    dtype=np.float32,
                )
            squared = np.add(
                np.multiply(
                    perpendicular[0],
                    perpendicular[0],
                    dtype=np.float32,
                ),
                np.multiply(
                    perpendicular[1],
                    perpendicular[1],
                    dtype=np.float32,
                ),
                dtype=np.float32,
            )
            length = np.float32(np.sqrt(squared))
            inverse_length = np.divide(
                _F32_ONE,
                length,
                dtype=np.float32,
            )
            result[output_index] = np.multiply(
                perpendicular,
                inverse_length,
                dtype=np.float32,
            )

        return result.reshape(indices.shape + (2,))

    def priority_at_waypoint(
        self,
        waypoint: float | Iterable[float],
    ) -> ByteArray:
        """Return path draw priority, with out-of-range indices yielding zero."""

        _, indices = self._waypoint_integers(waypoint)
        valid = (indices >= 0) & (indices <= self.end_waypoint)
        safe_indices = np.clip(indices, 0, self.end_waypoint)
        return np.where(valid, self.priorities[safe_indices], 0).astype(
            np.uint8,
        )

    def is_in_tunnel_at_waypoint(
        self,
        waypoint: float | Iterable[float],
    ) -> BoolArray:
        """Return tunnel state with the game's asymmetric bounds behavior.

        Waypoints before the path are considered inside a tunnel; waypoints
        beyond the last valid index are not.
        """

        _, indices = self._waypoint_integers(waypoint)
        valid = (indices >= 0) & (indices <= self.end_waypoint)
        safe_indices = np.clip(indices, 0, self.end_waypoint)
        return np.where(
            valid,
            self.in_tunnel[safe_indices],
            indices < 0,
        ).astype(np.bool_)

    def point_at_distance(
        self,
        distance: float | Iterable[float],
    ) -> Float64Array:
        """Interpolate by audit-only float64 geometric polyline distance."""

        values = np.asarray(distance, dtype=np.float64)
        clipped = np.clip(values, 0.0, self.total_length)
        x = np.interp(clipped, self.cumulative_distance, self.points[:, 0])
        y = np.interp(clipped, self.cumulative_distance, self.points[:, 1])
        return np.stack((x, y), axis=-1)

    def _indices_at_distance(
        self,
        distance: float | Iterable[float],
    ) -> NDArray[np.intp]:
        values = np.asarray(distance, dtype=np.float64)
        indices = np.searchsorted(
            self.cumulative_distance,
            np.clip(values, 0.0, self.total_length),
            side="right",
        ) - 1
        return np.clip(indices, 0, len(self.points) - 1)

    def priority_at_distance(
        self,
        distance: float | Iterable[float],
    ) -> ByteArray:
        """Return the original draw/collision priority for a path distance."""

        return self.priorities[self._indices_at_distance(distance)]

    def is_in_tunnel_at_distance(
        self,
        distance: float | Iterable[float],
    ) -> BoolArray:
        """Return the original tunnel bit for a path distance."""

        return self.in_tunnel[self._indices_at_distance(distance)]


def parse_original_curve(
    data: bytes,
    *,
    source_path: Path | None = None,
) -> OriginalCurve:
    """Parse a Revenge ``CURV`` file using the game's serialized layout."""

    if len(data) < 96 or data[:4] != _CURVE_MAGIC:
        raise OriginalDataError("not a Zuma's Revenge CURV file")
    version = struct.unpack_from("<I", data, 4)[0]
    if not _CURVE_MIN_SUPPORTED_VERSION <= version <= _CURVE_VERSION:
        raise OriginalDataError(
            "supported CURV versions are "
            f"{_CURVE_MIN_SUPPORTED_VERSION}-{_CURVE_VERSION}, got {version}"
        )

    cursor = 8

    def require(count: int) -> None:
        if cursor + count > len(data):
            raise OriginalDataError("unexpected end of CURV file")

    def boolean() -> bool:
        nonlocal cursor
        require(1)
        value = data[cursor] != 0
        cursor += 1
        return value

    def unsigned() -> int:
        nonlocal cursor
        require(4)
        value = struct.unpack_from("<I", data, cursor)[0]
        cursor += 4
        return int(value)

    def floating() -> float:
        nonlocal cursor
        require(4)
        value = struct.unpack_from("<f", data, cursor)[0]
        cursor += 4
        return float(value)

    linear = boolean()
    start_distance = unsigned()
    num_balls = unsigned()
    ball_repeat = unsigned()
    max_single = unsigned()
    colors = unsigned()
    speed = floating()
    slow_distance = unsigned()
    acceleration_rate = floating()
    max_speed = floating()
    score_target = unsigned()
    skull_rotation = unsigned()
    zuma_back = unsigned()
    zuma_slow = unsigned()
    slow_factor = floating() if version >= 13 else 4.0
    max_clump_size = unsigned() if version >= 14 else 10
    powerup_count = unsigned()
    if powerup_count > 14:
        raise OriginalDataError(
            f"CURV contains an invalid power-up count: {powerup_count}"
        )
    powerup_records = tuple(
        (unsigned(), unsigned() if version >= 12 else 100_000_000)
        for _ in range(powerup_count)
    )
    powerup_chance = unsigned() if version >= 12 else 0
    parameters = CurveParameters(
        start_distance_percent=start_distance,
        num_balls=num_balls,
        ball_repeat_chance=ball_repeat,
        max_single=max_single,
        colors=colors,
        speed=speed,
        slow_distance=slow_distance,
        acceleration_rate=acceleration_rate,
        max_speed=max_speed,
        zuma_score=score_target,
        skull_rotation_degrees=skull_rotation,
        zuma_back_distance=zuma_back,
        zuma_slow_duration=zuma_slow,
        slow_factor=slow_factor,
        max_clump_size=max_clump_size,
        powerup_records=powerup_records,
        powerup_chance=powerup_chance,
    )

    draw_curve = boolean()
    draw_tunnels = boolean()
    destroy_all = boolean()
    draw_pit = boolean()
    die_at_end = boolean()
    no_edit_payload = boolean()
    has_priorities = boolean()
    edit_type = 0
    if not no_edit_payload:
        edit_type = unsigned()
        payload_size = unsigned()
        require(payload_size)
        cursor += payload_size

    count = unsigned()
    if not 2 <= count <= 1_000_000:
        raise OriginalDataError(f"CURV contains an invalid point count: {count}")

    points = np.empty((count, 2), dtype=np.float32)
    point_flags = np.empty(count, dtype=np.uint8)
    in_tunnel = np.empty(count, dtype=np.bool_)
    priorities = np.zeros(count, dtype=np.uint8)
    absolute_anchors = np.empty(count, dtype=np.bool_)
    x = _F32_ZERO
    y = _F32_ZERO

    for index in range(count):
        require(1)
        flags = data[cursor]
        cursor += 1
        point_flags[index] = flags
        in_tunnel[index] = bool(flags & 1)
        absolute = bool(flags & 2)
        absolute_anchors[index] = absolute
        if has_priorities or version >= 15:
            require(1)
            priorities[index] = data[cursor]
            cursor += 1
        if absolute:
            require(8)
            absolute_x, absolute_y = struct.unpack_from("<ff", data, cursor)
            x = np.float32(absolute_x)
            y = np.float32(absolute_y)
            cursor += 8
        else:
            require(2)
            delta_x, delta_y = struct.unpack_from("<bb", data, cursor)
            cursor += 2
            # CurveData performs both the subpixel multiply and every write
            # back to its running coordinate fields as ``float``.  Rounding
            # only once after accumulating in float64 measurably drifts.
            x = np.add(
                x,
                np.multiply(
                    np.float32(delta_x),
                    _INV_SUBPIXEL_MULT,
                    dtype=np.float32,
                ),
                dtype=np.float32,
            )
            y = np.add(
                y,
                np.multiply(
                    np.float32(delta_y),
                    _INV_SUBPIXEL_MULT,
                    dtype=np.float32,
                ),
                dtype=np.float32,
            )
        points[index] = (x, y)

    if cursor != len(data):
        raise OriginalDataError(
            f"CURV has {len(data) - cursor} unparsed trailing bytes"
        )

    # Geometric distances are a diagnostic convenience, never runtime curve
    # state.  Derive them in float64 from the already float32-decoded samples
    # so audit accumulation is stable without changing waypoint semantics.
    audit_points = points.astype(np.float64)
    segment_lengths = np.linalg.norm(
        np.diff(audit_points, axis=0),
        axis=1,
    )
    if np.any(~np.isfinite(segment_lengths)) or np.any(segment_lengths <= 0.0):
        raise OriginalDataError("curve contains implausible sample spacing")
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    return OriginalCurve(
        source_path=source_path,
        version=version,
        linear=linear,
        parameters=parameters,
        draw_curve=draw_curve,
        draw_tunnels=draw_tunnels,
        destroy_all=destroy_all,
        draw_pit=draw_pit,
        die_at_end=die_at_end,
        edit_type=edit_type,
        points=points,
        point_flags=point_flags,
        in_tunnel=in_tunnel,
        priorities=priorities,
        absolute_anchors=absolute_anchors,
        cumulative_distance=cumulative,
    )


def load_original_curve(path: str | Path) -> OriginalCurve:
    source = Path(path)
    return parse_original_curve(source.read_bytes(), source_path=source)


@dataclass(frozen=True, slots=True)
class GunDefinition:
    type: str
    positions: tuple[tuple[float, float], ...]
    attributes: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class TunnelDefinition:
    layer: int
    priority: int
    hidden_from_thumbnail: bool


@dataclass(frozen=True, slots=True)
class ZoneDefinition:
    number: int
    start_level: str
    boss_level: str
    fruit_type: str
    attributes: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class FruitAssetMetadata:
    """Gameplay-relevant fruit dimensions and collection animation data."""

    fruit_type: str
    image_resource_id: str
    image_path: str
    sheet_columns: int
    sheet_rows: int
    logical_width: int
    logical_height: int
    collection_animation_frames: int
    collection_animation_fps: float
    high_resolution_gif_sha256: str
    low_resolution_gif_sha256: str
    collection_animation_sha256: str
    provenance: str


@dataclass(frozen=True, slots=True)
class LevelDefinition:
    id: str
    display_name: str
    curve_names: tuple[str, ...]
    attributes: Mapping[str, str]
    gun: GunDefinition
    tunnels: tuple[TunnelDefinition, ...]
    treasure_points: tuple[tuple[float, float], ...]
    # Point-major unlock percentages.  Each inner tuple has one entry per
    # curve (``dist1``, ``dist2``, ...); a missing XML attribute is represented
    # by zero, matching the retail selector's ``threshold <= 0`` skip.
    treasure_point_distances: tuple[tuple[int, ...], ...] = ()
    zone_number: int = 0
    fruit_type: str = ""


@dataclass(frozen=True, slots=True)
class LoadedOriginalLevel:
    definition: LevelDefinition
    curves: tuple[OriginalCurve, ...]
    hard: bool


class _PamCursor:
    """Bounds-checked reader for the retail version-5 PopAnim subset."""

    def __init__(self, data: bytes):
        self.data = data
        self.position = 0

    def take(self, count: int) -> bytes:
        if count < 0 or self.position + count > len(self.data):
            raise OriginalDataError("unexpected end of fruit PAM")
        value = self.data[self.position : self.position + count]
        self.position += count
        return value

    def u8(self) -> int:
        return self.take(1)[0]

    def i16(self) -> int:
        return int(struct.unpack("<h", self.take(2))[0])

    def u16(self) -> int:
        return int(struct.unpack("<H", self.take(2))[0])

    def i32(self) -> int:
        return int(struct.unpack("<i", self.take(4))[0])

    def u32(self) -> int:
        return int(struct.unpack("<I", self.take(4))[0])

    def string16(self) -> str:
        count = self.i16()
        if count < 0:
            raise OriginalDataError("fruit PAM contains a negative string size")
        try:
            return self.take(count).decode("utf-8")
        except UnicodeDecodeError as error:
            raise OriginalDataError("fruit PAM contains invalid text") from error


def _pam_variable_count(cursor: _PamCursor) -> int:
    count = cursor.u8()
    if count == 255:
        count = cursor.i16()
    if count < 0:
        raise OriginalDataError("fruit PAM contains a negative item count")
    return count


def _skip_pam_frame(cursor: _PamCursor, version: int) -> bool:
    flags = cursor.u8()
    if flags & ~0x3F:
        raise OriginalDataError("fruit PAM contains unknown frame flags")

    if flags & 0x01:  # removes
        for _ in range(_pam_variable_count(cursor)):
            index = cursor.i16()
            if index >= 2047:
                cursor.i32()

    if flags & 0x02:  # adds
        for _ in range(_pam_variable_count(cursor)):
            add_flags = cursor.u16()
            if add_flags & 2047 == 2047:
                cursor.i32()
            resource = cursor.u8()
            if version >= 6 and resource == 255:
                cursor.i16()
            if add_flags & 8192:
                cursor.i16()
            if add_flags & 4096:
                cursor.string16()
            if add_flags & 2048:
                cursor.i32()

    if flags & 0x04:  # moves
        for _ in range(_pam_variable_count(cursor)):
            move_flags = cursor.u16()
            if move_flags & 1023 == 1023:
                cursor.i32()
            if move_flags & 4096:
                cursor.take(16)
            elif move_flags & 16384:
                cursor.take(2)
            cursor.take(8 if move_flags & 2048 else 4)
            if move_flags & 32768:
                cursor.take(8)
            if move_flags & 8192:
                cursor.take(4)
            if move_flags & 1024:
                cursor.take(2)

    if flags & 0x08:
        cursor.string16()
    if flags & 0x20:
        for _ in range(cursor.u8()):
            cursor.string16()
            cursor.string16()
    return bool(flags & 0x10)


def _read_pam_sprite(
    cursor: _PamCursor,
    version: int,
) -> tuple[str, float, int, bool]:
    name = cursor.string16() if version >= 4 else ""
    if version >= 6:
        cursor.string16()
    frame_rate = cursor.i32() / 65536.0 if version >= 4 else -1.0
    frame_count = cursor.i16()
    if not 1 <= frame_count <= 10_000:
        raise OriginalDataError("fruit PAM sprite frame count is invalid")
    if version >= 5:
        work_start = cursor.i16()
        work_end = cursor.i16()
        if not 0 <= work_start < frame_count or work_end < work_start:
            raise OriginalDataError("fruit PAM sprite work area is invalid")
    last_frame_stops = False
    for frame_index in range(frame_count):
        stops = _skip_pam_frame(cursor, version)
        if frame_index == frame_count - 1:
            last_frame_stops = stops
    return name, frame_rate, frame_count, last_frame_stops


def parse_fruit_collection_animation(data: bytes) -> tuple[int, float]:
    """Return the frame count and FPS of the exact ``Main`` PAM sprite.

    Zuma's Revenge ships these effects as version-5 PopAnim files.  The parser
    consumes every byte and rejects unsupported variants so an accidental
    resource substitution cannot silently alter collection timing.
    """

    cursor = _PamCursor(data)
    if cursor.u32() != _PAM_MAGIC:
        raise OriginalDataError("fruit collection asset is not a PopAnim PAM")
    version = cursor.i32()
    if version != _PAM_RETAIL_VERSION:
        raise OriginalDataError(
            f"fruit collection PAM must be version {_PAM_RETAIL_VERSION}"
        )
    header_frame_rate = cursor.u8()
    cursor.take(8)  # position and size, four int16 values

    image_count = cursor.i16()
    if not 0 <= image_count <= 10_000:
        raise OriginalDataError("fruit PAM image count is invalid")
    for _ in range(image_count):
        cursor.string16()
        cursor.take(4)  # image width and height
        cursor.take(20)  # four fixed-point matrix values and x/y translation

    sprite_count = cursor.i16()
    if not 0 <= sprite_count <= 10_000:
        raise OriginalDataError("fruit PAM sprite count is invalid")
    sprites = tuple(
        _read_pam_sprite(cursor, version) for _ in range(sprite_count)
    )
    has_main_sprite = cursor.u8()
    if has_main_sprite not in (0, 1):
        raise OriginalDataError("fruit PAM main-sprite flag is invalid")
    if has_main_sprite:
        _read_pam_sprite(cursor, version)
    if cursor.position != len(data):
        raise OriginalDataError("fruit PAM contains unparsed trailing bytes")

    matches = [row for row in sprites if row[0].casefold() == "main"]
    if len(matches) != 1:
        raise OriginalDataError("fruit PAM needs exactly one Main sprite")
    _, sprite_frame_rate, frame_count, last_frame_stops = matches[0]
    if (
        sprite_frame_rate <= 0.0
        or sprite_frame_rate != float(header_frame_rate)
        or not last_frame_stops
    ):
        raise OriginalDataError("fruit PAM Main timing or stop frame is invalid")
    return frame_count, sprite_frame_rate


def _gif_dimensions(data: bytes, *, member: str) -> tuple[int, int]:
    if len(data) < 10 or data[:6] not in {b"GIF87a", b"GIF89a"}:
        raise OriginalDataError(f"fruit image is not a GIF: {member}")
    width, height = struct.unpack_from("<HH", data, 6)
    if width < 1 or height < 1:
        raise OriginalDataError(f"fruit GIF dimensions are invalid: {member}")
    return int(width), int(height)


def _windows_path_to_wsl(value: str) -> Path:
    normalized = value.replace("\\\\", "\\")
    match = re.fullmatch(r"([A-Za-z]):\\(.*)", normalized)
    if match and Path("/mnt").is_dir():
        drive = match.group(1).lower()
        tail = match.group(2).replace("\\", "/")
        return Path("/mnt") / drive / tail
    return Path(normalized)


def _steam_library_candidates() -> list[Path]:
    candidates: list[Path] = []
    library_files = [
        Path("/mnt/c/Program Files (x86)/Steam/steamapps/libraryfolders.vdf"),
        Path.home() / ".steam/steam/steamapps/libraryfolders.vdf",
        Path.home() / ".local/share/Steam/steamapps/libraryfolders.vdf",
    ]
    pattern = re.compile(r'"path"\s+"([^"]+)"', re.IGNORECASE)
    for library_file in library_files:
        if not library_file.is_file():
            continue
        text = library_file.read_text(encoding="utf-8", errors="replace")
        for match in pattern.finditer(text):
            candidates.append(_windows_path_to_wsl(match.group(1)))
        candidates.append(library_file.parent.parent)

    if Path("/mnt").is_dir():
        for drive in "cdefghijklmnopqrstuvwxyz":
            candidates.append(Path("/mnt") / drive / "SteamLibrary")
    return candidates


def find_original_installation(explicit: str | Path | None = None) -> Path:
    """Locate the user's installation without changing it."""

    roots: list[Path] = []
    if explicit is not None:
        roots.append(Path(explicit))
    environment_root = os.environ.get("ZUMA_REVENGE_ROOT")
    if environment_root:
        roots.append(Path(environment_root))

    for library in _steam_library_candidates():
        roots.append(library / "steamapps/common/Zuma's Revenge")
    roots.extend(
        (
            Path("C:/Program Files (x86)/Steam/steamapps/common/Zuma's Revenge"),
            Path("C:/Program Files/EA Games/Zuma's Revenge"),
        )
    )

    seen: set[str] = set()
    for root in roots:
        key = str(root).casefold()
        if key in seen:
            continue
        seen.add(key)
        if (
            (root / "ZumasRevenge.exe").is_file()
            and (root / "main.pak").is_file()
            and (root / "levels").is_dir()
        ):
            return root.resolve()
    raise OriginalDataError(
        "Zuma's Revenge installation was not found. Set ZUMA_REVENGE_ROOT "
        "to the directory containing ZumasRevenge.exe."
    )


class OriginalGameCatalog:
    """Level metadata and curve files sourced from the local installation."""

    def __init__(self, root: str | Path | None = None):
        self.root = find_original_installation(root)
        self.archive = PopCapPakArchive(self.root / "main.pak")
        xml_data = self.archive.read_member(r"levels\levels.xml")
        # PopCap's permissive XML reader accepts repeated ``--`` inside comments
        # and a custom ``&cr;`` line-break entity.  Standard XML deliberately
        # rejects both, so discard documentation comments and normalize the
        # sole custom entity before using ElementTree.
        xml_text = xml_data.decode("utf-8-sig")
        xml_text = re.sub(r"<!--.*?-->", "", xml_text, flags=re.DOTALL)
        xml_text = xml_text.replace("&cr;", "&#10;")
        self.xml_root = ET.fromstring(xml_text)
        self.resources_text = self.archive.read_member(
            r"properties\resources.xml"
        ).decode("utf-8-sig")
        defaults = self.xml_root.find("Defaults")
        self.defaults: Mapping[str, str] = (
            dict(defaults.attrib) if defaults is not None else {}
        )
        self.zones = self._parse_zones()
        self.levels = self._parse_levels()
        self._fruit_assets_cache: dict[str, FruitAssetMetadata] = {}
        self._curve_files = {
            str(path.relative_to(self.root / "levels"))
            .replace("\\", "/")
            .casefold(): path
            for path in (self.root / "levels").rglob("*")
            if path.is_file() and path.suffix.casefold() == ".dat"
        }
        self._curves_by_basename: dict[str, list[Path]] = {}
        for path in self._curve_files.values():
            self._curves_by_basename.setdefault(path.name.casefold(), []).append(
                path
            )

    def _parse_zones(self) -> tuple[ZoneDefinition, ...]:
        zones: list[ZoneDefinition] = []
        for element in self.xml_root.findall("Zone"):
            try:
                number = int(element.attrib["num"])
            except (KeyError, ValueError) as error:
                raise OriginalDataError("levels.xml contains an invalid Zone") from error
            fruit_type = element.attrib.get("fruit", "").strip().casefold()
            if number < 1 or not fruit_type:
                raise OriginalDataError("levels.xml contains an incomplete Zone")
            zones.append(
                ZoneDefinition(
                    number=number,
                    start_level=element.attrib.get("start", ""),
                    boss_level=element.attrib.get("boss", ""),
                    fruit_type=fruit_type,
                    attributes=dict(element.attrib),
                )
            )
        return tuple(zones)

    def _level_zone(self, level_id: str) -> ZoneDefinition | None:
        key = level_id.casefold()
        zones = getattr(self, "zones", ())
        for zone in zones:
            if zone.boss_level and key == zone.boss_level.casefold():
                return zone
            prefix = re.sub(r"\d+$", "", zone.start_level.casefold())
            if prefix and key.startswith(prefix):
                return zone
        return None

    def _parse_levels(self) -> Mapping[str, LevelDefinition]:
        levels: dict[str, LevelDefinition] = {}
        for element in self.xml_root.findall("Level"):
            level_id = element.attrib["id"]
            curve_names = tuple(
                value
                for key, value in sorted(element.attrib.items())
                if re.fullmatch(r"curve\d+", key, re.IGNORECASE)
            )
            gun_element = element.find("Gun")
            if gun_element is None:
                gun = GunDefinition(type="normal", positions=(), attributes={})
            else:
                positions: list[tuple[float, float]] = []
                index = 1
                while (
                    f"gx{index}" in gun_element.attrib
                    and f"gy{index}" in gun_element.attrib
                ):
                    positions.append(
                        (
                            float(gun_element.attrib[f"gx{index}"]),
                            float(gun_element.attrib[f"gy{index}"]),
                        )
                    )
                    index += 1
                if not positions and {
                    "startx",
                    "starty",
                } <= gun_element.attrib.keys():
                    positions.append(
                        (
                            float(gun_element.attrib["startx"]),
                            float(gun_element.attrib["starty"]),
                        )
                    )
                gun = GunDefinition(
                    type=gun_element.attrib.get("type", "normal"),
                    positions=tuple(positions),
                    attributes=dict(gun_element.attrib),
                )

            tunnels = tuple(
                TunnelDefinition(
                    layer=int(tunnel.attrib["layer"]),
                    priority=int(tunnel.attrib.get("pri", "0")),
                    hidden_from_thumbnail=tunnel.attrib.get(
                        "nothumb",
                        "false",
                    ).casefold()
                    == "true",
                )
                for tunnel in element.findall("Tunnel")
            )
            treasure_elements = tuple(element.findall("TreasurePoint"))
            treasures = tuple(
                (
                    float(point.attrib["x"]),
                    float(point.attrib["y"]),
                )
                for point in treasure_elements
            )
            treasure_distances = tuple(
                tuple(
                    int(point.attrib.get(f"dist{index}", "0"))
                    for index in range(1, len(curve_names) + 1)
                )
                for point in treasure_elements
            )
            zone = self._level_zone(level_id)
            levels[level_id.casefold()] = LevelDefinition(
                id=level_id,
                display_name=element.attrib.get("dispname", level_id),
                curve_names=curve_names,
                attributes=dict(element.attrib),
                gun=gun,
                tunnels=tunnels,
                treasure_points=treasures,
                treasure_point_distances=treasure_distances,
                zone_number=0 if zone is None else zone.number,
                fruit_type="" if zone is None else zone.fruit_type,
            )
        return levels

    def _resource_attributes(
        self,
        tag_name: str,
        resource_id: str,
    ) -> Mapping[str, str]:
        tag_pattern = re.compile(
            rf"<{re.escape(tag_name)}\b[^>]*>",
            flags=re.IGNORECASE,
        )
        for match in tag_pattern.finditer(self.resources_text):
            attributes = {
                key.casefold(): quoted if quoted != "" else unquoted
                for key, quoted, unquoted in re.findall(
                    r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"
                    r"(?:\"([^\"]*)\"|([^\s/>]+))",
                    match.group(0),
                )
            }
            if attributes.get("id", "").casefold() == resource_id.casefold():
                return attributes
        raise OriginalDataError(f"resource definition not found: {resource_id}")

    def fruit_assets(self, fruit_type: str) -> FruitAssetMetadata:
        """Load and validate gameplay-relevant fruit assets from ``main.pak``."""

        fruit = fruit_type.strip().casefold()
        if not fruit or not re.fullmatch(r"[a-z]+", fruit):
            raise OriginalDataError("fruit type is invalid")
        cached = self._fruit_assets_cache.get(fruit)
        if cached is not None:
            return cached

        image_id = f"IMAGE_FRUIT_{fruit.upper()}"
        image = self._resource_attributes("Image", image_id)
        try:
            high_path = image["path"]
            columns = int(image["cols"])
            rows = int(image["rows"])
        except (KeyError, ValueError) as error:
            raise OriginalDataError(
                f"fruit image metadata is incomplete: {image_id}"
            ) from error
        if columns < 1 or rows < 1:
            raise OriginalDataError("fruit sheet grid is invalid")

        low_path, replacements = re.subn(
            r"\\1200\\",
            r"\\600\\",
            high_path,
            count=1,
            flags=re.IGNORECASE,
        )
        if replacements != 1:
            raise OriginalDataError("fruit image path is not a 1200 resource")
        high_member = high_path + "_.gif"
        low_member = low_path + "_.gif"
        high_data = self.archive.read_member(high_member)
        low_data = self.archive.read_member(low_member)
        high_width, high_height = _gif_dimensions(
            high_data,
            member=high_member,
        )
        low_width, low_height = _gif_dimensions(low_data, member=low_member)
        if (
            high_width % columns
            or high_height % rows
            or low_width % columns
            or low_height % rows
        ):
            raise OriginalDataError("fruit sheet does not divide into its grid")
        high_cell = (high_width // columns, high_height // rows)
        low_cell = (low_width // columns, low_height // rows)
        if high_cell != (low_cell[0] * 2, low_cell[1] * 2):
            raise OriginalDataError("600/1200 fruit sheet scales disagree")

        animation_id = f"POPANIM_NONRESIZE_{fruit.upper()}MUSH"
        animation = self._resource_attributes("PopAnim", animation_id)
        try:
            animation_member = animation["path"]
        except KeyError as error:
            raise OriginalDataError(
                f"fruit animation metadata is incomplete: {animation_id}"
            ) from error
        animation_data = self.archive.read_member(animation_member)
        animation_frames, animation_fps = parse_fruit_collection_animation(
            animation_data
        )
        result = FruitAssetMetadata(
            fruit_type=fruit,
            image_resource_id=image_id,
            image_path=high_path,
            sheet_columns=columns,
            sheet_rows=rows,
            logical_width=low_cell[0],
            logical_height=low_cell[1],
            collection_animation_frames=animation_frames,
            collection_animation_fps=animation_fps,
            high_resolution_gif_sha256=_sha256_bytes(high_data),
            low_resolution_gif_sha256=_sha256_bytes(low_data),
            collection_animation_sha256=_sha256_bytes(animation_data),
            provenance=(
                f"installed main.pak resources {high_member}, {low_member}, "
                f"and {animation_member}; complete version-5 PAM parse"
            ),
        )
        self._fruit_assets_cache[fruit] = result
        return result

    def level(self, level_id: str) -> LevelDefinition:
        try:
            return self.levels[level_id.casefold()]
        except KeyError as error:
            raise OriginalDataError(f"unknown original level: {level_id}") from error

    def _curve_path(self, curve_name: str, hard: bool) -> Path:
        normalized = curve_name.replace("\\", "/").strip("/").casefold()
        suffix = "_hard.dat" if hard else ".dat"
        candidates = [normalized + suffix]
        if "/" not in normalized:
            candidates.append(f"{normalized}/{normalized}{suffix}")
        for candidate in candidates:
            if candidate in self._curve_files:
                return self._curve_files[candidate]
        basename = normalized.rsplit("/", 1)[-1] + suffix
        basename_matches = self._curves_by_basename.get(basename, [])
        if len(basename_matches) == 1:
            return basename_matches[0]
        raise OriginalDataError(
            f"curve file for {curve_name!r} (hard={hard}) was not found"
        )

    def load_level(
        self,
        level_id: str,
        *,
        hard: bool = False,
    ) -> LoadedOriginalLevel:
        definition = self.level(level_id)
        curves = tuple(
            load_original_curve(self._curve_path(name, hard))
            for name in definition.curve_names
        )
        if not curves:
            raise OriginalDataError(f"level {level_id!r} has no curve data")
        return LoadedOriginalLevel(
            definition=definition,
            curves=curves,
            hard=hard,
        )
