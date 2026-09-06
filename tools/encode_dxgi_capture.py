"""Validate a raw DXGI acquisition and encode it as lossless FFV1/MKV.

The acquisition directory is treated as immutable input.  Encoding is only
attempted after the metadata, canonical CSV, raw byte stream, offsets, and all
declared hashes have been cross-checked.  Matroska written by FFmpeg uses a
1/1000 time base, so QPC-relative presentation times are mapped to the nearest
millisecond with ties rounded upward.  The returned report records every PTS
and the mapping rule so a verifier can independently recompute the timeline.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import re
import stat
import sys
from contextlib import ExitStack
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, BinaryIO, Sequence

CAPTURE_SCHEMA = "zuma-rl.dxgi-bgra-capture"
CAPTURE_VERSION = 2
ENCODING_SCHEMA = "zuma-rl.dxgi-ffv1-encoding"
ENCODING_VERSION = 1
TIMELINE_SCHEMA = "zuma-rl.dxgi-qpc-timeline"
TIMELINE_VERSION = 1

RAW_FILE = "frames.bgra.raw"
CSV_FILE = "frames.csv"
METADATA_FILE = "metadata.json"
DIRECTORY_OUTPUT_FILE = "capture.mkv"
PIXEL_FORMAT = "bgra"
BYTES_PER_PIXEL = 4
VIDEO_CODEC = "ffv1"
VIDEO_TIME_BASE = Fraction(1, 1000)
AVERAGE_RATE_ABSOLUTE_TOLERANCE = Fraction(1, 1000)
AVERAGE_RATE_RELATIVE_TOLERANCE = Fraction(1, 10_000)
TIMESTAMP_MAPPING = "qpc_relative_nearest_millisecond_ties_up"
TIMELINE_METADATA_TAG = "ZUMA_DXGI_TIMELINE"
NOMINAL_FPS_MAX_DENOMINATOR = 1001
REQUIRED_RUNTIME_VERSIONS = {
    "comtypes": "1.4.16",
    "dxcam": "0.3.0",
    "numpy": "2.5.1",
}

MAX_METADATA_BYTES = 256 * 1024
MAX_CSV_BYTES = 4 * 1024**2
MAX_TIMELINE_TAG_BYTES = 128 * 1024
MAX_RAW_BYTES = 2 * 1024**3
MAX_VIDEO_BYTES = 4 * 1024**3
MAX_DIMENSION = 16_384
MAX_PIXELS_PER_FRAME = 33_554_432
MAX_FRAME_COUNT = 1_202
MAX_QPC_FREQUENCY = 1_000_000_000_000
MAX_REQUESTED_DURATION_SECONDS = 10.0
MAX_CAPTURE_ELAPSED_SECONDS = 10.0
READ_CHUNK_BYTES = 1024 * 1024

_PIXEL_HASH_DOMAIN = b"zuma-rl.dxgi-bgra.v1\0"
_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
_DECIMAL_PATTERN = re.compile(r"0|[1-9][0-9]*")
_WINDOW_HANDLE_PATTERN = re.compile(r"0x[0-9a-f]{16}")
_CSV_HEADER = (
    "sequence",
    "present_ticks",
    "qpc_frequency",
    "host_perf_counter_ns",
    "accumulated_frames",
    "raw_offset",
    "raw_bytes",
    "frame_sha256",
)
_METADATA_KEYS = {
    "schema",
    "version",
    "status",
    "target_process",
    "device_index",
    "output_index",
    "global_region",
    "output_local_region",
    "width",
    "height",
    "pixel_format",
    "bytes_per_pixel",
    "frame_budget_fps",
    "capture_mode",
    "capture_storage_mode",
    "capture_surface",
    "foreground_geometry_guard",
    "topmost_or_injected_overlay_detection",
    "requires_full_frame_visual_review",
    "requested_duration_seconds",
    "capture_start_perf_counter_ns",
    "capture_end_perf_counter_ns",
    "capture_elapsed_seconds",
    "estimated_raw_budget_bytes",
    "frame_budget",
    "frame_count",
    "warmup_baseline_present_ticks",
    "first_present_ticks",
    "last_present_ticks",
    "present_span_ticks",
    "present_span_seconds",
    "observed_mean_present_fps",
    "qpc_frequency",
    "missed_presentations",
    "idle_poll_count",
    "pointer_only_update_count",
    "warmup_idle_poll_count",
    "warmup_pointer_only_update_count",
    "runtime_versions",
    "runtime_versions_verified",
    "source_identity",
    "target_identity",
    "raw_bytes",
    "raw_sha256",
    "frames_csv_bytes",
    "frames_csv_rows",
    "frames_csv_sha256",
    "aggregate_pixel_sha256",
    "frames_csv",
    "raw_frames",
}


class EncodingError(RuntimeError):
    """A fixed-code error safe to print without disclosing a path."""

    def __init__(self, code: str) -> None:
        if re.fullmatch(r"[a-z0-9_]{1,80}", code) is None:
            code = "encoding_failed"
        self.code = code
        super().__init__(code)


class SafeArgumentParser(argparse.ArgumentParser):
    """Do not echo private command-line values in parser errors."""

    def error(self, message: str) -> None:
        del message
        raise EncodingError("invalid_arguments")


@dataclass(frozen=True, slots=True)
class FrameRow:
    sequence: int
    present_ticks: int
    qpc_frequency: int
    host_perf_counter_ns: int
    accumulated_frames: int
    raw_offset: int
    raw_bytes: int
    frame_sha256: str


@dataclass(frozen=True, slots=True)
class EncodingReport:
    artifact_bytes: int
    artifact_sha256: str
    width: int
    height: int
    frame_count: int
    frame_pts: tuple[int, ...]
    nominal_fps: tuple[int, int]
    nominal_fps_derivation: str
    encoded_pts_cfr: bool
    source_first_present_ticks: int
    source_present_ticks: tuple[int, ...]
    source_qpc_frequency: int
    source_mean_present_fps: tuple[int, int] | None
    source_metadata_sha256: str
    source_raw_sha256: str
    source_frames_csv_sha256: str
    maximum_timestamp_error_qpc_ticks: tuple[int, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": ENCODING_SCHEMA,
            "version": ENCODING_VERSION,
            "status": "encoding_complete",
            "artifact_bytes": self.artifact_bytes,
            "artifact_sha256": self.artifact_sha256,
            "width": self.width,
            "height": self.height,
            "codec": VIDEO_CODEC,
            "pixel_format": PIXEL_FORMAT,
            "time_base": [
                VIDEO_TIME_BASE.numerator,
                VIDEO_TIME_BASE.denominator,
            ],
            "nominal_fps": list(self.nominal_fps),
            "frame_count": self.frame_count,
            "first_pts": self.frame_pts[0],
            "last_pts": self.frame_pts[-1],
            "frame_pts": list(self.frame_pts),
            "encoded_pts_cfr": self.encoded_pts_cfr,
            "frame_resampling": "none",
            "source_missed_presentations": 0,
            "timestamp_quantization_collisions": 0,
            "capture_surface": "dxgi_desktop_client_region_crop",
            "foreground_geometry_guard": True,
            "topmost_or_injected_overlay_detection": False,
            "requires_full_frame_visual_review": True,
            "pc_golden_v3_compatible": False,
            "pc_golden_v3_incompatibility_reason": (
                "v3 requires an integer nominal cadence grid and cannot "
                "faithfully classify arbitrary QPC-derived VFR PTS"
            ),
            "nominal_fps_derivation": self.nominal_fps_derivation,
            "timestamp_mapping": TIMESTAMP_MAPPING,
            "matroska_timestamp_precision_seconds": [1, 1000],
            "source_first_present_ticks": (
                self.source_first_present_ticks
            ),
            "source_present_ticks": list(self.source_present_ticks),
            "source_qpc_frequency": self.source_qpc_frequency,
            "source_mean_present_fps": (
                list(self.source_mean_present_fps)
                if self.source_mean_present_fps is not None
                else None
            ),
            "source_metadata_sha256": self.source_metadata_sha256,
            "source_raw_sha256": self.source_raw_sha256,
            "source_frames_csv_sha256": (
                self.source_frames_csv_sha256
            ),
            "maximum_timestamp_error_qpc_ticks": list(
                self.maximum_timestamp_error_qpc_ticks
            ),
            "maximum_timestamp_error_seconds": list(
                Fraction(
                    self.maximum_timestamp_error_qpc_ticks[0],
                    self.maximum_timestamp_error_qpc_ticks[1]
                    * self.source_qpc_frequency,
                ).as_integer_ratio()
            ),
        }


@dataclass(frozen=True, slots=True)
class _FileSnapshot:
    device: int
    inode: int
    mode: int
    size: int
    mtime_ns: int
    ctime_ns: int


class _BoundedWriter:
    """A seekable file proxy that refuses to grow past a fixed limit."""

    def __init__(self, stream: BinaryIO, limit: int) -> None:
        self._stream = stream
        self._limit = limit
        self.maximum_extent = 0

    def write(self, value: bytes | bytearray | memoryview) -> int:
        try:
            length = memoryview(value).nbytes
            position = self._stream.tell()
        except (BufferError, OSError, TypeError, ValueError):
            raise EncodingError("video_write_failed") from None
        if (
            length < 0
            or position < 0
            or position + length > self._limit
        ):
            raise EncodingError("video_budget_exceeded")
        try:
            written = self._stream.write(value)
        except (MemoryError, OSError, OverflowError, ValueError):
            raise EncodingError("video_write_failed") from None
        if written != length:
            raise EncodingError("video_write_failed")
        self.maximum_extent = max(self.maximum_extent, position + written)
        return written

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        try:
            if whence == os.SEEK_SET:
                target = offset
            elif whence == os.SEEK_CUR:
                target = self._stream.tell() + offset
            elif whence == os.SEEK_END:
                current = self._stream.tell()
                self._stream.seek(0, os.SEEK_END)
                end = self._stream.tell()
                self._stream.seek(current, os.SEEK_SET)
                target = end + offset
            else:
                raise EncodingError("video_write_failed")
            if target < 0 or target > self._limit:
                raise EncodingError("video_budget_exceeded")
            return self._stream.seek(target, os.SEEK_SET)
        except EncodingError:
            raise
        except (OSError, OverflowError, TypeError, ValueError):
            raise EncodingError("video_write_failed") from None

    def tell(self) -> int:
        try:
            return self._stream.tell()
        except (OSError, ValueError):
            raise EncodingError("video_write_failed") from None

    def flush(self) -> None:
        try:
            self._stream.flush()
        except (OSError, ValueError):
            raise EncodingError("video_write_failed") from None

    def writable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True


def _strict_int(
    value: Any,
    *,
    code: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EncodingError(code)
    if minimum is not None and value < minimum:
        raise EncodingError(code)
    if maximum is not None and value > maximum:
        raise EncodingError(code)
    return value


def _strict_finite_number(
    value: Any,
    *,
    code: str,
    minimum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EncodingError(code)
    try:
        result = float(value)
    except (OverflowError, ValueError):
        raise EncodingError(code) from None
    if not math.isfinite(result):
        raise EncodingError(code)
    if minimum is not None and result < minimum:
        raise EncodingError(code)
    return result


def _strict_hash(value: Any, *, code: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise EncodingError(code)
    return value


def _is_junction(path: Path) -> bool:
    try:
        return bool(getattr(path, "is_junction", lambda: False)())
    except OSError:
        raise EncodingError("path_inspection_failed") from None


def _snapshot(info: os.stat_result) -> _FileSnapshot:
    return _FileSnapshot(
        device=int(info.st_dev),
        inode=int(info.st_ino),
        mode=int(info.st_mode),
        size=int(info.st_size),
        mtime_ns=int(info.st_mtime_ns),
        ctime_ns=int(info.st_ctime_ns),
    )


def _open_regular_read_only(path: Path, *, code: str) -> tuple[BinaryIO, _FileSnapshot]:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        if path.is_symlink() or _is_junction(path):
            raise EncodingError(code)
        descriptor = os.open(path, flags)
    except EncodingError:
        raise
    except (OSError, TypeError, ValueError):
        raise EncodingError(code) from None
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise EncodingError(code)
        return os.fdopen(descriptor, "rb", closefd=True), _snapshot(info)
    except EncodingError:
        os.close(descriptor)
        raise
    except (OSError, TypeError, ValueError):
        os.close(descriptor)
        raise EncodingError(code) from None


def _assert_unchanged(stream: BinaryIO, expected: _FileSnapshot) -> None:
    try:
        actual = _snapshot(os.fstat(stream.fileno()))
    except (OSError, TypeError, ValueError):
        raise EncodingError("input_changed") from None
    if actual != expected:
        raise EncodingError("input_changed")


def _read_bounded(
    stream: BinaryIO,
    *,
    maximum: int,
    expected_size: int | None,
    code: str,
) -> bytes:
    try:
        stream.seek(0)
        value = stream.read(maximum + 1)
    except (MemoryError, OSError, OverflowError, ValueError):
        raise EncodingError(code) from None
    if len(value) > maximum:
        raise EncodingError(code)
    if expected_size is not None and len(value) != expected_size:
        raise EncodingError(code)
    return value


def _duplicate_safe_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EncodingError("metadata_invalid")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    del value
    raise EncodingError("metadata_invalid")


def _parse_metadata(value: bytes) -> dict[str, Any]:
    try:
        text = value.decode("ascii")
        metadata = json.loads(
            text,
            object_pairs_hook=_duplicate_safe_object,
            parse_constant=_reject_json_constant,
        )
    except EncodingError:
        raise
    except (
        MemoryError,
        OverflowError,
        RecursionError,
        UnicodeError,
        ValueError,
    ):
        raise EncodingError("metadata_invalid") from None
    if not isinstance(metadata, dict) or set(metadata) != _METADATA_KEYS:
        raise EncodingError("metadata_invalid")
    try:
        canonical = (
            json.dumps(
                metadata,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")
    except (
        MemoryError,
        OverflowError,
        RecursionError,
        TypeError,
        ValueError,
    ):
        raise EncodingError("metadata_invalid") from None
    if canonical != value:
        raise EncodingError("metadata_not_canonical")
    return metadata


def _decimal(value: str, *, code: str) -> int:
    if _DECIMAL_PATTERN.fullmatch(value) is None:
        raise EncodingError(code)
    try:
        return int(value)
    except (OverflowError, ValueError):
        raise EncodingError(code) from None


def _parse_csv(value: bytes) -> tuple[FrameRow, ...]:
    try:
        text = value.decode("ascii")
    except UnicodeError:
        raise EncodingError("frames_csv_invalid") from None
    if "\r" in text or "\0" in text or not text.endswith("\n"):
        raise EncodingError("frames_csv_invalid")
    lines = text.splitlines()
    expected_header = ",".join(_CSV_HEADER)
    if not lines or lines[0] != expected_header:
        raise EncodingError("frames_csv_invalid")
    if len(lines) - 1 > MAX_FRAME_COUNT:
        raise EncodingError("frame_count_exceeded")
    rows: list[FrameRow] = []
    for line in lines[1:]:
        fields = line.split(",")
        if len(fields) != len(_CSV_HEADER):
            raise EncodingError("frames_csv_invalid")
        frame_sha256 = fields[7]
        if _SHA256_PATTERN.fullmatch(frame_sha256) is None:
            raise EncodingError("frames_csv_invalid")
        rows.append(
            FrameRow(
                sequence=_decimal(fields[0], code="frames_csv_invalid"),
                present_ticks=_decimal(
                    fields[1],
                    code="frames_csv_invalid",
                ),
                qpc_frequency=_decimal(
                    fields[2],
                    code="frames_csv_invalid",
                ),
                host_perf_counter_ns=_decimal(
                    fields[3],
                    code="frames_csv_invalid",
                ),
                accumulated_frames=_decimal(
                    fields[4],
                    code="frames_csv_invalid",
                ),
                raw_offset=_decimal(
                    fields[5],
                    code="frames_csv_invalid",
                ),
                raw_bytes=_decimal(
                    fields[6],
                    code="frames_csv_invalid",
                ),
                frame_sha256=frame_sha256,
            )
        )
    if not rows:
        raise EncodingError("no_frames")
    return tuple(rows)


def _validate_region(value: Any, *, code: str) -> tuple[int, int, int, int]:
    if not isinstance(value, list) or len(value) != 4:
        raise EncodingError(code)
    coordinates = tuple(
        _strict_int(item, code=code)
        for item in value
    )
    left, top, right, bottom = coordinates
    if right <= left or bottom <= top:
        raise EncodingError(code)
    return left, top, right, bottom


def _safe_identity_text(value: Any, *, code: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 256
        or any(ord(character) < 32 for character in value)
    ):
        raise EncodingError(code)
    return value


def _validate_source_identity(
    value: Any,
    *,
    global_region: tuple[int, int, int, int],
) -> None:
    expected_keys = {
        "adapter_description",
        "adapter_vendor_id",
        "adapter_vram_bytes",
        "output_device_name",
        "output_desktop_region",
        "output_rotation_degrees",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise EncodingError("source_identity_invalid")
    _safe_identity_text(
        value["adapter_description"],
        code="source_identity_invalid",
    )
    _safe_identity_text(
        value["output_device_name"],
        code="source_identity_invalid",
    )
    _strict_int(
        value["adapter_vendor_id"],
        code="source_identity_invalid",
        minimum=0,
        maximum=0xFFFFFFFF,
    )
    _strict_int(
        value["adapter_vram_bytes"],
        code="source_identity_invalid",
        minimum=0,
        maximum=1024**4,
    )
    if _strict_int(
        value["output_rotation_degrees"],
        code="source_identity_invalid",
    ) != 0:
        raise EncodingError("source_identity_invalid")
    desktop = _validate_region(
        value["output_desktop_region"],
        code="source_identity_invalid",
    )
    if not (
        desktop[0] <= global_region[0] < global_region[2]
        <= desktop[2]
        and desktop[1] <= global_region[1] < global_region[3]
        <= desktop[3]
    ):
        raise EncodingError("source_identity_invalid")


def _validate_target_identity(
    value: Any,
    *,
    global_region: tuple[int, int, int, int],
) -> None:
    expected_keys = {
        "process_id",
        "process_creation_filetime_100ns",
        "executable_bytes",
        "executable_sha256",
        "executable_hash_provenance",
        "window_handle_hex",
        "window_client_region",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise EncodingError("target_identity_invalid")
    _strict_int(
        value["process_id"],
        code="target_identity_invalid",
        minimum=1,
        maximum=0xFFFFFFFF,
    )
    _strict_int(
        value["process_creation_filetime_100ns"],
        code="target_identity_invalid",
        minimum=1,
        maximum=0xFFFFFFFFFFFFFFFF,
    )
    executable_bytes = _strict_int(
        value["executable_bytes"],
        code="target_identity_invalid",
        minimum=1,
        maximum=512 * 1024**2,
    )
    executable_sha256 = _strict_hash(
        value["executable_sha256"],
        code="target_identity_invalid",
    )
    _validate_executable_hash_provenance(
        value["executable_hash_provenance"],
        executable_bytes=executable_bytes,
        executable_sha256=executable_sha256,
    )
    handle = value["window_handle_hex"]
    if (
        not isinstance(handle, str)
        or _WINDOW_HANDLE_PATTERN.fullmatch(handle) is None
        or int(handle, 16) == 0
    ):
        raise EncodingError("target_identity_invalid")
    client = _validate_region(
        value["window_client_region"],
        code="target_identity_invalid",
    )
    if not (
        client[0] <= global_region[0] < global_region[2] <= client[2]
        and client[1] <= global_region[1] < global_region[3] <= client[3]
    ):
        raise EncodingError("target_identity_invalid")


def _validate_executable_hash_provenance(
    value: Any,
    *,
    executable_bytes: int,
    executable_sha256: str,
) -> None:
    code = "target_identity_invalid"
    if value == {"method": "direct_file_sha256"}:
        return
    expected_keys = {
        "method",
        "source_bytes",
        "source_sha256",
        "payload_offset",
        "payload_bytes",
        "payload_sha256",
        "payload_extent_basis",
        "trailing_source_bytes",
        "machine",
        "section_count",
        "pe_timestamp",
        "pe_checksum",
        "size_of_image",
        "size_of_headers",
        "certificate_table_offset",
        "certificate_table_bytes",
        "runtime_file_direct_hash_verified",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise EncodingError(code)
    if (
        value["method"] != "embedded_signed_pe"
        or value["payload_extent_basis"]
        != "certificate_table_end"
        or not isinstance(
            value["runtime_file_direct_hash_verified"],
            bool,
        )
    ):
        raise EncodingError(code)
    source_bytes = _strict_int(
        value["source_bytes"],
        code=code,
        minimum=1,
        maximum=512 * 1024**2,
    )
    _strict_hash(value["source_sha256"], code=code)
    payload_offset = _strict_int(
        value["payload_offset"],
        code=code,
        minimum=1,
        maximum=source_bytes,
    )
    payload_bytes = _strict_int(
        value["payload_bytes"],
        code=code,
        minimum=1,
        maximum=512 * 1024**2,
    )
    payload_sha256 = _strict_hash(
        value["payload_sha256"],
        code=code,
    )
    trailing_bytes = _strict_int(
        value["trailing_source_bytes"],
        code=code,
        minimum=0,
        maximum=source_bytes,
    )
    if (
        payload_bytes != executable_bytes
        or payload_sha256 != executable_sha256
        or payload_offset + payload_bytes + trailing_bytes
        != source_bytes
    ):
        raise EncodingError(code)
    _strict_int(
        value["machine"],
        code=code,
        minimum=0,
        maximum=0xFFFF,
    )
    _strict_int(
        value["section_count"],
        code=code,
        minimum=1,
        maximum=96,
    )
    _strict_int(
        value["pe_timestamp"],
        code=code,
        minimum=0,
        maximum=0xFFFFFFFF,
    )
    _strict_int(
        value["pe_checksum"],
        code=code,
        minimum=0,
        maximum=0xFFFFFFFF,
    )
    size_of_image = _strict_int(
        value["size_of_image"],
        code=code,
        minimum=1,
        maximum=512 * 1024**2,
    )
    size_of_headers = _strict_int(
        value["size_of_headers"],
        code=code,
        minimum=1,
        maximum=payload_bytes,
    )
    certificate_offset = _strict_int(
        value["certificate_table_offset"],
        code=code,
        minimum=1,
        maximum=payload_bytes,
    )
    certificate_bytes = _strict_int(
        value["certificate_table_bytes"],
        code=code,
        minimum=1,
        maximum=payload_bytes,
    )
    if (
        size_of_image < size_of_headers
        or certificate_offset < size_of_headers
        or certificate_offset + certificate_bytes != payload_bytes
    ):
        raise EncodingError(code)


def _validate_metadata(
    metadata: dict[str, Any],
    *,
    metadata_size: int,
    csv_size: int,
    raw_size: int,
) -> tuple[int, int, int, int]:
    if (
        metadata["schema"] != CAPTURE_SCHEMA
        or metadata["version"] != CAPTURE_VERSION
        or metadata["status"] != "acquisition_complete"
    ):
        raise EncodingError("unsupported_capture_schema")
    if (
        metadata["pixel_format"] != PIXEL_FORMAT
        or metadata["bytes_per_pixel"] != BYTES_PER_PIXEL
        or metadata["capture_mode"] != "every_new_present"
        or metadata["capture_storage_mode"] != "memory_then_publish"
        or metadata["capture_surface"]
        != "dxgi_desktop_client_region_crop"
        or metadata["frames_csv"] != CSV_FILE
        or metadata["raw_frames"] != RAW_FILE
    ):
        raise EncodingError("metadata_invalid")
    if metadata["target_process"] != "popcapgame1.exe":
        raise EncodingError("metadata_invalid")
    if metadata["runtime_versions"] != REQUIRED_RUNTIME_VERSIONS:
        raise EncodingError("metadata_invalid")
    if metadata["runtime_versions_verified"] is not True:
        raise EncodingError("metadata_invalid")
    if metadata["foreground_geometry_guard"] is not True:
        raise EncodingError("metadata_invalid")
    if metadata["topmost_or_injected_overlay_detection"] is not False:
        raise EncodingError("metadata_invalid")
    if metadata["requires_full_frame_visual_review"] is not True:
        raise EncodingError("metadata_invalid")

    width = _strict_int(
        metadata["width"],
        code="dimensions_invalid",
        minimum=1,
        maximum=MAX_DIMENSION,
    )
    height = _strict_int(
        metadata["height"],
        code="dimensions_invalid",
        minimum=1,
        maximum=MAX_DIMENSION,
    )
    pixels = width * height
    if pixels > MAX_PIXELS_PER_FRAME:
        raise EncodingError("dimensions_exceeded")
    global_region = _validate_region(
        metadata["global_region"],
        code="metadata_invalid",
    )
    local_region = _validate_region(
        metadata["output_local_region"],
        code="metadata_invalid",
    )
    if (
        global_region[2] - global_region[0] != width
        or global_region[3] - global_region[1] != height
        or local_region[2] - local_region[0] != width
        or local_region[3] - local_region[1] != height
    ):
        raise EncodingError("metadata_invalid")
    _validate_source_identity(
        metadata["source_identity"],
        global_region=global_region,
    )
    _validate_target_identity(
        metadata["target_identity"],
        global_region=global_region,
    )

    frame_count = _strict_int(
        metadata["frame_count"],
        code="frame_count_invalid",
        minimum=1,
        maximum=MAX_FRAME_COUNT,
    )
    qpc_frequency = _strict_int(
        metadata["qpc_frequency"],
        code="qpc_invalid",
        minimum=1,
        maximum=MAX_QPC_FREQUENCY,
    )
    frame_size = pixels * BYTES_PER_PIXEL
    expected_raw = frame_size * frame_count
    if expected_raw > MAX_RAW_BYTES:
        raise EncodingError("raw_budget_exceeded")
    if (
        _strict_int(
            metadata["raw_bytes"],
            code="raw_size_mismatch",
            minimum=1,
            maximum=MAX_RAW_BYTES,
        )
        != expected_raw
        or raw_size != expected_raw
    ):
        raise EncodingError("raw_size_mismatch")
    if (
        _strict_int(
            metadata["frames_csv_bytes"],
            code="frames_csv_size_mismatch",
            minimum=1,
            maximum=MAX_CSV_BYTES,
        )
        != csv_size
        or _strict_int(
            metadata["frames_csv_rows"],
            code="frames_csv_invalid",
            minimum=1,
            maximum=MAX_FRAME_COUNT,
        )
        != frame_count
    ):
        raise EncodingError("frames_csv_size_mismatch")
    if metadata_size <= 0 or metadata_size > MAX_METADATA_BYTES:
        raise EncodingError("metadata_size_exceeded")

    frame_budget_fps = _strict_int(
        metadata["frame_budget_fps"],
        code="metadata_invalid",
        minimum=60,
        maximum=240,
    )
    duration = _strict_finite_number(
        metadata["requested_duration_seconds"],
        code="metadata_invalid",
        minimum=0.25,
    )
    if duration > MAX_REQUESTED_DURATION_SECONDS:
        raise EncodingError("metadata_invalid")
    frame_budget = math.ceil(duration * frame_budget_fps) + 2
    if (
        _strict_int(
            metadata["frame_budget"],
            code="metadata_invalid",
            minimum=1,
            maximum=MAX_FRAME_COUNT,
        )
        != frame_budget
        or frame_count > frame_budget
    ):
        raise EncodingError("metadata_invalid")
    estimated_budget = frame_size * frame_budget
    if (
        _strict_int(
            metadata["estimated_raw_budget_bytes"],
            code="metadata_invalid",
            minimum=1,
            maximum=MAX_RAW_BYTES,
        )
        != estimated_budget
    ):
        raise EncodingError("metadata_invalid")
    capture_elapsed = _strict_finite_number(
        metadata["capture_elapsed_seconds"],
        code="metadata_invalid",
        minimum=0.0,
    )
    if (
        capture_elapsed < duration
        or capture_elapsed > MAX_CAPTURE_ELAPSED_SECONDS
    ):
        raise EncodingError("metadata_invalid")
    capture_start_ns = _strict_int(
        metadata["capture_start_perf_counter_ns"],
        code="metadata_invalid",
        minimum=1,
    )
    capture_end_ns = _strict_int(
        metadata["capture_end_perf_counter_ns"],
        code="metadata_invalid",
        minimum=capture_start_ns,
    )
    if (
        capture_elapsed
        != (capture_end_ns - capture_start_ns) / 1_000_000_000
    ):
        raise EncodingError("metadata_invalid")
    for name in ("device_index", "output_index"):
        _strict_int(
            metadata[name],
            code="metadata_invalid",
            minimum=0,
            maximum=15,
        )
    for name in (
        "idle_poll_count",
        "pointer_only_update_count",
        "warmup_idle_poll_count",
        "warmup_pointer_only_update_count",
    ):
        _strict_int(metadata[name], code="metadata_invalid", minimum=0)
    if _strict_int(
        metadata["missed_presentations"],
        code="missed_presentations_detected",
        minimum=0,
    ) != 0:
        raise EncodingError("missed_presentations_detected")

    first_ticks = _strict_int(
        metadata["first_present_ticks"],
        code="qpc_invalid",
        minimum=1,
    )
    last_ticks = _strict_int(
        metadata["last_present_ticks"],
        code="qpc_invalid",
        minimum=first_ticks,
    )
    baseline = _strict_int(
        metadata["warmup_baseline_present_ticks"],
        code="qpc_invalid",
        minimum=0,
    )
    span_ticks = last_ticks - first_ticks
    if (
        baseline >= first_ticks
        or _strict_int(
            metadata["present_span_ticks"],
            code="qpc_invalid",
            minimum=0,
        )
        != span_ticks
    ):
        raise EncodingError("qpc_invalid")
    if frame_count == 1:
        if (
            span_ticks != 0
            or metadata["present_span_seconds"] is not None
            or metadata["observed_mean_present_fps"] is not None
        ):
            raise EncodingError("qpc_invalid")
    else:
        span_seconds = _strict_finite_number(
            metadata["present_span_seconds"],
            code="qpc_invalid",
            minimum=0.0,
        )
        mean_fps = _strict_finite_number(
            metadata["observed_mean_present_fps"],
            code="qpc_invalid",
            minimum=0.0,
        )
        if (
            span_ticks <= 0
            or span_seconds != span_ticks / qpc_frequency
            or mean_fps != (frame_count - 1) / span_seconds
            or span_seconds > capture_elapsed
        ):
            raise EncodingError("qpc_invalid")

    _strict_hash(metadata["raw_sha256"], code="metadata_invalid")
    _strict_hash(metadata["frames_csv_sha256"], code="metadata_invalid")
    _strict_hash(
        metadata["aggregate_pixel_sha256"],
        code="metadata_invalid",
    )
    return width, height, frame_count, qpc_frequency


def _validate_rows(
    rows: tuple[FrameRow, ...],
    *,
    metadata: dict[str, Any],
    frame_size: int,
    frame_count: int,
    qpc_frequency: int,
) -> None:
    if len(rows) != frame_count:
        raise EncodingError("frames_csv_row_count_mismatch")
    previous_ticks: int | None = None
    previous_host_ns: int | None = None
    expected_offset = 0
    capture_start_ns = metadata["capture_start_perf_counter_ns"]
    capture_end_ns = metadata["capture_end_perf_counter_ns"]
    for sequence, row in enumerate(rows):
        if row.sequence != sequence:
            raise EncodingError("frame_sequence_invalid")
        if row.qpc_frequency != qpc_frequency:
            raise EncodingError("qpc_frequency_changed")
        if row.accumulated_frames != 1:
            raise EncodingError("missed_presentations_detected")
        if row.raw_offset != expected_offset or row.raw_bytes != frame_size:
            raise EncodingError("raw_offset_invalid")
        if row.present_ticks <= 0 or (
            previous_ticks is not None
            and row.present_ticks <= previous_ticks
        ):
            raise EncodingError("present_ticks_not_increasing")
        if (
            row.host_perf_counter_ns <= capture_start_ns
            or row.host_perf_counter_ns > capture_end_ns
            or (
                previous_host_ns is not None
                and row.host_perf_counter_ns <= previous_host_ns
            )
        ):
            raise EncodingError("host_clock_not_increasing")
        expected_offset += frame_size
        previous_ticks = row.present_ticks
        previous_host_ns = row.host_perf_counter_ns
    if (
        rows[0].present_ticks != metadata["first_present_ticks"]
        or rows[-1].present_ticks != metadata["last_present_ticks"]
        or rows[-1].present_ticks - rows[0].present_ticks
        != metadata["present_span_ticks"]
        or expected_offset != metadata["raw_bytes"]
    ):
        raise EncodingError("qpc_invalid")


def _sha256_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _read_exact(stream: BinaryIO, count: int) -> bytes:
    try:
        value = stream.read(count)
    except (MemoryError, OSError, OverflowError, ValueError):
        raise EncodingError("raw_read_failed") from None
    if len(value) != count:
        raise EncodingError("raw_size_mismatch")
    return value


def _validate_raw(
    stream: BinaryIO,
    *,
    rows: tuple[FrameRow, ...],
    metadata: dict[str, Any],
) -> None:
    raw_digest = hashlib.sha256()
    aggregate = hashlib.sha256(_PIXEL_HASH_DOMAIN)
    try:
        stream.seek(0)
    except (OSError, ValueError):
        raise EncodingError("raw_read_failed") from None
    for row in rows:
        try:
            if stream.tell() != row.raw_offset:
                raise EncodingError("raw_offset_invalid")
        except (OSError, ValueError):
            raise EncodingError("raw_read_failed") from None
        payload = _read_exact(stream, row.raw_bytes)
        frame_hash = _sha256_bytes(payload)
        if frame_hash != row.frame_sha256:
            raise EncodingError("frame_sha256_mismatch")
        raw_digest.update(payload)
        aggregate.update(len(payload).to_bytes(8, "big"))
        aggregate.update(payload)
    try:
        if stream.read(1):
            raise EncodingError("raw_size_mismatch")
    except EncodingError:
        raise
    except (OSError, ValueError):
        raise EncodingError("raw_read_failed") from None
    if (
        f"sha256:{raw_digest.hexdigest()}" != metadata["raw_sha256"]
        or f"sha256:{aggregate.hexdigest()}"
        != metadata["aggregate_pixel_sha256"]
    ):
        raise EncodingError("raw_sha256_mismatch")


def qpc_relative_pts(
    present_ticks: int,
    *,
    first_present_ticks: int,
    qpc_frequency: int,
) -> int:
    """Map one QPC tick to Matroska milliseconds using integer arithmetic."""

    delta = present_ticks - first_present_ticks
    if delta < 0 or qpc_frequency <= 0:
        raise EncodingError("qpc_invalid")
    quotient, remainder = divmod(
        delta * VIDEO_TIME_BASE.denominator,
        qpc_frequency,
    )
    if remainder * 2 >= qpc_frequency:
        quotient += 1
    return quotient


def _presentation_timeline(
    rows: tuple[FrameRow, ...],
    *,
    qpc_frequency: int,
) -> tuple[
    tuple[int, ...],
    Fraction,
    Fraction | None,
    str,
    bool,
    tuple[int, int],
]:
    first = rows[0].present_ticks
    pts = tuple(
        qpc_relative_pts(
            row.present_ticks,
            first_present_ticks=first,
            qpc_frequency=qpc_frequency,
        )
        for row in rows
    )
    if pts[0] != 0 or any(
        current <= previous
        for previous, current in zip(pts, pts[1:])
    ):
        raise EncodingError("timestamp_precision_collision")
    differences = tuple(
        current - previous
        for previous, current in zip(pts, pts[1:])
    )
    if len(rows) >= 2:
        source_mean_fps = Fraction(
            (len(rows) - 1) * qpc_frequency,
            rows[-1].present_ticks - rows[0].present_ticks,
        )
        nominal_fps = source_mean_fps.limit_denominator(
            NOMINAL_FPS_MAX_DENOMINATOR
        )
        nominal_derivation = (
            "qpc_span_mean_limited_to_denominator_1001"
        )
    else:
        source_mean_fps = None
        nominal_fps = Fraction(1, 1)
        nominal_derivation = "single_frame_fallback_1fps"
    cfr = not differences or all(
        difference == differences[0]
        for difference in differences
    )
    maximum_error_numerator = 0
    for row, frame_pts in zip(rows, pts):
        exact_numerator = (
            row.present_ticks - first
        ) * VIDEO_TIME_BASE.denominator
        error_numerator = abs(
            frame_pts * qpc_frequency - exact_numerator
        )
        maximum_error_numerator = max(
            maximum_error_numerator,
            error_numerator,
        )
    error = Fraction(
        maximum_error_numerator,
        VIDEO_TIME_BASE.denominator,
    )
    return (
        pts,
        nominal_fps,
        source_mean_fps,
        nominal_derivation,
        cfr,
        (error.numerator, error.denominator),
    )


def _load_runtime() -> tuple[Any, Any]:
    try:
        av_module = importlib.import_module("av")
        np_module = importlib.import_module("numpy")
    except (ImportError, OSError, RuntimeError):
        raise EncodingError("encoding_runtime_unavailable") from None
    try:
        major = int(str(av_module.__version__).split(".", 1)[0])
    except (AttributeError, TypeError, ValueError):
        raise EncodingError("unsupported_pyav_runtime") from None
    if major != 18:
        raise EncodingError("unsupported_pyav_runtime")
    return av_module, np_module


def _timeline_metadata_json(
    *,
    rows: tuple[FrameRow, ...],
    frame_pts: tuple[int, ...],
    qpc_frequency: int,
    source_mean_fps: Fraction | None,
    nominal_fps: Fraction,
    nominal_derivation: str,
    maximum_error: tuple[int, int],
    source_metadata_sha256: str,
    metadata: dict[str, Any],
) -> str:
    error_seconds = Fraction(
        maximum_error[0],
        maximum_error[1] * qpc_frequency,
    )
    timeline = {
        "schema": TIMELINE_SCHEMA,
        "version": TIMELINE_VERSION,
        "pc_golden_v3_compatible": False,
        "pc_golden_v3_incompatibility_reason": (
            "v3 requires an integer nominal cadence grid and cannot "
            "faithfully classify arbitrary QPC-derived VFR PTS"
        ),
        "time_base": [
            VIDEO_TIME_BASE.numerator,
            VIDEO_TIME_BASE.denominator,
        ],
        "matroska_timestamp_precision_seconds": [1, 1000],
        "timestamp_mapping": TIMESTAMP_MAPPING,
        "source_first_present_ticks": rows[0].present_ticks,
        "source_present_ticks": [
            row.present_ticks for row in rows
        ],
        "source_qpc_frequency": qpc_frequency,
        "source_mean_present_fps": (
            [source_mean_fps.numerator, source_mean_fps.denominator]
            if source_mean_fps is not None
            else None
        ),
        "stream_nominal_fps": [
            nominal_fps.numerator,
            nominal_fps.denominator,
        ],
        "stream_nominal_fps_derivation": nominal_derivation,
        "frame_pts": list(frame_pts),
        "encoded_pts_cfr": (
            len(frame_pts) <= 2
            or len(set(
                current - previous
                for previous, current in zip(
                    frame_pts,
                    frame_pts[1:],
                )
            )) == 1
        ),
        "frame_resampling": "none",
        "source_missed_presentations": 0,
        "timestamp_quantization_collisions": 0,
        "capture_surface": metadata["capture_surface"],
        "foreground_geometry_guard": metadata[
            "foreground_geometry_guard"
        ],
        "topmost_or_injected_overlay_detection": metadata[
            "topmost_or_injected_overlay_detection"
        ],
        "requires_full_frame_visual_review": metadata[
            "requires_full_frame_visual_review"
        ],
        "maximum_timestamp_error_qpc_ticks": list(maximum_error),
        "maximum_timestamp_error_seconds": [
            error_seconds.numerator,
            error_seconds.denominator,
        ],
        "source_raw_sha256": metadata["raw_sha256"],
        "source_metadata_sha256": source_metadata_sha256,
        "source_frames_csv_sha256": metadata[
            "frames_csv_sha256"
        ],
    }
    try:
        encoded = json.dumps(
            timeline,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        if len(encoded.encode("ascii")) > MAX_TIMELINE_TAG_BYTES:
            raise EncodingError("timeline_metadata_exceeded")
        return encoded
    except EncodingError:
        raise
    except (MemoryError, RecursionError, TypeError, UnicodeError, ValueError):
        raise EncodingError("timeline_metadata_invalid") from None


def _prepare_input_directory(path: str | Path) -> Path:
    directory = Path(path)
    if not directory.is_absolute():
        raise EncodingError("input_directory_not_absolute")
    try:
        if (
            directory.is_symlink()
            or _is_junction(directory)
            or not directory.is_dir()
        ):
            raise EncodingError("input_directory_invalid")
        names = {item.name for item in directory.iterdir()}
    except EncodingError:
        raise
    except OSError:
        raise EncodingError("input_directory_invalid") from None
    if any(name.casefold().endswith(".part") for name in names):
        raise EncodingError("input_contains_part")
    if names != {METADATA_FILE, CSV_FILE, RAW_FILE}:
        raise EncodingError("input_directory_invalid")
    return directory


def _path_exists(path: Path) -> bool:
    try:
        return os.path.lexists(path)
    except (OSError, TypeError, ValueError):
        raise EncodingError("output_path_invalid") from None


def _prepare_output(
    output: str | Path,
    *,
    input_directory: Path,
) -> tuple[Path, Path, Path | None]:
    supplied = Path(output)
    if not supplied.is_absolute():
        raise EncodingError("output_path_not_absolute")
    try:
        if _path_exists(supplied) and supplied.is_dir():
            if supplied.is_symlink() or _is_junction(supplied):
                raise EncodingError("output_directory_invalid")
            if any(supplied.iterdir()):
                raise EncodingError("output_directory_not_empty")
            if os.path.samefile(supplied, input_directory):
                raise EncodingError("output_overlaps_input")
            final = supplied / DIRECTORY_OUTPUT_FILE
            exclusive_directory: Path | None = supplied
        else:
            if _path_exists(supplied):
                raise EncodingError("output_already_exists")
            if supplied.suffix.casefold() != ".mkv":
                raise EncodingError("output_file_extension_invalid")
            parent = supplied.parent
            if (
                parent.is_symlink()
                or _is_junction(parent)
                or not parent.is_dir()
            ):
                raise EncodingError("output_parent_invalid")
            if os.path.samefile(parent, input_directory):
                raise EncodingError("output_overlaps_input")
            final = supplied
            exclusive_directory = None
        part = final.with_name(final.name + ".part")
        if _path_exists(final) or _path_exists(part):
            raise EncodingError("output_already_exists")
    except EncodingError:
        raise
    except OSError:
        raise EncodingError("output_path_invalid") from None
    return final, part, exclusive_directory


def _assert_output_directory_contents(
    directory: Path,
    expected_names: set[str],
) -> None:
    try:
        if (
            directory.is_symlink()
            or _is_junction(directory)
            or not directory.is_dir()
            or {item.name for item in directory.iterdir()}
            != expected_names
        ):
            raise EncodingError("output_directory_changed")
    except EncodingError:
        raise
    except OSError:
        raise EncodingError("output_directory_changed") from None


def _encode_video(
    *,
    av_module: Any,
    np_module: Any,
    raw_stream: BinaryIO,
    rows: tuple[FrameRow, ...],
    pts: tuple[int, ...],
    nominal_fps: Fraction,
    width: int,
    height: int,
    timeline_metadata_json: str,
    output_stream: BinaryIO,
) -> None:
    bounded = _BoundedWriter(output_stream, MAX_VIDEO_BYTES)
    container = None
    try:
        container = av_module.open(
            bounded,
            mode="w",
            format="matroska",
        )
        container.metadata[TIMELINE_METADATA_TAG] = (
            timeline_metadata_json
        )
        video = container.add_stream(VIDEO_CODEC, rate=nominal_fps)
        video.width = width
        video.height = height
        video.pix_fmt = PIXEL_FORMAT
        video.time_base = VIDEO_TIME_BASE
        # FFV1 otherwise derives its encoder clock from ``rate`` and can
        # silently snap VFR timestamps to nominal-frame boundaries.
        video.codec_context.time_base = VIDEO_TIME_BASE
        raw_stream.seek(0)
        for row, frame_pts in zip(rows, pts):
            payload = _read_exact(raw_stream, row.raw_bytes)
            if _sha256_bytes(payload) != row.frame_sha256:
                raise EncodingError("input_changed")
            pixels = np_module.frombuffer(
                payload,
                dtype=np_module.uint8,
            ).reshape((height, width, BYTES_PER_PIXEL))
            frame = av_module.VideoFrame.from_ndarray(
                pixels,
                format=PIXEL_FORMAT,
            )
            frame.pts = frame_pts
            frame.time_base = VIDEO_TIME_BASE
            for packet in video.encode(frame):
                container.mux(packet)
        if raw_stream.read(1):
            raise EncodingError("input_changed")
        for packet in video.encode():
            container.mux(packet)
        container.close()
        container = None
        bounded.flush()
        os.fsync(output_stream.fileno())
    except EncodingError:
        raise
    except (MemoryError, OSError, OverflowError, RecursionError):
        raise EncodingError("video_encoding_failed_safely") from None
    except Exception:
        raise EncodingError("video_encoding_failed") from None
    finally:
        if container is not None:
            try:
                container.close()
            except Exception:
                pass


def _average_rate_matches(
    observed: Any,
    expected: Fraction,
) -> bool:
    """Allow Matroska's advisory average-rate rational approximation.

    Exact per-frame PTS and decoded pixel hashes are validated separately.
    This field is only an advisory container rational, so its bound combines
    a small absolute floor with a one-basis-point relative tolerance.
    """

    if observed is None:
        return False
    try:
        value = Fraction(observed)
    except (TypeError, ValueError, ZeroDivisionError):
        return False
    tolerance = max(
        AVERAGE_RATE_ABSOLUTE_TOLERANCE,
        abs(expected) * AVERAGE_RATE_RELATIVE_TOLERANCE,
    )
    return value > 0 and abs(value - expected) <= tolerance


def _verify_video(
    *,
    av_module: Any,
    path: Path,
    rows: tuple[FrameRow, ...],
    expected_pts: tuple[int, ...],
    nominal_fps: Fraction,
    width: int,
    height: int,
    timeline_metadata_json: str,
) -> tuple[int, str]:
    stream, snapshot = _open_regular_read_only(
        path,
        code="video_validation_failed",
    )
    with stream:
        if snapshot.size <= 0 or snapshot.size > MAX_VIDEO_BYTES:
            raise EncodingError("video_budget_exceeded")
        try:
            container = av_module.open(
                stream,
                mode="r",
                format="matroska",
                options={
                    "err_detect": (
                        "explode+crccheck+bitstream+buffer+careful"
                    )
                },
            )
        except Exception:
            raise EncodingError("video_validation_failed") from None
        try:
            all_streams = tuple(container.streams)
            video_streams = tuple(container.streams.video)
            if (
                len(all_streams) != 1
                or len(video_streams) != 1
                or container.metadata.get(TIMELINE_METADATA_TAG)
                != timeline_metadata_json
            ):
                raise EncodingError("video_validation_failed")
            video = video_streams[0]
            context = video.codec_context
            if (
                str(context.name) != VIDEO_CODEC
                or str(context.format.name) != PIXEL_FORMAT
                or (int(context.width), int(context.height))
                != (width, height)
                or Fraction(video.time_base) != VIDEO_TIME_BASE
                or not _average_rate_matches(
                    video.average_rate,
                    nominal_fps,
                )
            ):
                raise EncodingError("video_validation_failed")
            decoded = 0
            for frame in container.decode(video):
                if decoded >= len(rows):
                    raise EncodingError("video_validation_failed")
                if (
                    frame.pts != expected_pts[decoded]
                    or Fraction(frame.time_base) != VIDEO_TIME_BASE
                    or (int(frame.width), int(frame.height))
                    != (width, height)
                    or str(frame.format.name) != PIXEL_FORMAT
                    or bool(frame.is_corrupt)
                ):
                    raise EncodingError("video_validation_failed")
                pixels = frame.to_ndarray(format=PIXEL_FORMAT)
                if (
                    pixels.dtype.name != "uint8"
                    or tuple(pixels.shape)
                    != (height, width, BYTES_PER_PIXEL)
                    or not bool(pixels.flags.c_contiguous)
                    or _sha256_bytes(memoryview(pixels))
                    != rows[decoded].frame_sha256
                ):
                    raise EncodingError("video_losslessness_failed")
                decoded += 1
            if decoded != len(rows):
                raise EncodingError("video_validation_failed")
        except EncodingError:
            raise
        except (MemoryError, OSError, OverflowError, RecursionError):
            raise EncodingError("video_validation_failed_safely") from None
        except Exception:
            raise EncodingError("video_validation_failed") from None
        finally:
            try:
                container.close()
            except Exception:
                pass
        _assert_unchanged(stream, snapshot)
        try:
            stream.seek(0)
            digest = hashlib.sha256()
            total = 0
            while chunk := stream.read(READ_CHUNK_BYTES):
                total += len(chunk)
                if total > MAX_VIDEO_BYTES:
                    raise EncodingError("video_budget_exceeded")
                digest.update(chunk)
        except EncodingError:
            raise
        except (MemoryError, OSError, OverflowError, ValueError):
            raise EncodingError("video_validation_failed_safely") from None
        if total != snapshot.size:
            raise EncodingError("video_validation_failed")
        return total, f"sha256:{digest.hexdigest()}"


def _publish_no_replace(
    part: Path,
    final: Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
) -> None:
    stream, snapshot = _open_regular_read_only(
        part,
        code="output_publish_failed",
    )
    linked = False
    try:
        with stream:
            stream.seek(0)
            digest = hashlib.sha256()
            total = 0
            while chunk := stream.read(READ_CHUNK_BYTES):
                total += len(chunk)
                if total > MAX_VIDEO_BYTES:
                    raise EncodingError("video_budget_exceeded")
                digest.update(chunk)
            if (
                total != expected_bytes
                or f"sha256:{digest.hexdigest()}" != expected_sha256
            ):
                raise EncodingError("output_changed")
            os.link(part, final, follow_symlinks=False)
            linked = True
            final_info = os.stat(final, follow_symlinks=False)
            if (
                not stat.S_ISREG(final_info.st_mode)
                or int(final_info.st_dev) != snapshot.device
                or int(final_info.st_ino) != snapshot.inode
                or int(final_info.st_size) != expected_bytes
            ):
                raise EncodingError("output_publish_failed")
            stream.seek(0)
            digest = hashlib.sha256()
            total = 0
            while chunk := stream.read(READ_CHUNK_BYTES):
                total += len(chunk)
                if total > MAX_VIDEO_BYTES:
                    raise EncodingError("video_budget_exceeded")
                digest.update(chunk)
            if (
                total != expected_bytes
                or f"sha256:{digest.hexdigest()}" != expected_sha256
            ):
                raise EncodingError("output_changed")
        part_info = os.stat(part, follow_symlinks=False)
        if (
            not stat.S_ISREG(part_info.st_mode)
            or int(part_info.st_dev) != snapshot.device
            or int(part_info.st_ino) != snapshot.inode
            or int(part_info.st_size) != expected_bytes
        ):
            raise EncodingError("output_changed")
        part.unlink()
        linked = False
    except EncodingError:
        if linked:
            try:
                final.unlink()
            except OSError:
                pass
        raise
    except (FileExistsError, OSError, TypeError, ValueError):
        if linked:
            try:
                final.unlink()
            except OSError:
                pass
        raise EncodingError("output_publish_failed") from None


def encode_capture(
    input_directory: str | Path,
    output: str | Path,
) -> EncodingReport:
    """Validate and encode one acquisition without modifying its directory."""

    source = _prepare_input_directory(input_directory)
    final, part, exclusive_output_directory = _prepare_output(
        output,
        input_directory=source,
    )
    av_module, np_module = _load_runtime()

    with ExitStack() as stack:
        metadata_stream, metadata_snapshot = _open_regular_read_only(
            source / METADATA_FILE,
            code="metadata_open_failed",
        )
        csv_stream, csv_snapshot = _open_regular_read_only(
            source / CSV_FILE,
            code="frames_csv_open_failed",
        )
        raw_stream, raw_snapshot = _open_regular_read_only(
            source / RAW_FILE,
            code="raw_open_failed",
        )
        stack.enter_context(metadata_stream)
        stack.enter_context(csv_stream)
        stack.enter_context(raw_stream)

        metadata_bytes = _read_bounded(
            metadata_stream,
            maximum=MAX_METADATA_BYTES,
            expected_size=metadata_snapshot.size,
            code="metadata_size_exceeded",
        )
        csv_bytes = _read_bounded(
            csv_stream,
            maximum=MAX_CSV_BYTES,
            expected_size=csv_snapshot.size,
            code="frames_csv_size_exceeded",
        )
        metadata = _parse_metadata(metadata_bytes)
        width, height, frame_count, qpc_frequency = _validate_metadata(
            metadata,
            metadata_size=metadata_snapshot.size,
            csv_size=csv_snapshot.size,
            raw_size=raw_snapshot.size,
        )
        if _sha256_bytes(csv_bytes) != metadata["frames_csv_sha256"]:
            raise EncodingError("frames_csv_sha256_mismatch")
        rows = _parse_csv(csv_bytes)
        frame_size = width * height * BYTES_PER_PIXEL
        _validate_rows(
            rows,
            metadata=metadata,
            frame_size=frame_size,
            frame_count=frame_count,
            qpc_frequency=qpc_frequency,
        )
        _validate_raw(raw_stream, rows=rows, metadata=metadata)
        (
            frame_pts,
            nominal_fps,
            source_mean_fps,
            nominal_derivation,
            encoded_pts_cfr,
            maximum_error,
        ) = _presentation_timeline(
            rows,
            qpc_frequency=qpc_frequency,
        )
        timeline_metadata_json = _timeline_metadata_json(
            rows=rows,
            frame_pts=frame_pts,
            qpc_frequency=qpc_frequency,
            source_mean_fps=source_mean_fps,
            nominal_fps=nominal_fps,
            nominal_derivation=nominal_derivation,
            maximum_error=maximum_error,
            source_metadata_sha256=_sha256_bytes(metadata_bytes),
            metadata=metadata,
        )

        if exclusive_output_directory is not None:
            _assert_output_directory_contents(
                exclusive_output_directory,
                set(),
            )
        try:
            output_stream = part.open("xb+")
        except (FileExistsError, OSError):
            raise EncodingError("output_creation_failed") from None
        with output_stream:
            _encode_video(
                av_module=av_module,
                np_module=np_module,
                raw_stream=raw_stream,
                rows=rows,
                pts=frame_pts,
                nominal_fps=nominal_fps,
                width=width,
                height=height,
                timeline_metadata_json=timeline_metadata_json,
                output_stream=output_stream,
            )

        _assert_unchanged(metadata_stream, metadata_snapshot)
        _assert_unchanged(csv_stream, csv_snapshot)
        _assert_unchanged(raw_stream, raw_snapshot)
        artifact_bytes, artifact_sha256 = _verify_video(
            av_module=av_module,
            path=part,
            rows=rows,
            expected_pts=frame_pts,
            nominal_fps=nominal_fps,
            width=width,
            height=height,
            timeline_metadata_json=timeline_metadata_json,
        )
        _assert_unchanged(metadata_stream, metadata_snapshot)
        _assert_unchanged(csv_stream, csv_snapshot)
        _assert_unchanged(raw_stream, raw_snapshot)
        try:
            names = {item.name for item in source.iterdir()}
        except OSError:
            raise EncodingError("input_changed") from None
        if names != {METADATA_FILE, CSV_FILE, RAW_FILE}:
            raise EncodingError("input_changed")

    if exclusive_output_directory is not None:
        _assert_output_directory_contents(
            exclusive_output_directory,
            {part.name},
        )
    _publish_no_replace(
        part,
        final,
        expected_bytes=artifact_bytes,
        expected_sha256=artifact_sha256,
    )
    return EncodingReport(
        artifact_bytes=artifact_bytes,
        artifact_sha256=artifact_sha256,
        width=width,
        height=height,
        frame_count=frame_count,
        frame_pts=frame_pts,
        nominal_fps=(
            nominal_fps.numerator,
            nominal_fps.denominator,
        ),
        nominal_fps_derivation=nominal_derivation,
        encoded_pts_cfr=encoded_pts_cfr,
        source_first_present_ticks=rows[0].present_ticks,
        source_present_ticks=tuple(
            row.present_ticks for row in rows
        ),
        source_qpc_frequency=qpc_frequency,
        source_mean_present_fps=(
            (
                source_mean_fps.numerator,
                source_mean_fps.denominator,
            )
            if source_mean_fps is not None
            else None
        ),
        source_metadata_sha256=_sha256_bytes(metadata_bytes),
        source_raw_sha256=metadata["raw_sha256"],
        source_frames_csv_sha256=metadata["frames_csv_sha256"],
        maximum_timestamp_error_qpc_ticks=maximum_error,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = SafeArgumentParser(
        description=(
            "Validate a completed raw DXGI acquisition and encode FFV1/MKV."
        )
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def run(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    report = encode_capture(arguments.input_dir, arguments.output)
    print(
        json.dumps(
            report.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(argv)
    except EncodingError as error:
        print(f"encode error: {error.code}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("encode error: interrupted", file=sys.stderr)
        return 130
    except Exception:
        print("encode error: unexpected_failure", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
