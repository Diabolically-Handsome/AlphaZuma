"""Strict, read-only semantic inspection of PC golden video evidence.

The manifest's file hash proves which bytes were supplied.  This module goes
further and asks FFmpeg (through PyAV) to open the file, validates the one
video stream against :class:`~zuma_rl.pc_golden.VideoMetadata`, decodes every
frame, validates every presentation timestamp, and hashes a canonical RGB24
pixel stream.

Pixel equality is deliberately *not* used to infer duplicate capture frames:
an unchanged game screen is perfectly valid.  Dropped and duplicate counts
are derived only from presentation timestamp cadence.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import struct
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, BinaryIO

import numpy as np

from zuma_rl.pc_golden import (
    ArtifactSpec,
    PcGoldenManifest,
    PcGoldenValidationError,
    VideoMetadata,
    _json_loads_strict,
    _mapping,
    _sha256,
    _strict_int,
)

try:  # Optional so the verifier can report INCOMPARABLE without PyAV.
    import av as _av
    if str(getattr(_av, "__version__", "")).split(".", 1)[0] != "18":
        _av = None
except (ImportError, OSError, RuntimeError):
    _av = None


MAX_VIDEO_DIMENSION = 16_384
MAX_VIDEO_PIXELS_PER_FRAME = 33_554_432
MAX_VIDEO_FRAME_COUNT = 100_000
MAX_NORMALIZED_DECODE_BYTES = 32 * 1024**3
MAX_VIDEO_ARTIFACT_BYTES = 64 * 1024**3
NORMALIZED_PIXEL_FORMAT = "rgb24"
_PIXEL_HASH_DOMAIN = b"zuma-rl.pc-video.rgb24.v1\0"
_FRAME_PIXEL_HASH_DOMAIN = b"zuma-rl.pc-video.rgb24-frame.v1\0"
_READ_CHUNK_BYTES = 1024 * 1024
_DXGI_TIMELINE_TAG = "ZUMA_DXGI_TIMELINE"
_DXGI_TIMELINE_SCHEMA = "zuma-rl.dxgi-qpc-timeline"
_DXGI_TIMELINE_VERSION = 1
_DXGI_TIMELINE_KEYS = {
    "schema",
    "version",
    "pc_golden_v3_compatible",
    "pc_golden_v3_incompatibility_reason",
    "time_base",
    "matroska_timestamp_precision_seconds",
    "timestamp_mapping",
    "source_first_present_ticks",
    "source_present_ticks",
    "source_qpc_frequency",
    "source_mean_present_fps",
    "stream_nominal_fps",
    "stream_nominal_fps_derivation",
    "frame_pts",
    "encoded_pts_cfr",
    "frame_resampling",
    "source_missed_presentations",
    "timestamp_quantization_collisions",
    "capture_surface",
    "foreground_geometry_guard",
    "topmost_or_injected_overlay_detection",
    "requires_full_frame_visual_review",
    "maximum_timestamp_error_qpc_ticks",
    "maximum_timestamp_error_seconds",
    "source_raw_sha256",
    "source_metadata_sha256",
    "source_frames_csv_sha256",
}


def canonical_rgb24_frame_sha256(pixels: np.ndarray[Any, Any]) -> str:
    """Hash one contiguous uint8 RGB24 frame in the verifier's domain."""

    if (
        not isinstance(pixels, np.ndarray)
        or pixels.dtype != np.uint8
        or pixels.ndim != 3
        or pixels.shape[2] != 3
    ):
        raise ValueError("RGB24 frame must be a height-by-width uint8 array")
    contiguous = np.ascontiguousarray(pixels)
    digest = hashlib.sha256(_FRAME_PIXEL_HASH_DOMAIN)
    digest.update(struct.pack(">Q", contiguous.nbytes))
    digest.update(memoryview(contiguous))
    return f"sha256:{digest.hexdigest()}"


class PcVideoError(RuntimeError):
    """Base class for safe PC-video inspection failures."""


class VideoDecoderUnavailable(PcVideoError):
    """PyAV/FFmpeg is unavailable, so video evidence is incomparable."""


class VideoArtifactError(PcVideoError):
    """The supplied file is not the content-addressed manifest artifact."""


class VideoResourceLimitError(PcVideoError):
    """Declared or decoded video exceeds a fixed inspection resource limit."""


class VideoDecodeError(PcVideoError):
    """The video container or one of its frames cannot be fully decoded."""


class VideoMetadataMismatch(PcVideoError):
    """Decoded video semantics differ from the immutable manifest."""


@dataclass(frozen=True, slots=True)
class PcVideoInspection:
    """Semantic facts proved by a successful full-stream decode."""

    artifact_sha256: str
    artifact_bytes: int
    width: int
    height: int
    codec: str
    pixel_format: str
    time_base: tuple[int, int]
    nominal_fps: tuple[int, int]
    frame_count: int
    first_pts: int
    last_pts: int
    frame_pts: tuple[int, ...]
    cfr: bool
    dropped_frames: int
    duplicate_frames: int
    normalized_pixel_format: str
    normalized_decoded_bytes: int
    decoded_pixel_sha256: str
    frame_pixel_sha256: tuple[str, ...] = ()
    frame_pixel_sha256_without_last_row: tuple[str, ...] = ()
    # Kept in-memory for the independent verifier's bounded edge comparison.
    # Raw row bytes are intentionally omitted from ``to_dict``.
    frame_last_row_rgb24: tuple[bytes, ...] = ()
    dxgi_source_metadata_sha256: str | None = None
    dxgi_source_raw_sha256: str | None = None
    dxgi_source_frames_csv_sha256: str | None = None
    dxgi_source_present_ticks: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible summary (including the validated PTS)."""

        return {
            "artifact_sha256": self.artifact_sha256,
            "artifact_bytes": self.artifact_bytes,
            "width": self.width,
            "height": self.height,
            "codec": self.codec,
            "pixel_format": self.pixel_format,
            "time_base": list(self.time_base),
            "nominal_fps": list(self.nominal_fps),
            "frame_count": self.frame_count,
            "first_pts": self.first_pts,
            "last_pts": self.last_pts,
            "frame_pts": list(self.frame_pts),
            "cfr": self.cfr,
            "dropped_frames": self.dropped_frames,
            "duplicate_frames": self.duplicate_frames,
            "normalized_pixel_format": self.normalized_pixel_format,
            "normalized_decoded_bytes": self.normalized_decoded_bytes,
            "decoded_pixel_sha256": self.decoded_pixel_sha256,
            "frame_pixel_sha256": list(self.frame_pixel_sha256),
            "frame_pixel_sha256_without_last_row": list(
                self.frame_pixel_sha256_without_last_row
            ),
            "dxgi_source_metadata_sha256": (
                self.dxgi_source_metadata_sha256
            ),
            "dxgi_source_raw_sha256": self.dxgi_source_raw_sha256,
            "dxgi_source_frames_csv_sha256": (
                self.dxgi_source_frames_csv_sha256
            ),
            "dxgi_source_present_ticks": list(
                self.dxgi_source_present_ticks
            ),
        }


def _safe_fraction(value: Any, field: str) -> tuple[int, int]:
    try:
        if value is None:
            raise VideoMetadataMismatch(f"video {field} is unavailable")
        result = Fraction(value)
        if result.numerator <= 0 or result.denominator <= 0:
            raise VideoMetadataMismatch(f"video {field} is invalid")
        return result.numerator, result.denominator
    except VideoMetadataMismatch:
        raise
    except (ArithmeticError, OverflowError, RecursionError, TypeError, ValueError):
        raise VideoMetadataMismatch(f"video {field} is invalid") from None


def _validate_declared_resources(
    metadata: VideoMetadata,
    *,
    artifact_bytes: int,
) -> int:
    if artifact_bytes > MAX_VIDEO_ARTIFACT_BYTES:
        raise VideoResourceLimitError(
            "video artifact exceeds the inspection byte limit"
        )
    if (
        metadata.width > MAX_VIDEO_DIMENSION
        or metadata.height > MAX_VIDEO_DIMENSION
    ):
        raise VideoResourceLimitError(
            "video dimensions exceed the inspection limit"
        )
    try:
        pixels_per_frame = metadata.width * metadata.height
        normalized_bytes = pixels_per_frame * 3 * metadata.frame_count
    except (OverflowError, RecursionError):
        raise VideoResourceLimitError(
            "video resource declaration cannot be represented safely"
        ) from None
    if pixels_per_frame > MAX_VIDEO_PIXELS_PER_FRAME:
        raise VideoResourceLimitError(
            "video pixels per frame exceed the inspection limit"
        )
    if metadata.frame_count > MAX_VIDEO_FRAME_COUNT:
        raise VideoResourceLimitError(
            "video frame count exceeds the inspection limit"
        )
    if normalized_bytes > MAX_NORMALIZED_DECODE_BYTES:
        raise VideoResourceLimitError(
            "video normalized decode size exceeds the inspection limit"
        )
    return normalized_bytes


def validate_pc_video_declarations(
    manifest: PcGoldenManifest,
) -> int:
    """Validate video resource declarations without opening any artifact.

    Returns the declared normalized RGB24 byte count.  Callers should run
    this preflight before hashing potentially oversized artifacts.
    """

    if not isinstance(manifest, PcGoldenManifest):
        raise VideoMetadataMismatch(
            "video preflight requires a validated PC golden manifest"
        )
    metadata = manifest.video
    try:
        artifact = manifest.artifacts[metadata.artifact]
    except (KeyError, OverflowError, RecursionError, TypeError):
        raise VideoMetadataMismatch(
            "manifest video artifact reference is invalid"
        ) from None
    return validate_pc_video_artifact_declaration(metadata, artifact)


def validate_pc_video_artifact_declaration(
    metadata: VideoMetadata,
    artifact: ArtifactSpec,
) -> int:
    """Validate one primary or replay video declaration without opening it."""

    if not isinstance(metadata, VideoMetadata):
        raise VideoMetadataMismatch(
            "video declaration requires VideoMetadata"
        )
    if not isinstance(artifact, ArtifactSpec):
        raise VideoMetadataMismatch(
            "video declaration requires ArtifactSpec"
        )
    return _validate_declared_resources(
        metadata,
        artifact_bytes=artifact.bytes,
    )


def _open_regular_read_only(path: str | os.PathLike[str]) -> BinaryIO:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    try:
        raw_path = os.fspath(path)
        descriptor = os.open(raw_path, flags)
    except Exception:
        raise VideoArtifactError("video artifact cannot be opened read-only") from None
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise VideoArtifactError("video artifact must be a regular file")
        return os.fdopen(descriptor, "rb", closefd=True)
    except VideoArtifactError:
        os.close(descriptor)
        raise
    except Exception:
        os.close(descriptor)
        raise VideoArtifactError("video artifact cannot be inspected") from None


def _digest_open_file(file_object: BinaryIO) -> tuple[int, str]:
    digest = hashlib.sha256()
    total = 0
    try:
        file_object.seek(0)
        while True:
            chunk = file_object.read(_READ_CHUNK_BYTES)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_VIDEO_ARTIFACT_BYTES:
                raise VideoResourceLimitError(
                    "video artifact exceeds the inspection byte limit"
                )
            digest.update(chunk)
        file_object.seek(0)
    except VideoResourceLimitError:
        raise
    except (MemoryError, OSError, OverflowError, RecursionError, TypeError,
            ValueError):
        raise VideoArtifactError("video artifact cannot be read safely") from None
    return total, f"sha256:{digest.hexdigest()}"


def _stream_header(
    stream: Any,
) -> tuple[int, int, str, str, tuple[int, int], tuple[int, int]]:
    try:
        context = stream.codec_context
        width = int(context.width)
        height = int(context.height)
        codec = str(context.name)
        pixel_format_value = context.format
        pixel_format = str(pixel_format_value.name)
        time_base = _safe_fraction(stream.time_base, "stream time_base")
        nominal_fps = _safe_fraction(stream.average_rate, "nominal frame rate")
    except PcVideoError:
        raise
    except (ArithmeticError, AttributeError, OverflowError, RecursionError,
            TypeError, ValueError):
        raise VideoDecodeError("video stream header is incomplete") from None
    return width, height, codec, pixel_format, time_base, nominal_fps


def _header_mismatches(
    metadata: VideoMetadata,
    *,
    width: int,
    height: int,
    codec: str,
    pixel_format: str,
    time_base: tuple[int, int],
    nominal_fps: tuple[int, int],
) -> tuple[str, ...]:
    mismatches: list[str] = []
    if (width, height) != (metadata.width, metadata.height):
        mismatches.append("dimensions")
    if codec != metadata.codec:
        mismatches.append("codec")
    if pixel_format != metadata.pixel_format:
        mismatches.append("pixel_format")
    if time_base != metadata.time_base:
        mismatches.append("time_base")
    if nominal_fps != metadata.nominal_fps:
        mismatches.append("nominal_fps")
    return tuple(mismatches)


def _cadence_counts(
    pts_values: tuple[int, ...],
    *,
    pts_step: int,
) -> tuple[bool, int, int]:
    """Return exact-CFR, dropped, duplicate counts using PTS only.

    A short interval is an extra presentation inside one nominal cadence and
    is counted as a duplicate.  A long interval must be an integer multiple of
    the cadence; its empty slots are dropped frames.  Pixel equality never
    enters this calculation.
    """

    if pts_step <= 0:
        raise VideoDecodeError("nominal video PTS cadence is invalid")
    dropped = 0
    duplicate = 0
    cfr = True
    if not pts_values:
        return cfr, dropped, duplicate
    previous = pts_values[0]
    expected_next = previous + pts_step
    for current in pts_values[1:]:
        if current <= previous:
            raise VideoDecodeError(
                "video frame PTS values must be strictly increasing"
            )
        cfr = False
        if current < expected_next:
            duplicate += 1
            previous = current
            continue
        if current == expected_next:
            if duplicate == 0 and dropped == 0:
                cfr = True
            expected_next += pts_step
            previous = current
            continue
        quotient, remainder = divmod(current - expected_next, pts_step)
        if remainder != 0:
            raise VideoDecodeError(
                "video frame PTS is off the nominal cadence grid"
            )
        dropped += quotient
        expected_next = current + pts_step
        previous = current
    return cfr, dropped, duplicate


def _decode_all_frames(
    container: Any,
    stream: Any,
    metadata: VideoMetadata,
) -> tuple[
    tuple[int, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[bytes, ...],
    int,
    str,
]:
    frame_pts: list[int] = []
    frame_hashes: list[str] = []
    frame_hashes_without_last_row: list[str] = []
    frame_last_rows: list[bytes] = []
    decoded_bytes = 0
    pixel_digest = hashlib.sha256(_PIXEL_HASH_DOMAIN)
    expected_shape = (metadata.height, metadata.width, 3)
    per_frame_bytes = metadata.width * metadata.height * 3
    try:
        for frame in container.decode(stream):
            if len(frame_pts) >= MAX_VIDEO_FRAME_COUNT:
                raise VideoResourceLimitError(
                    "decoded frame count exceeds the inspection limit"
                )
            if frame.pts is None:
                raise VideoDecodeError("decoded video frame is missing PTS")
            pts = int(frame.pts)
            if frame_pts and pts <= frame_pts[-1]:
                raise VideoDecodeError(
                    "video frame PTS values must be strictly increasing"
                )
            if _safe_fraction(frame.time_base, "frame time base") != (
                metadata.time_base
            ):
                raise VideoMetadataMismatch(
                    "decoded frame time_base differs from manifest"
                )
            if (int(frame.width), int(frame.height)) != (
                metadata.width,
                metadata.height,
            ):
                raise VideoMetadataMismatch(
                    "decoded frame dimensions differ from manifest"
                )
            if str(frame.format.name) != metadata.pixel_format:
                raise VideoMetadataMismatch(
                    "decoded frame pixel_format differs from manifest"
                )
            if bool(frame.is_corrupt):
                raise VideoDecodeError("decoded video frame is marked corrupt")
            pixels = frame.to_ndarray(format=NORMALIZED_PIXEL_FORMAT)
            if pixels.shape != expected_shape or pixels.dtype != np.uint8:
                raise VideoDecodeError(
                    "decoded frame cannot be normalized to RGB24"
                )
            pixels = np.ascontiguousarray(pixels)
            if pixels.nbytes != per_frame_bytes:
                raise VideoDecodeError(
                    "decoded frame has an invalid normalized byte count"
                )
            decoded_bytes += pixels.nbytes
            if decoded_bytes > MAX_NORMALIZED_DECODE_BYTES:
                raise VideoResourceLimitError(
                    "decoded pixels exceed the inspection byte limit"
                )
            frame_hashes.append(canonical_rgb24_frame_sha256(pixels))
            pixels_without_last_row = np.ascontiguousarray(pixels[:-1])
            interior_digest = hashlib.sha256(_FRAME_PIXEL_HASH_DOMAIN)
            interior_digest.update(
                struct.pack(">Q", pixels_without_last_row.nbytes)
            )
            interior_digest.update(memoryview(pixels_without_last_row))
            frame_hashes_without_last_row.append(
                f"sha256:{interior_digest.hexdigest()}"
            )
            frame_last_rows.append(
                memoryview(np.ascontiguousarray(pixels[-1:])).tobytes()
            )
            pixel_digest.update(struct.pack(">Q", pixels.nbytes))
            pixel_digest.update(memoryview(pixels))
            frame_pts.append(pts)
    except PcVideoError:
        raise
    except (MemoryError, OSError, OverflowError, RecursionError):
        raise VideoDecodeError("video frame decoding failed safely") from None
    except Exception:
        # PyAV exposes FFmpeg failures through a version-specific exception
        # hierarchy.  Never propagate its text because it can contain a path.
        raise VideoDecodeError("video frame decoding failed") from None

    if not frame_pts:
        raise VideoDecodeError("video stream decoded no frames")
    return (
        tuple(frame_pts),
        tuple(frame_hashes),
        tuple(frame_hashes_without_last_row),
        tuple(frame_last_rows),
        decoded_bytes,
        f"sha256:{pixel_digest.hexdigest()}",
    )


def _dxgi_timeline_provenance(
    container: Any,
    *,
    frame_pts: tuple[int, ...],
) -> tuple[
    str | None,
    str | None,
    str | None,
    tuple[int, ...],
]:
    """Parse the encoder's canonical source-provenance tag when present."""

    try:
        raw_value = container.metadata.get(_DXGI_TIMELINE_TAG)
    except Exception:
        raise VideoMetadataMismatch(
            "video container metadata cannot be inspected"
        ) from None
    if raw_value is None:
        return None, None, None, ()
    if not isinstance(raw_value, str):
        raise VideoMetadataMismatch(
            "DXGI timeline metadata tag is not text"
        )
    try:
        data = _mapping(
            _json_loads_strict(raw_value, "DXGI timeline metadata"),
            "DXGI timeline metadata",
        )
        if set(data) != _DXGI_TIMELINE_KEYS:
            raise PcGoldenValidationError(
                "DXGI timeline metadata fields do not match v1"
            )
        canonical = json.dumps(
            data,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        if raw_value != canonical:
            raise PcGoldenValidationError(
                "DXGI timeline metadata is not canonical"
            )
        if (
            data["schema"] != _DXGI_TIMELINE_SCHEMA
            or _strict_int(data["version"], "DXGI timeline version")
            != _DXGI_TIMELINE_VERSION
            or data["frame_resampling"] != "none"
            or data["source_missed_presentations"] != 0
            or data["timestamp_quantization_collisions"] != 0
            or data["foreground_geometry_guard"] is not True
            or data["topmost_or_injected_overlay_detection"] is not False
            or data["requires_full_frame_visual_review"] is not True
        ):
            raise PcGoldenValidationError(
                "DXGI timeline violates the lossless provenance contract"
            )
        declared_pts = tuple(
            _strict_int(item, "DXGI timeline frame PTS")
            for item in data["frame_pts"]
        )
        if declared_pts != frame_pts:
            raise PcGoldenValidationError(
                "DXGI timeline frame PTS differ from decoded video"
            )
        present_ticks = tuple(
            _strict_int(
                item,
                "DXGI timeline present tick",
                minimum=0,
            )
            for item in data["source_present_ticks"]
        )
        if (
            len(present_ticks) != len(frame_pts)
            or any(
                current <= previous
                for previous, current in zip(
                    present_ticks,
                    present_ticks[1:],
                )
            )
            or present_ticks[0]
            != _strict_int(
                data["source_first_present_ticks"],
                "DXGI first present tick",
                minimum=0,
            )
        ):
            raise PcGoldenValidationError(
                "DXGI timeline present ticks are inconsistent"
            )
        return (
            _sha256(
                data["source_metadata_sha256"],
                "DXGI timeline source metadata SHA-256",
            ),
            _sha256(
                data["source_raw_sha256"],
                "DXGI timeline source raw SHA-256",
            ),
            _sha256(
                data["source_frames_csv_sha256"],
                "DXGI timeline source CSV SHA-256",
            ),
            present_ticks,
        )
    except (PcGoldenValidationError, TypeError, ValueError):
        raise VideoMetadataMismatch(
            "DXGI timeline metadata is invalid"
        ) from None


def inspect_pc_video(
    path: str | os.PathLike[str],
    manifest: PcGoldenManifest,
) -> PcVideoInspection:
    """Fully decode and validate the manifest's content-addressed video.

    The file is opened through one read-only, no-follow descriptor and is
    never modified.  All external-library errors are translated to controlled
    messages that do not reveal ``path``.
    """

    if not isinstance(manifest, PcGoldenManifest):
        raise VideoMetadataMismatch(
            "video inspection requires a validated PC golden manifest"
        )
    metadata = manifest.video
    try:
        artifact = manifest.artifacts[metadata.artifact]
    except (KeyError, OverflowError, RecursionError, TypeError):
        raise VideoMetadataMismatch(
            "manifest video artifact reference is invalid"
        ) from None
    return inspect_pc_video_artifact(
        path,
        metadata=metadata,
        artifact=artifact,
    )


def inspect_pc_video_artifact(
    path: str | os.PathLike[str],
    *,
    metadata: VideoMetadata,
    artifact: ArtifactSpec,
) -> PcVideoInspection:
    """Fully decode one primary or replay video artifact read-only."""

    if _av is None:
        raise VideoDecoderUnavailable(
            "PyAV 18 is unavailable; video evidence cannot be decoded"
        )
    validate_pc_video_artifact_declaration(metadata, artifact)
    file_object = _open_regular_read_only(path)
    try:
        try:
            opened_size = os.fstat(file_object.fileno()).st_size
        except (MemoryError, OSError, OverflowError, RecursionError, TypeError,
                ValueError):
            raise VideoArtifactError(
                "video artifact size cannot be inspected"
            ) from None
        if opened_size != artifact.bytes:
            raise VideoArtifactError("video artifact size differs from manifest")
        actual_bytes, actual_sha256 = _digest_open_file(file_object)
        if actual_bytes != artifact.bytes:
            raise VideoArtifactError("video artifact size differs from manifest")
        if actual_sha256 != artifact.sha256:
            raise VideoArtifactError("video artifact SHA-256 differs from manifest")

        try:
            container = _av.open(
                file_object,
                mode="r",
                options={
                    "err_detect": (
                        "explode+crccheck+bitstream+buffer+careful"
                    )
                },
            )
        except (MemoryError, OSError, OverflowError, RecursionError):
            raise VideoDecodeError("video container cannot be opened safely") from None
        except Exception:
            raise VideoDecodeError("video container cannot be opened") from None
        try:
            try:
                video_streams = tuple(container.streams.video)
            except (MemoryError, OverflowError, RecursionError):
                raise VideoDecodeError(
                    "video stream inventory cannot be read safely"
                ) from None
            except Exception:
                raise VideoDecodeError(
                    "video stream inventory cannot be read"
                ) from None
            if len(video_streams) != 1:
                raise VideoMetadataMismatch(
                    "video artifact must contain exactly one video stream"
                )
            stream = video_streams[0]
            (
                width,
                height,
                codec,
                pixel_format,
                time_base,
                nominal_fps,
            ) = _stream_header(stream)
            if (
                width > MAX_VIDEO_DIMENSION
                or height > MAX_VIDEO_DIMENSION
                or width * height > MAX_VIDEO_PIXELS_PER_FRAME
            ):
                raise VideoResourceLimitError(
                    "decoded video dimensions exceed the inspection limit"
                )
            mismatches = _header_mismatches(
                metadata,
                width=width,
                height=height,
                codec=codec,
                pixel_format=pixel_format,
                time_base=time_base,
                nominal_fps=nominal_fps,
            )
            if mismatches:
                raise VideoMetadataMismatch(
                    "video stream header differs from manifest fields: "
                    + ", ".join(mismatches)
                )
            (
                frame_pts,
                frame_pixel_sha256,
                frame_pixel_sha256_without_last_row,
                frame_last_row_rgb24,
                decoded_bytes,
                pixel_sha256,
            ) = _decode_all_frames(container, stream, metadata)
            (
                dxgi_source_metadata_sha256,
                dxgi_source_raw_sha256,
                dxgi_source_frames_csv_sha256,
                dxgi_source_present_ticks,
            ) = _dxgi_timeline_provenance(
                container,
                frame_pts=frame_pts,
            )
        finally:
            try:
                container.close()
            except Exception:
                # Closing is best effort; no write-capable handle was opened.
                pass

        frame_count = len(frame_pts)
        first_pts = frame_pts[0]
        last_pts = frame_pts[-1]
        if metadata.timing_mode == "qpc_vfr_pts_table":
            cfr = (
                len(frame_pts) <= 2
                or len(
                    {
                        current - previous
                        for previous, current in zip(
                            frame_pts,
                            frame_pts[1:],
                        )
                    }
                )
                == 1
            )
            dropped_frames = 0
            duplicate_frames = 0
        else:
            cfr, dropped_frames, duplicate_frames = _cadence_counts(
                frame_pts,
                pts_step=metadata.cfr_pts_step,
            )
        decoded_mismatches: list[str] = []
        if frame_count != metadata.frame_count:
            decoded_mismatches.append("frame_count")
        if first_pts != metadata.first_pts:
            decoded_mismatches.append("first_pts")
        if last_pts != metadata.last_pts:
            decoded_mismatches.append("last_pts")
        if cfr != metadata.cfr:
            decoded_mismatches.append("cfr")
        if dropped_frames != metadata.dropped_frames:
            decoded_mismatches.append("dropped_frames")
        if duplicate_frames != metadata.duplicate_frames:
            decoded_mismatches.append("duplicate_frames")
        if decoded_bytes != (
            metadata.width * metadata.height * 3 * metadata.frame_count
        ):
            decoded_mismatches.append("normalized_decoded_bytes")
        if decoded_mismatches:
            raise VideoMetadataMismatch(
                "decoded video differs from manifest fields: "
                + ", ".join(decoded_mismatches)
            )

        # Detect a file changed during inspection without ever opening it for
        # writing.  Size/identity revalidation is cheap and uses the same fd.
        final_bytes, final_sha256 = _digest_open_file(file_object)
        if final_bytes != actual_bytes or final_sha256 != actual_sha256:
            raise VideoArtifactError(
                "video artifact changed during read-only inspection"
            )
        return PcVideoInspection(
            artifact_sha256=actual_sha256,
            artifact_bytes=actual_bytes,
            width=width,
            height=height,
            codec=codec,
            pixel_format=pixel_format,
            time_base=time_base,
            nominal_fps=nominal_fps,
            frame_count=frame_count,
            first_pts=first_pts,
            last_pts=last_pts,
            frame_pts=frame_pts,
            cfr=cfr,
            dropped_frames=dropped_frames,
            duplicate_frames=duplicate_frames,
            normalized_pixel_format=NORMALIZED_PIXEL_FORMAT,
            normalized_decoded_bytes=decoded_bytes,
            decoded_pixel_sha256=pixel_sha256,
            frame_pixel_sha256=frame_pixel_sha256,
            frame_pixel_sha256_without_last_row=(
                frame_pixel_sha256_without_last_row
            ),
            frame_last_row_rgb24=frame_last_row_rgb24,
            dxgi_source_metadata_sha256=(
                dxgi_source_metadata_sha256
            ),
            dxgi_source_raw_sha256=dxgi_source_raw_sha256,
            dxgi_source_frames_csv_sha256=(
                dxgi_source_frames_csv_sha256
            ),
            dxgi_source_present_ticks=dxgi_source_present_ticks,
        )
    except PcVideoError:
        raise
    except (MemoryError, OSError, OverflowError, RecursionError):
        raise VideoDecodeError("video inspection failed safely") from None
    except Exception:
        raise VideoDecodeError("video inspection failed") from None
    finally:
        try:
            file_object.close()
        except Exception:
            pass


def decode_pc_video_rgb24_at_pts(
    path: str | os.PathLike[str],
    *,
    metadata: VideoMetadata,
    artifact: ArtifactSpec,
    requested_pts: tuple[int, ...],
) -> dict[int, bytes]:
    """Read up to 16 declared frames as canonical RGB24 byte strings."""

    if _av is None:
        raise VideoDecoderUnavailable(
            "PyAV 18 is unavailable; video evidence cannot be decoded"
        )
    if (
        not requested_pts
        or len(requested_pts) > 16
        or len(set(requested_pts)) != len(requested_pts)
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not metadata.first_pts <= value <= metadata.last_pts
            for value in requested_pts
        )
    ):
        raise VideoMetadataMismatch("requested video PTS set is invalid")
    validate_pc_video_artifact_declaration(metadata, artifact)
    requested = frozenset(requested_pts)
    file_object = _open_regular_read_only(path)
    try:
        opened_size = os.fstat(file_object.fileno()).st_size
        if opened_size != artifact.bytes:
            raise VideoArtifactError("video artifact size differs from manifest")
        actual_bytes, actual_sha256 = _digest_open_file(file_object)
        if (
            actual_bytes != artifact.bytes
            or actual_sha256 != artifact.sha256
        ):
            raise VideoArtifactError(
                "video artifact identity differs from manifest"
            )
        try:
            container = _av.open(
                file_object,
                mode="r",
                options={
                    "err_detect": (
                        "explode+crccheck+bitstream+buffer+careful"
                    )
                },
            )
        except Exception:
            raise VideoDecodeError("video container cannot be opened") from None
        try:
            streams = tuple(container.streams.video)
            if len(streams) != 1 or len(tuple(container.streams)) != 1:
                raise VideoMetadataMismatch(
                    "video artifact must contain exactly one video stream"
                )
            stream = streams[0]
            width, height, codec, pixel_format, time_base, nominal_fps = (
                _stream_header(stream)
            )
            mismatches = _header_mismatches(
                metadata,
                width=width,
                height=height,
                codec=codec,
                pixel_format=pixel_format,
                time_base=time_base,
                nominal_fps=nominal_fps,
            )
            if mismatches:
                raise VideoMetadataMismatch(
                    "video stream header differs from manifest fields: "
                    + ", ".join(mismatches)
                )
            found: dict[int, bytes] = {}
            for frame in container.decode(stream):
                pts = int(frame.pts)
                if pts not in requested:
                    continue
                if pts in found:
                    raise VideoDecodeError(
                        "requested video PTS occurs more than once"
                    )
                try:
                    pixels = np.ascontiguousarray(
                        frame.to_ndarray(format=NORMALIZED_PIXEL_FORMAT)
                    )
                except Exception:
                    raise VideoDecodeError(
                        "requested frame cannot be normalized"
                    ) from None
                if (
                    pixels.shape != (metadata.height, metadata.width, 3)
                    or pixels.dtype != np.uint8
                ):
                    raise VideoDecodeError(
                        "requested normalized frame layout is invalid"
                    )
                found[pts] = pixels.tobytes(order="C")
                if len(found) == len(requested):
                    break
            if set(found) != set(requested):
                raise VideoMetadataMismatch(
                    "requested video PTS is absent from decoded stream"
                )
        finally:
            try:
                container.close()
            except Exception:
                pass
        final_bytes, final_sha256 = _digest_open_file(file_object)
        if final_bytes != actual_bytes or final_sha256 != actual_sha256:
            raise VideoArtifactError(
                "video artifact changed during read-only inspection"
            )
        return found
    except PcVideoError:
        raise
    except (MemoryError, OSError, OverflowError, RecursionError):
        raise VideoDecodeError("requested frame decode failed safely") from None
    except Exception:
        raise VideoDecodeError("requested frame decode failed") from None
    finally:
        try:
            file_object.close()
        except Exception:
            pass


__all__ = [
    "MAX_NORMALIZED_DECODE_BYTES",
    "MAX_VIDEO_ARTIFACT_BYTES",
    "MAX_VIDEO_DIMENSION",
    "MAX_VIDEO_FRAME_COUNT",
    "MAX_VIDEO_PIXELS_PER_FRAME",
    "NORMALIZED_PIXEL_FORMAT",
    "PcVideoError",
    "PcVideoInspection",
    "VideoArtifactError",
    "VideoDecodeError",
    "VideoDecoderUnavailable",
    "VideoMetadataMismatch",
    "VideoResourceLimitError",
    "canonical_rgb24_frame_sha256",
    "decode_pc_video_rgb24_at_pts",
    "inspect_pc_video",
    "inspect_pc_video_artifact",
    "validate_pc_video_artifact_declaration",
    "validate_pc_video_declarations",
]
