"""Package one completed collector session as a PC Golden v4 case.

The packager is deliberately narrow.  It accepts the collector's two raw
DXGI runs, finds the longest contiguous native-update interval for which both
runs contain an exactly equal gameplay BGRA viewport, and emits a
self-contained case.  By default this remains the complete 800x600 frame.  A
narrow explicit contract may exclude only the final native raster row while
bounding every mismatch there.
The original raw acquisitions remain outside the case because the verifier's
non-video artifact budget is intentionally smaller than one 800x600 capture.
Their identities stay bound through canonical DXGI metadata and the FFV1
container provenance tag.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from zuma_rl.original_data import OriginalGameCatalog
from zuma_rl.pc_calibration import (
    CalibrationControlPoint,
    PcCalibrationSidecar,
)
from zuma_rl.pc_evidence import PcTickMap, TickPts
from zuma_rl.pc_golden import (
    DMO_FILE_ID,
    DMO_FORMAT,
    DMO_VERSION,
    ArtifactSpec,
    ComparisonContract,
    CoordinateCalibration,
    CoverageRange,
    CoverageStatus,
    FrameRef,
    GoldenInput,
    GoldenTick,
    InputTimeline,
    Measurement,
    MeasurementStatus,
    PcEnvironment,
    PcGoldenManifest,
    PcGoldenTrace,
    ReplayDeterminismContract,
    ReplayPixelComparisonContract,
    ReplayRunContract,
    RenderSettledReplayEvidence,
    SAVE_VOLATILE_REGISTRY_ROLES,
    SaveRunContract,
    SaveTransactionContract,
    Scenario,
    TickClock,
    VideoMetadata,
    capture_contract_fingerprint,
)
from zuma_rl.pc_memory_evidence import (
    MEMORY_CURVE_GEOMETRY_ARTIFACT,
    MEMORY_TRANSITION_CONTRACT_ARTIFACT,
    PcMemoryTransitionContract,
    build_memory_curve_geometry_binding,
    validate_probe_payloads,
    validate_shot_transition,
)
from zuma_rl.pc_protocol_evidence import (
    PcDxgiCaptureMetadata,
    PcFrameworkUpdateMap,
    PcStateSnapshot,
)
from zuma_rl.pc_render_settle import (
    PcRenderSettledUpdateMap,
    RENDER_SETTLE_METHOD,
    verify_render_settled_update_map,
)
from zuma_rl.popcap_dmo import DemoCommand, PopCapDemo
from zuma_rl.retail_dmo_provenance import (
    PROVENANCE_ARTIFACT,
    RAW_RECORDING_DMO_ARTIFACT,
    RECORDING_REPORT_ARTIFACT,
    RetailDmoProvenanceError,
    build_certifying_provenance,
    canonical_provenance_bytes,
)


CASE_SCHEMA = "zuma-rl.pc-golden-v4-package"
CASE_VERSION = 1
_FRAME_CSV_FIELDS = (
    "sequence",
    "present_ticks",
    "qpc_frequency",
    "host_perf_counter_ns",
    "accumulated_frames",
    "raw_offset",
    "raw_bytes",
    "frame_sha256",
)
# Keep the score glyphs and the readout face interior while excluding the
# animated antialiased boundary scanlines.  Independent exact replays changed
# only row 7 and row 42 of the former (0, 0, 130, 43) crop; every pixel in this
# half-open interior remained invariant while the rendered score stayed 7950.
_SCORE_CROP = (0, 8, 130, 42)


class PackagingError(RuntimeError):
    """The collector session cannot support a fail-closed Golden case."""


@dataclass(frozen=True, slots=True)
class FrameCandidate:
    sequence: int
    update: int
    pts: int
    frame_sha256: str
    comparison_sha256: str | None = None
    excluded_edge_bgra: bytes = b""


@dataclass(frozen=True, slots=True)
class MatchingTick:
    update: int
    r1: FrameCandidate
    r2: FrameCandidate
    excluded_edge_mismatch_count: int = 0


@dataclass(frozen=True, slots=True)
class VideoFacts:
    width: int
    height: int
    codec: str
    pixel_format: str
    time_base: tuple[int, int]
    nominal_fps: tuple[int, int]
    frame_pts: tuple[int, ...]
    cfr: bool


@dataclass(frozen=True, slots=True)
class MemoryEvidenceSources:
    before_probe: Path
    after_probe: Path
    before_pixel_validation: Path
    after_pixel_validation: Path
    transition_validation: Path
    expected_before_update: int
    expected_after_update: int
    expected_chain_distance_delta: float
    distance_tolerance: float
    minimum_chain_count: int
    minimum_visible: int
    maximum_mismatches: int
    minimum_visible_after_fired_bullets: int
    maximum_fired_bullet_mismatches: int
    video_binding_excluded_bottom_rows: int
    maximum_excluded_edge_mismatches: int


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, ValueError) as error:
        raise PackagingError(f"invalid JSON artifact: {path.name}") from error
    if not isinstance(value, dict):
        raise PackagingError(f"JSON artifact is not an object: {path.name}")
    return value


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _sha256_bytes(value: bytes | memoryview) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _artifact(path: Path, relative_path: str) -> ArtifactSpec:
    return ArtifactSpec(
        path=relative_path,
        sha256=_sha256_path(path),
        bytes=path.stat().st_size,
    )


def _copy_new(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise PackagingError(f"required source is missing: {source.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise PackagingError(f"destination already exists: {destination.name}")
    with source.open("rb") as input_stream:
        with destination.open("xb") as output_stream:
            shutil.copyfileobj(input_stream, output_stream, 4 * 1024 * 1024)
    if (
        destination.stat().st_size != source.stat().st_size
        or _sha256_path(destination) != _sha256_path(source)
    ):
        raise PackagingError(f"copied artifact changed: {source.name}")


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)


def _inspect_video(path: Path) -> VideoFacts:
    try:
        import av
    except ImportError as error:
        raise PackagingError("PyAV 18 is required to package FFV1") from error

    try:
        with av.open(str(path), mode="r") as container:
            streams = tuple(container.streams.video)
            if len(streams) != 1 or len(tuple(container.streams)) != 1:
                raise PackagingError("encoded capture has extra streams")
            stream = streams[0]
            context = stream.codec_context
            frame_pts = tuple(
                int(frame.pts) for frame in container.decode(stream)
            )
            if (
                not frame_pts
                or any(
                    current <= previous
                    for previous, current in zip(
                        frame_pts,
                        frame_pts[1:],
                    )
                )
            ):
                raise PackagingError("encoded frame PTS are not increasing")
            time_base = Fraction(stream.time_base)
            nominal_fps = Fraction(stream.average_rate)
            return VideoFacts(
                width=int(context.width),
                height=int(context.height),
                codec=str(context.name),
                pixel_format=str(context.format.name),
                time_base=(time_base.numerator, time_base.denominator),
                nominal_fps=(
                    nominal_fps.numerator,
                    nominal_fps.denominator,
                ),
                frame_pts=frame_pts,
                cfr=(
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
                ),
            )
    except PackagingError:
        raise
    except Exception as error:
        raise PackagingError("encoded capture could not be inspected") from error


def _stable_candidates(
    run_root: Path,
    *,
    video: VideoFacts,
    excluded_bottom_rows: int = 0,
    render_settled_update_map: PcRenderSettledUpdateMap | None = None,
) -> Mapping[int, tuple[FrameCandidate, ...]]:
    if excluded_bottom_rows not in (0, 1):
        raise PackagingError(
            "replay comparison may exclude only the final raster row"
        )
    update_map = PcFrameworkUpdateMap.read(
        run_root / "framework-updates.json"
    )
    csv_path = run_root / "capture" / "frames.csv"
    try:
        with csv_path.open("r", encoding="ascii", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != _FRAME_CSV_FIELDS:
                raise PackagingError("frame CSV header differs from v2")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise PackagingError("frame CSV could not be parsed") from error
    if (
        len(rows) != len(update_map.records)
        or len(rows) != len(video.frame_pts)
    ):
        raise PackagingError("video/CSV/framework map lengths differ")
    if (
        render_settled_update_map is not None
        and len(render_settled_update_map.records) != len(rows)
    ):
        raise PackagingError(
            "render-settled/video/framework map lengths differ"
        )

    raw_frames: np.memmap | None = None
    if excluded_bottom_rows:
        raw_path = run_root / "capture" / "frames.bgra.raw"
        frame_bytes = video.width * video.height * 4
        raw_size = raw_path.stat().st_size
        if raw_size != len(rows) * frame_bytes:
            raise PackagingError(
                "raw capture size differs from its complete frame timeline"
            )
        raw_frames = np.memmap(
            raw_path,
            dtype=np.uint8,
            mode="r",
            shape=(len(rows), video.height, video.width, 4),
        )

    grouped: dict[int, list[FrameCandidate]] = {}
    settled_records: Sequence[Any]
    if render_settled_update_map is None:
        settled_records = (None,) * len(rows)
    else:
        settled_records = render_settled_update_map.records
    for expected_sequence, (row, update_record, settled_record, pts) in enumerate(
        zip(
            rows,
            update_map.records,
            settled_records,
            video.frame_pts,
            strict=True,
        )
    ):
        if (
            int(row["sequence"]) != expected_sequence
            or update_record.sequence != expected_sequence
            or int(row["present_ticks"]) != update_record.present_ticks
        ):
            raise PackagingError("frame sequence identity differs")
        if settled_record is None:
            if update_record.update_before != update_record.update_after:
                continue
            update = update_record.update_before
        else:
            if (
                settled_record.sequence != expected_sequence
                or settled_record.present_ticks != update_record.present_ticks
                or settled_record.source_update_before
                != update_record.update_before
                or settled_record.source_update_after
                != update_record.update_after
            ):
                raise PackagingError(
                    "render-settled frame identity differs from source map"
                )
            update = settled_record.assigned_framework_update
        frame_sha256 = row["frame_sha256"]
        comparison_sha256 = frame_sha256
        excluded_edge_bgra = b""
        if raw_frames is not None:
            frame = raw_frames[expected_sequence]
            frame_payload = memoryview(
                np.ascontiguousarray(frame)
            ).cast("B")
            if _sha256_bytes(frame_payload) != frame_sha256:
                raise PackagingError(
                    "stable raw frame differs from its CSV hash"
                )
            comparison = np.ascontiguousarray(
                frame[: video.height - excluded_bottom_rows]
            )
            comparison_sha256 = _sha256_bytes(
                memoryview(comparison).cast("B")
            )
            excluded_edge_bgra = memoryview(
                np.ascontiguousarray(
                    frame[video.height - excluded_bottom_rows :]
                )
            ).tobytes()
        grouped.setdefault(update, []).append(
            FrameCandidate(
                sequence=expected_sequence,
                update=update,
                pts=pts,
                frame_sha256=frame_sha256,
                comparison_sha256=comparison_sha256,
                excluded_edge_bgra=excluded_edge_bgra,
            )
        )
    return {
        update: tuple(candidates)
        for update, candidates in grouped.items()
    }


def _matching_ticks(
    left: Mapping[int, tuple[FrameCandidate, ...]],
    right: Mapping[int, tuple[FrameCandidate, ...]],
    *,
    maximum_excluded_edge_mismatches: int = 0,
) -> tuple[MatchingTick, ...]:
    if maximum_excluded_edge_mismatches < 0:
        raise PackagingError("replay edge mismatch budget is invalid")

    def edge_mismatches(
        left_candidate: FrameCandidate,
        right_candidate: FrameCandidate,
    ) -> int | None:
        left_edge = left_candidate.excluded_edge_bgra
        right_edge = right_candidate.excluded_edge_bgra
        if len(left_edge) != len(right_edge) or len(left_edge) % 4:
            return None
        if not left_edge:
            return 0
        left_pixels = np.frombuffer(left_edge, dtype=np.uint8).reshape((-1, 4))
        right_pixels = np.frombuffer(right_edge, dtype=np.uint8).reshape((-1, 4))
        return int(np.any(left_pixels != right_pixels, axis=1).sum())

    matches: list[MatchingTick] = []
    for update in sorted(set(left) & set(right)):
        candidates = []
        for left_candidate in left[update]:
            for right_candidate in right[update]:
                left_comparison = (
                    left_candidate.comparison_sha256
                    or left_candidate.frame_sha256
                )
                right_comparison = (
                    right_candidate.comparison_sha256
                    or right_candidate.frame_sha256
                )
                if left_comparison != right_comparison:
                    continue
                mismatches = edge_mismatches(
                    left_candidate,
                    right_candidate,
                )
                if (
                    mismatches is None
                    or mismatches > maximum_excluded_edge_mismatches
                ):
                    continue
                candidates.append(
                    (
                        left_candidate.sequence,
                        right_candidate.sequence,
                        mismatches,
                        left_comparison,
                        left_candidate,
                        right_candidate,
                    )
                )
        candidates.sort(key=lambda value: value[:4])
        if candidates:
            _, _, mismatches, _, l, r = candidates[0]
            matches.append(
                MatchingTick(
                    update=update,
                    r1=l,
                    r2=r,
                    excluded_edge_mismatch_count=mismatches,
                )
            )
    return tuple(matches)


def _longest_contiguous(
    matches: tuple[MatchingTick, ...],
    *,
    minimum_ticks: int,
) -> tuple[MatchingTick, ...]:
    runs: list[list[MatchingTick]] = []
    for match in matches:
        if not runs or match.update != runs[-1][-1].update + 1:
            runs.append([match])
        else:
            runs[-1].append(match)
    if not runs:
        raise PackagingError("the two captures have no exact matching tick")
    selected = min(
        runs,
        key=lambda run: (-len(run), run[0].update),
    )
    if len(selected) < minimum_ticks:
        raise PackagingError(
            "the longest exact replay window is shorter than required"
        )
    for previous, current in zip(selected, selected[1:]):
        if (
            current.r1.sequence <= previous.r1.sequence
            or current.r2.sequence <= previous.r2.sequence
            or current.r1.pts <= previous.r1.pts
            or current.r2.pts <= previous.r2.pts
        ):
            raise PackagingError("selected frames are not monotonic")
    return tuple(selected)


def _window_inputs(
    demo: PopCapDemo,
    *,
    first_update: int,
    last_update: int,
    allow_idle_window: bool = False,
) -> tuple[DemoCommand, ...]:
    commands = tuple(
        command
        for command in demo.input_commands
        if first_update <= command.update <= last_update
    )
    mouse_buttons = [
        command
        for command in commands
        if command.kind == "mouse_button"
    ]
    if not mouse_buttons and allow_idle_window:
        return commands
    if (
        not any(command.payload.get("down") is True for command in mouse_buttons)
        or not any(
            command.payload.get("down") is False
            for command in mouse_buttons
        )
    ):
        raise PackagingError(
            "selected window does not contain a complete mouse-button edge"
        )
    return commands


def _select_packaging_window(
    matches: tuple[MatchingTick, ...],
    demo: PopCapDemo,
    *,
    minimum_ticks: int,
    allow_idle_window: bool,
    required_update_range: tuple[int, int] | None = None,
    select_exact_required_range: bool = False,
) -> tuple[tuple[MatchingTick, ...], tuple[DemoCommand, ...]]:
    """Choose an eligible exact run whose input edges are self-contained."""

    runs: list[list[MatchingTick]] = []
    for match in matches:
        if not runs or match.update != runs[-1][-1].update + 1:
            runs.append([match])
        else:
            runs[-1].append(match)
    eligible_by_length = [run for run in runs if len(run) >= minimum_ticks]
    if not eligible_by_length:
        raise PackagingError(
            "the longest exact replay window is shorter than required"
        )
    if required_update_range is not None:
        required_start, required_end = required_update_range
        eligible_by_length = [
            run
            for run in eligible_by_length
            if run[0].update <= required_start
            and run[-1].update >= required_end
        ]
        if not eligible_by_length:
            raise PackagingError(
                "no exact replay window contains the preregistered update "
                "range"
            )
    if select_exact_required_range:
        if required_update_range is None:
            raise PackagingError(
                "exact required-range selection needs a required range"
            )
        required_start, required_end = required_update_range
        required_count = required_end - required_start + 1
        if required_count < minimum_ticks:
            raise PackagingError(
                "exact required range is shorter than minimum ticks"
            )
        eligible_by_length = [
            [
                row
                for row in run
                if required_start <= row.update <= required_end
            ]
            for run in eligible_by_length
        ]
        if any(len(run) != required_count for run in eligible_by_length):
            raise PackagingError(
                "exact required range is not completely replay matched"
            )
    for run in sorted(
        eligible_by_length,
        key=lambda value: (-len(value), value[0].update),
    ):
        selected = tuple(run)
        try:
            inputs = _window_inputs(
                demo,
                first_update=selected[0].update,
                last_update=selected[-1].update,
                allow_idle_window=allow_idle_window,
            )
        except PackagingError:
            continue
        for previous, current in zip(selected, selected[1:]):
            if (
                current.r1.sequence <= previous.r1.sequence
                or current.r2.sequence <= previous.r2.sequence
                or current.r1.pts <= previous.r1.pts
                or current.r2.pts <= previous.r2.pts
            ):
                raise PackagingError("selected frames are not monotonic")
        return selected, inputs
    raise PackagingError(
        "no exact replay window satisfies the input-edge contract"
    )


def _verify_selected_raw_frames(
    session_root: Path,
    selected: tuple[MatchingTick, ...],
    *,
    width: int,
    height: int,
    excluded_bottom_rows: int = 0,
    maximum_excluded_edge_mismatches: int = 0,
) -> str:
    if excluded_bottom_rows not in (0, 1):
        raise PackagingError(
            "replay comparison may exclude only the final raster row"
        )
    if maximum_excluded_edge_mismatches < 0:
        raise PackagingError("replay edge mismatch budget is invalid")
    if excluded_bottom_rows == 0 and maximum_excluded_edge_mismatches != 0:
        raise PackagingError(
            "replay edge mismatch budget requires an excluded raster row"
        )
    raw_paths = (
        session_root / "run-r1" / "capture" / "frames.bgra.raw",
        session_root / "run-r2" / "capture" / "frames.bgra.raw",
    )
    frame_bytes = width * height * 4
    arrays: list[np.memmap] = []
    for path in raw_paths:
        raw_size = path.stat().st_size
        if raw_size == 0 or raw_size % frame_bytes:
            raise PackagingError("raw capture size is not frame aligned")
        arrays.append(
            np.memmap(
                path,
                dtype=np.uint8,
                mode="r",
                shape=(raw_size // frame_bytes, height, width, 4),
            )
        )
    crop_hashes: set[str] = set()
    x0, y0, x1, y1 = _SCORE_CROP
    for match in selected:
        frames = (
            arrays[0][match.r1.sequence],
            arrays[1][match.r2.sequence],
        )
        expected_hashes = (
            match.r1.frame_sha256,
            match.r2.frame_sha256,
        )
        for frame, expected in zip(frames, expected_hashes, strict=True):
            payload = memoryview(np.ascontiguousarray(frame)).cast("B")
            if _sha256_bytes(payload) != expected:
                raise PackagingError(
                    "selected raw frame differs from its CSV hash"
                )
            crop = np.ascontiguousarray(frame[y0:y1, x0:x1])
            crop_hashes.add(
                _sha256_bytes(memoryview(crop).cast("B"))
            )
        difference = np.any(frames[0] != frames[1], axis=2)
        interior_end = height - excluded_bottom_rows
        interior_mismatches = int(difference[:interior_end].sum())
        edge_mismatches = int(difference[interior_end:].sum())
        if interior_mismatches:
            raise PackagingError(
                "selected replay frames differ inside gameplay viewport"
            )
        if edge_mismatches > maximum_excluded_edge_mismatches:
            raise PackagingError(
                "selected replay frames exceed edge mismatch budget"
            )
        if edge_mismatches != match.excluded_edge_mismatch_count:
            raise PackagingError(
                "selected replay edge mismatch count changed after selection"
            )
    if len(crop_hashes) != 1:
        raise PackagingError(
            "score annotation crop changes inside the selected window"
        )
    return next(iter(crop_hashes))


def _verify_memory_video_binding(
    session_root: Path,
    selected: tuple[MatchingTick, ...],
    *,
    before_update: int,
    before_frame: Path,
    after_update: int,
    after_frame: Path,
    excluded_bottom_rows: int,
    maximum_excluded_edge_mismatches: int,
) -> tuple[Mapping[str, Any], ...]:
    try:
        from PIL import Image
    except ImportError as error:
        raise PackagingError(
            "Pillow is required for memory/video pixel binding"
        ) from error
    if not 0 <= excluded_bottom_rows <= 4:
        raise PackagingError("memory/video edge exclusion is invalid")
    if not (
        0
        <= maximum_excluded_edge_mismatches
        <= 800 * excluded_bottom_rows
    ):
        raise PackagingError("memory/video edge mismatch budget is invalid")
    by_update = {match.update: match for match in selected}
    raw_path = (
        session_root / "run-r1" / "capture" / "frames.bgra.raw"
    )
    frame_bytes = 800 * 600 * 4
    raw_size = raw_path.stat().st_size
    if raw_size % frame_bytes:
        raise PackagingError("raw capture size is not frame aligned")
    frame_count = raw_size // frame_bytes
    raw = np.memmap(
        raw_path,
        dtype=np.uint8,
        mode="r",
        shape=(frame_count, 600, 800, 4),
    )
    rows: list[Mapping[str, Any]] = []
    for label, update, frame_path in (
        ("before", before_update, before_frame),
        ("after", after_update, after_frame),
    ):
        match = by_update.get(update)
        if match is None:
            raise PackagingError(
                f"memory {label} update is outside selected video ticks"
            )
        with Image.open(frame_path) as source:
            if source.format != "BMP" or source.size != (800, 600):
                raise PackagingError(
                    f"memory {label} frame is not native BMP"
                )
            probe_rgb = np.asarray(source.convert("RGB"))
        video_rgb = np.ascontiguousarray(
            raw[match.r1.sequence][:, :, [2, 1, 0]]
        )
        difference = np.any(probe_rgb != video_rgb, axis=2)
        interior_end = 600 - excluded_bottom_rows
        interior_mismatches = int(difference[:interior_end].sum())
        excluded_mismatches = int(difference[interior_end:].sum())
        if interior_mismatches != 0:
            raise PackagingError(
                f"memory {label} frame differs inside gameplay viewport"
            )
        if excluded_mismatches > maximum_excluded_edge_mismatches:
            raise PackagingError(
                f"memory {label} frame exceeds edge mismatch budget"
            )
        rows.append(
            {
                "phase": label,
                "framework_update": update,
                "native_tick": update - selected[0].update,
                "video_frame_sequence": match.r1.sequence,
                "video_pts": match.r1.pts,
                "interior_height": interior_end,
                "interior_mismatch_count": interior_mismatches,
                "excluded_bottom_rows": excluded_bottom_rows,
                "excluded_edge_mismatch_count": excluded_mismatches,
                "probe_rgb24_sha256": _sha256_bytes(
                    memoryview(np.ascontiguousarray(probe_rgb)).cast("B")
                ),
                "video_rgb24_sha256": _sha256_bytes(
                    memoryview(video_rgb).cast("B")
                ),
                "interior_rgb24_sha256": _sha256_bytes(
                    memoryview(
                        np.ascontiguousarray(probe_rgb[:interior_end])
                    ).cast("B")
                ),
            }
        )
    return tuple(rows)


def _identity_calibration() -> PcCalibrationSidecar:
    points = tuple(
        CalibrationControlPoint(raw=(x, y), logical=(x, y))
        for x in (0.0, 399.0, 799.0)
        for y in (0.0, 299.0, 599.0)
    )
    provisional = PcCalibrationSidecar(
        raw_width=800,
        raw_height=600,
        logical_width=800,
        logical_height=600,
        transform_kind="axis_aligned_affine",
        pixel_center_convention="center_at_integer",
        control_points=points,
        logical_from_raw=(
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        rms_error_px=0.0,
        max_error_px=0.0,
    )
    fit = provisional.recompute()
    return PcCalibrationSidecar(
        raw_width=800,
        raw_height=600,
        logical_width=800,
        logical_height=600,
        transform_kind="axis_aligned_affine",
        pixel_center_convention="center_at_integer",
        control_points=points,
        logical_from_raw=fit.logical_from_raw,
        rms_error_px=fit.rms_error_px,
        max_error_px=fit.max_error_px,
    )


def _video_metadata(
    artifact_name: str,
    facts: VideoFacts,
) -> VideoMetadata:
    return VideoMetadata(
        artifact=artifact_name,
        width=facts.width,
        height=facts.height,
        codec=facts.codec,
        pixel_format=facts.pixel_format,
        time_base=facts.time_base,
        nominal_fps=facts.nominal_fps,
        frame_count=len(facts.frame_pts),
        first_pts=facts.frame_pts[0],
        last_pts=facts.frame_pts[-1],
        cfr=facts.cfr,
        dropped_frames=0,
        duplicate_frames=0,
        timing_mode="qpc_vfr_pts_table",
    )


def _trace(
    *,
    case_id: str,
    environment_fingerprint: str,
    contract_fingerprint: str,
    selected: tuple[MatchingTick, ...],
    run_id: str,
    inputs: tuple[DemoCommand, ...],
    score: int,
) -> PcGoldenTrace:
    first_update = selected[0].update
    inputs_by_tick: dict[int, list[DemoCommand]] = {}
    for command in inputs:
        inputs_by_tick.setdefault(
            command.update - first_update,
            [],
        ).append(command)
    records: list[GoldenTick] = []
    for tick, match in enumerate(selected):
        frame = match.r1 if run_id == "r1" else match.r2
        commands = sorted(
            inputs_by_tick.get(tick, ()),
            key=lambda command: command.sequence,
        )
        records.append(
            GoldenTick(
                tick=tick,
                frames=(
                    FrameRef(
                        frame_index=frame.sequence,
                        pts=frame.pts,
                    ),
                ),
                inputs=tuple(
                    GoldenInput(
                        sequence=index,
                        kind=command.kind,
                        pts=frame.pts,
                        payload={
                            "dmo_sequence": command.sequence,
                            "dmo_update": command.update,
                        },
                    )
                    for index, command in enumerate(commands)
                ),
                events=(),
                measurements={
                    "score": Measurement(
                        status=MeasurementStatus.OBSERVED,
                        value=score,
                        source="frame_annotation",
                        uncertainty=0.0,
                    )
                },
            )
        )
    return PcGoldenTrace(
        case_id=case_id,
        pc_environment_fingerprint=environment_fingerprint,
        capture_contract_fingerprint=contract_fingerprint,
        records=tuple(records),
    )


def package_case(
    *,
    session_root: Path,
    original_root: Path,
    output_root: Path,
    level_id: str,
    observed_score: int,
    os_build: str,
    driver: str,
    minimum_ticks: int,
    case_id: str | None,
    memory_evidence: MemoryEvidenceSources | None = None,
    allow_idle_input_window: bool = False,
    required_update_range: tuple[int, int] | None = None,
    select_exact_required_range: bool = False,
    replay_excluded_bottom_rows: int = 0,
    replay_maximum_excluded_edge_mismatches: int = 0,
    retail_recording_dmo: Path | None = None,
    retail_recording_report: Path | None = None,
    render_settle_calibration_preregistration: Path | None = None,
    render_settle_calibration_execution_binding: Path | None = None,
    render_settle_calibration_holdout_report: Path | None = None,
) -> Mapping[str, Any]:
    if (
        required_update_range is not None
        and required_update_range[1] < required_update_range[0]
    ):
        raise PackagingError("required update range is reversed")
    if select_exact_required_range and required_update_range is None:
        raise PackagingError(
            "exact required-range selection needs both required updates"
        )
    if (retail_recording_dmo is None) != (retail_recording_report is None):
        raise PackagingError(
            "retail recording DMO and recording report must be supplied "
            "together"
        )
    render_settle_sources = (
        render_settle_calibration_preregistration,
        render_settle_calibration_execution_binding,
        render_settle_calibration_holdout_report,
    )
    render_settle_enabled = any(
        value is not None for value in render_settle_sources
    )
    if render_settle_enabled and not all(
        value is not None for value in render_settle_sources
    ):
        raise PackagingError(
            "all three render-settle calibration artifacts are required"
        )
    if replay_excluded_bottom_rows not in (0, 1):
        raise PackagingError(
            "replay comparison may exclude only the final raster row"
        )
    if replay_excluded_bottom_rows == 0:
        if replay_maximum_excluded_edge_mismatches != 0:
            raise PackagingError(
                "replay edge mismatch budget requires an excluded raster row"
            )
    elif replay_maximum_excluded_edge_mismatches <= 0:
        raise PackagingError(
            "excluded replay raster row requires a positive mismatch budget"
        )
    session_root = session_root.resolve()
    original_root = original_root.resolve()
    output_root = output_root.resolve()
    if retail_recording_dmo is not None:
        retail_recording_dmo = retail_recording_dmo.resolve()
        retail_recording_report = retail_recording_report.resolve()
        if not retail_recording_dmo.is_file():
            raise PackagingError("raw retail recording DMO does not exist")
        if not retail_recording_report.is_file():
            raise PackagingError("retail recording report does not exist")
    if render_settle_enabled:
        assert render_settle_calibration_preregistration is not None
        assert render_settle_calibration_execution_binding is not None
        assert render_settle_calibration_holdout_report is not None
        render_settle_calibration_preregistration = (
            render_settle_calibration_preregistration.resolve()
        )
        render_settle_calibration_execution_binding = (
            render_settle_calibration_execution_binding.resolve()
        )
        render_settle_calibration_holdout_report = (
            render_settle_calibration_holdout_report.resolve()
        )
        if not all(
            path.is_file()
            for path in (
                render_settle_calibration_preregistration,
                render_settle_calibration_execution_binding,
                render_settle_calibration_holdout_report,
            )
        ):
            raise PackagingError(
                "a render-settle calibration artifact is missing"
            )
    part_root = output_root.with_name(output_root.name + ".building")
    if output_root.exists() or part_root.exists():
        raise PackagingError("output or building directory already exists")
    part_root.parent.mkdir(parents=True, exist_ok=True)
    part_root.mkdir()

    collection = _read_json(session_root / "collection.json")
    plan = _read_json(session_root / "plan.json")
    if (
        collection.get("status") != "complete"
        or collection.get("version") != 5
        or plan.get("version") != 5
        or collection.get("session_nonce") != plan.get("session_nonce")
    ):
        raise PackagingError("collector session is not a complete v5 result")

    videos = {
        run_id: _inspect_video(
            session_root / f"run-{run_id}" / "capture.ffv1.mkv"
        )
        for run_id in ("r1", "r2")
    }
    if any(
        (facts.width, facts.height) != (800, 600)
        for facts in videos.values()
    ):
        raise PackagingError("capture is not the native 800x600 viewport")
    settled_maps: dict[str, PcRenderSettledUpdateMap] = {}
    if render_settle_enabled:
        assert render_settle_calibration_preregistration is not None
        assert render_settle_calibration_execution_binding is not None
        assert render_settle_calibration_holdout_report is not None
        for run_id in ("r1", "r2"):
            run_root = session_root / f"run-{run_id}"
            settled_path = run_root / "render-settled-updates.json"
            try:
                expected = PcRenderSettledUpdateMap.read(settled_path)
                settled_maps[run_id] = verify_render_settled_update_map(
                    expected,
                    capture_metadata_path=(
                        run_root / "capture" / "metadata.json"
                    ),
                    frames_csv_path=(
                        run_root / "capture" / "frames.csv"
                    ),
                    framework_update_map_path=(
                        run_root / "framework-updates.json"
                    ),
                    framework_state_sidecar_path=(
                        run_root / "framework-state-diagnostic.json"
                    ),
                    framework_poll_path=(
                        run_root / "framework-state-poll-diagnostic.json"
                    ),
                    calibration_preregistration_path=(
                        render_settle_calibration_preregistration
                    ),
                    calibration_execution_binding_path=(
                        render_settle_calibration_execution_binding
                    ),
                    calibration_holdout_report_path=(
                        render_settle_calibration_holdout_report
                    ),
                )
            except (OSError, ValueError) as error:
                raise PackagingError(
                    f"render-settled {run_id} evidence is invalid: {error}"
                ) from error
    candidates = {
        run_id: _stable_candidates(
            session_root / f"run-{run_id}",
            video=videos[run_id],
            excluded_bottom_rows=replay_excluded_bottom_rows,
            render_settled_update_map=settled_maps.get(run_id),
        )
        for run_id in ("r1", "r2")
    }
    all_matches = _matching_ticks(
        candidates["r1"],
        candidates["r2"],
        maximum_excluded_edge_mismatches=(
            replay_maximum_excluded_edge_mismatches
        ),
    )
    demo = PopCapDemo.read(session_root / "inputs" / "input.dmo")
    selected, window_inputs = _select_packaging_window(
        all_matches,
        demo,
        minimum_ticks=minimum_ticks,
        allow_idle_window=allow_idle_input_window,
        required_update_range=required_update_range,
        select_exact_required_range=select_exact_required_range,
    )
    crop_sha256 = _verify_selected_raw_frames(
        session_root,
        selected,
        width=800,
        height=600,
        excluded_bottom_rows=replay_excluded_bottom_rows,
        maximum_excluded_edge_mismatches=(
            replay_maximum_excluded_edge_mismatches
        ),
    )

    catalog = OriginalGameCatalog(original_root)
    loaded = catalog.load_level(level_id, hard=False)
    if len(loaded.curves) != 1:
        raise PackagingError("packager currently requires one original curve")
    runtime_hashes = {
        PcDxgiCaptureMetadata.read(
            session_root
            / f"run-{run_id}"
            / "capture"
            / "metadata.json"
        ).executable_sha256
        for run_id in ("r1", "r2")
    }
    if len(runtime_hashes) != 1:
        raise PackagingError("capture runtime identities differ")
    runtime_sha256 = next(iter(runtime_hashes))
    launcher_sha256 = _sha256_path(original_root / "ZumasRevenge.exe")
    main_pak_sha256 = _sha256_path(original_root / "main.pak")
    levels_xml_sha256 = _sha256_bytes(
        catalog.archive.read_member(r"levels\levels.xml")
    )
    curve_hashes = {
        str(curve.source_path.relative_to(original_root)).replace("\\", "/"):
        _sha256_path(curve.source_path)
        for curve in loaded.curves
        if curve.source_path is not None
    }
    if len(curve_hashes) != len(loaded.curves):
        raise PackagingError("original curve identities are incomplete")
    runtime_plan = plan["runtime"]
    if (
        runtime_plan["runtime_executable_sha256"] != runtime_sha256
        or runtime_plan["runtime_source_sha256"] != launcher_sha256
        or plan["dmo"]["sha256"] != demo.artifact_sha256
    ):
        raise PackagingError("collector plan identities differ from sources")

    dmo_provenance: Mapping[str, Any] | None = None
    dmo_provenance_bytes: bytes | None = None
    if retail_recording_dmo is not None:
        try:
            dmo_provenance = build_certifying_provenance(
                source_data=retail_recording_dmo.read_bytes(),
                output_data=(session_root / "inputs" / "input.dmo").read_bytes(),
                recording_report_data=retail_recording_report.read_bytes(),
                collector_plan_data=(session_root / "plan.json").read_bytes(),
                expected_runtime_sha256=runtime_sha256,
            )
            dmo_provenance_bytes = canonical_provenance_bytes(
                dmo_provenance
            )
        except RetailDmoProvenanceError as error:
            raise PackagingError(
                "raw retail recording provenance is not certifying"
            ) from error

    pre_snapshot = PcStateSnapshot.read(
        session_root / "protocol" / "pre.json"
    )
    capture_metadata = {
        run_id: PcDxgiCaptureMetadata.read(
            session_root
            / f"run-{run_id}"
            / "capture"
            / "metadata.json"
        )
        for run_id in ("r1", "r2")
    }
    gpu_names = {
        str(metadata.raw_mapping["source_identity"]["adapter_description"])
        for metadata in capture_metadata.values()
    }
    if len(gpu_names) != 1:
        raise PackagingError("capture adapters differ")

    final_case_id = case_id or (
        f"{loaded.definition.id.casefold()}_dmo_"
        f"{demo.artifact_sha256[7:19]}_"
        f"u{selected[0].update}_{selected[-1].update}"
    )
    artifacts: dict[str, ArtifactSpec] = {}

    def copy_artifact(
        name: str,
        source: Path,
        relative_path: str,
    ) -> None:
        destination = part_root / relative_path
        _copy_new(source, destination)
        artifacts[name] = _artifact(destination, relative_path)

    copy_artifact(
        "input.dmo",
        session_root / "inputs" / "input.dmo",
        "input.dmo",
    )
    if retail_recording_dmo is not None:
        copy_artifact(
            RAW_RECORDING_DMO_ARTIFACT,
            retail_recording_dmo,
            "evidence/input-recording.dmo",
        )
        copy_artifact(
            RECORDING_REPORT_ARTIFACT,
            retail_recording_report,
            "evidence/input-recording-report.json",
        )
        provenance_path = part_root / "evidence" / "dmo-provenance.json"
        _write_new(provenance_path, dmo_provenance_bytes)
        artifacts[PROVENANCE_ARTIFACT] = _artifact(
            provenance_path,
            "evidence/dmo-provenance.json",
        )
    for run_id in ("r1", "r2"):
        copy_artifact(
            f"video.{run_id}",
            session_root / f"run-{run_id}" / "capture.ffv1.mkv",
            f"capture-{run_id}.mkv",
        )
        copy_artifact(
            f"capture.metadata.{run_id}",
            session_root
            / f"run-{run_id}"
            / "capture"
            / "metadata.json",
            f"evidence/capture-metadata-{run_id}.json",
        )
        copy_artifact(
            f"capture.frames_csv.{run_id}",
            session_root
            / f"run-{run_id}"
            / "capture"
            / "frames.csv",
            f"evidence/frames-{run_id}.csv",
        )
        copy_artifact(
            f"framework.updates.{run_id}",
            session_root / f"run-{run_id}" / "framework-updates.json",
            f"evidence/framework-updates-{run_id}.json",
        )
        if render_settle_enabled:
            copy_artifact(
                f"render.settled.updates.{run_id}",
                session_root
                / f"run-{run_id}"
                / "render-settled-updates.json",
                f"evidence/render-settled-updates-{run_id}.json",
            )
            copy_artifact(
                f"framework.poll.{run_id}",
                session_root
                / f"run-{run_id}"
                / "framework-state-poll-diagnostic.json",
                f"evidence/framework-poll-{run_id}.json",
            )
            copy_artifact(
                f"framework.state.{run_id}",
                session_root
                / f"run-{run_id}"
                / "framework-state-diagnostic.json",
                f"evidence/framework-state-{run_id}.json",
            )
        copy_artifact(
            f"strict.replay.{run_id}",
            session_root / f"run-{run_id}" / "strict-replay.json",
            f"evidence/strict-replay-{run_id}.json",
        )
        copy_artifact(
            f"window.repaint.{run_id}",
            session_root / f"run-{run_id}" / "window-repaint.json",
            f"evidence/window-repaint-{run_id}.json",
        )
    if render_settle_enabled:
        assert render_settle_calibration_preregistration is not None
        assert render_settle_calibration_execution_binding is not None
        assert render_settle_calibration_holdout_report is not None
        copy_artifact(
            "render.settle.calibration.preregistration",
            render_settle_calibration_preregistration,
            "evidence/render-settle-calibration-preregistration.json",
        )
        copy_artifact(
            "render.settle.calibration.execution_binding",
            render_settle_calibration_execution_binding,
            "evidence/render-settle-calibration-execution-binding.json",
        )
        copy_artifact(
            "render.settle.calibration.holdout_report",
            render_settle_calibration_holdout_report,
            "evidence/render-settle-calibration-holdout-report.json",
        )
    copy_artifact(
        "collector.plan",
        session_root / "plan.json",
        "evidence/collector-plan.json",
    )
    copy_artifact(
        "collector.result",
        session_root / "collection.json",
        "evidence/collector-result.json",
    )
    copy_artifact(
        "save.journal",
        session_root / "protocol" / "save-journal.ndjson",
        "evidence/save-journal.ndjson",
    )
    copy_artifact(
        "process.timeline",
        session_root / "protocol" / "process-timeline.bin",
        "evidence/process-timeline.bin",
    )
    snapshot_sources = {
        "save.pre": session_root / "protocol" / "pre.json",
        "save.r1.start": session_root / "run-r1" / "start.json",
        "save.r1.end": session_root / "run-r1" / "end.json",
        "save.r2.start": session_root / "run-r2" / "start.json",
        "save.r2.end": session_root / "run-r2" / "end.json",
        "save.restored": session_root / "protocol" / "restored.json",
    }
    for name, source in snapshot_sources.items():
        copy_artifact(
            name,
            source,
            "evidence/state/" + name.removeprefix("save.") + ".json",
        )

    memory_video_binding: tuple[Mapping[str, Any], ...] = ()
    if memory_evidence is not None:
        before_probe = memory_evidence.before_probe.resolve()
        after_probe = memory_evidence.after_probe.resolve()
        before_pixel = (
            memory_evidence.before_pixel_validation.resolve()
        )
        after_pixel = memory_evidence.after_pixel_validation.resolve()
        transition_validation = (
            memory_evidence.transition_validation.resolve()
        )
        if not (
            selected[0].update
            <= memory_evidence.expected_before_update
            < memory_evidence.expected_after_update
            <= selected[-1].update
        ):
            raise PackagingError(
                "memory transition updates are outside selected video range"
            )
        recomputed_transition = validate_shot_transition(
            before_probe,
            after_probe,
            before_pixel,
            after_pixel,
            expected_before_update=(
                memory_evidence.expected_before_update
            ),
            expected_after_update=(
                memory_evidence.expected_after_update
            ),
            expected_score=observed_score,
            expected_chain_distance_delta=(
                memory_evidence.expected_chain_distance_delta
            ),
            distance_tolerance=memory_evidence.distance_tolerance,
            minimum_chain_count=memory_evidence.minimum_chain_count,
            minimum_visible=memory_evidence.minimum_visible,
            maximum_mismatches=memory_evidence.maximum_mismatches,
            minimum_visible_after_fired_bullets=(
                memory_evidence.minimum_visible_after_fired_bullets
            ),
            maximum_fired_bullet_mismatches=(
                memory_evidence.maximum_fired_bullet_mismatches
            ),
            expected_runtime_sha256=runtime_sha256,
            expected_dmo_sha256=demo.artifact_sha256,
        )
        if _read_json(transition_validation) != recomputed_transition:
            raise PackagingError(
                "stored memory transition report is not canonical"
            )
        before_raw = validate_probe_payloads(
            before_probe,
            expected_runtime_sha256=runtime_sha256,
            expected_dmo_sha256=demo.artifact_sha256,
            expected_update=memory_evidence.expected_before_update,
            expected_score=observed_score,
            require_freeze_state=True,
        )
        after_raw = validate_probe_payloads(
            after_probe,
            expected_runtime_sha256=runtime_sha256,
            expected_dmo_sha256=demo.artifact_sha256,
            expected_update=memory_evidence.expected_after_update,
            expected_score=observed_score,
            require_freeze_state=True,
        )
        curve_geometry_binding = build_memory_curve_geometry_binding(
            case_id=final_case_id,
            level_id=loaded.definition.id,
            hard=False,
            curve_index=0,
            curve=loaded.curves[0],
            phases=(
                (
                    "before",
                    memory_evidence.expected_before_update,
                    before_raw["active_chain"],
                ),
                (
                    "after",
                    memory_evidence.expected_after_update,
                    after_raw["active_chain"],
                ),
            ),
        )
        memory_video_binding = _verify_memory_video_binding(
            session_root,
            selected,
            before_update=memory_evidence.expected_before_update,
            before_frame=before_raw["frozen_frame_path"],
            after_update=memory_evidence.expected_after_update,
            after_frame=after_raw["frozen_frame_path"],
            excluded_bottom_rows=(
                memory_evidence.video_binding_excluded_bottom_rows
            ),
            maximum_excluded_edge_mismatches=(
                memory_evidence.maximum_excluded_edge_mismatches
            ),
        )

        def copy_memory_phase(
            phase: str,
            probe_path: Path,
            pixel_path: Path,
            raw: Mapping[str, Any],
        ) -> tuple[str, str]:
            probe_key = f"memory.{phase}.probe"
            pixel_key = f"memory.{phase}.pixel_validation"
            relative_root = f"evidence/memory/{phase}"
            copy_artifact(
                probe_key,
                probe_path,
                f"{relative_root}/{probe_path.name}",
            )
            for index, source in enumerate(
                sorted(
                    raw["used_artifact_paths"],
                    key=lambda path: path.name,
                )
            ):
                copy_artifact(
                    f"memory.{phase}.raw.{index:02d}.{source.name}",
                    source,
                    f"{relative_root}/{source.name}",
                )
            copy_artifact(
                pixel_key,
                pixel_path,
                f"{relative_root}/{pixel_path.name}",
            )
            return probe_key, pixel_key

        before_probe_key, before_pixel_key = copy_memory_phase(
            "before",
            before_probe,
            before_pixel,
            before_raw,
        )
        after_probe_key, after_pixel_key = copy_memory_phase(
            "after",
            after_probe,
            after_pixel,
            after_raw,
        )
        curve_binding_path = (
            part_root
            / "evidence"
            / "memory"
            / "curve-geometry-binding.json"
        )
        _write_new(
            curve_binding_path,
            _canonical_json(curve_geometry_binding),
        )
        artifacts[MEMORY_CURVE_GEOMETRY_ARTIFACT] = _artifact(
            curve_binding_path,
            "evidence/memory/curve-geometry-binding.json",
        )
        transition_key = "memory.transition_validation"
        copy_artifact(
            transition_key,
            transition_validation,
            (
                "evidence/memory/"
                + transition_validation.name
            ),
        )
        binding_key = "memory.video_binding"
        binding_report = {
            "schema": "zuma-rl.pc-memory-video-binding",
            "version": 1,
            "status": "PASS",
            "case_id": final_case_id,
            "primary_video_artifact": "video.r1",
            "selected_first_update": selected[0].update,
            "excluded_bottom_rows": (
                memory_evidence.video_binding_excluded_bottom_rows
            ),
            "maximum_excluded_edge_mismatches": (
                memory_evidence.maximum_excluded_edge_mismatches
            ),
            "bindings": list(memory_video_binding),
        }
        binding_path = (
            part_root
            / "evidence"
            / "memory"
            / "video-binding.json"
        )
        _write_new(binding_path, _canonical_json(binding_report))
        artifacts[binding_key] = _artifact(
            binding_path,
            "evidence/memory/video-binding.json",
        )
        memory_contract = PcMemoryTransitionContract(
            case_id=final_case_id,
            before_probe_artifact=before_probe_key,
            after_probe_artifact=after_probe_key,
            before_pixel_validation_artifact=before_pixel_key,
            after_pixel_validation_artifact=after_pixel_key,
            transition_validation_artifact=transition_key,
            video_binding_validation_artifact=binding_key,
            expected_before_update=(
                memory_evidence.expected_before_update
            ),
            expected_after_update=(
                memory_evidence.expected_after_update
            ),
            expected_score=observed_score,
            expected_chain_distance_delta=(
                memory_evidence.expected_chain_distance_delta
            ),
            distance_tolerance=memory_evidence.distance_tolerance,
            minimum_chain_count=memory_evidence.minimum_chain_count,
            minimum_visible=memory_evidence.minimum_visible,
            maximum_mismatches=memory_evidence.maximum_mismatches,
            minimum_visible_after_fired_bullets=(
                memory_evidence.minimum_visible_after_fired_bullets
            ),
            maximum_fired_bullet_mismatches=(
                memory_evidence.maximum_fired_bullet_mismatches
            ),
            video_binding_excluded_bottom_rows=(
                memory_evidence.video_binding_excluded_bottom_rows
            ),
            maximum_excluded_edge_mismatches=(
                memory_evidence.maximum_excluded_edge_mismatches
            ),
        )
        contract_path = (
            part_root / "evidence" / "memory" / "contract.json"
        )
        _write_new(
            contract_path,
            _canonical_json(memory_contract.to_dict()),
        )
        artifacts[MEMORY_TRANSITION_CONTRACT_ARTIFACT] = _artifact(
            contract_path,
            "evidence/memory/contract.json",
        )

    calibration = _identity_calibration()
    calibration_path = part_root / "calibration.json"
    _write_new(calibration_path, calibration.to_json().encode("utf-8"))
    artifacts["calibration"] = _artifact(
        calibration_path,
        "calibration.json",
    )
    fit = calibration.recompute()

    tick_maps: dict[str, PcTickMap] = {}
    for run_id in ("r1", "r2"):
        tick_map = PcTickMap(
            records=tuple(
                TickPts(
                    tick=tick,
                    pts=(
                        match.r1.pts
                        if run_id == "r1"
                        else match.r2.pts
                    ),
                )
                for tick, match in enumerate(selected)
            )
        )
        path = part_root / f"tick-map-{run_id}.csv"
        _write_new(path, tick_map.to_csv().encode("utf-8"))
        name = f"tick_map.{run_id}"
        artifacts[name] = _artifact(path, f"tick-map-{run_id}.csv")
        tick_maps[run_id] = tick_map

    mismatch_updates = sorted(
        (set(candidates["r1"]) & set(candidates["r2"]))
        - {match.update for match in all_matches}
    )
    analysis = {
        "schema": CASE_SCHEMA,
        "version": CASE_VERSION,
        "case_id": final_case_id,
        "frame_update_assignment": {
            "mode": (
                "render_settled_recomputed"
                if render_settle_enabled
                else "post_grab_framework_sample"
            ),
            "method": (
                RENDER_SETTLE_METHOD if render_settle_enabled else None
            ),
        },
        "selection": {
            "first_update": selected[0].update,
            "last_update": selected[-1].update,
            "native_tick_count": len(selected),
            "native_tick_offset": -selected[0].update,
            "matching_pairs": [
                {
                    "update": match.update,
                    "comparison_bgra_sha256": (
                        match.r1.comparison_sha256
                        or match.r1.frame_sha256
                    ),
                    "r1_frame_sha256": match.r1.frame_sha256,
                    "r2_frame_sha256": match.r2.frame_sha256,
                    "r1_sequence": match.r1.sequence,
                    "r2_sequence": match.r2.sequence,
                    "excluded_edge_mismatch_count": (
                        match.excluded_edge_mismatch_count
                    ),
                }
                for match in selected
            ],
        },
        "full_capture": {
            "common_gameplay_viewport_match_count": len(all_matches),
            "common_full_frame_exact_update_count": sum(
                match.r1.frame_sha256 == match.r2.frame_sha256
                for match in all_matches
            ),
            "common_mismatch_updates": mismatch_updates,
        },
        "pixel_comparison": {
            "excluded_bottom_rows": replay_excluded_bottom_rows,
            "maximum_excluded_edge_mismatches": (
                replay_maximum_excluded_edge_mismatches
            ),
            "maximum_observed_excluded_edge_mismatches": max(
                (
                    match.excluded_edge_mismatch_count
                    for match in selected
                ),
                default=0,
            ),
        },
    }
    analysis_path = part_root / "evidence" / "selection-analysis.json"
    _write_new(analysis_path, _canonical_json(analysis))
    artifacts["selection.analysis"] = _artifact(
        analysis_path,
        "evidence/selection-analysis.json",
    )

    annotation = {
        "schema": "zuma-rl.pc-frame-annotation",
        "version": 1,
        "case_id": final_case_id,
        "channel": "score",
        "status": "observed",
        "value": observed_score,
        "source": "frame_annotation",
        "method": "manual_full_resolution_visual_transcription",
        "crop": {
            "bounds_xyxy": list(_SCORE_CROP),
            "bgra_sha256": crop_sha256,
            "invariant_selected_frame_count": 2 * len(selected),
        },
        "selected_update_range": [
            selected[0].update,
            selected[-1].update,
        ],
    }
    annotation_path = part_root / "evidence" / "score-annotation.json"
    _write_new(annotation_path, _canonical_json(annotation))
    artifacts["annotation.score"] = _artifact(
        annotation_path,
        "evidence/score-annotation.json",
    )

    scenario = Scenario(
        level_id=loaded.definition.id,
        hard=False,
        curve_index=0,
        gun_index=0,
        mode="adventure",
        profile_mode="tutorials_completed",
    )
    environment = PcEnvironment(
        executable_sha256=runtime_sha256,
        main_pak_sha256=main_pak_sha256,
        levels_xml_sha256=levels_xml_sha256,
        curve_sha256=curve_hashes,
        pre_capture_save_sha256=pre_snapshot.state_root,
        profile_mode="tutorials_completed",
        renderer_api="Direct3D9",
        renderer_mode="windowed_800x600",
        ball_radius_branch=18,
        os_build=os_build,
        gpu=next(iter(gpu_names)),
        driver=driver,
        game_settings={
            "full_screen": False,
            "high_resolution": False,
            "is_3d": True,
            "launcher_registry_contract": "preserved_from_prestate",
            "runtime_source_executable_sha256": launcher_sha256,
            "runtime_payload_sha256": runtime_sha256,
            "wait_for_vsync": False,
        },
        dynamic_difficulty_state={},
    )
    timeline = InputTimeline(
        artifact="input.dmo",
        format=DMO_FORMAT,
        file_id=DMO_FILE_ID,
        dmo_version=DMO_VERSION,
        product_version=demo.product_version,
        random_seed=demo.random_seed,
        length_updates=demo.length_updates,
        native_tick_offset=-selected[0].update,
    )
    video_metadata = {
        run_id: _video_metadata(f"video.{run_id}", videos[run_id])
        for run_id in ("r1", "r2")
    }
    clocks = {
        run_id: TickClock(
            logic_hz=(100, 1),
            tick_start=0,
            tick_end=len(selected) - 1,
            tick0_video_pts=tick_maps[run_id].records[0].pts,
            sample_phase="post_update_presented",
            mapping_kind="per_tick_pts_table",
            tick_map_artifact=f"tick_map.{run_id}",
            uncertainty_ticks=0.0,
        )
        for run_id in ("r1", "r2")
    }
    coordinates = CoordinateCalibration(
        raw_width=800,
        raw_height=600,
        logical_width=800,
        logical_height=600,
        transform_kind="axis_aligned_affine",
        pixel_center_convention="center_at_integer",
        logical_from_raw=fit.logical_from_raw,
        rms_error_px=fit.rms_error_px,
        max_error_px=fit.max_error_px,
        calibration_artifact="calibration",
    )
    coverage = tuple(
        CoverageRange(
            channel=channel,
            start_tick=0,
            end_tick=len(selected) - 1,
            status=CoverageStatus.COMPLETE,
            required=True,
        )
        for channel in ("frames", "input_events", "score")
    )
    comparison = ComparisonContract(
        event_tick_tolerance=0,
        center_l2_tolerance_px=0.5,
        waypoint_abs_tolerance=0.01,
        max_drift_per_100_ticks=0.05,
    )
    save_runs = tuple(
        SaveRunContract(
            run_id=run_id,
            start_snapshot_artifact=f"save.{run_id}.start",
            end_snapshot_artifact=f"save.{run_id}.end",
            process_id=capture_metadata[run_id].process_id,
            process_creation_filetime_100ns=(
                capture_metadata[
                    run_id
                ].process_creation_filetime_100ns
            ),
        )
        for run_id in ("r1", "r2")
    )
    save_contract = SaveTransactionContract(
        journal_artifact="save.journal",
        process_timeline_artifact="process.timeline",
        pre_snapshot_artifact="save.pre",
        restored_snapshot_artifact="save.restored",
        runs=save_runs,
        volatile_registry_roles=SAVE_VOLATILE_REGISTRY_ROLES,
    )
    replay_contract = ReplayDeterminismContract(
        runs=tuple(
            ReplayRunContract(
                run_id=run_id,
                trace_artifact=f"trace.{run_id}",
                video=video_metadata[run_id],
                clock=clocks[run_id],
                capture_metadata_artifact=(
                    f"capture.metadata.{run_id}"
                ),
                framework_update_artifact=(
                    f"framework.updates.{run_id}"
                ),
                render_settled_evidence=(
                    RenderSettledReplayEvidence(
                        update_map_artifact=(
                            f"render.settled.updates.{run_id}"
                        ),
                        frames_csv_artifact=(
                            f"capture.frames_csv.{run_id}"
                        ),
                        framework_poll_artifact=(
                            f"framework.poll.{run_id}"
                        ),
                        framework_state_artifact=(
                            f"framework.state.{run_id}"
                        ),
                        calibration_preregistration_artifact=(
                            "render.settle.calibration.preregistration"
                        ),
                        calibration_execution_binding_artifact=(
                            "render.settle.calibration.execution_binding"
                        ),
                        calibration_holdout_report_artifact=(
                            "render.settle.calibration.holdout_report"
                        ),
                    )
                    if render_settle_enabled
                    else None
                ),
            )
            for run_id in ("r1", "r2")
        ),
        pixel_comparison=(
            ReplayPixelComparisonContract(
                mode="exact_except_bounded_bottom_raster_edge",
                excluded_bottom_rows=replay_excluded_bottom_rows,
                maximum_excluded_edge_mismatches=(
                    replay_maximum_excluded_edge_mismatches
                ),
            )
            if replay_excluded_bottom_rows
            else None
        ),
    )
    contract_fingerprint = capture_contract_fingerprint(
        case_id=final_case_id,
        scenario=scenario,
        pc_environment_fingerprint=environment.fingerprint,
        artifacts=artifacts,
        input_timeline=timeline,
        trace_artifact="trace.r1",
        video=video_metadata["r1"],
        clock=clocks["r1"],
        coordinates=coordinates,
        coverage=coverage,
        comparison_contract=comparison,
        save_transaction=save_contract,
        replay_determinism=replay_contract,
    )
    for run_id in ("r1", "r2"):
        trace = _trace(
            case_id=final_case_id,
            environment_fingerprint=environment.fingerprint,
            contract_fingerprint=contract_fingerprint,
            selected=selected,
            run_id=run_id,
            inputs=window_inputs,
            score=observed_score,
        )
        path = part_root / f"trace-{run_id}.ndjson"
        _write_new(path, trace.to_ndjson().encode("utf-8"))
        artifacts[f"trace.{run_id}"] = _artifact(
            path,
            f"trace-{run_id}.ndjson",
        )

    manifest = PcGoldenManifest(
        case_id=final_case_id,
        scenario=scenario,
        pc_environment=environment,
        pc_environment_fingerprint=environment.fingerprint,
        artifacts=artifacts,
        input_timeline=timeline,
        trace_artifact="trace.r1",
        video=video_metadata["r1"],
        clock=clocks["r1"],
        coordinates=coordinates,
        coverage=coverage,
        comparison_contract=comparison,
        producer={
            "tool": "tools/package_pc_golden_v4.py",
            "version": CASE_VERSION,
            "collector_schema": collection["schema"],
            "collector_version": collection["version"],
            "idle_input_window_authorized": allow_idle_input_window,
            "required_update_range": (
                list(required_update_range)
                if required_update_range is not None
                else None
            ),
            "selected_exact_required_update_range": (
                select_exact_required_range
            ),
            "memory_transition_evidence": (
                memory_evidence is not None
            ),
            "certifying_retail_dmo_provenance": (
                dmo_provenance is not None
            ),
            "render_settled_update_evidence": render_settle_enabled,
        },
        save_transaction=save_contract,
        replay_determinism=replay_contract,
    )
    if manifest.capture_contract_fingerprint != contract_fingerprint:
        raise PackagingError("final capture contract fingerprint changed")
    manifest_path = part_root / "manifest.json"
    _write_new(
        manifest_path,
        (manifest.to_json() + "\n").encode("utf-8"),
    )
    manifest_digest = _sha256_path(manifest_path)
    part_root.rename(output_root)
    return {
        "schema": CASE_SCHEMA,
        "version": CASE_VERSION,
        "status": "complete",
        "case_id": final_case_id,
        "case_root": str(output_root),
        "manifest": str(output_root / "manifest.json"),
        "manifest_sha256": manifest_digest,
        "selected_update_range": [
            selected[0].update,
            selected[-1].update,
        ],
        "native_tick_count": len(selected),
        "bound_input_count": len(window_inputs),
        "idle_input_window_authorized": allow_idle_input_window,
        "required_update_range": (
            list(required_update_range)
            if required_update_range is not None
            else None
        ),
        "selected_exact_required_update_range": (
            select_exact_required_range
        ),
        "replay_excluded_bottom_rows": replay_excluded_bottom_rows,
        "replay_maximum_excluded_edge_mismatches": (
            replay_maximum_excluded_edge_mismatches
        ),
        "artifact_count": len(artifacts),
        "memory_transition_evidence": memory_evidence is not None,
        "certifying_retail_dmo_provenance": dmo_provenance is not None,
        "render_settled_update_evidence": render_settle_enabled,
        "memory_video_binding": list(memory_video_binding),
        "video_bytes": sum(
            artifacts[f"video.{run_id}"].bytes
            for run_id in ("r1", "r2")
        ),
    }


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _nonnegative_int(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Package a complete two-run collector session as PC Golden v4."
        )
    )
    parser.add_argument("session_root", type=Path)
    parser.add_argument("--original-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--level-id", required=True)
    parser.add_argument(
        "--observed-score",
        required=True,
        type=_nonnegative_int,
    )
    parser.add_argument("--os-build", required=True)
    parser.add_argument("--driver", required=True)
    parser.add_argument("--minimum-ticks", type=_positive_int, default=50)
    parser.add_argument("--case-id")
    parser.add_argument(
        "--retail-recording-dmo",
        type=Path,
        help=(
            "immutable raw DMO produced by the retail -record path; required "
            "with --retail-recording-report for Gate certification"
        ),
    )
    parser.add_argument(
        "--retail-recording-report",
        type=Path,
        help=(
            "successful non-mutating retail recording result bound to the "
            "raw DMO"
        ),
    )
    parser.add_argument(
        "--allow-idle-input-window",
        action="store_true",
        help=(
            "explicitly allow a selected deterministic window with no "
            "mouse-button edge, for natural post-input mechanisms"
        ),
    )
    parser.add_argument(
        "--required-start-update",
        type=_nonnegative_int,
        help=(
            "preregistered first update that the selected deterministic "
            "window must contain; requires --required-end-update"
        ),
    )
    parser.add_argument(
        "--required-end-update",
        type=_nonnegative_int,
        help=(
            "preregistered final update that the selected deterministic "
            "window must contain; requires --required-start-update"
        ),
    )
    parser.add_argument(
        "--select-exact-required-range",
        action="store_true",
        help=(
            "select exactly the preregistered required update range instead "
            "of extending to the surrounding exact run"
        ),
    )
    parser.add_argument(
        "--render-settle-calibration-preregistration",
        type=Path,
        help=(
            "frozen independent render-transport preregistration; supplying "
            "any render-settle calibration option requires all three"
        ),
    )
    parser.add_argument(
        "--render-settle-calibration-execution-binding",
        type=Path,
    )
    parser.add_argument(
        "--render-settle-calibration-holdout-report",
        type=Path,
    )
    parser.add_argument(
        "--replay-excluded-bottom-rows",
        type=_nonnegative_int,
        default=0,
        help=(
            "exclude exactly one native bottom raster row from strict replay "
            "interior equality"
        ),
    )
    parser.add_argument(
        "--replay-maximum-excluded-edge-mismatches",
        type=_nonnegative_int,
        default=0,
        help="maximum differing pixels in the explicitly excluded raster row",
    )
    parser.add_argument("--memory-before-probe", type=Path)
    parser.add_argument("--memory-after-probe", type=Path)
    parser.add_argument("--memory-before-pixel", type=Path)
    parser.add_argument("--memory-after-pixel", type=Path)
    parser.add_argument("--memory-transition", type=Path)
    parser.add_argument("--memory-before-update", type=_nonnegative_int)
    parser.add_argument("--memory-after-update", type=_nonnegative_int)
    parser.add_argument("--memory-chain-distance-delta", type=float)
    parser.add_argument(
        "--memory-distance-tolerance",
        type=float,
        default=1e-6,
    )
    parser.add_argument(
        "--memory-minimum-chain-count",
        type=_positive_int,
        default=90,
    )
    parser.add_argument(
        "--memory-minimum-visible",
        type=_positive_int,
        default=90,
    )
    parser.add_argument(
        "--memory-maximum-mismatches",
        type=_nonnegative_int,
        default=0,
    )
    parser.add_argument(
        "--memory-minimum-visible-after-fired-bullets",
        type=_nonnegative_int,
        default=1,
    )
    parser.add_argument(
        "--memory-maximum-fired-bullet-mismatches",
        type=_nonnegative_int,
        default=0,
    )
    parser.add_argument(
        "--memory-video-excluded-bottom-rows",
        type=_nonnegative_int,
        default=1,
    )
    parser.add_argument(
        "--memory-maximum-excluded-edge-mismatches",
        type=_nonnegative_int,
        default=4,
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        required_updates = (
            arguments.required_start_update,
            arguments.required_end_update,
        )
        if any(value is not None for value in required_updates) and not all(
            value is not None for value in required_updates
        ):
            raise PackagingError(
                "required start and end updates must be supplied together"
            )
        required_update_range = (
            None
            if required_updates[0] is None
            else (required_updates[0], required_updates[1])
        )
        memory_paths = (
            arguments.memory_before_probe,
            arguments.memory_after_probe,
            arguments.memory_before_pixel,
            arguments.memory_after_pixel,
            arguments.memory_transition,
        )
        memory_evidence: MemoryEvidenceSources | None = None
        if any(path is not None for path in memory_paths):
            if (
                not all(path is not None for path in memory_paths)
                or arguments.memory_before_update is None
                or arguments.memory_after_update is None
                or arguments.memory_chain_distance_delta is None
            ):
                raise PackagingError(
                    "memory evidence paths, updates, and distance delta "
                    "must be supplied together"
                )
            memory_evidence = MemoryEvidenceSources(
                before_probe=arguments.memory_before_probe,
                after_probe=arguments.memory_after_probe,
                before_pixel_validation=arguments.memory_before_pixel,
                after_pixel_validation=arguments.memory_after_pixel,
                transition_validation=arguments.memory_transition,
                expected_before_update=arguments.memory_before_update,
                expected_after_update=arguments.memory_after_update,
                expected_chain_distance_delta=(
                    arguments.memory_chain_distance_delta
                ),
                distance_tolerance=(
                    arguments.memory_distance_tolerance
                ),
                minimum_chain_count=(
                    arguments.memory_minimum_chain_count
                ),
                minimum_visible=arguments.memory_minimum_visible,
                maximum_mismatches=(
                    arguments.memory_maximum_mismatches
                ),
                minimum_visible_after_fired_bullets=(
                    arguments
                    .memory_minimum_visible_after_fired_bullets
                ),
                maximum_fired_bullet_mismatches=(
                    arguments
                    .memory_maximum_fired_bullet_mismatches
                ),
                video_binding_excluded_bottom_rows=(
                    arguments.memory_video_excluded_bottom_rows
                ),
                maximum_excluded_edge_mismatches=(
                    arguments
                    .memory_maximum_excluded_edge_mismatches
                ),
            )
        result = package_case(
            session_root=arguments.session_root,
            original_root=arguments.original_root,
            output_root=arguments.output_root,
            level_id=arguments.level_id,
            observed_score=arguments.observed_score,
            os_build=arguments.os_build,
            driver=arguments.driver,
            minimum_ticks=arguments.minimum_ticks,
            case_id=arguments.case_id,
            memory_evidence=memory_evidence,
            allow_idle_input_window=(
                arguments.allow_idle_input_window
            ),
            required_update_range=required_update_range,
            select_exact_required_range=(
                arguments.select_exact_required_range
            ),
            replay_excluded_bottom_rows=(
                arguments.replay_excluded_bottom_rows
            ),
            replay_maximum_excluded_edge_mismatches=(
                arguments.replay_maximum_excluded_edge_mismatches
            ),
            retail_recording_dmo=arguments.retail_recording_dmo,
            retail_recording_report=arguments.retail_recording_report,
            render_settle_calibration_preregistration=(
                arguments.render_settle_calibration_preregistration
            ),
            render_settle_calibration_execution_binding=(
                arguments.render_settle_calibration_execution_binding
            ),
            render_settle_calibration_holdout_report=(
                arguments.render_settle_calibration_holdout_report
            ),
        )
    except (OSError, ValueError, PackagingError) as error:
        print(
            json.dumps(
                {
                    "schema": CASE_SCHEMA,
                    "version": CASE_VERSION,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "reason": str(error),
                },
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        return 1
    print(
        json.dumps(
            result,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
