from __future__ import annotations

import hashlib
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

av = pytest.importorskip("av", minversion="18")

from zuma_rl.pc_golden import (  # noqa: E402
    ArtifactSpec,
    ComparisonContract,
    CoordinateCalibration,
    CoverageRange,
    CoverageStatus,
    InputTimeline,
    PcEnvironment,
    PcGoldenManifest,
    Scenario,
    TickClock,
    VideoMetadata,
)
from zuma_rl.pc_video import (  # noqa: E402
    PcVideoInspection,
    VideoArtifactError,
    VideoDecodeError,
    VideoDecoderUnavailable,
    VideoMetadataMismatch,
    VideoResourceLimitError,
    inspect_pc_video,
)
from zuma_rl.pc_video import _cadence_counts  # noqa: E402
from zuma_rl.pc_video import _decode_all_frames  # noqa: E402
from zuma_rl.revenge_core import SUPPORTED_PROFILE_MODE  # noqa: E402


_DUMMY_SHA = "sha256:" + "7" * 64


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_ffv1(
    path: Path,
    *,
    width: int = 8,
    height: int = 6,
    pts: tuple[int, ...] = (0, 4, 8),
    static: bool = False,
    video_streams: int = 1,
) -> None:
    """Write a tiny lossless Matroska fixture with a 1/100 input clock.

    Matroska canonicalizes that clock to 1/1000, so the decoded PTS are ten
    times the supplied values and the nominal 25 fps cadence is 40 ticks.
    """

    container = av.open(str(path), mode="w")
    streams = []
    for _ in range(video_streams):
        stream = container.add_stream("ffv1", rate=Fraction(25, 1))
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv444p"
        stream.time_base = Fraction(1, 100)
        streams.append(stream)
    try:
        for index, source_pts in enumerate(pts):
            for stream_index, stream in enumerate(streams):
                value = 31 if static else 31 + index * 40 + stream_index
                pixels = np.full(
                    (height, width, 3),
                    value,
                    dtype=np.uint8,
                )
                frame = av.VideoFrame.from_ndarray(
                    pixels,
                    format="rgb24",
                )
                frame.pts = source_pts
                frame.time_base = Fraction(1, 100)
                for packet in stream.encode(frame):
                    container.mux(packet)
        for stream in streams:
            for packet in stream.encode():
                container.mux(packet)
    finally:
        container.close()


def _manifest(
    video_path: Path,
    *,
    metadata: VideoMetadata | None = None,
) -> PcGoldenManifest:
    actual_video = metadata or VideoMetadata(
        artifact="video",
        width=8,
        height=6,
        codec="ffv1",
        pixel_format="yuv444p",
        time_base=(1, 1000),
        nominal_fps=(25, 1),
        frame_count=3,
        first_pts=0,
        last_pts=80,
        cfr=True,
        dropped_frames=0,
        duplicate_frames=0,
    )
    artifacts = {
        "video": ArtifactSpec(
            path="capture.mkv",
            sha256=_sha256(video_path),
            bytes=video_path.stat().st_size,
        ),
        "trace": ArtifactSpec(
            path="trace.jsonl",
            sha256=_DUMMY_SHA,
            bytes=0,
        ),
        "dmo": ArtifactSpec(
            path="input.dmo",
            sha256=_DUMMY_SHA,
            bytes=0,
        ),
        "tick_map": ArtifactSpec(
            path="ticks.csv",
            sha256=_DUMMY_SHA,
            bytes=0,
        ),
        "calibration": ArtifactSpec(
            path="calibration.json",
            sha256=_DUMMY_SHA,
            bytes=0,
        ),
    }
    environment = PcEnvironment(
        executable_sha256=_DUMMY_SHA,
        main_pak_sha256=_DUMMY_SHA,
        levels_xml_sha256=_DUMMY_SHA,
        curve_sha256={"curve": _DUMMY_SHA},
        pre_capture_save_sha256=_DUMMY_SHA,
        profile_mode=SUPPORTED_PROFILE_MODE,
        renderer_api="Direct3D",
        renderer_mode="windowed",
        ball_radius_branch=18,
        os_build="test",
        gpu="test",
        driver="test",
    )
    return PcGoldenManifest(
        case_id="pc-video-test",
        scenario=Scenario(
            level_id="lvl11",
            hard=False,
            curve_index=0,
            gun_index=0,
            mode="adventure",
            profile_mode=SUPPORTED_PROFILE_MODE,
        ),
        pc_environment=environment,
        pc_environment_fingerprint=environment.fingerprint,
        artifacts=artifacts,
        input_timeline=InputTimeline(
            artifact="dmo",
            format="popcap_dmo_v2",
            file_id=0x42BEEF78,
            dmo_version=2,
            product_version="test",
            random_seed=1,
            length_updates=0,
            native_tick_offset=0,
        ),
        trace_artifact="trace",
        video=actual_video,
        clock=TickClock(
            logic_hz=(100, 1),
            tick_start=0,
            tick_end=0,
            tick0_video_pts=actual_video.first_pts,
            sample_phase="post_update_presented",
            mapping_kind="explicit_tick_pts_v1",
            tick_map_artifact="tick_map",
            uncertainty_ticks=0.0,
        ),
        coordinates=CoordinateCalibration(
            raw_width=actual_video.width,
            raw_height=actual_video.height,
            logical_width=800,
            logical_height=600,
            transform_kind="axis_aligned_affine",
            pixel_center_convention="center_at_integer",
            logical_from_raw=(
                (800.0 / actual_video.width, 0.0, 0.0),
                (0.0, 600.0 / actual_video.height, 0.0),
                (0.0, 0.0, 1.0),
            ),
            rms_error_px=0.0,
            max_error_px=0.0,
            calibration_artifact="calibration",
        ),
        coverage=(
            CoverageRange(
                channel="frames",
                start_tick=0,
                end_tick=0,
                status=CoverageStatus.COMPLETE,
                required=True,
            ),
        ),
        comparison_contract=ComparisonContract(
            event_tick_tolerance=0,
            center_l2_tolerance_px=0.5,
            waypoint_abs_tolerance=0.05,
            max_drift_per_100_ticks=0.05,
        ),
    )


def _replace_video_artifact(
    manifest: PcGoldenManifest,
    path: Path,
) -> PcGoldenManifest:
    artifacts = dict(manifest.artifacts)
    artifacts["video"] = replace(
        artifacts["video"],
        sha256=_sha256(path),
        bytes=path.stat().st_size,
    )
    return replace(manifest, artifacts=artifacts)


def test_inspect_decodes_all_frames_and_hashes_normalized_pixels(
    tmp_path: Path,
) -> None:
    path = tmp_path / "capture.mkv"
    _write_ffv1(path)
    report = inspect_pc_video(path, _manifest(path))

    assert isinstance(report, PcVideoInspection)
    assert report.frame_pts == (0, 40, 80)
    assert report.frame_count == 3
    assert report.time_base == (1, 1000)
    assert report.nominal_fps == (25, 1)
    assert report.normalized_pixel_format == "rgb24"
    assert report.normalized_decoded_bytes == 8 * 6 * 3 * 3
    assert report.decoded_pixel_sha256.startswith("sha256:")
    assert len(report.decoded_pixel_sha256) == 71
    assert len(report.frame_pixel_sha256) == report.frame_count
    assert all(
        item.startswith("sha256:") and len(item) == 71
        for item in report.frame_pixel_sha256
    )
    assert report.to_dict()["frame_pts"] == [0, 40, 80]


def test_static_frames_are_not_misclassified_as_capture_duplicates(
    tmp_path: Path,
) -> None:
    path = tmp_path / "static.mkv"
    _write_ffv1(path, static=True)
    report = inspect_pc_video(path, _manifest(path))

    assert report.cfr is True
    assert report.dropped_frames == 0
    assert report.duplicate_frames == 0


def test_qpc_vfr_mode_does_not_invent_cfr_drop_slots(
    tmp_path: Path,
) -> None:
    path = tmp_path / "qpc-vfr.mkv"
    _write_ffv1(path, pts=(0, 4, 12))
    metadata = VideoMetadata(
        artifact="video",
        width=8,
        height=6,
        codec="ffv1",
        pixel_format="yuv444p",
        time_base=(1, 1000),
        nominal_fps=(25, 1),
        frame_count=3,
        first_pts=0,
        last_pts=120,
        cfr=False,
        dropped_frames=0,
        duplicate_frames=0,
        timing_mode="qpc_vfr_pts_table",
    )

    report = inspect_pc_video(path, _manifest(path, metadata=metadata))

    assert report.frame_pts == (0, 40, 120)
    assert report.cfr is False
    assert report.dropped_frames == 0
    assert report.duplicate_frames == 0


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("codec", "huffyuv"),
        ("pixel_format", "yuv420p"),
        ("time_base", (1, 2000)),
        ("nominal_fps", (50, 1)),
    ),
)
def test_stream_metadata_mismatch_is_rejected(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    path = tmp_path / "metadata.mkv"
    _write_ffv1(path)
    manifest = _manifest(path)
    wrong_video = replace(
        manifest.video,
        cfr=False,
        **{field: value},
    )
    wrong_manifest = replace(manifest, video=wrong_video)

    with pytest.raises(VideoMetadataMismatch, match=field):
        inspect_pc_video(path, wrong_manifest)


def test_decoded_dimensions_must_match_manifest(tmp_path: Path) -> None:
    path = tmp_path / "dimensions.mkv"
    _write_ffv1(path, width=10)
    manifest = _manifest(path)

    with pytest.raises(VideoMetadataMismatch, match="dimensions"):
        inspect_pc_video(path, manifest)


def test_full_decode_rejects_truncated_artifact_even_when_rehashed(
    tmp_path: Path,
) -> None:
    original = tmp_path / "original.mkv"
    damaged = tmp_path / "damaged.mkv"
    _write_ffv1(original)
    payload = original.read_bytes()
    damaged.write_bytes(payload[: max(1, len(payload) * 3 // 4)])
    manifest = _replace_video_artifact(_manifest(original), damaged)

    with pytest.raises((VideoDecodeError, VideoMetadataMismatch)):
        inspect_pc_video(damaged, manifest)


def test_extra_video_stream_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "two-streams.mkv"
    _write_ffv1(path, video_streams=2)
    manifest = _manifest(path)

    with pytest.raises(VideoMetadataMismatch, match="exactly one"):
        inspect_pc_video(path, manifest)


def test_dropped_frame_count_comes_from_pts_cadence(tmp_path: Path) -> None:
    path = tmp_path / "gap.mkv"
    _write_ffv1(path, pts=(0, 4, 12))
    metadata = VideoMetadata(
        artifact="video",
        width=8,
        height=6,
        codec="ffv1",
        pixel_format="yuv444p",
        time_base=(1, 1000),
        nominal_fps=(25, 1),
        frame_count=3,
        first_pts=0,
        last_pts=120,
        cfr=False,
        dropped_frames=1,
        duplicate_frames=0,
    )
    report = inspect_pc_video(path, _manifest(path, metadata=metadata))
    assert report.cfr is False
    assert report.dropped_frames == 1

    false_claim = replace(
        _manifest(path, metadata=metadata),
        video=replace(metadata, dropped_frames=0),
    )
    with pytest.raises(VideoMetadataMismatch, match="dropped_frames"):
        inspect_pc_video(path, false_claim)


def test_artifact_is_only_read_and_path_is_not_disclosed(tmp_path: Path) -> None:
    path = tmp_path / "very-secret-capture-name.mkv"
    _write_ffv1(path)
    manifest = _manifest(path)
    before = path.read_bytes()
    path.chmod(0o444)
    before_mode = path.stat().st_mode

    inspect_pc_video(path, manifest)

    assert path.read_bytes() == before
    assert path.stat().st_mode == before_mode

    secret = tmp_path / "hidden-parent" / "private-video-name.mkv"
    with pytest.raises(VideoArtifactError) as caught:
        inspect_pc_video(secret, manifest)
    assert "hidden-parent" not in str(caught.value)
    assert "private-video-name" not in str(caught.value)


def test_wrong_artifact_hash_error_does_not_disclose_path(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sensitive-capture.mkv"
    _write_ffv1(path)
    manifest = _manifest(path)
    bad_artifacts = dict(manifest.artifacts)
    bad_artifacts["video"] = replace(
        bad_artifacts["video"],
        sha256="sha256:" + "0" * 64,
    )
    bad_manifest = replace(manifest, artifacts=bad_artifacts)

    with pytest.raises(VideoArtifactError) as caught:
        inspect_pc_video(path, bad_manifest)
    assert str(path) not in str(caught.value)
    assert path.name not in str(caught.value)


def test_decoder_unavailable_has_a_dedicated_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "capture.mkv"
    _write_ffv1(path)
    manifest = _manifest(path)
    monkeypatch.setattr("zuma_rl.pc_video._av", None)

    with pytest.raises(VideoDecoderUnavailable):
        inspect_pc_video(path, manifest)


def test_declared_resource_limit_is_checked_before_opening(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ordinary.mkv"
    _write_ffv1(path)
    manifest = _manifest(path)
    oversized_video = replace(
        manifest.video,
        width=16_385,
        height=1,
    )
    oversized_manifest = replace(
        manifest,
        video=oversized_video,
        coordinates=replace(
            manifest.coordinates,
            raw_width=16_385,
            raw_height=1,
            logical_from_raw=(
                (800.0 / 16_385, 0.0, 0.0),
                (0.0, 600.0, 0.0),
                (0.0, 0.0, 1.0),
            ),
        ),
    )

    with pytest.raises(VideoResourceLimitError):
        inspect_pc_video(tmp_path / "does-not-exist.mkv", oversized_manifest)


def test_irregular_off_grid_pts_are_rejected() -> None:
    with pytest.raises(VideoDecodeError, match="cadence"):
        _cadence_counts((0, 40, 90), pts_step=40)


def test_duplicate_cadence_count_does_not_use_adjacent_pixel_or_pts_pairs(
) -> None:
    assert _cadence_counts((0, 20, 40), pts_step=40) == (
        False,
        0,
        1,
    )


class _FakeFormat:
    name = "yuv444p"


class _FakeFrame:
    def __init__(self, pts: int | None) -> None:
        self.pts = pts
        self.time_base = Fraction(1, 1000)
        self.width = 8
        self.height = 6
        self.format = _FakeFormat()
        self.is_corrupt = False

    def to_ndarray(self, *, format: str) -> np.ndarray:
        assert format == "rgb24"
        return np.zeros((6, 8, 3), dtype=np.uint8)


class _FakeContainer:
    def __init__(self, frames: tuple[_FakeFrame, ...]) -> None:
        self.frames = frames

    def decode(self, stream: object) -> tuple[_FakeFrame, ...]:
        del stream
        return self.frames


def test_missing_and_nonincreasing_frame_pts_are_rejected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "metadata-source.mkv"
    _write_ffv1(path)
    metadata = _manifest(path).video

    with pytest.raises(VideoDecodeError, match="missing PTS"):
        _decode_all_frames(
            _FakeContainer((_FakeFrame(None),)),
            object(),
            metadata,
        )
    with pytest.raises(VideoDecodeError, match="strictly increasing"):
        _decode_all_frames(
            _FakeContainer((_FakeFrame(0), _FakeFrame(0))),
            object(),
            metadata,
        )
