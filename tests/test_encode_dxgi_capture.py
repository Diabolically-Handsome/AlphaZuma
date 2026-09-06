from __future__ import annotations

import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

av = pytest.importorskip("av", minversion="18")

from tools import encode_dxgi_capture as encoder  # noqa: E402


def _sha256(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _write_acquisition(
    directory: Path,
    *,
    ticks: tuple[int, ...] = (100, 160, 230),
) -> tuple[tuple[bytes, ...], dict[str, object]]:
    directory.mkdir()
    width = 2
    height = 1
    qpc_frequency = 10_000
    frames = tuple(
        np.array(
            [[[index, index + 1, index + 2, 255],
              [index + 3, index + 4, index + 5, 255]]],
            dtype=np.uint8,
        ).tobytes()
        for index in (10, 30, 50)
    )
    frame_size = width * height * 4
    raw = b"".join(frames)
    raw_path = directory / encoder.RAW_FILE
    raw_path.write_bytes(raw)

    header = ",".join(encoder._CSV_HEADER) + "\n"
    lines = [header]
    for sequence, (tick, payload) in enumerate(zip(ticks, frames)):
        lines.append(
            ",".join(
                (
                    str(sequence),
                    str(tick),
                    str(qpc_frequency),
                    str(1_000 + sequence * 100),
                    "1",
                    str(sequence * frame_size),
                    str(frame_size),
                    _sha256(payload),
                )
            )
            + "\n"
        )
    csv_bytes = "".join(lines).encode("ascii")
    csv_path = directory / encoder.CSV_FILE
    csv_path.write_bytes(csv_bytes)

    aggregate = hashlib.sha256(encoder._PIXEL_HASH_DOMAIN)
    for payload in frames:
        aggregate.update(len(payload).to_bytes(8, "big"))
        aggregate.update(payload)
    requested_duration = 0.25
    frame_budget_fps = 60
    frame_budget = math.ceil(
        requested_duration * frame_budget_fps
    ) + 2
    span = ticks[-1] - ticks[0]
    span_seconds = span / qpc_frequency
    metadata: dict[str, object] = {
        "schema": encoder.CAPTURE_SCHEMA,
        "version": encoder.CAPTURE_VERSION,
        "status": "acquisition_complete",
        "target_process": "popcapgame1.exe",
        "device_index": 0,
        "output_index": 0,
        "global_region": [10, 20, 12, 21],
        "output_local_region": [0, 0, 2, 1],
        "width": width,
        "height": height,
        "pixel_format": "bgra",
        "bytes_per_pixel": 4,
        "frame_budget_fps": frame_budget_fps,
        "capture_mode": "every_new_present",
        "capture_storage_mode": "memory_then_publish",
        "capture_surface": "dxgi_desktop_client_region_crop",
        "foreground_geometry_guard": True,
        "topmost_or_injected_overlay_detection": False,
        "requires_full_frame_visual_review": True,
        "requested_duration_seconds": requested_duration,
        "capture_start_perf_counter_ns": 900,
        "capture_end_perf_counter_ns": 250_000_900,
        "capture_elapsed_seconds": requested_duration,
        "estimated_raw_budget_bytes": frame_size * frame_budget,
        "frame_budget": frame_budget,
        "frame_count": len(frames),
        "warmup_baseline_present_ticks": ticks[0] - 1,
        "first_present_ticks": ticks[0],
        "last_present_ticks": ticks[-1],
        "present_span_ticks": span,
        "present_span_seconds": span_seconds,
        "observed_mean_present_fps": (
            (len(frames) - 1) / span_seconds
        ),
        "qpc_frequency": qpc_frequency,
        "missed_presentations": 0,
        "idle_poll_count": 0,
        "pointer_only_update_count": 0,
        "warmup_idle_poll_count": 0,
        "warmup_pointer_only_update_count": 0,
        "runtime_versions": {
            "comtypes": "1.4.16",
            "dxcam": "0.3.0",
            "numpy": "2.5.1",
        },
        "runtime_versions_verified": True,
        "source_identity": {
            "adapter_description": "Synthetic GPU",
            "adapter_vendor_id": 4318,
            "adapter_vram_bytes": 16 * 1024**3,
            "output_device_name": r"\\.\DISPLAY1",
            "output_desktop_region": [0, 0, 3840, 2160],
            "output_rotation_degrees": 0,
        },
        "target_identity": {
            "process_id": 1234,
            "process_creation_filetime_100ns": 133_000_000_000_000_000,
            "executable_bytes": 12_345,
            "executable_sha256": "sha256:" + "1" * 64,
            "executable_hash_provenance": {
                "method": "direct_file_sha256",
            },
            "window_handle_hex": "0x0000000000001234",
            "window_client_region": [10, 20, 12, 21],
        },
        "raw_bytes": len(raw),
        "raw_sha256": _sha256(raw),
        "frames_csv_bytes": len(csv_bytes),
        "frames_csv_rows": len(frames),
        "frames_csv_sha256": _sha256(csv_bytes),
        "aggregate_pixel_sha256": (
            "sha256:" + aggregate.hexdigest()
        ),
        "frames_csv": encoder.CSV_FILE,
        "raw_frames": encoder.RAW_FILE,
    }
    metadata_bytes = (
        json.dumps(
            metadata,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")
    (directory / encoder.METADATA_FILE).write_bytes(metadata_bytes)
    return frames, metadata


def _rewrite_metadata(directory: Path, metadata: dict[str, object]) -> None:
    (directory / encoder.METADATA_FILE).write_bytes(
        (
            json.dumps(
                metadata,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")
    )


def test_encodes_lossless_ffv1_with_recomputable_relative_pts(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    output.mkdir()
    frames, unused = _write_acquisition(source)
    del unused
    before = {
        item.name: item.read_bytes()
        for item in source.iterdir()
    }

    report = encoder.encode_capture(source, output)

    video_path = output / encoder.DIRECTORY_OUTPUT_FILE
    assert video_path.is_file()
    assert not list(output.glob("*.part"))
    assert report.frame_pts == (0, 6, 13)
    assert report.nominal_fps == (2000, 13)
    assert report.source_mean_present_fps == (2000, 13)
    assert report.encoded_pts_cfr is False
    assert report.artifact_bytes == video_path.stat().st_size
    assert report.artifact_sha256 == _sha256(video_path.read_bytes())
    assert report.to_dict()["timestamp_mapping"] == (
        "qpc_relative_nearest_millisecond_ties_up"
    )
    assert report.to_dict()["pc_golden_v3_compatible"] is False
    assert {
        item.name: item.read_bytes()
        for item in source.iterdir()
    } == before

    container = av.open(str(video_path), mode="r")
    stream = container.streams.video[0]
    assert stream.codec_context.name == "ffv1"
    assert stream.codec_context.format.name == "bgra"
    assert str(stream.time_base) == "1/1000"
    embedded_timeline = json.loads(
        container.metadata[encoder.TIMELINE_METADATA_TAG]
    )
    assert embedded_timeline["source_present_ticks"] == [100, 160, 230]
    assert embedded_timeline["frame_pts"] == [0, 6, 13]
    assert embedded_timeline["time_base"] == [1, 1000]
    assert embedded_timeline["pc_golden_v3_compatible"] is False
    assert embedded_timeline[
        "topmost_or_injected_overlay_detection"
    ] is False
    assert embedded_timeline[
        "requires_full_frame_visual_review"
    ] is True
    assert embedded_timeline["source_metadata_sha256"] == _sha256(
        (source / encoder.METADATA_FILE).read_bytes()
    )
    assert report.source_metadata_sha256 == embedded_timeline[
        "source_metadata_sha256"
    ]
    decoded = list(container.decode(stream))
    container.close()
    assert [frame.pts for frame in decoded] == [0, 6, 13]
    assert [
        frame.to_ndarray(format="bgra").tobytes()
        for frame in decoded
    ] == list(frames)


def test_accepts_an_explicit_new_mkv_file(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _write_acquisition(source, ticks=(100, 200, 300))
    output = tmp_path / "named.mkv"

    report = encoder.encode_capture(source, output)

    assert output.is_file()
    assert report.frame_pts == (0, 10, 20)
    assert report.nominal_fps == (100, 1)
    assert report.encoded_pts_cfr is True
    assert not output.with_name("named.mkv.part").exists()


def test_publish_closes_part_handle_before_unlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    part = tmp_path / "capture.mkv.part"
    final = tmp_path / "capture.mkv"
    payload = b"lossless-video-placeholder"
    part.write_bytes(payload)
    opened: dict[str, object] = {}
    original_open = encoder._open_regular_read_only
    original_unlink = Path.unlink

    def tracked_open(path: Path, *, code: str):
        stream, snapshot = original_open(path, code=code)
        opened["stream"] = stream
        return stream, snapshot

    def guarded_unlink(path: Path, *args, **kwargs):
        if path == part:
            assert getattr(opened["stream"], "closed") is True
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(encoder, "_open_regular_read_only", tracked_open)
    monkeypatch.setattr(Path, "unlink", guarded_unlink)

    encoder._publish_no_replace(
        part,
        final,
        expected_bytes=len(payload),
        expected_sha256=_sha256(payload),
    )

    assert final.read_bytes() == payload
    assert not part.exists()


def test_accepts_five_and_a_half_second_capture_within_frame_budget(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    frames, metadata = _write_acquisition(source)
    duration = 5.5
    frame_budget = math.ceil(
        duration * int(metadata["frame_budget_fps"])
    ) + 2
    metadata["requested_duration_seconds"] = duration
    metadata["capture_elapsed_seconds"] = duration
    metadata["capture_end_perf_counter_ns"] = 5_500_000_900
    metadata["frame_budget"] = frame_budget
    metadata["estimated_raw_budget_bytes"] = (
        len(frames[0]) * frame_budget
    )
    _rewrite_metadata(source, metadata)
    output = tmp_path / "capture.mkv"

    report = encoder.encode_capture(source, output)

    assert report.frame_count == len(frames)
    assert output.is_file()


def test_allows_matroska_average_rate_rational_rounding() -> None:
    expected = Fraction(103121, 625)
    observed = Fraction(25739, 156)

    assert encoder._average_rate_matches(observed, expected)
    assert encoder._average_rate_matches(
        Fraction(29864, 181),
        Fraction(150147, 910),
    )
    assert not encoder._average_rate_matches(
        expected + Fraction(1, 10),
        expected,
    )
    assert not encoder._average_rate_matches(None, expected)


def test_rejects_a_stale_part_before_creating_output(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _write_acquisition(source)
    (source / "stale.part").write_bytes(b"incomplete")
    output = tmp_path / "out.mkv"

    with pytest.raises(
        encoder.EncodingError,
        match="input_contains_part",
    ):
        encoder.encode_capture(source, output)

    assert not output.exists()
    assert not output.with_name("out.mkv.part").exists()


def test_rejects_per_frame_hash_mismatch(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    unused, metadata = _write_acquisition(source)
    del unused
    csv_path = source / encoder.CSV_FILE
    csv_text = csv_path.read_text(encoding="ascii")
    csv_text = csv_text.replace(
        csv_text.splitlines()[1].split(",")[-1],
        "sha256:" + "0" * 64,
        1,
    )
    csv_bytes = csv_text.encode("ascii")
    csv_path.write_bytes(csv_bytes)
    metadata["frames_csv_bytes"] = len(csv_bytes)
    metadata["frames_csv_sha256"] = _sha256(csv_bytes)
    _rewrite_metadata(source, metadata)
    output = tmp_path / "out.mkv"

    with pytest.raises(
        encoder.EncodingError,
        match="frame_sha256_mismatch",
    ):
        encoder.encode_capture(source, output)

    assert not output.exists()
    assert not output.with_name("out.mkv.part").exists()


def test_rejects_noncontiguous_raw_offsets(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    unused, metadata = _write_acquisition(source)
    del unused
    csv_path = source / encoder.CSV_FILE
    lines = csv_path.read_text(encoding="ascii").splitlines()
    fields = lines[2].split(",")
    fields[5] = "9"
    lines[2] = ",".join(fields)
    csv_bytes = ("\n".join(lines) + "\n").encode("ascii")
    csv_path.write_bytes(csv_bytes)
    metadata["frames_csv_bytes"] = len(csv_bytes)
    metadata["frames_csv_sha256"] = _sha256(csv_bytes)
    _rewrite_metadata(source, metadata)

    with pytest.raises(
        encoder.EncodingError,
        match="raw_offset_invalid",
    ):
        encoder.encode_capture(source, tmp_path / "out.mkv")


def test_never_overwrites_an_existing_output(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _write_acquisition(source)
    output = tmp_path / "out.mkv"
    output.write_bytes(b"keep me")

    with pytest.raises(
        encoder.EncodingError,
        match="output_already_exists",
    ):
        encoder.encode_capture(source, output)

    assert output.read_bytes() == b"keep me"


def test_rejects_a_new_output_file_inside_the_input_directory(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _write_acquisition(source)
    before_names = {item.name for item in source.iterdir()}

    with pytest.raises(
        encoder.EncodingError,
        match="output_overlaps_input",
    ):
        encoder.encode_capture(source, source / "capture.mkv")

    assert {item.name for item in source.iterdir()} == before_names
    assert not (source / "capture.mkv").exists()
    assert not (source / "capture.mkv.part").exists()


def test_requires_verified_pinned_capture_runtime(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    unused, metadata = _write_acquisition(source)
    del unused
    metadata["runtime_versions_verified"] = False
    _rewrite_metadata(source, metadata)

    with pytest.raises(
        encoder.EncodingError,
        match="metadata_invalid",
    ):
        encoder.encode_capture(source, tmp_path / "out.mkv")

    assert not (tmp_path / "out.mkv").exists()


def test_accepts_structurally_bound_embedded_runtime_identity() -> None:
    executable_sha256 = "sha256:" + "2" * 64
    provenance = {
        "method": "embedded_signed_pe",
        "source_bytes": 1_300,
        "source_sha256": "sha256:" + "1" * 64,
        "payload_offset": 100,
        "payload_bytes": 1_000,
        "payload_sha256": executable_sha256,
        "payload_extent_basis": "certificate_table_end",
        "trailing_source_bytes": 200,
        "machine": 0x14C,
        "section_count": 4,
        "pe_timestamp": 123,
        "pe_checksum": 456,
        "size_of_image": 2_000,
        "size_of_headers": 200,
        "certificate_table_offset": 900,
        "certificate_table_bytes": 100,
        "runtime_file_direct_hash_verified": False,
    }

    encoder._validate_executable_hash_provenance(
        provenance,
        executable_bytes=1_000,
        executable_sha256=executable_sha256,
    )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("source_bytes", 1_301),
        ("payload_bytes", 999),
        ("payload_sha256", "sha256:" + "3" * 64),
        ("payload_extent_basis", "section_table_end"),
        ("certificate_table_bytes", 99),
        ("runtime_file_direct_hash_verified", 0),
    ],
)
def test_rejects_inconsistent_embedded_runtime_identity(
    field: str,
    replacement: object,
) -> None:
    executable_sha256 = "sha256:" + "2" * 64
    provenance: dict[str, object] = {
        "method": "embedded_signed_pe",
        "source_bytes": 1_300,
        "source_sha256": "sha256:" + "1" * 64,
        "payload_offset": 100,
        "payload_bytes": 1_000,
        "payload_sha256": executable_sha256,
        "payload_extent_basis": "certificate_table_end",
        "trailing_source_bytes": 200,
        "machine": 0x14C,
        "section_count": 4,
        "pe_timestamp": 123,
        "pe_checksum": 456,
        "size_of_image": 2_000,
        "size_of_headers": 200,
        "certificate_table_offset": 900,
        "certificate_table_bytes": 100,
        "runtime_file_direct_hash_verified": False,
    }
    provenance[field] = replacement

    with pytest.raises(
        encoder.EncodingError,
        match="target_identity_invalid",
    ):
        encoder._validate_executable_hash_provenance(
            provenance,
            executable_bytes=1_000,
            executable_sha256=executable_sha256,
        )


def test_cli_error_never_echoes_private_paths(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    private_input = tmp_path / "private-case-name"
    private_output = tmp_path / "private-output-name.mkv"

    status = encoder.main(
        [
            "--input-dir",
            str(private_input),
            "--output",
            str(private_output),
        ]
    )
    captured = capsys.readouterr()

    assert status == 1
    assert captured.out == ""
    assert captured.err == "encode error: input_directory_invalid\n"
    assert str(private_input) not in captured.err
    assert str(private_output) not in captured.err


@pytest.mark.parametrize(
    ("tick", "expected"),
    [
        (100, 0),
        (104, 0),
        (105, 1),
        (114, 1),
        (115, 2),
    ],
)
def test_qpc_mapping_rounds_nearest_with_ties_up(
    tick: int,
    expected: int,
) -> None:
    assert encoder.qpc_relative_pts(
        tick,
        first_present_ticks=100,
        qpc_frequency=10_000,
    ) == expected
