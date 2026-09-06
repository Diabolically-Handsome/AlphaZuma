"""End-to-end tests for the read-only PC golden acceptance verifier."""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import zuma_rl.verify_pc_golden as verifier_module
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
    ComparisonStatus,
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
    PcGoldenValidationError,
    ReplayDeterminismContract,
    ReplayPixelComparisonContract,
    ReplayRunContract,
    SAVE_VOLATILE_REGISTRY_ROLES,
    SaveRunContract,
    SaveTransactionContract,
    Scenario,
    TickClock,
    VideoMetadata,
    capture_contract_fingerprint,
)
from zuma_rl.pc_protocol_evidence import (
    FrameworkUpdateRecord,
    REGISTRY_ROOTS,
    PcDxgiCaptureMetadata,
    PcFrameworkUpdateMap,
    PcProcessEvent,
    PcProcessTimeline,
    PcSaveJournal,
    PcStateSnapshot,
    SnapshotFile,
    SnapshotRegistryNode,
    SnapshotRegistryValue,
)
from zuma_rl.pc_video import (
    MAX_VIDEO_ARTIFACT_BYTES,
    PcVideoInspection,
    VideoDecoderUnavailable,
    VideoMetadataMismatch,
)
from zuma_rl.popcap_dmo import DEMO_FILE_ID
from zuma_rl.verify_pc_golden import (
    REPORT_SCHEMA,
    VerificationCheckStatus,
    main,
    verify_pc_golden_case,
)

_PRODUCT_VERSION = "1.0.4.9496"
_RANDOM_SEED = 0x12345678
_LENGTH_UPDATES = 2
_SENSITIVE_DMO_VALUE = b"PRIVATE_EMBEDDED_DMO_VALUE"


def _digest(payload: bytes | str) -> str:
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _matching_video_inspection(
    manifest: PcGoldenManifest,
    *,
    frame_pts: tuple[int, ...] = (0, 10, 20),
) -> PcVideoInspection:
    metadata = manifest.video
    artifact = manifest.artifacts[metadata.artifact]
    return PcVideoInspection(
        artifact_sha256=artifact.sha256,
        artifact_bytes=artifact.bytes,
        width=metadata.width,
        height=metadata.height,
        codec=metadata.codec,
        pixel_format=metadata.pixel_format,
        time_base=metadata.time_base,
        nominal_fps=metadata.nominal_fps,
        frame_count=metadata.frame_count,
        first_pts=metadata.first_pts,
        last_pts=metadata.last_pts,
        frame_pts=frame_pts,
        cfr=metadata.cfr,
        dropped_frames=metadata.dropped_frames,
        duplicate_frames=metadata.duplicate_frames,
        normalized_pixel_format="rgb24",
        normalized_decoded_bytes=(
            metadata.width
            * metadata.height
            * 3
            * metadata.frame_count
        ),
        decoded_pixel_sha256=_digest("decoded-rgb24"),
    )


@pytest.fixture(autouse=True)
def _stub_successful_video_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep verifier unit fixtures tiny; pc_video has real FFV1 tests."""

    def inspect(path: Path, manifest: PcGoldenManifest) -> PcVideoInspection:
        del path
        return _matching_video_inspection(manifest)

    monkeypatch.setattr(verifier_module, "inspect_pc_video", inspect)


class _BitWriter:
    def __init__(self) -> None:
        self.data = bytearray()
        self.bit_position = 0

    def bits(self, value: int, count: int) -> None:
        masked = value & ((1 << count) - 1)
        for index in range(count):
            if self.bit_position % 8 == 0:
                self.data.append(0)
            if masked & (1 << index):
                self.data[self.bit_position // 8] |= (
                    1 << (self.bit_position % 8)
                )
            self.bit_position += 1

    def byte(self, value: int) -> None:
        self.bits(value, 8)

    def i32(self, value: int) -> None:
        for shift in (0, 8, 16, 24):
            self.byte((value >> shift) & 0xFF)


def _command(
    writer: _BitWriter,
    *,
    delta: int,
    number: int,
    short: bool = False,
) -> None:
    writer.bits(delta, 4)
    writer.bits(int(short), 1)
    writer.bits(number, 1 if short else 5)


def _dmo_bytes(
    *,
    version: int = 2,
    product_version: str = _PRODUCT_VERSION,
    random_seed: int = _RANDOM_SEED,
    length_updates: int = _LENGTH_UPDATES,
    file_id: int = DEMO_FILE_ID,
    include_inputs: bool = True,
    input_update: int = 1,
) -> bytes:
    writer = _BitWriter()

    if include_inputs:
        _command(writer, delta=input_update, number=0)
        writer.bits(400, 12)
        writer.bits(300, 12)

        _command(writer, delta=0, number=1, short=True)
        writer.bits(1, 1)
        writer.bits(-1, 3)

    _command(writer, delta=0 if include_inputs else 1, number=18)
    writer.i32(len(_SENSITIVE_DMO_VALUE))
    for value in _SENSITIVE_DMO_VALUE:
        writer.byte(value)

    product = product_version.encode("latin-1")
    header = struct.pack(
        "<IIIH",
        file_id,
        version,
        random_seed,
        len(product),
    ) + product
    if version >= 2:
        marker_buffer = struct.pack("<i", 0)
        header += struct.pack("<I", len(marker_buffer)) + marker_buffer
    return (
        header
        + struct.pack("<I", length_updates)
        + bytes(writer.data)
    )


def _environment(
    *,
    profile_mode: str = "tutorials_completed",
) -> PcEnvironment:
    return PcEnvironment(
        executable_sha256=_digest("exe"),
        main_pak_sha256=_digest("main.pak"),
        levels_xml_sha256=_digest("levels.xml"),
        curve_sha256={"curve": _digest("curve")},
        pre_capture_save_sha256=_digest("save"),
        profile_mode=profile_mode,
        renderer_api="Direct3D 9",
        renderer_mode="hardware",
        ball_radius_branch=18,
        os_build="Windows 11 build 26100",
        gpu="NVIDIA GeForce RTX 5090",
        driver="576.80",
        game_settings={"resolution": [1600, 1200]},
        dynamic_difficulty_state={"lives": 3},
    )


def _calibration_sidecar() -> PcCalibrationSidecar:
    points = tuple(
        CalibrationControlPoint(
            raw=(raw_x, raw_y),
            logical=(raw_x * 0.5, raw_y * 0.5),
        )
        for raw_x in (0.0, 800.0, 1598.0)
        for raw_y in (0.0, 600.0, 1198.0)
    )
    return PcCalibrationSidecar(
        raw_width=1600,
        raw_height=1200,
        logical_width=800,
        logical_height=600,
        transform_kind="axis_aligned_affine",
        pixel_center_convention="center_at_integer",
        control_points=points,
        logical_from_raw=(
            (0.5, 0.0, 0.0),
            (0.0, 0.5, 0.0),
            (0.0, 0.0, 1.0),
        ),
        rms_error_px=0.0,
        max_error_px=0.0,
    )


def _trace(
    environment: PcEnvironment,
    *,
    case_id: str = "temple-1_golden_0001",
    capture_fingerprint: str | None = None,
    bound_inputs: bool = True,
    binding_changes: dict[int, dict[str, Any]] | None = None,
) -> PcGoldenTrace:
    first_payload: dict[str, Any]
    second_payload: dict[str, Any]
    first_kind: str
    second_kind: str
    if bound_inputs:
        first_payload = {"dmo_sequence": 0, "dmo_update": 1}
        second_payload = {"dmo_sequence": 1, "dmo_update": 1}
        first_kind = "mouse_move"
        second_kind = "mouse_button"
    else:
        first_payload = {"logical_x": 200.0, "logical_y": 150.0}
        second_payload = {"button": "left", "down": True}
        first_kind = "cursor_move"
        second_kind = "fire_down"
    for index, changes in (binding_changes or {}).items():
        target = first_payload if index == 0 else second_payload
        target.update(changes)

    return PcGoldenTrace(
        case_id=case_id,
        pc_environment_fingerprint=environment.fingerprint,
        capture_contract_fingerprint=(
            capture_fingerprint or _digest("synthetic-unbound-contract")
        ),
        records=(
            GoldenTick(
                tick=0,
                frames=(FrameRef(frame_index=0, pts=0),),
                inputs=(),
                events=(),
                measurements={
                    "score": Measurement(
                        status=MeasurementStatus.OBSERVED,
                        value=0,
                        source="pc_memory_probe",
                    )
                },
            ),
            GoldenTick(
                tick=1,
                frames=(FrameRef(frame_index=1, pts=10),),
                inputs=(
                    GoldenInput(
                        sequence=0,
                        kind=first_kind,
                        pts=10,
                        payload=first_payload,
                    ),
                    GoldenInput(
                        sequence=1,
                        kind=second_kind,
                        pts=10,
                        payload=second_payload,
                    ),
                ),
                events=(),
                measurements={
                    "score": Measurement(
                        status=MeasurementStatus.OBSERVED,
                        value=0,
                        source="pc_memory_probe",
                    )
                },
            ),
            GoldenTick(
                tick=2,
                frames=(FrameRef(frame_index=2, pts=20),),
                inputs=(),
                events=(),
                measurements={
                    "score": Measurement(
                        status=MeasurementStatus.OBSERVED,
                        value=0,
                        source="pc_memory_probe",
                    )
                },
            ),
        ),
    )


def _write_case(
    root: Path,
    *,
    dmo_data: bytes | None = None,
    video_data: bytes | None = None,
    tick_map_data: bytes | None = None,
    calibration_data: bytes | None = None,
    trace: PcGoldenTrace | None = None,
    manifest_case_id: str = "temple-1_golden_0001",
    timeline_changes: dict[str, Any] | None = None,
    coverage: tuple[CoverageRange, ...] | None = None,
    bound_inputs: bool = True,
    binding_changes: dict[int, dict[str, Any]] | None = None,
    profile_mode: str = "tutorials_completed",
) -> tuple[Path, PcGoldenManifest]:
    root.mkdir(parents=True, exist_ok=True)
    environment = _environment(profile_mode=profile_mode)
    dmo_data = _dmo_bytes() if dmo_data is None else dmo_data
    calibration_sidecar = _calibration_sidecar()
    calibration_fit = calibration_sidecar.recompute()
    calibration_data = (
        calibration_sidecar.to_json().encode("utf-8")
        if calibration_data is None
        else calibration_data
    )
    payloads = {
        "video": (
            b"lossless-video-evidence"
            if video_data is None
            else video_data
        ),
        "input_dmo": dmo_data,
        "tick_map": (
            b"tick,pts\n0,0\n1,10\n2,20\n"
            if tick_map_data is None
            else tick_map_data
        ),
        "calibration": calibration_data,
    }
    paths = {
        "video": "capture.mkv",
        "input_dmo": "input.dmo",
        "tick_map": "tick-map.csv",
        "calibration": "calibration.json",
        "trace": "trace.ndjson",
    }
    artifacts = {
        name: ArtifactSpec(
            path=paths[name],
            sha256=_digest(payload),
            bytes=len(payload),
        )
        for name, payload in payloads.items()
    }
    timeline_values: dict[str, Any] = {
        "artifact": "input_dmo",
        "format": DMO_FORMAT,
        "file_id": DMO_FILE_ID,
        "dmo_version": DMO_VERSION,
        "product_version": _PRODUCT_VERSION,
        "random_seed": _RANDOM_SEED,
        "length_updates": _LENGTH_UPDATES,
        "native_tick_offset": 0,
    }
    timeline_values.update(timeline_changes or {})
    coverage = coverage or (
        CoverageRange(
            channel="frames",
            start_tick=0,
            end_tick=2,
            status=CoverageStatus.COMPLETE,
            required=True,
        ),
        CoverageRange(
            channel="input_events",
            start_tick=0,
            end_tick=2,
            status=CoverageStatus.COMPLETE,
            required=True,
        ),
        CoverageRange(
            channel="score",
            start_tick=0,
            end_tick=2,
            status=CoverageStatus.COMPLETE,
            required=True,
        ),
    )
    scenario = Scenario(
        level_id="temple-1",
        hard=False,
        curve_index=0,
        gun_index=0,
        mode="adventure",
        profile_mode=profile_mode,
    )
    input_timeline = InputTimeline(**timeline_values)
    video = VideoMetadata(
        artifact="video",
        width=1600,
        height=1200,
        codec="ffv1",
        pixel_format="bgra",
        time_base=(1, 1000),
        nominal_fps=(100, 1),
        frame_count=3,
        first_pts=0,
        last_pts=20,
        cfr=True,
        dropped_frames=0,
        duplicate_frames=0,
    )
    clock = TickClock(
        logic_hz=(100, 1),
        tick_start=0,
        tick_end=2,
        tick0_video_pts=0,
        sample_phase="post_update_presented",
        mapping_kind="per_tick_pts_table",
        tick_map_artifact="tick_map",
        uncertainty_ticks=0.0,
    )
    coordinates = CoordinateCalibration(
        raw_width=1600,
        raw_height=1200,
        logical_width=800,
        logical_height=600,
        transform_kind="axis_aligned_affine",
        pixel_center_convention="center_at_integer",
        logical_from_raw=calibration_fit.logical_from_raw,
        rms_error_px=calibration_fit.rms_error_px,
        max_error_px=calibration_fit.max_error_px,
        calibration_artifact="calibration",
    )
    comparison_contract = ComparisonContract(
        event_tick_tolerance=0,
        center_l2_tolerance_px=0.5,
        waypoint_abs_tolerance=0.01,
        max_drift_per_100_ticks=0.05,
    )
    contract_fingerprint = capture_contract_fingerprint(
        case_id=manifest_case_id,
        scenario=scenario,
        pc_environment_fingerprint=environment.fingerprint,
        artifacts=artifacts,
        input_timeline=input_timeline,
        trace_artifact="trace",
        video=video,
        clock=clock,
        coordinates=coordinates,
        coverage=coverage,
        comparison_contract=comparison_contract,
    )
    if trace is None:
        trace = _trace(
            environment,
            capture_fingerprint=contract_fingerprint,
            bound_inputs=bound_inputs,
            binding_changes=binding_changes,
        )
    else:
        trace = replace(
            trace,
            capture_contract_fingerprint=contract_fingerprint,
        )
    payloads["trace"] = trace.to_ndjson().encode("utf-8")
    artifacts["trace"] = ArtifactSpec(
        path=paths["trace"],
        sha256=_digest(payloads["trace"]),
        bytes=len(payloads["trace"]),
    )
    manifest = PcGoldenManifest(
        case_id=manifest_case_id,
        scenario=scenario,
        pc_environment=environment,
        pc_environment_fingerprint=environment.fingerprint,
        artifacts=artifacts,
        input_timeline=input_timeline,
        trace_artifact="trace",
        video=video,
        clock=clock,
        coordinates=coordinates,
        coverage=coverage,
        comparison_contract=comparison_contract,
        producer={"tool": "test-capture", "version": 1},
    )
    for name, artifact in manifest.artifacts.items():
        (root / artifact.path).write_bytes(payloads[name])
    manifest_path = root / "manifest.json"
    manifest.write_json(manifest_path)
    return manifest_path, manifest


_PROTOCOL_NONCE = "0123456789abcdef0123456789abcdef"


def _protocol_snapshot(
    *,
    phase: str,
    perf_counter_ns: int,
    file_data: bytes,
    last_game_pid: int = 99,
    registration_data: int = 0x12345678,
) -> PcStateSnapshot:
    return PcStateSnapshot(
        session_nonce=_PROTOCOL_NONCE,
        phase=phase,
        captured_perf_counter_ns=perf_counter_ns,
        files=(
            SnapshotFile(
                path="users.dat",
                data=file_data,
                windows_attributes=0x20,
            ),
        ),
        registry_nodes=(
            SnapshotRegistryNode(
                root=REGISTRY_ROOTS[0],
                subkey="",
                exists=True,
                values=(
                    SnapshotRegistryValue(
                        name="InProgress",
                        value_type=4,
                        data=b"\0\0\0\0",
                    ),
                    SnapshotRegistryValue(
                        name="LastGamePID",
                        value_type=4,
                        data=last_game_pid.to_bytes(4, "little"),
                    ),
                    SnapshotRegistryValue(
                        name="RegExData",
                        value_type=4,
                        data=registration_data.to_bytes(4, "little"),
                    ),
                ),
            ),
            SnapshotRegistryNode(
                root=REGISTRY_ROOTS[1],
                subkey="",
                exists=True,
                values=(
                    SnapshotRegistryValue(
                        name="LastShutdownOK",
                        value_type=4,
                        data=b"\1\0\0\0",
                    ),
                ),
            ),
        ),
    )


def _protocol_capture_metadata(
    *,
    process_id: int,
    creation_filetime: int,
    executable_sha256: str,
    capture_start_ns: int,
    capture_end_ns: int,
    run_id: str,
) -> tuple[bytes, PcDxgiCaptureMetadata]:
    raw_sha = _digest(f"{run_id}-raw")
    csv_sha = _digest(f"{run_id}-csv")
    data = {
        "schema": "zuma-rl.dxgi-bgra-capture",
        "version": 2,
        "status": "acquisition_complete",
        "target_process": "popcapgame1.exe",
        "device_index": 0,
        "output_index": 0,
        "global_region": [0, 0, 1600, 1200],
        "output_local_region": [0, 0, 1600, 1200],
        "width": 1600,
        "height": 1200,
        "pixel_format": "bgra",
        "bytes_per_pixel": 4,
        "frame_budget_fps": 200,
        "capture_mode": "every_new_present",
        "capture_storage_mode": "memory_then_publish",
        "capture_surface": "dxgi_desktop_client_region_crop",
        "foreground_geometry_guard": True,
        "topmost_or_injected_overlay_detection": False,
        "requires_full_frame_visual_review": True,
        "requested_duration_seconds": 1.0,
        "capture_start_perf_counter_ns": capture_start_ns,
        "capture_end_perf_counter_ns": capture_end_ns,
        "capture_elapsed_seconds": (
            (capture_end_ns - capture_start_ns) / 1_000_000_000
        ),
        "estimated_raw_budget_bytes": 1_000_000,
        "frame_budget": 202,
        "frame_count": 3,
        "warmup_baseline_present_ticks": 100,
        "first_present_ticks": 110,
        "last_present_ticks": 130,
        "present_span_ticks": 20,
        "present_span_seconds": 0.000002,
        "observed_mean_present_fps": 1_000_000.0,
        "qpc_frequency": 10_000_000,
        "missed_presentations": 0,
        "idle_poll_count": 1,
        "pointer_only_update_count": 0,
        "warmup_idle_poll_count": 0,
        "warmup_pointer_only_update_count": 0,
        "runtime_versions": {
            "comtypes": "1.4.16",
            "dxcam": "0.3.0",
            "numpy": "2.5.1",
        },
        "runtime_versions_verified": True,
        "source_identity": {},
        "target_identity": {
            "process_id": process_id,
            "process_creation_filetime_100ns": creation_filetime,
            "executable_bytes": 10,
            "executable_sha256": executable_sha256,
            "executable_hash_provenance": {
                "method": "direct_file_sha256"
            },
            "window_handle_hex": "0x0000000000000001",
            "window_client_region": [0, 0, 1600, 1200],
        },
        "raw_bytes": 100,
        "raw_sha256": raw_sha,
        "frames_csv_bytes": 100,
        "frames_csv_rows": 3,
        "frames_csv_sha256": csv_sha,
        "aggregate_pixel_sha256": _digest(f"{run_id}-pixels"),
        "frames_csv": "frames.csv",
        "raw_frames": "frames.bgra.raw",
    }
    text = (
        json.dumps(
            data,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )
    parsed = PcDxgiCaptureMetadata.from_json(text)
    return text.encode("ascii"), parsed


def _write_protocol_case(
    root: Path,
    *,
    pixel_comparison: ReplayPixelComparisonContract | None = None,
) -> tuple[
    Path,
    PcGoldenManifest,
    dict[str, PcDxgiCaptureMetadata],
]:
    root.mkdir(parents=True, exist_ok=True)
    snapshots = {
        "pre": _protocol_snapshot(
            phase="pre",
            perf_counter_ns=100,
            file_data=b"pre-state",
        ),
        "r1-start": _protocol_snapshot(
            phase="r1-start",
            perf_counter_ns=200,
            file_data=b"pre-state",
        ),
        "r1-end": _protocol_snapshot(
            phase="r1-end",
            perf_counter_ns=500,
            file_data=b"end-state",
            last_game_pid=10,
            registration_data=0x11111111,
        ),
        "r2-start": _protocol_snapshot(
            phase="r2-start",
            perf_counter_ns=600,
            file_data=b"pre-state",
        ),
        "r2-end": _protocol_snapshot(
            phase="r2-end",
            perf_counter_ns=900,
            file_data=b"end-state",
            last_game_pid=20,
            registration_data=0x22222222,
        ),
        "restored": _protocol_snapshot(
            phase="restored",
            perf_counter_ns=1000,
            file_data=b"pre-state",
        ),
    }
    environment = replace(
        _environment(),
        pre_capture_save_sha256=snapshots["pre"].state_root,
    )
    process_timeline = PcProcessTimeline(
        session_nonce=_PROTOCOL_NONCE,
        events=(
            PcProcessEvent(
                sequence=0,
                kind="start",
                process_id=10,
                process_creation_filetime_100ns=1000,
                perf_counter_ns=250,
                exit_code=None,
                executable_sha256=environment.executable_sha256,
            ),
            PcProcessEvent(
                sequence=1,
                kind="stop",
                process_id=10,
                process_creation_filetime_100ns=1000,
                perf_counter_ns=450,
                exit_code=0,
                executable_sha256=environment.executable_sha256,
            ),
            PcProcessEvent(
                sequence=2,
                kind="start",
                process_id=20,
                process_creation_filetime_100ns=2000,
                perf_counter_ns=650,
                exit_code=None,
                executable_sha256=environment.executable_sha256,
            ),
            PcProcessEvent(
                sequence=3,
                kind="stop",
                process_id=20,
                process_creation_filetime_100ns=2000,
                perf_counter_ns=850,
                exit_code=0,
                executable_sha256=environment.executable_sha256,
            ),
        ),
    )
    phase_artifact_names = {
        phase: f"state_{phase.replace('-', '_')}"
        for phase in snapshots
    }
    journal = PcSaveJournal.build(
        session_nonce=_PROTOCOL_NONCE,
        records=(
            {
                "phase": "pre",
                "snapshot_artifact": phase_artifact_names["pre"],
                "snapshot_state_root": snapshots["pre"].state_root,
                "snapshot_perf_counter_ns": 100,
                "process_id": None,
                "process_creation_filetime_100ns": None,
                "exit_code": None,
            },
            {
                "phase": "r1-start",
                "snapshot_artifact": phase_artifact_names["r1-start"],
                "snapshot_state_root": snapshots["r1-start"].state_root,
                "snapshot_perf_counter_ns": 200,
                "process_id": 10,
                "process_creation_filetime_100ns": 1000,
                "exit_code": None,
            },
            {
                "phase": "r1-end",
                "snapshot_artifact": phase_artifact_names["r1-end"],
                "snapshot_state_root": snapshots["r1-end"].state_root,
                "snapshot_perf_counter_ns": 500,
                "process_id": 10,
                "process_creation_filetime_100ns": 1000,
                "exit_code": 0,
            },
            {
                "phase": "r2-start",
                "snapshot_artifact": phase_artifact_names["r2-start"],
                "snapshot_state_root": snapshots["r2-start"].state_root,
                "snapshot_perf_counter_ns": 600,
                "process_id": 20,
                "process_creation_filetime_100ns": 2000,
                "exit_code": None,
            },
            {
                "phase": "r2-end",
                "snapshot_artifact": phase_artifact_names["r2-end"],
                "snapshot_state_root": snapshots["r2-end"].state_root,
                "snapshot_perf_counter_ns": 900,
                "process_id": 20,
                "process_creation_filetime_100ns": 2000,
                "exit_code": 0,
            },
            {
                "phase": "restored",
                "snapshot_artifact": phase_artifact_names["restored"],
                "snapshot_state_root": snapshots["restored"].state_root,
                "snapshot_perf_counter_ns": 1000,
                "process_id": None,
                "process_creation_filetime_100ns": None,
                "exit_code": None,
            },
        ),
    )
    capture_r1, parsed_r1 = _protocol_capture_metadata(
        process_id=10,
        creation_filetime=1000,
        executable_sha256=environment.executable_sha256,
        capture_start_ns=300,
        capture_end_ns=400,
        run_id="r1",
    )
    capture_r2, parsed_r2 = _protocol_capture_metadata(
        process_id=20,
        creation_filetime=2000,
        executable_sha256=environment.executable_sha256,
        capture_start_ns=700,
        capture_end_ns=800,
        run_id="r2",
    )
    update_r1 = PcFrameworkUpdateMap(
        capture_metadata_sha256=_digest(capture_r1),
        process_id=10,
        process_creation_filetime_100ns=1000,
        executable_sha256=environment.executable_sha256,
        records=tuple(
            FrameworkUpdateRecord(
                sequence=index,
                present_ticks=110 + index * 10,
                host_perf_counter_ns=325 + index * 25,
                update_before=index,
                update_after=index,
            )
            for index in range(3)
        ),
    )
    update_r2 = PcFrameworkUpdateMap(
        capture_metadata_sha256=_digest(capture_r2),
        process_id=20,
        process_creation_filetime_100ns=2000,
        executable_sha256=environment.executable_sha256,
        records=tuple(
            FrameworkUpdateRecord(
                sequence=index,
                present_ticks=110 + index * 10,
                host_perf_counter_ns=725 + index * 25,
                update_before=index,
                update_after=index,
            )
            for index in range(3)
        ),
    )
    calibration = _calibration_sidecar()
    calibration_fit = calibration.recompute()
    scenario = Scenario(
        level_id="temple-1",
        hard=False,
        curve_index=0,
        gun_index=0,
        mode="adventure",
        profile_mode="tutorials_completed",
    )
    timeline = InputTimeline(
        artifact="input_dmo",
        format=DMO_FORMAT,
        file_id=DMO_FILE_ID,
        dmo_version=DMO_VERSION,
        product_version=_PRODUCT_VERSION,
        random_seed=_RANDOM_SEED,
        length_updates=_LENGTH_UPDATES,
        native_tick_offset=0,
    )
    video_r1 = VideoMetadata(
        artifact="video",
        width=1600,
        height=1200,
        codec="ffv1",
        pixel_format="bgra",
        time_base=(1, 1000),
        nominal_fps=(100, 1),
        frame_count=3,
        first_pts=0,
        last_pts=20,
        cfr=True,
        dropped_frames=0,
        duplicate_frames=0,
    )
    video_r2 = replace(video_r1, artifact="video_r2")
    clock_r1 = TickClock(
        logic_hz=(100, 1),
        tick_start=0,
        tick_end=2,
        tick0_video_pts=0,
        sample_phase="post_update_presented",
        mapping_kind="per_tick_pts_table",
        tick_map_artifact="tick_map",
        uncertainty_ticks=0.0,
    )
    clock_r2 = replace(clock_r1, tick_map_artifact="tick_map_r2")
    coordinates = CoordinateCalibration(
        raw_width=1600,
        raw_height=1200,
        logical_width=800,
        logical_height=600,
        transform_kind="axis_aligned_affine",
        pixel_center_convention="center_at_integer",
        logical_from_raw=calibration_fit.logical_from_raw,
        rms_error_px=calibration_fit.rms_error_px,
        max_error_px=calibration_fit.max_error_px,
        calibration_artifact="calibration",
    )
    coverage = (
        CoverageRange(
            channel="frames",
            start_tick=0,
            end_tick=2,
            status=CoverageStatus.COMPLETE,
            required=True,
        ),
        CoverageRange(
            channel="input_events",
            start_tick=0,
            end_tick=2,
            status=CoverageStatus.COMPLETE,
            required=True,
        ),
        CoverageRange(
            channel="score",
            start_tick=0,
            end_tick=2,
            status=CoverageStatus.COMPLETE,
            required=True,
        ),
    )
    comparison = ComparisonContract(
        event_tick_tolerance=0,
        center_l2_tolerance_px=0.5,
        waypoint_abs_tolerance=0.01,
        max_drift_per_100_ticks=0.05,
    )
    payloads: dict[str, bytes] = {
        "video": b"lossless-video-r1",
        "video_r2": b"lossless-video-r2",
        "input_dmo": _dmo_bytes(),
        "tick_map": b"tick,pts\n0,0\n1,10\n2,20\n",
        "tick_map_r2": b"tick,pts\n0,0\n1,10\n2,20\n",
        "calibration": calibration.to_json().encode("utf-8"),
        "capture_metadata_r1": capture_r1,
        "capture_metadata_r2": capture_r2,
        "framework_update_r1": update_r1.to_json().encode("ascii"),
        "framework_update_r2": update_r2.to_json().encode("ascii"),
        "save_journal": journal.to_ndjson().encode("ascii"),
        "process_timeline": process_timeline.to_bytes(),
    }
    paths = {
        "video": "capture-r1.mkv",
        "video_r2": "capture-r2.mkv",
        "input_dmo": "input.dmo",
        "tick_map": "tick-map-r1.csv",
        "tick_map_r2": "tick-map-r2.csv",
        "calibration": "calibration.json",
        "capture_metadata_r1": "capture-metadata-r1.json",
        "capture_metadata_r2": "capture-metadata-r2.json",
        "framework_update_r1": "framework-update-r1.json",
        "framework_update_r2": "framework-update-r2.json",
        "save_journal": "save-journal.ndjson",
        "process_timeline": "process-timeline.bin",
        "trace": "trace-r1.ndjson",
        "trace_r2": "trace-r2.ndjson",
    }
    for phase, snapshot in snapshots.items():
        name = phase_artifact_names[phase]
        payloads[name] = snapshot.to_json().encode("ascii")
        paths[name] = f"{name}.json"
    artifacts = {
        name: ArtifactSpec(
            path=paths[name],
            sha256=_digest(payload),
            bytes=len(payload),
        )
        for name, payload in payloads.items()
    }
    save_contract = SaveTransactionContract(
        journal_artifact="save_journal",
        process_timeline_artifact="process_timeline",
        pre_snapshot_artifact=phase_artifact_names["pre"],
        restored_snapshot_artifact=phase_artifact_names["restored"],
        runs=(
            SaveRunContract(
                run_id="r1",
                start_snapshot_artifact=phase_artifact_names["r1-start"],
                end_snapshot_artifact=phase_artifact_names["r1-end"],
                process_id=10,
                process_creation_filetime_100ns=1000,
            ),
            SaveRunContract(
                run_id="r2",
                start_snapshot_artifact=phase_artifact_names["r2-start"],
                end_snapshot_artifact=phase_artifact_names["r2-end"],
                process_id=20,
                process_creation_filetime_100ns=2000,
            ),
        ),
        volatile_registry_roles=SAVE_VOLATILE_REGISTRY_ROLES,
    )
    replay_contract = ReplayDeterminismContract(
        runs=(
            ReplayRunContract(
                run_id="r1",
                trace_artifact="trace",
                video=video_r1,
                clock=clock_r1,
                capture_metadata_artifact="capture_metadata_r1",
                framework_update_artifact="framework_update_r1",
            ),
            ReplayRunContract(
                run_id="r2",
                trace_artifact="trace_r2",
                video=video_r2,
                clock=clock_r2,
                capture_metadata_artifact="capture_metadata_r2",
                framework_update_artifact="framework_update_r2",
            ),
        ),
        pixel_comparison=pixel_comparison,
    )
    contract_fingerprint = capture_contract_fingerprint(
        case_id="temple-1_golden_0001",
        scenario=scenario,
        pc_environment_fingerprint=environment.fingerprint,
        artifacts=artifacts,
        input_timeline=timeline,
        trace_artifact="trace",
        video=video_r1,
        clock=clock_r1,
        coordinates=coordinates,
        coverage=coverage,
        comparison_contract=comparison,
        save_transaction=save_contract,
        replay_determinism=replay_contract,
    )
    trace_r1 = _trace(
        environment,
        capture_fingerprint=contract_fingerprint,
    )
    trace_r2 = _trace(
        environment,
        capture_fingerprint=contract_fingerprint,
    )
    for name, trace in (("trace", trace_r1), ("trace_r2", trace_r2)):
        payloads[name] = trace.to_ndjson().encode("utf-8")
        artifacts[name] = ArtifactSpec(
            path=paths[name],
            sha256=_digest(payloads[name]),
            bytes=len(payloads[name]),
        )
    manifest = PcGoldenManifest(
        case_id="temple-1_golden_0001",
        scenario=scenario,
        pc_environment=environment,
        pc_environment_fingerprint=environment.fingerprint,
        artifacts=artifacts,
        input_timeline=timeline,
        trace_artifact="trace",
        video=video_r1,
        clock=clock_r1,
        coordinates=coordinates,
        coverage=coverage,
        comparison_contract=comparison,
        producer={"tool": "protocol-test", "version": 1},
        save_transaction=save_contract,
        replay_determinism=replay_contract,
    )
    for name, artifact in manifest.artifacts.items():
        (root / artifact.path).write_bytes(payloads[name])
    manifest_path = root / "manifest.json"
    manifest.write_json(manifest_path)
    return (
        manifest_path,
        manifest,
        {"r1": parsed_r1, "r2": parsed_r2},
    )


def _check_by_name(report, name: str):
    return next(check for check in report.checks if check.name == name)


def test_certification_rejects_case_without_raw_recording_provenance(
    tmp_path: Path,
) -> None:
    manifest_path, _ = _write_case(tmp_path)

    report = verify_pc_golden_case(
        manifest_path,
        require_certifying_dmo_provenance=True,
    )

    assert report.status is ComparisonStatus.FAIL
    assert report.exit_code == 1
    assert report.checks[-1].name == "dmo_source_provenance"
    assert "provenance-bound raw retail recording" in report.reasons[0]


def test_structurally_valid_case_is_semantically_incomparable_and_read_only(
    tmp_path: Path,
) -> None:
    manifest_path, manifest = _write_case(tmp_path)
    sources = (manifest_path,) + tuple(
        tmp_path / artifact.path
        for artifact in manifest.artifacts.values()
    )
    before = {path: path.read_bytes() for path in sources}

    report = verify_pc_golden_case(manifest_path)

    assert report.status is ComparisonStatus.INCOMPARABLE
    assert report.exit_code == 2
    assert report.to_dict()["schema"] == REPORT_SCHEMA
    assert [check.name for check in report.checks] == [
        "manifest",
        "artifact_budget",
        "artifacts",
        "trace",
        "tick_map",
        "calibration",
        "dmo",
        "input_sequence",
        "coverage",
        "trace_coverage",
        "video_semantics",
        "scenario_semantics",
        "save_transaction",
        "replay_determinism",
    ]
    assert all(
        check.status is VerificationCheckStatus.PASS
        for check in report.checks[:11]
    )
    assert all(
        check.status is VerificationCheckStatus.INCOMPARABLE
        for check in report.checks[11:]
    )
    assert report.summary["dmo"]["file_id"] == "0x42BEEF78"
    assert report.summary["dmo"]["input_command_count"] == 2
    assert (
        report.summary["semantic_limits"]["save_restoration_attested"]
        is False
    )
    assert (
        report.summary["semantic_limits"][
            "deterministic_replay_attested"
        ]
        is False
    )
    assert report.summary["semantic_limits"]["video_stream_decoded"] is True
    assert report.summary["video"]["frame_count"] == 3
    serialized = report.to_json()
    assert _SENSITIVE_DMO_VALUE.decode("ascii") not in serialized
    assert '"payload"' not in serialized
    assert {path: path.read_bytes() for path in sources} == before


def _protocol_video_inspection(
    manifest: PcGoldenManifest,
    *,
    run_id: str,
    capture_metadata: PcDxgiCaptureMetadata,
    frame_hashes: tuple[str, ...] | None = None,
    interior_frame_hashes: tuple[str, ...] = (),
    last_rows_rgb24: tuple[bytes, ...] = (),
) -> PcVideoInspection:
    assert manifest.replay_determinism is not None
    run = next(
        item
        for item in manifest.replay_determinism.runs
        if item.run_id == run_id
    )
    artifact = manifest.artifacts[run.video.artifact]
    return PcVideoInspection(
        artifact_sha256=artifact.sha256,
        artifact_bytes=artifact.bytes,
        width=run.video.width,
        height=run.video.height,
        codec=run.video.codec,
        pixel_format=run.video.pixel_format,
        time_base=run.video.time_base,
        nominal_fps=run.video.nominal_fps,
        frame_count=run.video.frame_count,
        first_pts=run.video.first_pts,
        last_pts=run.video.last_pts,
        frame_pts=(0, 10, 20),
        cfr=run.video.cfr,
        dropped_frames=run.video.dropped_frames,
        duplicate_frames=run.video.duplicate_frames,
        normalized_pixel_format="rgb24",
        normalized_decoded_bytes=(
            run.video.width
            * run.video.height
            * 3
            * run.video.frame_count
        ),
        decoded_pixel_sha256=_digest("protocol-decoded-rgb24"),
        frame_pixel_sha256=(
            frame_hashes
            if frame_hashes is not None
            else (
                _digest("protocol-frame-0"),
                _digest("protocol-frame-1"),
                _digest("protocol-frame-2"),
            )
        ),
        frame_pixel_sha256_without_last_row=interior_frame_hashes,
        frame_last_row_rgb24=last_rows_rgb24,
        dxgi_source_metadata_sha256=(
            manifest.artifacts[run.capture_metadata_artifact].sha256
        ),
        dxgi_source_raw_sha256=capture_metadata.raw_sha256,
        dxgi_source_frames_csv_sha256=(
            capture_metadata.frames_csv_sha256
        ),
        dxgi_source_present_ticks=(110, 120, 130),
    )


def test_framework_update_map_skips_boundary_frames_and_accepts_any_stable_pts(
    tmp_path: Path,
) -> None:
    _, manifest, captures = _write_protocol_case(tmp_path)
    assert manifest.replay_determinism is not None
    run = manifest.replay_determinism.runs[0]
    capture = captures["r1"]
    update_map = PcFrameworkUpdateMap(
        capture_metadata_sha256=(
            manifest.artifacts[run.capture_metadata_artifact].sha256
        ),
        process_id=capture.process_id,
        process_creation_filetime_100ns=(
            capture.process_creation_filetime_100ns
        ),
        executable_sha256=manifest.pc_environment.executable_sha256,
        records=(
            FrameworkUpdateRecord(0, 110, 320, 0, 0),
            FrameworkUpdateRecord(1, 115, 330, 0, 0),
            FrameworkUpdateRecord(2, 118, 340, 0, 1),
            FrameworkUpdateRecord(3, 120, 350, 1, 1),
            FrameworkUpdateRecord(4, 130, 375, 2, 2),
        ),
    )
    inspection = replace(
        _protocol_video_inspection(
            manifest,
            run_id="r1",
            capture_metadata=capture,
        ),
        frame_count=5,
        frame_pts=(0, 5, 6, 10, 20),
        normalized_decoded_bytes=(
            run.video.width * run.video.height * 3 * 5
        ),
        frame_pixel_sha256=tuple(
            _digest(f"candidate-frame-{index}")
            for index in range(5)
        ),
        dxgi_source_present_ticks=(110, 115, 118, 120, 130),
    )
    tick_map = PcTickMap(
        records=(
            TickPts(0, 5),
            TickPts(1, 10),
            TickPts(2, 20),
        )
    )

    verifier_module._validate_framework_update_map(
        manifest,
        run=run,
        update_map=update_map,
        capture_metadata=capture,
        capture_metadata_sha256=(
            manifest.artifacts[run.capture_metadata_artifact].sha256
        ),
        inspection=inspection,
        tick_map=tick_map,
    )

    with pytest.raises(
        PcGoldenValidationError,
        match="not stable framework-update/video candidates",
    ):
        verifier_module._validate_framework_update_map(
            manifest,
            run=run,
            update_map=update_map,
            capture_metadata=capture,
            capture_metadata_sha256=(
                manifest.artifacts[
                    run.capture_metadata_artifact
                ].sha256
            ),
            inspection=inspection,
            tick_map=PcTickMap(
                records=(
                    TickPts(0, 7),
                    TickPts(1, 10),
                    TickPts(2, 20),
                )
            ),
        )


def test_v4_raw_save_and_double_replay_evidence_can_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, manifest, captures = _write_protocol_case(tmp_path)

    def inspect_primary(
        path: Path,
        supplied_manifest: PcGoldenManifest,
    ) -> PcVideoInspection:
        del path
        assert supplied_manifest.case_id == manifest.case_id
        return _protocol_video_inspection(
            manifest,
            run_id="r1",
            capture_metadata=captures["r1"],
        )

    def inspect_replay(
        path: Path,
        *,
        metadata: VideoMetadata,
        artifact: ArtifactSpec,
    ) -> PcVideoInspection:
        del path, artifact
        run_id = "r2" if metadata.artifact == "video_r2" else "r1"
        return _protocol_video_inspection(
            manifest,
            run_id=run_id,
            capture_metadata=captures[run_id],
        )

    monkeypatch.setattr(verifier_module, "inspect_pc_video", inspect_primary)
    monkeypatch.setattr(
        verifier_module,
        "inspect_pc_video_artifact",
        inspect_replay,
    )
    monkeypatch.setattr(
        verifier_module,
        "_scenario_semantics_mismatches",
        lambda supplied_manifest, original_root: (),
    )

    report = verify_pc_golden_case(
        manifest_path,
        original_root=tmp_path,
    )

    assert report.status is ComparisonStatus.PASS
    assert report.exit_code == 0
    assert all(
        check.status is VerificationCheckStatus.PASS
        for check in report.checks
    )
    assert _check_by_name(
        report,
        "save_transaction",
    ).details["pre_restored_root_matched"] is True
    assert _check_by_name(
        report,
        "replay_determinism",
    ).details["full_viewport_rgb24_per_tick_matched"] is True
    assert (
        report.summary["semantic_limits"]["save_restoration_attested"]
        is True
    )
    assert (
        report.summary["semantic_limits"][
            "deterministic_replay_attested"
        ]
        is True
    )


@pytest.mark.parametrize(
    ("edge_budget", "expected_status"),
    ((5, ComparisonStatus.PASS), (4, ComparisonStatus.FAIL)),
)
def test_v4_replay_bottom_raster_edge_is_independently_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    edge_budget: int,
    expected_status: ComparisonStatus,
) -> None:
    pixel_comparison = ReplayPixelComparisonContract(
        mode="exact_except_bounded_bottom_raster_edge",
        excluded_bottom_rows=1,
        maximum_excluded_edge_mismatches=edge_budget,
    )
    manifest_path, manifest, captures = _write_protocol_case(
        tmp_path,
        pixel_comparison=pixel_comparison,
    )
    interior_hashes = tuple(
        _digest(f"interior-frame-{index}") for index in range(3)
    )
    reference_row = bytes(manifest.video.width * 3)
    changed_row = bytearray(reference_row)
    for pixel in range(5):
        changed_row[pixel * 3] = 1
    reference_rows = (reference_row,) * 3
    changed_rows = (bytes(changed_row),) * 3

    def inspection(run_id: str) -> PcVideoInspection:
        return _protocol_video_inspection(
            manifest,
            run_id=run_id,
            capture_metadata=captures[run_id],
            frame_hashes=tuple(
                _digest(f"{run_id}-full-frame-{index}")
                for index in range(3)
            ),
            interior_frame_hashes=interior_hashes,
            last_rows_rgb24=(
                reference_rows if run_id == "r1" else changed_rows
            ),
        )

    monkeypatch.setattr(
        verifier_module,
        "inspect_pc_video",
        lambda path, supplied_manifest: inspection("r1"),
    )
    monkeypatch.setattr(
        verifier_module,
        "inspect_pc_video_artifact",
        lambda path, *, metadata, artifact: inspection("r2"),
    )
    monkeypatch.setattr(
        verifier_module,
        "_scenario_semantics_mismatches",
        lambda supplied_manifest, original_root: (),
    )

    report = verify_pc_golden_case(
        manifest_path,
        original_root=tmp_path,
    )

    assert report.status is expected_status
    replay_check = _check_by_name(report, "replay_determinism")
    if expected_status is ComparisonStatus.PASS:
        assert replay_check.status is VerificationCheckStatus.PASS
        assert replay_check.details[
            "full_viewport_rgb24_per_tick_matched"
        ] is False
        assert replay_check.details[
            "gameplay_viewport_rgb24_per_tick_matched"
        ] is True
        assert replay_check.details[
            "maximum_observed_excluded_edge_mismatches"
        ] == 5
    else:
        assert replay_check.status is VerificationCheckStatus.FAIL


def test_v4_double_replay_pixel_difference_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, manifest, captures = _write_protocol_case(tmp_path)

    monkeypatch.setattr(
        verifier_module,
        "inspect_pc_video",
        lambda path, supplied_manifest: _protocol_video_inspection(
            manifest,
            run_id="r1",
            capture_metadata=captures["r1"],
        ),
    )

    def inspect_replay(
        path: Path,
        *,
        metadata: VideoMetadata,
        artifact: ArtifactSpec,
    ) -> PcVideoInspection:
        del path, artifact
        assert metadata.artifact == "video_r2"
        return _protocol_video_inspection(
            manifest,
            run_id="r2",
            capture_metadata=captures["r2"],
            frame_hashes=(
                _digest("protocol-frame-0"),
                _digest("different-frame"),
                _digest("protocol-frame-2"),
            ),
        )

    monkeypatch.setattr(
        verifier_module,
        "inspect_pc_video_artifact",
        inspect_replay,
    )

    report = verify_pc_golden_case(manifest_path)

    assert report.status is ComparisonStatus.FAIL
    assert report.exit_code == 1
    assert report.checks[-1].name == "replay_determinism"
    assert report.checks[-1].status is VerificationCheckStatus.FAIL


def test_unavailable_video_decoder_is_incomparable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, _ = _write_case(tmp_path)

    def unavailable(path: Path, manifest: PcGoldenManifest) -> None:
        del path, manifest
        raise VideoDecoderUnavailable("decoder unavailable")

    monkeypatch.setattr(verifier_module, "inspect_pc_video", unavailable)
    report = verify_pc_golden_case(manifest_path)

    video_check = _check_by_name(report, "video_semantics")
    assert report.status is ComparisonStatus.INCOMPARABLE
    assert video_check.status is VerificationCheckStatus.INCOMPARABLE
    assert video_check.details["decoder_available"] is False
    assert (
        report.summary["semantic_limits"]["video_stream_decoded"]
        is False
    )


def test_video_semantic_mismatch_is_fail_closed_and_path_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, _ = _write_case(tmp_path)
    private_value = "PRIVATE-VIDEO-PATH/capture.mkv"

    def mismatch(path: Path, manifest: PcGoldenManifest) -> None:
        del path, manifest
        raise VideoMetadataMismatch(private_value)

    monkeypatch.setattr(verifier_module, "inspect_pc_video", mismatch)
    report = verify_pc_golden_case(manifest_path)

    assert report.status is ComparisonStatus.FAIL
    assert report.checks[-1].name == "video_semantics"
    assert report.checks[-1].details["error_type"] == (
        "VideoMetadataMismatch"
    )
    assert private_value not in report.to_json()


def test_real_ffv1_video_decoder_is_integrated_with_verifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    av = pytest.importorskip("av", minversion="18")
    np = pytest.importorskip("numpy")
    from fractions import Fraction

    from zuma_rl.pc_video import inspect_pc_video as real_inspect_pc_video

    source = tmp_path / "source.mkv"
    container = av.open(str(source), mode="w")
    stream = container.add_stream("ffv1", rate=Fraction(100, 1))
    stream.width = 1600
    stream.height = 1200
    stream.pix_fmt = "bgra"
    stream.time_base = Fraction(1, 1000)
    try:
        for index, pts in enumerate((0, 10, 20)):
            pixels = np.full(
                (1200, 1600, 4),
                index * 30,
                dtype=np.uint8,
            )
            frame = av.VideoFrame.from_ndarray(pixels, format="bgra")
            frame.pts = pts
            frame.time_base = Fraction(1, 1000)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()

    manifest_path, _ = _write_case(
        tmp_path / "case",
        video_data=source.read_bytes(),
    )
    monkeypatch.setattr(
        verifier_module,
        "inspect_pc_video",
        real_inspect_pc_video,
    )

    report = verify_pc_golden_case(manifest_path)

    assert report.status is ComparisonStatus.INCOMPARABLE
    assert (
        _check_by_name(report, "video_semantics").status
        is VerificationCheckStatus.PASS
    )
    assert report.summary["video"][
        "referenced_frame_pts_bound_to_trace"
    ] is True
    assert report.summary["video"]["decoded_pixel_sha256"].startswith(
        "sha256:"
    )


def test_decoded_frame_pts_are_bound_to_trace_and_tick_map(
    tmp_path: Path,
) -> None:
    environment = _environment()
    reference = _trace(environment)
    tick_one = replace(
        reference.records[1],
        frames=(FrameRef(frame_index=1, pts=9),),
        inputs=tuple(
            replace(item, pts=9) for item in reference.records[1].inputs
        ),
    )
    altered_trace = replace(
        reference,
        records=(
            reference.records[0],
            tick_one,
            reference.records[2],
        ),
    )
    manifest_path, _ = _write_case(
        tmp_path,
        trace=altered_trace,
        tick_map_data=b"tick,pts\n0,0\n1,9\n2,20\n",
    )

    report = verify_pc_golden_case(manifest_path)

    assert report.status is ComparisonStatus.FAIL
    assert report.checks[-1].name == "video_semantics"
    assert "frame PTS" in report.reasons[0]


def test_incomplete_required_coverage_is_incomparable(tmp_path: Path) -> None:
    coverage = (
        CoverageRange(
            channel="input_events",
            start_tick=0,
            end_tick=2,
            status=CoverageStatus.PARTIAL,
            required=True,
            reason="capture transition obscures one interval",
        ),
    )
    manifest_path, _ = _write_case(tmp_path, coverage=coverage)

    report = verify_pc_golden_case(manifest_path)

    assert report.status is ComparisonStatus.INCOMPARABLE
    assert report.exit_code == 2
    assert _check_by_name(
        report,
        "coverage",
    ).status is VerificationCheckStatus.INCOMPARABLE
    assert "required channel" in report.reasons[0]


def test_artifact_tampering_is_a_structured_failure(tmp_path: Path) -> None:
    manifest_path, manifest = _write_case(tmp_path)
    (tmp_path / manifest.artifacts["video"].path).write_bytes(b"tampered")

    report = verify_pc_golden_case(manifest_path)

    assert report.status is ComparisonStatus.FAIL
    assert report.exit_code == 1
    assert report.checks[-1].name == "artifacts"
    assert report.checks[-1].status is VerificationCheckStatus.FAIL
    assert report.checks[-1].details["error_type"] == "PcGoldenArtifactError"


def test_oversized_video_is_rejected_before_artifact_content_is_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, manifest = _write_case(tmp_path)
    artifacts = dict(manifest.artifacts)
    artifacts["video"] = replace(
        artifacts["video"],
        bytes=MAX_VIDEO_ARTIFACT_BYTES + 1,
    )
    oversized = replace(manifest, artifacts=artifacts)
    oversized.write_json(manifest_path)

    def forbidden_verify(self: PcGoldenManifest, root: Path) -> None:
        del self, root
        raise AssertionError("artifact content must not be read")

    monkeypatch.setattr(
        PcGoldenManifest,
        "verify_artifacts",
        forbidden_verify,
    )
    report = verify_pc_golden_case(manifest_path)

    assert report.status is ComparisonStatus.INCOMPARABLE
    assert [check.name for check in report.checks] == [
        "manifest",
        "artifact_budget",
    ]
    assert report.checks[-1].details["artifact_content_read"] is False


def test_content_addressed_but_fake_calibration_fails_semantic_recompute(
    tmp_path: Path,
) -> None:
    manifest_path, _ = _write_case(
        tmp_path,
        calibration_data=b'{"matrix":"not-control-point-evidence"}\n',
    )

    report = verify_pc_golden_case(manifest_path)

    assert report.status is ComparisonStatus.FAIL
    assert report.checks[-1].name == "calibration"
    assert (
        report.checks[-1].status
        is VerificationCheckStatus.FAIL
    )


@pytest.mark.parametrize(
    ("timeline_changes", "field"),
    [
        ({"product_version": "different"}, "product_version"),
        ({"random_seed": 7}, "random_seed"),
        ({"length_updates": 99}, "length_updates"),
    ],
)
def test_dmo_header_is_cross_checked_against_input_timeline(
    tmp_path: Path,
    timeline_changes: dict[str, Any],
    field: str,
) -> None:
    manifest_path, _ = _write_case(
        tmp_path,
        timeline_changes=timeline_changes,
    )

    report = verify_pc_golden_case(manifest_path)

    assert report.status is ComparisonStatus.FAIL
    assert report.checks[-1].name == "dmo"
    assert field in report.reasons[0]
    assert _SENSITIVE_DMO_VALUE.decode("ascii") not in report.to_json()


def test_dmo_product_version_text_is_not_echoed_to_reports(
    tmp_path: Path,
) -> None:
    private_version = "PRIVATE_PRODUCT_PATH_CAPTURE"
    manifest_path, _ = _write_case(
        tmp_path,
        dmo_data=_dmo_bytes(product_version=private_version),
        timeline_changes={"product_version": private_version},
    )

    report = verify_pc_golden_case(manifest_path)

    assert report.status is ComparisonStatus.INCOMPARABLE
    assert report.summary["dmo"]["product_version_matched"] is True
    assert (
        report.summary["dmo"]["product_version_bytes"]
        == len(private_version)
    )
    assert private_version not in report.to_json()


def test_v1_or_invalid_file_id_dmo_cannot_satisfy_v2_timeline(
    tmp_path: Path,
) -> None:
    version_one_path, _ = _write_case(
        tmp_path / "v1",
        dmo_data=_dmo_bytes(version=1),
    )
    version_one = verify_pc_golden_case(version_one_path)
    assert version_one.status is ComparisonStatus.FAIL
    assert version_one.checks[-1].name == "dmo"
    assert "format" in version_one.reasons[0]
    assert "dmo_version" in version_one.reasons[0]

    bad_id_path, _ = _write_case(
        tmp_path / "bad-id",
        dmo_data=_dmo_bytes(file_id=DEMO_FILE_ID + 1),
    )
    bad_id = verify_pc_golden_case(bad_id_path)
    assert bad_id.status is ComparisonStatus.FAIL
    assert bad_id.reasons == ("DMO parsing failed",)
    assert _SENSITIVE_DMO_VALUE.decode("ascii") not in bad_id.to_json()


def test_explicit_dmo_input_binding_mismatch_fails_safely(
    tmp_path: Path,
) -> None:
    manifest_path, _ = _write_case(
        tmp_path,
        binding_changes={
            0: {
                "dmo_sequence": 1,
                "private_note": "TRACE_PRIVATE_VALUE",
            }
        },
    )

    report = verify_pc_golden_case(manifest_path)

    assert report.status is ComparisonStatus.FAIL
    assert report.checks[-1].name == "input_sequence"
    assert "input index 0" in report.reasons[0]
    serialized = report.to_json()
    assert "TRACE_PRIVATE_VALUE" not in serialized
    assert _SENSITIVE_DMO_VALUE.decode("ascii") not in serialized


def test_dmo_update_must_map_to_the_declared_native_tick(
    tmp_path: Path,
) -> None:
    manifest_path, _ = _write_case(
        tmp_path,
        timeline_changes={"native_tick_offset": 1},
    )

    report = verify_pc_golden_case(manifest_path)

    assert report.status is ComparisonStatus.FAIL
    assert report.checks[-1].name == "input_sequence"
    assert "different native tick" in report.reasons[0]


def test_tick_map_rejects_input_moved_to_a_later_trace_tick(
    tmp_path: Path,
) -> None:
    environment = _environment()
    reference = _trace(environment)
    moved = PcGoldenTrace(
        case_id=reference.case_id,
        pc_environment_fingerprint=reference.pc_environment_fingerprint,
        capture_contract_fingerprint=(
            reference.capture_contract_fingerprint
        ),
        records=(
            reference.records[0],
            replace(reference.records[1], inputs=()),
            replace(
                reference.records[2],
                inputs=reference.records[1].inputs,
            ),
        ),
    )
    manifest_path, _ = _write_case(tmp_path, trace=moved)

    report = verify_pc_golden_case(manifest_path)

    assert report.status is ComparisonStatus.FAIL
    assert report.checks[-1].name == "tick_map"
    assert "tick-map" in report.reasons[0]


def test_tick_map_parsed_content_is_rebound_to_artifact_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, _ = _write_case(tmp_path)
    tampered = PcTickMap(
        records=(
            TickPts(0, 0),
            TickPts(1, 9),
            TickPts(2, 20),
        )
    )
    monkeypatch.setattr(
        verifier_module.PcTickMap,
        "read",
        lambda path: tampered,
    )

    report = verify_pc_golden_case(manifest_path)

    assert report.status is ComparisonStatus.FAIL
    assert report.checks[-1].name == "tick_map"


def test_unbound_trace_inputs_are_incomparable_without_guessing_phase(
    tmp_path: Path,
) -> None:
    manifest_path, _ = _write_case(tmp_path, bound_inputs=False)

    report = verify_pc_golden_case(manifest_path)

    assert report.status is ComparisonStatus.INCOMPARABLE
    assert report.exit_code == 2
    input_check = _check_by_name(report, "input_sequence")
    assert input_check.status is VerificationCheckStatus.INCOMPARABLE
    assert input_check.details["mode"] == "unbound"


def test_dmo_inputs_outside_clipped_trace_range_are_ignored(
    tmp_path: Path,
) -> None:
    reference = _trace(_environment())
    clipped = replace(
        reference,
        records=(
            reference.records[0],
            replace(reference.records[1], inputs=()),
            reference.records[2],
        ),
    )
    manifest_path, _ = _write_case(
        tmp_path,
        dmo_data=_dmo_bytes(
            length_updates=10,
            input_update=5,
        ),
        timeline_changes={"length_updates": 10},
        trace=clipped,
    )

    report = verify_pc_golden_case(manifest_path)

    input_check = _check_by_name(report, "input_sequence")
    assert input_check.status is VerificationCheckStatus.PASS
    assert input_check.details["trace_input_count"] == 0
    assert input_check.details["dmo_input_count"] == 0
    assert input_check.details["dmo_total_input_count"] == 2


def test_unknown_required_channel_and_unsupported_profile_are_incomparable(
    tmp_path: Path,
) -> None:
    coverage = (
        CoverageRange(
            channel="frames",
            start_tick=0,
            end_tick=2,
            status=CoverageStatus.COMPLETE,
            required=True,
        ),
        CoverageRange(
            channel="input_events",
            start_tick=0,
            end_tick=2,
            status=CoverageStatus.COMPLETE,
            required=True,
        ),
        CoverageRange(
            channel="score",
            start_tick=0,
            end_tick=2,
            status=CoverageStatus.COMPLETE,
            required=True,
        ),
        CoverageRange(
            channel="made_up_channel",
            start_tick=0,
            end_tick=2,
            status=CoverageStatus.COMPLETE,
            required=True,
        ),
    )
    unknown_path, _ = _write_case(
        tmp_path / "unknown",
        coverage=coverage,
    )
    unsupported_path, _ = _write_case(
        tmp_path / "profile",
        profile_mode="adventure",
    )

    unknown = verify_pc_golden_case(unknown_path)
    unsupported = verify_pc_golden_case(unsupported_path)

    assert unknown.status is ComparisonStatus.INCOMPARABLE
    assert any("unknown to PC golden" in reason for reason in unknown.reasons)
    assert unsupported.status is ComparisonStatus.INCOMPARABLE
    assert any("profile mode" in reason for reason in unsupported.reasons)


def test_complete_measurement_rejects_simulator_provenance(
    tmp_path: Path,
) -> None:
    environment = _environment()
    reference = _trace(environment)
    records = tuple(
        replace(
            record,
            measurements={
                "score": replace(
                    record.measurements["score"],
                    source="simulator",
                )
            },
        )
        for record in reference.records
    )
    untrusted = replace(reference, records=records)
    manifest_path, _ = _write_case(tmp_path, trace=untrusted)

    report = verify_pc_golden_case(manifest_path)

    assert report.status is ComparisonStatus.INCOMPARABLE
    assert _check_by_name(
        report,
        "trace_coverage",
    ).status is VerificationCheckStatus.INCOMPARABLE
    assert any("untrusted provenance" in reason for reason in report.reasons)


def test_one_sided_input_evidence_never_passes(tmp_path: Path) -> None:
    environment = _environment()
    reference = _trace(environment)
    trace_without_inputs = replace(
        reference,
        records=(
            reference.records[0],
            replace(reference.records[1], inputs=()),
            reference.records[2],
        ),
    )
    missing_trace_path, _ = _write_case(
        tmp_path / "trace-empty",
        trace=trace_without_inputs,
    )
    missing_trace = verify_pc_golden_case(missing_trace_path)
    assert missing_trace.status is ComparisonStatus.INCOMPARABLE
    assert _check_by_name(
        missing_trace,
        "input_sequence",
    ).status is VerificationCheckStatus.INCOMPARABLE

    missing_dmo_path, _ = _write_case(
        tmp_path / "dmo-empty",
        dmo_data=_dmo_bytes(include_inputs=False),
    )
    missing_dmo = verify_pc_golden_case(missing_dmo_path)
    assert missing_dmo.status is ComparisonStatus.FAIL
    assert missing_dmo.checks[-1].name == "input_sequence"

    both_empty_path, _ = _write_case(
        tmp_path / "both-empty",
        dmo_data=_dmo_bytes(include_inputs=False),
        trace=trace_without_inputs,
    )
    both_empty = verify_pc_golden_case(both_empty_path)
    assert both_empty.status is ComparisonStatus.INCOMPARABLE
    assert _check_by_name(
        both_empty,
        "input_sequence",
    ).status is VerificationCheckStatus.PASS


def test_trace_manifest_mismatch_is_rejected_after_artifact_verification(
    tmp_path: Path,
) -> None:
    environment = _environment()
    trace = _trace(environment, case_id="different-case")
    manifest_path, _ = _write_case(tmp_path, trace=trace)

    report = verify_pc_golden_case(manifest_path)

    assert report.status is ComparisonStatus.FAIL
    assert [check.name for check in report.checks] == [
        "manifest",
        "artifact_budget",
        "artifacts",
        "trace",
    ]


def test_explicit_original_scenario_resolution_is_fail_closed_and_path_safe(
    tmp_path: Path,
) -> None:
    manifest_path, _ = _write_case(tmp_path / "case")
    missing_root = tmp_path / "PRIVATE-NONEXISTENT-ORIGINAL"

    report = verify_pc_golden_case(
        manifest_path,
        original_root=missing_root,
    )

    assert report.status is ComparisonStatus.FAIL
    assert report.checks[-1].name == "scenario_semantics"
    serialized = report.to_json()
    assert str(missing_root) not in serialized


def test_successful_original_resolution_only_clears_scenario_blocker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, _ = _write_case(tmp_path / "case")
    monkeypatch.setattr(
        verifier_module,
        "_scenario_semantics_mismatches",
        lambda manifest, original_root: (),
    )

    report = verify_pc_golden_case(
        manifest_path,
        original_root=tmp_path / "synthetic-original",
    )

    assert report.status is ComparisonStatus.INCOMPARABLE
    assert (
        _check_by_name(report, "scenario_semantics").status
        is VerificationCheckStatus.PASS
    )
    assert report.summary["semantic_limits"][
        "scenario_resolved_against_original_data"
    ] is True


def test_bad_manifest_returns_report_instead_of_raising(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text('{"schema":NaN}', encoding="utf-8")

    report = verify_pc_golden_case(manifest_path)

    assert report.status is ComparisonStatus.FAIL
    assert report.case_id is None
    assert report.checks[0].name == "manifest"
    assert report.checks[0].details["error_type"] == (
        "PcGoldenValidationError"
    )


def test_pathological_json_nesting_is_a_structured_cli_failure(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest_path = tmp_path / "PRIVATE-DEEP-MANIFEST.json"
    nested = "[" * 2000 + "0" + "]" * 2000
    manifest_path.write_text(nested, encoding="utf-8")

    assert main([str(manifest_path), "--compact"]) == 1
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert report["status"] == "FAIL"
    assert report["checks"][0]["name"] == "manifest"
    assert "Traceback" not in captured.err
    assert str(manifest_path) not in captured.out


def test_cli_emits_json_and_uses_fail_closed_exit_codes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    structural_path, _ = _write_case(tmp_path / "structural")
    incomplete = (
        CoverageRange(
            channel="input_events",
            start_tick=0,
            end_tick=2,
            status=CoverageStatus.ABSENT,
            required=True,
            reason="not captured",
        ),
    )
    incomparable_path, _ = _write_case(
        tmp_path / "incomparable",
        coverage=incomplete,
    )
    fail_path, fail_manifest = _write_case(tmp_path / "fail")
    (fail_path.parent / fail_manifest.artifacts["video"].path).write_bytes(
        b"tampered"
    )

    for path, expected_status, expected_code in (
        (structural_path, "INCOMPARABLE", 2),
        (incomparable_path, "INCOMPARABLE", 2),
        (fail_path, "FAIL", 1),
    ):
        assert main([str(path), "--compact"]) == expected_code
        output = json.loads(capsys.readouterr().out)
        assert output["status"] == expected_status
        assert output["schema"] == REPORT_SCHEMA
