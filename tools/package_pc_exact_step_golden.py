"""Package two formal exact-step retail runs as a PC Golden v5 case.

The source BMPs remain the authoritative full-BGRA evidence.  FFV1 videos are
lossless, deterministic presentation wrappers whose decoded RGB pixels are
independently rebound to every retained BMP by ``verify_pc_golden_case``.
Nothing is published until that read-only verifier returns PASS.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import Any, Iterable, Mapping

import numpy as np

from zuma_rl.original_data import OriginalGameCatalog
from zuma_rl.pc_calibration import (
    CalibrationControlPoint,
    PcCalibrationSidecar,
)
from zuma_rl.pc_evidence import PcTickMap, TickPts
from zuma_rl.pc_exact_step_evidence import (
    ExactStepLoadedRun,
    canonical_json_bytes,
    compare_formal_exact_step_runs,
    decode_top_down_bgra_bmp,
    load_formal_exact_step_run,
    read_canonical_json,
)
from zuma_rl.pc_golden import (
    DMO_FILE_ID,
    DMO_FORMAT,
    DMO_VERSION,
    EXACT_STEP_MANIFEST_VERSION,
    ArtifactSpec,
    ComparisonContract,
    ComparisonStatus,
    CoordinateCalibration,
    CoverageRange,
    CoverageStatus,
    ExactStepReplayContract,
    ExactStepRunContract,
    FrameRef,
    GoldenInput,
    GoldenTick,
    InputTimeline,
    Measurement,
    MeasurementStatus,
    PcEnvironment,
    PcGoldenManifest,
    PcGoldenTrace,
    Scenario,
    TickClock,
    VideoMetadata,
    exact_step_capture_contract_fingerprint,
)
from zuma_rl.pc_protocol_evidence import PcStateSnapshot
from zuma_rl.popcap_dmo import DemoCommand, PopCapDemo
from zuma_rl.retail_dmo_provenance import (
    COLLECTOR_PLAN_ARTIFACT,
    PLAYBACK_DMO_ARTIFACT,
    PROVENANCE_ARTIFACT,
    RAW_RECORDING_DMO_ARTIFACT,
    RECORDING_REPORT_ARTIFACT,
    RetailDmoProvenanceError,
    build_certifying_provenance,
    canonical_provenance_bytes,
)
from zuma_rl.verify_pc_golden import verify_pc_golden_case


PACKAGE_SCHEMA = "zuma-rl.pc-golden-exact-step-package"
PACKAGE_VERSION = 1


class PackagingError(RuntimeError):
    """The source campaign cannot support a formal v5 case."""


def _validated_run_ids(values: tuple[str, str]) -> tuple[str, str]:
    if (
        len(values) != 2
        or len(set(values)) != 2
        or any(
            re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", value) is None
            for value in values
        )
    ):
        raise PackagingError("formal run ids must be two distinct run ids")
    return values


@dataclass(frozen=True, slots=True)
class RunSource:
    run_id: str
    root: Path
    post_snapshot: Path
    selected_attempt: int
    probe_path: Path
    index_path: Path
    process_id: int
    process_creation_filetime_100ns: int
    loaded: ExactStepLoadedRun


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _sha256_bytes(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _artifact(path: Path, relative_path: str) -> ArtifactSpec:
    return ArtifactSpec(
        path=relative_path,
        sha256=_sha256_path(path),
        bytes=path.stat().st_size,
    )


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)


def _copy_new(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise PackagingError(f"required source is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as input_stream, destination.open("xb") as output:
        shutil.copyfileobj(input_stream, output, 4 * 1024 * 1024)
    if (
        destination.stat().st_size != source.stat().st_size
        or _sha256_path(destination) != _sha256_path(source)
    ):
        raise PackagingError(f"copied artifact changed: {source.name}")


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


def _strict_int(row: Mapping[str, Any], field: str) -> int:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PackagingError(f"source field {field!r} is invalid")
    return value


def _load_run_source(
    *,
    run_id: str,
    root: Path,
    post_snapshot: Path,
    runtime_sha256: str,
    dmo_sha256: str,
    freeze_update: int,
    start_update: int,
    end_update: int,
    maximum_attempts: int,
) -> RunSource:
    attempts_path = root / "attempts.json"
    attempts = read_canonical_json(attempts_path)
    if not isinstance(attempts, list) or not attempts:
        raise PackagingError(f"{run_id} attempts are missing")
    selected_attempt = len(attempts)
    selected = attempts[-1]
    if not isinstance(selected, dict):
        raise PackagingError(f"{run_id} selected attempt is invalid")
    process_id = _strict_int(selected, "process_id")
    creation_time = _strict_int(
        selected, "process_creation_filetime_100ns"
    )
    probe_path = (
        root / f"attempt-{selected_attempt:02d}" / "memory-probe.json"
    )
    index_path = (
        root
        / f"attempt-{selected_attempt:02d}"
        / "trajectory"
        / "index.json"
    )
    loaded = load_formal_exact_step_run(
        run_id=run_id,
        selected_attempt=selected_attempt,
        attempts_path=attempts_path,
        probe_path=probe_path,
        index_path=index_path,
        expected_process_id=process_id,
        expected_process_creation_filetime_100ns=creation_time,
        expected_runtime_sha256=runtime_sha256,
        expected_dmo_sha256=dmo_sha256,
        expected_freeze_update=freeze_update,
        expected_start_update=start_update,
        expected_end_update=end_update,
        expected_warmup_tick_count=start_update - freeze_update,
        maximum_startup_attempts=maximum_attempts,
    )
    snapshot = PcStateSnapshot.read(post_snapshot)
    if snapshot.captured_perf_counter_ns <= loaded.attempt_finished_perf_counter_ns:
        raise PackagingError(f"{run_id} post snapshot predates capture finish")
    return RunSource(
        run_id=run_id,
        root=root,
        post_snapshot=post_snapshot,
        selected_attempt=selected_attempt,
        probe_path=probe_path,
        index_path=index_path,
        process_id=process_id,
        process_creation_filetime_100ns=creation_time,
        loaded=loaded,
    )


def _encode_ffv1(path: Path, source: ExactStepLoadedRun) -> None:
    try:
        import av
    except ImportError as error:
        raise PackagingError("PyAV is required for FFV1 packaging") from error

    path.parent.mkdir(parents=True, exist_ok=True)
    time_base = Fraction(1, 100)
    try:
        with av.open(str(path), mode="w", format="avi") as container:
            stream = container.add_stream("ffv1", rate=Fraction(100, 1))
            stream.width = 800
            stream.height = 600
            stream.pix_fmt = "bgra"
            stream.time_base = time_base
            for tick, source_frame in enumerate(source.visual_frames):
                pixels = np.ascontiguousarray(
                    decode_top_down_bgra_bmp(source_frame.path)
                )
                frame = av.VideoFrame.from_ndarray(pixels, format="bgra")
                frame.pts = tick
                frame.time_base = time_base
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
    except Exception as error:
        if path.exists():
            path.unlink()
        raise PackagingError("lossless FFV1 encoding failed") from error


def _trace(
    *,
    case_id: str,
    environment_fingerprint: str,
    contract_fingerprint: str,
    source: ExactStepLoadedRun,
    commands: tuple[DemoCommand, ...],
) -> PcGoldenTrace:
    inputs_by_tick: dict[int, list[DemoCommand]] = {}
    for command in commands:
        inputs_by_tick.setdefault(
            command.update - source.start_update,
            [],
        ).append(command)
    if inputs_by_tick.get(0):
        raise PackagingError(
            "exact-step source start update contains an input edge"
        )
    records: list[GoldenTick] = []
    for tick, row in enumerate(source.tick_rows):
        inputs = sorted(
            inputs_by_tick.get(tick, ()),
            key=lambda command: command.sequence,
        )
        records.append(
            GoldenTick(
                tick=tick,
                frames=(FrameRef(frame_index=tick, pts=tick),),
                inputs=tuple(
                    GoldenInput(
                        sequence=sequence,
                        kind=command.kind,
                        pts=tick,
                        payload={
                            "dmo_sequence": command.sequence,
                            "dmo_update": command.update,
                        },
                    )
                    for sequence, command in enumerate(inputs)
                ),
                events=(),
                measurements={
                    "score": Measurement(
                        status=MeasurementStatus.OBSERVED,
                        value=_strict_int(row, "score"),
                        source="pc_memory_probe",
                        uncertainty=0.0,
                    ),
                    "chain_count": Measurement(
                        status=MeasurementStatus.OBSERVED,
                        value=_strict_int(row, "chain_ball_count"),
                        source="pc_memory_probe",
                        uncertainty=0.0,
                    ),
                    "projectile_count": Measurement(
                        status=MeasurementStatus.OBSERVED,
                        value=_strict_int(row, "fired_bullet_count"),
                        source="pc_memory_probe",
                        uncertainty=0.0,
                    ),
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
    run_ids: tuple[str, str],
    run_roots: tuple[Path, Path],
    post_snapshots: tuple[Path, Path],
    replay_pre_snapshot_path: Path,
    host_pre_snapshot_path: Path,
    preregistration_path: Path,
    execution_binding_path: Path,
    input_dmo_path: Path,
    retail_recording_dmo_path: Path,
    retail_recording_report_path: Path,
    collector_plan_path: Path,
    original_root: Path,
    output_root: Path,
    level_id: str,
    case_id: str,
    freeze_update: int,
    start_update: int,
    end_update: int,
    maximum_attempts: int,
    os_build: str,
    gpu: str,
    driver: str,
) -> Mapping[str, Any]:
    run_ids = _validated_run_ids(run_ids)
    paths = (
        *run_roots,
        *post_snapshots,
        replay_pre_snapshot_path,
        host_pre_snapshot_path,
        preregistration_path,
        execution_binding_path,
        input_dmo_path,
        retail_recording_dmo_path,
        retail_recording_report_path,
        collector_plan_path,
        original_root,
        output_root,
    )
    (
        run_roots,
        post_snapshots,
        replay_pre_snapshot_path,
        host_pre_snapshot_path,
        preregistration_path,
        execution_binding_path,
        input_dmo_path,
        retail_recording_dmo_path,
        retail_recording_report_path,
        collector_plan_path,
        original_root,
        output_root,
    ) = (
        tuple(path.resolve() for path in run_roots),
        tuple(path.resolve() for path in post_snapshots),
        replay_pre_snapshot_path.resolve(),
        host_pre_snapshot_path.resolve(),
        preregistration_path.resolve(),
        execution_binding_path.resolve(),
        input_dmo_path.resolve(),
        retail_recording_dmo_path.resolve(),
        retail_recording_report_path.resolve(),
        collector_plan_path.resolve(),
        original_root.resolve(),
        output_root.resolve(),
    )
    del paths
    part_root = output_root.with_name(output_root.name + ".building")
    verification_path = output_root.with_name(
        output_root.name + ".verification.json"
    )
    verification_part = verification_path.with_name(
        verification_path.name + ".building"
    )
    if any(
        path.exists()
        for path in (
            output_root,
            part_root,
            verification_path,
            verification_part,
        )
    ):
        raise PackagingError("output, building, or verification path exists")
    if not freeze_update < start_update <= end_update:
        raise PackagingError("exact-step source update range is invalid")
    if not 1 <= maximum_attempts <= 3:
        raise PackagingError("maximum attempts must be between one and three")

    demo = PopCapDemo.read(input_dmo_path)
    dmo_sha256 = demo.artifact_sha256
    preregistration = read_canonical_json(preregistration_path)
    execution_binding = read_canonical_json(execution_binding_path)
    if not isinstance(preregistration, dict) or not isinstance(
        execution_binding, dict
    ):
        raise PackagingError("formal control evidence is invalid")

    first_attempts = read_canonical_json(run_roots[0] / "attempts.json")
    if not isinstance(first_attempts, list) or not first_attempts:
        raise PackagingError("first formal attempts receipt is invalid")
    first_probe = read_canonical_json(
        run_roots[0]
        / f"attempt-{len(first_attempts):02d}"
        / "memory-probe.json"
    )
    if not isinstance(first_probe, dict):
        raise PackagingError("first formal memory probe is invalid")
    runtime_sha256 = first_probe.get("runtime_executable_sha256")
    if not isinstance(runtime_sha256, str):
        raise PackagingError("formal runtime identity is missing")
    runs = tuple(
        _load_run_source(
            run_id=run_id,
            root=root,
            post_snapshot=post,
            runtime_sha256=runtime_sha256,
            dmo_sha256=dmo_sha256,
            freeze_update=freeze_update,
            start_update=start_update,
            end_update=end_update,
            maximum_attempts=maximum_attempts,
        )
        for run_id, root, post in zip(
            run_ids, run_roots, post_snapshots, strict=True
        )
    )
    comparison_payload = compare_formal_exact_step_runs(
        run.loaded for run in runs
    )
    if comparison_payload.get("status") != "PASS":
        raise PackagingError("formal independent source comparison failed")

    replay_pre_snapshot = PcStateSnapshot.read(replay_pre_snapshot_path)
    host_pre_snapshot = PcStateSnapshot.read(host_pre_snapshot_path)
    run_post_snapshots = tuple(
        PcStateSnapshot.read(path) for path in post_snapshots
    )
    if any(
        snapshot.state_root != host_pre_snapshot.state_root
        for snapshot in run_post_snapshots
    ):
        raise PackagingError("formal campaign did not restore host state")
    if host_pre_snapshot.captured_perf_counter_ns >= min(
        run.loaded.campaign_started_perf_counter_ns for run in runs
    ):
        raise PackagingError("pre snapshot was not captured before both runs")

    try:
        provenance_payload = build_certifying_provenance(
            source_data=retail_recording_dmo_path.read_bytes(),
            output_data=input_dmo_path.read_bytes(),
            recording_report_data=retail_recording_report_path.read_bytes(),
            collector_plan_data=collector_plan_path.read_bytes(),
            expected_runtime_sha256=runtime_sha256,
        )
        provenance_bytes = canonical_provenance_bytes(provenance_payload)
    except (OSError, RetailDmoProvenanceError) as error:
        raise PackagingError("retail DMO provenance is not certifying") from error

    catalog = OriginalGameCatalog(original_root)
    loaded_level = catalog.load_level(level_id, hard=False)
    if len(loaded_level.curves) != 1:
        raise PackagingError("exact-step packager requires one original curve")
    runtime_source = original_root / "ZumasRevenge.exe"
    main_pak = original_root / "main.pak"
    launcher_sha256 = _sha256_path(runtime_source)
    main_pak_sha256 = _sha256_path(main_pak)
    levels_xml_sha256 = _sha256_bytes(
        catalog.archive.read_member(r"levels\levels.xml")
    )
    curve_hashes = {
        str(curve.source_path.relative_to(original_root)).replace("\\", "/"):
        _sha256_path(curve.source_path)
        for curve in loaded_level.curves
        if curve.source_path is not None
    }
    if len(curve_hashes) != len(loaded_level.curves):
        raise PackagingError("original curve identities are incomplete")

    part_root.parent.mkdir(parents=True, exist_ok=True)
    part_root.mkdir()
    artifacts: dict[str, ArtifactSpec] = {}

    def copy_artifact(name: str, source: Path, relative: str) -> None:
        destination = part_root / relative
        _copy_new(source, destination)
        artifacts[name] = _artifact(destination, relative)

    copy_artifact(
        PLAYBACK_DMO_ARTIFACT, input_dmo_path, "input.dmo"
    )
    copy_artifact(
        RAW_RECORDING_DMO_ARTIFACT,
        retail_recording_dmo_path,
        "evidence/input-recording.dmo",
    )
    copy_artifact(
        RECORDING_REPORT_ARTIFACT,
        retail_recording_report_path,
        "evidence/input-recording-report.json",
    )
    copy_artifact(
        COLLECTOR_PLAN_ARTIFACT,
        collector_plan_path,
        "evidence/collector-plan.json",
    )
    provenance_path = part_root / "evidence/dmo-provenance.json"
    _write_new(provenance_path, provenance_bytes)
    artifacts[PROVENANCE_ARTIFACT] = _artifact(
        provenance_path, "evidence/dmo-provenance.json"
    )
    copy_artifact(
        "exact.preregistration",
        preregistration_path,
        "evidence/exact-step-preregistration.json",
    )
    copy_artifact(
        "exact.execution_binding",
        execution_binding_path,
        "evidence/exact-step-execution-binding.json",
    )
    copy_artifact(
        "save.replay_pre",
        replay_pre_snapshot_path,
        "evidence/state/replay-pre.json",
    )
    copy_artifact(
        "save.host_pre",
        host_pre_snapshot_path,
        "evidence/state/host-pre.json",
    )
    comparison_path = part_root / "evidence/exact-step-comparison.json"
    _write_new(comparison_path, canonical_json_bytes(comparison_payload))
    artifacts["exact.comparison"] = _artifact(
        comparison_path, "evidence/exact-step-comparison.json"
    )

    source_artifact_names: dict[tuple[str, str], str] = {}
    for run in runs:
        for source in sorted(
            (path for path in run.root.rglob("*") if path.is_file()),
            key=lambda path: path.relative_to(run.root).as_posix(),
        ):
            relative_source = source.relative_to(run.root).as_posix()
            name = f"source.{run.run_id}/{relative_source}"
            relative = f"evidence/runs/{run.run_id}/{relative_source}"
            copy_artifact(name, source, relative)
            source_artifact_names[(run.run_id, relative_source)] = name
        copy_artifact(
            f"save.{run.run_id}.post",
            run.post_snapshot,
            f"evidence/state/{run.run_id}-post.json",
        )
        video_path = part_root / f"video-{run.run_id}.avi"
        _encode_ffv1(video_path, run.loaded)
        artifacts[f"video.{run.run_id}"] = _artifact(
            video_path, f"video-{run.run_id}.avi"
        )

    calibration = _identity_calibration()
    fit = calibration.recompute()
    calibration_path = part_root / "calibration.json"
    _write_new(calibration_path, calibration.to_json().encode("utf-8"))
    artifacts["calibration"] = _artifact(
        calibration_path, "calibration.json"
    )
    tick_count = end_update - start_update + 1
    tick_maps: dict[str, PcTickMap] = {}
    for run in runs:
        tick_map = PcTickMap(
            records=tuple(TickPts(tick=tick, pts=tick) for tick in range(tick_count))
        )
        tick_map_path = part_root / f"tick-map-{run.run_id}.csv"
        _write_new(tick_map_path, tick_map.to_csv().encode("utf-8"))
        artifacts[f"tick_map.{run.run_id}"] = _artifact(
            tick_map_path, f"tick-map-{run.run_id}.csv"
        )
        tick_maps[run.run_id] = tick_map

    scenario = Scenario(
        level_id=loaded_level.definition.id,
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
        pre_capture_save_sha256=replay_pre_snapshot.state_root,
        profile_mode="tutorials_completed",
        renderer_api="Direct3D9",
        renderer_mode="windowed_800x600_exact_step",
        ball_radius_branch=18,
        os_build=os_build,
        gpu=gpu,
        driver=driver,
        game_settings={
            "full_screen": False,
            "high_resolution": False,
            "is_3d": True,
            "wait_for_vsync": False,
            "dwm_window_corner_preference": "DWMWCP_DONOTROUND",
            "runtime_source_executable_sha256": launcher_sha256,
            "runtime_payload_sha256": runtime_sha256,
        },
        dynamic_difficulty_state={},
    )
    timeline = InputTimeline(
        artifact=PLAYBACK_DMO_ARTIFACT,
        format=DMO_FORMAT,
        file_id=DMO_FILE_ID,
        dmo_version=DMO_VERSION,
        product_version=demo.product_version,
        random_seed=demo.random_seed,
        length_updates=demo.length_updates,
        native_tick_offset=-start_update,
    )
    videos = {
        run.run_id: VideoMetadata(
            artifact=f"video.{run.run_id}",
            width=800,
            height=600,
            codec="ffv1",
            pixel_format="bgra",
            time_base=(1, 100),
            nominal_fps=(100, 1),
            frame_count=tick_count,
            first_pts=0,
            last_pts=tick_count - 1,
            cfr=True,
            dropped_frames=0,
            duplicate_frames=0,
        )
        for run in runs
    }
    clocks = {
        run.run_id: TickClock(
            logic_hz=(100, 1),
            tick_start=0,
            tick_end=tick_count - 1,
            tick0_video_pts=0,
            sample_phase="post_update_presented",
            mapping_kind="per_tick_pts_table",
            tick_map_artifact=f"tick_map.{run.run_id}",
            uncertainty_ticks=0.0,
        )
        for run in runs
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
            end_tick=tick_count - 1,
            status=CoverageStatus.COMPLETE,
            required=True,
        )
        for channel in (
            "frames",
            "input_events",
            "score",
            "chain_count",
            "projectile_count",
        )
    )
    comparison_contract = ComparisonContract(
        event_tick_tolerance=0,
        center_l2_tolerance_px=0.5,
        waypoint_abs_tolerance=0.01,
        max_drift_per_100_ticks=0.05,
    )
    for run in runs:
        artifacts[f"trace.{run.run_id}"] = ArtifactSpec(
            path=f"trace-{run.run_id}.ndjson",
            sha256="sha256:" + "0" * 64,
            bytes=1,
        )
    run_contracts = tuple(
        ExactStepRunContract(
            run_id=run.run_id,
            selected_attempt=run.selected_attempt,
            attempts_artifact=source_artifact_names[
                (run.run_id, "attempts.json")
            ],
            memory_probe_artifact=source_artifact_names[
                (
                    run.run_id,
                    f"attempt-{run.selected_attempt:02d}/memory-probe.json",
                )
            ],
            trajectory_index_artifact=source_artifact_names[
                (
                    run.run_id,
                    f"attempt-{run.selected_attempt:02d}/trajectory/index.json",
                )
            ],
            post_snapshot_artifact=f"save.{run.run_id}.post",
            trace_artifact=f"trace.{run.run_id}",
            video=videos[run.run_id],
            clock=clocks[run.run_id],
            process_id=run.process_id,
            process_creation_filetime_100ns=(
                run.process_creation_filetime_100ns
            ),
        )
        for run in runs
    )
    exact_contract = ExactStepReplayContract(
        preregistration_artifact="exact.preregistration",
        execution_binding_artifact="exact.execution_binding",
        comparison_artifact="exact.comparison",
        pre_snapshot_artifact="save.replay_pre",
        host_pre_snapshot_artifact="save.host_pre",
        runs=run_contracts,
        freeze_update=freeze_update,
        source_start_update=start_update,
        source_end_update=end_update,
        warmup_tick_count=start_update - freeze_update,
        maximum_startup_attempts=maximum_attempts,
        sample_phase="frozen_post_replay_update_barrier",
        pixel_comparison="full_800x600_bgra_exact",
    )
    primary_run = runs[0]
    contract_fingerprint = exact_step_capture_contract_fingerprint(
        case_id=case_id,
        scenario=scenario,
        pc_environment_fingerprint=environment.fingerprint,
        artifacts=artifacts,
        input_timeline=timeline,
        trace_artifact=f"trace.{primary_run.run_id}",
        video=videos[primary_run.run_id],
        clock=clocks[primary_run.run_id],
        coordinates=coordinates,
        coverage=coverage,
        comparison_contract=comparison_contract,
        exact_step_replay=exact_contract,
    )
    commands = tuple(
        command
        for command in demo.input_commands
        if start_update <= command.update <= end_update
    )
    for run in runs:
        trace = _trace(
            case_id=case_id,
            environment_fingerprint=environment.fingerprint,
            contract_fingerprint=contract_fingerprint,
            source=run.loaded,
            commands=commands,
        )
        trace_path = part_root / f"trace-{run.run_id}.ndjson"
        _write_new(trace_path, trace.to_ndjson().encode("utf-8"))
        artifacts[f"trace.{run.run_id}"] = _artifact(
            trace_path, f"trace-{run.run_id}.ndjson"
        )

    manifest = PcGoldenManifest(
        case_id=case_id,
        scenario=scenario,
        pc_environment=environment,
        pc_environment_fingerprint=environment.fingerprint,
        artifacts=artifacts,
        input_timeline=timeline,
        trace_artifact=f"trace.{primary_run.run_id}",
        video=videos[primary_run.run_id],
        clock=clocks[primary_run.run_id],
        coordinates=coordinates,
        coverage=coverage,
        comparison_contract=comparison_contract,
        producer={
            "tool": "tools/package_pc_exact_step_golden.py",
            "version": PACKAGE_VERSION,
            "transport": "formal_exact_step_external_lossless",
            "source_bmp_authoritative": True,
            "ffv1_role": "lossless_presentation_wrapper",
            "certifying_retail_dmo_provenance": True,
        },
        manifest_version=EXACT_STEP_MANIFEST_VERSION,
        exact_step_replay=exact_contract,
    )
    if manifest.capture_contract_fingerprint != contract_fingerprint:
        raise PackagingError("final exact-step contract fingerprint changed")
    manifest_path = part_root / "manifest.json"
    _write_new(
        manifest_path,
        (manifest.to_json() + "\n").encode("utf-8"),
    )

    verification = verify_pc_golden_case(
        manifest_path,
        case_root=part_root,
        original_root=original_root,
        require_certifying_dmo_provenance=True,
    )
    if verification.status is not ComparisonStatus.PASS:
        raise PackagingError(
            "independent verifier rejected the building case: "
            + verification.status.value
        )
    verification_payload = (verification.to_json(indent=2) + "\n").encode(
        "utf-8"
    )
    _write_new(verification_part, verification_payload)
    manifest_sha256 = _sha256_path(manifest_path)
    part_root.rename(output_root)
    verification_part.rename(verification_path)
    return {
        "schema": PACKAGE_SCHEMA,
        "version": PACKAGE_VERSION,
        "status": "complete",
        "case_id": case_id,
        "case_root": str(output_root),
        "manifest": str(output_root / "manifest.json"),
        "manifest_sha256": manifest_sha256,
        "verification": str(verification_path),
        "verification_sha256": _sha256_path(verification_path),
        "verification_status": verification.status.value,
        "source_update_range": [start_update, end_update],
        "native_tick_count": tick_count,
        "run_ids": list(run_ids),
        "run_count": len(runs),
        "artifact_count": len(artifacts),
        "full_800x600_bgra_exact": True,
        "host_state_restored": True,
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
        description="Package two formal exact-step runs as PC Golden v5."
    )
    parser.add_argument("--run-a-id", required=True)
    parser.add_argument("--run-b-id", required=True)
    parser.add_argument("--run-a-root", required=True, type=Path)
    parser.add_argument("--run-b-root", required=True, type=Path)
    parser.add_argument("--run-a-post-snapshot", required=True, type=Path)
    parser.add_argument("--run-b-post-snapshot", required=True, type=Path)
    parser.add_argument("--replay-pre-snapshot", required=True, type=Path)
    parser.add_argument("--host-pre-snapshot", required=True, type=Path)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--execution-binding", required=True, type=Path)
    parser.add_argument("--input-dmo", required=True, type=Path)
    parser.add_argument("--retail-recording-dmo", required=True, type=Path)
    parser.add_argument("--retail-recording-report", required=True, type=Path)
    parser.add_argument("--collector-plan", required=True, type=Path)
    parser.add_argument("--original-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--level-id", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--freeze-update", required=True, type=_nonnegative_int)
    parser.add_argument("--start-update", required=True, type=_nonnegative_int)
    parser.add_argument("--end-update", required=True, type=_nonnegative_int)
    parser.add_argument("--maximum-attempts", type=_positive_int, default=3)
    parser.add_argument("--os-build", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--driver", required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        result = package_case(
            run_ids=(arguments.run_a_id, arguments.run_b_id),
            run_roots=(arguments.run_a_root, arguments.run_b_root),
            post_snapshots=(
                arguments.run_a_post_snapshot,
                arguments.run_b_post_snapshot,
            ),
            replay_pre_snapshot_path=arguments.replay_pre_snapshot,
            host_pre_snapshot_path=arguments.host_pre_snapshot,
            preregistration_path=arguments.preregistration,
            execution_binding_path=arguments.execution_binding,
            input_dmo_path=arguments.input_dmo,
            retail_recording_dmo_path=arguments.retail_recording_dmo,
            retail_recording_report_path=arguments.retail_recording_report,
            collector_plan_path=arguments.collector_plan,
            original_root=arguments.original_root,
            output_root=arguments.output_root,
            level_id=arguments.level_id,
            case_id=arguments.case_id,
            freeze_update=arguments.freeze_update,
            start_update=arguments.start_update,
            end_update=arguments.end_update,
            maximum_attempts=arguments.maximum_attempts,
            os_build=arguments.os_build,
            gpu=arguments.gpu,
            driver=arguments.driver,
        )
    except Exception as error:
        print(f"packaging failed: {type(error).__name__}: {error}")
        return 1
    print(json.dumps(result, ensure_ascii=True, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
