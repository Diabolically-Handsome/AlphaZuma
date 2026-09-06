from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from zuma_rl.pc_golden import (
    DMO_FILE_ID,
    DMO_FORMAT,
    DMO_VERSION,
    EXACT_STEP_MANIFEST_VERSION,
    MANIFEST_SCHEMA,
    TRACE_SCHEMA,
    ArtifactSpec,
    ComparisonContract,
    ComparisonResult,
    ComparisonStatus,
    CoordinateCalibration,
    CoverageRange,
    CoverageStatus,
    ExactStepReplayContract,
    ExactStepRunContract,
    FrameRef,
    GoldenEvent,
    GoldenInput,
    GoldenTick,
    InputTimeline,
    Measurement,
    MeasurementStatus,
    PcEnvironment,
    PcGoldenArtifactError,
    PcGoldenManifest,
    PcGoldenTrace,
    PcGoldenValidationError,
    ReplayPixelComparisonContract,
    Scenario,
    TickClock,
    VideoMetadata,
    canonical_sha256,
    capture_contract_fingerprint,
    pc_golden_native_source_fingerprint,
)


def test_replay_pixel_comparison_is_a_narrow_one_row_contract() -> None:
    contract = ReplayPixelComparisonContract(
        mode="exact_except_bounded_bottom_raster_edge",
        excluded_bottom_rows=1,
        maximum_excluded_edge_mismatches=5,
    )

    assert ReplayPixelComparisonContract.from_dict(
        contract.to_dict()
    ) == contract
    with pytest.raises(
        PcGoldenValidationError,
        match="exactly one bottom row",
    ):
        ReplayPixelComparisonContract(
            mode="exact_except_bounded_bottom_raster_edge",
            excluded_bottom_rows=2,
            maximum_excluded_edge_mismatches=5,
        )
    with pytest.raises(
        PcGoldenValidationError,
        match="mode is unsupported",
    ):
        ReplayPixelComparisonContract(
            mode="arbitrary_mask",
            excluded_bottom_rows=1,
            maximum_excluded_edge_mismatches=5,
        )


def _digest(payload: bytes | str) -> str:
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _tag_digest(tag: str) -> str:
    return _digest(tag)


def _environment(*, reverse_nested_order: bool = False) -> PcEnvironment:
    curves = (
        {"curve-b": _tag_digest("curve-b"), "curve-a": _tag_digest("curve-a")}
        if reverse_nested_order
        else {"curve-a": _tag_digest("curve-a"), "curve-b": _tag_digest("curve-b")}
    )
    settings = (
        {"music": 0.0, "fullscreen": False, "resolution": [1600, 1200]}
        if reverse_nested_order
        else {"resolution": [1600, 1200], "fullscreen": False, "music": 0.0}
    )
    difficulty = (
        {"danger": {"meter": 0.25, "stage": 2}, "lives": 3}
        if reverse_nested_order
        else {"lives": 3, "danger": {"stage": 2, "meter": 0.25}}
    )
    return PcEnvironment(
        executable_sha256=_tag_digest("exe"),
        main_pak_sha256=_tag_digest("main.pak"),
        levels_xml_sha256=_tag_digest("levels.xml"),
        curve_sha256=curves,
        pre_capture_save_sha256=_tag_digest("save"),
        profile_mode="tutorials_completed",
        renderer_api="Direct3D 9",
        renderer_mode="hardware",
        ball_radius_branch=18,
        os_build="Windows 11 24H2 build 26100.3915",
        gpu="NVIDIA GeForce RTX 5090",
        driver="576.80",
        game_settings=settings,
        dynamic_difficulty_state=difficulty,
    )


def _base_artifacts() -> tuple[dict[str, ArtifactSpec], dict[str, bytes]]:
    payloads = {
        "video": b"raw-video-evidence",
        "input_dmo": b"\x78\xef\xbe\x42\x02\x00\x00\x00dmo-v2-evidence",
        "tick_map": b"tick,pts\n0,0\n1,10\n2,20\n",
        "calibration": b'{"logical_from_raw":[[0.5,0,0],[0,0.5,0],[0,0,1]]}\n',
    }
    paths = {
        "video": "capture.mkv",
        "input_dmo": "input.dmo",
        "tick_map": "tick-map.csv",
        "calibration": "calibration.json",
    }
    specs = {
        name: ArtifactSpec(
            path=paths[name],
            sha256=_digest(payload),
            bytes=len(payload),
        )
        for name, payload in payloads.items()
    }
    return specs, payloads


def _case_semantics(
    *,
    coverage: tuple[CoverageRange, ...] | None = None,
    uncertainty_ticks: float = 0.0,
    dropped_frames: int = 0,
    duplicate_frames: int = 0,
) -> dict[str, object]:
    coverage = coverage or (
        CoverageRange(
            channel="events",
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
            required=False,
        ),
    )
    return {
        "scenario": Scenario(
            level_id="temple-1",
            hard=False,
            curve_index=0,
            gun_index=0,
            mode="adventure",
            profile_mode="tutorials_completed",
        ),
        "input_timeline": InputTimeline(
            artifact="input_dmo",
            format=DMO_FORMAT,
            file_id=DMO_FILE_ID,
            dmo_version=DMO_VERSION,
            product_version="1.0.5.600",
            random_seed=0xDEADBEEF,
            length_updates=2,
            native_tick_offset=0,
        ),
        "video": VideoMetadata(
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
            dropped_frames=dropped_frames,
            duplicate_frames=duplicate_frames,
        ),
        "clock": TickClock(
            logic_hz=(100, 1),
            tick_start=0,
            tick_end=2,
            tick0_video_pts=0,
            sample_phase="post_update_presented",
            mapping_kind="per_tick_pts_table",
            tick_map_artifact="tick_map",
            uncertainty_ticks=uncertainty_ticks,
        ),
        "coordinates": CoordinateCalibration(
            raw_width=1600,
            raw_height=1200,
            logical_width=800,
            logical_height=600,
            transform_kind="axis_aligned_affine",
            pixel_center_convention="center_at_integer",
            logical_from_raw=(
                (0.5, 0.0, 0.0),
                (0.0, 0.5, 0.0),
                (0.0, 0.0, 1.0),
            ),
            rms_error_px=0.1,
            max_error_px=0.25,
            calibration_artifact="calibration",
        ),
        "coverage": coverage,
        "comparison_contract": ComparisonContract(
            event_tick_tolerance=0,
            center_l2_tolerance_px=0.5,
            waypoint_abs_tolerance=0.01,
            max_drift_per_100_ticks=0.05,
        ),
    }


def _contract_fingerprint(
    environment: PcEnvironment,
    semantics: dict[str, object],
) -> str:
    artifacts, _ = _base_artifacts()
    return capture_contract_fingerprint(
        case_id="temple-1_reference_0001",
        scenario=semantics["scenario"],
        pc_environment_fingerprint=environment.fingerprint,
        artifacts=artifacts,
        input_timeline=semantics["input_timeline"],
        trace_artifact="trace",
        video=semantics["video"],
        clock=semantics["clock"],
        coordinates=semantics["coordinates"],
        coverage=semantics["coverage"],
        comparison_contract=semantics["comparison_contract"],
    )


def _trace(
    environment: PcEnvironment | None = None,
    *,
    semantics: dict[str, object] | None = None,
) -> PcGoldenTrace:
    environment = environment or _environment()
    semantics = semantics or _case_semantics()
    records = (
        GoldenTick(
            tick=0,
            frames=(FrameRef(frame_index=0, pts=0),),
            inputs=(),
            events=(),
            measurements={
                "score": Measurement(
                    status=MeasurementStatus.OBSERVED,
                    value=0,
                    source="memory_probe",
                ),
                "chain_centers": Measurement(
                    status=MeasurementStatus.NOT_CAPTURED,
                    reason="capture begins before segmentation is available",
                ),
            },
        ),
        GoldenTick(
            tick=1,
            frames=(FrameRef(frame_index=1, pts=10),),
            inputs=(
                GoldenInput(
                    sequence=0,
                    kind="cursor_move",
                    pts=9,
                    payload={"raw_x": 800, "raw_y": 300},
                ),
                GoldenInput(
                    sequence=1,
                    kind="fire_down",
                    pts=10,
                    payload={"button": "left"},
                ),
            ),
            events=(
                GoldenEvent(
                    sequence=0,
                    kind="shot_fired",
                    subjects=("frog", "projectile-0"),
                    payload={"color": 2},
                    source="frame_annotation",
                ),
            ),
            measurements={
                "score": Measurement(
                    status=MeasurementStatus.INFERRED,
                    value=0,
                    source="ocr",
                    uncertainty=0.1,
                    reason="digit contour is partially anti-aliased",
                )
            },
        ),
        GoldenTick(
            tick=2,
            frames=(FrameRef(frame_index=2, pts=20),),
            inputs=(
                GoldenInput(
                    sequence=0,
                    kind="fire_up",
                    pts=20,
                    payload={"button": "left"},
                ),
            ),
            events=(),
            measurements={
                "score": Measurement(
                    status=MeasurementStatus.OCCLUDED,
                    source="ocr",
                    reason="score is covered by the pause overlay",
                )
            },
        ),
    )
    return PcGoldenTrace(
        case_id="temple-1_reference_0001",
        pc_environment_fingerprint=environment.fingerprint,
        capture_contract_fingerprint=_contract_fingerprint(
            environment,
            semantics,
        ),
        records=records,
    )


def _artifact_specs(
    trace: PcGoldenTrace,
) -> tuple[dict[str, ArtifactSpec], dict[str, bytes]]:
    specs, payloads = _base_artifacts()
    payloads["trace"] = trace.to_ndjson().encode("utf-8")
    specs["trace"] = ArtifactSpec(
        path="trace.ndjson",
        sha256=_digest(payloads["trace"]),
        bytes=len(payloads["trace"]),
    )
    return specs, payloads


def _manifest(
    *,
    environment: PcEnvironment | None = None,
    trace: PcGoldenTrace | None = None,
    coverage: tuple[CoverageRange, ...] | None = None,
    uncertainty_ticks: float = 0.0,
    dropped_frames: int = 0,
    duplicate_frames: int = 0,
) -> PcGoldenManifest:
    environment = environment or _environment()
    semantics = _case_semantics(
        coverage=coverage,
        uncertainty_ticks=uncertainty_ticks,
        dropped_frames=dropped_frames,
        duplicate_frames=duplicate_frames,
    )
    trace = trace or _trace(environment, semantics=semantics)
    artifacts, _ = _artifact_specs(trace)
    return PcGoldenManifest(
        case_id=trace.case_id,
        scenario=semantics["scenario"],
        pc_environment=environment,
        pc_environment_fingerprint=environment.fingerprint,
        artifacts=artifacts,
        input_timeline=semantics["input_timeline"],
        trace_artifact="trace",
        video=semantics["video"],
        clock=semantics["clock"],
        coordinates=semantics["coordinates"],
        coverage=semantics["coverage"],
        comparison_contract=semantics["comparison_contract"],
        producer={"tool": "pc-capture", "version": 1},
    )


def _exact_step_manifest() -> PcGoldenManifest:
    environment = _environment()
    semantics = _case_semantics()
    artifact_names = {
        "input_dmo",
        "calibration",
        "video_r1",
        "video_r2",
        "tick_map_r1",
        "tick_map_r2",
        "trace_r1",
        "trace_r2",
        "attempts_r1",
        "attempts_r2",
        "probe_r1",
        "probe_r2",
        "index_r1",
        "index_r2",
        "post_r1",
        "post_r2",
        "pre",
        "host_pre",
        "preregistration",
        "execution_binding",
        "comparison",
    }
    artifacts = {
        name: ArtifactSpec(
            path=f"evidence/{name}.bin",
            sha256=_tag_digest(name),
            bytes=len(name),
        )
        for name in artifact_names
    }
    videos = {
        run_id: VideoMetadata(
            artifact=f"video_{run_id}",
            width=800,
            height=600,
            codec="ffv1",
            pixel_format="bgra",
            time_base=(1, 100),
            nominal_fps=(100, 1),
            frame_count=3,
            first_pts=0,
            last_pts=2,
            cfr=True,
            dropped_frames=0,
            duplicate_frames=0,
        )
        for run_id in ("r1", "r2")
    }
    clocks = {
        run_id: TickClock(
            logic_hz=(100, 1),
            tick_start=0,
            tick_end=2,
            tick0_video_pts=0,
            sample_phase="post_update_presented",
            mapping_kind="per_tick_pts_table",
            tick_map_artifact=f"tick_map_{run_id}",
            uncertainty_ticks=0.0,
        )
        for run_id in ("r1", "r2")
    }
    runs = tuple(
        ExactStepRunContract(
            run_id=run_id,
            selected_attempt=1,
            attempts_artifact=f"attempts_{run_id}",
            memory_probe_artifact=f"probe_{run_id}",
            trajectory_index_artifact=f"index_{run_id}",
            post_snapshot_artifact=f"post_{run_id}",
            trace_artifact=f"trace_{run_id}",
            video=videos[run_id],
            clock=clocks[run_id],
            process_id=100 + index,
            process_creation_filetime_100ns=10_000 + index,
        )
        for index, run_id in enumerate(("r1", "r2"), start=1)
    )
    exact_step = ExactStepReplayContract(
        preregistration_artifact="preregistration",
        execution_binding_artifact="execution_binding",
        comparison_artifact="comparison",
        pre_snapshot_artifact="pre",
        host_pre_snapshot_artifact="host_pre",
        runs=runs,
        freeze_update=97,
        source_start_update=98,
        source_end_update=100,
        warmup_tick_count=1,
        maximum_startup_attempts=3,
        sample_phase="frozen_post_replay_update_barrier",
        pixel_comparison="full_800x600_bgra_exact",
    )
    coordinates = CoordinateCalibration(
        raw_width=800,
        raw_height=600,
        logical_width=800,
        logical_height=600,
        transform_kind="axis_aligned_affine",
        pixel_center_convention="center_at_integer",
        logical_from_raw=(
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        rms_error_px=0.0,
        max_error_px=0.0,
        calibration_artifact="calibration",
    )
    return PcGoldenManifest(
        case_id="temple-1_exact_step_0001",
        scenario=semantics["scenario"],
        pc_environment=environment,
        pc_environment_fingerprint=environment.fingerprint,
        artifacts=artifacts,
        input_timeline=semantics["input_timeline"],
        trace_artifact="trace_r1",
        video=videos["r1"],
        clock=clocks["r1"],
        coordinates=coordinates,
        coverage=semantics["coverage"],
        comparison_contract=semantics["comparison_contract"],
        producer={"tool": "exact-step-test"},
        manifest_version=EXACT_STEP_MANIFEST_VERSION,
        exact_step_replay=exact_step,
    )


def test_exact_step_manifest_v5_is_separate_and_round_trips() -> None:
    manifest = _exact_step_manifest()

    assert manifest.manifest_version == EXACT_STEP_MANIFEST_VERSION
    assert manifest.save_transaction is None
    assert manifest.replay_determinism is None
    assert manifest.exact_step_replay is not None
    assert PcGoldenManifest.from_json(manifest.to_json()).to_dict() == (
        manifest.to_dict()
    )
    assert manifest.capture_contract_excluded_artifacts == (
        "trace_r1",
        "trace_r2",
    )
    assert pc_golden_native_source_fingerprint(manifest).startswith(
        "sha256:"
    )

    relabeled = replace(
        manifest,
        exact_step_replay=replace(
            manifest.exact_step_replay,
            freeze_update=96,
            source_start_update=97,
            source_end_update=99,
            warmup_tick_count=1,
        ),
    )
    assert relabeled.capture_contract_fingerprint != (
        manifest.capture_contract_fingerprint
    )


def test_manifest_versions_cannot_cross_label_transport_contracts() -> None:
    exact = _exact_step_manifest()
    with pytest.raises(PcGoldenValidationError, match="cannot declare v5"):
        replace(exact, manifest_version=4)
    with pytest.raises(PcGoldenValidationError, match="require an ExactStep"):
        replace(exact, exact_step_replay=None)
    with pytest.raises(PcGoldenValidationError, match="cannot relabel"):
        replace(
            exact,
            save_transaction=object(),
        )


def test_manifest_v3_round_trip_and_canonical_environment_hash(tmp_path: Path) -> None:
    environment_a = _environment()
    environment_b = _environment(reverse_nested_order=True)
    assert json.dumps(environment_a.to_dict()) != json.dumps(
        environment_b.to_dict()
    )
    assert environment_a.fingerprint == environment_b.fingerprint
    assert environment_a.fingerprint == canonical_sha256(environment_a.to_dict())
    fingerprint_before_export = environment_a.fingerprint
    exported = environment_a.to_dict()
    exported["dynamic_difficulty_state"]["danger"]["meter"] = 99
    assert environment_a.fingerprint == fingerprint_before_export
    assert environment_a.dynamic_difficulty_state["danger"]["meter"] == 0.25
    with pytest.raises(TypeError):
        environment_a.dynamic_difficulty_state["danger"]["meter"] = 99

    manifest = _manifest(environment=environment_a)
    assert manifest.to_dict()["schema"] == MANIFEST_SCHEMA
    decoded = PcGoldenManifest.from_json(manifest.to_json())
    assert decoded.to_dict() == manifest.to_dict()

    destination = tmp_path / "manifest.json"
    manifest.write_json(destination)
    assert PcGoldenManifest.read_json(destination).to_dict() == manifest.to_dict()

    tampered = manifest.to_dict()
    tampered["pc_environment_fingerprint"] = _tag_digest("wrong")
    with pytest.raises(PcGoldenValidationError, match="does not match canonical"):
        PcGoldenManifest.from_dict(tampered)


@pytest.mark.parametrize(
    ("scope", "field"),
    [
        ("manifest", "future_field"),
        ("scenario", "future_field"),
        ("pc_environment", "future_field"),
        ("artifact", "future_field"),
        ("input_timeline", "future_field"),
        ("video", "future_field"),
        ("clock", "future_field"),
        ("coordinates", "future_field"),
        ("coverage", "future_field"),
        ("comparison_contract", "future_field"),
    ],
)
def test_manifest_rejects_unknown_fields(scope: str, field: str) -> None:
    data = _manifest().to_dict()
    if scope == "manifest":
        target = data
    elif scope == "artifact":
        target = data["artifacts"]["video"]
    elif scope == "coverage":
        target = data["coverage"][0]
    else:
        target = data[scope]
    target[field] = True
    with pytest.raises(PcGoldenValidationError, match="unknown fields"):
        PcGoldenManifest.from_dict(data)


def test_strict_json_rejects_duplicate_keys_and_nonfinite_constants() -> None:
    with pytest.raises(PcGoldenValidationError, match="duplicate JSON key"):
        PcGoldenManifest.from_json('{"schema":"a","schema":"b"}')
    with pytest.raises(PcGoldenValidationError, match="forbidden constant NaN"):
        PcGoldenManifest.from_json('{"value":NaN}')
    with pytest.raises(PcGoldenValidationError, match="invalid manifest JSON"):
        PcGoldenManifest.from_json('{"schema":' + "9" * 5000 + "}")
    with pytest.raises(PcGoldenValidationError, match="NaN or infinity"):
        Measurement(
            status=MeasurementStatus.OBSERVED,
            value={"x": np.inf},
            source="probe",
        )
    deeply_nested: object = 0
    for _ in range(140):
        deeply_nested = {"value": deeply_nested}
    manifest_data = _manifest().to_dict()
    manifest_data["producer"] = deeply_nested
    with pytest.raises(PcGoldenValidationError, match="nesting depth"):
        PcGoldenManifest.from_json(json.dumps(manifest_data))

    huge_number = _manifest().to_dict()
    huge_number["clock"]["uncertainty_ticks"] = 10**3000
    with pytest.raises(PcGoldenValidationError, match="must be finite"):
        PcGoldenManifest.from_dict(huge_number)


def test_measurement_statuses_distinguish_values_from_missing_evidence() -> None:
    observed = Measurement(
        status=MeasurementStatus.OBSERVED,
        value={"center": [12.0, 14.0]},
        source="segmentation",
        uncertainty=0.25,
    )
    assert Measurement.from_dict(observed.to_dict()) == observed

    observed_null = Measurement(
        status=MeasurementStatus.OBSERVED,
        value=None,
        source="probe",
    )
    assert observed_null.to_dict()["value"] is None
    with pytest.raises(PcGoldenValidationError, match="contain a value field"):
        Measurement.from_dict({"status": "observed", "source": "probe"})
    with pytest.raises(PcGoldenValidationError, match="reason"):
        Measurement(
            status=MeasurementStatus.INFERRED,
            value=5,
            source="ocr",
        )
    with pytest.raises(PcGoldenValidationError, match="must not contain a value"):
        Measurement(
            status=MeasurementStatus.OCCLUDED,
            value=0,
            reason="hidden",
        )
    with pytest.raises(PcGoldenValidationError, match="must not contain a value"):
        Measurement.from_dict(
            {"status": "not_captured", "value": None, "reason": "not recorded"}
        )
    with pytest.raises(PcGoldenValidationError, match="reason"):
        Measurement(status=MeasurementStatus.NOT_APPLICABLE)


def test_trace_measurement_vocabulary_and_count_geometry_are_closed() -> None:
    with pytest.raises(PcGoldenValidationError, match="unknown"):
        GoldenTick(
            tick=0,
            frames=(),
            inputs=(),
            events=(),
            measurements={
                "future_private_state": Measurement(
                    status=MeasurementStatus.OBSERVED,
                    value=1,
                    source="synthetic",
                )
            },
        )

    with pytest.raises(PcGoldenValidationError, match="does not match"):
        GoldenTick(
            tick=0,
            frames=(),
            inputs=(),
            events=(),
            measurements={
                "chain_count": Measurement(
                    status=MeasurementStatus.OBSERVED,
                    value=2,
                    source="synthetic",
                ),
                "chain_centers": Measurement(
                    status=MeasurementStatus.OBSERVED,
                    value=[[1.0, 2.0]],
                    source="synthetic",
                ),
            },
        )


def test_input_timeline_strictly_identifies_popcap_dmo_v2() -> None:
    timeline = _manifest().input_timeline
    assert InputTimeline.from_dict(timeline.to_dict()) == timeline
    assert timeline.file_id == 0x42BEEF78

    invalid_cases = (
        ({"format": "json_inputs"}, "format"),
        ({"file_id": 0x42BEEF79}, "file_id"),
        ({"dmo_version": 1}, "dmo_version"),
        ({"product_version": " "}, "product_version"),
        ({"random_seed": -1}, "random_seed"),
        ({"random_seed": 0x1_0000_0000}, "unsigned 32-bit"),
        ({"length_updates": -1}, "length_updates"),
    )
    for changes, expected_message in invalid_cases:
        with pytest.raises(PcGoldenValidationError, match=expected_message):
            replace(timeline, **changes)

    unknown = timeline.to_dict()
    unknown["future_field"] = 1
    with pytest.raises(PcGoldenValidationError, match="unknown fields"):
        InputTimeline.from_dict(unknown)

    manifest_data = _manifest().to_dict()
    del manifest_data["input_timeline"]
    with pytest.raises(PcGoldenValidationError, match="missing fields"):
        PcGoldenManifest.from_dict(manifest_data)


def test_artifact_verification_checks_path_size_and_sha(tmp_path: Path) -> None:
    trace = _trace()
    manifest = _manifest(trace=trace)
    _, payloads = _artifact_specs(trace)
    for name, spec in manifest.artifacts.items():
        (tmp_path / spec.path).write_bytes(payloads[name])

    verified = manifest.verify_artifacts(tmp_path)
    assert set(verified) == {
        "video",
        "input_dmo",
        "tick_map",
        "calibration",
        "trace",
    }

    video_path = tmp_path / manifest.artifacts["video"].path
    video_path.write_bytes(b"x")
    with pytest.raises(PcGoldenArtifactError, match="size differs"):
        manifest.verify_artifacts(tmp_path)

    expected_size = manifest.artifacts["video"].bytes
    video_path.write_bytes(b"x" * expected_size)
    with pytest.raises(PcGoldenArtifactError, match="SHA-256 differs"):
        manifest.verify_artifacts(tmp_path)

    with pytest.raises(PcGoldenValidationError, match="relative POSIX"):
        ArtifactSpec(
            path="../escape.bin",
            sha256=_tag_digest("escape"),
            bytes=1,
        )


def test_coordinate_calibration_maps_raw_pixels_and_rejects_bad_matrices() -> None:
    calibration = _manifest().coordinates
    actual = calibration.raw_to_logical(
        np.asarray([[0.0, 0.0], [800.0, 600.0], [1600.0, 1200.0]])
    )
    np.testing.assert_allclose(
        actual,
        np.asarray([[0.0, 0.0], [400.0, 300.0], [800.0, 600.0]]),
    )

    base = calibration.to_dict()
    for bad_matrix in (
        [[1.0, 0.0, 0.0], [0.0, np.nan, 0.0], [0.0, 0.0, 1.0]],
        [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        [[1.0, 0.1, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    ):
        invalid = dict(base)
        invalid["logical_from_raw"] = bad_matrix
        with pytest.raises(PcGoldenValidationError):
            CoordinateCalibration.from_dict(invalid)
    for malformed in (
        [[1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "not-a-matrix",
        [[object(), 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    ):
        with pytest.raises(PcGoldenValidationError, match="finite 3x3"):
            replace(calibration, logical_from_raw=malformed)
    with pytest.raises(PcGoldenValidationError, match="numeric array"):
        calibration.raw_to_logical(["not-a-point"])


def test_trace_v2_is_canonical_contiguous_ndjson_and_round_trips(
    tmp_path: Path,
) -> None:
    trace = _trace()
    text = trace.to_ndjson()
    lines = text.splitlines()
    assert json.loads(lines[0])["schema"] == TRACE_SCHEMA
    assert [json.loads(line)["tick"] for line in lines[1:]] == [0, 1, 2]
    assert PcGoldenTrace.from_ndjson(text).to_ndjson() == text
    assert trace.canonical_content_sha256 == _digest(text)
    noncanonical = "\n".join(
        json.dumps(json.loads(line), sort_keys=False)
        for line in text.splitlines()
    ) + "\n"
    assert noncanonical != text
    with pytest.raises(PcGoldenValidationError, match="not in canonical"):
        PcGoldenTrace.from_ndjson(noncanonical)

    destination = tmp_path / "trace.ndjson"
    trace.write_ndjson(destination)
    assert PcGoldenTrace.read_ndjson(destination).to_ndjson() == text

    record = json.loads(lines[2])
    record["future_field"] = 1
    lines[2] = json.dumps(record)
    with pytest.raises(PcGoldenValidationError, match="unknown fields"):
        PcGoldenTrace.from_ndjson("\n".join(lines) + "\n")


def test_trace_requires_tick_zero_baseline_and_continuity() -> None:
    trace = _trace()
    with pytest.raises(PcGoldenValidationError, match="expected 1, got 2"):
        PcGoldenTrace(
            case_id=trace.case_id,
            pc_environment_fingerprint=trace.pc_environment_fingerprint,
            capture_contract_fingerprint=(
                trace.capture_contract_fingerprint
            ),
            records=(trace.records[0], trace.records[2]),
        )

    with pytest.raises(PcGoldenValidationError, match="tick zero"):
        GoldenTick(
            tick=0,
            frames=(),
            inputs=(
                GoldenInput(
                    sequence=0,
                    kind="fire_down",
                    pts=0,
                    payload={},
                ),
            ),
            events=(),
            measurements={},
        )


def test_input_sequence_and_pts_are_strict() -> None:
    with pytest.raises(PcGoldenValidationError, match="contiguous"):
        GoldenTick(
            tick=1,
            frames=(),
            inputs=(
                GoldenInput(sequence=1, kind="fire", pts=10, payload={}),
            ),
            events=(),
            measurements={},
        )
    with pytest.raises(PcGoldenValidationError, match="non-decreasing"):
        GoldenTick(
            tick=1,
            frames=(),
            inputs=(
                GoldenInput(sequence=0, kind="move", pts=20, payload={}),
                GoldenInput(sequence=1, kind="fire", pts=10, payload={}),
            ),
            events=(),
            measurements={},
        )

    environment = _environment()
    reference = _trace(environment)
    reversed_pts = PcGoldenTrace(
        case_id=reference.case_id,
        pc_environment_fingerprint=reference.pc_environment_fingerprint,
        capture_contract_fingerprint=(
            reference.capture_contract_fingerprint
        ),
        records=(
            reference.records[0],
            replace(
                reference.records[1],
                inputs=(
                    GoldenInput(sequence=0, kind="fire", pts=20, payload={}),
                ),
            ),
            replace(
                reference.records[2],
                inputs=(
                    GoldenInput(sequence=0, kind="fire", pts=10, payload={}),
                ),
            ),
        ),
    )
    manifest = _manifest(environment=environment, trace=reversed_pts)
    with pytest.raises(PcGoldenValidationError, match="across trace ticks"):
        reversed_pts.validate_against_manifest(manifest)

    outside_video = replace(
        reference,
        records=(
            reference.records[0],
            replace(
                reference.records[1],
                inputs=(
                    GoldenInput(sequence=0, kind="fire", pts=21, payload={}),
                ),
            ),
            reference.records[2],
        ),
    )
    outside_manifest = _manifest(environment=environment, trace=outside_video)
    with pytest.raises(PcGoldenValidationError, match="outside video"):
        outside_video.validate_against_manifest(outside_manifest)


def test_trace_cross_validates_manifest_and_trace_artifact_identity() -> None:
    environment = _environment()
    trace = _trace(environment)
    manifest = _manifest(environment=environment, trace=trace)
    trace.validate_against_manifest(manifest)

    with pytest.raises(PcGoldenValidationError, match="case_id"):
        replace(trace, case_id="different").validate_against_manifest(manifest)
    with pytest.raises(PcGoldenValidationError, match="environment fingerprint"):
        replace(
            trace,
            pc_environment_fingerprint=_tag_digest("different"),
        ).validate_against_manifest(manifest)
    with pytest.raises(PcGoldenValidationError, match="final tick"):
        replace(trace, records=trace.records[:2]).validate_against_manifest(manifest)

    specs = dict(manifest.artifacts)
    specs["trace"] = replace(specs["trace"], sha256=_tag_digest("different trace"))
    bad_manifest = replace(manifest, artifacts=specs)
    with pytest.raises(PcGoldenValidationError, match="trace artifact SHA-256"):
        trace.validate_against_manifest(bad_manifest)

    repeated_frame = replace(
        trace,
        records=(
            trace.records[0],
            trace.records[1],
            replace(
                trace.records[2],
                frames=(FrameRef(frame_index=1, pts=20),),
            ),
        ),
    )
    repeated_frame_manifest = _manifest(
        environment=environment,
        trace=repeated_frame,
    )
    with pytest.raises(PcGoldenValidationError, match="indices must increase"):
        repeated_frame.validate_against_manifest(repeated_frame_manifest)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda manifest: replace(
            manifest,
            scenario=replace(manifest.scenario, hard=True),
        ),
        lambda manifest: replace(
            manifest,
            input_timeline=replace(
                manifest.input_timeline,
                native_tick_offset=1,
            ),
        ),
        lambda manifest: replace(
            manifest,
            video=replace(manifest.video, codec="huffyuv"),
        ),
        lambda manifest: replace(
            manifest,
            clock=replace(manifest.clock, tick0_video_pts=10),
        ),
        lambda manifest: replace(
            manifest,
            coordinates=replace(
                manifest.coordinates,
                rms_error_px=0.11,
            ),
        ),
        lambda manifest: replace(
            manifest,
            coverage=tuple(
                replace(item, required=not item.required)
                for item in manifest.coverage
            ),
        ),
        lambda manifest: replace(
            manifest,
            comparison_contract=replace(
                manifest.comparison_contract,
                center_l2_tolerance_px=0.4,
            ),
        ),
    ],
)
def test_trace_capture_contract_rejects_semantic_relabeling(mutation) -> None:
    environment = _environment()
    trace = _trace(environment)
    manifest = _manifest(environment=environment, trace=trace)
    relabeled = mutation(manifest)

    assert (
        relabeled.capture_contract_fingerprint
        != manifest.capture_contract_fingerprint
    )
    with pytest.raises(
        PcGoldenValidationError,
        match="capture-contract fingerprint",
    ):
        trace.validate_against_manifest(relabeled)


def test_serialized_capture_contract_fingerprint_is_strict_and_producer_free() -> None:
    manifest = _manifest()
    data = manifest.to_dict()
    data["scenario"]["hard"] = True
    with pytest.raises(
        PcGoldenValidationError,
        match="capture_contract_fingerprint",
    ):
        PcGoldenManifest.from_dict(data)

    producer_only = replace(
        manifest,
        producer={"tool": "different-exporter", "version": 99},
    )
    assert (
        producer_only.capture_contract_fingerprint
        == manifest.capture_contract_fingerprint
    )


def test_manifest_rejects_bad_cross_references_and_coverage() -> None:
    manifest = _manifest()
    with pytest.raises(PcGoldenValidationError, match="unknown artifacts"):
        replace(manifest, trace_artifact="unknown")
    with pytest.raises(PcGoldenValidationError, match="unknown artifacts"):
        replace(
            manifest,
            input_timeline=replace(
                manifest.input_timeline,
                artifact="unknown",
            ),
        )

    duplicate_paths = dict(manifest.artifacts)
    duplicate_paths["video"] = replace(
        duplicate_paths["video"],
        path=duplicate_paths["trace"].path,
    )
    with pytest.raises(PcGoldenValidationError, match="reuse the same"):
        replace(manifest, artifacts=duplicate_paths)

    overlapping = (
        CoverageRange(
            channel="events",
            start_tick=0,
            end_tick=1,
            status=CoverageStatus.COMPLETE,
            required=True,
        ),
        CoverageRange(
            channel="events",
            start_tick=1,
            end_tick=2,
            status=CoverageStatus.COMPLETE,
            required=True,
        ),
    )
    with pytest.raises(PcGoldenValidationError, match="overlap"):
        replace(manifest, coverage=overlapping)

    exceeds = (
        CoverageRange(
            channel="events",
            start_tick=0,
            end_tick=3,
            status=CoverageStatus.COMPLETE,
            required=True,
        ),
    )
    with pytest.raises(PcGoldenValidationError, match="exceeds"):
        replace(manifest, coverage=exceeds)

    gap = (
        CoverageRange(
            channel="events",
            start_tick=0,
            end_tick=0,
            status=CoverageStatus.COMPLETE,
            required=True,
        ),
        CoverageRange(
            channel="events",
            start_tick=2,
            end_tick=2,
            status=CoverageStatus.COMPLETE,
            required=True,
        ),
    )
    with pytest.raises(PcGoldenValidationError, match="gap"):
        replace(manifest, coverage=gap)


def test_readiness_and_comparison_result_have_three_explicit_outcomes() -> None:
    ready = _manifest().coverage_readiness()
    assert ready.status is ComparisonStatus.PASS
    assert ready.reasons == ()

    incomplete_coverage = (
        CoverageRange(
            channel="events",
            start_tick=0,
            end_tick=2,
            status=CoverageStatus.PARTIAL,
            required=True,
            reason="two frames are obscured by a transition",
        ),
    )
    incomparable = _manifest(
        coverage=incomplete_coverage,
        uncertainty_ticks=0.25,
        dropped_frames=1,
        duplicate_frames=1,
    ).coverage_readiness()
    assert incomparable.status is ComparisonStatus.INCOMPARABLE
    assert len(incomparable.reasons) == 4

    failed = ComparisonResult(
        status=ComparisonStatus.FAIL,
        reasons=("event mismatch at tick 120",),
        metrics={"event_mismatches": 1},
    )
    assert ComparisonResult.from_dict(failed.to_dict()) == failed
    with pytest.raises(PcGoldenValidationError, match="require at least one reason"):
        ComparisonResult(status=ComparisonStatus.FAIL)
    with pytest.raises(PcGoldenValidationError, match="JSON array"):
        ComparisonResult(
            status=ComparisonStatus.FAIL,
            reasons="not a sequence of reasons",
        )


def test_manifest_rejects_invalid_native_clock_and_coordinate_dimensions() -> None:
    manifest = _manifest()
    with pytest.raises(PcGoldenValidationError, match="100/1"):
        replace(manifest.clock, logic_hz=(60, 1))
    with pytest.raises(PcGoldenValidationError, match="800x600"):
        replace(manifest.coordinates, logical_width=1920)
    with pytest.raises(PcGoldenValidationError, match="must match video"):
        replace(
            manifest,
            coordinates=replace(manifest.coordinates, raw_width=1920),
        )
    with pytest.raises(PcGoldenValidationError, match="outside the video"):
        replace(
            manifest,
            clock=replace(manifest.clock, tick0_video_pts=21),
        )
