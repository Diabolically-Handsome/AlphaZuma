from __future__ import annotations

import hashlib
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from zuma_rl.golden_diff import (
    SimulatorTick,
    _compare_pc_trace,
    compare_pc_golden_case,
    compare_pc_trace,
    snapshot_revenge_simulator,
)
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
    GoldenEvent,
    GoldenTick,
    InputTimeline,
    Measurement,
    MeasurementStatus,
    PcEnvironment,
    PcGoldenManifest,
    PcGoldenTrace,
    PcGoldenValidationError,
    Scenario,
    TickClock,
    VideoMetadata,
    capture_contract_fingerprint,
)
from zuma_rl.revenge_core import GunState, RevengeSimulator


def _digest(value: bytes | str) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _sim_tick(
    tick: int,
    events: tuple[str, ...] = (),
    measurements: dict[str, object] | None = None,
) -> SimulatorTick:
    return SimulatorTick(
        tick=tick,
        source_tick=tick,
        sample_phase="post_update_presented",
        events=events,
        measurements={} if measurements is None else measurements,
    )


def _measurement(
    value: object,
    *,
    inferred: bool = False,
) -> Measurement:
    return Measurement(
        status=(
            MeasurementStatus.INFERRED
            if inferred
            else MeasurementStatus.OBSERVED
        ),
        value=value,
        source="synthetic_test",
        reason="synthetic inference" if inferred else None,
    )


def _environment() -> PcEnvironment:
    return PcEnvironment(
        executable_sha256=_digest("exe"),
        main_pak_sha256=_digest("pak"),
        levels_xml_sha256=_digest("levels"),
        curve_sha256={"curve": _digest("curve")},
        pre_capture_save_sha256=_digest("save"),
        profile_mode="tutorials_completed",
        renderer_api="Direct3D 9",
        renderer_mode="hardware",
        ball_radius_branch=18,
        os_build="synthetic",
        gpu="synthetic",
        driver="synthetic",
    )


def _coverage(
    channel: str,
    tick_end: int,
    *,
    required: bool = True,
    status: CoverageStatus = CoverageStatus.COMPLETE,
) -> CoverageRange:
    return CoverageRange(
        channel=channel,
        start_tick=0,
        end_tick=tick_end,
        status=status,
        required=required,
        reason=None if status is CoverageStatus.COMPLETE else "synthetic gap",
    )


def _make_case(
    records: tuple[GoldenTick, ...],
    *,
    coverage: tuple[CoverageRange, ...],
    center_tolerance: float = 0.5,
    waypoint_tolerance: float = 0.05,
) -> tuple[PcGoldenManifest, PcGoldenTrace]:
    environment = _environment()
    case_id = "synthetic-diff"
    tick_end = records[-1].tick
    payloads = {
        "video": b"video",
        "input": b"input",
        "tick_map": b"ticks",
        "calibration": b"calibration",
    }
    artifacts = {
        name: ArtifactSpec(
            path=f"{name}.bin",
            sha256=_digest(payload),
            bytes=len(payload),
        )
        for name, payload in payloads.items()
    }
    scenario = Scenario(
        level_id="synthetic",
        hard=False,
        curve_index=0,
        gun_index=0,
        mode="adventure",
        profile_mode="tutorials_completed",
    )
    input_timeline = InputTimeline(
        artifact="input",
        format=DMO_FORMAT,
        file_id=DMO_FILE_ID,
        dmo_version=DMO_VERSION,
        product_version="synthetic",
        random_seed=1,
        length_updates=tick_end,
        native_tick_offset=0,
    )
    video = VideoMetadata(
        artifact="video",
        width=800,
        height=600,
        codec="ffv1",
        pixel_format="bgra",
        time_base=(1, 100),
        nominal_fps=(100, 1),
        frame_count=max(1, len(records)),
        first_pts=0,
        last_pts=tick_end,
        cfr=True,
        dropped_frames=0,
        duplicate_frames=0,
    )
    clock = TickClock(
        logic_hz=(100, 1),
        tick_start=0,
        tick_end=tick_end,
        tick0_video_pts=0,
        sample_phase="post_update_presented",
        mapping_kind="synthetic",
        tick_map_artifact="tick_map",
        uncertainty_ticks=0.0,
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
    comparison_contract = ComparisonContract(
        event_tick_tolerance=0,
        center_l2_tolerance_px=center_tolerance,
        waypoint_abs_tolerance=waypoint_tolerance,
        max_drift_per_100_ticks=0.05,
    )
    contract_fingerprint = capture_contract_fingerprint(
        case_id=case_id,
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
    trace = PcGoldenTrace(
        case_id=case_id,
        pc_environment_fingerprint=environment.fingerprint,
        capture_contract_fingerprint=contract_fingerprint,
        records=records,
    )
    trace_payload = trace.to_ndjson().encode("utf-8")
    artifacts["trace"] = ArtifactSpec(
        path="trace.bin",
        sha256=_digest(trace_payload),
        bytes=len(trace_payload),
    )
    manifest = PcGoldenManifest(
        case_id=trace.case_id,
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
        producer={"tool": "synthetic_test"},
    )
    return manifest, trace


def _all_measurements(
    *,
    score: int,
    chain_centers: list[list[float]],
    projectile_centers: list[list[float]],
    chain_waypoints: list[float],
    inferred_score: bool = False,
) -> dict[str, Measurement]:
    return {
        "score": _measurement(score, inferred=inferred_score),
        "current_color": _measurement(0),
        "next_color": _measurement(1),
        "gun_state": _measurement("normal"),
        "outcome": _measurement(None),
        "zuma_reached": _measurement(False),
        "chain_count": _measurement(len(chain_centers)),
        "projectile_count": _measurement(len(projectile_centers)),
        "chain_centers": _measurement(chain_centers),
        "projectile_centers": _measurement(projectile_centers),
        "chain_waypoints": _measurement(chain_waypoints),
    }


def _all_channels_coverage(tick_end: int) -> tuple[CoverageRange, ...]:
    channels = (
        "events",
        "score",
        "current_color",
        "next_color",
        "gun_state",
        "outcome",
        "zuma_reached",
        "chain_count",
        "projectile_count",
        "chain_centers",
        "chain_waypoints",
    )
    return tuple(_coverage(channel, tick_end) for channel in channels)


def test_compare_passes_exact_scalars_inferred_values_and_ordered_geometry() -> None:
    records = (
        GoldenTick(
            tick=0,
            frames=(),
            inputs=(),
            events=(),
            measurements=_all_measurements(
                score=0,
                chain_centers=[[10.0, 20.0], [30.0, 40.0]],
                projectile_centers=[],
                chain_waypoints=[100.0, 136.0],
            ),
        ),
        GoldenTick(
            tick=1,
            frames=(),
            inputs=(),
            events=(
                GoldenEvent(
                    kind="shot_fired",
                    subjects=(),
                    payload={"ignored": "by-kind comparison"},
                    source="synthetic_test",
                ),
            ),
            measurements=_all_measurements(
                score=10,
                inferred_score=True,
                chain_centers=[[11.0, 20.0], [31.0, 40.0]],
                projectile_centers=[[400.0, 300.0]],
                chain_waypoints=[101.0, 137.0],
            ),
        ),
    )
    manifest, trace = _make_case(
        records,
        coverage=_all_channels_coverage(1),
    )
    simulator_ticks = (
        _sim_tick(
            0,
            (),
            {
                "score": 0,
                "current_color": 0,
                "next_color": 1,
                "gun_state": "normal",
                "outcome": None,
                "zuma_reached": False,
                "chain_count": 2,
                "projectile_count": 0,
                "chain_centers": [[10.3, 20.4], [30.3, 40.4]],
                "projectile_centers": [],
                "chain_waypoints": [100.04, 136.0],
            },
        ),
        _sim_tick(
            1,
            ("shot_fired",),
            {
                "score": 10,
                "current_color": 0,
                "next_color": 1,
                "gun_state": "normal",
                "outcome": None,
                "zuma_reached": False,
                "chain_count": 2,
                "projectile_count": 1,
                "chain_centers": [[11.3, 20.4], [31.3, 40.4]],
                "projectile_centers": [[400.3, 300.4]],
                "chain_waypoints": [101.04, 137.0],
            },
        ),
    )

    result = _compare_pc_trace(manifest, trace, simulator_ticks)

    assert result.status is ComparisonStatus.PASS
    assert result.metrics["scalar_comparisons"] == 16
    assert result.metrics["events_matched"] == 1
    assert result.metrics["max_center_l2_error_px"] == pytest.approx(0.5)
    assert result.metrics["max_waypoint_abs_error"] == pytest.approx(0.04)


def test_accumulating_ordered_chain_error_exceeds_drift_contract() -> None:
    tick_end = 100
    records = tuple(
        GoldenTick(
            tick=tick,
            frames=(),
            inputs=(),
            events=(),
            measurements={
                "chain_centers": _measurement([[10.0, 20.0]]),
                "chain_waypoints": _measurement([100.0]),
            },
        )
        for tick in range(tick_end + 1)
    )
    manifest, trace = _make_case(
        records,
        coverage=(
            _coverage("chain_centers", tick_end),
            _coverage("chain_waypoints", tick_end),
        ),
        center_tolerance=0.5,
        waypoint_tolerance=0.05,
    )
    simulator_ticks = tuple(
        _sim_tick(
            tick,
            (),
            {
                "chain_centers": [[10.0 + tick * 0.0006, 20.0]],
                "chain_waypoints": [100.0 + tick * 0.0006],
            },
        )
        for tick in range(tick_end + 1)
    )

    result = _compare_pc_trace(manifest, trace, simulator_ticks)

    assert result.metrics["max_center_l2_error_px"] < 1.0
    assert result.metrics["max_waypoint_abs_error"] < 1.0
    assert result.metrics["max_drift_per_100_ticks"] == pytest.approx(0.06)
    assert result.status is ComparisonStatus.FAIL
    assert any("drift tolerance" in reason for reason in result.reasons)


def test_transient_error_spike_is_not_misreported_as_net_drift() -> None:
    tick_end = 150
    records = tuple(
        GoldenTick(
            tick=tick,
            frames=(),
            inputs=(),
            events=(),
            measurements={
                "chain_centers": _measurement([[10.0, 20.0]]),
                "chain_waypoints": _measurement([100.0]),
            },
        )
        for tick in range(tick_end + 1)
    )
    manifest, trace = _make_case(
        records,
        coverage=(
            _coverage("chain_centers", tick_end),
            _coverage("chain_waypoints", tick_end),
        ),
        center_tolerance=0.5,
        waypoint_tolerance=0.05,
    )
    simulator_ticks = tuple(
        _sim_tick(
            tick,
            measurements={
                "chain_centers": [
                    [10.1 if tick == 50 else 10.0, 20.0]
                ],
                "chain_waypoints": [
                    100.04 if tick == 50 else 100.0
                ],
            },
        )
        for tick in range(tick_end + 1)
    )

    result = _compare_pc_trace(manifest, trace, simulator_ticks)

    assert result.status is ComparisonStatus.PASS
    assert result.metrics["max_center_l2_error_px"] == pytest.approx(0.1)
    assert result.metrics["max_waypoint_abs_error"] == pytest.approx(0.04)
    assert result.metrics["max_drift_per_100_ticks"] == 0.0


def test_topology_event_cuts_the_exact_drift_window() -> None:
    tick_end = 150
    topology_event = GoldenEvent(
        kind="inserted",
        subjects=(),
        payload={},
        source="synthetic_test",
    )
    records = tuple(
        GoldenTick(
            tick=tick,
            frames=(),
            inputs=(),
            events=(topology_event,) if tick == 75 else (),
            measurements={
                "chain_centers": _measurement([[10.0, 20.0]]),
                "chain_waypoints": _measurement([100.0]),
            },
        )
        for tick in range(tick_end + 1)
    )
    manifest, trace = _make_case(
        records,
        coverage=(
            _coverage("events", tick_end),
            _coverage("chain_centers", tick_end),
            _coverage("chain_waypoints", tick_end),
        ),
        center_tolerance=0.5,
        waypoint_tolerance=0.05,
    )
    simulator_ticks = tuple(
        _sim_tick(
            tick,
            ("inserted",) if tick == 75 else (),
            {
                "chain_centers": [[10.0 + tick * 0.0006, 20.0]],
                "chain_waypoints": [100.0],
            },
        )
        for tick in range(tick_end + 1)
    )

    result = _compare_pc_trace(manifest, trace, simulator_ticks)

    assert result.status is ComparisonStatus.PASS
    assert result.metrics["events_matched"] == 1
    assert result.metrics["topology_segment_resets"] == 2
    assert result.metrics["max_drift_per_100_ticks"] == 0.0


def test_chain_length_change_starts_a_new_drift_segment() -> None:
    records = (
        GoldenTick(
            tick=0,
            frames=(),
            inputs=(),
            events=(),
            measurements={
                "chain_centers": _measurement([[10.0, 20.0]]),
                "chain_waypoints": _measurement([100.0]),
            },
        ),
        GoldenTick(
            tick=1,
            frames=(),
            inputs=(),
            events=(),
            measurements={
                "chain_centers": _measurement(
                    [[11.0, 20.0], [30.0, 40.0]]
                ),
                "chain_waypoints": _measurement([101.0, 140.0]),
            },
        ),
    )
    manifest, trace = _make_case(
        records,
        coverage=(
            _coverage("chain_centers", 1),
            _coverage("chain_waypoints", 1),
        ),
        center_tolerance=0.5,
        waypoint_tolerance=0.05,
    )
    simulator_ticks = (
        _sim_tick(
            0,
            (),
            {
                "chain_centers": [[10.0, 20.0]],
                "chain_waypoints": [100.0],
            },
        ),
        _sim_tick(
            1,
            (),
            {
                "chain_centers": [[11.5, 20.0], [30.5, 40.0]],
                "chain_waypoints": [101.0, 140.0],
            },
        ),
    )

    result = _compare_pc_trace(manifest, trace, simulator_ticks)

    assert result.status is ComparisonStatus.PASS
    assert result.metrics["max_drift_per_100_ticks"] == 0.0


def test_comparable_scalar_and_geometry_deviations_fail() -> None:
    records = (
        GoldenTick(
            tick=0,
            frames=(),
            inputs=(),
            events=(),
            measurements={
                "score": _measurement(10),
                "chain_centers": _measurement([[1.0, 2.0]]),
                "chain_waypoints": _measurement([25.0]),
            },
        ),
    )
    manifest, trace = _make_case(
        records,
        coverage=(
            _coverage("score", 0),
            _coverage("chain_centers", 0),
            _coverage("chain_waypoints", 0),
        ),
        center_tolerance=0.5,
        waypoint_tolerance=0.05,
    )
    simulator_ticks = (
        _sim_tick(
            0,
            (),
            {
                "score": 11,
                "chain_centers": [[1.0, 2.51]],
                "chain_waypoints": [25.051],
            },
        ),
    )

    result = _compare_pc_trace(manifest, trace, simulator_ticks)

    assert result.status is ComparisonStatus.FAIL
    assert result.metrics["mismatch_count"] == 3
    assert result.metrics["max_center_l2_error_px"] == pytest.approx(0.51)
    assert result.metrics["max_waypoint_abs_error"] == pytest.approx(0.051)
    assert all("synthetic_test" not in value for value in result.metrics)


def test_required_unavailable_or_missing_measurement_is_incomparable() -> None:
    records = (
        GoldenTick(
            tick=0,
            frames=(),
            inputs=(),
            events=(),
            measurements={
                "score": Measurement(
                    status=MeasurementStatus.OCCLUDED,
                    reason="synthetic occlusion",
                )
            },
        ),
        GoldenTick(
            tick=1,
            frames=(),
            inputs=(),
            events=(),
            measurements={},
        ),
    )
    manifest, trace = _make_case(
        records,
        coverage=(_coverage("score", 1),),
    )

    result = _compare_pc_trace(
        manifest,
        trace,
        (
            _sim_tick(0, (), {"score": 0}),
            _sim_tick(1, (), {"score": 0}),
        ),
    )

    assert result.status is ComparisonStatus.INCOMPARABLE
    assert result.metrics["incomparable_count"] == 2
    assert any("unavailable" in reason for reason in result.reasons)
    assert any("missing" in reason for reason in result.reasons)


def test_optional_complete_measurement_cannot_silently_pass_when_missing() -> None:
    records = (
        GoldenTick(
            tick=0,
            frames=(),
            inputs=(),
            events=(),
            measurements={"score": _measurement(0)},
        ),
    )
    manifest, trace = _make_case(
        records,
        coverage=(
            _coverage("score", 0),
            _coverage("chain_centers", 0, required=False),
        ),
    )

    result = _compare_pc_trace(
        manifest,
        trace,
        (_sim_tick(0, measurements={"score": 0}),),
    )

    assert result.status is ComparisonStatus.INCOMPARABLE
    assert result.metrics["incomparable_count"] == 1
    assert any(
        "'chain_centers' is missing" in reason
        for reason in result.reasons
    )


def test_required_projectile_centers_are_incomparable_without_identity() -> None:
    records = (
        GoldenTick(
            tick=0,
            frames=(),
            inputs=(),
            events=(),
            measurements={
                "projectile_centers": _measurement([[1.0, 2.0]])
            },
        ),
    )
    manifest, trace = _make_case(
        records,
        coverage=(_coverage("projectile_centers", 0),),
    )

    result = _compare_pc_trace(
        manifest,
        trace,
        (
            _sim_tick(
                0,
                measurements={"projectile_centers": [[1.0, 2.0]]},
            ),
        ),
    )

    assert result.status is ComparisonStatus.INCOMPARABLE
    assert result.metrics["measurement_comparisons"] == 0
    assert result.metrics["projectile_center_ranges_skipped"] == 1
    assert any("stable" in reason for reason in result.reasons)


def test_structural_channels_are_supported_without_scalar_comparison() -> None:
    records = (
        GoldenTick(
            tick=0,
            frames=(),
            inputs=(),
            events=(),
            measurements={"score": _measurement(0)},
        ),
    )
    manifest, trace = _make_case(
        records,
        coverage=(
            _coverage("frames", 0),
            _coverage("input_events", 0),
            _coverage("events", 0),
            _coverage("score", 0),
        ),
    )

    result = _compare_pc_trace(
        manifest,
        trace,
        (_sim_tick(0, measurements={"score": 0}),),
    )

    assert result.status is ComparisonStatus.PASS
    assert result.metrics["unsupported_optional_ranges"] == 0
    assert result.metrics["scalar_comparisons"] == 1


def test_central_value_contract_rejects_invalid_simulator_category() -> None:
    records = (
        GoldenTick(
            tick=0,
            frames=(),
            inputs=(),
            events=(),
            measurements={"gun_state": _measurement("normal")},
        ),
    )
    manifest, trace = _make_case(
        records,
        coverage=(_coverage("gun_state", 0),),
    )

    result = _compare_pc_trace(
        manifest,
        trace,
        (_sim_tick(0, measurements={"gun_state": "spinning"}),),
    )

    assert result.status is ComparisonStatus.INCOMPARABLE
    assert result.metrics["scalar_comparisons"] == 0
    assert any("invalid value" in reason for reason in result.reasons)


def test_readiness_and_unknown_required_channels_cannot_silently_pass() -> None:
    records = (
        GoldenTick(
            tick=0,
            frames=(),
            inputs=(),
            events=(),
            measurements={},
        ),
    )
    incomplete_manifest, incomplete_trace = _make_case(
        records,
        coverage=(
            _coverage(
                "future_state",
                0,
                status=CoverageStatus.PARTIAL,
            ),
        ),
    )
    readiness_result = _compare_pc_trace(
        incomplete_manifest,
        incomplete_trace,
        (_sim_tick(0, (), {"future_state": 1}),),
    )
    assert readiness_result.status is ComparisonStatus.INCOMPARABLE
    assert any("partial" in reason for reason in readiness_result.reasons)

    complete_manifest, complete_trace = _make_case(
        records,
        coverage=(_coverage("future_state", 0),),
    )
    unknown_result = _compare_pc_trace(
        complete_manifest,
        complete_trace,
        (_sim_tick(0, (), {"future_state": 1}),),
    )
    assert unknown_result.status is ComparisonStatus.INCOMPARABLE
    assert any(
        "unknown to PC golden" in reason
        for reason in unknown_result.reasons
    )


def test_unknown_event_sequence_uses_one_to_one_multiset_matching() -> None:
    one_event_records = (
        GoldenTick(
            tick=0,
            frames=(),
            inputs=(),
            events=(),
            measurements={},
        ),
        GoldenTick(
            tick=1,
            frames=(),
            inputs=(),
            events=(
                GoldenEvent(
                    kind="hit",
                    subjects=("private-tracker-id",),
                    payload={"private": "ignored"},
                    source="synthetic_test",
                ),
            ),
            measurements={},
        ),
        GoldenTick(
            tick=2,
            frames=(),
            inputs=(),
            events=(),
            measurements={},
        ),
    )
    manifest, trace = _make_case(
        one_event_records,
        coverage=(_coverage("events", 2),),
    )
    simulator_ticks = (
        _sim_tick(0),
        _sim_tick(1, ("hit",)),
        _sim_tick(2),
    )

    result = _compare_pc_trace(manifest, trace, simulator_ticks)

    assert result.status is ComparisonStatus.PASS
    assert result.metrics["events_matched"] == 1
    assert result.metrics["max_event_tick_error"] == 0
    assert "private-tracker-id" not in str(result.metrics)

    duplicate_event = GoldenEvent(
        kind="hit",
        subjects=(),
        payload={},
        source="synthetic_test",
    )
    two_event_records = (
        one_event_records[0],
        GoldenTick(
            tick=1,
            frames=(),
            inputs=(),
            events=(duplicate_event, duplicate_event),
            measurements={},
        ),
        one_event_records[2],
    )
    duplicate_manifest, duplicate_trace = _make_case(
        two_event_records,
        coverage=(_coverage("events", 2),),
    )
    duplicate_result = _compare_pc_trace(
        duplicate_manifest,
        duplicate_trace,
        simulator_ticks,
    )
    assert duplicate_result.status is ComparisonStatus.FAIL
    assert duplicate_result.metrics["events_matched"] == 1
    assert duplicate_result.metrics["mismatch_count"] == 1


def test_known_event_sequence_is_strict_but_unknown_order_is_not() -> None:
    known_events = (
        GoldenEvent(
            kind="shot_fired",
            subjects=(),
            payload={},
            source="synthetic_test",
            sequence=0,
        ),
        GoldenEvent(
            kind="hit",
            subjects=(),
            payload={},
            source="synthetic_test",
            sequence=1,
        ),
    )
    known_records = (
        GoldenTick(0, (), (), (), {}),
        GoldenTick(1, (), (), known_events, {}),
    )
    manifest, trace = _make_case(
        known_records,
        coverage=(_coverage("events", 1),),
    )
    reversed_ticks = (
        _sim_tick(0),
        _sim_tick(1, ("hit", "shot_fired")),
    )

    ordered_result = _compare_pc_trace(manifest, trace, reversed_ticks)

    assert ordered_result.status is ComparisonStatus.FAIL
    assert ordered_result.metrics["events_matched"] == 2
    assert any("sequence" in reason for reason in ordered_result.reasons)

    unknown_records = (
        known_records[0],
        GoldenTick(
            1,
            (),
            (),
            tuple(replace(event, sequence=None) for event in known_events),
            {},
        ),
    )
    unknown_manifest, unknown_trace = _make_case(
        unknown_records,
        coverage=(_coverage("events", 1),),
    )
    unknown_result = _compare_pc_trace(
        unknown_manifest,
        unknown_trace,
        reversed_ticks,
    )
    assert unknown_result.status is ComparisonStatus.PASS


def test_simulator_ticks_must_cover_zero_through_tick_end() -> None:
    records = (
        GoldenTick(0, (), (), (), {"score": _measurement(0)}),
        GoldenTick(1, (), (), (), {"score": _measurement(0)}),
    )
    manifest, trace = _make_case(
        records,
        coverage=(_coverage("score", 1),),
    )

    with pytest.raises(PcGoldenValidationError, match="tick 0 through tick_end"):
        _compare_pc_trace(
            manifest,
            trace,
            (_sim_tick(0, (), {"score": 0}),),
        )
    with pytest.raises(PcGoldenValidationError, match="expected 0, got 1"):
        _compare_pc_trace(
            manifest,
            trace,
            (
                _sim_tick(1, (), {"score": 0}),
                _sim_tick(2, (), {"score": 0}),
            ),
        )


def test_simulator_tick_requires_bound_source_tick_and_sample_phase() -> None:
    with pytest.raises(PcGoldenValidationError, match="source_tick"):
        SimulatorTick(
            tick=1,
            source_tick=0,
            sample_phase="post_update_presented",
        )
    with pytest.raises(PcGoldenValidationError, match="sample_phase"):
        SimulatorTick(
            tick=0,
            source_tick=0,
            sample_phase="pre_update",
        )


def test_unverified_public_comparison_is_rejected() -> None:
    records = (
        GoldenTick(
            tick=0,
            frames=(),
            inputs=(),
            events=(),
            measurements={"score": _measurement(0)},
        ),
    )
    manifest, trace = _make_case(
        records,
        coverage=(_coverage("score", 0),),
    )

    with pytest.raises(
        PcGoldenValidationError,
        match="compare_pc_golden_case",
    ):
        compare_pc_trace(
            manifest,
            trace,
            (_sim_tick(0, measurements={"score": 0}),),
        )


def test_safe_case_entry_fails_closed_before_consuming_ticks(tmp_path) -> None:
    consumed = False

    def simulator_ticks():
        nonlocal consumed
        consumed = True
        yield _sim_tick(0)

    result = compare_pc_golden_case(
        tmp_path / "missing-manifest.json",
        simulator_ticks(),
    )

    assert result.status is ComparisonStatus.FAIL
    assert result.metrics["verification_status"] == "FAIL"
    assert consumed is False


def test_snapshot_exposes_ordered_detached_simulator_channels() -> None:
    simulator = RevengeSimulator.__new__(RevengeSimulator)
    simulator.tick_count = 37
    simulator.score = 30
    simulator.current_color = 2
    simulator.next_color = 1
    simulator.gun_state = GunState.RELOADING
    simulator.outcome = None
    simulator.zuma_reached = True
    simulator.balls = [
        SimpleNamespace(waypoint=np.float32(10.5), center=(1.0, 2.0)),
        SimpleNamespace(waypoint=np.float32(20.5), center=(3.0, 4.0)),
    ]
    simulator.free_projectiles = [
        SimpleNamespace(position=np.array((5.0, 6.0), dtype=np.float32))
    ]
    simulator.merging_projectiles = [
        SimpleNamespace(position=np.array((7.0, 8.0), dtype=np.float32))
    ]
    simulator.ball_position = lambda ball: np.asarray(
        ball.center,
        dtype=np.float32,
    )

    snapshot = snapshot_revenge_simulator(
        simulator,
        events=("shot_fired",),
    )

    assert snapshot.tick == 37
    assert snapshot.source_tick == 37
    assert snapshot.sample_phase == "post_update_presented"
    assert snapshot.events == ("shot_fired",)
    assert dict(snapshot.measurements) == {
        "score": 30,
        "current_color": 2,
        "next_color": 1,
        "gun_state": "reloading",
        "outcome": None,
        "zuma_reached": True,
        "chain_count": 2,
        "projectile_count": 2,
        "chain_centers": ((1.0, 2.0), (3.0, 4.0)),
        "projectile_centers": ((5.0, 6.0), (7.0, 8.0)),
        "chain_waypoints": (10.5, 20.5),
    }
