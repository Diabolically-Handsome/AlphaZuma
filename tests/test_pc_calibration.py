from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from zuma_rl.pc_calibration import (
    CALIBRATION_SCHEMA,
    CALIBRATION_VERSION,
    CalibrationControlPoint,
    PcCalibrationSidecar,
)
from zuma_rl.pc_golden import (
    DMO_FILE_ID,
    DMO_FORMAT,
    DMO_VERSION,
    ArtifactSpec,
    ComparisonContract,
    CoordinateCalibration,
    CoverageRange,
    CoverageStatus,
    InputTimeline,
    PcEnvironment,
    PcGoldenManifest,
    PcGoldenValidationError,
    Scenario,
    TickClock,
    VideoMetadata,
)


def _digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _point_values(*, noise: float = 0.0) -> list[dict[str, list[float]]]:
    values: list[dict[str, list[float]]] = []
    for raw_x in (0.0, 800.0, 1598.0):
        for raw_y in (0.0, 600.0, 1198.0):
            logical_x = raw_x * 0.5
            logical_y = raw_y * 0.5
            if raw_x == 800.0 and raw_y == 600.0:
                logical_x += noise
            values.append(
                {
                    "raw": [raw_x, raw_y],
                    "logical": [logical_x, logical_y],
                }
            )
    return values


def _fitted_declarations(
    points: list[dict[str, list[float]]],
) -> tuple[list[list[float]], float, float]:
    raw = np.asarray([point["raw"] for point in points], dtype=np.float64)
    logical = np.asarray(
        [point["logical"] for point in points],
        dtype=np.float64,
    )
    x_fit = np.linalg.lstsq(
        np.column_stack((raw[:, 0], np.ones(len(raw)))),
        logical[:, 0],
        rcond=None,
    )[0]
    y_fit = np.linalg.lstsq(
        np.column_stack((raw[:, 1], np.ones(len(raw)))),
        logical[:, 1],
        rcond=None,
    )[0]
    matrix = [
        [float(x_fit[0]), 0.0, float(x_fit[1])],
        [0.0, float(y_fit[0]), float(y_fit[1])],
        [0.0, 0.0, 1.0],
    ]
    predicted = np.column_stack(
        (
            x_fit[0] * raw[:, 0] + x_fit[1],
            y_fit[0] * raw[:, 1] + y_fit[1],
        )
    )
    errors = np.linalg.norm(predicted - logical, axis=1)
    return (
        matrix,
        float(np.sqrt(np.mean(np.square(errors)))),
        float(np.max(errors)),
    )


def _sidecar(*, noise: float = 0.0) -> PcCalibrationSidecar:
    point_values = _point_values(noise=noise)
    matrix, rms, maximum = _fitted_declarations(point_values)
    return PcCalibrationSidecar(
        raw_width=1600,
        raw_height=1200,
        logical_width=800,
        logical_height=600,
        transform_kind="axis_aligned_affine",
        pixel_center_convention="center_at_integer",
        control_points=tuple(
            CalibrationControlPoint.from_dict(point)
            for point in point_values
        ),
        logical_from_raw=tuple(tuple(row) for row in matrix),
        rms_error_px=rms,
        max_error_px=maximum,
    )


def _environment() -> PcEnvironment:
    digest = "sha256:" + "1" * 64
    return PcEnvironment(
        executable_sha256=digest,
        main_pak_sha256=digest,
        levels_xml_sha256=digest,
        curve_sha256={"curve": digest},
        pre_capture_save_sha256=digest,
        profile_mode="tutorials_completed",
        renderer_api="Direct3D9",
        renderer_mode="windowed",
        ball_radius_branch=17,
        os_build="test",
        gpu="test",
        driver="test",
    )


def _manifest(
    sidecar: PcCalibrationSidecar,
    *,
    raw_width: int | None = None,
    raw_height: int | None = None,
    convention: str | None = None,
    matrix: tuple[tuple[float, float, float], ...] | None = None,
    rms_limit: float | None = None,
    max_limit: float | None = None,
) -> PcGoldenManifest:
    calibration_payload = sidecar.to_json().encode("utf-8")
    payloads = {
        "input": b"dmo",
        "video": b"video",
        "tick_map": b"tick,pts\n0,0\n",
        "calibration": calibration_payload,
        "trace": b"trace",
    }
    artifacts = {
        name: ArtifactSpec(
            path=f"{name}.bin",
            sha256=_digest(payload),
            bytes=len(payload),
        )
        for name, payload in payloads.items()
    }
    actual_raw_width = (
        sidecar.raw_width if raw_width is None else raw_width
    )
    actual_raw_height = (
        sidecar.raw_height if raw_height is None else raw_height
    )
    coordinates = CoordinateCalibration(
        raw_width=actual_raw_width,
        raw_height=actual_raw_height,
        logical_width=sidecar.logical_width,
        logical_height=sidecar.logical_height,
        transform_kind="axis_aligned_affine",
        pixel_center_convention=(
            sidecar.pixel_center_convention
            if convention is None
            else convention
        ),
        logical_from_raw=(
            sidecar.logical_from_raw if matrix is None else matrix
        ),
        rms_error_px=(
            sidecar.rms_error_px
            if rms_limit is None
            else rms_limit
        ),
        max_error_px=(
            sidecar.max_error_px
            if max_limit is None
            else max_limit
        ),
        calibration_artifact="calibration",
    )
    environment = _environment()
    return PcGoldenManifest(
        case_id="calibration_test",
        scenario=Scenario(
            level_id="temple-1",
            hard=False,
            curve_index=0,
            gun_index=0,
            mode="adventure",
            profile_mode="tutorials_completed",
        ),
        pc_environment=environment,
        pc_environment_fingerprint=environment.fingerprint,
        artifacts=artifacts,
        input_timeline=InputTimeline(
            artifact="input",
            format=DMO_FORMAT,
            file_id=DMO_FILE_ID,
            dmo_version=DMO_VERSION,
            product_version="test",
            random_seed=1,
            length_updates=0,
            native_tick_offset=0,
        ),
        trace_artifact="trace",
        video=VideoMetadata(
            artifact="video",
            width=actual_raw_width,
            height=actual_raw_height,
            codec="ffv1",
            pixel_format="bgra",
            time_base=(1, 100),
            nominal_fps=(100, 1),
            frame_count=1,
            first_pts=0,
            last_pts=0,
            cfr=True,
            dropped_frames=0,
            duplicate_frames=0,
        ),
        clock=TickClock(
            logic_hz=(100, 1),
            tick_start=0,
            tick_end=0,
            tick0_video_pts=0,
            sample_phase="post_update_presented",
            mapping_kind="per_tick_pts_table",
            tick_map_artifact="tick_map",
            uncertainty_ticks=0.0,
        ),
        coordinates=coordinates,
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


def _canonical(value: dict[str, Any]) -> str:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )


def test_canonical_round_trip_read_and_manifest_validation_are_read_only(
    tmp_path: Path,
) -> None:
    sidecar = _sidecar()
    manifest = _manifest(sidecar)
    path = tmp_path / "private-capture-coordinates.json"
    path.write_bytes(sidecar.to_json().encode("utf-8"))
    before = (path.read_bytes(), path.stat().st_mtime_ns)

    parsed = PcCalibrationSidecar.read(path)
    fit = parsed.validate_against(manifest)

    assert parsed == sidecar
    assert parsed.to_json().endswith("\n")
    assert parsed.to_dict()["schema"] == CALIBRATION_SCHEMA
    assert parsed.to_dict()["version"] == CALIBRATION_VERSION
    assert np.allclose(fit.matrix, sidecar.matrix, rtol=0.0, atol=1e-9)
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before


@pytest.mark.parametrize(
    "mutate",
    [
        lambda text: text[:-1],
        lambda text: text.replace("\n", "\r\n"),
        lambda text: json.dumps(json.loads(text), indent=2) + "\n",
        lambda text: text.replace("e-13", "e-013", 1),
    ],
)
def test_noncanonical_json_is_rejected(mutate) -> None:
    text = _sidecar().to_json()
    changed = mutate(text)
    assert changed != text
    with pytest.raises(PcGoldenValidationError, match="canonical"):
        PcCalibrationSidecar.from_json(changed)


def test_duplicate_unknown_and_nonfinite_json_fields_are_rejected() -> None:
    sidecar = _sidecar()
    value = sidecar.to_dict()
    value["unknown"] = True
    with pytest.raises(PcGoldenValidationError, match="unknown fields"):
        PcCalibrationSidecar.from_json(_canonical(value))

    point_unknown = sidecar.to_dict()
    point_unknown["control_points"][0]["unknown"] = True
    with pytest.raises(PcGoldenValidationError, match="unknown fields"):
        PcCalibrationSidecar.from_json(_canonical(point_unknown))

    duplicate = sidecar.to_json().replace(
        '{"control_points"',
        '{"version":1,"control_points"',
        1,
    )
    with pytest.raises(PcGoldenValidationError, match="strict JSON"):
        PcCalibrationSidecar.from_json(duplicate)

    nonfinite_value = sidecar.to_dict()
    nonfinite_value["rms_error_px"] = float("nan")
    nonfinite = (
        json.dumps(
            nonfinite_value,
            ensure_ascii=False,
            allow_nan=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    with pytest.raises(PcGoldenValidationError, match="strict JSON"):
        PcCalibrationSidecar.from_json(nonfinite)


def test_control_points_reject_insufficient_duplicate_and_degenerate_sets() -> None:
    sidecar = _sidecar()

    insufficient = sidecar.to_dict()
    insufficient["control_points"] = insufficient["control_points"][:8]
    with pytest.raises(PcGoldenValidationError, match="at least nine"):
        PcCalibrationSidecar.from_dict(insufficient)

    duplicate_raw = sidecar.to_dict()
    duplicate_raw["control_points"][1]["raw"] = list(
        duplicate_raw["control_points"][0]["raw"]
    )
    with pytest.raises(PcGoldenValidationError, match="duplicate raw"):
        PcCalibrationSidecar.from_dict(duplicate_raw)

    duplicate_logical = sidecar.to_dict()
    duplicate_logical["control_points"][1]["logical"] = list(
        duplicate_logical["control_points"][0]["logical"]
    )
    with pytest.raises(PcGoldenValidationError, match="duplicate logical"):
        PcCalibrationSidecar.from_dict(duplicate_logical)

    degenerate_points = tuple(
        CalibrationControlPoint(
            raw=(float(index % 2), float(index * 100)),
            logical=(float(index % 2), float(index * 50)),
        )
        for index in range(9)
    )
    with pytest.raises(PcGoldenValidationError, match="degenerate"):
        replace(
            sidecar,
            control_points=tuple(
                sorted(degenerate_points, key=lambda point: point.raw)
            ),
        )


def test_nonfinite_out_of_bounds_and_noncanonical_point_order_are_rejected() -> None:
    with pytest.raises(PcGoldenValidationError, match="finite"):
        CalibrationControlPoint(raw=(float("nan"), 0.0), logical=(0.0, 0.0))

    sidecar = _sidecar()
    outside = sidecar.to_dict()
    outside["control_points"][-1]["raw"][0] = 2000
    with pytest.raises(PcGoldenValidationError, match="outside"):
        PcCalibrationSidecar.from_dict(outside)

    with pytest.raises(PcGoldenValidationError, match="ordered canonically"):
        replace(
            sidecar,
            control_points=tuple(reversed(sidecar.control_points)),
        )


def test_declared_matrix_and_errors_must_be_reproducible() -> None:
    sidecar = _sidecar()

    wrong_matrix = sidecar.to_dict()
    wrong_matrix["logical_from_raw"][0][0] = 0.6
    with pytest.raises(PcGoldenValidationError, match="matrix"):
        PcCalibrationSidecar.from_dict(wrong_matrix)

    wrong_rms = sidecar.to_dict()
    wrong_rms["rms_error_px"] = 0.1
    wrong_rms["max_error_px"] = 0.1
    with pytest.raises(PcGoldenValidationError, match="RMS"):
        PcCalibrationSidecar.from_dict(wrong_rms)

    wrong_maximum = sidecar.to_dict()
    wrong_maximum["max_error_px"] = 0.1
    with pytest.raises(PcGoldenValidationError, match="maximum"):
        PcCalibrationSidecar.from_dict(wrong_maximum)


def test_validate_against_rejects_manifest_semantic_mismatches() -> None:
    sidecar = _sidecar()

    with pytest.raises(PcGoldenValidationError, match="dimensions"):
        sidecar.validate_against(_manifest(sidecar, raw_width=1599))

    with pytest.raises(PcGoldenValidationError, match="conventions"):
        sidecar.validate_against(
            _manifest(sidecar, convention="center_at_half")
        )

    different_matrix = (
        (0.500001, 0.0, 0.0),
        (0.0, 0.5, 0.0),
        (0.0, 0.0, 1.0),
    )
    with pytest.raises(PcGoldenValidationError, match="matrix"):
        sidecar.validate_against(
            _manifest(sidecar, matrix=different_matrix)
        )


def test_recomputed_errors_must_exactly_match_manifest_declarations() -> None:
    sidecar = _sidecar(noise=0.25)
    manifest = _manifest(
        sidecar,
        rms_limit=0.0,
        max_limit=0.0,
    )

    with pytest.raises(PcGoldenValidationError, match="do not match"):
        sidecar.validate_against(manifest)


def test_calibration_artifact_identity_is_bound_to_manifest() -> None:
    sidecar = _sidecar()
    manifest = _manifest(sidecar)
    artifacts = dict(manifest.artifacts)
    artifacts["calibration"] = replace(
        artifacts["calibration"],
        sha256="sha256:" + "0" * 64,
    )
    changed = replace(manifest, artifacts=artifacts)

    with pytest.raises(PcGoldenValidationError, match="artifact identity"):
        sidecar.validate_against(changed)


def test_read_errors_never_echo_path(tmp_path: Path) -> None:
    sensitive = tmp_path / "SECRET-CONTROL-POINTS-CASE.json"

    with pytest.raises(PcGoldenValidationError) as captured:
        PcCalibrationSidecar.read(sensitive)

    assert str(sensitive) not in str(captured.value)
    assert sensitive.name not in str(captured.value)
