from __future__ import annotations

import hashlib
from pathlib import Path
import struct

import pytest

from zuma_rl.pc_exact_step_evidence import (
    FORMAL_CLASSIFICATION,
    FORMAL_SAMPLE_PHASE,
    FORMAL_SNAPSHOT_SEMANTICS,
    canonical_json_bytes,
    compare_formal_exact_step_runs,
    load_formal_exact_step_run,
)
from zuma_rl.pc_memory_trajectory import STEPPED_SAMPLE_BARRIER
from zuma_rl.pc_external_input import (
    EXTERNAL_INPUT_GUARD_BINDING_SCHEMA,
    EXTERNAL_INPUT_GUARD_BINDING_VERSION,
    EXTERNAL_INPUT_GUARD_SCHEMA,
    EXTERNAL_INPUT_GUARD_VERSION,
    EXTERNAL_INPUT_POLICY,
    EXTERNAL_INPUT_PRIVACY,
    REPAINT_INPUT_EXTRA_INFO,
    SELF_TEST_INPUT_EXTRA_INFO,
)
from zuma_rl.pc_golden import ArtifactSpec, VideoMetadata
from zuma_rl.pc_video import inspect_pc_video_artifact
from tools.package_pc_exact_step_golden import (
    PackagingError,
    _encode_ffv1,
    _validated_run_ids,
)


RUNTIME_SHA256 = "sha256:" + "a" * 64
DMO_SHA256 = "sha256:" + "b" * 64


def _digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def _bmp(pixel: bytes) -> bytes:
    pixels = pixel * (800 * 600)
    file_size = 54 + len(pixels)
    return (
        struct.pack("<2sIHHI", b"BM", file_size, 0, 0, 54)
        + struct.pack(
            "<IiiHHIIiiII",
            40,
            800,
            -600,
            1,
            32,
            0,
            len(pixels),
            0,
            0,
            0,
            0,
        )
        + pixels
    )


def _snapshot(
    *,
    artifact: str,
    payload: bytes,
    process_id: int,
    window: str,
    update: int,
    warmup: bool,
    repaint_mechanism: str = (
        "nonclient_titlebar_drag_then_exact_origin_restore"
    ),
) -> dict[str, object]:
    result: dict[str, object] = {
        "artifact": artifact,
        "artifact_bytes": len(payload),
        "artifact_sha256": _digest(payload),
        "capture": {
            "process_id": process_id,
            "window_handle_hex": window,
            "client_region": [0, 0, 800, 600],
            "width": 800,
            "height": 600,
            "bytes": len(payload),
            "semantics": FORMAL_SNAPSHOT_SEMANTICS,
        },
        "repaint": {
            "schema": "zuma-rl.pc-window-repaint-handshake",
            "version": 1,
            "mechanism": repaint_mechanism,
            "restoration_mechanism": "set_window_pos_exact_origin",
            "geometry_restored": True,
            "process_id": process_id,
            "window_handle_hex": window,
            "outer_rect_before": [0, 0, 816, 639],
            "outer_rect_after": [0, 0, 816, 639],
            "started_perf_counter_ns": 12,
            "moved_perf_counter_ns": 13,
            "held_perf_counter_ns": 14,
            "restored_perf_counter_ns": 15,
            "finished_perf_counter_ns": 16,
        },
        "framework_update": update,
    }
    if repaint_mechanism == (
        "set_window_pos_temporary_translation_then_exact_origin_restore"
    ):
        repaint = result["repaint"]
        assert isinstance(repaint, dict)
        repaint.update(
            {
                "outer_rect_moved": [32, 0, 848, 639],
                "requested_temporary_delta": [32, 0],
                "actual_temporary_delta": [32, 0],
                "input_transport": "none_window_manager_api_only",
                "synthetic_input_event_count": 0,
                "dwm_flush_count": 2,
            }
        )
    if warmup:
        result["acceptance_role"] = "discarded_dxgi_surface_warmup"
    else:
        result["schema"] = "zuma-rl.pc-step-visual-snapshot"
        result["version"] = 1
    return result


def _build_run(
    root: Path,
    *,
    process_id: int,
    creation_time: int,
    retained_pixel: bytes = b"\x03\x02\x01\xff",
    repaint_mechanism: str = (
        "nonclient_titlebar_drag_then_exact_origin_restore"
    ),
    allowed_repaint_event_count: int = 24,
) -> tuple[Path, Path, Path]:
    attempt = root / "attempt-01"
    trajectory = attempt / "trajectory"
    frames = trajectory / "frames"
    frames.mkdir(parents=True)
    window = f"0x{process_id:016x}"
    identity = {
        "process_id": process_id,
        "process_creation_filetime_100ns": creation_time,
        "executable_bytes": 1234,
        "executable_sha256": RUNTIME_SHA256,
        "window_handle_hex": window,
        "window_client_region": [0, 0, 800, 600],
    }
    warmup_payload = _bmp(b"\x00\x00\x00\xff")
    retained_payload = _bmp(retained_pixel)
    (frames / "warmup-u00000100.bmp").write_bytes(warmup_payload)
    (frames / "u00000101.bmp").write_bytes(retained_payload)

    mtrand = struct.pack("<625I", *([0] * 625))
    (trajectory / "global-mtrand-frames.bin").write_bytes(mtrand)
    topology = "sha256:" + "0" * 64
    curve_lists = [
        {
            "curve_index": 0,
            "container_offset": offset,
            "declared_count": 0,
            "traversed_count": 0,
            "entities": [],
            "topology_sha256": topology,
        }
        for offset in (0x50, 0x5C, 0x68)
    ]
    warmup = _snapshot(
        artifact="frames/warmup-u00000100.bmp",
        payload=warmup_payload,
        process_id=process_id,
        window=window,
        update=100,
        warmup=True,
        repaint_mechanism=repaint_mechanism,
    )
    retained = _snapshot(
        artifact="frames/u00000101.bmp",
        payload=retained_payload,
        process_id=process_id,
        window=window,
        update=101,
        warmup=False,
        repaint_mechanism=repaint_mechanism,
    )
    index = {
        "schema": "zuma-rl.pc-rng-trajectory",
        "version": 2,
        "evidence_classification": FORMAL_CLASSIFICATION,
        "process_identity": identity,
        "sample_phase": FORMAL_SAMPLE_PHASE,
        "freeze_update": 100,
        "start_update": 101,
        "end_update": 101,
        "warmup_tick_count": 1,
        "tick_count": 1,
        "visual_snapshots": {
            "enabled": True,
            "frame_count": 1,
            "sample_phase": FORMAL_SAMPLE_PHASE,
            "repaint_handshake": f"{repaint_mechanism}_per_frame",
            "semantics": FORMAL_SNAPSHOT_SEMANTICS,
            "warmup_frame": warmup,
        },
        "window_transport": {
            "square_dwm_corners": {
                "schema": "zuma-rl.pc-dwm-corner-preference",
                "version": 1,
                "window_handle_hex": window,
                "attribute": "DWMWA_WINDOW_CORNER_PREFERENCE",
                "requested": "DWMWCP_DONOTROUND",
                "observed_value": 1,
                "status": "verified",
            }
        },
        "global_mtrand": {
            "artifact": "global-mtrand-frames.bin",
            "artifact_bytes": len(mtrand),
            "artifact_sha256": _digest(mtrand),
            "state_word_count": 624,
            "record_bytes": 2500,
        },
        "ticks": [
            {
                "framework_update": 101,
                "sample_phase": FORMAL_SAMPLE_PHASE,
                "sample_barrier": STEPPED_SAMPLE_BARRIER,
                "replay_state": {
                    "update_count": 101,
                    "frame_time_ms": 10,
                    "update_multiplier": 0.0,
                    "fast_forward_target": 101,
                    "fast_forward_to_marker": False,
                    "fast_forward_step": False,
                },
                "global_mtrand_record_offset": 0,
                "global_mtrand_record_bytes": len(mtrand),
                "global_mtrand_record_sha256": _digest(mtrand),
                "global_mtrand_index": 0,
                "score": 10,
                "displayed_score": 10,
                "score_target": 100,
                "board_color_counts": [0, 0, 0, 0, 0, 0],
                "chain_ball_count": 0,
                "pending_ball_count": 0,
                "inserting_ball_count": 0,
                "fired_bullet_count": 0,
                "current_ball": None,
                "next_ball": None,
                "qrand": {
                    "update_count": 0,
                    "selected_index": -1,
                    "vectors": {
                        "weights": [],
                        "sways": [],
                        "last_hit": [],
                        "previous_hit": [],
                    },
                },
                "thread_crt_rand_state": None,
                "curve_lists": curve_lists,
                "visual_snapshot": retained,
            }
        ],
    }
    index_path = trajectory / "index.json"
    _write_json(index_path, index)
    guard_receipt = {
        "schema": EXTERNAL_INPUT_GUARD_SCHEMA,
        "version": EXTERNAL_INPUT_GUARD_VERSION,
        "status": "PASS",
        "policy": EXTERNAL_INPUT_POLICY,
        "privacy": EXTERNAL_INPUT_PRIVACY,
        "coverage": {
            "start_perf_counter_ns": 11,
            "end_perf_counter_ns": 17,
        },
        "hooks": {
            "keyboard_installed": True,
            "mouse_installed": True,
            "keyboard_unhooked": True,
            "mouse_unhooked": True,
        },
        "self_tests": [
            {
                "label": label,
                "status": "PASS",
                "keyboard_event_count": 2,
                "mouse_event_count": 2,
            }
            for label in ("start", "end")
        ],
        "markers": {
            "repaint_input_extra_info_hex": (
                f"0x{REPAINT_INPUT_EXTRA_INFO:08x}"
            ),
            "self_test_input_extra_info_hex": (
                f"0x{SELF_TEST_INPUT_EXTRA_INFO:08x}"
            ),
        },
        "event_counts": {
            "external": 0,
            "allowed_repaint": allowed_repaint_event_count,
            "self_test": 8,
        },
        "external_events": [],
        "protocol_errors": [],
    }
    guard_path = attempt / "external-input-guard.json"
    _write_json(guard_path, guard_receipt)
    guard_sha256 = _digest(guard_path.read_bytes())
    probe = {
        "schema": "zuma-rl.pc-memory-int32-probe",
        "version": 1,
        "evidence_classification": FORMAL_CLASSIFICATION,
        "process_id": process_id,
        "process_identity": identity,
        "external_input_guard": {
            "schema": EXTERNAL_INPUT_GUARD_BINDING_SCHEMA,
            "version": EXTERNAL_INPUT_GUARD_BINDING_VERSION,
            "status": "PASS",
            "artifact": guard_path.name,
            "artifact_bytes": guard_path.stat().st_size,
            "artifact_sha256": guard_sha256,
            "coverage_start_perf_counter_ns": 11,
            "coverage_end_perf_counter_ns": 17,
            "external_event_count": 0,
            "allowed_repaint_event_count": allowed_repaint_event_count,
        },
        "runtime_executable_sha256": RUNTIME_SHA256,
        "dmo_sha256": DMO_SHA256,
        "diagnostic_mutation": None,
        "diagnostic_observation": None,
        "diagnostic_process_affinity": None,
        "global_rng_call_trace": None,
        "live_rng_monitor": None,
        "gameplay_mtrand_sync": None,
        "repaint": {
            "status": "SUPERSEDED_BY_PER_TICK_REPAINT_HANDSHAKE"
        },
        "frozen_frame": {
            "status": "SUPERSEDED_BY_RETAINED_EXACT_STEP_FRAMES"
        },
        "trajectory": {
            "artifact": "trajectory/index.json",
            "artifact_sha256": _digest(index_path.read_bytes()),
            "freeze_update": 100,
            "start_update": 101,
            "end_update": 101,
            "warmup_tick_count": 1,
            "tick_count": 1,
        },
    }
    probe_path = attempt / "memory-probe.json"
    _write_json(probe_path, probe)
    attempts = [
        {
            "attempt": 1,
            "status": "PASS",
            "process_id": process_id,
            "process_creation_filetime_100ns": creation_time,
            "started_perf_counter_ns": 10,
            "finished_perf_counter_ns": 20,
            "probe_sha256": _digest(probe_path.read_bytes()),
            "external_input_guard_sha256": guard_sha256,
            "external_input_guard_coverage_start_perf_counter_ns": 11,
            "external_input_guard_coverage_end_perf_counter_ns": 17,
        }
    ]
    attempts_path = root / "attempts.json"
    _write_json(attempts_path, attempts)
    return attempts_path, probe_path, index_path


def _load(run_id: str, paths: tuple[Path, Path, Path], pid: int, creation: int):
    return load_formal_exact_step_run(
        run_id=run_id,
        selected_attempt=1,
        attempts_path=paths[0],
        probe_path=paths[1],
        index_path=paths[2],
        expected_process_id=pid,
        expected_process_creation_filetime_100ns=creation,
        expected_runtime_sha256=RUNTIME_SHA256,
        expected_dmo_sha256=DMO_SHA256,
        expected_freeze_update=100,
        expected_start_update=101,
        expected_end_update=101,
        expected_warmup_tick_count=1,
        maximum_startup_attempts=3,
        declared_artifact_paths=(
            path for path in paths[0].parent.rglob("*") if path.is_file()
        ),
    )


def test_formal_exact_step_pair_recomputes_full_pass(tmp_path: Path) -> None:
    left_paths = _build_run(tmp_path / "left", process_id=101, creation_time=1001)
    right_paths = _build_run(tmp_path / "right", process_id=202, creation_time=2002)

    report = compare_formal_exact_step_runs(
        (_load("left", left_paths, 101, 1001), _load("right", right_paths, 202, 2002))
    )

    assert report["status"] == "PASS"
    assert report["independence"]["independent_process_instances"] is True
    assert report["pixels"]["full_800x600_bgra_bmp_exact"] is True
    assert report["memory"]["normalized_gameplay_and_rng_exact"] is True


def test_formal_exact_step_accepts_no_input_set_window_pos_repaint(
    tmp_path: Path,
) -> None:
    mechanism = (
        "set_window_pos_temporary_translation_then_exact_origin_restore"
    )
    paths = _build_run(
        tmp_path / "setpos",
        process_id=303,
        creation_time=3003,
        repaint_mechanism=mechanism,
        allowed_repaint_event_count=0,
    )

    loaded = _load("setpos", paths, 303, 3003)

    assert loaded.process_id == 303
    assert loaded.tick_rows[0]["visual_snapshot"]["repaint"][
        "synthetic_input_event_count"
    ] == 0


def test_formal_exact_step_pair_reports_real_pixel_difference(tmp_path: Path) -> None:
    left_paths = _build_run(tmp_path / "left", process_id=101, creation_time=1001)
    right_paths = _build_run(
        tmp_path / "right",
        process_id=202,
        creation_time=2002,
        retained_pixel=b"\x04\x02\x01\xff",
    )

    report = compare_formal_exact_step_runs(
        (_load("left", left_paths, 101, 1001), _load("right", right_paths, 202, 2002))
    )

    assert report["status"] == "FAIL"
    assert report["pixels"]["full_800x600_bgra_bmp_exact"] is False
    assert report["pixels"]["full_800x600_rgb24_exact"] is False


def test_exact_step_ffv1_wrapper_round_trips_source_pixels(tmp_path: Path) -> None:
    paths = _build_run(tmp_path / "source", process_id=101, creation_time=1001)
    loaded = _load("source", paths, 101, 1001)
    video_path = tmp_path / "source.avi"

    _encode_ffv1(video_path, loaded)

    metadata = VideoMetadata(
        artifact="video",
        width=800,
        height=600,
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
    )
    artifact = ArtifactSpec(
        path="source.avi",
        sha256=_digest(video_path.read_bytes()),
        bytes=video_path.stat().st_size,
    )
    inspection = inspect_pc_video_artifact(
        video_path,
        metadata=metadata,
        artifact=artifact,
    )

    assert inspection.frame_pts == (0,)
    assert inspection.frame_pixel_sha256 == (
        loaded.visual_frames[0].pc_video_rgb24_sha256,
    )


def test_exact_step_packager_accepts_fresh_distinct_run_ids() -> None:
    assert _validated_run_ids(("c89", "c90")) == ("c89", "c90")

    with pytest.raises(PackagingError, match="two distinct run ids"):
        _validated_run_ids(("c89", "c89"))
