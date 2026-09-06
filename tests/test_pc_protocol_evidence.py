"""Tests for fail-closed PC golden protocol-v4 raw evidence."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from zuma_rl.pc_golden import PcGoldenValidationError
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

_NONCE = "0123456789abcdef0123456789abcdef"
_EXE_SHA = "sha256:" + "ab" * 32


def _snapshot(
    phase: str,
    perf_counter_ns: int,
    *,
    payload: bytes = b"save-state",
) -> PcStateSnapshot:
    return PcStateSnapshot(
        session_nonce=_NONCE,
        phase=phase,
        captured_perf_counter_ns=perf_counter_ns,
        files=(
            SnapshotFile(
                path="users.dat",
                data=payload,
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


def test_state_snapshot_round_trip_recomputes_root_and_rejects_tampering() -> None:
    snapshot = _snapshot("pre", 100)
    decoded = PcStateSnapshot.from_json(snapshot.to_json())

    assert decoded.state_root == snapshot.state_root
    assert decoded.users_bytes == len(b"save-state")

    tampered = json.loads(snapshot.to_json())
    tampered["users_files"][0]["data_base64"] = "dGFtcGVyZWQ="
    tampered_text = (
        json.dumps(
            tampered,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    with pytest.raises(PcGoldenValidationError, match="state_root"):
        PcStateSnapshot.from_json(tampered_text)

    with pytest.raises(PcGoldenValidationError, match="relative POSIX"):
        SnapshotFile(
            path="../users.dat",
            data=b"x",
            windows_attributes=0x20,
        )
    with pytest.raises(PcGoldenValidationError, match="reparse"):
        SnapshotFile(
            path="linked.dat",
            data=b"x",
            windows_attributes=0x400,
        )


def test_state_projection_excludes_only_declared_registry_values() -> None:
    base = _snapshot("r1-end", 100)
    first_node = base.registry_nodes[0]
    with_launcher_values = replace(
        base,
        registry_nodes=(
            replace(
                first_node,
                values=(
                    *first_node.values,
                    SnapshotRegistryValue(
                        name="LastGamePID",
                        value_type=4,
                        data=(10).to_bytes(4, "little"),
                    ),
                    SnapshotRegistryValue(
                        name="RegExData",
                        value_type=4,
                        data=(100).to_bytes(4, "little"),
                    ),
                ),
            ),
            base.registry_nodes[1],
        ),
    )
    changed_launcher_values = replace(
        with_launcher_values,
        registry_nodes=(
            replace(
                with_launcher_values.registry_nodes[0],
                values=(
                    with_launcher_values.registry_nodes[0].values[0],
                    SnapshotRegistryValue(
                        name="LastGamePID",
                        value_type=4,
                        data=(20).to_bytes(4, "little"),
                    ),
                    SnapshotRegistryValue(
                        name="RegExData",
                        value_type=4,
                        data=(200).to_bytes(4, "little"),
                    ),
                ),
            ),
            with_launcher_values.registry_nodes[1],
        ),
    )
    exclusions = (
        (REGISTRY_ROOTS[0], "", "LastGamePID"),
        (REGISTRY_ROOTS[0], "", "RegExData"),
    )

    assert (
        with_launcher_values.projected_state_root(exclusions)
        == changed_launcher_values.projected_state_root(exclusions)
    )
    assert (
        with_launcher_values.state_root
        != changed_launcher_values.state_root
    )
    changed_game_state = replace(
        changed_launcher_values,
        files=(
            replace(
                changed_launcher_values.files[0],
                data=b"different-save",
            ),
        ),
    )
    assert (
        with_launcher_values.projected_state_root(exclusions)
        != changed_game_state.projected_state_root(exclusions)
    )


def test_state_snapshot_rejects_windows_aliases_and_registry_scope() -> None:
    first = SnapshotFile(
        path="User.dat",
        data=b"a",
        windows_attributes=0x20,
    )
    second = SnapshotFile(
        path="user.dat",
        data=b"b",
        windows_attributes=0x20,
    )
    with pytest.raises(PcGoldenValidationError, match="duplicate Windows"):
        replace(_snapshot("pre", 100), files=(first, second))

    with pytest.raises(PcGoldenValidationError, match="fixed Zuma scope"):
        SnapshotRegistryNode(
            root=r"HKCU\Software\Unrelated",
            subkey="",
            exists=True,
            values=(),
        )


def test_save_journal_round_trip_recomputes_hash_chain() -> None:
    snapshots = (
        _snapshot("pre", 100),
        _snapshot("r1-start", 200),
        _snapshot("r1-end", 400, payload=b"end"),
        _snapshot("r2-start", 500),
        _snapshot("r2-end", 700, payload=b"end"),
        _snapshot("restored", 800),
    )
    process_fields = (
        (None, None, None),
        (10, 1000, None),
        (10, 1000, 0),
        (20, 2000, None),
        (20, 2000, 0),
        (None, None, None),
    )
    journal = PcSaveJournal.build(
        session_nonce=_NONCE,
        records=tuple(
            {
                "phase": snapshot.phase,
                "snapshot_artifact": f"state_{snapshot.phase}",
                "snapshot_state_root": snapshot.state_root,
                "snapshot_perf_counter_ns": (
                    snapshot.captured_perf_counter_ns
                ),
                "process_id": process_id,
                "process_creation_filetime_100ns": creation,
                "exit_code": exit_code,
            }
            for snapshot, (
                process_id,
                creation,
                exit_code,
            ) in zip(snapshots, process_fields, strict=True)
        ),
    )

    decoded = PcSaveJournal.from_ndjson(journal.to_ndjson())
    assert decoded == journal

    lines = journal.to_ndjson().splitlines()
    record = json.loads(lines[3])
    record["snapshot_state_root"] = "sha256:" + "00" * 32
    lines[3] = json.dumps(
        record,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    with pytest.raises(PcGoldenValidationError, match="hash chain"):
        PcSaveJournal.from_ndjson("\n".join(lines) + "\n")


def _process_timeline() -> PcProcessTimeline:
    return PcProcessTimeline(
        session_nonce=_NONCE,
        events=(
            PcProcessEvent(
                sequence=0,
                kind="start",
                process_id=10,
                process_creation_filetime_100ns=1000,
                perf_counter_ns=250,
                exit_code=None,
                executable_sha256=_EXE_SHA,
            ),
            PcProcessEvent(
                sequence=1,
                kind="stop",
                process_id=10,
                process_creation_filetime_100ns=1000,
                perf_counter_ns=350,
                exit_code=0,
                executable_sha256=_EXE_SHA,
            ),
            PcProcessEvent(
                sequence=2,
                kind="start",
                process_id=20,
                process_creation_filetime_100ns=2000,
                perf_counter_ns=550,
                exit_code=None,
                executable_sha256=_EXE_SHA,
            ),
            PcProcessEvent(
                sequence=3,
                kind="stop",
                process_id=20,
                process_creation_filetime_100ns=2000,
                perf_counter_ns=650,
                exit_code=0,
                executable_sha256=_EXE_SHA,
            ),
        ),
    )


def test_process_timeline_binary_round_trip_and_pairing() -> None:
    timeline = _process_timeline()
    payload = timeline.to_bytes()
    decoded = PcProcessTimeline.from_bytes(payload)

    assert decoded == timeline
    start, stop = decoded.event_pair((20, 2000))
    assert start.kind == "start"
    assert stop.kind == "stop"
    assert stop.exit_code == 0

    corrupted = bytearray(payload)
    corrupted[-1] ^= 1
    with pytest.raises(PcGoldenValidationError, match="identity changed"):
        PcProcessTimeline.from_bytes(bytes(corrupted))


def test_framework_update_map_round_trip_preserves_boundary_samples() -> None:
    update_map = PcFrameworkUpdateMap(
        capture_metadata_sha256="sha256:" + "01" * 32,
        process_id=10,
        process_creation_filetime_100ns=1000,
        executable_sha256=_EXE_SHA,
        records=(
            FrameworkUpdateRecord(
                sequence=0,
                present_ticks=100,
                host_perf_counter_ns=200,
                update_before=50,
                update_after=50,
            ),
            FrameworkUpdateRecord(
                sequence=1,
                present_ticks=110,
                host_perf_counter_ns=210,
                update_before=51,
                update_after=52,
            ),
        ),
    )

    decoded = PcFrameworkUpdateMap.from_json(update_map.to_json())
    assert decoded == update_map
    assert decoded.records[0].stable_update == 50
    assert decoded.records[1].stable_update is None


def _dxgi_metadata_text() -> str:
    data = {
        "schema": "zuma-rl.dxgi-bgra-capture",
        "version": 2,
        "status": "acquisition_complete",
        "target_process": "popcapgame1.exe",
        "device_index": 0,
        "output_index": 0,
        "global_region": [0, 0, 800, 600],
        "output_local_region": [0, 0, 800, 600],
        "width": 800,
        "height": 600,
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
        "capture_start_perf_counter_ns": 260,
        "capture_end_perf_counter_ns": 300,
        "capture_elapsed_seconds": 0.00000004,
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
            "process_id": 10,
            "process_creation_filetime_100ns": 1000,
            "executable_bytes": 10,
            "executable_sha256": _EXE_SHA,
            "executable_hash_provenance": {
                "method": "direct_file_sha256"
            },
            "window_handle_hex": "0x0000000000000001",
            "window_client_region": [0, 0, 800, 600],
        },
        "raw_bytes": 100,
        "raw_sha256": "sha256:" + "01" * 32,
        "frames_csv_bytes": 100,
        "frames_csv_rows": 3,
        "frames_csv_sha256": "sha256:" + "02" * 32,
        "aggregate_pixel_sha256": "sha256:" + "03" * 32,
        "frames_csv": "frames.csv",
        "raw_frames": "frames.bgra.raw",
    }
    return (
        json.dumps(
            data,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )


def test_dxgi_metadata_strictly_binds_process_and_capture_interval() -> None:
    metadata = PcDxgiCaptureMetadata.from_json(_dxgi_metadata_text())

    assert metadata.process_instance == (10, 1000)
    assert metadata.capture_start_perf_counter_ns == 260
    assert metadata.capture_end_perf_counter_ns == 300
    assert metadata.frame_count == 3

    changed = json.loads(_dxgi_metadata_text())
    changed["missed_presentations"] = 1
    text = (
        json.dumps(
            changed,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    with pytest.raises(PcGoldenValidationError, match="lossless"):
        PcDxgiCaptureMetadata.from_json(text)
