from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from tools import analyze_pc_capture_draw_alignment as alignment


_RUNTIME_SHA = "sha256:" + "ab" * 32


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")


def _write_bytes(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: object) -> str:
    return _write_bytes(path, _canonical_bytes(value))


def _snapshot(update: int, draw: int) -> dict[str, int]:
    return {"update_count": update, "draw_count": draw}


def _make_run(
    root: Path,
    *,
    pid: int,
    creation: int,
    hashes: list[str],
    updates: list[tuple[int, int]],
    draws: list[tuple[int, int]],
) -> None:
    run_root = root / f"run-r{pid}"
    csv_path = run_root / "capture" / "frames.csv"
    csv_path.parent.mkdir(parents=True)
    with csv_path.open("w", encoding="ascii", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "sequence",
                "present_ticks",
                "host_perf_counter_ns",
                "frame_sha256",
            ],
        )
        writer.writeheader()
        for sequence, digest in enumerate(hashes):
            writer.writerow(
                {
                    "sequence": sequence,
                    "present_ticks": 1000 + sequence,
                    "host_perf_counter_ns": 100000 + sequence,
                    "frame_sha256": digest,
                }
            )
    csv_sha = "sha256:" + hashlib.sha256(csv_path.read_bytes()).hexdigest()
    metadata = {
        "schema": "zuma-rl.dxgi-bgra-capture",
        "status": "acquisition_complete",
        "frame_count": len(hashes),
        "width": 800,
        "height": 600,
        "frames_csv_sha256": csv_sha,
        "target_identity": {
            "process_id": pid,
            "process_creation_filetime_100ns": creation,
            "executable_sha256": _RUNTIME_SHA,
        },
    }
    metadata_sha = _write_json(run_root / "capture" / "metadata.json", metadata)
    update_records = []
    state_records = []
    for sequence, ((update_before, update_after), (draw_before, draw_after)) in enumerate(
        zip(updates, draws, strict=True)
    ):
        common = {
            "sequence": sequence,
            "present_ticks": 1000 + sequence,
            "host_perf_counter_ns": 100000 + sequence,
        }
        update_records.append(
            {
                **common,
                "update_before": update_before,
                "update_after": update_after,
            }
        )
        state_records.append(
            {
                **common,
                "before": _snapshot(update_before, draw_before),
                "after": _snapshot(update_after, draw_after),
            }
        )
    update_map = {
        "schema": "zuma-rl.pc-framework-update-map",
        "version": 1,
        "records": update_records,
    }
    update_sha = _write_json(run_root / "framework-updates.json", update_map)
    state = {
        "schema": "zuma-rl.pc-framework-state-diagnostic",
        "version": 1,
        "diagnostic_only": True,
        "capture_metadata_sha256": metadata_sha,
        "framework_update_map_sha256": update_sha,
        "process_id": pid,
        "process_creation_filetime_100ns": creation,
        "executable_sha256": _RUNTIME_SHA,
        "records": state_records,
    }
    _write_json(run_root / "framework-state-diagnostic.json", state)


def _digest(byte: int) -> str:
    return "sha256:" + f"{byte:02x}" * 32


def _session(tmp_path: Path) -> Path:
    root = tmp_path / "session"
    _make_run(
        root,
        pid=1,
        creation=101,
        hashes=[
            _digest(1),
            _digest(1),
            _digest(2),
            _digest(3),
            _digest(4),
            _digest(5),
        ],
        updates=[
            (10, 10),
            (10, 10),
            (11, 11),
            (12, 12),
            (13, 13),
            (14, 14),
        ],
        draws=[
            (20, 20),
            (20, 21),
            (21, 21),
            (22, 22),
            (23, 23),
            (24, 24),
        ],
    )
    _make_run(
        root,
        pid=2,
        creation=202,
        hashes=[_digest(1), _digest(2), _digest(3), _digest(4), _digest(9)],
        updates=[(10, 10), (11, 11), (13, 13), (14, 14), (15, 15)],
        draws=[(15, 15), (16, 16), (17, 17), (18, 18), (19, 19)],
    )
    return root


def test_analyze_reports_modal_draw_alignment_without_certifying(
    tmp_path: Path,
) -> None:
    report = alignment.analyze(_session(tmp_path))

    assert report["classification"] == "diagnostic-only-not-pc-golden"
    assert report["gate_effect"] == "none"
    assert report["status"] == "OBSERVED_DRAW_ALIGNMENT_INCOMPLETE"
    ordered = report["ordered_present_visuals"]
    assert ordered["modal_draw_after_delta"] == 5
    assert ordered["matched_collapsed_visual_count"] == 4
    assert ordered["modal_draw_after_delta_support"] == 4
    baseline = report["same_stable_update_baseline"]
    assert baseline["common_stable_update_count"] == 4
    assert baseline["exact_hash_overlap_count"] == 2
    draw = report["normalized_draw_candidate_alignment"]
    assert draw["paired_draw_count"] == 5
    assert draw["exact_hash_overlap_count"] == 4
    assert draw["unresolved_pair_count"] == 1


def test_analyze_rejects_broken_state_binding(tmp_path: Path) -> None:
    root = _session(tmp_path)
    state_path = root / "run-r1" / "framework-state-diagnostic.json"
    state = json.loads(state_path.read_text(encoding="ascii"))
    state["capture_metadata_sha256"] = "sha256:" + "00" * 32
    state_path.write_bytes(_canonical_bytes(state))

    with pytest.raises(
        alignment.DrawAlignmentError,
        match="draw_alignment_state_binding_mismatch",
    ):
        alignment.analyze(root)


def test_main_writes_canonical_output_exclusively(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _session(tmp_path)
    output = tmp_path / "report.json"

    assert alignment.main(
        ["--session-root", str(root), "--output", str(output)]
    ) == 0
    payload = output.read_bytes()
    assert payload.endswith(b"\n")
    assert json.loads(payload)["schema"] == alignment.SCHEMA
    assert "sha256:" in capsys.readouterr().out

    assert alignment.main(
        ["--session-root", str(root), "--output", str(output)]
    ) == 2
