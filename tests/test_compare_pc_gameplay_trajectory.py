from __future__ import annotations

from types import SimpleNamespace
import json
from pathlib import Path

import pytest

import tools.compare_pc_gameplay_trajectory as gameplay_cli
from tools.compare_pc_gameplay_trajectory import _select_frames
from zuma_rl.pc_memory_trajectory import PcMemoryTrajectoryError


def test_select_frames_requires_an_exact_contiguous_window() -> None:
    frames = tuple(SimpleNamespace(update=update) for update in range(10, 16))

    selected = _select_frames(
        frames,
        start_update=11,
        end_update=14,
    )

    assert [frame.update for frame in selected] == [11, 12, 13, 14]

    with pytest.raises(
        PcMemoryTrajectoryError,
        match="gameplay_diff_window_incomplete",
    ):
        _select_frames(frames, start_update=9, end_update=14)

    with pytest.raises(
        PcMemoryTrajectoryError,
        match="gameplay_diff_window_invalid",
    ):
        _select_frames(frames, start_update=14, end_update=13)


def test_main_binds_raw_trajectory_and_dmo_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = tmp_path / "index.json"
    dmo = tmp_path / "input.dmo"
    original_root = tmp_path / "original"
    output = tmp_path / "report.json"
    index.write_bytes(b"trajectory-index\n")
    dmo.write_bytes(b"dmo\n")
    original_root.mkdir()
    frames = tuple(
        SimpleNamespace(update=update) for update in range(10, 13)
    )
    monkeypatch.setattr(
        gameplay_cli,
        "load_memory_trajectory",
        lambda path: frames,
    )
    monkeypatch.setattr(
        gameplay_cli.PopCapDemo,
        "read",
        lambda path: object(),
    )
    monkeypatch.setattr(
        gameplay_cli,
        "compare_gameplay_trajectory",
        lambda *args, **kwargs: {
            "schema": "zuma-rl.pc-gameplay-simulator-diff",
            "version": 1,
            "status": "PASS",
            "start_update": 10,
            "end_update": 12,
        },
    )

    result = gameplay_cli.main(
        [
            str(index),
            "--dmo",
            str(dmo),
            "--original-root",
            str(original_root),
            "--no-synchronize-shooter",
            "--output",
            str(output),
        ]
    )

    assert result == 0
    report = json.loads(output.read_text(encoding="ascii"))
    assert report["trajectory"] == {
        "artifact": str(index.resolve()),
        "artifact_sha256": gameplay_cli.sha256_path(index),
        "captured_start_update": 10,
        "captured_end_update": 12,
        "selected_start_update": 10,
        "selected_end_update": 12,
    }
    assert report["dmo"]["artifact_sha256"] == gameplay_cli.sha256_path(
        dmo
    )


def test_main_derives_source_proofs_from_a_bound_enclosing_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = tmp_path / "index.json"
    dmo = tmp_path / "input.dmo"
    original_root = tmp_path / "original"
    output = tmp_path / "report.json"
    index.write_bytes(b"trajectory-index\n")
    dmo.write_bytes(b"dmo\n")
    original_root.mkdir()
    all_frames = tuple(
        SimpleNamespace(update=update) for update in range(9, 15)
    )
    proof_windows: list[list[int]] = []
    monkeypatch.setattr(
        gameplay_cli,
        "load_memory_trajectory",
        lambda path: all_frames,
    )
    monkeypatch.setattr(
        gameplay_cli.PopCapDemo,
        "read",
        lambda path: object(),
    )
    monkeypatch.setattr(
        gameplay_cli,
        "compare_gameplay_trajectory",
        lambda frames, *args, **kwargs: {
            "schema": "zuma-rl.pc-gameplay-simulator-diff",
            "version": 5,
            "status": "PASS",
            "start_update": frames[0].update,
            "end_update": frames[-1].update,
        },
    )

    def derive(frames, **kwargs):
        proof_windows.append([frame.update for frame in frames])
        return (
            ("powerup_proximity_bomb",),
            (
                {
                    "feature": "powerup_proximity_bomb",
                    "status": "PASS",
                    "framework_update": 12,
                },
            ),
        )

    monkeypatch.setattr(
        gameplay_cli,
        "derive_native_mechanism_features",
        derive,
    )

    result = gameplay_cli.main(
        [
            str(index),
            "--dmo",
            str(dmo),
            "--original-root",
            str(original_root),
            "--start-update",
            "10",
            "--end-update",
            "12",
            "--source-feature-proofs",
            "--source-proof-start-update",
            "9",
            "--source-proof-end-update",
            "14",
            "--evidence-root",
            str(tmp_path),
            "--output",
            str(output),
        ]
    )

    assert result == 0
    report = json.loads(output.read_text(encoding="ascii"))
    assert proof_windows == [[9, 10, 11, 12, 13, 14]]
    assert report["trajectory"]["selected_start_update"] == 10
    assert report["trajectory"]["selected_end_update"] == 12
    assert report["trajectory"]["source_proof_start_update"] == 9
    assert report["trajectory"]["source_proof_end_update"] == 14
    assert report["source_authorized_features"] == [
        "powerup_proximity_bomb"
    ]
