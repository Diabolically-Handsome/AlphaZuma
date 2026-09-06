from pathlib import Path

from tools import snapshot_alphazuma_55_weekend as snapshot


def test_read_json_accepts_windows_utf8_bom(tmp_path: Path) -> None:
    path = tmp_path / "guard-receipt.json"
    path.write_bytes(b"\xef\xbb\xbf{\"status\":\"RUNNING\"}")

    assert snapshot._read_json(path) == {"status": "RUNNING"}


def test_latest_checkpoint_prefers_largest_counter(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    (checkpoint_root / "run_1048576_steps.zip").write_bytes(b"a")
    (checkpoint_root / "run_2097152_steps.zip").write_bytes(b"b")
    (checkpoint_root / "unrelated.zip").write_bytes(b"c")

    value = snapshot._latest_checkpoint(tmp_path)

    assert value is not None
    assert value["steps"] == 2_097_152
    assert value["path"].endswith("run_2097152_steps.zip")


def test_evaluation_snapshot_reports_progress_and_eta(tmp_path: Path) -> None:
    root = tmp_path / "evaluation"
    root.mkdir()
    (root / "matrix-shard-00-of-02.json").write_text(
        """{
          "shard_index": 0,
          "status": "RUNNING",
          "completed_attempts": 20,
          "expected_attempts": 100,
          "runtime": {"wall_seconds": 10.0},
          "error": null
        }""",
        encoding="utf-8",
    )
    (root / "matrix-shard-01-of-02.json").write_text(
        """{
          "shard_index": 1,
          "status": "RUNNING",
          "completed_attempts": 30,
          "expected_attempts": 90,
          "runtime": {"wall_seconds": 15.0},
          "error": null
        }""",
        encoding="utf-8",
    )

    value = snapshot._evaluation_snapshot(root)

    assert value["status"] == "RUNNING"
    assert value["completed_attempts"] == 50
    assert value["expected_attempts"] == 190
    assert value["eta_seconds"] == 40.0


def test_matching_processes_requires_every_needle() -> None:
    inventory = [
        {"pid": 10, "command": "python train.py --run-id alpha"},
        {"pid": 11, "command": "python train.py --run-id beta"},
    ]

    assert snapshot._matching_processes(inventory, "train.py", "alpha") == [
        inventory[0]
    ]
