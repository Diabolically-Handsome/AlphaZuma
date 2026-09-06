from tools import snapshot_alphazuma_55_weekend_v3 as subject


def test_matching_processes_accepts_v2_and_v3_postprocess_names() -> None:
    inventory = [
        {"pid": 1, "command": "python run_alphazuma_55_postprocess_parallel_v2.py"},
        {"pid": 2, "command": "python run_alphazuma_55_postprocess_parallel_v3.py"},
        {"pid": 3, "command": "python unrelated.py"},
    ]

    matches = subject._matching_processes(
        inventory, "run_alphazuma_55_postprocess_parallel_v2.py"
    )

    assert [row["pid"] for row in matches] == [1, 2]


def test_matching_processes_preserves_all_needle_semantics() -> None:
    inventory = [
        {"pid": 1, "command": "python worker.py --route expected.json"},
        {"pid": 2, "command": "python worker.py --route other.json"},
    ]

    matches = subject._matching_processes(inventory, "worker.py", "expected.json")

    assert [row["pid"] for row in matches] == [1]
