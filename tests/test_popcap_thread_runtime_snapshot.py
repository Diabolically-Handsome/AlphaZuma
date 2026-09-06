from __future__ import annotations

from tools.popcap_thread_runtime_snapshot import compute_thread_runtime_delta


def _snapshot(
    *,
    captured: int,
    rows: list[dict[str, int | bool]],
) -> dict[str, object]:
    return {
        "process_id": 10,
        "main_thread_id": 100,
        "captured_perf_counter_ns": captured,
        "threads": rows,
    }


def test_thread_runtime_delta_tracks_lifecycle_and_groups_start_address() -> None:
    ready = _snapshot(
        captured=1_000,
        rows=[
            {
                "thread_id": 100,
                "is_main_thread": True,
                "start_address": 0x1000,
                "kernel_time_100ns": 10,
                "user_time_100ns": 20,
                "cycle_time": 30,
            },
            {
                "thread_id": 200,
                "is_main_thread": False,
                "start_address": 0x2000,
                "kernel_time_100ns": 40,
                "user_time_100ns": 50,
                "cycle_time": 60,
            },
            {
                "thread_id": 300,
                "is_main_thread": False,
                "start_address": 0x2000,
                "kernel_time_100ns": 70,
                "user_time_100ns": 80,
                "cycle_time": 90,
            },
        ],
    )
    boundary = _snapshot(
        captured=3_000,
        rows=[
            {
                "thread_id": 100,
                "is_main_thread": True,
                "start_address": 0x1000,
                "kernel_time_100ns": 15,
                "user_time_100ns": 27,
                "cycle_time": 41,
            },
            {
                "thread_id": 200,
                "is_main_thread": False,
                "start_address": 0x2000,
                "kernel_time_100ns": 44,
                "user_time_100ns": 56,
                "cycle_time": 68,
            },
            {
                "thread_id": 400,
                "is_main_thread": False,
                "start_address": 0x3000,
                "kernel_time_100ns": 1,
                "user_time_100ns": 2,
                "cycle_time": 3,
            },
        ],
    )

    result = compute_thread_runtime_delta(ready, boundary)

    assert result["elapsed_perf_counter_ns"] == 2_000
    assert result["persistent_thread_count"] == 2
    assert result["created_thread_count"] == 1
    assert result["exited_thread_count"] == 1
    by_thread = {row["thread_id"]: row for row in result["threads"]}
    assert by_thread[100]["cycle_time_delta"] == 11
    assert by_thread[200]["user_time_100ns_delta"] == 6
    assert by_thread[300]["present_at_boundary"] is False
    assert by_thread[400]["present_at_ready"] is False
    assert by_thread[400]["cycle_time_delta"] == 3
    groups = {
        row["start_address"]: row
        for row in result["aggregate_by_start_address"]
    }
    assert groups[0x2000]["persistent_count"] == 1
    assert groups[0x2000]["exited_count"] == 1
    assert groups[0x2000]["cycle_time_delta"] == 8
    assert groups[0x3000]["cycle_time_delta"] == 3


def test_thread_runtime_delta_rejects_identity_mismatch() -> None:
    ready = _snapshot(captured=1, rows=[])
    boundary = _snapshot(captured=2, rows=[])
    boundary["process_id"] = 11

    try:
        compute_thread_runtime_delta(ready, boundary)
    except ValueError as error:
        assert str(error) == "thread runtime snapshot identity mismatch"
    else:
        raise AssertionError("identity mismatch must fail")
