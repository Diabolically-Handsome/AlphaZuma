from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tools.compare_live_zuma_rng_monitors import (
    MonitorComparisonError,
    compare_monitors,
)


def _rng_row(
    update: int,
    *,
    mt_index: int = 10,
    mt_sha: str = "a" * 64,
    crt: int = 20,
    qrand_update: int = 3,
    score: int = 100,
    address: int = 0x1000,
) -> dict[str, Any]:
    return {
        "type": "rng",
        "perf_counter_ns": update * 100,
        "framework_update": update,
        "native_game_time": update - 100,
        "score": score,
        "displayed_score": score,
        "score_target": 200,
        "board_color_counts": [1, 0, 0, 0, 0, 0],
        "chain_ball_count": 1,
        "pending_ball_count": 2,
        "inserting_ball_count": 0,
        "fired_bullet_count": 0,
        "chain": [{"ball_id": 7, "color_id": 0}],
        "current_ball": {
            "address": address,
            "address_hex": hex(address),
            "ball_id": 8,
            "color_id": 1,
            "kind": "bullet",
            "powerup_previous_type": 14,
            "powerup_primary_type": 14,
            "powerup_secondary_type": 14,
            "vtable": 0x1234,
        },
        "next_ball": None,
        "qrand": {
            "address": address + 4,
            "update_count": qrand_update,
            "selected_index": 0,
            "vectors": {
                "weights": [1.0, 0.0],
                "sways": [1.0, 0.0],
                "last_hit": [qrand_update, 0],
                "previous_hit": [qrand_update - 1, 0],
            },
        },
        "thread_crt_rand_state": crt,
        "global_mtrand_index": mt_index,
        "global_mtrand_sha256": mt_sha,
    }


def _write_monitor(path: Path, rows: list[dict[str, Any]]) -> None:
    header = {
        "schema": "zuma-rl.live-rng-change-monitor",
        "version": 1,
        "classification": "read-only-localization-diagnostic",
        "process_id": 123,
        "process_memory_writes": 0,
    }
    path.write_text(
        "\n".join(
            json.dumps(value, sort_keys=True)
            for value in [header, *rows]
        )
        + "\n",
        encoding="utf-8",
    )


def test_comparison_ignores_process_addresses_and_sampling_time(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left.ndjson"
    right = tmp_path / "right.ndjson"
    _write_monitor(left, [_rng_row(101, address=0x1000)])
    row = _rng_row(101, address=0x9000)
    row["perf_counter_ns"] = 999999
    _write_monitor(right, [row])

    report = compare_monitors(left, right)

    for component in report["components"].values():
        assert component["shared_common_update_count"] == 1
        assert component["terminal_divergence_observed"] is False


def test_comparison_uses_any_same_update_state_and_terminal_bound(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left.ndjson"
    right = tmp_path / "right.ndjson"
    left_rows = [
        _rng_row(10, mt_index=10),
        _rng_row(11, mt_index=11),
        _rng_row(12, mt_index=12),
        _rng_row(13, mt_index=13),
        _rng_row(14, mt_index=14),
    ]
    right_rows = [
        _rng_row(10, mt_index=10),
        _rng_row(11, mt_index=99),
        _rng_row(12, mt_index=98),
        _rng_row(12, mt_index=12, address=0x9000),
        _rng_row(13, mt_index=97),
        _rng_row(14, mt_index=96),
    ]
    _write_monitor(left, left_rows)
    _write_monitor(right, right_rows)

    report = compare_monitors(left, right)
    mt = report["components"]["global_mtrand"]

    assert mt["first_observed_disjoint_update"] == 11
    assert mt["last_observed_shared_update"] == 12
    assert mt["terminal_divergence_observed"] is True
    assert mt["terminal_divergence_bound"][
        "first_later_common_disjoint_update"
    ] == 13
    assert mt["terminal_divergence_bound"][
        "all_later_common_updates_disjoint"
    ] is True
    assert mt["terminal_divergence_bound"][
        "later_common_disjoint_update_count"
    ] == 2


def test_comparison_rejects_non_read_only_monitor(tmp_path: Path) -> None:
    left = tmp_path / "left.ndjson"
    right = tmp_path / "right.ndjson"
    _write_monitor(left, [_rng_row(10)])
    _write_monitor(right, [_rng_row(10, mt_index=11)])
    lines = left.read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])
    header["process_memory_writes"] = 1
    lines[0] = json.dumps(header)
    left.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(
        MonitorComparisonError,
        match="monitor_header_contract_invalid",
    ):
        compare_monitors(left, right)


def test_comparison_rejects_no_common_updates(tmp_path: Path) -> None:
    left = tmp_path / "left.ndjson"
    right = tmp_path / "right.ndjson"
    _write_monitor(left, [_rng_row(10)])
    _write_monitor(right, [_rng_row(11)])

    with pytest.raises(
        MonitorComparisonError,
        match="monitors_have_no_common_updates",
    ):
        compare_monitors(left, right)
