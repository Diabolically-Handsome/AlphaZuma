from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct

import pytest

from tools.popcap_mtrand_frame_schedule import (
    load_frame_mtrand_sync_schedule,
)
from zuma_rl.revenge_core import PopCapMTRandom


def _payload(rng: PopCapMTRandom) -> bytes:
    return struct.pack("<625I", *rng.words, rng.index)


def _row(update: int, native: int, rng: PopCapMTRandom) -> dict[str, object]:
    payload = _payload(rng)
    return {
        "type": "rng",
        "framework_update": update,
        "native_game_time": native,
        "global_mtrand_index": rng.index,
        "global_mtrand_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _write_log(path: Path, *, omit_update: int | None = None) -> None:
    rng = PopCapMTRandom(123)
    rows: list[dict[str, object]] = [
        {
            "schema": "zuma-rl.live-rng-change-monitor",
            "version": 1,
            "process_id": 77,
            "process_memory_writes": 0,
        }
    ]
    rows.append(_row(10, 100, rng))
    for update in range(11, 15):
        rng.next_u31()
        if update != omit_update:
            rows.append(_row(update, update + 90, rng))
        if update == 12:
            rng.next_u31()
            if update != omit_update:
                rows.append(_row(update, update + 90, rng))
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def test_load_frame_schedule_uses_last_prior_update_state(
    tmp_path: Path,
) -> None:
    path = tmp_path / "rng.ndjson"
    _write_log(path)

    schedule = load_frame_mtrand_sync_schedule(
        path,
        seed=123,
        start_update=11,
        end_update=14,
        maximum_draws=20,
    )

    assert schedule.source_process_id == 77
    assert len(schedule.entries) == 4
    assert schedule.entries[0].source_draw_count == 0
    assert schedule.entries[0].post_draw_count == 1
    assert schedule.entries[0].post_state_observed is True
    assert schedule.entries[2].source_draw_count == 3
    assert schedule.entries[2].post_draw_count == 4
    assert schedule.entries[0].post_state_sha256.startswith("sha256:")
    assert schedule.semantic_sha256.startswith("sha256:")


def test_frame_schedule_rejects_missing_update(tmp_path: Path) -> None:
    path = tmp_path / "rng.ndjson"
    _write_log(path, omit_update=12)

    with pytest.raises(ValueError, match="cover every requested update"):
        load_frame_mtrand_sync_schedule(
            path,
            seed=123,
            start_update=11,
            end_update=14,
            maximum_draws=20,
        )


def test_frame_schedule_rejects_unproven_first_call(
    tmp_path: Path,
) -> None:
    path = tmp_path / "rng.ndjson"
    _write_log(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines]
    rows = [
        row
        for row in rows
        if not (
            row.get("framework_update") == 13
            and row.get("type") == "rng"
        )
    ]
    # Keep coverage but substitute a state two calls beyond the prior
    # update. It proves neither "no call" nor the expected first call.
    rng = PopCapMTRandom(123)
    for _ in range(2):
        rng.next_u31()
    replacement = _row(13, 103, rng)
    replacement["framework_update"] = 13
    replacement["native_game_time"] = 103
    rows.append(replacement)
    rows.sort(key=lambda row: int(row.get("framework_update", -1)))
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="non-forward"):
        load_frame_mtrand_sync_schedule(
            path,
            seed=123,
            start_update=11,
            end_update=14,
            maximum_draws=20,
        )
