from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct

import pytest

from tools.popcap_mtrand_call_oracle import (
    EXPECTED_RETAIL_RUNTIME_SHA256,
    load_gameplay_mtrand_oracle,
)
from zuma_rl.revenge_core import PopCapMTRandom


def _payload(rng: PopCapMTRandom) -> bytes:
    return struct.pack("<625I", *rng.words, rng.index)


def _trace(path: Path, *, bad_output: bool = False) -> None:
    rng = PopCapMTRandom(1234)
    calls = []
    for order, ambient_draws in enumerate((2, 0, 3)):
        for _ in range(ambient_draws):
            rng.next_u31()
        output = rng.next_u31()
        state = _payload(rng)
        calls.append(
            {
                "order": order,
                "caller": 0x004B5ADF,
                "framework_update": 100 + order,
                "native_game_time": 20 + order,
                "score": 8000,
                "score_target": 9650,
                "output": (
                    output ^ 1
                    if bad_output and order == 1
                    else output
                ),
                "post_index": rng.index,
                "post_state_sha256": (
                    "sha256:" + hashlib.sha256(state).hexdigest()
                ),
            }
        )
    path.write_text(
        json.dumps(
            {
                "schema": "zuma-rl.pc-gameplay-mtrand-call-trace",
                "version": 1,
                "status": "PASS",
                "failure": None,
                "process_id": 42,
                "runtime_executable_sha256": (
                    EXPECTED_RETAIL_RUNTIME_SHA256
                ),
                "process_memory_writes": 0,
                "persistent_file_modified": False,
                "hardware_breakpoint_restored": True,
                "call_count": len(calls),
                "calls": calls,
            },
            sort_keys=True,
        ),
        encoding="ascii",
    )


def test_load_gameplay_mtrand_oracle_reconstructs_exact_states(
    tmp_path: Path,
) -> None:
    path = tmp_path / "oracle.json"
    _trace(path)

    oracle = load_gameplay_mtrand_oracle(
        path,
        seed=1234,
        maximum_draws=20,
    )

    assert [entry.post_draw_count for entry in oracle.entries] == [
        3,
        4,
        8,
    ]
    assert {entry.caller for entry in oracle.entries} == {0x004B5ADF}
    assert all(
        hashlib.sha256(entry.post_payload).hexdigest()
        == entry.post_state_sha256[7:]
        for entry in oracle.entries
    )
    assert oracle.semantic_sha256.startswith("sha256:")


def test_load_gameplay_mtrand_oracle_rejects_output_mismatch(
    tmp_path: Path,
) -> None:
    path = tmp_path / "oracle.json"
    _trace(path, bad_output=True)

    with pytest.raises(
        ValueError,
        match="seeded reconstruction mismatch",
    ):
        load_gameplay_mtrand_oracle(
            path,
            seed=1234,
            maximum_draws=20,
        )
