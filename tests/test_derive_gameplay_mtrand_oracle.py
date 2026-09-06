from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct

import pytest

from tools.derive_gameplay_mtrand_oracle import (
    DEFAULT_GAMEPLAY_RETURN,
    derive_gameplay_oracle,
)
from tools.popcap_mtrand_call_oracle import (
    EXPECTED_RETAIL_RUNTIME_SHA256,
    load_gameplay_mtrand_oracle,
)
from zuma_rl.revenge_core import PopCapMTRandom


def _payload(rng: PopCapMTRandom) -> bytes:
    return struct.pack("<625I", *rng.words, rng.index)


def _sha(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _full_trace(
    path: Path,
    *,
    duplicate_target: bool = False,
) -> int:
    seed = 12345
    rng = PopCapMTRandom(seed)
    calls = []
    order = 0
    for update in (10, 11, 12):
        callers = [0x11111111, DEFAULT_GAMEPLAY_RETURN]
        if duplicate_target and update == 11:
            callers.append(DEFAULT_GAMEPLAY_RETURN)
        for caller in callers:
            pre = _payload(rng)
            output = rng.next_u31()
            post = _payload(rng)
            calls.append(
                {
                    "order": order,
                    "framework_update": update,
                    "native_game_time": update + 100,
                    "score": 50,
                    "score_target": 100,
                    "thread_id": 456,
                    "caller": caller,
                    "output": output,
                    "pre_index": struct.unpack_from(
                        "<I", pre, 624 * 4
                    )[0],
                    "post_index": struct.unpack_from(
                        "<I", post, 624 * 4
                    )[0],
                    "pre_state_sha256": _sha(pre),
                    "post_state_sha256": _sha(post),
                }
            )
            order += 1
    value = {
        "schema": "zuma-rl.pc-gameplay-mtrand-call-trace",
        "version": 1,
        "status": "PASS",
        "failure": None,
        "breakpoint_kind": "global_wrapper_entry",
        "process_memory_writes": 0,
        "persistent_file_modified": False,
        "hardware_breakpoint_restored": True,
        "stop_break_requested": True,
        "stop_break_observed": True,
        "runtime_executable_sha256": EXPECTED_RETAIL_RUNTIME_SHA256,
        "process_id": 123,
        "main_thread_id": 456,
        "call_count": len(calls),
        "calls": calls,
    }
    path.write_text(json.dumps(value), encoding="ascii")
    return seed


def test_derivation_filters_and_reindexes_contiguous_gameplay_calls(
    tmp_path: Path,
) -> None:
    source = tmp_path / "full.json"
    seed = _full_trace(source)

    report = derive_gameplay_oracle(
        source,
        seed=seed,
        start_after_update=10,
        end_at_update=12,
        maximum_draws=100,
    )

    assert report["call_count"] == 2
    assert [row["order"] for row in report["calls"]] == [0, 1]
    assert [
        row["framework_update"] for row in report["calls"]
    ] == [11, 12]
    assert all(
        row["caller"] == DEFAULT_GAMEPLAY_RETURN
        for row in report["calls"]
    )
    output = tmp_path / "gameplay.json"
    output.write_text(json.dumps(report), encoding="ascii")
    oracle = load_gameplay_mtrand_oracle(
        output,
        seed=seed,
        maximum_draws=100,
    )
    assert len(oracle.entries) == 2
    assert oracle.entries[-1].framework_update == 12


def test_derivation_prepends_exact_bootstrap_source_calls(
    tmp_path: Path,
) -> None:
    source = tmp_path / "full.json"
    seed = _full_trace(source)

    report = derive_gameplay_oracle(
        source,
        seed=seed,
        start_after_update=9,
        end_at_update=12,
        bootstrap_source_orders=(0,),
        maximum_draws=100,
    )

    assert report["call_count"] == 4
    assert [row["source_order"] for row in report["calls"]] == [
        0,
        1,
        3,
        5,
    ]
    assert report["calls"][0]["caller"] == 0x11111111
    assert report["provenance"]["bootstrap_source_orders"] == [0]
    output = tmp_path / "gameplay-with-bootstrap.json"
    output.write_text(json.dumps(report), encoding="ascii")
    oracle = load_gameplay_mtrand_oracle(
        output,
        seed=seed,
        maximum_draws=100,
    )
    assert [entry.caller for entry in oracle.entries] == [
        0x11111111,
        DEFAULT_GAMEPLAY_RETURN,
        DEFAULT_GAMEPLAY_RETURN,
        DEFAULT_GAMEPLAY_RETURN,
    ]


def test_derivation_rejects_duplicate_gameplay_call_in_update(
    tmp_path: Path,
) -> None:
    source = tmp_path / "full.json"
    seed = _full_trace(source, duplicate_target=True)

    with pytest.raises(
        ValueError,
        match="exactly one contiguous call per update",
    ):
        derive_gameplay_oracle(
            source,
            seed=seed,
            start_after_update=10,
            end_at_update=12,
            maximum_draws=100,
        )
