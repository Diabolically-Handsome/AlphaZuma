from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct

import pytest

from tools.popcap_global_mtrand_call_oracle import (
    load_initial_global_mtrand_call_oracle,
    load_global_mtrand_call_oracle,
    load_startup_global_mtrand_observation_oracle,
    reconstruct_mtrand_draw_counts,
)
from tools.popcap_mtrand_call_oracle import (
    EXPECTED_RETAIL_RUNTIME_SHA256,
)
from zuma_rl.revenge_core import PopCapMTRandom


def _payload(rng: PopCapMTRandom) -> bytes:
    return struct.pack("<625I", *rng.words, rng.index)


def _trace(path: Path, *, gap: bool = False) -> None:
    rng = PopCapMTRandom(4321)
    calls = []
    for order in range(4):
        pre = _payload(rng)
        output = rng.next_u31()
        post = _payload(rng)
        calls.append(
            {
                "order": order,
                "thread_id": 8,
                "framework_update": 100 + order,
                "native_game_time": 20 + order,
                "score": 8000,
                "score_target": 9650,
                "caller": 0x401000,
                "output": output,
                "pre_index": struct.unpack_from("<I", pre, 2496)[0],
                "pre_state_sha256": (
                    "sha256:" + hashlib.sha256(pre).hexdigest()
                ),
                "post_index": struct.unpack_from(
                    "<I",
                    post,
                    2496,
                )[0],
                "post_state_sha256": (
                    "sha256:" + hashlib.sha256(post).hexdigest()
                ),
            }
        )
        if gap and order == 1:
            rng.next_u31()
    path.write_text(
        json.dumps(
            {
                "schema": "zuma-rl.pc-gameplay-mtrand-call-trace",
                "version": 1,
                "status": "PASS",
                "failure": None,
                "breakpoint_kind": "global_wrapper_entry",
                "address": 0x617490,
                "process_id": 7,
                "main_thread_id": 8,
                "runtime_executable_sha256": (
                    EXPECTED_RETAIL_RUNTIME_SHA256
                ),
                "process_memory_writes": 0,
                "persistent_file_modified": False,
                "hardware_breakpoint_restored": True,
                "stop_break_requested": True,
                "stop_break_observed": True,
                "call_count": len(calls),
                "calls": calls,
            },
            sort_keys=True,
        ),
        encoding="ascii",
    )


def test_load_global_mtrand_call_oracle_is_contiguous(
    tmp_path: Path,
) -> None:
    path = tmp_path / "oracle.json"
    _trace(path)

    oracle = load_global_mtrand_call_oracle(
        path,
        seed=4321,
        start_after_update=100,
        maximum_draws=20,
    )

    assert len(oracle.entries) == 3
    assert oracle.entries[0].source_order == 1
    assert [
        entry.pre_draw_count for entry in oracle.entries
    ] == [1, 2, 3]
    assert oracle.semantic_sha256.startswith("sha256:")


def test_load_global_mtrand_call_oracle_rejects_hidden_draw(
    tmp_path: Path,
) -> None:
    path = tmp_path / "oracle.json"
    _trace(path, gap=True)

    with pytest.raises(
        ValueError,
        match="unobserved inter-call draw",
    ):
        load_global_mtrand_call_oracle(
            path,
            seed=4321,
            start_after_update=100,
            maximum_draws=20,
        )


def test_reconstruct_mtrand_draw_counts_maps_seeded_states() -> None:
    seed = 4321
    rng = PopCapMTRandom(seed)
    expected: dict[str, int] = {}
    for draw_count in range(8):
        if draw_count in {0, 2, 7}:
            digest = "sha256:" + hashlib.sha256(_payload(rng)).hexdigest()
            expected[digest] = draw_count
        rng.next_u31()

    observed = reconstruct_mtrand_draw_counts(
        seed=seed,
        state_sha256=set(expected),
        maximum_draws=7,
    )

    assert observed == expected


def test_load_startup_global_mtrand_observation_allows_hidden_draws(
    tmp_path: Path,
) -> None:
    path = tmp_path / "startup-oracle.json"
    _trace(path, gap=True)

    oracle = load_startup_global_mtrand_observation_oracle(
        path,
        seed=4321,
        end_at_update=102,
        maximum_draws=20,
    )

    assert oracle.wrapper_address == 0x617490
    assert [entry.source_order for entry in oracle.entries] == [0, 1, 2]
    assert [entry.pre_draw_count for entry in oracle.entries] == [0, 1, 3]
    assert [entry.post_draw_count for entry in oracle.entries] == [1, 2, 4]
    assert oracle.semantic_sha256.startswith("sha256:")


def test_load_initial_global_mtrand_call_oracle_binds_order_zero(
    tmp_path: Path,
) -> None:
    path = tmp_path / "initial-oracle.json"
    seed = 4321
    rng = PopCapMTRandom(seed)
    pre = _payload(rng)
    output = rng.next_u31()
    post = _payload(rng)
    caller = 0x4E4A66
    path.write_text(
        json.dumps(
            {
                "schema": "zuma-rl.pc-gameplay-mtrand-call-trace",
                "version": 1,
                "status": "PASS",
                "failure": None,
                "breakpoint_kind": "global_wrapper_entry",
                "address": 0x617490,
                "process_id": 7,
                "main_thread_id": 8,
                "runtime_executable_sha256": (
                    EXPECTED_RETAIL_RUNTIME_SHA256
                ),
                "process_memory_writes": 0,
                "persistent_file_modified": False,
                "hardware_breakpoint_restored": True,
                "stop_break_requested": True,
                "stop_break_observed": True,
                "call_count": 1,
                "calls": [
                    {
                        "order": 0,
                        "thread_id": 8,
                        "framework_update": 0,
                        "native_game_time": None,
                        "score": None,
                        "score_target": None,
                        "caller": caller,
                        "output": output,
                        "pre_index": struct.unpack_from(
                            "<I", pre, 2496
                        )[0],
                        "pre_state_sha256": (
                            "sha256:" + hashlib.sha256(pre).hexdigest()
                        ),
                        "post_index": struct.unpack_from(
                            "<I", post, 2496
                        )[0],
                        "post_state_sha256": (
                            "sha256:" + hashlib.sha256(post).hexdigest()
                        ),
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="ascii",
    )

    oracle = load_initial_global_mtrand_call_oracle(path, seed=seed)

    assert oracle.call_address == caller - 5
    assert oracle.wrapper_address == 0x617490
    assert oracle.pre_payload == pre
    assert oracle.post_payload == post
    assert oracle.semantic_sha256.startswith("sha256:")


def test_load_initial_global_mtrand_call_oracle_rejects_wrong_seed(
    tmp_path: Path,
) -> None:
    path = tmp_path / "initial-oracle.json"
    seed = 4321
    rng = PopCapMTRandom(seed)
    pre = _payload(rng)
    output = rng.next_u31()
    post = _payload(rng)
    path.write_text(
        json.dumps(
            {
                "schema": "zuma-rl.pc-gameplay-mtrand-call-trace",
                "version": 1,
                "status": "PASS",
                "failure": None,
                "breakpoint_kind": "global_wrapper_entry",
                "address": 0x617490,
                "process_id": 7,
                "main_thread_id": 8,
                "runtime_executable_sha256": (
                    EXPECTED_RETAIL_RUNTIME_SHA256
                ),
                "process_memory_writes": 0,
                "persistent_file_modified": False,
                "hardware_breakpoint_restored": True,
                "stop_break_requested": True,
                "stop_break_observed": True,
                "call_count": 1,
                "calls": [
                    {
                        "order": 0,
                        "thread_id": 8,
                        "framework_update": 0,
                        "caller": 0x4E4A66,
                        "output": output,
                        "pre_index": struct.unpack_from(
                            "<I", pre, 2496
                        )[0],
                        "pre_state_sha256": (
                            "sha256:" + hashlib.sha256(pre).hexdigest()
                        ),
                        "post_index": struct.unpack_from(
                            "<I", post, 2496
                        )[0],
                        "post_state_sha256": (
                            "sha256:" + hashlib.sha256(post).hexdigest()
                        ),
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="ascii",
    )

    with pytest.raises(
        ValueError,
        match="seeded transition mismatch",
    ):
        load_initial_global_mtrand_call_oracle(path, seed=seed + 1)
