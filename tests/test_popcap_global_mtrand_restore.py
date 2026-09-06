from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct

import pytest

from tools.popcap_global_mtrand_restore import (
    MTRAND_STATE_BYTES,
    load_global_mtrand_restore_state,
    load_source_bound_global_mtrand_restore_state,
    reconstruct_global_mtrand_state,
)
from zuma_rl.revenge_core import PopCapMTRandom


SEED = 160_147_421
DRAW_COUNT = 989
UPDATE = 1740


def _captured_state() -> tuple[int, str]:
    rng = PopCapMTRandom(SEED)
    for _ in range(DRAW_COUNT):
        rng.next_u31()
    payload = struct.pack("<625I", *rng.words, rng.index)
    return rng.index, hashlib.sha256(payload).hexdigest()


def _monitor(*, writes: int = 0, state_hash: str | None = None) -> str:
    index, expected_hash = _captured_state()
    header = {
        "schema": "zuma-rl.live-rng-change-monitor",
        "version": 1,
        "process_id": 123,
        "process_memory_writes": writes,
    }
    row = {
        "type": "rng",
        "framework_update": UPDATE,
        "native_game_time": 417,
        "global_mtrand_index": index,
        "global_mtrand_sha256": state_hash or expected_hash,
    }
    return json.dumps(header) + "\n" + json.dumps(row) + "\n"


def test_reconstruct_global_mtrand_state_binds_seed_draws_and_hash() -> None:
    index, state_hash = _captured_state()

    draw_count, payload = reconstruct_global_mtrand_state(
        seed=SEED,
        index=index,
        state_sha256=state_hash,
        maximum_draws=2_000,
    )

    assert draw_count == DRAW_COUNT
    assert len(payload) == MTRAND_STATE_BYTES
    assert hashlib.sha256(payload).hexdigest() == state_hash


def test_load_global_mtrand_restore_state_binds_read_only_evidence(
    tmp_path: Path,
) -> None:
    path = tmp_path / "rng.ndjson"
    path.write_text(_monitor(), encoding="utf-8")

    observed = load_global_mtrand_restore_state(
        path,
        framework_update=UPDATE,
        seed=SEED,
        maximum_draws=2_000,
    )

    assert observed.source_process_id == 123
    assert observed.seed == SEED
    assert observed.captured_draw_count == DRAW_COUNT
    assert observed.rewind_draws == 0
    assert observed.draw_count == DRAW_COUNT
    assert observed.index == 365
    assert observed.state_sha256.startswith("sha256:")
    assert observed.semantic_sha256.startswith("sha256:")


def test_load_global_mtrand_restore_state_can_derive_prior_boundary(
    tmp_path: Path,
) -> None:
    path = tmp_path / "rng.ndjson"
    path.write_text(_monitor(), encoding="utf-8")

    observed = load_global_mtrand_restore_state(
        path,
        framework_update=UPDATE,
        seed=SEED,
        maximum_draws=2_000,
        rewind_draws=1,
    )

    assert observed.captured_draw_count == DRAW_COUNT
    assert observed.captured_index == 365
    assert observed.rewind_draws == 1
    assert observed.draw_count == DRAW_COUNT - 1
    assert observed.index == 364
    assert observed.state_sha256 != observed.captured_state_sha256


def test_load_global_mtrand_restore_state_rejects_unmatched_hash(
    tmp_path: Path,
) -> None:
    path = tmp_path / "rng.ndjson"
    path.write_text(_monitor(state_hash="0" * 64), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly one seeded reconstruction"):
        load_global_mtrand_restore_state(
            path,
            framework_update=UPDATE,
            seed=SEED,
            maximum_draws=2_000,
        )


def test_load_global_mtrand_restore_state_rejects_mutating_source(
    tmp_path: Path,
) -> None:
    path = tmp_path / "rng.ndjson"
    path.write_text(_monitor(writes=1), encoding="utf-8")

    with pytest.raises(ValueError, match="header contract"):
        load_global_mtrand_restore_state(
            path,
            framework_update=UPDATE,
            seed=SEED,
            maximum_draws=2_000,
        )


def _source_bound_global_files(
    tmp_path: Path,
    *,
    monitor_receipt_hash: str | None = None,
) -> tuple[Path, Path, Path]:
    monitor_path = tmp_path / "rng.ndjson"
    monitor_path.write_text(_monitor(), encoding="utf-8")
    monitor_sha256 = (
        "sha256:" + hashlib.sha256(monitor_path.read_bytes()).hexdigest()
    )

    rng = PopCapMTRandom(SEED)
    for _ in range(DRAW_COUNT - 1):
        rng.next_u31()
    pre_payload = struct.pack("<625I", *rng.words, rng.index)
    pre_index = rng.index
    output = rng.next_u31()
    post_payload = struct.pack("<625I", *rng.words, rng.index)

    dmo_path = tmp_path / "input.dmo"
    dmo_path.write_bytes(b"natural retail DMO")
    dmo_sha256 = "sha256:" + hashlib.sha256(dmo_path.read_bytes()).hexdigest()
    trace_path = tmp_path / "global-mtrand-calls.json"
    trace = {
        "schema": "zuma-rl.pc-gameplay-mtrand-call-trace",
        "version": 1,
        "status": "PASS",
        "failure": None,
        "process_id": 123,
        "main_thread_id": 456,
        "process_memory_writes": 0,
        "persistent_file_modified": False,
        "hardware_breakpoint_restored": True,
        "hardware_breakpoint_restore_error": None,
        "debugger_detach_error": None,
        "runtime_executable_sha256": "sha256:" + "a" * 64,
        "call_count": 1,
        "calls": [
            {
                "order": 0,
                "framework_update": 3120,
                "native_game_time": 109,
                "thread_id": 456,
                "caller": 0x65B821,
                "pre_index": pre_index,
                "pre_state_sha256": (
                    "sha256:" + hashlib.sha256(pre_payload).hexdigest()
                ),
                "output": output,
                "post_index": rng.index,
                "post_state_sha256": (
                    "sha256:" + hashlib.sha256(post_payload).hexdigest()
                ),
                "thread_crt_rand_state": 1_165_183_229,
                "thread_crt_snapshot_error": None,
                "thread_crt_state": {
                    "ptd_thread_id": 456,
                    "rand_state_address": 0x123400,
                },
            }
        ],
    }
    trace_path.write_text(json.dumps(trace), encoding="utf-8")
    trace_sha256 = (
        "sha256:" + hashlib.sha256(trace_path.read_bytes()).hexdigest()
    )
    report_path = tmp_path / "result.json"
    report = {
        "schema": "zuma-rl.retail-autoplay-recording",
        "version": 1,
        "status": "PASS",
        "controlled_crt_rand_seed": None,
        "process_id": 123,
        "process_memory_writes": 0,
        "normal_exit": True,
        "exit_method": "retail_ui",
        "process_exit_code": 0,
        "host_restored": True,
        "host_pre_state_root": "sha256:" + "b" * 64,
        "host_restored_state_root": "sha256:" + "b" * 64,
        "runtime_executable_sha256": "sha256:" + "a" * 64,
        "dmo": str(dmo_path),
        "dmo_sha256": dmo_sha256,
        "gameplay": {"status": "PASS", "outcome": "natural_win"},
        "main_thread": {
            "source": "verified_retail_window",
            "process_memory_writes": 0,
            "thread_id": 456,
        },
        "gameplay_mtrand_trace": {
            "status": "PASS",
            "failure": None,
            "process_memory_writes": 0,
            "hardware_breakpoint_restored": True,
            "output": str(trace_path),
            "output_sha256": trace_sha256,
        },
        "rng_monitor": {
            "exit_code": 0,
            "process_memory_writes": 0,
            "output": str(monitor_path),
            "output_sha256": (
                monitor_sha256
                if monitor_receipt_hash is None
                else monitor_receipt_hash
            ),
            "ui_thread_id": 456,
            "stop_after_native_game_time": 500,
        },
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return monitor_path, trace_path, report_path


def test_load_source_bound_global_mtrand_restore_binds_natural_call(
    tmp_path: Path,
) -> None:
    monitor_path, trace_path, report_path = _source_bound_global_files(
        tmp_path
    )

    observed = load_source_bound_global_mtrand_restore_state(
        monitor_path,
        trace_path,
        report_path,
        monitor_framework_update=UPDATE,
        seed=SEED,
        rewind_draws=1,
        source_order=0,
        framework_update=3120,
        caller=0x65B821,
        maximum_draws=2_000,
    )

    assert observed.source_kind == (
        "natural_retail_hardware_trace_and_monitor"
    )
    assert observed.draw_count == DRAW_COUNT - 1
    assert observed.source_call_order == 0
    assert observed.source_caller == 0x65B821
    assert observed.source_trace_path == trace_path.resolve()
    assert observed.source_dmo_path == (tmp_path / "input.dmo").resolve()


def test_load_source_bound_global_mtrand_restore_rejects_unbound_monitor(
    tmp_path: Path,
) -> None:
    monitor_path, trace_path, report_path = _source_bound_global_files(
        tmp_path,
        monitor_receipt_hash="sha256:" + "0" * 64,
    )

    with pytest.raises(ValueError, match="monitor receipt mismatch"):
        load_source_bound_global_mtrand_restore_state(
            monitor_path,
            trace_path,
            report_path,
            monitor_framework_update=UPDATE,
            seed=SEED,
            rewind_draws=1,
            source_order=0,
            framework_update=3120,
            caller=0x65B821,
            maximum_draws=2_000,
        )
