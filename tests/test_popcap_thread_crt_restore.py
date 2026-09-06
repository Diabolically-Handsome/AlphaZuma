from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.popcap_thread_crt_restore import (
    load_source_bound_thread_crt_restore_state,
    load_thread_crt_restore_state,
)


def _header(*, writes: int = 0) -> dict[str, object]:
    return {
        "schema": "zuma-rl.live-rng-change-monitor",
        "version": 1,
        "process_id": 123,
        "process_memory_writes": writes,
        "thread_crt_state": {
            "thread_id": 456,
            "ptd_thread_id": 456,
            "rand_state_address": 0x123400,
        },
    }


def _row(*, update: int = 1740) -> dict[str, object]:
    return {
        "type": "rng",
        "framework_update": update,
        "native_game_time": 417,
        "thread_crt_rand_state": 1_021_395_514,
    }


def test_load_thread_crt_restore_state_binds_thread_and_value(
    tmp_path: Path,
) -> None:
    path = tmp_path / "rng.ndjson"
    path.write_text(
        json.dumps(_header()) + "\n" + json.dumps(_row()) + "\n",
        encoding="utf-8",
    )

    observed = load_thread_crt_restore_state(
        path,
        framework_update=1740,
    )

    assert observed.source_process_id == 123
    assert observed.source_thread_id == 456
    assert observed.captured_state == 1_021_395_514
    assert observed.rewind_draws == 0
    assert observed.state == 1_021_395_514
    assert observed.semantic_sha256.startswith("sha256:")


def test_load_thread_crt_restore_state_can_derive_prior_boundary(
    tmp_path: Path,
) -> None:
    path = tmp_path / "rng.ndjson"
    path.write_text(
        json.dumps(_header()) + "\n" + json.dumps(_row()) + "\n",
        encoding="utf-8",
    )

    observed = load_thread_crt_restore_state(
        path,
        framework_update=1740,
        rewind_draws=2,
    )

    assert observed.captured_state == 0x3CE1423A
    assert observed.rewind_draws == 2
    assert observed.state == 0xF62785C0


def test_load_thread_crt_restore_state_requires_unique_row(
    tmp_path: Path,
) -> None:
    path = tmp_path / "rng.ndjson"
    row = json.dumps(_row())
    path.write_text(
        json.dumps(_header()) + "\n" + row + "\n" + row + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exactly one row"):
        load_thread_crt_restore_state(path, framework_update=1740)


def test_load_thread_crt_restore_state_rejects_unresolved_thread(
    tmp_path: Path,
) -> None:
    path = tmp_path / "rng.ndjson"
    header = _header()
    header["thread_crt_state"] = None
    path.write_text(
        json.dumps(header) + "\n" + json.dumps(_row()) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="resolved thread state"):
        load_thread_crt_restore_state(path, framework_update=1740)


def test_load_thread_crt_restore_state_rejects_mutating_source(
    tmp_path: Path,
) -> None:
    path = tmp_path / "rng.ndjson"
    path.write_text(
        json.dumps(_header(writes=1))
        + "\n"
        + json.dumps(_row())
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="header contract"):
        load_thread_crt_restore_state(path, framework_update=1740)


def _source_bound_files(
    tmp_path: Path,
    *,
    controlled_seed: int | None = None,
    trace_writes: int = 0,
) -> tuple[Path, Path]:
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
        "process_memory_writes": trace_writes,
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
        "controlled_crt_rand_seed": controlled_seed,
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
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return trace_path, report_path


def test_load_source_bound_thread_crt_restore_state_binds_natural_source(
    tmp_path: Path,
) -> None:
    trace_path, report_path = _source_bound_files(tmp_path)

    observed = load_source_bound_thread_crt_restore_state(
        trace_path,
        report_path,
        source_order=0,
        framework_update=3120,
        caller=0x65B821,
    )

    assert observed.source_kind == "natural_retail_hardware_trace"
    assert observed.source_recording_report_path == report_path.resolve()
    assert observed.source_dmo_path == (tmp_path / "input.dmo").resolve()
    assert observed.source_call_order == 0
    assert observed.source_caller == 0x65B821
    assert observed.state == 1_165_183_229


def test_load_source_bound_thread_crt_restore_state_rejects_controlled_source(
    tmp_path: Path,
) -> None:
    trace_path, report_path = _source_bound_files(
        tmp_path,
        controlled_seed=123,
    )

    with pytest.raises(ValueError, match="recording contract"):
        load_source_bound_thread_crt_restore_state(
            trace_path,
            report_path,
            source_order=0,
            framework_update=3120,
            caller=0x65B821,
        )


def test_load_source_bound_thread_crt_restore_state_rejects_mutating_trace(
    tmp_path: Path,
) -> None:
    trace_path, report_path = _source_bound_files(tmp_path, trace_writes=1)

    with pytest.raises(ValueError, match="trace contract"):
        load_source_bound_thread_crt_restore_state(
            trace_path,
            report_path,
            source_order=0,
            framework_update=3120,
            caller=0x65B821,
        )
