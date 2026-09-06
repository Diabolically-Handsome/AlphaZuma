from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import struct
from types import SimpleNamespace

import pytest

if os.name != "nt":
    pytest.skip(
        "retail debugger helpers require native Windows",
        allow_module_level=True,
    )

from tools import trace_popcap_demo_commands as tracer
from tools.trace_popcap_demo_commands import (
    TraceResult,
    _decode_near_relative_branch_target,
    _parser,
    _startup_hidden_draw_stack_code,
    _write_gameplay_mtrand_sync_receipt_json,
    _write_initial_global_mtrand_receipt_json,
    _write_startup_global_mtrand_observation_receipt_json,
)
from zuma_rl.revenge_core import PopCapMTRandom


def test_parser_exposes_direct_natural_seed_observation() -> None:
    args = _parser().parse_args(
        [
            "--dmo",
            "input.dmo",
            "--direct-runtime-exe",
            "runtime.exe",
            "--changedir",
            "assets",
            "--direct-natural-seed",
        ]
    )

    assert args.direct_natural_seed is True
    assert args.crt_rand_seed is None


def test_initial_global_dispatch_thunk_preserves_wrapper_return_address() -> None:
    dispatch_target = _decode_near_relative_branch_target(
        0x004E4A61,
        bytes.fromhex("e8cacaf1ff"),
        opcode=0xE8,
    )
    wrapper_target = _decode_near_relative_branch_target(
        0x0040156C,
        bytes.fromhex("e91f5f2100"),
        opcode=0xE9,
    )

    assert dispatch_target == 0x00401530
    assert wrapper_target == 0x00617490
    assert dispatch_target != wrapper_target


def test_startup_hidden_draw_stack_code_uses_bounded_candidate_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, int] = {}

    def fake_candidates(
        process: int,
        stack_pointer: int,
        *,
        size: int,
    ) -> tuple[tuple[int, int], ...]:
        observed.update(
            process=process,
            stack_pointer=stack_pointer,
            size=size,
        )
        return ((0, 0x401000), (20, 0x617490))

    monkeypatch.setattr(tracer, "_stack_code_candidates", fake_candidates)

    rows = _startup_hidden_draw_stack_code(123, 0xABCDEF00)

    assert observed == {
        "process": 123,
        "stack_pointer": 0xABCDEF00,
        "size": 256,
    }
    assert rows == [
        {
            "offset": 0,
            "address": 0x401000,
            "address_hex": "0x00401000",
        },
        {
            "offset": 20,
            "address": 0x617490,
            "address_hex": "0x00617490",
        },
    ]


def test_parser_exposes_gameplay_mtrand_receipt_contract() -> None:
    args = _parser().parse_args(
        [
            "--dmo",
            "input.dmo",
            "--gameplay-mtrand-oracle",
            "oracle.json",
            "--gameplay-mtrand-oracle-seed",
            "85961375",
            "--gameplay-mtrand-oracle-maximum-draws",
            "20000",
            "--gameplay-mtrand-receipt-json",
            "receipt.json",
        ]
    )

    assert args.gameplay_mtrand_oracle == Path("oracle.json")
    assert args.gameplay_mtrand_oracle_seed == 85961375
    assert args.gameplay_mtrand_oracle_maximum_draws == 20000
    assert args.gameplay_mtrand_receipt_json == Path("receipt.json")


def test_gameplay_mtrand_receipt_writer_persists_pre_blackout_pass(
    tmp_path: Path,
) -> None:
    dmo = tmp_path / "input.dmo"
    dmo.write_bytes(b"dmo\n")
    output = tmp_path / "receipt.json"
    receipt = {
        "status": "PASS",
        "oracle_complete": True,
        "hardware_breakpoint_restored": True,
        "hardware_breakpoint_restore_error": None,
        "expected_hit_count": 1,
        "hit_count": 1,
        "process_memory_mutation": True,
        "process_context_mutation": True,
    }
    result = TraceResult(
        hits=10,
        exited=False,
        exit_code=None,
        brokered_blocks=1,
        broker_bypassed_blocks=0,
        startup_post_bypass_worker_yields=(),
        broker_failures=(),
        boundary_failures=(),
        last_update=7001,
        stopped_at_update=True,
        stopped_at_command_order=False,
    )

    digest = _write_gameplay_mtrand_sync_receipt_json(
        output,
        dmo=dmo,
        runtime_exe=tmp_path / "runtime.exe",
        runtime_process_id=123,
        started_perf_counter_ns=10,
        finished_perf_counter_ns=20,
        receipt=receipt,
        observations=[{"order": 0, "framework_update": 4012}],
        result=result,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert digest.startswith("sha256:")
    assert payload["status"] == "PASS"
    assert payload["receipt"]["hit_count"] == 1
    assert payload["pre_blackout_trace_result"]["failure_count"] == 0


def test_parser_exposes_initial_global_mtrand_seed_contract() -> None:
    args = _parser().parse_args(
        [
            "--dmo",
            "input.dmo",
            "--initial-global-mtrand-oracle",
            "global-calls.json",
            "--initial-global-mtrand-seed",
            "23557968",
            "--initial-global-mtrand-receipt-json",
            "initial-seed.json",
        ]
    )

    assert args.initial_global_mtrand_oracle == Path("global-calls.json")
    assert args.initial_global_mtrand_seed == 23557968
    assert args.initial_global_mtrand_receipt_json == Path(
        "initial-seed.json"
    )


def test_initial_global_mtrand_receipt_binds_atomic_startup_seed(
    tmp_path: Path,
) -> None:
    dmo = tmp_path / "input.dmo"
    dmo.write_bytes(b"dmo\n")
    output = tmp_path / "initial-seed.json"
    oracle = SimpleNamespace(
        source_path=(tmp_path / "global-calls.json").resolve(),
        source_sha256="sha256:" + "1" * 64,
        source_process_id=111,
        source_main_thread_id=222,
        runtime_executable_sha256="sha256:" + "2" * 64,
        seed=23557968,
        semantic_sha256="sha256:" + "3" * 64,
        wrapper_address=0x617490,
        call_address=0x4E4A61,
        caller=0x4E4A66,
        output=1030539336,
        pre_index=624,
        pre_state_sha256="sha256:" + "4" * 64,
        post_index=1,
        post_state_sha256="sha256:" + "5" * 64,
    )
    observation = {
        "restored_pre_state_sha256": oracle.pre_state_sha256,
        "expected_post_state_sha256": oracle.post_state_sha256,
        "changed": True,
        "bytes_written": 2500,
        "process_memory_mutation": True,
        "process_context_mutation": True,
        "persistent_file_modified": False,
        "atomic_window_completed": True,
        "call_target_entered": True,
        "call_return_observed": True,
        "other_threads_resumed_after_call_return": True,
        "observed_output": oracle.output,
        "observed_post_state_sha256": oracle.post_state_sha256,
    }
    result = TraceResult(
        hits=1,
        exited=False,
        exit_code=None,
        brokered_blocks=0,
        broker_bypassed_blocks=0,
        startup_post_bypass_worker_yields=(),
        broker_failures=(),
        boundary_failures=(),
        last_update=0,
        stopped_at_update=True,
        stopped_at_command_order=False,
    )

    digest = _write_initial_global_mtrand_receipt_json(
        output,
        dmo=dmo,
        runtime_exe=tmp_path / "runtime.exe",
        runtime_process_id=123,
        started_perf_counter_ns=10,
        finished_perf_counter_ns=20,
        oracle=oracle,
        observations=[observation],
        result=result,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert digest.startswith("sha256:")
    assert payload["status"] == "PASS"
    assert payload["state_correction_count"] == 1
    assert payload["bytes_written_total"] == 2500
    assert payload["process_memory_mutation"] is True


def test_parser_exposes_startup_global_mtrand_observation_contract() -> None:
    args = _parser().parse_args(
        [
            "--dmo",
            "input.dmo",
            "--startup-global-mtrand-observation-oracle",
            "global-calls.json",
            "--startup-global-mtrand-observation-seed",
            "23557968",
            "--startup-global-mtrand-observation-end-at-update",
            "1702",
            "--startup-global-mtrand-observation-maximum-draws",
            "20000",
            "--startup-global-mtrand-observation-receipt-json",
            "startup-observation.json",
            "--startup-global-mtrand-hidden-draw-start-after-source-order",
            "308",
            "--startup-global-mtrand-hidden-draw-stop-before-source-order",
            "309",
        ]
    )

    assert args.startup_global_mtrand_observation_oracle == Path(
        "global-calls.json"
    )
    assert args.startup_global_mtrand_observation_seed == 23557968
    assert args.startup_global_mtrand_observation_end_at_update == 1702
    assert args.startup_global_mtrand_observation_maximum_draws == 20000
    assert args.startup_global_mtrand_observation_receipt_json == Path(
        "startup-observation.json"
    )
    assert (
        args.startup_global_mtrand_hidden_draw_start_after_source_order == 308
    )
    assert (
        args.startup_global_mtrand_hidden_draw_stop_before_source_order == 309
    )


def test_startup_global_mtrand_observation_receipt_localizes_mismatch(
    tmp_path: Path,
) -> None:
    dmo = tmp_path / "input.dmo"
    dmo.write_bytes(b"dmo\n")
    output = tmp_path / "startup-observation.json"
    seed = 4321
    rng = PopCapMTRandom(seed)
    transitions = []
    for order in range(2):
        pre = struct.pack("<625I", *rng.words, rng.index)
        value = rng.next_u31()
        post = struct.pack("<625I", *rng.words, rng.index)
        transitions.append(
            SimpleNamespace(
                source_order=order,
                framework_update=100 + order,
                caller=0x401000,
                output=value,
                pre_draw_count=order,
                pre_index=struct.unpack_from("<I", pre, 2496)[0],
                pre_state_sha256=(
                    "sha256:" + hashlib.sha256(pre).hexdigest()
                ),
                post_draw_count=order + 1,
                post_index=struct.unpack_from("<I", post, 2496)[0],
                post_state_sha256=(
                    "sha256:" + hashlib.sha256(post).hexdigest()
                ),
            )
        )
    oracle = SimpleNamespace(
        source_path=(tmp_path / "global-calls.json").resolve(),
        source_sha256="sha256:" + "1" * 64,
        source_process_id=111,
        source_main_thread_id=222,
        runtime_executable_sha256="sha256:" + "2" * 64,
        seed=seed,
        semantic_sha256="sha256:" + "3" * 64,
        wrapper_address=0x617490,
        end_at_update=101,
        maximum_draws=20,
        entries=tuple(transitions),
    )
    observations = []
    for order, entry in enumerate(transitions):
        observations.append(
            {
                "order": order,
                "pre_state_sha256": entry.pre_state_sha256,
                "inferred_post_state_sha256": entry.post_state_sha256,
                "caller_match": order == 0,
                "framework_update_match": True,
                "pre_state_match": True,
                "transition_match": True,
                "exact_match": order == 0,
            }
        )
    receipt = {
        "status": "PASS",
        "oracle_complete": True,
        "hardware_breakpoint_restored": True,
        "hardware_breakpoint_restore_error": None,
        "expected_hit_count": 2,
        "hit_count": 2,
        "process_memory_writes": 0,
        "process_memory_mutation": False,
        "process_context_mutation": True,
    }
    result = TraceResult(
        hits=10,
        exited=False,
        exit_code=None,
        brokered_blocks=1,
        broker_bypassed_blocks=0,
        startup_post_bypass_worker_yields=(),
        broker_failures=(),
        boundary_failures=(),
        last_update=7001,
        stopped_at_update=True,
        stopped_at_command_order=False,
    )

    digest = _write_startup_global_mtrand_observation_receipt_json(
        output,
        dmo=dmo,
        runtime_exe=tmp_path / "runtime.exe",
        runtime_process_id=123,
        started_perf_counter_ns=10,
        finished_perf_counter_ns=20,
        oracle=oracle,
        receipt=receipt,
        observations=observations,
        result=result,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert digest.startswith("sha256:")
    assert payload["status"] == "PASS"
    assert payload["comparison"]["first_exact_mismatch_order"] == 1
    assert payload["comparison"]["first_caller_mismatch_order"] == 1
    assert payload["comparison"]["first_draw_count_mismatch_order"] is None
    assert [
        row["observed_pre_draw_count"] for row in payload["observations"]
    ] == [0, 1]
    assert payload["process_memory_writes"] == 0


def test_startup_global_mtrand_hidden_draw_receipt_preserves_deficit(
    tmp_path: Path,
) -> None:
    dmo = tmp_path / "input.dmo"
    dmo.write_bytes(b"dmo\n")
    output = tmp_path / "startup-hidden-draws.json"
    seed = 9876
    rng = PopCapMTRandom(seed)

    def state() -> tuple[int, str]:
        data = struct.pack("<625I", *rng.words, rng.index)
        return (
            rng.index,
            "sha256:" + hashlib.sha256(data).hexdigest(),
        )

    source_pre_zero_index, source_pre_zero_hash = state()
    source_output_zero = rng.next_u31()
    source_post_zero_index, source_post_zero_hash = state()
    for _ in range(3):
        rng.next_u31()
    source_pre_one_index, source_pre_one_hash = state()
    source_output_one = rng.next_u31()
    source_post_one_index, source_post_one_hash = state()
    entries = (
        SimpleNamespace(
            source_order=308,
            framework_update=1675,
            caller=0x40AE62,
            output=source_output_zero,
            pre_draw_count=0,
            pre_index=source_pre_zero_index,
            pre_state_sha256=source_pre_zero_hash,
            post_draw_count=1,
            post_index=source_post_zero_index,
            post_state_sha256=source_post_zero_hash,
        ),
        SimpleNamespace(
            source_order=309,
            framework_update=3121,
            caller=0x40AE62,
            output=source_output_one,
            pre_draw_count=4,
            pre_index=source_pre_one_index,
            pre_state_sha256=source_pre_one_hash,
            post_draw_count=5,
            post_index=source_post_one_index,
            post_state_sha256=source_post_one_hash,
        ),
    )
    replay_rng = PopCapMTRandom(seed)
    replay_pre_zero = struct.pack(
        "<625I", *replay_rng.words, replay_rng.index
    )
    replay_rng.next_u31()
    hidden_states: list[tuple[int, str]] = []
    for _ in range(3):
        replay_state = struct.pack(
            "<625I", *replay_rng.words, replay_rng.index
        )
        hidden_states.append(
            (
                replay_rng.index,
                "sha256:" + hashlib.sha256(replay_state).hexdigest(),
            )
        )
        if len(hidden_states) < 3:
            replay_rng.next_u31()
    replay_pre_one = struct.pack(
        "<625I", *replay_rng.words, replay_rng.index
    )
    replay_rng.next_u31()
    replay_post_one = struct.pack(
        "<625I", *replay_rng.words, replay_rng.index
    )
    observations = [
        {
            "order": 0,
            "pre_state_sha256": (
                "sha256:" + hashlib.sha256(replay_pre_zero).hexdigest()
            ),
            "inferred_post_state_sha256": hidden_states[0][1],
            "caller_match": True,
            "framework_update_match": True,
            "pre_state_match": True,
            "transition_match": True,
            "exact_match": True,
        },
        {
            "order": 1,
            "pre_state_sha256": (
                "sha256:" + hashlib.sha256(replay_pre_one).hexdigest()
            ),
            "inferred_post_state_sha256": (
                "sha256:" + hashlib.sha256(replay_post_one).hexdigest()
            ),
            "caller_match": True,
            "framework_update_match": True,
            "pre_state_match": False,
            "transition_match": False,
            "exact_match": False,
        },
    ]
    hidden_observations = [
        {
            "order": order,
            "phase": (
                "start_boundary_wrapper_draw"
                if order == 0
                else "hidden_interval_draw"
            ),
            "framework_update": 1675 + order,
            "stack_return": 0x40AE62 if order == 0 else 0x4E6215,
            "stack_return_hex": (
                "0x0040ae62" if order == 0 else "0x004e6215"
            ),
            "instruction_pointer": 0x40158A,
            "instruction_pointer_hex": "0x0040158a",
            "post_index": post_index,
            "post_state_sha256": post_hash,
            "hardware_breakpoint_slot": 1,
            "hardware_breakpoint_access": "write",
            "hardware_breakpoint_length_bytes": 4,
            "process_memory_writes": 0,
            "process_memory_mutation": False,
            "process_context_mutation": True,
            "persistent_file_modified": False,
        }
        for order, (post_index, post_hash) in enumerate(hidden_states)
    ]
    oracle = SimpleNamespace(
        source_path=(tmp_path / "global-calls.json").resolve(),
        source_sha256="sha256:" + "1" * 64,
        source_process_id=111,
        source_main_thread_id=222,
        runtime_executable_sha256="sha256:" + "2" * 64,
        seed=seed,
        semantic_sha256="sha256:" + "3" * 64,
        wrapper_address=0x617490,
        end_at_update=3121,
        maximum_draws=20,
        entries=entries,
    )
    hidden_contract = {
        "status": "PASS",
        "failure": None,
        "start_after_source_order": 308,
        "start_entry_order": 0,
        "start_framework_update": 1675,
        "start_post_draw_count": 1,
        "start_post_state_sha256": source_post_zero_hash,
        "stop_before_source_order": 309,
        "stop_entry_order": 1,
        "stop_framework_update": 3121,
        "expected_stop_pre_draw_count": 4,
        "expected_stop_pre_state_sha256": source_pre_one_hash,
        "expected_hidden_draw_count": 3,
        "expected_total_index_write_count": 4,
        "start_boundary_observed": True,
        "stop_boundary_observed": True,
        "watch_armed": True,
        "watch_disarmed": True,
        "index_write_count": 3,
        "observed_hidden_write_count": 2,
        "process_memory_writes": 0,
        "process_memory_mutation": False,
        "process_context_mutation": True,
        "persistent_file_modified": False,
    }
    receipt = {
        "status": "PASS",
        "failure": None,
        "oracle_complete": True,
        "hardware_breakpoint_restored": True,
        "hardware_breakpoint_restore_error": None,
        "expected_hit_count": 2,
        "hit_count": 2,
        "process_memory_writes": 0,
        "process_memory_mutation": False,
        "process_context_mutation": True,
        "hidden_draw_interval": hidden_contract,
    }
    result = TraceResult(
        hits=10,
        exited=False,
        exit_code=None,
        brokered_blocks=1,
        broker_bypassed_blocks=0,
        startup_post_bypass_worker_yields=(),
        broker_failures=(),
        boundary_failures=(),
        last_update=3200,
        stopped_at_update=True,
        stopped_at_command_order=False,
    )

    _write_startup_global_mtrand_observation_receipt_json(
        output,
        dmo=dmo,
        runtime_exe=tmp_path / "runtime.exe",
        runtime_process_id=123,
        started_perf_counter_ns=10,
        finished_perf_counter_ns=20,
        oracle=oracle,
        receipt=receipt,
        observations=observations,
        result=result,
        hidden_draw_observations=hidden_observations,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    hidden = payload["hidden_draw_interval"]
    assert payload["status"] == "PASS"
    assert hidden["status"] == "PASS"
    assert hidden["summary"]["expected_hidden_draw_count"] == 3
    assert hidden["summary"]["observed_hidden_draw_count"] == 2
    assert hidden["summary"]["missing_hidden_draw_count"] == 1
    assert hidden["summary"]["deficit_detected"] is True
    assert [
        row["observed_post_draw_count"] for row in hidden["observations"]
    ] == [1, 2, 3]
    assert hidden["caller_histogram"][0]["count"] == 2
    assert hidden["process_memory_writes"] == 0
