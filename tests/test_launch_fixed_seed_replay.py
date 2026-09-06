from __future__ import annotations

from pathlib import Path
import struct

from tools import launch_fixed_seed_replay


def test_debugger_launch_evidence_binds_startup_trace_handoff() -> None:
    evidence = launch_fixed_seed_replay.FixedSeedLaunchEvidence(
        process_id=55,
        thread_id=66,
        seed=123,
        breakpoint_address=0x0068F572,
        original_instruction_hex="50",
        breakpoint_observed_perf_counter_ns=100,
        debugger_detached_perf_counter_ns=110,
        detach_pending_events_drained=0,
        detach_attempts=1,
        main_thread_suspended_on_detach=True,
        main_thread_suspend_previous_count=0,
        process_affinity={
            "application_stage": "created_suspended_before_first_resume",
            "requested_process_mask": 1,
            "observed_process_mask": 1,
            "handoff_observed_process_mask": 1,
            "handoff_verified": True,
        },
    )

    payload = evidence.to_dict()

    assert payload["thread_id"] == 66
    assert payload["main_thread_suspended_on_detach"] is True
    assert payload["main_thread_suspend_previous_count"] == 0
    assert payload["process_affinity"]["observed_process_mask"] == 1
    assert payload["process_affinity"]["handoff_verified"] is True


def test_natural_seed_evidence_attests_zero_rng_override() -> None:
    evidence = launch_fixed_seed_replay.NaturalSeedLaunchEvidence(
        process_id=55,
        thread_id=66,
        observed_seed=0x12345678,
        breakpoint_address=0x0068F572,
        original_instruction_hex="50",
        breakpoint_observed_perf_counter_ns=100,
        debugger_detached_perf_counter_ns=110,
        detach_pending_events_drained=0,
        detach_attempts=1,
        main_thread_suspended_on_detach=True,
        main_thread_suspend_previous_count=0,
    )

    payload = evidence.to_dict()

    assert payload["mode"] == "retail_natural_seed_observation"
    assert payload["observed_seed"] == 0x12345678
    assert payload["effective_seed"] == 0x12345678
    assert payload["register_override"] is None
    assert payload["rng_process_memory_writes"] == 0


def test_fixed_seed_iat_stub_filters_only_the_srand_tick_call() -> None:
    code = launch_fixed_seed_replay._fixed_seed_iat_stub_code(
        seed=0x11223344,
        original_pointer=0x55667788,
        seed_return_address=0x0068F577,
    )

    assert code == bytes.fromhex(
        "813c2477f56800"
        "7506"
        "b844332211"
        "c3"
        "b888776655"
        "ffe0"
    )
    assert len(code) == 22


def test_fixed_seed_iat_stub_rejects_invalid_forwarding_address() -> None:
    try:
        launch_fixed_seed_replay._fixed_seed_iat_stub_code(
            seed=1,
            original_pointer=0,
            seed_return_address=0x0068F577,
        )
    except ValueError as error:
        assert "must be nonzero" in str(error)
    else:
        raise AssertionError("zero forwarding address was accepted")


def test_fixed_seed_launch_command_supports_record_mode() -> None:
    command = launch_fixed_seed_replay._fixed_seed_launch_command(
        runtime_executable=Path(r"C:\runtime\popcapgame1.exe"),
        changedir=Path(r"C:\assets\Zuma's Revenge"),
        dmo=Path(r"C:\evidence\natural win.dmo"),
        launch_mode="record",
    )

    assert "-record" in command
    assert "-play" not in command
    assert "-demofile=" in command
    assert "natural win.dmo" in command
    assert "-changedir=" in command


def test_fixed_seed_launch_command_rejects_unknown_mode() -> None:
    try:
        launch_fixed_seed_replay._fixed_seed_launch_command(
            runtime_executable=Path("runtime.exe"),
            changedir=Path("assets"),
            dmo=Path("input.dmo"),
            launch_mode="observe",
        )
    except ValueError as error:
        assert "play or record" in str(error)
    else:
        raise AssertionError("unknown fixed-seed launch mode was accepted")


def test_startup_replay_state_reads_framework_demo_fields(
    monkeypatch,
) -> None:
    base = 0x12345000
    memory = {
        launch_fixed_seed_replay.G_SEXY_APP_BASE_ADDRESS: struct.pack(
            "<I", base
        ),
        base + launch_fixed_seed_replay.SEXY_APP_UPDATE_OFFSET: (
            struct.pack("<i", 1005)
        ),
        base
        + launch_fixed_seed_replay.SEXY_APP_DEMO_READ_BIT_POSITION_OFFSET: (
            struct.pack("<i", 712)
        ),
        base
        + launch_fixed_seed_replay.SEXY_APP_LAST_DEMO_UPDATE_OFFSET: (
            struct.pack("<i", 1005)
        ),
        base
        + launch_fixed_seed_replay.SEXY_APP_NEEDS_COMMAND_OFFSET: b"\x01",
        base
        + launch_fixed_seed_replay.SEXY_APP_COMMAND_NUMBER_OFFSET: (
            struct.pack("<i", 16)
        ),
        base
        + launch_fixed_seed_replay.SEXY_APP_COMMAND_ORDER_OFFSET: (
            struct.pack("<i", 91)
        ),
        base
        + launch_fixed_seed_replay.SEXY_APP_COMMAND_BIT_POSITION_OFFSET: (
            struct.pack("<i", 704)
        ),
        base
        + launch_fixed_seed_replay.SEXY_APP_DEMO_LOADING_COMPLETE_OFFSET: (
            b"\x00"
        ),
    }

    def fake_read_memory(
        unused_process_handle,
        address: int,
        size: int,
    ) -> bytes:
        value = memory[address]
        assert len(value) == size
        return value

    monkeypatch.setattr(
        launch_fixed_seed_replay,
        "read_memory",
        fake_read_memory,
    )

    assert launch_fixed_seed_replay._startup_replay_state(123) == {
        "available": True,
        "base_address_hex": "0x12345000",
        "framework_update": 1005,
        "buffer_read_bit_position": 712,
        "last_demo_update": 1005,
        "needs_command": True,
        "command_number": 16,
        "command_order": 91,
        "command_bit_position": 704,
        "demo_loading_complete": False,
    }


def test_startup_replay_state_is_best_effort(monkeypatch) -> None:
    def fail_read_memory(
        unused_process_handle,
        unused_address: int,
        unused_size: int,
    ) -> bytes:
        raise OSError("process unavailable")

    monkeypatch.setattr(
        launch_fixed_seed_replay,
        "read_memory",
        fail_read_memory,
    )

    state = launch_fixed_seed_replay._startup_replay_state(123)

    assert state["available"] is False
    assert state["reason"] == "OSError: process unavailable"
