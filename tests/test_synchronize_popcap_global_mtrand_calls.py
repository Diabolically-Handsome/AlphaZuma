from __future__ import annotations

from pathlib import Path

import pytest

from tools import synchronize_popcap_global_mtrand_calls as synchronizer


def test_parser_exposes_read_only_boundary_mode() -> None:
    args = synchronizer._parser().parse_args(
        [
            "--pid",
            "123",
            "--main-thread-id",
            "456",
            "--executable",
            "runtime.exe",
            "--oracle",
            "oracle.json",
            "--seed",
            "23557968",
            "--start-after-update",
            "1702",
            "--end-at-update",
            "3121",
            "--output",
            "result.json",
            "--ready",
            "ready.json",
            "--stop",
            "stop.txt",
            "--observe-boundary-only",
            "--observe-boundary-thread-runtime",
            "--observe-handoff-thread-runtime",
            "--observe-handoff-rng-state",
            "--handoff-rng-wait-target-words-sha256",
            "sha256:" + ("ab" * 32),
            "--handoff-rng-wait-min-index",
            "315",
            "--handoff-rng-wait-timeout-seconds",
            "0.25",
            "--handoff-rng-wait-poll-interval-seconds",
            "0.001",
        ]
    )

    assert args.observe_boundary_only is True
    assert args.observe_boundary_thread_runtime is True
    assert args.observe_handoff_thread_runtime is True
    assert args.observe_handoff_rng_state is True
    assert args.handoff_rng_wait_target_words_sha256 == (
        "sha256:" + ("ab" * 32)
    )
    assert args.handoff_rng_wait_minimum_index == 315
    assert args.handoff_rng_wait_timeout_seconds == 0.25
    assert args.handoff_rng_wait_poll_interval_seconds == 0.001
    assert args.maximum_draws == 100_000
    assert args.oracle == Path("oracle.json")


def test_read_only_boundary_policy_never_reads_back_or_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> bytes:
        raise AssertionError("read/write helper must not be called")

    monkeypatch.setattr(synchronizer, "read_memory", forbidden)
    monkeypatch.setattr(synchronizer, "write_memory", forbidden)

    natural_match, correction = (
        synchronizer._prepare_global_mtrand_pre_state(
            process=1,
            live_state=b"live",
            expected_state=b"source",
            observe_boundary_only=True,
        )
    )

    assert natural_match is False
    assert correction is False


def test_synchronizing_policy_reports_the_ephemeral_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writes: list[bytes] = []

    def fake_write(process: int, address: int, payload: bytes) -> None:
        assert process == 7
        assert address == synchronizer.DEFAULT_GLOBAL_RNG_STATE
        writes.append(payload)

    monkeypatch.setattr(synchronizer, "write_memory", fake_write)
    monkeypatch.setattr(
        synchronizer,
        "read_memory",
        lambda process, address, size: b"source",
    )

    natural_match, correction = (
        synchronizer._prepare_global_mtrand_pre_state(
            process=7,
            live_state=b"live",
            expected_state=b"source",
            observe_boundary_only=False,
        )
    )

    assert natural_match is False
    assert correction is True
    assert writes == [b"source"]


def test_handoff_rng_snapshot_is_one_read_and_zero_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = bytearray(synchronizer.MTRAND_STATE_BYTES)
    state[synchronizer.MTRAND_STATE_WORDS * 4 :] = (414).to_bytes(
        4,
        "little",
    )
    calls: list[tuple[int, int, int]] = []

    def fake_read(process: int, address: int, size: int) -> bytes:
        calls.append((process, address, size))
        return bytes(state)

    monkeypatch.setattr(synchronizer, "read_memory", fake_read)
    monkeypatch.setattr(
        synchronizer.time,
        "perf_counter_ns",
        lambda: 123456,
    )

    observed = synchronizer._capture_handoff_rng_state(7)

    assert calls == [
        (
            7,
            synchronizer.DEFAULT_GLOBAL_RNG_STATE,
            synchronizer.MTRAND_STATE_BYTES,
        )
    ]
    assert observed["index"] == 414
    assert observed["state_bytes"] == synchronizer.MTRAND_STATE_BYTES
    assert observed["words_sha256"] == synchronizer._sha256_bytes(
        bytes(state[: synchronizer.MTRAND_STATE_WORDS * 4])
    )
    assert observed["process_memory_reads"] == 1
    assert observed["process_memory_read_bytes"] == len(state)
    assert observed["process_memory_writes"] == 0
    assert observed["process_memory_mutation"] is False
    assert observed["captured_perf_counter_ns"] == 123456


def test_handoff_rng_readiness_wait_reaches_target_without_writes() -> None:
    target_words = "sha256:" + ("ab" * 32)
    observations = iter(
        [
            {
                "state_sha256": "sha256:" + ("01" * 32),
                "words_sha256": target_words,
                "index": 299,
                "captured_perf_counter_ns": 101,
            },
            {
                "state_sha256": "sha256:" + ("02" * 32),
                "words_sha256": target_words,
                "index": 315,
                "captured_perf_counter_ns": 103,
            },
        ]
    )
    clock = iter([100, 102, 104])
    sleeps: list[float] = []

    final, receipt = synchronizer._wait_for_handoff_rng_readiness(
        process=7,
        target_words_sha256=target_words,
        minimum_index=315,
        timeout_seconds=0.25,
        poll_interval_seconds=0.001,
        capture_state=lambda process: next(observations),
        perf_counter_ns=lambda: next(clock),
        sleep=sleeps.append,
    )

    assert final["index"] == 315
    assert receipt["status"] == "TARGET_REACHED"
    assert receipt["target_reached"] is True
    assert receipt["poll_count"] == 1
    assert receipt["process_memory_reads"] == 2
    assert receipt["process_memory_read_bytes"] == (
        2 * synchronizer.MTRAND_STATE_BYTES
    )
    assert receipt["process_memory_writes"] == 0
    assert receipt["process_memory_mutation"] is False
    assert sleeps == [0.001]


def test_handoff_rng_readiness_timeout_is_recorded_and_released() -> None:
    target_words = "sha256:" + ("ab" * 32)
    observation = {
        "state_sha256": "sha256:" + ("01" * 32),
        "words_sha256": target_words,
        "index": 299,
        "captured_perf_counter_ns": 101,
    }
    clock = iter([100, 250_000_101, 250_000_102])

    final, receipt = synchronizer._wait_for_handoff_rng_readiness(
        process=7,
        target_words_sha256=target_words,
        minimum_index=315,
        timeout_seconds=0.25,
        poll_interval_seconds=0.001,
        capture_state=lambda process: observation,
        perf_counter_ns=lambda: next(clock),
        sleep=lambda seconds: pytest.fail("expired wait must not sleep"),
    )

    assert final == observation
    assert receipt["status"] == "TIMEOUT"
    assert receipt["target_reached"] is False
    assert receipt["poll_count"] == 0
    assert receipt["process_memory_reads"] == 1
    assert receipt["process_memory_writes"] == 0


def test_ready_is_not_published_when_handoff_resume_is_unverified(
    tmp_path: Path,
) -> None:
    calls: list[object] = []

    with pytest.raises(
        RuntimeError,
        match="expected suspend count 1, observed 0",
    ):
        synchronizer._publish_ready_after_verified_handoff(
            main_thread_id=456,
            ready_path=tmp_path / "ready.json",
            ready_payload={"process_id": 123},
            resume_main_thread_on_ready=True,
            open_main_thread=lambda thread_id: calls.append(
                ("open", thread_id)
            )
            or 99,
            resume_thread=lambda handle: calls.append(("resume", handle))
            or 0,
            suspend_thread=lambda handle: calls.append(
                ("suspend", handle)
            )
            or 0,
            close_handle=lambda handle: calls.append(("close", handle)),
            write_ready=lambda path, payload: calls.append(
                ("write", path, payload)
            ),
            perf_counter_ns=lambda: 123456,
        )

    assert calls == [("open", 456), ("resume", 99), ("close", 99)]
    assert not (tmp_path / "ready.json").exists()


def test_ready_publication_follows_verified_handoff_resume(
    tmp_path: Path,
) -> None:
    calls: list[object] = []
    written: dict[str, object] = {}

    def write_ready(path: Path, payload: dict[str, object]) -> None:
        calls.append(("write", path))
        written.update(payload)

    result = synchronizer._publish_ready_after_verified_handoff(
        main_thread_id=456,
        ready_path=tmp_path / "ready.json",
        ready_payload={"process_id": 123},
        resume_main_thread_on_ready=True,
        open_main_thread=lambda thread_id: calls.append(
            ("open", thread_id)
        )
        or 99,
        resume_thread=lambda handle: calls.append(("resume", handle)) or 1,
        suspend_thread=lambda handle: pytest.fail(
            "verified handoff must not restore a suspension"
        ),
        close_handle=lambda handle: calls.append(("close", handle)),
        write_ready=write_ready,
        perf_counter_ns=lambda: 123456,
    )

    assert result == (True, 1, 123456)
    assert calls == [
        ("open", 456),
        ("resume", 99),
        ("close", 99),
        ("write", tmp_path / "ready.json"),
    ]
    assert written == {
        "process_id": 123,
        "schema": synchronizer.READY_SCHEMA,
        "version": synchronizer.READY_VERSION,
        "handoff_verified": True,
        "handoff_main_thread_resumed": True,
        "handoff_resume_previous_suspend_count": 1,
        "publication_order": "AFTER_VERIFIED_HANDOFF_RESUME",
        "published_perf_counter_ns": 123456,
    }


def test_over_suspended_handoff_is_restored_before_ready_is_rejected(
    tmp_path: Path,
) -> None:
    calls: list[object] = []

    with pytest.raises(
        RuntimeError,
        match="expected suspend count 1, observed 2",
    ):
        synchronizer._publish_ready_after_verified_handoff(
            main_thread_id=456,
            ready_path=tmp_path / "ready.json",
            ready_payload={"process_id": 123},
            resume_main_thread_on_ready=True,
            open_main_thread=lambda thread_id: 99,
            resume_thread=lambda handle: calls.append(("resume", handle))
            or 2,
            suspend_thread=lambda handle: calls.append(
                ("restore", handle)
            )
            or 1,
            close_handle=lambda handle: calls.append(("close", handle)),
            write_ready=lambda path, payload: calls.append(("write", path)),
            perf_counter_ns=lambda: 123456,
        )

    assert calls == [("resume", 99), ("restore", 99), ("close", 99)]
