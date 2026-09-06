"""Tests for authentic single-process retail source evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from zuma_rl import pc_source
from zuma_rl.pc_exact_step_evidence import canonical_json_bytes


def _write(path: Path, value: object) -> None:
    if isinstance(value, (bytes, bytearray)):
        path.write_bytes(bytes(value))
    else:
        path.write_bytes(canonical_json_bytes(value))


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _binding(path: Path, root: Path) -> dict[str, str]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _digest(path),
    }


def _strict_replay_probe(
    probe: Path,
    *,
    runtime: Path,
    dmo: Path,
    process_id: int,
    global_rng_seed: int | None = None,
    rng_process_memory_writes: int = 0,
    stop_after_update: int = 3151,
) -> None:
    startup_seed = 123456789
    null_controls = {
        name: None
        for name in (
            "crt_rand_seed",
            "startup_seed_transport",
            "board_seed",
            "global_rng_seed",
            "thread_crt_rng_seed",
            "gameplay_mtrand_oracle",
            "gameplay_mtrand_oracle_seed",
            "initial_global_mtrand_oracle",
            "initial_global_mtrand_seed",
            "startup_global_mtrand_observation_oracle",
            "startup_global_mtrand_observation_seed",
            "global_mtrand_restore_log",
            "global_mtrand_restore_seed",
            "global_mtrand_restore_command_order",
            "qrand_restore_log",
            "qrand_restore_command_order",
            "thread_crt_restore_log",
            "thread_crt_restore_command_order",
            "source_bound_board_seed",
        )
    }
    null_controls["global_rng_seed"] = global_rng_seed
    options = {
        **null_controls,
        "direct_runtime": True,
        "direct_natural_seed": True,
        "broker_service_blocks": True,
        "rng_seed_override_count": 0,
        "attach_at_update": 0,
        "stop_after_update": stop_after_update,
        "startup_trace_handoff": True,
        "allow_font_cache_manifest_completion_debt": True,
        "seed_board_before_attach": False,
        "allow_source_bound_board_global_correction": False,
        "source_bound_board_anchor": False,
        "source_bound_board_precall_global_restore": False,
    }
    result = {
        "startup_post_bypass_worker_yields": [
            {
                "process_id": process_id,
                "memory_writes": 0,
                "failure": None,
                "command_breakpoint_rearmed": True,
                "temporary_breakpoint_memory_writes": 4,
            }
        ],
        "deferred_file_write_discharges": [],
        "font_cache_manifest_completion_discharges": [],
        "terminal_file_write_payload_handoffs": [],
        "service_file_write_header_claims": [],
        "preloading_failed_file_write_tail_short_header_recoveries": [],
        "failure_count": 0,
        "boundary_failures": [],
        "broker_failures": [],
        "stopped_at_update": True,
        "last_update": stop_after_update,
        "font_cache_manifest_entry_count": 35,
        "font_cache_manifest_sha256": "1" * 64,
        "font_cache_manifest_main_pak_sha256": "2" * 64,
    }
    strict_path = probe.parent / "strict-replay.json"
    _write(
        strict_path,
        {
            "schema": "zuma.popcap_strict_replay.v3",
            "started_utc": "2026-08-09T00:00:00Z",
            "finished_utc": "2026-08-09T00:00:01Z",
            "trace_started_perf_counter_ns": 100,
            "trace_finished_perf_counter_ns": 200,
            "runtime_executable": str(runtime.resolve()),
            "runtime_process_id": process_id,
            "source_dmo": {
                "path": str(dmo.resolve()),
                "sha256": _digest(dmo).removeprefix("sha256:"),
                "size_bytes": dmo.stat().st_size,
            },
            "options": options,
            "startup_rng": {
                "mode": "retail_natural_seed_observation",
                "process_id": process_id,
                "thread_id": 7,
                "observed_seed": startup_seed,
                "effective_seed": startup_seed,
                "seed_source": "retail_eax_before_push_to_srand",
                "register_override": None,
                "rng_process_memory_writes": rng_process_memory_writes,
                "persistent_file_modified": False,
            },
            "result": result,
        },
    )
    _write(
        probe,
        {
            "strict_command_replay": {
                "schema": pc_source.STRICT_REPLAY_BINDING_SCHEMA,
                "version": pc_source.STRICT_REPLAY_BINDING_VERSION,
                "status": "PASS",
                "artifact": strict_path.name,
                "artifact_bytes": strict_path.stat().st_size,
                "artifact_sha256": _digest(strict_path),
                "runtime_process_id": process_id,
                "runtime_executable": str(runtime.resolve()),
                "runtime_executable_sha256": _digest(runtime),
                "source_dmo": str(dmo.resolve()),
                "source_dmo_sha256": _digest(dmo),
                "stop_after_update": stop_after_update,
                "failure_count": 0,
                "startup_seed": startup_seed,
                "seed_source": "retail_eax_before_push_to_srand",
                "register_override": None,
                "rng_seed_override_count": 0,
                "rng_process_memory_writes": rng_process_memory_writes,
                "font_cache_manifest": {
                    "entry_count": 35,
                    "manifest_sha256": "sha256:" + "1" * 64,
                    "main_pak_sha256": "sha256:" + "2" * 64,
                },
                "transport": {
                    "service_broker": True,
                    "startup_worker_yield_count": 1,
                    "temporary_code_breakpoint_memory_write_count": 4,
                    "callback_context_emulation_count": 0,
                    "replay_state_recovery_count": 0,
                    "replay_state_recovery_memory_write_count": 0,
                    "replay_state_recovery_memory_write_bytes": 0,
                    "gameplay_or_rng_process_memory_write_count": 0,
                },
            }
        },
    )


def _build_source(
    root: Path,
    original_root: Path,
    *,
    run_id: str,
    process_id: int,
    filetime: int,
) -> tuple[Path, dict[str, object]]:
    source_root = root / run_id
    source_root.mkdir()
    runtime = source_root / "runtime.exe"
    dmo = source_root / "replay.dmo"
    attempts = source_root / "attempts.json"
    probe = source_root / "probe.json"
    index = source_root / "index.json"
    pre = source_root / "pre.json"
    post = source_root / "post.json"
    _write(runtime, b"derived-runtime")
    _write(dmo, b"demo")
    _write(attempts, {"artifact": "attempts"})
    _write(probe, {"artifact": "probe"})
    _write(
        index,
        {
            "process_identity": {
                "executable_sha256": _digest(runtime),
                "executable_hash_provenance": {
                    "payload_sha256": _digest(runtime),
                    "source_sha256": _digest(
                        original_root / "ZumasRevenge.exe"
                    ),
                    "runtime_file_direct_hash_verified": True,
                },
            }
        },
    )
    _write(pre, {"phase": "pre"})
    _write(post, {"phase": "post"})

    manifest = source_root / "manifest.json"
    _write(
        manifest,
        {
            "schema": pc_source.SOURCE_SCHEMA,
            "version": pc_source.SOURCE_VERSION,
            "source_id": run_id,
            "scope": {
                "level_id": "Jungle2",
                "hard": False,
                "profile_mode": "tutorials_completed",
                "mode": "adventure",
            },
            "original_executable_sha256": _digest(
                original_root / "ZumasRevenge.exe"
            ),
            "runtime_payload": _binding(runtime, root),
            "dmo": _binding(dmo, root),
            "run": {
                "run_id": run_id,
                "selected_attempt": 1,
                "attempts": _binding(attempts, root),
                "memory_probe": _binding(probe, root),
                "trajectory_index": _binding(index, root),
                "process_id": process_id,
                "process_creation_filetime_100ns": filetime,
                "maximum_startup_attempts": 3,
            },
            "window": {
                "freeze_update": 100,
                "start_update": 104,
                "end_update": 915,
                "warmup_tick_count": 4,
            },
            "state_transaction": {
                "pre_snapshot": _binding(pre, root),
                "post_snapshot": _binding(post, root),
            },
        },
    )
    loaded = {
        "frames": tuple(range(812)),
        "campaign_started_perf_counter_ns": 200,
        "attempt_finished_perf_counter_ns": 300,
        "process_id": process_id,
        "process_creation_filetime_100ns": filetime,
        "probe_sha256": _digest(probe),
        "index_sha256": _digest(index),
    }
    return manifest, loaded


def test_independent_retail_sources_do_not_require_cross_process_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_root = tmp_path / "original"
    original_root.mkdir()
    _write(original_root / "ZumasRevenge.exe", b"retail-executable")
    manifest_a, loaded_a = _build_source(
        tmp_path,
        original_root,
        run_id="source-a",
        process_id=1001,
        filetime=10_001,
    )
    manifest_b, loaded_b = _build_source(
        tmp_path,
        original_root,
        run_id="source-b",
        process_id=1002,
        filetime=10_002,
    )
    loaded_by_id = {"source-a": loaded_a, "source-b": loaded_b}

    monkeypatch.setattr(
        pc_source,
        "load_formal_exact_step_run",
        lambda **kwargs: SimpleNamespace(**loaded_by_id[kwargs["run_id"]]),
    )
    monkeypatch.setattr(
        pc_source.PopCapDemo,
        "read",
        staticmethod(lambda path: SimpleNamespace(random_seed=777)),
    )

    def read_snapshot(path: Path) -> SimpleNamespace:
        is_pre = path.name == "pre.json"
        return SimpleNamespace(
            session_nonce="nonce-12345678",
            state_root="sha256:" + "a" * 64,
            captured_perf_counter_ns=100 if is_pre else 400,
        )

    monkeypatch.setattr(
        pc_source.PcStateSnapshot,
        "read",
        staticmethod(read_snapshot),
    )

    reports = [
        pc_source.verify_pc_source_manifest(
            manifest,
            evidence_root=tmp_path,
            original_root=original_root,
        )
        for manifest in (manifest_a, manifest_b)
    ]

    assert [report["status"] for report in reports] == ["PASS", "PASS"]
    assert all(
        report["cross_process_exact_replay_required"] is False
        for report in reports
    )
    assert reports[0]["source_fingerprint"] != reports[1]["source_fingerprint"]
    assert reports[0]["dmo_sha256"] == _digest(
        manifest_a.parent / "replay.dmo"
    )
    assert reports[0]["runtime_sha256"] == _digest(
        manifest_a.parent / "runtime.exe"
    )
    assert "long_horizon_drift" in reports[0]["authorized_features"]


def test_source_requires_exact_host_state_restoration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_root = tmp_path / "original"
    original_root.mkdir()
    _write(original_root / "ZumasRevenge.exe", b"retail-executable")
    manifest, loaded = _build_source(
        tmp_path,
        original_root,
        run_id="source-a",
        process_id=1001,
        filetime=10_001,
    )
    monkeypatch.setattr(
        pc_source,
        "load_formal_exact_step_run",
        lambda **kwargs: SimpleNamespace(**loaded),
    )
    monkeypatch.setattr(
        pc_source.PopCapDemo,
        "read",
        staticmethod(lambda path: SimpleNamespace(random_seed=777)),
    )
    monkeypatch.setattr(
        pc_source.PcStateSnapshot,
        "read",
        staticmethod(
            lambda path: SimpleNamespace(
                session_nonce="nonce-12345678",
                state_root=(
                    "sha256:" + ("a" if path.name == "pre.json" else "b") * 64
                ),
                captured_perf_counter_ns=(
                    100 if path.name == "pre.json" else 400
                ),
            )
        ),
    )

    with pytest.raises(
        pc_source.PcSourceValidationError,
        match="not restored exactly",
    ):
        pc_source.verify_pc_source_manifest(
            manifest,
            evidence_root=tmp_path,
            original_root=original_root,
        )


def test_full_state_source_uses_v2_loader_and_requires_strict_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_root = tmp_path / "original"
    original_root.mkdir()
    _write(original_root / "ZumasRevenge.exe", b"retail-executable")
    manifest, loaded = _build_source(
        tmp_path,
        original_root,
        run_id="full-source",
        process_id=1001,
        filetime=10_001,
    )
    payload = json.loads(manifest.read_text("ascii"))
    payload["version"] = pc_source.FULL_STATE_SOURCE_VERSION
    payload["run"]["source_kind"] = pc_source.FULL_STATE_SOURCE_KIND
    payload["window"] = {
        "freeze_update": 104,
        "start_update": 104,
        "end_update": 915,
        "warmup_tick_count": 0,
    }
    probe = manifest.parent / "probe.json"
    runtime = manifest.parent / "runtime.exe"
    dmo = manifest.parent / "replay.dmo"
    _strict_replay_probe(
        probe,
        runtime=runtime,
        dmo=dmo,
        process_id=1001,
    )
    payload["run"]["memory_probe"] = _binding(probe, tmp_path)
    _write(manifest, payload)
    loaded["probe_sha256"] = _digest(probe)
    monkeypatch.setattr(
        pc_source,
        "load_formal_full_state_run",
        lambda **kwargs: SimpleNamespace(**loaded),
    )
    monkeypatch.setattr(
        pc_source,
        "load_formal_exact_step_run",
        lambda **kwargs: pytest.fail("legacy exact-step loader was used"),
    )
    monkeypatch.setattr(
        pc_source.PopCapDemo,
        "read",
        staticmethod(lambda path: SimpleNamespace(random_seed=777)),
    )
    monkeypatch.setattr(
        pc_source.PcStateSnapshot,
        "read",
        staticmethod(
            lambda path: SimpleNamespace(
                session_nonce="nonce-12345678",
                state_root="sha256:" + "a" * 64,
                captured_perf_counter_ns=(
                    100 if path.name == "pre.json" else 400
                ),
            )
        ),
    )

    report = pc_source.verify_pc_source_manifest(
        manifest,
        evidence_root=tmp_path,
        original_root=original_root,
    )

    assert report["status"] == "PASS"
    assert report["transport"] == (
        "single_retail_process_full_state_exact_step_source"
    )
    assert report["strict_command_replay"]["status"] == "PASS"


def test_source_bound_full_state_source_uses_distinct_v3_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_root = tmp_path / "original"
    original_root.mkdir()
    _write(original_root / "ZumasRevenge.exe", b"retail-executable")
    manifest, loaded = _build_source(
        tmp_path,
        original_root,
        run_id="source-bound-full-state",
        process_id=1001,
        filetime=10_001,
    )
    payload = json.loads(manifest.read_text("ascii"))
    payload["version"] = pc_source.SOURCE_BOUND_FULL_STATE_SOURCE_VERSION
    payload["run"]["source_kind"] = (
        pc_source.SOURCE_BOUND_FULL_STATE_SOURCE_KIND
    )
    payload["window"] = {
        "freeze_update": 104,
        "start_update": 104,
        "end_update": 915,
        "warmup_tick_count": 0,
    }
    _write(manifest, payload)
    monkeypatch.setattr(
        pc_source,
        "load_formal_full_state_run",
        lambda **kwargs: SimpleNamespace(**loaded),
    )
    monkeypatch.setattr(
        pc_source,
        "_validate_strict_command_replay",
        lambda *args, **kwargs: None,
    )
    source_bound_binding = {
        "schema": pc_source.SOURCE_BOUND_REPLAY_BINDING_SCHEMA,
        "version": pc_source.SOURCE_BOUND_REPLAY_BINDING_VERSION,
        "status": "PASS",
    }
    monkeypatch.setattr(
        pc_source,
        "_validate_source_bound_command_replay",
        lambda *args, **kwargs: source_bound_binding,
    )
    monkeypatch.setattr(
        pc_source.PopCapDemo,
        "read",
        staticmethod(lambda path: SimpleNamespace(random_seed=777)),
    )
    monkeypatch.setattr(
        pc_source.PcStateSnapshot,
        "read",
        staticmethod(
            lambda path: SimpleNamespace(
                session_nonce="nonce-12345678",
                state_root="sha256:" + "a" * 64,
                captured_perf_counter_ns=(
                    100 if path.name == "pre.json" else 400
                ),
            )
        ),
    )

    report = pc_source.verify_pc_source_manifest(
        manifest,
        evidence_root=tmp_path,
        original_root=original_root,
    )

    assert report["status"] == "PASS"
    assert report["transport"] == (
        "single_retail_process_source_bound_full_state_exact_step_source"
    )
    assert report["strict_command_replay"] == source_bound_binding


def test_source_bound_font_manifest_accepts_richer_plan_receipt() -> None:
    compact = {
        "entry_count": 35,
        "manifest_sha256": "sha256:" + "1" * 64,
        "main_pak_sha256": "sha256:" + "2" * 64,
    }
    plan_receipt = {
        **compact,
        "loading_complete_row_index": 75,
        "members": [{"member": "cached/font.cfw2", "size": 123}],
        "startup_file_write_rows": [],
    }

    assert pc_source._font_manifest_binding_matches(
        compact,
        plan_receipt,
    )
    assert not pc_source._font_manifest_binding_matches(
        compact,
        {**plan_receipt, "entry_count": 34},
    )


def test_strict_replay_binding_is_independently_recomputed(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime.exe"
    dmo = tmp_path / "input.dmo"
    probe = tmp_path / "probe.json"
    _write(runtime, b"runtime")
    _write(dmo, b"demo")
    _strict_replay_probe(
        probe,
        runtime=runtime,
        dmo=dmo,
        process_id=321,
    )

    binding = pc_source._validate_strict_command_replay(
        probe,
        expected_process_id=321,
        expected_runtime_path=runtime,
        expected_runtime_sha256=_digest(runtime),
        expected_dmo_path=dmo,
        expected_dmo_sha256=_digest(dmo),
    )

    assert binding is not None
    assert binding["status"] == "PASS"
    assert binding["rng_process_memory_writes"] == 0


def test_strict_replay_accepts_internally_bound_natural_stop_update(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "runtime.exe"
    dmo = tmp_path / "input.dmo"
    probe = tmp_path / "probe.json"
    _write(runtime, b"runtime")
    _write(dmo, b"demo")
    _strict_replay_probe(
        probe,
        runtime=runtime,
        dmo=dmo,
        process_id=321,
        stop_after_update=3123,
    )

    binding = pc_source._validate_strict_command_replay(
        probe,
        expected_process_id=321,
        expected_runtime_path=runtime,
        expected_runtime_sha256=_digest(runtime),
        expected_dmo_path=dmo,
        expected_dmo_sha256=_digest(dmo),
    )

    assert binding is not None
    assert binding["stop_after_update"] == 3123


@pytest.mark.parametrize(
    ("global_rng_seed", "rng_process_memory_writes"),
    ((7, 0), (None, 1)),
)
def test_strict_replay_rejects_rehashed_rng_mutation_evidence(
    tmp_path: Path,
    global_rng_seed: int | None,
    rng_process_memory_writes: int,
) -> None:
    runtime = tmp_path / "runtime.exe"
    dmo = tmp_path / "input.dmo"
    probe = tmp_path / "probe.json"
    _write(runtime, b"runtime")
    _write(dmo, b"demo")
    _strict_replay_probe(
        probe,
        runtime=runtime,
        dmo=dmo,
        process_id=321,
        global_rng_seed=global_rng_seed,
        rng_process_memory_writes=rng_process_memory_writes,
    )

    with pytest.raises(
        pc_source.PcSourceValidationError,
        match="strict command replay differs",
    ):
        pc_source._validate_strict_command_replay(
            probe,
            expected_process_id=321,
            expected_runtime_path=runtime,
            expected_runtime_sha256=_digest(runtime),
            expected_dmo_path=dmo,
            expected_dmo_sha256=_digest(dmo),
        )
