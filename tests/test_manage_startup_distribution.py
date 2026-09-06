"""Tests for startup-distribution campaign management."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import struct
from types import SimpleNamespace

import pytest

from tools import manage_startup_distribution as manager
from tools.collect_pc_golden_v4 import (
    DIRECT_NATURAL_SEED_LAUNCH_MODE,
    DIRECT_NATURAL_STRICT_BROKER_LAUNCH_MODE,
    EXPECTED_RUNTIME_SHA256,
    PLAN_SCHEMA,
    PLAN_VERSION,
)
from zuma_rl.pc_exact_step_evidence import canonical_json_bytes, read_canonical_json
from zuma_rl import distribution_fidelity as distribution


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_bytes(canonical_json_bytes(value))


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_derive_plan_removes_every_seed_and_debugger_control(
    tmp_path: Path,
    monkeypatch,
) -> None:
    base = tmp_path / "base"
    runtime = tmp_path / "runtime.exe"
    source = tmp_path / "ZumasRevenge.exe"
    assets = tmp_path / "assets"
    assets.mkdir()
    _write(runtime, b"runtime")
    _write(source, b"source")
    monkeypatch.setattr(manager, "EXPECTED_RUNTIME_SHA256", _digest(runtime))
    _write(base / "inputs" / "input.dmo", b"dmo")
    _write(base / "protocol" / "pre-template.json", {"pre": True})
    _write(base / "safety" / "host-pre.json", {"host": True})
    _write(
        base / "plan.json",
        {
            "schema": PLAN_SCHEMA,
            "version": PLAN_VERSION,
            "dmo": {
                "artifact": "inputs/input.dmo",
                "sha256": _digest(base / "inputs" / "input.dmo"),
            },
            "runtime": {
                "runtime_source_executable": str(source),
                "runtime_source_sha256": _digest(source),
                "expected_runtime_sha256": _digest(runtime),
                "crt_rand_seed": 1,
                "board_seed": 2,
                "global_rng_seed": 3,
                "thread_crt_rng_seed": 4,
            },
            "trace": {
                "startup_trace_handoff": True,
                "seed_board_before_attach": True,
                "source_bound_board_anchor": {"seed": 1},
                "source_bound_board_precall_global_restore": {"call": 1},
                "gameplay_mtrand_sync": {"seed": 1},
                "allow_source_bound_board_global_correction": True,
            },
            "prestate": {
                "template_artifact": "protocol/pre-template.json",
            },
            "safety": {"host_pre_artifact": "safety/host-pre.json"},
            "state_comparison": {"volatile_registry_roles": ["role"]},
        },
    )

    output = tmp_path / "derived"
    plan_path = manager.derive_natural_direct_plan(
        base_root=base,
        output_root=output,
        direct_runtime=runtime,
        changedir=assets,
    )

    plan = read_canonical_json(plan_path)
    assert plan["runtime"]["launch_mode"] == DIRECT_NATURAL_SEED_LAUNCH_MODE
    assert plan["runtime"]["runtime_executable_sha256"] == _digest(runtime)
    for field in (
        "crt_rand_seed",
        "startup_seed_transport",
        "board_seed_call_address",
        "board_seed",
        "global_rng_seed",
        "thread_crt_rng_seed",
    ):
        assert plan["runtime"][field] is None
    assert plan["trace"]["startup_trace_handoff"] is False
    assert plan["trace"]["source_bound_board_anchor"] is None
    assert plan["trace"]["gameplay_mtrand_sync"] is None
    assert read_canonical_json(output / "derivation.json")[
        "command_debugger_attached"
    ] is False


def test_derive_plan_can_bind_natural_strict_command_broker(
    tmp_path: Path,
    monkeypatch,
) -> None:
    base = tmp_path / "base"
    runtime = tmp_path / "runtime.exe"
    source = tmp_path / "ZumasRevenge.exe"
    assets = tmp_path / "assets"
    assets.mkdir()
    _write(runtime, b"runtime")
    _write(source, b"source")
    monkeypatch.setattr(manager, "EXPECTED_RUNTIME_SHA256", _digest(runtime))
    manifest_receipt = {
        "entry_count": 35,
        "manifest_sha256": "sha256:" + "1" * 64,
        "main_pak_sha256": "sha256:" + "2" * 64,
    }
    monkeypatch.setattr(
        manager,
        "_strict_font_cache_manifest_receipt",
        lambda changedir, dmo: manifest_receipt,
    )
    _write(base / "inputs" / "input.dmo", b"dmo")
    _write(base / "protocol" / "pre-template.json", {"pre": True})
    _write(base / "safety" / "host-pre.json", {"host": True})
    _write(
        base / "plan.json",
        {
            "schema": PLAN_SCHEMA,
            "version": PLAN_VERSION,
            "dmo": {
                "artifact": "inputs/input.dmo",
                "sha256": _digest(base / "inputs" / "input.dmo"),
            },
            "runtime": {
                "runtime_source_executable": str(source),
                "runtime_source_sha256": _digest(source),
                "expected_runtime_sha256": _digest(runtime),
            },
            "trace": {"allow_pre_stream_commands": True},
            "prestate": {
                "template_artifact": "protocol/pre-template.json",
            },
            "safety": {"host_pre_artifact": "safety/host-pre.json"},
        },
    )

    output = tmp_path / "derived-strict"
    plan_path = manager.derive_natural_direct_plan(
        base_root=base,
        output_root=output,
        direct_runtime=runtime,
        changedir=assets,
        strict_command_broker_stop_update=3151,
    )

    plan = read_canonical_json(plan_path)
    receipt = read_canonical_json(output / "derivation.json")
    assert plan["runtime"]["launch_mode"] == (
        DIRECT_NATURAL_STRICT_BROKER_LAUNCH_MODE
    )
    assert plan["trace"]["startup_trace_handoff"] is True
    assert plan["trace"][
        "natural_command_broker_stop_after_update"
    ] == 3151
    assert receipt["command_debugger_attached"] is True
    assert receipt["command_broker"]["rng_seed_override_count"] == 0
    assert plan["trace"][
        "allow_font_cache_manifest_completion_debt"
    ] is True
    assert plan["trace"]["font_cache_manifest_receipt"] == manifest_receipt
    assert receipt["command_broker"][
        "font_cache_manifest_completion"
    ]["rng_process_memory_writes"] == 0


def test_derive_seeded_slot_changes_only_dmo_header_seed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    evidence = tmp_path / "evidence"
    base = evidence / "base"
    runtime = evidence / "runtime.exe"
    source = evidence / "ZumasRevenge.exe"
    assets = evidence / "assets"
    assets.mkdir(parents=True)
    _write(runtime, b"runtime")
    _write(source, b"source")
    monkeypatch.setattr(manager, "EXPECTED_RUNTIME_SHA256", _digest(runtime))
    manifest_receipt = {
        "entry_count": 35,
        "manifest_sha256": "sha256:" + "1" * 64,
        "main_pak_sha256": "sha256:" + "2" * 64,
    }
    monkeypatch.setattr(
        manager,
        "_strict_font_cache_manifest_receipt",
        lambda changedir, dmo: manifest_receipt,
    )

    base_dmo = bytearray(b"D" * 64)
    struct.pack_into("<I", base_dmo, distribution.DMO_RANDOM_SEED_OFFSET, 7)
    _write(base / "inputs" / "input.dmo", bytes(base_dmo))
    _write(base / "protocol" / "pre-template.json", {"pre": True})
    _write(base / "safety" / "host-pre.json", {"host": True})
    _write(
        base / "plan.json",
        {
            "schema": PLAN_SCHEMA,
            "version": PLAN_VERSION,
            "dmo": {
                "artifact": "inputs/input.dmo",
                "bytes": len(base_dmo),
                "product_version": "test",
                "random_seed": 7,
                "length_updates": 10,
                "sha256": _digest(base / "inputs" / "input.dmo"),
            },
            "runtime": {
                "runtime_source_executable": str(source),
                "runtime_source_sha256": _digest(source),
                "expected_runtime_sha256": _digest(runtime),
            },
            "trace": {},
            "prestate": {
                "template_artifact": "protocol/pre-template.json",
            },
            "safety": {"host_pre_artifact": "safety/host-pre.json"},
        },
    )

    def fake_demo(payload: bytes) -> SimpleNamespace:
        return SimpleNamespace(
            random_seed=struct.unpack_from(
                "<I", payload, distribution.DMO_RANDOM_SEED_OFFSET
            )[0],
            version=2,
            product_version="test",
            length_updates=10,
            markers=(),
            commands=(),
        )

    monkeypatch.setattr(
        manager.PopCapDemo,
        "from_bytes",
        staticmethod(fake_demo),
    )
    output = evidence / "slot-001"
    plan_path = manager.derive_natural_direct_plan(
        base_root=base,
        output_root=output,
        direct_runtime=runtime,
        changedir=assets,
        strict_command_broker_stop_update=3151,
        pc_slot_ordinal=1,
        evidence_root=evidence,
    )

    expected_seed = distribution.frozen_seed_schedules()[0][0]
    derived = (output / "inputs" / "input.dmo").read_bytes()
    assert struct.unpack_from(
        "<I", derived, distribution.DMO_RANDOM_SEED_OFFSET
    )[0] == expected_seed
    assert (
        derived[: distribution.DMO_RANDOM_SEED_OFFSET]
        == base_dmo[: distribution.DMO_RANDOM_SEED_OFFSET]
    )
    assert (
        derived[
            distribution.DMO_RANDOM_SEED_OFFSET
            + distribution.DMO_RANDOM_SEED_BYTES :
        ]
        == base_dmo[
            distribution.DMO_RANDOM_SEED_OFFSET
            + distribution.DMO_RANDOM_SEED_BYTES :
        ]
    )
    plan = read_canonical_json(plan_path)
    assert plan["dmo"]["random_seed"] == expected_seed
    assert plan["dmo"]["sha256"] == _digest(
        output / "inputs" / "input.dmo"
    )
    provenance = read_canonical_json(
        output / "inputs" / "input.dmo.seed.json"
    )
    assert provenance["base_dmo"]["path"] == "base/inputs/input.dmo"
    assert provenance["derived_dmo"]["path"] == (
        "slot-001/inputs/input.dmo"
    )
    assert provenance["random_seed"] == expected_seed
    assert set(provenance["changed_byte_offsets"]).issubset(
        set(
            range(
                distribution.DMO_RANDOM_SEED_OFFSET,
                distribution.DMO_RANDOM_SEED_OFFSET
                + distribution.DMO_RANDOM_SEED_BYTES,
            )
        )
    )
    receipt = read_canonical_json(output / "derivation.json")
    assert receipt["version"] == 3
    assert receipt["pc_slot_ordinal"] == 1
    assert receipt["gameplay_seed"] == expected_seed
    assert receipt["gameplay_seed_transport"] == (
        distribution.GAMEPLAY_SEED_TRANSPORT
    )
    assert receipt["same_dmo_replay_counted_as_distribution"] is False


def test_prepare_campaign_freezes_all_slots_before_data(
    tmp_path: Path,
    monkeypatch,
) -> None:
    base = tmp_path / "base"
    runtime = tmp_path / "runtime.exe"
    source = tmp_path / "ZumasRevenge.exe"
    assets = tmp_path / "assets"
    assets.mkdir()
    _write(runtime, b"runtime")
    _write(source, b"source")
    monkeypatch.setattr(manager, "EXPECTED_RUNTIME_SHA256", _digest(runtime))
    monkeypatch.setattr(
        manager,
        "_strict_font_cache_manifest_receipt",
        lambda changedir, dmo: {
            "entry_count": 35,
            "manifest_sha256": "sha256:" + "1" * 64,
            "main_pak_sha256": "sha256:" + "2" * 64,
        },
    )
    base_dmo = bytearray(b"D" * 64)
    struct.pack_into("<I", base_dmo, distribution.DMO_RANDOM_SEED_OFFSET, 7)
    _write(base / "inputs" / "input.dmo", bytes(base_dmo))
    _write(base / "protocol" / "pre-template.json", {"pre": True})
    _write(base / "safety" / "host-pre.json", {"host": True})
    _write(
        base / "plan.json",
        {
            "schema": PLAN_SCHEMA,
            "version": PLAN_VERSION,
            "dmo": {
                "artifact": "inputs/input.dmo",
                "bytes": len(base_dmo),
                "product_version": "test",
                "random_seed": 7,
                "length_updates": 10,
                "sha256": _digest(base / "inputs" / "input.dmo"),
            },
            "runtime": {
                "runtime_source_executable": str(source),
                "runtime_source_sha256": _digest(source),
                "expected_runtime_sha256": _digest(runtime),
            },
            "trace": {},
            "prestate": {
                "template_artifact": "protocol/pre-template.json",
            },
            "safety": {"host_pre_artifact": "safety/host-pre.json"},
        },
    )

    def fake_demo(payload: bytes) -> SimpleNamespace:
        return SimpleNamespace(
            random_seed=struct.unpack_from(
                "<I", payload, distribution.DMO_RANDOM_SEED_OFFSET
            )[0],
            version=2,
            product_version="test",
            length_updates=10,
            markers=(),
            commands=(),
        )

    monkeypatch.setattr(
        manager.PopCapDemo,
        "from_bytes",
        staticmethod(fake_demo),
    )
    output = tmp_path / "campaign"
    campaign_path = manager.prepare_distribution_campaign(
        base_root=base,
        output_root=output,
        direct_runtime=runtime,
        changedir=assets,
        frozen_utc="2026-08-09T00:00:00Z",
    )

    campaign = read_canonical_json(campaign_path)
    verification = read_canonical_json(
        output / campaign["preregistration_verification"]["path"]
    )
    schedule = read_canonical_json(
        output / campaign["seed_schedule"]["path"]
    )
    jobs = read_canonical_json(
        output / campaign["collection_jobs"]["path"]
    )
    preregistration = read_canonical_json(
        output / campaign["preregistration"]["path"]
    )
    assert campaign["status"] == "FROZEN_BEFORE_DATA"
    assert campaign["pc_samples_observed"] == 0
    assert campaign["pc_dataset"] is None
    assert campaign["audit"] is None
    assert verification["status"] == "PASS"
    assert verification["pc_slot_count"] == distribution.MINIMUM_PC_SAMPLES
    assert len(jobs["jobs"]) == distribution.MINIMUM_PC_SAMPLES
    assert jobs["replay_boundary_tool"] == preregistration["pc_capture"][
        "replay_boundary_tool"
    ]
    assert (
        preregistration["pc_capture"]["replay_boundary_tool"]["path"]
        == "frozen/tools/popcap_replay_boundary.py"
    )
    assert all(job["status"] == "NOT_STARTED" for job in jobs["jobs"])
    assert all(
        job["arguments"][job["arguments"].index("--maximum-attempts") + 1]
        == "1"
        for job in jobs["jobs"]
    )
    assert len(set(schedule["pc_seeds"])) == distribution.MINIMUM_PC_SAMPLES
    assert not set(schedule["pc_seeds"]) & set(schedule["simulator_seeds"])
    assert list((output / "captures").iterdir()) == []


def test_invalidate_failed_campaign_binds_exact_pre_sample_race(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "campaign"
    root.mkdir()
    (root / "captures").mkdir()
    (root / "transactions").mkdir()
    nonce = "1" * 32
    state_root = "sha256:" + "a" * 64
    campaign_id = "failed-campaign"
    successor_id = "successor-campaign"

    slots = []
    jobs = []
    for ordinal in range(1, distribution.MINIMUM_PC_SAMPLES + 1):
        name = f"pc-{ordinal:03d}"
        plan_rel = f"plans/{name}/plan.json"
        dmo_rel = f"plans/{name}/inputs/input.dmo"
        plan_binding = {
            "path": plan_rel,
            "sha256": "sha256:" + "0" * 64,
        }
        dmo_binding = {
            "path": dmo_rel,
            "sha256": "sha256:" + "0" * 64,
        }
        slots.append(
            {
                "ordinal": ordinal,
                "gameplay_seed": ordinal,
                "collector_plan": plan_binding,
                "dmo": dmo_binding,
                "source_manifest_path": f"captures/{name}/manifest.json",
                "output_root_path": f"captures/{name}",
            }
        )
        jobs.append(
            {
                "ordinal": ordinal,
                "status": "NOT_STARTED",
                "gameplay_seed": ordinal,
                "plan": plan_rel,
                "prestate": f"plans/{name}/protocol/pre-template.json",
                "host_restore": f"plans/{name}/safety/host-pre.json",
                "output_root": f"captures/{name}",
                "arguments": ["--maximum-attempts", "1"],
            }
        )

    first = root / "plans" / "pc-001"
    _write(first / "plan.json", {"session_nonce": nonce})
    _write(first / "inputs" / "input.dmo", b"dmo")
    _write(first / "protocol" / "pre-template.json", {})
    _write(first / "safety" / "host-pre.json", {})
    slots[0]["collector_plan"] = {
        "path": "plans/pc-001/plan.json",
        "sha256": _digest(first / "plan.json"),
    }
    slots[0]["dmo"] = {
        "path": "plans/pc-001/inputs/input.dmo",
        "sha256": _digest(first / "inputs" / "input.dmo"),
    }

    capture = root / "captures" / "pc-001"
    attempt_root = capture / "attempt-01"
    attempt_root.mkdir(parents=True)
    attempt = {
        "attempt": 1,
        "error": "natural_strict_trace_process_failed",
        "error_type": "ProbeError",
        "finished_perf_counter_ns": 500,
        "process_id": 55,
        "started_perf_counter_ns": 100,
        "status": "RETRY",
    }
    _write(capture / "attempts.json", [attempt])
    _write(
        attempt_root / "external-input-guard.json",
        {
            "schema": "zuma-rl.pc-external-input-guard",
            "version": 1,
            "status": "PASS",
            "coverage": {
                "start_perf_counter_ns": 101,
                "end_perf_counter_ns": 501,
            },
            "event_counts": {"external": 0},
            "external_events": [],
            "protocol_errors": [],
        },
    )
    dmo = first / "inputs" / "input.dmo"
    _write(
        attempt_root / "strict-replay.json",
        {
            "schema": "zuma.popcap_strict_replay.v3",
            "runtime_process_id": 55,
            "source_dmo": {
                "path": str(dmo.resolve()),
                "sha256": _digest(dmo).removeprefix("sha256:"),
            },
            "options": {
                "direct_natural_seed": True,
                "broker_service_blocks": True,
                "attach_at_update": 0,
                "stop_after_update": 3151,
                "startup_priority_bias_until_update": None,
                "startup_process_affinity_mask": None,
                "rng_seed_override_count": 0,
            },
            "startup_rng": {
                "mode": "retail_natural_seed_observation",
                "process_id": 55,
                "register_override": None,
                "rng_process_memory_writes": 0,
                "persistent_file_modified": False,
            },
            "result": {
                "failure_count": 1,
                "boundary_failures": [
                    "service payload forced read changed its prepared command"
                ],
                "broker_failures": [],
                "last_update": 390,
                "stopped_at_update": False,
                "stopped_at_command_order": False,
            },
        },
    )
    (attempt_root / "strict-replay.log").write_text(
        "offline_boundary_failure update=390 "
        "detail=service payload forced read changed its prepared command\n",
        encoding="utf-8",
    )

    _write(root / "transactions" / "pc-001-pre.json", {})
    _write(root / "transactions" / "pc-001-post.json", {})
    _write(
        root / "transactions" / "pc-001-failure.json",
        {
            "schema": manager.SLOT_EXECUTION_RECEIPT_SCHEMA,
            "version": manager.SLOT_EXECUTION_RECEIPT_VERSION,
            "status": "FAIL",
            "ordinal": 1,
            "collector_exit_code": 1,
            "pre_state_root": state_root,
            "post_state_root": state_root,
        },
    )
    _write(root / "preregistration-verification.json", {"status": "PASS"})
    _write(root / "collection-jobs.json", {"jobs": jobs})
    campaign = {
        "campaign_id": campaign_id,
        "preregistration_verification": {
            "path": "preregistration-verification.json",
            "sha256": _digest(root / "preregistration-verification.json"),
        },
        "collection_jobs": {
            "path": "collection-jobs.json",
            "sha256": _digest(root / "collection-jobs.json"),
        },
    }
    _write(root / "campaign.json", campaign)

    snapshot = SimpleNamespace(session_nonce=nonce, state_root=state_root)
    monkeypatch.setattr(
        manager.PcStateSnapshot,
        "read",
        staticmethod(lambda path: snapshot),
    )
    context = {
        "root": root.resolve(),
        "campaign": campaign,
        "slots": slots,
        "jobs": jobs,
    }
    monkeypatch.setattr(
        manager,
        "_load_campaign_context",
        lambda campaign_root, require_current_collection_tools=True: context,
    )

    path = manager.invalidate_failed_distribution_campaign(
        campaign_root=root,
        superseded_by_campaign_id=successor_id,
        invalidated_utc="2026-08-09T01:00:00Z",
    )

    receipt = read_canonical_json(path)
    assert receipt["status"] == "INVALIDATED"
    assert receipt["classification"] == (
        manager.PRE_SAMPLE_INFRASTRUCTURE_FAILURE
    )
    assert receipt["formal_samples_observed"] == 0
    assert receipt["superseded_by_campaign_id"] == successor_id
    assert receipt["failed_slot"]["last_framework_update"] == 390
    assert receipt["state_transaction"]["restored"] is True
    assert receipt["root_cause"]["code"] == manager.BROKER_RACE_FAILURE_CODE
    with pytest.raises(manager.DistributionCampaignError):
        manager.invalidate_failed_distribution_campaign(
            campaign_root=root,
            superseded_by_campaign_id=successor_id,
        )


def _rebased_late_payload_failure_fixture(
    tmp_path: Path,
) -> tuple[dict[str, object], str, dict[str, object], Path, dict[str, str]]:
    dmo = tmp_path / "input.dmo"
    _write(dmo, b"dmo")
    process_id = 55
    claim = {
        "process_id": process_id,
        "mechanism": "broker_adjacent_current_update_file_write",
        "framework_update": 387,
        "row_index": 61,
        "row_start": 1739,
        "row_end": 1750,
        "row_update": 387,
        "command_order": 61,
        "command_order_offset": 0,
        "prior_brokered_row_index": 60,
        "prior_brokered_command_order": 60,
        "prior_brokered_read_bit_position": 1739,
        "file_write_font_cache_member": (
            "cached\\fonts\\600\\shagexotica32_normal.txt.cfw2"
        ),
        "file_write_argument_size": 15623,
        "manifest_expected_size": 15623,
        "manifest_entry_count": 35,
        "recorded_success": False,
        "payload_completion_verified": True,
        "natural_payload_consumer": True,
        "payload_completion_framework_update": 387,
        "payload_completion_command_order": 61,
        "payload_completion_command_bit_position": 1739,
        "payload_completion_buffer_read_bit_position": 1750,
        "thread_id": 77,
        "payload_completion_thread_id": 88,
    }
    strict: dict[str, object] = {
        "schema": "zuma.popcap_strict_replay.v3",
        "runtime_process_id": process_id,
        "source_dmo": {
            "path": str(dmo.resolve()),
            "sha256": _digest(dmo).removeprefix("sha256:"),
        },
        "options": {
            "direct_natural_seed": True,
            "broker_service_blocks": True,
            "attach_at_update": 0,
            "stop_after_update": 3151,
            "startup_priority_bias_until_update": None,
            "startup_process_affinity_mask": None,
            "rng_seed_override_count": 0,
        },
        "startup_rng": {
            "mode": "retail_natural_seed_observation",
            "process_id": process_id,
            "register_override": None,
            "rng_process_memory_writes": 0,
            "persistent_file_modified": False,
        },
        "result": {
            "failure_count": 1,
            "boundary_failures": [
                "preloading failed file-write tail prefetch command_order "
                "mismatch: 67 != 66"
            ],
            "broker_failures": [],
            "last_update": 390,
            "stopped_at_update": False,
            "stopped_at_command_order": False,
            "command_order_offset": 4,
            "command_order_rebase_rows": [64, 65, 68, 69],
            "service_file_write_header_claims": [claim],
        },
    }
    strict_log = "\n".join(
        (
            "service_file_write_header_claim row=61 update=387",
            "service_file_write_payload_completion row=61 update=387",
            "service_command_order_rebase tid=88 offset=2 "
            "accounted_rows=64,65",
            "service_command_order_rebase tid=88 offset=4 "
            "accounted_rows=64,65,68,69",
            "service_payload_reentry tid=88 update=390 row=70 "
            "cmd_bitpos=1848 read_bitpos=1858",
            "offline_boundary_failure detail=preloading failed file-write "
            "tail prefetch command_order mismatch: 67 != 66",
        )
    )
    return (
        strict,
        strict_log,
        {"process_id": process_id},
        dmo,
        {"path": "input.dmo", "sha256": _digest(dmo)},
    )


def test_validate_rebased_late_payload_failure_trace_accepts_exact_pc008(
    tmp_path: Path,
) -> None:
    strict, strict_log, attempt, dmo, dmo_binding = (
        _rebased_late_payload_failure_fixture(tmp_path)
    )

    claim = manager._validate_rebased_late_payload_failure_trace(
        strict=strict,
        strict_log=strict_log,
        attempt=attempt,
        dmo_path=dmo,
        dmo_binding=dmo_binding,
    )

    assert claim["mechanism"] == "broker_adjacent_current_update_file_write"
    assert claim["payload_completion_verified"] is True


def test_validate_rebased_late_payload_failure_trace_rejects_wrong_registry(
    tmp_path: Path,
) -> None:
    strict, strict_log, attempt, dmo, dmo_binding = (
        _rebased_late_payload_failure_fixture(tmp_path)
    )
    mutated = deepcopy(strict)
    mutated["result"]["command_order_rebase_rows"] = [64, 65, 67, 69]

    with pytest.raises(
        manager.DistributionCampaignError,
        match="failed_slot_trace_not_exact_rebased_tail",
    ):
        manager._validate_rebased_late_payload_failure_trace(
            strict=mutated,
            strict_log=strict_log,
            attempt=attempt,
            dmo_path=dmo,
            dmo_binding=dmo_binding,
        )


def _tail_short_header_failure_fixture(
    tmp_path: Path,
) -> tuple[dict[str, object], str, dict[str, object], Path, dict[str, str]]:
    strict, _, attempt, dmo, dmo_binding = (
        _rebased_late_payload_failure_fixture(tmp_path)
    )
    observed_failure = (
        "service exit continuation is unaudited: kind=11, "
        "cmd_bitpos=1882, read_bitpos=1888, needs=0, short=1, "
        "num=1, order=68"
    )
    result = strict["result"]
    assert isinstance(result, dict)
    result["boundary_failures"] = [observed_failure]
    result["last_update"] = 401
    strict_log = "\n".join(
        (
            "service_file_write_header_claim row=61 update=387",
            "service_file_write_payload_completion row=61 update=387",
            "service_command_order_rebase tid=88 offset=2 "
            "accounted_rows=64,65",
            "service_command_order_rebase tid=88 offset=4 "
            "accounted_rows=64,65,68,69",
            "service_payload_reentry tid=88 update=390 row=70 "
            "cmd_bitpos=1848 read_bitpos=1858",
            "preloading_failed_file_write_tail_prefetch tid=88 "
            "update=390 row=73 cmd_bitpos=1848 read_bitpos=1882",
            "offline_boundary_failure tid=88 update=401 "
            f"cmd_bitpos=1882 detail={observed_failure}",
        )
    )
    return strict, strict_log, attempt, dmo, dmo_binding


def test_validate_tail_short_header_failure_trace_accepts_exact_pc007(
    tmp_path: Path,
) -> None:
    strict, strict_log, attempt, dmo, dmo_binding = (
        _tail_short_header_failure_fixture(tmp_path)
    )

    claim = manager._validate_tail_short_header_failure_trace(
        strict=strict,
        strict_log=strict_log,
        attempt=attempt,
        dmo_path=dmo,
        dmo_binding=dmo_binding,
    )

    assert claim["mechanism"] == "broker_adjacent_current_update_file_write"
    assert claim["payload_completion_verified"] is True


def test_validate_tail_short_header_failure_trace_rejects_executed_payload(
    tmp_path: Path,
) -> None:
    strict, strict_log, attempt, dmo, dmo_binding = (
        _tail_short_header_failure_fixture(tmp_path)
    )
    mutated = deepcopy(strict)
    mutated["result"][
        "preloading_failed_file_write_tail_short_header_recoveries"
    ] = []

    with pytest.raises(
        manager.DistributionCampaignError,
        match="failed_slot_trace_not_exact_tail_short_header",
    ):
        manager._validate_tail_short_header_failure_trace(
            strict=mutated,
            strict_log=strict_log,
            attempt=attempt,
            dmo_path=dmo,
            dmo_binding=dmo_binding,
        )


def _post_sample_finalization_oserror_fixture(
    tmp_path: Path,
) -> tuple[dict[str, object], str, dict[str, object], Path, dict[str, str]]:
    dmo = tmp_path / "input.dmo"
    _write(dmo, b"dmo")
    process_id = 22492
    attempt = {
        "attempt": 1,
        "status": "RETRY",
        "process_id": process_id,
        "started_perf_counter_ns": 100,
        "finished_perf_counter_ns": 200,
        "error_type": "OSError",
        "error": "[Errno 22] Invalid argument",
    }
    strict: dict[str, object] = {
        "schema": "zuma.popcap_strict_replay.v3",
        "runtime_process_id": process_id,
        "source_dmo": {
            "path": str(dmo.resolve()),
            "sha256": _digest(dmo).removeprefix("sha256:"),
        },
        "options": {
            "direct_natural_seed": True,
            "broker_service_blocks": True,
            "attach_at_update": 0,
            "stop_after_update": 3151,
            "startup_priority_bias_until_update": None,
            "startup_process_affinity_mask": None,
            "rng_seed_override_count": 0,
        },
        "startup_rng": {
            "mode": "retail_natural_seed_observation",
            "process_id": process_id,
            "register_override": None,
            "rng_process_memory_writes": 0,
            "persistent_file_modified": False,
        },
        "result": {
            "failure_count": 0,
            "boundary_failures": [],
            "broker_failures": [],
            "last_update": 3151,
            "stopped_at_update": True,
            "stopped_at_command_order": False,
            "command_order_offset": 0,
            "command_order_rebase_rows": [],
            "service_file_write_header_claims": [],
            "preloading_failed_file_write_tail_short_header_recoveries": [],
            "service_exit_timeline_commits": [],
            "debug_exception_observations": [],
            "service_continuation_verifications": [
                {
                    "process_id": process_id,
                    "framework_update": 406,
                    "row_index": 72,
                    "corridor_start_index": 45,
                    "corridor_end_index": 71,
                    "reentries": 3,
                    "command_order_offset": 0,
                    "service_exit_timeline_commit_count": 0,
                }
            ],
        },
    }
    strict_log = "\n".join(
        (
            "service_continuation_verified tid=1 update=406 row=72 reentries=3",
            "result hits=478 broker_failures=0 boundary_failures=0 "
            "last_update=3151 stopped_at_update=True command_order_offset=0 "
            "preloading_failed_file_write_tail_short_header_recoveries=0 "
            "service_file_write_header_claims=0",
        )
    )
    return (
        strict,
        strict_log,
        attempt,
        dmo,
        {"path": "input.dmo", "sha256": _digest(dmo)},
    )


def test_validate_post_sample_finalization_oserror_accepts_exact_pc018(
    tmp_path: Path,
) -> None:
    strict, strict_log, attempt, dmo, dmo_binding = (
        _post_sample_finalization_oserror_fixture(tmp_path)
    )

    continuation = manager._validate_post_sample_finalization_oserror_trace(
        strict=strict,
        strict_log=strict_log,
        attempt=attempt,
        dmo_path=dmo,
        dmo_binding=dmo_binding,
    )

    assert continuation["framework_update"] == 406
    assert continuation["row_index"] == 72


@pytest.mark.parametrize(
    ("mutation_path", "value"),
    (
        (("result", "failure_count"), 1),
        (("result", "preloading_failed_file_write_tail_short_header_recoveries"), [{}]),
        (("attempt", "error"), "[Errno 5] Access denied"),
    ),
)
def test_validate_post_sample_finalization_oserror_fails_closed(
    tmp_path: Path,
    mutation_path: tuple[str, str],
    value: object,
) -> None:
    strict, strict_log, attempt, dmo, dmo_binding = (
        _post_sample_finalization_oserror_fixture(tmp_path)
    )
    owner, field = mutation_path
    if owner == "result":
        mutated_result = strict["result"]
        assert isinstance(mutated_result, dict)
        mutated_result[field] = value
    else:
        attempt[field] = value

    with pytest.raises(
        manager.DistributionCampaignError,
        match="failed_slot_trace_not_exact_post_sample_oserror",
    ):
        manager._validate_post_sample_finalization_oserror_trace(
            strict=strict,
            strict_log=strict_log,
            attempt=attempt,
            dmo_path=dmo,
            dmo_binding=dmo_binding,
        )
