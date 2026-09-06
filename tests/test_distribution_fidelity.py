"""Tests for the preregistered stochastic-distribution lane."""

from __future__ import annotations

import hashlib
from pathlib import Path
import struct
from types import SimpleNamespace

import pytest

from zuma_rl import distribution_fidelity as distribution
from zuma_rl.pc_exact_step_evidence import canonical_json_bytes, read_canonical_json
from zuma_rl.pc_source import (
    STRICT_REPLAY_BINDING_SCHEMA,
    STRICT_REPLAY_BINDING_VERSION,
)


POLICY = "original-transfer-jungle2-v2"


def test_distribution_gate_uses_authoritative_strict_replay_binding() -> None:
    assert (
        distribution.STRICT_NATURAL_REPLAY_BINDING_SCHEMA
        == STRICT_REPLAY_BINDING_SCHEMA
    )
    assert (
        distribution.STRICT_NATURAL_REPLAY_BINDING_VERSION
        == STRICT_REPLAY_BINDING_VERSION
    )


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_bytes(canonical_json_bytes(value))


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _binding(path: Path, root: Path) -> dict[str, str]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _digest(path),
    }


def _scope() -> dict[str, object]:
    return {
        "policy": POLICY,
        "environment_id": "ZumaRevenge-v0",
        "level_id": "Jungle2",
        "hard": False,
        "profile_mode": "tutorials_completed",
    }


class _FakeSimulator:
    forced_pair: tuple[int, int] | None = None

    def __init__(self) -> None:
        self.curve_count = 1
        self.num_colors = distribution.NUM_COLORS
        _, simulator_seeds = distribution.frozen_seed_schedules()
        self.reset(seed=simulator_seeds[0])

    @classmethod
    def from_installed(cls, *args: object, **kwargs: object) -> "_FakeSimulator":
        return cls()

    def reset(self, *, seed: int) -> None:
        self.seed = seed
        self.tick_count = 0
        self.current_color = None
        self.next_color = None
        self.score = 0
        self.outcome = None

    def tick(self) -> None:
        self.tick_count = 1
        if self.forced_pair is not None:
            self.current_color, self.next_color = self.forced_pair
            return
        _, simulator_seeds = distribution.frozen_seed_schedules()
        index = simulator_seeds.index(self.seed)
        self.current_color = index % distribution.NUM_COLORS
        self.next_color = (
            (index // distribution.NUM_COLORS) % distribution.NUM_COLORS
        )


def _build_audit(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    simulator_pair: tuple[int, int] | None = None,
    maximum_total_variation: float = distribution.MAXIMUM_TOTAL_VARIATION,
    bad_pc_sample: bool = False,
    bad_strict_replay: bool = False,
) -> tuple[Path, Path]:
    original_root = root / "original"
    original_root.mkdir()
    (original_root / "ZumasRevenge.exe").write_bytes(b"retail")
    dmo = root / "inputs" / "base-input.dmo"
    runtime = root / "tools" / "runtime.exe"
    collector = root / "tools" / "collect_pc_memory_probe.py"
    trace_tool = root / "tools" / "trace_popcap_demo_commands.py"
    replay_boundary_tool = root / "tools" / "popcap_replay_boundary.py"
    seed_deriver = root / "tools" / "manage_startup_distribution.py"
    collector_plan = root / "inputs" / "base-plan.json"
    replay_prestate = root / "inputs" / "prestate.json"
    dmo.parent.mkdir()
    runtime.parent.mkdir()
    base_dmo_bytes = bytearray(b"D" * 64)
    struct.pack_into("<I", base_dmo_bytes, distribution.DMO_RANDOM_SEED_OFFSET, 7)
    dmo.write_bytes(base_dmo_bytes)
    runtime.write_bytes(b"runtime")
    collector.write_bytes(b"collector")
    trace_tool.write_bytes(b"trace")
    replay_boundary_tool.write_bytes(b"replay-boundary")
    seed_deriver.write_bytes(b"seed-deriver")
    _write(collector_plan, {"plan": "frozen"})
    _write(replay_prestate, {"state": "frozen"})
    dmo_binding = _binding(dmo, root)
    runtime_binding = _binding(runtime, root)
    monkeypatch.setattr(
        distribution.PopCapDemo,
        "from_bytes",
        staticmethod(
            lambda data: SimpleNamespace(
                random_seed=struct.unpack_from(
                    "<I", data, distribution.DMO_RANDOM_SEED_OFFSET
                )[0]
            )
        ),
    )
    pc_seeds, simulator_seeds = distribution.frozen_seed_schedules()

    slots: list[dict[str, object]] = []
    pc_rows: list[dict[str, object]] = []
    reports: dict[Path, dict[str, object]] = {}
    for ordinal in range(1, distribution.MINIMUM_PC_SAMPLES + 1):
        source_id = f"startup-dist-v2-pc-{ordinal:03d}"
        run_id = f"startup-dist-v2-run-{ordinal:03d}"
        source_root = root / "sources" / source_id
        gameplay_seed = pc_seeds[ordinal - 1]
        seeded_dmo = source_root / "inputs" / "input.dmo"
        seeded_bytes = bytearray(base_dmo_bytes)
        struct.pack_into(
            "<I",
            seeded_bytes,
            distribution.DMO_RANDOM_SEED_OFFSET,
            gameplay_seed,
        )
        _write(seeded_dmo, bytes(seeded_bytes))
        seeded_dmo_binding = _binding(seeded_dmo, root)
        provenance = source_root / "inputs" / "input.dmo.seed.json"
        changed_offsets = [
            index
            for index, (before, after) in enumerate(
                zip(base_dmo_bytes, seeded_bytes, strict=True)
            )
            if before != after
        ]
        _write(
            provenance,
            {
                "schema": distribution.SEEDED_DMO_PROVENANCE_SCHEMA,
                "version": distribution.SEEDED_DMO_PROVENANCE_VERSION,
                "status": "PASS",
                "base_dmo": dmo_binding,
                "derived_dmo": seeded_dmo_binding,
                "random_seed": gameplay_seed,
                "seed_offset": distribution.DMO_RANDOM_SEED_OFFSET,
                "seed_size_bytes": distribution.DMO_RANDOM_SEED_BYTES,
                "changed_byte_offsets": changed_offsets,
                "unchanged_byte_count": len(base_dmo_bytes)
                - len(changed_offsets),
            },
        )
        plan = source_root / "plan.json"
        _write(
            plan,
            {
                "schema": "zuma-rl.pc-golden-v4-collection-plan",
                "version": 5,
                "dmo": {
                    "artifact": "inputs/input.dmo",
                    "sha256": seeded_dmo_binding["sha256"],
                    "random_seed": gameplay_seed,
                },
                "runtime": {
                    "launch_mode": (
                        "direct_byte_identical_natural_seed_"
                        "strict_command_broker"
                    ),
                    "runtime_executable_sha256": runtime_binding["sha256"],
                    "expected_runtime_sha256": runtime_binding["sha256"],
                    "crt_rand_seed": None,
                    "startup_seed_transport": None,
                    "board_seed_call_address": None,
                    "board_seed": None,
                    "global_rng_seed": None,
                    "thread_crt_rng_seed": None,
                },
                "trace": {
                    "attach_at_update": 0,
                    "startup_trace_handoff": True,
                    "allow_pre_stream_commands": True,
                    "allow_font_cache_manifest_completion_debt": True,
                    "allow_pre_attach_file_write_debt": False,
                    "seed_board_before_attach": False,
                    "natural_command_broker_stop_after_update": 3151,
                    "font_cache_manifest_receipt": {"entry_count": 35},
                },
            },
        )
        attempts = source_root / "attempts.json"
        memory_probe = source_root / "attempt-01" / "memory-probe.json"
        trajectory_index = (
            source_root / "attempt-01" / "trajectory" / "index.json"
        )
        process_id = 10_000 + ordinal
        filetime = 100_000 + ordinal
        _write(
            attempts,
            [
                {
                    "attempt": 1,
                    "status": "PASS",
                    "process_id": process_id,
                    "started_perf_counter_ns": 1_000 + ordinal * 10,
                    "finished_perf_counter_ns": 1_005 + ordinal * 10,
                }
            ],
        )
        _write(memory_probe, {"artifact": "memory_probe"})
        _write(trajectory_index, {"artifact": "trajectory_index"})
        manifest = source_root / "manifest.json"
        _write(
            manifest,
            {
                "schema": "zuma-rl.pc-source-manifest",
                "version": 1,
                "source_id": source_id,
                "dmo": seeded_dmo_binding,
                "runtime_payload": runtime_binding,
                "window": {
                    "freeze_update": distribution.PC_FREEZE_UPDATE,
                    "start_update": distribution.PC_SAMPLE_UPDATE,
                    "end_update": distribution.PC_SAMPLE_UPDATE,
                    "warmup_tick_count": (
                        distribution.PC_WARMUP_TICK_COUNT
                    ),
                },
                "run": {
                    "run_id": run_id,
                    "selected_attempt": 1,
                    "maximum_startup_attempts": 1,
                    "process_id": process_id,
                    "process_creation_filetime_100ns": filetime,
                    "attempts": _binding(attempts, root),
                    "memory_probe": _binding(memory_probe, root),
                    "trajectory_index": _binding(trajectory_index, root),
                },
            },
        )
        slots.append(
            {
                "ordinal": ordinal,
                "source_id": source_id,
                "run_id": run_id,
                "source_manifest_path": manifest.relative_to(root).as_posix(),
                "output_root_path": source_root.relative_to(root).as_posix(),
                "gameplay_seed": gameplay_seed,
                "dmo": seeded_dmo_binding,
                "seed_provenance": _binding(provenance, root),
                "collector_plan": _binding(plan, root),
            }
        )
        pc_rows.append(
            {
                "ordinal": ordinal,
                "source_manifest": _binding(manifest, root),
            }
        )
        index = ordinal - 1
        reports[manifest.resolve()] = {
            "source_fingerprint": "sha256:" + f"{ordinal:064x}",
            "random_seed": gameplay_seed,
            "strict_command_replay": {
                "schema": distribution.STRICT_NATURAL_REPLAY_BINDING_SCHEMA,
                "version": (
                    distribution.STRICT_NATURAL_REPLAY_BINDING_VERSION
                ),
                "status": "PASS",
                "source_dmo_sha256": seeded_dmo_binding["sha256"],
                "runtime_executable_sha256": runtime_binding["sha256"],
                "runtime_process_id": process_id,
                "register_override": None,
                "rng_seed_override_count": 0,
                "rng_process_memory_writes": (
                    1 if bad_strict_replay and ordinal == 1 else 0
                ),
                "stop_after_update": 3151,
                "failure_count": 0,
            },
            "independent_process_identity": {
                "process_id": process_id,
                "process_creation_filetime_100ns": filetime,
            },
            "actor_visible_first_frame": {
                "framework_update": distribution.PC_SAMPLE_UPDATE,
                "score": (
                    distribution.PC_EXPECTED_SCORE + 1
                    if bad_pc_sample and ordinal == 1
                    else distribution.PC_EXPECTED_SCORE
                ),
                "displayed_score": distribution.PC_EXPECTED_SCORE,
                "chain_ball_count": 0,
                "inserting_ball_count": 0,
                "fired_bullet_count": 0,
                "current_ball_id": ordinal * 2,
                "current_color_id": index % distribution.NUM_COLORS,
                "next_ball_id": ordinal * 2 + 1,
                "next_color_id": (
                    (index // distribution.NUM_COLORS)
                    % distribution.NUM_COLORS
                ),
                "board_color_counts": [0, 0, 0, 0, 0, 0],
            },
        }

    monkeypatch.setattr(
        distribution,
        "verify_pc_source_manifest",
        lambda path, **kwargs: reports[Path(path).resolve()],
    )
    _FakeSimulator.forced_pair = simulator_pair
    monkeypatch.setattr(distribution, "RevengeSimulator", _FakeSimulator)

    preregistration = root / "preregistration.json"
    _write(
        preregistration,
        {
            "schema": distribution.PREREGISTRATION_SCHEMA,
            "version": distribution.PREREGISTRATION_VERSION,
            "status": "FROZEN_BEFORE_DATA",
            "frozen_utc": "2026-08-09T00:00:00Z",
            "scope": _scope(),
            "metric": {
                "name": distribution.METRIC_NAME,
                "kind": "categorical_pair",
                "minimum_pc_samples": distribution.MINIMUM_PC_SAMPLES,
                "minimum_simulator_samples": (
                    distribution.MINIMUM_SIMULATOR_SAMPLES
                ),
                "num_colors": distribution.NUM_COLORS,
                "maximum_total_variation": maximum_total_variation,
                "maximum_excess_total_variation": (
                    distribution.MAXIMUM_EXCESS_TOTAL_VARIATION
                ),
                "minimum_compatibility_p_value": (
                    distribution.MINIMUM_COMPATIBILITY_P_VALUE
                ),
                "permutation_count": distribution.PERMUTATION_COUNT,
                "permutation_seed": distribution.PERMUTATION_SEED,
            },
            "pc_capture": {
                "freeze_update": distribution.PC_FREEZE_UPDATE,
                "sample_update": distribution.PC_SAMPLE_UPDATE,
                "warmup_tick_count": distribution.PC_WARMUP_TICK_COUNT,
                "expected_score": distribution.PC_EXPECTED_SCORE,
                "base_dmo": dmo_binding,
                "runtime_payload": runtime_binding,
                "base_collector_plan": _binding(collector_plan, root),
                "replay_prestate": _binding(replay_prestate, root),
                "collector": _binding(collector, root),
                "trace_tool": _binding(trace_tool, root),
                "replay_boundary_tool": _binding(
                    replay_boundary_tool,
                    root,
                ),
                "seed_deriver": _binding(seed_deriver, root),
                "gameplay_seed_transport": (
                    distribution.GAMEPLAY_SEED_TRANSPORT
                ),
                "strict_command_replay_required": True,
            },
            "selection": {
                "campaign_id": "startup-actor-distribution-v3",
                "pc_process_count": distribution.MINIMUM_PC_SAMPLES,
                "pc_slots": slots,
                "seed_schedule": {
                    "id": distribution.SEED_SCHEDULE_ID,
                    "algorithm": distribution.SEED_SCHEDULE_ALGORITHM,
                    "encoding": distribution.SEED_SCHEDULE_ENCODING,
                    "pc_seed_count": distribution.MINIMUM_PC_SAMPLES,
                    "pc_seed_sha256": distribution.seed_sequence_sha256(
                        pc_seeds
                    ),
                    "simulator_seed_count": (
                        distribution.MINIMUM_SIMULATOR_SAMPLES
                    ),
                    "simulator_seed_sha256": (
                        distribution.seed_sequence_sha256(simulator_seeds)
                    ),
                    "populations_disjoint": True,
                },
                "simulator_seed_count": (
                    distribution.MINIMUM_SIMULATOR_SAMPLES
                ),
                "simulator_sample_tick": distribution.SIMULATOR_SAMPLE_TICK,
                "retain_every_started_pc_process": True,
                "replacement_after_observation_forbidden": True,
            },
        },
    )

    pc_dataset = root / "pc.json"
    _write(
        pc_dataset,
        {
            "schema": distribution.DATASET_SCHEMA,
            "version": distribution.DATASET_VERSION,
            "population": "pc",
            "metric": distribution.METRIC_NAME,
            "samples": pc_rows,
        },
    )
    simulator_dataset = root / "simulator.json"
    _write(
        simulator_dataset,
        {
            "schema": distribution.DATASET_SCHEMA,
            "version": distribution.DATASET_VERSION,
            "population": "simulator",
            "metric": distribution.METRIC_NAME,
            "samples": [
                {"seed": seed} for seed in simulator_seeds
            ],
        },
    )
    audit = root / "audit.json"
    _write(
        audit,
        {
            "schema": distribution.AUDIT_SCHEMA,
            "version": distribution.AUDIT_VERSION,
            "status": "READY_FOR_RECOMPUTE",
            "scope": _scope(),
            "preregistration": _binding(preregistration, root),
            "pc_dataset": _binding(pc_dataset, root),
            "simulator_dataset": _binding(simulator_dataset, root),
        },
    )
    return audit, original_root


def test_matching_actor_visible_startup_distribution_passes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit, original_root = _build_audit(tmp_path, monkeypatch)

    report = distribution.verify_distribution_audit(
        audit,
        evidence_root=tmp_path,
        original_root=original_root,
        policy_id=POLICY,
    )

    assert report["status"] == "PASS"
    assert report["feature"] == "startup_actor_distribution"
    assert report["total_variation"] == 0.0
    assert report["compatibility_p_value"] == 1.0
    assert report["raw_samples_recomputed"] is True
    assert len(report["source_fingerprints"]) == 32


def test_mechanism_probe_seed_is_excluded_from_formal_populations() -> None:
    pc_seeds, simulator_seeds = distribution.frozen_seed_schedules()
    probe_seed = distribution.mechanism_probe_seed()

    assert probe_seed not in set(pc_seeds)
    assert probe_seed not in set(simulator_seeds)


def test_materially_different_startup_distribution_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit, original_root = _build_audit(
        tmp_path,
        monkeypatch,
        simulator_pair=(3, 3),
    )

    report = distribution.verify_distribution_audit(
        audit,
        evidence_root=tmp_path,
        original_root=original_root,
        policy_id=POLICY,
    )

    assert report["status"] == "FAIL"
    assert report["total_variation"] > report["maximum_total_variation"]
    assert report["compatibility_p_value"] < 0.05


def test_preregistered_threshold_cannot_be_weakened(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit, original_root = _build_audit(
        tmp_path,
        monkeypatch,
        maximum_total_variation=0.99,
    )

    with pytest.raises(
        distribution.DistributionFidelityError,
        match="metric policy differs",
    ):
        distribution.verify_distribution_audit(
            audit,
            evidence_root=tmp_path,
            original_root=original_root,
            policy_id=POLICY,
        )


def test_preregistered_replay_boundary_tool_cannot_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit, original_root = _build_audit(tmp_path, monkeypatch)
    audit_payload = read_canonical_json(audit)
    preregistration_path = tmp_path / audit_payload["preregistration"]["path"]
    preregistration = read_canonical_json(preregistration_path)
    boundary_path = (
        tmp_path
        / preregistration["pc_capture"]["replay_boundary_tool"]["path"]
    )
    boundary_path.write_bytes(b"drifted-replay-boundary")

    with pytest.raises(
        distribution.DistributionFidelityError,
        match="replay_boundary_tool identity differs",
    ):
        distribution.verify_distribution_audit(
            audit,
            evidence_root=tmp_path,
            original_root=original_root,
            policy_id=POLICY,
        )


def test_pc_colour_claim_is_recomputed_from_exact_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit, original_root = _build_audit(
        tmp_path,
        monkeypatch,
        bad_pc_sample=True,
    )

    with pytest.raises(
        distribution.DistributionFidelityError,
        match="sample point differs",
    ):
        distribution.verify_distribution_audit(
            audit,
            evidence_root=tmp_path,
            original_root=original_root,
            policy_id=POLICY,
        )


def test_simulator_dataset_cannot_declare_an_observed_colour(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit_path, original_root = _build_audit(tmp_path, monkeypatch)
    audit = read_canonical_json(audit_path)
    simulator_path = tmp_path / audit["simulator_dataset"]["path"]
    simulator = read_canonical_json(simulator_path)
    simulator["samples"][0]["current_color_id"] = 3
    _write(simulator_path, simulator)
    audit["simulator_dataset"] = _binding(simulator_path, tmp_path)
    _write(audit_path, audit)

    with pytest.raises(
        distribution.DistributionFidelityError,
        match="simulator sample fields are invalid",
    ):
        distribution.verify_distribution_audit(
            audit_path,
            evidence_root=tmp_path,
            original_root=original_root,
            policy_id=POLICY,
        )


def test_pc_dataset_cannot_replace_a_frozen_slot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit_path, original_root = _build_audit(tmp_path, monkeypatch)
    audit = read_canonical_json(audit_path)
    pc_path = tmp_path / audit["pc_dataset"]["path"]
    pc_dataset = read_canonical_json(pc_path)
    pc_dataset["samples"][0]["source_manifest"] = pc_dataset["samples"][1][
        "source_manifest"
    ]
    _write(pc_path, pc_dataset)
    audit["pc_dataset"] = _binding(pc_path, tmp_path)
    _write(audit_path, audit)

    with pytest.raises(
        distribution.DistributionFidelityError,
        match="replaced a frozen source slot",
    ):
        distribution.verify_distribution_audit(
            audit_path,
            evidence_root=tmp_path,
            original_root=original_root,
            policy_id=POLICY,
        )


def test_pc_slot_cannot_change_the_frozen_gameplay_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit_path, original_root = _build_audit(tmp_path, monkeypatch)
    audit = read_canonical_json(audit_path)
    prereg_path = tmp_path / audit["preregistration"]["path"]
    prereg = read_canonical_json(prereg_path)
    prereg["selection"]["pc_slots"][0]["gameplay_seed"] = prereg[
        "selection"
    ]["pc_slots"][1]["gameplay_seed"]
    _write(prereg_path, prereg)
    audit["preregistration"] = _binding(prereg_path, tmp_path)
    _write(audit_path, audit)

    with pytest.raises(
        distribution.DistributionFidelityError,
        match="gameplay seeds differ from the frozen schedule",
    ):
        distribution.verify_distribution_audit(
            audit_path,
            evidence_root=tmp_path,
            original_root=original_root,
            policy_id=POLICY,
        )


def test_seeded_dmo_cannot_hide_a_second_byte_range_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit_path, original_root = _build_audit(tmp_path, monkeypatch)
    audit = read_canonical_json(audit_path)
    prereg_path = tmp_path / audit["preregistration"]["path"]
    prereg = read_canonical_json(prereg_path)
    slot = prereg["selection"]["pc_slots"][0]
    dmo_path = tmp_path / slot["dmo"]["path"]
    tampered = bytearray(dmo_path.read_bytes())
    tampered[20] ^= 1
    dmo_path.write_bytes(tampered)
    slot["dmo"] = _binding(dmo_path, tmp_path)

    provenance_path = tmp_path / slot["seed_provenance"]["path"]
    provenance = read_canonical_json(provenance_path)
    provenance["derived_dmo"] = slot["dmo"]
    provenance["changed_byte_offsets"] = sorted(
        set(provenance["changed_byte_offsets"] + [20])
    )
    provenance["unchanged_byte_count"] -= 1
    _write(provenance_path, provenance)
    slot["seed_provenance"] = _binding(provenance_path, tmp_path)
    _write(prereg_path, prereg)
    audit["preregistration"] = _binding(prereg_path, tmp_path)
    _write(audit_path, audit)

    with pytest.raises(
        distribution.DistributionFidelityError,
        match="changes more than its frozen gameplay seed",
    ):
        distribution.verify_distribution_audit(
            audit_path,
            evidence_root=tmp_path,
            original_root=original_root,
            policy_id=POLICY,
        )


def test_pc_slot_rejects_any_strict_replay_rng_memory_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit_path, original_root = _build_audit(
        tmp_path,
        monkeypatch,
        bad_strict_replay=True,
    )

    with pytest.raises(
        distribution.DistributionFidelityError,
        match="gameplay seed or strict replay differs",
    ):
        distribution.verify_distribution_audit(
            audit_path,
            evidence_root=tmp_path,
            original_root=original_root,
            policy_id=POLICY,
        )
