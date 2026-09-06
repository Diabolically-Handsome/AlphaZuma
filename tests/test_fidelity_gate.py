"""Fail-closed tests for the scope-specific transfer gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from zuma_rl import fidelity_gate
from zuma_rl.fidelity_gate import (
    EvidenceKind,
    FidelityGateStatus,
    FidelityGateValidationError,
    FidelitySuite,
    GatePolicy,
    GateRequirement,
    verify_fidelity_suite,
)
from zuma_rl.pc_golden import ComparisonStatus


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def _test_policy(
    *,
    requirements: tuple[GateRequirement, ...] | None = None,
    pc_cases: int = 1,
    simulator_cases: int = 1,
    random_seeds: int = 1,
) -> GatePolicy:
    return GatePolicy(
        id="test-policy",
        environment_id="ZumaRevenge-v0",
        levels=("Jungle2",),
        hard=False,
        profile_mode="tutorials_completed",
        observation_mode="actor",
        curve_count=1,
        minimum_pc_golden_cases=pc_cases,
        minimum_simulator_diff_cases=simulator_cases,
        minimum_random_seeds=random_seeds,
        requirements=requirements
        or (
            GateRequirement("actor_no_hidden_state", audit=1),
            GateRequirement(
                "shot_release",
                pc_golden=1,
                simulator_diff=1,
            ),
        ),
    )


def _install_test_policy(
    monkeypatch: pytest.MonkeyPatch,
    policy: GatePolicy,
) -> None:
    monkeypatch.setattr(
        fidelity_gate,
        "GATE_POLICIES",
        {policy.id: policy},
    )


def _golden_stub(
    *,
    evidence_digit: str = "1",
    source_digit: str = "2",
    random_seed: int = 12345,
) -> SimpleNamespace:
    artifacts = {
        "input": SimpleNamespace(sha256="sha256:" + source_digit * 64),
        "video-r1": SimpleNamespace(sha256="sha256:" + "3" * 64),
        "metadata-r1": SimpleNamespace(sha256="sha256:" + "4" * 64),
        "updates-r1": SimpleNamespace(sha256="sha256:" + "5" * 64),
        "video-r2": SimpleNamespace(sha256="sha256:" + "6" * 64),
        "metadata-r2": SimpleNamespace(sha256="sha256:" + "7" * 64),
        "updates-r2": SimpleNamespace(sha256="sha256:" + "8" * 64),
    }
    replay_determinism = SimpleNamespace(
        runs=(
            SimpleNamespace(
                video=SimpleNamespace(artifact="video-r1"),
                capture_metadata_artifact="metadata-r1",
                framework_update_artifact="updates-r1",
            ),
            SimpleNamespace(
                video=SimpleNamespace(artifact="video-r2"),
                capture_metadata_artifact="metadata-r2",
                framework_update_artifact="updates-r2",
            ),
        )
    )
    return SimpleNamespace(
        manifest_version=4,
        scenario=SimpleNamespace(
            level_id="Jungle2",
            hard=False,
            profile_mode="tutorials_completed",
            mode="adventure",
        ),
        clock=SimpleNamespace(tick_end=19),
        input_timeline=SimpleNamespace(
            artifact="input",
            random_seed=random_seed,
        ),
        artifacts=artifacts,
        replay_determinism=replay_determinism,
        evidence_set_fingerprint="sha256:" + evidence_digit * 64,
    )


def _golden_verification_stub() -> SimpleNamespace:
    return SimpleNamespace(
        status=ComparisonStatus.PASS,
        checks=(),
        summary={
            "input_sequence_status": "PASS",
            "semantic_limits": {
                "deterministic_replay_attested": True,
                "memory_measurement_provenance_attested": True,
            },
            "replay_determinism": {
                "machine_verifiable_evidence": True,
                "run_count": 2,
                "native_tick_count": 20,
                "full_viewport_rgb24_per_tick_matched": True,
                "normalized_trace_matched": True,
                "dmo_binding_matched": True,
                "independent_processes": True,
                "end_state_roots_matched": True,
            },
            "measurement_provenance": {
                "machine_verifiable_evidence": True,
                "fired_ball_id": 41,
                "fired_color_id": 2,
                "fired_speed": 0.9,
                "gameplay_viewport_pixels_matched": True,
                "video_bindings": [
                    {
                        "framework_update": 10,
                        "native_tick": 5,
                        "video_frame_sequence": 5,
                        "interior_mismatch_count": 0,
                    },
                    {
                        "framework_update": 11,
                        "native_tick": 6,
                        "video_frame_sequence": 6,
                        "interior_mismatch_count": 0,
                    },
                ],
            },
        },
    )


def test_long_horizon_requires_500_verified_native_ticks() -> None:
    short = _golden_verification_stub()
    assert (
        "long_horizon_drift"
        not in fidelity_gate._pc_golden_authorized_features(short)
    )

    long = _golden_verification_stub()
    long.summary["replay_determinism"]["native_tick_count"] = 500
    assert (
        "long_horizon_drift"
        in fidelity_gate._pc_golden_authorized_features(long)
    )


def test_original_pc_source_authentication_is_provenance_class_specific() -> None:
    source_bound_basis = (
        fidelity_gate._SOURCE_BOUND_PC_SOURCE_AUTHENTICATION_BASIS
    )

    def accepted(
        evidence_id: str,
        kind: EvidenceKind,
        dmo_digit: str,
        *,
        is_accepted: bool = True,
        authentication_basis: str | None = None,
    ) -> fidelity_gate._AcceptedEvidence:
        spec = fidelity_gate.EvidenceSpec.from_dict(
            {
                "id": evidence_id,
                "kind": kind.value,
                "path": f"{evidence_id}.json",
                "sha256": "sha256:" + "9" * 64,
                "features": ["shot_release"],
            }
        )
        return fidelity_gate._AcceptedEvidence(
            spec=spec,
            accepted=is_accepted,
            schema="test",
            version=1,
            status="PASS" if is_accepted else "FAIL",
            level_id="Jungle2",
            hard=False,
            profile_mode="tutorials_completed",
            random_seed=1,
            tick_count=1,
            source_fingerprint="sha256:" + evidence_id[-1] * 64,
            authorized_features=("shot_release",),
            unsupported_features=(),
            source_dmo_sha256="sha256:" + dmo_digit * 64,
            original_source_authentication_basis=authentication_basis,
        )

    rows = fidelity_gate._authenticate_original_sources(
        (
            accepted("golden-1", EvidenceKind.PC_GOLDEN, "a"),
            accepted("source-2", EvidenceKind.PC_SOURCE, "a"),
            accepted("source-3", EvidenceKind.PC_SOURCE, "b"),
            accepted(
                "golden-4",
                EvidenceKind.PC_GOLDEN,
                "c",
                is_accepted=False,
            ),
            accepted("source-5", EvidenceKind.PC_SOURCE, "c"),
            accepted(
                "source-6",
                EvidenceKind.PC_SOURCE,
                "d",
                authentication_basis=source_bound_basis,
            ),
            accepted(
                "source-7",
                EvidenceKind.PC_SOURCE,
                "e",
                authentication_basis="unrecognized_basis",
            ),
            accepted(
                "source-8",
                EvidenceKind.PC_SOURCE,
                "f",
                is_accepted=False,
                authentication_basis=source_bound_basis,
            ),
        )
    )

    assert [row.original_source_authenticated for row in rows] == [
        True,
        True,
        False,
        False,
        False,
        True,
        False,
        False,
    ]
    assert [row.original_source_authentication_basis for row in rows] == [
        fidelity_gate._PC_GOLDEN_AUTHENTICATION_BASIS,
        fidelity_gate._SAME_DMO_PC_GOLDEN_AUTHENTICATION_BASIS,
        None,
        None,
        None,
        source_bound_basis,
        None,
        None,
    ]


def test_source_bound_pc_source_self_authentication_contract_is_fail_closed(
) -> None:
    valid = {
        "status": "PASS",
        "transport": fidelity_gate._SOURCE_BOUND_PC_SOURCE_TRANSPORT,
        "cross_process_exact_replay_required": False,
        "strict_command_replay": {
            "schema": fidelity_gate._SOURCE_BOUND_REPLAY_SCHEMA,
            "version": 2,
            "status": "PASS",
            "classification": (
                "source_bound_original_natural_retail_outcome"
            ),
        },
    }

    assert fidelity_gate._pc_source_self_authentication_basis(valid) == (
        fidelity_gate._SOURCE_BOUND_PC_SOURCE_AUTHENTICATION_BASIS
    )

    invalid = []
    for key, value in (
        ("status", "FAIL"),
        ("transport", "single_retail_process_exact_step_source"),
        ("cross_process_exact_replay_required", True),
    ):
        invalid.append({**valid, key: value})
    for key, value in (
        ("schema", "zuma-rl.pc-natural-strict-command-replay-binding"),
        ("version", 3),
        ("status", "FAIL"),
        ("classification", "source_bound_unproven"),
    ):
        invalid.append(
            {
                **valid,
                "strict_command_replay": {
                    **valid["strict_command_replay"],
                    key: value,
                },
            }
        )

    assert all(
        fidelity_gate._pc_source_self_authentication_basis(report) is None
        for report in invalid
    )


def test_v3_repairs_transfer_authority_and_adds_fruit_contract() -> None:
    legacy = fidelity_gate.GATE_POLICIES[
        "original-transfer-jungle2-v1"
    ]
    superseded = fidelity_gate.GATE_POLICIES[
        "original-transfer-jungle2-v2"
    ]
    policy = fidelity_gate.GATE_POLICIES[
        "original-transfer-jungle2-v3"
    ]

    assert legacy.transfer_authorizing is False
    assert superseded.transfer_authorizing is False
    assert policy.transfer_authorizing is True
    assert policy.generation == 3
    features = {requirement.feature for requirement in policy.requirements}
    assert "deterministic_replay" not in features
    assert "startup_actor_distribution" in features
    assert "actor_visual_derivability" not in features
    assert "fruit_visual_oscillator" in features
    assert "fruit_powerup_collision" in features
    assert "fruit_actor_observation" in features
    assert "original-transfer-jungle2-v2" in (
        policy.compatible_audit_policies
    )
    assert "original-transfer-jungle2-v2" in (
        policy.compatible_distribution_policies
    )


def test_v4_removes_unreachable_jungle2_slow_powerup_requirement() -> None:
    historical = fidelity_gate.GATE_POLICIES[
        "original-transfer-jungle2-v3"
    ]
    policy = fidelity_gate.GATE_POLICIES[
        "original-transfer-jungle2-v4"
    ]

    historical_features = {
        requirement.feature for requirement in historical.requirements
    }
    features = {requirement.feature for requirement in policy.requirements}
    assert "powerup_slow" in historical_features
    assert "powerup_slow" not in features
    assert features == historical_features - {"powerup_slow"}
    assert policy.transfer_authorizing is True
    assert policy.generation == 4
    assert "original-transfer-jungle2-v3" in (
        policy.compatible_audit_policies
    )
    assert "original-transfer-jungle2-v3" in (
        policy.compatible_distribution_policies
    )


def _build_complete_suite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    simulator_status: str = "PASS",
    simulator_ticks: int = 20,
) -> Path:
    policy = _test_policy()
    _install_test_policy(monkeypatch, policy)

    golden = tmp_path / "golden.json"
    golden.write_text("{}", encoding="utf-8")
    simulator = tmp_path / "simulator.json"
    _write_json(
        simulator,
        {
            "schema": "zuma-rl.pc-gameplay-simulator-diff",
            "version": 1,
            "status": simulator_status,
            "level_id": "Jungle2",
            "hard": False,
            "compared_tick_count": simulator_ticks,
            "failure_reasons": (
                [] if simulator_status == "PASS" else ["mismatch"]
            ),
            "shooter_mismatch_updates": [],
            "synchronize_shooter": False,
            "ticks": [
                {
                    "status": "PASS",
                    "events": {"fired": 1, "hits": 0},
                    "fired_identity_match": True,
                    "fired_count_pc": 1,
                    "fired_count_simulator": 1,
                }
            ],
        },
    )
    audit = tmp_path / "audit.json"
    live_audit = {
        "schema": fidelity_gate.AUDIT_SCHEMA,
        "version": fidelity_gate.AUDIT_VERSION,
        "status": "PASS",
        "audit_type": "actor_no_hidden_state",
        "scope": {
            "policy": policy.id,
            "environment_id": policy.environment_id,
            "profile_mode": policy.profile_mode,
            "observation_mode": policy.observation_mode,
        },
        "checks": [{"name": "test", "status": "PASS"}],
        "summary": {
            "unmapped_feature_count": 0,
            "hidden_perturbation_actor_equal": True,
            "privileged_control_changed": True,
            "visible_powerup_channel_verified": True,
        },
        "audit_fingerprint": "sha256:" + "2" * 64,
    }
    _write_json(
        audit,
        live_audit,
    )
    suite = tmp_path / "suite.json"
    _write_json(
        suite,
        {
            "schema": fidelity_gate.SUITE_SCHEMA,
            "version": fidelity_gate.SUITE_VERSION,
            "policy": policy.id,
            "evidence": [
                {
                    "id": "a-golden",
                    "kind": EvidenceKind.PC_GOLDEN.value,
                    "path": golden.name,
                    "sha256": _digest(golden),
                    "features": ["shot_release"],
                },
                {
                    "id": "b-simulator",
                    "kind": EvidenceKind.SIMULATOR_DIFF.value,
                    "path": simulator.name,
                    "sha256": _digest(simulator),
                    "features": ["shot_release"],
                },
                {
                    "id": "c-audit",
                    "kind": EvidenceKind.AUDIT.value,
                    "path": audit.name,
                    "sha256": _digest(audit),
                    "features": ["actor_no_hidden_state"],
                },
            ],
        },
    )
    monkeypatch.setattr(
        fidelity_gate.PcGoldenManifest,
        "read_json",
        lambda path: _golden_stub(),
    )
    monkeypatch.setattr(
        fidelity_gate,
        "verify_pc_golden_case",
        lambda *args, **kwargs: _golden_verification_stub(),
    )
    monkeypatch.setattr(
        "zuma_rl.actor_audit.validate_actor_audit_report",
        lambda *args, **kwargs: None,
    )
    return suite


def test_complete_content_addressed_suite_opens_narrow_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite = _build_complete_suite(tmp_path, monkeypatch)

    report = verify_fidelity_suite(
        suite,
        original_root=tmp_path,
    )

    assert report.status is FidelityGateStatus.OPEN
    assert report.exit_code == 0
    assert report.summary["gate_open"] is True
    assert all(
        row["status"] == "PASS" for row in report.requirements
    )
    assert json.loads(report.to_json())["status"] == "OPEN"


def test_historical_policy_scope_is_not_accepted_for_actor_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite_path = _build_complete_suite(tmp_path, monkeypatch)
    audit_path = tmp_path / "audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["scope"]["policy"] = "original-transfer-jungle2-v1"
    _write_json(audit_path, audit)
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    suite["evidence"][2]["sha256"] = _digest(audit_path)
    _write_json(suite_path, suite)

    report = verify_fidelity_suite(
        suite_path,
        original_root=tmp_path,
    )

    assert report.status is FidelityGateStatus.CLOSED
    actor_row = next(
        row for row in report.evidence if row["id"] == "c-audit"
    )
    assert actor_row["accepted"] is False
    assert actor_row["reason"] == (
        "audit report scope differs from the gate policy"
    )


def test_repackaged_native_capture_counts_as_one_pc_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _test_policy(
        requirements=(GateRequirement("shot_release", pc_golden=1),),
        pc_cases=2,
        simulator_cases=0,
        random_seeds=1,
    )
    _install_test_policy(monkeypatch, policy)
    golden_a = tmp_path / "golden-a.json"
    golden_b = tmp_path / "golden-b.json"
    golden_a.write_text("{}", encoding="utf-8")
    golden_b.write_text("{}", encoding="utf-8")
    suite = tmp_path / "suite.json"
    _write_json(
        suite,
        {
            "schema": fidelity_gate.SUITE_SCHEMA,
            "version": fidelity_gate.SUITE_VERSION,
            "policy": policy.id,
            "evidence": [
                {
                    "id": "golden-a",
                    "kind": EvidenceKind.PC_GOLDEN.value,
                    "path": golden_a.name,
                    "sha256": _digest(golden_a),
                    "features": ["shot_release"],
                },
                {
                    "id": "golden-b",
                    "kind": EvidenceKind.PC_GOLDEN.value,
                    "path": golden_b.name,
                    "sha256": _digest(golden_b),
                    "features": ["shot_release"],
                },
            ],
        },
    )
    manifests = {
        golden_a: _golden_stub(evidence_digit="1"),
        golden_b: _golden_stub(evidence_digit="9"),
    }
    monkeypatch.setattr(
        fidelity_gate.PcGoldenManifest,
        "read_json",
        lambda path: manifests[Path(path)],
    )
    monkeypatch.setattr(
        fidelity_gate,
        "verify_pc_golden_case",
        lambda *args, **kwargs: _golden_verification_stub(),
    )

    report = verify_fidelity_suite(suite, original_root=tmp_path)

    assert report.status is FidelityGateStatus.CLOSED
    assert report.summary["accepted_pc_golden_cases"] == 1
    assert any(
        "independent PC Golden cases: 1/2" in reason
        for reason in report.reasons
    )
    assert (
        report.evidence[0]["source_fingerprint"]
        == report.evidence[1]["source_fingerprint"]
    )


def test_original_installation_is_required_to_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite = _build_complete_suite(tmp_path, monkeypatch)

    report = verify_fidelity_suite(suite)

    assert report.status is FidelityGateStatus.CLOSED
    assert report.evidence[0]["accepted"] is False
    assert report.requirements[0]["accepted"]["simulator_diff"] == 0
    assert any(
        "original_root is required" in reason
        for reason in report.reasons
    )


def test_pass_diagnostic_cannot_replace_simulator_or_pc_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _test_policy(
        requirements=(
            GateRequirement("shot_release", simulator_diff=1),
        ),
        pc_cases=0,
        simulator_cases=0,
        random_seeds=0,
    )
    _install_test_policy(monkeypatch, policy)
    diagnostic = tmp_path / "diagnostic.json"
    _write_json(
        diagnostic,
        {
            "schema": "zuma-rl.pc-gameplay-simulator-diff",
            "version": 1,
            "status": "PASS",
            "level_id": "Jungle2",
            "hard": False,
            "compared_tick_count": 200,
            "failure_reasons": [],
            "shooter_mismatch_updates": [10],
            "synchronize_shooter": True,
        },
    )
    suite = tmp_path / "suite.json"
    _write_json(
        suite,
        {
            "schema": fidelity_gate.SUITE_SCHEMA,
            "version": fidelity_gate.SUITE_VERSION,
            "policy": policy.id,
            "evidence": [
                {
                    "id": "diagnostic",
                    "kind": EvidenceKind.DIAGNOSTIC.value,
                    "path": diagnostic.name,
                    "sha256": _digest(diagnostic),
                    "features": ["shot_release"],
                }
            ],
        },
    )

    report = verify_fidelity_suite(suite)

    assert report.status is FidelityGateStatus.CLOSED
    assert report.summary["diagnostic_only_cases"] == 1
    assert report.requirements[0]["accepted"]["simulator_diff"] == 0


def test_tampered_report_hash_makes_suite_invalid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite = _build_complete_suite(tmp_path, monkeypatch)
    simulator = tmp_path / "simulator.json"
    simulator.write_text(
        simulator.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    report = verify_fidelity_suite(
        suite,
        original_root=tmp_path,
    )

    assert report.status is FidelityGateStatus.INVALID
    assert report.exit_code == 2
    assert "SHA-256 differs" in report.reasons[0]


def test_failing_simulator_report_keeps_gate_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite = _build_complete_suite(
        tmp_path,
        monkeypatch,
        simulator_status="FAIL",
    )

    report = verify_fidelity_suite(
        suite,
        original_root=tmp_path,
    )

    assert report.status is FidelityGateStatus.CLOSED
    assert any(
        "b-simulator" in reason for reason in report.reasons
    )


def test_minimum_tick_requirement_is_applied_per_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _test_policy(
        requirements=(
            GateRequirement(
                "long_horizon_drift",
                simulator_diff=1,
                minimum_ticks=500,
            ),
        ),
        pc_cases=0,
        simulator_cases=1,
        random_seeds=0,
    )
    _install_test_policy(monkeypatch, policy)
    report_path = tmp_path / "short.json"
    _write_json(
        report_path,
        {
            "schema": "zuma-rl.pc-gameplay-simulator-diff",
            "version": 1,
            "status": "PASS",
            "level_id": "Jungle2",
            "hard": False,
            "compared_tick_count": 499,
            "failure_reasons": [],
            "shooter_mismatch_updates": [],
            "synchronize_shooter": False,
        },
    )
    suite_path = tmp_path / "suite.json"
    _write_json(
        suite_path,
        {
            "schema": fidelity_gate.SUITE_SCHEMA,
            "version": fidelity_gate.SUITE_VERSION,
            "policy": policy.id,
            "evidence": [
                {
                    "id": "short",
                    "kind": EvidenceKind.SIMULATOR_DIFF.value,
                    "path": report_path.name,
                    "sha256": _digest(report_path),
                    "features": ["long_horizon_drift"],
                }
            ],
        },
    )

    report = verify_fidelity_suite(suite_path)

    assert report.status is FidelityGateStatus.CLOSED
    assert report.requirements[0]["status"] == "MISSING"


@pytest.mark.parametrize(
    "extra",
    [
        {
            "synchronize_shooter": True,
            "shooter_mismatch_updates": [12],
        },
        {
            "synchronize_shooter": False,
            "shooter_mismatch_updates": [],
            "scenario": {"injected_terminal_ball_id": 99},
        },
    ],
)
def test_intervened_simulator_pass_is_not_certifying(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extra: dict[str, object],
) -> None:
    policy = _test_policy(
        requirements=(GateRequirement("shot_release", simulator_diff=1),),
        pc_cases=0,
        simulator_cases=0,
        random_seeds=0,
    )
    _install_test_policy(monkeypatch, policy)
    payload: dict[str, object] = {
        "schema": "zuma-rl.pc-gameplay-simulator-diff",
        "version": 1,
        "status": "PASS",
        "level_id": "Jungle2",
        "hard": False,
        "compared_tick_count": 20,
        "failure_reasons": [],
    }
    payload.update(extra)
    evidence = tmp_path / "intervened.json"
    _write_json(evidence, payload)
    suite = tmp_path / "suite.json"
    _write_json(
        suite,
        {
            "schema": fidelity_gate.SUITE_SCHEMA,
            "version": fidelity_gate.SUITE_VERSION,
            "policy": policy.id,
            "evidence": [
                {
                    "id": "intervened",
                    "kind": EvidenceKind.SIMULATOR_DIFF.value,
                    "path": evidence.name,
                    "sha256": _digest(evidence),
                    "features": ["shot_release"],
                }
            ],
        },
    )

    report = verify_fidelity_suite(suite)

    assert report.status is FidelityGateStatus.CLOSED


def test_loss_v2_requires_live_source_recomputation_before_certifying(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from zuma_rl import pc_loss_sequence

    calls: list[tuple[Path, Path]] = []

    def validate(
        report: object,
        *,
        evidence_root: str | Path,
        original_root: str | Path,
    ) -> None:
        calls.append((Path(evidence_root), Path(original_root)))
        return None

    monkeypatch.setattr(
        pc_loss_sequence,
        "validate_loss_simulator_diff_report",
        validate,
    )
    report = {
        "schema": "zuma-rl.pc-loss-simulator-diff",
        "version": 2,
        "status": "PASS",
        "scenario": {
            "source_bound_terminal_ball_id": 99,
        },
    }

    assert (
        fidelity_gate._noncertifying_report_reason(
            report,
            schema=report["schema"],
            evidence_root=tmp_path,
            original_root=tmp_path,
        )
        is None
    )
    assert calls == [(tmp_path, tmp_path)]

    report["scenario"]["injected_terminal_ball_id"] = 99
    assert "injected or synthetic" in str(
        fidelity_gate._noncertifying_report_reason(
            report,
            schema=report["schema"],
            evidence_root=tmp_path,
            original_root=tmp_path,
        )
    )


def test_handwritten_feature_label_cannot_claim_unobserved_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _test_policy(
        requirements=(GateRequirement("swap", simulator_diff=1),),
        pc_cases=0,
        simulator_cases=1,
        random_seeds=0,
    )
    _install_test_policy(monkeypatch, policy)
    evidence = tmp_path / "input-only.json"
    _write_json(
        evidence,
        {
            "schema": "zuma-rl.pc-gameplay-simulator-diff",
            "version": 1,
            "status": "PASS",
            "level_id": "Jungle2",
            "hard": False,
            "compared_tick_count": 2,
            "failure_reasons": [],
            "shooter_mismatch_updates": [],
            "synchronize_shooter": False,
            "input_results": [
                {
                    "sequence": 1,
                    "framework_update": 10,
                    "kind": "mouse_button",
                    "accepted": True,
                }
            ],
            "ticks": [
                {
                    "status": "PASS",
                    "events": {"fired": 0, "hits": 0},
                }
            ],
        },
    )
    suite = tmp_path / "suite.json"
    _write_json(
        suite,
        {
            "schema": fidelity_gate.SUITE_SCHEMA,
            "version": fidelity_gate.SUITE_VERSION,
            "policy": policy.id,
            "evidence": [
                {
                    "id": "input-only",
                    "kind": EvidenceKind.SIMULATOR_DIFF.value,
                    "path": evidence.name,
                    "sha256": _digest(evidence),
                    "features": ["input_cadence", "swap"],
                }
            ],
        },
    )

    report = verify_fidelity_suite(suite)

    assert report.status is FidelityGateStatus.CLOSED
    assert report.evidence[0]["authorized_features"] == ["input_cadence"]
    assert report.evidence[0]["unsupported_features"] == ["swap"]
    assert report.summary["accepted_simulator_diff_cases"] == 1
    assert any(
        "unsupported evidence feature claim(s): swap" in reason
        for reason in report.reasons
    )


def test_pc_feature_label_requires_verifier_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _test_policy(
        requirements=(GateRequirement("shot_release", pc_golden=1),),
        pc_cases=0,
        simulator_cases=0,
        random_seeds=0,
    )
    _install_test_policy(monkeypatch, policy)
    golden = tmp_path / "golden.json"
    golden.write_text("{}", encoding="utf-8")
    suite = tmp_path / "suite.json"
    _write_json(
        suite,
        {
            "schema": fidelity_gate.SUITE_SCHEMA,
            "version": fidelity_gate.SUITE_VERSION,
            "policy": policy.id,
            "evidence": [
                {
                    "id": "golden",
                    "kind": EvidenceKind.PC_GOLDEN.value,
                    "path": golden.name,
                    "sha256": _digest(golden),
                    "features": ["shot_release"],
                }
            ],
        },
    )
    monkeypatch.setattr(
        fidelity_gate.PcGoldenManifest,
        "read_json",
        lambda path: _golden_stub(),
    )
    monkeypatch.setattr(
        fidelity_gate,
        "verify_pc_golden_case",
        lambda *args, **kwargs: SimpleNamespace(
            status=ComparisonStatus.PASS,
            checks=(),
            summary={},
        ),
    )

    report = verify_fidelity_suite(suite, original_root=tmp_path)

    assert report.status is FidelityGateStatus.CLOSED
    assert report.evidence[0]["accepted"] is True
    assert report.evidence[0]["authorized_features"] == []
    assert report.evidence[0]["unsupported_features"] == ["shot_release"]
    assert report.requirements[0]["status"] == "MISSING"


def test_passing_visual_audit_cross_authorizes_its_pc_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _test_policy(
        requirements=(
            GateRequirement(
                "actor_visual_derivability",
                pc_golden=1,
                audit=1,
            ),
        ),
        pc_cases=1,
        simulator_cases=0,
        random_seeds=1,
    )
    _install_test_policy(monkeypatch, policy)
    manifest = _golden_stub()
    source_fingerprint = fidelity_gate._pc_golden_source_fingerprint(
        manifest
    )
    golden = tmp_path / "golden.json"
    golden.write_text("{}", encoding="utf-8")
    audit = tmp_path / "visual-audit.json"
    _write_json(
        audit,
        {
            "schema": fidelity_gate.AUDIT_SCHEMA,
            "version": fidelity_gate.AUDIT_VERSION,
            "status": "PASS",
            "audit_type": "actor_visual_derivability",
            "scope": {
                "policy": policy.id,
                "environment_id": policy.environment_id,
                "profile_mode": policy.profile_mode,
                "observation_mode": policy.observation_mode,
            },
            "checks": [{"name": "coverage", "status": "PASS"}],
            "failure_reasons": [],
            "summary": {
                "missing_feature_count": 0,
                "pc_source_count": 1,
                "pc_source_fingerprints": [source_fingerprint],
            },
            "audit_fingerprint": "sha256:" + "9" * 64,
        },
    )
    suite = tmp_path / "suite.json"
    _write_json(
        suite,
        {
            "schema": fidelity_gate.SUITE_SCHEMA,
            "version": fidelity_gate.SUITE_VERSION,
            "policy": policy.id,
            "evidence": [
                {
                    "id": "a-golden",
                    "kind": EvidenceKind.PC_GOLDEN.value,
                    "path": golden.name,
                    "sha256": _digest(golden),
                    "features": ["actor_visual_derivability"],
                },
                {
                    "id": "b-visual-audit",
                    "kind": EvidenceKind.AUDIT.value,
                    "path": audit.name,
                    "sha256": _digest(audit),
                    "features": ["actor_visual_derivability"],
                },
            ],
        },
    )
    monkeypatch.setattr(
        fidelity_gate.PcGoldenManifest,
        "read_json",
        lambda path: manifest,
    )
    monkeypatch.setattr(
        fidelity_gate,
        "verify_pc_golden_case",
        lambda *args, **kwargs: SimpleNamespace(
            status=ComparisonStatus.PASS,
            checks=(),
            summary={},
        ),
    )
    monkeypatch.setattr(
        "zuma_rl.pc_visual_audit.validate_actor_visual_audit_report",
        lambda *args, **kwargs: None,
    )

    report = verify_fidelity_suite(suite, original_root=tmp_path)

    assert report.status is FidelityGateStatus.OPEN
    assert report.evidence[0]["authorized_features"] == [
        "actor_visual_derivability"
    ]
    assert report.evidence[0]["unsupported_features"] == []
    assert report.evidence[1]["supporting_source_fingerprints"] == [
        source_fingerprint
    ]


def test_mechanism_audit_cross_authorizes_only_its_pc_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _test_policy(
        requirements=(
            GateRequirement(
                "projectile_collision",
                pc_golden=1,
            ),
        ),
        pc_cases=1,
        simulator_cases=0,
        random_seeds=1,
    )
    _install_test_policy(monkeypatch, policy)
    manifest = _golden_stub()
    source_fingerprint = fidelity_gate._pc_golden_source_fingerprint(
        manifest
    )
    golden = tmp_path / "golden.json"
    golden.write_text("{}", encoding="utf-8")
    audit = tmp_path / "mechanism-audit.json"
    _write_json(
        audit,
        {
            "schema": fidelity_gate.AUDIT_SCHEMA,
            "version": fidelity_gate.AUDIT_VERSION,
            "status": "PASS",
            "audit_type": "pc_mechanism_coverage",
            "scope": {
                "policy": "original-transfer-jungle2-v1",
                "environment_id": policy.environment_id,
                "profile_mode": policy.profile_mode,
                "observation_mode": policy.observation_mode,
            },
            "checks": [{"name": "coverage", "status": "PASS"}],
            "failure_reasons": [],
            "summary": {
                "pc_source_count": 1,
                "pc_source_fingerprints": [source_fingerprint],
                "source_feature_authorizations": [
                    {
                        "source_fingerprint": source_fingerprint,
                        "authorized_features": [
                            "projectile_collision",
                        ],
                    }
                ],
            },
            "audit_fingerprint": "sha256:" + "9" * 64,
        },
    )
    suite = tmp_path / "suite.json"
    _write_json(
        suite,
        {
            "schema": fidelity_gate.SUITE_SCHEMA,
            "version": fidelity_gate.SUITE_VERSION,
            "policy": policy.id,
            "evidence": [
                {
                    "id": "a-golden",
                    "kind": EvidenceKind.PC_GOLDEN.value,
                    "path": golden.name,
                    "sha256": _digest(golden),
                    "features": ["projectile_collision"],
                },
                {
                    "id": "b-mechanism-audit",
                    "kind": EvidenceKind.AUDIT.value,
                    "path": audit.name,
                    "sha256": _digest(audit),
                    "features": ["pc_mechanism_coverage"],
                },
            ],
        },
    )
    monkeypatch.setattr(
        fidelity_gate.PcGoldenManifest,
        "read_json",
        lambda path: manifest,
    )
    monkeypatch.setattr(
        fidelity_gate,
        "verify_pc_golden_case",
        lambda *args, **kwargs: SimpleNamespace(
            status=ComparisonStatus.PASS,
            checks=(),
            summary={},
        ),
    )
    monkeypatch.setattr(
        "zuma_rl.pc_mechanism_audit."
        "validate_pc_mechanism_audit_report",
        lambda *args, **kwargs: None,
    )

    report = verify_fidelity_suite(suite, original_root=tmp_path)

    assert report.status is FidelityGateStatus.OPEN
    assert report.evidence[0]["authorized_features"] == [
        "projectile_collision"
    ]
    assert report.evidence[0]["unsupported_features"] == []
    assert report.evidence[1]["supporting_source_features"] == [
        {
            "source_fingerprint": source_fingerprint,
            "authorized_features": ["projectile_collision"],
        }
    ]


def test_source_bound_provenance_mechanism_and_diff_form_original_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = GatePolicy(
        id="test-policy",
        environment_id="ZumaRevenge-v0",
        levels=("Jungle2",),
        hard=False,
        profile_mode="tutorials_completed",
        observation_mode="actor",
        curve_count=1,
        minimum_pc_golden_cases=0,
        minimum_simulator_diff_cases=1,
        minimum_random_seeds=1,
        requirements=(
            GateRequirement(
                "natural_loss",
                original_source=1,
                simulator_diff=1,
            ),
        ),
        generation=2,
        transfer_authorizing=True,
        minimum_original_source_cases=1,
    )
    _install_test_policy(monkeypatch, policy)
    source_fingerprint = "sha256:" + "1" * 64
    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")
    audit = tmp_path / "mechanism-audit.json"
    _write_json(
        audit,
        {
            "schema": fidelity_gate.AUDIT_SCHEMA,
            "version": 2,
            "status": "PASS",
            "audit_type": "pc_mechanism_coverage",
            "scope": {
                "policy": policy.id,
                "environment_id": policy.environment_id,
                "profile_mode": policy.profile_mode,
                "observation_mode": policy.observation_mode,
            },
            "checks": [{"name": "coverage", "status": "PASS"}],
            "failure_reasons": [],
            "summary": {
                "pc_source_count": 1,
                "pc_source_fingerprints": [source_fingerprint],
                "source_feature_authorizations": [
                    {
                        "source_fingerprint": source_fingerprint,
                        "authorized_features": ["natural_loss"],
                    }
                ],
            },
            "audit_fingerprint": "sha256:" + "2" * 64,
        },
    )
    simulator = tmp_path / "simulator.json"
    _write_json(
        simulator,
        {
            "schema": "zuma-rl.pc-gameplay-simulator-diff",
            "version": 1,
            "status": "PASS",
            "level_id": "Jungle2",
            "hard": False,
            "compared_tick_count": 20,
            "failure_reasons": [],
            "synchronize_shooter": False,
            "shooter_mismatch_updates": [],
        },
    )
    suite = tmp_path / "suite.json"
    _write_json(
        suite,
        {
            "schema": fidelity_gate.SUITE_SCHEMA,
            "version": fidelity_gate.SUITE_VERSION,
            "policy": policy.id,
            "evidence": [
                {
                    "id": "a-source",
                    "kind": EvidenceKind.PC_SOURCE.value,
                    "path": source.name,
                    "sha256": _digest(source),
                    "features": ["natural_loss", "source_authenticity"],
                },
                {
                    "id": "b-mechanism-audit",
                    "kind": EvidenceKind.AUDIT.value,
                    "path": audit.name,
                    "sha256": _digest(audit),
                    "features": ["pc_mechanism_coverage"],
                },
                {
                    "id": "c-simulator",
                    "kind": EvidenceKind.SIMULATOR_DIFF.value,
                    "path": simulator.name,
                    "sha256": _digest(simulator),
                    "features": ["natural_loss"],
                },
            ],
        },
    )
    monkeypatch.setattr(
        "zuma_rl.pc_source.verify_pc_source_manifest",
        lambda *args, **kwargs: {
            "schema": "zuma-rl.pc-source-verification",
            "version": 1,
            "status": "PASS",
            "scope": {
                "level_id": "Jungle2",
                "hard": False,
                "profile_mode": "tutorials_completed",
                "mode": "adventure",
            },
            "random_seed": 123,
            "tick_count": 631,
            "source_fingerprint": source_fingerprint,
            "dmo_sha256": "sha256:" + "3" * 64,
            "authorized_features": ["source_authenticity"],
            "transport": fidelity_gate._SOURCE_BOUND_PC_SOURCE_TRANSPORT,
            "cross_process_exact_replay_required": False,
            "strict_command_replay": {
                "schema": fidelity_gate._SOURCE_BOUND_REPLAY_SCHEMA,
                "version": 2,
                "status": "PASS",
                "classification": (
                    "source_bound_original_natural_retail_outcome"
                ),
            },
        },
    )
    monkeypatch.setattr(
        "zuma_rl.pc_mechanism_audit.validate_pc_mechanism_audit_report",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        fidelity_gate,
        "_simulator_authorized_features",
        lambda *args, **kwargs: frozenset({"natural_loss"}),
    )

    report = verify_fidelity_suite(
        suite,
        original_root=tmp_path,
    )

    assert report.status is FidelityGateStatus.OPEN, report.reasons
    assert report.requirements[0]["accepted"]["original_source"] == 1
    assert report.requirements[0]["accepted"]["simulator_diff"] == 1
    assert report.evidence[0]["original_source_authenticated"] is True
    assert report.evidence[0]["original_source_authentication_basis"] == (
        fidelity_gate._SOURCE_BOUND_PC_SOURCE_AUTHENTICATION_BASIS
    )
    assert report.evidence[0]["authorized_features"] == [
        "natural_loss",
        "source_authenticity",
    ]
    assert report.summary["original_source_authentication_basis_counts"] == {
        fidelity_gate._SOURCE_BOUND_PC_SOURCE_AUTHENTICATION_BASIS: 1,
    }


def test_gameplay_features_are_derived_from_exact_tick_events() -> None:
    ticks = []
    for index in range(500):
        ticks.append(
            {
                "status": "PASS",
                "active_identity_match": True,
                "active_color_match": True,
                "fired_identity_match": True,
                "staging_identity_match": True,
                "fired_count_pc": 1 if index == 10 else 0,
                "fired_count_simulator": 1 if index == 10 else 0,
                "staging_count_pc": 1 if index == 20 else 0,
                "staging_count_simulator": 1 if index == 20 else 0,
                "qrand_observed_match": True,
                "qrand_selected_index_pc": 2,
                "qrand_selected_index_simulator": 2,
                "qrand_update_count_pc": 1 if index < 250 else 2,
                "qrand_update_count_simulator": 1 if index < 250 else 2,
                "thread_crt_rand_observed_match": True,
                "thread_crt_rand_state_pc": 123,
                "thread_crt_rand_state_simulator": 123,
                "events": {
                    "fired": 1 if index == 10 else 0,
                    "hits": 1 if index == 20 else 0,
                },
            }
        )
    report = {
        "compared_tick_count": 500,
        "shooter_rng_state_restored": True,
        "input_results": [
            {
                "sequence": 7,
                "framework_update": 10,
                "kind": "mouse_button",
                "accepted": True,
            }
        ],
        "ticks": ticks,
    }

    assert fidelity_gate._gameplay_authorized_features(report) == frozenset(
        {
            "input_cadence",
            "long_horizon_drift",
            "projectile_collision",
            "rng_pending",
            "shot_release",
        }
    )


def test_scored_match_and_rollback_proofs_fail_closed() -> None:
    ticks = [
        {
            "status": "PASS",
            "framework_update": update,
            "active_identity_match": True,
            "active_color_match": True,
            "latent_mismatches": {},
            "events": {
                "matches": 1 if update in {11, 14} else 0,
                "balls_exploded": (
                    3 if update == 11 else 4 if update == 14 else 0
                ),
                "score_delta": (
                    30 if update == 11 else 140 if update == 14 else 0
                ),
            },
            "score_delta_pc": (
                30 if update == 11 else 140 if update == 14 else 0
            ),
            "score_delta_simulator": (
                30 if update == 11 else 140 if update == 14 else 0
            ),
        }
        for update in range(10, 15)
    ]
    report = {
        "source_authorized_features": [
            "match3",
            "match4",
            "rollback_chain",
        ],
        "source_feature_proofs": [
            {
                "feature": "match3",
                "status": "PASS",
                "framework_update": 11,
                "score_delta": 30,
                "exploding_ball_ids": [1, 2, 3],
            },
            {
                "feature": "match4",
                "status": "PASS",
                "framework_update": 14,
                "score_delta": 140,
                "exploding_ball_ids": [7, 8, 9, 10],
            },
            {
                "feature": "rollback_chain",
                "status": "PASS",
                "start_update": 12,
                "stop_update": 14,
                "score_delta": 140,
            },
        ],
        "ticks": ticks,
    }

    assert fidelity_gate._gameplay_authorized_features(report) == frozenset(
        {"match3", "match4", "rollback_chain"}
    )

    ticks[-1]["latent_mismatches"] = {"explode_frame": "differs"}
    assert fidelity_gate._gameplay_authorized_features(report) == frozenset(
        {"match3"}
    )

    ticks[-1]["latent_mismatches"] = {}
    ticks[-1]["score_delta_simulator"] = 139
    assert fidelity_gate._gameplay_authorized_features(report) == frozenset(
        {"match3"}
    )


def _exact_v5_mechanism_tick(
    update: int,
    *,
    pending: list[int],
    mtrand_index: int,
) -> dict[str, object]:
    return {
        "status": "PASS",
        "framework_update": update,
        "active_count_pc": 53,
        "active_count_simulator": 53,
        "active_identity_match": True,
        "active_color_match": True,
        "pending_colors_pc": list(pending),
        "pending_colors_simulator": list(pending),
        "pending_colors_match": True,
        "fired_count_pc": 1,
        "fired_count_simulator": 1,
        "fired_identity_match": True,
        "staging_count_pc": 0,
        "staging_count_simulator": 0,
        "staging_identity_match": True,
        "shooter_observed_match": True,
        "qrand_observed_match": True,
        "thread_crt_rand_observed_match": True,
        "global_mtrand_observed_match": True,
        "gameplay_state_matched_before_global_rng": True,
        "global_mtrand_index_pc": mtrand_index,
        "global_mtrand_index_simulator": mtrand_index,
        "score_pc": 7_950,
        "score_simulator": 7_950,
        "score_delta_pc": 0,
        "score_delta_simulator": 0,
        "maximum_position_error_px": 0.0,
        "maximum_waypoint_error": 0.0,
        "fired_position_error_px": 0.0,
        "fired_waypoint_error": 0.0,
        "fired_progress_error": 0.0,
        "staging_position_error_px": 0.0,
        "staging_progress_error": 0.0,
        "latent_mismatches": {},
        "fired_latent_mismatches": {},
        "staging_latent_mismatches": {},
        "curve_state_mismatches": {},
        "fruit_state_mismatches": {},
        "events": {
            "fired": 0,
            "hits": 0,
            "inserted": 0,
            "matches": 0,
            "balls_exploded": 0,
            "balls_removed": 0,
            "score_delta": 0,
        },
    }


def _exact_v5_mechanism_report(
    *,
    feature: str,
    proof: dict[str, object],
    ticks: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "version": 5,
        "initial_free_projectile_state_restored": True,
        "global_mtrand_state_restored": True,
        "position_tolerance_px": 1e-3,
        "waypoint_tolerance": 1e-4,
        "progress_tolerance": 1e-6,
        "source_authorized_features": [feature],
        "source_feature_proofs": [proof],
        "ticks": ticks,
    }


def test_gap_shot_requires_exact_gap_latent_state_on_both_ticks() -> None:
    gap = {
        "curve_index": 0,
        "gap_distance": 36,
        "boundary_ball_id": 61,
    }
    proof: dict[str, object] = {
        "feature": "gap_shot",
        "status": "PASS",
        "from_update": 3_770,
        "framework_update": 3_771,
        "projectile_ball_id": 78,
        "curve_index": 0,
        "curve_point_before": 0,
        "curve_point_after": 397,
        "projectile_remained_free": True,
        "projectile_radius": 18,
        "new_gap_entry": gap,
        "simulations": [
            {"final_curve_point": 397, "new_entries": [gap]},
            {"final_curve_point": 397, "new_entries": [gap]},
        ],
    }
    ticks = [
        _exact_v5_mechanism_tick(
            update,
            pending=[1],
            mtrand_index=567,
        )
        for update in (3_770, 3_771)
    ]
    report = _exact_v5_mechanism_report(
        feature="gap_shot",
        proof=proof,
        ticks=ticks,
    )

    assert fidelity_gate._gameplay_authorized_features(report) == frozenset(
        {"gap_shot"}
    )

    ticks[1]["fired_latent_mismatches"] = {"gap_info": 1}
    assert fidelity_gate._gameplay_authorized_features(report) == frozenset()
    ticks[1]["fired_latent_mismatches"] = {}
    simulations = proof["simulations"]
    assert isinstance(simulations, list)
    simulations[0]["final_curve_point"] = 396
    assert fidelity_gate._gameplay_authorized_features(report) == frozenset()


def test_rng_rejection_requires_exact_draw_and_pending_transition() -> None:
    rejected = [574_727_091, 2_102_776_875, 1_864_971_879, 939_760_459]
    repeat = 1_816_503_049
    accepted = 738_518_352
    visual = 1_881_158_752
    proof: dict[str, object] = {
        "feature": "rng_rejection",
        "status": "PASS",
        "from_update": 3_581,
        "framework_update": 3_582,
        "previous_color_id": 3,
        "generated_color_id": 0,
        "generated_ball_id": 69,
        "rejection_count": 4,
        "rejected_candidate_outputs": rejected,
        "rejected_candidate_colors": [3, 3, 3, 3],
        "repeat_roll_output": repeat,
        "accepted_candidate_output": accepted,
        "accepted_candidate_color": 0,
        "ball_visual_frame_output": visual,
        "mtrand_outputs": [repeat, *rejected, accepted, visual],
        "mtrand_draw_count": 7,
        "mtrand_state_exact": True,
        "neighbouring_ambient_draw_count": 0,
        "ambient_before_output": None,
        "ambient_after_output": None,
        "rng_sequence_layout": (
            "repeat,candidate_rejection_loop,ball_visual_frame"
        ),
    }
    ticks = [
        _exact_v5_mechanism_tick(
            3_581,
            pending=[],
            mtrand_index=521,
        ),
        _exact_v5_mechanism_tick(
            3_582,
            pending=[0],
            mtrand_index=528,
        ),
    ]
    report = _exact_v5_mechanism_report(
        feature="rng_rejection",
        proof=proof,
        ticks=ticks,
    )

    assert fidelity_gate._gameplay_authorized_features(report) == frozenset(
        {"rng_rejection"}
    )

    ticks[1]["pending_colors_simulator"] = [1]
    assert fidelity_gate._gameplay_authorized_features(report) == frozenset()
    ticks[1]["pending_colors_simulator"] = [0]
    proof["rejected_candidate_colors"] = [3, 3, 3, 0]
    assert fidelity_gate._gameplay_authorized_features(report) == frozenset()


def _exact_powerup_proof(feature: str) -> dict[str, object]:
    powerup_type = {
        "powerup_proximity_bomb": 0,
        "powerup_slow": 1,
        "powerup_reverse": 3,
    }[feature]
    exploding_ids = [1, 2, 3, 4] if powerup_type == 0 else [1, 2, 3]
    score_delta = 10 * len(exploding_ids)
    effect: dict[str, object] = {
        "schema": "zuma-rl.pc-powerup-trigger-verification",
        "version": 1,
        "status": "PASS",
        "trajectory": {
            "start_update": 98,
            "end_update": 102,
            "tick_count": 5,
        },
        "target": {
            "curve_index": 0,
            "ball_id": 2,
            "color_id": 2,
            "powerup_type": powerup_type,
        },
        "trigger": {
            "update": 100,
            "native_game_time": 2_000,
            "newly_exploding_ball_ids": exploding_ids,
            "score_before": 1_000,
            "score_after": 1_000 + score_delta,
            "score_delta": score_delta,
            "counter_before": 0,
            "counter_after": 1,
            "cooldown_before": -1_000,
            "cooldown_after": 2_000,
            "active_color_count_before": 1,
            "active_color_count_after": 0,
            "target_lifetime_before": 500,
            "target_lifetime_after": 499,
            "target_explode_frame_at_trigger": 1,
        },
        "reverse": None,
        "slow": None,
        "bomb_geometry": None,
        "last_seen_update_by_exploding_ball": {
            str(ball_id): 102 for ball_id in exploding_ids
        },
        "trigger_flag_persists_through_update": 102,
    }
    slow_movement: dict[str, object] | None = None
    if powerup_type == 0:
        effect["bomb_geometry"] = {
            "collision_pad_per_ball": 56,
            "direct_match_ball_ids": [1, 2, 3],
            "farthest_selected": {
                "ball_id": 4,
                "distance": 147.0,
                "threshold": 148.0,
            },
            "nearest_rejected": {
                "ball_id": 99,
                "distance": 148.0,
                "threshold": 148.0,
            },
            "retail_radius_for_18px_balls": 148,
            "spatially_selected_ball_ids": exploding_ids,
            "strict_less_than_threshold": True,
        }
    elif powerup_type == 3:
        effect["reverse"] = {
            "ticks_at_trigger": 300,
            "ticks_next_update": 299,
            "speed": 1.0,
            "movement": {
                "reference_ball_id": 90,
                "sample_phase": "two_complete_post_trigger_intervals",
                "trigger_distance": 500.0,
                "next_distance": 499.0,
                "post_next_distance": 498.0,
                "distance_per_tick": 1.0,
            },
        }
    else:
        effect["slow"] = {
            "ticks_at_trigger": 800,
            "ticks_next_update": 799,
        }
        slow_movement = {
            "reference_ball_id": 90,
            "movement_before_trigger": 0.125,
            "movement_at_trigger": 0.0625,
            "movement_after_trigger": 0.0625,
            "strictly_slower": True,
        }
    return {
        "feature": feature,
        "status": "PASS",
        "framework_update": 100,
        "native_game_time": 2_000,
        "curve_index": 0,
        "powerup_type": powerup_type,
        "trigger_ball_id": 2,
        "trigger_color_id": 2,
        "newly_exploding_ball_ids": exploding_ids,
        "score_delta": score_delta,
        "manager_transition_exact": True,
        "effect_verification": effect,
        "slow_physical_movement": slow_movement,
    }


def _exact_powerup_report(feature: str) -> dict[str, object]:
    proof = _exact_powerup_proof(feature)
    trigger = proof["effect_verification"]["trigger"]
    score_before = trigger["score_before"]
    score_after = trigger["score_after"]
    ticks = [
        _exact_v5_mechanism_tick(
            update,
            pending=[1],
            mtrand_index=update,
        )
        for update in range(98, 103)
    ]
    for tick in ticks:
        is_trigger = tick["framework_update"] == 100
        score = score_after if tick["framework_update"] >= 100 else score_before
        tick["score_pc"] = score
        tick["score_simulator"] = score
        tick["score_delta_pc"] = proof["score_delta"] if is_trigger else 0
        tick["score_delta_simulator"] = (
            proof["score_delta"] if is_trigger else 0
        )
        tick["events"]["powerups_triggered"] = 1 if is_trigger else 0
        if is_trigger:
            tick["events"].update(
                {
                    "matches": 1,
                    "balls_exploded": len(
                        proof["newly_exploding_ball_ids"]
                    ),
                    "score_delta": proof["score_delta"],
                }
            )
    return _exact_v5_mechanism_report(
        feature=feature,
        proof=proof,
        ticks=ticks,
    )


@pytest.mark.parametrize(
    "feature",
    [
        "powerup_proximity_bomb",
        "powerup_reverse",
        "powerup_slow",
    ],
)
def test_powerup_effects_require_source_proof_and_exact_simulator_ticks(
    feature: str,
) -> None:
    report = _exact_powerup_report(feature)

    assert fidelity_gate._gameplay_authorized_features(report) == frozenset(
        {feature}
    )

    report["ticks"][2]["curve_state_mismatches"] = {"timer": "differs"}
    assert fidelity_gate._gameplay_authorized_features(report) == frozenset()


def test_powerup_effect_physics_are_checked_by_feature() -> None:
    bomb = _exact_powerup_report("powerup_proximity_bomb")
    bomb_proof = bomb["source_feature_proofs"][0]
    bomb_proof["effect_verification"]["bomb_geometry"][
        "nearest_rejected"
    ]["distance"] = 147.9
    assert fidelity_gate._gameplay_authorized_features(bomb) == frozenset()

    reverse = _exact_powerup_report("powerup_reverse")
    reverse_proof = reverse["source_feature_proofs"][0]
    reverse_proof["effect_verification"]["reverse"]["movement"][
        "post_next_distance"
    ] = 497.5
    assert fidelity_gate._gameplay_authorized_features(reverse) == frozenset()

    slow = _exact_powerup_report("powerup_slow")
    slow_proof = slow["source_feature_proofs"][0]
    slow_proof["slow_physical_movement"]["movement_after_trigger"] = 0.125
    assert fidelity_gate._gameplay_authorized_features(slow) == frozenset()


def test_v5_fruit_collection_features_require_source_and_exact_ticks() -> None:
    proofs = [
        {
            "feature": "fruit_projectile_collision",
            "status": "PASS",
            "framework_update": 101,
            "collision_cause": "free_projectile",
            "active_pointer_transition": "same_nonzero_to_collecting",
            "projectile_ball_id": 77,
            "distance_squared": 100.0,
            "collision_radius_squared": 1_936.0,
        },
        {
            "feature": "fruit_collection_score",
            "status": "PASS",
            "framework_update": 101,
            "score_before": 4_600,
            "score_after": 5_100,
            "score_delta": 500,
            "score_at_level_start": 4_000,
            "tier_divisor": 600,
            "tier_multiplier": 100,
            "minimum_points": 500,
            "isolated_projectile_collection": True,
        },
        {
            "feature": "fruit_collection_animation",
            "status": "PASS",
            "start_update": 101,
            "framework_update": 101,
            "clear_update": 103,
            "collection_ticks": 2,
            "tick_hz": 100,
            "active_pointer_clear_offset": 2,
            "alpha_step": -8,
            "glow_recurrence_exact": True,
            "cell_recurrence_exact": True,
            "float32_state_retention_exact": True,
        },
    ]
    ticks = [
        {
            "status": "PASS",
            "framework_update": update,
            "fruit_state_mismatches": {},
            "fired_identity_match": True,
            "fired_latent_mismatches": {},
            "score_pc": 5_100,
            "score_simulator": 5_100,
            "score_delta_pc": 500 if update == 101 else 0,
            "score_delta_simulator": 500 if update == 101 else 0,
            "events": {
                "score_delta": 500 if update == 101 else 0,
                "fruits_collected": 1 if update == 101 else 0,
                "fruits_expired": 0,
                "hits": 0,
                "matches": 0,
                "balls_exploded": 0,
                "powerups_triggered": 0,
            },
        }
        for update in range(101, 104)
    ]
    report = {
        "version": 5,
        "fruit_runtime_state_restored": True,
        "score_at_level_start": 4_000,
        "source_authorized_features": sorted(
            proof["feature"] for proof in proofs
        ),
        "source_feature_proofs": proofs,
        "ticks": ticks,
    }

    assert fidelity_gate._gameplay_authorized_features(report) == frozenset(
        {
            "fruit_projectile_collision",
            "fruit_collection_score",
            "fruit_collection_animation",
        }
    )

    ticks[0]["events"]["score_delta"] = 499
    assert fidelity_gate._gameplay_authorized_features(report) == frozenset(
        {"fruit_projectile_collision", "fruit_collection_animation"}
    )
    ticks[0]["events"]["score_delta"] = 500
    proofs[2]["glow_recurrence_exact"] = False
    assert fidelity_gate._gameplay_authorized_features(report) == frozenset(
        {"fruit_projectile_collision", "fruit_collection_score"}
    )


def test_v5_fruit_powerup_collision_requires_strict_bomb_geometry() -> None:
    proof = {
        "feature": "fruit_powerup_collision",
        "status": "PASS",
        "framework_update": 201,
        "collision_cause": "proximity_bomb",
        "powerup_type": 0,
        "collision_radius": 108.0,
        "active_pointer_transition": "same_nonzero_to_collecting",
        "linked_powerup_manager_transition_exact": True,
        "trigger_ball_id": 9,
        "distance_squared": 10_000.0,
        "strict_radius_squared": 11_664.0,
    }
    report = {
        "version": 5,
        "fruit_runtime_state_restored": True,
        "source_authorized_features": ["fruit_powerup_collision"],
        "source_feature_proofs": [proof],
        "ticks": [
            {
                "status": "PASS",
                "framework_update": 201,
                "fruit_state_mismatches": {},
                "active_identity_match": True,
                "latent_mismatches": {},
                "score_pc": 5_300,
                "score_simulator": 5_300,
                "score_delta_pc": 700,
                "score_delta_simulator": 700,
                "events": {
                    "fruits_collected": 1,
                    "powerups_triggered": 1,
                    "balls_exploded": 3,
                },
            }
        ],
    }

    assert fidelity_gate._gameplay_authorized_features(report) == frozenset(
        {"fruit_powerup_collision"}
    )

    proof["distance_squared"] = 11_664.0
    assert fidelity_gate._gameplay_authorized_features(report) == frozenset()


def test_v5_short_fruit_window_does_not_bypass_long_visual_contract() -> None:
    exact_tick = {
        "status": "PASS",
        "framework_update": 1,
        "fruit_state_mismatches": {},
    }
    short = {
        "version": 5,
        "fruit_runtime_state_restored": True,
        "ticks": [exact_tick],
    }
    assert fidelity_gate._gameplay_fruit_contract_reason(
        short,
        evidence_root=None,
    ) is None

    long = {
        **short,
        "ticks": [
            {**exact_tick, "framework_update": update}
            for update in range(500)
        ],
    }
    assert fidelity_gate._gameplay_fruit_contract_reason(
        long,
        evidence_root=None,
    ) == "long v4/v5 gameplay differential lacks a bound fruit visual proof"


def test_fruit_scheduler_core_diff_requires_exact_source_and_simulator_state() -> None:
    outputs = [1_593_818_000, 615_627_736, 467_837_514]
    report = {
        "start_update": 10_582,
        "end_update": 10_583,
        "compared_tick_count": 1,
        "fruit_runtime_state_restored": True,
        "source_authorized_features": ["fruit_scheduler_spawn"],
        "source_feature_proofs": [
            {
                "feature": "fruit_scheduler_spawn",
                "status": "PASS",
                "framework_update": 10_583,
                "selected_point_index": 1,
                "expiry_time": 7_994,
                "mtrand_outputs": outputs,
                "mtrand_total_draw_count": 3,
                "mtrand_scheduler_draw_count": 2,
                "mtrand_state_exact": True,
                "active_pointer_transition": "zero_to_nonzero",
                "fruit_reset_state_exact": True,
            }
        ],
        "transition": {
            "source_before_active": False,
            "source_after_active": True,
            "source_selected_point_index": 1,
            "simulator_selected_point_index": 1,
            "source_expiry_time": 7_994,
            "simulator_expiry_time": 7_994,
            "scheduler_phase_fruit_state_mismatches": {},
            "full_tick_fruit_state_mismatches": {},
            "scheduler_phase_fruit_chance_draws": 1,
            "scheduler_phase_fruits_spawned": 1,
            "full_tick_fruit_chance_draws": 1,
            "full_tick_fruits_spawned": 1,
        },
        "mtrand": {
            "source_total_draw_count": 3,
            "source_outputs": outputs,
            "scheduler_draw_count": 2,
            "scheduler_prefix_state_exact": True,
            "full_tick_final_state_exact": True,
        },
    }

    assert fidelity_gate._fruit_scheduler_authorized_features(
        report
    ) == frozenset({"fruit_scheduler_spawn"})

    report["transition"]["simulator_expiry_time"] = 7_995
    assert fidelity_gate._fruit_scheduler_authorized_features(
        report
    ) == frozenset()


def test_fruit_expiry_diff_requires_exact_natural_boundary() -> None:
    report = {
        "start_update": 10_289,
        "end_update": 10_290,
        "compared_tick_count": 1,
        "fruit_runtime_state_restored": True,
        "source_authorized_features": ["fruit_expiry"],
        "source_feature_proofs": [
            {
                "feature": "fruit_expiry",
                "status": "PASS",
                "start_update": 10_289,
                "framework_update": 10_290,
                "native_game_time_before": 6_700,
                "native_game_time_after": 6_701,
                "expiry_time": 6_701,
                "active_pointer_transition": "nonzero_to_zero",
                "collecting_before": False,
                "collecting_after": False,
                "score_delta": 0,
            }
        ],
        "transition": {
            "source_before_active": True,
            "source_after_active": False,
            "source_collecting_before": False,
            "source_collecting_after": False,
            "source_native_game_time_before": 6_700,
            "source_native_game_time_after": 6_701,
            "source_expiry_time": 6_701,
            "source_selected_point_index_after": 0,
            "source_score_delta": 0,
            "simulator_active_after_scheduler": False,
            "simulator_expiry_time": 6_701,
            "scheduler_phase_fruit_state_mismatches": {},
            "full_tick_fruit_state_mismatches": {},
            "scheduler_phase_fruit_chance_draws": 0,
            "scheduler_phase_fruits_spawned": 0,
            "scheduler_phase_fruits_expired": 1,
            "scheduler_phase_fruits_collected": 0,
            "full_tick_fruit_chance_draws": 0,
            "full_tick_fruits_spawned": 0,
            "full_tick_fruits_expired": 1,
            "full_tick_fruits_collected": 0,
        },
        "mtrand": {"authorization_dependency": "none"},
    }

    assert fidelity_gate._fruit_expiry_authorized_features(
        report
    ) == frozenset({"fruit_expiry"})

    report["transition"]["source_score_delta"] = 250
    assert fidelity_gate._fruit_expiry_authorized_features(
        report
    ) == frozenset()

def test_swap_proof_requires_exact_quiescent_simulator_transition() -> None:
    def tick(
        update: int,
        *,
        current_id: int,
        next_id: int,
        current_color: int,
        next_color: int,
    ) -> dict[str, object]:
        return {
            "status": "PASS",
            "framework_update": update,
            "active_identity_match": True,
            "active_color_match": True,
            "active_count_pc": 58,
            "active_count_simulator": 58,
            "shooter_observed_match": True,
            "current_ball_id_pc": current_id,
            "next_ball_id_pc": next_id,
            "current_color_id_pc": current_color,
            "current_color_id_simulator": current_color,
            "next_color_id_pc": next_color,
            "next_color_id_simulator": next_color,
            "fired_identity_match": True,
            "fired_count_pc": 0,
            "fired_count_simulator": 0,
            "staging_identity_match": True,
            "staging_count_pc": 0,
            "staging_count_simulator": 0,
            "latent_mismatches": {},
            "qrand_observed_match": True,
            "qrand_update_count_pc": 3,
            "qrand_update_count_simulator": 3,
            "qrand_selected_index_pc": 1,
            "qrand_selected_index_simulator": 1,
            "thread_crt_rand_observed_match": True,
            "thread_crt_rand_state_pc": 123,
            "thread_crt_rand_state_simulator": 123,
            "global_mtrand_observed_match": True,
            "global_mtrand_index_pc": 538,
            "global_mtrand_index_simulator": 538,
            "score_pc": 7950,
            "score_simulator": 7950,
            "events": {
                "fired": 0,
                "hits": 0,
                "inserted": 0,
                "matches": 0,
                "balls_exploded": 0,
                "balls_removed": 0,
                "score_delta": 0,
            },
        }

    report = {
        "source_authorized_features": ["swap"],
        "source_feature_proofs": [
            {
                "feature": "swap",
                "status": "PASS",
                "framework_update": 12,
                "current_ball_id_before": 47,
                "next_ball_id_before": 67,
                "current_ball_id_after": 67,
                "next_ball_id_after": 47,
                "qrand_unchanged": True,
                "thread_crt_rand_unchanged": True,
                "score_and_curve_lists_unchanged": True,
            }
        ],
        "input_results": [
            {
                "sequence": 1,
                "framework_update": 11,
                "kind": "mouse_button",
                "accepted": True,
            }
        ],
        "ticks": [
            tick(
                11,
                current_id=47,
                next_id=67,
                current_color=1,
                next_color=3,
            ),
            tick(
                12,
                current_id=67,
                next_id=47,
                current_color=3,
                next_color=1,
            ),
        ],
    }

    assert fidelity_gate._gameplay_authorized_features(report) == frozenset(
        {"input_cadence", "swap"}
    )

    report["ticks"][1]["qrand_update_count_simulator"] = 4
    assert fidelity_gate._gameplay_authorized_features(report) == frozenset(
        {"input_cadence"}
    )

    report["ticks"][1]["qrand_update_count_simulator"] = 3
    report["ticks"][1]["current_color_id_simulator"] = 1
    assert fidelity_gate._gameplay_authorized_features(report) == frozenset(
        {"input_cadence"}
    )


def test_gameplay_scope_boundary_must_bind_the_final_tick() -> None:
    report = {
        "end_update": 3479,
        "synchronize_shooter": False,
        "shooter_mismatch_updates": [],
        "scope_boundary": {
            "artifact_sha256": "sha256:" + "a" * 64,
            "schema": "zuma-rl.pc-ambient-mtrand-boundary",
            "version": 1,
            "status": "PASS",
            "classification": (
                "natural-gameplay-slice-boundary-before-shadow-canopy-"
                "shared-rng"
            ),
            "last_included_update": 3479,
            "first_excluded_update": 3480,
            "excluded_scope": (
                "shadow_canopy_ambient_visual_rng_consumers"
            ),
        },
    }

    assert (
        fidelity_gate._noncertifying_report_reason(
            report,
            schema="zuma-rl.pc-gameplay-simulator-diff",
        )
        is None
    )

    report["scope_boundary"]["first_excluded_update"] = 3481
    assert "scope-boundary proof" in str(
        fidelity_gate._noncertifying_report_reason(
            report,
            schema="zuma-rl.pc-gameplay-simulator-diff",
        )
    )


def test_legacy_external_mtrand_reconciliation_is_noncertifying() -> None:
    report = {
        "version": 2,
        "synchronize_shooter": False,
        "shooter_mismatch_updates": [],
        "global_mtrand_policy": (
            "conditional_external_visual_draw_reconciliation_v1"
        ),
        "global_mtrand_reconciliation": {
            "enabled": True,
            "classification": (
                "excluded_rendering_only_shared_rng_consumers"
            ),
            "maximum_draws_per_tick": 64,
            "reconciled_tick_count": 0,
            "total_reconciled_draw_count": 0,
            "rows": [],
        },
        "ticks": [],
    }

    reason = fidelity_gate._noncertifying_report_reason(
        report,
        schema="zuma-rl.pc-gameplay-simulator-diff",
    )

    assert "may conceal gameplay-relevant RNG" in str(reason)


def test_exact_v3_mtrand_transcript_remains_certifying() -> None:
    report = {
        "version": 3,
        "synchronize_shooter": False,
        "shooter_mismatch_updates": [],
        "global_mtrand_state_restored": True,
        "global_mtrand_policy": "exact_no_reconciliation",
        "global_mtrand_reconciliation": {
            "enabled": False,
            "classification": "no_reconciliation",
            "state_binding": (
                "sha256_of_624_words_plus_index_le_u32"
            ),
            "maximum_draws_per_tick": 64,
            "reconciled_tick_count": 0,
            "total_reconciled_draw_count": 0,
            "rows": [],
        },
        "ticks": [
            {
                "status": "PASS",
                "global_mtrand_observed_match": True,
                "global_mtrand_reconciled_draws": 0,
                "global_mtrand_leading_reconciled_draws": 0,
            }
        ],
    }

    assert (
        fidelity_gate._noncertifying_report_reason(
            report,
            schema="zuma-rl.pc-gameplay-simulator-diff",
        )
        is None
    )

    report["global_mtrand_reconciliation"][
        "maximum_draws_per_tick"
    ] = 65
    assert "non-exact transcript" in str(
        fidelity_gate._noncertifying_report_reason(
            report,
            schema="zuma-rl.pc-gameplay-simulator-diff",
        )
    )


def test_bound_v3_mtrand_reconciliation_requires_independent_proof() -> None:
    digest = "sha256:" + "a" * 64
    tick = {
        "status": "PASS",
        "framework_update": 11,
        "gameplay_state_matched_before_global_rng": True,
        "global_mtrand_observed_match": True,
        "global_mtrand_reconciled_draws": 1,
        "global_mtrand_leading_reconciled_draws": 0,
        "active_identity_match": True,
        "active_color_match": True,
        "pending_colors_match": True,
        "fired_identity_match": True,
        "staging_identity_match": True,
        "qrand_observed_match": True,
        "thread_crt_rand_observed_match": True,
        "score_pc": 10,
        "score_simulator": 10,
        "score_delta_pc": 0,
        "score_delta_simulator": 0,
        "latent_mismatches": {},
        "fired_latent_mismatches": {},
        "staging_latent_mismatches": {},
        "curve_state_mismatches": {},
    }
    report = {
        "version": 3,
        "synchronize_shooter": False,
        "shooter_mismatch_updates": [],
        "global_mtrand_state_restored": True,
        "global_mtrand_policy": (
            "conditional_external_mtrand_reconciliation_v2"
        ),
        "global_mtrand_reconciliation": {
            "enabled": True,
            "classification": (
                "unclassified_until_bound_call_trace_proof"
            ),
            "state_binding": (
                "sha256_of_624_words_plus_index_le_u32"
            ),
            "maximum_draws_per_tick": 2048,
            "reconciled_tick_count": 1,
            "total_reconciled_draw_count": 1,
            "rows": [
                {
                    "framework_update": 11,
                    "phase": "after_gameplay_state_match",
                    "draw_count": 1,
                    "simulator_state_sha256_before": digest,
                    "simulator_state_sha256_after": digest,
                    "pc_state_sha256": digest,
                    "gameplay_state_matched_before_reconciliation": True,
                    "full_state_reached_exactly": True,
                }
            ],
        },
        "ticks": [tick],
    }

    reason = fidelity_gate._noncertifying_report_reason(
        report,
        schema="zuma-rl.pc-gameplay-simulator-diff",
    )
    assert "independent bound call-trace proof" in str(reason)

    report["global_mtrand_reconciliation"][
        "total_reconciled_draw_count"
    ] = 2
    reason = fidelity_gate._noncertifying_report_reason(
        report,
        schema="zuma-rl.pc-gameplay-simulator-diff",
    )
    assert "draw total" in str(reason)


def test_bound_v3_mtrand_proof_is_loaded_hashed_and_recomputed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    digest = "sha256:" + "a" * 64
    tick = {
        "status": "PASS",
        "framework_update": 11,
        "gameplay_state_matched_before_global_rng": True,
        "global_mtrand_observed_match": True,
        "global_mtrand_reconciled_draws": 1,
        "global_mtrand_leading_reconciled_draws": 0,
        "active_identity_match": True,
        "active_color_match": True,
        "pending_colors_match": True,
        "fired_identity_match": True,
        "staging_identity_match": True,
        "qrand_observed_match": True,
        "thread_crt_rand_observed_match": True,
        "score_pc": 10,
        "score_simulator": 10,
        "score_delta_pc": 0,
        "score_delta_simulator": 0,
        "latent_mismatches": {},
        "fired_latent_mismatches": {},
        "staging_latent_mismatches": {},
        "curve_state_mismatches": {},
    }
    report = {
        "schema": "zuma-rl.pc-gameplay-simulator-diff",
        "version": 3,
        "status": "PASS",
        "start_update": 10,
        "end_update": 11,
        "compared_tick_count": 1,
        "trajectory": {"artifact_sha256": digest},
        "dmo": {"artifact_sha256": digest},
        "synchronize_shooter": False,
        "shooter_mismatch_updates": [],
        "global_mtrand_state_restored": True,
        "global_mtrand_policy": (
            "conditional_external_mtrand_reconciliation_v2"
        ),
        "global_mtrand_reconciliation": {
            "enabled": True,
            "classification": (
                "unclassified_until_bound_call_trace_proof"
            ),
            "state_binding": (
                "sha256_of_624_words_plus_index_le_u32"
            ),
            "maximum_draws_per_tick": 2048,
            "reconciled_tick_count": 1,
            "total_reconciled_draw_count": 1,
            "rows": [
                {
                    "framework_update": 11,
                    "phase": "after_gameplay_state_match",
                    "draw_count": 1,
                    "simulator_state_sha256_before": digest,
                    "simulator_state_sha256_after": digest,
                    "pc_state_sha256": digest,
                    "gameplay_state_matched_before_reconciliation": True,
                    "full_state_reached_exactly": True,
                }
            ],
        },
        "ticks": [tick],
    }
    transcript = fidelity_gate.reconciliation_transcript_sha256(report)
    proof_path = tmp_path / "proof.json"
    proof_path.write_text("{}\n", encoding="ascii")
    report["global_mtrand_reconciliation_proof"] = {
        "artifact": "proof.json",
        "artifact_sha256": _digest(proof_path),
        "schema": fidelity_gate._MTRAND_RECONCILIATION_PROOF_SCHEMA,
        "version": 1,
        "status": "PASS",
        "classification": (
            fidelity_gate.MTRAND_RECONCILIATION_PROOF_CLASSIFICATION
        ),
        "reconciliation_transcript_sha256": transcript,
    }
    calls: list[Path] = []

    def verify(
        path: Path,
        *,
        report: object,
        evidence_root: Path,
    ) -> dict[str, object]:
        calls.append(path)
        assert evidence_root == tmp_path
        return {
            "gameplay_diff_binding": {
                "reconciliation_transcript_sha256": transcript,
            },
            "reconciliation": {
                "reconciled_tick_count": 1,
                "reconciled_draw_count": 1,
            },
        }

    monkeypatch.setattr(
        fidelity_gate,
        "verify_mtrand_reconciliation_proof_binding",
        verify,
    )
    assert (
        fidelity_gate._noncertifying_report_reason(
            report,
            schema=report["schema"],
            evidence_root=tmp_path,
        )
        is None
    )
    assert calls == [proof_path]

    report["global_mtrand_reconciliation"][
        "maximum_draws_per_tick"
    ] = 2049
    assert "totals are invalid" in str(
        fidelity_gate._noncertifying_report_reason(
            report,
            schema=report["schema"],
            evidence_root=tmp_path,
        )
    )
    report["global_mtrand_reconciliation"][
        "maximum_draws_per_tick"
    ] = 2048

    proof_path.write_text("tampered\n", encoding="ascii")
    reason = fidelity_gate._noncertifying_report_reason(
        report,
        schema=report["schema"],
        evidence_root=tmp_path,
    )
    assert "proof hash mismatch" in str(reason)


def test_merge_direction_and_powerup_spawn_require_exact_fields() -> None:
    merge = {
        "initial_hit_in_front": False,
        "initial_hit_ball_id": 41,
        "initial_hit_index": 7,
        "pc_insertion_update": 100,
        "simulator_insertion_update": 100,
        "ticks": [
            {
                "status": "PASS",
                "framework_update": 100,
                "active_identity_match": True,
                "staging_identity_match": True,
                "events": {"inserted": 1},
            }
        ],
    }
    spawn = {
        "exact_ball_and_manager_state": True,
        "selected_ball_id": 86,
        "selected_color_id": 2,
        "selected_powerup_type": 3,
        "mtrand_index_before": 19,
        "mtrand_index_after": 20,
    }

    assert fidelity_gate._merge_authorized_features(merge) == frozenset(
        {"back_insertion"}
    )
    assert fidelity_gate._powerup_spawn_authorized_features(
        spawn
    ) == frozenset({"powerup_spawn"})
    spawn["selected_powerup_type"] = 0
    assert fidelity_gate._powerup_spawn_authorized_features(
        spawn
    ) == frozenset({"powerup_spawn"})
    spawn["selected_powerup_type"] = 14
    assert not fidelity_gate._powerup_spawn_authorized_features(spawn)
    assert fidelity_gate._simulator_authorized_features(
        {
            "status": "PASS",
            "target": {"powerup_type": 3},
        },
        schema="zuma-rl.pc-powerup-lifecycle-core-diff",
    ) == frozenset()


def test_natural_win_diff_requires_exact_two_tick_oracle() -> None:
    report = {
        "schema": "zuma-rl.pc-natural-win-simulator-diff",
        "version": 1,
        "status": "PASS",
        "start_update": 106,
        "empty_update": 107,
        "formal_transition_update": 108,
        "end_update": 108,
        "ticks_compared": 2,
        "initial_chain_count": 3,
        "balls_removed": 3,
        "maximum_waypoint_error": 0.0,
        "score_target_achieved": True,
        "plan_exhausted": True,
        "stop_adding_transplanted": True,
        "pc_empty_state_matched": True,
        "pc_formal_state_matched": True,
        "scenario": {
            "shooter_rng_state_restored": True,
            "global_mtrand_state_restored": True,
        },
        "oracle": {
            "schema": "zuma-rl.pc-natural-win-sequence",
            "version": 1,
            "status": "PASS",
            "score_target": 100,
            "score_cross_update": 102,
            "feed_exhaustion_update": 103,
            "empty_update": 107,
            "formal_transition_update": 108,
        },
    }

    assert fidelity_gate._simulator_authorized_features(
        report,
        schema=report["schema"],
    ) == frozenset({"natural_win", "zuma_transition"})

    report["balls_removed"] = 2
    assert fidelity_gate._simulator_authorized_features(
        report,
        schema=report["schema"],
    ) == frozenset()


def test_natural_win_v2_accepts_source_closed_feed_and_projectile_exit() -> None:
    report = {
        "schema": "zuma-rl.pc-natural-win-simulator-diff",
        "version": 2,
        "status": "PASS",
        "start_update": 7491,
        "empty_update": 7492,
        "formal_transition_update": 7493,
        "end_update": 7493,
        "ticks_compared": 2,
        "terminal_object_kind": "free_projectile",
        "initial_chain_count": 0,
        "initial_free_projectile_count": 1,
        "initial_gameplay_entity_count": 1,
        "gameplay_entities_removed": 1,
        "balls_removed": 0,
        "maximum_waypoint_error": 0.0,
        "score_target_achieved": True,
        "plan_exhausted": True,
        "stop_adding_transplanted": True,
        "pc_empty_state_matched": True,
        "pc_formal_state_matched": True,
        "pc_terminal_chamber_cleared": True,
        "scenario": {
            "shooter_rng_state_restored": True,
            "global_mtrand_state_restored": True,
        },
        "oracle": {
            "schema": "zuma-rl.pc-natural-win-sequence",
            "version": 1,
            "status": "PASS",
            "score_target": 9650,
            "score_cross_update": 7045,
            "feed_exhaustion_update": None,
            "feed_closed_at_capture_start": True,
            "empty_update": 7492,
            "formal_transition_update": 7493,
        },
    }

    assert fidelity_gate._simulator_authorized_features(
        report,
        schema=report["schema"],
    ) == frozenset({"natural_win", "zuma_transition"})

    report["pc_terminal_chamber_cleared"] = False
    assert fidelity_gate._simulator_authorized_features(
        report,
        schema=report["schema"],
    ) == frozenset()


def test_tunnel_collision_diff_requires_exact_suppressed_hit() -> None:
    proof = {
        "feature": "tunnel_collision",
        "status": "PASS",
        "from_update": 6952,
        "framework_update": 6953,
        "projectile_ball_id": 203,
        "projectile_color_id": 1,
        "projectile_position_before": [64.5, 110.5],
        "projectile_position_after": [57.6, 106.4],
        "projectile_velocity": [-6.8, -4.0],
        "tunnel_overlaps": [{"hit_ball_id": 200}],
        "non_tunnel_overlap_count": 0,
        "staging_list_empty": True,
        "projectile_remained_free": True,
    }
    report = {
        "schema": "zuma-rl.pc-tunnel-collision-simulator-diff",
        "version": 1,
        "status": "PASS",
        "start_update": 6952,
        "end_update": 6953,
        "compared_tick_count": 1,
        "source_authorized_features": ["tunnel_collision"],
        "source_feature_proofs": [proof],
        "source_projectile": {
            "ball_id": 203,
            "color_id": 1,
            "position_before": [64.5, 110.5],
            "position_after": [57.6, 106.4],
            "velocity": [-6.8, -4.0],
            "remained_free": True,
            "staging_count_before": 0,
            "staging_count_after": 0,
        },
        "simulator_transition": {
            "projectile_count_after": 1,
            "staging_count_after": 0,
            "projectile_identity_match": True,
            "projectile_position_error_px": 0.0,
            "projectile_waypoint_error": 0.0,
            "projectile_progress_error": 0.0,
            "projectile_latent_mismatches": {},
            "active_identity_match": True,
            "active_color_match": True,
            "active_waypoint_error": 0.0,
            "active_position_error_px": 1.9e-6,
            "active_latent_mismatches": {},
            "pending_colors_match": True,
            "curve_state_mismatches": {},
            "fruit_state_mismatches": {},
            "shooter_state_match": True,
            "rng_state_matches": {
                "qrand": True,
                "thread_crt_rand": True,
                "global_mtrand": True,
            },
            "score_source": 9630,
            "score_simulator": 9630,
            "events": {
                "fired": 0,
                "hits": 0,
                "inserted": 0,
                "matches": 0,
                "balls_exploded": 0,
                "balls_removed": 0,
                "score_delta": 0,
            },
            "collision_suppressed": True,
            "exact_native_state_match": True,
        },
        "scenario": {
            "shooter_rng_state_restored": True,
            "global_mtrand_state_restored": True,
        },
    }

    assert fidelity_gate._simulator_authorized_features(
        report,
        schema=report["schema"],
    ) == frozenset({"tunnel_collision"})

    report["simulator_transition"]["collision_suppressed"] = False
    assert fidelity_gate._simulator_authorized_features(
        report,
        schema=report["schema"],
    ) == frozenset()


def test_suite_rejects_path_escape_and_unsorted_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_test_policy(monkeypatch, _test_policy())
    base = {
        "schema": fidelity_gate.SUITE_SCHEMA,
        "version": fidelity_gate.SUITE_VERSION,
        "policy": "test-policy",
        "evidence": [
            {
                "id": "case",
                "kind": EvidenceKind.DIAGNOSTIC.value,
                "path": "../outside.json",
                "sha256": "sha256:" + "0" * 64,
                "features": ["shot_release"],
            }
        ],
    }
    with pytest.raises(
        FidelityGateValidationError,
        match="normalized relative POSIX",
    ):
        FidelitySuite.from_dict(base)

    base["evidence"][0]["path"] = "inside.json"
    base["evidence"][0]["features"] = ["swap", "shot_release"]
    with pytest.raises(
        FidelityGateValidationError,
        match="unique and sorted",
    ):
        FidelitySuite.from_dict(base)


def test_duplicate_json_keys_make_suite_invalid(tmp_path: Path) -> None:
    suite = tmp_path / "suite.json"
    suite.write_text(
        '{"schema":"a","schema":"b"}',
        encoding="utf-8",
    )

    report = verify_fidelity_suite(suite)

    assert report.status is FidelityGateStatus.INVALID
    assert "duplicate JSON key" in report.reasons[0]
