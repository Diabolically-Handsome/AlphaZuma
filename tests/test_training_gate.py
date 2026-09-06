"""Tests for the independent bounded-recipe Training Gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from zuma_rl import training_gate


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CURRENT_SUITE = PROJECT_ROOT / "diagnostics" / "training-suite-current.json"
VALIDATED_MODEL = Path(
    "D:/ZumaTraining/screening-teacher-offline-polar90-s20260873-v1/"
    "refined_model.zip"
)
VALIDATED_RUN = Path(
    "D:/ZumaTraining/"
    "scale-teacher-anchor-c1000-behavioral-s20260940-98k-n32-v1"
)


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.skipif(
    not VALIDATED_MODEL.is_file() or not VALIDATED_RUN.is_dir(),
    reason="local scale-stability artifacts are unavailable",
)
def test_legacy_training_suite_is_historical_after_gate_repair() -> None:
    report = training_gate.verify_training_suite(CURRENT_SUITE)

    assert report.status is training_gate.TrainingGateStatus.CLOSED
    assert report.exit_code == 1
    assert report.summary["transfer_authorizing_policy"] is False
    assert report.summary["required_fidelity_policy"] == (
        "original-transfer-jungle2-v2"
    )
    assert report.summary["authorized_effective_steps"] == 491_520
    assert report.summary["state_policy_training_only"] is True
    assert report.summary["visual_policy_training_authorized"] is False
    assert report.summary["original_game_deployment_authorized"] is False
    assert all(
        row["status"] == "PASS"
        for row in report.summary["validated_run_artifacts"]
    )


def test_recipe_contract_is_exact_and_bounded() -> None:
    request = training_gate.expected_training_request()

    assert training_gate.training_request_mismatches(request) == ()
    assert request["planned_effective_steps"] == 491_520
    assert request["seed"] == 20_260_950
    assert request["initial_model_sha256"] == (
        training_gate.EXPECTED_INITIAL_MODEL_SHA256
    )

    oversized = dict(request)
    oversized["total_steps"] = 10_000_000
    oversized["planned_effective_steps"] = 10_000_384
    mismatches = training_gate.training_request_mismatches(oversized)
    assert any("total_steps" in reason for reason in mismatches)
    assert any("planned_effective_steps" in reason for reason in mismatches)

    recipe_drift = dict(request)
    recipe_drift["teacher_kl_coef"] = 100.0
    recipe_drift["learning_rate"] = 0.0003
    mismatches = training_gate.training_request_mismatches(recipe_drift)
    assert any("teacher_kl_coef" in reason for reason in mismatches)
    assert any("learning_rate" in reason for reason in mismatches)

    migrated_sha256 = "sha256:" + "1" * 64
    bootstrap = training_gate.expected_training_request(
        policy=training_gate.TRANSFER_POLICY_ID,
        initial_model_sha256=migrated_sha256,
    )
    assert bootstrap["total_steps"] == 98_304
    assert bootstrap["planned_effective_steps"] == 98_304
    assert bootstrap["initial_model_sha256"] == migrated_sha256
    assert training_gate.training_request_mismatches(
        bootstrap,
        expected=bootstrap,
    ) == ()


def test_v2_transfer_policy_is_retained_but_historical(tmp_path: Path) -> None:
    suite = tmp_path / "legacy-transfer-suite.json"
    suite.write_text(
        json.dumps(
            {
                "schema": training_gate.SUITE_SCHEMA,
                "version": training_gate.SUITE_VERSION,
                "policy": training_gate.LEGACY_TRANSFER_POLICY_ID,
                "stage": training_gate.LEGACY_TRANSFER_STAGE_ID,
                "evidence": [],
            }
        ),
        encoding="utf-8",
    )

    report = training_gate.verify_training_suite(suite)

    assert report.status is training_gate.TrainingGateStatus.CLOSED
    assert report.summary["transfer_authorizing_policy"] is False
    assert any("historical" in reason for reason in report.reasons)


def test_unapproved_outcome_cannot_substitute_for_frozen_evidence(
    tmp_path: Path,
) -> None:
    diagnostics = PROJECT_ROOT / "diagnostics"
    preregistration = tmp_path / "preregistration.json"
    preregistration.write_bytes(
        (
            diagnostics
            / (
                "teacher-policy-anchor-c1000-scale98k-behavioral-"
                "s20260940-preregistration-v1.json"
            )
        ).read_bytes()
    )
    outcome = tmp_path / "outcome.json"
    outcome.write_text(
        json.dumps(
            {
                "schema": (
                    "zuma-rl.teacher-policy-anchor-scale-stability-outcome"
                ),
                "version": 2,
                "status": "BEHAVIORAL_SCALE_STABILITY_PASS",
            }
        ),
        encoding="utf-8",
    )
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "schema": training_gate.SUITE_SCHEMA,
                "version": training_gate.SUITE_VERSION,
                "policy": training_gate.POLICY_ID,
                "stage": training_gate.STAGE_ID,
                "evidence": [
                    {
                        "id": "a-outcome",
                        "kind": "scale_stability_outcome",
                        "path": outcome.name,
                        "sha256": _digest(outcome),
                    },
                    {
                        "id": "b-preregistration",
                        "kind": "preregistration",
                        "path": preregistration.name,
                        "sha256": _digest(preregistration),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    report = training_gate.verify_training_suite(suite)

    assert report.status is training_gate.TrainingGateStatus.CLOSED
    assert any("outcome identity is not approved" in item for item in report.reasons)


def test_content_address_mismatch_is_invalid(tmp_path: Path) -> None:
    evidence = tmp_path / "outcome.json"
    evidence.write_text("{}", encoding="utf-8")
    preregistration = tmp_path / "preregistration.json"
    preregistration.write_text("{}", encoding="utf-8")
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "schema": training_gate.SUITE_SCHEMA,
                "version": training_gate.SUITE_VERSION,
                "policy": training_gate.POLICY_ID,
                "stage": training_gate.STAGE_ID,
                "evidence": [
                    {
                        "id": "a-outcome",
                        "kind": "scale_stability_outcome",
                        "path": evidence.name,
                        "sha256": "sha256:" + "0" * 64,
                    },
                    {
                        "id": "b-preregistration",
                        "kind": "preregistration",
                        "path": preregistration.name,
                        "sha256": _digest(preregistration),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    report = training_gate.verify_training_suite(suite)

    assert report.status is training_gate.TrainingGateStatus.INVALID
    assert "SHA-256 differs" in report.reasons[0]
