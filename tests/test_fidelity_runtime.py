"""Tests for the native Windows Fidelity verifier bridge."""

from __future__ import annotations

import json
import subprocess

import pytest

from zuma_rl import fidelity_runtime
from zuma_rl.fidelity_gate import FidelityGateReport, FidelityGateStatus


def test_non_wsl_runtime_uses_in_process_verifier(monkeypatch) -> None:
    expected = FidelityGateReport(
        status=FidelityGateStatus.CLOSED,
        policy="original-transfer-jungle2-v3",
    )
    monkeypatch.setattr(fidelity_runtime, "_is_wsl", lambda: False)
    monkeypatch.setattr(
        fidelity_runtime,
        "verify_fidelity_suite",
        lambda *args, **kwargs: expected,
    )

    assert (
        fidelity_runtime.verify_fidelity_suite_for_training(
            "suite.json",
            suite_root="root",
            original_root="original",
        )
        is expected
    )


def test_native_report_parser_preserves_all_gate_rows() -> None:
    raw = {
        "schema": "zuma-rl.fidelity-gate-report",
        "version": 2,
        "status": "CLOSED",
        "policy": "original-transfer-jungle2-v3",
        "evidence": [{"id": "a"}],
        "requirements": [{"feature": "fruit_expiry", "status": "MISSING"}],
        "reasons": ["missing certifying evidence: fruit_expiry"],
        "summary": {"gate_open": False},
    }

    report = fidelity_runtime._report_from_mapping(raw)

    assert report.status is FidelityGateStatus.CLOSED
    assert report.evidence == ({"id": "a"},)
    assert report.requirements[0]["feature"] == "fruit_expiry"


def test_wsl_native_verifier_uses_full_suite_timeout(monkeypatch) -> None:
    raw = {
        "schema": "zuma-rl.fidelity-gate-report",
        "version": 2,
        "status": "OPEN",
        "policy": "original-transfer-jungle2-v4",
        "evidence": [],
        "requirements": [],
        "reasons": [],
        "summary": {"gate_open": True},
    }
    observed: dict[str, object] = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["timeout"] = kwargs["timeout"]
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(raw) + "\n",
            stderr="",
        )

    monkeypatch.setattr(fidelity_runtime, "_is_wsl", lambda: True)
    monkeypatch.setattr(
        fidelity_runtime,
        "_windows_python",
        lambda root: root / ".venv-win" / "Scripts" / "python.exe",
    )
    monkeypatch.setattr(
        fidelity_runtime,
        "_windows_path",
        lambda path: str(path),
    )
    monkeypatch.setattr(fidelity_runtime.subprocess, "run", fake_run)

    report = fidelity_runtime.verify_fidelity_suite_for_training(
        "suite.json",
        suite_root="root",
        original_root="original",
    )

    assert report.status is FidelityGateStatus.OPEN
    assert observed["timeout"] == 900
    assert (
        fidelity_runtime.NATIVE_FIDELITY_TIMEOUT_SECONDS
        == observed["timeout"]
    )


def test_wsl_native_verifier_timeout_is_domain_error(monkeypatch) -> None:
    def fake_run(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(fidelity_runtime, "_is_wsl", lambda: True)
    monkeypatch.setattr(
        fidelity_runtime,
        "_windows_python",
        lambda root: root / ".venv-win" / "Scripts" / "python.exe",
    )
    monkeypatch.setattr(
        fidelity_runtime,
        "_windows_path",
        lambda path: str(path),
    )
    monkeypatch.setattr(fidelity_runtime.subprocess, "run", fake_run)

    with pytest.raises(
        fidelity_runtime.FidelityRuntimeError,
        match="timed out after 900 seconds",
    ):
        fidelity_runtime.verify_fidelity_suite_for_training(
            "suite.json",
            suite_root="root",
            original_root="original",
        )
