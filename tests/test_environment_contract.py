"""Tests for the exact training-environment contract."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from zuma_rl import environment_contract


def test_v3_training_environment_configuration_is_bounded() -> None:
    config = environment_contract.training_environment_config()

    assert config.aim_bins == 180
    assert config.action_mode == "factorized"
    assert config.frame_skip == 1
    assert config.max_ticks == 12_000
    assert config.max_balls == 768


def test_runtime_source_boundary_excludes_independent_evidence_code(
    tmp_path: Path,
) -> None:
    expected = {
        "environment_runtime": set(environment_contract.ENVIRONMENT_SOURCE_PATHS),
        "policy_runtime": set(environment_contract.POLICY_SOURCE_PATHS),
    }
    for relative in sorted(set().union(*expected.values())):
        path = tmp_path.joinpath(*relative.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    evidence_only = tmp_path / "src" / "zuma_rl" / "pc_memory_trajectory.py"
    evidence_only.write_text("first evidence revision", encoding="utf-8")

    before = environment_contract._implementation_sources(tmp_path)
    evidence_only.write_text("second evidence revision", encoding="utf-8")
    after = environment_contract._implementation_sources(tmp_path)

    assert {group: set(rows) for group, rows in before.items()} == expected
    assert before == after
    assert all(
        not relative.startswith("src/zuma_rl/pc_")
        for rows in before.values()
        for relative in rows
    )


def test_environment_validator_requires_exact_live_dictionary(monkeypatch) -> None:
    live = {
        "schema": environment_contract.CONTRACT_SCHEMA,
        "version": environment_contract.CONTRACT_VERSION,
        "status": "PASS",
        "contract_fingerprint": "sha256:" + "1" * 64,
    }
    monkeypatch.setattr(
        environment_contract,
        "build_training_environment_contract",
        lambda **kwargs: deepcopy(live),
    )
    assert (
        environment_contract.validate_training_environment_contract(
            live,
            original_root="game",
        )
        is None
    )
    assert "differs" in environment_contract.validate_training_environment_contract(
        dict(live, status="FAIL"),
        original_root="game",
    )
