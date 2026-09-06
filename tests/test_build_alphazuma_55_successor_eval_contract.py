from __future__ import annotations

from pathlib import Path

import pytest

from tools import build_alphazuma_55_successor_eval_contract as successor
from tools.build_alphazuma_55_successor_eval_contract import (
    _validated_stage_spec,
)


def test_successor_final_blind_matrix_is_fixed_at_eight_attempts() -> None:
    assert _validated_stage_spec(
        stage="final_blind",
        base_seed=2_100_000_000,
        attempts_per_level=8,
    ) == {"base_seed": 2_100_000_000, "attempts": 8}
    with pytest.raises(ValueError, match="requires 8"):
        _validated_stage_spec(
            stage="final_blind",
            base_seed=2_100_000_000,
            attempts_per_level=4,
        )


def test_successor_continuous_matrix_is_fixed_at_four_campaigns() -> None:
    assert _validated_stage_spec(
        stage="continuous",
        base_seed=2_200_000_000,
        attempts_per_level=4,
    ) == {"base_seed": 2_200_000_000, "attempts": 4}


def test_successor_contract_rejects_unknown_stage_and_uint32_overflow() -> None:
    with pytest.raises(ValueError, match="stage"):
        _validated_stage_spec(
            stage="selection",
            base_seed=2_000_000_000,
            attempts_per_level=4,
        )
    with pytest.raises(ValueError, match="uint32"):
        _validated_stage_spec(
            stage="final_blind",
            base_seed=2**32 - 100,
            attempts_per_level=8,
        )


def test_successor_contract_marks_seed_matrix_frozen_before_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    master = tmp_path / "master.json"
    original = tmp_path / "original"
    evaluator = tmp_path / "evaluator.py"
    model = tmp_path / "model.zip"
    original.mkdir()
    master.write_text("{}", encoding="utf-8")
    evaluator.write_text("# evaluator\n", encoding="utf-8")
    model.write_bytes(b"model")
    master_value = {
        "schema": "zuma-rl.alphazuma-55-weekend-master-preregistration",
        "version": 1,
        "status": "FROZEN_BEFORE_SUCCESSOR_FINAL_BLIND",
        "seed_registry": {
            "final_blind": {"first": 2_100_000_000, "last": 2_100_000_439}
        },
        "execution": {
            "shard_count": 2,
            "devices": ["cuda:0", "cuda:0"],
            "parallel_envs_per_shard": 12,
        },
    }
    monkeypatch.setattr(successor.legacy, "_read_json", lambda _: master_value)

    def fake_build_contracts(**kwargs: object) -> tuple[dict, dict]:
        assert successor.legacy.STAGES["final_blind"] == {
            "base_seed": 2_100_000_000,
            "attempts": 8,
        }
        return (
            {"models": kwargs["models"]},
            {
                "attempts_per_level": 8,
                "total_attempts": 440,
                "seed_plan": {
                    "base_seed": 2_100_000_000,
                    "last_seed": 2_100_000_439,
                },
                "execution": {
                    "shard_count": 2,
                    "devices": ["cuda:0", "cuda:1"],
                    "parallel_envs_per_shard": 12,
                },
            },
        )

    monkeypatch.setattr(successor.legacy, "build_contracts", fake_build_contracts)
    model_spec = {
        "id": "candidate",
        "training_steps": 1,
        "path": str(model),
        "sha256": successor.legacy._sha256(model),
    }
    _, preregistration = successor.build_successor_contracts(
        master_path=master,
        original_root=original,
        evaluator_path=evaluator,
        stage="final_blind",
        base_seed=2_100_000_000,
        attempts_per_level=8,
        output_root=tmp_path / "output",
        models=[model_spec],
        parallel_envs_per_shard=12,
    )
    assert preregistration["seed_plan"][
        "all_seeds_frozen_before_policy_inference"
    ] is True


def test_successor_contract_rejects_changed_model_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    master = tmp_path / "master.json"
    original = tmp_path / "original"
    evaluator = tmp_path / "evaluator.py"
    model = tmp_path / "model.zip"
    original.mkdir()
    master.write_text("{}", encoding="utf-8")
    evaluator.write_text("# evaluator\n", encoding="utf-8")
    model.write_bytes(b"changed")
    monkeypatch.setattr(
        successor.legacy,
        "_read_json",
        lambda _: {
            "schema": "zuma-rl.alphazuma-55-weekend-master-preregistration",
            "version": 1,
            "status": "FROZEN_BEFORE_SUCCESSOR_FINAL_BLIND",
            "seed_registry": {
                "final_blind": {
                    "first": 2_100_000_000,
                    "last": 2_100_000_439,
                }
            },
            "execution": {
                "shard_count": 2,
                "devices": ["cuda:0", "cuda:0"],
                "parallel_envs_per_shard": 12,
            },
        },
    )
    with pytest.raises(ValueError, match="hash differs"):
        successor.build_successor_contracts(
            master_path=master,
            original_root=original,
            evaluator_path=evaluator,
            stage="final_blind",
            base_seed=2_100_000_000,
            attempts_per_level=8,
            output_root=tmp_path / "output",
            models=[
                {
                    "id": "candidate",
                    "training_steps": 1,
                    "path": str(model),
                    "sha256": "sha256:" + "0" * 64,
                }
            ],
            parallel_envs_per_shard=12,
        )
