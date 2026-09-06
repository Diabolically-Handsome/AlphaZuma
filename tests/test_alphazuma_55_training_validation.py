from pathlib import Path

import pytest

from tools import build_alphazuma_55_training_validation as validation


def _fake_base_contracts(**kwargs):
    models = kwargs["models"]
    return (
        {
            "schema": "zuma-rl.zero-shot-models-manifest",
            "version": 1,
            "status": "FROZEN",
            "stage": "engineering",
            "models": models,
        },
        {
            "schema": "zuma-rl.zero-shot-multilevel-preregistration",
            "version": 1,
            "status": "FROZEN_BEFORE_EVALUATION",
            "model": models[0],
            "levels": [{"id": f"level-{index}"} for index in range(55)],
            "attempts_per_level": 1,
            "total_attempts": 55,
            "seed_plan": {
                "registry": "engineering_and_calibration",
                "base_seed": 1_400_200_000,
                "last_seed": 1_400_200_054,
            },
            "environment": {"base_config": {"max_ticks": 12_000}},
            "execution": {
                "output_root": str(kwargs["output_root"]),
                "shard_count": 2,
                "devices": ["cuda:0", "cuda:1"],
                "parallel_envs_per_shard": kwargs["parallel_envs_per_shard"],
            },
        },
    )


def test_training_validation_rebinds_only_nonformal_seed_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(validation, "build_contracts", _fake_base_contracts)
    monkeypatch.setattr(
        validation,
        "_read_json",
        lambda _path: {
            "seed_registry": {
                "training_validation": {
                    "first": 1_550_000_000,
                    "last": 1_599_999_999,
                }
            }
        },
    )
    monkeypatch.setattr(validation, "_sha256", lambda _path: "sha256:test")

    models = [
        {"id": "source", "training_steps": 1, "path": "a", "sha256": "x"},
        {"id": "checkpoint", "training_steps": 2, "path": "b", "sha256": "y"},
    ]
    manifest, preregistration = validation.build_training_validation_contracts(
        master_path=tmp_path / "master.json",
        original_root=tmp_path / "retail",
        evaluator_path=tmp_path / "evaluator.py",
        output_root=tmp_path / "output",
        models=models,
        base_seed=1_550_000_000,
        device="cuda:0",
        parallel_envs=55,
    )

    assert manifest["stage"] == "training_validation"
    assert preregistration["seed_plan"] == {
        "registry": "training_validation",
        "base_seed": 1_550_000_000,
        "last_seed": 1_550_000_054,
    }
    assert preregistration["execution"]["shard_count"] == 1
    assert preregistration["execution"]["devices"] == ["cuda:0"]
    assert preregistration["diagnostic_contract"] == {
        "paired_models_share_identical_task_seeds": True,
        "formal_selection_authority": False,
        "training_recipe_change_authority": False,
        "formal_seed_ranges_consumed": False,
        "max_ticks": 12_000,
    }


def test_training_validation_rejects_seed_range_overflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(validation, "build_contracts", _fake_base_contracts)
    monkeypatch.setattr(
        validation,
        "_read_json",
        lambda _path: {
            "seed_registry": {
                "training_validation": {
                    "first": 1_550_000_000,
                    "last": 1_550_000_010,
                }
            }
        },
    )

    with pytest.raises(ValueError, match="exceeds its seed registry"):
        validation.build_training_validation_contracts(
            master_path=tmp_path / "master.json",
            original_root=tmp_path / "retail",
            evaluator_path=tmp_path / "evaluator.py",
            output_root=tmp_path / "output",
            models=[
                {
                    "id": "source",
                    "training_steps": 1,
                    "path": "a",
                    "sha256": "x",
                }
            ],
            base_seed=1_550_000_000,
            device="cuda:0",
            parallel_envs=55,
        )
