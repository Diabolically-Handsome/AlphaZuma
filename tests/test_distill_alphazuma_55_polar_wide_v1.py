from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import build_alphazuma_55_polar_wide as builder
from tools import distill_alphazuma_55_polar_wide_v1 as trainer


def test_wide_policy_kwargs_only_changes_learned_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        trainer,
        "revenge_polar_policy_kwargs",
        lambda _: {
            "features_extractor_kwargs": {
                "features_dim": 256,
                "polar_aim_bins": 180,
            },
            "net_arch": {"pi": [], "vf": [256]},
        },
    )
    kwargs = trainer._wide_policy_kwargs(object(), 1024)
    assert kwargs["features_extractor_kwargs"] == {
        "features_dim": 1024,
        "polar_aim_bins": 180,
    }
    assert kwargs["net_arch"] == {"pi": [], "vf": [256]}
    with pytest.raises(ValueError, match="dimension"):
        trainer._wide_policy_kwargs(object(), 512)


def test_collection_summary_aggregates_three_teacher_passes() -> None:
    collections = [
        {
            "retained_samples": 100 + index,
            "wins": 54,
            "losses": 1,
            "truncations": 0,
            "retained_samples_by_verb": {"wait": 40, "fire": 60 + index},
            "episodes": [{"steps": 10}, {"steps": 20}],
        }
        for index in range(3)
    ]
    summary = trainer._collection_summary(collections)
    assert summary["teacher_passes_completed"] == 3
    assert summary["wins"] == 162
    assert summary["losses"] == 3
    assert summary["retained_samples"] == 303
    assert summary["teacher_steps"] == 90
    assert summary["retained_samples_by_verb"] == {"wait": 120, "fire": 183}


def test_builder_freezes_three_nonoverlapping_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    master_path = tmp_path / "master.json"
    teacher_path = tmp_path / "teacher.json"
    master_path.write_text(
        json.dumps(
            {
                "schema": "zuma-rl.alphazuma-55-weekend-master-preregistration",
                "version": 1,
                "campaign_id": "campaign",
                "seed_registry": {
                    "training": {"first": 1_500_000_000, "last": 1_549_999_999},
                    "training_validation": {
                        "first": 1_550_000_000,
                        "last": 1_599_999_999,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    teacher_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(builder.base, "_validate_probe", lambda _: None)
    preregistration = builder.build(
        master_path=master_path,
        teacher_result_path=teacher_path,
        run_dir=tmp_path / "run",
        device="cuda:0",
        training_seed_base=1_542_002_000,
        model_seed=1_542_002_500,
        validation_seed_base=1_550_001_400,
    )
    run = preregistration["run"]
    assert run["learned_features_dim"] == 1024
    assert run["teacher_passes"] == 3
    assert run["training_seed_last_consumed"] == 1_542_002_164
    assert run["epochs_per_round"] == 60
    assert preregistration["authority_boundary"][
        "s99081535_successor_candidate_authority"
    ] is False
