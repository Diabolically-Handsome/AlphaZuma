from __future__ import annotations

from pathlib import Path

import pytest

from tools import build_alphazuma_55_motor_observable_replay_v2 as builder
from tools import distill_alphazuma_55_motor_observable_replay_v2 as trainer
from zuma_rl.alphazuma_55 import INCLUDED_LEVELS
from zuma_rl.motor_observable_migration import (
    transplant_appended_global_policy,
)


def test_builder_freezes_fresh_nonformal_route(tmp_path: Path) -> None:
    output = tmp_path / "preregistration.json"
    value = builder.build(
        project_root=builder.SCRIPT_PATH.parents[1],
        output=output,
    )
    run = value["run"]
    assert run["anchor_teacher_passes"] == 3
    assert run["teacher_execution_probabilities"] == [0.5, 0.1, 0.0]
    assert run["epochs_per_round"] == 4
    assert run["checkpoint_interval_epochs"] == 1
    assert run["anchor_seed_base"] == 1_543_001_000
    assert run["student_seed_last_consumed"] == 1_543_001_329
    assert value["frozen_engineering_validation"]["full55_gate"] == {
        "base_seed": 1_550_002_500,
        "last_seed": 1_550_002_554,
        "levels": 55,
        "maximum_candidates": 5,
        "minimum_wins": 35,
        "minimum_cleared_levels": 35,
    }
    assert value["authority_boundary"][
        "formal_selection_seed_consumption"
    ] is False
    validated = trainer._validate_preregistration(output)
    assert validated["campaign_id"] == builder.CAMPAIGN_ID


def test_real_source_transplants_with_equivalent_initial_logits(
    tmp_path: Path,
) -> None:
    from sb3_contrib import MaskablePPO

    original_root = Path(
        "/mnt/d/SteamLibrary/steamapps/common/Zuma's Revenge"
    )
    if not original_root.exists():
        pytest.skip("retail extraction is unavailable")
    output = tmp_path / "preregistration.json"
    value = builder.build(
        project_root=builder.SCRIPT_PATH.parents[1],
        output=output,
    )
    config = dict(value["run"])
    config["device"] = "cpu"
    prototype = trainer._prototype(
        original_root=original_root,
        max_ticks=int(config["max_ticks"]),
    )
    try:
        source = MaskablePPO.load(
            Path(value["source"]["model_path"]), device="cpu"
        )
        target = trainer._build_motor_model(prototype, config)
        receipt = transplant_appended_global_policy(
            source_policy=source.policy,
            target_policy=target.policy,
            source_global_feature_size=int(
                prototype.revenge_env.global_feature_size
            ),
            target_global_feature_size=int(prototype.global_feature_size),
        )
        equivalence = trainer._verify_initial_equivalence(
            source=source,
            target=target,
            prototype=prototype,
        )
        assert receipt["status"] == "PASS"
        assert equivalence["status"] == "PASS"
        assert equivalence["deterministic_component_argmax_exact"] is True
        assert all(
            row["within_absolute_tolerance"]
            for row in equivalence["components"]
        )
    finally:
        prototype.close()


def test_real_one_level_teacher_collection_keeps_motor_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_root = Path(
        "/mnt/d/SteamLibrary/steamapps/common/Zuma's Revenge"
    )
    if not original_root.exists():
        pytest.skip("retail extraction is unavailable")
    monkeypatch.setattr(
        trainer.legacy,
        "_make_env_factory",
        trainer._make_motor_env_factory,
    )
    rows, collection = trainer.intent._collect_intent_round(
        original_root=original_root,
        level_ids=(INCLUDED_LEVELS[0],),
        round_index=0,
        seed_base=1_400_000_000,
        parallel_envs=1,
        max_ticks=30_000,
        capacities={"wait": 8, "fire": 8, "swap": 8, "hop": 8},
        strides={"wait": 16, "fire": 8, "swap": 8, "hop": 8},
        model_seed=1_400_000_001,
    )
    assert collection["wins"] == 1
    assert collection["retained_samples"] == len(rows)
    assert rows
    assert rows[0][0].shape == (22_904,)
