from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from tools import build_alphazuma_55_motor_observable_replay_v3 as builder
from tools import distill_alphazuma_55_motor_observable_replay_v3 as trainer
from zuma_rl.alphazuma_55 import INCLUDED_LEVELS


def _collection(
    level_ids: tuple[str, ...], seed_base: int, *, lose: str | None
) -> tuple[list[Any], dict[str, Any]]:
    rows = [
        (
            np.asarray([index], dtype=np.float32),
            np.asarray([1, index % 8], dtype=np.int64),
            np.ones(12, dtype=np.bool_),
        )
        for index, _ in enumerate(level_ids)
    ]
    episodes = [
        {
            "level_id": level_id,
            "seed": seed_base + index,
            "steps": 10 + index,
            "outcome": "loss" if level_id == lose else "win",
            "time_limit_truncated": False,
            "observation_capacity_overflow": False,
        }
        for index, level_id in enumerate(level_ids)
    ]
    return rows, {
        "round_index": 0,
        "wall_seconds": 1.0,
        "episodes": episodes,
        "wins": sum(row["outcome"] == "win" for row in episodes),
        "losses": sum(row["outcome"] == "loss" for row in episodes),
        "truncations": 0,
        "capacity_overflows": 0,
        "retained_samples": len(rows),
        "retained_samples_by_verb": {"fire": len(rows)},
        "raw_intent_action_counts": {"fire": len(rows)},
        "effective_execution_action_counts": {"fire": len(rows)},
        "intent_mask_relaxations": 0,
        "teacher_policy_id": "teacher",
    }


def test_builder_freezes_retry_ledger_and_fresh_training_seeds(
    tmp_path: Path,
) -> None:
    output = tmp_path / "preregistration.json"
    value = builder.build(
        project_root=builder.SCRIPT_PATH.parents[1], output=output
    )
    run = value["run"]
    assert run["anchor_seed_base"] == 1_543_002_000
    assert run["anchor_seed_last_reserved"] == 1_543_002_494
    assert run["student_seed_base"] == 1_543_002_495
    assert run["student_seed_last_consumed"] == 1_543_002_659
    assert run["anchor_retry_policy"]["attempts_per_level"] == 3
    assert run["anchor_retry_policy"]["sample_retention"] == "all_attempts"
    assert value["post_hoc_disclosure"]["formal_gate_unchanged"] is True
    assert value["engineering_seed_reuse_evidence"][
        "matrix_artifacts_present"
    ] is False
    validated = trainer._validate_preregistration(output)
    assert validated["campaign_id"] == builder.CAMPAIGN_ID


def test_anchor_seed_formula_is_disjoint_and_covers_reserved_interval() -> None:
    config = {"anchor_seed_base": builder.ANCHOR_SEED_BASE}
    seeds = [
        trainer._anchor_seed(
            config,
            pass_index=pass_index,
            level_index=level_index,
            attempt_index=attempt_index,
        )
        for pass_index in range(3)
        for attempt_index in range(3)
        for level_index in range(len(INCLUDED_LEVELS))
    ]
    assert len(seeds) == len(set(seeds)) == 495
    assert min(seeds) == builder.ANCHOR_SEED_BASE
    assert max(seeds) == builder.ANCHOR_SEED_LAST


def test_anchor_collector_retries_only_failure_and_retains_both_attempts(
    tmp_path: Path,
) -> None:
    failed_level = INCLUDED_LEVELS[7]
    calls: list[dict[str, Any]] = []

    def fake_collector(**kwargs: Any) -> tuple[list[Any], dict[str, Any]]:
        calls.append(kwargs)
        level_ids = tuple(kwargs["level_ids"])
        lose = failed_level if len(level_ids) == len(INCLUDED_LEVELS) else None
        return _collection(level_ids, int(kwargs["seed_base"]), lose=lose)

    config = {
        "anchor_seed_base": builder.ANCHOR_SEED_BASE,
        "anchor_seed_last_reserved": builder.ANCHOR_SEED_LAST,
        "anchor_retry_policy": {
            "attempts_per_level": 3,
            "first_attempt_mode": "paired_full55_batch",
            "retry_mode": "failed_levels_only_fixed_seed",
            "sample_retention": "all_attempts",
            "pass_requirement": "at_least_one_win_for_each_of_55_levels",
        },
    }
    rows, result = trainer._collect_teacher_anchor_with_retries(
        base_collector=fake_collector,
        config=config,
        run_dir=tmp_path,
        original_root=tmp_path,
        level_ids=INCLUDED_LEVELS,
        round_index=0,
        seed_base=builder.ANCHOR_SEED_BASE,
        parallel_envs=24,
        max_ticks=30_000,
        capacities={"wait": 1, "fire": 1, "swap": 1, "hop": 1},
        strides={"wait": 1, "fire": 1, "swap": 1, "hop": 1},
        model_seed=123,
    )
    assert len(calls) == 2
    assert tuple(calls[1]["level_ids"]) == (failed_level,)
    assert calls[1]["seed_base"] == builder.ANCHOR_SEED_BASE + 55 + 7
    assert result["wins"] == 55
    assert result["level_coverage_wins"] == 55
    assert result["attempts_consumed"] == 56
    assert result["losses"] == 1
    assert result["missing_level_ids"] == []
    assert len(rows) == 56
    assert len(result["consumed_seeds"]) == len(
        set(result["consumed_seeds"])
    )
    assert len(result["attempt_receipts"]) == 2
    assert (tmp_path / "anchor_attempts/pass-00-summary.json").exists()
