from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools import build_alphazuma_55_polar_intent_wide as builder
from tools import distill_alphazuma_55_polar_intent_wide_v1 as trainer


def test_intent_label_relaxes_only_the_teacher_target_verb() -> None:
    exact_mask = np.ones(184, dtype=np.bool_)
    exact_mask[1] = False
    exact_mask[3] = False
    raw, effective, training_mask, forced = trainer._intent_label(
        np.asarray([1, 42], dtype=np.int64), exact_mask
    )
    np.testing.assert_array_equal(raw, np.asarray([1, 42]))
    np.testing.assert_array_equal(effective, np.asarray([0, 42]))
    assert forced is True
    assert bool(training_mask[1]) is True
    assert bool(training_mask[3]) is False
    np.testing.assert_array_equal(training_mask[4:], exact_mask[4:])
    assert bool(exact_mask[1]) is False


def test_intent_label_rejects_a_masked_aim() -> None:
    exact_mask = np.ones(184, dtype=np.bool_)
    exact_mask[4 + 42] = False
    with pytest.raises(RuntimeError, match="masked aim bin"):
        trainer._intent_label(np.asarray([1, 42]), exact_mask)


def test_builder_freezes_intent_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    master = tmp_path / "master.json"
    teacher = tmp_path / "teacher.json"
    master.write_text(
        json.dumps(
            {
                "schema": "zuma-rl.alphazuma-55-weekend-master-preregistration",
                "version": 1,
                "campaign_id": "campaign",
                "seed_registry": {
                    "training": {
                        "first": 1_500_000_000,
                        "last": 1_549_999_999,
                    },
                    "training_validation": {
                        "first": 1_550_000_000,
                        "last": 1_599_999_999,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    teacher.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(builder.effective_builder.base, "_validate_probe", lambda _: None)
    prereg = builder.build(
        master_path=master,
        teacher_result_path=teacher,
        run_dir=tmp_path / "run",
        device="cuda:0",
        training_seed_base=1_542_003_000,
        model_seed=1_542_003_500,
        validation_seed_base=1_550_001_600,
    )
    run = prereg["run"]
    assert run["policy_architecture"] == "entity_polar_intent_wide"
    assert run["raw_teacher_intent_labels"] is True
    assert run["training_mask_relaxation"] == (
        "unmask_only_the_teacher_target_verb"
    )
    assert run["exact_action_masks_for_environment_execution"] is True
    assert run["exact_action_masks_for_online_policy_inference"] is True
    assert run["epochs_per_round"] == 30
    assert prereg["frozen_engineering_validation"]["checkpoint_epochs"] == [
        10,
        20,
        30,
    ]
    assert prereg["authority_boundary"][
        "s99081535_successor_candidate_authority"
    ] is False
