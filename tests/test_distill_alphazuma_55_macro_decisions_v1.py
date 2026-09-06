"""Tests for the macro-decision BC trainer.

Covers, on small synthetic CPU data built through the COLLECTOR's real
recorder/manifest path:
(a) dataset loading with sha256 receipt verification, the per-level
    ``--levels`` filter, honest teacher-loss flags, and refusals
    (foreign manifest, tampered bytes, unknown level, mask-illegal
    labels); observation backing: the RAM-budget projection/refusal
    contract and the receipted memmap cache (extraction, reuse,
    per-selection identity, tamper refusals, bit-identical data on
    both paths, training on the memmap view);
(b) training: loss computes and decreases, per-epoch HOLDOUT metrics
    (verb accuracy, aim exact/within-3 overall and on fire decisions,
    fire recall/precision), episode-level split, checkpoints, receipts,
    and the frozen-run-dir refusal;
(c) round trip: the saved final_model.zip loads through plain
    ``MaskablePPO.load`` (extractor resolved from
    tools.distill_alphazuma_55_park_settle_v3), passes
    ``probe_alphazuma_55_macro_native_eval_v1.require_macro_native_
    model``, and runs that probe's stubbed-env path unchanged;
(d) verb class weights (the park-settle v1 reweighting hook): the
    weighted verb CE differs from unweighted on a synthetic imbalanced
    batch (and all-ones weights match unweighted exactly), the weights
    flow into the config.json/completion.json receipts, the validation
    contract refuses malformed weights, and the flag-free
    ``fire_rate_calibration`` block (student vs teacher deterministic
    fire fraction) is always present in the holdout/calibration
    metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tools import collect_alphazuma_55_macro_decisions_v1 as collector
from tools import distill_alphazuma_55 as legacy
from tools import distill_alphazuma_55_macro_decisions_v1 as trainer
from tools import distill_alphazuma_55_park_settle_v1 as v1
from tools import distill_alphazuma_55_park_settle_v3 as trainer_v3
from tools import probe_alphazuma_55_macro_native_eval_v1 as native_probe
import zuma_rl.park_settle_action_wrapper as wrapper_module
from zuma_rl.observation_stack_wrapper import stacked_box


STACK_LAGS = (0, 4, 8)
AIM_LABEL = 37
FIRE_EVERY = 3


def _toy_spaces() -> tuple[Any, dict[str, Any]]:
    observation_space, _, policy_kwargs = v1.tiny_entity_polar_interface(
        features_dim=16
    )
    return observation_space, policy_kwargs


def _toy_stacked_dim() -> int:
    observation_space, _ = _toy_spaces()
    return int(observation_space.shape[0]) * len(STACK_LAGS)


def _macro_info(verb: int, aim: int) -> dict[str, Any]:
    released = verb == wrapper_module.MACRO_FIRE
    return {
        "profile": "park-settle-v1",
        "macro_verb": wrapper_module.MACRO_VERB_NAMES[verb],
        "target_aim_bin": aim,
        "ticks_consumed": 20,
        "settle_ticks": 3,
        "settled": True,
        "button_executed": released,
        "released": released,
        "release_aim_bin": aim if released else None,
        "decision_point": True,
        "terminal": False,
    }


def _write_decision_episode(
    run_dir: Path,
    task: collector.EpisodeTask,
    *,
    decisions: int,
    obs_dim: int,
    outcome: str,
) -> dict[str, Any]:
    """Learnable teacher decisions through the collector's real recorder.

    ``obs[0]`` is +1 on fire decisions and -1 otherwise, and the aim
    label is the constant ``AIM_LABEL``, so a couple of tiny-model
    epochs measurably reduce both losses.
    """

    episodes_dir = run_dir / collector.EPISODES_DIRNAME
    spill_dir = run_dir / collector.SPILL_DIRNAME
    episodes_dir.mkdir(parents=True, exist_ok=True)
    spill_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(task.seed))
    recorder = collector._DecisionRecorder(
        spill_path=spill_dir / f"{task.file_name}.spill",
        observation_width=obs_dim,
    )
    for index in range(decisions):
        fire = index % FIRE_EVERY == 0
        verb = (
            wrapper_module.MACRO_FIRE
            if fire
            else wrapper_module.MACRO_WAIT_HOLD
        )
        observation = rng.uniform(-1.0, 1.0, obs_dim).astype(np.float32)
        observation[0] = np.float32(1.0 if fire else -1.0)
        recorder.record_decision(
            observation=observation,
            macro_action=np.asarray([verb, AIM_LABEL], dtype=np.int64),
            macro_mask=np.ones(collector.MACRO_MASK_WIDTH, dtype=np.bool_),
            teacher_per_tick_action=np.asarray(
                [verb, AIM_LABEL], dtype=np.int64
            ),
            fallback=None,
        )
        recorder.record_result(
            _macro_info(verb, AIM_LABEL), {"score": index * 5}
        )
    return collector._finalize_episode(
        task=task,
        recorder=recorder,
        final_info={"outcome": outcome, "score": decisions * 5},
        guard_exhausted=False,
        episodes_dir=episodes_dir,
        wall_seconds=0.1,
    )


def _write_collection(
    tmp_path: Path,
    *,
    levels: tuple[str, ...] = ("Jungle1", "Jungle2"),
    seeds_per_level: int = 2,
    decisions: int = 48,
) -> Path:
    run_dir = tmp_path / "collection"
    obs_dim = _toy_stacked_dim()
    tasks = collector.plan_tasks(
        levels=levels,
        seeds_per_level=seeds_per_level,
        seed_base=collector.DEFAULT_SEED_BASE,
    )
    entries: dict[tuple[str, int], dict[str, Any]] = {}
    for ordinal, task in enumerate(tasks):
        # The LAST planned episode is an honest teacher loss.
        outcome = "loss" if ordinal == len(tasks) - 1 else "win"
        entry = _write_decision_episode(
            run_dir,
            task,
            decisions=decisions,
            obs_dim=obs_dim,
            outcome=outcome,
        )
        entries[task.key] = entry
    collector._write_manifest(
        run_dir,
        levels=collector._validate_levels(levels),
        seeds_per_level=seeds_per_level,
        seed_base=collector.DEFAULT_SEED_BASE,
        max_ticks=200,
        hold_ticks=8,
        stack_lags=STACK_LAGS,
        entries_by_key=entries,
    )
    return run_dir


def _train_config(**overrides: Any) -> trainer.MacroDecisionTrainConfig:
    values: dict[str, Any] = dict(
        epochs=2,
        batch_size=64,
        learning_rate=1.0e-3,
        holdout_fraction=0.5,
        shuffle_seed=99_081_693,
        device="cpu",
        checkpoint_interval_epochs=1,
    )
    values.update(overrides)
    return trainer.MacroDecisionTrainConfig(**values)


def _toy_stacked_model(
    train_config: trainer.MacroDecisionTrainConfig,
) -> Any:
    observation_space, policy_kwargs = _toy_spaces()
    return trainer.build_macro_student_model(
        observation_space=stacked_box(observation_space, len(STACK_LAGS)),
        policy_kwargs=trainer_v3.stack_policy_kwargs(
            policy_kwargs, frames=len(STACK_LAGS)
        ),
        model_config=trainer.MacroDecisionModelConfig(
            learned_features_dim=16,
            aim_head_identity_scale=5.0,
            model_seed=99_081_690,
        ),
        train_config=train_config,
    )


# ---------------------------------------------------------------------------
# (a) Dataset loading: receipts, level filter, honest flags, refusals.
# ---------------------------------------------------------------------------


def test_load_verifies_receipts_filters_levels_and_flags_losses(
    tmp_path: Path,
) -> None:
    run_dir = _write_collection(tmp_path)
    manifest_path = run_dir / "episodes_manifest.json"
    dataset, receipt = trainer.load_decision_dataset(manifest_path)
    assert dataset.sample_count == 4 * 48
    assert receipt["episodes"] == 4
    assert receipt["stack_lags"] == list(STACK_LAGS)
    assert receipt["hold_ticks"] == 8
    assert receipt["sha256_verified"] is True
    # The teacher loss is flagged, its decisions loaded all the same.
    assert receipt["teacher_wins"] == 3
    assert receipt["teacher_losses_flagged"] == 1
    assert receipt["fire_decisions"] == 4 * 16
    assert dataset.observations.shape == (192, _toy_stacked_dim())
    assert dataset.masks.shape == (192, 183)
    assert len(np.unique(dataset.episode_indices)) == 4
    # Tick indices are per-episode native tick offsets.
    assert int(dataset.tick_indices[0]) == 0

    filtered, filtered_receipt = trainer.load_decision_dataset(
        manifest_path, levels=["Jungle2"]
    )
    assert filtered.sample_count == 2 * 48
    assert filtered_receipt["levels"] == ["Jungle2"]
    assert filtered_receipt["level_filter"] == ["Jungle2"]
    assert all(
        entry["level_id"] == "Jungle2"
        for entry in filtered_receipt["entries"]
    )

    with pytest.raises(ValueError, match="not in this collection"):
        trainer.load_decision_dataset(manifest_path, levels=["volcano10"])
    with pytest.raises(ValueError, match="selects nothing"):
        trainer.load_decision_dataset(manifest_path, levels=[])

    foreign = tmp_path / "foreign.json"
    foreign.write_text(
        json.dumps({"schema": "something-else", "episodes": [{}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="macro-decisions manifest"):
        trainer.load_decision_dataset(foreign)

    victim = Path(filtered_receipt["entries"][0]["path"])
    victim.write_bytes(victim.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="differ from the manifest receipt"):
        trainer.load_decision_dataset(manifest_path)


def test_dataset_validation_rejects_mask_illegal_labels() -> None:
    count = 2
    masks = np.ones((count, 183), dtype=np.bool_)
    masks[0, wrapper_module.MACRO_FIRE] = False
    dataset = trainer.MacroDecisionDataset(
        observations=np.zeros((count, 12), dtype=np.float32),
        actions=np.asarray(
            [[wrapper_module.MACRO_FIRE, 10], [0, 5]], dtype=np.int64
        ),
        masks=masks,
        episode_indices=np.zeros(count, dtype=np.int64),
        tick_indices=np.arange(count, dtype=np.int64),
    )
    with pytest.raises(ValueError, match="illegal under"):
        dataset.validate()


def test_split_holdout_is_by_episode() -> None:
    rows = 40
    dataset = trainer.MacroDecisionDataset(
        observations=np.zeros((rows, 4), dtype=np.float32),
        actions=np.zeros((rows, 2), dtype=np.int64),
        masks=np.ones((rows, 183), dtype=np.bool_),
        episode_indices=np.repeat(np.arange(4, dtype=np.int64), 10),
        tick_indices=np.tile(np.arange(10, dtype=np.int64), 4),
    )
    train, holdout, split = trainer.split_holdout_by_episode(
        dataset, holdout_fraction=0.25, seed=7
    )
    assert split["mode"] == "episode"
    assert split["holdout_units"] == 1
    assert holdout.size == 10
    assert train.size == 30
    # No episode straddles the split.
    train_units = set(dataset.episode_indices[train].tolist())
    holdout_units = set(dataset.episode_indices[holdout].tolist())
    assert not train_units & holdout_units


# ---------------------------------------------------------------------------
# Observation backing: RAM-budget contract and the memmap cache path.
# ---------------------------------------------------------------------------


def test_memmap_backing_matches_ram_reuses_cache_and_trains(
    tmp_path: Path,
) -> None:
    run_dir = _write_collection(tmp_path)
    manifest_path = run_dir / "episodes_manifest.json"
    ram_dataset, ram_receipt = trainer.load_decision_dataset(manifest_path)
    ram_backing = ram_receipt["observation_backing"]
    assert ram_backing["mode"] == "ram"
    assert ram_backing["requested"] == "auto"
    # The projection comes from the manifest's own receipts and equals
    # the bytes the RAM path actually materialized.
    assert (
        ram_backing["projected_observation_bytes"]
        == ram_dataset.observations.nbytes
    )

    dataset, receipt = trainer.load_decision_dataset(
        manifest_path, observation_backing="memmap"
    )
    backing = receipt["observation_backing"]
    assert backing["mode"] == "memmap"
    assert backing["storage"] == "raw_npy_memmap_read_only"
    assert backing["extraction"]["action"] == "extracted"
    npy_path = Path(backing["observations_npy"]["path"]).resolve()
    assert npy_path.is_file()
    assert npy_path.parent == (
        run_dir / trainer.OBSERVATION_CACHE_DIRNAME
    ).resolve()
    # Observations are a READ-ONLY memmap view, bit-identical to the
    # RAM path; the small arrays are bit-identical too.
    assert isinstance(dataset.observations, np.memmap) or isinstance(
        dataset.observations.base, np.memmap
    )
    assert not dataset.observations.flags.writeable
    assert np.array_equal(
        np.asarray(dataset.observations), ram_dataset.observations
    )
    assert np.array_equal(dataset.actions, ram_dataset.actions)
    assert np.array_equal(dataset.masks, ram_dataset.masks)
    assert np.array_equal(
        dataset.episode_indices, ram_dataset.episode_indices
    )
    assert np.array_equal(dataset.tick_indices, ram_dataset.tick_indices)

    # The training loop runs UNCHANGED on the memmap view, and the
    # backing lands in completion.json through the dataset receipt.
    train_config = _train_config(epochs=1, checkpoint_interval_epochs=0)
    model = _toy_stacked_model(train_config)
    completion = trainer.execute_macro_training(
        model=model,
        dataset=dataset,
        run_dir=tmp_path / "memmap-train",
        model_config=trainer.MacroDecisionModelConfig(
            learned_features_dim=16, model_seed=99_081_690
        ),
        train_config=train_config,
        dataset_receipt=receipt,
        stack_lags=STACK_LAGS,
    )
    assert completion["status"] == "COMPLETE"
    assert completion["dataset"]["observation_backing"]["mode"] == "memmap"

    # A second load REUSES the receipted cache (same selection digest).
    again, again_receipt = trainer.load_decision_dataset(
        manifest_path, observation_backing="memmap"
    )
    assert (
        again_receipt["observation_backing"]["extraction"]["action"]
        == "reused"
    )
    assert np.array_equal(
        np.asarray(again.observations), ram_dataset.observations
    )

    # A different --levels selection is a DIFFERENT cache identity.
    filtered, filtered_receipt = trainer.load_decision_dataset(
        manifest_path, levels=["Jungle2"], observation_backing="memmap"
    )
    filtered_backing = filtered_receipt["observation_backing"]
    assert filtered_backing["extraction"]["action"] == "extracted"
    assert (
        Path(filtered_backing["observations_npy"]["path"]).resolve()
        != npy_path
    )
    assert filtered.sample_count == 2 * 48

    # Honest cache refusals: tampered bytes, then an unreceipted .npy.
    receipt_path = Path(backing["extraction"]["receipt_path"])
    original = npy_path.read_bytes()
    npy_path.write_bytes(original[:-1] + bytes([original[-1] ^ 0xFF]))
    with pytest.raises(ValueError, match="sha256 changed"):
        trainer.load_decision_dataset(
            manifest_path, observation_backing="memmap"
        )
    npy_path.write_bytes(original)
    receipt_path.unlink()
    with pytest.raises(ValueError, match="unreceipted"):
        trainer.load_decision_dataset(
            manifest_path, observation_backing="memmap"
        )


def test_backing_budget_auto_switch_and_ram_refusal(tmp_path: Path) -> None:
    run_dir = _write_collection(tmp_path)
    manifest_path = run_dir / "episodes_manifest.json"
    # A tiny budget flips auto to memmap -- never a refusal, never OOM.
    dataset, receipt = trainer.load_decision_dataset(
        manifest_path, ram_budget_bytes=1
    )
    assert receipt["observation_backing"]["mode"] == "memmap"
    assert receipt["observation_backing"]["requested"] == "auto"
    assert dataset.sample_count == 4 * 48
    # Explicit RAM above the budget is a CONTRACT refusal pointing at
    # --levels sharding and memmap backing (not an OOM kill mid-run).
    with pytest.raises(ValueError, match=r"--levels"):
        trainer.load_decision_dataset(
            manifest_path, observation_backing="ram", ram_budget_bytes=1
        )
    # Unknown modes and non-positive budgets are refused.
    with pytest.raises(ValueError, match="observation_backing"):
        trainer.load_decision_dataset(
            manifest_path, observation_backing="mystery"
        )
    with pytest.raises(ValueError, match="ram_budget_bytes"):
        trainer.load_decision_dataset(manifest_path, ram_budget_bytes=0)


# ---------------------------------------------------------------------------
# Model container: distill-v3 extractor enforced, macro interface.
# ---------------------------------------------------------------------------


def test_model_container_enforces_the_distill_v3_extractor() -> None:
    observation_space, policy_kwargs = _toy_spaces()
    # An unstacked (base-class) extractor cannot produce a zip the
    # macro-native eval probe can resolve: refused by name.
    with pytest.raises(ValueError, match="pickle-reference"):
        trainer.build_macro_student_model(
            observation_space=observation_space,
            policy_kwargs=policy_kwargs,
            model_config=trainer.MacroDecisionModelConfig(
                learned_features_dim=16, model_seed=1
            ),
            train_config=_train_config(),
        )
    model = _toy_stacked_model(_train_config())
    assert tuple(int(v) for v in model.action_space.nvec) == (3, 180)
    assert int(model.observation_space.shape[0]) == _toy_stacked_dim()
    extractor = model.policy.features_extractor
    assert type(extractor) is trainer_v3.StackedRevengeEntityFeatureExtractor
    assert extractor.stacked_frames == len(STACK_LAGS)
    # The polar identity anchor landed on the 180 aim rows.
    import torch

    action_net = model.policy.action_net
    learned = int(extractor.learned_features_dim)
    with torch.no_grad():
        aim_rows = action_net.weight[3:, learned:]
        assert torch.equal(
            aim_rows, 5.0 * torch.eye(180, dtype=aim_rows.dtype)
        )


# ---------------------------------------------------------------------------
# (b) Training: losses, holdout metrics, checkpoints, receipts.
# ---------------------------------------------------------------------------


def test_training_reduces_loss_and_receipts_holdout_metrics(
    tmp_path: Path,
) -> None:
    run_dir = _write_collection(tmp_path)
    dataset, receipt = trainer.load_decision_dataset(
        run_dir / "episodes_manifest.json"
    )
    train_config = _train_config()
    model = _toy_stacked_model(train_config)
    out_dir = tmp_path / "train-run"
    completion = trainer.execute_macro_training(
        model=model,
        dataset=dataset,
        run_dir=out_dir,
        model_config=trainer.MacroDecisionModelConfig(
            learned_features_dim=16, model_seed=99_081_690
        ),
        train_config=train_config,
        dataset_receipt=receipt,
        stack_lags=STACK_LAGS,
    )
    assert completion["status"] == "COMPLETE"
    assert completion["schema"] == (
        "zuma-rl.alphazuma-55-macro-decisions-distillation-completion"
    )
    assert completion["formal_seed_consumption"] is False
    assert completion["model_spaces"]["action_space_nvec"] == [3, 180]
    assert completion["stack_lags"] == list(STACK_LAGS)
    assert completion["split"]["mode"] == "episode"
    assert completion["split"]["holdout_units"] == 2
    epochs = completion["optimization"]["epochs"]
    assert len(epochs) == 2
    for row in epochs:
        holdout = row["holdout"]
        assert holdout["slice"] == "holdout"
        assert holdout["sample_count"] == 96
        for key in (
            "verb_accuracy",
            "aim_exact_accuracy",
            "aim_within_three_accuracy",
            "fire_recall",
            "fire_precision",
            "teacher_fire_decisions",
            "student_fire_predictions",
        ):
            assert key in holdout
        assert holdout["deterministic_verbs_all_mask_legal"] is True
        fire_group = holdout["fire_decisions"]
        assert fire_group["sample_count"] == 32
        assert "aim_exact_accuracy" in fire_group
        assert "aim_within_three_accuracy" in fire_group
    # Loss decreases (or at the very least computes finitely).
    assert np.isfinite(epochs[0]["mean_loss"])
    assert epochs[-1]["mean_loss"] < epochs[0]["mean_loss"]
    # The categorical-only recipe is receipted (no smoothing exists).
    recipe = completion["recipe"]
    assert recipe["aim_loss_mode"] == "categorical"
    assert recipe["aim_loss_smoothing"].startswith("none")
    assert recipe["loss_weights"] == {"verb": 1.0, "aim": 1.0}
    assert recipe["value_head_init"].startswith("fresh")
    # Checkpoints at every epoch (interval 1), sha-receipted.
    checkpoints = completion["optimization"]["checkpoints"]
    assert [entry["epoch"] for entry in checkpoints] == [1, 2]
    for entry in checkpoints:
        path = Path(entry["path"])
        assert path.is_file()
        assert entry["sha256"] == legacy._sha256(path)
    final = completion["final_model"]
    final_path = Path(final["path"])
    assert final_path.name == "final_model.zip"
    assert final["sha256"] == legacy._sha256(final_path)
    round_trip = completion["round_trip"]
    assert round_trip["reloaded_action_nvec"] == [3, 180]
    assert round_trip["bitwise_state_dict_equal"] is True
    assert round_trip["loadable_by_macro_native_eval_v1"] is True
    # Receipts on disk and JSON-safe.
    assert (out_dir / "config.json").is_file()
    assert (out_dir / "completion.json").is_file()
    json.dumps(v1._jsonable(completion), allow_nan=False)
    # Frozen-output discipline: a used run dir is refused.
    with pytest.raises(FileExistsError):
        trainer.execute_macro_training(
            model=model,
            dataset=dataset,
            run_dir=out_dir,
            model_config=trainer.MacroDecisionModelConfig(
                learned_features_dim=16, model_seed=99_081_690
            ),
            train_config=train_config,
            dataset_receipt=receipt,
        )


# ---------------------------------------------------------------------------
# (d) Verb class weights and the flag-free fire-rate calibration.
# ---------------------------------------------------------------------------


def _synthetic_imbalanced_dataset(
    rows: int = 64, fire_rows: int = 6
) -> trainer.MacroDecisionDataset:
    """One deterministic batch with the real fire imbalance shape."""

    obs_dim = _toy_stacked_dim()
    rng = np.random.default_rng(20_260_818)
    observations = rng.uniform(-1.0, 1.0, (rows, obs_dim)).astype(
        np.float32
    )
    observations[:, 0] = np.float32(-1.0)
    observations[:fire_rows, 0] = np.float32(1.0)
    actions = np.zeros((rows, 2), dtype=np.int64)
    actions[:, 0] = wrapper_module.MACRO_WAIT_HOLD
    actions[:fire_rows, 0] = wrapper_module.MACRO_FIRE
    actions[:, 1] = AIM_LABEL
    dataset = trainer.MacroDecisionDataset(
        observations=observations,
        actions=actions,
        masks=np.ones((rows, 183), dtype=np.bool_),
        episode_indices=np.zeros(rows, dtype=np.int64),
        tick_indices=np.arange(rows, dtype=np.int64),
    )
    dataset.validate()
    return dataset


def _one_batch_metrics(
    dataset: trainer.MacroDecisionDataset,
    loss_config: trainer.MacroDecisionLossConfig,
) -> dict[str, Any]:
    """One _train_macro_batch step on a FRESH identically-seeded model."""

    model = _toy_stacked_model(_train_config())
    return trainer._train_macro_batch(
        model=model,
        dataset=dataset,
        rows=np.arange(dataset.sample_count, dtype=np.int64),
        loss=loss_config,
    )


def test_verb_class_weights_validation_contract() -> None:
    # Valid weights pass (and default None stays the current behavior).
    trainer.MacroDecisionLossConfig().validate()
    trainer.MacroDecisionLossConfig(
        verb_class_weights=(1.0, 8.0, 1.0)
    ).validate()
    with pytest.raises(ValueError, match="every macro verb"):
        trainer.MacroDecisionLossConfig(
            verb_class_weights=(1.0, 8.0)
        ).validate()
    with pytest.raises(
        ValueError, match="verb_class_weights must be non-negative"
    ):
        trainer.MacroDecisionLossConfig(
            verb_class_weights=(1.0, -1.0, 1.0)
        ).validate()
    with pytest.raises(
        ValueError, match="verb class weight must be positive"
    ):
        trainer.MacroDecisionLossConfig(
            verb_class_weights=(0.0, 0.0, 0.0)
        ).validate()


def test_weighted_verb_ce_changes_loss_on_imbalanced_batch() -> None:
    dataset = _synthetic_imbalanced_dataset(rows=64, fire_rows=6)
    unweighted = _one_batch_metrics(
        dataset, trainer.MacroDecisionLossConfig()
    )
    fire_weighted = _one_batch_metrics(
        dataset,
        trainer.MacroDecisionLossConfig(
            verb_class_weights=(1.0, 8.0, 1.0)
        ),
    )
    ones_weighted = _one_batch_metrics(
        dataset,
        trainer.MacroDecisionLossConfig(
            verb_class_weights=(1.0, 1.0, 1.0)
        ),
    )
    assert np.isfinite(unweighted["verb_loss"])
    assert np.isfinite(fire_weighted["verb_loss"])
    # Up-weighting the 8%-ish fire class CHANGES the verb CE (torch's
    # weight= is a weighted mean over the batch's per-row NLLs).
    assert fire_weighted["verb_loss"] != pytest.approx(
        unweighted["verb_loss"]
    )
    # All-ones weights are exactly the unweighted objective.
    assert ones_weighted["verb_loss"] == pytest.approx(
        unweighted["verb_loss"]
    )
    # The aim pathway is untouched by verb class weights.
    assert fire_weighted["aim_loss"] == pytest.approx(
        unweighted["aim_loss"]
    )


def test_calibration_always_reports_deterministic_fire_rate() -> None:
    dataset = _synthetic_imbalanced_dataset(rows=60, fire_rows=20)
    model = _toy_stacked_model(_train_config())
    metrics = trainer.evaluate_macro_decisions(
        model=model,
        dataset=dataset,
        indices=np.arange(dataset.sample_count, dtype=np.int64),
        batch_size=32,
    )
    calibration = metrics["fire_rate_calibration"]
    assert calibration["policy"] == "deterministic_masked_verb_argmax"
    assert calibration["teacher_fire_fraction"] == pytest.approx(20 / 60)
    assert calibration["student_fire_fraction"] == pytest.approx(
        metrics["student_fire_predictions"] / 60
    )
    ratio = calibration["student_to_teacher_fire_ratio"]
    assert ratio == pytest.approx(
        metrics["student_fire_predictions"]
        / metrics["teacher_fire_decisions"]
    )
    # A fire-free slice reports fraction 0 and an honest None ratio.
    no_fire = _synthetic_imbalanced_dataset(rows=16, fire_rows=0)
    quiet = trainer.evaluate_macro_decisions(
        model=model,
        dataset=no_fire,
        indices=np.arange(no_fire.sample_count, dtype=np.int64),
        batch_size=16,
    )
    quiet_calibration = quiet["fire_rate_calibration"]
    assert quiet_calibration["teacher_fire_fraction"] == 0.0
    assert quiet_calibration["student_to_teacher_fire_ratio"] is None


def test_verb_class_weights_flow_into_receipts_and_calibration(
    tmp_path: Path,
) -> None:
    run_dir = _write_collection(tmp_path, decisions=24)
    dataset, receipt = trainer.load_decision_dataset(
        run_dir / "episodes_manifest.json"
    )
    weights = (1.0, 6.0, 1.0)
    train_config = _train_config(
        epochs=1,
        checkpoint_interval_epochs=0,
        loss=trainer.MacroDecisionLossConfig(verb_class_weights=weights),
    )
    model = _toy_stacked_model(train_config)
    out_dir = tmp_path / "weighted-run"
    completion = trainer.execute_macro_training(
        model=model,
        dataset=dataset,
        run_dir=out_dir,
        model_config=trainer.MacroDecisionModelConfig(
            learned_features_dim=16, model_seed=99_081_690
        ),
        train_config=train_config,
        dataset_receipt=receipt,
        stack_lags=STACK_LAGS,
    )
    assert completion["status"] == "COMPLETE"
    # The weights land in BOTH receipts: config.json and completion.json.
    config_disk = json.loads(
        (out_dir / "config.json").read_text(encoding="utf-8")
    )
    assert config_disk["train_config"]["loss"]["verb_class_weights"] == list(
        weights
    )
    completion_disk = json.loads(
        (out_dir / "completion.json").read_text(encoding="utf-8")
    )
    assert completion_disk["train_config"]["loss"][
        "verb_class_weights"
    ] == list(weights)
    assert completion_disk["recipe"]["verb_class_weights"] == list(weights)
    assert completion_disk["recipe"]["verb_batch_rebalancing"] == "none"
    assert tuple(
        completion["train_config"]["loss"]["verb_class_weights"]
    ) == weights
    # The flag-free calibration block is present in EVERY epoch's
    # holdout metrics and in the completion's calibration section, and
    # its fractions agree with the counted fields on the same slice.
    for row in completion["optimization"]["epochs"]:
        assert "fire_rate_calibration" in row["holdout"]
    calibration = completion["calibration"]["fire_rate_calibration"]
    holdout = completion["final_holdout_metrics"]
    assert calibration["policy"] == "deterministic_masked_verb_argmax"
    assert calibration["teacher_fire_fraction"] == pytest.approx(
        holdout["teacher_fire_decisions"] / holdout["sample_count"]
    )
    assert calibration["student_fire_fraction"] == pytest.approx(
        holdout["student_fire_predictions"] / holdout["sample_count"]
    )
    json.dumps(v1._jsonable(completion), allow_nan=False)
    # CLI: default None (current behavior), three floats when given.
    parser = trainer.build_parser()
    args = parser.parse_args(["--collection-dir", "somewhere"])
    assert args.verb_class_weights is None
    args = parser.parse_args(
        [
            "--collection-dir",
            "somewhere",
            "--verb-class-weights",
            "1.0",
            "6.0",
            "1.0",
        ]
    )
    assert args.verb_class_weights == [1.0, 6.0, 1.0]


# ---------------------------------------------------------------------------
# (c) Round trip: the zip runs in the macro-native eval probe unchanged.
# ---------------------------------------------------------------------------


class _StubMacroVector:
    """Macro-wrapped vec-env stand-in for the probe's stubbed-env path."""

    def __init__(self, count: int, episode_macros: int, width: int) -> None:
        self.count = count
        self.episode_macros = episode_macros
        self.width = width
        self._step = 0

    def seed(self, value: int) -> None:
        self.seed_value = int(value)

    def reset(self) -> np.ndarray:
        self._step = 0
        return np.zeros((self.count, self.width), dtype=np.float32)

    def step(self, actions: Any) -> tuple[Any, Any, Any, Any]:
        actions = np.asarray(actions, dtype=np.int64)
        step = self._step
        self._step += 1
        done = step == self.episode_macros - 1
        infos = []
        for index in range(self.count):
            verb = int(actions[index][0])
            aim = int(actions[index][1])
            outcome = "win" if done else None
            info = {
                "score": (step + 1) * 10,
                "ticks": (step + 1) * 20,
                "outcome": outcome,
                "native_outcome": outcome,
                "TimeLimit.truncated": False,
                "park_settle": _macro_info(verb, aim),
            }
            infos.append(info)
        observations = np.zeros(
            (self.count, self.width), dtype=np.float32
        )
        rewards = np.zeros(self.count, dtype=np.float64)
        dones = np.full(self.count, done, dtype=np.bool_)
        return observations, rewards, dones, infos

    def close(self) -> None:
        pass


def test_round_trip_zip_runs_in_macro_native_eval_stub_path(
    tmp_path: Path,
) -> None:
    run_dir = _write_collection(tmp_path, decisions=24)
    dataset, receipt = trainer.load_decision_dataset(
        run_dir / "episodes_manifest.json"
    )
    train_config = _train_config(epochs=1, checkpoint_interval_epochs=0)
    model = _toy_stacked_model(train_config)
    completion = trainer.execute_macro_training(
        model=model,
        dataset=dataset,
        run_dir=tmp_path / "round-trip-run",
        model_config=trainer.MacroDecisionModelConfig(
            learned_features_dim=16, model_seed=99_081_690
        ),
        train_config=train_config,
        dataset_receipt=receipt,
        stack_lags=STACK_LAGS,
    )
    from sb3_contrib import MaskablePPO

    # Plain load: the pickled policy_kwargs resolve the extractor from
    # tools.distill_alphazuma_55_park_settle_v3 (the probe imports it
    # for exactly this side effect).
    reloaded = MaskablePPO.load(
        str(completion["final_model"]["path"]), device="cpu"
    )
    assert native_probe.require_macro_native_model(reloaded) == (3, 180)
    extractor = reloaded.policy.features_extractor
    assert type(extractor) is trainer_v3.StackedRevengeEntityFeatureExtractor
    signature = native_probe.model_space_signature(reloaded)
    assert signature["action_space"]["nvec"] == [3, 180]
    assert signature["observation_space"]["shape"] == [_toy_stacked_dim()]
    # The probe's own run loop executes the model UNCHANGED on its
    # stubbed-env path: identity mapping, zero fallbacks.
    episode_macros = 5
    result = native_probe._run_macro_native(
        model=reloaded,
        vector=_StubMacroVector(1, episode_macros, _toy_stacked_dim()),
        level_ids=("Jungle1",),
        seed_base=native_probe.DEFAULT_SEED_BASE,
        max_ticks=100,
        masks_provider=lambda _: np.ones((1, 183), dtype=np.bool_),
    )
    summary = result["summary"]
    assert summary["attempts"] == 1
    assert summary["wins"] == 1
    assert summary["mask_fallbacks"] == {}
    assert summary["total_macro_decisions"] == episode_macros
    assert result["episodes"][0]["mask_fallbacks"] == {}


# ---------------------------------------------------------------------------
# CLI contract: defaults, validate-only, stack-lags cross-check.
# ---------------------------------------------------------------------------


def test_cli_defaults_validate_only_and_stack_lag_cross_check(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    parser = trainer.build_parser()
    args = parser.parse_args(["--collection-dir", "somewhere"])
    assert args.epochs == 20
    assert args.batch_size == 512
    assert args.learning_rate == pytest.approx(3.0e-4)
    assert args.holdout_fraction == pytest.approx(0.2)
    assert args.checkpoint_interval_epochs == 4
    assert args.verb_loss_weight == pytest.approx(1.0)
    assert args.aim_loss_weight == pytest.approx(1.0)
    assert args.learned_features_dim == 2048
    assert args.device == "cpu"
    assert args.levels is None
    assert args.observation_backing == "auto"
    assert args.ram_budget_gib == pytest.approx(16.0)
    assert args.observation_cache_dir is None

    run_dir = _write_collection(tmp_path, decisions=6)
    assert (
        trainer.main(
            ["--collection-dir", str(run_dir), "--validate-only"]
        )
        == 0
    )
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "VALID"
    assert printed["dataset"]["sample_count"] == 4 * 6
    assert printed["dataset"]["stack_lags"] == list(STACK_LAGS)
    assert printed["dataset"]["observation_backing"]["mode"] == "ram"

    # Memmap backing and the cache-dir override are first-class flags.
    cache_dir = tmp_path / "cli-cache"
    assert (
        trainer.main(
            [
                "--collection-dir",
                str(run_dir),
                "--observation-backing",
                "memmap",
                "--observation-cache-dir",
                str(cache_dir),
                "--validate-only",
            ]
        )
        == 0
    )
    printed = json.loads(capsys.readouterr().out)
    backing = printed["dataset"]["observation_backing"]
    assert backing["mode"] == "memmap"
    assert backing["extraction"]["action"] == "extracted"
    assert (
        Path(backing["observations_npy"]["path"]).resolve().parent
        == cache_dir.resolve()
    )

    # The per-level filter is a first-class CLI path.
    assert (
        trainer.main(
            [
                "--collection-dir",
                str(run_dir),
                "--levels",
                "Jungle2",
                "--validate-only",
            ]
        )
        == 0
    )
    printed = json.loads(capsys.readouterr().out)
    assert printed["dataset"]["levels"] == ["Jungle2"]
    assert printed["dataset"]["sample_count"] == 2 * 6

    # A --stack-lags disagreement with the manifest is refused.
    with pytest.raises(SystemExit, match="disagrees with the collection"):
        trainer.main(
            [
                "--collection-dir",
                str(run_dir),
                "--stack-lags",
                "0,2",
                "--validate-only",
            ]
        )
