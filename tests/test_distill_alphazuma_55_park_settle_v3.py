"""Tests for the lag-stacked park-settle distillation wrapper (v3).

Covers, on small synthetic CPU data:
(a) gather-index correctness: exact lagged rows, nearest-earlier gap
    fallbacks, episode-start clamps (including the duplicate-current-row
    degenerate case), episode-boundary isolation, honest per-type
    statistics, and the stacked view's feature-axis concatenation;
(b) the degenerate stack (``--stack-lags 0``) reproduces the v1/v2
    training results bit-identically (same tiny dataset, same seeds,
    state_dict allclose) -- the stack must not change semantics;
(c) a real multi-lag run consumes the wider input end to end and the
    completion receipt records stack_lags plus the gather-fallback
    statistics;
(d) refusals: stacking without episode_ticks temporal structure, and
    malformed ``--stack-lags`` values.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools import distill_alphazuma_55 as legacy
from tools import distill_alphazuma_55_park_settle_v1 as v1
from tools import distill_alphazuma_55_park_settle_v2 as v2
from tools import distill_alphazuma_55_park_settle_v3 as v3


def _synthetic_arrays(
    *,
    seed: int,
    episodes: int = 3,
    ticks_per_episode: int = 30,
    obs_dim: int | None = None,
) -> dict[str, np.ndarray]:
    """Coverage-v4-style aggregate arrays (same shape family as v2 tests)."""

    if obs_dim is None:
        observation_space, _, _ = v1.tiny_entity_polar_interface()
        obs_dim = int(observation_space.shape[0])
    fire_tick = ticks_per_episode // 2
    illegal_fire_tick = (3 * ticks_per_episode) // 4
    rng = np.random.default_rng(seed)
    count = episodes * ticks_per_episode
    observations = rng.uniform(-1.0, 1.0, size=(count, obs_dim)).astype(
        np.float32
    )
    actions = np.zeros((count, 2), dtype=np.int64)
    exact_masks = np.ones((count, v1.MASK_WIDTH), dtype=np.bool_)
    episode_indices = np.repeat(
        7 * np.arange(episodes, dtype=np.int64) + 3, ticks_per_episode
    )
    tick_indices = np.tile(
        np.arange(ticks_per_episode, dtype=np.int64), episodes
    )
    for episode in range(episodes):
        start = episode * ticks_per_episode
        actions[start : start + ticks_per_episode, 1] = 10 + 7 * episode
        actions[start + fire_tick, 0] = v1.FIRE_VERB
        actions[start + illegal_fire_tick, 0] = v1.FIRE_VERB
        exact_masks[start + illegal_fire_tick, v1.FIRE_VERB] = False
    return {
        "observations": observations,
        "actions": actions,
        "exact_masks": exact_masks,
        "episode_indices": episode_indices,
        "tick_indices": tick_indices,
    }


def _write_coverage_dataset(
    directory: Path, arrays: dict[str, np.ndarray]
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    npz_path = directory / "replay.npz"
    np.savez_compressed(npz_path, **arrays)
    manifest_path = directory / "replay.dataset.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": (
                    "zuma-rl.alphazuma-55-motor-observable-replay-v4-dataset"
                ),
                "version": 4,
                "temporal_structure": "episode_ticks",
                "mask_semantics": {
                    "exact_masks": "exact_runtime_valid_action_mask",
                },
                "shards": [
                    {
                        "path": "replay.npz",
                        "sha256": legacy._sha256(npz_path),
                        "rows": int(arrays["observations"].shape[0]),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def _tiny_model(*, frames: int, seed: int = 7, batch_size: int = 32):
    observation_space, action_space, policy_kwargs = (
        v1.tiny_entity_polar_interface(features_dim=16)
    )
    from zuma_rl.observation_stack_wrapper import stacked_box

    if frames == 0:  # v1 baseline: untouched single-frame path
        space = observation_space
        kwargs = policy_kwargs
    else:
        space = stacked_box(observation_space, frames)
        kwargs = v3.stack_policy_kwargs(policy_kwargs, frames=frames)
    return v1.build_student_model(
        observation_space=space,
        action_space=action_space,
        policy_kwargs=kwargs,
        model_config=v1.ParkSettleModelConfig(
            learned_features_dim=16,
            aim_head_identity_scale=5.0,
            model_seed=seed,
        ),
        train_config=v1.ParkSettleTrainConfig(
            epochs=1,
            batch_size=batch_size,
            learning_rate=1.0e-3,
            holdout_fraction=0.5,
            shuffle_seed=seed,
            device="cpu",
        ),
    )


def test_gather_covers_exact_gap_and_episode_start_fallbacks() -> None:
    # Episode 7: ticks 0,1,2,3,10,11,12 (rows 0-6, gap 4..9);
    # episode 9: ticks 5,6,7 (rows 7-9, starts mid-stream).
    episode_indices = np.asarray([7] * 7 + [9] * 3, dtype=np.int64)
    tick_indices = np.asarray(
        [0, 1, 2, 3, 10, 11, 12, 5, 6, 7], dtype=np.int64
    )
    gather, stats = v3.build_stack_gather(
        episode_indices=episode_indices,
        tick_indices=tick_indices,
        stack_lags=(0, 2, 4),
    )
    assert gather.dtype == np.int32
    # Lag 0 is always the row itself.
    assert gather[:, 0].tolist() == list(range(10))
    # Lag 2: episode-start clamps at ticks 0,1 (and 5,6 of episode 9),
    # exact at ticks 2,3,12 and 7, nearest-earlier across the 4..9 gap.
    assert gather[:, 1].tolist() == [0, 0, 0, 1, 3, 3, 4, 7, 7, 7]
    # Lag 4: every episode-7 head row clamps to row 0; the gap rows all
    # fall back to tick 3 (row 3); episode 9 never crosses into episode 7.
    assert gather[:, 2].tolist() == [0, 0, 0, 0, 3, 3, 3, 7, 7, 7]

    assert stats["rows"] == 10
    assert stats["stack_lags"] == [0, 2, 4]
    assert stats["lagged_reads"] == 30
    assert stats["exact"] == 14
    assert stats["nearest_earlier"] == 5
    assert stats["episode_start_clamp"] == 11
    assert stats["fallback_reads"] == 16
    # Rows 0 and 7 are their episodes' first retained rows: their
    # episode-start clamps duplicate the current row (two lags each).
    assert stats["episode_start_duplicate_current_row"] == 4
    assert stats["per_lag"]["0"]["exact"] == 10
    assert stats["per_lag"]["2"] == {
        "exact": 4,
        "nearest_earlier": 2,
        "episode_start_clamp": 4,
    }
    assert stats["per_lag"]["4"] == {
        "exact": 0,
        "nearest_earlier": 3,
        "episode_start_clamp": 7,
    }

    # Train/runtime divergence is quantified and receipted: reads whose
    # gathered tick differs from the runtime wrapper's max(t - lag, 0).
    # Lag 2: ticks 10,11 gather tick 3 (gaps 5,6); episode-9 ticks 5,6
    # clamp to tick 5 while runtime holds ticks 3,4 (gaps 2,1).
    # Lag 4: ticks 10,11,12 gather tick 3 (gaps 3,4,5); episode-9 rows
    # clamp to tick 5 while runtime holds 1,2,3 (gaps 4,3,2).  Episode-7
    # head rows clamp to tick 0 == runtime, so they do NOT diverge.
    divergence = stats["train_runtime_divergence"]
    assert divergence["divergent_reads"] == 10
    assert divergence["divergent_fraction"] == pytest.approx(10 / 30)
    assert divergence["episodes"] == 2
    assert divergence["episodes_retaining_tick_zero"] == 1
    assert divergence["per_lag"]["0"] == {
        "divergent_reads": 0,
        "divergent_fraction": 0.0,
        "divergence_ticks": None,
    }
    lag2 = divergence["per_lag"]["2"]
    assert lag2["divergent_reads"] == 4
    assert lag2["divergent_fraction"] == pytest.approx(0.4)
    assert lag2["divergence_ticks"]["max"] == 6
    assert lag2["divergence_ticks"]["p50"] == pytest.approx(3.5)
    lag4 = divergence["per_lag"]["4"]
    assert lag4["divergent_reads"] == 6
    assert lag4["divergence_ticks"]["max"] == 5
    assert lag4["divergence_ticks"]["p50"] == pytest.approx(3.5)
    assert "preregistrations" in divergence["disclosure"]

    # Row order independence: interleaving the episodes must map the
    # SAME (episode, tick) pairs to the same source rows.
    permutation = np.asarray([7, 0, 8, 1, 9, 2, 3, 4, 5, 6])
    inverse = np.argsort(permutation)
    shuffled_gather, shuffled_stats = v3.build_stack_gather(
        episode_indices=episode_indices[permutation],
        tick_indices=tick_indices[permutation],
        stack_lags=(0, 2, 4),
    )
    assert np.array_equal(
        permutation[shuffled_gather[inverse]], gather
    )
    for key in ("exact", "nearest_earlier", "episode_start_clamp"):
        assert shuffled_stats[key] == stats[key]
    assert shuffled_stats["train_runtime_divergence"] == divergence

    # The stacked view concatenates the gathered rows along features.
    observations = (
        np.arange(10, dtype=np.float32).reshape(10, 1)
        * np.ones((10, 5), dtype=np.float32)
    )
    view = v3.StackedObservationView(observations, gather)
    assert view.shape == (10, 15)
    assert view.ndim == 2
    stacked = view[np.asarray([4, 9], dtype=np.int64)]
    assert stacked.dtype == np.float32
    assert np.array_equal(
        stacked,
        np.concatenate(
            [
                observations[[4, 9]],
                observations[[3, 7]],
                observations[[3, 7]],
            ],
            axis=1,
        ),
    )
    with pytest.raises(TypeError, match="1-D integer"):
        view[3:5]

    # Duplicate ticks inside one episode are refused, not misgathered.
    with pytest.raises(ValueError, match="unique"):
        v3.build_stack_gather(
            episode_indices=np.zeros(3, dtype=np.int64),
            tick_indices=np.asarray([0, 1, 1], dtype=np.int64),
            stack_lags=(0, 2),
        )


def test_degenerate_stack_training_bit_identical_to_v1(
    tmp_path: Path,
) -> None:
    import torch

    arrays = _synthetic_arrays(seed=19, episodes=4, ticks_per_episode=40)
    manifest_path = _write_coverage_dataset(tmp_path / "data", arrays)
    dataset, receipt = v2.load_replay_dataset_memmap(manifest_path)

    train_config = v1.ParkSettleTrainConfig(
        epochs=2,
        batch_size=32,
        learning_rate=1.0e-3,
        holdout_fraction=0.5,
        shuffle_seed=23,
        device="cpu",
        checkpoint_interval_epochs=1,
        loss=v1.ParkSettleLossConfig(
            fire_window_ticks=10, wait_aim_weight=0.25
        ),
    )
    model_config = v1.ParkSettleModelConfig(
        learned_features_dim=16, model_seed=7
    )

    model_a = _tiny_model(frames=0, seed=7)
    completion_a = v1.execute_training(
        model=model_a,
        dataset=dataset,
        run_dir=tmp_path / "run-v1",
        model_config=model_config,
        train_config=train_config,
        dataset_receipt=receipt,
    )

    stacked_dataset, gather, stats = v3.stack_dataset(
        dataset, stack_lags=(0,)
    )
    assert stats["fallback_reads"] == 0
    assert stats["train_runtime_divergence"]["divergent_reads"] == 0
    assert np.array_equal(
        gather[:, 0], np.arange(dataset.sample_count, dtype=np.int64)
    )
    model_b = _tiny_model(frames=1, seed=7)
    completion_b = v3.execute_stacked_training(
        model=model_b,
        dataset=stacked_dataset,
        run_dir=tmp_path / "run-v3",
        model_config=model_config,
        train_config=train_config,
        dataset_receipt=receipt,
        stack_lags=(0,),
        gather_stats=stats,
    )

    epochs_a = completion_a["optimization"]["epochs"]
    epochs_b = completion_b["optimization"]["epochs"]
    assert len(epochs_a) == len(epochs_b) == 2
    for row_a, row_b in zip(epochs_a, epochs_b):
        for metric in ("mean_loss", "mean_verb_loss", "mean_aim_loss"):
            assert row_b[metric] == pytest.approx(
                row_a[metric], rel=1.0e-6, abs=1.0e-9
            )
    state_a = model_a.policy.state_dict()
    state_b = model_b.policy.state_dict()
    assert state_a.keys() == state_b.keys()
    for key in state_a:
        assert torch.allclose(
            state_a[key], state_b[key], rtol=1.0e-6, atol=1.0e-7
        ), key
    assert completion_b["split"] == completion_a["split"]
    assert completion_b["calibration"][
        "student_deterministic_fire_rate"
    ] == pytest.approx(
        completion_a["calibration"]["student_deterministic_fire_rate"]
    )

    # The v3 completion keeps the v1 schema and additionally records the
    # stack; the amendment must land in the written completion.json too.
    assert completion_b["schema"] == completion_a["schema"]
    assert "observation_stack" not in completion_a
    stack = completion_b["observation_stack"]
    assert stack["stack_lags"] == [0]
    assert stack["gather_fallback"]["fallback_reads"] == 0
    written = json.loads(
        (tmp_path / "run-v3" / "completion.json").read_text(
            encoding="utf-8"
        )
    )
    assert written["observation_stack"]["stack_lags"] == [0]
    assert written["observation_stack"]["gather_fallback"]["exact"] == (
        dataset.sample_count
    )


def test_multi_lag_training_consumes_wider_input_and_records_stats(
    tmp_path: Path,
) -> None:
    arrays = _synthetic_arrays(seed=29, episodes=3, ticks_per_episode=24)
    manifest_path = _write_coverage_dataset(tmp_path / "data", arrays)
    dataset, receipt = v2.load_replay_dataset_memmap(manifest_path)
    obs_dim = int(dataset.observations.shape[1])

    stacked_dataset, _, stats = v3.stack_dataset(
        dataset, stack_lags=(0, 2)
    )
    assert stacked_dataset.observations.shape == (
        dataset.sample_count,
        2 * obs_dim,
    )
    # Each episode's first two ticks clamp their lag-2 read.
    assert stats["episode_start_clamp"] == 6
    assert stats["nearest_earlier"] == 0
    assert stats["exact"] == dataset.sample_count + (
        dataset.sample_count - 6
    )
    # Contiguous episodes retained from tick 0: the clamps land exactly
    # where the runtime wrapper clamps, so the receipted divergence is 0.
    divergence = stats["train_runtime_divergence"]
    assert divergence["divergent_reads"] == 0
    assert divergence["episodes"] == 3
    assert divergence["episodes_retaining_tick_zero"] == 3

    model = _tiny_model(frames=2, seed=11)
    extractor = model.policy.features_extractor
    assert isinstance(extractor, v3.StackedRevengeEntityFeatureExtractor)
    assert extractor.stacked_frames == 2
    assert extractor.frame_width == obs_dim
    assert extractor.learned_features_dim == 2 * 16
    # The polar aim head identity anchor survived the widened latent.
    assert model.policy.action_net.in_features == 2 * 16 + v1.AIM_BINS
    assert int(model.observation_space.shape[0]) == 2 * obs_dim

    train_config = v1.ParkSettleTrainConfig(
        epochs=1,
        batch_size=32,
        learning_rate=1.0e-3,
        holdout_fraction=0.5,
        shuffle_seed=11,
        device="cpu",
        checkpoint_interval_epochs=0,
        loss=v1.ParkSettleLossConfig(fire_window_ticks=6),
    )
    completion = v3.execute_stacked_training(
        model=model,
        dataset=stacked_dataset,
        run_dir=tmp_path / "run-stacked",
        model_config=v1.ParkSettleModelConfig(
            learned_features_dim=16, model_seed=11
        ),
        train_config=train_config,
        dataset_receipt=receipt,
        stack_lags=(0, 2),
        gather_stats=stats,
    )
    assert completion["status"] == "COMPLETE"
    assert completion["observation_stack"]["stack_lags"] == [0, 2]
    assert completion["observation_stack"]["gather_fallback"] == (
        v1._jsonable(stats)
    )
    assert completion["optimization"]["updates"] >= 1
    assert completion["calibration"]["sample_count"] > 0


def test_stack_refusals_and_validate_only_receipt(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Multi-lag stacking without temporal structure is refused.
    rows = [
        (
            np.zeros(8, dtype=np.float32),
            np.asarray([0, 3], dtype=np.int64),
            np.ones(v1.MASK_WIDTH, dtype=np.bool_),
        )
        for _ in range(4)
    ]
    flat = v1.ParkSettleDataset.from_rows(rows)
    with pytest.raises(ValueError, match="episode_ticks"):
        v3.stack_dataset(flat, stack_lags=(0, 4))
    # The degenerate stack stays available for absent structure.
    stacked, _, _ = v3.stack_dataset(flat, stack_lags=(0,))
    assert stacked.observations.shape == (4, 8)

    from zuma_rl.observation_stack_wrapper import parse_stack_lags

    with pytest.raises(ValueError, match="first stack lag"):
        parse_stack_lags("4,0,8")
    with pytest.raises(ValueError, match="unique"):
        parse_stack_lags((0, 4, 4))
    with pytest.raises(ValueError, match="non-empty"):
        parse_stack_lags("")
    with pytest.raises(ValueError, match="non-negative"):
        parse_stack_lags((0, -2))

    # --validate-only flows the stack receipt through the CLI without
    # touching an environment.
    arrays = _synthetic_arrays(seed=31, episodes=2, ticks_per_episode=12)
    manifest_path = _write_coverage_dataset(tmp_path / "data", arrays)
    assert (
        v3.main(
            [
                "--dataset-manifest",
                str(manifest_path),
                "--stack-lags",
                "0,2,4",
                "--validate-only",
            ]
        )
        == 0
    )
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "VALID"
    stack = printed["dataset"]["observation_stack"]
    assert stack["stack_lags"] == [0, 2, 4]
    assert stack["stacked_observation_width"] == 3 * int(
        arrays["observations"].shape[1]
    )
    assert stack["gather_fallback"]["lagged_reads"] == 3 * int(
        arrays["observations"].shape[0]
    )
    # The divergence disclosure rides every receipt, --validate-only too.
    divergence = stack["gather_fallback"]["train_runtime_divergence"]
    assert divergence["divergent_reads"] == 0
    assert "disclosure" in divergence
    assert printed["dataset"]["observation_backing"]["mode"] == "memmap"
