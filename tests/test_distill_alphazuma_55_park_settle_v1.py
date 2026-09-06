from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tools import distill_alphazuma_55_park_settle_v1 as park_settle


def _synthetic_dataset(
    *,
    seed: int,
    episodes: int = 4,
    ticks_per_episode: int = 60,
    fire_tick: int | None = None,
    illegal_fire_tick: int | None = None,
    obs_dim: int | None = None,
) -> park_settle.ParkSettleDataset:
    """Episode streams: parked waits, one legal fire, one illegal fire."""

    if fire_tick is None:
        fire_tick = ticks_per_episode // 2
    if illegal_fire_tick is None:
        illegal_fire_tick = (3 * ticks_per_episode) // 4
    if obs_dim is None:
        observation_space, _, _ = park_settle.tiny_entity_polar_interface()
        obs_dim = int(observation_space.shape[0])
    rng = np.random.default_rng(seed)
    count = episodes * ticks_per_episode
    observations = rng.uniform(-1.0, 1.0, size=(count, obs_dim)).astype(
        np.float32
    )
    raw_actions = np.zeros((count, 2), dtype=np.int64)
    masks = np.ones((count, park_settle.MASK_WIDTH), dtype=np.bool_)
    episode_indices = np.repeat(
        np.arange(episodes, dtype=np.int64), ticks_per_episode
    )
    tick_indices = np.tile(
        np.arange(ticks_per_episode, dtype=np.int64), episodes
    )
    for episode in range(episodes):
        base = episode * ticks_per_episode
        parked_bin = 10 + 7 * episode
        raw_actions[base : base + ticks_per_episode, 1] = parked_bin
        raw_actions[base + fire_tick, 0] = park_settle.FIRE_VERB
        raw_actions[base + illegal_fire_tick, 0] = park_settle.FIRE_VERB
        masks[base + illegal_fire_tick, park_settle.FIRE_VERB] = False
    return park_settle.ParkSettleDataset.from_episode_rows(
        observations=observations,
        raw_actions=raw_actions,
        masks=masks,
        episode_indices=episode_indices,
        tick_indices=tick_indices,
    )


def _tiny_model(
    *, learning_rate: float = 1.0e-3, batch_size: int = 32, seed: int = 7
):
    observation_space, action_space, policy_kwargs = (
        park_settle.tiny_entity_polar_interface(features_dim=16)
    )
    model = park_settle.build_student_model(
        observation_space=observation_space,
        action_space=action_space,
        policy_kwargs=policy_kwargs,
        model_config=park_settle.ParkSettleModelConfig(
            learned_features_dim=16,
            aim_head_identity_scale=5.0,
            model_seed=seed,
        ),
        train_config=park_settle.ParkSettleTrainConfig(
            epochs=1,
            batch_size=batch_size,
            learning_rate=learning_rate,
            holdout_fraction=0.5,
            shuffle_seed=seed,
            device="cpu",
        ),
    )
    return model


def test_annotations_weight_pre_fire_window_and_parked_waits() -> None:
    loss = park_settle.ParkSettleLossConfig(
        fire_window_ticks=10, wait_aim_weight=0.25
    )
    dataset = _synthetic_dataset(seed=11, episodes=1, ticks_per_episode=60)
    annotations = park_settle.annotate_park_settle(dataset, loss)

    fire_tick = 30
    illegal_tick = 45
    assert bool(annotations.fire_commit[fire_tick])
    assert bool(annotations.fire_edge[fire_tick])
    assert not bool(annotations.fire_commit[illegal_tick])
    assert bool(annotations.fire_intent[illegal_tick])
    assert not bool(annotations.verb_legal[illegal_tick])

    expected_window = np.zeros(60, dtype=np.bool_)
    expected_window[fire_tick - 10 : fire_tick + 1] = True
    assert np.array_equal(annotations.pre_fire_window, expected_window)
    # Wait ticks (parked target carried in the aim label) keep the reduced
    # weight; every tick inside the pre-fire window is weighted 1.0.
    assert np.allclose(
        annotations.aim_weights,
        np.where(expected_window, 1.0, 0.25).astype(np.float32),
    )
    # The illegal fire intent sits outside the window: parked weight.
    assert annotations.aim_weights[illegal_tick] == pytest.approx(0.25)
    # The aim label on wait ticks is the parked target bin.
    assert int(dataset.raw_actions[0, 1]) == int(
        dataset.raw_actions[fire_tick, 1]
    )


def test_batch_step_weights_aim_loss_and_drops_illegal_verbs() -> None:
    import torch
    import torch.nn.functional as functional

    loss = park_settle.ParkSettleLossConfig(
        fire_window_ticks=10, wait_aim_weight=0.25
    )
    dataset = _synthetic_dataset(seed=13, episodes=2, ticks_per_episode=40)
    annotations = park_settle.annotate_park_settle(dataset, loss)
    model = _tiny_model()
    rows = np.arange(dataset.sample_count, dtype=np.int64)

    observation_tensor, _ = model.policy.obs_to_tensor(
        dataset.observations[rows]
    )
    with torch.no_grad():
        verb_logits, aim_logits = park_settle._policy_logits(
            model, observation_tensor
        )
    action_tensor = torch.as_tensor(dataset.raw_actions[rows])
    weight_tensor = torch.as_tensor(annotations.aim_weights[rows])
    legal_tensor = torch.as_tensor(annotations.verb_legal[rows])
    aim_nll = functional.cross_entropy(
        aim_logits, action_tensor[:, 1], reduction="none"
    )
    expected_aim = float(
        (weight_tensor * aim_nll).sum() / weight_tensor.sum()
    )
    unweighted_aim = float(aim_nll.mean())
    masked_verb_logits = verb_logits.masked_fill(
        ~torch.as_tensor(dataset.masks[rows, : park_settle.VERB_COUNT]),
        park_settle.VERB_MASK_FILL,
    )
    expected_verb = float(
        functional.cross_entropy(
            masked_verb_logits[legal_tensor],
            action_tensor[legal_tensor, 0],
        )
    )
    parameter_before = model.policy.action_net.weight.detach().clone()

    metrics = park_settle._train_park_settle_batch(
        model=model,
        dataset=dataset,
        annotations=annotations,
        rows=rows,
        loss=loss,
    )

    # (a) wait ticks contribute at wait_aim_weight, pre-fire ticks at 1.0.
    window_count = int(annotations.pre_fire_window.sum())
    parked_count = dataset.sample_count - window_count
    assert metrics["aim_weight_sum"] == pytest.approx(
        window_count * 1.0 + parked_count * 0.25
    )
    assert metrics["aim_loss"] == pytest.approx(expected_aim, rel=1.0e-5)
    assert not math.isclose(expected_aim, unweighted_aim, abs_tol=1.0e-9)
    assert metrics["pre_fire_window_sample_count"] == window_count

    # (b) illegal-verb ticks are excluded from the verb loss but present in
    # the aim loss.
    illegal_count = int(np.sum(~annotations.verb_legal))
    assert illegal_count == 2
    assert metrics["dropped_illegal_verb_count"] == illegal_count
    assert metrics["verb_loss_sample_count"] == (
        dataset.sample_count - illegal_count
    )
    assert metrics["aim_loss_sample_count"] == dataset.sample_count
    assert metrics["verb_loss"] == pytest.approx(expected_verb, rel=1.0e-5)

    # The optimizer actually stepped.
    assert not torch.equal(
        parameter_before, model.policy.action_net.weight.detach()
    )
    # (c) masked logits never produce an illegal argmax.
    assert metrics["masked_argmax_all_mask_legal"] is True


def test_masked_deterministic_verbs_are_never_illegal() -> None:
    dataset = _synthetic_dataset(seed=17, episodes=2, ticks_per_episode=30)
    # Restrict the runtime mask hard: only wait legal on even rows, only
    # wait and swap legal on odd rows.
    dataset.masks[:, 1:park_settle.VERB_COUNT] = False
    dataset.masks[1::2, 2] = True
    loss = park_settle.ParkSettleLossConfig()
    annotations = park_settle.annotate_park_settle(dataset, loss)
    model = _tiny_model(seed=21)
    calibration = park_settle.calibrate_park_settle(
        model=model,
        dataset=dataset,
        annotations=annotations,
        indices=np.arange(dataset.sample_count, dtype=np.int64),
        batch_size=16,
    )
    assert calibration["deterministic_verbs_all_mask_legal"] is True
    # Fire is masked everywhere, so the deterministic policy cannot fire.
    assert calibration["student_deterministic_fire_rate"] == 0.0


def test_execute_training_writes_receipted_completion(
    tmp_path: Path,
) -> None:
    dataset = _synthetic_dataset(seed=19, episodes=4, ticks_per_episode=40)
    model = _tiny_model()
    train_config = park_settle.ParkSettleTrainConfig(
        epochs=1,
        batch_size=32,
        learning_rate=1.0e-3,
        holdout_fraction=0.5,
        shuffle_seed=23,
        device="cpu",
        checkpoint_interval_epochs=1,
        loss=park_settle.ParkSettleLossConfig(
            fire_window_ticks=10, wait_aim_weight=0.25
        ),
    )
    model_config = park_settle.ParkSettleModelConfig(
        learned_features_dim=16, model_seed=7
    )
    run_dir = tmp_path / "park-settle-run"
    completion = park_settle.execute_training(
        model=model,
        dataset=dataset,
        run_dir=run_dir,
        model_config=model_config,
        train_config=train_config,
    )

    written = json.loads(
        (run_dir / "completion.json").read_text(encoding="utf-8")
    )
    assert written["schema"] == (
        "zuma-rl.alphazuma-55-park-settle-distillation-completion"
    )
    assert written["status"] == "COMPLETE"
    assert written["recipe"]["verb_batch_rebalancing"] == "none"
    assert written["recipe"]["verb_class_weights"] is None
    assert written["recipe"]["aim_loss_scope"] == (
        "all_ticks_park_and_settle"
    )
    assert written["optimization"]["sampling"] == (
        "uniform_shuffle_no_verb_rebalance"
    )

    # (d) the calibration report is present with the required fields.
    calibration = written["calibration"]
    assert calibration["slice"] == "holdout"
    for name in (
        "student_deterministic_fire_rate",
        "teacher_executed_fire_rate",
        "student_teacher_fire_rate_ratio",
        "fire_cycle_upper_bound_fire_rate",
        "fire_commit_aim",
        "pre_fire_window_aim",
    ):
        assert name in calibration
    assert calibration["fire_cycle_upper_bound_fire_rate"] == (
        pytest.approx(1.0 / 21.0)
    )
    for group in ("fire_commit_aim", "pre_fire_window_aim"):
        for metric in (
            "sample_count",
            "exact_accuracy",
            "within_three_accuracy",
        ):
            assert metric in calibration[group]

    # The illegal-teacher-verb fraction is reported: one illegal fire per
    # 40-tick episode.
    assert written["illegal_teacher_verb_fraction"]["overall"] == (
        pytest.approx(1.0 / 40.0)
    )
    # sha256 receipts bind the trainer and the final model bytes.
    assert written["trainer"]["sha256"].startswith("sha256:")
    final_model = Path(written["final_model"]["path"])
    assert final_model.exists()
    assert written["final_model"]["sha256"].startswith("sha256:")
    assert written["optimization"]["checkpoints"]
    assert completion["status"] == "COMPLETE"


def test_load_replay_dataset_accepts_both_manifest_layouts(
    tmp_path: Path,
) -> None:
    observation_space, _, _ = park_settle.tiny_entity_polar_interface()
    obs_dim = int(observation_space.shape[0])
    rng = np.random.default_rng(29)

    def _arrays(rows: int) -> dict[str, np.ndarray]:
        actions = np.zeros((rows, 2), dtype=np.int64)
        actions[:, 1] = rng.integers(0, park_settle.AIM_BINS, size=rows)
        actions[::7, 0] = park_settle.FIRE_VERB
        return {
            "observations": rng.uniform(
                -1.0, 1.0, size=(rows, obs_dim)
            ).astype(np.float32),
            "actions": actions,
            "masks": np.ones(
                (rows, park_settle.MASK_WIDTH), dtype=np.bool_
            ),
        }

    # replay-v2 style: reservoir shards without temporal structure.
    shard_dir = tmp_path / "replay-v2"
    shard_dir.mkdir()
    for index in range(2):
        np.savez(shard_dir / f"shard_{index}.npz", **_arrays(12))
    manifest_v2 = shard_dir / "manifest.json"
    manifest_v2.write_text(
        json.dumps(
            {
                "schema": "zuma-rl.test-replay-v2-shards",
                "shards": [
                    {"path": "shard_0.npz"},
                    {"path": "shard_1.npz"},
                ],
            }
        ),
        encoding="utf-8",
    )
    dataset_v2, receipt_v2 = park_settle.load_replay_dataset(manifest_v2)
    assert receipt_v2["layout"] == "motor-observable-replay-v2-shards"
    assert dataset_v2.temporal_structure == "absent"
    assert dataset_v2.sample_count == 24

    # coverage-v4 style: one NPZ per episode, tick order preserved.
    episode_dir = tmp_path / "coverage-v4"
    episode_dir.mkdir()
    for index in range(2):
        arrays = _arrays(15)
        arrays["raw_actions"] = arrays.pop("actions")
        arrays["runtime_masks"] = arrays.pop("masks")
        arrays["tick_indices"] = np.arange(15, dtype=np.int64)
        np.savez(episode_dir / f"episode_{index}.npz", **arrays)
    manifest_v4 = episode_dir / "manifest.json"
    manifest_v4.write_text(
        json.dumps(
            {
                "schema": (
                    "zuma-rl.alphazuma-55-motor-observable-replay-dataset"
                ),
                "version": 4,
                "episodes": [
                    {"path": "episode_0.npz"},
                    {"path": "episode_1.npz"},
                ],
            }
        ),
        encoding="utf-8",
    )
    dataset_v4, receipt_v4 = park_settle.load_replay_dataset(manifest_v4)
    assert receipt_v4["layout"] == (
        "motor-observable-replay-v4-coverage-episodes"
    )
    assert dataset_v4.temporal_structure == "episode_ticks"
    assert dataset_v4.sample_count == 30
    assert len(np.unique(dataset_v4.episode_indices)) == 2
    assert receipt_v4["entries"][0]["keys"]["masks"] == "runtime_masks"

    # Declared shard hashes are enforced.
    bad_manifest = episode_dir / "bad_manifest.json"
    bad_manifest.write_text(
        json.dumps(
            {
                "episodes": [
                    {"path": "episode_0.npz", "sha256": "sha256:0000"}
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="shard bytes differ"):
        park_settle.load_replay_dataset(bad_manifest)


def test_load_replay_dataset_refuses_relaxed_training_masks(
    tmp_path: Path,
) -> None:
    """Relaxed replay-v2 training masks can never enter the trainer.

    Loading them would silently zero dropped_illegal_verb_count and
    report illegal_teacher_verb_fraction == 0.0 against the verified
    ~26% reality -- the blind-metric failure family this recipe exists
    to fix -- so the loader must fail closed on (a) an NPZ offering only
    'training_masks' and (b) a manifest whose mask_semantics marks the
    loaded key as anything but the exact runtime mask.
    """

    observation_space, _, _ = park_settle.tiny_entity_polar_interface()
    obs_dim = int(observation_space.shape[0])
    rng = np.random.default_rng(31)
    rows = 8
    observations = rng.uniform(-1.0, 1.0, size=(rows, obs_dim)).astype(
        np.float32
    )
    actions = np.zeros((rows, 2), dtype=np.int64)
    masks = np.ones((rows, park_settle.MASK_WIDTH), dtype=np.bool_)

    # (a) NPZ shard that only offers the relaxed training mask.
    relaxed_dir = tmp_path / "relaxed"
    relaxed_dir.mkdir()
    np.savez(
        relaxed_dir / "shard_0.npz",
        observations=observations,
        actions=actions,
        training_masks=masks,
    )
    relaxed_manifest = relaxed_dir / "manifest.json"
    relaxed_manifest.write_text(
        json.dumps({"shards": [{"path": "shard_0.npz"}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="relaxed training masks"):
        park_settle.load_replay_dataset(relaxed_manifest)

    # (b) manifest declares non-exact semantics for the loaded key.
    declared_dir = tmp_path / "declared"
    declared_dir.mkdir()
    np.savez(
        declared_dir / "shard_0.npz",
        observations=observations,
        actions=actions,
        masks=masks,
    )
    mislabeled = declared_dir / "mislabeled.json"
    mislabeled.write_text(
        json.dumps(
            {
                "shards": [{"path": "shard_0.npz"}],
                "mask_semantics": {
                    "masks": "relaxed_training_mask_teacher_verb_unmasked"
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="mask semantics"):
        park_settle.load_replay_dataset(mislabeled)

    # A declaration marking the loaded key as exact loads and is
    # recorded in the receipt (and thus in the completion receipt).
    exact = declared_dir / "exact.json"
    exact.write_text(
        json.dumps(
            {
                "shards": [{"path": "shard_0.npz"}],
                "mask_semantics": {
                    "masks": "exact_runtime_valid_action_mask"
                },
            }
        ),
        encoding="utf-8",
    )
    dataset, receipt = park_settle.load_replay_dataset(exact)
    assert dataset.sample_count == rows
    assert receipt["mask_semantics"] == {
        "masks": "exact_runtime_valid_action_mask"
    }
