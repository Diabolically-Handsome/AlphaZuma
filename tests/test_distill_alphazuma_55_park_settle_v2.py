"""Tests for the memmap-backed park-settle distillation wrapper (v2).

Covers, on small synthetic CPU data:
(a) receipted extraction: bit-exact round trip, reuse-on-matching
    receipts, refusal on any receipt mismatch or orphaned artifact;
(b) the memmap loading path produces IDENTICAL training results to
    v1's in-RAM path (same tiny dataset, same seeds, 2 epochs);
(c) RAM boundedness structurally: extraction never ``np.load``s the
    source NPZ, and the loader never decompresses the observations
    member (only the raw .npy is opened, and only with
    ``mmap_mode="r"``);
(d) v1 stays untouched: the CLI delegation restores
    ``v1.load_replay_dataset`` (v1's own test file is run alongside
    this one for the full regression).
"""

from __future__ import annotations

import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from tools import distill_alphazuma_55 as legacy
from tools import distill_alphazuma_55_park_settle_v1 as v1
from tools import distill_alphazuma_55_park_settle_v2 as v2


def _synthetic_arrays(
    *,
    seed: int,
    episodes: int = 3,
    ticks_per_episode: int = 30,
    obs_dim: int | None = None,
) -> dict[str, np.ndarray]:
    """Coverage-v4-style aggregate arrays: parked waits, legal fire,
    illegal fire (exact mask forbids it), non-contiguous episode ids."""

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
    training_masks = np.ones((count, v1.MASK_WIDTH), dtype=np.bool_)
    episode_indices = np.repeat(
        7 * np.arange(episodes, dtype=np.int64) + 3, ticks_per_episode
    )
    tick_indices = np.tile(
        np.arange(ticks_per_episode, dtype=np.int64), episodes
    )
    for episode in range(episodes):
        base = episode * ticks_per_episode
        actions[base : base + ticks_per_episode, 1] = 10 + 7 * episode
        actions[base + fire_tick, 0] = v1.FIRE_VERB
        actions[base + illegal_fire_tick, 0] = v1.FIRE_VERB
        exact_masks[base + illegal_fire_tick, v1.FIRE_VERB] = False
    return {
        "observations": observations,
        "actions": actions,
        "training_masks": training_masks,
        "exact_masks": exact_masks,
        "episode_indices": episode_indices,
        "tick_indices": tick_indices,
    }


def _write_coverage_dataset(
    directory: Path, arrays: dict[str, np.ndarray]
) -> Path:
    """One aggregate NPZ + the builder-shaped ``.dataset.json``."""

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
                "builder": "v4-coverage",
                "temporal_structure": "episode_ticks",
                "mask_semantics": {
                    "exact_masks": "exact_runtime_valid_action_mask",
                    "training_masks": (
                        "relaxed_training_mask_teacher_verb_unmasked"
                    ),
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


def _tiny_model(*, seed: int = 7, batch_size: int = 32):
    observation_space, action_space, policy_kwargs = (
        v1.tiny_entity_polar_interface(features_dim=16)
    )
    return v1.build_student_model(
        observation_space=observation_space,
        action_space=action_space,
        policy_kwargs=policy_kwargs,
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


def test_extraction_round_trip_is_bit_exact_and_receipted(
    tmp_path: Path,
) -> None:
    arrays = _synthetic_arrays(seed=11, obs_dim=24)
    _write_coverage_dataset(tmp_path, arrays)
    npz_path = tmp_path / "replay.npz"

    result = v2.extract_observations(npz_path)

    assert result["action"] == "extracted"
    npy_path = Path(result["npy_path"])
    assert npy_path == tmp_path / "replay.observations.npy"
    # Bit-exact: the raw .npy equals the NPZ zip member byte for byte.
    with zipfile.ZipFile(npz_path) as archive:
        member_bytes = archive.read("observations.npy")
    assert npy_path.read_bytes() == member_bytes
    assert np.array_equal(np.load(npy_path), arrays["observations"])

    receipt_path = Path(result["receipt_path"])
    assert receipt_path == tmp_path / "replay.observations.receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt == result["receipt"]
    assert receipt["schema"] == v2.EXTRACTION_SCHEMA
    assert receipt["source_npz"]["sha256"] == legacy._sha256(npz_path)
    assert receipt["source_npz"]["member"] == "observations.npy"
    assert receipt["observations_npy"]["sha256"] == legacy._sha256(npy_path)
    assert receipt["observations_npy"]["bytes"] == npy_path.stat().st_size
    assert receipt["observations_npy"]["shape"] == list(
        arrays["observations"].shape
    )


def test_extraction_reuses_receipts_and_refuses_mismatches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arrays = _synthetic_arrays(seed=13, obs_dim=16)
    _write_coverage_dataset(tmp_path, arrays)
    npz_path = tmp_path / "replay.npz"
    first = v2.extract_observations(npz_path)
    npy_path = Path(first["npy_path"])
    receipt_path = Path(first["receipt_path"])
    receipt_bytes = receipt_path.read_bytes()

    # Matching receipts are reused WITHOUT re-extraction.
    def _boom(*args, **kwargs):
        raise AssertionError("re-extracted despite a matching receipt")

    monkeypatch.setattr(v2, "_stream_extract_member", _boom)
    second = v2.extract_observations(npz_path)
    assert second["action"] == "reused"
    assert second["receipt"] == first["receipt"]
    monkeypatch.undo()

    # An unreceipted .npy is refused.
    receipt_path.unlink()
    with pytest.raises(ValueError, match="unreceipted"):
        v2.extract_observations(npz_path)
    receipt_path.write_bytes(receipt_bytes)

    # A corrupted .npy is refused (sha256 no longer matches the receipt).
    original_npy = npy_path.read_bytes()
    corrupted = bytearray(original_npy)
    corrupted[-1] ^= 0xFF
    npy_path.write_bytes(bytes(corrupted))
    with pytest.raises(ValueError, match="receipt mismatch"):
        v2.extract_observations(npz_path)
    npy_path.write_bytes(original_npy)

    # A receipt whose .npy vanished is refused, never overwritten.
    npy_path.unlink()
    with pytest.raises(ValueError, match="stale extraction receipt"):
        v2.extract_observations(npz_path)
    npy_path.write_bytes(original_npy)

    # A source NPZ whose bytes changed after extraction is refused.
    changed = dict(arrays)
    changed["observations"] = arrays["observations"] + 1.0
    npz_path.unlink()
    np.savez_compressed(npz_path, **changed)
    with pytest.raises(ValueError, match="source NPZ sha256 changed"):
        v2.extract_observations(npz_path)


def test_memmap_loader_matches_v1_loader_and_is_memmap_backed(
    tmp_path: Path,
) -> None:
    arrays = _synthetic_arrays(seed=17, obs_dim=20)
    manifest_path = _write_coverage_dataset(tmp_path / "data", arrays)

    dataset_v1, receipt_v1 = v1.load_replay_dataset(manifest_path)
    dataset_v2, receipt_v2 = v2.load_replay_dataset_memmap(manifest_path)

    # The v1 dataset object, validated by the v1 checks, with identical
    # contents -- only the observation backing differs.
    assert isinstance(dataset_v2, v1.ParkSettleDataset)
    assert isinstance(dataset_v2.observations.base, np.memmap)
    assert not isinstance(dataset_v1.observations, np.memmap)
    assert np.array_equal(dataset_v2.observations, dataset_v1.observations)
    assert np.array_equal(dataset_v2.raw_actions, dataset_v1.raw_actions)
    assert np.array_equal(dataset_v2.masks, dataset_v1.masks)
    assert np.array_equal(
        dataset_v2.episode_indices, dataset_v1.episode_indices
    )
    assert np.array_equal(dataset_v2.tick_indices, dataset_v1.tick_indices)
    assert dataset_v2.temporal_structure == dataset_v1.temporal_structure
    assert dataset_v2.temporal_structure == "episode_ticks"

    # Receipt parity: identical to v1's except the additive
    # observation_backing block (which lands in completion.json).
    stripped = {
        key: value
        for key, value in receipt_v2.items()
        if key != "observation_backing"
    }
    assert stripped == receipt_v1
    backing = receipt_v2["observation_backing"]
    assert backing["mode"] == "memmap"
    assert backing["extraction"]["action"] in {"extracted", "reused"}
    assert backing["extraction"]["receipt"]["schema"] == (
        v2.EXTRACTION_SCHEMA
    )

    # Memmap backing cannot span shards: multi-NPZ manifests are refused.
    multi_dir = tmp_path / "multi"
    multi_dir.mkdir()
    for name in ("shard_0.npz", "shard_1.npz"):
        np.savez_compressed(multi_dir / name, **arrays)
    multi_manifest = multi_dir / "manifest.json"
    multi_manifest.write_text(
        json.dumps(
            {
                "shards": [
                    {"path": "shard_0.npz"},
                    {"path": "shard_1.npz"},
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exactly one aggregate"):
        v2.load_replay_dataset_memmap(multi_manifest)


def test_memmap_training_results_identical_to_v1_in_ram(
    tmp_path: Path,
) -> None:
    import torch

    arrays = _synthetic_arrays(seed=19, episodes=4, ticks_per_episode=40)
    manifest_path = _write_coverage_dataset(tmp_path / "data", arrays)
    dataset_a, receipt_a = v1.load_replay_dataset(manifest_path)
    dataset_b, receipt_b = v2.load_replay_dataset_memmap(manifest_path)
    assert isinstance(dataset_b.observations.base, np.memmap)

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

    model_a = _tiny_model(seed=7)
    completion_a = v1.execute_training(
        model=model_a,
        dataset=dataset_a,
        run_dir=tmp_path / "run-a",
        model_config=model_config,
        train_config=train_config,
        dataset_receipt=receipt_a,
    )
    model_b = _tiny_model(seed=7)
    completion_b = v1.execute_training(
        model=model_b,
        dataset=dataset_b,
        run_dir=tmp_path / "run-b",
        model_config=model_config,
        train_config=train_config,
        dataset_receipt=receipt_b,
    )

    # Same tiny dataset, same seeds, same recipe: the memmap path must
    # reproduce the in-RAM path -- final losses and weights allclose.
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
    calibration_a = completion_a["calibration"]
    calibration_b = completion_b["calibration"]
    assert calibration_b["student_deterministic_fire_rate"] == (
        pytest.approx(calibration_a["student_deterministic_fire_rate"])
    )
    assert completion_b["split"] == completion_a["split"]

    # The completion receipt additionally records the memmap backing
    # and the extraction receipts; everything else keeps the v1 format.
    assert completion_b["schema"] == completion_a["schema"]
    assert "observation_backing" not in completion_a["dataset"]
    backing = completion_b["dataset"]["observation_backing"]
    assert backing["mode"] == "memmap"
    assert backing["extraction"]["receipt"]["schema"] == (
        v2.EXTRACTION_SCHEMA
    )
    written = json.loads(
        (tmp_path / "run-b" / "completion.json").read_text(
            encoding="utf-8"
        )
    )
    assert written["dataset"]["observation_backing"]["mode"] == "memmap"


def test_extraction_and_loading_are_structurally_ram_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arrays = _synthetic_arrays(seed=29, obs_dim=12)

    # (1) Extraction never calls np.load at all: the member is streamed
    # through zipfile, so the full observations array is never
    # materialized.
    _write_coverage_dataset(tmp_path / "extract", arrays)
    def _forbidden_load(*args, **kwargs):
        raise AssertionError("extraction called np.load")

    monkeypatch.setattr(np, "load", _forbidden_load)
    result = v2.extract_observations(tmp_path / "extract" / "replay.npz")
    monkeypatch.undo()
    assert result["action"] == "extracted"

    # (2) The loader opens the NPZ lazily and never indexes any
    # observation key (indexing would decompress ~73 GB at real scale),
    # and opens the raw .npy exactly once, memmapped read-only.
    manifest_path = _write_coverage_dataset(tmp_path / "load", arrays)
    real_load = np.load
    npy_calls: list[tuple[str, object]] = []

    class _ForbidObservationAccess:
        def __init__(self, archive) -> None:
            self._archive = archive

        @property
        def files(self):
            return self._archive.files

        def __getitem__(self, key):
            assert key not in v1._OBSERVATION_KEYS, (
                f"loader decompressed the observations member {key!r}"
            )
            return self._archive[key]

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self._archive.close()
            return False

    def _spy_load(path, *args, **kwargs):
        name = str(path)
        if name.endswith(".npz"):
            assert kwargs.get("mmap_mode") is None
            return _ForbidObservationAccess(
                real_load(path, *args, **kwargs)
            )
        npy_calls.append((name, kwargs.get("mmap_mode")))
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", _spy_load)
    dataset, receipt = v2.load_replay_dataset_memmap(manifest_path)
    monkeypatch.undo()

    expected_npy = str(
        (tmp_path / "load" / "replay.observations.npy").resolve()
    )
    assert npy_calls == [(expected_npy, "r")]
    assert isinstance(dataset.observations.base, np.memmap)
    assert np.array_equal(dataset.observations, arrays["observations"])
    assert receipt["observation_backing"]["mode"] == "memmap"


def test_main_delegates_to_v1_and_restores_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    arrays = _synthetic_arrays(seed=31, obs_dim=10)
    manifest_path = _write_coverage_dataset(tmp_path, arrays)
    original_loader = v1.load_replay_dataset

    # Extraction-only mode extracts, prints receipts, and exits 0.
    assert (
        v2.main(
            [
                "--extract-observations-only",
                "--dataset-manifest",
                str(manifest_path),
            ]
        )
        == 0
    )
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "EXTRACTED"
    assert printed["extractions"][0]["action"] == "extracted"
    assert (tmp_path / "replay.observations.npy").exists()

    # --validate-only flows through v1.main with the memmap loader; the
    # printed dataset receipt carries the observation backing, the
    # already-extracted .npy is reused, and v1 is restored afterwards.
    assert (
        v2.main(
            ["--dataset-manifest", str(manifest_path), "--validate-only"]
        )
        == 0
    )
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "VALID"
    backing = printed["dataset"]["observation_backing"]
    assert backing["mode"] == "memmap"
    assert backing["extraction"]["action"] == "reused"
    assert v1.load_replay_dataset is original_loader
