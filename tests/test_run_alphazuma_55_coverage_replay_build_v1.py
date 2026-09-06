from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tools import build_alphazuma_55_motor_observable_replay_v4 as builder
from tools import collect_alphazuma_55_park_settle_episodes_v1 as collector
from tools import distill_alphazuma_55 as legacy
from tools import distill_alphazuma_55_park_settle_v1 as park_settle
from tools import run_alphazuma_55_coverage_replay_build_v1 as runner


OBS_WIDTH = 16
RETAIL_ROOT = Path("/mnt/d/SteamLibrary/steamapps/common/Zuma's Revenge")


def _fill_recorder(
    recorder: collector._EpisodeRecorder,
    *,
    ticks: int,
    fire_edges: tuple[int, ...] = (),
    runtime_illegal_ticks: tuple[int, ...] = (),
    observation_seed: int = 0,
) -> collector._EpisodeRecorder:
    """Feed synthetic full-tick data through the real recorder path.

    Mirrors tests/test_collect_alphazuma_55_park_settle_episodes_v1 so
    the runner is exercised on byte-real collector NPZ output.
    """

    fire_set = {int(value) for value in fire_edges}
    illegal_set = {int(value) for value in runtime_illegal_ticks}
    rng = np.random.default_rng(observation_seed)
    for tick_index in range(ticks):
        verb = collector.FIRE_VERB_INDEX if tick_index in fire_set else 0
        aim = tick_index % collector.AIM_BINS
        exact = np.ones(collector.MASK_WIDTH, dtype=np.bool_)
        relaxed = tick_index in illegal_set
        if relaxed:
            exact[verb] = False
        training = exact.copy()
        training[verb] = True
        observation = rng.random(OBS_WIDTH, dtype=np.float32)
        observation[0] = np.float32(tick_index)
        recorder.record_intent(
            observation=observation,
            raw_action=np.asarray([verb, aim], dtype=np.int64),
            effective_action=np.asarray(
                [0 if relaxed else verb, aim], dtype=np.int64
            ),
            training_mask=training,
            exact_mask=exact,
            relaxed=relaxed,
        )
        recorder.record_info(
            {
                "score": tick_index * 10,
                "chain_length": 5 + tick_index,
                "visible_balls": 3,
                "human_speedrun": {
                    "desired_verb": verb,
                    "desired_aim_bin": aim,
                    "executed_verb": verb,
                    "executed_aim_bin": (aim + 1) % collector.AIM_BINS,
                    "button_enqueued": verb == collector.FIRE_VERB_INDEX,
                },
            }
        )
    return recorder


def _synthetic_episode(
    run_dir: Path,
    task: collector.EpisodeTask,
    *,
    ticks: int = 24,
    fire_edges: tuple[int, ...] = (6,),
    runtime_illegal_ticks: tuple[int, ...] = (),
) -> dict[str, Any]:
    episodes_dir = run_dir / collector.EPISODES_DIRNAME
    spill_dir = run_dir / collector.SPILL_DIRNAME
    episodes_dir.mkdir(parents=True, exist_ok=True)
    spill_dir.mkdir(parents=True, exist_ok=True)
    recorder = _fill_recorder(
        collector._EpisodeRecorder(
            spill_path=spill_dir / f"{task.file_name}.spill",
            observation_width=OBS_WIDTH,
        ),
        ticks=ticks,
        fire_edges=fire_edges,
        runtime_illegal_ticks=runtime_illegal_ticks,
        observation_seed=int(task.seed),
    )
    return collector._finalize_episode(
        task=task,
        recorder=recorder,
        final_info={"outcome": "win", "score": (ticks - 1) * 10},
        guard_exhausted=False,
        episodes_dir=episodes_dir,
        wall_seconds=0.25,
    )


def _write_synthetic_run(
    run_dir: Path,
    *,
    levels: tuple[str, ...] = ("Jungle1", "volcano10"),
    seeds_per_level: int = 2,
    ticks: int = 24,
    runtime_illegal_ticks: tuple[int, ...] = (),
) -> tuple[Path, list[dict[str, Any]]]:
    tasks = collector.plan_tasks(
        levels=levels,
        seeds_per_level=seeds_per_level,
        seed_base=collector.DEFAULT_SEED_BASE,
    )
    entries = [
        _synthetic_episode(
            run_dir,
            task,
            ticks=ticks,
            runtime_illegal_ticks=runtime_illegal_ticks,
        )
        for task in tasks
    ]
    manifest_path = collector._write_manifest(
        run_dir,
        levels=collector._validate_levels(levels),
        seeds_per_level=seeds_per_level,
        seed_base=collector.DEFAULT_SEED_BASE,
        max_ticks=100,
        entries_by_key={
            (entry["level_id"], entry["seed"]): entry for entry in entries
        },
    )
    # The manifest writer orders level-major; re-read to keep the test's
    # entry order identical to what the runner will consume.
    ordered = legacy._read_json(manifest_path)["episodes"]
    return manifest_path, ordered


def _runner_argv(
    run_dir: Path,
    prefix: Path,
    *,
    ratio: float,
    budget: int,
    danger_threshold: float | None = None,
    teacher_policy_id: str | None = None,
) -> list[str]:
    argv = [
        "--collection-run-dir",
        str(run_dir),
        "--output-prefix",
        str(prefix),
        "--target-coverage-ratio",
        str(ratio),
        "--total-budget",
        str(budget),
    ]
    if danger_threshold is not None:
        argv += ["--danger-threshold", str(danger_threshold)]
    if teacher_policy_id is not None:
        argv += ["--teacher-policy-id", teacher_policy_id]
    return argv


def _builder_main_reference(
    manifest_path: Path,
    prefix: Path,
    *,
    ratio: float,
    budget: int,
    danger_threshold: float | None = None,
) -> None:
    """Run the frozen builder.main() on the same collection via its
    inline-JSON hydration path, as the ground-truth reference."""

    episodes_json = prefix.parent / "builder-episodes.json"
    collector.emit_builder_episodes_json(
        manifest_path=manifest_path, output_path=episodes_json
    )
    argv = [
        "--episodes",
        str(episodes_json),
        "--output-prefix",
        str(prefix),
        "--target-coverage-ratio",
        str(ratio),
        "--total-budget",
        str(budget),
        "--teacher-policy-id",
        collector.TEACHER_POLICY_ID,
    ]
    if danger_threshold is not None:
        argv += ["--danger-threshold", str(danger_threshold)]
    assert builder.main(argv) == 0


def _assert_outputs_match_builder(out_prefix: Path, ref_prefix: Path) -> None:
    """Runner outputs must equal builder.main() outputs in everything
    but wall time and NPZ zip-container bytes (member timestamps)."""

    out_manifest = legacy._read_json(
        out_prefix.parent / (out_prefix.name + ".manifest.json")
    )
    ref_manifest = legacy._read_json(
        ref_prefix.parent / (ref_prefix.name + ".manifest.json")
    )
    assert out_manifest.pop("wall_seconds") >= 0.0
    assert ref_manifest.pop("wall_seconds") >= 0.0
    assert out_manifest == ref_manifest

    out_dataset = legacy._read_json(
        out_prefix.parent / (out_prefix.name + ".dataset.json")
    )
    ref_dataset = legacy._read_json(
        ref_prefix.parent / (ref_prefix.name + ".dataset.json")
    )
    out_shard = out_dataset.pop("shards")[0]
    ref_shard = ref_dataset.pop("shards")[0]
    assert out_dataset == ref_dataset
    assert out_shard["path"] == ref_shard["path"]
    assert out_shard["rows"] == ref_shard["rows"]

    with np.load(
        out_prefix.parent / (out_prefix.name + ".npz")
    ) as out_npz, np.load(
        ref_prefix.parent / (ref_prefix.name + ".npz")
    ) as ref_npz:
        assert sorted(out_npz.files) == sorted(ref_npz.files)
        for key in ref_npz.files:
            assert np.array_equal(out_npz[key], ref_npz[key]), key


def test_collection_schema_and_metadata_keys_match_the_collector() -> None:
    assert runner.COLLECTION_MANIFEST_SCHEMA == collector.MANIFEST_SCHEMA
    assert "observations" not in runner._METADATA_KEYS
    assert set(runner._METADATA_KEYS) < set(collector._EPISODE_ARRAY_KEYS)
    # Planning info covers every danger-priority field the collector NPZ
    # schema records, plus the score the builder reads.
    recordable = {"chain_length", "visible_balls"}
    priority = set(builder.DEFAULT_DANGER_FIELD_PRIORITY)
    assert recordable == priority & set(runner._PLANNING_INFO_FIELDS)
    assert "score" in runner._PLANNING_INFO_FIELDS


def test_sha256_tamper_refusal_lists_every_bad_file(tmp_path: Path) -> None:
    run_dir = tmp_path / "collection"
    _, entries = _write_synthetic_run(run_dir)
    tampered = run_dir / entries[0]["path"]
    tampered.write_bytes(tampered.read_bytes() + b"tampered")
    missing = run_dir / entries[1]["path"]
    missing.unlink()
    prefix = tmp_path / "out" / "replay"
    with pytest.raises(ValueError) as refusal:
        runner.main(_runner_argv(run_dir, prefix, ratio=1.0, budget=10_000))
    message = str(refusal.value)
    assert "2 episode file(s) failed" in message
    assert str(entries[0]["path"]) in message
    assert "sha256 mismatch" in message
    assert str(entries[1]["path"]) in message
    assert "missing on disk" in message
    # Refusal happens before any output is written.
    for suffix in (".npz", ".manifest.json", ".dataset.json",
                   ".completion.json"):
        assert not (prefix.parent / (prefix.name + suffix)).exists()


def test_refuses_foreign_schema_and_relaxed_mask_semantics(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "collection"
    manifest_path, _ = _write_synthetic_run(run_dir)
    manifest = legacy._read_json(manifest_path)

    foreign_dir = tmp_path / "foreign"
    foreign_dir.mkdir()
    legacy._write_json_atomic(
        foreign_dir / "episodes_manifest.json",
        {**manifest, "schema": "some-other-schema"},
    )
    with pytest.raises(ValueError, match="not a park-settle episodes"):
        runner.load_collection_manifest(
            foreign_dir / "episodes_manifest.json"
        )

    relaxed_dir = tmp_path / "relaxed"
    relaxed_dir.mkdir()
    legacy._write_json_atomic(
        relaxed_dir / "episodes_manifest.json",
        {
            **manifest,
            "mask_semantics": {
                "exact_masks": (
                    "relaxed_training_mask_teacher_verb_unmasked"
                )
            },
        },
    )
    with pytest.raises(ValueError, match="refusing to feed"):
        runner.load_collection_manifest(
            relaxed_dir / "episodes_manifest.json"
        )


def test_runner_matches_builder_main_and_park_settle_accepts(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "collection"
    manifest_path, entries = _write_synthetic_run(
        run_dir, ticks=40, runtime_illegal_ticks=(8, 9)
    )
    ref_prefix = tmp_path / "ref" / "replay"
    ref_prefix.parent.mkdir(parents=True)
    _builder_main_reference(
        manifest_path, ref_prefix, ratio=1.0, budget=1_000_000
    )
    out_prefix = tmp_path / "out" / "replay"
    # --teacher-policy-id is omitted: the runner must inherit the
    # collection manifest's id, matching the explicit reference run.
    assert (
        runner.main(
            _runner_argv(run_dir, out_prefix, ratio=1.0, budget=1_000_000)
        )
        == 0
    )
    _assert_outputs_match_builder(out_prefix, ref_prefix)

    # The trainer-facing dataset manifest is accepted end to end with
    # exact-mask semantics and nonzero fire windows.
    dataset, receipt = park_settle.load_replay_dataset(
        out_prefix.parent / (out_prefix.name + ".dataset.json")
    )
    assert dataset.temporal_structure == "episode_ticks"
    assert dataset.sample_count == 4 * 40
    assert len(np.unique(dataset.episode_indices)) == 4
    assert receipt["entries"][0]["keys"]["masks"] == "exact_masks"
    assert receipt["mask_semantics"]["exact_masks"] == (
        "exact_runtime_valid_action_mask"
    )
    annotations = park_settle.annotate_park_settle(
        dataset, park_settle.ParkSettleLossConfig()
    )
    assert int(np.sum(annotations.fire_edge)) > 0
    assert int(np.sum(annotations.fire_commit)) > 0
    assert int(np.sum(annotations.pre_fire_window)) > 0

    # Completion receipt: input receipts, config echo, coverage report.
    completion = legacy._read_json(
        out_prefix.parent / (out_prefix.name + ".completion.json")
    )
    assert completion["schema"] == runner.COMPLETION_SCHEMA
    assert completion["status"] == "BUILT"
    assert completion["wall_seconds"] >= 0.0
    assert completion["collection"]["episodes"] == 4
    assert completion["collection"]["episodes_sha256_verified"] == 4
    assert completion["collection"]["total_source_ticks"] == 160
    assert completion["collection"]["episodes_manifest"][
        "sha256"
    ] == legacy._sha256(manifest_path)
    echo = completion["builder_config"]
    assert echo["target_coverage_ratio"] == 1.0
    assert echo["total_budget"] == 1_000_000
    assert echo["teacher_policy_id"] == collector.TEACHER_POLICY_ID
    assert echo["teacher_policy_id_source"] == "collection_manifest"
    out_manifest = legacy._read_json(
        out_prefix.parent / (out_prefix.name + ".manifest.json")
    )
    assert completion["per_level_coverage"] == (
        out_manifest["per_level_coverage"]
    )
    assert completion["dataset_sha256"] == out_manifest["dataset_sha256"]
    assert completion["retained_samples"] == 160
    assert completion["formal_seed_consumption"] is False

    # Frozen-output discipline: the same prefix is never overwritten.
    with pytest.raises(FileExistsError):
        runner.main(
            _runner_argv(run_dir, out_prefix, ratio=1.0, budget=1_000_000)
        )


def test_budget_cap_and_uniform_sampling_match_builder_main(
    tmp_path: Path,
) -> None:
    """Budget rescaling + RNG uniform remainder must reproduce
    builder.main() exactly, and the mandatory tiers must survive."""

    run_dir = tmp_path / "collection"
    manifest_path, _ = _write_synthetic_run(
        run_dir, ticks=40, runtime_illegal_ticks=(8, 9)
    )
    ref_prefix = tmp_path / "ref" / "replay"
    ref_prefix.parent.mkdir(parents=True)
    _builder_main_reference(
        manifest_path, ref_prefix, ratio=0.6, budget=70,
        danger_threshold=0.95,
    )
    out_prefix = tmp_path / "out" / "replay"
    assert (
        runner.main(
            _runner_argv(
                run_dir, out_prefix, ratio=0.6, budget=70,
                danger_threshold=0.95,
            )
        )
        == 0
    )
    _assert_outputs_match_builder(out_prefix, ref_prefix)
    out_manifest = legacy._read_json(
        out_prefix.parent / (out_prefix.name + ".manifest.json")
    )
    equalization = out_manifest["coverage_equalization"]
    assert equalization["budget_cap_applied"] is True
    assert out_manifest["retained_samples"] == 70
    for level_report in out_manifest["per_level_coverage"].values():
        assert (
            level_report["retained_ticks"]
            >= level_report["mandatory_ticks"] > 0
        )
        assert level_report["fire_window_ticks"] > 0
        assert level_report["uniform_ticks"] > 0  # RNG path exercised
    dataset, _ = park_settle.load_replay_dataset(
        out_prefix.parent / (out_prefix.name + ".dataset.json")
    )
    assert dataset.sample_count == 70
    assert dataset.temporal_structure == "episode_ticks"


def test_two_pass_ram_safety_mechanism(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Planning never hydrates observations; gathering hydrates each
    episode exactly once, in plan order; the spill is cleaned up."""

    run_dir = tmp_path / "collection"
    manifest_path, entries = _write_synthetic_run(run_dir, ticks=24)

    # Unit level: the planning record carries the width-1 stub while
    # actions/masks/info stay byte-real.
    record = runner._metadata_episode_record(
        manifest_path.parent, entries[0]
    )
    assert all(
        tick["observation"].size == 1
        and tick["observation"] is runner._STUB_OBSERVATION
        for tick in record["ticks"]
    )
    with np.load(run_dir / entries[0]["path"]) as archive:
        assert archive["observations"].shape == (24, OBS_WIDTH)
        assert np.array_equal(
            np.stack([tick["raw_action"] for tick in record["ticks"]]),
            archive["raw_actions"],
        )
        assert np.array_equal(
            np.stack([tick["exact_mask"] for tick in record["ticks"]]),
            archive["exact_masks"],
        )
        assert [
            tick["info"]["chain_length"] for tick in record["ticks"]
        ] == archive["info_chain_length"].tolist()

    metadata_calls: list[str] = []
    observation_calls: list[str] = []
    real_metadata = runner._load_metadata_arrays
    real_observations = runner._load_episode_observations

    def spy_metadata(path: Path) -> dict[str, np.ndarray]:
        metadata_calls.append(Path(path).name)
        arrays = real_metadata(path)
        assert "observations" not in arrays
        return arrays

    def spy_observations(path: Path) -> np.ndarray:
        observation_calls.append(Path(path).name)
        return real_observations(path)

    monkeypatch.setattr(runner, "_load_metadata_arrays", spy_metadata)
    monkeypatch.setattr(
        runner, "_load_episode_observations", spy_observations
    )
    out_prefix = tmp_path / "out" / "replay"
    assert (
        runner.main(
            _runner_argv(run_dir, out_prefix, ratio=1.0, budget=1_000_000)
        )
        == 0
    )
    expected = [Path(str(entry["path"])).name for entry in entries]
    assert metadata_calls == expected  # once per episode, manifest order
    assert observation_calls == expected  # one hydration per episode
    # The observation spill is removed once the aggregate NPZ landed.
    leftovers = [
        path
        for path in out_prefix.parent.iterdir()
        if ".observations." in path.name
    ]
    assert leftovers == []
    completion = legacy._read_json(
        out_prefix.parent / (out_prefix.name + ".completion.json")
    )
    ram = completion["ram_safety"]
    assert ram["planning_pass"]["observations_resident"] is False
    assert ram["planning_pass"]["stub_observation_width"] == 1
    assert ram["gather_pass"]["episodes_hydrated"] == 4
    assert ram["gather_pass"]["max_single_episode_observation_bytes"] == (
        24 * OBS_WIDTH * 4
    )
    assert ram["gather_pass"]["retained_observation_bytes"] == (
        4 * 24 * OBS_WIDTH * 4
    )


def test_real_smoke_collector_runner_park_settle_chain(
    tmp_path: Path,
) -> None:
    """End-to-end on the real engine: collector main() -> runner main()
    -> park-settle load_replay_dataset with exact-mask semantics and
    nonzero fire windows (Jungle1, 1 seed, 300 ticks, 1 env, CPU)."""

    if not RETAIL_ROOT.exists():
        pytest.skip("retail extraction is unavailable")
    run_dir = tmp_path / "run"
    assert (
        collector.main(
            [
                "--original-root",
                str(RETAIL_ROOT),
                "--run-dir",
                str(run_dir),
                "--levels",
                "Jungle1",
                "--seeds-per-level",
                "1",
                "--parallel-envs",
                "1",
                "--max-ticks",
                "300",
            ]
        )
        == 0
    )
    completion = legacy._read_json(run_dir / "completion.json")
    assert completion["status"] == "COMPLETE"
    tick_count = int(completion["episodes"][0]["tick_count"])
    assert 0 < tick_count <= 300

    prefix = tmp_path / "out" / "replay"
    assert (
        runner.main(
            _runner_argv(run_dir, prefix, ratio=1.0, budget=1_000_000)
        )
        == 0
    )
    build_completion = legacy._read_json(
        prefix.parent / (prefix.name + ".completion.json")
    )
    assert build_completion["retained_samples"] == tick_count
    assert build_completion["builder_config"]["teacher_policy_id"] == (
        collector.TEACHER_POLICY_ID
    )
    assert (
        build_completion["per_level_coverage"]["Jungle1"]["fire_edges"] > 0
    )

    dataset, receipt = park_settle.load_replay_dataset(
        prefix.parent / (prefix.name + ".dataset.json")
    )
    assert dataset.temporal_structure == "episode_ticks"
    assert dataset.sample_count == tick_count
    assert receipt["entries"][0]["keys"]["masks"] == "exact_masks"
    assert receipt["mask_semantics"]["exact_masks"] == (
        "exact_runtime_valid_action_mask"
    )
    annotations = park_settle.annotate_park_settle(
        dataset, park_settle.ParkSettleLossConfig()
    )
    assert int(np.sum(annotations.fire_edge)) > 0
    assert int(np.sum(annotations.fire_commit)) > 0
    assert int(np.sum(annotations.pre_fire_window)) > 0
