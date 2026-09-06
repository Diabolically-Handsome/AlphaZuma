from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tools import build_alphazuma_55_motor_observable_replay_v4 as builder
from tools import collect_alphazuma_55_park_settle_episodes_v1 as collector
from tools import distill_alphazuma_55 as legacy
from tools import distill_alphazuma_55_park_settle_v1 as park_settle
from zuma_rl.alphazuma_55 import INCLUDED_LEVELS


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
    """Feed synthetic full-tick data through the real recorder path."""

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
    tmp_path: Path,
    task: collector.EpisodeTask,
    *,
    ticks: int = 24,
    fire_edges: tuple[int, ...] = (6,),
    runtime_illegal_ticks: tuple[int, ...] = (),
    final_info: dict[str, Any] | None = None,
    guard_exhausted: bool = False,
) -> dict[str, Any]:
    episodes_dir = tmp_path / collector.EPISODES_DIRNAME
    spill_dir = tmp_path / collector.SPILL_DIRNAME
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
    if final_info is None:
        final_info = {"outcome": "win", "score": (ticks - 1) * 10}
    return collector._finalize_episode(
        task=task,
        recorder=recorder,
        final_info=final_info,
        guard_exhausted=guard_exhausted,
        episodes_dir=episodes_dir,
        wall_seconds=0.25,
    )


def _write_synthetic_run(
    tmp_path: Path,
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
            tmp_path,
            task,
            ticks=ticks,
            runtime_illegal_ticks=runtime_illegal_ticks,
        )
        for task in tasks
    ]
    manifest_path = collector._write_manifest(
        tmp_path,
        levels=collector._validate_levels(levels),
        seeds_per_level=seeds_per_level,
        seed_base=collector.DEFAULT_SEED_BASE,
        max_ticks=100,
        entries_by_key={
            (entry["level_id"], entry["seed"]): entry for entry in entries
        },
    )
    return manifest_path, entries


def test_seed_plan_follows_replay_convention_and_avoids_embargo() -> None:
    tasks = collector.plan_tasks(
        levels=INCLUDED_LEVELS,
        seeds_per_level=3,
        seed_base=collector.DEFAULT_SEED_BASE,
    )
    assert len(tasks) == 3 * 55
    for task in tasks:
        assert task.seed == (
            collector.DEFAULT_SEED_BASE
            + task.pass_index * 55
            + task.level_position
        )
        for _, first, last in collector.EMBARGOED_SEED_RANGES:
            assert not first <= task.seed <= last
    collector.validate_seed_plan(tasks)
    # A base inside an embargo range must abort before any environment.
    with pytest.raises(ValueError, match="formal_selection embargo"):
        collector.validate_seed_plan(
            collector.plan_tasks(
                levels=INCLUDED_LEVELS,
                seeds_per_level=3,
                seed_base=1_600_000_000,
            )
        )
    # A plan that merely STRADDLES a range must abort too.
    with pytest.raises(ValueError, match="formal_final_blind embargo"):
        collector.validate_seed_plan(
            collector.plan_tasks(
                levels=INCLUDED_LEVELS,
                seeds_per_level=3,
                seed_base=1_699_999_900,
            )
        )


def test_levels_are_validated_and_reordered() -> None:
    ordered = collector._validate_levels(["volcano10", "Jungle1"])
    assert ordered == ("Jungle1", "volcano10")
    with pytest.raises(ValueError, match="frozen 55 scope"):
        collector._validate_levels(["Jungle1", "Jungle5"])
    with pytest.raises(ValueError, match="duplicate"):
        collector._validate_levels(["Jungle1", "Jungle1"])


def test_episode_npz_schema_and_sha256_receipt(tmp_path: Path) -> None:
    task = collector.plan_tasks(
        levels=("Jungle1",),
        seeds_per_level=1,
        seed_base=collector.DEFAULT_SEED_BASE,
    )[0]
    entry = _synthetic_episode(
        tmp_path,
        task,
        ticks=30,
        fire_edges=(5, 6, 20),
        runtime_illegal_ticks=(20, 25),
    )
    path = tmp_path / entry["path"]
    assert path.is_file()
    assert entry["sha256"] == legacy._sha256(path)
    assert entry["tick_count"] == 30
    # 6 is a held fire tick, not a rising edge.
    assert entry["fire_edges_raw_intent"] == 2
    assert entry["runtime_illegal_raw_verb_ticks"] == 2
    assert entry["intent_mask_relaxations"] == 2
    assert entry["raw_intent_action_counts"]["fire"] == 3
    assert entry["effective_execution_action_counts"]["wait"] == 28
    assert entry["outcome"] == "win"
    with np.load(path) as archive:
        assert archive["observations"].dtype == np.float32
        assert archive["observations"].shape == (30, OBS_WIDTH)
        assert archive["raw_actions"].dtype == np.int64
        assert archive["raw_actions"].shape == (30, 2)
        assert archive["training_masks"].dtype == np.bool_
        assert archive["exact_masks"].shape == (30, collector.MASK_WIDTH)
        assert np.array_equal(
            archive["tick_indices"], np.arange(30, dtype=np.int64)
        )
        # The exact mask stays UNRELAXED while training mask allows the verb.
        assert not bool(archive["exact_masks"][20][1])
        assert bool(archive["training_masks"][20][1])
        assert archive["info_chain_length"][7] == 12
        assert bool(archive["hs_button_enqueued"][5])
        # The spill file is gone once the NPZ landed.
    assert not list((tmp_path / collector.SPILL_DIRNAME).glob("*.spill"))


def test_truncation_loss_and_guard_outcomes_are_recorded(
    tmp_path: Path,
) -> None:
    tasks = collector.plan_tasks(
        levels=("Jungle1", "Jungle2", "Jungle3"),
        seeds_per_level=1,
        seed_base=collector.DEFAULT_SEED_BASE,
    )
    truncated = _synthetic_episode(
        tmp_path,
        tasks[0],
        final_info={"outcome": None, "TimeLimit.truncated": True, "score": 7},
    )
    assert truncated["outcome"] == "truncated"
    assert truncated["time_limit_truncated"] is True
    assert truncated["guard_exhausted"] is False
    assert truncated["score"] == 7
    lost = _synthetic_episode(
        tmp_path,
        tasks[1],
        final_info={"outcome": "loss", "score": 123},
    )
    assert lost["outcome"] == "loss"
    assert lost["time_limit_truncated"] is False
    guarded = _synthetic_episode(
        tmp_path, tasks[2], final_info={}, guard_exhausted=True
    )
    assert guarded["outcome"] == "truncated"
    assert guarded["guard_exhausted"] is True
    # Guard episodes keep their ticks and fall back to the recorded score.
    assert guarded["tick_count"] == 24
    assert guarded["score"] == 230


def test_manifest_hydrates_into_the_v4_builder_contract(
    tmp_path: Path,
) -> None:
    manifest_path, entries = _write_synthetic_run(
        tmp_path, ticks=40, runtime_illegal_ticks=(8, 9)
    )
    records = collector.load_episode_records(manifest_path)
    assert len(records) == len(entries) == 4
    rows, manifest, provenance = (
        builder.build_coverage_equalized_replay_dataset(
            records,
            config=builder.CoverageReplayV4Config(
                target_coverage_ratio=1.0, total_budget=1_000_000
            ),
            teacher_policy_id=collector.TEACHER_POLICY_ID,
        )
    )
    assert len(rows) == 4 * 40
    assert manifest["wins"] == 4
    assert manifest["teacher_policy_id"] == collector.TEACHER_POLICY_ID
    # The runtime-illegal raw verbs survive with the honest fraction.
    fraction = manifest["runtime_illegal_raw_verb_fraction"]
    assert fraction["source"] == pytest.approx(8 / 160)
    assert fraction["retained"] == pytest.approx(8 / 160)
    # The per-tick info carries the danger field the builder prioritizes.
    equalization = manifest["coverage_equalization"]
    assert equalization["danger_field_used"] == "chain_length"
    coverage = manifest["per_level_coverage"]
    assert set(coverage) == {"Jungle1", "volcano10"}
    assert all(entry["fire_edges"] > 0 for entry in coverage.values())
    assert provenance.exact_masks.shape == (160, collector.MASK_WIDTH)
    # Corrupted bytes are refused via the sha256 receipt.
    victim = tmp_path / entries[0]["path"]
    victim.write_bytes(victim.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="differ from the manifest receipt"):
        collector.load_episode_records(manifest_path)


def test_manifest_loads_directly_into_the_park_settle_trainer(
    tmp_path: Path,
) -> None:
    manifest_path, entries = _write_synthetic_run(tmp_path, ticks=32)
    dataset, receipt = park_settle.load_replay_dataset(manifest_path)
    assert dataset.temporal_structure == "episode_ticks"
    assert dataset.sample_count == 4 * 32
    assert len(np.unique(dataset.episode_indices)) == 4
    assert np.array_equal(
        dataset.tick_indices[:32], np.arange(32, dtype=np.int64)
    )
    assert receipt["entries"][0]["keys"]["masks"] == "exact_masks"
    assert receipt["mask_semantics"]["exact_masks"] == (
        "exact_runtime_valid_action_mask"
    )


def _fake_collect_batch_factory(
    record_log: list[tuple[str, int]], ticks: int = 12
):
    def fake_collect_batch(
        *,
        original_root: Path,
        tasks: list[collector.EpisodeTask],
        record_flags: list[bool],
        max_ticks: int,
        episodes_dir: Path,
        spill_dir: Path,
        on_episode,
    ) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        first_seed = int(tasks[0].seed)
        for offset, task in enumerate(tasks):
            assert int(task.seed) == first_seed + offset
        for task, record in zip(tasks, record_flags, strict=True):
            if not record:
                continue
            record_log.append(task.key)
            recorder = _fill_recorder(
                collector._EpisodeRecorder(
                    spill_path=spill_dir / f"{task.file_name}.spill",
                    observation_width=OBS_WIDTH,
                ),
                ticks=ticks,
                fire_edges=(3,),
                observation_seed=int(task.seed),
            )
            entry = collector._finalize_episode(
                task=task,
                recorder=recorder,
                final_info={"outcome": "win", "score": 999},
                guard_exhausted=False,
                episodes_dir=episodes_dir,
                wall_seconds=0.01,
            )
            entries.append(entry)
            on_episode(entry)
        return entries

    return fake_collect_batch


def test_resume_skips_completed_and_recollects_corrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    levels = ("Jungle1", "Jungle2", "Jungle3")
    kwargs = dict(
        original_root=tmp_path,
        run_dir=run_dir,
        levels=levels,
        seeds_per_level=2,
        seed_base=collector.DEFAULT_SEED_BASE,
        parallel_envs=2,
        max_ticks=64,
    )

    first_log: list[tuple[str, int]] = []
    monkeypatch.setattr(
        collector, "_collect_batch", _fake_collect_batch_factory(first_log)
    )
    completion = collector.collect(**kwargs)
    assert completion["status"] == "COMPLETE"
    assert len(first_log) == 6
    assert len(completion["episodes"]) == 6
    assert all(entry["reused"] is False for entry in completion["episodes"])
    assert completion["wins"] == 6
    assert (run_dir / "episodes_manifest.json").exists()
    assert (run_dir / "completion.json").exists()

    second_log: list[tuple[str, int]] = []
    monkeypatch.setattr(
        collector, "_collect_batch", _fake_collect_batch_factory(second_log)
    )
    completion = collector.collect(**kwargs)
    assert completion["status"] == "COMPLETE"
    assert second_log == []
    assert all(entry["reused"] is True for entry in completion["episodes"])

    # Corrupt one episode IN PLACE (no unlink): resume must self-heal by
    # quarantining the bad bytes and re-collecting only that (level,
    # seed) -- never wedging on write_npz's FileExistsError refusal.
    episodes_dir = run_dir / collector.EPISODES_DIRNAME
    victim = episodes_dir / "Jungle2-seed1543004001.npz"
    corrupted_bytes = b"corrupted-in-place"
    victim.write_bytes(corrupted_bytes)
    third_log: list[tuple[str, int]] = []
    monkeypatch.setattr(
        collector, "_collect_batch", _fake_collect_batch_factory(third_log)
    )
    completion = collector.collect(**kwargs)
    assert completion["status"] == "COMPLETE"
    assert completion["errors"] == []
    assert third_log == [("Jungle2", 1_543_004_001)]
    reused = {
        (entry["level_id"], entry["seed"]): entry["reused"]
        for entry in completion["episodes"]
    }
    assert reused[("Jungle2", 1_543_004_001)] is False
    assert sum(1 for value in reused.values() if value) == 5
    # The corrupted bytes were quarantined, not silently destroyed.
    quarantined = sorted(
        episodes_dir.glob("Jungle2-seed1543004001.npz.quarantined-*")
    )
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == corrupted_bytes
    assert completion["resume_quarantined_files"] == [
        {"original": victim.name, "quarantined_as": quarantined[0].name}
    ]
    # The re-collected NPZ matches its fresh manifest receipt again.
    manifest = legacy._read_json(run_dir / "episodes_manifest.json")
    healed = next(
        entry
        for entry in manifest["episodes"]
        if entry["level_id"] == "Jungle2" and entry["seed"] == 1_543_004_001
    )
    assert healed["sha256"] == legacy._sha256(victim)

    # An ORPHANED NPZ -- crash in the window between the NPZ os.replace
    # and the manifest rewrite: the file exists but no manifest entry
    # claims it.  Resume must quarantine it and re-collect, not wedge.
    orphan = episodes_dir / "Jungle3-seed1543004002.npz"
    orphan_bytes = orphan.read_bytes()
    manifest = legacy._read_json(run_dir / "episodes_manifest.json")
    manifest["episodes"] = [
        entry
        for entry in manifest["episodes"]
        if not (
            entry["level_id"] == "Jungle3"
            and entry["seed"] == 1_543_004_002
        )
    ]
    legacy._write_json_atomic(run_dir / "episodes_manifest.json", manifest)
    fourth_log: list[tuple[str, int]] = []
    monkeypatch.setattr(
        collector, "_collect_batch", _fake_collect_batch_factory(fourth_log)
    )
    completion = collector.collect(**kwargs)
    assert completion["status"] == "COMPLETE"
    assert completion["errors"] == []
    assert fourth_log == [("Jungle3", 1_543_004_002)]
    assert orphan.is_file()  # re-collected fresh at the same path
    orphan_quarantine = sorted(
        episodes_dir.glob("Jungle3-seed1543004002.npz.quarantined-*")
    )
    assert len(orphan_quarantine) == 1
    assert orphan_quarantine[0].read_bytes() == orphan_bytes
    # Earlier quarantined files are left untouched, not re-quarantined.
    assert completion["resume_quarantined_files"] == [
        {"original": orphan.name, "quarantined_as": orphan_quarantine[0].name}
    ]

    # A MISSING file must equally trigger recollection.
    victim.unlink()
    fifth_log: list[tuple[str, int]] = []
    monkeypatch.setattr(
        collector, "_collect_batch", _fake_collect_batch_factory(fifth_log)
    )
    completion = collector.collect(**kwargs)
    assert completion["status"] == "COMPLETE"
    assert fifth_log == [("Jungle2", 1_543_004_001)]
    assert completion["resume_quarantined_files"] == []

    # A different plan must refuse to reuse the run dir.
    with pytest.raises(ValueError, match="different collection contract"):
        collector.collect(**{**kwargs, "seed_base": 1_543_005_000})


def test_emit_builder_episodes_json_round_trips_bit_exact(
    tmp_path: Path,
) -> None:
    manifest_path, entries = _write_synthetic_run(
        tmp_path,
        levels=("Jungle1",),
        seeds_per_level=1,
        ticks=20,
        runtime_illegal_ticks=(4,),
    )
    output = tmp_path / "builder-episodes.json"
    receipt = collector.emit_builder_episodes_json(
        manifest_path=manifest_path, output_path=output
    )
    assert receipt["episodes"] == 1
    assert receipt["ticks"] == 20
    assert receipt["sha256"] == legacy._sha256(output)
    with pytest.raises(FileExistsError):
        collector.emit_builder_episodes_json(
            manifest_path=manifest_path, output_path=output
        )
    records = builder._load_episode_records(output)
    assert len(records) == 1
    record = records[0]
    assert record["level_id"] == "Jungle1"
    assert record["outcome"] == "win"
    assert record["ticks"][4]["intent_mask_relaxed"] is True
    assert record["ticks"][4]["info"]["human_speedrun"]["executed_verb"] == 0
    # Observation floats survive the JSON round trip bit-exactly.
    with np.load(tmp_path / entries[0]["path"]) as archive:
        stored = np.asarray(archive["observations"], dtype=np.float32)
    parsed = np.asarray(
        [tick["observation"] for tick in record["ticks"]], dtype=np.float32
    )
    assert np.array_equal(stored, parsed)
    rows, manifest = builder.build_coverage_equalized_replay(
        records,
        config=builder.CoverageReplayV4Config(
            target_coverage_ratio=1.0, total_budget=1_000_000
        ),
    )
    assert len(rows) == 20
    assert manifest["runtime_illegal_raw_verb_fraction"][
        "source"
    ] == pytest.approx(1 / 20)


def test_real_smoke_collector_builder_trainer_chain(tmp_path: Path) -> None:
    """End-to-end on the real engine: collector main() -> v4 builder
    main() -> park-settle load_replay_dataset, exact-mask semantics and
    nonzero fire windows all the way through."""

    if not RETAIL_ROOT.exists():
        pytest.skip("retail extraction is unavailable")
    run_dir = tmp_path / "run"
    episodes_json = tmp_path / "builder-episodes.json"
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
                "400",
                "--emit-builder-episodes",
                str(episodes_json),
            ]
        )
        == 0
    )
    completion = json.loads(
        (run_dir / "completion.json").read_text(encoding="utf-8")
    )
    assert completion["status"] == "COMPLETE"
    assert completion["formal_seed_consumption"] is False
    assert len(completion["episodes"]) == 1
    entry = completion["episodes"][0]
    assert entry["level_id"] == "Jungle1"
    assert entry["seed"] == collector.DEFAULT_SEED_BASE
    assert entry["outcome"] in {"win", "loss", "truncated"}
    if entry["outcome"] == "truncated":
        assert entry["time_limit_truncated"] is True
    assert 0 < entry["tick_count"] <= 400
    assert entry["fire_edges_raw_intent"] > 0
    assert entry["sha256"] == legacy._sha256(run_dir / entry["path"])

    prefix = tmp_path / "replay"
    assert (
        builder.main(
            [
                "--episodes",
                str(episodes_json),
                "--output-prefix",
                str(prefix),
                "--target-coverage-ratio",
                "1.0",
                "--total-budget",
                "1000000",
                "--teacher-policy-id",
                collector.TEACHER_POLICY_ID,
            ]
        )
        == 0
    )
    replay_manifest = json.loads(
        (tmp_path / "replay.manifest.json").read_text(encoding="utf-8")
    )
    assert (
        replay_manifest["per_level_coverage"]["Jungle1"]["fire_edges"] > 0
    )

    dataset, receipt = park_settle.load_replay_dataset(
        tmp_path / "replay.dataset.json"
    )
    assert dataset.temporal_structure == "episode_ticks"
    assert dataset.sample_count == entry["tick_count"]
    assert receipt["entries"][0]["keys"]["masks"] == "exact_masks"
    assert receipt["mask_semantics"]["exact_masks"] == (
        "exact_runtime_valid_action_mask"
    )
    annotations = park_settle.annotate_park_settle(
        dataset, park_settle.ParkSettleLossConfig()
    )
    # Nonzero fire windows with exact-mask legality accounting.
    assert int(np.sum(annotations.fire_edge)) > 0
    assert int(np.sum(annotations.fire_commit)) > 0
    assert int(np.sum(annotations.pre_fire_window)) > 0

    # The collector's own manifest is ALSO directly park-settle loadable,
    # and byte-identical to what the builder chain aggregated.
    direct, direct_receipt = park_settle.load_replay_dataset(
        run_dir / "episodes_manifest.json"
    )
    assert direct.temporal_structure == "episode_ticks"
    assert direct_receipt["entries"][0]["keys"]["masks"] == "exact_masks"
    assert np.array_equal(direct.observations, dataset.observations)
    assert np.array_equal(direct.raw_actions, dataset.raw_actions)
    assert np.array_equal(direct.masks, dataset.masks)
