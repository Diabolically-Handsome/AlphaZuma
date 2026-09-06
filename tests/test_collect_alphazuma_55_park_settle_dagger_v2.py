from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pytest

from tools import build_alphazuma_55_motor_observable_replay_v4 as builder
from tools import collect_alphazuma_55_park_settle_dagger_v2 as dagger2
from tools import collect_alphazuma_55_park_settle_episodes_v1 as v1
from tools import distill_alphazuma_55 as legacy
from tools import distill_alphazuma_55_park_settle_v1 as park_settle
from tools import merge_alphazuma_55_episode_manifests_v1 as merge_tool
from tools import run_alphazuma_55_coverage_replay_build_v1 as runner
from zuma_rl.alphazuma_55 import INCLUDED_LEVELS


OBS_WIDTH = 16
RETAIL_ROOT = Path("/mnt/d/SteamLibrary/steamapps/common/Zuma's Revenge")


# ---------------------------------------------------------------------------
# Synthetic helpers
# ---------------------------------------------------------------------------


def _record_tick(
    recorder: Any,
    *,
    tick: int,
    raw: tuple[int, int],
    executed: tuple[int, int] | None = None,
) -> None:
    """Feed one fully-legal synthetic tick through the real recorder path."""

    exact = np.ones(v1.MASK_WIDTH, dtype=np.bool_)
    training = exact.copy()
    observation = np.zeros(OBS_WIDTH, dtype=np.float32)
    observation[0] = np.float32(tick)
    kwargs: dict[str, Any] = dict(
        observation=observation,
        raw_action=np.asarray(raw, dtype=np.int64),
        effective_action=np.asarray(raw, dtype=np.int64),
        training_mask=training,
        exact_mask=exact,
        relaxed=False,
    )
    if executed is not None:
        kwargs["executed_action"] = np.asarray(executed, dtype=np.int64)
    recorder.record_intent(**kwargs)
    recorder.record_info(
        {
            "score": tick * 10,
            "chain_length": 3 + tick,
            "visible_balls": 2,
            "human_speedrun": {
                "desired_verb": int(raw[0]),
                "desired_aim_bin": int(raw[1]),
                "executed_verb": int(raw[0]),
                "executed_aim_bin": int(raw[1]),
                "button_enqueued": int(raw[0]) == v1.FIRE_VERB_INDEX,
            },
        }
    )


def _raw_for_tick(tick: int) -> tuple[int, int]:
    """Teacher stream with fire edges at ticks 5 and 15 (held one tick)."""

    verb = v1.FIRE_VERB_INDEX if tick in (5, 15) else 0
    return (verb, 40 + (tick % 3))


def _write_teacher_run(
    run_dir: Path,
    *,
    levels: tuple[str, ...] = ("Jungle1", "Jungle2"),
    seeds_per_level: int = 1,
    ticks: int = 30,
) -> Path:
    """A tiny v1 teacher-driven collection built via v1's own machinery."""

    tasks = v1.plan_tasks(
        levels=levels,
        seeds_per_level=seeds_per_level,
        seed_base=v1.DEFAULT_SEED_BASE,
    )
    episodes_dir = run_dir / v1.EPISODES_DIRNAME
    spill_dir = run_dir / v1.SPILL_DIRNAME
    episodes_dir.mkdir(parents=True)
    spill_dir.mkdir(parents=True)
    entries: dict[tuple[str, int], dict[str, Any]] = {}
    for task in tasks:
        recorder = v1._EpisodeRecorder(
            spill_path=spill_dir / f"{task.file_name}.spill",
            observation_width=OBS_WIDTH,
        )
        for tick in range(ticks):
            _record_tick(recorder, tick=tick, raw=_raw_for_tick(tick))
        entry = v1._finalize_episode(
            task=task,
            recorder=recorder,
            final_info={"outcome": "win", "score": (ticks - 1) * 10},
            guard_exhausted=False,
            episodes_dir=episodes_dir,
            wall_seconds=0.1,
        )
        entries[task.key] = entry
    v1._write_manifest(
        run_dir,
        levels=v1._validate_levels(levels),
        seeds_per_level=seeds_per_level,
        seed_base=v1.DEFAULT_SEED_BASE,
        max_ticks=100,
        entries_by_key=entries,
    )
    return run_dir


def _fill_dagger_recorder(
    recorder: dagger2.DaggerEpisodeRecorder,
    *,
    ticks: int,
    teacher: tuple[int, int] = (1, 42),
    student: tuple[int, int] = (0, 7),
) -> dagger2.DaggerEpisodeRecorder:
    for tick in range(ticks):
        _record_tick(recorder, tick=tick, raw=teacher, executed=student)
    return recorder


def _fake_dagger_batch_factory(
    record_log: list[tuple[str, int]], ticks: int = 10
):
    """Synthetic stand-in with _collect_batch_dagger's exact signature."""

    def fake(
        *,
        original_root: Path,
        tasks: list[v1.EpisodeTask],
        record_flags: list[bool],
        max_ticks: int,
        episodes_dir: Path,
        spill_dir: Path,
        on_episode: Any,
        policy_path: Path,
        device: str,
        model_cache: dict[Any, Any],
        **_ignored: Any,
    ) -> list[dict[str, Any]]:
        first_seed = int(tasks[0].seed)
        for offset, task in enumerate(tasks):
            assert int(task.seed) == first_seed + offset
        entries: list[dict[str, Any]] = []
        for task, record in zip(tasks, record_flags, strict=True):
            if not record:
                continue
            record_log.append(task.key)
            recorder = _fill_dagger_recorder(
                dagger2.DaggerEpisodeRecorder(
                    spill_path=spill_dir / f"{task.file_name}.spill",
                    observation_width=OBS_WIDTH,
                ),
                ticks=ticks,
            )
            entry = dagger2._finalize_dagger_episode(
                task=task,
                recorder=recorder,
                final_info={"outcome": "loss", "score": 55},
                guard_exhausted=False,
                episodes_dir=episodes_dir,
                wall_seconds=0.01,
            )
            entries.append(entry)
            on_episode(entry)
        return entries

    return fake


# ---------------------------------------------------------------------------
# Stubs for the injectable-seam batch loop (no engine, CPU-only)
# ---------------------------------------------------------------------------


class _StubVector:
    """Deterministic vector env recording exactly what was executed."""

    def __init__(
        self, *, count: int, obs_width: int, episode_ticks: tuple[int, ...]
    ) -> None:
        self.count = count
        self.obs_width = obs_width
        self.episode_ticks = list(episode_ticks)
        self.stepped_actions: list[np.ndarray] = []
        self.tick = 0
        self.seeded_with: int | None = None
        self.closed = False

    def seed(self, value: int) -> None:
        self.seeded_with = int(value)

    def _observations(self) -> np.ndarray:
        observations = np.zeros(
            (self.count, self.obs_width), dtype=np.float32
        )
        observations[:, 0] = np.float32(self.tick)
        for index in range(self.count):
            observations[index, 1] = np.float32(index)
        return observations

    def reset(self) -> np.ndarray:
        self.tick = 0
        return self._observations()

    def step(self, actions: np.ndarray):
        self.stepped_actions.append(
            np.asarray(actions, dtype=np.int64).copy()
        )
        self.tick += 1
        dones = np.asarray(
            [self.tick >= ticks for ticks in self.episode_ticks],
            dtype=np.bool_,
        )
        infos = []
        for index in range(self.count):
            info: dict[str, Any] = {
                "score": self.tick * 10,
                "chain_length": 4,
                "visible_balls": 3,
                "human_speedrun": {
                    "desired_verb": 0,
                    "desired_aim_bin": 0,
                    "executed_verb": 0,
                    "executed_aim_bin": 0,
                    "button_enqueued": False,
                },
            }
            if bool(dones[index]):
                info["outcome"] = "loss"
            infos.append(info)
        return (
            self._observations(),
            np.zeros(self.count, dtype=np.float32),
            dones,
            infos,
        )

    def close(self) -> None:
        self.closed = True


class _StubStudent:
    def __init__(self, action: tuple[int, int]) -> None:
        self.action = np.asarray(action, dtype=np.int64)

    def predict(
        self,
        observations: np.ndarray,
        deterministic: bool = False,
        action_masks: np.ndarray | None = None,
    ):
        assert deterministic is True
        assert action_masks is not None
        count = int(np.asarray(observations).shape[0])
        return np.tile(self.action, (count, 1)), None


class _StubTeacher:
    def __init__(self, action: tuple[int, int]) -> None:
        self.action = np.asarray(action, dtype=np.int64)

    def act(self, observation: np.ndarray) -> np.ndarray:
        return self.action.copy()


def _run_stub_batch(
    tmp_path: Path,
    *,
    episode_ticks: tuple[int, ...],
    max_ticks: int,
    student: tuple[int, int] = (0, 7),
    teachers: tuple[tuple[int, int], ...] = ((1, 42), (2, 99)),
) -> tuple[list[dict[str, Any]], _StubVector, Path]:
    levels = ("Jungle1", "Jungle2")[: len(episode_ticks)]
    tasks = v1.plan_tasks(
        levels=levels,
        seeds_per_level=1,
        seed_base=dagger2.DEFAULT_SEED_BASE,
    )
    episodes_dir = tmp_path / v1.EPISODES_DIRNAME
    spill_dir = tmp_path / v1.SPILL_DIRNAME
    episodes_dir.mkdir(exist_ok=True)
    spill_dir.mkdir(exist_ok=True)
    count = len(tasks)
    vector = _StubVector(
        count=count, obs_width=OBS_WIDTH, episode_ticks=episode_ticks
    )
    recorded: list[dict[str, Any]] = []
    entries = dagger2._collect_batch_dagger(
        original_root=tmp_path,
        tasks=tasks,
        record_flags=[True] * count,
        max_ticks=max_ticks,
        episodes_dir=episodes_dir,
        spill_dir=spill_dir,
        on_episode=recorded.append,
        policy_path=tmp_path / "never-loaded.zip",
        device="cpu",
        model_cache={},
        model=_StubStudent(student),
        vector_factory=lambda **kwargs: vector,
        teachers_factory=lambda _vector: [
            _StubTeacher(action) for action in teachers[:count]
        ],
        masks_provider=lambda _vector: np.ones(
            (count, v1.MASK_WIDTH), dtype=np.bool_
        ),
    )
    assert len(recorded) == len(entries)
    return entries, vector, episodes_dir


# ---------------------------------------------------------------------------
# (a) student executes / teacher labels separation
# ---------------------------------------------------------------------------


def test_student_executes_while_teacher_labels(tmp_path: Path) -> None:
    entries, vector, _ = _run_stub_batch(
        tmp_path, episode_ticks=(5, 8), max_ticks=20
    )
    assert vector.seeded_with == dagger2.DEFAULT_SEED_BASE
    assert vector.closed is True
    by_level = {entry["level_id"]: entry for entry in entries}
    assert set(by_level) == {"Jungle1", "Jungle2"}

    first = by_level["Jungle1"]
    assert first["tick_count"] == 5
    # Honest outcome: the student-driven episode lost, and that is kept.
    assert first["outcome"] == "loss"
    assert first["collection_mode"] == dagger2.COLLECTION_MODE
    assert first["executed_action_source"] == "student_policy"
    assert first["label_executed_verb_disagreement_ticks"] == 5
    assert first["label_executed_action_disagreement_ticks"] == 5
    assert first["executed_action_counts"] == {"wait": 5}
    # The teacher label stream drives fire-window semantics: verb 1 held
    # for the whole episode is exactly one rising edge.
    assert first["fire_edges_raw_intent"] == 1
    assert first["fire_edges_executed_student"] == 0
    with np.load(tmp_path / first["path"]) as archive:
        labels = archive["raw_actions"]
        executed = archive["executed_actions"]
        assert labels.shape == executed.shape == (5, 2)
        # LABEL = raw teacher intent; EXECUTED = student action.
        assert np.all(labels == np.asarray([1, 42], dtype=np.int64))
        assert np.all(executed == np.asarray([0, 7], dtype=np.int64))
        assert np.any(labels != executed)
        # Fully-legal mask: the teacher effective action equals its raw.
        assert np.array_equal(archive["effective_actions"], labels)

    second = by_level["Jungle2"]
    assert second["tick_count"] == 8
    with np.load(tmp_path / second["path"]) as archive:
        assert np.all(
            archive["raw_actions"] == np.asarray([2, 99], dtype=np.int64)
        )

    # THE ENVIRONMENT EXECUTED THE STUDENT'S ACTIONS on every tick, for
    # every env -- never the teacher's proposals.
    stepped = np.stack(vector.stepped_actions)
    assert np.all(stepped == np.asarray([0, 7], dtype=np.int64))


def test_guard_exhaustion_records_truncated_student_episode(
    tmp_path: Path,
) -> None:
    entries, _, _ = _run_stub_batch(
        tmp_path, episode_ticks=(100, 100), max_ticks=4
    )
    assert len(entries) == 2
    for entry in entries:
        assert entry["outcome"] == "truncated"
        assert entry["guard_exhausted"] is True
        assert entry["tick_count"] == 5  # max_ticks + 1 loop, v1 semantics
        assert entry["label_executed_verb_disagreement_ticks"] == 5


# ---------------------------------------------------------------------------
# (b) NPZ schema identical to v1 plus additive executed_actions
# ---------------------------------------------------------------------------


def test_npz_schema_is_v1_plus_additive_executed_actions(
    tmp_path: Path,
) -> None:
    task_v1, task_dagger = v1.plan_tasks(
        levels=("Jungle1", "Jungle2"),
        seeds_per_level=1,
        seed_base=dagger2.DEFAULT_SEED_BASE,
    )
    episodes_dir = tmp_path / v1.EPISODES_DIRNAME
    spill_dir = tmp_path / v1.SPILL_DIRNAME
    episodes_dir.mkdir()
    spill_dir.mkdir()
    ticks = 12

    reference = v1._EpisodeRecorder(
        spill_path=spill_dir / "reference.spill",
        observation_width=OBS_WIDTH,
    )
    for tick in range(ticks):
        _record_tick(reference, tick=tick, raw=_raw_for_tick(tick))
    entry_v1 = v1._finalize_episode(
        task=task_v1,
        recorder=reference,
        final_info={"outcome": "win", "score": 1},
        guard_exhausted=False,
        episodes_dir=episodes_dir,
        wall_seconds=0.1,
    )

    dagger_recorder = dagger2.DaggerEpisodeRecorder(
        spill_path=spill_dir / "dagger.spill",
        observation_width=OBS_WIDTH,
    )
    for tick in range(ticks):
        _record_tick(
            dagger_recorder,
            tick=tick,
            raw=_raw_for_tick(tick),
            executed=(0, 7),
        )
    entry_dagger = dagger2._finalize_dagger_episode(
        task=task_dagger,
        recorder=dagger_recorder,
        final_info={"outcome": "loss", "score": 1},
        guard_exhausted=False,
        episodes_dir=episodes_dir,
        wall_seconds=0.1,
    )

    with np.load(tmp_path / entry_v1["path"]) as archive:
        v1_files = set(archive.files)
        v1_arrays = {key: archive[key] for key in archive.files}
    with np.load(tmp_path / entry_dagger["path"]) as archive:
        dagger_files = set(archive.files)
        dagger_arrays = {key: archive[key] for key in archive.files}

    # Schema: EXACTLY v1's members plus the one additive key.
    assert dagger_files == v1_files | {dagger2.EXECUTED_ACTIONS_KEY}
    assert dagger_arrays[dagger2.EXECUTED_ACTIONS_KEY].dtype == np.int64
    assert dagger_arrays[dagger2.EXECUTED_ACTIONS_KEY].shape == (ticks, 2)
    # Identical synthetic streams: every shared member is byte-identical.
    for key in v1_files:
        assert np.array_equal(v1_arrays[key], dagger_arrays[key]), key

    # The park-settle trainer's key precedence still selects the LABEL.
    with np.load(tmp_path / entry_dagger["path"]) as archive:
        assert (
            park_settle._first_key(
                archive, park_settle._ACTION_KEYS, required=True
            )
            == "raw_actions"
        )

    # v1 hydration and the coverage runner's metadata pass both accept
    # the DAgger NPZ unchanged.
    entries_by_key = {
        task_v1.key: entry_v1,
        task_dagger.key: entry_dagger,
    }
    manifest_path = v1._write_manifest(
        tmp_path,
        levels=("Jungle1", "Jungle2"),
        seeds_per_level=1,
        seed_base=dagger2.DEFAULT_SEED_BASE,
        max_ticks=100,
        entries_by_key=entries_by_key,
    )
    records = v1.load_episode_records(manifest_path)
    assert len(records) == 2
    metadata = runner._load_metadata_arrays(tmp_path / entry_dagger["path"])
    assert int(metadata["raw_actions"].shape[0]) == ticks

    # The additive stream never leaks into the recorder unless provided:
    # the v1 recorder refuses nothing, but the DAgger recorder refuses a
    # misaligned executed stream.
    broken = dagger2.DaggerEpisodeRecorder(
        spill_path=spill_dir / "broken.spill",
        observation_width=OBS_WIDTH,
    )
    _record_tick(broken, tick=0, raw=(0, 1), executed=(0, 2))
    broken.executed_actions.append(np.asarray([0, 3], dtype=np.int64))
    with pytest.raises(RuntimeError, match="misaligned"):
        broken.write_npz(episodes_dir / "broken.npz")
    broken.discard()


# ---------------------------------------------------------------------------
# (c) merge utility -> coverage runner end-to-end
# ---------------------------------------------------------------------------


def test_merge_manifests_feeds_the_coverage_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    teacher_run = _write_teacher_run(
        tmp_path / "teacher", levels=("Jungle1", "Jungle2"), ticks=30
    )
    policy = tmp_path / "student.zip"
    policy.write_bytes(b"synthetic-student-policy-bytes")
    dagger_run = tmp_path / "dagger"
    monkeypatch.setattr(
        dagger2,
        "_collect_batch_dagger",
        _fake_dagger_batch_factory([], ticks=20),
    )
    completion = dagger2.collect(
        original_root=tmp_path,
        run_dir=dagger_run,
        policy_path=policy,
        levels=("Jungle1", "Jungle2"),
        seeds_per_level=1,
        seed_base=dagger2.DEFAULT_SEED_BASE,
        parallel_envs=2,
        max_ticks=64,
        device="cpu",
    )
    assert completion["status"] == "COMPLETE"

    merged_dir = tmp_path / "merged"
    receipt = merge_tool.merge_manifests(
        input_run_dirs=[teacher_run, dagger_run],
        output_run_dir=merged_dir,
    )
    assert receipt["status"] == "MERGED"
    assert receipt["episodes"] == 4
    assert receipt["total_ticks"] == 2 * 30 + 2 * 20
    modes = [block["collection_mode"] for block in receipt["sources"]]
    assert modes == ["teacher_driven_v1", dagger2.COLLECTION_MODE]

    merged = legacy._read_json(merged_dir / "episodes_manifest.json")
    assert merged["schema"] == runner.COLLECTION_MANIFEST_SCHEMA
    assert merged["mask_semantics"] == v1.MASK_SEMANTICS
    assert merged["teacher_policy_id"] == v1.TEACHER_POLICY_ID
    # Absolute paths that resolve WITHOUT copying; receipts pass through.
    for entry in merged["episodes"]:
        path = Path(entry["path"])
        assert path.is_absolute()
        assert path.is_file()
        assert entry["sha256"] == legacy._sha256(path)
    keys = [
        (entry["level_id"], entry["seed"]) for entry in merged["episodes"]
    ]
    assert keys == [
        ("Jungle1", v1.DEFAULT_SEED_BASE),
        ("Jungle1", dagger2.DEFAULT_SEED_BASE),
        ("Jungle2", v1.DEFAULT_SEED_BASE + 1),
        ("Jungle2", dagger2.DEFAULT_SEED_BASE + 1),
    ]

    # Duplicate (level, seed) keys across inputs are refused.
    with pytest.raises(ValueError, match="duplicate episode"):
        merge_tool.merge_manifests(
            input_run_dirs=[teacher_run, teacher_run],
            output_run_dir=tmp_path / "duplicates",
        )

    # A mask-semantics mismatch between inputs is refused.
    tampered = tmp_path / "tampered"
    shutil.copytree(dagger_run, tampered)
    document = legacy._read_json(tampered / "episodes_manifest.json")
    document["mask_semantics"] = {
        **document["mask_semantics"],
        "training_masks": "not_the_v1_semantics",
    }
    legacy._write_json_atomic(
        tampered / "episodes_manifest.json", document
    )
    with pytest.raises(ValueError, match="mask_semantics"):
        merge_tool.merge_manifests(
            input_run_dirs=[teacher_run, tampered],
            output_run_dir=tmp_path / "mismatched",
        )

    # The frozen-output discipline holds.
    with pytest.raises(FileExistsError):
        merge_tool.merge_manifests(
            input_run_dirs=[teacher_run, dagger_run],
            output_run_dir=merged_dir,
        )

    # END-TO-END: the coverage runner builds from the merged dir
    # unchanged, and the park-settle trainer loads the result.
    build = runner.run_coverage_replay_build(
        runner.CoverageReplayBuildRunConfig(
            collection_run_dir=merged_dir,
            output_prefix=tmp_path / "out" / "replay",
            builder_config=builder.CoverageReplayV4Config(
                target_coverage_ratio=1.0, total_budget=1_000_000
            ),
        )
    )
    assert build["status"] == "BUILT"
    assert build["collection"]["episodes"] == 4
    assert build["retained_samples"] == 2 * 30 + 2 * 20
    dataset, dataset_receipt = park_settle.load_replay_dataset(
        tmp_path / "out" / "replay.dataset.json"
    )
    assert dataset.sample_count == 100
    assert dataset_receipt["entries"][0]["keys"]["masks"] == "exact_masks"
    assert dataset_receipt["mask_semantics"]["exact_masks"] == (
        "exact_runtime_valid_action_mask"
    )


# ---------------------------------------------------------------------------
# (d) seed embargo + disjointness from the v1 teacher block
# ---------------------------------------------------------------------------


def test_seed_discipline_embargo_and_disjointness() -> None:
    assert dagger2.DEFAULT_SEED_BASE == 1_543_005_000
    dagger_tasks = v1.plan_tasks(
        levels=INCLUDED_LEVELS,
        seeds_per_level=dagger2.DEFAULT_SEEDS_PER_LEVEL,
        seed_base=dagger2.DEFAULT_SEED_BASE,
    )
    assert len(dagger_tasks) == 2 * 55
    dagger2.validate_dagger_seed_plan(dagger_tasks)
    dagger_seeds = {int(task.seed) for task in dagger_tasks}
    for _, first, last in v1.EMBARGOED_SEED_RANGES:
        assert not any(first <= seed <= last for seed in dagger_seeds)
    # Disjoint from the v1 teacher collection block (55 x 3 default plan).
    teacher_tasks = v1.plan_tasks(
        levels=INCLUDED_LEVELS,
        seeds_per_level=v1.DEFAULT_SEEDS_PER_LEVEL,
        seed_base=v1.DEFAULT_SEED_BASE,
    )
    assert dagger_seeds.isdisjoint(
        {int(task.seed) for task in teacher_tasks}
    )
    # The v1 teacher block and the probe block are refused outright.
    with pytest.raises(
        ValueError, match="park_settle_teacher_collection_v1"
    ):
        dagger2.validate_dagger_seed_plan(
            v1.plan_tasks(
                levels=INCLUDED_LEVELS,
                seeds_per_level=1,
                seed_base=v1.DEFAULT_SEED_BASE,
            )
        )
    with pytest.raises(ValueError, match="recovery_intervention_probe_v4"):
        dagger2.validate_dagger_seed_plan(
            v1.plan_tasks(
                levels=INCLUDED_LEVELS,
                seeds_per_level=1,
                seed_base=1_400_920_000,
            )
        )
    # v1's frozen formal embargo validation still applies unchanged.
    with pytest.raises(ValueError, match="formal_selection embargo"):
        dagger2.validate_dagger_seed_plan(
            v1.plan_tasks(
                levels=INCLUDED_LEVELS,
                seeds_per_level=1,
                seed_base=1_600_000_000,
            )
        )


# ---------------------------------------------------------------------------
# (e) resume + quarantine through the reused v1 machinery
# ---------------------------------------------------------------------------


def test_resume_quarantine_and_policy_pinning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy = tmp_path / "student.zip"
    policy.write_bytes(b"synthetic-student-policy-bytes")
    run_dir = tmp_path / "run"
    kwargs: dict[str, Any] = dict(
        original_root=tmp_path,
        run_dir=run_dir,
        policy_path=policy,
        levels=("Jungle1", "Jungle2"),
        seeds_per_level=2,
        seed_base=dagger2.DEFAULT_SEED_BASE,
        parallel_envs=2,
        max_ticks=64,
        device="cpu",
    )

    first_log: list[tuple[str, int]] = []
    monkeypatch.setattr(
        dagger2, "_collect_batch_dagger", _fake_dagger_batch_factory(first_log)
    )
    completion = dagger2.collect(**kwargs)
    assert completion["status"] == "COMPLETE"
    assert completion["collection_mode"] == dagger2.COLLECTION_MODE
    assert len(first_log) == 4
    assert completion["losses"] == 4
    assert completion["label_executed_verb_disagreement_ticks"] == 4 * 10
    assert completion["student_policy"]["sha256"] == legacy._sha256(policy)
    assert completion["formal_seed_consumption"] is False
    # The DAgger config owns the run dir; the manifest keeps the v1
    # schema (runner-compatible) plus honest additive keys.
    config = legacy._read_json(run_dir / "config.json")
    assert config["schema"] == dagger2.CONFIG_SCHEMA
    assert config["student_policy"]["sha256"] == legacy._sha256(policy)
    manifest = legacy._read_json(run_dir / "episodes_manifest.json")
    assert manifest["schema"] == v1.MANIFEST_SCHEMA
    assert manifest["mask_semantics"] == v1.MASK_SEMANTICS
    assert manifest["collection_mode"] == dagger2.COLLECTION_MODE
    assert manifest["environment_execution_semantics"] == (
        dagger2.EXECUTION_SEMANTICS
    )
    assert manifest["student_policy"]["sha256"] == legacy._sha256(policy)
    assert (run_dir / "dagger_completion.json").is_file()
    assert (run_dir / "completion.json").is_file()

    # Resume: nothing is re-collected, everything is reused.
    second_log: list[tuple[str, int]] = []
    monkeypatch.setattr(
        dagger2,
        "_collect_batch_dagger",
        _fake_dagger_batch_factory(second_log),
    )
    completion = dagger2.collect(**kwargs)
    assert completion["status"] == "COMPLETE"
    assert second_log == []
    # Reused entries keep the additive DAgger diagnostics.
    assert completion["label_executed_verb_disagreement_ticks"] == 4 * 10

    # Corrupt one episode IN PLACE: resume must quarantine and
    # re-collect exactly that (level, seed) via the reused machinery.
    episodes_dir = run_dir / v1.EPISODES_DIRNAME
    victim = episodes_dir / (
        f"Jungle2-seed{dagger2.DEFAULT_SEED_BASE + 1}.npz"
    )
    corrupted = b"corrupted-in-place"
    victim.write_bytes(corrupted)
    third_log: list[tuple[str, int]] = []
    monkeypatch.setattr(
        dagger2,
        "_collect_batch_dagger",
        _fake_dagger_batch_factory(third_log),
    )
    completion = dagger2.collect(**kwargs)
    assert completion["status"] == "COMPLETE"
    assert completion["errors"] == []
    assert third_log == [("Jungle2", dagger2.DEFAULT_SEED_BASE + 1)]
    quarantined = sorted(
        episodes_dir.glob(f"{victim.name}.quarantined-*")
    )
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == corrupted
    assert completion["resume_quarantined_files"] == [
        {"original": victim.name, "quarantined_as": quarantined[0].name}
    ]
    manifest = legacy._read_json(run_dir / "episodes_manifest.json")
    healed = next(
        entry
        for entry in manifest["episodes"]
        if entry["level_id"] == "Jungle2"
        and entry["seed"] == dagger2.DEFAULT_SEED_BASE + 1
    )
    assert healed["sha256"] == legacy._sha256(victim)

    # A different student policy must be refused on resume.
    other = tmp_path / "other-student.zip"
    other.write_bytes(b"a-different-policy")
    with pytest.raises(ValueError, match="different student policy"):
        dagger2.collect(**{**kwargs, "policy_path": other})

    # A different plan is refused by the reused v1 contract check.
    with pytest.raises(ValueError, match="different collection contract"):
        dagger2.collect(
            **{**kwargs, "seed_base": dagger2.DEFAULT_SEED_BASE + 500}
        )


# ---------------------------------------------------------------------------
# Real-engine smoke: student policy from the trainer's own constructor,
# DAgger main() -> merge main() -> coverage runner -> trainer loader.
# ---------------------------------------------------------------------------


def test_real_smoke_dagger_merge_runner_chain(tmp_path: Path) -> None:
    if not RETAIL_ROOT.exists():
        pytest.skip("retail extraction is unavailable")
    from tools import (
        distill_alphazuma_55_motor_observable_replay_v2 as motor,
    )

    prototype = motor._prototype(original_root=RETAIL_ROOT, max_ticks=300)
    try:
        policy_kwargs = park_settle.entity_polar_policy_kwargs(
            prototype, learned_features_dim=16
        )
        model = park_settle.build_student_model(
            observation_space=prototype.observation_space,
            action_space=prototype.action_space,
            policy_kwargs=policy_kwargs,
            model_config=park_settle.ParkSettleModelConfig(
                learned_features_dim=16,
                aim_head_identity_scale=5.0,
                model_seed=7,
            ),
            train_config=park_settle.ParkSettleTrainConfig(
                epochs=1,
                batch_size=32,
                learning_rate=1.0e-3,
                holdout_fraction=0.5,
                shuffle_seed=7,
                device="cpu",
            ),
        )
    finally:
        prototype.close()
    policy_path = tmp_path / "tiny-student.zip"
    model.save(str(policy_path))
    del model

    dagger_run = tmp_path / "dagger-run"
    assert (
        dagger2.main(
            [
                "--original-root",
                str(RETAIL_ROOT),
                "--run-dir",
                str(dagger_run),
                "--policy-path",
                str(policy_path),
                "--levels",
                "Jungle1",
                "--seeds-per-level",
                "1",
                "--parallel-envs",
                "1",
                "--max-ticks",
                "300",
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    completion = json.loads(
        (dagger_run / "dagger_completion.json").read_text(encoding="utf-8")
    )
    assert completion["status"] == "COMPLETE"
    assert completion["formal_seed_consumption"] is False
    assert len(completion["episodes"]) == 1
    entry = completion["episodes"][0]
    assert entry["level_id"] == "Jungle1"
    assert entry["seed"] == dagger2.DEFAULT_SEED_BASE
    assert entry["outcome"] in {"win", "loss", "truncated"}
    assert 0 < entry["tick_count"] <= 300
    with np.load(dagger_run / entry["path"]) as archive:
        labels = archive["raw_actions"]
        executed = archive["executed_actions"]
        assert labels.shape == executed.shape
        # Different policies produced genuinely different streams.
        assert np.any(labels != executed)
        # In a 300-tick student-driven smoke the settled teacher may
        # legitimately propose wait on every tick (measured: the
        # untrained tiny student keeps the cursor unsettled), so the
        # fire-edge receipts are checked for CONSISTENCY with the
        # stored label stream rather than for a minimum count; the
        # merged build below gets its fire windows from the
        # teacher-driven episode.
        fire = labels[:, 0] == v1.FIRE_VERB_INDEX
        edges = int(np.count_nonzero(fire & ~np.concatenate(([False], fire[:-1]))))
        assert entry["fire_edges_raw_intent"] == edges
        executed_fire = executed[:, 0] == v1.FIRE_VERB_INDEX
        executed_edges = int(
            np.count_nonzero(
                executed_fire
                & ~np.concatenate(([False], executed_fire[:-1]))
            )
        )
        assert entry["fire_edges_executed_student"] == executed_edges
    assert entry["label_executed_action_disagreement_ticks"] > 0
    assert entry["sha256"] == legacy._sha256(dagger_run / entry["path"])

    teacher_run = tmp_path / "teacher-run"
    assert (
        v1.main(
            [
                "--original-root",
                str(RETAIL_ROOT),
                "--run-dir",
                str(teacher_run),
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

    merged_dir = tmp_path / "merged"
    assert (
        merge_tool.main(
            [
                "--input-run-dirs",
                str(teacher_run),
                str(dagger_run),
                "--output-run-dir",
                str(merged_dir),
            ]
        )
        == 0
    )

    build = runner.run_coverage_replay_build(
        runner.CoverageReplayBuildRunConfig(
            collection_run_dir=merged_dir,
            output_prefix=tmp_path / "replay",
            builder_config=builder.CoverageReplayV4Config(
                target_coverage_ratio=1.0, total_budget=1_000_000
            ),
        )
    )
    assert build["status"] == "BUILT"
    assert build["collection"]["episodes"] == 2
    assert build["retained_samples"] > 0
    # The teacher-driven episode guarantees nonzero fire windows in the
    # merged coverage build even when the short student-driven episode
    # carried all-wait teacher labels.
    assert build["per_level_coverage"]["Jungle1"]["fire_edges"] > 0
    dataset, dataset_receipt = park_settle.load_replay_dataset(
        tmp_path / "replay.dataset.json"
    )
    assert dataset.sample_count == build["retained_samples"]
    assert dataset_receipt["mask_semantics"]["exact_masks"] == (
        "exact_runtime_valid_action_mask"
    )
