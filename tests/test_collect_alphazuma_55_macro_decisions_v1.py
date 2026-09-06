"""Tests for the teacher macro-decision collector.

Covers:
(a) real-engine smoke (Jungle1, one seed, small max-ticks, CPU, one
    env): decisions recorded with the correct shapes, macro-mask
    true-count sanity, the teacher deciding on the RAW leading slice
    while the recorded observations are STACKED, NPZ + manifest sha256
    receipts;
(b) the adapter mapping and per-decision honesty on stubbed
    vectors/teachers (parked-target passthrough, hop and mask
    fallbacks, cumulative native tick indices, honest loss flagging);
(c) seed embargo + reserved engineering block refusals;
(d) resume: completed episodes are skipped, corrupted/orphaned bytes
    are quarantined and re-collected, and a different plan is refused.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tools import collect_alphazuma_55_macro_decisions_v1 as collector
from tools import collect_alphazuma_55_park_settle_episodes_v1 as episodes_v1
from tools import distill_alphazuma_55 as legacy
from tools import probe_alphazuma_55_macro_eval_v1 as macro_eval
from tools import probe_alphazuma_55_recovery_intervention_v5 as v5
import zuma_rl.park_settle_action_wrapper as wrapper_module


RETAIL_ROOT = Path("/mnt/d/SteamLibrary/steamapps/common/Zuma's Revenge")
RAW_DIM = 6
STACK_LAGS = (0, 4, 8)
STACKED_DIM = RAW_DIM * len(STACK_LAGS)
TICKS_PER_MACRO = 20


# ---------------------------------------------------------------------------
# Synthetic recorder helpers (mirror the collector-v1 test conventions).
# ---------------------------------------------------------------------------


def _macro_info(
    verb: int, aim: int, *, release_error: int = 0
) -> dict[str, Any]:
    released = verb == wrapper_module.MACRO_FIRE
    return {
        "profile": "park-settle-v1",
        "macro_verb": wrapper_module.MACRO_VERB_NAMES[verb],
        "target_aim_bin": aim,
        "ticks_consumed": TICKS_PER_MACRO,
        "settle_ticks": 3,
        "settled": True,
        "button_executed": released,
        "released": released,
        "release_aim_bin": ((aim + release_error) % 180) if released else None,
        "decision_point": True,
        "terminal": False,
    }


def _fill_recorder(
    recorder: collector._DecisionRecorder,
    *,
    decisions: int,
    fire_every: int = 3,
    observation_seed: int = 0,
    width: int = STACKED_DIM,
) -> collector._DecisionRecorder:
    rng = np.random.default_rng(observation_seed)
    for index in range(decisions):
        verb = (
            wrapper_module.MACRO_FIRE
            if index % fire_every == 0
            else wrapper_module.MACRO_WAIT_HOLD
        )
        aim = (10 + index) % 180
        observation = rng.random(width, dtype=np.float32)
        observation[0] = np.float32(index)
        recorder.record_decision(
            observation=observation,
            macro_action=np.asarray([verb, aim], dtype=np.int64),
            macro_mask=np.ones(collector.MACRO_MASK_WIDTH, dtype=np.bool_),
            teacher_per_tick_action=np.asarray([verb, aim], dtype=np.int64),
            fallback=None,
        )
        recorder.record_result(
            _macro_info(verb, aim), {"score": index * 10}
        )
    return recorder


def _synthetic_episode(
    run_dir: Path,
    task: collector.EpisodeTask,
    *,
    decisions: int = 12,
    final_info: dict[str, Any] | None = None,
    guard_exhausted: bool = False,
    width: int = STACKED_DIM,
) -> dict[str, Any]:
    episodes_dir = run_dir / collector.EPISODES_DIRNAME
    spill_dir = run_dir / collector.SPILL_DIRNAME
    episodes_dir.mkdir(parents=True, exist_ok=True)
    spill_dir.mkdir(parents=True, exist_ok=True)
    recorder = _fill_recorder(
        collector._DecisionRecorder(
            spill_path=spill_dir / f"{task.file_name}.spill",
            observation_width=width,
        ),
        decisions=decisions,
        observation_seed=int(task.seed),
        width=width,
    )
    if final_info is None:
        final_info = {"outcome": "win", "score": (decisions - 1) * 10}
    return collector._finalize_episode(
        task=task,
        recorder=recorder,
        final_info=final_info,
        guard_exhausted=guard_exhausted,
        episodes_dir=episodes_dir,
        wall_seconds=0.25,
    )


# ---------------------------------------------------------------------------
# (c) Seed discipline: frozen embargo + reserved engineering blocks.
# ---------------------------------------------------------------------------


def test_seed_plan_defaults_avoid_embargo_and_reserved_blocks() -> None:
    tasks = collector.plan_tasks(
        levels=("Jungle1", "Jungle2"),
        seeds_per_level=40,
        seed_base=collector.DEFAULT_SEED_BASE,
    )
    assert len(tasks) == 80
    assert [int(task.seed) for task in tasks] == list(
        range(
            collector.DEFAULT_SEED_BASE, collector.DEFAULT_SEED_BASE + 80
        )
    )
    collector.validate_seed_plan(tasks)
    # The frozen formal embargo aborts before any environment work.
    with pytest.raises(ValueError, match="formal_selection embargo"):
        collector.validate_seed_plan(
            collector.plan_tasks(
                levels=("Jungle1",),
                seeds_per_level=1,
                seed_base=1_600_000_000,
            )
        )
    # Reserved engineering blocks are refused too, straddling included.
    with pytest.raises(ValueError, match="macro_ppo_engineering"):
        collector.validate_seed_plan(
            collector.plan_tasks(
                levels=("Jungle1", "Jungle2"),
                seeds_per_level=1,
                seed_base=1_493_999_999,
            )
        )
    with pytest.raises(
        ValueError, match="park_settle_teacher_collection_v1"
    ):
        collector.validate_seed_plan(
            collector.plan_tasks(
                levels=("Jungle1",),
                seeds_per_level=1,
                seed_base=1_543_004_500,
            )
        )
    # The default base sits outside every reserved block.
    for _, first, last in collector.RESERVED_ENGINEERING_SEED_BLOCKS:
        assert not first <= collector.DEFAULT_SEED_BASE <= last


def test_cli_defaults_match_the_requested_contract() -> None:
    parser = collector.build_parser()
    args = parser.parse_args(
        ["--original-root", "root", "--run-dir", "run"]
    )
    assert args.seeds_per_level == 40
    assert args.seed_base == 1_494_000_000
    assert args.parallel_envs == 12
    assert args.hold_ticks == 8
    assert args.stack_lags == "0,4,8"
    assert args.max_ticks == 30_000
    assert args.device == "cpu"
    assert collector.MACRO_MASK_WIDTH == 183
    # Reused v1 machinery is imported, never copied.
    assert collector.plan_tasks is episodes_v1.plan_tasks
    assert collector.EpisodeTask is episodes_v1.EpisodeTask
    source = Path(collector.__file__).read_text(encoding="utf-8")
    assert "def adapt_per_tick_action" not in source
    assert "def _make_stacked_macro_env_factory" not in source
    assert "class RawFrameTeacher" not in source


# ---------------------------------------------------------------------------
# NPZ schema, receipts, and honest outcomes on the synthetic recorder.
# ---------------------------------------------------------------------------


def test_decision_npz_schema_and_sha256_receipt(tmp_path: Path) -> None:
    task = collector.plan_tasks(
        levels=("Jungle1",),
        seeds_per_level=1,
        seed_base=collector.DEFAULT_SEED_BASE,
    )[0]
    entry = _synthetic_episode(tmp_path, task, decisions=12)
    path = tmp_path / entry["path"]
    assert path.is_file()
    assert entry["sha256"] == legacy._sha256(path)
    assert entry["decision_count"] == 12
    assert entry["native_ticks"] == 12 * TICKS_PER_MACRO
    assert entry["observation_width"] == STACKED_DIM
    assert entry["teacher_won"] is True
    assert entry["shots"] == 4  # fire every 3rd of 12 decisions
    assert entry["shots_on_target"] == 4
    assert entry["macro_verb_counts"] == {"fire": 4, "wait_hold": 8}
    assert entry["mask_fallbacks"] == {}
    with np.load(path) as archive:
        assert sorted(archive.files) == [
            "fallback_codes",
            "macro_actions",
            "macro_masks",
            "observations",
            "teacher_per_tick_actions",
            "tick_indices",
        ]
        assert archive["observations"].dtype == np.float32
        assert archive["observations"].shape == (12, STACKED_DIM)
        assert archive["macro_actions"].dtype == np.int64
        assert archive["macro_actions"].shape == (12, 2)
        assert archive["macro_masks"].dtype == np.bool_
        assert archive["macro_masks"].shape == (
            12,
            collector.MACRO_MASK_WIDTH,
        )
        assert archive["fallback_codes"].dtype == np.int8
        # Tick indices are native ticks consumed BEFORE each decision.
        assert np.array_equal(
            archive["tick_indices"],
            np.arange(12, dtype=np.int64) * TICKS_PER_MACRO,
        )
    # The spill file is gone once the NPZ landed.
    assert not list((tmp_path / collector.SPILL_DIRNAME).glob("*.spill"))


def test_teacher_losses_are_recorded_and_flagged(tmp_path: Path) -> None:
    tasks = collector.plan_tasks(
        levels=("Jungle1", "Jungle2", "Jungle3"),
        seeds_per_level=1,
        seed_base=collector.DEFAULT_SEED_BASE,
    )
    lost = _synthetic_episode(
        tmp_path, tasks[0], final_info={"outcome": "loss", "score": 123}
    )
    assert lost["outcome"] == "loss"
    assert lost["teacher_won"] is False
    # The loss's decisions are stored all the same.
    assert lost["decision_count"] == 12
    assert (tmp_path / lost["path"]).is_file()
    truncated = _synthetic_episode(
        tmp_path,
        tasks[1],
        final_info={"outcome": None, "TimeLimit.truncated": True, "score": 7},
    )
    assert truncated["outcome"] == "truncated"
    assert truncated["teacher_won"] is False
    assert truncated["time_limit_truncated"] is True
    guarded = _synthetic_episode(
        tmp_path, tasks[2], final_info={}, guard_exhausted=True
    )
    assert guarded["outcome"] == "truncated"
    assert guarded["guard_exhausted"] is True
    # Guard episodes keep their decisions and the recorded score.
    assert guarded["decision_count"] == 12
    assert guarded["score"] == 110


def test_recorder_enforces_decision_result_pairing(tmp_path: Path) -> None:
    recorder = collector._DecisionRecorder(
        spill_path=tmp_path / "pair.spill", observation_width=STACKED_DIM
    )
    observation = np.zeros(STACKED_DIM, dtype=np.float32)
    kwargs = dict(
        observation=observation,
        macro_action=np.asarray([0, 5], dtype=np.int64),
        macro_mask=np.ones(collector.MACRO_MASK_WIDTH, dtype=np.bool_),
        teacher_per_tick_action=np.asarray([0, 5], dtype=np.int64),
        fallback=None,
    )
    with pytest.raises(RuntimeError, match="record_decision"):
        recorder.record_result(_macro_info(0, 5), {})
    recorder.record_decision(**kwargs)
    with pytest.raises(RuntimeError, match="awaits its macro result"):
        recorder.record_decision(**kwargs)
    with pytest.raises(RuntimeError, match="missing its macro result"):
        recorder.write_npz(tmp_path / "pair.npz")
    recorder.record_result(_macro_info(0, 5), {})
    with pytest.raises(ValueError, match="unknown adapter fallback"):
        recorder.record_decision(**{**kwargs, "fallback": "bogus"})
    recorder.discard()


# ---------------------------------------------------------------------------
# (b) Stubbed batch: adapter mapping, raw-frame teacher slice, tick
# indices, and per-decision fallback honesty.
# ---------------------------------------------------------------------------


class _ScriptedTeacher:
    """Stub teacher replaying a fixed per-tick proposal script."""

    def __init__(self, script: list[tuple[int, int]]) -> None:
        self.script = list(script)
        self.calls = 0
        self.frames: list[np.ndarray] = []

    def act(self, observation: Any) -> np.ndarray:
        frame = np.asarray(observation).reshape(-1)
        self.frames.append(frame.copy())
        verb, aim = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return np.asarray((verb, aim), dtype=np.int64)


class _StubStackedMacroVector:
    """Stacked macro vec-env stand-in emitting park_settle telemetry.

    Observations are STACKED (width ``STACKED_DIM``); the leading raw
    slice carries a distinctive marker per step so the raw-frame
    teacher contract is checkable, and the lagged tail is negative so a
    teacher leaking past the leading slice is caught immediately.
    """

    def __init__(self, count: int, episode_macros: int) -> None:
        self.count = count
        self.episode_macros = episode_macros
        self.received: list[np.ndarray] = []
        self._step = 0

    def _observe(self, marker: float) -> np.ndarray:
        observations = np.full(
            (self.count, STACKED_DIM), -1.0, dtype=np.float32
        )
        observations[:, :RAW_DIM] = marker
        return observations

    def seed(self, value: int) -> None:
        self.seed_value = int(value)

    def reset(self) -> np.ndarray:
        self._step = 0
        return self._observe(100.0)

    def step(self, actions: Any) -> tuple[Any, Any, Any, Any]:
        actions = np.asarray(actions, dtype=np.int64)
        self.received.append(actions.copy())
        step = self._step
        self._step += 1
        done = step == self.episode_macros - 1
        infos = []
        for index in range(self.count):
            verb = int(actions[index][0])
            aim = int(actions[index][1])
            info = {
                "score": (step + 1) * 10,
                "ticks": (step + 1) * TICKS_PER_MACRO,
                "outcome": ("loss" if done else None),
                "TimeLimit.truncated": False,
                "park_settle": _macro_info(verb, aim),
            }
            infos.append(info)
        return (
            self._observe(float(step + 1)),
            np.zeros(self.count, dtype=np.float64),
            np.full(self.count, done, dtype=np.bool_),
            infos,
        )

    def close(self) -> None:
        pass


def test_stub_batch_maps_teacher_decisions_and_strips_raw_frames(
    tmp_path: Path,
) -> None:
    tasks = collector.plan_tasks(
        levels=("Jungle1",),
        seeds_per_level=1,
        seed_base=collector.DEFAULT_SEED_BASE,
    )
    episodes_dir = tmp_path / collector.EPISODES_DIRNAME
    spill_dir = tmp_path / collector.SPILL_DIRNAME
    episodes_dir.mkdir(parents=True)
    spill_dir.mkdir(parents=True)
    # wait parks 137; hop falls back (unsupported); swap is masked at
    # decision 2 (masked fallback); fire 70 executes and ends the
    # episode as a LOSS -- honesty must keep and flag it.
    teacher = _ScriptedTeacher([(0, 137), (3, 12), (2, 55), (1, 70)])
    masks = np.ones((1, collector.MACRO_MASK_WIDTH), dtype=np.bool_)
    mask_calls: list[int] = []

    def masks_provider(vector: Any) -> np.ndarray:
        mask_calls.append(len(mask_calls))
        provided = masks.copy()
        if len(mask_calls) == 3:  # the swap proposal's decision point
            provided[0, wrapper_module.MACRO_SWAP] = False
        return provided

    factory_raw_dims: list[int] = []

    def teachers_factory(vector: Any, *, raw_observation_dim: int) -> list:
        factory_raw_dims.append(int(raw_observation_dim))
        return v5.wrap_raw_frame_teachers(
            [teacher], raw_observation_dim=raw_observation_dim
        )

    manifest_entries: list[dict[str, Any]] = []
    vector = _StubStackedMacroVector(1, 4)
    entries = collector._collect_batch(
        original_root=tmp_path,
        tasks=tasks,
        record_flags=[True],
        max_ticks=200,
        hold_ticks=8,
        stack_lags=STACK_LAGS,
        episodes_dir=episodes_dir,
        spill_dir=spill_dir,
        on_episode=manifest_entries.append,
        vector_factory=lambda **kwargs: vector,
        teachers_factory=teachers_factory,
        masks_provider=masks_provider,
    )
    assert vector.seed_value == collector.DEFAULT_SEED_BASE
    # The default teacher plumbing received the derived RAW width.
    assert factory_raw_dims == [RAW_DIM]
    # The teacher decided on the RAW leading slice: width RAW_DIM (the
    # stacked observation is 3x wider) and the marker content, never
    # the negative lagged tail.
    assert len(teacher.frames) == 4
    assert all(frame.shape == (RAW_DIM,) for frame in teacher.frames)
    assert STACKED_DIM != RAW_DIM
    np.testing.assert_array_equal(
        np.asarray([float(frame[0]) for frame in teacher.frames]),
        np.asarray([100.0, 1.0, 2.0, 3.0]),
    )
    assert all(float(frame.min()) >= 0.0 for frame in teacher.frames)
    # The vector executed the adapter-mapped macros, verbatim.
    executed = np.concatenate(vector.received, axis=0)
    np.testing.assert_array_equal(
        executed,
        np.asarray(
            [[0, 137], [0, 12], [0, 55], [1, 70]], dtype=np.int64
        ),
    )
    assert len(entries) == len(manifest_entries) == 1
    entry = entries[0]
    assert entry["outcome"] == "loss"
    assert entry["teacher_won"] is False
    assert entry["decision_count"] == 4
    assert entry["native_ticks"] == 4 * TICKS_PER_MACRO
    assert entry["mask_fallbacks"] == {
        macro_eval.FALLBACK_UNSUPPORTED: 1,
        macro_eval.FALLBACK_MASKED: 1,
    }
    assert entry["shots"] == 1
    assert entry["shots_on_target"] == 1
    with np.load(episodes_dir / tasks[0].file_name) as archive:
        # The stored observations are the STACKED decision-point rows.
        assert archive["observations"].shape == (4, STACKED_DIM)
        np.testing.assert_array_equal(
            archive["observations"][:, 0],
            np.asarray([100.0, 1.0, 2.0, 3.0], dtype=np.float32),
        )
        assert float(archive["observations"][:, RAW_DIM:].max()) == -1.0
        # Macro actions carry the adapter mapping (parked targets kept).
        np.testing.assert_array_equal(archive["macro_actions"], executed)
        # The raw per-tick proposals and fallback codes stay honest.
        np.testing.assert_array_equal(
            archive["teacher_per_tick_actions"],
            np.asarray(
                [[0, 137], [3, 12], [2, 55], [1, 70]], dtype=np.int64
            ),
        )
        np.testing.assert_array_equal(
            archive["fallback_codes"],
            np.asarray(
                [
                    collector.FALLBACK_CODE_NONE,
                    collector.FALLBACK_CODE_UNSUPPORTED,
                    collector.FALLBACK_CODE_MASKED,
                    collector.FALLBACK_CODE_NONE,
                ],
                dtype=np.int8,
            ),
        )
        # The recorded mask is the decision point's own mask.
        assert not bool(
            archive["macro_masks"][2, wrapper_module.MACRO_SWAP]
        )
        assert int(archive["macro_masks"][2].sum()) == (
            collector.MACRO_MASK_WIDTH - 1
        )
        # Native tick indices accumulate the macro tick consumption.
        np.testing.assert_array_equal(
            archive["tick_indices"],
            np.asarray([0, 20, 40, 60], dtype=np.int64),
        )


# ---------------------------------------------------------------------------
# (d) Resume: skip, quarantine, re-collect, plan pinning.
# ---------------------------------------------------------------------------


def _fake_collect_batch_factory(record_log: list[tuple[str, int]]):
    def fake_collect_batch(
        *,
        original_root: Path,
        tasks: list[collector.EpisodeTask],
        record_flags: list[bool],
        max_ticks: int,
        hold_ticks: int,
        stack_lags: tuple[int, ...],
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
                collector._DecisionRecorder(
                    spill_path=spill_dir / f"{task.file_name}.spill",
                    observation_width=STACKED_DIM,
                ),
                decisions=6,
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


def test_resume_skips_completed_and_quarantines_corruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    kwargs = dict(
        original_root=tmp_path,
        run_dir=run_dir,
        levels=("Jungle1", "Jungle2"),
        seeds_per_level=2,
        seed_base=collector.DEFAULT_SEED_BASE,
        parallel_envs=2,
        max_ticks=64,
        hold_ticks=8,
        stack_lags=STACK_LAGS,
    )
    first_log: list[tuple[str, int]] = []
    monkeypatch.setattr(
        collector, "_collect_batch", _fake_collect_batch_factory(first_log)
    )
    completion = collector.collect(**kwargs)
    assert completion["status"] == "COMPLETE"
    assert len(first_log) == 4
    assert completion["wins"] == 4
    assert completion["teacher_losses_flagged"] == 0
    assert completion["total_decisions"] == 24
    manifest_path = run_dir / "episodes_manifest.json"
    manifest = legacy._read_json(manifest_path)
    assert manifest["schema"] == collector.MANIFEST_SCHEMA
    assert manifest["stack_lags"] == list(STACK_LAGS)
    assert manifest["hold_ticks"] == 8
    assert manifest["teacher_losses_are_recorded_and_flagged"] is True
    assert completion["episodes_manifest"]["sha256"] == legacy._sha256(
        manifest_path
    )

    # A clean resume steps NOTHING.
    second_log: list[tuple[str, int]] = []
    monkeypatch.setattr(
        collector, "_collect_batch", _fake_collect_batch_factory(second_log)
    )
    completion = collector.collect(**kwargs)
    assert completion["status"] == "COMPLETE"
    assert second_log == []
    assert all(entry["reused"] is True for entry in completion["episodes"])

    # Corrupt one NPZ in place: resume quarantines and re-collects it.
    episodes_dir = run_dir / collector.EPISODES_DIRNAME
    victim = episodes_dir / f"Jungle2-seed{collector.DEFAULT_SEED_BASE + 1}.npz"
    corrupted_bytes = b"corrupted-in-place"
    victim.write_bytes(corrupted_bytes)
    third_log: list[tuple[str, int]] = []
    monkeypatch.setattr(
        collector, "_collect_batch", _fake_collect_batch_factory(third_log)
    )
    completion = collector.collect(**kwargs)
    assert completion["status"] == "COMPLETE"
    assert completion["errors"] == []
    assert third_log == [("Jungle2", collector.DEFAULT_SEED_BASE + 1)]
    quarantined = sorted(episodes_dir.glob(f"{victim.name}.quarantined-*"))
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == corrupted_bytes
    assert completion["resume_quarantined_files"] == [
        {"original": victim.name, "quarantined_as": quarantined[0].name}
    ]
    healed = next(
        entry
        for entry in legacy._read_json(manifest_path)["episodes"]
        if entry["level_id"] == "Jungle2"
        and entry["seed"] == collector.DEFAULT_SEED_BASE + 1
    )
    assert healed["sha256"] == legacy._sha256(victim)

    # A missing file equally triggers re-collection.
    victim.unlink()
    fourth_log: list[tuple[str, int]] = []
    monkeypatch.setattr(
        collector, "_collect_batch", _fake_collect_batch_factory(fourth_log)
    )
    completion = collector.collect(**kwargs)
    assert completion["status"] == "COMPLETE"
    assert fourth_log == [("Jungle2", collector.DEFAULT_SEED_BASE + 1)]

    # A different plan must refuse to reuse the run dir.
    with pytest.raises(ValueError, match="different collection contract"):
        collector.collect(**{**kwargs, "hold_ticks": 16})
    with pytest.raises(ValueError, match="different collection contract"):
        collector.collect(**{**kwargs, "stack_lags": (0, 2)})

    # Embargoed and reserved seed bases abort before any environment.
    with pytest.raises(ValueError, match="embargo"):
        collector.collect(
            **{
                **kwargs,
                "run_dir": tmp_path / "embargoed",
                "seed_base": 1_700_000_000,
            }
        )
    with pytest.raises(ValueError, match="macro_ppo_engineering"):
        collector.collect(
            **{
                **kwargs,
                "run_dir": tmp_path / "reserved",
                "seed_base": 1_460_000_100,
            }
        )


# ---------------------------------------------------------------------------
# (a) Real-engine smoke: Jungle1, one seed, one env, CPU.
# ---------------------------------------------------------------------------


def test_real_engine_smoke_jungle1_records_stacked_decisions(
    tmp_path: Path,
) -> None:
    if not RETAIL_ROOT.exists():
        pytest.skip("retail extraction is unavailable")
    run_dir = tmp_path / "run"
    max_ticks = 600
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
                str(max_ticks),
            ]
        )
        == 0
    )
    completion = json.loads(
        (run_dir / "completion.json").read_text(encoding="utf-8")
    )
    assert completion["status"] == "COMPLETE"
    assert completion["formal_seed_consumption"] is False
    assert completion["training_authority"] is False
    assert completion["stack_lags"] == list(STACK_LAGS)
    assert len(completion["episodes"]) == 1
    entry = completion["episodes"][0]
    assert entry["level_id"] == "Jungle1"
    assert entry["seed"] == collector.DEFAULT_SEED_BASE
    assert entry["outcome"] in {"win", "loss", "truncated"}
    assert entry["teacher_won"] is (entry["outcome"] == "win")
    assert entry["decision_count"] >= 2
    assert 0 < entry["native_ticks"] <= max_ticks
    # The teacher fired through the macro protocol.
    assert entry["shots"] >= 1
    assert entry["shots_on_target"] >= 1
    path = run_dir / entry["path"]
    assert entry["sha256"] == legacy._sha256(path)

    manifest = json.loads(
        (run_dir / "episodes_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema"] == collector.MANIFEST_SCHEMA
    assert manifest["decision_interface"]["action_space_nvec"] == [3, 180]
    assert manifest["observation_stack"]["teacher_observation"] == (
        "raw_current_frame_leading_slice"
    )

    with np.load(path) as archive:
        observations = np.asarray(archive["observations"])
        actions = np.asarray(archive["macro_actions"])
        masks = np.asarray(archive["macro_masks"])
        ticks = np.asarray(archive["tick_indices"])
    count = int(entry["decision_count"])
    width = int(observations.shape[1])
    assert observations.shape == (count, width)
    assert observations.dtype == np.float32
    # The stored observation is STACKED: three frames of the raw width
    # the teacher decided on (stacked width != raw width).
    assert width % len(STACK_LAGS) == 0
    raw_dim = width // len(STACK_LAGS)
    assert width == 3 * raw_dim
    assert width != raw_dim
    assert entry["observation_width"] == width
    # Decision 0 is the reset frame: the ring buffer clamps every lag
    # to the only frame it holds, so the three slices are identical.
    first = observations[0]
    np.testing.assert_array_equal(
        first[:raw_dim], first[raw_dim : 2 * raw_dim]
    )
    np.testing.assert_array_equal(
        first[:raw_dim], first[2 * raw_dim : 3 * raw_dim]
    )
    # Once >= 8 native ticks passed, the lag-8 frame is real history:
    # at least one such decision differs from its current frame.
    matured = observations[ticks >= 8]
    assert matured.shape[0] >= 1
    assert any(
        not np.array_equal(
            row[:raw_dim], row[2 * raw_dim : 3 * raw_dim]
        )
        for row in matured
    )
    # Macro actions stay inside the wrapper interface and mask-legal.
    assert actions.shape == (count, 2)
    assert bool(np.all((actions[:, 0] >= 0) & (actions[:, 0] < 3)))
    assert bool(np.all((actions[:, 1] >= 0) & (actions[:, 1] < 180)))
    assert bool(np.all(masks[np.arange(count), actions[:, 0]]))
    # Mask true-count sanity: aim bins are always valid and wait_hold
    # is always legal at a decision point, so 181 <= sum <= 183.
    assert masks.shape == (count, collector.MACRO_MASK_WIDTH)
    true_counts = masks.sum(axis=1)
    assert bool(np.all(true_counts >= 181))
    assert bool(np.all(true_counts <= 183))
    assert bool(np.all(masks[:, wrapper_module.MACRO_WAIT_HOLD]))
    assert bool(np.all(masks[:, 3:]))
    # Native tick indices start at 0 and strictly increase.
    assert int(ticks[0]) == 0
    assert bool(np.all(np.diff(ticks) > 0))
    assert int(ticks[-1]) < entry["native_ticks"]
