"""Full-tick teacher episode collector for the v4 coverage replay builder.

Purpose
-------
Produce the raw material that
``tools/build_alphazuma_55_motor_observable_replay_v4.py`` consumes:
FULL-tick, teacher-driven episode records for the AlphaZuma 55 campaign
(default 3 seeds per level).  The 55/55 teacher
(``CurveAwareSettledStrategicRevengeTeacher``, deterministic code built
from each environment's ``teacher_spec()``; no checkpoint file exists)
executes EVERY tick through the motor-observable
``ObservableHumanSpeedrunWrapper`` stack -- exactly the elite-human input
regime the park-settle students train and run under.

Episode/tick contract (mirrors the builder's documented input record)
----------------------------------------------------------------------
Per tick the collector records, at the moment the teacher acted:

* ``observation`` -- the actor observation the teacher saw (float32).
* ``raw_action`` -- the RAW teacher intent ``[verb, aim_bin]`` (int64),
  BEFORE any legality relaxation; fire edges in this stream are what the
  builder's fire-window tier protects.
* ``training_mask`` -- the relaxed replay-v2 training mask produced by
  ``distill_alphazuma_55_polar_intent_wide_v1._intent_label`` (the exact
  runtime mask with only the teacher's target verb unmasked).
* ``exact_mask`` -- the UNRELAXED exact runtime valid-action mask as
  ``get_action_masks`` reported it.  Roughly a quarter of raw teacher
  intents are runtime-illegal; that signal is preserved, never validated
  away, so the park-settle trainer can report the honest
  ``illegal_teacher_verb_fraction``.
* ``effective_action`` -- the action actually executed by the wrapper
  stack (illegal teacher verbs execute as waits via ``_effective_label``).
* ``intent_mask_relaxed`` -- whether this tick needed the relaxation.
* per-tick ``info`` scalars -- ``score``, ``chain_length`` (the first
  field of the builder's default ``danger_field_priority`` that
  ``RevengeEnv._info`` actually emits), ``visible_balls``, and the
  ``human_speedrun`` executed-motor fields the wrapper publishes
  (``desired_verb``, ``desired_aim_bin``, ``executed_verb``,
  ``executed_aim_bin``, ``button_enqueued``).

Storage layout (disk honesty)
-----------------------------
Full-tick observations for a 55x3 collection are ~2M ticks x 22,904
float32 = ~183 GB raw, so observations are NOT inlined into JSON.  Each
episode is written as one compressed NPZ (``np.savez_compressed``; the
observations are mostly zeros, so deflate is large) under
``<run-dir>/episodes/``, and ``<run-dir>/episodes_manifest.json`` lists
every episode with a RELATIVE path plus a sha256 receipt.

Three consumers are served without modifying any frozen file:

1. ``tools.distill_alphazuma_55_park_settle_v1.load_replay_dataset``
   loads ``episodes_manifest.json`` DIRECTLY (its documented
   ``{"episodes": [{"path":..., "sha256":...}, ...]}`` one-NPZ-per-episode
   layout): each NPZ carries ``observations`` / ``raw_actions`` /
   ``exact_masks`` / ``tick_indices``, and the manifest declares
   ``mask_semantics`` so the exact-mask key is enforced at load time.
2. ``load_episode_records(manifest_path)`` (this module) hydrates the
   NPZ shards back into the builder's in-memory episode-record contract
   (numpy arrays per tick) for
   ``build_coverage_equalized_replay_dataset``.
3. ``--emit-builder-episodes <path>`` writes the builder's inline
   ``{"episodes": [...]}`` JSON document (streamed tick by tick, bounded
   memory) so ``build_alphazuma_55_motor_observable_replay_v4.main`` runs
   on the output UNCHANGED.  Observation floats round-trip bit-exactly
   (float32 -> shortest-repr double -> float32).  Inline JSON is only
   practical at smoke/subset scale; full-run consumers should hydrate via
   ``load_episode_records`` / ``iter_episode_records``.

Honest outcomes
---------------
A teacher loss or a max-ticks truncation is recorded in the episode's
``outcome`` field (``"win"`` / ``"loss"`` / ``"truncated"``) and is never
silently dropped -- the builder may still use those ticks.  If the
collection guard loop ever exhausts without the environment reporting
done, the episode is finalized with ``outcome == "truncated"`` and
``guard_exhausted: true`` instead of being discarded.

Crash safety and resume
-----------------------
Episode NPZs are written atomically (temp + ``os.replace``) the moment
each episode finishes, and the episodes manifest is rewritten atomically
after every episode, so one episode failing never loses completed
episodes.  On restart the collector verifies existing receipts
(existence + sha256) and skips already-collected episodes; a batch whose
episodes are all present is skipped without stepping any environment.
Partially complete batches re-run the whole seed-contiguous batch (the
vector seed convention needs contiguous seeds) but only the missing
episodes are recorded and written.

Resume also SELF-HEALS bad bytes on disk.  An episode NPZ that fails its
manifest receipt (corrupted in place) or that no manifest entry claims
(orphaned by a crash in the window between the NPZ ``os.replace`` and
the manifest rewrite) sits exactly where the re-collection must write,
and ``write_npz`` correctly refuses to overwrite existing files -- so
before any batch runs, every unverified file under
``<run-dir>/episodes/`` is quarantine-renamed (``*.quarantined-<utc>-
<pid>``; never silently deleted, the bytes stay on disk for forensics)
and the episode is re-collected.  The quarantine actions are recorded in
``completion.json`` under ``resume_quarantined_files``.

Seed discipline
---------------
Seeds follow the replay-anchor convention
(``build_alphazuma_55_motor_observable_replay`` v1/v2/v3 collection via
``_collect_intent_round``): episode seed = ``seed_base + pass_index *
len(levels) + level_position`` and the vector is seeded with the first
seed of each contiguous batch.  The default engineering base
``1_543_004_000`` continues the 1_543_xxx_xxx engineering training block
(v3 anchors ended at 1_543_002_329; the park-settle model/shuffle seeds
hold 1_543_003_1xx).  Every planned seed is validated against the frozen
formal embargo ranges -- selection ``[1_600_000_000, 1_600_000_219]``,
final blind ``[1_700_000_000, 1_700_000_439]``, continuous campaign
``[1_800_000_000, 1_800_000_219]`` -- and the run aborts before touching
any environment if the plan overlaps them.  ``formal_seed_consumption``
is recorded as ``false`` in every receipt.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Iterator, Mapping, Sequence

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import numpy as np

from tools import distill_alphazuma_55 as legacy
from tools import distill_alphazuma_55_motor_observable_replay_v2 as motor
from tools import distill_alphazuma_55_polar_intent_wide_v1 as intent
from zuma_rl.alphazuma_55 import INCLUDED_LEVELS
from zuma_rl.human_speedrun import EliteHumanInputConfig, WinFirstRewardConfig
from zuma_rl.revenge_settled_strategic_teacher_v3 import (
    CurveAwareSettledStrategicRevengeTeacher,
)


SCRIPT_PATH = Path(__file__).resolve()
BUILDER_PATH = (
    SCRIPT_PATH.parent / "build_alphazuma_55_motor_observable_replay_v4.py"
)
SCHEMA_PREFIX = "zuma-rl.alphazuma-55-park-settle-episodes"
CONFIG_SCHEMA = f"{SCHEMA_PREFIX}-collection-config"
MANIFEST_SCHEMA = f"{SCHEMA_PREFIX}-manifest"
STATUS_SCHEMA = f"{SCHEMA_PREFIX}-collection-status"
COMPLETION_SCHEMA = f"{SCHEMA_PREFIX}-collection-completion"
BUILDER_EPISODES_SCHEMA = f"{SCHEMA_PREFIX}-builder-episodes"
VERSION = 1
TEACHER_POLICY_ID = "curve-aware-settled-strategic-v3"
VERB_NAMES = ("wait", "fire", "swap", "hop")
FIRE_VERB_INDEX = VERB_NAMES.index("fire")
AIM_BINS = 180
MASK_WIDTH = len(VERB_NAMES) + AIM_BINS
# Keys must match tools.build_alphazuma_55_motor_observable_replay_v4
# MASK_SEMANTICS so the park-settle loader keeps refusing relaxed masks.
MASK_SEMANTICS = {
    "exact_masks": "exact_runtime_valid_action_mask",
    "training_masks": "relaxed_training_mask_teacher_verb_unmasked",
}
EPISODES_DIRNAME = "episodes"
SPILL_DIRNAME = "tmp-observation-spill"
DEFAULT_SEED_BASE = 1_543_004_000
DEFAULT_SEEDS_PER_LEVEL = 3
DEFAULT_PARALLEL_ENVS = 12
DEFAULT_MAX_TICKS = 30_000
# Frozen formal embargo ranges; engineering collection must stay outside.
EMBARGOED_SEED_RANGES = (
    ("formal_selection", 1_600_000_000, 1_600_000_219),
    ("formal_final_blind", 1_700_000_000, 1_700_000_439),
    ("continuous_campaign", 1_800_000_000, 1_800_000_219),
)


@dataclass(frozen=True)
class EpisodeTask:
    """One planned (level, seed) episode in the collection order."""

    level_id: str
    seed: int
    pass_index: int
    level_position: int

    @property
    def key(self) -> tuple[str, int]:
        return (self.level_id, int(self.seed))

    @property
    def file_name(self) -> str:
        return f"{self.level_id}-seed{int(self.seed)}.npz"


def _input_config() -> EliteHumanInputConfig:
    """The elite-human input profile every replay/probe route pins."""

    return EliteHumanInputConfig(
        profile_id="elite-human-v1",
        reaction_delay_ticks=12,
        max_aim_speed_degrees_per_second=1080.0,
        max_aim_acceleration_degrees_per_second_squared=18000.0,
        min_button_interval_ticks=5,
    )


def _reward_config() -> WinFirstRewardConfig:
    return WinFirstRewardConfig(
        profile_id="win-time-score-v1",
        win_reward=10.0,
        failure_reward=-10.0,
        time_penalty_per_native_tick=-0.0001,
        score_progress_reward_cap=0.01,
    )


def _validate_levels(levels: Sequence[str]) -> tuple[str, ...]:
    requested = list(levels)
    if not requested:
        raise ValueError("at least one level is required")
    unknown = [
        level for level in requested if level not in INCLUDED_LEVELS
    ]
    if unknown:
        raise ValueError(f"levels outside the frozen 55 scope: {unknown}")
    if len(set(requested)) != len(requested):
        raise ValueError("duplicate levels in the selection")
    selected = set(requested)
    return tuple(level for level in INCLUDED_LEVELS if level in selected)


def plan_tasks(
    *,
    levels: Sequence[str],
    seeds_per_level: int,
    seed_base: int,
) -> list[EpisodeTask]:
    """Pass-major task plan with the replay-anchor seed convention."""

    ordered = _validate_levels(levels)
    if int(seeds_per_level) < 1:
        raise ValueError("seeds_per_level must be positive")
    if int(seed_base) < 0:
        raise ValueError("seed_base cannot be negative")
    tasks: list[EpisodeTask] = []
    for pass_index in range(int(seeds_per_level)):
        for position, level_id in enumerate(ordered):
            tasks.append(
                EpisodeTask(
                    level_id=level_id,
                    seed=int(seed_base) + pass_index * len(ordered) + position,
                    pass_index=pass_index,
                    level_position=position,
                )
            )
    return tasks


def validate_seed_plan(tasks: Sequence[EpisodeTask]) -> None:
    """Abort before any environment work if the plan touches an embargo."""

    for task in tasks:
        for name, first, last in EMBARGOED_SEED_RANGES:
            if first <= int(task.seed) <= last:
                raise ValueError(
                    f"planned seed {int(task.seed)} for {task.level_id} "
                    f"falls inside the frozen {name} embargo "
                    f"[{first}, {last}]; choose another --seed-base"
                )


def _batches(
    tasks: Sequence[EpisodeTask], parallel_envs: int
) -> list[list[EpisodeTask]]:
    """Seed-contiguous batches: chunks never straddle a pass boundary."""

    if int(parallel_envs) < 1:
        raise ValueError("parallel_envs must be positive")
    by_pass: dict[int, list[EpisodeTask]] = {}
    for task in tasks:
        by_pass.setdefault(int(task.pass_index), []).append(task)
    batches: list[list[EpisodeTask]] = []
    for pass_index in sorted(by_pass):
        members = by_pass[pass_index]
        for start in range(0, len(members), int(parallel_envs)):
            batches.append(members[start : start + int(parallel_envs)])
    return batches


class _EpisodeRecorder:
    """Accumulates one episode; observations spill to disk immediately.

    Full-tick observations are far too large to hold for 12 concurrent
    long episodes (~2.75 GB per 30k-tick episode), so each observation
    row is appended raw to a spill file the moment it is recorded and
    memmapped back only while streaming the compressed NPZ.
    """

    def __init__(self, *, spill_path: Path, observation_width: int) -> None:
        if int(observation_width) < 1:
            raise ValueError("observation_width must be positive")
        self.spill_path = Path(spill_path)
        self.observation_width = int(observation_width)
        self.spill_path.parent.mkdir(parents=True, exist_ok=True)
        self._spill = self.spill_path.open("wb")
        self._pending_info = False
        self.ticks = 0
        self.raw_actions: list[np.ndarray] = []
        self.effective_actions: list[np.ndarray] = []
        self.training_masks: list[np.ndarray] = []
        self.exact_masks: list[np.ndarray] = []
        self.relaxed: list[bool] = []
        self.info_score: list[int] = []
        self.info_chain_length: list[int] = []
        self.info_visible_balls: list[int] = []
        self.hs_desired_verb: list[int] = []
        self.hs_desired_aim_bin: list[int] = []
        self.hs_executed_verb: list[int] = []
        self.hs_executed_aim_bin: list[int] = []
        self.hs_button_enqueued: list[bool] = []
        self.capacity_overflow = False

    def record_intent(
        self,
        *,
        observation: np.ndarray,
        raw_action: np.ndarray,
        effective_action: np.ndarray,
        training_mask: np.ndarray,
        exact_mask: np.ndarray,
        relaxed: bool,
    ) -> None:
        if self._pending_info:
            raise RuntimeError("previous tick still awaits its info record")
        row = np.ascontiguousarray(observation, dtype=np.float32).reshape(-1)
        if int(row.size) != self.observation_width:
            raise ValueError(
                f"observation width changed mid-episode: {int(row.size)} "
                f"!= {self.observation_width}"
            )
        training = np.asarray(training_mask, dtype=np.bool_).reshape(-1)
        exact = np.asarray(exact_mask, dtype=np.bool_).reshape(-1).copy()
        if training.size != exact.size:
            raise ValueError("training and exact masks disagree on width")
        self._spill.write(row.tobytes())
        self.raw_actions.append(
            np.asarray(raw_action, dtype=np.int64).reshape(2).copy()
        )
        self.effective_actions.append(
            np.asarray(effective_action, dtype=np.int64).reshape(2).copy()
        )
        self.training_masks.append(training.copy())
        self.exact_masks.append(exact)
        self.relaxed.append(bool(relaxed))
        self._pending_info = True

    def record_info(self, info: Mapping[str, Any]) -> None:
        if not self._pending_info:
            raise RuntimeError("record_intent must precede record_info")
        info = info if isinstance(info, Mapping) else {}
        speedrun = info.get("human_speedrun")
        speedrun = speedrun if isinstance(speedrun, Mapping) else {}
        self.info_score.append(int(info.get("score", 0) or 0))
        self.info_chain_length.append(int(info.get("chain_length", 0) or 0))
        self.info_visible_balls.append(int(info.get("visible_balls", 0) or 0))
        self.hs_desired_verb.append(int(speedrun.get("desired_verb", 0) or 0))
        self.hs_desired_aim_bin.append(
            int(speedrun.get("desired_aim_bin", 0) or 0)
        )
        self.hs_executed_verb.append(
            int(speedrun.get("executed_verb", 0) or 0)
        )
        self.hs_executed_aim_bin.append(
            int(speedrun.get("executed_aim_bin", 0) or 0)
        )
        self.hs_button_enqueued.append(
            bool(speedrun.get("button_enqueued", False))
        )
        self.capacity_overflow = self.capacity_overflow or bool(
            info.get("observation_capacity_overflow", False)
        )
        self._pending_info = False
        self.ticks += 1

    def write_npz(self, path: Path) -> tuple[str, int, int]:
        """Atomically write the episode NPZ; returns (sha256, ticks, bytes)."""

        if self._pending_info:
            raise RuntimeError("last tick is missing its info record")
        if self.ticks < 1:
            raise ValueError("episode recorded no ticks")
        path = Path(path).resolve()
        if path.exists():
            raise FileExistsError(f"episode output already exists: {path}")
        self._spill.flush()
        os.fsync(self._spill.fileno())
        self._spill.close()
        observations = np.memmap(
            self.spill_path,
            dtype=np.float32,
            mode="r",
            shape=(self.ticks, self.observation_width),
        )
        temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
        try:
            np.savez_compressed(
                temporary,
                observations=observations,
                raw_actions=np.stack(self.raw_actions),
                effective_actions=np.stack(self.effective_actions),
                training_masks=np.stack(self.training_masks),
                exact_masks=np.stack(self.exact_masks),
                intent_mask_relaxed=np.asarray(self.relaxed, dtype=np.bool_),
                tick_indices=np.arange(self.ticks, dtype=np.int64),
                info_score=np.asarray(self.info_score, dtype=np.int64),
                info_chain_length=np.asarray(
                    self.info_chain_length, dtype=np.int64
                ),
                info_visible_balls=np.asarray(
                    self.info_visible_balls, dtype=np.int64
                ),
                hs_desired_verb=np.asarray(
                    self.hs_desired_verb, dtype=np.int64
                ),
                hs_desired_aim_bin=np.asarray(
                    self.hs_desired_aim_bin, dtype=np.int64
                ),
                hs_executed_verb=np.asarray(
                    self.hs_executed_verb, dtype=np.int64
                ),
                hs_executed_aim_bin=np.asarray(
                    self.hs_executed_aim_bin, dtype=np.int64
                ),
                hs_button_enqueued=np.asarray(
                    self.hs_button_enqueued, dtype=np.bool_
                ),
            )
            del observations
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
        self.spill_path.unlink(missing_ok=True)
        return legacy._sha256(path), int(self.ticks), int(path.stat().st_size)

    def discard(self) -> None:
        """Drop an unfinished episode's spill without writing anything."""

        try:
            if not self._spill.closed:
                self._spill.close()
        finally:
            self.spill_path.unlink(missing_ok=True)


def _fire_edge_count(raw_verbs: np.ndarray) -> int:
    """Rising edges in the RAW intent verb stream (builder semantics)."""

    fire = np.asarray(raw_verbs, dtype=np.int64) == FIRE_VERB_INDEX
    previous = np.concatenate(([False], fire[:-1]))
    return int(np.count_nonzero(fire & ~previous))


def _finalize_episode(
    *,
    task: EpisodeTask,
    recorder: _EpisodeRecorder,
    final_info: Mapping[str, Any],
    guard_exhausted: bool,
    episodes_dir: Path,
    wall_seconds: float,
) -> dict[str, Any]:
    """Write the episode NPZ and return its manifest/completion entry.

    Outcome honesty: ``win``/``loss`` come from the environment; anything
    else that ended by time limit (or by the collection guard) is
    recorded as ``truncated``, never dropped.
    """

    final_info = final_info if isinstance(final_info, Mapping) else {}
    outcome_raw = final_info.get("outcome")
    truncated = bool(final_info.get("TimeLimit.truncated", False)) or bool(
        guard_exhausted
    )
    if outcome_raw in ("win", "loss"):
        outcome = str(outcome_raw)
    elif truncated:
        outcome = "truncated"
    elif outcome_raw:
        outcome = str(outcome_raw)
    else:
        outcome = "unknown"
    path = episodes_dir / task.file_name
    digest, tick_count, npz_bytes = recorder.write_npz(path)
    raw_verbs = np.stack(recorder.raw_actions)[:, 0]
    effective_verbs = np.stack(recorder.effective_actions)[:, 0]
    exact = np.stack(recorder.exact_masks)
    runtime_illegal = int(
        np.count_nonzero(~exact[np.arange(tick_count), raw_verbs])
    )
    raw_counts = Counter(VERB_NAMES[int(verb)] for verb in raw_verbs)
    effective_counts = Counter(
        VERB_NAMES[int(verb)] for verb in effective_verbs
    )
    score = int(final_info.get("score", 0) or 0)
    if not score and recorder.info_score:
        score = int(recorder.info_score[-1])
    return {
        "level_id": task.level_id,
        "seed": int(task.seed),
        "pass_index": int(task.pass_index),
        "outcome": outcome,
        "time_limit_truncated": truncated,
        "guard_exhausted": bool(guard_exhausted),
        "observation_capacity_overflow": bool(
            recorder.capacity_overflow
            or final_info.get("observation_capacity_overflow", False)
        ),
        "score": score,
        "tick_count": int(tick_count),
        "raw_intent_action_counts": dict(raw_counts),
        "effective_execution_action_counts": dict(effective_counts),
        "intent_mask_relaxations": int(sum(recorder.relaxed)),
        "runtime_illegal_raw_verb_ticks": runtime_illegal,
        "fire_edges_raw_intent": _fire_edge_count(raw_verbs),
        "path": f"{EPISODES_DIRNAME}/{task.file_name}",
        "sha256": digest,
        "npz_bytes": int(npz_bytes),
        "wall_seconds": float(wall_seconds),
    }


def _collect_batch(
    *,
    original_root: Path,
    tasks: Sequence[EpisodeTask],
    record_flags: Sequence[bool],
    max_ticks: int,
    episodes_dir: Path,
    spill_dir: Path,
    on_episode: Callable[[dict[str, Any]], None],
) -> list[dict[str, Any]]:
    """Run one seed-contiguous batch of teacher episodes.

    Mirrors ``distill_alphazuma_55_polar_intent_wide_v1
    ._collect_intent_round`` (SubprocVecEnv, ``vector.seed(first_seed)``,
    per-env settled strategic teachers from ``teacher_spec()``) on the
    motor-observable wrapper stack, but keeps EVERY tick instead of
    reservoir subsampling.  Episodes whose ``record_flags`` entry is
    False are stepped (the vector needs the full batch) but not recorded.
    """

    from sb3_contrib.common.maskable.utils import get_action_masks
    from stable_baselines3.common.vec_env import SubprocVecEnv

    count = len(tasks)
    if count != len(record_flags):
        raise ValueError("record_flags must align with tasks")
    first_seed = int(tasks[0].seed)
    for offset, task in enumerate(tasks):
        if int(task.seed) != first_seed + offset:
            raise ValueError(
                "batch seeds must be contiguous "
                "(replay-anchor vector seeding convention)"
            )
    factories = [
        motor._make_motor_env_factory(
            original_root=original_root,
            level_id=task.level_id,
            max_ticks=int(max_ticks),
            input_config=_input_config(),
            reward_config=_reward_config(),
        )
        for task in tasks
    ]
    vector = SubprocVecEnv(factories, start_method="forkserver")
    recorders: list[_EpisodeRecorder | None] = [None] * count
    entries: list[dict[str, Any]] = []
    try:
        vector.seed(first_seed)
        observations = np.asarray(vector.reset(), dtype=np.float32)
        specifications = vector.env_method("teacher_spec")
        teachers = [
            CurveAwareSettledStrategicRevengeTeacher(
                legacy._teacher_spec(value)
            )
            for value in specifications
        ]
        width = int(observations.shape[1])
        for position in range(count):
            if bool(record_flags[position]):
                recorders[position] = _EpisodeRecorder(
                    spill_path=spill_dir
                    / f"{tasks[position].file_name}.observations.spill",
                    observation_width=width,
                )
        batch_started = time.perf_counter()
        finished = np.zeros(count, dtype=np.bool_)
        last_infos: list[dict[str, Any]] = [{} for _ in range(count)]

        def finalize(index: int, info: Mapping[str, Any], guard: bool) -> None:
            recorder = recorders[index]
            if recorder is None:
                return
            recorders[index] = None
            entry = _finalize_episode(
                task=tasks[index],
                recorder=recorder,
                final_info=info,
                guard_exhausted=guard,
                episodes_dir=episodes_dir,
                wall_seconds=time.perf_counter() - batch_started,
            )
            entries.append(entry)
            on_episode(entry)

        for _ in range(int(max_ticks) + 1):
            masks = np.asarray(get_action_masks(vector), dtype=np.bool_)
            actions = np.zeros((count, 2), dtype=np.int64)
            active = ~finished
            for position in np.flatnonzero(active):
                index = int(position)
                teacher_raw = np.asarray(
                    teachers[index].act(observations[index]),
                    dtype=np.int64,
                )
                raw, effective, training_mask, forced = intent._intent_label(
                    teacher_raw, masks[index]
                )
                actions[index] = effective
                recorder = recorders[index]
                if recorder is not None:
                    recorder.record_intent(
                        observation=observations[index],
                        raw_action=raw,
                        effective_action=effective,
                        training_mask=training_mask,
                        exact_mask=masks[index],
                        relaxed=forced,
                    )
            step_observations, _, dones, infos = vector.step(actions)
            for position in np.flatnonzero(active):
                index = int(position)
                info = dict(infos[index])
                recorder = recorders[index]
                if recorder is not None:
                    recorder.record_info(info)
                last_infos[index] = info
                if bool(dones[index]):
                    finished[index] = True
                    finalize(index, info, False)
            observations = np.asarray(step_observations, dtype=np.float32)
            if bool(np.all(finished)):
                break
        for position in range(count):
            if not finished[position]:
                # Guard exhausted without done: record honestly, never drop.
                finalize(int(position), last_infos[int(position)], True)
    finally:
        for recorder in recorders:
            if recorder is not None:
                recorder.discard()
        vector.close()
    return entries


def _write_manifest(
    run_dir: Path,
    *,
    levels: Sequence[str],
    seeds_per_level: int,
    seed_base: int,
    max_ticks: int,
    entries_by_key: Mapping[tuple[str, int], Mapping[str, Any]],
) -> Path:
    """Rewrite the episodes manifest atomically, level-major order."""

    level_positions = {level: index for index, level in enumerate(levels)}
    ordered = sorted(
        entries_by_key.values(),
        key=lambda entry: (
            level_positions.get(str(entry["level_id"]), len(levels)),
            int(entry["seed"]),
        ),
    )
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "version": VERSION,
        "builder_contract": (
            "tools/build_alphazuma_55_motor_observable_replay_v4.py episode "
            "records; hydrate via collect_alphazuma_55_park_settle_episodes_"
            "v1.load_episode_records or --emit-builder-episodes"
        ),
        "teacher_policy_id": TEACHER_POLICY_ID,
        "environment_stack": (
            "RevengeEnv + ObservableHumanSpeedrunWrapper (motor-observable)"
        ),
        "training_label_semantics": "raw_teacher_intent",
        "environment_execution_semantics": "exact_mask_effective_action",
        "mask_semantics": dict(MASK_SEMANTICS),
        "levels": list(levels),
        "seeds_per_level": int(seeds_per_level),
        "seed_base": int(seed_base),
        "max_ticks": int(max_ticks),
        "formal_seed_consumption": False,
        "episodes": [dict(entry) for entry in ordered],
    }
    path = run_dir / "episodes_manifest.json"
    legacy._write_json_atomic(path, manifest)
    return path


def _load_completed_entries(
    run_dir: Path,
) -> dict[tuple[str, int], dict[str, Any]]:
    """Verified entries from a previous run: file present, sha256 intact."""

    path = run_dir / "episodes_manifest.json"
    if not path.exists():
        return {}
    manifest = legacy._read_json(path)
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"run dir holds a foreign manifest: {path}")
    completed: dict[tuple[str, int], dict[str, Any]] = {}
    for entry in manifest.get("episodes", []):
        episode_path = run_dir / str(entry.get("path", ""))
        if not episode_path.is_file():
            continue
        if legacy._sha256(episode_path) != entry.get("sha256"):
            continue
        if int(entry.get("tick_count", 0)) < 1:
            continue
        completed[(str(entry["level_id"]), int(entry["seed"]))] = dict(entry)
    return completed


def _quarantine_unverified_episode_files(
    episodes_dir: Path,
    verified_paths: set[Path],
) -> list[dict[str, str]]:
    """Quarantine-rename episode files that lack a verified receipt.

    Resume self-healing: a corrupted-in-place NPZ (sha256 receipt
    mismatch) or an orphaned NPZ (a crash in the window between the
    atomic ``os.replace`` of the NPZ and the manifest rewrite) leaves a
    file at exactly the path its re-collection must write to.
    ``_EpisodeRecorder.write_npz`` correctly refuses to overwrite
    existing files, so without this sweep the resume raises
    ``FileExistsError``, the exception aborts the whole batch (discarding
    every other in-flight recorder in it), and every subsequent resume
    wedges the same way until someone deletes the bad file by hand.

    Unverified bytes are renamed aside (never silently deleted -- they
    may matter forensically) BEFORE any batch runs, so the run always
    self-heals.  Already-quarantined files no longer match ``*.npz`` and
    are left untouched.
    """

    quarantined: list[dict[str, str]] = []
    if not episodes_dir.is_dir():
        return quarantined
    for path in sorted(episodes_dir.glob("*.npz")):
        if not path.is_file():
            continue
        if path.resolve() in verified_paths:
            continue
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        attempt = 0
        while True:
            tag = f"{stamp}-{os.getpid()}" + (
                f"-{attempt}" if attempt else ""
            )
            target = path.with_name(f"{path.name}.quarantined-{tag}")
            if not target.exists():
                break
            attempt += 1
        os.replace(path, target)
        quarantined.append(
            {"original": path.name, "quarantined_as": target.name}
        )
    return quarantined


def collect(
    *,
    original_root: Path,
    run_dir: Path,
    levels: Sequence[str] = INCLUDED_LEVELS,
    seeds_per_level: int = DEFAULT_SEEDS_PER_LEVEL,
    seed_base: int = DEFAULT_SEED_BASE,
    parallel_envs: int = DEFAULT_PARALLEL_ENVS,
    max_ticks: int = DEFAULT_MAX_TICKS,
    device: str = "cpu",
) -> dict[str, Any]:
    """Collect (or resume) the full-tick teacher episode archive."""

    started = time.perf_counter()
    original_root = Path(original_root).expanduser().resolve(strict=True)
    run_dir = Path(run_dir).expanduser().resolve()
    ordered_levels = _validate_levels(levels)
    if int(max_ticks) < 1:
        raise ValueError("max_ticks must be positive")
    tasks = plan_tasks(
        levels=ordered_levels,
        seeds_per_level=int(seeds_per_level),
        seed_base=int(seed_base),
    )
    validate_seed_plan(tasks)
    episodes_dir = run_dir / EPISODES_DIRNAME
    spill_dir = run_dir / SPILL_DIRNAME
    episodes_dir.mkdir(parents=True, exist_ok=True)
    spill_dir.mkdir(parents=True, exist_ok=True)
    for stale in spill_dir.glob("*.spill"):
        stale.unlink()

    plan = {
        "levels": list(ordered_levels),
        "seeds_per_level": int(seeds_per_level),
        "seed_base": int(seed_base),
        "max_ticks": int(max_ticks),
    }
    config_path = run_dir / "config.json"
    if config_path.exists():
        existing = legacy._read_json(config_path)
        previous = {key: existing.get(key) for key in plan}
        if previous != plan:
            raise ValueError(
                f"run dir {run_dir} was planned with a different collection "
                f"contract; refusing to mix plans: {previous} != {plan}"
            )
    else:
        legacy._write_json_atomic(
            config_path,
            {
                "schema": CONFIG_SCHEMA,
                "version": VERSION,
                "collector": {
                    "path": str(SCRIPT_PATH),
                    "sha256": legacy._sha256(SCRIPT_PATH),
                },
                "builder_contract": {
                    "path": str(BUILDER_PATH),
                    "sha256": legacy._sha256(BUILDER_PATH),
                },
                "teacher_policy_id": TEACHER_POLICY_ID,
                "teacher_construction": (
                    "deterministic code built from each environment's "
                    "teacher_spec(); no checkpoint file exists"
                ),
                "original_root": str(original_root),
                "device": str(device),
                "parallel_envs": int(parallel_envs),
                "input_profile": _input_config().profile_id,
                "reward_profile": _reward_config().profile_id,
                "environment_stack": (
                    "RevengeEnv + ObservableHumanSpeedrunWrapper "
                    "(motor-observable)"
                ),
                "seed_last": int(seed_base)
                + int(seeds_per_level) * len(ordered_levels)
                - 1,
                "seed_embargo": {
                    name: [first, last]
                    for name, first, last in EMBARGOED_SEED_RANGES
                },
                "formal_seed_consumption": False,
                **plan,
            },
        )

    entries_by_key = _load_completed_entries(run_dir)
    reused_keys = set(entries_by_key)
    # Self-heal resume: any episode file WITHOUT a verified receipt
    # (corrupted in place, or orphaned by a crash before the manifest
    # rewrite) blocks its own re-collection via write_npz's overwrite
    # refusal.  Quarantine-rename such files before any batch runs.
    verified_paths = {
        (run_dir / str(entry["path"])).resolve()
        for entry in entries_by_key.values()
    }
    quarantined_files = _quarantine_unverified_episode_files(
        episodes_dir, verified_paths
    )

    def write_manifest() -> Path:
        return _write_manifest(
            run_dir,
            levels=ordered_levels,
            seeds_per_level=int(seeds_per_level),
            seed_base=int(seed_base),
            max_ticks=int(max_ticks),
            entries_by_key=entries_by_key,
        )

    def on_episode(entry: dict[str, Any]) -> None:
        entries_by_key[(str(entry["level_id"]), int(entry["seed"]))] = entry
        write_manifest()

    manifest_path = write_manifest()
    errors: list[dict[str, Any]] = []
    batches = _batches(tasks, int(parallel_envs))
    for batch_index, batch in enumerate(batches):
        record_flags = [task.key not in entries_by_key for task in batch]
        if not any(record_flags):
            continue
        legacy._write_json_atomic(
            run_dir / "status.json",
            {
                "schema": STATUS_SCHEMA,
                "version": VERSION,
                "status": "RUNNING",
                "updated_utc": legacy._utc_now(),
                "batch_index": batch_index,
                "batch_count": len(batches),
                "batch_levels": [task.level_id for task in batch],
                "batch_first_seed": int(batch[0].seed),
                "completed_episodes": len(entries_by_key),
                "total_tasks": len(tasks),
                "errors": len(errors),
                "wall_seconds": time.perf_counter() - started,
            },
        )
        try:
            _collect_batch(
                original_root=original_root,
                tasks=batch,
                record_flags=record_flags,
                max_ticks=int(max_ticks),
                episodes_dir=episodes_dir,
                spill_dir=spill_dir,
                on_episode=on_episode,
            )
        except Exception as error:  # noqa: BLE001 - keep other batches alive
            errors.append(
                {
                    "batch_levels": [task.level_id for task in batch],
                    "batch_first_seed": int(batch[0].seed),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )

    manifest_path = write_manifest()
    ordered_entries = [
        dict(entries_by_key[task.key])
        for task in tasks
        if task.key in entries_by_key
    ]
    for entry in ordered_entries:
        entry["reused"] = (str(entry["level_id"]), int(entry["seed"])) in (
            reused_keys
        )
    missing = [
        {"level_id": task.level_id, "seed": int(task.seed)}
        for task in tasks
        if task.key not in entries_by_key
    ]
    per_level: dict[str, dict[str, Any]] = {}
    for level_id in ordered_levels:
        members = [
            entry
            for entry in ordered_entries
            if entry["level_id"] == level_id
        ]
        per_level[level_id] = {
            "episodes": len(members),
            "wins": sum(entry["outcome"] == "win" for entry in members),
            "losses": sum(entry["outcome"] == "loss" for entry in members),
            "truncations": sum(
                bool(entry["time_limit_truncated"]) for entry in members
            ),
            "ticks": sum(int(entry["tick_count"]) for entry in members),
            "scores": [int(entry["score"]) for entry in members],
            "seeds": [int(entry["seed"]) for entry in members],
        }
    total_ticks = sum(int(entry["tick_count"]) for entry in ordered_entries)
    total_bytes = sum(int(entry["npz_bytes"]) for entry in ordered_entries)
    completion = {
        "schema": COMPLETION_SCHEMA,
        "version": VERSION,
        "status": (
            "COMPLETE" if not missing and not errors else "PARTIAL"
        ),
        "completed_utc": legacy._utc_now(),
        "wall_seconds": time.perf_counter() - started,
        "collector": {
            "path": str(SCRIPT_PATH),
            "sha256": legacy._sha256(SCRIPT_PATH),
        },
        "teacher_policy_id": TEACHER_POLICY_ID,
        "original_root": str(original_root),
        "run_dir": str(run_dir),
        "device": str(device),
        "parallel_envs": int(parallel_envs),
        "levels": list(ordered_levels),
        "seeds_per_level": int(seeds_per_level),
        "seed_base": int(seed_base),
        "seed_last": int(seed_base)
        + int(seeds_per_level) * len(ordered_levels)
        - 1,
        "max_ticks": int(max_ticks),
        "seed_embargo": {
            name: [first, last] for name, first, last in EMBARGOED_SEED_RANGES
        },
        "formal_seed_consumption": False,
        "episodes": ordered_entries,
        "missing_episodes": missing,
        "resume_quarantined_files": quarantined_files,
        "errors": errors,
        "wins": sum(entry["outcome"] == "win" for entry in ordered_entries),
        "losses": sum(
            entry["outcome"] == "loss" for entry in ordered_entries
        ),
        "truncations": sum(
            bool(entry["time_limit_truncated"]) for entry in ordered_entries
        ),
        "guard_exhaustions": sum(
            bool(entry["guard_exhausted"]) for entry in ordered_entries
        ),
        "capacity_overflows": sum(
            bool(entry["observation_capacity_overflow"])
            for entry in ordered_entries
        ),
        "per_level": per_level,
        "total_ticks": total_ticks,
        "total_npz_bytes": total_bytes,
        "episodes_manifest": {
            "path": str(manifest_path),
            "sha256": legacy._sha256(manifest_path),
        },
    }
    legacy._write_json_atomic(run_dir / "completion.json", completion)
    legacy._write_json_atomic(
        run_dir / "status.json",
        {
            "schema": STATUS_SCHEMA,
            "version": VERSION,
            "status": completion["status"],
            "updated_utc": legacy._utc_now(),
            "completed_episodes": len(ordered_entries),
            "total_tasks": len(tasks),
            "errors": len(errors),
            "wall_seconds": completion["wall_seconds"],
        },
    )
    return completion


def _tick_info(
    arrays: Mapping[str, np.ndarray], tick_index: int
) -> dict[str, Any]:
    return {
        "score": int(arrays["info_score"][tick_index]),
        "chain_length": int(arrays["info_chain_length"][tick_index]),
        "visible_balls": int(arrays["info_visible_balls"][tick_index]),
        "human_speedrun": {
            "desired_verb": int(arrays["hs_desired_verb"][tick_index]),
            "desired_aim_bin": int(arrays["hs_desired_aim_bin"][tick_index]),
            "executed_verb": int(arrays["hs_executed_verb"][tick_index]),
            "executed_aim_bin": int(
                arrays["hs_executed_aim_bin"][tick_index]
            ),
            "button_enqueued": bool(arrays["hs_button_enqueued"][tick_index]),
        },
    }


_EPISODE_ARRAY_KEYS = (
    "observations",
    "raw_actions",
    "effective_actions",
    "training_masks",
    "exact_masks",
    "intent_mask_relaxed",
    "info_score",
    "info_chain_length",
    "info_visible_balls",
    "hs_desired_verb",
    "hs_desired_aim_bin",
    "hs_executed_verb",
    "hs_executed_aim_bin",
    "hs_button_enqueued",
)


def _read_episode_arrays(
    path: Path, *, expected_sha256: str | None
) -> dict[str, np.ndarray]:
    if expected_sha256 is not None:
        digest = legacy._sha256(path)
        if digest != expected_sha256:
            raise ValueError(
                f"episode bytes differ from the manifest receipt: {path}"
            )
    with np.load(path) as archive:
        missing = [
            key for key in _EPISODE_ARRAY_KEYS if key not in archive.files
        ]
        if missing:
            raise ValueError(f"episode NPZ misses {missing}: {path}")
        arrays = {key: archive[key] for key in _EPISODE_ARRAY_KEYS}
    count = int(arrays["observations"].shape[0])
    for key, value in arrays.items():
        if int(value.shape[0]) != count:
            raise ValueError(f"episode array {key} misaligned: {path}")
    return arrays


def _hydrate_episode(
    manifest_dir: Path,
    entry: Mapping[str, Any],
    *,
    verify_sha256: bool,
) -> dict[str, Any]:
    path = Path(str(entry["path"]))
    if not path.is_absolute():
        path = manifest_dir / path
    path = path.resolve(strict=True)
    arrays = _read_episode_arrays(
        path,
        expected_sha256=(
            str(entry["sha256"]) if verify_sha256 and "sha256" in entry
            else None
        ),
    )
    count = int(arrays["observations"].shape[0])
    ticks = [
        {
            "observation": arrays["observations"][tick_index],
            "raw_action": arrays["raw_actions"][tick_index],
            "training_mask": arrays["training_masks"][tick_index],
            "exact_mask": arrays["exact_masks"][tick_index],
            "effective_action": arrays["effective_actions"][tick_index],
            "intent_mask_relaxed": bool(
                arrays["intent_mask_relaxed"][tick_index]
            ),
            "info": _tick_info(arrays, tick_index),
        }
        for tick_index in range(count)
    ]
    return {
        "level_id": str(entry["level_id"]),
        "seed": int(entry["seed"]),
        "outcome": str(entry.get("outcome", "")),
        "time_limit_truncated": bool(entry.get("time_limit_truncated", False)),
        "observation_capacity_overflow": bool(
            entry.get("observation_capacity_overflow", False)
        ),
        "score": int(entry.get("score", 0)),
        "ticks": ticks,
    }


def _manifest_entries(manifest_path: Path) -> list[dict[str, Any]]:
    manifest = legacy._read_json(manifest_path)
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"not a park-settle episodes manifest: {manifest_path}")
    entries = manifest.get("episodes")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"manifest lists no episodes: {manifest_path}")
    return entries


def iter_episode_records(
    manifest_path: Path, *, verify_sha256: bool = True
) -> Iterator[dict[str, Any]]:
    """Hydrate episodes one at a time into the builder's record contract."""

    manifest_path = Path(manifest_path).resolve(strict=True)
    for entry in _manifest_entries(manifest_path):
        yield _hydrate_episode(
            manifest_path.parent, entry, verify_sha256=verify_sha256
        )


def load_episode_records(
    manifest_path: Path, *, verify_sha256: bool = True
) -> list[dict[str, Any]]:
    """Builder-ready episode records (numpy ticks) for
    ``build_alphazuma_55_motor_observable_replay_v4
    .build_coverage_equalized_replay_dataset``.

    Loads everything into memory: fine for smoke/subset runs; iterate
    ``iter_episode_records`` for anything larger.
    """

    return list(
        iter_episode_records(manifest_path, verify_sha256=verify_sha256)
    )


def emit_builder_episodes_json(
    *,
    manifest_path: Path,
    output_path: Path,
    verify_sha256: bool = True,
) -> dict[str, Any]:
    """Write the builder's inline ``{"episodes": [...]}`` JSON document.

    Streamed tick by tick so memory stays bounded; observation floats
    round-trip bit-exactly through the float32 -> repr(double) -> float32
    path the builder's ``np.asarray(..., dtype=np.float32)`` applies.
    The output must not already exist (frozen-output discipline).
    """

    manifest_path = Path(manifest_path).resolve(strict=True)
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise FileExistsError(f"output already exists: {output_path}")
    entries = _manifest_entries(manifest_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(
        f".{output_path.name}.{os.getpid()}.tmp"
    )
    total_ticks = 0
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write("{")
            stream.write(f'"schema": {json.dumps(BUILDER_EPISODES_SCHEMA)}, ')
            stream.write(f'"version": {VERSION}, ')
            stream.write(
                f'"teacher_policy_id": {json.dumps(TEACHER_POLICY_ID)}, '
            )
            stream.write(
                f'"source_manifest": {json.dumps(str(manifest_path))}, '
            )
            stream.write('"episodes": [\n')
            for index, entry in enumerate(entries):
                if index:
                    stream.write(",\n")
                path = Path(str(entry["path"]))
                if not path.is_absolute():
                    path = manifest_path.parent / path
                arrays = _read_episode_arrays(
                    path.resolve(strict=True),
                    expected_sha256=(
                        str(entry["sha256"])
                        if verify_sha256 and "sha256" in entry
                        else None
                    ),
                )
                count = int(arrays["observations"].shape[0])
                header = {
                    "level_id": str(entry["level_id"]),
                    "seed": int(entry["seed"]),
                    "outcome": str(entry.get("outcome", "")),
                    "time_limit_truncated": bool(
                        entry.get("time_limit_truncated", False)
                    ),
                    "observation_capacity_overflow": bool(
                        entry.get("observation_capacity_overflow", False)
                    ),
                    "score": int(entry.get("score", 0)),
                }
                stream.write("{")
                for key, value in header.items():
                    stream.write(
                        f"{json.dumps(key)}: {json.dumps(value)}, "
                    )
                stream.write('"ticks": [')
                for tick_index in range(count):
                    if tick_index:
                        stream.write(",")
                    json.dump(
                        {
                            "observation": arrays["observations"][
                                tick_index
                            ].tolist(),
                            "raw_action": arrays["raw_actions"][
                                tick_index
                            ].tolist(),
                            "training_mask": arrays["training_masks"][
                                tick_index
                            ].tolist(),
                            "exact_mask": arrays["exact_masks"][
                                tick_index
                            ].tolist(),
                            "effective_action": arrays["effective_actions"][
                                tick_index
                            ].tolist(),
                            "intent_mask_relaxed": bool(
                                arrays["intent_mask_relaxed"][tick_index]
                            ),
                            "info": _tick_info(arrays, tick_index),
                        },
                        stream,
                        allow_nan=False,
                    )
                stream.write("]}")
                total_ticks += count
            stream.write("\n]}\n")
            stream.flush()
            os.fsync(stream.fileno())
        if output_path.exists():
            raise FileExistsError(f"output already exists: {output_path}")
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "path": str(output_path),
        "sha256": legacy._sha256(output_path),
        "episodes": len(entries),
        "ticks": int(total_ticks),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--original-root",
        type=Path,
        default=None,
        help="retail extraction root (required unless --hydrate-only)",
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--levels",
        nargs="+",
        default=list(INCLUDED_LEVELS),
        help="subset filter; ordering is normalized to the frozen 55 order",
    )
    parser.add_argument(
        "--seeds-per-level", type=int, default=DEFAULT_SEEDS_PER_LEVEL
    )
    parser.add_argument("--seed-base", type=int, default=DEFAULT_SEED_BASE)
    parser.add_argument(
        "--parallel-envs", type=int, default=DEFAULT_PARALLEL_ENVS
    )
    parser.add_argument("--max-ticks", type=int, default=DEFAULT_MAX_TICKS)
    parser.add_argument(
        "--device",
        default="cpu",
        help=(
            "recorded for the receipt only; the settled strategic teacher "
            "is deterministic numpy code and needs no GPU"
        ),
    )
    parser.add_argument(
        "--emit-builder-episodes",
        type=Path,
        default=None,
        help=(
            "after collection, write the builder's inline episodes JSON "
            "here (smoke/subset scale only; ~90 bytes per observation "
            "float at full scale)"
        ),
    )
    parser.add_argument(
        "--hydrate-only",
        action="store_true",
        help=(
            "skip collection; only emit --emit-builder-episodes from the "
            "run dir's existing episodes manifest"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_dir = args.run_dir.expanduser().resolve()
    if args.hydrate_only:
        if args.emit_builder_episodes is None:
            parser.error("--hydrate-only requires --emit-builder-episodes")
        receipt = emit_builder_episodes_json(
            manifest_path=run_dir / "episodes_manifest.json",
            output_path=args.emit_builder_episodes,
        )
        print(
            json.dumps(
                {"status": "HYDRATED", "builder_episodes": receipt},
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        )
        return 0
    if args.original_root is None:
        parser.error("--original-root is required to collect episodes")
    completion = collect(
        original_root=args.original_root,
        run_dir=run_dir,
        levels=args.levels,
        seeds_per_level=int(args.seeds_per_level),
        seed_base=int(args.seed_base),
        parallel_envs=int(args.parallel_envs),
        max_ticks=int(args.max_ticks),
        device=str(args.device),
    )
    builder_receipt = None
    if args.emit_builder_episodes is not None:
        builder_receipt = emit_builder_episodes_json(
            manifest_path=run_dir / "episodes_manifest.json",
            output_path=args.emit_builder_episodes,
        )
    print(
        json.dumps(
            {
                "status": completion["status"],
                "episodes": len(completion["episodes"]),
                "missing_episodes": len(completion["missing_episodes"]),
                "errors": completion["errors"],
                "wins": completion["wins"],
                "losses": completion["losses"],
                "truncations": completion["truncations"],
                "total_ticks": completion["total_ticks"],
                "total_npz_bytes": completion["total_npz_bytes"],
                "run_dir": completion["run_dir"],
                "episodes_manifest": completion["episodes_manifest"],
                "builder_episodes": builder_receipt,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    return 0 if completion["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
