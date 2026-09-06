"""Teacher MACRO-DECISION collector for park-settle BC distillation.

Why this exists (verified 2026-08-17)
-------------------------------------
The teacher through the park-settle macro protocol wins 50/55 with 99.8%
shot execution (``probe_alphazuma_55_macro_teacher_v1``), while five
macro-PPO runs failed to discover or consolidate wins by exploration
(receipts in ``diagnostics/alphazuma-55-macro-ppo-*-outcome-v1.json``).
The historically winning recipe was dense TEACHER anchoring, so this
collector produces the raw material for supervising a macro-native
student DIRECTLY on the teacher's macro decisions: no exploration
lottery, one label per macro decision point.

What is recorded
----------------
The teacher plays through the STUDENT'S EXACT RUNTIME STACK -- the
motor-observable env wrapped by ``ObservationStackWrapper`` (lags
``0,4,8``) UNDER ``ParkSettleActionWrapper``
(``probe_alphazuma_55_macro_native_eval_v1._make_stacked_macro_env_
factory``, the proven trainer composition) -- while DECIDING on the RAW
current frame: the leading ``raw_observation_dim`` slice of each stacked
observation, stripped by probe-v5's ``RawFrameTeacher`` before
``teacher.act``.  Each per-tick teacher proposal is mapped through the
SAME adapter the teacher ceiling probe used
(``probe_alphazuma_55_macro_eval_v1.adapt_per_tick_action``):

* wait -> wait_hold; the teacher's wait action CARRIES its parked
  target bin, so the recorded macro parks exactly that bin;
* fire -> (fire, target); swap -> (swap, target);
* hop (per-tick verb 3) has no macro and falls back to wait_hold,
  counted per episode under ``mask_fallbacks`` and flagged per decision
  in ``fallback_codes`` -- never hidden;
* a macro-mask-illegal verb equally falls back to wait_hold, counted.

Per macro DECISION the NPZ stores: the STACKED observation the student
will see at runtime (float32), the adapter-mapped teacher macro action
``[verb, bin]`` (int64), the wrapper's own macro action mask (183
bool), the native tick index within the episode at the decision point,
the raw per-tick teacher proposal, and the fallback code.  Per EPISODE
the manifest stores level, seed, HONEST outcome (teacher losses are
recorded and flagged via ``teacher_won``; their decisions are stored
all the same), native ticks, shots, and shots-on-target.

Storage layout (one compressed NPZ per episode + receipts)
----------------------------------------------------------
Mirrors ``collect_alphazuma_55_park_settle_episodes_v1``: episode NPZs
are written atomically (spill -> temp -> ``os.replace``) the moment each
episode finishes, ``episodes_manifest.json`` is rewritten atomically
after every episode with a sha256 receipt per NPZ, resume verifies
receipts and skips completed episodes, and unverified NPZ bytes are
quarantine-renamed (never deleted) before re-collection.  The decision
volume is small: a 3k-tick teacher episode holds only ~300-500 macro
decisions (~130 MB raw stacked float32), and the mostly-zero
observations compress well.

Seed discipline
---------------
Default ``--seed-base`` 1_494_000_000: validated against the frozen
formal embargo ranges by the reused v1 validator AND refused if it
overlaps engineering blocks already consumed by sibling tools (probe-v4
1_400_920_xxx, teacher collection 1_543_004_xxx, DAgger 1_543_005_xxx,
macro-PPO engineering 1_460_000_000-1_493_999_999).
``formal_seed_consumption`` is ``false`` in every receipt; this
collector carries no training or formal-selection authority itself.

``--device`` is recorded for orchestration parity only: the teacher is
deterministic CPU code (built from each environment's
``teacher_spec()``; no checkpoint file exists) and no GPU is touched.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Mapping, Sequence

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import numpy as np

from tools import collect_alphazuma_55_park_settle_episodes_v1 as episodes_v1
from tools import distill_alphazuma_55 as legacy
from tools import probe_alphazuma_55_macro_eval_v1 as macro_eval
from tools import probe_alphazuma_55_macro_native_eval_v1 as native_eval
from tools import probe_alphazuma_55_recovery_intervention_v4 as v4
from tools import probe_alphazuma_55_recovery_intervention_v5 as v5
from zuma_rl.alphazuma_55 import INCLUDED_LEVELS
from zuma_rl.observation_stack_wrapper import (
    DEFAULT_STACK_LAGS,
    parse_stack_lags,
)
from zuma_rl.park_settle_action_wrapper import MACRO_VERB_NAMES


SCRIPT_PATH = Path(__file__).resolve()
SCHEMA_PREFIX = "zuma-rl.alphazuma-55-macro-decisions"
CONFIG_SCHEMA = f"{SCHEMA_PREFIX}-collection-config"
MANIFEST_SCHEMA = f"{SCHEMA_PREFIX}-manifest"
STATUS_SCHEMA = f"{SCHEMA_PREFIX}-collection-status"
COMPLETION_SCHEMA = f"{SCHEMA_PREFIX}-collection-completion"
VERSION = 1
TEACHER_POLICY_ID = episodes_v1.TEACHER_POLICY_ID
TEACHER_ID = v4.TEACHER_ID
AIM_BINS = macro_eval.AIM_BINS
MACRO_VERB_COUNT = len(MACRO_VERB_NAMES)
MACRO_MASK_WIDTH = MACRO_VERB_COUNT + AIM_BINS
MACRO_FIRE_NAME = "fire"
# Per-decision fallback codes stored in the NPZ (int8).
FALLBACK_CODE_NONE = 0
FALLBACK_CODE_UNSUPPORTED = 1  # per-tick hop: no macro exposes it
FALLBACK_CODE_MASKED = 2  # mapped macro verb illegal at the decision point
FALLBACK_CODES = {
    None: FALLBACK_CODE_NONE,
    macro_eval.FALLBACK_UNSUPPORTED: FALLBACK_CODE_UNSUPPORTED,
    macro_eval.FALLBACK_MASKED: FALLBACK_CODE_MASKED,
}
EPISODES_DIRNAME = episodes_v1.EPISODES_DIRNAME
SPILL_DIRNAME = episodes_v1.SPILL_DIRNAME
DEFAULT_SEED_BASE = 1_494_000_000
DEFAULT_SEEDS_PER_LEVEL = 40
DEFAULT_PARALLEL_ENVS = 12
DEFAULT_MAX_TICKS = 30_000
DEFAULT_HOLD_TICKS = macro_eval.DEFAULT_HOLD_TICKS
EMBARGOED_SEED_RANGES = episodes_v1.EMBARGOED_SEED_RANGES
# Engineering blocks already consumed by sibling tools; a plan straddling
# them could collide (level, seed) keys with existing collections.
RESERVED_ENGINEERING_SEED_BLOCKS = (
    ("recovery_intervention_probe_v4", 1_400_920_000, 1_400_920_999),
    ("park_settle_teacher_collection_v1", 1_543_004_000, 1_543_004_999),
    ("park_settle_dagger_v2", 1_543_005_000, 1_543_005_999),
    ("macro_ppo_engineering", 1_460_000_000, 1_493_999_999),
)

# Reused v1 planning machinery (imported, never copied).
EpisodeTask = episodes_v1.EpisodeTask
plan_tasks = episodes_v1.plan_tasks
_validate_levels = episodes_v1._validate_levels
_batches = episodes_v1._batches


def validate_seed_plan(tasks: Sequence[EpisodeTask]) -> None:
    """v1's frozen formal-embargo validation plus reserved-block refusals."""

    episodes_v1.validate_seed_plan(tasks)
    for task in tasks:
        for name, first, last in RESERVED_ENGINEERING_SEED_BLOCKS:
            if first <= int(task.seed) <= last:
                raise ValueError(
                    f"planned seed {int(task.seed)} for {task.level_id} "
                    f"overlaps the reserved {name} engineering block "
                    f"[{first}, {last}]; choose another --seed-base so "
                    f"collections keep unique (level, seed) keys"
                )


def _mean_or_none(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if len(values) else None


class _DecisionRecorder:
    """Accumulates one episode's macro decisions; observations spill.

    The stacked decision observations are appended raw to a spill file
    the moment they are recorded (12 concurrent long episodes could
    otherwise hold gigabytes in RAM) and memmapped back only while
    streaming the compressed NPZ, exactly the collector-v1 pattern.
    """

    def __init__(self, *, spill_path: Path, observation_width: int) -> None:
        if int(observation_width) < 1:
            raise ValueError("observation_width must be positive")
        self.spill_path = Path(spill_path)
        self.observation_width = int(observation_width)
        self.spill_path.parent.mkdir(parents=True, exist_ok=True)
        self._spill = self.spill_path.open("wb")
        self._pending_result = False
        self.decisions = 0
        self.native_ticks = 0
        self.macro_actions: list[np.ndarray] = []
        self.teacher_per_tick_actions: list[np.ndarray] = []
        self.macro_masks: list[np.ndarray] = []
        self.fallback_codes: list[int] = []
        self.tick_indices: list[int] = []
        self.verb_counts: Counter = Counter()
        self.fallback_counts: Counter = Counter()
        self.shots = 0
        self.shots_on_target = 0
        self.unsettled_fire_macros = 0
        self.release_errors: list[int] = []
        self.last_score = 0
        self.capacity_overflow = False

    def record_decision(
        self,
        *,
        observation: np.ndarray,
        macro_action: np.ndarray,
        macro_mask: np.ndarray,
        teacher_per_tick_action: np.ndarray,
        fallback: str | None,
    ) -> None:
        if self._pending_result:
            raise RuntimeError(
                "previous decision still awaits its macro result"
            )
        row = np.ascontiguousarray(observation, dtype=np.float32).reshape(-1)
        if int(row.size) != self.observation_width:
            raise ValueError(
                f"observation width changed mid-episode: {int(row.size)} "
                f"!= {self.observation_width}"
            )
        mask = np.asarray(macro_mask, dtype=np.bool_).reshape(-1)
        if mask.shape != (MACRO_MASK_WIDTH,):
            raise ValueError(
                f"macro mask width must be {MACRO_MASK_WIDTH}, got "
                f"{mask.shape}"
            )
        action = np.asarray(macro_action, dtype=np.int64).reshape(2)
        if not 0 <= int(action[0]) < MACRO_VERB_COUNT:
            raise ValueError(f"macro verb out of range: {int(action[0])}")
        if not 0 <= int(action[1]) < AIM_BINS:
            raise ValueError(f"macro aim bin out of range: {int(action[1])}")
        if fallback not in FALLBACK_CODES:
            raise ValueError(f"unknown adapter fallback reason: {fallback!r}")
        self._spill.write(row.tobytes())
        self.macro_actions.append(action.copy())
        self.teacher_per_tick_actions.append(
            np.asarray(teacher_per_tick_action, dtype=np.int64)
            .reshape(2)
            .copy()
        )
        self.macro_masks.append(mask.copy())
        self.fallback_codes.append(FALLBACK_CODES[fallback])
        self.tick_indices.append(int(self.native_ticks))
        if fallback is not None:
            self.fallback_counts[str(fallback)] += 1
        self._pending_result = True

    def record_result(
        self, macro: Mapping[str, Any], info: Mapping[str, Any]
    ) -> None:
        if not self._pending_result:
            raise RuntimeError("record_decision must precede record_result")
        self.native_ticks += int(macro["ticks_consumed"])
        self.verb_counts[str(macro["macro_verb"])] += 1
        if macro["macro_verb"] == MACRO_FIRE_NAME and not bool(
            macro["settled"]
        ):
            self.unsettled_fire_macros += 1
        if bool(macro.get("released", False)):
            self.shots += 1
            error = macro_eval._bin_distance(
                int(macro["release_aim_bin"]), int(macro["target_aim_bin"])
            )
            self.release_errors.append(int(error))
            if error <= macro_eval.SHOT_TOLERANCE_BINS:
                self.shots_on_target += 1
        self.last_score = int(info.get("score", self.last_score) or 0)
        self.capacity_overflow = self.capacity_overflow or bool(
            info.get("observation_capacity_overflow", False)
        )
        self._pending_result = False
        self.decisions += 1

    def write_npz(self, path: Path) -> tuple[str, int, int]:
        """Atomically write the episode NPZ; return (sha256, rows, bytes)."""

        if self._pending_result:
            raise RuntimeError("last decision is missing its macro result")
        if self.decisions < 1:
            raise ValueError("episode recorded no macro decisions")
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
            shape=(self.decisions, self.observation_width),
        )
        temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
        try:
            np.savez_compressed(
                temporary,
                observations=observations,
                macro_actions=np.stack(self.macro_actions),
                macro_masks=np.stack(self.macro_masks),
                tick_indices=np.asarray(self.tick_indices, dtype=np.int64),
                teacher_per_tick_actions=np.stack(
                    self.teacher_per_tick_actions
                ),
                fallback_codes=np.asarray(
                    self.fallback_codes, dtype=np.int8
                ),
            )
            del observations
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
        self.spill_path.unlink(missing_ok=True)
        return (
            legacy._sha256(path),
            int(self.decisions),
            int(path.stat().st_size),
        )

    def discard(self) -> None:
        """Drop an unfinished episode's spill without writing anything."""

        try:
            if not self._spill.closed:
                self._spill.close()
        finally:
            self.spill_path.unlink(missing_ok=True)


def _finalize_episode(
    *,
    task: EpisodeTask,
    recorder: _DecisionRecorder,
    final_info: Mapping[str, Any],
    guard_exhausted: bool,
    episodes_dir: Path,
    wall_seconds: float,
) -> dict[str, Any]:
    """Write the episode NPZ and return its manifest entry.

    Outcome honesty: ``win``/``loss`` come from the environment;
    anything that ended by time limit (or the collection guard) is
    recorded as ``truncated``.  Teacher losses are FLAGGED via
    ``teacher_won`` and their decisions are stored all the same.
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
    digest, decision_count, npz_bytes = recorder.write_npz(path)
    score = int(final_info.get("score", 0) or 0) or int(recorder.last_score)
    shots = int(recorder.shots)
    on_target = int(recorder.shots_on_target)
    return {
        "level_id": task.level_id,
        "seed": int(task.seed),
        "pass_index": int(task.pass_index),
        "outcome": outcome,
        "teacher_won": outcome == "win",
        "time_limit_truncated": truncated,
        "guard_exhausted": bool(guard_exhausted),
        "observation_capacity_overflow": bool(
            recorder.capacity_overflow
            or final_info.get("observation_capacity_overflow", False)
        ),
        "score": score,
        "decision_count": int(decision_count),
        "native_ticks": int(recorder.native_ticks),
        "observation_width": int(recorder.observation_width),
        "macro_verb_counts": dict(recorder.verb_counts),
        "mask_fallbacks": dict(recorder.fallback_counts),
        "shots": shots,
        "shots_on_target": on_target,
        "shots_on_target_rate": (
            on_target / shots if shots else None
        ),
        "mean_release_error_bins": _mean_or_none(recorder.release_errors),
        "unsettled_fire_macros": int(recorder.unsettled_fire_macros),
        "path": f"{EPISODES_DIRNAME}/{task.file_name}",
        "sha256": digest,
        "npz_bytes": int(npz_bytes),
        "wall_seconds": float(wall_seconds),
    }


def _default_vector_factory(
    *,
    original_root: Path,
    tasks: Sequence[EpisodeTask],
    max_ticks: int,
    hold_ticks: int,
    stack_lags: tuple[int, ...],
) -> Any:
    """The student's exact runtime stack: stacked motor env UNDER macros."""

    from stable_baselines3.common.vec_env import SubprocVecEnv

    factories = [
        native_eval._make_stacked_macro_env_factory(
            original_root=original_root,
            level_id=task.level_id,
            max_ticks=int(max_ticks),
            hold_ticks=int(hold_ticks),
            stack_lags=stack_lags,
        )
        for task in tasks
    ]
    return SubprocVecEnv(factories, start_method="forkserver")


def _default_teachers_factory(
    vector: Any, *, raw_observation_dim: int
) -> list[Any]:
    """Per-env settled strategic teachers on the RAW current frame.

    v4's construction (deterministic code from each environment's
    ``teacher_spec()``) wrapped in probe-v5's ``RawFrameTeacher``: the
    observations are stacked, so each teacher acts on the leading
    ``raw_observation_dim`` slice only.
    """

    return v5.wrap_raw_frame_teachers(
        v4._build_teachers(vector), raw_observation_dim=raw_observation_dim
    )


def _default_masks_provider(vector: Any) -> np.ndarray:
    from sb3_contrib.common.maskable.utils import get_action_masks

    return np.asarray(get_action_masks(vector), dtype=np.bool_)


def _collect_batch(
    *,
    original_root: Path,
    tasks: Sequence[EpisodeTask],
    record_flags: Sequence[bool],
    max_ticks: int,
    hold_ticks: int,
    stack_lags: tuple[int, ...],
    episodes_dir: Path,
    spill_dir: Path,
    on_episode: Callable[[dict[str, Any]], None],
    vector_factory: Callable[..., Any] | None = None,
    teachers_factory: Callable[..., list[Any]] | None = None,
    masks_provider: Callable[[Any], np.ndarray] | None = None,
) -> list[dict[str, Any]]:
    """Run one seed-contiguous batch of teacher macro-decision episodes.

    The run loop is macro_eval's decision-point loop with recording:
    one ``vector.step`` is one macro per environment, the teacher is
    queried on the RAW leading slice of each stacked decision-point
    observation, and every telemetry row comes from the wrapper's own
    ``info["park_settle"]`` release reports.  The vector/teachers/masks
    seams are injectable so the adapter-and-recording path is unit
    testable on CPU without an engine; the real run uses SubprocVecEnv
    over the stacked macro stacks.
    """

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
    lags = parse_stack_lags(stack_lags)
    if vector_factory is None:
        vector_factory = _default_vector_factory
    if teachers_factory is None:
        teachers_factory = _default_teachers_factory
    if masks_provider is None:
        masks_provider = _default_masks_provider

    vector = vector_factory(
        original_root=original_root,
        tasks=tasks,
        max_ticks=int(max_ticks),
        hold_ticks=int(hold_ticks),
        stack_lags=lags,
    )
    recorders: list[_DecisionRecorder | None] = [None] * count
    entries: list[dict[str, Any]] = []
    try:
        vector.seed(first_seed)
        observations = np.asarray(vector.reset(), dtype=np.float32)
        stacked_width = int(observations.shape[1])
        if stacked_width % len(lags):
            raise ValueError(
                f"stacked observation width {stacked_width} is not "
                f"divisible by {len(lags)} stack frames"
            )
        raw_observation_dim = stacked_width // len(lags)
        teachers = teachers_factory(
            vector, raw_observation_dim=raw_observation_dim
        )
        if len(teachers) != count:
            raise ValueError("one teacher per batch environment is required")
        for position in range(count):
            if bool(record_flags[position]):
                recorders[position] = _DecisionRecorder(
                    spill_path=spill_dir
                    / f"{tasks[position].file_name}.observations.spill",
                    observation_width=stacked_width,
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
            if bool(np.all(finished)):
                break
            macro_masks = np.asarray(
                masks_provider(vector), dtype=np.bool_
            ).reshape(count, -1)
            if macro_masks.shape[1] != MACRO_MASK_WIDTH:
                raise ValueError(
                    f"macro mask width must be {MACRO_MASK_WIDTH}, got "
                    f"{macro_masks.shape}"
                )
            # Finished environments (auto-reset by the vector) get an
            # inert wait_hold; their telemetry is never read again.
            actions = np.zeros((count, 2), dtype=np.int64)
            active = ~finished
            for position in np.flatnonzero(active):
                index = int(position)
                per_tick = np.asarray(
                    teachers[index].act(observations[index]),
                    dtype=np.int64,
                ).reshape(2)
                macro_action, fallback = macro_eval.adapt_per_tick_action(
                    per_tick, macro_masks[index]
                )
                actions[index] = macro_action
                recorder = recorders[index]
                if recorder is not None:
                    recorder.record_decision(
                        observation=observations[index],
                        macro_action=macro_action,
                        macro_mask=macro_masks[index],
                        teacher_per_tick_action=per_tick,
                        fallback=fallback,
                    )
            step_observations, _, dones, infos = vector.step(actions)
            for position in np.flatnonzero(active):
                index = int(position)
                info = dict(infos[index])
                macro = info.get("park_settle")
                if not isinstance(macro, Mapping):
                    raise RuntimeError(
                        "macro step returned no park_settle telemetry; "
                        "the vector is not a park-settle wrapped stack"
                    )
                recorder = recorders[index]
                if recorder is not None:
                    recorder.record_result(macro, info)
                last_infos[index] = info
                if bool(dones[index]):
                    finished[index] = True
                    finalize(index, info, False)
            observations = np.asarray(step_observations, dtype=np.float32)
        for position in range(count):
            if not finished[position]:
                # Guard exhausted without done: record honestly.
                finalize(int(position), last_infos[int(position)], True)
    finally:
        for recorder in recorders:
            if recorder is not None:
                recorder.discard()
        vector.close()
    return entries


def _decision_interface_contract(hold_ticks: int) -> dict[str, Any]:
    return {
        "action_space_nvec": [MACRO_VERB_COUNT, AIM_BINS],
        "mask_width": MACRO_MASK_WIDTH,
        "macro_verbs": list(MACRO_VERB_NAMES),
        "adapter": (
            "tools.probe_alphazuma_55_macro_eval_v1.adapt_per_tick_action: "
            "wait -> wait_hold CARRYING the teacher's parked target bin; "
            "fire -> fire; swap -> swap; hop and macro-mask-illegal verbs "
            "fall back to wait_hold, counted per episode and flagged per "
            "decision in fallback_codes"
        ),
        "mask_semantics": {
            "macro_masks": "park_settle_wrapper_action_masks"
        },
        "hold_ticks": int(hold_ticks),
        "tick_index_semantics": (
            "native ticks consumed in the episode BEFORE the decision "
            "(the runtime ObservationStackWrapper ring buffer has "
            "advanced exactly that many times)"
        ),
        "fallback_codes": {
            "0": "none",
            "1": macro_eval.FALLBACK_UNSUPPORTED,
            "2": macro_eval.FALLBACK_MASKED,
        },
    }


def _observation_stack_contract(
    stack_lags: Sequence[int],
) -> dict[str, Any]:
    return {
        "stack_lags": [int(lag) for lag in stack_lags],
        "frames": len(tuple(stack_lags)),
        "position": "under_park_settle_macro_wrapper",
        "ring_buffer_advances_every_native_tick": True,
        "policy_observation": "feature_axis_stacked_frames",
        "teacher_observation": "raw_current_frame_leading_slice",
    }


def _write_manifest(
    run_dir: Path,
    *,
    levels: Sequence[str],
    seeds_per_level: int,
    seed_base: int,
    max_ticks: int,
    hold_ticks: int,
    stack_lags: Sequence[int],
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
        "purpose": (
            "teacher macro decisions on the student's exact runtime "
            "stacked park-settle interface, for direct BC supervision by "
            "tools/distill_alphazuma_55_macro_decisions_v1.py"
        ),
        "teacher_policy_id": TEACHER_POLICY_ID,
        "environment_stack": (
            "RevengeEnv + ObservableHumanSpeedrunWrapper (motor-"
            "observable) + ObservationStackWrapper UNDER "
            "ParkSettleActionWrapper"
        ),
        "decision_interface": _decision_interface_contract(hold_ticks),
        "observation_stack": _observation_stack_contract(stack_lags),
        "levels": list(levels),
        "seeds_per_level": int(seeds_per_level),
        "seed_base": int(seed_base),
        "max_ticks": int(max_ticks),
        "hold_ticks": int(hold_ticks),
        "stack_lags": [int(lag) for lag in stack_lags],
        "teacher_losses_are_recorded_and_flagged": True,
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
        if int(entry.get("decision_count", 0)) < 1:
            continue
        completed[(str(entry["level_id"]), int(entry["seed"]))] = dict(entry)
    return completed


def collect(
    *,
    original_root: Path,
    run_dir: Path,
    levels: Sequence[str] = INCLUDED_LEVELS,
    seeds_per_level: int = DEFAULT_SEEDS_PER_LEVEL,
    seed_base: int = DEFAULT_SEED_BASE,
    parallel_envs: int = DEFAULT_PARALLEL_ENVS,
    max_ticks: int = DEFAULT_MAX_TICKS,
    hold_ticks: int = DEFAULT_HOLD_TICKS,
    stack_lags: Any = DEFAULT_STACK_LAGS,
    device: str = "cpu",
) -> dict[str, Any]:
    """Collect (or resume) the teacher macro-decision archive."""

    started = time.perf_counter()
    original_root = Path(original_root).expanduser().resolve(strict=True)
    run_dir = Path(run_dir).expanduser().resolve()
    ordered_levels = _validate_levels(levels)
    lags = parse_stack_lags(stack_lags)
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
        "hold_ticks": int(hold_ticks),
        "stack_lags": [int(lag) for lag in lags],
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
                "env_factory_module": {
                    "module": (
                        "tools.probe_alphazuma_55_macro_native_eval_v1"
                    ),
                    "path": str(native_eval.SCRIPT_PATH),
                    "sha256": legacy._sha256(native_eval.SCRIPT_PATH),
                },
                "adapter_module": {
                    "module": "tools.probe_alphazuma_55_macro_eval_v1",
                    "path": str(macro_eval.SCRIPT_PATH),
                    "sha256": legacy._sha256(macro_eval.SCRIPT_PATH),
                },
                "teacher_policy_id": TEACHER_POLICY_ID,
                "teacher_id": TEACHER_ID,
                "teacher_construction": (
                    "deterministic code built from each environment's "
                    "teacher_spec(); no checkpoint file exists"
                ),
                "teacher_observation": "raw_current_frame_leading_slice",
                "decision_interface": _decision_interface_contract(
                    int(hold_ticks)
                ),
                "observation_stack": _observation_stack_contract(lags),
                "original_root": str(original_root),
                "device": str(device),
                "device_role": (
                    "none: the teacher is deterministic CPU code; the "
                    "flag is recorded for orchestration parity only and "
                    "no GPU is touched"
                ),
                "parallel_envs": int(parallel_envs),
                "input_profile": episodes_v1._input_config().profile_id,
                "reward_profile": episodes_v1._reward_config().profile_id,
                "seed_last": int(seed_base)
                + int(seeds_per_level) * len(ordered_levels)
                - 1,
                "seed_embargo": {
                    name: [first, last]
                    for name, first, last in EMBARGOED_SEED_RANGES
                },
                "reserved_engineering_seed_blocks": {
                    name: [first, last]
                    for name, first, last in (
                        RESERVED_ENGINEERING_SEED_BLOCKS
                    )
                },
                "formal_seed_consumption": False,
                "training_authority": False,
                **plan,
            },
        )

    entries_by_key = _load_completed_entries(run_dir)
    reused_keys = set(entries_by_key)
    # Self-heal resume: quarantine-rename (never delete) any episode NPZ
    # without a verified receipt so write_npz's overwrite refusal cannot
    # wedge the re-collection (the collector-v1 pattern, reused).
    verified_paths = {
        (run_dir / str(entry["path"])).resolve()
        for entry in entries_by_key.values()
    }
    quarantined_files = episodes_v1._quarantine_unverified_episode_files(
        episodes_dir, verified_paths
    )

    def write_manifest() -> Path:
        return _write_manifest(
            run_dir,
            levels=ordered_levels,
            seeds_per_level=int(seeds_per_level),
            seed_base=int(seed_base),
            max_ticks=int(max_ticks),
            hold_ticks=int(hold_ticks),
            stack_lags=lags,
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
                hold_ticks=int(hold_ticks),
                stack_lags=lags,
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
            "decisions": sum(
                int(entry["decision_count"]) for entry in members
            ),
            "native_ticks": sum(
                int(entry["native_ticks"]) for entry in members
            ),
            "shots": sum(int(entry["shots"]) for entry in members),
            "shots_on_target": sum(
                int(entry["shots_on_target"]) for entry in members
            ),
            "seeds": [int(entry["seed"]) for entry in members],
        }
    total_shots = sum(int(entry["shots"]) for entry in ordered_entries)
    total_on_target = sum(
        int(entry["shots_on_target"]) for entry in ordered_entries
    )
    completion = {
        "schema": COMPLETION_SCHEMA,
        "version": VERSION,
        "status": ("COMPLETE" if not missing and not errors else "PARTIAL"),
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
        "hold_ticks": int(hold_ticks),
        "stack_lags": [int(lag) for lag in lags],
        "seed_embargo": {
            name: [first, last]
            for name, first, last in EMBARGOED_SEED_RANGES
        },
        "reserved_engineering_seed_blocks": {
            name: [first, last]
            for name, first, last in RESERVED_ENGINEERING_SEED_BLOCKS
        },
        "formal_seed_consumption": False,
        "training_authority": False,
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
        "teacher_losses_flagged": sum(
            not bool(entry["teacher_won"]) for entry in ordered_entries
        ),
        "guard_exhaustions": sum(
            bool(entry["guard_exhausted"]) for entry in ordered_entries
        ),
        "capacity_overflows": sum(
            bool(entry["observation_capacity_overflow"])
            for entry in ordered_entries
        ),
        "per_level": per_level,
        "total_decisions": sum(
            int(entry["decision_count"]) for entry in ordered_entries
        ),
        "total_native_ticks": sum(
            int(entry["native_ticks"]) for entry in ordered_entries
        ),
        "total_shots": total_shots,
        "shots_on_target": total_on_target,
        "shots_on_target_rate": (
            total_on_target / total_shots if total_shots else None
        ),
        "total_npz_bytes": sum(
            int(entry["npz_bytes"]) for entry in ordered_entries
        ),
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", required=True, type=Path)
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
    parser.add_argument(
        "--seed-base",
        type=int,
        default=DEFAULT_SEED_BASE,
        help=(
            "validated against the frozen formal embargo ranges and the "
            "reserved engineering blocks before any environment work"
        ),
    )
    parser.add_argument(
        "--parallel-envs", type=int, default=DEFAULT_PARALLEL_ENVS
    )
    parser.add_argument("--hold-ticks", type=int, default=DEFAULT_HOLD_TICKS)
    parser.add_argument(
        "--stack-lags",
        default=",".join(str(lag) for lag in DEFAULT_STACK_LAGS),
        help=(
            "comma-separated source-tick lags (feature-axis stack UNDER "
            "the macro wrapper); must start with 0 and match the student "
            "the decisions will supervise"
        ),
    )
    parser.add_argument("--max-ticks", type=int, default=DEFAULT_MAX_TICKS)
    parser.add_argument(
        "--device",
        default="cpu",
        help=(
            "recorded for orchestration parity only: the teacher is "
            "deterministic CPU code and no GPU is touched"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    completion = collect(
        original_root=args.original_root,
        run_dir=args.run_dir,
        levels=args.levels,
        seeds_per_level=int(args.seeds_per_level),
        seed_base=int(args.seed_base),
        parallel_envs=int(args.parallel_envs),
        max_ticks=int(args.max_ticks),
        hold_ticks=int(args.hold_ticks),
        stack_lags=str(args.stack_lags),
        device=str(args.device),
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
                "teacher_losses_flagged": completion[
                    "teacher_losses_flagged"
                ],
                "total_decisions": completion["total_decisions"],
                "total_native_ticks": completion["total_native_ticks"],
                "total_shots": completion["total_shots"],
                "shots_on_target_rate": completion["shots_on_target_rate"],
                "total_npz_bytes": completion["total_npz_bytes"],
                "run_dir": completion["run_dir"],
                "episodes_manifest": completion["episodes_manifest"],
                "formal_seed_consumption": False,
                "training_authority": False,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    return 0 if completion["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
