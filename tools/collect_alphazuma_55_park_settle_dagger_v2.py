"""DAgger episode collector: the STUDENT executes, the TEACHER labels.

Why this exists
---------------
The park-settle student is statically calibrated (fire rate 0.86x
teacher, aim within-3 54%) but collapses in closed loop (0/12,
shots-on-target 0%): its cursor-feedback features leave the training
distribution the moment the student drives the cursor itself
(aim-intent error p50 43 bins student-driven vs 18 teacher-driven,
measured by ``tools/probe_alphazuma_55_recovery_intervention_v4.py``).
Teacher-driven collection can never visit those states.  This collector
closes the DAgger loop: the student policy EXECUTES every tick (so the
episodes carry the student-visited, off-distribution cursor states) and
the 55/55 settled strategic teacher's counterfactual proposal for the
SAME state is computed every tick and stored as the LABEL.

Per-tick contract
-----------------
* STUDENT executes: deterministic masked inference, byte-identical
  semantics to ``probe_alphazuma_55_recovery_intervention_v4``
  student_only mode -- ``MaskablePPO.load(policy_path, device)``,
  ``torch.set_num_threads(1)``, ``model.predict(observations,
  deterministic=True, action_masks=get_action_masks(vector))``, int64
  ``reshape(count, 2)``, per-active-env
  ``distill_alphazuma_55_polar_dagger_v1._validate_student_action``,
  and the predicted batch (including finished rows) is what the vector
  steps.
* TEACHER labels: the same per-env
  ``CurveAwareSettledStrategicRevengeTeacher`` (deterministic code from
  each environment's ``teacher_spec()``) proposes on the observation the
  student saw; ``distill_alphazuma_55_polar_intent_wide_v1
  ._intent_label`` derives the exact v1 label tuple.  The NPZ therefore
  stores, under the UNCHANGED v1 keys: ``raw_actions`` = raw teacher
  intent ``[verb, aim_bin]`` (the training label; fire windows in the
  v4 coverage builder derive from THESE edges), ``effective_actions`` =
  the teacher's mask-effective counterfactual, ``training_masks`` /
  ``exact_masks`` / ``intent_mask_relaxed`` exactly as v1 defines them.
* ADDITIVE ONLY: the student's executed action is stored in a NEW
  ``executed_actions`` array (int64, shape ``(ticks, 2)``).  Every v1
  consumer is unaffected: the coverage runner and the v1 hydrators read
  explicit key tuples, and the park-settle trainer's ``_ACTION_KEYS``
  precedence starts at ``raw_actions``, so ``executed_actions`` can
  never shadow the label.

Reuse (import, never copy-paste)
--------------------------------
Everything that is not the per-tick DAgger loop is
``tools.collect_alphazuma_55_park_settle_episodes_v1`` machinery,
imported and delegated to: seed planning (``plan_tasks`` /
``validate_seed_plan``), the episode recorder (subclassed only to append
``executed_actions``), episode finalization/statistics
(``_finalize_episode``), the env stack (``_input_config`` /
``_reward_config`` / ``motor._make_motor_env_factory`` /
``SubprocVecEnv`` forkserver / ``vector.seed(first_seed)``), and the
whole manifest/receipts/resume/quarantine orchestration (``v1.collect``
called with this module's batch collector and manifest decorator
scoped-rebound in the exact pattern
``tools/distill_alphazuma_55_park_settle_v2.py`` established; the
bindings are restored in ``finally``).  The episodes manifest keeps v1's
schema string and ``mask_semantics``, so
``tools/run_alphazuma_55_coverage_replay_build_v1.py`` and
``tools.distill_alphazuma_55_park_settle_v1.load_replay_dataset``
consume DAgger collections UNCHANGED; honest additive keys
(``collection_mode``, ``student_policy``, corrected
``environment_execution_semantics``) record what really executed.

Honest outcomes
---------------
Student-driven episodes will mostly be losses -- that is the point;
they carry the off-distribution cursor states.  Outcomes flow through
v1's ``_finalize_episode`` untouched (``win``/``loss``/``truncated``,
guard exhaustion recorded, never dropped).  Additive per-episode
diagnostics count label-vs-executed disagreement so the DAgger value is
auditable per episode.

Seed discipline
---------------
Default ``--seed-base`` 1_543_005_000: validated by v1 against the
three frozen formal embargo ranges AND additionally refused if it
overlaps the reserved engineering blocks already consumed tonight --
the v1 teacher collection block at 1_543_004_000 and the recovery probe
block at 1_400_920_000 -- so teacher and DAgger collections can always
be merged without (level, seed) collisions.
``formal_seed_consumption`` stays ``false`` in every receipt.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import numpy as np

from tools import collect_alphazuma_55_park_settle_episodes_v1 as v1
from tools import distill_alphazuma_55 as legacy
from tools import distill_alphazuma_55_motor_observable_replay_v2 as motor
from tools import distill_alphazuma_55_polar_dagger_v1 as dagger
from tools import distill_alphazuma_55_polar_intent_wide_v1 as intent
from zuma_rl.alphazuma_55 import INCLUDED_LEVELS


SCRIPT_PATH = Path(__file__).resolve()
SCHEMA_PREFIX = "zuma-rl.alphazuma-55-park-settle-dagger"
CONFIG_SCHEMA = f"{SCHEMA_PREFIX}-collection-config"
COMPLETION_SCHEMA = f"{SCHEMA_PREFIX}-collection-completion"
VERSION = 2
COLLECTION_MODE = "dagger_student_executes_teacher_labels"
EXECUTED_ACTIONS_KEY = "executed_actions"
EXECUTION_SEMANTICS = "student_policy_executed_action"
EXECUTED_ACTIONS_SEMANTICS = (
    "student_policy_deterministic_masked_predict"
)
LABEL_SEMANTICS = "raw_teacher_intent_counterfactual_per_tick"
STUDENT_INFERENCE_SEMANTICS = (
    "MaskablePPO.load(policy_path, device); torch.set_num_threads(1); "
    "model.predict(observations, deterministic=True, "
    "action_masks=get_action_masks(vector)); np.asarray(int64)"
    ".reshape(count, 2); per-active-env _validate_student_action -- "
    "byte-identical to probe_alphazuma_55_recovery_intervention_v4 "
    "student_only"
)
DEFAULT_SEED_BASE = 1_543_005_000
DEFAULT_SEEDS_PER_LEVEL = 2
# Engineering blocks already consumed by sibling tools tonight; a DAgger
# plan straddling them could later collide (level, seed) keys with the
# teacher collection when the two are merged, so they are refused here
# on top of v1's frozen formal embargo validation.
RESERVED_ENGINEERING_SEED_BLOCKS = (
    # tools/collect_alphazuma_55_park_settle_episodes_v1.py teacher block
    ("park_settle_teacher_collection_v1", 1_543_004_000, 1_543_004_999),
    # tools/probe_alphazuma_55_recovery_intervention_v4.py probe block
    ("recovery_intervention_probe_v4", 1_400_920_000, 1_400_920_999),
)


def validate_dagger_seed_plan(tasks: Sequence[v1.EpisodeTask]) -> None:
    """v1's frozen embargo validation plus the reserved-block refusals."""

    v1.validate_seed_plan(tasks)
    for task in tasks:
        for name, first, last in RESERVED_ENGINEERING_SEED_BLOCKS:
            if first <= int(task.seed) <= last:
                raise ValueError(
                    f"planned seed {int(task.seed)} for {task.level_id} "
                    f"overlaps the reserved {name} engineering block "
                    f"[{first}, {last}]; choose another --seed-base so "
                    f"merged collections keep unique (level, seed) keys"
                )


class _SavezExecutedInjector:
    """Numpy proxy whose ``savez_compressed`` adds ``executed_actions``.

    ``v1._EpisodeRecorder.write_npz`` is reused verbatim (spill fsync,
    memmap streaming, atomic temp + ``os.replace``, overwrite refusal,
    sha256 receipt); the ONLY difference a DAgger episode needs is one
    additional array in the same ``np.savez_compressed`` call.  The
    subclass below rebinds ``v1.np`` to this proxy for exactly the
    duration of ``super().write_npz`` (single-threaded episode
    finalization; restored in ``finally``), so the v1 writer stays
    byte-for-byte unmodified while the NPZ gains the additive key.
    """

    def __init__(self, executed: np.ndarray) -> None:
        self._executed = np.asarray(executed, dtype=np.int64)

    def __getattr__(self, name: str) -> Any:
        return getattr(np, name)

    def savez_compressed(self, file: Any, **arrays: Any) -> None:
        if EXECUTED_ACTIONS_KEY in arrays:
            raise ValueError(
                f"NPZ already carries {EXECUTED_ACTIONS_KEY!r}"
            )
        np.savez_compressed(
            file, **arrays, **{EXECUTED_ACTIONS_KEY: self._executed}
        )


class DaggerEpisodeRecorder(v1._EpisodeRecorder):
    """v1 recorder plus the additive student ``executed_actions`` stream."""

    def __init__(self, *, spill_path: Path, observation_width: int) -> None:
        super().__init__(
            spill_path=spill_path, observation_width=observation_width
        )
        self.executed_actions: list[np.ndarray] = []

    def record_intent(  # type: ignore[override]
        self,
        *,
        observation: np.ndarray,
        raw_action: np.ndarray,
        effective_action: np.ndarray,
        training_mask: np.ndarray,
        exact_mask: np.ndarray,
        relaxed: bool,
        executed_action: np.ndarray,
    ) -> None:
        super().record_intent(
            observation=observation,
            raw_action=raw_action,
            effective_action=effective_action,
            training_mask=training_mask,
            exact_mask=exact_mask,
            relaxed=relaxed,
        )
        self.executed_actions.append(
            np.asarray(executed_action, dtype=np.int64).reshape(2).copy()
        )

    def write_npz(self, path: Path) -> tuple[str, int, int]:
        if not self.executed_actions:
            raise ValueError("episode recorded no ticks")
        if self._pending_info:
            raise RuntimeError("last tick is missing its info record")
        if len(self.executed_actions) != self.ticks:
            raise RuntimeError(
                f"executed action stream misaligned: "
                f"{len(self.executed_actions)} != {self.ticks}"
            )
        original = v1.np
        v1.np = _SavezExecutedInjector(np.stack(self.executed_actions))
        try:
            return super().write_npz(path)
        finally:
            v1.np = original


def _finalize_dagger_episode(
    *,
    task: v1.EpisodeTask,
    recorder: DaggerEpisodeRecorder,
    final_info: Mapping[str, Any],
    guard_exhausted: bool,
    episodes_dir: Path,
    wall_seconds: float,
) -> dict[str, Any]:
    """v1 finalization (honest outcomes) plus additive DAgger diagnostics."""

    executed = np.stack(recorder.executed_actions)
    labels = np.stack(recorder.raw_actions)
    entry = v1._finalize_episode(
        task=task,
        recorder=recorder,
        final_info=final_info,
        guard_exhausted=guard_exhausted,
        episodes_dir=episodes_dir,
        wall_seconds=wall_seconds,
    )
    executed_counts = Counter(
        v1.VERB_NAMES[int(verb)] for verb in executed[:, 0]
    )
    entry["collection_mode"] = COLLECTION_MODE
    entry["executed_action_source"] = "student_policy"
    entry["executed_action_counts"] = dict(executed_counts)
    entry["fire_edges_executed_student"] = v1._fire_edge_count(
        executed[:, 0]
    )
    entry["label_executed_verb_disagreement_ticks"] = int(
        np.count_nonzero(labels[:, 0] != executed[:, 0])
    )
    entry["label_executed_action_disagreement_ticks"] = int(
        np.count_nonzero(np.any(labels != executed, axis=1))
    )
    return entry


def _load_student_model(
    *, policy_path: Path, device: str, model_cache: dict[Any, Any]
) -> Any:
    """Cached probe-v4-identical student load (one load per collection)."""

    key = (str(policy_path), str(device))
    model = model_cache.get(key)
    if model is None:
        import torch
        from sb3_contrib import MaskablePPO

        torch.set_num_threads(1)
        model = MaskablePPO.load(str(policy_path), device=str(device))
        model_cache[key] = model
    return model


def _default_masks_provider(vector: Any) -> np.ndarray:
    from sb3_contrib.common.maskable.utils import get_action_masks

    return np.asarray(get_action_masks(vector), dtype=np.bool_)


def _default_vector_factory(
    *, original_root: Path, tasks: Sequence[v1.EpisodeTask], max_ticks: int
) -> Any:
    """The exact v1 batch env stack (motor-observable, forkserver)."""

    from stable_baselines3.common.vec_env import SubprocVecEnv

    factories = [
        motor._make_motor_env_factory(
            original_root=original_root,
            level_id=task.level_id,
            max_ticks=int(max_ticks),
            input_config=v1._input_config(),
            reward_config=v1._reward_config(),
        )
        for task in tasks
    ]
    return SubprocVecEnv(factories, start_method="forkserver")


def _default_teachers_factory(vector: Any) -> list[Any]:
    """Per-env settled strategic teachers, exactly as v1 constructs them."""

    specifications = vector.env_method("teacher_spec")
    return [
        v1.CurveAwareSettledStrategicRevengeTeacher(
            legacy._teacher_spec(value)
        )
        for value in specifications
    ]


def _collect_batch_dagger(
    *,
    original_root: Path,
    tasks: Sequence[v1.EpisodeTask],
    record_flags: Sequence[bool],
    max_ticks: int,
    episodes_dir: Path,
    spill_dir: Path,
    on_episode: Callable[[dict[str, Any]], None],
    policy_path: Path,
    device: str,
    model_cache: dict[Any, Any],
    model: Any | None = None,
    vector_factory: Callable[..., Any] | None = None,
    teachers_factory: Callable[[Any], list[Any]] | None = None,
    masks_provider: Callable[[Any], np.ndarray] | None = None,
) -> list[dict[str, Any]]:
    """One seed-contiguous DAgger batch: student steps, teacher labels.

    Signature-compatible with ``v1._collect_batch`` (the v1 orchestrator
    calls it through the scoped rebinding in :func:`collect`) plus the
    student policy arguments.  The vector/teachers/masks/model seams are
    injectable so the student-executes/teacher-labels separation is unit
    testable on CPU without an engine; the real run uses the exact v1
    env stack and the exact probe-v4 student inference.
    """

    import time

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
    if model is None:
        model = _load_student_model(
            policy_path=policy_path, device=device, model_cache=model_cache
        )
    if vector_factory is None:
        vector_factory = _default_vector_factory
    if teachers_factory is None:
        teachers_factory = _default_teachers_factory
    if masks_provider is None:
        masks_provider = _default_masks_provider

    vector = vector_factory(
        original_root=original_root, tasks=tasks, max_ticks=int(max_ticks)
    )
    recorders: list[DaggerEpisodeRecorder | None] = [None] * count
    entries: list[dict[str, Any]] = []
    try:
        vector.seed(first_seed)
        observations = vector.reset()
        teachers = teachers_factory(vector)
        if len(teachers) != count:
            raise ValueError("one teacher per batch environment is required")
        width = int(np.asarray(observations).shape[1])
        for position in range(count):
            if bool(record_flags[position]):
                recorders[position] = DaggerEpisodeRecorder(
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
            entry = _finalize_dagger_episode(
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
            masks = masks_provider(vector)
            # Probe-v4 student_only inference, byte-identical: batch
            # deterministic masked predict, int64, (count, 2).
            predicted, _ = model.predict(
                observations,
                deterministic=True,
                action_masks=masks,
            )
            student_actions = np.asarray(predicted, dtype=np.int64).reshape(
                count, 2
            )
            actions = student_actions.copy()
            active = ~finished
            for position in np.flatnonzero(active):
                index = int(position)
                dagger._validate_student_action(
                    student_actions[index], masks[index]
                )
                # Teacher counterfactual proposal for the SAME state the
                # student saw: the v1 label tuple, stored as the label.
                teacher_raw = np.asarray(
                    teachers[index].act(observations[index]),
                    dtype=np.int64,
                )
                raw, effective, training_mask, forced = intent._intent_label(
                    teacher_raw, masks[index]
                )
                recorder = recorders[index]
                if recorder is not None:
                    recorder.record_intent(
                        observation=observations[index],
                        raw_action=raw,
                        effective_action=effective,
                        training_mask=training_mask,
                        exact_mask=masks[index],
                        relaxed=forced,
                        executed_action=student_actions[index],
                    )
            # The STUDENT batch is what the vector executes.
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
            observations = step_observations
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


def _manifest_additions(
    *, policy_path: Path, policy_sha256: str, script_sha256: str
) -> dict[str, Any]:
    """Additive-and-honest keys layered onto every v1 manifest rewrite.

    The v1 schema string, ``mask_semantics``, ``teacher_policy_id`` and
    ``training_label_semantics`` (all consumed downstream) stay v1's;
    ``environment_execution_semantics`` is corrected because in DAgger
    the environment executed the STUDENT action, not the teacher's
    mask-effective action -- no downstream consumer reads that key, but
    the receipt must not lie.
    """

    return {
        "collection_mode": COLLECTION_MODE,
        "dagger_version": VERSION,
        "environment_execution_semantics": EXECUTION_SEMANTICS,
        "executed_actions_semantics": EXECUTED_ACTIONS_SEMANTICS,
        "label_semantics": LABEL_SEMANTICS,
        "student_inference_semantics": STUDENT_INFERENCE_SEMANTICS,
        "student_policy": {
            "path": str(policy_path),
            "sha256": policy_sha256,
        },
        "dagger_collector": {
            "path": str(SCRIPT_PATH),
            "sha256": script_sha256,
        },
    }


def collect(
    *,
    original_root: Path,
    run_dir: Path,
    policy_path: Path,
    levels: Sequence[str] = INCLUDED_LEVELS,
    seeds_per_level: int = DEFAULT_SEEDS_PER_LEVEL,
    seed_base: int = DEFAULT_SEED_BASE,
    parallel_envs: int = v1.DEFAULT_PARALLEL_ENVS,
    max_ticks: int = v1.DEFAULT_MAX_TICKS,
    device: str = "cpu",
) -> dict[str, Any]:
    """Collect (or resume) the DAgger episode archive via v1 machinery.

    The whole orchestration -- config plan pinning, resume receipts,
    quarantine self-healing, atomic manifest/status/completion writes --
    is ``v1.collect`` with this module's batch collector and manifest
    decorator scoped-rebound for the duration of the call (the
    ``distill_alphazuma_55_park_settle_v2`` rebinding pattern; restored
    in ``finally``).  A DAgger ``config.json`` is written FIRST so the
    run dir's primary receipt is honest about what executed; ``v1``
    then only validates the plan keys against it.  Resume refuses a
    different student policy: mixing policies in one run dir would
    poison the collection.
    """

    original_root = Path(original_root).expanduser().resolve(strict=True)
    run_dir = Path(run_dir).expanduser().resolve()
    policy_path = Path(policy_path).expanduser().resolve(strict=True)
    ordered_levels = v1._validate_levels(levels)
    tasks = v1.plan_tasks(
        levels=ordered_levels,
        seeds_per_level=int(seeds_per_level),
        seed_base=int(seed_base),
    )
    validate_dagger_seed_plan(tasks)
    policy_sha256 = legacy._sha256(policy_path)
    script_sha256 = legacy._sha256(SCRIPT_PATH)

    plan = {
        "levels": list(ordered_levels),
        "seeds_per_level": int(seeds_per_level),
        "seed_base": int(seed_base),
        "max_ticks": int(max_ticks),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    if config_path.exists():
        existing = legacy._read_json(config_path)
        if existing.get("schema") != CONFIG_SCHEMA:
            raise ValueError(
                f"run dir holds a foreign (non-DAgger) config: {config_path}"
            )
        previous_policy = existing.get("student_policy")
        previous_policy = (
            previous_policy if isinstance(previous_policy, Mapping) else {}
        )
        if previous_policy.get("sha256") != policy_sha256:
            raise ValueError(
                f"run dir {run_dir} was collected with a different student "
                f"policy (config {previous_policy.get('sha256')}, requested "
                f"{policy_sha256}); refusing to mix policies in one "
                f"collection"
            )
        # Plan mismatches are refused by the reused v1.collect check.
    else:
        legacy._write_json_atomic(
            config_path,
            {
                "schema": CONFIG_SCHEMA,
                "version": VERSION,
                "collection_mode": COLLECTION_MODE,
                "collector": {
                    "path": str(SCRIPT_PATH),
                    "sha256": script_sha256,
                },
                "reused_v1_machinery": {
                    "path": str(v1.SCRIPT_PATH),
                    "sha256": legacy._sha256(v1.SCRIPT_PATH),
                },
                "builder_contract": {
                    "path": str(v1.BUILDER_PATH),
                    "sha256": legacy._sha256(v1.BUILDER_PATH),
                },
                "teacher_policy_id": v1.TEACHER_POLICY_ID,
                "teacher_construction": (
                    "deterministic code built from each environment's "
                    "teacher_spec(); no checkpoint file exists"
                ),
                "label_semantics": LABEL_SEMANTICS,
                "student_policy": {
                    "path": str(policy_path),
                    "sha256": policy_sha256,
                },
                "student_inference_semantics": STUDENT_INFERENCE_SEMANTICS,
                "executed_actions_semantics": EXECUTED_ACTIONS_SEMANTICS,
                "environment_execution_semantics": EXECUTION_SEMANTICS,
                "original_root": str(original_root),
                "device": str(device),
                "parallel_envs": int(parallel_envs),
                "input_profile": v1._input_config().profile_id,
                "reward_profile": v1._reward_config().profile_id,
                "environment_stack": (
                    "RevengeEnv + ObservableHumanSpeedrunWrapper "
                    "(motor-observable)"
                ),
                "seed_last": int(seed_base)
                + int(seeds_per_level) * len(ordered_levels)
                - 1,
                "seed_embargo": {
                    name: [first, last]
                    for name, first, last in v1.EMBARGOED_SEED_RANGES
                },
                "reserved_engineering_seed_blocks": {
                    name: [first, last]
                    for name, first, last in (
                        RESERVED_ENGINEERING_SEED_BLOCKS
                    )
                },
                "formal_seed_consumption": False,
                **plan,
            },
        )

    model_cache: dict[Any, Any] = {}
    additions = _manifest_additions(
        policy_path=policy_path,
        policy_sha256=policy_sha256,
        script_sha256=script_sha256,
    )

    def _bound_batch(**kwargs: Any) -> list[dict[str, Any]]:
        # Module-global lookup keeps the batch collector monkeypatchable
        # in tests, mirroring how v1's own tests patch _collect_batch.
        return _collect_batch_dagger(
            policy_path=policy_path,
            device=str(device),
            model_cache=model_cache,
            **kwargs,
        )

    original_batch = v1._collect_batch
    original_write_manifest = v1._write_manifest

    def _write_manifest_with_dagger_keys(
        run_dir_argument: Path, **kwargs: Any
    ) -> Path:
        path = original_write_manifest(run_dir_argument, **kwargs)
        document = legacy._read_json(path)
        document.update(additions)
        legacy._write_json_atomic(path, document)
        return path

    v1._collect_batch = _bound_batch
    v1._write_manifest = _write_manifest_with_dagger_keys
    try:
        completion = v1.collect(
            original_root=original_root,
            run_dir=run_dir,
            levels=ordered_levels,
            seeds_per_level=int(seeds_per_level),
            seed_base=int(seed_base),
            parallel_envs=int(parallel_envs),
            max_ticks=int(max_ticks),
            device=str(device),
        )
    finally:
        v1._collect_batch = original_batch
        v1._write_manifest = original_write_manifest

    entries = completion["episodes"]
    dagger_completion = {
        "schema": COMPLETION_SCHEMA,
        "version": VERSION,
        "status": completion["status"],
        "completed_utc": legacy._utc_now(),
        "collection_mode": COLLECTION_MODE,
        "collector": {"path": str(SCRIPT_PATH), "sha256": script_sha256},
        "reused_v1_machinery": {
            "path": str(v1.SCRIPT_PATH),
            "sha256": legacy._sha256(v1.SCRIPT_PATH),
        },
        "student_policy": {
            "path": str(policy_path),
            "sha256": policy_sha256,
        },
        "student_inference_semantics": STUDENT_INFERENCE_SEMANTICS,
        "teacher_policy_id": v1.TEACHER_POLICY_ID,
        "label_semantics": LABEL_SEMANTICS,
        "executed_actions_semantics": EXECUTED_ACTIONS_SEMANTICS,
        "device": str(device),
        "run_dir": str(run_dir),
        "levels": list(ordered_levels),
        "seeds_per_level": int(seeds_per_level),
        "seed_base": int(seed_base),
        "max_ticks": int(max_ticks),
        "episodes": entries,
        "missing_episodes": completion["missing_episodes"],
        "errors": completion["errors"],
        "wins": completion["wins"],
        "losses": completion["losses"],
        "truncations": completion["truncations"],
        "total_ticks": completion["total_ticks"],
        "fire_edges_teacher_labels": sum(
            int(entry.get("fire_edges_raw_intent", 0)) for entry in entries
        ),
        "fire_edges_executed_student": sum(
            int(entry.get("fire_edges_executed_student", 0))
            for entry in entries
        ),
        "label_executed_verb_disagreement_ticks": sum(
            int(entry.get("label_executed_verb_disagreement_ticks", 0))
            for entry in entries
        ),
        "label_executed_action_disagreement_ticks": sum(
            int(entry.get("label_executed_action_disagreement_ticks", 0))
            for entry in entries
        ),
        "resume_quarantined_files": completion["resume_quarantined_files"],
        "episodes_manifest": completion["episodes_manifest"],
        "v1_completion": {
            "path": str(run_dir / "completion.json"),
            "sha256": legacy._sha256(run_dir / "completion.json"),
        },
        "formal_seed_consumption": False,
    }
    legacy._write_json_atomic(
        run_dir / "dagger_completion.json", dagger_completion
    )
    return dagger_completion


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--policy-path",
        required=True,
        type=Path,
        help="SB3 MaskablePPO zip of the student that EXECUTES every tick",
    )
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
        "--parallel-envs", type=int, default=v1.DEFAULT_PARALLEL_ENVS
    )
    parser.add_argument("--max-ticks", type=int, default=v1.DEFAULT_MAX_TICKS)
    parser.add_argument(
        "--device",
        default="cpu",
        help=(
            "student inference device; the teacher labeler is "
            "deterministic numpy code and needs no GPU"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    completion = collect(
        original_root=args.original_root,
        run_dir=args.run_dir,
        policy_path=args.policy_path,
        levels=args.levels,
        seeds_per_level=int(args.seeds_per_level),
        seed_base=int(args.seed_base),
        parallel_envs=int(args.parallel_envs),
        max_ticks=int(args.max_ticks),
        device=str(args.device),
    )
    print(
        json.dumps(
            {
                "status": completion["status"],
                "collection_mode": COLLECTION_MODE,
                "episodes": len(completion["episodes"]),
                "missing_episodes": len(completion["missing_episodes"]),
                "errors": completion["errors"],
                "wins": completion["wins"],
                "losses": completion["losses"],
                "truncations": completion["truncations"],
                "total_ticks": completion["total_ticks"],
                "label_executed_verb_disagreement_ticks": completion[
                    "label_executed_verb_disagreement_ticks"
                ],
                "student_policy": completion["student_policy"],
                "run_dir": completion["run_dir"],
                "episodes_manifest": completion["episodes_manifest"],
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    return 0 if completion["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
