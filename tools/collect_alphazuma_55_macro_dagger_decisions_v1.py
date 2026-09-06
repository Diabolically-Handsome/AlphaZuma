"""Macro-decision DAgger collector: the STUDENT drives, the teacher labels.

Why this exists (jungle9 diagnosis, 2026-08-21)
-----------------------------------------------
The strengthened-init recipe (240 teacher episodes -> BC) that took
Jungle3 from 2/8 to 16/16 produced a 0/16 baseline on Jungle9 despite
healthy offline calibration (fire_ratio 1.16, fire_recall 0.88,
aim_within_3 0.86).  All sixteen panel episodes were ACTIVE losses: the
student shoots well (61-83 on-target shots, real scores) but drifts
into deep-chain danger states the teacher never visits - the teacher
keeps the chain far back, so the 240-episode collection contains no
defuse-the-crisis supervision at all, and Jungle9's ~434 decisions per
episode (2x Jungle3) compound the drift.  That is the textbook DAgger
gap: supervision must cover the states the STUDENT actually reaches.

What this collector does
------------------------
``collect_alphazuma_55_macro_decisions_v1``'s decision-point loop with
exactly one change of authority: at every macro decision the EXECUTED
action comes from a student MaskablePPO checkpoint (masked
deterministic predict on the stacked observation, the native macro
interface - no adapter, no fallback possible), while the RECORDED
label remains the settled strategic teacher's adapter-mapped proposal
on the raw leading frame, byte-identical to collector v1's labels.

The episode NPZ, manifest, and completion schemas are therefore
UNCHANGED (``distill_alphazuma_55_macro_decisions_v1`` consumes the
output directly; aggregate it with the teacher collection for the
classic DAgger recipe).  ``outcome``/``teacher_won`` fields describe
the STUDENT-driven episode - config and completion record
``outcome_semantics: student_driven`` so no receipt can be misread.
Each episode additionally gets a SIDECAR
``<episode>.executed.json`` holding the executed student action per
decision plus label-agreement telemetry; the NPZ label stream stays
pure teacher.

``--teacher-drive-prob p`` optionally hands the wheel back to the
teacher for a Bernoulli(p) fraction of decisions (deterministic
per-episode rng seeded by the episode seed), for beta-mixed DAgger
rounds.  The default 0.0 is the classic round-1 configuration.

Seed discipline: the same embargo/reservation validation as collector
v1 (``validate_seed_plan``); runs must use a fresh engineering block
disjoint from every training reservation.  ``formal_seed_consumption``
stays false: no formal seed is ever touched.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import numpy as np

from tools import collect_alphazuma_55_macro_decisions_v1 as c1
from tools import collect_alphazuma_55_park_settle_episodes_v1 as episodes_v1
from tools import distill_alphazuma_55 as legacy
from tools import probe_alphazuma_55_macro_eval_v1 as macro_eval
from tools import probe_alphazuma_55_macro_native_eval_v1 as native_eval

SCRIPT_PATH = Path(__file__).resolve()
VERSION = 1
CONFIG_SCHEMA = "zuma-rl.alphazuma-55-macro-dagger-decisions-v1-config"
STATUS_SCHEMA = "zuma-rl.alphazuma-55-macro-dagger-decisions-v1-status"
COMPLETION_SCHEMA = "zuma-rl.alphazuma-55-macro-dagger-decisions-v1"
DEFAULT_TEACHER_DRIVE_PROB = 0.0


def load_student_model(model_path: Path, device: str) -> Any:
    """Load the macro-native student exactly as the eval probe does.

    Importing ``native_eval`` (module top) makes the pickled feature
    extractor class importable before ``MaskablePPO.load`` runs.
    """

    from sb3_contrib import MaskablePPO

    return MaskablePPO.load(str(Path(model_path).resolve(strict=True)), device=str(device))


def _collect_batch_dagger(
    *,
    original_root: Path,
    tasks: Sequence[c1.EpisodeTask],
    record_flags: Sequence[bool],
    max_ticks: int,
    hold_ticks: int,
    stack_lags: tuple[int, ...],
    episodes_dir: Path,
    spill_dir: Path,
    on_episode: Callable[[dict[str, Any]], None],
    student_model: Any,
    teacher_drive_prob: float,
    vector_factory: Callable[..., Any] | None = None,
    teachers_factory: Callable[..., list[Any]] | None = None,
    masks_provider: Callable[[Any], np.ndarray] | None = None,
) -> list[dict[str, Any]]:
    """Collector v1's batch loop with student-driven execution.

    Per active environment and decision point:

    * the teacher is queried on the raw leading slice and its proposal
      is adapter-mapped -- that mapped action IS the recorded label,
      byte-identical to v1;
    * the student model's masked deterministic predict on the stacked
      observation (batched over active envs) is the EXECUTED action,
      unless the per-episode rng hands this decision to the teacher
      (``teacher_drive_prob``);
    * the executed action, its authority, and label agreement land in
      the episode's ``.executed.json`` sidecar.
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
    prob = float(teacher_drive_prob)
    if not 0.0 <= prob <= 1.0:
        raise ValueError("teacher_drive_prob must be in [0, 1]")
    lags = c1.parse_stack_lags(stack_lags)
    if vector_factory is None:
        vector_factory = c1._default_vector_factory
    if teachers_factory is None:
        teachers_factory = c1._default_teachers_factory
    if masks_provider is None:
        masks_provider = c1._default_masks_provider

    vector = vector_factory(
        original_root=original_root,
        tasks=tasks,
        max_ticks=int(max_ticks),
        hold_ticks=int(hold_ticks),
        stack_lags=lags,
    )
    recorders: list[c1._DecisionRecorder | None] = [None] * count
    executed_rows: list[list[dict[str, Any]]] = [[] for _ in range(count)]
    drive_rngs = [np.random.default_rng(int(task.seed)) for task in tasks]
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
                recorders[position] = c1._DecisionRecorder(
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
            entry = c1._finalize_episode(
                task=tasks[index],
                recorder=recorder,
                final_info=info,
                guard_exhausted=guard,
                episodes_dir=episodes_dir,
                wall_seconds=time.perf_counter() - batch_started,
            )
            rows = executed_rows[index]
            agree = sum(
                1
                for row in rows
                if row["executed"] == row["teacher_label"]
            )
            sidecar = {
                "schema": "zuma-rl.alphazuma-55-macro-dagger-executed-v1",
                "version": VERSION,
                "episode_file": str(entry.get("path", "")),
                "level_id": tasks[index].level_id,
                "seed": int(tasks[index].seed),
                "outcome_semantics": "student_driven",
                "teacher_drive_prob": prob,
                "decisions": len(rows),
                "teacher_driven_decisions": sum(
                    1 for row in rows if row["authority"] == "teacher"
                ),
                "label_agreement": {
                    "exact": agree,
                    "exact_rate": (agree / len(rows)) if rows else None,
                },
                "executed_actions": rows,
            }
            sidecar_path = (
                episodes_dir / f"{tasks[index].file_name}.executed.json"
            )
            legacy._write_json_atomic(sidecar_path, sidecar)
            entry = dict(entry)
            entry["executed_sidecar"] = sidecar_path.name
            entry["executed_sidecar_sha256"] = legacy._sha256(sidecar_path)
            entries.append(entry)
            on_episode(entry)

        for _ in range(int(max_ticks) + 1):
            if bool(np.all(finished)):
                break
            macro_masks = np.asarray(
                masks_provider(vector), dtype=np.bool_
            ).reshape(count, -1)
            if macro_masks.shape[1] != c1.MACRO_MASK_WIDTH:
                raise ValueError(
                    f"macro mask width must be {c1.MACRO_MASK_WIDTH}, got "
                    f"{macro_masks.shape}"
                )
            actions = np.zeros((count, 2), dtype=np.int64)
            active = ~finished
            active_indices = [int(p) for p in np.flatnonzero(active)]
            student_actions: dict[int, np.ndarray] = {}
            if active_indices:
                predicted, _ = student_model.predict(
                    observations[active_indices],
                    deterministic=True,
                    action_masks=macro_masks[active_indices],
                )
                predicted = np.asarray(predicted, dtype=np.int64).reshape(
                    len(active_indices), 2
                )
                for row, index in enumerate(active_indices):
                    student_actions[index] = predicted[row]
            for index in active_indices:
                per_tick = np.asarray(
                    teachers[index].act(observations[index]),
                    dtype=np.int64,
                ).reshape(2)
                teacher_action, fallback = macro_eval.adapt_per_tick_action(
                    per_tick, macro_masks[index]
                )
                teacher_drives = bool(
                    prob > 0.0 and drive_rngs[index].random() < prob
                )
                executed = (
                    np.asarray(teacher_action, dtype=np.int64)
                    if teacher_drives
                    else student_actions[index]
                )
                actions[index] = executed
                recorder = recorders[index]
                if recorder is not None:
                    recorder.record_decision(
                        observation=observations[index],
                        macro_action=teacher_action,
                        macro_mask=macro_masks[index],
                        teacher_per_tick_action=per_tick,
                        fallback=fallback,
                    )
                    executed_rows[index].append(
                        {
                            "executed": [int(v) for v in executed],
                            "teacher_label": [
                                int(v) for v in np.asarray(teacher_action)
                            ],
                            "authority": (
                                "teacher" if teacher_drives else "student"
                            ),
                        }
                    )
            step_observations, _, dones, infos = vector.step(actions)
            for index in active_indices:
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
                finalize(int(position), last_infos[int(position)], True)
    finally:
        for recorder in recorders:
            if recorder is not None:
                recorder.discard()
        vector.close()
    return entries


def collect_dagger(
    *,
    original_root: Path,
    run_dir: Path,
    student_model_path: Path,
    levels: Sequence[str],
    seeds_per_level: int,
    seed_base: int,
    parallel_envs: int = c1.DEFAULT_PARALLEL_ENVS,
    max_ticks: int = c1.DEFAULT_MAX_TICKS,
    hold_ticks: int = c1.DEFAULT_HOLD_TICKS,
    stack_lags: Any = c1.DEFAULT_STACK_LAGS,
    teacher_drive_prob: float = DEFAULT_TEACHER_DRIVE_PROB,
    device: str = "cpu",
) -> dict[str, Any]:
    """Collect (or resume) a student-driven, teacher-labeled archive."""

    started = time.perf_counter()
    original_root = Path(original_root).expanduser().resolve(strict=True)
    run_dir = Path(run_dir).expanduser().resolve()
    student_model_path = Path(student_model_path).expanduser().resolve(
        strict=True
    )
    ordered_levels = c1._validate_levels(levels)
    lags = c1.parse_stack_lags(stack_lags)
    if int(max_ticks) < 1:
        raise ValueError("max_ticks must be positive")
    prob = float(teacher_drive_prob)
    if not 0.0 <= prob <= 1.0:
        raise ValueError("teacher_drive_prob must be in [0, 1]")
    tasks = c1.plan_tasks(
        levels=ordered_levels,
        seeds_per_level=int(seeds_per_level),
        seed_base=int(seed_base),
    )
    c1.validate_seed_plan(tasks)
    episodes_dir = run_dir / c1.EPISODES_DIRNAME
    spill_dir = run_dir / c1.SPILL_DIRNAME
    episodes_dir.mkdir(parents=True, exist_ok=True)
    spill_dir.mkdir(parents=True, exist_ok=True)
    for stale in spill_dir.glob("*.spill"):
        stale.unlink()

    student_sha = legacy._sha256(student_model_path)
    plan = {
        "levels": list(ordered_levels),
        "seeds_per_level": int(seeds_per_level),
        "seed_base": int(seed_base),
        "max_ticks": int(max_ticks),
        "hold_ticks": int(hold_ticks),
        "stack_lags": [int(lag) for lag in lags],
        "student_model_sha256": student_sha,
        "teacher_drive_prob": prob,
    }
    config_path = run_dir / "config.json"
    if config_path.exists():
        existing = legacy._read_json(config_path)
        previous = {key: existing.get(key) for key in plan}
        if previous != plan:
            raise ValueError(
                f"run dir {run_dir} was planned with a different DAgger "
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
                "base_collector": {
                    "module": (
                        "tools.collect_alphazuma_55_macro_decisions_v1"
                    ),
                    "path": str(c1.SCRIPT_PATH),
                    "sha256": legacy._sha256(c1.SCRIPT_PATH),
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
                "teacher_policy_id": c1.TEACHER_POLICY_ID,
                "teacher_observation": "raw_current_frame_leading_slice",
                "student_model": {
                    "path": str(student_model_path),
                    "sha256": student_sha,
                    "predict": (
                        "masked deterministic on the stacked observation "
                        "with the wrapper's own (3 + 180) macro mask - "
                        "the native interface, no adapter, no fallback"
                    ),
                },
                "outcome_semantics": "student_driven",
                "label_semantics": (
                    "NPZ macro_actions are PURE TEACHER labels "
                    "(collector-v1 byte-identical); executed student "
                    "actions live in per-episode .executed.json sidecars"
                ),
                "decision_interface": c1._decision_interface_contract(
                    int(hold_ticks)
                ),
                "observation_stack": c1._observation_stack_contract(lags),
                "original_root": str(original_root),
                "device": str(device),
                "parallel_envs": int(parallel_envs),
                "input_profile": episodes_v1._input_config().profile_id,
                "reward_profile": episodes_v1._reward_config().profile_id,
                "seed_last": int(seed_base)
                + int(seeds_per_level) * len(ordered_levels)
                - 1,
                "seed_embargo": {
                    name: [first, last]
                    for name, first, last in c1.EMBARGOED_SEED_RANGES
                },
                "formal_seed_consumption": False,
                "training_authority": False,
                **plan,
            },
        )

    entries_by_key = c1._load_completed_entries(run_dir)
    reused_keys = set(entries_by_key)
    verified_paths = {
        (run_dir / str(entry["path"])).resolve()
        for entry in entries_by_key.values()
    }
    quarantined_files = episodes_v1._quarantine_unverified_episode_files(
        episodes_dir, verified_paths
    )

    def write_manifest() -> Path:
        return c1._write_manifest(
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

    write_manifest()
    student_model = load_student_model(student_model_path, device)
    errors: list[dict[str, Any]] = []
    batches = c1._batches(tasks, int(parallel_envs))
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
            _collect_batch_dagger(
                original_root=original_root,
                tasks=batch,
                record_flags=record_flags,
                max_ticks=int(max_ticks),
                hold_ticks=int(hold_ticks),
                stack_lags=lags,
                episodes_dir=episodes_dir,
                spill_dir=spill_dir,
                on_episode=on_episode,
                student_model=student_model,
                teacher_drive_prob=prob,
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

    write_manifest()
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
            "student_wins": sum(
                entry["outcome"] == "win" for entry in members
            ),
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
            "seeds": [int(entry["seed"]) for entry in members],
        }
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
        "teacher_policy_id": c1.TEACHER_POLICY_ID,
        "student_model": {
            "path": str(student_model_path),
            "sha256": student_sha,
        },
        "outcome_semantics": "student_driven",
        "teacher_drive_prob": prob,
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
        "formal_seed_consumption": False,
        "training_authority": False,
        "episodes": ordered_entries,
        "missing_episodes": missing,
        "resume_quarantined_files": quarantined_files,
        "errors": errors,
        "student_wins": sum(
            entry["outcome"] == "win" for entry in ordered_entries
        ),
        "per_level": per_level,
    }
    legacy._write_json_atomic(run_dir / "completion.json", completion)
    return completion


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--student-model", type=Path, required=True)
    parser.add_argument("--levels", nargs="+", required=True)
    parser.add_argument("--seeds-per-level", type=int, required=True)
    parser.add_argument("--seed-base", type=int, required=True)
    parser.add_argument(
        "--parallel-envs", type=int, default=c1.DEFAULT_PARALLEL_ENVS
    )
    parser.add_argument("--max-ticks", type=int, default=c1.DEFAULT_MAX_TICKS)
    parser.add_argument(
        "--hold-ticks", type=int, default=c1.DEFAULT_HOLD_TICKS
    )
    parser.add_argument(
        "--stack-lags",
        default=",".join(str(lag) for lag in c1.DEFAULT_STACK_LAGS),
    )
    parser.add_argument(
        "--teacher-drive-prob",
        type=float,
        default=DEFAULT_TEACHER_DRIVE_PROB,
    )
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    completion = collect_dagger(
        original_root=args.original_root,
        run_dir=args.run_dir,
        student_model_path=args.student_model,
        levels=args.levels,
        seeds_per_level=args.seeds_per_level,
        seed_base=args.seed_base,
        parallel_envs=args.parallel_envs,
        max_ticks=args.max_ticks,
        hold_ticks=args.hold_ticks,
        stack_lags=args.stack_lags,
        teacher_drive_prob=args.teacher_drive_prob,
        device=args.device,
    )
    print(
        json.dumps(
            {
                "status": completion["status"],
                "episodes": len(completion["episodes"]),
                "student_wins": completion["student_wins"],
                "errors": len(completion["errors"]),
            }
        )
    )
    return 0 if completion["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
