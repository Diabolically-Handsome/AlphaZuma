"""Probe V4 override windows with honest closed-loop accounting.

Mirrors the paired student_only versus recovery_intervention structure of
tools/probe_alphazuma_55_recovery_intervention_v2.py (same fixed 12-level
cross-world panel, same engineering seed base, identical task seeds for
both modes, SubprocVecEnv execution) with three corrections:

1.  The executor is tools/alphazuma_55_recovery_intervention_v4.py:
    override windows where the teacher controls both verb and aim on
    every tick, event-level triggers/handback, and veto accounting.
2.  BOTH modes are instrumented with the W1 closed-loop metrics
    (tools/alphazuma_55_closed_loop_metrics_v1.py): decision-relevant
    disagreement rates, wait-inclusive aim intent error, release-angle
    attribution through the 18-tick wrapped commit-to-release delay,
    shots_on_target_rate, and honest execution fractions.
3.  The completion decision gate reads
    decision_relevant_teacher_fraction and
    teacher_initiated_shot_fraction; tick_teacher_fraction is reported
    for continuity but never gated.

The executed aim at release is observed through the info stream the
wrapped stack already publishes: HumanSpeedrunWrapper.step exposes
executed_verb, executed_aim_bin, desired_verb, and button_enqueued under
info["human_speedrun"] on every tick, so no wrapper subclass is needed
and src/zuma_rl/human_speedrun.py stays byte-identical.  Fire commits
are attributed to whichever side executed the enqueuing tick, matching
the wrapper's rising-edge button semantics.

The student model path is a direct CLI argument.  The 55/55 teacher
(CurveAwareSettledStrategicRevengeTeacher) has no checkpoint file: it is
deterministic code constructed from each environment's teacher_spec(),
pinned here by identifier.  An optional --preregistration binds a
zuma-rl.alphazuma-55-recovery-intervention-v4-probe-preregistration
document into the receipts; a DRAFT_NOT_FROZEN document is accepted but
recorded as unfrozen so the run cannot masquerade as a frozen probe.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any, Callable, Mapping, Sequence

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import numpy as np

from tools import distill_alphazuma_55 as legacy
from tools import distill_alphazuma_55_motor_observable_replay_v2 as motor
from tools import distill_alphazuma_55_polar_dagger_v1 as dagger
from tools import distill_alphazuma_55_polar_intent_wide_v1 as intent
from tools import probe_alphazuma_55_recovery_intervention_v1 as base
from tools.alphazuma_55_closed_loop_metrics_v1 import (
    AimIntentErrorAccumulator,
    HonestExecutionAccounting,
    ReleaseAngleConfig,
    ReleaseAngleTracker,
    ReleaseTickRecord,
    TickClassificationAccumulator,
    WRAPPED_STACK_COMMIT_TO_RELEASE_TICKS,
)
from tools.alphazuma_55_recovery_intervention_v4 import (
    DecisionGateConfigV4,
    RecoveryInterventionConfigV4,
    RecoveryInterventionControllerV4,
    evaluate_decision_gate,
)
from zuma_rl.alphazuma_55 import INCLUDED_LEVELS


SCRIPT_PATH = Path(__file__).resolve()
CONTROLLER_PATH = (
    SCRIPT_PATH.parent / "alphazuma_55_recovery_intervention_v4.py"
).resolve()
METRICS_PATH = (
    SCRIPT_PATH.parent / "alphazuma_55_closed_loop_metrics_v1.py"
).resolve()
EXPECTED_MODES = base.EXPECTED_MODES
COMPLETION_SCHEMA = (
    "zuma-rl.alphazuma-55-recovery-intervention-v4-probe-completion"
)
PREREGISTRATION_SCHEMA = (
    "zuma-rl.alphazuma-55-recovery-intervention-v4-probe-preregistration"
)
TEACHER_ID = "curve-aware-settled-strategic-v3"
DEFAULT_SEED_BASE = 1_400_920_000
DEFAULT_MAX_TICKS = 30_000
DEFAULT_LEVELS = (
    "Jungle1",
    "Jungle9",
    "village3",
    "village8",
    "city1",
    "city9",
    "Coast1",
    "Coast9",
    "grotto1",
    "grotto9",
    "volcano1",
    "volcano9",
)
SHOT_TOLERANCE_BINS = 2


def _fraction(numerator: int, denominator: int) -> float:
    """Honest ratio: an empty denominator yields NaN, never a free pass."""

    if int(denominator) <= 0:
        return float("nan")
    return int(numerator) / int(denominator)


def _json_safe(value: Any) -> Any:
    """Replace non-finite floats with None so receipts stay allow_nan=False.

    The honest fractions deliberately report NaN on empty denominators;
    the gate is evaluated on the in-memory NaN (which fails closed) and
    the serialized receipt records null.
    """

    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _validate_levels(level_ids: Sequence[str]) -> tuple[str, ...]:
    levels = tuple(str(value) for value in level_ids)
    if not levels or len(levels) != len(set(levels)):
        raise ValueError("probe levels must be non-empty and unique")
    canonical = {value.casefold() for value in INCLUDED_LEVELS}
    if any(level.casefold() not in canonical for level in levels):
        raise ValueError("probe level escaped the included full55 inventory")
    return levels


def _bind_preregistration(path: Path) -> dict[str, Any]:
    """Bind a V4 preregistration document (draft or frozen) by bytes."""

    path = path.resolve(strict=True)
    prereg = base._read(path)
    if prereg.get("schema") != PREREGISTRATION_SCHEMA:
        raise ValueError("unexpected recovery intervention V4 preregistration")
    status = str(prereg.get("status", ""))
    if status not in {"DRAFT_NOT_FROZEN", "FROZEN_BEFORE_PROBE"}:
        raise ValueError(f"unexpected V4 preregistration status: {status}")
    return {
        "path": str(path),
        "sha256": base._sha256(path),
        "status": status,
        "frozen": status == "FROZEN_BEFORE_PROBE",
    }


def _default_masks_provider(vector: Any) -> np.ndarray:
    from sb3_contrib.common.maskable.utils import get_action_masks

    return np.asarray(get_action_masks(vector), dtype=np.bool_)


def _release_payload(summary: Any) -> dict[str, Any]:
    """Serialize a ReleaseAngleSummary without the per-shot list."""

    return {
        "shot_count": int(summary.shot_count),
        "shots_on_target_rate": float(summary.shots_on_target_rate),
        "mean_release_angle_error_bins": float(
            summary.mean_release_angle_error_bins
        ),
        "pending_commits": int(summary.pending_commits),
        "commit_to_release_ticks": WRAPPED_STACK_COMMIT_TO_RELEASE_TICKS,
        "tolerance_bins": SHOT_TOLERANCE_BINS,
        "per_executor": {
            name: asdict(row) for name, row in summary.per_executor.items()
        },
    }


def _aim_error_payload(errors: Sequence[int]) -> dict[str, Any] | None:
    if not errors:
        return None
    merged = AimIntentErrorAccumulator()
    merged.errors.extend(int(value) for value in errors)
    return asdict(merged.summary())


def _run_mode(
    *,
    mode: str,
    model: Any,
    vector: Any,
    teachers: Sequence[Any],
    level_ids: Sequence[str],
    seed_base: int,
    max_ticks: int,
    controller_config: RecoveryInterventionConfigV4,
    masks_provider: Callable[[Any], np.ndarray] | None = None,
) -> dict[str, Any]:
    """Run one probe mode with full W1 closed-loop instrumentation.

    The vector, model, teachers, and mask provider are injectable so the
    completion contract can be exercised on CPU with stubs; the real run
    passes a SubprocVecEnv, a MaskablePPO student, and settled strategic
    teachers built from each environment's teacher_spec().
    """

    if mode not in EXPECTED_MODES:
        raise ValueError(f"unexpected probe mode: {mode}")
    count = len(level_ids)
    if len(teachers) != count:
        raise ValueError("one teacher per probe level is required")
    if masks_provider is None:
        masks_provider = _default_masks_provider
    started = time.perf_counter()
    vector.seed(seed_base)
    observations = vector.reset()
    controllers = [
        RecoveryInterventionControllerV4(controller_config)
        for _ in level_ids
    ]
    tick_classifiers = [
        TickClassificationAccumulator(
            fire_aim_threshold=controller_config.fire_aim_disagreement_bins,
        )
        for _ in level_ids
    ]
    aim_errors = [AimIntentErrorAccumulator() for _ in level_ids]
    release_trackers = [
        ReleaseAngleTracker(
            ReleaseAngleConfig(
                commit_to_release_ticks=WRAPPED_STACK_COMMIT_TO_RELEASE_TICKS,
                tolerance_bins=SHOT_TOLERANCE_BINS,
            )
        )
        for _ in level_ids
    ]
    accounting = [HonestExecutionAccounting() for _ in level_ids]
    previous_infos: list[dict[str, Any]] = [
        {"score": 0, "chain_length": 0} for _ in level_ids
    ]
    last_infos: list[dict[str, Any]] = [{} for _ in level_ids]
    phase_counts = [Counter() for _ in level_ids]
    finished = np.zeros(count, dtype=np.bool_)
    steps = np.zeros(count, dtype=np.int64)
    teacher_steps = np.zeros(count, dtype=np.int64)
    student_steps = np.zeros(count, dtype=np.int64)
    transitions = 0

    for _ in range(max_ticks + 1):
        masks = masks_provider(vector)
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
        tick_executors: list[str | None] = [None] * count
        tick_teacher_bins = np.zeros(count, dtype=np.int64)
        for position in np.flatnonzero(active):
            index = int(position)
            dagger._validate_student_action(
                student_actions[index], masks[index]
            )
            teacher_raw = np.asarray(
                teachers[index].act(observations[index]),
                dtype=np.int64,
            )
            _, effective, _, _ = intent._intent_label(
                teacher_raw, masks[index]
            )
            tick_teacher_bins[index] = int(effective[1])
            tick_class = tick_classifiers[index].add(
                effective, student_actions[index]
            )
            aim_errors[index].add(
                int(effective[1]), int(student_actions[index][1])
            )
            if mode == "recovery_intervention":
                decision = controllers[index].decide(
                    tick=int(steps[index]),
                    teacher_action=effective,
                    student_action=student_actions[index],
                    previous_info=previous_infos[index],
                )
                phase = decision.phase
                if decision.execute_teacher:
                    actions[index] = effective
                    teacher_steps[index] += 1
                    executor = "teacher"
                else:
                    student_steps[index] += 1
                    executor = "student"
            else:
                phase = (
                    "student_disagreement"
                    if tick_class.disagreement
                    else "student_nominal"
                )
                student_steps[index] += 1
                executor = "student"
            phase_counts[index][phase] += 1
            accounting[index].add(
                executor=executor,
                teacher_verb=int(effective[0]),
                student_verb=int(student_actions[index][0]),
                phase=phase,
            )
            tick_executors[index] = executor

        observations, _, dones, infos = vector.step(actions)
        transitions += int(np.count_nonzero(active))
        for position in np.flatnonzero(active):
            index = int(position)
            info = dict(infos[index])
            speedrun = info.get("human_speedrun")
            if isinstance(speedrun, Mapping):
                commit = (
                    bool(speedrun.get("button_enqueued", False))
                    and int(speedrun.get("desired_verb", 0)) == 1
                )
                release_trackers[index].observe(
                    ReleaseTickRecord(
                        tick=int(steps[index]),
                        executed_verb=int(speedrun.get("executed_verb", 0)),
                        executed_aim_bin=int(
                            speedrun.get("executed_aim_bin", 0)
                        ),
                        teacher_target_bin=int(tick_teacher_bins[index]),
                        fire_edge_executor=(
                            tick_executors[index] if commit else None
                        ),
                    )
                )
            previous_infos[index] = info
            last_infos[index] = info
            if bool(dones[index]):
                finished[index] = True
        steps[active] += 1
        if bool(np.all(finished)):
            break
    if not bool(np.all(finished)):
        missing = [
            level_ids[index]
            for index in range(count)
            if not finished[index]
        ]
        raise RuntimeError(f"probe episodes exceeded guard: {missing}")

    episodes: list[dict[str, Any]] = []
    for index, level_id in enumerate(level_ids):
        info = last_infos[index]
        controller_summary = (
            controllers[index].summary()
            if mode == "recovery_intervention"
            else None
        )
        release_summary = release_trackers[index].summary()
        episodes.append(
            {
                "level_id": level_id,
                "seed": seed_base + index,
                "outcome": info.get("outcome"),
                "native_outcome": info.get("native_outcome"),
                "time_limit_truncated": bool(
                    info.get("TimeLimit.truncated", False)
                ),
                "ticks": int(info.get("ticks", steps[index])),
                "score": int(info.get("score", 0)),
                "decisions": int(steps[index]),
                "student_execution_steps": int(student_steps[index]),
                "teacher_execution_steps": int(teacher_steps[index]),
                "student_fire_vetoes": (
                    int(controller_summary["student_fire_vetoes"])
                    if controller_summary is not None
                    else 0
                ),
                "closed_loop_metrics": {
                    "tick_classification": asdict(
                        tick_classifiers[index].summary()
                    ),
                    "aim_intent_error": _aim_error_payload(
                        aim_errors[index].errors
                    ),
                    "release_angle": _release_payload(release_summary),
                    "honest_execution": asdict(accounting[index].summary()),
                },
                "phase_counts": dict(phase_counts[index]),
                "intervention": controller_summary,
                "observation_capacity_overflow": bool(
                    info.get("observation_capacity_overflow", False)
                ),
            }
        )

    total_decisions = int(steps.sum())
    total_teacher = int(teacher_steps.sum())
    total_ticks = sum(row.total_ticks for row in tick_classifiers)
    disagreement_ticks = sum(
        row.disagreement_ticks for row in tick_classifiers
    )
    decision_relevant = sum(
        row.decision_relevant_ticks for row in accounting
    )
    decision_relevant_teacher = sum(
        row.decision_relevant_teacher_ticks for row in accounting
    )
    teacher_shots = sum(row.teacher_initiated_shots for row in accounting)
    student_shots = sum(row.student_initiated_shots for row in accounting)
    all_shots = [
        shot for tracker in release_trackers for shot in tracker.shots
    ]
    on_target = sum(1 for shot in all_shots if shot.on_target)
    mean_release_error = (
        float(
            np.mean([shot.release_angle_error_bins for shot in all_shots])
        )
        if all_shots
        else float("nan")
    )
    merged_errors = [
        error for row in aim_errors for error in row.errors
    ]
    interventions = (
        [row["intervention"] for row in episodes]
        if mode == "recovery_intervention"
        else []
    )
    return {
        "mode": mode,
        "episodes": episodes,
        "summary": {
            "attempts": len(episodes),
            "wins": sum(row["outcome"] == "win" for row in episodes),
            "losses": sum(row["outcome"] == "loss" for row in episodes),
            "truncations": sum(
                row["time_limit_truncated"] for row in episodes
            ),
            "total_score": sum(row["score"] for row in episodes),
            "total_decisions": total_decisions,
            "teacher_execution_steps": total_teacher,
            "tick_teacher_fraction": _fraction(
                total_teacher, total_decisions
            ),
            "decision_relevant_ticks": int(decision_relevant),
            "decision_relevant_teacher_ticks": int(
                decision_relevant_teacher
            ),
            "decision_relevant_teacher_fraction": _fraction(
                decision_relevant_teacher, decision_relevant
            ),
            "teacher_initiated_shots": int(teacher_shots),
            "student_initiated_shots": int(student_shots),
            "teacher_initiated_shot_fraction": _fraction(
                teacher_shots, teacher_shots + student_shots
            ),
            "per_tick_disagreement_rate": _fraction(
                disagreement_ticks, total_ticks
            ),
            "decision_relevant_disagreement_rate": _fraction(
                disagreement_ticks, decision_relevant
            ),
            "shot_count": len(all_shots),
            "shots_on_target_rate": _fraction(on_target, len(all_shots)),
            "mean_release_angle_error_bins": mean_release_error,
            "aim_intent_error": _aim_error_payload(merged_errors),
            "student_fire_vetoes": sum(
                row["student_fire_vetoes"] for row in episodes
            ),
            "teacher_fire_edges": sum(
                int(row["teacher_fire_edges"]) for row in interventions
            ),
            "interventions": sum(
                int(row["interventions"]) for row in interventions
            ),
            "recovery_successes": sum(
                int(row["recovery_successes"]) for row in interventions
            ),
            "agreement_handbacks": sum(
                int(row["agreement_handbacks"]) for row in interventions
            ),
            "forced_handoffs": sum(
                int(row["forced_handoffs"]) for row in interventions
            ),
        },
        "runtime": {
            "wall_seconds": time.perf_counter() - started,
            "active_vector_transitions": transitions,
            "parallel_envs": count,
        },
    }


def _decision(
    *,
    gate: DecisionGateConfigV4,
    level_ids: Sequence[str],
    modes: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    by_mode = {str(row["mode"]): row for row in modes}
    baseline = by_mode["student_only"]
    recovery = by_mode["recovery_intervention"]
    baseline_by_level = {
        str(row["level_id"]): row for row in baseline["episodes"]
    }
    recovery_by_level = {
        str(row["level_id"]): row for row in recovery["episodes"]
    }
    paired_rows = []
    for level_id in level_ids:
        left = baseline_by_level[level_id]
        right = recovery_by_level[level_id]
        paired_rows.append(
            {
                "level_id": level_id,
                "seed": left["seed"],
                "student_only_outcome": left["outcome"],
                "recovery_outcome": right["outcome"],
                "student_only_score": left["score"],
                "recovery_score": right["score"],
                "score_delta": right["score"] - left["score"],
            }
        )
    verdict = evaluate_decision_gate(
        gate=gate,
        student_only_wins=int(baseline["summary"]["wins"]),
        student_only_total_score=int(baseline["summary"]["total_score"]),
        intervention_wins=int(recovery["summary"]["wins"]),
        intervention_total_score=int(recovery["summary"]["total_score"]),
        paired_score_deltas=[row["score_delta"] for row in paired_rows],
        interventions=int(recovery["summary"]["interventions"]),
        recovery_successes=int(recovery["summary"]["recovery_successes"]),
        tick_teacher_fraction=float(
            recovery["summary"]["tick_teacher_fraction"]
        ),
        decision_relevant_teacher_fraction=float(
            recovery["summary"]["decision_relevant_teacher_fraction"]
        ),
        teacher_initiated_shot_fraction=float(
            recovery["summary"]["teacher_initiated_shot_fraction"]
        ),
    )
    return {
        "status": verdict["status"],
        "checks": verdict["checks"],
        "gate": verdict["gate"],
        "reported_not_gated": verdict["reported_not_gated"],
        "paired_rows": paired_rows,
        "training_authorized_by_this_probe": False,
        "next_step_if_pass": (
            "freeze the DRAFT preregistration thresholds and rerun as a "
            "frozen probe before any recovery-priority training pilot"
        ),
    }


def _build_completion(
    *,
    modes: Sequence[dict[str, Any]],
    decision: dict[str, Any],
    source: dict[str, Any],
    runtime: dict[str, Any],
    wall_seconds: float,
    controller_config: RecoveryInterventionConfigV4,
    gate: DecisionGateConfigV4,
    preregistration: dict[str, Any] | None,
) -> dict[str, Any]:
    """Assemble the completion receipt; NaN metrics serialize as null."""

    return _json_safe(
        {
            "schema": COMPLETION_SCHEMA,
            "version": 1,
            "status": "COMPLETE",
            "completed_utc": base._utc_now(),
            "wall_seconds": float(wall_seconds),
            "runtime": dict(runtime),
            "teacher": {
                "id": TEACHER_ID,
                "construction": (
                    "deterministic code built from each environment's "
                    "teacher_spec(); no checkpoint file exists"
                ),
            },
            "source": dict(source),
            "controller_config": controller_config.contract(),
            "decision_gate": gate.contract(),
            "preregistration": preregistration,
            "modes": list(modes),
            "decision": decision,
            "formal_seed_consumption": False,
            "formal_candidate_authority": False,
            "training_authority": False,
        }
    )


def _build_vector(
    *,
    original_root: Path,
    level_ids: Sequence[str],
    max_ticks: int,
) -> Any:
    from stable_baselines3.common.vec_env import SubprocVecEnv

    factories = [
        motor._make_motor_env_factory(
            original_root=original_root,
            level_id=level_id,
            max_ticks=max_ticks,
            input_config=base._input_config(),
            reward_config=base._reward_config(),
        )
        for level_id in level_ids
    ]
    return SubprocVecEnv(factories, start_method="forkserver")


def _build_teachers(vector: Any) -> list[Any]:
    specifications = vector.env_method("teacher_spec")
    return [
        dagger.CurveAwareSettledStrategicRevengeTeacher(
            legacy._teacher_spec(value)
        )
        for value in specifications
    ]


def run(
    *,
    model_path: Path,
    original_root: Path,
    run_dir: Path,
    level_ids: Sequence[str] = DEFAULT_LEVELS,
    seed_base: int = DEFAULT_SEED_BASE,
    max_ticks: int = DEFAULT_MAX_TICKS,
    device: str = "cpu",
    controller_config: RecoveryInterventionConfigV4 | None = None,
    gate: DecisionGateConfigV4 | None = None,
    preregistration_path: Path | None = None,
) -> dict[str, Any]:
    model_path = model_path.resolve(strict=True)
    original_root = original_root.resolve(strict=True)
    run_dir = run_dir.resolve()
    if run_dir.exists():
        raise FileExistsError(f"V4 probe output already exists: {run_dir}")
    levels = _validate_levels(level_ids)
    controller_config = (
        controller_config
        if controller_config is not None
        else RecoveryInterventionConfigV4()
    )
    gate = gate if gate is not None else DecisionGateConfigV4()
    preregistration = (
        _bind_preregistration(preregistration_path)
        if preregistration_path is not None
        else None
    )
    source = {
        "model_path": str(model_path),
        "model_sha256": base._sha256(model_path),
    }
    run_dir.mkdir(parents=True)
    started = time.perf_counter()
    base._write_atomic(
        run_dir / "config.json",
        {
            "schema": (
                "zuma-rl.alphazuma-55-recovery-intervention-v4-probe-config"
            ),
            "version": 1,
            "status": "FROZEN",
            "probe": {
                "path": str(SCRIPT_PATH),
                "sha256": base._sha256(SCRIPT_PATH),
            },
            "controller": {
                "path": str(CONTROLLER_PATH),
                "sha256": base._sha256(CONTROLLER_PATH),
            },
            "metrics_module": {
                "path": str(METRICS_PATH),
                "sha256": base._sha256(METRICS_PATH),
            },
            "preregistration": preregistration,
            "source": source,
            "teacher_id": TEACHER_ID,
            "levels": list(levels),
            "seed_base": int(seed_base),
            "seed_last": int(seed_base) + len(levels) - 1,
            "max_ticks": int(max_ticks),
            "device": str(device),
            "controller_config": controller_config.contract(),
            "decision_gate": gate.contract(),
        },
    )
    model: Any | None = None
    try:
        import torch
        from sb3_contrib import MaskablePPO

        torch.set_num_threads(1)
        model = MaskablePPO.load(str(model_path), device=str(device))
        modes = []
        for mode in EXPECTED_MODES:
            base._write_atomic(
                run_dir / "status.json",
                {
                    "schema": (
                        "zuma-rl.alphazuma-55-recovery-intervention"
                        "-v4-probe-status"
                    ),
                    "version": 1,
                    "status": "RUNNING",
                    "stage": mode,
                    "updated_utc": base._utc_now(),
                    "completed_modes": [row["mode"] for row in modes],
                    "wall_seconds": time.perf_counter() - started,
                },
            )
            vector = _build_vector(
                original_root=original_root,
                level_ids=levels,
                max_ticks=max_ticks,
            )
            try:
                teachers = _build_teachers(vector)
                result = _run_mode(
                    mode=mode,
                    model=model,
                    vector=vector,
                    teachers=teachers,
                    level_ids=levels,
                    seed_base=int(seed_base),
                    max_ticks=int(max_ticks),
                    controller_config=controller_config,
                )
            finally:
                vector.close()
            modes.append(result)
            base._write_atomic(run_dir / f"{mode}.json", _json_safe(result))
        decision = _decision(gate=gate, level_ids=levels, modes=modes)
        runtime = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": str(device),
        }
        if str(device).startswith("cuda"):
            runtime["cuda_device"] = torch.cuda.get_device_name(
                torch.device(str(device))
            )
        completion = _build_completion(
            modes=modes,
            decision=decision,
            source=source,
            runtime=runtime,
            wall_seconds=time.perf_counter() - started,
            controller_config=controller_config,
            gate=gate,
            preregistration=preregistration,
        )
        base._write_atomic(run_dir / "completion.json", completion)
        return completion
    except BaseException as error:
        base._write_atomic(
            run_dir / "failure.json",
            {
                "schema": (
                    "zuma-rl.alphazuma-55-recovery-intervention"
                    "-v4-probe-failure"
                ),
                "version": 1,
                "status": "FAILED",
                "failed_utc": base._utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
                "formal_seed_consumption": False,
                "training_authority": False,
            },
        )
        raise
    finally:
        del model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--original-root", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--teacher-id",
        default=TEACHER_ID,
        help=(
            "identifier of the deterministic teacher implementation; the "
            "settled strategic teacher has no checkpoint file"
        ),
    )
    parser.add_argument(
        "--levels", nargs="+", default=list(DEFAULT_LEVELS)
    )
    parser.add_argument("--seed-base", type=int, default=DEFAULT_SEED_BASE)
    parser.add_argument("--max-ticks", type=int, default=DEFAULT_MAX_TICKS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--preregistration", type=Path, default=None)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if str(args.teacher_id) != TEACHER_ID:
        raise ValueError(
            f"only the {TEACHER_ID} teacher implementation exists"
        )
    model_path = args.model_path.expanduser().resolve(strict=True)
    original_root = args.original_root.expanduser().resolve(strict=True)
    levels = _validate_levels(args.levels)
    preregistration = (
        _bind_preregistration(args.preregistration.expanduser())
        if args.preregistration is not None
        else None
    )
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "VALID",
                    "model": {
                        "path": str(model_path),
                        "sha256": base._sha256(model_path),
                    },
                    "preregistration": preregistration,
                    "levels": len(levels),
                    "modes": list(EXPECTED_MODES),
                    "seed_base": int(args.seed_base),
                    "formal_seed_consumption": False,
                    "training_authority": False,
                },
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        )
        return 0
    result = run(
        model_path=model_path,
        original_root=original_root,
        run_dir=args.run_dir.expanduser(),
        level_ids=levels,
        seed_base=int(args.seed_base),
        max_ticks=int(args.max_ticks),
        device=str(args.device),
        preregistration_path=args.preregistration,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "decision": result["decision"]["status"],
                "formal_seed_consumption": False,
                "training_authority": False,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
