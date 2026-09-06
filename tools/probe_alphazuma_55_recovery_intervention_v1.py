"""Run a paired student-only versus targeted-recovery engineering probe."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import numpy as np

from tools import distill_alphazuma_55 as legacy
from tools import distill_alphazuma_55_motor_observable_replay_v2 as motor
from tools import distill_alphazuma_55_polar_dagger_v1 as dagger
from tools import distill_alphazuma_55_polar_intent_wide_v1 as intent
from tools.alphazuma_55_recovery_intervention_v1 import (
    RecoveryInterventionConfig,
    RecoveryInterventionController,
    semantic_action_disagreement,
)
from zuma_rl.alphazuma_55 import INCLUDED_LEVELS


SCRIPT_PATH = Path(__file__).resolve()
CONTROLLER_PATH = (
    SCRIPT_PATH.parent / "alphazuma_55_recovery_intervention_v1.py"
).resolve()
EXPECTED_MODES = ("student_only", "recovery_intervention")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _write_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _bound(reference: Any, expected: Path, label: str) -> Path:
    if not isinstance(reference, dict):
        raise ValueError(f"{label} reference must be an object")
    path = Path(str(reference.get("path", ""))).resolve(strict=True)
    if path != expected.resolve(strict=True):
        raise ValueError(f"{label} path changed")
    if reference.get("sha256") != _sha256(path):
        raise ValueError(f"{label} bytes changed")
    return path


def _validate_preregistration(path: Path) -> dict[str, Any]:
    prereg = _read(path)
    if not (
        prereg.get("schema")
        == "zuma-rl.alphazuma-55-recovery-intervention-probe-preregistration"
        and prereg.get("version") == 1
        and prereg.get("status") == "FROZEN_BEFORE_PROBE"
    ):
        raise ValueError("unexpected recovery intervention preregistration")
    _bound(prereg.get("probe"), SCRIPT_PATH, "probe")
    _bound(prereg.get("controller"), CONTROLLER_PATH, "controller")

    master = prereg.get("master_preregistration", {})
    if not isinstance(master, dict):
        raise ValueError("master reference must be an object")
    master_path = Path(str(master.get("path", ""))).resolve(strict=True)
    if master.get("sha256") != _sha256(master_path):
        raise ValueError("master preregistration bytes changed")
    master_value = _read(master_path)
    if master_value.get("schema") != (
        "zuma-rl.alphazuma-55-weekend-master-preregistration"
    ):
        raise ValueError("unexpected AlphaZuma 55 master")

    source = prereg.get("source", {})
    if not isinstance(source, dict):
        raise ValueError("source must be an object")
    model_path = Path(str(source.get("model_path", ""))).resolve(strict=True)
    completion_path = Path(
        str(source.get("completion_path", ""))
    ).resolve(strict=True)
    if source.get("model_sha256") != _sha256(model_path):
        raise ValueError("source model bytes changed")
    if source.get("completion_sha256") != _sha256(completion_path):
        raise ValueError("source completion bytes changed")
    completion = _read(completion_path)
    final = completion.get("final_model", {})
    if not (
        completion.get("status") == "COMPLETE"
        and completion.get("route_version") == "v3_retry_audited"
        and completion.get("formal_seed_consumption") is False
        and Path(str(final.get("path", ""))).resolve() == model_path
        and final.get("sha256") == source.get("model_sha256")
    ):
        raise ValueError("source is not the frozen motor-observable V3 model")

    levels = tuple(str(value) for value in prereg.get("levels", ()))
    if not levels or len(levels) != len(set(levels)):
        raise ValueError("probe levels must be non-empty and unique")
    canonical = {value.casefold(): value for value in INCLUDED_LEVELS}
    if any(level.casefold() not in canonical for level in levels):
        raise ValueError("probe level escaped the included full55 inventory")

    probe = prereg.get("paired_probe", {})
    if not isinstance(probe, dict):
        raise ValueError("paired_probe must be an object")
    modes = tuple(str(value) for value in probe.get("modes", ()))
    if modes != EXPECTED_MODES:
        raise ValueError("paired probe modes changed")
    seed_base = int(probe.get("seed_base", -1))
    seed_last = int(probe.get("seed_last", -1))
    if seed_last != seed_base + len(levels) - 1:
        raise ValueError("paired probe seed interval changed")
    registry = master_value["seed_registry"]["engineering_and_calibration"]
    if not int(registry["first"]) <= seed_base <= seed_last <= int(
        registry["last"]
    ):
        raise ValueError("probe seeds escaped engineering_and_calibration")
    if not (
        int(probe.get("parallel_envs", -1)) == len(levels)
        and int(probe.get("max_ticks", -1)) == 30000
        and probe.get("paired_modes_share_identical_task_seeds") is True
        and probe.get("deterministic_student_inference") is True
    ):
        raise ValueError("paired probe execution contract changed")
    frozen_controller = RecoveryInterventionConfig(
        **dict(probe.get("controller_config", {}))
    )
    if frozen_controller.contract() != probe.get("controller_config"):
        raise ValueError("controller configuration is not canonical")

    acceptance = prereg.get("acceptance", {})
    required_acceptance = {
        "intervention_wins_must_exceed_student_only": True,
        "intervention_total_score_must_exceed_student_only": True,
        "minimum_paired_score_improvements": 6,
        "minimum_interventions": 6,
        "minimum_recovery_successes": 1,
        "maximum_teacher_execution_fraction": 0.6,
    }
    if acceptance != required_acceptance:
        raise ValueError("probe acceptance contract changed")
    boundary = prereg.get("authority_boundary", {})
    for name in (
        "training_authority",
        "candidate_authority",
        "formal_selection_seed_consumption",
        "formal_final_blind_seed_consumption",
        "continuous_campaign_seed_consumption",
        "power_change_authority",
    ):
        if boundary.get(name) is not False:
            raise ValueError(f"probe authority changed: {name}")
    return prereg


def _input_config() -> Any:
    return dagger.EliteHumanInputConfig(
        profile_id="elite-human-v1",
        reaction_delay_ticks=12,
        max_aim_speed_degrees_per_second=1080.0,
        max_aim_acceleration_degrees_per_second_squared=18000.0,
        min_button_interval_ticks=5,
    )


def _reward_config() -> Any:
    return dagger.WinFirstRewardConfig(
        profile_id="win-time-score-v1",
        win_reward=10.0,
        failure_reward=-10.0,
        time_penalty_per_native_tick=-0.0001,
        score_progress_reward_cap=0.01,
    )


def _run_mode(
    *,
    mode: str,
    model: Any,
    original_root: Path,
    level_ids: Sequence[str],
    seed_base: int,
    max_ticks: int,
    controller_config: RecoveryInterventionConfig,
) -> dict[str, Any]:
    from sb3_contrib.common.maskable.utils import get_action_masks
    from stable_baselines3.common.vec_env import SubprocVecEnv

    if mode not in EXPECTED_MODES:
        raise ValueError(f"unexpected probe mode: {mode}")
    factories = [
        motor._make_motor_env_factory(
            original_root=original_root,
            level_id=level_id,
            max_ticks=max_ticks,
            input_config=_input_config(),
            reward_config=_reward_config(),
        )
        for level_id in level_ids
    ]
    vector = SubprocVecEnv(factories, start_method="forkserver")
    started = time.perf_counter()
    try:
        vector.seed(seed_base)
        observations = vector.reset()
        specifications = vector.env_method("teacher_spec")
        teachers = [
            dagger.CurveAwareSettledStrategicRevengeTeacher(
                legacy._teacher_spec(value)
            )
            for value in specifications
        ]
        controllers = [
            RecoveryInterventionController(controller_config)
            for _ in level_ids
        ]
        previous_infos: list[dict[str, Any]] = [
            {"score": 0, "chain_length": 0} for _ in level_ids
        ]
        finished = np.zeros(len(level_ids), dtype=np.bool_)
        steps = np.zeros(len(level_ids), dtype=np.int64)
        teacher_steps = np.zeros(len(level_ids), dtype=np.int64)
        student_steps = np.zeros(len(level_ids), dtype=np.int64)
        disagreement_steps = np.zeros(len(level_ids), dtype=np.int64)
        phase_counts = [Counter() for _ in level_ids]
        disagreement_reasons = [Counter() for _ in level_ids]
        last_infos: list[dict[str, Any]] = [{} for _ in level_ids]
        transitions = 0

        for _ in range(max_ticks + 1):
            masks = np.asarray(get_action_masks(vector), dtype=np.bool_)
            predicted, _ = model.predict(
                observations,
                deterministic=True,
                action_masks=masks,
            )
            student_actions = np.asarray(predicted, dtype=np.int64).reshape(
                len(level_ids), 2
            )
            teacher_actions = np.zeros((len(level_ids), 2), dtype=np.int64)
            actions = student_actions.copy()
            active = ~finished
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
                teacher_actions[index] = effective
                disagreement, reason = semantic_action_disagreement(
                    effective,
                    student_actions[index],
                    fire_aim_threshold=(
                        controller_config.fire_aim_disagreement_bins
                    ),
                )
                if disagreement:
                    disagreement_steps[index] += 1
                    disagreement_reasons[index][str(reason)] += 1
                if mode == "recovery_intervention":
                    decision = controllers[index].decide(
                        tick=int(steps[index]),
                        teacher_action=effective,
                        student_action=student_actions[index],
                        previous_info=previous_infos[index],
                    )
                    phase_counts[index][decision.phase] += 1
                    if decision.execute_teacher:
                        actions[index] = effective
                        teacher_steps[index] += 1
                    else:
                        student_steps[index] += 1
                else:
                    phase = (
                        "student_disagreement"
                        if disagreement
                        else "student_nominal"
                    )
                    phase_counts[index][phase] += 1
                    student_steps[index] += 1

            observations, _, dones, infos = vector.step(actions)
            transitions += int(np.count_nonzero(active))
            steps[active] += 1
            for position, (done, info) in enumerate(
                zip(dones, infos, strict=True)
            ):
                if finished[position]:
                    continue
                previous_infos[position] = dict(info)
                last_infos[position] = dict(info)
                if bool(done):
                    finished[position] = True
            if bool(np.all(finished)):
                break
        if not bool(np.all(finished)):
            missing = [
                level_ids[index]
                for index in range(len(level_ids))
                if not finished[index]
            ]
            raise RuntimeError(f"probe episodes exceeded guard: {missing}")

        episodes: list[dict[str, Any]] = []
        for index, level_id in enumerate(level_ids):
            info = last_infos[index]
            total = int(teacher_steps[index] + student_steps[index])
            controller_summary = controllers[index].summary()
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
                    "teacher_execution_fraction": (
                        float(teacher_steps[index]) / total if total else 0.0
                    ),
                    "semantic_disagreement_steps": int(
                        disagreement_steps[index]
                    ),
                    "semantic_disagreement_fraction": (
                        float(disagreement_steps[index]) / total
                        if total
                        else 0.0
                    ),
                    "disagreement_reasons": dict(
                        disagreement_reasons[index]
                    ),
                    "phase_counts": dict(phase_counts[index]),
                    "intervention": (
                        controller_summary
                        if mode == "recovery_intervention"
                        else None
                    ),
                    "observation_capacity_overflow": bool(
                        info.get("observation_capacity_overflow", False)
                    ),
                }
            )
        total_decisions = sum(row["decisions"] for row in episodes)
        total_teacher = sum(
            row["teacher_execution_steps"] for row in episodes
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
                "teacher_execution_fraction": (
                    total_teacher / total_decisions if total_decisions else 0.0
                ),
                "semantic_disagreement_steps": sum(
                    row["semantic_disagreement_steps"] for row in episodes
                ),
                "interventions": sum(
                    int((row["intervention"] or {}).get("interventions", 0))
                    for row in episodes
                ),
                "recovery_successes": sum(
                    int(
                        (row["intervention"] or {}).get(
                            "recovery_successes", 0
                        )
                    )
                    for row in episodes
                ),
                "forced_handoffs": sum(
                    int((row["intervention"] or {}).get("forced_handoffs", 0))
                    for row in episodes
                ),
            },
            "runtime": {
                "wall_seconds": time.perf_counter() - started,
                "active_vector_transitions": transitions,
                "parallel_envs": len(level_ids),
            },
        }
    finally:
        vector.close()


def _decision(prereg: dict[str, Any], modes: Sequence[dict[str, Any]]) -> dict[str, Any]:
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
    for level_id in prereg["levels"]:
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
    acceptance = prereg["acceptance"]
    checks = {
        "intervention_wins_exceed_student_only": (
            recovery["summary"]["wins"] > baseline["summary"]["wins"]
        ),
        "intervention_total_score_exceeds_student_only": (
            recovery["summary"]["total_score"]
            > baseline["summary"]["total_score"]
        ),
        "paired_score_improvements": (
            sum(row["score_delta"] > 0 for row in paired_rows)
            >= int(acceptance["minimum_paired_score_improvements"])
        ),
        "minimum_interventions": (
            recovery["summary"]["interventions"]
            >= int(acceptance["minimum_interventions"])
        ),
        "minimum_recovery_successes": (
            recovery["summary"]["recovery_successes"]
            >= int(acceptance["minimum_recovery_successes"])
        ),
        "maximum_teacher_execution_fraction": (
            recovery["summary"]["teacher_execution_fraction"]
            <= float(acceptance["maximum_teacher_execution_fraction"])
        ),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "paired_rows": paired_rows,
        "training_authorized_by_this_probe": False,
        "next_step_if_pass": (
            "freeze a separate recovery-priority DAgger training pilot"
        ),
    }


def run(*, preregistration_path: Path, original_root: Path) -> dict[str, Any]:
    preregistration_path = preregistration_path.resolve(strict=True)
    original_root = original_root.resolve(strict=True)
    prereg = _validate_preregistration(preregistration_path)
    run_dir = Path(str(prereg["run_dir"])).resolve()
    if run_dir.exists():
        raise FileExistsError(f"probe output already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    started = time.perf_counter()
    _write_atomic(
        run_dir / "config.json",
        {
            "schema": "zuma-rl.alphazuma-55-recovery-intervention-probe-config",
            "version": 1,
            "status": "FROZEN",
            "preregistration": {
                "path": str(preregistration_path),
                "sha256": _sha256(preregistration_path),
            },
            "probe": {"path": str(SCRIPT_PATH), "sha256": _sha256(SCRIPT_PATH)},
            "controller": {
                "path": str(CONTROLLER_PATH),
                "sha256": _sha256(CONTROLLER_PATH),
            },
            "paired_probe": prereg["paired_probe"],
        },
    )
    model: Any | None = None
    try:
        import torch
        from sb3_contrib import MaskablePPO

        torch.set_num_threads(1)
        model = MaskablePPO.load(
            prereg["source"]["model_path"],
            device=str(prereg["paired_probe"]["device"]),
        )
        controller_config = RecoveryInterventionConfig(
            **dict(prereg["paired_probe"]["controller_config"])
        )
        modes = []
        for mode in EXPECTED_MODES:
            _write_atomic(
                run_dir / "status.json",
                {
                    "schema": (
                        "zuma-rl.alphazuma-55-recovery-intervention-probe-status"
                    ),
                    "version": 1,
                    "status": "RUNNING",
                    "stage": mode,
                    "updated_utc": _utc_now(),
                    "completed_modes": [row["mode"] for row in modes],
                    "wall_seconds": time.perf_counter() - started,
                },
            )
            result = _run_mode(
                mode=mode,
                model=model,
                original_root=original_root,
                level_ids=prereg["levels"],
                seed_base=int(prereg["paired_probe"]["seed_base"]),
                max_ticks=int(prereg["paired_probe"]["max_ticks"]),
                controller_config=controller_config,
            )
            modes.append(result)
            _write_atomic(run_dir / f"{mode}.json", result)
        decision = _decision(prereg, modes)
        completion = {
            "schema": "zuma-rl.alphazuma-55-recovery-intervention-probe-completion",
            "version": 1,
            "status": "COMPLETE",
            "completed_utc": _utc_now(),
            "wall_seconds": time.perf_counter() - started,
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "torch": torch.__version__,
                "device": str(prereg["paired_probe"]["device"]),
                "cuda_device": torch.cuda.get_device_name(
                    torch.device(str(prereg["paired_probe"]["device"]))
                ),
            },
            "source": prereg["source"],
            "modes": modes,
            "decision": decision,
            "formal_seed_consumption": False,
            "formal_candidate_authority": False,
            "training_authority": False,
        }
        _write_atomic(run_dir / "completion.json", completion)
        return completion
    except BaseException as error:
        _write_atomic(
            run_dir / "failure.json",
            {
                "schema": "zuma-rl.alphazuma-55-recovery-intervention-probe-failure",
                "version": 1,
                "status": "FAILED",
                "failed_utc": _utc_now(),
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
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--original-root", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prereg_path = args.preregistration.expanduser().resolve(strict=True)
    original_root = args.original_root.expanduser().resolve(strict=True)
    prereg = _validate_preregistration(prereg_path)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "VALID",
                    "preregistration": {
                        "path": str(prereg_path),
                        "sha256": _sha256(prereg_path),
                    },
                    "levels": len(prereg["levels"]),
                    "modes": list(EXPECTED_MODES),
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
        preregistration_path=prereg_path,
        original_root=original_root,
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
