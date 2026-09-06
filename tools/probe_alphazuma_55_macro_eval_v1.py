"""Evaluate existing per-tick BC students through park-settle macros.

Finding (2026-08-16): per-tick BC students pick good aim targets when
they sit on the teacher's cursor trajectory (aim-intent p50 = 2 bins in
teacher-driven states) but their self-driven per-tick aim stream
diverges (p50 = 37 bins, shots-on-target 4-9%, zero wins).  The per-tick
cursor feedback loop -- not target selection -- is the wall.

This probe removes that loop from the measurement.  The environment is
the exact probe-v4 motor-observable stack wrapped in
src/zuma_rl/park_settle_action_wrapper.py: the policy is queried only
at macro decision points (gun NORMAL, no pending button intention) and
the wrapper itself executes the teacher's motor protocol -- park the
delayed slew-limited cursor on the target, gate the fire edge on the
executed cursor settling within 2 bins, hold the target through the
reaction delay plus the firing animation so the release cannot be
corrupted by a later intent.

Adapter contract (per macro decision point):

* masked deterministic predict on the CURRENT observation; the per-tick
  (4 + 180) mask is derived from the wrapper's macro (3 + 180) mask --
  the three macro verbs are the actuator's first three per-tick verbs
  in order, and hop (per-tick verb 3) has no macro so it is masked off;
* verb mapping wait -> wait_hold, fire -> fire, swap -> swap; the aim
  target is always the model's own aim-head output;
* a wait becomes wait_hold parking the model's aim bin for hold_ticks
  native ticks, after which the model is re-queried at the next
  decision point;
* a macro-mask-illegal verb falls back to wait_hold (counted per
  episode under mask_fallbacks, never hidden).

student_only mode ONLY: no teacher, no recovery controller -- this is a
pure autonomy evaluation of existing per-tick checkpoints through the
macro motor program.  shots_on_target here measures the WRAPPER's
execution fidelity (release within 2 bins of the commissioning macro's
target, expected near 1.0), NOT strategic quality; the real outcome
signals are wins, score, and shot count.

Existing modules are imported, never modified: the macro wrapper, the
probe-v4 helpers/panel/seed base, and the motor env factory keep their
bytes.  This probe consumes engineering seeds only and carries no
training or formal-selection authority.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
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

from tools import distill_alphazuma_55_motor_observable_replay_v2 as motor
from tools import probe_alphazuma_55_recovery_intervention_v1 as base
from tools import probe_alphazuma_55_recovery_intervention_v4 as v4
from zuma_rl.park_settle_action_wrapper import (
    MACRO_FIRE,
    MACRO_SWAP,
    MACRO_VERB_NAMES,
    MACRO_WAIT_HOLD,
    ParkSettleActionConfig,
    ParkSettleActionWrapper,
)


SCRIPT_PATH = Path(__file__).resolve()
WRAPPER_PATH = (
    SCRIPT_PATH.parents[1]
    / "src"
    / "zuma_rl"
    / "park_settle_action_wrapper.py"
).resolve()
COMPLETION_SCHEMA = "zuma-rl.alphazuma-55-macro-eval-v1"
MODE = "student_only"
AIM_BINS = 180
# full55-v1 per-tick interface: MultiDiscrete([4, 180]) = wait/fire/swap/hop.
PER_TICK_VERB_COUNT = 4
PER_TICK_TO_MACRO_VERB = {
    0: MACRO_WAIT_HOLD,  # wait -> wait_hold parking the model's aim bin
    1: MACRO_FIRE,
    2: MACRO_SWAP,
}
FALLBACK_UNSUPPORTED = "per_tick_verb_outside_macro_interface"
FALLBACK_MASKED = "macro_verb_masked"
# Mirrors ParkSettleActionConfig.settle_tolerance_bins: on-target means
# the release landed within the wrapper's own settle gate.
SHOT_TOLERANCE_BINS = 2
DEFAULT_LEVELS = v4.DEFAULT_LEVELS
DEFAULT_SEED_BASE = v4.DEFAULT_SEED_BASE
DEFAULT_MAX_TICKS = v4.DEFAULT_MAX_TICKS
DEFAULT_HOLD_TICKS = 8


def _bin_distance(first: int, second: int) -> int:
    direct = abs(int(first) - int(second)) % AIM_BINS
    return min(direct, AIM_BINS - direct)


def per_tick_masks_from_macro(macro_masks: Any) -> np.ndarray:
    """Derive the per-tick model's (4 + 180) mask from the macro mask.

    The wrapper's action_masks() truncates the actuator's per-tick verb
    mask to the three macro verbs, so the first three per-tick verbs
    recover exactly; hop (per-tick verb 3) has no macro and is masked
    off so the masked deterministic predict can never select it.  Aim
    bins pass through unchanged (always all-valid in both interfaces).
    """

    masks = np.asarray(macro_masks, dtype=np.bool_)
    squeeze = masks.ndim == 1
    if squeeze:
        masks = masks[np.newaxis, :]
    expected = len(MACRO_VERB_NAMES) + AIM_BINS
    if masks.ndim != 2 or masks.shape[1] != expected:
        raise ValueError(
            f"macro mask width must be {expected}, got {masks.shape}"
        )
    verbs = np.zeros((masks.shape[0], PER_TICK_VERB_COUNT), dtype=np.bool_)
    verbs[:, : len(MACRO_VERB_NAMES)] = masks[:, : len(MACRO_VERB_NAMES)]
    result = np.concatenate(
        (verbs, masks[:, len(MACRO_VERB_NAMES) :]), axis=1
    )
    return result[0] if squeeze else result


def adapt_per_tick_action(
    per_tick_action: Any, macro_mask: Any
) -> tuple[np.ndarray, str | None]:
    """Map one per-tick (verb, aim) prediction to a macro action.

    Returns ``(macro_action, fallback_reason)``: the fallback reason is
    None on a direct mapping, FALLBACK_UNSUPPORTED when the per-tick
    verb has no macro (hop), and FALLBACK_MASKED when the mapped macro
    verb is invalid under the wrapper's decision-point mask.  Both
    fallbacks execute wait_hold with the model's aim bin as the parked
    target, so the model is simply re-queried at the next decision
    point with its cursor already heading toward its stated intent.
    """

    values = np.asarray(per_tick_action, dtype=np.int64).reshape(-1)
    if values.shape != (2,):
        raise ValueError(f"invalid per-tick action: {per_tick_action!r}")
    verb, aim = int(values[0]), int(values[1])
    if not 0 <= verb < PER_TICK_VERB_COUNT:
        raise ValueError(f"per-tick verb out of range: {verb}")
    if not 0 <= aim < AIM_BINS:
        raise ValueError(f"per-tick aim bin out of range: {aim}")
    mask = np.asarray(macro_mask, dtype=np.bool_).reshape(-1)
    if mask.shape != (len(MACRO_VERB_NAMES) + AIM_BINS,):
        raise ValueError(f"invalid macro mask shape: {mask.shape}")
    if not bool(mask[MACRO_WAIT_HOLD]):
        raise RuntimeError(
            "wait_hold is masked at a decision point; the actuator "
            "contract guarantees wait is always available"
        )
    macro_verb = PER_TICK_TO_MACRO_VERB.get(verb)
    fallback: str | None = None
    if macro_verb is None:
        fallback = FALLBACK_UNSUPPORTED
        macro_verb = MACRO_WAIT_HOLD
    elif not bool(mask[macro_verb]):
        fallback = FALLBACK_MASKED
        macro_verb = MACRO_WAIT_HOLD
    return np.asarray((macro_verb, aim), dtype=np.int64), fallback


def _default_masks_provider(vector: Any) -> np.ndarray:
    from sb3_contrib.common.maskable.utils import get_action_masks

    return np.asarray(get_action_masks(vector), dtype=np.bool_)


def _run_student_only(
    *,
    model: Any,
    vector: Any,
    level_ids: Sequence[str],
    seed_base: int,
    max_ticks: int,
    masks_provider: Callable[[Any], np.ndarray] | None = None,
) -> dict[str, Any]:
    """Run the macro-adapted student over one vectorized panel.

    The model, vector, and mask provider are injectable so the adapter
    loop is testable on CPU with stubs; the real run passes a
    SubprocVecEnv over macro-wrapped motor stacks and a MaskablePPO
    per-tick student.  One vector.step is one macro per environment;
    every telemetry row comes from the wrapper's own info["park_settle"]
    release reports.
    """

    if masks_provider is None:
        masks_provider = _default_masks_provider
    count = len(level_ids)
    started = time.perf_counter()
    vector.seed(int(seed_base))
    observations = vector.reset()
    finished = np.zeros(count, dtype=np.bool_)
    macro_decisions = np.zeros(count, dtype=np.int64)
    native_ticks = np.zeros(count, dtype=np.int64)
    shots = np.zeros(count, dtype=np.int64)
    on_target = np.zeros(count, dtype=np.int64)
    unsettled_fire = np.zeros(count, dtype=np.int64)
    verb_counts: list[Counter] = [Counter() for _ in level_ids]
    fallback_counts: list[Counter] = [Counter() for _ in level_ids]
    release_errors: list[list[int]] = [[] for _ in level_ids]
    last_infos: list[dict[str, Any]] = [{} for _ in level_ids]
    macro_transitions = 0

    for _ in range(int(max_ticks) + 1):
        if bool(np.all(finished)):
            break
        macro_masks = np.asarray(
            masks_provider(vector), dtype=np.bool_
        ).reshape(count, -1)
        per_tick_masks = per_tick_masks_from_macro(macro_masks)
        predicted, _ = model.predict(
            observations,
            deterministic=True,
            action_masks=per_tick_masks,
        )
        per_tick_actions = np.asarray(predicted, dtype=np.int64).reshape(
            count, 2
        )
        # Finished environments (auto-reset by the vector) get an inert
        # wait_hold; their telemetry is never read again.
        actions = np.zeros((count, 2), dtype=np.int64)
        active = ~finished
        for position in np.flatnonzero(active):
            index = int(position)
            macro_action, fallback = adapt_per_tick_action(
                per_tick_actions[index], macro_masks[index]
            )
            actions[index] = macro_action
            if fallback is not None:
                fallback_counts[index][fallback] += 1
        observations, _, dones, infos = vector.step(actions)
        macro_transitions += int(np.count_nonzero(active))
        for position in np.flatnonzero(active):
            index = int(position)
            info = dict(infos[index])
            macro = info.get("park_settle")
            if not isinstance(macro, Mapping):
                raise RuntimeError(
                    "macro step returned no park_settle telemetry; the "
                    "vector is not a park-settle wrapped stack"
                )
            macro_decisions[index] += 1
            native_ticks[index] += int(macro["ticks_consumed"])
            verb_counts[index][str(macro["macro_verb"])] += 1
            if macro["macro_verb"] == "fire" and not bool(macro["settled"]):
                unsettled_fire[index] += 1
            if bool(macro.get("released", False)):
                shots[index] += 1
                error = _bin_distance(
                    int(macro["release_aim_bin"]),
                    int(macro["target_aim_bin"]),
                )
                release_errors[index].append(error)
                if error <= SHOT_TOLERANCE_BINS:
                    on_target[index] += 1
            last_infos[index] = info
            if bool(dones[index]):
                finished[index] = True

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
        errors = release_errors[index]
        episodes.append(
            {
                "level_id": str(level_id),
                "seed": int(seed_base) + index,
                "outcome": info.get("outcome"),
                "native_outcome": info.get("native_outcome"),
                "time_limit_truncated": bool(
                    info.get("TimeLimit.truncated", False)
                ),
                "ticks": int(info.get("ticks", native_ticks[index])),
                "score": int(info.get("score", 0)),
                "macro_decisions": int(macro_decisions[index]),
                "native_ticks_consumed": int(native_ticks[index]),
                "macro_verb_counts": dict(verb_counts[index]),
                "mask_fallbacks": dict(fallback_counts[index]),
                "shots": int(shots[index]),
                "shots_on_target": int(on_target[index]),
                "shots_on_target_rate": v4._fraction(
                    int(on_target[index]), int(shots[index])
                ),
                "mean_release_error_bins": (
                    float(np.mean(errors)) if errors else float("nan")
                ),
                "unsettled_fire_macros": int(unsettled_fire[index]),
                "observation_capacity_overflow": bool(
                    info.get("observation_capacity_overflow", False)
                ),
            }
        )

    all_errors = [error for row in release_errors for error in row]
    merged_verbs: Counter = Counter()
    for row in verb_counts:
        merged_verbs.update(row)
    merged_fallbacks: Counter = Counter()
    for row in fallback_counts:
        merged_fallbacks.update(row)
    total_shots = int(shots.sum())
    summary = {
        "attempts": len(episodes),
        "wins": sum(row["outcome"] == "win" for row in episodes),
        "losses": sum(row["outcome"] == "loss" for row in episodes),
        "truncations": sum(row["time_limit_truncated"] for row in episodes),
        "total_score": sum(row["score"] for row in episodes),
        "total_shots": total_shots,
        "shots_on_target": int(on_target.sum()),
        "shots_on_target_rate": v4._fraction(
            int(on_target.sum()), total_shots
        ),
        "mean_release_error_bins": (
            float(np.mean(all_errors)) if all_errors else float("nan")
        ),
        "unsettled_fire_macros": int(unsettled_fire.sum()),
        "mean_ticks": float(np.mean([row["ticks"] for row in episodes])),
        "total_macro_decisions": int(macro_decisions.sum()),
        "total_native_ticks": int(native_ticks.sum()),
        "mean_native_ticks_per_macro": v4._fraction(
            int(native_ticks.sum()), int(macro_decisions.sum())
        ),
        "macro_verb_counts": dict(merged_verbs),
        "mask_fallbacks": dict(merged_fallbacks),
    }
    return {
        "mode": MODE,
        "episodes": episodes,
        "summary": summary,
        "runtime": {
            "wall_seconds": time.perf_counter() - started,
            "active_macro_transitions": macro_transitions,
            "parallel_envs": count,
        },
    }


def _adapter_contract(hold_ticks: int) -> dict[str, Any]:
    return {
        "per_tick_interface": (
            "MultiDiscrete([4, 180]) masked deterministic predict on the "
            "current decision-point observation"
        ),
        "verb_mapping": {
            "wait": "wait_hold",
            "fire": "fire",
            "swap": "swap",
            "hop": "wait_hold (fallback: no macro exposes hop)",
        },
        "aim_target": "the per-tick model's own aim-head output",
        "wait_semantics": (
            "wait_hold parks the model's aim bin for "
            f"hold_ticks={int(hold_ticks)} native ticks; the model is "
            "re-queried at the next decision point"
        ),
        "mask_rule": (
            "per-tick masks are derived from the wrapper's macro masks "
            "(hop always masked off); macro-mask-illegal verbs fall back "
            "to wait_hold and are counted under mask_fallbacks"
        ),
        "on_target_semantics": (
            "release within "
            f"{SHOT_TOLERANCE_BINS} bins of the commissioning macro's "
            "target: this measures the wrapper's motor-protocol execution "
            "fidelity (expected near 1.0), not strategic quality; wins, "
            "score, and shot count are the outcome signals"
        ),
    }


def _build_completion(
    *,
    result: Mapping[str, Any],
    source: Mapping[str, Any],
    wrapper_contract: Mapping[str, Any],
    runtime: Mapping[str, Any],
    wall_seconds: float,
    levels: Sequence[str],
    seed_base: int,
    max_ticks: int,
    hold_ticks: int,
) -> dict[str, Any]:
    """Assemble the completion receipt; NaN metrics serialize as null."""

    return v4._json_safe(
        {
            "schema": COMPLETION_SCHEMA,
            "version": 1,
            "status": "COMPLETE",
            "completed_utc": base._utc_now(),
            "wall_seconds": float(wall_seconds),
            "runtime": dict(runtime),
            "mode": MODE,
            "source": dict(source),
            "adapter": _adapter_contract(hold_ticks),
            "wrapper_contract": dict(wrapper_contract),
            "levels": list(levels),
            "seed_base": int(seed_base),
            "seed_last": int(seed_base) + len(levels) - 1,
            "max_ticks": int(max_ticks),
            "hold_ticks": int(hold_ticks),
            "episodes": list(result["episodes"]),
            "summary": dict(result["summary"]),
            "mode_runtime": dict(result["runtime"]),
            "formal_seed_consumption": False,
            "formal_candidate_authority": False,
            "training_authority": False,
        }
    )


def _make_macro_env_factory(
    *,
    original_root: Path,
    level_id: str,
    max_ticks: int,
    hold_ticks: int,
) -> Any:
    """Wrap the exact probe-v4 motor stack in park-settle macros."""

    motor_factory = motor._make_motor_env_factory(
        original_root=original_root,
        level_id=level_id,
        max_ticks=int(max_ticks),
        input_config=base._input_config(),
        reward_config=base._reward_config(),
    )
    config = ParkSettleActionConfig(hold_ticks=int(hold_ticks))

    def make_env() -> ParkSettleActionWrapper:
        return ParkSettleActionWrapper(motor_factory(), config=config)

    return make_env


def _build_vector(
    *,
    original_root: Path,
    level_ids: Sequence[str],
    max_ticks: int,
    hold_ticks: int,
) -> Any:
    from stable_baselines3.common.vec_env import SubprocVecEnv

    factories = [
        _make_macro_env_factory(
            original_root=original_root,
            level_id=level_id,
            max_ticks=int(max_ticks),
            hold_ticks=int(hold_ticks),
        )
        for level_id in level_ids
    ]
    return SubprocVecEnv(factories, start_method="forkserver")


def run(
    *,
    model_path: Path,
    original_root: Path,
    run_dir: Path,
    level_ids: Sequence[str] = DEFAULT_LEVELS,
    seed_base: int = DEFAULT_SEED_BASE,
    max_ticks: int = DEFAULT_MAX_TICKS,
    hold_ticks: int = DEFAULT_HOLD_TICKS,
    device: str = "cpu",
) -> dict[str, Any]:
    model_path = model_path.resolve(strict=True)
    original_root = original_root.resolve(strict=True)
    run_dir = run_dir.resolve()
    if run_dir.exists():
        raise FileExistsError(f"macro eval output already exists: {run_dir}")
    levels = v4._validate_levels(level_ids)
    source = {
        "model_path": str(model_path),
        "model_sha256": base._sha256(model_path),
    }
    run_dir.mkdir(parents=True)
    started = time.perf_counter()
    base._write_atomic(
        run_dir / "config.json",
        {
            "schema": f"{COMPLETION_SCHEMA}-config",
            "version": 1,
            "status": "FROZEN",
            "probe": {
                "path": str(SCRIPT_PATH),
                "sha256": base._sha256(SCRIPT_PATH),
            },
            "wrapper_module": {
                "path": str(WRAPPER_PATH),
                "sha256": base._sha256(WRAPPER_PATH),
            },
            "source": source,
            "mode": MODE,
            "levels": list(levels),
            "seed_base": int(seed_base),
            "seed_last": int(seed_base) + len(levels) - 1,
            "max_ticks": int(max_ticks),
            "hold_ticks": int(hold_ticks),
            "device": str(device),
            "adapter": _adapter_contract(hold_ticks),
        },
    )
    model: Any | None = None
    try:
        import torch
        from sb3_contrib import MaskablePPO

        torch.set_num_threads(1)
        model = MaskablePPO.load(str(model_path), device=str(device))
        base._write_atomic(
            run_dir / "status.json",
            {
                "schema": f"{COMPLETION_SCHEMA}-status",
                "version": 1,
                "status": "RUNNING",
                "stage": MODE,
                "updated_utc": base._utc_now(),
            },
        )
        vector = _build_vector(
            original_root=original_root,
            level_ids=levels,
            max_ticks=int(max_ticks),
            hold_ticks=int(hold_ticks),
        )
        try:
            wrapper_contract = vector.env_method("contract")[0]
            result = _run_student_only(
                model=model,
                vector=vector,
                level_ids=levels,
                seed_base=int(seed_base),
                max_ticks=int(max_ticks),
            )
        finally:
            vector.close()
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
            result=result,
            source=source,
            wrapper_contract=wrapper_contract,
            runtime=runtime,
            wall_seconds=time.perf_counter() - started,
            levels=levels,
            seed_base=int(seed_base),
            max_ticks=int(max_ticks),
            hold_ticks=int(hold_ticks),
        )
        base._write_atomic(run_dir / "completion.json", completion)
        return completion
    except BaseException as error:
        base._write_atomic(
            run_dir / "failure.json",
            {
                "schema": f"{COMPLETION_SCHEMA}-failure",
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
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--levels", nargs="+", default=list(DEFAULT_LEVELS))
    parser.add_argument("--seed-base", type=int, default=DEFAULT_SEED_BASE)
    parser.add_argument("--max-ticks", type=int, default=DEFAULT_MAX_TICKS)
    parser.add_argument(
        "--hold-ticks", type=int, default=DEFAULT_HOLD_TICKS
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(
        model_path=args.model_path.expanduser().resolve(strict=True),
        original_root=args.original_root.expanduser().resolve(strict=True),
        run_dir=args.run_dir.expanduser(),
        level_ids=tuple(args.levels),
        seed_base=int(args.seed_base),
        max_ticks=int(args.max_ticks),
        hold_ticks=int(args.hold_ticks),
        device=str(args.device),
    )
    summary = result["summary"]
    print(
        json.dumps(
            {
                "status": result["status"],
                "wins": summary["wins"],
                "losses": summary["losses"],
                "truncations": summary["truncations"],
                "total_score": summary["total_score"],
                "total_shots": summary["total_shots"],
                "shots_on_target_rate": summary["shots_on_target_rate"],
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
