"""Teacher-through-macro ceiling probe for the park-settle interface.

The macro-eval probe (tools/probe_alphazuma_55_macro_eval_v1) showed the
park-settle wrapper executes the motor protocol perfectly (100%
shots-on-target, 0.0-bin release error) while existing per-tick BC
students fail on DECISION quality at macro decision points.  Before
spending an afternoon of PPO on the macro interface, this probe measures
the interface's CEILING: the decision-maker is the 55/55 TEACHER
(CurveAwareSettledStrategicRevengeTeacher), not a student model.

At each macro decision point the per-env teacher is queried on the
current RAW frame (no observation stacking exists anywhere in this
stack, so the vector's observations ARE raw single frames -- the probe-v5
RawFrameTeacher shim is unnecessary and deliberately absent) and its
per-tick proposal is mapped through the SAME adapter the student eval
uses (tools.probe_alphazuma_55_macro_eval_v1.adapt_per_tick_action):

* wait -> wait_hold; the teacher's wait action CARRIES its parked
  target bin, so the wait_hold macro parks exactly the teacher's
  carried target;
* fire -> (fire, target); swap -> (swap, target);
* hop (per-tick verb 3) has no macro: it falls back to wait_hold and is
  counted under mask_fallbacks["per_tick_verb_outside_macro_interface"]
  -- the dual-position levels (zuma_rl.alphazuma_55.DUAL_POSITION_LEVELS)
  are therefore EXPECTED to fail through this interface;
* a macro-mask-illegal verb falls back to wait_hold, counted.

Interpretation contract: if the teacher wins most levels through the
macro interface, macro-interface PPO has measured headroom; per-level
outcomes identify exactly which levels the interface itself blocks.
Teacher wins here are TEACHER wins -- this probe carries no autonomy,
training, or formal-selection authority and consumes engineering seeds
only.

Existing modules are imported, never modified: the macro wrapper, the
macro-eval env factory/loop/adapter, the probe-v4 teacher construction
and level/seed constants all keep their bytes.
"""

from __future__ import annotations

import argparse
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

from tools import probe_alphazuma_55_macro_eval_v1 as macro_eval
from tools import probe_alphazuma_55_recovery_intervention_v1 as base
from tools import probe_alphazuma_55_recovery_intervention_v4 as v4
from zuma_rl.alphazuma_55 import DUAL_POSITION_LEVELS


SCRIPT_PATH = Path(__file__).resolve()
BASE_PROBE_PATH = Path(macro_eval.__file__).resolve()
WRAPPER_PATH = macro_eval.WRAPPER_PATH
COMPLETION_SCHEMA = "zuma-rl.alphazuma-55-macro-teacher-v1"
MODE = "teacher_only"
TEACHER_ID = v4.TEACHER_ID
DEFAULT_LEVELS = macro_eval.DEFAULT_LEVELS
DEFAULT_SEED_BASE = macro_eval.DEFAULT_SEED_BASE
DEFAULT_MAX_TICKS = macro_eval.DEFAULT_MAX_TICKS
DEFAULT_HOLD_TICKS = macro_eval.DEFAULT_HOLD_TICKS


class TeacherPanelDecisionPolicy:
    """``model.predict``-compatible facade over per-env teachers.

    The macro-eval loop queries its decision-maker once per macro
    decision point with the CURRENT observations; this facade answers
    with each environment's deterministic teacher proposal on its raw
    frame.  The per-tick masks the loop derives are accepted but
    ignored: the teacher is deterministic code, and macro-mask
    legality is enforced downstream by the shared adapter (illegal or
    unsupported proposals fall back to wait_hold, counted, exactly as
    for a student).
    """

    def __init__(self, teachers: Sequence[Any]) -> None:
        if not teachers:
            raise ValueError("at least one teacher is required")
        self._teachers = tuple(teachers)
        self.decision_queries = 0

    def predict(
        self,
        observations: Any,
        deterministic: bool = True,
        action_masks: Any = None,
    ) -> tuple[np.ndarray, None]:
        if not deterministic:
            raise ValueError("teacher decisions are deterministic only")
        frames = np.asarray(observations)
        if frames.ndim != 2 or frames.shape[0] != len(self._teachers):
            raise ValueError(
                f"expected {len(self._teachers)} raw frames, got "
                f"{frames.shape}"
            )
        actions = np.stack(
            [
                np.asarray(
                    teacher.act(frames[index]), dtype=np.int64
                ).reshape(2)
                for index, teacher in enumerate(self._teachers)
            ]
        )
        self.decision_queries += 1
        return actions, None


def _run_teacher_only(
    *,
    teachers: Sequence[Any],
    vector: Any,
    level_ids: Sequence[str],
    seed_base: int,
    max_ticks: int,
    masks_provider: Callable[[Any], np.ndarray] | None = None,
) -> dict[str, Any]:
    """The macro-eval loop with the teacher facade as decision-maker.

    Every accounting rule (macro telemetry, release errors, mask
    fallbacks, verb counts) is macro-eval's own code; only the
    decision-maker and the reported mode differ.
    """

    result = dict(
        macro_eval._run_student_only(
            model=TeacherPanelDecisionPolicy(teachers),
            vector=vector,
            level_ids=level_ids,
            seed_base=seed_base,
            max_ticks=max_ticks,
            masks_provider=masks_provider,
        )
    )
    result["mode"] = MODE
    return result


def _adapter_contract(hold_ticks: int) -> dict[str, Any]:
    """Macro-eval's adapter contract with the teacher-specific semantics."""

    contract = dict(macro_eval._adapter_contract(hold_ticks))
    contract["decision_maker"] = (
        f"{TEACHER_ID} teacher queried on the current RAW frame at each "
        "macro decision point; no student model exists in this probe"
    )
    contract["per_tick_interface"] = (
        "teacher.act(raw_frame) -> per-tick (verb, aim); the settle-gated "
        "teacher's wait action CARRIES its parked target bin, so "
        "wait -> wait_hold parks exactly the teacher's carried target"
    )
    contract["aim_target"] = (
        "the teacher's commanded aim bin (the carried parked target on "
        "wait proposals)"
    )
    contract["hop_expectation"] = (
        "dual-position levels "
        f"{list(DUAL_POSITION_LEVELS)} need the hop verb, which the macro "
        "interface does not expose; teacher hop proposals fall back to "
        "wait_hold and are counted under mask_fallbacks["
        f"{macro_eval.FALLBACK_UNSUPPORTED!r}], so those levels are "
        "expected to fail through this interface"
    )
    return contract


def _build_completion(
    *,
    result: Mapping[str, Any],
    wrapper_contract: Mapping[str, Any],
    runtime: Mapping[str, Any],
    wall_seconds: float,
    levels: Sequence[str],
    seed_base: int,
    max_ticks: int,
    hold_ticks: int,
) -> dict[str, Any]:
    """Assemble the ceiling-probe receipt; NaN metrics serialize as null."""

    episodes = list(result["episodes"])
    dual = {level.casefold() for level in DUAL_POSITION_LEVELS}
    return v4._json_safe(
        {
            "schema": COMPLETION_SCHEMA,
            "version": 1,
            "status": "COMPLETE",
            "completed_utc": base._utc_now(),
            "wall_seconds": float(wall_seconds),
            "runtime": dict(runtime),
            "mode": MODE,
            "purpose": (
                "macro-interface ceiling measurement: if the teacher wins "
                "most levels through the park-settle macro interface, PPO "
                "on that interface has measured headroom; per-level "
                "outcomes identify interface-blocked levels"
            ),
            "teacher": {
                "id": TEACHER_ID,
                "construction": (
                    "deterministic code built from each environment's "
                    "teacher_spec(); no checkpoint file exists"
                ),
                "observation": "raw_current_frame",
            },
            "adapter": _adapter_contract(hold_ticks),
            "wrapper_contract": dict(wrapper_contract),
            "levels": list(levels),
            "seed_base": int(seed_base),
            "seed_last": int(seed_base) + len(levels) - 1,
            "max_ticks": int(max_ticks),
            "hold_ticks": int(hold_ticks),
            "episodes": episodes,
            "summary": dict(result["summary"]),
            "per_level_outcomes": {
                str(row["level_id"]): row["outcome"] for row in episodes
            },
            "dual_position_levels_in_panel": [
                str(row["level_id"])
                for row in episodes
                if str(row["level_id"]).casefold() in dual
            ],
            "mode_runtime": dict(result["runtime"]),
            "teacher_wins_are_not_autonomy": True,
            "formal_seed_consumption": False,
            "formal_candidate_authority": False,
            "training_authority": False,
        }
    )


def run(
    *,
    original_root: Path,
    run_dir: Path,
    level_ids: Sequence[str] = DEFAULT_LEVELS,
    seed_base: int = DEFAULT_SEED_BASE,
    max_ticks: int = DEFAULT_MAX_TICKS,
    hold_ticks: int = DEFAULT_HOLD_TICKS,
    device: str = "cpu",
) -> dict[str, Any]:
    original_root = original_root.resolve(strict=True)
    run_dir = run_dir.resolve()
    if run_dir.exists():
        raise FileExistsError(
            f"macro teacher probe output already exists: {run_dir}"
        )
    levels = v4._validate_levels(level_ids)
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
            "base_probe": {
                "path": str(BASE_PROBE_PATH),
                "sha256": base._sha256(BASE_PROBE_PATH),
            },
            "wrapper_module": {
                "path": str(WRAPPER_PATH),
                "sha256": base._sha256(WRAPPER_PATH),
            },
            "mode": MODE,
            "teacher_id": TEACHER_ID,
            "teacher_observation": "raw_current_frame",
            "levels": list(levels),
            "seed_base": int(seed_base),
            "seed_last": int(seed_base) + len(levels) - 1,
            "max_ticks": int(max_ticks),
            "hold_ticks": int(hold_ticks),
            "device": str(device),
            "device_role": (
                "none: the teacher is deterministic CPU code; the flag is "
                "recorded for orchestration parity only"
            ),
            "adapter": _adapter_contract(hold_ticks),
        },
    )
    try:
        import torch

        torch.set_num_threads(1)
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
        vector = macro_eval._build_vector(
            original_root=original_root,
            level_ids=levels,
            max_ticks=int(max_ticks),
            hold_ticks=int(hold_ticks),
        )
        try:
            wrapper_contract = vector.env_method("contract")[0]
            teachers = v4._build_teachers(vector)
            result = _run_teacher_only(
                teachers=teachers,
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
            "device_role": "none (teacher is deterministic CPU code)",
        }
        completion = _build_completion(
            result=result,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--device",
        default="cpu",
        help="recorded only: the teacher is deterministic CPU code",
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        default=list(DEFAULT_LEVELS),
        help="probe panel; pass the full 55-level inventory for a ceiling run",
    )
    parser.add_argument("--seed-base", type=int, default=DEFAULT_SEED_BASE)
    parser.add_argument("--max-ticks", type=int, default=DEFAULT_MAX_TICKS)
    parser.add_argument("--hold-ticks", type=int, default=DEFAULT_HOLD_TICKS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(
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
                "mode": result["mode"],
                "wins": summary["wins"],
                "losses": summary["losses"],
                "truncations": summary["truncations"],
                "total_score": summary["total_score"],
                "total_shots": summary["total_shots"],
                "shots_on_target_rate": summary["shots_on_target_rate"],
                "mask_fallbacks": summary["mask_fallbacks"],
                "per_level_outcomes": result["per_level_outcomes"],
                "teacher_wins_are_not_autonomy": True,
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
