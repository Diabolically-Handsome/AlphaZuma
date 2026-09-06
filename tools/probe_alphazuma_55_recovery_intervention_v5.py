"""Probe V5: the V4 override-window probe for lag-stacked students.

THIN wrapper over ``tools.probe_alphazuma_55_recovery_intervention_v4``.
Teacher construction, the V4 override-window controller, the W1
closed-loop metrics, the mode loop (``_run_mode``), the paired decision
gate, and all accounting are v4's, imported and delegated to, never
re-implemented.  v5 exists for exactly one reason: park-settle v3
students consume lag-stacked observations
(``obs_dim x n_lags`` feature concatenation, see
``tools.distill_alphazuma_55_park_settle_v3`` and
``zuma_rl.observation_stack_wrapper``), so the probe must

1. wrap EVERY env with ``ObservationStackWrapper`` inside the env
   factory (the existing per-env wrapping convention), so the stacked
   policy sees the training-time input layout; and
2. keep the deterministic teacher on the RAW current frame.  v4's
   ``_run_mode`` hands ``observations[index]`` -- now stacked -- to
   ``teacher.act``; because the first stack lag is forced to 0, the raw
   current frame is exactly the leading ``raw_observation_dim`` slice,
   and each teacher is wrapped in ``RawFrameTeacher`` which strips the
   lagged tail before delegating.  Teacher and intervention logic are
   byte-identical to v4.

The completion schema bumps to v5 and records ``observation_stack``
(stack_lags, frame widths, the raw-frame teacher contract).  An
optional ``--preregistration`` still binds a V4 preregistration
document (the executor and gate are v4's).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, Callable, Sequence

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import numpy as np

from tools import distill_alphazuma_55_motor_observable_replay_v2 as motor
# Imported for the side effect of making the stacked student's pickled
# feature-extractor class importable before MaskablePPO.load runs.
from tools import distill_alphazuma_55_park_settle_v3 as trainer_v3  # noqa: F401
from tools import probe_alphazuma_55_recovery_intervention_v1 as base
from tools import probe_alphazuma_55_recovery_intervention_v4 as v4
from tools.alphazuma_55_recovery_intervention_v4 import (
    DecisionGateConfigV4,
    RecoveryInterventionConfigV4,
)
import zuma_rl.observation_stack_wrapper as observation_stack_wrapper
from zuma_rl.observation_stack_wrapper import (
    DEFAULT_STACK_LAGS,
    ObservationStackWrapper,
    parse_stack_lags,
)


SCRIPT_PATH = Path(__file__).resolve()
BASE_PROBE_PATH = Path(v4.__file__).resolve()
WRAPPER_PATH = Path(observation_stack_wrapper.__file__).resolve()
EXPECTED_MODES = v4.EXPECTED_MODES
TEACHER_ID = v4.TEACHER_ID
DEFAULT_SEED_BASE = v4.DEFAULT_SEED_BASE
DEFAULT_MAX_TICKS = v4.DEFAULT_MAX_TICKS
DEFAULT_LEVELS = v4.DEFAULT_LEVELS
COMPLETION_SCHEMA = (
    "zuma-rl.alphazuma-55-recovery-intervention-v5-probe-completion"
)
CONFIG_SCHEMA = "zuma-rl.alphazuma-55-recovery-intervention-v5-probe-config"
STATUS_SCHEMA = "zuma-rl.alphazuma-55-recovery-intervention-v5-probe-status"
FAILURE_SCHEMA = "zuma-rl.alphazuma-55-recovery-intervention-v5-probe-failure"


class RawFrameTeacher:
    """Delegate ``act`` on the RAW current frame of a stacked observation.

    The first stack lag is forced to 0, so the current raw frame is the
    leading ``raw_observation_dim`` slice; the deterministic teacher
    never sees the lagged tail.
    """

    def __init__(self, teacher: Any, *, raw_observation_dim: int) -> None:
        raw_observation_dim = int(raw_observation_dim)
        if raw_observation_dim < 1:
            raise ValueError("raw observation width must be positive")
        self._teacher = teacher
        self._raw_observation_dim = raw_observation_dim

    def act(self, observation: Any) -> Any:
        frame = np.asarray(observation).reshape(-1)
        if frame.shape[0] < self._raw_observation_dim:
            raise ValueError(
                f"stacked observation narrower ({frame.shape[0]}) than the "
                f"raw frame ({self._raw_observation_dim})"
            )
        return self._teacher.act(frame[: self._raw_observation_dim])


def wrap_raw_frame_teachers(
    teachers: Sequence[Any], *, raw_observation_dim: int
) -> list[RawFrameTeacher]:
    return [
        RawFrameTeacher(teacher, raw_observation_dim=raw_observation_dim)
        for teacher in teachers
    ]


def _stacked_env_factory(
    *,
    original_root: Path,
    level_id: str,
    max_ticks: int,
    stack_lags: tuple[int, ...],
) -> Any:
    base_factory = motor._make_motor_env_factory(
        original_root=original_root,
        level_id=level_id,
        max_ticks=max_ticks,
        input_config=base._input_config(),
        reward_config=base._reward_config(),
    )

    def make_env() -> ObservationStackWrapper:
        return ObservationStackWrapper(
            base_factory(), stack_lags=stack_lags
        )

    return make_env


def _build_stacked_vector(
    *,
    original_root: Path,
    level_ids: Sequence[str],
    max_ticks: int,
    stack_lags: tuple[int, ...],
) -> Any:
    from stable_baselines3.common.vec_env import SubprocVecEnv

    factories = [
        _stacked_env_factory(
            original_root=original_root,
            level_id=level_id,
            max_ticks=max_ticks,
            stack_lags=stack_lags,
        )
        for level_id in level_ids
    ]
    return SubprocVecEnv(factories, start_method="forkserver")


def _run_stacked_mode(
    *,
    mode: str,
    model: Any,
    vector: Any,
    teachers: Sequence[Any],
    level_ids: Sequence[str],
    seed_base: int,
    max_ticks: int,
    controller_config: RecoveryInterventionConfigV4,
    raw_observation_dim: int,
    masks_provider: Callable[[Any], np.ndarray] | None = None,
) -> dict[str, Any]:
    """v4's ``_run_mode`` with raw-frame teachers; nothing else changes."""

    return v4._run_mode(
        mode=mode,
        model=model,
        vector=vector,
        teachers=wrap_raw_frame_teachers(
            teachers, raw_observation_dim=raw_observation_dim
        ),
        level_ids=level_ids,
        seed_base=seed_base,
        max_ticks=max_ticks,
        controller_config=controller_config,
        masks_provider=masks_provider,
    )


def _observation_stack_payload(
    *, stack_lags: tuple[int, ...], raw_observation_dim: int
) -> dict[str, Any]:
    return {
        "stack_lags": [int(lag) for lag in stack_lags],
        "frames": len(stack_lags),
        "raw_observation_dim": int(raw_observation_dim),
        "stacked_observation_dim": int(raw_observation_dim)
        * len(stack_lags),
        "policy_observation": "feature_axis_stacked_frames",
        "teacher_observation": "raw_current_frame_leading_slice",
        "wrapper_module": {
            "path": str(WRAPPER_PATH),
            "sha256": base._sha256(WRAPPER_PATH),
        },
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
    stack_lags: tuple[int, ...],
    raw_observation_dim: int,
) -> dict[str, Any]:
    """v4's completion receipt, schema bumped to v5 plus the stack block."""

    completion = v4._build_completion(
        modes=modes,
        decision=decision,
        source=source,
        runtime=runtime,
        wall_seconds=wall_seconds,
        controller_config=controller_config,
        gate=gate,
        preregistration=preregistration,
    )
    completion["schema"] = COMPLETION_SCHEMA
    completion["base_probe"] = {
        "module": "tools.probe_alphazuma_55_recovery_intervention_v4",
        "path": str(BASE_PROBE_PATH),
        "sha256": base._sha256(BASE_PROBE_PATH),
    }
    completion["observation_stack"] = _observation_stack_payload(
        stack_lags=stack_lags, raw_observation_dim=raw_observation_dim
    )
    return completion


def run(
    *,
    model_path: Path,
    original_root: Path,
    run_dir: Path,
    level_ids: Sequence[str] = DEFAULT_LEVELS,
    seed_base: int = DEFAULT_SEED_BASE,
    max_ticks: int = DEFAULT_MAX_TICKS,
    device: str = "cpu",
    stack_lags: Any = DEFAULT_STACK_LAGS,
    controller_config: RecoveryInterventionConfigV4 | None = None,
    gate: DecisionGateConfigV4 | None = None,
    preregistration_path: Path | None = None,
) -> dict[str, Any]:
    """Mirror of ``v4.run`` with stacked envs and raw-frame teachers."""

    model_path = model_path.resolve(strict=True)
    original_root = original_root.resolve(strict=True)
    run_dir = run_dir.resolve()
    if run_dir.exists():
        raise FileExistsError(f"V5 probe output already exists: {run_dir}")
    lags = parse_stack_lags(stack_lags)
    levels = v4._validate_levels(level_ids)
    controller_config = (
        controller_config
        if controller_config is not None
        else RecoveryInterventionConfigV4()
    )
    gate = gate if gate is not None else DecisionGateConfigV4()
    preregistration = (
        v4._bind_preregistration(preregistration_path)
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
            "schema": CONFIG_SCHEMA,
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
            "controller": {
                "path": str(v4.CONTROLLER_PATH),
                "sha256": base._sha256(v4.CONTROLLER_PATH),
            },
            "metrics_module": {
                "path": str(v4.METRICS_PATH),
                "sha256": base._sha256(v4.METRICS_PATH),
            },
            "observation_stack_wrapper": {
                "path": str(WRAPPER_PATH),
                "sha256": base._sha256(WRAPPER_PATH),
            },
            "stack_lags": [int(lag) for lag in lags],
            "preregistration": preregistration,
            "source": source,
            "teacher_id": TEACHER_ID,
            "teacher_observation": "raw_current_frame_leading_slice",
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
        stacked_width = int(model.observation_space.shape[0])
        if stacked_width % len(lags):
            raise ValueError(
                f"model observation width {stacked_width} is not divisible "
                f"by {len(lags)} stack frames; --stack-lags disagrees with "
                f"the trained student"
            )
        raw_observation_dim = stacked_width // len(lags)
        modes: list[dict[str, Any]] = []
        for mode in EXPECTED_MODES:
            base._write_atomic(
                run_dir / "status.json",
                {
                    "schema": STATUS_SCHEMA,
                    "version": 1,
                    "status": "RUNNING",
                    "stage": mode,
                    "updated_utc": base._utc_now(),
                    "completed_modes": [row["mode"] for row in modes],
                    "wall_seconds": time.perf_counter() - started,
                },
            )
            vector = _build_stacked_vector(
                original_root=original_root,
                level_ids=levels,
                max_ticks=max_ticks,
                stack_lags=lags,
            )
            try:
                vector_width = int(vector.observation_space.shape[0])
                if vector_width != stacked_width:
                    raise ValueError(
                        f"stacked env width {vector_width} disagrees with "
                        f"the model's {stacked_width}"
                    )
                teachers = v4._build_teachers(vector)
                result = _run_stacked_mode(
                    mode=mode,
                    model=model,
                    vector=vector,
                    teachers=teachers,
                    level_ids=levels,
                    seed_base=int(seed_base),
                    max_ticks=int(max_ticks),
                    controller_config=controller_config,
                    raw_observation_dim=raw_observation_dim,
                )
            finally:
                vector.close()
            modes.append(result)
            base._write_atomic(
                run_dir / f"{mode}.json", v4._json_safe(result)
            )
        decision = v4._decision(gate=gate, level_ids=levels, modes=modes)
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
            stack_lags=lags,
            raw_observation_dim=raw_observation_dim,
        )
        base._write_atomic(run_dir / "completion.json", completion)
        return completion
    except BaseException as error:
        base._write_atomic(
            run_dir / "failure.json",
            {
                "schema": FAILURE_SCHEMA,
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
    """v4's CLI plus ``--stack-lags``."""

    parser = v4.build_parser()
    parser.description = __doc__
    parser.add_argument(
        "--stack-lags",
        default=",".join(str(lag) for lag in DEFAULT_STACK_LAGS),
        help=(
            "comma-separated source-tick lags the student was trained "
            "with; must match tools/distill_alphazuma_55_park_settle_v3 "
            "and start with 0"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if str(args.teacher_id) != TEACHER_ID:
        raise ValueError(
            f"only the {TEACHER_ID} teacher implementation exists"
        )
    lags = parse_stack_lags(args.stack_lags)
    model_path = args.model_path.expanduser().resolve(strict=True)
    original_root = args.original_root.expanduser().resolve(strict=True)
    levels = v4._validate_levels(args.levels)
    preregistration = (
        v4._bind_preregistration(args.preregistration.expanduser())
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
                    "stack_lags": [int(lag) for lag in lags],
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
        stack_lags=lags,
        preregistration_path=args.preregistration,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "decision": result["decision"]["status"],
                "stack_lags": result["observation_stack"]["stack_lags"],
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
