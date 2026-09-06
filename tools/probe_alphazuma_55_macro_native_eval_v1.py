"""Evaluate MACRO-NATIVE stacked policies through park-settle macros.

THIN delegate of :mod:`tools.probe_alphazuma_55_macro_eval_v1`.  That
probe evaluates PER-TICK students (``MultiDiscrete([4, 180])``) through
the park-settle macro wrapper via an adapter: it widens the wrapper's
(3 + 180) macro mask to the per-tick (4 + 180) interface (hop masked
off), lets the per-tick model predict, and maps the prediction back to
a macro (with wait_hold fallbacks for hop or macro-mask-illegal verbs).

This probe evaluates MACRO-NATIVE policies -- checkpoints trained by
``tools/train_overnight_multilevel_v2b.py`` whose action space is the
wrapper's own ``MultiDiscrete([3, 180])`` and whose observations are
feature-axis stacked frames (``zuma_rl.observation_stack_wrapper``,
lags ``0,4,8`` by default, extractor
``tools.distill_alphazuma_55_park_settle_v3
.StackedRevengeEntityFeatureExtractor``).  Differences from the
adapter probe, and nothing else:

* NO per-tick adapter: the model's masked deterministic predict output
  IS the macro action, and the masks it sees are the wrapper's own
  ``action_masks()`` (3 + 180), bit for bit.  The run loop itself is
  the frozen ``macro_eval._run_student_only``; a tiny predict shim
  inverts the loop's lossless per-tick mask widening (drop the
  always-False hop column) before the model predicts, so the adapter's
  verb map degenerates to the identity and no fallback can trigger.
  Any recorded fallback fails the run loudly instead of being hidden.
* The env factory composes ``ObservationStackWrapper`` UNDER
  ``ParkSettleActionWrapper`` over the probes' motor-observable stack
  (probe-v5's stacked factory), mirroring the proven trainer
  composition of ``train_overnight_multilevel_v2b.make_stacked_macro_env``:
  the ring buffer advances on EVERY native tick a macro consumes, so
  the policy sees true ``frame(t - lag)`` context at each decision
  point.
* The completion schema bumps to ``zuma-rl.alphazuma-55-macro-native-
  eval-v1`` and additionally records the stack lags and the model's
  action/observation space signature.

Telemetry (wins/losses/score/shots/on-target/macro counts), the
student_only mode semantics, and the on-target meaning (wrapper motor
fidelity, not strategic quality) are macro_eval's, unchanged.  Existing
modules are imported, never modified; this probe consumes engineering
seeds only and carries no training or formal-selection authority.
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

# Imported for the side effect of making the stacked policy's pickled
# feature-extractor class importable before MaskablePPO.load runs.
from tools import distill_alphazuma_55_park_settle_v3 as trainer_v3  # noqa: F401
from tools import probe_alphazuma_55_macro_eval_v1 as macro_eval
from tools import probe_alphazuma_55_recovery_intervention_v1 as base
from tools import probe_alphazuma_55_recovery_intervention_v4 as v4
from tools import probe_alphazuma_55_recovery_intervention_v5 as v5
import zuma_rl.observation_stack_wrapper as observation_stack_wrapper
from zuma_rl.observation_stack_wrapper import (
    DEFAULT_STACK_LAGS,
    parse_stack_lags,
)
from zuma_rl.park_settle_action_wrapper import (
    MACRO_VERB_NAMES,
    ParkSettleActionConfig,
    ParkSettleActionWrapper,
)


SCRIPT_PATH = Path(__file__).resolve()
BASE_PROBE_PATH = Path(macro_eval.__file__).resolve()
STACK_WRAPPER_PATH = Path(observation_stack_wrapper.__file__).resolve()
COMPLETION_SCHEMA = "zuma-rl.alphazuma-55-macro-native-eval-v1"
MODE = macro_eval.MODE
AIM_BINS = macro_eval.AIM_BINS
# The wrapper's native interface: MultiDiscrete([3, 180]).
MACRO_NVEC = (len(MACRO_VERB_NAMES), AIM_BINS)
# The full55-v1 per-tick interface this probe REFUSES: wait/fire/swap/hop.
PER_TICK_NVEC = (macro_eval.PER_TICK_VERB_COUNT, AIM_BINS)
# Column index of hop in macro_eval's widened per-tick mask.
_HOP_COLUMN = len(MACRO_VERB_NAMES)
DEFAULT_LEVELS = macro_eval.DEFAULT_LEVELS
DEFAULT_SEED_BASE = macro_eval.DEFAULT_SEED_BASE
DEFAULT_MAX_TICKS = macro_eval.DEFAULT_MAX_TICKS
DEFAULT_HOLD_TICKS = macro_eval.DEFAULT_HOLD_TICKS


def require_macro_native_model(model: Any) -> tuple[int, ...]:
    """Accept only the wrapper-native ``MultiDiscrete([3, 180])`` model.

    A per-tick full55-v1 model (``MultiDiscrete([4, 180])``) is refused
    by name with a pointer to the adapter probe that exists for it.
    """

    space = getattr(model, "action_space", None)
    nvec = getattr(space, "nvec", None)
    signature = (
        tuple(int(value) for value in np.asarray(nvec).reshape(-1))
        if nvec is not None
        else None
    )
    if signature == MACRO_NVEC:
        return signature
    if signature == PER_TICK_NVEC:
        raise ValueError(
            "per-tick model detected (MultiDiscrete([4, 180]) = "
            "wait/fire/swap/hop): this probe evaluates MACRO-NATIVE "
            "policies only; evaluate per-tick checkpoints through the "
            "adapter probe tools/probe_alphazuma_55_macro_eval_v1.py "
            "(tools.probe_alphazuma_55_macro_eval_v1) instead"
        )
    raise ValueError(
        "macro-native eval requires the wrapper's action space "
        f"MultiDiscrete([{MACRO_NVEC[0]}, {MACRO_NVEC[1]}]); the loaded "
        f"model exposes {space!r}"
    )


def model_space_signature(model: Any) -> dict[str, Any]:
    """Serializable action/observation space signature for receipts."""

    action = model.action_space
    observation = model.observation_space
    return {
        "action_space": {
            "type": type(action).__name__,
            "nvec": [
                int(value) for value in np.asarray(action.nvec).reshape(-1)
            ],
        },
        "observation_space": {
            "type": type(observation).__name__,
            "shape": [int(value) for value in observation.shape],
            "dtype": str(observation.dtype),
        },
    }


class MacroNativePredictShim:
    """Present a macro-native model to macro_eval's frozen run loop.

    ``macro_eval._run_student_only`` widens the wrapper's (3 + 180)
    macro mask through ``per_tick_masks_from_macro`` (an always-False
    hop column at index 3) before predict, then maps the prediction
    back through ``adapt_per_tick_action`` (the identity on verbs
    0..2).  The widening is lossless, so this shim inverts it exactly:
    it strips the hop column and hands the model's masked deterministic
    predict the wrapper's own (3 + 180) mask, bit for bit.  The model's
    output IS the macro action -- and because the model's masked
    distribution enforced the true macro mask, no adapter fallback can
    ever trigger.
    """

    def __init__(self, model: Any) -> None:
        self._model = model

    def predict(
        self,
        observations: Any,
        deterministic: bool = True,
        action_masks: Any = None,
    ) -> tuple[Any, Any]:
        masks = np.asarray(action_masks, dtype=np.bool_)
        expected = macro_eval.PER_TICK_VERB_COUNT + AIM_BINS
        if masks.shape[-1] != expected:
            raise ValueError(
                f"widened per-tick mask width must be {expected}, got "
                f"{masks.shape}"
            )
        if bool(np.any(masks[..., _HOP_COLUMN])):
            raise ValueError(
                "hop column is set: the mask was not derived from the "
                "wrapper's (3 + 180) macro mask"
            )
        macro_masks = np.concatenate(
            (masks[..., :_HOP_COLUMN], masks[..., _HOP_COLUMN + 1 :]),
            axis=-1,
        )
        return self._model.predict(
            observations,
            deterministic=deterministic,
            action_masks=macro_masks,
        )


def _run_macro_native(
    *,
    model: Any,
    vector: Any,
    level_ids: Sequence[str],
    seed_base: int,
    max_ticks: int,
    masks_provider: Callable[[Any], np.ndarray] | None = None,
) -> dict[str, Any]:
    """macro_eval's run loop with the model's output as the macro action.

    Delegates to the frozen ``macro_eval._run_student_only`` through
    :class:`MacroNativePredictShim`, then enforces the native contract:
    the adapter degenerates to the identity, so ANY recorded mask
    fallback means the interface is broken and the run fails loudly.
    """

    result = macro_eval._run_student_only(
        model=MacroNativePredictShim(model),
        vector=vector,
        level_ids=level_ids,
        seed_base=seed_base,
        max_ticks=max_ticks,
        masks_provider=masks_provider,
    )
    fallbacks = dict(result["summary"].get("mask_fallbacks", {}))
    if fallbacks:
        raise RuntimeError(
            "macro-native run recorded adapter fallbacks, which the "
            f"identity mapping makes impossible: {fallbacks}"
        )
    return result


def _make_stacked_macro_env_factory(
    *,
    original_root: Path,
    level_id: str,
    max_ticks: int,
    hold_ticks: int,
    stack_lags: Any = DEFAULT_STACK_LAGS,
) -> Any:
    """Probe-v5's stacked motor stack under the park-settle macro wrapper.

    Mirrors the proven composition order of
    ``tools.train_overnight_multilevel_v2b.make_stacked_macro_env``
    (stack UNDER the macro wrapper) over the probes' motor-observable
    env: motor stack -> ObservationStackWrapper ->
    ParkSettleActionWrapper, so the ring buffer advances on every
    native tick a macro consumes.
    """

    lags = parse_stack_lags(stack_lags)
    stacked_factory = v5._stacked_env_factory(
        original_root=original_root,
        level_id=level_id,
        max_ticks=int(max_ticks),
        stack_lags=lags,
    )
    config = ParkSettleActionConfig(hold_ticks=int(hold_ticks))

    def make_env() -> ParkSettleActionWrapper:
        return ParkSettleActionWrapper(stacked_factory(), config=config)

    return make_env


def _build_vector(
    *,
    original_root: Path,
    level_ids: Sequence[str],
    max_ticks: int,
    hold_ticks: int,
    stack_lags: Any,
) -> Any:
    from stable_baselines3.common.vec_env import SubprocVecEnv

    factories = [
        _make_stacked_macro_env_factory(
            original_root=original_root,
            level_id=level_id,
            max_ticks=int(max_ticks),
            hold_ticks=int(hold_ticks),
            stack_lags=stack_lags,
        )
        for level_id in level_ids
    ]
    return SubprocVecEnv(factories, start_method="forkserver")


def _native_interface_contract(hold_ticks: int) -> dict[str, Any]:
    adapter = macro_eval._adapter_contract(int(hold_ticks))
    return {
        "macro_interface": (
            "MultiDiscrete([3, 180]) masked deterministic predict on the "
            "current stacked decision-point observation"
        ),
        "action_mapping": (
            "none: the model's prediction IS the macro action executed "
            "by the wrapper (no per-tick adapter, no verb remapping, no "
            "fallback path; a recorded fallback fails the run)"
        ),
        "mask_rule": (
            "masks are the wrapper's own action_masks() (3 + 180), "
            "enforced inside the model's masked distribution"
        ),
        "wait_semantics": adapter["wait_semantics"],
        "on_target_semantics": adapter["on_target_semantics"],
    }


def _observation_stack_payload(
    *, stack_lags: Sequence[int], raw_observation_dim: int
) -> dict[str, Any]:
    return {
        "stack_lags": [int(lag) for lag in stack_lags],
        "frames": len(stack_lags),
        "raw_observation_dim": int(raw_observation_dim),
        "stacked_observation_dim": (
            int(raw_observation_dim) * len(stack_lags)
        ),
        "position": "under_park_settle_macro_wrapper",
        "ring_buffer_advances_every_native_tick": True,
        "wrapper_module": {
            "path": str(STACK_WRAPPER_PATH),
            "sha256": base._sha256(STACK_WRAPPER_PATH),
        },
    }


def _build_completion(
    *,
    result: Any,
    source: Any,
    wrapper_contract: Any,
    runtime: Any,
    wall_seconds: float,
    levels: Sequence[str],
    seed_base: int,
    max_ticks: int,
    hold_ticks: int,
    stack_lags: Sequence[int],
    raw_observation_dim: int,
    model_spaces: Any,
) -> dict[str, Any]:
    """macro_eval's completion receipt, schema bumped to macro-native.

    The per-tick adapter block is nulled (no adapter exists on this
    path) and replaced by the native interface contract; the stack lags
    and the model's space signature are recorded alongside.
    """

    completion = macro_eval._build_completion(
        result=result,
        source=source,
        wrapper_contract=wrapper_contract,
        runtime=runtime,
        wall_seconds=wall_seconds,
        levels=levels,
        seed_base=seed_base,
        max_ticks=max_ticks,
        hold_ticks=hold_ticks,
    )
    completion["schema"] = COMPLETION_SCHEMA
    completion["adapter"] = None
    completion["native_interface"] = _native_interface_contract(hold_ticks)
    completion["base_probe"] = {
        "module": "tools.probe_alphazuma_55_macro_eval_v1",
        "path": str(BASE_PROBE_PATH),
        "sha256": base._sha256(BASE_PROBE_PATH),
    }
    completion["observation_stack"] = _observation_stack_payload(
        stack_lags=stack_lags, raw_observation_dim=raw_observation_dim
    )
    completion["model_spaces"] = dict(model_spaces)
    return completion


def run(
    *,
    model_path: Path,
    original_root: Path,
    run_dir: Path,
    level_ids: Sequence[str] = DEFAULT_LEVELS,
    seed_base: int = DEFAULT_SEED_BASE,
    max_ticks: int = DEFAULT_MAX_TICKS,
    hold_ticks: int = DEFAULT_HOLD_TICKS,
    stack_lags: Any = DEFAULT_STACK_LAGS,
    device: str = "cpu",
) -> dict[str, Any]:
    model_path = model_path.resolve(strict=True)
    original_root = original_root.resolve(strict=True)
    run_dir = run_dir.resolve()
    if run_dir.exists():
        raise FileExistsError(
            f"macro-native eval output already exists: {run_dir}"
        )
    lags = parse_stack_lags(stack_lags)
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
            "base_probe": {
                "path": str(BASE_PROBE_PATH),
                "sha256": base._sha256(BASE_PROBE_PATH),
            },
            "wrapper_module": {
                "path": str(macro_eval.WRAPPER_PATH),
                "sha256": base._sha256(macro_eval.WRAPPER_PATH),
            },
            "observation_stack_wrapper": {
                "path": str(STACK_WRAPPER_PATH),
                "sha256": base._sha256(STACK_WRAPPER_PATH),
            },
            "source": source,
            "mode": MODE,
            "levels": list(levels),
            "seed_base": int(seed_base),
            "seed_last": int(seed_base) + len(levels) - 1,
            "max_ticks": int(max_ticks),
            "hold_ticks": int(hold_ticks),
            "stack_lags": [int(lag) for lag in lags],
            "device": str(device),
            "native_interface": _native_interface_contract(hold_ticks),
        },
    )
    model: Any | None = None
    try:
        import torch
        from sb3_contrib import MaskablePPO

        torch.set_num_threads(1)
        model = MaskablePPO.load(str(model_path), device=str(device))
        require_macro_native_model(model)
        stacked_width = int(model.observation_space.shape[0])
        if stacked_width % len(lags):
            raise ValueError(
                f"model observation width {stacked_width} is not "
                f"divisible by {len(lags)} stack frames; --stack-lags "
                f"disagrees with the trained policy"
            )
        raw_observation_dim = stacked_width // len(lags)
        model_spaces = model_space_signature(model)
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
            stack_lags=lags,
        )
        try:
            vector_width = int(vector.observation_space.shape[0])
            if vector_width != stacked_width:
                raise ValueError(
                    f"stacked env width {vector_width} disagrees with "
                    f"the model's {stacked_width}"
                )
            wrapper_contract = vector.env_method("contract")[0]
            result = _run_macro_native(
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
            stack_lags=lags,
            raw_observation_dim=raw_observation_dim,
            model_spaces=model_spaces,
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
    """macro_eval's CLI plus ``--stack-lags``."""

    parser = macro_eval.build_parser()
    parser.description = __doc__
    parser.add_argument(
        "--stack-lags",
        default=",".join(str(lag) for lag in DEFAULT_STACK_LAGS),
        help=(
            "comma-separated source-tick lags the policy was trained "
            "with (feature-axis stack UNDER the macro wrapper); must "
            "start with 0 and match the checkpoint's training lags"
        ),
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
        stack_lags=str(args.stack_lags),
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
