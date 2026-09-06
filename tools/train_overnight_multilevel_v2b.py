"""v2b: v2 PPO training with observation stacking UNDER the macro wrapper.

Why this file exists.  ``tools/train_overnight_multilevel_v2.py`` fixes
the verified v1 pathologies (configurable gamma, per-level max_ticks,
truncation-neutral reward, live warm-start defaults) and can train
through the park-settle macro wrapper (``--use-park-settle-wrapper``),
but it was built BEFORE ``zuma_rl.observation_stack_wrapper`` existed:
its env stack has no stacking hook of any kind (no TrainerV2Config
field, no wrapper import), so composing the frame-stacked BC init with
the macro interface is impossible via configuration alone.  This v2b is
the thinnest possible delegate over v2's frozen bytes, in the same
rebinding style the distill v2/v3 lineage used: v2's preregistration
validation, PPO construction, rollout/learn loop, receipts, telemetry,
and checkpointing all execute UNCHANGED inside v2; v2b only swaps the
per-worker environment factory (and, during prevalidation, the
per-level env maker) for one that inserts the stack wrapper, and
injects an ``--init-model`` into the in-memory preregistration mapping
so v2's existing bitwise initial-model copy path runs on the
transplanted macro policy.

Composition order (per level)::

    RevengeEnv (per-level max_ticks)
      -> DenseShaping(Observable)HumanSpeedrunWrapper       (v2)
      -> TruncationNeutralRewardWrapper                     (v2)
      -> ObservationStackWrapper(stack_lags)                (v2b insert)
      -> ParkSettleActionWrapper                            (macro)

The stack sits UNDER the macro wrapper, so the ring buffer advances on
EVERY native tick a macro consumes and the policy sees true
``frame(t - lag)`` context at each macro decision point -- the same
input layout the distill-v3 student was trained on and the probe-v5
runtime convention.  Stacking OVER the macro would instead lag by whole
macros (~20-70 native ticks each) and is deliberately not offered.

What v2b adds to the CLI: ``--stack-lags`` (default ``0,4,8``; must
match the init model's training lags) and ``--init-model`` (the
transplanted macro zip from
``tools/build_alphazuma_55_macro_policy_bootstrap_v1.py``; recorded
with its sha256 in v2's own receipts).  Everything else is v2's parser.

New files only: v2's module-level names are temporarily rebound in
memory around the delegated calls and always restored; no frozen file
changes bytes.  A ``delegate_v2b.json`` receipt records the delegation,
the composition order, and the wrapper hashes next to v2's receipts.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from functools import partial
import json
from pathlib import Path
import sys
from typing import Any, Mapping

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import gymnasium as gym

# Imported for the side effect of making the transplanted model's pickled
# feature-extractor class importable before MaskablePPO.load runs.
from tools import distill_alphazuma_55_park_settle_v3 as trainer_v3  # noqa: F401
from tools import train_overnight_multilevel_v2 as v2
from tools.train_overnight_multilevel import (
    _sha256,
    _validate_catalog,
    _write_json_atomic,
)
import zuma_rl.observation_stack_wrapper as observation_stack_wrapper
from zuma_rl.observation_stack_wrapper import (
    DEFAULT_STACK_LAGS,
    ObservationStackWrapper,
    parse_stack_lags,
)
import zuma_rl.park_settle_action_wrapper as park_settle_action_wrapper
from zuma_rl.park_settle_action_wrapper import (
    ParkSettleActionConfig,
    ParkSettleActionWrapper,
)


SCRIPT_PATH = Path(__file__).resolve()
V2_PATH = Path(v2.__file__).resolve()

# Captured ONCE at import, before any rebinding.  make_stacked_macro_env
# must build its inner stack from v2's pristine per-level maker: while
# prevalidate_spaces_v2b has ``v2._make_human_env_v2`` temporarily
# rebound to the stacked maker itself, resolving that name at call time
# would make the stacked maker call itself (RecursionError).  Subprocess
# training workers import this module fresh and capture the same
# pristine original, so training behavior is unchanged.
_V2_MAKE_HUMAN_ENV_ORIGINAL = v2._make_human_env_v2
DELEGATE_SCHEMA = "zuma-rl.overnight-multilevel-delegate-v2b"
COMPOSITION_ORDER = (
    "RevengeEnv",
    "DenseShaping(Observable)HumanSpeedrunWrapper",
    "TruncationNeutralRewardWrapper",
    "ObservationStackWrapper",
    "ParkSettleActionWrapper",
)


def make_stacked_macro_env(
    *,
    level_id: str,
    base_config: Any,
    input_config: Any,
    trainer_config: v2.TrainerV2Config,
    reward_profile: Mapping[str, Any] | None = None,
    original_root: Path | None = None,
    park_settle_config: ParkSettleActionConfig | None = None,
    stack_lags: Any = DEFAULT_STACK_LAGS,
) -> gym.Env:
    """v2's per-level stack with ObservationStackWrapper UNDER the macro.

    The inner stack is built by v2's own ``_make_human_env_v2`` (the
    import-time capture, immune to the prevalidation rebinding) with the
    park-settle flag forced off, then stacking is inserted, then the
    macro wrapper goes on top iff the trainer config asks for it -- so
    the ring buffer advances every native tick a macro consumes.
    """

    lags = parse_stack_lags(stack_lags)
    inner = _V2_MAKE_HUMAN_ENV_ORIGINAL(
        level_id=level_id,
        base_config=base_config,
        input_config=input_config,
        trainer_config=replace(trainer_config, use_park_settle_wrapper=False),
        reward_profile=reward_profile,
        original_root=original_root,
    )
    env: gym.Env = ObservationStackWrapper(inner, stack_lags=lags)
    if trainer_config.use_park_settle_wrapper:
        env = ParkSettleActionWrapper(env, config=park_settle_config)
    return env


class StackedMultiLevelSpeedrunEnvV2(v2.MultiLevelSpeedrunEnvV2):
    """v2's multi-level env whose per-level stacks carry the lag stack."""

    def __init__(self, *, stack_lags: Any = DEFAULT_STACK_LAGS, **kwargs: Any) -> None:
        # Parsed BEFORE super().__init__, which builds the first level.
        self.stack_lags = parse_stack_lags(stack_lags)
        super().__init__(**kwargs)

    def _make_env(self, level_id: str) -> gym.Env:
        return make_stacked_macro_env(
            level_id=level_id,
            base_config=self.base_config,
            input_config=self.input_config,
            trainer_config=self.trainer_config,
            reward_profile=self.reward_profile,
            original_root=self.original_root,
            stack_lags=self.stack_lags,
        )


def _make_worker_v2b(
    *,
    stack_lags: Any,
    original_root: Path | None,
    levels: Any,
    weights: Mapping[str, float],
    base_config: Any,
    input_config: Any,
    trainer_config: v2.TrainerV2Config,
    reward_profile: Mapping[str, Any] | None,
    selector_seed: int,
    episode_seed_base: int,
    episode_seed_stride: int,
    worker_rank: int,
) -> gym.Env:
    """Mirror of v2's worker factory over the stacked multi-level env."""

    from stable_baselines3.common.monitor import Monitor

    environment = StackedMultiLevelSpeedrunEnvV2(
        stack_lags=stack_lags,
        original_root=original_root,
        levels=levels,
        weights=weights,
        base_config=base_config,
        input_config=input_config,
        trainer_config=trainer_config,
        reward_profile=reward_profile,
        selector_seed=selector_seed + worker_rank,
        episode_seed_base=episode_seed_base,
        episode_seed_stride=episode_seed_stride,
        worker_rank=worker_rank,
    )
    return Monitor(environment)


def inject_init_model(
    prereg: dict[str, Any], init_model_path: Path
) -> dict[str, Any]:
    """Bind ``--init-model`` into the in-memory preregistration mapping.

    v2's existing machinery then does all the work: prevalidation loads
    the model and enforces space equality against the (stacked, macro)
    env stack, and the fresh-start path copies the policy bitwise while
    the optimizer starts clean.  The injected block is recorded verbatim
    in v2's run receipt.  A CLI init model wins over a preregistered
    ``initial_model`` (the replacement is recorded).
    """

    path = init_model_path.resolve(strict=True)
    replaced = prereg.get("initial_model")
    prereg["initial_model"] = {
        "path": str(path),
        "sha256": _sha256(path),
        "injected_by": "tools/train_overnight_multilevel_v2b.py --init-model",
        "replaced_preregistration_initial_model": replaced,
    }
    return prereg["initial_model"]


def prevalidate_spaces_v2b(
    *,
    prereg: dict[str, Any],
    original_root: Path | None,
    base_config: Any,
    input_config: Any,
    reward_profile: Mapping[str, Any] | None,
    trainer_config: v2.TrainerV2Config,
    stack_lags: Any,
) -> dict[str, Any]:
    """v2's prevalidation over the stacked composition.

    ``v2._prevalidate_spaces_v2`` resolves the per-level env maker as a
    module global, so it is rebound to the stacked maker for the
    duration of the call and always restored.
    """

    lags = parse_stack_lags(stack_lags)
    original = v2._make_human_env_v2
    v2._make_human_env_v2 = partial(make_stacked_macro_env, stack_lags=lags)
    try:
        validation = v2._prevalidate_spaces_v2(
            prereg=prereg,
            original_root=original_root,
            base_config=base_config,
            input_config=input_config,
            reward_profile=reward_profile,
            trainer_config=trainer_config,
        )
    finally:
        v2._make_human_env_v2 = original
    raw_width = None
    shape = validation.get("observation_shape")
    if shape and int(shape[0]) % len(lags) == 0:
        raw_width = int(shape[0]) // len(lags)
    validation["observation_stack"] = {
        "stack_lags": [int(lag) for lag in lags],
        "frames": len(lags),
        "stacked_observation_dim": int(shape[0]) if shape else None,
        "raw_observation_dim": raw_width,
        "position": "under_park_settle_macro_wrapper",
    }
    return validation


def _write_delegate_receipt(
    *,
    run_spec: Mapping[str, Any],
    stack_lags: Any,
    prereg: Mapping[str, Any],
) -> None:
    """Record the delegation next to v2's receipts (post-run, additive)."""

    run_dir = Path(str(run_spec["run_dir"])).resolve()
    if not run_dir.exists():
        return
    lags = parse_stack_lags(stack_lags)
    wrapper_path = Path(observation_stack_wrapper.__file__).resolve()
    macro_path = Path(park_settle_action_wrapper.__file__).resolve()
    _write_json_atomic(
        run_dir / "delegate_v2b.json",
        {
            "schema": DELEGATE_SCHEMA,
            "version": 1,
            "delegate": {
                "path": str(SCRIPT_PATH),
                "sha256": _sha256(SCRIPT_PATH),
            },
            "delegated_trainer": {
                "path": str(V2_PATH),
                "sha256": _sha256(V2_PATH),
            },
            "observation_stack_wrapper": {
                "path": str(wrapper_path),
                "sha256": _sha256(wrapper_path),
            },
            "park_settle_action_wrapper": {
                "path": str(macro_path),
                "sha256": _sha256(macro_path),
            },
            "stack_lags": [int(lag) for lag in lags],
            "composition_order": list(COMPOSITION_ORDER),
            "stack_position": "under_park_settle_macro_wrapper",
            "ring_buffer_advances_every_native_tick": True,
            "initial_model": prereg.get("initial_model"),
            "delegation": (
                "preregistration validation, PPO construction, "
                "rollout/learn loop, receipts, telemetry, and "
                "checkpointing are v2's code; v2b rebinds only the env "
                "factories and injects --init-model"
            ),
        },
    )


def run_training_v2b(
    *,
    prereg_path: Path,
    prereg: dict[str, Any],
    run_spec: dict[str, Any],
    original_root: Path | None,
    trainer_config: v2.TrainerV2Config,
    validation: dict[str, Any],
    stack_lags: Any,
) -> dict[str, Any]:
    """v2's training run with the worker factory rebound to the stack.

    ``v2._run_training_v2`` resolves ``_make_worker_v2`` as a module
    global when it builds the SubprocVecEnv factories, so rebinding it
    here routes every worker through the stacked composition while the
    entire loop, receipt, and checkpoint machinery stay v2's.  The
    rebinding is restored even on failure; the child processes never
    depend on it (they unpickle ``_make_worker_v2b`` by qualified name
    from this module).
    """

    lags = parse_stack_lags(stack_lags)
    original = v2._make_worker_v2
    v2._make_worker_v2 = partial(_make_worker_v2b, stack_lags=lags)
    try:
        completion = v2._run_training_v2(
            prereg_path=prereg_path,
            prereg=prereg,
            run_spec=run_spec,
            original_root=original_root,
            trainer_config=trainer_config,
            validation=validation,
        )
    finally:
        v2._make_worker_v2 = original
        _write_delegate_receipt(
            run_spec=run_spec, stack_lags=lags, prereg=prereg
        )
    return completion


def build_parser() -> argparse.ArgumentParser:
    """v2's CLI plus ``--stack-lags`` and ``--init-model``."""

    parser = v2.build_parser()
    parser.description = __doc__
    parser.add_argument(
        "--stack-lags",
        default=",".join(str(lag) for lag in DEFAULT_STACK_LAGS),
        help=(
            "comma-separated source-tick lags stacked along the feature "
            "axis UNDER the macro wrapper; must start with 0 and match "
            "the init model's training lags"
        ),
    )
    parser.add_argument(
        "--init-model",
        type=Path,
        default=None,
        help=(
            "transplanted macro-interface MaskablePPO zip (from "
            "tools/build_alphazuma_55_macro_policy_bootstrap_v1.py); "
            "injected as the preregistration's initial_model, so v2's "
            "bitwise policy copy and space prevalidation apply to it"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    lags = parse_stack_lags(args.stack_lags)
    prereg_path = args.preregistration.expanduser().resolve(strict=True)
    original_root = args.original_root.expanduser().resolve(strict=True)
    prereg = v2._validate_preregistration_v2(prereg_path)
    _validate_catalog(prereg=prereg, original_root=original_root)
    injected = None
    if args.init_model is not None:
        injected = inject_init_model(
            prereg, args.init_model.expanduser()
        )
    base_config, input_config, reward_profile = v2._configs_v2(prereg)
    cli_overrides = v2._cli_trainer_overrides(args)
    run_spec: dict[str, Any] | None = None
    if args.run_id:
        matches = [run for run in prereg["runs"] if run["id"] == args.run_id]
        if len(matches) != 1:
            raise SystemExit(f"unknown or duplicate run id: {args.run_id}")
        run_spec = dict(matches[0])
        if args.learning_rate is not None:
            run_spec["learning_rate"] = float(args.learning_rate)
        if args.entropy_coef is not None:
            run_spec["entropy_coef"] = float(args.entropy_coef)
    trainer_config = v2._trainer_config_for_run(
        prereg, run_spec, cli_overrides
    )
    validation = prevalidate_spaces_v2b(
        prereg=prereg,
        original_root=original_root,
        base_config=base_config,
        input_config=input_config,
        reward_profile=reward_profile,
        trainer_config=trainer_config,
        stack_lags=lags,
    )
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "VALID",
                    "preregistration_sha256": _sha256(prereg_path),
                    "trainer_sha256": _sha256(V2_PATH),
                    "delegate_sha256": _sha256(SCRIPT_PATH),
                    "injected_init_model": injected,
                    **validation,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if run_spec is None:
        raise SystemExit("--run-id is required unless --validate-only is used")
    completion = run_training_v2b(
        prereg_path=prereg_path,
        prereg=prereg,
        run_spec=run_spec,
        original_root=original_root,
        trainer_config=trainer_config,
        validation=validation,
        stack_lags=lags,
    )
    print(json.dumps(completion, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
