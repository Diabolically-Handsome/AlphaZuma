"""Run diagnostic or scope-certified PPO training.

Closed-gate runs remain capped and explicitly non-transferable.  Transfer mode
is available only when independent fidelity and training-recipe suites both
evaluate to OPEN and the request exactly matches their fixed policies.
"""

from __future__ import annotations

import argparse
from functools import partial
import hashlib
import importlib.metadata
import json
import math
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gymnasium.wrappers import TimeLimit

from zuma_rl.config import ZumaConfig
from zuma_rl.env import ZumaEnv
from zuma_rl.fidelity_gate import (
    FidelityGateReport,
    FidelityGateStatus,
    GATE_POLICIES,
)
from zuma_rl.fidelity_runtime import (
    verify_fidelity_suite_for_training as verify_fidelity_suite,
)
from zuma_rl.original_data import OriginalDataError, find_original_installation
from zuma_rl.revenge_core import SUPPORTED_PROFILE_MODE
from zuma_rl.revenge_env import RevengeEnv, RevengeEnvConfig
from zuma_rl.training_gate import (
    TRANSFER_POLICY_ID,
    TRANSFER_STAGE_ID,
    TrainingGateReport,
    TrainingGateStatus,
    training_request_mismatches,
    verify_training_suite,
)

FIDELITY_GATE = "closed"
DEFAULT_DIAGNOSTIC_STEPS = 10_000
CLOSED_GATE_MAX_EFFECTIVE_STEPS = 100_000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--environment",
        choices=("revenge", "simple"),
        default="revenge",
        help=(
            "Diagnostic environment. 'revenge' reads original Jungle1 data; "
            "'simple' is only a non-transferable reference smoke test."
        ),
    )
    parser.add_argument(
        "--acknowledge-fidelity-gate",
        action="store_true",
        help=(
            "Acknowledge that the fidelity gate is CLOSED and the run is "
            "diagnostic/non-transferable."
        ),
    )
    parser.add_argument(
        "--transfer-training",
        action="store_true",
        help=(
            "Request bounded state-policy transfer training. This requires "
            "both --fidelity-suite and --training-suite to evaluate to OPEN."
        ),
    )
    parser.add_argument(
        "--fidelity-suite",
        type=Path,
        help=(
            "Content-addressed multi-case suite required by "
            "--transfer-training."
        ),
    )
    parser.add_argument(
        "--fidelity-suite-root",
        type=Path,
        help=(
            "Evidence root for --fidelity-suite; defaults to the suite "
            "directory."
        ),
    )
    parser.add_argument(
        "--training-suite",
        type=Path,
        help=(
            "Content-addressed scale-stability and recipe suite required by "
            "--transfer-training."
        ),
    )
    parser.add_argument(
        "--training-suite-root",
        type=Path,
        help=(
            "Evidence root for --training-suite; defaults to the suite "
            "directory."
        ),
    )
    parser.add_argument(
        "--allow-reference-smoke",
        action="store_true",
        help=(
            "Additionally permit --environment simple. This never marks the "
            "reference simulator as suitable for transferable training."
        ),
    )

    parser.add_argument("--level", default="Jungle1")
    parser.add_argument(
        "--original-root",
        type=Path,
        default=None,
        help=(
            "Original Zuma's Revenge directory containing ZumasRevenge.exe, "
            "main.pak, and levels/. Otherwise auto-detect or use "
            "ZUMA_REVENGE_ROOT."
        ),
    )
    parser.add_argument("--hard", action="store_true")
    parser.add_argument("--curve-index", type=int, default=0)
    parser.add_argument("--aim-bins", type=int, default=180)
    parser.add_argument(
        "--flat-actions",
        action="store_true",
        help=(
            "Use the legacy 3*aim_bins Discrete encoding. The default "
            "MultiDiscrete [verb, aim] encoding avoids duplicate wait/swap "
            "categories without changing game inputs."
        ),
    )
    parser.add_argument(
        "--mask-rejected-actions",
        action="store_true",
        help=(
            "Use MaskablePPO to suppress fire/swap requests only when the "
            "retail state machine would reject them. The mask is derived "
            "entirely from actor-observable state and does not alter physics."
        ),
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=1,
        help=(
            "Native 100 Hz ticks per policy decision. Fidelity default is 1; "
            "larger values are an explicit diagnostic control abstraction."
        ),
    )
    parser.add_argument("--max-ticks", type=int, default=180_000)
    parser.add_argument(
        "--training-time-limit-steps",
        type=int,
        default=0,
        help=(
            "Diagnostic curriculum boundary applied only around each training "
            "environment. Zero disables it. Evaluation keeps the full "
            "--max-ticks horizon, and the wrapped simulator observations, "
            "rewards, actions, and native tick dynamics are unchanged."
        ),
    )
    parser.add_argument(
        "--max-balls",
        type=int,
        default=192,
        help=(
            "Fixed visible-ball capacity in the actor observation. The "
            "environment fails closed instead of silently truncating when "
            "this capacity is exceeded."
        ),
    )
    parser.add_argument("--simple-initial-balls", type=int, default=24)
    parser.add_argument("--simple-max-steps", type=int, default=600)

    parser.add_argument(
        "--total-steps",
        type=int,
        default=DEFAULT_DIAGNOSTIC_STEPS,
        help=(
            "Requested diagnostic timesteps. While the fidelity gate is "
            "closed, complete PPO rollouts may not exceed 100,000 effective "
            "timesteps."
        ),
    )
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-dir", type=Path, default=Path("runs/ppo"))
    parser.add_argument(
        "--initial-model",
        type=Path,
        default=None,
        help=(
            "Initialize Revenge MaskablePPO from a saved model. Transfer "
            "mode additionally requires the exact model identity authorized "
            "by the Training Gate."
        ),
    )
    parser.add_argument(
        "--freeze-aim-path",
        action="store_true",
        help=(
            "For an initialized factorized MaskablePPO diagnostic, freeze "
            "the feature extractor and all aim-logit rows while leaving "
            "verb logits and the value network trainable. Bitwise freeze "
            "receipts are required for completion."
        ),
    )
    parser.add_argument(
        "--reset-optimizer-state",
        action="store_true",
        help=(
            "Discard all optimizer moments loaded with --initial-model while "
            "leaving policy/value parameters bitwise unchanged. This is "
            "intended for a supervised-to-PPO objective transition and writes "
            "a required reset receipt."
        ),
    )
    parser.add_argument(
        "--teacher-kl-coef",
        type=float,
        default=0.0,
        help=(
            "Forward-KL coefficient anchoring each PPO minibatch to the "
            "immutable --initial-model policy on current rollout states. "
            "Zero disables anchoring."
        ),
    )
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1_024)
    parser.add_argument(
        "--ppo-epochs",
        type=int,
        default=10,
        help="Optimization epochs per PPO rollout batch.",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--reward-scale",
        type=float,
        default=0.01,
        help=(
            "Uniformly scale score, step, win, and loss rewards for PPO "
            "numerics. This does not change the reward-optimal policy."
        ),
    )
    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=0.001,
        help=(
            "PPO entropy coefficient. The factorized action has about 6.29 "
            "nats at maximum entropy, so the previous 0.01 was overly strong."
        ),
    )
    parser.add_argument("--checkpoint-every", type=int, default=1_000_000)
    parser.add_argument("--eval-every", type=int, default=250_000)
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=20,
        help="Episodes per periodic EvalCallback evaluation.",
    )
    parser.add_argument(
        "--boundary-eval-episodes",
        type=int,
        default=0,
        help=(
            "Run this many deterministic episodes on the same evaluation "
            "seed sequence immediately before and after learning, writing "
            "paired JSON receipts. Zero disables boundary evaluation."
        ),
    )
    parser.add_argument(
        "--eval-num-envs",
        type=int,
        default=1,
        help=(
            "Parallel environments used by periodic and boundary evaluation. "
            "Values above one shorten evidence runs without changing native "
            "tick semantics."
        ),
    )
    parser.add_argument(
        "--also-evaluate-stochastic-boundaries",
        action="store_true",
        help=(
            "In addition to deterministic boundary evaluation, write paired "
            "stochastic-policy receipts using an isolated action RNG stream."
        ),
    )
    parser.add_argument(
        "--single-process",
        action="store_true",
        help="Use DummyVecEnv; useful for debugging and constrained platforms.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _planned_effective_steps(args: argparse.Namespace) -> int:
    """Return SB3's full-rollout timestep count for the requested run."""

    rollout_size = args.num_envs * args.rollout_steps
    return (
        (args.total_steps + rollout_size - 1) // rollout_size
    ) * rollout_size


def _validate_args(args: argparse.Namespace) -> None:
    """Validate safety and numeric arguments before importing SB3."""

    if args.transfer_training:
        if args.acknowledge_fidelity_gate:
            raise SystemExit(
                "--transfer-training cannot be combined with the closed-gate "
                "--acknowledge-fidelity-gate flag"
            )
        if args.fidelity_suite is None:
            raise SystemExit(
                "--transfer-training requires --fidelity-suite PATH"
            )
        if args.training_suite is None:
            raise SystemExit(
                "--transfer-training requires --training-suite PATH"
            )
        if args.environment != "revenge":
            raise SystemExit(
                "--transfer-training is only valid for the revenge "
                "environment"
            )
    elif not args.acknowledge_fidelity_gate:
        raise SystemExit(
            "Fidelity gate is CLOSED. Training is disabled by default because "
            "the current environment is diagnostic and policies are not yet "
            "expected to transfer to the original game. Re-run only for a "
            "diagnostic experiment with --acknowledge-fidelity-gate."
        )
    elif args.fidelity_suite is not None:
        raise SystemExit(
            "--fidelity-suite is only used with --transfer-training"
        )
    elif args.training_suite is not None:
        raise SystemExit(
            "--training-suite is only used with --transfer-training"
        )
    if (
        args.fidelity_suite_root is not None
        and args.fidelity_suite is None
    ):
        raise SystemExit(
            "--fidelity-suite-root requires --fidelity-suite"
        )
    if (
        args.training_suite_root is not None
        and args.training_suite is None
    ):
        raise SystemExit(
            "--training-suite-root requires --training-suite"
        )
    if args.environment == "simple" and not args.allow_reference_smoke:
        raise SystemExit(
            "ZumaSimple-v0 is a reference-only toy environment. To run its "
            "non-transferable smoke test, explicitly pass both "
            "--environment simple and --allow-reference-smoke."
        )
    if args.environment != "simple" and args.allow_reference_smoke:
        raise SystemExit(
            "--allow-reference-smoke is only valid with --environment simple"
        )
    if args.environment == "simple" and args.flat_actions:
        raise SystemExit(
            "--flat-actions only applies to the revenge environment"
        )
    if args.environment == "simple" and args.mask_rejected_actions:
        raise SystemExit(
            "--mask-rejected-actions only applies to the revenge environment"
        )
    if args.environment == "simple" and args.training_time_limit_steps:
        raise SystemExit(
            "--training-time-limit-steps only applies to the revenge "
            "environment"
        )
    if args.initial_model is not None:
        if args.environment != "revenge":
            raise SystemExit("--initial-model only supports the Revenge environment")
        if not args.mask_rejected_actions:
            raise SystemExit(
                "--initial-model requires --mask-rejected-actions"
            )
        if args.flat_actions:
            raise SystemExit("--initial-model requires factorized actions")
    if args.freeze_aim_path and args.initial_model is None:
        raise SystemExit("--freeze-aim-path requires --initial-model")
    if args.reset_optimizer_state and args.initial_model is None:
        raise SystemExit("--reset-optimizer-state requires --initial-model")
    if not math.isfinite(args.teacher_kl_coef) or args.teacher_kl_coef < 0.0:
        raise SystemExit("teacher-kl-coef must be finite and non-negative")
    if args.teacher_kl_coef > 0.0:
        if args.initial_model is None:
            raise SystemExit("--teacher-kl-coef requires --initial-model")
        if not args.mask_rejected_actions:
            raise SystemExit(
                "--teacher-kl-coef requires --mask-rejected-actions"
            )
        if not args.reset_optimizer_state:
            raise SystemExit(
                "--teacher-kl-coef requires --reset-optimizer-state"
            )
        if args.freeze_aim_path:
            raise SystemExit(
                "--teacher-kl-coef cannot be combined with "
                "--freeze-aim-path"
            )
    if args.transfer_training and args.training_time_limit_steps:
        raise SystemExit(
            "--training-time-limit-steps is a closed-gate diagnostic "
            "curriculum and cannot be used for transfer training"
        )
    if args.total_steps < 1:
        raise SystemExit("total-steps must be positive")
    if (
        not args.transfer_training
        and args.total_steps > CLOSED_GATE_MAX_EFFECTIVE_STEPS
    ):
        raise SystemExit(
            "Fidelity gate is CLOSED; diagnostic runs are capped at "
            f"{CLOSED_GATE_MAX_EFFECTIVE_STEPS:,} effective timesteps"
        )
    if args.num_envs < 1:
        raise SystemExit("num-envs must be positive")
    if args.rollout_steps < 2:
        raise SystemExit("rollout-steps must be at least 2")
    if not 1 <= args.ppo_epochs <= 100:
        raise SystemExit("ppo-epochs must be in [1, 100]")
    rollout_size = args.num_envs * args.rollout_steps
    if not 1 < args.batch_size <= rollout_size:
        raise SystemExit(
            f"batch-size must be in [2, {rollout_size}] for this rollout"
        )
    effective_steps = _planned_effective_steps(args)
    if (
        not args.transfer_training
        and effective_steps > CLOSED_GATE_MAX_EFFECTIVE_STEPS
    ):
        raise SystemExit(
            "Fidelity gate is CLOSED; the requested run expands to "
            f"{effective_steps:,} timesteps because PPO collects complete "
            "rollouts. Diagnostic runs are capped at "
            f"{CLOSED_GATE_MAX_EFFECTIVE_STEPS:,} effective timesteps"
        )
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0.0:
        raise SystemExit("learning-rate must be positive")
    if not math.isfinite(args.reward_scale) or args.reward_scale <= 0.0:
        raise SystemExit("reward-scale must be positive")
    if not math.isfinite(args.entropy_coef) or args.entropy_coef < 0.0:
        raise SystemExit("entropy-coef must be finite and non-negative")
    if args.checkpoint_every < 1:
        raise SystemExit("checkpoint-every must be positive")
    if args.eval_every < 1:
        raise SystemExit("eval-every must be positive")
    if args.eval_episodes < 1:
        raise SystemExit("eval-episodes must be positive")
    if args.boundary_eval_episodes < 0:
        raise SystemExit("boundary-eval-episodes cannot be negative")
    if args.eval_num_envs < 1:
        raise SystemExit("eval-num-envs must be positive")
    if args.eval_num_envs > args.eval_episodes:
        raise SystemExit("eval-num-envs cannot exceed eval-episodes")
    if (
        args.boundary_eval_episodes
        and args.eval_num_envs > args.boundary_eval_episodes
    ):
        raise SystemExit(
            "eval-num-envs cannot exceed boundary-eval-episodes when "
            "boundary evaluation is enabled"
        )
    if (
        args.also_evaluate_stochastic_boundaries
        and not args.boundary_eval_episodes
    ):
        raise SystemExit(
            "--also-evaluate-stochastic-boundaries requires a positive "
            "--boundary-eval-episodes"
        )
    if args.aim_bins < 16:
        raise SystemExit("aim-bins must be at least 16")
    if args.frame_skip < 1:
        raise SystemExit("frame-skip must be positive")
    if args.max_ticks < 1:
        raise SystemExit("max-ticks must be positive")
    if args.training_time_limit_steps < 0:
        raise SystemExit("training-time-limit-steps cannot be negative")
    if args.training_time_limit_steps >= args.max_ticks:
        raise SystemExit(
            "training-time-limit-steps must be zero or shorter than max-ticks"
        )
    if args.max_balls < 160:
        raise SystemExit("max-balls must be at least 160")
    if args.curve_index < 0:
        raise SystemExit("curve-index cannot be negative")
    if args.simple_initial_balls < 1:
        raise SystemExit("simple-initial-balls must be positive")
    if args.simple_max_steps < 1:
        raise SystemExit("simple-max-steps must be positive")


def _resolve_original_root(explicit: Path | None) -> Path:
    """Resolve and strictly validate an explicitly supplied game directory."""

    if explicit is not None:
        root = explicit.expanduser().resolve()
        required = (
            root / "ZumasRevenge.exe",
            root / "main.pak",
            root / "levels",
        )
        if not (
            required[0].is_file()
            and required[1].is_file()
            and required[2].is_dir()
        ):
            raise SystemExit(
                "The --original-root directory is not a usable Zuma's Revenge "
                f"installation: {root}\n"
                "Expected ZumasRevenge.exe, main.pak, and levels/."
            )
        return root
    try:
        return find_original_installation()
    except OriginalDataError as error:
        raise SystemExit(
            "The original Zuma's Revenge installation is required for the "
            "fidelity-first diagnostic environment, but it was not found.\n"
            "Pass --original-root PATH or set ZUMA_REVENGE_ROOT to the "
            "directory containing ZumasRevenge.exe, main.pak, and levels/."
        ) from error


def _verify_transfer_gate(
    args: argparse.Namespace,
    *,
    original_root: Path,
) -> FidelityGateReport | None:
    """Open only the exact scope approved by a complete suite."""

    if not args.transfer_training:
        return None
    report = verify_fidelity_suite(
        args.fidelity_suite,
        suite_root=args.fidelity_suite_root,
        original_root=original_root,
    )
    if report.status is not FidelityGateStatus.OPEN:
        reasons = "; ".join(report.reasons[:5])
        if len(report.reasons) > 5:
            reasons += f"; and {len(report.reasons) - 5} more"
        raise SystemExit(
            "Transfer training refused: fidelity suite is "
            f"{report.status.value}. {reasons}"
        )
    if report.policy is None or report.policy not in GATE_POLICIES:
        raise SystemExit(
            "Transfer training refused: gate report has no supported policy"
        )
    policy = GATE_POLICIES[report.policy]
    if not policy.transfer_authorizing or policy.generation < 2:
        raise SystemExit(
            "Transfer training refused: fidelity policy "
            f"{policy.id!r} is historical evidence only and does not "
            "authorize transfer training"
        )
    mismatches: list[str] = []
    if args.level not in policy.levels:
        mismatches.append(
            f"level {args.level!r} is outside {policy.levels!r}"
        )
    if bool(args.hard) is not policy.hard:
        mismatches.append("difficulty differs from the approved scope")
    if args.curve_index != 0:
        mismatches.append("approved scope requires curve-index 0")
    if args.frame_skip != 1:
        mismatches.append("approved scope requires native frame-skip 1")
    if mismatches:
        raise SystemExit(
            "Transfer training refused: requested environment is outside "
            f"policy {policy.id!r}: {'; '.join(mismatches)}"
        )
    return report


def _training_request(
    args: argparse.Namespace,
    *,
    initial_model_sha256: str | None,
) -> dict[str, Any]:
    """Build the path-free request compared with the fixed training policy."""

    return {
        "environment": args.environment,
        "level": args.level,
        "hard": bool(args.hard),
        "curve_index": args.curve_index,
        "aim_bins": args.aim_bins,
        "flat_actions": bool(args.flat_actions),
        "mask_rejected_actions": bool(args.mask_rejected_actions),
        "frame_skip": args.frame_skip,
        "max_ticks": args.max_ticks,
        "training_time_limit_steps": args.training_time_limit_steps,
        "max_balls": args.max_balls,
        "initial_model_sha256": initial_model_sha256,
        "freeze_aim_path": bool(args.freeze_aim_path),
        "reset_optimizer_state": bool(args.reset_optimizer_state),
        "teacher_kl_coef": args.teacher_kl_coef,
        "total_steps": args.total_steps,
        "planned_effective_steps": _planned_effective_steps(args),
        "num_envs": args.num_envs,
        "seed": args.seed,
        "device": args.device,
        "rollout_steps": args.rollout_steps,
        "batch_size": args.batch_size,
        "ppo_epochs": args.ppo_epochs,
        "learning_rate": args.learning_rate,
        "reward_scale": args.reward_scale,
        "entropy_coef": args.entropy_coef,
        "checkpoint_every": args.checkpoint_every,
        "eval_every": args.eval_every,
        "eval_episodes": args.eval_episodes,
        "boundary_eval_episodes": args.boundary_eval_episodes,
        "eval_num_envs": args.eval_num_envs,
        "also_evaluate_stochastic_boundaries": bool(
            args.also_evaluate_stochastic_boundaries
        ),
        "single_process": bool(args.single_process),
    }


def _verify_training_gate(
    args: argparse.Namespace,
    *,
    fidelity_report: FidelityGateReport | None,
    original_root: Path,
) -> TrainingGateReport | None:
    """Authorize only the exact bounded recipe supported by prior evidence."""

    if not args.transfer_training:
        return None
    report = verify_training_suite(
        args.training_suite,
        suite_root=args.training_suite_root,
        original_root=original_root,
    )
    if report.status is not TrainingGateStatus.OPEN:
        reasons = "; ".join(report.reasons[:5])
        if len(report.reasons) > 5:
            reasons += f"; and {len(report.reasons) - 5} more"
        raise SystemExit(
            "Transfer training refused: training suite is "
            f"{report.status.value}. {reasons}"
        )
    required_fidelity_policy = report.summary.get(
        "required_fidelity_policy"
    )
    if (
        report.policy != TRANSFER_POLICY_ID
        or report.stage != TRANSFER_STAGE_ID
        or report.summary.get("transfer_authorizing_policy") is not True
    ):
        raise SystemExit(
            "Transfer training refused: Training Gate policy is historical "
            "and cannot authorize a new run"
        )
    if (
        fidelity_report is None
        or not isinstance(required_fidelity_policy, str)
        or fidelity_report.policy != required_fidelity_policy
    ):
        raise SystemExit(
            "Transfer training refused: Training Gate requires fidelity "
            f"policy {required_fidelity_policy!r}"
        )
    fidelity_suite_path = Path(args.fidelity_suite).resolve(strict=True)
    fidelity_suite_sha256 = "sha256:" + hashlib.sha256(
        fidelity_suite_path.read_bytes()
    ).hexdigest()
    if (
        report.summary.get("required_fidelity_suite_sha256")
        != fidelity_suite_sha256
    ):
        raise SystemExit(
            "Transfer training refused: Training Gate binds a different "
            "Fidelity suite"
        )
    initial_model_sha256: str | None = None
    if args.initial_model is not None and args.initial_model.is_file():
        initial_model_sha256 = (
            "sha256:" + hashlib.sha256(args.initial_model.read_bytes()).hexdigest()
        )
    mismatches = training_request_mismatches(
        _training_request(
            args,
            initial_model_sha256=initial_model_sha256,
        ),
        expected=report.recipe,
    )
    if mismatches:
        details = "; ".join(mismatches[:8])
        if len(mismatches) > 8:
            details += f"; and {len(mismatches) - 8} more"
        raise SystemExit(
            "Transfer training refused: request differs from the fixed "
            f"Training Gate recipe: {details}"
        )
    return report


def _verify_new_transfer_run_directory(args: argparse.Namespace) -> None:
    """Prevent a certified run from overwriting or mixing prior artifacts."""

    if not args.transfer_training:
        return
    run_dir = args.run_dir.expanduser().resolve()
    if run_dir.exists():
        raise SystemExit(
            "Transfer training refused: --run-dir must not already exist: "
            f"{run_dir}"
        )


def _environment_spec(
    args: argparse.Namespace,
) -> tuple[type[RevengeEnv] | type[ZumaEnv], dict[str, Any], dict[str, Any]]:
    """Return the environment class, constructor kwargs, and serializable spec."""

    if args.environment == "simple":
        config = ZumaConfig(
            aim_bins=args.aim_bins,
            initial_balls=args.simple_initial_balls,
            max_steps=args.simple_max_steps,
        )
        return (
            ZumaEnv,
            {"config": config},
            {
                "id": "ZumaSimple-v0",
                "role": "reference_smoke_only",
                "config": config.to_dict(),
            },
        )

    root = _resolve_original_root(args.original_root)
    config = RevengeEnvConfig(
        aim_bins=args.aim_bins,
        action_mode="flat" if args.flat_actions else "factorized",
        frame_skip=args.frame_skip,
        max_ticks=args.max_ticks,
        max_balls=args.max_balls,
        score_reward_scale=args.reward_scale,
        step_penalty=-0.01 * args.reward_scale,
        win_reward=1_000.0 * args.reward_scale,
        loss_reward=-1_000.0 * args.reward_scale,
    )
    kwargs: dict[str, Any] = {
        "config": config,
        "level_id": args.level,
        "root": root,
        "hard": args.hard,
        "curve_index": args.curve_index,
        "profile_mode": SUPPORTED_PROFILE_MODE,
    }

    # Fail in the parent process with a concise error instead of letting every
    # vector worker fail independently after SB3 is imported.
    try:
        probe = RevengeEnv(**kwargs)
    except (OriginalDataError, IndexError, NotImplementedError, ValueError) as error:
        raise SystemExit(
            "Could not construct the fidelity-first diagnostic environment "
            f"for level {args.level!r}: {error}"
        ) from error
    else:
        structure_spec = {
            "shared_board": True,
            "curve_count": probe.sim.curve_count,
            "simulated_curve_indices": list(range(probe.sim.curve_count)),
            "curve_end_waypoints": [
                int(curve.end_waypoint) for curve in probe.sim.curves
            ],
            "curve_score_targets": [
                int(getattr(curve.parameters, "zuma_score", 0))
                for curve in probe.sim.curves
            ],
            "board_score_target": int(probe.sim.score_target),
            "num_colors": int(probe.sim.num_colors),
        }
        probe.close()

    return (
        RevengeEnv,
        kwargs,
        {
            "id": "ZumaRevenge-v0",
            "role": (
                "fidelity_first_transfer"
                if args.transfer_training
                else "fidelity_first_diagnostic"
            ),
            "action_filter": {
                "enabled": bool(args.mask_rejected_actions),
                "kind": (
                    "actor_observable_retail_rejection_mask"
                    if args.mask_rejected_actions
                    else "none"
                ),
                "simulator_semantics_changed": False,
            },
            "training_episode_boundary": {
                "enabled": bool(args.training_time_limit_steps),
                "kind": (
                    "gymnasium_time_limit"
                    if args.training_time_limit_steps
                    else "none"
                ),
                "max_episode_steps": (
                    int(args.training_time_limit_steps)
                    if args.training_time_limit_steps
                    else None
                ),
                "applies_to_evaluation": False,
                "native_dynamics_changed": False,
                "observation_changed": False,
            },
            "level": args.level,
            "hard": bool(args.hard),
            "requested_curve_index": int(args.curve_index),
            "profile_mode": SUPPORTED_PROFILE_MODE,
            "tutorial_contract": {
                "tutorials_completed_required": True,
                "fresh_profile_supported": False,
            },
            "level_structure": structure_spec,
            "original_root": str(root),
            "config": asdict(config),
        },
    )


def _run_config(
    args: argparse.Namespace,
    environment_spec: dict[str, Any],
    run_dir: Path,
    training_runtime: dict[str, Any] | None = None,
    gate_report: FidelityGateReport | None = None,
    training_gate_report: TrainingGateReport | None = None,
) -> dict[str, Any]:
    training = vars(args).copy()
    training["run_dir"] = str(run_dir)
    if training["original_root"] is not None:
        training["original_root"] = str(training["original_root"])
    for name in (
        "fidelity_suite",
        "fidelity_suite_root",
        "training_suite",
        "training_suite_root",
    ):
        if training[name] is not None:
            training[name] = str(training[name])
    initial_model_identity: dict[str, Any] | None = None
    if training["initial_model"] is not None:
        initial_model_path = Path(training["initial_model"]).resolve()
        training["initial_model"] = str(initial_model_path)
        initial_model_identity = {
            "path": str(initial_model_path),
            "sha256": (
                "sha256:"
                + hashlib.sha256(initial_model_path.read_bytes()).hexdigest()
            ),
        }
    suite_identity: dict[str, Any] | None = None
    if gate_report is not None:
        suite_path = Path(args.fidelity_suite).resolve()
        suite_identity = {
            "path": str(suite_path),
            "sha256": (
                "sha256:"
                + hashlib.sha256(suite_path.read_bytes()).hexdigest()
            ),
            "report": gate_report.to_dict(),
        }
    training_suite_identity: dict[str, Any] | None = None
    if training_gate_report is not None:
        training_suite_path = Path(args.training_suite).resolve()
        training_suite_identity = {
            "path": str(training_suite_path),
            "sha256": (
                "sha256:"
                + hashlib.sha256(training_suite_path.read_bytes()).hexdigest()
            ),
            "report": training_gate_report.to_dict(),
        }
    transfer_authorized = (
        gate_report is not None and training_gate_report is not None
    )
    return {
        "algorithm": (
            "teacher_anchored_maskable_ppo"
            if args.teacher_kl_coef > 0.0
            else (
                "maskable_ppo" if args.mask_rejected_actions else "ppo"
            )
        ),
        "diagnostic_non_transferable": not transfer_authorized,
        "fidelity_gate": (
            "open" if gate_report is not None else FIDELITY_GATE
        ),
        "fidelity_gate_acknowledged": bool(args.acknowledge_fidelity_gate),
        "training_gate": (
            "open" if training_gate_report is not None else "closed"
        ),
        "closed_gate_max_effective_steps": (
            None
            if transfer_authorized
            else CLOSED_GATE_MAX_EFFECTIVE_STEPS
        ),
        "fidelity_suite": suite_identity,
        "training_suite": training_suite_identity,
        "transfer_scope": (
            "bounded_state_policy_bootstrap"
            if transfer_authorized
            else "diagnostic_only"
        ),
        "visual_policy_training_authorized": False,
        "original_game_deployment_authorized": False,
        "initial_model": initial_model_identity,
        "planned_effective_steps": _planned_effective_steps(args),
        "environment": environment_spec,
        "runtime": training_runtime or {},
        "training": training,
    }


def _training_wrapper_class(args: argparse.Namespace) -> Any | None:
    """Return the diagnostic training-only episode wrapper, if requested."""

    if not args.training_time_limit_steps:
        return None
    return partial(
        TimeLimit,
        max_episode_steps=int(args.training_time_limit_steps),
    )


def _load_training_dependencies() -> dict[str, Any]:
    try:
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.maskable.callbacks import (
            MaskableEvalCallback,
        )
        from sb3_contrib.common.maskable.evaluation import (
            evaluate_policy as evaluate_maskable_policy,
        )
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import (
            CheckpointCallback,
            EvalCallback,
        )
        from stable_baselines3.common.evaluation import evaluate_policy
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
        from zuma_rl.teacher_anchor import (
            make_teacher_anchored_maskable_ppo,
        )
    except ImportError as error:
        raise SystemExit(
            "Training dependencies are missing. Install them with:\n"
            "  python -m pip install -e '.[train]'"
        ) from error
    TeacherAnchoredMaskablePPO = make_teacher_anchored_maskable_ppo(
        MaskablePPO
    )
    return {
        "PPO": PPO,
        "MaskablePPO": MaskablePPO,
        "TeacherAnchoredMaskablePPO": TeacherAnchoredMaskablePPO,
        "CheckpointCallback": CheckpointCallback,
        "EvalCallback": EvalCallback,
        "MaskableEvalCallback": MaskableEvalCallback,
        "evaluate_policy": evaluate_policy,
        "evaluate_maskable_policy": evaluate_maskable_policy,
        "make_vec_env": make_vec_env,
        "DummyVecEnv": DummyVecEnv,
        "SubprocVecEnv": SubprocVecEnv,
    }


def _run_boundary_evaluation(
    *,
    phase: str,
    output_path: Path,
    model: Any,
    eval_env: Any,
    evaluate_policy: Any,
    episode_count: int,
    seed: int,
    deterministic: bool = True,
    action_sampling_seed: int | None = None,
) -> dict[str, Any]:
    """Evaluate one model boundary on a repeatable episode-seed sequence."""

    if phase not in {"pretrain", "posttrain"}:
        raise ValueError("boundary evaluation phase is invalid")
    if episode_count < 1:
        raise ValueError("boundary evaluation episode count must be positive")
    if output_path.exists():
        raise RuntimeError(f"boundary evaluation output exists: {output_path}")
    if deterministic and action_sampling_seed is not None:
        raise ValueError(
            "deterministic evaluation cannot use an action sampling seed"
        )
    if not deterministic and action_sampling_seed is None:
        raise ValueError(
            "stochastic evaluation requires an action sampling seed"
        )

    episode_identities: list[dict[str, Any]] = []

    def capture_episode_identity(
        evaluator_locals: dict[str, Any],
        _evaluator_globals: dict[str, Any],
    ) -> None:
        """Mirror the evaluator's append order with a stable pairing key."""

        if not bool(evaluator_locals["done"]):
            return
        info = evaluator_locals["info"]
        if (
            bool(evaluator_locals["is_monitor_wrapped"])
            and "episode" not in info
        ):
            return
        vector_env_index = int(evaluator_locals["i"])
        episode_ordinal = int(
            evaluator_locals["episode_counts"][vector_env_index]
        )
        terminal_outcome = info.get("outcome")
        if terminal_outcome not in {"win", "loss", None}:
            raise RuntimeError("boundary terminal outcome is invalid")
        if not isinstance(info.get("score"), int):
            raise RuntimeError("boundary terminal score is missing")
        if not isinstance(info.get("ticks"), int):
            raise RuntimeError("boundary terminal tick is missing")
        time_limit_truncated = bool(info.get("TimeLimit.truncated", False))
        if time_limit_truncated == (terminal_outcome is not None):
            raise RuntimeError("boundary terminal outcome is inconsistent")
        episode_identities.append(
            {
                "pairing_key": (
                    f"env-{vector_env_index}:episode-{episode_ordinal}"
                ),
                "vector_env_index": vector_env_index,
                "episode_ordinal_within_env": episode_ordinal,
                "initial_env_seed": int(seed + vector_env_index),
                "terminal_outcome": terminal_outcome,
                "native_outcome": info.get("native_outcome"),
                "terminal_score": int(info["score"]),
                "terminal_ticks": int(info["ticks"]),
                "time_limit_truncated": time_limit_truncated,
            }
        )

    eval_env.seed(seed)
    if deterministic:
        rewards, lengths = evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=episode_count,
            deterministic=True,
            return_episode_rewards=True,
            warn=False,
            callback=capture_episode_identity,
        )
    else:
        # Stochastic boundary evaluation must not perturb the global Torch RNG
        # used by the subsequent training run.  Preserve both CPU and every
        # visible CUDA generator, then replay the same isolated action stream
        # at the pre- and post-training boundaries.
        try:
            import torch
        except ImportError as error:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "Torch is required for stochastic boundary evaluation"
            ) from error
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_states = (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else []
        )
        try:
            torch.manual_seed(int(action_sampling_seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(action_sampling_seed))
            rewards, lengths = evaluate_policy(
                model,
                eval_env,
                n_eval_episodes=episode_count,
                deterministic=False,
                return_episode_rewards=True,
                warn=False,
                callback=capture_episode_identity,
            )
        finally:
            torch.set_rng_state(cpu_rng_state)
            if cuda_rng_states:
                torch.cuda.set_rng_state_all(cuda_rng_states)
    reward_values = [float(value) for value in rewards]
    length_values = [int(value) for value in lengths]
    if (
        len(reward_values) != episode_count
        or len(length_values) != episode_count
        or len(episode_identities) != episode_count
    ):
        raise RuntimeError("boundary evaluation episode count mismatch")
    pairing_keys = [
        str(identity["pairing_key"]) for identity in episode_identities
    ]
    if len(set(pairing_keys)) != episode_count:
        raise RuntimeError("boundary evaluation episode identity mismatch")
    mean_reward = math.fsum(reward_values) / episode_count
    reward_variance = math.fsum(
        (value - mean_reward) ** 2 for value in reward_values
    ) / episode_count
    mean_length = math.fsum(length_values) / episode_count
    payload = {
        "schema": "zuma-rl.ppo-boundary-evaluation",
        "version": 4,
        "phase": phase,
        "deterministic": bool(deterministic),
        "seed": seed,
        "seed_protocol": "vec_env_seed_once_continuous_resets",
        "action_sampling_seed": action_sampling_seed,
        "action_sampling_rng_isolated": not deterministic,
        "vector_env_count": int(getattr(eval_env, "num_envs", 1)),
        "episode_count": episode_count,
        "model_num_timesteps": int(getattr(model, "num_timesteps", 0)),
        "episode_identity_protocol": (
            "vector_env_index_per_env_episode_ordinal_and_terminal_info"
        ),
        "episode_identities": episode_identities,
        "episode_rewards": reward_values,
        "episode_lengths": length_values,
        "mean_reward": mean_reward,
        "reward_std": math.sqrt(reward_variance),
        "mean_episode_length": mean_length,
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def _tensor_sha256(tensor: Any) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return "sha256:" + hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _named_tensors_sha256(named_tensors: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(named_tensors.items()):
        encoded_name = name.encode("utf-8")
        array = tensor.detach().cpu().contiguous().numpy()
        encoded_shape = json.dumps(list(array.shape)).encode("ascii")
        encoded_dtype = str(array.dtype).encode("ascii")
        for value in (encoded_name, encoded_shape, encoded_dtype):
            digest.update(len(value).to_bytes(8, "big"))
            digest.update(value)
        raw = array.tobytes(order="C")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return "sha256:" + digest.hexdigest()


def _write_json_exclusive(output_path: Path, payload: dict[str, Any]) -> None:
    encoded = (
        json.dumps(payload, allow_nan=False, ensure_ascii=False, indent=2)
        + "\n"
    ).encode("utf-8")
    with output_path.open("xb") as stream:
        stream.write(encoded)


def _install_aim_path_freeze(
    *,
    model: Any,
    output_path: Path,
) -> dict[str, Any]:
    """Freeze actor features and aim rows, including loaded Adam momentum."""

    try:
        import torch
    except ImportError as error:  # pragma: no cover - dependency guard
        raise RuntimeError("Torch is required for aim-path freezing") from error

    action_sizes = [int(value) for value in model.action_space.nvec]
    if len(action_sizes) != 2 or action_sizes[0] < 1 or action_sizes[1] < 1:
        raise RuntimeError("aim-path freezing requires [verb, aim] actions")
    verb_bins, aim_bins = action_sizes
    action_net = model.policy.action_net
    if (
        action_net.weight.ndim != 2
        or action_net.bias.ndim != 1
        or action_net.weight.shape[0] != verb_bins + aim_bins
        or action_net.bias.shape[0] != verb_bins + aim_bins
    ):
        raise RuntimeError("factorized action-head shape is incompatible")

    feature_parameters = {
        name: parameter
        for name, parameter in model.policy.features_extractor.named_parameters()
    }
    if not feature_parameters:
        raise RuntimeError("aim-path freezing found no feature parameters")
    feature_snapshots = {
        name: parameter.detach().clone()
        for name, parameter in feature_parameters.items()
    }
    aim_weight_snapshot = action_net.weight[verb_bins:].detach().clone()
    aim_bias_snapshot = action_net.bias[verb_bins:].detach().clone()

    optimizer_state_tensors_cleared: list[str] = []
    with torch.no_grad():
        for name, parameter in feature_parameters.items():
            parameter.requires_grad_(False)
            for state_name, value in model.policy.optimizer.state.get(
                parameter, {}
            ).items():
                if torch.is_tensor(value) and value.shape == parameter.shape:
                    value.zero_()
                    optimizer_state_tensors_cleared.append(
                        f"features_extractor.{name}:{state_name}"
                    )
        for parameter_name, parameter in (
            ("action_net.weight", action_net.weight),
            ("action_net.bias", action_net.bias),
        ):
            for state_name, value in model.policy.optimizer.state.get(
                parameter, {}
            ).items():
                if torch.is_tensor(value) and value.shape == parameter.shape:
                    value[verb_bins:].zero_()
                    optimizer_state_tensors_cleared.append(
                        f"{parameter_name}:{state_name}"
                    )

    def zero_aim_rows(gradient: Any) -> Any:
        frozen_gradient = gradient.clone()
        frozen_gradient[verb_bins:].zero_()
        return frozen_gradient

    hook_handles = [
        action_net.weight.register_hook(zero_aim_rows),
        action_net.bias.register_hook(zero_aim_rows),
    ]
    payload = {
        "schema": "zuma-rl.ppo-aim-path-freeze",
        "version": 1,
        "status": "INSTALLED",
        "action_space_nvec": action_sizes,
        "verb_bins_trainable": verb_bins,
        "aim_bins_frozen": aim_bins,
        "feature_parameter_count_frozen": sum(
            parameter.numel() for parameter in feature_parameters.values()
        ),
        "feature_tensor_count_frozen": len(feature_parameters),
        "initial_hashes": {
            "feature_parameters": _named_tensors_sha256(
                feature_snapshots
            ),
            "aim_weight_rows": _tensor_sha256(aim_weight_snapshot),
            "aim_bias_rows": _tensor_sha256(aim_bias_snapshot),
        },
        "optimizer_state_tensors_cleared": sorted(
            optimizer_state_tensors_cleared
        ),
        "guarantee": (
            "feature parameters use requires_grad=False; aim output rows use "
            "gradient masking after their loaded Adam state is zeroed"
        ),
    }
    _write_json_exclusive(output_path, payload)
    return {
        "verb_bins": verb_bins,
        "feature_snapshots": feature_snapshots,
        "aim_weight_snapshot": aim_weight_snapshot,
        "aim_bias_snapshot": aim_bias_snapshot,
        "hook_handles": hook_handles,
    }


def _reset_loaded_optimizer_state(
    *,
    model: Any,
    output_path: Path,
) -> dict[str, Any]:
    """Drop loaded optimizer moments without changing any model parameter."""

    try:
        import torch
    except ImportError as error:  # pragma: no cover - dependency guard
        raise RuntimeError("Torch is required for optimizer reset") from error

    named_parameters = dict(model.policy.named_parameters())
    parameter_hash_before = _named_tensors_sha256(named_parameters)
    optimizer_state = model.policy.optimizer.state
    state_parameter_count = len(optimizer_state)
    state_tensor_count = 0
    state_element_count = 0
    exp_avg_squared_norm = 0.0
    max_step = 0.0
    for state in optimizer_state.values():
        for state_name, value in state.items():
            if not torch.is_tensor(value):
                continue
            state_tensor_count += 1
            state_element_count += value.numel()
            if state_name == "exp_avg":
                exp_avg_squared_norm += float(
                    torch.sum(value.detach().double().square()).item()
                )
            if state_name == "step" and value.numel() == 1:
                max_step = max(max_step, float(value.detach().cpu().item()))

    optimizer_state.clear()
    parameter_hash_after = _named_tensors_sha256(named_parameters)
    state_cleared = len(optimizer_state) == 0
    parameters_unchanged = parameter_hash_before == parameter_hash_after
    payload = {
        "schema": "zuma-rl.ppo-optimizer-state-reset",
        "version": 1,
        "status": (
            "PASS" if state_cleared and parameters_unchanged else "FAIL"
        ),
        "objective_transition": "supervised_teacher_to_on_policy_ppo",
        "state_before": {
            "parameter_count": state_parameter_count,
            "tensor_count": state_tensor_count,
            "element_count": state_element_count,
            "aggregate_exp_avg_norm": math.sqrt(exp_avg_squared_norm),
            "max_step": max_step,
        },
        "state_after": {"parameter_count": len(optimizer_state)},
        "parameters": {
            "sha256_before": parameter_hash_before,
            "sha256_after": parameter_hash_after,
            "bitwise_unchanged": parameters_unchanged,
        },
    }
    _write_json_exclusive(output_path, payload)
    if not state_cleared or not parameters_unchanged:
        raise RuntimeError("optimizer-state reset verification failed")
    return payload


def _install_teacher_policy_anchor(
    *,
    model: Any,
    coefficient: float,
    source_model_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Clone and freeze the initial policy used by the PPO KL anchor."""

    import copy

    try:
        import torch
    except ImportError as error:  # pragma: no cover - dependency guard
        raise RuntimeError("Torch is required for teacher anchoring") from error

    if coefficient <= 0.0 or not math.isfinite(coefficient):
        raise RuntimeError("teacher anchor coefficient must be positive")
    if not bool(
        getattr(model, "_supports_teacher_policy_anchor", False)
    ):
        raise RuntimeError("model does not support teacher policy anchoring")
    if not source_model_path.is_file():
        raise RuntimeError("teacher anchor source model is missing")

    current_parameters = dict(model.policy.named_parameters())
    current_hash = _named_tensors_sha256(current_parameters)
    teacher_policy = copy.deepcopy(model.policy)
    teacher_policy.set_training_mode(False)
    for parameter in teacher_policy.parameters():
        parameter.requires_grad_(False)
    teacher_parameters = dict(teacher_policy.named_parameters())
    teacher_hash = _named_tensors_sha256(teacher_parameters)
    all_frozen = all(
        not parameter.requires_grad
        for parameter in teacher_policy.parameters()
    )
    hashes_match = current_hash == teacher_hash

    model.teacher_kl_coef = float(coefficient)
    model._teacher_policy = teacher_policy
    model._teacher_anchor_sample_count = 0
    model._teacher_anchor_kl_sum = 0.0
    model._teacher_anchor_kl_max = 0.0
    model._teacher_anchor_minibatch_count = 0
    model._teacher_anchor_train_calls = 0

    source_hash = "sha256:" + hashlib.sha256(
        source_model_path.read_bytes()
    ).hexdigest()
    installed = hashes_match and all_frozen
    payload = {
        "schema": "zuma-rl.ppo-teacher-policy-anchor",
        "version": 1,
        "status": "INSTALLED" if installed else "FAIL",
        "direction": "forward_kl_initial_teacher_to_current_policy",
        "state_source": "current_rollout_observations",
        "action_masks_applied_to_both_policies": True,
        "coefficient": float(coefficient),
        "source_model": {
            "path": str(source_model_path),
            "sha256": source_hash,
        },
        "initial_policy": {
            "parameter_count": sum(
                parameter.numel()
                for parameter in current_parameters.values()
            ),
            "tensor_count": len(current_parameters),
            "current_sha256": current_hash,
            "teacher_sha256": teacher_hash,
            "bitwise_equal": hashes_match,
            "teacher_requires_grad_all_false": all_frozen,
        },
        "optimizer_includes_teacher_parameters": any(
            parameter in model.policy.optimizer.state
            for parameter in teacher_policy.parameters()
        ),
    }
    _write_json_exclusive(output_path, payload)
    if not installed or payload["optimizer_includes_teacher_parameters"]:
        raise RuntimeError("teacher policy anchor installation failed")
    return {
        "initial_policy_sha256": current_hash,
        "teacher_policy_sha256": teacher_hash,
        "coefficient": float(coefficient),
    }


def _verify_teacher_policy_anchor(
    *,
    model: Any,
    anchor_state: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    """Prove that the teacher stayed immutable and anchoring was exercised."""

    teacher_policy = getattr(model, "_teacher_policy", None)
    if teacher_policy is None:
        raise RuntimeError("teacher policy anchor disappeared before verification")
    teacher_parameters = dict(teacher_policy.named_parameters())
    teacher_hash = _named_tensors_sha256(teacher_parameters)
    teacher_unchanged = (
        teacher_hash == anchor_state["teacher_policy_sha256"]
    )
    all_frozen = all(
        not parameter.requires_grad
        for parameter in teacher_policy.parameters()
    )
    sample_count = int(model._teacher_anchor_sample_count)
    minibatch_count = int(model._teacher_anchor_minibatch_count)
    train_calls = int(model._teacher_anchor_train_calls)
    kl_sum = float(model._teacher_anchor_kl_sum)
    kl_max = float(model._teacher_anchor_kl_max)
    mean_kl = kl_sum / sample_count if sample_count else math.nan
    current_hash = _named_tensors_sha256(
        dict(model.policy.named_parameters())
    )
    current_changed = current_hash != anchor_state["initial_policy_sha256"]
    exercised = (
        sample_count > 0
        and minibatch_count > 0
        and train_calls > 0
        and math.isfinite(mean_kl)
        and math.isfinite(kl_max)
        and mean_kl >= 0.0
        and kl_max >= 0.0
    )
    passed = teacher_unchanged and all_frozen and exercised and current_changed
    payload = {
        "schema": "zuma-rl.ppo-teacher-policy-anchor-verification",
        "version": 1,
        "status": "PASS" if passed else "FAIL",
        "coefficient": float(anchor_state["coefficient"]),
        "teacher_policy": {
            "initial_sha256": anchor_state["teacher_policy_sha256"],
            "final_sha256": teacher_hash,
            "bitwise_unchanged": teacher_unchanged,
            "requires_grad_all_false": all_frozen,
        },
        "current_policy": {
            "initial_sha256": anchor_state["initial_policy_sha256"],
            "final_sha256": current_hash,
            "bitwise_changed": current_changed,
        },
        "training": {
            "train_calls": train_calls,
            "minibatch_count": minibatch_count,
            "sample_count": sample_count,
            "mean_forward_kl": mean_kl,
            "max_sample_forward_kl": kl_max,
        },
        "teacher_policy_detached_before_model_save": True,
    }
    _write_json_exclusive(output_path, payload)
    model._teacher_policy = None
    if not passed:
        raise RuntimeError("teacher policy anchor verification failed")
    return payload


def _verify_aim_path_freeze(
    *,
    model: Any,
    freeze_state: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    """Write a bitwise post-training receipt and fail on any frozen drift."""

    try:
        import torch
    except ImportError as error:  # pragma: no cover - dependency guard
        raise RuntimeError("Torch is required for aim-path verification") from error

    verb_bins = int(freeze_state["verb_bins"])
    current_features = {
        name: parameter.detach()
        for name, parameter in model.policy.features_extractor.named_parameters()
    }
    feature_snapshots = freeze_state["feature_snapshots"]
    changed_features = sorted(
        name
        for name, snapshot in feature_snapshots.items()
        if name not in current_features
        or not torch.equal(snapshot, current_features[name])
    )
    action_net = model.policy.action_net
    aim_weight_unchanged = torch.equal(
        freeze_state["aim_weight_snapshot"],
        action_net.weight[verb_bins:],
    )
    aim_bias_unchanged = torch.equal(
        freeze_state["aim_bias_snapshot"],
        action_net.bias[verb_bins:],
    )
    bitwise_unchanged = (
        not changed_features
        and aim_weight_unchanged
        and aim_bias_unchanged
    )
    payload = {
        "schema": "zuma-rl.ppo-aim-path-freeze-verification",
        "version": 1,
        "status": "PASS" if bitwise_unchanged else "FAIL",
        "bitwise_unchanged": bitwise_unchanged,
        "changed_feature_parameters": changed_features,
        "aim_weight_rows_unchanged": aim_weight_unchanged,
        "aim_bias_rows_unchanged": aim_bias_unchanged,
        "final_hashes": {
            "feature_parameters": _named_tensors_sha256(current_features),
            "aim_weight_rows": _tensor_sha256(
                action_net.weight[verb_bins:]
            ),
            "aim_bias_rows": _tensor_sha256(action_net.bias[verb_bins:]),
        },
    }
    _write_json_exclusive(output_path, payload)
    if not bitwise_unchanged:
        raise RuntimeError("aim-path freeze verification failed")
    return payload


def _force_close_vector_env(env: Any) -> None:
    """Bound cleanup after a subprocess worker has already failed."""

    processes = tuple(getattr(env, "processes", ()))
    remotes = tuple(getattr(env, "remotes", ()))
    if not processes:
        try:
            env.close()
        except BaseException:
            pass
        return

    for remote in remotes:
        try:
            remote.close()
        except BaseException:
            pass
    for process in processes:
        try:
            if process.is_alive():
                process.terminate()
        except BaseException:
            pass
    for process in processes:
        try:
            process.join(timeout=5.0)
        except BaseException:
            pass
    for process in processes:
        try:
            if process.is_alive():
                process.kill()
                process.join(timeout=2.0)
        except BaseException:
            pass
    if hasattr(env, "waiting"):
        env.waiting = False
    if hasattr(env, "closed"):
        env.closed = True


def _write_training_failure_receipt(
    *,
    output_path: Path,
    error: BaseException,
    model_num_timesteps: int,
    emergency_model_saved: bool,
) -> dict[str, Any]:
    """Persist the parent-visible failure without masking the exception."""

    payload = {
        "schema": "zuma-rl.ppo-training-failure",
        "version": 1,
        "status": "ABORTED",
        "exception_type": type(error).__name__,
        "exception": str(error),
        "model_num_timesteps": int(model_num_timesteps),
        "emergency_model_saved": bool(emergency_model_saved),
    }
    if not output_path.exists():
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return payload


def _artifact_receipt(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"required completion artifact is missing: {path}")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "bytes": path.stat().st_size,
        "sha256": "sha256:" + digest,
    }


def _write_training_completion_receipt(
    *,
    output_path: Path,
    run_dir: Path,
    model_num_timesteps: int,
    expected_effective_steps: int,
    algorithm: str,
    boundary_evaluation_enabled: bool,
    stochastic_boundary_evaluation_enabled: bool,
    aim_path_freeze_enabled: bool = False,
    optimizer_state_reset_enabled: bool = False,
    teacher_policy_anchor_enabled: bool = False,
) -> dict[str, Any]:
    """Write the final in-process success receipt after vector cleanup.

    This is intentionally the last fallible operation in ``main``.  Missing
    artifacts, a timestep mismatch, vector-close failures, or an existing
    receipt all prevent a false COMPLETE claim.
    """

    if output_path.exists():
        raise RuntimeError(f"completion receipt already exists: {output_path}")
    if model_num_timesteps != expected_effective_steps:
        raise RuntimeError(
            "completion timestep mismatch: "
            f"{model_num_timesteps} != {expected_effective_steps}"
        )
    if (run_dir / "failure.json").exists():
        raise RuntimeError("failure receipt exists at completion boundary")

    artifact_paths = {
        "config": run_dir / "config.json",
        "final_model": run_dir / "final_model.zip",
    }
    if boundary_evaluation_enabled:
        artifact_paths.update(
            {
                "posttrain_deterministic": (
                    run_dir / "posttrain_evaluation.json"
                ),
                "pretrain_deterministic": (
                    run_dir / "pretrain_evaluation.json"
                ),
            }
        )
    if stochastic_boundary_evaluation_enabled:
        artifact_paths.update(
            {
                "posttrain_stochastic": (
                    run_dir / "posttrain_evaluation_stochastic.json"
                ),
                "pretrain_stochastic": (
                    run_dir / "pretrain_evaluation_stochastic.json"
                ),
            }
        )
    if aim_path_freeze_enabled:
        artifact_paths.update(
            {
                "aim_path_freeze": run_dir / "aim_path_freeze.json",
                "aim_path_freeze_verification": (
                    run_dir / "aim_path_freeze_verification.json"
                ),
            }
        )
    if optimizer_state_reset_enabled:
        artifact_paths["optimizer_state_reset"] = (
            run_dir / "optimizer_state_reset.json"
        )
    if teacher_policy_anchor_enabled:
        artifact_paths.update(
            {
                "teacher_policy_anchor": (
                    run_dir / "teacher_policy_anchor.json"
                ),
                "teacher_policy_anchor_verification": (
                    run_dir / "teacher_policy_anchor_verification.json"
                ),
            }
        )
    artifacts = {
        name: _artifact_receipt(path)
        for name, path in sorted(artifact_paths.items())
    }
    payload = {
        "algorithm": algorithm,
        "artifacts": artifacts,
        "completed_utc": datetime.now(timezone.utc).isoformat(
            timespec="milliseconds"
        ).replace("+00:00", "Z"),
        "effective_steps": model_num_timesteps,
        "process_contract": {
            "completion_receipt_is_last_main_operation": True,
            "expected_process_exit_code": 0,
            "vector_env_close_status": "PASS",
        },
        "schema": "zuma-rl.ppo-training-completion",
        "status": "COMPLETE",
        "version": 1,
    }
    encoded = (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    with output_path.open("xb") as stream:
        stream.write(encoded)
    return payload


def _training_runtime() -> dict[str, Any]:
    """Return a path-free runtime manifest for reproducible diagnostics."""

    try:
        import torch
    except ImportError as error:  # pragma: no cover - guarded by dependency load
        raise SystemExit(
            "PyTorch is unavailable after dependency loading"
        ) from error

    package_names = (
        "zuma-rl",
        "numpy",
        "gymnasium",
        "stable-baselines3",
        "sb3-contrib",
        "torch",
        "tensorboard",
    )
    packages: dict[str, str] = {}
    for name in package_names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise SystemExit(
                f"Training runtime package metadata is missing: {name}"
            ) from error
    cuda_available = bool(torch.cuda.is_available())
    cuda_device_count = int(torch.cuda.device_count()) if cuda_available else 0
    cuda_device_names = [
        str(torch.cuda.get_device_name(index))
        for index in range(cuda_device_count)
    ]
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "torch_build_cuda": torch.version.cuda,
        "torch_cuda_available": cuda_available,
        "torch_cuda_device_count": cuda_device_count,
        "torch_cuda_device_names": cuda_device_names,
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": int(torch.get_num_interop_threads()),
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    _validate_args(args)
    if args.environment == "revenge":
        args.original_root = _resolve_original_root(args.original_root)
    if args.initial_model is not None:
        args.initial_model = args.initial_model.expanduser().resolve()
        if not args.initial_model.is_file():
            raise SystemExit(
                f"The --initial-model file does not exist: {args.initial_model}"
            )
    gate_report = _verify_transfer_gate(
        args,
        original_root=(
            Path(args.original_root)
            if args.original_root is not None
            else Path()
        ),
    )
    training_gate_report = _verify_training_gate(
        args,
        fidelity_report=gate_report,
        original_root=(
            Path(args.original_root)
            if args.original_root is not None
            else Path()
        ),
    )
    _verify_new_transfer_run_directory(args)
    env_class, env_kwargs, environment_spec = _environment_spec(args)
    dependencies = _load_training_dependencies()
    training_runtime = _training_runtime()

    if args.mask_rejected_actions:
        if args.teacher_kl_coef > 0.0:
            PPO = dependencies["TeacherAnchoredMaskablePPO"]
            algorithm_name = "teacher_anchored_maskable_ppo"
        else:
            PPO = dependencies["MaskablePPO"]
            algorithm_name = "maskable_ppo"
        EvalCallback = dependencies["MaskableEvalCallback"]
        evaluate_policy = dependencies["evaluate_maskable_policy"]
    else:
        PPO = dependencies["PPO"]
        EvalCallback = dependencies["EvalCallback"]
        evaluate_policy = dependencies.get("evaluate_policy")
        algorithm_name = "ppo"
    CheckpointCallback = dependencies["CheckpointCallback"]
    make_vec_env = dependencies["make_vec_env"]
    DummyVecEnv = dependencies["DummyVecEnv"]
    SubprocVecEnv = dependencies["SubprocVecEnv"]

    run_dir = args.run_dir.resolve()
    checkpoints = run_dir / "checkpoints"
    evaluations = run_dir / "evaluations"
    checkpoints.mkdir(parents=True, exist_ok=True)
    evaluations.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(
            _run_config(
                args,
                environment_spec,
                run_dir,
                training_runtime=training_runtime,
                gate_report=gate_report,
                training_gate_report=training_gate_report,
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if gate_report is None:
        print(
            "WARNING: fidelity gate is CLOSED; this PPO run is diagnostic "
            "and non-transferable."
        )
    else:
        print(
            "Fidelity and Training gates OPEN for policies "
            f"{gate_report.policy!r} and {training_gate_report.policy!r}; "
            "this run is limited to bounded state-policy calibration."
        )
    vector_class = (
        DummyVecEnv
        if args.single_process or args.num_envs == 1
        else SubprocVecEnv
    )
    training_wrapper = _training_wrapper_class(args)
    train_env_options: dict[str, Any] = {
        "n_envs": args.num_envs,
        "seed": args.seed,
        "env_kwargs": env_kwargs,
        "vec_env_cls": vector_class,
    }
    if training_wrapper is not None:
        train_env_options["wrapper_class"] = training_wrapper
    train_env = make_vec_env(env_class, **train_env_options)
    eval_env = make_vec_env(
        env_class,
        n_envs=args.eval_num_envs,
        seed=args.seed + 10_000,
        env_kwargs=env_kwargs,
        # EvalCallback compares vector wrapper types.  Matching the training
        # wrapper avoids a false mismatch warning; evaluation workers remain
        # isolated from the training workers.
        vec_env_cls=vector_class,
    )

    run_role = (
        "transfer"
        if gate_report is not None and training_gate_report is not None
        else "diagnostic"
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, args.checkpoint_every // args.num_envs),
        save_path=str(checkpoints),
        name_prefix=f"zuma_{algorithm_name}_{run_role}",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / "best"),
        log_path=str(evaluations),
        eval_freq=max(1, args.eval_every // args.num_envs),
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
    )

    if args.initial_model is not None:
        load_options: dict[str, Any] = {}
        if args.teacher_kl_coef > 0.0:
            load_options["teacher_kl_coef"] = args.teacher_kl_coef
        model = PPO.load(
            args.initial_model,
            env=train_env,
            learning_rate=args.learning_rate,
            n_steps=args.rollout_steps,
            batch_size=args.batch_size,
            n_epochs=args.ppo_epochs,
            ent_coef=args.entropy_coef,
            tensorboard_log=str(run_dir / "tensorboard"),
            seed=args.seed,
            device=args.device,
            verbose=1,
            force_reset=True,
            **load_options,
        )
    else:
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=args.learning_rate,
            n_steps=args.rollout_steps,
            batch_size=args.batch_size,
            n_epochs=args.ppo_epochs,
            gamma=0.995,
            gae_lambda=0.95,
            ent_coef=args.entropy_coef,
            policy_kwargs={"net_arch": [256, 256]},
            tensorboard_log=str(run_dir / "tensorboard"),
            seed=args.seed,
            device=args.device,
            verbose=1,
        )
    freeze_state: dict[str, Any] | None = None
    anchor_state: dict[str, Any] | None = None
    failed = False
    try:
        if args.reset_optimizer_state:
            _reset_loaded_optimizer_state(
                model=model,
                output_path=run_dir / "optimizer_state_reset.json",
            )
        if args.freeze_aim_path:
            freeze_state = _install_aim_path_freeze(
                model=model,
                output_path=run_dir / "aim_path_freeze.json",
            )
        if args.teacher_kl_coef > 0.0:
            anchor_state = _install_teacher_policy_anchor(
                model=model,
                coefficient=args.teacher_kl_coef,
                source_model_path=args.initial_model,
                output_path=run_dir / "teacher_policy_anchor.json",
            )
        if args.boundary_eval_episodes:
            _run_boundary_evaluation(
                phase="pretrain",
                output_path=run_dir / "pretrain_evaluation.json",
                model=model,
                eval_env=eval_env,
                evaluate_policy=evaluate_policy,
                episode_count=args.boundary_eval_episodes,
                seed=args.seed + 10_000,
            )
            if args.also_evaluate_stochastic_boundaries:
                _run_boundary_evaluation(
                    phase="pretrain",
                    output_path=(
                        run_dir / "pretrain_evaluation_stochastic.json"
                    ),
                    model=model,
                    eval_env=eval_env,
                    evaluate_policy=evaluate_policy,
                    episode_count=args.boundary_eval_episodes,
                    seed=args.seed + 10_000,
                    deterministic=False,
                    action_sampling_seed=args.seed + 20_000,
                )
        model.learn(
            total_timesteps=args.total_steps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True,
        )
        if freeze_state is not None:
            _verify_aim_path_freeze(
                model=model,
                freeze_state=freeze_state,
                output_path=run_dir / "aim_path_freeze_verification.json",
            )
        if anchor_state is not None:
            _verify_teacher_policy_anchor(
                model=model,
                anchor_state=anchor_state,
                output_path=(
                    run_dir / "teacher_policy_anchor_verification.json"
                ),
            )
        model.save(run_dir / "final_model")
        if args.boundary_eval_episodes:
            _run_boundary_evaluation(
                phase="posttrain",
                output_path=run_dir / "posttrain_evaluation.json",
                model=model,
                eval_env=eval_env,
                evaluate_policy=evaluate_policy,
                episode_count=args.boundary_eval_episodes,
                seed=args.seed + 10_000,
            )
            if args.also_evaluate_stochastic_boundaries:
                _run_boundary_evaluation(
                    phase="posttrain",
                    output_path=(
                        run_dir / "posttrain_evaluation_stochastic.json"
                    ),
                    model=model,
                    eval_env=eval_env,
                    evaluate_policy=evaluate_policy,
                    episode_count=args.boundary_eval_episodes,
                    seed=args.seed + 10_000,
                    deterministic=False,
                    action_sampling_seed=args.seed + 20_000,
                )
    except BaseException as error:
        failed = True
        if anchor_state is not None:
            model._teacher_policy = None
        emergency_model_saved = False
        model_num_timesteps = int(getattr(model, "num_timesteps", 0))
        if model_num_timesteps > 0:
            try:
                model.save(run_dir / "emergency_model")
                emergency_model_saved = True
            except BaseException:
                emergency_model_saved = False
        try:
            _write_training_failure_receipt(
                output_path=run_dir / "failure.json",
                error=error,
                model_num_timesteps=model_num_timesteps,
                emergency_model_saved=emergency_model_saved,
            )
        except BaseException:
            pass
        raise
    finally:
        if failed:
            _force_close_vector_env(train_env)
            _force_close_vector_env(eval_env)
        else:
            train_env.close()
            eval_env.close()
    _write_training_completion_receipt(
        output_path=run_dir / "completion.json",
        run_dir=run_dir,
        model_num_timesteps=int(getattr(model, "num_timesteps", 0)),
        expected_effective_steps=_planned_effective_steps(args),
        algorithm=algorithm_name,
        boundary_evaluation_enabled=bool(args.boundary_eval_episodes),
        stochastic_boundary_evaluation_enabled=bool(
            args.also_evaluate_stochastic_boundaries
        ),
        aim_path_freeze_enabled=bool(args.freeze_aim_path),
        optimizer_state_reset_enabled=bool(args.reset_optimizer_state),
        teacher_policy_anchor_enabled=bool(args.teacher_kl_coef > 0.0),
    )


if __name__ == "__main__":
    main()
