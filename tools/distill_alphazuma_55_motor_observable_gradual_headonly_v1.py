"""Run gradual motor DAgger while training only the policy action head.

The frozen fire-only recipe updates the complete shared policy and its early
checkpoints outperformed later checkpoints in the engineering Gate.  This
registered ablation keeps the environment, teacher, labels, DAgger schedule,
losses, batches, and checkpoints unchanged while preventing imitation
gradients from overwriting the source representation.  Only
``action_net.weight`` and ``action_net.bias`` are trainable.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tools import distill_alphazuma_55 as legacy
from tools import distill_alphazuma_55_motor_observable_gradual_v1 as gradual
from tools import distill_alphazuma_55_motor_observable_gradual_v2 as compat
from tools import distill_alphazuma_55_polar_dagger_v1 as dagger
from tools import distill_alphazuma_55_settled_v3 as settled


SCRIPT_PATH = Path(__file__).resolve()
EXPECTED_SEED_BASE = 1_546_000_000
EXPECTED_VARIANT = "action_head_only_fire_aim_v1"
TRAINABLE_PARAMETER_NAMES = ("action_net.weight", "action_net.bias")
_BASE_GRADUAL_VALIDATE = gradual._validate_preregistration
_BASE_OPTIMIZE = settled._optimize
_MODEL_STATES: dict[int, dict[str, Any]] = {}


def _validate_headonly_preregistration(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    original = legacy._read_json(resolved)
    run = original.get("run", {})
    if not (
        run.get("aim_loss_scope") == "fire_only"
        and run.get("route_variant") == EXPECTED_VARIANT
        and run.get("optimizer_parameter_scope") == "action_net_only"
        and tuple(run.get("trainable_parameter_names", ()))
        == TRAINABLE_PARAMETER_NAMES
        and run.get("frozen_backbone") is True
        and run.get("optimizer_state_reset_before_first_update") is True
        and int(run.get("training_seed_base", -1)) == EXPECTED_SEED_BASE
        and run.get("device") == "cuda:0"
    ):
        raise ValueError("head-only gradual contingency contract changed")

    original_script = gradual.SCRIPT_PATH
    original_seed_base = gradual.EXPECTED_SEED_BASE
    gradual.SCRIPT_PATH = SCRIPT_PATH
    gradual.EXPECTED_SEED_BASE = EXPECTED_SEED_BASE
    try:
        return _BASE_GRADUAL_VALIDATE(resolved)
    finally:
        gradual.EXPECTED_SEED_BASE = original_seed_base
        gradual.SCRIPT_PATH = original_script


def _configure_head_only_optimizer(
    *, model: Any, run: dict[str, Any]
) -> dict[str, Any]:
    import torch

    key = id(model)
    if key in _MODEL_STATES:
        return _MODEL_STATES[key]
    named = dict(model.policy.named_parameters())
    missing = [name for name in TRAINABLE_PARAMETER_NAMES if name not in named]
    if missing:
        raise ValueError(f"head-only trainable parameters missing: {missing}")

    trainable = []
    frozen_snapshot: dict[str, Any] = {}
    for name, parameter in named.items():
        is_trainable = name in TRAINABLE_PARAMETER_NAMES
        parameter.requires_grad_(is_trainable)
        if is_trainable:
            trainable.append(parameter)
        else:
            frozen_snapshot[name] = parameter.detach().cpu().clone()
    requires_grad_names = tuple(
        name for name, value in named.items() if value.requires_grad
    )
    if (
        len(requires_grad_names) != len(TRAINABLE_PARAMETER_NAMES)
        or set(requires_grad_names) != set(TRAINABLE_PARAMETER_NAMES)
    ):
        raise ValueError("head-only requires_grad inventory differs")

    optimizer_class = model.policy.optimizer_class
    optimizer_kwargs = dict(model.policy.optimizer_kwargs)
    model.policy.optimizer = optimizer_class(
        trainable,
        lr=float(run["learning_rate"]),
        **optimizer_kwargs,
    )
    optimizer_ids = {
        id(parameter)
        for group in model.policy.optimizer.param_groups
        for parameter in group["params"]
    }
    if optimizer_ids != {id(parameter) for parameter in trainable}:
        raise ValueError("head-only optimizer escaped trainable inventory")

    state = {
        "frozen_snapshot": frozen_snapshot,
        "frozen_parameter_names": tuple(frozen_snapshot),
        "trainable_parameter_names": TRAINABLE_PARAMETER_NAMES,
        "trainable_parameter_count": sum(value.numel() for value in trainable),
        "frozen_parameter_count": sum(
            value.numel()
            for name, value in named.items()
            if name not in TRAINABLE_PARAMETER_NAMES
        ),
        "torch": torch,
    }
    _MODEL_STATES[key] = state
    return state


def _verify_frozen_parameters(*, model: Any, state: dict[str, Any]) -> None:
    torch = state["torch"]
    named = dict(model.policy.named_parameters())
    for name, expected in state["frozen_snapshot"].items():
        actual = named[name].detach().cpu()
        if not torch.equal(actual, expected):
            raise RuntimeError(f"frozen head-only parameter changed: {name}")


def _head_only_optimize(
    *,
    model: Any,
    dataset: list[tuple[Any, Any, Any]],
    run: dict[str, Any],
    run_dir: Path,
    status_callback: Any,
) -> dict[str, Any]:
    state = _configure_head_only_optimizer(model=model, run=run)
    _verify_frozen_parameters(model=model, state=state)
    result = _BASE_OPTIMIZE(
        model=model,
        dataset=dataset,
        run=run,
        run_dir=run_dir,
        status_callback=status_callback,
    )
    _verify_frozen_parameters(model=model, state=state)
    result.update(
        {
            "optimizer_parameter_scope": "action_net_only",
            "trainable_parameter_names": list(
                state["trainable_parameter_names"]
            ),
            "trainable_parameter_count": int(
                state["trainable_parameter_count"]
            ),
            "frozen_parameter_count": int(state["frozen_parameter_count"]),
            "frozen_parameters_unchanged": True,
            "optimizer_state_reset_before_first_update": True,
        }
    )
    return result


def _annotate_completion(preregistration: dict[str, Any]) -> None:
    completion_path = Path(str(preregistration["outputs"]["completion"]))
    completion = legacy._read_json(completion_path.resolve(strict=True))
    rounds = completion.get("rounds", ())
    if not (
        completion.get("status") == "COMPLETE"
        and rounds
        and all(
            row.get("optimization", {}).get("optimizer_parameter_scope")
            == "action_net_only"
            and row.get("optimization", {}).get(
                "frozen_parameters_unchanged"
            )
            is True
            for row in rounds
        )
    ):
        raise ValueError("head-only completion contains mixed optimizer scope")
    completion.update(
        {
            "route_variant": EXPECTED_VARIANT,
            "optimizer_parameter_scope": "action_net_only",
            "trainable_parameter_names": list(TRAINABLE_PARAMETER_NAMES),
            "frozen_backbone": True,
            "frozen_parameters_unchanged": True,
            "optimizer_state_reset_before_first_update": True,
            "formal_seed_consumption": False,
            "formal_candidate_authority": False,
        }
    )
    dagger._write_atomic(completion_path, completion)


def main(argv: list[str] | None = None) -> int:
    parsed = gradual.build_parser().parse_args(argv)
    original_validate = gradual._validate_preregistration
    original_script = gradual.SCRIPT_PATH
    original_dagger_run = dagger.run
    original_optimize = settled._optimize
    try:
        gradual._validate_preregistration = _validate_headonly_preregistration
        gradual.SCRIPT_PATH = SCRIPT_PATH
        dagger.run = compat._compatibility_run
        settled._optimize = _head_only_optimize
        result = gradual.main(argv)
        if result == 0 and not parsed.validate_only:
            preregistration = _validate_headonly_preregistration(
                parsed.preregistration.expanduser().resolve(strict=True)
            )
            _annotate_completion(preregistration)
        return result
    finally:
        settled._optimize = original_optimize
        dagger.run = original_dagger_run
        gradual.SCRIPT_PATH = original_script
        gradual._validate_preregistration = original_validate
        _MODEL_STATES.clear()


if __name__ == "__main__":
    raise SystemExit(main())
