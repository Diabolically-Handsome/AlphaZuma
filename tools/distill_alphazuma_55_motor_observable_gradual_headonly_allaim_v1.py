"""Run the registered head-only plus all-actions-aim DAgger factorial cell.

This engineering route combines the two independently registered causal
interventions: every raw teacher intent supervises the aim component, while
only ``action_net.weight`` and ``action_net.bias`` are trainable.  The source
representation, environment, teacher, DAgger schedule, batches, and gates are
otherwise unchanged.
"""

from __future__ import annotations

import copy
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tools import distill_alphazuma_55 as legacy
from tools import distill_alphazuma_55_motor_observable_gradual_headonly_v1 as headonly
from tools import distill_alphazuma_55_motor_observable_gradual_v1 as gradual
from tools import distill_alphazuma_55_motor_observable_gradual_v2 as compat
from tools import distill_alphazuma_55_polar_dagger_v1 as dagger
from tools import distill_alphazuma_55_settled_v3 as settled


SCRIPT_PATH = Path(__file__).resolve()
EXPECTED_SEED_BASE = 1_546_001_000
EXPECTED_VARIANT = "action_head_only_all_actions_aim_v1"
TRAINABLE_PARAMETER_NAMES = headonly.TRAINABLE_PARAMETER_NAMES
_BASE_GRADUAL_VALIDATE = gradual._validate_preregistration


def _validate_preregistration(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    original = legacy._read_json(resolved)
    run = original.get("run", {})
    if not (
        run.get("aim_loss_scope") == "all_actions"
        and run.get("all_action_aim_supervision") is True
        and run.get("aim_loss_sample_semantics")
        == "raw_teacher_intent_on_every_retained_action"
        and run.get("route_variant") == EXPECTED_VARIANT
        and run.get("optimizer_parameter_scope") == "action_net_only"
        and tuple(run.get("trainable_parameter_names", ()))
        == TRAINABLE_PARAMETER_NAMES
        and run.get("frozen_backbone") is True
        and run.get("optimizer_state_reset_before_first_update") is True
        and int(run.get("training_seed_base", -1)) == EXPECTED_SEED_BASE
        and run.get("device") == "cuda:1"
    ):
        raise ValueError("head-only all-actions factorial contract changed")

    original_read = legacy._read_json
    original_script = gradual.SCRIPT_PATH
    original_seed_base = gradual.EXPECTED_SEED_BASE

    def read_with_fire_only_alias(candidate: Path) -> dict[str, Any]:
        value = original_read(candidate)
        if Path(candidate).resolve() != resolved:
            return value
        adjusted = copy.deepcopy(value)
        adjusted["run"]["aim_loss_scope"] = "fire_only"
        return adjusted

    legacy._read_json = read_with_fire_only_alias
    gradual.SCRIPT_PATH = SCRIPT_PATH
    gradual.EXPECTED_SEED_BASE = EXPECTED_SEED_BASE
    try:
        validated = _BASE_GRADUAL_VALIDATE(resolved)
    finally:
        gradual.EXPECTED_SEED_BASE = original_seed_base
        gradual.SCRIPT_PATH = original_script
        legacy._read_json = original_read

    validated["run"].update(
        {
            "aim_loss_scope": "all_actions",
            "all_action_aim_supervision": True,
            "aim_loss_sample_semantics": (
                "raw_teacher_intent_on_every_retained_action"
            ),
        }
    )
    return validated


def _annotate_completion(preregistration: dict[str, Any]) -> None:
    completion_path = Path(str(preregistration["outputs"]["completion"]))
    completion = legacy._read_json(completion_path.resolve(strict=True))
    rounds = completion.get("rounds", ())
    if not (
        completion.get("status") == "COMPLETE"
        and rounds
        and all(
            row.get("optimization", {}).get("aim_loss_scope")
            == "all_actions"
            and row.get("optimization", {}).get(
                "optimizer_parameter_scope"
            )
            == "action_net_only"
            and row.get("optimization", {}).get(
                "frozen_parameters_unchanged"
            )
            is True
            for row in rounds
        )
    ):
        raise ValueError("head-only all-actions completion is ineligible")
    completion.update(
        {
            "route_variant": EXPECTED_VARIANT,
            "aim_loss_scope": "all_actions",
            "all_action_aim_supervision": True,
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
        gradual._validate_preregistration = _validate_preregistration
        gradual.SCRIPT_PATH = SCRIPT_PATH
        dagger.run = compat._compatibility_run
        settled._optimize = headonly._head_only_optimize
        result = gradual.main(argv)
        if result == 0 and not parsed.validate_only:
            preregistration = _validate_preregistration(
                parsed.preregistration.expanduser().resolve(strict=True)
            )
            _annotate_completion(preregistration)
        return result
    finally:
        settled._optimize = original_optimize
        dagger.run = original_dagger_run
        gradual.SCRIPT_PATH = original_script
        gradual._validate_preregistration = original_validate
        headonly._MODEL_STATES.clear()


if __name__ == "__main__":
    raise SystemExit(main())
