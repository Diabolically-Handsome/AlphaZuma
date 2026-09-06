"""Run gradual motor DAgger with aim supervision on every actor tick.

This engineering-only contingency keeps the frozen V2 environment, teacher,
student architecture, DAgger schedule, action masks, and optimizer unchanged.
Its sole recipe change is to supervise the teacher's aim component for every
raw intent label instead of only for labels whose verb is ``fire``.  That
matches the elite-human actuator: every action, including ``wait``, enqueues an
aim target into the reaction-delay queue.
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
from tools import distill_alphazuma_55_motor_observable_gradual_v1 as gradual
from tools import distill_alphazuma_55_motor_observable_gradual_v2 as compat
from tools import distill_alphazuma_55_polar_dagger_v1 as dagger


SCRIPT_PATH = Path(__file__).resolve()
EXPECTED_SEED_BASE = 1_546_000_000
EXPECTED_VARIANT = "all_actions_aim_supervision_v1"
_BASE_GRADUAL_VALIDATE = gradual._validate_preregistration


def _validate_allaim_preregistration(path: Path) -> dict[str, Any]:
    """Reuse every V1 invariant while explicitly validating the one change."""

    resolved = path.resolve(strict=True)
    original = legacy._read_json(resolved)
    run = original.get("run", {})
    if not (
        run.get("aim_loss_scope") == "all_actions"
        and run.get("route_variant") == EXPECTED_VARIANT
        and run.get("all_action_aim_supervision") is True
        and int(run.get("training_seed_base", -1)) == EXPECTED_SEED_BASE
    ):
        raise ValueError("all-action aim contingency contract changed")

    original_read = legacy._read_json
    original_script = gradual.SCRIPT_PATH
    original_seed_base = gradual.EXPECTED_SEED_BASE

    def read_with_validation_alias(candidate: Path) -> dict[str, Any]:
        value = original_read(candidate)
        if Path(candidate).resolve() != resolved:
            return value
        adjusted = copy.deepcopy(value)
        # The inherited validator has a frozen fire-only literal.  Present that
        # literal only to reuse its complete invariant audit, then restore the
        # independently checked all-actions value in the returned object.
        adjusted["run"]["aim_loss_scope"] = "fire_only"
        return adjusted

    legacy._read_json = read_with_validation_alias
    gradual.SCRIPT_PATH = SCRIPT_PATH
    gradual.EXPECTED_SEED_BASE = EXPECTED_SEED_BASE
    try:
        validated = _BASE_GRADUAL_VALIDATE(resolved)
    finally:
        gradual.EXPECTED_SEED_BASE = original_seed_base
        gradual.SCRIPT_PATH = original_script
        legacy._read_json = original_read

    validated["run"]["aim_loss_scope"] = "all_actions"
    return validated


def _annotate_completion(preregistration: dict[str, Any]) -> None:
    completion_path = Path(str(preregistration["outputs"]["completion"]))
    completion = legacy._read_json(completion_path.resolve(strict=True))
    if completion.get("status") != "COMPLETE":
        raise ValueError("all-action aim completion is not complete")
    rounds = completion.get("rounds", ())
    if not rounds or any(
        row.get("optimization", {}).get("aim_loss_scope") != "all_actions"
        for row in rounds
    ):
        raise ValueError("all-action aim completion contains mixed loss scopes")
    completion.update(
        {
            "route_variant": EXPECTED_VARIANT,
            "aim_loss_scope": "all_actions",
            "all_action_aim_supervision": True,
            "actuator_alignment": (
                "every actor action enqueues an aim target before motor advance"
            ),
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
    try:
        gradual._validate_preregistration = _validate_allaim_preregistration
        gradual.SCRIPT_PATH = SCRIPT_PATH
        dagger.run = compat._compatibility_run
        result = gradual.main(argv)
        if result == 0 and not parsed.validate_only:
            preregistration = _validate_allaim_preregistration(
                parsed.preregistration.expanduser().resolve(strict=True)
            )
            _annotate_completion(preregistration)
        return result
    finally:
        dagger.run = original_dagger_run
        gradual.SCRIPT_PATH = original_script
        gradual._validate_preregistration = original_validate


if __name__ == "__main__":
    raise SystemExit(main())
