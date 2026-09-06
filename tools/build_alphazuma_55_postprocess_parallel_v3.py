"""Freeze the capacity-safe AlphaZuma 55 postprocess contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    import sys

    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tools import build_alphazuma_55_postprocess_parallel_v2 as legacy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTROLLER = PROJECT_ROOT / "tools/run_alphazuma_55_postprocess_parallel_v3.py"
EVALUATOR = PROJECT_ROOT / "tools/evaluate_zero_shot_multilevel_v2.py"
INCIDENT = PROJECT_ROOT / "diagnostics/alphazuma-55-horizon-probe-s99081501-incident-v1.json"
EVALUATOR_TEST = PROJECT_ROOT / "tests/test_evaluate_zero_shot_multilevel_v2.py"
CONTROLLER_TEST = PROJECT_ROOT / "tests/test_run_alphazuma_55_postprocess_parallel_v3.py"


def build(**kwargs: Any) -> dict[str, Any]:
    value = legacy.build(**kwargs)
    value["implementation"]["controller"] = legacy._artifact(CONTROLLER)
    value["implementation"]["evaluator"] = legacy._artifact(EVALUATOR)
    value["capacity_fail_closed_contract"] = {
        "actor_observation_capacity_balls": 768,
        "observation_shape_unchanged": True,
        "silent_state_truncation_forbidden": True,
        "exact_capacity_overflow_outcome": "loss",
        "only_affected_policy_seed_attempt_fails": True,
        "sibling_attempts_must_continue": True,
        "all_other_exceptions_fail_the_shard": True,
        "trigger_incident": legacy._artifact(INCIDENT),
        "unit_tests": [
            legacy._artifact(EVALUATOR_TEST),
            legacy._artifact(CONTROLLER_TEST),
        ],
        "formal_seed_consumption_before_contract": False,
    }
    return value


def main(argv: list[str] | None = None) -> int:
    args = legacy.build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen preregistration: {output}")
    value = build(
        master_path=args.master_preregistration.expanduser().resolve(strict=True),
        original_root=args.original_root.expanduser().resolve(strict=True),
        migrations=args.migration,
        route_paths=[
            path.expanduser().resolve(strict=True) for path in args.training_route
        ],
        controller_root=args.controller_root.expanduser().resolve(),
        selection_root=args.selection_root.expanduser().resolve(),
        final_root=args.final_blind_root.expanduser().resolve(),
        continuous_root=args.continuous_root.expanduser().resolve(),
    )
    if args.validate_only:
        print(
            json.dumps(
                {**value, "status": "VALID_PREVIEW"},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    legacy._write_exclusive(output, value)
    print(
        json.dumps(
            {
                "status": "FROZEN",
                "output": str(output),
                "sha256": legacy._sha256(output),
                "routes": len(value["training_routes"]),
                "migrations": len(value["migrations"]),
                "controller": value["implementation"]["controller"],
                "evaluator": value["implementation"]["evaluator"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
