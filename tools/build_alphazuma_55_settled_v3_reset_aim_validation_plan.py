"""Freeze fresh validation for the settled-V3 reset-aim clone."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools import build_alphazuma_55_settled_v3_validation_plan as base


SCRIPT_PATH = Path(__file__).resolve()
MATERIALIZER = SCRIPT_PATH.with_name(
    "materialize_alphazuma_55_settled_v3_reset_aim_validation.py"
)
PRIOR_PLANS = (
    SCRIPT_PATH.parents[1]
    / "diagnostics/alphazuma-55-settled-v3-validation-s99081518-plan-v1.json",
    SCRIPT_PATH.parents[1]
    / (
        "diagnostics/alphazuma-55-settled-v3-high-fidelity-validation-"
        "s99081520-plan-v1.json"
    ),
)
CHECKPOINT_EPOCHS = (4, 8, 12, 16, 20, 24)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-preregistration", required=True, type=Path)
    parser.add_argument("--distillation-preregistration", required=True, type=Path)
    parser.add_argument("--baseline-model", required=True, type=Path)
    parser.add_argument("--baseline-migration", required=True, type=Path)
    parser.add_argument("--target-run-dir", required=True, type=Path)
    parser.add_argument("--original-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--models-manifest", required=True, type=Path)
    parser.add_argument("--preregistration-output", required=True, type=Path)
    parser.add_argument("--audit-output", required=True, type=Path)
    parser.add_argument("--base-seed", required=True, type=int)
    parser.add_argument("--device", choices=("cuda:0", "cuda:1"), default="cuda:1")
    parser.add_argument("--parallel-envs", type=int, default=24)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    distillation_path = args.distillation_preregistration.expanduser().resolve(
        strict=True
    )
    distillation = base._read(distillation_path)
    reset = distillation.get("run", {}).get("aim_head_reset", {})
    if reset.get("enabled") is not True or reset.get("rows") != "aim_component_only":
        raise ValueError("distillation preregistration is not the reset-aim route")
    value = base.build(
        master_path=args.master_preregistration.expanduser().resolve(strict=True),
        distillation_preregistration=distillation_path,
        baseline_model=args.baseline_model.expanduser().resolve(strict=True),
        baseline_migration=args.baseline_migration.expanduser().resolve(strict=True),
        target_run_dir=args.target_run_dir.expanduser().resolve(),
        original_root=args.original_root.expanduser().resolve(strict=True),
        output_root=args.output_root.expanduser().resolve(),
        models_manifest=args.models_manifest.expanduser().resolve(),
        preregistration_output=args.preregistration_output.expanduser().resolve(),
        audit_output=args.audit_output.expanduser().resolve(),
        base_seed=args.base_seed,
        device=args.device,
        parallel_envs=args.parallel_envs,
    )
    prior_ranges = []
    for prior_path in PRIOR_PLANS:
        prior = base._read(prior_path.resolve(strict=True))
        prior_range = (
            int(prior["matrix"]["base_seed"]),
            int(prior["matrix"]["last_seed"]),
        )
        if base._overlap(
            int(value["matrix"]["base_seed"]),
            int(value["matrix"]["last_seed"]),
            prior_range,
        ):
            raise ValueError("reset-aim validation overlaps an earlier V3 matrix")
        prior_ranges.append(prior_range)

    target_run = Path(str(value["models"]["target"]["run_dir"])).resolve()
    value["schema"] = "zuma-rl.alphazuma-55-settled-v3-reset-aim-validation-plan"
    value["created_utc"] = datetime.now(timezone.utc).isoformat()
    value["purpose"] = (
        "Paired fresh-seed checkpoint comparison of the precommitted reset-aim "
        "clone against its unchanged V11 initialization."
    )
    value["models"]["target"]["expected_checkpoints"] = [
        {
            "id": f"settled-v3-reset-aim-epoch-{epoch:02d}",
            "epoch": epoch,
            "expected_path": str(
                (target_run / f"epoch_{epoch:02d}_model.zip").resolve()
            ),
        }
        for epoch in CHECKPOINT_EPOCHS
    ]
    value["models"]["target"]["expected_final"] = {
        "id": "settled-v3-reset-aim-final",
        "expected_path": str((target_run / "final_model.zip").resolve()),
    }
    for prior_range in prior_ranges:
        encoded = list(prior_range)
        if encoded not in value["matrix"]["prior_reserved_or_consumed_ranges"]:
            value["matrix"]["prior_reserved_or_consumed_ranges"].append(encoded)
    value["implementation"]["base_planner"] = base._artifact(
        Path(base.__file__).resolve()
    )
    value["implementation"]["planner"] = base._artifact(SCRIPT_PATH)
    value["implementation"]["materializer"] = base._artifact(MATERIALIZER)
    value["decision_rule"]["ranking"] = [
        "cleared_levels_descending",
        "wins_descending",
        "median_winning_ticks_ascending",
        "earlier_epoch_then_final",
    ]
    base._write_exclusive(output, value)
    print(
        json.dumps(
            {
                "status": value["status"],
                "output": str(output),
                "sha256": base._sha256(output),
                "seed_range": [
                    value["matrix"]["base_seed"],
                    value["matrix"]["last_seed"],
                ],
                "target_absent": True,
                "models_after_materialization": 8,
                "formal_seed_consumption": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
