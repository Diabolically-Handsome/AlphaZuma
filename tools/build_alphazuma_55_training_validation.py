"""Freeze a non-formal AlphaZuma 55 training-validation comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools.build_alphazuma_55_eval_contract import (
    _model_spec,
    _read_json,
    _sha256,
    _write_json_exclusive,
    build_contracts,
)


def build_training_validation_contracts(
    *,
    master_path: Path,
    original_root: Path,
    evaluator_path: Path,
    output_root: Path,
    models: list[dict[str, Any]],
    base_seed: int,
    device: str,
    parallel_envs: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a paired diagnostic without consuming any formal seed range."""

    master = _read_json(master_path)
    registry = master["seed_registry"]["training_validation"]
    manifest, preregistration = build_contracts(
        master_path=master_path,
        original_root=original_root,
        evaluator_path=evaluator_path,
        stage="engineering",
        output_root=output_root,
        models=models,
        parallel_envs_per_shard=parallel_envs,
    )

    total_attempts = int(preregistration["total_attempts"])
    last_seed = base_seed + total_attempts - 1
    if not (
        int(registry["first"]) <= base_seed
        and last_seed <= int(registry["last"])
    ):
        raise ValueError("training-validation matrix exceeds its seed registry")
    if device not in {"cuda:0", "cuda:1"}:
        raise ValueError("training validation requires cuda:0 or cuda:1")
    if parallel_envs < 1:
        raise ValueError("parallel_envs must be positive")

    builder_path = Path(__file__).resolve()
    manifest["stage"] = "training_validation"
    manifest["purpose"] = (
        "Paired early-health diagnostic only; it cannot select the formal finalist."
    )
    manifest["builder"] = {
        "path": str(builder_path),
        "sha256": _sha256(builder_path),
    }

    preregistration["stage"] = "training_validation"
    preregistration["seed_plan"] = {
        "registry": "training_validation",
        "base_seed": base_seed,
        "last_seed": last_seed,
    }
    preregistration["execution"] = {
        "output_root": str(output_root),
        "shard_count": 1,
        "devices": [device],
        "parallel_envs_per_shard": parallel_envs,
    }
    preregistration["diagnostic_contract"] = {
        "paired_models_share_identical_task_seeds": True,
        "formal_selection_authority": False,
        "training_recipe_change_authority": False,
        "formal_seed_ranges_consumed": False,
        "max_ticks": int(
            preregistration["environment"]["base_config"]["max_ticks"]
        ),
    }
    preregistration["builder"] = dict(manifest["builder"])
    return manifest, preregistration


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-preregistration", required=True, type=Path)
    parser.add_argument("--original-root", required=True, type=Path)
    parser.add_argument(
        "--evaluator",
        type=Path,
        default=Path(__file__).resolve().with_name(
            "evaluate_zero_shot_multilevel.py"
        ),
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--manifest-output", required=True, type=Path)
    parser.add_argument("--preregistration-output", required=True, type=Path)
    parser.add_argument("--model", action="append", type=_model_spec, required=True)
    parser.add_argument("--base-seed", required=True, type=int)
    parser.add_argument("--device", choices=("cuda:0", "cuda:1"), default="cuda:0")
    parser.add_argument("--parallel-envs", type=int, default=55)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_path = args.manifest_output.expanduser().resolve()
    preregistration_path = args.preregistration_output.expanduser().resolve()
    for path in (manifest_path, preregistration_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite frozen contract: {path}")

    manifest, preregistration = build_training_validation_contracts(
        master_path=args.master_preregistration.expanduser().resolve(strict=True),
        original_root=args.original_root.expanduser().resolve(strict=True),
        evaluator_path=args.evaluator.expanduser().resolve(strict=True),
        output_root=args.output_root.expanduser().resolve(),
        models=args.model,
        base_seed=args.base_seed,
        device=args.device,
        parallel_envs=args.parallel_envs,
    )
    _write_json_exclusive(manifest_path, manifest)
    _write_json_exclusive(preregistration_path, preregistration)
    print(
        json.dumps(
            {
                "status": "FROZEN",
                "stage": "training_validation",
                "manifest": str(manifest_path),
                "manifest_sha256": _sha256(manifest_path),
                "preregistration": str(preregistration_path),
                "preregistration_sha256": _sha256(preregistration_path),
                "models": len(manifest["models"]),
                "levels": len(preregistration["levels"]),
                "attempts_per_model": preregistration["total_attempts"],
                "seed_plan": preregistration["seed_plan"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
