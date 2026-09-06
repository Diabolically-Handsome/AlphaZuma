"""Freeze fresh validation before the settled-V3 CNN checkpoints exist."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()
MATERIALIZER = PROJECT_ROOT / "tools/materialize_alphazuma_55_settled_v3_validation.py"
CONTRACT_BUILDER = PROJECT_ROOT / "tools/build_alphazuma_55_training_validation.py"
EVALUATOR = PROJECT_ROOT / "tools/evaluate_zero_shot_multilevel_v2.py"
AUDITOR = PROJECT_ROOT / "tools/audit_alphazuma_55_training_validation.py"
PRIOR_VALIDATIONS = (
    PROJECT_ROOT
    / "diagnostics/alphazuma-55-distilled-first-checkpoint-validation-s92081501-preregistration-v1.json",
    PROJECT_ROOT
    / "diagnostics/alphazuma-55-distilled-milestone-validation-s92081502-plan-v1.json",
    PROJECT_ROOT
    / "diagnostics/alphazuma-55-balanced-distillation-validation-s99081511-plan-v1.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _artifact(path: Path) -> dict[str, str]:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "sha256": _sha256(resolved)}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite V3 validation plan: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _range_from_prior(value: dict[str, Any]) -> tuple[int, int]:
    matrix = value.get("matrix", value.get("seed_plan", {}))
    if "base_seed" not in matrix or "last_seed" not in matrix:
        raise ValueError("prior validation does not expose a seed range")
    return int(matrix["base_seed"]), int(matrix["last_seed"])


def _overlap(first: int, last: int, other: tuple[int, int]) -> bool:
    return max(first, other[0]) <= min(last, other[1])


def build(
    *,
    master_path: Path,
    distillation_preregistration: Path,
    baseline_model: Path,
    baseline_migration: Path,
    target_run_dir: Path,
    original_root: Path,
    output_root: Path,
    models_manifest: Path,
    preregistration_output: Path,
    audit_output: Path,
    base_seed: int,
    device: str,
    parallel_envs: int,
) -> dict[str, Any]:
    master = _read(master_path)
    distillation = _read(distillation_preregistration)
    migration = _read(baseline_migration)
    migrated = migration.get("migrated_model", {})
    if (
        master.get("schema")
        != "zuma-rl.alphazuma-55-weekend-master-preregistration"
        or master.get("version") != 1
    ):
        raise ValueError("unexpected master preregistration")
    if (
        distillation.get("schema")
        != "zuma-rl.alphazuma-55-settled-v3-distillation-preregistration"
        or distillation.get("status") != "FROZEN_BEFORE_TRAINING"
    ):
        raise ValueError("settled-V3 distillation is not frozen")
    if (
        migration.get("status") != "PASS"
        or migrated.get("sha256") != _sha256(baseline_model)
        or Path(str(migrated.get("path", ""))).resolve() != baseline_model
    ):
        raise ValueError("baseline migration does not bind the baseline model")
    expected_run = Path(str(distillation["run"]["run_dir"])).resolve()
    if expected_run != target_run_dir:
        raise ValueError("validation target differs from distillation run")
    expected_paths = [
        target_run_dir / "epoch_02_model.zip",
        target_run_dir / "epoch_04_model.zip",
        target_run_dir / "epoch_06_model.zip",
        target_run_dir / "epoch_08_model.zip",
        target_run_dir / "final_model.zip",
        target_run_dir / "completion.json",
    ]
    if target_run_dir.exists() or any(path.exists() for path in expected_paths):
        raise FileExistsError("V3 validation must freeze before target training")
    if any(
        path.exists()
        for path in (
            output_root,
            models_manifest,
            preregistration_output,
            audit_output,
        )
    ):
        raise FileExistsError("V3 validation output already exists")
    if device not in {"cuda:0", "cuda:1"} or not 1 <= parallel_envs <= 55:
        raise ValueError("invalid V3 validation execution")

    last_seed = base_seed + 54
    registry = master["seed_registry"]["training_validation"]
    if not int(registry["first"]) <= base_seed <= last_seed <= int(
        registry["last"]
    ):
        raise ValueError("V3 validation escaped training-validation seeds")
    prior_ranges = [_range_from_prior(_read(path)) for path in PRIOR_VALIDATIONS]
    if any(_overlap(base_seed, last_seed, value) for value in prior_ranges):
        raise ValueError("V3 validation overlaps a prior validation matrix")

    return {
        "schema": "zuma-rl.alphazuma-55-settled-v3-validation-plan",
        "version": 1,
        "status": "FROZEN_BEFORE_TARGET_MODEL",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_id": master["campaign_id"],
        "purpose": (
            "Paired fresh-seed engineering comparison of the V11 initialization "
            "against four precommitted V3 checkpoints and the final clone."
        ),
        "master_preregistration": _artifact(master_path),
        "distillation_preregistration": _artifact(
            distillation_preregistration
        ),
        "original_root": str(original_root.resolve(strict=True)),
        "models": {
            "baseline": {
                "id": "v11-full55-v1",
                "training_steps": int(migrated["num_timesteps"]),
                **_artifact(baseline_model),
            },
            "baseline_migration": _artifact(baseline_migration),
            "target": {
                "run_dir": str(target_run_dir),
                "expected_completion_path": str(
                    (target_run_dir / "completion.json").resolve()
                ),
                "expected_checkpoints": [
                    {
                        "id": f"settled-v3-epoch-{epoch:02d}",
                        "epoch": epoch,
                        "expected_path": str(
                            (target_run_dir / f"epoch_{epoch:02d}_model.zip").resolve()
                        ),
                    }
                    for epoch in (2, 4, 6, 8)
                ],
                "expected_final": {
                    "id": "settled-v3-final",
                    "expected_path": str(
                        (target_run_dir / "final_model.zip").resolve()
                    ),
                },
                "all_paths_must_be_absent_at_freeze": True,
                "hashes_frozen_after_completion": True,
            },
        },
        "matrix": {
            "levels": 55,
            "attempts_per_level_per_model": 1,
            "paired_models_share_identical_task_seeds": True,
            "seed_registry": "training_validation",
            "base_seed": base_seed,
            "last_seed": last_seed,
            "max_ticks": 30_000,
            "prior_reserved_or_consumed_ranges": [
                list(value) for value in prior_ranges
            ],
        },
        "execution": {
            "device": device,
            "parallel_envs": parallel_envs,
            "shard_count": 1,
        },
        "implementation": {
            "planner": _artifact(SCRIPT_PATH),
            "materializer": _artifact(MATERIALIZER),
            "contract_builder": _artifact(CONTRACT_BUILDER),
            "evaluator": _artifact(EVALUATOR),
            "independent_auditor": _artifact(AUDITOR),
        },
        "outputs": {
            "run_root": str(output_root.resolve()),
            "models_manifest": str(models_manifest.resolve()),
            "preregistration": str(preregistration_output.resolve()),
            "audit_receipt": str(audit_output.resolve()),
        },
        "decision_rule": {
            "ranking": [
                "cleared_levels_descending",
                "wins_descending",
                "median_winning_ticks_ascending",
                "earlier_epoch_then_final",
            ],
            "evidence_scope": "engineering_checkpoint_choice_only",
            "formal_candidate_registration_authorized": False,
            "formal_selection_seed_consumption": False,
            "negative_result_must_be_retained": True,
        },
        "authority_boundary": {
            "formal_seed_consumption": False,
            "formal_selection_authority": False,
            "training_recipe_change_authority": False,
        },
    }


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
    value = build(
        master_path=args.master_preregistration.expanduser().resolve(strict=True),
        distillation_preregistration=args.distillation_preregistration.expanduser().resolve(strict=True),
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
    _write_exclusive(output, value)
    print(
        json.dumps(
            {
                "status": value["status"],
                "output": str(output),
                "sha256": _sha256(output),
                "seed_range": [
                    value["matrix"]["base_seed"],
                    value["matrix"]["last_seed"],
                ],
                "target_absent": True,
                "models_after_materialization": 6,
                "formal_seed_consumption": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
