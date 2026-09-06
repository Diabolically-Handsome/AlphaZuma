"""Freeze validation semantics before the balanced-distillation model exists."""

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
MATERIALIZER = PROJECT_ROOT / "tools/materialize_alphazuma_55_balanced_validation.py"
CONTRACT_BUILDER = PROJECT_ROOT / "tools/build_alphazuma_55_training_validation.py"
EVALUATOR = PROJECT_ROOT / "tools/evaluate_zero_shot_multilevel_v2.py"
AUDITOR = PROJECT_ROOT / "tools/audit_alphazuma_55_training_validation.py"
PRIOR_FIRST = PROJECT_ROOT / "diagnostics/alphazuma-55-distilled-first-checkpoint-validation-s92081501-preregistration-v1.json"
PRIOR_MILESTONE = PROJECT_ROOT / "diagnostics/alphazuma-55-distilled-milestone-validation-s92081502-plan-v1.json"


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
        raise FileExistsError(f"refusing to overwrite balanced validation plan: {path}")
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


def _overlap(first: int, last: int, other_first: int, other_last: int) -> bool:
    return max(first, other_first) <= min(last, other_last)


def build(
    *,
    master_path: Path,
    balanced_preregistration: Path,
    source_model: Path,
    source_completion: Path,
    target_expected_path: Path,
    target_expected_completion: Path,
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
    balanced = _read(balanced_preregistration)
    completion = _read(source_completion)
    if balanced.get("status") != "FROZEN_BEFORE_TRAINING":
        raise ValueError("balanced distillation preregistration is not frozen")
    if completion.get("status") != "COMPLETE":
        raise ValueError("source distillation completion is not complete")
    if completion["final_model"]["sha256"] != _sha256(source_model):
        raise ValueError("source completion differs from source model")
    if target_expected_path.exists() or target_expected_completion.exists():
        raise FileExistsError("balanced validation must freeze before target completion")
    if any(path.exists() for path in (output_root, models_manifest, preregistration_output, audit_output)):
        raise FileExistsError("balanced validation output already exists")
    if device not in {"cuda:0", "cuda:1"} or parallel_envs < 1:
        raise ValueError("invalid balanced validation execution")

    last_seed = base_seed + 54
    registry = master["seed_registry"]["training_validation"]
    if not (int(registry["first"]) <= base_seed <= last_seed <= int(registry["last"])):
        raise ValueError("balanced validation escaped training-validation seeds")
    prior_first = _read(PRIOR_FIRST)["seed_plan"]
    prior_milestone = _read(PRIOR_MILESTONE)["matrix"]
    prior_ranges = [
        (int(prior_first["base_seed"]), int(prior_first["last_seed"])),
        (int(prior_milestone["base_seed"]), int(prior_milestone["last_seed"])),
    ]
    if any(_overlap(base_seed, last_seed, first, last) for first, last in prior_ranges):
        raise ValueError("balanced validation overlaps a prior validation matrix")

    return {
        "schema": "zuma-rl.alphazuma-55-balanced-distillation-validation-plan",
        "version": 1,
        "status": "FROZEN_BEFORE_TARGET_MODEL",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_id": master["campaign_id"],
        "purpose": (
            "Paired fresh-seed engineering test of whether balanced fire sampling "
            "improves the distilled initialization; it cannot select a finalist."
        ),
        "master_preregistration": _artifact(master_path),
        "balanced_distillation_preregistration": _artifact(
            balanced_preregistration
        ),
        "original_root": str(original_root.resolve(strict=True)),
        "models": {
            "baseline": {
                "id": "specialist-distilled-v1",
                "training_steps": int(completion["final_model"]["num_timesteps"]),
                **_artifact(source_model),
            },
            "baseline_completion": _artifact(source_completion),
            "target": {
                "id": "balanced-distilled-v1",
                "expected_path": str(target_expected_path.resolve()),
                "expected_completion_path": str(
                    target_expected_completion.resolve()
                ),
                "must_be_absent_at_freeze": True,
                "hash_and_training_steps_frozen_after_completion": True,
            },
        },
        "matrix": {
            "levels": 55,
            "attempts_per_level_per_model": 1,
            "paired_models_share_identical_task_seeds": True,
            "seed_registry": "training_validation",
            "base_seed": base_seed,
            "last_seed": last_seed,
            "max_ticks": 12_000,
            "prior_consumed_ranges": [list(value) for value in prior_ranges],
        },
        "execution": {"device": device, "parallel_envs": parallel_envs},
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
            "evidence_scope": "engineering_only",
            "positive_signal": (
                "target paired cleared-level coverage exceeds baseline and no "
                "capacity overflow is hidden"
            ),
            "positive_signal_authorizes": "separate PPO route preregistration only",
            "formal_candidate_registration_authorized": False,
            "negative_result_must_be_retained": True,
        },
        "authority_boundary": {
            "formal_seed_consumption": False,
            "formal_selection_authority": False,
            "training_recipe_change_authority": False,
            "candidate_addition_or_removal_authority": False,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-preregistration", required=True, type=Path)
    parser.add_argument("--balanced-preregistration", required=True, type=Path)
    parser.add_argument("--source-model", required=True, type=Path)
    parser.add_argument("--source-completion", required=True, type=Path)
    parser.add_argument("--target-expected-path", required=True, type=Path)
    parser.add_argument("--target-expected-completion", required=True, type=Path)
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
        balanced_preregistration=args.balanced_preregistration.expanduser().resolve(strict=True),
        source_model=args.source_model.expanduser().resolve(strict=True),
        source_completion=args.source_completion.expanduser().resolve(strict=True),
        target_expected_path=args.target_expected_path.expanduser().resolve(),
        target_expected_completion=args.target_expected_completion.expanduser().resolve(),
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
                "formal_seed_consumption": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
