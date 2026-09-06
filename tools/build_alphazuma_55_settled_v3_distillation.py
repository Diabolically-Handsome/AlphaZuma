"""Freeze one mask-aware settled-V3 behavior-cloning run for AlphaZuma 55."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from zuma_rl.alphazuma_55 import INCLUDED_LEVELS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "sha256": _sha256(resolved)}


def _validate_probe(
    path: Path,
    *,
    policy_id: str,
) -> None:
    result = _read_json(path)
    summary = result.get("summary", {})
    if (
        result.get("status") != "COMPLETE"
        or result.get("formal_seed_consumption") is not False
        or result.get("teacher", {}).get("policy_id") != policy_id
        or int(summary.get("attempts", -1)) != 55
        or int(summary.get("wins", -1)) != 55
        or int(summary.get("level_coverage", -1)) != 55
        or int(summary.get("losses", -1)) != 0
        or int(summary.get("truncations", -1)) != 0
        or int(summary.get("capacity_overflows", -1)) != 0
    ):
        raise ValueError(f"teacher evidence is not a clean 55/55 result: {path}")


def build(
    *,
    master_path: Path,
    source_model_path: Path,
    source_migration_path: Path,
    unmasked_result_path: Path,
    masked_result_path: Path,
    run_dir: Path,
    device: str,
    training_seed_base: int,
    model_seed: int,
) -> dict[str, Any]:
    master = _read_json(master_path)
    if (
        master.get("schema")
        != "zuma-rl.alphazuma-55-weekend-master-preregistration"
        or master.get("version") != 1
    ):
        raise ValueError("unexpected master preregistration")
    migration = _read_json(source_migration_path)
    migrated = migration.get("migrated_model", {})
    source_hash = _sha256(source_model_path)
    if (
        migration.get("status") != "PASS"
        or Path(str(migrated.get("path", ""))).resolve() != source_model_path
        or migrated.get("sha256") != source_hash
        or tuple(migrated.get("observation_shape", ())) != (22_833,)
        or tuple(migrated.get("action_nvec", ())) != (4, 180)
    ):
        raise ValueError("source migration does not bind a full55-v1 model")
    _validate_probe(
        unmasked_result_path,
        policy_id="curve-aware-settled-strategic-actor-observable-v3",
    )
    _validate_probe(
        masked_result_path,
        policy_id="curve-aware-settled-strategic-masked-v3",
    )
    if run_dir.exists():
        raise FileExistsError(f"settled-V3 run already exists: {run_dir}")
    training_seed_last = training_seed_base + len(INCLUDED_LEVELS) - 1
    training = master["seed_registry"]["training"]
    if not (
        int(training["first"])
        <= training_seed_base
        <= training_seed_last
        <= int(training["last"])
    ):
        raise ValueError("settled-V3 seeds escaped the training registry")

    root = Path(__file__).resolve().parents[1]
    trainer = root / "tools/distill_alphazuma_55_settled_v3.py"
    implementation_paths = {
        "legacy_distillation_runtime": root / "tools/distill_alphazuma_55.py",
        "behavior_clone_batch_primitive": root
        / "tools/pretrain_maskable_ppo_teacher.py",
        "settled_teacher_v3": root
        / "src/zuma_rl/revenge_settled_strategic_teacher_v3.py",
        "settled_teacher_v1": root
        / "src/zuma_rl/revenge_settled_strategic_teacher.py",
        "strategic_teacher": root / "src/zuma_rl/revenge_strategic_teacher.py",
        "geometric_teacher": root / "src/zuma_rl/revenge_teacher.py",
        "exact_mask_evaluator": root
        / "tools/evaluate_alphazuma_55_masked_teacher_v3.py",
        "environment": root / "src/zuma_rl/revenge_env.py",
        "core": root / "src/zuma_rl/revenge_core.py",
        "features": root / "src/zuma_rl/revenge_features.py",
        "scope": root / "src/zuma_rl/alphazuma_55.py",
    }
    return {
        "schema": "zuma-rl.alphazuma-55-settled-v3-distillation-preregistration",
        "version": 1,
        "status": "FROZEN_BEFORE_TRAINING",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_id": master["campaign_id"],
        "objective": (
            "Clone the actor-observable V3 policy into one full55 CNN, including "
            "its purposeful wait-and-aim trajectory under exact action masks."
        ),
        "master_preregistration": _artifact(master_path),
        "trainer": _artifact(trainer),
        "implementation": {
            name: _artifact(path) for name, path in implementation_paths.items()
        },
        "teacher_evidence": {
            "unmasked_fresh_55": _artifact(unmasked_result_path),
            "exact_mask_fresh_55": _artifact(masked_result_path),
            "classification": "engineering_only_not_formal_blind",
        },
        "source_migration_receipt": _artifact(source_migration_path),
        "student_source_model": {
            "id": "v11-full55-v1",
            "training_steps": int(migrated["num_timesteps"]),
            "path": str(source_model_path),
            "sha256": source_hash,
        },
        "levels": list(INCLUDED_LEVELS),
        "run": {
            "id": "alphazuma55-settled-v3-distillation-5090",
            "run_dir": str(run_dir),
            "device": device,
            "model_seed": model_seed,
            "training_seed_base": training_seed_base,
            "training_seed_last_consumed": training_seed_last,
            "rounds": 1,
            "parallel_envs": 24,
            "max_ticks": 30_000,
            "reservoir_capacity_by_verb": {
                "wait": 256,
                "fire": 256,
                "swap": 192,
                "hop": 192,
            },
            "sample_stride_by_verb": {
                "wait": 4,
                "fire": 1,
                "swap": 1,
                "hop": 1,
            },
            "target_batch_fraction_by_verb": {
                "wait": 0.70,
                "fire": 0.22,
                "swap": 0.07,
                "hop": 0.01,
            },
            "batch_size": 512,
            "epochs_per_round": 8,
            "checkpoint_interval_epochs": 2,
            "learning_rate": 0.00001,
            "verb_loss_weight": 2.0,
            "max_grad_norm": 0.5,
            "aim_loss_scope": "all_actions",
            "aim_loss_mode": "circular_smoothed",
            "aim_smoothing_sigma": 1.5,
            "aim_distance_weight": 0.5,
            "teacher_execution_probability": 1.0,
            "exact_action_masks_before_every_decision": True,
            "masked_verb_fallback": "wait_with_teacher_aim",
            "verb_balanced_optimization": True,
            "final_runtime_teacher_calls_forbidden": True,
        },
        "authority_boundary": {
            "classification": "engineering_training_only",
            "formal_selection_seed_consumption": False,
            "formal_final_blind_seed_consumption": False,
            "continuous_campaign_seed_consumption": False,
            "formal_candidate_registration_authorized": False,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-preregistration", required=True, type=Path)
    parser.add_argument("--source-model", required=True, type=Path)
    parser.add_argument("--source-migration", required=True, type=Path)
    parser.add_argument("--unmasked-result", required=True, type=Path)
    parser.add_argument("--masked-result", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--device", choices=("cuda:0", "cuda:1"), default="cuda:0")
    parser.add_argument("--training-seed-base", required=True, type=int)
    parser.add_argument("--model-seed", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite preregistration: {output}")
    result = build(
        master_path=args.master_preregistration.expanduser().resolve(strict=True),
        source_model_path=args.source_model.expanduser().resolve(strict=True),
        source_migration_path=args.source_migration.expanduser().resolve(strict=True),
        unmasked_result_path=args.unmasked_result.expanduser().resolve(strict=True),
        masked_result_path=args.masked_result.expanduser().resolve(strict=True),
        run_dir=args.run_dir.expanduser().resolve(),
        device=args.device,
        training_seed_base=args.training_seed_base,
        model_seed=args.model_seed,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(output),
                "sha256": _sha256(output),
                "run": result["run"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
