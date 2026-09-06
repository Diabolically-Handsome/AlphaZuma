"""Freeze a specialist-to-single-policy AlphaZuma 55 distillation run."""

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
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def build_preregistration(
    *,
    master_path: Path,
    teacher_map_path: Path,
    source_model_path: Path,
    source_id: str,
    source_training_steps: int,
    source_migration_receipt: Path,
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
    teacher_map = _read_json(teacher_map_path)
    if teacher_map.get("status") != "FROZEN":
        raise ValueError("teacher map is not frozen")
    if [row["level_id"] for row in teacher_map["levels"]] != list(INCLUDED_LEVELS):
        raise ValueError("teacher map scope differs from AlphaZuma 55")
    migration = _read_json(source_migration_receipt)
    if (
        migration.get("status") != "PASS"
        or migration["migrated_model"]["sha256"] != _sha256(source_model_path)
    ):
        raise ValueError("source migration receipt differs from student source")
    if source_training_steps < 0:
        raise ValueError("source training steps cannot be negative")
    if run_dir.exists():
        raise FileExistsError(f"distillation run already exists: {run_dir}")

    rounds = 2
    consumed_last = training_seed_base + rounds * len(INCLUDED_LEVELS) - 1
    registry = master["seed_registry"]["training"]
    if not (
        int(registry["first"]) <= training_seed_base
        and consumed_last <= int(registry["last"])
    ):
        raise ValueError("distillation seeds exceed the training registry")

    root = Path(__file__).resolve().parents[1]
    distiller = root / "tools" / "distill_alphazuma_55.py"
    implementation_paths = {
        "behavior_clone_batch_primitive": root
        / "tools"
        / "pretrain_maskable_ppo_teacher.py",
        "teacher": root / "src" / "zuma_rl" / "revenge_teacher.py",
        "environment": root / "src" / "zuma_rl" / "revenge_env.py",
        "core": root / "src" / "zuma_rl" / "revenge_core.py",
        "features": root / "src" / "zuma_rl" / "revenge_features.py",
        "scope": root / "src" / "zuma_rl" / "alphazuma_55.py",
        "source_migration_receipt": source_migration_receipt,
    }
    return {
        "schema": "zuma-rl.alphazuma-55-distillation-preregistration",
        "version": 1,
        "status": "FROZEN_BEFORE_TRAINING",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_id": master["campaign_id"],
        "master_preregistration": {
            "path": str(master_path),
            "sha256": _sha256(master_path),
        },
        "trainer": {"path": str(distiller), "sha256": _sha256(distiller)},
        "implementation": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in implementation_paths.items()
        },
        "teacher_map": {
            "path": str(teacher_map_path),
            "sha256": _sha256(teacher_map_path),
        },
        "student_source_model": {
            "id": source_id,
            "training_steps": source_training_steps,
            "path": str(source_model_path),
            "sha256": _sha256(source_model_path),
        },
        "run": {
            "id": "alphazuma55-specialist-distillation-5090",
            "run_dir": str(run_dir),
            "device": device,
            "model_seed": model_seed,
            "training_seed_base": training_seed_base,
            "training_seed_last_consumed": consumed_last,
            "rounds": rounds,
            "samples_per_level_per_round": 192,
            "sample_stride": 8,
            "batch_size": 512,
            "epochs_per_round": 3,
            "parallel_envs": 24,
            "max_ticks": 12_000,
            "teacher_execution_probability_after_round_zero": 0.25,
            "learning_rate": 0.00002,
            "verb_loss_weight": 2.0,
            "max_grad_norm": 0.5,
            "geometric_teacher_for_unproven_levels": True,
            "geometric_hop_override_on_all_dual_position_levels": True,
            "final_runtime_teacher_calls_forbidden": True,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-preregistration", required=True, type=Path)
    parser.add_argument("--teacher-map", required=True, type=Path)
    parser.add_argument("--source-model", required=True, type=Path)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--source-training-steps", required=True, type=int)
    parser.add_argument("--source-migration-receipt", required=True, type=Path)
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
    result = build_preregistration(
        master_path=args.master_preregistration.expanduser().resolve(strict=True),
        teacher_map_path=args.teacher_map.expanduser().resolve(strict=True),
        source_model_path=args.source_model.expanduser().resolve(strict=True),
        source_id=args.source_id,
        source_training_steps=args.source_training_steps,
        source_migration_receipt=args.source_migration_receipt.expanduser().resolve(
            strict=True
        ),
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
