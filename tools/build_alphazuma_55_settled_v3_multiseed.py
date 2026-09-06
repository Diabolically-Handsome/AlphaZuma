"""Freeze four-round fresh-seed continuation of a reset-aim V3 clone."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools import build_alphazuma_55_settled_v3_distillation as base


SCRIPT_PATH = Path(__file__).resolve()
TRAINER = SCRIPT_PATH.with_name("distill_alphazuma_55_settled_v3_multiseed.py")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-preregistration", required=True, type=Path)
    parser.add_argument("--source-completion", required=True, type=Path)
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
    parent_path = args.parent_preregistration.expanduser().resolve(strict=True)
    parent = base._read_json(parent_path)
    completion_path = args.source_completion.expanduser().resolve(strict=True)
    completion = base._read_json(completion_path)
    final = completion.get("final_model", {})
    source_path = Path(str(final.get("path", ""))).resolve(strict=True)
    if (
        parent.get("status") != "FROZEN_BEFORE_TRAINING"
        or parent.get("run", {}).get("aim_head_reset", {}).get("enabled") is not True
        or completion.get("status") != "COMPLETE"
        or completion.get("aim_head_reset", {}).get("status")
        != "APPLIED_BEFORE_OPTIMIZER_REPLACEMENT_AND_COLLECTION"
        or final.get("sha256") != base._sha256(source_path)
        or Path(str(parent["run"]["run_dir"])).resolve() != completion_path.parent
    ):
        raise ValueError("parent does not bind a completed reset-aim source")
    result = copy.deepcopy(parent)
    result["schema"] = "zuma-rl.alphazuma-55-settled-v3-multiseed-preregistration"
    result["created_utc"] = datetime.now(timezone.utc).isoformat()
    result["objective"] = (
        "Improve seed generalization by aggregating four fresh 55-level teacher "
        "rounds before joint optimization of the reset-aim neural policy."
    )
    result["trainer"] = base._artifact(TRAINER)
    result["implementation"]["multiseed_builder"] = base._artifact(SCRIPT_PATH)
    result["implementation"]["parent_preregistration"] = base._artifact(parent_path)
    result["implementation"]["source_completion"] = base._artifact(completion_path)
    result.pop("source_migration_receipt", None)
    result["student_source_model"] = {
        "id": "settled-v3-reset-aim-v2-final",
        "training_steps": int(final["num_timesteps"]),
        "path": str(source_path),
        "sha256": str(final["sha256"]),
    }
    result["source_completion"] = base._artifact(completion_path)
    result["continuation_lineage"] = {
        "parent_preregistration": base._artifact(parent_path),
        "parent_completion": base._artifact(completion_path),
        "source_aim_head_was_reset": True,
        "new_aim_head_reset": False,
    }
    result["development_disclosure"] = {
        "motivated_by_single_round_seed_generalization_risk": True,
        "source_training_collection_wins": int(completion["collection"]["wins"]),
        "source_training_collection_truncations": int(
            completion["collection"]["truncations"]
        ),
        "source_final_mean_loss": float(
            completion["optimization"]["epochs"][-1]["mean_loss"]
        ),
        "formal_evidence": False,
    }
    rounds = 4
    run = result["run"]
    run.update(
        {
            "id": "alphazuma55-settled-v3-multiseed-5090",
            "run_dir": str(args.run_dir.expanduser().resolve()),
            "device": args.device,
            "model_seed": args.model_seed,
            "training_seed_base": args.training_seed_base,
            "training_seed_last_consumed": args.training_seed_base + rounds * 55 - 1,
            "rounds": rounds,
            "parallel_envs": 24,
            "epochs_per_round": 16,
            "checkpoint_interval_epochs": 4,
            "learning_rate": 0.00003,
            "verb_loss_weight": 1.0,
            "max_grad_norm": 1.0,
            "dataset_aggregation": "four_fresh_teacher_rounds_before_joint_optimization",
            "aim_head_reset": {
                "enabled": False,
                "reason": "source_model_already_has_reset_and_trained_aim_head",
            },
        }
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
                "sha256": base._sha256(output),
                "run": run,
                "formal_seed_consumption": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
