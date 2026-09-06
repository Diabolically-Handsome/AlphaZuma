"""Freeze the recovered settled-V3 reset-aim clone after startup failure."""

from __future__ import annotations

import argparse
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
TRAINER = SCRIPT_PATH.with_name("distill_alphazuma_55_settled_v3_reset_aim_v2.py")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-preregistration", required=True, type=Path)
    parser.add_argument("--source-model", required=True, type=Path)
    parser.add_argument("--source-migration", required=True, type=Path)
    parser.add_argument("--unmasked-result", required=True, type=Path)
    parser.add_argument("--masked-result", required=True, type=Path)
    parser.add_argument("--prior-completion", required=True, type=Path)
    parser.add_argument("--startup-incident", required=True, type=Path)
    parser.add_argument("--failed-preregistration", required=True, type=Path)
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
    prior_path = args.prior_completion.expanduser().resolve(strict=True)
    prior = base._read_json(prior_path)
    last_epoch = prior.get("optimization", {}).get("epochs", [])[-1]
    if (
        prior.get("status") != "COMPLETE"
        or int(last_epoch.get("epoch", -1)) != 24
        or float(last_epoch.get("mean_aim_within_one_accuracy", 1.0)) >= 0.20
        or float(last_epoch.get("mean_loss", 0.0)) <= 6.0
    ):
        raise ValueError("prior high-fidelity run does not trigger aim-head reset")
    incident_path = args.startup_incident.expanduser().resolve(strict=True)
    incident = base._read_json(incident_path)
    failed_prereg_path = args.failed_preregistration.expanduser().resolve(strict=True)
    failed_prereg = base._read_json(failed_prereg_path)
    evidence = incident.get("evidence", {})
    if (
        incident.get("status") != "FAILED_BEFORE_RUN_INITIALIZATION"
        or evidence.get("run_dir_exists_after_exit") is not False
        or evidence.get("environment_collection_started") is not False
        or evidence.get("training_seed_consumption") is not False
        or incident.get("preregistration", {}).get("sha256")
        != base._sha256(failed_prereg_path)
        or failed_prereg.get("run", {}).get("training_seed_base") == args.training_seed_base
    ):
        raise ValueError("startup incident does not authorize this recovery")
    result = base.build(
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
    result["created_utc"] = datetime.now(timezone.utc).isoformat()
    result["objective"] = (
        "Recover the precommitted aim-head reset after a seed-free startup "
        "wrapper failure, using a new run and new engineering seeds."
    )
    original_trainer = result["trainer"]
    result["trainer"] = base._artifact(TRAINER)
    result["implementation"]["base_settled_v3_distiller"] = original_trainer
    result["implementation"]["variant_builder"] = base._artifact(SCRIPT_PATH)
    result["implementation"]["prior_high_fidelity_completion"] = base._artifact(
        prior_path
    )
    result["implementation"]["failed_startup_incident"] = base._artifact(
        incident_path
    )
    result["implementation"]["failed_preregistration"] = base._artifact(
        failed_prereg_path
    )
    result["development_disclosure"] = {
        "selection_conditioned_on_prior_training_metrics": True,
        "prior_epoch": int(last_epoch["epoch"]),
        "prior_mean_loss": float(last_epoch["mean_loss"]),
        "prior_mean_aim_within_one_accuracy": float(
            last_epoch["mean_aim_within_one_accuracy"]
        ),
        "formal_evidence": False,
    }
    result["startup_recovery"] = {
        "failed_before_run_initialization": True,
        "failed_seed_consumption": False,
        "failed_preregistration": base._artifact(failed_prereg_path),
        "incident": base._artifact(incident_path),
        "reuse_failed_training_seed_range": False,
    }
    run = result["run"]
    run.update(
        {
            "id": "alphazuma55-settled-v3-reset-aim-v2-5090",
            "epochs_per_round": 24,
            "checkpoint_interval_epochs": 4,
            "learning_rate": 0.00003,
            "verb_loss_weight": 1.0,
            "max_grad_norm": 1.0,
            "aim_head_reset": {
                "enabled": True,
                "rows": "aim_component_only",
                "weight": "zero",
                "bias": "zero",
                "verb_head_unchanged": True,
                "feature_extractor_unchanged_before_training": True,
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
                "startup_recovery": result["startup_recovery"],
                "run": run,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
