"""Freeze a small engineering probe for polar aim-head re-anchoring."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MASTER = (
    PROJECT_ROOT
    / "diagnostics/alphazuma-55-weekend-s81081401-preregistration-v1.json"
)
DEFAULT_SOURCE_COMPLETION = Path(
    "/mnt/d/ZumaTraining/alphazuma-55-settled-v3-multiseed-s99081525-v1/"
    "completion.json"
)
DEFAULT_CANONICAL_VALIDATION = (
    PROJECT_ROOT
    / "diagnostics/alphazuma-55-settled-v3-multiseed-validation-"
    "s99081526-preregistration-v1.json"
)
DEFAULT_RUNNER = PROJECT_ROOT / "tools/create_alphazuma_55_polar_reanchor_variants.py"
DEFAULT_MATERIALIZER = (
    PROJECT_ROOT / "tools/materialize_alphazuma_55_polar_reanchor_probe.py"
)
DEFAULT_EVALUATOR = PROJECT_ROOT / "tools/evaluate_zero_shot_multilevel_v2.py"
DEFAULT_PLAN_OUTPUT = (
    PROJECT_ROOT
    / "diagnostics/alphazuma-55-polar-reanchor-probe-s99081527-plan-v1.json"
)

PROBE_LEVELS = (
    "Jungle2",
    "Jungle9",
    "village6",
    "city4",
    "city10",
    "Coast10",
    "grotto10",
    "volcano1",
    "volcano5",
    "volcano8",
    "volcano9",
    "volcano10",
)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def _write_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-preregistration", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--source-completion", type=Path, default=DEFAULT_SOURCE_COMPLETION)
    parser.add_argument(
        "--canonical-validation", type=Path, default=DEFAULT_CANONICAL_VALIDATION
    )
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument("--materializer", type=Path, default=DEFAULT_MATERIALIZER)
    parser.add_argument("--evaluator", type=Path, default=DEFAULT_EVALUATOR)
    parser.add_argument("--output", type=Path, default=DEFAULT_PLAN_OUTPUT)
    parser.add_argument("--base-seed", type=int, default=1_550_000_700)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen plan: {output}")

    master_path = args.master_preregistration.expanduser().resolve(strict=True)
    source_completion_path = args.source_completion.expanduser().resolve(strict=True)
    canonical_path = args.canonical_validation.expanduser().resolve(strict=True)
    runner_path = args.runner.expanduser().resolve(strict=True)
    materializer_path = args.materializer.expanduser().resolve(strict=True)
    evaluator_path = args.evaluator.expanduser().resolve(strict=True)
    master = _read_json(master_path)
    completion = _read_json(source_completion_path)
    canonical = _read_json(canonical_path)
    if completion.get("status") != "COMPLETE":
        raise ValueError("source multiseed completion is not COMPLETE")
    if completion.get("formal_seed_consumption") is not False:
        raise ValueError("source completion does not preserve the formal embargo")
    source_model = dict(completion["final_model"])
    source_model_path = Path(str(source_model["path"])).resolve(strict=True)
    if source_model.get("sha256") != _sha256(source_model_path):
        raise ValueError("source model hash differs from completion")
    if canonical.get("status") != "FROZEN_BEFORE_EVALUATION":
        raise ValueError("canonical validation contract is not frozen")

    levels_by_id = {str(row["id"]): dict(row) for row in canonical["levels"]}
    levels = [levels_by_id[level_id] for level_id in PROBE_LEVELS]
    if len(levels) != len(PROBE_LEVELS):
        raise ValueError("probe level inventory is incomplete")
    registry = master["seed_registry"]["training_validation"]
    base_seed = int(args.base_seed)
    last_seed = base_seed + len(levels) - 1
    if not int(registry["first"]) <= base_seed <= last_seed <= int(registry["last"]):
        raise ValueError("probe seeds escaped the training-validation registry")

    plan = {
        "schema": "zuma-rl.alphazuma-55-polar-reanchor-probe-plan",
        "version": 1,
        "status": "FROZEN_BEFORE_VARIANT_CREATION",
        "created_utc": _utc_now(),
        "campaign_id": "alphazuma-55-weekend-s81081401-v1",
        "objective": (
            "Test whether the actor-visible 180-bin polar basis can repair the "
            "weak learned aim head without changing the verb policy or environment."
        ),
        "master_preregistration": {
            "path": str(master_path),
            "sha256": _sha256(master_path),
        },
        "source_completion": {
            "path": str(source_completion_path),
            "sha256": _sha256(source_completion_path),
        },
        "source_model": {
            "id": "settled-v3-multiseed-final",
            "path": str(source_model_path),
            "sha256": _sha256(source_model_path),
            "training_steps": int(source_model["num_timesteps"]),
        },
        "canonical_validation": {
            "path": str(canonical_path),
            "sha256": _sha256(canonical_path),
        },
        "implementation": {
            "runner": {"path": str(runner_path), "sha256": _sha256(runner_path)},
            "materializer": {
                "path": str(materializer_path),
                "sha256": _sha256(materializer_path),
            },
            "evaluator": {
                "path": str(evaluator_path),
                "sha256": _sha256(evaluator_path),
            },
        },
        "variants": [
            {
                "id": "polar-reset-identity-scale-5",
                "mode": "reset_identity",
                "scale": 5.0,
            },
            {
                "id": "polar-additive-identity-scale-1",
                "mode": "additive_identity",
                "scale": 1.0,
            },
            {
                "id": "polar-additive-identity-scale-2",
                "mode": "additive_identity",
                "scale": 2.0,
            },
        ],
        "variant_output_root": (
            "/mnt/d/ZumaTraining/alphazuma-55-polar-reanchor-variants-"
            "s99081527-v1"
        ),
        "validation": {
            "levels": levels,
            "attempts_per_level": 1,
            "seed_plan": {
                "registry": "training_validation",
                "base_seed": base_seed,
                "last_seed": last_seed,
            },
            "execution": {
                "output_root": (
                    "/mnt/d/ZumaTraining/alphazuma-55-polar-reanchor-probe-"
                    "s99081527-v1"
                ),
                "shard_count": 1,
                "devices": ["cuda:0"],
                "parallel_envs_per_shard": 24,
            },
            "manifest_output": str(
                PROJECT_ROOT
                / "diagnostics/alphazuma-55-polar-reanchor-probe-"
                "s99081527-models-v1.json"
            ),
            "preregistration_output": str(
                PROJECT_ROOT
                / "diagnostics/alphazuma-55-polar-reanchor-probe-"
                "s99081527-preregistration-v1.json"
            ),
        },
        "authority_boundary": {
            "classification": "posthoc_engineering_probe_only",
            "formal_candidate_registration_authorized": False,
            "formal_selection_seed_consumption": False,
            "formal_final_blind_seed_consumption": False,
            "continuous_campaign_seed_consumption": False,
        },
    }
    _write_exclusive(output, plan)
    print(
        json.dumps(
            {
                "status": plan["status"],
                "path": str(output),
                "sha256": _sha256(output),
                "levels": len(levels),
                "variants": len(plan["variants"]),
                "seed_plan": plan["validation"]["seed_plan"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
