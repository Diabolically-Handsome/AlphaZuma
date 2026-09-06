"""Build hash-bound configured successor evaluation contracts.

Unlike the first DAgger-only successor builder, this variant reads the seed
range and execution topology from the frozen successor master.  It therefore
cannot silently reuse another campaign's seeds or GPU layout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools import build_alphazuma_55_eval_contract as legacy


SCRIPT_PATH = Path(__file__).resolve()
EXPECTED_ATTEMPTS = {"final_blind": 8, "continuous": 4}
REGISTRY_BY_STAGE = {
    "final_blind": "final_blind",
    "continuous": "continuous_campaign_challenge",
}


def _validated_stage_spec(
    *, stage: str, base_seed: int, attempts_per_level: int
) -> dict[str, int]:
    if stage not in EXPECTED_ATTEMPTS:
        raise ValueError("configured successor stage must be final_blind or continuous")
    attempts = int(attempts_per_level)
    if attempts != EXPECTED_ATTEMPTS[stage]:
        raise ValueError(
            f"configured successor {stage} requires {EXPECTED_ATTEMPTS[stage]} attempts"
        )
    seed = int(base_seed)
    if not 0 <= seed <= (2**32 - 1):
        raise ValueError("configured successor seed is outside uint32")
    last = seed + 55 * attempts - 1
    if last > 2**32 - 1:
        raise ValueError("configured successor seed matrix exceeds uint32")
    return {"base_seed": seed, "attempts": attempts}


def build_successor_contracts(
    *,
    master_path: Path,
    original_root: Path,
    evaluator_path: Path,
    stage: str,
    base_seed: int,
    attempts_per_level: int,
    output_root: Path,
    models: list[dict[str, Any]],
    parallel_envs_per_shard: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    master_path = master_path.resolve(strict=True)
    master = legacy._read_json(master_path)
    if (
        master.get("schema")
        != "zuma-rl.alphazuma-55-configured-successor-master-preregistration"
        or master.get("version") != 1
        or master.get("status") != "FROZEN_BEFORE_SUCCESSOR_FINAL_BLIND"
    ):
        raise ValueError("unexpected configured successor master preregistration")
    stage_spec = _validated_stage_spec(
        stage=stage,
        base_seed=base_seed,
        attempts_per_level=attempts_per_level,
    )
    if not models or len({str(model.get("id", "")) for model in models}) != len(
        models
    ):
        raise ValueError("configured successor model ids must be non-empty and unique")
    for index, model in enumerate(models):
        model_id = str(model.get("id", ""))
        model_path = Path(str(model.get("path", ""))).expanduser().resolve(
            strict=True
        )
        if not model_id:
            raise ValueError(f"configured successor models[{index}] has an empty id")
        if int(model.get("training_steps", -1)) < 0:
            raise ValueError(
                f"configured successor models[{index}] has invalid training steps"
            )
        if model.get("sha256") != legacy._sha256(model_path):
            raise ValueError(f"configured successor models[{index}] hash differs")
        declared_bytes = model.get("bytes")
        if declared_bytes is not None and int(declared_bytes) != model_path.stat().st_size:
            raise ValueError(f"configured successor models[{index}] byte count differs")

    registry_name = REGISTRY_BY_STAGE[stage]
    registry = master["seed_registry"][registry_name]
    last_seed = stage_spec["base_seed"] + 55 * stage_spec["attempts"] - 1
    if not (
        int(registry["first"]) == stage_spec["base_seed"]
        and int(registry["last"]) == last_seed
    ):
        raise ValueError("configured successor master seed registry differs from matrix")

    execution = master.get("execution", {})
    devices = list(execution.get("devices", []))
    shard_count = int(execution.get("shard_count", -1))
    if not (
        shard_count == 2
        and len(devices) == shard_count
        and all(device in {"cuda:0", "cuda:1"} for device in devices)
        and int(execution.get("parallel_envs_per_shard", -1))
        == int(parallel_envs_per_shard)
    ):
        raise ValueError("configured successor execution topology differs from master")

    original = dict(legacy.STAGES[stage])
    original_reader = legacy._read_json

    def _read_for_legacy(path: Path) -> dict[str, Any]:
        value = original_reader(path)
        if Path(path).resolve() == master_path:
            value = dict(value)
            value["schema"] = "zuma-rl.alphazuma-55-weekend-master-preregistration"
        return value

    legacy.STAGES[stage] = dict(stage_spec)
    legacy._read_json = _read_for_legacy
    try:
        manifest, preregistration = legacy.build_contracts(
            master_path=master_path,
            original_root=original_root.resolve(strict=True),
            evaluator_path=evaluator_path.resolve(strict=True),
            stage=stage,
            output_root=output_root.resolve(),
            models=models,
            parallel_envs_per_shard=parallel_envs_per_shard,
        )
    finally:
        legacy.STAGES[stage] = original
        legacy._read_json = original_reader

    preregistration["execution"]["devices"] = devices
    binding = {
        "builder": {
            "path": str(SCRIPT_PATH),
            "sha256": legacy._sha256(SCRIPT_PATH),
        },
        "stage": stage,
        "registry": registry_name,
        "base_seed": stage_spec["base_seed"],
        "last_seed": last_seed,
        "attempts_per_level": stage_spec["attempts"],
        "formal_seed_consumption_before_contract": False,
    }
    manifest["configured_successor_contract"] = binding
    preregistration["configured_successor_contract"] = binding
    preregistration["seed_plan"][
        "all_seeds_frozen_before_policy_inference"
    ] = True
    if (
        int(preregistration["attempts_per_level"]) != stage_spec["attempts"]
        or int(preregistration["total_attempts"])
        != 55 * stage_spec["attempts"]
        or int(preregistration["seed_plan"]["base_seed"])
        != stage_spec["base_seed"]
        or int(preregistration["seed_plan"]["last_seed"]) != last_seed
        or preregistration["seed_plan"].get(
            "all_seeds_frozen_before_policy_inference"
        )
        is not True
        or preregistration["execution"].get("devices") != devices
    ):
        raise RuntimeError("materialized configured successor matrix differs from binding")
    return manifest, preregistration


def _model_spec(value: str) -> dict[str, Any]:
    return legacy._model_spec(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-preregistration", required=True, type=Path)
    parser.add_argument("--original-root", required=True, type=Path)
    parser.add_argument("--evaluator", required=True, type=Path)
    parser.add_argument("--stage", choices=tuple(EXPECTED_ATTEMPTS), required=True)
    parser.add_argument("--base-seed", required=True, type=int)
    parser.add_argument("--attempts-per-level", required=True, type=int)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--manifest-output", required=True, type=Path)
    parser.add_argument("--preregistration-output", required=True, type=Path)
    parser.add_argument("--model", action="append", type=_model_spec, required=True)
    parser.add_argument("--parallel-envs-per-shard", type=int, default=12)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_path = args.manifest_output.expanduser().resolve()
    preregistration_path = args.preregistration_output.expanduser().resolve()
    for path in (manifest_path, preregistration_path):
        if path.exists():
            raise FileExistsError(
                f"refusing to overwrite configured successor contract: {path}"
            )
    manifest, preregistration = build_successor_contracts(
        master_path=args.master_preregistration.expanduser(),
        original_root=args.original_root.expanduser(),
        evaluator_path=args.evaluator.expanduser(),
        stage=args.stage,
        base_seed=args.base_seed,
        attempts_per_level=args.attempts_per_level,
        output_root=args.output_root.expanduser(),
        models=args.model,
        parallel_envs_per_shard=args.parallel_envs_per_shard,
    )
    legacy._write_json_exclusive(manifest_path, manifest)
    legacy._write_json_exclusive(preregistration_path, preregistration)
    print(
        json.dumps(
            {
                "status": "FROZEN",
                "stage": args.stage,
                "manifest": {
                    "path": str(manifest_path),
                    "sha256": legacy._sha256(manifest_path),
                },
                "preregistration": {
                    "path": str(preregistration_path),
                    "sha256": legacy._sha256(preregistration_path),
                },
                "models": len(manifest["models"]),
                "attempts": preregistration["total_attempts"],
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
