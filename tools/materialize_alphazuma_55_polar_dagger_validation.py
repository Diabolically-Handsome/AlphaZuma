"""Bind frozen polar DAgger round models into paired full55 validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools import build_alphazuma_55_training_validation as contracts
from tools.build_alphazuma_55_eval_contract import _write_json_exclusive


SCRIPT_PATH = Path(__file__).resolve()
EXPECTED_ROUNDS = (0, 1, 2)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _bound_path(reference: Any, name: str) -> Path:
    _require(isinstance(reference, dict), f"{name} must be an object")
    path = Path(str(reference.get("path", ""))).resolve(strict=True)
    _require(
        reference.get("sha256") == contracts._sha256(path),
        f"{name} hash mismatch",
    )
    return path


def validate_plan_static(plan_path: Path, expected_sha256: str) -> dict[str, Any]:
    plan_path = plan_path.resolve(strict=True)
    _require(contracts._sha256(plan_path) == expected_sha256, "plan hash mismatch")
    plan = _read(plan_path)
    _require(
        plan.get("schema")
        == "zuma-rl.alphazuma-55-polar-dagger-validation-plan"
        and plan.get("version") == 1
        and plan.get("status") == "FROZEN_DURING_TARGET_TRAINING",
        "unexpected polar DAgger validation plan",
    )
    _bound_path(plan["master_preregistration"], "master preregistration")
    training_path = _bound_path(
        plan["training_preregistration"], "training preregistration"
    )
    training = _read(training_path)
    _require(
        training.get("schema")
        == "zuma-rl.alphazuma-55-polar-dagger-preregistration"
        and tuple(
            training["frozen_engineering_validation"]["models"]
        )
        == (
            "source-final",
            "dagger-round-00",
            "dagger-round-01",
            "dagger-round-02",
        ),
        "plan does not bind the frozen DAgger training contract",
    )
    _bound_path(plan["dagger_plan"], "DAgger plan")
    for name, reference in plan["implementation"].items():
        implementation = _bound_path(reference, f"implementation {name}")
        if name == "materializer":
            _require(
                implementation == SCRIPT_PATH,
                "plan binds another materializer",
            )
    matrix = plan["matrix"]
    _require(
        int(matrix["levels"]) == 55
        and int(matrix["models"]) == 4
        and int(matrix["expected_attempts"]) == 220
        and int(matrix["last_seed"]) == int(matrix["base_seed"]) + 54
        and int(matrix["max_ticks"]) == 30_000
        and matrix["paired_models_share_identical_task_seeds"] is True,
        "invalid polar DAgger validation matrix",
    )
    _require(
        plan["authority_boundary"]["formal_seed_consumption"] is False
        and plan["authority_boundary"][
            "current_campaign_candidate_authority"
        ]
        is False,
        "plan claims formal authority",
    )
    return plan


def materialize(plan_path: Path, expected_sha256: str) -> tuple[Path, Path]:
    plan_path = plan_path.resolve(strict=True)
    plan = validate_plan_static(plan_path, expected_sha256)
    completion_path = Path(plan["target"]["expected_completion"]).resolve(
        strict=True
    )
    completion = _read(completion_path)
    _require(
        completion.get("schema")
        == "zuma-rl.alphazuma-55-polar-dagger-completion"
        and completion.get("status") == "COMPLETE"
        and completion.get("policy_architecture") == "entity_polar"
        and completion.get("formal_seed_consumption") is False
        and completion.get("formal_candidate_authority") is False,
        "polar DAgger completion is invalid",
    )
    training = _read(
        Path(plan["training_preregistration"]["path"]).resolve(strict=True)
    )
    source_completion_path = Path(
        training["source"]["completion_path"]
    ).resolve(strict=True)
    source_completion = _read(source_completion_path)
    source_model = Path(training["source"]["model_path"]).resolve(strict=True)
    source_digest = contracts._sha256(source_model)
    _require(
        source_digest == training["source"]["model_sha256"]
        and source_completion["final_model"]["sha256"] == source_digest,
        "source model hash changed before validation",
    )
    models = [
        {
            "id": "polar-source-final",
            "training_steps": int(
                source_completion["final_model"]["num_timesteps"]
            ),
            "path": str(source_model),
            "sha256": source_digest,
        }
    ]
    round_by_index = {
        int(row["round_index"]): row for row in completion["rounds"]
    }
    expected_paths = tuple(
        Path(value).resolve() for value in plan["target"]["round_model_paths"]
    )
    cumulative_steps = int(source_completion["final_model"]["num_timesteps"])
    for round_index, expected_path in zip(
        EXPECTED_ROUNDS, expected_paths, strict=True
    ):
        _require(round_index in round_by_index, "DAgger completion misses a round")
        row = round_by_index[round_index]
        path = Path(str(row["model"]["path"])).resolve(strict=True)
        _require(path == expected_path, f"round {round_index} path differs")
        digest = contracts._sha256(path)
        _require(row["model"]["sha256"] == digest, "round model hash mismatch")
        cumulative_steps += sum(
            int(episode["steps"]) for episode in row["collection"]["episodes"]
        )
        models.append(
            {
                "id": f"polar-dagger-round-{round_index:02d}",
                "training_steps": cumulative_steps,
                "path": str(path),
                "sha256": digest,
            }
        )

    outputs = plan["outputs"]
    manifest_path = Path(outputs["models_manifest"]).resolve()
    preregistration_path = Path(outputs["evaluation_preregistration"]).resolve()
    run_root = Path(outputs["run_root"]).resolve()
    for path in (manifest_path, preregistration_path, run_root):
        _require(not path.exists(), f"validation output already exists: {path}")
    master_path = Path(plan["master_preregistration"]["path"]).resolve(
        strict=True
    )
    evaluator_path = Path(plan["implementation"]["evaluator"]["path"]).resolve(
        strict=True
    )
    manifest, preregistration = contracts.build_training_validation_contracts(
        master_path=master_path,
        original_root=Path(plan["original_root"]).resolve(strict=True),
        evaluator_path=evaluator_path,
        output_root=run_root,
        models=models,
        base_seed=int(plan["matrix"]["base_seed"]),
        device=str(plan["execution"]["device"]),
        parallel_envs=int(plan["execution"]["parallel_envs"]),
    )
    preregistration["environment"]["base_config"]["max_ticks"] = int(
        plan["matrix"]["max_ticks"]
    )
    preregistration["diagnostic_contract"]["max_ticks"] = int(
        plan["matrix"]["max_ticks"]
    )
    plan_reference = {"path": str(plan_path), "sha256": expected_sha256}
    completion_reference = {
        "path": str(completion_path),
        "sha256": contracts._sha256(completion_path),
    }
    manifest["polar_dagger_validation_plan"] = plan_reference
    manifest["target_completion"] = completion_reference
    preregistration["polar_dagger_validation_plan"] = plan_reference
    preregistration["target_completion"] = completion_reference
    _require(
        int(preregistration["seed_plan"]["base_seed"])
        == int(plan["matrix"]["base_seed"])
        and int(preregistration["seed_plan"]["last_seed"])
        == int(plan["matrix"]["last_seed"]),
        "materialized seed matrix differs from plan",
    )
    _write_json_exclusive(manifest_path, manifest)
    _write_json_exclusive(preregistration_path, preregistration)
    return manifest_path, preregistration_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--expected-plan-sha256", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_path, preregistration_path = materialize(
        args.plan.expanduser().resolve(strict=True),
        str(args.expected_plan_sha256),
    )
    print(
        json.dumps(
            {
                "status": "FROZEN_AFTER_TARGET_MODEL",
                "manifest": {
                    "path": str(manifest_path),
                    "sha256": contracts._sha256(manifest_path),
                },
                "preregistration": {
                    "path": str(preregistration_path),
                    "sha256": contracts._sha256(preregistration_path),
                },
                "models": 4,
                "expected_attempts": 220,
                "formal_seed_consumption": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
