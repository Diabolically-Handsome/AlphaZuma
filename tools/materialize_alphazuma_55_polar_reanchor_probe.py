"""Bind completed polar re-anchor variants into their frozen probe."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--completion", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan_path = args.plan.expanduser().resolve(strict=True)
    completion_path = args.completion.expanduser().resolve(strict=True)
    if _sha256(plan_path) != args.expected_plan_sha256:
        raise ValueError("plan hash differs from the expected frozen hash")
    plan = _read_json(plan_path)
    completion = _read_json(completion_path)
    materializer = plan["implementation"]["materializer"]
    if Path(str(materializer["path"])).resolve() != Path(__file__).resolve():
        raise ValueError("plan binds another materializer")
    if materializer["sha256"] != _sha256(Path(__file__).resolve()):
        raise ValueError("materializer bytes differ from the plan")
    if completion.get("status") != "COMPLETE":
        raise ValueError("variant completion is not COMPLETE")
    if completion.get("formal_seed_consumption") is not False:
        raise ValueError("variant completion crossed the formal boundary")
    if completion["plan"]["sha256"] != _sha256(plan_path):
        raise ValueError("variant completion binds another plan")

    expected_specs = {row["id"]: row for row in plan["variants"]}
    variants = completion.get("variants")
    if not isinstance(variants, list) or {row["id"] for row in variants} != set(
        expected_specs
    ):
        raise ValueError("variant completion does not match the frozen variants")
    for row in variants:
        path = Path(str(row["path"])).resolve(strict=True)
        if row["sha256"] != _sha256(path):
            raise ValueError(f"variant hash differs: {row['id']}")
        spec = expected_specs[row["id"]]
        transform = row["transform"]
        if transform["mode"] != spec["mode"] or float(transform["scale"]) != float(
            spec["scale"]
        ):
            raise ValueError(f"variant transform differs: {row['id']}")
        if not transform["verb_rows_bitwise_unchanged"]:
            raise ValueError(f"variant changed verb rows: {row['id']}")

    source = dict(plan["source_model"])
    models = [source] + [
        {
            "id": str(row["id"]),
            "training_steps": int(row["training_steps"]),
            "path": str(Path(str(row["path"])).resolve()),
            "sha256": str(row["sha256"]),
        }
        for row in variants
    ]
    validation = plan["validation"]
    manifest_path = Path(str(validation["manifest_output"])).resolve()
    prereg_path = Path(str(validation["preregistration_output"])).resolve()
    for path in (manifest_path, prereg_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite frozen contract: {path}")

    canonical_path = Path(str(plan["canonical_validation"]["path"])).resolve(
        strict=True
    )
    if plan["canonical_validation"]["sha256"] != _sha256(canonical_path):
        raise ValueError("canonical validation bytes changed")
    preregistration = copy.deepcopy(_read_json(canonical_path))
    preregistration.update(
        {
            "created_utc": _utc_now(),
            "stage": "engineering_polar_reanchor_probe",
            "model": source,
            "levels": validation["levels"],
            "attempts_per_level": int(validation["attempts_per_level"]),
            "total_attempts": len(validation["levels"])
            * int(validation["attempts_per_level"]),
            "seed_plan": validation["seed_plan"],
            "execution": validation["execution"],
            "evaluator": plan["implementation"]["evaluator"],
            "polar_reanchor_probe": {
                "plan": {"path": str(plan_path), "sha256": _sha256(plan_path)},
                "completion": {
                    "path": str(completion_path),
                    "sha256": _sha256(completion_path),
                },
                "posthoc_engineering_only": True,
            },
        }
    )
    preregistration["diagnostic_contract"] = {
        "paired_models_share_identical_task_seeds": True,
        "formal_selection_authority": False,
        "training_recipe_change_authority": False,
        "formal_seed_ranges_consumed": False,
        "max_ticks": int(preregistration["environment"]["base_config"]["max_ticks"]),
        "probe_winner_cannot_enter_original_formal_campaign": True,
    }
    manifest = {
        "schema": "zuma-rl.zero-shot-models-manifest",
        "version": 1,
        "status": "FROZEN",
        "created_utc": _utc_now(),
        "stage": "engineering_polar_reanchor_probe",
        "purpose": (
            "Posthoc paired engineering smoke test; no original formal selection authority."
        ),
        "models": models,
        "plan": {"path": str(plan_path), "sha256": _sha256(plan_path)},
        "completion": {
            "path": str(completion_path),
            "sha256": _sha256(completion_path),
        },
    }
    _write_exclusive(manifest_path, manifest)
    _write_exclusive(prereg_path, preregistration)
    print(
        json.dumps(
            {
                "status": "FROZEN_AFTER_VARIANT_CREATION",
                "manifest": {
                    "path": str(manifest_path),
                    "sha256": _sha256(manifest_path),
                },
                "preregistration": {
                    "path": str(prereg_path),
                    "sha256": _sha256(prereg_path),
                },
                "models": len(models),
                "levels": len(validation["levels"]),
                "formal_seed_consumption": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
