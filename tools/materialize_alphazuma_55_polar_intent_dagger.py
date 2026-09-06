"""Bind the frozen raw-intent final model into its DAgger preregistration."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

from tools import distill_alphazuma_55 as legacy
from tools.distill_alphazuma_55_polar_intent_dagger_v1 import (
    _validate_probability_schedule,
    _validate_preregistration,
)
from zuma_rl.alphazuma_55 import INCLUDED_LEVELS


SCRIPT_PATH = Path(__file__).resolve()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _write_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists():
        temporary.unlink()
        raise FileExistsError(f"refusing to overwrite preregistration: {path}")
    os.replace(temporary, path)


def _bound_path(reference: Any, label: str) -> Path:
    if not isinstance(reference, dict):
        raise ValueError(f"{label} reference must be an object")
    path = Path(str(reference.get("path", ""))).resolve(strict=True)
    if reference.get("sha256") != legacy._sha256(path):
        raise ValueError(f"{label} bytes differ: {path}")
    return path


def validate_plan_static(path: Path, expected_sha256: str) -> dict[str, Any]:
    path = path.resolve(strict=True)
    if legacy._sha256(path) != expected_sha256:
        raise ValueError("raw-intent DAgger plan hash differs")
    plan = _read(path)
    if not (
        plan.get("schema")
        == "zuma-rl.alphazuma-55-polar-intent-dagger-plan"
        and plan.get("version") == 1
        and plan.get("status") == "FROZEN_BEFORE_SOURCE_RESULT_INSPECTION"
    ):
        raise ValueError("unexpected raw-intent DAgger plan")
    master_path = _bound_path(plan.get("master_preregistration"), "master")
    master = _read(master_path)
    if (
        master.get("schema")
        != "zuma-rl.alphazuma-55-weekend-master-preregistration"
    ):
        raise ValueError("unexpected raw-intent DAgger master")
    source_prereg_path = _bound_path(
        plan.get("source", {}).get("preregistration"),
        "raw-intent source preregistration",
    )
    source_prereg = _read(source_prereg_path)
    if not (
        source_prereg.get("schema")
        == (
            "zuma-rl.alphazuma-55-polar-intent-wide-"
            "distillation-preregistration"
        )
        and source_prereg.get("status") == "FROZEN_BEFORE_TRAINING"
        and source_prereg.get("run", {}).get("raw_teacher_intent_labels")
        is True
    ):
        raise ValueError("unexpected raw-intent source preregistration")
    for name, artifact in plan.get("implementation", {}).items():
        implementation = _bound_path(artifact, f"implementation {name}")
        if name == "materializer" and implementation != SCRIPT_PATH:
            raise ValueError("plan binds another raw-intent DAgger materializer")
    for name, artifact in plan.get("implementation_dependencies", {}).items():
        _bound_path(artifact, f"dependency {name}")
    _bound_path(
        plan.get("teacher_evidence", {}).get("exact_mask_fresh_55", {}),
        "teacher evidence",
    )
    if tuple(plan.get("levels", ())) != tuple(INCLUDED_LEVELS):
        raise ValueError("raw-intent DAgger plan level order changed")
    run = plan.get("run", {})
    schedule = _validate_probability_schedule(
        run.get("teacher_execution_probabilities", ())
    )
    if int(run.get("rounds", -1)) != len(schedule):
        raise ValueError("raw-intent DAgger plan round count changed")
    first = int(run["training_seed_base"])
    last = int(run["training_seed_last_consumed"])
    if last != first + len(INCLUDED_LEVELS) * len(schedule) - 1:
        raise ValueError("raw-intent DAgger plan training seeds changed")
    training_registry = master["seed_registry"]["training"]
    if not (
        int(training_registry["first"])
        <= first
        <= last
        <= int(training_registry["last"])
    ):
        raise ValueError("raw-intent DAgger plan escaped training registry")
    frozen = plan.get("frozen_engineering_validation", {})
    validation_first = int(frozen.get("base_seed", -1))
    validation_last = int(frozen.get("last_seed", -1))
    validation_registry = master["seed_registry"]["training_validation"]
    if not (
        frozen.get("registry") == "training_validation"
        and validation_last == validation_first + len(INCLUDED_LEVELS) - 1
        and int(validation_registry["first"])
        <= validation_first
        <= validation_last
        <= int(validation_registry["last"])
    ):
        raise ValueError("raw-intent DAgger validation seeds changed")
    boundary = plan.get("authority_boundary", {})
    for name in (
        "current_campaign_candidate_authority",
        "s99081545_successor_candidate_authority",
        "formal_selection_seed_consumption",
        "formal_final_blind_seed_consumption",
        "continuous_campaign_seed_consumption",
    ):
        if boundary.get(name) is not False:
            raise ValueError(f"raw-intent DAgger plan authority changed: {name}")
    outputs = plan.get("outputs", {})
    for key in ("run_dir", "preregistration", "status_root"):
        if not str(outputs.get(key, "")):
            raise ValueError(f"raw-intent DAgger output is missing: {key}")
    source = plan.get("source", {})
    if not str(source.get("expected_completion", "")) or not str(
        source.get("expected_failure", "")
    ):
        raise ValueError("raw-intent source receipts are missing")
    return plan


def materialize(plan_path: Path, expected_plan_sha256: str) -> Path:
    plan_path = plan_path.resolve(strict=True)
    plan = validate_plan_static(plan_path, expected_plan_sha256)
    output = Path(str(plan["outputs"]["preregistration"])).resolve()
    if output.exists():
        _validate_preregistration(output)
        existing = _read(output)
        if existing.get("materialization_plan", {}).get("sha256") != (
            expected_plan_sha256
        ):
            raise ValueError("existing preregistration binds another plan")
        return output

    source = plan["source"]
    completion_path = Path(str(source["expected_completion"])).resolve(
        strict=True
    )
    completion = _read(completion_path)
    if not (
        completion.get("schema")
        == (
            "zuma-rl.alphazuma-55-polar-intent-wide-"
            "distillation-completion"
        )
        and completion.get("status") == "COMPLETE"
        and completion.get("policy_architecture")
        == "entity_polar_intent_wide"
        and completion.get("training_label_semantics")
        == "raw_teacher_intent"
        and completion.get("online_policy_inference_semantics")
        == "exact_action_masks"
        and completion.get("formal_seed_consumption") is False
    ):
        raise ValueError("raw-intent completion is not eligible for DAgger")
    model = completion.get("final_model", {})
    model_path = Path(str(model.get("path", ""))).resolve(strict=True)
    model_sha256 = legacy._sha256(model_path)
    if model.get("sha256") != model_sha256:
        raise ValueError("raw-intent model differs from completion receipt")
    preregistration = {
        "schema": "zuma-rl.alphazuma-55-polar-intent-dagger-preregistration",
        "version": 1,
        "status": "FROZEN_BEFORE_TRAINING",
        "created_utc": _utc_now(),
        "campaign_id": plan["campaign_id"],
        "objective": plan["objective"],
        "decision_evidence_available_before_freeze": plan[
            "decision_evidence_available_before_freeze"
        ],
        "hypotheses": plan["hypotheses"],
        "master_preregistration": plan["master_preregistration"],
        "materialization_plan": {
            "path": str(plan_path),
            "sha256": expected_plan_sha256,
        },
        "trainer": plan["implementation"]["trainer"],
        "implementation": plan["implementation_dependencies"],
        "teacher_evidence": plan["teacher_evidence"],
        "source": {
            "completion_path": str(completion_path),
            "completion_sha256": legacy._sha256(completion_path),
            "model_path": str(model_path),
            "model_sha256": model_sha256,
            "model_bytes": model_path.stat().st_size,
        },
        "levels": list(INCLUDED_LEVELS),
        "run": plan["run"],
        "frozen_engineering_validation": plan[
            "frozen_engineering_validation"
        ],
        "authority_boundary": plan["authority_boundary"],
    }
    _write_atomic(output, preregistration)
    _validate_preregistration(output)
    return output
