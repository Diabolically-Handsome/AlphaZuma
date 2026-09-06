"""Bind the frozen intent-wide source into replay-anchored DAgger."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

from tools import distill_alphazuma_55 as legacy
from tools.distill_alphazuma_55_polar_intent_replay_anchor_v1 import (
    EXPECTED_ANCHOR_PASSES,
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


def _write_atomic_new(path: Path, value: dict[str, Any]) -> None:
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
        raise ValueError("replay-anchor plan hash differs")
    plan = _read(path)
    if not (
        plan.get("schema")
        == "zuma-rl.alphazuma-55-polar-intent-replay-anchor-plan"
        and plan.get("version") == 1
        and plan.get("status")
        == "FROZEN_AFTER_FAILURE_DIAGNOSIS_BEFORE_ROUTE_INFERENCE"
    ):
        raise ValueError("unexpected replay-anchor plan")
    master_path = _bound_path(plan.get("master_preregistration"), "master")
    master = _read(master_path)
    if (
        master.get("schema")
        != "zuma-rl.alphazuma-55-weekend-master-preregistration"
    ):
        raise ValueError("unexpected replay-anchor master")
    source_prereg_path = _bound_path(
        plan.get("source", {}).get("preregistration"),
        "source intent-wide preregistration",
    )
    source_prereg = _read(source_prereg_path)
    if not (
        source_prereg.get("schema")
        == "zuma-rl.alphazuma-55-polar-intent-wide-distillation-preregistration"
        and source_prereg.get("status") == "FROZEN_BEFORE_TRAINING"
        and source_prereg.get("run", {}).get("raw_teacher_intent_labels")
        is True
    ):
        raise ValueError("unexpected replay-anchor source preregistration")
    for name, artifact in plan.get("implementation", {}).items():
        implementation = _bound_path(artifact, f"implementation {name}")
        if name == "materializer" and implementation != SCRIPT_PATH:
            raise ValueError("plan binds another replay-anchor materializer")
    for name, artifact in plan.get("implementation_dependencies", {}).items():
        _bound_path(artifact, f"dependency {name}")
    _bound_path(
        plan.get("teacher_evidence", {}).get("exact_mask_fresh_55", {}),
        "teacher evidence",
    )
    if tuple(plan.get("levels", ())) != tuple(INCLUDED_LEVELS):
        raise ValueError("replay-anchor level order changed")
    run = plan.get("run", {})
    schedule = _validate_probability_schedule(
        run.get("teacher_execution_probabilities", ())
    )
    if not (
        int(run.get("anchor_teacher_passes", -1))
        == EXPECTED_ANCHOR_PASSES
        and int(run.get("student_rounds", -1)) == len(schedule)
    ):
        raise ValueError("replay-anchor round counts changed")
    first = int(run["anchor_seed_base"])
    anchor_last = int(run["anchor_seed_last_consumed"])
    student_first = int(run["student_seed_base"])
    last = int(run["student_seed_last_consumed"])
    if not (
        anchor_last
        == first + len(INCLUDED_LEVELS) * EXPECTED_ANCHOR_PASSES - 1
        and student_first == anchor_last + 1
        and last == student_first + len(INCLUDED_LEVELS) * len(schedule) - 1
    ):
        raise ValueError("replay-anchor seed intervals changed")
    registry = master["seed_registry"]["training"]
    if not (
        int(registry["first"]) <= first <= last <= int(registry["last"])
    ):
        raise ValueError("replay-anchor escaped training registry")
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
        and int(frozen.get("expected_attempts", -1)) == 275
    ):
        raise ValueError("replay-anchor validation contract changed")
    boundary = plan.get("authority_boundary", {})
    for name in (
        "current_campaign_candidate_authority",
        "formal_selection_seed_consumption",
        "formal_final_blind_seed_consumption",
        "continuous_campaign_seed_consumption",
    ):
        if boundary.get(name) is not False:
            raise ValueError(f"replay-anchor plan authority changed: {name}")
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
        == "zuma-rl.alphazuma-55-polar-intent-wide-distillation-completion"
        and completion.get("status") == "COMPLETE"
        and completion.get("policy_architecture")
        == "entity_polar_intent_wide"
        and completion.get("training_label_semantics")
        == "raw_teacher_intent"
        and completion.get("online_policy_inference_semantics")
        == "exact_action_masks"
        and completion.get("formal_seed_consumption") is False
        and completion.get("formal_candidate_authority") is False
    ):
        raise ValueError("source intent-wide completion is not eligible")
    model = completion.get("final_model", {})
    model_path = Path(str(model.get("path", ""))).resolve(strict=True)
    model_sha256 = legacy._sha256(model_path)
    if model.get("sha256") != model_sha256:
        raise ValueError("source intent-wide model differs from receipt")
    preregistration = {
        "schema": (
            "zuma-rl.alphazuma-55-polar-intent-replay-anchor-preregistration"
        ),
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
        "source": {
            "preregistration": source["preregistration"],
            "completion_path": str(completion_path),
            "completion_sha256": legacy._sha256(completion_path),
            "model_path": str(model_path),
            "model_sha256": model_sha256,
            "selection_rule": source["selection_rule"],
        },
        "teacher_evidence": plan["teacher_evidence"],
        "levels": plan["levels"],
        "run": plan["run"],
        "frozen_engineering_validation": plan[
            "frozen_engineering_validation"
        ],
        "authority_boundary": plan["authority_boundary"],
    }
    _write_atomic_new(output, preregistration)
    _validate_preregistration(output)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--expected-plan-sha256", required=True)
    args = parser.parse_args(argv)
    output = materialize(
        args.plan.expanduser(), str(args.expected_plan_sha256)
    )
    print(
        json.dumps(
            {
                "status": "MATERIALIZED",
                "preregistration": {
                    "path": str(output),
                    "sha256": legacy._sha256(output),
                },
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
