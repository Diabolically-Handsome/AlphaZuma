"""Freeze postprocessing for the retry-audited motor-observable route."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tools import run_alphazuma_55_motor_observable_postprocess_v2 as controller


SCRIPT_PATH = Path(__file__).resolve()
CAMPAIGN_ID = "alphazuma-55-motor-observable-postprocess-s99081617-v2"
EXPECTED_V1_PLAN_SHA256 = (
    "sha256:3c50df94556ab4d54e957bce369cfa7e2d5f151dce7c93b0c01ec677f07faddc"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _reference(path: Path) -> dict[str, str]:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "sha256": controller._sha256(resolved)}


def _write_new(path: Path, value: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists():
        temporary.unlink()
        raise FileExistsError(f"postprocess plan already exists: {path}")
    os.replace(temporary, path)


def _predecessor_incident(root: Path) -> dict[str, Any]:
    plan_path = (
        root
        / "diagnostics/alphazuma-55-motor-observable-postprocess-"
        "s99081615-plan-v1.json"
    )
    if controller._sha256(plan_path) != EXPECTED_V1_PLAN_SHA256:
        raise ValueError("V1 postprocess plan bytes changed")
    output = Path(
        "/mnt/d/ZumaTraining/alphazuma-55-motor-observable-postprocess-"
        "s99081615-v1"
    )
    status = controller._read(output / "controller_status.json")
    failure = controller._read(output / "failure.json")
    unexpected = [
        path
        for path in output.iterdir()
        if path.name not in {"controller_status.json", "failure.json"}
    ]
    if not (
        status.get("status") == "ERROR"
        and status.get("phase") == "ERROR"
        and status.get("formal_seed_consumption") == "NONE"
        and failure.get("error") == "motor-observable training reported failure"
        and failure.get("formal_seed_consumption") == "NONE"
        and not unexpected
    ):
        raise ValueError("V1 postprocess incident facts changed")
    return {
        "classification": (
            "V1_WAITING_CONTROLLER_ABORTED_WITHOUT_MATRIX_INFERENCE"
        ),
        "plan": _reference(plan_path),
        "controller_status": _reference(output / "controller_status.json"),
        "failure": _reference(output / "failure.json"),
        "matrix_artifacts_present": False,
        "formal_seed_consumption": "NONE",
    }


def build(*, project_root: Path, output: Path) -> dict[str, Any]:
    root = project_root.resolve(strict=True)
    old_plan_path = (
        root
        / "diagnostics/alphazuma-55-motor-observable-postprocess-"
        "s99081615-plan-v1.json"
    )
    plan = copy.deepcopy(controller._read(old_plan_path))
    route = (
        root
        / "diagnostics/alphazuma-55-motor-observable-replay-"
        "s99081616-preregistration-v3.json"
    )
    route_value = controller._read(route)
    run_dir = Path(str(route_value["run"]["run_dir"]))
    output_root = Path(f"/mnt/d/ZumaTraining/{CAMPAIGN_ID}")
    plan.update(
        {
            "created_utc": _utc_now(),
            "campaign_id": CAMPAIGN_ID,
            "objective": (
                "Select every early checkpoint from the retry-audited "
                "motor-observable route, then apply the unchanged 35-win and "
                "35-level fresh full55 engineering promotion gate."
            ),
            "training_route": _reference(route),
            "expected_training_completion": str(run_dir / "completion.json"),
            "expected_training_failure": str(run_dir / "failure.json"),
            "predecessor_postprocess_incident": _predecessor_incident(root),
            "deadline_utc": "2026-08-17T14:45:00+00:00",
        }
    )
    plan["implementation"].update(
        {
            "builder": _reference(SCRIPT_PATH),
            "controller": _reference(
                root
                / "tools/run_alphazuma_55_motor_observable_postprocess_v2.py"
            ),
            "auditor": _reference(
                root
                / "tools/audit_alphazuma_55_motor_observable_postprocess_v2.py"
            ),
            "v1_controller_runtime": _reference(
                root
                / "tools/run_alphazuma_55_motor_observable_postprocess_v1.py"
            ),
            "v1_auditor_runtime": _reference(
                root
                / "tools/audit_alphazuma_55_motor_observable_postprocess_v1.py"
            ),
        }
    )
    plan["outputs"] = {
        "output_root": str(output_root),
        "independent_audit_receipt": str(
            root
            / "diagnostics/alphazuma-55-motor-observable-postprocess-"
            "s99081617-independent-audit-v2.json"
        ),
    }
    plan["seed_reuse_disclosure"] = {
        "classification": (
            "unchanged frozen training-validation seeds reused only because "
            "the predecessor stopped before creating any matrix artifact"
        ),
        "screen_or_gate_outcomes_seen": False,
        "formal_gate_unchanged": True,
        "formal_seed_consumption": "NONE",
    }
    _write_new(output, plan)
    return plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    plan = build(project_root=args.project_root, output=args.output)
    print(json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
