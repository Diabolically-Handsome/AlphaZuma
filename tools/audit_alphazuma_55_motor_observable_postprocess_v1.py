"""Independently recompute the frozen motor-observable engineering result."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tools import run_alphazuma_55_motor_observable_postprocess_v1 as controller


SCRIPT_PATH = Path(__file__).resolve()


def _write_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists():
        temporary.unlink()
        raise FileExistsError(f"refusing to overwrite audit receipt: {path}")
    os.replace(temporary, path)


def audit(*, plan_path: Path, expected_plan_sha256: str) -> dict[str, Any]:
    plan_path = plan_path.resolve(strict=True)
    plan = controller.validate_plan(plan_path, expected_plan_sha256)
    output_root = Path(str(plan["outputs"]["output_root"])).resolve(strict=True)
    status_path = output_root / "preaudit-controller-status.json"
    result_path = output_root / "final_report.json"
    status = controller._read(status_path)
    result = controller._read(result_path)
    status_result_hash = (
        status.get("result", {}).get("sha256")
        if status.get("phase") == "COMPLETE"
        else status.get("final_report", {}).get("sha256")
    )
    if not (
        status.get("status") in {"RUNNING", "COMPLETE"}
        and status.get("phase") in {"RUNNING_INDEPENDENT_AUDIT", "COMPLETE"}
        and status_result_hash == controller._sha256(result_path)
        and result.get("schema")
        == "zuma-rl.alphazuma-55-motor-observable-postprocess-result"
        and result.get("formal_seed_consumption") == "NONE"
    ):
        raise ValueError("motor-observable controller result is incomplete")

    screen_prereg = output_root / "screen-preregistration.json"
    screen_manifest = output_root / "screen-models-manifest.json"
    screen = controller.audit_matrix(
        prereg_path=screen_prereg,
        manifest_path=screen_manifest,
    )
    selected_ids = [
        str(row["model_id"])
        for row in screen["ranking"][: controller.MAXIMUM_GATE_CANDIDATES]
    ]
    recorded_screen = controller._read(output_root / "screen-audit.json")
    if recorded_screen.get("selected_for_full55_gate") != selected_ids:
        raise ValueError("motor-observable screen selection differs")

    gate_prereg = output_root / "full55-preregistration.json"
    gate_manifest = output_root / "full55-models-manifest.json"
    gate = controller.audit_matrix(
        prereg_path=gate_prereg,
        manifest_path=gate_manifest,
    )
    promotable = [row for row in gate["ranking"] if bool(row["promotable"])]
    if not promotable:
        raise ValueError("motor-observable gate has no promotable model")
    winner = promotable[0]
    passed = (
        int(winner["wins"]) >= int(plan["full55_gate"]["minimum_wins"])
        and int(winner["cleared_levels"])
        >= int(plan["full55_gate"]["minimum_cleared_levels"])
    )
    expected_status = "COMPLETE_GATE_PASS" if passed else "COMPLETE_NO_PROMOTION"
    recorded_gate = result.get("promotion_gate", {})
    if not (
        result.get("status") == expected_status
        and bool(recorded_gate.get("passed")) is passed
        and recorded_gate.get("winner", {}).get("model_id")
        == winner["model_id"]
        and recorded_gate.get("winner", {}).get("model_sha256")
        == winner["model_sha256"]
        and bool(result.get("formal_candidate_authority")) is passed
    ):
        raise ValueError("motor-observable promotion decision differs")
    return {
        "schema": "zuma-rl.alphazuma-55-motor-observable-independent-audit",
        "version": 1,
        "status": "PASS",
        "audited_utc": controller._utc_now(),
        "controller_result": expected_status,
        "gate_passed": passed,
        "winner": winner,
        "artifacts": {
            "plan": {"path": str(plan_path), "sha256": expected_plan_sha256},
            "controller_status": {
                "path": str(status_path),
                "sha256": controller._sha256(status_path),
            },
            "final_report": {
                "path": str(result_path),
                "sha256": controller._sha256(result_path),
            },
            "screen_preregistration": {
                "path": str(screen_prereg),
                "sha256": controller._sha256(screen_prereg),
            },
            "screen_manifest": {
                "path": str(screen_manifest),
                "sha256": controller._sha256(screen_manifest),
            },
            "full55_preregistration": {
                "path": str(gate_prereg),
                "sha256": controller._sha256(gate_prereg),
            },
            "full55_manifest": {
                "path": str(gate_manifest),
                "sha256": controller._sha256(gate_manifest),
            },
            "auditor": {
                "path": str(SCRIPT_PATH),
                "sha256": controller._sha256(SCRIPT_PATH),
            },
        },
        "recomputed": {
            "screen_attempts": screen["attempts"],
            "selected_for_full55_gate": selected_ids,
            "full55_attempts": gate["attempts"],
            "winner_wins": winner["wins"],
            "winner_cleared_levels": winner["cleared_levels"],
        },
        "formal_seed_consumption": "NONE",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    receipt = audit(
        plan_path=args.plan.expanduser(),
        expected_plan_sha256=str(args.expected_plan_sha256),
    )
    output = args.output.expanduser().resolve()
    _write_new(output, receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
