"""Safely replace a waiting AlphaZuma switch before it materializes outputs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tools import switch_alphazuma_55_waiting_deployment as legacy


SCRIPT_PATH = Path(__file__).resolve()


def _matching_switches(
    inventory: list[dict[str, Any]], *, script_name: str, preregistration_name: str
) -> list[dict[str, Any]]:
    return [
        row
        for row in inventory
        if script_name
        in str(row.get("command", row.get("cmdline", "")))
        and preregistration_name
        in str(row.get("command", row.get("cmdline", "")))
    ]


def _artifact(path: Path) -> dict[str, str]:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "sha256": legacy._sha256(resolved)}


def _controller_boundary(current: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    controller_root = Path(str(current["formal_outputs"]["controller"])).resolve(strict=True)
    status_path = controller_root / "controller_status.json"
    status = legacy._read_json(status_path)
    legacy._require(
        status.get("status") == "RUNNING"
        and status.get("phase") == "WAITING_FOR_TRAINING"
        and status.get("error") is None,
        "current formal controller is not cleanly waiting",
    )
    for name in ("selection", "final_blind", "continuous"):
        legacy._require(
            not Path(str(current["formal_outputs"][name])).exists(),
            f"current formal {name} root is no longer absent",
        )
    return status_path, status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-switch-preregistration", required=True, type=Path)
    parser.add_argument("--expected-old-switch-sha256", required=True)
    parser.add_argument("--new-switch-preregistration", required=True, type=Path)
    parser.add_argument("--expected-new-switch-sha256", required=True)
    parser.add_argument("--prestop-evidence", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--exit-timeout-seconds", type=float, default=30.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    old_path = args.old_switch_preregistration.expanduser().resolve(strict=True)
    new_path = args.new_switch_preregistration.expanduser().resolve(strict=True)
    legacy._require(
        legacy._sha256(old_path) == args.expected_old_switch_sha256,
        "old switch preregistration hash mismatch",
    )
    legacy._require(
        legacy._sha256(new_path) == args.expected_new_switch_sha256,
        "new switch preregistration hash mismatch",
    )
    old = legacy._validate_preregistration(old_path)
    new = legacy._validate_preregistration(new_path)
    supersedes = new.get("supersedes_waiting_switch", {}).get("preregistration", {})
    legacy._require(
        supersedes.get("path") == str(old_path)
        and supersedes.get("sha256") == args.expected_old_switch_sha256,
        "new switch does not bind the old switch",
    )

    old_root = Path(str(old["outputs"]["switch_output_root"])).resolve(strict=True)
    old_status_path = old_root / "controller_status.json"
    old_status = legacy._read_json(old_status_path)
    legacy._require(
        old_status.get("status") == "RUNNING"
        and old_status.get("phase") == "WAITING_FOR_MILESTONE",
        "old switch is not safely waiting for the milestone",
    )
    for forbidden in ("prestop_evidence.json", "handoff.json", "failure.json"):
        legacy._require(
            not (old_root / forbidden).exists(),
            f"old switch already materialized {forbidden}",
        )

    current_path = Path(str(old["current_deployment"]["path"])).resolve(strict=True)
    current = legacy._read_json(current_path)
    controller_status_path, controller_status = _controller_boundary(current)
    expansion_path = Path(str(old["capacity_expansion"]["path"])).resolve(strict=True)
    expansion = legacy._read_json(expansion_path)
    supersession = expansion["postprocess_supersession"]
    prospective_paths = [
        Path(str(old["outputs"]["withdrawal_receipt"])).resolve(),
        Path(str(old["outputs"]["new_deployment_preregistration"])).resolve(),
        Path(str(supersession["new_postprocess_preregistration"])).resolve(),
        Path(str(supersession["new_independent_audit_receipt"])).resolve(),
        Path(str(supersession["new_deployment_output_root"])).resolve(),
        *[
            Path(str(path)).resolve()
            for path in supersession["new_formal_outputs"].values()
        ],
    ]
    legacy._require(
        not any(path.exists() for path in prospective_paths),
        "a prospective six-route output already exists",
    )
    new_root = Path(str(new["outputs"]["switch_output_root"])).resolve()
    legacy._require(not new_root.exists(), "new switch output root already exists")

    old_script = Path(str(old["implementation"]["orchestrator"]["path"])).name
    matches = _matching_switches(
        legacy._process_inventory(),
        script_name=old_script,
        preregistration_name=old_path.name,
    )
    legacy._require(len(matches) == 1, "expected exactly one old waiting switch process")
    old_pid = int(matches[0]["pid"])
    legacy._require(old_pid != os.getpid(), "refusing to signal the superseder itself")

    evidence_path = args.prestop_evidence.expanduser().resolve()
    receipt_path = args.receipt.expanduser().resolve()
    legacy._write_exclusive(
        evidence_path,
        {
            "schema": "zuma-rl.alphazuma-55-waiting-switch-supersession-prestop",
            "version": 1,
            "status": "PASS",
            "recorded_utc": datetime.now(timezone.utc).isoformat(),
            "superseder": _artifact(SCRIPT_PATH),
            "old_switch": _artifact(old_path),
            "new_switch": _artifact(new_path),
            "old_switch_status": _artifact(old_status_path),
            "formal_controller_status": _artifact(controller_status_path),
            "formal_controller_phase": controller_status["phase"],
            "formal_seed_consumption": False,
            "prospective_outputs_absent": True,
            "old_process": matches[0],
        },
    )

    os.kill(old_pid, signal.SIGTERM)
    deadline = time.monotonic() + args.exit_timeout_seconds
    while time.monotonic() < deadline:
        remaining = _matching_switches(
            legacy._process_inventory(),
            script_name=old_script,
            preregistration_name=old_path.name,
        )
        if not remaining:
            break
        time.sleep(0.5)
    else:
        raise RuntimeError("old waiting switch did not exit after SIGTERM")

    legacy._write_exclusive(
        receipt_path,
        {
            "schema": "zuma-rl.alphazuma-55-waiting-switch-supersession",
            "version": 1,
            "status": "PASS",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "prestop_evidence": _artifact(evidence_path),
            "old_switch": _artifact(old_path),
            "new_switch": _artifact(new_path),
            "old_pid": old_pid,
            "old_process_absent": True,
            "formal_seed_consumption": False,
            "training_processes_modified": False,
            "power_restoration_requested": False,
        },
    )

    orchestrator = Path(str(new["implementation"]["orchestrator"]["path"])).resolve(strict=True)
    os.execv(
        sys.executable,
        [sys.executable, str(orchestrator), "--preregistration", str(new_path)],
    )


if __name__ == "__main__":
    raise SystemExit(main())
