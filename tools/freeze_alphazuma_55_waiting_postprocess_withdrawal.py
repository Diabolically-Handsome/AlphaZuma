"""Freeze evidence that a waiting AlphaZuma 55 deployment was withdrawn."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON document {path}: {error}") from error
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _write_json_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError:
        raise FileExistsError(f"refusing to overwrite withdrawal receipt: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _process_inventory() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        command = raw.replace(b"\0", b" ").decode("utf-8", errors="replace").strip()
        if command:
            rows.append({"pid": int(entry.name), "command": command})
    return rows


def _old_processes(
    inventory: list[dict[str, Any]],
    *,
    deployment_name: str,
    postprocess_name: str,
) -> dict[str, list[dict[str, Any]]]:
    return {
        "deployer": [
            row
            for row in inventory
            if "deploy_alphazuma_55_postprocess_parallel_v2.py" in row["command"]
            and deployment_name in row["command"]
        ],
        "controller": [
            row
            for row in inventory
            if "run_alphazuma_55_postprocess_parallel_v2.py" in row["command"]
            and postprocess_name in row["command"]
        ],
    }


def _gpu_limits() -> dict[str, float]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,power.limit",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    values: dict[str, float] = {}
    for line in completed.stdout.splitlines():
        index, limit = [part.strip() for part in line.split(",")]
        values[index] = float(limit)
    return values


def _active_power_plan() -> str:
    raw = subprocess.run(
        ["powercfg.exe", "/getactivescheme"],
        check=True,
        capture_output=True,
    ).stdout
    return raw.decode("ascii", errors="ignore").strip()


def freeze_withdrawal(
    *,
    expansion_path: Path,
    expected_expansion_sha256: str,
    current_deployment_path: Path,
    expected_current_deployment_sha256: str,
    process_inventory: list[dict[str, Any]] | None = None,
    gpu_limits: dict[str, float] | None = None,
    active_power_plan: str | None = None,
) -> dict[str, Any]:
    _require(_sha256(expansion_path) == expected_expansion_sha256, "capacity expansion hash mismatch")
    _require(
        _sha256(current_deployment_path) == expected_current_deployment_sha256,
        "current deployment hash mismatch",
    )
    expansion = _read_json(expansion_path)
    current = _read_json(current_deployment_path)
    expansion_version = expansion.get("version")
    expansion_status = expansion.get("status")
    _require(
        expansion.get("schema")
        == "zuma-rl.alphazuma-55-capacity-expansion-preregistration"
        and (
            (expansion_version == 1 and expansion_status == "FROZEN_BEFORE_REPLICATE_TRAINING")
            or (
                expansion_version == 2
                and expansion_status == "FROZEN_BEFORE_HARD_FRONTIER_TRAINING"
            )
        ),
        "unexpected capacity expansion",
    )
    _require(
        current.get("schema")
        == "zuma-rl.alphazuma-55-postprocess-deployment-preregistration"
        and current.get("status") == "FROZEN_BEFORE_FORMAL_SEEDS",
        "unexpected current deployment",
    )
    current_reference = expansion["postprocess_supersession"]["current_deployment"]
    _require(
        Path(str(current_reference["path"])).resolve(strict=True) == current_deployment_path
        and current_reference["sha256"] == expected_current_deployment_sha256,
        "capacity expansion binds another current deployment",
    )

    milestone_path = Path(str(expansion["safe_switch_gate"]["wait_for_milestone_controller_to_reach_terminal_complete_or_error"])).resolve(strict=True)
    milestone = _read_json(milestone_path)
    _require(milestone.get("status") in {"COMPLETE", "ERROR"}, "milestone controller is not terminal")
    controller_status_path = Path(str(current["formal_outputs"]["controller"])).resolve(strict=True) / "controller_status.json"
    deployment_status_path = Path(str(current["deployment_output_root"])).resolve(strict=True) / "deployment_status.json"
    controller_status = _read_json(controller_status_path)
    deployment_status = _read_json(deployment_status_path)
    _require(
        controller_status.get("phase") == "WAITING_FOR_TRAINING"
        and controller_status.get("error") is None,
        "old controller was not cleanly waiting",
    )

    old_outputs = current["formal_outputs"]
    old_formal_absent = not any(
        Path(str(path)).exists()
        for name, path in old_outputs.items()
        if name != "controller"
    )
    new_paths = expansion["postprocess_supersession"]
    new_formal_absent = not any(
        Path(str(path)).exists()
        for path in [
            new_paths["new_postprocess_preregistration"],
            new_paths["new_deployment_preregistration"],
            new_paths["new_independent_audit_receipt"],
            new_paths["new_deployment_output_root"],
            *new_paths["new_formal_outputs"].values(),
        ]
    )
    _require(old_formal_absent, "old formal evaluation roots are not absent")
    _require(new_formal_absent, "new expansion artifacts appeared before withdrawal")

    inventory = _process_inventory() if process_inventory is None else process_inventory
    matches = _old_processes(
        inventory,
        deployment_name=current_deployment_path.name,
        postprocess_name=Path(str(current["postprocess_preregistration"])).name,
    )
    _require(not matches["controller"] and not matches["deployer"], "old postprocess processes remain alive")

    limits = _gpu_limits() if gpu_limits is None else gpu_limits
    _require(
        abs(float(limits.get("0", -1.0)) - 550.0) < 0.6
        and abs(float(limits.get("1", -1.0)) - 250.0) < 0.6,
        "training GPU power limits changed during withdrawal",
    )
    active = _active_power_plan() if active_power_plan is None else active_power_plan
    master_path = Path(str(current["master_preregistration"]["path"])).resolve(strict=True)
    _require(current["master_preregistration"]["sha256"] == _sha256(master_path), "master hash mismatch")
    master = _read_json(master_path)
    training_plan_guid = str(master["hardware_contract"]["windows_training_power_plan"]).split()[-1]
    _require(training_plan_guid.casefold() in active.casefold(), "Windows training power plan changed")

    return {
        "schema": "zuma-rl.alphazuma-55-waiting-postprocess-withdrawal",
        "version": 1,
        "status": "PASS",
        "withdrawn_utc": datetime.now(timezone.utc).isoformat(),
        "reason": (
            "Superseded before formal seeds by the preregistered "
            f"{int(expansion['postprocess_supersession']['registered_route_count'])}-route "
            "capacity expansion."
        ),
        "capacity_expansion": {"path": str(expansion_path), "sha256": expected_expansion_sha256},
        "withdrawal_builder": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "withdrawn_deployment": {
            "path": str(current_deployment_path),
            "sha256": expected_current_deployment_sha256,
        },
        "milestone_controller": {
            "path": str(milestone_path),
            "sha256": _sha256(milestone_path),
            "status": milestone["status"],
        },
        "preconditions": {
            "old_controller_phase": controller_status["phase"],
            "old_formal_roots_absent": old_formal_absent,
            "new_expansion_artifacts_absent": new_formal_absent,
            "formal_seed_consumption": False,
        },
        "processes": {
            "old_controller_absent": not matches["controller"],
            "old_deployer_absent": not matches["deployer"],
            "matches": matches,
        },
        "old_status_artifacts": {
            "controller": {"path": str(controller_status_path), "sha256": _sha256(controller_status_path)},
            "deployment": {
                "path": str(deployment_status_path),
                "sha256": _sha256(deployment_status_path),
                "status": deployment_status.get("status"),
                "stage": deployment_status.get("stage"),
            },
        },
        "power_state": {
            "restoration_requested": False,
            "gpu_power_limits_watts": limits,
            "active_windows_power_plan": active,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expansion-preregistration", required=True, type=Path)
    parser.add_argument("--expected-expansion-sha256", required=True)
    parser.add_argument("--current-deployment", required=True, type=Path)
    parser.add_argument("--expected-current-deployment-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite withdrawal receipt: {output}")
    receipt = freeze_withdrawal(
        expansion_path=args.expansion_preregistration.expanduser().resolve(strict=True),
        expected_expansion_sha256=args.expected_expansion_sha256,
        current_deployment_path=args.current_deployment.expanduser().resolve(strict=True),
        expected_current_deployment_sha256=args.expected_current_deployment_sha256,
    )
    _write_json_exclusive(output, receipt)
    print(
        json.dumps(
            {"status": "PASS", "output": str(output), "sha256": _sha256(output)},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
