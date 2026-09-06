"""Emit one strict, read-only AlphaZuma 55 weekend health snapshot."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any


CHECKPOINT_RE = re.compile(r"_(\d+)_steps\.zip$")


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
    # Windows PowerShell guard receipts are emitted with a UTF-8 BOM.  Treat
    # that encoding marker as transport metadata while retaining strict JSON
    # parsing for the document itself.
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON document {path}: {error}") from error
    _require(isinstance(value, dict), f"expected a JSON object: {path}")
    return value


def _utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    _require(parsed.tzinfo is not None, f"timestamp lacks timezone: {value}")
    return parsed.astimezone(timezone.utc)


def _bound_path(reference: Any, name: str) -> Path:
    _require(isinstance(reference, dict), f"{name} must be an object")
    path = Path(str(reference.get("path", ""))).expanduser().resolve(strict=True)
    _require(reference.get("sha256") == _sha256(path), f"{name} hash mismatch")
    return path


def _process_inventory() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
            ).strip()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if command:
            rows.append({"pid": int(entry.name), "command": command})
    return rows


def _matching_processes(
    inventory: list[dict[str, Any]], *needles: str
) -> list[dict[str, Any]]:
    return [
        row
        for row in inventory
        if all(needle in str(row["command"]) for needle in needles)
    ]


def _latest_checkpoint(run_dir: Path) -> dict[str, Any] | None:
    checkpoints = run_dir / "checkpoints"
    if not checkpoints.exists():
        return None
    rows: list[tuple[int, Path]] = []
    for path in checkpoints.glob("*_steps.zip"):
        match = CHECKPOINT_RE.search(path.name)
        if match:
            rows.append((int(match.group(1)), path))
    if not rows:
        return None
    steps, path = max(rows, key=lambda row: (row[0], row[1].name))
    stat = path.stat()
    return {
        "path": str(path),
        "steps": steps,
        "bytes": stat.st_size,
        "modified_utc": datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).isoformat(),
    }


def _training_snapshot(
    postprocess: dict[str, Any], now: datetime, inventory: list[dict[str, Any]]
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for index, reference in enumerate(postprocess["training_routes"]):
        route_path = _bound_path(reference, f"training_routes[{index}]")
        route = _read_json(route_path)
        runs = route.get("runs")
        _require(isinstance(runs, list) and len(runs) == 1, "route must have one run")
        run = runs[0]
        run_id = str(run["id"])
        run_dir = Path(str(run["run_dir"])).resolve(strict=True)
        status_path = run_dir / "training_status.json"
        completion_path = run_dir / "completion.json"
        failure_path = run_dir / "failure.json"
        status = _read_json(status_path)
        _require(status.get("run_id") == run_id, f"run id mismatch: {run_id}")
        updated = _utc(str(status["updated_utc"]))
        per_level = status.get("per_level")
        _require(isinstance(per_level, dict), f"per_level invalid: {run_id}")
        processes = _matching_processes(
            inventory, "train_alphazuma_55.py", "--run-id", run_id
        )
        rows.append(
            {
                "run_id": run_id,
                "route_preregistration": {
                    "path": str(route_path),
                    "sha256": _sha256(route_path),
                },
                "run_dir": str(run_dir),
                "device": str(run["device"]),
                "status": str(status["status"]),
                "updated_utc": updated.isoformat(),
                "age_seconds": (now - updated).total_seconds(),
                "timesteps": int(status["timesteps"]),
                "steps_this_run": int(status["steps_this_run"]),
                "steps_per_second": float(status["steps_per_second"]),
                "episodes": int(status["episodes"]),
                "wins": int(status["wins"]),
                "losses": int(status["losses"]),
                "truncations": int(status["truncations"]),
                "training_win_coverage": sum(
                    int(value.get("wins", 0)) > 0 for value in per_level.values()
                ),
                "latest_checkpoint": _latest_checkpoint(run_dir),
                "completion_exists": completion_path.exists(),
                "failure_exists": failure_path.exists(),
                "processes": processes,
            }
        )
    return {
        "route_count": len(rows),
        "running_routes": sum(row["status"] == "RUNNING" for row in rows),
        "completed_routes": sum(row["completion_exists"] for row in rows),
        "failed_routes": sum(row["failure_exists"] for row in rows),
        "total_steps_this_run": sum(row["steps_this_run"] for row in rows),
        "aggregate_steps_per_second": sum(
            row["steps_per_second"] for row in rows if row["status"] == "RUNNING"
        ),
        "max_running_age_seconds": max(
            (row["age_seconds"] for row in rows if row["status"] == "RUNNING"),
            default=0.0,
        ),
        "missing_running_processes": [
            row["run_id"]
            for row in rows
            if row["status"] == "RUNNING" and not row["processes"]
        ],
        "routes": rows,
    }


def _evaluation_snapshot(root: Path) -> dict[str, Any]:
    files = sorted(root.glob("matrix-shard-*.json")) if root.exists() else []
    if not files:
        return {
            "root": str(root),
            "status": "NOT_STARTED",
            "shard_count": 0,
            "completed_attempts": 0,
            "expected_attempts": 0,
            "eta_seconds": None,
            "shards": [],
        }
    rows: list[dict[str, Any]] = []
    for path in files:
        value = _read_json(path)
        expected = int(value["expected_attempts"])
        completed = int(value["completed_attempts"])
        wall = float(value["runtime"]["wall_seconds"])
        _require(0 <= completed <= expected and expected > 0, "invalid shard counts")
        _require(math.isfinite(wall) and wall >= 0, "invalid shard wall time")
        rate = completed / wall if completed and wall else None
        eta = (expected - completed) / rate if rate and completed < expected else (
            0.0 if completed == expected else None
        )
        rows.append(
            {
                "path": str(path),
                "shard_index": int(value["shard_index"]),
                "status": str(value["status"]),
                "completed_attempts": completed,
                "expected_attempts": expected,
                "wall_seconds": wall,
                "eta_seconds": eta,
                "error": value.get("error"),
            }
        )
    rows.sort(key=lambda row: row["shard_index"])
    etas = [float(row["eta_seconds"]) for row in rows if row["eta_seconds"] is not None]
    status = (
        "ERROR"
        if any(row["status"] == "ERROR" or row["error"] is not None for row in rows)
        else (
            "COMPLETE"
            if all(row["status"] == "COMPLETE" for row in rows)
            else "RUNNING"
        )
    )
    return {
        "root": str(root),
        "status": status,
        "shard_count": len(rows),
        "completed_attempts": sum(row["completed_attempts"] for row in rows),
        "expected_attempts": sum(row["expected_attempts"] for row in rows),
        "eta_seconds": max(etas) if etas else None,
        "shards": rows,
    }


def _gpu_snapshot() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,power.draw,power.limit,utilization.gpu,memory.used,memory.total,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    rows: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        _require(len(parts) == 8, "unexpected nvidia-smi CSV")
        rows.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "power_draw_watts": float(parts[2]),
                "power_limit_watts": float(parts[3]),
                "utilization_percent": int(parts[4]),
                "memory_used_mib": int(parts[5]),
                "memory_total_mib": int(parts[6]),
                "temperature_c": int(parts[7]),
            }
        )
    return rows


def _memory_snapshot() -> dict[str, Any]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, raw = line.split(":", 1)
        values[key] = int(raw.strip().split()[0]) * 1024
    swap_used = values["SwapTotal"] - values["SwapFree"]
    return {
        "total_bytes": values["MemTotal"],
        "available_bytes": values["MemAvailable"],
        "swap_total_bytes": values["SwapTotal"],
        "swap_used_bytes": swap_used,
    }


def _disk_snapshot(path: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(path)
    return {
        "path": str(path),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
    }


def _power_plan() -> str:
    output = subprocess.run(
        ["powercfg.exe", "/getactivescheme"], check=True, capture_output=True
    ).stdout
    return output.decode("ascii", errors="ignore").strip()


def build_snapshot(
    *,
    master_path: Path,
    deployment_path: Path,
    stale_seconds: float,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    master = _read_json(master_path)
    deployment = _read_json(deployment_path)
    _require(
        master.get("schema")
        == "zuma-rl.alphazuma-55-weekend-master-preregistration",
        "unexpected master preregistration",
    )
    _require(
        deployment.get("schema")
        == "zuma-rl.alphazuma-55-postprocess-deployment-preregistration"
        and deployment.get("status") == "FROZEN_BEFORE_FORMAL_SEEDS",
        "unexpected deployment preregistration",
    )
    _require(
        deployment["master_preregistration"]["sha256"] == _sha256(master_path),
        "deployment does not bind the supplied master",
    )
    for name, reference in deployment["implementation"].items():
        _bound_path(reference, f"deployment implementation {name}")

    postprocess_path = Path(str(deployment["postprocess_preregistration"])).resolve(
        strict=True
    )
    postprocess = _read_json(postprocess_path)
    inventory = _process_inventory()
    training = _training_snapshot(postprocess, now, inventory)

    outputs = postprocess["outputs"]
    controller_root = Path(str(outputs["controller"])).resolve(strict=True)
    controller_path = controller_root / "controller_status.json"
    controller = _read_json(controller_path)
    deployment_root = Path(str(deployment["deployment_output_root"])).resolve(
        strict=True
    )
    deployment_status_path = deployment_root / "deployment_status.json"
    deployment_status = _read_json(deployment_status_path)
    phase = str(controller["phase"])

    phase_root = {
        "SELECTION_EVALUATION": Path(str(outputs["selection"])),
        "FINAL_BLIND_EVALUATION": Path(str(outputs["final_blind"])),
        "CONTINUOUS_CAMPAIGNS": Path(str(outputs["continuous"])),
    }.get(phase)
    evaluation = _evaluation_snapshot(phase_root) if phase_root is not None else None

    formal_roots = {
        name: {"path": str(outputs[name]), "exists": Path(str(outputs[name])).exists()}
        for name in ("selection", "final_blind", "continuous")
    }
    embargo_intact = phase != "WAITING_FOR_TRAINING" or not any(
        row["exists"] for row in formal_roots.values()
    )

    hardware = master["hardware_contract"]
    gpus = _gpu_snapshot()
    memory = _memory_snapshot()
    root_disk = _disk_snapshot(Path("/"))
    d_disk = _disk_snapshot(Path("/mnt/d"))
    gpu_guard_path = Path(str(deployment["restore"]["gpu_receipt"])).resolve(
        strict=True
    )
    plan_guard_path = Path(
        str(deployment["restore"]["power_plan_receipt"])
    ).resolve(strict=True)
    gpu_guard = _read_json(gpu_guard_path)
    plan_guard = _read_json(plan_guard_path)
    active_plan = _power_plan()

    expected_limits = {
        0: float(hardware["gpu_0"]["training_power_limit_watts"]),
        1: float(hardware["gpu_1"]["training_power_limit_watts"]),
    }
    power_limits_match = all(
        abs(row["power_limit_watts"] - expected_limits[row["index"]]) < 0.6
        for row in gpus
    )
    warning_temperature = int(hardware["thermal_warning_c"])
    failure_temperature = int(hardware["thermal_failure_c"])
    root_min = int(hardware["wsl_root_free_failure_gib"]) * 1024**3
    d_min = int(hardware["d_drive_free_failure_gib"]) * 1024**3

    controller_processes = _matching_processes(
        inventory, "run_alphazuma_55_postprocess_parallel_v2.py"
    )
    deployer_processes = _matching_processes(
        inventory, "deploy_alphazuma_55_postprocess_parallel_v2.py"
    )
    errors: list[str] = []
    warnings: list[str] = []
    if deployment_status.get("status") == "FAILED" or controller.get("error") is not None:
        errors.append("deployment_or_controller_error")
    if training["failed_routes"]:
        errors.append("training_failure_receipt")
    if root_disk["free_bytes"] < root_min:
        errors.append("wsl_root_free_below_threshold")
    if d_disk["free_bytes"] < d_min:
        errors.append("d_drive_free_below_threshold")
    if any(row["temperature_c"] >= failure_temperature for row in gpus):
        errors.append("gpu_temperature_failure")
    if evaluation is not None and evaluation["status"] == "ERROR":
        errors.append("evaluation_error")
    if not embargo_intact:
        errors.append("formal_seed_embargo_boundary_violation")

    if training["max_running_age_seconds"] > stale_seconds:
        warnings.append("stale_training_status")
    if training["missing_running_processes"]:
        warnings.append("missing_running_training_process")
    if not controller_processes or not deployer_processes:
        warnings.append("missing_controller_or_deployer_process")
    if any(row["temperature_c"] >= warning_temperature for row in gpus):
        warnings.append("gpu_temperature_warning")
    if memory["swap_used_bytes"] > 0:
        warnings.append("swap_in_use")
    if not power_limits_match:
        warnings.append("gpu_power_limit_mismatch")
    if str(hardware["windows_training_power_plan"].split()[-1]).casefold() not in active_plan.casefold():
        warnings.append("windows_power_plan_mismatch")
    if gpu_guard.get("error") or plan_guard.get("error"):
        warnings.append("hardware_guard_error")

    status = "ERROR" if errors else ("WARNING" if warnings else "PASS")
    return {
        "schema": "zuma-rl.alphazuma-55-weekend-live-snapshot",
        "version": 1,
        "status": status,
        "checked_utc": now.isoformat(),
        "errors": errors,
        "warnings": warnings,
        "master_preregistration": {
            "path": str(master_path),
            "sha256": _sha256(master_path),
        },
        "deployment_preregistration": {
            "path": str(deployment_path),
            "sha256": _sha256(deployment_path),
        },
        "postprocess_preregistration": {
            "path": str(postprocess_path),
            "sha256": _sha256(postprocess_path),
        },
        "deployment": {
            "path": str(deployment_status_path),
            "status": deployment_status.get("status"),
            "stage": deployment_status.get("stage"),
            "error": deployment_status.get("error"),
            "processes": deployer_processes,
        },
        "controller": {
            "path": str(controller_path),
            "status": controller.get("status"),
            "phase": phase,
            "updated_utc": controller.get("updated_utc"),
            "error": controller.get("error"),
            "processes": controller_processes,
        },
        "training": training,
        "formal_output_boundary": {
            "roots": formal_roots,
            "embargo_boundary_intact": embargo_intact,
        },
        "evaluation": evaluation,
        "hardware": {
            "gpus": gpus,
            "power_limits_match": power_limits_match,
            "memory": memory,
            "wsl_root": root_disk,
            "d_drive": d_disk,
            "gpu_guard": {
                "path": str(gpu_guard_path),
                "sha256": _sha256(gpu_guard_path),
                "status": gpu_guard.get("status"),
                "error": gpu_guard.get("error"),
            },
            "power_plan_guard": {
                "path": str(plan_guard_path),
                "sha256": _sha256(plan_guard_path),
                "status": plan_guard.get("status"),
                "error": plan_guard.get("error"),
            },
            "active_power_plan": active_plan,
        },
        "classification": "read_only_operational_snapshot",
        "formal_seed_consumption": "none_by_snapshot",
        "formal_gate_effect": "none",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-preregistration", required=True, type=Path)
    parser.add_argument("--deployment-preregistration", required=True, type=Path)
    parser.add_argument("--stale-seconds", type=float, default=180.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not math.isfinite(args.stale_seconds) or args.stale_seconds <= 0:
        raise ValueError("--stale-seconds must be finite and positive")
    try:
        snapshot = build_snapshot(
            master_path=args.master_preregistration.expanduser().resolve(strict=True),
            deployment_path=args.deployment_preregistration.expanduser().resolve(
                strict=True
            ),
            stale_seconds=args.stale_seconds,
        )
    except BaseException as error:
        snapshot = {
            "schema": "zuma-rl.alphazuma-55-weekend-live-snapshot",
            "version": 1,
            "status": "ERROR",
            "checked_utc": datetime.now(timezone.utc).isoformat(),
            "errors": [
                {"type": type(error).__name__, "message": str(error)}
            ],
            "classification": "read_only_operational_snapshot",
            "formal_seed_consumption": "none_by_snapshot",
            "formal_gate_effect": "none",
        }
    print(json.dumps(snapshot, ensure_ascii=False, allow_nan=False, sort_keys=True))
    return 0 if snapshot["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
