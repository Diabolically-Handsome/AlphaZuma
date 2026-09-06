"""Emit one strict, read-only health snapshot for the AlphaZuma V1.1 portfolio."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"timestamp is not timezone-aware: {value}")
    return parsed.astimezone(timezone.utc)


def summarize_evaluation_shards(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate and summarize shard progress from already parsed JSON rows."""

    shard_rows: list[dict[str, Any]] = []
    indices: set[int] = set()
    for row in rows:
        shard_index = int(row["shard_index"])
        if shard_index in indices:
            raise ValueError("duplicate shard index")
        indices.add(shard_index)
        expected = int(row["expected_attempts"])
        completed = int(row["completed_attempts"])
        wall_seconds = float(row["runtime"]["wall_seconds"])
        if expected <= 0 or completed < 0 or completed > expected:
            raise ValueError("invalid shard attempt counts")
        if not math.isfinite(wall_seconds) or wall_seconds < 0:
            raise ValueError("invalid shard wall time")
        rate = completed / wall_seconds if completed and wall_seconds else None
        eta_seconds = (
            (expected - completed) / rate
            if rate is not None and completed < expected
            else (0.0 if completed == expected else None)
        )
        shard_rows.append(
            {
                "shard_index": shard_index,
                "status": str(row["status"]),
                "completed_attempts": completed,
                "expected_attempts": expected,
                "progress_fraction": completed / expected,
                "wall_seconds": wall_seconds,
                "attempts_per_second": rate,
                "eta_seconds": eta_seconds,
                "error": row.get("error"),
            }
        )
    shard_rows.sort(key=lambda row: row["shard_index"])
    expected_total = sum(row["expected_attempts"] for row in shard_rows)
    completed_total = sum(row["completed_attempts"] for row in shard_rows)
    eta_values = [
        float(row["eta_seconds"])
        for row in shard_rows
        if row["eta_seconds"] is not None
    ]
    return {
        "shard_count": len(shard_rows),
        "status": (
            "ERROR"
            if any(row["status"] == "ERROR" for row in shard_rows)
            else (
                "COMPLETE"
                if shard_rows
                and all(row["status"] == "COMPLETE" for row in shard_rows)
                else "RUNNING"
            )
        ),
        "completed_attempts": completed_total,
        "expected_attempts": expected_total,
        "progress_fraction": (
            completed_total / expected_total if expected_total else None
        ),
        "eta_seconds": max(eta_values) if eta_values else None,
        "shards": shard_rows,
    }


def _evaluation_snapshot(path: Path) -> dict[str, Any]:
    files = sorted(path.glob("matrix-shard-*.json")) if path.exists() else []
    if not files:
        return {
            "directory": str(path),
            "shard_count": 0,
            "status": "NOT_STARTED",
            "completed_attempts": 0,
            "expected_attempts": 0,
            "progress_fraction": None,
            "eta_seconds": None,
            "shards": [],
        }
    summary = summarize_evaluation_shards([_read_json(path) for path in files])
    summary["directory"] = str(path)
    return summary


def summarize_training_liveness(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize route liveness without treating completed routes as stale."""

    if not rows:
        raise ValueError("training rows must not be empty")
    running_rows = [
        row
        for row in rows
        if row["status"] == "RUNNING" and not row["completion_exists"]
    ]
    unexpected_rows = [
        row
        for row in rows
        if row["status"] != "RUNNING"
        and not row["completion_exists"]
        and not row["failure_exists"]
    ]
    return {
        "route_count": len(rows),
        "running_routes": len(running_rows),
        "completed_routes": sum(row["completion_exists"] for row in rows),
        "failed_routes": sum(row["failure_exists"] for row in rows),
        "unexpected_incomplete_routes": len(unexpected_rows),
        "total_steps_this_run": sum(row["steps_this_run"] for row in rows),
        "max_status_age_seconds": max(row["age_seconds"] for row in rows),
        "max_running_status_age_seconds": (
            max(row["age_seconds"] for row in running_rows)
            if running_rows
            else 0.0
        ),
    }


def classify_preselection_entries(names: list[str]) -> dict[str, Any]:
    """Classify output-root entries allowed before candidate selection starts."""

    allowed = {"controller_status.json"}
    temporary_pattern = re.compile(r"^\.controller_status\.json\.\d+\.tmp$")
    unique = sorted(set(names))
    temporary = [name for name in unique if temporary_pattern.fullmatch(name)]
    unexpected = [
        name
        for name in unique
        if name not in allowed and name not in temporary
    ]
    return {
        "entries": unique,
        "temporary_atomic_write_entries": temporary,
        "unexpected_entries": unexpected,
        "embargo_boundary_intact": not unexpected,
    }


def _training_snapshot(master: dict[str, Any], now: datetime) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for run in master["runs"]:
        run_dir = Path(str(run["run_dir"]))
        status_path = run_dir / "training_status.json"
        completion_path = run_dir / "completion.json"
        failure_path = run_dir / "failure.json"
        status = _read_json(status_path)
        if str(status.get("run_id")) != str(run["training_run_id"]):
            raise ValueError(f"training status run id mismatch: {run['id']}")
        updated = _utc(str(status["updated_utc"]))
        rows.append(
            {
                "run": str(run["id"]),
                "status": str(status["status"]),
                "updated_utc": updated.isoformat().replace("+00:00", "Z"),
                "age_seconds": (now - updated).total_seconds(),
                "steps_this_run": int(status["steps_this_run"]),
                "episodes": int(status["episodes"]),
                "wins": int(status["wins"]),
                "completion_exists": completion_path.exists(),
                "failure_exists": failure_path.exists(),
            }
        )
    return {**summarize_training_liveness(rows), "routes": rows}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-preregistration", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--forecast-receipt", type=Path)
    parser.add_argument("--stale-seconds", type=float, default=180.0)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not math.isfinite(args.stale_seconds) or args.stale_seconds <= 0:
        raise ValueError("--stale-seconds must be finite and positive")
    master_path = args.master_preregistration.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve(strict=True)
    master = _read_json(master_path)
    controller_path = output_root / "controller_status.json"
    controller = _read_json(controller_path)
    now = datetime.now(timezone.utc)
    training = _training_snapshot(master, now)
    phase = str(controller["phase"])
    preselection_inventory = classify_preselection_entries(
        [path.name for path in output_root.iterdir()]
    )
    evaluation: dict[str, Any] | None = None
    projected_remaining: float | None = None
    if phase == "SELECTION_EVALUATION":
        evaluation = _evaluation_snapshot(output_root / "selection-evaluation")
        projected_remaining = evaluation["eta_seconds"]
        if args.forecast_receipt is not None and projected_remaining is not None:
            forecast = _read_json(args.forecast_receipt.resolve(strict=True))
            projected_remaining += (
                float(forecast["forecast"]["final_blind"]["forecast_seconds"])
                * 1.20
                + 300.0
            )
    elif phase == "FINAL_BLIND_EVALUATION":
        evaluation = _evaluation_snapshot(output_root / "final-blind-evaluation")
        projected_remaining = evaluation["eta_seconds"]
        if projected_remaining is not None:
            projected_remaining += 120.0

    goal_end = _utc(str(master["schedule"]["goal_end_utc"]))
    seconds_to_goal_end = (goal_end - now).total_seconds()
    projected_buffer = (
        seconds_to_goal_end - projected_remaining
        if projected_remaining is not None
        else None
    )
    controller_error = controller.get("error") is not None
    evaluation_error = evaluation is not None and evaluation["status"] == "ERROR"
    stale_training = (
        phase == "WAITING_FOR_TRAINING"
        and training["max_running_status_age_seconds"] > args.stale_seconds
    )
    failed_training = training["failed_routes"] > 0
    unexpected_training = training["unexpected_incomplete_routes"] > 0
    embargo_violation = (
        phase == "WAITING_FOR_TRAINING"
        and not preselection_inventory["embargo_boundary_intact"]
    )
    status = (
        "ERROR"
        if controller_error or evaluation_error or failed_training
        else (
            "WARNING"
            if stale_training or unexpected_training or embargo_violation
            else "PASS"
        )
    )
    snapshot = {
        "schema": "zuma-rl.alphazuma-v11-frontier-live-snapshot",
        "version": 1,
        "checked_utc": now.isoformat().replace("+00:00", "Z"),
        "status": status,
        "master_preregistration": {
            "path": str(master_path),
            "sha256": _sha256(master_path),
        },
        "controller": {
            "path": str(controller_path),
            "status": str(controller["status"]),
            "phase": phase,
            "updated_utc": str(controller["updated_utc"]),
            "error": controller.get("error"),
        },
        "training": training,
        "preselection_output_boundary": preselection_inventory,
        "evaluation": evaluation,
        "deadline": {
            "goal_end_utc": goal_end.isoformat().replace("+00:00", "Z"),
            "seconds_to_goal_end": seconds_to_goal_end,
            "projected_remaining_seconds": projected_remaining,
            "projected_buffer_seconds": projected_buffer,
        },
        "classification": "read_only_operational_snapshot",
        "formal_seed_consumption": "none",
        "formal_gate_effect": "none",
    }
    print(json.dumps(snapshot, ensure_ascii=False, allow_nan=False, sort_keys=True))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
