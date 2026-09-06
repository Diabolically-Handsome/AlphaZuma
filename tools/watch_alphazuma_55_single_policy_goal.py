"""Watch frozen AlphaZuma 55 audits and finalize the single-policy goal."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools import audit_alphazuma_55_single_policy_goal as goal


SCRIPT_PATH = Path(__file__).resolve()
RESTORE_TIMEOUT_SECONDS = 180.0


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _write_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _write_marker(path: Path, reason: str) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="ascii", newline="\n") as stream:
        stream.write(f"{_now().isoformat()} {reason}\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError:
        pass
    finally:
        temporary.unlink(missing_ok=True)


def _receipt_signature(campaigns: list[dict[str, Any]]) -> tuple[Any, ...]:
    signature: list[Any] = []
    for campaign in campaigns:
        path = Path(str(campaign["independent_receipt"])).expanduser().resolve()
        if path.exists():
            stat = path.stat()
            signature.append((str(path), stat.st_mtime_ns, stat.st_size))
        else:
            signature.append((str(path), None, None))
    return tuple(signature)


def _ensure_goal_receipt(path: Path, result: dict[str, Any]) -> dict[str, Any]:
    goal._require(result.get("status") == "PASS", "goal result is not PASS")
    if path.exists():
        existing = goal._read(path)
        goal._require(existing.get("status") == "PASS", "existing goal receipt is not PASS")
        goal._require(
            existing.get("plan") == result.get("plan")
            and existing.get("selected_campaign_id")
            == result.get("selected_campaign_id")
            and existing.get("selected_single_policy")
            == result.get("selected_single_policy"),
            "existing goal receipt differs",
        )
        return existing
    goal._write_exclusive(path, result)
    return goal._read(path)


def _restore_evidence(
    gpu_guard_path: Path,
    power_plan_guard_path: Path,
    *,
    timeout_seconds: float = RESTORE_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        gpu = goal._read(gpu_guard_path) if gpu_guard_path.exists() else None
        plan = goal._read(power_plan_guard_path) if power_plan_guard_path.exists() else None
        if gpu and gpu.get("status") == "ERROR":
            raise RuntimeError(f"GPU power guard failed: {gpu.get('error')}")
        if plan and plan.get("status") == "ERROR":
            raise RuntimeError(f"Windows power-plan guard failed: {plan.get('error')}")
        gpu_restored = bool(
            gpu
            and gpu.get("status") in {"RESTORED_ON_REQUEST", "RESTORED_AT_DEADLINE"}
            and {
                int(row["index"]): int(float(row["power_limit_watts"]))
                for row in gpu.get("gpu_state", [])
            }
            == {0: 600, 1: 400}
        )
        plan_restored = bool(
            plan
            and plan.get("status")
            in {"RESTORED_ON_REQUEST", "RESTORED_AT_DEADLINE"}
            and str(plan.get("active_guid", "")).lower()
            == "381b4222-f694-41f0-9685-ff5bb260df2e"
        )
        if gpu_restored and plan_restored:
            return {
                "gpu_power_guard": goal._reference(gpu_guard_path),
                "windows_power_plan_guard": goal._reference(power_plan_guard_path),
                "gpu_limits_watts": {"0": 600, "1": 400},
                "windows_power_plan_guid": "381b4222-f694-41f0-9685-ff5bb260df2e",
                "verified": True,
            }
        time.sleep(5.0)
    raise TimeoutError("power restoration was not verified within the frozen timeout")


def run(finalizer_path: Path, expected_sha256: str, poll_seconds: float) -> int:
    finalizer_path = finalizer_path.expanduser().resolve(strict=True)
    if goal._sha256(finalizer_path) != expected_sha256:
        raise goal.GoalAuditError("goal finalizer plan hash differs")
    finalizer = goal._read(finalizer_path)
    goal._require(
        finalizer.get("schema") == "zuma-rl.alphazuma-55-single-policy-goal-finalizer"
        and finalizer.get("version") == 1
        and finalizer.get("status") == "FROZEN_BEFORE_GOAL_COMPLETION",
        "goal finalizer identity differs",
    )
    implementation = finalizer.get("implementation")
    goal._require(isinstance(implementation, dict), "finalizer implementation is absent")
    goal._require(
        goal._artifact(implementation.get("watcher"), "goal watcher") == SCRIPT_PATH,
        "finalizer binds another watcher",
    )
    goal._require(
        goal._artifact(implementation.get("goal_auditor"), "goal auditor")
        == goal.SCRIPT_PATH,
        "finalizer binds another goal auditor",
    )
    goal_plan_path = goal._artifact(finalizer.get("goal_audit_plan"), "goal audit plan")
    goal_plan = goal._read(goal_plan_path)
    deadline = goal._utc(str(finalizer["deadline_utc"]))
    goal._require(
        deadline == goal._utc(str(goal_plan["deadline_utc"])),
        "finalizer deadline differs from goal audit plan",
    )
    outputs = finalizer.get("outputs")
    goal._require(isinstance(outputs, dict), "finalizer outputs are absent")
    status_root = Path(str(outputs["status_root"])).expanduser().resolve()
    status_root.mkdir(parents=True, exist_ok=False)
    status_path = status_root / "watcher_status.json"
    receipt_path = Path(str(outputs["goal_receipt"])).expanduser().resolve()
    goal._require(
        receipt_path
        == Path(str(goal_plan["outputs"]["goal_independent_receipt"]))
        .expanduser()
        .resolve(),
        "finalizer goal receipt differs",
    )
    hardware = finalizer.get("hardware_restoration")
    goal._require(isinstance(hardware, dict), "hardware restoration contract is absent")
    gpu_marker = Path(str(hardware["gpu_restore_request"])).expanduser().resolve()
    plan_marker = Path(str(hardware["power_plan_restore_request"])).expanduser().resolve()
    gpu_guard = Path(str(hardware["gpu_guard_receipt"])).expanduser().resolve()
    plan_guard = Path(str(hardware["power_plan_guard_receipt"])).expanduser().resolve()
    state: dict[str, Any] = {
        "schema": "zuma-rl.alphazuma-55-single-policy-goal-watcher-status",
        "version": 1,
        "status": "RUNNING",
        "phase": "WAITING_FOR_CAMPAIGN_AUDITS",
        "started_utc": _now().isoformat(),
        "updated_utc": _now().isoformat(),
        "finalizer": goal._reference(finalizer_path),
        "goal_audit_plan": goal._reference(goal_plan_path),
        "deadline_utc": deadline.isoformat(),
        "error": None,
    }
    _write_atomic(status_path, state)
    last_signature: tuple[Any, ...] | None = None
    try:
        while True:
            now = _now()
            signature = _receipt_signature(list(goal_plan["campaigns"]))
            if signature != last_signature or now >= deadline:
                result = goal.audit(goal_plan_path, now=now)
                last_signature = signature
                state.update(
                    {
                        "updated_utc": now.isoformat(),
                        "goal_status": result["status"],
                        "campaigns": [
                            {
                                "id": row["id"],
                                "status": row["status"],
                                "eligible": row["eligible"],
                            }
                            for row in result["campaigns"]
                        ],
                        "selection_blocked_by": result[
                            "selection_blocked_by_pending_higher_priority"
                        ],
                    }
                )
                _write_atomic(status_path, state)
                if result["status"] == "PASS":
                    receipt = _ensure_goal_receipt(receipt_path, result)
                    state.update(
                        {
                            "phase": "RESTORING_POWER",
                            "updated_utc": _now().isoformat(),
                            "goal_receipt": goal._reference(receipt_path),
                            "selected_campaign_id": receipt["selected_campaign_id"],
                            "selected_single_policy": receipt["selected_single_policy"],
                        }
                    )
                    _write_atomic(status_path, state)
                    _write_marker(gpu_marker, "goal-independent-audit-pass")
                    _write_marker(plan_marker, "goal-independent-audit-pass")
                    restored = _restore_evidence(gpu_guard, plan_guard)
                    state.update(
                        {
                            "status": "COMPLETE",
                            "phase": "COMPLETE",
                            "updated_utc": _now().isoformat(),
                            "power_restoration": restored,
                        }
                    )
                    _write_atomic(status_path, state)
                    return 0
                if now >= deadline:
                    state.update(
                        {
                            "phase": "RESTORING_POWER_AT_DEADLINE",
                            "updated_utc": _now().isoformat(),
                        }
                    )
                    _write_atomic(status_path, state)
                    _write_marker(gpu_marker, "goal-deadline-reached")
                    _write_marker(plan_marker, "goal-deadline-reached")
                    restored = _restore_evidence(gpu_guard, plan_guard)
                    state.update(
                        {
                            "status": "COMPLETE_GOAL_NOT_ACHIEVED",
                            "phase": "COMPLETE",
                            "updated_utc": _now().isoformat(),
                            "power_restoration": restored,
                        }
                    )
                    _write_atomic(status_path, state)
                    return 2
            remaining = max(0.0, (deadline - _now()).total_seconds())
            time.sleep(min(poll_seconds, remaining) if remaining else 0.1)
    except BaseException as error:
        state.update(
            {
                "status": "FAILED",
                "phase": "FAILED",
                "updated_utc": _now().isoformat(),
                "error": {"type": type(error).__name__, "message": str(error)},
            }
        )
        _write_atomic(status_path, state)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finalizer", required=True, type=Path)
    parser.add_argument("--expected-finalizer-sha256", required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args.finalizer, str(args.expected_finalizer_sha256), args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
