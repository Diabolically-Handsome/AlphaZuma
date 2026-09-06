"""Resume frozen AlphaZuma engineering evaluators in causal-priority order.

This watcher does not select models, consume evaluation seeds, or change any
training/evaluation recipe.  It only implements the SIGCONT order already
frozen in the weekend resource schedule after the preceding route has an
independently audited negative engineering Gate.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEDULE_PATH = PROJECT_ROOT / (
    "diagnostics/alphazuma-55-weekend-evaluation-resource-schedule-"
    "s99081648-v1.json"
)
EXPECTED_SCHEDULE_SHA256 = (
    "sha256:4e60fdde646e89eddf8072af09d463dcf0f3dc02e6767f333c55eca2a92f1413"
)
PAUSE_RECEIPT_PATH = PROJECT_ROOT / (
    "diagnostics/alphazuma-55-weekend-evaluation-resource-schedule-"
    "s99081648-pause-receipt-v1.json"
)


@dataclass(frozen=True)
class Route:
    name: str
    output_root: Path
    audit_path: Path
    successor_status_path: Path
    postprocess_pid: int | None = None
    expected_process_fragment: str | None = None
    resume_receipt_path: Path | None = None

    @property
    def final_report_path(self) -> Path:
        return self.output_root / "final_report.json"

    @property
    def controller_status_path(self) -> Path:
        return self.output_root / "controller_status.json"


COMBINED = Route(
    name="head-only plus all-actions factorial",
    output_root=Path(
        "/mnt/d/ZumaTraining/alphazuma-55-motor-observable-gradual-"
        "headonly-allaim-postprocess-s99081647-v1"
    ),
    audit_path=PROJECT_ROOT
    / (
        "diagnostics/alphazuma-55-motor-observable-gradual-headonly-allaim-"
        "postprocess-s99081647-independent-audit-v1.json"
    ),
    successor_status_path=Path(
        "/mnt/d/ZumaTraining/alphazuma-55-motor-observable-gradual-"
        "headonly-allaim-successor-s99081649-v1/controller/"
        "controller_status.json"
    ),
)

HEAD_ONLY = Route(
    name="head-only fire-aim",
    output_root=Path(
        "/mnt/d/ZumaTraining/alphazuma-55-motor-observable-gradual-"
        "headonly-postprocess-s99081643-v2"
    ),
    audit_path=PROJECT_ROOT
    / (
        "diagnostics/alphazuma-55-motor-observable-gradual-headonly-"
        "postprocess-s99081643-independent-audit-v2.json"
    ),
    successor_status_path=Path(
        "/mnt/d/ZumaTraining/alphazuma-55-motor-observable-gradual-"
        "headonly-successor-s99081644-v1/controller/controller_status.json"
    ),
    postprocess_pid=404830,
    expected_process_fragment=(
        "run_alphazuma_55_motor_observable_gradual_headonly_postprocess_v1.py"
    ),
    resume_receipt_path=PROJECT_ROOT
    / (
        "diagnostics/alphazuma-55-weekend-evaluation-resource-schedule-"
        "s99081648-resume-headonly-receipt-v1.json"
    ),
)

ALL_ACTIONS = Route(
    name="full-policy all-actions aim",
    output_root=Path(
        "/mnt/d/ZumaTraining/alphazuma-55-motor-observable-gradual-"
        "allaim-postprocess-s99081634-v2"
    ),
    audit_path=PROJECT_ROOT
    / (
        "diagnostics/alphazuma-55-motor-observable-gradual-allaim-"
        "postprocess-s99081634-independent-audit-v2.json"
    ),
    successor_status_path=Path(
        "/mnt/d/ZumaTraining/alphazuma-55-motor-observable-gradual-"
        "allaim-successor-s99081636-v2/controller/controller_status.json"
    ),
    postprocess_pid=393431,
    expected_process_fragment=(
        "run_alphazuma_55_motor_observable_gradual_allaim_postprocess_v2.py"
    ),
    resume_receipt_path=PROJECT_ROOT
    / (
        "diagnostics/alphazuma-55-weekend-evaluation-resource-schedule-"
        "s99081648-resume-allaim-receipt-v1.json"
    ),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"), parse_constant=_reject_constant
    )
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_new(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(
            payload,
            stream,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        stream.write("\n")


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("deadline must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def _process_snapshot(pid: int) -> dict[str, Any]:
    proc = Path("/proc") / str(pid)
    status_path = proc / "status"
    cmdline_path = proc / "cmdline"
    if not status_path.exists() or not cmdline_path.exists():
        raise RuntimeError(f"required controller pid {pid} is absent")
    fields: dict[str, str] = {}
    for line in status_path.read_text(encoding="utf-8").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key] = value.strip()
    cmdline = cmdline_path.read_bytes().replace(b"\x00", b" ").decode(
        "utf-8", errors="replace"
    )
    return {
        "pid": pid,
        "state": fields.get("State", ""),
        "cmdline": cmdline.strip(),
    }


def _validate_schedule() -> dict[str, Any]:
    if _sha256(SCHEDULE_PATH) != EXPECTED_SCHEDULE_SHA256:
        raise ValueError("frozen evaluation resource schedule hash differs")
    schedule = _read(SCHEDULE_PATH)
    pause = _read(PAUSE_RECEIPT_PATH)
    if not (
        schedule.get("status")
        == "FROZEN_BEFORE_ANY_CURRENT_ROUTE_ENGINEERING_INFERENCE"
        and schedule.get("evaluation_resume_order")
        == [COMBINED.name, HEAD_ONLY.name, ALL_ACTIONS.name]
        and schedule.get("suspension_contract", {}).get("mechanism")
        == "Linux SIGSTOP followed by SIGCONT on idle postprocess controller only"
        and schedule.get("safety", {}).get(
            "resume_only_one_dual-shard_postprocessor_at_a_time"
        )
        is True
        and pause.get("status") == "PASS"
        and pause.get("resume_required") is True
        and pause.get("schedule", {}).get("sha256")
        == EXPECTED_SCHEDULE_SHA256
    ):
        raise ValueError("resource schedule or pause receipt contract differs")
    return schedule


def _validate_paused_route(route: Route) -> dict[str, Any]:
    if route.postprocess_pid is None or route.expected_process_fragment is None:
        raise ValueError(f"route is not resumable: {route.name}")
    snapshot = _process_snapshot(route.postprocess_pid)
    status = _read(route.controller_status_path)
    if not (
        snapshot["state"].startswith("T")
        and route.expected_process_fragment in snapshot["cmdline"]
        and status.get("status") == "RUNNING"
        and status.get("phase") == "WAITING_FOR_TRAINING"
        and status.get("formal_seed_consumption") == "NONE"
        and status.get("error") is None
    ):
        raise ValueError(f"paused controller contract differs: {route.name}")
    return {"process": snapshot, "controller_status": status}


def _wait_for_gate_decision(
    route: Route, *, deadline: datetime, poll_seconds: float
) -> dict[str, Any]:
    while datetime.now(timezone.utc) < deadline:
        if route.audit_path.exists() and route.final_report_path.exists():
            audit = _read(route.audit_path)
            report = _read(route.final_report_path)
            gate_passed = audit.get("gate_passed")
            if not (
                audit.get("status") == "PASS"
                and isinstance(gate_passed, bool)
                and report.get("status")
                == (
                    "COMPLETE_GATE_PASS"
                    if gate_passed
                    else "COMPLETE_NO_PROMOTION"
                )
                and report.get("promotion_gate", {}).get("passed")
                is gate_passed
            ):
                raise ValueError(f"invalid audited Gate decision: {route.name}")
            return {
                "route": route.name,
                "gate_passed": gate_passed,
                "audit": {
                    "path": str(route.audit_path),
                    "sha256": _sha256(route.audit_path),
                },
                "final_report": {
                    "path": str(route.final_report_path),
                    "sha256": _sha256(route.final_report_path),
                },
            }
        if route.controller_status_path.exists():
            status = _read(route.controller_status_path)
            if status.get("status") == "ERROR" or status.get("phase") == "ERROR":
                raise RuntimeError(
                    f"engineering controller failed before an audit: {route.name}: "
                    f"{status.get('error')}"
                )
        time.sleep(poll_seconds)
    raise TimeoutError(f"deadline reached before Gate decision: {route.name}")


def _wait_for_no_promotion_successor(
    route: Route, *, deadline: datetime, poll_seconds: float
) -> dict[str, Any]:
    while datetime.now(timezone.utc) < deadline:
        if route.successor_status_path.exists():
            status = _read(route.successor_status_path)
            if (
                status.get("status") == "COMPLETE_NO_PROMOTION"
                and status.get("phase") == "COMPLETE_NO_PROMOTION"
                and status.get("formal_seed_consumption") == "NONE"
                and status.get("error") is None
            ):
                return status
            if status.get("status") == "ERROR" or status.get("phase") == "ERROR":
                raise RuntimeError(
                    f"successor failed after negative Gate: {route.name}: "
                    f"{status.get('error')}"
                )
        time.sleep(poll_seconds)
    raise TimeoutError(
        f"deadline reached before no-promotion successor sealed: {route.name}"
    )


def _resume_route(
    route: Route,
    *,
    trigger: dict[str, Any],
    predecessor_route: Route,
    predecessor_successor: dict[str, Any],
) -> dict[str, Any]:
    if route.postprocess_pid is None or route.resume_receipt_path is None:
        raise ValueError(f"route has no resume contract: {route.name}")
    if route.resume_receipt_path.exists():
        raise FileExistsError(f"resume receipt already exists: {route.resume_receipt_path}")
    before = _validate_paused_route(route)
    os.kill(route.postprocess_pid, signal.SIGCONT)
    after: dict[str, Any] | None = None
    for _attempt in range(50):
        time.sleep(0.1)
        after = _process_snapshot(route.postprocess_pid)
        if not str(after["state"]).startswith("T"):
            break
    if after is None or str(after["state"]).startswith("T"):
        raise RuntimeError(f"SIGCONT did not resume controller: {route.name}")
    receipt = {
        "schema": "zuma-rl.alphazuma-55-evaluation-controller-resume-receipt",
        "version": 1,
        "status": "PASS",
        "observed_utc": _utc_now(),
        "schedule": {
            "path": str(SCHEDULE_PATH),
            "sha256": EXPECTED_SCHEDULE_SHA256,
        },
        "trigger": trigger,
        "predecessor_successor": {
            "route": predecessor_route.name,
            "path": str(predecessor_route.successor_status_path),
            "sha256": _sha256(predecessor_route.successor_status_path),
            "status": predecessor_successor,
        },
        "controller": {
            "route": route.name,
            "pid": route.postprocess_pid,
            "signal": "SIGCONT",
            "before": before,
            "after_process": after,
        },
        "single_postprocessor_resumed": True,
        "formal_seed_consumption_before_resume": "NONE",
        "model_or_recipe_change": False,
        "candidate_or_gate_change": False,
        "power_restore_authority": False,
    }
    _write_new(route.resume_receipt_path, receipt)
    return receipt


def run(*, deadline: datetime, poll_seconds: float) -> int:
    schedule = _validate_schedule()
    initial = {
        HEAD_ONLY.name: _validate_paused_route(HEAD_ONLY),
        ALL_ACTIONS.name: _validate_paused_route(ALL_ACTIONS),
    }
    print(
        json.dumps(
            {
                "status": "WAITING_FOR_COMBINED_GATE",
                "observed_utc": _utc_now(),
                "deadline_utc": deadline.isoformat(),
                "schedule_sha256": EXPECTED_SCHEDULE_SHA256,
                "initial": initial,
            },
            ensure_ascii=False,
            allow_nan=False,
        ),
        flush=True,
    )

    combined = _wait_for_gate_decision(
        COMBINED, deadline=deadline, poll_seconds=poll_seconds
    )
    print(json.dumps(combined, ensure_ascii=False, allow_nan=False), flush=True)
    if combined["gate_passed"]:
        return 0
    combined_successor = _wait_for_no_promotion_successor(
        COMBINED, deadline=deadline, poll_seconds=poll_seconds
    )
    _resume_route(
        HEAD_ONLY,
        trigger=combined,
        predecessor_route=COMBINED,
        predecessor_successor=combined_successor,
    )

    head_only = _wait_for_gate_decision(
        HEAD_ONLY, deadline=deadline, poll_seconds=poll_seconds
    )
    print(json.dumps(head_only, ensure_ascii=False, allow_nan=False), flush=True)
    if head_only["gate_passed"]:
        return 0
    head_only_successor = _wait_for_no_promotion_successor(
        HEAD_ONLY, deadline=deadline, poll_seconds=poll_seconds
    )
    _resume_route(
        ALL_ACTIONS,
        trigger=head_only,
        predecessor_route=HEAD_ONLY,
        predecessor_successor=head_only_successor,
    )

    all_actions = _wait_for_gate_decision(
        ALL_ACTIONS, deadline=deadline, poll_seconds=poll_seconds
    )
    print(json.dumps(all_actions, ensure_ascii=False, allow_nan=False), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deadline-utc", required=True)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    deadline = _parse_utc(str(args.deadline_utc))
    schedule = _validate_schedule()
    initial = {
        HEAD_ONLY.name: _validate_paused_route(HEAD_ONLY),
        ALL_ACTIONS.name: _validate_paused_route(ALL_ACTIONS),
    }
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "VALID",
                    "deadline_utc": deadline.isoformat(),
                    "schedule": {
                        "path": str(SCHEDULE_PATH),
                        "sha256": EXPECTED_SCHEDULE_SHA256,
                        "resume_order": schedule["evaluation_resume_order"],
                    },
                    "paused_routes": initial,
                    "formal_seed_consumption": "NONE",
                },
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        )
        return 0
    return run(deadline=deadline, poll_seconds=max(1.0, args.poll_seconds))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
