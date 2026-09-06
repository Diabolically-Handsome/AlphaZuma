"""Build the frozen eight-route AlphaZuma 55 goal audit and finalizer plans."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools import audit_alphazuma_55_motor_observable_successor_v1 as motor_auditor
from tools import audit_alphazuma_55_single_policy_goal_v5 as goal_auditor
from tools import watch_alphazuma_55_single_policy_goal_v5 as goal_watcher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()
CREATED_UTC = "2026-08-16T04:38:00Z"
OLD_GOAL_SHA256 = "sha256:8bfa79a5977be5f5380bf261a7beff36bda8e0b816e8a165727d0342407b76d5"
OLD_FINALIZER_SHA256 = "sha256:da425431a1eca688f67b3590706e88be018b152ab44b78885034c7e8e7fe4db6"
MOTOR_MASTER_SHA256 = "sha256:fef69405434c91a9da7f3ab3f78439072a2e6b071520229169aad4185ff12a46"
MOTOR_AUDITOR_SHA256 = "sha256:ede1646441c47707e7239ff87aad3b5f58965bc167553f554811164e3b8c0aa5"
REGISTRY_SHA256 = "sha256:2f6996cd46778455e1069936da7a72b041882265a2c533b0db7faeddc63c5680"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _strict_read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value


def _ref(path: Path) -> dict[str, str]:
    path = path.resolve(strict=True)
    return {"path": str(path), "sha256": _sha256(path)}


def _expect(path: Path, expected: str, label: str) -> None:
    actual = _sha256(path.resolve(strict=True))
    if actual != expected:
        raise RuntimeError(f"{label} hash differs: {actual}")


def _encoded(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def _bytes_sha256(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError:
        raise FileExistsError(f"refusing to overwrite frozen plan: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def build(goal_output: Path, finalizer_output: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    old_goal_path = PROJECT_ROOT / "diagnostics/alphazuma-55-single-policy-goal-audit-s99081611-preregistration-v4.json"
    old_finalizer_path = PROJECT_ROOT / "diagnostics/alphazuma-55-single-policy-goal-finalizer-s99081612-preregistration-v4.json"
    motor_master_path = PROJECT_ROOT / "diagnostics/alphazuma-55-motor-observable-successor-s99081618-preregistration-v1.json"
    registry_path = PROJECT_ROOT / "diagnostics/alphazuma-55-weekend-formal-seed-registry-s99081619-v5.json"
    _expect(old_goal_path, OLD_GOAL_SHA256, "V4 goal plan")
    _expect(old_finalizer_path, OLD_FINALIZER_SHA256, "V4 finalizer")
    _expect(motor_master_path, MOTOR_MASTER_SHA256, "motor successor master")
    _expect(motor_auditor.SCRIPT_PATH, MOTOR_AUDITOR_SHA256, "motor auditor")
    _expect(registry_path, REGISTRY_SHA256, "formal seed registry V5")

    goal_output = goal_output.expanduser().resolve()
    finalizer_output = finalizer_output.expanduser().resolve()
    old_goal = _strict_read(old_goal_path)
    goal = deepcopy(old_goal)
    goal.update(
        {
            "campaign_id": "alphazuma-55-single-policy-goal-audit-s99081620",
            "created_utc": CREATED_UTC,
            "supersedes": _ref(old_goal_path),
            "formal_seed_registry": _ref(registry_path),
        }
    )
    motor_campaign = {
        "id": "motor-observable-successor-s99081618",
        "kind": "motor_observable_successor_v1",
        "seed_registry_campaign_id": "motor-observable-successor-s99081618",
        "campaign_master": _ref(motor_master_path),
        "audit_input": _ref(motor_master_path),
        "independent_auditor": _ref(motor_auditor.SCRIPT_PATH),
        "independent_receipt": str(
            PROJECT_ROOT
            / "diagnostics/alphazuma-55-motor-observable-successor-s99081618-independent-audit-v1.json"
        ),
    }
    goal["campaigns"] = [motor_campaign, *goal["campaigns"]]
    goal["selection"] = {
        **goal["selection"],
        "campaign_priority": [row["id"] for row in goal["campaigns"]],
        "rationale": (
            "Prefer the motor-observable closed-loop successor because it exposes "
            "the hidden actuator state while preserving one frozen policy; then "
            "retain the seven already frozen fallbacks in their V4 order."
        ),
    }
    goal["implementation"] = {
        "goal_auditor": _ref(goal_auditor.SCRIPT_PATH),
        "builder": _ref(SCRIPT_PATH),
    }
    goal["outputs"] = {
        "goal_independent_receipt": str(
            PROJECT_ROOT
            / "diagnostics/alphazuma-55-single-policy-goal-independent-audit-s99081620-v5.json"
        )
    }
    goal_payload = _encoded(goal)
    goal_reference = {
        "path": str(goal_output),
        "sha256": _bytes_sha256(goal_payload),
    }

    old_finalizer = _strict_read(old_finalizer_path)
    finalizer = deepcopy(old_finalizer)
    finalizer.update(
        {
            "campaign_id": "alphazuma-55-single-policy-goal-finalizer-s99081621",
            "created_utc": CREATED_UTC,
            "supersedes": _ref(old_finalizer_path),
            "goal_audit_plan": goal_reference,
        }
    )
    finalizer["implementation"] = {
        "goal_auditor": _ref(goal_auditor.SCRIPT_PATH),
        "watcher": _ref(goal_watcher.SCRIPT_PATH),
        "builder": _ref(SCRIPT_PATH),
    }
    finalizer["outputs"] = {
        "status_root": "/mnt/d/ZumaTraining/alphazuma-55-single-policy-goal-finalizer-s99081621-v5",
        "goal_receipt": goal["outputs"]["goal_independent_receipt"],
    }
    return goal, finalizer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal-output", required=True, type=Path)
    parser.add_argument("--finalizer-output", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    goal, finalizer = build(args.goal_output, args.finalizer_output)
    goal_payload = _encoded(goal)
    finalizer_payload = _encoded(finalizer)
    result = {
        "status": "VALID" if args.validate_only else "FROZEN",
        "goal_plan": {
            "path": str(args.goal_output.expanduser().resolve()),
            "sha256": _bytes_sha256(goal_payload),
            "campaigns": len(goal["campaigns"]),
        },
        "finalizer": {
            "path": str(args.finalizer_output.expanduser().resolve()),
            "sha256": _bytes_sha256(finalizer_payload),
        },
    }
    if not args.validate_only:
        if args.goal_output.expanduser().resolve().exists() or args.finalizer_output.expanduser().resolve().exists():
            raise FileExistsError("refusing to overwrite a frozen V5 goal artifact")
        _write_new(args.goal_output.expanduser().resolve(), goal_payload)
        _write_new(args.finalizer_output.expanduser().resolve(), finalizer_payload)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
