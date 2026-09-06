"""Bind the AlphaZuma 55 goal watcher to a superseding source postprocess."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
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


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _write_exclusive(path: Path, value: dict[str, Any]) -> None:
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
        raise FileExistsError(f"refusing to overwrite goal artifact: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def build(
    *,
    source_plan_path: Path,
    expected_source_plan_sha256: str,
    source_finalizer_path: Path,
    expected_source_finalizer_sha256: str,
    postprocess_path: Path,
    expected_postprocess_sha256: str,
    source_independent_receipt: Path,
    goal_receipt: Path,
    status_root: Path,
    plan_output: Path,
    finalizer_output: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    _require(_sha256(source_plan_path) == expected_source_plan_sha256, "source goal plan hash differs")
    _require(_sha256(source_finalizer_path) == expected_source_finalizer_sha256, "source finalizer hash differs")
    _require(_sha256(postprocess_path) == expected_postprocess_sha256, "postprocess hash differs")
    source_plan = _read(source_plan_path)
    source_finalizer = _read(source_finalizer_path)
    postprocess = _read(postprocess_path)
    _require(
        source_plan.get("schema") == "zuma-rl.alphazuma-55-single-policy-goal-audit-plan"
        and source_plan.get("status") == "FROZEN_BEFORE_GOAL_COMPLETION",
        "unexpected source goal plan",
    )
    _require(
        source_finalizer.get("schema") == "zuma-rl.alphazuma-55-single-policy-goal-finalizer"
        and source_finalizer.get("status") == "FROZEN_BEFORE_GOAL_COMPLETION",
        "unexpected source finalizer",
    )
    _require(
        postprocess.get("schema") == "zuma-rl.alphazuma-55-postprocess-preregistration"
        and postprocess.get("status") == "FROZEN_BEFORE_SELECTION"
        and len(postprocess.get("training_routes", [])) == 7,
        "unexpected superseding postprocess",
    )
    _require(not source_independent_receipt.exists(), "source audit receipt already exists")
    _require(not goal_receipt.exists(), "goal receipt already exists")
    _require(not status_root.exists(), "goal watcher status root already exists")
    _require(not plan_output.exists() and not finalizer_output.exists(), "goal outputs already exist")

    now = datetime.now(timezone.utc).isoformat()
    plan = copy.deepcopy(source_plan)
    plan["created_utc"] = now
    plan["campaign_id"] = "alphazuma-55-single-policy-goal-audit-s99081624"
    plan["supersedes"] = {
        "path": str(source_plan_path),
        "sha256": expected_source_plan_sha256,
    }
    source_rows = [row for row in plan["campaigns"] if row["id"] == "source-weekend-s81081401"]
    _require(len(source_rows) == 1, "source campaign entry differs")
    source = source_rows[0]
    source["audit_input"] = {
        "path": str(postprocess_path),
        "sha256": expected_postprocess_sha256,
    }
    source["independent_receipt"] = str(source_independent_receipt)
    plan["source_postprocess_supersession"] = {
        "old_audit_input": source_plan["campaigns"][-1]["audit_input"],
        "new_audit_input": source["audit_input"],
        "formal_seed_ranges_and_campaign_master_unchanged": True,
        "goal_requirements_and_priority_unchanged": True,
    }
    plan["implementation"]["builder"] = {
        "path": str(Path(__file__).resolve()),
        "sha256": _sha256(Path(__file__).resolve()),
    }
    plan["outputs"]["goal_independent_receipt"] = str(goal_receipt)
    _write_exclusive(plan_output, plan)
    plan_sha256 = _sha256(plan_output)

    finalizer = copy.deepcopy(source_finalizer)
    finalizer["created_utc"] = now
    finalizer["campaign_id"] = "alphazuma-55-single-policy-goal-finalizer-s99081625"
    finalizer["supersedes"] = {
        "path": str(source_finalizer_path),
        "sha256": expected_source_finalizer_sha256,
    }
    finalizer["goal_audit_plan"] = {"path": str(plan_output), "sha256": plan_sha256}
    finalizer["implementation"]["builder"] = {
        "path": str(Path(__file__).resolve()),
        "sha256": _sha256(Path(__file__).resolve()),
    }
    finalizer["outputs"] = {
        "status_root": str(status_root),
        "goal_receipt": str(goal_receipt),
    }
    _write_exclusive(finalizer_output, finalizer)
    return plan, finalizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-plan", required=True, type=Path)
    parser.add_argument("--expected-source-plan-sha256", required=True)
    parser.add_argument("--source-finalizer", required=True, type=Path)
    parser.add_argument("--expected-source-finalizer-sha256", required=True)
    parser.add_argument("--postprocess", required=True, type=Path)
    parser.add_argument("--expected-postprocess-sha256", required=True)
    parser.add_argument("--source-independent-receipt", required=True, type=Path)
    parser.add_argument("--goal-receipt", required=True, type=Path)
    parser.add_argument("--status-root", required=True, type=Path)
    parser.add_argument("--plan-output", required=True, type=Path)
    parser.add_argument("--finalizer-output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan_output = args.plan_output.expanduser().resolve()
    finalizer_output = args.finalizer_output.expanduser().resolve()
    plan, finalizer = build(
        source_plan_path=args.source_plan.expanduser().resolve(strict=True),
        expected_source_plan_sha256=str(args.expected_source_plan_sha256),
        source_finalizer_path=args.source_finalizer.expanduser().resolve(strict=True),
        expected_source_finalizer_sha256=str(args.expected_source_finalizer_sha256),
        postprocess_path=args.postprocess.expanduser().resolve(strict=True),
        expected_postprocess_sha256=str(args.expected_postprocess_sha256),
        source_independent_receipt=args.source_independent_receipt.expanduser().resolve(),
        goal_receipt=args.goal_receipt.expanduser().resolve(),
        status_root=args.status_root.expanduser().resolve(),
        plan_output=plan_output,
        finalizer_output=finalizer_output,
    )
    print(
        json.dumps(
            {
                "status": "FROZEN_BEFORE_GOAL_COMPLETION",
                "plan": {"path": str(plan_output), "sha256": _sha256(plan_output)},
                "finalizer": {"path": str(finalizer_output), "sha256": _sha256(finalizer_output)},
                "source_audit_input": plan["campaigns"][-1]["audit_input"],
                "outputs": finalizer["outputs"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
