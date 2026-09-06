"""Freeze delayed launch of intent-wide training after effective-wide exits."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools.build_alphazuma_55_eval_contract import _read_json, _sha256


SCRIPT_PATH = Path(__file__).resolve()


def _ref(path: Path) -> dict[str, str]:
    path = path.resolve(strict=True)
    return {"path": str(path), "sha256": _sha256(path)}


def build(
    *,
    intent_preregistration: Path,
    expected_intent_sha256: str,
    source_launch_plan: Path,
    expected_source_plan_sha256: str,
    source_watcher_pid: int,
    original_root: Path,
    status_root: Path,
) -> dict[str, Any]:
    intent_preregistration = intent_preregistration.resolve(strict=True)
    source_launch_plan = source_launch_plan.resolve(strict=True)
    original_root = original_root.resolve(strict=True)
    if _sha256(intent_preregistration) != expected_intent_sha256:
        raise ValueError("intent-wide preregistration hash differs")
    if _sha256(source_launch_plan) != expected_source_plan_sha256:
        raise ValueError("effective-wide launch plan hash differs")
    intent = _read_json(intent_preregistration)
    source = _read_json(source_launch_plan)
    if (
        intent.get("schema")
        != "zuma-rl.alphazuma-55-polar-intent-wide-distillation-preregistration"
        or intent.get("status") != "FROZEN_BEFORE_TRAINING"
    ):
        raise ValueError("unexpected intent-wide preregistration")
    if (
        source.get("schema") != "zuma-rl.alphazuma-55-polar-wide-launch-plan"
        or source.get("status") != "FROZEN_BEFORE_WAIT"
    ):
        raise ValueError("unexpected effective-wide launch plan")
    target_run = Path(str(intent["run"]["run_dir"])).resolve()
    source_run = Path(str(source["target"]["run_dir"])).resolve()
    status_root = status_root.resolve()
    for path in (target_run, status_root):
        if path.exists():
            raise FileExistsError(f"intent-wide delayed output exists: {path}")
    root = SCRIPT_PATH.parents[1]
    watcher = root / "tools/watch_alphazuma_55_polar_intent_wide.py"
    return {
        "schema": "zuma-rl.alphazuma-55-polar-intent-wide-launch-plan",
        "version": 1,
        "status": "FROZEN_BEFORE_WAIT",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "intent_preregistration": _ref(intent_preregistration),
        "source_effective_wide_launch_plan": _ref(source_launch_plan),
        "source_terminal": {
            "completion": str(source_run / "completion.json"),
            "failure": str(source_run / "failure.json"),
            "watcher_pid_at_registration": int(source_watcher_pid),
            "launch_after_either_terminal_receipt_and_watcher_exit": True,
        },
        "target": {
            "run_dir": str(target_run),
            "completion": str(target_run / "completion.json"),
            "failure": str(target_run / "failure.json"),
        },
        "original_root": str(original_root),
        "outputs": {"status_root": str(status_root)},
        "implementation": {
            "builder": _ref(SCRIPT_PATH),
            "watcher": _ref(watcher),
            "trainer": intent["trainer"],
        },
        "authority_boundary": {
            "formal_seed_consumption": False,
            "current_campaign_candidate_authority": False,
            "s99081535_successor_candidate_authority": False,
            "power_restore_authority": False,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intent-preregistration", required=True, type=Path)
    parser.add_argument("--expected-intent-sha256", required=True)
    parser.add_argument("--source-launch-plan", required=True, type=Path)
    parser.add_argument("--expected-source-plan-sha256", required=True)
    parser.add_argument("--source-watcher-pid", required=True, type=int)
    parser.add_argument("--original-root", required=True, type=Path)
    parser.add_argument("--status-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite intent-wide launch plan: {output}")
    value = build(
        intent_preregistration=args.intent_preregistration.expanduser(),
        expected_intent_sha256=str(args.expected_intent_sha256),
        source_launch_plan=args.source_launch_plan.expanduser(),
        expected_source_plan_sha256=str(args.expected_source_plan_sha256),
        source_watcher_pid=args.source_watcher_pid,
        original_root=args.original_root.expanduser(),
        status_root=args.status_root.expanduser(),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {
                "status": value["status"],
                "output": {"path": str(output), "sha256": _sha256(output)},
                "source_terminal": value["source_terminal"],
                "target": value["target"],
                "formal_seed_consumption": False,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
