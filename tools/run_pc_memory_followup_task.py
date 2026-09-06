"""Show or execute one verified post-PC-Golden memory follow-up task.

``show`` is headless and read-only.  ``collect`` launches one visible original
replay, freezes it, and records a schema-v2 full Board trajectory.  Collection
must only be used after the source session's two PC Golden replays completed
and the user yielded the desktop.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

from zuma_rl.pc_memory_followup import verify_pc_memory_followup_plan


def _task_row(
    plan: Path,
    *,
    evidence_root: Path,
    task_id: str,
) -> Mapping[str, Any]:
    report = verify_pc_memory_followup_plan(
        plan,
        evidence_root=evidence_root,
    )
    if report.status != "PASS":
        raise ValueError(
            "memory follow-up plan failed verification: "
            + "; ".join(report.reasons)
        )
    matches = [row for row in report.tasks if row.get("id") == task_id]
    if len(matches) != 1:
        raise ValueError("memory follow-up task ID is missing or duplicated")
    return matches[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("show", "collect"):
        command = subparsers.add_parser(name)
        command.add_argument("plan", type=Path)
        command.add_argument("--evidence-root", required=True, type=Path)
        command.add_argument("--task", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    row = _task_row(
        args.plan,
        evidence_root=args.evidence_root,
        task_id=args.task,
    )
    if args.command == "show":
        print(
            json.dumps(
                dict(row),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                indent=2,
            )
        )
        return 0

    if row.get("status") != "ready_for_memory_collection":
        raise SystemExit(
            "memory collection requires a completed PC Golden source session; "
            f"current status is {row.get('status')}"
        )
    if os.name != "nt":
        raise SystemExit("memory collection requires native Windows")
    collector_arguments = row.get("collector_arguments")
    if (
        not isinstance(collector_arguments, list)
        or not collector_arguments
        or any(not isinstance(item, str) for item in collector_arguments)
    ):
        raise SystemExit("verified collector arguments are missing")
    from tools import collect_pc_memory_probe

    return collect_pc_memory_probe.main(collector_arguments)


if __name__ == "__main__":
    raise SystemExit(main())
