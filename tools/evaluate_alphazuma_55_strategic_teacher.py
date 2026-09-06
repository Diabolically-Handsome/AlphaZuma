"""Evaluate the strategic observation-only teacher with the frozen probe runner."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tools import evaluate_alphazuma_55_geometric_teacher as legacy
from zuma_rl.revenge_strategic_teacher import (
    StrategicActorObservableRevengeTeacher,
)


SCRIPT_PATH = Path(__file__).resolve()
_BASE_RUN_TASK = legacy._run_task


def _run_task(payload: dict[str, Any]) -> dict[str, Any]:
    legacy.ActorObservableRevengeTeacher = StrategicActorObservableRevengeTeacher
    row = _BASE_RUN_TASK(payload)
    row["teacher_policy_id"] = "strategic-actor-observable-v1"
    return row


def main(argv: list[str] | None = None) -> int:
    legacy.SCRIPT_PATH = SCRIPT_PATH
    legacy._run_task = _run_task
    return legacy.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
