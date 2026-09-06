"""Watch and finalize the eight-route AlphaZuma 55 goal."""

from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools import audit_alphazuma_55_single_policy_goal_v5 as goal
from tools import watch_alphazuma_55_single_policy_goal as base


SCRIPT_PATH = Path(__file__).resolve()


def run(finalizer_path: Path, expected_sha256: str, poll_seconds: float) -> int:
    original_goal = base.goal
    original_script = base.SCRIPT_PATH
    try:
        base.goal = goal
        base.SCRIPT_PATH = SCRIPT_PATH
        return base.run(finalizer_path, expected_sha256, poll_seconds)
    finally:
        base.goal = original_goal
        base.SCRIPT_PATH = original_script


def main(argv: list[str] | None = None) -> int:
    args = base.build_parser().parse_args(argv)
    return run(args.finalizer, str(args.expected_finalizer_sha256), args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
