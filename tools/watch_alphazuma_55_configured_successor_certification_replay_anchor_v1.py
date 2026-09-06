"""Watch the replay-anchor successor and invoke its independent auditor."""

from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools import (
    run_alphazuma_55_configured_successor_certification_replay_anchor_v1 as controller,
)
from tools import watch_alphazuma_55_configured_successor_certification as base


SCRIPT_PATH = Path(__file__).resolve()


def run(
    master_path: Path,
    expected_master_sha256: str,
    controller_pid: int,
    poll_seconds: float,
) -> int:
    original_controller = base.controller
    original_script = base.SCRIPT_PATH
    try:
        base.controller = controller
        base.SCRIPT_PATH = SCRIPT_PATH
        return base.run(
            master_path,
            expected_master_sha256,
            controller_pid,
            poll_seconds,
        )
    finally:
        base.controller = original_controller
        base.SCRIPT_PATH = original_script


def main(argv: list[str] | None = None) -> int:
    args = base.build_parser().parse_args(argv)
    return run(
        args.master_preregistration.expanduser(),
        str(args.expected_master_sha256),
        args.controller_pid,
        args.poll_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
