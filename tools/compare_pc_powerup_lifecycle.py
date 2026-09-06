"""Diff the simulator power-up state machine against every captured PC tick."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from zuma_rl.pc_memory_trajectory import load_memory_trajectory
from zuma_rl.pc_powerup_lifecycle import (
    compare_powerup_lifecycle_with_simulator,
    verify_powerup_lifecycle,
)
from zuma_rl.revenge_core import RevengeSimulator


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--level", default="Jungle2")
    parser.add_argument("--hard", action="store_true")
    parser.add_argument("--curve-index", type=int, default=0)
    parser.add_argument("--ball-id", required=True, type=int)
    parser.add_argument("--color-id", required=True, type=int)
    parser.add_argument("--powerup-type", required=True, type=int)
    parser.add_argument("--expected-expiration-update", type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    trajectory = args.trajectory.resolve()
    output = args.output.resolve()
    if output.exists() or not output.parent.is_dir():
        raise FileExistsError(
            "output path must be a new file in an existing directory"
        )
    lifecycle = verify_powerup_lifecycle(
        trajectory,
        ball_id=args.ball_id,
        curve_index=args.curve_index,
        color_id=args.color_id,
        powerup_type=args.powerup_type,
        expected_expiration_update=args.expected_expiration_update,
    )
    simulator = RevengeSimulator.from_installed(
        args.level,
        root=args.root.resolve(),
        hard=args.hard,
        curve_index=args.curve_index,
        seed=1,
    )
    report = compare_powerup_lifecycle_with_simulator(
        load_memory_trajectory(trajectory),
        simulator,
        ball_id=args.ball_id,
        curve_index=args.curve_index,
        color_id=args.color_id,
        powerup_type=args.powerup_type,
    )
    report["trajectory"] = lifecycle["trajectory"]
    report["scenario"] = {
        "level_id": args.level,
        "hard": args.hard,
        "curve_index": args.curve_index,
    }
    report["lifecycle"] = {
        "expiration_update": lifecycle["expiration"]["update"],
        "primary_clear_update": lifecycle["primary_clear_update"],
        "previous_marker_clear_update": (
            lifecycle["previous_marker_clear_update"]
        ),
    }
    output.write_bytes(
        (
            json.dumps(
                report,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("ascii")
    )
    print(
        f"status={report['status']} "
        f"transitions={report['transition_count']} "
        f"updates={report['start_update']}..{report['end_update']}"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
