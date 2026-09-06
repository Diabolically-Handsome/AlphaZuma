"""Diff any captured power-up state-machine interval against the simulator."""

from __future__ import annotations

import argparse
import hashlib
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
)
from zuma_rl.revenge_core import RevengeSimulator


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--level", default="Jungle2")
    parser.add_argument("--hard", action="store_true")
    parser.add_argument("--curve-index", type=int, default=0)
    parser.add_argument("--start-update", required=True, type=int)
    parser.add_argument("--end-update", required=True, type=int)
    parser.add_argument("--ball-id", required=True, type=int)
    parser.add_argument("--color-id", required=True, type=int)
    parser.add_argument("--powerup-type", required=True, type=int)
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
    all_frames = load_memory_trajectory(trajectory)
    frames = tuple(
        frame
        for frame in all_frames
        if args.start_update <= frame.update <= args.end_update
    )
    if (
        not frames
        or frames[0].update != args.start_update
        or frames[-1].update != args.end_update
        or len(frames) != args.end_update - args.start_update + 1
    ):
        raise ValueError(
            "requested update interval is not fully covered"
        )
    simulator = RevengeSimulator.from_installed(
        args.level,
        root=args.root.resolve(),
        hard=args.hard,
        curve_index=args.curve_index,
        seed=1,
    )
    report = compare_powerup_lifecycle_with_simulator(
        frames,
        simulator,
        ball_id=args.ball_id,
        curve_index=args.curve_index,
        color_id=args.color_id,
        powerup_type=args.powerup_type,
    )
    report["trajectory"] = {
        "artifact": str(trajectory),
        "artifact_sha256": _sha256_path(trajectory),
        "captured_start_update": all_frames[0].update,
        "captured_end_update": all_frames[-1].update,
        "compared_start_update": frames[0].update,
        "compared_end_update": frames[-1].update,
    }
    report["scenario"] = {
        "level_id": args.level,
        "hard": args.hard,
        "curve_index": args.curve_index,
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
