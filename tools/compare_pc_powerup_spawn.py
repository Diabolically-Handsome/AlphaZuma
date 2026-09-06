"""Diff one full-object PC power-up spawn against the calibrated simulator."""

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
    compare_powerup_spawn_with_simulator,
)
from zuma_rl.revenge_core import (
    PowerupSpawnCalibration,
    RevengeSimulator,
)


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
    parser.add_argument("--from-update", required=True, type=int)
    parser.add_argument("--to-update", required=True, type=int)
    parser.add_argument("--expected-ball-id", required=True, type=int)
    parser.add_argument("--expected-powerup-type", required=True, type=int)
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
    frames = load_memory_trajectory(trajectory)
    before_matches = [
        frame for frame in frames if frame.update == args.from_update
    ]
    after_matches = [
        frame for frame in frames if frame.update == args.to_update
    ]
    if len(before_matches) != 1 or len(after_matches) != 1:
        raise ValueError("requested transition is not uniquely covered")
    calibration = PowerupSpawnCalibration.jungle2_retail_v1()
    simulator = RevengeSimulator.from_installed(
        args.level,
        root=args.root.resolve(),
        hard=args.hard,
        curve_index=args.curve_index,
        seed=1,
        powerup_calibration=calibration,
    )
    report = compare_powerup_spawn_with_simulator(
        before_matches[0],
        after_matches[0],
        simulator,
        curve_index=args.curve_index,
        expected_ball_id=args.expected_ball_id,
        expected_powerup_type=args.expected_powerup_type,
    )
    report["trajectory"] = {
        "artifact": str(trajectory),
        "artifact_sha256": _sha256_path(trajectory),
        "start_update": frames[0].update,
        "end_update": frames[-1].update,
    }
    report["scenario"] = {
        "level_id": args.level,
        "hard": args.hard,
        "curve_index": args.curve_index,
        "calibration": {
            "chance_denominator": calibration.chance_denominator,
            "initial_delay_ticks": calibration.initial_delay_ticks,
            "spawn_delay_ticks": calibration.spawn_delay_ticks,
            "cooldown_ticks": calibration.cooldown_ticks,
            "unique_color": calibration.unique_color,
            "supported_types": list(calibration.supported_types),
            "provenance": calibration.provenance,
        },
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
        f"update={report['to_update']} "
        f"type={report['selected_powerup_type']} "
        f"color={report['selected_color_id']} "
        f"ball={report['selected_ball_id']}"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
