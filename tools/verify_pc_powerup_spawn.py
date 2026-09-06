"""Verify one retail power-up spawn against RNG state and call traces."""

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

from zuma_rl.pc_rng_trajectory import (
    load_rng_trajectory,
    verify_powerup_spawn_transition,
)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _weights(value: str) -> tuple[int, ...]:
    try:
        weights = tuple(int(part, 10) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "weights must be comma-separated base-10 integers"
        ) from error
    if len(weights) != 14 or any(weight < 0 for weight in weights):
        raise argparse.ArgumentTypeError(
            "weights must contain exactly 14 non-negative integers"
        )
    return weights


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--call-trace", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--to-update", required=True, type=int)
    parser.add_argument("--curve-index", type=int, default=0)
    parser.add_argument("--num-colors", required=True, type=int)
    parser.add_argument("--chance-denominator", required=True, type=int)
    parser.add_argument("--weights", required=True, type=_weights)
    parser.add_argument("--expected-powerup-type", type=int)
    parser.add_argument("--expected-ball-id", type=int)
    parser.add_argument("--maximum-calls-per-tick", type=int, default=64)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    trajectory = args.trajectory.resolve()
    call_trace = args.call_trace.resolve()
    output = args.output.resolve()
    if output.exists() or not output.parent.is_dir():
        raise FileExistsError(
            "output path must be a new file in an existing directory"
        )
    frames = load_rng_trajectory(trajectory)
    report = verify_powerup_spawn_transition(
        frames,
        call_trace,
        to_update=args.to_update,
        curve_index=args.curve_index,
        num_colors=args.num_colors,
        chance_denominator=args.chance_denominator,
        powerup_weights=args.weights,
        expected_powerup_type=args.expected_powerup_type,
        expected_ball_id=args.expected_ball_id,
        maximum_calls_per_tick=args.maximum_calls_per_tick,
    )
    report["trajectory"] = {
        "artifact": str(trajectory),
        "artifact_sha256": _sha256_path(trajectory),
        "start_update": frames[0].update,
        "end_update": frames[-1].update,
        "tick_count": len(frames),
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
        f"type={report['weighted_type_roll']['selected_powerup_type']} "
        f"color={report['eligible_color_roll']['selected_color']} "
        f"ball={report['eligible_ball_roll']['selected_ball_id']}"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
