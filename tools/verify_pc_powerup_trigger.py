"""Verify one isolated retail power-up trigger trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from zuma_rl.pc_powerup_trigger import verify_powerup_trigger


def _ball_ids(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "ball IDs must be comma-separated integers"
        ) from exc
    if not result or len(result) != len(set(result)) or min(result) < 0:
        raise argparse.ArgumentTypeError("ball IDs must be unique and nonnegative")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--curve-index", type=int, default=0)
    parser.add_argument("--trigger-ball-id", required=True, type=int)
    parser.add_argument("--trigger-color-id", required=True, type=int)
    parser.add_argument("--powerup-type", required=True, type=int)
    parser.add_argument(
        "--expected-newly-exploding-ids",
        required=True,
        type=_ball_ids,
    )
    parser.add_argument("--expected-direct-match-ids", type=_ball_ids)
    parser.add_argument("--expected-score-delta", required=True, type=int)
    parser.add_argument("--movement-reference-ball-id", type=int)
    parser.add_argument("--expected-trigger-update", type=int)
    parser.add_argument("--expected-trigger-count", type=int, default=1)
    parser.add_argument("--reverse-ticks", type=int, default=300)
    parser.add_argument("--reverse-speed", type=float, default=1.0)
    parser.add_argument("--slow-ticks", type=int, default=800)
    parser.add_argument("--bomb-collision-pad", type=int, default=56)
    parser.add_argument(
        "--allow-unremoved-explosions",
        action="store_true",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output.resolve()
    if output.exists() or not output.parent.is_dir():
        raise FileExistsError(
            "output path must be a new file in an existing directory"
        )
    report = verify_powerup_trigger(
        args.trajectory.resolve(),
        curve_index=args.curve_index,
        trigger_ball_id=args.trigger_ball_id,
        trigger_color_id=args.trigger_color_id,
        powerup_type=args.powerup_type,
        expected_newly_exploding_ids=(
            args.expected_newly_exploding_ids
        ),
        expected_direct_match_ids=args.expected_direct_match_ids,
        expected_score_delta=args.expected_score_delta,
        movement_reference_ball_id=args.movement_reference_ball_id,
        expected_trigger_update=args.expected_trigger_update,
        expected_trigger_count=args.expected_trigger_count,
        reverse_ticks=args.reverse_ticks,
        reverse_speed=args.reverse_speed,
        slow_ticks=args.slow_ticks,
        bomb_collision_pad=args.bomb_collision_pad,
        require_explosion_removal=not args.allow_unremoved_explosions,
    )
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
    trigger = report["trigger"]
    print(
        f"status={report['status']} "
        f"trigger={trigger['update']} "
        f"score_delta={trigger['score_delta']} "
        "exploding="
        f"{','.join(map(str, trigger['newly_exploding_ball_ids']))}"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
