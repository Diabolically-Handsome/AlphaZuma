"""Verify a retail power-up expiry and cleanup from a full trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from zuma_rl.pc_powerup_lifecycle import verify_powerup_lifecycle


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ball-id", required=True, type=int)
    parser.add_argument("--curve-index", type=int, default=0)
    parser.add_argument("--color-id", required=True, type=int)
    parser.add_argument("--powerup-type", required=True, type=int)
    parser.add_argument("--expected-expiration-update", type=int)
    parser.add_argument("--transition-ticks", type=int, default=100)
    parser.add_argument("--previous-retention-ticks", type=int, default=150)
    parser.add_argument("--visual-step", type=float, default=0.04)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output.resolve()
    if output.exists() or not output.parent.is_dir():
        raise FileExistsError(
            "output path must be a new file in an existing directory"
        )
    report = verify_powerup_lifecycle(
        args.trajectory.resolve(),
        ball_id=args.ball_id,
        curve_index=args.curve_index,
        color_id=args.color_id,
        powerup_type=args.powerup_type,
        expected_expiration_update=args.expected_expiration_update,
        transition_ticks=args.transition_ticks,
        previous_retention_ticks=args.previous_retention_ticks,
        visual_step=args.visual_step,
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
    print(
        f"status={report['status']} "
        f"expire={report['expiration']['update']} "
        f"primary_clear={report['primary_clear_update']} "
        "previous_clear="
        f"{report['previous_marker_clear_update']}"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
