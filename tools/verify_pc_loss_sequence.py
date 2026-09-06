"""Verify the complete retail skull-entry loss sequence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from zuma_rl.pc_loss_sequence import verify_loss_sequence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--dmo", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--curve-index", type=int, default=0)
    parser.add_argument("--trigger-ball-id", required=True, type=int)
    parser.add_argument("--decoded-curve-end", required=True, type=int)
    parser.add_argument("--expected-trigger-update", type=int)
    parser.add_argument("--expected-chain-empty-update", type=int)
    parser.add_argument("--expected-pending-count", type=int, default=1)
    parser.add_argument(
        "--expected-pretrigger-advance",
        type=float,
        default=0.125,
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output.resolve()
    if output.exists() or not output.parent.is_dir():
        raise FileExistsError(
            "output path must be a new file in an existing directory"
        )
    report = verify_loss_sequence(
        args.trajectory.resolve(),
        args.dmo.resolve(),
        curve_index=args.curve_index,
        trigger_ball_id=args.trigger_ball_id,
        decoded_curve_end=args.decoded_curve_end,
        expected_trigger_update=args.expected_trigger_update,
        expected_chain_empty_update=args.expected_chain_empty_update,
        expected_pending_count=args.expected_pending_count,
        expected_pretrigger_advance=args.expected_pretrigger_advance,
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
    suction = report["suction"]
    print(
        f"status={report['status']} "
        f"trigger={trigger['update']} "
        f"empty={suction['first_chain_empty_update']} "
        f"ticks={suction['ticks_from_trigger_to_empty']} "
        f"clicks_ignored={len(report['dmo']['ignored_left_down_updates'])}"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
