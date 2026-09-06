"""Verify the retail empty-chain victory transition and input lock."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from zuma_rl.pc_clear_sequence import verify_clear_sequence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--mutation", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--level", default="Jungle2")
    parser.add_argument("--hard", action="store_true")
    parser.add_argument("--curve-index", type=int, default=0)
    parser.add_argument("--expected-initial-active-count", type=int)
    parser.add_argument("--expected-initial-pending-count", type=int, default=1)
    parser.add_argument("--expected-clear-update", type=int)
    parser.add_argument("--expected-formal-transition-update", type=int)
    parser.add_argument("--expected-delayed-award-count", type=int)
    parser.add_argument("--expected-award-points", type=int, default=100)
    parser.add_argument("--expected-award-period", type=int, default=5)
    parser.add_argument("--minimum-stable-tail-ticks", type=int, default=0)
    parser.add_argument("--click-trajectory", type=Path)
    parser.add_argument("--click-dmo", type=Path)
    parser.add_argument("--click-provenance", type=Path)
    parser.add_argument("--click-down-update", type=int)
    parser.add_argument("--click-up-update", type=int)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output.resolve()
    if output.exists() or not output.parent.is_dir():
        raise FileExistsError(
            "output path must be a new file in an existing directory"
        )
    report = verify_clear_sequence(
        args.trajectory.resolve(),
        args.mutation.resolve(),
        original_root=args.root.resolve(),
        click_trajectory_path=(
            None
            if args.click_trajectory is None
            else args.click_trajectory.resolve()
        ),
        click_dmo_path=(
            None if args.click_dmo is None else args.click_dmo.resolve()
        ),
        click_provenance_path=(
            None
            if args.click_provenance is None
            else args.click_provenance.resolve()
        ),
        click_down_update=args.click_down_update,
        click_up_update=args.click_up_update,
        curve_index=args.curve_index,
        expected_initial_active_count=args.expected_initial_active_count,
        expected_initial_pending_count=args.expected_initial_pending_count,
        expected_clear_update=args.expected_clear_update,
        expected_formal_transition_update=(
            args.expected_formal_transition_update
        ),
        expected_delayed_award_count=args.expected_delayed_award_count,
        expected_award_points=args.expected_award_points,
        expected_award_period=args.expected_award_period,
        minimum_stable_tail_ticks=args.minimum_stable_tail_ticks,
        level_id=args.level,
        hard=args.hard,
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
    transition = report["transition"]
    tail = report["diagnostic_score_tail"]
    print(
        f"status={report['status']} "
        f"empty={transition['empty_update']} "
        f"formal={transition['formal_transition_update']} "
        f"awards={tail['award_count']} "
        f"input_locked={report.get('input_lock', {}).get('accepted') is False}"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
