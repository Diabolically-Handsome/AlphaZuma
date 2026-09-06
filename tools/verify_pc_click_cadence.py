"""Verify retail continuous-click acceptance and firing cadence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from zuma_rl.pc_input_cadence import verify_click_cadence


def _updates(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "updates must be comma-separated integers"
        ) from exc
    if (
        not result
        or min(result) < 0
        or tuple(sorted(set(result))) != result
    ):
        raise argparse.ArgumentTypeError(
            "updates must be sorted, unique, and nonnegative"
        )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--dmo", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--expected-accepted-down-updates",
        required=True,
        type=_updates,
    )
    parser.add_argument("--release-delay-ticks", type=int, default=6)
    parser.add_argument("--expected-cycle-ticks", type=int, default=21)
    parser.add_argument(
        "--expected-free-projectile-ticks",
        type=int,
        default=10,
    )
    parser.add_argument("--allow-score-change", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output.resolve()
    if output.exists() or not output.parent.is_dir():
        raise FileExistsError(
            "output path must be a new file in an existing directory"
        )
    report = verify_click_cadence(
        args.trajectory.resolve(),
        args.dmo.resolve(),
        expected_accepted_down_updates=(
            args.expected_accepted_down_updates
        ),
        release_delay_ticks=args.release_delay_ticks,
        expected_cycle_ticks=args.expected_cycle_ticks,
        expected_free_projectile_ticks=(
            args.expected_free_projectile_ticks
        ),
        require_stable_score=not args.allow_score_change,
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
    cadence = report["cadence"]
    input_state = report["input"]
    print(
        f"status={report['status']} "
        f"clicks={input_state['down_count']} "
        f"accepted={input_state['accepted_count']} "
        f"rejected={input_state['rejected_count']} "
        f"cycle={cadence['expected_cycle_ticks']} "
        f"release_delay={cadence['release_delay_ticks']}"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
