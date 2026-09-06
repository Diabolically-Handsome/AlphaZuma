"""Validate a compact retail RNG trajectory and emit exact MT transitions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from zuma_rl.pc_rng_trajectory import (
    analyze_rng_trajectory,
    load_rng_trajectory,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--maximum-calls-per-tick", type=int, default=64)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.maximum_calls_per_tick < 1:
        raise SystemExit("--maximum-calls-per-tick must be positive")
    output = args.output.resolve()
    if output.exists() or not output.parent.is_dir():
        raise FileExistsError("output path must be a new file in an existing directory")
    frames = load_rng_trajectory(args.index.resolve())
    report = analyze_rng_trajectory(
        frames,
        maximum_calls_per_tick=args.maximum_calls_per_tick,
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
        f"status={report['status']} ticks={report['tick_count']} "
        f"transitions={report['transition_count']} "
        f"mtrand_calls={report['total_mtrand_calls']} "
        f"non_two={report['non_two_call_transition_count']}"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
