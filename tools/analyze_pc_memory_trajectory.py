"""Validate and summarize one frozen retail memory trajectory."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from zuma_rl.pc_memory_trajectory import (
    PcMemoryTrajectoryError,
    analyze_memory_trajectory,
    canonical_analysis_bytes,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        report = analyze_memory_trajectory(arguments.index.resolve())
        payload = canonical_analysis_bytes(report)
        if arguments.output is not None:
            output = arguments.output.resolve()
            if output.exists() or not output.parent.is_dir():
                raise PcMemoryTrajectoryError(
                    "trajectory_analysis_output_invalid"
                )
            output.write_bytes(payload)
        sys.stdout.buffer.write(payload)
        return 0
    except PcMemoryTrajectoryError as error:
        print(f"trajectory analysis error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
