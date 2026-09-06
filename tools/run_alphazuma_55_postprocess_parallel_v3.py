"""Capacity-safe AlphaZuma 55 postprocess controller.

This is a narrow versioned wrapper around the frozen V2 controller.  Candidate
freezing, rankings, seed matrices, gates, and reporting remain unchanged.  The
only execution change is binding the capacity-safe multilevel evaluator so an
actor observation overflow fails one policy/seed attempt instead of killing a
whole vector shard.
"""

from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tools import run_alphazuma_55_postprocess_parallel_v2 as legacy


EVALUATOR = Path(__file__).resolve().with_name(
    "evaluate_zero_shot_multilevel_v2.py"
)
SCRIPT_PATH = Path(__file__).resolve()


def main(argv: list[str] | None = None) -> int:
    legacy.SCRIPT_PATH = SCRIPT_PATH
    legacy.EVALUATOR = EVALUATOR
    # _evaluate_and_summarize was imported from the mature helper module; its
    # own global evaluator path must point to the same versioned executable.
    legacy._evaluate_and_summarize.__globals__["EVALUATOR"] = EVALUATOR
    return legacy.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
