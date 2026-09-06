"""Seven-route capacity-safe AlphaZuma 55 postprocess controller.

V4 changes exactly one structural bound from the frozen V3 controller: the
candidate inventory may contain seven training routes instead of six.  Seed
matrices, rankings, gates, evaluator semantics, and all artifact audits remain
unchanged.
"""

from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tools import run_alphazuma_55_postprocess_parallel_v2 as legacy


EVALUATOR = Path(__file__).resolve().with_name("evaluate_zero_shot_multilevel_v2.py")
SCRIPT_PATH = Path(__file__).resolve()
_ORIGINAL_REQUIRE = legacy._require


def _seven_route_require(condition: bool, message: str) -> None:
    if not condition and message == "route count is outside [1, 6]":
        return
    _ORIGINAL_REQUIRE(condition, message)


def main(argv: list[str] | None = None) -> int:
    preview_args = legacy.build_parser().parse_args(argv)
    preview = legacy._read_json(
        preview_args.preregistration.expanduser().resolve(strict=True)
    )
    _ORIGINAL_REQUIRE(
        isinstance(preview.get("training_routes"), list)
        and len(preview["training_routes"]) == 7,
        "V4 requires exactly seven training routes",
    )
    original_script = legacy.SCRIPT_PATH
    original_evaluator = legacy.EVALUATOR
    original_require = legacy._require
    helper_globals = legacy._evaluate_and_summarize.__globals__
    original_helper_evaluator = helper_globals["EVALUATOR"]
    try:
        legacy.SCRIPT_PATH = SCRIPT_PATH
        legacy.EVALUATOR = EVALUATOR
        legacy._require = _seven_route_require
        helper_globals["EVALUATOR"] = EVALUATOR
        return legacy.main(argv)
    finally:
        legacy.SCRIPT_PATH = original_script
        legacy.EVALUATOR = original_evaluator
        legacy._require = original_require
        helper_globals["EVALUATOR"] = original_helper_evaluator


if __name__ == "__main__":
    raise SystemExit(main())
