"""Freeze a paired strategic-teacher probe on consumed engineering seeds."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tools import build_alphazuma_55_geometric_teacher_probe as legacy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()
EVALUATOR = PROJECT_ROOT / "tools/evaluate_alphazuma_55_strategic_teacher.py"
TEACHER = PROJECT_ROOT / "src/zuma_rl/revenge_strategic_teacher.py"
PAIRED_GEOMETRIC_RESULT = Path(
    "/mnt/d/ZumaTraining/alphazuma-55-weekend-s81081401-v1/engineering/"
    "geometric-teacher-probe-s99081506-v1/result.json"
)


def build(*, output_root: Path, workers: int) -> dict[str, Any]:
    value = legacy.build(output_root=output_root, workers=workers)
    value["created_utc"] = datetime.now(timezone.utc).isoformat()
    value["objective"] = (
        "Paired screen of an actor-observable strategic ranking teacher using "
        "the already-consumed geometric-probe seeds; no fresh evidence claim."
    )
    value["implementation"]["builder"] = legacy._artifact(SCRIPT_PATH)
    value["implementation"]["evaluator"] = legacy._artifact(EVALUATOR)
    value["implementation"]["teacher"] = legacy._artifact(TEACHER)
    value["teacher_policy_id"] = "strategic-actor-observable-v1"
    value["paired_seed_reuse"] = {
        "fresh_evidence": False,
        "reason": "paired policy engineering on the frozen geometric probe matrix",
        "geometric_result": legacy._artifact(PAIRED_GEOMETRIC_RESULT),
    }
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"strategic teacher output root exists: {output_root}")
    value = build(output_root=output_root, workers=args.workers)
    legacy._write_exclusive(output, value)
    print(
        json.dumps(
            {
                "status": "FROZEN_PAIRED_ENGINEERING",
                "output": str(output),
                "sha256": legacy._sha256(output),
                "levels": len(value["levels"]),
                "seed_range": value["seed_range"],
                "workers": value["execution"]["workers"],
                "fresh_evidence": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
