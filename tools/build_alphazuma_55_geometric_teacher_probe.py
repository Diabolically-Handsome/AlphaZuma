"""Freeze the engineering-only 55-level geometric-teacher probe."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()
MASTER = PROJECT_ROOT / "diagnostics/alphazuma-55-weekend-s81081401-preregistration-v1.json"
SOURCE = PROJECT_ROOT / "diagnostics/alphazuma-55-engineering-baseline-s82081402-preregistration-v1.json"
EVALUATOR = PROJECT_ROOT / "tools/evaluate_alphazuma_55_geometric_teacher.py"
TEACHER = PROJECT_ROOT / "src/zuma_rl/revenge_teacher.py"
CAPACITY_WRAPPER = PROJECT_ROOT / "tools/evaluate_zero_shot_multilevel_v2.py"
SEED_FIRST = 1_400_500_000


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(path: Path) -> dict[str, str]:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "sha256": _sha256(resolved)}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen probe: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build(*, output_root: Path, workers: int) -> dict[str, Any]:
    if workers < 1 or workers > 24:
        raise ValueError("workers must be in [1, 24]")
    master = _read(MASTER.resolve(strict=True))
    source = _read(SOURCE.resolve(strict=True))
    levels = source.get("levels")
    if not isinstance(levels, list) or len(levels) != 55:
        raise ValueError("source evaluation contract does not contain 55 levels")
    engineering = master["seed_registry"]["engineering_and_calibration"]
    seed_last = SEED_FIRST + len(levels) - 1
    if not (int(engineering["first"]) <= SEED_FIRST <= seed_last <= int(engineering["last"])):
        raise ValueError("teacher probe seeds escaped engineering namespace")
    result_path = output_root / "result.json"
    return {
        "schema": "zuma-rl.alphazuma-55-geometric-teacher-probe-preregistration",
        "version": 1,
        "status": "FROZEN_BEFORE_ENGINEERING_PROBE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_id": master["campaign_id"],
        "objective": (
            "Screen the actor-observable geometric teacher for winning hard-level "
            "demonstrations without consuming selection, final-blind, or continuous seeds."
        ),
        "master_preregistration": _artifact(MASTER),
        "source_evaluation_contract": _artifact(SOURCE),
        "levels": levels,
        "tasks": [
            {"level_id": level["id"], "seed": SEED_FIRST + index}
            for index, level in enumerate(levels)
        ],
        "seed_range": [SEED_FIRST, seed_last],
        "execution": {
            "device": "cpu",
            "workers": workers,
            "attempts_per_level": 1,
            "human_input_wrapper": "elite-human-v1",
            "capacity_overflow": "fail_closed_affected_attempt_only",
        },
        "implementation": {
            "builder": _artifact(SCRIPT_PATH),
            "evaluator": _artifact(EVALUATOR),
            "teacher": _artifact(TEACHER),
            "capacity_wrapper": _artifact(CAPACITY_WRAPPER),
        },
        "outputs": {
            "root": str(output_root.resolve()),
            "result": str(result_path.resolve()),
        },
        "authority_boundary": {
            "formal_seed_consumption": False,
            "training_recipe_change_authorized": False,
            "formal_candidate_registration_authorized": False,
            "winning_rows_may_support_a_separately_preregistered_demonstration_route": True,
        },
    }


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
        raise FileExistsError(f"teacher probe output root exists: {output_root}")
    value = build(output_root=output_root, workers=args.workers)
    _write_exclusive(output, value)
    print(
        json.dumps(
            {
                "status": "FROZEN",
                "output": str(output),
                "sha256": _sha256(output),
                "levels": len(value["levels"]),
                "seed_range": value["seed_range"],
                "workers": value["execution"]["workers"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
