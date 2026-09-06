"""Freeze the specialist plus observation-teacher map for balanced distillation."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from zuma_rl.alphazuma_55 import INCLUDED_LEVELS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()
BASE_TEACHER_MAP = (
    PROJECT_ROOT / "diagnostics/alphazuma-55-teacher-map-s85081501-v1.json"
)
HYBRID_PREREGISTRATION = (
    PROJECT_ROOT
    / "diagnostics/alphazuma-55-hybrid-teacher-fresh-probe-s99081508-preregistration-v1.json"
)
HYBRID_RESULT = Path(
    "/mnt/d/ZumaTraining/alphazuma-55-weekend-s81081401-v1/engineering/"
    "hybrid-teacher-fresh-probe-s99081508-v1/result.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


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
        raise FileExistsError(f"refusing to overwrite frozen teacher map: {path}")
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


def build() -> dict[str, Any]:
    base = _read(BASE_TEACHER_MAP.resolve(strict=True))
    hybrid_prereg = _read(HYBRID_PREREGISTRATION.resolve(strict=True))
    hybrid_result = _read(HYBRID_RESULT.resolve(strict=True))
    if base.get("status") != "FROZEN":
        raise ValueError("base teacher map is not frozen")
    if [row["level_id"] for row in base["levels"]] != list(INCLUDED_LEVELS):
        raise ValueError("base teacher map differs from AlphaZuma 55")
    if hybrid_prereg.get("status") != "FROZEN_BEFORE_FRESH_ENGINEERING_PROBE":
        raise ValueError("hybrid teacher preregistration is not frozen")
    if hybrid_result.get("status") != "COMPLETE":
        raise ValueError("hybrid teacher result is not complete")
    if hybrid_result["preregistration"]["sha256"] != _sha256(
        HYBRID_PREREGISTRATION
    ):
        raise ValueError("hybrid teacher result binds another preregistration")
    policy_rows = hybrid_prereg.get("teacher_policy_map")
    if not isinstance(policy_rows, list) or [
        row["level_id"] for row in policy_rows
    ] != list(INCLUDED_LEVELS):
        raise ValueError("hybrid teacher policy map differs from AlphaZuma 55")
    policy_by_level = {str(row["level_id"]): row for row in policy_rows}
    levels: list[dict[str, Any]] = []
    for source_row in base["levels"]:
        row = deepcopy(source_row)
        policy = policy_by_level[str(row["level_id"])]
        row["observation_teacher_policy_id"] = str(
            policy["teacher_policy_id"]
        )
        row["observation_teacher_selection_reason"] = str(
            policy["selection_reason"]
        )
        levels.append(row)
    return {
        "schema": "zuma-rl.alphazuma-55-balanced-teacher-map",
        "version": 1,
        "status": "FROZEN",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_id": base["campaign_id"],
        "objective": (
            "Retain proven specialist labels while routing unproven and relocation "
            "labels through the frozen geometric/strategic observation-teacher map."
        ),
        "builder": _artifact(SCRIPT_PATH),
        "base_specialist_map": _artifact(BASE_TEACHER_MAP),
        "hybrid_teacher_evidence": {
            "preregistration": _artifact(HYBRID_PREREGISTRATION),
            "result": _artifact(HYBRID_RESULT),
            "result_summary": hybrid_result["summary"],
            "formal_evidence": False,
            "selection_conditioned_engineering": True,
        },
        "levels": levels,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    value = build()
    _write_exclusive(output, value)
    counts: dict[str, int] = {}
    for row in value["levels"]:
        policy_id = str(row["observation_teacher_policy_id"])
        counts[policy_id] = counts.get(policy_id, 0) + 1
    print(
        json.dumps(
            {
                "status": "FROZEN",
                "output": str(output),
                "sha256": _sha256(output),
                "levels": len(value["levels"]),
                "observation_teacher_counts": counts,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
