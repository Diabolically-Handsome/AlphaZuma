"""Append the gradual-motor successor to the formal seed registry."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tools import run_alphazuma_55_motor_observable_gradual_successor_v1 as controller


SCRIPT_PATH = Path(__file__).resolve()
EXPECTED_V5_SHA256 = (
    "sha256:2f6996cd46778455e1069936da7a72b041882265a2c533b0db7faeddc63c5680"
)
CAMPAIGN_ID = "motor-observable-gradual-successor-s99081631"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _reference(path: Path) -> dict[str, str]:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "sha256": controller._sha256(resolved)}


def _write_new(path: Path, value: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists():
        temporary.unlink()
        raise FileExistsError(f"formal seed registry already exists: {path}")
    os.replace(temporary, path)


def build(*, project_root: Path, master_path: Path, output: Path) -> dict[str, Any]:
    root = project_root.resolve(strict=True)
    predecessor = (
        root
        / "diagnostics/alphazuma-55-weekend-formal-seed-registry-"
        "s99081619-v5.json"
    )
    if controller._sha256(predecessor) != EXPECTED_V5_SHA256:
        raise ValueError("formal seed registry V5 bytes changed")
    master_path = master_path.resolve(strict=True)
    master = controller._read(master_path)
    if not (
        master.get("schema") == controller.MASTER_SCHEMA
        and master.get("status") == "FROZEN_BEFORE_ENGINEERING_RESULT"
    ):
        raise ValueError("unexpected gradual successor master")
    registry = copy.deepcopy(controller._read(predecessor))
    registry.update(
        {
            "created_utc": _utc_now(),
            "supersedes": _reference(predecessor),
            "implementation": {"builder": _reference(SCRIPT_PATH)},
        }
    )
    registry["campaigns"].append(
        {
            "id": CAMPAIGN_ID,
            "master": _reference(master_path),
            "ranges": [
                {
                    "stage": "final_blind",
                    "first": controller.FINAL_SEED_BASE,
                    "last": (
                        controller.FINAL_SEED_BASE
                        + controller.LEVEL_COUNT * controller.FINAL_ATTEMPTS
                        - 1
                    ),
                },
                {
                    "stage": "continuous_campaign_challenge",
                    "first": controller.CONTINUOUS_SEED_BASE,
                    "last": (
                        controller.CONTINUOUS_SEED_BASE
                        + controller.LEVEL_COUNT * controller.CONTINUOUS_CAMPAIGNS
                        - 1
                    ),
                },
            ],
        }
    )
    ranges: list[tuple[int, int, str]] = []
    for campaign in registry["campaigns"]:
        for row in campaign["ranges"]:
            first, last = int(row["first"]), int(row["last"])
            if not 0 <= first <= last <= 0xFFFFFFFF:
                raise ValueError("formal range escaped uint32")
            ranges.append((first, last, str(campaign["id"])))
    ranges.sort()
    for left, right in zip(ranges, ranges[1:], strict=False):
        if left[1] >= right[0]:
            raise ValueError(f"formal ranges overlap: {left} {right}")
    registry["global_checks"] = {
        **registry.get("global_checks", {}),
        "ranges_strictly_non_overlapping": True,
        "all_ranges_within_uint32": True,
        "all_policy_inference_requires_its_own_frozen_master": True,
        "negative_promotion_result_leaves_its_ranges_unconsumed": True,
    }
    _write_new(output, registry)
    return registry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--master", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    value = build(
        project_root=args.project_root,
        master_path=args.master,
        output=args.output,
    )
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
