"""Extend the formal seed registry for the factorial successor."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
PREDECESSOR = PROJECT_ROOT / (
    "diagnostics/alphazuma-55-weekend-formal-seed-registry-s99081645-v9.json"
)
PREDECESSOR_SHA256 = (
    "sha256:39f52356d105cfe195f43d77ded9b8ffea8235fa939b97b890497fb551c5aa6d"
)
MASTER = PROJECT_ROOT / (
    "diagnostics/alphazuma-55-motor-observable-gradual-headonly-allaim-"
    "successor-s99081649-preregistration-v1.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / (
    "diagnostics/alphazuma-55-weekend-formal-seed-registry-s99081650-v10.json"
)
CAMPAIGN_ID = (
    "motor-observable-gradual-headonly-allaim-successor-s99081649-v1"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _reference(path: Path, expected: str | None = None) -> dict[str, str]:
    resolved = path.resolve(strict=True)
    actual = _sha256(resolved)
    if expected is not None and actual != expected:
        raise ValueError(f"frozen bytes differ: {resolved}")
    return {"path": str(resolved), "sha256": actual}


def _write_new(path: Path, value: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value, ensure_ascii=False, indent=2, allow_nan=False
    ) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _validate_ranges(campaigns: list[dict[str, Any]]) -> None:
    rows: list[tuple[int, int, str, str]] = []
    for campaign in campaigns:
        for row in campaign.get("ranges", []):
            first, last = int(row["first"]), int(row["last"])
            if not 0 <= first <= last <= 0xFFFFFFFF:
                raise ValueError("formal seed range escaped uint32")
            rows.append(
                (first, last, str(campaign["id"]), str(row["stage"]))
            )
    rows.sort()
    for left, right in zip(rows, rows[1:]):
        if left[1] >= right[0]:
            raise ValueError(f"formal seed ranges overlap: {left} {right}")


def build(*, expected_master_sha256: str, output: Path) -> dict[str, Any]:
    if output.resolve().exists():
        raise FileExistsError(f"formal seed registry already exists: {output}")
    predecessor_ref = _reference(PREDECESSOR, PREDECESSOR_SHA256)
    predecessor = _read(PREDECESSOR)
    master_ref = _reference(MASTER, expected_master_sha256)
    master = _read(MASTER)
    final = master["seed_registry"]["final_blind"]
    continuous = master["seed_registry"]["continuous_campaign_challenge"]
    if not (
        master.get("campaign_id")
        == "alphazuma-55-motor-observable-gradual-headonly-allaim-"
        "successor-s99081649-v1"
        and Path(
            master["formal_seed_registry"]["active_registry_path"]
        ).resolve()
        == output.resolve()
        and [int(final["first"]), int(final["last"])]
        == [4_100_000_000, 4_100_000_439]
        and [int(continuous["first"]), int(continuous["last"])]
        == [4_200_000_000, 4_200_000_219]
    ):
        raise ValueError("factorial successor seed contract differs")
    campaigns = list(predecessor.get("campaigns", []))
    if any(str(row.get("id")) == CAMPAIGN_ID for row in campaigns):
        raise ValueError("factorial successor already registered")
    campaigns.append(
        {
            "id": CAMPAIGN_ID,
            "master": master_ref,
            "ranges": [
                {
                    "stage": "final_blind",
                    "first": 4_100_000_000,
                    "last": 4_100_000_439,
                },
                {
                    "stage": "continuous_campaign_challenge",
                    "first": 4_200_000_000,
                    "last": 4_200_000_219,
                },
            ],
        }
    )
    _validate_ranges(campaigns)
    registry = {
        "schema": "zuma-rl.alphazuma-55-weekend-formal-seed-registry",
        "version": 1,
        "status": "FROZEN_BEFORE_ANY_SUCCESSOR_FORMAL_INFERENCE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "supersedes": predecessor_ref,
        "campaigns": campaigns,
        "global_checks": {
            "ranges_strictly_non_overlapping": True,
            "all_ranges_within_uint32": True,
            "all_policy_inference_requires_its_own_frozen_master": True,
            "negative_promotion_result_leaves_its_ranges_unconsumed": True,
        },
        "authority_boundary": {
            "this_registry_does_not_authorize_policy_inference": True,
            "this_registry_does_not_change_any_campaign_recipe": True,
            "power_restore_authority": False,
        },
        "implementation": {"builder": _reference(SCRIPT_PATH)},
    }
    _write_new(output, registry)
    return registry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-master-sha256", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    value = build(
        expected_master_sha256=str(args.expected_master_sha256),
        output=args.output,
    )
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
