"""Freeze the engineering-only AlphaZuma 55 long-horizon probe.

The probe reuses no formal selection, final-blind, or continuous-campaign
seeds.  It asks a narrow question: can the frozen V1 frontier checkpoint clear
training levels that were dominated by 12,000-tick truncations when allowed a
30,000-tick engineering horizon?
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MASTER = PROJECT_ROOT / "diagnostics/alphazuma-55-weekend-s81081401-preregistration-v1.json"
ANALYSIS = PROJECT_ROOT / "diagnostics/alphazuma-55-hard-frontier-analysis-s95081501-v1.json"
TEMPLATE = PROJECT_ROOT / "diagnostics/alphazuma-55-engineering-baseline-s82081402-preregistration-v1.json"
EVALUATOR = PROJECT_ROOT / "tools/evaluate_zero_shot_multilevel.py"
SOURCE_MODEL = Path(
    "/mnt/d/ZumaTraining/alphazuma-55-v1-frontier-s87081501-v1/checkpoints/"
    "alphazuma55-v1-frontier-5080_4194304_steps.zip"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "diagnostics/alphazuma-55-horizon-probe-s99081501-preregistration-v1.json"
DEFAULT_RUN_ROOT = Path(
    "/mnt/d/ZumaTraining/alphazuma-55-weekend-s81081401-v1/engineering/"
    "horizon-probe-s99081501-v1"
)

EXPECTED_HASHES = {
    MASTER: "sha256:f43e10c299332b3e4b3497ff057596775c1bf3011e1ce08fc23f832a6652f4cb",
    ANALYSIS: "sha256:49ae136c38458fe0f452975257085ccdaee765a7e35cbcf517f7a2cc6766572e",
    TEMPLATE: "sha256:c6c00d29dfebeafa347aa7995608d53c73b5b8d29eba7d9a056ff72568b9c6f8",
    EVALUATOR: "sha256:d0011f9ad5c6837315007a7fb31e128fb315e4baddb45959c5f3b98d11a6a7e3",
    SOURCE_MODEL: "sha256:4324ccf39b822ad946359e34c73a1afe62c7f00d73d4a2d1375a5b9e64a955bc",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected an object in {path}")
    return value


def _write_new(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _verify_inputs() -> None:
    for path, expected in EXPECTED_HASHES.items():
        actual = _sha256(path.resolve(strict=True))
        if actual != expected:
            raise RuntimeError(f"hash mismatch for {path}: {actual} != {expected}")


def _selected_level_ids(analysis: dict[str, Any]) -> set[str]:
    selected: set[str] = set()
    for row in analysis["per_level"]:
        episodes = int(row["episodes"])
        truncations = int(row["truncations"])
        if (
            int(row["wins"]) == 0
            and episodes >= 10
            and truncations / episodes >= 0.75
        ):
            selected.add(str(row["level_id"]))
    if len(selected) != 17:
        raise RuntimeError(f"expected 17 truncation-dominant levels, got {len(selected)}")
    return selected


def build_contract(
    *,
    max_ticks: int,
    seed_base: int,
    run_root: Path,
    device: str,
) -> dict[str, Any]:
    if max_ticks <= 12_000:
        raise ValueError("probe horizon must exceed the frozen training horizon")
    if not 1_400_000_000 <= seed_base <= 1_400_999_999:
        raise ValueError("seed base must remain in engineering_and_calibration")

    master = _load(MASTER)
    analysis = _load(ANALYSIS)
    template = _load(TEMPLATE)
    selected = _selected_level_ids(analysis)
    master_level_ids = [
        str(level_id)
        for level_id in master["scope"]["included_levels_in_adventure_order"]
    ]
    template_by_id = {
        str(level["id"]): level for level in template["levels"]
    }
    if set(template_by_id) != set(master_level_ids):
        raise RuntimeError("template levels differ from the master 55-level scope")
    levels = [
        copy.deepcopy(template_by_id[level_id])
        for level_id in master_level_ids
        if level_id in selected
    ]
    if {str(level["id"]) for level in levels} != selected:
        raise RuntimeError("selected levels are not an exact subset of the master")

    environment = copy.deepcopy(template["environment"])
    environment["base_config"]["max_ticks"] = int(max_ticks)
    last_seed = seed_base + len(levels) - 1
    if last_seed > 1_400_999_999:
        raise ValueError("probe seed range exceeds engineering registry")

    return {
        "schema": "zuma-rl.zero-shot-multilevel-preregistration",
        "version": 1,
        "status": "FROZEN_BEFORE_EVALUATION",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_id": str(master["campaign_id"]),
        "stage": "engineering_horizon_probe",
        "question": (
            "Does the frozen V1 frontier checkpoint clear any level whose live "
            "training outcomes were dominated by 12,000-tick truncations when "
            f"allowed {max_ticks:,} ticks?"
        ),
        "classification": "engineering_only_not_formal_selection_evidence",
        "master_preregistration": {
            "path": str(MASTER),
            "sha256": EXPECTED_HASHES[MASTER],
        },
        "source_analysis": {
            "path": str(ANALYSIS),
            "sha256": EXPECTED_HASHES[ANALYSIS],
            "selection_rule": (
                "wins == 0 and episodes >= 10 and truncations / episodes >= 0.75"
            ),
        },
        "model": {
            "id": "v1-frontier-4194304",
            "training_steps": 4_194_304,
            "path": str(SOURCE_MODEL),
            "sha256": EXPECTED_HASHES[SOURCE_MODEL],
        },
        "levels": levels,
        "attempts_per_level": 1,
        "total_attempts": len(levels),
        "seed_plan": {
            "registry": "engineering_and_calibration",
            "base_seed": seed_base,
            "last_seed": last_seed,
            "formal_seed_consumption": False,
        },
        "environment": environment,
        "execution": {
            "output_root": str(run_root),
            "shard_count": 1,
            "devices": [device],
            "parallel_envs_per_shard": len(levels),
        },
        "interpretation": {
            "positive": (
                "A win after tick 12000 supports testing a separately frozen "
                "long-horizon training route; it does not promote this model."
            ),
            "negative": (
                "No wins by the probe ceiling does not prove longer horizons are "
                "useless; it rejects only this frozen checkpoint and seed matrix."
            ),
            "formal_authority": False,
        },
        "evaluator": {
            "path": str(EVALUATOR),
            "sha256": EXPECTED_HASHES[EVALUATOR],
        },
        "builder": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--max-ticks", type=int, default=30_000)
    parser.add_argument("--seed-base", type=int, default=1_400_400_000)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()

    _verify_inputs()
    output = args.output.expanduser().resolve()
    run_root = args.run_root.expanduser().resolve()
    if run_root.exists():
        raise FileExistsError(f"probe output root already exists: {run_root}")
    contract = build_contract(
        max_ticks=args.max_ticks,
        seed_base=args.seed_base,
        run_root=run_root,
        device=str(args.device),
    )
    _write_new(output, contract)
    print(
        json.dumps(
            {
                "status": "FROZEN",
                "path": str(output),
                "sha256": _sha256(output),
                "levels": len(contract["levels"]),
                "max_ticks": args.max_ticks,
                "seed_range": [
                    contract["seed_plan"]["base_seed"],
                    contract["seed_plan"]["last_seed"],
                ],
                "output_root_absent": not run_root.exists(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
