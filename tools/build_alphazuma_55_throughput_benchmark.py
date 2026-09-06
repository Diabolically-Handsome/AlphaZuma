"""Freeze a non-formal AlphaZuma 55 evaluator throughput benchmark.

The benchmark deliberately reuses the engineering/calibration seed registry so
that selection, final-blind, and continuous-campaign seeds remain embargoed.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any


ENGINEERING_SEED_FIRST = 1_400_000_000
ENGINEERING_SEED_LAST = 1_400_999_999
FORMAL_SEED_FIRST = 1_600_000_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError:
        raise FileExistsError(f"refusing to overwrite frozen benchmark: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def build(
    *,
    source_path: Path,
    output_root: Path,
    attempts_per_level: int,
    base_seed: int,
    parallel_envs_per_shard: int,
) -> dict[str, Any]:
    source = _read_json(source_path)
    if (
        source.get("schema") != "zuma-rl.zero-shot-multilevel-preregistration"
        or source.get("version") != 1
        or source.get("status") != "FROZEN_BEFORE_EVALUATION"
        or source.get("stage") != "engineering"
    ):
        raise ValueError("source is not a frozen engineering evaluation contract")
    if attempts_per_level < 1 or parallel_envs_per_shard < 1:
        raise ValueError("attempt and parallel-environment counts must be positive")
    level_count = len(source.get("levels", []))
    if level_count != 55:
        raise ValueError("source contract does not contain exactly 55 levels")
    last_seed = base_seed + level_count * attempts_per_level - 1
    if not ENGINEERING_SEED_FIRST <= base_seed <= last_seed <= ENGINEERING_SEED_LAST:
        raise ValueError("benchmark seeds escape the engineering/calibration registry")
    if last_seed >= FORMAL_SEED_FIRST:
        raise ValueError("benchmark would consume a formal seed")
    evaluator = Path(str(source["evaluator"]["path"])).resolve(strict=True)
    if source["evaluator"]["sha256"] != _sha256(evaluator):
        raise ValueError("evaluator bytes differ from the source contract")
    model = Path(str(source["model"]["path"])).resolve(strict=True)
    if source["model"]["sha256"] != _sha256(model):
        raise ValueError("model bytes differ from the source contract")
    master = Path(str(source["master_preregistration"]["path"])).resolve(strict=True)
    if source["master_preregistration"]["sha256"] != _sha256(master):
        raise ValueError("master preregistration differs from the source contract")
    if output_root.exists():
        raise FileExistsError(f"benchmark output root already exists: {output_root}")

    benchmark = deepcopy(source)
    benchmark["created_utc"] = datetime.now(timezone.utc).isoformat()
    benchmark["stage"] = "engineering_throughput_benchmark"
    benchmark["attempts_per_level"] = attempts_per_level
    benchmark["total_attempts"] = level_count * attempts_per_level
    benchmark["seed_plan"] = {
        "registry": "engineering_and_calibration",
        "base_seed": base_seed,
        "last_seed": last_seed,
        "formal_seed_embargo_intact": True,
    }
    benchmark["execution"]["output_root"] = str(output_root)
    benchmark["execution"]["parallel_envs_per_shard"] = parallel_envs_per_shard
    benchmark["benchmark_provenance"] = {
        "purpose": "measure evaluator throughput only; never rank or select a model",
        "source_preregistration": {
            "path": str(source_path),
            "sha256": _sha256(source_path),
        },
        "selection_seed_registry_consumed": False,
        "final_blind_seed_registry_consumed": False,
        "continuous_campaign_seed_registry_consumed": False,
    }
    return benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-preregistration", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--attempts-per-level", type=int, default=4)
    parser.add_argument("--base-seed", type=int, default=1_400_300_000)
    parser.add_argument("--parallel-envs-per-shard", type=int, default=64)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source_path = args.source_preregistration.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen benchmark: {output}")
    benchmark = build(
        source_path=source_path,
        output_root=output_root,
        attempts_per_level=args.attempts_per_level,
        base_seed=args.base_seed,
        parallel_envs_per_shard=args.parallel_envs_per_shard,
    )
    _write_json_exclusive(output, benchmark)
    print(
        json.dumps(
            {
                "status": "FROZEN_NONFORMAL_BENCHMARK",
                "output": str(output),
                "sha256": _sha256(output),
                "attempts": benchmark["total_attempts"],
                "parallel_envs_per_shard": args.parallel_envs_per_shard,
                "seed_plan": benchmark["seed_plan"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
