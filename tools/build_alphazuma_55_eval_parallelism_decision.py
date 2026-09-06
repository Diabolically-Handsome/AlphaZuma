"""Freeze the evidence-based AlphaZuma 55 formal-evaluation width decision."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any


EXPECTED_MASTER_SHA256 = (
    "sha256:f43e10c299332b3e4b3497ff057596775c1bf3011e1ce08fc23f832a6652f4cb"
)
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


def _artifact(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


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
        raise FileExistsError(f"refusing to overwrite frozen decision: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _benchmark(
    *,
    preregistration_path: Path,
    result_path: Path,
    stderr_path: Path,
    expected_parallel_envs: int,
) -> dict[str, Any]:
    preregistration = _read_json(preregistration_path)
    result = _read_json(result_path)
    if preregistration.get("stage") != "engineering_throughput_benchmark":
        raise ValueError("throughput input is not an engineering benchmark")
    seed_plan = preregistration.get("seed_plan", {})
    if (
        seed_plan.get("registry") != "engineering_and_calibration"
        or seed_plan.get("formal_seed_embargo_intact") is not True
        or int(seed_plan.get("last_seed", FORMAL_SEED_FIRST)) >= FORMAL_SEED_FIRST
    ):
        raise ValueError("throughput benchmark consumed or could consume formal seeds")
    if int(preregistration["execution"]["parallel_envs_per_shard"]) != expected_parallel_envs:
        raise ValueError("benchmark preregistration has a different parallel width")
    if (
        result.get("status") != "COMPLETE"
        or result.get("error") is not None
        or result.get("completed_attempts") != result.get("expected_attempts")
        or int(result["runtime"]["parallel_envs"]) != expected_parallel_envs
    ):
        raise ValueError("throughput result is not a complete matching run")
    if result["preregistration"]["sha256"] != _sha256(preregistration_path):
        raise ValueError("throughput result is bound to another preregistration")
    if stderr_path.stat().st_size != 0:
        raise ValueError("throughput benchmark stderr is non-empty")
    return {
        "parallel_envs_per_shard": expected_parallel_envs,
        "preregistration": _artifact(preregistration_path),
        "result": _artifact(result_path),
        "stderr": _artifact(stderr_path),
        "shard_index": int(result["shard_index"]),
        "shard_count": int(result["shard_count"]),
        "attempts": int(result["completed_attempts"]),
        "vector_transitions": int(result["runtime"]["total_vector_transitions"]),
        "wall_seconds": float(result["runtime"]["wall_seconds"]),
        "seed_plan": seed_plan,
        "max_ticks": int(preregistration["environment"]["base_config"]["max_ticks"]),
        "model_sha256": str(preregistration["model"]["sha256"]),
        "evaluator_sha256": str(preregistration["evaluator"]["sha256"]),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    master = args.master.expanduser().resolve(strict=True)
    if _sha256(master) != EXPECTED_MASTER_SHA256:
        raise ValueError("master preregistration hash differs")
    sixty_four = _benchmark(
        preregistration_path=args.preregistration_64.expanduser().resolve(strict=True),
        result_path=args.result_64.expanduser().resolve(strict=True),
        stderr_path=args.stderr_64.expanduser().resolve(strict=True),
        expected_parallel_envs=64,
    )
    one_twelve = _benchmark(
        preregistration_path=args.preregistration_112.expanduser().resolve(strict=True),
        result_path=args.result_112.expanduser().resolve(strict=True),
        stderr_path=args.stderr_112.expanduser().resolve(strict=True),
        expected_parallel_envs=112,
    )
    for key in ("attempts", "vector_transitions", "max_ticks", "model_sha256", "evaluator_sha256"):
        if sixty_four[key] != one_twelve[key]:
            raise ValueError(f"benchmarks are not comparable: {key}")
    speedup = 1.0 - one_twelve["wall_seconds"] / sixty_four["wall_seconds"]
    if speedup < 0.10:
        raise ValueError("112 environments did not clear the frozen 10% speedup floor")
    return {
        "schema": "zuma-rl.alphazuma-55-evaluation-parallelism-decision",
        "version": 1,
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_id": "alphazuma-55-weekend-s81081401-v1",
        "master_preregistration": _artifact(master),
        "formal_seed_embargo_intact": True,
        "benchmarks": [sixty_four, one_twelve],
        "comparison": {
            "same_attempts": True,
            "same_vector_transitions": True,
            "wall_seconds_64": sixty_four["wall_seconds"],
            "wall_seconds_112": one_twelve["wall_seconds"],
            "relative_wall_reduction": speedup,
            "minimum_required_relative_wall_reduction": 0.10,
        },
        "decision": {
            "formal_parallel_envs_per_shard": 112,
            "formal_shard_count": 2,
            "devices": ["cuda:0", "cuda:1"],
            "selection_batches_per_model_per_shard": 1,
            "final_blind_batches_per_model_per_shard": 2,
            "continuous_batches_per_model_per_shard": 1,
            "training_semantics_changed": False,
            "evaluation_task_or_seed_semantics_changed": False,
            "selection_or_gate_semantics_changed": False,
        },
        "scope_limit": (
            "The benchmark proves width safety and relative throughput at the "
            "12,000-tick engineering guard. It does not by itself prove the "
            "absolute runtime of 180,000-tick formal matrices."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", required=True, type=Path)
    parser.add_argument("--preregistration-64", required=True, type=Path)
    parser.add_argument("--result-64", required=True, type=Path)
    parser.add_argument("--stderr-64", required=True, type=Path)
    parser.add_argument("--preregistration-112", required=True, type=Path)
    parser.add_argument("--result-112", required=True, type=Path)
    parser.add_argument("--stderr-112", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen decision: {output}")
    decision = build(args)
    _write_json_exclusive(output, decision)
    print(
        json.dumps(
            {
                "status": decision["status"],
                "output": str(output),
                "sha256": _sha256(output),
                "decision": decision["decision"],
                "comparison": decision["comparison"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
