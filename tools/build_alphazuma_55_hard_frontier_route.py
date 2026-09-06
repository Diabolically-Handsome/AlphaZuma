"""Freeze one AlphaZuma 55 hard-frontier rescue route from a checkpoint."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any


CHECKPOINT_RE = re.compile(r"_(\d+)_steps\.zip$")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _write_json_exclusive(path: Path, value: Any) -> None:
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
        raise FileExistsError(f"refusing to overwrite rescue route: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _range(route: dict[str, Any]) -> tuple[str, int, int]:
    run = route["runs"][0]
    return str(run["id"]), int(run["episode_seed_base"]), int(run["episode_seed_last"])


def _require_disjoint(candidate: tuple[str, int, int], existing: list[tuple[str, int, int]]) -> None:
    candidate_id, first, last = candidate
    _require(first <= last, "candidate seed range is invalid")
    for route_id, route_first, route_last in existing:
        _require(
            last < route_first or route_last < first,
            f"candidate seed range overlaps {route_id}",
        )
        _require(candidate_id != route_id, f"candidate route id duplicates {route_id}")


def build_route(
    *,
    analysis_path: Path,
    expected_analysis_sha256: str,
    route_id: str,
    run_dir: Path,
    device: str,
    model_seed: int,
    episode_seed_base: int,
    num_envs: int,
) -> dict[str, Any]:
    _require(_sha256(analysis_path) == expected_analysis_sha256, "analysis hash mismatch")
    analysis = _read_json(analysis_path)
    _require(
        analysis.get("schema") == "zuma-rl.alphazuma-55-curriculum-frontier-analysis"
        and analysis.get("status") == "FROZEN_BEFORE_SOURCE_CHECKPOINT",
        "unexpected frontier analysis",
    )
    _require(
        analysis["authority_boundary"]["may_freeze_one_sixth_route_after_exact_checkpoint_exists"] is True,
        "analysis does not authorize one rescue route",
    )
    source = analysis["source_checkpoint"]
    checkpoint_path = Path(str(source["expected_path"])).resolve(strict=True)
    match = CHECKPOINT_RE.search(checkpoint_path.name)
    checkpoint_steps = int(source["training_steps"])
    _require(match is not None and int(match.group(1)) == checkpoint_steps, "checkpoint counter differs")
    source_route_path = Path(str(source["source_route_preregistration"]["path"])).resolve(strict=True)
    _require(
        source["source_route_preregistration"]["sha256"] == _sha256(source_route_path),
        "source route preregistration hash mismatch",
    )
    base = _read_json(source_route_path)
    _require(
        base.get("schema") == "zuma-rl.overnight-multilevel-preregistration"
        and base.get("status") == "FROZEN_BEFORE_TRAINING",
        "source route is not frozen",
    )
    _require(not run_dir.exists(), "rescue run directory already exists")
    _require(device in {"cuda:0", "cuda:1"} and num_envs > 0, "invalid rescue execution")

    master_path = Path(str(base["master_preregistration"]["path"])).resolve(strict=True)
    _require(base["master_preregistration"]["sha256"] == _sha256(master_path), "master hash mismatch")
    master = _read_json(master_path)
    stride = 250_000
    episode_seed_last = episode_seed_base + num_envs * stride - 1
    registry = master["seed_registry"]["training"]
    _require(
        int(registry["first"]) <= episode_seed_base
        and episode_seed_last <= int(registry["last"]),
        "rescue seed range is outside training registry",
    )
    existing_ranges: list[tuple[str, int, int]] = []
    for receipt in analysis["route_status_snapshots"]:
        path = Path(str(receipt["preregistration"]["path"])).resolve(strict=True)
        _require(receipt["preregistration"]["sha256"] == _sha256(path), "analysis route hash mismatch")
        existing_ranges.append(_range(_read_json(path)))
    _require_disjoint((route_id, episode_seed_base, episode_seed_last), existing_ranges)

    weights = analysis["rescue_initial_weights"]
    level_ids = [str(row["id"]) for row in base["levels"]]
    _require(list(weights) == level_ids and len(weights) == 55, "rescue weights differ from level order")
    _require(all(0.5 <= float(value) <= 4.75 for value in weights.values()), "rescue weight is out of range")

    result = copy.deepcopy(base)
    result["created_utc"] = datetime.now(timezone.utc).isoformat()
    result["route_family"] = "alphazuma-55-single-policy-hard-frontier-rescue"
    result["initial_model"] = {
        "id": f"{source['source_route_id']}-checkpoint-{checkpoint_steps}",
        "training_steps": int(base["initial_model"]["training_steps"]) + checkpoint_steps,
        "path": str(checkpoint_path),
        "sha256": _sha256(checkpoint_path),
    }
    result["implementation"]["hard_frontier_analysis"] = {
        "path": str(analysis_path),
        "sha256": expected_analysis_sha256,
    }
    result["implementation"]["hard_frontier_route_builder"] = {
        "path": str(Path(__file__).resolve()),
        "sha256": _sha256(Path(__file__).resolve()),
    }
    source_run = base["runs"][0]
    result["runs"] = [
        {
            "id": route_id,
            "description": (
                "AlphaZuma 55 hard-frontier rescue initialized from the frozen "
                f"{source['source_route_id']} {checkpoint_steps}-step checkpoint."
            ),
            "run_dir": str(run_dir),
            "device": device,
            "seed": model_seed,
            "episode_seed_base": episode_seed_base,
            "episode_seed_stride": stride,
            "episode_seed_last": episode_seed_last,
            "total_steps": int(source_run["total_steps"]),
            "num_envs": num_envs,
            "rollout_steps": int(source_run["rollout_steps"]),
            "batch_size": int(source_run["batch_size"]),
            "ppo_epochs": int(source_run["ppo_epochs"]),
            "learning_rate": float(source_run["learning_rate"]),
            "entropy_coef": float(source_run["entropy_coef"]),
            "checkpoint_every": int(source_run["checkpoint_every"]),
            "initial_weights": {level_id: float(weights[level_id]) for level_id in level_ids},
            "adaptive": copy.deepcopy(source_run["adaptive"]),
        }
    ]
    result["route_seed_namespace"] = {
        "first": episode_seed_base,
        "last": episode_seed_last,
        "registry": "training",
    }
    result["evidence_anchor"] = {
        "frontier_analysis": {"path": str(analysis_path), "sha256": expected_analysis_sha256},
        "source_route": {"path": str(source_route_path), "sha256": _sha256(source_route_path)},
        "source_checkpoint": {"path": str(checkpoint_path), "sha256": _sha256(checkpoint_path)},
        "source_checkpoint_steps": checkpoint_steps,
        "formal_seed_consumption_before_freeze": False,
    }
    _require(result["environment"] == base["environment"], "environment changed in rescue route")
    _require(result["levels"] == base["levels"], "level inventory changed in rescue route")
    _require(result["trainer"] == base["trainer"], "trainer changed in rescue route")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--expected-analysis-sha256", required=True)
    parser.add_argument("--route-id", required=True)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--device", choices=("cuda:0", "cuda:1"), required=True)
    parser.add_argument("--model-seed", required=True, type=int)
    parser.add_argument("--episode-seed-base", required=True, type=int)
    parser.add_argument("--num-envs", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite rescue route: {output}")
    value = build_route(
        analysis_path=args.analysis.expanduser().resolve(strict=True),
        expected_analysis_sha256=args.expected_analysis_sha256,
        route_id=args.route_id,
        run_dir=args.run_dir.expanduser().resolve(),
        device=args.device,
        model_seed=args.model_seed,
        episode_seed_base=args.episode_seed_base,
        num_envs=args.num_envs,
    )
    _write_json_exclusive(output, value)
    print(
        json.dumps(
            {
                "status": "FROZEN",
                "output": str(output),
                "sha256": _sha256(output),
                "route_id": value["runs"][0]["id"],
                "source_model": value["initial_model"],
                "seed_namespace": value["route_seed_namespace"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
