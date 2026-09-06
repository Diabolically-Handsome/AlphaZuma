"""Freeze a seven-route postprocess before any AlphaZuma 55 formal seed."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _write_exclusive(path: Path, value: dict[str, Any]) -> None:
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
        raise FileExistsError(f"refusing to overwrite postprocess preregistration: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def build(
    *,
    source_path: Path,
    expected_source_sha256: str,
    route_path: Path,
    expected_route_sha256: str,
    controller_path: Path,
    source_controller_status: Path,
    output_root: Path,
) -> dict[str, Any]:
    _require(_sha256(source_path) == expected_source_sha256, "source postprocess hash differs")
    _require(_sha256(route_path) == expected_route_sha256, "seventh route hash differs")
    source = _read(source_path)
    route = _read(route_path)
    _require(
        source.get("schema") == "zuma-rl.alphazuma-55-postprocess-preregistration"
        and source.get("status") == "FROZEN_BEFORE_SELECTION",
        "unexpected source postprocess",
    )
    _require(len(source.get("training_routes", [])) == 6, "source route inventory differs")
    _require(
        route.get("schema") == "zuma-rl.overnight-multilevel-preregistration"
        and route.get("status") == "FROZEN_BEFORE_TRAINING"
        and len(route.get("runs", [])) == 1,
        "unexpected seventh route",
    )
    _require(route["campaign_id"] == source["campaign_id"], "seventh route campaign differs")
    _require(controller_path.resolve(strict=True).name == "run_alphazuma_55_postprocess_parallel_v4.py", "controller path differs")
    _require(source_controller_status.resolve(strict=True).is_file(), "source controller status is absent")
    source_status = _read(source_controller_status)
    _require(
        source_status.get("status") == "RUNNING"
        and source_status.get("phase") == "WAITING_FOR_TRAINING",
        "source controller crossed the formal-seed boundary",
    )
    for key in ("selection", "final_blind", "continuous"):
        _require(not Path(str(source["outputs"][key])).exists(), f"source {key} root already exists")
    _require(not output_root.exists(), "seven-route output root already exists")

    result = copy.deepcopy(source)
    result["created_utc"] = datetime.now(timezone.utc).isoformat()
    result["supersedes"] = {
        "path": str(source_path),
        "sha256": expected_source_sha256,
        "reason": "add one frozen engineering rescue route before any formal seed consumption",
    }
    result["training_routes"].append(
        {"path": str(route_path), "sha256": expected_route_sha256}
    )
    _require(len(result["training_routes"]) == 7, "seven-route inventory was not materialized")
    result["implementation"]["controller"] = {
        "path": str(controller_path.resolve(strict=True)),
        "sha256": _sha256(controller_path.resolve(strict=True)),
    }
    result["outputs"] = {
        "controller": str(output_root / "controller"),
        "selection": str(output_root / "selection"),
        "final_blind": str(output_root / "final-blind"),
        "continuous": str(output_root / "continuous"),
    }
    result["candidate_set_amendment"] = {
        "classification": "frozen_before_selection",
        "previous_candidate_routes": 6,
        "new_candidate_routes": 7,
        "seventh_route": {
            "path": str(route_path),
            "sha256": expected_route_sha256,
            "run_id": route["runs"][0]["id"],
        },
        "source_controller_status_snapshot": {
            "path": str(source_controller_status),
            "sha256_at_freeze": _sha256(source_controller_status),
            "status": source_status["status"],
            "phase": source_status["phase"],
            "updated_utc": source_status["updated_utc"],
        },
        "formal_seed_consumption_before_freeze": False,
        "selection_ranking_gate_and_seed_matrices_unchanged": True,
    }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--route", required=True, type=Path)
    parser.add_argument("--expected-route-sha256", required=True)
    parser.add_argument("--controller", required=True, type=Path)
    parser.add_argument("--source-controller-status", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite postprocess preregistration: {output}")
    result = build(
        source_path=args.source.expanduser().resolve(strict=True),
        expected_source_sha256=str(args.expected_source_sha256),
        route_path=args.route.expanduser().resolve(strict=True),
        expected_route_sha256=str(args.expected_route_sha256),
        controller_path=args.controller.expanduser().resolve(strict=True),
        source_controller_status=args.source_controller_status.expanduser().resolve(strict=True),
        output_root=args.output_root.expanduser().resolve(),
    )
    _write_exclusive(output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(output),
                "sha256": _sha256(output),
                "route_count": len(result["training_routes"]),
                "outputs": result["outputs"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
