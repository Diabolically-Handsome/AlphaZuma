"""Independently audit the frozen 12-level polar re-anchor engineering probe."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
from typing import Any


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _bound_path(reference: Any, name: str) -> Path:
    _require(isinstance(reference, dict), f"{name} reference must be an object")
    path = Path(str(reference.get("path", ""))).expanduser().resolve(strict=True)
    _require(reference.get("sha256") == _sha256(path), f"{name} hash mismatch")
    return path


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
        raise FileExistsError(f"refusing to overwrite audit receipt: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def audit(
    *,
    master_path: Path,
    polar_plan_path: Path,
    preregistration_path: Path,
    manifest_path: Path,
    matrix_path: Path,
) -> dict[str, Any]:
    master = _read_json(master_path)
    plan = _read_json(polar_plan_path)
    prereg = _read_json(preregistration_path)
    manifest = _read_json(manifest_path)
    matrix = _read_json(matrix_path)

    _require(
        master.get("schema") == "zuma-rl.alphazuma-55-weekend-master-preregistration"
        and master.get("version") == 1,
        "unexpected master preregistration",
    )
    _require(
        plan.get("schema") == "zuma-rl.alphazuma-55-polar-reanchor-probe-plan"
        and plan.get("version") == 1
        and plan.get("status") == "FROZEN_BEFORE_VARIANT_CREATION",
        "unexpected polar plan",
    )
    _require(
        prereg.get("schema") == "zuma-rl.zero-shot-multilevel-preregistration"
        and prereg.get("version") == 1
        and prereg.get("status") == "FROZEN_BEFORE_EVALUATION"
        and prereg.get("stage") == "engineering_polar_reanchor_probe",
        "unexpected polar preregistration",
    )
    _require(
        manifest.get("schema") == "zuma-rl.zero-shot-models-manifest"
        and manifest.get("version") == 1
        and manifest.get("status") == "FROZEN"
        and manifest.get("stage") == "engineering_polar_reanchor_probe",
        "unexpected polar manifest",
    )
    _require(
        matrix.get("schema") == "zuma-rl.zero-shot-multimodel-shard"
        and matrix.get("version") == 1
        and matrix.get("status") == "COMPLETE"
        and matrix.get("error") is None,
        "polar matrix is not a clean COMPLETE result",
    )

    _require(
        plan["master_preregistration"]["sha256"] == _sha256(master_path),
        "polar plan does not bind the supplied master",
    )
    _require(
        prereg["polar_reanchor_probe"]["plan"]["sha256"]
        == _sha256(polar_plan_path),
        "polar preregistration does not bind the supplied plan",
    )
    _require(
        manifest["plan"]["sha256"] == _sha256(polar_plan_path),
        "polar manifest does not bind the supplied plan",
    )
    _require(
        matrix["preregistration"]["sha256"] == _sha256(preregistration_path),
        "matrix does not bind the supplied preregistration",
    )
    _require(
        matrix["models_manifest"]["sha256"] == _sha256(manifest_path),
        "matrix does not bind the supplied manifest",
    )
    evaluator_path = _bound_path(prereg["evaluator"], "evaluator")
    _require(
        matrix["evaluator_sha256"] == _sha256(evaluator_path),
        "matrix evaluator differs from the frozen evaluator",
    )

    expected_level_rows = plan["validation"]["levels"]
    expected_levels = [str(row["id"]) for row in expected_level_rows]
    actual_levels = [str(row["id"]) for row in prereg["levels"]]
    _require(actual_levels == expected_levels, "polar level order differs from plan")
    _require(len(expected_levels) == 12, "polar probe must contain 12 levels")

    models = manifest.get("models")
    _require(isinstance(models, list) and len(models) == 4, "expected four models")
    model_ids = [str(row["id"]) for row in models]
    _require(len(set(model_ids)) == len(model_ids), "duplicate model ids")
    for index, model in enumerate(models):
        _bound_path(model, f"models[{index}]")
    _require(matrix.get("models") == models, "matrix models differ from manifest")

    _require(
        int(matrix.get("shard_index", -1)) == 0
        and int(matrix.get("shard_count", -1)) == 1
        and int(matrix["runtime"]["parallel_envs"])
        == int(prereg["execution"]["parallel_envs_per_shard"]),
        "polar execution width or shard layout differs",
    )
    attempts_per_level = int(prereg["attempts_per_level"])
    _require(attempts_per_level == 1, "polar probe must use one paired attempt")
    expected_attempts = len(models) * len(expected_levels)
    rows = matrix.get("attempts")
    _require(
        isinstance(rows, list)
        and len(rows) == expected_attempts
        and int(matrix["expected_attempts"]) == expected_attempts
        and int(matrix["completed_attempts"]) == expected_attempts,
        "polar attempt matrix is incomplete",
    )

    base_seed = int(prereg["seed_plan"]["base_seed"])
    registry = master["seed_registry"]["training_validation"]
    _require(
        prereg["seed_plan"]["registry"] == "training_validation"
        and int(registry["first"]) <= base_seed
        and int(prereg["seed_plan"]["last_seed"]) <= int(registry["last"]),
        "polar seeds escaped the training-validation registry",
    )
    _require(
        prereg["diagnostic_contract"]["formal_selection_authority"] is False
        and prereg["diagnostic_contract"]["formal_seed_ranges_consumed"] is False,
        "polar non-authority boundary is missing",
    )

    by_model = {model_id: [] for model_id in model_ids}
    model_by_id = {str(row["id"]): row for row in models}
    level_index = {level_id: index for index, level_id in enumerate(expected_levels)}
    seen: set[tuple[str, str, int]] = set()
    for index, row in enumerate(rows):
        _require(isinstance(row, dict), f"attempt[{index}] is not an object")
        model_id = str(row.get("model_id", ""))
        level_id = str(row.get("level_id", ""))
        attempt_index = int(row.get("attempt_index", -1))
        _require(model_id in model_by_id, f"unknown model in attempt[{index}]")
        _require(level_id in level_index, f"unknown level in attempt[{index}]")
        key = (model_id, level_id, attempt_index)
        _require(attempt_index == 0 and key not in seen, f"duplicate/invalid key {key}")
        seen.add(key)
        spec = model_by_id[model_id]
        _require(int(row["seed"]) == base_seed + level_index[level_id], f"seed mismatch for {key}")
        _require(
            row["model_sha256"] == spec["sha256"]
            and int(row["training_steps"]) == int(spec["training_steps"]),
            f"model binding mismatch for {key}",
        )
        outcome = row.get("outcome")
        truncated = bool(row.get("time_limit_truncated"))
        _require(
            outcome in {"win", "loss", None} and ((outcome is None) == truncated),
            f"outcome/truncation mismatch for {key}",
        )
        ticks = int(row.get("ticks", -1))
        seconds = float(row.get("seconds", math.nan))
        _require(ticks >= 0 and math.isfinite(seconds), f"invalid timing for {key}")
        _require(abs(seconds - ticks / 100.0) < 1e-9, f"seconds mismatch for {key}")
        trajectory = str(row.get("trajectory_sha256", ""))
        _require(trajectory.startswith("sha256:") and len(trajectory) == 71, f"invalid trajectory for {key}")
        by_model[model_id].append(row)

    summaries: list[dict[str, Any]] = []
    for model_id in model_ids:
        model_rows = by_model[model_id]
        wins = [row for row in model_rows if row["outcome"] == "win"]
        summaries.append(
            {
                "model_id": model_id,
                "model_sha256": model_by_id[model_id]["sha256"],
                "attempts": len(model_rows),
                "wins": len(wins),
                "losses": sum(row["outcome"] == "loss" for row in model_rows),
                "truncations": sum(bool(row["time_limit_truncated"]) for row in model_rows),
                "total_score": sum(int(row["score"]) for row in model_rows),
                "median_winning_ticks": (
                    statistics.median(int(row["ticks"]) for row in wins)
                    if wins
                    else None
                ),
                "winning_levels": [str(row["level_id"]) for row in wins],
            }
        )

    return {
        "schema": "zuma-rl.alphazuma-55-polar-reanchor-probe-independent-audit",
        "version": 1,
        "status": "PASS",
        "audited_utc": datetime.now(timezone.utc).isoformat(),
        "formal_selection_authority": False,
        "artifacts": {
            "master": {"path": str(master_path), "sha256": _sha256(master_path)},
            "polar_plan": {"path": str(polar_plan_path), "sha256": _sha256(polar_plan_path)},
            "preregistration": {"path": str(preregistration_path), "sha256": _sha256(preregistration_path)},
            "models_manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
            "matrix": {"path": str(matrix_path), "sha256": _sha256(matrix_path)},
            "auditor": {"path": str(Path(__file__).resolve()), "sha256": _sha256(Path(__file__).resolve())},
        },
        "matrix": {
            "models": len(models),
            "levels": len(expected_levels),
            "completed_attempts": expected_attempts,
            "seed_plan": prereg["seed_plan"],
            "runtime": matrix["runtime"],
        },
        "model_summaries": summaries,
        "interpretation_boundary": (
            "This PASS proves integrity of a posthoc 12-level engineering probe; "
            "it is not original formal selection, final blind, or continuous evidence."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-preregistration", required=True, type=Path)
    parser.add_argument("--polar-plan", required=True, type=Path)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--models-manifest", required=True, type=Path)
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite audit receipt: {output}")
    receipt = audit(
        master_path=args.master_preregistration.expanduser().resolve(strict=True),
        polar_plan_path=args.polar_plan.expanduser().resolve(strict=True),
        preregistration_path=args.preregistration.expanduser().resolve(strict=True),
        manifest_path=args.models_manifest.expanduser().resolve(strict=True),
        matrix_path=args.matrix.expanduser().resolve(strict=True),
    )
    _write_exclusive(output, receipt)
    print(json.dumps({
        "status": receipt["status"],
        "output": {"path": str(output), "sha256": _sha256(output)},
        "matrix": receipt["matrix"],
        "model_summaries": receipt["model_summaries"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
