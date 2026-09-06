"""Independently audit one non-formal AlphaZuma 55 validation matrix."""

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


def _resolve_bound_file(reference: Any, name: str) -> Path:
    _require(isinstance(reference, dict), f"{name} must be an object")
    path = Path(str(reference.get("path", ""))).expanduser().resolve(strict=True)
    _require(reference.get("sha256") == _sha256(path), f"{name} hash mismatch")
    return path


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
        raise FileExistsError(f"refusing to overwrite audit receipt: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def audit(
    *,
    master_path: Path,
    preregistration_path: Path,
    manifest_path: Path,
    shard_path: Path,
) -> dict[str, Any]:
    master = _read_json(master_path)
    preregistration = _read_json(preregistration_path)
    manifest = _read_json(manifest_path)
    shard = _read_json(shard_path)

    _require(
        master.get("schema")
        == "zuma-rl.alphazuma-55-weekend-master-preregistration"
        and master.get("version") == 1,
        "unexpected master preregistration",
    )
    _require(
        preregistration.get("schema")
        == "zuma-rl.zero-shot-multilevel-preregistration"
        and preregistration.get("version") == 1
        and preregistration.get("status") == "FROZEN_BEFORE_EVALUATION"
        and preregistration.get("stage") == "training_validation",
        "unexpected training-validation preregistration",
    )
    _require(
        manifest.get("schema") == "zuma-rl.zero-shot-models-manifest"
        and manifest.get("version") == 1
        and manifest.get("status") == "FROZEN"
        and manifest.get("stage") == "training_validation",
        "unexpected training-validation manifest",
    )
    _require(
        shard.get("schema") == "zuma-rl.zero-shot-multimodel-shard"
        and shard.get("version") == 1
        and shard.get("status") == "COMPLETE"
        and shard.get("error") is None,
        "validation shard is not a clean COMPLETE result",
    )

    _require(
        preregistration["master_preregistration"]["sha256"]
        == _sha256(master_path),
        "preregistration does not bind the supplied master",
    )
    _require(
        manifest["master_preregistration"]["sha256"] == _sha256(master_path),
        "manifest does not bind the supplied master",
    )
    _resolve_bound_file(preregistration["evaluator"], "evaluator")
    _resolve_bound_file(preregistration["builder"], "preregistration builder")
    _resolve_bound_file(manifest["builder"], "manifest builder")

    _require(
        shard["preregistration"]["sha256"] == _sha256(preregistration_path),
        "shard does not bind the supplied preregistration",
    )
    _require(
        shard["models_manifest"]["sha256"] == _sha256(manifest_path),
        "shard does not bind the supplied model manifest",
    )
    _require(
        shard["evaluator_sha256"] == preregistration["evaluator"]["sha256"],
        "shard evaluator hash differs from the contract",
    )

    frozen_levels = master["scope"]["included_levels_in_adventure_order"]
    prereg_levels = preregistration.get("levels")
    _require(isinstance(prereg_levels, list), "preregistration levels are invalid")
    level_ids = [str(level.get("id", "")) for level in prereg_levels]
    _require(level_ids == frozen_levels, "validation level order differs from master")

    model_specs = manifest.get("models")
    _require(isinstance(model_specs, list) and model_specs, "manifest models invalid")
    model_ids = [str(model.get("id", "")) for model in model_specs]
    _require(len(set(model_ids)) == len(model_ids), "duplicate model ids")
    for index, model in enumerate(model_specs):
        _resolve_bound_file(model, f"models[{index}]")
    _require(shard.get("models") == model_specs, "shard model specs differ from manifest")

    execution = preregistration["execution"]
    _require(
        int(execution["shard_count"]) == 1
        and len(execution["devices"]) == 1
        and shard.get("shard_count") == 1
        and shard.get("shard_index") == 0,
        "training validation must be one complete shard",
    )
    _require(
        int(shard["runtime"]["parallel_envs"])
        == int(execution["parallel_envs_per_shard"]),
        "runtime parallel width differs from preregistration",
    )

    attempts_per_level = int(preregistration["attempts_per_level"])
    attempts_per_model = len(level_ids) * attempts_per_level
    expected_attempts = attempts_per_model * len(model_specs)
    rows = shard.get("attempts")
    _require(isinstance(rows, list), "shard attempts are invalid")
    _require(
        len(rows) == expected_attempts
        and int(shard["expected_attempts"]) == expected_attempts
        and int(shard["completed_attempts"]) == expected_attempts,
        "attempt matrix is incomplete",
    )

    base_seed = int(preregistration["seed_plan"]["base_seed"])
    registry = master["seed_registry"]["training_validation"]
    _require(
        preregistration["seed_plan"]["registry"] == "training_validation"
        and int(registry["first"]) <= base_seed
        and int(preregistration["seed_plan"]["last_seed"])
        <= int(registry["last"]),
        "seed plan is outside the training-validation registry",
    )
    _require(
        preregistration["diagnostic_contract"]["formal_selection_authority"]
        is False
        and preregistration["diagnostic_contract"]["training_recipe_change_authority"]
        is False
        and preregistration["diagnostic_contract"]["formal_seed_ranges_consumed"]
        is False,
        "diagnostic non-authority contract is missing",
    )

    spec_by_id = {str(model["id"]): model for model in model_specs}
    level_index = {level_id: index for index, level_id in enumerate(level_ids)}
    seen: set[tuple[str, str, int]] = set()
    rows_by_model: dict[str, list[dict[str, Any]]] = {
        model_id: [] for model_id in model_ids
    }
    for row_index, row in enumerate(rows):
        _require(isinstance(row, dict), f"attempt[{row_index}] is not an object")
        model_id = str(row.get("model_id", ""))
        level_id = str(row.get("level_id", ""))
        attempt_index = int(row.get("attempt_index", -1))
        _require(model_id in spec_by_id, f"unknown model in attempt[{row_index}]")
        _require(level_id in level_index, f"unknown level in attempt[{row_index}]")
        _require(
            0 <= attempt_index < attempts_per_level,
            f"invalid attempt index in attempt[{row_index}]",
        )
        key = (model_id, level_id, attempt_index)
        _require(key not in seen, f"duplicate attempt key: {key}")
        seen.add(key)

        expected_seed = (
            base_seed + level_index[level_id] * attempts_per_level + attempt_index
        )
        spec = spec_by_id[model_id]
        _require(int(row["seed"]) == expected_seed, f"seed mismatch for {key}")
        _require(
            row["model_sha256"] == spec["sha256"]
            and int(row["training_steps"]) == int(spec["training_steps"]),
            f"model binding mismatch for {key}",
        )
        trajectory = str(row.get("trajectory_sha256", ""))
        _require(
            trajectory.startswith("sha256:") and len(trajectory) == 71,
            f"invalid trajectory hash for {key}",
        )
        outcome = row.get("outcome")
        truncated = bool(row.get("time_limit_truncated"))
        _require(
            outcome in {"win", "loss", None}
            and ((outcome is None) == truncated),
            f"outcome/truncation mismatch for {key}",
        )
        ticks = int(row.get("ticks", -1))
        seconds = float(row.get("seconds", math.nan))
        _require(ticks >= 0 and math.isfinite(seconds), f"invalid timing for {key}")
        _require(abs(seconds - ticks / 100.0) < 1e-9, f"seconds mismatch for {key}")
        rows_by_model[model_id].append(row)

    summaries: list[dict[str, Any]] = []
    for model_id in model_ids:
        model_rows = rows_by_model[model_id]
        wins = [row for row in model_rows if row["outcome"] == "win"]
        win_ticks = [int(row["ticks"]) for row in wins]
        summaries.append(
            {
                "model_id": model_id,
                "model_sha256": spec_by_id[model_id]["sha256"],
                "attempts": len(model_rows),
                "wins": len(wins),
                "losses": sum(row["outcome"] == "loss" for row in model_rows),
                "truncations": sum(
                    bool(row["time_limit_truncated"]) for row in model_rows
                ),
                "cleared_levels": len({str(row["level_id"]) for row in wins}),
                "winning_levels": [
                    {
                        "level_id": row["level_id"],
                        "seed": int(row["seed"]),
                        "ticks": int(row["ticks"]),
                        "score": int(row["score"]),
                        "trajectory_sha256": row["trajectory_sha256"],
                    }
                    for row in wins
                ],
                "median_winning_ticks": (
                    statistics.median(win_ticks) if win_ticks else None
                ),
            }
        )

    paired_comparison = None
    if len(summaries) == 2:
        first, second = summaries
        paired_comparison = {
            "baseline_model_id": first["model_id"],
            "candidate_model_id": second["model_id"],
            "wins_delta": int(second["wins"]) - int(first["wins"]),
            "cleared_levels_delta": int(second["cleared_levels"])
            - int(first["cleared_levels"]),
            "candidate_only_winning_levels": sorted(
                set(row["level_id"] for row in second["winning_levels"])
                - set(row["level_id"] for row in first["winning_levels"])
            ),
        }

    return {
        "schema": "zuma-rl.alphazuma-55-training-validation-audit",
        "version": 1,
        "status": "PASS",
        "audited_utc": datetime.now(timezone.utc).isoformat(),
        "formal_selection_authority": False,
        "training_recipe_change_authority": False,
        "artifacts": {
            "master": {"path": str(master_path), "sha256": _sha256(master_path)},
            "preregistration": {
                "path": str(preregistration_path),
                "sha256": _sha256(preregistration_path),
            },
            "models_manifest": {
                "path": str(manifest_path),
                "sha256": _sha256(manifest_path),
            },
            "matrix_shard": {"path": str(shard_path), "sha256": _sha256(shard_path)},
            "auditor": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256(Path(__file__).resolve()),
            },
        },
        "matrix": {
            "levels": len(level_ids),
            "attempts_per_model": attempts_per_model,
            "models": len(model_specs),
            "completed_attempts": expected_attempts,
            "seed_plan": preregistration["seed_plan"],
            "runtime": shard["runtime"],
        },
        "model_summaries": summaries,
        "paired_comparison": paired_comparison,
        "interpretation_boundary": (
            "This PASS proves artifact integrity and a paired training-health signal only; "
            "it is not selection, final blind, or continuous-campaign evidence."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-preregistration", required=True, type=Path)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--models-manifest", required=True, type=Path)
    parser.add_argument("--matrix-shard", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite audit receipt: {output}")
    receipt = audit(
        master_path=args.master_preregistration.expanduser().resolve(strict=True),
        preregistration_path=args.preregistration.expanduser().resolve(strict=True),
        manifest_path=args.models_manifest.expanduser().resolve(strict=True),
        shard_path=args.matrix_shard.expanduser().resolve(strict=True),
    )
    _write_json_exclusive(output, receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "output": str(output),
                "sha256": _sha256(output),
                "model_summaries": receipt["model_summaries"],
                "paired_comparison": receipt["paired_comparison"],
                "interpretation_boundary": receipt["interpretation_boundary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
