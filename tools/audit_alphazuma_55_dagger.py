"""Independently audit one completed two-round AlphaZuma 55 DAgger run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
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


def audit(preregistration_path: Path, completion_path: Path) -> dict[str, Any]:
    prereg = _read_json(preregistration_path)
    completion = _read_json(completion_path)
    _require(
        prereg.get("schema") == "zuma-rl.alphazuma-55-dagger-preregistration"
        and prereg.get("version") == 1
        and prereg.get("status") == "FROZEN_BEFORE_TRAINING",
        "unexpected DAgger preregistration",
    )
    _require(
        completion.get("schema") == "zuma-rl.alphazuma-55-dagger-completion"
        and completion.get("version") == 1
        and completion.get("status") == "COMPLETE"
        and completion.get("formal_seed_consumption") is False,
        "DAgger completion is not clean",
    )
    master_path = _bound_path(prereg["master_preregistration"], "master")
    source_path = _bound_path(prereg["student_source_model"], "source model")
    selection_path = _bound_path(prereg["source_selection"], "source selection")
    trainer_path = _bound_path(prereg["trainer"], "trainer")
    for name, reference in prereg["implementation"].items():
        _bound_path(reference, f"implementation {name}")
    _require(
        completion["source_model"] == prereg["student_source_model"]
        and completion["source_selection"] == prereg["source_selection"],
        "completion source binding differs from preregistration",
    )
    _require(
        completion.get("teacher_policy_id")
        == "curve-aware-settled-strategic-masked-v3",
        "completion used another teacher policy",
    )
    run = prereg["run"]
    run_dir = Path(str(run["run_dir"])).resolve(strict=True)
    _require(completion_path.parent == run_dir, "completion is outside the frozen run")
    levels = [str(value) for value in prereg["levels"]]
    _require(len(levels) == 55 and len(set(levels)) == 55, "DAgger level scope is not 55 unique levels")
    registry = _read_json(master_path)["seed_registry"]["training"]
    seed_base = int(run["training_seed_base"])
    seed_last = int(run["training_seed_last_consumed"])
    _require(
        seed_last == seed_base + 2 * 55 - 1
        and int(registry["first"]) <= seed_base <= seed_last <= int(registry["last"]),
        "DAgger seed range is invalid",
    )

    collections = completion.get("collections")
    _require(isinstance(collections, list) and len(collections) == 2, "expected two DAgger collections")
    expected_probabilities = [float(value) for value in run["teacher_execution_probabilities"]]
    total_steps = 0
    total_samples = 0
    collection_summaries: list[dict[str, Any]] = []
    for round_index, collection in enumerate(collections):
        _require(int(collection["round_index"]) == round_index, "collection round index mismatch")
        _require(
            math.isclose(
                float(collection["teacher_execution_probability"]),
                expected_probabilities[round_index],
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            "teacher execution probability differs",
        )
        episodes = collection.get("episodes")
        _require(isinstance(episodes, list) and len(episodes) == 55, "collection must contain 55 episodes")
        _require([str(row["level_id"]) for row in episodes] == levels, "collection level order differs")
        for level_index, episode in enumerate(episodes):
            expected_seed = seed_base + round_index * 55 + level_index
            _require(int(episode["seed"]) == expected_seed, "collection seed mismatch")
            _require(int(episode["steps"]) > 0, "collection episode has no steps")
            _require(
                episode.get("outcome") in {"win", "loss", None}
                and ((episode.get("outcome") is None) == bool(episode["time_limit_truncated"])),
                "collection outcome/truncation mismatch",
            )
            _require(episode.get("observation_capacity_overflow") is False, "collection capacity overflow")
            comparisons = int(episode["teacher_executions"]) + int(episode["student_executions"])
            _require(comparisons == int(episode["steps"]), "execution count differs from steps")
            for key in ("exact_action_agreement", "verb_agreement"):
                value = float(episode[key])
                _require(0.0 <= value <= 1.0 and math.isfinite(value), f"invalid {key}")
        computed = {
            "wins": sum(row["outcome"] == "win" for row in episodes),
            "losses": sum(row["outcome"] == "loss" for row in episodes),
            "truncations": sum(bool(row["time_limit_truncated"]) for row in episodes),
            "capacity_overflows": sum(bool(row["observation_capacity_overflow"]) for row in episodes),
            "retained_samples": sum(int(row["retained_samples"]) for row in episodes),
            "teacher_executions": sum(int(row["teacher_executions"]) for row in episodes),
            "student_executions": sum(int(row["student_executions"]) for row in episodes),
        }
        for key, value in computed.items():
            _require(int(collection[key]) == value, f"collection aggregate mismatch: {key}")
        total_steps += sum(int(row["steps"]) for row in episodes)
        total_samples += int(collection["retained_samples"])
        collection_summaries.append({"round_index": round_index, **computed})

    aggregate = completion["aggregate_collection"]
    _require(
        int(aggregate["rounds"]) == 2
        and int(aggregate["episodes"]) == 110
        and int(aggregate["wins"]) == sum(row["wins"] for row in collection_summaries)
        and int(aggregate["losses"]) == sum(row["losses"] for row in collection_summaries)
        and int(aggregate["truncations"]) == sum(row["truncations"] for row in collection_summaries)
        and int(aggregate["capacity_overflows"]) == 0
        and int(aggregate["retained_samples"]) == total_samples,
        "completion aggregate collection mismatch",
    )

    optimizations = completion.get("optimizations")
    _require(isinstance(optimizations, list) and len(optimizations) == 2, "expected two optimizations")
    cumulative_samples = 0
    optimization_summaries: list[dict[str, Any]] = []
    source_steps = int(prereg["student_source_model"]["training_steps"])
    cumulative_steps = source_steps
    for round_index, optimization in enumerate(optimizations):
        cumulative_samples += int(collections[round_index]["retained_samples"])
        cumulative_steps += sum(int(row["steps"]) for row in collections[round_index]["episodes"])
        _require(
            int(optimization["round_index"]) == round_index
            and int(optimization["aggregate_samples"]) == cumulative_samples,
            "optimization aggregation differs",
        )
        _require(len(optimization["epochs"]) == int(run["epochs_per_round"]), "optimization epoch count differs")
        expected_checkpoint_epochs = list(
            range(
                int(run["checkpoint_interval_epochs"]),
                int(run["epochs_per_round"]) + 1,
                int(run["checkpoint_interval_epochs"]),
            )
        )
        checkpoints = optimization["checkpoints"]
        _require([int(row["epoch"]) for row in checkpoints] == expected_checkpoint_epochs, "checkpoint schedule differs")
        for checkpoint in checkpoints:
            _bound_path(checkpoint, "optimization checkpoint")
        round_model_path = _bound_path(optimization["round_model"], "round model")
        _require(round_model_path.parent == run_dir / f"round_{round_index + 1:02d}", "round model path differs")
        _require(int(optimization["round_model"]["num_timesteps"]) == cumulative_steps, "round model timestep count differs")
        optimization_summaries.append(
            {
                "round_index": round_index,
                "aggregate_samples": cumulative_samples,
                "epochs": len(optimization["epochs"]),
                "round_model": optimization["round_model"],
            }
        )

    final_path = _bound_path(completion["final_model"], "final model")
    _require(final_path == run_dir / "final_model.zip", "final model path differs")
    _require(
        int(completion["final_model"]["num_timesteps"]) == source_steps + total_steps,
        "final model timestep count differs",
    )
    _require(_sha256(source_path) == prereg["student_source_model"]["sha256"], "source changed")
    _require(_read_json(selection_path).get("status") == "SELECTED", "selection receipt status changed")

    return {
        "schema": "zuma-rl.alphazuma-55-dagger-independent-audit",
        "version": 1,
        "status": "PASS",
        "audited_utc": datetime.now(timezone.utc).isoformat(),
        "formal_selection_authority": False,
        "artifacts": {
            "preregistration": {"path": str(preregistration_path), "sha256": _sha256(preregistration_path)},
            "completion": {"path": str(completion_path), "sha256": _sha256(completion_path)},
            "trainer": {"path": str(trainer_path), "sha256": _sha256(trainer_path)},
            "auditor": {"path": str(Path(__file__).resolve()), "sha256": _sha256(Path(__file__).resolve())},
            "final_model": completion["final_model"],
        },
        "collections": collection_summaries,
        "optimizations": optimization_summaries,
        "total_student_visited_steps": total_steps,
        "final_num_timesteps": int(completion["final_model"]["num_timesteps"]),
        "interpretation_boundary": (
            "This PASS proves DAgger artifact, seed, and recipe integrity only; "
            "fresh validation is required for performance claims."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--completion", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite audit receipt: {output}")
    receipt = audit(
        args.preregistration.expanduser().resolve(strict=True),
        args.completion.expanduser().resolve(strict=True),
    )
    _write_exclusive(output, receipt)
    print(json.dumps({
        "status": receipt["status"],
        "output": {"path": str(output), "sha256": _sha256(output)},
        "collections": receipt["collections"],
        "final_num_timesteps": receipt["final_num_timesteps"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
