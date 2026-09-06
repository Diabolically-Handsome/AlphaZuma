"""Run the frozen AlphaZuma speed-essence selection and final blind campaign.

This controller is deliberately bound to the immutable S70081401 master hash.
It waits for all six training routes to become terminal, freezes the two
baseline plus midpoint/final route candidates, performs the paired selection
matrix, freezes independent robust and speed winners, then performs the paired
17-level final blind matrix.  It never derives evaluation semantics from a
route's shaping reward; the independently audited V1.1 canonical environment
is used for both evaluation phases.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import re
import statistics
import sys
import time
from typing import Any, Iterable, Mapping

try:
    from run_overnight_v11_postprocess import (
        EVALUATOR,
        _evaluate_and_summarize,
        _evaluation_preregistration,
        _level,
        _model_summary,
        _parse_utc,
        _read_json,
        _replace_json,
        _run_logged,
        _sha256,
        _utc_now,
        _write_new_json,
        _write_new_text,
    )
    from audit_alphazuma_v11_frontier_result import (
        _audit_evaluation as _strict_audit_evaluation,
    )
except ModuleNotFoundError:  # Imported as tools.<module> in tests.
    from tools.run_overnight_v11_postprocess import (
        EVALUATOR,
        _evaluate_and_summarize,
        _evaluation_preregistration,
        _level,
        _model_summary,
        _parse_utc,
        _read_json,
        _replace_json,
        _run_logged,
        _sha256,
        _utc_now,
        _write_new_json,
        _write_new_text,
    )
    from tools.audit_alphazuma_v11_frontier_result import (
        _audit_evaluation as _strict_audit_evaluation,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()
RECORDER = PROJECT_ROOT / "tools" / "record_paired_multimodel_level_best.py"
INDEPENDENT_AUDITOR = PROJECT_ROOT / "tools" / "audit_alphazuma_speed_essence_result.py"
EXPECTED_MASTER_SHA256 = (
    "sha256:5de3a8e4205d5a665caa840f1f3f8115da792ee22a3dd6c29d5691cb64cf204c"
)
EXPECTED_TRAINER_SHA256 = (
    "sha256:2886a72a7348e492f4d34f4ef9733eee1668df24e17cc896ace0094267d371ba"
)
EXPECTED_EVALUATOR_SHA256 = (
    "sha256:d0011f9ad5c6837315007a7fb31e128fb315e4baddb45959c5f3b98d11a6a7e3"
)
EXPECTED_TRAINING_PREREGISTRATIONS = (
    (
        "alphazuma-speed-essence-v1-lowdrift-s71081401-preregistration-v1.json",
        "sha256:b40b0bfda1fbe492b5cd66354342e852a129069f828b8ef74a34564f16a47fca",
    ),
    (
        "alphazuma-speed-essence-v1-speed-s72081401-preregistration-v1.json",
        "sha256:d0014e7d253d31bd0d3e003410cacdfdbcaaaa811f88785a361d0a7bee82edf3",
    ),
    (
        "alphazuma-speed-essence-v11-repair-s73081401-preregistration-v1.json",
        "sha256:bbb84c51ed600664b61d18222d6136e7914f793ad05adf977bfcdde3cf103ba9",
    ),
)
HARD_LEVEL_IDS = ("Jungle7", "village6", "village10")
SPEED_LEVEL_IDS = ("Jungle2", "Jungle4", "Jungle9", "village7")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _validate_execution_receipt(
    receipt_path: Path, *, master_path: Path, output_root: Path
) -> dict[str, Any]:
    receipt = _read_json(receipt_path)
    _require(
        receipt.get("schema")
        == "zuma-rl.alphazuma-speed-essence-postprocess-execution-receipt",
        "unexpected execution receipt schema",
    )
    _require(
        receipt.get("version") == 1
        and receipt.get("status") == "FROZEN_BEFORE_SELECTION",
        "execution receipt is not frozen before selection",
    )
    master = _mapping(receipt.get("master_preregistration"), "execution master")
    _require(
        Path(str(master.get("path", ""))).resolve(strict=True) == master_path,
        "execution receipt master path differs",
    )
    _require(
        master.get("sha256") == EXPECTED_MASTER_SHA256 == _sha256(master_path),
        "execution receipt master hash differs",
    )
    _require(
        Path(str(receipt.get("output_root", ""))).resolve() == output_root,
        "execution receipt output root differs",
    )
    expected_paths = {
        "controller": SCRIPT_PATH,
        "independent_auditor": INDEPENDENT_AUDITOR,
        "evaluator": EVALUATOR,
        "summarizer": PROJECT_ROOT / "tools" / "summarize_multimodel_evaluation.py",
        "matrix_auditor": PROJECT_ROOT / "tools" / "audit_alphazuma_v11_frontier_result.py",
        "record_renderer": RECORDER,
        "postprocess_helper": PROJECT_ROOT / "tools" / "run_overnight_v11_postprocess.py",
    }
    for name, expected_path in expected_paths.items():
        artifact = _mapping(receipt.get(name), f"execution {name}")
        actual_path = Path(str(artifact.get("path", ""))).resolve(strict=True)
        _require(actual_path == expected_path.resolve(strict=True), f"execution {name} path differs")
        _require(artifact.get("sha256") == _sha256(actual_path), f"execution {name} hash differs")
    return receipt


def _resolve_reference(path_text: str, *, fallback: Path, expected_sha256: str) -> Path:
    """Resolve a frozen reference, permitting only basename+hash path recovery.

    Some Windows console transcripts can display the Chinese project component
    incorrectly.  The artifact identity is therefore the frozen SHA-256; a
    fallback is accepted only when it has the same basename and exact hash.
    """

    requested = Path(path_text).expanduser()
    candidates = [requested, fallback]
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
        except (FileNotFoundError, OSError):
            continue
        if resolved.name == requested.name and _sha256(resolved) == expected_sha256:
            return resolved
    raise FileNotFoundError(
        f"frozen reference is missing or hash-mismatched: {path_text} ({expected_sha256})"
    )


def _load_contract(master_path: Path) -> dict[str, Any]:
    _require(_sha256(master_path) == EXPECTED_MASTER_SHA256, "master preregistration hash changed")
    master = _read_json(master_path)
    _require(
        master.get("schema") == "zuma-rl.alphazuma-speed-essence-campaign-preregistration",
        "unexpected master preregistration schema",
    )
    _require(master.get("version") == 1, "unexpected master preregistration version")
    _require(master.get("status") == "FROZEN_BEFORE_TRAINING", "master is not frozen")

    references = master.get("training_preregistrations")
    _require(isinstance(references, list) and len(references) == 3, "expected three training preregistrations")
    training_specs: list[dict[str, Any]] = []
    training_receipts: list[dict[str, Any]] = []
    for index, (filename, expected_hash) in enumerate(EXPECTED_TRAINING_PREREGISTRATIONS):
        reference = _mapping(references[index], f"training_preregistrations[{index}]")
        _require(reference.get("sha256") == expected_hash, "training preregistration receipt changed")
        path = _resolve_reference(
            str(reference.get("path", "")),
            fallback=master_path.parent / filename,
            expected_sha256=expected_hash,
        )
        spec = _read_json(path)
        _require(spec.get("schema") == "zuma-rl.overnight-multilevel-preregistration", "wrong training schema")
        _require(spec.get("status") == "FROZEN_BEFORE_TRAINING", "training preregistration is not frozen")
        _require(_mapping(spec.get("trainer"), "trainer").get("sha256") == EXPECTED_TRAINER_SHA256, "trainer hash differs")
        training_specs.append(spec)
        training_receipts.append({"path": str(path), "sha256": expected_hash})

    _require(_sha256(EVALUATOR) == EXPECTED_EVALUATOR_SHA256, "evaluator bytes changed")
    _require(_mapping(master.get("evaluator"), "evaluator").get("sha256") == EXPECTED_EVALUATOR_SHA256, "master evaluator receipt differs")

    anchor = _mapping(training_specs[0].get("evidence_anchor"), "evidence_anchor")
    source_master_hash = str(anchor.get("source_master_sha256", ""))
    source_master_name = Path(str(anchor.get("source_master_path", ""))).name
    source_master_path = _resolve_reference(
        str(anchor.get("source_master_path", "")),
        fallback=master_path.parent / source_master_name,
        expected_sha256=source_master_hash,
    )
    canonical = _read_json(source_master_path)
    levels = canonical.get("levels")
    environment = canonical.get("environment")
    _require(isinstance(levels, list) and len(levels) == 17, "canonical inventory is not seventeen levels")
    _require(isinstance(environment, dict), "canonical environment is absent")
    level_ids = [str(row["id"]) for row in levels]
    _require(len({value.casefold() for value in level_ids}) == 17, "canonical level ids are duplicated")
    for spec in training_specs:
        _require(spec.get("levels") == levels, "training level inventory differs from canonical V1.1")
        route_environment = _mapping(spec.get("environment"), "training environment")
        _require(route_environment.get("base_config") == environment.get("base_config"), "base environment differs from canonical V1.1")
        _require(route_environment.get("input_profile") == environment.get("input_profile"), "human input profile differs from canonical V1.1")

    selection = _mapping(master.get("selection"), "selection")
    final_blind = _mapping(master.get("final_blind"), "final_blind")
    selection_ids = [str(value) for value in selection.get("level_ids", [])]
    _require(len(selection_ids) == 9 and len({value.casefold() for value in selection_ids}) == 9, "selection inventory differs")
    _require(all(value.casefold() in {item.casefold() for item in level_ids} for value in selection_ids), "selection level missing")
    _require(int(selection.get("attempts_per_level", -1)) == 4, "selection attempts differ")
    _require(int(selection.get("seed_base", -1)) == 1_200_000_000, "selection seed base differs")
    _require(int(selection.get("last_seed", -1)) == 1_200_000_035, "selection last seed differs")
    _require(int(final_blind.get("attempts_per_level", -1)) == 8, "final attempts differ")
    _require(int(final_blind.get("seed_base", -1)) == 1_300_000_000, "final seed base differs")
    _require(int(final_blind.get("last_seed", -1)) == 1_300_000_135, "final last seed differs")
    references_ticks = _mapping(selection.get("frozen_v1_best_tick_references"), "speed references")
    _require(list(references_ticks) == list(SPEED_LEVEL_IDS), "speed reference level order differs")
    _require(all(int(references_ticks[value]) > 0 for value in SPEED_LEVEL_IDS), "invalid speed reference")

    baselines = _mapping(master.get("baselines"), "baselines")
    for name in ("v1", "v11"):
        model = _mapping(baselines.get(name), f"baselines.{name}")
        model_path = Path(str(model.get("path", ""))).resolve(strict=True)
        _require(model.get("sha256") == _sha256(model_path), f"{name} baseline model hash changed")

    runs: list[dict[str, Any]] = []
    for prereg_index, (spec, prereg_receipt) in enumerate(zip(training_specs, training_receipts, strict=True)):
        raw_runs = spec.get("runs")
        _require(isinstance(raw_runs, list) and len(raw_runs) == 2, "each training preregistration must contain two routes")
        initial_model = _mapping(spec.get("initial_model"), "initial_model")
        for run_index, raw_run in enumerate(raw_runs):
            run = dict(_mapping(raw_run, f"runs[{run_index}]"))
            run["family"] = str(spec["id"])
            run["preregistration"] = prereg_receipt
            run["initial_model"] = initial_model
            run["manifest_index"] = len(runs)
            runs.append(run)
    _require(len(runs) == 6, "expected six formal routes")
    _require(len({str(run["id"]) for run in runs}) == 6, "route ids are duplicated")
    _require(len({str(run["run_dir"]) for run in runs}) == 6, "route directories are duplicated")

    return {
        "master": master,
        "master_path": master_path,
        "training_specs": training_specs,
        "training_receipts": training_receipts,
        "canonical_source": {"path": str(source_master_path), "sha256": source_master_hash},
        "levels": levels,
        "environment": environment,
        "runs": runs,
    }


def _checkpoint_steps(path: Path) -> int:
    match = re.search(r"_(\d+)_steps\.zip$", path.name)
    if match is None:
        raise ValueError(f"cannot parse checkpoint steps: {path}")
    return int(match.group(1))


def _terminal_snapshot(contract: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    terminal = True
    snapshots: dict[str, Any] = {}
    for run in contract["runs"]:
        root = Path(str(run["run_dir"])).resolve()
        completion = root / "completion.json"
        failure = root / "failure.json"
        training = root / "training_status.json"
        if completion.exists():
            value = _read_json(completion)
        elif failure.exists():
            value = _read_json(failure)
        elif training.exists():
            value = _read_json(training)
            terminal = False
        else:
            value = {"status": "STARTING"}
            terminal = False
        snapshots[str(run["id"])] = value
    return terminal, snapshots


def _route_candidates(run: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    run_id = str(run["id"])
    run_dir = Path(str(run["run_dir"])).resolve(strict=True)
    failure_path = run_dir / "failure.json"
    if failure_path.exists():
        return [], {
            "route_id": run_id,
            "status": "FAILED",
            "failure": {"path": str(failure_path), "sha256": _sha256(failure_path)},
        }
    completion_path = run_dir / "completion.json"
    config_path = run_dir / "config.json"
    completion = _read_json(completion_path)
    config = _read_json(config_path)
    _require(completion.get("status") == "COMPLETE", f"route completion is not complete: {run_id}")
    _require(_mapping(config.get("preregistration"), "config.preregistration").get("sha256") == run["preregistration"]["sha256"], f"route preregistration differs: {run_id}")
    _require(_mapping(config.get("run"), "config.run").get("id") == run_id, f"route id differs: {run_id}")
    _require(config.get("trainer_sha256") == EXPECTED_TRAINER_SHA256, f"route trainer differs: {run_id}")
    source_counter = int(_mapping(config.get("training_source_model"), "training_source_model").get("timesteps", 0))
    final_counter = int(completion.get("actual_steps", -1))
    completed_new_steps = int(completion.get("actual_steps_this_run", -1))
    _require(int(completion.get("source_steps", -1)) == source_counter, f"source counter differs: {run_id}")
    _require(completed_new_steps > 0 and final_counter - source_counter == completed_new_steps, f"step accounting differs: {run_id}")
    final_receipt = _mapping(completion.get("final_model"), "completion.final_model")
    final_path = Path(str(final_receipt.get("path", ""))).resolve(strict=True)
    _require(final_receipt.get("sha256") == _sha256(final_path), f"final model hash changed: {run_id}")

    midpoint_target = source_counter + completed_new_steps / 2.0
    checkpoint_dir = run_dir / "checkpoints"
    checkpoints = [
        path
        for path in checkpoint_dir.glob("*_steps.zip")
        if source_counter < _checkpoint_steps(path) <= final_counter
    ]
    if not checkpoints:
        return [], {
            "route_id": run_id,
            "status": "NO_ELIGIBLE_CHECKPOINT",
            "config": {"path": str(config_path), "sha256": _sha256(config_path)},
            "completion": {"path": str(completion_path), "sha256": _sha256(completion_path)},
            "actual_steps_this_run": completed_new_steps,
        }
    midpoint = min(
        checkpoints,
        key=lambda path: (abs(_checkpoint_steps(path) - midpoint_target), _checkpoint_steps(path)),
    )
    midpoint_counter = _checkpoint_steps(midpoint)
    initial_steps = int(_mapping(run.get("initial_model"), "initial_model").get("training_steps", 0))
    candidates: list[dict[str, Any]] = []
    for stage, path, counter in (
        ("mid", midpoint, midpoint_counter),
        ("final", final_path, final_counter),
    ):
        new_steps = counter - source_counter
        candidates.append(
            {
                "id": f"{run_id}-{stage}-{counter}",
                "training_steps": initial_steps + new_steps,
                "campaign_steps": new_steps,
                "path": str(path),
                "sha256": _sha256(path),
                "source": f"{run_id}:{stage}",
                "route_family": run["family"],
                "device": run["device"],
            }
        )
    receipt = {
        "route_id": run_id,
        "status": "COMPLETE",
        "family": run["family"],
        "device": run["device"],
        "preregistration": run["preregistration"],
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "completion": {"path": str(completion_path), "sha256": _sha256(completion_path)},
        "source_counter_timesteps": source_counter,
        "actual_counter_timesteps": final_counter,
        "actual_steps_this_run": completed_new_steps,
        "midpoint_target_counter": midpoint_target,
        "midpoint_checkpoint": {"path": str(midpoint), "sha256": _sha256(midpoint)},
        "midpoint_counter_timesteps": midpoint_counter,
        "final_model": {"path": str(final_path), "sha256": _sha256(final_path)},
    }
    return candidates, receipt


def _freeze_candidates(contract: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    master = contract["master"]
    candidates: list[dict[str, Any]] = []
    for name in ("v1", "v11"):
        model = dict(master["baselines"][name])
        model["path"] = str(Path(str(model["path"])).resolve(strict=True))
        model["source"] = f"frozen_{name}_baseline"
        candidates.append(model)
    receipts: list[dict[str, Any]] = []
    for run in contract["runs"]:
        additions, receipt = _route_candidates(run)
        candidates.extend(additions)
        receipts.append(receipt)
    expected = 2 + 2 * sum(receipt["status"] == "COMPLETE" for receipt in receipts)
    _require(len(candidates) == expected, "candidate count differs from terminal route receipts")
    _require(len({str(model["id"]) for model in candidates}) == len(candidates), "candidate ids are duplicated")
    return candidates, receipts


def _median_best_ticks(summary: Mapping[str, Any]) -> float | None:
    values = [
        int(level["best_attempt"]["ticks"])
        for level in summary["level_summaries"]
        if isinstance(level.get("best_attempt"), dict)
    ]
    return statistics.median(values) if values else None


def _selection_decision(
    master: Mapping[str, Any], aggregate: Mapping[str, Any], candidates: list[dict[str, Any]]
) -> dict[str, Any]:
    references = master["selection"]["frozen_v1_best_tick_references"]
    robust_rows: list[dict[str, Any]] = []
    speed_rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        summary = _model_summary(aggregate, str(candidate["id"]))
        anchor = _level(summary, "Jungle2")
        hard = [_level(summary, level_id) for level_id in HARD_LEVEL_IDS]
        speed = [_level(summary, level_id) for level_id in SPEED_LEVEL_IDS]
        all_levels = list(summary["level_summaries"])
        median_ticks = _median_best_ticks(summary)
        ratios = [
            int(level["best_attempt"]["ticks"]) / int(references[level_id])
            for level_id, level in zip(SPEED_LEVEL_IDS, speed, strict=True)
            if isinstance(level.get("best_attempt"), dict)
        ]
        speed_complete = len(ratios) == len(SPEED_LEVEL_IDS)
        geometric_mean = math.prod(ratios) ** (1.0 / len(ratios)) if speed_complete else None
        common = {
            "anchor_wins": int(anchor["wins"]),
            "anchor_retained": int(anchor["wins"]) >= 3,
            "selection_levels_cleared": sum(int(level["wins"]) > 0 for level in all_levels),
            "selection_wins": sum(int(level["wins"]) for level in all_levels),
        }
        robust_metrics = {
            **common,
            "hard_levels_cleared": sum(int(level["wins"]) > 0 for level in hard),
            "median_best_win_ticks": median_ticks,
        }
        robust_key = (
            int(robust_metrics["anchor_retained"]),
            int(robust_metrics["hard_levels_cleared"]),
            int(robust_metrics["selection_levels_cleared"]),
            int(robust_metrics["selection_wins"]),
            -float(median_ticks) if median_ticks is not None else -1_000_000_000.0,
            -index,
        )
        robust_rows.append(
            {"manifest_index": index, "model": candidate, "metrics": robust_metrics, "rank_key": list(robust_key), "_key": robust_key}
        )
        speed_metrics = {
            **common,
            "speed_levels_cleared": sum(int(level["wins"]) > 0 for level in speed),
            "speed_reference_levels_complete": speed_complete,
            "speed_geometric_mean_ratio": geometric_mean,
            "speed_reference_ticks": {level_id: references[level_id] for level_id in SPEED_LEVEL_IDS},
        }
        speed_key = (
            int(speed_metrics["anchor_retained"]),
            int(speed_metrics["speed_levels_cleared"]),
            int(speed_metrics["selection_wins"]),
            -float(geometric_mean) if geometric_mean is not None else -1_000_000_000.0,
            -index,
        )
        speed_rows.append(
            {"manifest_index": index, "model": candidate, "metrics": speed_metrics, "rank_key": list(speed_key), "_key": speed_key}
        )

    def ranked(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result = sorted(rows, key=lambda row: row["_key"], reverse=True)
        for rank, row in enumerate(result, start=1):
            row["rank"] = rank
            del row["_key"]
        return result

    robust = ranked(robust_rows)
    speed = ranked(speed_rows)
    return {
        "schema": "zuma-rl.alphazuma-speed-essence-selection-decision",
        "version": 1,
        "status": "COMPLETE",
        "completed_utc": _utc_now(),
        "aggregate_receipts": {
            "created_utc": aggregate["created_utc"],
            "preregistration": aggregate["preregistration"],
            "models_manifest": aggregate["models_manifest"],
            "validation": aggregate["validation"],
        },
        "frozen_rankings": {
            "robust": master["selection"]["robust_ranking"],
            "speed": master["selection"]["speed_ranking"],
        },
        "robust_selected_model": robust[0]["model"],
        "speed_selected_model": speed[0]["model"],
        "robust_ranking": robust,
        "speed_ranking": speed,
    }


def _levels(contract: Mapping[str, Any], requested: Iterable[str]) -> list[dict[str, Any]]:
    by_id = {str(level["id"]).casefold(): level for level in contract["levels"]}
    result = []
    for level_id in requested:
        key = str(level_id).casefold()
        _require(key in by_id, f"frozen level is missing: {level_id}")
        result.append(by_id[key])
    return result


def _paired_result(rows: Iterable[dict[str, Any]], first_id: str, second_id: str) -> dict[str, Any]:
    by_key: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        model_id = str(row["model_id"])
        if model_id in {first_id, second_id}:
            by_key[(str(row["level_id"]), int(row["attempt_index"]))][model_id] = row
    counts = {"both_win": 0, "first_only_win": 0, "second_only_win": 0, "both_fail": 0}
    for pair in by_key.values():
        if first_id == second_id:
            _require(first_id in pair, "same-model paired row is missing")
            first = second = pair[first_id]
        else:
            _require(set(pair) == {first_id, second_id}, "paired attempt is incomplete")
            first, second = pair[first_id], pair[second_id]
        first_win = first.get("outcome") == "win"
        second_win = second.get("outcome") == "win"
        key = "both_win" if first_win and second_win else "first_only_win" if first_win else "second_only_win" if second_win else "both_fail"
        counts[key] += 1
    counts["pairs"] = len(by_key)
    counts["same_model"] = first_id == second_id
    return counts


def _all_rows(evaluation_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(2):
        shard = _read_json(evaluation_root / f"matrix-shard-{index:02d}-of-02.json")
        rows.extend(shard["attempts"])
    return rows


def _summary_metrics(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model": summary["model"],
        "levels": int(summary["levels"]),
        "levels_cleared": int(summary["levels_cleared"]),
        "attempts": int(summary["attempts"]),
        "wins": int(summary["wins"]),
        "losses": int(summary["losses"]),
        "truncations": int(summary["truncations"]),
        "win_rate": float(summary["win_rate"]),
        "median_win_ticks": summary.get("median_win_ticks"),
        "median_win_seconds": summary.get("median_win_seconds"),
        "best_attempt": summary.get("best_attempt"),
        "level_summaries": summary["level_summaries"],
    }


def _new_level_bests(
    aggregate: Mapping[str, Any], roles: Mapping[str, str]
) -> list[dict[str, Any]]:
    summaries = {str(item["model"]["id"]): item for item in aggregate["models"]}
    baseline_ids = {roles["v1_baseline"], roles["v11_baseline"]}
    selected_ids = {roles["robust_selected"], roles["speed_selected"]} - baseline_ids
    if not selected_ids:
        return []
    level_ids = [str(level["level_id"]) for level in next(iter(summaries.values()))["level_summaries"]]
    results = []
    for level_id in level_ids:
        baseline_attempts = [
            _level(summaries[model_id], level_id).get("best_attempt") for model_id in baseline_ids
        ]
        selected_attempts = [
            _level(summaries[model_id], level_id).get("best_attempt") for model_id in selected_ids
        ]
        baseline_wins = [value for value in baseline_attempts if isinstance(value, dict)]
        selected_wins = [value for value in selected_attempts if isinstance(value, dict)]
        if not selected_wins:
            continue
        selected = min(selected_wins, key=lambda value: (int(value["ticks"]), str(value["model_id"])))
        baseline = min(baseline_wins, key=lambda value: (int(value["ticks"]), str(value["model_id"]))) if baseline_wins else None
        if baseline is None or int(selected["ticks"]) < int(baseline["ticks"]):
            results.append(
                {
                    "level_id": level_id,
                    "selected_attempt": selected,
                    "best_baseline_attempt": baseline,
                    "improvement_ticks": None if baseline is None else int(baseline["ticks"]) - int(selected["ticks"]),
                }
            )
    return results


def _render_new_bests(
    *,
    new_bests: list[dict[str, Any]],
    aggregate_path: Path,
    preregistration_path: Path,
    manifest_path: Path,
    original_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    video_root = output_root / "record-videos"
    video_root.mkdir()
    artifacts = []
    for index, best in enumerate(new_bests):
        level_id = str(best["level_id"])
        model_id = str(best["selected_attempt"]["model_id"])
        safe_level = re.sub(r"[^A-Za-z0-9_-]+", "-", level_id).strip("-").lower()
        output = video_root / f"{index + 1:02d}-{safe_level}.mp4"
        stdout = video_root / f"{index + 1:02d}-{safe_level}.stdout.json"
        stderr = video_root / f"{index + 1:02d}-{safe_level}.stderr.log"
        _run_logged(
            [
                sys.executable,
                str(RECORDER),
                "--aggregate",
                str(aggregate_path),
                "--preregistration",
                str(preregistration_path),
                "--models-manifest",
                str(manifest_path),
                "--original-root",
                str(original_root),
                "--level-id",
                level_id,
                "--model-id",
                model_id,
                "--output",
                str(output),
                "--device",
                "cuda:0",
            ],
            stdout_path=stdout,
            stderr_path=stderr,
        )
        sidecar = output.with_suffix(".json")
        poster = output.with_name(output.stem + "-poster.png")
        metadata = _read_json(sidecar)
        _require(metadata.get("status") == "COMPLETE", "record video sidecar is incomplete")
        _require(metadata["candidate"].get("trajectory_sha256") == best["selected_attempt"].get("trajectory_sha256"), "record trajectory hash differs from blind attempt")
        artifacts.append(
            {
                "level_id": level_id,
                "model_id": model_id,
                "trajectory_sha256": best["selected_attempt"]["trajectory_sha256"],
                "video": {"path": str(output), "sha256": _sha256(output)},
                "poster": {"path": str(poster), "sha256": _sha256(poster)},
                "sidecar": {"path": str(sidecar), "sha256": _sha256(sidecar)},
            }
        )
    return {
        "schema": "zuma-rl.alphazuma-speed-essence-record-videos",
        "version": 1,
        "status": "COMPLETE",
        "created_utc": _utc_now(),
        "rule": "selected candidate is faster than both frozen baselines on the paired final-blind level",
        "artifacts": artifacts,
    }


def _final_report(
    *,
    master: Mapping[str, Any],
    selection: Mapping[str, Any],
    aggregate: Mapping[str, Any],
    evaluation_root: Path,
    roles: Mapping[str, str],
    new_bests: list[dict[str, Any]],
    video_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    summaries = {str(item["model"]["id"]): item for item in aggregate["models"]}
    rows = _all_rows(evaluation_root)
    role_metrics = {role: _summary_metrics(summaries[model_id]) for role, model_id in roles.items()}
    paired: dict[str, Any] = {}
    for selected_role in ("robust_selected", "speed_selected"):
        for baseline_role in ("v1_baseline", "v11_baseline"):
            key = f"{selected_role}_vs_{baseline_role}"
            paired[key] = _paired_result(rows, roles[selected_role], roles[baseline_role])
    return {
        "schema": "zuma-rl.alphazuma-speed-essence-final-report",
        "version": 1,
        "status": "COMPLETE",
        "created_utc": _utc_now(),
        "master_preregistration_sha256": EXPECTED_MASTER_SHA256,
        "scientifically_valid": bool(aggregate.get("validation")) and all(aggregate["validation"].values()),
        "scope": {
            "simulator_state_policy": True,
            "original_client_world_record_claim": False,
            "best_seed_speed_reported_separately_from_robustness": True,
        },
        "selection": {
            "robust_selected_model": selection["robust_selected_model"],
            "speed_selected_model": selection["speed_selected_model"],
        },
        "roles": dict(roles),
        "final_blind": role_metrics,
        "paired_results": paired,
        "new_level_bests_against_both_paired_baselines": new_bests,
        "record_video_manifest": video_manifest,
        "deadline_utc": master["schedule"]["goal_end_utc"],
    }


def _brief(report: Mapping[str, Any], *, chinese: bool) -> str:
    roles = report["roles"]
    lines = [
        "# AlphaZuma Speed Essence 最终简报" if chinese else "# AlphaZuma Speed Essence Final Brief",
        "",
        (
            "口径：冻结模拟器状态策略、精英人类输入限制；不宣称原版客户端世界纪录。"
            if chinese
            else "Scope: frozen simulator state policies under elite-human input limits; this is not an original-client world-record claim."
        ),
        "",
    ]
    labels = {
        "v1_baseline": "V1 基线" if chinese else "V1 baseline",
        "v11_baseline": "V1.1 基线" if chinese else "V1.1 baseline",
        "robust_selected": "稳健榜入选" if chinese else "robust-selected",
        "speed_selected": "速度榜入选" if chinese else "speed-selected",
    }
    for role in ("v1_baseline", "v11_baseline", "robust_selected", "speed_selected"):
        metrics = report["final_blind"][role]
        best = metrics.get("best_attempt")
        best_text = "无胜局" if chinese else "no win"
        if isinstance(best, dict):
            best_text = f"{best['level_id']} {float(best['seconds']):.2f}s"
        median = metrics.get("median_win_seconds")
        median_text = "—" if median is None else f"{float(median):.2f}s"
        lines.append(
            f"- {labels[role]} (`{roles[role]}`): "
            + (
                f"通关 {metrics['levels_cleared']}/17，胜局 {metrics['wins']}/{metrics['attempts']}，胜局中位时间 {median_text}，best-seed {best_text}。"
                if chinese
                else f"coverage {metrics['levels_cleared']}/17, wins {metrics['wins']}/{metrics['attempts']}, median winning time {median_text}, best-seed {best_text}."
            )
        )
    lines.extend(["", "## 配对结果" if chinese else "## Paired results", ""])
    for name, paired in report["paired_results"].items():
        lines.append(
            f"- `{name}`: first-only={paired['first_only_win']}, second-only={paired['second_only_win']}, both-win={paired['both_win']}, both-fail={paired['both_fail']}."
        )
    count = len(report["new_level_bests_against_both_paired_baselines"])
    lines.extend(
        [
            "",
            (f"新关卡最佳（相对本次两基线）：{count}。" if chinese else f"New level bests versus both paired baselines: {count}."),
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-preregistration", type=Path, required=True)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--execution-receipt", type=Path)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    master_path = args.master_preregistration.expanduser().resolve(strict=True)
    original_root = args.original_root.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve()
    contract = _load_contract(master_path)
    master = contract["master"]
    execution_receipt = None
    if args.execution_receipt is not None:
        execution_receipt_path = args.execution_receipt.expanduser().resolve(strict=True)
        execution_receipt = _validate_execution_receipt(
            execution_receipt_path,
            master_path=master_path,
            output_root=output_root,
        )
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "VALID",
                    "master_preregistration": str(master_path),
                    "master_sha256": _sha256(master_path),
                    "controller": {"path": str(SCRIPT_PATH), "sha256": _sha256(SCRIPT_PATH)},
                    "routes": len(contract["runs"]),
                    "normal_candidate_count": 14,
                    "selection_attempts_per_model": 36,
                    "final_attempts_per_model": 136,
                    "canonical_environment_source": contract["canonical_source"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if execution_receipt is None:
        raise ValueError("--execution-receipt is required for formal postprocessing")
    if output_root.exists():
        raise FileExistsError(f"refusing to reuse controller output root: {output_root}")
    selection_root = Path(str(master["execution"]["selection_output_root"])).resolve()
    final_root = Path(str(master["execution"]["final_output_root"])).resolve()
    for path in (selection_root, final_root):
        if path.exists():
            raise FileExistsError(f"refusing to reuse formal evaluation root: {path}")
    output_root.mkdir(parents=True)
    status_path = output_root / "controller_status.json"
    state: dict[str, Any] = {
        "schema": "zuma-rl.alphazuma-speed-essence-controller-status",
        "version": 1,
        "status": "RUNNING",
        "phase": "WAITING_FOR_TRAINING",
        "started_utc": _utc_now(),
        "updated_utc": _utc_now(),
        "master_preregistration": {"path": str(master_path), "sha256": _sha256(master_path)},
        "controller": {"path": str(SCRIPT_PATH), "sha256": _sha256(SCRIPT_PATH)},
        "execution_receipt": {
            "path": str(args.execution_receipt.expanduser().resolve(strict=True)),
            "sha256": _sha256(args.execution_receipt.expanduser().resolve(strict=True)),
        },
        "training": {},
        "error": None,
    }
    _replace_json(status_path, state)
    try:
        latest_terminal = _parse_utc(master["schedule"]["training_stop_utc"]) + timedelta(minutes=20)
        while True:
            terminal, snapshot = _terminal_snapshot(contract)
            state["training"] = snapshot
            state["updated_utc"] = _utc_now()
            _replace_json(status_path, state)
            if terminal:
                break
            if datetime.now(timezone.utc) > latest_terminal:
                raise TimeoutError("six training routes did not become terminal within twenty minutes")
            time.sleep(args.poll_seconds)

        state["phase"] = "FREEZING_CANDIDATES"
        state["updated_utc"] = _utc_now()
        _replace_json(status_path, state)
        candidates, route_receipts = _freeze_candidates(contract)
        selection_manifest = output_root / "selection-models-manifest.json"
        _write_new_json(
            selection_manifest,
            {
                "schema": "zuma-rl.zero-shot-models-manifest",
                "version": 1,
                "status": "FROZEN",
                "frozen_utc": _utc_now(),
                "purpose": "speed-essence paired candidate selection on isolated frozen seeds",
                "master_preregistration": {"path": str(master_path), "sha256": _sha256(master_path)},
                "candidate_rule": master["candidate_rule"],
                "models": candidates,
                "route_receipts": route_receipts,
            },
        )
        selection_prereg = output_root / "selection-preregistration.json"
        _write_new_json(
            selection_prereg,
            _evaluation_preregistration(
                baseline_model=candidates[0],
                levels=_levels(contract, master["selection"]["level_ids"]),
                attempts_per_level=4,
                base_seed=1_200_000_000,
                output_root=selection_root,
                shard_count=2,
                devices=["cuda:0", "cuda:1"],
                environment=contract["environment"],
                purpose="speed-essence paired selection; not final blind evidence",
            ),
        )
        state["phase"] = "SELECTION_EVALUATION"
        state["updated_utc"] = _utc_now()
        _replace_json(status_path, state)
        selection_aggregate = _evaluate_and_summarize(
            phase="selection",
            preregistration=selection_prereg,
            manifest=selection_manifest,
            original_root=original_root,
            output_root=selection_root,
            shard_count=2,
            purpose="selection",
        )
        selection_audit = _strict_audit_evaluation(
            root=selection_root,
            preregistration_path=selection_prereg,
            manifest_path=selection_manifest,
            aggregate_path=selection_root / "aggregate.json",
            expected_level_ids=[str(value) for value in master["selection"]["level_ids"]],
            expected_attempts_per_level=4,
            expected_seed_base=1_200_000_000,
            expected_model_ids=[str(model["id"]) for model in candidates],
        )
        selection = _selection_decision(master, selection_aggregate, candidates)
        selection["strict_matrix_audit"] = {
            key: selection_audit[key]
            for key in ("attempts", "seed_range", "preregistration_sha256", "manifest_sha256", "aggregate_sha256", "shard_sha256")
        }
        selection_path = output_root / "selection-decision.json"
        _write_new_json(selection_path, selection)

        robust = dict(selection["robust_selected_model"])
        speed = dict(selection["speed_selected_model"])
        candidate_by_id = {str(model["id"]): model for model in candidates}
        source_roles = {
            "v1_baseline": str(master["baselines"]["v1"]["id"]),
            "v11_baseline": str(master["baselines"]["v11"]["id"]),
            "robust_selected": str(robust["id"]),
            "speed_selected": str(speed["id"]),
        }
        ordered_roles = (
            "v1_baseline",
            "v11_baseline",
            "robust_selected",
            "speed_selected",
        )
        final_models: list[dict[str, Any]] = []
        retained_by_hash: dict[str, str] = {}
        roles: dict[str, str] = {}
        for role in ordered_roles:
            model = dict(candidate_by_id[source_roles[role]])
            model_hash = str(model["sha256"])
            if model_hash in retained_by_hash:
                roles[role] = retained_by_hash[model_hash]
                continue
            model["path"] = str(Path(str(model["path"])).resolve(strict=True))
            final_models.append(model)
            retained_by_hash[model_hash] = str(model["id"])
            roles[role] = str(model["id"])
        final_manifest = output_root / "final-blind-models-manifest.json"
        _write_new_json(
            final_manifest,
            {
                "schema": "zuma-rl.zero-shot-models-manifest",
                "version": 1,
                "status": "FROZEN",
                "frozen_utc": _utc_now(),
                "purpose": "speed-essence paired final blind comparison",
                "models": final_models,
                "roles": roles,
                "deduplicated_by_model_sha256": True,
                "selection_decision": {"path": str(selection_path), "sha256": _sha256(selection_path)},
                "selection_aggregate": {"path": str(selection_root / "aggregate.json"), "sha256": _sha256(selection_root / "aggregate.json")},
            },
        )
        final_prereg = output_root / "final-blind-preregistration.json"
        _write_new_json(
            final_prereg,
            _evaluation_preregistration(
                baseline_model=final_models[0],
                levels=contract["levels"],
                attempts_per_level=8,
                base_seed=1_300_000_000,
                output_root=final_root,
                shard_count=2,
                devices=["cuda:0", "cuda:1"],
                environment=contract["environment"],
                purpose="speed-essence paired final blind all-seventeen evaluation",
            ),
        )
        state["phase"] = "FINAL_BLIND_EVALUATION"
        state["updated_utc"] = _utc_now()
        _replace_json(status_path, state)
        final_aggregate = _evaluate_and_summarize(
            phase="final-blind",
            preregistration=final_prereg,
            manifest=final_manifest,
            original_root=original_root,
            output_root=final_root,
            shard_count=2,
            purpose="final-blind",
        )
        final_audit = _strict_audit_evaluation(
            root=final_root,
            preregistration_path=final_prereg,
            manifest_path=final_manifest,
            aggregate_path=final_root / "aggregate.json",
            expected_level_ids=[str(level["id"]) for level in contract["levels"]],
            expected_attempts_per_level=8,
            expected_seed_base=1_300_000_000,
            expected_model_ids=[str(model["id"]) for model in final_models],
        )

        state["phase"] = "RECORD_VIDEOS"
        state["updated_utc"] = _utc_now()
        _replace_json(status_path, state)
        new_bests = _new_level_bests(final_aggregate, roles)
        video_manifest_value = _render_new_bests(
            new_bests=new_bests,
            aggregate_path=final_root / "aggregate.json",
            preregistration_path=final_prereg,
            manifest_path=final_manifest,
            original_root=original_root,
            output_root=output_root,
        )
        video_manifest = output_root / "record-videos-manifest.json"
        _write_new_json(video_manifest, video_manifest_value)

        state["phase"] = "FINAL_REPORT"
        state["updated_utc"] = _utc_now()
        _replace_json(status_path, state)
        report = _final_report(
            master=master,
            selection=selection,
            aggregate=final_aggregate,
            evaluation_root=final_root,
            roles=roles,
            new_bests=new_bests,
            video_manifest={"path": str(video_manifest), "sha256": _sha256(video_manifest), "artifacts": video_manifest_value["artifacts"]},
        )
        report["strict_final_matrix_audit"] = {
            key: final_audit[key]
            for key in ("attempts", "seed_range", "preregistration_sha256", "manifest_sha256", "aggregate_sha256", "shard_sha256")
        }
        report_path = output_root / "final_report.json"
        chinese_path = output_root / "FINAL_BRIEF.zh-CN.md"
        english_path = output_root / "FINAL_BRIEF.en.md"
        _write_new_json(report_path, report)
        _write_new_text(chinese_path, _brief(report, chinese=True))
        _write_new_text(english_path, _brief(report, chinese=False))
        state.update(
            {
                "status": "COMPLETE",
                "phase": "COMPLETE",
                "updated_utc": _utc_now(),
                "selection_decision": {"path": str(selection_path), "sha256": _sha256(selection_path)},
                "final_report": {"path": str(report_path), "sha256": _sha256(report_path)},
                "brief_zh_cn": {"path": str(chinese_path), "sha256": _sha256(chinese_path)},
                "brief_en": {"path": str(english_path), "sha256": _sha256(english_path)},
            }
        )
        _replace_json(status_path, state)
    except BaseException as error:
        state.update(
            {
                "status": "FAILED",
                "phase": "FAILED",
                "updated_utc": _utc_now(),
                "error": {"type": type(error).__name__, "message": str(error)},
            }
        )
        _replace_json(status_path, state)
        raise


if __name__ == "__main__":
    main()
