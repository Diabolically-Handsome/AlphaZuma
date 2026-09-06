"""Independently audit the frozen AlphaZuma 55 campaign result."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

try:
    from audit_alphazuma_v11_frontier_result import (
        _audit_evaluation as _strict_audit_evaluation,
    )
except ModuleNotFoundError:  # Imported as tools.<module> in tests.
    from tools.audit_alphazuma_v11_frontier_result import (
        _audit_evaluation as _strict_audit_evaluation,
    )


SCRIPT_PATH = Path(__file__).resolve()
EXPECTED_MASTER_SHA256 = (
    "sha256:f43e10c299332b3e4b3497ff057596775c1bf3011e1ce08fc23f832a6652f4cb"
)
EXPECTED_TRAINER_SHA256 = (
    "sha256:2886a72a7348e492f4d34f4ef9733eee1668df24e17cc896ace0094267d371ba"
)


class AuditError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _mapping(value: Any, name: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{name} is not an object")
    return value


def _artifact(reference: Mapping[str, Any], name: str) -> Path:
    path = Path(str(reference.get("path", ""))).expanduser().resolve(strict=True)
    _require(reference.get("sha256") == _sha256(path), f"{name} hash differs")
    return path


def _checkpoint_steps(path: Path) -> int:
    match = re.search(r"_(\d+)_steps\.zip$", path.name)
    _require(match is not None, f"cannot parse checkpoint steps: {path}")
    return int(match.group(1))


def _route_candidates(
    spec: Mapping[str, Any], preregistration: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_runs = spec.get("runs")
    _require(isinstance(raw_runs, list) and len(raw_runs) == 1, "route run count differs")
    run = _mapping(raw_runs[0], "route")
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
    _require(completion.get("status") == "COMPLETE", f"route is incomplete: {run_id}")
    _require(
        _mapping(config.get("preregistration"), "config preregistration").get("sha256")
        == preregistration["sha256"],
        f"route preregistration differs: {run_id}",
    )
    _require(_mapping(config.get("run"), "config run").get("id") == run_id, "route id differs")
    _require(config.get("trainer_sha256") == EXPECTED_TRAINER_SHA256, "route trainer differs")
    source_counter = int(_mapping(config.get("training_source_model"), "training source").get("timesteps", 0))
    final_counter = int(completion.get("actual_steps", -1))
    completed_new_steps = int(completion.get("actual_steps_this_run", -1))
    _require(int(completion.get("source_steps", -1)) == source_counter, "source counter differs")
    _require(completed_new_steps > 0 and final_counter - source_counter == completed_new_steps, "route step accounting differs")
    final_receipt = _mapping(completion.get("final_model"), "final model")
    final_path = Path(str(final_receipt.get("path", ""))).resolve(strict=True)
    _require(final_receipt.get("sha256") == _sha256(final_path), "final model changed")
    checkpoints = [
        path
        for path in (run_dir / "checkpoints").glob("*_steps.zip")
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
    midpoint_target = source_counter + completed_new_steps / 2.0
    midpoint = min(
        checkpoints,
        key=lambda path: (abs(_checkpoint_steps(path) - midpoint_target), _checkpoint_steps(path)),
    )
    midpoint_counter = _checkpoint_steps(midpoint)
    initial_steps = int(_mapping(spec.get("initial_model"), "initial_model").get("training_steps", 0))
    family = str(spec["route_family"])
    candidates = []
    for stage, model_path, counter in (
        ("mid", midpoint, midpoint_counter),
        ("final", final_path, final_counter),
    ):
        new_steps = counter - source_counter
        candidates.append(
            {
                "id": f"{run_id}-{stage}-{counter}",
                "training_steps": initial_steps + new_steps,
                "campaign_steps": new_steps,
                "path": str(model_path),
                "sha256": _sha256(model_path),
                "source": f"{run_id}:{stage}",
                "route_family": family,
                "device": run["device"],
            }
        )
    receipt = {
        "route_id": run_id,
        "status": "COMPLETE",
        "family": family,
        "device": run["device"],
        "preregistration": dict(preregistration),
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


def _candidate_inventory(post: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    for index, raw in enumerate(post["migrations"]):
        reference = _mapping(raw, f"migration {index}")
        receipt_path = _artifact(_mapping(reference["receipt"], "migration receipt"), "migration receipt")
        receipt = _read_json(receipt_path)
        _require(receipt.get("status") == "PASS", "migration is not PASS")
        model = _mapping(receipt.get("migrated_model"), "migrated model")
        model_path = Path(str(model["path"])).resolve(strict=True)
        _require(model.get("sha256") == _sha256(model_path), "migrated model changed")
        candidates.append(
            {
                "id": str(reference["id"]),
                "role": str(reference["role"]),
                "training_steps": int(model["num_timesteps"]),
                "path": str(model_path),
                "sha256": str(model["sha256"]),
                "source": f"migrated_{reference['role']}",
            }
        )
    route_receipts = []
    for raw in post["training_routes"]:
        path = _artifact(_mapping(raw, "training route"), "training route")
        spec = _read_json(path)
        additions, receipt = _route_candidates(
            spec, {"path": str(path), "sha256": _sha256(path)}
        )
        candidates.extend(additions)
        route_receipts.append(receipt)
    return candidates, route_receipts


def _summary(aggregate: Mapping[str, Any], model_id: str) -> dict[str, Any]:
    for row in aggregate["models"]:
        if row["model"]["id"] == model_id:
            return row
    raise AuditError(f"model summary missing: {model_id}")


def _rankings(
    aggregate: Mapping[str, Any], candidates: list[dict[str, Any]]
) -> tuple[list[str], list[str]]:
    robust = []
    speed = []
    for model in candidates:
        summary = _summary(aggregate, str(model["id"]))
        wins = [int(row["wins"]) for row in summary["level_summaries"]]
        coverage = sum(value > 0 for value in wins)
        minimum = min(wins)
        total = sum(wins)
        median = summary.get("median_win_ticks")
        robust.append(
            ((-coverage, -minimum, -total, float(median) if median is not None else float("inf"), str(model["sha256"])), str(model["id"]))
        )
        speed.append(
            ((-coverage, -total, float(median) if median is not None else float("inf"), str(model["sha256"])), str(model["id"]))
        )
    return (
        [model_id for _, model_id in sorted(robust)],
        [model_id for _, model_id in sorted(speed)],
    )


def _continuous(
    rows: Iterable[Mapping[str, Any]], level_ids: list[str], model_id: str
) -> dict[str, Any]:
    campaigns: dict[int, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        _require(row["model_id"] == model_id, "continuous model differs")
        campaigns[int(row["attempt_index"])][str(row["level_id"])] = row
    _require(set(campaigns) == set(range(4)), "continuous campaign ids differ")
    result = []
    for index in range(4):
        _require(set(campaigns[index]) == set(level_ids), "continuous level inventory differs")
        ordered = [campaigns[index][level_id] for level_id in level_ids]
        wins = sum(row.get("outcome") == "win" for row in ordered)
        result.append(
            {
                "campaign_index": index,
                "levels_attempted": 55,
                "wins": wins,
                "cleared_all_55": wins == 55,
                "first_failure_level": next(
                    (str(row["level_id"]) for row in ordered if row.get("outcome") != "win"),
                    None,
                ),
                "seeds": [int(row["seed"]) for row in ordered],
            }
        )
    return {
        "campaigns": result,
        "passed": any(row["cleared_all_55"] for row in result),
        "campaigns_cleared": sum(row["cleared_all_55"] for row in result),
    }


def _rows(root: Path) -> list[dict[str, Any]]:
    result = []
    for index in range(2):
        result.extend(_read_json(root / f"matrix-shard-{index:02d}-of-02.json")["attempts"])
    return result


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
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


def audit(preregistration_path: Path) -> dict[str, Any]:
    post = _read_json(preregistration_path)
    _require(
        post.get("schema") == "zuma-rl.alphazuma-55-postprocess-preregistration"
        and post.get("status") == "FROZEN_BEFORE_SELECTION",
        "postprocess preregistration differs",
    )
    master_path = _artifact(_mapping(post["master_preregistration"], "master"), "master")
    _require(_sha256(master_path) == EXPECTED_MASTER_SHA256, "master hash differs")
    master = _read_json(master_path)
    level_ids = list(master["scope"]["included_levels_in_adventure_order"])
    _require(len(level_ids) == 55, "level inventory differs")
    rules = _mapping(post.get("selection_rules"), "selection rules")
    _require(
        rules.get("capability_ranking")
        == master["candidate_and_selection_rules"]["selection_ranking_lexicographic"],
        "capability ranking contract differs",
    )
    _require(rules.get("single_policy_finalist") == "rank 1 under capability_ranking", "finalist contract differs")
    embargo = _mapping(post.get("seed_embargo"), "seed embargo")
    _require(embargo.get("selection") == [1_600_000_000, 1_600_000_219], "selection embargo differs")
    _require(embargo.get("final_blind") == [1_700_000_000, 1_700_000_439], "final embargo differs")
    _require(embargo.get("continuous") == [1_800_000_000, 1_800_000_219], "continuous embargo differs")
    auditor_ref = _mapping(post["implementation"]["independent_auditor"], "auditor")
    _require(_artifact(auditor_ref, "auditor") == SCRIPT_PATH.resolve(strict=True), "auditor path differs")
    for name, reference in _mapping(post.get("implementation"), "implementation").items():
        _artifact(_mapping(reference, f"implementation {name}"), f"implementation {name}")
    controller_root = Path(str(post["outputs"]["controller"])).resolve(strict=True)
    selection_root = Path(str(post["outputs"]["selection"])).resolve(strict=True)
    final_root = Path(str(post["outputs"]["final_blind"])).resolve(strict=True)
    continuous_root = Path(str(post["outputs"]["continuous"])).resolve(strict=True)
    status = _read_json(controller_root / "controller_status.json")
    _require(status.get("status") == "COMPLETE" and status.get("phase") == "COMPLETE", "controller is incomplete")

    candidates, route_receipts = _candidate_inventory(post)
    selection_manifest_path = controller_root / "selection-models-manifest.json"
    selection_prereg_path = controller_root / "selection-preregistration.json"
    selection_manifest = _read_json(selection_manifest_path)
    _require(selection_manifest.get("models") == candidates, "candidate manifest differs")
    _require(selection_manifest.get("route_receipts") == route_receipts, "route receipts differ")
    selection_eval = _strict_audit_evaluation(
        root=selection_root,
        preregistration_path=selection_prereg_path,
        manifest_path=selection_manifest_path,
        aggregate_path=selection_root / "aggregate.json",
        expected_level_ids=level_ids,
        expected_attempts_per_level=4,
        expected_seed_base=1_600_000_000,
        expected_model_ids=[str(row["id"]) for row in candidates],
    )
    robust_order, speed_order = _rankings(selection_eval["aggregate"], candidates)
    decision_path = controller_root / "selection-decision.json"
    decision = _read_json(decision_path)
    _require([row["model"]["id"] for row in decision["robust_ranking"]] == robust_order, "robust ranking differs")
    _require([row["model"]["id"] for row in decision["speed_ranking"]] == speed_order, "speed ranking differs")
    _require(decision["selected_single_policy"]["id"] == robust_order[0], "selected policy differs")
    _require(decision["speed_report_model"]["id"] == speed_order[0], "speed report model differs")

    by_id = {str(row["id"]): row for row in candidates}
    role_ids = {
        **{str(raw["role"]): str(raw["id"]) for raw in post["migrations"]},
        "selected_single_policy": robust_order[0],
        "speed_report_model": speed_order[0],
    }
    expected_models = []
    expected_roles = {}
    retained = {}
    for role, model_id in role_ids.items():
        model = by_id[model_id]
        if model["sha256"] not in retained:
            retained[model["sha256"]] = model_id
            expected_models.append(model)
        expected_roles[role] = retained[model["sha256"]]
    final_manifest_path = controller_root / "final-blind-models-manifest.json"
    final_prereg_path = controller_root / "final-blind-preregistration.json"
    final_manifest = _read_json(final_manifest_path)
    _require(final_manifest.get("models") == expected_models, "final models differ")
    _require(final_manifest.get("roles") == expected_roles, "final roles differ")
    final_eval = _strict_audit_evaluation(
        root=final_root,
        preregistration_path=final_prereg_path,
        manifest_path=final_manifest_path,
        aggregate_path=final_root / "aggregate.json",
        expected_level_ids=level_ids,
        expected_attempts_per_level=8,
        expected_seed_base=1_700_000_000,
        expected_model_ids=[str(row["id"]) for row in expected_models],
    )
    selected_summary = _summary(final_eval["aggregate"], expected_roles["selected_single_policy"])
    level_wins = [int(row["wins"]) for row in selected_summary["level_summaries"]]
    capability = {
        "levels_cleared": int(selected_summary["levels_cleared"]),
        "minimum_wins_per_level": min(level_wins),
        "total_wins": int(selected_summary["wins"]),
        "attempts": int(selected_summary["attempts"]),
    }
    capability_pass = (
        capability["levels_cleared"] == 55
        and capability["minimum_wins_per_level"] >= 1
        and capability["total_wins"] >= 220
        and capability["attempts"] == 440
    )

    continuous_manifest_path = controller_root / "continuous-models-manifest.json"
    continuous_prereg_path = controller_root / "continuous-preregistration.json"
    continuous_manifest = _read_json(continuous_manifest_path)
    _require(continuous_manifest.get("models") == [by_id[robust_order[0]]], "continuous model differs")
    continuous_eval = _strict_audit_evaluation(
        root=continuous_root,
        preregistration_path=continuous_prereg_path,
        manifest_path=continuous_manifest_path,
        aggregate_path=continuous_root / "aggregate.json",
        expected_level_ids=level_ids,
        expected_attempts_per_level=4,
        expected_seed_base=1_800_000_000,
        expected_model_ids=[robust_order[0]],
    )
    campaigns = _continuous(_rows(continuous_root), level_ids, robust_order[0])
    report_path = controller_root / "final_report.json"
    report = _read_json(report_path)
    _require(report.get("selected_single_policy") == by_id[robust_order[0]], "reported selected policy differs")
    gates = _mapping(report.get("success_gates"), "success gates")
    _require(gates["single_policy_capability_gate"]["actual"] == capability, "reported capability differs")
    _require(gates["single_policy_capability_gate"]["passed"] is capability_pass, "capability gate differs")
    _require(report.get("continuous_campaigns") == campaigns, "continuous campaigns differ")
    _require(gates["continuous_full_game_gate"]["passed"] is campaigns["passed"], "continuous gate differs")
    return {
        "schema": "zuma-rl.alphazuma-55-independent-final-audit",
        "version": 1,
        "status": "PASS",
        "preregistration": {"path": str(preregistration_path), "sha256": _sha256(preregistration_path)},
        "controller_status": {"path": str(controller_root / "controller_status.json"), "sha256": _sha256(controller_root / "controller_status.json")},
        "final_report": {"path": str(report_path), "sha256": _sha256(report_path)},
        "selected_single_policy": by_id[robust_order[0]],
        "candidate_count": len(candidates),
        "selection_attempts": selection_eval["attempts"],
        "final_blind_attempts": final_eval["attempts"],
        "continuous_attempts": continuous_eval["attempts"],
        "single_policy_capability_gate": capability_pass,
        "continuous_full_game_gate": campaigns["passed"],
        "checks": [
            "candidate_inventory_reconstructed",
            "selection_seed_and_attempt_matrix",
            "robust_and_speed_rankings_recomputed",
            "final_manifest_roles_and_deduplication",
            "final_blind_seed_and_attempt_matrix",
            "continuous_four_campaign_matrix",
            "single_policy_gates_recomputed",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    preregistration = args.preregistration.expanduser().resolve(strict=True)
    receipt = args.receipt.expanduser().resolve()
    if receipt.exists():
        existing = _read_json(receipt)
        _require(existing.get("status") == "PASS", "existing receipt is not PASS")
        print(json.dumps(existing, ensure_ascii=False, indent=2))
        return 0
    result = audit(preregistration)
    _write_exclusive(receipt, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
