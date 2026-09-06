"""Independently audit a configured AlphaZuma 55 successor result."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping

try:
    from audit_alphazuma_55_training_validation import audit as audit_engineering
    from audit_alphazuma_v11_frontier_result import (
        _audit_evaluation as audit_evaluation,
    )
    from build_alphazuma_55_eval_contract import _sha256
except ModuleNotFoundError:
    from tools.audit_alphazuma_55_training_validation import (
        audit as audit_engineering,
    )
    from tools.audit_alphazuma_v11_frontier_result import (
        _audit_evaluation as audit_evaluation,
    )
    from tools.build_alphazuma_55_eval_contract import _sha256


SCRIPT_PATH = Path(__file__).resolve()
LEVEL_COUNT = 55
EXPECTED_RANKING = [
    "wins_desc",
    "level_coverage_desc",
    "total_score_desc",
    "median_winning_ticks_asc",
    "smaller_model_first_as_tie_break",
]


class AuditError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _artifact(reference: Any, name: str) -> Path:
    _require(isinstance(reference, dict), f"{name} must be an object")
    path = Path(str(reference.get("path", ""))).expanduser().resolve(strict=True)
    _require(reference.get("sha256") == _sha256(path), f"{name} hash differs")
    return path


def _model_bytes(model: Mapping[str, Any]) -> int:
    if model.get("bytes") is not None:
        size = int(model["bytes"])
    else:
        size = Path(str(model["path"])).resolve(strict=True).stat().st_size
    _require(size > 0, "engineering model byte count is invalid")
    return size


def _rank(
    models: list[dict[str, Any]], rows: Iterable[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    order = {str(model["id"]): index for index, model in enumerate(models)}
    grouped: dict[str, list[Mapping[str, Any]]] = {
        model_id: [] for model_id in order
    }
    for row in rows:
        model_id = str(row["model_id"])
        _require(model_id in grouped, f"unknown engineering model: {model_id}")
        grouped[model_id].append(row)
    ranked = []
    for model in models:
        model_id = str(model["id"])
        model_rows = grouped[model_id]
        _require(len(model_rows) == LEVEL_COUNT, f"{model_id} lacks 55 attempts")
        wins = [row for row in model_rows if row.get("outcome") == "win"]
        ticks = [int(row["ticks"]) for row in wins]
        metrics = {
            "wins": len(wins),
            "cleared_levels": len({str(row["level_id"]) for row in wins}),
            "total_score": sum(int(row["score"]) for row in model_rows),
            "median_winning_ticks": statistics.median(ticks) if ticks else None,
            "model_bytes": _model_bytes(model),
        }
        ranked.append(
            {
                "model": dict(model),
                "metrics": metrics,
                "_key": (
                    -metrics["wins"],
                    -metrics["cleared_levels"],
                    -metrics["total_score"],
                    (
                        float(metrics["median_winning_ticks"])
                        if metrics["median_winning_ticks"] is not None
                        else math.inf
                    ),
                    metrics["model_bytes"],
                    order[model_id],
                ),
            }
        )
    ranked.sort(key=lambda row: row["_key"])
    for index, row in enumerate(ranked, start=1):
        row["rank"] = index
        del row["_key"]
    return ranked


def _rows(root: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index in range(2):
        result.extend(_read(root / f"matrix-shard-{index:02d}-of-02.json")["attempts"])
    return result


def _continuous(
    rows: Iterable[Mapping[str, Any]], level_ids: list[str], model_id: str
) -> dict[str, Any]:
    campaigns: dict[int, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        _require(str(row["model_id"]) == model_id, "continuous model differs")
        campaign = int(row["attempt_index"])
        level_id = str(row["level_id"])
        _require(level_id not in campaigns[campaign], "continuous level repeats")
        campaigns[campaign][level_id] = row
    _require(set(campaigns) == set(range(4)), "continuous campaign ids differ")
    summaries = []
    for campaign in range(4):
        mapping = campaigns[campaign]
        _require(set(mapping) == set(level_ids), "continuous level inventory differs")
        ordered = [mapping[level_id] for level_id in level_ids]
        wins = sum(row.get("outcome") == "win" for row in ordered)
        summaries.append(
            {
                "campaign_index": campaign,
                "levels_attempted": LEVEL_COUNT,
                "wins": wins,
                "cleared_all_55": wins == LEVEL_COUNT,
                "first_failure_level": next(
                    (
                        str(row["level_id"])
                        for row in ordered
                        if row.get("outcome") != "win"
                    ),
                    None,
                ),
                "seeds": [int(row["seed"]) for row in ordered],
            }
        )
    return {
        "campaigns": summaries,
        "passed": any(row["cleared_all_55"] for row in summaries),
        "campaigns_cleared": sum(row["cleared_all_55"] for row in summaries),
    }


def _summary(aggregate: Mapping[str, Any], model_id: str) -> dict[str, Any]:
    for row in aggregate["models"]:
        if str(row["model"]["id"]) == model_id:
            return dict(row)
    raise AuditError(f"model summary absent: {model_id}")


def _capability(summary: Mapping[str, Any]) -> tuple[dict[str, int], bool]:
    wins_by_level = [int(row["wins"]) for row in summary["level_summaries"]]
    actual = {
        "levels_cleared": int(summary["levels_cleared"]),
        "minimum_wins_per_level": min(wins_by_level),
        "total_wins": int(summary["wins"]),
        "attempts": int(summary["attempts"]),
    }
    passed = (
        actual["levels_cleared"] == LEVEL_COUNT
        and actual["minimum_wins_per_level"] >= 1
        and actual["total_wins"] >= 220
        and actual["attempts"] == 440
    )
    return actual, passed


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


def audit(master_path: Path) -> dict[str, Any]:
    master_path = master_path.resolve(strict=True)
    master = _read(master_path)
    _require(
        master.get("schema")
        == "zuma-rl.alphazuma-55-configured-successor-master-preregistration"
        and master.get("version") == 1
        and master.get("status") == "FROZEN_BEFORE_SUCCESSOR_FINAL_BLIND",
        "unexpected configured successor master",
    )
    level_ids = list(master.get("scope", {}).get("included_levels_in_adventure_order", []))
    _require(
        len(level_ids) == LEVEL_COUNT and len(set(level_ids)) == LEVEL_COUNT,
        "configured successor level scope differs",
    )
    implementation = master.get("implementation")
    _require(isinstance(implementation, dict), "implementation bindings are absent")
    _require(
        _artifact(implementation.get("independent_auditor"), "independent auditor")
        == SCRIPT_PATH,
        "master binds another independent auditor",
    )
    _artifact(implementation.get("master_builder"), "master builder")
    controller_path = _artifact(implementation.get("controller"), "controller")
    _artifact(implementation.get("watcher"), "watcher")
    builder_path = _artifact(implementation.get("evaluation_builder"), "evaluation builder")
    evaluator_path = _artifact(implementation.get("evaluator"), "evaluator")
    _artifact(implementation.get("matrix_auditor"), "matrix auditor")
    engineering_auditor_path = _artifact(
        implementation.get("engineering_auditor"), "engineering auditor"
    )
    successor = master.get("successor")
    _require(isinstance(successor, dict), "configured successor contract is absent")
    _artifact(successor.get("intent"), "successor intent")
    source_master_path = _artifact(
        successor.get("source_campaign_master"), "source campaign master"
    )
    engineering = successor.get("engineering_validation")
    _require(isinstance(engineering, dict), "engineering contract is absent")
    plan_path = _artifact(engineering.get("plan"), "engineering plan")
    expected_ids = [str(value) for value in engineering.get("eligible_model_ids", [])]
    _require(
        len(expected_ids) == 4 and len(set(expected_ids)) == 4,
        "eligible ids differ",
    )
    _require(engineering.get("ranking") == EXPECTED_RANKING, "ranking differs")
    manifest_key = str(engineering.get("manifest_plan_binding_key", ""))
    _require(manifest_key, "engineering manifest binding key is absent")
    audit_path = Path(str(engineering["audit_receipt"])).resolve(strict=True)
    engineering_receipt = _read(audit_path)
    _require(
        engineering_receipt.get("status") == "PASS"
        and engineering_receipt.get("formal_selection_authority") is False,
        "engineering audit is not a non-formal PASS",
    )
    artifacts = engineering_receipt.get("artifacts")
    _require(isinstance(artifacts, dict), "engineering artifacts are absent")
    _require(
        _artifact(artifacts.get("master"), "engineering source master")
        == source_master_path,
        "engineering source master differs",
    )
    prereg_path = _artifact(artifacts.get("preregistration"), "engineering preregistration")
    manifest_path = _artifact(artifacts.get("models_manifest"), "engineering manifest")
    shard_path = _artifact(artifacts.get("matrix_shard"), "engineering shard")
    _require(
        _artifact(artifacts.get("auditor"), "engineering receipt auditor")
        == engineering_auditor_path,
        "engineering receipt binds another auditor",
    )
    recomputed_engineering = audit_engineering(
        master_path=source_master_path,
        preregistration_path=prereg_path,
        manifest_path=manifest_path,
        shard_path=shard_path,
    )
    _require(
        recomputed_engineering["model_summaries"]
        == engineering_receipt.get("model_summaries")
        and recomputed_engineering["matrix"] == engineering_receipt.get("matrix"),
        "engineering receipt cannot be reproduced",
    )
    manifest = _read(manifest_path)
    _require(
        manifest.get(manifest_key)
        == {"path": str(plan_path), "sha256": _sha256(plan_path)},
        "engineering manifest does not bind its plan",
    )
    models = list(manifest["models"])
    _require([str(row["id"]) for row in models] == expected_ids, "model order differs")
    ranking = _rank(models, _read(shard_path)["attempts"])
    gate = successor.get("promotion_gate")
    _require(isinstance(gate, dict), "promotion gate is absent")
    promotable = sorted(str(value) for value in gate["selected_model_must_be_one_of"])
    minimum_levels = int(gate["minimum_cleared_levels"])
    minimum_wins = int(gate["minimum_wins"])
    promotion_pass = (
        str(ranking[0]["model"]["id"]) in set(promotable)
        and int(ranking[0]["metrics"]["cleared_levels"]) >= minimum_levels
        and int(ranking[0]["metrics"]["wins"]) >= minimum_wins
    )
    outputs = master.get("outputs")
    _require(isinstance(outputs, dict), "configured successor outputs are absent")
    controller_root = Path(str(outputs["controller"])).resolve(strict=True)
    final_root = Path(str(outputs["final_blind"])).resolve()
    continuous_root = Path(str(outputs["continuous"])).resolve()
    controller_status = _read(controller_root / "controller_status.json")
    decision_path = controller_root / "promotion-decision.json"
    decision = _read(decision_path)
    _require(decision.get("ranking_rule") == EXPECTED_RANKING, "reported ranking rule differs")
    _require(decision.get("ranking") == ranking, "reported engineering ranking differs")
    _require(decision.get("selected_policy") == ranking[0]["model"], "selected policy differs")
    _require(
        decision.get("promotion_gate")
        == {
            "actual": ranking[0]["metrics"],
            "required": {
                "selected_model_must_be_one_of": promotable,
                "minimum_cleared_levels": minimum_levels,
                "minimum_wins": minimum_wins,
            },
            "passed": promotion_pass,
        },
        "reported promotion gate differs",
    )
    _require(
        decision.get("successor_formal_seed_consumption_authorized")
        is promotion_pass,
        "reported seed authorization differs",
    )
    report_path = controller_root / "final_report.json"
    report = _read(report_path)
    common = {
        "master": {"path": str(master_path), "sha256": _sha256(master_path)},
        "controller": {"path": str(controller_path), "sha256": _sha256(controller_path)},
        "promotion_decision": {
            "path": str(decision_path),
            "sha256": _sha256(decision_path),
        },
        "engineering_audit": {"path": str(audit_path), "sha256": _sha256(audit_path)},
    }
    if not promotion_pass:
        _require(
            controller_status.get("status") == "COMPLETE_NO_PROMOTION"
            and controller_status.get("formal_seed_consumption") == "NONE",
            "controller no-promotion status differs",
        )
        _require(
            report.get("status") == "COMPLETE_NO_PROMOTION"
            and report.get("master_preregistration")
            == {"path": str(master_path), "sha256": _sha256(master_path)}
            and report.get("promotion_decision") == common["promotion_decision"]
            and report.get("selected_engineering_policy") == ranking[0]["model"]
            and report.get("formal_seed_consumption") == "NONE",
            "no-promotion report differs",
        )
        _require(
            not final_root.exists() and not continuous_root.exists(),
            "formal configured successor output exists despite failed promotion",
        )
        return {
            "schema": "zuma-rl.alphazuma-55-configured-successor-independent-audit",
            "version": 1,
            "status": "PASS",
            "audited_utc": datetime.now(timezone.utc).isoformat(),
            "controller_result": "COMPLETE_NO_PROMOTION",
            "formal_seed_consumption": "NONE",
            "promotion_gate": decision["promotion_gate"],
            "artifacts": common,
            "checks": [
                "engineering_matrix_recomputed",
                "promotion_ranking_recomputed",
                "promotable_inventory_recomputed",
                "successor_seed_roots_absent",
            ],
        }

    _require(
        controller_status.get("status") == "COMPLETE"
        and controller_status.get("phase") == "COMPLETE",
        "configured successor controller is not COMPLETE",
    )
    _require(report.get("status") == "COMPLETE", "configured successor report is incomplete")
    selected = dict(ranking[0]["model"])
    final_manifest_path = controller_root / "final-blind-models-manifest.json"
    final_prereg_path = controller_root / "final-blind-preregistration.json"
    continuous_manifest_path = controller_root / "continuous-models-manifest.json"
    continuous_prereg_path = controller_root / "continuous-preregistration.json"
    promotion_reference = {"path": str(decision_path), "sha256": _sha256(decision_path)}
    final_seed = int(master["seed_registry"]["final_blind"]["first"])
    continuous_seed = int(
        master["seed_registry"]["continuous_campaign_challenge"]["first"]
    )
    devices = list(master["execution"]["devices"])
    parallel_envs = int(master["execution"]["parallel_envs_per_shard"])
    for stage, manifest_file, prereg_file, seed, attempts in (
        ("final_blind", final_manifest_path, final_prereg_path, final_seed, 8),
        ("continuous", continuous_manifest_path, continuous_prereg_path, continuous_seed, 4),
    ):
        stage_manifest = _read(manifest_file)
        stage_prereg = _read(prereg_file)
        _require(stage_manifest.get("models") == [selected], f"{stage} model differs")
        _require(stage_manifest.get("promotion_decision") == promotion_reference, f"{stage} promotion receipt differs")
        _require(stage_prereg.get("promotion_decision") == promotion_reference, f"{stage} prereg promotion receipt differs")
        binding = stage_prereg.get("configured_successor_contract")
        _require(isinstance(binding, dict), f"{stage} configured binding is absent")
        _require(
            binding.get("builder")
            == {"path": str(builder_path), "sha256": _sha256(builder_path)}
            and binding.get("stage") == stage
            and int(binding.get("base_seed", -1)) == seed
            and int(binding.get("last_seed", -1))
            == seed + LEVEL_COUNT * attempts - 1
            and int(binding.get("attempts_per_level", -1)) == attempts,
            f"{stage} configured successor binding differs",
        )
        _require(
            stage_prereg.get("evaluator")
            == {"path": str(evaluator_path), "sha256": _sha256(evaluator_path)}
            and stage_prereg.get("execution", {}).get("devices") == devices
            and int(
                stage_prereg.get("execution", {}).get(
                    "parallel_envs_per_shard", -1
                )
            )
            == parallel_envs
            and stage_prereg.get("seed_plan", {}).get(
                "all_seeds_frozen_before_policy_inference"
            )
            is True,
            f"{stage} evaluator, topology, or seed freeze differs",
        )
    final_eval = audit_evaluation(
        root=final_root.resolve(strict=True),
        preregistration_path=final_prereg_path,
        manifest_path=final_manifest_path,
        aggregate_path=(final_root / "aggregate.json").resolve(strict=True),
        expected_level_ids=level_ids,
        expected_attempts_per_level=8,
        expected_seed_base=final_seed,
        expected_model_ids=[str(selected["id"])],
    )
    continuous_eval = audit_evaluation(
        root=continuous_root.resolve(strict=True),
        preregistration_path=continuous_prereg_path,
        manifest_path=continuous_manifest_path,
        aggregate_path=(continuous_root / "aggregate.json").resolve(strict=True),
        expected_level_ids=level_ids,
        expected_attempts_per_level=4,
        expected_seed_base=continuous_seed,
        expected_model_ids=[str(selected["id"])],
    )
    selected_summary = _summary(final_eval["aggregate"], str(selected["id"]))
    capability, capability_pass = _capability(selected_summary)
    campaigns = _continuous(_rows(continuous_root), level_ids, str(selected["id"]))
    gates = report.get("success_gates")
    _require(isinstance(gates, dict), "reported gates are absent")
    _require(
        gates.get("single_policy_capability_gate", {}).get("actual") == capability
        and gates.get("single_policy_capability_gate", {}).get("passed")
        is capability_pass,
        "reported capability gate differs",
    )
    _require(report.get("continuous_campaigns") == campaigns, "reported campaigns differ")
    _require(
        gates.get("continuous_full_game_gate", {}).get("passed")
        is campaigns["passed"],
        "reported continuous gate differs",
    )
    _require(report.get("selected_single_policy") == selected, "reported final policy differs")
    return {
        "schema": "zuma-rl.alphazuma-55-configured-successor-independent-audit",
        "version": 1,
        "status": "PASS",
        "audited_utc": datetime.now(timezone.utc).isoformat(),
        "controller_result": "COMPLETE",
        "promotion_gate": decision["promotion_gate"],
        "single_policy_capability_gate": capability_pass,
        "continuous_full_game_gate": campaigns["passed"],
        "final_blind_attempts": final_eval["attempts"],
        "continuous_attempts": continuous_eval["attempts"],
        "artifacts": {
            **common,
            "final_report": {"path": str(report_path), "sha256": _sha256(report_path)},
            "final_aggregate": {
                "path": str(final_root / "aggregate.json"),
                "sha256": _sha256(final_root / "aggregate.json"),
            },
            "continuous_aggregate": {
                "path": str(continuous_root / "aggregate.json"),
                "sha256": _sha256(continuous_root / "aggregate.json"),
            },
        },
        "checks": [
            "engineering_matrix_recomputed",
            "promotion_ranking_recomputed",
            "promotable_inventory_recomputed",
            "configured_successor_contract_hashes",
            "final_blind_seed_and_attempt_matrix",
            "continuous_four_campaign_matrix",
            "single_policy_gates_recomputed",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-preregistration", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    receipt_path = args.receipt.expanduser().resolve()
    if receipt_path.exists():
        raise FileExistsError(f"refusing to overwrite audit receipt: {receipt_path}")
    result = audit(args.master_preregistration.expanduser())
    _write_exclusive(receipt_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
