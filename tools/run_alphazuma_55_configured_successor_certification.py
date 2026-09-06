"""Run a hash-frozen, intent-configured AlphaZuma 55 certification.

The controller waits for the independent engineering receipt.  It freezes one
promotion decision before creating either formal contract and consumes formal
seeds only when the selected non-baseline policy clears the frozen promotion
gate.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Iterable, Mapping

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tools import audit_alphazuma_55_training_validation as engineering_audit
from tools import build_alphazuma_55_configured_successor_eval_contract as contracts
from tools import run_alphazuma_55_successor_certification as legacy
from tools import run_overnight_v11_postprocess as evaluation
from tools.audit_alphazuma_v11_frontier_result import (
    _audit_evaluation as _strict_audit_evaluation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()
LEVEL_COUNT = 55
FINAL_ATTEMPTS = 8
CONTINUOUS_CAMPAIGNS = 4
SUPPORTED_RANKING = [
    "wins_desc",
    "level_coverage_desc",
    "total_score_desc",
    "median_winning_ticks_asc",
    "smaller_model_first_as_tie_break",
]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    return contracts.legacy._sha256(path)


def _artifact(reference: Any, name: str) -> Path:
    _require(isinstance(reference, dict), f"{name} must be an object")
    path = Path(str(reference.get("path", ""))).expanduser().resolve(strict=True)
    _require(reference.get("sha256") == _sha256(path), f"{name} hash differs")
    return path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    _require(parsed.tzinfo is not None, "deadline must contain a timezone")
    return parsed.astimezone(timezone.utc)


def _load_master(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    master = _read(path)
    _require(
        master.get("schema")
        == "zuma-rl.alphazuma-55-configured-successor-master-preregistration"
        and master.get("version") == 1
        and master.get("status") == "FROZEN_BEFORE_SUCCESSOR_FINAL_BLIND",
        "unexpected configured successor master preregistration",
    )
    levels = list(master.get("scope", {}).get("included_levels_in_adventure_order", []))
    _require(
        len(levels) == LEVEL_COUNT and len(set(levels)) == LEVEL_COUNT,
        "configured successor scope is not exactly 55 unique levels",
    )
    successor = master.get("successor")
    _require(isinstance(successor, dict), "configured successor contract is absent")
    intent_path = _artifact(successor.get("intent"), "successor intent")
    intent = _read(intent_path)
    _require(
        intent.get("status") == "FROZEN_BEFORE_ENGINEERING_RESULT"
        and intent.get("authority_boundary", {}).get("formal_seed_consumption")
        is False,
        "configured successor intent authority differs",
    )
    source_master_path = _artifact(
        successor.get("source_campaign_master"), "source campaign master"
    )
    source_master = _read(source_master_path)
    _require(
        source_master.get("scope", {}).get("included_levels_in_adventure_order")
        == levels,
        "source and configured successor level scopes differ",
    )
    engineering = successor.get("engineering_validation")
    _require(isinstance(engineering, dict), "engineering validation contract is absent")
    plan_path = _artifact(engineering.get("plan"), "engineering validation plan")
    plan = _read(plan_path)
    _require(
        plan.get("schema") == engineering.get("plan_schema")
        and plan.get("status") == "FROZEN_DURING_TARGET_TRAINING",
        "unexpected configured engineering validation plan",
    )
    _require(
        Path(str(plan["master_preregistration"]["path"])).resolve(strict=True)
        == source_master_path
        and plan["master_preregistration"]["sha256"]
        == _sha256(source_master_path),
        "engineering plan does not bind the source master",
    )
    audit_path = Path(str(engineering.get("audit_receipt", ""))).resolve()
    _require(
        audit_path == Path(str(plan["outputs"]["audit_receipt"])).resolve(),
        "engineering audit path differs from its frozen plan",
    )
    expected_ids = [str(value) for value in engineering.get("eligible_model_ids", [])]
    ranking = list(engineering.get("ranking", []))
    manifest_key = str(engineering.get("manifest_plan_binding_key", ""))
    _require(
        len(expected_ids) == 4
        and len(set(expected_ids)) == 4
        and ranking == SUPPORTED_RANKING
        and plan.get("decision_rule", {}).get("ranking") == ranking
        and manifest_key,
        "configured successor engineering inventory or ranking differs",
    )
    gate = successor.get("promotion_gate")
    _require(isinstance(gate, dict), "promotion gate is absent")
    minimum_levels = int(gate.get("minimum_cleared_levels", -1))
    minimum_wins = int(gate.get("minimum_wins", -1))
    promotable_ids = [
        str(value) for value in gate.get("selected_model_must_be_one_of", [])
    ]
    _require(
        1 <= minimum_levels <= LEVEL_COUNT
        and 1 <= minimum_wins <= LEVEL_COUNT
        and promotable_ids
        and set(promotable_ids).issubset(set(expected_ids))
        and gate.get("evaluated_before_successor_seed_consumption") is True,
        "configured successor promotion gate is invalid",
    )

    registries = master.get("seed_registry", {})
    final_registry = registries.get("final_blind", {})
    continuous_registry = registries.get("continuous_campaign_challenge", {})
    final_seed = int(final_registry.get("first", -1))
    continuous_seed = int(continuous_registry.get("first", -1))
    _require(
        int(final_registry.get("last", -1)) == final_seed + 439
        and int(continuous_registry.get("last", -1)) == continuous_seed + 219
        and final_registry.get("embargo_until_promotion_decision_is_frozen")
        is True
        and continuous_registry.get("embargo_until_successor_policy_hash_is_frozen")
        is True,
        "configured successor seed registries differ",
    )
    execution = master.get("execution", {})
    devices = list(execution.get("devices", []))
    shard_count = int(execution.get("shard_count", -1))
    parallel_envs = int(execution.get("parallel_envs_per_shard", -1))
    _require(
        shard_count == 2
        and len(devices) == shard_count
        and all(device in {"cuda:0", "cuda:1"} for device in devices)
        and parallel_envs > 0
        and int(execution.get("final_blind_attempts_per_level", -1))
        == FINAL_ATTEMPTS
        and int(execution.get("continuous_campaigns", -1))
        == CONTINUOUS_CAMPAIGNS,
        "configured successor execution topology differs",
    )
    implementation = master.get("implementation")
    _require(isinstance(implementation, dict), "implementation bindings are absent")
    expected_paths = {
        "master_builder": PROJECT_ROOT
        / "tools"
        / "build_alphazuma_55_configured_successor_master.py",
        "controller": SCRIPT_PATH,
        "watcher": PROJECT_ROOT
        / "tools"
        / "watch_alphazuma_55_configured_successor_certification.py",
        "independent_auditor": PROJECT_ROOT
        / "tools"
        / "audit_alphazuma_55_configured_successor_result.py",
        "evaluation_builder": PROJECT_ROOT
        / "tools"
        / "build_alphazuma_55_configured_successor_eval_contract.py",
        "evaluator": PROJECT_ROOT / "tools" / "evaluate_zero_shot_multilevel_v2.py",
        "evaluation_helper": PROJECT_ROOT / "tools" / "run_overnight_v11_postprocess.py",
        "matrix_auditor": PROJECT_ROOT
        / "tools"
        / "audit_alphazuma_v11_frontier_result.py",
        "engineering_auditor": PROJECT_ROOT
        / "tools"
        / "audit_alphazuma_55_training_validation.py",
        "summarizer": PROJECT_ROOT / "tools" / "summarize_multimodel_evaluation.py",
    }
    resolved_implementation: dict[str, Path] = {}
    for name, expected in expected_paths.items():
        actual = _artifact(implementation.get(name), f"implementation {name}")
        _require(actual == expected.resolve(strict=True), f"{name} path differs")
        resolved_implementation[name] = actual
    outputs = master.get("outputs")
    _require(isinstance(outputs, dict), "configured successor outputs are absent")
    resolved_outputs = {
        name: Path(str(outputs[name])).resolve()
        for name in (
            "root",
            "controller",
            "watcher",
            "final_blind",
            "continuous",
            "independent_audit_receipt",
        )
    }
    _require(
        resolved_outputs["controller"].parent == resolved_outputs["root"]
        and resolved_outputs["watcher"].parent == resolved_outputs["root"]
        and resolved_outputs["final_blind"].parent == resolved_outputs["root"]
        and resolved_outputs["continuous"].parent == resolved_outputs["root"],
        "configured successor output roots do not share the frozen root",
    )
    original_root = Path(str(master.get("original_root", ""))).resolve(strict=True)
    _require(
        original_root == Path(str(plan["original_root"])).resolve(strict=True),
        "configured successor original root differs from engineering plan",
    )
    return {
        "path": path,
        "master": master,
        "levels": levels,
        "successor": successor,
        "intent_path": intent_path,
        "source_master_path": source_master_path,
        "engineering_plan_path": plan_path,
        "engineering_plan": plan,
        "engineering_audit_path": audit_path,
        "manifest_plan_binding_key": manifest_key,
        "expected_model_ids": expected_ids,
        "ranking": ranking,
        "promotable_model_ids": promotable_ids,
        "minimum_levels": minimum_levels,
        "minimum_wins": minimum_wins,
        "final_seed": final_seed,
        "continuous_seed": continuous_seed,
        "devices": devices,
        "parallel_envs_per_shard": parallel_envs,
        "implementation": resolved_implementation,
        "outputs": resolved_outputs,
        "original_root": original_root,
        "deadline": _parse_utc(str(master["deadline_utc"])),
    }


def _model_bytes(model: Mapping[str, Any]) -> int:
    if model.get("bytes") is not None:
        size = int(model["bytes"])
    else:
        size = Path(str(model["path"])).resolve(strict=True).stat().st_size
    _require(size > 0, "engineering model byte count is invalid")
    return size


def _engineering_decision(
    *,
    models: list[dict[str, Any]],
    rows: Iterable[Mapping[str, Any]],
    minimum_levels: int,
    minimum_wins: int,
    promotable_model_ids: Iterable[str],
    ranking_rule: list[str] | None = None,
) -> dict[str, Any]:
    ranking_rule = list(ranking_rule or SUPPORTED_RANKING)
    _require(ranking_rule == SUPPORTED_RANKING, "unsupported engineering ranking")
    order = {str(model["id"]): index for index, model in enumerate(models)}
    rows_by_model: dict[str, list[Mapping[str, Any]]] = {
        model_id: [] for model_id in order
    }
    for row in rows:
        model_id = str(row["model_id"])
        _require(model_id in rows_by_model, f"unknown engineering model: {model_id}")
        rows_by_model[model_id].append(row)
    ranking: list[dict[str, Any]] = []
    for model in models:
        model_id = str(model["id"])
        model_rows = rows_by_model[model_id]
        _require(len(model_rows) == LEVEL_COUNT, f"{model_id} lacks 55 attempts")
        wins = [row for row in model_rows if row.get("outcome") == "win"]
        win_ticks = [int(row["ticks"]) for row in wins]
        metrics = {
            "wins": len(wins),
            "cleared_levels": len({str(row["level_id"]) for row in wins}),
            "total_score": sum(int(row["score"]) for row in model_rows),
            "median_winning_ticks": (
                statistics.median(win_ticks) if win_ticks else None
            ),
            "model_bytes": _model_bytes(model),
        }
        median_key = (
            float(metrics["median_winning_ticks"])
            if metrics["median_winning_ticks"] is not None
            else math.inf
        )
        ranking.append(
            {
                "model": dict(model),
                "metrics": metrics,
                "_key": (
                    -metrics["wins"],
                    -metrics["cleared_levels"],
                    -metrics["total_score"],
                    median_key,
                    metrics["model_bytes"],
                    order[model_id],
                ),
            }
        )
    ranking.sort(key=lambda row: row["_key"])
    for index, row in enumerate(ranking, start=1):
        row["rank"] = index
        del row["_key"]
    selected = ranking[0]
    promotable = set(str(value) for value in promotable_model_ids)
    passed = (
        str(selected["model"]["id"]) in promotable
        and int(selected["metrics"]["cleared_levels"]) >= minimum_levels
        and int(selected["metrics"]["wins"]) >= minimum_wins
    )
    return {
        "schema": "zuma-rl.alphazuma-55-configured-successor-promotion-decision",
        "version": 1,
        "status": "PROMOTED" if passed else "NO_PROMOTION",
        "completed_utc": _utc_now(),
        "ranking_rule": ranking_rule,
        "ranking": ranking,
        "selected_policy": selected["model"],
        "promotion_gate": {
            "actual": selected["metrics"],
            "required": {
                "selected_model_must_be_one_of": sorted(promotable),
                "minimum_cleared_levels": minimum_levels,
                "minimum_wins": minimum_wins,
            },
            "passed": passed,
        },
        "successor_formal_seed_consumption_authorized": passed,
    }


def _load_engineering_evidence(
    contract: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    audit_path = Path(contract["engineering_audit_path"])
    audit = _read(audit_path)
    _require(
        audit.get("schema") == "zuma-rl.alphazuma-55-training-validation-audit"
        and audit.get("status") == "PASS"
        and audit.get("formal_selection_authority") is False,
        "engineering audit is not a clean non-formal PASS",
    )
    artifacts = audit.get("artifacts")
    _require(isinstance(artifacts, dict), "engineering audit artifacts are absent")
    master_path = _artifact(artifacts.get("master"), "engineering audit master")
    prereg_path = _artifact(
        artifacts.get("preregistration"), "engineering preregistration"
    )
    manifest_path = _artifact(
        artifacts.get("models_manifest"), "engineering models manifest"
    )
    shard_path = _artifact(artifacts.get("matrix_shard"), "engineering matrix")
    _require(
        master_path == contract["source_master_path"],
        "engineering audit uses another source master",
    )
    recomputed = engineering_audit.audit(
        master_path=master_path,
        preregistration_path=prereg_path,
        manifest_path=manifest_path,
        shard_path=shard_path,
    )
    _require(
        recomputed["model_summaries"] == audit.get("model_summaries")
        and recomputed["matrix"] == audit.get("matrix"),
        "engineering audit cannot be independently reproduced",
    )
    manifest = _read(manifest_path)
    binding_key = str(contract["manifest_plan_binding_key"])
    _require(
        manifest.get(binding_key)
        == {
            "path": str(contract["engineering_plan_path"]),
            "sha256": _sha256(Path(contract["engineering_plan_path"])),
        },
        "engineering manifest does not bind the configured frozen plan",
    )
    models = list(manifest["models"])
    _require(
        [str(model["id"]) for model in models] == contract["expected_model_ids"],
        "engineering model order differs from the frozen inventory",
    )
    rows = list(_read(shard_path)["attempts"])
    return models, rows, audit


def _write_contract(
    *,
    contract: Mapping[str, Any],
    stage: str,
    model: dict[str, Any],
    output_root: Path,
    manifest_path: Path,
    preregistration_path: Path,
    decision_path: Path,
) -> tuple[Path, Path]:
    attempts = FINAL_ATTEMPTS if stage == "final_blind" else CONTINUOUS_CAMPAIGNS
    base_seed = (
        int(contract["final_seed"])
        if stage == "final_blind"
        else int(contract["continuous_seed"])
    )
    manifest, preregistration = contracts.build_successor_contracts(
        master_path=Path(contract["path"]),
        original_root=Path(contract["original_root"]),
        evaluator_path=Path(contract["implementation"]["evaluator"]),
        stage=stage,
        base_seed=base_seed,
        attempts_per_level=attempts,
        output_root=output_root,
        models=[model],
        parallel_envs_per_shard=int(contract["parallel_envs_per_shard"]),
    )
    decision_reference = {
        "path": str(decision_path),
        "sha256": _sha256(decision_path),
    }
    manifest["promotion_decision"] = decision_reference
    manifest["single_policy_only"] = True
    manifest["runtime_model_switching_forbidden"] = True
    preregistration["promotion_decision"] = decision_reference
    contracts.legacy._write_json_exclusive(manifest_path, manifest)
    contracts.legacy._write_json_exclusive(preregistration_path, preregistration)
    return manifest_path, preregistration_path


_all_rows = legacy._all_rows
_continuous_campaigns = legacy._continuous_campaigns
_capability = legacy._capability
_brief = legacy._brief


def _seed_consumption_label(contract: Mapping[str, Any], include_continuous: bool) -> str:
    final_last = int(contract["final_seed"]) + LEVEL_COUNT * FINAL_ATTEMPTS - 1
    if not include_continuous:
        return f"FINAL_BLIND_{contract['final_seed']}_{final_last}"
    continuous_last = (
        int(contract["continuous_seed"]) + LEVEL_COUNT * CONTINUOUS_CAMPAIGNS - 1
    )
    return (
        f"FINAL_BLIND_{contract['final_seed']}_{final_last}_AND_CONTINUOUS_"
        f"{contract['continuous_seed']}_{continuous_last}"
    )


def run(master_path: Path, original_root: Path, poll_seconds: float) -> int:
    contract = _load_master(master_path)
    original_root = original_root.resolve(strict=True)
    _require(
        original_root == contract["original_root"],
        "original retail root differs from the configured successor contract",
    )
    outputs = contract["outputs"]
    for name in ("root", "controller", "final_blind", "continuous"):
        _require(
            not outputs[name].exists(),
            f"configured successor output already exists: {name}",
        )
    outputs["root"].mkdir(parents=True)
    outputs["controller"].mkdir()
    status_path = outputs["controller"] / "controller_status.json"
    state: dict[str, Any] = {
        "schema": "zuma-rl.alphazuma-55-configured-successor-controller-status",
        "version": 1,
        "status": "RUNNING",
        "phase": "WAITING_FOR_ENGINEERING_AUDIT",
        "started_utc": _utc_now(),
        "updated_utc": _utc_now(),
        "master_preregistration": {
            "path": str(contract["path"]),
            "sha256": _sha256(Path(contract["path"])),
        },
        "formal_seed_consumption": "NONE",
        "error": None,
    }
    evaluation._replace_json(status_path, state)
    try:
        audit_path = Path(contract["engineering_audit_path"])
        failure_path = Path(
            str(contract["engineering_plan"]["target"]["expected_failure"])
        )
        while not audit_path.exists():
            if failure_path.exists():
                raise RuntimeError(
                    f"frozen configured training failed before validation: {failure_path}"
                )
            if datetime.now(timezone.utc) > contract["deadline"]:
                raise TimeoutError(
                    "configured successor engineering audit missed the campaign deadline"
                )
            state["updated_utc"] = _utc_now()
            state["waiting_for"] = str(audit_path)
            evaluation._replace_json(status_path, state)
            time.sleep(poll_seconds)

        state["phase"] = "FREEZING_PROMOTION_DECISION"
        state["updated_utc"] = _utc_now()
        evaluation._replace_json(status_path, state)
        models, rows, engineering_receipt = _load_engineering_evidence(contract)
        decision = _engineering_decision(
            models=models,
            rows=rows,
            minimum_levels=contract["minimum_levels"],
            minimum_wins=contract["minimum_wins"],
            promotable_model_ids=contract["promotable_model_ids"],
            ranking_rule=contract["ranking"],
        )
        decision["engineering_audit"] = {
            "path": str(audit_path),
            "sha256": _sha256(audit_path),
            "status": engineering_receipt["status"],
        }
        decision_path = outputs["controller"] / "promotion-decision.json"
        contracts.legacy._write_json_exclusive(decision_path, decision)
        if not decision["promotion_gate"]["passed"]:
            report = {
                "schema": "zuma-rl.alphazuma-55-configured-successor-final-report",
                "version": 1,
                "status": "COMPLETE_NO_PROMOTION",
                "completed_utc": _utc_now(),
                "master_preregistration": state["master_preregistration"],
                "promotion_decision": {
                    "path": str(decision_path),
                    "sha256": _sha256(decision_path),
                },
                "selected_engineering_policy": decision["selected_policy"],
                "formal_seed_consumption": "NONE",
                "interpretation": (
                    "The frozen engineering winner missed the preregistered "
                    "configured promotion gate; both formal seed ranges remain "
                    "unconsumed."
                ),
            }
            report_path = outputs["controller"] / "final_report.json"
            contracts.legacy._write_json_exclusive(report_path, report)
            state.update(
                {
                    "status": "COMPLETE_NO_PROMOTION",
                    "phase": "COMPLETE_NO_PROMOTION",
                    "updated_utc": _utc_now(),
                    "formal_seed_consumption": "NONE",
                    "final_report": {
                        "path": str(report_path),
                        "sha256": _sha256(report_path),
                    },
                }
            )
            evaluation._replace_json(status_path, state)
            return 0

        selected = dict(decision["selected_policy"])
        final_manifest = outputs["controller"] / "final-blind-models-manifest.json"
        final_prereg = outputs["controller"] / "final-blind-preregistration.json"
        continuous_manifest = outputs["controller"] / "continuous-models-manifest.json"
        continuous_prereg = outputs["controller"] / "continuous-preregistration.json"
        _write_contract(
            contract=contract,
            stage="final_blind",
            model=selected,
            output_root=outputs["final_blind"],
            manifest_path=final_manifest,
            preregistration_path=final_prereg,
            decision_path=decision_path,
        )
        _write_contract(
            contract=contract,
            stage="continuous",
            model=selected,
            output_root=outputs["continuous"],
            manifest_path=continuous_manifest,
            preregistration_path=continuous_prereg,
            decision_path=decision_path,
        )
        state.update(
            {
                "phase": "FINAL_BLIND_EVALUATION",
                "updated_utc": _utc_now(),
                "formal_seed_consumption": _seed_consumption_label(
                    contract, include_continuous=False
                ),
                "selected_single_policy": selected,
            }
        )
        evaluation._replace_json(status_path, state)
        evaluation.EVALUATOR = contract["implementation"]["evaluator"]
        evaluation.SUMMARIZER = contract["implementation"]["summarizer"]
        final_aggregate = evaluation._evaluate_and_summarize(
            phase="configured-successor-final-blind",
            preregistration=final_prereg,
            manifest=final_manifest,
            original_root=original_root,
            output_root=outputs["final_blind"],
            shard_count=2,
            purpose="alphazuma-55-configured-successor-final-blind",
        )
        final_audit = _strict_audit_evaluation(
            root=outputs["final_blind"],
            preregistration_path=final_prereg,
            manifest_path=final_manifest,
            aggregate_path=outputs["final_blind"] / "aggregate.json",
            expected_level_ids=contract["levels"],
            expected_attempts_per_level=FINAL_ATTEMPTS,
            expected_seed_base=int(contract["final_seed"]),
            expected_model_ids=[str(selected["id"])],
        )
        state.update(
            {
                "phase": "CONTINUOUS_CAMPAIGNS",
                "updated_utc": _utc_now(),
                "formal_seed_consumption": _seed_consumption_label(
                    contract, include_continuous=True
                ),
            }
        )
        evaluation._replace_json(status_path, state)
        continuous_aggregate = evaluation._evaluate_and_summarize(
            phase="configured-successor-continuous",
            preregistration=continuous_prereg,
            manifest=continuous_manifest,
            original_root=original_root,
            output_root=outputs["continuous"],
            shard_count=2,
            purpose="alphazuma-55-configured-successor-continuous-campaigns",
        )
        continuous_audit = _strict_audit_evaluation(
            root=outputs["continuous"],
            preregistration_path=continuous_prereg,
            manifest_path=continuous_manifest,
            aggregate_path=outputs["continuous"] / "aggregate.json",
            expected_level_ids=contract["levels"],
            expected_attempts_per_level=CONTINUOUS_CAMPAIGNS,
            expected_seed_base=int(contract["continuous_seed"]),
            expected_model_ids=[str(selected["id"])],
        )
        selected_summary = evaluation._model_summary(
            final_aggregate, str(selected["id"])
        )
        capability_actual, capability_pass = _capability(selected_summary)
        campaigns = _continuous_campaigns(
            _all_rows(outputs["continuous"]),
            level_ids=contract["levels"],
            model_id=str(selected["id"]),
        )
        report = {
            "schema": "zuma-rl.alphazuma-55-configured-successor-final-report",
            "version": 1,
            "status": "COMPLETE",
            "completed_utc": _utc_now(),
            "master_preregistration": state["master_preregistration"],
            "promotion_decision": {
                "path": str(decision_path),
                "sha256": _sha256(decision_path),
            },
            "selected_single_policy": selected,
            "final_blind": {
                "aggregate": {
                    "path": str(outputs["final_blind"] / "aggregate.json"),
                    "sha256": _sha256(outputs["final_blind"] / "aggregate.json"),
                },
                "summary": selected_summary,
            },
            "continuous_campaigns": campaigns,
            "success_gates": {
                "single_policy_capability_gate": {
                    "actual": capability_actual,
                    "required": {
                        "levels_cleared": 55,
                        "minimum_wins_per_level": 1,
                        "total_wins": 220,
                        "attempts": 440,
                    },
                    "passed": capability_pass,
                },
                "continuous_full_game_gate": {
                    "actual": {
                        "campaigns_cleared": campaigns["campaigns_cleared"],
                        "campaigns_attempted": CONTINUOUS_CAMPAIGNS,
                    },
                    "required": {"campaigns_cleared": 1},
                    "passed": campaigns["passed"],
                },
            },
            "strict_audits": {
                "final_blind": final_audit,
                "continuous": continuous_audit,
            },
            "claims": {
                "one_frozen_simulator_state_policy": True,
                "runtime_model_switching": False,
                "original_client_world_record_claim": False,
                "continuous_full_game_claim_supported": campaigns["passed"],
            },
        }
        report_path = outputs["controller"] / "final_report.json"
        contracts.legacy._write_json_exclusive(report_path, report)
        evaluation._write_new_text(
            outputs["controller"] / "FINAL_BRIEF.zh-CN.md",
            _brief(report, chinese=True),
        )
        evaluation._write_new_text(
            outputs["controller"] / "FINAL_BRIEF.en.md",
            _brief(report, chinese=False),
        )
        state.update(
            {
                "status": "COMPLETE",
                "phase": "COMPLETE",
                "updated_utc": _utc_now(),
                "final_report": {
                    "path": str(report_path),
                    "sha256": _sha256(report_path),
                },
            }
        )
        evaluation._replace_json(status_path, state)
        return 0
    except BaseException as error:
        state.update(
            {
                "status": "FAILED",
                "phase": "FAILED",
                "updated_utc": _utc_now(),
                "error": {"type": type(error).__name__, "message": str(error)},
            }
        )
        evaluation._replace_json(status_path, state)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-preregistration", required=True, type=Path)
    parser.add_argument("--original-root", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    contract = _load_master(args.master_preregistration.expanduser())
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "VALID",
                    "master": {
                        "path": str(contract["path"]),
                        "sha256": _sha256(Path(contract["path"])),
                    },
                    "levels": len(contract["levels"]),
                    "eligible_models": contract["expected_model_ids"],
                    "promotable_models": contract["promotable_model_ids"],
                    "promotion_gate": {
                        "minimum_cleared_levels": contract["minimum_levels"],
                        "minimum_wins": contract["minimum_wins"],
                    },
                    "final_blind_seed_range": [
                        contract["final_seed"],
                        contract["final_seed"] + 439,
                    ],
                    "continuous_seed_range": [
                        contract["continuous_seed"],
                        contract["continuous_seed"] + 219,
                    ],
                    "devices": contract["devices"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    return run(
        args.master_preregistration.expanduser(),
        args.original_root.expanduser(),
        args.poll_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
