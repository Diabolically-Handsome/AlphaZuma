"""Run the frozen AlphaZuma 55 selection, blind, and continuous campaign.

The controller is intentionally launched before the training deadline.  It
waits for every registered route to become terminal, freezes baseline plus
midpoint/final candidates without reading any evaluation seed, ranks them on
the paired 55 x 4 selection matrix, freezes one capability finalist, then runs
the 55 x 8 final blind matrix and four one-shot 55-level campaigns.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Iterable, Mapping

try:
    from run_overnight_v11_postprocess import (
        EVALUATOR,
        _evaluate_and_summarize,
        _model_summary,
        _parse_utc,
        _read_json,
        _replace_json,
        _sha256,
        _utc_now,
        _write_new_json,
        _write_new_text,
    )
    from audit_alphazuma_v11_frontier_result import (
        _audit_evaluation as _strict_audit_evaluation,
    )
    from build_alphazuma_55_eval_contract import build_contracts
    from run_alphazuma_speed_essence_postprocess import (
        _route_candidates as _frozen_route_candidates,
    )
except ModuleNotFoundError:  # Imported as tools.<module> in tests.
    from tools.run_overnight_v11_postprocess import (
        EVALUATOR,
        _evaluate_and_summarize,
        _model_summary,
        _parse_utc,
        _read_json,
        _replace_json,
        _sha256,
        _utc_now,
        _write_new_json,
        _write_new_text,
    )
    from tools.audit_alphazuma_v11_frontier_result import (
        _audit_evaluation as _strict_audit_evaluation,
    )
    from tools.build_alphazuma_55_eval_contract import build_contracts
    from tools.run_alphazuma_speed_essence_postprocess import (
        _route_candidates as _frozen_route_candidates,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()
EXPECTED_MASTER_SHA256 = (
    "sha256:f43e10c299332b3e4b3497ff057596775c1bf3011e1ce08fc23f832a6652f4cb"
)
EXPECTED_TRAINER_SHA256 = (
    "sha256:2886a72a7348e492f4d34f4ef9733eee1668df24e17cc896ace0094267d371ba"
)
LEVEL_COUNT = 55


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _artifact(reference: Mapping[str, Any], name: str) -> Path:
    path = Path(str(reference.get("path", ""))).expanduser().resolve(strict=True)
    _require(reference.get("sha256") == _sha256(path), f"{name} hash differs")
    return path


def _load_contract(path: Path) -> dict[str, Any]:
    post = _read_json(path)
    _require(
        post.get("schema") == "zuma-rl.alphazuma-55-postprocess-preregistration",
        "unexpected postprocess preregistration schema",
    )
    _require(
        post.get("version") == 1
        and post.get("status") == "FROZEN_BEFORE_SELECTION",
        "postprocess preregistration is not frozen",
    )
    master_path = _artifact(
        _mapping(post.get("master_preregistration"), "master_preregistration"),
        "master preregistration",
    )
    _require(_sha256(master_path) == EXPECTED_MASTER_SHA256, "master hash differs")
    master = _read_json(master_path)
    _require(
        master.get("schema")
        == "zuma-rl.alphazuma-55-weekend-master-preregistration",
        "unexpected master schema",
    )
    level_ids = list(master["scope"]["included_levels_in_adventure_order"])
    _require(len(level_ids) == LEVEL_COUNT and len(set(level_ids)) == LEVEL_COUNT, "55-level scope differs")
    rules = _mapping(post.get("selection_rules"), "selection_rules")
    _require(
        rules.get("capability_ranking")
        == master["candidate_and_selection_rules"]["selection_ranking_lexicographic"],
        "capability ranking differs from the master",
    )
    _require(
        rules.get("speed_ranking")
        == [
            "number of levels with at least one win out of four",
            "total wins",
            "paired median winning ticks",
            "model sha256 ascending as deterministic tie break",
        ],
        "speed ranking differs",
    )
    _require(rules.get("single_policy_finalist") == "rank 1 under capability_ranking", "finalist rule differs")
    _require(rules.get("final_blind_may_not_change_finalist") is True, "final blind may change finalist")
    embargo = _mapping(post.get("seed_embargo"), "seed_embargo")
    _require(embargo.get("selection") == [1_600_000_000, 1_600_000_219], "selection seed embargo differs")
    _require(embargo.get("final_blind") == [1_700_000_000, 1_700_000_439], "final seed embargo differs")
    _require(embargo.get("continuous") == [1_800_000_000, 1_800_000_219], "continuous seed embargo differs")
    evaluation_execution = _mapping(
        post.get("evaluation_execution"), "evaluation_execution"
    )
    _require(
        evaluation_execution.get("shard_count") == 2
        and evaluation_execution.get("devices") == ["cuda:0", "cuda:1"],
        "evaluation shard topology differs",
    )
    parallel_envs_per_shard = int(
        evaluation_execution.get("parallel_envs_per_shard", 0)
    )
    _require(
        1 <= parallel_envs_per_shard <= 256,
        "evaluation parallel environment count is outside [1, 256]",
    )
    decision_path = _artifact(
        _mapping(
            evaluation_execution.get("decision_receipt"),
            "evaluation_execution.decision_receipt",
        ),
        "evaluation parallelism decision",
    )
    _require(
        decision_path
        == (
            PROJECT_ROOT
            / "diagnostics"
            / "alphazuma-55-evaluation-parallelism-decision-s90081503-v1.json"
        ).resolve(strict=True),
        "evaluation parallelism decision path differs",
    )
    parallelism_decision = _read_json(decision_path)
    _require(
        parallelism_decision.get("status") == "PASS"
        and parallelism_decision.get("formal_seed_embargo_intact") is True
        and parallelism_decision.get("decision", {}).get(
            "formal_parallel_envs_per_shard"
        )
        == parallel_envs_per_shard,
        "evaluation parallelism decision does not authorize this width",
    )

    implementation = _mapping(post.get("implementation"), "implementation")
    expected_implementation = {
        "controller": SCRIPT_PATH,
        "independent_auditor": PROJECT_ROOT / "tools" / "audit_alphazuma_55_result.py",
        "evaluator": EVALUATOR,
        "summarizer": PROJECT_ROOT / "tools" / "summarize_multimodel_evaluation.py",
        "matrix_auditor": PROJECT_ROOT / "tools" / "audit_alphazuma_v11_frontier_result.py",
        "evaluation_builder": PROJECT_ROOT / "tools" / "build_alphazuma_55_eval_contract.py",
        "postprocess_helper": PROJECT_ROOT / "tools" / "run_overnight_v11_postprocess.py",
        "candidate_helper": PROJECT_ROOT / "tools" / "run_alphazuma_speed_essence_postprocess.py",
        "gpu_power_guard": PROJECT_ROOT / "tools" / "manage_alphazuma_55_power.ps1",
        "windows_power_plan_guard": PROJECT_ROOT / "tools" / "manage_alphazuma_55_power_plan.ps1",
    }
    for name, expected in expected_implementation.items():
        actual = _artifact(_mapping(implementation.get(name), f"implementation.{name}"), name)
        _require(actual == expected.resolve(strict=True), f"{name} path differs")

    migrations: list[dict[str, Any]] = []
    raw_migrations = post.get("migrations")
    _require(isinstance(raw_migrations, list) and len(raw_migrations) == 3, "expected three migrations")
    for index, raw in enumerate(raw_migrations):
        reference = _mapping(raw, f"migrations[{index}]")
        receipt_path = _artifact(
            _mapping(reference.get("receipt"), f"migrations[{index}].receipt"),
            f"migration receipt {index}",
        )
        receipt = _read_json(receipt_path)
        _require(
            receipt.get("schema") == "zuma-rl.alphazuma-55-model-migration"
            and receipt.get("status") == "PASS",
            f"migration {index} is not PASS",
        )
        migrated = _mapping(receipt.get("migrated_model"), "migrated_model")
        model_path = Path(str(migrated.get("path", ""))).resolve(strict=True)
        _require(migrated.get("sha256") == _sha256(model_path), "migrated model changed")
        _require(migrated.get("observation_shape") == [22833], "migrated observation space differs")
        _require(migrated.get("action_nvec") == [4, 180], "migrated action space differs")
        migrations.append(
            {
                "id": str(reference["id"]),
                "role": str(reference["role"]),
                "training_steps": int(migrated["num_timesteps"]),
                "path": str(model_path),
                "sha256": str(migrated["sha256"]),
                "source": f"migrated_{reference['role']}",
                "receipt": {"path": str(receipt_path), "sha256": _sha256(receipt_path)},
            }
        )
    _require(len({row["id"] for row in migrations}) == 3, "migration ids are duplicated")

    routes: list[dict[str, Any]] = []
    raw_routes = post.get("training_routes")
    _require(isinstance(raw_routes, list) and 1 <= len(raw_routes) <= 6, "route count is outside [1, 6]")
    for index, raw in enumerate(raw_routes):
        reference = _mapping(raw, f"training_routes[{index}]")
        prereg_path = _artifact(reference, f"training route {index}")
        spec = _read_json(prereg_path)
        _require(
            spec.get("schema") == "zuma-rl.overnight-multilevel-preregistration"
            and spec.get("status") == "FROZEN_BEFORE_TRAINING",
            f"route {index} is not frozen",
        )
        _require(spec.get("campaign_id") == master["campaign_id"], "route campaign differs")
        _require(spec.get("levels") and len(spec["levels"]) == LEVEL_COUNT, "route level inventory differs")
        _require([row["id"] for row in spec["levels"]] == level_ids, "route level order differs")
        route_implementation = _mapping(spec.get("implementation"), "route implementation")
        for name, implementation_reference in route_implementation.items():
            _artifact(
                _mapping(implementation_reference, f"route implementation {name}"),
                f"route implementation {name}",
            )
        trainer = _mapping(spec.get("trainer"), "route trainer")
        _require(trainer.get("sha256") == EXPECTED_TRAINER_SHA256, "route trainer hash differs")
        raw_runs = spec.get("runs")
        _require(isinstance(raw_runs, list) and len(raw_runs) == 1, "each route preregistration must contain one run")
        run = dict(_mapping(raw_runs[0], "route run"))
        run.update(
            {
                "family": str(spec["route_family"]),
                "preregistration": {"path": str(prereg_path), "sha256": _sha256(prereg_path)},
                "initial_model": _mapping(spec.get("initial_model"), "initial_model"),
            }
        )
        routes.append(run)
    _require(len({str(run["id"]) for run in routes}) == len(routes), "route ids are duplicated")
    _require(len({str(run["run_dir"]) for run in routes}) == len(routes), "route directories are duplicated")

    outputs = _mapping(post.get("outputs"), "outputs")
    output_paths = {name: Path(str(value)).resolve() for name, value in outputs.items()}
    _require(
        set(output_paths) == {"controller", "selection", "final_blind", "continuous"},
        "formal output roots differ",
    )
    _require(len(set(output_paths.values())) == 4, "formal output roots overlap")
    return {
        "post": post,
        "post_path": path,
        "master": master,
        "master_path": master_path,
        "level_ids": level_ids,
        "migrations": migrations,
        "routes": routes,
        "outputs": output_paths,
        "parallel_envs_per_shard": parallel_envs_per_shard,
    }


def _terminal_snapshot(routes: Iterable[Mapping[str, Any]]) -> tuple[bool, dict[str, Any]]:
    terminal = True
    snapshots: dict[str, Any] = {}
    for run in routes:
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


def _freeze_candidates(contract: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = [{key: value for key, value in model.items() if key != "receipt"} for model in contract["migrations"]]
    receipts: list[dict[str, Any]] = []
    for run in contract["routes"]:
        additions, receipt = _frozen_route_candidates(run)
        candidates.extend(additions)
        receipts.append(receipt)
    _require(len({str(row["id"]) for row in candidates}) == len(candidates), "candidate ids are duplicated")
    return candidates, receipts


def _selection_decision(
    aggregate: Mapping[str, Any], candidates: list[dict[str, Any]]
) -> dict[str, Any]:
    robust: list[dict[str, Any]] = []
    speed: list[dict[str, Any]] = []
    for model in candidates:
        summary = _model_summary(aggregate, str(model["id"]))
        levels = list(summary["level_summaries"])
        wins = [int(row["wins"]) for row in levels]
        median = summary.get("median_win_ticks")
        metrics = {
            "levels_cleared": sum(value > 0 for value in wins),
            "minimum_wins_per_level": min(wins),
            "total_wins": sum(wins),
            "median_winning_ticks": median,
        }
        robust.append(
            {
                "model": model,
                "metrics": metrics,
                "_key": (
                    -metrics["levels_cleared"],
                    -metrics["minimum_wins_per_level"],
                    -metrics["total_wins"],
                    float(median) if median is not None else float("inf"),
                    str(model["sha256"]),
                ),
            }
        )
        speed.append(
            {
                "model": model,
                "metrics": metrics,
                "_key": (
                    -metrics["levels_cleared"],
                    -metrics["total_wins"],
                    float(median) if median is not None else float("inf"),
                    str(model["sha256"]),
                ),
            }
        )

    def ranked(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result = sorted(rows, key=lambda row: row["_key"])
        for index, row in enumerate(result, start=1):
            row["rank"] = index
            del row["_key"]
        return result

    robust = ranked(robust)
    speed = ranked(speed)
    return {
        "schema": "zuma-rl.alphazuma-55-selection-decision",
        "version": 1,
        "status": "COMPLETE",
        "completed_utc": _utc_now(),
        "robust_rule": [
            "levels cleared descending",
            "minimum wins per level descending",
            "total wins descending",
            "median winning ticks ascending",
            "model sha256 ascending",
        ],
        "speed_rule": [
            "levels cleared descending",
            "total wins descending",
            "median winning ticks ascending",
            "model sha256 ascending",
        ],
        "selected_single_policy": robust[0]["model"],
        "speed_report_model": speed[0]["model"],
        "robust_ranking": robust,
        "speed_ranking": speed,
    }


def _deduplicated_manifest_models(
    candidates: Iterable[Mapping[str, Any]], role_ids: Mapping[str, str]
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    by_id = {str(row["id"]): dict(row) for row in candidates}
    retained_by_hash: dict[str, str] = {}
    models: list[dict[str, Any]] = []
    roles: dict[str, str] = {}
    for role, source_id in role_ids.items():
        model = dict(by_id[source_id])
        digest = str(model["sha256"])
        if digest not in retained_by_hash:
            retained_by_hash[digest] = str(model["id"])
            models.append(model)
        roles[role] = retained_by_hash[digest]
    return models, roles


def _write_evaluation_contracts(
    *,
    contract: Mapping[str, Any],
    stage: str,
    models: list[dict[str, Any]],
    output_root: Path,
    controller_root: Path,
    manifest_name: str,
    preregistration_name: str,
    extra_manifest: Mapping[str, Any],
) -> tuple[Path, Path]:
    manifest, prereg = build_contracts(
        master_path=contract["master_path"],
        original_root=contract["original_root"],
        evaluator_path=EVALUATOR,
        stage=stage,
        output_root=output_root,
        models=models,
        parallel_envs_per_shard=int(contract["parallel_envs_per_shard"]),
    )
    manifest.update(extra_manifest)
    manifest_path = controller_root / manifest_name
    prereg_path = controller_root / preregistration_name
    _write_new_json(manifest_path, manifest)
    _write_new_json(prereg_path, prereg)
    return manifest_path, prereg_path


def _all_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(2):
        rows.extend(_read_json(root / f"matrix-shard-{index:02d}-of-02.json")["attempts"])
    return rows


def _summary_metrics(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: summary.get(key)
        for key in (
            "levels",
            "levels_cleared",
            "attempts",
            "wins",
            "losses",
            "truncations",
            "win_rate",
            "median_win_ticks",
            "median_win_seconds",
            "best_attempt",
            "level_summaries",
        )
    }


def _continuous_campaigns(
    rows: Iterable[Mapping[str, Any]], *, level_ids: list[str], model_id: str
) -> dict[str, Any]:
    by_campaign: dict[int, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        _require(str(row["model_id"]) == model_id, "continuous manifest contains another model")
        index = int(row["attempt_index"])
        level_id = str(row["level_id"])
        _require(level_id not in by_campaign[index], "continuous campaign repeats a level")
        by_campaign[index][level_id] = row
    _require(set(by_campaign) == set(range(4)), "continuous campaign ids differ")
    campaigns: list[dict[str, Any]] = []
    for index in range(4):
        campaign = by_campaign[index]
        _require(set(campaign) == set(level_ids), "continuous campaign level inventory differs")
        ordered = [campaign[level_id] for level_id in level_ids]
        wins = sum(row.get("outcome") == "win" for row in ordered)
        campaigns.append(
            {
                "campaign_index": index,
                "levels_attempted": LEVEL_COUNT,
                "wins": wins,
                "cleared_all_55": wins == LEVEL_COUNT,
                "first_failure_level": next(
                    (str(row["level_id"]) for row in ordered if row.get("outcome") != "win"),
                    None,
                ),
                "seeds": [int(row["seed"]) for row in ordered],
            }
        )
    return {
        "campaigns": campaigns,
        "passed": any(row["cleared_all_55"] for row in campaigns),
        "campaigns_cleared": sum(row["cleared_all_55"] for row in campaigns),
    }


def _brief(report: Mapping[str, Any], *, chinese: bool) -> str:
    gate = report["success_gates"]["single_policy_capability_gate"]
    continuous = report["success_gates"]["continuous_full_game_gate"]
    selected = report["selected_single_policy"]
    if chinese:
        return (
            "# AlphaZuma 55 最终报告\n\n"
            f"冻结单策略：`{selected['id']}`（`{selected['sha256']}`）。\n\n"
            f"最终盲测覆盖 {gate['actual']['levels_cleared']}/55，"
            f"总胜局 {gate['actual']['total_wins']}/440，"
            f"能力 Gate：{'PASS' if gate['passed'] else 'FAIL'}。\n\n"
            f"四套连续挑战中完整通关 {continuous['actual']['campaigns_cleared']}/4，"
            f"连续全流程 Gate：{'PASS' if continuous['passed'] else 'FAIL'}。\n"
        )
    return (
        "# AlphaZuma 55 final report\n\n"
        f"Frozen single policy: `{selected['id']}` (`{selected['sha256']}`).\n\n"
        f"Final-blind coverage {gate['actual']['levels_cleared']}/55, "
        f"wins {gate['actual']['total_wins']}/440, "
        f"capability gate: {'PASS' if gate['passed'] else 'FAIL'}.\n\n"
        f"Complete continuous campaigns {continuous['actual']['campaigns_cleared']}/4, "
        f"continuous gate: {'PASS' if continuous['passed'] else 'FAIL'}.\n"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prereg_path = args.preregistration.expanduser().resolve(strict=True)
    contract = _load_contract(prereg_path)
    original_root = args.original_root.expanduser().resolve(strict=True)
    _require(
        Path(str(contract["post"].get("original_root", ""))).resolve(strict=True)
        == original_root,
        "original retail root differs from the frozen postprocess contract",
    )
    contract["original_root"] = original_root
    controller_root = contract["outputs"]["controller"]
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "VALID",
                    "preregistration": {"path": str(prereg_path), "sha256": _sha256(prereg_path)},
                    "routes": len(contract["routes"]),
                    "baseline_candidates": len(contract["migrations"]),
                    "levels": len(contract["level_ids"]),
                    "selection_seed_range": [1_600_000_000, 1_600_000_219],
                    "final_blind_seed_range": [1_700_000_000, 1_700_000_439],
                    "continuous_seed_range": [1_800_000_000, 1_800_000_219],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    for output in contract["outputs"].values():
        if output.exists():
            raise FileExistsError(f"refusing to reuse formal output root: {output}")
    controller_root.mkdir(parents=True)
    status_path = controller_root / "controller_status.json"
    state: dict[str, Any] = {
        "schema": "zuma-rl.alphazuma-55-postprocess-controller-status",
        "version": 1,
        "status": "RUNNING",
        "phase": "WAITING_FOR_TRAINING",
        "started_utc": _utc_now(),
        "updated_utc": _utc_now(),
        "preregistration": {"path": str(prereg_path), "sha256": _sha256(prereg_path)},
        "training": {},
        "error": None,
    }
    _replace_json(status_path, state)
    try:
        latest_terminal = _parse_utc(contract["master"]["schedule"]["formal_training_stop_utc"]) + timedelta(minutes=20)
        while True:
            terminal, snapshot = _terminal_snapshot(contract["routes"])
            state["training"] = snapshot
            state["updated_utc"] = _utc_now()
            _replace_json(status_path, state)
            if terminal:
                break
            if datetime.now(timezone.utc) > latest_terminal:
                raise TimeoutError("registered routes did not become terminal within twenty minutes")
            time.sleep(args.poll_seconds)

        state["phase"] = "FREEZING_CANDIDATES"
        state["updated_utc"] = _utc_now()
        _replace_json(status_path, state)
        candidates, route_receipts = _freeze_candidates(contract)
        selection_manifest, selection_prereg = _write_evaluation_contracts(
            contract=contract,
            stage="selection",
            models=candidates,
            output_root=contract["outputs"]["selection"],
            controller_root=controller_root,
            manifest_name="selection-models-manifest.json",
            preregistration_name="selection-preregistration.json",
            extra_manifest={
                "candidate_rule": contract["master"]["candidate_and_selection_rules"],
                "migration_receipts": [row["receipt"] for row in contract["migrations"]],
                "route_receipts": route_receipts,
            },
        )
        state["phase"] = "SELECTION_EVALUATION"
        state["updated_utc"] = _utc_now()
        _replace_json(status_path, state)
        selection_aggregate = _evaluate_and_summarize(
            phase="selection",
            preregistration=selection_prereg,
            manifest=selection_manifest,
            original_root=contract["original_root"],
            output_root=contract["outputs"]["selection"],
            shard_count=2,
            purpose="alphazuma-55-selection",
        )
        selection_audit = _strict_audit_evaluation(
            root=contract["outputs"]["selection"],
            preregistration_path=selection_prereg,
            manifest_path=selection_manifest,
            aggregate_path=contract["outputs"]["selection"] / "aggregate.json",
            expected_level_ids=contract["level_ids"],
            expected_attempts_per_level=4,
            expected_seed_base=1_600_000_000,
            expected_model_ids=[str(row["id"]) for row in candidates],
        )
        decision = _selection_decision(selection_aggregate, candidates)
        decision["strict_matrix_audit"] = selection_audit
        decision_path = controller_root / "selection-decision.json"
        _write_new_json(decision_path, decision)

        baseline_roles = {str(row["role"]): str(row["id"]) for row in contract["migrations"]}
        role_ids = {
            **baseline_roles,
            "selected_single_policy": str(decision["selected_single_policy"]["id"]),
            "speed_report_model": str(decision["speed_report_model"]["id"]),
        }
        final_models, final_roles = _deduplicated_manifest_models(candidates, role_ids)
        final_manifest, final_prereg = _write_evaluation_contracts(
            contract=contract,
            stage="final_blind",
            models=final_models,
            output_root=contract["outputs"]["final_blind"],
            controller_root=controller_root,
            manifest_name="final-blind-models-manifest.json",
            preregistration_name="final-blind-preregistration.json",
            extra_manifest={
                "roles": final_roles,
                "deduplicated_by_model_sha256": True,
                "selection_decision": {"path": str(decision_path), "sha256": _sha256(decision_path)},
            },
        )
        state["phase"] = "FINAL_BLIND_EVALUATION"
        state["updated_utc"] = _utc_now()
        _replace_json(status_path, state)
        final_aggregate = _evaluate_and_summarize(
            phase="final-blind",
            preregistration=final_prereg,
            manifest=final_manifest,
            original_root=contract["original_root"],
            output_root=contract["outputs"]["final_blind"],
            shard_count=2,
            purpose="alphazuma-55-final-blind",
        )
        final_audit = _strict_audit_evaluation(
            root=contract["outputs"]["final_blind"],
            preregistration_path=final_prereg,
            manifest_path=final_manifest,
            aggregate_path=contract["outputs"]["final_blind"] / "aggregate.json",
            expected_level_ids=contract["level_ids"],
            expected_attempts_per_level=8,
            expected_seed_base=1_700_000_000,
            expected_model_ids=[str(row["id"]) for row in final_models],
        )

        selected = dict(decision["selected_single_policy"])
        continuous_manifest, continuous_prereg = _write_evaluation_contracts(
            contract=contract,
            stage="continuous",
            models=[selected],
            output_root=contract["outputs"]["continuous"],
            controller_root=controller_root,
            manifest_name="continuous-models-manifest.json",
            preregistration_name="continuous-preregistration.json",
            extra_manifest={
                "single_policy_only": True,
                "runtime_model_switching_forbidden": True,
                "selection_decision": {"path": str(decision_path), "sha256": _sha256(decision_path)},
            },
        )
        state["phase"] = "CONTINUOUS_CAMPAIGNS"
        state["updated_utc"] = _utc_now()
        _replace_json(status_path, state)
        continuous_aggregate = _evaluate_and_summarize(
            phase="continuous",
            preregistration=continuous_prereg,
            manifest=continuous_manifest,
            original_root=contract["original_root"],
            output_root=contract["outputs"]["continuous"],
            shard_count=2,
            purpose="alphazuma-55-continuous-campaigns",
        )
        continuous_audit = _strict_audit_evaluation(
            root=contract["outputs"]["continuous"],
            preregistration_path=continuous_prereg,
            manifest_path=continuous_manifest,
            aggregate_path=contract["outputs"]["continuous"] / "aggregate.json",
            expected_level_ids=contract["level_ids"],
            expected_attempts_per_level=4,
            expected_seed_base=1_800_000_000,
            expected_model_ids=[str(selected["id"])],
        )
        campaigns = _continuous_campaigns(
            _all_rows(contract["outputs"]["continuous"]),
            level_ids=contract["level_ids"],
            model_id=str(selected["id"]),
        )

        final_role_metrics = {
            role: _summary_metrics(_model_summary(final_aggregate, model_id))
            for role, model_id in final_roles.items()
        }
        selected_summary = _model_summary(final_aggregate, final_roles["selected_single_policy"])
        level_wins = [int(row["wins"]) for row in selected_summary["level_summaries"]]
        capability_actual = {
            "levels_cleared": int(selected_summary["levels_cleared"]),
            "minimum_wins_per_level": min(level_wins),
            "total_wins": int(selected_summary["wins"]),
            "attempts": int(selected_summary["attempts"]),
        }
        capability_pass = (
            capability_actual["levels_cleared"] == 55
            and capability_actual["minimum_wins_per_level"] >= 1
            and capability_actual["total_wins"] >= 220
            and capability_actual["attempts"] == 440
        )
        report = {
            "schema": "zuma-rl.alphazuma-55-final-report",
            "version": 1,
            "status": "COMPLETE",
            "completed_utc": _utc_now(),
            "preregistration": {"path": str(prereg_path), "sha256": _sha256(prereg_path)},
            "selected_single_policy": selected,
            "speed_report_model": decision["speed_report_model"],
            "selection": {
                "decision": {"path": str(decision_path), "sha256": _sha256(decision_path)},
                "candidate_count": len(candidates),
                "strict_matrix_audit": selection_audit,
            },
            "final_blind": final_role_metrics,
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
                        "campaigns_attempted": 4,
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
                "simulator_state_policy": True,
                "original_client_world_record_claim": False,
                "continuous_full_game_claim_supported": campaigns["passed"],
            },
        }
        report_path = controller_root / "final_report.json"
        brief_zh = controller_root / "FINAL_BRIEF.zh-CN.md"
        brief_en = controller_root / "FINAL_BRIEF.en.md"
        _write_new_json(report_path, report)
        _write_new_text(brief_zh, _brief(report, chinese=True))
        _write_new_text(brief_en, _brief(report, chinese=False))
        state.update(
            {
                "status": "COMPLETE",
                "phase": "COMPLETE",
                "updated_utc": _utc_now(),
                "final_report": {"path": str(report_path), "sha256": _sha256(report_path)},
                "selected_single_policy": selected,
            }
        )
        _replace_json(status_path, state)
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
        _replace_json(status_path, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
