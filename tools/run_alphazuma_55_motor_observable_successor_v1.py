"""Conditionally certify one retry-audited motor-observable full55 policy."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tools import (
    audit_alphazuma_55_motor_observable_postprocess_v2 as engineering_auditor,
)
from tools import run_alphazuma_55_motor_observable_postprocess_v2 as engineering
from tools import run_alphazuma_55_successor_certification as successor_helpers
from tools import run_overnight_v11_postprocess as evaluation
from tools.audit_alphazuma_v11_frontier_result import (
    _audit_evaluation as _strict_audit_evaluation,
)
from zuma_rl.alphazuma_55 import INCLUDED_LEVELS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()
MASTER_SCHEMA = "zuma-rl.alphazuma-55-motor-observable-successor-master"
LEVEL_COUNT = 55
FINAL_ATTEMPTS = 8
CONTINUOUS_CAMPAIGNS = 4
FINAL_SEED_BASE = 3_300_000_000
CONTINUOUS_SEED_BASE = 3_400_000_000


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    return engineering._sha256(path)


def _artifact(reference: Any, label: str) -> Path:
    _require(isinstance(reference, dict), f"{label} reference must be an object")
    path = Path(str(reference.get("path", ""))).resolve(strict=True)
    _require(reference.get("sha256") == _sha256(path), f"{label} hash differs")
    return path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    _require(parsed.tzinfo is not None, "deadline timezone is absent")
    return parsed.astimezone(timezone.utc)


def _write_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists():
        temporary.unlink()
        raise FileExistsError(f"refusing to overwrite frozen artifact: {path}")
    os.replace(temporary, path)


def _replace_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _validate_registry(
    *, registry_path: Path, master_path: Path
) -> dict[str, Any]:
    registry_path = registry_path.resolve(strict=True)
    registry = _read(registry_path)
    _require(
        registry.get("schema")
        == "zuma-rl.alphazuma-55-weekend-formal-seed-registry"
        and registry.get("version") == 1
        and registry.get("status")
        == "FROZEN_BEFORE_ANY_SUCCESSOR_FORMAL_INFERENCE",
        "unexpected active formal seed registry",
    )
    implementation = registry.get("implementation", {})
    if implementation:
        _artifact(implementation.get("builder"), "formal registry builder")
    all_ranges: list[tuple[int, int, str, str]] = []
    matched = None
    for campaign in registry.get("campaigns", []):
        campaign_id = str(campaign.get("id", ""))
        campaign_master = _artifact(
            campaign.get("master"), f"registry campaign master {campaign_id}"
        )
        ranges = list(campaign.get("ranges", []))
        for row in ranges:
            first = int(row.get("first", -1))
            last = int(row.get("last", -1))
            _require(0 <= first <= last <= 0xFFFFFFFF, "formal seed range is invalid")
            all_ranges.append((first, last, campaign_id, str(row.get("stage"))))
        if campaign_master == master_path:
            matched = campaign
    ordered = sorted(all_ranges)
    for left, right in zip(ordered, ordered[1:], strict=False):
        _require(left[1] < right[0], f"formal seed ranges overlap: {left} {right}")
    _require(matched is not None, "active registry does not bind this master")
    expected = [
        {
            "stage": "final_blind",
            "first": FINAL_SEED_BASE,
            "last": FINAL_SEED_BASE + LEVEL_COUNT * FINAL_ATTEMPTS - 1,
        },
        {
            "stage": "continuous_campaign_challenge",
            "first": CONTINUOUS_SEED_BASE,
            "last": (
                CONTINUOUS_SEED_BASE
                + LEVEL_COUNT * CONTINUOUS_CAMPAIGNS
                - 1
            ),
        },
    ]
    _require(matched.get("ranges") == expected, "active registry ranges differ")
    return {
        "path": registry_path,
        "sha256": _sha256(registry_path),
        "registry": registry,
    }


def _load_master(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    master = _read(path)
    _require(
        master.get("schema") == MASTER_SCHEMA
        and master.get("version") == 1
        and master.get("status") == "FROZEN_BEFORE_ENGINEERING_RESULT",
        "unexpected motor successor master",
    )
    levels = list(master.get("scope", {}).get("included_levels", []))
    _require(levels == list(INCLUDED_LEVELS), "motor successor level scope differs")
    engineering_contract = master.get("engineering", {})
    plan_path = _artifact(engineering_contract.get("plan"), "engineering plan")
    plan = engineering.validate_plan(
        plan_path, str(engineering_contract["plan"]["sha256"])
    )
    audit_path = Path(str(engineering_contract.get("independent_audit"))).resolve()
    _require(
        audit_path
        == Path(str(plan["outputs"]["independent_audit_receipt"])).resolve(),
        "engineering independent audit path differs",
    )
    gate = engineering_contract.get("promotion_gate", {})
    _require(
        int(gate.get("minimum_wins", -1)) == 35
        and int(gate.get("minimum_cleared_levels", -1)) == 35
        and gate.get("must_be_promotable_motor_checkpoint") is True
        and gate.get("evaluated_before_formal_seed_consumption") is True,
        "motor successor promotion gate changed",
    )
    registries = master.get("seed_registry", {})
    final = registries.get("final_blind", {})
    continuous = registries.get("continuous_campaign_challenge", {})
    _require(
        int(final.get("first", -1)) == FINAL_SEED_BASE
        and int(final.get("last", -1))
        == FINAL_SEED_BASE + LEVEL_COUNT * FINAL_ATTEMPTS - 1
        and int(continuous.get("first", -1)) == CONTINUOUS_SEED_BASE
        and int(continuous.get("last", -1))
        == CONTINUOUS_SEED_BASE + LEVEL_COUNT * CONTINUOUS_CAMPAIGNS - 1
        and final.get("embargo_until_engineering_promotion_is_frozen") is True
        and continuous.get("embargo_until_policy_hash_is_frozen") is True,
        "motor successor formal seed contract changed",
    )
    execution = master.get("execution", {})
    devices = list(execution.get("devices", []))
    _require(
        devices == ["cuda:0", "cuda:1"]
        and int(execution.get("shard_count", -1)) == 2
        and int(execution.get("parallel_envs_per_shard", -1)) == 24
        and int(execution.get("final_blind_attempts_per_level", -1))
        == FINAL_ATTEMPTS
        and int(execution.get("continuous_campaigns", -1))
        == CONTINUOUS_CAMPAIGNS,
        "motor successor execution topology changed",
    )
    expected_implementation = {
        "master_builder": PROJECT_ROOT
        / "tools/build_alphazuma_55_motor_observable_successor_master_v1.py",
        "controller": SCRIPT_PATH,
        "independent_auditor": PROJECT_ROOT
        / "tools/audit_alphazuma_55_motor_observable_successor_v1.py",
        "evaluator": PROJECT_ROOT
        / "tools/evaluate_alphazuma_55_motor_observable.py",
        "evaluation_helper": PROJECT_ROOT / "tools/run_overnight_v11_postprocess.py",
        "matrix_auditor": PROJECT_ROOT
        / "tools/audit_alphazuma_v11_frontier_result.py",
        "summarizer": PROJECT_ROOT / "tools/summarize_multimodel_evaluation.py",
        "engineering_controller": PROJECT_ROOT
        / "tools/run_alphazuma_55_motor_observable_postprocess_v2.py",
        "engineering_auditor": PROJECT_ROOT
        / "tools/audit_alphazuma_55_motor_observable_postprocess_v2.py",
    }
    implementation: dict[str, Path] = {}
    for name, expected in expected_implementation.items():
        actual = _artifact(master.get("implementation", {}).get(name), name)
        _require(actual == expected.resolve(strict=True), f"{name} path differs")
        implementation[name] = actual
    template_path = _artifact(
        master.get("environment_template"), "environment template"
    )
    template = _read(template_path)
    _require(
        [str(row["id"]) for row in template.get("levels", [])] == levels,
        "environment template level order differs",
    )
    outputs_value = master.get("outputs", {})
    outputs = {
        name: Path(str(outputs_value[name])).resolve()
        for name in (
            "root",
            "controller",
            "final_blind",
            "continuous",
            "independent_audit_receipt",
        )
    }
    _require(
        outputs["controller"].parent == outputs["root"]
        and outputs["final_blind"].parent == outputs["root"]
        and outputs["continuous"].parent == outputs["root"],
        "motor successor output topology differs",
    )
    original_root = Path(str(master.get("original_root", ""))).resolve(strict=True)
    _require(
        original_root == Path(str(plan["original_root"])).resolve(strict=True),
        "motor successor original root differs",
    )
    active_registry_path = Path(
        str(master.get("formal_seed_registry", {}).get("active_registry_path", ""))
    )
    registry = _validate_registry(
        registry_path=active_registry_path, master_path=path
    )
    return {
        "path": path,
        "master": master,
        "levels": levels,
        "engineering_plan_path": plan_path,
        "engineering_plan": plan,
        "engineering_audit_path": audit_path,
        "template_path": template_path,
        "template": template,
        "implementation": implementation,
        "outputs": outputs,
        "original_root": original_root,
        "devices": devices,
        "parallel_envs_per_shard": int(execution["parallel_envs_per_shard"]),
        "deadline": _parse_utc(str(master["deadline_utc"])),
        "registry": registry,
    }


def _engineering_result(contract: Mapping[str, Any]) -> dict[str, Any]:
    audit_path = Path(contract["engineering_audit_path"])
    stored_audit = _read(audit_path)
    recomputed = engineering_auditor.audit(
        plan_path=Path(contract["engineering_plan_path"]),
        expected_plan_sha256=_sha256(Path(contract["engineering_plan_path"])),
    )
    _require(
        stored_audit.get("status") == "PASS"
        and stored_audit.get("controller_result")
        == recomputed.get("controller_result")
        and stored_audit.get("gate_passed") == recomputed.get("gate_passed")
        and stored_audit.get("winner", {}).get("model_id")
        == recomputed.get("winner", {}).get("model_id")
        and stored_audit.get("winner", {}).get("model_sha256")
        == recomputed.get("winner", {}).get("model_sha256"),
        "engineering independent audit cannot be reproduced",
    )
    output_root = Path(
        str(contract["engineering_plan"]["outputs"]["output_root"])
    ).resolve(strict=True)
    report_path = output_root / "final_report.json"
    status_path = output_root / "controller_status.json"
    report = _read(report_path)
    status = _read(status_path)
    _require(
        status.get("status") == "COMPLETE"
        and status.get("phase") == "COMPLETE"
        and status.get("result", {}).get("sha256") == _sha256(report_path)
        and status.get("independent_audit", {}).get("sha256")
        == _sha256(audit_path),
        "engineering controller did not complete cleanly",
    )
    passed = bool(recomputed["gate_passed"])
    _require(
        report.get("status")
        == ("COMPLETE_GATE_PASS" if passed else "COMPLETE_NO_PROMOTION")
        and bool(report.get("formal_candidate_authority")) is passed,
        "engineering promotion result differs",
    )
    model = dict(report["promotion_gate"]["winner_model"])
    model_path = Path(str(model.get("path", ""))).resolve(strict=True)
    _require(
        model.get("sha256") == _sha256(model_path)
        and model.get("observation_profile") == "motor-observable-v1"
        and bool(model.get("promotable")) is True,
        "engineering winner is not an eligible frozen motor policy",
    )
    winner = report["promotion_gate"]["winner"]
    _require(
        winner.get("model_id") == model.get("id")
        and winner.get("model_sha256") == model.get("sha256"),
        "engineering winner model and ranking row differ",
    )
    return {
        "passed": passed,
        "model": model,
        "report_path": report_path,
        "report_sha256": _sha256(report_path),
        "audit_path": audit_path,
        "audit_sha256": _sha256(audit_path),
        "recomputed_audit": recomputed,
    }


def _build_contracts(
    *,
    contract: Mapping[str, Any],
    stage: str,
    model: dict[str, Any],
    decision_path: Path,
) -> tuple[Path, Path]:
    if stage == "final_blind":
        attempts = FINAL_ATTEMPTS
        base_seed = FINAL_SEED_BASE
        output_root = Path(contract["outputs"]["final_blind"])
        purpose = "alphazuma-55-motor-successor-final-blind"
    elif stage == "continuous":
        attempts = CONTINUOUS_CAMPAIGNS
        base_seed = CONTINUOUS_SEED_BASE
        output_root = Path(contract["outputs"]["continuous"])
        purpose = "alphazuma-55-motor-successor-continuous-campaigns"
    else:
        raise ValueError(f"unsupported formal stage: {stage}")
    controller_root = Path(contract["outputs"]["controller"])
    manifest_path = controller_root / f"{stage}-models-manifest.json"
    prereg_path = controller_root / f"{stage}-preregistration.json"
    master_ref = {
        "path": str(contract["path"]),
        "sha256": _sha256(Path(contract["path"])),
    }
    decision_ref = {
        "path": str(decision_path),
        "sha256": _sha256(decision_path),
    }
    evaluator_path = Path(contract["implementation"]["evaluator"])
    manifest = {
        "schema": "zuma-rl.zero-shot-models-manifest",
        "version": 1,
        "status": "FROZEN",
        "created_utc": _utc_now(),
        "purpose": purpose,
        "master_preregistration": master_ref,
        "promotion_decision": decision_ref,
        "single_policy_only": True,
        "runtime_model_switching_forbidden": True,
        "models": [dict(model)],
    }
    preregistration = {
        "schema": "zuma-rl.zero-shot-multilevel-preregistration",
        "version": 1,
        "status": "FROZEN_BEFORE_EVALUATION",
        "frozen_utc": _utc_now(),
        "purpose": purpose,
        "stage": stage,
        "master_preregistration": master_ref,
        "promotion_decision": decision_ref,
        "evaluator": {
            "path": str(evaluator_path),
            "sha256": _sha256(evaluator_path),
        },
        "model": dict(model),
        "levels": [dict(row) for row in contract["template"]["levels"]],
        "attempts_per_level": attempts,
        "total_attempts": LEVEL_COUNT * attempts,
        "seed_plan": {
            "registry": (
                "motor_successor_final_blind"
                if stage == "final_blind"
                else "motor_successor_continuous_campaign_challenge"
            ),
            "base_seed": base_seed,
            "last_seed": base_seed + LEVEL_COUNT * attempts - 1,
            "all_seeds_frozen_before_policy_inference": True,
            "seed_formula": "base_seed + level_index*attempts_per_level + attempt_index",
        },
        "environment": contract["template"]["environment"],
        "execution": {
            "output_root": str(output_root),
            "shard_count": 2,
            "devices": list(contract["devices"]),
            "parallel_envs_per_shard": int(
                contract["parallel_envs_per_shard"]
            ),
        },
        "reporting": {"all_attempts_must_be_reported": True},
        "authority_boundary": {
            "classification": stage,
            "single_frozen_policy": True,
            "runtime_model_switching": False,
        },
    }
    _write_new(manifest_path, manifest)
    _write_new(prereg_path, preregistration)
    return manifest_path, prereg_path


def _run_independent_audit(
    *, contract: Mapping[str, Any], report_path: Path, status_path: Path
) -> dict[str, Any]:
    output = Path(contract["outputs"]["independent_audit_receipt"])
    auditor = Path(contract["implementation"]["independent_auditor"])
    stdout_path = Path(contract["outputs"]["controller"]) / "audit.stdout.log"
    stderr_path = Path(contract["outputs"]["controller"]) / "audit.stderr.log"
    with stdout_path.open("x", encoding="utf-8") as stdout, stderr_path.open(
        "x", encoding="utf-8"
    ) as stderr:
        result = subprocess.run(
            [
                sys.executable,
                str(auditor),
                "--master",
                str(contract["path"]),
                "--expected-master-sha256",
                _sha256(Path(contract["path"])),
                "--controller-report",
                str(report_path),
                "--controller-status",
                str(status_path),
                "--output",
                str(output),
            ],
            cwd=PROJECT_ROOT,
            stdout=stdout,
            stderr=stderr,
            check=False,
            text=True,
        )
    if result.returncode != 0:
        raise RuntimeError("motor successor independent audit failed")
    receipt = _read(output)
    _require(receipt.get("status") == "PASS", "independent audit is not PASS")
    return {
        "path": str(output),
        "sha256": _sha256(output),
        "receipt": receipt,
    }


def run(*, master_path: Path, original_root: Path, poll_seconds: float) -> int:
    contract = _load_master(master_path)
    original_root = original_root.resolve(strict=True)
    _require(original_root == contract["original_root"], "retail root differs")
    outputs = contract["outputs"]
    for name in ("root", "controller", "final_blind", "continuous"):
        _require(not outputs[name].exists(), f"successor output exists: {name}")
    outputs["root"].mkdir(parents=True)
    outputs["controller"].mkdir()
    status_path = outputs["controller"] / "controller_status.json"
    state: dict[str, Any] = {
        "schema": "zuma-rl.alphazuma-55-motor-successor-controller-status",
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
    _replace_json(status_path, state)
    try:
        audit_path = Path(contract["engineering_audit_path"])
        engineering_root = Path(
            str(contract["engineering_plan"]["outputs"]["output_root"])
        )
        while not audit_path.exists():
            failure_path = engineering_root / "failure.json"
            if failure_path.exists():
                raise RuntimeError("motor engineering route failed before audit")
            if datetime.now(timezone.utc) >= contract["deadline"]:
                raise TimeoutError("motor successor missed the campaign deadline")
            state.update(
                {
                    "phase": "WAITING_FOR_ENGINEERING_AUDIT",
                    "updated_utc": _utc_now(),
                    "waiting_for": str(audit_path),
                    "formal_seed_consumption": "NONE",
                }
            )
            _replace_json(status_path, state)
            time.sleep(poll_seconds)

        state.update(
            {"phase": "FREEZING_PROMOTION_DECISION", "updated_utc": _utc_now()}
        )
        _replace_json(status_path, state)
        engineering_result = _engineering_result(contract)
        selected = dict(engineering_result["model"])
        decision = {
            "schema": "zuma-rl.alphazuma-55-motor-successor-promotion-decision",
            "version": 1,
            "status": (
                "PROMOTED" if engineering_result["passed"] else "NO_PROMOTION"
            ),
            "completed_utc": _utc_now(),
            "engineering_report": {
                "path": str(engineering_result["report_path"]),
                "sha256": engineering_result["report_sha256"],
            },
            "engineering_independent_audit": {
                "path": str(engineering_result["audit_path"]),
                "sha256": engineering_result["audit_sha256"],
            },
            "selected_policy": selected,
            "promotion_gate_passed": bool(engineering_result["passed"]),
            "successor_formal_seed_consumption_authorized": bool(
                engineering_result["passed"]
            ),
        }
        decision_path = outputs["controller"] / "promotion-decision.json"
        _write_new(decision_path, decision)

        if not engineering_result["passed"]:
            report = {
                "schema": "zuma-rl.alphazuma-55-motor-successor-final-report",
                "version": 1,
                "status": "COMPLETE_NO_PROMOTION",
                "completed_utc": _utc_now(),
                "master_preregistration": state["master_preregistration"],
                "promotion_decision": {
                    "path": str(decision_path),
                    "sha256": _sha256(decision_path),
                },
                "selected_engineering_policy": selected,
                "formal_seed_consumption": "NONE",
                "goal_gates_passed": False,
            }
            report_path = outputs["controller"] / "final_report.json"
            _write_new(report_path, report)
            state.update(
                {
                    "status": "RUNNING",
                    "phase": "RUNNING_INDEPENDENT_AUDIT",
                    "updated_utc": _utc_now(),
                    "final_report": {
                        "path": str(report_path),
                        "sha256": _sha256(report_path),
                    },
                    "formal_seed_consumption": "NONE",
                }
            )
            _replace_json(status_path, state)
            audit = _run_independent_audit(
                contract=contract,
                report_path=report_path,
                status_path=status_path,
            )
            state.update(
                {
                    "status": "COMPLETE_NO_PROMOTION",
                    "phase": "COMPLETE_NO_PROMOTION",
                    "updated_utc": _utc_now(),
                    "independent_audit": {
                        "path": audit["path"],
                        "sha256": audit["sha256"],
                    },
                }
            )
            _replace_json(status_path, state)
            return 0

        final_manifest, final_prereg = _build_contracts(
            contract=contract,
            stage="final_blind",
            model=selected,
            decision_path=decision_path,
        )
        continuous_manifest, continuous_prereg = _build_contracts(
            contract=contract,
            stage="continuous",
            model=selected,
            decision_path=decision_path,
        )
        state.update(
            {
                "phase": "FINAL_BLIND_EVALUATION",
                "updated_utc": _utc_now(),
                "selected_single_policy": selected,
                "formal_seed_consumption": (
                    f"FINAL_BLIND_{FINAL_SEED_BASE}_"
                    f"{FINAL_SEED_BASE + LEVEL_COUNT * FINAL_ATTEMPTS - 1}"
                ),
            }
        )
        _replace_json(status_path, state)
        evaluation.EVALUATOR = contract["implementation"]["evaluator"]
        evaluation.SUMMARIZER = contract["implementation"]["summarizer"]
        final_aggregate = evaluation._evaluate_and_summarize(
            phase="motor-successor-final-blind",
            preregistration=final_prereg,
            manifest=final_manifest,
            original_root=original_root,
            output_root=outputs["final_blind"],
            shard_count=2,
            purpose="alphazuma-55-motor-successor-final-blind",
        )
        final_audit = _strict_audit_evaluation(
            root=outputs["final_blind"],
            preregistration_path=final_prereg,
            manifest_path=final_manifest,
            aggregate_path=outputs["final_blind"] / "aggregate.json",
            expected_level_ids=contract["levels"],
            expected_attempts_per_level=FINAL_ATTEMPTS,
            expected_seed_base=FINAL_SEED_BASE,
            expected_model_ids=[str(selected["id"])],
        )
        final_summary = next(
            row
            for row in final_aggregate["models"]
            if str(row["model_id"]) == str(selected["id"])
        )
        capability_actual, capability_passed = successor_helpers._capability(
            final_summary
        )

        state.update(
            {
                "phase": "CONTINUOUS_CAMPAIGNS",
                "updated_utc": _utc_now(),
                "formal_seed_consumption": (
                    f"FINAL_BLIND_{FINAL_SEED_BASE}_"
                    f"{FINAL_SEED_BASE + LEVEL_COUNT * FINAL_ATTEMPTS - 1}_"
                    f"AND_CONTINUOUS_{CONTINUOUS_SEED_BASE}_"
                    f"{CONTINUOUS_SEED_BASE + LEVEL_COUNT * CONTINUOUS_CAMPAIGNS - 1}"
                ),
            }
        )
        _replace_json(status_path, state)
        continuous_aggregate = evaluation._evaluate_and_summarize(
            phase="motor-successor-continuous",
            preregistration=continuous_prereg,
            manifest=continuous_manifest,
            original_root=original_root,
            output_root=outputs["continuous"],
            shard_count=2,
            purpose="alphazuma-55-motor-successor-continuous-campaigns",
        )
        continuous_audit = _strict_audit_evaluation(
            root=outputs["continuous"],
            preregistration_path=continuous_prereg,
            manifest_path=continuous_manifest,
            aggregate_path=outputs["continuous"] / "aggregate.json",
            expected_level_ids=contract["levels"],
            expected_attempts_per_level=CONTINUOUS_CAMPAIGNS,
            expected_seed_base=CONTINUOUS_SEED_BASE,
            expected_model_ids=[str(selected["id"])],
        )
        campaigns = successor_helpers._continuous_campaigns(
            successor_helpers._all_rows(outputs["continuous"]),
            level_ids=contract["levels"],
            model_id=str(selected["id"]),
        )
        goal_passed = capability_passed and bool(campaigns["passed"])
        report = {
            "schema": "zuma-rl.alphazuma-55-motor-successor-final-report",
            "version": 1,
            "status": (
                "COMPLETE_FORMAL_PASS" if goal_passed else "COMPLETE_FORMAL_FAIL"
            ),
            "completed_utc": _utc_now(),
            "master_preregistration": state["master_preregistration"],
            "formal_seed_registry": {
                "path": str(contract["registry"]["path"]),
                "sha256": contract["registry"]["sha256"],
            },
            "promotion_decision": {
                "path": str(decision_path),
                "sha256": _sha256(decision_path),
            },
            "selected_single_policy": selected,
            "final_blind": {
                "manifest": {
                    "path": str(final_manifest),
                    "sha256": _sha256(final_manifest),
                },
                "preregistration": {
                    "path": str(final_prereg),
                    "sha256": _sha256(final_prereg),
                },
                "aggregate": {
                    "path": str(outputs["final_blind"] / "aggregate.json"),
                    "sha256": _sha256(outputs["final_blind"] / "aggregate.json"),
                },
                "summary": final_summary,
            },
            "continuous": {
                "manifest": {
                    "path": str(continuous_manifest),
                    "sha256": _sha256(continuous_manifest),
                },
                "preregistration": {
                    "path": str(continuous_prereg),
                    "sha256": _sha256(continuous_prereg),
                },
                "aggregate": {
                    "path": str(outputs["continuous"] / "aggregate.json"),
                    "sha256": _sha256(outputs["continuous"] / "aggregate.json"),
                },
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
                    "passed": capability_passed,
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
            "formal_seed_consumption": (
                f"FINAL_BLIND_{FINAL_SEED_BASE}_"
                f"{FINAL_SEED_BASE + LEVEL_COUNT * FINAL_ATTEMPTS - 1}_"
                f"AND_CONTINUOUS_{CONTINUOUS_SEED_BASE}_"
                f"{CONTINUOUS_SEED_BASE + LEVEL_COUNT * CONTINUOUS_CAMPAIGNS - 1}"
            ),
            "goal_gates_passed": goal_passed,
            "claims": {
                "one_frozen_simulator_state_policy": True,
                "runtime_model_switching": False,
                "original_client_world_record_claim": False,
                "continuous_full_game_claim_supported": bool(campaigns["passed"]),
            },
        }
        report_path = outputs["controller"] / "final_report.json"
        _write_new(report_path, report)
        state.update(
            {
                "status": "RUNNING",
                "phase": "RUNNING_INDEPENDENT_AUDIT",
                "updated_utc": _utc_now(),
                "goal_gates_passed": goal_passed,
                "final_report": {
                    "path": str(report_path),
                    "sha256": _sha256(report_path),
                },
            }
        )
        _replace_json(status_path, state)
        audit = _run_independent_audit(
            contract=contract,
            report_path=report_path,
            status_path=status_path,
        )
        state.update(
            {
                "status": (
                    "COMPLETE_FORMAL_PASS"
                    if goal_passed
                    else "COMPLETE_FORMAL_FAIL"
                ),
                "phase": "COMPLETE",
                "updated_utc": _utc_now(),
                "independent_audit": {
                    "path": audit["path"],
                    "sha256": audit["sha256"],
                },
            }
        )
        _replace_json(status_path, state)
        return 0
    except BaseException as error:
        failure = {
            "schema": "zuma-rl.alphazuma-55-motor-successor-failure",
            "version": 1,
            "status": "ERROR",
            "failed_utc": _utc_now(),
            "phase": state.get("phase"),
            "error_type": type(error).__name__,
            "error": str(error),
            "formal_seed_consumption": state.get(
                "formal_seed_consumption", "NONE"
            ),
        }
        _replace_json(outputs["controller"] / "failure.json", failure)
        state.update(
            {
                "status": "ERROR",
                "phase": "ERROR",
                "updated_utc": _utc_now(),
                "error": failure,
            }
        )
        _replace_json(status_path, state)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", required=True, type=Path)
    parser.add_argument("--expected-master-sha256", required=True)
    parser.add_argument("--original-root", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    master_path = args.master.expanduser().resolve(strict=True)
    _require(
        _sha256(master_path) == str(args.expected_master_sha256),
        "motor successor master hash differs",
    )
    contract = _load_master(master_path)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "VALID",
                    "master": {
                        "path": str(master_path),
                        "sha256": _sha256(master_path),
                    },
                    "engineering_audit": str(contract["engineering_audit_path"]),
                    "final_blind_attempts": LEVEL_COUNT * FINAL_ATTEMPTS,
                    "continuous_attempts": LEVEL_COUNT * CONTINUOUS_CAMPAIGNS,
                    "formal_seed_registry": {
                        "path": str(contract["registry"]["path"]),
                        "sha256": contract["registry"]["sha256"],
                    },
                    "formal_seed_consumption": "NONE",
                },
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        )
        return 0
    return run(
        master_path=master_path,
        original_root=args.original_root.expanduser(),
        poll_seconds=max(1.0, float(args.poll_seconds)),
    )


if __name__ == "__main__":
    raise SystemExit(main())
