"""Audit the final AlphaZuma 55 single-policy goal across frozen campaigns.

This auditor has no policy-inference authority.  It only accepts a campaign
after that campaign's own independent auditor can reproduce its receipt, the
formal 55-level capability gate passes, and at least one continuous 55-level
campaign is cleared by the same hash-verified neural-policy archive.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping
import zipfile

try:
    from audit_alphazuma_55_result import audit as audit_source_campaign
    from audit_alphazuma_55_successor_result import audit as audit_successor
    from audit_alphazuma_55_configured_successor_result import (
        audit as audit_configured_successor,
    )
except ModuleNotFoundError:  # Imported as tools.<module> in tests.
    from tools.audit_alphazuma_55_result import audit as audit_source_campaign
    from tools.audit_alphazuma_55_successor_result import audit as audit_successor
    from tools.audit_alphazuma_55_configured_successor_result import (
        audit as audit_configured_successor,
    )


SCRIPT_PATH = Path(__file__).resolve()
LEVEL_COUNT = 55
FINAL_ATTEMPTS = 440
CONTINUOUS_ATTEMPTS = 220
_PENDING = {"PENDING", "MISSING_AT_DEADLINE"}


class GoalAuditError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GoalAuditError(message)


def _reject_constant(value: str) -> None:
    raise GoalAuditError(f"non-finite JSON constant is forbidden: {value}")


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise GoalAuditError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8-sig"),
            object_pairs_hook=_pairs,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GoalAuditError(f"cannot read strict JSON {path}: {exc}") from exc
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _reference(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": _sha256(path)}


def _artifact(reference: Any, name: str) -> Path:
    _require(isinstance(reference, dict), f"{name} reference is not an object")
    path = Path(str(reference.get("path", ""))).expanduser().resolve(strict=True)
    _require(reference.get("sha256") == _sha256(path), f"{name} hash differs")
    return path


def _utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    _require(parsed.tzinfo is not None, "deadline lacks a timezone")
    return parsed.astimezone(timezone.utc)


def _without_audit_time(value: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(value))
    result.pop("audited_utc", None)
    return result


def _verify_policy(model: Any) -> dict[str, Any]:
    _require(isinstance(model, dict), "selected policy is not an object")
    _require(str(model.get("id", "")), "selected policy id is absent")
    path = Path(str(model.get("path", ""))).expanduser().resolve(strict=True)
    _require(path.is_file() and path.stat().st_size > 0, "selected policy is empty")
    digest = _sha256(path)
    _require(model.get("sha256") == digest, "selected policy hash differs")
    try:
        with zipfile.ZipFile(path) as archive:
            members = set(archive.namelist())
            _require(
                {"data", "policy.pth"}.issubset(members),
                "selected policy is not a Stable-Baselines neural archive",
            )
            _require(archive.testzip() is None, "selected policy ZIP CRC differs")
    except (OSError, zipfile.BadZipFile) as exc:
        raise GoalAuditError(f"selected policy archive is invalid: {exc}") from exc
    return {
        **dict(model),
        "path": str(path),
        "sha256": digest,
        "bytes": path.stat().st_size,
        "neural_archive_verified": True,
    }


def _production_recomputers() -> dict[str, dict[str, Any]]:
    return {
        "source": {
            "path": Path(audit_source_campaign.__code__.co_filename).resolve(),
            "call": audit_source_campaign,
        },
        "successor": {
            "path": Path(audit_successor.__code__.co_filename).resolve(),
            "call": audit_successor,
        },
        "configured_successor": {
            "path": Path(audit_configured_successor.__code__.co_filename).resolve(),
            "call": audit_configured_successor,
        },
    }


def _validate_seed_registry(
    reference: Any, campaigns: list[dict[str, Any]]
) -> dict[str, Any]:
    path = _artifact(reference, "formal seed registry")
    registry = _read(path)
    _require(
        registry.get("schema")
        == "zuma-rl.alphazuma-55-weekend-formal-seed-registry"
        and registry.get("version") == 1
        and registry.get("status")
        == "FROZEN_BEFORE_ANY_SUCCESSOR_FORMAL_INFERENCE",
        "formal seed registry identity differs",
    )
    checks = registry.get("global_checks")
    _require(isinstance(checks, dict) and all(checks.values()), "seed registry checks fail")
    registry_campaigns = {
        str(row["id"]): row for row in registry.get("campaigns", [])
    }
    _require(
        len(registry_campaigns) == len(registry.get("campaigns", [])),
        "seed registry campaign ids repeat",
    )
    ranges: list[tuple[int, int, str, str]] = []
    for campaign in campaigns:
        registry_id = str(campaign.get("seed_registry_campaign_id", ""))
        _require(registry_id in registry_campaigns, "campaign is absent from seed registry")
        registered = registry_campaigns[registry_id]
        _require(
            registered.get("master") == campaign.get("campaign_master"),
            f"seed registry master differs for {campaign['id']}",
        )
        for row in registered.get("ranges", []):
            first = int(row["first"])
            last = int(row["last"])
            _require(0 <= first <= last <= 0xFFFFFFFF, "seed range is invalid")
            ranges.append((first, last, registry_id, str(row["stage"])))
    ranges.sort()
    for left, right in zip(ranges, ranges[1:]):
        _require(left[1] < right[0], f"formal seed ranges overlap: {left} / {right}")
    return {
        "artifact": _reference(path),
        "campaigns_verified": len(campaigns),
        "ranges_verified": len(ranges),
        "strictly_non_overlapping": True,
    }


def _campaign_result(
    campaign: dict[str, Any],
    *,
    deadline: datetime,
    now: datetime,
    recomputers: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    campaign_id = str(campaign["id"])
    kind = str(campaign["kind"])
    _artifact(campaign.get("campaign_master"), f"{campaign_id} campaign master")
    audit_input = _artifact(campaign.get("audit_input"), f"{campaign_id} audit input")
    auditor_path = _artifact(
        campaign.get("independent_auditor"), f"{campaign_id} independent auditor"
    )
    _require(kind in recomputers, f"unsupported campaign kind: {kind}")
    recomputer = recomputers[kind]
    expected_auditor = Path(str(recomputer["path"])).resolve(strict=True)
    _require(auditor_path == expected_auditor, f"{campaign_id} binds another auditor")
    callback = recomputer.get("call")
    _require(callable(callback), f"{campaign_id} recomputer is not callable")
    receipt_path = Path(str(campaign["independent_receipt"])).expanduser().resolve()
    if not receipt_path.exists():
        return {
            "id": campaign_id,
            "kind": kind,
            "status": "MISSING_AT_DEADLINE" if now >= deadline else "PENDING",
            "independent_receipt": str(receipt_path),
            "eligible": False,
        }
    receipt = _read(receipt_path)
    recomputed = callback(audit_input)
    _require(isinstance(recomputed, dict), f"{campaign_id} auditor returned no object")
    _require(
        _without_audit_time(receipt) == _without_audit_time(recomputed),
        f"{campaign_id} receipt cannot be reproduced",
    )
    _require(receipt.get("status") == "PASS", f"{campaign_id} audit is not PASS")
    receipt_reference = _reference(receipt_path)
    controller_result = receipt.get("controller_result")
    if kind != "source" and controller_result == "COMPLETE_NO_PROMOTION":
        _require(
            receipt.get("formal_seed_consumption") == "NONE",
            f"{campaign_id} no-promotion seed boundary differs",
        )
        return {
            "id": campaign_id,
            "kind": kind,
            "status": "TERMINAL_NO_PROMOTION",
            "independent_receipt": receipt_reference,
            "promotion_gate": receipt.get("promotion_gate"),
            "eligible": False,
        }
    if kind != "source":
        _require(controller_result == "COMPLETE", f"{campaign_id} controller is incomplete")
        _require(
            receipt.get("promotion_gate", {}).get("passed") is True,
            f"{campaign_id} formal inference lacks a passing promotion gate",
        )
    final_attempts = int(receipt.get("final_blind_attempts", -1))
    continuous_attempts = int(receipt.get("continuous_attempts", -1))
    _require(
        final_attempts == FINAL_ATTEMPTS,
        f"{campaign_id} final-blind attempt count differs",
    )
    _require(
        continuous_attempts == CONTINUOUS_ATTEMPTS,
        f"{campaign_id} continuous attempt count differs",
    )
    capability_pass = receipt.get("single_policy_capability_gate") is True
    continuous_pass = receipt.get("continuous_full_game_gate") is True
    if not (capability_pass and continuous_pass):
        return {
            "id": campaign_id,
            "kind": kind,
            "status": "TERMINAL_GATE_FAIL",
            "independent_receipt": receipt_reference,
            "final_blind_attempts": final_attempts,
            "continuous_attempts": continuous_attempts,
            "single_policy_capability_gate": capability_pass,
            "continuous_full_game_gate": continuous_pass,
            "eligible": False,
        }
    if kind == "source":
        report_path = _artifact(receipt.get("final_report"), f"{campaign_id} final report")
        selected = receipt.get("selected_single_policy")
    else:
        artifacts = receipt.get("artifacts")
        _require(isinstance(artifacts, dict), f"{campaign_id} artifacts are absent")
        report_path = _artifact(artifacts.get("final_report"), f"{campaign_id} final report")
        selected = _read(report_path).get("selected_single_policy")
    report = _read(report_path)
    _require(report.get("status") == "COMPLETE", f"{campaign_id} report is incomplete")
    _require(
        report.get("selected_single_policy") == selected,
        f"{campaign_id} selected policy differs between artifacts",
    )
    policy = _verify_policy(selected)
    gates = report.get("success_gates")
    _require(isinstance(gates, dict), f"{campaign_id} report gates are absent")
    capability = gates.get("single_policy_capability_gate", {}).get("actual")
    _require(isinstance(capability, dict), f"{campaign_id} capability metrics are absent")
    _require(
        int(capability.get("levels_cleared", -1)) == LEVEL_COUNT
        and int(capability.get("minimum_wins_per_level", -1)) >= 1
        and int(capability.get("total_wins", -1)) >= 220
        and int(capability.get("attempts", -1)) == FINAL_ATTEMPTS,
        f"{campaign_id} capability metrics do not prove the 55-level goal",
    )
    continuous = report.get("continuous_campaigns")
    _require(isinstance(continuous, dict), f"{campaign_id} continuous evidence is absent")
    campaigns = continuous.get("campaigns")
    _require(
        isinstance(campaigns, list)
        and len(campaigns) == 4
        and any(row.get("cleared_all_55") is True for row in campaigns),
        f"{campaign_id} has no complete continuous 55-level campaign",
    )
    return {
        "id": campaign_id,
        "kind": kind,
        "status": "ELIGIBLE",
        "independent_receipt": receipt_reference,
        "final_report": _reference(report_path),
        "selected_single_policy": policy,
        "capability": capability,
        "continuous_campaigns_cleared": int(continuous.get("campaigns_cleared", 0)),
        "final_blind_attempts": final_attempts,
        "continuous_attempts": continuous_attempts,
        "single_policy_capability_gate": True,
        "continuous_full_game_gate": True,
        "eligible": True,
    }


def audit(
    plan_path: Path,
    *,
    now: datetime | None = None,
    recomputers: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    plan_path = plan_path.expanduser().resolve(strict=True)
    plan = _read(plan_path)
    _require(
        plan.get("schema") == "zuma-rl.alphazuma-55-single-policy-goal-audit-plan"
        and plan.get("version") == 1
        and plan.get("status") == "FROZEN_BEFORE_GOAL_COMPLETION",
        "goal audit plan identity differs",
    )
    implementation = plan.get("implementation")
    _require(isinstance(implementation, dict), "goal auditor binding is absent")
    _require(
        _artifact(implementation.get("goal_auditor"), "goal auditor") == SCRIPT_PATH,
        "plan binds another goal auditor",
    )
    requirements = plan.get("requirements")
    _require(isinstance(requirements, dict), "goal requirements are absent")
    _require(
        requirements
        == {
            "one_frozen_neural_policy": True,
            "included_levels": LEVEL_COUNT,
            "final_blind_attempts": FINAL_ATTEMPTS,
            "minimum_final_blind_total_wins": 220,
            "minimum_win_per_level": 1,
            "continuous_campaigns": 4,
            "continuous_attempts": CONTINUOUS_ATTEMPTS,
            "minimum_complete_continuous_55_level_campaigns": 1,
        },
        "goal requirements differ",
    )
    deadline = _utc(str(plan["deadline_utc"]))
    checked = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    campaigns = plan.get("campaigns")
    _require(isinstance(campaigns, list) and campaigns, "goal campaigns are absent")
    ids = [str(row["id"]) for row in campaigns]
    _require(len(ids) == len(set(ids)), "goal campaign ids repeat")
    selection = plan.get("selection")
    _require(isinstance(selection, dict), "goal selection contract is absent")
    _require(
        selection.get("campaign_priority") == ids
        and selection.get("higher_priority_must_be_terminal_before_lower_selection")
        is True,
        "goal campaign priority differs",
    )
    seed_registry = _validate_seed_registry(plan.get("formal_seed_registry"), campaigns)
    active_recomputers = recomputers or _production_recomputers()
    results = [
        _campaign_result(
            dict(campaign),
            deadline=deadline,
            now=checked,
            recomputers=active_recomputers,
        )
        for campaign in campaigns
    ]
    selected: dict[str, Any] | None = None
    blocked_by: str | None = None
    for result in results:
        if result["status"] == "PENDING":
            blocked_by = str(result["id"])
            break
        if result["status"] == "ELIGIBLE":
            selected = result
            break
    if selected is not None:
        status = "PASS"
    elif checked >= deadline:
        status = "FAIL_AT_DEADLINE"
    else:
        status = "WAITING"
    output: dict[str, Any] = {
        "schema": "zuma-rl.alphazuma-55-single-policy-goal-independent-audit",
        "version": 1,
        "status": status,
        "goal_achieved": status == "PASS",
        "checked_utc": checked.isoformat(),
        "deadline_utc": deadline.isoformat(),
        "plan": _reference(plan_path),
        "formal_seed_registry": seed_registry,
        "requirements": requirements,
        "campaign_priority": ids,
        "campaigns": results,
        "selection_blocked_by_pending_higher_priority": blocked_by,
    }
    if selected is not None:
        output.update(
            {
                "selected_campaign_id": selected["id"],
                "selected_single_policy": selected["selected_single_policy"],
                "proof": {
                    "independent_receipt": selected["independent_receipt"],
                    "final_report": selected["final_report"],
                    "final_blind_attempts": FINAL_ATTEMPTS,
                    "continuous_attempts": CONTINUOUS_ATTEMPTS,
                    "single_policy_capability_gate": True,
                    "continuous_full_game_gate": True,
                },
            }
        )
    return output


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
        raise FileExistsError(f"refusing to overwrite goal receipt: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="Print WAITING/PASS/FAIL evidence without creating the receipt.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    receipt_path = args.receipt.expanduser().resolve()
    result = audit(args.plan)
    if args.status_only:
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    if receipt_path.exists():
        existing = _read(receipt_path)
        _require(existing.get("status") == "PASS", "existing goal receipt is not PASS")
        _require(result.get("status") == "PASS", "current evidence no longer passes")
        _require(
            existing.get("plan") == result.get("plan")
            and existing.get("selected_campaign_id")
            == result.get("selected_campaign_id")
            and existing.get("selected_single_policy")
            == result.get("selected_single_policy"),
            "existing goal receipt differs from current evidence",
        )
        print(json.dumps(existing, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    if result["status"] != "PASS":
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 2
    _write_exclusive(receipt_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
