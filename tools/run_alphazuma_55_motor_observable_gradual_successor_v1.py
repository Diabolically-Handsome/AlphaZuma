"""Conditionally certify the gradual motor-observable engineering winner."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tools import audit_alphazuma_55_motor_observable_gradual_postprocess_v1 as engineering_auditor
from tools import run_alphazuma_55_motor_observable_gradual_postprocess_v1 as engineering
from tools import run_alphazuma_55_motor_observable_successor_v1 as base
from zuma_rl.alphazuma_55 import INCLUDED_LEVELS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()
MASTER_SCHEMA = base.MASTER_SCHEMA
LEVEL_COUNT = base.LEVEL_COUNT
FINAL_ATTEMPTS = base.FINAL_ATTEMPTS
CONTINUOUS_CAMPAIGNS = base.CONTINUOUS_CAMPAIGNS
FINAL_SEED_BASE = 3_500_000_000
CONTINUOUS_SEED_BASE = 3_600_000_000

_require = base._require
_read = base._read
_sha256 = base._sha256
_artifact = base._artifact
_utc_now = base._utc_now
_parse_utc = base._parse_utc
_write_new = base._write_new
_replace_json = base._replace_json
successor_helpers = base.successor_helpers
_strict_audit_evaluation = base._strict_audit_evaluation
_base_engineering_result = base._engineering_result


def _validate_registry(
    *, registry_path: Path, master_path: Path
) -> dict[str, Any]:
    original_final = base.FINAL_SEED_BASE
    original_continuous = base.CONTINUOUS_SEED_BASE
    try:
        base.FINAL_SEED_BASE = FINAL_SEED_BASE
        base.CONTINUOUS_SEED_BASE = CONTINUOUS_SEED_BASE
        return base._validate_registry(
            registry_path=registry_path,
            master_path=master_path,
        )
    finally:
        base.CONTINUOUS_SEED_BASE = original_continuous
        base.FINAL_SEED_BASE = original_final


def _load_master(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    master = _read(path)
    _require(
        master.get("schema") == MASTER_SCHEMA
        and master.get("version") == 1
        and master.get("status") == "FROZEN_BEFORE_ENGINEERING_RESULT",
        "unexpected gradual motor successor master",
    )
    levels = list(master.get("scope", {}).get("included_levels", []))
    _require(levels == list(INCLUDED_LEVELS), "gradual successor level scope differs")
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
        "gradual successor promotion gate changed",
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
        "gradual successor formal seed contract changed",
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
        "gradual successor execution topology changed",
    )
    expected_implementation = {
        "master_builder": PROJECT_ROOT
        / "tools/build_alphazuma_55_motor_observable_gradual_successor_master_v1.py",
        "controller": SCRIPT_PATH,
        "independent_auditor": PROJECT_ROOT
        / "tools/audit_alphazuma_55_motor_observable_gradual_successor_v1.py",
        "evaluator": PROJECT_ROOT
        / "tools/evaluate_alphazuma_55_motor_observable.py",
        "evaluation_helper": PROJECT_ROOT / "tools/run_overnight_v11_postprocess.py",
        "matrix_auditor": PROJECT_ROOT
        / "tools/audit_alphazuma_v11_frontier_result.py",
        "summarizer": PROJECT_ROOT / "tools/summarize_multimodel_evaluation.py",
        "engineering_controller": PROJECT_ROOT
        / "tools/run_alphazuma_55_motor_observable_gradual_postprocess_v1.py",
        "engineering_auditor": PROJECT_ROOT
        / "tools/audit_alphazuma_55_motor_observable_gradual_postprocess_v1.py",
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
        "gradual successor output topology differs",
    )
    original_root = Path(str(master.get("original_root", ""))).resolve(strict=True)
    _require(
        original_root == Path(str(plan["original_root"])).resolve(strict=True),
        "gradual successor original root differs",
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
    original_engineering = base.engineering
    original_auditor = base.engineering_auditor
    try:
        base.engineering = engineering
        base.engineering_auditor = engineering_auditor
        return _base_engineering_result(contract)
    finally:
        base.engineering_auditor = original_auditor
        base.engineering = original_engineering


@contextmanager
def _patched_base() -> Iterator[None]:
    originals = {
        "FINAL_SEED_BASE": base.FINAL_SEED_BASE,
        "CONTINUOUS_SEED_BASE": base.CONTINUOUS_SEED_BASE,
        "SCRIPT_PATH": base.SCRIPT_PATH,
        "engineering": base.engineering,
        "engineering_auditor": base.engineering_auditor,
        "_load_master": base._load_master,
        "_engineering_result": base._engineering_result,
    }
    try:
        base.FINAL_SEED_BASE = FINAL_SEED_BASE
        base.CONTINUOUS_SEED_BASE = CONTINUOUS_SEED_BASE
        base.SCRIPT_PATH = SCRIPT_PATH
        base.engineering = engineering
        base.engineering_auditor = engineering_auditor
        base._load_master = _load_master
        base._engineering_result = _engineering_result
        yield
    finally:
        for name, value in originals.items():
            setattr(base, name, value)


def run(*, master_path: Path, original_root: Path, poll_seconds: float) -> int:
    with _patched_base():
        return base.run(
            master_path=master_path,
            original_root=original_root,
            poll_seconds=poll_seconds,
        )


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
        "gradual successor master hash differs",
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
