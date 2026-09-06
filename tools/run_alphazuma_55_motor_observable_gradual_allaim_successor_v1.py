"""Conditionally certify the all-actions motor-observable engineering winner."""

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

from tools import (
    audit_alphazuma_55_motor_observable_gradual_allaim_postprocess_v1
    as engineering_auditor,
)
from tools import (
    run_alphazuma_55_motor_observable_gradual_allaim_postprocess_v1
    as engineering,
)
from tools import run_alphazuma_55_motor_observable_gradual_successor_v1 as legacy
from zuma_rl.alphazuma_55 import INCLUDED_LEVELS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()
MASTER_SCHEMA = legacy.MASTER_SCHEMA
LEVEL_COUNT = legacy.LEVEL_COUNT
FINAL_ATTEMPTS = legacy.FINAL_ATTEMPTS
CONTINUOUS_CAMPAIGNS = legacy.CONTINUOUS_CAMPAIGNS
FINAL_SEED_BASE = 3_700_000_000
CONTINUOUS_SEED_BASE = 3_800_000_000
EXPECTED_CAMPAIGN = (
    "alphazuma-55-motor-observable-gradual-allaim-successor-s99081636-v1"
)
EXPECTED_ENGINEERING_CAMPAIGN = (
    "alphazuma-55-motor-observable-gradual-allaim-postprocess-s99081634-v1"
)

base = legacy.base
_require = legacy._require
_read = legacy._read
_sha256 = legacy._sha256
_artifact = legacy._artifact
_utc_now = legacy._utc_now
_parse_utc = legacy._parse_utc
_write_new = legacy._write_new
_replace_json = legacy._replace_json
successor_helpers = legacy.successor_helpers
_strict_audit_evaluation = legacy._strict_audit_evaluation
_base_engineering_result = legacy._base_engineering_result


def _implementation_paths() -> dict[str, Path]:
    return {
        "master_builder": PROJECT_ROOT
        / "tools/build_alphazuma_55_motor_observable_gradual_allaim_"
        "successor_master_v1.py",
        "controller": SCRIPT_PATH,
        "independent_auditor": PROJECT_ROOT
        / "tools/audit_alphazuma_55_motor_observable_gradual_allaim_"
        "successor_v1.py",
        "evaluator": PROJECT_ROOT
        / "tools/evaluate_alphazuma_55_motor_observable.py",
        "evaluation_helper": PROJECT_ROOT
        / "tools/run_overnight_v11_postprocess.py",
        "matrix_auditor": PROJECT_ROOT
        / "tools/audit_alphazuma_v11_frontier_result.py",
        "summarizer": PROJECT_ROOT / "tools/summarize_multimodel_evaluation.py",
        "engineering_controller": PROJECT_ROOT
        / "tools/run_alphazuma_55_motor_observable_gradual_allaim_"
        "postprocess_v1.py",
        "engineering_auditor": PROJECT_ROOT
        / "tools/audit_alphazuma_55_motor_observable_gradual_allaim_"
        "postprocess_v1.py",
    }


def _legacy_implementation_paths() -> dict[str, Path]:
    return {
        "master_builder": PROJECT_ROOT
        / "tools/build_alphazuma_55_motor_observable_gradual_"
        "successor_master_v1.py",
        "independent_auditor": PROJECT_ROOT
        / "tools/audit_alphazuma_55_motor_observable_gradual_successor_v1.py",
        "engineering_controller": PROJECT_ROOT
        / "tools/run_alphazuma_55_motor_observable_gradual_postprocess_v1.py",
        "engineering_auditor": PROJECT_ROOT
        / "tools/audit_alphazuma_55_motor_observable_gradual_postprocess_v1.py",
    }


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
        master.get("campaign_id") == EXPECTED_CAMPAIGN,
        "all-action successor campaign differs",
    )
    actual_paths = _implementation_paths()
    legacy_paths = _legacy_implementation_paths()
    original_artifact = legacy._artifact
    original_engineering = legacy.engineering
    original_script = legacy.SCRIPT_PATH
    original_final = legacy.FINAL_SEED_BASE
    original_continuous = legacy.CONTINUOUS_SEED_BASE

    def artifact_with_lineage(reference: Any, label: str) -> Path:
        actual = original_artifact(reference, label)
        if label not in legacy_paths:
            return actual
        expected = actual_paths[label].resolve(strict=True)
        _require(actual == expected, f"all-action {label} path differs")
        return legacy_paths[label].resolve(strict=True)

    legacy._artifact = artifact_with_lineage
    legacy.engineering = engineering
    legacy.SCRIPT_PATH = SCRIPT_PATH
    legacy.FINAL_SEED_BASE = FINAL_SEED_BASE
    legacy.CONTINUOUS_SEED_BASE = CONTINUOUS_SEED_BASE
    try:
        contract = legacy._load_master(path)
    finally:
        legacy.CONTINUOUS_SEED_BASE = original_continuous
        legacy.FINAL_SEED_BASE = original_final
        legacy.SCRIPT_PATH = original_script
        legacy.engineering = original_engineering
        legacy._artifact = original_artifact

    plan = contract["engineering_plan"]
    _require(
        plan.get("campaign_id") == EXPECTED_ENGINEERING_CAMPAIGN,
        "all-action engineering campaign differs",
    )
    contract["implementation"].update(
        {
            name: path.resolve(strict=True)
            for name, path in actual_paths.items()
        }
    )
    contract["registry"] = _validate_registry(
        registry_path=Path(
            str(
                master.get("formal_seed_registry", {}).get(
                    "active_registry_path", ""
                )
            )
        ),
        master_path=path,
    )
    return contract


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
        "all-action successor master hash differs",
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
                    "engineering_audit": str(
                        contract["engineering_audit_path"]
                    ),
                    "final_blind_attempts": LEVEL_COUNT * FINAL_ATTEMPTS,
                    "continuous_attempts": (
                        LEVEL_COUNT * CONTINUOUS_CAMPAIGNS
                    ),
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
