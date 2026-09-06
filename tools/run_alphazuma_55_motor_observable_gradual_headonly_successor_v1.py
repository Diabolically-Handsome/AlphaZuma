"""Conditionally certify the frozen-backbone head-only engineering winner."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys
from typing import Any, Iterator

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tools import (
    audit_alphazuma_55_motor_observable_gradual_headonly_postprocess_v1
    as engineering_auditor,
)
from tools import (
    run_alphazuma_55_motor_observable_gradual_headonly_postprocess_v1
    as engineering,
)
from tools import (
    run_alphazuma_55_motor_observable_gradual_allaim_successor_v1 as legacy,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()
MASTER_SCHEMA = legacy.MASTER_SCHEMA
LEVEL_COUNT = legacy.LEVEL_COUNT
FINAL_ATTEMPTS = legacy.FINAL_ATTEMPTS
CONTINUOUS_CAMPAIGNS = legacy.CONTINUOUS_CAMPAIGNS
FINAL_SEED_BASE = 3_900_000_000
CONTINUOUS_SEED_BASE = 4_000_000_000
EXPECTED_CAMPAIGN = (
    "alphazuma-55-motor-observable-gradual-headonly-successor-s99081644-v1"
)
EXPECTED_ENGINEERING_CAMPAIGN = (
    "alphazuma-55-motor-observable-gradual-headonly-postprocess-s99081643-v2"
)

base = legacy.base
_require = legacy._require
_read = legacy._read
_sha256 = legacy._sha256
_utc_now = legacy._utc_now
_write_new = legacy._write_new
_replace_json = legacy._replace_json
successor_helpers = legacy.successor_helpers
_strict_audit_evaluation = legacy._strict_audit_evaluation


def _implementation_paths() -> dict[str, Path]:
    return {
        "master_builder": PROJECT_ROOT
        / "tools/build_alphazuma_55_motor_observable_gradual_headonly_"
        "successor_master_v1.py",
        "controller": SCRIPT_PATH,
        "independent_auditor": PROJECT_ROOT
        / "tools/audit_alphazuma_55_motor_observable_gradual_headonly_"
        "successor_v1.py",
        "evaluator": PROJECT_ROOT
        / "tools/evaluate_alphazuma_55_motor_observable.py",
        "evaluation_helper": PROJECT_ROOT
        / "tools/run_overnight_v11_postprocess.py",
        "matrix_auditor": PROJECT_ROOT
        / "tools/audit_alphazuma_v11_frontier_result.py",
        "summarizer": PROJECT_ROOT
        / "tools/summarize_multimodel_evaluation.py",
        "engineering_controller": engineering.SCRIPT_PATH,
        "engineering_auditor": engineering_auditor.SCRIPT_PATH,
    }


@contextmanager
def _patched_legacy() -> Iterator[None]:
    names = {
        "SCRIPT_PATH": SCRIPT_PATH,
        "EXPECTED_CAMPAIGN": EXPECTED_CAMPAIGN,
        "EXPECTED_ENGINEERING_CAMPAIGN": EXPECTED_ENGINEERING_CAMPAIGN,
        "FINAL_SEED_BASE": FINAL_SEED_BASE,
        "CONTINUOUS_SEED_BASE": CONTINUOUS_SEED_BASE,
        "engineering": engineering,
        "engineering_auditor": engineering_auditor,
        "_implementation_paths": _implementation_paths,
    }
    originals = {name: getattr(legacy, name) for name in names}
    try:
        for name, value in names.items():
            setattr(legacy, name, value)
        yield
    finally:
        for name, value in originals.items():
            setattr(legacy, name, value)


def _load_master(path: Path) -> dict[str, Any]:
    with _patched_legacy():
        return legacy._load_master(path)


def run(*, master_path: Path, original_root: Path, poll_seconds: float) -> int:
    with _patched_legacy():
        return legacy.run(
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
        "head-only successor master hash differs",
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
