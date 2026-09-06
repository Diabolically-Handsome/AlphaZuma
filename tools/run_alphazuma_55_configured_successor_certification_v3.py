"""Run raw-intent DAgger certification with the frozen V3 contract."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools import run_alphazuma_55_configured_successor_certification as base
from tools import run_alphazuma_55_configured_successor_certification_v2 as v2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()
EXPECTED_RANKING = list(v2.EXPECTED_RANKING)
_sha256 = base._sha256


def _artifact(path: Path) -> dict[str, str]:
    path = path.resolve(strict=True)
    return {"path": str(path), "sha256": _sha256(path)}


def _load_master(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    actual_master = base._read(path)
    implementation = actual_master.get("implementation", {})
    expected_actual = {
        "master_builder": PROJECT_ROOT
        / "tools/build_alphazuma_55_configured_successor_master_v3.py",
        "controller": SCRIPT_PATH,
        "watcher": PROJECT_ROOT
        / "tools/watch_alphazuma_55_configured_successor_certification_v3.py",
        "independent_auditor": PROJECT_ROOT
        / "tools/audit_alphazuma_55_configured_successor_result_v3.py",
    }
    for name, expected in expected_actual.items():
        reference = implementation.get(name, {})
        resolved = Path(str(reference.get("path", ""))).resolve(strict=True)
        base._require(
            resolved == expected.resolve(strict=True)
            and reference.get("sha256") == _sha256(resolved),
            f"configured successor V3 {name} binding differs",
        )
    preflight = actual_master.get("preflight_correction", {})
    base._require(
        preflight.get("formal_seed_consumption_before_correction") is False
        and preflight.get("policy_inference_before_correction") is False
        and preflight.get("correction") == "forward poll_seconds to base.run",
        "configured successor V3 preflight correction differs",
    )
    plan_path = Path(
        actual_master["successor"]["engineering_validation"]["plan"]["path"]
    ).resolve(strict=True)
    actual_plan = base._read(plan_path)
    original_read = base._read
    original_v2_script = v2.SCRIPT_PATH

    def adapted_read(candidate: Path) -> dict[str, Any]:
        resolved = Path(candidate).resolve(strict=True)
        value = original_read(resolved)
        if resolved == path:
            value = copy.deepcopy(value)
            translated = value["implementation"]
            translated["master_builder"] = _artifact(
                PROJECT_ROOT
                / "tools/build_alphazuma_55_configured_successor_master_v2.py"
            )
            translated["watcher"] = _artifact(
                PROJECT_ROOT
                / "tools/watch_alphazuma_55_configured_successor_certification_v2.py"
            )
            translated["independent_auditor"] = _artifact(
                PROJECT_ROOT
                / "tools/audit_alphazuma_55_configured_successor_result_v2.py"
            )
        return value

    try:
        base._read = adapted_read
        v2.SCRIPT_PATH = SCRIPT_PATH
        contract = v2._load_master(path)
    finally:
        base._read = original_read
        v2.SCRIPT_PATH = original_v2_script
    contract["master"] = actual_master
    contract["engineering_plan"] = actual_plan
    contract["ranking"] = list(EXPECTED_RANKING)
    return contract


def run(
    *,
    master_path: Path,
    original_root: Path,
    poll_seconds: float,
) -> int:
    base._require(poll_seconds > 0, "poll_seconds must be positive")
    original_load = base._load_master
    original_decision = base._engineering_decision
    original_ranking = base.SUPPORTED_RANKING
    try:
        base._load_master = _load_master
        base._engineering_decision = v2._engineering_decision
        base.SUPPORTED_RANKING = list(EXPECTED_RANKING)
        return base.run(
            master_path=master_path,
            original_root=original_root,
            poll_seconds=poll_seconds,
        )
    finally:
        base._load_master = original_load
        base._engineering_decision = original_decision
        base.SUPPORTED_RANKING = original_ranking


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-preregistration", required=True, type=Path)
    parser.add_argument("--original-root", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(
        master_path=args.master_preregistration.expanduser(),
        original_root=args.original_root.expanduser(),
        poll_seconds=args.poll_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
