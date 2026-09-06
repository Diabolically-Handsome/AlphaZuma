"""Freeze delayed launch of the wide polar run after the active DAgger run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools.build_alphazuma_55_eval_contract import _read_json, _sha256


SCRIPT_PATH = Path(__file__).resolve()


def _ref(path: Path) -> dict[str, str]:
    path = path.resolve(strict=True)
    return {"path": str(path), "sha256": _sha256(path)}


def build(
    *, wide_preregistration: Path, expected_wide_sha256: str,
    dagger_preregistration: Path, expected_dagger_sha256: str,
    source_trainer_pid: int, original_root: Path, status_root: Path
) -> dict[str, Any]:
    wide_preregistration = wide_preregistration.resolve(strict=True)
    dagger_preregistration = dagger_preregistration.resolve(strict=True)
    original_root = original_root.resolve(strict=True)
    if _sha256(wide_preregistration) != expected_wide_sha256:
        raise ValueError("wide preregistration hash differs")
    if _sha256(dagger_preregistration) != expected_dagger_sha256:
        raise ValueError("DAgger preregistration hash differs")
    wide = _read_json(wide_preregistration)
    dagger = _read_json(dagger_preregistration)
    if (
        wide.get("schema")
        != "zuma-rl.alphazuma-55-polar-wide-distillation-preregistration"
        or wide.get("status") != "FROZEN_BEFORE_TRAINING"
    ):
        raise ValueError("unexpected wide preregistration")
    if (
        dagger.get("schema")
        != "zuma-rl.alphazuma-55-polar-dagger-preregistration"
        or dagger.get("status") != "FROZEN_BEFORE_TRAINING"
    ):
        raise ValueError("unexpected DAgger preregistration")
    wide_run = Path(str(wide["run"]["run_dir"])).resolve()
    dagger_run = Path(str(dagger["run"]["run_dir"])).resolve()
    status_root = status_root.resolve()
    for path in (wide_run, status_root):
        if path.exists():
            raise FileExistsError(f"delayed wide output already exists: {path}")
    root = SCRIPT_PATH.parents[1]
    watcher = root / "tools/watch_alphazuma_55_polar_wide.py"
    return {
        "schema": "zuma-rl.alphazuma-55-polar-wide-launch-plan",
        "version": 1,
        "status": "FROZEN_BEFORE_WAIT",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "wide_preregistration": _ref(wide_preregistration),
        "source_dagger_preregistration": _ref(dagger_preregistration),
        "source_terminal": {
            "completion": str(dagger_run / "completion.json"),
            "failure": str(dagger_run / "failure.json"),
            "trainer_pid_at_registration": int(source_trainer_pid),
            "launch_after_either_terminal_receipt_and_process_exit": True,
        },
        "target": {
            "run_dir": str(wide_run),
            "completion": str(wide_run / "completion.json"),
            "failure": str(wide_run / "failure.json"),
        },
        "original_root": str(original_root),
        "outputs": {"status_root": str(status_root)},
        "implementation": {
            "builder": _ref(SCRIPT_PATH),
            "watcher": _ref(watcher),
            "trainer": wide["trainer"],
        },
        "authority_boundary": {
            "formal_seed_consumption": False,
            "current_campaign_candidate_authority": False,
            "s99081535_successor_candidate_authority": False,
            "power_restore_authority": False,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wide-preregistration", required=True, type=Path)
    parser.add_argument("--expected-wide-sha256", required=True)
    parser.add_argument("--dagger-preregistration", required=True, type=Path)
    parser.add_argument("--expected-dagger-sha256", required=True)
    parser.add_argument("--source-trainer-pid", required=True, type=int)
    parser.add_argument("--original-root", required=True, type=Path)
    parser.add_argument("--status-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite wide launch plan: {output}")
    value = build(
        wide_preregistration=args.wide_preregistration.expanduser(),
        expected_wide_sha256=str(args.expected_wide_sha256),
        dagger_preregistration=args.dagger_preregistration.expanduser(),
        expected_dagger_sha256=str(args.expected_dagger_sha256),
        source_trainer_pid=args.source_trainer_pid,
        original_root=args.original_root.expanduser(),
        status_root=args.status_root.expanduser(),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {
                "status": value["status"],
                "output": {"path": str(output), "sha256": _sha256(output)},
                "source_terminal": value["source_terminal"],
                "target": value["target"],
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
