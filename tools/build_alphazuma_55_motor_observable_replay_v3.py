"""Freeze the retry-audited actor-visible motor-state replay route."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tools import distill_alphazuma_55 as legacy
from zuma_rl.alphazuma_55 import INCLUDED_LEVELS


SCRIPT_PATH = Path(__file__).resolve()
CAMPAIGN_ID = "alphazuma-55-motor-observable-replay-s99081616-v3"
ANCHOR_SEED_BASE = 1_543_002_000
ANCHOR_ATTEMPTS_PER_LEVEL = 3
ANCHOR_SEED_LAST = (
    ANCHOR_SEED_BASE
    + 3 * ANCHOR_ATTEMPTS_PER_LEVEL * len(INCLUDED_LEVELS)
    - 1
)
STUDENT_SEED_BASE = ANCHOR_SEED_LAST + 1
STUDENT_SEED_LAST = STUDENT_SEED_BASE + 3 * len(INCLUDED_LEVELS) - 1
MODEL_SEED = 1_543_002_800
EXPECTED_V2_PREREG_SHA256 = (
    "sha256:d67a45937fbeba955f275dda83330588e1b50d24591f4cd232696b50cfac1431"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _reference(path: Path) -> dict[str, str]:
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "sha256": legacy._sha256(resolved)}


def _write_new(path: Path, value: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists():
        temporary.unlink()
        raise FileExistsError(f"preregistration already exists: {path}")
    os.replace(temporary, path)


def _predecessor_incident(root: Path) -> dict[str, Any]:
    prereg_path = (
        root
        / "diagnostics/alphazuma-55-motor-observable-replay-"
        "s99081614-preregistration-v2.json"
    )
    if legacy._sha256(prereg_path) != EXPECTED_V2_PREREG_SHA256:
        raise ValueError("V2 preregistration bytes changed")
    run_dir = Path(
        "/mnt/d/ZumaTraining/alphazuma-55-motor-observable-replay-"
        "s99081614-v2"
    )
    artifacts = {
        "preregistration": _reference(prereg_path),
        "config": _reference(run_dir / "config.json"),
        "migration": _reference(run_dir / "migration.json"),
        "training_status": _reference(run_dir / "training_status.json"),
        "failure": _reference(run_dir / "failure.json"),
        "stderr": _reference(
            Path(
                "/mnt/d/ZumaTraining/alphazuma-55-motor-observable-"
                "replay-s99081614-launch.stderr.log"
            )
        ),
    }
    failure = legacy._read_json(run_dir / "failure.json")
    status = legacy._read_json(run_dir / "training_status.json")
    migration = legacy._read_json(run_dir / "migration.json")
    forbidden = [
        run_dir / "completion.json",
        run_dir / "final_model.zip",
        *(run_dir.glob("round_*")),
    ]
    if not (
        failure.get("status") == "ERROR"
        and failure.get("error")
        == "motor-observable teacher anchor failed a level: pass=0 wins=54"
        and failure.get("formal_seed_consumption") is False
        and status.get("stage") == "COLLECTING_TEACHER_ANCHOR"
        and int(status.get("completed_anchor_passes", -1)) == 0
        and int(status.get("completed_student_rounds", -1)) == 0
        and migration.get("status") == "PASS"
        and not any(path.exists() for path in forbidden)
    ):
        raise ValueError("V2 runtime incident facts changed")
    return {
        "classification": "V2_ANCHOR_54_OF_55_ABORT_BEFORE_ANY_OPTIMIZATION",
        "observed_utc": failure["failed_utc"],
        "artifacts": artifacts,
        "facts": {
            "anchor_pass": 0,
            "wins": 54,
            "expected_wins": 55,
            "completed_anchor_passes": 0,
            "completed_student_rounds": 0,
            "model_checkpoints_created": 0,
            "formal_seed_consumption": False,
        },
    }


def _engineering_seed_reuse_evidence() -> dict[str, Any]:
    output = Path(
        "/mnt/d/ZumaTraining/alphazuma-55-motor-observable-postprocess-"
        "s99081615-v1"
    )
    status = legacy._read_json(output / "controller_status.json")
    failure = legacy._read_json(output / "failure.json")
    unexpected = [
        path
        for path in output.iterdir()
        if path.name not in {"controller_status.json", "failure.json"}
    ]
    if not (
        status.get("status") == "ERROR"
        and status.get("phase") == "ERROR"
        and status.get("formal_seed_consumption") == "NONE"
        and failure.get("error") == "motor-observable training reported failure"
        and failure.get("formal_seed_consumption") == "NONE"
        and not unexpected
    ):
        raise ValueError("predecessor postprocess consumed or wrote matrix data")
    return {
        "classification": "UNCONSUMED_TRAINING_VALIDATION_SEEDS",
        "controller_status": _reference(output / "controller_status.json"),
        "failure": _reference(output / "failure.json"),
        "matrix_artifacts_present": False,
        "formal_seed_consumption": "NONE",
    }


def build(*, project_root: Path, output: Path) -> dict[str, Any]:
    root = project_root.resolve(strict=True)
    predecessor_path = (
        root
        / "diagnostics/alphazuma-55-motor-observable-replay-"
        "s99081614-preregistration-v2.json"
    )
    predecessor = legacy._read_json(predecessor_path)
    master_path = Path(str(predecessor["master_preregistration"]["path"]))
    master = legacy._read_json(master_path)
    training = master["seed_registry"]["training"]
    if not (
        int(training["first"])
        <= ANCHOR_SEED_BASE
        <= STUDENT_SEED_LAST
        <= int(training["last"])
        and ANCHOR_SEED_LAST
        == ANCHOR_SEED_BASE
        + 3 * ANCHOR_ATTEMPTS_PER_LEVEL * len(INCLUDED_LEVELS)
        - 1
        and STUDENT_SEED_BASE == ANCHOR_SEED_LAST + 1
        and STUDENT_SEED_LAST
        == STUDENT_SEED_BASE + 3 * len(INCLUDED_LEVELS) - 1
    ):
        raise ValueError("V3 training seed interval is invalid")

    preregistration = copy.deepcopy(predecessor)
    preregistration.update(
        {
            "version": 2,
            "created_utc": _utc_now(),
            "campaign_id": CAMPAIGN_ID,
            "objective": (
                "Test actor-visible deterministic motor state after replacing "
                "the brittle all-or-nothing teacher-anchor precondition with "
                "a frozen per-level retry ledger that still requires 55/55 "
                "coverage before optimization."
            ),
            "predecessor_runtime_incident": _predecessor_incident(root),
            "engineering_seed_reuse_evidence": (
                _engineering_seed_reuse_evidence()
            ),
            "post_hoc_disclosure": {
                "trigger": "observed_v2_anchor_54_of_55",
                "classification": (
                    "post_hoc_training-precondition correction frozen before "
                    "any V3 seed or model update"
                ),
                "new_training_seeds_unseen_before_freeze": True,
                "all_attempt_samples_retained": True,
                "per_pass_55_of_55_teacher_coverage_still_required": True,
                "formal_gate_unchanged": True,
            },
            "builder": _reference(SCRIPT_PATH),
            "trainer": _reference(
                root
                / "tools/distill_alphazuma_55_motor_observable_replay_v3.py"
            ),
        }
    )
    preregistration.pop("predecessor_startup_incident", None)
    preregistration["implementation"]["v2_replay_runtime"] = _reference(
        root / "tools/distill_alphazuma_55_motor_observable_replay_v2.py"
    )
    preregistration["run"].update(
        {
            "id": "alphazuma55-motor-observable-replay-v3-5080",
            "run_dir": f"/mnt/d/ZumaTraining/{CAMPAIGN_ID}",
            "model_seed": MODEL_SEED,
            "anchor_seed_base": ANCHOR_SEED_BASE,
            "anchor_seed_last_reserved": ANCHOR_SEED_LAST,
            "student_seed_base": STUDENT_SEED_BASE,
            "student_seed_last_consumed": STUDENT_SEED_LAST,
            "anchor_retry_policy": {
                "attempts_per_level": ANCHOR_ATTEMPTS_PER_LEVEL,
                "first_attempt_mode": "paired_full55_batch",
                "retry_mode": "failed_levels_only_fixed_seed",
                "sample_retention": "all_attempts",
                "pass_requirement": (
                    "at_least_one_win_for_each_of_55_levels"
                ),
                "seed_formula": (
                    "anchor_seed_base + pass_index*(55*attempts_per_level) + "
                    "attempt_index*55 + level_index"
                ),
                "failed_attempts_are_not_hidden": True,
                "retry_selection_uses_only_current_attempt_outcome": True,
            },
        }
    )
    preregistration["run"].pop("anchor_seed_last_consumed", None)
    preregistration["authority_boundary"]["classification"] = (
        "motor_observable_retry_audited_engineering_training_only"
    )
    preregistration["frozen_engineering_validation"][
        "seed_reuse_reason"
    ] = (
        "V2 postprocess failed while waiting for training; no candidate, "
        "screen, gate, or formal matrix artifact was created"
    )
    _write_new(output, preregistration)
    return preregistration


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    value = build(project_root=args.project_root, output=args.output)
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
