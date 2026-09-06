"""Run the frozen gradual-motor checkpoint screen and full55 engineering gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tools import run_alphazuma_55_motor_observable_postprocess_v1 as base


SCRIPT_PATH = Path(__file__).resolve()
PLAN_SCHEMA = base.PLAN_SCHEMA
SCREEN_LEVEL_IDS = base.SCREEN_LEVEL_IDS
EXPECTED_SCREEN_CANDIDATES = 17
MAXIMUM_GATE_CANDIDATES = 5
EXPECTED_SCHEDULE = (1.0, 0.9, 0.7, 0.5)
EXPECTED_CHECKPOINT_EPOCHS = (3, 6, 9, 12)
EXPECTED_ROUTE_CAMPAIGN = "alphazuma-55-motor-observable-gradual-s99081629-v1"
EXPECTED_SCREEN_SEED = 1_550_002_600
EXPECTED_GATE_SEED = 1_550_002_700

_utc_now = base._utc_now
_sha256 = base._sha256
_read = base._read
_write_atomic = base._write_atomic
_write_new = base._write_new
_bound = base._bound
_parse_utc = base._parse_utc
_shard_bounds = base._shard_bounds
_manifest = base._manifest
_evaluation_preregistration = base._evaluation_preregistration
_rank_key = base._rank_key
audit_matrix = base.audit_matrix


def validate_plan(
    path: Path, expected_sha256: str | None = None
) -> dict[str, Any]:
    path = path.resolve(strict=True)
    if expected_sha256 is not None and _sha256(path) != expected_sha256:
        raise ValueError("gradual-motor postprocess plan hash differs")
    plan = _read(path)
    if not (
        plan.get("schema") == PLAN_SCHEMA
        and plan.get("version") == 1
        and plan.get("status")
        == "FROZEN_DURING_TRAINING_BEFORE_CHECKPOINT_INFERENCE"
    ):
        raise ValueError("unexpected gradual-motor postprocess plan")

    implementation = plan.get("implementation", {})
    controller_path = _bound(implementation.get("controller"), "controller")
    if controller_path != SCRIPT_PATH:
        raise ValueError("gradual-motor plan binds another controller")
    for name, reference in implementation.items():
        _bound(reference, f"implementation {name}")

    master_path = _bound(plan.get("master_preregistration"), "master")
    master = _read(master_path)
    route_path = _bound(plan.get("training_route"), "training route")
    route = _read(route_path)
    template_path = _bound(plan.get("environment_template"), "environment template")
    template = _read(template_path)
    trainer_path = _bound(route.get("trainer"), "gradual trainer")
    _bound(route.get("builder"), "gradual builder")
    _bound(route.get("source", {}).get("completion"), "source completion")
    _bound(route.get("source", {}).get("model"), "source model")
    for name, reference in route.get("implementation", {}).items():
        _bound(reference, f"gradual implementation {name}")
    run = route.get("run", {})
    boundary = route.get("authority_boundary", {})
    if not (
        master.get("schema")
        == "zuma-rl.alphazuma-55-weekend-master-preregistration"
        and route.get("schema")
        == "zuma-rl.alphazuma-55-motor-observable-gradual-preregistration"
        and route.get("version") == 1
        and route.get("status") == "FROZEN_BEFORE_TRAINING"
        and route.get("campaign_id") == EXPECTED_ROUTE_CAMPAIGN
        and trainer_path.name
        == "distill_alphazuma_55_motor_observable_gradual_v2.py"
        and run.get("motor_observation_profile") == "motor-observable-v1"
        and tuple(float(value) for value in run.get(
            "teacher_execution_probabilities", ()
        ))
        == EXPECTED_SCHEDULE
        and int(run.get("rounds", -1)) == len(EXPECTED_SCHEDULE)
        and int(run.get("epochs_per_round", -1)) == 12
        and int(run.get("checkpoint_interval_epochs", -1)) == 3
        and run.get("final_runtime_teacher_calls_forbidden") is True
        and route.get("recovery_lineage", {}).get("training_recipe_changed")
        is False
        and route.get("recovery_lineage", {}).get("seed_interval_changed")
        is False
        and all(value is False for value in boundary.values())
        and template.get("schema")
        == "zuma-rl.zero-shot-multilevel-preregistration"
    ):
        raise ValueError("gradual-motor training contract changed")

    full_levels = plan.get("full55_levels", [])
    screen_levels = plan.get("screen_levels", [])
    if len(full_levels) != 55 or tuple(
        row.get("id") for row in screen_levels
    ) != SCREEN_LEVEL_IDS:
        raise ValueError("gradual-motor level inventory changed")
    full_ids = {str(row.get("id")) for row in full_levels}
    if not set(SCREEN_LEVEL_IDS).issubset(full_ids):
        raise ValueError("gradual-motor screen levels escaped full55")

    screen = plan.get("screen", {})
    gate = plan.get("full55_gate", {})
    if not (
        int(screen.get("base_seed", -1)) == EXPECTED_SCREEN_SEED
        and int(screen.get("last_seed", -1))
        == EXPECTED_SCREEN_SEED + len(screen_levels) - 1
        and int(screen.get("expected_candidates", -1))
        == EXPECTED_SCREEN_CANDIDATES
        and int(screen.get("expected_attempts", -1))
        == EXPECTED_SCREEN_CANDIDATES * len(screen_levels)
        and int(screen.get("select_top", -1)) == MAXIMUM_GATE_CANDIDATES
        and int(gate.get("base_seed", -1)) == EXPECTED_GATE_SEED
        and int(gate.get("last_seed", -1))
        == EXPECTED_GATE_SEED + len(full_levels) - 1
        and int(gate.get("maximum_candidates", -1))
        == MAXIMUM_GATE_CANDIDATES
        and int(gate.get("maximum_expected_attempts", -1))
        == MAXIMUM_GATE_CANDIDATES * len(full_levels)
        and int(gate.get("minimum_wins", -1)) == 35
        and int(gate.get("minimum_cleared_levels", -1)) == 35
    ):
        raise ValueError("gradual-motor matrix contract changed")
    registry = master["seed_registry"]["training_validation"]
    for first, last in (
        (int(screen["base_seed"]), int(screen["last_seed"])),
        (int(gate["base_seed"]), int(gate["last_seed"])),
    ):
        if not int(registry["first"]) <= first <= last <= int(registry["last"]):
            raise ValueError("gradual-motor seeds escaped training_validation")
    if int(screen["last_seed"]) >= int(gate["base_seed"]):
        raise ValueError("gradual-motor screen and gate seeds overlap")
    post_boundary = plan.get("authority_boundary", {})
    for name in (
        "formal_selection_seed_consumption",
        "formal_final_blind_seed_consumption",
        "continuous_campaign_seed_consumption",
    ):
        if post_boundary.get(name) is not False:
            raise ValueError(f"gradual-motor postprocess authority changed: {name}")
    rule = plan.get("candidate_inventory_rule", {})
    if not (
        rule.get("source_baseline_included") is True
        and rule.get("source_baseline_promotable") is False
        and tuple(int(value) for value in rule.get("round_indices", ()))
        == tuple(range(4))
        and tuple(int(value) for value in rule.get("checkpoint_epochs", ()))
        == EXPECTED_CHECKPOINT_EPOCHS
        and rule.get("final_model_duplicate_excluded") is True
    ):
        raise ValueError("gradual-motor candidate rule changed")
    return plan


def _candidate_inventory(
    *, plan: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    route_path = _bound(plan["training_route"], "training route")
    route = _read(route_path)
    completion_path = Path(str(plan["expected_training_completion"])).resolve(
        strict=True
    )
    completion = _read(completion_path)
    if not (
        completion.get("schema")
        == "zuma-rl.alphazuma-55-motor-observable-gradual-completion"
        and completion.get("status") == "COMPLETE"
        and completion.get("motor_observation_profile") == "motor-observable-v1"
        and tuple(float(value) for value in completion.get(
            "teacher_execution_probabilities", ()
        ))
        == EXPECTED_SCHEDULE
        and completion.get("state_distribution_semantics")
        == "gradual_teacher_student_dagger"
        and completion.get("final_runtime_teacher_calls_forbidden") is True
        and completion.get("formal_seed_consumption") is False
        and completion.get("formal_candidate_authority") is False
    ):
        raise ValueError("gradual-motor training completion is ineligible")
    source_path = _bound(route["source"]["model"], "source model")
    candidates: list[dict[str, Any]] = [
        {
            "id": "motor-v3-source-final",
            "training_steps": 0,
            "path": str(source_path),
            "sha256": _sha256(source_path),
            "observation_profile": "motor-observable-v1",
            "promotable": False,
            "candidate_order": 0,
            "source": "frozen_v3_source_baseline",
        }
    ]
    rounds = completion.get("rounds", [])
    if len(rounds) != len(EXPECTED_SCHEDULE):
        raise ValueError("gradual-motor completion round count changed")
    cumulative_steps = 0
    for expected_round, row in enumerate(rounds):
        if not (
            int(row.get("round_index", -1)) == expected_round
            and float(row.get("teacher_execution_probability", -1.0))
            == EXPECTED_SCHEDULE[expected_round]
        ):
            raise ValueError("gradual-motor completion round order changed")
        episodes = row.get("collection", {}).get("episodes", [])
        if len(episodes) != 55:
            raise ValueError("gradual-motor round is not a full55 collection")
        cumulative_steps += sum(int(episode["steps"]) for episode in episodes)
        checkpoints = row.get("optimization", {}).get("checkpoints", [])
        if len(checkpoints) != len(EXPECTED_CHECKPOINT_EPOCHS):
            raise ValueError("gradual-motor checkpoint count changed")
        for expected_epoch, checkpoint in zip(
            EXPECTED_CHECKPOINT_EPOCHS, checkpoints, strict=True
        ):
            if int(checkpoint.get("epoch", -1)) != expected_epoch:
                raise ValueError("gradual-motor checkpoint epoch changed")
            model_path = Path(str(checkpoint.get("path", ""))).resolve(strict=True)
            model_hash = _sha256(model_path)
            if checkpoint.get("sha256") != model_hash:
                raise ValueError("gradual-motor checkpoint bytes changed")
            candidates.append(
                {
                    "id": f"gradual-r{expected_round:02d}-e{expected_epoch:02d}",
                    "training_steps": cumulative_steps,
                    "path": str(model_path),
                    "sha256": model_hash,
                    "observation_profile": "motor-observable-v1",
                    "promotable": True,
                    "candidate_order": len(candidates),
                    "round_index": expected_round,
                    "epoch": expected_epoch,
                    "teacher_execution_probability": EXPECTED_SCHEDULE[
                        expected_round
                    ],
                }
            )
    if len(candidates) != EXPECTED_SCREEN_CANDIDATES:
        raise ValueError("gradual-motor candidate inventory changed")
    return candidates, {
        "path": str(completion_path),
        "sha256": _sha256(completion_path),
    }


def _gradual_processes(process_finder: Any, fragment: str) -> list[int]:
    if fragment == "distill_alphazuma_55_motor_observable_replay_v2.py":
        fragment = "distill_alphazuma_55_motor_observable_gradual_v2.py"
    return list(process_finder(fragment))


def run(*, plan_path: Path, expected_plan_sha256: str, poll_seconds: float) -> int:
    plan_path = plan_path.resolve(strict=True)
    plan = validate_plan(plan_path, expected_plan_sha256)
    original_validate = base.validate_plan
    original_inventory = base._candidate_inventory
    original_processes = base._processes_containing
    original_screen_candidates = base.EXPECTED_SCREEN_CANDIDATES
    original_gate_candidates = base.MAXIMUM_GATE_CANDIDATES
    try:
        base.validate_plan = lambda _path, _hash=None: plan
        base._candidate_inventory = _candidate_inventory
        base._processes_containing = lambda fragment: _gradual_processes(
            original_processes, fragment
        )
        base.EXPECTED_SCREEN_CANDIDATES = EXPECTED_SCREEN_CANDIDATES
        base.MAXIMUM_GATE_CANDIDATES = MAXIMUM_GATE_CANDIDATES
        return base.run(
            plan_path=plan_path,
            expected_plan_sha256=expected_plan_sha256,
            poll_seconds=poll_seconds,
        )
    finally:
        base.MAXIMUM_GATE_CANDIDATES = original_gate_candidates
        base.EXPECTED_SCREEN_CANDIDATES = original_screen_candidates
        base._processes_containing = original_processes
        base._candidate_inventory = original_inventory
        base.validate_plan = original_validate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    plan_path = args.plan.expanduser().resolve(strict=True)
    if args.validate_only:
        plan = validate_plan(plan_path, str(args.expected_plan_sha256))
        print(
            json.dumps(
                {
                    "status": "VALID",
                    "plan": {"path": str(plan_path), "sha256": _sha256(plan_path)},
                    "screen_candidates": int(plan["screen"]["expected_candidates"]),
                    "screen_attempts": int(plan["screen"]["expected_attempts"]),
                    "maximum_full55_attempts": int(
                        plan["full55_gate"]["maximum_expected_attempts"]
                    ),
                    "formal_seed_consumption": False,
                },
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        )
        return 0
    return run(
        plan_path=plan_path,
        expected_plan_sha256=str(args.expected_plan_sha256),
        poll_seconds=max(1.0, float(args.poll_seconds)),
    )


if __name__ == "__main__":
    raise SystemExit(main())
