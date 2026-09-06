"""Screen all all-actions motor DAgger checkpoints on fresh validation seeds."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tools import run_alphazuma_55_motor_observable_gradual_postprocess_v1 as base


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
EXPECTED_ROUTE_CAMPAIGN = (
    "alphazuma-55-motor-observable-gradual-allaim-s99081633-v1"
)
EXPECTED_TRAINER_NAME = (
    "distill_alphazuma_55_motor_observable_gradual_allaim_v1.py"
)
EXPECTED_LAUNCH_RECEIPT_BUILDER_NAME = (
    "capture_alphazuma_55_motor_observable_gradual_allaim_launch_receipt_v1.py"
)
LEGACY_TRAINER_NAME = "distill_alphazuma_55_motor_observable_gradual_v2.py"
EXPECTED_SCREEN_SEED = 1_550_002_800
EXPECTED_GATE_SEED = 1_550_002_900
PREREG_AUDIT = (
    PROJECT_ROOT
    / "diagnostics/alphazuma-55-motor-observable-gradual-allaim-"
    "s99081633-independent-prereg-audit-v1.json"
)
PREREG_AUDIT_SHA256 = (
    "sha256:61be913b30601f2e529cc8112dd18aab8b434d82a7364a794e1ff756c5d99b6d"
)
_BASE_VALIDATE = base.validate_plan
_BASE_INVENTORY = base._candidate_inventory


def _valid_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 71 and text.startswith("sha256:") and all(
        character in "0123456789abcdef" for character in text[7:]
    )


def _validate_launch_receipt(
    plan: dict[str, Any], route: dict[str, Any], route_path: Path
) -> None:
    receipt_path = base._bound(
        plan.get("training_route_launch_receipt"),
        "training route launch receipt",
    )
    receipt = base._read(receipt_path)
    bindings = receipt.get("bindings", {})
    preregistration_path = base._bound(
        bindings.get("preregistration"), "launch receipt preregistration"
    )
    trainer_path = base._bound(
        bindings.get("trainer"), "launch receipt trainer"
    )
    builder_path = base._bound(
        bindings.get("builder"), "launch receipt builder"
    )
    audit_path = base._bound(
        bindings.get("preregistration_independent_audit"),
        "launch receipt preregistration independent audit",
    )
    receipt_builder_path = base._bound(
        bindings.get("launch_receipt_builder"),
        "launch receipt builder",
    )
    route_trainer = base._bound(route.get("trainer"), "all-action trainer")
    route_builder = base._bound(route.get("builder"), "all-action builder")
    plan_route_sha256 = str(plan.get("training_route", {}).get("sha256", ""))
    run = route.get("run", {})
    run_dir = Path(str(run.get("run_dir", ""))).resolve()
    validation = receipt.get("validation", {})
    processes = receipt.get("processes", {})
    initial = receipt.get("initial_artifacts", {})
    initial_status = initial.get("training_status", {})
    logs = receipt.get("logs", {})
    resources = receipt.get("resource_snapshot", {})
    gpus = resources.get("gpus", ())
    boundary = receipt.get("authority_boundary", {})

    if not (
        receipt.get("schema")
        == "zuma-rl.alphazuma-55-motor-observable-gradual-launch-receipt"
        and receipt.get("version") == 1
        and receipt.get("status") == "LAUNCHED_AND_COLLECTING"
        and receipt.get("campaign_id") == EXPECTED_ROUTE_CAMPAIGN
        and base._parse_utc(str(receipt.get("observed_utc", "")))
        and preregistration_path == route_path
        and bindings.get("preregistration", {}).get("sha256")
        == plan_route_sha256
        and trainer_path == route_trainer
        and builder_path == route_builder
        and audit_path == PREREG_AUDIT.resolve(strict=True)
        and receipt_builder_path.name == EXPECTED_LAUNCH_RECEIPT_BUILDER_NAME
        and validation.get("status") == "VALID"
        and validation.get("observation_shape") == [22904]
        and tuple(
            float(value)
            for value in validation.get(
                "teacher_execution_probabilities", ()
            )
        )
        == base.EXPECTED_SCHEDULE
        and int(validation.get("epochs_per_round", -1)) == 12
        and validation.get("formal_seed_consumption") is False
        and isinstance(processes.get("windows_launcher_pid"), int)
        and int(processes["windows_launcher_pid"]) > 0
        and isinstance(processes.get("wsl_trainer_pid"), int)
        and int(processes["wsl_trainer_pid"]) > 0
        and processes.get("windows_launcher_alive") is True
        and processes.get("wsl_trainer_alive") is True
        and Path(str(initial.get("config", {}).get("path", ""))).resolve()
        == run_dir / "config.json"
        and Path(str(initial_status.get("path", ""))).resolve()
        == run_dir / "training_status.json"
        and _valid_sha256(initial.get("config", {}).get("sha256"))
        and _valid_sha256(initial_status.get("sha256_at_snapshot"))
        and initial_status.get("stage") == "COLLECTING"
        and int(initial_status.get("current_round", -1)) == 0
        and float(initial_status.get("teacher_execution_probability", -1.0))
        == 1.0
        and int(initial_status.get("completed_rounds", -1)) == 0
        and int(initial_status.get("expected_rounds", -1)) == 4
        and int(resources.get("wsl_available_memory_bytes", -1))
        >= 12 * 1024**3
        and int(resources.get("wsl_swap_used_bytes", -1)) == 0
        and int(resources.get("wsl_root_available_bytes", -1))
        >= 12 * 1024**3
        and int(resources.get("d_drive_available_bytes", -1)) > 0
        and len(gpus) == 2
        and [int(row.get("index", -1)) for row in gpus] == [0, 1]
        and [float(row.get("power_limit_watts", -1.0)) for row in gpus]
        == [550.0, 250.0]
        and all(int(row.get("temperature_c", 999)) < 85 for row in gpus)
        and boundary.get("engineering_training_route_only") is True
        and boundary.get("formal_selection_seed_consumption") == "NONE"
        and boundary.get("formal_final_blind_seed_consumption") == "NONE"
        and boundary.get("continuous_campaign_seed_consumption") == "NONE"
        and boundary.get("formal_candidate_authority") is False
        and boundary.get("power_restore_authority") is False
    ):
        raise ValueError("all-action training launch receipt is ineligible")

    config_path = base._bound(
        initial.get("config"), "launch receipt initial config"
    )
    if config_path != run_dir / "config.json":
        raise ValueError("all-action launch config escaped expected run dir")
    for stream in ("stdout", "stderr"):
        row = logs.get(stream, {})
        if not (
            str(row.get("path", "")).strip()
            and int(row.get("bytes_at_snapshot", -1)) >= 0
            and _valid_sha256(row.get("sha256_at_snapshot"))
        ):
            raise ValueError(f"all-action launch {stream} evidence differs")


def _validate_prereg_audit(plan: dict[str, Any], route: dict[str, Any]) -> None:
    reference = plan.get("training_route_independent_prereg_audit", {})
    path = base._bound(reference, "training route independent prereg audit")
    if path != PREREG_AUDIT.resolve(strict=True):
        raise ValueError("all-action plan binds another prereg audit")
    if base._sha256(path) != PREREG_AUDIT_SHA256:
        raise ValueError("all-action prereg audit bytes differ")
    audit = base._read(path)
    if not (
        audit.get("status") == "PASS"
        and audit.get("candidate", {}).get("sha256")
        == plan.get("training_route", {}).get("sha256")
        and audit.get("all_other_recipe_fields_byte_equivalent_after_normalization")
        is True
        and audit.get("formal_seed_consumption") == "NONE"
        and audit.get("formal_candidate_authority") is False
    ):
        raise ValueError("all-action prereg independent audit is ineligible")
    run = route.get("run", {})
    if not (
        route.get("campaign_id") == EXPECTED_ROUTE_CAMPAIGN
        and Path(str(route.get("trainer", {}).get("path", ""))).name
        == EXPECTED_TRAINER_NAME
        and run.get("aim_loss_scope") == "all_actions"
        and run.get("route_variant") == "all_actions_aim_supervision_v1"
        and run.get("all_action_aim_supervision") is True
        and route.get("experimental_lineage", {}).get("single_recipe_change")
        == "aim_loss_scope fire_only -> all_actions"
        and all(
            value is False
            for value in route.get("authority_boundary", {}).values()
        )
    ):
        raise ValueError("all-action training route contract differs")


def validate_plan(
    path: Path, expected_sha256: str | None = None
) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if expected_sha256 is not None and base._sha256(resolved) != expected_sha256:
        raise ValueError("all-action postprocess plan hash differs")
    plan = base._read(resolved)
    route_path = base._bound(plan.get("training_route"), "training route")
    route = base._read(route_path)
    _validate_prereg_audit(plan, route)
    _validate_launch_receipt(plan, route, route_path)

    original_script = base.SCRIPT_PATH
    original_campaign = base.EXPECTED_ROUTE_CAMPAIGN
    original_screen_seed = base.EXPECTED_SCREEN_SEED
    original_gate_seed = base.EXPECTED_GATE_SEED
    original_read = base._read
    original_bound = base._bound

    def read_with_legacy_lineage(candidate: Path) -> dict[str, Any]:
        value = original_read(candidate)
        if Path(candidate).resolve() != route_path:
            return value
        adjusted = copy.deepcopy(value)
        adjusted["recovery_lineage"] = {
            "training_recipe_changed": False,
            "seed_interval_changed": False,
        }
        return adjusted

    def bound_with_trainer_name(reference: Any, label: str) -> Path:
        actual = original_bound(reference, label)
        if label != "gradual trainer":
            return actual
        if actual.name != EXPECTED_TRAINER_NAME:
            raise ValueError("all-action plan binds another trainer")
        return actual.with_name(LEGACY_TRAINER_NAME)

    base.SCRIPT_PATH = SCRIPT_PATH
    base.EXPECTED_ROUTE_CAMPAIGN = EXPECTED_ROUTE_CAMPAIGN
    base.EXPECTED_SCREEN_SEED = EXPECTED_SCREEN_SEED
    base.EXPECTED_GATE_SEED = EXPECTED_GATE_SEED
    base._read = read_with_legacy_lineage
    base._bound = bound_with_trainer_name
    try:
        return _BASE_VALIDATE(resolved, expected_sha256)
    finally:
        base._bound = original_bound
        base._read = original_read
        base.EXPECTED_GATE_SEED = original_gate_seed
        base.EXPECTED_SCREEN_SEED = original_screen_seed
        base.EXPECTED_ROUTE_CAMPAIGN = original_campaign
        base.SCRIPT_PATH = original_script


def _candidate_inventory(
    *, plan: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    completion_path = Path(str(plan["expected_training_completion"])).resolve(
        strict=True
    )
    completion = base._read(completion_path)
    rounds = completion.get("rounds", ())
    if not (
        completion.get("status") == "COMPLETE"
        and completion.get("route_variant")
        == "all_actions_aim_supervision_v1"
        and completion.get("aim_loss_scope") == "all_actions"
        and completion.get("all_action_aim_supervision") is True
        and rounds
        and all(
            row.get("optimization", {}).get("aim_loss_scope") == "all_actions"
            for row in rounds
        )
        and completion.get("formal_seed_consumption") is False
        and completion.get("formal_candidate_authority") is False
    ):
        raise ValueError("all-action training completion is ineligible")
    candidates, evidence = _BASE_INVENTORY(plan=plan)
    for candidate in candidates:
        if candidate.get("promotable") is True:
            candidate["id"] = "allaim-" + str(candidate["id"])
            candidate["route_variant"] = "all_actions_aim_supervision_v1"
    return candidates, evidence


def _allaim_processes(process_finder: Any, fragment: str) -> list[int]:
    if fragment == "distill_alphazuma_55_motor_observable_replay_v2.py":
        fragment = EXPECTED_TRAINER_NAME
    return list(process_finder(fragment))


def run(*, plan_path: Path, expected_plan_sha256: str, poll_seconds: float) -> int:
    plan_path = plan_path.resolve(strict=True)
    plan = validate_plan(plan_path, expected_plan_sha256)
    original_validate = base.validate_plan
    original_inventory = base._candidate_inventory
    original_processes = base._gradual_processes
    original_script = base.SCRIPT_PATH
    original_campaign = base.EXPECTED_ROUTE_CAMPAIGN
    original_screen_seed = base.EXPECTED_SCREEN_SEED
    original_gate_seed = base.EXPECTED_GATE_SEED
    try:
        base.validate_plan = lambda _path, _hash=None: plan
        base._candidate_inventory = _candidate_inventory
        base._gradual_processes = _allaim_processes
        base.SCRIPT_PATH = SCRIPT_PATH
        base.EXPECTED_ROUTE_CAMPAIGN = EXPECTED_ROUTE_CAMPAIGN
        base.EXPECTED_SCREEN_SEED = EXPECTED_SCREEN_SEED
        base.EXPECTED_GATE_SEED = EXPECTED_GATE_SEED
        return base.run(
            plan_path=plan_path,
            expected_plan_sha256=expected_plan_sha256,
            poll_seconds=poll_seconds,
        )
    finally:
        base.EXPECTED_GATE_SEED = original_gate_seed
        base.EXPECTED_SCREEN_SEED = original_screen_seed
        base.EXPECTED_ROUTE_CAMPAIGN = original_campaign
        base.SCRIPT_PATH = original_script
        base._gradual_processes = original_processes
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
                    "plan": {
                        "path": str(plan_path),
                        "sha256": base._sha256(plan_path),
                    },
                    "screen_candidates": int(
                        plan["screen"]["expected_candidates"]
                    ),
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
