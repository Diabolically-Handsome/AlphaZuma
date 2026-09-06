"""Screen head-only DAgger checkpoints with canonical timeout support."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tools import (
    audit_alphazuma_55_motor_observable_timeout_recovery_v1 as timeout_reader,
)
from tools import (
    run_alphazuma_55_motor_observable_gradual_allaim_postprocess_v1 as legacy,
)


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
EXPECTED_ROUTE_CAMPAIGN = (
    "alphazuma-55-motor-observable-gradual-headonly-s99081642-v1"
)
EXPECTED_TRAINER_NAME = (
    "distill_alphazuma_55_motor_observable_gradual_headonly_v1.py"
)
EXPECTED_LAUNCH_RECEIPT_BUILDER_NAME = (
    "capture_alphazuma_55_motor_observable_gradual_headonly_"
    "launch_receipt_v1.py"
)
LEGACY_TRAINER_NAME = "distill_alphazuma_55_motor_observable_gradual_v2.py"
EXPECTED_SCREEN_SEED = 1_550_003_000
EXPECTED_GATE_SEED = 1_550_003_100
PREREG_AUDIT = (
    PROJECT_ROOT
    / "diagnostics/alphazuma-55-motor-observable-gradual-headonly-"
    "s99081642-independent-prereg-audit-v1.json"
)
PREREG_AUDIT_SHA256 = (
    "sha256:9efd8a02d81df0cce8cd9f339d3f16faf1012b8ccca358194bdeafef5bd5be09"
)
MAXIMUM_GATE_CANDIDATES = legacy.base.MAXIMUM_GATE_CANDIDATES
audit_matrix = timeout_reader.audit_matrix_timeout_compatible
_read = legacy.base._read
_sha256 = legacy.base._sha256
_utc_now = legacy.base._utc_now
_write_new = legacy.base._write_new


def _valid_sha256(value: Any) -> bool:
    text = str(value)
    return len(text) == 71 and text.startswith("sha256:") and all(
        character in "0123456789abcdef" for character in text[7:]
    )


def _validate_launch_receipt(
    plan: dict[str, Any], route: dict[str, Any], route_path: Path
) -> None:
    receipt_path = legacy.base._bound(
        plan.get("training_route_launch_receipt"),
        "training route launch receipt",
    )
    receipt = legacy.base._read(receipt_path)
    bindings = receipt.get("bindings", {})
    preregistration_path = legacy.base._bound(
        bindings.get("preregistration"), "launch receipt preregistration"
    )
    trainer_path = legacy.base._bound(
        bindings.get("trainer"), "launch receipt trainer"
    )
    builder_path = legacy.base._bound(
        bindings.get("builder"), "launch receipt builder"
    )
    audit_path = legacy.base._bound(
        bindings.get("preregistration_independent_audit"),
        "launch receipt preregistration independent audit",
    )
    receipt_builder_path = legacy.base._bound(
        bindings.get("launch_receipt_builder"),
        "launch receipt builder",
    )
    route_trainer = legacy.base._bound(route.get("trainer"), "head-only trainer")
    route_builder = legacy.base._bound(route.get("builder"), "head-only builder")
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
        and legacy.base._parse_utc(str(receipt.get("observed_utc", "")))
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
            for value in validation.get("teacher_execution_probabilities", ())
        )
        == legacy.base.EXPECTED_SCHEDULE
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
        and boundary.get("training_seed_reuse_disclosed") is True
        and boundary.get("formal_selection_seed_consumption") == "NONE"
        and boundary.get("formal_final_blind_seed_consumption") == "NONE"
        and boundary.get("continuous_campaign_seed_consumption") == "NONE"
        and boundary.get("formal_candidate_authority") is False
        and boundary.get("power_restore_authority") is False
    ):
        raise ValueError("head-only training launch receipt is ineligible")

    config_path = legacy.base._bound(
        initial.get("config"), "launch receipt initial config"
    )
    if config_path != run_dir / "config.json":
        raise ValueError("head-only launch config escaped expected run dir")
    initial_config = legacy.base._read(config_path)
    initial_run = initial_config.get("run", {})
    if not (
        initial_run.get("aim_loss_scope") == "fire_only"
        and initial_run.get("route_variant")
        == "action_head_only_fire_aim_v1"
        and initial_run.get("optimizer_parameter_scope") == "action_net_only"
        and initial_run.get("frozen_backbone") is True
    ):
        raise ValueError("head-only initial launch config contract differs")
    for stream in ("stdout", "stderr"):
        row = logs.get(stream, {})
        if not (
            str(row.get("path", "")).strip()
            and int(row.get("bytes_at_snapshot", -1)) >= 0
            and _valid_sha256(row.get("sha256_at_snapshot"))
        ):
            raise ValueError(f"head-only launch {stream} evidence differs")


def _validate_prereg_audit(
    plan: dict[str, Any], route: dict[str, Any]
) -> None:
    reference = plan.get("training_route_independent_prereg_audit", {})
    path = legacy.base._bound(
        reference, "training route independent prereg audit"
    )
    if path != PREREG_AUDIT.resolve(strict=True):
        raise ValueError("head-only plan binds another prereg audit")
    if legacy.base._sha256(path) != PREREG_AUDIT_SHA256:
        raise ValueError("head-only prereg audit bytes differ")
    audit = legacy.base._read(path)
    run = route.get("run", {})
    registry = route.get("seed_registry", {})
    if not (
        audit.get("status") == "PASS"
        and audit.get("candidate", {}).get("sha256")
        == plan.get("training_route", {}).get("sha256")
        and audit.get("all_other_recipe_fields_byte_equivalent_after_normalization")
        is True
        and audit.get("formal_seed_consumption") == "NONE"
        and audit.get("formal_candidate_authority") is False
        and route.get("campaign_id") == EXPECTED_ROUTE_CAMPAIGN
        and Path(str(route.get("trainer", {}).get("path", ""))).name
        == EXPECTED_TRAINER_NAME
        and run.get("aim_loss_scope") == "fire_only"
        and run.get("route_variant") == "action_head_only_fire_aim_v1"
        and run.get("optimizer_parameter_scope") == "action_net_only"
        and tuple(run.get("trainable_parameter_names", ()))
        == ("action_net.weight", "action_net.bias")
        and run.get("frozen_backbone") is True
        and registry.get("fresh_training_randomness_claim") is False
        and registry.get("formal_selection_seed_consumption") == "NONE"
        and all(
            value is False
            for value in route.get("authority_boundary", {}).values()
        )
    ):
        raise ValueError("head-only training preregistration is ineligible")


def validate_plan(
    path: Path, expected_sha256: str | None = None
) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    original_script = legacy.SCRIPT_PATH
    original_campaign = legacy.EXPECTED_ROUTE_CAMPAIGN
    original_trainer = legacy.EXPECTED_TRAINER_NAME
    original_receipt_builder = legacy.EXPECTED_LAUNCH_RECEIPT_BUILDER_NAME
    original_screen_seed = legacy.EXPECTED_SCREEN_SEED
    original_gate_seed = legacy.EXPECTED_GATE_SEED
    original_audit_path = legacy.PREREG_AUDIT
    original_audit_hash = legacy.PREREG_AUDIT_SHA256
    original_prereg_audit = legacy._validate_prereg_audit
    original_launch_receipt = legacy._validate_launch_receipt
    try:
        legacy.SCRIPT_PATH = SCRIPT_PATH
        legacy.EXPECTED_ROUTE_CAMPAIGN = EXPECTED_ROUTE_CAMPAIGN
        legacy.EXPECTED_TRAINER_NAME = EXPECTED_TRAINER_NAME
        legacy.EXPECTED_LAUNCH_RECEIPT_BUILDER_NAME = (
            EXPECTED_LAUNCH_RECEIPT_BUILDER_NAME
        )
        legacy.EXPECTED_SCREEN_SEED = EXPECTED_SCREEN_SEED
        legacy.EXPECTED_GATE_SEED = EXPECTED_GATE_SEED
        legacy.PREREG_AUDIT = PREREG_AUDIT
        legacy.PREREG_AUDIT_SHA256 = PREREG_AUDIT_SHA256
        legacy._validate_prereg_audit = _validate_prereg_audit
        legacy._validate_launch_receipt = _validate_launch_receipt
        plan = legacy.validate_plan(resolved, expected_sha256)
    finally:
        legacy._validate_launch_receipt = original_launch_receipt
        legacy._validate_prereg_audit = original_prereg_audit
        legacy.PREREG_AUDIT_SHA256 = original_audit_hash
        legacy.PREREG_AUDIT = original_audit_path
        legacy.EXPECTED_GATE_SEED = original_gate_seed
        legacy.EXPECTED_SCREEN_SEED = original_screen_seed
        legacy.EXPECTED_LAUNCH_RECEIPT_BUILDER_NAME = original_receipt_builder
        legacy.EXPECTED_TRAINER_NAME = original_trainer
        legacy.EXPECTED_ROUTE_CAMPAIGN = original_campaign
        legacy.SCRIPT_PATH = original_script

    correction = plan.get("timeout_reader_correction", {})
    auditor_ref = plan.get("implementation", {}).get(
        "timeout_compatible_matrix_auditor", {}
    )
    auditor_path = Path(str(auditor_ref.get("path", ""))).resolve(strict=True)
    disclosure = plan.get("fresh_seed_disclosure", {})
    if not (
        correction.get("classification")
        == "PRE_EVALUATION_READER_CONTRACT_CORRECTION"
        and correction.get("inference_already_consumed") is False
        and correction.get("seed_ranges_changed") is False
        and correction.get("promotion_gate_changed") is False
        and correction.get("canonical_timeout_tuple")
        == {
            "outcome": None,
            "time_limit_truncated": True,
            "ticks_equals_max_ticks": True,
        }
        and auditor_path == timeout_reader.SCRIPT_PATH.resolve(strict=True)
        and auditor_ref.get("sha256") == _sha256(auditor_path)
        and disclosure.get("screen")
        == [EXPECTED_SCREEN_SEED, EXPECTED_SCREEN_SEED + 11]
        and disclosure.get("full55_gate")
        == [EXPECTED_GATE_SEED, EXPECTED_GATE_SEED + 54]
        and disclosure.get("formal_seed_consumption") == "NONE"
    ):
        raise ValueError("head-only postprocess correction or seeds differ")
    return plan


def _candidate_inventory(
    *, plan: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    completion_path = Path(str(plan["expected_training_completion"])).resolve(
        strict=True
    )
    completion = legacy.base._read(completion_path)
    rounds = completion.get("rounds", ())
    if not (
        completion.get("status") == "COMPLETE"
        and completion.get("route_variant") == "action_head_only_fire_aim_v1"
        and completion.get("optimizer_parameter_scope") == "action_net_only"
        and completion.get("frozen_backbone") is True
        and completion.get("frozen_parameters_unchanged") is True
        and completion.get("formal_seed_consumption") is False
        and completion.get("formal_candidate_authority") is False
        and rounds
        and all(
            row.get("optimization", {}).get("aim_loss_scope") == "fire_only"
            and row.get("optimization", {}).get("optimizer_parameter_scope")
            == "action_net_only"
            and row.get("optimization", {}).get("frozen_parameters_unchanged")
            is True
            for row in rounds
        )
    ):
        raise ValueError("head-only training completion is ineligible")
    candidates, evidence = legacy._BASE_INVENTORY(plan=plan)
    for candidate in candidates:
        if candidate.get("promotable") is True:
            candidate["id"] = "headonly-" + str(candidate["id"])
            candidate["route_variant"] = "action_head_only_fire_aim_v1"
    return candidates, evidence


def run(*, plan_path: Path, expected_plan_sha256: str, poll_seconds: float) -> int:
    plan_path = plan_path.resolve(strict=True)
    plan = validate_plan(plan_path, expected_plan_sha256)
    original_validate = legacy.validate_plan
    original_inventory = legacy._candidate_inventory
    original_script = legacy.SCRIPT_PATH
    original_campaign = legacy.EXPECTED_ROUTE_CAMPAIGN
    original_trainer = legacy.EXPECTED_TRAINER_NAME
    original_screen_seed = legacy.EXPECTED_SCREEN_SEED
    original_gate_seed = legacy.EXPECTED_GATE_SEED
    original_audit = legacy.base.base.audit_matrix
    try:
        legacy.validate_plan = lambda _path, _hash=None: plan
        legacy._candidate_inventory = _candidate_inventory
        legacy.SCRIPT_PATH = SCRIPT_PATH
        legacy.EXPECTED_ROUTE_CAMPAIGN = EXPECTED_ROUTE_CAMPAIGN
        legacy.EXPECTED_TRAINER_NAME = EXPECTED_TRAINER_NAME
        legacy.EXPECTED_SCREEN_SEED = EXPECTED_SCREEN_SEED
        legacy.EXPECTED_GATE_SEED = EXPECTED_GATE_SEED
        legacy.base.base.audit_matrix = audit_matrix
        return legacy.run(
            plan_path=plan_path,
            expected_plan_sha256=expected_plan_sha256,
            poll_seconds=poll_seconds,
        )
    finally:
        legacy.base.base.audit_matrix = original_audit
        legacy.EXPECTED_GATE_SEED = original_gate_seed
        legacy.EXPECTED_SCREEN_SEED = original_screen_seed
        legacy.EXPECTED_TRAINER_NAME = original_trainer
        legacy.EXPECTED_ROUTE_CAMPAIGN = original_campaign
        legacy.SCRIPT_PATH = original_script
        legacy._candidate_inventory = original_inventory
        legacy.validate_plan = original_validate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    plan_path = args.plan.expanduser().resolve(strict=True)
    plan = validate_plan(plan_path, str(args.expected_plan_sha256))
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "VALID",
                    "plan": {
                        "path": str(plan_path),
                        "sha256": _sha256(plan_path),
                    },
                    "screen_candidates": int(
                        plan["screen"]["expected_candidates"]
                    ),
                    "screen_attempts": int(
                        plan["screen"]["expected_attempts"]
                    ),
                    "maximum_full55_attempts": int(
                        plan["full55_gate"]["maximum_expected_attempts"]
                    ),
                    "timeout_reader": "canonical_v1",
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
