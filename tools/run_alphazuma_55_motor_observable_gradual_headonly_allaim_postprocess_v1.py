"""Screen the head-only all-actions factorial route with frozen semantics."""

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
    run_alphazuma_55_motor_observable_gradual_headonly_postprocess_v1
    as legacy,
)


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
EXPECTED_ROUTE_CAMPAIGN = (
    "alphazuma-55-motor-observable-gradual-headonly-allaim-s99081646-v1"
)
EXPECTED_TRAINER_NAME = (
    "distill_alphazuma_55_motor_observable_gradual_headonly_allaim_v1.py"
)
EXPECTED_LAUNCH_RECEIPT_BUILDER_NAME = (
    "capture_alphazuma_55_motor_observable_gradual_headonly_allaim_"
    "launch_receipt_v1.py"
)
EXPECTED_SCREEN_SEED = 1_550_005_000
EXPECTED_GATE_SEED = 1_550_005_100
PREREG_AUDIT = PROJECT_ROOT / (
    "diagnostics/alphazuma-55-motor-observable-gradual-headonly-allaim-"
    "s99081646-independent-prereg-audit-v1.json"
)
PREREG_AUDIT_SHA256 = (
    "sha256:b0c17f5a4d96bf940fbe267b0a1472bcc31b6282119d02f551744a36f18a430c"
)
MAXIMUM_GATE_CANDIDATES = legacy.MAXIMUM_GATE_CANDIDATES
audit_matrix = legacy.audit_matrix
_read = legacy._read
_sha256 = legacy._sha256
_utc_now = legacy._utc_now
_write_new = legacy._write_new


def _bound(reference: Any, label: str) -> Path:
    return legacy.legacy.base._bound(reference, label)


def _valid_sha256(value: Any) -> bool:
    return legacy._valid_sha256(value)


def _validate_launch_receipt(
    plan: dict[str, Any], route: dict[str, Any], route_path: Path
) -> None:
    receipt_path = _bound(
        plan.get("training_route_launch_receipt"),
        "training route launch receipt",
    )
    receipt = _read(receipt_path)
    bindings = receipt.get("bindings", {})
    preregistration_path = _bound(
        bindings.get("preregistration"), "launch receipt preregistration"
    )
    trainer_path = _bound(bindings.get("trainer"), "launch receipt trainer")
    builder_path = _bound(bindings.get("builder"), "launch receipt builder")
    audit_path = _bound(
        bindings.get("preregistration_independent_audit"),
        "launch receipt preregistration independent audit",
    )
    receipt_builder_path = _bound(
        bindings.get("launch_receipt_builder"),
        "launch receipt builder",
    )
    route_trainer = _bound(route.get("trainer"), "factorial trainer")
    route_builder = _bound(route.get("builder"), "factorial builder")
    run = route.get("run", {})
    run_dir = Path(str(run.get("run_dir", ""))).resolve()
    validation = receipt.get("validation", {})
    processes = receipt.get("processes", {})
    initial = receipt.get("initial_artifacts", {})
    initial_status = initial.get("training_status", {})
    resources = receipt.get("resource_snapshot", {})
    gpus = resources.get("gpus", ())
    boundary = receipt.get("authority_boundary", {})
    logs = receipt.get("logs", {})
    if not (
        receipt.get("schema")
        == "zuma-rl.alphazuma-55-motor-observable-gradual-launch-receipt"
        and receipt.get("version") == 1
        and receipt.get("status") == "LAUNCHED_AND_COLLECTING"
        and receipt.get("campaign_id") == EXPECTED_ROUTE_CAMPAIGN
        and preregistration_path == route_path
        and bindings.get("preregistration", {}).get("sha256")
        == plan.get("training_route", {}).get("sha256")
        and trainer_path == route_trainer
        and builder_path == route_builder
        and audit_path == PREREG_AUDIT.resolve(strict=True)
        and receipt_builder_path.name
        == EXPECTED_LAUNCH_RECEIPT_BUILDER_NAME
        and validation.get("status") == "VALID"
        and validation.get("observation_shape") == [22904]
        and tuple(
            float(value)
            for value in validation.get("teacher_execution_probabilities", ())
        )
        == (1.0, 0.9, 0.7, 0.5)
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
        and boundary.get("training_seed_range_is_fresh") is True
        and boundary.get("formal_selection_seed_consumption") == "NONE"
        and boundary.get("formal_final_blind_seed_consumption") == "NONE"
        and boundary.get("continuous_campaign_seed_consumption") == "NONE"
        and boundary.get("formal_candidate_authority") is False
        and boundary.get("power_restore_authority") is False
    ):
        raise ValueError("factorial training launch receipt is ineligible")
    initial_config = _read(
        _bound(initial.get("config"), "launch receipt initial config")
    )
    initial_run = initial_config.get("run", {})
    if not (
        initial_run.get("aim_loss_scope") == "all_actions"
        and initial_run.get("all_action_aim_supervision") is True
        and initial_run.get("route_variant")
        == "action_head_only_all_actions_aim_v1"
        and initial_run.get("optimizer_parameter_scope") == "action_net_only"
        and initial_run.get("frozen_backbone") is True
    ):
        raise ValueError("factorial initial launch config contract differs")
    for stream in ("stdout", "stderr"):
        row = logs.get(stream, {})
        if not (
            str(row.get("path", "")).strip()
            and int(row.get("bytes_at_snapshot", -1)) >= 0
            and _valid_sha256(row.get("sha256_at_snapshot"))
        ):
            raise ValueError(f"factorial launch {stream} evidence differs")


def _validate_prereg_audit(
    plan: dict[str, Any], route: dict[str, Any]
) -> None:
    reference = plan.get("training_route_independent_prereg_audit", {})
    path = _bound(reference, "factorial independent prereg audit")
    if path != PREREG_AUDIT.resolve(strict=True):
        raise ValueError("factorial plan binds another prereg audit")
    if _sha256(path) != PREREG_AUDIT_SHA256:
        raise ValueError("factorial prereg audit bytes differ")
    audit = _read(path)
    run = route.get("run", {})
    registry = route.get("seed_registry", {})
    if not (
        audit.get("status") == "PASS"
        and audit.get("candidate", {}).get("sha256")
        == plan.get("training_route", {}).get("sha256")
        and audit.get("factorial_lineage_verified") is True
        and audit.get("all_other_recipe_fields_byte_equivalent_after_normalization")
        is True
        and audit.get("formal_seed_consumption") == "NONE"
        and audit.get("formal_candidate_authority") is False
        and route.get("campaign_id") == EXPECTED_ROUTE_CAMPAIGN
        and Path(str(route.get("trainer", {}).get("path", ""))).name
        == EXPECTED_TRAINER_NAME
        and run.get("aim_loss_scope") == "all_actions"
        and run.get("all_action_aim_supervision") is True
        and run.get("route_variant")
        == "action_head_only_all_actions_aim_v1"
        and run.get("optimizer_parameter_scope") == "action_net_only"
        and tuple(run.get("trainable_parameter_names", ()))
        == ("action_net.weight", "action_net.bias")
        and run.get("frozen_backbone") is True
        and registry.get("classification")
        == "fresh_engineering_training_subrange"
        and registry.get("formal_selection_seed_consumption") == "NONE"
        and all(
            value is False
            for value in route.get("authority_boundary", {}).values()
        )
    ):
        raise ValueError("factorial training preregistration is ineligible")


def _candidate_inventory(
    *, plan: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    completion_path = Path(str(plan["expected_training_completion"])).resolve(
        strict=True
    )
    completion = _read(completion_path)
    rounds = completion.get("rounds", ())
    if not (
        completion.get("status") == "COMPLETE"
        and completion.get("route_variant")
        == "action_head_only_all_actions_aim_v1"
        and completion.get("aim_loss_scope") == "all_actions"
        and completion.get("all_action_aim_supervision") is True
        and completion.get("optimizer_parameter_scope") == "action_net_only"
        and completion.get("frozen_backbone") is True
        and completion.get("frozen_parameters_unchanged") is True
        and completion.get("formal_seed_consumption") is False
        and completion.get("formal_candidate_authority") is False
        and rounds
        and all(
            row.get("optimization", {}).get("aim_loss_scope")
            == "all_actions"
            and row.get("optimization", {}).get(
                "optimizer_parameter_scope"
            )
            == "action_net_only"
            and row.get("optimization", {}).get(
                "frozen_parameters_unchanged"
            )
            is True
            for row in rounds
        )
    ):
        raise ValueError("factorial training completion is ineligible")
    candidates, evidence = legacy.legacy._BASE_INVENTORY(plan=plan)
    for candidate in candidates:
        if candidate.get("promotable") is True:
            candidate["id"] = "headonly-allaim-" + str(candidate["id"])
            candidate["route_variant"] = (
                "action_head_only_all_actions_aim_v1"
            )
    return candidates, evidence


@contextmanager
def _patched_legacy() -> Iterator[None]:
    names = {
        "SCRIPT_PATH": SCRIPT_PATH,
        "EXPECTED_ROUTE_CAMPAIGN": EXPECTED_ROUTE_CAMPAIGN,
        "EXPECTED_TRAINER_NAME": EXPECTED_TRAINER_NAME,
        "EXPECTED_LAUNCH_RECEIPT_BUILDER_NAME": (
            EXPECTED_LAUNCH_RECEIPT_BUILDER_NAME
        ),
        "EXPECTED_SCREEN_SEED": EXPECTED_SCREEN_SEED,
        "EXPECTED_GATE_SEED": EXPECTED_GATE_SEED,
        "PREREG_AUDIT": PREREG_AUDIT,
        "PREREG_AUDIT_SHA256": PREREG_AUDIT_SHA256,
        "_validate_prereg_audit": _validate_prereg_audit,
        "_validate_launch_receipt": _validate_launch_receipt,
        "_candidate_inventory": _candidate_inventory,
    }
    originals = {name: getattr(legacy, name) for name in names}
    try:
        for name, value in names.items():
            setattr(legacy, name, value)
        yield
    finally:
        for name, value in originals.items():
            setattr(legacy, name, value)


def validate_plan(
    path: Path, expected_sha256: str | None = None
) -> dict[str, Any]:
    with _patched_legacy():
        return legacy.validate_plan(path, expected_sha256)


def run(*, plan_path: Path, expected_plan_sha256: str, poll_seconds: float) -> int:
    with _patched_legacy():
        return legacy.run(
            plan_path=plan_path,
            expected_plan_sha256=expected_plan_sha256,
            poll_seconds=poll_seconds,
        )


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
