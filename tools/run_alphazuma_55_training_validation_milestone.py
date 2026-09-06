"""Run one frozen AlphaZuma 55 training-health milestone unattended."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools import snapshot_alphazuma_55_weekend as snapshot_module


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    _require(parsed.tzinfo is not None, f"timestamp lacks timezone: {value}")
    return parsed.astimezone(timezone.utc)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON document {path}: {error}") from error
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _expected_hash(path: Path, expected: str, name: str) -> None:
    _require(
        expected.startswith("sha256:") and len(expected) == 71,
        f"invalid expected hash for {name}",
    )
    _require(_sha256(path) == expected, f"{name} hash mismatch")


def _bound_path(reference: Any, name: str) -> Path:
    _require(isinstance(reference, dict), f"{name} must be an object")
    path = Path(str(reference.get("path", ""))).expanduser().resolve(strict=True)
    _expected_hash(path, str(reference.get("sha256", "")), name)
    return path


def _validate_activation(snapshot: dict[str, Any], plan: dict[str, Any]) -> None:
    gate = plan["activation_gate"]
    _require(snapshot.get("status") == "PASS", "live snapshot is not PASS")
    training = snapshot["training"]
    _require(
        int(training["route_count"]) == 4
        and int(training["failed_routes"]) == 0
        and int(training["running_routes"]) + int(training["completed_routes"]) == 4,
        "not all four frozen training routes are healthy",
    )
    _require(
        snapshot["controller"]["phase"]
        == gate["formal_controller_phase_must_equal"],
        "formal controller is not waiting for training",
    )
    roots = snapshot["formal_output_boundary"]["roots"]
    _require(
        snapshot["formal_output_boundary"]["embargo_boundary_intact"] is True
        and not any(bool(item["exists"]) for item in roots.values()),
        "formal output embargo boundary is not intact",
    )
    hardware = snapshot["hardware"]
    available = int(hardware["memory"]["available_bytes"])
    required = int(gate["wsl_available_memory_gib_at_least"]) * 1024**3
    _require(available >= required, "available WSL memory is below activation gate")
    _require(
        int(hardware["memory"]["swap_used_bytes"])
        == int(gate["wsl_swap_used_bytes_must_equal"]),
        "WSL swap use differs from activation gate",
    )
    temperature_limit = int(gate["gpu_temperature_c_below"])
    _require(
        all(int(gpu["temperature_c"]) < temperature_limit for gpu in hardware["gpus"]),
        "GPU temperature is above activation gate",
    )


def _validate_frozen_contracts(
    *,
    plan_path: Path,
    expected_plan_sha256: str,
    deployment_path: Path,
    expected_deployment_sha256: str,
    snapshot_tool_path: Path,
    expected_snapshot_tool_sha256: str,
    expected_controller_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    _expected_hash(plan_path, expected_plan_sha256, "milestone plan")
    _expected_hash(deployment_path, expected_deployment_sha256, "deployment preregistration")
    _expected_hash(snapshot_tool_path, expected_snapshot_tool_sha256, "snapshot tool")
    _expected_hash(Path(__file__).resolve(), expected_controller_sha256, "milestone controller")
    _require(
        snapshot_tool_path == Path(snapshot_module.__file__).resolve(),
        "validated snapshot path is not the imported snapshot module",
    )

    plan = _read_json(plan_path)
    deployment = _read_json(deployment_path)
    _require(
        plan.get("schema")
        == "zuma-rl.alphazuma-55-training-validation-milestone-plan"
        and plan.get("version") == 1
        and plan.get("status") == "FROZEN_BEFORE_TARGET_CHECKPOINT",
        "unexpected milestone plan",
    )
    _require(
        deployment.get("schema")
        == "zuma-rl.alphazuma-55-postprocess-deployment-preregistration"
        and deployment.get("status") == "FROZEN_BEFORE_FORMAL_SEEDS",
        "unexpected deployment preregistration",
    )
    _require(
        deployment.get("campaign_id") == plan.get("campaign_id"),
        "campaign id differs between milestone and deployment",
    )

    master_path = _bound_path(plan["master_preregistration"], "master preregistration")
    _bound_path(plan["training_route"], "training route")
    _bound_path(plan["prior_validation_audit"], "prior validation audit")
    _bound_path(plan["models"]["baseline"], "baseline model")
    for name, reference in plan["implementation"].items():
        _bound_path(reference, f"implementation {name}")
    _require(
        deployment["master_preregistration"]["sha256"] == _sha256(master_path),
        "deployment does not bind milestone master",
    )
    _require(plan["activation_gate"]["run_at_most_once"] is True, "plan is not one-shot")
    return plan, deployment, master_path


def _validate_generated_contracts(
    *,
    plan: dict[str, Any],
    target_path: Path,
    target_sha256: str,
    manifest_path: Path,
    preregistration_path: Path,
) -> None:
    manifest = _read_json(manifest_path)
    prereg = _read_json(preregistration_path)
    baseline = plan["models"]["baseline"]
    target = plan["models"]["target"]
    expected_models = [
        {
            "id": baseline["id"],
            "training_steps": int(baseline["training_steps"]),
            "path": str(Path(str(baseline["path"])).resolve(strict=True)),
            "sha256": baseline["sha256"],
        },
        {
            "id": target["id"],
            "training_steps": int(target["training_steps"]),
            "path": str(target_path),
            "sha256": target_sha256,
        },
    ]
    _require(manifest.get("models") == expected_models, "generated model manifest differs from plan")
    matrix = plan["matrix"]
    _require(
        prereg.get("stage") == "training_validation"
        and int(prereg["total_attempts"]) == int(matrix["levels"])
        and int(prereg["attempts_per_level"]) == int(matrix["attempts_per_level_per_model"]),
        "generated validation matrix differs from plan",
    )
    _require(
        int(prereg["seed_plan"]["base_seed"]) == int(matrix["base_seed"])
        and int(prereg["seed_plan"]["last_seed"]) == int(matrix["last_seed"]),
        "generated validation seeds differ from plan",
    )
    execution = prereg["execution"]
    _require(
        execution["devices"] == [matrix["device"]]
        and int(execution["parallel_envs_per_shard"]) == int(matrix["parallel_envs"])
        and Path(str(execution["output_root"])).resolve()
        == Path(str(plan["outputs"]["run_root"])).resolve(),
        "generated validation execution differs from plan",
    )
    _require(
        prereg["diagnostic_contract"]["formal_selection_authority"] is False
        and prereg["diagnostic_contract"]["training_recipe_change_authority"] is False
        and prereg["diagnostic_contract"]["formal_seed_ranges_consumed"] is False,
        "generated contract exceeds training-health authority",
    )


class _StatusWriter:
    def __init__(self, path: Path, plan_path: Path) -> None:
        self.path = path
        self.plan_path = plan_path
        self.started_utc = _utc_now()

    def update(self, status: str, **extra: Any) -> None:
        value = {
            "schema": "zuma-rl.alphazuma-55-training-validation-milestone-controller",
            "version": 1,
            "status": status,
            "started_utc": self.started_utc,
            "updated_utc": _utc_now(),
            "plan": {"path": str(self.plan_path), "sha256": _sha256(self.plan_path)},
            "controller": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256(Path(__file__).resolve()),
                "pid": os.getpid(),
            },
            "formal_selection_authority": False,
            "training_recipe_change_authority": False,
            **extra,
        }
        _write_json_atomic(self.path, value)


def _run_checked(command: list[str], *, cwd: Path, log_path: Path) -> None:
    environment = os.environ.copy()
    environment.update({"TMPDIR": "/tmp", "TEMP": "/tmp", "PYTHONUNBUFFERED": "1"})
    with log_path.open("x", encoding="utf-8", newline="\n") as log:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    _require(completed.returncode == 0, f"command failed ({completed.returncode}): {command[1]}")


def _run_evaluator(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
    shard_path: Path,
    writer: _StatusWriter,
    poll_seconds: float,
) -> None:
    environment = os.environ.copy()
    environment.update({"TMPDIR": "/tmp", "TEMP": "/tmp", "PYTHONUNBUFFERED": "1"})
    with log_path.open("x", encoding="utf-8", newline="\n") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        while process.poll() is None:
            progress: dict[str, Any] = {"process_pid": process.pid}
            if shard_path.exists():
                try:
                    shard = _read_json(shard_path)
                    progress.update(
                        {
                            "shard_status": shard.get("status"),
                            "completed_attempts": shard.get("completed_attempts"),
                            "expected_attempts": shard.get("expected_attempts"),
                            "wall_seconds": shard.get("runtime", {}).get("wall_seconds"),
                            "shard_error": shard.get("error"),
                        }
                    )
                except ValueError:
                    progress["shard_status"] = "ATOMIC_UPDATE_IN_PROGRESS"
            writer.update("RUNNING_EVALUATION", evaluation=progress)
            time.sleep(poll_seconds)
        _require(process.returncode == 0, f"evaluator exited with code {process.returncode}")


def _verify_complete_receipt(
    *, plan: dict[str, Any], receipt_path: Path, target_sha256: str
) -> dict[str, Any]:
    receipt = _read_json(receipt_path)
    _require(receipt.get("status") == "PASS", "milestone audit receipt is not PASS")
    model_ids = [item["model_id"] for item in receipt["model_summaries"]]
    _require(
        model_ids == [plan["models"]["baseline"]["id"], plan["models"]["target"]["id"]],
        "audit model order differs from milestone plan",
    )
    _require(
        receipt["model_summaries"][1]["model_sha256"] == target_sha256,
        "audit target hash differs from frozen checkpoint",
    )
    _require(
        receipt["formal_selection_authority"] is False
        and receipt["training_recipe_change_authority"] is False,
        "audit receipt exceeds milestone authority",
    )
    return receipt


def execute(args: argparse.Namespace) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    plan_path = args.plan.expanduser().resolve(strict=True)
    deployment_path = args.deployment_preregistration.expanduser().resolve(strict=True)
    snapshot_tool_path = args.snapshot_tool.expanduser().resolve(strict=True)
    original_root = args.original_root.expanduser().resolve(strict=True)
    status_root = args.status_root.expanduser().resolve()
    status_root.mkdir(parents=True, exist_ok=True)
    status_path = status_root / "controller_status.json"
    writer = _StatusWriter(status_path, plan_path)

    plan, deployment, master_path = _validate_frozen_contracts(
        plan_path=plan_path,
        expected_plan_sha256=args.expected_plan_sha256,
        deployment_path=deployment_path,
        expected_deployment_sha256=args.expected_deployment_sha256,
        snapshot_tool_path=snapshot_tool_path,
        expected_snapshot_tool_sha256=args.expected_snapshot_tool_sha256,
        expected_controller_sha256=args.expected_controller_sha256,
    )
    outputs = plan["outputs"]
    run_root = Path(str(outputs["run_root"])).resolve()
    manifest_path = Path(str(outputs["models_manifest"])).resolve()
    preregistration_path = Path(str(outputs["preregistration"])).resolve()
    receipt_path = Path(str(outputs["audit_receipt"])).resolve()
    shard_path = run_root / "matrix-shard-00-of-01.json"
    target_path = Path(str(plan["models"]["target"]["expected_path"])).resolve()
    deadline = _parse_utc(_read_json(master_path)["schedule"]["formal_training_stop_utc"])

    if receipt_path.exists():
        _require(target_path.exists(), "audit exists but target checkpoint is absent")
        target_sha256 = _sha256(target_path)
        receipt = _verify_complete_receipt(
            plan=plan, receipt_path=receipt_path, target_sha256=target_sha256
        )
        writer.update(
            "COMPLETE",
            completed_utc=_utc_now(),
            target={"path": str(target_path), "sha256": target_sha256},
            audit={"path": str(receipt_path), "sha256": _sha256(receipt_path)},
            paired_comparison=receipt.get("paired_comparison"),
        )
        return receipt

    while not target_path.exists():
        _require(
            not any(path.exists() for path in (run_root, manifest_path, preregistration_path)),
            "milestone outputs appeared before the target checkpoint",
        )
        now = datetime.now(timezone.utc)
        _require(now < deadline, "target checkpoint did not appear before training stop")
        route_status_path = Path(str(plan["training_route"]["path"])).resolve(strict=True)
        route = _read_json(route_status_path)
        training_status_path = Path(str(route["runs"][0]["run_dir"])) / "training_status.json"
        training_status = _read_json(training_status_path.resolve(strict=True))
        steps = int(training_status["steps_this_run"])
        rate = float(training_status["steps_per_second"])
        target_steps = int(plan["models"]["target"]["training_steps"])
        eta_seconds = max(0.0, (target_steps - steps) / rate) if rate > 0 else None
        writer.update(
            "WAITING_FOR_TARGET",
            target={"path": str(target_path), "training_steps": target_steps},
            route={
                "status": training_status["status"],
                "steps_this_run": steps,
                "steps_per_second": rate,
                "eta_seconds": eta_seconds,
            },
            deadline_utc=deadline.isoformat(),
        )
        time.sleep(args.poll_seconds)

    target_sha256 = _sha256(target_path)
    _require(
        datetime.now(timezone.utc) < deadline,
        "target checkpoint appeared only after the frozen training stop",
    )
    snapshot = snapshot_module.build_snapshot(
        master_path=master_path,
        deployment_path=deployment_path,
        stale_seconds=args.stale_seconds,
    )
    _validate_activation(snapshot, plan)
    writer.update(
        "ACTIVATED",
        target={"path": str(target_path), "sha256": target_sha256},
        activation_snapshot={
            "checked_utc": snapshot["checked_utc"],
            "status": snapshot["status"],
            "controller_phase": snapshot["controller"]["phase"],
            "embargo_boundary_intact": snapshot["formal_output_boundary"]["embargo_boundary_intact"],
        },
    )

    builder_path = _bound_path(plan["implementation"]["contract_builder"], "contract builder")
    evaluator_path = _bound_path(plan["implementation"]["evaluator"], "evaluator")
    auditor_path = _bound_path(plan["implementation"]["independent_auditor"], "independent auditor")
    baseline = plan["models"]["baseline"]
    target = plan["models"]["target"]
    matrix = plan["matrix"]

    if manifest_path.exists() or preregistration_path.exists():
        _require(
            manifest_path.exists() and preregistration_path.exists(),
            "only one generated contract exists",
        )
    else:
        _require(not run_root.exists(), "validation run root already exists")
        build_command = [
            sys.executable,
            str(builder_path),
            "--master-preregistration",
            str(master_path),
            "--original-root",
            str(original_root),
            "--evaluator",
            str(evaluator_path),
            "--output-root",
            str(run_root),
            "--manifest-output",
            str(manifest_path),
            "--preregistration-output",
            str(preregistration_path),
            "--model",
            f"{baseline['id']}={baseline['training_steps']}={baseline['path']}",
            "--model",
            f"{target['id']}={target['training_steps']}={target_path}",
            "--base-seed",
            str(matrix["base_seed"]),
            "--device",
            str(matrix["device"]),
            "--parallel-envs",
            str(matrix["parallel_envs"]),
        ]
        _run_checked(build_command, cwd=project_root, log_path=status_root / "builder.log")

    _validate_generated_contracts(
        plan=plan,
        target_path=target_path,
        target_sha256=target_sha256,
        manifest_path=manifest_path,
        preregistration_path=preregistration_path,
    )
    writer.update(
        "CONTRACTS_FROZEN",
        target={"path": str(target_path), "sha256": target_sha256},
        manifest={"path": str(manifest_path), "sha256": _sha256(manifest_path)},
        preregistration={"path": str(preregistration_path), "sha256": _sha256(preregistration_path)},
    )

    validate_command = [
        sys.executable,
        str(evaluator_path),
        "--preregistration",
        str(preregistration_path),
        "--original-root",
        str(original_root),
        "--models-manifest",
        str(manifest_path),
        "--validate-only",
    ]
    if not shard_path.exists():
        _run_checked(
            validate_command,
            cwd=project_root,
            log_path=status_root / "validate-only.log",
        )
        evaluate_command = [
            sys.executable,
            str(evaluator_path),
            "--preregistration",
            str(preregistration_path),
            "--original-root",
            str(original_root),
            "--shard-index",
            "0",
            "--models-manifest",
            str(manifest_path),
        ]
        _run_evaluator(
            evaluate_command,
            cwd=project_root,
            log_path=status_root / "evaluation.log",
            shard_path=shard_path,
            writer=writer,
            poll_seconds=args.poll_seconds,
        )

    shard = _read_json(shard_path.resolve(strict=True))
    _require(
        shard.get("status") == "COMPLETE" and shard.get("error") is None,
        "existing validation shard is not a clean COMPLETE result",
    )
    writer.update(
        "AUDITING",
        shard={"path": str(shard_path), "sha256": _sha256(shard_path)},
    )
    if not receipt_path.exists():
        audit_command = [
            sys.executable,
            str(auditor_path),
            "--master-preregistration",
            str(master_path),
            "--preregistration",
            str(preregistration_path),
            "--models-manifest",
            str(manifest_path),
            "--matrix-shard",
            str(shard_path),
            "--output",
            str(receipt_path),
        ]
        _run_checked(audit_command, cwd=project_root, log_path=status_root / "audit.log")
    receipt = _verify_complete_receipt(
        plan=plan, receipt_path=receipt_path, target_sha256=target_sha256
    )
    writer.update(
        "COMPLETE",
        completed_utc=_utc_now(),
        target={"path": str(target_path), "sha256": target_sha256},
        audit={"path": str(receipt_path), "sha256": _sha256(receipt_path)},
        paired_comparison=receipt.get("paired_comparison"),
    )
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--deployment-preregistration", required=True, type=Path)
    parser.add_argument("--expected-deployment-sha256", required=True)
    parser.add_argument("--snapshot-tool", required=True, type=Path)
    parser.add_argument("--expected-snapshot-tool-sha256", required=True)
    parser.add_argument("--expected-controller-sha256", required=True)
    parser.add_argument("--original-root", required=True, type=Path)
    parser.add_argument("--status-root", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--stale-seconds", type=float, default=180.0)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def _validate_only(args: argparse.Namespace) -> dict[str, Any]:
    plan_path = args.plan.expanduser().resolve(strict=True)
    deployment_path = args.deployment_preregistration.expanduser().resolve(strict=True)
    plan, _deployment, master_path = _validate_frozen_contracts(
        plan_path=plan_path,
        expected_plan_sha256=args.expected_plan_sha256,
        deployment_path=deployment_path,
        expected_deployment_sha256=args.expected_deployment_sha256,
        snapshot_tool_path=args.snapshot_tool.expanduser().resolve(strict=True),
        expected_snapshot_tool_sha256=args.expected_snapshot_tool_sha256,
        expected_controller_sha256=args.expected_controller_sha256,
    )
    snapshot = snapshot_module.build_snapshot(
        master_path=master_path,
        deployment_path=deployment_path,
        stale_seconds=args.stale_seconds,
    )
    _validate_activation(snapshot, plan)
    target_path = Path(str(plan["models"]["target"]["expected_path"])).resolve()
    return {
        "status": "VALID",
        "campaign_id": plan["campaign_id"],
        "plan": {"path": str(plan_path), "sha256": _sha256(plan_path)},
        "controller": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "snapshot": {
            "status": snapshot["status"],
            "checked_utc": snapshot["checked_utc"],
            "formal_controller_phase": snapshot["controller"]["phase"],
            "embargo_boundary_intact": snapshot["formal_output_boundary"]["embargo_boundary_intact"],
        },
        "target": {
            "path": str(target_path),
            "training_steps": int(plan["models"]["target"]["training_steps"]),
            "exists": target_path.exists(),
        },
        "classification": "read_only_preflight",
        "formal_seed_consumption": "none",
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if (
        not math.isfinite(args.poll_seconds)
        or args.poll_seconds <= 0
        or not math.isfinite(args.stale_seconds)
        or args.stale_seconds <= 0
    ):
        raise ValueError("poll and stale intervals must be finite and positive")
    if args.validate_only:
        try:
            result = _validate_only(args)
        except BaseException as error:
            result = {
                "status": "ERROR",
                "error": {"type": type(error).__name__, "message": str(error)},
                "classification": "read_only_preflight",
                "formal_seed_consumption": "none",
            }
            print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True))
            return 1
        print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True))
        return 0
    status_root = args.status_root.expanduser().resolve()
    status_root.mkdir(parents=True, exist_ok=True)
    status_path = status_root / "controller_status.json"
    lock_path = status_root / "controller.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(
                json.dumps(
                    {"status": "ALREADY_RUNNING", "lock": str(lock_path)},
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return 0
        try:
            receipt = execute(args)
        except BaseException as error:
            failure = {
                "schema": "zuma-rl.alphazuma-55-training-validation-milestone-controller",
                "version": 1,
                "status": "ERROR",
                "updated_utc": _utc_now(),
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                },
                "formal_selection_authority": False,
                "training_recipe_change_authority": False,
            }
            _write_json_atomic(status_path, failure)
            print(json.dumps(failure, ensure_ascii=False, allow_nan=False, sort_keys=True))
            return 1
    summary = {
        "status": "COMPLETE",
        "audit_receipt": _read_json(args.plan)["outputs"]["audit_receipt"],
        "paired_comparison": receipt.get("paired_comparison"),
        "interpretation_boundary": receipt.get("interpretation_boundary"),
    }
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
