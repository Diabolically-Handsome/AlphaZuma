"""Run the frozen motor-observable checkpoint screen and full55 gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any, Sequence

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))


SCRIPT_PATH = Path(__file__).resolve()
PLAN_SCHEMA = "zuma-rl.alphazuma-55-motor-observable-postprocess-plan"
SCREEN_LEVEL_IDS = (
    "Jungle1",
    "Jungle9",
    "village3",
    "village8",
    "city1",
    "city9",
    "Coast1",
    "Coast9",
    "grotto1",
    "grotto9",
    "volcano1",
    "volcano9",
)
EXPECTED_MOTOR_CHECKPOINTS = 12
EXPECTED_SCREEN_CANDIDATES = 13
MAXIMUM_GATE_CANDIDATES = 5


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _write_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _write_new(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen artifact: {path}")
    _write_atomic(path, value)


def _bound(reference: Any, label: str) -> Path:
    if not isinstance(reference, dict):
        raise ValueError(f"{label} reference must be an object")
    path = Path(str(reference.get("path", ""))).resolve(strict=True)
    if reference.get("sha256") != _sha256(path):
        raise ValueError(f"{label} bytes differ: {path}")
    return path


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _shard_bounds(count: int, index: int, shards: int) -> tuple[int, int]:
    return count * index // shards, count * (index + 1) // shards


def validate_plan(path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    path = path.resolve(strict=True)
    if expected_sha256 is not None and _sha256(path) != expected_sha256:
        raise ValueError("motor-observable postprocess plan hash differs")
    plan = _read(path)
    if not (
        plan.get("schema") == PLAN_SCHEMA
        and plan.get("version") == 1
        and plan.get("status")
        == "FROZEN_DURING_TRAINING_BEFORE_CHECKPOINT_INFERENCE"
    ):
        raise ValueError("unexpected motor-observable postprocess plan")
    implementation = plan.get("implementation", {})
    controller_path = _bound(implementation.get("controller"), "controller")
    if controller_path != SCRIPT_PATH:
        raise ValueError("postprocess plan binds another controller")
    for name, reference in implementation.items():
        _bound(reference, f"implementation {name}")
    master_path = _bound(plan.get("master_preregistration"), "master")
    master = _read(master_path)
    route_path = _bound(plan.get("training_route"), "training route")
    route = _read(route_path)
    template_path = _bound(plan.get("environment_template"), "environment template")
    template = _read(template_path)
    if not (
        master.get("schema")
        == "zuma-rl.alphazuma-55-weekend-master-preregistration"
        and route.get("schema")
        == "zuma-rl.alphazuma-55-motor-observable-replay-preregistration"
        and route.get("status") == "FROZEN_BEFORE_TRAINING"
        and route.get("run", {}).get("motor_observation_profile")
        == "motor-observable-v1"
        and template.get("schema")
        == "zuma-rl.zero-shot-multilevel-preregistration"
    ):
        raise ValueError("postprocess input contract changed")
    full_levels = plan.get("full55_levels", [])
    screen_levels = plan.get("screen_levels", [])
    if len(full_levels) != 55 or tuple(
        row.get("id") for row in screen_levels
    ) != SCREEN_LEVEL_IDS:
        raise ValueError("postprocess level inventory changed")
    full_ids = {str(row.get("id")) for row in full_levels}
    if not set(SCREEN_LEVEL_IDS).issubset(full_ids):
        raise ValueError("screen levels escaped full55")
    screen = plan.get("screen", {})
    gate = plan.get("full55_gate", {})
    if not (
        int(screen.get("base_seed", -1)) == 1_550_002_400
        and int(screen.get("last_seed", -1))
        == int(screen["base_seed"]) + len(screen_levels) - 1
        and int(screen.get("expected_candidates", -1))
        == EXPECTED_SCREEN_CANDIDATES
        and int(gate.get("base_seed", -1)) == 1_550_002_500
        and int(gate.get("last_seed", -1))
        == int(gate["base_seed"]) + len(full_levels) - 1
        and int(gate.get("maximum_candidates", -1))
        == MAXIMUM_GATE_CANDIDATES
        and int(gate.get("minimum_wins", -1)) == 35
        and int(gate.get("minimum_cleared_levels", -1)) == 35
    ):
        raise ValueError("postprocess matrix contract changed")
    registry = master["seed_registry"]["training_validation"]
    for first, last in (
        (int(screen["base_seed"]), int(screen["last_seed"])),
        (int(gate["base_seed"]), int(gate["last_seed"])),
    ):
        if not int(registry["first"]) <= first <= last <= int(registry["last"]):
            raise ValueError("postprocess seeds escaped training_validation")
    if int(screen["last_seed"]) >= int(gate["base_seed"]):
        raise ValueError("screen and gate seed ranges overlap")
    boundary = plan.get("authority_boundary", {})
    for name in (
        "formal_selection_seed_consumption",
        "formal_final_blind_seed_consumption",
        "continuous_campaign_seed_consumption",
    ):
        if boundary.get(name) is not False:
            raise ValueError(f"postprocess authority changed: {name}")
    return plan


def _processes_containing(fragment: str) -> list[int]:
    result: list[int] = []
    for candidate in Path("/proc").iterdir():
        if not candidate.name.isdigit():
            continue
        try:
            command = (candidate / "cmdline").read_bytes().replace(b"\0", b" ")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if fragment.encode("utf-8") in command:
            result.append(int(candidate.name))
    return sorted(result)


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
        == "zuma-rl.alphazuma-55-motor-observable-replay-completion"
        and completion.get("status") == "COMPLETE"
        and completion.get("motor_observation_profile")
        == "motor-observable-v1"
        and completion.get("formal_seed_consumption") is False
        and completion.get("formal_candidate_authority") is False
    ):
        raise ValueError("motor-observable training completion is ineligible")
    source_path = Path(str(route["source"]["model_path"])).resolve(strict=True)
    if route["source"]["model_sha256"] != _sha256(source_path):
        raise ValueError("motor-observable source changed")
    candidates: list[dict[str, Any]] = [
        {
            "id": "base-polar-intent-wide-epoch-10",
            "training_steps": 0,
            "path": str(source_path),
            "sha256": _sha256(source_path),
            "observation_profile": "human-speedrun-v1",
            "promotable": False,
            "candidate_order": 0,
            "source": "frozen_base_baseline",
        }
    ]
    anchor_steps = sum(
        int(episode["steps"])
        for collection in completion.get("anchor_collections", [])
        for episode in collection.get("episodes", [])
    )
    cumulative_steps = anchor_steps
    rounds = completion.get("student_rounds", [])
    if len(rounds) != 3:
        raise ValueError("motor-observable completion does not contain three rounds")
    for expected_round, row in enumerate(rounds):
        if int(row.get("round_index", -1)) != expected_round:
            raise ValueError("motor-observable round order changed")
        cumulative_steps += sum(
            int(episode["steps"])
            for episode in row.get("collection", {}).get("episodes", [])
        )
        checkpoints = row.get("optimization", {}).get("checkpoints", [])
        if len(checkpoints) != 4:
            raise ValueError("motor-observable checkpoint count changed")
        for expected_epoch, checkpoint in enumerate(checkpoints, start=1):
            if int(checkpoint.get("epoch", -1)) != expected_epoch:
                raise ValueError("motor-observable checkpoint epoch changed")
            model_path = Path(str(checkpoint.get("path", ""))).resolve(strict=True)
            model_hash = _sha256(model_path)
            if checkpoint.get("sha256") != model_hash:
                raise ValueError("motor-observable checkpoint bytes changed")
            candidates.append(
                {
                    "id": f"motor-r{expected_round:02d}-e{expected_epoch:02d}",
                    "training_steps": cumulative_steps,
                    "path": str(model_path),
                    "sha256": model_hash,
                    "observation_profile": "motor-observable-v1",
                    "promotable": True,
                    "candidate_order": len(candidates),
                    "round_index": expected_round,
                    "epoch": expected_epoch,
                }
            )
    if len(candidates) != EXPECTED_SCREEN_CANDIDATES:
        raise ValueError("motor-observable candidate inventory changed")
    return candidates, {
        "path": str(completion_path),
        "sha256": _sha256(completion_path),
    }


def _manifest(
    *,
    plan_path: Path,
    plan_sha256: str,
    stage: str,
    candidates: Sequence[dict[str, Any]],
    completion: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "zuma-rl.zero-shot-models-manifest",
        "version": 1,
        "status": "FROZEN",
        "created_utc": _utc_now(),
        "campaign_id": str(_read(plan_path)["campaign_id"]),
        "stage": stage,
        "postprocess_plan": {"path": str(plan_path), "sha256": plan_sha256},
        "training_completion": completion,
        "models": [dict(row) for row in candidates],
        "paired_models_share_identical_task_seeds": True,
        "formal_seed_consumption": False,
    }


def _evaluation_preregistration(
    *,
    plan: dict[str, Any],
    plan_path: Path,
    plan_sha256: str,
    stage: str,
    candidates: Sequence[dict[str, Any]],
    levels: Sequence[dict[str, Any]],
    base_seed: int,
    output_root: Path,
) -> dict[str, Any]:
    template_path = _bound(plan["environment_template"], "environment template")
    template = _read(template_path)
    evaluator = plan["implementation"]["evaluator"]
    devices = list(plan["execution"]["devices"])
    return {
        "schema": "zuma-rl.zero-shot-multilevel-preregistration",
        "version": 1,
        "status": "FROZEN_BEFORE_EVALUATION",
        "created_utc": _utc_now(),
        "campaign_id": plan["campaign_id"],
        "stage": stage,
        "master_preregistration": plan["master_preregistration"],
        "postprocess_plan": {"path": str(plan_path), "sha256": plan_sha256},
        "evaluator": evaluator,
        "model": dict(candidates[0]),
        "levels": [dict(row) for row in levels],
        "attempts_per_level": 1,
        "total_attempts": len(levels),
        "seed_plan": {
            "registry": "training_validation",
            "base_seed": int(base_seed),
            "last_seed": int(base_seed) + len(levels) - 1,
            "paired_models_share_identical_task_seeds": True,
        },
        "environment": template["environment"],
        "execution": {
            "output_root": str(output_root),
            "shard_count": len(devices),
            "devices": devices,
            "parallel_envs_per_shard": int(
                plan["execution"]["parallel_envs_per_shard"]
            ),
        },
        "authority_boundary": {
            "classification": "training_validation_only",
            "formal_selection_seed_consumption": False,
            "formal_final_blind_seed_consumption": False,
            "continuous_campaign_seed_consumption": False,
        },
    }


def _rank_key(row: dict[str, Any]) -> tuple[Any, ...]:
    median = row.get("median_winning_ticks")
    return (
        -int(row["wins"]),
        -int(row["cleared_levels"]),
        -int(row["total_score"]),
        float(median) if median is not None else math.inf,
        int(row["candidate_order"]),
    )


def audit_matrix(
    *,
    prereg_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    prereg = _read(prereg_path)
    manifest = _read(manifest_path)
    models = manifest["models"]
    levels = prereg["levels"]
    shard_count = int(prereg["execution"]["shard_count"])
    output_root = Path(str(prereg["execution"]["output_root"])).resolve(
        strict=True
    )
    evaluator_path = Path(str(prereg["evaluator"]["path"])).resolve(strict=True)
    if prereg["evaluator"]["sha256"] != _sha256(evaluator_path):
        raise ValueError("evaluation preregistration evaluator changed")
    expected: dict[tuple[str, str], tuple[int, str]] = {}
    seed_base = int(prereg["seed_plan"]["base_seed"])
    for level_index, level in enumerate(levels):
        for model in models:
            expected[(str(model["id"]), str(level["id"]))] = (
                seed_base + level_index,
                str(model["sha256"]),
            )
    observed: dict[tuple[str, str], dict[str, Any]] = {}
    shards: list[dict[str, Any]] = []
    for shard_index in range(shard_count):
        shard_path = output_root / (
            f"matrix-shard-{shard_index:02d}-of-{shard_count:02d}.json"
        )
        shard = _read(shard_path)
        start, end = _shard_bounds(len(levels), shard_index, shard_count)
        expected_attempts = (end - start) * len(models)
        if not (
            shard.get("status") == "COMPLETE"
            and shard.get("error") is None
            and int(shard.get("completed_attempts", -1)) == expected_attempts
            and int(shard.get("expected_attempts", -1)) == expected_attempts
            and shard.get("preregistration", {}).get("sha256")
            == _sha256(prereg_path)
            and shard.get("models_manifest", {}).get("sha256")
            == _sha256(manifest_path)
            and shard.get("evaluator_sha256") == _sha256(evaluator_path)
        ):
            raise ValueError(f"evaluation shard failed validation: {shard_path}")
        for row in shard.get("attempts", []):
            key = (str(row.get("model_id")), str(row.get("level_id")))
            if key in observed or key not in expected:
                raise ValueError(f"unexpected or duplicated evaluation attempt: {key}")
            expected_seed, expected_hash = expected[key]
            if not (
                int(row.get("seed", -1)) == expected_seed
                and row.get("model_sha256") == expected_hash
                and row.get("outcome") in {"win", "loss"}
            ):
                raise ValueError(f"evaluation attempt contract changed: {key}")
            observed[key] = row
        shards.append({"path": str(shard_path), "sha256": _sha256(shard_path)})
    if set(observed) != set(expected):
        raise ValueError("evaluation matrix is incomplete")
    summaries: list[dict[str, Any]] = []
    for model in models:
        rows = [
            observed[(str(model["id"]), str(level["id"]))]
            for level in levels
        ]
        wins = [row for row in rows if row["outcome"] == "win"]
        summaries.append(
            {
                "model_id": model["id"],
                "model_sha256": model["sha256"],
                "observation_profile": model["observation_profile"],
                "promotable": bool(model.get("promotable", False)),
                "candidate_order": int(model["candidate_order"]),
                "attempts": len(rows),
                "wins": len(wins),
                "losses": sum(row["outcome"] == "loss" for row in rows),
                "cleared_levels": len({row["level_id"] for row in wins}),
                "total_score": sum(int(row.get("score", 0)) for row in rows),
                "median_winning_ticks": (
                    statistics.median(int(row["ticks"]) for row in wins)
                    if wins
                    else None
                ),
                "capacity_overflows": sum(
                    bool(row.get("observation_capacity_overflow", False))
                    for row in rows
                ),
            }
        )
    ranking = sorted(summaries, key=_rank_key)
    for rank, row in enumerate(ranking, start=1):
        row["rank"] = rank
    return {
        "schema": "zuma-rl.alphazuma-55-motor-observable-matrix-audit",
        "version": 1,
        "status": "PASS",
        "audited_utc": _utc_now(),
        "preregistration": {
            "path": str(prereg_path),
            "sha256": _sha256(prereg_path),
        },
        "models_manifest": {
            "path": str(manifest_path),
            "sha256": _sha256(manifest_path),
        },
        "shards": shards,
        "attempts": len(observed),
        "model_summaries": summaries,
        "ranking": ranking,
        "formal_seed_consumption": "NONE",
    }


def _wait_processes(
    *,
    processes: Sequence[subprocess.Popen[Any]],
    stage: str,
    output_root: Path,
    status_callback: Any,
    poll_seconds: float,
    deadline: datetime,
) -> None:
    while True:
        exits = [process.poll() for process in processes]
        status_callback(
            f"{stage}_EVALUATION",
            shard_processes=[
                {"pid": process.pid, "exit_code": exits[index]}
                for index, process in enumerate(processes)
            ],
        )
        if all(code is not None for code in exits):
            if any(code != 0 for code in exits):
                raise RuntimeError(f"{stage} evaluator process failed: {exits}")
            return
        if datetime.now(timezone.utc) >= deadline:
            raise TimeoutError(f"{stage} evaluation exceeded deadline")
        time.sleep(poll_seconds)


def _run_evaluation(
    *,
    plan: dict[str, Any],
    prereg_path: Path,
    manifest_path: Path,
    stage: str,
    status_callback: Any,
    poll_seconds: float,
    deadline: datetime,
) -> dict[str, Any]:
    evaluator_path = _bound(plan["implementation"]["evaluator"], "evaluator")
    original_root = str(plan["original_root"])
    stage_root = Path(str(_read(prereg_path)["execution"]["output_root"]))
    stage_root.mkdir(parents=True, exist_ok=True)
    validate_stdout = stage_root / "validate.stdout.log"
    validate_stderr = stage_root / "validate.stderr.log"
    command = [
        sys.executable,
        str(evaluator_path),
        "--preregistration",
        str(prereg_path),
        "--models-manifest",
        str(manifest_path),
        "--original-root",
        original_root,
        "--validate-only",
    ]
    with validate_stdout.open("x", encoding="utf-8") as stdout, validate_stderr.open(
        "x", encoding="utf-8"
    ) as stderr:
        validated = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
    if validated.returncode != 0:
        raise RuntimeError(f"{stage} evaluator validation failed")
    prereg = _read(prereg_path)
    shards = int(prereg["execution"]["shard_count"])
    processes: list[subprocess.Popen[Any]] = []
    handles: list[Any] = []
    try:
        for shard_index in range(shards):
            stdout = (stage_root / f"shard-{shard_index:02d}.stdout.log").open(
                "x", encoding="utf-8"
            )
            stderr = (stage_root / f"shard-{shard_index:02d}.stderr.log").open(
                "x", encoding="utf-8"
            )
            handles.extend((stdout, stderr))
            processes.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        str(evaluator_path),
                        "--preregistration",
                        str(prereg_path),
                        "--models-manifest",
                        str(manifest_path),
                        "--original-root",
                        original_root,
                        "--shard-index",
                        str(shard_index),
                    ],
                    stdout=stdout,
                    stderr=stderr,
                    text=True,
                )
            )
        _wait_processes(
            processes=processes,
            stage=stage,
            output_root=stage_root,
            status_callback=status_callback,
            poll_seconds=poll_seconds,
            deadline=deadline,
        )
    finally:
        for handle in handles:
            handle.close()
    return audit_matrix(prereg_path=prereg_path, manifest_path=manifest_path)


def run(*, plan_path: Path, expected_plan_sha256: str, poll_seconds: float) -> int:
    plan_path = plan_path.resolve(strict=True)
    plan = validate_plan(plan_path, expected_plan_sha256)
    output_root = Path(str(plan["outputs"]["output_root"])).resolve()
    if output_root.exists():
        raise FileExistsError(f"postprocess output root already exists: {output_root}")
    output_root.mkdir(parents=True)
    status_path = output_root / "controller_status.json"
    deadline = _parse_utc(plan["deadline_utc"])
    started = time.perf_counter()

    def write_status(phase: str, **extra: Any) -> None:
        _write_atomic(
            status_path,
            {
                "schema": "zuma-rl.alphazuma-55-motor-observable-postprocess-status",
                "version": 1,
                "status": "RUNNING",
                "phase": phase,
                "updated_utc": _utc_now(),
                "wall_seconds": time.perf_counter() - started,
                "plan": {"path": str(plan_path), "sha256": expected_plan_sha256},
                "formal_seed_consumption": "NONE",
                "error": None,
                **extra,
            },
        )

    try:
        completion_path = Path(str(plan["expected_training_completion"]))
        failure_path = Path(str(plan["expected_training_failure"]))
        missing_process_rounds = 0
        while not completion_path.exists():
            if failure_path.exists():
                raise RuntimeError("motor-observable training reported failure")
            processes = _processes_containing(
                "distill_alphazuma_55_motor_observable_replay_v2.py"
            )
            missing_process_rounds = 0 if processes else missing_process_rounds + 1
            write_status(
                "WAITING_FOR_TRAINING",
                training_processes=processes,
                missing_process_rounds=missing_process_rounds,
            )
            if missing_process_rounds >= 3:
                raise RuntimeError("motor-observable trainer disappeared")
            if datetime.now(timezone.utc) >= deadline:
                raise TimeoutError("motor-observable training exceeded deadline")
            time.sleep(poll_seconds)

        write_status("FREEZING_SCREEN_CANDIDATES")
        candidates, completion_reference = _candidate_inventory(plan=plan)
        plan_sha256 = _sha256(plan_path)
        screen_manifest_path = output_root / "screen-models-manifest.json"
        screen_prereg_path = output_root / "screen-preregistration.json"
        screen_root = output_root / "screen"
        _write_new(
            screen_manifest_path,
            _manifest(
                plan_path=plan_path,
                plan_sha256=plan_sha256,
                stage="motor_observable_checkpoint_screen",
                candidates=candidates,
                completion=completion_reference,
            ),
        )
        _write_new(
            screen_prereg_path,
            _evaluation_preregistration(
                plan=plan,
                plan_path=plan_path,
                plan_sha256=plan_sha256,
                stage="motor_observable_checkpoint_screen",
                candidates=candidates,
                levels=plan["screen_levels"],
                base_seed=int(plan["screen"]["base_seed"]),
                output_root=screen_root,
            ),
        )
        screen_audit = _run_evaluation(
            plan=plan,
            prereg_path=screen_prereg_path,
            manifest_path=screen_manifest_path,
            stage="SCREEN",
            status_callback=write_status,
            poll_seconds=poll_seconds,
            deadline=deadline,
        )
        selected_ids = [
            str(row["model_id"])
            for row in screen_audit["ranking"][:MAXIMUM_GATE_CANDIDATES]
        ]
        selected = [
            next(candidate for candidate in candidates if candidate["id"] == model_id)
            for model_id in selected_ids
        ]
        screen_report = {
            **screen_audit,
            "selected_for_full55_gate": selected_ids,
            "selection_count": len(selected),
            "selection_rule": (
                "wins_desc, coverage_desc, score_desc, median_winning_ticks_asc, "
                "pre-frozen candidate order"
            ),
        }
        screen_report_path = output_root / "screen-audit.json"
        _write_new(screen_report_path, screen_report)

        write_status("FREEZING_FULL55_GATE", selected_model_ids=selected_ids)
        gate_manifest_path = output_root / "full55-models-manifest.json"
        gate_prereg_path = output_root / "full55-preregistration.json"
        gate_root = output_root / "full55"
        _write_new(
            gate_manifest_path,
            _manifest(
                plan_path=plan_path,
                plan_sha256=plan_sha256,
                stage="motor_observable_full55_gate",
                candidates=selected,
                completion=completion_reference,
            ),
        )
        _write_new(
            gate_prereg_path,
            _evaluation_preregistration(
                plan=plan,
                plan_path=plan_path,
                plan_sha256=plan_sha256,
                stage="motor_observable_full55_gate",
                candidates=selected,
                levels=plan["full55_levels"],
                base_seed=int(plan["full55_gate"]["base_seed"]),
                output_root=gate_root,
            ),
        )
        gate_audit = _run_evaluation(
            plan=plan,
            prereg_path=gate_prereg_path,
            manifest_path=gate_manifest_path,
            stage="FULL55_GATE",
            status_callback=write_status,
            poll_seconds=poll_seconds,
            deadline=deadline,
        )
        promotable = [
            row for row in gate_audit["ranking"] if bool(row["promotable"])
        ]
        if not promotable:
            raise RuntimeError("full55 gate contains no promotable motor model")
        winner = promotable[0]
        passed = (
            int(winner["wins"]) >= int(plan["full55_gate"]["minimum_wins"])
            and int(winner["cleared_levels"])
            >= int(plan["full55_gate"]["minimum_cleared_levels"])
        )
        winner_model = next(
            row for row in selected if row["id"] == winner["model_id"]
        )
        final_report = {
            "schema": "zuma-rl.alphazuma-55-motor-observable-postprocess-result",
            "version": 1,
            "status": "COMPLETE_GATE_PASS" if passed else "COMPLETE_NO_PROMOTION",
            "completed_utc": _utc_now(),
            "plan": {"path": str(plan_path), "sha256": plan_sha256},
            "training_completion": completion_reference,
            "screen_audit": {
                "path": str(screen_report_path),
                "sha256": _sha256(screen_report_path),
            },
            "full55_gate": gate_audit,
            "promotion_gate": {
                "passed": passed,
                "required": {
                    "minimum_wins": int(plan["full55_gate"]["minimum_wins"]),
                    "minimum_cleared_levels": int(
                        plan["full55_gate"]["minimum_cleared_levels"]
                    ),
                },
                "winner": winner,
                "winner_model": winner_model,
            },
            "formal_seed_consumption": "NONE",
            "formal_candidate_authority": passed,
        }
        final_path = output_root / "final_report.json"
        _write_new(final_path, final_report)
        write_status(
            "RUNNING_INDEPENDENT_AUDIT",
            gate_passed=passed,
            final_report={"path": str(final_path), "sha256": _sha256(final_path)},
        )
        preaudit_status_path = output_root / "preaudit-controller-status.json"
        _write_new(preaudit_status_path, _read(status_path))
        auditor_path = _bound(plan["implementation"]["auditor"], "auditor")
        audit_output = Path(str(plan["outputs"]["independent_audit_receipt"]))
        audit_stdout = output_root / "independent-audit.stdout.log"
        audit_stderr = output_root / "independent-audit.stderr.log"
        with audit_stdout.open("x", encoding="utf-8") as stdout, audit_stderr.open(
            "x", encoding="utf-8"
        ) as stderr:
            audited = subprocess.run(
                [
                    sys.executable,
                    str(auditor_path),
                    "--plan",
                    str(plan_path),
                    "--expected-plan-sha256",
                    plan_sha256,
                    "--output",
                    str(audit_output),
                ],
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
        if audited.returncode != 0:
            raise RuntimeError("motor-observable independent audit failed")
        audit_receipt = _read(audit_output)
        if audit_receipt.get("status") != "PASS":
            raise ValueError("motor-observable independent audit is not PASS")
        _write_atomic(
            status_path,
            {
                "schema": "zuma-rl.alphazuma-55-motor-observable-postprocess-status",
                "version": 1,
                "status": "COMPLETE",
                "phase": "COMPLETE",
                "updated_utc": _utc_now(),
                "wall_seconds": time.perf_counter() - started,
                "plan": {"path": str(plan_path), "sha256": plan_sha256},
                "result": {"path": str(final_path), "sha256": _sha256(final_path)},
                "independent_audit": {
                    "path": str(audit_output),
                    "sha256": _sha256(audit_output),
                },
                "gate_passed": passed,
                "formal_seed_consumption": "NONE",
                "error": None,
            },
        )
        return 0
    except BaseException as error:
        failure = {
            "schema": "zuma-rl.alphazuma-55-motor-observable-postprocess-failure",
            "version": 1,
            "status": "ERROR",
            "failed_utc": _utc_now(),
            "wall_seconds": time.perf_counter() - started,
            "error_type": type(error).__name__,
            "error": str(error),
            "formal_seed_consumption": "NONE",
        }
        _write_atomic(output_root / "failure.json", failure)
        _write_atomic(
            status_path,
            {
                **failure,
                "schema": "zuma-rl.alphazuma-55-motor-observable-postprocess-status",
                "phase": "ERROR",
            },
        )
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_only:
        plan_path = args.plan.expanduser().resolve(strict=True)
        plan = validate_plan(plan_path, str(args.expected_plan_sha256))
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
        plan_path=args.plan.expanduser(),
        expected_plan_sha256=str(args.expected_plan_sha256),
        poll_seconds=max(1.0, float(args.poll_seconds)),
    )


if __name__ == "__main__":
    raise SystemExit(main())
