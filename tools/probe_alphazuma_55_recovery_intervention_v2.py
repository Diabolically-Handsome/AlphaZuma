"""Probe the rolling-error, deep-recovery AlphaZuma controller."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tools import probe_alphazuma_55_recovery_intervention_v1 as base
from tools.alphazuma_55_recovery_intervention_v2 import (
    RecoveryInterventionConfigV2,
    RecoveryInterventionControllerV2,
)
from zuma_rl.alphazuma_55 import INCLUDED_LEVELS


SCRIPT_PATH = Path(__file__).resolve()
CONTROLLER_PATH = (
    SCRIPT_PATH.parent / "alphazuma_55_recovery_intervention_v2.py"
).resolve()
EXPECTED_MODES = base.EXPECTED_MODES


def _validate_preregistration(path: Path) -> dict[str, Any]:
    prereg = base._read(path)
    if not (
        prereg.get("schema")
        == "zuma-rl.alphazuma-55-recovery-intervention-v2-probe-preregistration"
        and prereg.get("version") == 1
        and prereg.get("status") == "FROZEN_BEFORE_PROBE"
    ):
        raise ValueError("unexpected recovery intervention V2 preregistration")
    base._bound(prereg.get("probe"), SCRIPT_PATH, "probe")
    base._bound(prereg.get("controller"), CONTROLLER_PATH, "controller")

    master = prereg.get("master_preregistration", {})
    if not isinstance(master, dict):
        raise ValueError("master reference must be an object")
    master_path = Path(str(master.get("path", ""))).resolve(strict=True)
    if master.get("sha256") != base._sha256(master_path):
        raise ValueError("master preregistration bytes changed")
    master_value = base._read(master_path)
    if master_value.get("schema") != (
        "zuma-rl.alphazuma-55-weekend-master-preregistration"
    ):
        raise ValueError("unexpected AlphaZuma 55 master")

    evidence = prereg.get("v1_failure_evidence", {})
    if not isinstance(evidence, dict):
        raise ValueError("V1 failure evidence must be an object")
    evidence_path = Path(str(evidence.get("path", ""))).resolve(strict=True)
    if evidence.get("sha256") != base._sha256(evidence_path):
        raise ValueError("V1 failure evidence bytes changed")
    evidence_value = base._read(evidence_path)
    v1_recovery = next(
        row
        for row in evidence_value.get("modes", [])
        if row.get("mode") == "recovery_intervention"
    )
    if not (
        evidence_value.get("status") == "COMPLETE"
        and evidence_value.get("decision", {}).get("status") == "FAIL"
        and int(v1_recovery.get("summary", {}).get("wins", -1)) == 0
        and int(v1_recovery.get("summary", {}).get("interventions", -1)) == 19
    ):
        raise ValueError("V1 failure evidence no longer matches the correction")

    source = prereg.get("source", {})
    if not isinstance(source, dict):
        raise ValueError("source must be an object")
    model_path = Path(str(source.get("model_path", ""))).resolve(strict=True)
    completion_path = Path(
        str(source.get("completion_path", ""))
    ).resolve(strict=True)
    if source.get("model_sha256") != base._sha256(model_path):
        raise ValueError("source model bytes changed")
    if source.get("completion_sha256") != base._sha256(completion_path):
        raise ValueError("source completion bytes changed")
    completion = base._read(completion_path)
    final = completion.get("final_model", {})
    if not (
        completion.get("status") == "COMPLETE"
        and completion.get("route_version") == "v3_retry_audited"
        and completion.get("formal_seed_consumption") is False
        and Path(str(final.get("path", ""))).resolve() == model_path
        and final.get("sha256") == source.get("model_sha256")
    ):
        raise ValueError("source is not the frozen motor-observable V3 model")

    levels = tuple(str(value) for value in prereg.get("levels", ()))
    if not levels or len(levels) != len(set(levels)):
        raise ValueError("probe levels must be non-empty and unique")
    canonical = {value.casefold() for value in INCLUDED_LEVELS}
    if any(level.casefold() not in canonical for level in levels):
        raise ValueError("probe level escaped the included full55 inventory")

    probe = prereg.get("paired_probe", {})
    if not isinstance(probe, dict):
        raise ValueError("paired_probe must be an object")
    if tuple(str(value) for value in probe.get("modes", ())) != EXPECTED_MODES:
        raise ValueError("paired probe modes changed")
    seed_base = int(probe.get("seed_base", -1))
    seed_last = int(probe.get("seed_last", -1))
    if seed_last != seed_base + len(levels) - 1:
        raise ValueError("paired probe seed interval changed")
    registry = master_value["seed_registry"]["engineering_and_calibration"]
    if not int(registry["first"]) <= seed_base <= seed_last <= int(
        registry["last"]
    ):
        raise ValueError("probe seeds escaped engineering_and_calibration")
    if not (
        int(probe.get("parallel_envs", -1)) == len(levels)
        and int(probe.get("max_ticks", -1)) == 30000
        and probe.get("paired_modes_share_identical_task_seeds") is True
        and probe.get("deterministic_student_inference") is True
    ):
        raise ValueError("paired probe execution contract changed")
    frozen_controller = RecoveryInterventionConfigV2(
        **dict(probe.get("controller_config", {}))
    )
    if frozen_controller.contract() != probe.get("controller_config"):
        raise ValueError("V2 controller configuration is not canonical")

    expected_acceptance = {
        "intervention_wins_must_exceed_student_only": True,
        "intervention_total_score_must_exceed_student_only": True,
        "minimum_paired_score_improvements": 6,
        "minimum_interventions": 6,
        "minimum_recovery_successes": 1,
        "maximum_teacher_execution_fraction": 0.6,
    }
    if prereg.get("acceptance") != expected_acceptance:
        raise ValueError("V2 probe acceptance contract changed")
    boundary = prereg.get("authority_boundary", {})
    for name in (
        "training_authority",
        "candidate_authority",
        "formal_selection_seed_consumption",
        "formal_final_blind_seed_consumption",
        "continuous_campaign_seed_consumption",
        "power_change_authority",
    ):
        if boundary.get(name) is not False:
            raise ValueError(f"V2 probe authority changed: {name}")
    return prereg


def run(*, preregistration_path: Path, original_root: Path) -> dict[str, Any]:
    preregistration_path = preregistration_path.resolve(strict=True)
    original_root = original_root.resolve(strict=True)
    prereg = _validate_preregistration(preregistration_path)
    run_dir = Path(str(prereg["run_dir"])).resolve()
    if run_dir.exists():
        raise FileExistsError(f"V2 probe output already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    started = time.perf_counter()
    base._write_atomic(
        run_dir / "config.json",
        {
            "schema": "zuma-rl.alphazuma-55-recovery-intervention-v2-probe-config",
            "version": 1,
            "status": "FROZEN",
            "preregistration": {
                "path": str(preregistration_path),
                "sha256": base._sha256(preregistration_path),
            },
            "probe": {"path": str(SCRIPT_PATH), "sha256": base._sha256(SCRIPT_PATH)},
            "controller": {
                "path": str(CONTROLLER_PATH),
                "sha256": base._sha256(CONTROLLER_PATH),
            },
            "paired_probe": prereg["paired_probe"],
        },
    )
    model: Any | None = None
    original_controller = base.RecoveryInterventionController
    try:
        import torch
        from sb3_contrib import MaskablePPO

        torch.set_num_threads(1)
        model = MaskablePPO.load(
            prereg["source"]["model_path"],
            device=str(prereg["paired_probe"]["device"]),
        )
        controller_config = RecoveryInterventionConfigV2(
            **dict(prereg["paired_probe"]["controller_config"])
        )
        base.RecoveryInterventionController = RecoveryInterventionControllerV2
        modes = []
        for mode in EXPECTED_MODES:
            base._write_atomic(
                run_dir / "status.json",
                {
                    "schema": (
                        "zuma-rl.alphazuma-55-recovery-intervention-v2-probe-status"
                    ),
                    "version": 1,
                    "status": "RUNNING",
                    "stage": mode,
                    "updated_utc": base._utc_now(),
                    "completed_modes": [row["mode"] for row in modes],
                    "wall_seconds": time.perf_counter() - started,
                },
            )
            result = base._run_mode(
                mode=mode,
                model=model,
                original_root=original_root,
                level_ids=prereg["levels"],
                seed_base=int(prereg["paired_probe"]["seed_base"]),
                max_ticks=int(prereg["paired_probe"]["max_ticks"]),
                controller_config=controller_config,
            )
            modes.append(result)
            base._write_atomic(run_dir / f"{mode}.json", result)
        decision = base._decision(prereg, modes)
        completion = {
            "schema": "zuma-rl.alphazuma-55-recovery-intervention-v2-probe-completion",
            "version": 1,
            "status": "COMPLETE",
            "completed_utc": base._utc_now(),
            "wall_seconds": time.perf_counter() - started,
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "torch": torch.__version__,
                "device": str(prereg["paired_probe"]["device"]),
                "cuda_device": torch.cuda.get_device_name(
                    torch.device(str(prereg["paired_probe"]["device"]))
                ),
            },
            "source": prereg["source"],
            "v1_failure_evidence": prereg["v1_failure_evidence"],
            "modes": modes,
            "decision": decision,
            "formal_seed_consumption": False,
            "formal_candidate_authority": False,
            "training_authority": False,
        }
        base._write_atomic(run_dir / "completion.json", completion)
        return completion
    except BaseException as error:
        base._write_atomic(
            run_dir / "failure.json",
            {
                "schema": "zuma-rl.alphazuma-55-recovery-intervention-v2-probe-failure",
                "version": 1,
                "status": "FAILED",
                "failed_utc": base._utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
                "formal_seed_consumption": False,
                "training_authority": False,
            },
        )
        raise
    finally:
        base.RecoveryInterventionController = original_controller
        del model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--original-root", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prereg_path = args.preregistration.expanduser().resolve(strict=True)
    original_root = args.original_root.expanduser().resolve(strict=True)
    prereg = _validate_preregistration(prereg_path)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "VALID",
                    "preregistration": {
                        "path": str(prereg_path),
                        "sha256": base._sha256(prereg_path),
                    },
                    "levels": len(prereg["levels"]),
                    "modes": list(EXPECTED_MODES),
                    "formal_seed_consumption": False,
                    "training_authority": False,
                },
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        )
        return 0
    result = run(
        preregistration_path=prereg_path,
        original_root=original_root,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "decision": result["decision"]["status"],
                "formal_seed_consumption": False,
                "training_authority": False,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
