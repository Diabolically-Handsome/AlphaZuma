"""Audit recoverability of every formal AlphaZuma V1.1 training route."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_TIMESTEP_PATTERN = re.compile(r"_(\d+)_steps\.zip$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _checkpoint_timesteps(path: Path) -> int:
    match = _TIMESTEP_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"checkpoint filename has no timestep suffix: {path}")
    return int(match.group(1))


def _select_latest_stable_checkpoint(
    checkpoint_dir: Path,
    *,
    now_ns: int,
    stable_age_seconds: float,
) -> tuple[Path, int, float]:
    candidates: list[tuple[int, Path, float]] = []
    for path in checkpoint_dir.glob("*.zip"):
        try:
            timesteps = _checkpoint_timesteps(path)
        except ValueError:
            continue
        age_seconds = (now_ns - path.stat().st_mtime_ns) / 1_000_000_000
        if age_seconds >= stable_age_seconds:
            candidates.append((timesteps, path, age_seconds))
    if not candidates:
        raise FileNotFoundError(
            f"no checkpoint at least {stable_age_seconds}s old in {checkpoint_dir}"
        )
    candidates.sort(key=lambda row: (row[0], row[1].name))
    if len(candidates) >= 2 and candidates[-2][0] == candidates[-1][0]:
        raise RuntimeError(
            f"ambiguous latest checkpoint timestep in {checkpoint_dir}"
        )
    timesteps, path, age_seconds = candidates[-1]
    return path, timesteps, age_seconds


def _write_json_atomic(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    os.replace(temporary, path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-preregistration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stable-age-seconds", type=float, default=30.0)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not math.isfinite(args.stable_age_seconds) or args.stable_age_seconds < 0:
        raise ValueError("--stable-age-seconds must be finite and non-negative")
    master_path = args.master_preregistration.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    master = json.loads(master_path.read_text(encoding="utf-8"))
    if master.get("status") != "FROZEN_BEFORE_SELECTION":
        raise ValueError("master preregistration is not frozen")
    runs = master.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("master preregistration has no formal runs")

    from sb3_contrib import MaskablePPO

    now_ns = time.time_ns()
    rows: list[dict[str, Any]] = []
    for run in runs:
        run_id = str(run["id"])
        preregistration_path = Path(run["preregistration_path"])
        preregistration_sha256 = _sha256(preregistration_path)
        if preregistration_sha256 != run["preregistration_sha256"]:
            raise ValueError(f"training preregistration hash changed for {run_id}")
        run_dir = Path(run["run_dir"])
        checkpoint_path, filename_timesteps, age_seconds = (
            _select_latest_stable_checkpoint(
                run_dir / "checkpoints",
                now_ns=now_ns,
                stable_age_seconds=args.stable_age_seconds,
            )
        )
        before = checkpoint_path.stat()
        checkpoint_sha256 = _sha256(checkpoint_path)
        model = MaskablePPO.load(checkpoint_path, device=args.device)
        after = checkpoint_path.stat()
        if (before.st_size, before.st_mtime_ns) != (
            after.st_size,
            after.st_mtime_ns,
        ):
            raise RuntimeError(f"checkpoint changed during audit: {checkpoint_path}")
        observation_shape = list(model.observation_space.shape or ())
        action_nvec = [int(value) for value in model.action_space.nvec]
        optimizer_state_entries = len(model.policy.optimizer.state)
        parameter_tensors = list(model.policy.parameters())
        all_parameters_finite = all(
            bool(parameter.detach().isfinite().all().item())
            for parameter in parameter_tensors
        )
        checks = {
            "filename_matches_internal_timesteps": (
                filename_timesteps == int(model.num_timesteps)
            ),
            "observation_shape_matches": observation_shape == [21220],
            "action_nvec_matches": action_nvec == [3, 180],
            "optimizer_state_entries_match": optimizer_state_entries == 24,
            "all_policy_parameters_finite": all_parameters_finite,
            "file_unchanged_during_audit": True,
        }
        rows.append(
            {
                "run": run_id,
                "path": str(checkpoint_path),
                "bytes": before.st_size,
                "sha256": checkpoint_sha256,
                "stable_age_seconds_at_selection": age_seconds,
                "filename_timesteps": filename_timesteps,
                "model_timesteps": int(model.num_timesteps),
                "observation_shape": observation_shape,
                "action_nvec": action_nvec,
                "optimizer_state_entries": optimizer_state_entries,
                "policy_parameter_tensor_count": len(parameter_tensors),
                "checks": checks,
                "status": "PASS" if all(checks.values()) else "FAIL",
            }
        )

    passed = sum(row["status"] == "PASS" for row in rows)
    report = {
        "schema": "zuma-rl.alphazuma-v11-checkpoint-recovery-audit",
        "version": 2,
        "completed_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "status": "PASS" if passed == len(runs) else "FAIL",
        "classification": "training_integrity_diagnostic_only",
        "method": (
            "Load the latest stable checkpoint from every formal route on CPU; "
            "verify immutable file metadata, SHA-256, filename/internal timestep, "
            "observation/action contracts, optimizer state, and finite parameters."
        ),
        "inputs": {
            "master_preregistration": {
                "path": str(master_path),
                "sha256": _sha256(master_path),
            },
            "auditor": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256(Path(__file__).resolve()),
            },
            "stable_age_seconds": args.stable_age_seconds,
            "device": args.device,
        },
        "validation": {
            "formal_routes": len(runs),
            "audited_routes": len(rows),
            "passed_routes": passed,
            "all_formal_routes_passed": passed == len(runs),
        },
        "checkpoints": rows,
        "formal_gate_effect": "none",
    }
    _write_json_atomic(output_path, report)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
