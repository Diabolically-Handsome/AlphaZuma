"""Launch and supervise intent-wide training after effective-wide exits."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools.build_alphazuma_55_eval_contract import _read_json, _sha256


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pid_alive(pid: int) -> bool:
    return pid > 0 and Path(f"/proc/{pid}").exists()


def _atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _bound(reference: Any, name: str) -> Path:
    if not isinstance(reference, dict):
        raise ValueError(f"{name} reference is absent")
    path = Path(str(reference.get("path", ""))).resolve(strict=True)
    if reference.get("sha256") != _sha256(path):
        raise ValueError(f"{name} hash differs")
    return path


def run(plan_path: Path, expected_plan_sha256: str, poll_seconds: float) -> int:
    plan_path = plan_path.resolve(strict=True)
    if _sha256(plan_path) != expected_plan_sha256:
        raise ValueError("intent-wide launch plan hash differs")
    plan = _read_json(plan_path)
    if (
        plan.get("schema")
        != "zuma-rl.alphazuma-55-polar-intent-wide-launch-plan"
        or plan.get("version") != 1
        or plan.get("status") != "FROZEN_BEFORE_WAIT"
    ):
        raise ValueError("unexpected intent-wide launch plan")
    watcher = _bound(plan["implementation"]["watcher"], "watcher")
    if watcher != SCRIPT_PATH:
        raise ValueError("launch plan binds another watcher")
    preregistration = _bound(
        plan["intent_preregistration"], "intent-wide preregistration"
    )
    _bound(
        plan["source_effective_wide_launch_plan"],
        "effective-wide launch plan",
    )
    trainer = _bound(plan["implementation"]["trainer"], "intent-wide trainer")
    original_root = Path(str(plan["original_root"])).resolve(strict=True)
    status_root = Path(str(plan["outputs"]["status_root"])).resolve()
    status_root.mkdir(parents=True, exist_ok=False)
    status_path = status_root / "watcher_status.json"
    source = plan["source_terminal"]
    source_completion = Path(str(source["completion"])).resolve()
    source_failure = Path(str(source["failure"])).resolve()
    source_pid = int(source["watcher_pid_at_registration"])
    state: dict[str, Any] = {
        "schema": "zuma-rl.alphazuma-55-polar-intent-wide-watcher-status",
        "version": 1,
        "status": "RUNNING",
        "stage": "WAITING_FOR_EFFECTIVE_WIDE_TERMINAL",
        "started_utc": _utc_now(),
        "updated_utc": _utc_now(),
        "plan": {"path": str(plan_path), "sha256": expected_plan_sha256},
        "source_watcher_pid": source_pid,
        "formal_seed_consumption": False,
        "error": None,
    }
    _atomic(status_path, state)
    try:
        while not source_completion.exists() and not source_failure.exists():
            if not _pid_alive(source_pid):
                raise RuntimeError(
                    "effective-wide watcher exited without a terminal receipt"
                )
            state["updated_utc"] = _utc_now()
            _atomic(status_path, state)
            time.sleep(poll_seconds)
        source_receipt_path = (
            source_completion if source_completion.exists() else source_failure
        )
        source_receipt = _read_json(source_receipt_path)
        if source_receipt.get("status") not in {"COMPLETE", "FAILED"}:
            raise RuntimeError("effective-wide terminal receipt is invalid")
        while _pid_alive(source_pid):
            time.sleep(min(poll_seconds, 5.0))
        state.update(
            {
                "stage": "TRAINING",
                "updated_utc": _utc_now(),
                "source_terminal": {
                    "path": str(source_receipt_path),
                    "sha256": _sha256(source_receipt_path),
                    "status": source_receipt["status"],
                },
            }
        )
        command = [
            sys.executable,
            str(trainer),
            "--preregistration",
            str(preregistration),
            "--original-root",
            str(original_root),
        ]
        environment = os.environ.copy()
        environment.update(
            {
                "TMPDIR": "/tmp",
                "TEMP": "/tmp",
                "PYTHONPATH": str(PROJECT_ROOT / "src"),
                "PYTHONUNBUFFERED": "1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            }
        )
        with (status_root / "trainer.stdout.log").open(
            "x", encoding="utf-8", newline="\n"
        ) as stdout, (status_root / "trainer.stderr.log").open(
            "x", encoding="utf-8", newline="\n"
        ) as stderr:
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                text=True,
            )
            state["trainer"] = {"pid": process.pid, "command": command}
            _atomic(status_path, state)
            return_code = process.wait()
        target_completion = Path(str(plan["target"]["completion"])).resolve()
        target_failure = Path(str(plan["target"]["failure"])).resolve()
        if return_code != 0 or not target_completion.exists():
            raise RuntimeError(
                f"intent-wide trainer failed with code {return_code}; "
                f"failure={target_failure}"
            )
        completion = _read_json(target_completion)
        if (
            completion.get("status") != "COMPLETE"
            or completion.get("training_label_semantics")
            != "raw_teacher_intent"
        ):
            raise RuntimeError("intent-wide completion receipt is invalid")
        state.update(
            {
                "status": "COMPLETE",
                "stage": "COMPLETE",
                "updated_utc": _utc_now(),
                "completion": {
                    "path": str(target_completion),
                    "sha256": _sha256(target_completion),
                    "model": completion["final_model"],
                    "policy_parameter_count": completion[
                        "policy_parameter_count"
                    ],
                    "training_label_semantics": completion[
                        "training_label_semantics"
                    ],
                },
            }
        )
        _atomic(status_path, state)
        return 0
    except BaseException as error:
        state.update(
            {
                "status": "FAILED",
                "stage": "FAILED",
                "updated_utc": _utc_now(),
                "error": {"type": type(error).__name__, "message": str(error)},
                "formal_seed_consumption": False,
            }
        )
        _atomic(status_path, state)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.poll_seconds <= 0:
        raise SystemExit("poll-seconds must be positive")
    return run(
        args.plan.expanduser(), str(args.expected_plan_sha256), args.poll_seconds
    )


if __name__ == "__main__":
    raise SystemExit(main())
