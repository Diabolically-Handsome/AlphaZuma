"""Wait for one frozen AlphaZuma 55 validation matrix and audit it."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


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


def _write_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _write_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError:
        raise FileExistsError(f"refusing to overwrite immutable evidence: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _bound_path(path: Path, expected_hash: str, name: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    _require(_sha256(resolved) == expected_hash, f"{name} hash mismatch")
    return resolved


def _process_command(pid: int) -> str | None:
    try:
        raw = (Path("/proc") / str(pid) / "cmdline").read_bytes()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    return raw.replace(b"\0", b" ").decode("utf-8", errors="replace").strip()


def _expected_matrix_attempts(
    preregistration: dict[str, Any], manifest: dict[str, Any]
) -> int:
    models = manifest.get("models")
    _require(isinstance(models, list) and models, "models manifest is empty")
    attempts_per_model = int(preregistration["total_attempts"])
    _require(attempts_per_model > 0, "per-model attempt count must be positive")
    return attempts_per_model * len(models)


class StatusWriter:
    def __init__(self, root: Path, bindings: dict[str, Any]) -> None:
        self.root = root
        self.status_path = root / "controller_status.json"
        self.started_utc = datetime.now(timezone.utc).isoformat()
        self.bindings = bindings

    def update(self, status: str, **extra: Any) -> None:
        _write_atomic(
            self.status_path,
            {
                "schema": "zuma-rl.alphazuma-55-training-validation-watcher",
                "version": 1,
                "status": status,
                "started_utc": self.started_utc,
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "controller": {
                    "path": str(SCRIPT_PATH),
                    "sha256": _sha256(SCRIPT_PATH),
                    "pid": os.getpid(),
                },
                "bindings": self.bindings,
                "formal_selection_authority": False,
                "training_recipe_change_authority": False,
                "formal_seed_consumption": False,
                **extra,
            },
        )


def execute(args: argparse.Namespace) -> dict[str, Any]:
    project_root = SCRIPT_PATH.parents[1]
    master = _bound_path(
        args.master_preregistration, args.expected_master_sha256, "master preregistration"
    )
    prereg = _bound_path(
        args.preregistration, args.expected_preregistration_sha256, "validation preregistration"
    )
    manifest = _bound_path(
        args.models_manifest, args.expected_models_manifest_sha256, "models manifest"
    )
    auditor = _bound_path(args.auditor, args.expected_auditor_sha256, "auditor")
    _require(
        _sha256(SCRIPT_PATH) == args.expected_controller_sha256,
        "watcher hash mismatch",
    )
    matrix_path = args.matrix_shard.expanduser().resolve(strict=True)
    audit_output = args.audit_output.expanduser().resolve()
    status_root = args.status_root.expanduser().resolve()
    _require(not status_root.exists(), "watcher status root already exists")
    status_root.mkdir(parents=True)

    prereg_value = _read_json(prereg)
    manifest_value = _read_json(manifest)
    expected_attempts = _expected_matrix_attempts(prereg_value, manifest_value)
    expected_matrix = (
        Path(str(prereg_value["execution"]["output_root"])).resolve()
        / "matrix-shard-00-of-01.json"
    )
    _require(matrix_path == expected_matrix, "matrix path differs from preregistration")
    _require(
        prereg_value["diagnostic_contract"]["formal_selection_authority"] is False
        and prereg_value["diagnostic_contract"]["formal_seed_ranges_consumed"] is False,
        "validation contract exceeds engineering authority",
    )
    _require(not audit_output.exists(), "audit receipt already exists")

    bindings = {
        "master_preregistration": {"path": str(master), "sha256": _sha256(master)},
        "preregistration": {"path": str(prereg), "sha256": _sha256(prereg)},
        "models_manifest": {"path": str(manifest), "sha256": _sha256(manifest)},
        "matrix_shard": {"path": str(matrix_path)},
        "auditor": {"path": str(auditor), "sha256": _sha256(auditor)},
        "audit_output": str(audit_output),
        "evaluator_pid": int(args.evaluator_pid),
    }
    writer = StatusWriter(status_root, bindings)
    last_completed = -1
    while True:
        matrix = _read_json(matrix_path)
        completed = int(matrix["completed_attempts"])
        _require(
            int(matrix["expected_attempts"]) == expected_attempts,
            "matrix expected attempts changed",
        )
        _require(completed >= last_completed, "matrix progress moved backwards")
        last_completed = completed
        status = str(matrix["status"])
        if status == "COMPLETE":
            _require(
                completed == expected_attempts and matrix.get("error") is None,
                "matrix completed without the full frozen attempt count",
            )
            break
        _require(status == "RUNNING", f"matrix entered terminal status {status}")
        command = _process_command(int(args.evaluator_pid))
        _require(command is not None, "evaluator process disappeared before completion")
        _require(
            prereg.name in command and "evaluate_zero_shot_multilevel_v2.py" in command,
            "bound PID is not the frozen evaluator",
        )
        writer.update(
            "WAITING_FOR_MATRIX",
            matrix={
                "status": status,
                "completed_attempts": completed,
                "expected_attempts": expected_attempts,
                "wall_seconds": matrix.get("runtime", {}).get("wall_seconds"),
                "error": matrix.get("error"),
            },
            evaluator={"pid": int(args.evaluator_pid), "command": command},
        )
        time.sleep(args.poll_seconds)

    writer.update(
        "RUNNING_INDEPENDENT_AUDIT",
        matrix={
            "status": "COMPLETE",
            "completed_attempts": expected_attempts,
            "expected_attempts": expected_attempts,
            "sha256": _sha256(matrix_path),
        },
    )
    log_path = status_root / "independent-audit.log"
    environment = os.environ.copy()
    environment.update({"TMPDIR": "/tmp", "TEMP": "/tmp", "PYTHONUNBUFFERED": "1"})
    command = [
        sys.executable,
        str(auditor),
        "--master-preregistration",
        str(master),
        "--preregistration",
        str(prereg),
        "--models-manifest",
        str(manifest),
        "--matrix-shard",
        str(matrix_path),
        "--output",
        str(audit_output),
    ]
    with log_path.open("x", encoding="utf-8", newline="\n") as log:
        completed_process = subprocess.run(
            command,
            cwd=project_root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    _require(completed_process.returncode == 0, "independent auditor failed")
    receipt = _read_json(audit_output.resolve(strict=True))
    _require(receipt.get("status") == "PASS", "independent audit did not PASS")
    writer.update(
        "COMPLETE",
        completed_utc=datetime.now(timezone.utc).isoformat(),
        matrix={"path": str(matrix_path), "sha256": _sha256(matrix_path)},
        audit={"path": str(audit_output), "sha256": _sha256(audit_output)},
    )
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-preregistration", required=True, type=Path)
    parser.add_argument("--expected-master-sha256", required=True)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--expected-preregistration-sha256", required=True)
    parser.add_argument("--models-manifest", required=True, type=Path)
    parser.add_argument("--expected-models-manifest-sha256", required=True)
    parser.add_argument("--matrix-shard", required=True, type=Path)
    parser.add_argument("--auditor", required=True, type=Path)
    parser.add_argument("--expected-auditor-sha256", required=True)
    parser.add_argument("--expected-controller-sha256", required=True)
    parser.add_argument("--audit-output", required=True, type=Path)
    parser.add_argument("--evaluator-pid", required=True, type=int)
    parser.add_argument("--status-root", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        execute(args)
    except BaseException as error:
        root = args.status_root.expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        failure_path = root / "failure.json"
        if not failure_path.exists():
            _write_exclusive(
                failure_path,
                {
                    "schema": "zuma-rl.alphazuma-55-training-validation-watcher-failure",
                    "version": 1,
                    "status": "ERROR",
                    "failed_utc": datetime.now(timezone.utc).isoformat(),
                    "error": {"type": type(error).__name__, "message": str(error)},
                },
            )
        _write_atomic(
            root / "controller_status.json",
            {
                "schema": "zuma-rl.alphazuma-55-training-validation-watcher",
                "version": 1,
                "status": "ERROR",
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "failure": {"path": str(failure_path), "sha256": _sha256(failure_path)},
            },
        )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
