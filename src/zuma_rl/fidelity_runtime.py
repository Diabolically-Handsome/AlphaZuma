"""Run the PC-evidence Fidelity Gate in its native Windows filesystem runtime."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import platform
import subprocess
from typing import Any, Mapping

from zuma_rl.fidelity_gate import (
    REPORT_SCHEMA,
    REPORT_VERSION,
    FidelityGateReport,
    FidelityGateStatus,
    verify_fidelity_suite,
)


class FidelityRuntimeError(ValueError):
    """The native verifier runtime or its machine-readable output is invalid."""


NATIVE_FIDELITY_TIMEOUT_SECONDS = 900


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _is_wsl() -> bool:
    return os.name != "nt" and (
        "microsoft" in platform.release().lower()
        or "WSL_DISTRO_NAME" in os.environ
    )


def _windows_path(path: Path) -> str:
    resolved = path.resolve(strict=True)
    parts = resolved.parts
    if len(parts) < 4 or parts[0] != "/" or parts[1] != "mnt":
        raise FidelityRuntimeError("WSL evidence path is not on a Windows drive")
    drive = parts[2]
    if len(drive) != 1 or not drive.isalpha():
        raise FidelityRuntimeError("WSL evidence path has an invalid drive")
    return str(PureWindowsPath(f"{drive.upper()}:\\", *parts[3:]))


def _windows_python(project_root: Path) -> Path:
    path = project_root / ".venv-win" / "Scripts" / "python.exe"
    if not path.is_file():
        raise FidelityRuntimeError("native Windows fidelity runtime is missing")
    return path


def windows_fidelity_runtime_identity() -> dict[str, Any]:
    """Return a recomputable identity for the Windows verifier interpreter."""

    project_root = Path(__file__).resolve().parents[2]
    executable = _windows_python(project_root)
    query = (
        "import importlib.metadata,json,platform,sys;"
        "names=('av','numpy','Pillow');"
        "print(json.dumps({'python':platform.python_version(),"
        "'implementation':platform.python_implementation(),"
        "'packages':{n:importlib.metadata.version(n) for n in names}},"
        "sort_keys=True,separators=(',',':')))"
    )
    completed = subprocess.run(
        [str(executable), "-c", query],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    try:
        runtime = json.loads(completed.stdout.strip())
    except json.JSONDecodeError as error:
        raise FidelityRuntimeError(
            "native Windows runtime identity is invalid"
        ) from error
    return {
        "role": "native_ntfs_pc_evidence_verifier",
        "path": executable.relative_to(project_root).as_posix(),
        "sha256": _sha256_path(executable),
        "runtime": runtime,
    }


def _report_from_mapping(value: Mapping[str, Any]) -> FidelityGateReport:
    required = {
        "schema",
        "version",
        "status",
        "policy",
        "evidence",
        "requirements",
        "reasons",
        "summary",
    }
    if (
        set(value) != required
        or value.get("schema") != REPORT_SCHEMA
        or value.get("version") != REPORT_VERSION
        or not isinstance(value.get("evidence"), list)
        or not isinstance(value.get("requirements"), list)
        or not isinstance(value.get("reasons"), list)
        or not isinstance(value.get("summary"), Mapping)
    ):
        raise FidelityRuntimeError("native Fidelity report shape is invalid")
    try:
        status = FidelityGateStatus(value["status"])
    except (TypeError, ValueError) as error:
        raise FidelityRuntimeError("native Fidelity status is invalid") from error
    if not all(isinstance(item, Mapping) for item in value["evidence"]):
        raise FidelityRuntimeError("native Fidelity evidence rows are invalid")
    if not all(isinstance(item, Mapping) for item in value["requirements"]):
        raise FidelityRuntimeError("native Fidelity requirement rows are invalid")
    if not all(isinstance(item, str) for item in value["reasons"]):
        raise FidelityRuntimeError("native Fidelity reasons are invalid")
    policy = value["policy"]
    if policy is not None and not isinstance(policy, str):
        raise FidelityRuntimeError("native Fidelity policy is invalid")
    return FidelityGateReport(
        status=status,
        policy=policy,
        evidence=tuple(dict(item) for item in value["evidence"]),
        requirements=tuple(dict(item) for item in value["requirements"]),
        reasons=tuple(value["reasons"]),
        summary=dict(value["summary"]),
    )


def verify_fidelity_suite_for_training(
    suite_path: str | Path,
    *,
    suite_root: str | Path | None,
    original_root: str | Path | None,
) -> FidelityGateReport:
    """Use Windows for PC evidence under WSL; otherwise verify in-process."""

    if not _is_wsl():
        return verify_fidelity_suite(
            suite_path,
            suite_root=suite_root,
            original_root=original_root,
        )
    if suite_root is None or original_root is None:
        raise FidelityRuntimeError(
            "WSL native Fidelity verification requires both roots"
        )
    project_root = Path(__file__).resolve().parents[2]
    executable = _windows_python(project_root)
    command = [
        str(executable),
        "-m",
        "zuma_rl.fidelity_gate",
        _windows_path(Path(suite_path)),
        "--suite-root",
        _windows_path(Path(suite_root)),
        "--original-root",
        _windows_path(Path(original_root)),
        "--compact",
    ]
    environment = dict(os.environ)
    environment["PYTHONUTF8"] = "1"
    try:
        completed = subprocess.run(
            command,
            cwd=project_root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=NATIVE_FIDELITY_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as error:
        raise FidelityRuntimeError(
            "native Windows Fidelity verifier timed out after "
            f"{NATIVE_FIDELITY_TIMEOUT_SECONDS} seconds"
        ) from error
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if completed.returncode not in {0, 1, 2} or not lines:
        raise FidelityRuntimeError(
            "native Windows Fidelity verifier did not return a report"
        )
    try:
        raw = json.loads(lines[-1])
    except json.JSONDecodeError as error:
        raise FidelityRuntimeError(
            "native Windows Fidelity output is invalid JSON"
        ) from error
    if not isinstance(raw, Mapping):
        raise FidelityRuntimeError("native Windows Fidelity output is not an object")
    report = _report_from_mapping(raw)
    if report.exit_code != completed.returncode:
        raise FidelityRuntimeError(
            "native Windows Fidelity exit code differs from its report"
        )
    return report


__all__ = [
    "FidelityRuntimeError",
    "NATIVE_FIDELITY_TIMEOUT_SECONDS",
    "verify_fidelity_suite_for_training",
    "windows_fidelity_runtime_identity",
]
