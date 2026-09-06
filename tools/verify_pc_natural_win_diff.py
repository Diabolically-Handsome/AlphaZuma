"""Freeze a source-bound two-tick natural-victory simulator differential."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from zuma_rl.pc_mechanism_audit import (
    validate_pc_mechanism_audit_report,
)
from zuma_rl.pc_memory_trajectory import load_memory_trajectory
from zuma_rl.pc_natural_win import compare_natural_win_with_simulator
from zuma_rl.pc_source import verify_pc_source_manifest


IMPLEMENTATION_PATHS = (
    "src/zuma_rl/original_data.py",
    "src/zuma_rl/revenge_core.py",
    "src/zuma_rl/pc_memory_evidence.py",
    "src/zuma_rl/pc_memory_trajectory.py",
    "src/zuma_rl/pc_natural_win.py",
    "src/zuma_rl/pc_source.py",
    "src/zuma_rl/pc_mechanism_audit.py",
    "src/zuma_rl/fidelity_gate.py",
    "tools/verify_pc_natural_win_diff.py",
)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _relative(path: Path, root: Path) -> str:
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(root.resolve(strict=True))
    except ValueError as error:
        raise ValueError(
            f"artifact is outside evidence root: {resolved}"
        ) from error
    pure = PurePosixPath(*relative.parts)
    if any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError("artifact path is not normalized")
    return pure.as_posix()


def _binding(path: Path, root: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"bound path is not a file: {resolved}")
    return {
        "artifact": _relative(resolved, root),
        "artifact_sha256": _sha256_path(resolved),
    }


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    root = args.evidence_root.resolve(strict=True)
    original_root = args.original_root.resolve(strict=True)
    trajectory = args.trajectory.resolve(strict=True)
    source_manifest = args.source_manifest.resolve(strict=True)
    mechanism_audit = args.mechanism_audit.resolve(strict=True)

    source_report = verify_pc_source_manifest(
        source_manifest,
        evidence_root=root,
        original_root=original_root,
    )
    if source_report.get("status") != "PASS":
        raise ValueError("source manifest did not independently pass")

    source_payload = _mapping(
        json.loads(source_manifest.read_text(encoding="utf-8")),
        "source manifest",
    )
    run = _mapping(source_payload.get("run"), "source run")
    trajectory_row = _mapping(
        run.get("trajectory_index"),
        "source trajectory binding",
    )
    if (
        trajectory_row.get("path") != _relative(trajectory, root)
        or trajectory_row.get("sha256") != _sha256_path(trajectory)
    ):
        raise ValueError("trajectory differs from source manifest")

    audit_payload = _mapping(
        json.loads(mechanism_audit.read_text(encoding="utf-8")),
        "mechanism audit",
    )
    reason = validate_pc_mechanism_audit_report(
        audit_payload,
        evidence_root=root,
        original_root=original_root,
        level_id=args.level,
        hard=args.hard,
        profile_mode=args.profile_mode,
    )
    if reason is not None:
        raise ValueError(f"mechanism audit did not pass: {reason}")
    summary = _mapping(audit_payload.get("summary"), "audit summary")
    authorizations = summary.get("source_feature_authorizations")
    expected_source = source_report.get("source_fingerprint")
    if (
        not isinstance(authorizations, list)
        or not any(
            isinstance(row, Mapping)
            and row.get("source_fingerprint") == expected_source
            and {"natural_win", "zuma_transition"}.issubset(
                set(row.get("authorized_features", ()))
            )
            for row in authorizations
        )
    ):
        raise ValueError(
            "mechanism audit does not authorize the natural-win source"
        )

    index = _mapping(
        json.loads(trajectory.read_text(encoding="utf-8")),
        "trajectory index",
    )
    frames = load_memory_trajectory(trajectory)
    report = dict(
        compare_natural_win_with_simulator(
            frames,
            original_root=original_root,
            level_id=args.level,
            hard=args.hard,
            curve_index=args.curve_index,
        )
    )
    report["profile_mode"] = args.profile_mode
    report["trajectory"] = {
        **_binding(trajectory, root),
        "source_version": index.get("version"),
        "captured_start_update": index.get("start_update"),
        "captured_end_update": index.get("end_update"),
        "captured_tick_count": index.get("tick_count"),
    }
    report["source_manifest"] = {
        **_binding(source_manifest, root),
        "source_fingerprint": expected_source,
        "source_version": source_payload.get("version"),
    }
    report["mechanism_audit"] = {
        **_binding(mechanism_audit, root),
        "audit_fingerprint": audit_payload.get("audit_fingerprint"),
        "authorized_features": ["natural_win", "zuma_transition"],
    }
    project_root = Path(__file__).resolve().parents[1]
    report["implementation"] = [
        {
            "path": relative,
            "sha256": _sha256_path(
                project_root.joinpath(
                    *PurePosixPath(relative).parts
                ).resolve(strict=True)
            ),
        }
        for relative in IMPLEMENTATION_PATHS
    ]
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--mechanism-audit", required=True, type=Path)
    parser.add_argument("--original-root", required=True, type=Path)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--level", default="Jungle2")
    parser.add_argument("--profile-mode", default="tutorials_completed")
    parser.add_argument("--curve-index", default=0, type=int)
    parser.add_argument("--hard", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output.resolve(strict=False)
    if output.exists() or not output.parent.is_dir():
        raise ValueError("output must be absent with an existing parent")
    report = build_report(args)
    data = (
        json.dumps(
            report,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")
    with output.open("xb") as stream:
        stream.write(data)
    print(
        json.dumps(
            {
                "status": report["status"],
                "path": str(output),
                "sha256": "sha256:" + hashlib.sha256(data).hexdigest(),
                "ticks_compared": report["ticks_compared"],
                "terminal_object_kind": report["terminal_object_kind"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
