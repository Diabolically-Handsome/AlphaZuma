"""Freeze an exact retail-Board fruit-visual simulator differential."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

from zuma_rl.pc_fruit_visual_diff import (
    canonical_fruit_visual_report_bytes,
    compare_fruit_visual_trajectory,
)
from zuma_rl.pc_memory_trajectory import load_memory_trajectory


IMPLEMENTATION_PATHS = (
    "src/zuma_rl/original_data.py",
    "src/zuma_rl/revenge_core.py",
    "src/zuma_rl/pc_memory_trajectory.py",
    "src/zuma_rl/pc_gameplay_diff.py",
    "src/zuma_rl/pc_fruit_visual_diff.py",
    "tools/verify_pc_fruit_visual.py",
)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _binding(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"bound path is not a file: {resolved}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256_path(resolved),
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    trajectory = args.trajectory.resolve(strict=True)
    original_root = args.original_root.resolve(strict=True)
    runtime = args.runtime.resolve(strict=True)
    source_manifest = args.source_manifest.resolve(strict=True)
    frames = load_memory_trajectory(
        trajectory,
        start_update=args.start_update,
        end_update=args.end_update,
    )
    report = dict(
        compare_fruit_visual_trajectory(
            frames,
            original_root=original_root,
            level_id=args.level,
            hard=args.hard,
            curve_index=args.curve_index,
        )
    )
    project_root = Path(__file__).resolve().parents[1]
    implementation = []
    for relative in IMPLEMENTATION_PATHS:
        pure = PurePosixPath(relative)
        path = project_root.joinpath(*pure.parts).resolve(strict=True)
        implementation.append(
            {
                "path": relative,
                "sha256": _sha256_path(path),
            }
        )
    source_value = json.loads(source_manifest.read_text(encoding="utf-8"))
    if not isinstance(source_value, dict):
        raise ValueError("source manifest must be a JSON object")
    report["provenance"] = {
        "trajectory_index": _binding(trajectory),
        "source_manifest": {
            **_binding(source_manifest),
            "schema": source_value.get("schema"),
            "version": source_value.get("version"),
            "status": source_value.get("status"),
        },
        "original_root": str(original_root),
        "runtime_executable": _binding(runtime),
        "implementation": implementation,
    }
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--original-root", required=True, type=Path)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--level", default="Jungle2")
    parser.add_argument("--curve-index", default=0, type=int)
    parser.add_argument("--hard", action="store_true")
    parser.add_argument("--start-update", type=int)
    parser.add_argument("--end-update", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output.resolve(strict=False)
    if output.exists() or not output.parent.is_dir():
        raise ValueError("output must be absent with an existing parent")
    report = build_report(args)
    data = canonical_fruit_visual_report_bytes(report)
    with output.open("xb") as stream:
        stream.write(data)
    print(
        json.dumps(
            {
                "status": report["status"],
                "path": str(output),
                "sha256": "sha256:" + hashlib.sha256(data).hexdigest(),
                "transition_count": report["transition_count"],
                "exact_transition_count": report["exact_transition_count"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
