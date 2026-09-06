"""Recompute and freeze a source-bound full-state gameplay differential."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

from zuma_rl.pc_gameplay_diff import compare_gameplay_trajectory
from zuma_rl.pc_mechanism_audit import derive_native_mechanism_features
from zuma_rl.pc_memory_trajectory import load_memory_trajectory
from zuma_rl.popcap_dmo import PopCapDemo


IMPLEMENTATION_PATHS = (
    "src/zuma_rl/original_data.py",
    "src/zuma_rl/revenge_core.py",
    "src/zuma_rl/popcap_dmo.py",
    "src/zuma_rl/pc_memory_trajectory.py",
    "src/zuma_rl/pc_merge_diff.py",
    "src/zuma_rl/pc_gameplay_diff.py",
    "src/zuma_rl/pc_mechanism_audit.py",
    "tools/verify_pc_gameplay_diff.py",
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
        raise ValueError(f"artifact is outside evidence root: {resolved}") from error
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


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    root = args.evidence_root.resolve(strict=True)
    trajectory = args.trajectory.resolve(strict=True)
    dmo_path = args.dmo.resolve(strict=True)
    original_root = args.original_root.resolve(strict=True)
    runtime = args.runtime.resolve(strict=True)
    source_manifest = args.source_manifest.resolve(strict=True)
    fruit_visual_diff = args.fruit_visual_diff.resolve(strict=True)
    index = json.loads(trajectory.read_text(encoding="utf-8"))
    if not isinstance(index, dict):
        raise ValueError("trajectory index must be a JSON object")
    frames = load_memory_trajectory(
        trajectory,
        start_update=args.start_update,
        end_update=args.end_update,
    )
    report = dict(
        compare_gameplay_trajectory(
            frames,
            demo=PopCapDemo.read(dmo_path),
            original_root=original_root,
            hard=args.hard,
            level_id=args.level,
            curve_index=args.curve_index,
            synchronize_shooter=False,
            reconcile_external_visual_mtrand=True,
        )
    )
    report["trajectory"] = {
        **_binding(trajectory, root),
        "source_version": index.get("version"),
        "captured_start_update": index.get("start_update"),
        "captured_end_update": index.get("end_update"),
        "selected_start_update": frames[0].update,
        "selected_end_update": frames[-1].update,
    }
    source_features, source_proofs = derive_native_mechanism_features(
        frames,
        original_root=original_root,
        level_id=args.level,
        hard=args.hard,
        curve_index=args.curve_index,
    )
    report["source_authorized_features"] = list(source_features)
    report["source_feature_proofs"] = list(source_proofs)
    report["dmo"] = _binding(dmo_path, root)
    report["source_manifest"] = _binding(source_manifest, root)
    report["runtime_executable"] = _binding(runtime, root)
    report["fruit_visual_diff"] = _binding(fruit_visual_diff, root)
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
    parser.add_argument("--dmo", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--fruit-visual-diff", required=True, type=Path)
    parser.add_argument("--original-root", required=True, type=Path)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--level", default="Jungle2")
    parser.add_argument("--curve-index", default=0, type=int)
    parser.add_argument("--hard", action="store_true")
    parser.add_argument("--start-update", required=True, type=int)
    parser.add_argument("--end-update", required=True, type=int)
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
                "compared_tick_count": report["compared_tick_count"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
