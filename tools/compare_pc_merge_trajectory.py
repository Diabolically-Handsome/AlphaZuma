"""Transplant a retail staged merge and compare simulator state per tick."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from zuma_rl.pc_memory_trajectory import (
    PcMemoryTrajectoryError,
    load_legacy_memory_trajectory_v1,
    load_memory_trajectory,
)
from zuma_rl.pc_mechanism_audit import (
    PcMechanismAuditError,
    derive_native_mechanism_features,
)
from zuma_rl.pc_merge_diff import compare_staged_merge_trajectory


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=Path)
    parser.add_argument("--original-root", required=True, type=Path)
    parser.add_argument("--level-id")
    parser.add_argument("--hard", action="store_true")
    parser.add_argument("--curve-index", type=int)
    parser.add_argument("--start-update", type=int)
    parser.add_argument("--end-update", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--legacy-v1",
        action="store_true",
        help="Explicitly decode an immutable legacy trajectory-v1 source.",
    )
    parser.add_argument(
        "--source-feature-proofs",
        action="store_true",
        help="Recompute conservative native mechanism proofs for the window.",
    )
    parser.add_argument(
        "--evidence-root",
        type=Path,
        help="Root used to bind source-proof trajectory paths relatively.",
    )
    parser.add_argument("--waypoint-tolerance", type=float, default=1e-4)
    parser.add_argument("--position-tolerance", type=float, default=1e-3)
    parser.add_argument("--progress-tolerance", type=float, default=1e-6)
    return parser


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _source_provenance(
    index: Path,
    *,
    trajectory_sha256: str,
) -> Mapping[str, Any] | None:
    """Bind the derived report to the collector's adjacent source probe."""

    probe_path = index.parent.parent / "memory-probe.json"
    if not probe_path.is_file():
        return None
    try:
        probe = json.loads(probe_path.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PcMemoryTrajectoryError(
            "merge_diff_source_probe_invalid"
        ) from error
    if not isinstance(probe, Mapping):
        raise PcMemoryTrajectoryError("merge_diff_source_probe_invalid")
    trajectory = probe.get("trajectory")
    if (
        not isinstance(trajectory, Mapping)
        or trajectory.get("artifact_sha256") != trajectory_sha256
    ):
        raise PcMemoryTrajectoryError(
            "merge_diff_source_probe_trajectory_mismatch"
        )
    return {
        "diagnostic_mutation": probe.get("diagnostic_mutation"),
        "dmo_sha256": probe.get("dmo_sha256"),
        "probe_sha256": _sha256_path(probe_path),
        "runtime_executable_sha256": probe.get(
            "runtime_executable_sha256"
        ),
        "sample_phase": trajectory.get("sample_phase"),
    }


def main(argv: Iterable[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        index = arguments.index.resolve()
        trajectory_sha256 = _sha256_path(index)
        if arguments.source_feature_proofs and arguments.evidence_root is None:
            raise PcMemoryTrajectoryError(
                "merge_diff_source_proof_root_missing"
            )
        loader = (
            load_legacy_memory_trajectory_v1
            if arguments.legacy_v1
            else load_memory_trajectory
        )
        frames = loader(
            index,
            start_update=arguments.start_update,
            end_update=arguments.end_update,
        )
        report = dict(
            compare_staged_merge_trajectory(
                frames,
                original_root=arguments.original_root.resolve(),
                hard=arguments.hard,
                level_id=arguments.level_id,
                curve_index=arguments.curve_index,
                waypoint_tolerance=arguments.waypoint_tolerance,
                position_tolerance=arguments.position_tolerance,
                progress_tolerance=arguments.progress_tolerance,
            )
        )
        report["source_trajectory_sha256"] = trajectory_sha256
        if arguments.source_feature_proofs:
            evidence_root = arguments.evidence_root.resolve(strict=True)
            try:
                relative_index = index.relative_to(evidence_root)
            except ValueError as error:
                raise PcMemoryTrajectoryError(
                    "merge_diff_source_proof_path_invalid"
                ) from error
            features, proofs = derive_native_mechanism_features(
                frames,
                original_root=arguments.original_root.resolve(),
                level_id=arguments.level_id,
                hard=arguments.hard,
                curve_index=(
                    0 if arguments.curve_index is None else arguments.curve_index
                ),
                allow_initial_staging_observation=True,
            )
            report["trajectory"] = {
                "artifact": relative_index.as_posix(),
                "artifact_sha256": trajectory_sha256,
                "source_version": 1 if arguments.legacy_v1 else 2,
                "selected_start_update": frames[0].update,
                "selected_end_update": frames[-1].update,
            }
            report["source_authorized_features"] = list(features)
            report["source_feature_proofs"] = list(proofs)
        provenance = _source_provenance(
            index,
            trajectory_sha256=trajectory_sha256,
        )
        if provenance is not None:
            report["source_provenance"] = provenance
        payload = _canonical(report)
        if arguments.output is not None:
            output = arguments.output.resolve()
            if output.exists() or not output.parent.is_dir():
                raise PcMemoryTrajectoryError(
                    "merge_diff_output_invalid"
                )
            output.write_bytes(payload)
        sys.stdout.buffer.write(payload)
        return 0 if report["status"] == "PASS" else 1
    except (PcMechanismAuditError, PcMemoryTrajectoryError) as error:
        print(f"merge diff error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
