"""Replay DMO input from a frozen PC midstate and diff every captured tick."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from zuma_rl.pc_gameplay_diff import compare_gameplay_trajectory
from zuma_rl.pc_ambient_rng_boundary import (
    AmbientRngBoundaryError,
    bind_ambient_rng_boundary_to_diff,
    load_ambient_rng_boundary_report,
    sha256_path,
)
from zuma_rl.pc_memory_trajectory import (
    PcMemoryTrajectoryError,
    load_legacy_memory_trajectory_v1,
    load_memory_trajectory,
)
from zuma_rl.pc_mechanism_audit import (
    PcMechanismAuditError,
    derive_native_mechanism_features,
)
from zuma_rl.popcap_dmo import PopCapDemo, PopCapDemoError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=Path)
    parser.add_argument("--dmo", required=True, type=Path)
    parser.add_argument("--original-root", required=True, type=Path)
    parser.add_argument("--level-id")
    parser.add_argument("--hard", action="store_true")
    parser.add_argument("--curve-index", type=int)
    parser.add_argument("--start-update", type=int)
    parser.add_argument("--end-update", type=int)
    parser.add_argument("--no-synchronize-shooter", action="store_true")
    parser.add_argument(
        "--reconcile-external-visual-mtrand",
        action="store_true",
        help=(
            "Conditionally consume short source-observed MTRand suffixes. "
            "The output remains non-certifying until an independent bound "
            "call-trace proof classifies every reconciled caller."
        ),
    )
    parser.add_argument(
        "--maximum-external-visual-mtrand-draws-per-tick",
        type=int,
        default=64,
    )
    parser.add_argument("--scope-boundary-proof", type=Path)
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
        "--source-proof-start-update",
        type=int,
        help=(
            "Optional start of a wider native observation window used only "
            "to derive source feature proofs."
        ),
    )
    parser.add_argument(
        "--source-proof-end-update",
        type=int,
        help=(
            "Optional end of a wider native observation window used only "
            "to derive source feature proofs."
        ),
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


def _select_frames(
    frames: tuple[object, ...],
    *,
    start_update: int | None,
    end_update: int | None,
) -> tuple[object, ...]:
    if not frames:
        raise PcMemoryTrajectoryError("gameplay_diff_frames_empty")
    start = (
        int(getattr(frames[0], "update"))
        if start_update is None
        else start_update
    )
    end = (
        int(getattr(frames[-1], "update"))
        if end_update is None
        else end_update
    )
    if start < 0 or end < start:
        raise PcMemoryTrajectoryError("gameplay_diff_window_invalid")
    selected = tuple(
        frame
        for frame in frames
        if start <= int(getattr(frame, "update")) <= end
    )
    if (
        len(selected) < 2
        or int(getattr(selected[0], "update")) != start
        or int(getattr(selected[-1], "update")) != end
        or len(selected) != end - start + 1
    ):
        raise PcMemoryTrajectoryError("gameplay_diff_window_incomplete")
    return selected


def main(argv: Iterable[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        index = arguments.index.resolve()
        dmo_path = arguments.dmo.resolve()
        if arguments.source_feature_proofs and arguments.evidence_root is None:
            raise PcMemoryTrajectoryError(
                "gameplay_diff_source_proof_root_missing"
            )
        if (
            arguments.source_proof_start_update is not None
            or arguments.source_proof_end_update is not None
        ) and not arguments.source_feature_proofs:
            raise PcMemoryTrajectoryError(
                "gameplay_diff_source_proof_window_without_proofs"
            )
        loader = (
            load_legacy_memory_trajectory_v1
            if arguments.legacy_v1
            else load_memory_trajectory
        )
        all_frames = loader(index)
        frames = _select_frames(
            all_frames,
            start_update=arguments.start_update,
            end_update=arguments.end_update,
        )
        report = dict(compare_gameplay_trajectory(
            frames,
            demo=PopCapDemo.read(dmo_path),
            original_root=arguments.original_root.resolve(),
            hard=arguments.hard,
            level_id=arguments.level_id,
            curve_index=arguments.curve_index,
            synchronize_shooter=(
                not arguments.no_synchronize_shooter
            ),
            reconcile_external_visual_mtrand=(
                arguments.reconcile_external_visual_mtrand
            ),
            maximum_external_visual_mtrand_draws_per_tick=(
                arguments.maximum_external_visual_mtrand_draws_per_tick
            ),
            waypoint_tolerance=arguments.waypoint_tolerance,
            position_tolerance=arguments.position_tolerance,
            progress_tolerance=arguments.progress_tolerance,
        ))
        trajectory_sha256 = sha256_path(index)
        dmo_sha256 = sha256_path(dmo_path)
        report["trajectory"] = {
            "artifact": str(index),
            "artifact_sha256": trajectory_sha256,
            "captured_start_update": all_frames[0].update,
            "captured_end_update": all_frames[-1].update,
            "selected_start_update": frames[0].update,
            "selected_end_update": frames[-1].update,
        }
        if arguments.source_feature_proofs:
            evidence_root = arguments.evidence_root.resolve(strict=True)
            try:
                relative_index = index.relative_to(evidence_root)
            except ValueError as error:
                raise PcMemoryTrajectoryError(
                    "gameplay_diff_source_proof_path_invalid"
                ) from error
            proof_frames = _select_frames(
                all_frames,
                start_update=(
                    frames[0].update
                    if arguments.source_proof_start_update is None
                    else arguments.source_proof_start_update
                ),
                end_update=(
                    frames[-1].update
                    if arguments.source_proof_end_update is None
                    else arguments.source_proof_end_update
                ),
            )
            if (
                proof_frames[0].update > frames[0].update
                or proof_frames[-1].update < frames[-1].update
            ):
                raise PcMemoryTrajectoryError(
                    "gameplay_diff_source_proof_window_not_enclosing"
                )
            features, proofs = derive_native_mechanism_features(
                proof_frames,
                original_root=arguments.original_root.resolve(),
                level_id=arguments.level_id,
                hard=arguments.hard,
                curve_index=(
                    0 if arguments.curve_index is None else arguments.curve_index
                ),
            )
            report["trajectory"]["artifact"] = relative_index.as_posix()
            report["trajectory"]["source_version"] = (
                1 if arguments.legacy_v1 else 2
            )
            report["trajectory"]["source_proof_start_update"] = (
                proof_frames[0].update
            )
            report["trajectory"]["source_proof_end_update"] = (
                proof_frames[-1].update
            )
            report["source_authorized_features"] = list(features)
            report["source_feature_proofs"] = list(proofs)
        report["dmo"] = {
            "artifact": str(dmo_path),
            "artifact_sha256": dmo_sha256,
        }
        if arguments.scope_boundary_proof is not None:
            boundary_path = arguments.scope_boundary_proof.resolve()
            boundary = load_ambient_rng_boundary_report(boundary_path)
            report["scope_boundary"] = bind_ambient_rng_boundary_to_diff(
                boundary,
                boundary_path=boundary_path,
                trajectory_sha256=trajectory_sha256,
                dmo_sha256=dmo_sha256,
                selected_end_update=frames[-1].update,
            )
        payload = _canonical(report)
        if arguments.output is not None:
            output = arguments.output.resolve()
            if output.exists() or not output.parent.is_dir():
                raise PcMemoryTrajectoryError(
                    "gameplay_diff_output_invalid"
                )
            output.write_bytes(payload)
        sys.stdout.buffer.write(payload)
        return 0 if report["status"] == "PASS" else 1
    except (
        OSError,
        AmbientRngBoundaryError,
        PcMechanismAuditError,
        PcMemoryTrajectoryError,
        PopCapDemoError,
    ) as error:
        print(f"gameplay diff error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
