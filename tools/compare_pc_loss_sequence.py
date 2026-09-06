"""Diff the simulator skull-entry loss path against every captured PC tick."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Iterable

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from zuma_rl.pc_loss_sequence import (
    compare_loss_sequence_with_simulator,
    validate_loss_simulator_diff_report,
    verify_loss_sequence_frames,
)
from zuma_rl.pc_memory_trajectory import load_memory_trajectory


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--level", default="Jungle2")
    parser.add_argument("--hard", action="store_true")
    parser.add_argument("--curve-index", type=int, default=0)
    parser.add_argument("--start-update", type=int)
    parser.add_argument("--end-update", type=int)
    parser.add_argument("--trigger-ball-id", required=True, type=int)
    parser.add_argument("--decoded-curve-end", required=True, type=int)
    parser.add_argument("--expected-trigger-update", type=int)
    parser.add_argument("--expected-chain-empty-update", type=int)
    parser.add_argument("--expected-pending-count", type=int, default=1)
    parser.add_argument(
        "--expected-pretrigger-advance",
        type=float,
        default=0.125,
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    trajectory = args.trajectory.resolve()
    evidence_root = args.evidence_root.resolve(strict=True)
    output = args.output.resolve()
    if output.exists() or not output.parent.is_dir():
        raise FileExistsError(
            "output path must be a new file in an existing directory"
        )
    captured_frames = load_memory_trajectory(trajectory)
    if (args.start_update is None) != (args.end_update is None):
        raise ValueError(
            "selected start and end updates must be provided together"
        )
    if args.start_update is None:
        frames = captured_frames
    else:
        assert args.end_update is not None
        frames = tuple(
            frame
            for frame in captured_frames
            if args.start_update <= frame.update <= args.end_update
        )
        if (
            not frames
            or frames[0].update != args.start_update
            or frames[-1].update != args.end_update
            or len(frames) != args.end_update - args.start_update + 1
        ):
            raise ValueError(
                "selected loss window is not contiguous in the trajectory"
            )
    oracle = verify_loss_sequence_frames(
        frames,
        curve_index=args.curve_index,
        trigger_ball_id=args.trigger_ball_id,
        decoded_curve_end=args.decoded_curve_end,
        expected_trigger_update=args.expected_trigger_update,
        expected_chain_empty_update=args.expected_chain_empty_update,
        expected_pending_count=args.expected_pending_count,
        expected_pretrigger_advance=args.expected_pretrigger_advance,
    )
    report = compare_loss_sequence_with_simulator(
        frames,
        original_root=args.root.resolve(),
        hard=args.hard,
        level_id=args.level,
        curve_index=args.curve_index,
    )
    try:
        relative_trajectory = trajectory.relative_to(evidence_root)
    except ValueError as error:
        raise ValueError(
            "trajectory must be contained under the evidence root"
        ) from error
    report["trajectory"] = {
        "artifact": relative_trajectory.as_posix(),
        "artifact_sha256": _sha256_path(trajectory),
        "captured_start_update": captured_frames[0].update,
        "captured_end_update": captured_frames[-1].update,
        "selected_start_update": frames[0].update,
        "selected_end_update": frames[-1].update,
    }
    report["oracle"] = {
        "schema": oracle["schema"],
        "status": oracle["status"],
        "trigger_update": oracle["trigger"]["update"],
        "empty_update": oracle["suction"]["first_chain_empty_update"],
        "ticks_from_trigger_to_empty": (
            oracle["suction"]["ticks_from_trigger_to_empty"]
        ),
    }
    validation_reason = validate_loss_simulator_diff_report(
        report,
        evidence_root=evidence_root,
        original_root=args.root.resolve(strict=True),
    )
    if validation_reason is not None:
        raise ValueError(validation_reason)
    output.write_bytes(
        (
            json.dumps(
                report,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("ascii")
    )
    print(
        f"status={report['status']} "
        f"updates={report['start_update']}..{report['end_update']} "
        f"ticks={report['ticks_compared']} "
        f"waypoint_error={report['maximum_waypoint_error']}"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
