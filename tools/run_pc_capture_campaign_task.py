"""Prepare or collect one verified PC capture-campaign task on Windows.

The ``prepare`` command is read-mostly with respect to the game: it verifies
the campaign, snapshots host state, and creates a new immutable session plan.
The ``collect`` command launches two visible original-game replays and must be
run only when the user has yielded the desktop and DXGI output.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tools import collect_pc_golden_v4
from zuma_rl.pc_capture_campaign import verify_pc_capture_campaign


def _ready_task(
    report: Any,
    task_id: str,
) -> Mapping[str, Any]:
    matches = [row for row in report.tasks if row.get("id") == task_id]
    if len(matches) != 1:
        raise ValueError("campaign task id was not found exactly once")
    task = matches[0]
    if task.get("status") != "ready_for_collection":
        raise ValueError("campaign task is not ready for collection")
    options = task.get("collector_prepare_options")
    if not isinstance(options, Mapping):
        raise ValueError("campaign task has no collector options")
    return task


def _root_member(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("campaign collector path is invalid")
    candidate = root.joinpath(*relative.split("/"))
    resolved = candidate.resolve(strict=True)
    resolved.relative_to(root)
    return resolved


def _prepare_arguments(args: argparse.Namespace) -> list[str]:
    report = verify_pc_capture_campaign(
        args.campaign,
        evidence_root=args.evidence_root,
    )
    if report.status != "PASS":
        raise ValueError(
            "campaign verification failed: " + "; ".join(report.reasons)
        )
    task = _ready_task(report, args.task)
    options = task["collector_prepare_options"]

    evidence_root = args.evidence_root.resolve(strict=True)
    original_root = args.original_root.resolve(strict=True)
    runtime_source = original_root / "ZumasRevenge.exe"
    if not runtime_source.is_file():
        raise ValueError("original_root has no ZumasRevenge.exe")
    session_root = args.session_root.resolve(strict=False)
    if session_root.exists():
        raise ValueError("session_root already exists")
    if not session_root.parent.is_dir():
        raise ValueError("session_root parent does not exist")

    nonce = uuid.uuid4().hex if args.session_nonce is None else args.session_nonce
    if (
        len(nonce) != 32
        or any(character not in "0123456789abcdef" for character in nonce)
    ):
        raise ValueError("session_nonce must be 32 lowercase hex characters")

    result = [
        "prepare",
        "--session-root",
        str(session_root),
        "--session-nonce",
        nonce,
        "--dmo",
        str(_root_member(evidence_root, options["dmo"])),
        "--prestate-dir",
        str(_root_member(evidence_root, options["prestate_dir"])),
        "--runtime-source-executable",
        str(runtime_source),
        "--direct-runtime-executable",
        str(
            _root_member(
                evidence_root,
                options["direct_runtime_executable"],
            )
        ),
        "--changedir",
        str(original_root),
        "--crt-rand-seed",
        str(options["crt_rand_seed"]),
        "--board-seed",
        str(options["board_seed"]),
        "--global-rng-seed",
        str(options["global_rng_seed"]),
        "--thread-crt-rng-seed",
        str(options["thread_crt_rng_seed"]),
        "--duration-seconds",
        str(options["duration_seconds"]),
        "--frame-budget-fps",
        str(options["frame_budget_fps"]),
        "--wait-until-framework-update",
        str(options["wait_until_framework_update"]),
        "--window-repaint-update",
        str(options["window_repaint_update"]),
        "--attach-at-update",
        str(options["attach_at_update"]),
        "--detach-at-update",
        str(options["detach_at_update"]),
        "--reattach-at-update",
        str(options["reattach_at_update"]),
        "--maximum-hits",
        str(args.maximum_hits),
        "--trace-timeout-seconds",
        str(args.trace_timeout_seconds),
        "--device-index",
        str(args.device_index),
        "--output-index",
        str(args.output_index),
    ]
    if options.get("allow_pre_stream_commands") is True:
        result.append("--allow-pre-stream-commands")
        if options.get("attach_at_update") == 0:
            result.append("--startup-trace-handoff")
    elif options.get("attach_at_update", 0) > 0:
        result.extend(
            [
                "--allow-attach-stabilization",
                "--allow-pre-attach-file-write-debt",
                "--allow-font-cache-manifest-completion-debt",
            ]
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one fail-closed PC capture-campaign task."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("campaign", type=Path)
    prepare.add_argument("--task", required=True)
    prepare.add_argument("--evidence-root", required=True, type=Path)
    prepare.add_argument("--original-root", required=True, type=Path)
    prepare.add_argument("--session-root", required=True, type=Path)
    prepare.add_argument("--session-nonce")
    prepare.add_argument("--maximum-hits", type=int, default=10_000)
    prepare.add_argument("--trace-timeout-seconds", type=float, default=900.0)
    prepare.add_argument("--device-index", type=int, default=0)
    prepare.add_argument("--output-index", type=int, default=0)

    collect = subparsers.add_parser("collect")
    collect.add_argument("--session-root", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "collect":
        return collect_pc_golden_v4.main(
            ["collect", "--session-root", str(args.session_root)]
        )
    try:
        collector_arguments = _prepare_arguments(args)
    except (OSError, ValueError) as error:
        print(
            json.dumps(
                {
                    "status": "FAIL",
                    "reason": str(error),
                },
                sort_keys=True,
            )
        )
        return 1
    return collect_pc_golden_v4.main(collector_arguments)


if __name__ == "__main__":
    raise SystemExit(main())
