"""Collect one preregistered formal PC fruit-lifecycle source.

The source watches the first fruit lifecycle after a fixed monitor boundary,
freezes at the first retained update whose natural lifetime is below a frozen
threshold, and records a fixed-size exact-step full-Board suffix.  Monitoring
is read-only, every observed update is retained, and a lifecycle may not be
skipped after its outcome is known.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tools.collect_pc_memory_probe import collect_probe
from tools.pc_state_transaction import (
    capture_state,
    write_snapshot_exclusive,
)


class FormalFruitLifecycleCollectionError(RuntimeError):
    """The bounded formal source transaction could not be completed."""


def collect_formal_fruit_lifecycle(
    *,
    plan_path: Path,
    prestate_path: Path,
    output_root: Path,
    host_pre_output: Path,
    host_post_output: Path,
    session_nonce: str,
    monitor_start_update: int,
    maximum_framework_update: int,
    remaining_threshold_ticks: int,
    slowdown_lead_updates: int,
    freeze_lead_updates: int,
    trajectory_tick_count: int,
    poll_interval_seconds: float,
) -> Path:
    """Run one no-retry source and restore the exact caller state."""

    paths = (
        plan_path,
        prestate_path,
    )
    if any(not path.resolve().is_file() for path in paths):
        raise FormalFruitLifecycleCollectionError("fixed_input_missing")
    outputs = (output_root, host_pre_output, host_post_output)
    if any(path.exists() for path in outputs):
        raise FormalFruitLifecycleCollectionError("output_already_exists")
    if (
        not output_root.parent.is_dir()
        or not host_pre_output.parent.is_dir()
        or not host_post_output.parent.is_dir()
        or not session_nonce
    ):
        raise FormalFruitLifecycleCollectionError("output_contract_invalid")

    host_pre = capture_state(
        session_nonce=session_nonce,
        phase="host-pre",
    )
    write_snapshot_exclusive(host_pre, host_pre_output)
    source_path: Path | None = None
    collection_error: BaseException | None = None
    try:
        source_path = collect_probe(
            plan_path=plan_path.resolve(),
            prestate_path=prestate_path.resolve(),
            host_restore_path=host_pre_output.resolve(),
            output_root=output_root.resolve(),
            probe_update=monitor_start_update,
            slowdown_update=monitor_start_update - 20,
            int32_value=None,
            maximum_attempts=1,
            trajectory_end_update=(
                monitor_start_update + trajectory_tick_count - 1
            ),
            trajectory_mode="full",
            formal_full_state_evidence=True,
            allow_discovered_display_mismatch=True,
            formal_independent_score_binding=True,
            formal_fruit_lifecycle_trigger={
                "monitor_start_update": monitor_start_update,
                "maximum_framework_update": maximum_framework_update,
                "remaining_threshold_ticks": remaining_threshold_ticks,
                "slowdown_lead_updates": slowdown_lead_updates,
                "freeze_lead_updates": freeze_lead_updates,
                "trajectory_tick_count": trajectory_tick_count,
                "poll_interval_seconds": poll_interval_seconds,
            },
        )
    except BaseException as error:
        collection_error = error
    host_post = capture_state(
        session_nonce=session_nonce,
        phase="host-post",
    )
    write_snapshot_exclusive(host_post, host_post_output)
    if host_post.state_root != host_pre.state_root:
        raise FormalFruitLifecycleCollectionError(
            "host_state_restore_mismatch"
        ) from collection_error
    if collection_error is not None:
        raise FormalFruitLifecycleCollectionError(
            f"source_collection_failed:{type(collection_error).__name__}:"
            f"{str(collection_error)}"
        ) from collection_error
    if source_path is None:
        raise FormalFruitLifecycleCollectionError(
            "source_collection_result_missing"
        )
    return source_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--prestate", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--host-pre-output", required=True, type=Path)
    parser.add_argument("--host-post-output", required=True, type=Path)
    parser.add_argument("--session-nonce", required=True)
    parser.add_argument("--monitor-start-update", type=int, default=9527)
    parser.add_argument(
        "--maximum-framework-update",
        type=int,
        default=17200,
    )
    parser.add_argument(
        "--remaining-threshold-ticks",
        type=int,
        default=384,
    )
    parser.add_argument(
        "--slowdown-lead-updates",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--freeze-lead-updates",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--trajectory-tick-count",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=0.002,
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        source = collect_formal_fruit_lifecycle(
            plan_path=args.plan,
            prestate_path=args.prestate,
            output_root=args.output_root,
            host_pre_output=args.host_pre_output,
            host_post_output=args.host_post_output,
            session_nonce=args.session_nonce,
            monitor_start_update=args.monitor_start_update,
            maximum_framework_update=args.maximum_framework_update,
            remaining_threshold_ticks=args.remaining_threshold_ticks,
            slowdown_lead_updates=args.slowdown_lead_updates,
            freeze_lead_updates=args.freeze_lead_updates,
            trajectory_tick_count=args.trajectory_tick_count,
            poll_interval_seconds=args.poll_interval_seconds,
        )
    except FormalFruitLifecycleCollectionError as error:
        print(f"formal fruit lifecycle collection error: {error}", file=sys.stderr)
        return 1
    print(source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
