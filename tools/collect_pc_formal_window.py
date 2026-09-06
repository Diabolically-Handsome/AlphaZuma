"""Collect one fixed, preregistered formal PC full-state window.

The wrapper captures a fresh host-state snapshot, runs exactly one formal
read-only retail source attempt, captures the restored host state, and fails
closed unless both state roots are identical.  Feature presence is deliberately
not an acceptance criterion; it is derived only after the source is complete.
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
from tools.pc_state_transaction import capture_state, write_snapshot_exclusive


class FormalWindowCollectionError(RuntimeError):
    """The fixed formal-source transaction could not be completed."""


def collect_formal_window(
    *,
    plan_path: Path,
    prestate_path: Path,
    output_root: Path,
    host_pre_output: Path,
    host_post_output: Path,
    session_nonce: str,
    freeze_update: int,
    slowdown_update: int,
    trajectory_end_update: int,
) -> Path:
    """Run one no-retry source and restore the exact caller state."""

    if any(
        not path.resolve().is_file()
        for path in (plan_path, prestate_path)
    ):
        raise FormalWindowCollectionError("fixed_input_missing")
    if any(
        path.exists()
        for path in (output_root, host_pre_output, host_post_output)
    ):
        raise FormalWindowCollectionError("output_already_exists")
    if (
        not output_root.parent.is_dir()
        or not host_pre_output.parent.is_dir()
        or not host_post_output.parent.is_dir()
        or not session_nonce
        or slowdown_update < 0
        or freeze_update <= slowdown_update
        or trajectory_end_update < freeze_update
    ):
        raise FormalWindowCollectionError("output_contract_invalid")

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
            probe_update=freeze_update,
            slowdown_update=slowdown_update,
            int32_value=None,
            maximum_attempts=1,
            trajectory_end_update=trajectory_end_update,
            trajectory_mode="full",
            formal_full_state_evidence=True,
            allow_discovered_display_mismatch=True,
            formal_independent_score_binding=True,
        )
    except BaseException as error:
        collection_error = error

    host_post = capture_state(
        session_nonce=session_nonce,
        phase="host-post",
    )
    write_snapshot_exclusive(host_post, host_post_output)
    if host_post.state_root != host_pre.state_root:
        raise FormalWindowCollectionError(
            "host_state_restore_mismatch"
        ) from collection_error
    if collection_error is not None:
        raise FormalWindowCollectionError(
            f"source_collection_failed:{type(collection_error).__name__}:"
            f"{str(collection_error)}"
        ) from collection_error
    if source_path is None:
        raise FormalWindowCollectionError("source_collection_result_missing")
    return source_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--prestate", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--host-pre-output", required=True, type=Path)
    parser.add_argument("--host-post-output", required=True, type=Path)
    parser.add_argument("--session-nonce", required=True)
    parser.add_argument("--freeze-update", required=True, type=int)
    parser.add_argument("--slowdown-update", required=True, type=int)
    parser.add_argument(
        "--trajectory-end-update",
        required=True,
        type=int,
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        source = collect_formal_window(
            plan_path=args.plan,
            prestate_path=args.prestate,
            output_root=args.output_root,
            host_pre_output=args.host_pre_output,
            host_post_output=args.host_post_output,
            session_nonce=args.session_nonce,
            freeze_update=args.freeze_update,
            slowdown_update=args.slowdown_update,
            trajectory_end_update=args.trajectory_end_update,
        )
    except FormalWindowCollectionError as error:
        print(f"formal window collection error: {error}", file=sys.stderr)
        return 1
    print(source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
