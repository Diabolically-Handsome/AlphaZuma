"""Run one command under an exact temporary Windows display mode."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from zuma_rl.windows_display_mode import (
    DisplayMode,
    DisplayModeError,
    apply_temporary_display_mode,
    available_display_modes,
    current_display_mode,
    select_exact_display_mode,
    test_display_mode,
)


REPORT_SCHEMA = "zuma-rl.windows-display-mode-transaction"
REPORT_VERSION = 1


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")


def _write_exclusive(path: Path, value: Any) -> None:
    path = path.resolve()
    if not path.parent.is_dir() or path.exists():
        raise DisplayModeError("display_mode_report_path_invalid")
    payload = _canonical_json(value)
    try:
        with path.open("xb") as stream:
            if stream.write(payload) != len(payload):
                raise DisplayModeError("display_mode_report_write_incomplete")
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as error:
        raise DisplayModeError("display_mode_report_write_failed") from error


def _select(args: argparse.Namespace) -> tuple[DisplayMode, tuple[DisplayMode, ...]]:
    current = current_display_mode()
    modes = available_display_modes()
    selected = select_exact_display_mode(
        modes,
        width=args.width,
        height=args.height,
        refresh_rate_hz=args.refresh_rate,
        current=current,
    )
    return selected, modes


def _restore_with_retry(mode: DisplayMode) -> DisplayMode:
    last_error: DisplayModeError | None = None
    for _ in range(3):
        try:
            observed = apply_temporary_display_mode(mode)
            if observed != mode:
                raise DisplayModeError(
                    "display_mode_restore_verification_failed"
                )
            return observed
        except DisplayModeError as error:
            last_error = error
            time.sleep(0.25)
    raise DisplayModeError("display_mode_restore_failed") from last_error


def _list_modes(args: argparse.Namespace) -> int:
    current = current_display_mode()
    modes = available_display_modes()
    matching = [
        mode.to_dict()
        for mode in modes
        if (args.width is None or mode.width == args.width)
        and (args.height is None or mode.height == args.height)
        and (
            args.refresh_rate is None
            or mode.refresh_rate_hz == args.refresh_rate
        )
    ]
    print(
        json.dumps(
            {
                "schema": "zuma-rl.windows-display-mode-list",
                "version": 1,
                "current": current.to_dict(),
                "matching_modes": matching,
                "matching_mode_count": len(matching),
                "total_unique_mode_count": len(modes),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _run(args: argparse.Namespace) -> int:
    command = list(args.command)
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        raise DisplayModeError("display_mode_child_command_missing")
    before = current_display_mode()
    target, modes = _select(args)
    test_display_mode(target)
    switched_at_ns: int | None = None
    applied: DisplayMode | None = None
    child_returncode: int | None = None
    child_error: str | None = None
    restored: DisplayMode | None = None
    restore_error: str | None = None
    started_ns = time.perf_counter_ns()
    try:
        applied = apply_temporary_display_mode(target)
        switched_at_ns = time.perf_counter_ns()
        time.sleep(args.settle_seconds)
        if current_display_mode() != target:
            raise DisplayModeError("display_mode_drift_before_child")
        try:
            completed = subprocess.run(command, check=False)
            child_returncode = int(completed.returncode)
        except OSError as error:
            child_error = f"{type(error).__name__}:{error}"
    finally:
        try:
            restored = _restore_with_retry(before)
        except DisplayModeError as error:
            restore_error = str(error)
    finished_ns = time.perf_counter_ns()
    report = {
        "schema": REPORT_SCHEMA,
        "version": REPORT_VERSION,
        "status": (
            "complete"
            if restored == before
            and child_error is None
            and child_returncode == 0
            else "failed"
        ),
        "before": before.to_dict(),
        "requested": {
            "width": args.width,
            "height": args.height,
            "refresh_rate_hz": args.refresh_rate,
        },
        "selected": target.to_dict(),
        "selected_from_unique_mode_count": len(modes),
        "applied": None if applied is None else applied.to_dict(),
        "restored": None if restored is None else restored.to_dict(),
        "restore_exact": restored == before,
        "restore_error": restore_error,
        "child_command": command,
        "child_returncode": child_returncode,
        "child_error": child_error,
        "settle_seconds": args.settle_seconds,
        "started_perf_counter_ns": started_ns,
        "switched_perf_counter_ns": switched_at_ns,
        "finished_perf_counter_ns": finished_ns,
    }
    _write_exclusive(args.report, report)
    if restore_error is not None:
        print(restore_error, file=sys.stderr)
        return 2
    if child_error is not None:
        print(child_error, file=sys.stderr)
        return 3
    assert child_returncode is not None
    return child_returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    list_parser = subparsers.add_parser("list")
    list_parser.add_argument("--width", type=int)
    list_parser.add_argument("--height", type=int)
    list_parser.add_argument("--refresh-rate", type=int)
    list_parser.set_defaults(handler=_list_modes)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--width", required=True, type=int)
    run_parser.add_argument("--height", required=True, type=int)
    run_parser.add_argument("--refresh-rate", required=True, type=int)
    run_parser.add_argument("--settle-seconds", default=1.0, type=float)
    run_parser.add_argument("--report", required=True, type=Path)
    run_parser.add_argument("command", nargs=argparse.REMAINDER)
    run_parser.set_defaults(handler=_run)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if (
        getattr(args, "settle_seconds", 0.0) < 0
        or getattr(args, "settle_seconds", 0.0) > 10.0
    ):
        print("error: settle seconds must be in [0, 10]", file=sys.stderr)
        return 2
    try:
        return int(args.handler(args))
    except (DisplayModeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
