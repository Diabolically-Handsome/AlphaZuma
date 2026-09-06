"""Run one reversible retail F11 screenshot probe at a frozen DMO update."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tools.collect_pc_memory_probe import collect_probe
from tools.collect_pc_golden_v4 import (
    _activate_startup_window,
    _wait_for_capture_window,
)
from tools.control_popcap_replay import main_window_for_pid
from tools.popcap_internal_screenshot import (
    DEFAULT_SCREENSHOT_ROOT,
    capture_internal_screenshot,
    send_virtual_key,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--prestate", required=True, type=Path)
    parser.add_argument("--host-restore", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--slowdown-update", required=True, type=int)
    parser.add_argument("--probe-update", required=True, type=int)
    parser.add_argument("--maximum-attempts", default=1, type=int)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    plan = json.loads(args.plan.read_text(encoding="ascii"))
    changedir = Path(plan["runtime"]["changedir"])
    runtime_parent = Path(plan["runtime"]["runtime_executable"]).parent

    def capture(
        pid: int,
        unused_board: Mapping[str, Any],
        attempt_root: Path,
    ) -> Mapping[str, Any]:
        del unused_board
        target = _wait_for_capture_window(pid)
        activation = _activate_startup_window(target)
        hwnd = main_window_for_pid(pid)
        screenshot = capture_internal_screenshot(
            window_handle=hwnd,
            output=attempt_root / "internal-f11.png",
            screenshot_root=DEFAULT_SCREENSHOT_ROOT,
            additional_screenshot_roots=(
                changedir / "_screenshots",
                runtime_parent / "_screenshots",
                Path.cwd() / "_screenshots",
            ),
            post_key=send_virtual_key,
        )
        return {
            "schema": "zuma-rl.pc-internal-screenshot-probe",
            "version": 1,
            "activation": activation,
            "screenshot": screenshot,
        }

    path = collect_probe(
        plan_path=args.plan.resolve(),
        prestate_path=args.prestate.resolve(),
        host_restore_path=args.host_restore.resolve(),
        output_root=args.output_root.resolve(),
        probe_update=args.probe_update,
        slowdown_update=args.slowdown_update,
        int32_value=None,
        maximum_attempts=args.maximum_attempts,
        trajectory_end_update=args.probe_update,
        trajectory_mode="rng",
        skip_repaint_guard=True,
        skip_frozen_snapshot=True,
        allow_discovered_display_mismatch=True,
        diagnostic_observer=capture,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
