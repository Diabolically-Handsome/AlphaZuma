"""Log read-only changes in a live retail Board for event localization."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tools.autoplay_live_zuma import (
    LiveBoardUnavailable,
    read_framework_state,
    read_live_board,
)


def monitor(
    pid: int,
    *,
    output: Path,
    interval_seconds: float,
    maximum_seconds: float,
) -> dict[str, Any]:
    if os.name != "nt":
        raise RuntimeError("live board monitoring requires Windows")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    if interval_seconds <= 0 or maximum_seconds <= 0:
        raise ValueError("monitor timing is invalid")
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    previous: tuple[Any, ...] | None = None
    change_count = 0
    sample_count = 0
    unavailable_count = 0
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(
            json.dumps(
                {
                    "schema": "zuma-rl.live-board-change-monitor",
                    "version": 1,
                    "classification": "read-only-localization-diagnostic",
                    "process_id": pid,
                    "process_memory_writes": 0,
                    "started_perf_counter_ns": time.perf_counter_ns(),
                },
                sort_keys=True,
            )
            + "\n"
        )
        while time.monotonic() - started < maximum_seconds:
            try:
                framework = read_framework_state(pid)
                board = read_live_board(pid)
            except (LiveBoardUnavailable, OSError) as error:
                unavailable_count += 1
                row = (
                    "unavailable",
                    str(error),
                )
                if row != previous:
                    stream.write(
                        json.dumps(
                            {
                                "type": "unavailable",
                                "perf_counter_ns": time.perf_counter_ns(),
                                "reason": str(error),
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    stream.flush()
                    change_count += 1
                    previous = row
                time.sleep(interval_seconds)
                continue
            sample_count += 1
            row = (
                "board",
                framework["framework_update"],
                board["native_game_time"],
                board["score"],
                board["displayed_score"],
                board["score_target"],
                board["runtime_active"],
                board["loss_counter"],
                board["curve_plan_exhausted"],
                board["active_ball_count"],
                board["inserting_ball_count"],
                board["pending_ball_count"],
                board["fired_ball_count"],
                board["current"]["ball_id"],
                board["current"]["color_id"],
                board["next"]["ball_id"],
                board["next"]["color_id"],
            )
            if row != previous:
                stream.write(
                    json.dumps(
                        {
                            "type": "board",
                            "perf_counter_ns": time.perf_counter_ns(),
                            "framework_update": framework[
                                "framework_update"
                            ],
                            "native_game_time": board["native_game_time"],
                            "score": board["score"],
                            "displayed_score": board["displayed_score"],
                            "score_target": board["score_target"],
                            "runtime_active": board["runtime_active"],
                            "loss_counter": board["loss_counter"],
                            "curve_plan_exhausted": board[
                                "curve_plan_exhausted"
                            ],
                            "active_ball_count": board[
                                "active_ball_count"
                            ],
                            "inserting_ball_count": board[
                                "inserting_ball_count"
                            ],
                            "pending_ball_count": board[
                                "pending_ball_count"
                            ],
                            "fired_ball_count": board[
                                "fired_ball_count"
                            ],
                            "current_ball_id": board["current"]["ball_id"],
                            "current_color_id": board["current"][
                                "color_id"
                            ],
                            "next_ball_id": board["next"]["ball_id"],
                            "next_color_id": board["next"]["color_id"],
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                stream.flush()
                change_count += 1
                previous = row
            time.sleep(interval_seconds)
    return {
        "status": "PASS",
        "process_id": pid,
        "sample_count": sample_count,
        "change_count": change_count,
        "unavailable_count": unavailable_count,
        "elapsed_seconds": time.monotonic() - started,
        "output": str(output),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--interval-seconds", type=float, default=0.05)
    parser.add_argument("--maximum-seconds", type=float, default=900.0)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = monitor(
        args.pid,
        output=args.output.resolve(),
        interval_seconds=args.interval_seconds,
        maximum_seconds=args.maximum_seconds,
    )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
