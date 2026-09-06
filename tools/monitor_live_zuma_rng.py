"""Log read-only retail Board, QRand, and global-MTRand state changes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Mapping

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tools.autoplay_live_zuma import (
    LiveBoardUnavailable,
    read_framework_state,
    read_live_board,
)
from tools.collect_pc_memory_probe import (
    ProbeError,
    _read_rng_trajectory_tick,
    _resolve_thread_crt_state,
)
from tools.inspect_popcap_replay import (
    close_process,
    open_process_readonly,
)


def _chain_rows(
    curve_lists: Iterable[Mapping[str, Any]],
) -> list[dict[str, int]]:
    rows: list[dict[str, int]] = []
    for curve_list in curve_lists:
        if int(curve_list["container_offset"]) != 0x5C:
            continue
        for entity in curve_list["entities"]:
            if not isinstance(entity, Mapping):
                continue
            rows.append(
                {
                    "ball_id": int(entity["ball_id"]),
                    "color_id": int(entity["color_id"]),
                }
            )
    return rows


def monitor(
    pid: int,
    *,
    output: Path,
    interval_seconds: float,
    maximum_seconds: float,
    stop_after_native_game_time: int | None,
    thread_id: int | None = None,
    stop_file: Path | None = None,
) -> dict[str, Any]:
    """Capture compact RNG/topology changes without mutating the process."""

    if os.name != "nt":
        raise RuntimeError("live RNG monitoring requires Windows")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite: {output}")
    if interval_seconds <= 0 or maximum_seconds <= 0:
        raise ValueError("monitor timing is invalid")
    if (
        stop_after_native_game_time is not None
        and stop_after_native_game_time < 0
    ):
        raise ValueError("native-game-time stop must not be negative")
    if thread_id is not None and thread_id <= 0:
        raise ValueError("thread ID must be positive")
    if stop_file is not None and stop_file.exists():
        raise FileExistsError(f"stop file already exists: {stop_file}")

    output.parent.mkdir(parents=True, exist_ok=True)
    thread_crt_state = (
        _resolve_thread_crt_state(pid=pid, thread_id=thread_id)
        if thread_id is not None
        else None
    )
    handle = open_process_readonly(pid)
    started = time.monotonic()
    previous: tuple[Any, ...] | None = None
    sample_count = 0
    change_count = 0
    unavailable_count = 0
    last_native_game_time: int | None = None
    try:
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(
                json.dumps(
                    {
                        "schema": "zuma-rl.live-rng-change-monitor",
                        "version": 1,
                        "classification": (
                            "read-only-localization-diagnostic"
                        ),
                        "process_id": pid,
                        "process_memory_writes": 0,
                        "thread_crt_state": thread_crt_state,
                        "started_perf_counter_ns": time.perf_counter_ns(),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            while time.monotonic() - started < maximum_seconds:
                if stop_file is not None and stop_file.is_file():
                    break
                try:
                    framework = read_framework_state(pid)
                    board = read_live_board(pid)
                    rng, global_mtrand = _read_rng_trajectory_tick(
                        handle,
                        update=int(framework["framework_update"]),
                        thread_crt_state=thread_crt_state,
                    )
                except (
                    LiveBoardUnavailable,
                    OSError,
                    ProbeError,
                    RuntimeError,
                    ValueError,
                ) as error:
                    unavailable_count += 1
                    key = ("unavailable", str(error))
                    if key != previous:
                        stream.write(
                            json.dumps(
                                {
                                    "type": "unavailable",
                                    "perf_counter_ns": (
                                        time.perf_counter_ns()
                                    ),
                                    "reason": str(error),
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        )
                        stream.flush()
                        change_count += 1
                        previous = key
                    time.sleep(interval_seconds)
                    continue

                sample_count += 1
                native_game_time = int(board["native_game_time"])
                last_native_game_time = native_game_time
                chain = _chain_rows(rng["curve_lists"])
                current = rng["current_ball"]
                following = rng["next_ball"]
                qrand = rng["qrand"]
                global_sha256 = hashlib.sha256(global_mtrand).hexdigest()
                key = (
                    "rng",
                    int(framework["framework_update"]),
                    native_game_time,
                    int(rng["score"]),
                    tuple(int(value) for value in rng["board_color_counts"]),
                    tuple(
                        (row["ball_id"], row["color_id"]) for row in chain
                    ),
                    (
                        int(current["ball_id"]),
                        int(current["color_id"]),
                    )
                    if isinstance(current, Mapping)
                    else None,
                    (
                        int(following["ball_id"]),
                        int(following["color_id"]),
                    )
                    if isinstance(following, Mapping)
                    else None,
                    int(qrand["update_count"]),
                    int(qrand["selected_index"]),
                    rng["thread_crt_rand_state"],
                    tuple(
                        (
                            name,
                            tuple(values),
                        )
                        for name, values in sorted(
                            qrand["vectors"].items()
                        )
                    ),
                    int(rng["global_mtrand_index"]),
                    global_sha256,
                )
                if key != previous:
                    stream.write(
                        json.dumps(
                            {
                                "type": "rng",
                                "perf_counter_ns": time.perf_counter_ns(),
                                "framework_update": int(
                                    framework["framework_update"]
                                ),
                                "native_game_time": native_game_time,
                                "score": int(rng["score"]),
                                "displayed_score": int(
                                    rng["displayed_score"]
                                ),
                                "score_target": int(rng["score_target"]),
                                "board_color_counts": [
                                    int(value)
                                    for value in rng[
                                        "board_color_counts"
                                    ]
                                ],
                                "chain_ball_count": int(
                                    rng["chain_ball_count"]
                                ),
                                "pending_ball_count": int(
                                    rng["pending_ball_count"]
                                ),
                                "inserting_ball_count": int(
                                    rng["inserting_ball_count"]
                                ),
                                "fired_bullet_count": int(
                                    rng["fired_bullet_count"]
                                ),
                                "current_ball": current,
                                "next_ball": following,
                                "chain": chain,
                                "qrand": qrand,
                                "thread_crt_rand_state": rng[
                                    "thread_crt_rand_state"
                                ],
                                "global_mtrand_index": int(
                                    rng["global_mtrand_index"]
                                ),
                                "global_mtrand_sha256": global_sha256,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    stream.flush()
                    change_count += 1
                    previous = key

                if (
                    stop_after_native_game_time is not None
                    and native_game_time >= stop_after_native_game_time
                ):
                    break
                time.sleep(interval_seconds)
    finally:
        close_process(handle)

    return {
        "status": "PASS",
        "process_id": pid,
        "sample_count": sample_count,
        "change_count": change_count,
        "unavailable_count": unavailable_count,
        "last_native_game_time": last_native_game_time,
        "thread_id": thread_id,
        "thread_crt_state": thread_crt_state,
        "stopped_by_file": (
            stop_file is not None and stop_file.is_file()
        ),
        "elapsed_seconds": time.monotonic() - started,
        "output": str(output),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--interval-seconds", type=float, default=0.01)
    parser.add_argument("--maximum-seconds", type=float, default=900.0)
    parser.add_argument("--stop-after-native-game-time", type=int)
    parser.add_argument(
        "--thread-id",
        type=int,
        help=(
            "Resolve and capture the proven WOW64 thread-local MSVC CRT "
            "rand state on every RNG row."
        ),
    )
    parser.add_argument(
        "--stop-file",
        type=Path,
        help="stop cleanly when this new sentinel file appears",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = monitor(
        args.pid,
        output=args.output.resolve(),
        interval_seconds=args.interval_seconds,
        maximum_seconds=args.maximum_seconds,
        stop_after_native_game_time=args.stop_after_native_game_time,
        thread_id=args.thread_id,
        stop_file=(
            args.stop_file.resolve()
            if args.stop_file is not None
            else None
        ),
    )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
