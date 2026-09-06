"""Measure DXGI one-shot capture cadence without publishing artifacts."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from typing import Any

import dxcam
import numpy as np


def _sample(*, copy_mode: bool, duration_seconds: float) -> dict[str, Any]:
    camera = dxcam.create(
        device_idx=0,
        output_idx=0,
        output_color="BGRA",
        backend="dxgi",
        processor_backend="numpy",
    )
    accumulated: list[int] = []
    grab_ms: list[float] = []
    idle_polls = 0
    try:
        desktop = camera._output.desc.DesktopCoordinates
        region = (
            int(desktop.left),
            int(desktop.top),
            int(desktop.right),
            int(desktop.bottom),
        )
        deadline = time.perf_counter() + duration_seconds
        while time.perf_counter() < deadline:
            started_ns = time.perf_counter_ns()
            frame = camera.grab(
                region=region,
                copy=copy_mode,
                new_frame_only=True,
            )
            ended_ns = time.perf_counter_ns()
            if frame is None:
                idle_polls += 1
                time.sleep(0.0005)
                continue
            accumulated.append(int(camera._duplicator.accumulated_frames))
            grab_ms.append((ended_ns - started_ns) / 1_000_000)
    finally:
        camera.release()
    return {
        "mode": f"grab_copy_{str(copy_mode).lower()}",
        "frames": len(accumulated),
        "idle_polls": idle_polls,
        "accumulated_histogram": dict(sorted(Counter(accumulated).items())),
        "grab_ms_min": min(grab_ms, default=None),
        "grab_ms_mean": (
            sum(grab_ms) / len(grab_ms) if grab_ms else None
        ),
        "grab_ms_max": max(grab_ms, default=None),
    }


def _sample_preallocated(*, duration_seconds: float) -> dict[str, Any]:
    camera = dxcam.create(
        device_idx=0,
        output_idx=0,
        output_color="BGRA",
        backend="dxgi",
        processor_backend="numpy",
    )
    accumulated: list[int] = []
    grab_ms: list[float] = []
    idle_polls = 0
    try:
        desktop = camera._output.desc.DesktopCoordinates
        region = (
            int(desktop.left),
            int(desktop.top),
            int(desktop.right),
            int(desktop.bottom),
        )
        width = region[2] - region[0]
        height = region[3] - region[1]
        frame_budget = math.ceil(duration_seconds * 180) + 4
        buffers = [
            np.zeros((height, width, 4), dtype=np.uint8)
            for _ in range(frame_budget)
        ]
        deadline = time.perf_counter() + duration_seconds
        while time.perf_counter() < deadline:
            if len(accumulated) >= len(buffers):
                raise RuntimeError("preallocated diagnostic budget exceeded")
            started_ns = time.perf_counter_ns()
            captured, _ticks, frame_width, frame_height = (
                camera._grab_into(
                    region,
                    buffers[len(accumulated)],
                )
            )
            ended_ns = time.perf_counter_ns()
            if not captured:
                if frame_width or frame_height:
                    raise RuntimeError("preallocated destination mismatch")
                idle_polls += 1
                time.sleep(0.0005)
                continue
            accumulated.append(int(camera._duplicator.accumulated_frames))
            grab_ms.append((ended_ns - started_ns) / 1_000_000)
    finally:
        camera.release()
    return {
        "mode": "preallocated_grab_into",
        "frames": len(accumulated),
        "idle_polls": idle_polls,
        "accumulated_histogram": dict(sorted(Counter(accumulated).items())),
        "grab_ms_min": min(grab_ms, default=None),
        "grab_ms_mean": (
            sum(grab_ms) / len(grab_ms) if grab_ms else None
        ),
        "grab_ms_max": max(grab_ms, default=None),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=0.35,
    )
    args = parser.parse_args()
    if not 0 < args.duration_seconds <= 5:
        parser.error("--duration-seconds must be in (0, 5]")
    print(
        json.dumps(
            [
                _sample(
                    copy_mode=True,
                    duration_seconds=args.duration_seconds,
                ),
                _sample(
                    copy_mode=False,
                    duration_seconds=args.duration_seconds,
                ),
                _sample_preallocated(
                    duration_seconds=args.duration_seconds,
                ),
            ],
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
