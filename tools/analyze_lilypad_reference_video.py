#!/usr/bin/env python3
"""Create timestamped, hash-anchored contact sheets for lily-pad calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import av
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class Sample:
    requested_seconds: float
    decoded_seconds: float
    frame_index: int
    image: Image.Image


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float)
    parser.add_argument("--step", type=float, default=5.0)
    parser.add_argument(
        "--source-offset",
        type=float,
        default=0.0,
        help="Seconds in the source video corresponding to local time zero.",
    )
    parser.add_argument("--columns", type=int, default=6)
    parser.add_argument("--thumb-width", type=int, default=320)
    return parser.parse_args()


def _format_time(seconds: float) -> str:
    milliseconds = int(round(max(0.0, seconds) * 1000.0))
    minutes, remainder = divmod(milliseconds, 60_000)
    whole_seconds, millis = divmod(remainder, 1000)
    return f"{minutes:02d}:{whole_seconds:02d}.{millis:03d}"


def _sample_video(
    video: Path,
    *,
    start: float,
    end: float,
    step: float,
) -> tuple[list[Sample], dict[str, object]]:
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        average_rate = float(stream.average_rate) if stream.average_rate else None
        duration = (
            float(stream.duration * stream.time_base)
            if stream.duration is not None
            else float(container.duration / av.time_base)
        )
        effective_end = min(end, duration)
        targets: list[float] = []
        value = start
        while value <= effective_end + 1e-9:
            targets.append(value)
            value += step

        samples: list[Sample] = []
        target_index = 0
        frame_index = -1
        for frame in container.decode(stream):
            frame_index += 1
            if frame.time is None:
                continue
            decoded_seconds = float(frame.time)
            while target_index < len(targets) and decoded_seconds + 1e-9 >= targets[target_index]:
                samples.append(
                    Sample(
                        requested_seconds=targets[target_index],
                        decoded_seconds=decoded_seconds,
                        frame_index=frame_index,
                        image=frame.to_image().convert("RGB"),
                    )
                )
                target_index += 1
            if target_index >= len(targets):
                break

        metadata: dict[str, object] = {
            "duration_seconds": duration,
            "average_rate_fps": average_rate,
            "width": int(stream.codec_context.width),
            "height": int(stream.codec_context.height),
            "time_base": str(stream.time_base),
            "requested_start_seconds": start,
            "requested_end_seconds": effective_end,
            "requested_step_seconds": step,
            "requested_sample_count": len(targets),
            "decoded_sample_count": len(samples),
        }
        return samples, metadata


def _render_sheet(
    samples: list[Sample],
    *,
    columns: int,
    thumb_width: int,
    source_offset: float,
) -> Image.Image:
    if not samples:
        raise ValueError("no video samples were decoded")
    source_width, source_height = samples[0].image.size
    thumb_height = max(1, round(thumb_width * source_height / source_width))
    label_height = 38
    rows = math.ceil(len(samples) / columns)
    sheet = Image.new("RGB", (columns * thumb_width, rows * (thumb_height + label_height)), "#111111")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=16)

    for index, sample in enumerate(samples):
        column = index % columns
        row = index // columns
        x = column * thumb_width
        y = row * (thumb_height + label_height)
        thumbnail = sample.image.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS)
        sheet.paste(thumbnail, (x, y))
        local = _format_time(sample.decoded_seconds)
        source = _format_time(sample.decoded_seconds + source_offset)
        draw.text((x + 4, y + thumb_height + 2), f"local {local}  source {source}", fill="white", font=font)
    return sheet


def main() -> int:
    args = _parse_args()
    if args.start < 0.0:
        raise ValueError("--start must be non-negative")
    if args.step <= 0.0:
        raise ValueError("--step must be positive")
    if args.columns <= 0 or args.thumb_width <= 0:
        raise ValueError("--columns and --thumb-width must be positive")
    video = args.video.resolve(strict=True)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        duration = (
            float(stream.duration * stream.time_base)
            if stream.duration is not None
            else float(container.duration / av.time_base)
        )
    end = duration if args.end is None else args.end
    if end < args.start:
        raise ValueError("--end must not precede --start")

    samples, metadata = _sample_video(
        video,
        start=args.start,
        end=end,
        step=args.step,
    )
    sheet = _render_sheet(
        samples,
        columns=args.columns,
        thumb_width=args.thumb_width,
        source_offset=args.source_offset,
    )
    sheet.save(output, format="PNG", optimize=True)

    receipt = {
        "schema": "zuma-rl.lilypad-reference-contact-sheet",
        "version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "video": {
            "path": str(video),
            "sha256": f"sha256:{_sha256(video)}",
        },
        "contact_sheet": {
            "path": str(output),
            "sha256": f"sha256:{_sha256(output)}",
        },
        "source_offset_seconds": args.source_offset,
        "metadata": metadata,
        "samples": [
            {
                "requested_seconds": sample.requested_seconds,
                "decoded_seconds": sample.decoded_seconds,
                "source_seconds": sample.decoded_seconds + args.source_offset,
                "frame_index": sample.frame_index,
            }
            for sample in samples
        ],
    }
    receipt_path = output.with_suffix(".json")
    receipt_path.write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
