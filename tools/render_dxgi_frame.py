"""Render one hash-verified raw DXGI frame to a review PNG."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable

from PIL import Image


def render(capture_root: Path, sequence: int, output: Path) -> None:
    metadata = json.loads(
        (capture_root / "metadata.json").read_text(encoding="ascii")
    )
    if (
        metadata.get("schema") != "zuma-rl.dxgi-bgra-capture"
        or metadata.get("status") != "acquisition_complete"
        or metadata.get("pixel_format") != "bgra"
        or metadata.get("bytes_per_pixel") != 4
    ):
        raise ValueError("capture metadata is unsupported")
    width = int(metadata["width"])
    height = int(metadata["height"])
    frame_bytes = width * height * 4

    with (capture_root / metadata["frames_csv"]).open(
        "r",
        encoding="ascii",
        newline="",
    ) as stream:
        rows = list(csv.DictReader(stream))
    if not 0 <= sequence < len(rows):
        raise ValueError("sequence is outside the capture")
    row = rows[sequence]
    if (
        int(row["sequence"]) != sequence
        or int(row["raw_bytes"]) != frame_bytes
    ):
        raise ValueError("frame row identity is invalid")

    with (capture_root / metadata["raw_frames"]).open("rb") as stream:
        stream.seek(int(row["raw_offset"]))
        payload = stream.read(frame_bytes)
    if len(payload) != frame_bytes:
        raise ValueError("raw frame is truncated")
    digest = f"sha256:{hashlib.sha256(payload).hexdigest()}"
    if digest != row["frame_sha256"]:
        raise ValueError("raw frame hash differs from frames.csv")
    if output.exists() or not output.parent.is_dir():
        raise FileExistsError("output must be a new file in an existing folder")

    image = Image.frombytes(
        "RGBA",
        (width, height),
        payload,
        "raw",
        "BGRA",
    ).convert("RGB")
    image.save(output, format="PNG", optimize=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture_root", type=Path)
    parser.add_argument("sequence", type=int)
    parser.add_argument("output", type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    render(
        args.capture_root.resolve(),
        args.sequence,
        args.output.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
