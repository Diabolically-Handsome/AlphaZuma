"""Remap a full-screen PopCap DMO mouse trajectory into windowed coordinates.

The retail ``-play`` path creates a window even when the recorded DMO's
``video_data`` command says that recording was full-screen.  Absolute and
short-form mouse commands are not scaled by the framework, so a full-screen
recording can replay every command correctly while all clicks land outside
the 800x600 playback client.

Wide-screen full-screen recordings may contain a centred 4:3 game viewport.
In that case, coordinates are first translated out of the pillarbox margins,
clamped to the game viewport, and then scaled into the windowed client.

This diagnostic tool changes only mouse coordinate payload bits.  Command
headers, updates, button states, service payloads, and total bit length must
remain identical.  The source is never overwritten, the caller must bind its
SHA-256, and the output is reparsed and compared command-by-command before it
is written with exclusive-create semantics.

The result is diagnostic calibration material, not unmodified PC evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
from typing import Iterable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.align_popcap_dmo_startup import (
    _command_stream_offset,
    _load_popcap_dmo_module,
    _scan_commands,
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _scale_coordinate(value: int, source_size: int, target_size: int) -> int:
    if source_size <= 1 or target_size <= 1:
        raise ValueError("coordinate spaces must be at least 2 pixels")
    if not 0 <= value < source_size:
        raise ValueError(
            f"coordinate {value} is outside source extent {source_size}"
        )
    source_max = source_size - 1
    target_max = target_size - 1
    return (value * target_max + source_max // 2) // source_max


def _scale_viewport_coordinate(
    value: int,
    source_size: int,
    target_size: int,
    *,
    viewport_origin: int,
    viewport_size: int,
) -> int:
    if source_size <= 1:
        raise ValueError("source coordinate space must be at least 2 pixels")
    if not 0 <= value < source_size:
        raise ValueError(
            f"coordinate {value} is outside source extent {source_size}"
        )
    if viewport_origin < 0:
        raise ValueError("viewport origin must be non-negative")
    if viewport_size <= 1:
        raise ValueError("viewport extent must be at least 2 pixels")
    if viewport_origin + viewport_size > source_size:
        raise ValueError("viewport extends beyond the source coordinate space")
    relative = min(
        max(value - viewport_origin, 0),
        viewport_size - 1,
    )
    return _scale_coordinate(relative, viewport_size, target_size)


def _write_lsb_bits(
    data: bytearray,
    start: int,
    width: int,
    value: int,
    *,
    signed: bool = False,
) -> None:
    if width <= 0:
        raise ValueError("bit width must be positive")
    if signed:
        minimum = -(1 << (width - 1))
        maximum = (1 << (width - 1)) - 1
        if not minimum <= value <= maximum:
            raise ValueError(
                f"signed {width}-bit value out of range: {value}"
            )
        encoded = value & ((1 << width) - 1)
    else:
        if not 0 <= value < (1 << width):
            raise ValueError(
                f"unsigned {width}-bit value out of range: {value}"
            )
        encoded = value
    for offset in range(width):
        position = start + offset
        mask = 1 << (position % 8)
        if (encoded >> offset) & 1:
            data[position // 8] |= mask
        else:
            data[position // 8] &= ~mask


def _plain_payload(command) -> dict[str, object]:
    return dict(command.payload)


def remap_mouse_space(
    data: bytes,
    module,
    *,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
    source_viewport_left: int = 0,
    source_viewport_top: int = 0,
    source_viewport_width: int | None = None,
    source_viewport_height: int | None = None,
) -> tuple[bytes, dict[str, object]]:
    viewport_width = (
        source_width
        if source_viewport_width is None
        else source_viewport_width
    )
    viewport_height = (
        source_height
        if source_viewport_height is None
        else source_viewport_height
    )
    # Validate the viewport even if the DMO contains no mouse commands.
    _scale_viewport_coordinate(
        source_viewport_left,
        source_width,
        target_width,
        viewport_origin=source_viewport_left,
        viewport_size=viewport_width,
    )
    _scale_viewport_coordinate(
        source_viewport_top,
        source_height,
        target_height,
        viewport_origin=source_viewport_top,
        viewport_size=viewport_height,
    )

    offset = _command_stream_offset(data)
    length_updates = struct.unpack_from("<I", data, offset - 4)[0]
    command_data = data[offset:]
    rows, meaningful_bits = _scan_commands(
        module,
        command_data,
        length_updates,
    )
    parsed_source = module.PopCapDemo.from_bytes(data)
    if len(rows) != len(parsed_source.commands):
        raise RuntimeError("scanner/parser command counts differ")

    output_command_data = bytearray(command_data)
    target_mouse_x = 0
    target_mouse_y = 0
    transformed_absolute = 0
    transformed_relative = 0
    button_commands = 0
    clamped_mouse_commands = 0

    for index, (row, command) in enumerate(
        zip(rows, parsed_source.commands, strict=True)
    ):
        row_kind = (
            "mouse_move"
            if not bool(row["short_form"])
            and int(row["command_number"]) == 0
            else str(row["kind"])
        )
        if (
            int(row["update"]) != command.update
            or bool(row["short_form"]) != command.short_form
            or int(row["command_number"]) != command.command_number
            or row_kind != command.kind
        ):
            raise RuntimeError(f"scanner/parser mismatch at command {index}")

        payload = _plain_payload(command)
        if command.kind == "mouse_move":
            source_x = int(payload["x"])
            source_y = int(payload["y"])
            next_target_x = _scale_viewport_coordinate(
                source_x,
                source_width,
                target_width,
                viewport_origin=source_viewport_left,
                viewport_size=viewport_width,
            )
            next_target_y = _scale_viewport_coordinate(
                source_y,
                source_height,
                target_height,
                viewport_origin=source_viewport_top,
                viewport_size=viewport_height,
            )
            if not (
                source_viewport_left
                <= source_x
                < source_viewport_left + viewport_width
                and source_viewport_top
                <= source_y
                < source_viewport_top + viewport_height
            ):
                clamped_mouse_commands += 1
            start = int(row["start"])
            if command.short_form:
                delta_x = next_target_x - target_mouse_x
                delta_y = next_target_y - target_mouse_y
                _write_lsb_bits(
                    output_command_data,
                    start + 6,
                    6,
                    delta_x,
                    signed=True,
                )
                _write_lsb_bits(
                    output_command_data,
                    start + 12,
                    6,
                    delta_y,
                    signed=True,
                )
                transformed_relative += 1
            else:
                _write_lsb_bits(
                    output_command_data,
                    start + 10,
                    12,
                    next_target_x,
                )
                _write_lsb_bits(
                    output_command_data,
                    start + 22,
                    12,
                    next_target_y,
                )
                transformed_absolute += 1
            target_mouse_x = next_target_x
            target_mouse_y = next_target_y
        elif command.kind == "mouse_button":
            source_x = int(payload["x"])
            source_y = int(payload["y"])
            expected_x = _scale_viewport_coordinate(
                source_x,
                source_width,
                target_width,
                viewport_origin=source_viewport_left,
                viewport_size=viewport_width,
            )
            expected_y = _scale_viewport_coordinate(
                source_y,
                source_height,
                target_height,
                viewport_origin=source_viewport_top,
                viewport_size=viewport_height,
            )
            if (target_mouse_x, target_mouse_y) != (
                expected_x,
                expected_y,
            ):
                raise RuntimeError(
                    f"mouse state mismatch before button command {index}"
                )
            button_commands += 1

    output = data[:offset] + bytes(output_command_data)
    if len(output) != len(data):
        raise RuntimeError("mouse remap changed DMO byte length")
    parsed_output = module.PopCapDemo.from_bytes(output)
    if len(parsed_output.commands) != len(parsed_source.commands):
        raise RuntimeError("mouse remap changed DMO command count")
    if parsed_output.length_updates != parsed_source.length_updates:
        raise RuntimeError("mouse remap changed DMO length_updates")

    previous_target_x = 0
    previous_target_y = 0
    for index, (source_command, output_command) in enumerate(
        zip(
            parsed_source.commands,
            parsed_output.commands,
            strict=True,
        )
    ):
        source_identity = (
            source_command.update,
            source_command.kind,
            source_command.short_form,
            source_command.command_number,
        )
        output_identity = (
            output_command.update,
            output_command.kind,
            output_command.short_form,
            output_command.command_number,
        )
        if output_identity != source_identity:
            raise RuntimeError(
                f"command identity changed at command {index}"
            )
        source_payload = _plain_payload(source_command)
        output_payload = _plain_payload(output_command)
        if source_command.kind == "mouse_move":
            expected_x = _scale_viewport_coordinate(
                int(source_payload["x"]),
                source_width,
                target_width,
                viewport_origin=source_viewport_left,
                viewport_size=viewport_width,
            )
            expected_y = _scale_viewport_coordinate(
                int(source_payload["y"]),
                source_height,
                target_height,
                viewport_origin=source_viewport_top,
                viewport_size=viewport_height,
            )
            if (
                int(output_payload["x"]),
                int(output_payload["y"]),
            ) != (expected_x, expected_y):
                raise RuntimeError(
                    f"scaled mouse position mismatch at command {index}"
                )
            if source_command.short_form and (
                int(output_payload["delta_x"]),
                int(output_payload["delta_y"]),
            ) != (
                expected_x - previous_target_x,
                expected_y - previous_target_y,
            ):
                raise RuntimeError(
                    f"scaled mouse delta mismatch at command {index}"
                )
            previous_target_x = expected_x
            previous_target_y = expected_y
        elif source_command.kind == "mouse_button":
            expected_payload = {
                **source_payload,
                "x": previous_target_x,
                "y": previous_target_y,
            }
            if output_payload != expected_payload:
                raise RuntimeError(
                    f"mouse button semantics changed at command {index}"
                )
        elif output_payload != source_payload:
            raise RuntimeError(
                f"non-mouse payload changed at command {index}"
            )

    provenance = {
        "schema": "zuma-rl.diagnostic-dmo-mouse-space-remap",
        "version": 1,
        "classification": "diagnostic-not-unmodified-pc-evidence",
        "source": {
            "bytes": len(data),
            "sha256": _sha256(data),
            "commands": len(parsed_source.commands),
            "length_updates": parsed_source.length_updates,
            "meaningful_command_bits": meaningful_bits,
            "mouse_space": [source_width, source_height],
            "game_viewport": [
                source_viewport_left,
                source_viewport_top,
                viewport_width,
                viewport_height,
            ],
        },
        "transformation": {
            "mapping": (
                "clamp(round-half-up((value - viewport_origin) * "
                "(target_size - 1) / (viewport_size - 1)), "
                "0, target_size - 1)"
            ),
            "target_mouse_space": [target_width, target_height],
            "absolute_mouse_commands": transformed_absolute,
            "relative_mouse_commands": transformed_relative,
            "button_commands_verified": button_commands,
            "clamped_mouse_commands": clamped_mouse_commands,
            "preserved": [
                "command headers",
                "command updates",
                "button states",
                "non-mouse payloads",
                "command bit length",
            ],
            "reason": (
                "retail -play creates an 800x600 window for a DMO "
                "recorded in full-screen mode"
            ),
        },
        "output": {
            "bytes": len(output),
            "sha256": _sha256(output),
            "commands": len(parsed_output.commands),
            "length_updates": parsed_output.length_updates,
            "meaningful_command_bits": meaningful_bits,
            "mouse_space": [target_width, target_height],
        },
    }
    return output, provenance


def _positive_int(text: str) -> int:
    value = int(text, 0)
    if value <= 1:
        raise argparse.ArgumentTypeError("dimension must be at least 2")
    return value


def _non_negative_int(text: str) -> int:
    value = int(text, 0)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--source-width", type=_positive_int, required=True)
    parser.add_argument("--source-height", type=_positive_int, required=True)
    parser.add_argument("--target-width", type=_positive_int, required=True)
    parser.add_argument("--target-height", type=_positive_int, required=True)
    parser.add_argument(
        "--source-viewport-left",
        type=_non_negative_int,
    )
    parser.add_argument(
        "--source-viewport-top",
        type=_non_negative_int,
    )
    parser.add_argument(
        "--source-viewport-width",
        type=_positive_int,
    )
    parser.add_argument(
        "--source-viewport-height",
        type=_positive_int,
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    viewport_args = (
        args.source_viewport_left,
        args.source_viewport_top,
        args.source_viewport_width,
        args.source_viewport_height,
    )
    if any(value is not None for value in viewport_args) and not all(
        value is not None for value in viewport_args
    ):
        parser.error(
            "source viewport left, top, width, and height must be "
            "provided together"
        )
    if not args.input.is_file():
        raise FileNotFoundError(f"input DMO does not exist: {args.input}")
    output = args.output
    provenance_path = output.with_name(output.name + ".provenance.json")
    for path in (output, provenance_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite: {path}")
    source = args.input.read_bytes()
    source_sha256 = _sha256(source)
    expected = args.expected_source_sha256.lower()
    if source_sha256 != expected:
        raise ValueError(
            f"source SHA-256 mismatch: {source_sha256} != {expected}"
        )
    module = _load_popcap_dmo_module()
    transformed, provenance = remap_mouse_space(
        source,
        module,
        source_width=args.source_width,
        source_height=args.source_height,
        target_width=args.target_width,
        target_height=args.target_height,
        source_viewport_left=(
            0
            if args.source_viewport_left is None
            else args.source_viewport_left
        ),
        source_viewport_top=(
            0
            if args.source_viewport_top is None
            else args.source_viewport_top
        ),
        source_viewport_width=args.source_viewport_width,
        source_viewport_height=args.source_viewport_height,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(transformed)
    with provenance_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(
            provenance,
            stream,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        stream.write("\n")
    print(f"output={output}")
    print(f"provenance={provenance_path}")
    print(f"sha256={provenance['output']['sha256']}")
    print(f"bytes={provenance['output']['bytes']}")
    print(f"commands={provenance['output']['commands']}")
    print(f"length_updates={provenance['output']['length_updates']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
