"""Create a provenance-bound diagnostic DMO with a selected video mode.

Only the ``video_data.windowed`` payload byte is changed.  The source is
never overwritten, every command is reparsed, and all command identities,
updates, refresh rates, and non-video payloads must remain identical.

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
from tools.transform_popcap_dmo_mouse_space import _write_lsb_bits


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def set_windowed(
    data: bytes,
    module,
    *,
    windowed: bool,
) -> tuple[bytes, dict[str, object]]:
    offset = _command_stream_offset(data)
    length_updates = struct.unpack_from("<I", data, offset - 4)[0]
    command_data = data[offset:]
    rows, meaningful_bits = _scan_commands(
        module,
        command_data,
        length_updates,
    )
    source = module.PopCapDemo.from_bytes(data)
    if len(rows) != len(source.commands):
        raise RuntimeError("scanner/parser command counts differ")

    output_commands = bytearray(command_data)
    transformed: list[int] = []
    for index, (row, command) in enumerate(
        zip(rows, source.commands, strict=True)
    ):
        if command.kind != "video_data":
            continue
        if (
            bool(row["short_form"])
            or int(row["command_number"]) != 23
            or int(row["end"]) - int(row["start"]) != 26
        ):
            raise RuntimeError("video command encoding is unexpected")
        payload = dict(command.payload)
        before_windowed = payload.get("windowed")
        if not isinstance(before_windowed, bool):
            raise RuntimeError("video command windowed value is invalid")
        if before_windowed is not windowed:
            _write_lsb_bits(
                output_commands,
                int(row["start"]) + 10,
                8,
                int(windowed),
            )
            transformed.append(index)

    if not transformed:
        mode = "windowed" if windowed else "fullscreen"
        raise ValueError(f"DMO has no video_data command to force {mode}")
    output = data[:offset] + bytes(output_commands)
    if len(output) != len(data):
        raise RuntimeError("video transformation changed DMO byte length")

    parsed = module.PopCapDemo.from_bytes(output)
    if (
        parsed.length_updates != source.length_updates
        or len(parsed.commands) != len(source.commands)
    ):
        raise RuntimeError("video transformation changed DMO structure")
    transformed_set = set(transformed)
    for index, (before, after) in enumerate(
        zip(source.commands, parsed.commands, strict=True)
    ):
        if (
            before.update,
            before.kind,
            before.short_form,
            before.command_number,
        ) != (
            after.update,
            after.kind,
            after.short_form,
            after.command_number,
        ):
            raise RuntimeError(f"command identity changed at command {index}")
        before_payload = dict(before.payload)
        after_payload = dict(after.payload)
        if index in transformed_set:
            expected = {**before_payload, "windowed": windowed}
            if before_payload.get("windowed") is windowed:
                raise RuntimeError("transformed video source mode was unchanged")
        else:
            expected = before_payload
        if after_payload != expected:
            raise RuntimeError(f"command payload changed at command {index}")

    return output, {
        "meaningful_command_bits": meaningful_bits,
        "transformed_command_indices": transformed,
        "transformed_video_commands": len(transformed),
        "target_windowed": windowed,
    }


def force_windowed(
    data: bytes,
    module,
) -> tuple[bytes, dict[str, object]]:
    return set_windowed(data, module, windowed=True)


def force_fullscreen(
    data: bytes,
    module,
) -> tuple[bytes, dict[str, object]]:
    return set_windowed(data, module, windowed=False)


def _write_exclusive(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument(
        "--mode",
        choices=("windowed", "fullscreen"),
        default="windowed",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    source_path = args.source.resolve(strict=True)
    output_path = args.output.resolve()
    provenance_path = args.provenance.resolve()
    if source_path in {output_path, provenance_path}:
        raise ValueError("source and output paths must be distinct")
    if output_path.parent != provenance_path.parent:
        raise ValueError("output and provenance must share a directory")
    if not output_path.parent.is_dir():
        raise FileNotFoundError("output directory does not exist")
    module = _load_popcap_dmo_module()
    source = source_path.read_bytes()
    target_windowed = args.mode == "windowed"
    output, details = set_windowed(
        source,
        module,
        windowed=target_windowed,
    )
    parsed_source = module.PopCapDemo.from_bytes(source)
    parsed_output = module.PopCapDemo.from_bytes(output)
    provenance = {
        "schema": f"zuma-rl.diagnostic-dmo-{args.mode}-video",
        "version": 1,
        "classification": "diagnostic-not-unmodified-pc-evidence",
        "source": {
            "bytes": len(source),
            "sha256": _sha256(source),
            "commands": len(parsed_source.commands),
            "length_updates": parsed_source.length_updates,
        },
        "output": {
            "bytes": len(output),
            "sha256": _sha256(output),
            "commands": len(parsed_output.commands),
            "length_updates": parsed_output.length_updates,
        },
        "transformation": {
            "reason": (
                "force the recorded video_data command to use the retail "
                f"{args.mode} replay mode"
            ),
            "preserved": [
                "command count",
                "command headers",
                "command updates",
                "refresh rates",
                "all non-video payloads",
                "command bit length",
            ],
            **details,
        },
    }
    provenance_bytes = (
        json.dumps(
            provenance,
            ensure_ascii=True,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")
    _write_exclusive(output_path, output)
    try:
        _write_exclusive(provenance_path, provenance_bytes)
    except BaseException:
        output_path.unlink(missing_ok=True)
        raise
    print(
        json.dumps(
            provenance,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
