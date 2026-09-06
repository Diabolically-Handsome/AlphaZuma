"""Extract one provenance-bound ``file_read`` payload from a PopCap DMO.

The normal DMO parser deliberately hashes service payloads instead of
returning their bytes.  This recovery tool is narrower: the caller must bind
the exact source artifact, command index, payload length, and payload hash.
It writes a new file and provenance record with exclusive-create semantics.

This is intended for reconstructing a known local pre-replay state.  It does
not infer filenames, modify the source DMO, or overwrite an existing target.
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
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _normalise_sha256(value: str) -> str:
    result = value.lower()
    if result.startswith("sha256:"):
        result = result.removeprefix("sha256:")
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise ValueError("SHA-256 must contain exactly 64 hexadecimal digits")
    return result


def extract_file_read(
    data: bytes,
    *,
    command_index: int,
) -> tuple[bytes, dict[str, int | bool]]:
    """Return one successful file-read payload and its decoded coordinates."""

    if command_index < 0:
        raise ValueError("command index must be non-negative")
    module = _load_popcap_dmo_module()
    offset = _command_stream_offset(data)
    length_updates = struct.unpack_from("<I", data, offset - 4)[0]
    reader = module._BitReader(data[offset:])
    update = 0
    index = 0
    while reader.remaining_bits:
        if (
            reader.remaining_bits <= 7
            and reader.remaining_is_zero_padding()
        ):
            break
        start = reader.bit_position
        update += reader.read_bits(4, context="command timing")
        short_form = bool(
            reader.read_bits(1, context="command short-form flag")
        )
        command_number = reader.read_bits(
            1 if short_form else 5,
            context="command number",
        )
        if short_form:
            if command_number == 0:
                reader.read_bits(
                    6,
                    signed=True,
                    context="short mouse delta x",
                )
                reader.read_bits(
                    6,
                    signed=True,
                    context="short mouse delta y",
                )
            elif command_number == 1:
                reader.read_bits(1, context="mouse button state")
                reader.read_bits(
                    3,
                    signed=True,
                    context="mouse button number",
                )
            else:  # pragma: no cover - one-bit command number
                raise ValueError(
                    f"unsupported short command: {command_number}"
                )
        elif command_number == 15:
            success = bool(
                reader.read_bits(1, context="file_read success")
            )
            payload = b""
            if success:
                size = reader.read_i32("file data size")
                if size < 0:
                    raise ValueError("negative file-read payload size")
                payload = reader.read_bytes(size, "file data")
            if index == command_index:
                if not success:
                    raise ValueError(
                        f"command {command_index} is an unsuccessful file_read"
                    )
                return payload, {
                    "command_index": index,
                    "command_start_bit": start,
                    "command_end_bit": reader.bit_position,
                    "update": update,
                    "short_form": short_form,
                    "command_number": command_number,
                    "length_updates": length_updates,
                }
        else:
            module._read_long_command_payload(reader, command_number)
        if index == command_index:
            raise ValueError(
                f"command {command_index} is not a successful file_read"
            )
        if update > length_updates:
            raise ValueError("command update exceeds declared DMO length")
        index += 1
    raise ValueError(f"command index {command_index} does not exist")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--command-index", required=True, type=int)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-payload-bytes", required=True, type=int)
    parser.add_argument("--expected-payload-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.input.is_file():
        raise FileNotFoundError(f"input DMO does not exist: {args.input}")
    if args.expected_payload_bytes < 0:
        raise ValueError("expected payload bytes must be non-negative")
    provenance_path = args.output.with_name(
        args.output.name + ".provenance.json"
    )
    for path in (args.output, provenance_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite: {path}")

    source = args.input.read_bytes()
    source_sha256 = _sha256(source)
    expected_source = _normalise_sha256(args.expected_source_sha256)
    if source_sha256 != expected_source:
        raise ValueError(
            f"source SHA-256 mismatch: {source_sha256} != {expected_source}"
        )

    payload, coordinates = extract_file_read(
        source,
        command_index=args.command_index,
    )
    payload_sha256 = _sha256(payload)
    expected_payload = _normalise_sha256(
        args.expected_payload_sha256
    )
    if len(payload) != args.expected_payload_bytes:
        raise ValueError(
            "payload byte count mismatch: "
            f"{len(payload)} != {args.expected_payload_bytes}"
        )
    if payload_sha256 != expected_payload:
        raise ValueError(
            f"payload SHA-256 mismatch: {payload_sha256} != "
            f"{expected_payload}"
        )

    provenance = {
        "schema": "zuma-rl.dmo-file-read-extraction",
        "version": 1,
        "classification": "reconstructed-local-prestate-from-dmo-payload",
        "source": {
            "path": str(args.input.resolve()),
            "bytes": len(source),
            "sha256": source_sha256,
        },
        "command": coordinates,
        "output": {
            "path": str(args.output.resolve()),
            "bytes": len(payload),
            "sha256": payload_sha256,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(payload)
    with provenance_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(
            provenance,
            stream,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        stream.write("\n")

    print(f"output={args.output}")
    print(f"provenance={provenance_path}")
    print(f"bytes={len(payload)}")
    print(f"sha256={payload_sha256}")
    print(f"command_index={args.command_index}")
    print(f"update={coordinates['update']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
