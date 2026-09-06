"""Create a provenance-bound diagnostic DMO with one duplicate startup read removed.

Some Zuma's Revenge recordings begin with two bit-identical ``Is3D`` registry
reads while the retail playback path requests only one before its three D3D
sync commands.  The extra command shifts the shared DMO bit stream and makes
the unmodified recording unplayable.

This tool is deliberately narrow.  It refuses to transform a stream unless:

* the caller supplies the exact source SHA-256;
* commands zero and one are bit-identical 107-bit ``registry_read`` commands;
* both payloads are ``REG_DWORD(1)``; and
* command two is ``sync`` at update zero.

The source is never changed.  A new DMO and JSON provenance record are created
with exclusive-create semantics, then parsed again before success is reported.
The result is diagnostic calibration material, not an unmodified PC capture.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
from typing import Iterable


def _load_popcap_dmo_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "zuma_rl"
        / "popcap_dmo.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_zuma_rl_popcap_dmo_alignment",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load PopCap DMO parser")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _command_stream_offset(data: bytes) -> int:
    if len(data) < 20:
        raise ValueError("DMO is too short")
    if struct.unpack_from("<I", data, 0)[0] != 0x42BEEF78:
        raise ValueError("invalid DMO file id")
    version = struct.unpack_from("<I", data, 4)[0]
    if version not in (1, 2):
        raise ValueError(f"unsupported DMO version: {version}")
    offset = 12
    product_length = struct.unpack_from("<H", data, offset)[0]
    offset += 2 + product_length
    if version >= 2:
        marker_size = struct.unpack_from("<I", data, offset)[0]
        offset += 4 + marker_size
    offset += 4
    if offset >= len(data):
        raise ValueError("DMO has no command stream")
    return offset


def _bit_tuple(data: bytes, start: int, end: int) -> tuple[int, ...]:
    return tuple(
        (data[position // 8] >> (position % 8)) & 1
        for position in range(start, end)
    )


def _pack_lsb_bits(bits: Iterable[int]) -> bytes:
    values = tuple(bits)
    output = bytearray((len(values) + 7) // 8)
    for position, value in enumerate(values):
        if value not in (0, 1):
            raise ValueError("bit values must be zero or one")
        if value:
            output[position // 8] |= 1 << (position % 8)
    return bytes(output)


def _scan_commands(module, command_data: bytes, length_updates: int):
    reader = module._BitReader(command_data)
    update = 0
    rows: list[dict[str, object]] = []
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
                reader.read_bits(6, signed=True, context="mouse delta x")
                reader.read_bits(6, signed=True, context="mouse delta y")
                kind = "mouse_move"
            elif command_number == 1:
                reader.read_bits(1, context="mouse button state")
                reader.read_bits(3, signed=True, context="mouse button")
                kind = "mouse_button"
            else:
                raise ValueError(
                    f"unsupported short command: {command_number}"
                )
            payload: dict[str, object] = {}
        else:
            try:
                kind = module._COMMAND_NAMES[command_number]
            except KeyError as error:
                raise ValueError(
                    f"unsupported long command: {command_number}"
                ) from error
            payload = module._read_long_command_payload(
                reader,
                command_number,
            )
        rows.append(
            {
                "start": start,
                "end": reader.bit_position,
                "update": update,
                "short_form": short_form,
                "command_number": command_number,
                "kind": kind,
                "payload": payload,
            }
        )
        if update > length_updates:
            raise ValueError("command update exceeds declared DMO length")
    return rows, reader.bit_position


def align_startup(data: bytes, module) -> tuple[bytes, dict[str, object]]:
    offset = _command_stream_offset(data)
    length_updates = struct.unpack_from("<I", data, offset - 4)[0]
    command_data = data[offset:]
    rows, meaningful_bits = _scan_commands(
        module,
        command_data,
        length_updates,
    )
    if len(rows) < 3:
        raise ValueError("DMO has fewer than three commands")
    first, second, third = rows[:3]
    expected_registry = {
        "update": 0,
        "short_form": False,
        "command_number": 11,
        "kind": "registry_read",
    }
    for index, row in enumerate((first, second)):
        for key, expected in expected_registry.items():
            if row[key] != expected:
                raise ValueError(
                    f"startup command {index} {key} mismatch: "
                    f"{row[key]!r} != {expected!r}"
                )
        payload = dict(row["payload"])
        if (
            payload.get("success") is not True
            or payload.get("value_type") != 4
            or payload.get("value", {}).get("bytes") != 4
        ):
            raise ValueError(
                f"startup registry payload {index} is not REG_DWORD"
            )
    if (
        first["start"] != 0
        or first["end"] != 107
        or second["start"] != 107
        or second["end"] != 214
    ):
        raise ValueError("startup registry command spans are not 107 bits each")
    first_bits = _bit_tuple(
        command_data,
        int(first["start"]),
        int(first["end"]),
    )
    second_bits = _bit_tuple(
        command_data,
        int(second["start"]),
        int(second["end"]),
    )
    if first_bits != second_bits:
        raise ValueError("startup registry commands are not bit-identical")
    if (
        third["start"] != 214
        or third["update"] != 0
        or third["short_form"] is not False
        or third["command_number"] != 18
        or third["kind"] != "sync"
    ):
        raise ValueError("third startup command is not the expected sync")

    source_bits = _bit_tuple(command_data, 0, meaningful_bits)
    aligned_bits = source_bits[int(first["end"]) :]
    aligned_command_data = _pack_lsb_bits(aligned_bits)
    aligned = data[:offset] + aligned_command_data
    parsed_source = module.PopCapDemo.from_bytes(data)
    parsed_aligned = module.PopCapDemo.from_bytes(aligned)
    if parsed_aligned.length_updates != parsed_source.length_updates:
        raise RuntimeError("aligned DMO length changed")
    if len(parsed_aligned.commands) != len(parsed_source.commands) - 1:
        raise RuntimeError("aligned DMO command count did not decrease by one")
    if parsed_aligned.commands[0].kind != "registry_read":
        raise RuntimeError("aligned DMO no longer begins with registry_read")
    if parsed_aligned.commands[1].kind != "sync":
        raise RuntimeError("aligned DMO second command is not sync")

    provenance = {
        "schema": "zuma-rl.diagnostic-dmo-startup-alignment",
        "version": 1,
        "classification": "diagnostic-not-unmodified-pc-evidence",
        "source": {
            "bytes": len(data),
            "sha256": _sha256(data),
            "commands": len(parsed_source.commands),
            "length_updates": parsed_source.length_updates,
            "meaningful_command_bits": meaningful_bits,
        },
        "transformation": {
            "removed_command_index": 0,
            "removed_bit_start": int(first["start"]),
            "removed_bit_end": int(first["end"]),
            "removed_bits": int(first["end"]) - int(first["start"]),
            "reason": (
                "bit-identical duplicate startup registry_read before "
                "the first D3D sync command"
            ),
        },
        "output": {
            "bytes": len(aligned),
            "sha256": _sha256(aligned),
            "commands": len(parsed_aligned.commands),
            "length_updates": parsed_aligned.length_updates,
            "meaningful_command_bits": len(aligned_bits),
        },
    }
    return aligned, provenance


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a diagnostic DMO after verifying and removing one "
            "bit-identical duplicate startup registry read."
        )
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-source-sha256", required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
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
    aligned, provenance = align_startup(source, module)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(aligned)
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
