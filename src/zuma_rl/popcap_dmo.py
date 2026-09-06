"""Read-only parser for PopCap SexyAppFramework ``.dmo`` recordings.

The original PC executable contains the framework's ``-record``, ``-play``
and ``-demofile`` switches.  A DMO is substantially better input evidence
than reconstructing clicks from video: commands are timestamped in framework
update counts and the file header stores the framework RNG seed.

This module deliberately does not claim that a framework update is already
aligned to a :class:`~zuma_rl.revenge_core.RevengeSimulator` native tick.
That phase relationship must be established by a visible event in each PC
capture.  The parser therefore calls the timestamp ``update``, not ``tick``.

Binary payloads and strings that may contain local file, registry or network
data are represented by length and SHA-256 only.  Mouse/key commands and the
small header fields needed for calibration remain directly inspectable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

DEMO_FILE_ID = 0x42BEEF78
SUPPORTED_DEMO_VERSIONS = frozenset((1, 2))
DMO_INDEX_SCHEMA = "zuma-rl.popcap-dmo-index"
DMO_INDEX_VERSION = 3

_COMMAND_NAMES = {
    0: "mouse_position",
    1: "activate_app",
    2: "size",
    3: "key_down",
    4: "key_up",
    5: "key_char",
    6: "close",
    7: "mouse_enter",
    8: "mouse_exit",
    9: "loading_complete",
    10: "registry_get_subkeys",
    11: "registry_read",
    12: "registry_write",
    13: "registry_erase",
    14: "file_exists",
    15: "file_read",
    16: "file_write",
    17: "http_result",
    18: "sync",
    19: "assert_string_equal",
    20: "assert_int_equal",
    21: "mouse_wheel",
    22: "handle_complete",
    23: "video_data",
    31: "idle",
}
_INPUT_KINDS = frozenset(
    (
        "mouse_move",
        "mouse_button",
        "mouse_wheel",
        "key_down",
        "key_up",
        "key_char",
        "mouse_enter",
        "mouse_exit",
    )
)


class PopCapDemoError(ValueError):
    """Raised when a DMO is malformed, truncated, or unsupported."""


def _sha256(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _payload_descriptor(data: bytes) -> Mapping[str, Any]:
    return MappingProxyType(
        {
            "bytes": len(data),
            "sha256": _sha256(data),
        }
    )


class _BitReader:
    """Least-significant-bit-first reader matching Sexy ``Buffer``."""

    __slots__ = ("_data", "bit_position")

    def __init__(self, data: bytes):
        self._data = data
        self.bit_position = 0

    @property
    def total_bits(self) -> int:
        return len(self._data) * 8

    @property
    def remaining_bits(self) -> int:
        return self.total_bits - self.bit_position

    def _require(self, bits: int, context: str) -> None:
        if bits < 0 or self.remaining_bits < bits:
            raise PopCapDemoError(
                f"truncated DMO while reading {context} at bit "
                f"{self.bit_position}"
            )

    def read_bits(
        self,
        bits: int,
        *,
        signed: bool = False,
        context: str = "bits",
    ) -> int:
        self._require(bits, context)
        value = 0
        for output_bit in range(bits):
            byte_index, bit_index = divmod(self.bit_position, 8)
            if self._data[byte_index] & (1 << bit_index):
                value |= 1 << output_bit
            self.bit_position += 1
        if signed and bits and value & (1 << (bits - 1)):
            value -= 1 << bits
        return value

    def read_u8(self, context: str) -> int:
        return self.read_bits(8, context=context)

    def read_u16(self, context: str) -> int:
        low = self.read_u8(context)
        high = self.read_u8(context)
        return low | (high << 8)

    def read_u32(self, context: str) -> int:
        return sum(self.read_u8(context) << shift for shift in (0, 8, 16, 24))

    def read_i32(self, context: str) -> int:
        value = self.read_u32(context)
        return value - (1 << 32) if value & (1 << 31) else value

    def read_bytes(self, count: int, context: str) -> bytes:
        if count < 0:
            raise PopCapDemoError(f"negative byte count for {context}")
        self._require(count * 8, context)
        return bytes(self.read_u8(context) for _ in range(count))

    def read_string(self, context: str) -> bytes:
        length = self.read_u16(f"{context} length")
        return self.read_bytes(length, context)

    def remaining_is_zero_padding(self) -> bool:
        position = self.bit_position
        while position < self.total_bits:
            byte_index, bit_index = divmod(position, 8)
            if self._data[byte_index] & (1 << bit_index):
                return False
            position += 1
        return True


@dataclass(frozen=True, slots=True)
class DemoMarker:
    """A named marker and its framework update count."""

    name: str
    update: int

    def __post_init__(self) -> None:
        if self.update < 0:
            raise PopCapDemoError("demo marker update must not be negative")


@dataclass(frozen=True, slots=True)
class DemoCommand:
    """One decoded command in original stream order."""

    update: int
    sequence: int
    kind: str
    payload: Mapping[str, Any]
    short_form: bool
    command_number: int

    @property
    def is_input(self) -> bool:
        return self.kind in _INPUT_KINDS

    def to_dict(
        self,
        *,
        include_sensitive_input: bool = False,
    ) -> dict[str, Any]:
        payload = dict(self.payload)
        if not include_sensitive_input:
            if self.kind in {"key_down", "key_up"}:
                payload.pop("key_code", None)
                payload["key_code_redacted"] = True
            elif self.kind == "key_char":
                payload.pop("codepoint", None)
                payload["codepoint_redacted"] = True
        return {
            "update": self.update,
            "sequence": self.sequence,
            "kind": self.kind,
            "payload": payload,
            "short_form": self.short_form,
            "command_number": self.command_number,
        }


@dataclass(frozen=True, slots=True)
class PopCapDemo:
    """Parsed DMO header, markers, and command stream."""

    version: int
    random_seed: int
    product_version: str
    length_updates: int
    markers: tuple[DemoMarker, ...]
    commands: tuple[DemoCommand, ...]
    artifact_bytes: int
    artifact_sha256: str

    @property
    def input_commands(self) -> tuple[DemoCommand, ...]:
        return tuple(command for command in self.commands if command.is_input)

    def to_dict(
        self,
        *,
        inputs_only: bool = False,
        include_sensitive_input: bool = False,
    ) -> dict[str, Any]:
        commands: Sequence[DemoCommand] = (
            self.input_commands if inputs_only else self.commands
        )
        return {
            "schema": DMO_INDEX_SCHEMA,
            "version": DMO_INDEX_VERSION,
            "dmo_version": self.version,
            "random_seed": self.random_seed,
            "product_version": self.product_version,
            "length_updates": self.length_updates,
            "artifact": {
                "bytes": self.artifact_bytes,
                "sha256": self.artifact_sha256,
            },
            "markers": [
                {
                    "name": {
                        "bytes": len(marker.name.encode("latin-1")),
                        "sha256": _sha256(
                            marker.name.encode("latin-1")
                        ),
                    },
                    "update": marker.update,
                }
                for marker in self.markers
            ],
            "commands": [
                command.to_dict(
                    include_sensitive_input=include_sensitive_input,
                )
                for command in commands
            ],
        }

    def to_json(
        self,
        *,
        inputs_only: bool = False,
        include_sensitive_input: bool = False,
        indent: int | None = 2,
    ) -> str:
        return json.dumps(
            self.to_dict(
                inputs_only=inputs_only,
                include_sensitive_input=include_sensitive_input,
            ),
            ensure_ascii=False,
            allow_nan=False,
            indent=indent,
            sort_keys=True,
            separators=(",", ":") if indent is None else None,
        )

    @classmethod
    def read(cls, path: str | Path) -> "PopCapDemo":
        source = Path(path)
        try:
            data = source.read_bytes()
        except OSError as error:
            raise PopCapDemoError(f"could not read DMO: {source}") from error
        return cls.from_bytes(data)

    @classmethod
    def from_bytes(cls, data: bytes) -> "PopCapDemo":
        if not isinstance(data, bytes):
            raise TypeError("DMO data must be bytes")
        # Even the smallest payload-free long command occupies ten bits and
        # therefore two bytes after Buffer padding.
        if len(data) < 20:
            raise PopCapDemoError("DMO is too short")

        offset = 0

        def unpack_u32(context: str) -> int:
            nonlocal offset
            if offset + 4 > len(data):
                raise PopCapDemoError(f"truncated DMO {context}")
            value = struct.unpack_from("<I", data, offset)[0]
            offset += 4
            return value

        file_id = unpack_u32("file id")
        if file_id != DEMO_FILE_ID:
            raise PopCapDemoError(
                f"invalid DMO file id: {file_id:#010x}"
            )
        version = unpack_u32("version")
        if version not in SUPPORTED_DEMO_VERSIONS:
            raise PopCapDemoError(f"unsupported DMO version: {version}")
        random_seed = unpack_u32("random seed")

        if offset + 2 > len(data):
            raise PopCapDemoError("truncated DMO product-version length")
        product_length = struct.unpack_from("<H", data, offset)[0]
        offset += 2
        if product_length > 255:
            raise PopCapDemoError(
                "DMO product-version string exceeds framework limit"
            )
        if offset + product_length > len(data):
            raise PopCapDemoError("truncated DMO product-version string")
        product_raw = data[offset : offset + product_length]
        offset += product_length
        try:
            product_version = product_raw.decode("latin-1")
        except UnicodeDecodeError as error:  # pragma: no cover - latin-1 total
            raise PopCapDemoError("invalid DMO product-version string") from error
        markers: tuple[DemoMarker, ...] = ()
        if version >= 2:
            marker_size = unpack_u32("marker-buffer size")
            # At least the four-byte demo length and one command byte must
            # remain, matching the original reader's strict inequality.
            if marker_size >= len(data) - offset:
                raise PopCapDemoError("invalid DMO marker-buffer size")
            marker_end = offset + marker_size
            if marker_end + 5 > len(data):
                raise PopCapDemoError("truncated DMO after marker buffer")
            markers = _parse_markers(data[offset:marker_end])
            offset = marker_end

        length_updates = unpack_u32("length")
        previous_marker_update = -1
        for marker in markers:
            if marker.update > length_updates:
                raise PopCapDemoError(
                    "DMO marker update exceeds declared demo length"
                )
            if marker.update < previous_marker_update:
                raise PopCapDemoError(
                    "DMO marker updates must be non-decreasing"
                )
            previous_marker_update = marker.update
        command_data = data[offset:]
        if not command_data:
            raise PopCapDemoError("DMO has no command stream")
        commands = _parse_commands(command_data, length_updates)
        return cls(
            version=version,
            random_seed=random_seed,
            product_version=product_version,
            length_updates=length_updates,
            markers=markers,
            commands=commands,
            artifact_bytes=len(data),
            artifact_sha256=_sha256(data),
        )


def _parse_markers(data: bytes) -> tuple[DemoMarker, ...]:
    reader = _BitReader(data)
    if reader.remaining_bits < 32:
        raise PopCapDemoError("DMO marker buffer is too short")
    count = reader.read_i32("marker count")
    if count < 0:
        raise PopCapDemoError("negative DMO marker count")
    markers: list[DemoMarker] = []
    for index in range(count):
        raw_name = reader.read_string(f"marker {index} name")
        name = raw_name.decode("latin-1")
        update = reader.read_i32(f"marker {index} update")
        markers.append(DemoMarker(name=name, update=update))
    if not reader.remaining_is_zero_padding():
        raise PopCapDemoError("non-zero trailing data in DMO marker buffer")
    return tuple(markers)


def _parse_commands(
    data: bytes,
    length_updates: int,
) -> tuple[DemoCommand, ...]:
    reader = _BitReader(data)
    update = 0
    sequence = 0
    mouse_x = 0
    mouse_y = 0
    commands: list[DemoCommand] = []

    while reader.remaining_bits:
        # Sexy Buffer rounds its final partial byte up and zero-initialises
        # unused bits.  A valid command needs at least six header bits.
        if (
            reader.remaining_bits <= 7
            and reader.remaining_is_zero_padding()
        ):
            break
        if reader.remaining_bits < 6:
            raise PopCapDemoError("non-zero truncated DMO command header")

        update += reader.read_bits(4, context="command timing")
        if update > length_updates:
            raise PopCapDemoError(
                "DMO command update exceeds declared demo length"
            )
        short_form = bool(
            reader.read_bits(1, context="command short-form flag")
        )
        command_number = reader.read_bits(
            1 if short_form else 5,
            context="command number",
        )
        payload: dict[str, Any]
        if short_form:
            if command_number == 0:
                delta_x = reader.read_bits(
                    6,
                    signed=True,
                    context="short mouse delta x",
                )
                delta_y = reader.read_bits(
                    6,
                    signed=True,
                    context="short mouse delta y",
                )
                mouse_x += delta_x
                mouse_y += delta_y
                kind = "mouse_move"
                payload = {
                    "x": mouse_x,
                    "y": mouse_y,
                    "delta_x": delta_x,
                    "delta_y": delta_y,
                }
            elif command_number == 1:
                down = bool(
                    reader.read_bits(1, context="mouse button state")
                )
                button = reader.read_bits(
                    3,
                    signed=True,
                    context="mouse button number",
                )
                kind = "mouse_button"
                payload = {
                    "x": mouse_x,
                    "y": mouse_y,
                    "button": button,
                    "down": down,
                }
            else:  # pragma: no cover - one-bit command
                raise PopCapDemoError(
                    f"unsupported short DMO command: {command_number}"
                )
        else:
            try:
                kind = _COMMAND_NAMES[command_number]
            except KeyError as error:
                raise PopCapDemoError(
                    f"unsupported DMO command number: {command_number}"
                ) from error
            payload = _read_long_command_payload(
                reader,
                command_number,
            )
            if command_number == 0:
                mouse_x = int(payload["x"])
                mouse_y = int(payload["y"])
                kind = "mouse_move"

        commands.append(
            DemoCommand(
                update=update,
                sequence=sequence,
                kind=kind,
                payload=MappingProxyType(payload),
                short_form=short_form,
                command_number=command_number,
            )
        )
        sequence += 1

    return tuple(commands)


def _read_success(reader: _BitReader, context: str) -> bool:
    return bool(reader.read_bits(1, context=f"{context} success"))


def _read_sized_blob(
    reader: _BitReader,
    context: str,
) -> Mapping[str, Any]:
    size = reader.read_i32(f"{context} size")
    if size < 0:
        raise PopCapDemoError(f"negative DMO {context} size")
    return _payload_descriptor(reader.read_bytes(size, context))


def _read_long_command_payload(
    reader: _BitReader,
    command: int,
) -> dict[str, Any]:
    if command == 0:
        return {
            "x": reader.read_bits(12, context="absolute mouse x"),
            "y": reader.read_bits(12, context="absolute mouse y"),
        }
    if command == 1:
        return {"active": bool(reader.read_bits(1, context="active state"))}
    if command == 2:
        return {"minimized": bool(reader.read_u8("minimized state"))}
    if command in (3, 4):
        return {"key_code": reader.read_u8("key code")}
    if command == 5:
        wide = bool(reader.read_bits(1, context="key-char width"))
        bits = 16 if wide else 8
        return {
            "codepoint": reader.read_bits(bits, context="key character"),
            "bits": bits,
        }
    if command in (6, 7, 8, 9, 31):
        return {}
    if command == 10:
        success = _read_success(reader, "registry_get_subkeys")
        payload: dict[str, Any] = {"success": success}
        if success:
            count = reader.read_i32("registry subkey count")
            if count < 0:
                raise PopCapDemoError("negative registry subkey count")
            encoded = bytearray()
            for index in range(count):
                value = reader.read_string(f"registry subkey {index}")
                encoded.extend(struct.pack("<H", len(value)))
                encoded.extend(value)
            payload.update(
                {
                    "count": count,
                    "subkeys": dict(_payload_descriptor(bytes(encoded))),
                }
            )
        return payload
    if command == 11:
        success = _read_success(reader, "registry_read")
        payload = {"success": success}
        if success:
            value_type = reader.read_i32("registry value type")
            blob = _read_sized_blob(reader, "registry value")
            payload.update(
                {
                    "value_type": value_type,
                    "value": dict(blob),
                }
            )
        return payload
    if command in (12, 13, 16):
        return {"success": _read_success(reader, _COMMAND_NAMES[command])}
    if command == 14:
        return {"exists": bool(reader.read_bits(1, context="file exists"))}
    if command == 15:
        success = _read_success(reader, "file_read")
        payload = {"success": success}
        if success:
            payload["data"] = dict(_read_sized_blob(reader, "file data"))
        return payload
    if command == 17:
        transfer_id = reader.read_i32("HTTP transfer id")
        result = reader.read_u8("HTTP result")
        content = reader.read_string("HTTP content")
        return {
            "transfer_id": transfer_id,
            "result": result,
            "content": dict(_payload_descriptor(content)),
        }
    if command == 18:
        return {"data": dict(_read_sized_blob(reader, "sync data"))}
    if command == 19:
        value = reader.read_string("assert string")
        return {"value": dict(_payload_descriptor(value))}
    if command in (20, 22):
        key = "value" if command == 20 else "handle"
        return {key: reader.read_i32(key)}
    if command == 21:
        return {
            "delta": reader.read_bits(
                8,
                signed=True,
                context="mouse-wheel delta",
            )
        }
    if command == 23:
        return {
            "windowed": bool(reader.read_u8("video windowed state")),
            "refresh_rate": reader.read_u8("video refresh rate"),
        }
    raise PopCapDemoError(f"unsupported DMO command number: {command}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit a PopCap DMO without exposing embedded local data.",
    )
    parser.add_argument("path", type=Path)
    parser.add_argument(
        "--inputs-only",
        action="store_true",
        help="Only emit mouse/key commands; the artifact hash still binds all data.",
    )
    parser.add_argument(
        "--include-sensitive-input",
        action="store_true",
        help=(
            "Include raw keyboard key codes/characters in JSON. By default "
            "they are redacted because they can reconstruct typed text."
        ),
    )
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        demo = PopCapDemo.read(args.path)
    except (OSError, PopCapDemoError):
        print(
            "error: DMO could not be read or parsed",
            file=sys.stderr,
        )
        return 1
    if args.json:
        print(
            demo.to_json(
                inputs_only=args.inputs_only,
                include_sensitive_input=args.include_sensitive_input,
            )
        )
        return 0
    print(f"artifact         : {demo.artifact_sha256}")
    print(f"bytes            : {demo.artifact_bytes:,}")
    print(f"DMO version      : {demo.version}")
    print(f"product version  : {demo.product_version}")
    print(f"random seed      : {demo.random_seed}")
    print(f"length updates   : {demo.length_updates:,}")
    print(f"markers          : {len(demo.markers):,}")
    print(f"commands         : {len(demo.commands):,}")
    print(f"input commands   : {len(demo.input_commands):,}")
    return 0


__all__ = [
    "DEMO_FILE_ID",
    "DMO_INDEX_SCHEMA",
    "DMO_INDEX_VERSION",
    "DemoCommand",
    "DemoMarker",
    "PopCapDemo",
    "PopCapDemoError",
    "SUPPORTED_DEMO_VERSIONS",
]


if __name__ == "__main__":
    raise SystemExit(main())
