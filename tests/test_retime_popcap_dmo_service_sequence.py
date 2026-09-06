from __future__ import annotations

import hashlib
import json
import struct

import pytest

from tools.retime_popcap_dmo_service_sequence import (
    retime_service_sequence,
)
from zuma_rl import popcap_dmo


class _Bits:
    def __init__(self) -> None:
        self.data = bytearray()
        self.position = 0

    def write(self, value: int, width: int) -> None:
        for bit in range(width):
            if self.position % 8 == 0:
                self.data.append(0)
            if value & (1 << bit):
                self.data[self.position // 8] |= 1 << (
                    self.position % 8
                )
            self.position += 1


def _command(writer: _Bits, delta: int, number: int) -> None:
    writer.write(delta, 4)
    writer.write(0, 1)
    writer.write(number, 5)


def _demo() -> bytes:
    commands = _Bits()
    _command(commands, 15, 31)
    _command(commands, 4, 16)
    commands.write(0, 1)
    _command(commands, 0, 16)
    commands.write(0, 1)
    _command(commands, 1, 16)
    commands.write(0, 1)
    _command(commands, 15, 31)
    product = b"1.0.4.9496"
    return b"".join(
        (
            struct.pack(
                "<IIIH",
                popcap_dmo.DEMO_FILE_ID,
                1,
                123,
                len(product),
            ),
            product,
            struct.pack("<I", 35),
            bytes(commands.data),
        )
    )


def _false_payload_sha256() -> str:
    payload = json.dumps(
        {"success": False},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def test_retime_service_sequence_preserves_payload_and_later_update() -> None:
    output, provenance = retime_service_sequence(
        _demo(),
        start_index=1,
        end_index=3,
        expected_updates=(19, 19, 20),
        new_updates=(19, 20, 22),
        expected_start_bit=10,
        expected_end_bit=43,
        expected_kind="file_write",
        expected_payload_sha256=_false_payload_sha256(),
    )
    parsed = popcap_dmo.PopCapDemo.from_bytes(output)

    assert [
        (command.update, command.kind, dict(command.payload))
        for command in parsed.commands
    ] == [
        (15, "idle", {}),
        (19, "file_write", {"success": False}),
        (20, "file_write", {"success": False}),
        (22, "file_write", {"success": False}),
        (35, "idle", {}),
    ]
    assert provenance["transformation"]["later_timeline_preserved"] is True


def test_retime_service_sequence_rejects_update_pattern() -> None:
    with pytest.raises(ValueError, match="update pattern mismatch"):
        retime_service_sequence(
            _demo(),
            start_index=1,
            end_index=3,
            expected_updates=(19, 20, 20),
            new_updates=(19, 20, 22),
            expected_start_bit=10,
            expected_end_bit=43,
            expected_kind="file_write",
            expected_payload_sha256=_false_payload_sha256(),
        )
