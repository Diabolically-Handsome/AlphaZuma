from __future__ import annotations

import struct

import pytest

from tools.delay_popcap_dmo_suffix import delay_suffix
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
    _command(commands, 15, 16)
    commands.write(0, 1)
    _command(commands, 0, 16)
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
            struct.pack("<I", 45),
            bytes(commands.data),
        )
    )


def test_delay_suffix_inserts_idle_and_preserves_payloads() -> None:
    source = _demo()
    output, provenance = delay_suffix(
        source,
        popcap_dmo,
        anchor_index=0,
        expected_anchor_kind="idle",
        expected_anchor_update=15,
        expected_anchor_start_bit=0,
        expected_anchor_end_bit=10,
        target_index=1,
        expected_target_kind="file_write",
        expected_target_update=30,
        expected_target_start_bit=10,
        expected_target_end_bit=21,
        delay_updates=4,
    )
    before = popcap_dmo.PopCapDemo.from_bytes(source)
    after = popcap_dmo.PopCapDemo.from_bytes(output)

    assert after.length_updates == 49
    assert [
        (command.update, command.kind, dict(command.payload))
        for command in after.commands
    ] == [
        (15, "idle", {}),
        (30, "idle", {}),
        (34, "file_write", {"success": False}),
        (34, "file_write", {"success": False}),
        (49, "idle", {}),
    ]
    assert [
        (command.kind, dict(command.payload))
        for command in after.commands[2:]
    ] == [
        (command.kind, dict(command.payload))
        for command in before.commands[1:]
    ]
    assert provenance["transformation"]["delay_updates"] == 4
    assert (
        provenance["transformation"]["source_payload_bits_preserved"]
        is True
    )


def test_delay_suffix_rejects_unbound_target() -> None:
    with pytest.raises(ValueError, match="target kind mismatch"):
        delay_suffix(
            _demo(),
            popcap_dmo,
            anchor_index=0,
            expected_anchor_kind="idle",
            expected_anchor_update=15,
            expected_anchor_start_bit=0,
            expected_anchor_end_bit=10,
            target_index=1,
            expected_target_kind="idle",
            expected_target_update=30,
            expected_target_start_bit=10,
            expected_target_end_bit=21,
            delay_updates=4,
        )
