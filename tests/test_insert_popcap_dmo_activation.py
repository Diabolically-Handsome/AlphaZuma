from __future__ import annotations

import struct

import pytest

from tools.insert_popcap_dmo_activation import insert_activation
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
    _command(commands, 0, 16)
    commands.write(0, 1)
    _command(commands, 0, 0)
    commands.write(400, 12)
    commands.write(300, 12)
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
            struct.pack("<I", 20),
            bytes(commands.data),
        )
    )


def test_insert_activation_preserves_existing_commands_and_timeline() -> None:
    source = _demo()
    before = popcap_dmo.PopCapDemo.from_bytes(source)
    output, provenance = insert_activation(
        source,
        popcap_dmo,
        anchor_index=0,
        expected_anchor_kind="file_write",
        expected_anchor_update=0,
        expected_anchor_start_bit=0,
        expected_anchor_end_bit=11,
    )
    after = popcap_dmo.PopCapDemo.from_bytes(output)

    assert [
        (command.update, command.kind, dict(command.payload))
        for command in after.commands
    ] == [
        (0, "file_write", {"success": False}),
        (0, "activate_app", {"active": True}),
        (0, "mouse_move", {"x": 400, "y": 300}),
        (15, "idle", {}),
    ]
    assert after.commands[0] == before.commands[0]
    assert [
        (
            command.update,
            command.kind,
            dict(command.payload),
            command.short_form,
            command.command_number,
        )
        for command in after.commands[2:]
    ] == [
        (
            command.update,
            command.kind,
            dict(command.payload),
            command.short_form,
            command.command_number,
        )
        for command in before.commands[1:]
    ]
    assert [command.sequence for command in after.commands] == [0, 1, 2, 3]
    assert provenance["transformation"]["timeline_preserved"] is True


def test_insert_activation_rejects_unbound_anchor() -> None:
    with pytest.raises(ValueError, match="anchor 0 kind mismatch"):
        insert_activation(
            _demo(),
            popcap_dmo,
            anchor_index=0,
            expected_anchor_kind="idle",
            expected_anchor_update=0,
            expected_anchor_start_bit=0,
            expected_anchor_end_bit=11,
        )
