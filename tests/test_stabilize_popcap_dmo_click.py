from __future__ import annotations

import struct

import pytest

from tools.stabilize_popcap_dmo_click import stabilize_click
from zuma_rl import popcap_dmo


class _Bits:
    def __init__(self) -> None:
        self.data = bytearray()
        self.position = 0

    def write(self, value: int, width: int) -> None:
        encoded = value & ((1 << width) - 1)
        for bit in range(width):
            if self.position % 8 == 0:
                self.data.append(0)
            if encoded & (1 << bit):
                self.data[self.position // 8] |= 1 << (
                    self.position % 8
                )
            self.position += 1


def _command(
    writer: _Bits,
    delta: int,
    number: int,
    *,
    short: bool = False,
) -> None:
    writer.write(delta, 4)
    writer.write(int(short), 1)
    writer.write(number, 1 if short else 5)


def _demo(*, restore_delta_x: int = 0) -> bytes:
    commands = _Bits()
    _command(commands, 0, 0)
    commands.write(100, 12)
    commands.write(100, 12)
    _command(commands, 1, 1, short=True)
    commands.write(1, 1)
    commands.write(1, 3)
    _command(commands, 1, 0, short=True)
    commands.write(2, 6)
    commands.write(3, 6)
    _command(commands, 1, 0, short=True)
    commands.write(-1, 6)
    commands.write(1, 6)
    _command(commands, 1, 1, short=True)
    commands.write(0, 1)
    commands.write(1, 3)
    _command(commands, 1, 31)
    _command(commands, 1, 0, short=True)
    commands.write(restore_delta_x, 6)
    commands.write(0, 6)
    _command(commands, 1, 31)
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
            struct.pack("<I", 10),
            bytes(commands.data),
        )
    )


def test_stabilize_click_restores_original_mouse_trajectory() -> None:
    source = _demo()
    before = popcap_dmo.PopCapDemo.from_bytes(source)
    output, provenance = stabilize_click(
        source,
        popcap_dmo,
        down_index=1,
        up_index=4,
        restore_index=6,
        expected_down_update=1,
        expected_up_update=4,
        expected_restore_update=6,
        expected_down_x=100,
        expected_down_y=100,
        expected_up_x=101,
        expected_up_y=104,
        expected_restore_x=101,
        expected_restore_y=104,
    )
    after = popcap_dmo.PopCapDemo.from_bytes(output)

    assert len(output) == len(source)
    assert dict(after.commands[2].payload) == {
        "x": 100,
        "y": 100,
        "delta_x": 0,
        "delta_y": 0,
    }
    assert dict(after.commands[3].payload) == {
        "x": 100,
        "y": 100,
        "delta_x": 0,
        "delta_y": 0,
    }
    assert dict(after.commands[4].payload) == {
        "x": 100,
        "y": 100,
        "button": 1,
        "down": False,
    }
    assert dict(after.commands[6].payload) == {
        "x": 101,
        "y": 104,
        "delta_x": 1,
        "delta_y": 4,
    }
    assert after.commands[7:] == before.commands[7:]
    assert provenance["transformation"][
        "post_restoration_trajectory_preserved"
    ] is True


def test_stabilize_click_rejects_unencodable_restoration() -> None:
    source = _demo(restore_delta_x=31)
    before = popcap_dmo.PopCapDemo.from_bytes(source)

    with pytest.raises(ValueError, match="restoration delta"):
        stabilize_click(
            source,
            popcap_dmo,
            down_index=1,
            up_index=4,
            restore_index=6,
            expected_down_update=1,
            expected_up_update=4,
            expected_restore_update=6,
            expected_down_x=100,
            expected_down_y=100,
            expected_up_x=101,
            expected_up_y=104,
            expected_restore_x=int(before.commands[6].payload["x"]),
            expected_restore_y=int(before.commands[6].payload["y"]),
        )
