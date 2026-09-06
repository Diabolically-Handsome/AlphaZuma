from __future__ import annotations

import struct

import pytest

from tools import transform_popcap_dmo_mouse_space
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


def _mouse_demo() -> bytes:
    commands = _Bits()
    _command(commands, 0, 0)
    commands.write(960, 12)
    commands.write(540, 12)
    _command(commands, 1, 0, short=True)
    commands.write(30, 6)
    commands.write(0, 6)
    _command(commands, 1, 1, short=True)
    commands.write(1, 1)
    commands.write(1, 3)
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
            struct.pack("<I", 2),
            bytes(commands.data),
        )
    )


def test_viewport_coordinate_scales_and_clamps_pillarbox() -> None:
    scale = transform_popcap_dmo_mouse_space._scale_viewport_coordinate

    assert scale(
        240,
        1920,
        800,
        viewport_origin=240,
        viewport_size=1440,
    ) == 0
    assert scale(
        960,
        1920,
        800,
        viewport_origin=240,
        viewport_size=1440,
    ) == 400
    assert scale(
        1679,
        1920,
        800,
        viewport_origin=240,
        viewport_size=1440,
    ) == 799
    assert scale(
        100,
        1920,
        800,
        viewport_origin=240,
        viewport_size=1440,
    ) == 0
    assert scale(
        1800,
        1920,
        800,
        viewport_origin=240,
        viewport_size=1440,
    ) == 799


def test_viewport_coordinate_rejects_invalid_viewport() -> None:
    with pytest.raises(ValueError, match="extends beyond"):
        transform_popcap_dmo_mouse_space._scale_viewport_coordinate(
            0,
            1920,
            800,
            viewport_origin=500,
            viewport_size=1500,
        )


def test_remap_mouse_space_uses_viewport_for_moves_and_buttons() -> None:
    source = _mouse_demo()
    output, provenance = (
        transform_popcap_dmo_mouse_space.remap_mouse_space(
            source,
            popcap_dmo,
            source_width=1920,
            source_height=1080,
            target_width=800,
            target_height=600,
            source_viewport_left=240,
            source_viewport_top=0,
            source_viewport_width=1440,
            source_viewport_height=1080,
        )
    )

    commands = popcap_dmo.PopCapDemo.from_bytes(output).commands
    assert dict(commands[0].payload) == {"x": 400, "y": 300}
    assert dict(commands[1].payload) == {
        "x": 416,
        "y": 300,
        "delta_x": 16,
        "delta_y": 0,
    }
    assert dict(commands[2].payload) == {
        "x": 416,
        "y": 300,
        "button": 1,
        "down": True,
    }
    assert provenance["source"]["game_viewport"] == [
        240,
        0,
        1440,
        1080,
    ]
    assert provenance["transformation"]["clamped_mouse_commands"] == 0
