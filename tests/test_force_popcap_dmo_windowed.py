from __future__ import annotations

import struct

import pytest

from tools import force_popcap_dmo_windowed
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
                self.data[self.position // 8] |= 1 << (self.position % 8)
            self.position += 1


def _demo(*, windowed: bool) -> bytes:
    commands = _Bits()
    commands.write(0, 4)
    commands.write(0, 1)
    commands.write(23, 5)
    commands.write(int(windowed), 8)
    commands.write(100, 8)
    product = b"1.0.4.9496"
    markers = struct.pack("<i", 0)
    return b"".join(
        (
            struct.pack(
                "<IIIH",
                popcap_dmo.DEMO_FILE_ID,
                2,
                123,
                len(product),
            ),
            product,
            struct.pack("<I", len(markers)),
            markers,
            struct.pack("<I", 1),
            bytes(commands.data),
        )
    )


def test_force_windowed_changes_only_video_payload() -> None:
    source = _demo(windowed=False)
    output, details = force_popcap_dmo_windowed.force_windowed(
        source,
        popcap_dmo,
    )

    before = popcap_dmo.PopCapDemo.from_bytes(source)
    after = popcap_dmo.PopCapDemo.from_bytes(output)
    assert len(output) == len(source)
    assert before.commands[0].payload == {
        "windowed": False,
        "refresh_rate": 100,
    }
    assert after.commands[0].payload == {
        "windowed": True,
        "refresh_rate": 100,
    }
    assert details["transformed_command_indices"] == [0]


def test_force_windowed_rejects_already_windowed_demo() -> None:
    with pytest.raises(
        ValueError,
        match="no video_data command to force windowed",
    ):
        force_popcap_dmo_windowed.force_windowed(
            _demo(windowed=True),
            popcap_dmo,
        )


def test_force_fullscreen_changes_only_video_payload() -> None:
    source = _demo(windowed=True)
    output, details = force_popcap_dmo_windowed.force_fullscreen(
        source,
        popcap_dmo,
    )

    before = popcap_dmo.PopCapDemo.from_bytes(source)
    after = popcap_dmo.PopCapDemo.from_bytes(output)
    assert len(output) == len(source)
    assert before.commands[0].payload == {
        "windowed": True,
        "refresh_rate": 100,
    }
    assert after.commands[0].payload == {
        "windowed": False,
        "refresh_rate": 100,
    }
    assert details["transformed_command_indices"] == [0]
    assert details["target_windowed"] is False


def test_force_fullscreen_rejects_already_fullscreen_demo() -> None:
    with pytest.raises(
        ValueError,
        match="no video_data command to force fullscreen",
    ):
        force_popcap_dmo_windowed.force_fullscreen(
            _demo(windowed=False),
            popcap_dmo,
        )
