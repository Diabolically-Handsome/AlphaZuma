from __future__ import annotations

import struct

import pytest

from zuma_rl.popcap_dmo import DEMO_FILE_ID, PopCapDemo
from tools.inject_popcap_dmo_click_burst import inject_click_burst


class _Writer:
    def __init__(self) -> None:
        self.data = bytearray()
        self.position = 0

    def bits(self, value: int, width: int) -> None:
        encoded = value & ((1 << width) - 1)
        for offset in range(width):
            if self.position % 8 == 0:
                self.data.append(0)
            if encoded & (1 << offset):
                self.data[self.position // 8] |= 1 << (self.position % 8)
            self.position += 1


def _command(
    writer: _Writer,
    delta: int,
    number: int,
    *,
    short: bool = False,
) -> None:
    writer.bits(delta, 4)
    writer.bits(int(short), 1)
    writer.bits(number, 1 if short else 5)


def _source_demo() -> bytes:
    writer = _Writer()
    _command(writer, 15, 31)
    _command(writer, 5, 0)
    writer.bits(400, 12)
    writer.bits(300, 12)
    _command(writer, 1, 1, short=True)
    writer.bits(1, 1)
    writer.bits(1, 3)
    _command(writer, 1, 1, short=True)
    writer.bits(0, 1)
    writer.bits(1, 3)
    _command(writer, 15, 31)
    _command(writer, 3, 6)
    product = b"1.0.4.9496"
    return b"".join(
        (
            struct.pack(
                "<IIIH",
                DEMO_FILE_ID,
                1,
                0x12345678,
                len(product),
            ),
            product,
            struct.pack("<I", 40),
            bytes(writer.data),
        )
    )


def test_click_burst_preserves_source_commands_and_inserts_edges() -> None:
    import zuma_rl.popcap_dmo as module

    source = _source_demo()
    original = PopCapDemo.from_bytes(source)
    output, provenance = inject_click_burst(
        source,
        module,
        pairs=((24, 25), (27, 28)),
    )
    transformed = PopCapDemo.from_bytes(output)

    assert transformed.random_seed == original.random_seed
    assert transformed.length_updates == original.length_updates
    assert len(transformed.commands) == len(original.commands) + 4
    inserted = [
        command
        for command in transformed.commands
        if (
            command.kind == "mouse_button"
            and 24 <= command.update <= 28
        )
    ]
    assert [
        (command.update, command.payload["down"])
        for command in inserted
    ] == [(24, True), (25, False), (27, True), (28, False)]
    assert all(command.payload["button"] == 1 for command in inserted)
    assert all(
        (command.payload["x"], command.payload["y"]) == (400, 300)
        for command in inserted
    )
    assert provenance["source"]["commands"] == len(original.commands)
    assert provenance["output"]["commands"] == len(transformed.commands)
    assert provenance["classification"] == (
        "diagnostic-not-unmodified-pc-evidence"
    )


def test_click_burst_rejects_overlap_with_source_button_edge() -> None:
    import zuma_rl.popcap_dmo as module

    with pytest.raises(
        ValueError,
        match="source left-button edge overlaps injection",
    ):
        inject_click_burst(
            _source_demo(),
            module,
            pairs=((21, 23),),
        )
