"""Synthetic binary tests for the read-only PopCap DMO parser."""

from __future__ import annotations

import hashlib
import struct

import pytest

from zuma_rl.popcap_dmo import (
    DEMO_FILE_ID,
    PopCapDemo,
    PopCapDemoError,
    main,
)


class _BitWriter:
    def __init__(self) -> None:
        self.data = bytearray()
        self.bit_position = 0

    def bits(self, value: int, count: int) -> None:
        masked = value & ((1 << count) - 1)
        for index in range(count):
            if self.bit_position % 8 == 0:
                self.data.append(0)
            if masked & (1 << index):
                self.data[self.bit_position // 8] |= (
                    1 << (self.bit_position % 8)
                )
            self.bit_position += 1

    def byte(self, value: int) -> None:
        self.bits(value, 8)

    def i32(self, value: int) -> None:
        for shift in (0, 8, 16, 24):
            self.byte((value >> shift) & 0xFF)

    def string(self, value: bytes) -> None:
        self.byte(len(value) & 0xFF)
        self.byte((len(value) >> 8) & 0xFF)
        for byte in value:
            self.byte(byte)


def _command(
    writer: _BitWriter,
    *,
    delta: int,
    number: int,
    short: bool = False,
) -> None:
    writer.bits(delta, 4)
    writer.bits(int(short), 1)
    writer.bits(number, 1 if short else 5)


def _marker_buffer(
    name: bytes = b"first-shot",
    *,
    update: int = 12,
) -> bytes:
    writer = _BitWriter()
    writer.i32(1)
    writer.string(name)
    writer.i32(update)
    return bytes(writer.data)


def _demo_bytes(
    commands: bytes,
    *,
    length: int = 50,
    product: bytes = b"1.0.4.9496",
    marker_name: bytes = b"first-shot",
    marker_update: int = 12,
) -> bytes:
    markers = _marker_buffer(marker_name, update=marker_update)
    return b"".join(
        (
            struct.pack("<IIIH", DEMO_FILE_ID, 2, 0x12345678, len(product)),
            product,
            struct.pack("<I", len(markers)),
            markers,
            struct.pack("<I", length),
            commands,
        )
    )


def test_header_markers_and_ordered_input_commands_decode() -> None:
    writer = _BitWriter()

    _command(writer, delta=0, number=23)
    writer.byte(0)
    writer.byte(164)

    _command(writer, delta=3, number=0)
    writer.bits(400, 12)
    writer.bits(300, 12)

    _command(writer, delta=0, number=0, short=True)
    writer.bits(5, 6)
    writer.bits(-3, 6)

    _command(writer, delta=1, number=1, short=True)
    writer.bits(1, 1)
    writer.bits(-1, 3)

    _command(writer, delta=0, number=1, short=True)
    writer.bits(0, 1)
    writer.bits(-1, 3)

    _command(writer, delta=2, number=3)
    writer.bits(0x20, 8)
    _command(writer, delta=0, number=4)
    writer.bits(0x20, 8)

    raw = _demo_bytes(bytes(writer.data))
    demo = PopCapDemo.from_bytes(raw)

    assert demo.version == 2
    assert demo.random_seed == 0x12345678
    assert demo.product_version == "1.0.4.9496"
    assert demo.length_updates == 50
    assert demo.markers[0].name == "first-shot"
    assert demo.markers[0].update == 12
    assert demo.artifact_sha256 == f"sha256:{hashlib.sha256(raw).hexdigest()}"

    assert [command.kind for command in demo.input_commands] == [
        "mouse_move",
        "mouse_move",
        "mouse_button",
        "mouse_button",
        "key_down",
        "key_up",
    ]
    assert [command.update for command in demo.input_commands] == [
        3,
        3,
        4,
        4,
        6,
        6,
    ]
    assert dict(demo.input_commands[1].payload) == {
        "x": 405,
        "y": 297,
        "delta_x": 5,
        "delta_y": -3,
    }
    assert dict(demo.input_commands[2].payload) == {
        "x": 405,
        "y": 297,
        "button": -1,
        "down": True,
    }
    assert demo.commands[0].kind == "video_data"
    assert dict(demo.commands[0].payload) == {
        "windowed": False,
        "refresh_rate": 164,
    }


def test_idle_blocks_accumulate_framework_updates() -> None:
    writer = _BitWriter()
    _command(writer, delta=15, number=31)
    _command(writer, delta=15, number=31)
    _command(writer, delta=2, number=6)
    demo = PopCapDemo.from_bytes(_demo_bytes(bytes(writer.data), length=32))

    assert [command.update for command in demo.commands] == [15, 30, 32]
    assert [command.kind for command in demo.commands] == [
        "idle",
        "idle",
        "close",
    ]


def test_embedded_payloads_are_hashed_not_exposed() -> None:
    writer = _BitWriter()
    _command(writer, delta=0, number=18)
    writer.i32(6)
    for value in b"secret":
        writer.byte(value)
    _command(writer, delta=0, number=19)
    writer.string(b"private-local-value")

    demo = PopCapDemo.from_bytes(_demo_bytes(bytes(writer.data)))
    sync = dict(demo.commands[0].payload["data"])
    asserted = dict(demo.commands[1].payload["value"])

    assert sync == {
        "bytes": 6,
        "sha256": f"sha256:{hashlib.sha256(b'secret').hexdigest()}",
    }
    assert asserted["bytes"] == len(b"private-local-value")
    serialized = demo.to_json()
    assert "secret" not in serialized
    assert "private-local-value" not in serialized


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda raw: b"bad!" + raw[4:], "file id"),
        (
            lambda raw: raw[:4] + struct.pack("<I", 99) + raw[8:],
            "unsupported DMO version",
        ),
        (
            lambda raw: raw[:-1],
            "truncated",
        ),
    ],
)
def test_malformed_dmo_is_rejected(mutation, message: str) -> None:
    writer = _BitWriter()
    _command(writer, delta=1, number=0)
    writer.bits(10, 12)
    writer.bits(20, 12)
    raw = _demo_bytes(bytes(writer.data))

    with pytest.raises(PopCapDemoError, match=message):
        PopCapDemo.from_bytes(mutation(raw))


def test_command_cannot_exceed_declared_length() -> None:
    writer = _BitWriter()
    _command(writer, delta=5, number=6)
    with pytest.raises(PopCapDemoError, match="exceeds declared"):
        PopCapDemo.from_bytes(
            _demo_bytes(
                bytes(writer.data),
                length=4,
                marker_update=0,
            )
        )


def test_marker_cannot_exceed_declared_length() -> None:
    writer = _BitWriter()
    _command(writer, delta=0, number=6)
    with pytest.raises(PopCapDemoError, match="marker update exceeds"):
        PopCapDemo.from_bytes(
            _demo_bytes(
                bytes(writer.data),
                length=4,
                marker_update=5,
            )
        )


def test_unknown_long_command_is_rejected() -> None:
    writer = _BitWriter()
    _command(writer, delta=0, number=24)
    with pytest.raises(PopCapDemoError, match="unsupported DMO command"):
        PopCapDemo.from_bytes(_demo_bytes(bytes(writer.data)))


def test_framework_permits_empty_product_version_and_marker_name() -> None:
    writer = _BitWriter()
    _command(writer, delta=0, number=6)

    demo = PopCapDemo.from_bytes(
        _demo_bytes(
            bytes(writer.data),
            product=b"",
            marker_name=b"",
        )
    )

    assert demo.product_version == ""
    assert demo.markers[0].name == ""


def test_minimal_v1_with_empty_product_version_is_accepted() -> None:
    writer = _BitWriter()
    _command(writer, delta=0, number=6)
    raw = (
        struct.pack("<IIIH", DEMO_FILE_ID, 1, 7, 0)
        + struct.pack("<I", 0)
        + bytes(writer.data)
    )

    assert len(raw) == 20
    demo = PopCapDemo.from_bytes(raw)
    assert demo.version == 1
    assert demo.product_version == ""
    assert demo.commands[0].kind == "close"


def test_all_long_framework_command_payloads_are_consumed_exactly() -> None:
    writer = _BitWriter()

    _command(writer, delta=0, number=1)
    writer.bits(1, 1)
    _command(writer, delta=0, number=2)
    writer.byte(1)
    _command(writer, delta=0, number=5)
    writer.bits(1, 1)
    writer.bits(0x20AC, 16)
    for number in (7, 8, 9):
        _command(writer, delta=0, number=number)

    _command(writer, delta=0, number=10)
    writer.bits(1, 1)
    writer.i32(2)
    writer.string(b"one")
    writer.string(b"two")

    _command(writer, delta=0, number=11)
    writer.bits(1, 1)
    writer.i32(3)
    writer.i32(4)
    for value in b"rval":
        writer.byte(value)

    for number in (12, 13):
        _command(writer, delta=0, number=number)
        writer.bits(1, 1)

    _command(writer, delta=0, number=14)
    writer.bits(1, 1)

    _command(writer, delta=0, number=15)
    writer.bits(1, 1)
    writer.i32(3)
    for value in b"xyz":
        writer.byte(value)

    _command(writer, delta=0, number=16)
    writer.bits(0, 1)

    _command(writer, delta=0, number=17)
    writer.i32(7)
    writer.byte(2)
    writer.string(b"http-body")

    _command(writer, delta=0, number=20)
    writer.i32(-123)
    _command(writer, delta=0, number=21)
    writer.bits(-7, 8)
    _command(writer, delta=0, number=22)
    writer.i32(42)

    demo = PopCapDemo.from_bytes(_demo_bytes(bytes(writer.data)))
    commands = {command.command_number: command for command in demo.commands}

    assert dict(commands[1].payload) == {"active": True}
    assert dict(commands[2].payload) == {"minimized": True}
    assert dict(commands[5].payload) == {"codepoint": 0x20AC, "bits": 16}
    assert [commands[number].kind for number in (7, 8, 9)] == [
        "mouse_enter",
        "mouse_exit",
        "loading_complete",
    ]
    registry_subkeys = dict(commands[10].payload)
    assert registry_subkeys["success"] is True
    assert registry_subkeys["count"] == 2
    assert registry_subkeys["subkeys"]["bytes"] == 10
    registry_value = dict(commands[11].payload)
    assert registry_value["value_type"] == 3
    assert registry_value["value"]["bytes"] == 4
    assert dict(commands[12].payload) == {"success": True}
    assert dict(commands[13].payload) == {"success": True}
    assert dict(commands[14].payload) == {"exists": True}
    assert commands[15].payload["data"]["bytes"] == 3
    assert dict(commands[16].payload) == {"success": False}
    assert commands[17].payload["transfer_id"] == 7
    assert commands[17].payload["result"] == 2
    assert commands[17].payload["content"]["bytes"] == 9
    assert dict(commands[20].payload) == {"value": -123}
    assert dict(commands[21].payload) == {"delta": -7}
    assert dict(commands[22].payload) == {"handle": 42}

    serialized = demo.to_json()
    for secret in ("one", "two", "rval", "xyz", "http-body"):
        assert secret not in serialized


def test_json_redacts_marker_names_and_all_keyboard_values_by_default() -> None:
    writer = _BitWriter()
    _command(writer, delta=0, number=3)
    writer.byte(ord("A"))
    _command(writer, delta=0, number=4)
    writer.byte(ord("A"))
    _command(writer, delta=0, number=5)
    writer.bits(0, 1)
    writer.bits(ord("Q"), 8)

    demo = PopCapDemo.from_bytes(
        _demo_bytes(
            bytes(writer.data),
            marker_name=b"PRIVATE_MARKER",
        )
    )
    serialized = demo.to_json(inputs_only=True)

    assert "PRIVATE_MARKER" not in serialized
    assert '"key_code":' not in serialized
    assert serialized.count('"key_code_redacted": true') == 2
    assert '"codepoint":' not in serialized
    assert '"codepoint_redacted": true' in serialized

    explicit = demo.to_json(
        inputs_only=True,
        include_sensitive_input=True,
    )
    assert '"key_code": 65' in explicit
    assert '"codepoint": 81' in explicit


def test_cli_parse_error_is_safe_and_has_no_traceback(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["/definitely/not/a/demo.dmo"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "error: DMO could not be read or parsed\n"
    assert "Traceback" not in captured.err
