"""Tests for provenance-bound diagnostic DMO command removal."""

from __future__ import annotations

import copy
import hashlib
import struct

import pytest

from tools.align_popcap_dmo_startup import (
    _command_stream_offset,
    _load_popcap_dmo_module,
    _scan_commands,
)
from tools.insert_popcap_dmo_failed_file_writes import (
    insert_failed_file_writes,
    validate_successful_file_write_padding_provenance,
)
from tools.remove_popcap_dmo_command import remove_command
from tools.relocate_popcap_dmo_service_block import (
    relocate_service_block,
)
from tools.retime_popcap_dmo_service_block import retime_service_block
from tools.replace_popcap_dmo_command_with_idle import (
    replace_command_with_idle,
)
from tools.replace_popcap_dmo_service_sequence_with_idle import (
    replace_service_sequence_with_idle,
)
from zuma_rl.popcap_dmo import DEMO_FILE_ID, PopCapDemo


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


def _command(writer: _BitWriter, *, delta: int, number: int) -> None:
    writer.bits(delta, 4)
    writer.bits(0, 1)
    writer.bits(number, 5)


def _demo_bytes(commands: bytes, *, length: int = 50) -> bytes:
    product = b"1.0.4.9496"
    empty_markers = struct.pack("<I", 0)
    return b"".join(
        (
            struct.pack(
                "<IIIH",
                DEMO_FILE_ID,
                2,
                0x12345678,
                len(product),
            ),
            product,
            struct.pack("<I", len(empty_markers)),
            empty_markers,
            struct.pack("<I", length),
            commands,
        )
    )


def _service_result_demo(*, following_delta: int = 0) -> tuple[bytes, bytes]:
    writer = _BitWriter()
    _command(writer, delta=15, number=31)
    _command(writer, delta=15, number=31)

    value = struct.pack("<I", 1)
    _command(writer, delta=5, number=11)
    writer.bits(1, 1)
    writer.i32(4)
    writer.i32(len(value))
    for byte in value:
        writer.byte(byte)

    _command(writer, delta=following_delta, number=15)
    writer.bits(1, 1)
    writer.i32(2)
    writer.byte(0xAA)
    writer.byte(0x55)

    _command(writer, delta=3, number=6)
    return _demo_bytes(bytes(writer.data), length=80), value


def _failed_file_write_demo() -> bytes:
    writer = _BitWriter()
    _command(writer, delta=15, number=31)
    _command(writer, delta=5, number=16)
    writer.bits(0, 1)
    _command(writer, delta=15, number=31)
    return _demo_bytes(bytes(writer.data), length=50)


def _failed_file_write_sequence_demo() -> bytes:
    writer = _BitWriter()
    _command(writer, delta=15, number=31)
    _command(writer, delta=5, number=16)
    writer.bits(0, 1)
    _command(writer, delta=0, number=16)
    writer.bits(0, 1)
    _command(writer, delta=2, number=16)
    writer.bits(0, 1)
    _command(writer, delta=13, number=31)
    return _demo_bytes(bytes(writer.data), length=50)


def _loading_complete_demo() -> bytes:
    writer = _BitWriter()
    _command(writer, delta=15, number=31)
    _command(writer, delta=5, number=9)
    _command(writer, delta=15, number=31)
    return _demo_bytes(bytes(writer.data), length=50)


def _relocatable_service_demo() -> tuple[bytes, bytes]:
    writer = _BitWriter()
    _command(writer, delta=15, number=31)
    _command(writer, delta=15, number=31)

    value = struct.pack("<I", 1)
    _command(writer, delta=5, number=11)
    writer.bits(1, 1)
    writer.i32(4)
    writer.i32(len(value))
    for byte in value:
        writer.byte(byte)
    _command(writer, delta=0, number=15)
    writer.bits(1, 1)
    writer.i32(2)
    writer.byte(0xAA)
    writer.byte(0x55)

    _command(writer, delta=5, number=31)
    _command(writer, delta=5, number=31)
    _command(writer, delta=3, number=6)
    return _demo_bytes(bytes(writer.data), length=80), value


def _rows(data: bytes):
    module = _load_popcap_dmo_module()
    offset = _command_stream_offset(data)
    length = struct.unpack_from("<I", data, offset - 4)[0]
    rows, _ = _scan_commands(module, data[offset:], length)
    return rows


def test_removal_transfers_timing_to_following_command() -> None:
    source, value = _service_result_demo()
    source_rows = _rows(source)
    target = source_rows[2]

    output, provenance = remove_command(
        source,
        command_index=2,
        expected_kind="registry_read",
        expected_update=35,
        expected_start_bit=int(target["start"]),
        expected_end_bit=int(target["end"]),
        expected_payload_sha256=hashlib.sha256(value).hexdigest(),
    )

    parsed = PopCapDemo.from_bytes(output)
    assert [command.kind for command in parsed.commands] == [
        "idle",
        "idle",
        "file_read",
        "close",
    ]
    assert [command.update for command in parsed.commands] == [
        15,
        30,
        35,
        38,
    ]
    assert provenance["version"] == 2
    transformation = provenance["transformation"]
    assert transformation["removed_timing_delta"] == 5
    assert transformation["following_original_timing_delta"] == 0
    assert transformation["following_replacement_timing_delta"] == 5
    assert transformation["timeline_preserved"] is True


def test_removal_rejects_a_later_following_command() -> None:
    source, value = _service_result_demo(following_delta=1)
    target = _rows(source)[2]

    with pytest.raises(ValueError, match="following command update differs"):
        remove_command(
            source,
            command_index=2,
            expected_kind="registry_read",
            expected_update=35,
            expected_start_bit=int(target["start"]),
            expected_end_bit=int(target["end"]),
            expected_payload_sha256=hashlib.sha256(value).hexdigest(),
        )


def test_removal_can_transfer_a_later_following_timing() -> None:
    source, value = _service_result_demo(following_delta=1)
    target = _rows(source)[2]

    output, provenance = remove_command(
        source,
        command_index=2,
        expected_kind="registry_read",
        expected_update=35,
        expected_start_bit=int(target["start"]),
        expected_end_bit=int(target["end"]),
        expected_payload_sha256=hashlib.sha256(value).hexdigest(),
        allow_timing_transfer=True,
    )

    parsed = PopCapDemo.from_bytes(output)
    assert [command.kind for command in parsed.commands] == [
        "idle",
        "idle",
        "file_read",
        "close",
    ]
    assert [command.update for command in parsed.commands] == [
        15,
        30,
        36,
        39,
    ]
    assert (
        provenance["transformation"][
            "later_following_command_allowed"
        ]
        is True
    )


def test_idle_replacement_preserves_a_saturated_following_delta() -> None:
    source, value = _service_result_demo(following_delta=15)
    target = _rows(source)[2]

    output, provenance = replace_command_with_idle(
        source,
        command_index=2,
        expected_kind="registry_read",
        expected_update=35,
        expected_start_bit=int(target["start"]),
        expected_end_bit=int(target["end"]),
        expected_payload_sha256=hashlib.sha256(value).hexdigest(),
    )

    parsed = PopCapDemo.from_bytes(output)
    assert [command.kind for command in parsed.commands] == [
        "idle",
        "idle",
        "idle",
        "file_read",
        "close",
    ]
    assert [command.update for command in parsed.commands] == [
        15,
        30,
        35,
        50,
        53,
    ]
    transformation = provenance["transformation"]
    assert transformation["replacement_kind"] == "idle"
    assert transformation["replacement_timing_delta"] == 5
    assert transformation["timeline_preserved"] is True


def test_idle_replacement_binds_a_payload_without_an_embedded_blob() -> None:
    source = _failed_file_write_demo()
    target = _rows(source)[1]
    payload_sha256 = hashlib.sha256(b'{"success":false}').hexdigest()

    output, provenance = replace_command_with_idle(
        source,
        command_index=1,
        expected_kind="file_write",
        expected_update=20,
        expected_start_bit=int(target["start"]),
        expected_end_bit=int(target["end"]),
        expected_payload_sha256=payload_sha256,
    )

    parsed = PopCapDemo.from_bytes(output)
    assert [command.kind for command in parsed.commands] == [
        "idle",
        "idle",
        "idle",
    ]
    assert [command.update for command in parsed.commands] == [15, 20, 35]
    assert (
        provenance["transformation"]["replaced_payload_sha256"]
        == payload_sha256
    )


def test_service_sequence_idle_replacement_is_atomic_and_bound() -> None:
    source = _failed_file_write_sequence_demo()
    rows = _rows(source)
    sequence = rows[1:4]
    payload_sha256 = hashlib.sha256(b'{"success":false}').hexdigest()

    output, provenance = replace_service_sequence_with_idle(
        source,
        start_index=1,
        end_index=3,
        expected_updates=(20, 20, 22),
        expected_start_bit=int(sequence[0]["start"]),
        expected_end_bit=int(sequence[-1]["end"]),
        expected_kind="file_write",
        expected_payload_sha256=payload_sha256,
    )

    parsed = PopCapDemo.from_bytes(output)
    assert [command.kind for command in parsed.commands] == [
        "idle",
        "idle",
        "idle",
        "idle",
        "idle",
    ]
    assert [command.update for command in parsed.commands] == [
        15,
        20,
        20,
        22,
        35,
    ]
    transformation = provenance["transformation"]
    assert transformation["command_count"] == 3
    assert transformation["source_bits"] == 33
    assert transformation["replacement_bits"] == 30
    assert transformation["recorded_updates"] == [20, 20, 22]
    assert transformation["timeline_preserved"] is True
    assert transformation["command_count_preserved"] is True
    assert transformation["non_target_command_bits_preserved"] is True


def test_service_sequence_idle_replacement_rejects_update_drift() -> None:
    source = _failed_file_write_sequence_demo()
    rows = _rows(source)
    sequence = rows[1:4]

    with pytest.raises(
        ValueError,
        match="service sequence update pattern mismatch",
    ):
        replace_service_sequence_with_idle(
            source,
            start_index=1,
            end_index=3,
            expected_updates=(20, 21, 22),
            expected_start_bit=int(sequence[0]["start"]),
            expected_end_bit=int(sequence[-1]["end"]),
            expected_kind="file_write",
            expected_payload_sha256=hashlib.sha256(
                b'{"success":false}'
            ).hexdigest(),
        )


def test_failed_file_write_insertion_preserves_source_commands() -> None:
    source = _loading_complete_demo()
    anchor = _rows(source)[1]

    output, provenance = insert_failed_file_writes(
        source,
        anchor_index=1,
        expected_anchor_kind="loading_complete",
        expected_anchor_update=20,
        expected_anchor_start_bit=int(anchor["start"]),
        expected_anchor_end_bit=int(anchor["end"]),
        count=3,
    )

    parsed = PopCapDemo.from_bytes(output)
    assert [command.kind for command in parsed.commands] == [
        "idle",
        "loading_complete",
        "file_write",
        "file_write",
        "file_write",
        "idle",
    ]
    assert [command.update for command in parsed.commands] == [
        15,
        20,
        20,
        20,
        20,
        35,
    ]
    assert all(
        command.payload["success"] is False
        for command in parsed.commands[2:5]
    )
    transformation = provenance["transformation"]
    assert transformation["inserted_command_count"] == 3
    assert transformation["first_inserted_timing_delta"] == 0
    assert transformation["following_inserted_timing_delta"] == 0
    assert transformation["source_command_semantics_preserved"] is True


def test_failed_file_write_insertion_can_preload_a_future_result() -> None:
    source = _loading_complete_demo()
    anchor = _rows(source)[1]

    output, provenance = insert_failed_file_writes(
        source,
        anchor_index=1,
        expected_anchor_kind="loading_complete",
        expected_anchor_update=20,
        expected_anchor_start_bit=int(anchor["start"]),
        expected_anchor_end_bit=int(anchor["end"]),
        count=2,
        inserted_update=21,
    )

    parsed = PopCapDemo.from_bytes(output)
    assert [command.update for command in parsed.commands] == [
        15,
        20,
        21,
        21,
        35,
    ]
    transformation = provenance["transformation"]
    assert transformation["inserted_update"] == 21
    assert transformation["first_inserted_timing_delta"] == 1


def test_successful_file_write_insertion_is_explicit_and_bit_exact() -> None:
    source = _loading_complete_demo()
    anchor = _rows(source)[1]

    output, provenance = insert_failed_file_writes(
        source,
        anchor_index=1,
        expected_anchor_kind="loading_complete",
        expected_anchor_update=20,
        expected_anchor_start_bit=int(anchor["start"]),
        expected_anchor_end_bit=int(anchor["end"]),
        count=1,
        inserted_success=True,
    )

    before_rows = _rows(source)
    after_rows = _rows(output)
    inserted = after_rows[2]
    assert inserted["kind"] == "file_write"
    assert inserted["payload"] == {"success": True}
    assert int(inserted["end"]) - int(inserted["start"]) == 11
    assert [
        (row["update"], row["kind"], row["payload"])
        for row in after_rows[3:]
    ] == [
        (row["update"], row["kind"], row["payload"])
        for row in before_rows[2:]
    ]
    transformation = provenance["transformation"]
    assert transformation["inserted_payload"] == {"success": True}
    assert transformation["inserted_bits_per_command"] == 11
    assert transformation["source_command_semantics_preserved"] is True
    assert validate_successful_file_write_padding_provenance(
        output,
        provenance,
    ) == (2,)


def test_successful_padding_provenance_rejects_identity_and_transform_drift(
) -> None:
    source = _loading_complete_demo()
    anchor = _rows(source)[1]
    output, provenance = insert_failed_file_writes(
        source,
        anchor_index=1,
        expected_anchor_kind="loading_complete",
        expected_anchor_update=20,
        expected_anchor_start_bit=int(anchor["start"]),
        expected_anchor_end_bit=int(anchor["end"]),
        count=1,
        inserted_success=True,
    )

    bad_hash = copy.deepcopy(provenance)
    bad_hash["output"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="output identity"):
        validate_successful_file_write_padding_provenance(output, bad_hash)

    bad_payload = copy.deepcopy(provenance)
    bad_payload["transformation"]["inserted_payload"] = {"success": False}
    with pytest.raises(ValueError, match="exact successful padding"):
        validate_successful_file_write_padding_provenance(output, bad_payload)

    bad_anchor = copy.deepcopy(provenance)
    bad_anchor["transformation"]["anchor_bit_start"] += 1
    with pytest.raises(ValueError, match="anchor or inserted row"):
        validate_successful_file_write_padding_provenance(output, bad_anchor)

    changed_output = bytearray(output)
    changed_output[-1] ^= 0x80
    with pytest.raises(ValueError, match="output identity"):
        validate_successful_file_write_padding_provenance(
            bytes(changed_output),
            provenance,
        )


def test_failed_file_write_insertion_rejects_anchor_drift() -> None:
    source = _loading_complete_demo()
    anchor = _rows(source)[1]

    with pytest.raises(ValueError, match="anchor 1 update mismatch"):
        insert_failed_file_writes(
            source,
            anchor_index=1,
            expected_anchor_kind="loading_complete",
            expected_anchor_update=21,
            expected_anchor_start_bit=int(anchor["start"]),
            expected_anchor_end_bit=int(anchor["end"]),
            count=3,
        )


def test_service_block_retiming_preserves_payloads_and_later_updates() -> None:
    source, value = _service_result_demo()
    rows = _rows(source)
    block = rows[2:4]
    payload_hashes = [
        hashlib.sha256(value).hexdigest(),
        hashlib.sha256(bytes((0xAA, 0x55))).hexdigest(),
    ]

    output, provenance = retime_service_block(
        source,
        start_index=2,
        end_index=3,
        expected_update=35,
        new_update=36,
        expected_start_bit=int(block[0]["start"]),
        expected_end_bit=int(block[-1]["end"]),
        expected_kinds=("registry_read", "file_read"),
        expected_payload_sha256=payload_hashes,
    )

    parsed = PopCapDemo.from_bytes(output)
    assert [command.update for command in parsed.commands] == [
        15,
        30,
        36,
        36,
        38,
    ]
    assert [command.kind for command in parsed.commands] == [
        "idle",
        "idle",
        "registry_read",
        "file_read",
        "close",
    ]
    transformation = provenance["transformation"]
    assert transformation["command_payloads_preserved"] is True
    assert transformation["later_timeline_preserved"] is True
    assert transformation["first_original_timing_delta"] == 5
    assert transformation["first_replacement_timing_delta"] == 6
    assert transformation["following_original_timing_delta"] == 3
    assert transformation["following_replacement_timing_delta"] == 2


def test_service_block_relocation_copies_exact_payload_after_anchor() -> None:
    source, value = _relocatable_service_demo()
    rows = _rows(source)
    block = rows[2:4]
    anchor = rows[5]

    output, provenance = relocate_service_block(
        source,
        start_index=2,
        end_index=3,
        expected_update=35,
        expected_start_bit=int(block[0]["start"]),
        expected_end_bit=int(block[-1]["end"]),
        expected_kinds=("registry_read", "file_read"),
        expected_payload_sha256=(
            hashlib.sha256(value).hexdigest(),
            hashlib.sha256(bytes((0xAA, 0x55))).hexdigest(),
        ),
        anchor_index=5,
        expected_anchor_kind="idle",
        expected_anchor_update=45,
        expected_anchor_start_bit=int(anchor["start"]),
        expected_anchor_end_bit=int(anchor["end"]),
        new_update=46,
    )

    parsed = PopCapDemo.from_bytes(output)
    assert [command.kind for command in parsed.commands] == [
        "idle",
        "idle",
        "idle",
        "idle",
        "idle",
        "idle",
        "registry_read",
        "file_read",
        "close",
    ]
    assert [command.update for command in parsed.commands] == [
        15,
        30,
        35,
        35,
        40,
        45,
        46,
        46,
        48,
    ]
    transformation = provenance["transformation"]
    assert transformation["payload_bits_copied_exactly"] is True
    assert transformation["later_timeline_preserved"] is True
    assert transformation["placeholder_count"] == 2
