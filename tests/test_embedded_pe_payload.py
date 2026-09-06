"""Tests for signed PE payload identity derivation."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import pytest

from tools.embedded_pe_payload import (
    EmbeddedPeError,
    inspect_embedded_signed_pe,
    main,
)


def _signed_pe(
    *,
    timestamp: int = 123,
    checksum: int = 0x12345678,
    certificate_bytes: int = 0x20,
) -> bytes:
    pe = bytearray(0x400 + certificate_bytes)
    pe[:2] = b"MZ"
    struct.pack_into("<I", pe, 0x3C, 0x80)
    pe[0x80:0x84] = b"PE\0\0"
    struct.pack_into(
        "<HHIIIHH",
        pe,
        0x84,
        0x14C,
        1,
        timestamp,
        0,
        0,
        0xE0,
        0x0103,
    )
    optional = 0x98
    struct.pack_into("<H", pe, optional, 0x10B)
    struct.pack_into("<I", pe, optional + 36, 0x200)
    struct.pack_into("<I", pe, optional + 56, 0x2000)
    struct.pack_into("<I", pe, optional + 60, 0x200)
    struct.pack_into("<I", pe, optional + 64, checksum)
    struct.pack_into("<I", pe, optional + 92, 16)
    struct.pack_into(
        "<II",
        pe,
        optional + 96 + 4 * 8,
        0x400,
        certificate_bytes,
    )
    section = optional + 0xE0
    pe[section : section + 8] = b".text\0\0\0"
    struct.pack_into(
        "<IIII",
        pe,
        section + 8,
        0x180,
        0x1000,
        0x200,
        0x200,
    )
    pe[0x200:0x400] = bytes(index % 251 for index in range(0x200))
    pe[0x400:] = b"C" * certificate_bytes
    return bytes(pe)


def _wrapper(
    payloads: tuple[bytes, ...],
    *,
    prefix: bytes = b"persistent-wrapper:",
    trailer: bytes = b":launcher-trailer",
) -> tuple[bytes, tuple[int, ...]]:
    result = bytearray(prefix)
    offsets: list[int] = []
    for payload in payloads:
        offsets.append(len(result))
        result.extend(payload)
        result.extend(b":between:")
    result.extend(trailer)
    return bytes(result), tuple(offsets)


def test_derives_unique_embedded_signed_payload_identity(
    tmp_path: Path,
) -> None:
    payload = _signed_pe()
    wrapper, offsets = _wrapper((payload,))
    path = tmp_path / "launcher.exe"
    path.write_bytes(wrapper)

    identity = inspect_embedded_signed_pe(
        path,
        expected_payload_bytes=len(payload),
    )

    assert identity.source_bytes == len(wrapper)
    assert identity.source_sha256 == (
        "sha256:" + hashlib.sha256(wrapper).hexdigest()
    )
    assert identity.payload_offset == offsets[0]
    assert identity.payload_bytes == len(payload)
    assert identity.payload_sha256 == (
        "sha256:" + hashlib.sha256(payload).hexdigest()
    )
    assert identity.payload_extent_basis == "certificate_table_end"
    assert identity.trailing_source_bytes == (
        len(wrapper) - offsets[0] - len(payload)
    )
    assert identity.machine == 0x14C
    assert identity.section_count == 1
    assert identity.pe_timestamp == 123
    assert identity.pe_checksum == 0x12345678
    assert identity.size_of_image == 0x2000
    assert identity.size_of_headers == 0x200
    assert identity.certificate_table_offset == 0x400
    assert identity.certificate_table_bytes == 0x20


def test_expected_size_selects_one_of_multiple_payloads(
    tmp_path: Path,
) -> None:
    first = _signed_pe(certificate_bytes=0x20)
    second = _signed_pe(certificate_bytes=0x28)
    wrapper, offsets = _wrapper((first, second))
    path = tmp_path / "launcher.exe"
    path.write_bytes(wrapper)

    identity = inspect_embedded_signed_pe(
        path,
        expected_payload_bytes=len(second),
    )

    assert identity.payload_offset == offsets[1]
    assert identity.payload_bytes == len(second)


@pytest.mark.parametrize(
    "expected",
    (0, -1, True, 513 * 1024**2),
)
def test_rejects_invalid_expected_payload_size(
    tmp_path: Path,
    expected: int,
) -> None:
    wrapper, unused = _wrapper((_signed_pe(),))
    del unused
    path = tmp_path / "launcher.exe"
    path.write_bytes(wrapper)

    with pytest.raises(
        EmbeddedPeError,
        match="expected_payload_bytes_invalid",
    ):
        inspect_embedded_signed_pe(
            path,
            expected_payload_bytes=expected,
        )


def test_rejects_ambiguous_or_mismatched_payload(
    tmp_path: Path,
) -> None:
    payload = _signed_pe()
    wrapper, unused = _wrapper((payload, payload))
    del unused
    path = tmp_path / "launcher.exe"
    path.write_bytes(wrapper)

    with pytest.raises(
        EmbeddedPeError,
        match="embedded_signed_pe_not_unique",
    ):
        inspect_embedded_signed_pe(path)
    with pytest.raises(
        EmbeddedPeError,
        match="embedded_signed_pe_not_unique",
    ):
        inspect_embedded_signed_pe(
            path,
            expected_payload_bytes=len(payload) + 1,
        )


def test_rejects_certificate_extent_past_source(
    tmp_path: Path,
) -> None:
    payload = bytearray(_signed_pe())
    optional = 0x98
    struct.pack_into(
        "<II",
        payload,
        optional + 96 + 4 * 8,
        0x400,
        0x1000,
    )
    wrapper, unused = _wrapper((bytes(payload),))
    del unused
    path = tmp_path / "launcher.exe"
    path.write_bytes(wrapper)

    with pytest.raises(
        EmbeddedPeError,
        match="embedded_signed_pe_not_unique",
    ):
        inspect_embedded_signed_pe(path)


def test_json_cli_reports_machine_readable_identity(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = _signed_pe()
    wrapper, unused = _wrapper((payload,))
    del unused
    path = tmp_path / "launcher.exe"
    path.write_bytes(wrapper)

    assert main(
        [
            str(path),
            "--expected-payload-bytes",
            str(len(payload)),
            "--json",
        ]
    ) == 0
    result = json.loads(capsys.readouterr().out)

    assert result["payload_bytes"] == len(payload)
    assert result["payload_sha256"] == (
        "sha256:" + hashlib.sha256(payload).hexdigest()
    )


def test_cli_fails_closed_without_echoing_source_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "private-name.exe"

    assert main([str(path)]) == 1
    captured = capsys.readouterr()

    assert captured.out == ""
    assert captured.err == "error: source_executable_unavailable\n"
    assert str(path) not in captured.err
