"""Audit a signed PE payload embedded inside a persistent launcher.

The Steam build of Zuma's Revenge keeps the visible game's executable inside
``ZumasRevenge.exe`` and extracts it to a locked, temporary
``popcapgame1.exe`` while running.  This module derives the complete embedded
file extent from the inner PE section table and certificate directory, then
hashes that exact byte range without extracting or publishing the payload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

PE32_MAGIC = 0x10B
PE32_PLUS_MAGIC = 0x20B
MAX_EXECUTABLE_BYTES = 512 * 1024**2
MAX_PE_SECTIONS = 96
SECURITY_DIRECTORY_INDEX = 4


class EmbeddedPeError(ValueError):
    """Raised when a unique, bounded signed PE payload cannot be proven."""


@dataclass(frozen=True, slots=True)
class EmbeddedPeIdentity:
    """Content identity and structural provenance for one embedded PE."""

    source_bytes: int
    source_sha256: str
    payload_offset: int
    payload_bytes: int
    payload_sha256: str
    payload_extent_basis: str
    trailing_source_bytes: int
    machine: int
    section_count: int
    pe_timestamp: int
    pe_checksum: int
    size_of_image: int
    size_of_headers: int
    certificate_table_offset: int
    certificate_table_bytes: int

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _PeCandidate:
    offset: int
    file_bytes: int
    machine: int
    section_count: int
    timestamp: int
    checksum: int
    size_of_image: int
    size_of_headers: int
    certificate_offset: int
    certificate_bytes: int


def _u16(data: bytes, offset: int) -> int:
    return struct.unpack_from("<H", data, offset)[0]


def _u32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def _checked_end(start: int, size: int, limit: int) -> int | None:
    if start < 0 or size < 0:
        return None
    end = start + size
    if end < start or end > limit:
        return None
    return end


def _parse_signed_candidate(
    data: bytes,
    payload_offset: int,
) -> _PeCandidate | None:
    available = len(data) - payload_offset
    if payload_offset <= 0 or available < 0x40:
        return None
    if data[payload_offset : payload_offset + 2] != b"MZ":
        return None

    pe_relative = _u32(data, payload_offset + 0x3C)
    if pe_relative < 0x40:
        return None
    pe_offset = payload_offset + pe_relative
    if _checked_end(pe_offset, 24, len(data)) is None:
        return None
    if data[pe_offset : pe_offset + 4] != b"PE\0\0":
        return None

    coff_offset = pe_offset + 4
    (
        machine,
        section_count,
        timestamp,
        _symbol_table,
        _symbol_count,
        optional_bytes,
        _characteristics,
    ) = struct.unpack_from("<HHIIIHH", data, coff_offset)
    if not 1 <= section_count <= MAX_PE_SECTIONS:
        return None

    optional_offset = pe_offset + 24
    optional_end = _checked_end(
        optional_offset,
        optional_bytes,
        len(data),
    )
    if optional_end is None or optional_bytes < 2:
        return None
    magic = _u16(data, optional_offset)
    if magic == PE32_MAGIC:
        minimum_optional_bytes = 96
        directory_count_offset = 92
        directories_offset = 96
    elif magic == PE32_PLUS_MAGIC:
        minimum_optional_bytes = 112
        directory_count_offset = 108
        directories_offset = 112
    else:
        return None
    if optional_bytes < minimum_optional_bytes:
        return None

    file_alignment = _u32(data, optional_offset + 36)
    size_of_image = _u32(data, optional_offset + 56)
    size_of_headers = _u32(data, optional_offset + 60)
    checksum = _u32(data, optional_offset + 64)
    if (
        file_alignment == 0
        or size_of_image == 0
        or size_of_headers == 0
        or size_of_headers > available
    ):
        return None

    directory_count = _u32(
        data,
        optional_offset + directory_count_offset,
    )
    maximum_directory_count = (
        optional_bytes - directories_offset
    ) // 8
    if directory_count > maximum_directory_count:
        return None
    if directory_count <= SECURITY_DIRECTORY_INDEX:
        return None
    security_entry = (
        optional_offset
        + directories_offset
        + SECURITY_DIRECTORY_INDEX * 8
    )
    certificate_offset = _u32(data, security_entry)
    certificate_bytes = _u32(data, security_entry + 4)
    if certificate_offset == 0 or certificate_bytes == 0:
        return None

    section_table = optional_offset + optional_bytes
    section_table_end = _checked_end(
        section_table,
        section_count * 40,
        len(data),
    )
    if section_table_end is None:
        return None

    raw_end = size_of_headers
    for index in range(section_count):
        section_offset = section_table + index * 40
        raw_bytes = _u32(data, section_offset + 16)
        raw_offset = _u32(data, section_offset + 20)
        if raw_bytes == 0:
            continue
        section_end = _checked_end(raw_offset, raw_bytes, available)
        if section_end is None:
            return None
        raw_end = max(raw_end, section_end)

    certificate_end = _checked_end(
        certificate_offset,
        certificate_bytes,
        available,
    )
    if certificate_end is None or certificate_offset < raw_end:
        return None
    if certificate_end < size_of_headers:
        return None

    return _PeCandidate(
        offset=payload_offset,
        file_bytes=certificate_end,
        machine=machine,
        section_count=section_count,
        timestamp=timestamp,
        checksum=checksum,
        size_of_image=size_of_image,
        size_of_headers=size_of_headers,
        certificate_offset=certificate_offset,
        certificate_bytes=certificate_bytes,
    )


def _signed_candidates(data: bytes) -> tuple[_PeCandidate, ...]:
    candidates: list[_PeCandidate] = []
    offset = data.find(b"MZ", 1)
    while offset >= 0:
        candidate = _parse_signed_candidate(data, offset)
        if candidate is not None:
            candidates.append(candidate)
        offset = data.find(b"MZ", offset + 1)
    return tuple(candidates)


def inspect_embedded_signed_pe(
    path: str | Path,
    *,
    expected_payload_bytes: int | None = None,
) -> EmbeddedPeIdentity:
    """Return a unique embedded signed PE identity without extracting it."""

    source = Path(path)
    try:
        source_bytes = source.stat().st_size
        if (
            source_bytes <= 0
            or source_bytes > MAX_EXECUTABLE_BYTES
            or not source.is_file()
        ):
            raise EmbeddedPeError("source_executable_invalid")
        data = source.read_bytes()
    except EmbeddedPeError:
        raise
    except OSError:
        raise EmbeddedPeError("source_executable_unavailable") from None
    if len(data) != source_bytes:
        raise EmbeddedPeError("source_executable_changed")
    if (
        expected_payload_bytes is not None
        and (
            isinstance(expected_payload_bytes, bool)
            or expected_payload_bytes <= 0
            or expected_payload_bytes > MAX_EXECUTABLE_BYTES
        )
    ):
        raise EmbeddedPeError("expected_payload_bytes_invalid")

    candidates = _signed_candidates(data)
    if expected_payload_bytes is not None:
        candidates = tuple(
            candidate
            for candidate in candidates
            if candidate.file_bytes == expected_payload_bytes
        )
    if len(candidates) != 1:
        raise EmbeddedPeError("embedded_signed_pe_not_unique")
    candidate = candidates[0]
    payload_end = candidate.offset + candidate.file_bytes
    payload = data[candidate.offset : payload_end]
    return EmbeddedPeIdentity(
        source_bytes=source_bytes,
        source_sha256=(
            "sha256:" + hashlib.sha256(data).hexdigest()
        ),
        payload_offset=candidate.offset,
        payload_bytes=candidate.file_bytes,
        payload_sha256=(
            "sha256:" + hashlib.sha256(payload).hexdigest()
        ),
        payload_extent_basis="certificate_table_end",
        trailing_source_bytes=source_bytes - payload_end,
        machine=candidate.machine,
        section_count=candidate.section_count,
        pe_timestamp=candidate.timestamp,
        pe_checksum=candidate.checksum,
        size_of_image=candidate.size_of_image,
        size_of_headers=candidate.size_of_headers,
        certificate_table_offset=candidate.certificate_offset,
        certificate_table_bytes=candidate.certificate_bytes,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit a signed PE embedded in a persistent executable "
            "without extracting the payload."
        ),
    )
    parser.add_argument("path", type=Path)
    parser.add_argument("--expected-payload-bytes", type=int)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        identity = inspect_embedded_signed_pe(
            args.path,
            expected_payload_bytes=args.expected_payload_bytes,
        )
    except EmbeddedPeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    if args.json:
        print(
            json.dumps(
                identity.to_dict(),
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        return 0
    print(f"source bytes      : {identity.source_bytes:,}")
    print(f"source SHA-256    : {identity.source_sha256}")
    print(f"payload offset    : {identity.payload_offset:,}")
    print(f"payload bytes     : {identity.payload_bytes:,}")
    print(f"payload SHA-256   : {identity.payload_sha256}")
    print(f"extent basis      : {identity.payload_extent_basis}")
    print(f"certificate bytes : {identity.certificate_table_bytes:,}")
    print(f"trailing bytes    : {identity.trailing_source_bytes:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
