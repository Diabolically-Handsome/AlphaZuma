from __future__ import annotations

import struct

import pytest

from zuma_rl.pc_internal_screenshot_capability import (
    BASE_PATH_GETTER_VA,
    DUMP_DIRECTORY_VA,
    DUMP_FUNCTION_PREFIX,
    DUMP_IMAGELIST_VA,
    DUMP_PATH_GETTER_CALLS,
    DUMP_PATH_PUSHES,
    F11_KEY_HANDLER_BYTES,
    F11_KEY_HANDLER_VA,
    F11_TARGET_FUNCTION_VA,
    GENERIC_SCREENSHOT_FUNCTION_PREFIX,
    GENERIC_SCREENSHOT_FUNCTION_VA,
    InternalScreenshotCapabilityError,
    SCREENSHOT_DIRECTORY_VA,
    SCREENSHOT_FILENAME_PUSH_VA,
    SCREENSHOT_FILENAME_VA,
    SCREENSHOT_GLOB_PUSH_VA,
    SCREENSHOT_GLOB_VA,
    SCREENSHOT_PATH_GETTER_CALL_VA,
    SCREENSHOT_PATH_PUSH_VA,
    verify_runtime_static_routes,
)


_IMAGE_BASE = 0x00400000
_SECTION_VA = 0x1000


def _synthetic_runtime() -> bytearray:
    maximum_va = max(DUMP_IMAGELIST_VA + 64, SCREENSHOT_FILENAME_VA + 32)
    raw_size = maximum_va - _IMAGE_BASE - _SECTION_VA + 0x100
    raw_offset = 0x200
    payload = bytearray(raw_offset + raw_size)
    payload[:2] = b"MZ"
    pe_offset = 0x80
    struct.pack_into("<I", payload, 0x3C, pe_offset)
    payload[pe_offset : pe_offset + 4] = b"PE\0\0"
    struct.pack_into("<H", payload, pe_offset + 4, 0x014C)
    struct.pack_into("<H", payload, pe_offset + 6, 1)
    struct.pack_into("<H", payload, pe_offset + 20, 0xE0)
    optional = pe_offset + 24
    struct.pack_into("<H", payload, optional, 0x10B)
    struct.pack_into("<I", payload, optional + 28, _IMAGE_BASE)
    section = optional + 0xE0
    payload[section : section + 8] = b".all\0\0\0\0"
    struct.pack_into(
        "<IIII",
        payload,
        section + 8,
        raw_size,
        _SECTION_VA,
        raw_size,
        raw_offset,
    )
    struct.pack_into("<I", payload, section + 36, 0x60000020)

    def offset(virtual_address: int) -> int:
        return raw_offset + virtual_address - _IMAGE_BASE - _SECTION_VA

    def write(virtual_address: int, content: bytes) -> None:
        start = offset(virtual_address)
        payload[start : start + len(content)] = content

    def call(source: int, target: int) -> None:
        write(source, b"\xe8" + struct.pack("<i", target - source - 5))

    def push(source: int, value: int) -> None:
        write(source, b"\x68" + struct.pack("<I", value))

    write(F11_KEY_HANDLER_VA, F11_KEY_HANDLER_BYTES)
    write(F11_TARGET_FUNCTION_VA, DUMP_FUNCTION_PREFIX)
    for source in DUMP_PATH_GETTER_CALLS:
        call(source, BASE_PATH_GETTER_VA)
    for source in DUMP_PATH_PUSHES:
        push(source, DUMP_DIRECTORY_VA)
    write(DUMP_DIRECTORY_VA, b"_dump\0\0\0_dump\\imagelist.html\0")

    write(
        GENERIC_SCREENSHOT_FUNCTION_VA,
        GENERIC_SCREENSHOT_FUNCTION_PREFIX,
    )
    call(SCREENSHOT_PATH_GETTER_CALL_VA, BASE_PATH_GETTER_VA)
    push(SCREENSHOT_PATH_PUSH_VA, SCREENSHOT_DIRECTORY_VA)
    push(SCREENSHOT_GLOB_PUSH_VA, SCREENSHOT_GLOB_VA)
    push(SCREENSHOT_FILENAME_PUSH_VA, SCREENSHOT_FILENAME_VA)
    write(
        SCREENSHOT_DIRECTORY_VA,
        b"_screenshots\0\0\0\0*.png\0\0\0%d.png\0",
    )
    return payload


def _offset(virtual_address: int) -> int:
    return 0x200 + virtual_address - _IMAGE_BASE - _SECTION_VA


def test_static_routes_distinguish_f11_dump_from_png_exporter() -> None:
    proof = verify_runtime_static_routes(bytes(_synthetic_runtime()))

    assert proof["f11_target"]["directory"] == "_dump"
    assert proof["f11_target"]["imagelist"] == "_dump\\imagelist.html"
    assert proof["generic_png_exporter"]["directory"] == "_screenshots"
    assert proof["generic_png_exporter"]["filename_format"] == "%d.png"
    assert proof["generic_png_exporter"][
        "direct_rel32_call_or_jump_xrefs"
    ] == []


def test_static_routes_reject_changed_f11_target() -> None:
    payload = _synthetic_runtime()
    call_offset = _offset(F11_KEY_HANDLER_VA) + 23
    payload[call_offset + 1 : call_offset + 5] = struct.pack("<i", 0)

    with pytest.raises(
        InternalScreenshotCapabilityError,
        match="runtime_f11_handler_bytes_mismatch",
    ):
        verify_runtime_static_routes(bytes(payload))


def test_static_routes_reject_direct_generic_exporter_xref() -> None:
    payload = _synthetic_runtime()
    source = 0x00500000
    payload[_offset(source)] = 0xE8
    struct.pack_into(
        "<i",
        payload,
        _offset(source) + 1,
        GENERIC_SCREENSHOT_FUNCTION_VA - source - 5,
    )

    with pytest.raises(
        InternalScreenshotCapabilityError,
        match="runtime_screenshot_function_direct_xref_present",
    ):
        verify_runtime_static_routes(bytes(payload))
