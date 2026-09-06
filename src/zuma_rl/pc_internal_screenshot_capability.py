"""Fail-closed static proof of the retail F11 key route.

The pinned Zuma's Revenge payload contains a generic PNG-export helper and
separate screenshot-looking strings.  Their mere presence does not establish
that F11 reaches the helper.  This module verifies the retail key handler,
resolves its direct call, validates both path families, and enumerates direct
static references so the distinction is machine-checkable.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import struct
from typing import Any, Mapping, Sequence


CAPABILITY_SCHEMA = "zuma-rl.pc-internal-screenshot-capability"
CAPABILITY_VERSION = 1
CAPABILITY_CLASSIFICATION = (
    "retail_f11_routes_to_debug_image_dump_not_gameplay_screenshot"
)
EXPECTED_RETAIL_RUNTIME_BYTES = 6_657_328
EXPECTED_RETAIL_RUNTIME_SHA256 = (
    "sha256:2181ce2bfbfcb4678bf69a1474e08d3db941311aa768176a88453cc69692af20"
)

EXPECTED_IMAGE_BASE = 0x00400000
F11_KEY_HANDLER_VA = 0x0068844E
F11_KEY_HANDLER_BYTES = bytes.fromhex(
    "83f87a751d8b8e4003000080b9040100000074078bce"
    "e85744ffffb001e9bf000000"
)
F11_VIRTUAL_KEY = 0x7A
F11_DEBUG_OBJECT_OFFSET = 0x340
F11_DEBUG_FLAG_OFFSET = 0x104
F11_TARGET_CALL_VA = 0x00688464
F11_TARGET_FUNCTION_VA = 0x0067C8C0

BASE_PATH_GETTER_VA = 0x00617D60
DUMP_PATH_GETTER_CALLS = (0x0067C911, 0x0067C9A7)
DUMP_PATH_PUSHES = (0x0067C916, 0x0067C9AC)
DUMP_DIRECTORY_VA = 0x00988C90
DUMP_IMAGELIST_VA = 0x00988C98
DUMP_FUNCTION_PREFIX = bytes.fromhex(
    "6aff6809b3930064a1000000005081ec"
)

GENERIC_SCREENSHOT_FUNCTION_VA = 0x00487DA0
GENERIC_SCREENSHOT_FUNCTION_PREFIX = bytes.fromhex("558bec")
SCREENSHOT_PATH_GETTER_CALL_VA = 0x00487DF8
SCREENSHOT_PATH_PUSH_VA = 0x00487DFD
SCREENSHOT_GLOB_PUSH_VA = 0x00487E82
SCREENSHOT_FILENAME_PUSH_VA = 0x00487EF1
SCREENSHOT_DIRECTORY_VA = 0x00965B24
SCREENSHOT_GLOB_VA = 0x00965B34
SCREENSHOT_FILENAME_VA = 0x00965B3C


class InternalScreenshotCapabilityError(RuntimeError):
    """The runtime or corroborating probe does not match the frozen proof."""


def _fail(reason: str) -> None:
    raise InternalScreenshotCapabilityError(reason)


def _sha256_bytes(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _sha256_path(path: Path) -> str:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
    except OSError as error:
        raise InternalScreenshotCapabilityError(
            "artifact_unreadable"
        ) from error
    return f"sha256:{digest.hexdigest()}"


@dataclass(frozen=True)
class _PESection:
    name: str
    virtual_address: int
    virtual_size: int
    raw_offset: int
    raw_size: int
    characteristics: int


@dataclass(frozen=True)
class _PELayout:
    image_base: int
    sections: tuple[_PESection, ...]


def _parse_pe32(payload: bytes) -> _PELayout:
    if len(payload) < 0x40 or payload[:2] != b"MZ":
        _fail("runtime_dos_header_invalid")
    pe_offset = struct.unpack_from("<I", payload, 0x3C)[0]
    if (
        pe_offset + 24 > len(payload)
        or payload[pe_offset : pe_offset + 4] != b"PE\0\0"
    ):
        _fail("runtime_pe_header_invalid")
    machine, section_count = struct.unpack_from("<HH", payload, pe_offset + 4)
    optional_size = struct.unpack_from("<H", payload, pe_offset + 20)[0]
    optional_offset = pe_offset + 24
    if (
        machine != 0x014C
        or section_count <= 0
        or optional_size < 0x60
        or optional_offset + optional_size > len(payload)
        or struct.unpack_from("<H", payload, optional_offset)[0] != 0x10B
    ):
        _fail("runtime_pe32_header_invalid")
    image_base = struct.unpack_from("<I", payload, optional_offset + 28)[0]
    section_table = optional_offset + optional_size
    sections: list[_PESection] = []
    for index in range(section_count):
        offset = section_table + index * 40
        if offset + 40 > len(payload):
            _fail("runtime_pe_section_table_invalid")
        name_bytes = payload[offset : offset + 8].split(b"\0", 1)[0]
        try:
            name = name_bytes.decode("ascii")
        except UnicodeDecodeError:
            _fail("runtime_pe_section_name_invalid")
        virtual_size, virtual_address, raw_size, raw_offset = struct.unpack_from(
            "<IIII", payload, offset + 8
        )
        characteristics = struct.unpack_from("<I", payload, offset + 36)[0]
        if raw_size <= 0 or raw_offset + raw_size > len(payload):
            _fail("runtime_pe_section_bounds_invalid")
        sections.append(
            _PESection(
                name=name,
                virtual_address=virtual_address,
                virtual_size=virtual_size,
                raw_offset=raw_offset,
                raw_size=raw_size,
                characteristics=characteristics,
            )
        )
    return _PELayout(image_base=image_base, sections=tuple(sections))


def _read_va(
    payload: bytes,
    layout: _PELayout,
    virtual_address: int,
    size: int,
) -> bytes:
    if size < 0 or virtual_address < layout.image_base:
        _fail("runtime_virtual_address_invalid")
    relative = virtual_address - layout.image_base
    for section in layout.sections:
        local = relative - section.virtual_address
        if 0 <= local and local + size <= section.raw_size:
            offset = section.raw_offset + local
            return payload[offset : offset + size]
    _fail("runtime_virtual_address_unmapped")


def _cstring(
    payload: bytes,
    layout: _PELayout,
    virtual_address: int,
    *,
    maximum_bytes: int = 128,
) -> str:
    result = bytearray()
    for index in range(maximum_bytes):
        value = _read_va(payload, layout, virtual_address + index, 1)[0]
        if value == 0:
            try:
                return result.decode("ascii")
            except UnicodeDecodeError:
                _fail("runtime_string_not_ascii")
        result.append(value)
    _fail("runtime_string_unterminated")


def _rel32_target(
    payload: bytes,
    layout: _PELayout,
    call_address: int,
) -> tuple[bytes, int]:
    instruction = _read_va(payload, layout, call_address, 5)
    if instruction[0] != 0xE8:
        _fail("runtime_call_opcode_mismatch")
    displacement = struct.unpack_from("<i", instruction, 1)[0]
    return instruction, call_address + 5 + displacement


def _push_imm32(
    payload: bytes,
    layout: _PELayout,
    instruction_address: int,
) -> tuple[bytes, int]:
    instruction = _read_va(payload, layout, instruction_address, 5)
    if instruction[0] != 0x68:
        _fail("runtime_push_opcode_mismatch")
    return instruction, struct.unpack_from("<I", instruction, 1)[0]


def _direct_rel32_xrefs(
    payload: bytes,
    layout: _PELayout,
    target_address: int,
) -> tuple[Mapping[str, Any], ...]:
    hits: list[Mapping[str, Any]] = []
    for section in layout.sections:
        if not section.characteristics & 0x20000000:
            continue
        raw = payload[
            section.raw_offset : section.raw_offset + section.raw_size
        ]
        for local in range(max(0, len(raw) - 4)):
            opcode = raw[local]
            if opcode not in (0xE8, 0xE9):
                continue
            source = layout.image_base + section.virtual_address + local
            displacement = struct.unpack_from("<i", raw, local + 1)[0]
            if source + 5 + displacement == target_address:
                hits.append(
                    {
                        "source_virtual_address": source,
                        "opcode": "call_rel32" if opcode == 0xE8 else "jmp_rel32",
                    }
                )
    return tuple(hits)


def _absolute_va_file_offsets(
    payload: bytes,
    virtual_address: int,
) -> tuple[int, ...]:
    needle = struct.pack("<I", virtual_address)
    offsets: list[int] = []
    start = 0
    while True:
        found = payload.find(needle, start)
        if found < 0:
            return tuple(offsets)
        offsets.append(found)
        start = found + 1


def _verify_call(
    payload: bytes,
    layout: _PELayout,
    address: int,
    expected_target: int,
) -> Mapping[str, Any]:
    instruction, target = _rel32_target(payload, layout, address)
    if target != expected_target:
        _fail("runtime_call_target_mismatch")
    return {
        "instruction_virtual_address": address,
        "instruction_hex": instruction.hex(),
        "target_virtual_address": target,
    }


def _verify_push(
    payload: bytes,
    layout: _PELayout,
    address: int,
    expected_value: int,
) -> Mapping[str, Any]:
    instruction, value = _push_imm32(payload, layout, address)
    if value != expected_value:
        _fail("runtime_push_value_mismatch")
    return {
        "instruction_virtual_address": address,
        "instruction_hex": instruction.hex(),
        "value_virtual_address": value,
    }


def verify_runtime_static_routes(payload: bytes) -> Mapping[str, Any]:
    """Verify and describe the two distinct retail code paths."""

    layout = _parse_pe32(payload)
    if layout.image_base != EXPECTED_IMAGE_BASE:
        _fail("runtime_image_base_mismatch")

    handler = _read_va(
        payload,
        layout,
        F11_KEY_HANDLER_VA,
        len(F11_KEY_HANDLER_BYTES),
    )
    if handler != F11_KEY_HANDLER_BYTES:
        _fail("runtime_f11_handler_bytes_mismatch")
    f11_call = _verify_call(
        payload,
        layout,
        F11_TARGET_CALL_VA,
        F11_TARGET_FUNCTION_VA,
    )

    dump_prefix = _read_va(
        payload,
        layout,
        F11_TARGET_FUNCTION_VA,
        len(DUMP_FUNCTION_PREFIX),
    )
    if dump_prefix != DUMP_FUNCTION_PREFIX:
        _fail("runtime_dump_function_prefix_mismatch")
    dump_getters = [
        _verify_call(
            payload,
            layout,
            address,
            BASE_PATH_GETTER_VA,
        )
        for address in DUMP_PATH_GETTER_CALLS
    ]
    dump_pushes = [
        _verify_push(payload, layout, address, DUMP_DIRECTORY_VA)
        for address in DUMP_PATH_PUSHES
    ]
    dump_directory = _cstring(payload, layout, DUMP_DIRECTORY_VA)
    dump_imagelist = _cstring(payload, layout, DUMP_IMAGELIST_VA)
    if dump_directory != "_dump" or dump_imagelist != "_dump\\imagelist.html":
        _fail("runtime_dump_path_strings_mismatch")

    screenshot_prefix = _read_va(
        payload,
        layout,
        GENERIC_SCREENSHOT_FUNCTION_VA,
        len(GENERIC_SCREENSHOT_FUNCTION_PREFIX),
    )
    if screenshot_prefix != GENERIC_SCREENSHOT_FUNCTION_PREFIX:
        _fail("runtime_screenshot_function_prefix_mismatch")
    screenshot_getter = _verify_call(
        payload,
        layout,
        SCREENSHOT_PATH_GETTER_CALL_VA,
        BASE_PATH_GETTER_VA,
    )
    screenshot_pushes = [
        _verify_push(
            payload,
            layout,
            SCREENSHOT_PATH_PUSH_VA,
            SCREENSHOT_DIRECTORY_VA,
        ),
        _verify_push(
            payload,
            layout,
            SCREENSHOT_GLOB_PUSH_VA,
            SCREENSHOT_GLOB_VA,
        ),
        _verify_push(
            payload,
            layout,
            SCREENSHOT_FILENAME_PUSH_VA,
            SCREENSHOT_FILENAME_VA,
        ),
    ]
    screenshot_directory = _cstring(payload, layout, SCREENSHOT_DIRECTORY_VA)
    screenshot_glob = _cstring(payload, layout, SCREENSHOT_GLOB_VA)
    screenshot_filename = _cstring(payload, layout, SCREENSHOT_FILENAME_VA)
    if (
        screenshot_directory != "_screenshots"
        or screenshot_glob != "*.png"
        or screenshot_filename != "%d.png"
    ):
        _fail("runtime_screenshot_path_strings_mismatch")

    dump_xrefs = _direct_rel32_xrefs(payload, layout, F11_TARGET_FUNCTION_VA)
    if dump_xrefs != (
        {
            "source_virtual_address": F11_TARGET_CALL_VA,
            "opcode": "call_rel32",
        },
    ):
        _fail("runtime_f11_target_xrefs_mismatch")
    screenshot_xrefs = _direct_rel32_xrefs(
        payload,
        layout,
        GENERIC_SCREENSHOT_FUNCTION_VA,
    )
    screenshot_absolute_refs = _absolute_va_file_offsets(
        payload,
        GENERIC_SCREENSHOT_FUNCTION_VA,
    )
    if screenshot_xrefs or screenshot_absolute_refs:
        _fail("runtime_screenshot_function_direct_xref_present")

    return {
        "image_base": layout.image_base,
        "f11_handler": {
            "virtual_address": F11_KEY_HANDLER_VA,
            "bytes": handler.hex(),
            "bytes_sha256": _sha256_bytes(handler),
            "virtual_key": F11_VIRTUAL_KEY,
            "debug_object_offset": F11_DEBUG_OBJECT_OFFSET,
            "debug_flag_offset": F11_DEBUG_FLAG_OFFSET,
            "guarded_call": f11_call,
        },
        "f11_target": {
            "classification": "debug_image_resource_dump",
            "function_virtual_address": F11_TARGET_FUNCTION_VA,
            "function_prefix_hex": dump_prefix.hex(),
            "base_path_getter_calls": dump_getters,
            "directory_pushes": dump_pushes,
            "directory": dump_directory,
            "imagelist": dump_imagelist,
            "direct_rel32_xrefs": list(dump_xrefs),
        },
        "generic_png_exporter": {
            "function_virtual_address": GENERIC_SCREENSHOT_FUNCTION_VA,
            "function_prefix_hex": screenshot_prefix.hex(),
            "base_path_getter_call": screenshot_getter,
            "path_pushes": screenshot_pushes,
            "directory": screenshot_directory,
            "glob": screenshot_glob,
            "filename_format": screenshot_filename,
            "direct_rel32_call_or_jump_xrefs": list(screenshot_xrefs),
            "absolute_va_file_offsets": list(screenshot_absolute_refs),
        },
    }


def _load_negative_live_probe(
    path: Path,
    *,
    delivery_method: str,
) -> Mapping[str, Any]:
    path = path.resolve()
    try:
        payload = path.read_bytes()
        decoded = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InternalScreenshotCapabilityError(
            "live_probe_unreadable"
        ) from error
    if not isinstance(decoded, list) or len(decoded) != 1:
        _fail("live_probe_attempt_count_invalid")
    attempt = decoded[0]
    if not isinstance(attempt, Mapping):
        _fail("live_probe_attempt_invalid")
    if (
        attempt.get("attempt") != 1
        or attempt.get("status") != "RETRY"
        or attempt.get("error_type") != "InternalScreenshotError"
        or attempt.get("error") != "internal_screenshot_not_created"
        or not isinstance(attempt.get("process_id"), int)
    ):
        _fail("live_probe_result_mismatch")
    return {
        "artifact": str(path),
        "artifact_bytes": len(payload),
        "artifact_sha256": _sha256_bytes(payload),
        "delivery_method": delivery_method,
        "result": "no_png_created_in_any_monitored_root",
        "error": attempt["error"],
        "process_id": attempt["process_id"],
    }


def derive_internal_screenshot_capability(
    *,
    runtime_executable: Path,
    post_message_probe: Path | None = None,
    send_input_probe: Path | None = None,
) -> Mapping[str, Any]:
    """Bind the static route proof to the pinned runtime and live negatives."""

    runtime_executable = runtime_executable.resolve()
    try:
        payload = runtime_executable.read_bytes()
    except OSError as error:
        raise InternalScreenshotCapabilityError(
            "runtime_executable_unreadable"
        ) from error
    runtime_sha256 = _sha256_bytes(payload)
    if len(payload) != EXPECTED_RETAIL_RUNTIME_BYTES:
        _fail("runtime_executable_size_mismatch")
    if runtime_sha256 != EXPECTED_RETAIL_RUNTIME_SHA256:
        _fail("runtime_executable_sha256_mismatch")
    static_routes = verify_runtime_static_routes(payload)

    live_probes: list[Mapping[str, Any]] = []
    if post_message_probe is not None:
        live_probes.append(
            _load_negative_live_probe(
                post_message_probe,
                delivery_method="post_message_wm_keydown_keyup",
            )
        )
    if send_input_probe is not None:
        live_probes.append(
            _load_negative_live_probe(
                send_input_probe,
                delivery_method="foreground_send_input_keydown_keyup",
            )
        )

    return {
        "schema": CAPABILITY_SCHEMA,
        "version": CAPABILITY_VERSION,
        "status": "PASS",
        "failure": None,
        "classification": CAPABILITY_CLASSIFICATION,
        "conclusion": {
            "f11_gameplay_screenshot_supported": False,
            "f11_route": "debug_image_resource_dump",
            "f11_route_guarded_by_runtime_debug_flag": True,
            "generic_png_exporter_present": True,
            "direct_static_route_from_f11_to_generic_png_exporter": False,
            "formal_capture_path": "external_dxgi_full_viewport_rgb24",
        },
        "runtime": {
            "artifact": str(runtime_executable),
            "artifact_bytes": len(payload),
            "artifact_sha256": runtime_sha256,
        },
        "static_routes": static_routes,
        "corroborating_live_probes": live_probes,
        "scope": {
            "proves": (
                "the pinned retail F11 handler directly targets the _dump "
                "image-resource exporter rather than the generic PNG exporter"
            ),
            "does_not_prove": (
                "that no indirect or otherwise undiscovered caller can ever "
                "reach the generic PNG exporter"
            ),
            "pc_golden_authorization": False,
        },
    }


def write_capability_report(report: Mapping[str, Any], output: Path) -> None:
    """Create one canonical report without replacing prior evidence."""

    output = output.resolve()
    if output.exists() or not output.parent.is_dir():
        _fail("output_path_invalid")
    encoded = (
        json.dumps(
            report,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")
    try:
        with output.open("xb") as stream:
            if stream.write(encoded) != len(encoded):
                _fail("output_write_incomplete")
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as error:
        raise InternalScreenshotCapabilityError(
            "output_write_failed"
        ) from error


__all__ = [
    "CAPABILITY_CLASSIFICATION",
    "CAPABILITY_SCHEMA",
    "CAPABILITY_VERSION",
    "EXPECTED_RETAIL_RUNTIME_BYTES",
    "EXPECTED_RETAIL_RUNTIME_SHA256",
    "InternalScreenshotCapabilityError",
    "derive_internal_screenshot_capability",
    "verify_runtime_static_routes",
    "write_capability_report",
]
