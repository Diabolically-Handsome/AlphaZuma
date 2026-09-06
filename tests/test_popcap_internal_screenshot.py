from __future__ import annotations

import struct
import zlib

import pytest

from tools.popcap_internal_screenshot import (
    InternalScreenshotError,
    VK_F11,
    _png_facts,
    capture_internal_screenshot,
)


def _chunk(name: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(name)
    crc = zlib.crc32(payload, crc)
    return (
        struct.pack(">I", len(payload))
        + name
        + payload
        + struct.pack(">I", crc & 0xFFFFFFFF)
    )


def _png(width: int = 800, height: int = 600) -> bytes:
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    rows = b"".join(b"\x00" + b"\x00" * (width * 3) for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(rows))
        + _chunk(b"IEND", b"")
    )


def test_png_facts_require_800_by_600_crc_valid_image() -> None:
    assert _png_facts(_png())["color_type"] == 2
    with pytest.raises(
        InternalScreenshotError,
        match="internal_screenshot_png_geometry_invalid",
    ):
        _png_facts(_png(width=799))
    corrupted = bytearray(_png())
    corrupted[-1] ^= 1
    with pytest.raises(
        InternalScreenshotError,
        match="internal_screenshot_png_crc_invalid",
    ):
        _png_facts(bytes(corrupted))


def test_capture_copies_and_removes_only_new_retail_png(tmp_path) -> None:
    root = tmp_path / "_screenshots"
    output = tmp_path / "evidence" / "frame.png"
    output.parent.mkdir()
    observed: list[tuple[int, int]] = []

    def post_key(hwnd: int, key: int) -> None:
        observed.append((hwnd, key))
        root.mkdir()
        (root / "1.png").write_bytes(_png())

    receipt = capture_internal_screenshot(
        window_handle=123,
        output=output,
        screenshot_root=root,
        post_key=post_key,
        delay=lambda unused: None,
    )

    assert observed == [(123, VK_F11)]
    assert output.read_bytes() == _png()
    assert receipt["png"]["width"] == 800
    assert receipt["source_file_removed_after_verified_copy"] is True
    assert receipt["new_source_directory_removed"] is True
    assert not root.exists()


def test_capture_preserves_preexisting_screenshot_files(tmp_path) -> None:
    root = tmp_path / "_screenshots"
    root.mkdir()
    original = root / "7.png"
    original.write_bytes(b"preexisting")
    output = tmp_path / "frame.png"

    def post_key(unused_hwnd: int, unused_key: int) -> None:
        (root / "8.png").write_bytes(_png())

    receipt = capture_internal_screenshot(
        window_handle=123,
        output=output,
        screenshot_root=root,
        post_key=post_key,
        delay=lambda unused: None,
    )

    assert original.read_bytes() == b"preexisting"
    assert not (root / "8.png").exists()
    assert root.is_dir()
    assert receipt["preexisting_file_count"] == 1
    assert receipt["new_source_directory_removed"] is False
