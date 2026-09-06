from __future__ import annotations

import json
from pathlib import Path
import struct
from typing import Any

import pytest


pytest.importorskip("PIL")
from PIL import Image, ImageDraw  # noqa: E402

from tools.validate_pc_ball_pixels import (  # noqa: E402
    BALL_OBJECT_SIZE,
    BALL_VTABLE,
    BULLET_OBJECT_SIZE,
    BULLET_VTABLE,
    PixelValidationError,
    validate,
)


_RGB = {
    0: (0, 90, 255),
    1: (255, 225, 0),
    2: (255, 0, 0),
    3: (0, 255, 0),
    4: (190, 0, 255),
}


def _object(
    *,
    size: int,
    vtable: int,
    ball_id: int,
    color_id: int,
    x: float,
    y: float,
) -> bytes:
    payload = bytearray(size)
    struct.pack_into("<I", payload, 0x00, vtable)
    struct.pack_into("<I", payload, 0x10, ball_id)
    struct.pack_into("<i", payload, 0x14, color_id)
    struct.pack_into("<ff", payload, 0x2C, x, y)
    return bytes(payload)


def _records(
    values: list[tuple[int, int, float, float]],
    *,
    object_size: int,
) -> list[dict[str, Any]]:
    return [
        {
            "artifact_offset": index * object_size,
            "artifact_bytes": object_size,
        }
        for index, _ in enumerate(values)
    ]


def _case(
    tmp_path: Path,
    *,
    fired_memory_color: int = 3,
) -> Path:
    active_values = [
        (10, 0, 35.0, 130.0),
        (11, 1, 80.0, 130.0),
        (12, 2, 125.0, 130.0),
    ]
    fired_values = [(104, fired_memory_color, 170.0, 60.0)]
    active_payload = b"".join(
        _object(
            size=BALL_OBJECT_SIZE,
            vtable=BALL_VTABLE,
            ball_id=ball_id,
            color_id=color_id,
            x=x,
            y=y,
        )
        for ball_id, color_id, x, y in active_values
    )
    fired_payload = b"".join(
        _object(
            size=BULLET_OBJECT_SIZE,
            vtable=BULLET_VTABLE,
            ball_id=ball_id,
            color_id=color_id,
            x=x,
            y=y,
        )
        for ball_id, color_id, x, y in fired_values
    )
    (tmp_path / "active.bin").write_bytes(active_payload)
    (tmp_path / "fired.bin").write_bytes(fired_payload)

    image = Image.new("RGB", (210, 170), (12, 12, 12))
    draw = ImageDraw.Draw(image)
    for _, color_id, x, y in active_values:
        draw.ellipse(
            (x - 15, y - 15, x + 15, y + 15),
            fill=_RGB[color_id],
        )
    fired_render_color = 3
    _, _, x, y = fired_values[0]
    draw.ellipse(
        (x - 15, y - 15, x + 15, y + 15),
        fill=_RGB[fired_render_color],
    )
    image.save(tmp_path / "frame.bmp")

    probe = {
        "schema": "zuma-rl.pc-memory-int32-probe",
        "framework_update": 7750,
        "frozen_frame": {"artifact": "frame.bmp"},
        "active_board": {
            "curve_manager": {
                "curves": [
                    {
                        "intrusive_lists": [
                            {
                                "container_offset": 0x5C,
                                "artifact": "active.bin",
                                "payload_count": len(active_values),
                                "records": _records(
                                    active_values,
                                    object_size=BALL_OBJECT_SIZE,
                                ),
                            }
                        ]
                    }
                ]
            },
            "fired_bullets": {
                "artifact": "fired.bin",
                "board_container_offset_hex": "0x698",
                "declared_count": len(fired_values),
                "traversed_count": len(fired_values),
                "records": _records(
                    fired_values,
                    object_size=BULLET_OBJECT_SIZE,
                ),
            },
        },
    }
    path = tmp_path / "memory-probe.json"
    path.write_text(json.dumps(probe), encoding="ascii")
    return path


def test_pixel_validator_checks_active_and_fired_bullets(
    tmp_path: Path,
) -> None:
    report = validate(
        _case(tmp_path),
        minimum_visible=3,
        maximum_mismatches=0,
        minimum_visible_fired_bullets=1,
        maximum_fired_bullet_mismatches=0,
        verify_raw_payloads=False,
    )

    assert report["status"] == "PASS"
    assert report["version"] == 2
    assert report["visible_count"] == 3
    assert report["match_count"] == 3
    assert report["fired_bullets"]["visible_count"] == 1
    assert report["fired_bullets"]["match_count"] == 1
    assert report["fired_bullets"]["results"][0]["ball_id"] == 104


def test_pixel_validator_rejects_fired_bullet_color_mismatch(
    tmp_path: Path,
) -> None:
    path = _case(tmp_path, fired_memory_color=2)

    with pytest.raises(
        PixelValidationError,
        match="fired_bullet_pixel_mismatch_limit_exceeded",
    ):
        validate(
            path,
            minimum_visible=3,
            maximum_mismatches=0,
            minimum_visible_fired_bullets=1,
            maximum_fired_bullet_mismatches=0,
            verify_raw_payloads=False,
        )
