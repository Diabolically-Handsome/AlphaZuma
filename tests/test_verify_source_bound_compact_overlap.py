from __future__ import annotations

import struct

import pytest

from tools.verify_source_bound_compact_overlap import (
    BULLET_VTABLE,
    OverlapVerificationError,
    _ball_projection,
    _difference_paths,
)


def _bullet() -> dict[str, object]:
    floats = (0.0, 1.25, 1.0, 0.0, 320.5, 240.25, 1.0, 18.0)
    return {
        "ball_id": 47,
        "color_id": 1,
        "curve_distance": floats[0],
        "orientation_radians": floats[1],
        "previous_orientation_radians": floats[2],
        "angular_step_radians": floats[3],
        "position_x": floats[4],
        "position_y": floats[5],
        "scale": floats[6],
        "radius": floats[7],
        "powerup_previous_type": 14,
        "powerup_primary_type": 14,
        "powerup_secondary_type": 14,
        "flags_b4_c2_hex": "00" * 15,
        "render_float32_bits_hex": struct.pack("<8f", *floats).hex(),
    }


def test_ball_projection_recomputes_render_bits() -> None:
    bullet = _bullet()

    projected = _ball_projection(
        bullet,
        kind="bullet",
        vtable=BULLET_VTABLE,
        verify_render_bits=True,
    )

    assert projected["ball_id"] == 47
    assert projected["position_x"] == 320.5
    assert "render_float32_bits_hex" not in projected


def test_ball_projection_rejects_false_render_bits() -> None:
    bullet = _bullet()
    bullet["render_float32_bits_hex"] = "00" * 32

    with pytest.raises(OverlapVerificationError, match="ball_render_bits_mismatch"):
        _ball_projection(
            bullet,
            kind="bullet",
            vtable=BULLET_VTABLE,
            verify_render_bits=True,
        )


def test_difference_paths_are_leaf_exact() -> None:
    left = {"a": [1, {"b": 2}], "c": True}
    right = {"a": [1, {"b": 3}], "c": True}

    assert _difference_paths(left, right) == ["/a/1/b"]
