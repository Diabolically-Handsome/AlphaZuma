from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import struct

import numpy as np
import pytest

from tools.compare_pc_step_visual_trajectories import (
    StepVisualComparisonError,
    _decode_bgra_bmp,
    _process_local_topology_signature,
    _semantic_rng_frame,
    _stable_render_identity,
)
from zuma_rl.pc_rng_trajectory import (
    RngCurveList,
    RngIdentity,
    RngTrajectoryFrame,
)


def _bmp(pixels: bytes, *, width: int = 800, height: int = 600) -> bytes:
    expected = width * height * 4
    assert len(pixels) == expected
    offset = 54
    return (
        struct.pack("<2sIHHI", b"BM", offset + expected, 0, 0, offset)
        + struct.pack(
            "<IiiHHIIiiII",
            40,
            width,
            -height,
            1,
            32,
            0,
            expected,
            2835,
            2835,
            0,
            0,
        )
        + pixels
    )


def _rng_frame() -> RngTrajectoryFrame:
    identity = RngIdentity(ball_id=7, color_id=2, kind="ball")
    curve_list = RngCurveList(
        curve_index=0,
        container_offset=0x5C,
        entities=(identity,),
        topology_sha256="sha256:" + "1" * 64,
    )
    return RngTrajectoryFrame(
        update=3429,
        score=8320,
        displayed_score=8320,
        score_target=8320,
        board_color_counts=(1, 2, 3, 4, 5, 6),
        chain_ball_count=1,
        pending_ball_count=0,
        inserting_ball_count=0,
        fired_bullet_count=0,
        current_ball=identity,
        next_ball=replace(identity, ball_id=8),
        qrand_update_count=11,
        qrand_selected_index=2,
        qrand_vectors=(("weights", (1.0, 2.0)),),
        thread_crt_rand_state=1234,
        curve_lists=(curve_list,),
        mtrand_words=tuple(range(624)),
        mtrand_index=420,
    )


def test_semantic_rng_frame_excludes_only_process_local_topology_hash() -> None:
    left = _rng_frame()
    right = replace(
        left,
        curve_lists=(
            replace(
                left.curve_lists[0],
                topology_sha256="sha256:" + "2" * 64,
            ),
        ),
    )

    assert _semantic_rng_frame(left) == _semantic_rng_frame(right)
    assert _process_local_topology_signature(left) != (
        _process_local_topology_signature(right)
    )


def test_semantic_rng_frame_retains_gameplay_and_rng_fields() -> None:
    left = _rng_frame()

    assert _semantic_rng_frame(left) != _semantic_rng_frame(
        replace(left, score=left.score + 10)
    )
    assert _semantic_rng_frame(left) != _semantic_rng_frame(
        replace(left, thread_crt_rand_state=left.thread_crt_rand_state + 1)
    )
    assert _semantic_rng_frame(left) != _semantic_rng_frame(
        replace(left, mtrand_index=left.mtrand_index + 1)
    )


def _render_identity() -> dict[str, object]:
    floats = (12.5, 0.25, 0.125, -0.03125, 400.5, 588.25, 1.0, 16.0)
    return {
        "address": 0x12340000,
        "address_hex": "0x12340000",
        "vtable": 0x00960160,
        "kind": "ball",
        "ball_id": 7,
        "color_id": 2,
        "powerup_previous_type": 14,
        "powerup_primary_type": 14,
        "powerup_secondary_type": 14,
        "curve_distance": floats[0],
        "orientation_radians": floats[1],
        "previous_orientation_radians": floats[2],
        "angular_step_radians": floats[3],
        "position_x": floats[4],
        "position_y": floats[5],
        "scale": floats[6],
        "radius": floats[7],
        "render_float32_bits_hex": struct.pack("<8f", *floats).hex(),
        "flags_b4_c2_hex": bytes(range(15)).hex(),
    }


def test_stable_render_identity_excludes_addresses_but_keeps_float_bits() -> None:
    left = _render_identity()
    right = {**left, "address": 0x56780000, "address_hex": "0x56780000"}

    assert _stable_render_identity(left, code="invalid") == (
        _stable_render_identity(right, code="invalid")
    )

    right["position_y"] = 588.5
    with pytest.raises(StepVisualComparisonError, match="invalid"):
        _stable_render_identity(right, code="invalid")


def test_decode_bgra_bmp_preserves_exact_transport_bytes(
    tmp_path: Path,
) -> None:
    pixels = bytearray(800 * 600 * 4)
    pixels[:8] = bytes((1, 2, 3, 4, 5, 6, 7, 8))
    path = tmp_path / "frame.bmp"
    path.write_bytes(_bmp(bytes(pixels)))

    decoded = _decode_bgra_bmp(path)

    assert decoded.shape == (600, 800, 4)
    assert decoded.dtype == np.uint8
    assert decoded[0, 0].tolist() == [1, 2, 3, 4]
    assert decoded[0, 1].tolist() == [5, 6, 7, 8]


def test_decode_bgra_bmp_rejects_non_800x600_layout(
    tmp_path: Path,
) -> None:
    path = tmp_path / "frame.bmp"
    path.write_bytes(_bmp(bytes(4), width=1, height=1))

    with pytest.raises(
        StepVisualComparisonError,
        match="step_visual_bmp_layout_invalid",
    ):
        _decode_bgra_bmp(path)
