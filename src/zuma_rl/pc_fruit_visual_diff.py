"""Exact retail-Board versus simulator fruit-visual transition proof."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from zuma_rl.pc_gameplay_diff import (
    _fruit_state_mismatches,
    _restore_fruit_state,
)
from zuma_rl.pc_memory_trajectory import (
    PcMemoryTrajectoryError,
    TrajectoryFrame,
)
from zuma_rl.revenge_core import RevengeSimulator


FRUIT_VISUAL_DIFF_SCHEMA = "zuma-rl.pc-fruit-visual-simulator-diff"
FRUIT_VISUAL_DIFF_VERSION = 2


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def compare_fruit_visual_trajectory(
    frames: Sequence[TrajectoryFrame],
    *,
    original_root: str | Path,
    level_id: str = "Jungle2",
    hard: bool = False,
    curve_index: int = 0,
) -> Mapping[str, Any]:
    """Replay only the fruit visual update against content-addressed Boards."""

    if len(frames) < 2:
        raise PcMemoryTrajectoryError("fruit_visual_diff_window_too_short")
    if any(frame.fruit_state is None for frame in frames):
        raise PcMemoryTrajectoryError("fruit_visual_diff_fruit_state_missing")
    if any(frame.native_game_time is None for frame in frames):
        raise PcMemoryTrajectoryError("fruit_visual_diff_native_time_missing")
    if any(frame.board_update_count is None for frame in frames):
        raise PcMemoryTrajectoryError(
            "fruit_visual_diff_board_update_count_missing"
        )
    simulator = RevengeSimulator.from_installed(
        level_id,
        root=Path(original_root),
        hard=hard,
        curve_index=curve_index,
        seed=0,
    )
    calibration = simulator.fruit_calibration
    if calibration is None:
        raise PcMemoryTrajectoryError("fruit_visual_diff_calibration_missing")
    first = frames[0]
    assert first.fruit_state is not None
    assert first.native_game_time is not None
    assert first.board_update_count is not None
    _restore_fruit_state(simulator, first.fruit_state)
    simulator.tick_count = first.board_update_count
    simulator.native_game_time = first.native_game_time

    transcript: list[dict[str, Any]] = [
        {
            "framework_update": first.update,
            "native_game_time": first.native_game_time,
            "retail_state": first.fruit_state.to_dict(),
        }
    ]
    mismatch_rows: list[dict[str, Any]] = []
    exact_transition_count = 0
    for previous, frame in zip(frames, frames[1:], strict=False):
        assert frame.fruit_state is not None
        assert frame.native_game_time is not None
        assert frame.board_update_count is not None
        assert previous.native_game_time is not None
        assert previous.board_update_count is not None
        if (
            frame.update != previous.update + 1
            or frame.native_game_time != previous.native_game_time + 1
            or frame.board_update_count
            != previous.board_update_count + 1
        ):
            raise PcMemoryTrajectoryError(
                "fruit_visual_diff_frames_not_contiguous"
            )
        simulator.tick_count = frame.board_update_count
        simulator.native_game_time = frame.native_game_time
        simulator._update_fruit_visual()
        mismatches = _fruit_state_mismatches(
            simulator,
            frame.fruit_state,
        )
        if mismatches:
            if len(mismatch_rows) < 32:
                mismatch_rows.append(
                    {
                        "framework_update": frame.update,
                        "native_game_time": frame.native_game_time,
                        "mismatches": mismatches,
                    }
                )
        else:
            exact_transition_count += 1
        transcript.append(
            {
                "framework_update": frame.update,
                "native_game_time": frame.native_game_time,
                "retail_state": frame.fruit_state.to_dict(),
            }
        )

    states = [frame.fruit_state for frame in frames]
    assert all(state is not None for state in states)
    typed_states = [state for state in states if state is not None]
    transition_count = len(frames) - 1
    status = "PASS" if exact_transition_count == transition_count else "FAIL"
    return {
        "schema": FRUIT_VISUAL_DIFF_SCHEMA,
        "version": FRUIT_VISUAL_DIFF_VERSION,
        "status": status,
        "level_id": level_id,
        "hard": hard,
        "curve_index": curve_index,
        "start_update": first.update,
        "end_update": frames[-1].update,
        "start_native_game_time": first.native_game_time,
        "end_native_game_time": frames[-1].native_game_time,
        "start_board_update_count": first.board_update_count,
        "end_board_update_count": frames[-1].board_update_count,
        "frame_count": len(frames),
        "transition_count": transition_count,
        "exact_transition_count": exact_transition_count,
        "mismatch_transition_count": transition_count - exact_transition_count,
        "mismatch_rows": mismatch_rows,
        "comparison_contract": {
            "update_phase": "Board fruit visual update before projectile iteration",
            "animation_clock": "Board+0x028 update counter",
            "float32_policy": "little-endian IEEE-754 bit exact",
            "integer_policy": "exact",
            "float32_fields": [
                "velocity",
                "max_velocity",
                "acceleration",
                "vertical_offset",
                "lower_bound",
                "upper_bound",
            ],
            "integer_fields": [
                "active_point_pointer_presence",
                "selected_point_index",
                "collecting",
                "glow_alpha",
                "glow_step",
                "alpha",
                "expiry_time",
                "cell_index",
            ],
        },
        "retail_board_offsets": {
            "active_point_pointer": "0x0b4",
            "selected_point_index": "0x118",
            "lower_bound": "0x130",
            "upper_bound": "0x134",
            "collecting": "0x153",
            "velocity": "0xe90",
            "max_velocity": "0xe94",
            "acceleration": "0xe98",
            "vertical_offset": "0xe9c",
            "glow_alpha": "0xeac",
            "glow_step": "0xeb0",
            "alpha": "0xeb4",
            "expiry_time": "0xeb8",
            "cell_index": "0xebc",
            "native_game_time": "0xec8",
            "board_update_count": "0x028",
        },
        "calibration": {
            "fruit_type": calibration.fruit_type,
            "logical_width": calibration.logical_width,
            "logical_height": calibration.logical_height,
            "sheet_columns": calibration.sheet_columns,
            "sheet_rows": calibration.sheet_rows,
            "collection_animation_frames": (
                calibration.collection_animation_frames
            ),
            "collection_animation_fps": calibration.collection_animation_fps,
            "provenance": calibration.provenance,
        },
        "coverage": {
            "active_frame_count": sum(
                state.active for state in typed_states
            ),
            "collecting_frame_count": sum(
                state.collecting for state in typed_states
            ),
            "vertical_offset_min": min(
                state.vertical_offset for state in typed_states
            ),
            "vertical_offset_max": max(
                state.vertical_offset for state in typed_states
            ),
            "cell_index_min": min(state.cell_index for state in typed_states),
            "cell_index_max": max(state.cell_index for state in typed_states),
        },
        "initial_state": first.fruit_state.to_dict(),
        "final_state": frames[-1].fruit_state.to_dict(),
        "retail_state_transcript_sha256": _sha256_bytes(
            _canonical_bytes(transcript)
        ),
    }


def canonical_fruit_visual_report_bytes(report: Mapping[str, Any]) -> bytes:
    return _canonical_bytes(report)


__all__ = [
    "FRUIT_VISUAL_DIFF_SCHEMA",
    "FRUIT_VISUAL_DIFF_VERSION",
    "canonical_fruit_visual_report_bytes",
    "compare_fruit_visual_trajectory",
]
