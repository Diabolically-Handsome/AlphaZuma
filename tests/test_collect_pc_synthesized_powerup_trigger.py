from __future__ import annotations

import json
from pathlib import Path
import struct
from typing import Any

import tools.collect_pc_synthesized_powerup_trigger as synthesis


class _FakeKernel32:
    def __init__(self) -> None:
        self.closed: list[int] = []

    def CloseHandle(self, handle: int) -> None:
        self.closed.append(handle)


def _board() -> dict[str, Any]:
    return {
        "primary_child": {
            "bullets": [
                {
                    "shooter_pointer_offset": 0x130,
                    "address": 0x1000,
                    "ball": {"ball_id": 119, "color_id": 0},
                },
                {
                    "shooter_pointer_offset": 0x134,
                    "address": 0x1100,
                    "ball": {"ball_id": 122, "color_id": 1},
                },
            ]
        },
        "curve_manager": {
            "curves": [
                {
                    "index": 0,
                    "address": 0x3000,
                    "intrusive_lists": [
                        {
                            "container_offset": 0x5C,
                            "records": [
                                {
                                    "index": 86,
                                    "payload_address": 0x2000,
                                    "ball": {
                                        "ball_id": 29,
                                        "color_id": 3,
                                    },
                                }
                            ],
                        }
                    ],
                }
            ]
        },
    }


def _load_bytes(memory: dict[int, int], address: int, payload: bytes) -> None:
    for offset, value in enumerate(payload):
        memory[address + offset] = value


def _read_bytes(memory: dict[int, int], address: int, size: int) -> bytes:
    return bytes(memory[address + offset] for offset in range(size))


def test_synthesis_is_exactly_guarded_and_emits_byte_audit(
    monkeypatch,
    tmp_path: Path,
) -> None:
    memory: dict[int, int] = {}
    expected_fields = {
        0x1000 + synthesis.BALL_COLOR_OFFSET: struct.pack("<i", 0),
        0x2000
        + synthesis.BALL_POWERUP_PREVIOUS_TICKS_OFFSET: struct.pack("<i", 0),
        0x2000
        + synthesis.BALL_POWERUP_PREVIOUS_TYPE_OFFSET: struct.pack("<i", 14),
        0x2000
        + synthesis.BALL_POWERUP_LIFETIME_OFFSET: struct.pack("<i", 0),
        0x2000
        + synthesis.BALL_POWERUP_TRANSITION_OFFSET: struct.pack("<i", 0),
        0x2000
        + synthesis.BALL_POWERUP_VISUAL_SCALE_OFFSET: struct.pack("<f", 1.0),
        0x2000
        + synthesis.BALL_POWERUP_VISUAL_STEP_OFFSET: struct.pack("<f", 0.0),
        0x2000
        + synthesis.BALL_POWERUP_VISUAL_INDEX_OFFSET: struct.pack("<i", -1),
        0x2000
        + synthesis.BALL_POWERUP_PRIMARY_TYPE_OFFSET: struct.pack("<i", 14),
        0x2000
        + synthesis.BALL_POWERUP_SECONDARY_TYPE_OFFSET: struct.pack("<i", 14),
        0x3000
        + synthesis.CURVE_ACTIVE_COLOR_COUNTS_OFFSET
        + 3 * 4: struct.pack("<i", 0),
    }
    for address, payload in expected_fields.items():
        _load_bytes(memory, address, payload)

    kernel32 = _FakeKernel32()
    monkeypatch.setattr(synthesis, "_kernel32", lambda: kernel32)
    monkeypatch.setattr(synthesis, "_open_process", lambda _kernel, _pid: 77)
    monkeypatch.setattr(
        synthesis,
        "_read_memory",
        lambda _kernel, _handle, address, size: _read_bytes(
            memory,
            address,
            size,
        ),
    )

    def write_memory(
        _kernel,
        _handle: int,
        address: int,
        payload: bytes,
    ) -> None:
        _load_bytes(memory, address, payload)

    monkeypatch.setattr(synthesis, "_write_memory", write_memory)

    result = synthesis.synthesize_powerup_trigger_state(
        1234,
        _board(),
        tmp_path,
        curve_index=0,
        target_ball_id=29,
        expected_current_color=0,
        synthesized_current_color=3,
        expected_target_color=3,
        powerup_type=3,
        powerup_lifetime=500,
    )

    assert result["mutation_count"] == 11
    assert kernel32.closed == [77]
    assert _read_bytes(
        memory,
        0x1000 + synthesis.BALL_COLOR_OFFSET,
        4,
    ) == struct.pack("<i", 3)
    assert _read_bytes(
        memory,
        0x2000 + synthesis.BALL_POWERUP_PRIMARY_TYPE_OFFSET,
        4,
    ) == struct.pack("<i", 3)
    assert _read_bytes(
        memory,
        0x2000 + synthesis.BALL_POWERUP_LIFETIME_OFFSET,
        4,
    ) == struct.pack("<i", 500)
    assert _read_bytes(
        memory,
        0x3000 + synthesis.CURVE_ACTIVE_COLOR_COUNTS_OFFSET + 3 * 4,
        4,
    ) == struct.pack("<i", 1)

    artifact = json.loads(
        (tmp_path / "diagnostic-mutation.json").read_text(encoding="ascii")
    )
    assert artifact["classification"] == (
        "diagnostic-not-unmodified-pc-evidence"
    )
    assert artifact["persistent_files_modified"] is False
    assert artifact["target_ball_id"] == 29
    assert len(artifact["mutations"]) == 11
    assert all(
        mutation["before_hex"] and mutation["after_hex"]
        for mutation in artifact["mutations"]
    )
