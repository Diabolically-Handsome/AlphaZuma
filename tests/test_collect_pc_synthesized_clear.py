from __future__ import annotations

import json
from pathlib import Path
import struct
from typing import Any

import tools.collect_pc_synthesized_clear as synthesis


class _FakeKernel32:
    def __init__(self) -> None:
        self.closed: list[int] = []

    def CloseHandle(self, handle: int) -> None:
        self.closed.append(handle)


def _record(
    *,
    index: int,
    address: int,
    ball_id: int,
    distance: float,
) -> dict[str, Any]:
    return {
        "index": index,
        "payload_address": address,
        "ball": {
            "ball_id": ball_id,
            "curve_distance": distance,
            "flags_b4_c2_hex": "000000000000000000000000000001",
        },
    }


def _board() -> dict[str, Any]:
    return {
        "curve_manager": {
            "curves": [
                {
                    "index": 0,
                    "intrusive_lists": [
                        {
                            "container_offset": 0x5C,
                            "records": [
                                _record(
                                    index=0,
                                    address=0x1000,
                                    ball_id=1,
                                    distance=31.875,
                                ),
                                _record(
                                    index=1,
                                    address=0x2000,
                                    ball_id=2,
                                    distance=67.875,
                                ),
                            ],
                        },
                        {
                            "container_offset": 0x68,
                            "records": [
                                _record(
                                    index=0,
                                    address=0x3000,
                                    ball_id=3,
                                    distance=1.0,
                                ),
                            ],
                        },
                    ],
                }
            ]
        }
    }


def _load_bytes(memory: dict[int, int], address: int, payload: bytes) -> None:
    for offset, value in enumerate(payload):
        memory[address + offset] = value


def _read_bytes(memory: dict[int, int], address: int, size: int) -> bytes:
    return bytes(memory[address + offset] for offset in range(size))


def test_clear_synthesis_validates_then_marks_every_curve_ball(
    monkeypatch,
    tmp_path: Path,
) -> None:
    memory: dict[int, int] = {}
    distance_address = 0x1000 + synthesis.BALL_CURVE_DISTANCE_OFFSET
    _load_bytes(memory, distance_address, struct.pack("<f", 31.875))
    flag_addresses = [
        address + synthesis.BALL_SHOULD_REMOVE_OFFSET
        for address in (0x1000, 0x2000, 0x3000)
    ]
    for address in flag_addresses:
        _load_bytes(memory, address, b"\x00")
    memory[0x4000] = 0xA5

    kernel32 = _FakeKernel32()
    writes: list[tuple[int, bytes]] = []
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
        writes.append((address, payload))
        _load_bytes(memory, address, payload)

    monkeypatch.setattr(synthesis, "_write_memory", write_memory)

    result = synthesis.synthesize_clear_state(
        1234,
        _board(),
        tmp_path,
        curve_index=0,
        expected_active_count=2,
        expected_pending_count=1,
        expected_rear_curve_distance=31.875,
        synthesized_rear_curve_distance=40.0,
    )

    assert result["mutation_count"] == 4
    assert kernel32.closed == [77]
    assert _read_bytes(memory, distance_address, 4) == struct.pack("<f", 40.0)
    assert [_read_bytes(memory, address, 1) for address in flag_addresses] == [
        b"\x01",
        b"\x01",
        b"\x01",
    ]
    assert memory[0x4000] == 0xA5
    assert len(writes) == 4
    artifact = json.loads(
        (tmp_path / "diagnostic-mutation.json").read_text(
            encoding="ascii"
        )
    )
    assert artifact["persistent_files_modified"] is False
    assert artifact["active_ball_count"] == 2
    assert artifact["pending_ball_count"] == 1
    assert len(artifact["mutations"]) == 4
