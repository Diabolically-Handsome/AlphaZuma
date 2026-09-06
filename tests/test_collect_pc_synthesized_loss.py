from __future__ import annotations

import json
from pathlib import Path
import struct
from typing import Any

import tools.collect_pc_synthesized_loss as synthesis


class _FakeKernel32:
    def __init__(self) -> None:
        self.closed: list[int] = []

    def CloseHandle(self, handle: int) -> None:
        self.closed.append(handle)


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
                                {
                                    "index": 0,
                                    "payload_address": 0x1000,
                                    "ball": {
                                        "ball_id": 1,
                                        "curve_distance": 3523.875,
                                        "flags_b4_c2_hex": (
                                            "010000000000000000000000000001"
                                        ),
                                    },
                                },
                                {
                                    "index": 1,
                                    "payload_address": 0x2000,
                                    "ball": {
                                        "ball_id": 176,
                                        "curve_distance": 3559.875,
                                        "flags_b4_c2_hex": (
                                            "000000000000000000000000000001"
                                        ),
                                    },
                                },
                            ],
                        }
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


def test_loss_synthesis_mutates_only_guarded_front_distance(
    monkeypatch,
    tmp_path: Path,
) -> None:
    memory: dict[int, int] = {}
    address = 0x2000 + synthesis.BALL_CURVE_DISTANCE_OFFSET
    _load_bytes(memory, address, struct.pack("<f", 3559.875))

    kernel32 = _FakeKernel32()
    monkeypatch.setattr(synthesis, "_kernel32", lambda: kernel32)
    monkeypatch.setattr(synthesis, "_open_process", lambda _kernel, _pid: 77)
    monkeypatch.setattr(
        synthesis,
        "_read_memory",
        lambda _kernel, _handle, read_address, size: _read_bytes(
            memory,
            read_address,
            size,
        ),
    )

    def write_memory(
        _kernel,
        _handle: int,
        write_address: int,
        payload: bytes,
    ) -> None:
        _load_bytes(memory, write_address, payload)

    monkeypatch.setattr(synthesis, "_write_memory", write_memory)

    result = synthesis.synthesize_loss_state(
        1234,
        _board(),
        tmp_path,
        curve_index=0,
        target_ball_id=176,
        expected_curve_distance=3559.875,
        synthesized_curve_distance=3757.0,
        decoded_curve_end=3757,
    )

    assert result["mutation_count"] == 1
    assert kernel32.closed == [77]
    assert _read_bytes(memory, address, 4) == struct.pack("<f", 3757.0)
    artifact = json.loads(
        (tmp_path / "diagnostic-mutation.json").read_text(
            encoding="ascii"
        )
    )
    assert artifact["persistent_files_modified"] is False
    assert artifact["target_ball_id"] == 176
    assert artifact["decoded_curve_end"] == 3757
    assert artifact["mutation"]["before_value"] == 3559.875
    assert artifact["mutation"]["after_value"] == 3757.0
