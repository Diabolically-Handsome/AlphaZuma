from __future__ import annotations

import json
from pathlib import Path
import struct

import pytest

from tools.popcap_qrand_restore import (
    QRAND_OBJECT_SIZE,
    QRandRestoreState,
    load_qrand_restore_state,
    qrand_scalar_payload,
    qrand_vector_write_plan,
)


def _state() -> QRandRestoreState:
    return QRandRestoreState(
        source_path=Path("rng.ndjson"),
        source_sha256="sha256:" + "0" * 64,
        source_framework_update=1759,
        source_native_game_time=435,
        update_count=2,
        selected_index=2,
        vectors={
            "weights": (0.0, 0.0, 0.5, 0.5, 0.0, 0.0),
            "sways": (0.0, 0.0, 0.25, 0.4375, 0.0, 0.0),
            "last_hit": (0, 0, 2, 0, 0, 0),
            "previous_hit": (0, 0, 1, 0, 0, 0),
        },
    )


def _object_bytes() -> bytes:
    payload = bytearray(QRAND_OBJECT_SIZE)
    for offset, address in zip(
        (0x08, 0x18, 0x28, 0x38),
        (0x1000, 0x2000, 0x3000, 0x4000),
        strict=True,
    ):
        struct.pack_into(
            "<III",
            payload,
            offset + 4,
            address,
            address + 24,
            address + 32,
        )
    return bytes(payload)


def test_qrand_write_plan_binds_all_four_live_vectors() -> None:
    writes = qrand_vector_write_plan(_object_bytes(), _state())

    assert [address for address, _ in writes] == [
        0x1000,
        0x2000,
        0x3000,
        0x4000,
    ]
    assert all(len(payload) == 24 for _, payload in writes)
    assert qrand_scalar_payload(_state()) == struct.pack("<ii", 2, 2)


def test_qrand_write_plan_rejects_live_length_drift() -> None:
    payload = bytearray(_object_bytes())
    struct.pack_into("<I", payload, 0x08 + 8, 0x1000 + 20)

    with pytest.raises(ValueError, match="weights vector layout"):
        qrand_vector_write_plan(bytes(payload), _state())


def test_qrand_write_plan_accepts_uninitialized_empty_vectors() -> None:
    state = QRandRestoreState(
        source_path=Path("rng.ndjson"),
        source_sha256="sha256:" + "0" * 64,
        source_framework_update=1740,
        source_native_game_time=417,
        update_count=0,
        selected_index=-1,
        vectors={
            "weights": (),
            "sways": (),
            "last_hit": (),
            "previous_hit": (),
        },
    )

    assert qrand_vector_write_plan(bytes(QRAND_OBJECT_SIZE), state) == ()
    assert qrand_scalar_payload(state) == struct.pack("<ii", 0, -1)


def test_load_qrand_restore_state_requires_unique_read_only_row(
    tmp_path: Path,
) -> None:
    path = tmp_path / "rng.ndjson"
    header = {
        "schema": "zuma-rl.live-rng-change-monitor",
        "version": 1,
        "process_memory_writes": 0,
    }
    row = {
        "type": "rng",
        "framework_update": 1759,
        "native_game_time": 435,
        "qrand": _state().semantic_dict(),
    }
    path.write_text(
        json.dumps(header) + "\n" + json.dumps(row) + "\n",
        encoding="utf-8",
    )

    observed = load_qrand_restore_state(
        path,
        framework_update=1759,
    )

    assert observed.semantic_dict() == _state().semantic_dict()
    assert observed.semantic_sha256.startswith("sha256:")


def test_load_qrand_restore_state_rejects_mutating_source(
    tmp_path: Path,
) -> None:
    path = tmp_path / "rng.ndjson"
    path.write_text(
        json.dumps(
            {
                "schema": "zuma-rl.live-rng-change-monitor",
                "version": 1,
                "process_memory_writes": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="header contract"):
        load_qrand_restore_state(path, framework_update=1759)
