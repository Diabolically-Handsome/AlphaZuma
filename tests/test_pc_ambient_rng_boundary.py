from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct

import pytest

from zuma_rl.pc_ambient_rng_boundary import (
    AMBIENT_RNG_BOUNDARY_CLASSIFICATION,
    AMBIENT_RNG_BOUNDARY_SCHEMA,
    AMBIENT_RNG_BOUNDARY_VERSION,
    EXPECTED_BOUNDARY_CALLERS,
    GLOBAL_MTRAND_WRAPPER_VA,
    SHADOW_CANOPY_CALLERS,
    SHADOW_CANOPY_NAME_VA,
    SHADOW_CANOPY_UPDATE_VA,
    SHADOW_CANOPY_VTABLE_SLOT_VA,
    AmbientRngBoundaryError,
    _select_boundary_calls,
    _verify_shadow_canopy_runtime,
    bind_ambient_rng_boundary_to_diff,
    load_ambient_rng_boundary_report,
    mtrand_state_sha256,
)
from zuma_rl.pc_memory_trajectory import TrajectoryMTRandState


def _hash(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode("ascii")).hexdigest()


def test_mtrand_state_sha256_uses_exact_retail_memory_layout() -> None:
    state = TrajectoryMTRandState(
        index=17,
        words=tuple(range(624)),
    )

    expected = "sha256:" + hashlib.sha256(
        struct.pack("<625I", *range(624), 17)
    ).hexdigest()

    assert mtrand_state_sha256(state) == expected


def test_select_boundary_calls_requires_exact_shadow_canopy_suffix() -> None:
    hashes = [_hash(f"state-{index}") for index in range(6)]
    calls = tuple(
        {
            "order": 100 + index,
            "framework_update": 3480,
            "native_game_time": 468,
            "score": 7950,
            "score_target": 9650,
            "caller": caller,
            "pre_state_sha256": hashes[index],
            "post_state_sha256": hashes[index + 1],
        }
        for index, caller in enumerate(EXPECTED_BOUNDARY_CALLERS)
    )

    selected = _select_boundary_calls(
        calls,
        first_excluded_update=3480,
        before_state_sha256=hashes[0],
        after_state_sha256=hashes[-1],
        native_game_time=468,
        score=7950,
        score_target=9650,
    )

    assert tuple(row["caller"] for row in selected[-2:]) == (
        SHADOW_CANOPY_CALLERS
    )

    altered = [dict(row) for row in calls]
    altered[-1]["caller"] += 1
    with pytest.raises(
        AmbientRngBoundaryError,
        match="ambient_boundary_call_sequence_mismatch",
    ):
        _select_boundary_calls(
            altered,
            first_excluded_update=3480,
            before_state_sha256=hashes[0],
            after_state_sha256=hashes[-1],
            native_game_time=468,
            score=7950,
            score_target=9650,
        )


def _synthetic_retail_pe() -> bytes:
    image_base = 0x00400000
    section_va = 0x1000
    maximum_va = max(
        SHADOW_CANOPY_NAME_VA + len(b"ShadowCanopy1\0"),
        SHADOW_CANOPY_VTABLE_SLOT_VA + 4,
        max(SHADOW_CANOPY_CALLERS),
    )
    raw_size = maximum_va - image_base - section_va + 0x100
    raw_offset = 0x200
    payload = bytearray(raw_offset + raw_size)
    pe_offset = 0x80
    struct.pack_into("<I", payload, 0x3C, pe_offset)
    payload[pe_offset : pe_offset + 4] = b"PE\0\0"
    struct.pack_into("<H", payload, pe_offset + 6, 1)
    struct.pack_into("<H", payload, pe_offset + 20, 0xE0)
    optional = pe_offset + 24
    struct.pack_into("<H", payload, optional, 0x10B)
    struct.pack_into("<I", payload, optional + 28, image_base)
    section = optional + 0xE0
    payload[section : section + 8] = b".all\0\0\0\0"
    struct.pack_into(
        "<IIII",
        payload,
        section + 8,
        raw_size,
        section_va,
        raw_size,
        raw_offset,
    )

    def offset(virtual_address: int) -> int:
        return raw_offset + virtual_address - image_base - section_va

    payload[
        offset(SHADOW_CANOPY_UPDATE_VA) :
        offset(SHADOW_CANOPY_UPDATE_VA) + 3
    ] = b"\x55\x8b\xec"
    payload[
        offset(SHADOW_CANOPY_NAME_VA) :
        offset(SHADOW_CANOPY_NAME_VA) + len(b"ShadowCanopy1\0")
    ] = b"ShadowCanopy1\0"
    struct.pack_into(
        "<I",
        payload,
        offset(SHADOW_CANOPY_VTABLE_SLOT_VA),
        SHADOW_CANOPY_UPDATE_VA,
    )
    for return_address in SHADOW_CANOPY_CALLERS:
        call_address = return_address - 5
        displacement = GLOBAL_MTRAND_WRAPPER_VA - return_address
        payload[offset(call_address)] = 0xE8
        struct.pack_into(
            "<i",
            payload,
            offset(call_address) + 1,
            displacement,
        )
    return bytes(payload)


def test_shadow_canopy_static_proof_binds_vtable_name_and_calls() -> None:
    proof = _verify_shadow_canopy_runtime(_synthetic_retail_pe())

    assert proof["class_name"] == "ShadowCanopy1"
    assert proof["vtable_slot_target"] == SHADOW_CANOPY_UPDATE_VA
    assert [row["return_address"] for row in proof["call_sites"]] == list(
        SHADOW_CANOPY_CALLERS
    )
    assert all(
        row["target_address"] == GLOBAL_MTRAND_WRAPPER_VA
        for row in proof["call_sites"]
    )


def _boundary_report() -> dict[str, object]:
    return {
        "schema": AMBIENT_RNG_BOUNDARY_SCHEMA,
        "version": AMBIENT_RNG_BOUNDARY_VERSION,
        "status": "PASS",
        "failure": None,
        "classification": AMBIENT_RNG_BOUNDARY_CLASSIFICATION,
        "last_included_update": 3479,
        "first_excluded_update": 3480,
        "scope": {
            "excluded": "shadow_canopy_ambient_visual_rng_consumers"
        },
        "trajectory": {"artifact_sha256": _hash("trajectory")},
        "replay_dmo": {"artifact_sha256": _hash("dmo")},
    }


def test_boundary_report_binds_exact_diff_source_and_end(tmp_path: Path) -> None:
    path = tmp_path / "boundary.json"
    path.write_text(
        json.dumps(_boundary_report(), sort_keys=True) + "\n",
        encoding="ascii",
    )
    report = load_ambient_rng_boundary_report(path)

    binding = bind_ambient_rng_boundary_to_diff(
        report,
        boundary_path=path,
        trajectory_sha256=_hash("trajectory"),
        dmo_sha256=_hash("dmo"),
        selected_end_update=3479,
    )

    assert binding["status"] == "PASS"
    assert binding["first_excluded_update"] == 3480

    with pytest.raises(
        AmbientRngBoundaryError,
        match="ambient_boundary_diff_binding_mismatch",
    ):
        bind_ambient_rng_boundary_to_diff(
            report,
            boundary_path=path,
            trajectory_sha256=_hash("different-trajectory"),
            dmo_sha256=_hash("dmo"),
            selected_end_update=3479,
        )
