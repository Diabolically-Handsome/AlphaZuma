from __future__ import annotations

import struct

import pytest

from zuma_rl.pc_mtrand_reconciliation import (
    MTRandReconciliationProofError,
    _call_requires_pil_particle_static_surface,
    _verify_pil_particle_runtime_surface,
    classify_native_call,
    reconciliation_transcript_sha256,
)


def _call(caller: int, stack: list[int]) -> dict[str, object]:
    return {
        "caller": caller,
        "caller_hex": f"0x{caller:08x}",
        "stack_code": [
            {
                "address": address,
                "address_hex": f"0x{address:08x}",
                "stack_offset": offset * 4,
            }
            for offset, address in enumerate(stack)
        ],
    }


@pytest.mark.parametrize(
    ("call", "expected"),
    [
        (
            _call(0x004B5ADF, [0x004B5ADF, 0x0041C490]),
            ("modeled_gameplay", "fruit_chance_scheduler"),
        ),
        (
            _call(
                0x0045CEE0,
                [0x0045CEE0, 0x0045E195, 0x0065DE83],
            ),
            ("modeled_gameplay", "powerup_spawn_chance_roll"),
        ),
        (
            _call(
                0x00458CD1,
                [0x00458CD1, 0x00459031, 0x0045DE5E, 0x0065DE83],
            ),
            ("modeled_gameplay", "pending_ball_repeat_roll"),
        ),
        (
            _call(
                0x00402111,
                [
                    0x00402111,
                    0x00458D6A,
                    0x00459031,
                    0x0045DE5E,
                    0x0065DE83,
                ],
            ),
            ("modeled_gameplay", "pending_ball_visual_frame_roll"),
        ),
        (
            _call(0x0040CADF, [0x0040CADF, 0x004171AD]),
            ("omitted_visual", "score_digit_transition"),
        ),
        (
            _call(
                0x0040CDDE,
                [0x0040CDDE, 0x0041FD7D, 0x004227F5],
            ),
            ("omitted_visual", "score_digit_wobble"),
        ),
        (
            _call(
                0x0065B821,
                [0x0065B821],
            ),
            ("omitted_visual", "pi_effect_particle_update"),
        ),
        (
            _call(
                0x0040AE62,
                [0x0040AE62, 0x005A6496, 0x004B6A68],
            ),
            ("omitted_visual", "shadow_canopy_animation"),
        ),
        (
            _call(0x005A6630, [0x005A6630, 0x004B6A68]),
            ("omitted_visual", "shadow_canopy_animation"),
        ),
    ],
)
def test_native_call_classification_is_stack_bound(
    call: dict[str, object],
    expected: tuple[str, str],
) -> None:
    assert classify_native_call(call) == expected


@pytest.mark.parametrize(
    ("call", "expected"),
    [
        (
            _call(
                0x00458CD1,
                [0x00458CD1, 0x0092E91B, 0x00459031, 0x0045DE5E],
            ),
            ("modeled_gameplay", "pending_ball_repeat_roll"),
        ),
        (
            _call(
                0x00402111,
                [
                    0x00402111,
                    0x00458D6A,
                    0x0092E91B,
                    0x00459031,
                    0x0045DE5E,
                ],
            ),
            ("modeled_gameplay", "pending_ball_visual_frame_roll"),
        ),
    ],
)
def test_pending_ball_classification_accepts_stable_near_stack(
    call: dict[str, object],
    expected: tuple[str, str],
) -> None:
    assert classify_native_call(call) == expected


@pytest.mark.parametrize(
    ("caller", "stack"),
    [
        (0x0040AE62, [0x0040AE62, 0x00901E46]),
        (0x0040AE62, [0x0040AE62, 0x009022EC, 0x0090DCD5]),
        *[
            (caller, [caller, 0x00901E46])
            for caller in (
                0x00907084,
                0x0090715D,
                0x00907222,
                0x0090739C,
                0x009073AB,
                0x00907458,
                0x00907462,
                0x009074EE,
            )
        ],
        (0x00909626, [0x00909626, 0x00901E46]),
        (
            0x00909626,
            [0x00909626, 0x0090934D, 0x009145BB, 0x00906F86],
        ),
        (
            0x0090972F,
            [0x0090972F, 0x0090934D, 0x009145BB, 0x00906F86],
        ),
        (
            0x0090973E,
            [0x0090973E, 0x0090934D, 0x009145BB, 0x00906F86],
        ),
    ],
)
def test_pil_particle_calls_require_frozen_stack_markers(
    caller: int,
    stack: list[int],
) -> None:
    assert classify_native_call(_call(caller, stack)) == (
        "omitted_visual",
        "pil_particle_emitter_update",
    )


@pytest.mark.parametrize(
    ("caller", "stack"),
    [
        (0x0040AE62, [0x0040AE62, 0x009022EC]),
        (0x0040AE62, [0x0040AE62, 0x0090DCD5]),
        (0x00907084, [0x00907084, 0x0090934D, 0x009145BB, 0x00906F86]),
        (0x00909626, [0x00909626, 0x0090934D, 0x00906F86]),
        (0x0090972F, [0x0090972F, 0x00901E46]),
        (0x0090973E, [0x0090973E, 0x0090934D, 0x009145BB]),
    ],
)
def test_pil_particle_calls_reject_partial_or_wrong_markers(
    caller: int,
    stack: list[int],
) -> None:
    with pytest.raises(
        MTRandReconciliationProofError,
        match="unclassified_native_mtrand_call",
    ):
        classify_native_call(_call(caller, stack))


def _pil_runtime_reader(
    *,
    corrupt_rtti: bool = False,
):
    wrapper = 0x00617490
    direct_callers = (
        0x00907084,
        0x0090715D,
        0x00907222,
        0x0090739C,
        0x009073AB,
        0x00907458,
        0x00907462,
        0x009074EE,
        0x00909626,
        0x0090972F,
        0x0090973E,
    )
    values: dict[tuple[int, int], bytes] = {
        (return_address - 5, 5): b"\xe8"
        + struct.pack("<i", wrapper - return_address)
        for return_address in direct_callers
    }
    values.update(
        {
            (0x009F0D18, len(b".?AVEmitter@PIL@@\0")): (
                b".?AVEmitteX@PIL@@\0"
                if corrupt_rtti
                else b".?AVEmitter@PIL@@\0"
            ),
            (0x009F0DD4, len(b".?AVParticle@PIL@@\0")): (
                b".?AVParticle@PIL@@\0"
            ),
            (0x009F0CF8, len(b".?AVMovableObject@PIL@@\0")): (
                b".?AVMovableObject@PIL@@\0"
            ),
            (0x00994CE8, 4): struct.pack("<I", 0x009A56F0),
            (0x00994D88, 4): struct.pack("<I", 0x009A58DC),
            (0x00994E10, 4): struct.pack("<I", 0x009A59BC),
            (0x00994D38, 4): struct.pack("<I", 0x0090796F),
            (0x00994DAC, 4): struct.pack("<I", 0x0090DCA4),
            (0x00994E34, 4): struct.pack("<I", 0x009021F3),
            (0x0090796F, 8): bytes.fromhex("558bec83e4f86aff"),
            (0x0090DCA4, 9): bytes.fromhex("558bec83e4f883ec3c"),
            (0x009021F3, 6): bytes.fromhex("558bec83ec34"),
            (0x00906915, 6): bytes.fromhex("c703ec4c9900"),
            (0x0090DF11, 6): bytes.fromhex("c7068c4d9900"),
            (0x0090242B, 6): bytes.fromhex("c707144e9900"),
            (0x00901E41, 5): bytes.fromhex("e89a41b6ff"),
            (0x0090DCD0, 5): bytes.fromhex("e81e45ffff"),
            (0x00906F81, 5): bytes.fromhex("e8b7260000"),
        }
    )

    def read(address: int, size: int) -> bytes:
        return values[(address, size)]

    return read


def test_pil_particle_static_surface_binds_calls_rtti_and_vtables() -> None:
    result = _verify_pil_particle_runtime_surface(_pil_runtime_reader())
    assert result["semantic"] == "pil_particle_emitter_update"
    assert len(result["verified_call_sites"]) == 11
    assert result["functions"] == {
        "emitter_vtable_slot_19": 0x0090796F,
        "particle_update_vtable_slot_8": 0x0090DCA4,
        "movable_object_update_vtable_slot_8": 0x009021F3,
    }


def test_pil_particle_static_surface_rejects_rtti_drift() -> None:
    with pytest.raises(
        MTRandReconciliationProofError,
        match="runtime_pil_particle_semantics_mismatch",
    ):
        _verify_pil_particle_runtime_surface(
            _pil_runtime_reader(corrupt_rtti=True)
        )


def test_pil_static_surface_activation_is_conditional() -> None:
    assert _call_requires_pil_particle_static_surface(
        _call(0x0040AE62, [0x0040AE62, 0x00901E46])
    )
    assert _call_requires_pil_particle_static_surface(
        _call(0x00907084, [0x00907084])
    )
    assert not _call_requires_pil_particle_static_surface(
        _call(0x0040AE62, [0x0040AE62, 0x005A6496, 0x004B6A68])
    )


def test_native_call_classification_rejects_caller_only_label() -> None:
    with pytest.raises(
        MTRandReconciliationProofError,
        match="unclassified_native_mtrand_call",
    ):
        classify_native_call(_call(0x0040CADF, [0x0040CADF, 0x00401234]))

    with pytest.raises(
        MTRandReconciliationProofError,
        match="unclassified_native_mtrand_call",
    ):
        classify_native_call(_call(0x0045CEE0, [0x0045CEE0, 0x00401234]))

    with pytest.raises(
        MTRandReconciliationProofError,
        match="unclassified_native_mtrand_call",
    ):
        classify_native_call(_call(0x00458CD1, [0x00458CD1, 0x00401234]))

    with pytest.raises(
        MTRandReconciliationProofError,
        match="unclassified_native_mtrand_call",
    ):
        classify_native_call(
            _call(0x00458CD1, [0x00458CD1, 0x00459031, 0x00401234])
        )

    with pytest.raises(
        MTRandReconciliationProofError,
        match="unclassified_native_mtrand_call",
    ):
        classify_native_call(
            _call(
                0x00402111,
                [0x00402111, 0x00458D6A, 0x00459031, 0x00401234],
            )
        )


def test_reconciliation_transcript_binds_every_nonproof_field() -> None:
    report = {
        "schema": "zuma-rl.pc-gameplay-simulator-diff",
        "version": 3,
        "status": "PASS",
        "start_update": 10,
        "end_update": 11,
        "compared_tick_count": 1,
        "trajectory": {"artifact_sha256": "sha256:" + "a" * 64},
        "dmo": {"artifact_sha256": "sha256:" + "b" * 64},
        "synchronize_shooter": False,
        "shooter_mismatch_updates": [],
        "global_mtrand_policy": (
            "conditional_external_mtrand_reconciliation_v2"
        ),
        "global_mtrand_state_restored": True,
        "global_mtrand_reconciliation": {"rows": []},
        "ticks": [],
        "source_authorized_features": ["shot_release"],
    }
    original = reconciliation_transcript_sha256(report)
    report["synchronize_shooter"] = True
    assert reconciliation_transcript_sha256(report) != original
    report["synchronize_shooter"] = False
    report["global_mtrand_reconciliation_proof"] = {"artifact": "proof"}
    assert reconciliation_transcript_sha256(report) == original
