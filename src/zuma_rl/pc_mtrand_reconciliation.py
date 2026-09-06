"""Verify native call-trace proof for gameplay MTRand reconciliation.

The retail game shares one global Mersenne-Twister stream between gameplay
and several presentation systems.  A headless simulator intentionally omits
those presentation systems, so a PC differential may advance the simulator
through the omitted draws *after* comparing gameplay state.  Such a report is
certifying only when native traces prove, for every reconciled tick, that the
draw-count difference is exactly the number of omitted presentation calls.

This module deliberately verifies the complete per-tick native call sequence,
not merely the suffix consumed by the differential.  That distinction matters
when a presentation call precedes a modeled gameplay call in the same tick.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
from pathlib import Path, PurePosixPath
import struct
from typing import Any

from zuma_rl.pc_memory_trajectory import (
    TrajectoryMTRandState,
    load_memory_trajectory,
)
from zuma_rl.revenge_core import PopCapMTRandom
from zuma_rl.retail_dmo_provenance import (
    RetailDmoProvenanceError,
    read_certifying_provenance,
)


MTRAND_RECONCILIATION_PROOF_SCHEMA = (
    "zuma-rl.pc-mtrand-reconciliation-proof"
)
MTRAND_RECONCILIATION_PROOF_VERSION = 1
MTRAND_RECONCILIATION_PROOF_CLASSIFICATION = (
    "complete-native-call-sequence-visual-omission-decomposition"
)
EXPECTED_RETAIL_RUNTIME_SHA256 = (
    "sha256:"
    "2181ce2bfbfcb4678bf69a1474e08d3db941311aa768176a88453cc69692af20"
)
EXPECTED_DMO_SHA256 = (
    "sha256:"
    "216eb662e060ea4a6cbded03018bc193247298c3ba9f57fbb06c0787a9f951a7"
)
GLOBAL_TRACE_SCHEMA = "zuma-rl.pc-global-mtrand-call-trace"
TRACE_ALIGNMENT_SCHEMA = "zuma-rl.pc-rng-trace-alignment"
COMPACT_OVERLAP_SCHEMA = (
    "zuma-rl.pc-source-bound-compact-overlap-verification"
)
MEMORY_PROBE_SCHEMA = "zuma-rl.pc-memory-int32-probe"
DIAGNOSTIC_MUTATION_SCHEMA = (
    "zuma-rl.pc-diagnostic-source-bound-compact-state-restore"
)
GAMEPLAY_DIFF_SCHEMA = "zuma-rl.pc-gameplay-simulator-diff"
BOUND_MTRAND_POLICY = "conditional_external_mtrand_reconciliation_v2"
BOUND_MTRAND_CLASSIFICATION = (
    "unclassified_until_bound_call_trace_proof"
)
MTRAND_STATE_BINDING = "sha256_of_624_words_plus_index_le_u32"
GLOBAL_MTRAND_WRAPPER_VA = 0x00617490
GLOBAL_MTRAND_COMPAT_WRAPPER_VA = 0x00401530
GLOBAL_MTRAND_ADDRESS = 0x00A313C0
MTRAND_WORD_COUNT = 624

_GAMEPLAY_CALLERS = {
    0x00402111: "pending_ball_visual_frame_roll",
    0x00458CD1: "pending_ball_repeat_roll",
    0x004B5ADF: "fruit_chance_scheduler",
    0x0045CEE0: "powerup_spawn_chance_roll",
}
_SCORE_PAIR_CALLERS = {0x0040CADF, 0x0040CBC0}
_SCORE_JITTER_CALLERS = {0x0040CDDE}
_PARTICLE_CALLERS = {0x0065B821}
_SHADOW_GENERIC_CALLERS = {0x0040AE62, 0x00454C4E}
_SHADOW_DIRECT_CALLERS = {0x005A64A4, 0x005A6630, 0x005A66A4}
_PIL_PARTICLE_GENERIC_CALLERS = {0x0040AE62}
_PIL_PARTICLE_PRIMARY_DIRECT_CALLERS = {
    0x00907084,
    0x0090715D,
    0x00907222,
    0x0090739C,
    0x009073AB,
    0x00907458,
    0x00907462,
    0x009074EE,
}
_PIL_PARTICLE_MIXED_DIRECT_CALLERS = {0x00909626}
_PIL_PARTICLE_NESTED_DIRECT_CALLERS = {0x0090972F, 0x0090973E}
_PIL_PARTICLE_DIRECT_CALLERS = (
    _PIL_PARTICLE_PRIMARY_DIRECT_CALLERS
    | _PIL_PARTICLE_MIXED_DIRECT_CALLERS
    | _PIL_PARTICLE_NESTED_DIRECT_CALLERS
)
_PIL_PARTICLE_PRIMARY_STACK_MARKER = 0x00901E46
_PIL_PARTICLE_MOVABLE_STACK_MARKERS = {0x009022EC, 0x0090DCD5}
_PIL_PARTICLE_NESTED_STACK_MARKERS = {
    0x0090934D,
    0x009145BB,
    0x00906F86,
}
_VISUAL_CALLERS = (
    _SCORE_PAIR_CALLERS
    | _SCORE_JITTER_CALLERS
    | _PARTICLE_CALLERS
    | _SHADOW_GENERIC_CALLERS
    | _SHADOW_DIRECT_CALLERS
)

_DIRECT_CALL_TARGETS = {
    0x0040AE62: GLOBAL_MTRAND_COMPAT_WRAPPER_VA,
    0x0040CADF: GLOBAL_MTRAND_WRAPPER_VA,
    0x0040CBC0: GLOBAL_MTRAND_WRAPPER_VA,
    0x0040CDDE: GLOBAL_MTRAND_COMPAT_WRAPPER_VA,
    0x00454C4E: GLOBAL_MTRAND_COMPAT_WRAPPER_VA,
    0x004B5ADF: GLOBAL_MTRAND_COMPAT_WRAPPER_VA,
    0x005A64A4: GLOBAL_MTRAND_COMPAT_WRAPPER_VA,
    0x005A6630: GLOBAL_MTRAND_WRAPPER_VA,
    0x005A66A4: GLOBAL_MTRAND_WRAPPER_VA,
    0x0065B821: GLOBAL_MTRAND_WRAPPER_VA,
}
_CONDITIONAL_DIRECT_CALL_TARGETS = {
    0x0045CEE0: GLOBAL_MTRAND_COMPAT_WRAPPER_VA,
}
_PENDING_BALL_DIRECT_CALL_TARGETS = {
    0x00402111: GLOBAL_MTRAND_COMPAT_WRAPPER_VA,
    0x00458CD1: GLOBAL_MTRAND_COMPAT_WRAPPER_VA,
}
_PIL_PARTICLE_DIRECT_CALL_TARGETS = {
    caller: GLOBAL_MTRAND_WRAPPER_VA
    for caller in _PIL_PARTICLE_DIRECT_CALLERS
}


class MTRandReconciliationProofError(ValueError):
    """A reconciliation proof or one of its source artifacts is invalid."""


def _fail(code: str) -> None:
    raise MTRandReconciliationProofError(code)


def _reject_constant(value: str) -> None:
    _fail(f"nonfinite_json_number:{value}")


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate_json_key:{key}")
        result[key] = value
    return result


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        raw = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=_reject_constant,
        )
    except MTRandReconciliationProofError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MTRandReconciliationProofError(
            f"invalid_json:{path}"
        ) from error
    if not isinstance(raw, Mapping):
        _fail(f"json_root_not_object:{path}")
    return raw


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise MTRandReconciliationProofError(
            f"artifact_unreadable:{path}"
        ) from error
    return "sha256:" + digest.hexdigest()


def _mapping(value: Any, code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(code)
    return value


def _sequence(value: Any, code: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _fail(code)
    return value


def _integer(value: Any, code: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(code)
    if minimum is not None and value < minimum:
        _fail(code)
    return value


def _sha256(value: Any, code: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 71
        or not value.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        _fail(code)
    return value


def _relative_path(value: Any, code: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        _fail(code)
    result = PurePosixPath(value)
    if (
        result.is_absolute()
        or not result.parts
        or any(part in {"", ".", ".."} for part in result.parts)
    ):
        _fail(code)
    return result


def _resolve(root: Path, value: Any, code: str) -> Path:
    relative = _relative_path(value, code)
    root = root.resolve()
    candidate = root.joinpath(*relative.parts)
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise MTRandReconciliationProofError(code) from error
    if not resolved.is_file():
        _fail(code)
    return resolved


def _canonical_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise MTRandReconciliationProofError(
            "canonical_json_invalid"
        ) from error
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def reconciliation_transcript_sha256(report: Mapping[str, Any]) -> str:
    """Hash the complete differential except its circular proof reference."""

    required = (
        "schema",
        "version",
        "status",
        "start_update",
        "end_update",
        "compared_tick_count",
        "trajectory",
        "dmo",
        "synchronize_shooter",
        "shooter_mismatch_updates",
        "global_mtrand_policy",
        "global_mtrand_state_restored",
        "global_mtrand_reconciliation",
        "ticks",
    )
    if any(field not in report for field in required):
        _fail("gameplay_diff_reconciliation_transcript_incomplete")
    return _canonical_sha256(
        {
            key: value
            for key, value in report.items()
            if key != "global_mtrand_reconciliation_proof"
        }
    )


def mtrand_state_sha256(state: TrajectoryMTRandState) -> str:
    if (
        len(state.words) != MTRAND_WORD_COUNT
        or not 0 <= state.index <= MTRAND_WORD_COUNT
    ):
        _fail("trajectory_global_mtrand_state_invalid")
    payload = struct.pack(
        f"<{MTRAND_WORD_COUNT + 1}I",
        *state.words,
        state.index,
    )
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _rng_state_sha256(rng: PopCapMTRandom) -> str:
    payload = struct.pack(
        f"<{MTRAND_WORD_COUNT + 1}I",
        *rng.words,
        rng.index,
    )
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _stack_addresses(call: Mapping[str, Any]) -> tuple[int, ...]:
    stack = _sequence(call.get("stack_code"), "trace_call_stack_invalid")
    addresses: list[int] = []
    previous_offset = -1
    for value in stack:
        frame = _mapping(value, "trace_call_stack_frame_invalid")
        address = _integer(
            frame.get("address"),
            "trace_call_stack_address_invalid",
            minimum=1,
        )
        offset = _integer(
            frame.get("stack_offset"),
            "trace_call_stack_offset_invalid",
            minimum=0,
        )
        if offset <= previous_offset:
            _fail("trace_call_stack_offsets_not_increasing")
        if frame.get("address_hex") != f"0x{address:08x}":
            _fail("trace_call_stack_address_hex_mismatch")
        addresses.append(address)
        previous_offset = offset
    if not addresses:
        _fail("trace_call_stack_empty")
    return tuple(addresses)


def _is_pil_particle_call(
    caller: int,
    addresses: Sequence[int],
) -> bool:
    address_set = set(addresses)
    has_primary = _PIL_PARTICLE_PRIMARY_STACK_MARKER in address_set
    has_movable_pair = _PIL_PARTICLE_MOVABLE_STACK_MARKERS <= address_set
    has_nested = _PIL_PARTICLE_NESTED_STACK_MARKERS <= address_set
    if caller in _PIL_PARTICLE_GENERIC_CALLERS:
        return has_primary or has_movable_pair
    if caller in _PIL_PARTICLE_PRIMARY_DIRECT_CALLERS:
        return has_primary
    if caller in _PIL_PARTICLE_MIXED_DIRECT_CALLERS:
        return has_primary or has_nested
    if caller in _PIL_PARTICLE_NESTED_DIRECT_CALLERS:
        return has_nested
    return False


def _call_requires_pil_particle_static_surface(
    call: Mapping[str, Any],
) -> bool:
    caller = _integer(call.get("caller"), "trace_call_caller_invalid")
    if caller in _PIL_PARTICLE_DIRECT_CALLERS:
        return True
    if caller not in _PIL_PARTICLE_GENERIC_CALLERS:
        return False
    return _is_pil_particle_call(caller, _stack_addresses(call))


def classify_native_call(call: Mapping[str, Any]) -> tuple[str, str]:
    """Return ``(scope, semantic)`` for one exact-retail trace call."""

    caller = _integer(call.get("caller"), "trace_call_caller_invalid")
    addresses = _stack_addresses(call)
    if addresses[0] != caller:
        _fail("trace_call_stack_caller_mismatch")
    if caller == 0x004B5ADF and 0x0041C490 in addresses:
        return "modeled_gameplay", _GAMEPLAY_CALLERS[caller]
    # The outer 0x0065DE83 update frame is not stable under asynchronous
    # debugger stack sampling.  The nearer 0x00459031 -> 0x0045DE5E chain is
    # the retail pending-ball update path; proof derivation separately pins
    # the executable hash and verifies the direct call opcodes and operands.
    if (
        caller == 0x00458CD1
        and 0x00459031 in addresses
        and 0x0045DE5E in addresses
    ):
        return "modeled_gameplay", _GAMEPLAY_CALLERS[caller]
    if (
        caller == 0x00402111
        and 0x00458D6A in addresses
        and 0x00459031 in addresses
        and 0x0045DE5E in addresses
    ):
        return "modeled_gameplay", _GAMEPLAY_CALLERS[caller]
    if (
        caller == 0x0045CEE0
        and 0x0045E195 in addresses
        and 0x0065DE83 in addresses
    ):
        return "modeled_gameplay", _GAMEPLAY_CALLERS[caller]
    if caller in _SCORE_PAIR_CALLERS and 0x004171AD in addresses:
        return "omitted_visual", "score_digit_transition"
    if (
        caller in _SCORE_JITTER_CALLERS
        and 0x0041FD7D in addresses
        and 0x004227F5 in addresses
    ):
        return "omitted_visual", "score_digit_wobble"
    # Some PIEffect update frames have already unwound by the time the
    # debugger samples the caller stack.  The exact return address is itself
    # sufficient here: the proof separately pins the retail executable hash,
    # verifies the call opcode/target at 0x0065B81C, and verifies the
    # PIEffect::Update symbol surface and function entry.
    if caller in _PARTICLE_CALLERS:
        return "omitted_visual", "pi_effect_particle_update"
    if (
        caller in _SHADOW_GENERIC_CALLERS
        and 0x004B6A68 in addresses
        and ({0x005A6496, 0x005A653A} & set(addresses))
    ):
        return "omitted_visual", "shadow_canopy_animation"
    if caller in _SHADOW_DIRECT_CALLERS and 0x004B6A68 in addresses:
        return "omitted_visual", "shadow_canopy_animation"
    if _is_pil_particle_call(caller, addresses):
        return "omitted_visual", "pil_particle_emitter_update"
    _fail(f"unclassified_native_mtrand_call:0x{caller:08x}")


def _pe_layout(payload: bytes) -> tuple[int, tuple[tuple[int, int, int], ...]]:
    if len(payload) < 0x40:
        _fail("runtime_pe_header_invalid")
    pe_offset = struct.unpack_from("<I", payload, 0x3C)[0]
    if (
        pe_offset + 24 > len(payload)
        or payload[pe_offset : pe_offset + 4] != b"PE\0\0"
    ):
        _fail("runtime_pe_header_invalid")
    section_count = struct.unpack_from("<H", payload, pe_offset + 6)[0]
    optional_size = struct.unpack_from("<H", payload, pe_offset + 20)[0]
    optional_offset = pe_offset + 24
    if (
        section_count <= 0
        or optional_offset + optional_size > len(payload)
        or struct.unpack_from("<H", payload, optional_offset)[0] != 0x10B
    ):
        _fail("runtime_pe_optional_header_invalid")
    image_base = struct.unpack_from("<I", payload, optional_offset + 28)[0]
    section_offset = optional_offset + optional_size
    sections: list[tuple[int, int, int]] = []
    for index in range(section_count):
        offset = section_offset + index * 40
        if offset + 40 > len(payload):
            _fail("runtime_pe_section_table_invalid")
        virtual_size, virtual_address, raw_size, raw_offset = struct.unpack_from(
            "<IIII", payload, offset + 8
        )
        if raw_offset + raw_size > len(payload):
            _fail("runtime_pe_section_bounds_invalid")
        sections.append(
            (virtual_address, max(virtual_size, raw_size), raw_offset)
        )
    return image_base, tuple(sections)


def _read_runtime_va(
    payload: bytes,
    *,
    image_base: int,
    sections: Sequence[tuple[int, int, int]],
    virtual_address: int,
    size: int,
) -> bytes:
    relative = virtual_address - image_base
    for section_va, section_size, raw_offset in sections:
        if section_va <= relative and relative + size <= section_va + section_size:
            offset = raw_offset + relative - section_va
            if offset + size <= len(payload):
                return payload[offset : offset + size]
    _fail("runtime_virtual_address_unmapped")


def _verify_pil_particle_runtime_surface(
    read: Callable[[int, int], bytes],
) -> Mapping[str, Any]:
    call_sites: list[dict[str, Any]] = []
    for return_address, expected_target in sorted(
        _PIL_PARTICLE_DIRECT_CALL_TARGETS.items()
    ):
        instruction = read(return_address - 5, 5)
        if instruction[0] != 0xE8:
            _fail("runtime_pil_particle_mtrand_call_opcode_mismatch")
        target = return_address + struct.unpack("<i", instruction[1:])[0]
        if target != expected_target:
            _fail("runtime_pil_particle_mtrand_call_target_mismatch")
        call_sites.append(
            {
                "return_address": return_address,
                "instruction_hex": instruction.hex(),
                "target_address": target,
            }
        )

    if (
        read(0x009F0D18, len(b".?AVEmitter@PIL@@\0"))
        != b".?AVEmitter@PIL@@\0"
        or read(0x009F0DD4, len(b".?AVParticle@PIL@@\0"))
        != b".?AVParticle@PIL@@\0"
        or read(0x009F0CF8, len(b".?AVMovableObject@PIL@@\0"))
        != b".?AVMovableObject@PIL@@\0"
        or struct.unpack("<I", read(0x00994CE8, 4))[0] != 0x009A56F0
        or struct.unpack("<I", read(0x00994D88, 4))[0] != 0x009A58DC
        or struct.unpack("<I", read(0x00994E10, 4))[0] != 0x009A59BC
        or struct.unpack("<I", read(0x00994D38, 4))[0] != 0x0090796F
        or struct.unpack("<I", read(0x00994DAC, 4))[0] != 0x0090DCA4
        or struct.unpack("<I", read(0x00994E34, 4))[0] != 0x009021F3
        or read(0x0090796F, 8) != bytes.fromhex("558bec83e4f86aff")
        or read(0x0090DCA4, 9) != bytes.fromhex("558bec83e4f883ec3c")
        or read(0x009021F3, 6) != bytes.fromhex("558bec83ec34")
        or read(0x00906915, 6) != bytes.fromhex("c703ec4c9900")
        or read(0x0090DF11, 6) != bytes.fromhex("c7068c4d9900")
        or read(0x0090242B, 6) != bytes.fromhex("c707144e9900")
        or read(0x00901E41, 5) != bytes.fromhex("e89a41b6ff")
        or read(0x0090DCD0, 5) != bytes.fromhex("e81e45ffff")
        or read(0x00906F81, 5) != bytes.fromhex("e8b7260000")
    ):
        _fail("runtime_pil_particle_semantics_mismatch")

    return {
        "semantic": "pil_particle_emitter_update",
        "rtti": {
            "emitter": {
                "name": ".?AVEmitter@PIL@@",
                "name_virtual_address": 0x009F0D18,
                "complete_object_locator_virtual_address": 0x009A56F0,
                "vtable_virtual_address": 0x00994CEC,
            },
            "particle": {
                "name": ".?AVParticle@PIL@@",
                "name_virtual_address": 0x009F0DD4,
                "complete_object_locator_virtual_address": 0x009A58DC,
                "vtable_virtual_address": 0x00994D8C,
            },
            "movable_object": {
                "name": ".?AVMovableObject@PIL@@",
                "name_virtual_address": 0x009F0CF8,
                "complete_object_locator_virtual_address": 0x009A59BC,
                "vtable_virtual_address": 0x00994E14,
            },
        },
        "functions": {
            "emitter_vtable_slot_19": 0x0090796F,
            "particle_update_vtable_slot_8": 0x0090DCA4,
            "movable_object_update_vtable_slot_8": 0x009021F3,
        },
        "stack_markers": {
            "primary": _PIL_PARTICLE_PRIMARY_STACK_MARKER,
            "particle_movable_pair": sorted(
                _PIL_PARTICLE_MOVABLE_STACK_MARKERS
            ),
            "nested_emitter": sorted(_PIL_PARTICLE_NESTED_STACK_MARKERS),
        },
        "verified_call_sites": call_sites,
    }


def _verify_runtime_static_surface(
    path: Path,
    *,
    required_callers: set[int] | None = None,
    require_pil_particle_surface: bool = False,
) -> Mapping[str, Any]:
    runtime_sha256 = sha256_path(path)
    if runtime_sha256 != EXPECTED_RETAIL_RUNTIME_SHA256:
        _fail("runtime_executable_sha256_mismatch")
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise MTRandReconciliationProofError(
            "runtime_executable_unreadable"
        ) from error
    image_base, sections = _pe_layout(payload)
    if image_base != 0x00400000:
        _fail("runtime_image_base_mismatch")

    call_targets = dict(_DIRECT_CALL_TARGETS)
    for caller, target in _CONDITIONAL_DIRECT_CALL_TARGETS.items():
        if required_callers is not None and caller in required_callers:
            call_targets[caller] = target
    call_sites: list[dict[str, Any]] = []
    for return_address, expected_target in sorted(call_targets.items()):
        instruction = _read_runtime_va(
            payload,
            image_base=image_base,
            sections=sections,
            virtual_address=return_address - 5,
            size=5,
        )
        if instruction[0] != 0xE8:
            _fail("runtime_mtrand_call_opcode_mismatch")
        target = return_address + struct.unpack("<i", instruction[1:])[0]
        if target != expected_target:
            _fail("runtime_mtrand_call_target_mismatch")
        call_sites.append(
            {
                "return_address": return_address,
                "instruction_hex": instruction.hex(),
                "target_address": target,
            }
        )

    def read(virtual_address: int, size: int) -> bytes:
        return _read_runtime_va(
            payload,
            image_base=image_base,
            sections=sections,
            virtual_address=virtual_address,
            size=size,
        )

    pending_ball_callers = {
        0x00402111,
        0x00458CD1,
    } & (required_callers or set())
    if (
        read(0x00982E34, len(b"ShadowCanopy1\0"))
        != b"ShadowCanopy1\0"
        or struct.unpack("<I", read(0x00982E5C, 4))[0] != 0x005A63F0
        or read(0x005A63F0, 3) != b"\x55\x8b\xec"
        or read(0x0098843C, len(b"PIEffect::Update\0"))
        != b"PIEffect::Update\0"
        or read(0x0065B730, 3) != b"\x55\x8b\xec"
        or read(0x0096194C, len(b"POINT TARGET REACHED!\0"))
        != b"POINT TARGET REACHED!\0"
        or read(0x0096198C, len(b"ACE TARGET REACHED!\0"))
        != b"ACE TARGET REACHED!\0"
        or read(0x009619C0, len(b"New High Score!\0"))
        != b"New High Score!\0"
        or read(0x00416F9A, 5) != bytes.fromhex("e8315affff")
        or read(0x0041FD78, 5) != bytes.fromhex("e8c3cefeff")
        or read(0x0040CDD9, 21)
        != bytes.fromhex("e85247ffff99dd0508769900b903000000d9eef7f9")
    ):
        _fail("runtime_visual_callsite_semantics_mismatch")
    if pending_ball_callers:
        # These two callers were added after the original long-horizon proof
        # format was frozen.  Verify them when observed, but deliberately keep
        # them out of the serialized ``verified_call_sites`` surface so that a
        # stronger verifier does not invalidate immutable older proofs whose
        # gameplay transcript is unchanged.
        for return_address in sorted(pending_ball_callers):
            instruction = read(return_address - 5, 5)
            if instruction[0] != 0xE8:
                _fail("runtime_pending_ball_call_opcode_mismatch")
            target = return_address + struct.unpack("<i", instruction[1:])[0]
            if target != _PENDING_BALL_DIRECT_CALL_TARGETS[return_address]:
                _fail("runtime_pending_ball_call_target_mismatch")
        if (
            read(0x00458CCC, 15)
            != bytes.fromhex("e85f88faff99b964000000f7f93bd6")
            or read(0x0040210C, 17)
            != bytes.fromhex("e81ff4ffff99f77e345e8997f4000000c3")
        ):
            _fail("runtime_pending_ball_callsite_semantics_mismatch")

    result: dict[str, Any] = {
        "image_base": image_base,
        "global_mtrand_wrapper_virtual_address": GLOBAL_MTRAND_WRAPPER_VA,
        "global_mtrand_compat_wrapper_virtual_address": (
            GLOBAL_MTRAND_COMPAT_WRAPPER_VA
        ),
        "verified_call_sites": call_sites,
        "shadow_canopy": {
            "class_name": "ShadowCanopy1",
            "update_function_virtual_address": 0x005A63F0,
            "vtable_slot_virtual_address": 0x00982E5C,
        },
        "particle_effect": {
            "trace_name": "PIEffect::Update",
            "update_function_virtual_address": 0x0065B730,
        },
        "score_display": {
            "score_update_function_virtual_address": 0x00416D00,
            "digit_transition_function_virtual_address": 0x0040C9D0,
            "digit_wobble_function_virtual_address": 0x0040CC40,
        },
        "fruit_scheduler": {
            "board_update_call_return_address": 0x004B5ADF,
        },
    }
    if require_pil_particle_surface:
        result["pil_particle_emitter"] = (
            _verify_pil_particle_runtime_surface(read)
        )
    return result


def _absolute_file(value: Any, code: str) -> Path:
    if not isinstance(value, str) or not value:
        _fail(code)
    candidate = Path(value)
    if not candidate.is_absolute():
        _fail(code)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise MTRandReconciliationProofError(code) from error
    if not resolved.is_file():
        _fail(code)
    return resolved


def _bound_absolute_descriptor(value: Any, code: str) -> Path:
    descriptor = _mapping(value, code)
    path = _absolute_file(descriptor.get("path"), code)
    expected_bytes = _integer(
        descriptor.get("bytes"),
        code,
        minimum=0,
    )
    expected_sha256 = _sha256(descriptor.get("sha256"), code)
    if path.stat().st_size != expected_bytes or sha256_path(path) != expected_sha256:
        _fail(code)
    return path


def _verify_v2_retail_dmo_provenance(
    *,
    replay: Mapping[str, Any],
    source_recording: Mapping[str, Any],
    collector_plan_path: Path,
    playback_dmo_path: Path,
    playback_dmo_sha256: str,
) -> Mapping[str, Any]:
    provenance_path = _bound_absolute_descriptor(
        replay.get("retail_dmo_provenance"),
        "trace_probe_retail_dmo_provenance_invalid",
    )
    raw_dmo_path = _bound_absolute_descriptor(
        source_recording.get("raw_dmo"),
        "trace_probe_raw_recording_dmo_invalid",
    )
    recording_report_path = _bound_absolute_descriptor(
        source_recording.get("report"),
        "trace_probe_recording_report_invalid",
    )
    if sha256_path(playback_dmo_path) != playback_dmo_sha256:
        _fail("trace_probe_playback_dmo_hash_mismatch")
    try:
        facts = read_certifying_provenance(
            source_path=raw_dmo_path,
            output_path=playback_dmo_path,
            recording_report_path=recording_report_path,
            collector_plan_path=collector_plan_path,
            provenance_path=provenance_path,
            expected_runtime_sha256=EXPECTED_RETAIL_RUNTIME_SHA256,
        )
    except (OSError, RetailDmoProvenanceError, ValueError) as error:
        raise MTRandReconciliationProofError(
            "trace_probe_retail_dmo_provenance_invalid"
        ) from error
    if (
        facts.get("provenance_version") != 2
        or facts.get("playback_dmo_sha256") != playback_dmo_sha256
        or facts.get("source_dmo_sha256") != sha256_path(raw_dmo_path)
        or facts.get("removed_startup_command_count") != 1
        or facts.get("gameplay_input_commands_preserved") is not True
        or facts.get("recording_process_memory_writes") != 0
        or facts.get("recording_outcome") != source_recording.get("outcome")
        or facts.get("recording_natural_outcome") is not True
        or facts.get("recording_normal_retail_exit") is not True
        or facts.get("recording_host_restored_exactly") is not True
        or facts.get("collector_prestate_matched") is not True
        or facts.get("collector_board_seed_mutation") is not False
    ):
        _fail("trace_probe_retail_dmo_provenance_contract_mismatch")
    return {
        "artifact_sha256": sha256_path(provenance_path),
        "provenance_version": facts["provenance_version"],
        "source_dmo_sha256": facts["source_dmo_sha256"],
        "playback_dmo_sha256": facts["playback_dmo_sha256"],
        "recording_outcome": facts["recording_outcome"],
        "gameplay_input_commands_preserved": True,
        "recording_process_memory_writes": 0,
        "recording_natural_outcome": True,
        "recording_normal_retail_exit": True,
        "recording_host_restored_exactly": True,
        "collector_board_seed_mutation": False,
    }


def _verify_trace_probe_source_binding(
    *,
    window_root: Path,
    probe: Mapping[str, Any],
    process_id: int,
    source_probe_sha256: str,
    dmo_sha256: str,
    start_update: int,
) -> Mapping[str, Any] | None:
    """Validate either the legacy compact restore or exact source replay."""

    mutation = probe.get("diagnostic_mutation")
    if isinstance(mutation, Mapping):
        if (
            mutation.get("schema") != DIAGNOSTIC_MUTATION_SCHEMA
            or mutation.get("version") != 1
            or mutation.get("status") != "PASS"
            or mutation.get("process_id") != process_id
            or mutation.get("source_probe_sha256") != source_probe_sha256
            or mutation.get("post_compact_state_exact_source") is not True
            or mutation.get("transactional_rollback_on_failure") is not True
            or mutation.get("writeback_verified") is not True
            or mutation.get("persistent_file_modified") is not False
        ):
            _fail("trace_window_source_binding_mismatch")
        return None
    if mutation is not None:
        _fail("trace_window_source_binding_mismatch")

    replay = _mapping(
        probe.get("strict_command_replay"),
        "trace_probe_source_replay_missing",
    )
    external_guard = _mapping(
        probe.get("external_input_guard"),
        "trace_probe_external_guard_missing",
    )
    process_identity = _mapping(
        probe.get("process_identity"),
        "trace_probe_process_identity_missing",
    )
    normalization = _mapping(
        replay.get("normalization"),
        "trace_probe_replay_normalization_missing",
    )
    transport = _mapping(
        replay.get("transport"),
        "trace_probe_replay_transport_missing",
    )
    source_recording = _mapping(
        replay.get("source_recording"),
        "trace_probe_source_recording_missing",
    )
    collector_plan = _mapping(
        replay.get("collector_plan"),
        "trace_probe_collector_plan_missing",
    )
    replay_path = _resolve(
        window_root,
        replay.get("artifact"),
        "trace_probe_replay_artifact_invalid",
    )
    guard_path = _resolve(
        window_root,
        external_guard.get("artifact"),
        "trace_probe_external_guard_artifact_invalid",
    )
    replay_sha256 = sha256_path(replay_path)
    guard_sha256 = sha256_path(guard_path)
    plan_path = Path(str(collector_plan.get("path"))).resolve()
    try:
        plan_sha256 = sha256_path(plan_path)
    except MTRandReconciliationProofError:
        _fail("trace_probe_collector_plan_invalid")
    replay_version = _integer(
        replay.get("version"),
        "trace_probe_source_replay_version_invalid",
        minimum=1,
    )
    expected_replay_classification = {
        1: "source_bound_original_natural_retail_recording",
        2: "source_bound_original_natural_retail_outcome",
    }.get(replay_version)
    if expected_replay_classification is None:
        _fail("trace_probe_source_replay_version_invalid")
    strict = _read_json(replay_path)
    strict_result = _mapping(
        strict.get("result"),
        "trace_probe_strict_result_missing",
    )
    strict_dmo = _mapping(
        strict.get("source_dmo"),
        "trace_probe_strict_dmo_missing",
    )
    playback_dmo_path = _absolute_file(
        replay.get("source_dmo"),
        "trace_probe_playback_dmo_invalid",
    )
    strict_dmo_path = _absolute_file(
        strict_dmo.get("path"),
        "trace_probe_strict_dmo_invalid",
    )
    startup_rng = _mapping(
        strict.get("startup_rng"),
        "trace_probe_startup_rng_missing",
    )
    anchor = _mapping(
        strict.get("source_bound_board_anchor"),
        "trace_probe_source_anchor_missing",
    )
    observations = _sequence(
        anchor.get("observations"),
        "trace_probe_source_anchor_observations_invalid",
    )
    if len(observations) != 1:
        _fail("trace_probe_source_anchor_observations_invalid")
    observation = _mapping(
        observations[0],
        "trace_probe_source_anchor_observation_invalid",
    )
    global_state = _mapping(
        observation.get("global"),
        "trace_probe_source_anchor_global_invalid",
    )
    thread_crt = _mapping(
        observation.get("thread_crt"),
        "trace_probe_source_anchor_crt_invalid",
    )
    global_precall = _mapping(
        observation.get("global_precall_restore"),
        "trace_probe_source_anchor_precall_invalid",
    )
    guard = _read_json(guard_path)
    guard_counts = _mapping(
        guard.get("event_counts"),
        "trace_probe_external_guard_counts_invalid",
    )

    stop_after = _integer(
        replay.get("stop_after_update"),
        "trace_probe_replay_stop_invalid",
        minimum=0,
    )
    last_mutation = _integer(
        normalization.get("last_mutation_update"),
        "trace_probe_replay_mutation_update_invalid",
        minimum=0,
    )
    if (
        probe.get("evidence_classification")
        not in {
            "diagnostic",
            "formal_pc_full_state_exact_step_source",
        }
        or probe.get("process_id") != process_id
        or probe.get("dmo_sha256") != dmo_sha256
        or probe.get("runtime_executable_sha256")
        != EXPECTED_RETAIL_RUNTIME_SHA256
        or process_identity.get("process_id") != process_id
        or process_identity.get("executable_sha256")
        != EXPECTED_RETAIL_RUNTIME_SHA256
        or replay.get("schema")
        != "zuma-rl.pc-source-bound-original-command-replay-binding"
        or replay.get("version") != replay_version
        or replay.get("status") != "PASS"
        or replay.get("classification")
        != expected_replay_classification
        or replay.get("failure_count") != 0
        or replay.get("runtime_process_id") != process_id
        or replay.get("runtime_executable_sha256")
        != EXPECTED_RETAIL_RUNTIME_SHA256
        or replay.get("source_dmo_sha256") != dmo_sha256
        or replay.get("artifact_sha256") != replay_sha256
        or replay.get("artifact_bytes") != replay_path.stat().st_size
        or collector_plan.get("sha256") != plan_sha256
        or stop_after >= start_update
        or last_mutation > stop_after
        or normalization.get("global_postcall_memory_write_bytes") != 0
        or normalization.get("global_precall_memory_write_bytes") != 0
        or normalization.get("thread_crt_memory_write_bytes") != 4
        or normalization.get("gameplay_or_rng_process_memory_write_count")
        != 1
        or normalization.get("gameplay_or_rng_process_memory_write_bytes")
        != 4
        or normalization.get("persistent_file_modified") is not False
        or transport.get(
            "post_board_construction_process_memory_write_count"
        )
        != 0
        or source_recording.get("host_restored_exactly") is not True
        or source_recording.get("normal_exit") is not True
        or source_recording.get("process_memory_writes") != 0
        or strict.get("schema") != "zuma.popcap_strict_replay.v3"
        or strict_dmo_path != playback_dmo_path
        or sha256_path(playback_dmo_path) != dmo_sha256
        or strict_dmo.get("sha256") != dmo_sha256[7:]
        or strict_dmo.get("size_bytes") != playback_dmo_path.stat().st_size
        or startup_rng.get("process_id") != process_id
        or startup_rng.get("seed") != normalization.get("startup_seed")
        or startup_rng.get("register_override") != "eax_before_push_to_srand"
        or startup_rng.get("persistent_file_modified") is not False
        or strict_result.get("failure_count") != 0
        or strict_result.get("stopped_at_update") is not True
        or strict_result.get("last_update") != stop_after
        or anchor.get("mechanism")
        != "exact_post_global_call_board_constructor_seed_and_main_thread_crt_anchor"
        or anchor.get("persistent_file_modified") is not False
        or observation.get("process_id") != process_id
        or observation.get("framework_update") != last_mutation
        or observation.get("effective_seed") != normalization.get("board_seed")
        or observation.get("bytes_written") != 4
        or observation.get("observed_seed_matches_source") is not True
        or observation.get("persistent_file_modified") is not False
        or global_state.get("bytes_written") != 0
        or global_state.get("changed") is not False
        or global_state.get("natural_post_state_match") is not True
        or global_precall.get("bytes_written") != 0
        or global_precall.get("process_memory_mutation") is not False
        or global_precall.get("main_thread_match") is not True
        or thread_crt.get("bytes_written") != 4
        or observation.get("writeback_verified") is not True
        or thread_crt.get("restored_state")
        != normalization.get("thread_crt_state")
        or external_guard.get("schema")
        != "zuma-rl.pc-external-input-guard-binding"
        or external_guard.get("version") != 1
        or external_guard.get("status") != "PASS"
        or external_guard.get("external_event_count") != 0
        or external_guard.get("artifact_sha256") != guard_sha256
        or external_guard.get("artifact_bytes") != guard_path.stat().st_size
        or guard.get("schema") != "zuma-rl.pc-external-input-guard"
        or guard.get("version") != 1
        or guard.get("status") != "PASS"
        or guard_counts.get("external") != 0
    ):
        _fail("trace_window_source_binding_mismatch")
    provenance_summary: Mapping[str, Any] | None = None
    if replay_version == 2:
        provenance_summary = _verify_v2_retail_dmo_provenance(
            replay=replay,
            source_recording=source_recording,
            collector_plan_path=plan_path,
            playback_dmo_path=playback_dmo_path,
            playback_dmo_sha256=dmo_sha256,
        )
    result: dict[str, Any] = {
        "kind": "source_bound_original_command_replay",
        "strict_replay_sha256": replay_sha256,
        "collector_plan_sha256": plan_sha256,
        "external_input_guard_sha256": guard_sha256,
        "last_process_memory_mutation_update": last_mutation,
        "post_board_construction_process_memory_write_count": 0,
    }
    if provenance_summary is not None:
        result["retail_dmo_provenance"] = provenance_summary
    return result


def _verify_trace_call(
    call: Mapping[str, Any],
    *,
    expected_order: int,
) -> None:
    caller = _integer(call.get("caller"), "trace_call_caller_invalid", minimum=1)
    before = _integer(
        call.get("mtrand_index_before"),
        "trace_call_pre_index_invalid",
        minimum=0,
    )
    after = _integer(
        call.get("mtrand_index_after"),
        "trace_call_post_index_invalid",
        minimum=0,
    )
    _integer(call.get("framework_update"), "trace_call_update_invalid", minimum=0)
    _integer(call.get("mtrand_output"), "trace_call_output_invalid", minimum=0)
    _sha256(
        call.get("mtrand_state_sha256_before"),
        "trace_call_state_sha256_invalid",
    )
    if (
        _integer(call.get("order"), "trace_call_order_invalid")
        != expected_order
        or call.get("caller_hex") != f"0x{caller:08x}"
        or not 0 <= before <= MTRAND_WORD_COUNT
        or not 0 <= after <= MTRAND_WORD_COUNT
        or after != (1 if before == MTRAND_WORD_COUNT else before + 1)
    ):
        _fail("trace_call_contract_mismatch")
    addresses = _stack_addresses(call)
    if addresses[0] != caller:
        _fail("trace_call_stack_caller_mismatch")


def _verify_trace_window(
    *,
    root: Path,
    window_root_value: Any,
    source_index_path: Path,
    source_index_sha256: str,
    source_probe_sha256: str,
    dmo_sha256: str,
) -> tuple[Mapping[str, Any], dict[int, tuple[Mapping[str, Any], ...]]]:
    window_relative = _relative_path(
        window_root_value,
        "trace_window_root_invalid",
    )
    window_root = root.joinpath(*window_relative.parts).resolve()
    try:
        window_root.relative_to(root.resolve())
    except ValueError as error:
        raise MTRandReconciliationProofError(
            "trace_window_root_escapes_evidence_root"
        ) from error
    if not window_root.is_dir():
        _fail("trace_window_root_missing")

    trace_path = window_root / "global-rng-call-trace.json"
    alignment_path = window_root / "global-rng-call-trace-alignment.json"
    overlap_path = window_root / "c111-compact-overlap-verification.json"
    probe_path = window_root / "memory-probe.json"
    for path in (trace_path, alignment_path, overlap_path, probe_path):
        if not path.is_file():
            _fail("trace_window_artifact_missing")

    trace_sha256 = sha256_path(trace_path)
    alignment_sha256 = sha256_path(alignment_path)
    overlap_sha256 = sha256_path(overlap_path)
    probe_sha256 = sha256_path(probe_path)
    trace = _read_json(trace_path)
    alignment = _read_json(alignment_path)
    overlap = _read_json(overlap_path)
    probe = _read_json(probe_path)

    calls_value = _sequence(trace.get("calls"), "trace_calls_invalid")
    calls = tuple(_mapping(call, "trace_call_invalid") for call in calls_value)
    start_update = _integer(trace.get("start_update"), "trace_start_invalid")
    end_update = _integer(trace.get("end_update"), "trace_end_invalid")
    process_id = _integer(trace.get("process_id"), "trace_process_id_invalid")
    if (
        trace.get("schema") != GLOBAL_TRACE_SCHEMA
        or trace.get("version") != 1
        or trace.get("status") != "PASS"
        or trace.get("failure") is not None
        or trace.get("runtime_executable_sha256")
        != EXPECTED_RETAIL_RUNTIME_SHA256
        or trace.get("address") != GLOBAL_MTRAND_WRAPPER_VA
        or trace.get("global_mtrand_address") != GLOBAL_MTRAND_ADDRESS
        or trace.get("process_memory_mutation") is not False
        or trace.get("stop_reason") != "stop_requested"
        or trace.get("access_denied_thread_skip_event_count") != 0
        or trace.get("access_denied_thread_skip_ids") != []
        or _integer(trace.get("call_count"), "trace_call_count_invalid")
        != len(calls)
        or start_update > end_update
        or process_id <= 0
    ):
        _fail("global_trace_contract_mismatch")
    for order, call in enumerate(calls):
        _verify_trace_call(call, expected_order=order)
        update = int(call["framework_update"])
        if not start_update <= update <= end_update:
            _fail("trace_call_outside_window")

    transitions = _sequence(
        alignment.get("transitions"),
        "trace_alignment_transitions_invalid",
    )
    flattened: list[tuple[int, int, int, int, int]] = []
    for expected_from, value in enumerate(
        transitions,
        start=start_update,
    ):
        transition = _mapping(value, "trace_alignment_transition_invalid")
        transition_calls = _sequence(
            transition.get("calls"),
            "trace_alignment_transition_calls_invalid",
        )
        if (
            transition.get("from_update") != expected_from
            or transition.get("to_update") != expected_from + 1
            or transition.get("call_count") != len(transition_calls)
        ):
            _fail("trace_alignment_transition_contract_mismatch")
        for raw in transition_calls:
            item = _mapping(raw, "trace_alignment_call_invalid")
            caller_hex = item.get("caller")
            if not isinstance(caller_hex, str):
                _fail("trace_alignment_caller_invalid")
            flattened.append(
                (
                    _integer(item.get("order"), "trace_alignment_order_invalid"),
                    expected_from + 1,
                    int(caller_hex, 16),
                    _integer(
                        item.get("mtrand_index_before"),
                        "trace_alignment_pre_index_invalid",
                    ),
                    _integer(
                        item.get("mtrand_index_after"),
                        "trace_alignment_post_index_invalid",
                    ),
                )
            )
    expected_flattened = [
        (
            int(call["order"]),
            int(call["framework_update"]),
            int(call["caller"]),
            int(call["mtrand_index_before"]),
            int(call["mtrand_index_after"]),
        )
        for call in calls
    ]
    if (
        alignment.get("schema") != TRACE_ALIGNMENT_SCHEMA
        or alignment.get("version") != 1
        or alignment.get("status") != "PASS"
        or alignment.get("runtime_executable_sha256")
        != EXPECTED_RETAIL_RUNTIME_SHA256
        or alignment.get("trace_artifact") != trace_path.name
        or alignment.get("trace_artifact_sha256") != trace_sha256
        or alignment.get("start_update") != start_update
        or alignment.get("end_update") != end_update
        or alignment.get("tick_count") != end_update - start_update + 1
        or alignment.get("transition_count") != len(transitions)
        or len(transitions) != end_update - start_update
        or alignment.get("call_count") != len(calls)
        or flattened != expected_flattened
    ):
        _fail("trace_alignment_contract_mismatch")

    overlap_source = _mapping(
        overlap.get("source"),
        "trace_overlap_source_invalid",
    )
    overlap_diagnostic = _mapping(
        overlap.get("diagnostic"),
        "trace_overlap_diagnostic_invalid",
    )
    probe_trajectory = _mapping(
        probe.get("trajectory"),
        "trace_probe_trajectory_invalid",
    )
    probe_trace = _mapping(
        probe.get("global_rng_call_trace"),
        "trace_probe_trace_invalid",
    )
    source_binding_transport = _verify_trace_probe_source_binding(
        window_root=window_root,
        probe=probe,
        process_id=process_id,
        source_probe_sha256=source_probe_sha256,
        dmo_sha256=dmo_sha256,
        start_update=start_update,
    )
    if (
        overlap.get("schema") != COMPACT_OVERLAP_SCHEMA
        or overlap.get("version") != 1
        or overlap.get("status") != "PASS"
        or overlap.get("start_update") != start_update
        or overlap.get("end_update") != end_update
        or overlap.get("tick_count") != end_update - start_update + 1
        or overlap.get("all_compact_projections_exact") is not True
        or overlap.get("all_global_mtrand_states_exact") is not True
        or overlap.get("diagnostic_compact_artifacts_revalidated") is not True
        or overlap.get("source_full_state_artifacts_revalidated") is not True
        or overlap.get("render_float32_bits_recomputed") is not True
        or overlap.get("first_mismatch") is not None
        or overlap_source.get("index_sha256") != source_index_sha256
        or Path(str(overlap_source.get("index"))).resolve()
        != source_index_path.resolve()
        or overlap_diagnostic.get("index_sha256")
        != probe_trajectory.get("artifact_sha256")
        or probe.get("schema") != MEMORY_PROBE_SCHEMA
        or probe.get("version") != 1
        or probe.get("runtime_executable_sha256")
        != EXPECTED_RETAIL_RUNTIME_SHA256
        or probe.get("dmo_sha256") != dmo_sha256
        or probe_trajectory.get("start_update") != start_update
        or probe_trajectory.get("end_update") != end_update
        or probe_trajectory.get("tick_count") != end_update - start_update + 1
        or probe_trace.get("schema") != GLOBAL_TRACE_SCHEMA
        or probe_trace.get("version") != 1
        or probe_trace.get("status") != "PASS"
        or probe_trace.get("artifact_sha256") != trace_sha256
        or probe_trace.get("call_count") != len(calls)
    ):
        _fail("trace_window_source_binding_mismatch")

    by_update: dict[int, list[Mapping[str, Any]]] = {}
    for call in calls:
        by_update.setdefault(int(call["framework_update"]), []).append(call)
    window_summary = {
            "root": window_relative.as_posix(),
            "start_update": start_update,
            "end_update": end_update,
            "tick_count": end_update - start_update + 1,
            "call_count": len(calls),
            "trace_sha256": trace_sha256,
            "alignment_sha256": alignment_sha256,
            "overlap_sha256": overlap_sha256,
            "memory_probe_sha256": probe_sha256,
            "diagnostic_trajectory_sha256": probe_trajectory[
                "artifact_sha256"
            ],
            "tick_digest_root": overlap.get("tick_digest_root"),
        }
    if source_binding_transport is not None:
        window_summary["source_binding_transport"] = source_binding_transport
    return (
        window_summary,
        {update: tuple(items) for update, items in by_update.items()},
    )


def _verify_window_against_source(
    *,
    window: Mapping[str, Any],
    calls_by_update: Mapping[int, tuple[Mapping[str, Any], ...]],
    frames_by_update: Mapping[int, Any],
) -> None:
    start = int(window["start_update"])
    end = int(window["end_update"])
    for update in range(start + 1, end + 1):
        previous = frames_by_update.get(update - 1)
        current = frames_by_update.get(update)
        if (
            previous is None
            or current is None
            or previous.global_mtrand is None
            or current.global_mtrand is None
        ):
            _fail("trace_window_source_frames_missing")
        rng = PopCapMTRandom(1)
        rng.load_state(
            previous.global_mtrand.words,
            previous.global_mtrand.index,
        )
        for call in calls_by_update.get(update, ()):
            if (
                call.get("mtrand_state_sha256_before")
                != _rng_state_sha256(rng)
                or call.get("mtrand_index_before") != rng.index
            ):
                _fail("trace_call_source_pre_state_mismatch")
            output = rng.next_u31()
            if (
                call.get("mtrand_output") != output
                or call.get("mtrand_index_after") != rng.index
            ):
                _fail("trace_call_source_output_mismatch")
        if (
            rng.words != current.global_mtrand.words
            or rng.index != current.global_mtrand.index
        ):
            _fail("trace_window_call_sequence_incomplete")


def _derive_from_report(
    report: Mapping[str, Any],
    *,
    evidence_root: Path,
    trace_window_roots: Sequence[str],
    runtime_artifact: str,
) -> dict[str, Any]:
    if (
        report.get("schema") != GAMEPLAY_DIFF_SCHEMA
        or report.get("version") not in {3, 4, 5}
        or report.get("status") != "PASS"
        or report.get("global_mtrand_policy") != BOUND_MTRAND_POLICY
        or report.get("global_mtrand_state_restored") is not True
    ):
        _fail("gameplay_diff_contract_mismatch")
    reconciliation = _mapping(
        report.get("global_mtrand_reconciliation"),
        "gameplay_diff_reconciliation_missing",
    )
    rows_value = _sequence(
        reconciliation.get("rows"),
        "gameplay_diff_reconciliation_rows_invalid",
    )
    rows = tuple(
        _mapping(value, "gameplay_diff_reconciliation_row_invalid")
        for value in rows_value
    )
    if (
        reconciliation.get("enabled") is not True
        or reconciliation.get("classification")
        != BOUND_MTRAND_CLASSIFICATION
        or reconciliation.get("state_binding") != MTRAND_STATE_BINDING
        or not rows
        or reconciliation.get("reconciled_tick_count") != len(rows)
    ):
        _fail("gameplay_diff_reconciliation_header_mismatch")

    trajectory = _mapping(
        report.get("trajectory"),
        "gameplay_diff_trajectory_missing",
    )
    source_index_path = _resolve(
        evidence_root,
        trajectory.get("artifact"),
        "gameplay_diff_trajectory_path_invalid",
    )
    source_index_sha256 = sha256_path(source_index_path)
    if trajectory.get("artifact_sha256") != source_index_sha256:
        _fail("gameplay_diff_trajectory_sha256_mismatch")
    source_probe_path = source_index_path.parent.parent / "memory-probe.json"
    if not source_probe_path.is_file():
        _fail("source_memory_probe_missing")
    source_probe_sha256 = sha256_path(source_probe_path)
    source_probe = _read_json(source_probe_path)
    source_probe_trajectory = _mapping(
        source_probe.get("trajectory"),
        "source_memory_probe_trajectory_missing",
    )
    dmo = _mapping(report.get("dmo"), "gameplay_diff_dmo_missing")
    dmo_sha256 = _sha256(
        dmo.get("artifact_sha256"),
        "gameplay_diff_dmo_sha256_invalid",
    )
    if (
        source_probe.get("schema") != MEMORY_PROBE_SCHEMA
        or source_probe.get("version") != 1
        or source_probe.get("runtime_executable_sha256")
        != EXPECTED_RETAIL_RUNTIME_SHA256
        or source_probe.get("dmo_sha256") != dmo_sha256
        or source_probe_trajectory.get("artifact_sha256")
        != source_index_sha256
    ):
        _fail("source_memory_probe_contract_mismatch")
    source_binding_transport: Mapping[str, Any] | None = None
    if dmo_sha256 != EXPECTED_DMO_SHA256:
        source_start_update = _integer(
            source_probe_trajectory.get("start_update"),
            "source_memory_probe_start_update_invalid",
            minimum=0,
        )
        source_process_id = _integer(
            source_probe.get("process_id"),
            "source_memory_probe_process_id_invalid",
            minimum=1,
        )
        source_binding_transport = _verify_trace_probe_source_binding(
            window_root=source_probe_path.parent,
            probe=source_probe,
            process_id=source_process_id,
            source_probe_sha256=source_probe_sha256,
            dmo_sha256=dmo_sha256,
            start_update=source_start_update,
        )
        if (
            source_binding_transport is None
            or not isinstance(
                source_binding_transport.get("retail_dmo_provenance"),
                Mapping,
            )
        ):
            _fail("source_memory_probe_retail_provenance_missing")

    frames = load_memory_trajectory(source_index_path)
    frames_by_update = {frame.update: frame for frame in frames}
    if len(frames_by_update) != len(frames):
        _fail("source_trajectory_duplicate_updates")

    if not trace_window_roots:
        _fail("trace_windows_missing")
    if len(set(trace_window_roots)) != len(trace_window_roots):
        _fail("trace_window_roots_duplicate")
    verified_windows: list[Mapping[str, Any]] = []
    window_calls: list[dict[int, tuple[Mapping[str, Any], ...]]] = []
    for window_root in trace_window_roots:
        window, calls_by_update = _verify_trace_window(
            root=evidence_root,
            window_root_value=window_root,
            source_index_path=source_index_path,
            source_index_sha256=source_index_sha256,
            source_probe_sha256=source_probe_sha256,
            dmo_sha256=dmo_sha256,
        )
        _verify_window_against_source(
            window=window,
            calls_by_update=calls_by_update,
            frames_by_update=frames_by_update,
        )
        verified_windows.append(window)
        window_calls.append(calls_by_update)
    if source_binding_transport is not None:
        for window in verified_windows:
            window_transport = window.get("source_binding_transport")
            window_provenance = (
                window_transport.get("retail_dmo_provenance")
                if isinstance(window_transport, Mapping)
                else None
            )
            if (
                not isinstance(window_provenance, Mapping)
                or window_provenance.get("playback_dmo_sha256")
                != dmo_sha256
            ):
                _fail("trace_window_retail_provenance_missing")

    runtime_relative = _relative_path(
        runtime_artifact,
        "runtime_artifact_path_invalid",
    )
    runtime_path = _resolve(
        evidence_root,
        runtime_relative.as_posix(),
        "runtime_artifact_path_invalid",
    )
    required_callers = {
        int(call["caller"])
        for calls_by_update in window_calls
        for calls in calls_by_update.values()
        for call in calls
    }
    require_pil_particle_surface = any(
        _call_requires_pil_particle_static_surface(call)
        for calls_by_update in window_calls
        for calls in calls_by_update.values()
        for call in calls
    )
    static_surface = _verify_runtime_static_surface(
        runtime_path,
        required_callers=required_callers,
        require_pil_particle_surface=require_pil_particle_surface,
    )

    ticks = _sequence(report.get("ticks"), "gameplay_diff_ticks_invalid")
    ticks_by_update: dict[int, Mapping[str, Any]] = {}
    for value in ticks:
        tick = _mapping(value, "gameplay_diff_tick_invalid")
        update = _integer(
            tick.get("framework_update"),
            "gameplay_diff_tick_update_invalid",
        )
        if update in ticks_by_update:
            _fail("gameplay_diff_tick_update_duplicate")
        ticks_by_update[update] = tick

    proof_rows: list[dict[str, Any]] = []
    visual_total = 0
    gameplay_total = 0
    semantic_counts: Counter[str] = Counter()
    seen_updates: set[int] = set()
    for reconciliation_row in rows:
        update = _integer(
            reconciliation_row.get("framework_update"),
            "reconciliation_update_invalid",
        )
        draw_count = _integer(
            reconciliation_row.get("draw_count"),
            "reconciliation_draw_count_invalid",
            minimum=1,
        )
        if (
            update in seen_updates
            or reconciliation_row.get("phase")
            != "after_gameplay_state_match"
        ):
            _fail("reconciliation_row_scope_unsupported")
        seen_updates.add(update)
        matching_windows = [
            (window, calls_by_update)
            for window, calls_by_update in zip(
                verified_windows,
                window_calls,
            )
            if int(window["start_update"])
            <= update
            <= int(window["end_update"])
        ]
        if len(matching_windows) != 1:
            _fail("reconciliation_update_trace_coverage_not_unique")
        window, calls_by_update = matching_windows[0]
        calls = calls_by_update.get(update, ())
        if not calls:
            _fail("reconciliation_update_has_no_native_calls")

        previous = frames_by_update.get(update - 1)
        current = frames_by_update.get(update)
        tick = ticks_by_update.get(update)
        if (
            previous is None
            or current is None
            or previous.global_mtrand is None
            or current.global_mtrand is None
            or tick is None
        ):
            _fail("reconciliation_source_state_missing")

        classified: list[dict[str, Any]] = []
        scopes: list[str] = []
        for call in calls:
            scope, semantic = classify_native_call(call)
            scopes.append(scope)
            semantic_counts[semantic] += 1
            classified.append(
                {
                    "source_order": call["order"],
                    "caller": call["caller"],
                    "caller_hex": call["caller_hex"],
                    "pre_index": call["mtrand_index_before"],
                    "post_index": call["mtrand_index_after"],
                    "pre_state_sha256": call[
                        "mtrand_state_sha256_before"
                    ],
                    "output": call["mtrand_output"],
                    "scope": scope,
                    "semantic": semantic,
                }
            )
        visual_count = scopes.count("omitted_visual")
        gameplay_count = scopes.count("modeled_gameplay")
        if (
            visual_count != draw_count
            or gameplay_count != len(calls) - draw_count
            or gameplay_count > 1
        ):
            _fail("reconciliation_visual_omission_count_mismatch")

        simulator_before = PopCapMTRandom(1)
        simulator_before.load_state(
            previous.global_mtrand.words,
            previous.global_mtrand.index,
        )
        for _ in range(gameplay_count):
            simulator_before.next_u31()
        source_previous_sha256 = mtrand_state_sha256(
            previous.global_mtrand
        )
        source_current_sha256 = mtrand_state_sha256(current.global_mtrand)
        if (
            reconciliation_row.get("simulator_state_sha256_before")
            != _rng_state_sha256(simulator_before)
            or reconciliation_row.get("simulator_index_before")
            != simulator_before.index
            or reconciliation_row.get("pc_state_sha256")
            != source_current_sha256
            or reconciliation_row.get("simulator_state_sha256_after")
            != source_current_sha256
            or reconciliation_row.get("simulator_index_after")
            != current.global_mtrand.index
            or tick.get("global_mtrand_reconciled_draws") != draw_count
            or tick.get("global_mtrand_leading_reconciled_draws") != 0
            or tick.get("gameplay_state_matched_before_global_rng") is not True
            or tick.get("global_mtrand_observed_match") is not True
        ):
            _fail("reconciliation_diff_state_binding_mismatch")

        proof_rows.append(
            {
                "framework_update": update,
                "phase": "after_gameplay_state_match",
                "trace_window_root": window["root"],
                "source_previous_state_sha256": source_previous_sha256,
                "simulator_state_before_reconciliation_sha256": (
                    reconciliation_row["simulator_state_sha256_before"]
                ),
                "source_current_state_sha256": source_current_sha256,
                "native_call_count": len(calls),
                "modeled_gameplay_call_count": gameplay_count,
                "omitted_visual_call_count": visual_count,
                "reconciled_draw_count": draw_count,
                "native_call_order": classified,
            }
        )
        visual_total += visual_count
        gameplay_total += gameplay_count

    declared_total = _integer(
        reconciliation.get("total_reconciled_draw_count"),
        "reconciliation_total_draw_count_invalid",
    )
    if visual_total != declared_total:
        _fail("reconciliation_total_visual_count_mismatch")

    source_trajectory_result: dict[str, Any] = {
        "artifact": _relative_path(
            trajectory["artifact"],
            "gameplay_diff_trajectory_path_invalid",
        ).as_posix(),
        "artifact_sha256": source_index_sha256,
        "memory_probe_sha256": source_probe_sha256,
        "captured_start_update": trajectory.get("captured_start_update"),
        "captured_end_update": trajectory.get("captured_end_update"),
    }
    if source_binding_transport is not None:
        source_trajectory_result["source_binding_transport"] = (
            source_binding_transport
        )

    headless_systems_omitted = [
        "pi_effect_particle_update",
        "score_digit_transition",
        "score_digit_wobble",
        "shadow_canopy_animation",
    ]
    if semantic_counts["pil_particle_emitter_update"] > 0:
        headless_systems_omitted.append("pil_particle_emitter_update")

    return {
        "schema": MTRAND_RECONCILIATION_PROOF_SCHEMA,
        "version": MTRAND_RECONCILIATION_PROOF_VERSION,
        "status": "PASS",
        "failure": None,
        "classification": MTRAND_RECONCILIATION_PROOF_CLASSIFICATION,
        "gameplay_diff_binding": {
            "schema": report["schema"],
            "version": report["version"],
            "start_update": report["start_update"],
            "end_update": report["end_update"],
            "compared_tick_count": report["compared_tick_count"],
            "trajectory_sha256": source_index_sha256,
            "dmo_sha256": dmo_sha256,
            "reconciliation_transcript_sha256": (
                reconciliation_transcript_sha256(report)
            ),
        },
        "scope": {
            "headless_systems_omitted": headless_systems_omitted,
            "modeled_gameplay_consumers": sorted(
                semantic
                for semantic, count in semantic_counts.items()
                if count > 0 and semantic in set(_GAMEPLAY_CALLERS.values())
            ),
            "full_native_state_exact_after_each_reconciliation": True,
            "standalone_seed_exact_without_visual_rng_model": False,
            "claim": (
                "gameplay-state and draw-distribution fidelity modulo "
                "proven visual-only shared-stream consumers"
            ),
        },
        "source_trajectory": source_trajectory_result,
        "runtime": {
            "artifact": runtime_relative.as_posix(),
            "artifact_sha256": EXPECTED_RETAIL_RUNTIME_SHA256,
            "static_surface": static_surface,
        },
        "trace_windows": verified_windows,
        "reconciliation": {
            "reconciled_tick_count": len(proof_rows),
            "reconciled_draw_count": visual_total,
            "modeled_gameplay_call_count": gameplay_total,
            "native_call_count": visual_total + gameplay_total,
            "semantic_call_counts": dict(sorted(semantic_counts.items())),
            "rows": proof_rows,
        },
    }


def derive_mtrand_reconciliation_proof(
    report: Mapping[str, Any],
    *,
    evidence_root: Path,
    trace_window_roots: Sequence[str],
    runtime_artifact: str,
) -> dict[str, Any]:
    """Derive a deterministic proof from immutable native trace artifacts."""

    return _derive_from_report(
        report,
        evidence_root=evidence_root.resolve(),
        trace_window_roots=trace_window_roots,
        runtime_artifact=runtime_artifact,
    )


def verify_mtrand_reconciliation_proof_binding(
    proof_path: Path,
    *,
    report: Mapping[str, Any],
    evidence_root: Path,
) -> Mapping[str, Any]:
    """Recompute a proof and require byte-semantic equality with its artifact."""

    proof = _read_json(proof_path.resolve())
    if (
        proof.get("schema") != MTRAND_RECONCILIATION_PROOF_SCHEMA
        or proof.get("version") != MTRAND_RECONCILIATION_PROOF_VERSION
        or proof.get("status") != "PASS"
        or proof.get("failure") is not None
        or proof.get("classification")
        != MTRAND_RECONCILIATION_PROOF_CLASSIFICATION
    ):
        _fail("mtrand_reconciliation_proof_header_mismatch")
    runtime = _mapping(proof.get("runtime"), "proof_runtime_missing")
    windows = _sequence(proof.get("trace_windows"), "proof_windows_missing")
    roots = [
        _mapping(window, "proof_window_invalid").get("root")
        for window in windows
    ]
    if any(not isinstance(root, str) for root in roots):
        _fail("proof_window_root_invalid")
    expected = _derive_from_report(
        report,
        evidence_root=evidence_root.resolve(),
        trace_window_roots=[str(root) for root in roots],
        runtime_artifact=str(runtime.get("artifact")),
    )
    if proof != expected:
        _fail("mtrand_reconciliation_proof_recomputation_mismatch")
    return proof


__all__ = [
    "EXPECTED_RETAIL_RUNTIME_SHA256",
    "MTRandReconciliationProofError",
    "MTRAND_RECONCILIATION_PROOF_CLASSIFICATION",
    "MTRAND_RECONCILIATION_PROOF_SCHEMA",
    "MTRAND_RECONCILIATION_PROOF_VERSION",
    "classify_native_call",
    "derive_mtrand_reconciliation_proof",
    "mtrand_state_sha256",
    "reconciliation_transcript_sha256",
    "sha256_path",
    "verify_mtrand_reconciliation_proof_binding",
]
