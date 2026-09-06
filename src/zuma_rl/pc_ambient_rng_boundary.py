"""Prove a natural gameplay slice ends at a retail ambient-RNG boundary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
from typing import Any, Mapping, Sequence

from zuma_rl.pc_golden import PcGoldenManifest
from zuma_rl.pc_memory_trajectory import (
    TrajectoryFrame,
    TrajectoryMTRandState,
    load_memory_trajectory,
)
from zuma_rl.retail_dmo_provenance import (
    RetailDmoProvenanceError,
    certifying_recording_outcome,
)
from zuma_rl.revenge_core import PopCapMTRandom


AMBIENT_RNG_BOUNDARY_SCHEMA = "zuma-rl.pc-ambient-mtrand-boundary"
AMBIENT_RNG_BOUNDARY_VERSION = 1
AMBIENT_RNG_BOUNDARY_CLASSIFICATION = (
    "natural-gameplay-slice-boundary-before-shadow-canopy-shared-rng"
)
EXPECTED_RETAIL_RUNTIME_SHA256 = (
    "sha256:"
    "2181ce2bfbfcb4678bf69a1474e08d3db941311aa768176a88453cc69692af20"
)
GLOBAL_TRACE_SCHEMA = "zuma-rl.pc-gameplay-mtrand-call-trace"
EXPECTED_BOUNDARY_CALLERS = (
    0x00458CD1,
    0x004B4B78,
    0x00402111,
    0x005A6630,
    0x005A66A4,
)
SHADOW_CANOPY_CALLERS = EXPECTED_BOUNDARY_CALLERS[-2:]
SHADOW_CANOPY_UPDATE_VA = 0x005A63F0
SHADOW_CANOPY_NAME_VA = 0x00982E34
SHADOW_CANOPY_VTABLE_SLOT_VA = 0x00982E5C
GLOBAL_MTRAND_WRAPPER_VA = 0x00617490
MTRAND_WORD_COUNT = 624


class AmbientRngBoundaryError(ValueError):
    """A boundary input or its provenance chain is invalid."""


def _fail(code: str) -> None:
    raise AmbientRngBoundaryError(code)


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
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AmbientRngBoundaryError(
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
        raise AmbientRngBoundaryError(f"artifact_unreadable:{path}") from error
    return "sha256:" + digest.hexdigest()


def _mapping(value: Any, code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
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


def mtrand_state_sha256(state: TrajectoryMTRandState) -> str:
    if (
        len(state.words) != MTRAND_WORD_COUNT
        or not 0 <= state.index <= MTRAND_WORD_COUNT
        or any(not 0 <= word <= 0xFFFFFFFF for word in state.words)
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


def _verify_trace_contract(
    trace: Mapping[str, Any],
    *,
    trace_sha256: str,
) -> tuple[Mapping[str, Any], ...]:
    if (
        trace.get("schema") != GLOBAL_TRACE_SCHEMA
        or trace.get("version") != 1
        or trace.get("status") != "PASS"
        or trace.get("failure") is not None
        or trace.get("breakpoint_kind") != "global_wrapper_entry"
        or trace.get("runtime_executable_sha256")
        != EXPECTED_RETAIL_RUNTIME_SHA256
        or trace.get("process_memory_writes") != 0
        or trace.get("persistent_file_modified") is not False
        or trace.get("hardware_breakpoint_restored") is not True
        or trace.get("stop_break_requested") is not True
        or trace.get("stop_break_observed") is not True
        or _integer(trace.get("address"), "trace_address_invalid")
        != GLOBAL_MTRAND_WRAPPER_VA
    ):
        _fail("global_trace_contract_mismatch")
    if _sha256(trace_sha256, "trace_sha256_invalid") != trace_sha256:
        _fail("trace_sha256_invalid")
    main_thread = _integer(
        trace.get("main_thread_id"),
        "trace_main_thread_invalid",
        minimum=1,
    )
    calls_value = trace.get("calls")
    if not isinstance(calls_value, list):
        _fail("trace_calls_invalid")
    call_count = _integer(
        trace.get("call_count"),
        "trace_call_count_invalid",
        minimum=1,
    )
    if call_count != len(calls_value):
        _fail("trace_call_count_mismatch")
    calls: list[Mapping[str, Any]] = []
    previous_update = -1
    previous_native = -1
    for order, value in enumerate(calls_value):
        call = _mapping(value, "trace_call_invalid")
        pre_hash = _sha256(
            call.get("pre_state_sha256"),
            "trace_pre_state_sha256_invalid",
        )
        post_hash = _sha256(
            call.get("post_state_sha256"),
            "trace_post_state_sha256_invalid",
        )
        update = _integer(
            call.get("framework_update"),
            "trace_framework_update_invalid",
            minimum=0,
        )
        native_value = call.get("native_game_time")
        if native_value is None:
            if previous_native >= 0:
                _fail("trace_native_game_time_invalid")
            native = -1
        else:
            native = _integer(
                native_value,
                "trace_native_game_time_invalid",
                minimum=0,
            )
        if (
            _integer(call.get("order"), "trace_order_invalid") != order
            or _integer(call.get("thread_id"), "trace_thread_invalid")
            != main_thread
            or not 0
            <= _integer(call.get("pre_index"), "trace_pre_index_invalid")
            <= MTRAND_WORD_COUNT
            or not 0
            <= _integer(call.get("post_index"), "trace_post_index_invalid")
            <= MTRAND_WORD_COUNT
            or _integer(call.get("caller"), "trace_caller_invalid", minimum=1)
            <= 0
            or update < previous_update
            or native < previous_native
        ):
            _fail("global_trace_call_contract_mismatch")
        previous_update = update
        previous_native = native
        calls.append(call)
    return tuple(calls)


def _select_boundary_calls(
    calls: Sequence[Mapping[str, Any]],
    *,
    first_excluded_update: int,
    before_state_sha256: str,
    after_state_sha256: str,
    native_game_time: int,
    score: int,
    score_target: int,
) -> tuple[Mapping[str, Any], ...]:
    selected = tuple(
        call
        for call in calls
        if call.get("framework_update") == first_excluded_update
    )
    if (
        len(selected) != len(EXPECTED_BOUNDARY_CALLERS)
        or tuple(call.get("caller") for call in selected)
        != EXPECTED_BOUNDARY_CALLERS
        or selected[0].get("pre_state_sha256") != before_state_sha256
        or selected[-1].get("post_state_sha256") != after_state_sha256
        or any(
            left.get("post_state_sha256")
            != right.get("pre_state_sha256")
            for left, right in zip(selected, selected[1:])
        )
        or any(
            call.get("native_game_time") != native_game_time
            or call.get("score") != score
            or call.get("score_target") != score_target
            for call in selected
        )
    ):
        _fail("ambient_boundary_call_sequence_mismatch")
    return selected


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


def _verify_shadow_canopy_runtime(payload: bytes) -> Mapping[str, Any]:
    image_base, sections = _pe_layout(payload)
    function_prefix = _read_runtime_va(
        payload,
        image_base=image_base,
        sections=sections,
        virtual_address=SHADOW_CANOPY_UPDATE_VA,
        size=3,
    )
    name = _read_runtime_va(
        payload,
        image_base=image_base,
        sections=sections,
        virtual_address=SHADOW_CANOPY_NAME_VA,
        size=len(b"ShadowCanopy1\0"),
    )
    vtable_target = struct.unpack(
        "<I",
        _read_runtime_va(
            payload,
            image_base=image_base,
            sections=sections,
            virtual_address=SHADOW_CANOPY_VTABLE_SLOT_VA,
            size=4,
        ),
    )[0]
    if (
        image_base != 0x00400000
        or function_prefix != b"\x55\x8b\xec"
        or name != b"ShadowCanopy1\0"
        or vtable_target != SHADOW_CANOPY_UPDATE_VA
    ):
        _fail("shadow_canopy_runtime_identity_mismatch")

    call_sites: list[dict[str, Any]] = []
    for return_address in SHADOW_CANOPY_CALLERS:
        call_address = return_address - 5
        instruction = _read_runtime_va(
            payload,
            image_base=image_base,
            sections=sections,
            virtual_address=call_address,
            size=5,
        )
        if instruction[0] != 0xE8:
            _fail("shadow_canopy_rng_call_opcode_mismatch")
        displacement = struct.unpack("<i", instruction[1:])[0]
        target = return_address + displacement
        if target != GLOBAL_MTRAND_WRAPPER_VA:
            _fail("shadow_canopy_rng_call_target_mismatch")
        call_sites.append(
            {
                "call_address": call_address,
                "return_address": return_address,
                "instruction_hex": instruction.hex(),
                "target_address": target,
            }
        )
    return {
        "class_name": "ShadowCanopy1",
        "class_name_virtual_address": SHADOW_CANOPY_NAME_VA,
        "update_function_virtual_address": SHADOW_CANOPY_UPDATE_VA,
        "vtable_slot_virtual_address": SHADOW_CANOPY_VTABLE_SLOT_VA,
        "vtable_slot_target": vtable_target,
        "global_mtrand_wrapper_virtual_address": GLOBAL_MTRAND_WRAPPER_VA,
        "call_sites": call_sites,
    }


def _verified_case_artifact(
    manifest: PcGoldenManifest,
    root: Path,
    name: str,
) -> Path:
    artifact = manifest.artifacts.get(name)
    if artifact is None:
        _fail(f"pc_golden_artifact_missing:{name}")
    try:
        return artifact.verify(root)
    except (OSError, ValueError) as error:
        raise AmbientRngBoundaryError(
            f"pc_golden_artifact_invalid:{name}"
        ) from error


def _verify_seed_anchor(
    plan: Mapping[str, Any],
    *,
    calls: Sequence[Mapping[str, Any]],
    source_dmo_sha256: str,
    recording_report_sha256: str,
    trace_sha256: str,
) -> Mapping[str, Any]:
    trace_plan = _mapping(
        plan.get("trace"),
        "collector_plan_trace_invalid",
    )
    anchor = _mapping(
        trace_plan.get("source_bound_board_anchor"),
        "collector_plan_source_anchor_missing",
    )
    pre_call = _mapping(
        trace_plan.get("source_bound_board_precall_global_restore"),
        "collector_plan_source_precall_missing",
    )
    global_state = _mapping(
        pre_call.get("global_state"),
        "collector_plan_source_global_state_missing",
    )
    source_order = _integer(
        anchor.get("source_order"),
        "collector_plan_source_order_invalid",
        minimum=0,
    )
    if source_order >= len(calls):
        _fail("collector_plan_source_order_out_of_range")
    call = calls[source_order]
    seed = _integer(
        anchor.get("global_seed"),
        "collector_plan_global_seed_invalid",
        minimum=0,
    )
    draw_count = _integer(
        global_state.get("draw_count"),
        "collector_plan_global_draw_count_invalid",
        minimum=0,
    )
    rng = PopCapMTRandom(seed)
    for _ in range(draw_count):
        rng.next_u31()
    pre_hash = _rng_state_sha256(rng)
    output = rng.next_u31()
    post_hash = _rng_state_sha256(rng)
    if (
        anchor.get("source_dmo_sha256") != source_dmo_sha256
        or anchor.get("recording_report_sha256")
        != recording_report_sha256
        or anchor.get("trace_sha256") != trace_sha256
        or global_state.get("seed") != seed
        or global_state.get("state_sha256") != pre_hash
        or call.get("pre_state_sha256") != pre_hash
        or call.get("post_state_sha256") != post_hash
        or call.get("output") != output
        or call.get("caller") != anchor.get("caller")
        or call.get("framework_update") != anchor.get("framework_update")
        or call.get("post_index") != anchor.get("expected_post_index")
        or output != anchor.get("expected_output")
        or post_hash != anchor.get("expected_post_state_sha256")
    ):
        _fail("collector_plan_source_anchor_mismatch")
    return {
        "global_seed": seed,
        "draw_count_before_board_seed": draw_count,
        "source_order": source_order,
        "framework_update": call["framework_update"],
        "caller": call["caller"],
        "output": output,
        "pre_state_sha256": pre_hash,
        "post_state_sha256": post_hash,
    }


def derive_ambient_rng_boundary(
    *,
    trajectory_path: Path,
    memory_probe_path: Path,
    pc_golden_manifest_path: Path,
    trace_path: Path,
    runtime_executable_path: Path,
    last_included_update: int,
    first_excluded_update: int,
) -> dict[str, Any]:
    """Derive an immutable proof for one natural ambient-RNG boundary."""

    trajectory_path = trajectory_path.resolve()
    memory_probe_path = memory_probe_path.resolve()
    pc_golden_manifest_path = pc_golden_manifest_path.resolve()
    trace_path = trace_path.resolve()
    runtime_executable_path = runtime_executable_path.resolve()
    if (
        last_included_update < 0
        or first_excluded_update != last_included_update + 1
    ):
        _fail("ambient_boundary_update_range_invalid")

    trajectory_sha256 = sha256_path(trajectory_path)
    frames = load_memory_trajectory(trajectory_path)
    by_update = {frame.update: frame for frame in frames}
    before = by_update.get(last_included_update)
    after = by_update.get(first_excluded_update)
    if before is None or after is None:
        _fail("ambient_boundary_frames_missing")
    if before.global_mtrand is None or after.global_mtrand is None:
        _fail("ambient_boundary_global_mtrand_missing")
    if after.native_game_time is None:
        _fail("ambient_boundary_native_game_time_missing")
    before_hash = mtrand_state_sha256(before.global_mtrand)
    after_hash = mtrand_state_sha256(after.global_mtrand)

    memory_probe_sha256 = sha256_path(memory_probe_path)
    memory_probe = _read_json(memory_probe_path)
    probe_trajectory = _mapping(
        memory_probe.get("trajectory"),
        "memory_probe_trajectory_missing",
    )
    probe_trajectory_path = (
        memory_probe_path.parent / str(probe_trajectory.get("artifact"))
    ).resolve()
    if (
        memory_probe.get("schema") != "zuma-rl.pc-memory-int32-probe"
        or memory_probe.get("version") != 1
        or probe_trajectory_path != trajectory_path
        or probe_trajectory.get("artifact_sha256") != trajectory_sha256
        or probe_trajectory.get("start_update") != frames[0].update
        or probe_trajectory.get("end_update") != frames[-1].update
        or probe_trajectory.get("tick_count") != len(frames)
        or memory_probe.get("runtime_executable_sha256")
        != EXPECTED_RETAIL_RUNTIME_SHA256
        or memory_probe.get("diagnostic_mutation") is not None
        or memory_probe.get("gameplay_mtrand_sync") is not None
        or memory_probe.get("global_rng_call_trace") is not None
    ):
        _fail("memory_probe_provenance_mismatch")

    try:
        manifest = PcGoldenManifest.read_json(pc_golden_manifest_path)
    except (OSError, ValueError) as error:
        raise AmbientRngBoundaryError("pc_golden_manifest_invalid") from error
    case_root = pc_golden_manifest_path.parent
    replay_dmo = _verified_case_artifact(
        manifest,
        case_root,
        manifest.input_timeline.artifact,
    )
    source_dmo = _verified_case_artifact(
        manifest,
        case_root,
        "input.recording.dmo",
    )
    recording_report_path = _verified_case_artifact(
        manifest,
        case_root,
        "input.recording.report",
    )
    normalization_path = _verified_case_artifact(
        manifest,
        case_root,
        "input.transport.normalization",
    )
    collector_plan_path = _verified_case_artifact(
        manifest,
        case_root,
        "collector.plan",
    )
    replay_dmo_sha256 = sha256_path(replay_dmo)
    source_dmo_sha256 = sha256_path(source_dmo)
    recording_report_sha256 = sha256_path(recording_report_path)
    collector_plan_sha256 = sha256_path(collector_plan_path)
    if memory_probe.get("dmo_sha256") != replay_dmo_sha256:
        _fail("memory_probe_replay_dmo_mismatch")

    normalization = _read_json(normalization_path)
    norm_source = _mapping(
        normalization.get("source"),
        "dmo_normalization_source_missing",
    )
    norm_output = _mapping(
        normalization.get("output"),
        "dmo_normalization_output_missing",
    )
    transformation = _mapping(
        normalization.get("transformation"),
        "dmo_normalization_transformation_missing",
    )
    norm_plan = _mapping(
        normalization.get("collector_plan"),
        "dmo_normalization_plan_missing",
    )
    if (
        normalization.get("schema")
        != "zuma-rl.retail-dmo-transport-normalization"
        or normalization.get("version") not in {1, 2}
        or normalization.get("classification")
        != "certifying-retail-recording-transport-normalization"
        or norm_source.get("sha256") != source_dmo_sha256
        or norm_output.get("sha256") != replay_dmo_sha256
        or norm_plan.get("sha256") != collector_plan_sha256
        or transformation.get("kind")
        != "remove_bit_identical_duplicate_startup_registry_read"
        or transformation.get("gameplay_input_commands_preserved") is not True
        or transformation.get("remaining_command_bits_preserved") is not True
        or transformation.get("random_seed_preserved") is not True
        or transformation.get("length_updates_preserved") is not True
    ):
        _fail("dmo_normalization_contract_mismatch")

    recording = _read_json(recording_report_path)
    try:
        certifying_recording_outcome(recording)
    except RetailDmoProvenanceError as error:
        raise AmbientRngBoundaryError(
            "recording_outcome_contract_mismatch"
        ) from error
    recording_trace = _mapping(
        recording.get("gameplay_mtrand_trace"),
        "recording_global_trace_missing",
    )
    recording_main_thread = _mapping(
        recording.get("main_thread"),
        "recording_main_thread_missing",
    )
    trace_sha256 = sha256_path(trace_path)
    trace = _read_json(trace_path)
    calls = _verify_trace_contract(trace, trace_sha256=trace_sha256)
    if (
        recording.get("schema") != "zuma-rl.retail-autoplay-recording"
        or recording.get("version") not in {1, 2}
        or recording.get("status") != "PASS"
        or recording.get("normal_exit") is not True
        or recording.get("host_restored") is not True
        or recording.get("process_memory_writes") != 0
        or recording.get("dmo_sha256") != source_dmo_sha256
        or recording.get("runtime_executable_sha256")
        != EXPECTED_RETAIL_RUNTIME_SHA256
        or recording_main_thread.get("source") != "verified_retail_window"
        or recording_main_thread.get("process_memory_writes") != 0
        or recording_main_thread.get("thread_id")
        != trace.get("main_thread_id")
        or recording_trace.get("status") != "PASS"
        or recording_trace.get("failure") is not None
        or recording_trace.get("breakpoint_kind") != "global_wrapper_entry"
        or recording_trace.get("output_sha256") != trace_sha256
        or recording_trace.get("call_count") != len(calls)
        or recording_trace.get("process_memory_writes") != 0
        or recording_trace.get("hardware_breakpoint_restored") is not True
    ):
        _fail("recording_trace_provenance_mismatch")

    collector_plan = _read_json(collector_plan_path)
    if (
        collector_plan.get("schema")
        != "zuma-rl.pc-golden-v4-collection-plan"
        or collector_plan.get("version") != 5
        or _mapping(
            collector_plan.get("dmo"),
            "collector_plan_dmo_missing",
        ).get("sha256")
        != replay_dmo_sha256
        or _mapping(
            collector_plan.get("runtime"),
            "collector_plan_runtime_missing",
        ).get("runtime_executable_sha256")
        != EXPECTED_RETAIL_RUNTIME_SHA256
    ):
        _fail("collector_plan_contract_mismatch")
    seed_anchor = _verify_seed_anchor(
        collector_plan,
        calls=calls,
        source_dmo_sha256=source_dmo_sha256,
        recording_report_sha256=recording_report_sha256,
        trace_sha256=trace_sha256,
    )

    boundary_calls = _select_boundary_calls(
        calls,
        first_excluded_update=first_excluded_update,
        before_state_sha256=before_hash,
        after_state_sha256=after_hash,
        native_game_time=after.native_game_time,
        score=after.score,
        score_target=after.score_target,
    )
    anchor_order = int(seed_anchor["source_order"])
    boundary_end_order = int(boundary_calls[-1]["order"])
    if any(
        left.get("post_state_sha256") != right.get("pre_state_sha256")
        for left, right in zip(
            calls[anchor_order : boundary_end_order + 1],
            calls[anchor_order + 1 : boundary_end_order + 1],
        )
    ):
        _fail("source_anchor_to_boundary_trace_not_contiguous")

    runtime_sha256 = sha256_path(runtime_executable_path)
    if runtime_sha256 != EXPECTED_RETAIL_RUNTIME_SHA256:
        _fail("runtime_executable_sha256_mismatch")
    try:
        runtime_payload = runtime_executable_path.read_bytes()
    except OSError as error:
        raise AmbientRngBoundaryError(
            "runtime_executable_unreadable"
        ) from error
    static_proof = _verify_shadow_canopy_runtime(runtime_payload)

    return {
        "schema": AMBIENT_RNG_BOUNDARY_SCHEMA,
        "version": AMBIENT_RNG_BOUNDARY_VERSION,
        "status": "PASS",
        "failure": None,
        "classification": AMBIENT_RNG_BOUNDARY_CLASSIFICATION,
        "last_included_update": last_included_update,
        "first_excluded_update": first_excluded_update,
        "scope": {
            "included": "chain_shooter_projectile_collision_and_score",
            "excluded": "shadow_canopy_ambient_visual_rng_consumers",
            "long_horizon_fidelity_claimed": False,
        },
        "trajectory": {
            "artifact": str(trajectory_path),
            "artifact_sha256": trajectory_sha256,
            "captured_start_update": frames[0].update,
            "captured_end_update": frames[-1].update,
            "last_included_global_mtrand_index": before.global_mtrand.index,
            "last_included_global_mtrand_sha256": before_hash,
            "first_excluded_global_mtrand_index": after.global_mtrand.index,
            "first_excluded_global_mtrand_sha256": after_hash,
        },
        "memory_probe": {
            "artifact": str(memory_probe_path),
            "artifact_sha256": memory_probe_sha256,
            "framework_update": memory_probe.get("framework_update"),
        },
        "pc_golden": {
            "case_id": manifest.case_id,
            "manifest": str(pc_golden_manifest_path),
            "manifest_sha256": sha256_path(pc_golden_manifest_path),
            "evidence_set_fingerprint": manifest.evidence_set_fingerprint,
            "collector_plan_sha256": collector_plan_sha256,
        },
        "replay_dmo": {
            "artifact": str(replay_dmo),
            "artifact_sha256": replay_dmo_sha256,
        },
        "source_recording": {
            "dmo_artifact": str(source_dmo),
            "dmo_sha256": source_dmo_sha256,
            "report_artifact": str(recording_report_path),
            "report_sha256": recording_report_sha256,
        },
        "seed_anchor": seed_anchor,
        "global_mtrand_call_trace": {
            "artifact": str(trace_path),
            "artifact_sha256": trace_sha256,
            "call_count": len(calls),
            "process_id": trace.get("process_id"),
            "main_thread_id": trace.get("main_thread_id"),
            "first_boundary_source_order": boundary_calls[0]["order"],
            "last_boundary_source_order": boundary_calls[-1]["order"],
        },
        "boundary_calls": [
            {
                "order": call["order"],
                "framework_update": call["framework_update"],
                "native_game_time": call["native_game_time"],
                "caller": call["caller"],
                "caller_hex": f"0x{call['caller']:08x}",
                "pre_index": call["pre_index"],
                "post_index": call["post_index"],
                "pre_state_sha256": call["pre_state_sha256"],
                "post_state_sha256": call["post_state_sha256"],
                "scope": (
                    "shadow_canopy_ambient_visual"
                    if call["caller"] in SHADOW_CANOPY_CALLERS
                    else "gameplay_or_framework"
                ),
            }
            for call in boundary_calls
        ],
        "runtime": {
            "artifact": str(runtime_executable_path),
            "artifact_sha256": runtime_sha256,
            "static_shadow_canopy_proof": static_proof,
        },
    }


def load_ambient_rng_boundary_report(path: Path) -> Mapping[str, Any]:
    """Load the compact contract needed to bind a gameplay diff."""

    report = _read_json(path.resolve())
    if (
        report.get("schema") != AMBIENT_RNG_BOUNDARY_SCHEMA
        or report.get("version") != AMBIENT_RNG_BOUNDARY_VERSION
        or report.get("status") != "PASS"
        or report.get("failure") is not None
        or report.get("classification")
        != AMBIENT_RNG_BOUNDARY_CLASSIFICATION
    ):
        _fail("ambient_boundary_report_contract_mismatch")
    trajectory = _mapping(
        report.get("trajectory"),
        "ambient_boundary_report_trajectory_missing",
    )
    replay_dmo = _mapping(
        report.get("replay_dmo"),
        "ambient_boundary_report_dmo_missing",
    )
    _sha256(
        trajectory.get("artifact_sha256"),
        "ambient_boundary_report_trajectory_sha256_invalid",
    )
    _sha256(
        replay_dmo.get("artifact_sha256"),
        "ambient_boundary_report_dmo_sha256_invalid",
    )
    _integer(
        report.get("last_included_update"),
        "ambient_boundary_report_last_update_invalid",
        minimum=0,
    )
    _integer(
        report.get("first_excluded_update"),
        "ambient_boundary_report_first_update_invalid",
        minimum=1,
    )
    return report


def bind_ambient_rng_boundary_to_diff(
    boundary: Mapping[str, Any],
    *,
    boundary_path: Path,
    trajectory_sha256: str,
    dmo_sha256: str,
    selected_end_update: int,
) -> Mapping[str, Any]:
    trajectory = _mapping(
        boundary.get("trajectory"),
        "ambient_boundary_report_trajectory_missing",
    )
    replay_dmo = _mapping(
        boundary.get("replay_dmo"),
        "ambient_boundary_report_dmo_missing",
    )
    if (
        trajectory.get("artifact_sha256") != trajectory_sha256
        or replay_dmo.get("artifact_sha256") != dmo_sha256
        or boundary.get("last_included_update") != selected_end_update
        or boundary.get("first_excluded_update") != selected_end_update + 1
    ):
        _fail("ambient_boundary_diff_binding_mismatch")
    return {
        "artifact": str(boundary_path.resolve()),
        "artifact_sha256": sha256_path(boundary_path.resolve()),
        "schema": boundary["schema"],
        "version": boundary["version"],
        "status": boundary["status"],
        "classification": boundary["classification"],
        "last_included_update": boundary["last_included_update"],
        "first_excluded_update": boundary["first_excluded_update"],
        "excluded_scope": _mapping(
            boundary.get("scope"),
            "ambient_boundary_report_scope_missing",
        ).get("excluded"),
    }


__all__ = [
    "AMBIENT_RNG_BOUNDARY_CLASSIFICATION",
    "AMBIENT_RNG_BOUNDARY_SCHEMA",
    "AMBIENT_RNG_BOUNDARY_VERSION",
    "AmbientRngBoundaryError",
    "bind_ambient_rng_boundary_to_diff",
    "derive_ambient_rng_boundary",
    "load_ambient_rng_boundary_report",
    "mtrand_state_sha256",
    "sha256_path",
]
