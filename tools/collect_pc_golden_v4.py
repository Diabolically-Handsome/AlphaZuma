"""Fail-closed Windows collector for a two-run PC Golden v4 transaction.

The collector has two explicit phases:

``prepare``
    Read-only with respect to the live game state.  It snapshots the current
    users/registry state, copies the selected DMO and reconstructed prestate
    inputs into a new ASCII session directory, and creates a protocol
    prestate template.

``collect``
    Restores that template before each of two independent retail replay
    processes, captures lossless DXGI evidence plus framework-update samples,
    records start/end state snapshots and process evidence, restores the
    protocol prestate, and finally restores the user's original host state in
    a ``finally`` path.

No existing evidence file is overwritten.  A failed session is intentionally
left in place for diagnosis and cannot be resumed as if it had completed.
"""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import struct
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from zuma_rl.pc_protocol_evidence import (
    PcDxgiCaptureMetadata,
    PcFrameworkUpdateMap,
    PcProcessEvent,
    PcProcessTimeline,
    PcSaveJournal,
    PcStateSnapshot,
)
from zuma_rl.pc_external_input import REPAINT_INPUT_EXTRA_INFO
from zuma_rl.pc_golden import SAVE_VOLATILE_REGISTRY_ROLES
from zuma_rl.popcap_dmo import PopCapDemo
from zuma_rl.original_data import PopCapPakArchive
from zuma_rl.revenge_core import PopCapMTRandom
from tools.capture_dxgi import (
    CaptureError,
    FrameworkUpdateReader,
    WindowTarget,
    estimated_raw_bytes,
    parse_duration,
    parse_frame_budget_fps,
    resolve_windows_target,
    verify_windows_target,
)
from tools.launch_popcap_replay import (
    DEFAULT_RUNTIME_EXE,
    DEFAULT_STEAM_EXE,
    process_ids_by_name,
    process_image_path,
    same_windows_path,
)
from tools.insert_popcap_dmo_failed_file_writes import (
    validate_successful_file_write_padding_provenance,
)
from tools.popcap_replay_boundary import (
    ATTACH_STABILIZATION_MAX_SKIPPED_HITS,
    ATTACH_STABILIZATION_MAX_UPDATE_DELTA,
    FONT_CACHE_MANIFEST_EXPECTED_ENTRIES,
    MAIN_FILE_READ_CORRIDOR_MAX_BYTES,
    MAIN_FILE_READ_CORRIDOR_MIN_BYTES,
    POST_BLACKOUT_FILE_WRITE_SPILLOVER_MAX_ROWS,
    SERVICE_EXIT_TIMELINE_MAX_GAP_UPDATES,
    SOURCE_BOUND_GLOBAL_CORRECTION_MAX_DRAWS,
    _normalize_font_cache_member,
)
from tools.popcap_global_mtrand_restore import (
    GlobalMTRandRestoreState,
    load_source_bound_global_mtrand_restore_state,
)
from tools.popcap_mtrand_call_oracle import (
    GameplayMTRandOracle,
    load_gameplay_mtrand_oracle,
)
from tools.popcap_thread_crt_restore import (
    ThreadCrtRestoreState,
    load_source_bound_thread_crt_restore_state,
)
from tools.pc_state_transaction import (
    StateTransactionError,
    capture_state,
    overlay_snapshot_files,
    overlay_snapshot_registry_values,
    restore_state,
    write_snapshot_exclusive,
)


PLAN_SCHEMA = "zuma-rl.pc-golden-v4-collection-plan"
PLAN_VERSION = 5
RESULT_SCHEMA = "zuma-rl.pc-golden-v4-collection-result"
RESULT_VERSION = 5
REPAINT_SCHEMA = "zuma-rl.pc-window-repaint-handshake"
REPAINT_VERSION = 1
NONCLIENT_DRAG_REPAINT_MECHANISM = (
    "nonclient_titlebar_drag_then_exact_origin_restore"
)
SET_WINDOW_POS_REPAINT_MECHANISM = (
    "set_window_pos_temporary_translation_then_exact_origin_restore"
)
EXPECTED_RUNTIME_SHA256 = (
    "sha256:2181ce2bfbfcb4678bf69a1474e08d3db941311aa768176a88453cc69692af20"
)
RUN_IDS = ("r1", "r2")
OVERLAY_NAMES = ("adv_in_game2.sav", "user2.dat", "users.dat")
MIN_FREE_BYTES = 8 * 1024**3
MAX_INTER_RUN_COOLDOWN_SECONDS = 180.0
DEFAULT_DMO = Path(
    r"D:\ZumaGolden\session-20260728-203153"
    r"\native-windowed-record-001-aligned-diagnostic\input.dmo"
)
DEFAULT_PRESTATE = Path(
    r"D:\ZumaGolden\session-20260728-203153"
    r"\native-windowed-record-001-prestate-reconstructed"
)
DEFAULT_RUNTIME_SOURCE = Path(
    r"D:\SteamLibrary\steamapps\common"
    r"\Zuma's Revenge\ZumasRevenge.exe"
)
DEFAULT_DIRECT_RUNTIME = Path(
    r"D:\ZumaGolden\tools\direct-runtime\popcapgame1.exe"
)
STEAM_LAUNCH_MODE = "steam_embedded_runtime"
DIRECT_FIXED_SEED_LAUNCH_MODE = "direct_byte_identical_fixed_seed"
DIRECT_NATURAL_SEED_LAUNCH_MODE = "direct_byte_identical_natural_seed"
DIRECT_NATURAL_STRICT_BROKER_LAUNCH_MODE = (
    "direct_byte_identical_natural_seed_strict_command_broker"
)
STARTUP_SEED_TRANSPORTS = frozenset(("debugger_register", "iat_stub"))
DEFAULT_BOARD_RESEED_CALL = 0x0065B828
ZUMA_REGISTRY_ROOT = r"HKCU\Software\SteamPopCap\ZumasRevenge"
RUNTIME_SCREEN_MODE_VALUE = "ScreenMode"
RUNTIME_SCREEN_MODES = frozenset((0, 1))
SOURCE_BOUND_GLOBAL_PRECALL_ADDRESS = 0x0065B81C
SOURCE_BOUND_GLOBAL_PRECALL_RETURN_ADDRESS = 0x0065B821
SOURCE_BOUND_GLOBAL_PRECALL_TARGET_ADDRESS = 0x00617490
SOURCE_BOUND_GLOBAL_PRECALL_INSTRUCTION_HEX = "e86fbcfbff"
BOARD_VTABLE = 0x0096356C
BOARD_SEED_OWNER_VTABLE = 0x009884A4
BOARD_MTRAND_OFFSET = 0x4C
GLOBAL_MTRAND_ADDRESS = 0x00A313C0
MTRAND_STATE_WORDS = 624
MTRAND_STATE_BYTES = (MTRAND_STATE_WORDS + 1) * 4
CRT_FLS_INDEX_ADDRESS = 0x009E5184
CRT_PTD_RAND_STATE_OFFSET = 0x14
FLS_MAXIMUM_AVAILABLE = 0x0FF0
STRICT_REPLAY_SERVICE_WAIT_TIMEOUT_SECONDS = 60
CAPTURE_WINDOW_TIMEOUT_SECONDS = 240.0
BLACKOUT_FILE_WRITE_REBASE_EXACT_MODE = "exact_all_rows"
BLACKOUT_FILE_WRITE_REBASE_ENVELOPE_MODE = "candidate_envelope"
BLACKOUT_FILE_WRITE_REBASE_ENVELOPE_SET_MODE = "candidate_envelope_set"
GAMEPLAY_MTRAND_ORACLE_ARTIFACT = "inputs/gameplay-mtrand-oracle.json"
GAMEPLAY_MTRAND_SYNC_CLASSIFICATION = (
    "diagnostic_ephemeral_process_state_synchronization"
)
FRAMEWORK_POLL_SCHEMA = "zuma-rl.pc-framework-state-poll-diagnostic"
FRAMEWORK_POLL_READY_SCHEMA = "zuma-rl.pc-framework-state-poll-ready"
FRAMEWORK_POLL_STOP_SCHEMA = "zuma-rl.pc-framework-state-poll-stop"
MIN_FRAMEWORK_POLL_HZ = 2_000.0
MAX_FRAMEWORK_POLL_P99_INTERVAL_NS = 2_000_000


class CollectionError(RuntimeError):
    """A stable, non-sensitive collection failure code."""

    def __init__(self, code: str) -> None:
        if not isinstance(code, str) or not code.replace("_", "").isalnum():
            code = "collection_failed"
        self.code = code
        super().__init__(code)


def _state_comparison_contract(launch_mode: str) -> dict[str, Any]:
    if launch_mode == STEAM_LAUNCH_MODE:
        launcher_registry_contract = "runtime_process_bound"
    elif launch_mode == DIRECT_FIXED_SEED_LAUNCH_MODE:
        launcher_registry_contract = "preserved_from_prestate"
    else:
        raise CollectionError("launch_mode_invalid")
    return {
        "projection": "gameplay_state_projection_v1",
        "volatile_registry_roles": list(SAVE_VOLATILE_REGISTRY_ROLES),
        "launcher_registry_contract": launcher_registry_contract,
    }


def _runtime_screen_mode_overlay(mode: int) -> dict[str, Any]:
    if isinstance(mode, bool) or mode not in RUNTIME_SCREEN_MODES:
        raise CollectionError("runtime_screen_mode_invalid")
    return {
        "root": ZUMA_REGISTRY_ROOT,
        "subkey": "",
        "value_name": RUNTIME_SCREEN_MODE_VALUE,
        "value_type": 4,
        "data_hex": struct.pack("<I", mode).hex(),
        "screen_mode": mode,
        "semantic": "fullscreen_800x600" if mode == 1 else "windowed_800x600",
    }


@dataclass(frozen=True, slots=True)
class RunEvidence:
    run_id: str
    start_snapshot: PcStateSnapshot
    end_snapshot: PcStateSnapshot
    process_id: int
    process_creation_filetime_100ns: int
    executable_sha256: str
    process_start_perf_counter_ns: int
    process_stop_perf_counter_ns: int
    exit_code: int
    capture_metadata: PcDxgiCaptureMetadata
    framework_updates: PcFrameworkUpdateMap
    window_repaint_sha256: str
    end_gameplay_state_root: str
    gameplay_mtrand_sync: Mapping[str, Any] | None
    framework_state_poll: Mapping[str, Any] | None

    @property
    def process_instance(self) -> tuple[int, int]:
        return self.process_id, self.process_creation_filetime_100ns


def _require_windows() -> None:
    if os.name != "nt":
        raise CollectionError("windows_required")


def _sha256_bytes(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _mtrand_state_sha256(seed: int) -> str:
    effective_seed = 4357 if seed == 0 else seed
    words = [effective_seed]
    for index in range(1, MTRAND_STATE_WORDS):
        previous = words[-1]
        words.append(
            (
                1812433253
                * (previous ^ (previous >> 30))
                + index
            )
            & 0xFFFFFFFF
        )
    words.append(MTRAND_STATE_WORDS)
    return _sha256_bytes(
        b"".join(word.to_bytes(4, "little") for word in words)
    )


def _source_bound_global_transition(
    state: GlobalMTRandRestoreState,
) -> tuple[int, int, str]:
    if len(state.payload) != MTRAND_STATE_BYTES:
        raise CollectionError("source_bound_anchor_invalid")
    unpacked = struct.unpack(f"<{MTRAND_STATE_WORDS + 1}I", state.payload)
    if unpacked[-1] > MTRAND_STATE_WORDS:
        raise CollectionError("source_bound_anchor_invalid")
    rng = PopCapMTRandom(1)
    rng.load_state(unpacked[:MTRAND_STATE_WORDS], unpacked[-1])
    output = rng.next_u31()
    post_payload = struct.pack(
        f"<{MTRAND_STATE_WORDS + 1}I",
        *rng.words,
        rng.index,
    )
    return output, rng.index, _sha256_bytes(post_payload)


def _load_source_bound_board_anchor(
    *,
    monitor_path: Path,
    trace_path: Path,
    recording_report_path: Path,
    monitor_framework_update: int,
    global_seed: int,
    rewind_draws: int,
    source_order: int,
    framework_update: int,
    caller: int,
) -> tuple[
    GlobalMTRandRestoreState,
    ThreadCrtRestoreState,
    dict[str, Any],
]:
    monitor_path = monitor_path.resolve()
    trace_path = trace_path.resolve()
    recording_report_path = recording_report_path.resolve()
    try:
        global_state = load_source_bound_global_mtrand_restore_state(
            monitor_path,
            trace_path,
            recording_report_path,
            monitor_framework_update=monitor_framework_update,
            seed=global_seed,
            rewind_draws=rewind_draws,
            source_order=source_order,
            framework_update=framework_update,
            caller=caller,
        )
        thread_state = load_source_bound_thread_crt_restore_state(
            trace_path,
            recording_report_path,
            source_order=source_order,
            framework_update=framework_update,
            caller=caller,
        )
        expected_output, post_index, post_sha256 = (
            _source_bound_global_transition(global_state)
        )
    except (OSError, TypeError, ValueError, CollectionError):
        raise CollectionError("source_bound_anchor_invalid") from None
    if (
        global_state.source_kind
        != "natural_retail_hardware_trace_and_monitor"
        or thread_state.source_kind != "natural_retail_hardware_trace"
        or global_state.source_trace_path != thread_state.source_path
        or global_state.source_trace_sha256 != thread_state.source_sha256
        or global_state.source_recording_report_path
        != thread_state.source_recording_report_path
        or global_state.source_recording_report_sha256
        != thread_state.source_recording_report_sha256
        or global_state.source_dmo_path != thread_state.source_dmo_path
        or global_state.source_dmo_sha256 != thread_state.source_dmo_sha256
        or global_state.source_process_id != thread_state.source_process_id
        or global_state.source_call_order != source_order
        or thread_state.source_call_order != source_order
        or global_state.source_caller != caller
        or thread_state.source_caller != caller
        or thread_state.source_framework_update != framework_update
        or caller != 0x0065B821
    ):
        raise CollectionError("source_bound_anchor_invalid")
    assert global_state.source_dmo_path is not None
    assert global_state.source_dmo_sha256 is not None
    anchor = {
        "monitor_path": str(monitor_path),
        "monitor_sha256": global_state.source_sha256,
        "trace_path": str(trace_path),
        "trace_sha256": global_state.source_trace_sha256,
        "recording_report_path": str(recording_report_path),
        "recording_report_sha256": (
            global_state.source_recording_report_sha256
        ),
        "source_dmo_path": str(global_state.source_dmo_path),
        "source_dmo_sha256": global_state.source_dmo_sha256,
        "monitor_framework_update": monitor_framework_update,
        "global_seed": global_seed,
        "rewind_draws": rewind_draws,
        "source_order": source_order,
        "framework_update": framework_update,
        "caller": caller,
        "expected_output": expected_output,
        "expected_post_index": post_index,
        "expected_post_state_sha256": post_sha256,
        "thread_crt_state": thread_state.state,
    }
    return global_state, thread_state, anchor


def _source_bound_board_precall_plan(
    global_state: GlobalMTRandRestoreState,
    *,
    framework_update: int,
    caller: int,
) -> dict[str, Any]:
    """Freeze the exact source-derived global RNG CALL boundary."""

    if (
        caller != SOURCE_BOUND_GLOBAL_PRECALL_RETURN_ADDRESS
        or framework_update < 0
    ):
        raise CollectionError("source_bound_board_precall_invalid")
    return {
        "call_address": SOURCE_BOUND_GLOBAL_PRECALL_ADDRESS,
        "return_address": caller,
        "target_address": SOURCE_BOUND_GLOBAL_PRECALL_TARGET_ADDRESS,
        "instruction_hex": SOURCE_BOUND_GLOBAL_PRECALL_INSTRUCTION_HEX,
        "framework_update": framework_update,
        "allow_dynamic_before": True,
        "suspend_other_threads_until_board_call": True,
        "source_process_id": global_state.source_process_id,
        "source_native_game_time": global_state.source_native_game_time,
        "global_state": {
            **global_state.semantic_dict(),
            "semantic_sha256": global_state.semantic_sha256,
        },
    }


def _sha256_file(path: Path) -> str:
    try:
        return _sha256_bytes(path.read_bytes())
    except OSError:
        raise CollectionError("artifact_read_failed") from None


def _load_gameplay_mtrand_oracle_for_plan(
    path: Path,
    *,
    seed: int,
    maximum_draws: int,
) -> GameplayMTRandOracle:
    try:
        return load_gameplay_mtrand_oracle(
            path,
            seed=seed,
            maximum_draws=maximum_draws,
        )
    except (OSError, TypeError, ValueError):
        raise CollectionError("gameplay_mtrand_oracle_invalid") from None


def _gameplay_mtrand_sync_plan(
    path: Path,
    *,
    seed: int,
    maximum_draws: int,
    artifact: str = GAMEPLAY_MTRAND_ORACLE_ARTIFACT,
) -> dict[str, Any]:
    oracle = _load_gameplay_mtrand_oracle_for_plan(
        path,
        seed=seed,
        maximum_draws=maximum_draws,
    )
    entries = oracle.entries
    return {
        "artifact": artifact,
        "sha256": oracle.source_sha256,
        "seed": seed,
        "maximum_draws": maximum_draws,
        "source_process_id": oracle.source_process_id,
        "entry_count": len(entries),
        "first_update": entries[0].framework_update,
        "last_update": entries[-1].framework_update,
        "semantic_sha256": oracle.semantic_sha256,
        "classification": GAMEPLAY_MTRAND_SYNC_CLASSIFICATION,
    }


def _gameplay_mtrand_trace_args(
    *,
    session_root: Path,
    run_root: Path,
    trace: Mapping[str, Any],
) -> list[str]:
    sync = trace.get("gameplay_mtrand_sync")
    if sync is None:
        return []
    oracle_path = _session_member(session_root, sync["artifact"])
    return [
        "--gameplay-mtrand-oracle",
        str(oracle_path),
        "--gameplay-mtrand-oracle-seed",
        str(sync["seed"]),
        "--gameplay-mtrand-oracle-maximum-draws",
        str(sync["maximum_draws"]),
        "--gameplay-mtrand-receipt-json",
        str(run_root / "gameplay-mtrand-sync.json"),
    ]


def _load_gameplay_mtrand_sync_receipt(
    path: Path,
    *,
    expected_process_id: int,
    expected_oracle: Path,
    expected_dmo: Path,
    artifact: str,
) -> dict[str, Any]:
    """Validate and compact the pre-blackout RNG synchronization receipt."""

    try:
        data = path.read_bytes()
        payload = json.loads(data.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise CollectionError(
            "gameplay_mtrand_sync_receipt_invalid"
        ) from None
    if not isinstance(payload, Mapping):
        raise CollectionError("gameplay_mtrand_sync_receipt_invalid")
    receipt = payload.get("receipt")
    observations = payload.get("observations")
    trace_result = payload.get("pre_blackout_trace_result")
    source_dmo = payload.get("source_dmo")
    if (
        payload.get("schema")
        != "zuma.popcap_gameplay_mtrand_sync_receipt.v1"
        or payload.get("status") != "PASS"
        or payload.get("classification")
        != GAMEPLAY_MTRAND_SYNC_CLASSIFICATION
        or payload.get("runtime_process_id") != expected_process_id
        or payload.get("persistent_file_modified") is not False
        or payload.get("process_context_mutation") is not True
        or not isinstance(receipt, Mapping)
        or not isinstance(observations, list)
        or not isinstance(trace_result, Mapping)
        or not isinstance(source_dmo, Mapping)
    ):
        raise CollectionError(
            "gameplay_mtrand_sync_receipt_contract_failed"
        )
    oracle = receipt.get("oracle")
    expected_hits = receipt.get("expected_hit_count")
    if (
        not isinstance(oracle, Mapping)
        or receipt.get("status") != "PASS"
        or receipt.get("failure") is not None
        or receipt.get("oracle_complete") is not True
        or receipt.get("hardware_breakpoint_armed") is not False
        or receipt.get("hardware_breakpoint_restored") is not True
        or receipt.get("hardware_breakpoint_restore_error") is not None
        or isinstance(expected_hits, bool)
        or not isinstance(expected_hits, int)
        or expected_hits <= 0
        or receipt.get("hit_count") != expected_hits
        or receipt.get("register_mutation_count") != expected_hits
        or len(observations) != expected_hits
        or trace_result.get("failure_count") != 0
        or trace_result.get("stopped_at_update") is not True
    ):
        raise CollectionError(
            "gameplay_mtrand_sync_receipt_contract_failed"
        )
    if any(
        not isinstance(row, Mapping) or row.get("order") != order
        for order, row in enumerate(observations)
    ):
        raise CollectionError(
            "gameplay_mtrand_sync_observation_order_invalid"
        )
    oracle_path = oracle.get("source_path")
    oracle_sha256 = oracle.get("source_sha256")
    if (
        not isinstance(oracle_path, str)
        or Path(oracle_path).resolve() != expected_oracle.resolve()
        or oracle_sha256 != _sha256_file(expected_oracle)
        or oracle.get("entry_count") != expected_hits
        or observations[0].get("framework_update")
        != oracle.get("first_update")
        or observations[-1].get("framework_update")
        != oracle.get("last_update")
        or source_dmo.get("sha256") != _sha256_file(expected_dmo)
    ):
        raise CollectionError("gameplay_mtrand_sync_provenance_mismatch")
    correction_count = receipt.get("state_correction_count")
    bytes_written = receipt.get("bytes_written_total")
    if (
        isinstance(correction_count, bool)
        or not isinstance(correction_count, int)
        or not 0 <= correction_count <= expected_hits
        or isinstance(bytes_written, bool)
        or not isinstance(bytes_written, int)
        or bytes_written != correction_count * MTRAND_STATE_BYTES
    ):
        raise CollectionError(
            "gameplay_mtrand_sync_mutation_count_invalid"
        )
    return {
        "schema": "zuma-rl.pc-gameplay-mtrand-sync-artifact",
        "version": 1,
        "status": "PASS",
        "classification": payload.get("classification"),
        "artifact": artifact,
        "artifact_bytes": len(data),
        "artifact_sha256": _sha256_bytes(data),
        "oracle": dict(oracle),
        "receipt": dict(receipt),
        "observation_count": len(observations),
        "pre_blackout_trace": {
            "last_update": trace_result.get("last_update"),
            "stopped_at_update": trace_result.get("stopped_at_update"),
            "failure_count": trace_result.get("failure_count"),
        },
        "persistent_file_modified": False,
    }


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    try:
        text = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError):
        raise CollectionError("json_not_canonicalizable") from None
    return (text + "\n").encode("ascii")


def _font_cache_manifest_receipt(
    changedir: Path,
    demo: PopCapDemo,
    *,
    attach_at_update: int,
) -> dict[str, Any]:
    main_pak = changedir / "main.pak"
    archive = PopCapPakArchive(main_pak)
    manifest: dict[str, int] = {}
    for entry in archive.entries:
        normalized_name = entry.name.replace("/", "\\").casefold()
        member = _normalize_font_cache_member(normalized_name)
        if member is None or member != normalized_name:
            continue
        if member in manifest:
            raise CollectionError("font_cache_manifest_invalid")
        manifest[member] = entry.size
    loading_complete = next(
        (
            command.sequence
            for command in demo.commands
            if command.kind == "loading_complete"
            and command.command_number == 9
        ),
        None,
    )
    if loading_complete is None:
        raise CollectionError("font_cache_manifest_invalid")
    startup_rows = [
        {
            "row_index": command.sequence,
            "update": command.update,
            "command_number": command.command_number,
            "kind": command.kind,
            "short_form": command.short_form,
            "success": command.payload.get("success"),
        }
        for command in demo.commands
        if (
            command.sequence < loading_complete
            and command.kind == "file_write"
            and command.command_number == 16
            and not command.short_form
            and set(command.payload) == {"success"}
            and isinstance(command.payload["success"], bool)
        )
    ]
    members = [
        {"member": member, "size": manifest[member]}
        for member in sorted(manifest)
    ]
    manifest_bytes = json.dumps(
        [(row["member"], row["size"]) for row in members],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
    if (
        len(members) != FONT_CACHE_MANIFEST_EXPECTED_ENTRIES
        or len(startup_rows) != FONT_CACHE_MANIFEST_EXPECTED_ENTRIES
        or any(row["success"] is not False for row in startup_rows)
    ):
        raise CollectionError("font_cache_manifest_invalid")
    pre_attach_rows = [
        row for row in startup_rows if row["update"] < attach_at_update
    ]
    if (
        attach_at_update < 0
        or (attach_at_update == 0 and pre_attach_rows)
        or (attach_at_update > 0 and not pre_attach_rows)
    ):
        raise CollectionError("font_cache_manifest_invalid")
    return {
        "main_pak": str(main_pak.resolve()),
        "main_pak_sha256": _sha256_file(main_pak),
        "manifest_sha256": _sha256_bytes(manifest_bytes),
        "entry_count": len(members),
        "members": members,
        "loading_complete_row_index": loading_complete,
        "startup_file_write_rows": startup_rows,
        "pre_attach_file_write_rows": pre_attach_rows,
    }


def _post_blackout_overdue_idle_reentry_candidates(
    demo: PopCapDemo,
    *,
    reattach_at_update: int,
) -> list[dict[str, Any]]:
    """Freeze exact post-blackout write/idle shapes eligible for audit."""

    if reattach_at_update < 0:
        raise ValueError("reattach update must not be negative")

    def successful_write(command: Any, update: int | None = None) -> bool:
        return (
            command.kind == "file_write"
            and command.command_number == 16
            and command.short_form is False
            and dict(command.payload) == {"success": True}
            and (update is None or command.update == update)
        )

    def exact_idle(command: Any) -> bool:
        return (
            command.kind == "idle"
            and command.command_number == 31
            and command.short_form is False
            and dict(command.payload) == {}
        )

    commands = demo.commands
    candidates: list[dict[str, Any]] = []
    index = 0
    while index + 2 < len(commands):
        first = commands[index]
        if (
            first.update <= reattach_at_update
            or not successful_write(first)
        ):
            index += 1
            continue
        corridor_update = first.update
        corridor_end = index
        while (
            corridor_end + 1 < len(commands)
            and successful_write(
                commands[corridor_end + 1],
                corridor_update,
            )
        ):
            corridor_end += 1
        bridge_index = corridor_end + 1
        following_index = bridge_index + 1
        if following_index >= len(commands):
            break
        bridge = commands[bridge_index]
        following = commands[following_index]
        corridor_row_count = corridor_end - index + 1
        if (
            corridor_row_count
            <= POST_BLACKOUT_FILE_WRITE_SPILLOVER_MAX_ROWS
            and exact_idle(bridge)
            and exact_idle(following)
            and bridge.update - corridor_update
            == SERVICE_EXIT_TIMELINE_MAX_GAP_UPDATES
            and following.update - bridge.update
            == SERVICE_EXIT_TIMELINE_MAX_GAP_UPDATES
        ):
            candidates.append(
                {
                    "corridor_start_index": index,
                    "corridor_end_index": corridor_end,
                    "corridor_row_count": corridor_row_count,
                    "corridor_recorded_update": corridor_update,
                    "bridge_row_index": bridge_index,
                    "bridge_recorded_update": bridge.update,
                    "bridge_kind": bridge.kind,
                    "bridge_command_number": bridge.command_number,
                    "bridge_short_form": bridge.short_form,
                    "following_row_index": following_index,
                    "following_recorded_update": following.update,
                    "following_kind": following.kind,
                    "following_command_number": (
                        following.command_number
                    ),
                    "following_short_form": following.short_form,
                }
            )
        index = corridor_end + 1
    return candidates


def _write_exclusive(path: Path, payload: bytes) -> None:
    if not path.is_absolute() or path.exists() or not path.parent.is_dir():
        raise CollectionError("output_path_invalid")
    part = path.with_name(path.name + ".part")
    if part.exists():
        raise CollectionError("output_path_invalid")
    try:
        with part.open("xb") as stream:
            if stream.write(payload) != len(payload):
                raise CollectionError("output_write_failed")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(part, path)
        part.unlink()
    except CollectionError:
        raise
    except (FileExistsError, OSError):
        raise CollectionError("output_publish_failed") from None


def _copy_exclusive(source: Path, destination: Path) -> None:
    try:
        payload = source.read_bytes()
    except OSError:
        raise CollectionError("source_artifact_unavailable") from None
    _write_exclusive(destination, payload)


def _ascii_absolute(path: Path, name: str) -> Path:
    if not path.is_absolute():
        raise CollectionError(f"{name}_not_absolute")
    try:
        str(path).encode("ascii")
    except UnicodeEncodeError:
        raise CollectionError(f"{name}_not_ascii") from None
    return path


def _new_directory(path: Path) -> None:
    try:
        path.mkdir(parents=False, exist_ok=False)
    except (FileExistsError, OSError):
        raise CollectionError("session_directory_create_failed") from None


def _new_child_directory(path: Path) -> None:
    try:
        path.mkdir(exist_ok=False)
    except (FileExistsError, OSError):
        raise CollectionError("session_layout_create_failed") from None


def _session_member(session_root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative:
        raise CollectionError("plan_artifact_path_invalid")
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or ".." in pure.parts
        or "." in pure.parts
        or "\\" in relative
        or str(pure) != relative
    ):
        raise CollectionError("plan_artifact_path_invalid")
    result = session_root.joinpath(*pure.parts)
    try:
        result.resolve().relative_to(session_root.resolve())
    except (OSError, ValueError):
        raise CollectionError("plan_artifact_path_invalid") from None
    return result


def _game_processes() -> dict[str, tuple[int, ...]]:
    return {
        name: process_ids_by_name(name)
        for name in (DEFAULT_RUNTIME_EXE.name, "ZumasRevenge.exe")
    }


def _require_game_stopped() -> None:
    if any(_game_processes().values()):
        raise CollectionError("game_process_already_running")


def _read_json_object(path: Path, code: str) -> Mapping[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise CollectionError(code) from None
    if not isinstance(value, dict):
        raise CollectionError(code)
    return value


def _read_plan(session_root: Path) -> Mapping[str, Any]:
    plan_path = session_root / "plan.json"
    plan = _read_json_object(plan_path, "plan_invalid")
    expected_keys = {
        "schema",
        "version",
        "session_nonce",
        "project_root",
        "dmo",
        "runtime",
        "prestate",
        "safety",
        "capture",
        "trace",
        "state_comparison",
        "runs",
    }
    if (
        set(plan) != expected_keys
        or plan.get("schema") != PLAN_SCHEMA
        or plan.get("version") != PLAN_VERSION
        or plan.get("runs") != list(RUN_IDS)
        or _canonical_json(plan) != plan_path.read_bytes()
    ):
        raise CollectionError("plan_invalid")
    dmo = plan.get("dmo")
    base_dmo_keys = {
        "artifact",
        "sha256",
        "bytes",
        "product_version",
        "random_seed",
        "length_updates",
    }
    provenance_dmo_keys = {
        "provenance_artifact",
        "provenance_sha256",
        "diagnostic_successful_file_write_padding_rows",
    }
    if (
        not isinstance(dmo, dict)
        or set(dmo) not in (
            base_dmo_keys,
            base_dmo_keys | provenance_dmo_keys,
        )
        or dmo.get("artifact") != "inputs/input.dmo"
        or not isinstance(dmo.get("sha256"), str)
        or not dmo["sha256"].startswith("sha256:")
        or len(dmo["sha256"]) != 71
        or isinstance(dmo.get("bytes"), bool)
        or not isinstance(dmo.get("bytes"), int)
        or dmo["bytes"] <= 0
    ):
        raise CollectionError("plan_invalid")
    if provenance_dmo_keys.issubset(dmo):
        provenance_artifact = dmo["provenance_artifact"]
        provenance_sha256 = dmo["provenance_sha256"]
        padding_rows = dmo[
            "diagnostic_successful_file_write_padding_rows"
        ]
        if (
            not isinstance(padding_rows, list)
            or any(
                isinstance(row_index, bool)
                or not isinstance(row_index, int)
                or row_index < 0
                for row_index in padding_rows
            )
            or padding_rows != sorted(set(padding_rows))
            or (provenance_artifact is None)
            != (provenance_sha256 is None)
            or (provenance_artifact is None) != (not padding_rows)
            or (
                provenance_artifact is not None
                and (
                    provenance_artifact
                    != "inputs/input.dmo.provenance.json"
                    or not isinstance(provenance_sha256, str)
                    or not provenance_sha256.startswith("sha256:")
                    or len(provenance_sha256) != 71
                )
            )
        ):
            raise CollectionError("plan_invalid")
        if provenance_artifact is not None:
            try:
                provenance_bytes, observed_padding_rows = (
                    _validate_dmo_padding_provenance(
                        _session_member(session_root, dmo["artifact"]),
                        _session_member(session_root, provenance_artifact),
                    )
                )
            except CollectionError:
                raise CollectionError("plan_invalid") from None
            if (
                _sha256_bytes(provenance_bytes) != provenance_sha256
                or list(observed_padding_rows) != padding_rows
            ):
                raise CollectionError("plan_invalid")
    nonce = plan.get("session_nonce")
    if (
        not isinstance(nonce, str)
        or len(nonce) != 32
        or any(character not in "0123456789abcdef" for character in nonce)
    ):
        raise CollectionError("plan_invalid")
    prestate = plan.get("prestate")
    base_prestate_keys = {
        "overlays",
        "template_artifact",
        "template_state_root",
    }
    registry_overlay_key = "registry_overlays"
    if (
        not isinstance(prestate, dict)
        or set(prestate)
        not in (
            base_prestate_keys,
            base_prestate_keys | {registry_overlay_key},
        )
    ):
        raise CollectionError("plan_invalid")
    registry_overlays = prestate.get(registry_overlay_key, [])
    if not isinstance(registry_overlays, list) or len(registry_overlays) > 1:
        raise CollectionError("plan_invalid")
    if registry_overlays:
        row = registry_overlays[0]
        if (
            not isinstance(row, dict)
            or isinstance(row.get("screen_mode"), bool)
            or row.get("screen_mode") not in RUNTIME_SCREEN_MODES
            or row != _runtime_screen_mode_overlay(row["screen_mode"])
        ):
            raise CollectionError("plan_invalid")
    runtime = plan.get("runtime")
    if (
        not isinstance(runtime, dict)
        or runtime.get("launch_mode")
        not in {STEAM_LAUNCH_MODE, DIRECT_FIXED_SEED_LAUNCH_MODE}
    ):
        raise CollectionError("plan_invalid")
    crt_seed = runtime.get("crt_rand_seed")
    startup_seed_transport = runtime.get(
        "startup_seed_transport",
        (
            "debugger_register"
            if runtime["launch_mode"] == DIRECT_FIXED_SEED_LAUNCH_MODE
            else None
        ),
    )
    board_address = runtime.get("board_seed_call_address")
    board_seed = runtime.get("board_seed")
    global_rng_seed = runtime.get("global_rng_seed")
    thread_crt_rng_seed = runtime.get("thread_crt_rng_seed")
    if runtime["launch_mode"] == STEAM_LAUNCH_MODE:
        if any(
            value is not None
            for value in (
                crt_seed,
                startup_seed_transport,
                board_address,
                board_seed,
                global_rng_seed,
                thread_crt_rng_seed,
            )
        ):
            raise CollectionError("plan_invalid")
    else:
        if (
            isinstance(crt_seed, bool)
            or not isinstance(crt_seed, int)
            or not 0 <= crt_seed <= 0xFFFFFFFF
            or startup_seed_transport not in STARTUP_SEED_TRANSPORTS
            or (board_address is None) != (board_seed is None)
        ):
            raise CollectionError("plan_invalid")
        if board_seed is not None and (
            isinstance(board_address, bool)
            or not isinstance(board_address, int)
            or not 0 < board_address <= 0xFFFFFFFF
            or isinstance(board_seed, bool)
            or not isinstance(board_seed, int)
            or not 0 <= board_seed <= 0xFFFFFFFF
        ):
            raise CollectionError("plan_invalid")
        if global_rng_seed is not None and (
            board_seed is None
            or isinstance(global_rng_seed, bool)
            or not isinstance(global_rng_seed, int)
            or not 0 <= global_rng_seed <= 0xFFFFFFFF
        ):
            raise CollectionError("plan_invalid")
        if thread_crt_rng_seed is not None and (
            board_seed is None
            or isinstance(thread_crt_rng_seed, bool)
            or not isinstance(thread_crt_rng_seed, int)
            or not 0 <= thread_crt_rng_seed <= 0xFFFFFFFF
        ):
            raise CollectionError("plan_invalid")
    capture = plan.get("capture")
    startup_window_activation = (
        capture.get("startup_window_activation", True)
        if isinstance(capture, dict)
        else None
    )
    inter_run_cooldown_seconds = (
        capture.get("inter_run_cooldown_seconds", 0.0)
        if isinstance(capture, dict)
        else None
    )
    framework_state_diagnostic = (
        capture.get("framework_state_diagnostic", False)
        if isinstance(capture, dict)
        else None
    )
    framework_state_poll_diagnostic = (
        capture.get("framework_state_poll_diagnostic", False)
        if isinstance(capture, dict)
        else None
    )
    capture_wait_update = (
        capture.get("wait_until_framework_update")
        if isinstance(capture, dict)
        else None
    )
    framework_state_poll_affinity_mask = (
        capture.get("framework_state_poll_affinity_mask", 4)
        if isinstance(capture, dict)
        else None
    )
    framework_state_poll_arm_update = (
        capture.get(
            "framework_state_poll_arm_update",
            (
                max(0, capture_wait_update - 5)
                if isinstance(capture_wait_update, int)
                and not isinstance(capture_wait_update, bool)
                else None
            ),
        )
        if isinstance(capture, dict)
        else None
    )
    framework_state_poll_arm_deadline_update = (
        capture.get(
            "framework_state_poll_arm_deadline_update",
            framework_state_poll_arm_update,
        )
        if isinstance(capture, dict)
        else None
    )
    framework_state_poll_target_hz = (
        capture.get("framework_state_poll_target_hz", 5_000)
        if isinstance(capture, dict)
        else None
    )
    if (
        not isinstance(startup_window_activation, bool)
        or not isinstance(framework_state_diagnostic, bool)
        or not isinstance(framework_state_poll_diagnostic, bool)
        or isinstance(capture_wait_update, bool)
        or not isinstance(capture_wait_update, int)
        or capture_wait_update < 0
        or (
            framework_state_poll_diagnostic
            and not framework_state_diagnostic
        )
        or isinstance(framework_state_poll_affinity_mask, bool)
        or not isinstance(framework_state_poll_affinity_mask, int)
        or framework_state_poll_affinity_mask <= 0
        or framework_state_poll_affinity_mask
        > ctypes.c_size_t(-1).value
        or isinstance(framework_state_poll_arm_update, bool)
        or not isinstance(framework_state_poll_arm_update, int)
        or framework_state_poll_arm_update < 0
        or isinstance(framework_state_poll_arm_deadline_update, bool)
        or not isinstance(
            framework_state_poll_arm_deadline_update,
            int,
        )
        or framework_state_poll_arm_deadline_update
        < framework_state_poll_arm_update
        or (
            framework_state_poll_diagnostic
            and framework_state_poll_arm_deadline_update
            >= capture_wait_update
        )
        or isinstance(framework_state_poll_target_hz, bool)
        or not isinstance(framework_state_poll_target_hz, int)
        or not 1_000 <= framework_state_poll_target_hz <= 20_000
        or isinstance(inter_run_cooldown_seconds, bool)
        or not isinstance(inter_run_cooldown_seconds, (int, float))
        or not math.isfinite(float(inter_run_cooldown_seconds))
        or not 0.0
        <= float(inter_run_cooldown_seconds)
        <= MAX_INTER_RUN_COOLDOWN_SECONDS
    ):
        raise CollectionError("plan_invalid")
    trace = plan.get("trace")
    startup_priority_bias = (
        trace.get("startup_priority_bias_until_update")
        if isinstance(trace, dict)
        else None
    )
    startup_process_affinity = (
        trace.get("startup_process_affinity_mask")
        if isinstance(trace, dict)
        else None
    )
    if framework_state_poll_diagnostic and (
        not isinstance(startup_process_affinity, int)
        or isinstance(startup_process_affinity, bool)
        or startup_process_affinity <= 0
        or startup_process_affinity & (startup_process_affinity - 1)
    ):
        raise CollectionError("plan_invalid")
    allow_blackout_rebase = (
        trace.get("allow_blackout_file_write_order_rebase", False)
        if isinstance(trace, dict)
        else False
    )
    allow_post_blackout_stabilization = (
        trace.get("allow_post_blackout_attach_stabilization", False)
        if isinstance(trace, dict)
        else None
    )
    blackout_rebase_rows = (
        trace.get("blackout_file_write_order_rebase_rows", [])
        if isinstance(trace, dict)
        else None
    )
    blackout_rebase_mode = (
        trace.get("blackout_file_write_rebase_mode")
        if isinstance(trace, dict)
        else None
    )
    blackout_expected_command_order_offset = (
        trace.get("blackout_expected_command_order_offset")
        if isinstance(trace, dict)
        else None
    )
    blackout_expected_native_timeline_offset = (
        trace.get("blackout_expected_native_timeline_offset")
        if isinstance(trace, dict)
        else None
    )
    blackout_allowed_offset_pairs = (
        trace.get("blackout_allowed_offset_pairs", [])
        if isinstance(trace, dict)
        else None
    )
    blackout_allowed_pairs_valid = (
        isinstance(blackout_allowed_offset_pairs, list)
        and all(
            isinstance(pair, dict)
            and set(pair)
            == {"command_order_offset", "native_timeline_offset"}
            and not isinstance(pair.get("command_order_offset"), bool)
            and isinstance(pair.get("command_order_offset"), int)
            and pair["command_order_offset"] >= 0
            and not isinstance(pair.get("native_timeline_offset"), bool)
            and isinstance(pair.get("native_timeline_offset"), int)
            and pair["native_timeline_offset"] >= 0
            for pair in blackout_allowed_offset_pairs
        )
    )
    blackout_allowed_pair_values = (
        tuple(
            (
                pair["command_order_offset"],
                pair["native_timeline_offset"],
            )
            for pair in blackout_allowed_offset_pairs
        )
        if blackout_allowed_pairs_valid
        else ()
    )
    allow_post_blackout_overdue_idle_reentry = (
        trace.get(
            "allow_post_blackout_overdue_idle_reentry",
            False,
        )
        if isinstance(trace, dict)
        else None
    )
    post_blackout_overdue_idle_reentry_candidates = (
        trace.get(
            "post_blackout_overdue_idle_reentry_candidates",
            [],
        )
        if isinstance(trace, dict)
        else None
    )
    allow_attach_stabilization = (
        trace.get("allow_attach_stabilization", False)
        if isinstance(trace, dict)
        else None
    )
    startup_trace_handoff = (
        trace.get("startup_trace_handoff", False)
        if isinstance(trace, dict)
        else None
    )
    allow_pre_attach_file_write_debt = (
        trace.get("allow_pre_attach_file_write_debt", False)
        if isinstance(trace, dict)
        else None
    )
    allow_font_cache_manifest_completion_debt = (
        trace.get("allow_font_cache_manifest_completion_debt", False)
        if isinstance(trace, dict)
        else None
    )
    close_after_terminal_command = (
        trace.get("close_after_terminal_command", False)
        if isinstance(trace, dict)
        else None
    )
    terminal_close_command = (
        trace.get("terminal_close_command")
        if isinstance(trace, dict)
        else None
    )
    source_bound_anchor = (
        trace.get("source_bound_board_anchor")
        if isinstance(trace, dict)
        else None
    )
    source_bound_precall = (
        trace.get("source_bound_board_precall_global_restore")
        if isinstance(trace, dict)
        else None
    )
    allow_source_bound_board_global_correction = (
        trace.get(
            "allow_source_bound_board_global_correction",
            False,
        )
        if isinstance(trace, dict)
        else None
    )
    gameplay_mtrand_sync = (
        trace.get("gameplay_mtrand_sync")
        if isinstance(trace, dict)
        else None
    )
    if (
        not isinstance(trace, dict)
        or not isinstance(
            trace.get("allow_pre_stream_commands"),
            bool,
        )
        or not isinstance(
            trace.get("seed_board_before_attach", False),
            bool,
        )
        or not isinstance(
            allow_blackout_rebase,
            bool,
        )
        or not isinstance(allow_post_blackout_stabilization, bool)
        or not isinstance(
            allow_post_blackout_overdue_idle_reentry,
            bool,
        )
        or not isinstance(
            post_blackout_overdue_idle_reentry_candidates,
            list,
        )
        or bool(post_blackout_overdue_idle_reentry_candidates)
        is not allow_post_blackout_overdue_idle_reentry
        or (
            allow_post_blackout_overdue_idle_reentry
            and not allow_post_blackout_stabilization
        )
        or (
            allow_blackout_rebase
            and not allow_post_blackout_stabilization
        )
        or (
            allow_post_blackout_stabilization
            and runtime.get("launch_mode")
            != DIRECT_FIXED_SEED_LAUNCH_MODE
        )
        or not isinstance(allow_attach_stabilization, bool)
        or not isinstance(startup_trace_handoff, bool)
        or not isinstance(allow_pre_attach_file_write_debt, bool)
        or not isinstance(
            allow_font_cache_manifest_completion_debt, bool
        )
        or not isinstance(close_after_terminal_command, bool)
        or source_bound_anchor is not None
        and not isinstance(source_bound_anchor, dict)
        or source_bound_precall is not None
        and not isinstance(source_bound_precall, dict)
        or source_bound_precall is not None
        and source_bound_anchor is None
        or not isinstance(
            allow_source_bound_board_global_correction,
            bool,
        )
        or (
            allow_source_bound_board_global_correction
            and source_bound_anchor is None
        )
        or (
            source_bound_precall is not None
            and allow_source_bound_board_global_correction
        )
        or gameplay_mtrand_sync is not None
        and not isinstance(gameplay_mtrand_sync, dict)
        or bool(terminal_close_command)
        != close_after_terminal_command
        or (
            terminal_close_command is not None
            and (
                not isinstance(terminal_close_command, dict)
                or set(terminal_close_command)
                != {
                    "command_order",
                    "update",
                    "kind",
                    "command_number",
                    "short_form",
                }
                or isinstance(
                    terminal_close_command.get("command_order"), bool
                )
                or not isinstance(
                    terminal_close_command.get("command_order"), int
                )
                or terminal_close_command["command_order"] < 0
                or isinstance(terminal_close_command.get("update"), bool)
                or not isinstance(terminal_close_command.get("update"), int)
                or terminal_close_command["update"] < 0
                or terminal_close_command.get("kind") != "idle"
                or isinstance(
                    terminal_close_command.get("command_number"), bool
                )
                or not isinstance(
                    terminal_close_command.get("command_number"), int
                )
                or not 0
                <= terminal_close_command["command_number"]
                <= 31
                or terminal_close_command.get("short_form") is not False
            )
        )
        or not isinstance(blackout_rebase_rows, list)
        or bool(blackout_rebase_rows) != allow_blackout_rebase
        or not blackout_allowed_pairs_valid
        or blackout_rebase_mode
        not in {
            None,
            BLACKOUT_FILE_WRITE_REBASE_EXACT_MODE,
            BLACKOUT_FILE_WRITE_REBASE_ENVELOPE_MODE,
            BLACKOUT_FILE_WRITE_REBASE_ENVELOPE_SET_MODE,
        }
        or (
            not allow_blackout_rebase
            and (
                blackout_rebase_mode is not None
                or blackout_expected_command_order_offset is not None
                or blackout_expected_native_timeline_offset is not None
                or blackout_allowed_pair_values
            )
        )
        or (
            allow_blackout_rebase
            and (
                blackout_rebase_mode
                in {None, BLACKOUT_FILE_WRITE_REBASE_EXACT_MODE}
                and (
                    blackout_expected_command_order_offset is not None
                    or blackout_expected_native_timeline_offset is not None
                    or blackout_allowed_pair_values
                )
                or blackout_rebase_mode
                == BLACKOUT_FILE_WRITE_REBASE_ENVELOPE_MODE
                and (
                    isinstance(
                        blackout_expected_command_order_offset, bool
                    )
                    or not isinstance(
                        blackout_expected_command_order_offset, int
                    )
                    or blackout_expected_command_order_offset <= 0
                    or isinstance(
                        blackout_expected_native_timeline_offset, bool
                    )
                    or not isinstance(
                        blackout_expected_native_timeline_offset, int
                    )
                    or blackout_expected_native_timeline_offset < 0
                    or blackout_expected_command_order_offset
                    > len(blackout_rebase_rows)
                    or blackout_allowed_pair_values
                )
                or blackout_rebase_mode
                == BLACKOUT_FILE_WRITE_REBASE_ENVELOPE_SET_MODE
                and (
                    blackout_expected_command_order_offset is not None
                    or blackout_expected_native_timeline_offset is not None
                    or len(blackout_allowed_pair_values) < 2
                    or tuple(
                        sorted(set(blackout_allowed_pair_values))
                    )
                    != blackout_allowed_pair_values
                    or not any(
                        command_offset > 0
                        for command_offset, _
                        in blackout_allowed_pair_values
                    )
                    or any(
                        command_offset > len(blackout_rebase_rows)
                        for command_offset, _
                        in blackout_allowed_pair_values
                    )
                )
            )
        )
        or (
            trace["allow_pre_stream_commands"]
            and trace.get("attach_at_update") != 0
        )
        or (
            allow_attach_stabilization
            and (
                runtime.get("launch_mode")
                != DIRECT_FIXED_SEED_LAUNCH_MODE
                or trace.get("attach_at_update", 0) <= 0
                or trace.get("allow_pre_stream_commands", False)
                or trace.get("seed_board_before_attach", False)
            )
        )
        or (
            startup_trace_handoff
            and (
                runtime.get("launch_mode")
                != DIRECT_FIXED_SEED_LAUNCH_MODE
                or startup_seed_transport != "debugger_register"
                or trace.get("attach_at_update") != 0
                or not trace.get("allow_pre_stream_commands", False)
                or trace.get("seed_board_before_attach", False)
                or allow_attach_stabilization
                or allow_pre_attach_file_write_debt
            )
        )
        or (
            allow_pre_attach_file_write_debt
            and (
                runtime.get("launch_mode")
                != DIRECT_FIXED_SEED_LAUNCH_MODE
                or trace.get("attach_at_update", 0) <= 0
                or trace.get("allow_pre_stream_commands", False)
                or trace.get("seed_board_before_attach", False)
            )
        )
        or (
            allow_font_cache_manifest_completion_debt
            and not (
                (
                    allow_pre_attach_file_write_debt
                    and runtime.get("launch_mode")
                    == DIRECT_FIXED_SEED_LAUNCH_MODE
                    and trace.get("attach_at_update", 0) > 0
                    and not trace.get(
                        "allow_pre_stream_commands", False
                    )
                    and not trace.get(
                        "seed_board_before_attach", False
                    )
                )
                or (
                    startup_trace_handoff
                    and not allow_pre_attach_file_write_debt
                    and runtime.get("launch_mode")
                    == DIRECT_FIXED_SEED_LAUNCH_MODE
                    and trace.get("attach_at_update") == 0
                    and trace.get("allow_pre_stream_commands", False)
                    and not trace.get(
                        "seed_board_before_attach", False
                    )
                )
            )
        )
        or (
            trace.get("seed_board_before_attach", False)
            and (
                runtime.get("board_seed") is None
                or trace.get("attach_at_update", 0) <= 0
            )
        )
        or (
            startup_priority_bias is not None
            and (
                isinstance(startup_priority_bias, bool)
                or not isinstance(startup_priority_bias, int)
                or startup_priority_bias <= 0
                or runtime.get("launch_mode")
                != DIRECT_FIXED_SEED_LAUNCH_MODE
                or trace.get("seed_board_before_attach", False)
                or (
                    trace.get("attach_at_update", 0) > 0
                    and startup_priority_bias
                    >= trace.get("attach_at_update", 0)
                )
            )
        )
        or (
            startup_process_affinity is not None
            and (
                isinstance(startup_process_affinity, bool)
                or not isinstance(startup_process_affinity, int)
                or startup_process_affinity <= 0
                or startup_process_affinity
                > ctypes.c_size_t(-1).value
                or runtime.get("launch_mode")
                != DIRECT_FIXED_SEED_LAUNCH_MODE
                or startup_trace_handoff is not True
                or trace.get("seed_board_before_attach", False)
            )
        )
    ):
        raise CollectionError("plan_invalid")
    if source_bound_anchor is not None:
        source_bound_keys = {
            "monitor_path",
            "monitor_sha256",
            "trace_path",
            "trace_sha256",
            "recording_report_path",
            "recording_report_sha256",
            "source_dmo_path",
            "source_dmo_sha256",
            "monitor_framework_update",
            "global_seed",
            "rewind_draws",
            "source_order",
            "framework_update",
            "caller",
            "expected_output",
            "expected_post_index",
            "expected_post_state_sha256",
            "thread_crt_state",
        }
        path_keys = (
            "monitor_path",
            "trace_path",
            "recording_report_path",
            "source_dmo_path",
        )
        digest_keys = (
            "monitor_sha256",
            "trace_sha256",
            "recording_report_sha256",
            "source_dmo_sha256",
            "expected_post_state_sha256",
        )
        integer_keys = (
            "monitor_framework_update",
            "global_seed",
            "rewind_draws",
            "source_order",
            "framework_update",
            "caller",
            "expected_output",
            "expected_post_index",
            "thread_crt_state",
        )
        if (
            set(source_bound_anchor) != source_bound_keys
            or any(
                not isinstance(source_bound_anchor.get(key), str)
                or not source_bound_anchor[key]
                for key in path_keys
            )
            or any(
                not isinstance(source_bound_anchor.get(key), str)
                or not source_bound_anchor[key].startswith("sha256:")
                or len(source_bound_anchor[key]) != 71
                for key in digest_keys
            )
            or any(
                isinstance(source_bound_anchor.get(key), bool)
                or not isinstance(source_bound_anchor.get(key), int)
                or source_bound_anchor[key] < 0
                for key in integer_keys
            )
        ):
            raise CollectionError("plan_invalid")
    if source_bound_precall is not None:
        assert isinstance(source_bound_anchor, dict)
        try:
            rebuilt_global_state, _, rebuilt_anchor = (
                _load_source_bound_board_anchor(
                    monitor_path=Path(source_bound_anchor["monitor_path"]),
                    trace_path=Path(source_bound_anchor["trace_path"]),
                    recording_report_path=Path(
                        source_bound_anchor["recording_report_path"]
                    ),
                    monitor_framework_update=(
                        source_bound_anchor["monitor_framework_update"]
                    ),
                    global_seed=source_bound_anchor["global_seed"],
                    rewind_draws=source_bound_anchor["rewind_draws"],
                    source_order=source_bound_anchor["source_order"],
                    framework_update=source_bound_anchor[
                        "framework_update"
                    ],
                    caller=source_bound_anchor["caller"],
                )
            )
            rebuilt_precall = _source_bound_board_precall_plan(
                rebuilt_global_state,
                framework_update=source_bound_anchor["framework_update"],
                caller=source_bound_anchor["caller"],
            )
        except (OSError, ValueError, KeyError, TypeError, CollectionError):
            raise CollectionError("plan_invalid") from None
        if (
            rebuilt_anchor != source_bound_anchor
            or rebuilt_precall != source_bound_precall
            or source_bound_precall["framework_update"]
            != source_bound_anchor["framework_update"]
            or source_bound_precall["framework_update"]
            >= trace.get("detach_at_update", -1)
            or _sha256_file(Path(source_bound_anchor["source_dmo_path"]))
            != source_bound_anchor["source_dmo_sha256"]
        ):
            raise CollectionError("plan_invalid")
    if gameplay_mtrand_sync is not None:
        sync_keys = {
            "artifact",
            "sha256",
            "seed",
            "maximum_draws",
            "source_process_id",
            "entry_count",
            "first_update",
            "last_update",
            "semantic_sha256",
            "classification",
        }
        integer_keys = (
            "seed",
            "maximum_draws",
            "source_process_id",
            "entry_count",
            "first_update",
            "last_update",
        )
        if (
            set(gameplay_mtrand_sync) != sync_keys
            or gameplay_mtrand_sync.get("artifact")
            != GAMEPLAY_MTRAND_ORACLE_ARTIFACT
            or gameplay_mtrand_sync.get("classification")
            != GAMEPLAY_MTRAND_SYNC_CLASSIFICATION
            or any(
                isinstance(gameplay_mtrand_sync.get(key), bool)
                or not isinstance(gameplay_mtrand_sync.get(key), int)
                or gameplay_mtrand_sync[key] < 0
                for key in integer_keys
            )
            or gameplay_mtrand_sync["seed"] > 0xFFFFFFFF
            or gameplay_mtrand_sync["maximum_draws"] <= 0
            or gameplay_mtrand_sync["source_process_id"] <= 0
            or gameplay_mtrand_sync["entry_count"] <= 0
            or gameplay_mtrand_sync["first_update"]
            > gameplay_mtrand_sync["last_update"]
            or any(
                not isinstance(gameplay_mtrand_sync.get(key), str)
                or not gameplay_mtrand_sync[key].startswith("sha256:")
                or len(gameplay_mtrand_sync[key]) != 71
                for key in ("sha256", "semantic_sha256")
            )
            or runtime.get("launch_mode")
            != DIRECT_FIXED_SEED_LAUNCH_MODE
            or startup_trace_handoff is not True
            or source_bound_anchor is None
            or source_bound_precall is None
            or gameplay_mtrand_sync["seed"] != dmo.get("random_seed")
            or gameplay_mtrand_sync["first_update"]
            <= source_bound_anchor["framework_update"]
            or gameplay_mtrand_sync["last_update"]
            >= trace.get("detach_at_update", -1)
        ):
            raise CollectionError("plan_invalid")
        try:
            oracle_path = _session_member(
                session_root,
                gameplay_mtrand_sync["artifact"],
            )
            rebuilt_sync = _gameplay_mtrand_sync_plan(
                oracle_path,
                seed=gameplay_mtrand_sync["seed"],
                maximum_draws=gameplay_mtrand_sync["maximum_draws"],
                artifact=gameplay_mtrand_sync["artifact"],
            )
        except CollectionError:
            raise CollectionError("plan_invalid") from None
        if rebuilt_sync != gameplay_mtrand_sync:
            raise CollectionError("plan_invalid")
    if not startup_window_activation:
        if source_bound_anchor is None:
            raise CollectionError("plan_invalid")
        try:
            startup_demo = PopCapDemo.read(
                _session_member(session_root, plan["dmo"]["artifact"])
            )
        except (OSError, ValueError, KeyError, TypeError):
            raise CollectionError("plan_invalid") from None
        activations = [
            command
            for command in startup_demo.commands
            if command.kind == "activate_app"
            and command.update == 0
            and command.payload == {"active": True}
        ]
        if len(activations) != 1:
            raise CollectionError("plan_invalid")
        try:
            _, _, rebuilt_anchor = _load_source_bound_board_anchor(
                monitor_path=Path(source_bound_anchor["monitor_path"]),
                trace_path=Path(source_bound_anchor["trace_path"]),
                recording_report_path=Path(
                    source_bound_anchor["recording_report_path"]
                ),
                monitor_framework_update=(
                    source_bound_anchor["monitor_framework_update"]
                ),
                global_seed=source_bound_anchor["global_seed"],
                rewind_draws=source_bound_anchor["rewind_draws"],
                source_order=source_bound_anchor["source_order"],
                framework_update=source_bound_anchor["framework_update"],
                caller=source_bound_anchor["caller"],
            )
        except CollectionError:
            raise CollectionError("plan_invalid") from None
        if (
            rebuilt_anchor != source_bound_anchor
            or _sha256_file(Path(source_bound_anchor["source_dmo_path"]))
            != source_bound_anchor["source_dmo_sha256"]
            or runtime.get("launch_mode")
            != DIRECT_FIXED_SEED_LAUNCH_MODE
            or runtime.get("board_seed_call_address")
            != DEFAULT_BOARD_RESEED_CALL
            or runtime.get("board_seed")
            != source_bound_anchor["expected_output"]
            or runtime.get("global_rng_seed") is not None
            or runtime.get("thread_crt_rng_seed") is not None
            or trace.get("attach_at_update") != 0
            or trace.get("allow_pre_stream_commands") is not True
            or startup_trace_handoff is not True
            or trace.get("seed_board_before_attach", False)
            or source_bound_anchor["framework_update"]
            >= trace.get("detach_at_update", -1)
        ):
            raise CollectionError("plan_invalid")
    previous_rebase_row = -1
    for row in blackout_rebase_rows:
        if (
            not isinstance(row, dict)
            or set(row)
            != {
                "row_index",
                "update",
                "command_number",
                "kind",
                "short_form",
                "success",
            }
            or isinstance(row.get("row_index"), bool)
            or not isinstance(row.get("row_index"), int)
            or row["row_index"] <= previous_rebase_row
            or isinstance(row.get("update"), bool)
            or not isinstance(row.get("update"), int)
            or not (
                trace["detach_at_update"]
                < row["update"]
                <= trace["reattach_at_update"]
            )
            or row.get("command_number") != 16
            or row.get("kind") != "file_write"
            or row.get("short_form") is not False
            or not isinstance(row.get("success"), bool)
        ):
            raise CollectionError("plan_invalid")
        previous_rebase_row = row["row_index"]
    if allow_post_blackout_overdue_idle_reentry:
        try:
            overdue_demo = PopCapDemo.read(
                _session_member(session_root, plan["dmo"]["artifact"])
            )
        except (OSError, ValueError, KeyError, TypeError):
            raise CollectionError("plan_invalid") from None
        if post_blackout_overdue_idle_reentry_candidates != (
            _post_blackout_overdue_idle_reentry_candidates(
                overdue_demo,
                reattach_at_update=trace["reattach_at_update"],
            )
        ):
            raise CollectionError("plan_invalid")
    if close_after_terminal_command:
        try:
            planned_demo = PopCapDemo.read(
                _session_member(session_root, plan["dmo"]["artifact"])
            )
        except (OSError, ValueError, KeyError, TypeError):
            raise CollectionError("plan_invalid") from None
        if not planned_demo.commands:
            raise CollectionError("plan_invalid")
        terminal = planned_demo.commands[-1]
        assert isinstance(terminal_close_command, dict)
        if terminal_close_command != {
            "command_order": terminal.sequence,
            "update": terminal.update,
            "kind": terminal.kind,
            "command_number": terminal.command_number,
            "short_form": terminal.short_form,
        }:
            raise CollectionError("plan_invalid")
    return plan


def _validate_prestate_provenance(
    source_dir: Path,
    name: str,
) -> tuple[bytes, bytes]:
    source = source_dir / name
    provenance_path = source_dir / f"{name}.provenance.json"
    try:
        payload = source.read_bytes()
        provenance_bytes = provenance_path.read_bytes()
        provenance = json.loads(provenance_bytes.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise CollectionError("prestate_provenance_invalid") from None
    output = provenance.get("output") if isinstance(provenance, dict) else None
    if (
        provenance.get("schema") != "zuma-rl.dmo-file-read-extraction"
        or provenance.get("version") != 1
        or provenance.get("classification")
        != "reconstructed-local-prestate-from-dmo-payload"
        or not isinstance(output, dict)
        or output.get("bytes") != len(payload)
        or output.get("sha256") != hashlib.sha256(payload).hexdigest()
    ):
        raise CollectionError("prestate_provenance_invalid")
    return payload, provenance_bytes


def _validate_dmo_padding_provenance(
    dmo_path: Path,
    provenance_path: Path,
) -> tuple[bytes, tuple[int, ...]]:
    try:
        dmo_data = dmo_path.read_bytes()
        provenance_bytes = provenance_path.read_bytes()
        provenance = json.loads(provenance_bytes.decode("utf-8"))
        if not isinstance(provenance, dict):
            raise ValueError("provenance root is not an object")
        padding_rows = validate_successful_file_write_padding_provenance(
            dmo_data,
            provenance,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, TypeError):
        raise CollectionError("dmo_provenance_invalid") from None
    return provenance_bytes, padding_rows


def _validated_capture_settings(
    duration_value: Any,
    frame_budget_value: Any,
) -> tuple[float, int]:
    try:
        duration_seconds = parse_duration(duration_value)
        frame_budget_fps = parse_frame_budget_fps(frame_budget_value)
        estimated_raw_bytes(
            (0, 0, 800, 600),
            duration_seconds,
            frame_budget_fps,
        )
    except CaptureError as error:
        raise CollectionError(error.code) from None
    return duration_seconds, frame_budget_fps


def prepare_session(args: argparse.Namespace) -> Mapping[str, Any]:
    _require_windows()
    duration_seconds, frame_budget_fps = _validated_capture_settings(
        args.duration_seconds,
        args.frame_budget_fps,
    )
    inter_run_cooldown_seconds = float(
        args.inter_run_cooldown_seconds
    )
    session_root = _ascii_absolute(args.session_root, "session_root")
    dmo_source = args.dmo.resolve()
    dmo_provenance_source = (
        None
        if getattr(args, "dmo_provenance", None) is None
        else args.dmo_provenance.resolve()
    )
    prestate_source = args.prestate_dir.resolve()
    runtime_source = args.runtime_source_executable.resolve()
    direct_runtime = (
        None
        if args.direct_runtime_executable is None
        else args.direct_runtime_executable.resolve()
    )
    gameplay_mtrand_oracle_source = (
        None
        if getattr(args, "gameplay_mtrand_oracle", None) is None
        else args.gameplay_mtrand_oracle.resolve()
    )
    gameplay_mtrand_oracle_seed = getattr(
        args,
        "gameplay_mtrand_oracle_seed",
        None,
    )
    gameplay_mtrand_oracle_maximum_draws = getattr(
        args,
        "gameplay_mtrand_oracle_maximum_draws",
        100_000,
    )
    gameplay_mtrand_requested = (
        gameplay_mtrand_oracle_source is not None
        or gameplay_mtrand_oracle_seed is not None
    )
    if gameplay_mtrand_requested and (
        gameplay_mtrand_oracle_source is None
        or gameplay_mtrand_oracle_seed is None
        or gameplay_mtrand_oracle_maximum_draws <= 0
    ):
        raise CollectionError("gameplay_mtrand_oracle_invalid")
    source_bound_arg_names = (
        "source_bound_board_monitor",
        "source_bound_board_trace",
        "source_bound_board_recording_report",
        "source_bound_board_monitor_update",
        "source_bound_board_seed",
        "source_bound_board_rewind_draws",
        "source_bound_board_source_order",
        "source_bound_board_framework_update",
        "source_bound_board_caller",
    )
    source_bound_values = tuple(
        getattr(args, name, None) for name in source_bound_arg_names
    )
    source_bound_requested = any(
        value is not None for value in source_bound_values
    )
    if source_bound_requested and any(
        value is None for value in source_bound_values
    ):
        raise CollectionError("source_bound_anchor_invalid")
    source_bound_precall_requested = getattr(
        args,
        "source_bound_board_precall_global_restore",
        False,
    )
    if (
        source_bound_precall_requested
        and not source_bound_requested
    ):
        raise CollectionError("source_bound_board_precall_invalid")
    for path in (dmo_source, runtime_source):
        if not path.is_file():
            raise CollectionError("source_artifact_unavailable")
    if (
        dmo_provenance_source is not None
        and not dmo_provenance_source.is_file()
    ):
        raise CollectionError("source_artifact_unavailable")
    if (
        gameplay_mtrand_oracle_source is not None
        and not gameplay_mtrand_oracle_source.is_file()
    ):
        raise CollectionError("source_artifact_unavailable")
    if direct_runtime is None and (
        args.crt_rand_seed is not None
        or args.startup_seed_transport is not None
        or args.board_seed is not None
        or args.global_rng_seed is not None
        or args.thread_crt_rng_seed is not None
        or source_bound_requested
        or gameplay_mtrand_requested
    ):
        raise CollectionError("fixed_seed_launch_invalid")
    if direct_runtime is not None and not direct_runtime.is_file():
        raise CollectionError("source_artifact_unavailable")
    if not prestate_source.is_dir():
        raise CollectionError("prestate_directory_unavailable")
    _require_game_stopped()
    if not DEFAULT_STEAM_EXE.is_file():
        raise CollectionError("steam_executable_unavailable")
    if shutil.disk_usage(session_root.parent).free < MIN_FREE_BYTES:
        raise CollectionError("insufficient_free_space")

    demo = PopCapDemo.read(dmo_source)
    gameplay_mtrand_oracle: GameplayMTRandOracle | None = None
    if gameplay_mtrand_requested:
        assert gameplay_mtrand_oracle_source is not None
        assert gameplay_mtrand_oracle_seed is not None
        if gameplay_mtrand_oracle_seed != demo.random_seed:
            raise CollectionError("gameplay_mtrand_oracle_invalid")
        gameplay_mtrand_oracle = _load_gameplay_mtrand_oracle_for_plan(
            gameplay_mtrand_oracle_source,
            seed=gameplay_mtrand_oracle_seed,
            maximum_draws=gameplay_mtrand_oracle_maximum_draws,
        )
    dmo_provenance_bytes: bytes | None = None
    diagnostic_padding_rows: tuple[int, ...] = ()
    if dmo_provenance_source is not None:
        dmo_provenance_bytes, diagnostic_padding_rows = (
            _validate_dmo_padding_provenance(
                dmo_source,
                dmo_provenance_source,
            )
        )
    if demo.length_updates <= args.wait_until_framework_update:
        raise CollectionError("capture_update_outside_dmo")
    startup_window_activation = not getattr(
        args,
        "skip_startup_window_activation",
        False,
    )
    recorded_startup_activations = [
        command
        for command in demo.commands
        if command.kind == "activate_app"
        and command.update == 0
        and command.payload == {"active": True}
    ]
    if (
        not startup_window_activation
        and len(recorded_startup_activations) != 1
    ):
        raise CollectionError("startup_window_activation_invalid")
    close_after_terminal_command = getattr(
        args,
        "close_after_terminal_command",
        False,
    )
    terminal_close_command: dict[str, Any] | None = None
    if close_after_terminal_command:
        if not demo.commands:
            raise CollectionError("terminal_close_command_invalid")
        terminal = demo.commands[-1]
        if (
            terminal.kind != "idle"
            or terminal.short_form
            or terminal.payload
        ):
            raise CollectionError("terminal_close_command_invalid")
        terminal_close_command = {
            "command_order": terminal.sequence,
            "update": terminal.update,
            "kind": terminal.kind,
            "command_number": terminal.command_number,
            "short_form": terminal.short_form,
        }
    effectful_command_updates = sorted(
        {
            command.update
            for command in demo.commands
            if command.kind != "idle"
        }
    )
    previous_commands = [
        update
        for update in effectful_command_updates
        if update < args.window_repaint_update
    ]
    next_commands = [
        update
        for update in effectful_command_updates
        if update >= args.window_repaint_update
    ]
    if not previous_commands or not next_commands:
        raise CollectionError("window_repaint_update_outside_dmo")
    repaint_previous_command = previous_commands[-1]
    repaint_next_command = next_commands[0]
    guarded_idle_count = sum(
        command.kind == "idle"
        and repaint_previous_command
        < command.update
        < repaint_next_command
        for command in demo.commands
    )
    if not (
        repaint_previous_command
        < args.window_repaint_update
        <= args.wait_until_framework_update
        < repaint_next_command
    ):
        raise CollectionError("window_repaint_guard_invalid")
    if not (
        0
        <= args.attach_at_update
        < args.detach_at_update
        < args.window_repaint_update
        <= args.wait_until_framework_update
        < args.reattach_at_update
        < demo.length_updates
    ):
        raise CollectionError("trace_capture_blackout_invalid")
    if (
        args.allow_pre_stream_commands
        and args.attach_at_update != 0
    ):
        raise CollectionError("trace_capture_blackout_invalid")
    seed_board_before_attach = getattr(
        args,
        "seed_board_before_attach",
        False,
    )
    allow_attach_stabilization = getattr(
        args,
        "allow_attach_stabilization",
        False,
    )
    allow_pre_attach_file_write_debt = getattr(
        args,
        "allow_pre_attach_file_write_debt",
        False,
    )
    allow_font_cache_manifest_completion_debt = getattr(
        args,
        "allow_font_cache_manifest_completion_debt",
        False,
    )
    allow_post_blackout_overdue_idle_reentry = getattr(
        args,
        "allow_post_blackout_overdue_idle_reentry",
        False,
    )
    allow_blackout_file_write_order_rebase = getattr(
        args,
        "allow_blackout_file_write_order_rebase",
        False,
    )
    blackout_expected_command_order_offset = getattr(
        args,
        "blackout_expected_command_order_offset",
        None,
    )
    blackout_expected_native_timeline_offset = getattr(
        args,
        "blackout_expected_native_timeline_offset",
        None,
    )
    blackout_envelope_values = (
        blackout_expected_command_order_offset,
        blackout_expected_native_timeline_offset,
    )
    blackout_allowed_offset_pairs = tuple(
        getattr(args, "blackout_allowed_offset_pairs", [])
    )
    if any(value is not None for value in blackout_envelope_values) and (
        any(value is None for value in blackout_envelope_values)
        or not allow_blackout_file_write_order_rebase
    ):
        raise CollectionError("trace_capture_blackout_invalid")
    if blackout_allowed_offset_pairs and (
        any(value is not None for value in blackout_envelope_values)
        or not allow_blackout_file_write_order_rebase
        or len(blackout_allowed_offset_pairs) < 2
        or tuple(sorted(set(blackout_allowed_offset_pairs)))
        != blackout_allowed_offset_pairs
        or not any(
            command_offset > 0
            for command_offset, _ in blackout_allowed_offset_pairs
        )
    ):
        raise CollectionError("trace_capture_blackout_invalid")
    post_blackout_overdue_idle_reentry_candidates = (
        _post_blackout_overdue_idle_reentry_candidates(
            demo,
            reattach_at_update=args.reattach_at_update,
        )
        if allow_post_blackout_overdue_idle_reentry
        else []
    )
    if (
        allow_post_blackout_overdue_idle_reentry
        and not post_blackout_overdue_idle_reentry_candidates
    ):
        raise CollectionError("post_blackout_overdue_idle_reentry_invalid")
    allow_post_blackout_stabilization = (
        getattr(
            args,
            "allow_post_blackout_attach_stabilization",
            False,
        )
        or allow_blackout_file_write_order_rebase
        or allow_post_blackout_overdue_idle_reentry
    )
    startup_trace_handoff = getattr(
        args,
        "startup_trace_handoff",
        False,
    )
    startup_priority_bias = getattr(
        args,
        "startup_priority_bias_until_update",
        None,
    )
    startup_process_affinity = getattr(
        args,
        "startup_process_affinity_mask",
        None,
    )
    allow_source_bound_board_global_correction = getattr(
        args,
        "allow_source_bound_board_global_correction",
        False,
    )
    if (
        allow_source_bound_board_global_correction
        and not source_bound_requested
    ):
        raise CollectionError("source_bound_anchor_invalid")
    if (
        source_bound_precall_requested
        and allow_source_bound_board_global_correction
    ):
        raise CollectionError("source_bound_board_precall_invalid")
    if seed_board_before_attach and (
        args.board_seed is None
        or args.attach_at_update <= 0
        or direct_runtime is None
    ):
        raise CollectionError("trace_capture_blackout_invalid")
    if allow_attach_stabilization and (
        direct_runtime is None
        or args.attach_at_update <= 0
        or args.allow_pre_stream_commands
        or seed_board_before_attach
    ):
        raise CollectionError("trace_capture_blackout_invalid")
    if allow_post_blackout_stabilization and direct_runtime is None:
        raise CollectionError("trace_capture_blackout_invalid")
    if allow_pre_attach_file_write_debt and (
        direct_runtime is None
        or args.attach_at_update <= 0
        or args.allow_pre_stream_commands
        or seed_board_before_attach
    ):
        raise CollectionError("trace_capture_blackout_invalid")
    if startup_trace_handoff and (
        direct_runtime is None
        or args.startup_seed_transport not in (None, "debugger_register")
        or args.attach_at_update != 0
        or not args.allow_pre_stream_commands
        or seed_board_before_attach
        or allow_attach_stabilization
        or allow_pre_attach_file_write_debt
    ):
        raise CollectionError("trace_capture_blackout_invalid")
    if allow_font_cache_manifest_completion_debt:
        legacy_manifest_mode = (
            allow_pre_attach_file_write_debt
            and direct_runtime is not None
            and args.attach_at_update > 0
            and not args.allow_pre_stream_commands
            and not seed_board_before_attach
        )
        startup_handoff_manifest_mode = (
            startup_trace_handoff
            and not allow_pre_attach_file_write_debt
            and direct_runtime is not None
            and args.attach_at_update == 0
            and args.allow_pre_stream_commands
            and not seed_board_before_attach
        )
        if not (legacy_manifest_mode or startup_handoff_manifest_mode):
            raise CollectionError("trace_capture_blackout_invalid")
    if startup_priority_bias is not None and (
        direct_runtime is None
        or seed_board_before_attach
        or startup_priority_bias <= 0
        or (
            args.attach_at_update > 0
            and startup_priority_bias >= args.attach_at_update
        )
    ):
        raise CollectionError("trace_capture_blackout_invalid")
    if startup_process_affinity is not None and (
        direct_runtime is None
        or not startup_trace_handoff
        or seed_board_before_attach
        or isinstance(startup_process_affinity, bool)
        or startup_process_affinity <= 0
        or startup_process_affinity > ctypes.c_size_t(-1).value
    ):
        raise CollectionError("trace_capture_blackout_invalid")
    source_bound_anchor: dict[str, Any] | None = None
    source_bound_precall: dict[str, Any] | None = None
    if source_bound_requested:
        if (
            direct_runtime is None
            or args.board_seed is None
            or args.board_seed_address != DEFAULT_BOARD_RESEED_CALL
            or args.global_rng_seed is not None
            or args.thread_crt_rng_seed is not None
            or args.attach_at_update != 0
            or not args.allow_pre_stream_commands
            or not startup_trace_handoff
            or seed_board_before_attach
        ):
            raise CollectionError("source_bound_anchor_invalid")
        (
            source_bound_global_state,
            _,
            source_bound_anchor,
        ) = _load_source_bound_board_anchor(
            monitor_path=args.source_bound_board_monitor,
            trace_path=args.source_bound_board_trace,
            recording_report_path=(
                args.source_bound_board_recording_report
            ),
            monitor_framework_update=(
                args.source_bound_board_monitor_update
            ),
            global_seed=args.source_bound_board_seed,
            rewind_draws=args.source_bound_board_rewind_draws,
            source_order=args.source_bound_board_source_order,
            framework_update=(
                args.source_bound_board_framework_update
            ),
            caller=args.source_bound_board_caller,
        )
        if (
            source_bound_anchor["expected_output"] != args.board_seed
            or source_bound_anchor["framework_update"]
            >= args.detach_at_update
        ):
            raise CollectionError("source_bound_anchor_invalid")
        if source_bound_precall_requested:
            source_bound_precall = _source_bound_board_precall_plan(
                source_bound_global_state,
                framework_update=source_bound_anchor["framework_update"],
                caller=source_bound_anchor["caller"],
            )
            if (
                source_bound_precall["framework_update"]
                != source_bound_anchor["framework_update"]
                or source_bound_precall["framework_update"]
                >= args.detach_at_update
            ):
                raise CollectionError("source_bound_board_precall_invalid")
    if gameplay_mtrand_oracle is not None:
        if (
            direct_runtime is None
            or not startup_trace_handoff
            or source_bound_anchor is None
            or source_bound_precall is None
            or gameplay_mtrand_oracle.entries[0].framework_update
            <= source_bound_anchor["framework_update"]
            or gameplay_mtrand_oracle.entries[-1].framework_update
            >= args.detach_at_update
        ):
            raise CollectionError("gameplay_mtrand_oracle_invalid")
    if not startup_window_activation and source_bound_anchor is None:
        raise CollectionError("startup_window_activation_invalid")
    blackout_rebase_rows = [
        {
            "row_index": command.sequence,
            "update": command.update,
            "command_number": command.command_number,
            "kind": command.kind,
            "short_form": command.short_form,
            "success": command.payload.get("success"),
        }
        for command in demo.commands
        if (
            args.detach_at_update
            < command.update
            <= args.reattach_at_update
            and command.kind == "file_write"
            and command.command_number == 16
            and not command.short_form
            and set(command.payload) == {"success"}
            and isinstance(command.payload["success"], bool)
        )
    ]
    if (
        allow_blackout_file_write_order_rebase
        and not blackout_rebase_rows
    ):
        raise CollectionError("trace_capture_blackout_invalid")
    if not allow_blackout_file_write_order_rebase:
        blackout_rebase_rows = []
    if (
        blackout_expected_command_order_offset is not None
        and blackout_expected_command_order_offset
        > len(blackout_rebase_rows)
    ):
        raise CollectionError("trace_capture_blackout_invalid")
    if any(
        command_offset > len(blackout_rebase_rows)
        for command_offset, _ in blackout_allowed_offset_pairs
    ):
        raise CollectionError("trace_capture_blackout_invalid")
    blackout_file_write_rebase_mode = (
        BLACKOUT_FILE_WRITE_REBASE_ENVELOPE_SET_MODE
        if blackout_allowed_offset_pairs
        else (
            BLACKOUT_FILE_WRITE_REBASE_ENVELOPE_MODE
            if blackout_expected_command_order_offset is not None
            else (
                BLACKOUT_FILE_WRITE_REBASE_EXACT_MODE
                if allow_blackout_file_write_order_rebase
                else None
            )
        )
    )
    runtime_source_sha256 = _sha256_file(runtime_source)
    launch_mode = (
        STEAM_LAUNCH_MODE
        if direct_runtime is None
        else DIRECT_FIXED_SEED_LAUNCH_MODE
    )
    runtime_executable = (
        DEFAULT_RUNTIME_EXE
        if direct_runtime is None
        else direct_runtime
    )
    runtime_executable_sha256 = (
        EXPECTED_RUNTIME_SHA256
        if direct_runtime is None
        else _sha256_file(direct_runtime)
    )
    if runtime_executable_sha256 != EXPECTED_RUNTIME_SHA256:
        raise CollectionError("runtime_executable_identity_mismatch")
    changedir = (
        None
        if direct_runtime is None
        else (
            runtime_source.parent.resolve()
            if args.changedir is None
            else args.changedir.resolve()
        )
    )
    if changedir is not None and not changedir.is_dir():
        raise CollectionError("runtime_asset_directory_unavailable")
    font_cache_manifest_receipt = (
        _font_cache_manifest_receipt(
            changedir,
            demo,
            attach_at_update=args.attach_at_update,
        )
        if allow_font_cache_manifest_completion_debt
        and changedir is not None
        else None
    )
    crt_rand_seed = (
        None
        if direct_runtime is None
        else (
            demo.random_seed
            if args.crt_rand_seed is None
            else args.crt_rand_seed
        )
    )
    if crt_rand_seed is not None and crt_rand_seed > 0xFFFFFFFF:
        raise CollectionError("fixed_seed_launch_invalid")
    startup_seed_transport = (
        None
        if direct_runtime is None
        else (args.startup_seed_transport or "debugger_register")
    )
    board_seed = (
        None if direct_runtime is None else args.board_seed
    )
    global_rng_seed = (
        None if direct_runtime is None else args.global_rng_seed
    )
    thread_crt_rng_seed = (
        None
        if direct_runtime is None
        else args.thread_crt_rng_seed
    )
    if (
        board_seed is not None
        and (
            board_seed > 0xFFFFFFFF
            or args.board_seed_address > 0xFFFFFFFF
        )
    ):
        raise CollectionError("fixed_seed_launch_invalid")
    if global_rng_seed is not None and (
        board_seed is None or global_rng_seed > 0xFFFFFFFF
    ):
        raise CollectionError("fixed_seed_launch_invalid")
    if thread_crt_rng_seed is not None and (
        board_seed is None or thread_crt_rng_seed > 0xFFFFFFFF
    ):
        raise CollectionError("fixed_seed_launch_invalid")

    _new_directory(session_root)
    for relative in ("inputs", "protocol", "safety"):
        _new_child_directory(session_root / relative)
    _new_child_directory(session_root / "inputs" / "prestate")

    dmo_destination = session_root / "inputs" / "input.dmo"
    _copy_exclusive(dmo_source, dmo_destination)
    dmo_provenance_destination = (
        session_root / "inputs" / "input.dmo.provenance.json"
    )
    if dmo_provenance_bytes is not None:
        _write_exclusive(
            dmo_provenance_destination,
            dmo_provenance_bytes,
        )
    gameplay_mtrand_oracle_destination = (
        session_root / GAMEPLAY_MTRAND_ORACLE_ARTIFACT
    )
    gameplay_mtrand_sync: dict[str, Any] | None = None
    if gameplay_mtrand_oracle_source is not None:
        assert gameplay_mtrand_oracle_seed is not None
        _copy_exclusive(
            gameplay_mtrand_oracle_source,
            gameplay_mtrand_oracle_destination,
        )
        gameplay_mtrand_sync = _gameplay_mtrand_sync_plan(
            gameplay_mtrand_oracle_destination,
            seed=gameplay_mtrand_oracle_seed,
            maximum_draws=gameplay_mtrand_oracle_maximum_draws,
        )

    overlays: list[dict[str, Any]] = []
    replacement_paths: list[tuple[str, Path]] = []
    for name in OVERLAY_NAMES:
        payload, provenance = _validate_prestate_provenance(
            prestate_source,
            name,
        )
        destination = session_root / "inputs" / "prestate" / name
        provenance_destination = destination.with_name(
            destination.name + ".provenance.json"
        )
        _write_exclusive(destination, payload)
        _write_exclusive(provenance_destination, provenance)
        replacements = (name, destination)
        replacement_paths.append(replacements)
        overlays.append(
            {
                "relative_path": name,
                "artifact": f"inputs/prestate/{name}",
                "sha256": _sha256_bytes(payload),
                "bytes": len(payload),
                "provenance_artifact": (
                    f"inputs/prestate/{name}.provenance.json"
                ),
                "provenance_sha256": _sha256_bytes(provenance),
            }
        )

    nonce = args.session_nonce
    host_pre = capture_state(session_nonce=nonce, phase="host-pre")
    host_pre_path = session_root / "safety" / "host-pre.json"
    write_snapshot_exclusive(host_pre, host_pre_path)
    template = overlay_snapshot_files(
        host_pre,
        replacement_paths,
        phase="pre-template",
        captured_perf_counter_ns=time.perf_counter_ns(),
    )
    runtime_screen_mode = getattr(args, "runtime_screen_mode", None)
    registry_overlays: list[dict[str, Any]] = []
    if runtime_screen_mode is not None:
        overlay = _runtime_screen_mode_overlay(runtime_screen_mode)
        try:
            template = overlay_snapshot_registry_values(
                template,
                (
                    (
                        overlay["root"],
                        overlay["subkey"],
                        overlay["value_name"],
                        overlay["value_type"],
                        bytes.fromhex(overlay["data_hex"]),
                    ),
                ),
            )
        except StateTransactionError:
            raise CollectionError(
                "runtime_screen_mode_overlay_invalid"
            ) from None
        registry_overlays.append(overlay)
    template_path = session_root / "protocol" / "pre-template.json"
    write_snapshot_exclusive(template, template_path)

    framework_state_poll_diagnostic = getattr(
        args, "framework_state_poll_diagnostic", False
    )
    framework_state_poll_affinity_mask = getattr(
        args, "framework_state_poll_affinity_mask", 4
    )
    framework_state_poll_arm_update = getattr(
        args, "framework_state_poll_arm_update", None
    )
    if framework_state_poll_arm_update is None:
        framework_state_poll_arm_update = max(
            0, args.wait_until_framework_update - 5
        )
    framework_state_poll_arm_deadline_update = getattr(
        args,
        "framework_state_poll_arm_deadline_update",
        None,
    )
    if framework_state_poll_arm_deadline_update is None:
        framework_state_poll_arm_deadline_update = (
            framework_state_poll_arm_update
        )
    framework_state_poll_target_hz = getattr(
        args, "framework_state_poll_target_hz", 5_000
    )
    if (
        not isinstance(framework_state_poll_diagnostic, bool)
        or (
            framework_state_poll_diagnostic
            and (
                not isinstance(startup_process_affinity, int)
                or isinstance(startup_process_affinity, bool)
                or startup_process_affinity <= 0
                or startup_process_affinity
                & (startup_process_affinity - 1)
            )
        )
        or isinstance(framework_state_poll_affinity_mask, bool)
        or not isinstance(framework_state_poll_affinity_mask, int)
        or framework_state_poll_affinity_mask <= 0
        or framework_state_poll_affinity_mask
        > ctypes.c_size_t(-1).value
        or isinstance(framework_state_poll_arm_update, bool)
        or not isinstance(framework_state_poll_arm_update, int)
        or framework_state_poll_arm_update < 0
        or isinstance(framework_state_poll_arm_deadline_update, bool)
        or not isinstance(
            framework_state_poll_arm_deadline_update,
            int,
        )
        or framework_state_poll_arm_deadline_update
        < framework_state_poll_arm_update
        or (
            framework_state_poll_diagnostic
            and framework_state_poll_arm_deadline_update
            >= args.wait_until_framework_update
        )
        or isinstance(framework_state_poll_target_hz, bool)
        or not isinstance(framework_state_poll_target_hz, int)
        or not 1_000 <= framework_state_poll_target_hz <= 20_000
    ):
        raise CollectionError("framework_state_poll_configuration_invalid")

    capture_plan: dict[str, Any] = {
        "duration_seconds": duration_seconds,
        "frame_budget_fps": frame_budget_fps,
        "inter_run_cooldown_seconds": inter_run_cooldown_seconds,
        "wait_until_framework_update": args.wait_until_framework_update,
        "window_repaint_update": args.window_repaint_update,
        "startup_window_activation": startup_window_activation,
        "window_repaint_guard": {
            "previous_effectful_command_update": repaint_previous_command,
            "next_effectful_command_update": repaint_next_command,
            "allowed_command_kinds": ["idle"],
            "guarded_idle_command_count": guarded_idle_count,
        },
        "device_index": args.device_index,
        "output_index": args.output_index,
        "expected_width": 800,
        "expected_height": 600,
    }
    if (
        getattr(args, "framework_state_diagnostic", False)
        or framework_state_poll_diagnostic
    ):
        capture_plan["framework_state_diagnostic"] = True
    if framework_state_poll_diagnostic:
        capture_plan["framework_state_poll_diagnostic"] = True
        capture_plan["framework_state_poll_affinity_mask"] = (
            framework_state_poll_affinity_mask
        )
        capture_plan["framework_state_poll_arm_update"] = (
            framework_state_poll_arm_update
        )
        capture_plan["framework_state_poll_arm_deadline_update"] = (
            framework_state_poll_arm_deadline_update
        )
        capture_plan["framework_state_poll_target_hz"] = (
            framework_state_poll_target_hz
        )

    plan: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "version": PLAN_VERSION,
        "session_nonce": nonce,
        "project_root": str(Path(__file__).resolve().parents[1]),
        "dmo": {
            "artifact": "inputs/input.dmo",
            "sha256": _sha256_file(dmo_destination),
            "bytes": dmo_destination.stat().st_size,
            "product_version": demo.product_version,
            "random_seed": demo.random_seed,
            "length_updates": demo.length_updates,
            "provenance_artifact": (
                "inputs/input.dmo.provenance.json"
                if dmo_provenance_bytes is not None
                else None
            ),
            "provenance_sha256": (
                _sha256_bytes(dmo_provenance_bytes)
                if dmo_provenance_bytes is not None
                else None
            ),
            "diagnostic_successful_file_write_padding_rows": list(
                diagnostic_padding_rows
            ),
        },
        "runtime": {
            "launch_mode": launch_mode,
            "steam_executable": str(DEFAULT_STEAM_EXE.resolve()),
            "runtime_executable": str(runtime_executable),
            "runtime_executable_sha256": runtime_executable_sha256,
            "runtime_source_executable": str(runtime_source),
            "runtime_source_sha256": runtime_source_sha256,
            "expected_runtime_sha256": EXPECTED_RUNTIME_SHA256,
            "changedir": (
                None if changedir is None else str(changedir)
            ),
            "crt_rand_seed": crt_rand_seed,
            "startup_seed_transport": startup_seed_transport,
            "board_seed_call_address": (
                None
                if board_seed is None
                else args.board_seed_address
            ),
            "board_seed": board_seed,
            "global_rng_seed": global_rng_seed,
            "thread_crt_rng_seed": thread_crt_rng_seed,
        },
        "prestate": {
            "template_artifact": "protocol/pre-template.json",
            "template_state_root": template.state_root,
            "overlays": overlays,
            "registry_overlays": registry_overlays,
        },
        "safety": {
            "host_pre_artifact": "safety/host-pre.json",
            "host_pre_state_root": host_pre.state_root,
        },
        "capture": capture_plan,
        "trace": {
            "attach_at_update": args.attach_at_update,
            "detach_at_update": args.detach_at_update,
            "reattach_at_update": args.reattach_at_update,
            "maximum_hits": args.maximum_hits,
            "trace_timeout_seconds": args.trace_timeout_seconds,
            "close_after_terminal_command": (
                close_after_terminal_command
            ),
            "terminal_close_command": terminal_close_command,
            "launch_timeout_seconds": 30.0,
            "allow_pre_stream_commands": (
                args.allow_pre_stream_commands
            ),
            "startup_trace_handoff": startup_trace_handoff,
            "allow_attach_stabilization": (
                allow_attach_stabilization
            ),
            "allow_pre_attach_file_write_debt": (
                allow_pre_attach_file_write_debt
            ),
            "allow_font_cache_manifest_completion_debt": (
                allow_font_cache_manifest_completion_debt
            ),
            "font_cache_manifest_receipt": (
                font_cache_manifest_receipt
            ),
            "seed_board_before_attach": (
                seed_board_before_attach
            ),
            "startup_priority_bias_until_update": (
                startup_priority_bias
            ),
            "startup_process_affinity_mask": (
                startup_process_affinity
            ),
            "source_bound_board_anchor": source_bound_anchor,
            "source_bound_board_precall_global_restore": (
                source_bound_precall
            ),
            "gameplay_mtrand_sync": gameplay_mtrand_sync,
            "allow_source_bound_board_global_correction": (
                allow_source_bound_board_global_correction
            ),
            "allow_blackout_file_write_order_rebase": (
                allow_blackout_file_write_order_rebase
            ),
            "blackout_file_write_rebase_mode": (
                blackout_file_write_rebase_mode
            ),
            "blackout_expected_command_order_offset": (
                blackout_expected_command_order_offset
            ),
            "blackout_expected_native_timeline_offset": (
                blackout_expected_native_timeline_offset
            ),
            "blackout_allowed_offset_pairs": [
                {
                    "command_order_offset": command_offset,
                    "native_timeline_offset": timeline_offset,
                }
                for command_offset, timeline_offset
                in blackout_allowed_offset_pairs
            ],
            "allow_post_blackout_overdue_idle_reentry": (
                allow_post_blackout_overdue_idle_reentry
            ),
            "post_blackout_overdue_idle_reentry_candidates": (
                post_blackout_overdue_idle_reentry_candidates
            ),
            "allow_post_blackout_attach_stabilization": (
                allow_post_blackout_stabilization
            ),
            "blackout_file_write_order_rebase_rows": (
                blackout_rebase_rows
            ),
        },
        "state_comparison": _state_comparison_contract(launch_mode),
        "runs": list(RUN_IDS),
    }
    _write_exclusive(session_root / "plan.json", _canonical_json(plan))
    return {
        "status": "prepared",
        "session_root": str(session_root),
        "session_nonce": nonce,
        "host_pre_state_root": host_pre.state_root,
        "protocol_prestate_template_root": template.state_root,
        "dmo_sha256": plan["dmo"]["sha256"],
        "free_bytes_after_prepare": shutil.disk_usage(session_root).free,
    }


def _activate_window(window_handle: int) -> None:
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    user32.ShowWindow.argtypes = (wintypes.HWND, ctypes.c_int)
    user32.ShowWindow.restype = wintypes.BOOL
    user32.GetForegroundWindow.argtypes = ()
    user32.GetForegroundWindow.restype = wintypes.HWND
    user32.GetWindowThreadProcessId.argtypes = (
        wintypes.HWND,
        ctypes.POINTER(wintypes.DWORD),
    )
    user32.GetWindowThreadProcessId.restype = wintypes.DWORD
    user32.AttachThreadInput.argtypes = (
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.BOOL,
    )
    user32.AttachThreadInput.restype = wintypes.BOOL
    user32.BringWindowToTop.argtypes = (wintypes.HWND,)
    user32.BringWindowToTop.restype = wintypes.BOOL
    user32.SetForegroundWindow.argtypes = (wintypes.HWND,)
    user32.SetForegroundWindow.restype = wintypes.BOOL
    user32.SetActiveWindow.argtypes = (wintypes.HWND,)
    user32.SetActiveWindow.restype = wintypes.HWND
    user32.SetFocus.argtypes = (wintypes.HWND,)
    user32.SetFocus.restype = wintypes.HWND
    kernel32.GetCurrentThreadId.argtypes = ()
    kernel32.GetCurrentThreadId.restype = wintypes.DWORD

    target = wintypes.HWND(window_handle)
    user32.ShowWindow(target, 9)
    current_thread = int(kernel32.GetCurrentThreadId())
    target_thread = int(
        user32.GetWindowThreadProcessId(target, None)
    )
    foreground = user32.GetForegroundWindow()
    foreground_thread = (
        int(user32.GetWindowThreadProcessId(foreground, None))
        if foreground
        else 0
    )
    attached: list[int] = []
    try:
        for thread_id in (foreground_thread, target_thread):
            if (
                thread_id
                and thread_id != current_thread
                and thread_id not in attached
                and user32.AttachThreadInput(
                    current_thread,
                    thread_id,
                    True,
                )
            ):
                attached.append(thread_id)
        user32.BringWindowToTop(target)
        user32.SetForegroundWindow(target)
        user32.SetActiveWindow(target)
        user32.SetFocus(target)
    finally:
        for thread_id in reversed(attached):
            user32.AttachThreadInput(
                current_thread,
                thread_id,
                False,
            )


def _activate_verified_window(
    target: WindowTarget,
    *,
    activate_window: Any,
    verify_target: Any,
    delay: Any = time.sleep,
    maximum_attempts: int = 4,
) -> int:
    """Activate one exact target and retry only foreground-lock failures."""

    if maximum_attempts <= 0:
        raise ValueError("maximum activation attempts must be positive")
    last_error: CaptureError | None = None
    for attempt in range(1, maximum_attempts + 1):
        activate_window(target.window_handle)
        try:
            verify_target(target)
            return attempt
        except CaptureError as error:
            if not error.code.endswith("not_foreground"):
                raise
            last_error = error
            if attempt < maximum_attempts:
                delay(0.025)
    assert last_error is not None
    raise last_error


def _window_outer_rect(window_handle: int) -> tuple[int, int, int, int]:
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.GetWindowRect.argtypes = (
        wintypes.HWND,
        ctypes.POINTER(wintypes.RECT),
    )
    user32.GetWindowRect.restype = wintypes.BOOL
    rect = wintypes.RECT()
    if not user32.GetWindowRect(
        wintypes.HWND(window_handle),
        ctypes.byref(rect),
    ):
        raise CollectionError("window_repaint_geometry_unavailable")
    result = (
        int(rect.left),
        int(rect.top),
        int(rect.right),
        int(rect.bottom),
    )
    if result[2] <= result[0] or result[3] <= result[1]:
        raise CollectionError("window_repaint_geometry_invalid")
    return result


def _drag_window_titlebar_once(
    target: WindowTarget,
    delta_x: int,
) -> None:
    class MouseInput(ctypes.Structure):
        _fields_ = (
            ("dx", wintypes.LONG),
            ("dy", wintypes.LONG),
            ("mouseData", wintypes.DWORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ctypes.c_size_t),
        )

    class InputUnion(ctypes.Union):
        _fields_ = (("mi", MouseInput),)

    class Input(ctypes.Structure):
        _anonymous_ = ("payload",)
        _fields_ = (
            ("type", wintypes.DWORD),
            ("payload", InputUnion),
        )

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.GetCursorPos.argtypes = (
        ctypes.POINTER(wintypes.POINT),
    )
    user32.GetCursorPos.restype = wintypes.BOOL
    user32.GetSystemMetrics.argtypes = (ctypes.c_int,)
    user32.GetSystemMetrics.restype = ctypes.c_int
    user32.SendInput.argtypes = (
        wintypes.UINT,
        ctypes.POINTER(Input),
        ctypes.c_int,
    )
    user32.SendInput.restype = wintypes.UINT

    virtual_left = user32.GetSystemMetrics(76)
    virtual_top = user32.GetSystemMetrics(77)
    virtual_width = user32.GetSystemMetrics(78)
    virtual_height = user32.GetSystemMetrics(79)
    if virtual_width <= 1 or virtual_height <= 1:
        raise CollectionError("window_repaint_virtual_desktop_invalid")

    def send(flags: int, x: int = 0, y: int = 0) -> None:
        if flags & 0x8000:
            dx = round(
                (x - virtual_left) * 65535 / (virtual_width - 1)
            )
            dy = round(
                (y - virtual_top) * 65535 / (virtual_height - 1)
            )
        else:
            dx = 0
            dy = 0
        item = Input(
            type=0,
            payload=InputUnion(
                mi=MouseInput(
                    dx=dx,
                    dy=dy,
                    mouseData=0,
                    dwFlags=flags,
                    time=0,
                    dwExtraInfo=REPAINT_INPUT_EXTRA_INFO,
                )
            ),
        )
        if user32.SendInput(1, ctypes.byref(item), ctypes.sizeof(Input)) != 1:
            raise CollectionError("window_repaint_send_input_failed")

    rect = _window_outer_rect(target.window_handle)
    title_height = target.client_region[1] - rect[1]
    if title_height < 20:
        raise CollectionError("window_repaint_titlebar_invalid")
    start_x = rect[0] + min(200, (rect[2] - rect[0]) // 2)
    start_y = rect[1] + min(20, title_height - 8)
    original_cursor = wintypes.POINT()
    if not user32.GetCursorPos(ctypes.byref(original_cursor)):
        raise CollectionError("window_repaint_cursor_unavailable")
    button_down = False
    try:
        # MOVE | MOVE_NOCOALESCE | VIRTUALDESK | ABSOLUTE
        absolute_move_flags = 0x0001 | 0x2000 | 0x4000 | 0x8000
        send(absolute_move_flags, start_x, start_y)
        time.sleep(0.02)
        positioned_cursor = wintypes.POINT()
        if (
            not user32.GetCursorPos(ctypes.byref(positioned_cursor))
            or (
                int(positioned_cursor.x),
                int(positioned_cursor.y),
            )
            != (start_x, start_y)
        ):
            raise CollectionError("window_repaint_cursor_position_invalid")
        send(0x0002)
        button_down = True
        time.sleep(0.02)
        drag_steps = 8
        for step in range(1, drag_steps + 1):
            drag_x = start_x + round(delta_x * step / drag_steps)
            send(absolute_move_flags, drag_x, start_y)
            time.sleep(0.01)
        send(0x0004)
        button_down = False
        time.sleep(0.02)
    finally:
        if button_down:
            send(0x0004)
        send(
            absolute_move_flags,
            int(original_cursor.x),
            int(original_cursor.y),
        )


def _restore_window_origin(
    window_handle: int,
    left: int,
    top: int,
) -> None:
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.SetWindowPos.argtypes = (
        wintypes.HWND,
        wintypes.HWND,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        wintypes.UINT,
    )
    user32.SetWindowPos.restype = wintypes.BOOL
    # Preserve size, z-order and activation.  Callers use this for a bounded
    # temporary translation and/or exact restoration.
    flags = 0x0001 | 0x0004 | 0x0010
    if not user32.SetWindowPos(
        wintypes.HWND(window_handle),
        wintypes.HWND(0),
        left,
        top,
        0,
        0,
        flags,
    ):
        raise CollectionError("window_repaint_restore_failed")


def _virtual_screen_rect() -> tuple[int, int, int, int]:
    """Return the physical virtual-desktop bounds used for a safe move."""

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.GetSystemMetrics.argtypes = (ctypes.c_int,)
    user32.GetSystemMetrics.restype = ctypes.c_int
    left = int(user32.GetSystemMetrics(76))  # SM_XVIRTUALSCREEN
    top = int(user32.GetSystemMetrics(77))  # SM_YVIRTUALSCREEN
    width = int(user32.GetSystemMetrics(78))  # SM_CXVIRTUALSCREEN
    height = int(user32.GetSystemMetrics(79))  # SM_CYVIRTUALSCREEN
    if width <= 0 or height <= 0:
        raise CollectionError("window_repaint_virtual_screen_invalid")
    return left, top, left + width, top + height


def _dwm_flush() -> None:
    """Wait until all queued DirectX/DWM surface updates are composed."""

    dwmapi = ctypes.WinDLL("dwmapi", use_last_error=True)
    dwmapi.DwmFlush.argtypes = ()
    dwmapi.DwmFlush.restype = ctypes.c_long
    if int(dwmapi.DwmFlush()) != 0:
        raise CollectionError("window_repaint_dwm_flush_failed")


def _window_position_repaint_handshake(
    target: WindowTarget,
    *,
    get_outer_rect: Any = _window_outer_rect,
    get_virtual_screen_rect: Any = _virtual_screen_rect,
    set_window_origin: Any = _restore_window_origin,
    flush_dwm: Any = _dwm_flush,
    monotonic_ns: Any = time.perf_counter_ns,
    delay: Any = time.sleep,
) -> Mapping[str, Any]:
    """Recompose a window without synthesizing keyboard or mouse input."""

    before = get_outer_rect(target.window_handle)
    virtual = get_virtual_screen_rect()
    if (
        len(before) != 4
        or len(virtual) != 4
        or before[0] >= before[2]
        or before[1] >= before[3]
        or virtual[0] >= virtual[2]
        or virtual[1] >= virtual[3]
    ):
        raise CollectionError("window_repaint_geometry_invalid")
    if before[2] + 32 <= virtual[2]:
        delta_x = 32
    elif before[0] - 32 >= virtual[0]:
        delta_x = -32
    else:
        raise CollectionError("window_repaint_no_bounded_translation")

    expected_moved = (
        before[0] + delta_x,
        before[1],
        before[2] + delta_x,
        before[3],
    )
    hold_seconds = 0.05
    started_ns = monotonic_ns()
    moved_rect = before
    moved_ns = started_ns
    held_ns = started_ns
    restored_ns = started_ns
    try:
        set_window_origin(
            target.window_handle,
            expected_moved[0],
            expected_moved[1],
        )
        flush_dwm()
        moved_ns = monotonic_ns()
        moved_rect = get_outer_rect(target.window_handle)
        if moved_rect != expected_moved:
            raise CollectionError("window_repaint_translation_mismatch")
        delay(hold_seconds)
        held_ns = monotonic_ns()
        set_window_origin(target.window_handle, before[0], before[1])
        flush_dwm()
        restored_ns = monotonic_ns()
    except Exception:
        if get_outer_rect(target.window_handle) != before:
            try:
                set_window_origin(target.window_handle, before[0], before[1])
                flush_dwm()
            except Exception as restore_error:
                raise CollectionError(
                    "window_repaint_emergency_restore_failed"
                ) from restore_error
        raise

    delay(hold_seconds)
    after = get_outer_rect(target.window_handle)
    finished_ns = monotonic_ns()
    if after != before:
        raise CollectionError("window_repaint_geometry_not_restored")
    if not (
        started_ns < moved_ns < held_ns < restored_ns < finished_ns
    ):
        raise CollectionError("window_repaint_clock_invalid")
    return {
        "schema": REPAINT_SCHEMA,
        "version": REPAINT_VERSION,
        "mechanism": SET_WINDOW_POS_REPAINT_MECHANISM,
        "restoration_mechanism": "set_window_pos_exact_origin",
        "input_transport": "none_window_manager_api_only",
        "synthetic_input_event_count": 0,
        "dwm_flush_count": 2,
        "process_id": target.process_id,
        "window_handle_hex": f"0x{target.window_handle:016x}",
        "client_region": list(target.client_region),
        "virtual_screen_rect": list(virtual),
        "outer_rect_before": list(before),
        "outer_rect_moved": list(moved_rect),
        "requested_temporary_delta": [delta_x, 0],
        "actual_temporary_delta": [delta_x, 0],
        "temporary_hold_seconds": hold_seconds,
        "outer_rect_after": list(after),
        "started_perf_counter_ns": started_ns,
        "moved_perf_counter_ns": moved_ns,
        "held_perf_counter_ns": held_ns,
        "restored_perf_counter_ns": restored_ns,
        "finished_perf_counter_ns": finished_ns,
        "geometry_restored": True,
    }


def _window_repaint_handshake(
    target: WindowTarget,
    *,
    get_outer_rect: Any = _window_outer_rect,
    drag_titlebar_once: Any = _drag_window_titlebar_once,
    restore_window_origin: Any = _restore_window_origin,
    monotonic_ns: Any = time.perf_counter_ns,
    delay: Any = time.sleep,
) -> Mapping[str, Any]:
    """Force a real non-client drag, then restore the exact window origin."""

    before = get_outer_rect(target.window_handle)
    started_ns = monotonic_ns()
    hold_seconds = 0.05
    attempts: list[dict[str, Any]] = []
    moved_rect = before
    moved_ns = started_ns
    delta_x = 0
    for attempt, requested_delta_x in enumerate((32, 48, 64), start=1):
        drag_titlebar_once(target, requested_delta_x)
        attempt_finished_ns = monotonic_ns()
        observed_rect = get_outer_rect(target.window_handle)
        attempts.append(
            {
                "attempt": attempt,
                "requested_delta": [requested_delta_x, 0],
                "outer_rect_after": list(observed_rect),
                "finished_perf_counter_ns": attempt_finished_ns,
                "movement_detected": observed_rect != before,
            }
        )
        if observed_rect != before:
            delta_x = requested_delta_x
            moved_rect = observed_rect
            moved_ns = attempt_finished_ns
            break
        delay(0.02)
    if moved_rect == before:
        raise CollectionError("window_repaint_drag_no_movement")
    actual_delta = (
        moved_rect[0] - before[0],
        moved_rect[1] - before[1],
    )
    rigid_delta = (
        moved_rect[2] - before[2],
        moved_rect[3] - before[3],
    )
    delay(hold_seconds)
    held_ns = monotonic_ns()
    restore_window_origin(target.window_handle, before[0], before[1])
    restored_ns = monotonic_ns()
    delay(hold_seconds)
    finished_ns = monotonic_ns()
    after = get_outer_rect(target.window_handle)
    if after != before:
        raise CollectionError("window_repaint_geometry_not_restored")
    if rigid_delta != actual_delta:
        raise CollectionError("window_repaint_drag_nonrigid")
    if actual_delta[1] != 0:
        raise CollectionError("window_repaint_drag_vertical_shift")
    if not 1 <= abs(actual_delta[0]) <= 128:
        raise CollectionError("window_repaint_drag_range_invalid")
    if not (
        started_ns < moved_ns < held_ns < restored_ns < finished_ns
    ):
        raise CollectionError("window_repaint_clock_invalid")
    return {
        "schema": REPAINT_SCHEMA,
        "version": REPAINT_VERSION,
        "mechanism": NONCLIENT_DRAG_REPAINT_MECHANISM,
        "restoration_mechanism": "set_window_pos_exact_origin",
        "process_id": target.process_id,
        "window_handle_hex": f"0x{target.window_handle:016x}",
        "client_region": list(target.client_region),
        "outer_rect_before": list(before),
        "outer_rect_moved": list(moved_rect),
        "requested_temporary_delta": [delta_x, 0],
        "actual_temporary_delta": list(actual_delta),
        "drag_attempts": attempts,
        "temporary_hold_seconds": hold_seconds,
        "outer_rect_after": list(after),
        "started_perf_counter_ns": started_ns,
        "moved_perf_counter_ns": moved_ns,
        "held_perf_counter_ns": held_ns,
        "restored_perf_counter_ns": restored_ns,
        "finished_perf_counter_ns": finished_ns,
        "geometry_restored": True,
    }


def _wait_for_runtime(
    trace_process: subprocess.Popen[bytes],
    *,
    existing: set[int],
    expected_runtime: Path,
    timeout: float,
) -> tuple[int, int]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if trace_process.poll() is not None:
            raise CollectionError("trace_exited_before_runtime")
        candidates: list[int] = []
        for pid in process_ids_by_name(expected_runtime.name):
            if pid in existing:
                continue
            image = process_image_path(pid)
            if image is not None and same_windows_path(
                image,
                expected_runtime,
            ):
                candidates.append(pid)
        if len(candidates) == 1:
            return candidates[0], time.perf_counter_ns()
        if len(candidates) > 1:
            raise CollectionError("runtime_process_not_unique")
        time.sleep(0.02)
    raise CollectionError("runtime_launch_timeout")


def _wait_for_capture_window(
    pid: int,
    timeout: float = CAPTURE_WINDOW_TIMEOUT_SECONDS,
) -> WindowTarget:
    deadline = time.monotonic() + timeout
    last_code = "target_window_missing"
    last_dimensions: tuple[int, int] | None = None
    while time.monotonic() < deadline:
        if pid not in process_ids_by_name(DEFAULT_RUNTIME_EXE.name):
            raise CollectionError("runtime_exited_before_capture")
        try:
            target, region = resolve_windows_target(
                process_name=DEFAULT_RUNTIME_EXE.name,
                explicit_region=None,
            )
            if target.process_id != pid:
                raise CollectionError("runtime_process_identity_changed")
            dimensions = (region[2] - region[0], region[3] - region[1])
            if dimensions != last_dimensions:
                print(
                    "capture_window_observation "
                    f"pid={pid} "
                    f"client_width={dimensions[0]} "
                    f"client_height={dimensions[1]}",
                    flush=True,
                )
                last_dimensions = dimensions
            if dimensions != (800, 600):
                last_code = "runtime_client_size_mismatch"
                time.sleep(0.05)
                continue
            verify_windows_target(target, require_foreground=False)
            return target
        except CaptureError as error:
            last_code = error.code
            time.sleep(0.05)
    raise CollectionError(f"capture_window_{last_code}")


def _wait_for_window_acquisition_update(
    *,
    trace_process: subprocess.Popen[bytes],
    pid: int,
    target_update: int,
    timeout: float = CAPTURE_WINDOW_TIMEOUT_SECONDS,
) -> int:
    deadline = time.monotonic() + timeout
    reader: FrameworkUpdateReader | None = None
    try:
        while time.monotonic() < deadline:
            if trace_process.poll() is not None:
                raise CollectionError("trace_exited_before_capture")
            if pid not in process_ids_by_name(DEFAULT_RUNTIME_EXE.name):
                raise CollectionError("runtime_exited_before_capture")
            try:
                if reader is None:
                    reader = FrameworkUpdateReader(pid)
                update = reader.sample()
            except CaptureError:
                if reader is not None:
                    reader.close()
                    reader = None
                time.sleep(0.01)
                continue
            if update >= target_update:
                return update
            time.sleep(0.002)
    finally:
        if reader is not None:
            reader.close()
    raise CollectionError("capture_window_update_timeout")


def _window_target_evidence(target: WindowTarget) -> Mapping[str, Any]:
    return {
        "process_id": target.process_id,
        "window_handle_hex": f"0x{target.window_handle:016x}",
        "client_region": list(target.client_region),
    }


def _activate_startup_window(
    target: WindowTarget,
    *,
    activate_window: Any = _activate_window,
    verify_target: Any = verify_windows_target,
    monotonic_ns: Any = time.perf_counter_ns,
) -> Mapping[str, Any]:
    activation_started_ns = monotonic_ns()
    try:
        activation_attempts = _activate_verified_window(
            target,
            activate_window=activate_window,
            verify_target=verify_target,
        )
    except CaptureError as error:
        raise CollectionError(
            f"startup_activation_{error.code}"
        ) from None
    activation_finished_ns = monotonic_ns()
    return {
        "schema": "zuma-rl.pc-golden-startup-window-activation",
        "version": 1,
        "window_target": _window_target_evidence(target),
        "mechanism": (
            "show_restore_attach_input_then_set_foreground"
        ),
        "activation_attempts": activation_attempts,
        "activation_started_perf_counter_ns": activation_started_ns,
        "activation_finished_perf_counter_ns": activation_finished_ns,
        "foreground_activation_verified": True,
        "reason": (
            "reproduce the already-active application state of a "
            "full-screen recording before replayed mouse input"
        ),
    }


def _validate_rebound_target(
    *,
    expected_process_id: int,
    target: WindowTarget,
) -> None:
    if target.process_id != expected_process_id:
        raise CollectionError("runtime_process_identity_changed")
    if (
        target.client_region[2] - target.client_region[0],
        target.client_region[3] - target.client_region[1],
    ) != (800, 600):
        raise CollectionError("runtime_client_size_mismatch")


def _wait_for_repaint_guard(
    *,
    target: WindowTarget,
    trace_process: subprocess.Popen[bytes],
    trigger_update: int,
    previous_effectful_command_update: int,
    next_effectful_command_update: int,
    guarded_idle_command_count: int,
    timeout: float = 120.0,
    update_reader_factory: Any = FrameworkUpdateReader,
    verify_target: Any = verify_windows_target,
    refresh_target: Any = _wait_for_capture_window,
    activate_window: Any = _activate_window,
    repaint_handshake: Any = _window_repaint_handshake,
    monotonic: Any = time.monotonic,
    monotonic_ns: Any = time.perf_counter_ns,
    delay: Any = time.sleep,
) -> Mapping[str, Any]:
    deadline = monotonic() + timeout
    initial_target = target
    bound_target = target
    rebindings: list[dict[str, Any]] = []

    def verify_or_rebind(observed_update: int) -> None:
        nonlocal bound_target
        try:
            verify_target(
                bound_target,
                require_foreground=False,
            )
            return
        except CaptureError as error:
            if error.code != "target_window_changed":
                raise
        if trace_process.poll() is not None:
            raise CollectionError("strict_replay_failed")
        refreshed = refresh_target(
            initial_target.process_id,
            timeout=5.0,
        )
        _validate_rebound_target(
            expected_process_id=initial_target.process_id,
            target=refreshed,
        )
        previous_target = bound_target
        bound_target = refreshed
        verify_target(
            bound_target,
            require_foreground=False,
        )
        if bound_target != previous_target:
            rebindings.append(
                {
                    "observed_framework_update": (
                        None if observed_update < 0 else observed_update
                    ),
                    "observed_perf_counter_ns": monotonic_ns(),
                    "previous": _window_target_evidence(previous_target),
                    "replacement": _window_target_evidence(bound_target),
                    "same_process_verified": True,
                    "expected_client_size_verified": True,
                }
            )

    try:
        with update_reader_factory(target.process_id) as update_reader:
            observed_update = -1
            while monotonic() < deadline:
                if trace_process.poll() is not None:
                    raise CollectionError("strict_replay_failed")
                verify_or_rebind(observed_update)
                observed_update = update_reader.sample()
                if observed_update >= trigger_update:
                    break
                delay(0.002)
            else:
                raise CollectionError("window_repaint_update_timeout")
            if observed_update >= next_effectful_command_update:
                raise CollectionError("window_repaint_guard_missed")
            verify_or_rebind(observed_update)
            activation_started_ns = monotonic_ns()
            activation_attempts = _activate_verified_window(
                bound_target,
                activate_window=activate_window,
                verify_target=verify_target,
                delay=delay,
            )
            activation_finished_ns = monotonic_ns()
            repaint = dict(repaint_handshake(bound_target))
            after_update = update_reader.sample()
    except CaptureError as error:
        raise CollectionError(f"repaint_{error.code}") from None
    verify_target(bound_target)
    if after_update >= next_effectful_command_update:
        raise CollectionError("window_repaint_guard_overrun")
    return {
        **repaint,
        "wait_timeout_seconds": timeout,
        "initial_window_target": _window_target_evidence(initial_target),
        "capture_window_target": _window_target_evidence(bound_target),
        "window_rebind_count": len(rebindings),
        "window_rebindings": rebindings,
        "requested_framework_update": trigger_update,
        "framework_update_before": observed_update,
        "framework_update_after": after_update,
        "activation_started_perf_counter_ns": activation_started_ns,
        "activation_finished_perf_counter_ns": activation_finished_ns,
        "activation_attempts": activation_attempts,
        "foreground_activation_mechanism": (
            "show_restore_attach_input_then_set_foreground"
        ),
        "foreground_activation_verified": True,
        "previous_effectful_command_update": (
            previous_effectful_command_update
        ),
        "next_effectful_command_update": next_effectful_command_update,
        "allowed_overlapping_command_kinds": ["idle"],
        "guarded_idle_command_count": guarded_idle_command_count,
        "effectful_command_guard_satisfied": True,
    }


def _validate_run_timeline(
    *,
    start_snapshot_ns: int,
    observed_process_start_ns: int,
    capture_start_ns: int,
    capture_end_ns: int,
    trace_start_ns: int,
    trace_stop_ns: int,
) -> None:
    if not (
        start_snapshot_ns
        < observed_process_start_ns
        < capture_start_ns
        < capture_end_ns
        < trace_stop_ns
    ):
        raise CollectionError("capture_process_lifetime_mismatch")
    if not (
        start_snapshot_ns
        < trace_start_ns
        <= observed_process_start_ns
        < trace_stop_ns
    ):
        raise CollectionError("trace_process_lifetime_mismatch")


def _terminate_exact_process(pid: int, expected_image: Path) -> None:
    image = process_image_path(pid)
    if image is None or not same_windows_path(image, expected_image):
        return
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = (
        wintypes.DWORD,
        wintypes.BOOL,
        wintypes.DWORD,
    )
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.TerminateProcess.argtypes = (wintypes.HANDLE, wintypes.UINT)
    kernel32.TerminateProcess.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL
    handle = kernel32.OpenProcess(0x0001 | 0x00100000, False, pid)
    if not handle:
        return
    try:
        kernel32.TerminateProcess(handle, 1)
    finally:
        kernel32.CloseHandle(handle)


def _terminate_exact_runtime(
    pid: int,
    expected_runtime: Path = DEFAULT_RUNTIME_EXE,
) -> None:
    _terminate_exact_process(pid, expected_runtime)


def _wait_game_stopped(timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not any(_game_processes().values()):
            return
        time.sleep(0.05)
    raise CollectionError("game_process_did_not_stop")


def _child_environment(project_root: Path) -> dict[str, str]:
    environment = dict(os.environ)
    src = str(project_root / "src")
    current = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = src if not current else src + os.pathsep + current
    return environment


def _tail(path: Path, limit: int = 4096) -> str:
    try:
        payload = path.read_bytes()
    except OSError:
        return ""
    return payload[-limit:].decode("utf-8", errors="replace")


def _validate_font_cache_trace_receipt(
    payload: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    expected_startup_handoff: bool,
    expected_allow_pre_attach_debt: bool,
    expected_allow_manifest_completion: bool,
    expected_manifest_receipt: Mapping[str, Any] | None,
) -> None:
    options = payload.get("options")
    if (
        not isinstance(options, dict)
        or options.get("allow_pre_attach_file_write_debt")
        is not expected_allow_pre_attach_debt
        or options.get("allow_font_cache_manifest_completion_debt")
        is not expected_allow_manifest_completion
        or options.get("startup_trace_handoff", False)
        is not expected_startup_handoff
    ):
        raise CollectionError("trace_result_invalid")
    pre_attach_rows = result.get("pre_attach_file_write_debt_rows", [])
    if (
        not isinstance(pre_attach_rows, list)
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in pre_attach_rows
        )
        or pre_attach_rows != sorted(set(pre_attach_rows))
        or bool(pre_attach_rows) is not expected_allow_pre_attach_debt
    ):
        raise CollectionError("trace_result_invalid")
    if not expected_allow_manifest_completion:
        if (
            expected_manifest_receipt is not None
            or result.get(
                "font_cache_manifest_completion_discharges",
                [],
            )
            != []
            or result.get(
                "direct_font_cache_file_write_observations",
                [],
            )
            != []
            or result.get("font_cache_manifest_entry_count", 0) != 0
            or result.get("font_cache_manifest_sha256") is not None
            or result.get("font_cache_manifest_main_pak_sha256") is not None
        ):
            raise CollectionError("trace_result_invalid")
        return
    if not isinstance(expected_manifest_receipt, Mapping):
        raise CollectionError("trace_result_invalid")
    members = expected_manifest_receipt.get("members")
    startup_rows = expected_manifest_receipt.get(
        "startup_file_write_rows"
    )
    expected_pre_attach = expected_manifest_receipt.get(
        "pre_attach_file_write_rows"
    )
    if (
        expected_manifest_receipt.get("entry_count")
        != FONT_CACHE_MANIFEST_EXPECTED_ENTRIES
        or not isinstance(members, list)
        or len(members) != FONT_CACHE_MANIFEST_EXPECTED_ENTRIES
        or not isinstance(startup_rows, list)
        or len(startup_rows) != FONT_CACHE_MANIFEST_EXPECTED_ENTRIES
        or not isinstance(expected_pre_attach, list)
        or bool(expected_pre_attach)
        is not expected_allow_pre_attach_debt
        or expected_allow_pre_attach_debt == expected_startup_handoff
    ):
        raise CollectionError("trace_result_invalid")
    member_sizes: dict[str, int] = {}
    for row in members:
        if not isinstance(row, dict):
            raise CollectionError("trace_result_invalid")
        member = row.get("member")
        size = row.get("size")
        if (
            not isinstance(member, str)
            or _normalize_font_cache_member(member) != member
            or member in member_sizes
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
        ):
            raise CollectionError("trace_result_invalid")
        member_sizes[member] = size
    startup_by_index: dict[int, Mapping[str, Any]] = {}
    for row in startup_rows:
        if (
            not isinstance(row, dict)
            or set(row)
            != {
                "row_index",
                "update",
                "command_number",
                "kind",
                "short_form",
                "success",
            }
            or isinstance(row.get("row_index"), bool)
            or not isinstance(row.get("row_index"), int)
            or row["row_index"] < 0
            or row["row_index"] in startup_by_index
            or row.get("command_number") != 16
            or row.get("kind") != "file_write"
            or row.get("short_form") is not False
            or row.get("success") is not False
        ):
            raise CollectionError("trace_result_invalid")
        startup_by_index[row["row_index"]] = row
    expected_pre_attach_indices = [
        row.get("row_index")
        for row in expected_pre_attach
        if isinstance(row, dict)
    ]
    if (
        len(expected_pre_attach_indices) != len(expected_pre_attach)
        or pre_attach_rows != expected_pre_attach_indices
        or any(
            index not in startup_by_index
            for index in expected_pre_attach_indices
        )
    ):
        raise CollectionError("trace_result_invalid")
    expected_manifest_sha256 = expected_manifest_receipt.get(
        "manifest_sha256"
    )
    expected_pak_sha256 = expected_manifest_receipt.get(
        "main_pak_sha256"
    )
    if (
        result.get("font_cache_manifest_entry_count")
        != FONT_CACHE_MANIFEST_EXPECTED_ENTRIES
        or f"sha256:{result.get('font_cache_manifest_sha256')}"
        != expected_manifest_sha256
        or f"sha256:{result.get('font_cache_manifest_main_pak_sha256')}"
        != expected_pak_sha256
    ):
        raise CollectionError("trace_result_invalid")
    rebase_rows = result.get("command_order_rebase_rows")
    command_order_offset = result.get("command_order_offset")
    if (
        not isinstance(rebase_rows, list)
        or rebase_rows != sorted(set(rebase_rows))
        or isinstance(command_order_offset, bool)
        or not isinstance(command_order_offset, int)
        or command_order_offset != len(rebase_rows)
        or any(index not in startup_by_index for index in rebase_rows)
        or set(rebase_rows) & set(pre_attach_rows)
    ):
        raise CollectionError("trace_result_invalid")
    terminal = result.get("terminal_file_write_payload_handoffs", [])
    service_claims = result.get("service_file_write_header_claims", [])
    direct = result.get(
        "direct_font_cache_file_write_observations", []
    )
    deferred = result.get("deferred_file_write_discharges", [])
    completions = result.get(
        "font_cache_manifest_completion_discharges",
        [],
    )
    service_continuations = result.get(
        "service_continuation_verifications", []
    )
    service_exit_commits = result.get("service_exit_timeline_commits", [])
    if (
        not isinstance(terminal, list)
        or len(terminal) > 1
        or not isinstance(service_claims, list)
        or not isinstance(direct, list)
        or not isinstance(deferred, list)
        or not isinstance(completions, list)
        or not isinstance(service_continuations, list)
        or not isinstance(service_exit_commits, list)
        or len(deferred) != len(rebase_rows) + len(pre_attach_rows)
    ):
        raise CollectionError("trace_result_invalid")
    for observation in service_claims:
        if not isinstance(observation, dict):
            raise CollectionError("trace_result_invalid")
        mechanism = observation.get("mechanism")
        if mechanism is None:
            continue
        integer_fields = (
            "process_id",
            "thread_id",
            "framework_update",
            "row_index",
            "row_start",
            "row_end",
            "row_update",
            "command_order",
            "prior_brokered_row_index",
            "prior_brokered_read_bit_position",
            "prior_brokered_command_order",
            "payload_completion_thread_id",
            "payload_completion_framework_update",
            "payload_completion_command_bit_position",
            "payload_completion_buffer_read_bit_position",
            "payload_completion_command_order",
        )
        values = {
            name: observation.get(name) for name in integer_fields
        }
        if (
            mechanism
            != "broker_adjacent_current_update_file_write"
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in values.values()
            )
            or values["process_id"] <= 0
            or values["thread_id"] <= 0
            or values["payload_completion_thread_id"] <= 0
            or values["row_index"] <= 0
            or values["row_end"] - values["row_start"] != 11
            or values["row_update"] != values["framework_update"]
            or values["prior_brokered_row_index"]
            != values["row_index"] - 1
            or values["prior_brokered_read_bit_position"]
            != values["row_start"]
            or values["prior_brokered_command_order"]
            != values["command_order"] - 1
            or observation.get("payload_completion_verified") is not True
            or values["payload_completion_framework_update"]
            != values["framework_update"]
            or values["payload_completion_command_bit_position"]
            != values["row_start"]
            or values["payload_completion_buffer_read_bit_position"]
            != values["row_end"]
            or values["payload_completion_command_order"]
            != values["command_order"]
            or observation.get("natural_payload_consumer") is not True
        ):
            raise CollectionError("trace_result_invalid")
    expected_debt_rows = rebase_rows + pre_attach_rows
    for index, observation in enumerate(deferred):
        if (
            not isinstance(observation, dict)
            or observation.get("debt_row_index")
            != expected_debt_rows[index]
            or observation.get("recorded_success") is not False
            or observation.get("pre_attach_debt")
            is not (expected_debt_rows[index] in pre_attach_rows)
        ):
            raise CollectionError("trace_result_invalid")
    pre_loading_observations: list[Mapping[str, Any]] = []
    for observation in deferred + completions:
        if not isinstance(observation, dict):
            raise CollectionError("trace_result_invalid")
        loading_complete = observation.get("demo_loading_complete", 1)
        pre_loading = observation.get(
            "pre_loading_service_continuation", False
        )
        if (
            loading_complete not in (0, 1)
            or not isinstance(pre_loading, bool)
            or pre_loading is not (loading_complete == 0)
        ):
            raise CollectionError("trace_result_invalid")
        if pre_loading:
            pre_loading_observations.append(observation)
    if pre_loading_observations:
        if (
            not expected_startup_handoff
            or expected_allow_pre_attach_debt
            or not service_continuations
            or not service_exit_commits
        ):
            raise CollectionError("trace_result_invalid")
        verified_updates = {
            observation.get("service_continuation_verified_update")
            for observation in pre_loading_observations
        }
        if (
            len(verified_updates) != 1
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in verified_updates
            )
        ):
            raise CollectionError("trace_result_invalid")
        verified_update = next(iter(verified_updates))
        matching_verifications = [
            verification
            for verification in service_continuations
            if isinstance(verification, dict)
            and verification.get("framework_update") == verified_update
        ]
        if len(matching_verifications) != 1:
            raise CollectionError("trace_result_invalid")
        verification = matching_verifications[0]
        row_index = verification.get("row_index")
        corridor_end_index = verification.get("corridor_end_index")
        if (
            verified_update < 0
            or isinstance(row_index, bool)
            or not isinstance(row_index, int)
            or isinstance(corridor_end_index, bool)
            or not isinstance(corridor_end_index, int)
            or row_index <= corridor_end_index
            or verification.get(
                "service_exit_timeline_commit_count"
            )
            != len(service_exit_commits)
        ):
            raise CollectionError("trace_result_invalid")
        for observation in pre_loading_observations:
            if (
                isinstance(observation.get("framework_update"), bool)
                or not isinstance(
                    observation.get("framework_update"), int
                )
                or observation["framework_update"] < verified_update
            ):
                raise CollectionError("trace_result_invalid")
    direct_row_indices: list[int] = []
    for observation in direct:
        if not isinstance(observation, dict):
            raise CollectionError("trace_result_invalid")
        row_index = observation.get("row_index")
        if (
            isinstance(row_index, bool)
            or not isinstance(row_index, int)
            or row_index not in startup_by_index
            or row_index in expected_debt_rows
            or observation.get("row_update")
            != startup_by_index[row_index]["update"]
            or observation.get("recorded_success") is not False
        ):
            raise CollectionError("trace_result_invalid")
        direct_row_indices.append(row_index)
    if direct_row_indices != sorted(set(direct_row_indices)):
        raise CollectionError("trace_result_invalid")
    observed_groups = (
        terminal,
        service_claims,
        direct,
        deferred,
        completions,
    )
    observed_members: set[str] = set()
    for group in observed_groups:
        for observation in group:
            if not isinstance(observation, dict):
                raise CollectionError("trace_result_invalid")
            member = observation.get("file_write_font_cache_member")
            size = observation.get("file_write_argument_size")
            if (
                not isinstance(member, str)
                or member in observed_members
                or member_sizes.get(member) != size
                or observation.get("recorded_success") is not False
            ):
                raise CollectionError("trace_result_invalid")
            observed_members.add(member)
    if (
        len(pre_attach_rows) + len(observed_members)
        > FONT_CACHE_MANIFEST_EXPECTED_ENTRIES
    ):
        raise CollectionError("trace_result_invalid")
    prior_completion_base = (
        len(terminal) + len(service_claims) + len(direct) + len(deferred)
    )
    for index, observation in enumerate(completions):
        member = observation["file_write_font_cache_member"]
        if (
            observation.get("manifest_entry_count")
            != FONT_CACHE_MANIFEST_EXPECTED_ENTRIES
            or f"sha256:{observation.get('manifest_sha256')}"
            != expected_manifest_sha256
            or f"sha256:{observation.get('main_pak_sha256')}"
            != expected_pak_sha256
            or observation.get("manifest_expected_size")
            != member_sizes[member]
            or observation.get("startup_file_write_row_count")
            != FONT_CACHE_MANIFEST_EXPECTED_ENTRIES
            or observation.get("explicit_debt_discharge_count")
            != len(deferred)
            or observation.get("prior_observed_member_count")
            != prior_completion_base + index
        ):
                raise CollectionError("trace_result_invalid")


def _validate_main_file_read_corridor_trace_receipts(
    result: Mapping[str, Any],
    *,
    expected_pid: int,
) -> None:
    observations = result.get(
        "main_file_read_corridor_native_replays", []
    )
    if not isinstance(observations, list):
        raise CollectionError("trace_result_invalid")
    seen: set[tuple[int, int, int]] = set()
    integer_fields = (
        "process_id",
        "thread_id",
        "start_index",
        "end_index",
        "update",
        "file_read_count",
        "large_file_read_row_index",
        "large_file_read_bytes",
        "matching_large_payload_occurrences",
        "activated_framework_update",
        "command_order_offset",
        "activation_read_bit_position",
    )
    for observation in observations:
        if not isinstance(observation, dict):
            raise CollectionError("trace_result_invalid")
        values = {name: observation.get(name) for name in integer_fields}
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in values.values()
        ):
            raise CollectionError("trace_result_invalid")
        key = (
            values["start_index"],
            values["end_index"],
            values["update"],
        )
        digest = observation.get("large_file_read_sha256")
        if (
            key in seen
            or values["process_id"] != expected_pid
            or values["thread_id"] <= 0
            or values["start_index"] < 0
            or values["end_index"] < values["start_index"]
            or not (
                values["start_index"]
                <= values["large_file_read_row_index"]
                <= values["end_index"]
            )
            or values["file_read_count"] <= 0
            or not (
                MAIN_FILE_READ_CORRIDOR_MIN_BYTES
                <= values["large_file_read_bytes"]
                <= MAIN_FILE_READ_CORRIDOR_MAX_BYTES
            )
            or values["matching_large_payload_occurrences"] != 2
            or values["activated_framework_update"] != values["update"]
            or values["command_order_offset"] < 0
            or values["activation_read_bit_position"] < 0
            or observation.get("mode")
            != "exact_main_thread_native_replay"
            or not isinstance(digest, str)
            or len(digest) != 71
            or not digest.startswith("sha256:")
            or any(
                character not in "0123456789abcdef"
                for character in digest.removeprefix("sha256:")
            )
        ):
            raise CollectionError("trace_result_invalid")
        seen.add(key)


def _validate_diagnostic_successful_file_write_receipts(
    result: Mapping[str, Any],
    *,
    expected_pid: int,
    padding_rows: list[int],
) -> bool:
    """Bind the diagnostic padding path to its complete audited receipts."""

    if not padding_rows:
        return False

    rebase_rows = result.get("command_order_rebase_rows")
    command_order_offset = result.get("command_order_offset")
    service_rows = result.get("service_consumed_file_write_debt_rows")
    surplus_rows = result.get(
        "service_consumed_file_write_surplus_rows"
    )
    effective_debt_rows = result.get(
        "effective_deferred_file_write_debt_rows"
    )
    verifications = result.get("service_continuation_verifications")
    commits = result.get("service_exit_timeline_commits")
    suffix_rebases = result.get(
        "service_exit_timeline_suffix_rebases"
    )
    eof_exits = result.get("terminal_registry_write_eof_exits")
    barriers = result.get("successful_file_write_debt_barriers")
    if (
        not isinstance(rebase_rows, list)
        or rebase_rows != sorted(set(rebase_rows))
        or not all(
            not isinstance(value, bool)
            and isinstance(value, int)
            and value >= 0
            for value in rebase_rows
        )
        or isinstance(command_order_offset, bool)
        or not isinstance(command_order_offset, int)
        or command_order_offset != len(rebase_rows)
        or command_order_offset <= 0
        or not isinstance(service_rows, list)
        or not isinstance(surplus_rows, list)
        or not isinstance(effective_debt_rows, list)
        or not isinstance(verifications, list)
        or len(verifications) != 2
        or not isinstance(commits, list)
        or len(commits) != 1
        or not isinstance(suffix_rebases, list)
        or not isinstance(eof_exits, list)
        or len(eof_exits) != 1
        or not isinstance(barriers, list)
        or len(barriers) != 1
    ):
        raise CollectionError("trace_result_invalid")

    bridge, deferred_exit = verifications
    bridge_keys = {
        "process_id",
        "thread_id",
        "framework_update",
        "row_index",
        "corridor_start_index",
        "corridor_end_index",
        "reentries",
        "command_order_offset",
        "service_exit_timeline_commit_count",
        "bridge_recorded_update",
        "bridge_following_row_index",
        "bridge_following_recorded_update",
        "bridge_following_native_update",
        "bridge_late_updates",
        "native_timeline_rebase_start_row",
        "native_timeline_offset_before",
        "native_timeline_offset_delta",
        "native_timeline_offset_after",
        "verification_mode",
    }
    deferred_exit_keys = {
        "process_id",
        "thread_id",
        "framework_update",
        "row_index",
        "corridor_start_index",
        "corridor_end_index",
        "reentries",
        "command_order_offset",
        "service_exit_timeline_commit_count",
        "service_consumed_file_write_debt_rows",
        "verification_mode",
    }
    if (
        not isinstance(bridge, dict)
        or set(bridge) != bridge_keys
        or not isinstance(deferred_exit, dict)
        or set(deferred_exit) != deferred_exit_keys
    ):
        raise CollectionError("trace_result_invalid")
    for receipt, string_fields in (
        (bridge, {"verification_mode"}),
        (
            deferred_exit,
            {
                "verification_mode",
                "service_consumed_file_write_debt_rows",
            },
        ),
    ):
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for key, value in receipt.items()
            if key not in string_fields
        ):
            raise CollectionError("trace_result_invalid")

    bridge_start = bridge["corridor_start_index"]
    bridge_end = bridge["corridor_end_index"]
    bridge_offset = bridge["command_order_offset"]
    deferred_start = deferred_exit["corridor_start_index"]
    deferred_end = deferred_exit["corridor_end_index"]
    first_rebase_rows = [
        row_index
        for row_index in rebase_rows
        if bridge_start <= row_index <= bridge_end
    ]
    deferred_rebase_rows = [
        row_index
        for row_index in rebase_rows
        if deferred_start <= row_index <= deferred_end
    ]
    if (
        bridge.get("process_id") != expected_pid
        or bridge["thread_id"] <= 0
        or bridge_start < 0
        or bridge_end < bridge_start
        or bridge["row_index"] != bridge_end + 1
        or bridge["bridge_following_row_index"]
        != bridge["row_index"] + 1
        or bridge["native_timeline_rebase_start_row"]
        != bridge["bridge_following_row_index"]
        or bridge["reentries"] <= 0
        or not 0 < bridge_offset < command_order_offset
        or first_rebase_rows != rebase_rows[:bridge_offset]
        or len(first_rebase_rows) != bridge_offset
        or bridge["service_exit_timeline_commit_count"] != 0
        or bridge["verification_mode"]
        != "successful_file_write_late_idle_bridge"
        or bridge["bridge_late_updates"] <= 0
        or bridge["framework_update"]
        - bridge["bridge_recorded_update"]
        != bridge["bridge_late_updates"]
        or bridge["bridge_following_recorded_update"]
        != bridge["framework_update"]
        or bridge["bridge_following_native_update"]
        - bridge["bridge_following_recorded_update"]
        != bridge["bridge_late_updates"]
        or bridge["native_timeline_offset_before"] < 0
        or bridge["native_timeline_offset_delta"]
        != bridge["bridge_late_updates"]
        or bridge["native_timeline_offset_after"]
        != bridge["native_timeline_offset_before"]
        + bridge["native_timeline_offset_delta"]
        or deferred_exit.get("process_id") != expected_pid
        or deferred_exit["thread_id"] != bridge["thread_id"]
        or deferred_start <= bridge_end
        or deferred_end < deferred_start
        or deferred_exit["row_index"] != deferred_end + 1
        or deferred_exit["reentries"] <= 0
        or deferred_exit["command_order_offset"]
        != command_order_offset
        or deferred_exit["service_exit_timeline_commit_count"] != 1
        or deferred_exit["verification_mode"]
        != "successful_file_write_deferred_exit"
        or deferred_exit[
            "service_consumed_file_write_debt_rows"
        ]
        != ",".join(str(row_index) for row_index in service_rows)
        or not set(padding_rows).issubset(deferred_rebase_rows)
        or set(service_rows) & set(deferred_rebase_rows)
        or set(range(deferred_start, deferred_end + 1))
        != set(service_rows) | set(deferred_rebase_rows)
        or set(rebase_rows)
        != set(first_rebase_rows) | set(deferred_rebase_rows)
    ):
        raise CollectionError("trace_result_invalid")

    expected_effective_debt_rows = sorted(
        (set(rebase_rows) | set(service_rows))
        - set(surplus_rows)
        - set(padding_rows)
    )
    if effective_debt_rows != expected_effective_debt_rows:
        raise CollectionError("trace_result_invalid")

    commit = commits[0]
    commit_keys = {
        "process_id",
        "thread_id",
        "framework_update",
        "row_index",
        "command_bit_position",
        "buffer_read_bit_position",
        "last_demo_update_before",
        "last_demo_update_after",
        "recorded_update",
        "effective_committed_update",
        "timeline_delta_updates",
        "recorded_timeline_delta_updates",
        "late_commit_clamp_updates",
        "corridor_end_row_index",
        "corridor_end_recorded_update",
        "bytes_written",
    }
    if (
        not isinstance(commit, dict)
        or set(commit) != commit_keys
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in commit.values()
        )
    ):
        raise CollectionError("trace_result_invalid")
    recorded_update = commit["recorded_update"]
    framework_update = commit["framework_update"]
    effective_update = commit["effective_committed_update"]
    clamp_updates = commit["late_commit_clamp_updates"]
    if (
        commit["process_id"] != expected_pid
        or commit["thread_id"] != bridge["thread_id"]
        or commit["row_index"] != deferred_exit["row_index"]
        or commit["corridor_end_row_index"] != deferred_end
        or commit["command_bit_position"] < 0
        or commit["buffer_read_bit_position"]
        - commit["command_bit_position"]
        != 10
        or commit["bytes_written"] != 4
        or recorded_update < 0
        or framework_update < 0
        or commit["corridor_end_recorded_update"] > recorded_update
        or effective_update != max(recorded_update, framework_update)
        or clamp_updates != effective_update - recorded_update
        or clamp_updates not in (0, 1)
        or commit["recorded_timeline_delta_updates"]
        != recorded_update - framework_update
        or commit["timeline_delta_updates"]
        != effective_update - framework_update
        or commit["last_demo_update_after"] != effective_update
        or commit["last_demo_update_before"] < effective_update
        or deferred_exit["framework_update"] != framework_update
    ):
        raise CollectionError("trace_result_invalid")

    if clamp_updates == 0:
        if suffix_rebases:
            raise CollectionError("trace_result_invalid")
    else:
        suffix_keys = {
            "process_id",
            "thread_id",
            "framework_update",
            "commit_row_index",
            "suffix_start_row_index",
            "timeline_offset_before",
            "timeline_offset_delta",
            "timeline_offset_after",
            "mechanism",
        }
        if (
            len(suffix_rebases) != 1
            or not isinstance(suffix_rebases[0], dict)
            or set(suffix_rebases[0]) != suffix_keys
        ):
            raise CollectionError("trace_result_invalid")
        suffix = suffix_rebases[0]
        suffix_integer_fields = suffix_keys - {"mechanism"}
        if (
            any(
                isinstance(suffix.get(name), bool)
                or not isinstance(suffix.get(name), int)
                for name in suffix_integer_fields
            )
            or suffix.get("process_id") != expected_pid
            or suffix.get("thread_id") != commit["thread_id"]
            or suffix.get("commit_row_index") != commit["row_index"]
            or suffix.get("suffix_start_row_index")
            != commit["row_index"] + 1
            or suffix.get("framework_update") <= framework_update
            or suffix.get("timeline_offset_before")
            != bridge["native_timeline_offset_after"]
            or suffix.get("timeline_offset_delta") != 1
            or suffix.get("timeline_offset_after")
            != suffix.get("timeline_offset_before") + 1
            or suffix.get("mechanism")
            != "one_tick_clamped_deferred_exit_suffix"
        ):
            raise CollectionError("trace_result_invalid")

    eof = eof_exits[0]
    eof_keys = {
        "process_id",
        "thread_id",
        "framework_update",
        "start_index",
        "end_index",
        "terminal_row_index",
        "terminal_row_start_bit_position",
        "terminal_payload_bit_position",
        "terminal_row_end_bit_position",
        "last_command_bit_position",
        "last_read_bit_position",
        "last_command_order",
        "command_order_offset",
        "reentries",
        "unconsumed_result_bits",
        "recorded_success",
        "exit_code",
        "mechanism",
    }
    if (
        not isinstance(eof, dict)
        or set(eof) != eof_keys
        or any(
            isinstance(eof.get(name), bool)
            or not isinstance(eof.get(name), int)
            for name in eof_keys - {"recorded_success", "mechanism"}
        )
        or eof.get("recorded_success") is not True
        or eof.get("mechanism")
        != "normal_exit_at_terminal_registry_write_result_bit"
        or eof.get("process_id") != expected_pid
        or eof.get("thread_id") != commit["thread_id"]
        or eof.get("framework_update") < effective_update
        or eof.get("start_index") <= commit["row_index"]
        or eof.get("end_index") <= eof.get("start_index")
        or eof.get("terminal_row_index") != eof.get("end_index")
        or eof.get("reentries")
        != eof.get("end_index") - eof.get("start_index") + 1
        or eof.get("terminal_payload_bit_position")
        != eof.get("terminal_row_start_bit_position") + 10
        or eof.get("terminal_row_end_bit_position")
        != eof.get("terminal_payload_bit_position") + 1
        or eof.get("last_read_bit_position")
        != eof.get("terminal_payload_bit_position")
        or eof.get("last_command_bit_position")
        != eof.get("terminal_row_start_bit_position") - 1
        or eof.get("last_command_order")
        + eof.get("command_order_offset")
        != eof.get("end_index")
        or eof.get("command_order_offset") != command_order_offset
        or eof.get("unconsumed_result_bits") != 1
        or eof.get("exit_code") != 0
    ):
        raise CollectionError("trace_result_invalid")

    barrier = barriers[0]
    barrier_keys = {
        "process_id",
        "thread_id",
        "framework_update",
        "barrier_row_index",
        "barrier_start_bit_position",
        "barrier_read_bit_position",
        "barrier_command_number",
        "discharged_row_count",
        "surplus_rows",
        "mechanism",
    }
    if (
        not isinstance(barrier, dict)
        or set(barrier) != barrier_keys
        or any(
            isinstance(barrier.get(name), bool)
            or not isinstance(barrier.get(name), int)
            for name in barrier_keys - {"surplus_rows", "mechanism"}
        )
        or barrier.get("process_id") != expected_pid
        or barrier.get("thread_id") != commit["thread_id"]
        or barrier.get("framework_update") != eof["framework_update"]
        or barrier.get("barrier_row_index") != eof["start_index"]
        or barrier.get("barrier_read_bit_position")
        != barrier.get("barrier_start_bit_position") + 10
        or barrier.get("barrier_command_number") != 12
        or barrier.get("discharged_row_count")
        != len(effective_debt_rows)
        or barrier.get("surplus_rows") != surplus_rows
        or barrier.get("mechanism")
        != "later_non_file_write_prepared_service_barrier"
    ):
        raise CollectionError("trace_result_invalid")
    return True


def _validate_source_bound_board_trace_receipt(
    payload: Mapping[str, Any],
    *,
    expected_pid: int,
    expected_anchor: Mapping[str, Any] | None,
    expected_allow_global_correction: bool,
) -> None:
    if not isinstance(expected_allow_global_correction, bool):
        raise CollectionError("trace_result_invalid")
    receipt = payload.get("source_bound_board_anchor")
    if expected_anchor is None:
        if receipt is not None or expected_allow_global_correction:
            raise CollectionError("trace_result_invalid")
        return
    if not isinstance(receipt, dict):
        raise CollectionError("trace_result_invalid")
    observations = receipt.get("observations")
    if (
        receipt.get("mechanism")
        != (
            "exact_post_global_call_board_constructor_seed_and_"
            "main_thread_crt_anchor"
        )
        or receipt.get("persistent_file_modified") is not False
        or not isinstance(observations, list)
        or len(observations) != 1
        or not isinstance(observations[0], dict)
    ):
        raise CollectionError("trace_result_invalid")
    row = observations[0]
    global_row = row.get("global")
    crt_row = row.get("thread_crt")
    global_source = (
        global_row.get("restored_pre_call_source")
        if isinstance(global_row, dict)
        else None
    )
    crt_source = (
        crt_row.get("restored_source")
        if isinstance(crt_row, dict)
        else None
    )
    correction_applied = (
        global_row.get("bounded_global_correction_applied")
        if isinstance(global_row, dict)
        else None
    )
    natural_anchor = correction_applied is False
    global_bytes_written = (
        global_row.get("bytes_written")
        if isinstance(global_row, dict)
        else None
    )
    live_before_index = (
        global_row.get("live_before_index")
        if isinstance(global_row, dict)
        else None
    )
    live_before_sha256 = (
        global_row.get("live_before_state_sha256")
        if isinstance(global_row, dict)
        else None
    )
    correction_index_delta = (
        global_row.get("correction_index_delta")
        if isinstance(global_row, dict)
        else None
    )
    if (
        row.get("classification")
        != (
            "source-bound-natural-retail-post-call-board-anchor"
            if natural_anchor
            else "source-bound-bounded-global-correction-board-anchor"
        )
        or row.get("process_id") != expected_pid
        or isinstance(row.get("thread_id"), bool)
        or not isinstance(row.get("thread_id"), int)
        or row["thread_id"] <= 0
        or row.get("framework_update")
        != expected_anchor["framework_update"]
        or row.get("board_seed_call_address")
        != DEFAULT_BOARD_RESEED_CALL
        or row.get("source_call_return_address")
        != expected_anchor["caller"]
        or row.get("instruction_bridge_hex") != "8d564c8bc88bc2"
        or row.get("instruction_bridge_contains_call") is not False
        or row.get("effective_seed")
        != expected_anchor["expected_output"]
        or row.get("observed_seed_matches_source") is not natural_anchor
        or row.get("identity_mechanism")
        != "board_rng_owner_vtable_and_source_call_state"
        or row.get("expected_main_thread_id") != row.get("thread_id")
        or isinstance(row.get("rng_owner_address"), bool)
        or not isinstance(row.get("rng_owner_address"), int)
        or row["rng_owner_address"] <= 0
        or row.get("rng_owner_vtable") != BOARD_SEED_OWNER_VTABLE
        or row.get("rng_object_address")
        != row["rng_owner_address"] + BOARD_MTRAND_OFFSET
        or isinstance(row.get("active_board_address"), bool)
        or not isinstance(row.get("active_board_address"), int)
        or not 0 <= row["active_board_address"] <= 0xFFFFFFFF
        or row.get("active_board_address", 0) <= 0
        or row.get("active_board_vtable") != BOARD_VTABLE
        or row.get("rng_owner_is_active_board")
        is not (row["active_board_address"] == row["rng_owner_address"])
        or row.get("writeback_verified") is not True
        or row.get("transactional_rollback_on_failure") is not True
        or row.get("persistent_file_modified") is not False
        or not isinstance(global_row, dict)
        or global_row.get("state_address") != GLOBAL_MTRAND_ADDRESS
        or not isinstance(correction_applied, bool)
        or correction_applied
        and not expected_allow_global_correction
        or global_row.get("global_correction_authorized")
        is not expected_allow_global_correction
        or global_row.get("natural_post_state_match") is not natural_anchor
        or global_row.get("live_post_draw_verified") is not True
        or global_row.get("same_state_words") is not True
        or global_row.get("changed") is not correction_applied
        or global_bytes_written
        != (MTRAND_STATE_BYTES if correction_applied else 0)
        or isinstance(live_before_index, bool)
        or not isinstance(live_before_index, int)
        or not 0 <= live_before_index <= MTRAND_STATE_WORDS
        or not isinstance(live_before_sha256, str)
        or len(live_before_sha256) != 71
        or not live_before_sha256.startswith("sha256:")
        or any(
            character not in "0123456789abcdef"
            for character in live_before_sha256[7:]
        )
        or global_row.get("maximum_correction_draw_delta")
        != SOURCE_BOUND_GLOBAL_CORRECTION_MAX_DRAWS
        or isinstance(correction_index_delta, bool)
        or not isinstance(correction_index_delta, int)
        or correction_index_delta
        != expected_anchor["expected_post_index"] - live_before_index
        or (
            natural_anchor
            and (
                row.get("observed_seed")
                != expected_anchor["expected_output"]
                or correction_index_delta != 0
                or live_before_sha256
                != expected_anchor["expected_post_state_sha256"]
            )
        )
        or (
            correction_applied
            and (
                row.get("observed_seed")
                == expected_anchor["expected_output"]
                or not 0
                < abs(correction_index_delta)
                <= SOURCE_BOUND_GLOBAL_CORRECTION_MAX_DRAWS
                or live_before_sha256
                == expected_anchor["expected_post_state_sha256"]
            )
        )
        or global_row.get("source_call_output")
        != expected_anchor["expected_output"]
        or global_row.get("source_call_post_index")
        != expected_anchor["expected_post_index"]
        or global_row.get("source_call_post_state_sha256")
        != expected_anchor["expected_post_state_sha256"]
        or not isinstance(global_source, dict)
        or global_source.get("source_path")
        != expected_anchor["monitor_path"]
        or global_source.get("source_sha256")
        != expected_anchor["monitor_sha256"]
        or global_source.get("source_trace_path")
        != expected_anchor["trace_path"]
        or global_source.get("source_trace_sha256")
        != expected_anchor["trace_sha256"]
        or global_source.get("source_recording_report_path")
        != expected_anchor["recording_report_path"]
        or global_source.get("source_recording_report_sha256")
        != expected_anchor["recording_report_sha256"]
        or global_source.get("source_dmo_path")
        != expected_anchor["source_dmo_path"]
        or global_source.get("source_dmo_sha256")
        != expected_anchor["source_dmo_sha256"]
        or global_source.get("source_call_order")
        != expected_anchor["source_order"]
        or global_source.get("source_caller")
        != expected_anchor["caller"]
        or not isinstance(crt_row, dict)
        or crt_row.get("restored_state")
        != expected_anchor["thread_crt_state"]
        or crt_row.get("bytes_written") not in (0, 4)
        or crt_row.get("changed")
        is not (crt_row.get("bytes_written") == 4)
        or crt_row.get("thread_id") != row.get("thread_id")
        or crt_row.get("ptd_thread_id") != row.get("thread_id")
        or not isinstance(crt_source, dict)
        or crt_source.get("source_path")
        != expected_anchor["trace_path"]
        or crt_source.get("source_sha256")
        != expected_anchor["trace_sha256"]
        or crt_source.get("source_recording_report_path")
        != expected_anchor["recording_report_path"]
        or crt_source.get("source_recording_report_sha256")
        != expected_anchor["recording_report_sha256"]
        or crt_source.get("source_dmo_path")
        != expected_anchor["source_dmo_path"]
        or crt_source.get("source_dmo_sha256")
        != expected_anchor["source_dmo_sha256"]
        or crt_source.get("source_call_order")
        != expected_anchor["source_order"]
        or crt_source.get("source_caller")
        != expected_anchor["caller"]
        or crt_source.get("source_framework_update")
        != expected_anchor["framework_update"]
        or crt_source.get("state")
        != expected_anchor["thread_crt_state"]
        or row.get("bytes_written")
        != global_bytes_written + crt_row.get("bytes_written")
        or row.get("process_memory_mutation")
        is not (
            global_bytes_written > 0
            or crt_row.get("bytes_written") == 4
        )
        or receipt.get("process_memory_mutation")
        is not (
            global_bytes_written > 0
            or crt_row.get("bytes_written") == 4
        )
    ):
        raise CollectionError("trace_result_invalid")


def _validate_source_bound_board_precall_trace_receipt(
    payload: Mapping[str, Any],
    *,
    expected_pid: int,
    expected_anchor: Mapping[str, Any] | None,
    expected_precall: Mapping[str, Any] | None,
) -> None:
    """Bind the atomic seed-producing CALL restore to the natural source."""

    options = payload.get("options")
    source_receipt = payload.get("source_bound_board_anchor")
    source_rows = (
        source_receipt.get("observations")
        if isinstance(source_receipt, dict)
        else None
    )
    source_row = (
        source_rows[0]
        if isinstance(source_rows, list)
        and len(source_rows) == 1
        and isinstance(source_rows[0], dict)
        else None
    )
    receipt = (
        source_row.get("global_precall_restore")
        if isinstance(source_row, dict)
        else None
    )
    if expected_precall is None:
        if (
            receipt is not None
            or payload.get("global_mtrand_restore") is not None
            or isinstance(options, dict)
            and options.get(
                "source_bound_board_precall_global_restore",
                False,
            )
            is not False
        ):
            raise CollectionError("trace_result_invalid")
        return
    if (
        expected_anchor is None
        or not isinstance(options, dict)
        or options.get("source_bound_board_precall_global_restore")
        is not True
        or payload.get("global_mtrand_restore") is not None
        or not isinstance(receipt, dict)
    ):
        raise CollectionError("trace_result_invalid")
    try:
        expected_restored = {
            "source_path": expected_anchor["monitor_path"],
            "source_sha256": expected_anchor["monitor_sha256"],
            "source_process_id": expected_precall["source_process_id"],
            "source_framework_update": expected_anchor[
                "monitor_framework_update"
            ],
            "source_native_game_time": expected_precall[
                "source_native_game_time"
            ],
            **dict(expected_precall["global_state"]),
            "source_kind": "natural_retail_hardware_trace_and_monitor",
            "source_trace_path": expected_anchor["trace_path"],
            "source_trace_sha256": expected_anchor["trace_sha256"],
            "source_recording_report_path": expected_anchor[
                "recording_report_path"
            ],
            "source_recording_report_sha256": expected_anchor[
                "recording_report_sha256"
            ],
            "source_dmo_path": expected_anchor["source_dmo_path"],
            "source_dmo_sha256": expected_anchor["source_dmo_sha256"],
            "source_call_order": expected_anchor["source_order"],
            "source_caller": expected_anchor["caller"],
            "source_caller_hex": f"0x{expected_anchor['caller']:08x}",
        }
        expected_framework_update = expected_precall["framework_update"]
        expected_dynamic_before = expected_precall[
            "allow_dynamic_before"
        ]
    except (KeyError, TypeError, ValueError):
        raise CollectionError("trace_result_invalid") from None
    expected_row_keys = {
        "mechanism",
        "process_id",
        "thread_id",
        "perf_counter_ns",
        "framework_update",
        "state_address",
        "expected_before",
        "live_before_index",
        "live_before_state_sha256",
        "dynamic_before_allowed",
        "restored",
        "changed",
        "bytes_written",
        "transactional_rollback",
        "writeback_verified",
        "persistent_file_modified",
        "call_address",
        "return_address",
        "target_address",
        "instruction_hex",
        "expected_main_thread_id",
        "main_thread_match",
        "active_board_address",
        "active_board_vtable",
        "process_memory_mutation",
        "classification",
        "other_threads_suspended_count",
        "other_threads_suspended_until_board_call",
        "atomic_window_completed",
        "board_call_reached",
        "board_call_address",
        "board_call_framework_update",
        "board_call_thread_id",
        "other_threads_resumed_after_board_call",
        "atomic_window_finished_perf_counter_ns",
    }
    live_sha256 = receipt.get("live_before_state_sha256")
    live_reference_match = (
        live_sha256 == expected_restored.get("state_sha256")
    )
    expected_before = {
        **expected_restored,
        "live_reference_match": live_reference_match,
    }
    changed = not live_reference_match
    if (
        set(receipt) != expected_row_keys
        or receipt.get("mechanism")
        != "exact_source_bound_global_rng_precall_state_restore"
        or receipt.get("process_id") != expected_pid
        or isinstance(receipt.get("thread_id"), bool)
        or not isinstance(receipt.get("thread_id"), int)
        or receipt["thread_id"] <= 0
        or isinstance(receipt.get("perf_counter_ns"), bool)
        or not isinstance(receipt.get("perf_counter_ns"), int)
        or receipt["perf_counter_ns"] <= 0
        or receipt.get("framework_update") != expected_framework_update
        or receipt.get("state_address") != GLOBAL_MTRAND_ADDRESS
        or receipt.get("expected_before") != expected_before
        or isinstance(receipt.get("live_before_index"), bool)
        or not isinstance(receipt.get("live_before_index"), int)
        or not 0 <= receipt["live_before_index"] <= MTRAND_STATE_WORDS
        or not isinstance(live_sha256, str)
        or len(live_sha256) != 71
        or not live_sha256.startswith("sha256:")
        or any(
            character not in "0123456789abcdef"
            for character in live_sha256[7:]
        )
        or receipt.get("dynamic_before_allowed")
        is not expected_dynamic_before
        or receipt.get("restored") != expected_restored
        or receipt.get("changed") is not changed
        or receipt.get("bytes_written")
        != (MTRAND_STATE_BYTES if changed else 0)
        or receipt.get("transactional_rollback") is not True
        or receipt.get("writeback_verified") is not True
        or receipt.get("persistent_file_modified") is not False
        or receipt.get("call_address")
        != expected_precall["call_address"]
        or receipt.get("return_address")
        != expected_precall["return_address"]
        or receipt.get("target_address")
        != expected_precall["target_address"]
        or receipt.get("instruction_hex")
        != expected_precall["instruction_hex"]
        or receipt.get("expected_main_thread_id")
        != receipt.get("thread_id")
        or receipt.get("main_thread_match") is not True
        or isinstance(receipt.get("active_board_address"), bool)
        or not isinstance(receipt.get("active_board_address"), int)
        or not 0 < receipt["active_board_address"] <= 0xFFFFFFFF
        or receipt.get("active_board_vtable") != BOARD_VTABLE
        or receipt.get("process_memory_mutation") is not changed
        or receipt.get("classification")
        != "source-bound-natural-retail-global-precall-restore"
        or isinstance(
            receipt.get("other_threads_suspended_count"), bool
        )
        or not isinstance(
            receipt.get("other_threads_suspended_count"), int
        )
        or receipt["other_threads_suspended_count"] < 0
        or receipt.get("other_threads_suspended_until_board_call")
        is not expected_precall["suspend_other_threads_until_board_call"]
        or receipt.get("atomic_window_completed") is not True
        or receipt.get("board_call_reached") is not True
        or receipt.get("board_call_address") != DEFAULT_BOARD_RESEED_CALL
        or receipt.get("board_call_framework_update")
        != expected_framework_update
        or receipt.get("board_call_thread_id")
        != receipt.get("thread_id")
        or receipt.get("other_threads_resumed_after_board_call")
        is not True
        or isinstance(
            receipt.get("atomic_window_finished_perf_counter_ns"), bool
        )
        or not isinstance(
            receipt.get("atomic_window_finished_perf_counter_ns"), int
        )
        or receipt["atomic_window_finished_perf_counter_ns"]
        <= receipt["perf_counter_ns"]
    ):
        raise CollectionError("trace_result_invalid")


def _validate_post_blackout_overdue_idle_reentry_receipts(
    result: Mapping[str, Any],
    *,
    expected_pid: int,
    enabled: bool,
    candidates: list[Mapping[str, Any]],
) -> None:
    """Bind every overdue idle acceptance to one frozen DMO candidate."""

    receipts = result.get("service_exit_overdue_idle_reentries", [])
    if (
        not isinstance(receipts, list)
        or not isinstance(candidates, list)
        or bool(candidates) is not enabled
        or (receipts and not enabled)
    ):
        raise CollectionError("trace_result_invalid")

    candidate_keys = {
        "corridor_start_index",
        "corridor_end_index",
        "corridor_row_count",
        "corridor_recorded_update",
        "bridge_row_index",
        "bridge_recorded_update",
        "bridge_kind",
        "bridge_command_number",
        "bridge_short_form",
        "following_row_index",
        "following_recorded_update",
        "following_kind",
        "following_command_number",
        "following_short_form",
    }
    receipt_keys = candidate_keys | {
        "process_id",
        "thread_id",
        "framework_update",
        "last_demo_update",
        "corridor_end_bit_position",
        "bridge_payload_empty",
        "bridge_start_bit_position",
        "bridge_end_bit_position",
        "following_payload_empty",
        "following_start_bit_position",
        "following_end_bit_position",
        "bridge_late_updates",
        "following_remaining_updates",
        "command_order_offset",
        "observed_command_order",
        "reentries",
        "prepared_reentry_verified",
        "prepared_framework_update",
        "prepared_last_demo_update",
        "prepared_needs_command",
        "prepared_command_order",
        "prepared_command_bit_position",
        "prepared_buffer_read_bit_position",
        "memory_write_bytes",
        "process_memory_mutation",
        "verification_mode",
    }
    boolean_fields = {
        "bridge_short_form",
        "bridge_payload_empty",
        "following_short_form",
        "following_payload_empty",
        "prepared_reentry_verified",
        "process_memory_mutation",
    }
    string_fields = {
        "bridge_kind",
        "following_kind",
        "verification_mode",
    }
    integer_fields = receipt_keys - boolean_fields - string_fields
    frozen_candidates = [dict(candidate) for candidate in candidates]
    if any(
        set(candidate) != candidate_keys
        for candidate in frozen_candidates
    ):
        raise CollectionError("trace_result_invalid")

    seen_candidates: set[int] = set()
    for receipt in receipts:
        if (
            not isinstance(receipt, dict)
            or set(receipt) != receipt_keys
            or any(
                isinstance(receipt.get(key), bool)
                or not isinstance(receipt.get(key), int)
                for key in integer_fields
            )
            or any(
                not isinstance(receipt.get(key), bool)
                for key in boolean_fields
            )
            or any(
                not isinstance(receipt.get(key), str)
                for key in string_fields
            )
        ):
            raise CollectionError("trace_result_invalid")
        frozen_shape = {key: receipt[key] for key in candidate_keys}
        try:
            candidate_index = frozen_candidates.index(frozen_shape)
        except ValueError:
            raise CollectionError("trace_result_invalid") from None
        if candidate_index in seen_candidates:
            raise CollectionError("trace_result_invalid")
        seen_candidates.add(candidate_index)

        corridor_start = receipt["corridor_start_index"]
        corridor_end = receipt["corridor_end_index"]
        corridor_update = receipt["corridor_recorded_update"]
        bridge_index = receipt["bridge_row_index"]
        bridge_update = receipt["bridge_recorded_update"]
        following_index = receipt["following_row_index"]
        following_update = receipt["following_recorded_update"]
        framework_update = receipt["framework_update"]
        bridge_lateness = receipt["bridge_late_updates"]
        following_remaining = receipt["following_remaining_updates"]
        if (
            receipt["process_id"] != expected_pid
            or receipt["thread_id"] <= 0
            or framework_update != receipt["last_demo_update"]
            or corridor_start < 0
            or corridor_end < corridor_start
            or receipt["corridor_row_count"]
            != corridor_end - corridor_start + 1
            or receipt["corridor_row_count"]
            > POST_BLACKOUT_FILE_WRITE_SPILLOVER_MAX_ROWS
            or bridge_index != corridor_end + 1
            or following_index != bridge_index + 1
            or bridge_update - corridor_update
            != SERVICE_EXIT_TIMELINE_MAX_GAP_UPDATES
            or following_update - bridge_update
            != SERVICE_EXIT_TIMELINE_MAX_GAP_UPDATES
            or not bridge_update < framework_update < following_update
            or bridge_lateness != framework_update - bridge_update
            or following_remaining != following_update - framework_update
            or not 1
            <= bridge_lateness
            < SERVICE_EXIT_TIMELINE_MAX_GAP_UPDATES
            or not 1
            <= following_remaining
            < SERVICE_EXIT_TIMELINE_MAX_GAP_UPDATES
            or bridge_lateness + following_remaining
            != SERVICE_EXIT_TIMELINE_MAX_GAP_UPDATES
            or receipt["bridge_kind"] != "idle"
            or receipt["bridge_command_number"] != 31
            or receipt["bridge_short_form"] is not False
            or receipt["bridge_payload_empty"] is not True
            or receipt["following_kind"] != "idle"
            or receipt["following_command_number"] != 31
            or receipt["following_short_form"] is not False
            or receipt["following_payload_empty"] is not True
            or receipt["corridor_end_bit_position"]
            != receipt["bridge_start_bit_position"]
            or receipt["bridge_end_bit_position"]
            - receipt["bridge_start_bit_position"]
            != 10
            or receipt["following_start_bit_position"]
            != receipt["bridge_end_bit_position"]
            or receipt["following_end_bit_position"]
            - receipt["following_start_bit_position"]
            != 10
            or receipt["command_order_offset"] < 0
            or receipt["observed_command_order"]
            != bridge_index - receipt["command_order_offset"]
            or receipt["reentries"] <= 0
            or receipt["prepared_reentry_verified"] is not True
            or receipt["prepared_framework_update"]
            != receipt["framework_update"]
            or receipt["prepared_last_demo_update"]
            != receipt["last_demo_update"]
            or receipt["prepared_needs_command"] != 1
            or receipt["prepared_command_order"]
            != receipt["observed_command_order"]
            or receipt["prepared_command_bit_position"]
            != receipt["bridge_start_bit_position"]
            or receipt["prepared_buffer_read_bit_position"]
            != receipt["bridge_end_bit_position"]
            or receipt["memory_write_bytes"] != 0
            or receipt["process_memory_mutation"] is not False
            or receipt["verification_mode"]
            != (
                "post_blackout_successful_file_write_"
                "overdue_idle_reentry"
            )
        ):
            raise CollectionError("trace_result_invalid")


def _validate_trace_result(
    path: Path,
    *,
    expected_pid: int,
    expected_dmo_sha256: str,
    expected_attach_at_update: int | None = None,
    expected_detach_at_update: int | None = None,
    expected_reattach_at_update: int | None = None,
    expected_allow_pre_stream_commands: bool | None = None,
    expected_startup_trace_handoff: bool | None = None,
    expected_allow_attach_stabilization: bool | None = None,
    expected_allow_pre_attach_file_write_debt: bool | None = None,
    expected_allow_font_cache_manifest_completion_debt: (
        bool | None
    ) = None,
    expected_font_cache_manifest_receipt: (
        Mapping[str, Any] | None
    ) = None,
    expected_seed_board_before_attach: bool | None = None,
    expected_startup_priority_bias_until_update: int | None = None,
    expected_startup_process_affinity_mask: int | None = None,
    expected_crt_rand_seed: int | None = None,
    expected_startup_seed_transport: str | None = None,
    expected_board_seed_address: int | None = None,
    expected_board_seed: int | None = None,
    expected_global_rng_seed: int | None = None,
    expected_thread_crt_rng_seed: int | None = None,
    expected_source_bound_board_anchor: (
        Mapping[str, Any] | None
    ) = None,
    expected_source_bound_board_precall_global_restore: (
        Mapping[str, Any] | None
    ) = None,
    expected_allow_source_bound_board_global_correction: (
        bool | None
    ) = None,
    expected_post_blackout_attach_stabilization: bool | None = None,
    expected_blackout_file_write_order_rebase: bool | None = None,
    expected_blackout_file_write_order_rebase_rows: (
        list[Mapping[str, Any]] | None
    ) = None,
    expected_blackout_file_write_rebase_mode: str | None = None,
    expected_blackout_command_order_offset: int | None = None,
    expected_blackout_native_timeline_offset: int | None = None,
    expected_blackout_allowed_offset_pairs: (
        list[Mapping[str, Any]] | None
    ) = None,
    expected_allow_post_blackout_overdue_idle_reentry: (
        bool | None
    ) = None,
    expected_post_blackout_overdue_idle_reentry_candidates: (
        list[Mapping[str, Any]] | None
    ) = None,
    expected_close_after_terminal_command: bool | None = None,
    expected_terminal_close_command: Mapping[str, Any] | None = None,
    expected_diagnostic_successful_file_write_padding_rows: (
        list[int] | None
    ) = None,
    expected_stop_after_update: int | None = None,
) -> Mapping[str, Any]:
    payload = _read_json_object(path, "trace_result_invalid")
    result = payload.get("result")
    source_dmo = payload.get("source_dmo")
    stopped_trace = expected_stop_after_update is not None
    termination_valid = isinstance(result, dict) and (
        (
            not stopped_trace
            and result.get("exited") is True
            and result.get("exit_code") == 0
        )
        or (
            stopped_trace
            and isinstance(expected_stop_after_update, int)
            and not isinstance(expected_stop_after_update, bool)
            and expected_stop_after_update >= 0
            and result.get("exited") is False
            and result.get("exit_code") is None
            and result.get("stopped_at_update") is True
            and result.get("stopped_at_command_order") is False
            and result.get("last_update") == expected_stop_after_update
        )
    )
    options = payload.get("options")
    if (
        payload.get("schema") != "zuma.popcap_strict_replay.v3"
        or payload.get("runtime_process_id") != expected_pid
        or not isinstance(result, dict)
        or not isinstance(source_dmo, dict)
        or f"sha256:{source_dmo.get('sha256')}" != expected_dmo_sha256
        or not termination_valid
        or (
            stopped_trace
            and (
                not isinstance(options, dict)
                or options.get("stop_after_update")
                != expected_stop_after_update
                or options.get("stop_after_command_order") is not None
            )
        )
        or result.get("failure_count") != 0
        or result.get("boundary_failures") != []
        or result.get("broker_failures") != []
    ):
        raise CollectionError("trace_result_invalid")
    if expected_allow_source_bound_board_global_correction is not None:
        options = payload.get("options")
        if (
            not isinstance(options, dict)
            or options.get(
                "allow_source_bound_board_global_correction",
                False,
            )
            is not expected_allow_source_bound_board_global_correction
            or (
                expected_allow_source_bound_board_global_correction
                and expected_source_bound_board_anchor is None
            )
        ):
            raise CollectionError("trace_result_invalid")
    diagnostic_dynamic_service_rebase = False
    if expected_diagnostic_successful_file_write_padding_rows is not None:
        options = payload.get("options")
        padding_rows = result.get(
            "diagnostic_successful_file_write_padding_rows"
        )
        service_rows = result.get(
            "service_consumed_file_write_debt_rows"
        )
        surplus_rows = result.get(
            "service_consumed_file_write_surplus_rows"
        )
        effective_debt_rows = result.get(
            "effective_deferred_file_write_debt_rows"
        )
        debt_barriers = result.get(
            "successful_file_write_debt_barriers"
        )
        if (
            not isinstance(options, dict)
            or options.get(
                "diagnostic_successful_file_write_padding_rows"
            )
            != expected_diagnostic_successful_file_write_padding_rows
            or padding_rows
            != expected_diagnostic_successful_file_write_padding_rows
            or not isinstance(service_rows, list)
            or any(
                isinstance(row_index, bool)
                or not isinstance(row_index, int)
                or row_index < 0
                for row_index in service_rows
            )
            or service_rows != sorted(set(service_rows))
            or set(service_rows) & set(padding_rows)
            or not isinstance(surplus_rows, list)
            or surplus_rows != sorted(set(surplus_rows))
            or not isinstance(effective_debt_rows, list)
            or effective_debt_rows != sorted(set(effective_debt_rows))
            or not isinstance(debt_barriers, list)
        ):
            raise CollectionError("trace_result_invalid")
        if padding_rows:
            accounted_rows = result.get("command_order_rebase_rows")
            pre_attach_rows = result.get(
                "pre_attach_file_write_debt_rows"
            )
            discharges = result.get("deferred_file_write_discharges")
            if (
                not isinstance(accounted_rows, list)
                or not set(padding_rows).issubset(accounted_rows)
                or not isinstance(pre_attach_rows, list)
                or not isinstance(discharges, list)
                or any(not isinstance(receipt, dict) for receipt in discharges)
            ):
                raise CollectionError("trace_result_invalid")
            observed_debt_rows = [
                receipt.get("debt_row_index") for receipt in discharges
            ]
            if (
                observed_debt_rows != effective_debt_rows
                or not set(surplus_rows).issubset(
                    set(accounted_rows) | set(service_rows)
                )
                or set(surplus_rows) & set(padding_rows)
                or bool(surplus_rows) != bool(debt_barriers)
                or (
                    surplus_rows
                    and (
                        len(debt_barriers) != 1
                        or not isinstance(debt_barriers[0], dict)
                        or debt_barriers[0].get("surplus_rows")
                        != surplus_rows
                        or debt_barriers[0].get("mechanism")
                        != (
                            "later_non_file_write_prepared_service_barrier"
                        )
                    )
                )
            ):
                raise CollectionError("trace_result_invalid")
            diagnostic_dynamic_service_rebase = (
                _validate_diagnostic_successful_file_write_receipts(
                    result,
                    expected_pid=expected_pid,
                    padding_rows=padding_rows,
                )
            )
    terminal_close_requests = result.get("terminal_close_requests", [])
    terminal_close_expected = (
        expected_close_after_terminal_command is True
    )
    if expected_close_after_terminal_command is not None:
        options = payload.get("options")
        if (
            not isinstance(options, dict)
            or options.get("close_after_terminal_command", False)
            is not terminal_close_expected
            or not isinstance(terminal_close_requests, list)
            or bool(terminal_close_requests) is not terminal_close_expected
            or terminal_close_expected
            != (expected_terminal_close_command is not None)
        ):
            raise CollectionError("trace_result_invalid")
    if expected_close_after_terminal_command is True:
        assert expected_terminal_close_command is not None
        if (
            len(terminal_close_requests) != 1
            or result.get("stopped_at_command_order") is not True
        ):
            raise CollectionError("trace_result_invalid")
        receipt = terminal_close_requests[0]
        if not isinstance(receipt, dict):
            raise CollectionError("trace_result_invalid")
        integer_fields = (
            "thread_id",
            "target_start_bit_position",
            "target_end_bit_position",
            "observed_command_order",
            "observed_command_order_offset",
            "observed_command_bit_position",
            "observed_read_bit_position",
            "main_window_handle",
            "requested_perf_counter_ns",
        )
        if (
            receipt.get("process_id") != expected_pid
            or receipt.get("mechanism")
            != "wm_close_after_exact_terminal_command"
            or receipt.get("target_command_order")
            != expected_terminal_close_command.get("command_order")
            or receipt.get("target_update")
            != expected_terminal_close_command.get("update")
            or receipt.get("target_kind")
            != expected_terminal_close_command.get("kind")
            or receipt.get("target_command_number")
            != expected_terminal_close_command.get("command_number")
            or receipt.get("target_short_form")
            is not expected_terminal_close_command.get("short_form")
            or receipt.get("post_message_succeeded") is not True
            or any(
                isinstance(receipt.get(name), bool)
                or not isinstance(receipt.get(name), int)
                or receipt[name] < 0
                for name in integer_fields
            )
            or receipt["thread_id"] <= 0
            or receipt["main_window_handle"] <= 0
            or receipt["requested_perf_counter_ns"] <= 0
            or receipt["target_end_bit_position"]
            - receipt["target_start_bit_position"]
            != 10
            or receipt["observed_command_order"]
            + receipt["observed_command_order_offset"]
            != receipt["target_command_order"]
            or receipt["observed_update"] != receipt["target_update"]
            or receipt["observed_command_bit_position"]
            != receipt["target_start_bit_position"]
            or receipt["observed_read_bit_position"]
            != receipt["target_end_bit_position"]
        ):
            raise CollectionError("trace_result_invalid")
    _validate_main_file_read_corridor_trace_receipts(
        result,
        expected_pid=expected_pid,
    )
    for name in (
        "trace_started_perf_counter_ns",
        "trace_finished_perf_counter_ns",
    ):
        value = payload.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise CollectionError("trace_result_invalid")
    if (
        payload["trace_finished_perf_counter_ns"]
        <= payload["trace_started_perf_counter_ns"]
    ):
        raise CollectionError("trace_result_invalid")
    font_cache_expectations = (
        expected_allow_pre_attach_file_write_debt,
        expected_allow_font_cache_manifest_completion_debt,
    )
    if any(value is not None for value in font_cache_expectations) or (
        expected_font_cache_manifest_receipt is not None
    ):
        if (
            any(value is None for value in font_cache_expectations)
            or expected_startup_trace_handoff is None
        ):
            raise CollectionError("trace_result_invalid")
        font_cache_result = result
        trace_phases = payload.get("trace_phases")
        if (
            isinstance(trace_phases, list)
            and len(trace_phases) == 2
            and isinstance(trace_phases[0], dict)
            and trace_phases[0].get("name") == "pre_capture"
            and isinstance(trace_phases[0].get("result"), dict)
            and isinstance(trace_phases[1], dict)
            and trace_phases[1].get("name") == "post_capture"
        ):
            font_cache_result = trace_phases[0]["result"]
        _validate_font_cache_trace_receipt(
            payload,
            font_cache_result,
            expected_startup_handoff=(
                expected_startup_trace_handoff
            ),
            expected_allow_pre_attach_debt=(
                expected_allow_pre_attach_file_write_debt
            ),
            expected_allow_manifest_completion=(
                expected_allow_font_cache_manifest_completion_debt
            ),
            expected_manifest_receipt=(
                expected_font_cache_manifest_receipt
            ),
        )
    if expected_allow_pre_stream_commands is not None:
        options = payload.get("options")
        if (
            not isinstance(options, dict)
            or options.get("allow_pre_stream_commands")
            is not expected_allow_pre_stream_commands
        ):
            raise CollectionError("trace_result_invalid")
    if expected_startup_trace_handoff is not None:
        options = payload.get("options")
        startup_rng = payload.get("startup_rng")
        observed_enabled = (
            options.get("startup_trace_handoff", False)
            if isinstance(options, dict)
            else None
        )
        if observed_enabled is not expected_startup_trace_handoff:
            raise CollectionError("trace_result_invalid")
        if expected_startup_trace_handoff:
            main_thread_id = (
                startup_rng.get("thread_id")
                if isinstance(startup_rng, dict)
                else None
            )
            natural_seed_handoff = (
                expected_startup_seed_transport is None
                and isinstance(options, dict)
                and options.get("direct_natural_seed") is True
                and isinstance(startup_rng, dict)
                and startup_rng.get("mode")
                == "retail_natural_seed_observation"
            )
            if (
                expected_attach_at_update != 0
                or expected_allow_pre_stream_commands is not True
                or not (
                    expected_startup_seed_transport == "debugger_register"
                    or natural_seed_handoff
                )
                or not isinstance(startup_rng, dict)
                or startup_rng.get("main_thread_suspended_on_detach")
                is not True
                or startup_rng.get("main_thread_suspend_previous_count")
                != 0
                or isinstance(main_thread_id, bool)
                or not isinstance(main_thread_id, int)
                or main_thread_id <= 0
                or result.get("startup_handoff_main_thread_id")
                != main_thread_id
                or result.get(
                    "startup_handoff_launcher_suspend_previous_count"
                )
                != 0
                or result.get(
                    "startup_handoff_tracer_resume_previous_count"
                )
                not in (1, 2)
                or result.get(
                    "startup_handoff_resumed_after_breakpoints"
                )
                is not True
            ):
                raise CollectionError("trace_result_invalid")
        elif (
            (
                isinstance(startup_rng, dict)
                and (
                    startup_rng.get(
                        "main_thread_suspended_on_detach",
                        False,
                    )
                    is not False
                    or startup_rng.get(
                        "main_thread_suspend_previous_count"
                    )
                    is not None
                )
            )
            or result.get("startup_handoff_main_thread_id") is not None
            or result.get(
                "startup_handoff_launcher_suspend_previous_count"
            )
            is not None
            or result.get(
                "startup_handoff_tracer_resume_previous_count"
            )
            is not None
            or result.get(
                "startup_handoff_resumed_after_breakpoints",
                False,
            )
            is not False
        ):
            raise CollectionError("trace_result_invalid")
    options = payload.get("options")
    startup_rng = payload.get("startup_rng")
    process_affinity = (
        startup_rng.get("process_affinity")
        if isinstance(startup_rng, dict)
        else None
    )
    affinity_option_declared = (
        isinstance(options, dict)
        and "startup_process_affinity_mask" in options
    )
    if (
        expected_startup_process_affinity_mask is not None
        or affinity_option_declared
        or process_affinity is not None
    ):
        if (
            not isinstance(options, dict)
            or options.get("startup_process_affinity_mask")
            != expected_startup_process_affinity_mask
        ):
            raise CollectionError("trace_result_invalid")
    if expected_startup_process_affinity_mask is None:
        if process_affinity is not None:
            raise CollectionError("trace_result_invalid")
    else:
        requested_mask = expected_startup_process_affinity_mask
        previous_mask = (
            process_affinity.get("previous_process_mask")
            if isinstance(process_affinity, dict)
            else None
        )
        system_mask = (
            process_affinity.get("system_mask")
            if isinstance(process_affinity, dict)
            else None
        )
        handoff_system_mask = (
            process_affinity.get("handoff_system_mask")
            if isinstance(process_affinity, dict)
            else None
        )
        handoff_ns = (
            process_affinity.get(
                "handoff_verification_perf_counter_ns"
            )
            if isinstance(process_affinity, dict)
            else None
        )
        debugger_detached_ns = (
            startup_rng.get("debugger_detached_perf_counter_ns")
            if isinstance(startup_rng, dict)
            else None
        )
        if (
            isinstance(requested_mask, bool)
            or not isinstance(requested_mask, int)
            or requested_mask <= 0
            or requested_mask > ctypes.c_size_t(-1).value
            or not isinstance(options, dict)
            or options.get("direct_runtime") is not True
            or options.get("startup_trace_handoff") is not True
            or not isinstance(startup_rng, dict)
            or startup_rng.get("process_id") != expected_pid
            or not isinstance(process_affinity, dict)
            or process_affinity.get("application_stage")
            != "created_suspended_before_first_resume"
            or process_affinity.get("requested_process_mask")
            != requested_mask
            or process_affinity.get("observed_process_mask")
            != requested_mask
            or isinstance(previous_mask, bool)
            or not isinstance(previous_mask, int)
            or previous_mask <= 0
            or isinstance(system_mask, bool)
            or not isinstance(system_mask, int)
            or system_mask <= 0
            or requested_mask & ~system_mask
            or process_affinity.get("single_logical_processor")
            is not (requested_mask.bit_count() == 1)
            or process_affinity.get("persistent_host_modification")
            is not False
            or process_affinity.get("handoff_observed_process_mask")
            != requested_mask
            or handoff_system_mask != system_mask
            or process_affinity.get("handoff_verification_stage")
            != "after_debugger_detach_before_tracer_handoff"
            or process_affinity.get("handoff_verified") is not True
            or isinstance(debugger_detached_ns, bool)
            or not isinstance(debugger_detached_ns, int)
            or debugger_detached_ns <= 0
            or isinstance(handoff_ns, bool)
            or not isinstance(handoff_ns, int)
            or not (
                debugger_detached_ns
                < handoff_ns
                < payload["trace_finished_perf_counter_ns"]
            )
        ):
            raise CollectionError("trace_result_invalid")
    if expected_allow_attach_stabilization is not None:
        options = payload.get("options")
        observed_enabled = (
            options.get("allow_attach_stabilization", False)
            if isinstance(options, dict)
            else None
        )
        if observed_enabled is not expected_allow_attach_stabilization:
            raise CollectionError("trace_result_invalid")
        if expected_allow_attach_stabilization:
            observations = result.get(
                "attach_stabilization_observations"
            )
            skipped_hits = result.get(
                "attach_stabilization_skipped_hits"
            )
            verified_update = result.get(
                "attach_stabilization_verified_update"
            )
            verified_order = result.get(
                "attach_stabilization_verified_command_order"
            )
            if (
                expected_attach_at_update is None
                or expected_detach_at_update is None
                or not isinstance(options, dict)
                or options.get(
                    "attach_stabilization_max_skipped_hits"
                )
                != ATTACH_STABILIZATION_MAX_SKIPPED_HITS
                or options.get(
                    "attach_stabilization_max_update_delta"
                )
                != ATTACH_STABILIZATION_MAX_UPDATE_DELTA
                or result.get("attach_stabilization_verified") is not True
                or isinstance(skipped_hits, bool)
                or not isinstance(skipped_hits, int)
                or not (
                    0
                    <= skipped_hits
                    < ATTACH_STABILIZATION_MAX_SKIPPED_HITS
                )
                or result.get("attach_stabilization_start_update")
                != expected_attach_at_update
                or isinstance(verified_update, bool)
                or not isinstance(verified_update, int)
                or not (
                    expected_attach_at_update
                    <= verified_update
                    < expected_detach_at_update
                )
                or verified_update - expected_attach_at_update
                > ATTACH_STABILIZATION_MAX_UPDATE_DELTA
                or isinstance(verified_order, bool)
                or not isinstance(verified_order, int)
                or verified_order < 0
                or not isinstance(observations, list)
                or len(observations) != skipped_hits + 1
                or not all(
                    isinstance(row, dict)
                    for row in observations
                )
                or any(
                    row.get("accepted") is not False
                    for row in observations[:-1]
                )
                or observations[-1].get("accepted") is not True
                or observations[-1].get("reason")
                != "exact_complete_main_boundary"
                or observations[-1].get("framework_update")
                != verified_update
                or observations[-1].get("command_order")
                != verified_order
                or observations[-1].get(
                    "candidate_command_order_offset"
                )
                != late_result.get("command_order_offset")
            ):
                raise CollectionError("trace_result_invalid")
        elif (
            expected_post_blackout_attach_stabilization is not True
            and (
                result.get("attach_stabilization_verified", False)
                is not False
                or result.get("attach_stabilization_skipped_hits", 0) != 0
            )
        ):
            raise CollectionError("trace_result_invalid")
    if expected_seed_board_before_attach is not None:
        options = payload.get("options")
        board_phase = payload.get("board_seed_phase")
        if (
            not isinstance(options, dict)
            or options.get("seed_board_before_attach")
            is not expected_seed_board_before_attach
        ):
            raise CollectionError("trace_result_invalid")
        if expected_seed_board_before_attach:
            board_phase_result = (
                board_phase.get("result")
                if isinstance(board_phase, dict)
                else None
            )
            if (
                not isinstance(board_phase, dict)
                or board_phase.get("name")
                != "pre_attach_board_seed"
                or not isinstance(board_phase_result, dict)
                or board_phase_result.get("failure_count") != 0
                or board_phase_result.get("exited") is not False
            ):
                raise CollectionError("trace_result_invalid")
            phase_times = (
                board_phase.get("started_perf_counter_ns"),
                board_phase.get("finished_perf_counter_ns"),
            )
            if (
                any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value <= 0
                    for value in phase_times
                )
                or not (
                    payload["trace_started_perf_counter_ns"]
                    < phase_times[0]
                    < phase_times[1]
                    < payload["trace_finished_perf_counter_ns"]
                )
            ):
                raise CollectionError("trace_result_invalid")
        elif board_phase is not None:
            raise CollectionError("trace_result_invalid")
    options = payload.get("options")
    startup_priority_phase = payload.get("startup_priority_phase")
    startup_priority_option_declared = (
        isinstance(options, dict)
        and "startup_priority_bias_until_update" in options
    )
    if (
        expected_startup_priority_bias_until_update is not None
        or startup_priority_option_declared
        or startup_priority_phase is not None
    ):
        if (
            not isinstance(options, dict)
            or options.get("startup_priority_bias_until_update")
            != expected_startup_priority_bias_until_update
        ):
            raise CollectionError("trace_result_invalid")
    if expected_startup_priority_bias_until_update is None:
        if startup_priority_phase is not None:
            raise CollectionError("trace_result_invalid")
    else:
        threads = (
            startup_priority_phase.get("threads")
            if isinstance(startup_priority_phase, dict)
            else None
        )
        if (
            not isinstance(startup_priority_phase, dict)
            or startup_priority_phase.get("name")
            != "startup_service_priority_bias"
            or startup_priority_phase.get("mechanism")
            != "transient_main_thread_priority_bias"
            or startup_priority_phase.get("process_id") != expected_pid
            or isinstance(
                startup_priority_phase.get("main_thread_id"),
                bool,
            )
            or not isinstance(
                startup_priority_phase.get("main_thread_id"),
                int,
            )
            or startup_priority_phase["main_thread_id"] <= 0
            or startup_priority_phase.get("requested_until_update")
            != expected_startup_priority_bias_until_update
            or startup_priority_phase.get(
                "persistent_process_modification"
            )
            is not False
            or not isinstance(threads, list)
            or not threads
        ):
            raise CollectionError("trace_result_invalid")
        first_applied = startup_priority_phase.get(
            "first_applied_update"
        )
        restored_at = startup_priority_phase.get(
            "restored_at_update"
        )
        if (
            isinstance(first_applied, bool)
            or not isinstance(first_applied, int)
            or first_applied < 0
            or first_applied
            >= expected_startup_priority_bias_until_update
            or isinstance(restored_at, bool)
            or not isinstance(restored_at, int)
            or restored_at
            < expected_startup_priority_bias_until_update
            or (
                expected_attach_at_update is not None
                and expected_attach_at_update > 0
                and restored_at >= expected_attach_at_update
            )
        ):
            raise CollectionError("trace_result_invalid")
        priority_times = (
            startup_priority_phase.get("started_perf_counter_ns"),
            startup_priority_phase.get("restored_perf_counter_ns"),
            startup_priority_phase.get("finished_perf_counter_ns"),
        )
        if (
            any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in priority_times
            )
            or not (
                payload["trace_started_perf_counter_ns"]
                < priority_times[0]
                < priority_times[1]
                <= priority_times[2]
                < payload["trace_finished_perf_counter_ns"]
            )
        ):
            raise CollectionError("trace_result_invalid")
        main_rows = 0
        for row in threads:
            if (
                not isinstance(row, dict)
                or isinstance(row.get("thread_id"), bool)
                or not isinstance(row.get("thread_id"), int)
                or row["thread_id"] <= 0
                or row.get("role") != "main"
                or isinstance(row.get("original_priority"), bool)
                or not isinstance(row.get("original_priority"), int)
                or isinstance(row.get("biased_priority"), bool)
                or not isinstance(row.get("biased_priority"), int)
                or row.get("restored") is not True
                or row.get("restore_status")
                not in {"restored", "thread_exited"}
            ):
                raise CollectionError("trace_result_invalid")
            main_rows += 1
            if (
                row["thread_id"]
                != startup_priority_phase["main_thread_id"]
                or row["biased_priority"] != -2
            ):
                raise CollectionError("trace_result_invalid")
            if (
                row["restore_status"] == "restored"
                and row.get("restored_priority")
                != row["original_priority"]
            ):
                raise CollectionError("trace_result_invalid")
        if main_rows != 1:
            raise CollectionError("trace_result_invalid")
        if expected_startup_trace_handoff is True:
            handoff_resume_ns = result.get(
                "startup_handoff_resumed_perf_counter_ns"
            )
            if (
                first_applied != 0
                or startup_priority_phase["main_thread_id"]
                != result.get("startup_handoff_main_thread_id")
                or isinstance(handoff_resume_ns, bool)
                or not isinstance(handoff_resume_ns, int)
                or not (
                    priority_times[0]
                    < handoff_resume_ns
                    < priority_times[1]
                )
            ):
                raise CollectionError("trace_result_invalid")
    expected_allowed_pairs_valid = (
        expected_blackout_allowed_offset_pairs is None
        or (
            isinstance(expected_blackout_allowed_offset_pairs, list)
            and all(
                isinstance(pair, Mapping)
                and set(pair)
                == {"command_order_offset", "native_timeline_offset"}
                and not isinstance(
                    pair.get("command_order_offset"), bool
                )
                and isinstance(pair.get("command_order_offset"), int)
                and pair["command_order_offset"] >= 0
                and not isinstance(
                    pair.get("native_timeline_offset"), bool
                )
                and isinstance(pair.get("native_timeline_offset"), int)
                and pair["native_timeline_offset"] >= 0
                for pair in expected_blackout_allowed_offset_pairs
            )
        )
    )
    if not expected_allowed_pairs_valid:
        raise CollectionError("trace_result_invalid")
    expected_allowed_pair_values = tuple(
        (
            pair["command_order_offset"],
            pair["native_timeline_offset"],
        )
        for pair in (expected_blackout_allowed_offset_pairs or [])
    )
    effective_blackout_rebase_mode: str | None = None
    if expected_blackout_file_write_order_rebase is None:
        if any(
            value is not None
            for value in (
                expected_blackout_file_write_rebase_mode,
                expected_blackout_command_order_offset,
                expected_blackout_native_timeline_offset,
                expected_blackout_allowed_offset_pairs,
            )
        ):
            raise CollectionError("trace_result_invalid")
    elif expected_blackout_file_write_order_rebase:
        effective_blackout_rebase_mode = (
            expected_blackout_file_write_rebase_mode
            or BLACKOUT_FILE_WRITE_REBASE_EXACT_MODE
        )
        if effective_blackout_rebase_mode not in {
            BLACKOUT_FILE_WRITE_REBASE_EXACT_MODE,
            BLACKOUT_FILE_WRITE_REBASE_ENVELOPE_MODE,
            BLACKOUT_FILE_WRITE_REBASE_ENVELOPE_SET_MODE,
        }:
            raise CollectionError("trace_result_invalid")
        if effective_blackout_rebase_mode == (
            BLACKOUT_FILE_WRITE_REBASE_EXACT_MODE
        ):
            if (
                expected_blackout_command_order_offset is not None
                or expected_blackout_native_timeline_offset is not None
                or expected_allowed_pair_values
            ):
                raise CollectionError("trace_result_invalid")
        elif effective_blackout_rebase_mode == (
            BLACKOUT_FILE_WRITE_REBASE_ENVELOPE_MODE
        ) and (
            isinstance(expected_blackout_command_order_offset, bool)
            or not isinstance(
                expected_blackout_command_order_offset, int
            )
            or expected_blackout_command_order_offset <= 0
            or isinstance(expected_blackout_native_timeline_offset, bool)
            or not isinstance(
                expected_blackout_native_timeline_offset, int
            )
            or expected_blackout_native_timeline_offset < 0
            or expected_allowed_pair_values
        ):
            raise CollectionError("trace_result_invalid")
        elif effective_blackout_rebase_mode == (
            BLACKOUT_FILE_WRITE_REBASE_ENVELOPE_SET_MODE
        ) and (
            expected_blackout_command_order_offset is not None
            or expected_blackout_native_timeline_offset is not None
            or len(expected_allowed_pair_values) < 2
            or tuple(sorted(set(expected_allowed_pair_values)))
            != expected_allowed_pair_values
            or not any(
                command_offset > 0
                for command_offset, _ in expected_allowed_pair_values
            )
        ):
            raise CollectionError("trace_result_invalid")
    elif any(
        value is not None
        for value in (
            expected_blackout_file_write_rebase_mode,
            expected_blackout_command_order_offset,
            expected_blackout_native_timeline_offset,
        )
    ) or expected_allowed_pair_values:
        raise CollectionError("trace_result_invalid")
    if expected_blackout_file_write_order_rebase is not None:
        options = payload.get("options")
        if (
            not isinstance(options, dict)
            or options.get(
                "allow_blackout_file_write_order_rebase"
            )
            is not expected_blackout_file_write_order_rebase
            or options.get("blackout_expected_command_order_offset")
            != expected_blackout_command_order_offset
            or options.get("blackout_expected_native_timeline_offset")
            != expected_blackout_native_timeline_offset
            or (
                expected_blackout_allowed_offset_pairs is not None
                and options.get("blackout_allowed_offset_pairs", [])
                != expected_blackout_allowed_offset_pairs
            )
        ):
            raise CollectionError("trace_result_invalid")
    if expected_allow_post_blackout_overdue_idle_reentry is not None:
        options = payload.get("options")
        candidates = (
            expected_post_blackout_overdue_idle_reentry_candidates
        )
        if (
            not isinstance(options, dict)
            or options.get(
                "allow_post_blackout_overdue_idle_reentry",
                False,
            )
            is not expected_allow_post_blackout_overdue_idle_reentry
            or candidates is None
        ):
            raise CollectionError("trace_result_invalid")
        _validate_post_blackout_overdue_idle_reentry_receipts(
            result,
            expected_pid=expected_pid,
            enabled=expected_allow_post_blackout_overdue_idle_reentry,
            candidates=candidates,
        )
    if expected_post_blackout_attach_stabilization is not None:
        options = payload.get("options")
        if (
            not isinstance(options, dict)
            or options.get(
                "post_blackout_attach_stabilization", False
            )
            is not expected_post_blackout_attach_stabilization
            or (
                (
                    expected_blackout_file_write_order_rebase is True
                    or expected_allow_post_blackout_overdue_idle_reentry
                    is True
                )
                and not expected_post_blackout_attach_stabilization
            )
        ):
            raise CollectionError("trace_result_invalid")
    if (
        expected_blackout_file_write_order_rebase_rows is not None
        and expected_blackout_file_write_order_rebase is None
    ):
        raise CollectionError("trace_result_invalid")
    if (
        effective_blackout_rebase_mode
        == BLACKOUT_FILE_WRITE_REBASE_ENVELOPE_MODE
        and (
            expected_blackout_file_write_order_rebase_rows is None
            or expected_blackout_command_order_offset
            > len(expected_blackout_file_write_order_rebase_rows)
        )
    ):
        raise CollectionError("trace_result_invalid")
    if (
        effective_blackout_rebase_mode
        == BLACKOUT_FILE_WRITE_REBASE_ENVELOPE_SET_MODE
        and (
            expected_blackout_file_write_order_rebase_rows is None
            or any(
                command_offset
                > len(expected_blackout_file_write_order_rebase_rows)
                for command_offset, _ in expected_allowed_pair_values
            )
        )
    ):
        raise CollectionError("trace_result_invalid")
    if (
        expected_post_blackout_overdue_idle_reentry_candidates
        is not None
        and expected_allow_post_blackout_overdue_idle_reentry is None
    ):
        raise CollectionError("trace_result_invalid")
    single_phase_attach = (
        expected_attach_at_update is not None
        and expected_detach_at_update is None
        and expected_reattach_at_update is None
    )
    if single_phase_attach:
        options = payload.get("options")
        if (
            not isinstance(options, dict)
            or options.get("attach_at_update")
            != expected_attach_at_update
            or options.get("detach_at_update") is not None
            or options.get("reattach_at_update") is not None
            or payload.get("trace_phases") is not None
            or payload.get("debugger_blackout") is not None
        ):
            raise CollectionError("trace_result_invalid")
    expected_phases = (
        (None, None, None)
        if single_phase_attach
        else (
            expected_attach_at_update,
            expected_detach_at_update,
            expected_reattach_at_update,
        )
    )
    if any(value is not None for value in expected_phases):
        if any(value is None for value in expected_phases):
            raise CollectionError("trace_result_invalid")
        options = payload.get("options")
        phases = payload.get("trace_phases")
        blackout = payload.get("debugger_blackout")
        if (
            not isinstance(options, dict)
            or options.get("attach_at_update")
            != expected_attach_at_update
            or options.get("detach_at_update")
            != expected_detach_at_update
            or options.get("reattach_at_update")
            != expected_reattach_at_update
            or not isinstance(phases, list)
            or len(phases) != 2
            or not isinstance(blackout, dict)
        ):
            raise CollectionError("trace_result_invalid")
        early, late = phases
        early_result = (
            early.get("result") if isinstance(early, dict) else None
        )
        late_result = (
            late.get("result") if isinstance(late, dict) else None
        )
        if (
            not isinstance(early, dict)
            or not isinstance(late, dict)
            or early.get("name") != "pre_capture"
            or late.get("name") != "post_capture"
            or not isinstance(early_result, dict)
            or not isinstance(late_result, dict)
            or early_result.get("stopped_at_update") is not True
            or early_result.get("exited") is not False
            or early_result.get("failure_count") != 0
            or late_result.get("exited") is not True
            or late_result.get("exit_code") != 0
            or late_result.get("failure_count") != 0
            or blackout.get("requested_detach_update")
            != expected_detach_at_update
            or blackout.get("requested_reattach_update")
            != expected_reattach_at_update
            or blackout.get("read_only_wait") is not True
            or blackout.get("process_id") != expected_pid
        ):
            raise CollectionError("trace_result_invalid")
        if terminal_close_expected and (
            early_result.get("terminal_close_requests", []) != []
            or late_result.get("terminal_close_requests")
            != terminal_close_requests
            or late_result.get("stopped_at_command_order") is not True
        ):
            raise CollectionError("trace_result_invalid")
        if expected_allow_attach_stabilization is True:
            if (
                early_result.get("attach_stabilization_verified")
                is not True
                or early_result.get("attach_stabilization_start_update")
                != expected_attach_at_update
                or (
                    expected_post_blackout_attach_stabilization is not True
                    and (
                        late_result.get(
                            "attach_stabilization_verified",
                            False,
                        )
                        is not False
                        or late_result.get(
                            "attach_stabilization_skipped_hits",
                            0,
                        )
                        != 0
                    )
                )
            ):
                raise CollectionError("trace_result_invalid")
        if (
            expected_post_blackout_attach_stabilization is True
        ):
            observations = late_result.get(
                "attach_stabilization_observations"
            )
            skipped_hits = late_result.get(
                "attach_stabilization_skipped_hits"
            )
            verified_update = late_result.get(
                "attach_stabilization_verified_update"
            )
            verified_order = late_result.get(
                "attach_stabilization_verified_command_order"
            )
            stabilization_start = late_result.get(
                "attach_stabilization_start_update"
            )
            if (
                late_result.get(
                    "attach_stabilization_verified"
                )
                is not True
                or stabilization_start
                != blackout.get("reattach_observed_update")
                or isinstance(skipped_hits, bool)
                or not isinstance(skipped_hits, int)
                or not (
                    0
                    <= skipped_hits
                    < ATTACH_STABILIZATION_MAX_SKIPPED_HITS
                )
                or isinstance(verified_update, bool)
                or not isinstance(verified_update, int)
                or verified_update < stabilization_start
                or verified_update - stabilization_start
                > ATTACH_STABILIZATION_MAX_UPDATE_DELTA
                or isinstance(verified_order, bool)
                or not isinstance(verified_order, int)
                or verified_order < 0
                or not isinstance(observations, list)
                or len(observations) != skipped_hits + 1
                or not all(
                    isinstance(row, dict) for row in observations
                )
                or any(
                    row.get("accepted") is not False
                    for row in observations[:-1]
                )
                or observations[-1].get("accepted") is not True
                or observations[-1].get("reason")
                != "exact_complete_main_boundary"
                or observations[-1].get("framework_update")
                != verified_update
                or observations[-1].get("command_order")
                != verified_order
                or observations[-1].get(
                    "candidate_command_order_offset"
                )
                != late_result.get("command_order_offset")
            ):
                raise CollectionError("trace_result_invalid")
            if expected_allow_attach_stabilization is not True and (
                early_result.get(
                    "attach_stabilization_verified", False
                )
                is not False
                or early_result.get(
                    "attach_stabilization_skipped_hits", 0
                )
                != 0
                or result.get("attach_stabilization_verified")
                is not True
                or result.get("attach_stabilization_skipped_hits")
                != skipped_hits
                or result.get("attach_stabilization_start_update")
                != stabilization_start
                or result.get(
                    "attach_stabilization_verified_update"
                )
                != verified_update
                or result.get(
                    "attach_stabilization_verified_command_order"
                )
                != verified_order
                or result.get("attach_stabilization_observations")
                != observations
            ):
                raise CollectionError("trace_result_invalid")
        elif (
            late_result.get("attach_stabilization_verified", False)
            is not False
            or late_result.get("attach_stabilization_skipped_hits", 0) != 0
        ):
            raise CollectionError("trace_result_invalid")
        overdue_receipts = result.get(
            "service_exit_overdue_idle_reentries",
            [],
        )
        if (
            early_result.get(
                "service_exit_overdue_idle_reentries",
                [],
            )
            != []
            or late_result.get(
                "service_exit_overdue_idle_reentries",
                [],
            )
            != overdue_receipts
        ):
            raise CollectionError("trace_result_invalid")
        if expected_blackout_file_write_order_rebase is not None:
            rebase = blackout.get("command_order_rebase")
            expected_rebase_rows = (
                expected_blackout_file_write_order_rebase_rows
            )
            if expected_rebase_rows is None:
                raise CollectionError("trace_result_invalid")
            if (
                expected_blackout_file_write_order_rebase
                and effective_blackout_rebase_mode
                in {
                    BLACKOUT_FILE_WRITE_REBASE_ENVELOPE_MODE,
                    BLACKOUT_FILE_WRITE_REBASE_ENVELOPE_SET_MODE,
                }
            ):
                envelope_set_mode = effective_blackout_rebase_mode == (
                    BLACKOUT_FILE_WRITE_REBASE_ENVELOPE_SET_MODE
                )
                if envelope_set_mode:
                    selected_command_order_offset = late_result.get(
                        "command_order_offset"
                    )
                    selected_native_timeline_offset = late_result.get(
                        "blackout_native_timeline_offset"
                    )
                    if (
                        isinstance(selected_command_order_offset, bool)
                        or not isinstance(
                            selected_command_order_offset, int
                        )
                        or isinstance(
                            selected_native_timeline_offset, bool
                        )
                        or not isinstance(
                            selected_native_timeline_offset, int
                        )
                        or (
                            selected_command_order_offset,
                            selected_native_timeline_offset,
                        )
                        not in expected_allowed_pair_values
                    ):
                        raise CollectionError("trace_result_invalid")
                else:
                    assert isinstance(
                        expected_blackout_command_order_offset, int
                    )
                    assert isinstance(
                        expected_blackout_native_timeline_offset, int
                    )
                    selected_command_order_offset = (
                        expected_blackout_command_order_offset
                    )
                    selected_native_timeline_offset = (
                        expected_blackout_native_timeline_offset
                    )
                expected_mechanism = (
                    "bounded_long_form_file_write_candidate_envelope_set"
                    if envelope_set_mode
                    else "bounded_long_form_file_write_candidate_envelope"
                )
                expected_verification_mode = (
                    "preregistered_finite_offset_set_and_fixed_suffix"
                    if envelope_set_mode
                    else "preregistered_candidate_envelope_and_fixed_suffix"
                )
                selected_offset_pair = {
                    "command_order_offset": selected_command_order_offset,
                    "native_timeline_offset": (
                        selected_native_timeline_offset
                    ),
                }
                expected_indices = [
                    row["row_index"] for row in expected_rebase_rows
                ]
                candidate_rows = (
                    rebase.get("candidate_rows")
                    if isinstance(rebase, dict)
                    else None
                )
                first_live_order = (
                    rebase.get("first_live_command_order")
                    if isinstance(rebase, dict)
                    else None
                )
                offset = (
                    rebase.get("observed_offset")
                    if isinstance(rebase, dict)
                    else None
                )
                timeline_offset = (
                    rebase.get("observed_native_timeline_offset")
                    if isinstance(rebase, dict)
                    else None
                )
                late_observations = late_result.get(
                    "attach_stabilization_observations"
                )
                if (
                    not expected_rebase_rows
                    or not isinstance(rebase, dict)
                    or rebase.get("mechanism")
                    != expected_mechanism
                    or rebase.get("detached_after_update")
                    != blackout.get("detached_after_update")
                    or isinstance(offset, bool)
                    or not isinstance(offset, int)
                    or offset != selected_command_order_offset
                    or isinstance(timeline_offset, bool)
                    or not isinstance(timeline_offset, int)
                    or timeline_offset
                    != selected_native_timeline_offset
                    or isinstance(first_live_order, bool)
                    or not isinstance(first_live_order, int)
                    or first_live_order < 0
                    or rebase.get("first_observed_live_command_order")
                    != late_result.get("first_command_order")
                    or rebase.get("first_offline_command_order")
                    != first_live_order + offset
                    or rebase.get("candidate_row_count")
                    != len(expected_rebase_rows)
                    or rebase.get(
                        "fixed_for_entire_post_capture_phase"
                    )
                    is not True
                    or rebase.get(
                        "exact_missing_row_identity_available"
                    )
                    is not False
                    or rebase.get("verification_mode")
                    != expected_verification_mode
                    or (
                        envelope_set_mode
                        and (
                            rebase.get("allowed_offset_pairs")
                            != expected_blackout_allowed_offset_pairs
                            or rebase.get("selected_offset_pair")
                            != selected_offset_pair
                        )
                    )
                    or (
                        not envelope_set_mode
                        and (
                            "allowed_offset_pairs" in rebase
                            or "selected_offset_pair" in rebase
                        )
                    )
                    or not isinstance(candidate_rows, list)
                    or len(candidate_rows) != len(expected_rebase_rows)
                    or early_result.get("command_order_offset") != 0
                    or early_result.get("command_order_rebase_rows")
                    != []
                    or early_result.get(
                        "blackout_file_write_rebase_candidate_rows", []
                    )
                    != []
                    or early_result.get(
                        "blackout_native_timeline_offset", 0
                    )
                    != 0
                    or late_result.get("command_order_offset")
                    != selected_command_order_offset
                    or late_result.get("command_order_rebase_rows")
                    != []
                    or late_result.get(
                        "blackout_file_write_rebase_candidate_rows"
                    )
                    != expected_indices
                    or late_result.get(
                        "blackout_native_timeline_offset"
                    )
                    != selected_native_timeline_offset
                    or result.get("command_order_offset")
                    != selected_command_order_offset
                    or result.get("command_order_rebase_rows") != []
                    or result.get(
                        "blackout_file_write_rebase_candidate_rows"
                    )
                    != expected_indices
                    or result.get("blackout_native_timeline_offset")
                    != selected_native_timeline_offset
                    or late_result.get(
                        "attach_stabilization_verified_command_order"
                    )
                    != first_live_order
                    or not isinstance(
                        late_result.get("first_command_order"), int
                    )
                    or late_result.get("first_command_order") < 0
                    or late_result.get("first_command_order")
                    > first_live_order
                    or not isinstance(late_observations, list)
                    or not late_observations
                    or late_observations[-1].get(
                        "candidate_command_order_offset"
                    )
                    != selected_command_order_offset
                    or late_observations[-1].get(
                        "candidate_native_timeline_offset"
                    )
                    != selected_native_timeline_offset
                ):
                    raise CollectionError("trace_result_invalid")
                previous_end = -1
                for actual, expected in zip(
                    candidate_rows,
                    expected_rebase_rows,
                    strict=True,
                ):
                    if (
                        not isinstance(actual, dict)
                        or set(actual)
                        != {
                            "row_index",
                            "start_bit_position",
                            "end_bit_position",
                            "update",
                            "command_number",
                            "kind",
                            "short_form",
                            "success",
                        }
                        or {
                            key: actual.get(key)
                            for key in (
                                "row_index",
                                "update",
                                "command_number",
                                "kind",
                                "short_form",
                                "success",
                            )
                        }
                        != dict(expected)
                        or isinstance(
                            actual.get("start_bit_position"), bool
                        )
                        or not isinstance(
                            actual.get("start_bit_position"), int
                        )
                        or isinstance(
                            actual.get("end_bit_position"), bool
                        )
                        or not isinstance(
                            actual.get("end_bit_position"), int
                        )
                        or actual["start_bit_position"] < previous_end
                        or (
                            actual["end_bit_position"]
                            - actual["start_bit_position"]
                        )
                        != 11
                    ):
                        raise CollectionError("trace_result_invalid")
                    previous_end = actual["end_bit_position"]
            elif expected_blackout_file_write_order_rebase:
                accounted_rows = (
                    rebase.get("accounted_rows")
                    if isinstance(rebase, dict)
                    else None
                )
                expected_indices = [
                    row["row_index"] for row in expected_rebase_rows
                ]
                first_live_order = (
                    rebase.get("first_live_command_order")
                    if isinstance(rebase, dict)
                    else None
                )
                offset = (
                    rebase.get("observed_offset")
                    if isinstance(rebase, dict)
                    else None
                )
                if (
                    not expected_rebase_rows
                    or not isinstance(rebase, dict)
                    or rebase.get("mechanism")
                    != (
                        "direct_long_form_file_write_result_consumption"
                    )
                    or rebase.get("detached_after_update")
                    != blackout.get("detached_after_update")
                    or isinstance(offset, bool)
                    or not isinstance(offset, int)
                    or offset != len(expected_rebase_rows)
                    or isinstance(first_live_order, bool)
                    or not isinstance(first_live_order, int)
                    or first_live_order < 0
                    or rebase.get(
                        "first_observed_live_command_order"
                    )
                    != late_result.get("first_command_order")
                    or rebase.get("first_offline_command_order")
                    != first_live_order + offset
                    or rebase.get("accounted_row_count") != offset
                    or rebase.get(
                        "fixed_for_entire_post_capture_phase"
                    )
                    is not True
                    or not isinstance(accounted_rows, list)
                    or len(accounted_rows) != offset
                    or early_result.get("command_order_offset") != 0
                    or early_result.get("command_order_rebase_rows")
                    != []
                    or late_result.get("command_order_offset") != offset
                    or late_result.get("command_order_rebase_rows")
                    != expected_indices
                    or late_result.get(
                        "attach_stabilization_verified_command_order"
                    )
                    != first_live_order
                    or not isinstance(
                        late_result.get("first_command_order"), int
                    )
                    or late_result.get("first_command_order") < 0
                    or late_result.get("first_command_order")
                    > first_live_order
                    or not isinstance(
                        late_result.get(
                            "attach_stabilization_observations"
                        ),
                        list,
                    )
                    or late_result[
                        "attach_stabilization_observations"
                    ][-1].get("candidate_command_order_offset")
                    != offset
                    or late_result.get(
                        "blackout_file_write_rebase_candidate_rows", []
                    )
                    != []
                    or late_result.get(
                        "blackout_native_timeline_offset", 0
                    )
                    != 0
                    or result.get(
                        "blackout_file_write_rebase_candidate_rows", []
                    )
                    != []
                    or result.get("blackout_native_timeline_offset", 0)
                    != 0
                ):
                    raise CollectionError("trace_result_invalid")
                previous_end = -1
                for actual, expected in zip(
                    accounted_rows,
                    expected_rebase_rows,
                    strict=True,
                ):
                    if (
                        not isinstance(actual, dict)
                        or set(actual)
                        != {
                            "row_index",
                            "start_bit_position",
                            "end_bit_position",
                            "update",
                            "command_number",
                            "kind",
                            "short_form",
                            "success",
                        }
                        or {
                            key: actual.get(key)
                            for key in (
                                "row_index",
                                "update",
                                "command_number",
                                "kind",
                                "short_form",
                                "success",
                            )
                        }
                        != dict(expected)
                        or isinstance(
                            actual.get("start_bit_position"),
                            bool,
                        )
                        or not isinstance(
                            actual.get("start_bit_position"),
                            int,
                        )
                        or isinstance(
                            actual.get("end_bit_position"),
                            bool,
                        )
                        or not isinstance(
                            actual.get("end_bit_position"),
                            int,
                        )
                        or actual["start_bit_position"] < previous_end
                        or (
                            actual["end_bit_position"]
                            - actual["start_bit_position"]
                        )
                        != 11
                    ):
                        raise CollectionError("trace_result_invalid")
                    previous_end = actual["end_bit_position"]
            else:
                if (
                    early_result.get(
                        "blackout_file_write_rebase_candidate_rows", []
                    )
                    != []
                    or early_result.get(
                        "blackout_native_timeline_offset", 0
                    )
                    != 0
                    or late_result.get(
                        "blackout_file_write_rebase_candidate_rows", []
                    )
                    != []
                    or late_result.get(
                        "blackout_native_timeline_offset", 0
                    )
                    != 0
                    or result.get(
                        "blackout_file_write_rebase_candidate_rows", []
                    )
                    != []
                    or result.get("blackout_native_timeline_offset", 0)
                    != 0
                ):
                    raise CollectionError("trace_result_invalid")
                if diagnostic_dynamic_service_rebase:
                    diagnostic_phase_fields = (
                        "service_consumed_file_write_debt_rows",
                        "service_consumed_file_write_surplus_rows",
                        "effective_deferred_file_write_debt_rows",
                        "deferred_file_write_discharges",
                        "successful_file_write_debt_barriers",
                        "service_continuation_verifications",
                        "service_exit_timeline_commits",
                        "service_exit_timeline_suffix_rebases",
                        "terminal_registry_write_eof_exits",
                    )
                    if (
                        expected_rebase_rows
                        or rebase is not None
                        or early_result.get("command_order_offset") != 0
                        or early_result.get("command_order_rebase_rows")
                        != []
                        or early_result.get(
                            "diagnostic_successful_file_write_padding_rows"
                        )
                        != expected_diagnostic_successful_file_write_padding_rows
                        or any(
                            early_result.get(name, []) != []
                            for name in diagnostic_phase_fields
                        )
                        or late_result.get("command_order_offset")
                        != result.get("command_order_offset")
                        or late_result.get("command_order_rebase_rows")
                        != result.get("command_order_rebase_rows")
                        or late_result.get(
                            "diagnostic_successful_file_write_padding_rows"
                        )
                        != expected_diagnostic_successful_file_write_padding_rows
                        or any(
                            late_result.get(name) != result.get(name)
                            for name in diagnostic_phase_fields
                        )
                    ):
                        raise CollectionError("trace_result_invalid")
                elif (
                    expected_rebase_rows
                    or rebase is not None
                    or (
                        expected_allow_font_cache_manifest_completion_debt
                        is not True
                        and (
                            early_result.get("command_order_offset") != 0
                            or early_result.get(
                                "command_order_rebase_rows"
                            )
                            != []
                        )
                    )
                    or late_result.get("command_order_offset")
                    != early_result.get("command_order_offset")
                    or late_result.get("command_order_rebase_rows")
                    != early_result.get("command_order_rebase_rows")
                    or result.get("command_order_offset")
                    != early_result.get("command_order_offset")
                    or result.get("command_order_rebase_rows")
                    != early_result.get("command_order_rebase_rows")
                ):
                    raise CollectionError("trace_result_invalid")
        timing_values = (
            early.get("started_perf_counter_ns"),
            early.get("finished_perf_counter_ns"),
            blackout.get("started_perf_counter_ns"),
            blackout.get("finished_perf_counter_ns"),
            late.get("started_perf_counter_ns"),
            late.get("finished_perf_counter_ns"),
        )
        if (
            any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in timing_values
            )
            or not (
                timing_values[0]
                < timing_values[1]
                <= timing_values[2]
                < timing_values[3]
                <= timing_values[4]
                < timing_values[5]
            )
        ):
            raise CollectionError("trace_result_invalid")
    if expected_crt_rand_seed is not None:
        startup_rng = payload.get("startup_rng")
        options = payload.get("options")
        startup_seed_transport = (
            expected_startup_seed_transport or "debugger_register"
        )
        if (
            not isinstance(startup_rng, dict)
            or not isinstance(options, dict)
            or startup_seed_transport not in STARTUP_SEED_TRANSPORTS
            or options.get("direct_runtime") is not True
            or options.get("crt_rand_seed") != expected_crt_rand_seed
            or (
                expected_startup_seed_transport is not None
                and options.get("startup_seed_transport")
                != startup_seed_transport
            )
            or startup_rng.get("process_id") != expected_pid
            or startup_rng.get("seed") != expected_crt_rand_seed
            or startup_rng.get("compatibility_layer") != "HIGHDPIAWARE"
            or startup_rng.get("persistent_file_modified") is not False
        ):
            raise CollectionError("trace_result_invalid")
        if startup_seed_transport == "debugger_register":
            if (
                startup_rng.get("register_override")
                != "eax_before_push_to_srand"
                or startup_rng.get("original_instruction_hex") != "50"
                or isinstance(
                    startup_rng.get("detach_pending_events_drained"),
                    bool,
                )
                or not isinstance(
                    startup_rng.get("detach_pending_events_drained"),
                    int,
                )
                or startup_rng["detach_pending_events_drained"] < 0
                or isinstance(startup_rng.get("detach_attempts"), bool)
                or not isinstance(startup_rng.get("detach_attempts"), int)
                or not 1 <= startup_rng["detach_attempts"] <= 8
            ):
                raise CollectionError("trace_result_invalid")
            startup_times = (
                startup_rng.get("breakpoint_observed_perf_counter_ns"),
                startup_rng.get("debugger_detached_perf_counter_ns"),
            )
            if (
                any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value <= 0
                    for value in startup_times
                )
                or not (
                    payload["trace_started_perf_counter_ns"]
                    < startup_times[0]
                    < startup_times[1]
                    < payload["trace_finished_perf_counter_ns"]
                )
            ):
                raise CollectionError("trace_result_invalid")
        else:
            stub_hex = startup_rng.get("stub_machine_code_hex")
            if (
                startup_rng.get("seed_transport")
                != "temporary_get_tick_count_iat_stub"
                or startup_rng.get("seed_call_return_address_hex")
                != "0x0068f577"
                or startup_rng.get("iat_address_hex") != "0x0094b1d4"
                or startup_rng.get("entry_point_address_hex")
                != "0x008d1883"
                or startup_rng.get("entry_original_hex") != "e82e"
                or startup_rng.get("stub_machine_code")
                != "caller_return_address_filter_then_forward"
                or not isinstance(stub_hex, str)
                or len(stub_hex) < 20
                or len(stub_hex) % 2 != 0
                or startup_rng.get("iat_restored") is not True
                or isinstance(
                    startup_rng.get("restored_at_framework_update"),
                    bool,
                )
                or not isinstance(
                    startup_rng.get("restored_at_framework_update"),
                    int,
                )
                or startup_rng["restored_at_framework_update"] < 0
            ):
                raise CollectionError("trace_result_invalid")
            startup_times = (
                startup_rng.get("loader_gate_perf_counter_ns"),
                startup_rng.get("iat_patched_perf_counter_ns"),
                startup_rng.get("restoration_guard_perf_counter_ns"),
                startup_rng.get("iat_restored_perf_counter_ns"),
            )
            if (
                any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value <= 0
                    for value in startup_times
                )
                or not (
                    payload["trace_started_perf_counter_ns"]
                    < startup_times[0]
                    <= startup_times[1]
                    < startup_times[2]
                    <= startup_times[3]
                    < payload["trace_finished_perf_counter_ns"]
                )
            ):
                raise CollectionError("trace_result_invalid")
    board_expectations = (
        expected_board_seed_address,
        expected_board_seed,
    )
    if (
        expected_global_rng_seed is not None
        and expected_board_seed is None
    ):
        raise CollectionError("trace_result_invalid")
    if (
        expected_thread_crt_rng_seed is not None
        and expected_board_seed is None
    ):
        raise CollectionError("trace_result_invalid")
    if any(value is not None for value in board_expectations):
        if any(value is None for value in board_expectations):
            raise CollectionError("trace_result_invalid")
        options = payload.get("options")
        board_rng = payload.get("board_rng")
        observations = (
            board_rng.get("observations")
            if isinstance(board_rng, dict)
            else None
        )
        if (
            not isinstance(options, dict)
            or options.get("board_seed_address")
            != expected_board_seed_address
            or options.get("board_seed") != expected_board_seed
            or options.get("global_rng_seed")
            != expected_global_rng_seed
            or options.get("thread_crt_rng_seed")
            != expected_thread_crt_rng_seed
            or not isinstance(board_rng, dict)
            or board_rng.get("call_site_address")
            != expected_board_seed_address
            or board_rng.get("effective_seed")
            != expected_board_seed
            or board_rng.get("global_rng_seed")
            != expected_global_rng_seed
            or board_rng.get("thread_crt_rng_seed")
            != expected_thread_crt_rng_seed
            or board_rng.get("original_instruction_hex") != "e8"
            or board_rng.get("register_override")
            != "ecx_seed_before_mtrand_srand_call"
            or board_rng.get("persistent_file_modified") is not False
            or not isinstance(observations, list)
        ):
            raise CollectionError("trace_result_invalid")
        assert isinstance(observations, list)
        if expected_source_bound_board_anchor is not None:
            applied_rows = [
                row
                for row in observations
                if isinstance(row, dict)
                and row.get("source_bound_board_anchor_applied") is True
            ]
            skipped_rows = observations[:-1]
            if (
                not observations
                or len(applied_rows) != 1
                or observations[-1] is not applied_rows[0]
                or any(
                    not isinstance(row, dict)
                    or row.get("overridden") is not False
                    or row.get("skipped_before_override") is not True
                    or row.get("source_bound_board_anchor_applied")
                    is not None
                    or row.get("framework_update", -1)
                    > expected_source_bound_board_anchor[
                        "framework_update"
                    ]
                    for row in skipped_rows
                )
            ):
                raise CollectionError("trace_result_invalid")
            observation = applied_rows[0]
            source_bound_global_post_state_match = observation.get(
                "source_bound_global_post_state_match"
            )
            source_bound_correction_index_delta = observation.get(
                "source_bound_correction_index_delta"
            )
            source_bound_bounded_correction_eligible = observation.get(
                "source_bound_bounded_correction_eligible"
            )
            source_bound_live_global_post_index = observation.get(
                "source_bound_live_global_post_index"
            )
            source_bound_live_global_post_sha256 = observation.get(
                "source_bound_live_global_post_sha256"
            )
            expected_source_bound_global_post_index = (
                expected_source_bound_board_anchor["expected_post_index"]
            )
            expected_source_bound_global_post_sha256 = (
                expected_source_bound_board_anchor[
                    "expected_post_state_sha256"
                ]
            )
            natural_source_bound_anchor = (
                source_bound_global_post_state_match is True
            )
            corrected_source_bound_anchor = (
                source_bound_global_post_state_match is False
            )
            if (
                observation.get("rng_owner_vtable")
                != BOARD_SEED_OWNER_VTABLE
                or observation.get("rng_object_address")
                != observation.get("rng_owner_address", -1)
                + BOARD_MTRAND_OFFSET
                or observation.get("source_bound_target_update_match")
                is not True
                or observation.get("source_bound_main_thread_match")
                is not True
                or observation.get("source_bound_rng_owner_board_match")
                is not True
                or observation.get(
                    "source_bound_active_board_context_match"
                )
                is not True
                or observation.get(
                    "source_bound_attach_stabilization_clear"
                )
                is not True
                or observation.get("source_bound_live_post_draw_verified")
                is not True
                or not isinstance(
                    source_bound_global_post_state_match,
                    bool,
                )
                or isinstance(source_bound_correction_index_delta, bool)
                or not isinstance(source_bound_correction_index_delta, int)
                or not isinstance(
                    source_bound_bounded_correction_eligible,
                    bool,
                )
                or observation.get(
                    "source_bound_expected_global_post_index"
                )
                != expected_source_bound_global_post_index
                or observation.get(
                    "source_bound_expected_global_post_sha256"
                )
                != expected_source_bound_global_post_sha256
            ):
                raise CollectionError("trace_result_invalid")
            if natural_source_bound_anchor:
                if (
                    observation.get("source_bound_observed_seed_match")
                    is not True
                    or source_bound_live_global_post_index
                    != expected_source_bound_global_post_index
                    or source_bound_live_global_post_sha256
                    != expected_source_bound_global_post_sha256
                    or source_bound_correction_index_delta != 0
                    or source_bound_bounded_correction_eligible is not False
                ):
                    raise CollectionError("trace_result_invalid")
            elif corrected_source_bound_anchor:
                if (
                    expected_allow_source_bound_board_global_correction
                    is not True
                    or observation.get("source_bound_observed_seed_match")
                    is not False
                    or isinstance(source_bound_live_global_post_index, bool)
                    or not isinstance(
                        source_bound_live_global_post_index,
                        int,
                    )
                    or not 0
                    <= source_bound_live_global_post_index
                    <= MTRAND_STATE_WORDS
                    or source_bound_correction_index_delta
                    != (
                        expected_source_bound_global_post_index
                        - source_bound_live_global_post_index
                    )
                    or not 0
                    < abs(source_bound_correction_index_delta)
                    <= SOURCE_BOUND_GLOBAL_CORRECTION_MAX_DRAWS
                    or source_bound_bounded_correction_eligible is not True
                    or not isinstance(
                        source_bound_live_global_post_sha256,
                        str,
                    )
                    or len(source_bound_live_global_post_sha256) != 71
                    or not source_bound_live_global_post_sha256.startswith(
                        "sha256:"
                    )
                    or any(
                        character not in "0123456789abcdef"
                        for character in (
                            source_bound_live_global_post_sha256[7:]
                        )
                    )
                    or source_bound_live_global_post_sha256
                    == expected_source_bound_global_post_sha256
                ):
                    raise CollectionError("trace_result_invalid")
            else:
                raise CollectionError("trace_result_invalid")
            source_bound_receipt = payload.get(
                "source_bound_board_anchor"
            )
            source_bound_receipt_observations = (
                source_bound_receipt.get("observations")
                if isinstance(source_bound_receipt, dict)
                else None
            )
            source_bound_receipt_observation = (
                source_bound_receipt_observations[0]
                if isinstance(source_bound_receipt_observations, list)
                and len(source_bound_receipt_observations) == 1
                and isinstance(source_bound_receipt_observations[0], dict)
                else None
            )
            source_bound_receipt_global = (
                source_bound_receipt_observation.get("global")
                if isinstance(source_bound_receipt_observation, dict)
                else None
            )
            if (
                not isinstance(source_bound_receipt_observation, dict)
                or not isinstance(source_bound_receipt_global, dict)
                or source_bound_receipt_observation.get("classification")
                != (
                    "source-bound-natural-retail-post-call-board-anchor"
                    if natural_source_bound_anchor
                    else (
                        "source-bound-bounded-global-correction-"
                        "board-anchor"
                    )
                )
                or source_bound_receipt_observation.get("process_id")
                != observation.get("process_id")
                or source_bound_receipt_observation.get("thread_id")
                != observation.get("thread_id")
                or source_bound_receipt_observation.get("framework_update")
                != observation.get("framework_update")
                or source_bound_receipt_observation.get("observed_seed")
                != observation.get("observed_seed")
                or source_bound_receipt_observation.get("effective_seed")
                != observation.get("effective_seed")
                or source_bound_receipt_observation.get(
                    "active_board_address"
                )
                != observation.get("active_board_address")
                or source_bound_receipt_observation.get(
                    "rng_object_address"
                )
                != observation.get("rng_object_address")
                or source_bound_receipt_observation.get(
                    "rng_owner_address"
                )
                != observation.get("rng_owner_address")
                or source_bound_receipt_global.get("live_before_index")
                != source_bound_live_global_post_index
                or source_bound_receipt_global.get(
                    "live_before_state_sha256"
                )
                != source_bound_live_global_post_sha256
                or source_bound_receipt_global.get(
                    "natural_post_state_match"
                )
                is not natural_source_bound_anchor
                or source_bound_receipt_global.get(
                    "live_post_draw_verified"
                )
                is not True
                or source_bound_receipt_global.get(
                    "correction_index_delta"
                )
                != source_bound_correction_index_delta
                or source_bound_receipt_global.get(
                    "bounded_global_correction_applied"
                )
                is not corrected_source_bound_anchor
                or source_bound_receipt_observation.get("bytes_written")
                != observation.get(
                    "source_bound_board_anchor_bytes_written"
                )
            ):
                raise CollectionError("trace_result_invalid")
        else:
            if (
                len(observations) != 1
                or not isinstance(observations[0], dict)
            ):
                raise CollectionError("trace_result_invalid")
            observation = observations[0]
        unsigned_values = (
            observation.get("process_id"),
            observation.get("thread_id"),
            observation.get("perf_counter_ns"),
            observation.get("framework_update"),
            observation.get("call_site_address"),
            observation.get("rng_object_address"),
            observation.get("observed_seed"),
            observation.get("effective_seed"),
        )
        if expected_seed_board_before_attach:
            board_phase = payload.get("board_seed_phase")
            observation_time_valid = (
                isinstance(board_phase, dict)
                and isinstance(
                    board_phase.get("started_perf_counter_ns"),
                    int,
                )
                and isinstance(
                    board_phase.get("finished_perf_counter_ns"),
                    int,
                )
                and board_phase["started_perf_counter_ns"]
                < observation.get("perf_counter_ns", -1)
                < board_phase["finished_perf_counter_ns"]
            )
        else:
            observation_time_valid = (
                payload["trace_started_perf_counter_ns"]
                < observation.get("perf_counter_ns", -1)
                < payload["trace_finished_perf_counter_ns"]
            )
        if (
            any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in unsigned_values
            )
            or observation["process_id"] != expected_pid
            or observation["thread_id"] <= 0
            or observation["call_site_address"]
            != expected_board_seed_address
            or observation["rng_object_address"] <= 0
            or observation["rng_object_address"] > 0xFFFFFFFF
            or observation["observed_seed"] > 0xFFFFFFFF
            or observation["effective_seed"] != expected_board_seed
            or observation.get("overridden") is not True
            or (
                expected_source_bound_board_anchor is not None
                and observation.get("skipped_before_override") is not False
            )
            or not observation_time_valid
        ):
            raise CollectionError("trace_result_invalid")
        if (
            not expected_seed_board_before_attach
            and expected_attach_at_update is not None
            and (
            observation["framework_update"]
            < expected_attach_at_update
            )
        ):
            raise CollectionError("trace_result_invalid")
        if expected_detach_at_update is not None and (
            observation["framework_update"]
            >= expected_detach_at_update
        ):
            raise CollectionError("trace_result_invalid")
        if expected_global_rng_seed is not None:
            global_before = observation.get(
                "global_rng_state_sha256_before"
            )
            global_after = observation.get(
                "global_rng_state_sha256_after"
            )
            if (
                observation.get("global_rng_address")
                != GLOBAL_MTRAND_ADDRESS
                or observation.get("global_rng_seed")
                != expected_global_rng_seed
                or observation.get("global_rng_state_bytes")
                != MTRAND_STATE_BYTES
                or observation.get("global_rng_overridden") is not True
                or not isinstance(global_before, str)
                or len(global_before) != 71
                or not global_before.startswith("sha256:")
                or any(
                    character not in "0123456789abcdef"
                    for character in global_before[7:]
                )
                or global_after
                != _mtrand_state_sha256(expected_global_rng_seed)
            ):
                raise CollectionError("trace_result_invalid")
        if expected_thread_crt_rng_seed is not None:
            thread_crt_unsigned = (
                observation.get("fs_selector"),
                observation.get("fs_segment_base"),
                observation.get("teb_address"),
                observation.get("fls_index_address"),
                observation.get("fls_index"),
                observation.get("fls_data_address"),
                observation.get("fls_block_index"),
                observation.get("fls_entry_index"),
                observation.get("fls_block_slot_address"),
                observation.get("fls_block_address"),
                observation.get("fls_value_address"),
                observation.get("ptd_address"),
                observation.get("ptd_thread_id"),
                observation.get("rand_state_address"),
                observation.get("thread_crt_rng_state_before"),
                observation.get("thread_crt_rng_state_after"),
                observation.get("thread_crt_rng_seed"),
            )
            if any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value <= 0xFFFFFFFF
                for value in thread_crt_unsigned
            ):
                raise CollectionError("trace_result_invalid")
            shifted_index = observation["fls_index"] + 0x10
            expected_block_index = shifted_index.bit_length() - 1
            expected_entry_index = (
                shifted_index ^ (1 << expected_block_index)
            )
            if (
                not 0 < observation["fs_selector"] <= 0xFFFF
                or observation["fs_segment_base"] <= 0
                or observation["teb_address"] <= 0
                or observation["fls_index_address"]
                != CRT_FLS_INDEX_ADDRESS
                or not 0 < observation["fls_index"]
                < FLS_MAXIMUM_AVAILABLE
                or observation["fls_data_address"] <= 0
                or observation["fls_block_index"]
                != expected_block_index
                or observation["fls_entry_index"]
                != expected_entry_index
                or observation["fls_block_slot_address"]
                != (
                    observation["fls_data_address"]
                    + expected_block_index * 4
                    - 8
                )
                or observation["fls_block_address"] <= 0
                or observation["fls_value_address"]
                != (
                    observation["fls_block_address"]
                    + expected_entry_index * 4
                    + 4
                )
                or observation["ptd_address"] <= 0
                or observation["ptd_thread_id"]
                != observation["thread_id"]
                or observation["rand_state_address"]
                != (
                    observation["ptd_address"]
                    + CRT_PTD_RAND_STATE_OFFSET
                )
                or observation["thread_crt_rng_state_after"]
                != expected_thread_crt_rng_seed
                or observation["thread_crt_rng_seed"]
                != expected_thread_crt_rng_seed
                or observation.get("thread_crt_rng_overridden")
                is not True
            ):
                raise CollectionError("trace_result_invalid")
    _validate_source_bound_board_trace_receipt(
        payload,
        expected_pid=expected_pid,
        expected_anchor=expected_source_bound_board_anchor,
        expected_allow_global_correction=bool(
            expected_allow_source_bound_board_global_correction
        ),
    )
    _validate_source_bound_board_precall_trace_receipt(
        payload,
        expected_pid=expected_pid,
        expected_anchor=expected_source_bound_board_anchor,
        expected_precall=(
            expected_source_bound_board_precall_global_restore
        ),
    )
    return payload


_ZUMA_REGISTRY_ROOT = r"HKCU\Software\SteamPopCap\ZumasRevenge"
_VOLATILE_END_STATE_VALUES = (
    (_ZUMA_REGISTRY_ROOT, "", "LastGamePID"),
    (_ZUMA_REGISTRY_ROOT, "", "RegExData"),
)


def _projected_end_state_root(
    snapshot: PcStateSnapshot,
    *,
    pre_snapshot: PcStateSnapshot,
    process_id: int,
    launch_mode: str,
) -> str:
    def launcher_controls(
        source: PcStateSnapshot,
    ) -> tuple[Any, Any]:
        nodes = [
            node
            for node in source.registry_nodes
            if node.root == _ZUMA_REGISTRY_ROOT and node.subkey == ""
        ]
        if len(nodes) != 1 or not nodes[0].exists:
            raise CollectionError("end_state_launcher_root_invalid")
        values = {
            value.name.casefold(): value for value in nodes[0].values
        }
        last_pid = values.get("lastgamepid")
        registration_data = values.get("regexdata")
        if (
            last_pid is None
            or last_pid.name != "LastGamePID"
            or last_pid.value_type != 4
            or len(last_pid.data) != 4
        ):
            raise CollectionError("end_state_process_binding_invalid")
        if (
            registration_data is None
            or registration_data.name != "RegExData"
            or registration_data.value_type != 4
            or len(registration_data.data) != 4
        ):
            raise CollectionError("end_state_launcher_data_invalid")
        return last_pid, registration_data

    end_controls = launcher_controls(snapshot)
    if launch_mode == STEAM_LAUNCH_MODE:
        if int.from_bytes(end_controls[0].data, "little") != process_id:
            raise CollectionError("end_state_process_binding_invalid")
    elif launch_mode == DIRECT_FIXED_SEED_LAUNCH_MODE:
        if end_controls != launcher_controls(pre_snapshot):
            raise CollectionError("end_state_launcher_controls_changed")
    else:
        raise CollectionError("launch_mode_invalid")
    try:
        return snapshot.projected_state_root(
            _VOLATILE_END_STATE_VALUES
        )
    except Exception:
        raise CollectionError("end_state_projection_invalid") from None


def _wait_for_framework_poll_ready(
    process: subprocess.Popen[bytes],
    path: Path,
    *,
    expected_process_id: int,
    expected_executable_sha256: str,
    timeout: float = 10.0,
) -> Mapping[str, Any]:
    deadline = time.monotonic() + timeout
    while True:
        if path.is_file():
            ready = _read_json_object(
                path, "framework_state_poll_ready_invalid"
            )
            if (
                _canonical_json(ready) != path.read_bytes()
                or ready.get("schema") != FRAMEWORK_POLL_READY_SCHEMA
                or ready.get("version") != 1
                or ready.get("process_id") != expected_process_id
                or ready.get("executable_sha256")
                != expected_executable_sha256
                or isinstance(
                    ready.get("process_creation_filetime_100ns"), bool
                )
                or not isinstance(
                    ready.get("process_creation_filetime_100ns"), int
                )
                or ready["process_creation_filetime_100ns"] <= 0
                or isinstance(
                    ready.get("sample_before_perf_counter_ns"), bool
                )
                or not isinstance(
                    ready.get("sample_before_perf_counter_ns"), int
                )
                or isinstance(
                    ready.get("sample_after_perf_counter_ns"), bool
                )
                or not isinstance(
                    ready.get("sample_after_perf_counter_ns"), int
                )
                or ready["sample_after_perf_counter_ns"]
                < ready["sample_before_perf_counter_ns"]
            ):
                raise CollectionError(
                    "framework_state_poll_ready_invalid"
                )
            return ready
        return_code = process.poll()
        if return_code is not None:
            raise CollectionError("framework_state_poll_failed")
        if time.monotonic() >= deadline:
            raise CollectionError("framework_state_poll_ready_timeout")
        time.sleep(0.01)


def _signal_framework_poll_stop(path: Path, *, process_id: int) -> None:
    if path.exists():
        return
    part = path.with_name(path.name + ".part")
    _write_exclusive(
        part,
        _canonical_json(
            {
                "schema": FRAMEWORK_POLL_STOP_SCHEMA,
                "version": 1,
                "process_id": process_id,
                "requested_perf_counter_ns": time.perf_counter_ns(),
            }
        ),
    )
    try:
        os.replace(part, path)
    except OSError:
        raise CollectionError("framework_state_poll_stop_failed") from None


def _validate_framework_poll_evidence(
    path: Path,
    *,
    metadata: PcDxgiCaptureMetadata,
    update_map_path: Path,
    state_sidecar_path: Path,
    expected_arm_update: int,
    expected_arm_deadline_update: int,
    expected_target_hz: int,
    expected_observer_affinity_mask: int,
    expected_game_affinity_mask: int,
) -> Mapping[str, Any]:
    report = _read_json_object(
        path, "framework_state_poll_evidence_invalid"
    )
    try:
        canonical = _canonical_json(report) == path.read_bytes()
    except OSError:
        canonical = False
    identity = report.get("process_identity")
    observer = report.get("observer")
    binding = report.get("capture_binding")
    control = report.get("control")
    initial = report.get("initial_sample")
    final_state = report.get("final_state")
    if not all(
        isinstance(value, dict)
        for value in (
            identity,
            observer,
            binding,
            control,
            initial,
            final_state,
        )
    ):
        raise CollectionError("framework_state_poll_evidence_invalid")
    assert isinstance(identity, dict)
    assert isinstance(observer, dict)
    assert isinstance(binding, dict)
    assert isinstance(control, dict)
    assert isinstance(initial, dict)
    assert isinstance(final_state, dict)
    sample_intervals = observer.get("sample_completion_interval_ns")
    affinity_topology = observer.get("affinity_topology")
    initial_state = initial.get("state")
    if (
        not isinstance(sample_intervals, dict)
        or not isinstance(affinity_topology, dict)
        or not isinstance(initial_state, dict)
    ):
        raise CollectionError("framework_state_poll_evidence_invalid")
    mean_hz = observer.get("mean_polls_per_second")
    p99_interval = sample_intervals.get("p99")
    poll_count = observer.get("poll_count")
    draw_event_count = report.get("draw_event_count")
    transition_count = report.get("transition_count")
    initial_draw = initial_state.get("draw_count")
    final_draw = final_state.get("draw_count")
    stop_ns = control.get("stop_requested_perf_counter_ns")
    armed_update = observer.get("armed_at_framework_update")
    if (
        not canonical
        or report.get("schema") != FRAMEWORK_POLL_SCHEMA
        or report.get("version") != 1
        or report.get("classification")
        != "diagnostic-only-not-pc-golden"
        or report.get("gate_effect") != "none"
        or report.get("status") != "complete"
        or observer.get("arm_at_framework_update")
        != expected_arm_update
        or observer.get("arm_no_later_than_framework_update")
        != expected_arm_deadline_update
        or isinstance(armed_update, bool)
        or not isinstance(armed_update, int)
        or not expected_arm_update
        <= armed_update
        <= expected_arm_deadline_update
        or observer.get("target_poll_hz") != float(expected_target_hz)
        or observer.get("process_affinity_mask")
        != expected_observer_affinity_mask
        or affinity_topology.get("observer_logical_affinity_mask")
        != expected_observer_affinity_mask
        or affinity_topology.get("game_logical_affinity_mask")
        != expected_game_affinity_mask
        or affinity_topology.get("physical_cores_disjoint") is not True
        or affinity_topology.get("observer_physical_core_logical_mask")
        == affinity_topology.get("game_physical_core_logical_mask")
        or isinstance(observer.get("prearm_sample_count"), bool)
        or not isinstance(observer.get("prearm_sample_count"), int)
        or observer["prearm_sample_count"] <= 0
        or identity.get("process_id") != metadata.process_id
        or identity.get("process_creation_filetime_100ns")
        != metadata.process_creation_filetime_100ns
        or identity.get("executable_sha256")
        != metadata.executable_sha256
        or binding.get("capture_metadata_sha256") != metadata.sha256
        or binding.get("framework_update_map_sha256")
        != _sha256_file(update_map_path)
        or binding.get("framework_state_sidecar_sha256")
        != _sha256_file(state_sidecar_path)
        or binding.get("capture_interval_fully_observed") is not True
        or binding.get("capture_start_perf_counter_ns")
        != metadata.capture_start_perf_counter_ns
        or binding.get("capture_end_perf_counter_ns")
        != metadata.capture_end_perf_counter_ns
        or isinstance(mean_hz, bool)
        or not isinstance(mean_hz, (int, float))
        or not math.isfinite(float(mean_hz))
        or float(mean_hz) < MIN_FRAMEWORK_POLL_HZ
        or isinstance(p99_interval, bool)
        or not isinstance(p99_interval, int)
        or p99_interval <= 0
        or p99_interval > MAX_FRAMEWORK_POLL_P99_INTERVAL_NS
        or isinstance(poll_count, bool)
        or not isinstance(poll_count, int)
        or poll_count <= 0
        or isinstance(draw_event_count, bool)
        or not isinstance(draw_event_count, int)
        or draw_event_count <= 0
        or isinstance(transition_count, bool)
        or not isinstance(transition_count, int)
        or transition_count < draw_event_count
        or report.get("skipped_draw_increment_count") != 0
        or isinstance(initial_draw, bool)
        or not isinstance(initial_draw, int)
        or initial_state.get("update_count") != armed_update
        or isinstance(final_draw, bool)
        or not isinstance(final_draw, int)
        or final_draw - initial_draw != draw_event_count
        or isinstance(stop_ns, bool)
        or not isinstance(stop_ns, int)
        or stop_ns < metadata.capture_end_perf_counter_ns
    ):
        raise CollectionError("framework_state_poll_evidence_invalid")
    return {
        "artifact": path.name,
        "sha256": _sha256_file(path),
        "classification": report["classification"],
        "gate_effect": report["gate_effect"],
        "status": report["status"],
        "arm_at_framework_update": expected_arm_update,
        "arm_no_later_than_framework_update": (
            expected_arm_deadline_update
        ),
        "armed_at_framework_update": armed_update,
        "target_poll_hz": expected_target_hz,
        "observer_affinity_mask": expected_observer_affinity_mask,
        "game_affinity_mask": expected_game_affinity_mask,
        "physical_cores_disjoint": True,
        "poll_count": poll_count,
        "mean_polls_per_second": mean_hz,
        "p99_sample_completion_interval_ns": p99_interval,
        "draw_event_count": draw_event_count,
        "skipped_draw_increment_count": 0,
        "capture_interval_fully_observed": True,
    }


def _run_one(
    *,
    session_root: Path,
    plan: Mapping[str, Any],
    pre: PcStateSnapshot,
    run_id: str,
) -> RunEvidence:
    nonce = plan["session_nonce"]
    run_root = session_root / f"run-{run_id}"
    _new_child_directory(run_root)
    capture_root = run_root / "capture"
    _new_child_directory(capture_root)

    restored = restore_state(pre)
    if restored.state_root != pre.state_root:
        raise CollectionError("run_prestate_restore_mismatch")
    start = capture_state(session_nonce=nonce, phase=f"{run_id}-start")
    if start.state_root != pre.state_root:
        raise CollectionError("run_start_state_mismatch")
    write_snapshot_exclusive(start, run_root / "start.json")

    project_root = Path(plan["project_root"])
    dmo_path = _session_member(session_root, plan["dmo"]["artifact"])
    runtime_source = Path(
        plan["runtime"]["runtime_source_executable"]
    )
    runtime_executable = Path(
        plan["runtime"]["runtime_executable"]
    )
    direct_fixed_seed = (
        plan["runtime"]["launch_mode"]
        == DIRECT_FIXED_SEED_LAUNCH_MODE
    )
    trace_result_path = run_root / "strict-replay.json"
    trace_log_path = run_root / "strict-replay.log"
    capture_log_path = run_root / "capture.log"
    framework_path = run_root / "framework-updates.json"
    framework_state_path = run_root / "framework-state-diagnostic.json"
    framework_poll_path = run_root / "framework-state-poll-diagnostic.json"
    framework_poll_ready_path = run_root / "framework-state-poll-ready.json"
    framework_poll_stop_path = run_root / "framework-state-poll-stop.json"
    framework_poll_log_path = run_root / "framework-state-poll.log"
    trace_args = [
        sys.executable,
        str(project_root / "tools" / "trace_popcap_demo_commands.py"),
        "--dmo",
        str(dmo_path),
        "--steam-exe",
        plan["runtime"]["steam_executable"],
        "--runtime-exe",
        str(runtime_executable),
        "--maximum-hits",
        str(plan["trace"]["maximum_hits"]),
        "--launch-timeout",
        str(plan["trace"]["launch_timeout_seconds"]),
        "--trace-timeout",
        str(plan["trace"]["trace_timeout_seconds"]),
        "--attach-at-update",
        str(plan["trace"]["attach_at_update"]),
        "--detach-at-update",
        str(plan["trace"]["detach_at_update"]),
        "--reattach-at-update",
        str(plan["trace"]["reattach_at_update"]),
        "--attach-timeout",
        "120",
        "--broker-service-blocks",
        "--service-wait-timeout",
        str(STRICT_REPLAY_SERVICE_WAIT_TIMEOUT_SECONDS),
        "--quiet-nonservice",
        "--progress-every-updates",
        "1000",
        "--accept-normal-exit",
        "--result-json",
        str(trace_result_path),
    ]
    for row_index in plan["dmo"].get(
        "diagnostic_successful_file_write_padding_rows",
        [],
    ):
        trace_args.extend(
            ["--successful-file-write-padding-row", str(row_index)]
        )
    if plan["trace"]["allow_pre_stream_commands"]:
        trace_args.append("--allow-pre-stream-commands")
    if plan["trace"].get("startup_trace_handoff", False):
        trace_args.append("--startup-trace-handoff")
    if plan["trace"].get("allow_attach_stabilization", False):
        trace_args.append("--allow-attach-stabilization")
    if plan["trace"].get("allow_pre_attach_file_write_debt", False):
        trace_args.append("--allow-pre-attach-file-write-debt")
    if plan["trace"].get(
        "allow_font_cache_manifest_completion_debt",
        False,
    ):
        trace_args.append(
            "--allow-font-cache-manifest-completion-debt"
        )
    if plan["trace"].get("seed_board_before_attach", False):
        trace_args.append("--seed-board-before-attach")
    if plan["trace"].get(
        "allow_post_blackout_attach_stabilization",
        False,
    ):
        trace_args.append(
            "--allow-post-blackout-attach-stabilization"
        )
    if plan["trace"].get(
        "allow_blackout_file_write_order_rebase",
        False,
    ):
        trace_args.append(
            "--allow-blackout-file-write-order-rebase"
        )
    blackout_expected_command_order_offset = plan["trace"].get(
        "blackout_expected_command_order_offset"
    )
    blackout_expected_native_timeline_offset = plan["trace"].get(
        "blackout_expected_native_timeline_offset"
    )
    if blackout_expected_command_order_offset is not None:
        trace_args.extend(
            [
                "--blackout-expected-command-order-offset",
                str(blackout_expected_command_order_offset),
                "--blackout-expected-native-timeline-offset",
                str(blackout_expected_native_timeline_offset),
            ]
        )
    for offset_pair in plan["trace"].get(
        "blackout_allowed_offset_pairs", []
    ):
        trace_args.extend(
            [
                "--blackout-allowed-offset-pair",
                (
                    f"{offset_pair['command_order_offset']}:"
                    f"{offset_pair['native_timeline_offset']}"
                ),
            ]
        )
    if plan["trace"].get(
        "allow_post_blackout_overdue_idle_reentry",
        False,
    ):
        trace_args.append(
            "--allow-post-blackout-overdue-idle-reentry"
        )
    if plan["trace"].get("close_after_terminal_command", False):
        trace_args.append("--close-after-terminal-command")
    startup_priority_bias = plan["trace"].get(
        "startup_priority_bias_until_update"
    )
    if startup_priority_bias is not None:
        trace_args.extend(
            [
                "--startup-priority-bias-until-update",
                str(startup_priority_bias),
            ]
        )
    startup_process_affinity = plan["trace"].get(
        "startup_process_affinity_mask"
    )
    if startup_process_affinity is not None:
        trace_args.extend(
            [
                "--startup-process-affinity-mask",
                str(startup_process_affinity),
            ]
        )
    if direct_fixed_seed:
        trace_args.extend(
            [
                "--direct-runtime-exe",
                str(runtime_executable),
                "--changedir",
                plan["runtime"]["changedir"],
                "--crt-rand-seed",
                str(plan["runtime"]["crt_rand_seed"]),
                "--startup-seed-transport",
                plan["runtime"].get(
                    "startup_seed_transport",
                    "debugger_register",
                ),
            ]
        )
        if plan["runtime"]["board_seed"] is not None:
            trace_args.extend(
                [
                    "--board-seed-address",
                    str(
                        plan["runtime"][
                            "board_seed_call_address"
                        ]
                    ),
                    "--board-seed",
                    str(plan["runtime"]["board_seed"]),
                ]
            )
            if plan["runtime"]["global_rng_seed"] is not None:
                trace_args.extend(
                    [
                        "--global-rng-seed",
                        str(plan["runtime"]["global_rng_seed"]),
                    ]
                )
            if plan["runtime"]["thread_crt_rng_seed"] is not None:
                trace_args.extend(
                    [
                        "--thread-crt-rng-seed",
                        str(
                            plan["runtime"][
                                "thread_crt_rng_seed"
                            ]
                        ),
                    ]
                )
    source_bound_anchor = plan["trace"].get(
        "source_bound_board_anchor"
    )
    if source_bound_anchor is not None:
        trace_args.extend(
            [
                "--source-bound-board-monitor",
                source_bound_anchor["monitor_path"],
                "--source-bound-board-trace",
                source_bound_anchor["trace_path"],
                "--source-bound-board-recording-report",
                source_bound_anchor["recording_report_path"],
                "--source-bound-board-monitor-update",
                str(source_bound_anchor["monitor_framework_update"]),
                "--source-bound-board-seed",
                str(source_bound_anchor["global_seed"]),
                "--source-bound-board-rewind-draws",
                str(source_bound_anchor["rewind_draws"]),
                "--source-bound-board-source-order",
                str(source_bound_anchor["source_order"]),
                "--source-bound-board-framework-update",
                str(source_bound_anchor["framework_update"]),
                "--source-bound-board-caller",
                hex(source_bound_anchor["caller"]),
            ]
        )
        if plan["trace"].get(
            "allow_source_bound_board_global_correction",
            False,
        ):
            trace_args.append(
                "--allow-source-bound-board-global-correction"
            )
    source_bound_precall = plan["trace"].get(
        "source_bound_board_precall_global_restore"
    )
    if source_bound_precall is not None:
        assert source_bound_anchor is not None
        trace_args.append("--source-bound-board-precall-global-restore")
    trace_args.extend(
        _gameplay_mtrand_trace_args(
            session_root=session_root,
            run_root=run_root,
            trace=plan["trace"],
        )
    )
    existing = set(process_ids_by_name(runtime_executable.name))
    runtime_pid: int | None = None
    trace_process: subprocess.Popen[bytes] | None = None
    capture_process: subprocess.Popen[bytes] | None = None
    framework_poll_process: subprocess.Popen[bytes] | None = None
    gameplay_mtrand_sync_evidence: dict[str, Any] | None = None
    framework_poll_evidence: Mapping[str, Any] | None = None
    try:
        with trace_log_path.open("xb") as trace_log:
            trace_process = subprocess.Popen(
                trace_args,
                cwd=project_root,
                env=_child_environment(project_root),
                stdin=subprocess.DEVNULL,
                stdout=trace_log,
                stderr=subprocess.STDOUT,
                close_fds=True,
            )
            runtime_pid, observed_start_ns = _wait_for_runtime(
                trace_process,
                existing=existing,
                expected_runtime=runtime_executable,
                timeout=float(plan["trace"]["launch_timeout_seconds"]) + 5,
            )
            if plan["capture"].get(
                "startup_window_activation",
                True,
            ):
                target = _wait_for_capture_window(runtime_pid)
                startup_activation = _activate_startup_window(target)
                startup_activation_path = (
                    run_root / "startup-window-activation.json"
                )
                _write_exclusive(
                    startup_activation_path,
                    _canonical_json(startup_activation),
                )
            else:
                acquisition_update = _wait_for_window_acquisition_update(
                    trace_process=trace_process,
                    pid=runtime_pid,
                    target_update=max(
                        0,
                        plan["capture"]["window_repaint_update"] - 50,
                    ),
                )
                target = _wait_for_capture_window(runtime_pid, timeout=60.0)
                _write_exclusive(
                    run_root / "startup-window-acquisition.json",
                    _canonical_json(
                        {
                            "schema": (
                                "zuma-rl.pc-golden-deferred-window-acquisition"
                            ),
                            "version": 1,
                            "window_target": _window_target_evidence(target),
                            "observed_framework_update": acquisition_update,
                            "external_startup_activation": False,
                            "recorded_activate_app_update": 0,
                            "persistent_file_modified": False,
                        }
                    ),
                )
            if plan["capture"].get(
                "framework_state_poll_diagnostic", False
            ):
                framework_poll_args = [
                    sys.executable,
                    str(
                        project_root
                        / "tools"
                        / "poll_pc_framework_state.py"
                    ),
                    "--pid",
                    str(runtime_pid),
                    "--process-name",
                    runtime_executable.name,
                    "--expected-executable-sha256",
                    plan["runtime"]["expected_runtime_sha256"],
                    "--affinity-mask",
                    str(
                        plan["capture"][
                            "framework_state_poll_affinity_mask"
                        ]
                    ),
                    "--game-process-affinity-mask",
                    str(plan["trace"]["startup_process_affinity_mask"]),
                    "--arm-at-framework-update",
                    str(
                        plan["capture"].get(
                            "framework_state_poll_arm_update",
                            max(
                                0,
                                plan["capture"][
                                    "wait_until_framework_update"
                                ]
                                - 5,
                            ),
                        )
                    ),
                    "--arm-no-later-than-framework-update",
                    str(
                        plan["capture"].get(
                            "framework_state_poll_arm_deadline_update",
                            plan["capture"].get(
                                "framework_state_poll_arm_update",
                                max(
                                    0,
                                    plan["capture"][
                                        "wait_until_framework_update"
                                    ]
                                    - 5,
                                ),
                            ),
                        )
                    ),
                    "--target-poll-hz",
                    str(
                        plan["capture"].get(
                            "framework_state_poll_target_hz", 5_000
                        )
                    ),
                    "--maximum-duration-seconds",
                    "120",
                    "--output",
                    str(framework_poll_path),
                    "--ready-marker",
                    str(framework_poll_ready_path),
                    "--stop-marker",
                    str(framework_poll_stop_path),
                    "--capture-metadata",
                    str(capture_root / "metadata.json"),
                    "--framework-update-map",
                    str(framework_path),
                    "--framework-state-sidecar",
                    str(framework_state_path),
                ]
                with framework_poll_log_path.open("xb") as poll_log:
                    framework_poll_process = subprocess.Popen(
                        framework_poll_args,
                        cwd=project_root,
                        env=_child_environment(project_root),
                        stdin=subprocess.DEVNULL,
                        stdout=poll_log,
                        stderr=subprocess.STDOUT,
                        close_fds=True,
                    )
                _wait_for_framework_poll_ready(
                    framework_poll_process,
                    framework_poll_ready_path,
                    expected_process_id=int(runtime_pid),
                    expected_executable_sha256=plan["runtime"][
                        "expected_runtime_sha256"
                    ],
                )
            repaint_guard = plan["capture"]["window_repaint_guard"]
            repaint = _wait_for_repaint_guard(
                target=target,
                trace_process=trace_process,
                trigger_update=plan["capture"][
                    "window_repaint_update"
                ],
                previous_effectful_command_update=repaint_guard[
                    "previous_effectful_command_update"
                ],
                next_effectful_command_update=repaint_guard[
                    "next_effectful_command_update"
                ],
                guarded_idle_command_count=repaint_guard[
                    "guarded_idle_command_count"
                ],
                # Full command tracing is intentionally much slower than the
                # native game.  The immutable trace deadline is the enclosing
                # upper bound; a shorter fixed repaint deadline can terminate
                # an otherwise healthy replay before the capture blackout.
                timeout=float(plan["trace"]["trace_timeout_seconds"]),
            )
            repaint_path = run_root / "window-repaint.json"
            repaint_bytes = _canonical_json(repaint)
            _write_exclusive(repaint_path, repaint_bytes)
            repaint_sha256 = _sha256_bytes(repaint_bytes)
            capture_args = [
                sys.executable,
                str(project_root / "tools" / "capture_dxgi.py"),
                "--output-dir",
                str(capture_root),
                "--duration-seconds",
                str(plan["capture"]["duration_seconds"]),
                "--frame-budget-fps",
                str(plan["capture"]["frame_budget_fps"]),
                "--process",
                runtime_executable.name,
                "--device-index",
                str(plan["capture"]["device_index"]),
                "--output-index",
                str(plan["capture"]["output_index"]),
                "--framework-update-output",
                str(framework_path),
                "--wait-until-framework-update",
                str(plan["capture"]["wait_until_framework_update"]),
            ]
            if plan["capture"].get("framework_state_diagnostic", False):
                capture_args.extend(
                    [
                        "--framework-state-output",
                        str(framework_state_path),
                    ]
                )
            if not direct_fixed_seed:
                capture_args.extend(
                    [
                        "--runtime-source-executable",
                        str(runtime_source),
                    ]
                )
            with capture_log_path.open("xb") as capture_log:
                capture_process = subprocess.Popen(
                    capture_args,
                    cwd=project_root,
                    env=_child_environment(project_root),
                    stdin=subprocess.DEVNULL,
                    stdout=capture_log,
                    stderr=subprocess.STDOUT,
                    close_fds=True,
                )
                capture_deadline = time.monotonic() + 240
                while True:
                    capture_return_code = capture_process.poll()
                    trace_early_return_code = trace_process.poll()
                    if capture_return_code is not None:
                        break
                    if trace_early_return_code is not None:
                        capture_process.terminate()
                        try:
                            capture_process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            capture_process.kill()
                            capture_process.wait(timeout=10)
                        raise CollectionError("strict_replay_failed")
                    if time.monotonic() >= capture_deadline:
                        raise CollectionError("child_process_timeout")
                    time.sleep(0.05)
            if framework_poll_process is not None:
                _signal_framework_poll_stop(
                    framework_poll_stop_path,
                    process_id=int(runtime_pid),
                )
                try:
                    framework_poll_return_code = (
                        framework_poll_process.wait(timeout=15)
                    )
                except subprocess.TimeoutExpired:
                    framework_poll_process.terminate()
                    try:
                        framework_poll_process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        framework_poll_process.kill()
                        framework_poll_process.wait(timeout=5)
                    raise CollectionError(
                        "framework_state_poll_timeout"
                    ) from None
            if capture_return_code != 0:
                raise CollectionError("dxgi_capture_failed")
            if (
                framework_poll_process is not None
                and framework_poll_return_code != 0
            ):
                raise CollectionError("framework_state_poll_failed")
            trace_return_code = trace_process.wait(
                timeout=float(plan["trace"]["trace_timeout_seconds"]) + 30
            )
        if trace_return_code != 0:
            raise CollectionError("strict_replay_failed")
    except subprocess.TimeoutExpired:
        raise CollectionError("child_process_timeout") from None
    finally:
        if capture_process is not None and capture_process.poll() is None:
            capture_process.terminate()
            try:
                capture_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                capture_process.kill()
                capture_process.wait(timeout=10)
        if (
            framework_poll_process is not None
            and framework_poll_process.poll() is None
        ):
            if runtime_pid is not None:
                _signal_framework_poll_stop(
                    framework_poll_stop_path,
                    process_id=int(runtime_pid),
                )
            try:
                framework_poll_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                framework_poll_process.terminate()
                try:
                    framework_poll_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    framework_poll_process.kill()
                    framework_poll_process.wait(timeout=5)
        if runtime_pid is not None:
            _terminate_exact_runtime(runtime_pid, runtime_executable)
        if trace_process is not None and trace_process.poll() is None:
            trace_process.terminate()
            try:
                trace_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                trace_process.kill()
                trace_process.wait(timeout=10)

    _wait_game_stopped()
    gameplay_mtrand_sync = plan["trace"].get("gameplay_mtrand_sync")
    if gameplay_mtrand_sync is not None:
        gameplay_mtrand_sync_evidence = (
            _load_gameplay_mtrand_sync_receipt(
                run_root / "gameplay-mtrand-sync.json",
                expected_process_id=int(runtime_pid),
                expected_oracle=_session_member(
                    session_root,
                    gameplay_mtrand_sync["artifact"],
                ),
                expected_dmo=dmo_path,
                artifact=(
                    f"run-{run_id}/gameplay-mtrand-sync.json"
                ),
            )
        )
    trace_payload = _validate_trace_result(
        trace_result_path,
        expected_pid=int(runtime_pid),
        expected_dmo_sha256=plan["dmo"]["sha256"],
        expected_attach_at_update=plan["trace"]["attach_at_update"],
        expected_detach_at_update=plan["trace"]["detach_at_update"],
        expected_reattach_at_update=plan["trace"][
            "reattach_at_update"
        ],
        expected_allow_pre_stream_commands=plan["trace"][
            "allow_pre_stream_commands"
        ],
        expected_startup_trace_handoff=plan["trace"].get(
            "startup_trace_handoff",
            False,
        ),
        expected_allow_attach_stabilization=plan["trace"].get(
            "allow_attach_stabilization",
            False,
        ),
        expected_allow_pre_attach_file_write_debt=plan["trace"].get(
            "allow_pre_attach_file_write_debt",
            False,
        ),
        expected_allow_font_cache_manifest_completion_debt=plan[
            "trace"
        ].get(
            "allow_font_cache_manifest_completion_debt",
            False,
        ),
        expected_font_cache_manifest_receipt=plan["trace"].get(
            "font_cache_manifest_receipt"
        ),
        expected_seed_board_before_attach=plan["trace"].get(
            "seed_board_before_attach",
            False,
        ),
        expected_startup_priority_bias_until_update=plan["trace"].get(
            "startup_priority_bias_until_update"
        ),
        expected_startup_process_affinity_mask=plan["trace"].get(
            "startup_process_affinity_mask"
        ),
        expected_crt_rand_seed=plan["runtime"]["crt_rand_seed"],
        expected_startup_seed_transport=plan["runtime"].get(
            "startup_seed_transport",
            "debugger_register",
        ),
        expected_board_seed_address=plan["runtime"][
            "board_seed_call_address"
        ],
        expected_board_seed=plan["runtime"]["board_seed"],
        expected_global_rng_seed=plan["runtime"][
            "global_rng_seed"
        ],
        expected_thread_crt_rng_seed=plan["runtime"][
            "thread_crt_rng_seed"
        ],
        expected_source_bound_board_anchor=plan["trace"].get(
            "source_bound_board_anchor"
        ),
        expected_source_bound_board_precall_global_restore=plan[
            "trace"
        ].get(
            "source_bound_board_precall_global_restore"
        ),
        expected_allow_source_bound_board_global_correction=plan[
            "trace"
        ].get(
            "allow_source_bound_board_global_correction",
            False,
        ),
        expected_post_blackout_attach_stabilization=plan[
            "trace"
        ].get(
            "allow_post_blackout_attach_stabilization",
            False,
        ),
        expected_blackout_file_write_order_rebase=plan["trace"].get(
            "allow_blackout_file_write_order_rebase",
            False,
        ),
        expected_blackout_file_write_order_rebase_rows=plan[
            "trace"
        ].get(
            "blackout_file_write_order_rebase_rows",
            [],
        ),
        expected_blackout_file_write_rebase_mode=plan["trace"].get(
            "blackout_file_write_rebase_mode"
        ),
        expected_blackout_command_order_offset=plan["trace"].get(
            "blackout_expected_command_order_offset"
        ),
        expected_blackout_native_timeline_offset=plan["trace"].get(
            "blackout_expected_native_timeline_offset"
        ),
        expected_blackout_allowed_offset_pairs=plan["trace"].get(
            "blackout_allowed_offset_pairs", []
        ),
        expected_allow_post_blackout_overdue_idle_reentry=plan[
            "trace"
        ].get(
            "allow_post_blackout_overdue_idle_reentry",
            False,
        ),
        expected_post_blackout_overdue_idle_reentry_candidates=plan[
            "trace"
        ].get(
            "post_blackout_overdue_idle_reentry_candidates",
            [],
        ),
        expected_close_after_terminal_command=plan["trace"].get(
            "close_after_terminal_command",
            False,
        ),
        expected_terminal_close_command=plan["trace"].get(
            "terminal_close_command"
        ),
        expected_diagnostic_successful_file_write_padding_rows=plan[
            "dmo"
        ].get(
            "diagnostic_successful_file_write_padding_rows",
            [],
        ),
    )
    try:
        metadata = PcDxgiCaptureMetadata.read(
            capture_root / "metadata.json"
        )
        updates = PcFrameworkUpdateMap.read(framework_path)
    except Exception:
        raise CollectionError("capture_evidence_invalid") from None
    if (
        metadata.process_id != runtime_pid
        or updates.process_instance != metadata.process_instance
        or updates.executable_sha256 != metadata.executable_sha256
        or updates.capture_metadata_sha256 != metadata.sha256
        or metadata.executable_sha256
        != plan["runtime"]["expected_runtime_sha256"]
        or (metadata.width, metadata.height)
        != (
            plan["capture"]["expected_width"],
            plan["capture"]["expected_height"],
        )
    ):
        raise CollectionError("capture_identity_mismatch")
    if plan["capture"].get(
        "framework_state_poll_diagnostic", False
    ):
        framework_poll_evidence = _validate_framework_poll_evidence(
            framework_poll_path,
            metadata=metadata,
            update_map_path=framework_path,
            state_sidecar_path=framework_state_path,
            expected_arm_update=plan["capture"].get(
                "framework_state_poll_arm_update",
                max(
                    0,
                    plan["capture"]["wait_until_framework_update"] - 5,
                ),
            ),
            expected_arm_deadline_update=plan["capture"].get(
                "framework_state_poll_arm_deadline_update",
                plan["capture"].get(
                    "framework_state_poll_arm_update",
                    max(
                        0,
                        plan["capture"]["wait_until_framework_update"] - 5,
                    ),
                ),
            ),
            expected_target_hz=plan["capture"].get(
                "framework_state_poll_target_hz", 5_000
            ),
            expected_observer_affinity_mask=plan["capture"].get(
                "framework_state_poll_affinity_mask", 4
            ),
            expected_game_affinity_mask=plan["trace"][
                "startup_process_affinity_mask"
            ],
        )
    trace_start = int(trace_payload["trace_started_perf_counter_ns"])
    trace_stop = int(trace_payload["trace_finished_perf_counter_ns"])
    debugger_blackout = trace_payload["debugger_blackout"]
    if not (
        debugger_blackout["started_perf_counter_ns"]
        < metadata.capture_start_perf_counter_ns
        < metadata.capture_end_perf_counter_ns
        < debugger_blackout["finished_perf_counter_ns"]
    ):
        raise CollectionError("capture_debugger_blackout_mismatch")
    process_start_ns = observed_start_ns
    _validate_run_timeline(
        start_snapshot_ns=start.captured_perf_counter_ns,
        observed_process_start_ns=process_start_ns,
        capture_start_ns=metadata.capture_start_perf_counter_ns,
        capture_end_ns=metadata.capture_end_perf_counter_ns,
        trace_start_ns=trace_start,
        trace_stop_ns=trace_stop,
    )

    end = capture_state(session_nonce=nonce, phase=f"{run_id}-end")
    write_snapshot_exclusive(end, run_root / "end.json")
    if trace_stop >= end.captured_perf_counter_ns:
        raise CollectionError("run_end_snapshot_order_invalid")
    end_gameplay_state_root = _projected_end_state_root(
        end,
        pre_snapshot=pre,
        process_id=int(runtime_pid),
        launch_mode=plan["runtime"]["launch_mode"],
    )
    return RunEvidence(
        run_id=run_id,
        start_snapshot=start,
        end_snapshot=end,
        process_id=metadata.process_id,
        process_creation_filetime_100ns=(
            metadata.process_creation_filetime_100ns
        ),
        executable_sha256=metadata.executable_sha256,
        process_start_perf_counter_ns=process_start_ns,
        process_stop_perf_counter_ns=trace_stop,
        exit_code=0,
        capture_metadata=metadata,
        framework_updates=updates,
        window_repaint_sha256=repaint_sha256,
        end_gameplay_state_root=end_gameplay_state_root,
        gameplay_mtrand_sync=gameplay_mtrand_sync_evidence,
        framework_state_poll=framework_poll_evidence,
    )


def _build_protocol_controls(
    *,
    session_root: Path,
    nonce: str,
    pre: PcStateSnapshot,
    runs: tuple[RunEvidence, ...],
    restored: PcStateSnapshot,
) -> Mapping[str, Any]:
    snapshots: list[tuple[str, str, PcStateSnapshot, RunEvidence | None]] = [
        ("pre", "save.pre", pre, None)
    ]
    for run in runs:
        snapshots.extend(
            (
                (
                    f"{run.run_id}-start",
                    f"save.{run.run_id}.start",
                    run.start_snapshot,
                    run,
                ),
                (
                    f"{run.run_id}-end",
                    f"save.{run.run_id}.end",
                    run.end_snapshot,
                    run,
                ),
            )
        )
    snapshots.append(("restored", "save.restored", restored, None))
    journal_records: list[dict[str, Any]] = []
    for phase, artifact_name, snapshot, run in snapshots:
        is_end = phase.endswith("-end")
        journal_records.append(
            {
                "phase": phase,
                "snapshot_artifact": artifact_name,
                "snapshot_state_root": snapshot.state_root,
                "snapshot_perf_counter_ns": (
                    snapshot.captured_perf_counter_ns
                ),
                "process_id": None if run is None else run.process_id,
                "process_creation_filetime_100ns": (
                    None
                    if run is None
                    else run.process_creation_filetime_100ns
                ),
                "exit_code": run.exit_code if is_end and run else None,
            }
        )
    journal = PcSaveJournal.build(
        session_nonce=nonce,
        records=journal_records,
    )
    journal_path = session_root / "protocol" / "save-journal.ndjson"
    _write_exclusive(journal_path, journal.to_ndjson().encode("ascii"))

    process_events: list[PcProcessEvent] = []
    for run in runs:
        process_events.extend(
            (
                PcProcessEvent(
                    sequence=len(process_events),
                    kind="start",
                    process_id=run.process_id,
                    process_creation_filetime_100ns=(
                        run.process_creation_filetime_100ns
                    ),
                    perf_counter_ns=run.process_start_perf_counter_ns,
                    exit_code=None,
                    executable_sha256=run.executable_sha256,
                ),
                PcProcessEvent(
                    sequence=len(process_events) + 1,
                    kind="stop",
                    process_id=run.process_id,
                    process_creation_filetime_100ns=(
                        run.process_creation_filetime_100ns
                    ),
                    perf_counter_ns=run.process_stop_perf_counter_ns,
                    exit_code=run.exit_code,
                    executable_sha256=run.executable_sha256,
                ),
            )
        )
    timeline = PcProcessTimeline(
        session_nonce=nonce,
        events=tuple(process_events),
    )
    timeline_path = session_root / "protocol" / "process-timeline.bin"
    _write_exclusive(timeline_path, timeline.to_bytes())
    return {
        "journal_artifact": "protocol/save-journal.ndjson",
        "journal_sha256": _sha256_file(journal_path),
        "process_timeline_artifact": "protocol/process-timeline.bin",
        "process_timeline_sha256": _sha256_file(timeline_path),
        "pre_snapshot_artifact": "protocol/pre.json",
        "restored_snapshot_artifact": "protocol/restored.json",
        "run_end_gameplay_state_root": (
            runs[0].end_gameplay_state_root
        ),
    }


def _restore_host_with_retry(host_pre: PcStateSnapshot) -> None:
    last_error: Exception | None = None
    for _ in range(3):
        try:
            _require_game_stopped()
            restored = restore_state(host_pre)
            if restored.state_root != host_pre.state_root:
                raise CollectionError("host_restore_root_mismatch")
            return
        except (CollectionError, StateTransactionError) as error:
            last_error = error
            time.sleep(0.25)
    raise CollectionError("host_restore_failed") from last_error


def _wait_inter_run_cooldown(
    seconds: float,
    *,
    delay: Any = time.sleep,
    clock: Any = time.monotonic,
    require_game_stopped: Any = _require_game_stopped,
) -> None:
    """Wait out one preregistered external presentation cooldown."""

    if (
        isinstance(seconds, bool)
        or not isinstance(seconds, (int, float))
        or not math.isfinite(float(seconds))
        or not 0.0 <= float(seconds) <= MAX_INTER_RUN_COOLDOWN_SECONDS
    ):
        raise CollectionError("inter_run_cooldown_invalid")
    require_game_stopped()
    deadline = clock() + float(seconds)
    while True:
        remaining = deadline - clock()
        if remaining <= 0:
            break
        delay(min(1.0, remaining))
        require_game_stopped()


def collect_session(args: argparse.Namespace) -> Mapping[str, Any]:
    _require_windows()
    session_root = _ascii_absolute(args.session_root, "session_root")
    if not session_root.is_dir():
        raise CollectionError("session_directory_unavailable")
    plan = _read_plan(session_root)
    if (session_root / "collection.json").exists():
        raise CollectionError("session_already_collected")
    _write_exclusive(
        session_root / "collection.lock",
        _canonical_json(
            {
                "schema": "zuma-rl.pc-golden-v4-collection-lock",
                "version": 1,
                "session_nonce": plan["session_nonce"],
            }
        ),
    )
    _require_game_stopped()
    host_pre_path = _session_member(
        session_root,
        plan["safety"]["host_pre_artifact"],
    )
    template_path = _session_member(
        session_root,
        plan["prestate"]["template_artifact"],
    )
    host_pre = PcStateSnapshot.read(host_pre_path)
    template = PcStateSnapshot.read(template_path)
    dmo_provenance_identity_valid = True
    provenance_artifact = plan["dmo"].get("provenance_artifact")
    if provenance_artifact is not None:
        try:
            provenance_bytes, padding_rows = (
                _validate_dmo_padding_provenance(
                    _session_member(
                        session_root,
                        plan["dmo"]["artifact"],
                    ),
                    _session_member(session_root, provenance_artifact),
                )
            )
            dmo_provenance_identity_valid = (
                _sha256_bytes(provenance_bytes)
                == plan["dmo"].get("provenance_sha256")
                and list(padding_rows)
                == plan["dmo"].get(
                    "diagnostic_successful_file_write_padding_rows"
                )
            )
        except CollectionError:
            dmo_provenance_identity_valid = False
    if (
        host_pre.session_nonce != plan["session_nonce"]
        or template.session_nonce != plan["session_nonce"]
        or host_pre.state_root != plan["safety"]["host_pre_state_root"]
        or template.state_root != plan["prestate"]["template_state_root"]
        or _sha256_file(
            _session_member(session_root, plan["dmo"]["artifact"])
        )
        != plan["dmo"]["sha256"]
        or _sha256_file(
            Path(plan["runtime"]["runtime_source_executable"])
        )
        != plan["runtime"]["runtime_source_sha256"]
        or (
            plan["runtime"]["launch_mode"]
            == DIRECT_FIXED_SEED_LAUNCH_MODE
            and _sha256_file(
                Path(plan["runtime"]["runtime_executable"])
            )
            != plan["runtime"]["runtime_executable_sha256"]
        )
        or plan.get("state_comparison")
        != _state_comparison_contract(plan["runtime"]["launch_mode"])
        or not dmo_provenance_identity_valid
    ):
        raise CollectionError("prepared_artifact_identity_mismatch")
    live_precheck = capture_state(
        session_nonce=plan["session_nonce"],
        phase="host-precheck",
    )
    if live_precheck.state_root != host_pre.state_root:
        raise CollectionError("host_state_changed_since_prepare")

    result: Mapping[str, Any] | None = None
    primary_error: CollectionError | None = None
    try:
        restore_state(template)
        pre = capture_state(
            session_nonce=plan["session_nonce"],
            phase="pre",
        )
        if pre.state_root != template.state_root:
            raise CollectionError("protocol_prestate_mismatch")
        write_snapshot_exclusive(pre, session_root / "protocol" / "pre.json")
        run_evidence_list: list[RunEvidence] = []
        for run_index, run_id in enumerate(RUN_IDS):
            if run_index:
                between_runs = restore_state(pre)
                if between_runs.state_root != pre.state_root:
                    raise CollectionError(
                        "inter_run_prestate_restore_mismatch"
                    )
                _wait_inter_run_cooldown(
                    float(
                        plan["capture"].get(
                            "inter_run_cooldown_seconds",
                            0.0,
                        )
                    )
                )
            run_evidence_list.append(
                _run_one(
                session_root=session_root,
                plan=plan,
                pre=pre,
                run_id=run_id,
            )
            )
        run_evidence = tuple(run_evidence_list)
        if (
            run_evidence[0].end_gameplay_state_root
            != run_evidence[1].end_gameplay_state_root
        ):
            raise CollectionError(
                "independent_run_gameplay_states_differ"
            )
        restore_state(pre)
        restored = capture_state(
            session_nonce=plan["session_nonce"],
            phase="restored",
        )
        write_snapshot_exclusive(
            restored,
            session_root / "protocol" / "restored.json",
        )
        if restored.state_root != pre.state_root:
            raise CollectionError("protocol_restore_root_mismatch")
        controls = _build_protocol_controls(
            session_root=session_root,
            nonce=plan["session_nonce"],
            pre=pre,
            runs=run_evidence,
            restored=restored,
        )
        result = {
            "schema": RESULT_SCHEMA,
            "version": RESULT_VERSION,
            "status": "complete",
            "session_nonce": plan["session_nonce"],
            "protocol_pre_state_root": pre.state_root,
            "protocol_restored_state_root": restored.state_root,
            "host_pre_state_root": host_pre.state_root,
            "state_comparison": plan["state_comparison"],
            "runs": [
                {
                    "run_id": run.run_id,
                    "process_id": run.process_id,
                    "process_creation_filetime_100ns": (
                        run.process_creation_filetime_100ns
                    ),
                    "executable_sha256": run.executable_sha256,
                    "start_snapshot_artifact": f"run-{run.run_id}/start.json",
                    "end_snapshot_artifact": f"run-{run.run_id}/end.json",
                    "end_state_root": run.end_snapshot.state_root,
                    "end_gameplay_state_root": (
                        run.end_gameplay_state_root
                    ),
                    "capture_metadata_artifact": (
                        f"run-{run.run_id}/capture/metadata.json"
                    ),
                    "framework_update_artifact": (
                        f"run-{run.run_id}/framework-updates.json"
                    ),
                    "window_repaint_artifact": (
                        f"run-{run.run_id}/window-repaint.json"
                    ),
                    "window_repaint_sha256": (
                        run.window_repaint_sha256
                    ),
                    "gameplay_mtrand_sync": run.gameplay_mtrand_sync,
                    "framework_state_poll": (
                        None
                        if run.framework_state_poll is None
                        else {
                            **run.framework_state_poll,
                            "artifact": (
                                f"run-{run.run_id}/"
                                "framework-state-poll-diagnostic.json"
                            ),
                        }
                    ),
                    "frame_count": run.capture_metadata.frame_count,
                }
                for run in run_evidence
            ],
            "protocol_controls": controls,
        }
    except CollectionError as error:
        primary_error = error
    except Exception:
        primary_error = CollectionError("unexpected_collection_failure")
    finally:
        restore_error: CollectionError | None = None
        try:
            if any(_game_processes().values()):
                expected_runtime = Path(
                    plan["runtime"]["runtime_executable"]
                )
                for pid in process_ids_by_name(DEFAULT_RUNTIME_EXE.name):
                    _terminate_exact_runtime(pid, expected_runtime)
                runtime_source = Path(
                    plan["runtime"]["runtime_source_executable"]
                )
                for pid in process_ids_by_name(runtime_source.name):
                    _terminate_exact_process(pid, runtime_source)
                _wait_game_stopped()
            _restore_host_with_retry(host_pre)
            host_restored = capture_state(
                session_nonce=plan["session_nonce"],
                phase="host-restored",
            )
            if host_restored.state_root != host_pre.state_root:
                raise CollectionError("host_restore_root_mismatch")
            host_restored_path = (
                session_root / "safety" / "host-restored.json"
            )
            if not host_restored_path.exists():
                write_snapshot_exclusive(host_restored, host_restored_path)
        except CollectionError as error:
            restore_error = error
        if restore_error is not None:
            raise restore_error

    if primary_error is not None:
        failure = {
            "schema": RESULT_SCHEMA,
            "version": RESULT_VERSION,
            "status": "failed",
            "session_nonce": plan["session_nonce"],
            "failure_code": primary_error.code,
            "host_restored_state_root": host_pre.state_root,
        }
        failure_path = session_root / "collection-failure.json"
        if not failure_path.exists():
            _write_exclusive(failure_path, _canonical_json(failure))
        raise primary_error
    assert result is not None
    final_result = {
        **result,
        "host_restored_state_root": host_pre.state_root,
    }
    _write_exclusive(
        session_root / "collection.json",
        _canonical_json(final_result),
    )
    return final_result


def _positive_float(text: str) -> float:
    value = float(text)
    if not (value > 0):
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _inter_run_cooldown_seconds(text: str) -> float:
    value = float(text)
    if (
        not math.isfinite(value)
        or not 0.0 <= value <= MAX_INTER_RUN_COOLDOWN_SECONDS
    ):
        raise argparse.ArgumentTypeError(
            "inter-run cooldown must be between 0 and 180 seconds"
        )
    return value


def _capture_duration(text: str) -> float:
    try:
        return parse_duration(text)
    except CaptureError:
        raise argparse.ArgumentTypeError("invalid capture duration") from None


def _capture_frame_budget_fps(text: str) -> int:
    try:
        return parse_frame_budget_fps(text)
    except CaptureError:
        raise argparse.ArgumentTypeError(
            "invalid capture frame budget"
        ) from None


def _positive_int(text: str) -> int:
    value = int(text, 0)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _non_negative_int(text: str) -> int:
    value = int(text, 0)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return value


def _blackout_offset_pair(text: str) -> tuple[int, int]:
    try:
        command_text, timeline_text = text.split(":", 1)
        command_offset = int(command_text, 0)
        timeline_offset = int(timeline_text, 0)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(
            "offset pair must be COMMAND:TIMELINE"
        ) from None
    if command_offset < 0 or timeline_offset < 0:
        raise argparse.ArgumentTypeError(
            "offset pair values must be non-negative"
        )
    return command_offset, timeline_offset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare or collect a fail-closed PC Golden v4 session."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--session-root", required=True, type=Path)
    prepare.add_argument("--session-nonce", required=True)
    prepare.add_argument("--dmo", type=Path, default=DEFAULT_DMO)
    prepare.add_argument(
        "--dmo-provenance",
        type=Path,
        help=(
            "Signed diagnostic DMO transformation sidecar; copied and "
            "revalidated before every collection."
        ),
    )
    prepare.add_argument(
        "--prestate-dir",
        type=Path,
        default=DEFAULT_PRESTATE,
    )
    prepare.add_argument(
        "--runtime-screen-mode",
        type=int,
        choices=tuple(sorted(RUNTIME_SCREEN_MODES)),
        help=(
            "Transactionally overlay the existing retail ScreenMode DWORD "
            "inside the protocol prestate (0=windowed, 1=fullscreen); the "
            "original host value is restored in the collector finally path."
        ),
    )
    prepare.add_argument(
        "--runtime-source-executable",
        type=Path,
        default=DEFAULT_RUNTIME_SOURCE,
    )
    prepare.add_argument(
        "--direct-runtime-executable",
        type=Path,
        help=(
            "Use the extracted byte-identical retail payload with a fixed "
            "startup C RNG seed."
        ),
    )
    prepare.add_argument(
        "--changedir",
        type=Path,
        help="Original game asset directory for direct runtime mode.",
    )
    prepare.add_argument(
        "--crt-rand-seed",
        type=_non_negative_int,
        help=(
            "Fixed 32-bit C RNG seed; defaults to the DMO framework seed "
            "when direct runtime mode is selected."
        ),
    )
    prepare.add_argument(
        "--startup-seed-transport",
        choices=tuple(sorted(STARTUP_SEED_TRANSPORTS)),
        help=(
            "Use the temporary IAT stub to avoid a startup debugger, or "
            "the legacy debugger register override."
        ),
    )
    prepare.add_argument(
        "--board-seed-address",
        type=_positive_int,
        default=DEFAULT_BOARD_RESEED_CALL,
        help="Retail CALL instruction that seeds the board MTRand.",
    )
    prepare.add_argument(
        "--board-seed",
        type=_non_negative_int,
        help=(
            "Fixed 32-bit board MTRand seed for direct runtime mode."
        ),
    )
    prepare.add_argument(
        "--global-rng-seed",
        type=_non_negative_int,
        help=(
            "Reset the global framework MTRand at the board-seed "
            "breakpoint; requires --board-seed."
        ),
    )
    prepare.add_argument(
        "--thread-crt-rng-seed",
        type=_non_negative_int,
        help=(
            "Reset the paused board thread's CRT rand state at the "
            "board-seed breakpoint; requires --board-seed."
        ),
    )
    prepare.add_argument("--source-bound-board-monitor", type=Path)
    prepare.add_argument("--source-bound-board-trace", type=Path)
    prepare.add_argument(
        "--source-bound-board-recording-report",
        type=Path,
    )
    prepare.add_argument(
        "--source-bound-board-monitor-update",
        type=_non_negative_int,
    )
    prepare.add_argument(
        "--source-bound-board-seed",
        type=_non_negative_int,
    )
    prepare.add_argument(
        "--source-bound-board-rewind-draws",
        type=_non_negative_int,
    )
    prepare.add_argument(
        "--source-bound-board-source-order",
        type=_non_negative_int,
    )
    prepare.add_argument(
        "--source-bound-board-framework-update",
        type=_non_negative_int,
    )
    prepare.add_argument(
        "--source-bound-board-caller",
        type=_positive_int,
    )
    prepare.add_argument(
        "--source-bound-board-precall-global-restore",
        action="store_true",
        help=(
            "At the exact retail global RNG CALL immediately before the "
            "source-bound Board seed, atomically restore the complete "
            "natural pre-call MTRand state and suspend competing threads "
            "until the Board call."
        ),
    )
    prepare.add_argument(
        "--allow-source-bound-board-global-correction",
        action="store_true",
        help=(
            "At the exact source-bound board constructor only, permit a "
            "bounded same-state-word global MTRand index correction."
        ),
    )
    prepare.add_argument(
        "--gameplay-mtrand-oracle",
        type=Path,
        help=(
            "Verified retail gameplay-return oracle synchronized with "
            "hardware breakpoints before the capture blackout."
        ),
    )
    prepare.add_argument(
        "--gameplay-mtrand-oracle-seed",
        type=_non_negative_int,
        help="Global MTRand seed used to reconstruct the gameplay oracle.",
    )
    prepare.add_argument(
        "--gameplay-mtrand-oracle-maximum-draws",
        type=_positive_int,
        default=100_000,
        help="Maximum seeded draws used to validate the gameplay oracle.",
    )
    prepare.add_argument(
        "--duration-seconds",
        type=_capture_duration,
        default=3.0,
    )
    prepare.add_argument(
        "--frame-budget-fps",
        type=_capture_frame_budget_fps,
        default=240,
    )
    prepare.add_argument(
        "--framework-state-diagnostic",
        action="store_true",
        help=(
            "Record a diagnostic-only per-present SexyApp scheduling-state "
            "sidecar without changing the formal framework-update map."
        ),
    )
    prepare.add_argument(
        "--framework-state-poll-diagnostic",
        action="store_true",
        help=(
            "Run an independent high-rate, diagnostic-only observer around "
            "DXGI capture; this implies --framework-state-diagnostic."
        ),
    )
    prepare.add_argument(
        "--framework-state-poll-affinity-mask",
        type=lambda value: int(value, 0),
        default=4,
        help=(
            "Process affinity mask for the independent high-rate observer "
            "(default: logical CPU 2, mask 0x4)."
        ),
    )
    prepare.add_argument(
        "--framework-state-poll-arm-update",
        type=_non_negative_int,
        help=(
            "Enter controlled high-rate observation only at this exact "
            "stable framework update (default: capture wait update minus 5)."
        ),
    )
    prepare.add_argument(
        "--framework-state-poll-arm-deadline-update",
        type=_non_negative_int,
        help=(
            "Latest accepted stable framework update for bounded observer "
            "arming (default: exact arm update)."
        ),
    )
    prepare.add_argument(
        "--framework-state-poll-target-hz",
        type=_positive_int,
        default=5_000,
        help=(
            "Scheduled read rate for the diagnostic observer; accepted "
            "range is 1000 through 20000 Hz."
        ),
    )
    prepare.add_argument(
        "--inter-run-cooldown-seconds",
        type=_inter_run_cooldown_seconds,
        default=0.0,
        help=(
            "After restoring the frozen prestate, keep the game stopped "
            "for this preregistered interval before launching run r2."
        ),
    )
    prepare.add_argument(
        "--wait-until-framework-update",
        type=_non_negative_int,
        # A prior live diagnostic at update 7617 captured the native 800x600
        # client at a stable ~165 Hz.  Update 9000 can sit in a no-present
        # transition long enough for the strict DXGI warmup to fail.
        default=7617,
    )
    prepare.add_argument(
        "--window-repaint-update",
        type=_non_negative_int,
        # The source DMO has no non-idle command from update 7156 through
        # 7738. Triggering near 7607 leaves a measured effect-free window
        # for foreground activation, a held move, DWM flush, and capture.
        default=7607,
    )
    prepare.add_argument(
        "--skip-startup-window-activation",
        action="store_true",
        help=(
            "Rely on an update-zero recorded activate_app command and defer "
            "window acquisition until the pre-capture idle corridor."
        ),
    )
    prepare.add_argument(
        "--attach-at-update",
        type=_non_negative_int,
        # Attach before the first replay service block so it can be brokered.
        default=230,
    )
    prepare.add_argument(
        "--allow-pre-stream-commands",
        action="store_true",
        help=(
            "Allow the known initial negative-order framework command; "
            "requires --attach-at-update=0."
        ),
    )
    prepare.add_argument(
        "--startup-trace-handoff",
        action="store_true",
        help=(
            "Suspend the direct-launch main thread across startup debugger "
            "detach, arm the strict successor trace, then resume exactly "
            "once; requires update-zero pre-stream tracing."
        ),
    )
    prepare.add_argument(
        "--allow-attach-stabilization",
        action="store_true",
        help=(
            "At a nonzero direct-runtime attach, ignore only a bounded set "
            "of already in-flight calls until the first exact complete "
            "original-main-thread DMO boundary."
        ),
    )
    prepare.add_argument(
        "--allow-pre-attach-file-write-debt",
        action="store_true",
        help=(
            "Allow exact pre-attach long file-write results to satisfy only "
            "separately labeled bounded late file-write calls."
        ),
    )
    prepare.add_argument(
        "--allow-font-cache-manifest-completion-debt",
        action="store_true",
        help=(
            "Permit only bounded startup font-cache completion results after "
            "exact main.pak manifest, path, size, and DMO-cardinality "
            "validation."
        ),
    )
    prepare.add_argument(
        "--seed-board-before-attach",
        action="store_true",
        help=(
            "Apply the board/global/thread RNG overrides in a dedicated "
            "debugger phase before a nonzero late trace attachment."
        ),
    )
    prepare.add_argument(
        "--startup-priority-bias-until-update",
        type=_positive_int,
        help=(
            "Temporarily lower only the replay main thread through this "
            "startup update, restoring its exact prior priority before "
            "trace attachment."
        ),
    )
    prepare.add_argument(
        "--startup-process-affinity-mask",
        type=lambda value: int(value, 0),
        help=(
            "Apply this process affinity while the byte-identical retail "
            "process is still suspended, before its first instruction, "
            "and bind the verified handoff receipt into both runs."
        ),
    )
    prepare.add_argument(
        "--detach-at-update",
        type=_non_negative_int,
        # Stop breakpoints well before repaint/capture so update cadence has
        # time to return to the game's native scheduler.
        default=7300,
    )
    prepare.add_argument(
        "--reattach-at-update",
        type=_non_negative_int,
        # Resume strict tracing only after the complete video interval.
        default=8500,
    )
    prepare.add_argument(
        "--allow-post-blackout-attach-stabilization",
        action="store_true",
        help=(
            "After debugger-blackout reattachment, accept only the first "
            "bounded exact complete original-main-thread DMO boundary."
        ),
    )
    prepare.add_argument(
        "--allow-blackout-file-write-order-rebase",
        action="store_true",
        help=(
            "Permit one audited post-blackout command-order offset only "
            "for exact long-form file-write result rows."
        ),
    )
    prepare.add_argument(
        "--blackout-expected-command-order-offset",
        type=_positive_int,
        help=(
            "Preregister the exact post-blackout command-entry deficit for "
            "the frozen long-form file-write candidate envelope."
        ),
    )
    prepare.add_argument(
        "--blackout-expected-native-timeline-offset",
        type=_non_negative_int,
        help=(
            "Preregister the exact post-blackout native-update surplus for "
            "the frozen long-form file-write candidate envelope."
        ),
    )
    prepare.add_argument(
        "--blackout-allowed-offset-pair",
        action="append",
        type=_blackout_offset_pair,
        default=[],
        dest="blackout_allowed_offset_pairs",
        metavar="COMMAND:TIMELINE",
        help=(
            "Preregister one member of a finite post-blackout offset set; "
            "repeat for every allowed native scheduling branch."
        ),
    )
    prepare.add_argument(
        "--allow-post-blackout-overdue-idle-reentry",
        action="store_true",
        help=(
            "Permit only a pre-enumerated post-blackout successful-write "
            "exit whose first payload-free long idle is overdue inside the "
            "open interval before the next saturated long idle."
        ),
    )
    prepare.add_argument(
        "--close-after-terminal-command",
        action="store_true",
        help=(
            "After strict consumption of a payload-free final idle, request "
            "one normal game-window close and require exit code zero."
        ),
    )
    prepare.add_argument("--maximum-hits", type=_positive_int, default=6000)
    prepare.add_argument(
        "--trace-timeout-seconds",
        type=_positive_float,
        default=600.0,
    )
    prepare.add_argument("--device-index", type=_non_negative_int, default=0)
    prepare.add_argument("--output-index", type=_non_negative_int, default=0)
    collect = subparsers.add_parser("collect")
    collect.add_argument("--session-root", required=True, type=Path)
    return parser


def run(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare_session(args)
    else:
        result = collect_session(args)
    print(
        json.dumps(
            result,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    try:
        return run(argv)
    except CollectionError as error:
        print(f"collection error: {error.code}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("collection error: interrupted", file=sys.stderr)
        return 130
    except Exception:
        print("collection error: unexpected_failure", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
