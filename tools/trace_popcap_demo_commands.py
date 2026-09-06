"""Trace early retail PopCap demo-command requests.

The SexyAppFramework DMO stream is shared by the main and loading threads.
If playback invokes even one registry/file/sync helper in a different order
from recording, every later command is decoded at the wrong bit position.

This Windows-only diagnostic launches the verified Steam replay, attaches as
soon as the temporary runtime process exists, and places a software breakpoint
on ``SexyAppBase::PrepareDemoCommand``.  It records the caller, object pointer,
framework update, and DMO bit positions for a bounded number of calls.

The original instruction byte is restored while single-stepping and on every
normal cleanup path.  The executable path is verified before process memory is
changed.
"""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zuma_rl.original_data import PopCapPakArchive
from zuma_rl.revenge_core import PopCapMTRandom

from tools.align_popcap_dmo_startup import (
    _command_stream_offset,
    _load_popcap_dmo_module,
    _scan_commands,
)
from tools.launch_popcap_replay import (
    DEFAULT_RUNTIME_EXE,
    DEFAULT_STEAM_EXE,
    process_ids_by_name,
    process_image_path,
    same_windows_path,
)
from tools.popcap_replay_boundary import (
    ATTACH_STABILIZATION_MAX_SKIPPED_HITS,
    ATTACH_STABILIZATION_MAX_UPDATE_DELTA,
    DEFERRED_FILE_WRITE_CALLER,
    DEFERRED_FILE_WRITE_RESUME,
    FONT_CACHE_MANIFEST_EXPECTED_ENTRIES,
    MAIN_DEMO_CALLER,
    SERVICE_COMMAND_NUMBERS,
    SERVICE_PROGRESS_PROBE_TIMEOUT_SECONDS,
    SOURCE_BOUND_GLOBAL_CORRECTION_MAX_DRAWS,
    _audited_service_payload_reentry,
    _audited_terminal_registry_write_eof_exit,
    _audited_service_prepared_reentry,
    _audited_service_header_reentry,
    _audited_service_command_order_rebase,
    _audited_successful_file_write_corridor_prepared_exit,
    _audited_successful_file_write_deferred_exit_committed_boundary_snapshot,
    _audited_successful_file_write_deferred_exit_clamped_suffix_rebase,
    _audited_successful_file_write_deferred_exit_timeline_commit,
    _audited_successful_file_write_debt_barrier,
    _audited_service_exit_late_idle_bridge,
    _audited_service_exit_overdue_idle_reentry,
    _audited_service_exit_overdue_idle_prepared_reentry,
    _audited_service_exit_late_idle_bridge_prepared_reentry,
    _audited_late_idle_native_timeline_rebase,
    _audited_successful_file_write_tail_prefetch,
    _audited_preloading_failed_file_write_tail_prefetch,
    _audited_preloading_failed_file_write_tail_short_header_recovery,
    _audited_service_exit_reentry,
    _audited_service_exit_prepared_reentry,
    _audited_service_exit_partial_header,
    _audited_service_exit_forced_read,
    _audited_service_file_write_header_claim,
    _audited_service_file_write_payload_completion,
    _audited_file_write_command_order_rebase,
    _audited_post_blackout_attach_stabilization,
    _audited_post_blackout_file_write_offset_envelope,
    _audited_post_blackout_file_write_offset_envelope_set,
    _normalize_blackout_native_timeline_snapshot,
    _audited_deferred_file_write_result,
    _audited_font_cache_manifest_completion,
    _audited_main_file_read_corridor,
    _audited_terminal_file_write_payload_handoff,
    _attach_stabilization_boundary_failure,
    _board_seed_override_applies,
    _is_exact_startup_font_cache_failed_write_partition,
    _is_uniform_successful_file_write_corridor,
    _may_adopt_prepared_service_block,
    _may_broker_terminal_prepared_service_row,
    _probe_service_progress,
    _may_bypass_stalled_service_block,
    _normalize_font_cache_member,
    _service_block,
    _service_continuation_corridor,
    _service_settle_end,
    _snapshot_boundary_failure,
    _source_bound_board_anchor_applies,
    _source_bound_global_correction_eligible,
    _source_bound_global_post_correction_delta,
    _wait_for_service_boundary,
    _wait_for_startup_post_bypass_worker_boundary,
    _commit_startup_post_bypass_worker_continuation,
    _consume_startup_post_bypass_worker_echo,
)
from tools.popcap_global_mtrand_restore import (
    DEFAULT_MAXIMUM_RECONSTRUCTION_DRAWS,
    GlobalMTRandRestoreState,
    load_global_mtrand_restore_state,
    load_source_bound_global_mtrand_restore_state,
)
from tools.popcap_global_mtrand_call_oracle import (
    InitialGlobalMTRandCallOracle,
    StartupGlobalMTRandObservationOracle,
    load_initial_global_mtrand_call_oracle,
    load_startup_global_mtrand_observation_oracle,
    reconstruct_mtrand_draw_counts,
)
from tools.popcap_qrand_restore import (
    QRAND_OBJECT_SIZE,
    QRandRestoreState,
    load_qrand_restore_state,
    qrand_scalar_payload,
    qrand_vector_write_plan,
)
from tools.popcap_mtrand_call_oracle import (
    GameplayMTRandOracle,
    load_gameplay_mtrand_oracle,
)
from tools.popcap_thread_crt_restore import (
    ThreadCrtRestoreState,
    load_source_bound_thread_crt_restore_state,
    load_thread_crt_restore_state,
)
from zuma_rl.pc_memory_trajectory import (
    BOARD_NATIVE_GAME_TIME_OFFSET,
    BOARD_SCORE_OFFSET,
    BOARD_SCORE_TARGET_OFFSET,
)
from tools.inspect_popcap_replay import (
    close_process,
    open_process_readonly,
    read_process_bytes,
)
from tools.control_popcap_replay import main_window_for_pid
from tools.launch_fixed_seed_replay import (
    FixedSeedIatLaunchEvidence,
    FixedSeedLaunchEvidence,
    NaturalSeedLaunchEvidence,
    launch_fixed_seed_replay,
    launch_fixed_seed_replay_iat_stub,
    launch_natural_seed_replay,
)
from tools.trace_popcap_shutdown import (
    CONTEXT_FULL,
    CREATE_PROCESS_DEBUG_EVENT,
    DBG_CONTINUE,
    DBG_EXCEPTION_NOT_HANDLED,
    DEBUG_EVENT,
    EXCEPTION_BREAKPOINT,
    EXCEPTION_DEBUG_EVENT,
    EXIT_PROCESS_DEBUG_EVENT,
    LOAD_DLL_DEBUG_EVENT,
    MICROSOFT_CPP_EXCEPTION,
    PROCESS_ACCESS,
    THREAD_ACCESS,
    WOW64_CONTEXT,
    kernel32,
    read_memory,
    write_memory,
)


EXCEPTION_SINGLE_STEP = 0x80000004
STATUS_WX86_SINGLE_STEP = 0x4000001E
STATUS_WX86_BREAKPOINT = 0x4000001F
ERROR_SEM_TIMEOUT = 121
ERROR_ACCESS_DENIED = 5
TRAP_FLAG = 0x100
DEFAULT_PREPARE_DEMO_COMMAND = 0x00686DA0
DEFAULT_BOARD_SEED_CALL = 0x006568D1
DEFAULT_BOARD_RESEED_CALL = 0x0065B828
SOURCE_BOUND_GLOBAL_PRECALL_ADDRESS = 0x0065B81C
SOURCE_BOUND_GLOBAL_PRECALL_RETURN_ADDRESS = 0x0065B821
SOURCE_BOUND_GLOBAL_PRECALL_TARGET_ADDRESS = 0x00617490
SOURCE_BOUND_GLOBAL_PRECALL_INSTRUCTION = bytes.fromhex("e86fbcfbff")
G_SEXY_APP_BASE_ADDRESS = 0x009FC740
G_FRAMEWORK_MTRAND_ADDRESS = 0x00A313C0
ACTIVE_BOARD_OFFSET = 0x834
BOARD_VTABLE = 0x0096356C
BOARD_SEED_OWNER_VTABLE = 0x009884A4
BOARD_MTRAND_OFFSET = 0x4C
BOARD_QRAND_POINTER_OFFSET = 0x7A8
MTRAND_STATE_WORDS = 624
MTRAND_STATE_BYTES = (MTRAND_STATE_WORDS + 1) * 4
CRT_FLS_INDEX_ADDRESS = 0x009E5184
CRT_PTD_THREAD_ID_OFFSET = 0x00
CRT_PTD_RAND_STATE_OFFSET = 0x14
WOW64_TEB_FLS_DATA_OFFSET = 0x0FB4
FLS_MAXIMUM_AVAILABLE = 0x0FF0
THREAD_SUSPEND_RESUME = 0x0002
THREAD_SET_INFORMATION = 0x0020
THREAD_QUERY_LIMITED_INFORMATION = 0x0800
INVALID_SUSPEND_COUNT = 0xFFFFFFFF
TH32CS_SNAPTHREAD = 0x00000004
ERROR_INVALID_PARAMETER = 87
THREAD_SUSPEND_RETRY_ATTEMPTS = 5
THREAD_SUSPEND_RETRY_DELAY_SECONDS = 0.01
THREAD_PRIORITY_ERROR_RETURN = 0x7FFFFFFF
THREAD_PRIORITY_LOWEST = -2
STARTUP_PRIORITY_REFRESH_SECONDS = 0.01
WM_CLOSE = 0x0010
CONTEXT_DEBUG_REGISTERS = 0x00010010
CONTEXT_FULL_AND_DEBUG = CONTEXT_FULL | CONTEXT_DEBUG_REGISTERS
RESUME_FLAG = 0x00010000


class THREADENTRY32(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("cntUsage", wintypes.DWORD),
        ("th32ThreadID", wintypes.DWORD),
        ("th32OwnerProcessID", wintypes.DWORD),
        ("tpBasePri", wintypes.LONG),
        ("tpDeltaPri", wintypes.LONG),
        ("dwFlags", wintypes.DWORD),
    ]


class WOW64_LDT_ENTRY_BYTES(ctypes.Structure):
    _fields_ = [
        ("BaseMid", ctypes.c_ubyte),
        ("Flags1", ctypes.c_ubyte),
        ("Flags2", ctypes.c_ubyte),
        ("BaseHi", ctypes.c_ubyte),
    ]


class WOW64_LDT_ENTRY_HIGH_WORD(ctypes.Union):
    _fields_ = [
        ("Bytes", WOW64_LDT_ENTRY_BYTES),
        ("Raw", wintypes.DWORD),
    ]


class WOW64_LDT_ENTRY(ctypes.Structure):
    _fields_ = [
        ("LimitLow", wintypes.WORD),
        ("BaseLow", wintypes.WORD),
        ("HighWord", WOW64_LDT_ENTRY_HIGH_WORD),
    ]


kernel32.SuspendThread.argtypes = [wintypes.HANDLE]
kernel32.SuspendThread.restype = wintypes.DWORD
kernel32.ResumeThread.argtypes = [wintypes.HANDLE]
kernel32.ResumeThread.restype = wintypes.DWORD
kernel32.Wow64GetThreadSelectorEntry.argtypes = [
    wintypes.HANDLE,
    wintypes.DWORD,
    ctypes.POINTER(WOW64_LDT_ENTRY),
]
kernel32.Wow64GetThreadSelectorEntry.restype = wintypes.BOOL
kernel32.CreateToolhelp32Snapshot.argtypes = [
    wintypes.DWORD,
    wintypes.DWORD,
]
kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
kernel32.Thread32First.argtypes = [
    wintypes.HANDLE,
    ctypes.POINTER(THREADENTRY32),
]
kernel32.Thread32First.restype = wintypes.BOOL
kernel32.Thread32Next.argtypes = [
    wintypes.HANDLE,
    ctypes.POINTER(THREADENTRY32),
]
kernel32.Thread32Next.restype = wintypes.BOOL
kernel32.GetThreadPriority.argtypes = [wintypes.HANDLE]
kernel32.GetThreadPriority.restype = ctypes.c_int
kernel32.SetThreadPriority.argtypes = [wintypes.HANDLE, ctypes.c_int]
kernel32.SetThreadPriority.restype = wintypes.BOOL


@dataclass(frozen=True)
class TraceResult:
    hits: int
    exited: bool
    exit_code: int | None
    brokered_blocks: int
    broker_bypassed_blocks: int
    startup_post_bypass_worker_yields: tuple[Mapping[str, Any], ...]
    broker_failures: tuple[str, ...]
    boundary_failures: tuple[str, ...]
    last_update: int
    stopped_at_update: bool
    stopped_at_command_order: bool
    access_denied_thread_skip_events: tuple[int, ...] = ()
    handoff_main_thread_suspended: bool = False
    handoff_main_thread_id: int | None = None
    handoff_previous_suspend_count: int | None = None
    first_command_order: int | None = None
    last_command_order: int | None = None
    command_order_offset: int = 0
    command_order_rebase_rows: tuple[int, ...] = ()
    blackout_file_write_rebase_candidate_rows: tuple[int, ...] = ()
    blackout_native_timeline_offset: int = 0
    diagnostic_successful_file_write_padding_rows: tuple[int, ...] = ()
    service_consumed_file_write_debt_rows: tuple[int, ...] = ()
    service_consumed_file_write_surplus_rows: tuple[int, ...] = ()
    effective_deferred_file_write_debt_rows: tuple[int, ...] = ()
    successful_file_write_debt_barriers: tuple[Mapping[str, Any], ...] = ()
    terminal_registry_write_eof_exits: tuple[Mapping[str, Any], ...] = ()
    service_exit_timeline_suffix_rebases: tuple[Mapping[str, Any], ...] = ()
    attach_stabilization_verified: bool = False
    attach_stabilization_skipped_hits: int = 0
    attach_stabilization_start_update: int | None = None
    attach_stabilization_verified_update: int | None = None
    attach_stabilization_verified_command_order: int | None = None
    attach_stabilization_observations: tuple[
        Mapping[str, int | str | bool], ...
    ] = ()
    service_exit_timeline_commits: tuple[Mapping[str, int], ...] = ()
    service_continuation_verifications: tuple[
        Mapping[str, int | str], ...
    ] = ()
    service_exit_overdue_idle_reentries: tuple[
        Mapping[str, Any], ...
    ] = ()
    preloading_failed_file_write_tail_short_header_recoveries: tuple[
        Mapping[str, Any], ...
    ] = ()
    main_file_read_corridor_native_replays: tuple[
        Mapping[str, int | str], ...
    ] = ()
    deferred_file_write_discharges: tuple[
        Mapping[str, int | bool | str], ...
    ] = ()
    terminal_file_write_payload_handoffs: tuple[
        Mapping[str, int | bool | str], ...
    ] = ()
    service_file_write_header_claims: tuple[
        Mapping[str, int | bool | str], ...
    ] = ()
    direct_font_cache_file_write_observations: tuple[
        Mapping[str, int | bool | str], ...
    ] = ()
    pre_attach_file_write_debt_rows: tuple[int, ...] = ()
    font_cache_manifest_completion_discharges: tuple[
        Mapping[str, int | bool | str], ...
    ] = ()
    font_cache_manifest_entry_count: int = 0
    font_cache_manifest_sha256: str | None = None
    font_cache_manifest_main_pak_sha256: str | None = None
    startup_handoff_main_thread_id: int | None = None
    startup_handoff_launcher_suspend_previous_count: int | None = None
    startup_handoff_tracer_resume_previous_count: int | None = None
    startup_handoff_resumed_after_breakpoints: bool = False
    startup_handoff_resumed_perf_counter_ns: int | None = None
    terminal_close_requests: tuple[
        Mapping[str, int | str | bool], ...
    ] = ()
    debug_exception_observations: tuple[Mapping[str, Any], ...] = ()

    @property
    def failure_count(self) -> int:
        return len(self.broker_failures) + len(self.boundary_failures)


@dataclass
class StartupPriorityBiasRuntime:
    """Mutable lifecycle for an in-trace transient main-thread bias."""

    process_id: int
    main_thread_id: int
    requested_until_update: int
    started_perf_counter_ns: int
    first_applied_update: int
    records: dict[int, dict[str, int | str | bool | None]]
    restored_at_update: int | None = None
    restored_perf_counter_ns: int | None = None


def _merge_trace_results(
    first: TraceResult,
    second: TraceResult,
) -> TraceResult:
    """Combine two debugger phases around one explicit untraced interval."""

    return TraceResult(
        hits=first.hits + second.hits,
        exited=second.exited,
        exit_code=second.exit_code,
        brokered_blocks=(
            first.brokered_blocks + second.brokered_blocks
        ),
        broker_bypassed_blocks=(
            first.broker_bypassed_blocks
            + second.broker_bypassed_blocks
        ),
        startup_post_bypass_worker_yields=(
            first.startup_post_bypass_worker_yields
            + second.startup_post_bypass_worker_yields
        ),
        broker_failures=(
            first.broker_failures + second.broker_failures
        ),
        boundary_failures=(
            first.boundary_failures + second.boundary_failures
        ),
        last_update=max(first.last_update, second.last_update),
        stopped_at_update=second.stopped_at_update,
        stopped_at_command_order=second.stopped_at_command_order,
        access_denied_thread_skip_events=(
            first.access_denied_thread_skip_events
            + second.access_denied_thread_skip_events
        ),
        handoff_main_thread_suspended=(
            second.handoff_main_thread_suspended
        ),
        handoff_main_thread_id=second.handoff_main_thread_id,
        handoff_previous_suspend_count=(
            second.handoff_previous_suspend_count
        ),
        first_command_order=first.first_command_order,
        last_command_order=second.last_command_order,
        command_order_offset=second.command_order_offset,
        command_order_rebase_rows=(
            second.command_order_rebase_rows
        ),
        blackout_file_write_rebase_candidate_rows=(
            second.blackout_file_write_rebase_candidate_rows
        ),
        blackout_native_timeline_offset=(
            second.blackout_native_timeline_offset
        ),
        diagnostic_successful_file_write_padding_rows=(
            second.diagnostic_successful_file_write_padding_rows
            or first.diagnostic_successful_file_write_padding_rows
        ),
        service_consumed_file_write_debt_rows=(
            second.service_consumed_file_write_debt_rows
            or first.service_consumed_file_write_debt_rows
        ),
        service_consumed_file_write_surplus_rows=(
            second.service_consumed_file_write_surplus_rows
            or first.service_consumed_file_write_surplus_rows
        ),
        effective_deferred_file_write_debt_rows=(
            second.effective_deferred_file_write_debt_rows
            or first.effective_deferred_file_write_debt_rows
        ),
        successful_file_write_debt_barriers=(
            first.successful_file_write_debt_barriers
            + second.successful_file_write_debt_barriers
        ),
        terminal_registry_write_eof_exits=(
            first.terminal_registry_write_eof_exits
            + second.terminal_registry_write_eof_exits
        ),
        service_exit_timeline_suffix_rebases=(
            first.service_exit_timeline_suffix_rebases
            + second.service_exit_timeline_suffix_rebases
        ),
        attach_stabilization_verified=(
            first.attach_stabilization_verified
            or second.attach_stabilization_verified
        ),
        attach_stabilization_skipped_hits=(
            first.attach_stabilization_skipped_hits
            + second.attach_stabilization_skipped_hits
        ),
        attach_stabilization_start_update=(
            first.attach_stabilization_start_update
            if first.attach_stabilization_start_update is not None
            else second.attach_stabilization_start_update
        ),
        attach_stabilization_verified_update=(
            first.attach_stabilization_verified_update
            if first.attach_stabilization_verified
            else second.attach_stabilization_verified_update
        ),
        attach_stabilization_verified_command_order=(
            first.attach_stabilization_verified_command_order
            if first.attach_stabilization_verified
            else second.attach_stabilization_verified_command_order
        ),
        attach_stabilization_observations=(
            first.attach_stabilization_observations
            + second.attach_stabilization_observations
        ),
        service_exit_timeline_commits=(
            first.service_exit_timeline_commits
            + second.service_exit_timeline_commits
        ),
        service_continuation_verifications=(
            first.service_continuation_verifications
            + second.service_continuation_verifications
        ),
        service_exit_overdue_idle_reentries=(
            first.service_exit_overdue_idle_reentries
            + second.service_exit_overdue_idle_reentries
        ),
        preloading_failed_file_write_tail_short_header_recoveries=(
            first.preloading_failed_file_write_tail_short_header_recoveries
            + second.preloading_failed_file_write_tail_short_header_recoveries
        ),
        main_file_read_corridor_native_replays=(
            first.main_file_read_corridor_native_replays
            + second.main_file_read_corridor_native_replays
        ),
        deferred_file_write_discharges=(
            first.deferred_file_write_discharges
            + second.deferred_file_write_discharges
        ),
        terminal_file_write_payload_handoffs=(
            first.terminal_file_write_payload_handoffs
            + second.terminal_file_write_payload_handoffs
        ),
        service_file_write_header_claims=(
            first.service_file_write_header_claims
            + second.service_file_write_header_claims
        ),
        direct_font_cache_file_write_observations=(
            first.direct_font_cache_file_write_observations
            + second.direct_font_cache_file_write_observations
        ),
        pre_attach_file_write_debt_rows=(
            first.pre_attach_file_write_debt_rows
            + second.pre_attach_file_write_debt_rows
        ),
        font_cache_manifest_completion_discharges=(
            first.font_cache_manifest_completion_discharges
            + second.font_cache_manifest_completion_discharges
        ),
        font_cache_manifest_entry_count=(
            first.font_cache_manifest_entry_count
            or second.font_cache_manifest_entry_count
        ),
        font_cache_manifest_sha256=(
            first.font_cache_manifest_sha256
            or second.font_cache_manifest_sha256
        ),
        font_cache_manifest_main_pak_sha256=(
            first.font_cache_manifest_main_pak_sha256
            or second.font_cache_manifest_main_pak_sha256
        ),
        startup_handoff_main_thread_id=(
            first.startup_handoff_main_thread_id
            if first.startup_handoff_main_thread_id is not None
            else second.startup_handoff_main_thread_id
        ),
        startup_handoff_launcher_suspend_previous_count=(
            first.startup_handoff_launcher_suspend_previous_count
            if first.startup_handoff_launcher_suspend_previous_count
            is not None
            else second.startup_handoff_launcher_suspend_previous_count
        ),
        startup_handoff_tracer_resume_previous_count=(
            first.startup_handoff_tracer_resume_previous_count
            if first.startup_handoff_tracer_resume_previous_count is not None
            else second.startup_handoff_tracer_resume_previous_count
        ),
        startup_handoff_resumed_after_breakpoints=(
            first.startup_handoff_resumed_after_breakpoints
            or second.startup_handoff_resumed_after_breakpoints
        ),
        startup_handoff_resumed_perf_counter_ns=(
            first.startup_handoff_resumed_perf_counter_ns
            if first.startup_handoff_resumed_perf_counter_ns is not None
            else second.startup_handoff_resumed_perf_counter_ns
        ),
        terminal_close_requests=(
            first.terminal_close_requests
            + second.terminal_close_requests
        ),
        debug_exception_observations=(
            first.debug_exception_observations
            + second.debug_exception_observations
        ),
    )


def _trace_phase_evidence(
    *,
    name: str,
    started_perf_counter_ns: int,
    finished_perf_counter_ns: int,
    result: TraceResult,
) -> Mapping[str, Any]:
    return {
        "name": name,
        "started_perf_counter_ns": started_perf_counter_ns,
        "finished_perf_counter_ns": finished_perf_counter_ns,
        "result": {
            **asdict(result),
            "failure_count": result.failure_count,
        },
    }


def _offline_rows(dmo: Path) -> list[dict[str, object]]:
    module = _load_popcap_dmo_module()
    data = dmo.read_bytes()
    offset = _command_stream_offset(data)
    length_updates = struct.unpack_from("<I", data, offset - 4)[0]
    rows, _ = _scan_commands(module, data[offset:], length_updates)
    return rows


def _post_terminal_close(pid: int) -> tuple[int, int]:
    """Post one normal close only to the exact verified replay window."""

    hwnd = main_window_for_pid(pid)
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.PostMessageW.argtypes = (
        wintypes.HWND,
        wintypes.UINT,
        wintypes.WPARAM,
        wintypes.LPARAM,
    )
    user32.PostMessageW.restype = wintypes.BOOL
    requested_ns = time.perf_counter_ns()
    if not user32.PostMessageW(
        wintypes.HWND(hwnd),
        WM_CLOSE,
        0,
        0,
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    return hwnd, requested_ns


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _load_font_cache_manifest(
    changedir: Path,
) -> tuple[dict[str, int], str, str]:
    """Load and hash the immutable retail 600-DPI CFW2 catalog."""

    main_pak = changedir / "main.pak"
    archive = PopCapPakArchive(main_pak)
    manifest: dict[str, int] = {}
    for entry in archive.entries:
        normalized_name = entry.name.replace("/", "\\").casefold()
        member = _normalize_font_cache_member(normalized_name)
        if (
            member is None
            or member != normalized_name
            or not member.startswith("cached\\fonts\\600\\")
        ):
            continue
        if member in manifest:
            raise ValueError(
                f"duplicate retail font-cache member: {member}"
            )
        manifest[member] = entry.size
    canonical = json.dumps(
        sorted(manifest.items()),
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
    return (
        manifest,
        hashlib.sha256(canonical).hexdigest(),
        _sha256_path(main_pak),
    )


def _thread_ids_for_pid(pid: int) -> tuple[int, ...]:
    snapshot = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0)
    invalid_handle = ctypes.c_void_p(-1).value
    if not snapshot or int(snapshot) == invalid_handle:
        raise ctypes.WinError(ctypes.get_last_error())
    found: list[int] = []
    try:
        entry = THREADENTRY32()
        entry.dwSize = ctypes.sizeof(entry)
        ok = kernel32.Thread32First(snapshot, ctypes.byref(entry))
        while ok:
            if int(entry.th32OwnerProcessID) == pid:
                found.append(int(entry.th32ThreadID))
            ok = kernel32.Thread32Next(snapshot, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snapshot)
    return tuple(sorted(found))


def _resume_suspended_threads(
    suspended: list[tuple[int, int]],
) -> None:
    failure: OSError | None = None
    while suspended:
        _, handle = suspended.pop()
        result = int(kernel32.ResumeThread(handle))
        if result == INVALID_SUSPEND_COUNT and failure is None:
            failure = ctypes.WinError(ctypes.get_last_error())
        kernel32.CloseHandle(handle)
    if failure is not None:
        raise failure


def _suspend_other_threads(
    pid: int,
    *,
    excluded_thread_id: int,
    access_denied_skip_events: list[int] | None = None,
) -> list[tuple[int, int]]:
    """Increment every other live thread's suspend count exactly once."""

    suspended: list[tuple[int, int]] = []
    try:
        for thread_id in _thread_ids_for_pid(pid):
            if thread_id == excluded_thread_id:
                continue
            for attempt in range(1, THREAD_SUSPEND_RETRY_ATTEMPTS + 1):
                handle = kernel32.OpenThread(
                    THREAD_SUSPEND_RESUME,
                    False,
                    thread_id,
                )
                if not handle:
                    error_code = ctypes.get_last_error()
                else:
                    previous = int(kernel32.SuspendThread(handle))
                    if previous != INVALID_SUSPEND_COUNT:
                        suspended.append((thread_id, int(handle)))
                        break
                    error_code = ctypes.get_last_error()
                    kernel32.CloseHandle(handle)
                if (
                    error_code == ERROR_INVALID_PARAMETER
                    or thread_id not in _thread_ids_for_pid(pid)
                ):
                    break
                if (
                    error_code == ERROR_ACCESS_DENIED
                    and attempt < THREAD_SUSPEND_RETRY_ATTEMPTS
                ):
                    print(
                        "thread_suspend_retry "
                        f"tid={thread_id} attempt={attempt} "
                        f"error={error_code}",
                        flush=True,
                    )
                    # The outstanding debug event still freezes every process
                    # thread here. Reopening the handle is therefore a bounded
                    # retry without an untraced execution window.
                    time.sleep(THREAD_SUSPEND_RETRY_DELAY_SECONDS)
                    continue
                if (
                    error_code == ERROR_ACCESS_DENIED
                    and access_denied_skip_events is not None
                ):
                    # A thread already in kernel teardown can remain in a
                    # Toolhelp snapshot while OpenThread(SUSPEND_RESUME)
                    # consistently returns ACCESS_DENIED.  The global RNG
                    # call tracer may explicitly retain this evidence and
                    # continue; any actually runnable skipped thread entering
                    # the temporarily unarmed wrapper still trips the trace's
                    # breakpoint-state collision guard.
                    access_denied_skip_events.append(thread_id)
                    print(
                        "thread_suspend_skip_access_denied "
                        f"tid={thread_id} attempts="
                        f"{THREAD_SUSPEND_RETRY_ATTEMPTS}",
                        flush=True,
                    )
                    break
                raise ctypes.WinError(error_code)
    except Exception:
        _resume_suspended_threads(suspended)
        raise
    return suspended


def wait_for_new_runtime(
    *,
    existing_pids: set[int],
    runtime_exe: Path,
    timeout: float,
) -> int:
    """Return a newly created path-verified runtime before it has a window."""

    deadline = time.monotonic() + timeout
    observed: set[int] = set()
    while time.monotonic() < deadline:
        current = set(process_ids_by_name(runtime_exe.name))
        for pid in sorted(current - existing_pids):
            observed.add(pid)
            image = process_image_path(pid)
            if image is not None and same_windows_path(image, runtime_exe):
                return pid
        time.sleep(0.0005)
    raise TimeoutError(
        "timed out waiting for a new path-verified PopCap runtime; "
        f"observed PIDs={sorted(observed)}"
    )


def wait_for_update_before_attach(
    *,
    pid: int,
    target_update: int,
    timeout: float,
) -> tuple[int, int]:
    """Wait read-only until a replay reaches a safe late-attach boundary."""

    if target_update < 0 or timeout <= 0:
        raise ValueError("target_update must be non-negative and timeout positive")
    handle = open_process_readonly(pid)
    deadline = time.monotonic() + timeout
    last_update = -1
    base = 0
    try:
        while time.monotonic() < deadline:
            try:
                if not base:
                    base = struct.unpack(
                        "<I",
                        read_process_bytes(
                            handle,
                            G_SEXY_APP_BASE_ADDRESS,
                            4,
                        ),
                    )[0]
                    if not base:
                        time.sleep(0.001)
                        continue
                last_update = struct.unpack(
                    "<i",
                    read_process_bytes(handle, base + 0x4C4, 4),
                )[0]
                if last_update >= target_update:
                    return base, last_update
            except (OSError, RuntimeError):
                if process_image_path(pid) is None:
                    break
            time.sleep(0.001)
    finally:
        close_process(handle)
    raise TimeoutError(
        f"PID {pid} did not reach update {target_update}; "
        f"last_update={last_update}"
    )


def _apply_startup_thread_priority_bias(
    *,
    pid: int,
    main_thread_id: int,
    framework_update: int,
    records: dict[int, dict[str, int | str | bool | None]],
) -> int:
    """Lower only the main thread until the startup service block."""

    applied = 0
    for thread_id in _thread_ids_for_pid(pid):
        if thread_id != main_thread_id or thread_id in records:
            continue
        target = THREAD_PRIORITY_LOWEST
        handle = kernel32.OpenThread(
            THREAD_SET_INFORMATION | THREAD_QUERY_LIMITED_INFORMATION,
            False,
            thread_id,
        )
        if not handle:
            error = ctypes.get_last_error()
            if error == ERROR_INVALID_PARAMETER:
                continue
            raise ctypes.WinError(error)
        try:
            original = int(kernel32.GetThreadPriority(handle))
            if original == THREAD_PRIORITY_ERROR_RETURN:
                raise ctypes.WinError(ctypes.get_last_error())
            if original != target and not kernel32.SetThreadPriority(
                handle,
                target,
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            observed = int(kernel32.GetThreadPriority(handle))
            if observed != target:
                raise RuntimeError(
                    "startup thread priority bias did not verify"
                )
            records[thread_id] = {
                "thread_id": thread_id,
                "role": "main",
                "original_priority": original,
                "biased_priority": target,
                "applied_framework_update": framework_update,
                "restored": None,
                "restore_status": "pending",
            }
            applied += 1
        finally:
            kernel32.CloseHandle(handle)
    return applied


def _restore_startup_thread_priorities(
    records: dict[int, dict[str, int | str | bool | None]],
) -> None:
    """Restore every still-live thread to its exact observed priority."""

    failures: list[str] = []
    for thread_id in sorted(records):
        record = records[thread_id]
        if record.get("restored") is True:
            continue
        handle = kernel32.OpenThread(
            THREAD_SET_INFORMATION | THREAD_QUERY_LIMITED_INFORMATION,
            False,
            thread_id,
        )
        if not handle:
            error = ctypes.get_last_error()
            if error == ERROR_INVALID_PARAMETER:
                record["restored"] = True
                record["restore_status"] = "thread_exited"
                continue
            record["restored"] = False
            record["restore_status"] = f"open_failed:{error}"
            failures.append(f"thread {thread_id}: open failed {error}")
            continue
        try:
            original = int(record["original_priority"])
            if not kernel32.SetThreadPriority(handle, original):
                error = ctypes.get_last_error()
                if error == ERROR_INVALID_PARAMETER:
                    record["restored"] = True
                    record["restore_status"] = "thread_exited"
                    continue
                record["restored"] = False
                record["restore_status"] = f"set_failed:{error}"
                failures.append(f"thread {thread_id}: set failed {error}")
                continue
            observed = int(kernel32.GetThreadPriority(handle))
            if observed == THREAD_PRIORITY_ERROR_RETURN:
                error = ctypes.get_last_error()
                if error == ERROR_INVALID_PARAMETER:
                    record["restored"] = True
                    record["restore_status"] = "thread_exited"
                    continue
                record["restored"] = False
                record["restore_status"] = f"query_failed:{error}"
                failures.append(
                    f"thread {thread_id}: query failed {error}"
                )
                continue
            record["restored_priority"] = observed
            record["restored"] = observed == original
            record["restore_status"] = (
                "restored"
                if observed == original
                else "verify_failed"
            )
            if observed != original:
                failures.append(
                    f"thread {thread_id}: restored {observed}, "
                    f"expected {original}"
                )
        finally:
            kernel32.CloseHandle(handle)
    if failures:
        raise RuntimeError(
            "startup thread priority restoration failed: "
            + "; ".join(failures)
        )


def wait_for_update_with_startup_priority_bias(
    *,
    pid: int,
    main_thread_id: int,
    target_update: int,
    bias_until_update: int,
    timeout: float,
) -> tuple[int, int, Mapping[str, Any]]:
    """Cross the first DMO service block under a transient scheduler bias."""

    if (
        main_thread_id <= 0
        or bias_until_update <= 0
        or bias_until_update >= target_update
        or timeout <= 0
    ):
        raise ValueError("invalid startup priority bias boundary")
    handle = open_process_readonly(pid)
    deadline = time.monotonic() + timeout
    started_ns = time.perf_counter_ns()
    last_update = -1
    base = 0
    first_applied_update: int | None = None
    restored_at_update: int | None = None
    restored_ns: int | None = None
    records: dict[int, dict[str, int | str | bool | None]] = {}
    next_priority_refresh = 0.0
    try:
        while time.monotonic() < deadline:
            try:
                if not base:
                    base = struct.unpack(
                        "<I",
                        read_process_bytes(
                            handle,
                            G_SEXY_APP_BASE_ADDRESS,
                            4,
                        ),
                    )[0]
                    if not base:
                        time.sleep(0.001)
                        continue
                last_update = struct.unpack(
                    "<i",
                    read_process_bytes(handle, base + 0x4C4, 4),
                )[0]
                now = time.monotonic()
                if (
                    restored_at_update is None
                    and last_update < bias_until_update
                    and now >= next_priority_refresh
                ):
                    applied = _apply_startup_thread_priority_bias(
                        pid=pid,
                        main_thread_id=main_thread_id,
                        framework_update=last_update,
                        records=records,
                    )
                    if applied:
                        if first_applied_update is None:
                            first_applied_update = last_update
                        print(
                            "startup_priority_bias_applied "
                            f"update={last_update} "
                            f"new_threads={applied} "
                            f"total_threads={len(records)}",
                            flush=True,
                        )
                    next_priority_refresh = (
                        now + STARTUP_PRIORITY_REFRESH_SECONDS
                    )
                if (
                    restored_at_update is None
                    and last_update >= bias_until_update
                ):
                    _restore_startup_thread_priorities(records)
                    restored_at_update = last_update
                    restored_ns = time.perf_counter_ns()
                    print(
                        "startup_priority_bias_restored "
                        f"update={last_update} "
                        f"threads={len(records)}",
                        flush=True,
                    )
                if last_update >= target_update:
                    if restored_at_update is None:
                        _restore_startup_thread_priorities(records)
                        restored_at_update = last_update
                        restored_ns = time.perf_counter_ns()
                    evidence = {
                        "name": "startup_service_priority_bias",
                        "mechanism": (
                            "transient_main_thread_priority_bias"
                        ),
                        "process_id": pid,
                        "main_thread_id": main_thread_id,
                        "requested_until_update": bias_until_update,
                        "first_applied_update": first_applied_update,
                        "restored_at_update": restored_at_update,
                        "started_perf_counter_ns": started_ns,
                        "restored_perf_counter_ns": restored_ns,
                        "finished_perf_counter_ns": time.perf_counter_ns(),
                        "persistent_process_modification": False,
                        "threads": [
                            dict(records[thread_id])
                            for thread_id in sorted(records)
                        ],
                    }
                    return base, last_update, evidence
            except (OSError, RuntimeError):
                if process_image_path(pid) is None:
                    break
                raise
            time.sleep(0.001)
    finally:
        if records and restored_at_update is None:
            _restore_startup_thread_priorities(records)
        close_process(handle)
    raise TimeoutError(
        f"PID {pid} did not reach update {target_update} under startup "
        f"priority bias; last_update={last_update}"
    )


def _begin_in_trace_startup_priority_bias(
    *,
    pid: int,
    main_thread_id: int,
    framework_update: int,
    bias_until_update: int,
) -> StartupPriorityBiasRuntime:
    """Apply the startup bias before attaching the command debugger."""

    if (
        pid <= 0
        or main_thread_id <= 0
        or framework_update < 0
        or bias_until_update <= framework_update
    ):
        raise ValueError("invalid in-trace startup priority bias")
    records: dict[int, dict[str, int | str | bool | None]] = {}
    started_ns = time.perf_counter_ns()
    applied = _apply_startup_thread_priority_bias(
        pid=pid,
        main_thread_id=main_thread_id,
        framework_update=framework_update,
        records=records,
    )
    if applied != 1 or main_thread_id not in records:
        if records:
            _restore_startup_thread_priorities(records)
        raise RuntimeError(
            "failed to apply the in-trace main-thread priority bias"
        )
    print(
        "startup_priority_bias_applied "
        f"update={framework_update} "
        f"new_threads={applied} total_threads={len(records)}",
        flush=True,
    )
    return StartupPriorityBiasRuntime(
        process_id=pid,
        main_thread_id=main_thread_id,
        requested_until_update=bias_until_update,
        started_perf_counter_ns=started_ns,
        first_applied_update=framework_update,
        records=records,
    )


def _restore_in_trace_startup_priority_bias(
    runtime: StartupPriorityBiasRuntime,
    *,
    framework_update: int,
) -> None:
    """Restore the exact original priority once, recording the boundary."""

    if runtime.restored_at_update is not None:
        return
    _restore_startup_thread_priorities(runtime.records)
    runtime.restored_at_update = framework_update
    runtime.restored_perf_counter_ns = time.perf_counter_ns()
    print(
        "startup_priority_bias_restored "
        f"update={framework_update} "
        f"threads={len(runtime.records)}",
        flush=True,
    )


def _in_trace_startup_priority_evidence(
    runtime: StartupPriorityBiasRuntime,
) -> Mapping[str, Any]:
    """Return the same closed lifecycle evidence as the late-attach path."""

    return {
        "name": "startup_service_priority_bias",
        "mechanism": "transient_main_thread_priority_bias",
        "process_id": runtime.process_id,
        "main_thread_id": runtime.main_thread_id,
        "requested_until_update": runtime.requested_until_update,
        "first_applied_update": runtime.first_applied_update,
        "restored_at_update": runtime.restored_at_update,
        "started_perf_counter_ns": runtime.started_perf_counter_ns,
        "restored_perf_counter_ns": runtime.restored_perf_counter_ns,
        "finished_perf_counter_ns": time.perf_counter_ns(),
        "persistent_process_modification": False,
        "threads": [
            dict(runtime.records[thread_id])
            for thread_id in sorted(runtime.records)
        ],
    }


def _read_u32(process: int, address: int) -> int:
    return struct.unpack("<I", read_memory(process, address, 4))[0]


def _decode_near_relative_branch_target(
    instruction_address: int,
    instruction: bytes,
    *,
    opcode: int,
) -> int:
    """Decode one five-byte x86 near CALL or JMP target."""

    if (
        instruction_address <= 0
        or opcode not in {0xE8, 0xE9}
        or len(instruction) != 5
        or instruction[0] != opcode
    ):
        raise ValueError("near relative branch identity mismatch")
    displacement = struct.unpack_from("<i", instruction, 1)[0]
    return instruction_address + len(instruction) + displacement


def _read_i32(process: int, address: int) -> int:
    return struct.unpack("<i", read_memory(process, address, 4))[0]


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _read_debug_context(thread: int) -> WOW64_CONTEXT:
    context = WOW64_CONTEXT()
    context.ContextFlags = CONTEXT_FULL_AND_DEBUG
    if not kernel32.Wow64GetThreadContext(
        thread,
        ctypes.byref(context),
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    return context


def _write_debug_context(thread: int, context: WOW64_CONTEXT) -> None:
    context.ContextFlags = CONTEXT_FULL_AND_DEBUG
    if not kernel32.Wow64SetThreadContext(
        thread,
        ctypes.byref(context),
    ):
        raise ctypes.WinError(ctypes.get_last_error())


def _arm_gameplay_mtrand_breakpoints(
    *,
    main_thread_id: int,
    addresses: tuple[int, ...],
) -> tuple[int, int, int, int, int, int]:
    """Arm up to four return addresses while startup holds the process."""

    if (
        not addresses
        or len(addresses) > 4
        or len(set(addresses)) != len(addresses)
        or any(address <= 0 for address in addresses)
    ):
        raise ValueError("invalid gameplay MTRand breakpoint addresses")

    thread = kernel32.OpenThread(
        THREAD_ACCESS | THREAD_SUSPEND_RESUME,
        False,
        main_thread_id,
    )
    if not thread:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        context = _read_debug_context(thread)
        original = (
            int(context.Dr0),
            int(context.Dr1),
            int(context.Dr2),
            int(context.Dr3),
            int(context.Dr6),
            int(context.Dr7),
        )
        enable_mask = sum(0x3 << (slot * 2) for slot in range(len(addresses)))
        config_mask = sum(
            0xF << (16 + slot * 4) for slot in range(len(addresses))
        )
        if int(context.Dr7) & enable_mask:
            raise RuntimeError(
                "gameplay MTRand hardware breakpoint slot is occupied"
            )
        for slot, address in enumerate(addresses):
            setattr(context, f"Dr{slot}", address)
        context.Dr6 = 0
        context.Dr7 &= ~(enable_mask | config_mask)
        for slot in range(len(addresses)):
            context.Dr7 |= 0x1 << (slot * 2)
        _write_debug_context(thread, context)
        verified = _read_debug_context(thread)
        if (
            any(
                int(getattr(verified, f"Dr{slot}")) != address
                for slot, address in enumerate(addresses)
            )
            or any(
                int(verified.Dr7) & (0x1 << (slot * 2)) == 0
                for slot in range(len(addresses))
            )
            or int(verified.Dr7) & config_mask
        ):
            raise RuntimeError(
                "gameplay MTRand hardware breakpoint writeback mismatch"
            )
        return original
    finally:
        kernel32.CloseHandle(thread)


def _restore_gameplay_mtrand_breakpoints(
    *,
    main_thread_id: int,
    original: tuple[int, int, int, int, int, int],
) -> None:
    """Restore all debug registers while the debugger still owns the PID."""

    thread = kernel32.OpenThread(
        THREAD_ACCESS | THREAD_SUSPEND_RESUME,
        False,
        main_thread_id,
    )
    if not thread:
        raise ctypes.WinError(ctypes.get_last_error())
    suspended = False
    resume_error: OSError | None = None
    try:
        previous = int(kernel32.SuspendThread(thread))
        if previous == INVALID_SUSPEND_COUNT:
            raise ctypes.WinError(ctypes.get_last_error())
        suspended = True
        context = _read_debug_context(thread)
        (
            context.Dr0,
            context.Dr1,
            context.Dr2,
            context.Dr3,
            context.Dr6,
            context.Dr7,
        ) = original
        context.EFlags |= RESUME_FLAG
        _write_debug_context(thread, context)
        verified = _read_debug_context(thread)
        observed = (
            int(verified.Dr0),
            int(verified.Dr1),
            int(verified.Dr2),
            int(verified.Dr3),
            int(verified.Dr6),
            int(verified.Dr7),
        )
        if observed != original:
            raise RuntimeError(
                "gameplay MTRand hardware breakpoint restore mismatch"
            )
    finally:
        if suspended:
            result = int(kernel32.ResumeThread(thread))
            if result == INVALID_SUSPEND_COUNT:
                resume_error = ctypes.WinError(ctypes.get_last_error())
        kernel32.CloseHandle(thread)
        if resume_error is not None:
            raise resume_error


def _gameplay_mtrand_board_snapshot(process: int) -> dict[str, int]:
    app = _read_u32(process, G_SEXY_APP_BASE_ADDRESS)
    if not app:
        raise RuntimeError("gameplay MTRand app pointer is null")
    board = _read_u32(process, app + ACTIVE_BOARD_OFFSET)
    if not board:
        raise RuntimeError("gameplay MTRand Board pointer is null")
    return {
        "framework_update": _read_i32(process, app + 0x4C4),
        "native_game_time": _read_i32(
            process,
            board + BOARD_NATIVE_GAME_TIME_OFFSET,
        ),
        "score": _read_i32(process, board + BOARD_SCORE_OFFSET),
        "score_target": _read_i32(
            process,
            board + BOARD_SCORE_TARGET_OFFSET,
        ),
    }


def _startup_global_mtrand_framework_update(process: int) -> int:
    """Read the safest available startup timeline without requiring a Board."""

    app = _read_u32(process, G_SEXY_APP_BASE_ADDRESS)
    if not app:
        return -1
    return _read_i32(process, app + 0x4C4)


def _mtrand_state(seed: int) -> bytes:
    """Return the exact 32-bit retail MTRand state after SRand(seed)."""

    if not 0 <= seed <= 0xFFFFFFFF:
        raise ValueError("MTRand seed must fit uint32")
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
    return struct.pack(f"<{len(words)}I", *words)


def _global_mtrand_source_evidence(
    state: GlobalMTRandRestoreState,
) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "source_path": str(state.source_path),
        "source_sha256": state.source_sha256,
        "source_process_id": state.source_process_id,
        "source_framework_update": state.source_framework_update,
        "source_native_game_time": state.source_native_game_time,
        "seed": state.seed,
        "captured_draw_count": state.captured_draw_count,
        "captured_index": state.captured_index,
        "captured_state_sha256": state.captured_state_sha256,
        "rewind_draws": state.rewind_draws,
        "draw_count": state.draw_count,
        "index": state.index,
        "state_sha256": state.state_sha256,
        "semantic_sha256": state.semantic_sha256,
    }
    if state.source_kind != "live_rng_monitor":
        evidence["source_kind"] = state.source_kind
    for name, value in (
        ("source_trace_path", state.source_trace_path),
        ("source_trace_sha256", state.source_trace_sha256),
        (
            "source_recording_report_path",
            state.source_recording_report_path,
        ),
        (
            "source_recording_report_sha256",
            state.source_recording_report_sha256,
        ),
        ("source_dmo_path", state.source_dmo_path),
        ("source_dmo_sha256", state.source_dmo_sha256),
        ("source_call_order", state.source_call_order),
        ("source_caller", state.source_caller),
    ):
        if value is not None:
            evidence[name] = str(value) if isinstance(value, Path) else value
    if state.source_caller is not None:
        evidence["source_caller_hex"] = f"0x{state.source_caller:08x}"
    return evidence


def _thread_crt_source_evidence(
    state: ThreadCrtRestoreState,
) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "source_path": str(state.source_path),
        "source_sha256": state.source_sha256,
        "source_process_id": state.source_process_id,
        "source_thread_id": state.source_thread_id,
        "source_framework_update": state.source_framework_update,
        "source_native_game_time": state.source_native_game_time,
        "captured_state": state.captured_state,
        "rewind_draws": state.rewind_draws,
        "state": state.state,
        "semantic_sha256": state.semantic_sha256,
    }
    if state.source_kind != "live_rng_monitor":
        evidence["source_kind"] = state.source_kind
    for name, value in (
        (
            "source_recording_report_path",
            state.source_recording_report_path,
        ),
        (
            "source_recording_report_sha256",
            state.source_recording_report_sha256,
        ),
        ("source_dmo_path", state.source_dmo_path),
        ("source_dmo_sha256", state.source_dmo_sha256),
        ("source_call_order", state.source_call_order),
        ("source_caller", state.source_caller),
    ):
        if value is not None:
            evidence[name] = str(value) if isinstance(value, Path) else value
    if state.source_caller is not None:
        evidence["source_caller_hex"] = f"0x{state.source_caller:08x}"
    return evidence


def _global_mtrand_call_transition(
    payload: bytes,
) -> tuple[int, bytes, int, str]:
    """Execute one exact retail MTRand draw from a captured state."""

    if len(payload) != MTRAND_STATE_BYTES:
        raise RuntimeError("source-bound global MTRand payload size mismatch")
    unpacked = struct.unpack(f"<{MTRAND_STATE_WORDS + 1}I", payload)
    index = unpacked[-1]
    if index > MTRAND_STATE_WORDS:
        raise RuntimeError("source-bound global MTRand index is invalid")
    rng = PopCapMTRandom(1)
    rng.load_state(unpacked[:MTRAND_STATE_WORDS], index)
    output = rng.next_u31()
    post_payload = struct.pack(
        f"<{MTRAND_STATE_WORDS + 1}I",
        *rng.words,
        rng.index,
    )
    return (
        output,
        post_payload,
        rng.index,
        "sha256:" + hashlib.sha256(post_payload).hexdigest(),
    )


def _live_global_post_matches_observed_draw(
    live_post: bytes,
    observed_output: int,
) -> bool:
    """Verify one live post-state by reconstructing its exact prior draw."""

    if (
        len(live_post) != MTRAND_STATE_BYTES
        or isinstance(observed_output, bool)
        or not isinstance(observed_output, int)
        or not 0 <= observed_output <= 0x7FFFFFFF
    ):
        return False
    live_index = struct.unpack_from(
        "<I",
        live_post,
        MTRAND_STATE_WORDS * 4,
    )[0]
    if not 1 < live_index <= MTRAND_STATE_WORDS:
        return False
    reconstructed_pre = bytearray(live_post)
    struct.pack_into(
        "<I",
        reconstructed_pre,
        MTRAND_STATE_WORDS * 4,
        live_index - 1,
    )
    try:
        output, reconstructed_post, _, _ = (
            _global_mtrand_call_transition(bytes(reconstructed_pre))
        )
    except RuntimeError:
        return False
    return output == observed_output and reconstructed_post == live_post


def _validate_source_bound_board_anchor_states(
    global_state: GlobalMTRandRestoreState,
    thread_state: ThreadCrtRestoreState,
) -> None:
    """Require both desired states to name one immutable natural call."""

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
        or global_state.source_call_order != thread_state.source_call_order
        or global_state.source_caller != thread_state.source_caller
        or global_state.source_call_order is None
        or global_state.source_caller is None
        or thread_state.source_framework_update < 0
    ):
        raise ValueError("source-bound board anchor source identity mismatch")


def _apply_source_bound_board_anchor(
    *,
    process: int,
    process_id: int,
    thread: int,
    thread_id: int,
    context: WOW64_CONTEXT,
    framework_update: int,
    expected_main_thread_id: int,
    board_seed_address: int,
    board_original: bytes,
    active_board_address: int,
    rng_object_address: int,
    rng_owner_address: int,
    rng_owner_vtable: int,
    observed_seed: int,
    expected_effective_seed: int,
    allow_global_correction: bool,
    global_state: GlobalMTRandRestoreState,
    thread_state: ThreadCrtRestoreState,
) -> dict[str, Any]:
    """Anchor replay after one natural or bounded-corrected source call."""

    _validate_source_bound_board_anchor_states(global_state, thread_state)
    source_caller = global_state.source_caller
    assert source_caller is not None
    expected_bridge = bytes.fromhex("8d564c8bc88bc2")
    if (
        board_seed_address != DEFAULT_BOARD_RESEED_CALL
        or source_caller != 0x0065B821
        or board_seed_address - source_caller != len(expected_bridge)
        or board_original != b"\xE8"
        or read_memory(process, source_caller, len(expected_bridge))
        != expected_bridge
    ):
        raise RuntimeError(
            "source-bound board anchor instruction bridge mismatch"
        )
    (
        source_output,
        desired_global_post,
        desired_global_post_index,
        desired_global_post_sha256,
    ) = _global_mtrand_call_transition(global_state.payload)
    if (
        framework_update != thread_state.source_framework_update
        or thread_id != expected_main_thread_id
        or rng_owner_address <= 0
        or rng_owner_vtable != BOARD_SEED_OWNER_VTABLE
        or rng_object_address != rng_owner_address + BOARD_MTRAND_OFFSET
        or active_board_address <= 0
        or _read_u32(process, active_board_address) != BOARD_VTABLE
        or not isinstance(allow_global_correction, bool)
    ):
        raise RuntimeError("source-bound board anchor live identity mismatch")
    if source_output != expected_effective_seed:
        raise RuntimeError("source-bound board seed does not match source call")

    active_board_native_game_time = _read_i32(
        process,
        active_board_address + BOARD_NATIVE_GAME_TIME_OFFSET,
    )
    if active_board_native_game_time < 0:
        raise RuntimeError(
            "source-bound board native game time is invalid"
        )

    live_global = read_memory(
        process,
        G_FRAMEWORK_MTRAND_ADDRESS,
        MTRAND_STATE_BYTES,
    )
    live_global_index = struct.unpack_from(
        "<I",
        live_global,
        MTRAND_STATE_WORDS * 4,
    )[0]
    if live_global_index > MTRAND_STATE_WORDS:
        raise RuntimeError("live post-call global MTRand index is invalid")
    live_global_sha256 = "sha256:" + hashlib.sha256(live_global).hexdigest()
    live_post_draw_verified = _live_global_post_matches_observed_draw(
        live_global,
        observed_seed,
    )
    correction_index_delta = _source_bound_global_post_correction_delta(
        live_global,
        desired_global_post,
    )
    natural_source_call = (
        observed_seed == source_output
        and live_global == desired_global_post
    )
    if (
        not live_post_draw_verified
        or (
            not natural_source_call
            and (
                not allow_global_correction
                or correction_index_delta is None
            )
        )
    ):
        raise RuntimeError(
            "source-bound board anchor global correction is ineligible"
        )

    replay_thread_state = _thread_crt_rng_state(
        process=process,
        thread=thread,
        context=context,
        expected_thread_id=thread_id,
    )
    crt_address = replay_thread_state["rand_state_address"]
    live_crt = _read_u32(process, crt_address)
    desired_crt = thread_state.state

    global_changed = live_global != desired_global_post
    bounded_global_correction_applied = (
        not natural_source_call and global_changed
    )
    crt_changed = live_crt != desired_crt
    applied: list[tuple[int, bytes]] = []
    try:
        if global_changed:
            write_memory(
                process,
                G_FRAMEWORK_MTRAND_ADDRESS,
                desired_global_post,
            )
            applied.append((G_FRAMEWORK_MTRAND_ADDRESS, live_global))
        if crt_changed:
            write_memory(process, crt_address, struct.pack("<I", desired_crt))
            applied.append((crt_address, struct.pack("<I", live_crt)))
        if (
            read_memory(
                process,
                G_FRAMEWORK_MTRAND_ADDRESS,
                MTRAND_STATE_BYTES,
            )
            != desired_global_post
            or _read_u32(process, crt_address) != desired_crt
        ):
            raise RuntimeError(
                "source-bound board anchor writeback verification mismatch"
            )
    except BaseException:
        rollback_failures: list[str] = []
        for address, before in reversed(applied):
            try:
                write_memory(process, address, before)
                if read_memory(process, address, len(before)) != before:
                    rollback_failures.append(f"0x{address:08X}:verify")
            except BaseException as error:
                rollback_failures.append(
                    f"0x{address:08X}:{type(error).__name__}"
                )
        if rollback_failures:
            raise RuntimeError(
                "source-bound board anchor rollback was incomplete: "
                + ",".join(rollback_failures)
            )
        raise

    return {
        "classification": (
            (
                "source-bound-bounded-global-correction-board-anchor"
                if bounded_global_correction_applied
                else "source-bound-natural-retail-post-call-board-anchor"
            )
        ),
        "process_id": process_id,
        "thread_id": thread_id,
        "perf_counter_ns": time.perf_counter_ns(),
        "framework_update": framework_update,
        "board_seed_call_address": board_seed_address,
        "source_call_return_address": source_caller,
        "source_call_return_address_hex": f"0x{source_caller:08x}",
        "instruction_bridge_hex": expected_bridge.hex(),
        "instruction_bridge_contains_call": False,
        "observed_seed": observed_seed,
        "effective_seed": expected_effective_seed,
        "observed_seed_matches_source": observed_seed == source_output,
        "identity_mechanism": (
            "board_rng_owner_vtable_and_source_call_state"
        ),
        "expected_main_thread_id": expected_main_thread_id,
        "active_board_address": active_board_address,
        "active_board_native_game_time": (
            active_board_native_game_time
        ),
        "rng_object_address": rng_object_address,
        "rng_owner_address": rng_owner_address,
        "rng_owner_vtable": rng_owner_vtable,
        "active_board_vtable": _read_u32(process, active_board_address),
        "rng_owner_is_active_board": active_board_address == rng_owner_address,
        "global": {
            "state_address": G_FRAMEWORK_MTRAND_ADDRESS,
            "live_before_index": live_global_index,
            "live_before_state_sha256": live_global_sha256,
            "natural_post_state_match": live_global == desired_global_post,
            "live_post_draw_verified": live_post_draw_verified,
            "same_state_words": (
                live_global[: MTRAND_STATE_WORDS * 4]
                == desired_global_post[: MTRAND_STATE_WORDS * 4]
            ),
            "global_correction_authorized": allow_global_correction,
            "bounded_global_correction_applied": (
                bounded_global_correction_applied
            ),
            "correction_index_delta": correction_index_delta,
            "maximum_correction_draw_delta": (
                SOURCE_BOUND_GLOBAL_CORRECTION_MAX_DRAWS
            ),
            "changed": global_changed,
            "bytes_written": MTRAND_STATE_BYTES if global_changed else 0,
            "restored_pre_call_source": _global_mtrand_source_evidence(
                global_state
            ),
            "source_call_output": source_output,
            "source_call_post_index": desired_global_post_index,
            "source_call_post_state_sha256": desired_global_post_sha256,
        },
        "thread_crt": {
            **replay_thread_state,
            "live_before_state": live_crt,
            "changed": crt_changed,
            "bytes_written": 4 if crt_changed else 0,
            "restored_state": desired_crt,
            "restored_source": _thread_crt_source_evidence(thread_state),
        },
        "bytes_written": (
            (MTRAND_STATE_BYTES if global_changed else 0)
            + (4 if crt_changed else 0)
        ),
        "writeback_verified": True,
        "transactional_rollback_on_failure": True,
        "process_memory_mutation": global_changed or crt_changed,
        "persistent_file_modified": False,
    }


def _apply_global_mtrand_restore(
    *,
    process: int,
    process_id: int,
    thread_id: int,
    snapshot: Mapping[str, int],
    desired: GlobalMTRandRestoreState,
    expected_before: GlobalMTRandRestoreState,
    command_order: int,
    allow_dynamic_before: bool,
) -> dict[str, Any]:
    """Transactionally restore the shared global MTRand at a DMO boundary."""

    board_address = _read_u32(
        process,
        int(snapshot["base"]) + ACTIVE_BOARD_OFFSET,
    )
    if not board_address:
        raise RuntimeError("global MTRand restore active Board pointer is null")
    if _read_u32(process, board_address) != BOARD_VTABLE:
        raise RuntimeError(
            "global MTRand restore active Board identity mismatch"
        )
    before = read_memory(
        process,
        G_FRAMEWORK_MTRAND_ADDRESS,
        MTRAND_STATE_BYTES,
    )
    if (
        len(desired.payload) != MTRAND_STATE_BYTES
        or len(expected_before.payload) != MTRAND_STATE_BYTES
    ):
        raise RuntimeError("global MTRand restore payload size mismatch")
    reference_match = before == expected_before.payload
    if not reference_match and not allow_dynamic_before:
        raise RuntimeError(
            "global MTRand restore live baseline differs from captured "
            "expectation"
        )
    before_index = struct.unpack_from(
        "<I",
        before,
        MTRAND_STATE_WORDS * 4,
    )[0]
    if before_index > MTRAND_STATE_WORDS:
        raise RuntimeError("live global MTRand index is invalid")
    before_sha256 = "sha256:" + hashlib.sha256(before).hexdigest()
    changed = before != desired.payload
    try:
        if changed:
            write_memory(
                process,
                G_FRAMEWORK_MTRAND_ADDRESS,
                desired.payload,
            )
        after = read_memory(
            process,
            G_FRAMEWORK_MTRAND_ADDRESS,
            MTRAND_STATE_BYTES,
        )
        if after != desired.payload:
            raise RuntimeError(
                "global MTRand restore writeback verification mismatch"
            )
    except Exception as error:
        rollback_error: Exception | None = None
        try:
            if changed:
                write_memory(
                    process,
                    G_FRAMEWORK_MTRAND_ADDRESS,
                    before,
                )
            if read_memory(
                process,
                G_FRAMEWORK_MTRAND_ADDRESS,
                MTRAND_STATE_BYTES,
            ) != before:
                raise RuntimeError(
                    "global MTRand rollback verification failed"
                )
        except Exception as caught:
            rollback_error = caught
        if rollback_error is not None:
            raise RuntimeError(
                "global MTRand restore failed and rollback was incomplete: "
                f"{rollback_error}"
            ) from error
        raise
    return {
        "process_id": process_id,
        "thread_id": thread_id,
        "perf_counter_ns": time.perf_counter_ns(),
        "framework_update": int(snapshot["update"]),
        "command_order": command_order,
        "board_address": board_address,
        "state_address": G_FRAMEWORK_MTRAND_ADDRESS,
        "expected_before": {
            **_global_mtrand_source_evidence(expected_before),
            "live_reference_match": reference_match,
        },
        "live_before_index": before_index,
        "live_before_state_sha256": before_sha256,
        "dynamic_before_allowed": allow_dynamic_before,
        "restored": _global_mtrand_source_evidence(desired),
        "changed": changed,
        "bytes_written": MTRAND_STATE_BYTES if changed else 0,
        "transactional_rollback": True,
        "writeback_verified": True,
        "persistent_file_modified": False,
        "classification": "diagnostic-ephemeral-state-restore",
    }


def _apply_source_bound_global_precall_restore(
    *,
    process: int,
    process_id: int,
    thread_id: int,
    snapshot: Mapping[str, int],
    desired: GlobalMTRandRestoreState,
    expected_main_thread_id: int,
    target_framework_update: int,
    call_address: int,
    call_instruction: bytes,
) -> dict[str, Any]:
    """Restore the natural global state at the exact seed-producing CALL."""

    if (
        thread_id != expected_main_thread_id
        or int(snapshot["update"]) != target_framework_update
        or call_address != SOURCE_BOUND_GLOBAL_PRECALL_ADDRESS
        or call_instruction != SOURCE_BOUND_GLOBAL_PRECALL_INSTRUCTION
        or call_address + len(call_instruction)
        != SOURCE_BOUND_GLOBAL_PRECALL_RETURN_ADDRESS
    ):
        raise RuntimeError("source-bound global precall identity mismatch")
    relative_target = struct.unpack_from("<i", call_instruction, 1)[0]
    call_target = call_address + len(call_instruction) + relative_target
    if call_target != SOURCE_BOUND_GLOBAL_PRECALL_TARGET_ADDRESS:
        raise RuntimeError("source-bound global precall target mismatch")
    base_receipt = _apply_global_mtrand_restore(
        process=process,
        process_id=process_id,
        thread_id=thread_id,
        snapshot=snapshot,
        desired=desired,
        expected_before=desired,
        command_order=-1,
        allow_dynamic_before=True,
    )
    base_receipt.pop("command_order")
    base_receipt.pop("classification")
    active_board_address = base_receipt.pop("board_address")
    active_board_vtable = _read_u32(process, active_board_address)
    if active_board_vtable != BOARD_VTABLE:
        raise RuntimeError("source-bound global precall Board identity mismatch")
    return {
        "mechanism": "exact_source_bound_global_rng_precall_state_restore",
        **base_receipt,
        "call_address": call_address,
        "return_address": SOURCE_BOUND_GLOBAL_PRECALL_RETURN_ADDRESS,
        "target_address": call_target,
        "instruction_hex": call_instruction.hex(),
        "expected_main_thread_id": expected_main_thread_id,
        "main_thread_match": True,
        "active_board_address": active_board_address,
        "active_board_vtable": active_board_vtable,
        "process_memory_mutation": bool(base_receipt["changed"]),
        "classification": "source-bound-natural-retail-global-precall-restore",
    }


def _apply_qrand_restore(
    *,
    process: int,
    process_id: int,
    thread_id: int,
    snapshot: Mapping[str, int],
    desired: QRandRestoreState,
    expected_before: QRandRestoreState,
    command_order: int,
) -> dict[str, Any]:
    """Transactionally restore one captured QRand state at a DMO boundary."""

    board_address = _read_u32(
        process,
        int(snapshot["base"]) + ACTIVE_BOARD_OFFSET,
    )
    if not board_address:
        raise RuntimeError("QRand restore active Board pointer is null")
    if _read_u32(process, board_address) != BOARD_VTABLE:
        raise RuntimeError("QRand restore active Board identity mismatch")
    qrand_address = _read_u32(
        process,
        board_address + BOARD_QRAND_POINTER_OFFSET,
    )
    if not qrand_address:
        raise RuntimeError("QRand restore object pointer is null")
    object_bytes = read_memory(
        process,
        qrand_address,
        QRAND_OBJECT_SIZE,
    )
    expected_writes = (
        (qrand_address, qrand_scalar_payload(expected_before)),
        *qrand_vector_write_plan(object_bytes, expected_before),
    )
    desired_writes = (
        (qrand_address, qrand_scalar_payload(desired)),
        *qrand_vector_write_plan(object_bytes, desired),
    )
    if tuple(address for address, _ in expected_writes) != tuple(
        address for address, _ in desired_writes
    ):
        raise RuntimeError("QRand restore write-plan address mismatch")

    spans: list[tuple[int, bytes, bytes]] = []
    for (
        expected_address,
        expected_payload,
    ), (
        desired_address,
        desired_payload,
    ) in zip(expected_writes, desired_writes, strict=True):
        if len(expected_payload) != len(desired_payload):
            raise RuntimeError("QRand restore write-plan size mismatch")
        live_payload = read_memory(
            process,
            expected_address,
            len(expected_payload),
        )
        if live_payload != expected_payload:
            raise RuntimeError(
                "QRand restore live baseline differs from captured "
                f"expectation at 0x{expected_address:08X}"
            )
        spans.append(
            (desired_address, live_payload, desired_payload)
        )

    changed = [
        span
        for span in spans
        if span[1] != span[2]
    ]
    applied: list[tuple[int, bytes, bytes]] = []
    try:
        for address, before, after in changed:
            write_memory(process, address, after)
            applied.append((address, before, after))
        for address, _, after in spans:
            if read_memory(process, address, len(after)) != after:
                raise RuntimeError(
                    "QRand restore writeback verification mismatch"
                )
    except BaseException:
        rollback_failures: list[str] = []
        for address, before, _ in reversed(applied):
            try:
                write_memory(process, address, before)
                if read_memory(process, address, len(before)) != before:
                    rollback_failures.append(
                        f"0x{address:08X}:verify"
                    )
            except BaseException as error:
                rollback_failures.append(
                    f"0x{address:08X}:{type(error).__name__}"
                )
        if rollback_failures:
            raise RuntimeError(
                "QRand restore failed and rollback was incomplete: "
                + ",".join(rollback_failures)
            )
        raise

    return {
        "process_id": process_id,
        "thread_id": thread_id,
        "perf_counter_ns": time.perf_counter_ns(),
        "framework_update": int(snapshot["update"]),
        "command_order": command_order,
        "board_address": board_address,
        "qrand_address": qrand_address,
        "expected_before": {
            "source_path": str(expected_before.source_path),
            "source_sha256": expected_before.source_sha256,
            "source_framework_update": (
                expected_before.source_framework_update
            ),
            "source_native_game_time": (
                expected_before.source_native_game_time
            ),
            "semantic_sha256": expected_before.semantic_sha256,
        },
        "restored": {
            "source_path": str(desired.source_path),
            "source_sha256": desired.source_sha256,
            "source_framework_update": desired.source_framework_update,
            "source_native_game_time": desired.source_native_game_time,
            "semantic_sha256": desired.semantic_sha256,
        },
        "validated_span_count": len(spans),
        "changed_span_count": len(changed),
        "bytes_written": sum(len(after) for _, _, after in changed),
        "transactional_rollback": True,
        "writeback_verified": True,
        "persistent_file_modified": False,
        "classification": "diagnostic-ephemeral-state-restore",
    }


def _apply_thread_crt_restore(
    *,
    process: int,
    process_id: int,
    thread: int,
    thread_id: int,
    context: WOW64_CONTEXT,
    snapshot: Mapping[str, int],
    desired: ThreadCrtRestoreState,
    expected_before: ThreadCrtRestoreState,
    command_order: int,
    allow_dynamic_before: bool,
) -> dict[str, Any]:
    """Transactionally restore the paused main thread's CRT rand state."""

    board_address = _read_u32(
        process,
        snapshot["base"] + ACTIVE_BOARD_OFFSET,
    )
    if not board_address:
        raise RuntimeError("thread CRT restore active Board pointer is null")
    if _read_u32(process, board_address) != BOARD_VTABLE:
        raise RuntimeError("thread CRT restore active Board identity mismatch")
    thread_state = _thread_crt_rng_state(
        process=process,
        thread=thread,
        context=context,
        expected_thread_id=thread_id,
    )
    state_address = thread_state["rand_state_address"]
    before = _read_u32(process, state_address)
    reference_match = before == expected_before.state
    if not reference_match and not allow_dynamic_before:
        raise RuntimeError(
            "thread CRT restore live baseline differs from captured "
            f"expected state: live=0x{before:08X}, "
            f"expected=0x{expected_before.state:08X}"
        )
    changed = before != desired.state
    try:
        if changed:
            write_memory(
                process,
                state_address,
                struct.pack("<I", desired.state),
            )
        after = _read_u32(process, state_address)
        if after != desired.state:
            raise RuntimeError(
                "thread CRT restore writeback verification mismatch"
            )
    except Exception as error:
        rollback_error: Exception | None = None
        try:
            if changed:
                write_memory(
                    process,
                    state_address,
                    struct.pack("<I", before),
                )
            if _read_u32(process, state_address) != before:
                raise RuntimeError("thread CRT rollback verification failed")
        except Exception as caught:
            rollback_error = caught
        if rollback_error is not None:
            raise RuntimeError(
                "thread CRT restore failed and rollback was incomplete: "
                f"{rollback_error}"
            ) from error
        raise
    return {
        "process_id": process_id,
        "thread_id": thread_id,
        "perf_counter_ns": time.perf_counter_ns(),
        "framework_update": int(snapshot["update"]),
        "command_order": command_order,
        "board_address": board_address,
        "state_address": state_address,
        "thread_resolution": thread_state,
        "expected_before": {
            "source_path": str(expected_before.source_path),
            "source_sha256": expected_before.source_sha256,
            "source_process_id": expected_before.source_process_id,
            "source_thread_id": expected_before.source_thread_id,
            "source_framework_update": (
                expected_before.source_framework_update
            ),
            "source_native_game_time": (
                expected_before.source_native_game_time
            ),
            "captured_state": expected_before.captured_state,
            "captured_state_hex": (
                f"0x{expected_before.captured_state:08X}"
            ),
            "rewind_draws": expected_before.rewind_draws,
            "state": expected_before.state,
            "state_hex": f"0x{expected_before.state:08X}",
            "semantic_sha256": expected_before.semantic_sha256,
            "live_reference_match": reference_match,
        },
        "live_before_state": before,
        "live_before_state_hex": f"0x{before:08X}",
        "dynamic_before_allowed": allow_dynamic_before,
        "restored": {
            "source_path": str(desired.source_path),
            "source_sha256": desired.source_sha256,
            "source_process_id": desired.source_process_id,
            "source_thread_id": desired.source_thread_id,
            "source_framework_update": desired.source_framework_update,
            "source_native_game_time": desired.source_native_game_time,
            "captured_state": desired.captured_state,
            "captured_state_hex": f"0x{desired.captured_state:08X}",
            "rewind_draws": desired.rewind_draws,
            "state": desired.state,
            "state_hex": f"0x{desired.state:08X}",
            "semantic_sha256": desired.semantic_sha256,
        },
        "changed": changed,
        "bytes_written": 4 if changed else 0,
        "transactional_rollback": True,
        "writeback_verified": True,
        "persistent_file_modified": False,
        "classification": "diagnostic-ephemeral-state-restore",
    }


def _wow64_segment_base(
    thread: int,
    selector: int,
) -> int:
    """Resolve a WOW64 selector to the linear base used by the debugger."""

    entry = WOW64_LDT_ENTRY()
    if not kernel32.Wow64GetThreadSelectorEntry(
        thread,
        selector,
        ctypes.byref(entry),
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    return (
        int(entry.BaseLow)
        | (int(entry.HighWord.Bytes.BaseMid) << 16)
        | (int(entry.HighWord.Bytes.BaseHi) << 24)
    )


def _thread_crt_rng_state(
    *,
    process: int,
    thread: int,
    context: WOW64_CONTEXT,
    expected_thread_id: int,
) -> dict[str, int]:
    """Locate and validate the retail CRT PTD for one paused WOW64 thread."""

    fs_selector = int(context.SegFs)
    fs_segment_base = _wow64_segment_base(thread, fs_selector)
    teb_address = _read_u32(process, fs_segment_base + 0x18)
    if not teb_address:
        raise RuntimeError("WOW64 FS:[0x18] resolved to a null TEB")
    fls_index = _read_u32(process, CRT_FLS_INDEX_ADDRESS)
    if not 0 < fls_index < FLS_MAXIMUM_AVAILABLE:
        raise RuntimeError(f"CRT FLS index is invalid: {fls_index}")
    fls_data_address = _read_u32(
        process,
        teb_address + WOW64_TEB_FLS_DATA_OFFSET,
    )
    if not fls_data_address:
        raise RuntimeError("WOW64 TEB contains a null FLS data pointer")
    shifted_index = fls_index + 0x10
    fls_block_index = shifted_index.bit_length() - 1
    fls_entry_index = shifted_index ^ (1 << fls_block_index)
    fls_block_slot_address = (
        fls_data_address + fls_block_index * 4 - 8
    )
    fls_block_address = _read_u32(
        process,
        fls_block_slot_address,
    )
    if not fls_block_address:
        raise RuntimeError("CRT FLS block pointer is null")
    fls_value_address = (
        fls_block_address + fls_entry_index * 4 + 4
    )
    ptd_address = _read_u32(process, fls_value_address)
    if not ptd_address:
        raise RuntimeError("CRT FLS value contains a null PTD")
    ptd_thread_id = _read_u32(
        process,
        ptd_address + CRT_PTD_THREAD_ID_OFFSET,
    )
    if ptd_thread_id != expected_thread_id:
        raise RuntimeError(
            "CRT PTD thread identity mismatch: "
            f"expected={expected_thread_id}, observed={ptd_thread_id}, "
            f"fs_selector=0x{fs_selector:04X}, "
            f"fs_segment_base=0x{fs_segment_base:08X}, "
            f"teb=0x{teb_address:08X}, fls_index={fls_index}, "
            f"fls_data=0x{fls_data_address:08X}, "
            f"fls_block=0x{fls_block_address:08X}, "
            f"fls_value=0x{fls_value_address:08X}, "
            f"ptd=0x{ptd_address:08X}"
        )
    return {
        "thread_id": expected_thread_id,
        "fs_selector": fs_selector,
        "fs_segment_base": fs_segment_base,
        "teb_address": teb_address,
        "fls_index_address": CRT_FLS_INDEX_ADDRESS,
        "fls_index": fls_index,
        "fls_data_address": fls_data_address,
        "fls_block_index": fls_block_index,
        "fls_entry_index": fls_entry_index,
        "fls_block_slot_address": fls_block_slot_address,
        "fls_block_address": fls_block_address,
        "fls_value_address": fls_value_address,
        "ptd_address": ptd_address,
        "ptd_thread_id": ptd_thread_id,
        "rand_state_address": (
            ptd_address + CRT_PTD_RAND_STATE_OFFSET
        ),
    }


def _apply_board_rng_overrides(
    *,
    process: int,
    thread: int,
    context: WOW64_CONTEXT,
    process_id: int,
    thread_id: int,
    board_seed_address: int,
    board_seed_override: int | None,
    global_rng_seed_override: int | None,
    thread_crt_rng_seed_override: int | None,
) -> dict[str, Any]:
    """Apply and verify all one-shot board RNG overrides at the retail CALL."""

    observed_seed = int(context.Ecx)
    effective_seed = (
        observed_seed
        if board_seed_override is None
        else board_seed_override
    )
    base = _read_u32(process, G_SEXY_APP_BASE_ADDRESS)
    framework_update = (
        _read_i32(process, base + 0x4C4)
        if base
        else -1
    )
    global_rng_evidence: dict[str, Any] = {}
    if global_rng_seed_override is not None:
        global_state_before = read_memory(
            process,
            G_FRAMEWORK_MTRAND_ADDRESS,
            MTRAND_STATE_BYTES,
        )
        expected_global_state = _mtrand_state(
            global_rng_seed_override
        )
        write_memory(
            process,
            G_FRAMEWORK_MTRAND_ADDRESS,
            expected_global_state,
        )
        global_state_after = read_memory(
            process,
            G_FRAMEWORK_MTRAND_ADDRESS,
            MTRAND_STATE_BYTES,
        )
        if global_state_after != expected_global_state:
            raise RuntimeError(
                "global framework RNG state writeback mismatch"
            )
        global_rng_evidence = {
            "global_rng_address": G_FRAMEWORK_MTRAND_ADDRESS,
            "global_rng_seed": global_rng_seed_override,
            "global_rng_state_bytes": MTRAND_STATE_BYTES,
            "global_rng_state_sha256_before": (
                "sha256:"
                + hashlib.sha256(global_state_before).hexdigest()
            ),
            "global_rng_state_sha256_after": (
                "sha256:"
                + hashlib.sha256(global_state_after).hexdigest()
            ),
            "global_rng_overridden": True,
        }
    thread_crt_rng_evidence: dict[str, Any] = {}
    if thread_crt_rng_seed_override is not None:
        thread_crt_state = _thread_crt_rng_state(
            process=process,
            thread=thread,
            context=context,
            expected_thread_id=thread_id,
        )
        rand_state_address = thread_crt_state[
            "rand_state_address"
        ]
        rand_state_before = _read_u32(
            process,
            rand_state_address,
        )
        write_memory(
            process,
            rand_state_address,
            struct.pack("<I", thread_crt_rng_seed_override),
        )
        rand_state_after = _read_u32(
            process,
            rand_state_address,
        )
        if rand_state_after != thread_crt_rng_seed_override:
            raise RuntimeError(
                "thread CRT RNG state writeback mismatch"
            )
        thread_crt_rng_evidence = {
            **thread_crt_state,
            "thread_crt_rng_state_before": rand_state_before,
            "thread_crt_rng_state_after": rand_state_after,
            "thread_crt_rng_seed": thread_crt_rng_seed_override,
            "thread_crt_rng_overridden": True,
        }
    observation = {
        "process_id": process_id,
        "thread_id": thread_id,
        "perf_counter_ns": time.perf_counter_ns(),
        "framework_update": framework_update,
        "call_site_address": board_seed_address,
        "rng_object_address": int(context.Eax),
        "observed_seed": observed_seed,
        "effective_seed": effective_seed,
        "overridden": board_seed_override is not None,
        **global_rng_evidence,
        **thread_crt_rng_evidence,
    }
    context.Ecx = effective_seed
    return observation


def _command_snapshot(
    process: int,
    context: WOW64_CONTEXT,
) -> dict[str, int | str | bool]:
    base = int(context.Edi)
    return_address, required = struct.unpack(
        "<II", read_memory(process, context.Esp, 8)
    )
    snapshot = {
        "return_address": return_address,
        "required": required,
        "base": base,
        "vtable": _read_u32(process, base),
        "update": _read_i32(process, base + 0x4C4),
        "buffer_read_bit_position": _read_i32(process, base + 0x608),
        "last_demo_update": _read_i32(process, base + 0x61C),
        "needs_command": read_memory(process, base + 0x620, 1)[0],
        "is_short": read_memory(process, base + 0x621, 1)[0],
        "command_number": _read_i32(process, base + 0x624),
        "command_order": _read_i32(process, base + 0x628),
        "command_bit_position": _read_i32(process, base + 0x62C),
        "demo_loading_complete": read_memory(process, base + 0x630, 1)[0],
    }
    if return_address == DEFERRED_FILE_WRITE_CALLER:
        snapshot["file_write_argument_pointer"] = _read_u32(
            process, int(context.Esp) + 0xB4
        )
        snapshot["file_write_argument_size"] = _read_u32(
            process, int(context.Esp) + 0xB8
        )
        object_address = int(context.Ebx)
        snapshot["file_write_object_address"] = object_address
        try:
            object_length = _read_u32(process, object_address + 0x14)
            object_capacity = _read_u32(process, object_address + 0x18)
            if object_length > 1_048_576 or object_capacity > 1_048_576:
                raise ValueError("file-write object string is implausibly large")
            object_data_address = (
                _read_u32(process, object_address + 0x4)
                if object_capacity >= 16
                else object_address + 0x4
            )
            object_data = read_memory(
                process, object_data_address, object_length
            )
            snapshot["file_write_object_length"] = object_length
            snapshot["file_write_object_capacity"] = object_capacity
            snapshot["file_write_object_data_address"] = object_data_address
            snapshot["file_write_object_sha256"] = hashlib.sha256(
                object_data
            ).hexdigest()
            snapshot["file_write_object_prefix_hex"] = object_data[:64].hex()
            snapshot["file_write_object_full_hex"] = object_data[:512].hex()
            snapshot["file_write_object_full_truncated"] = (
                len(object_data) > 512
            )
            object_text = object_data.decode("latin-1")
            snapshot["file_write_font_cache_member"] = (
                _normalize_font_cache_member(object_text) or ""
            )
        except (OSError, ValueError):
            snapshot["file_write_object_length"] = -1
            snapshot["file_write_object_capacity"] = -1
            snapshot["file_write_object_data_address"] = 0
            snapshot["file_write_object_sha256"] = ""
            snapshot["file_write_object_prefix_hex"] = ""
            snapshot["file_write_object_full_hex"] = ""
            snapshot["file_write_object_full_truncated"] = False
            snapshot["file_write_font_cache_member"] = ""
    return snapshot


def _stack_code_candidates(
    process: int,
    stack_pointer: int,
    *,
    size: int = 256,
) -> tuple[tuple[int, int], ...]:
    words = struct.unpack(f"<{size // 4}I", read_memory(process, stack_pointer, size))
    return tuple(
        (index * 4, value)
        for index, value in enumerate(words)
        if 0x00401000 <= value < 0x0094AA9C
    )


def _startup_hidden_draw_stack_code(
    process: int,
    stack_pointer: int,
) -> list[dict[str, int | str]]:
    """Format bounded executable stack words for a hidden-draw receipt."""

    return [
        {
            "offset": offset,
            "address": value,
            "address_hex": f"0x{value:08x}",
        }
        for offset, value in _stack_code_candidates(
            process,
            stack_pointer,
            size=256,
        )
    ]


def trace_demo_commands(
    *,
    pid: int,
    executable: Path,
    address: int,
    maximum_hits: int,
    timeout: float,
    offline_rows: list[dict[str, object]] | None = None,
    broker_service_blocks: bool = False,
    service_wait_timeout: float = 5.0,
    quiet_nonservice: bool = False,
    stop_after_update: int | None = None,
    stop_after_command_order: int | None = None,
    close_after_terminal_command: bool = False,
    progress_every_updates: int | None = None,
    board_seed_address: int | None = None,
    board_seed_override: int | None = None,
    board_seed_skip_count: int = 0,
    global_rng_seed_override: int | None = None,
    thread_crt_rng_seed_override: int | None = None,
    board_seed_observations: list[dict[str, Any]] | None = None,
    source_bound_board_global_state: GlobalMTRandRestoreState | None = None,
    source_bound_board_thread_crt_state: ThreadCrtRestoreState | None = None,
    source_bound_board_observations: list[dict[str, Any]] | None = None,
    allow_source_bound_board_global_correction: bool = False,
    source_bound_board_precall_global_restore: bool = False,
    stop_after_board_seed: bool = False,
    allow_pre_stream_commands: bool = False,
    trace_command_entries: bool = True,
    startup_priority_bias: StartupPriorityBiasRuntime | None = None,
    global_mtrand_restore_state: GlobalMTRandRestoreState | None = None,
    global_mtrand_expected_before_state: (
        GlobalMTRandRestoreState | None
    ) = None,
    global_mtrand_restore_command_order: int | None = None,
    global_mtrand_restore_observations: (
        list[dict[str, Any]] | None
    ) = None,
    global_mtrand_allow_dynamic_before: bool = False,
    qrand_restore_state: QRandRestoreState | None = None,
    qrand_expected_before_state: QRandRestoreState | None = None,
    qrand_restore_command_order: int | None = None,
    qrand_restore_observations: list[dict[str, Any]] | None = None,
    thread_crt_restore_state: ThreadCrtRestoreState | None = None,
    thread_crt_expected_before_state: ThreadCrtRestoreState | None = None,
    thread_crt_restore_command_order: int | None = None,
    thread_crt_restore_observations: (
        list[dict[str, Any]] | None
    ) = None,
    thread_crt_allow_dynamic_before: bool = False,
    suspend_main_thread_on_stop: bool = False,
    allow_orphan_prepared_service_block: bool = False,
    allow_post_blackout_overdue_idle_reentry: bool = False,
    allow_initial_file_write_order_rebase: bool = False,
    command_order_rebase_after_update: int | None = None,
    blackout_expected_command_order_offset: int | None = None,
    blackout_expected_native_timeline_offset: int | None = None,
    blackout_allowed_offset_pairs: tuple[tuple[int, int], ...] = (),
    initial_command_order_offset: int = 0,
    initial_command_order_rebase_rows: tuple[int, ...] = (),
    diagnostic_successful_file_write_padding_rows: tuple[int, ...] = (),
    initial_service_consumed_file_write_debt_rows: tuple[int, ...] = (),
    attach_stabilization_after_update: int | None = None,
    attach_stabilization_main_thread_id: int | None = None,
    pre_attach_file_write_debt_before_update: int | None = None,
    font_cache_manifest: Mapping[str, int] | None = None,
    font_cache_manifest_sha256: str | None = None,
    font_cache_manifest_main_pak_sha256: str | None = None,
    startup_handoff_main_thread_id: int | None = None,
    startup_handoff_launcher_suspend_previous_count: int | None = None,
    gameplay_mtrand_oracle: GameplayMTRandOracle | None = None,
    gameplay_mtrand_sync_observations: (
        list[dict[str, Any]] | None
    ) = None,
    gameplay_mtrand_sync_receipt: dict[str, Any] | None = None,
    initial_global_mtrand_oracle: (
        InitialGlobalMTRandCallOracle | None
    ) = None,
    initial_global_mtrand_observations: (
        list[dict[str, Any]] | None
    ) = None,
    startup_global_mtrand_observation_oracle: (
        StartupGlobalMTRandObservationOracle | None
    ) = None,
    startup_global_mtrand_observations: (
        list[dict[str, Any]] | None
    ) = None,
    startup_global_mtrand_observation_receipt: (
        dict[str, Any] | None
    ) = None,
    startup_global_mtrand_hidden_draw_start_after_source_order: (
        int | None
    ) = None,
    startup_global_mtrand_hidden_draw_stop_before_source_order: (
        int | None
    ) = None,
    startup_global_mtrand_hidden_draw_observations: (
        list[dict[str, Any]] | None
    ) = None,
) -> TraceResult:
    """Trace a bounded number of PrepareDemoCommand entries."""

    observed = process_image_path(pid)
    if observed is None or not same_windows_path(observed, executable):
        raise RuntimeError(
            f"PID {pid} path mismatch: observed={observed}, expected={executable}"
        )
    if (
        address <= 0
        or maximum_hits <= 0
        or timeout <= 0
        or service_wait_timeout <= 0
    ):
        raise ValueError("address, maximum_hits, and timeout must be positive")
    if broker_service_blocks and offline_rows is None:
        raise ValueError("service broker requires offline DMO rows")
    attach_stabilization_requested = any(
        value is not None
        for value in (
            attach_stabilization_after_update,
            attach_stabilization_main_thread_id,
        )
    )
    if attach_stabilization_requested and (
        attach_stabilization_after_update is None
        or attach_stabilization_after_update <= 0
        or attach_stabilization_main_thread_id is None
        or attach_stabilization_main_thread_id <= 0
        or offline_rows is None
        or not trace_command_entries
    ):
        raise ValueError(
            "attach stabilization requires offline rows, command tracing, "
            "a positive late-attach update, and the original main thread"
        )
    if allow_post_blackout_overdue_idle_reentry and (
        not broker_service_blocks
        or not allow_orphan_prepared_service_block
        or not attach_stabilization_requested
        or offline_rows is None
        or not trace_command_entries
    ):
        raise ValueError(
            "post-blackout overdue-idle re-entry requires stabilized "
            "post-blackout service-brokered command tracing"
        )
    startup_handoff_requested = any(
        value is not None
        for value in (
            startup_handoff_main_thread_id,
            startup_handoff_launcher_suspend_previous_count,
        )
    )
    if startup_handoff_requested and (
        startup_handoff_main_thread_id is None
        or startup_handoff_main_thread_id <= 0
        or startup_handoff_launcher_suspend_previous_count != 0
        or not trace_command_entries
    ):
        raise ValueError(
            "startup trace handoff requires one launcher-suspended main "
            "thread and command tracing"
        )
    if startup_global_mtrand_observation_oracle is not None and (
        not startup_handoff_requested
        or startup_handoff_main_thread_id is None
        or not startup_global_mtrand_observation_oracle.entries
        or startup_global_mtrand_observations is None
        or startup_global_mtrand_observation_receipt is None
    ):
        raise ValueError(
            "startup global MTRand observation requires a non-empty oracle, "
            "startup handoff, observations, and a receipt"
        )
    if startup_global_mtrand_observation_oracle is None and (
        startup_global_mtrand_observations is not None
        or startup_global_mtrand_observation_receipt is not None
    ):
        raise ValueError(
            "startup global MTRand observation evidence requires an oracle"
        )
    startup_hidden_draw_values = (
        startup_global_mtrand_hidden_draw_start_after_source_order,
        startup_global_mtrand_hidden_draw_stop_before_source_order,
    )
    startup_hidden_draw_requested = any(
        value is not None for value in startup_hidden_draw_values
    )
    if startup_hidden_draw_requested and (
        any(value is None for value in startup_hidden_draw_values)
        or startup_global_mtrand_observation_oracle is None
        or startup_global_mtrand_hidden_draw_observations is None
    ):
        raise ValueError(
            "startup hidden-draw observation requires two source-order "
            "boundaries, a startup oracle, and observations"
        )
    if (
        not startup_hidden_draw_requested
        and startup_global_mtrand_hidden_draw_observations is not None
    ):
        raise ValueError(
            "startup hidden-draw observations require source boundaries"
        )
    startup_hidden_draw_start_entry_order: int | None = None
    startup_hidden_draw_stop_entry_order: int | None = None
    startup_hidden_draw_expected_count: int | None = None
    if startup_hidden_draw_requested:
        assert startup_global_mtrand_observation_oracle is not None
        assert (
            startup_global_mtrand_hidden_draw_start_after_source_order
            is not None
        )
        assert (
            startup_global_mtrand_hidden_draw_stop_before_source_order
            is not None
        )
        source_order_to_entry_order = {
            entry.source_order: order
            for order, entry in enumerate(
                startup_global_mtrand_observation_oracle.entries
            )
        }
        startup_hidden_draw_start_entry_order = (
            source_order_to_entry_order.get(
                startup_global_mtrand_hidden_draw_start_after_source_order
            )
        )
        startup_hidden_draw_stop_entry_order = (
            source_order_to_entry_order.get(
                startup_global_mtrand_hidden_draw_stop_before_source_order
            )
        )
        if (
            startup_hidden_draw_start_entry_order is None
            or startup_hidden_draw_stop_entry_order is None
            or startup_hidden_draw_stop_entry_order
            != startup_hidden_draw_start_entry_order + 1
        ):
            raise ValueError(
                "startup hidden-draw boundaries must be consecutive oracle "
                "entries"
            )
        start_entry = startup_global_mtrand_observation_oracle.entries[
            startup_hidden_draw_start_entry_order
        ]
        stop_entry = startup_global_mtrand_observation_oracle.entries[
            startup_hidden_draw_stop_entry_order
        ]
        startup_hidden_draw_expected_count = (
            stop_entry.pre_draw_count - start_entry.post_draw_count
        )
        if startup_hidden_draw_expected_count < 0:
            raise ValueError(
                "startup hidden-draw source interval is invalid"
            )
    if startup_global_mtrand_observation_oracle is not None and (
        initial_global_mtrand_oracle is not None
        or gameplay_mtrand_oracle is not None
    ):
        raise ValueError(
            "startup global MTRand observation cannot share debugger "
            "breakpoints with startup state control or gameplay sync"
        )
    if (
        startup_global_mtrand_observation_oracle is not None
        and stop_after_update is not None
        and startup_global_mtrand_observation_oracle.entries[-1]
        .framework_update
        >= stop_after_update
    ):
        raise ValueError(
            "startup global MTRand observation oracle must complete before "
            "the trace stop"
        )
    if gameplay_mtrand_oracle is not None and (
        not startup_handoff_requested
        or startup_handoff_main_thread_id is None
        or not gameplay_mtrand_oracle.entries
        or gameplay_mtrand_sync_observations is None
        or gameplay_mtrand_sync_receipt is None
    ):
        raise ValueError(
            "gameplay MTRand synchronization requires a non-empty oracle, "
            "startup handoff, observations, and a receipt"
        )
    if gameplay_mtrand_oracle is None and (
        gameplay_mtrand_sync_observations is not None
        or gameplay_mtrand_sync_receipt is not None
    ):
        raise ValueError(
            "gameplay MTRand synchronization evidence requires an oracle"
        )
    gameplay_mtrand_breakpoint_addresses = (
        tuple(
            dict.fromkeys(
                entry.caller for entry in gameplay_mtrand_oracle.entries
            )
        )
        if gameplay_mtrand_oracle is not None
        else ()
    )
    if len(gameplay_mtrand_breakpoint_addresses) > 4:
        raise ValueError(
            "gameplay MTRand oracle needs more than four hardware slots"
        )
    gameplay_mtrand_last_order_by_caller = (
        {
            entry.caller: entry.order
            for entry in gameplay_mtrand_oracle.entries
        }
        if gameplay_mtrand_oracle is not None
        else {}
    )
    if (
        gameplay_mtrand_oracle is not None
        and stop_after_update is not None
        and gameplay_mtrand_oracle.entries[-1].framework_update
        >= stop_after_update
    ):
        raise ValueError(
            "gameplay MTRand oracle must complete before the trace stop"
        )
    if (
        allow_initial_file_write_order_rebase
        != (command_order_rebase_after_update is not None)
    ):
        raise ValueError(
            "file-write order rebase requires one detached-after update"
        )
    if (
        allow_initial_file_write_order_rebase
        and (
            offline_rows is None
            or command_order_rebase_after_update is None
            or command_order_rebase_after_update < 0
            or stop_after_command_order is not None
            or global_mtrand_restore_state is not None
            or qrand_restore_state is not None
            or thread_crt_restore_state is not None
        )
    ):
        raise ValueError("invalid file-write command-order rebase")
    blackout_static_envelope_requested = any(
        value is not None
        for value in (
            blackout_expected_command_order_offset,
            blackout_expected_native_timeline_offset,
        )
    )
    blackout_envelope_set_requested = bool(
        blackout_allowed_offset_pairs
    )
    if (
        blackout_static_envelope_requested
        and blackout_envelope_set_requested
    ):
        raise ValueError(
            "single and finite-set blackout envelopes are mutually exclusive"
        )
    if blackout_static_envelope_requested and (
        not allow_initial_file_write_order_rebase
        or blackout_expected_command_order_offset is None
        or blackout_expected_command_order_offset <= 0
        or blackout_expected_native_timeline_offset is None
        or blackout_expected_native_timeline_offset < 0
    ):
        raise ValueError(
            "blackout offset envelope requires one positive command offset "
            "and one non-negative native timeline offset"
        )
    if blackout_envelope_set_requested and (
        not allow_initial_file_write_order_rebase
        or len(blackout_allowed_offset_pairs) < 2
        or tuple(sorted(set(blackout_allowed_offset_pairs)))
        != blackout_allowed_offset_pairs
        or not any(
            command_offset > 0
            for command_offset, _ in blackout_allowed_offset_pairs
        )
        or any(
            isinstance(command_offset, bool)
            or not isinstance(command_offset, int)
            or command_offset < 0
            or isinstance(timeline_offset, bool)
            or not isinstance(timeline_offset, int)
            or timeline_offset < 0
            for command_offset, timeline_offset
            in blackout_allowed_offset_pairs
        )
    ):
        raise ValueError(
            "blackout offset envelope set must contain canonical distinct "
            "non-negative pairs and at least one positive command offset"
        )
    blackout_envelope_requested = (
        blackout_static_envelope_requested
        or blackout_envelope_set_requested
    )
    if not trace_command_entries and (
        broker_service_blocks
        or stop_after_update is not None
        or stop_after_command_order is not None
        or close_after_terminal_command
        or allow_pre_stream_commands
        or global_mtrand_restore_state is not None
        or qrand_restore_state is not None
        or thread_crt_restore_state is not None
    ):
        raise ValueError(
            "board-only tracing cannot use command-entry options"
        )
    if (
        startup_priority_bias is not None
        and (
            not trace_command_entries
            or startup_priority_bias.process_id != pid
            or startup_priority_bias.requested_until_update <= 0
            or startup_priority_bias.restored_at_update is not None
        )
    ):
        raise ValueError("invalid startup priority bias trace lifecycle")
    if stop_after_command_order is not None and offline_rows is None:
        raise ValueError("command-order stop requires offline DMO rows")
    if close_after_terminal_command and (
        offline_rows is None
        or not trace_command_entries
        or stop_after_update is not None
        or stop_after_command_order is not None
        or suspend_main_thread_on_stop
    ):
        raise ValueError(
            "terminal close requires offline command tracing without another "
            "stop or handoff target"
        )
    if (
        initial_command_order_offset < 0
        or initial_command_order_offset
        != len(initial_command_order_rebase_rows)
        or any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in initial_command_order_rebase_rows
        )
    ):
        raise ValueError("initial command-order rebase is invalid")
    if suspend_main_thread_on_stop and (
        stop_after_update is None
        and stop_after_command_order is None
        and not stop_after_board_seed
    ):
        raise ValueError(
            "main-thread handoff suspension requires a bounded stop"
        )
    global_mtrand_values = (
        global_mtrand_restore_state,
        global_mtrand_expected_before_state,
        global_mtrand_restore_command_order,
    )
    if any(value is not None for value in global_mtrand_values) and any(
        value is None for value in global_mtrand_values
    ):
        raise ValueError(
            "global MTRand restore arguments must be used together"
        )
    if (
        global_mtrand_restore_state is not None
        and offline_rows is None
    ):
        raise ValueError("global MTRand restore requires offline DMO rows")
    if (
        global_mtrand_allow_dynamic_before
        and global_mtrand_restore_state is None
    ):
        raise ValueError(
            "dynamic global MTRand baseline requires a restore"
        )
    qrand_values = (
        qrand_restore_state,
        qrand_expected_before_state,
        qrand_restore_command_order,
    )
    if any(value is not None for value in qrand_values) and any(
        value is None for value in qrand_values
    ):
        raise ValueError("QRand restore arguments must be used together")
    if qrand_restore_state is not None and offline_rows is None:
        raise ValueError("QRand restore requires offline DMO rows")
    thread_crt_values = (
        thread_crt_restore_state,
        thread_crt_expected_before_state,
        thread_crt_restore_command_order,
    )
    if any(value is not None for value in thread_crt_values) and any(
        value is None for value in thread_crt_values
    ):
        raise ValueError("thread CRT restore arguments must be used together")
    if thread_crt_restore_state is not None and offline_rows is None:
        raise ValueError("thread CRT restore requires offline DMO rows")
    if (
        thread_crt_allow_dynamic_before
        and thread_crt_restore_state is None
    ):
        raise ValueError(
            "dynamic thread CRT baseline requires a thread CRT restore"
        )
    if board_seed_address is not None and board_seed_address <= 0:
        raise ValueError("board seed address must be positive")
    if board_seed_override is not None and not (
        0 <= board_seed_override <= 0xFFFFFFFF
    ):
        raise ValueError("board seed override must fit uint32")
    if board_seed_override is not None and board_seed_address is None:
        raise ValueError("board seed override requires a breakpoint address")
    if board_seed_skip_count < 0:
        raise ValueError("board seed skip count must be non-negative")
    if board_seed_skip_count and board_seed_address is None:
        raise ValueError("board seed skips require a breakpoint address")
    if global_rng_seed_override is not None and not (
        0 <= global_rng_seed_override <= 0xFFFFFFFF
    ):
        raise ValueError("global RNG seed override must fit uint32")
    if global_rng_seed_override is not None and board_seed_address is None:
        raise ValueError(
            "global RNG seed override requires a board breakpoint"
        )
    if thread_crt_rng_seed_override is not None and not (
        0 <= thread_crt_rng_seed_override <= 0xFFFFFFFF
    ):
        raise ValueError("thread CRT RNG seed override must fit uint32")
    if (
        thread_crt_rng_seed_override is not None
        and board_seed_address is None
    ):
        raise ValueError(
            "thread CRT RNG seed override requires a board breakpoint"
        )
    source_bound_board_values = (
        source_bound_board_global_state,
        source_bound_board_thread_crt_state,
    )
    if any(value is not None for value in source_bound_board_values) and any(
        value is None for value in source_bound_board_values
    ):
        raise ValueError(
            "source-bound board anchor states must be supplied together"
        )
    if (
        allow_source_bound_board_global_correction
        and source_bound_board_global_state is None
    ):
        raise ValueError(
            "source-bound board global correction requires source states"
        )
    if source_bound_board_precall_global_restore and (
        source_bound_board_global_state is None
        or allow_source_bound_board_global_correction
        or global_mtrand_restore_state is not None
    ):
        raise ValueError(
            "source-bound Board precall restore requires source states and "
            "cannot use another global correction or restore"
        )
    source_bound_board_expected_output: int | None = None
    source_bound_board_expected_post: bytes | None = None
    source_bound_board_expected_post_index: int | None = None
    source_bound_board_expected_post_sha256: str | None = None
    if source_bound_board_global_state is not None:
        assert source_bound_board_thread_crt_state is not None
        _validate_source_bound_board_anchor_states(
            source_bound_board_global_state,
            source_bound_board_thread_crt_state,
        )
        (
            source_output,
            source_bound_board_expected_post,
            source_bound_board_expected_post_index,
            source_bound_board_expected_post_sha256,
        ) = _global_mtrand_call_transition(source_bound_board_global_state.payload)
        source_bound_board_expected_output = source_output
        if (
            board_seed_address != DEFAULT_BOARD_RESEED_CALL
            or board_seed_override != source_output
            or board_seed_skip_count != 0
            or global_rng_seed_override is not None
            or thread_crt_rng_seed_override is not None
            or not trace_command_entries
            or startup_handoff_main_thread_id is None
        ):
            raise ValueError(
                "source-bound board anchor requires the exact board reseed "
                "site and source output without competing RNG overrides"
            )
    if stop_after_board_seed and board_seed_address is None:
        raise ValueError("board seed stop requires a breakpoint address")
    # Timeline rebases below are verifier-local.  Never mutate the caller's
    # decoded DMO rows or the source artifact they represent.
    rows = [dict(row) for row in (offline_rows or [])]
    if (
        tuple(
            sorted(set(diagnostic_successful_file_write_padding_rows))
        )
        != diagnostic_successful_file_write_padding_rows
        or tuple(
            sorted(set(initial_service_consumed_file_write_debt_rows))
        )
        != initial_service_consumed_file_write_debt_rows
        or (
            diagnostic_successful_file_write_padding_rows
            and (
                offline_rows is None
                or not trace_command_entries
                or not broker_service_blocks
            )
        )
        or (
            initial_service_consumed_file_write_debt_rows
            and not diagnostic_successful_file_write_padding_rows
        )
        or set(diagnostic_successful_file_write_padding_rows)
        & set(initial_service_consumed_file_write_debt_rows)
        or set(initial_command_order_rebase_rows)
        & set(initial_service_consumed_file_write_debt_rows)
    ):
        raise ValueError("successful file-write padding registry is invalid")
    for registry_name, registry_rows in (
        (
            "padding",
            diagnostic_successful_file_write_padding_rows,
        ),
        (
            "service-consumed",
            initial_service_consumed_file_write_debt_rows,
        ),
    ):
        for row_index in registry_rows:
            if row_index < 0 or row_index >= len(rows):
                raise ValueError(
                    f"successful file-write {registry_name} row is out of range"
                )
            row = rows[row_index]
            payload = row.get("payload")
            if (
                row.get("kind") != "file_write"
                or bool(row["short_form"])
                or int(row["command_number"]) != 16
                or int(row["end"]) - int(row["start"]) != 11
                or payload != {"success": True}
            ):
                raise ValueError(
                    f"successful file-write {registry_name} row is not exact"
                )

    def is_exact_successful_file_write_debt_row(row_index: int) -> bool:
        if row_index < 0 or row_index >= len(rows):
            return False
        row = rows[row_index]
        return (
            row.get("kind") == "file_write"
            and not bool(row["short_form"])
            and int(row["command_number"]) == 16
            and int(row["end"]) - int(row["start"]) == 11
            and row.get("payload") == {"success": True}
        )
    if close_after_terminal_command:
        terminal_row = rows[-1] if rows else None
        if (
            terminal_row is None
            or terminal_row.get("kind") != "idle"
            or bool(terminal_row.get("short_form"))
            or int(terminal_row["end"]) - int(terminal_row["start"]) != 10
        ):
            raise ValueError(
                "terminal close requires one payload-free ten-bit final idle"
            )
    if pre_attach_file_write_debt_before_update is not None and (
        pre_attach_file_write_debt_before_update <= 0
        or offline_rows is None
        or not trace_command_entries
    ):
        raise ValueError(
            "pre-attach file-write debt requires offline command tracing and "
            "a positive late-attach update"
        )
    pre_attach_file_write_debt_rows = (
        tuple(
            index
            for index, row in enumerate(rows)
            if (
                int(row["update"])
                < pre_attach_file_write_debt_before_update
                and not bool(row["short_form"])
                and int(row["command_number"]) == 16
                and isinstance(row.get("payload"), dict)
                and isinstance(row["payload"].get("success"), bool)
            )
        )
        if pre_attach_file_write_debt_before_update is not None
        else ()
    )
    if (
        pre_attach_file_write_debt_before_update is not None
        and not pre_attach_file_write_debt_rows
    ):
        raise ValueError("pre-attach file-write debt found no exact rows")
    font_cache_manifest_values = (
        font_cache_manifest,
        font_cache_manifest_sha256,
        font_cache_manifest_main_pak_sha256,
    )
    font_cache_manifest_requested = any(
        value is not None for value in font_cache_manifest_values
    )
    if font_cache_manifest_requested and (
        any(value is None for value in font_cache_manifest_values)
        or offline_rows is None
        or (
            pre_attach_file_write_debt_before_update is None
            and not startup_handoff_requested
        )
        or not trace_command_entries
    ):
        raise ValueError(
            "font-cache manifest completion requires complete manifest "
            "evidence, offline command tracing, and either pre-attach debt "
            "or a gapless startup handoff"
        )
    startup_file_write_rows = (
        tuple(
            index
            for index, row in enumerate(rows)
            if (
                index
                < next(
                    (
                        loading_index
                        for loading_index, loading_row in enumerate(rows)
                        if loading_row.get("kind") == "loading_complete"
                        and int(loading_row["command_number"]) == 9
                    ),
                    len(rows),
                )
                and not bool(row["short_form"])
                and int(row["command_number"]) == 16
                and isinstance(row.get("payload"), dict)
                and isinstance(row["payload"].get("success"), bool)
            )
        )
        if font_cache_manifest_requested
        else ()
    )
    if font_cache_manifest_requested:
        assert font_cache_manifest is not None
        if (
            len(font_cache_manifest)
            != FONT_CACHE_MANIFEST_EXPECTED_ENTRIES
            or len(startup_file_write_rows)
            != FONT_CACHE_MANIFEST_EXPECTED_ENTRIES
            or any(
                bool(rows[row_index]["payload"]["success"])
                for row_index in startup_file_write_rows
            )
        ):
            raise ValueError(
                "font-cache manifest and startup DMO do not form the exact "
                "35-operation uniformly false receipt"
            )
    if (
        stop_after_command_order is not None
        and stop_after_command_order >= len(rows)
    ):
        raise ValueError(
            "stop-after command order is outside the offline DMO"
        )
    if (
        global_mtrand_restore_command_order is not None
        and (
            global_mtrand_restore_command_order < 0
            or global_mtrand_restore_command_order >= len(rows)
        )
    ):
        raise ValueError(
            "global MTRand restore command order is outside the DMO"
        )
    if (
        qrand_restore_command_order is not None
        and (
            qrand_restore_command_order < 0
            or qrand_restore_command_order >= len(rows)
        )
    ):
        raise ValueError("QRand restore command order is outside the DMO")
    if (
        thread_crt_restore_command_order is not None
        and (
            thread_crt_restore_command_order < 0
            or thread_crt_restore_command_order >= len(rows)
        )
    ):
        raise ValueError(
            "thread CRT restore command order is outside the DMO"
        )
    rows_by_start = {
        int(row["start"]): (index, row)
        for index, row in enumerate(rows)
    }
    board_observations = (
        board_seed_observations
        if board_seed_observations is not None
        else []
    )
    source_bound_board_rows = (
        source_bound_board_observations
        if source_bound_board_observations is not None
        else []
    )
    global_mtrand_observations = (
        global_mtrand_restore_observations
        if global_mtrand_restore_observations is not None
        else []
    )
    qrand_observations = (
        qrand_restore_observations
        if qrand_restore_observations is not None
        else []
    )
    thread_crt_observations = (
        thread_crt_restore_observations
        if thread_crt_restore_observations is not None
        else []
    )

    attached = False
    handoff_main_thread_suspended = False
    handoff_main_thread_id: int | None = None
    handoff_previous_suspend_count: int | None = None
    preserve_handoff_suspend = False
    exited = False
    exit_code: int | None = None
    process = 0
    original: bytes | None = None
    breakpoint_armed = False
    board_original: bytes | None = None
    board_breakpoint_armed = False
    source_bound_precall_original: bytes | None = None
    source_bound_precall_breakpoint_armed = False
    source_bound_precall_observations: list[dict[str, Any]] = []
    source_bound_precall_suspended_threads: list[tuple[int, int]] = []
    initial_global_original: bytes | None = None
    initial_global_breakpoint_armed = False
    initial_global_dispatch_instruction: bytes | None = None
    initial_global_dispatch_target: int | None = None
    initial_global_return_original: bytes | None = None
    initial_global_return_breakpoint_armed = False
    initial_global_suspended_threads: list[tuple[int, int]] = []
    initial_global_atomic_window_active = False
    initial_global_rows = (
        initial_global_mtrand_observations
        if initial_global_mtrand_observations is not None
        else []
    )
    startup_hidden_draw_rows = (
        startup_global_mtrand_hidden_draw_observations
        if startup_global_mtrand_hidden_draw_observations is not None
        else []
    )
    startup_hidden_draw_watch_active = False
    startup_hidden_draw_start_boundary_observed = False
    startup_hidden_draw_stop_boundary_observed = False
    startup_global_mtrand_hardware_original: (
        tuple[int, int, int, int, int, int] | None
    ) = None
    startup_global_mtrand_hardware_armed = False
    startup_global_mtrand_hardware_restore_error: str | None = None
    gameplay_mtrand_hardware_original: (
        tuple[int, int, int, int, int, int] | None
    ) = None
    gameplay_mtrand_hardware_armed = False
    gameplay_mtrand_hardware_restore_error: str | None = None
    startup_handoff_tracer_resume_previous_count: int | None = None
    startup_handoff_resumed_after_breakpoints = False
    startup_handoff_resumed_perf_counter_ns: int | None = None
    stepping_thread: int | None = None
    stepping_kind: str | None = None
    stepping_suspended_threads: list[tuple[int, int]] = []
    stop_after_step = False
    terminal_close_after_step: dict[str, int | str | bool] | None = None
    terminal_close_deadline: float | None = None
    held_thread = 0
    pending_broker: dict[str, int] | None = None
    pending_service_continuation: dict[str, int] | None = None
    post_bypass_worker_yield_request: dict[str, int] | None = None
    post_bypass_return_breakpoint: dict[str, int] | None = None
    post_bypass_return_breakpoint_original: bytes | None = None
    pending_post_bypass_worker_yield: dict[str, int] | None = None
    post_bypass_worker_yield_thread = 0
    broker_failures: list[str] = []
    boundary_failures: list[str] = []
    brokered_blocks = 0
    broker_bypassed_blocks = 0
    startup_post_bypass_worker_yields: list[Mapping[str, Any]] = []
    bypassed_service_blocks: set[tuple[int, int, int]] = set()
    orphan_prepared_service_block_used = False
    access_denied_thread_skip_events: list[int] = []
    stopped_at_update = False
    stopped_at_command_order = False
    last_snapshot_update = -1
    first_command_order: int | None = None
    last_command_order: int | None = None
    command_order_offset = initial_command_order_offset
    command_order_rebase_rows = initial_command_order_rebase_rows
    blackout_file_write_rebase_candidate_rows: tuple[int, ...] = ()
    blackout_native_timeline_offset = 0
    service_consumed_file_write_debt_rows = (
        initial_service_consumed_file_write_debt_rows
    )
    service_consumed_file_write_surplus_rows: tuple[int, ...] = ()
    native_timeline_offset = 0
    initial_offline_boundary_seen = False
    last_progress_bucket = -1
    offline_stream_started = False
    hits = 0
    board_seed_hits_seen = 0
    attach_stabilization_pending = attach_stabilization_requested
    attach_stabilization_verified = False
    attach_stabilization_skipped_hits = 0
    attach_stabilization_verified_update: int | None = None
    attach_stabilization_verified_command_order: int | None = None
    attach_stabilization_observations: list[
        Mapping[str, int | str | bool]
    ] = []
    service_exit_timeline_commits: list[Mapping[str, int]] = []
    service_continuation_verifications: list[
        Mapping[str, int | str]
    ] = []
    service_exit_overdue_idle_reentries: list[
        Mapping[str, Any]
    ] = []
    preloading_failed_file_write_tail_short_header_recoveries: list[
        Mapping[str, Any]
    ] = []
    main_file_read_corridor_native_replays: list[
        Mapping[str, int | str]
    ] = []
    deferred_file_write_discharges: list[
        Mapping[str, int | bool | str]
    ] = []
    successful_file_write_debt_barriers: list[Mapping[str, Any]] = []
    terminal_registry_write_eof_exits: list[Mapping[str, Any]] = []
    service_exit_timeline_suffix_rebases: list[Mapping[str, Any]] = []
    terminal_file_write_payload_handoffs: list[
        Mapping[str, int | bool | str]
    ] = []
    service_file_write_header_claims: list[
        Mapping[str, int | bool | str]
    ] = []
    direct_font_cache_file_write_observations: list[
        Mapping[str, int | bool | str]
    ] = []
    font_cache_manifest_completion_discharges: list[
        Mapping[str, int | bool | str]
    ] = []
    terminal_close_requests: list[Mapping[str, int | str | bool]] = []
    debug_exception_observations: list[Mapping[str, Any]] = []
    startup_global_mtrand_rows = (
        startup_global_mtrand_observations
        if startup_global_mtrand_observations is not None
        else []
    )
    if startup_global_mtrand_observation_oracle is not None:
        assert startup_global_mtrand_observation_receipt is not None
        startup_global_mtrand_observation_receipt.update(
            {
                "status": "RUNNING",
                "failure": None,
                "oracle_complete": False,
                "mechanism": (
                    "in_trace_main_thread_global_wrapper_hardware_"
                    "breakpoint_read_only"
                ),
                "process_id": pid,
                "main_thread_id": startup_handoff_main_thread_id,
                "wrapper_address": (
                    startup_global_mtrand_observation_oracle.wrapper_address
                ),
                "wrapper_address_hex": (
                    "0x"
                    f"{startup_global_mtrand_observation_oracle.wrapper_address:08x}"
                ),
                "hardware_breakpoint_count": 1,
                "oracle": {
                    "source_path": str(
                        startup_global_mtrand_observation_oracle.source_path
                    ),
                    "source_sha256": (
                        startup_global_mtrand_observation_oracle.source_sha256
                    ),
                    "source_process_id": (
                        startup_global_mtrand_observation_oracle
                        .source_process_id
                    ),
                    "source_main_thread_id": (
                        startup_global_mtrand_observation_oracle
                        .source_main_thread_id
                    ),
                    "runtime_executable_sha256": (
                        startup_global_mtrand_observation_oracle
                        .runtime_executable_sha256
                    ),
                    "seed": startup_global_mtrand_observation_oracle.seed,
                    "end_at_update": (
                        startup_global_mtrand_observation_oracle.end_at_update
                    ),
                    "maximum_draws": (
                        startup_global_mtrand_observation_oracle.maximum_draws
                    ),
                    "entry_count": len(
                        startup_global_mtrand_observation_oracle.entries
                    ),
                    "semantic_sha256": (
                        startup_global_mtrand_observation_oracle
                        .semantic_sha256
                    ),
                },
                "expected_hit_count": len(
                    startup_global_mtrand_observation_oracle.entries
                ),
                "hit_count": 0,
                "hardware_breakpoint_armed": False,
                "hardware_breakpoint_restored": False,
                "hardware_breakpoint_restore_error": None,
                "process_memory_writes": 0,
                "process_memory_mutation": False,
                "process_context_mutation": False,
                "process_state_mutation": False,
                "register_housekeeping_count": 0,
                "persistent_file_modified": False,
            }
        )
        if startup_hidden_draw_requested:
            assert startup_hidden_draw_start_entry_order is not None
            assert startup_hidden_draw_stop_entry_order is not None
            assert startup_hidden_draw_expected_count is not None
            start_entry = (
                startup_global_mtrand_observation_oracle.entries[
                    startup_hidden_draw_start_entry_order
                ]
            )
            stop_entry = (
                startup_global_mtrand_observation_oracle.entries[
                    startup_hidden_draw_stop_entry_order
                ]
            )
            startup_global_mtrand_observation_receipt[
                "hidden_draw_interval"
            ] = {
                "status": "RUNNING",
                "failure": None,
                "index_address": (
                    G_FRAMEWORK_MTRAND_ADDRESS
                    + MTRAND_STATE_WORDS * 4
                ),
                "index_address_hex": (
                    "0x"
                    f"{G_FRAMEWORK_MTRAND_ADDRESS + MTRAND_STATE_WORDS * 4:08x}"
                ),
                "hardware_breakpoint_slot": 1,
                "hardware_breakpoint_access": "write",
                "hardware_breakpoint_length_bytes": 4,
                "start_after_source_order": start_entry.source_order,
                "start_entry_order": startup_hidden_draw_start_entry_order,
                "start_framework_update": start_entry.framework_update,
                "start_post_draw_count": start_entry.post_draw_count,
                "start_post_state_sha256": start_entry.post_state_sha256,
                "stop_before_source_order": stop_entry.source_order,
                "stop_entry_order": startup_hidden_draw_stop_entry_order,
                "stop_framework_update": stop_entry.framework_update,
                "expected_stop_pre_draw_count": stop_entry.pre_draw_count,
                "expected_stop_pre_state_sha256": (
                    stop_entry.pre_state_sha256
                ),
                "expected_hidden_draw_count": (
                    startup_hidden_draw_expected_count
                ),
                "expected_total_index_write_count": (
                    startup_hidden_draw_expected_count + 1
                ),
                "start_boundary_observed": False,
                "stop_boundary_observed": False,
                "watch_armed": False,
                "watch_disarmed": False,
                "index_write_count": 0,
                "process_memory_writes": 0,
                "process_memory_mutation": False,
                "process_context_mutation": True,
                "persistent_file_modified": False,
            }
    gameplay_mtrand_rows = (
        gameplay_mtrand_sync_observations
        if gameplay_mtrand_sync_observations is not None
        else []
    )
    if gameplay_mtrand_oracle is not None:
        assert gameplay_mtrand_sync_receipt is not None
        gameplay_mtrand_sync_receipt.update(
            {
                "status": "RUNNING",
                "failure": None,
                "oracle_complete": False,
                "mechanism": (
                    "in_trace_main_thread_gameplay_return_hardware_"
                    "breakpoint"
                ),
                "process_id": pid,
                "main_thread_id": startup_handoff_main_thread_id,
                "addresses": list(gameplay_mtrand_breakpoint_addresses),
                "address_hex": [
                    f"0x{address:08x}"
                    for address in gameplay_mtrand_breakpoint_addresses
                ],
                "hardware_breakpoint_count": len(
                    gameplay_mtrand_breakpoint_addresses
                ),
                "oracle": {
                    "source_path": str(
                        gameplay_mtrand_oracle.source_path
                    ),
                    "source_sha256": gameplay_mtrand_oracle.source_sha256,
                    "source_process_id": (
                        gameplay_mtrand_oracle.source_process_id
                    ),
                    "seed": gameplay_mtrand_oracle.seed,
                    "entry_count": len(
                        gameplay_mtrand_oracle.entries
                    ),
                    "first_update": (
                        gameplay_mtrand_oracle.entries[0]
                        .framework_update
                    ),
                    "last_update": (
                        gameplay_mtrand_oracle.entries[-1]
                        .framework_update
                    ),
                    "semantic_sha256": (
                        gameplay_mtrand_oracle.semantic_sha256
                    ),
                },
                "expected_hit_count": len(
                    gameplay_mtrand_oracle.entries
                ),
                "hit_count": 0,
                "state_correction_count": 0,
                "bytes_written_total": 0,
                "hardware_breakpoint_armed": False,
                "hardware_breakpoint_restored": False,
                "hardware_breakpoint_restore_error": None,
                "process_memory_mutation": False,
                "process_context_mutation": False,
                "process_state_mutation": False,
                "register_mutation_count": 0,
                "persistent_file_modified": False,
            }
        )

    if not kernel32.DebugActiveProcess(pid):
        raise ctypes.WinError(ctypes.get_last_error())
    attached = True
    try:
        if not kernel32.DebugSetProcessKillOnExit(False):
            raise ctypes.WinError(ctypes.get_last_error())
        process = kernel32.OpenProcess(PROCESS_ACCESS, False, pid)
        if not process:
            raise ctypes.WinError(ctypes.get_last_error())
        if trace_command_entries:
            original = read_memory(process, address, 1)
            if original == b"\xCC":
                raise RuntimeError("target address already contains INT3")
            write_memory(process, address, b"\xCC")
            breakpoint_armed = True
            kernel32.FlushInstructionCache(
                process,
                ctypes.c_void_p(address),
                1,
            )
            print(
                f"armed pid={pid} address=0x{address:08X} "
                f"original={original.hex()}",
                flush=True,
            )
        if initial_global_mtrand_oracle is not None:
            if startup_handoff_main_thread_id is None:
                raise ValueError(
                    "initial global MTRand control requires the startup "
                    "main-thread handoff"
                )
            initial_wrapper_address = (
                initial_global_mtrand_oracle.wrapper_address
            )
            initial_return_address = initial_global_mtrand_oracle.caller
            if (
                initial_wrapper_address in {
                    address,
                    board_seed_address,
                    SOURCE_BOUND_GLOBAL_PRECALL_ADDRESS,
                }
                or initial_return_address in {
                    address,
                    board_seed_address,
                    SOURCE_BOUND_GLOBAL_PRECALL_ADDRESS,
                    initial_wrapper_address,
                }
            ):
                raise ValueError(
                    "initial global MTRand wrapper and return breakpoints "
                    "must be unique"
                )
            initial_global_dispatch_instruction = read_memory(
                process,
                initial_global_mtrand_oracle.call_address,
                5,
            )
            if initial_global_dispatch_instruction[:1] != b"\xE8":
                raise RuntimeError(
                    "initial global MTRand source dispatch is not a near "
                    "CALL"
                )
            initial_global_dispatch_target = (
                _decode_near_relative_branch_target(
                    initial_global_mtrand_oracle.call_address,
                    initial_global_dispatch_instruction,
                    opcode=0xE8,
                )
            )
            initial_global_original = read_memory(
                process,
                initial_wrapper_address,
                5,
            )
            expected_wrapper_prefix = b"\xBA" + struct.pack(
                "<I",
                G_FRAMEWORK_MTRAND_ADDRESS,
            )
            if initial_global_original != expected_wrapper_prefix:
                raise RuntimeError(
                    "initial global MTRand wrapper prefix mismatch"
                )
            write_memory(process, initial_wrapper_address, b"\xCC")
            initial_global_breakpoint_armed = True
            kernel32.FlushInstructionCache(
                process,
                ctypes.c_void_p(initial_wrapper_address),
                1,
            )
            print(
                "armed_initial_global_mtrand "
                f"pid={pid} wrapper=0x{initial_wrapper_address:08X} "
                "source_call=0x"
                f"{initial_global_mtrand_oracle.call_address:08X} "
                f"dispatch_target=0x{initial_global_dispatch_target:08X} "
                f"source={initial_global_mtrand_oracle.source_sha256}",
                flush=True,
            )
        if board_seed_address is not None:
            if trace_command_entries and board_seed_address == address:
                raise ValueError(
                    "board seed and command breakpoints must differ"
                )
            board_original = read_memory(process, board_seed_address, 1)
            if board_original != b"\xE8":
                raise RuntimeError(
                    "board seed target is not the expected CALL instruction"
                )
            write_memory(process, board_seed_address, b"\xCC")
            board_breakpoint_armed = True
            kernel32.FlushInstructionCache(
                process,
                ctypes.c_void_p(board_seed_address),
                1,
            )
            print(
                f"armed_board_seed pid={pid} "
                f"address=0x{board_seed_address:08X} "
                f"original={board_original.hex()}",
                flush=True,
            )
        if source_bound_board_precall_global_restore:
            if (
                SOURCE_BOUND_GLOBAL_PRECALL_ADDRESS == address
                or SOURCE_BOUND_GLOBAL_PRECALL_ADDRESS
                == board_seed_address
            ):
                raise ValueError(
                    "source-bound global precall breakpoint must be unique"
                )
            source_bound_precall_original = read_memory(
                process,
                SOURCE_BOUND_GLOBAL_PRECALL_ADDRESS,
                len(SOURCE_BOUND_GLOBAL_PRECALL_INSTRUCTION),
            )
            if (
                source_bound_precall_original
                != SOURCE_BOUND_GLOBAL_PRECALL_INSTRUCTION
            ):
                raise RuntimeError(
                    "source-bound global precall instruction mismatch"
                )
            write_memory(
                process,
                SOURCE_BOUND_GLOBAL_PRECALL_ADDRESS,
                b"\xCC",
            )
            source_bound_precall_breakpoint_armed = True
            kernel32.FlushInstructionCache(
                process,
                ctypes.c_void_p(SOURCE_BOUND_GLOBAL_PRECALL_ADDRESS),
                1,
            )
            print(
                "armed_source_bound_global_precall "
                f"pid={pid} address=0x"
                f"{SOURCE_BOUND_GLOBAL_PRECALL_ADDRESS:08X} "
                f"instruction={source_bound_precall_original.hex()}",
                flush=True,
            )
        if startup_global_mtrand_observation_oracle is not None:
            assert startup_handoff_main_thread_id is not None
            assert startup_global_mtrand_observation_receipt is not None
            startup_wrapper_address = (
                startup_global_mtrand_observation_oracle.wrapper_address
            )
            observed_wrapper_prefix = read_memory(
                process,
                startup_wrapper_address,
                5,
            )
            expected_wrapper_prefix = b"\xBA" + struct.pack(
                "<I",
                G_FRAMEWORK_MTRAND_ADDRESS,
            )
            if observed_wrapper_prefix != expected_wrapper_prefix:
                raise RuntimeError(
                    "startup global MTRand observation wrapper prefix "
                    "mismatch"
                )
            startup_global_mtrand_hardware_original = (
                _arm_gameplay_mtrand_breakpoints(
                    main_thread_id=startup_handoff_main_thread_id,
                    addresses=(startup_wrapper_address,),
                )
            )
            startup_global_mtrand_hardware_armed = True
            if (
                startup_hidden_draw_requested
                and startup_global_mtrand_hardware_original[5]
                & (0x3 << 2)
            ):
                raise RuntimeError(
                    "startup hidden-draw hardware breakpoint slot is "
                    "occupied"
                )
            startup_global_mtrand_observation_receipt[
                "hardware_breakpoint_armed"
            ] = True
            if startup_hidden_draw_requested:
                startup_global_mtrand_observation_receipt[
                    "hardware_breakpoint_count"
                ] = 2
            startup_global_mtrand_observation_receipt[
                "process_context_mutation"
            ] = True
            startup_global_mtrand_observation_receipt[
                "process_state_mutation"
            ] = True
            print(
                "armed_startup_global_mtrand_observation "
                f"tid={startup_handoff_main_thread_id} "
                f"wrapper=0x{startup_wrapper_address:08X} "
                "expected_hits="
                f"{len(startup_global_mtrand_observation_oracle.entries)}",
                flush=True,
            )
        if gameplay_mtrand_oracle is not None:
            assert startup_handoff_main_thread_id is not None
            assert gameplay_mtrand_sync_receipt is not None
            gameplay_mtrand_hardware_original = (
                _arm_gameplay_mtrand_breakpoints(
                    main_thread_id=startup_handoff_main_thread_id,
                    addresses=gameplay_mtrand_breakpoint_addresses,
                )
            )
            gameplay_mtrand_hardware_armed = True
            gameplay_mtrand_sync_receipt[
                "hardware_breakpoint_armed"
            ] = True
            print(
                "armed_gameplay_mtrand_sync "
                f"tid={startup_handoff_main_thread_id} "
                "addresses="
                + ",".join(
                    f"0x{address:08X}"
                    for address in gameplay_mtrand_breakpoint_addresses
                )
                + " "
                "expected_hits="
                f"{len(gameplay_mtrand_oracle.entries)}",
                flush=True,
            )
        if startup_handoff_requested:
            assert startup_handoff_main_thread_id is not None
            startup_thread = kernel32.OpenThread(
                THREAD_ACCESS | THREAD_SUSPEND_RESUME,
                False,
                startup_handoff_main_thread_id,
            )
            if not startup_thread:
                raise ctypes.WinError(ctypes.get_last_error())
            try:
                startup_handoff_tracer_resume_previous_count = int(
                    kernel32.ResumeThread(startup_thread)
                )
            finally:
                kernel32.CloseHandle(startup_thread)
            if (
                startup_handoff_tracer_resume_previous_count
                == INVALID_SUSPEND_COUNT
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            startup_handoff_resumed_after_breakpoints = True
            startup_handoff_resumed_perf_counter_ns = time.perf_counter_ns()
            if startup_handoff_tracer_resume_previous_count not in (1, 2):
                raise RuntimeError(
                    "startup trace handoff expected the explicit launcher "
                    "suspend, optionally plus one temporary debugger-attach "
                    "suspend; observed count="
                    f"{startup_handoff_tracer_resume_previous_count}"
                )
            print(
                "startup_trace_handoff_resumed "
                f"tid={startup_handoff_main_thread_id} "
                "launcher_previous_count="
                f"{startup_handoff_launcher_suspend_previous_count} "
                "tracer_previous_count="
                f"{startup_handoff_tracer_resume_previous_count} "
                f"command_breakpoint_armed={breakpoint_armed} "
                f"board_breakpoint_armed={board_breakpoint_armed} "
                "initial_global_mtrand_breakpoint_armed="
                f"{initial_global_breakpoint_armed} "
                "resumed_perf_counter_ns="
                f"{startup_handoff_resumed_perf_counter_ns}",
                flush=True,
            )

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            event = DEBUG_EVENT()
            if not kernel32.WaitForDebugEvent(ctypes.byref(event), 100):
                error = ctypes.get_last_error()
                if error == ERROR_SEM_TIMEOUT:
                    if (
                        terminal_close_deadline is not None
                        and time.monotonic() >= terminal_close_deadline
                    ):
                        boundary_failures.append(
                            "normal terminal close did not exit within ten "
                            "seconds"
                        )
                        print(
                            "terminal_close_failure detail="
                            + boundary_failures[-1],
                            flush=True,
                        )
                        break
                    continue
                raise ctypes.WinError(error)

            status = DBG_CONTINUE
            should_stop = False
            if event.dwDebugEventCode == EXCEPTION_DEBUG_EVENT:
                exception = event.u.Exception
                code = int(exception.ExceptionRecord.ExceptionCode)
                exception_address = int(
                    exception.ExceptionRecord.ExceptionAddress or 0
                )
                if (
                    post_bypass_return_breakpoint is not None
                    and code
                    in (EXCEPTION_BREAKPOINT, STATUS_WX86_BREAKPOINT)
                    and exception_address
                    == post_bypass_return_breakpoint["return_address"]
                ):
                    return_breakpoint = post_bypass_return_breakpoint
                    return_original = (
                        post_bypass_return_breakpoint_original
                    )
                    if (
                        return_original is None
                        or int(event.dwThreadId)
                        != return_breakpoint["thread_id"]
                        or stepping_thread is not None
                        or stepping_kind is not None
                        or stepping_suspended_threads
                        or held_thread
                        or pending_broker is not None
                        or pending_post_bypass_worker_yield is not None
                        or post_bypass_worker_yield_thread
                        or original is None
                        or not breakpoint_armed
                        or pending_service_continuation is None
                    ):
                        raise RuntimeError(
                            "post-bypass return breakpoint state is invalid"
                        )
                    thread = kernel32.OpenThread(
                        THREAD_ACCESS | THREAD_SUSPEND_RESUME,
                        False,
                        event.dwThreadId,
                    )
                    if not thread:
                        raise ctypes.WinError(ctypes.get_last_error())
                    try:
                        context = WOW64_CONTEXT()
                        context.ContextFlags = CONTEXT_FULL
                        if not kernel32.Wow64GetThreadContext(
                            thread,
                            ctypes.byref(context),
                        ):
                            raise ctypes.WinError(
                                ctypes.get_last_error()
                            )
                        return_address = return_breakpoint[
                            "return_address"
                        ]
                        write_memory(
                            process,
                            return_address,
                            return_original,
                        )
                        kernel32.FlushInstructionCache(
                            process,
                            ctypes.c_void_p(return_address),
                            1,
                        )
                        write_memory(process, address, original)
                        breakpoint_armed = False
                        kernel32.FlushInstructionCache(
                            process,
                            ctypes.c_void_p(address),
                            1,
                        )
                        context.Eip = return_address
                        context.EFlags &= ~TRAP_FLAG
                        if not kernel32.Wow64SetThreadContext(
                            thread,
                            ctypes.byref(context),
                        ):
                            raise ctypes.WinError(
                                ctypes.get_last_error()
                            )
                        previous_suspend_count = int(
                            kernel32.SuspendThread(thread)
                        )
                        if (
                            previous_suspend_count
                            == INVALID_SUSPEND_COUNT
                        ):
                            raise ctypes.WinError(
                                ctypes.get_last_error()
                            )
                        if previous_suspend_count != 0:
                            kernel32.ResumeThread(thread)
                            raise RuntimeError(
                                "post-bypass return yield expected suspend "
                                "count 0, observed "
                                f"{previous_suspend_count}"
                            )
                        resume_handle = kernel32.OpenThread(
                            THREAD_ACCESS | THREAD_SUSPEND_RESUME,
                            False,
                            event.dwThreadId,
                        )
                        if not resume_handle:
                            kernel32.ResumeThread(thread)
                            raise ctypes.WinError(
                                ctypes.get_last_error()
                            )
                        post_bypass_worker_yield_thread = int(
                            resume_handle
                        )
                        armed_ns = time.perf_counter_ns()
                        pending_post_bypass_worker_yield = {
                            **return_breakpoint,
                            "return_breakpoint_hit_perf_counter_ns": (
                                armed_ns
                            ),
                            "command_breakpoint_disarmed_perf_counter_ns": (
                                armed_ns
                            ),
                            "probe_started_perf_counter_ns": armed_ns,
                            "temporary_breakpoint_memory_writes": 3,
                        }
                        post_bypass_return_breakpoint = None
                        post_bypass_return_breakpoint_original = None
                        print(
                            "startup_post_bypass_worker_yield_armed "
                            f"tid={event.dwThreadId} "
                            "return_address=0x"
                            f"{return_address:08X} "
                            "initial_read_bitpos="
                            f"{pending_post_bypass_worker_yield['initial_read_bit_position']} "
                            "corridor_end_bitpos="
                            f"{pending_post_bypass_worker_yield['corridor_end_bit_position']}",
                            flush=True,
                        )
                    finally:
                        kernel32.CloseHandle(thread)
                elif (
                    initial_global_mtrand_oracle is not None
                    and code
                    in (EXCEPTION_BREAKPOINT, STATUS_WX86_BREAKPOINT)
                    and exception_address
                    == initial_global_mtrand_oracle.caller
                    and initial_global_return_breakpoint_armed
                ):
                    if (
                        initial_global_return_original is None
                        or initial_global_breakpoint_armed
                        or len(initial_global_rows) != 1
                        or not initial_global_atomic_window_active
                    ):
                        raise RuntimeError(
                            "unexpected initial global MTRand return state"
                        )
                    if int(event.dwThreadId) != startup_handoff_main_thread_id:
                        raise RuntimeError(
                            "initial global MTRand return was not on the "
                            "startup main thread"
                        )
                    thread = kernel32.OpenThread(
                        THREAD_ACCESS | THREAD_SUSPEND_RESUME,
                        False,
                        event.dwThreadId,
                    )
                    if not thread:
                        raise ctypes.WinError(ctypes.get_last_error())
                    try:
                        context = WOW64_CONTEXT()
                        context.ContextFlags = CONTEXT_FULL
                        if not kernel32.Wow64GetThreadContext(
                            thread,
                            ctypes.byref(context),
                        ):
                            raise ctypes.WinError(ctypes.get_last_error())
                        post_state = read_memory(
                            process,
                            G_FRAMEWORK_MTRAND_ADDRESS,
                            MTRAND_STATE_BYTES,
                        )
                        observed_output = int(context.Eax) & 0xFFFFFFFF
                        if (
                            observed_output
                            != initial_global_mtrand_oracle.output
                            or post_state
                            != initial_global_mtrand_oracle.post_payload
                        ):
                            raise RuntimeError(
                                "initial global MTRand seeded transition "
                                "mismatch"
                            )
                        write_memory(
                            process,
                            initial_global_mtrand_oracle.caller,
                            initial_global_return_original,
                        )
                        initial_global_return_breakpoint_armed = False
                        kernel32.FlushInstructionCache(
                            process,
                            ctypes.c_void_p(
                                initial_global_mtrand_oracle.caller
                            ),
                            1,
                        )
                        context.Eip = initial_global_mtrand_oracle.caller
                        context.EFlags &= ~TRAP_FLAG
                        if not kernel32.Wow64SetThreadContext(
                            thread,
                            ctypes.byref(context),
                        ):
                            raise ctypes.WinError(ctypes.get_last_error())
                        row = initial_global_rows[0]
                        row["observed_output"] = observed_output
                        row["observed_post_index"] = struct.unpack_from(
                            "<I",
                            post_state,
                            MTRAND_STATE_WORDS * 4,
                        )[0]
                        row["observed_post_state_sha256"] = _sha256_bytes(
                            post_state
                        )
                        row["call_return_observed"] = True
                        row["atomic_window_completed"] = True
                        _resume_suspended_threads(
                            initial_global_suspended_threads
                        )
                        initial_global_atomic_window_active = False
                        row[
                            "other_threads_resumed_after_call_return"
                        ] = True
                        row["finished_perf_counter_ns"] = (
                            time.perf_counter_ns()
                        )
                        print(
                            "initial_global_mtrand_return "
                            f"tid={event.dwThreadId} "
                            f"output={observed_output} "
                            "post_index="
                            f"{row['observed_post_index']}",
                            flush=True,
                        )
                    finally:
                        kernel32.CloseHandle(thread)
                elif (
                    initial_global_mtrand_oracle is not None
                    and code
                    in (EXCEPTION_BREAKPOINT, STATUS_WX86_BREAKPOINT)
                    and exception_address
                    == initial_global_mtrand_oracle.wrapper_address
                ):
                    if (
                        not initial_global_breakpoint_armed
                        or initial_global_original is None
                        or initial_global_dispatch_instruction is None
                        or initial_global_dispatch_target is None
                        or initial_global_return_original is not None
                        or initial_global_return_breakpoint_armed
                        or initial_global_suspended_threads
                        or initial_global_atomic_window_active
                        or initial_global_rows
                    ):
                        raise RuntimeError(
                            "unexpected initial global MTRand wrapper state"
                        )
                    if (
                        stepping_thread is not None
                        or stepping_kind is not None
                        or stepping_suspended_threads
                        or held_thread
                        or pending_broker is not None
                        or post_bypass_return_breakpoint is not None
                        or pending_post_bypass_worker_yield is not None
                        or post_bypass_worker_yield_thread
                    ):
                        raise RuntimeError(
                            "initial global MTRand wrapper collided with "
                            "serialized command tracing"
                        )
                    if int(event.dwThreadId) != startup_handoff_main_thread_id:
                        raise RuntimeError(
                            "initial global MTRand wrapper was not on the "
                            "startup main thread"
                        )
                    thread = kernel32.OpenThread(
                        THREAD_ACCESS | THREAD_SUSPEND_RESUME,
                        False,
                        event.dwThreadId,
                    )
                    if not thread:
                        raise ctypes.WinError(ctypes.get_last_error())
                    try:
                        context = WOW64_CONTEXT()
                        context.ContextFlags = CONTEXT_FULL
                        if not kernel32.Wow64GetThreadContext(
                            thread,
                            ctypes.byref(context),
                        ):
                            raise ctypes.WinError(ctypes.get_last_error())
                        stack_return = _read_u32(process, int(context.Esp))
                        if stack_return != initial_global_mtrand_oracle.caller:
                            raise RuntimeError(
                                "initial global MTRand wrapper caller "
                                "mismatch"
                            )
                        live_state = read_memory(
                            process,
                            G_FRAMEWORK_MTRAND_ADDRESS,
                            MTRAND_STATE_BYTES,
                        )
                        live_index = struct.unpack_from(
                            "<I",
                            live_state,
                            MTRAND_STATE_WORDS * 4,
                        )[0]
                        changed = (
                            live_state
                            != initial_global_mtrand_oracle.pre_payload
                        )
                        if changed:
                            write_memory(
                                process,
                                G_FRAMEWORK_MTRAND_ADDRESS,
                                initial_global_mtrand_oracle.pre_payload,
                            )
                        if (
                            read_memory(
                                process,
                                G_FRAMEWORK_MTRAND_ADDRESS,
                                MTRAND_STATE_BYTES,
                            )
                            != initial_global_mtrand_oracle.pre_payload
                        ):
                            raise RuntimeError(
                                "initial global MTRand writeback mismatch"
                            )
                        initial_global_suspended_threads = (
                            _suspend_other_threads(
                                pid,
                                excluded_thread_id=int(event.dwThreadId),
                                access_denied_skip_events=(
                                    access_denied_thread_skip_events
                                ),
                            )
                        )
                        initial_global_atomic_window_active = True
                        initial_global_return_original = read_memory(
                            process,
                            initial_global_mtrand_oracle.caller,
                            1,
                        )
                        if initial_global_return_original == b"\xCC":
                            raise RuntimeError(
                                "initial global MTRand return already has "
                                "INT3"
                            )
                        write_memory(
                            process,
                            initial_global_mtrand_oracle.caller,
                            b"\xCC",
                        )
                        initial_global_return_breakpoint_armed = True
                        kernel32.FlushInstructionCache(
                            process,
                            ctypes.c_void_p(
                                initial_global_mtrand_oracle.caller
                            ),
                            1,
                        )
                        write_memory(
                            process,
                            initial_global_mtrand_oracle.wrapper_address,
                            initial_global_original[:1],
                        )
                        initial_global_breakpoint_armed = False
                        kernel32.FlushInstructionCache(
                            process,
                            ctypes.c_void_p(
                                initial_global_mtrand_oracle.wrapper_address
                            ),
                            1,
                        )
                        context.Eip = (
                            initial_global_mtrand_oracle.wrapper_address
                        )
                        context.EFlags &= ~TRAP_FLAG
                        if not kernel32.Wow64SetThreadContext(
                            thread,
                            ctypes.byref(context),
                        ):
                            raise ctypes.WinError(ctypes.get_last_error())
                        row = {
                            "thread_id": int(event.dwThreadId),
                            "source_path": str(
                                initial_global_mtrand_oracle.source_path
                            ),
                            "source_sha256": (
                                initial_global_mtrand_oracle.source_sha256
                            ),
                            "source_process_id": (
                                initial_global_mtrand_oracle.source_process_id
                            ),
                            "source_main_thread_id": (
                                initial_global_mtrand_oracle
                                .source_main_thread_id
                            ),
                            "semantic_sha256": (
                                initial_global_mtrand_oracle.semantic_sha256
                            ),
                            "seed": initial_global_mtrand_oracle.seed,
                            "call_address": (
                                initial_global_mtrand_oracle.call_address
                            ),
                            "call_instruction_hex": (
                                initial_global_dispatch_instruction.hex()
                            ),
                            "dispatch_target": (
                                initial_global_dispatch_target
                            ),
                            "wrapper_address": (
                                initial_global_mtrand_oracle.wrapper_address
                            ),
                            "wrapper_instruction_hex": (
                                initial_global_original.hex()
                            ),
                            "caller": initial_global_mtrand_oracle.caller,
                            "wrapper_entry_stack_return_address": (
                                stack_return
                            ),
                            "expected_output": (
                                initial_global_mtrand_oracle.output
                            ),
                            "live_pre_index": live_index,
                            "live_pre_state_sha256": _sha256_bytes(
                                live_state
                            ),
                            "restored_pre_index": (
                                initial_global_mtrand_oracle.pre_index
                            ),
                            "restored_pre_state_sha256": (
                                initial_global_mtrand_oracle
                                .pre_state_sha256
                            ),
                            "expected_post_index": (
                                initial_global_mtrand_oracle.post_index
                            ),
                            "expected_post_state_sha256": (
                                initial_global_mtrand_oracle
                                .post_state_sha256
                            ),
                            "changed": changed,
                            "bytes_written": (
                                MTRAND_STATE_BYTES if changed else 0
                            ),
                            "other_threads_suspended_count": len(
                                initial_global_suspended_threads
                            ),
                            "wrapper_entry_observed": True,
                            "call_target_entered": True,
                            "call_return_observed": False,
                            "atomic_window_completed": False,
                            "other_threads_resumed_after_call_return": False,
                            "process_memory_mutation": changed,
                            "process_context_mutation": True,
                            "persistent_file_modified": False,
                            "started_perf_counter_ns": (
                                time.perf_counter_ns()
                            ),
                        }
                        initial_global_rows.append(row)
                        print(
                            "initial_global_mtrand_restore "
                            f"tid={event.dwThreadId} "
                            f"live_index={live_index} "
                            "restored_index="
                            f"{initial_global_mtrand_oracle.pre_index} "
                            f"changed={changed} "
                            f"bytes={row['bytes_written']} "
                            "suspended="
                            f"{len(initial_global_suspended_threads)}",
                            flush=True,
                        )
                    finally:
                        kernel32.CloseHandle(thread)
                elif (
                    source_bound_board_precall_global_restore
                    and code
                    in (EXCEPTION_BREAKPOINT, STATUS_WX86_BREAKPOINT)
                    and exception_address
                    == SOURCE_BOUND_GLOBAL_PRECALL_ADDRESS
                ):
                    if (
                        not source_bound_precall_breakpoint_armed
                        or source_bound_precall_original is None
                    ):
                        raise RuntimeError(
                            "unexpected source-bound global precall "
                            "breakpoint state"
                        )
                    if (
                        stepping_thread is not None
                        or stepping_kind is not None
                        or stepping_suspended_threads
                        or source_bound_precall_suspended_threads
                        or held_thread
                        or pending_broker is not None
                        or post_bypass_return_breakpoint is not None
                        or pending_post_bypass_worker_yield is not None
                        or post_bypass_worker_yield_thread
                    ):
                        raise RuntimeError(
                            "source-bound global precall collided with "
                            "serialized command stepping"
                        )
                    assert source_bound_board_global_state is not None
                    assert source_bound_board_thread_crt_state is not None
                    assert startup_handoff_main_thread_id is not None
                    thread = kernel32.OpenThread(
                        THREAD_ACCESS | THREAD_SUSPEND_RESUME,
                        False,
                        event.dwThreadId,
                    )
                    if not thread:
                        raise ctypes.WinError(ctypes.get_last_error())
                    try:
                        context = WOW64_CONTEXT()
                        context.ContextFlags = CONTEXT_FULL
                        if not kernel32.Wow64GetThreadContext(
                            thread,
                            ctypes.byref(context),
                        ):
                            raise ctypes.WinError(ctypes.get_last_error())
                        base = _read_u32(
                            process,
                            G_SEXY_APP_BASE_ADDRESS,
                        )
                        framework_update = (
                            _read_i32(process, base + 0x4C4)
                            if base
                            else -1
                        )
                        source_target_update = (
                            source_bound_board_thread_crt_state
                            .source_framework_update
                        )
                        if (
                            not source_bound_precall_observations
                            and framework_update > source_target_update
                        ):
                            raise RuntimeError(
                                "source-bound global precall target was skipped"
                            )
                        active_board_address = (
                            _read_u32(
                                process,
                                base + ACTIVE_BOARD_OFFSET,
                            )
                            if base
                            else 0
                        )
                        active_board_vtable = (
                            _read_u32(process, active_board_address)
                            if active_board_address > 0
                            else 0
                        )
                        apply_precall_restore = (
                            not source_bound_precall_observations
                            and framework_update == source_target_update
                            and int(event.dwThreadId)
                            == startup_handoff_main_thread_id
                            and active_board_address > 0
                            and active_board_vtable == BOARD_VTABLE
                        )
                        if apply_precall_restore:
                            precall_observation = (
                                _apply_source_bound_global_precall_restore(
                                    process=process,
                                    process_id=pid,
                                    thread_id=int(event.dwThreadId),
                                    snapshot={
                                        "base": base,
                                        "update": framework_update,
                                    },
                                    desired=(
                                        source_bound_board_global_state
                                    ),
                                    expected_main_thread_id=(
                                        startup_handoff_main_thread_id
                                    ),
                                    target_framework_update=(
                                        source_target_update
                                    ),
                                    call_address=(
                                        SOURCE_BOUND_GLOBAL_PRECALL_ADDRESS
                                    ),
                                    call_instruction=(
                                        source_bound_precall_original
                                    ),
                                )
                            )
                            source_bound_precall_observations.append(
                                precall_observation
                            )
                        write_memory(
                            process,
                            SOURCE_BOUND_GLOBAL_PRECALL_ADDRESS,
                            source_bound_precall_original[:1],
                        )
                        source_bound_precall_breakpoint_armed = False
                        kernel32.FlushInstructionCache(
                            process,
                            ctypes.c_void_p(
                                SOURCE_BOUND_GLOBAL_PRECALL_ADDRESS
                            ),
                            1,
                        )
                        context.Eip = SOURCE_BOUND_GLOBAL_PRECALL_ADDRESS
                        context.EFlags |= TRAP_FLAG
                        if not kernel32.Wow64SetThreadContext(
                            thread,
                            ctypes.byref(context),
                        ):
                            raise ctypes.WinError(ctypes.get_last_error())
                        stepping_suspended_threads = (
                            _suspend_other_threads(
                                pid,
                                excluded_thread_id=int(event.dwThreadId),
                                access_denied_skip_events=(
                                    access_denied_thread_skip_events
                                ),
                            )
                        )
                        if apply_precall_restore:
                            precall_observation[
                                "other_threads_suspended_count"
                            ] = len(stepping_suspended_threads)
                            precall_observation[
                                "other_threads_suspended_until_board_call"
                            ] = True
                            precall_observation[
                                "atomic_window_completed"
                            ] = False
                            print(
                                "source_bound_global_precall_restore "
                                f"tid={event.dwThreadId} "
                                f"update={framework_update} bytes="
                                f"{precall_observation['bytes_written']} "
                                "before="
                                f"{precall_observation['live_before_state_sha256']} "
                                "after="
                                f"{precall_observation['restored']['state_sha256']} "
                                "suspended="
                                f"{len(stepping_suspended_threads)}",
                                flush=True,
                            )
                        stepping_thread = int(event.dwThreadId)
                        stepping_kind = (
                            "source_bound_global_precall"
                            if apply_precall_restore
                            else "source_bound_global_precall_skip"
                        )
                    finally:
                        kernel32.CloseHandle(thread)
                elif (
                    board_seed_address is not None
                    and code
                    in (EXCEPTION_BREAKPOINT, STATUS_WX86_BREAKPOINT)
                    and exception_address == board_seed_address
                ):
                    if (
                        not board_breakpoint_armed
                        or board_original is None
                    ):
                        raise RuntimeError(
                            "unexpected board seed breakpoint state"
                        )
                    if (
                        stepping_thread is not None
                        or stepping_kind is not None
                        or stepping_suspended_threads
                        or held_thread
                        or pending_broker is not None
                        or post_bypass_return_breakpoint is not None
                        or pending_post_bypass_worker_yield is not None
                        or post_bypass_worker_yield_thread
                    ):
                        raise RuntimeError(
                            "board seed breakpoint collided with serialized "
                            "command stepping"
                        )
                    thread = kernel32.OpenThread(
                        THREAD_ACCESS | THREAD_SUSPEND_RESUME,
                        False,
                        event.dwThreadId,
                    )
                    if not thread:
                        raise ctypes.WinError(ctypes.get_last_error())
                    try:
                        context = WOW64_CONTEXT()
                        context.ContextFlags = CONTEXT_FULL
                        if not kernel32.Wow64GetThreadContext(
                            thread, ctypes.byref(context)
                        ):
                            raise ctypes.WinError(ctypes.get_last_error())
                        observed_seed = int(context.Ecx)
                        attach_stabilization_board_collision = (
                            attach_stabilization_pending
                        )
                        apply_board_seed_overrides = (
                            _board_seed_override_applies(
                                board_seed_hits_seen,
                                board_seed_skip_count,
                            )
                            and not attach_stabilization_board_collision
                        )
                        board_seed_hits_seen += 1
                        base = _read_u32(
                            process,
                            G_SEXY_APP_BASE_ADDRESS,
                        )
                        framework_update = (
                            _read_i32(process, base + 0x4C4)
                            if base
                            else -1
                        )
                        active_board_address = (
                            _read_u32(
                                process,
                                base + ACTIVE_BOARD_OFFSET,
                            )
                            if base
                            else 0
                        )
                        rng_object_address = int(context.Eax)
                        rng_owner_address = (
                            rng_object_address - BOARD_MTRAND_OFFSET
                        )
                        rng_owner_vtable = (
                            _read_u32(process, rng_owner_address)
                            if rng_owner_address > 0
                            else 0
                        )
                        active_board_vtable = (
                            _read_u32(process, active_board_address)
                            if active_board_address > 0
                            else 0
                        )
                        source_bound_live_post_index: int | None = None
                        source_bound_live_post_sha256: str | None = None
                        source_bound_global_post_match: bool | None = None
                        source_bound_live_post_draw_verified: (
                            bool | None
                        ) = None
                        source_bound_correction_index_delta: (
                            int | None
                        ) = None
                        source_bound_bounded_correction_eligible: (
                            bool | None
                        ) = None
                        if source_bound_board_global_state is not None:
                            assert (
                                source_bound_board_thread_crt_state
                                is not None
                            )
                            source_target_update = (
                                source_bound_board_thread_crt_state
                                .source_framework_update
                            )
                            if (
                                source_bound_board_precall_global_restore
                                and framework_update == source_target_update
                                and (
                                    len(
                                        source_bound_precall_observations
                                    )
                                    != 1
                                    or not (
                                        source_bound_precall_suspended_threads
                                    )
                                    or source_bound_precall_observations[0].get(
                                        "thread_id"
                                    )
                                    != int(event.dwThreadId)
                                    or source_bound_precall_observations[0].get(
                                        "framework_update"
                                    )
                                    != framework_update
                                )
                            ):
                                raise RuntimeError(
                                    "source-bound Board call is not paired "
                                    "with one atomic global precall restore"
                                )
                            if source_bound_board_rows:
                                raise RuntimeError(
                                    "source-bound board anchor was observed "
                                    "more than once"
                                )
                            if framework_update > source_target_update:
                                raise RuntimeError(
                                    "source-bound board anchor target was "
                                    "skipped"
                                )
                            assert startup_handoff_main_thread_id is not None
                            assert source_bound_board_expected_output is not None
                            assert source_bound_board_expected_post is not None
                            assert (
                                source_bound_board_expected_post_index is not None
                            )
                            assert (
                                source_bound_board_expected_post_sha256 is not None
                            )
                            live_source_bound_global_post = read_memory(
                                process,
                                G_FRAMEWORK_MTRAND_ADDRESS,
                                MTRAND_STATE_BYTES,
                            )
                            source_bound_live_post_index = struct.unpack_from(
                                "<I",
                                live_source_bound_global_post,
                                MTRAND_STATE_WORDS * 4,
                            )[0]
                            source_bound_live_post_sha256 = (
                                "sha256:"
                                + hashlib.sha256(
                                    live_source_bound_global_post
                                ).hexdigest()
                            )
                            source_bound_global_post_match = (
                                live_source_bound_global_post
                                == source_bound_board_expected_post
                            )
                            source_bound_live_post_draw_verified = (
                                _live_global_post_matches_observed_draw(
                                    live_source_bound_global_post,
                                    observed_seed,
                                )
                            )
                            source_bound_correction_index_delta = (
                                _source_bound_global_post_correction_delta(
                                    live_source_bound_global_post,
                                    source_bound_board_expected_post,
                                )
                            )
                            source_seed_match = (
                                observed_seed
                                == source_bound_board_expected_output
                            )
                            source_bound_bounded_correction_eligible = (
                                _source_bound_global_correction_eligible(
                                    allow_correction=(
                                        allow_source_bound_board_global_correction
                                    ),
                                    live_post_draw_verified=(
                                        source_bound_live_post_draw_verified
                                    ),
                                    correction_index_delta=(
                                        source_bound_correction_index_delta
                                    ),
                                    observed_seed_match=source_seed_match,
                                    global_post_state_match=(
                                        source_bound_global_post_match
                                    ),
                                )
                            )
                            replay_main_thread_match = (
                                int(event.dwThreadId)
                                == startup_handoff_main_thread_id
                            )
                            rng_owner_board_match = (
                                rng_owner_vtable == BOARD_SEED_OWNER_VTABLE
                                and rng_object_address
                                == rng_owner_address + BOARD_MTRAND_OFFSET
                            )
                            active_board_context_match = (
                                active_board_address > 0
                                and active_board_vtable == BOARD_VTABLE
                            )
                            apply_board_seed_overrides = (
                                _source_bound_board_anchor_applies(
                                    framework_update=framework_update,
                                    target_framework_update=(
                                        source_target_update
                                    ),
                                    thread_id=int(event.dwThreadId),
                                    main_thread_id=(
                                        startup_handoff_main_thread_id
                                    ),
                                    rng_owner_vtable=rng_owner_vtable,
                                    expected_board_vtable=(
                                        BOARD_SEED_OWNER_VTABLE
                                    ),
                                    observed_seed=observed_seed,
                                    expected_seed=(
                                        source_bound_board_expected_output
                                    ),
                                    global_post_state_match=(
                                        source_bound_global_post_match
                                    ),
                                    bounded_global_correction=bool(
                                        source_bound_bounded_correction_eligible
                                    ),
                                    active_board_context_match=(
                                        active_board_context_match
                                    ),
                                    attach_stabilization_collision=(
                                        attach_stabilization_board_collision
                                    ),
                                )
                                and rng_owner_board_match
                            )
                            if (
                                source_bound_board_precall_global_restore
                                and framework_update == source_target_update
                                and not apply_board_seed_overrides
                            ):
                                raise RuntimeError(
                                    "source-bound global precall did not "
                                    "produce the natural Board anchor"
                                )
                        effective_seed = (
                            observed_seed
                            if (
                                board_seed_override is None
                                or not apply_board_seed_overrides
                            )
                            else board_seed_override
                        )
                        if attach_stabilization_board_collision:
                            attach_stabilization_pending = False
                            stabilization_failure = (
                                "board seed boundary arrived before attach "
                                "stabilization: framework_update="
                                f"{framework_update}, skipped_hits="
                                f"{attach_stabilization_skipped_hits}"
                            )
                            boundary_failures.append(
                                stabilization_failure
                            )
                            print(
                                "attach_stabilization_failure "
                                f"detail={stabilization_failure}",
                                flush=True,
                            )
                        global_rng_evidence: dict[str, Any] = {}
                        if (
                            global_rng_seed_override is not None
                            and apply_board_seed_overrides
                        ):
                            global_state_before = read_memory(
                                process,
                                G_FRAMEWORK_MTRAND_ADDRESS,
                                MTRAND_STATE_BYTES,
                            )
                            expected_global_state = _mtrand_state(
                                global_rng_seed_override
                            )
                            write_memory(
                                process,
                                G_FRAMEWORK_MTRAND_ADDRESS,
                                expected_global_state,
                            )
                            global_state_after = read_memory(
                                process,
                                G_FRAMEWORK_MTRAND_ADDRESS,
                                MTRAND_STATE_BYTES,
                            )
                            if global_state_after != expected_global_state:
                                raise RuntimeError(
                                    "global framework RNG state writeback "
                                    "mismatch"
                                )
                            global_rng_evidence = {
                                "global_rng_address": (
                                    G_FRAMEWORK_MTRAND_ADDRESS
                                ),
                                "global_rng_seed": (
                                    global_rng_seed_override
                                ),
                                "global_rng_state_bytes": (
                                    MTRAND_STATE_BYTES
                                ),
                                "global_rng_state_sha256_before": (
                                    "sha256:"
                                    + hashlib.sha256(
                                        global_state_before
                                    ).hexdigest()
                                ),
                                "global_rng_state_sha256_after": (
                                    "sha256:"
                                    + hashlib.sha256(
                                        global_state_after
                                    ).hexdigest()
                                ),
                                "global_rng_overridden": True,
                            }
                        thread_crt_rng_evidence: dict[str, Any] = {}
                        if (
                            thread_crt_rng_seed_override is not None
                            and apply_board_seed_overrides
                        ):
                            thread_crt_state = _thread_crt_rng_state(
                                process=process,
                                thread=thread,
                                context=context,
                                expected_thread_id=int(
                                    event.dwThreadId
                                ),
                            )
                            rand_state_address = thread_crt_state[
                                "rand_state_address"
                            ]
                            rand_state_before = _read_u32(
                                process,
                                rand_state_address,
                            )
                            write_memory(
                                process,
                                rand_state_address,
                                struct.pack(
                                    "<I",
                                    thread_crt_rng_seed_override,
                                ),
                            )
                            rand_state_after = _read_u32(
                                process,
                                rand_state_address,
                            )
                            if (
                                rand_state_after
                                != thread_crt_rng_seed_override
                            ):
                                raise RuntimeError(
                                    "thread CRT RNG state writeback "
                                    "mismatch"
                                )
                            thread_crt_rng_evidence = {
                                **thread_crt_state,
                                "thread_crt_rng_state_before": (
                                    rand_state_before
                                ),
                                "thread_crt_rng_state_after": (
                                    rand_state_after
                                ),
                                "thread_crt_rng_seed": (
                                    thread_crt_rng_seed_override
                                ),
                                "thread_crt_rng_overridden": True,
                            }
                        source_bound_board_evidence: dict[str, Any] = {}
                        if (
                            source_bound_board_global_state is not None
                            and apply_board_seed_overrides
                        ):
                            assert (
                                source_bound_board_thread_crt_state
                                is not None
                            )
                            source_bound_observation = (
                                _apply_source_bound_board_anchor(
                                    process=process,
                                    process_id=pid,
                                    thread=thread,
                                    thread_id=int(event.dwThreadId),
                                    context=context,
                                    framework_update=framework_update,
                                    expected_main_thread_id=(
                                        startup_handoff_main_thread_id
                                    ),
                                    board_seed_address=board_seed_address,
                                    board_original=board_original,
                                    active_board_address=(
                                        active_board_address
                                    ),
                                    rng_object_address=rng_object_address,
                                    rng_owner_address=rng_owner_address,
                                    rng_owner_vtable=rng_owner_vtable,
                                    observed_seed=observed_seed,
                                    expected_effective_seed=(
                                        effective_seed
                                    ),
                                    allow_global_correction=(
                                        allow_source_bound_board_global_correction
                                    ),
                                    global_state=(
                                        source_bound_board_global_state
                                    ),
                                    thread_state=(
                                        source_bound_board_thread_crt_state
                                    ),
                                )
                            )
                            if source_bound_board_precall_global_restore:
                                precall_observation = (
                                    source_bound_precall_observations[0]
                                )
                                precall_observation.update(
                                    {
                                        "board_call_reached": True,
                                        "board_call_address": (
                                            board_seed_address
                                        ),
                                        "board_call_framework_update": (
                                            framework_update
                                        ),
                                        "board_call_thread_id": int(
                                            event.dwThreadId
                                        ),
                                    }
                                )
                                source_bound_observation[
                                    "global_precall_restore"
                                ] = precall_observation
                            source_bound_board_rows.append(
                                source_bound_observation
                            )
                            source_bound_board_evidence = {
                                "source_bound_board_anchor_applied": True,
                                "source_bound_board_anchor_bytes_written": (
                                    source_bound_observation["bytes_written"]
                                ),
                            }
                            print(
                                "source_bound_board_anchor "
                                f"tid={event.dwThreadId} "
                                f"update={framework_update} "
                                "native_game_time="
                                f"{source_bound_observation['active_board_native_game_time']} "
                                f"observed_seed={observed_seed} "
                                f"effective_seed={effective_seed} "
                                "global_bytes="
                                f"{source_bound_observation['global']['bytes_written']} "
                                "crt_bytes="
                                f"{source_bound_observation['thread_crt']['bytes_written']}",
                                flush=True,
                            )
                        observation = {
                            "process_id": pid,
                            "thread_id": int(event.dwThreadId),
                            "perf_counter_ns": time.perf_counter_ns(),
                            "framework_update": framework_update,
                            "call_site_address": board_seed_address,
                            "rng_object_address": rng_object_address,
                            "rng_owner_address": rng_owner_address,
                            "rng_owner_vtable": rng_owner_vtable,
                            "rng_owner_is_board": (
                                rng_owner_vtable == BOARD_SEED_OWNER_VTABLE
                            ),
                            "active_board_address": active_board_address,
                            "active_board_vtable": active_board_vtable,
                            "matches_active_board_mtrand": (
                                active_board_address != 0
                                and rng_object_address
                                == active_board_address + BOARD_MTRAND_OFFSET
                            ),
                            "source_bound_target_update_match": (
                                framework_update == source_target_update
                                if source_bound_board_global_state is not None
                                else None
                            ),
                            "source_bound_main_thread_match": (
                                replay_main_thread_match
                                if source_bound_board_global_state is not None
                                else None
                            ),
                            "source_bound_rng_owner_board_match": (
                                rng_owner_board_match
                                if source_bound_board_global_state is not None
                                else None
                            ),
                            "source_bound_active_board_context_match": (
                                active_board_context_match
                                if source_bound_board_global_state is not None
                                else None
                            ),
                            "source_bound_observed_seed_match": (
                                source_seed_match
                                if source_bound_board_global_state is not None
                                else None
                            ),
                            "source_bound_attach_stabilization_clear": (
                                not attach_stabilization_board_collision
                                if source_bound_board_global_state is not None
                                else None
                            ),
                            "source_bound_live_global_post_index": (
                                source_bound_live_post_index
                            ),
                            "source_bound_live_global_post_sha256": (
                                source_bound_live_post_sha256
                            ),
                            "source_bound_expected_global_post_index": (
                                source_bound_board_expected_post_index
                            ),
                            "source_bound_expected_global_post_sha256": (
                                source_bound_board_expected_post_sha256
                            ),
                            "source_bound_global_post_state_match": (
                                source_bound_global_post_match
                            ),
                            "source_bound_live_post_draw_verified": (
                                source_bound_live_post_draw_verified
                            ),
                            "source_bound_correction_index_delta": (
                                source_bound_correction_index_delta
                            ),
                            "source_bound_bounded_correction_eligible": (
                                source_bound_bounded_correction_eligible
                            ),
                            "observed_seed": observed_seed,
                            "effective_seed": effective_seed,
                            "overridden": (
                                board_seed_override is not None
                                and apply_board_seed_overrides
                            ),
                            "skipped_before_override": (
                                not apply_board_seed_overrides
                            ),
                            **global_rng_evidence,
                            **thread_crt_rng_evidence,
                            **source_bound_board_evidence,
                        }
                        board_observations.append(observation)
                        print(
                            "board_seed_entry "
                            f"tid={observation['thread_id']} "
                            f"update={framework_update} "
                            f"call=0x{board_seed_address:08X} "
                            f"object=0x"
                            f"{observation['rng_object_address']:08X} "
                            "owner_vtable=0x"
                            f"{observation['rng_owner_vtable']:08X} "
                            "active=0x"
                            f"{observation['active_board_address']:08X} "
                            "active_vtable=0x"
                            f"{observation['active_board_vtable']:08X} "
                            "global_post_index="
                            f"{observation['source_bound_live_global_post_index']} "
                            "global_post_match="
                            f"{observation['source_bound_global_post_state_match']} "
                            f"observed={observed_seed} "
                            f"effective={effective_seed} "
                            f"overridden={observation['overridden']} "
                            "skipped_before_override="
                            f"{observation['skipped_before_override']} "
                            "global_rng_seed="
                            f"{global_rng_seed_override} "
                            "thread_crt_rng_seed="
                            f"{thread_crt_rng_seed_override}",
                            flush=True,
                        )
                        context.Ecx = effective_seed
                        write_memory(
                            process,
                            board_seed_address,
                            board_original,
                        )
                        board_breakpoint_armed = False
                        kernel32.FlushInstructionCache(
                            process,
                            ctypes.c_void_p(board_seed_address),
                            1,
                        )
                        context.Eip = board_seed_address
                        context.EFlags |= TRAP_FLAG
                        if not kernel32.Wow64SetThreadContext(
                            thread, ctypes.byref(context)
                        ):
                            raise ctypes.WinError(ctypes.get_last_error())
                        if (
                            source_bound_board_precall_global_restore
                            and apply_board_seed_overrides
                        ):
                            if not source_bound_precall_suspended_threads:
                                raise RuntimeError(
                                    "source-bound global precall atomic "
                                    "suspension was lost"
                                )
                            stepping_suspended_threads = (
                                source_bound_precall_suspended_threads
                            )
                            source_bound_precall_suspended_threads = []
                        else:
                            stepping_suspended_threads = (
                                _suspend_other_threads(
                                    pid,
                                    excluded_thread_id=int(
                                        event.dwThreadId
                                    ),
                                    access_denied_skip_events=(
                                        access_denied_thread_skip_events
                                    ),
                                )
                            )
                        stepping_thread = int(event.dwThreadId)
                        stepping_kind = (
                            "attach_stabilization_board_failure"
                            if attach_stabilization_board_collision
                            else (
                                "board_seed"
                                if apply_board_seed_overrides
                                else "board_seed_skip"
                            )
                        )
                    finally:
                        kernel32.CloseHandle(thread)
                elif (
                    trace_command_entries
                    and code
                    in (EXCEPTION_BREAKPOINT, STATUS_WX86_BREAKPOINT)
                    and exception_address == address
                    and pending_broker is not None
                    and bool(
                        pending_broker.get("event_driven_terminal", False)
                    )
                    and not bool(
                        pending_broker.get("event_driven_ready", False)
                    )
                ):
                    if not breakpoint_armed or original is None:
                        raise RuntimeError(
                            "terminal handoff breakpoint was not armed"
                        )
                    if int(event.dwThreadId) == pending_broker["thread_id"]:
                        raise RuntimeError(
                            "held main thread reached terminal handoff "
                            "breakpoint"
                        )
                    thread = kernel32.OpenThread(
                        THREAD_ACCESS | THREAD_SUSPEND_RESUME,
                        False,
                        event.dwThreadId,
                    )
                    if not thread:
                        raise ctypes.WinError(ctypes.get_last_error())
                    try:
                        context = WOW64_CONTEXT()
                        context.ContextFlags = CONTEXT_FULL
                        if not kernel32.Wow64GetThreadContext(
                            thread, ctypes.byref(context)
                        ):
                            raise ctypes.WinError(ctypes.get_last_error())
                        snapshot = _command_snapshot(process, context)
                        if blackout_native_timeline_offset:
                            snapshot = (
                                _normalize_blackout_native_timeline_snapshot(
                                    snapshot,
                                    native_timeline_offset=(
                                        blackout_native_timeline_offset
                                    ),
                                )
                            )
                        target_end = pending_broker[
                            "target_end_bit_position"
                        ]
                        settle_end = pending_broker[
                            "settle_end_bit_position"
                        ]
                        terminal_failure: str | None = None
                        terminal_payload_success: bool | None = None
                        if snapshot["base"] != pending_broker["base"]:
                            terminal_failure = (
                                "terminal service handoff replay base changed"
                            )
                        elif (
                            snapshot["buffer_read_bit_position"]
                            == target_end - 1
                        ):
                            (
                                terminal_payload_success,
                                terminal_failure,
                            ) = _audited_terminal_file_write_payload_handoff(
                                rows,
                                snapshot,
                                row_index=pending_broker["end_index"],
                                command_order_offset=command_order_offset,
                            )
                            if (
                                terminal_failure is None
                                and service_exit_timeline_commits
                            ):
                                terminal_failure = (
                                    "terminal file-write handoff found an "
                                    "existing service timeline commit"
                                )
                            if terminal_failure is None:
                                assert terminal_payload_success is not None
                                terminal_row_index = pending_broker[
                                    "end_index"
                                ]
                                terminal_row = rows[terminal_row_index]
                                last_demo_update_before = snapshot[
                                    "last_demo_update"
                                ]
                                recorded_update = int(
                                    terminal_row["update"]
                                )
                                write_memory(
                                    process,
                                    snapshot["base"] + 0x608,
                                    struct.pack("<i", target_end),
                                )
                                write_memory(
                                    process,
                                    snapshot["base"] + 0x61C,
                                    struct.pack("<i", recorded_update),
                                )
                                committed_read = _read_i32(
                                    process, snapshot["base"] + 0x608
                                )
                                committed_update = _read_i32(
                                    process, snapshot["base"] + 0x61C
                                )
                                if (
                                    committed_read != target_end
                                    or committed_update != recorded_update
                                ):
                                    terminal_failure = (
                                        "terminal file-write handoff write "
                                        "verification failed: read="
                                        f"{committed_read}/{target_end}, "
                                        "last_update="
                                        f"{committed_update}/{recorded_update}"
                                    )
                                else:
                                    before_eip = int(context.Eip)
                                    before_esp = int(context.Esp)
                                    context.Eax = (
                                        int(context.Eax) & 0xFFFFFF00
                                    ) | int(terminal_payload_success)
                                    context.Esp = before_esp + 4
                                    context.Eip = DEFERRED_FILE_WRITE_RESUME
                                    context.EFlags &= ~TRAP_FLAG
                                    snapshot[
                                        "buffer_read_bit_position"
                                    ] = committed_read
                                    snapshot[
                                        "last_demo_update"
                                    ] = committed_update
                                    handoff_observation = {
                                        "process_id": pid,
                                        "thread_id": int(event.dwThreadId),
                                        "framework_update": snapshot["update"],
                                        "row_index": terminal_row_index,
                                        "row_start": int(
                                            terminal_row["start"]
                                        ),
                                        "row_end": int(terminal_row["end"]),
                                        "recorded_success": (
                                            terminal_payload_success
                                        ),
                                        "read_before": target_end - 1,
                                        "read_after": committed_read,
                                        "last_demo_update_before": (
                                            last_demo_update_before
                                        ),
                                        "last_demo_update_after": (
                                            committed_update
                                        ),
                                        "instruction_pointer_before": (
                                            before_eip
                                        ),
                                        "instruction_pointer_after": (
                                            DEFERRED_FILE_WRITE_RESUME
                                        ),
                                        "stack_pointer_before": before_esp,
                                        "stack_pointer_after": before_esp + 4,
                                        "file_write_argument_pointer": snapshot[
                                            "file_write_argument_pointer"
                                        ],
                                        "file_write_argument_size": snapshot[
                                            "file_write_argument_size"
                                        ],
                                        "file_write_object_address": snapshot[
                                            "file_write_object_address"
                                        ],
                                        "file_write_object_length": snapshot[
                                            "file_write_object_length"
                                        ],
                                        "file_write_object_capacity": snapshot[
                                            "file_write_object_capacity"
                                        ],
                                        "file_write_object_data_address": snapshot[
                                            "file_write_object_data_address"
                                        ],
                                        "file_write_object_sha256": snapshot[
                                            "file_write_object_sha256"
                                        ],
                                        "file_write_object_prefix_hex": snapshot[
                                            "file_write_object_prefix_hex"
                                        ],
                                        "file_write_object_full_hex": snapshot[
                                            "file_write_object_full_hex"
                                        ],
                                        "file_write_object_full_truncated": snapshot[
                                            "file_write_object_full_truncated"
                                        ],
                                        "file_write_font_cache_member": snapshot[
                                            "file_write_font_cache_member"
                                        ],
                                    }
                                    terminal_file_write_payload_handoffs.append(
                                        handoff_observation
                                    )
                                    service_exit_timeline_commits.append(
                                        {
                                            "process_id": pid,
                                            "thread_id": int(
                                                event.dwThreadId
                                            ),
                                            "framework_update": snapshot[
                                                "update"
                                            ],
                                            "row_index": terminal_row_index,
                                            "command_bit_position": snapshot[
                                                "command_bit_position"
                                            ],
                                            "buffer_read_bit_position": (
                                                committed_read
                                            ),
                                            "last_demo_update_before": (
                                                last_demo_update_before
                                            ),
                                            "last_demo_update_after": (
                                                committed_update
                                            ),
                                            "recorded_update": recorded_update,
                                            "bytes_written": 4,
                                        }
                                    )
                                    print(
                                        "terminal_file_write_payload_handoff "
                                        f"tid={event.dwThreadId} "
                                        f"row={terminal_row_index} "
                                        f"success={int(terminal_payload_success)} "
                                        f"read={target_end - 1}->{committed_read} "
                                        "last_update="
                                        f"{last_demo_update_before}->"
                                        f"{committed_update}",
                                        flush=True,
                                    )
                        elif snapshot[
                            "buffer_read_bit_position"
                        ] not in (target_end, settle_end):
                            terminal_failure = (
                                "terminal service handoff missed an exact "
                                "boundary: read="
                                f"{snapshot['buffer_read_bit_position']}, "
                                f"allowed={target_end},{settle_end}"
                            )
                        pending_broker["event_driven_ready"] = True
                        pending_broker["event_observed_read"] = snapshot[
                            "buffer_read_bit_position"
                        ]
                        pending_broker["event_observed_needs"] = snapshot[
                            "needs_command"
                        ]
                        pending_broker["event_failure"] = terminal_failure
                        pending_broker["event_thread_id"] = int(
                            event.dwThreadId
                        )
                        print(
                            "service_terminal_handoff_event "
                            f"tid={event.dwThreadId} "
                            f"caller=0x{snapshot['return_address']:08X} "
                            f"update={snapshot['update']} "
                            "read_bitpos="
                            f"{snapshot['buffer_read_bit_position']} "
                            f"needs={snapshot['needs_command']} "
                            f"short={snapshot['is_short']} "
                            f"num={snapshot['command_number']} "
                            f"order={snapshot['command_order']} "
                            "cmd_bitpos="
                            f"{snapshot['command_bit_position']} "
                            f"failure={terminal_failure}",
                            flush=True,
                        )
                        write_memory(process, address, original)
                        breakpoint_armed = False
                        kernel32.FlushInstructionCache(
                            process, ctypes.c_void_p(address), 1
                        )
                        if terminal_payload_success is None:
                            context.Eip = address
                            context.EFlags &= ~TRAP_FLAG
                        if not kernel32.Wow64SetThreadContext(
                            thread, ctypes.byref(context)
                        ):
                            raise ctypes.WinError(ctypes.get_last_error())
                        if stepping_suspended_threads:
                            raise RuntimeError(
                                "terminal handoff suspension state leaked"
                            )
                        stepping_suspended_threads = _suspend_other_threads(
                            pid,
                            excluded_thread_id=pending_broker["thread_id"],
                            access_denied_skip_events=(
                                access_denied_thread_skip_events
                            ),
                        )
                    finally:
                        kernel32.CloseHandle(thread)
                elif (
                    trace_command_entries
                    and
                    code in (EXCEPTION_BREAKPOINT, STATUS_WX86_BREAKPOINT)
                    and exception_address == address
                ):
                    thread = kernel32.OpenThread(
                        THREAD_ACCESS | THREAD_SUSPEND_RESUME,
                        False,
                        event.dwThreadId,
                    )
                    if not thread:
                        raise ctypes.WinError(ctypes.get_last_error())
                    close_thread = True
                    try:
                        context = WOW64_CONTEXT()
                        context.ContextFlags = CONTEXT_FULL
                        if not kernel32.Wow64GetThreadContext(
                            thread, ctypes.byref(context)
                        ):
                            raise ctypes.WinError(ctypes.get_last_error())
                        snapshot = _command_snapshot(process, context)
                        if blackout_native_timeline_offset:
                            snapshot = (
                                _normalize_blackout_native_timeline_snapshot(
                                    snapshot,
                                    native_timeline_offset=(
                                        blackout_native_timeline_offset
                                    ),
                                )
                            )
                        if first_command_order is None:
                            first_command_order = snapshot["command_order"]
                        last_command_order = snapshot["command_order"]
                        last_snapshot_update = max(
                            last_snapshot_update,
                            snapshot["update"],
                        )
                        if (
                            startup_priority_bias is not None
                            and snapshot["update"]
                            >= startup_priority_bias.requested_until_update
                        ):
                            _restore_in_trace_startup_priority_bias(
                                startup_priority_bias,
                                framework_update=snapshot["update"],
                            )
                        if progress_every_updates is not None:
                            progress_bucket = (
                                snapshot["update"]
                                // progress_every_updates
                            )
                            if progress_bucket > last_progress_bucket:
                                last_progress_bucket = progress_bucket
                                print(
                                    "trace_progress "
                                    f"hit={hits + 1} "
                                    f"update={snapshot['update']} "
                                    "read_bitpos="
                                    f"{snapshot['buffer_read_bit_position']}",
                                    flush=True,
                                )
                        if (
                            pending_service_continuation is not None
                            and snapshot["return_address"]
                            == DEFERRED_FILE_WRITE_CALLER
                            and snapshot["base"]
                            == pending_service_continuation["base"]
                        ):
                            post_corridor_row = rows_by_start.get(
                                snapshot["command_bit_position"]
                            )
                            if post_corridor_row is not None:
                                exact_index = post_corridor_row[0]
                                exact_boundary_failure = (
                                    _snapshot_boundary_failure(
                                        rows_by_start,
                                        snapshot,
                                    command_order_offset=(
                                        command_order_offset
                                    ),
                                    )
                                )
                            else:
                                exact_index = -1
                                exact_boundary_failure = (
                                    "deferred service exit has no exact row"
                                )
                            if (
                                exact_index
                                > pending_service_continuation["end_index"]
                                and exact_boundary_failure is None
                            ):
                                service_continuation_verifications.append(
                                    {
                                        "process_id": pid,
                                        "thread_id": int(event.dwThreadId),
                                        "framework_update": snapshot[
                                            "update"
                                        ],
                                        "row_index": exact_index,
                                        "corridor_start_index": (
                                            pending_service_continuation[
                                                "start_index"
                                            ]
                                        ),
                                        "corridor_end_index": (
                                            pending_service_continuation[
                                                "end_index"
                                            ]
                                        ),
                                        "reentries": (
                                            pending_service_continuation[
                                                "reentries"
                                            ]
                                        ),
                                        "command_order_offset": (
                                            command_order_offset
                                        ),
                                        "service_exit_timeline_commit_count": len(
                                            service_exit_timeline_commits
                                        ),
                                    }
                                )
                                print(
                                    "service_continuation_verified "
                                    f"tid={event.dwThreadId} "
                                    f"update={snapshot['update']} "
                                    f"row={exact_index} "
                                    "reentries="
                                    f"{pending_service_continuation['reentries']} "
                                    "phase=file_write",
                                    flush=True,
                                )
                                pending_service_continuation = None
                            elif (
                                exact_index
                                == pending_service_continuation["end_index"]
                                + 1
                                and pending_service_continuation[
                                    "last_reentry_kind"
                                ]
                                == 1
                                and bool(
                                    pending_service_continuation.get(
                                        "allow_bounded_late_successful_file_write_corridor",
                                        False,
                                    )
                                )
                            ):
                                (
                                    candidate_order_offset,
                                    candidate_rebase_rows,
                                    candidate_rebase_failure,
                                ) = _audited_service_command_order_rebase(
                                    rows_by_start,
                                    rows,
                                    snapshot,
                                    corridor_start_index=(
                                        pending_service_continuation[
                                            "start_index"
                                        ]
                                    ),
                                    corridor_end_index=(
                                        pending_service_continuation[
                                            "end_index"
                                        ]
                                    ),
                                    command_order_offset=command_order_offset,
                                    accounted_rows=command_order_rebase_rows,
                                    require_last_demo_update=False,
                                    allow_exit_row=True,
                                )
                                deferred_exit_row: int | None = None
                                deferred_exit_commit_update: int | None = None
                                candidate_service_consumed_rows = (
                                    service_consumed_file_write_debt_rows
                                )
                                deferred_exit_failure: str | None = (
                                    candidate_rebase_failure
                                )
                                if deferred_exit_failure is None:
                                    (
                                        deferred_exit_row,
                                        deferred_exit_commit_update,
                                        deferred_exit_failure,
                                    ) = _audited_successful_file_write_deferred_exit_timeline_commit(
                                        rows,
                                        snapshot,
                                        corridor_start_index=(
                                            pending_service_continuation[
                                                "start_index"
                                            ]
                                        ),
                                        corridor_end_index=(
                                            pending_service_continuation[
                                                "end_index"
                                            ]
                                        ),
                                        previous_read_bit_position=(
                                            pending_service_continuation[
                                                "last_read_bit_position"
                                            ]
                                        ),
                                        previous_reentry_kind=(
                                            pending_service_continuation[
                                                "last_reentry_kind"
                                            ]
                                        ),
                                        command_order_offset=(
                                            candidate_order_offset
                                        ),
                                    )
                                if deferred_exit_failure is None:
                                    corridor_start = int(
                                        pending_service_continuation[
                                            "start_index"
                                        ]
                                    )
                                    corridor_end = int(
                                        pending_service_continuation[
                                            "end_index"
                                        ]
                                    )
                                    corridor_padding_rows = tuple(
                                        row_index
                                        for row_index in (
                                            diagnostic_successful_file_write_padding_rows
                                        )
                                        if corridor_start
                                        <= row_index
                                        <= corridor_end
                                    )
                                    if corridor_padding_rows:
                                        if not set(
                                            corridor_padding_rows
                                        ).issubset(candidate_rebase_rows):
                                            deferred_exit_failure = (
                                                "successful file-write padding "
                                                "row was not consumed by the "
                                                "audited service corridor"
                                            )
                                        else:
                                            supplemental_rows = tuple(
                                                row_index
                                                for row_index in range(
                                                    corridor_start,
                                                    corridor_end + 1,
                                                )
                                                if row_index
                                                not in corridor_padding_rows
                                                and row_index
                                                not in candidate_rebase_rows
                                            )
                                            candidate_service_consumed_rows = (
                                                tuple(
                                                    sorted(
                                                        set(
                                                            service_consumed_file_write_debt_rows
                                                        )
                                                        | set(
                                                            supplemental_rows
                                                        )
                                                    )
                                                )
                                            )
                                            if (
                                                set(
                                                    candidate_service_consumed_rows
                                                )
                                                & set(candidate_rebase_rows)
                                            ):
                                                deferred_exit_failure = (
                                                    "successful file-write "
                                                    "service debt overlaps the "
                                                    "command-order rebase"
                                                )
                                if deferred_exit_failure is None:
                                    assert deferred_exit_row is not None
                                    assert deferred_exit_commit_update is not None
                                    recorded_update = int(
                                        rows[deferred_exit_row]["update"]
                                    )
                                    live_last_demo_update = _read_i32(
                                        process,
                                        snapshot["base"] + 0x61C,
                                    )
                                    duplicate_commit = any(
                                        observation.get("row_index")
                                        == deferred_exit_row
                                        for observation in (
                                            service_exit_timeline_commits
                                        )
                                    )
                                    if (
                                        duplicate_commit
                                        or live_last_demo_update
                                        != snapshot["last_demo_update"]
                                    ):
                                        deferred_exit_failure = (
                                            "successful file-write deferred "
                                            "exit timeline precondition changed"
                                        )
                                if deferred_exit_failure is None:
                                    write_memory(
                                        process,
                                        snapshot["base"] + 0x61C,
                                        struct.pack(
                                            "<i", deferred_exit_commit_update
                                        ),
                                    )
                                    committed_update = _read_i32(
                                        process,
                                        snapshot["base"] + 0x61C,
                                    )
                                    if (
                                        committed_update
                                        != deferred_exit_commit_update
                                    ):
                                        raise RuntimeError(
                                            "successful file-write deferred "
                                            "exit timeline commit failed"
                                        )
                                    command_order_offset = (
                                        candidate_order_offset
                                    )
                                    command_order_rebase_rows = (
                                        candidate_rebase_rows
                                    )
                                    service_consumed_file_write_debt_rows = (
                                        candidate_service_consumed_rows
                                    )
                                    service_exit_timeline_commits.append(
                                        {
                                            "process_id": pid,
                                            "thread_id": int(event.dwThreadId),
                                            "framework_update": snapshot[
                                                "update"
                                            ],
                                            "row_index": deferred_exit_row,
                                            "command_bit_position": snapshot[
                                                "command_bit_position"
                                            ],
                                            "buffer_read_bit_position": snapshot[
                                                "buffer_read_bit_position"
                                            ],
                                            "last_demo_update_before": live_last_demo_update,
                                            "last_demo_update_after": committed_update,
                                            "recorded_update": recorded_update,
                                            "effective_committed_update": (
                                                committed_update
                                            ),
                                            "timeline_delta_updates": (
                                                committed_update
                                                - snapshot["update"]
                                            ),
                                            "recorded_timeline_delta_updates": (
                                                recorded_update
                                                - snapshot["update"]
                                            ),
                                            "late_commit_clamp_updates": (
                                                committed_update
                                                - recorded_update
                                            ),
                                            "corridor_end_row_index": pending_service_continuation[
                                                "end_index"
                                            ],
                                            "corridor_end_recorded_update": int(
                                                rows[
                                                    pending_service_continuation[
                                                        "end_index"
                                                    ]
                                                ]["update"]
                                            ),
                                            "bytes_written": 4,
                                        }
                                    )
                                    snapshot["last_demo_update"] = (
                                        committed_update
                                    )
                                    service_continuation_verifications.append(
                                        {
                                            "process_id": pid,
                                            "thread_id": int(event.dwThreadId),
                                            "framework_update": snapshot[
                                                "update"
                                            ],
                                            "row_index": deferred_exit_row,
                                            "corridor_start_index": pending_service_continuation[
                                                "start_index"
                                            ],
                                            "corridor_end_index": pending_service_continuation[
                                                "end_index"
                                            ],
                                            "reentries": pending_service_continuation[
                                                "reentries"
                                            ],
                                            "command_order_offset": command_order_offset,
                                            "service_exit_timeline_commit_count": len(
                                                service_exit_timeline_commits
                                            ),
                                            "service_consumed_file_write_debt_rows": (
                                                ",".join(
                                                    str(row_index)
                                                    for row_index in (
                                                        service_consumed_file_write_debt_rows
                                                    )
                                                )
                                            ),
                                            "verification_mode": (
                                                "successful_file_write_deferred_exit"
                                            ),
                                        }
                                    )
                                    print(
                                        "successful_file_write_deferred_exit "
                                        f"tid={event.dwThreadId} "
                                        f"update={snapshot['update']} "
                                        f"row={deferred_exit_row} "
                                        f"last={live_last_demo_update}->"
                                        f"{committed_update} "
                                        f"offset={command_order_offset} "
                                        "service_debt_rows="
                                        f"{service_consumed_file_write_debt_rows}",
                                        flush=True,
                                    )
                                    pending_service_continuation = None
                        deferred_file_write_failure: str | None = None
                        direct_font_cache_row_index: int | None = None
                        direct_font_cache_recorded_success: bool | None = None
                        allow_pre_loading_service_continuation = (
                            font_cache_manifest_requested
                            and startup_handoff_requested
                            and bool(service_exit_timeline_commits)
                            and bool(service_continuation_verifications)
                        )
                        if (
                            snapshot["return_address"]
                            == DEFERRED_FILE_WRITE_CALLER
                            and pending_service_continuation is None
                        ):
                            current_deferred_boundary = rows_by_start.get(
                                snapshot["command_bit_position"]
                            )
                            latest_timeline_commit = (
                                service_exit_timeline_commits[-1]
                                if service_exit_timeline_commits
                                else None
                            )
                            allow_one_tick_overdue_prepared = (
                                current_deferred_boundary is not None
                                and latest_timeline_commit is not None
                                and latest_timeline_commit.get("row_index")
                                == current_deferred_boundary[0]
                                and latest_timeline_commit.get(
                                    "framework_update"
                                )
                                == snapshot["update"]
                                and latest_timeline_commit.get(
                                    "recorded_update"
                                )
                                == snapshot["update"] - 1
                                and latest_timeline_commit.get(
                                    "effective_committed_update"
                                )
                                == snapshot["update"]
                                and latest_timeline_commit.get(
                                    "last_demo_update_after"
                                )
                                == snapshot["update"]
                                and latest_timeline_commit.get(
                                    "late_commit_clamp_updates"
                                )
                                == 1
                            )
                            (
                                debt_row_index,
                                recorded_success,
                                deferred_file_write_failure,
                            ) = _audited_deferred_file_write_result(
                                rows_by_start,
                                rows,
                                snapshot,
                                command_order_offset=command_order_offset,
                                accounted_rows=command_order_rebase_rows,
                                excluded_accounted_rows=tuple(
                                    row_index
                                    for row_index in command_order_rebase_rows
                                    if (
                                        row_index
                                        in diagnostic_successful_file_write_padding_rows
                                        or row_index
                                        in service_consumed_file_write_surplus_rows
                                        or not is_exact_successful_file_write_debt_row(
                                            row_index
                                        )
                                    )
                                ),
                                service_consumed_rows=(
                                    tuple(
                                        row_index
                                        for row_index in (
                                            service_consumed_file_write_debt_rows
                                        )
                                        if row_index
                                        not in service_consumed_file_write_surplus_rows
                                    )
                                ),
                                allow_one_tick_overdue_prepared=(
                                    allow_one_tick_overdue_prepared
                                ),
                                discharge_count=len(
                                    deferred_file_write_discharges
                                ),
                                pre_attach_rows=(
                                    pre_attach_file_write_debt_rows
                                ),
                                pre_attach_before_update=(
                                    pre_attach_file_write_debt_before_update
                                ),
                                allow_pre_loading_after_service_continuation=(
                                    allow_pre_loading_service_continuation
                                ),
                            )
                            if (
                                deferred_file_write_failure is None
                                and debt_row_index is None
                                and font_cache_manifest_requested
                            ):
                                current_index, _ = rows_by_start[
                                    snapshot["command_bit_position"]
                                ]
                                direct_font_cache_row_index = (
                                    current_index
                                    if snapshot["needs_command"] == 0
                                    else current_index + 1
                                )
                                if direct_font_cache_row_index >= len(rows):
                                    raise RuntimeError(
                                        "direct file-write result row is out of "
                                        "range"
                                    )
                                direct_row = rows[
                                    direct_font_cache_row_index
                                ]
                                direct_payload = direct_row.get("payload")
                                if (
                                    bool(direct_row["short_form"])
                                    or int(direct_row["command_number"]) != 16
                                    or int(direct_row["update"])
                                    != snapshot["update"]
                                    or not isinstance(direct_payload, dict)
                                    or not isinstance(
                                        direct_payload.get("success"), bool
                                    )
                                ):
                                    raise RuntimeError(
                                        "direct file-write result invariant "
                                        "failed"
                                    )
                                direct_font_cache_recorded_success = bool(
                                    direct_payload["success"]
                                )
                            if (
                                deferred_file_write_failure is None
                                and debt_row_index is not None
                            ):
                                assert recorded_success is not None
                                if (
                                    stepping_thread is not None
                                    or stepping_suspended_threads
                                    or pending_broker is not None
                                    or pending_service_continuation is not None
                                    or held_thread
                                ):
                                    raise RuntimeError(
                                        "deferred file-write discharge "
                                        "overlapped debugger serialization"
                                    )
                                before_eip = int(context.Eip)
                                before_esp = int(context.Esp)
                                context.Eax = (
                                    int(context.Eax) & 0xFFFFFF00
                                ) | int(recorded_success)
                                context.Esp = before_esp + 4
                                context.Eip = DEFERRED_FILE_WRITE_RESUME
                                context.EFlags &= ~TRAP_FLAG
                                if not kernel32.Wow64SetThreadContext(
                                    thread, ctypes.byref(context)
                                ):
                                    raise ctypes.WinError(
                                        ctypes.get_last_error()
                                    )
                                debt_row = rows[debt_row_index]
                                discharge_observation = {
                                    "process_id": pid,
                                    "thread_id": int(event.dwThreadId),
                                    "framework_update": snapshot["update"],
                                    "caller": snapshot["return_address"],
                                    "debt_row_index": debt_row_index,
                                    "debt_row_start": int(debt_row["start"]),
                                    "debt_row_end": int(debt_row["end"]),
                                    "debt_row_update": int(debt_row["update"]),
                                    "debt_command_number": int(
                                        debt_row["command_number"]
                                    ),
                                    "recorded_success": recorded_success,
                                    "demo_loading_complete": snapshot[
                                        "demo_loading_complete"
                                    ],
                                    "pre_loading_service_continuation": (
                                        snapshot["demo_loading_complete"] == 0
                                    ),
                                    "pre_attach_debt": (
                                        debt_row_index
                                        in pre_attach_file_write_debt_rows
                                    ),
                                    "service_consumed_debt": (
                                        debt_row_index
                                        in service_consumed_file_write_debt_rows
                                    ),
                                    "command_order": snapshot["command_order"],
                                    "command_order_offset": command_order_offset,
                                    "command_bit_position": snapshot[
                                        "command_bit_position"
                                    ],
                                    "buffer_read_bit_position": snapshot[
                                        "buffer_read_bit_position"
                                    ],
                                    "instruction_pointer_before": before_eip,
                                    "instruction_pointer_after": (
                                        DEFERRED_FILE_WRITE_RESUME
                                    ),
                                    "stack_pointer_before": before_esp,
                                    "stack_pointer_after": before_esp + 4,
                                    "file_write_argument_pointer": snapshot[
                                        "file_write_argument_pointer"
                                    ],
                                    "file_write_argument_size": snapshot[
                                        "file_write_argument_size"
                                    ],
                                    "file_write_object_address": snapshot[
                                        "file_write_object_address"
                                    ],
                                    "file_write_object_length": snapshot[
                                        "file_write_object_length"
                                    ],
                                    "file_write_object_capacity": snapshot[
                                        "file_write_object_capacity"
                                    ],
                                    "file_write_object_data_address": snapshot[
                                        "file_write_object_data_address"
                                    ],
                                    "file_write_object_sha256": snapshot[
                                        "file_write_object_sha256"
                                    ],
                                    "file_write_object_prefix_hex": snapshot[
                                        "file_write_object_prefix_hex"
                                    ],
                                    "file_write_object_full_hex": snapshot[
                                        "file_write_object_full_hex"
                                    ],
                                    "file_write_object_full_truncated": snapshot[
                                        "file_write_object_full_truncated"
                                    ],
                                    "file_write_font_cache_member": snapshot[
                                        "file_write_font_cache_member"
                                    ],
                                }
                                if snapshot["demo_loading_complete"] == 0:
                                    if not allow_pre_loading_service_continuation:
                                        raise RuntimeError(
                                            "pre-loading debt discharge lacks "
                                            "a verified service continuation"
                                        )
                                    discharge_observation[
                                        "service_continuation_verified_update"
                                    ] = int(
                                        service_continuation_verifications[-1][
                                            "framework_update"
                                        ]
                                    )
                                deferred_file_write_discharges.append(
                                    discharge_observation
                                )
                                hits += 1
                                print(
                                    "deferred_file_write_discharge "
                                    f"hit={hits} tid={event.dwThreadId} "
                                    f"update={snapshot['update']} "
                                    f"debt_row={debt_row_index} "
                                    f"success={int(recorded_success)} "
                                    "cmd_bitpos="
                                    f"{snapshot['command_bit_position']} "
                                    "read_bitpos="
                                    f"{snapshot['buffer_read_bit_position']}",
                                    flush=True,
                                )
                                if not kernel32.ContinueDebugEvent(
                                    event.dwProcessId,
                                    event.dwThreadId,
                                    status,
                                ):
                                    raise ctypes.WinError(
                                        ctypes.get_last_error()
                                    )
                                continue
                            if (
                                deferred_file_write_failure is not None
                                and font_cache_manifest_requested
                            ):
                                assert font_cache_manifest is not None
                                assert font_cache_manifest_sha256 is not None
                                assert (
                                    font_cache_manifest_main_pak_sha256
                                    is not None
                                )
                                observed_members = tuple(
                                    (
                                        str(
                                            observation[
                                                "file_write_font_cache_member"
                                            ]
                                        ),
                                        int(
                                            observation[
                                                "file_write_argument_size"
                                            ]
                                        ),
                                    )
                                    for observation in (
                                        terminal_file_write_payload_handoffs
                                        + service_file_write_header_claims
                                        + direct_font_cache_file_write_observations
                                        + deferred_file_write_discharges
                                        + font_cache_manifest_completion_discharges
                                    )
                                )
                                (
                                    manifest_recorded_success,
                                    manifest_completion_failure,
                                ) = _audited_font_cache_manifest_completion(
                                    rows_by_start,
                                    rows,
                                    snapshot,
                                    command_order_offset=(
                                        command_order_offset
                                    ),
                                    accounted_rows=(
                                        command_order_rebase_rows
                                    ),
                                    excluded_accounted_rows=tuple(
                                        row_index
                                        for row_index in (
                                            command_order_rebase_rows
                                        )
                                        if (
                                            row_index
                                            in diagnostic_successful_file_write_padding_rows
                                            or row_index
                                            in service_consumed_file_write_surplus_rows
                                            or not is_exact_successful_file_write_debt_row(
                                                row_index
                                            )
                                        )
                                    ),
                                    service_consumed_rows=(
                                        tuple(
                                            row_index
                                            for row_index in (
                                                service_consumed_file_write_debt_rows
                                            )
                                            if row_index
                                            not in service_consumed_file_write_surplus_rows
                                        )
                                    ),
                                    allow_one_tick_overdue_prepared=(
                                        allow_one_tick_overdue_prepared
                                    ),
                                    discharge_count=len(
                                        deferred_file_write_discharges
                                    ),
                                    pre_attach_rows=(
                                        pre_attach_file_write_debt_rows
                                    ),
                                    pre_attach_before_update=(
                                        pre_attach_file_write_debt_before_update
                                    ),
                                    startup_file_write_rows=(
                                        startup_file_write_rows
                                    ),
                                    manifest=font_cache_manifest,
                                    observed_members=observed_members,
                                    current_member=str(
                                        snapshot[
                                            "file_write_font_cache_member"
                                        ]
                                    ),
                                    current_size=int(
                                        snapshot[
                                            "file_write_argument_size"
                                        ]
                                    ),
                                    completion_count=len(
                                        font_cache_manifest_completion_discharges
                                    ),
                                    service_handoff_count=(
                                        len(
                                            terminal_file_write_payload_handoffs
                                        )
                                        + len(
                                            service_file_write_header_claims
                                        )
                                    ),
                                    direct_match_count=len(
                                        direct_font_cache_file_write_observations
                                    ),
                                    allow_pre_loading_after_service_continuation=(
                                        allow_pre_loading_service_continuation
                                    ),
                                )
                                if manifest_recorded_success is not None:
                                    if (
                                        stepping_thread is not None
                                        or stepping_suspended_threads
                                        or pending_broker is not None
                                        or pending_service_continuation
                                        is not None
                                        or held_thread
                                    ):
                                        raise RuntimeError(
                                            "font-cache manifest completion "
                                            "overlapped debugger serialization"
                                        )
                                    before_eip = int(context.Eip)
                                    before_esp = int(context.Esp)
                                    context.Eax = (
                                        int(context.Eax) & 0xFFFFFF00
                                    ) | int(manifest_recorded_success)
                                    context.Esp = before_esp + 4
                                    context.Eip = DEFERRED_FILE_WRITE_RESUME
                                    context.EFlags &= ~TRAP_FLAG
                                    if not kernel32.Wow64SetThreadContext(
                                        thread, ctypes.byref(context)
                                    ):
                                        raise ctypes.WinError(
                                            ctypes.get_last_error()
                                        )
                                    normalized_member = (
                                        _normalize_font_cache_member(
                                            str(
                                                snapshot[
                                                    "file_write_font_cache_member"
                                                ]
                                            )
                                        )
                                    )
                                    assert normalized_member is not None
                                    completion_observation = {
                                        "process_id": pid,
                                        "thread_id": int(event.dwThreadId),
                                        "framework_update": snapshot[
                                            "update"
                                        ],
                                        "caller": snapshot["return_address"],
                                        "recorded_success": (
                                            manifest_recorded_success
                                        ),
                                        "demo_loading_complete": snapshot[
                                            "demo_loading_complete"
                                        ],
                                        "pre_loading_service_continuation": (
                                            snapshot[
                                                "demo_loading_complete"
                                            ]
                                            == 0
                                        ),
                                        "command_order": snapshot[
                                            "command_order"
                                        ],
                                        "command_order_offset": (
                                            command_order_offset
                                        ),
                                        "command_bit_position": snapshot[
                                            "command_bit_position"
                                        ],
                                        "buffer_read_bit_position": snapshot[
                                            "buffer_read_bit_position"
                                        ],
                                        "instruction_pointer_before": (
                                            before_eip
                                        ),
                                        "instruction_pointer_after": (
                                            DEFERRED_FILE_WRITE_RESUME
                                        ),
                                        "stack_pointer_before": before_esp,
                                        "stack_pointer_after": before_esp + 4,
                                        "file_write_argument_pointer": snapshot[
                                            "file_write_argument_pointer"
                                        ],
                                        "file_write_argument_size": snapshot[
                                            "file_write_argument_size"
                                        ],
                                        "file_write_object_sha256": snapshot[
                                            "file_write_object_sha256"
                                        ],
                                        "file_write_font_cache_member": (
                                            normalized_member
                                        ),
                                        "manifest_expected_size": (
                                            font_cache_manifest[
                                                normalized_member
                                            ]
                                        ),
                                        "manifest_entry_count": len(
                                            font_cache_manifest
                                        ),
                                        "manifest_sha256": (
                                            font_cache_manifest_sha256
                                        ),
                                        "main_pak_sha256": (
                                            font_cache_manifest_main_pak_sha256
                                        ),
                                        "startup_file_write_row_count": len(
                                            startup_file_write_rows
                                        ),
                                        "explicit_debt_discharge_count": len(
                                            deferred_file_write_discharges
                                        ),
                                        "prior_observed_member_count": len(
                                            observed_members
                                        ),
                                    }
                                    if snapshot["demo_loading_complete"] == 0:
                                        if not allow_pre_loading_service_continuation:
                                            raise RuntimeError(
                                                "pre-loading manifest completion "
                                                "lacks a verified service "
                                                "continuation"
                                            )
                                        completion_observation[
                                            "service_continuation_verified_update"
                                        ] = int(
                                            service_continuation_verifications[-1][
                                                "framework_update"
                                            ]
                                        )
                                    font_cache_manifest_completion_discharges.append(
                                        completion_observation
                                    )
                                    hits += 1
                                    print(
                                        "font_cache_manifest_completion_discharge "
                                        f"hit={hits} tid={event.dwThreadId} "
                                        f"update={snapshot['update']} "
                                        f"member={normalized_member} "
                                        "size="
                                        f"{snapshot['file_write_argument_size']} "
                                        "success="
                                        f"{int(manifest_recorded_success)}",
                                        flush=True,
                                    )
                                    if not kernel32.ContinueDebugEvent(
                                        event.dwProcessId,
                                        event.dwThreadId,
                                        status,
                                    ):
                                        raise ctypes.WinError(
                                            ctypes.get_last_error()
                                        )
                                    continue
                                deferred_file_write_failure = (
                                    manifest_completion_failure
                                )
                        attach_stabilization_call = False
                        attach_stabilization_failure: str | None = None
                        if attach_stabilization_pending:
                            assert (
                                attach_stabilization_after_update is not None
                            )
                            assert (
                                attach_stabilization_main_thread_id is not None
                            )
                            if (
                                int(event.dwThreadId)
                                != attach_stabilization_main_thread_id
                            ):
                                stabilization_reason = (
                                    "attach stabilization thread mismatch: "
                                    f"{int(event.dwThreadId)} != "
                                    f"{attach_stabilization_main_thread_id}"
                                )
                            else:
                                stabilization_rebase_offset = (
                                    command_order_offset
                                )
                                stabilization_rebase_rows = (
                                    command_order_rebase_rows
                                )
                                stabilization_candidate_rows = (
                                    blackout_file_write_rebase_candidate_rows
                                )
                                stabilization_timeline_offset = (
                                    blackout_native_timeline_offset
                                )
                                if (
                                    allow_initial_file_write_order_rebase
                                    and not initial_offline_boundary_seen
                                ):
                                    assert (
                                        command_order_rebase_after_update
                                        is not None
                                    )
                                    if blackout_envelope_set_requested:
                                        (
                                            stabilization_rebase_offset,
                                            stabilization_candidate_rows,
                                            stabilization_timeline_offset,
                                            stabilization_reason,
                                        ) = _audited_post_blackout_file_write_offset_envelope_set(
                                            rows_by_start,
                                            rows,
                                            snapshot,
                                            detached_after_update=(
                                                command_order_rebase_after_update
                                            ),
                                            allowed_offset_pairs=(
                                                blackout_allowed_offset_pairs
                                            ),
                                        )
                                    elif blackout_envelope_requested:
                                        assert (
                                            blackout_expected_command_order_offset
                                            is not None
                                        )
                                        assert (
                                            blackout_expected_native_timeline_offset
                                            is not None
                                        )
                                        (
                                            stabilization_rebase_offset,
                                            stabilization_candidate_rows,
                                            stabilization_timeline_offset,
                                            stabilization_reason,
                                        ) = _audited_post_blackout_file_write_offset_envelope(
                                            rows_by_start,
                                            rows,
                                            snapshot,
                                            detached_after_update=(
                                                command_order_rebase_after_update
                                            ),
                                            expected_command_order_offset=(
                                                blackout_expected_command_order_offset
                                            ),
                                            expected_native_timeline_offset=(
                                                blackout_expected_native_timeline_offset
                                            ),
                                        )
                                    else:
                                        (
                                            stabilization_rebase_offset,
                                            stabilization_rebase_rows,
                                            stabilization_reason,
                                        ) = _audited_post_blackout_attach_stabilization(
                                            rows_by_start,
                                            rows,
                                            snapshot,
                                            detached_after_update=(
                                                command_order_rebase_after_update
                                            ),
                                        )
                                else:
                                    stabilization_reason = (
                                        _attach_stabilization_boundary_failure(
                                            rows_by_start,
                                            snapshot,
                                            command_order_offset=(
                                                command_order_offset
                                            ),
                                        )
                                    )
                            stabilization_observation: dict[
                                str, int | str | bool
                            ] = {
                                "thread_id": int(event.dwThreadId),
                                "framework_update": snapshot["update"],
                                "return_address": snapshot[
                                    "return_address"
                                ],
                                "buffer_read_bit_position": snapshot[
                                    "buffer_read_bit_position"
                                ],
                                "needs_command": snapshot[
                                    "needs_command"
                                ],
                                "command_number": snapshot[
                                    "command_number"
                                ],
                                "command_order": snapshot[
                                    "command_order"
                                ],
                                "command_bit_position": snapshot[
                                    "command_bit_position"
                                ],
                                "candidate_command_order_offset": (
                                    stabilization_rebase_offset
                                    if int(event.dwThreadId)
                                    == attach_stabilization_main_thread_id
                                    else command_order_offset
                                ),
                                "candidate_native_timeline_offset": (
                                    stabilization_timeline_offset
                                    if int(event.dwThreadId)
                                    == attach_stabilization_main_thread_id
                                    else blackout_native_timeline_offset
                                ),
                                "accepted": stabilization_reason is None,
                                "reason": (
                                    "exact_complete_main_boundary"
                                    if stabilization_reason is None
                                    else stabilization_reason
                                ),
                            }
                            attach_stabilization_observations.append(
                                stabilization_observation
                            )
                            if stabilization_reason is None:
                                if (
                                    allow_initial_file_write_order_rebase
                                    and not initial_offline_boundary_seen
                                ):
                                    command_order_offset = (
                                        stabilization_rebase_offset
                                    )
                                    if blackout_envelope_requested:
                                        blackout_file_write_rebase_candidate_rows = (
                                            stabilization_candidate_rows
                                        )
                                        blackout_native_timeline_offset = (
                                            stabilization_timeline_offset
                                        )
                                    else:
                                        command_order_rebase_rows = (
                                            stabilization_rebase_rows
                                        )
                                    initial_offline_boundary_seen = True
                                    rebase_label = (
                                        "command_order_rebase_envelope_set"
                                        if blackout_envelope_set_requested
                                        else (
                                            "command_order_rebase_envelope"
                                            if blackout_envelope_requested
                                            else "command_order_rebase"
                                        )
                                    )
                                    rebase_rows_label = (
                                        "candidate_rows"
                                        if blackout_envelope_requested
                                        else "accounted_rows"
                                    )
                                    rebase_rows_values = (
                                        blackout_file_write_rebase_candidate_rows
                                        if blackout_envelope_requested
                                        else command_order_rebase_rows
                                    )
                                    timeline_label = (
                                        " native_timeline_offset="
                                        f"{blackout_native_timeline_offset}"
                                        if blackout_envelope_requested
                                        else ""
                                    )
                                    print(
                                        f"{rebase_label} "
                                        f"tid={event.dwThreadId} "
                                        f"update={snapshot['update']} "
                                        "live_order="
                                        f"{snapshot['command_order']} "
                                        f"offset={command_order_offset} "
                                        "offline_order="
                                        f"{snapshot['command_order'] + command_order_offset} "
                                        f"{rebase_rows_label}="
                                        + ",".join(
                                            str(index)
                                            for index in rebase_rows_values
                                        )
                                        + timeline_label
                                        + " phase=post_blackout_stabilization",
                                        flush=True,
                                    )
                                attach_stabilization_pending = False
                                attach_stabilization_verified = True
                                attach_stabilization_verified_update = (
                                    snapshot["update"]
                                )
                                attach_stabilization_verified_command_order = (
                                    snapshot["command_order"]
                                )
                                print(
                                    "attach_stabilization_verified "
                                    f"tid={event.dwThreadId} "
                                    f"update={snapshot['update']} "
                                    "command_order="
                                    f"{snapshot['command_order']} "
                                    "command_bitpos="
                                    f"{snapshot['command_bit_position']} "
                                    "skipped_hits="
                                    f"{attach_stabilization_skipped_hits}",
                                    flush=True,
                                )
                                if blackout_native_timeline_offset:
                                    snapshot = (
                                        _normalize_blackout_native_timeline_snapshot(
                                            snapshot,
                                            native_timeline_offset=(
                                                blackout_native_timeline_offset
                                            ),
                                        )
                                    )
                                    print(
                                        "blackout_native_timeline_normalized "
                                        f"native_update={snapshot['native_update']} "
                                        f"update={snapshot['update']} "
                                        "native_last_demo_update="
                                        f"{snapshot['native_last_demo_update']} "
                                        "last_demo_update="
                                        f"{snapshot['last_demo_update']} "
                                        "offset="
                                        f"{blackout_native_timeline_offset}",
                                        flush=True,
                                    )
                            else:
                                attach_stabilization_call = True
                                attach_stabilization_skipped_hits += 1
                                update_delta = (
                                    snapshot["update"]
                                    - attach_stabilization_after_update
                                )
                                budget_exhausted = (
                                    attach_stabilization_skipped_hits
                                    >= ATTACH_STABILIZATION_MAX_SKIPPED_HITS
                                    or update_delta
                                    > ATTACH_STABILIZATION_MAX_UPDATE_DELTA
                                )
                                print(
                                    "attach_stabilization_observation "
                                    f"hit={attach_stabilization_skipped_hits} "
                                    f"tid={event.dwThreadId} "
                                    f"update={snapshot['update']} "
                                    f"update_delta={update_delta} "
                                    "command_order="
                                    f"{snapshot['command_order']} "
                                    "command_bitpos="
                                    f"{snapshot['command_bit_position']} "
                                    f"reason={stabilization_reason}",
                                    flush=True,
                                )
                                if budget_exhausted:
                                    attach_stabilization_pending = False
                                    attach_stabilization_failure = (
                                        "attach stabilization budget exhausted "
                                        "before an exact complete main "
                                        "boundary: skipped_hits="
                                        f"{attach_stabilization_skipped_hits}, "
                                        f"update_delta={update_delta}, "
                                        f"last_reason={stabilization_reason}"
                                    )
                        pre_stream_call = (
                            allow_pre_stream_commands
                            and not offline_stream_started
                            and snapshot["command_order"] < 0
                        )
                        untracked_command_call = (
                            pre_stream_call or attach_stabilization_call
                        )
                        if snapshot["command_order"] >= 0:
                            offline_stream_started = True
                        service_transient_call = False
                        service_terminal_prepared_call = False
                        service_terminal_prepared_row: int | None = None
                        startup_worker_echo_service_block: (
                            tuple[int, int, int] | None
                        ) = None
                        service_transient_failure: str | None = None
                        exact_offline_row = rows_by_start.get(
                            snapshot["command_bit_position"]
                        )
                        if pending_service_continuation is not None:
                            if (
                                snapshot["base"]
                                != pending_service_continuation["base"]
                            ):
                                service_transient_failure = (
                                    "service continuation replay base changed"
                                )
                            elif pending_service_continuation[
                                "last_reentry_kind"
                            ] == 12:
                                (
                                    committed_continuation,
                                    worker_echo_receipt,
                                    service_transient_failure,
                                ) = _consume_startup_post_bypass_worker_echo(
                                    rows,
                                    pending_service_continuation,
                                    snapshot,
                                    thread_id=int(event.dwThreadId),
                                    command_order_offset=(
                                        command_order_offset
                                    ),
                                )
                                if service_transient_failure is None:
                                    assert committed_continuation is not None
                                    assert worker_echo_receipt is not None
                                    pending_service_continuation = (
                                        committed_continuation
                                    )
                                    echo_block_start = int(
                                        worker_echo_receipt[
                                            "same_update_service_block_start_index"
                                        ]
                                    )
                                    if echo_block_start >= 0:
                                        startup_worker_echo_service_block = (
                                            echo_block_start,
                                            int(
                                                worker_echo_receipt[
                                                    "same_update_service_block_end_index"
                                                ]
                                            ),
                                            int(
                                                worker_echo_receipt[
                                                    "same_update_service_block_target_end_bit_position"
                                                ]
                                            ),
                                        )
                                    print(
                                        "startup_post_bypass_worker_echo "
                                        f"tid={event.dwThreadId} "
                                        "update="
                                        f"{worker_echo_receipt['framework_update']} "
                                        "row="
                                        f"{worker_echo_receipt['row_index']} "
                                        "read_bitpos="
                                        f"{worker_echo_receipt['read_bit_position']} "
                                        "same_update_service_rows="
                                        f"{worker_echo_receipt['same_update_service_block_start_index']}:"
                                        f"{worker_echo_receipt['same_update_service_block_end_index']}",
                                        flush=True,
                                    )
                            elif pending_service_continuation[
                                "last_reentry_kind"
                            ] == 13:
                                claim_row_index = pending_service_continuation.get(
                                    "active_file_write_claim_row_index"
                                )
                                claim_receipt_index = (
                                    pending_service_continuation.get(
                                        "active_file_write_claim_receipt_index"
                                    )
                                )
                                if (
                                    not isinstance(claim_row_index, int)
                                    or isinstance(claim_row_index, bool)
                                    or not isinstance(claim_receipt_index, int)
                                    or isinstance(claim_receipt_index, bool)
                                    or int(event.dwThreadId)
                                    != pending_service_continuation["thread_id"]
                                ):
                                    service_transient_failure = (
                                        "service file-write payload completion "
                                        "claim state is invalid"
                                    )
                                else:
                                    (
                                        completed_claim_row,
                                        service_transient_failure,
                                    ) = _audited_service_file_write_payload_completion(
                                        rows,
                                        snapshot,
                                        row_index=claim_row_index,
                                        previous_read_bit_position=(
                                            pending_service_continuation[
                                                "last_read_bit_position"
                                            ]
                                        ),
                                        previous_command_bit_position=(
                                            pending_service_continuation[
                                                "last_command_bit_position"
                                            ]
                                        ),
                                        previous_command_order=(
                                            pending_service_continuation[
                                                "last_command_order"
                                            ]
                                        ),
                                        previous_reentry_kind=(
                                            pending_service_continuation[
                                                "last_reentry_kind"
                                            ]
                                        ),
                                        command_order_offset=(
                                            command_order_offset
                                        ),
                                    )
                                    if service_transient_failure is None:
                                        assert completed_claim_row is not None
                                        receipt = (
                                            service_file_write_header_claims[
                                                claim_receipt_index
                                            ]
                                            if 0
                                            <= claim_receipt_index
                                            < len(
                                                service_file_write_header_claims
                                            )
                                            else None
                                        )
                                        if (
                                            not isinstance(receipt, dict)
                                            or receipt.get("mechanism")
                                            != "broker_adjacent_current_update_file_write"
                                            or receipt.get("process_id") != pid
                                            or receipt.get("row_index")
                                            != completed_claim_row
                                            or receipt.get(
                                                "payload_completion_verified"
                                            )
                                            is not False
                                        ):
                                            service_transient_failure = (
                                                "service file-write payload "
                                                "completion receipt identity "
                                                "mismatch"
                                            )
                                        else:
                                            receipt.update(
                                                {
                                                    "payload_completion_verified": True,
                                                    "payload_completion_thread_id": int(
                                                        event.dwThreadId
                                                    ),
                                                    "payload_completion_framework_update": snapshot[
                                                        "update"
                                                    ],
                                                    "payload_completion_command_bit_position": snapshot[
                                                        "command_bit_position"
                                                    ],
                                                    "payload_completion_buffer_read_bit_position": snapshot[
                                                        "buffer_read_bit_position"
                                                    ],
                                                    "payload_completion_command_order": snapshot[
                                                        "command_order"
                                                    ],
                                                }
                                            )
                                            service_transient_call = True
                                            pending_service_continuation[
                                                "last_read_bit_position"
                                            ] = snapshot[
                                                "buffer_read_bit_position"
                                            ]
                                            pending_service_continuation[
                                                "last_command_bit_position"
                                            ] = snapshot[
                                                "command_bit_position"
                                            ]
                                            pending_service_continuation[
                                                "last_command_order"
                                            ] = snapshot["command_order"]
                                            pending_service_continuation[
                                                "last_reentry_update"
                                            ] = snapshot["update"]
                                            pending_service_continuation[
                                                "last_reentry_kind"
                                            ] = 1
                                            pending_service_continuation[
                                                "reentries"
                                            ] += 1
                                            pending_service_continuation[
                                                "last_brokered_row_index"
                                            ] = completed_claim_row
                                            pending_service_continuation[
                                                "last_brokered_read_bit_position"
                                            ] = snapshot[
                                                "buffer_read_bit_position"
                                            ]
                                            pending_service_continuation[
                                                "last_brokered_command_bit_position"
                                            ] = snapshot[
                                                "command_bit_position"
                                            ]
                                            pending_service_continuation[
                                                "last_brokered_command_order"
                                            ] = snapshot["command_order"]
                                            pending_service_continuation.pop(
                                                "active_file_write_claim_row_index",
                                                None,
                                            )
                                            pending_service_continuation.pop(
                                                "active_file_write_claim_receipt_index",
                                                None,
                                            )
                                            print(
                                                "service_file_write_payload_completion "
                                                f"tid={event.dwThreadId} "
                                                f"row={completed_claim_row} "
                                                f"update={snapshot['update']} "
                                                "read_bitpos="
                                                f"{snapshot['buffer_read_bit_position']}",
                                                flush=True,
                                            )
                            elif exact_offline_row is not None:
                                exact_index, _ = exact_offline_row
                                exact_row = rows[exact_index]
                                prepared_ahead_of_update = (
                                    exact_index
                                    <= pending_service_continuation[
                                        "end_index"
                                    ]
                                    and snapshot["last_demo_update"]
                                    < int(exact_row["update"])
                                )
                                exact_from_payload = (
                                    exact_index
                                    <= pending_service_continuation[
                                        "end_index"
                                    ]
                                    and snapshot["needs_command"] in (0, 1)
                                    and pending_service_continuation[
                                        "last_reentry_kind"
                                    ]
                                    == 1
                                    and snapshot["command_bit_position"]
                                    == pending_service_continuation[
                                        "last_read_bit_position"
                                    ]
                                )
                                prepared_from_header = (
                                    exact_index
                                    <= pending_service_continuation[
                                        "end_index"
                                    ]
                                    and snapshot["needs_command"] == 1
                                    and pending_service_continuation[
                                        "last_reentry_kind"
                                    ]
                                    == 3
                                    and snapshot["command_bit_position"]
                                    == pending_service_continuation[
                                        "last_command_bit_position"
                                    ]
                                    and snapshot[
                                        "buffer_read_bit_position"
                                    ]
                                    == pending_service_continuation[
                                        "last_read_bit_position"
                                    ]
                                )
                                last_brokered_row_index = (
                                    pending_service_continuation.get(
                                        "last_brokered_row_index"
                                    )
                                )
                                broker_adjacent_file_write_claim = (
                                    isinstance(last_brokered_row_index, int)
                                    and not isinstance(
                                        last_brokered_row_index, bool
                                    )
                                    and exact_index
                                    == last_brokered_row_index + 1
                                    and all(
                                        isinstance(
                                            pending_service_continuation.get(
                                                name
                                            ),
                                            int,
                                        )
                                        and not isinstance(
                                            pending_service_continuation.get(
                                                name
                                            ),
                                            bool,
                                        )
                                        for name in (
                                            "last_brokered_read_bit_position",
                                            "last_brokered_command_order",
                                        )
                                    )
                                    and exact_index
                                    <= pending_service_continuation[
                                        "end_index"
                                    ]
                                    and snapshot["return_address"]
                                    == DEFERRED_FILE_WRITE_CALLER
                                    and snapshot["update"]
                                    == int(exact_row["update"])
                                    and snapshot["last_demo_update"]
                                    == int(exact_row["update"])
                                    and font_cache_manifest_requested
                                    and bool(
                                        pending_service_continuation.get(
                                            "allow_bounded_late_preloading_failed_file_write_corridor",
                                            False,
                                        )
                                    )
                                )
                                if (
                                    prepared_ahead_of_update
                                    or exact_from_payload
                                    or prepared_from_header
                                    or broker_adjacent_file_write_claim
                                ):
                                    ahead_reentry_phase = (
                                        "header_broker_adjacent"
                                        if broker_adjacent_file_write_claim
                                        else (
                                            (
                                                "header"
                                                if prepared_ahead_of_update
                                                else "header_payload"
                                            )
                                            if snapshot["needs_command"] == 0
                                            else (
                                                "prepared"
                                                if prepared_ahead_of_update
                                                else (
                                                    "prepared_header"
                                                    if prepared_from_header
                                                    else "prepared_payload"
                                                )
                                            )
                                        )
                                    )
                                    ahead_reentry_auditor = (
                                        _audited_service_file_write_header_claim
                                        if (
                                            ahead_reentry_phase.startswith(
                                                "header"
                                            )
                                            and snapshot["return_address"]
                                            == DEFERRED_FILE_WRITE_CALLER
                                            and font_cache_manifest_requested
                                        )
                                        else (
                                            _audited_service_header_reentry
                                            if ahead_reentry_phase.startswith(
                                                "header"
                                            )
                                            else _audited_service_prepared_reentry
                                        )
                                    )
                                    service_file_write_header_claim = (
                                        ahead_reentry_auditor
                                        is _audited_service_file_write_header_claim
                                    )
                                    ahead_reentry_kwargs = {
                                        "start_index": (
                                            pending_service_continuation[
                                                "start_index"
                                            ]
                                        ),
                                        "end_index": (
                                            pending_service_continuation[
                                                "end_index"
                                            ]
                                        ),
                                        "previous_read_bit_position": (
                                            pending_service_continuation[
                                                "last_brokered_read_bit_position"
                                                if broker_adjacent_file_write_claim
                                                else "last_read_bit_position"
                                            ]
                                        ),
                                        "previous_command_order": (
                                            pending_service_continuation[
                                                "last_brokered_command_order"
                                                if broker_adjacent_file_write_claim
                                                else "last_command_order"
                                            ]
                                        ),
                                        "command_order_offset": (
                                            command_order_offset
                                        ),
                                    }
                                    if ahead_reentry_phase.startswith(
                                        "header"
                                    ):
                                        ahead_reentry_kwargs[
                                            "previous_reentry_kind"
                                        ] = (
                                            1
                                            if broker_adjacent_file_write_claim
                                            else pending_service_continuation[
                                                "last_reentry_kind"
                                            ]
                                        )
                                        ahead_reentry_kwargs[
                                            "require_before_recorded_update"
                                        ] = (
                                            prepared_ahead_of_update
                                            and not broker_adjacent_file_write_claim
                                        )
                                    else:
                                        ahead_reentry_kwargs[
                                            "previous_command_bit_position"
                                        ] = pending_service_continuation[
                                            "last_command_bit_position"
                                        ]
                                        ahead_reentry_kwargs[
                                            "previous_reentry_kind"
                                        ] = pending_service_continuation[
                                            "last_reentry_kind"
                                        ]
                                        ahead_reentry_kwargs[
                                            "require_before_recorded_update"
                                        ] = prepared_ahead_of_update
                                    if ahead_reentry_auditor in (
                                        _audited_service_header_reentry,
                                        _audited_service_prepared_reentry,
                                    ):
                                        ahead_reentry_kwargs[
                                            "allow_bounded_late_successful_file_write_corridor"
                                        ] = bool(
                                            pending_service_continuation.get(
                                                "allow_bounded_late_successful_file_write_corridor",
                                                False,
                                            )
                                        )
                                        ahead_reentry_kwargs[
                                            "allow_bounded_late_preloading_failed_file_write_corridor"
                                        ] = bool(
                                            pending_service_continuation.get(
                                                "allow_bounded_late_preloading_failed_file_write_corridor",
                                                False,
                                            )
                                        )
                                    (
                                        prepared_row,
                                        service_transient_failure,
                                    ) = ahead_reentry_auditor(
                                        rows,
                                        snapshot,
                                        **ahead_reentry_kwargs,
                                    )
                                    if service_transient_failure is None:
                                        assert prepared_row is not None
                                        if snapshot["command_order"] != (
                                            exact_index
                                            - command_order_offset
                                        ):
                                            (
                                                command_order_offset,
                                                command_order_rebase_rows,
                                                service_transient_failure,
                                            ) = _audited_service_command_order_rebase(
                                                rows_by_start,
                                                rows,
                                                snapshot,
                                                corridor_start_index=(
                                                    pending_service_continuation[
                                                        "start_index"
                                                    ]
                                                ),
                                                corridor_end_index=(
                                                    pending_service_continuation[
                                                        "end_index"
                                                    ]
                                                ),
                                                command_order_offset=(
                                                    command_order_offset
                                                ),
                                                accounted_rows=(
                                                    command_order_rebase_rows
                                                ),
                                                require_last_demo_update=False,
                                            )
                                            if service_transient_failure is None:
                                                print(
                                                    "service_command_order_rebase "
                                                    f"tid={event.dwThreadId} "
                                                    f"update={snapshot['update']} "
                                                    f"phase={ahead_reentry_phase} "
                                                    f"offset={command_order_offset} "
                                                    "accounted_rows="
                                                    + ",".join(
                                                        str(index)
                                                        for index in (
                                                            command_order_rebase_rows
                                                        )
                                                    ),
                                                    flush=True,
                                                )
                                        if (
                                            service_transient_failure is None
                                            and service_file_write_header_claim
                                        ):
                                            assert font_cache_manifest is not None
                                            member = _normalize_font_cache_member(
                                                str(
                                                    snapshot[
                                                        "file_write_font_cache_member"
                                                    ]
                                                )
                                            )
                                            argument_size = int(
                                                snapshot[
                                                    "file_write_argument_size"
                                                ]
                                            )
                                            prior_members = {
                                                str(
                                                    observation[
                                                        "file_write_font_cache_member"
                                                    ]
                                                )
                                                for observation in (
                                                    terminal_file_write_payload_handoffs
                                                    + service_file_write_header_claims
                                                    + direct_font_cache_file_write_observations
                                                    + deferred_file_write_discharges
                                                    + font_cache_manifest_completion_discharges
                                                )
                                            }
                                            if (
                                                member is None
                                                or member in prior_members
                                                or font_cache_manifest.get(
                                                    member
                                                )
                                                != argument_size
                                                or len(
                                                    pre_attach_file_write_debt_rows
                                                )
                                                + len(prior_members)
                                                + 1
                                                > len(font_cache_manifest)
                                                or bool(
                                                    rows[prepared_row][
                                                        "payload"
                                                    ]["success"]
                                                )
                                            ):
                                                service_transient_failure = (
                                                    "service file-write claim "
                                                    "failed manifest identity, "
                                                    "size, uniqueness, result, "
                                                    "or cardinality validation"
                                                )
                                            else:
                                                claim_observation = {
                                                    "process_id": pid,
                                                    "thread_id": int(
                                                        event.dwThreadId
                                                    ),
                                                    "framework_update": snapshot[
                                                        "update"
                                                    ],
                                                    "row_index": prepared_row,
                                                    "row_start": int(
                                                        rows[prepared_row]["start"]
                                                    ),
                                                    "row_end": int(
                                                        rows[prepared_row]["end"]
                                                    ),
                                                    "row_update": int(
                                                        rows[prepared_row]["update"]
                                                    ),
                                                    "recorded_success": False,
                                                    "command_order": snapshot[
                                                        "command_order"
                                                    ],
                                                    "command_order_offset": (
                                                        command_order_offset
                                                    ),
                                                    "file_write_argument_pointer": snapshot[
                                                        "file_write_argument_pointer"
                                                    ],
                                                    "file_write_argument_size": (
                                                        argument_size
                                                    ),
                                                    "file_write_font_cache_member": (
                                                        member
                                                    ),
                                                    "file_write_object_sha256": snapshot[
                                                        "file_write_object_sha256"
                                                    ],
                                                    "manifest_expected_size": (
                                                        font_cache_manifest[
                                                            member
                                                        ]
                                                    ),
                                                    "manifest_entry_count": len(
                                                        font_cache_manifest
                                                    ),
                                                    "natural_payload_consumer": True,
                                                }
                                                if (
                                                    broker_adjacent_file_write_claim
                                                ):
                                                    claim_observation.update(
                                                        {
                                                            "mechanism": "broker_adjacent_current_update_file_write",
                                                            "prior_brokered_row_index": last_brokered_row_index,
                                                            "prior_brokered_read_bit_position": pending_service_continuation[
                                                                "last_brokered_read_bit_position"
                                                            ],
                                                            "prior_brokered_command_order": pending_service_continuation[
                                                                "last_brokered_command_order"
                                                            ],
                                                            "payload_completion_verified": False,
                                                        }
                                                    )
                                                    pending_service_continuation[
                                                        "active_file_write_claim_row_index"
                                                    ] = prepared_row
                                                    pending_service_continuation[
                                                        "active_file_write_claim_receipt_index"
                                                    ] = len(
                                                        service_file_write_header_claims
                                                    )
                                                service_file_write_header_claims.append(
                                                    claim_observation
                                                )
                                                print(
                                                    "service_file_write_header_claim "
                                                    f"tid={event.dwThreadId} "
                                                    f"row={prepared_row} "
                                                    f"update={snapshot['update']} "
                                                    f"member={member} "
                                                    f"size={argument_size}",
                                                    flush=True,
                                                )
                                        if service_transient_failure is not None:
                                            prepared_row = None
                                    if service_transient_failure is None:
                                        assert prepared_row is not None
                                        service_transient_call = True
                                        service_terminal_prepared_call = (
                                            _may_broker_terminal_prepared_service_row(
                                                rows,
                                                snapshot,
                                                prepared_row=prepared_row,
                                                corridor_end_index=(
                                                    pending_service_continuation[
                                                        "end_index"
                                                    ]
                                                ),
                                            )
                                        )
                                        if service_terminal_prepared_call:
                                            service_terminal_prepared_row = (
                                                prepared_row
                                            )
                                        if prepared_row == (
                                            pending_service_continuation[
                                                "end_index"
                                            ]
                                        ):
                                            print(
                                                "service_terminal_prepared_decision "
                                                f"tid={event.dwThreadId} "
                                                f"row={prepared_row} "
                                                "corridor_end="
                                                f"{pending_service_continuation['end_index']} "
                                                f"needs={snapshot['needs_command']} "
                                                f"short={snapshot['is_short']} "
                                                f"num={snapshot['command_number']} "
                                                "cmd_bitpos="
                                                f"{snapshot['command_bit_position']} "
                                                "read_bitpos="
                                                f"{snapshot['buffer_read_bit_position']} "
                                                "eligible="
                                                f"{service_terminal_prepared_call}",
                                                flush=True,
                                            )
                                        pending_service_continuation[
                                            "last_read_bit_position"
                                        ] = snapshot[
                                            "buffer_read_bit_position"
                                        ]
                                        pending_service_continuation[
                                            "last_command_bit_position"
                                        ] = snapshot[
                                            "command_bit_position"
                                        ]
                                        pending_service_continuation[
                                            "last_command_order"
                                        ] = snapshot["command_order"]
                                        pending_service_continuation[
                                            "last_reentry_update"
                                        ] = snapshot["update"]
                                        pending_service_continuation[
                                            "last_reentry_kind"
                                        ] = (
                                            13
                                            if broker_adjacent_file_write_claim
                                            else (
                                                3
                                                if ahead_reentry_phase.startswith(
                                                    "header"
                                                )
                                                else 2
                                            )
                                        )
                                        pending_service_continuation[
                                            "reentries"
                                        ] += 1
                                        print(
                                            f"service_{ahead_reentry_phase}_reentry "
                                            f"tid={event.dwThreadId} "
                                            f"update={snapshot['update']} "
                                            f"row={prepared_row} "
                                            "recorded_update="
                                            f"{exact_row['update']} "
                                            "read_bitpos="
                                            f"{snapshot['buffer_read_bit_position']}",
                                            flush=True,
                                        )
                                elif snapshot["command_order"] != (
                                    exact_index - command_order_offset
                                ):
                                    service_exit_row = (
                                        exact_index
                                        == pending_service_continuation[
                                            "end_index"
                                        ]
                                        + 1
                                    )
                                    (
                                        command_order_offset,
                                        command_order_rebase_rows,
                                        service_transient_failure,
                                    ) = _audited_service_command_order_rebase(
                                        rows_by_start,
                                        rows,
                                        snapshot,
                                        corridor_start_index=(
                                            pending_service_continuation[
                                                "start_index"
                                            ]
                                        ),
                                        corridor_end_index=(
                                            pending_service_continuation[
                                                "end_index"
                                            ]
                                        ),
                                        command_order_offset=(
                                            command_order_offset
                                        ),
                                        accounted_rows=(
                                            command_order_rebase_rows
                                        ),
                                        require_last_demo_update=(
                                            not service_exit_row
                                        ),
                                        allow_exit_row=service_exit_row,
                                    )
                                    if service_transient_failure is None:
                                        print(
                                            "service_command_order_rebase "
                                            f"tid={event.dwThreadId} "
                                            f"update={snapshot['update']} "
                                            "phase="
                                            f"{'exit' if service_exit_row else 'exact'} "
                                            f"offset={command_order_offset} "
                                            "accounted_rows="
                                            + ",".join(
                                                str(index)
                                                for index in (
                                                    command_order_rebase_rows
                                                )
                                            ),
                                            flush=True,
                                        )
                                if (
                                    service_transient_failure is None
                                    and not service_transient_call
                                    and exact_index
                                    > pending_service_continuation["end_index"]
                                ):
                                    exit_boundary_failure = (
                                        _snapshot_boundary_failure(
                                            rows_by_start,
                                            snapshot,
                                            command_order_offset=(
                                                command_order_offset
                                            ),
                                        )
                                    )
                                    if exit_boundary_failure is None:
                                        service_continuation_verifications.append(
                                            {
                                                "process_id": pid,
                                                "thread_id": int(
                                                    event.dwThreadId
                                                ),
                                                "framework_update": snapshot[
                                                    "update"
                                                ],
                                                "row_index": exact_index,
                                                "corridor_start_index": (
                                                    pending_service_continuation[
                                                        "start_index"
                                                    ]
                                                ),
                                                "corridor_end_index": (
                                                    pending_service_continuation[
                                                        "end_index"
                                                    ]
                                                ),
                                                "reentries": (
                                                    pending_service_continuation[
                                                        "reentries"
                                                    ]
                                                ),
                                                "command_order_offset": (
                                                    command_order_offset
                                                ),
                                                "service_exit_timeline_commit_count": len(
                                                    service_exit_timeline_commits
                                                ),
                                            }
                                        )
                                        print(
                                            "service_continuation_verified "
                                            f"tid={event.dwThreadId} "
                                            f"update={snapshot['update']} "
                                            f"row={exact_index} "
                                            "reentries="
                                            f"{pending_service_continuation['reentries']}",
                                            flush=True,
                                        )
                                        pending_service_continuation = None
                                    elif (
                                        exact_index
                                        == pending_service_continuation[
                                            "end_index"
                                        ]
                                        + 1
                                        and pending_service_continuation[
                                            "last_reentry_kind"
                                        ]
                                        == 10
                                    ):
                                        (
                                            exit_row,
                                            service_transient_failure,
                                        ) = _audited_service_exit_overdue_idle_prepared_reentry(
                                            rows,
                                            snapshot,
                                            corridor_start_index=(
                                                pending_service_continuation[
                                                    "start_index"
                                                ]
                                            ),
                                            corridor_end_index=(
                                                pending_service_continuation[
                                                    "end_index"
                                                ]
                                            ),
                                            previous_read_bit_position=(
                                                pending_service_continuation[
                                                    "last_read_bit_position"
                                                ]
                                            ),
                                            previous_command_bit_position=(
                                                pending_service_continuation[
                                                    "last_command_bit_position"
                                                ]
                                            ),
                                            previous_command_order=(
                                                pending_service_continuation[
                                                    "last_command_order"
                                                ]
                                            ),
                                            previous_update=(
                                                pending_service_continuation[
                                                    "last_reentry_update"
                                                ]
                                            ),
                                            previous_reentry_kind=(
                                                pending_service_continuation[
                                                    "last_reentry_kind"
                                                ]
                                            ),
                                            command_order_offset=(
                                                command_order_offset
                                            ),
                                        )
                                        if service_transient_failure is None:
                                            assert exit_row is not None
                                            receipt_index = (
                                                pending_service_continuation.get(
                                                    "overdue_idle_receipt_index"
                                                )
                                            )
                                            receipt = (
                                                service_exit_overdue_idle_reentries[
                                                    receipt_index
                                                ]
                                                if isinstance(
                                                    receipt_index,
                                                    int,
                                                )
                                                and not isinstance(
                                                    receipt_index,
                                                    bool,
                                                )
                                                and 0
                                                <= receipt_index
                                                < len(
                                                    service_exit_overdue_idle_reentries
                                                )
                                                else None
                                            )
                                            if (
                                                not isinstance(receipt, dict)
                                                or receipt.get(
                                                    "bridge_row_index"
                                                )
                                                != exit_row
                                                or receipt.get("process_id")
                                                != pid
                                                or receipt.get("thread_id")
                                                != int(event.dwThreadId)
                                                or receipt.get(
                                                    "framework_update"
                                                )
                                                != snapshot["update"]
                                                or receipt.get(
                                                    "prepared_reentry_verified"
                                                )
                                                is not False
                                            ):
                                                service_transient_failure = (
                                                    "service overdue-idle "
                                                    "prepared receipt identity "
                                                    "mismatch"
                                                )
                                            else:
                                                service_transient_call = True
                                                pending_service_continuation[
                                                    "reentries"
                                                ] += 1
                                                receipt.update(
                                                    {
                                                        "prepared_reentry_verified": True,
                                                        "prepared_framework_update": snapshot[
                                                            "update"
                                                        ],
                                                        "prepared_last_demo_update": snapshot[
                                                            "last_demo_update"
                                                        ],
                                                        "prepared_needs_command": snapshot[
                                                            "needs_command"
                                                        ],
                                                        "prepared_command_order": snapshot[
                                                            "command_order"
                                                        ],
                                                        "prepared_command_bit_position": snapshot[
                                                            "command_bit_position"
                                                        ],
                                                        "prepared_buffer_read_bit_position": snapshot[
                                                            "buffer_read_bit_position"
                                                        ],
                                                        "reentries": pending_service_continuation[
                                                            "reentries"
                                                        ],
                                                    }
                                                )
                                                print(
                                                    "successful_file_write_"
                                                    "overdue_idle_prepared_"
                                                    "reentry_verified "
                                                    f"tid={event.dwThreadId} "
                                                    f"update={snapshot['update']} "
                                                    f"row={exit_row} "
                                                    "read_bitpos="
                                                    f"{snapshot['buffer_read_bit_position']}",
                                                    flush=True,
                                                )
                                                pending_service_continuation = (
                                                    None
                                                )
                                    elif (
                                        exact_index
                                        == pending_service_continuation[
                                            "end_index"
                                        ]
                                        + 1
                                        and pending_service_continuation[
                                            "last_reentry_kind"
                                        ]
                                        == 8
                                    ):
                                        (
                                            exit_row,
                                            service_transient_failure,
                                        ) = _audited_service_exit_late_idle_bridge_prepared_reentry(
                                            rows,
                                            snapshot,
                                            corridor_start_index=(
                                                pending_service_continuation[
                                                    "start_index"
                                                ]
                                            ),
                                            corridor_end_index=(
                                                pending_service_continuation[
                                                    "end_index"
                                                ]
                                            ),
                                            previous_read_bit_position=(
                                                pending_service_continuation[
                                                    "last_read_bit_position"
                                                ]
                                            ),
                                            previous_command_bit_position=(
                                                pending_service_continuation[
                                                    "last_command_bit_position"
                                                ]
                                            ),
                                            previous_command_order=(
                                                pending_service_continuation[
                                                    "last_command_order"
                                                ]
                                            ),
                                            previous_update=(
                                                pending_service_continuation[
                                                    "last_reentry_update"
                                                ]
                                            ),
                                            previous_reentry_kind=(
                                                pending_service_continuation[
                                                    "last_reentry_kind"
                                                ]
                                            ),
                                            command_order_offset=(
                                                command_order_offset
                                            ),
                                        )
                                        if service_transient_failure is None:
                                            assert exit_row is not None
                                            following_row = exit_row + 1
                                            bridge_recorded_update = int(
                                                rows[exit_row]["update"]
                                            )
                                            following_recorded_update = int(
                                                rows[following_row]["update"]
                                            )
                                            (
                                                timeline_rebase_start_row,
                                                timeline_rebase_delta,
                                                service_transient_failure,
                                            ) = _audited_late_idle_native_timeline_rebase(
                                                rows,
                                                corridor_start_index=(
                                                    pending_service_continuation[
                                                        "start_index"
                                                    ]
                                                ),
                                                corridor_end_index=(
                                                    pending_service_continuation[
                                                        "end_index"
                                                    ]
                                                ),
                                                bridge_index=exit_row,
                                                live_bridge_update=snapshot[
                                                    "update"
                                                ],
                                            )
                                        if service_transient_failure is None:
                                            assert exit_row is not None
                                            assert timeline_rebase_start_row is not None
                                            assert timeline_rebase_delta is not None
                                            following_row = exit_row + 1
                                            native_timeline_offset_before = (
                                                native_timeline_offset
                                            )
                                            native_timeline_offset += (
                                                timeline_rebase_delta
                                            )
                                            service_transient_call = True
                                            service_continuation_verifications.append(
                                                {
                                                    "process_id": pid,
                                                    "thread_id": int(
                                                        event.dwThreadId
                                                    ),
                                                    "framework_update": snapshot[
                                                        "update"
                                                    ],
                                                    "row_index": exit_row,
                                                    "corridor_start_index": pending_service_continuation[
                                                        "start_index"
                                                    ],
                                                    "corridor_end_index": pending_service_continuation[
                                                        "end_index"
                                                    ],
                                                    "reentries": pending_service_continuation[
                                                        "reentries"
                                                    ],
                                                    "command_order_offset": command_order_offset,
                                                    "service_exit_timeline_commit_count": len(
                                                        service_exit_timeline_commits
                                                    ),
                                                    "verification_mode": (
                                                        "successful_file_write_late_idle_bridge"
                                                    ),
                                                    "bridge_recorded_update": int(
                                                        bridge_recorded_update
                                                    ),
                                                    "bridge_following_row_index": following_row,
                                                    "bridge_following_recorded_update": int(
                                                        following_recorded_update
                                                    ),
                                                    "bridge_late_updates": (
                                                        snapshot["update"]
                                                        - bridge_recorded_update
                                                    ),
                                                    "native_timeline_rebase_start_row": timeline_rebase_start_row,
                                                    "native_timeline_offset_before": native_timeline_offset_before,
                                                    "native_timeline_offset_delta": timeline_rebase_delta,
                                                    "native_timeline_offset_after": native_timeline_offset,
                                                    "bridge_following_native_update": int(
                                                        rows[following_row][
                                                            "update"
                                                        ]
                                                    ),
                                                }
                                            )
                                            print(
                                                "successful_file_write_late_idle_bridge_prepared "
                                                f"tid={event.dwThreadId} "
                                                f"update={snapshot['update']} "
                                                f"row={exit_row} "
                                                "following_row="
                                                f"{following_row} "
                                                "read_bitpos="
                                                f"{snapshot['buffer_read_bit_position']} "
                                                "timeline_offset="
                                                f"{native_timeline_offset_before}->"
                                                f"{native_timeline_offset}",
                                                flush=True,
                                            )
                                            pending_service_continuation = None
                                    elif (
                                        exact_index
                                        == pending_service_continuation[
                                            "end_index"
                                        ]
                                        + 1
                                        and pending_service_continuation[
                                            "last_reentry_kind"
                                        ]
                                        == 1
                                    ):
                                        prepared_write_exit = (
                                            snapshot["needs_command"] == 1
                                            and bool(
                                                pending_service_continuation.get(
                                                    "allow_bounded_late_successful_file_write_corridor",
                                                    False,
                                                )
                                            )
                                        )
                                        if prepared_write_exit:
                                            (
                                                exit_row,
                                                service_transient_failure,
                                            ) = _audited_successful_file_write_corridor_prepared_exit(
                                                rows,
                                                snapshot,
                                                corridor_start_index=(
                                                    pending_service_continuation[
                                                        "start_index"
                                                    ]
                                                ),
                                                corridor_end_index=(
                                                    pending_service_continuation[
                                                        "end_index"
                                                    ]
                                                ),
                                                previous_read_bit_position=(
                                                    pending_service_continuation[
                                                        "last_read_bit_position"
                                                    ]
                                                ),
                                                previous_reentry_kind=(
                                                    pending_service_continuation[
                                                        "last_reentry_kind"
                                                    ]
                                                ),
                                                command_order_offset=(
                                                    command_order_offset
                                                ),
                                            )
                                            if service_transient_failure is None:
                                                assert exit_row is not None
                                                recorded_update = int(
                                                    rows[exit_row]["update"]
                                                )
                                                live_last_demo_update = _read_i32(
                                                    process,
                                                    snapshot["base"] + 0x61C,
                                                )
                                                duplicate_commit = any(
                                                    observation.get(
                                                        "row_index"
                                                    )
                                                    == exit_row
                                                    for observation in (
                                                        service_exit_timeline_commits
                                                    )
                                                )
                                                if duplicate_commit:
                                                    service_transient_failure = (
                                                        "successful file-write "
                                                        "exit timeline commit "
                                                        "was requested twice"
                                                    )
                                                elif (
                                                    live_last_demo_update
                                                    != snapshot[
                                                        "last_demo_update"
                                                    ]
                                                ):
                                                    service_transient_failure = (
                                                        "successful file-write "
                                                        "exit timeline commit "
                                                        "pre-write value changed: "
                                                        f"snapshot={snapshot['last_demo_update']}, "
                                                        f"live={live_last_demo_update}"
                                                    )
                                                else:
                                                    write_memory(
                                                        process,
                                                        snapshot["base"]
                                                        + 0x61C,
                                                        struct.pack(
                                                            "<i",
                                                            recorded_update,
                                                        ),
                                                    )
                                                    committed_update = _read_i32(
                                                        process,
                                                        snapshot["base"]
                                                        + 0x61C,
                                                    )
                                                    if (
                                                        committed_update
                                                        != recorded_update
                                                    ):
                                                        service_transient_failure = (
                                                            "successful file-write "
                                                            "exit timeline commit "
                                                            "post-write verification failed: "
                                                            f"expected={recorded_update}, "
                                                            f"actual={committed_update}"
                                                        )
                                                    else:
                                                        service_exit_timeline_commits.append(
                                                            {
                                                                "process_id": pid,
                                                                "thread_id": int(
                                                                    event.dwThreadId
                                                                ),
                                                                "framework_update": snapshot[
                                                                    "update"
                                                                ],
                                                                "row_index": exit_row,
                                                                "command_bit_position": snapshot[
                                                                    "command_bit_position"
                                                                ],
                                                                "buffer_read_bit_position": snapshot[
                                                                    "buffer_read_bit_position"
                                                                ],
                                                                "last_demo_update_before": live_last_demo_update,
                                                                "last_demo_update_after": committed_update,
                                                                "recorded_update": recorded_update,
                                                                "timeline_delta_updates": (
                                                                    recorded_update
                                                                    - snapshot[
                                                                        "update"
                                                                    ]
                                                                ),
                                                                "corridor_end_row_index": (
                                                                    pending_service_continuation[
                                                                        "end_index"
                                                                    ]
                                                                ),
                                                                "corridor_end_recorded_update": int(
                                                                    rows[
                                                                        pending_service_continuation[
                                                                            "end_index"
                                                                        ]
                                                                    ]["update"]
                                                                ),
                                                                "bytes_written": 4,
                                                            }
                                                        )
                                                        snapshot[
                                                            "last_demo_update"
                                                        ] = committed_update
                                                        print(
                                                            "successful_file_write_exit_timeline_commit "
                                                            f"tid={event.dwThreadId} "
                                                            f"update={snapshot['update']} "
                                                            f"row={exit_row} "
                                                            f"before={live_last_demo_update} "
                                                            f"after={committed_update} "
                                                            "bytes_written=4",
                                                            flush=True,
                                                        )
                                            if service_transient_failure is None:
                                                assert exit_row is not None
                                                service_transient_call = True
                                                service_continuation_verifications.append(
                                                    {
                                                        "process_id": pid,
                                                        "thread_id": int(
                                                            event.dwThreadId
                                                        ),
                                                        "framework_update": snapshot[
                                                            "update"
                                                        ],
                                                        "row_index": exit_row,
                                                        "corridor_start_index": (
                                                            pending_service_continuation[
                                                                "start_index"
                                                            ]
                                                        ),
                                                        "corridor_end_index": (
                                                            pending_service_continuation[
                                                                "end_index"
                                                            ]
                                                        ),
                                                        "reentries": (
                                                            pending_service_continuation[
                                                                "reentries"
                                                            ]
                                                        ),
                                                        "command_order_offset": (
                                                            command_order_offset
                                                        ),
                                                        "service_exit_timeline_commit_count": len(
                                                            service_exit_timeline_commits
                                                        ),
                                                    }
                                                )
                                                print(
                                                    "successful_file_write_corridor_prepared_exit "
                                                    f"tid={event.dwThreadId} "
                                                    f"update={snapshot['update']} "
                                                    f"row={exit_row} "
                                                    "reentries="
                                                    f"{pending_service_continuation['reentries']} "
                                                    "offset="
                                                    f"{command_order_offset}",
                                                    flush=True,
                                                )
                                                pending_service_continuation = None
                                        else:
                                            corridor_end_index = (
                                                pending_service_continuation[
                                                    "end_index"
                                                ]
                                            )
                                            bridge_index = corridor_end_index + 1
                                            overdue_idle_reentry_candidate = (
                                                allow_post_blackout_overdue_idle_reentry
                                                and snapshot["needs_command"] == 0
                                                and bool(
                                                    pending_service_continuation.get(
                                                        "allow_bounded_late_successful_file_write_corridor",
                                                        False,
                                                    )
                                                )
                                                and bridge_index + 1 < len(rows)
                                                and int(
                                                    rows[bridge_index]["update"]
                                                )
                                                < snapshot["update"]
                                                < int(
                                                    rows[bridge_index + 1][
                                                        "update"
                                                    ]
                                                )
                                            )
                                            late_idle_bridge_candidate = (
                                                snapshot["needs_command"] == 0
                                                and bool(
                                                    pending_service_continuation.get(
                                                        "allow_bounded_late_successful_file_write_corridor",
                                                        False,
                                                    )
                                                )
                                                and bridge_index + 1 < len(rows)
                                                and rows[bridge_index].get(
                                                    "kind"
                                                )
                                                == "idle"
                                                and rows[
                                                    bridge_index + 1
                                                ].get("kind")
                                                == "idle"
                                                and int(
                                                    rows[bridge_index + 1][
                                                        "update"
                                                    ]
                                                )
                                                == snapshot["update"]
                                            )
                                            if overdue_idle_reentry_candidate:
                                                (
                                                    exit_row,
                                                    service_transient_failure,
                                                ) = _audited_service_exit_overdue_idle_reentry(
                                                    rows,
                                                    snapshot,
                                                    corridor_start_index=(
                                                        pending_service_continuation[
                                                            "start_index"
                                                        ]
                                                    ),
                                                    corridor_end_index=(
                                                        corridor_end_index
                                                    ),
                                                    previous_read_bit_position=(
                                                        pending_service_continuation[
                                                            "last_read_bit_position"
                                                        ]
                                                    ),
                                                    previous_reentry_kind=(
                                                        pending_service_continuation[
                                                            "last_reentry_kind"
                                                        ]
                                                    ),
                                                    command_order_offset=(
                                                        command_order_offset
                                                    ),
                                                )
                                                if (
                                                    service_transient_failure
                                                    is None
                                                ):
                                                    assert exit_row is not None
                                                    following_row = exit_row + 1
                                                    corridor_start_index = int(
                                                        pending_service_continuation[
                                                            "start_index"
                                                        ]
                                                    )
                                                    reentries = int(
                                                        pending_service_continuation[
                                                            "reentries"
                                                        ]
                                                    ) + 1
                                                    service_transient_call = True
                                                    overdue_receipt_index = len(
                                                        service_exit_overdue_idle_reentries
                                                    )
                                                    service_exit_overdue_idle_reentries.append(
                                                        {
                                                            "process_id": pid,
                                                            "thread_id": int(
                                                                event.dwThreadId
                                                            ),
                                                            "framework_update": snapshot[
                                                                "update"
                                                            ],
                                                            "last_demo_update": snapshot[
                                                                "last_demo_update"
                                                            ],
                                                            "corridor_start_index": corridor_start_index,
                                                            "corridor_end_index": corridor_end_index,
                                                            "corridor_row_count": (
                                                                corridor_end_index
                                                                - corridor_start_index
                                                                + 1
                                                            ),
                                                            "corridor_recorded_update": int(
                                                                rows[
                                                                    corridor_end_index
                                                                ]["update"]
                                                            ),
                                                            "corridor_end_bit_position": int(
                                                                rows[
                                                                    corridor_end_index
                                                                ]["end"]
                                                            ),
                                                            "bridge_row_index": exit_row,
                                                            "bridge_recorded_update": int(
                                                                rows[exit_row][
                                                                    "update"
                                                                ]
                                                            ),
                                                            "bridge_kind": str(
                                                                rows[exit_row][
                                                                    "kind"
                                                                ]
                                                            ),
                                                            "bridge_command_number": int(
                                                                rows[exit_row][
                                                                    "command_number"
                                                                ]
                                                            ),
                                                            "bridge_short_form": bool(
                                                                rows[exit_row][
                                                                    "short_form"
                                                                ]
                                                            ),
                                                            "bridge_payload_empty": (
                                                                rows[exit_row].get(
                                                                    "payload"
                                                                )
                                                                == {}
                                                            ),
                                                            "bridge_start_bit_position": int(
                                                                rows[exit_row][
                                                                    "start"
                                                                ]
                                                            ),
                                                            "bridge_end_bit_position": int(
                                                                rows[exit_row][
                                                                    "end"
                                                                ]
                                                            ),
                                                            "following_row_index": following_row,
                                                            "following_recorded_update": int(
                                                                rows[
                                                                    following_row
                                                                ]["update"]
                                                            ),
                                                            "following_kind": str(
                                                                rows[
                                                                    following_row
                                                                ]["kind"]
                                                            ),
                                                            "following_command_number": int(
                                                                rows[
                                                                    following_row
                                                                ][
                                                                    "command_number"
                                                                ]
                                                            ),
                                                            "following_short_form": bool(
                                                                rows[
                                                                    following_row
                                                                ]["short_form"]
                                                            ),
                                                            "following_payload_empty": (
                                                                rows[
                                                                    following_row
                                                                ].get("payload")
                                                                == {}
                                                            ),
                                                            "following_start_bit_position": int(
                                                                rows[
                                                                    following_row
                                                                ]["start"]
                                                            ),
                                                            "following_end_bit_position": int(
                                                                rows[
                                                                    following_row
                                                                ]["end"]
                                                            ),
                                                            "bridge_late_updates": (
                                                                snapshot["update"]
                                                                - int(
                                                                    rows[
                                                                        exit_row
                                                                    ]["update"]
                                                                )
                                                            ),
                                                            "following_remaining_updates": (
                                                                int(
                                                                    rows[
                                                                        following_row
                                                                    ]["update"]
                                                                )
                                                                - snapshot[
                                                                    "update"
                                                                ]
                                                            ),
                                                            "command_order_offset": command_order_offset,
                                                            "observed_command_order": snapshot[
                                                                "command_order"
                                                            ],
                                                            "reentries": reentries,
                                                            "prepared_reentry_verified": False,
                                                            "prepared_framework_update": None,
                                                            "prepared_last_demo_update": None,
                                                            "prepared_needs_command": None,
                                                            "prepared_command_order": None,
                                                            "prepared_command_bit_position": None,
                                                            "prepared_buffer_read_bit_position": None,
                                                            "memory_write_bytes": 0,
                                                            "process_memory_mutation": False,
                                                            "verification_mode": (
                                                                "post_blackout_successful_file_write_overdue_idle_reentry"
                                                            ),
                                                        }
                                                    )
                                                    print(
                                                        "successful_file_write_overdue_idle_reentry_verified "
                                                        f"tid={event.dwThreadId} "
                                                        f"update={snapshot['update']} "
                                                        f"row={exit_row} "
                                                        "recorded_update="
                                                        f"{rows[exit_row]['update']} "
                                                        "following_update="
                                                        f"{rows[following_row]['update']} "
                                                        "late_updates="
                                                        f"{snapshot['update'] - int(rows[exit_row]['update'])}",
                                                        flush=True,
                                                    )
                                                    pending_service_continuation[
                                                        "last_read_bit_position"
                                                    ] = snapshot[
                                                        "buffer_read_bit_position"
                                                    ]
                                                    pending_service_continuation[
                                                        "last_command_bit_position"
                                                    ] = snapshot[
                                                        "command_bit_position"
                                                    ]
                                                    pending_service_continuation[
                                                        "last_command_order"
                                                    ] = snapshot[
                                                        "command_order"
                                                    ]
                                                    pending_service_continuation[
                                                        "last_reentry_update"
                                                    ] = snapshot["update"]
                                                    pending_service_continuation[
                                                        "last_reentry_kind"
                                                    ] = 10
                                                    pending_service_continuation[
                                                        "reentries"
                                                    ] = reentries
                                                    pending_service_continuation[
                                                        "overdue_idle_receipt_index"
                                                    ] = overdue_receipt_index
                                            elif late_idle_bridge_candidate:
                                                (
                                                    exit_row,
                                                    service_transient_failure,
                                                ) = _audited_service_exit_late_idle_bridge(
                                                    rows,
                                                    snapshot,
                                                    corridor_start_index=(
                                                        pending_service_continuation[
                                                            "start_index"
                                                        ]
                                                    ),
                                                    corridor_end_index=(
                                                        corridor_end_index
                                                    ),
                                                    previous_read_bit_position=(
                                                        pending_service_continuation[
                                                            "last_read_bit_position"
                                                        ]
                                                    ),
                                                    previous_reentry_kind=(
                                                        pending_service_continuation[
                                                            "last_reentry_kind"
                                                        ]
                                                    ),
                                                    command_order_offset=(
                                                        command_order_offset
                                                    ),
                                                )
                                                if (
                                                    service_transient_failure
                                                    is None
                                                ):
                                                    assert exit_row is not None
                                                    following_row = exit_row + 1
                                                    service_transient_call = True
                                                    pending_service_continuation[
                                                        "last_read_bit_position"
                                                    ] = snapshot[
                                                        "buffer_read_bit_position"
                                                    ]
                                                    pending_service_continuation[
                                                        "last_command_bit_position"
                                                    ] = snapshot[
                                                        "command_bit_position"
                                                    ]
                                                    pending_service_continuation[
                                                        "last_command_order"
                                                    ] = snapshot[
                                                        "command_order"
                                                    ]
                                                    pending_service_continuation[
                                                        "last_reentry_update"
                                                    ] = snapshot["update"]
                                                    pending_service_continuation[
                                                        "last_reentry_kind"
                                                    ] = 8
                                                    pending_service_continuation[
                                                        "reentries"
                                                    ] += 1
                                                    print(
                                                        "successful_file_write_late_idle_bridge_observed "
                                                        f"tid={event.dwThreadId} "
                                                        f"update={snapshot['update']} "
                                                        f"row={exit_row} "
                                                        "recorded_update="
                                                        f"{rows[exit_row]['update']} "
                                                        "following_row="
                                                        f"{following_row} "
                                                        "late_updates="
                                                        f"{snapshot['update'] - int(rows[exit_row]['update'])}",
                                                        flush=True,
                                                    )
                                            else:
                                                (
                                                    exit_row,
                                                    service_transient_failure,
                                                ) = _audited_service_exit_reentry(
                                                    rows,
                                                    snapshot,
                                                    corridor_end_index=(
                                                        corridor_end_index
                                                    ),
                                                    previous_read_bit_position=(
                                                        pending_service_continuation[
                                                            "last_read_bit_position"
                                                        ]
                                                    ),
                                                    previous_reentry_kind=(
                                                        pending_service_continuation[
                                                            "last_reentry_kind"
                                                        ]
                                                    ),
                                                    command_order_offset=(
                                                        command_order_offset
                                                    ),
                                                )
                                                if (
                                                    service_transient_failure
                                                    is None
                                                ):
                                                    assert exit_row is not None
                                                    service_transient_call = True
                                                    pending_service_continuation[
                                                        "last_read_bit_position"
                                                    ] = snapshot[
                                                        "buffer_read_bit_position"
                                                    ]
                                                    pending_service_continuation[
                                                        "last_command_bit_position"
                                                    ] = snapshot[
                                                        "command_bit_position"
                                                    ]
                                                    pending_service_continuation[
                                                        "last_command_order"
                                                    ] = snapshot[
                                                        "command_order"
                                                    ]
                                                    pending_service_continuation[
                                                        "last_reentry_update"
                                                    ] = snapshot["update"]
                                                    pending_service_continuation[
                                                        "last_reentry_kind"
                                                    ] = 4
                                                    pending_service_continuation[
                                                        "reentries"
                                                    ] += 1
                                                    print(
                                                        "service_exit_reentry "
                                                        f"tid={event.dwThreadId} "
                                                        f"update={snapshot['update']} "
                                                        f"row={exit_row} "
                                                        "recorded_update="
                                                        f"{exact_row['update']} "
                                                        "read_bitpos="
                                                        f"{snapshot['buffer_read_bit_position']}",
                                                        flush=True,
                                                    )
                                    elif (
                                        exact_index
                                        == pending_service_continuation[
                                            "end_index"
                                        ]
                                        + 1
                                        and pending_service_continuation[
                                            "last_reentry_kind"
                                        ]
                                        == 4
                                    ):
                                        (
                                            exit_row,
                                            service_transient_failure,
                                        ) = _audited_service_exit_prepared_reentry(
                                            rows,
                                            snapshot,
                                            corridor_end_index=(
                                                pending_service_continuation[
                                                    "end_index"
                                                ]
                                            ),
                                            previous_read_bit_position=(
                                                pending_service_continuation[
                                                    "last_read_bit_position"
                                                ]
                                            ),
                                            previous_command_bit_position=(
                                                pending_service_continuation[
                                                    "last_command_bit_position"
                                                ]
                                            ),
                                            previous_command_order=(
                                                pending_service_continuation[
                                                    "last_command_order"
                                                ]
                                            ),
                                            previous_update=(
                                                pending_service_continuation[
                                                    "last_reentry_update"
                                                ]
                                            ),
                                            previous_reentry_kind=(
                                                pending_service_continuation[
                                                    "last_reentry_kind"
                                                ]
                                            ),
                                            command_order_offset=(
                                                command_order_offset
                                            ),
                                        )
                                        if service_transient_failure is None:
                                            assert exit_row is not None
                                            recorded_update = int(
                                                exact_row["update"]
                                            )
                                            live_last_demo_update = _read_i32(
                                                process,
                                                snapshot["base"] + 0x61C,
                                            )
                                            if any(
                                                observation.get("row_index")
                                                == exit_row
                                                for observation in (
                                                    service_exit_timeline_commits
                                                )
                                            ):
                                                service_transient_failure = (
                                                    "service exit timeline commit "
                                                    "was requested twice for "
                                                    "the same row"
                                                )
                                            elif (
                                                live_last_demo_update
                                                != snapshot["last_demo_update"]
                                            ):
                                                service_transient_failure = (
                                                    "service exit timeline commit "
                                                    "pre-write value changed: "
                                                    f"snapshot={snapshot['last_demo_update']}, "
                                                    f"live={live_last_demo_update}"
                                                )
                                            else:
                                                write_memory(
                                                    process,
                                                    snapshot["base"] + 0x61C,
                                                    struct.pack(
                                                        "<i", recorded_update
                                                    ),
                                                )
                                                committed_update = _read_i32(
                                                    process,
                                                    snapshot["base"] + 0x61C,
                                                )
                                                if (
                                                    committed_update
                                                    != recorded_update
                                                ):
                                                    service_transient_failure = (
                                                        "service exit timeline commit "
                                                        "post-write verification failed: "
                                                        f"expected={recorded_update}, "
                                                        f"actual={committed_update}"
                                                    )
                                                else:
                                                    commit_observation = {
                                                        "process_id": pid,
                                                        "thread_id": int(
                                                            event.dwThreadId
                                                        ),
                                                        "framework_update": snapshot[
                                                            "update"
                                                        ],
                                                        "row_index": exit_row,
                                                        "command_bit_position": snapshot[
                                                            "command_bit_position"
                                                        ],
                                                        "buffer_read_bit_position": snapshot[
                                                            "buffer_read_bit_position"
                                                        ],
                                                        "last_demo_update_before": live_last_demo_update,
                                                        "last_demo_update_after": committed_update,
                                                        "recorded_update": recorded_update,
                                                        "timeline_delta_updates": (
                                                            recorded_update
                                                            - snapshot["update"]
                                                        ),
                                                        "corridor_end_row_index": (
                                                            pending_service_continuation[
                                                                "end_index"
                                                            ]
                                                        ),
                                                        "corridor_end_recorded_update": int(
                                                            rows[
                                                                pending_service_continuation[
                                                                    "end_index"
                                                                ]
                                                            ]["update"]
                                                        ),
                                                        "bytes_written": 4,
                                                    }
                                                    service_exit_timeline_commits.append(
                                                        commit_observation
                                                    )
                                                    snapshot[
                                                        "last_demo_update"
                                                    ] = committed_update
                                                    print(
                                                        "service_exit_timeline_commit "
                                                        f"tid={event.dwThreadId} "
                                                        f"update={snapshot['update']} "
                                                        f"row={exit_row} "
                                                        f"before={live_last_demo_update} "
                                                        f"after={committed_update} "
                                                        "bytes_written=4",
                                                        flush=True,
                                                    )
                                            if service_transient_failure is None:
                                                service_transient_call = True
                                                pending_service_continuation[
                                                    "last_read_bit_position"
                                                ] = snapshot[
                                                    "buffer_read_bit_position"
                                                ]
                                                pending_service_continuation[
                                                    "last_command_bit_position"
                                                ] = snapshot[
                                                    "command_bit_position"
                                                ]
                                                pending_service_continuation[
                                                    "last_command_order"
                                                ] = snapshot["command_order"]
                                                pending_service_continuation[
                                                    "last_reentry_update"
                                                ] = snapshot["update"]
                                                pending_service_continuation[
                                                    "last_reentry_kind"
                                                ] = 7
                                                pending_service_continuation[
                                                    "reentries"
                                                ] += 1
                                                print(
                                                    "service_exit_prepared_reentry "
                                                    f"tid={event.dwThreadId} "
                                                    f"update={snapshot['update']} "
                                                    f"row={exit_row} "
                                                    "recorded_update="
                                                    f"{exact_row['update']} "
                                                    "read_bitpos="
                                                    f"{snapshot['buffer_read_bit_position']}",
                                                    flush=True,
                                                )
                                    else:
                                        service_transient_failure = (
                                            exit_boundary_failure
                                        )
                            else:
                                successful_write_tail_prefetch_candidate = (
                                    pending_service_continuation[
                                        "last_reentry_kind"
                                    ]
                                    == 1
                                    and snapshot["needs_command"] == 1
                                    and snapshot[
                                        "buffer_read_bit_position"
                                    ]
                                    > pending_service_continuation[
                                        "end_bit_position"
                                    ]
                                    and bool(
                                        pending_service_continuation.get(
                                            "allow_bounded_late_successful_file_write_corridor",
                                            False,
                                        )
                                    )
                                )
                                preloading_failed_write_tail_prefetch_candidate = (
                                    pending_service_continuation[
                                        "last_reentry_kind"
                                    ]
                                    in (1, 2)
                                    and snapshot["needs_command"] == 1
                                    and snapshot[
                                        "buffer_read_bit_position"
                                    ]
                                    > pending_service_continuation[
                                        "end_bit_position"
                                    ]
                                    and bool(
                                        pending_service_continuation.get(
                                            "allow_bounded_late_preloading_failed_file_write_corridor",
                                            False,
                                        )
                                    )
                                )
                                preloading_failed_write_tail_short_header_recovery_candidate = (
                                    pending_service_continuation[
                                        "last_reentry_kind"
                                    ]
                                    == 11
                                )
                                partial_exit_candidate = (
                                    pending_service_continuation[
                                        "last_reentry_kind"
                                    ]
                                    == 2
                                    and snapshot["command_bit_position"]
                                    == int(
                                        rows[
                                            pending_service_continuation[
                                                "end_index"
                                            ]
                                        ]["start"]
                                    )
                                    + 10
                                    and snapshot[
                                        "buffer_read_bit_position"
                                    ]
                                    > pending_service_continuation[
                                        "end_bit_position"
                                    ]
                                )
                                forced_exit_candidate = (
                                    pending_service_continuation[
                                        "last_reentry_kind"
                                    ]
                                    in (5, 6)
                                )
                                if successful_write_tail_prefetch_candidate:
                                    (
                                        transient_row,
                                        service_transient_failure,
                                    ) = _audited_successful_file_write_tail_prefetch(
                                        rows,
                                        snapshot,
                                        corridor_start_index=(
                                            pending_service_continuation[
                                                "start_index"
                                            ]
                                        ),
                                        corridor_end_index=(
                                            pending_service_continuation[
                                                "end_index"
                                            ]
                                        ),
                                        previous_read_bit_position=(
                                            pending_service_continuation[
                                                "last_read_bit_position"
                                            ]
                                        ),
                                        previous_command_bit_position=(
                                            pending_service_continuation[
                                                "last_command_bit_position"
                                            ]
                                        ),
                                        previous_command_order=(
                                            pending_service_continuation[
                                                "last_command_order"
                                            ]
                                        ),
                                        previous_update=(
                                            pending_service_continuation[
                                                "last_reentry_update"
                                            ]
                                        ),
                                        previous_reentry_kind=(
                                            pending_service_continuation[
                                                "last_reentry_kind"
                                            ]
                                        ),
                                        command_order_offset=(
                                            command_order_offset
                                        ),
                                    )
                                elif (
                                    preloading_failed_write_tail_prefetch_candidate
                                ):
                                    (
                                        transient_row,
                                        service_transient_failure,
                                    ) = _audited_preloading_failed_file_write_tail_prefetch(
                                        rows,
                                        snapshot,
                                        corridor_start_index=(
                                            pending_service_continuation[
                                                "start_index"
                                            ]
                                        ),
                                        corridor_end_index=(
                                            pending_service_continuation[
                                                "end_index"
                                            ]
                                        ),
                                        previous_read_bit_position=(
                                            pending_service_continuation[
                                                "last_read_bit_position"
                                            ]
                                        ),
                                        previous_command_bit_position=(
                                            pending_service_continuation[
                                                "last_command_bit_position"
                                            ]
                                        ),
                                        previous_command_order=(
                                            pending_service_continuation[
                                                "last_command_order"
                                            ]
                                        ),
                                        previous_update=(
                                            pending_service_continuation[
                                                "last_reentry_update"
                                            ]
                                        ),
                                        previous_reentry_kind=(
                                            pending_service_continuation[
                                                "last_reentry_kind"
                                            ]
                                        ),
                                        command_order_offset=(
                                            command_order_offset
                                        ),
                                        command_order_rebase_rows=(
                                            command_order_rebase_rows
                                        ),
                                    )
                                elif (
                                    preloading_failed_write_tail_short_header_recovery_candidate
                                ):
                                    recovery_state: dict[str, int] | None
                                    (
                                        transient_row,
                                        recovery_state,
                                        service_transient_failure,
                                    ) = _audited_preloading_failed_file_write_tail_short_header_recovery(
                                        rows,
                                        snapshot,
                                        corridor_start_index=(
                                            pending_service_continuation[
                                                "start_index"
                                            ]
                                        ),
                                        corridor_end_index=(
                                            pending_service_continuation[
                                                "end_index"
                                            ]
                                        ),
                                        previous_read_bit_position=(
                                            pending_service_continuation[
                                                "last_read_bit_position"
                                            ]
                                        ),
                                        previous_command_bit_position=(
                                            pending_service_continuation[
                                                "last_command_bit_position"
                                            ]
                                        ),
                                        previous_command_order=(
                                            pending_service_continuation[
                                                "last_command_order"
                                            ]
                                        ),
                                        previous_update=(
                                            pending_service_continuation[
                                                "last_reentry_update"
                                            ]
                                        ),
                                        previous_reentry_kind=(
                                            pending_service_continuation[
                                                "last_reentry_kind"
                                            ]
                                        ),
                                        command_order_offset=(
                                            command_order_offset
                                        ),
                                        command_order_rebase_rows=(
                                            command_order_rebase_rows
                                        ),
                                    )
                                    if service_transient_failure is None:
                                        assert recovery_state is not None
                                        if (
                                            preloading_failed_file_write_tail_short_header_recoveries
                                        ):
                                            service_transient_failure = (
                                                "preloading failed file-write "
                                                "tail short-header recovery "
                                                "was requested twice"
                                            )
                                    if service_transient_failure is None:
                                        before_recovery = {
                                            name: int(snapshot[name])
                                            for name in recovery_state
                                        }
                                        recovery_writes = (
                                            (
                                                "buffer_read_bit_position",
                                                0x608,
                                                "<i",
                                            ),
                                            (
                                                "last_demo_update",
                                                0x61C,
                                                "<i",
                                            ),
                                            ("needs_command", 0x620, "<B"),
                                            ("is_short", 0x621, "<B"),
                                            ("command_number", 0x624, "<i"),
                                            ("command_order", 0x628, "<i"),
                                            (
                                                "command_bit_position",
                                                0x62C,
                                                "<i",
                                            ),
                                            (
                                                "demo_loading_complete",
                                                0x630,
                                                "<B",
                                            ),
                                        )
                                        for name, offset, encoding in (
                                            recovery_writes
                                        ):
                                            write_memory(
                                                process,
                                                snapshot["base"] + offset,
                                                struct.pack(
                                                    encoding,
                                                    recovery_state[name],
                                                ),
                                            )
                                        recovered_snapshot = _command_snapshot(
                                            process,
                                            context,
                                        )
                                        changed_immutable = next(
                                            (
                                                name
                                                for name in (
                                                    "base",
                                                    "return_address",
                                                    "required",
                                                    "vtable",
                                                    "update",
                                                )
                                                if recovered_snapshot[name]
                                                != snapshot[name]
                                            ),
                                            None,
                                        )
                                        changed_recovery = next(
                                            (
                                                name
                                                for name, expected in (
                                                    recovery_state.items()
                                                )
                                                if recovered_snapshot[name]
                                                != expected
                                            ),
                                            None,
                                        )
                                        if changed_immutable is not None:
                                            service_transient_failure = (
                                                "preloading failed file-write "
                                                "tail short-header recovery "
                                                "changed immutable field: "
                                                f"{changed_immutable}"
                                            )
                                        elif changed_recovery is not None:
                                            service_transient_failure = (
                                                "preloading failed file-write "
                                                "tail short-header recovery "
                                                "write verification failed: "
                                                f"{changed_recovery}"
                                            )
                                        else:
                                            recovery_receipt = {
                                                "mechanism": (
                                                    "audited_false_short_header_"
                                                    "pre_payload_recovery"
                                                ),
                                                "process_id": pid,
                                                "thread_id": int(
                                                    event.dwThreadId
                                                ),
                                                "framework_update": snapshot[
                                                    "update"
                                                ],
                                                "false_header_row_index": (
                                                    transient_row
                                                ),
                                                "corridor_start_index": (
                                                    pending_service_continuation[
                                                        "start_index"
                                                    ]
                                                ),
                                                "corridor_end_index": (
                                                    pending_service_continuation[
                                                        "end_index"
                                                    ]
                                                ),
                                                "command_order_offset": (
                                                    command_order_offset
                                                ),
                                                "command_order_rebase_rows": list(
                                                    command_order_rebase_rows
                                                ),
                                                "before": before_recovery,
                                                "after": dict(
                                                    recovery_state
                                                ),
                                                "false_payload_executed": False,
                                                "write_count": len(
                                                    recovery_writes
                                                ),
                                                "bytes_written": sum(
                                                    struct.calcsize(encoding)
                                                    for _, _, encoding in (
                                                        recovery_writes
                                                    )
                                                ),
                                            }
                                            preloading_failed_file_write_tail_short_header_recoveries.append(
                                                recovery_receipt
                                            )
                                            snapshot.update(recovery_state)
                                elif partial_exit_candidate:
                                    (
                                        transient_row,
                                        service_transient_failure,
                                    ) = _audited_service_exit_partial_header(
                                        rows,
                                        snapshot,
                                        corridor_end_index=(
                                            pending_service_continuation[
                                                "end_index"
                                            ]
                                        ),
                                        previous_read_bit_position=(
                                            pending_service_continuation[
                                                "last_read_bit_position"
                                            ]
                                        ),
                                        previous_command_bit_position=(
                                            pending_service_continuation[
                                                "last_command_bit_position"
                                            ]
                                        ),
                                        previous_command_order=(
                                            pending_service_continuation[
                                                "last_command_order"
                                            ]
                                        ),
                                        previous_reentry_kind=(
                                            pending_service_continuation[
                                                "last_reentry_kind"
                                            ]
                                        ),
                                        command_order_offset=(
                                            command_order_offset
                                        ),
                                    )
                                elif forced_exit_candidate:
                                    (
                                        transient_row,
                                        service_transient_failure,
                                    ) = _audited_service_exit_forced_read(
                                        rows,
                                        snapshot,
                                        corridor_end_index=(
                                            pending_service_continuation[
                                                "end_index"
                                            ]
                                        ),
                                        previous_read_bit_position=(
                                            pending_service_continuation[
                                                "last_read_bit_position"
                                            ]
                                        ),
                                        previous_command_bit_position=(
                                            pending_service_continuation[
                                                "last_command_bit_position"
                                            ]
                                        ),
                                        previous_command_order=(
                                            pending_service_continuation[
                                                "last_command_order"
                                            ]
                                        ),
                                        previous_update=(
                                            pending_service_continuation[
                                                "last_reentry_update"
                                            ]
                                        ),
                                        previous_reentry_kind=(
                                            pending_service_continuation[
                                                "last_reentry_kind"
                                            ]
                                        ),
                                        command_order_offset=(
                                            command_order_offset
                                        ),
                                    )
                                elif pending_service_continuation[
                                    "last_reentry_kind"
                                ] in (4, 7, 9, 10, 11, 14):
                                    transient_row = None
                                    service_transient_failure = (
                                        "service exit continuation is unaudited: "
                                        "kind="
                                        f"{pending_service_continuation['last_reentry_kind']}, "
                                        "cmd_bitpos="
                                        f"{snapshot['command_bit_position']}, "
                                        "read_bitpos="
                                        f"{snapshot['buffer_read_bit_position']}, "
                                        f"needs={snapshot['needs_command']}, "
                                        f"short={snapshot['is_short']}, "
                                        f"num={snapshot['command_number']}, "
                                        f"order={snapshot['command_order']}"
                                    )
                                else:
                                    (
                                        transient_row,
                                        service_transient_failure,
                                    ) = _audited_service_payload_reentry(
                                        rows,
                                        snapshot,
                                        start_index=(
                                            pending_service_continuation[
                                                "start_index"
                                            ]
                                        ),
                                        end_index=(
                                            pending_service_continuation[
                                                "end_index"
                                            ]
                                        ),
                                        previous_read_bit_position=(
                                            pending_service_continuation[
                                                "last_read_bit_position"
                                            ]
                                        ),
                                        previous_command_bit_position=(
                                            pending_service_continuation[
                                                "last_command_bit_position"
                                            ]
                                        ),
                                        previous_command_order=(
                                            pending_service_continuation[
                                                "last_command_order"
                                            ]
                                        ),
                                        previous_reentry_kind=(
                                            pending_service_continuation[
                                                "last_reentry_kind"
                                            ]
                                        ),
                                        command_order_offset=(
                                            command_order_offset
                                        ),
                                    )
                                if service_transient_failure is None:
                                    assert transient_row is not None
                                    service_transient_call = True
                                    pending_service_continuation[
                                        "last_read_bit_position"
                                    ] = snapshot[
                                        "buffer_read_bit_position"
                                    ]
                                    pending_service_continuation[
                                        "last_command_bit_position"
                                    ] = snapshot["command_bit_position"]
                                    pending_service_continuation[
                                        "last_command_order"
                                    ] = snapshot["command_order"]
                                    pending_service_continuation[
                                        "last_reentry_update"
                                    ] = snapshot["update"]
                                    pending_service_continuation[
                                        "last_reentry_kind"
                                    ] = (
                                        14
                                        if preloading_failed_write_tail_short_header_recovery_candidate
                                        else (
                                            9
                                            if successful_write_tail_prefetch_candidate
                                            else (
                                                11
                                                if preloading_failed_write_tail_prefetch_candidate
                                                else (
                                                    5
                                                    if partial_exit_candidate
                                                    else (
                                                        6
                                                        if forced_exit_candidate
                                                        else 1
                                                    )
                                                )
                                            )
                                        )
                                    )
                                    pending_service_continuation[
                                        "reentries"
                                    ] += 1
                                    reentry_label = (
                                        "preloading_failed_file_write_tail_short_header_recovery "
                                        if preloading_failed_write_tail_short_header_recovery_candidate
                                        else (
                                            "successful_file_write_tail_prefetch "
                                            if successful_write_tail_prefetch_candidate
                                            else (
                                                "preloading_failed_file_write_tail_prefetch "
                                                if preloading_failed_write_tail_prefetch_candidate
                                                else (
                                                    "service_exit_partial_header "
                                                    if partial_exit_candidate
                                                    else (
                                                        "service_exit_forced_read "
                                                        if forced_exit_candidate
                                                        else "service_payload_reentry "
                                                    )
                                                )
                                            )
                                        )
                                    )
                                    print(
                                        f"{reentry_label}tid={event.dwThreadId} "
                                        f"update={snapshot['update']} "
                                        f"row={transient_row} "
                                        "cmd_bitpos="
                                        f"{snapshot['command_bit_position']} "
                                        "read_bitpos="
                                        f"{snapshot['buffer_read_bit_position']}",
                                        flush=True,
                                    )
                        orphan_prepared_block = False
                        service_block = (
                            _service_block(
                                rows_by_start,
                                rows,
                                snapshot,
                            )
                            if broker_service_blocks
                            and not untracked_command_call
                            and deferred_file_write_failure is None
                            and (
                                not service_transient_call
                                or service_terminal_prepared_call
                            )
                            else None
                        )
                        if startup_worker_echo_service_block is not None:
                            if service_block is not None:
                                service_transient_failure = (
                                    "startup worker echo produced two "
                                    "competing service blocks"
                                )
                                service_block = None
                                service_transient_call = True
                            else:
                                service_block = (
                                    startup_worker_echo_service_block
                                )
                        if (
                            service_block is None
                            and broker_service_blocks
                            and not untracked_command_call
                            and (
                                not service_transient_call
                                or service_terminal_prepared_call
                            )
                            and snapshot["needs_command"] == 1
                        ):
                            prepared_block = (
                                (
                                    service_terminal_prepared_row,
                                    service_terminal_prepared_row,
                                    int(
                                        rows[
                                            service_terminal_prepared_row
                                        ]["end"]
                                    ),
                                )
                                if (
                                    service_terminal_prepared_call
                                    and service_terminal_prepared_row
                                    is not None
                                )
                                else _service_block(
                                    rows_by_start,
                                    rows,
                                    snapshot,
                                    allow_prepared_command=True,
                                )
                            )
                            if prepared_block is not None:
                                prepared_start, _, prepared_end = (
                                    prepared_block
                                )
                                audited_debt_barrier = False
                                if (
                                    diagnostic_successful_file_write_padding_rows
                                    and deferred_file_write_discharges
                                    and not service_consumed_file_write_surplus_rows
                                ):
                                    (
                                        candidate_surplus_rows,
                                        debt_barrier_failure,
                                    ) = _audited_successful_file_write_debt_barrier(
                                        rows,
                                        snapshot,
                                        command_order_offset=(
                                            command_order_offset
                                        ),
                                        accounted_rows=(
                                            command_order_rebase_rows
                                        ),
                                        excluded_accounted_rows=tuple(
                                            row_index
                                            for row_index in (
                                                diagnostic_successful_file_write_padding_rows
                                            )
                                            if row_index
                                            in command_order_rebase_rows
                                        ),
                                        service_consumed_rows=(
                                            service_consumed_file_write_debt_rows
                                        ),
                                        discharged_rows=tuple(
                                            int(
                                                observation[
                                                    "debt_row_index"
                                                ]
                                            )
                                            for observation in (
                                                deferred_file_write_discharges
                                            )
                                        ),
                                        diagnostic_padding_rows=(
                                            diagnostic_successful_file_write_padding_rows
                                        ),
                                    )
                                    if debt_barrier_failure is None:
                                        service_consumed_file_write_surplus_rows = (
                                            candidate_surplus_rows
                                        )
                                        audited_debt_barrier = True
                                        successful_file_write_debt_barriers.append(
                                            {
                                                "process_id": pid,
                                                "thread_id": int(
                                                    event.dwThreadId
                                                ),
                                                "framework_update": snapshot[
                                                    "update"
                                                ],
                                                "barrier_row_index": (
                                                    prepared_start
                                                ),
                                                "barrier_command_number": int(
                                                    rows[prepared_start][
                                                        "command_number"
                                                    ]
                                                ),
                                                "barrier_start_bit_position": int(
                                                    rows[prepared_start][
                                                        "start"
                                                    ]
                                                ),
                                                "barrier_read_bit_position": snapshot[
                                                    "buffer_read_bit_position"
                                                ],
                                                "discharged_row_count": len(
                                                    deferred_file_write_discharges
                                                ),
                                                "surplus_rows": list(
                                                    candidate_surplus_rows
                                                ),
                                                "mechanism": (
                                                    "later_non_file_write_prepared_service_barrier"
                                                ),
                                            }
                                        )
                                prepared_key = (
                                    snapshot["base"],
                                    int(rows[prepared_start]["start"]),
                                    prepared_end,
                                )
                                prepared_was_bypassed = (
                                    prepared_key in bypassed_service_blocks
                                )
                                if (
                                    service_terminal_prepared_call
                                    or audited_debt_barrier
                                    or _may_adopt_prepared_service_block(
                                        already_bypassed=(
                                            prepared_was_bypassed
                                        ),
                                        allow_orphan=(
                                            allow_orphan_prepared_service_block
                                        ),
                                        orphan_already_used=(
                                            orphan_prepared_service_block_used
                                        ),
                                    )
                                ):
                                    service_block = prepared_block
                                if (
                                    service_block is not None
                                    and not prepared_was_bypassed
                                    and not service_terminal_prepared_call
                                ):
                                    orphan_prepared_block = True
                                    if audited_debt_barrier:
                                        print(
                                            "successful_file_write_debt_barrier "
                                            f"tid={event.dwThreadId} "
                                            f"update={snapshot['update']} "
                                            f"row={prepared_start} "
                                            "surplus_rows="
                                            f"{service_consumed_file_write_surplus_rows} "
                                            f"target_end_bitpos={prepared_end}",
                                            flush=True,
                                        )
                                    else:
                                        orphan_prepared_service_block_used = True
                                        print(
                                            "service_broker_reattach_prepared "
                                            f"tid={event.dwThreadId} "
                                            f"update={snapshot['update']} "
                                            "start_bitpos="
                                            f"{rows[prepared_start]['start']} "
                                            f"target_end_bitpos={prepared_end}",
                                            flush=True,
                                        )
                                if service_terminal_prepared_call:
                                    print(
                                        "service_broker_terminal_prepared "
                                        f"tid={event.dwThreadId} "
                                        f"update={snapshot['update']} "
                                        f"row={prepared_start} "
                                        "read_bitpos="
                                        f"{snapshot['buffer_read_bit_position']} "
                                        "target_end_bitpos="
                                        f"{prepared_end}",
                                        flush=True,
                                    )
                        boundary_failure: str | None = (
                            attach_stabilization_failure
                            or service_transient_failure
                            or deferred_file_write_failure
                        )
                        if (
                            rows
                            and not untracked_command_call
                            and not service_transient_call
                        ):
                            if (
                                allow_initial_file_write_order_rebase
                                and not initial_offline_boundary_seen
                            ):
                                assert (
                                    command_order_rebase_after_update
                                    is not None
                                )
                                (
                                    command_order_offset,
                                    command_order_rebase_rows,
                                    boundary_failure,
                                ) = (
                                    _audited_file_write_command_order_rebase(
                                        rows_by_start,
                                        rows,
                                        snapshot,
                                        detached_after_update=(
                                            command_order_rebase_after_update
                                        ),
                                    )
                                )
                                if boundary_failure is None:
                                    print(
                                        "command_order_rebase "
                                        f"tid={event.dwThreadId} "
                                        f"update={snapshot['update']} "
                                        "live_order="
                                        f"{snapshot['command_order']} "
                                        f"offset={command_order_offset} "
                                        "offline_order="
                                        f"{snapshot['command_order'] + command_order_offset} "
                                        "accounted_rows="
                                        + ",".join(
                                            str(index)
                                            for index in (
                                                command_order_rebase_rows
                                            )
                                        ),
                                        flush=True,
                                    )
                            if boundary_failure is None:
                                matched_boundary = rows_by_start.get(
                                    snapshot["command_bit_position"]
                                )
                                latest_commit = (
                                    service_exit_timeline_commits[-1]
                                    if service_exit_timeline_commits
                                    else None
                                )
                                clamped_suffix_candidate = (
                                    latest_commit is not None
                                    and not service_exit_timeline_suffix_rebases
                                    and latest_commit.get(
                                        "late_commit_clamp_updates"
                                    )
                                    == 1
                                    and matched_boundary is not None
                                    and matched_boundary[0]
                                    == latest_commit.get("row_index", -2) + 1
                                )
                                if clamped_suffix_candidate:
                                    (
                                        suffix_start_row,
                                        suffix_delta,
                                        suffix_failure,
                                    ) = _audited_successful_file_write_deferred_exit_clamped_suffix_rebase(
                                        rows,
                                        snapshot,
                                        latest_commit,
                                        process_id=pid,
                                        thread_id=int(event.dwThreadId),
                                        command_order_offset=(
                                            command_order_offset
                                        ),
                                    )
                                    if suffix_failure is not None:
                                        boundary_failure = suffix_failure
                                    else:
                                        assert suffix_start_row is not None
                                        assert suffix_delta is not None
                                        timeline_offset_before = (
                                            native_timeline_offset
                                        )
                                        native_timeline_offset += suffix_delta
                                        service_exit_timeline_suffix_rebases.append(
                                            {
                                                "process_id": pid,
                                                "thread_id": int(
                                                    event.dwThreadId
                                                ),
                                                "framework_update": snapshot[
                                                    "update"
                                                ],
                                                "commit_row_index": latest_commit[
                                                    "row_index"
                                                ],
                                                "suffix_start_row_index": (
                                                    suffix_start_row
                                                ),
                                                "timeline_offset_before": (
                                                    timeline_offset_before
                                                ),
                                                "timeline_offset_delta": (
                                                    suffix_delta
                                                ),
                                                "timeline_offset_after": (
                                                    native_timeline_offset
                                                ),
                                                "mechanism": (
                                                    "one_tick_clamped_deferred_exit_suffix"
                                                ),
                                            }
                                        )
                                        print(
                                            "successful_file_write_deferred_exit_"
                                            "suffix_rebase "
                                            f"tid={event.dwThreadId} "
                                            f"update={snapshot['update']} "
                                            f"start_row={suffix_start_row} "
                                            "timeline_offset="
                                            f"{timeline_offset_before}->"
                                            f"{native_timeline_offset}",
                                            flush=True,
                                        )
                            if boundary_failure is None:
                                boundary_snapshot = (
                                    _normalize_blackout_native_timeline_snapshot(
                                        snapshot,
                                        native_timeline_offset=(
                                            blackout_native_timeline_offset
                                        ),
                                    )
                                    if blackout_native_timeline_offset
                                    else snapshot
                                )
                                committed_boundary_normalized = False
                                if service_exit_timeline_commits:
                                    (
                                        boundary_snapshot,
                                        committed_boundary_normalized,
                                    ) = _audited_successful_file_write_deferred_exit_committed_boundary_snapshot(
                                        rows_by_start,
                                        snapshot,
                                        service_exit_timeline_commits[-1],
                                        process_id=pid,
                                        thread_id=int(event.dwThreadId),
                                    )
                                boundary_failure = (
                                    _snapshot_boundary_failure(
                                        rows_by_start,
                                        boundary_snapshot,
                                        command_order_offset=(
                                            command_order_offset
                                        ),
                                    )
                                )
                                if (
                                    boundary_failure is None
                                    and committed_boundary_normalized
                                ):
                                    print(
                                        "successful_file_write_deferred_exit_"
                                        "committed_boundary "
                                        f"tid={event.dwThreadId} "
                                        f"update={snapshot['update']} "
                                        "row="
                                        f"{service_exit_timeline_commits[-1]['row_index']} "
                                        "last="
                                        f"{snapshot['last_demo_update']}->"
                                        f"{boundary_snapshot['last_demo_update']}",
                                        flush=True,
                                    )
                            initial_offline_boundary_seen = True
                        if (
                            boundary_failure is None
                            and font_cache_manifest_requested
                            and direct_font_cache_row_index is not None
                            and direct_font_cache_row_index
                            in startup_file_write_rows
                            and not untracked_command_call
                            and not service_transient_call
                        ):
                            assert font_cache_manifest is not None
                            assert direct_font_cache_recorded_success is not None
                            normalized_member = _normalize_font_cache_member(
                                str(
                                    snapshot[
                                        "file_write_font_cache_member"
                                    ]
                                )
                            )
                            argument_size = int(
                                snapshot["file_write_argument_size"]
                            )
                            prior_members = {
                                str(
                                    observation[
                                        "file_write_font_cache_member"
                                    ]
                                )
                                for observation in (
                                    terminal_file_write_payload_handoffs
                                    + service_file_write_header_claims
                                    + direct_font_cache_file_write_observations
                                    + deferred_file_write_discharges
                                    + font_cache_manifest_completion_discharges
                                )
                            }
                            if (
                                normalized_member is None
                                or normalized_member in prior_members
                                or font_cache_manifest.get(normalized_member)
                                != argument_size
                                or direct_font_cache_recorded_success
                                or len(pre_attach_file_write_debt_rows)
                                + len(prior_members)
                                + 1
                                > len(font_cache_manifest)
                            ):
                                boundary_failure = (
                                    "direct font-cache file-write observation "
                                    "failed manifest identity, size, "
                                    "uniqueness, result, or cardinality "
                                    "validation"
                                )
                            else:
                                direct_row = rows[
                                    direct_font_cache_row_index
                                ]
                                direct_observation = {
                                    "process_id": pid,
                                    "thread_id": int(event.dwThreadId),
                                    "framework_update": snapshot["update"],
                                    "row_index": direct_font_cache_row_index,
                                    "row_start": int(direct_row["start"]),
                                    "row_end": int(direct_row["end"]),
                                    "row_update": int(direct_row["update"]),
                                    "recorded_success": (
                                        direct_font_cache_recorded_success
                                    ),
                                    "command_order": snapshot[
                                        "command_order"
                                    ],
                                    "command_order_offset": (
                                        command_order_offset
                                    ),
                                    "command_bit_position": snapshot[
                                        "command_bit_position"
                                    ],
                                    "buffer_read_bit_position": snapshot[
                                        "buffer_read_bit_position"
                                    ],
                                    "needs_command": snapshot[
                                        "needs_command"
                                    ],
                                    "file_write_argument_pointer": snapshot[
                                        "file_write_argument_pointer"
                                    ],
                                    "file_write_argument_size": argument_size,
                                    "file_write_object_address": snapshot[
                                        "file_write_object_address"
                                    ],
                                    "file_write_object_length": snapshot[
                                        "file_write_object_length"
                                    ],
                                    "file_write_object_capacity": snapshot[
                                        "file_write_object_capacity"
                                    ],
                                    "file_write_object_data_address": snapshot[
                                        "file_write_object_data_address"
                                    ],
                                    "file_write_object_sha256": snapshot[
                                        "file_write_object_sha256"
                                    ],
                                    "file_write_object_prefix_hex": snapshot[
                                        "file_write_object_prefix_hex"
                                    ],
                                    "file_write_object_full_hex": snapshot[
                                        "file_write_object_full_hex"
                                    ],
                                    "file_write_object_full_truncated": snapshot[
                                        "file_write_object_full_truncated"
                                    ],
                                    "file_write_font_cache_member": (
                                        normalized_member
                                    ),
                                    "manifest_expected_size": (
                                        font_cache_manifest[normalized_member]
                                    ),
                                }
                                direct_font_cache_file_write_observations.append(
                                    direct_observation
                                )
                                print(
                                    "direct_font_cache_file_write_observation "
                                    f"tid={event.dwThreadId} "
                                    f"update={snapshot['update']} "
                                    f"row={direct_font_cache_row_index} "
                                    f"member={normalized_member} "
                                    f"size={argument_size}",
                                    flush=True,
                                )
                        if pre_stream_call:
                            print(
                                "pre_stream_command "
                                f"tid={event.dwThreadId} "
                                f"update={snapshot['update']} "
                                "command_order="
                                f"{snapshot['command_order']}",
                                flush=True,
                            )
                        if (
                            snapshot["return_address"]
                            == DEFERRED_FILE_WRITE_CALLER
                        ):
                            print(
                                "file_write_call_arguments "
                                f"hit={hits} tid={event.dwThreadId} "
                                "pointer=0x"
                                f"{snapshot['file_write_argument_pointer']:08X} "
                                "size="
                                f"{snapshot['file_write_argument_size']} "
                                "object_length="
                                f"{snapshot['file_write_object_length']} "
                                "object_sha256="
                                f"{snapshot['file_write_object_sha256']} "
                                "object_prefix="
                                f"{snapshot['file_write_object_prefix_hex']} "
                                "object_full="
                                f"{snapshot['file_write_object_full_hex']} "
                                "object_full_truncated="
                                f"{int(snapshot['file_write_object_full_truncated'])}",
                                flush=True,
                            )
                        if boundary_failure is not None:
                            boundary_failures.append(boundary_failure)
                            print(
                                "offline_boundary_failure "
                                f"tid={event.dwThreadId} "
                                f"update={snapshot['update']} "
                                f"cmd_bitpos="
                                f"{snapshot['command_bit_position']} "
                                f"detail={boundary_failure}",
                                flush=True,
                            )
                        if (
                            not untracked_command_call
                            and not service_transient_call
                            and
                            global_mtrand_restore_command_order is not None
                            and snapshot["command_order"]
                            == global_mtrand_restore_command_order
                        ):
                            if boundary_failure is not None:
                                raise RuntimeError(
                                    "global MTRand restore boundary is invalid"
                                )
                            if not global_mtrand_observations:
                                assert (
                                    global_mtrand_restore_state is not None
                                )
                                assert (
                                    global_mtrand_expected_before_state
                                    is not None
                                )
                                mtrand_observation = (
                                    _apply_global_mtrand_restore(
                                        process=process,
                                        process_id=pid,
                                        thread_id=int(event.dwThreadId),
                                        snapshot=snapshot,
                                        desired=(
                                            global_mtrand_restore_state
                                        ),
                                        expected_before=(
                                            global_mtrand_expected_before_state
                                        ),
                                        command_order=(
                                            global_mtrand_restore_command_order
                                        ),
                                        allow_dynamic_before=(
                                            global_mtrand_allow_dynamic_before
                                        ),
                                    )
                                )
                                global_mtrand_observations.append(
                                    mtrand_observation
                                )
                                print(
                                    "global_mtrand_restore "
                                    f"tid={mtrand_observation['thread_id']} "
                                    "update="
                                    f"{mtrand_observation['framework_update']} "
                                    "order="
                                    f"{mtrand_observation['command_order']} "
                                    "state_address=0x"
                                    f"{mtrand_observation['state_address']:08X} "
                                    "bytes="
                                    f"{mtrand_observation['bytes_written']} "
                                    "before="
                                    f"{mtrand_observation['live_before_state_sha256']} "
                                    "after="
                                    f"{mtrand_observation['restored']['state_sha256']}",
                                    flush=True,
                                )
                        elif (
                            not untracked_command_call
                            and not service_transient_call
                            and
                            global_mtrand_restore_command_order is not None
                            and not global_mtrand_observations
                            and snapshot["command_order"]
                            > global_mtrand_restore_command_order
                        ):
                            raise RuntimeError(
                                "global MTRand restore command boundary was "
                                "skipped"
                            )
                        if (
                            not untracked_command_call
                            and not service_transient_call
                            and
                            thread_crt_restore_command_order is not None
                            and snapshot["command_order"]
                            == thread_crt_restore_command_order
                        ):
                            if boundary_failure is not None:
                                raise RuntimeError(
                                    "thread CRT restore boundary is invalid"
                                )
                            if not thread_crt_observations:
                                assert thread_crt_restore_state is not None
                                assert (
                                    thread_crt_expected_before_state
                                    is not None
                                )
                                crt_observation = (
                                    _apply_thread_crt_restore(
                                        process=process,
                                        process_id=pid,
                                        thread=thread,
                                        thread_id=int(event.dwThreadId),
                                        context=context,
                                        snapshot=snapshot,
                                        desired=thread_crt_restore_state,
                                        expected_before=(
                                            thread_crt_expected_before_state
                                        ),
                                        command_order=(
                                            thread_crt_restore_command_order
                                        ),
                                        allow_dynamic_before=(
                                            thread_crt_allow_dynamic_before
                                        ),
                                    )
                                )
                                thread_crt_observations.append(
                                    crt_observation
                                )
                                print(
                                    "thread_crt_restore "
                                    f"tid={crt_observation['thread_id']} "
                                    "update="
                                    f"{crt_observation['framework_update']} "
                                    "order="
                                    f"{crt_observation['command_order']} "
                                    "state_address=0x"
                                    f"{crt_observation['state_address']:08X} "
                                    "bytes="
                                    f"{crt_observation['bytes_written']} "
                                    "before="
                                    f"{crt_observation['expected_before']['state_hex']} "
                                    "after="
                                    f"{crt_observation['restored']['state_hex']}",
                                    flush=True,
                                )
                        elif (
                            not untracked_command_call
                            and not service_transient_call
                            and
                            thread_crt_restore_command_order is not None
                            and not thread_crt_observations
                            and snapshot["command_order"]
                            > thread_crt_restore_command_order
                        ):
                            raise RuntimeError(
                                "thread CRT restore command boundary was "
                                "skipped"
                            )
                        if (
                            not untracked_command_call
                            and not service_transient_call
                            and
                            qrand_restore_command_order is not None
                            and snapshot["command_order"]
                            == qrand_restore_command_order
                        ):
                            if boundary_failure is not None:
                                raise RuntimeError(
                                    "QRand restore boundary is invalid"
                                )
                            # PrepareDemoCommand may re-enter for the same
                            # command while its payload is being consumed.
                            # Restore only on the first exact boundary hit;
                            # later hits are normal command re-entry.
                            if not qrand_observations:
                                assert qrand_restore_state is not None
                                assert qrand_expected_before_state is not None
                                observation = _apply_qrand_restore(
                                    process=process,
                                    process_id=pid,
                                    thread_id=int(event.dwThreadId),
                                    snapshot=snapshot,
                                    desired=qrand_restore_state,
                                    expected_before=(
                                        qrand_expected_before_state
                                    ),
                                    command_order=(
                                        qrand_restore_command_order
                                    ),
                                )
                                qrand_observations.append(observation)
                                print(
                                    "qrand_restore "
                                    f"tid={observation['thread_id']} "
                                    f"update={observation['framework_update']} "
                                    f"order={observation['command_order']} "
                                    f"object=0x"
                                    f"{observation['qrand_address']:08X} "
                                    f"bytes={observation['bytes_written']} "
                                    "before="
                                    f"{observation['expected_before']['semantic_sha256']} "
                                    "after="
                                    f"{observation['restored']['semantic_sha256']}",
                                    flush=True,
                                )
                        elif (
                            not untracked_command_call
                            and not service_transient_call
                            and
                            qrand_restore_command_order is not None
                            and not qrand_observations
                            and snapshot["command_order"]
                            > qrand_restore_command_order
                        ):
                            raise RuntimeError(
                                "QRand restore command boundary was skipped"
                            )
                        stack_candidates = _stack_code_candidates(
                            process, context.Esp
                        )
                        hits += 1
                        if (
                            not quiet_nonservice
                            or snapshot["command_number"]
                            in SERVICE_COMMAND_NUMBERS
                            or service_transient_call
                            or attach_stabilization_call
                            or boundary_failure is not None
                        ):
                            print(
                                "command_entry "
                                f"hit={hits} tid={event.dwThreadId} "
                                f"caller=0x{snapshot['return_address']:08X} "
                                f"required={snapshot['required']} "
                                f"base=0x{snapshot['base']:08X} "
                                f"vtable=0x{snapshot['vtable']:08X} "
                                f"update={snapshot['update']} "
                                "read_bitpos="
                                f"{snapshot['buffer_read_bit_position']} "
                                f"last_update={snapshot['last_demo_update']} "
                                f"needs={snapshot['needs_command']} "
                                f"short={snapshot['is_short']} "
                                f"num={snapshot['command_number']} "
                                f"order={snapshot['command_order']} "
                                "cmd_bitpos="
                                f"{snapshot['command_bit_position']} "
                                f"loading_complete="
                                f"{snapshot['demo_loading_complete']} "
                                "stack_code="
                                + ",".join(
                                    f"+0x{offset:02X}:0x{value:08X}"
                                    for offset, value in stack_candidates
                                ),
                                flush=True,
                            )
                        write_memory(process, address, original)
                        breakpoint_armed = False
                        kernel32.FlushInstructionCache(
                            process, ctypes.c_void_p(address), 1
                        )
                        context.Eip = address
                        context.EFlags |= TRAP_FLAG
                        if not kernel32.Wow64SetThreadContext(
                            thread, ctypes.byref(context)
                        ):
                            raise ctypes.WinError(
                                ctypes.get_last_error()
                            )
                        if service_block is None:
                            if stepping_suspended_threads:
                                raise RuntimeError(
                                    "serialized single-step state leaked"
                                )
                            stepping_suspended_threads = (
                                _suspend_other_threads(
                                    pid,
                                    excluded_thread_id=int(
                                        event.dwThreadId
                                    ),
                                    access_denied_skip_events=(
                                        access_denied_thread_skip_events
                                    ),
                                )
                            )
                            stepping_thread = int(event.dwThreadId)
                            stepping_kind = "command"
                        else:
                            (
                                start_index,
                                end_index,
                                target_end,
                            ) = service_block
                            service_start = int(
                                rows[start_index]["start"]
                            )
                            service_update = int(
                                rows[start_index]["update"]
                            )
                            settle_end = _service_settle_end(
                                rows,
                                end_index,
                                target_end,
                            )
                            previous_suspend_count = int(
                                kernel32.SuspendThread(thread)
                            )
                            if (
                                previous_suspend_count
                                == INVALID_SUSPEND_COUNT
                            ):
                                raise ctypes.WinError(
                                    ctypes.get_last_error()
                                )
                            held_thread = int(thread)
                            close_thread = False
                            pending_broker = {
                                "thread_id": int(event.dwThreadId),
                                "base": snapshot["base"],
                                "start_index": start_index,
                                "end_index": end_index,
                                "start_bit_position": service_start,
                                "target_end_bit_position": target_end,
                                "settle_end_bit_position": settle_end,
                                "initial_read_bit_position": snapshot[
                                    "buffer_read_bit_position"
                                ],
                                "initial_command_bit_position": snapshot[
                                    "command_bit_position"
                                ],
                                "initial_command_order": snapshot[
                                    "command_order"
                                ],
                                "return_address": snapshot[
                                    "return_address"
                                ],
                                "needs_command": snapshot[
                                    "needs_command"
                                ],
                                "demo_loading_complete": snapshot[
                                    "demo_loading_complete"
                                ],
                                "orphan_prepared_block": (
                                    orphan_prepared_block
                                ),
                                "allow_intermediate_prepared": (
                                    not service_terminal_prepared_call
                                ),
                                "event_driven_terminal": (
                                    service_terminal_prepared_call
                                ),
                                "event_driven_ready": False,
                                "update": service_update,
                                "previous_suspend_count": (
                                    previous_suspend_count
                                ),
                            }
                            print(
                                "service_broker_hold "
                                f"tid={event.dwThreadId} "
                                f"update={service_update} "
                                f"rows={start_index}:{end_index} "
                                f"start_bitpos={service_start} "
                                f"target_end_bitpos={target_end} "
                                f"settle_end_bitpos={settle_end}",
                                flush=True,
                            )
                            if service_terminal_prepared_call:
                                write_memory(process, address, b"\xCC")
                                breakpoint_armed = True
                                kernel32.FlushInstructionCache(
                                    process,
                                    ctypes.c_void_p(address),
                                    1,
                                )
                                print(
                                    "service_terminal_handoff_armed "
                                    f"held_tid={event.dwThreadId} "
                                    f"address=0x{address:08X}",
                                    flush=True,
                                )
                        stop_after_step = hits >= maximum_hits
                        if boundary_failure is not None:
                            stop_after_step = True
                        if (
                            stop_after_update is not None
                            and not untracked_command_call
                            and snapshot["update"] >= stop_after_update
                        ):
                            stop_after_step = True
                            stopped_at_update = True
                        if (
                            not untracked_command_call
                            and not service_transient_call
                            and
                            stop_after_command_order is not None
                            and snapshot["command_order"]
                            >= stop_after_command_order
                            and snapshot["buffer_read_bit_position"]
                            >= int(
                                rows[stop_after_command_order]["end"]
                            )
                        ):
                            stop_after_step = True
                            stopped_at_command_order = True
                        if (
                            close_after_terminal_command
                            and not stop_after_step
                            and not untracked_command_call
                            and not service_transient_call
                            and boundary_failure is None
                            and terminal_close_after_step is None
                            and not terminal_close_requests
                            and snapshot["command_order"]
                            + command_order_offset
                            == len(rows) - 1
                            and snapshot["command_bit_position"]
                            == int(rows[-1]["start"])
                            and snapshot["buffer_read_bit_position"]
                            == int(rows[-1]["end"])
                        ):
                            terminal_close_after_step = {
                                "process_id": pid,
                                "thread_id": int(event.dwThreadId),
                                "mechanism": (
                                    "wm_close_after_exact_terminal_command"
                                ),
                                "target_command_order": len(rows) - 1,
                                "target_update": int(rows[-1]["update"]),
                                "target_kind": str(rows[-1]["kind"]),
                                "target_command_number": int(
                                    rows[-1]["command_number"]
                                ),
                                "target_short_form": bool(
                                    rows[-1]["short_form"]
                                ),
                                "target_start_bit_position": int(
                                    rows[-1]["start"]
                                ),
                                "target_end_bit_position": int(
                                    rows[-1]["end"]
                                ),
                                "observed_command_order": snapshot[
                                    "command_order"
                                ],
                                "observed_command_order_offset": (
                                    command_order_offset
                                ),
                                "observed_update": snapshot["update"],
                                "observed_command_bit_position": snapshot[
                                    "command_bit_position"
                                ],
                                "observed_read_bit_position": snapshot[
                                    "buffer_read_bit_position"
                                ],
                            }
                            stopped_at_command_order = True
                            print(
                                "terminal_close_armed "
                                f"tid={event.dwThreadId} "
                                f"update={snapshot['update']} "
                                "command_order="
                                f"{snapshot['command_order']} "
                                "command_order_offset="
                                f"{command_order_offset} "
                                "read_bitpos="
                                f"{snapshot['buffer_read_bit_position']}",
                                flush=True,
                            )
                    finally:
                        if close_thread:
                            kernel32.CloseHandle(thread)
                elif (
                    startup_global_mtrand_observation_oracle is not None
                    and code
                    in (EXCEPTION_SINGLE_STEP, STATUS_WX86_SINGLE_STEP)
                    and int(event.dwThreadId)
                    == startup_handoff_main_thread_id
                    and stepping_thread is None
                ):
                    if not startup_global_mtrand_hardware_armed:
                        raise RuntimeError(
                            "startup global MTRand hardware hit while "
                            "disarmed"
                        )
                    thread = kernel32.OpenThread(
                        THREAD_ACCESS | THREAD_SUSPEND_RESUME,
                        False,
                        event.dwThreadId,
                    )
                    if not thread:
                        raise ctypes.WinError(ctypes.get_last_error())
                    try:
                        context = _read_debug_context(thread)
                        wrapper_address = (
                            startup_global_mtrand_observation_oracle
                            .wrapper_address
                        )
                        hardware_hit = bool(int(context.Dr6) & 0x1) or any(
                            candidate == wrapper_address
                            for candidate in (
                                exception_address,
                                int(context.Eip),
                            )
                        )
                        hidden_write_hit = (
                            startup_hidden_draw_requested
                            and startup_hidden_draw_watch_active
                            and bool(int(context.Dr6) & 0x2)
                        )
                        if hidden_write_hit and hardware_hit:
                            raise RuntimeError(
                                "startup wrapper and hidden-draw watchpoints "
                                "triggered together"
                            )
                        if hidden_write_hit:
                            framework_update = (
                                _startup_global_mtrand_framework_update(
                                    process
                                )
                            )
                            post_state = read_memory(
                                process,
                                G_FRAMEWORK_MTRAND_ADDRESS,
                                MTRAND_STATE_BYTES,
                            )
                            post_index = struct.unpack_from(
                                "<I",
                                post_state,
                                MTRAND_STATE_WORDS * 4,
                            )[0]
                            stack_return = struct.unpack(
                                "<I",
                                read_memory(process, int(context.Esp), 4),
                            )[0]
                            stack_code = _startup_hidden_draw_stack_code(
                                process,
                                int(context.Esp),
                            )
                            row = {
                                "order": len(startup_hidden_draw_rows),
                                "phase": (
                                    "start_boundary_wrapper_draw"
                                    if not startup_hidden_draw_rows
                                    else "hidden_interval_draw"
                                ),
                                "thread_id": int(event.dwThreadId),
                                "exception_address": exception_address,
                                "instruction_pointer": int(context.Eip),
                                "instruction_pointer_hex": (
                                    f"0x{int(context.Eip):08x}"
                                ),
                                "stack_pointer": int(context.Esp),
                                "stack_return": stack_return,
                                "stack_return_hex": (
                                    f"0x{stack_return:08x}"
                                ),
                                "stack_code": stack_code,
                                "framework_update": framework_update,
                                "post_index": post_index,
                                "post_state_sha256": _sha256_bytes(
                                    post_state
                                ),
                                "hardware_breakpoint_slot": 1,
                                "hardware_breakpoint_access": "write",
                                "hardware_breakpoint_length_bytes": 4,
                                "process_memory_writes": 0,
                                "process_memory_mutation": False,
                                "process_context_mutation": True,
                                "persistent_file_modified": False,
                            }
                            context.Dr6 = 0
                            context.EFlags |= RESUME_FLAG
                            _write_debug_context(thread, context)
                            startup_hidden_draw_rows.append(row)
                            assert (
                                startup_global_mtrand_observation_receipt
                                is not None
                            )
                            hidden_receipt = (
                                startup_global_mtrand_observation_receipt[
                                    "hidden_draw_interval"
                                ]
                            )
                            hidden_receipt["index_write_count"] = len(
                                startup_hidden_draw_rows
                            )
                            if len(startup_hidden_draw_rows) % 50 == 0:
                                print(
                                    "startup_hidden_draw_progress writes="
                                    f"{len(startup_hidden_draw_rows)} "
                                    f"update={framework_update} "
                                    f"post_index={post_index}",
                                    flush=True,
                                )
                        elif not hardware_hit:
                            status = DBG_EXCEPTION_NOT_HANDLED
                        else:
                            if (
                                stepping_kind is not None
                                or stepping_suspended_threads
                                or held_thread
                                or pending_broker is not None
                                or post_bypass_return_breakpoint is not None
                                or pending_post_bypass_worker_yield is not None
                                or post_bypass_worker_yield_thread
                            ):
                                raise RuntimeError(
                                    "startup global MTRand hardware hit "
                                    "collided with serialized command tracing"
                                )
                            order = len(startup_global_mtrand_rows)
                            entries = (
                                startup_global_mtrand_observation_oracle
                                .entries
                            )
                            if order >= len(entries):
                                raise RuntimeError(
                                    "unexpected extra startup global MTRand "
                                    "wrapper call"
                                )
                            expected = entries[order]
                            stack_return = struct.unpack(
                                "<I",
                                read_memory(process, int(context.Esp), 4),
                            )[0]
                            framework_update = (
                                _startup_global_mtrand_framework_update(
                                    process
                                )
                            )
                            live_state = read_memory(
                                process,
                                G_FRAMEWORK_MTRAND_ADDRESS,
                                MTRAND_STATE_BYTES,
                            )
                            live_index = struct.unpack_from(
                                "<I",
                                live_state,
                                MTRAND_STATE_WORDS * 4,
                            )[0]
                            live_hash = _sha256_bytes(live_state)
                            (
                                inferred_output,
                                inferred_post_state,
                                inferred_post_index,
                                inferred_post_hash,
                            ) = _global_mtrand_call_transition(live_state)
                            caller_match = stack_return == expected.caller
                            update_match = (
                                framework_update
                                == expected.framework_update
                            )
                            pre_state_match = (
                                live_index == expected.pre_index
                                and live_hash
                                == expected.pre_state_sha256
                            )
                            transition_match = (
                                inferred_output == expected.output
                                and inferred_post_index
                                == expected.post_index
                                and inferred_post_hash
                                == expected.post_state_sha256
                            )
                            exact_match = (
                                caller_match
                                and update_match
                                and pre_state_match
                                and transition_match
                            )
                            completed = order + 1 == len(entries)
                            if (
                                startup_hidden_draw_requested
                                and order
                                == startup_hidden_draw_start_entry_order
                            ):
                                if (
                                    startup_hidden_draw_watch_active
                                    or startup_hidden_draw_rows
                                ):
                                    raise RuntimeError(
                                        "startup hidden-draw watch start "
                                        "boundary repeated"
                                    )
                                context.Dr1 = (
                                    G_FRAMEWORK_MTRAND_ADDRESS
                                    + MTRAND_STATE_WORDS * 4
                                )
                                context.Dr7 &= ~(
                                    (0x3 << 2) | (0xF << 20)
                                )
                                context.Dr7 |= (0x1 << 2) | (0xD << 20)
                                startup_hidden_draw_watch_active = True
                                startup_hidden_draw_start_boundary_observed = (
                                    True
                                )
                                assert (
                                    startup_global_mtrand_observation_receipt
                                    is not None
                                )
                                hidden_receipt = (
                                    startup_global_mtrand_observation_receipt[
                                        "hidden_draw_interval"
                                    ]
                                )
                                hidden_receipt[
                                    "start_boundary_observed"
                                ] = True
                                hidden_receipt["watch_armed"] = True
                            if (
                                startup_hidden_draw_requested
                                and order
                                == startup_hidden_draw_stop_entry_order
                            ):
                                if (
                                    not startup_hidden_draw_watch_active
                                    or not startup_hidden_draw_rows
                                    or not startup_hidden_draw_start_boundary_observed
                                ):
                                    raise RuntimeError(
                                        "startup hidden-draw watch stop "
                                        "boundary arrived before data"
                                    )
                                context.Dr7 &= ~(0x3 << 2)
                                startup_hidden_draw_watch_active = False
                                startup_hidden_draw_stop_boundary_observed = (
                                    True
                                )
                                assert (
                                    startup_global_mtrand_observation_receipt
                                    is not None
                                )
                                hidden_receipt = (
                                    startup_global_mtrand_observation_receipt[
                                        "hidden_draw_interval"
                                    ]
                                )
                                hidden_receipt[
                                    "stop_boundary_observed"
                                ] = True
                                hidden_receipt["watch_disarmed"] = True
                            context.Dr6 = 0
                            if completed:
                                context.Dr7 &= ~0x3
                            context.EFlags |= RESUME_FLAG
                            _write_debug_context(thread, context)
                            row = {
                                "order": order,
                                "thread_id": int(event.dwThreadId),
                                "exception_address": exception_address,
                                "instruction_pointer": int(context.Eip),
                                "stack_pointer": int(context.Esp),
                                "caller": stack_return,
                                "caller_hex": f"0x{stack_return:08x}",
                                "framework_update": framework_update,
                                "pre_index": live_index,
                                "pre_state_sha256": live_hash,
                                "inferred_output": inferred_output,
                                "inferred_post_index": inferred_post_index,
                                "inferred_post_state_sha256": (
                                    inferred_post_hash
                                ),
                                "expected_source_order": (
                                    expected.source_order
                                ),
                                "expected_framework_update": (
                                    expected.framework_update
                                ),
                                "expected_caller": expected.caller,
                                "expected_caller_hex": (
                                    f"0x{expected.caller:08x}"
                                ),
                                "expected_output": expected.output,
                                "expected_pre_draw_count": (
                                    expected.pre_draw_count
                                ),
                                "expected_pre_index": expected.pre_index,
                                "expected_pre_state_sha256": (
                                    expected.pre_state_sha256
                                ),
                                "expected_post_draw_count": (
                                    expected.post_draw_count
                                ),
                                "expected_post_index": expected.post_index,
                                "expected_post_state_sha256": (
                                    expected.post_state_sha256
                                ),
                                "caller_match": caller_match,
                                "framework_update_match": update_match,
                                "pre_state_match": pre_state_match,
                                "transition_match": transition_match,
                                "exact_match": exact_match,
                                "hardware_breakpoint_slot": 0,
                                "hardware_breakpoint_disabled_after_hit": (
                                    completed
                                ),
                                "process_memory_writes": 0,
                                "process_memory_mutation": False,
                                "process_context_mutation": True,
                                "persistent_file_modified": False,
                            }
                            startup_global_mtrand_rows.append(row)
                            assert (
                                startup_global_mtrand_observation_receipt
                                is not None
                            )
                            startup_global_mtrand_observation_receipt[
                                "hit_count"
                            ] = len(startup_global_mtrand_rows)
                            startup_global_mtrand_observation_receipt[
                                "register_housekeeping_count"
                            ] = len(startup_global_mtrand_rows)
                            startup_global_mtrand_observation_receipt[
                                "exact_match_count"
                            ] = sum(
                                bool(value["exact_match"])
                                for value in startup_global_mtrand_rows
                            )
                            startup_global_mtrand_observation_receipt[
                                "mismatch_count"
                            ] = sum(
                                not bool(value["exact_match"])
                                for value in startup_global_mtrand_rows
                            )
                            if (
                                len(startup_global_mtrand_rows) % 100 == 0
                                or completed
                            ):
                                print(
                                    "startup_global_mtrand_observation_"
                                    "progress hits="
                                    f"{len(startup_global_mtrand_rows)}/"
                                    f"{len(entries)} update="
                                    f"{framework_update} mismatches="
                                    f"{startup_global_mtrand_observation_receipt['mismatch_count']}",
                                    flush=True,
                                )
                    finally:
                        kernel32.CloseHandle(thread)
                elif (
                    gameplay_mtrand_oracle is not None
                    and code
                    in (EXCEPTION_SINGLE_STEP, STATUS_WX86_SINGLE_STEP)
                    and int(event.dwThreadId)
                    == startup_handoff_main_thread_id
                    and stepping_thread is None
                ):
                    if not gameplay_mtrand_hardware_armed:
                        raise RuntimeError(
                            "gameplay MTRand hardware hit while disarmed"
                        )
                    thread = kernel32.OpenThread(
                        THREAD_ACCESS | THREAD_SUSPEND_RESUME,
                        False,
                        event.dwThreadId,
                    )
                    if not thread:
                        raise ctypes.WinError(ctypes.get_last_error())
                    try:
                        context = _read_debug_context(thread)
                        triggered_slots = [
                            slot
                            for slot in range(
                                len(gameplay_mtrand_breakpoint_addresses)
                            )
                            if int(context.Dr6) & (0x1 << slot)
                        ]
                        if len(triggered_slots) > 1:
                            raise RuntimeError(
                                "multiple gameplay MTRand hardware slots "
                                "triggered together"
                            )
                        hit_address: int | None = None
                        hit_slot: int | None = None
                        if triggered_slots:
                            hit_slot = triggered_slots[0]
                            hit_address = (
                                gameplay_mtrand_breakpoint_addresses[
                                    hit_slot
                                ]
                            )
                        else:
                            for candidate in (
                                exception_address,
                                int(context.Eip),
                            ):
                                if candidate in (
                                    gameplay_mtrand_breakpoint_addresses
                                ):
                                    hit_address = candidate
                                    hit_slot = (
                                        gameplay_mtrand_breakpoint_addresses
                                        .index(candidate)
                                    )
                                    break
                        if hit_address is None or hit_slot is None:
                            status = DBG_EXCEPTION_NOT_HANDLED
                        else:
                            if (
                                stepping_kind is not None
                                or stepping_suspended_threads
                                or held_thread
                                or pending_broker is not None
                                or post_bypass_return_breakpoint is not None
                                or pending_post_bypass_worker_yield is not None
                                or post_bypass_worker_yield_thread
                            ):
                                raise RuntimeError(
                                    "gameplay MTRand hardware hit collided "
                                    "with serialized command tracing"
                                )
                            order = len(gameplay_mtrand_rows)
                            if order >= len(
                                gameplay_mtrand_oracle.entries
                            ):
                                raise RuntimeError(
                                    "unexpected extra gameplay MTRand call"
                                )
                            entry = gameplay_mtrand_oracle.entries[order]
                            if entry.caller != hit_address:
                                raise RuntimeError(
                                    "gameplay MTRand oracle caller mismatch: "
                                    f"observed=0x{hit_address:08X}, "
                                    f"expected=0x{entry.caller:08X}, "
                                    f"order={order}"
                                )
                            snapshot = _gameplay_mtrand_board_snapshot(
                                process
                            )
                            observed_contract = {
                                "framework_update": snapshot[
                                    "framework_update"
                                ],
                                "native_game_time": snapshot[
                                    "native_game_time"
                                ],
                                "score": snapshot["score"],
                                "score_target": snapshot[
                                    "score_target"
                                ],
                            }
                            expected_contract = {
                                "framework_update": (
                                    entry.framework_update
                                ),
                                "native_game_time": (
                                    entry.native_game_time
                                ),
                                "score": entry.score,
                                "score_target": entry.score_target,
                            }
                            if observed_contract != expected_contract:
                                raise RuntimeError(
                                    "gameplay MTRand oracle contract "
                                    "mismatch: observed="
                                    f"{observed_contract}, expected="
                                    f"{expected_contract}"
                                )
                            live_state = read_memory(
                                process,
                                G_FRAMEWORK_MTRAND_ADDRESS,
                                MTRAND_STATE_BYTES,
                            )
                            live_index = struct.unpack_from(
                                "<I",
                                live_state,
                                MTRAND_STATE_WORDS * 4,
                            )[0]
                            live_eax = int(context.Eax)
                            changed = live_state != entry.post_payload
                            caller_completed = (
                                gameplay_mtrand_last_order_by_caller[
                                    hit_address
                                ]
                                == order
                            )
                            try:
                                if changed:
                                    write_memory(
                                        process,
                                        G_FRAMEWORK_MTRAND_ADDRESS,
                                        entry.post_payload,
                                    )
                                if (
                                    read_memory(
                                        process,
                                        G_FRAMEWORK_MTRAND_ADDRESS,
                                        MTRAND_STATE_BYTES,
                                    )
                                    != entry.post_payload
                                ):
                                    raise RuntimeError(
                                        "gameplay MTRand sync writeback "
                                        "mismatch"
                                    )
                                context.Eax = entry.output
                                context.Dr6 = 0
                                if caller_completed:
                                    context.Dr7 &= ~(
                                        0x3 << (hit_slot * 2)
                                    )
                                context.EFlags |= RESUME_FLAG
                                _write_debug_context(thread, context)
                            except Exception:
                                if changed:
                                    write_memory(
                                        process,
                                        G_FRAMEWORK_MTRAND_ADDRESS,
                                        live_state,
                                    )
                                raise
                            row = {
                                "order": order,
                                "thread_id": int(event.dwThreadId),
                                "caller": hit_address,
                                "caller_hex": f"0x{hit_address:08x}",
                                "hardware_breakpoint_slot": hit_slot,
                                "hardware_breakpoint_disabled_after_hit": (
                                    caller_completed
                                ),
                                **observed_contract,
                                "live_eax": live_eax,
                                "restored_eax": entry.output,
                                "live_index": live_index,
                                "live_state_sha256": _sha256_bytes(
                                    live_state
                                ),
                                "post_draw_count": (
                                    entry.post_draw_count
                                ),
                                "post_index": entry.post_index,
                                "post_state_sha256": (
                                    entry.post_state_sha256
                                ),
                                "bytes_written": (
                                    MTRAND_STATE_BYTES if changed else 0
                                ),
                                "changed": changed,
                            }
                            gameplay_mtrand_rows.append(row)
                            assert gameplay_mtrand_sync_receipt is not None
                            gameplay_mtrand_sync_receipt[
                                "hit_count"
                            ] = len(gameplay_mtrand_rows)
                            gameplay_mtrand_sync_receipt[
                                "state_correction_count"
                            ] = sum(
                                bool(value["changed"])
                                for value in gameplay_mtrand_rows
                            )
                            gameplay_mtrand_sync_receipt[
                                "bytes_written_total"
                            ] = sum(
                                int(value["bytes_written"])
                                for value in gameplay_mtrand_rows
                            )
                            gameplay_mtrand_sync_receipt[
                                "process_memory_mutation"
                            ] = any(
                                bool(value["changed"])
                                for value in gameplay_mtrand_rows
                            )
                            gameplay_mtrand_sync_receipt[
                                "process_context_mutation"
                            ] = bool(gameplay_mtrand_rows)
                            gameplay_mtrand_sync_receipt[
                                "process_state_mutation"
                            ] = bool(gameplay_mtrand_rows)
                            gameplay_mtrand_sync_receipt[
                                "register_mutation_count"
                            ] = len(gameplay_mtrand_rows)
                            if len(gameplay_mtrand_rows) % 250 == 0:
                                print(
                                    "gameplay_mtrand_sync_progress "
                                    f"hits={len(gameplay_mtrand_rows)}/"
                                    f"{len(gameplay_mtrand_oracle.entries)} "
                                    "update="
                                    f"{entry.framework_update} "
                                    "corrections="
                                    f"{gameplay_mtrand_sync_receipt['state_correction_count']}",
                                    flush=True,
                                )
                    finally:
                        kernel32.CloseHandle(thread)
                elif (
                    code in (EXCEPTION_SINGLE_STEP, STATUS_WX86_SINGLE_STEP)
                    and stepping_thread == int(event.dwThreadId)
                ):
                    thread = kernel32.OpenThread(
                        THREAD_ACCESS | THREAD_SUSPEND_RESUME,
                        False,
                        event.dwThreadId,
                    )
                    if not thread:
                        raise ctypes.WinError(ctypes.get_last_error())
                    try:
                        context = WOW64_CONTEXT()
                        context.ContextFlags = CONTEXT_FULL
                        if not kernel32.Wow64GetThreadContext(
                            thread, ctypes.byref(context)
                        ):
                            raise ctypes.WinError(ctypes.get_last_error())
                        context.EFlags &= ~TRAP_FLAG
                        if not kernel32.Wow64SetThreadContext(
                            thread, ctypes.byref(context)
                        ):
                            raise ctypes.WinError(ctypes.get_last_error())
                        completed_kind = stepping_kind
                        stepping_thread = None
                        stepping_kind = None
                        if completed_kind in {
                            "source_bound_global_precall",
                            "source_bound_global_precall_skip",
                        }:
                            if (
                                source_bound_precall_original is None
                                or source_bound_precall_breakpoint_armed
                            ):
                                raise RuntimeError(
                                    "unexpected source-bound global precall "
                                    "single-step state"
                                )
                            if (
                                completed_kind
                                == "source_bound_global_precall_skip"
                            ):
                                write_memory(
                                    process,
                                    SOURCE_BOUND_GLOBAL_PRECALL_ADDRESS,
                                    b"\xCC",
                                )
                                source_bound_precall_breakpoint_armed = True
                                kernel32.FlushInstructionCache(
                                    process,
                                    ctypes.c_void_p(
                                        SOURCE_BOUND_GLOBAL_PRECALL_ADDRESS
                                    ),
                                    1,
                                )
                            else:
                                source_bound_precall_suspended_threads = (
                                    stepping_suspended_threads
                                )
                                stepping_suspended_threads = []
                        elif completed_kind == "board_seed":
                            if stop_after_board_seed:
                                should_stop = True
                        elif (
                            completed_kind
                            == "attach_stabilization_board_failure"
                        ):
                            should_stop = True
                        elif completed_kind == "board_seed_skip":
                            if (
                                board_seed_address is None
                                or board_original is None
                                or board_breakpoint_armed
                            ):
                                raise RuntimeError(
                                    "unexpected skipped board seed state"
                                )
                            write_memory(
                                process,
                                board_seed_address,
                                b"\xCC",
                            )
                            board_breakpoint_armed = True
                            kernel32.FlushInstructionCache(
                                process,
                                ctypes.c_void_p(board_seed_address),
                                1,
                            )
                        elif completed_kind == "command":
                            if terminal_close_after_step is not None:
                                hwnd, requested_ns = _post_terminal_close(pid)
                                terminal_close_requests.append(
                                    {
                                        **terminal_close_after_step,
                                        "main_window_handle": hwnd,
                                        "requested_perf_counter_ns": (
                                            requested_ns
                                        ),
                                        "post_message_succeeded": True,
                                    }
                                )
                                terminal_close_after_step = None
                                terminal_close_deadline = (
                                    time.monotonic() + 10.0
                                )
                                print(
                                    "terminal_close_requested "
                                    f"pid={pid} hwnd={hwnd} "
                                    f"requested_perf_counter_ns={requested_ns}",
                                    flush=True,
                                )
                            elif stop_after_step:
                                should_stop = True
                            else:
                                write_memory(process, address, b"\xCC")
                                breakpoint_armed = True
                                kernel32.FlushInstructionCache(
                                    process,
                                    ctypes.c_void_p(address),
                                    1,
                                )
                        else:
                            raise RuntimeError(
                                "single-step kind is missing"
                            )
                        if post_bypass_worker_yield_request is not None:
                            if (
                                completed_kind != "command"
                                or post_bypass_worker_yield_request[
                                    "thread_id"
                                ]
                                != int(event.dwThreadId)
                                or held_thread
                                or post_bypass_worker_yield_thread
                                or post_bypass_return_breakpoint is not None
                                or post_bypass_return_breakpoint_original
                                is not None
                                or pending_post_bypass_worker_yield
                                is not None
                            ):
                                raise RuntimeError(
                                    "post-bypass worker yield state is invalid"
                                )
                            return_address = (
                                post_bypass_worker_yield_request[
                                    "return_address"
                                ]
                            )
                            if (
                                return_address != MAIN_DEMO_CALLER
                                or return_address == address
                                or return_address == board_seed_address
                            ):
                                raise RuntimeError(
                                    "post-bypass return breakpoint address "
                                    "is invalid"
                                )
                            return_original = read_memory(
                                process,
                                return_address,
                                1,
                            )
                            if return_original == b"\xCC":
                                raise RuntimeError(
                                    "post-bypass return breakpoint already "
                                    "contains INT3"
                                )
                            write_memory(
                                process,
                                return_address,
                                b"\xCC",
                            )
                            kernel32.FlushInstructionCache(
                                process,
                                ctypes.c_void_p(return_address),
                                1,
                            )
                            post_bypass_return_breakpoint_original = (
                                return_original
                            )
                            post_bypass_return_breakpoint = {
                                **post_bypass_worker_yield_request,
                                "return_breakpoint_original_byte": int(
                                    return_original[0]
                                ),
                                "return_breakpoint_armed_perf_counter_ns": (
                                    time.perf_counter_ns()
                                ),
                            }
                            post_bypass_worker_yield_request = None
                            print(
                                "startup_post_bypass_return_breakpoint_armed "
                                f"tid={event.dwThreadId} "
                                "return_address=0x"
                                f"{return_address:08X} "
                                "initial_read_bitpos="
                                f"{post_bypass_return_breakpoint['initial_read_bit_position']} "
                                "corridor_end_bitpos="
                                f"{post_bypass_return_breakpoint['corridor_end_bit_position']}",
                                flush=True,
                            )
                        _resume_suspended_threads(
                            stepping_suspended_threads
                        )
                        if (
                            completed_kind == "board_seed"
                            and source_bound_board_precall_global_restore
                            and source_bound_precall_observations
                        ):
                            precall_observation = (
                                source_bound_precall_observations[0]
                            )
                            precall_observation[
                                "atomic_window_completed"
                            ] = True
                            precall_observation[
                                "other_threads_resumed_after_board_call"
                            ] = True
                            precall_observation[
                                "atomic_window_finished_perf_counter_ns"
                            ] = time.perf_counter_ns()
                        if (
                            should_stop
                            and suspend_main_thread_on_stop
                        ):
                            previous_suspend_count = int(
                                kernel32.SuspendThread(thread)
                            )
                            if (
                                previous_suspend_count
                                == INVALID_SUSPEND_COUNT
                            ):
                                raise ctypes.WinError(
                                    ctypes.get_last_error()
                                )
                            if previous_suspend_count != 0:
                                restore_count = int(
                                    kernel32.ResumeThread(thread)
                                )
                                if (
                                    restore_count
                                    == INVALID_SUSPEND_COUNT
                                ):
                                    raise ctypes.WinError(
                                        ctypes.get_last_error()
                                    )
                                raise RuntimeError(
                                    "main-thread handoff expected suspend "
                                    f"count 0, observed "
                                    f"{previous_suspend_count}"
                                )
                            handoff_main_thread_suspended = True
                            handoff_main_thread_id = int(
                                event.dwThreadId
                            )
                            handoff_previous_suspend_count = (
                                previous_suspend_count
                            )
                            print(
                                "handoff_main_thread_suspended "
                                f"tid={handoff_main_thread_id} "
                                f"update={last_snapshot_update} "
                                "previous_suspend_count="
                                f"{previous_suspend_count}",
                                flush=True,
                            )
                    finally:
                        kernel32.CloseHandle(thread)
                elif code == MICROSOFT_CPP_EXCEPTION:
                    status = DBG_EXCEPTION_NOT_HANDLED
                elif code not in (
                    EXCEPTION_BREAKPOINT,
                    STATUS_WX86_BREAKPOINT,
                ):
                    record = exception.ExceptionRecord
                    parameter_count = min(
                        int(record.NumberParameters),
                        len(record.ExceptionInformation),
                    )
                    information = [
                        int(record.ExceptionInformation[index])
                        for index in range(parameter_count)
                    ]
                    observation: dict[str, Any] = {
                        "process_id": int(event.dwProcessId),
                        "thread_id": int(event.dwThreadId),
                        "exception_code": code,
                        "exception_code_hex": f"0x{code:08X}",
                        "exception_flags": int(record.ExceptionFlags),
                        "exception_address": exception_address,
                        "exception_address_hex": (
                            f"0x{exception_address:08X}"
                        ),
                        "first_chance": bool(exception.dwFirstChance),
                        "parameters": information,
                        "last_framework_update": last_snapshot_update,
                        "last_command_order": last_command_order,
                        "perf_counter_ns": time.perf_counter_ns(),
                    }
                    if code == 0xC0000005 and len(information) >= 2:
                        access_kind = {
                            0: "read",
                            1: "write",
                            8: "execute",
                        }.get(information[0], "unknown")
                        observation.update(
                            {
                                "access_kind": access_kind,
                                "access_address": information[1],
                                "access_address_hex": (
                                    f"0x{information[1]:08X}"
                                ),
                            }
                        )
                    thread = kernel32.OpenThread(
                        THREAD_ACCESS,
                        False,
                        event.dwThreadId,
                    )
                    if thread:
                        try:
                            context = WOW64_CONTEXT()
                            context.ContextFlags = CONTEXT_FULL
                            if kernel32.Wow64GetThreadContext(
                                thread,
                                ctypes.byref(context),
                            ):
                                observation["registers"] = {
                                    name.lower(): int(getattr(context, name))
                                    for name in (
                                        "Eax",
                                        "Ebx",
                                        "Ecx",
                                        "Edx",
                                        "Esi",
                                        "Edi",
                                        "Ebp",
                                        "Esp",
                                        "Eip",
                                        "EFlags",
                                    )
                                }
                                frame_chain: list[dict[str, Any]] = []
                                frame_pointer = int(context.Ebp)
                                for _ in range(16):
                                    try:
                                        frame = read_memory(
                                            process,
                                            frame_pointer,
                                            24,
                                        )
                                    except OSError:
                                        break
                                    previous = int.from_bytes(
                                        frame[0:4],
                                        "little",
                                    )
                                    return_address = int.from_bytes(
                                        frame[4:8],
                                        "little",
                                    )
                                    frame_chain.append(
                                        {
                                            "frame_pointer": frame_pointer,
                                            "return_address": return_address,
                                            "return_address_hex": (
                                                f"0x{return_address:08X}"
                                            ),
                                            "arguments": [
                                                int.from_bytes(
                                                    frame[offset : offset + 4],
                                                    "little",
                                                )
                                                for offset in range(8, 24, 4)
                                            ],
                                        }
                                    )
                                    if (
                                        previous <= frame_pointer
                                        or previous - frame_pointer
                                        > 1024 * 1024
                                    ):
                                        break
                                    frame_pointer = previous
                                observation["frame_chain"] = frame_chain
                                object_fields: dict[str, int] = {}
                                object_address = int(context.Edi)
                                for offset in (
                                    0x0,
                                    0x8C,
                                    0x9C,
                                    0x104,
                                    0x688,
                                    0x68C,
                                    0x6A4,
                                    0x7A8,
                                    0x7BC,
                                    0xEFC,
                                ):
                                    try:
                                        object_fields[f"0x{offset:03X}"] = (
                                            int.from_bytes(
                                                read_memory(
                                                    process,
                                                    object_address + offset,
                                                    4,
                                                ),
                                                "little",
                                            )
                                        )
                                    except OSError:
                                        break
                                observation["edi_object_address"] = (
                                    object_address
                                )
                                observation["edi_object_fields"] = (
                                    object_fields
                                )
                                try:
                                    observation["stack_hex"] = read_memory(
                                        process,
                                        int(context.Esp),
                                        64,
                                    ).hex()
                                except OSError as error:
                                    observation["stack_read_error"] = (
                                        f"{type(error).__name__}: {error}"
                                    )
                            else:
                                observation["context_read_error"] = (
                                    f"winerror:{ctypes.get_last_error()}"
                                )
                        finally:
                            kernel32.CloseHandle(thread)
                    else:
                        observation["thread_open_error"] = (
                            f"winerror:{ctypes.get_last_error()}"
                        )
                    try:
                        observation["instruction_hex"] = read_memory(
                            process,
                            exception_address,
                            16,
                        ).hex()
                    except OSError as error:
                        observation["instruction_read_error"] = (
                            f"{type(error).__name__}: {error}"
                        )
                    debug_exception_observations.append(observation)
                    print(
                        "debug_exception "
                        f"pid={observation['process_id']} "
                        f"tid={observation['thread_id']} "
                        f"code={observation['exception_code_hex']} "
                        f"address={observation['exception_address_hex']} "
                        f"first_chance={int(observation['first_chance'])} "
                        f"access_kind={observation.get('access_kind')} "
                        "access_address="
                        f"{observation.get('access_address_hex')}",
                        flush=True,
                    )
                    status = DBG_EXCEPTION_NOT_HANDLED
            elif event.dwDebugEventCode == CREATE_PROCESS_DEBUG_EVENT:
                file_handle = event.u.CreateProcessInfo.hFile
                if file_handle:
                    kernel32.CloseHandle(file_handle)
            elif event.dwDebugEventCode == LOAD_DLL_DEBUG_EVENT:
                file_handle = event.u.LoadDll.hFile
                if file_handle:
                    kernel32.CloseHandle(file_handle)
            elif event.dwDebugEventCode == EXIT_PROCESS_DEBUG_EVENT:
                observed_exit_code = int(
                    event.u.ExitProcess.dwExitCode
                )
                if int(event.dwProcessId) == pid:
                    exited = True
                    exit_code = observed_exit_code
                    print(
                        f"exit_process pid={pid} code={exit_code}",
                        flush=True,
                    )
                else:
                    print(
                        "exit_child_process "
                        f"pid={int(event.dwProcessId)} "
                        f"code={observed_exit_code}",
                        flush=True,
                    )

            if not kernel32.ContinueDebugEvent(
                event.dwProcessId, event.dwThreadId, status
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            terminal_handoff_waiting = (
                pending_broker is not None
                and bool(
                    pending_broker.get("event_driven_terminal", False)
                )
                and not bool(
                    pending_broker.get("event_driven_ready", False)
                )
            )
            if pending_broker is not None and not terminal_handoff_waiting:
                target_end = pending_broker[
                    "target_end_bit_position"
                ]
                settle_end = pending_broker[
                    "settle_end_bit_position"
                ]

                def read_service_state() -> tuple[int, int]:
                    return (
                        _read_i32(
                            process,
                            pending_broker["base"] + 0x608,
                        ),
                        read_memory(
                            process,
                            pending_broker["base"] + 0x620,
                            1,
                        )[0],
                    )

                event_driven_ready = bool(
                    pending_broker.get("event_driven_ready", False)
                )
                if event_driven_ready:
                    observed_read = pending_broker[
                        "event_observed_read"
                    ]
                    observed_needs = pending_broker[
                        "event_observed_needs"
                    ]
                    failure = pending_broker.get("event_failure")
                    probe_status = (
                        "boundary" if failure is None else "failure"
                    )
                else:
                    probe_timeout = min(
                        service_wait_timeout,
                        SERVICE_PROGRESS_PROBE_TIMEOUT_SECONDS,
                    )
                    (
                        observed_read,
                        observed_needs,
                        probe_status,
                        failure,
                    ) = _probe_service_progress(
                        initial_read=pending_broker[
                            "initial_read_bit_position"
                        ],
                        target_end=target_end,
                        settle_end=settle_end,
                        timeout=probe_timeout,
                        read_state=read_service_state,
                        monotonic=time.monotonic,
                        delay=time.sleep,
                        allow_intermediate_prepared=bool(
                            pending_broker[
                                "allow_intermediate_prepared"
                            ]
                        ),
                    )
                broker_bypassed = False
                if probe_status == "no_progress":
                    block_key = (
                        pending_broker["base"],
                        pending_broker["start_bit_position"],
                        target_end,
                    )
                    broker_bypassed = _may_bypass_stalled_service_block(
                        pending_broker,
                        already_bypassed=(
                            block_key in bypassed_service_blocks
                        ),
                        allow_prepared_orphan=bool(
                            pending_broker["orphan_prepared_block"]
                        ),
                    )
                    if broker_bypassed:
                        bypassed_service_blocks.add(block_key)
                        failure = None
                    else:
                        failure = (
                            "service broker made no progress: "
                            f"read={observed_read}, target={target_end}"
                        )
                elif probe_status == "progress":
                    observed_read, observed_needs, failure = (
                        _wait_for_service_boundary(
                            target_end=target_end,
                            settle_end=settle_end,
                            timeout=service_wait_timeout,
                            read_state=read_service_state,
                            monotonic=time.monotonic,
                            delay=time.sleep,
                            allow_intermediate_prepared=bool(
                                pending_broker[
                                    "allow_intermediate_prepared"
                                ]
                            ),
                        )
                    )

                block_key = (
                    pending_broker["base"],
                    pending_broker["start_bit_position"],
                    target_end,
                )
                arm_service_continuation = (
                    broker_bypassed
                    or (
                        failure is None
                        and block_key in bypassed_service_blocks
                    )
                    or (
                        failure is None
                        and target_end < observed_read <= settle_end
                    )
                )
                if arm_service_continuation:
                    continuation_end_index, continuation_end = (
                        _service_continuation_corridor(
                            rows,
                            pending_broker["start_index"],
                        )
                    )
                    allow_successful_write_spillover = (
                        allow_orphan_prepared_service_block
                        and pending_broker["demo_loading_complete"] == 1
                        and _is_uniform_successful_file_write_corridor(
                            rows,
                            pending_broker["start_index"],
                            continuation_end_index,
                        )
                    )
                    direct_font_cache_rows = tuple(
                        int(observation["row_index"])
                        for observation in (
                            direct_font_cache_file_write_observations
                        )
                    )
                    direct_font_cache_members = tuple(
                        str(
                            observation[
                                "file_write_font_cache_member"
                            ]
                        )
                        for observation in (
                            direct_font_cache_file_write_observations
                        )
                    )
                    exact_direct_font_cache_receipt = (
                        font_cache_manifest is not None
                        and len(set(direct_font_cache_members))
                        == len(direct_font_cache_members)
                        and all(
                            observation.get("recorded_success") is False
                            and int(
                                observation[
                                    "file_write_argument_size"
                                ]
                            )
                            == int(
                                observation["manifest_expected_size"]
                            )
                            and font_cache_manifest.get(
                                str(
                                    observation[
                                        "file_write_font_cache_member"
                                    ]
                                )
                            )
                            == int(
                                observation[
                                    "file_write_argument_size"
                                ]
                            )
                            for observation in (
                                direct_font_cache_file_write_observations
                            )
                        )
                    )
                    allow_preloading_failed_write_spillover = (
                        # This is the exact update-zero startup handoff, not
                        # an orphaned post-blackout attachment.  Requiring
                        # the generic orphan-adoption switch here prevented
                        # the manifest-bound startup partition from ever
                        # authorizing its bounded worker yield.
                        startup_handoff_requested
                        and startup_handoff_resumed_after_breakpoints
                        and startup_handoff_main_thread_id
                        == pending_broker["thread_id"]
                        and pending_broker["return_address"]
                        == MAIN_DEMO_CALLER
                        and font_cache_manifest_requested
                        and pending_broker["demo_loading_complete"] == 0
                        and pre_attach_file_write_debt_before_update is None
                        and not pre_attach_file_write_debt_rows
                        and command_order_offset == 0
                        and not command_order_rebase_rows
                        and not deferred_file_write_discharges
                        and not terminal_file_write_payload_handoffs
                        and not service_file_write_header_claims
                        and not font_cache_manifest_completion_discharges
                        and exact_direct_font_cache_receipt
                        and _is_exact_startup_font_cache_failed_write_partition(
                            rows,
                            start_index=pending_broker["start_index"],
                            end_index=continuation_end_index,
                            direct_row_indices=direct_font_cache_rows,
                            manifest_entry_count=(
                                len(font_cache_manifest)
                                if font_cache_manifest is not None
                                else 0
                            ),
                        )
                    )
                    initial_header_continuation = (
                        allow_successful_write_spillover
                        and pending_broker["needs_command"] == 0
                        and pending_broker[
                            "initial_command_bit_position"
                        ]
                        == int(
                            rows[pending_broker["start_index"]]["start"]
                        )
                        and observed_read
                        == pending_broker[
                            "initial_command_bit_position"
                        ]
                        + 10
                        and pending_broker["initial_command_order"]
                        == pending_broker["start_index"]
                        - command_order_offset
                    )
                    pending_service_continuation = {
                        "base": pending_broker["base"],
                        "thread_id": pending_broker["thread_id"],
                        "start_index": pending_broker["start_index"],
                        "end_index": continuation_end_index,
                        "end_bit_position": continuation_end,
                        "last_read_bit_position": observed_read,
                        "last_command_bit_position": (
                            pending_broker[
                                "initial_command_bit_position"
                            ]
                            if initial_header_continuation
                            else -1
                        ),
                        "last_command_order": (
                            pending_broker["initial_command_order"]
                            if initial_header_continuation
                            else -1
                        ),
                        "last_reentry_update": pending_broker["update"],
                        "last_reentry_kind": (
                            3 if initial_header_continuation else 0
                        ),
                        "reentries": 0,
                        "allow_bounded_late_successful_file_write_corridor": (
                            allow_successful_write_spillover
                        ),
                        "allow_bounded_late_preloading_failed_file_write_corridor": (
                            allow_preloading_failed_write_spillover
                        ),
                    }
                    if (
                        broker_bypassed
                        and allow_preloading_failed_write_spillover
                    ):
                        if post_bypass_worker_yield_request is not None:
                            raise RuntimeError(
                                "post-bypass worker yield request leaked"
                            )
                        alternate_target_row_index = -1
                        next_target_index = pending_broker["end_index"] + 1
                        if next_target_index <= continuation_end_index:
                            target_update = int(
                                rows[pending_broker["end_index"]]["update"]
                            )
                            next_update = int(
                                rows[next_target_index]["update"]
                            )
                            if next_update == target_update + 1:
                                alternate_target_row_index = next_target_index
                                while (
                                    alternate_target_row_index + 1
                                    <= continuation_end_index
                                    and int(
                                        rows[
                                            alternate_target_row_index + 1
                                        ]["update"]
                                    )
                                    == next_update
                                ):
                                    alternate_target_row_index += 1
                        post_bypass_worker_yield_request = {
                            "process_id": pid,
                            "thread_id": pending_broker["thread_id"],
                            "base": pending_broker["base"],
                            "corridor_start_index": pending_broker[
                                "start_index"
                            ],
                            "corridor_end_index": continuation_end_index,
                            "target_row_index": pending_broker["end_index"],
                            "alternate_target_row_index": (
                                alternate_target_row_index
                            ),
                            "initial_read_bit_position": observed_read,
                            "corridor_end_bit_position": continuation_end,
                            "command_order_offset": command_order_offset,
                            "return_address": pending_broker[
                                "return_address"
                            ],
                            "memory_writes": 0,
                        }
                    print(
                        "service_continuation_armed "
                        f"tid={pending_broker['thread_id']} "
                        f"update={pending_broker['update']} "
                        f"rows={pending_broker['start_index']}:"
                        f"{continuation_end_index} "
                        f"read_bitpos={observed_read} "
                        f"corridor_end_bitpos={continuation_end} "
                        "preloading_failed_write_spillover="
                        f"{int(allow_preloading_failed_write_spillover)}",
                        flush=True,
                    )

                stepping_thread = pending_broker["thread_id"]
                stepping_kind = "command"
                if stepping_suspended_threads and not event_driven_ready:
                    raise RuntimeError(
                        "serialized single-step state leaked"
                    )
                if not stepping_suspended_threads:
                    stepping_suspended_threads = _suspend_other_threads(
                        pid,
                        excluded_thread_id=stepping_thread,
                        access_denied_skip_events=(
                            access_denied_thread_skip_events
                        ),
                    )
                resume_count = int(
                    kernel32.ResumeThread(held_thread)
                )
                if resume_count == INVALID_SUSPEND_COUNT:
                    raise ctypes.WinError(ctypes.get_last_error())
                kernel32.CloseHandle(held_thread)
                held_thread = 0
                if broker_bypassed:
                    broker_bypassed_blocks += 1
                    print(
                        "service_broker_bypass "
                        f"tid={pending_broker['thread_id']} "
                        f"update={pending_broker['update']} "
                        f"read_bitpos={observed_read} "
                        f"target_end_bitpos={target_end} "
                        f"settle_end_bitpos={settle_end} "
                        "reason="
                        + (
                            "one_reattach_prepared_continuation"
                            if pending_broker[
                                "orphan_prepared_block"
                            ]
                            else "one_same_thread_retry"
                        ),
                        flush=True,
                    )
                elif failure is None:
                    brokered_blocks += 1
                    if (
                        pending_service_continuation is not None
                        and pending_broker["base"]
                        == pending_service_continuation["base"]
                        and bool(
                            pending_service_continuation.get(
                                "allow_bounded_late_preloading_failed_file_write_corridor",
                                False,
                            )
                        )
                        and pending_service_continuation["start_index"]
                        <= pending_broker["start_index"]
                        <= pending_broker["end_index"]
                        <= pending_service_continuation["end_index"]
                        and observed_read == target_end
                        and observed_needs == 1
                        and all(
                            rows[row_index].get("kind") == "file_write"
                            and not bool(rows[row_index]["short_form"])
                            and int(rows[row_index]["command_number"]) == 16
                            and rows[row_index].get("payload")
                            == {"success": False}
                            and int(rows[row_index]["end"])
                            - int(rows[row_index]["start"])
                            == 11
                            for row_index in range(
                                pending_broker["start_index"],
                                pending_broker["end_index"] + 1,
                            )
                        )
                    ):
                        brokered_end_index = pending_broker["end_index"]
                        brokered_end_row = rows[brokered_end_index]
                        if int(brokered_end_row["end"]) != observed_read:
                            raise RuntimeError(
                                "broker checkpoint read position drifted"
                            )
                        pending_service_continuation[
                            "last_brokered_row_index"
                        ] = brokered_end_index
                        pending_service_continuation[
                            "last_brokered_read_bit_position"
                        ] = observed_read
                        pending_service_continuation[
                            "last_brokered_command_bit_position"
                        ] = int(brokered_end_row["start"])
                        pending_service_continuation[
                            "last_brokered_command_order"
                        ] = brokered_end_index - command_order_offset
                        pending_service_continuation[
                            "last_brokered_update"
                        ] = pending_broker["update"]
                    if (
                        stop_after_command_order is not None
                        and pending_broker["end_index"]
                        >= stop_after_command_order
                        and observed_read
                        >= int(rows[stop_after_command_order]["end"])
                    ):
                        stop_after_step = True
                        stopped_at_command_order = True
                    print(
                        "service_broker_release "
                        f"tid={pending_broker['thread_id']} "
                        f"update={pending_broker['update']} "
                        f"read_bitpos={observed_read} "
                        f"needs={observed_needs} "
                        f"settle_end_bitpos={settle_end} "
                        f"resume_count={resume_count}",
                        flush=True,
                    )
                else:
                    broker_failures.append(failure)
                    stop_after_step = True
                    print(
                        "service_broker_failure "
                        f"tid={pending_broker['thread_id']} "
                        f"update={pending_broker['update']} "
                        f"read_bitpos={observed_read} "
                        f"needs={observed_needs} "
                        f"settle_end_bitpos={settle_end} "
                        f"detail={failure}",
                        flush=True,
                    )
                pending_broker = None
            if pending_post_bypass_worker_yield is not None:
                worker_yield = pending_post_bypass_worker_yield
                if (
                    not post_bypass_worker_yield_thread
                    or breakpoint_armed
                    or original is None
                ):
                    raise RuntimeError(
                        "post-bypass worker yield execution state is invalid"
                    )

                def read_post_bypass_worker_state() -> Mapping[str, int]:
                    base = worker_yield["base"]
                    return {
                        "buffer_read_bit_position": _read_i32(
                            process,
                            base + 0x608,
                        ),
                        "needs_command": read_memory(
                            process,
                            base + 0x620,
                            1,
                        )[0],
                        "is_short": read_memory(
                            process,
                            base + 0x621,
                            1,
                        )[0],
                        "command_number": _read_i32(
                            process,
                            base + 0x624,
                        ),
                        "command_order": _read_i32(
                            process,
                            base + 0x628,
                        ),
                        "command_bit_position": _read_i32(
                            process,
                            base + 0x62C,
                        ),
                        "demo_loading_complete": read_memory(
                            process,
                            base + 0x630,
                            1,
                        )[0],
                        "update": _read_i32(
                            process,
                            base + 0x4C4,
                        ),
                    }

                worker_suspended_threads: list[tuple[int, int]] = []
                worker_window_closed = False
                worker_continuation_commit: Mapping[
                    str, int | str
                ] | None = None

                def close_post_bypass_worker_window() -> None:
                    nonlocal worker_window_closed
                    if worker_window_closed:
                        return
                    worker_suspended_threads.extend(
                        _suspend_other_threads(
                            pid,
                            excluded_thread_id=worker_yield["thread_id"],
                            access_denied_skip_events=(
                                access_denied_thread_skip_events
                            ),
                        )
                    )
                    worker_window_closed = True

                closing_boundary_advanced = False
                try:
                    (
                        worker_observed,
                        worker_probe_status,
                        worker_failure,
                    ) = _wait_for_startup_post_bypass_worker_boundary(
                        rows,
                        corridor_start_index=worker_yield[
                            "corridor_start_index"
                        ],
                        corridor_end_index=worker_yield[
                            "corridor_end_index"
                        ],
                        target_row_index=worker_yield[
                            "target_row_index"
                        ],
                        initial_read_bit_position=worker_yield[
                            "initial_read_bit_position"
                        ],
                        command_order_offset=worker_yield[
                            "command_order_offset"
                        ],
                        alternate_target_row_index=(
                            None
                            if worker_yield[
                                "alternate_target_row_index"
                            ] < 0
                            else worker_yield[
                                "alternate_target_row_index"
                            ]
                        ),
                        timeout=min(
                            service_wait_timeout,
                            SERVICE_PROGRESS_PROBE_TIMEOUT_SECONDS,
                        ),
                        read_state=read_post_bypass_worker_state,
                        monotonic=time.monotonic,
                        delay=time.sleep,
                        close_window=close_post_bypass_worker_window,
                    )
                    preclose_worker_observed = dict(worker_observed)
                    if not worker_window_closed:
                        close_post_bypass_worker_window()
                    final_worker_observed = dict(
                        read_post_bypass_worker_state()
                    )
                    if (
                        worker_failure is None
                        and any(
                            final_worker_observed.get(name)
                            != worker_observed.get(name)
                            for name in (
                                "buffer_read_bit_position",
                                "needs_command",
                                "command_order",
                                "command_bit_position",
                                "is_short",
                                "command_number",
                                "demo_loading_complete",
                                "update",
                            )
                        )
                    ):
                        if final_worker_observed[
                            "buffer_read_bit_position"
                        ] < worker_observed[
                            "buffer_read_bit_position"
                        ]:
                            worker_failure = (
                                "startup post-bypass worker boundary "
                                "regressed while closing its bounded window"
                            )
                            worker_probe_status = "failure"
                        else:
                            (
                                closing_observed,
                                closing_status,
                                closing_failure,
                            ) = _wait_for_startup_post_bypass_worker_boundary(
                                rows,
                                corridor_start_index=worker_yield[
                                    "corridor_start_index"
                                ],
                                corridor_end_index=worker_yield[
                                    "corridor_end_index"
                                ],
                                target_row_index=worker_yield[
                                    "target_row_index"
                                ],
                                initial_read_bit_position=worker_yield[
                                    "initial_read_bit_position"
                                ],
                                command_order_offset=worker_yield[
                                    "command_order_offset"
                                ],
                                alternate_target_row_index=(
                                    None
                                    if worker_yield[
                                        "alternate_target_row_index"
                                    ] < 0
                                    else worker_yield[
                                        "alternate_target_row_index"
                                    ]
                                ),
                                timeout=0.05,
                                settle_interval=0.01,
                                read_state=lambda: final_worker_observed,
                                monotonic=time.monotonic,
                                delay=time.sleep,
                            )
                            if closing_failure is None:
                                worker_observed = closing_observed
                                worker_probe_status = closing_status
                                closing_boundary_advanced = True
                            else:
                                worker_failure = (
                                    "startup post-bypass worker boundary "
                                    "changed while closing its bounded "
                                    f"window: {closing_failure}"
                                )
                                worker_probe_status = "failure"
                    if worker_failure is None:
                        if (
                            pending_service_continuation is None
                            or pending_service_continuation["base"]
                            != worker_yield["base"]
                            or pending_service_continuation[
                                "thread_id"
                            ]
                            != worker_yield["thread_id"]
                            or pending_service_continuation[
                                "start_index"
                            ]
                            != worker_yield["corridor_start_index"]
                            or pending_service_continuation[
                                "end_index"
                            ]
                            != worker_yield["corridor_end_index"]
                        ):
                            worker_failure = (
                                "startup post-bypass worker continuation "
                                "identity changed before its atomic commit"
                            )
                            worker_probe_status = "failure"
                        else:
                            (
                                committed_continuation,
                                worker_continuation_commit,
                                worker_failure,
                            ) = _commit_startup_post_bypass_worker_continuation(
                                rows,
                                pending_service_continuation,
                                worker_observed,
                                target_row_index=worker_yield[
                                    "target_row_index"
                                ],
                                alternate_target_row_index=(
                                    None
                                    if worker_yield[
                                        "alternate_target_row_index"
                                    ] < 0
                                    else worker_yield[
                                        "alternate_target_row_index"
                                    ]
                                ),
                                initial_read_bit_position=worker_yield[
                                    "initial_read_bit_position"
                                ],
                                command_order_offset=worker_yield[
                                    "command_order_offset"
                                ],
                            )
                            if worker_failure is None:
                                assert committed_continuation is not None
                                assert worker_continuation_commit is not None
                                pending_service_continuation = (
                                    committed_continuation
                                )
                            else:
                                worker_probe_status = "failure"
                    write_memory(process, address, b"\xCC")
                    breakpoint_armed = True
                    kernel32.FlushInstructionCache(
                        process,
                        ctypes.c_void_p(address),
                        1,
                    )
                    command_breakpoint_rearmed_ns = (
                        time.perf_counter_ns()
                    )
                finally:
                    if worker_suspended_threads:
                        _resume_suspended_threads(
                            worker_suspended_threads
                        )
                resume_count = int(
                    kernel32.ResumeThread(
                        post_bypass_worker_yield_thread
                    )
                )
                if resume_count == INVALID_SUSPEND_COUNT:
                    raise ctypes.WinError(ctypes.get_last_error())
                if resume_count != 1:
                    raise RuntimeError(
                        "post-bypass worker yield resume count mismatch: "
                        f"{resume_count}"
                    )
                kernel32.CloseHandle(
                    post_bypass_worker_yield_thread
                )
                post_bypass_worker_yield_thread = 0
                worker_yield_finished_ns = time.perf_counter_ns()
                startup_post_bypass_worker_yields.append(
                    {
                        **worker_yield,
                        "observed_read_bit_position": worker_observed[
                            "buffer_read_bit_position"
                        ],
                        "observed_needs_command": worker_observed[
                            "needs_command"
                        ],
                        "observed_command_order": worker_observed[
                            "command_order"
                        ],
                        "observed_command_bit_position": worker_observed[
                            "command_bit_position"
                        ],
                        "observed_is_short": worker_observed[
                            "is_short"
                        ],
                        "observed_command_number": worker_observed[
                            "command_number"
                        ],
                        "observed_demo_loading_complete": worker_observed[
                            "demo_loading_complete"
                        ],
                        "observed_update": worker_observed["update"],
                        "settled_row_index": worker_observed.get(
                            "settled_row_index",
                            -1,
                        ),
                        "settled_after_payload": worker_observed.get(
                            "settled_after_payload",
                            0,
                        ),
                        "settled_at_corridor_end": worker_observed.get(
                            "settled_at_corridor_end",
                            0,
                        ),
                        "probe_status": worker_probe_status,
                        "failure": worker_failure,
                        "continuation_commit": (
                            worker_continuation_commit
                        ),
                        "preclose_observed_read_bit_position": (
                            preclose_worker_observed[
                                "buffer_read_bit_position"
                            ]
                        ),
                        "closing_boundary_advanced": (
                            closing_boundary_advanced
                        ),
                        "command_breakpoint_rearmed": True,
                        "command_breakpoint_rearmed_perf_counter_ns": (
                            command_breakpoint_rearmed_ns
                        ),
                        "probe_finished_perf_counter_ns": (
                            worker_yield_finished_ns
                        ),
                        "temporary_breakpoint_memory_writes": (
                            worker_yield[
                                "temporary_breakpoint_memory_writes"
                            ]
                            + 1
                        ),
                        "resume_count": resume_count,
                    }
                )
                print(
                    "startup_post_bypass_worker_yield "
                    f"tid={worker_yield['thread_id']} "
                    "initial_read_bitpos="
                    f"{worker_yield['initial_read_bit_position']} "
                    "observed_read_bitpos="
                    f"{worker_observed['buffer_read_bit_position']} "
                    f"needs={worker_observed['needs_command']} "
                    f"order={worker_observed['command_order']} "
                    "settled_row="
                    f"{worker_observed.get('settled_row_index', -1)} "
                    f"status={worker_probe_status} "
                    f"failure={worker_failure}",
                    flush=True,
                )
                pending_post_bypass_worker_yield = None
                if worker_failure is not None:
                    boundary_failures.append(worker_failure)
                    should_stop = True
            if should_stop or exited:
                break

        if (
            post_bypass_worker_yield_request is not None
            or post_bypass_return_breakpoint is not None
            or post_bypass_return_breakpoint_original is not None
            or pending_post_bypass_worker_yield is not None
            or post_bypass_worker_yield_thread
        ):
            worker_yield_failure = (
                "startup post-bypass worker yield ended before its bounded "
                "probe completed"
            )
            boundary_failures.append(worker_yield_failure)
            print(
                "startup_post_bypass_worker_yield_failure "
                f"detail={worker_yield_failure}",
                flush=True,
            )
        if (
            pending_service_continuation is not None
            and not broker_failures
            and not boundary_failures
        ):
            terminal_registry_receipt: Mapping[str, Any] | None = None
            if exited and exit_code == 0:
                (
                    terminal_registry_receipt,
                    _,
                ) = _audited_terminal_registry_write_eof_exit(
                    rows,
                    pending_service_continuation,
                    process_id=pid,
                    exited=exited,
                    exit_code=exit_code,
                    command_order_offset=command_order_offset,
                )
            if terminal_registry_receipt is not None:
                terminal_registry_write_eof_exits.append(
                    terminal_registry_receipt
                )
                print(
                    "terminal_registry_write_eof_exit "
                    f"tid={terminal_registry_receipt['thread_id']} "
                    f"update={terminal_registry_receipt['framework_update']} "
                    f"rows={terminal_registry_receipt['start_index']}:"
                    f"{terminal_registry_receipt['end_index']} "
                    "payload_bitpos="
                    f"{terminal_registry_receipt['terminal_payload_bit_position']} "
                    f"exit_code={terminal_registry_receipt['exit_code']}",
                    flush=True,
                )
                pending_service_continuation = None
        if (
            pending_service_continuation is not None
            and not broker_failures
            and not boundary_failures
        ):
            continuation_failure = (
                "service continuation ended before an exact offline boundary: "
                f"rows={pending_service_continuation['start_index']}:"
                f"{pending_service_continuation['end_index']}, "
                "last_read="
                f"{pending_service_continuation['last_read_bit_position']}"
            )
            broker_failures.append(continuation_failure)
            print(
                f"service_continuation_failure detail={continuation_failure}",
                flush=True,
            )
        if (
            attach_stabilization_pending
            and not attach_stabilization_verified
        ):
            stabilization_failure = (
                "attach stabilization ended before an exact complete main "
                "boundary: skipped_hits="
                f"{attach_stabilization_skipped_hits}"
            )
            boundary_failures.append(stabilization_failure)
            print(
                "attach_stabilization_failure "
                f"detail={stabilization_failure}",
                flush=True,
            )
        incomplete_broker_adjacent_claims = [
            observation
            for observation in service_file_write_header_claims
            if observation.get("mechanism")
            == "broker_adjacent_current_update_file_write"
            and observation.get("payload_completion_verified") is not True
        ]
        if incomplete_broker_adjacent_claims:
            claim_failure = (
                "broker-adjacent current-update file-write claim did not "
                "complete its exact payload handoff"
            )
            boundary_failures.append(claim_failure)
            print(
                "service_file_write_payload_completion_failure "
                f"count={len(incomplete_broker_adjacent_claims)} "
                f"detail={claim_failure}",
                flush=True,
            )
        file_write_accounted_rows = tuple(
            row_index
            for row_index in command_order_rebase_rows
            if is_exact_successful_file_write_debt_row(row_index)
        )
        effective_debt_rows = tuple(
            sorted(
                (
                    set(file_write_accounted_rows)
                    | set(service_consumed_file_write_debt_rows)
                    | set(pre_attach_file_write_debt_rows)
                )
                - set(diagnostic_successful_file_write_padding_rows)
                - set(service_consumed_file_write_surplus_rows)
            )
        )
        if (
            exited
            and diagnostic_successful_file_write_padding_rows
            and not broker_failures
            and not boundary_failures
        ):
            accounted_padding_rows = tuple(
                row_index
                for row_index in (
                    diagnostic_successful_file_write_padding_rows
                )
                if row_index in command_order_rebase_rows
            )
            observed_debt_rows = tuple(
                int(observation["debt_row_index"])
                for observation in deferred_file_write_discharges
            )
            if (
                accounted_padding_rows
                != diagnostic_successful_file_write_padding_rows
                or observed_debt_rows != effective_debt_rows
                or (
                    service_consumed_file_write_surplus_rows
                    and not successful_file_write_debt_barriers
                )
            ):
                debt_audit_failure = (
                    "provenance-bound successful file-write debt audit "
                    "did not close: padding="
                    f"{accounted_padding_rows}/"
                    f"{diagnostic_successful_file_write_padding_rows}, "
                    f"observed={observed_debt_rows}, "
                    f"expected={effective_debt_rows}"
                )
                boundary_failures.append(debt_audit_failure)
                print(
                    "successful_file_write_debt_audit_failure "
                    f"detail={debt_audit_failure}",
                    flush=True,
                )
        if (
            source_bound_board_precall_global_restore
            and not broker_failures
            and not boundary_failures
            and last_snapshot_update
            >= source_bound_board_thread_crt_state.source_framework_update
            and (
                len(source_bound_precall_observations) != 1
                or source_bound_precall_observations[0].get(
                    "atomic_window_completed"
                )
                is not True
                or source_bound_precall_suspended_threads
            )
        ):
            raise RuntimeError(
                "source-bound global precall atomic window did not complete"
            )
        if startup_global_mtrand_observation_oracle is not None:
            expected_startup_hits = len(
                startup_global_mtrand_observation_oracle.entries
            )
            assert startup_global_mtrand_observation_receipt is not None
            if len(startup_global_mtrand_rows) != expected_startup_hits:
                startup_failure = (
                    "startup global MTRand observation ended before all "
                    "source-indexed calls: observed="
                    f"{len(startup_global_mtrand_rows)}, "
                    f"expected={expected_startup_hits}"
                )
                boundary_failures.append(startup_failure)
                startup_global_mtrand_observation_receipt["failure"] = (
                    "missing_startup_global_mtrand_calls"
                )
                print(
                    "startup_global_mtrand_observation_failure "
                    f"detail={startup_failure}",
                    flush=True,
                )
            else:
                startup_global_mtrand_observation_receipt[
                    "oracle_complete"
                ] = True
            if startup_hidden_draw_requested:
                assert startup_hidden_draw_start_entry_order is not None
                assert startup_hidden_draw_stop_entry_order is not None
                hidden_receipt = (
                    startup_global_mtrand_observation_receipt[
                        "hidden_draw_interval"
                    ]
                )
                start_entry = (
                    startup_global_mtrand_observation_oracle.entries[
                        startup_hidden_draw_start_entry_order
                    ]
                )
                hidden_contract_passed = (
                    startup_hidden_draw_start_boundary_observed
                    and startup_hidden_draw_stop_boundary_observed
                    and not startup_hidden_draw_watch_active
                    and bool(startup_hidden_draw_rows)
                    and startup_hidden_draw_rows[0].get(
                        "post_state_sha256"
                    )
                    == start_entry.post_state_sha256
                    and startup_hidden_draw_rows[0].get("post_index")
                    == start_entry.post_index
                )
                hidden_receipt["index_write_count"] = len(
                    startup_hidden_draw_rows
                )
                hidden_receipt["observed_hidden_write_count"] = max(
                    0,
                    len(startup_hidden_draw_rows) - 1,
                )
                hidden_receipt["status"] = (
                    "PASS" if hidden_contract_passed else "FAIL"
                )
                if not hidden_contract_passed:
                    hidden_receipt["failure"] = (
                        "hidden_draw_watch_boundary_contract_failed"
                    )
                    startup_global_mtrand_observation_receipt["failure"] = (
                        "hidden_draw_watch_boundary_contract_failed"
                    )
                    boundary_failures.append(
                        "startup hidden-draw watch boundary contract failed"
                    )
            if (
                startup_global_mtrand_hardware_armed
                and startup_global_mtrand_hardware_original is not None
                and startup_handoff_main_thread_id is not None
                and not exited
            ):
                try:
                    _restore_gameplay_mtrand_breakpoints(
                        main_thread_id=startup_handoff_main_thread_id,
                        original=startup_global_mtrand_hardware_original,
                    )
                    startup_global_mtrand_hardware_armed = False
                    startup_global_mtrand_observation_receipt[
                        "hardware_breakpoint_restored"
                    ] = True
                except Exception as error:
                    startup_global_mtrand_hardware_restore_error = (
                        f"{type(error).__name__}: {error}"
                    )
                    startup_global_mtrand_observation_receipt[
                        "hardware_breakpoint_restore_error"
                    ] = startup_global_mtrand_hardware_restore_error
                    startup_global_mtrand_observation_receipt["failure"] = (
                        "hardware_breakpoint_restore_failed"
                    )
                    boundary_failures.append(
                        "startup global MTRand hardware breakpoint restore "
                        "failed: "
                        f"{startup_global_mtrand_hardware_restore_error}"
                    )
            elif exited:
                startup_global_mtrand_hardware_armed = False
                startup_global_mtrand_observation_receipt[
                    "hardware_breakpoint_restored"
                ] = True
            startup_global_mtrand_observation_receipt[
                "hardware_breakpoint_armed"
            ] = startup_global_mtrand_hardware_armed
            if (
                startup_global_mtrand_observation_receipt.get("failure")
                is None
                and startup_global_mtrand_observation_receipt.get(
                    "oracle_complete"
                )
                is True
                and startup_global_mtrand_observation_receipt.get(
                    "hardware_breakpoint_restored"
                )
                is True
                and (
                    not startup_hidden_draw_requested
                    or startup_global_mtrand_observation_receipt.get(
                        "hidden_draw_interval",
                        {},
                    ).get("status")
                    == "PASS"
                )
            ):
                startup_global_mtrand_observation_receipt["status"] = "PASS"
            else:
                startup_global_mtrand_observation_receipt["status"] = "FAIL"
        if gameplay_mtrand_oracle is not None:
            expected_gameplay_hits = len(
                gameplay_mtrand_oracle.entries
            )
            if len(gameplay_mtrand_rows) != expected_gameplay_hits:
                gameplay_failure = (
                    "gameplay MTRand oracle ended before all calls: "
                    f"observed={len(gameplay_mtrand_rows)}, "
                    f"expected={expected_gameplay_hits}"
                )
                boundary_failures.append(gameplay_failure)
                assert gameplay_mtrand_sync_receipt is not None
                gameplay_mtrand_sync_receipt["failure"] = (
                    "missing_gameplay_mtrand_calls"
                )
                print(
                    "gameplay_mtrand_sync_failure "
                    f"detail={gameplay_failure}",
                    flush=True,
                )
            else:
                assert gameplay_mtrand_sync_receipt is not None
                gameplay_mtrand_sync_receipt["oracle_complete"] = True
            if (
                gameplay_mtrand_hardware_armed
                and gameplay_mtrand_hardware_original is not None
                and startup_handoff_main_thread_id is not None
                and not exited
            ):
                try:
                    _restore_gameplay_mtrand_breakpoints(
                        main_thread_id=startup_handoff_main_thread_id,
                        original=gameplay_mtrand_hardware_original,
                    )
                    gameplay_mtrand_hardware_armed = False
                    gameplay_mtrand_sync_receipt[
                        "hardware_breakpoint_restored"
                    ] = True
                except Exception as error:
                    gameplay_mtrand_hardware_restore_error = (
                        f"{type(error).__name__}: {error}"
                    )
                    gameplay_mtrand_sync_receipt[
                        "hardware_breakpoint_restore_error"
                    ] = gameplay_mtrand_hardware_restore_error
                    gameplay_mtrand_sync_receipt["failure"] = (
                        "hardware_breakpoint_restore_failed"
                    )
                    boundary_failures.append(
                        "gameplay MTRand hardware breakpoint restore "
                        f"failed: {gameplay_mtrand_hardware_restore_error}"
                    )
            elif exited:
                gameplay_mtrand_hardware_armed = False
                gameplay_mtrand_sync_receipt[
                    "hardware_breakpoint_restored"
                ] = True
            gameplay_mtrand_sync_receipt[
                "hardware_breakpoint_armed"
            ] = gameplay_mtrand_hardware_armed
            if (
                gameplay_mtrand_sync_receipt.get("failure") is None
                and gameplay_mtrand_sync_receipt.get("oracle_complete")
                is True
                and gameplay_mtrand_sync_receipt.get(
                    "hardware_breakpoint_restored"
                )
                is True
            ):
                gameplay_mtrand_sync_receipt["status"] = "PASS"
            else:
                gameplay_mtrand_sync_receipt["status"] = "FAIL"
        if (
            initial_global_mtrand_oracle is not None
            and (
                len(initial_global_rows) != 1
                or initial_global_rows[0].get("atomic_window_completed")
                is not True
                or initial_global_rows[0].get("call_return_observed")
                is not True
                or initial_global_rows[0].get(
                    "other_threads_resumed_after_call_return"
                )
                is not True
            )
        ):
            boundary_failures.append(
                "initial global MTRand source call was not completed"
            )
        print(
            f"result hits={hits} exited={exited} "
            f"brokered_blocks={brokered_blocks} "
            f"broker_bypassed_blocks={broker_bypassed_blocks} "
            "startup_post_bypass_worker_yields="
            f"{len(startup_post_bypass_worker_yields)} "
            f"broker_failures={len(broker_failures)} "
            f"boundary_failures={len(boundary_failures)} "
            "access_denied_thread_skip_events="
            f"{len(access_denied_thread_skip_events)} "
            f"last_update={last_snapshot_update} "
            f"stopped_at_update={stopped_at_update} "
            "stopped_at_command_order="
            f"{stopped_at_command_order} "
            f"command_order_offset={command_order_offset} "
            f"native_timeline_offset={native_timeline_offset} "
            "blackout_native_timeline_offset="
            f"{blackout_native_timeline_offset} "
            "command_order_rebase_rows="
            f"{len(command_order_rebase_rows)} "
            "blackout_file_write_rebase_candidate_rows="
            f"{len(blackout_file_write_rebase_candidate_rows)} "
            "successful_file_write_padding_rows="
            f"{len(diagnostic_successful_file_write_padding_rows)} "
            "service_consumed_file_write_debt_rows="
            f"{len(service_consumed_file_write_debt_rows)} "
            "service_consumed_file_write_surplus_rows="
            f"{len(service_consumed_file_write_surplus_rows)} "
            "effective_deferred_file_write_debt_rows="
            f"{len(effective_debt_rows)} "
            "successful_file_write_debt_barriers="
            f"{len(successful_file_write_debt_barriers)} "
            "terminal_registry_write_eof_exits="
            f"{len(terminal_registry_write_eof_exits)} "
            "service_exit_timeline_suffix_rebases="
            f"{len(service_exit_timeline_suffix_rebases)} "
            "attach_stabilization_verified="
            f"{attach_stabilization_verified} "
            "attach_stabilization_skipped_hits="
            f"{attach_stabilization_skipped_hits} "
            "service_exit_timeline_commits="
            f"{len(service_exit_timeline_commits)} "
            "service_continuation_verifications="
            f"{len(service_continuation_verifications)} "
            "service_exit_overdue_idle_reentries="
            f"{len(service_exit_overdue_idle_reentries)} "
            "preloading_failed_file_write_tail_short_header_recoveries="
            f"{len(preloading_failed_file_write_tail_short_header_recoveries)} "
            "main_file_read_corridor_native_replays="
            f"{len(main_file_read_corridor_native_replays)} "
            "deferred_file_write_discharges="
            f"{len(deferred_file_write_discharges)} "
            "terminal_file_write_payload_handoffs="
            f"{len(terminal_file_write_payload_handoffs)} "
            "service_file_write_header_claims="
            f"{len(service_file_write_header_claims)} "
            "direct_font_cache_file_write_observations="
            f"{len(direct_font_cache_file_write_observations)} "
            "font_cache_manifest_completion_discharges="
            f"{len(font_cache_manifest_completion_discharges)} "
            "terminal_close_requests="
            f"{len(terminal_close_requests)} "
            "debug_exception_observations="
            f"{len(debug_exception_observations)}",
            flush=True,
        )
        preserve_handoff_suspend = handoff_main_thread_suspended
        return TraceResult(
            hits=hits,
            exited=exited,
            exit_code=exit_code,
            brokered_blocks=brokered_blocks,
            broker_bypassed_blocks=broker_bypassed_blocks,
            startup_post_bypass_worker_yields=tuple(
                startup_post_bypass_worker_yields
            ),
            broker_failures=tuple(broker_failures),
            boundary_failures=tuple(boundary_failures),
            last_update=last_snapshot_update,
            stopped_at_update=stopped_at_update,
            stopped_at_command_order=stopped_at_command_order,
            access_denied_thread_skip_events=tuple(
                access_denied_thread_skip_events
            ),
            handoff_main_thread_suspended=(
                handoff_main_thread_suspended
            ),
            handoff_main_thread_id=handoff_main_thread_id,
            handoff_previous_suspend_count=(
                handoff_previous_suspend_count
            ),
            first_command_order=first_command_order,
            last_command_order=last_command_order,
            command_order_offset=command_order_offset,
            command_order_rebase_rows=command_order_rebase_rows,
            blackout_file_write_rebase_candidate_rows=(
                blackout_file_write_rebase_candidate_rows
            ),
            blackout_native_timeline_offset=(
                blackout_native_timeline_offset
            ),
            diagnostic_successful_file_write_padding_rows=(
                diagnostic_successful_file_write_padding_rows
            ),
            service_consumed_file_write_debt_rows=(
                service_consumed_file_write_debt_rows
            ),
            service_consumed_file_write_surplus_rows=(
                service_consumed_file_write_surplus_rows
            ),
            effective_deferred_file_write_debt_rows=(
                effective_debt_rows
            ),
            successful_file_write_debt_barriers=tuple(
                successful_file_write_debt_barriers
            ),
            terminal_registry_write_eof_exits=tuple(
                terminal_registry_write_eof_exits
            ),
            service_exit_timeline_suffix_rebases=tuple(
                service_exit_timeline_suffix_rebases
            ),
            attach_stabilization_verified=(
                attach_stabilization_verified
            ),
            attach_stabilization_skipped_hits=(
                attach_stabilization_skipped_hits
            ),
            attach_stabilization_start_update=(
                attach_stabilization_after_update
            ),
            attach_stabilization_verified_update=(
                attach_stabilization_verified_update
            ),
            attach_stabilization_verified_command_order=(
                attach_stabilization_verified_command_order
            ),
            attach_stabilization_observations=tuple(
                attach_stabilization_observations
            ),
            service_exit_timeline_commits=tuple(
                service_exit_timeline_commits
            ),
            service_continuation_verifications=tuple(
                service_continuation_verifications
            ),
            service_exit_overdue_idle_reentries=tuple(
                service_exit_overdue_idle_reentries
            ),
            preloading_failed_file_write_tail_short_header_recoveries=tuple(
                preloading_failed_file_write_tail_short_header_recoveries
            ),
            main_file_read_corridor_native_replays=tuple(
                main_file_read_corridor_native_replays
            ),
            deferred_file_write_discharges=tuple(
                deferred_file_write_discharges
            ),
            terminal_file_write_payload_handoffs=tuple(
                terminal_file_write_payload_handoffs
            ),
            service_file_write_header_claims=tuple(
                service_file_write_header_claims
            ),
            direct_font_cache_file_write_observations=tuple(
                direct_font_cache_file_write_observations
            ),
            pre_attach_file_write_debt_rows=(
                pre_attach_file_write_debt_rows
            ),
            font_cache_manifest_completion_discharges=tuple(
                font_cache_manifest_completion_discharges
            ),
            font_cache_manifest_entry_count=(
                len(font_cache_manifest)
                if font_cache_manifest is not None
                else 0
            ),
            font_cache_manifest_sha256=font_cache_manifest_sha256,
            font_cache_manifest_main_pak_sha256=(
                font_cache_manifest_main_pak_sha256
            ),
            startup_handoff_main_thread_id=(
                startup_handoff_main_thread_id
            ),
            startup_handoff_launcher_suspend_previous_count=(
                startup_handoff_launcher_suspend_previous_count
            ),
            startup_handoff_tracer_resume_previous_count=(
                startup_handoff_tracer_resume_previous_count
            ),
            startup_handoff_resumed_after_breakpoints=(
                startup_handoff_resumed_after_breakpoints
            ),
            startup_handoff_resumed_perf_counter_ns=(
                startup_handoff_resumed_perf_counter_ns
            ),
            terminal_close_requests=tuple(terminal_close_requests),
            debug_exception_observations=tuple(
                debug_exception_observations
            ),
        )
    finally:
        if (
            startup_priority_bias is not None
            and startup_priority_bias.restored_at_update is None
        ):
            try:
                _restore_in_trace_startup_priority_bias(
                    startup_priority_bias,
                    framework_update=last_snapshot_update,
                )
            except (OSError, RuntimeError):
                pass
        if stepping_suspended_threads:
            try:
                _resume_suspended_threads(
                    stepping_suspended_threads
                )
            except OSError:
                pass
        if source_bound_precall_suspended_threads:
            try:
                _resume_suspended_threads(
                    source_bound_precall_suspended_threads
                )
            except OSError:
                pass
        if initial_global_suspended_threads:
            try:
                _resume_suspended_threads(
                    initial_global_suspended_threads
                )
            except OSError:
                pass
        if held_thread:
            try:
                kernel32.ResumeThread(held_thread)
            finally:
                kernel32.CloseHandle(held_thread)
        if post_bypass_worker_yield_thread:
            try:
                kernel32.ResumeThread(
                    post_bypass_worker_yield_thread
                )
            finally:
                kernel32.CloseHandle(
                    post_bypass_worker_yield_thread
                )
        if (
            startup_handoff_requested
            and not startup_handoff_resumed_after_breakpoints
            and startup_handoff_main_thread_id is not None
        ):
            startup_thread = kernel32.OpenThread(
                THREAD_ACCESS | THREAD_SUSPEND_RESUME,
                False,
                startup_handoff_main_thread_id,
            )
            if startup_thread:
                try:
                    kernel32.ResumeThread(startup_thread)
                finally:
                    kernel32.CloseHandle(startup_thread)
        if (
            handoff_main_thread_suspended
            and not preserve_handoff_suspend
            and handoff_main_thread_id is not None
        ):
            handoff_thread = kernel32.OpenThread(
                THREAD_ACCESS | THREAD_SUSPEND_RESUME,
                False,
                handoff_main_thread_id,
            )
            if handoff_thread:
                try:
                    kernel32.ResumeThread(handoff_thread)
                finally:
                    kernel32.CloseHandle(handoff_thread)
        if (
            startup_global_mtrand_hardware_armed
            and startup_global_mtrand_hardware_original is not None
            and startup_handoff_main_thread_id is not None
            and not exited
        ):
            try:
                _restore_gameplay_mtrand_breakpoints(
                    main_thread_id=startup_handoff_main_thread_id,
                    original=startup_global_mtrand_hardware_original,
                )
                startup_global_mtrand_hardware_armed = False
                if startup_global_mtrand_observation_receipt is not None:
                    startup_global_mtrand_observation_receipt[
                        "hardware_breakpoint_armed"
                    ] = False
                    startup_global_mtrand_observation_receipt[
                        "hardware_breakpoint_restored"
                    ] = True
            except Exception as error:
                startup_global_mtrand_hardware_restore_error = (
                    f"{type(error).__name__}: {error}"
                )
                if startup_global_mtrand_observation_receipt is not None:
                    startup_global_mtrand_observation_receipt[
                        "hardware_breakpoint_restore_error"
                    ] = startup_global_mtrand_hardware_restore_error
                    startup_global_mtrand_observation_receipt[
                        "status"
                    ] = "FAIL"
                    startup_global_mtrand_observation_receipt["failure"] = (
                        "hardware_breakpoint_restore_failed"
                    )
        if (
            gameplay_mtrand_hardware_armed
            and gameplay_mtrand_hardware_original is not None
            and startup_handoff_main_thread_id is not None
            and not exited
        ):
            try:
                _restore_gameplay_mtrand_breakpoints(
                    main_thread_id=startup_handoff_main_thread_id,
                    original=gameplay_mtrand_hardware_original,
                )
                gameplay_mtrand_hardware_armed = False
                if gameplay_mtrand_sync_receipt is not None:
                    gameplay_mtrand_sync_receipt[
                        "hardware_breakpoint_armed"
                    ] = False
                    gameplay_mtrand_sync_receipt[
                        "hardware_breakpoint_restored"
                    ] = True
            except Exception as error:
                gameplay_mtrand_hardware_restore_error = (
                    f"{type(error).__name__}: {error}"
                )
                if gameplay_mtrand_sync_receipt is not None:
                    gameplay_mtrand_sync_receipt[
                        "hardware_breakpoint_restore_error"
                    ] = gameplay_mtrand_hardware_restore_error
                    gameplay_mtrand_sync_receipt["status"] = "FAIL"
                    gameplay_mtrand_sync_receipt["failure"] = (
                        "hardware_breakpoint_restore_failed"
                    )
        if process:
            if (
                post_bypass_return_breakpoint is not None
                and post_bypass_return_breakpoint_original is not None
            ):
                try:
                    return_address = post_bypass_return_breakpoint[
                        "return_address"
                    ]
                    write_memory(
                        process,
                        return_address,
                        post_bypass_return_breakpoint_original,
                    )
                    kernel32.FlushInstructionCache(
                        process,
                        ctypes.c_void_p(return_address),
                        1,
                    )
                except OSError:
                    pass
            if original is not None and breakpoint_armed:
                try:
                    write_memory(process, address, original)
                    kernel32.FlushInstructionCache(
                        process, ctypes.c_void_p(address), 1
                    )
                except OSError:
                    pass
            if (
                initial_global_mtrand_oracle is not None
                and initial_global_original is not None
                and initial_global_breakpoint_armed
            ):
                try:
                    write_memory(
                        process,
                        initial_global_mtrand_oracle.wrapper_address,
                        initial_global_original[:1],
                    )
                    kernel32.FlushInstructionCache(
                        process,
                        ctypes.c_void_p(
                            initial_global_mtrand_oracle.wrapper_address
                        ),
                        1,
                    )
                except OSError:
                    pass
            if (
                initial_global_mtrand_oracle is not None
                and initial_global_return_original is not None
                and initial_global_return_breakpoint_armed
            ):
                try:
                    write_memory(
                        process,
                        initial_global_mtrand_oracle.caller,
                        initial_global_return_original,
                    )
                    kernel32.FlushInstructionCache(
                        process,
                        ctypes.c_void_p(
                            initial_global_mtrand_oracle.caller
                        ),
                        1,
                    )
                except OSError:
                    pass
            if (
                board_seed_address is not None
                and board_original is not None
                and board_breakpoint_armed
            ):
                try:
                    write_memory(
                        process,
                        board_seed_address,
                        board_original,
                    )
                    kernel32.FlushInstructionCache(
                        process,
                        ctypes.c_void_p(board_seed_address),
                        1,
                    )
                except OSError:
                    pass
            if (
                source_bound_precall_original is not None
                and source_bound_precall_breakpoint_armed
            ):
                try:
                    write_memory(
                        process,
                        SOURCE_BOUND_GLOBAL_PRECALL_ADDRESS,
                        source_bound_precall_original[:1],
                    )
                    kernel32.FlushInstructionCache(
                        process,
                        ctypes.c_void_p(
                            SOURCE_BOUND_GLOBAL_PRECALL_ADDRESS
                        ),
                        1,
                    )
                except OSError:
                    pass
            kernel32.CloseHandle(process)
        if attached and not exited:
            kernel32.DebugActiveProcessStop(pid)


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


def _write_result_json(
    path: Path,
    *,
    dmo: Path,
    runtime_exe: Path,
    runtime_process_id: int,
    started_utc: str,
    finished_utc: str,
    trace_started_perf_counter_ns: int,
    trace_finished_perf_counter_ns: int,
    args: argparse.Namespace,
    result: TraceResult,
    trace_phases: Iterable[Mapping[str, Any]] = (),
    debugger_blackout: Mapping[str, Any] | None = None,
    board_seed_phase: Mapping[str, Any] | None = None,
    startup_priority_phase: Mapping[str, Any] | None = None,
    startup_rng: (
        FixedSeedLaunchEvidence
        | FixedSeedIatLaunchEvidence
        | NaturalSeedLaunchEvidence
        | None
    ) = None,
    board_rng_observations: Iterable[Mapping[str, Any]] = (),
    source_bound_board_observations: (
        Iterable[Mapping[str, Any]]
    ) = (),
    gameplay_mtrand_sync_observations: (
        Iterable[Mapping[str, Any]]
    ) = (),
    gameplay_mtrand_sync_receipt: Mapping[str, Any] | None = None,
    initial_global_mtrand_observations: (
        Iterable[Mapping[str, Any]]
    ) = (),
    startup_global_mtrand_observations: (
        Iterable[Mapping[str, Any]]
    ) = (),
    startup_global_mtrand_hidden_draw_observations: (
        Iterable[Mapping[str, Any]]
    ) = (),
    startup_global_mtrand_observation_receipt: (
        Mapping[str, Any] | None
    ) = None,
    global_mtrand_restore_observations: (
        Iterable[Mapping[str, Any]]
    ) = (),
    qrand_restore_observations: Iterable[Mapping[str, Any]] = (),
    thread_crt_restore_observations: Iterable[Mapping[str, Any]] = (),
) -> str:
    source = dmo.resolve()
    data = source.read_bytes()
    payload = {
        "schema": "zuma.popcap_strict_replay.v3",
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "trace_started_perf_counter_ns": trace_started_perf_counter_ns,
        "trace_finished_perf_counter_ns": trace_finished_perf_counter_ns,
        "source_dmo": {
            "path": str(source),
            "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        },
        "runtime_executable": str(runtime_exe.resolve()),
        "runtime_process_id": runtime_process_id,
        "options": {
            "address": args.address,
            "maximum_hits": args.maximum_hits,
            "attach_at_update": args.attach_at_update,
            "startup_trace_handoff": args.startup_trace_handoff,
            "allow_attach_stabilization": (
                args.allow_attach_stabilization
            ),
            "allow_pre_attach_file_write_debt": (
                args.allow_pre_attach_file_write_debt
            ),
            "allow_font_cache_manifest_completion_debt": (
                args.allow_font_cache_manifest_completion_debt
            ),
            "attach_stabilization_max_skipped_hits": (
                ATTACH_STABILIZATION_MAX_SKIPPED_HITS
            ),
            "attach_stabilization_max_update_delta": (
                ATTACH_STABILIZATION_MAX_UPDATE_DELTA
            ),
            "detach_at_update": args.detach_at_update,
            "reattach_at_update": args.reattach_at_update,
            "allow_blackout_file_write_order_rebase": (
                args.allow_blackout_file_write_order_rebase
            ),
            "blackout_expected_command_order_offset": (
                args.blackout_expected_command_order_offset
            ),
            "blackout_expected_native_timeline_offset": (
                args.blackout_expected_native_timeline_offset
            ),
            "blackout_allowed_offset_pairs": [
                {
                    "command_order_offset": command_offset,
                    "native_timeline_offset": timeline_offset,
                }
                for command_offset, timeline_offset
                in args.blackout_allowed_offset_pairs
            ],
            "allow_post_blackout_overdue_idle_reentry": (
                args.allow_post_blackout_overdue_idle_reentry
            ),
            "post_blackout_attach_stabilization": (
                args.allow_post_blackout_attach_stabilization
                or args.allow_blackout_file_write_order_rebase
                or args.allow_post_blackout_overdue_idle_reentry
            ),
            "direct_runtime": args.direct_runtime_exe is not None,
            "direct_natural_seed": args.direct_natural_seed,
            "crt_rand_seed": args.crt_rand_seed,
            "startup_seed_transport": (
                None
                if args.direct_natural_seed
                else args.startup_seed_transport
            ),
            "rng_seed_override_count": (
                0 if args.direct_natural_seed else None
            ),
            "board_seed_address": args.board_seed_address,
            "board_seed": args.board_seed,
            "board_seed_skip_count": args.board_seed_skip_count,
            "global_rng_seed": args.global_rng_seed,
            "initial_global_mtrand_oracle": (
                str(args.initial_global_mtrand_oracle.resolve())
                if args.initial_global_mtrand_oracle is not None
                else None
            ),
            "initial_global_mtrand_seed": (
                args.initial_global_mtrand_seed
            ),
            "initial_global_mtrand_receipt_json": (
                str(args.initial_global_mtrand_receipt_json.resolve())
                if args.initial_global_mtrand_receipt_json is not None
                else None
            ),
            "startup_global_mtrand_observation_oracle": (
                str(
                    args.startup_global_mtrand_observation_oracle.resolve()
                )
                if args.startup_global_mtrand_observation_oracle is not None
                else None
            ),
            "startup_global_mtrand_observation_seed": (
                args.startup_global_mtrand_observation_seed
            ),
            "startup_global_mtrand_observation_end_at_update": (
                args.startup_global_mtrand_observation_end_at_update
            ),
            "startup_global_mtrand_observation_maximum_draws": (
                args.startup_global_mtrand_observation_maximum_draws
            ),
            "startup_global_mtrand_observation_receipt_json": (
                str(
                    args.startup_global_mtrand_observation_receipt_json
                    .resolve()
                )
                if args.startup_global_mtrand_observation_receipt_json
                is not None
                else None
            ),
            "startup_global_mtrand_hidden_draw_start_after_source_order": (
                args.startup_global_mtrand_hidden_draw_start_after_source_order
            ),
            "startup_global_mtrand_hidden_draw_stop_before_source_order": (
                args.startup_global_mtrand_hidden_draw_stop_before_source_order
            ),
            "thread_crt_rng_seed": args.thread_crt_rng_seed,
            "source_bound_board_anchor": (
                args.source_bound_board_monitor is not None
            ),
            "allow_source_bound_board_global_correction": (
                args.allow_source_bound_board_global_correction
            ),
            "source_bound_board_precall_global_restore": (
                args.source_bound_board_precall_global_restore
            ),
            "source_bound_board_monitor_update": (
                args.source_bound_board_monitor_update
            ),
            "source_bound_board_seed": args.source_bound_board_seed,
            "source_bound_board_rewind_draws": (
                args.source_bound_board_rewind_draws
            ),
            "source_bound_board_source_order": (
                args.source_bound_board_source_order
            ),
            "source_bound_board_framework_update": (
                args.source_bound_board_framework_update
            ),
            "source_bound_board_caller": (
                args.source_bound_board_caller
            ),
            "gameplay_mtrand_oracle": (
                str(args.gameplay_mtrand_oracle.resolve())
                if args.gameplay_mtrand_oracle is not None
                else None
            ),
            "gameplay_mtrand_oracle_seed": (
                args.gameplay_mtrand_oracle_seed
            ),
            "gameplay_mtrand_oracle_maximum_draws": (
                args.gameplay_mtrand_oracle_maximum_draws
            ),
            "gameplay_mtrand_receipt_json": (
                str(args.gameplay_mtrand_receipt_json.resolve())
                if args.gameplay_mtrand_receipt_json is not None
                else None
            ),
            "global_mtrand_restore_log": (
                str(args.global_mtrand_restore_log.resolve())
                if args.global_mtrand_restore_log is not None
                else None
            ),
            "global_mtrand_restore_source_update": (
                args.global_mtrand_restore_source_update
            ),
            "global_mtrand_expected_before_log": (
                str(args.global_mtrand_expected_before_log.resolve())
                if args.global_mtrand_expected_before_log is not None
                else None
            ),
            "global_mtrand_expected_before_source_update": (
                args.global_mtrand_expected_before_source_update
            ),
            "global_mtrand_restore_seed": (
                args.global_mtrand_restore_seed
            ),
            "global_mtrand_restore_rewind_draws": (
                args.global_mtrand_restore_rewind_draws
            ),
            "global_mtrand_restore_maximum_draws": (
                args.global_mtrand_restore_maximum_draws
            ),
            "global_mtrand_restore_command_order": (
                args.global_mtrand_restore_command_order
            ),
            "global_mtrand_allow_dynamic_before": (
                args.global_mtrand_allow_dynamic_before
            ),
            "qrand_restore_log": (
                str(args.qrand_restore_log.resolve())
                if args.qrand_restore_log is not None
                else None
            ),
            "qrand_restore_source_update": (
                args.qrand_restore_source_update
            ),
            "qrand_expected_before_log": (
                str(args.qrand_expected_before_log.resolve())
                if args.qrand_expected_before_log is not None
                else None
            ),
            "qrand_expected_before_source_update": (
                args.qrand_expected_before_source_update
            ),
            "qrand_restore_command_order": (
                args.qrand_restore_command_order
            ),
            "thread_crt_restore_log": (
                str(args.thread_crt_restore_log.resolve())
                if args.thread_crt_restore_log is not None
                else None
            ),
            "thread_crt_restore_source_update": (
                args.thread_crt_restore_source_update
            ),
            "thread_crt_expected_before_log": (
                str(args.thread_crt_expected_before_log.resolve())
                if args.thread_crt_expected_before_log is not None
                else None
            ),
            "thread_crt_expected_before_source_update": (
                args.thread_crt_expected_before_source_update
            ),
            "thread_crt_restore_command_order": (
                args.thread_crt_restore_command_order
            ),
            "thread_crt_restore_rewind_draws": (
                args.thread_crt_restore_rewind_draws
            ),
            "thread_crt_allow_dynamic_before": (
                args.thread_crt_allow_dynamic_before
            ),
            "seed_board_before_attach": args.seed_board_before_attach,
            "startup_priority_bias_until_update": (
                args.startup_priority_bias_until_update
            ),
            "startup_process_affinity_mask": (
                args.startup_process_affinity_mask
            ),
            "broker_service_blocks": args.broker_service_blocks,
            "diagnostic_successful_file_write_padding_rows": (
                args.diagnostic_successful_file_write_padding_rows
            ),
            "service_wait_timeout": args.service_wait_timeout,
            "quiet_nonservice": args.quiet_nonservice,
            "stop_after_update": args.stop_after_update,
            "stop_after_command_order": args.stop_after_command_order,
            "close_after_terminal_command": (
                args.close_after_terminal_command
            ),
            "progress_every_updates": args.progress_every_updates,
            "allow_pre_stream_commands": (
                args.allow_pre_stream_commands
            ),
            "accept_normal_exit": args.accept_normal_exit,
            "suspend_main_thread_on_stop": (
                args.suspend_main_thread_on_stop
            ),
        },
        "result": {
            **asdict(result),
            "failure_count": result.failure_count,
        },
    }
    phases = list(trace_phases)
    if phases:
        payload["trace_phases"] = phases
    if debugger_blackout is not None:
        payload["debugger_blackout"] = dict(debugger_blackout)
    if board_seed_phase is not None:
        payload["board_seed_phase"] = dict(board_seed_phase)
    if startup_priority_phase is not None:
        payload["startup_priority_phase"] = dict(
            startup_priority_phase
        )
    if startup_rng is not None:
        payload["startup_rng"] = startup_rng.to_dict()
    if args.board_seed is not None:
        payload["board_rng"] = {
            "call_site_address": args.board_seed_address,
            "effective_seed": args.board_seed,
            "skip_count": args.board_seed_skip_count,
            "global_rng_seed": args.global_rng_seed,
            "thread_crt_rng_seed": args.thread_crt_rng_seed,
            "original_instruction_hex": "e8",
            "register_override": (
                "ecx_seed_before_mtrand_srand_call"
            ),
            "persistent_file_modified": False,
            "observations": [
                dict(observation)
                for observation in board_rng_observations
            ],
        }
    initial_global_rows = [
        dict(observation)
        for observation in initial_global_mtrand_observations
    ]
    if initial_global_rows:
        payload["initial_global_mtrand_seed"] = {
            "mechanism": (
                "one_shot_source_order_zero_precall_state_restore"
            ),
            "process_memory_mutation": any(
                bool(row.get("process_memory_mutation"))
                for row in initial_global_rows
            ),
            "process_context_mutation": True,
            "persistent_file_modified": False,
            "observations": initial_global_rows,
        }
    startup_global_rows = [
        dict(observation)
        for observation in startup_global_mtrand_observations
    ]
    startup_hidden_draw_rows = [
        dict(observation)
        for observation in startup_global_mtrand_hidden_draw_observations
    ]
    if startup_global_mtrand_observation_receipt is not None:
        startup_receipt = dict(
            startup_global_mtrand_observation_receipt
        )
        payload["startup_global_mtrand_observation"] = {
            "schema": "zuma.popcap_startup_global_mtrand_observation.v1",
            "classification": (
                "diagnostic_read_only_startup_rng_schedule_observation"
            ),
            "mechanism": startup_receipt.get("mechanism"),
            "process_memory_writes": 0,
            "process_memory_mutation": False,
            "process_context_mutation": bool(
                startup_receipt.get("process_context_mutation")
            ),
            "persistent_file_modified": False,
            "receipt": startup_receipt,
            "observations": startup_global_rows,
            "hidden_draw_observations": startup_hidden_draw_rows,
        }
    source_bound_board_rows = [
        dict(observation)
        for observation in source_bound_board_observations
    ]
    if source_bound_board_rows:
        payload["source_bound_board_anchor"] = {
            "mechanism": (
                "exact_post_global_call_board_constructor_seed_and_"
                "main_thread_crt_anchor"
            ),
            "process_memory_mutation": any(
                bool(row.get("process_memory_mutation"))
                for row in source_bound_board_rows
            ),
            "persistent_file_modified": False,
            "observations": source_bound_board_rows,
        }
    gameplay_mtrand_rows = [
        dict(observation)
        for observation in gameplay_mtrand_sync_observations
    ]
    if gameplay_mtrand_sync_receipt is not None:
        receipt = dict(gameplay_mtrand_sync_receipt)
        payload["gameplay_mtrand_sync"] = {
            "schema": "zuma.popcap_gameplay_mtrand_sync.v1",
            "classification": (
                "diagnostic_ephemeral_process_state_synchronization"
            ),
            "mechanism": receipt.get("mechanism"),
            "process_memory_mutation": bool(
                receipt.get("process_memory_mutation")
            ),
            "process_context_mutation": bool(
                receipt.get("process_context_mutation")
            ),
            "persistent_file_modified": False,
            "receipt": receipt,
            "observations": gameplay_mtrand_rows,
        }
    global_mtrand_rows = [
        dict(observation)
        for observation in global_mtrand_restore_observations
    ]
    if global_mtrand_rows:
        payload["global_mtrand_restore"] = {
            "mechanism": (
                "transactional_live_global_mtrand_state_restore"
            ),
            "process_memory_mutation": True,
            "persistent_file_modified": False,
            "observations": global_mtrand_rows,
        }
    qrand_rows = [
        dict(observation)
        for observation in qrand_restore_observations
    ]
    if qrand_rows:
        payload["qrand_restore"] = {
            "mechanism": (
                "transactional_live_qrand_semantic_state_restore"
            ),
            "process_memory_mutation": True,
            "persistent_file_modified": False,
            "observations": qrand_rows,
        }
    thread_crt_rows = [
        dict(observation)
        for observation in thread_crt_restore_observations
    ]
    if thread_crt_rows:
        payload["thread_crt_restore"] = {
            "mechanism": (
                "transactional_live_main_thread_crt_rand_state_restore"
            ),
            "process_memory_mutation": True,
            "persistent_file_modified": False,
            "observations": thread_crt_rows,
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    ) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(serialized)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _write_gameplay_mtrand_sync_receipt_json(
    path: Path,
    *,
    dmo: Path,
    runtime_exe: Path,
    runtime_process_id: int,
    started_perf_counter_ns: int,
    finished_perf_counter_ns: int,
    receipt: Mapping[str, Any],
    observations: Iterable[Mapping[str, Any]],
    result: TraceResult,
    hidden_draw_observations: Iterable[Mapping[str, Any]] = (),
) -> str:
    """Persist the completed pre-blackout sync before later replay capture."""

    source = dmo.resolve()
    source_data = source.read_bytes()
    receipt_row = dict(receipt)
    observation_rows = [dict(row) for row in observations]
    expected_hits = receipt_row.get("expected_hit_count")
    passed = (
        result.failure_count == 0
        and result.stopped_at_update
        and receipt_row.get("status") == "PASS"
        and receipt_row.get("oracle_complete") is True
        and receipt_row.get("hardware_breakpoint_restored") is True
        and receipt_row.get("hardware_breakpoint_restore_error") is None
        and receipt_row.get("hit_count") == expected_hits
        and len(observation_rows) == expected_hits
    )
    payload = {
        "schema": "zuma.popcap_gameplay_mtrand_sync_receipt.v1",
        "status": "PASS" if passed else "FAIL",
        "classification": (
            "diagnostic_ephemeral_process_state_synchronization"
        ),
        "started_perf_counter_ns": started_perf_counter_ns,
        "finished_perf_counter_ns": finished_perf_counter_ns,
        "runtime_process_id": runtime_process_id,
        "runtime_executable": str(runtime_exe.resolve()),
        "source_dmo": {
            "path": str(source),
            "size_bytes": len(source_data),
            "sha256": _sha256_bytes(source_data),
        },
        "process_memory_mutation": bool(
            receipt_row.get("process_memory_mutation")
        ),
        "process_context_mutation": bool(
            receipt_row.get("process_context_mutation")
        ),
        "persistent_file_modified": False,
        "receipt": receipt_row,
        "observations": observation_rows,
        "pre_blackout_trace_result": {
            **asdict(result),
            "failure_count": result.failure_count,
        },
    }
    serialized = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    ) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(serialized)
    return _sha256_bytes(serialized.encode("utf-8"))


def _write_initial_global_mtrand_receipt_json(
    path: Path,
    *,
    dmo: Path,
    runtime_exe: Path,
    runtime_process_id: int,
    started_perf_counter_ns: int,
    finished_perf_counter_ns: int,
    oracle: InitialGlobalMTRandCallOracle,
    observations: Iterable[Mapping[str, Any]],
    result: TraceResult,
) -> str:
    """Persist the one-shot source-bound startup seed before blackout."""

    source = dmo.resolve()
    source_data = source.read_bytes()
    rows = [dict(row) for row in observations]
    passed = (
        result.failure_count == 0
        and result.stopped_at_update
        and len(rows) == 1
        and rows[0].get("atomic_window_completed") is True
        and rows[0].get("call_target_entered") is True
        and rows[0].get("call_return_observed") is True
        and rows[0].get("other_threads_resumed_after_call_return") is True
        and rows[0].get("observed_output") == oracle.output
        and rows[0].get("restored_pre_state_sha256")
        == oracle.pre_state_sha256
        and rows[0].get("observed_post_state_sha256")
        == oracle.post_state_sha256
        and rows[0].get("expected_post_state_sha256")
        == oracle.post_state_sha256
        and rows[0].get("persistent_file_modified") is False
    )
    payload = {
        "schema": "zuma.popcap_initial_global_mtrand_seed_receipt.v1",
        "status": "PASS" if passed else "FAIL",
        "classification": "diagnostic_source_bound_rng_initialization",
        "started_perf_counter_ns": started_perf_counter_ns,
        "finished_perf_counter_ns": finished_perf_counter_ns,
        "runtime_process_id": runtime_process_id,
        "runtime_executable": str(runtime_exe.resolve()),
        "source_dmo": {
            "path": str(source),
            "size_bytes": len(source_data),
            "sha256": _sha256_bytes(source_data),
        },
        "oracle": {
            "source_path": str(oracle.source_path),
            "source_sha256": oracle.source_sha256,
            "source_process_id": oracle.source_process_id,
            "source_main_thread_id": oracle.source_main_thread_id,
            "runtime_executable_sha256": (
                oracle.runtime_executable_sha256
            ),
            "seed": oracle.seed,
            "semantic_sha256": oracle.semantic_sha256,
            "wrapper_address": oracle.wrapper_address,
            "call_address": oracle.call_address,
            "caller": oracle.caller,
            "output": oracle.output,
            "pre_index": oracle.pre_index,
            "pre_state_sha256": oracle.pre_state_sha256,
            "post_index": oracle.post_index,
            "post_state_sha256": oracle.post_state_sha256,
        },
        "state_correction_count": sum(
            bool(row.get("changed")) for row in rows
        ),
        "bytes_written_total": sum(
            int(row.get("bytes_written", 0)) for row in rows
        ),
        "process_memory_mutation": any(
            bool(row.get("process_memory_mutation")) for row in rows
        ),
        "process_context_mutation": bool(rows),
        "persistent_file_modified": False,
        "observations": rows,
        "pre_blackout_trace_result": {
            **asdict(result),
            "failure_count": result.failure_count,
        },
    }
    serialized = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    ) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(serialized)
    return _sha256_bytes(serialized.encode("utf-8"))


def _write_startup_global_mtrand_observation_receipt_json(
    path: Path,
    *,
    dmo: Path,
    runtime_exe: Path,
    runtime_process_id: int,
    started_perf_counter_ns: int,
    finished_perf_counter_ns: int,
    oracle: StartupGlobalMTRandObservationOracle,
    receipt: Mapping[str, Any],
    observations: Iterable[Mapping[str, Any]],
    result: TraceResult,
    hidden_draw_observations: Iterable[Mapping[str, Any]] = (),
) -> str:
    """Persist a read-only startup wrapper schedule comparison."""

    source = dmo.resolve()
    source_data = source.read_bytes()
    receipt_row = dict(receipt)
    raw_rows = [dict(row) for row in observations]
    raw_hidden_draw_rows = [
        dict(row) for row in hidden_draw_observations
    ]
    state_hashes = {
        str(row[key])
        for row in raw_rows
        for key in (
            "pre_state_sha256",
            "inferred_post_state_sha256",
        )
        if isinstance(row.get(key), str)
    }
    state_hashes.update(
        str(row["post_state_sha256"])
        for row in raw_hidden_draw_rows
        if isinstance(row.get("post_state_sha256"), str)
    )
    reconstruction_failure: str | None = None
    draw_counts: dict[str, int] = {}
    if state_hashes:
        try:
            draw_counts = reconstruct_mtrand_draw_counts(
                seed=oracle.seed,
                state_sha256=state_hashes,
                maximum_draws=oracle.maximum_draws,
            )
        except ValueError as error:
            reconstruction_failure = str(error)
    else:
        reconstruction_failure = "no observed MTRand states"

    rows: list[dict[str, Any]] = []
    previous_observed_post_draw: int | None = None
    previous_expected_post_draw: int | None = None
    for order, raw_row in enumerate(raw_rows):
        row = dict(raw_row)
        expected = oracle.entries[order] if order < len(oracle.entries) else None
        observed_pre_draw = draw_counts.get(
            str(row.get("pre_state_sha256"))
        )
        observed_post_draw = draw_counts.get(
            str(row.get("inferred_post_state_sha256"))
        )
        expected_pre_draw = (
            expected.pre_draw_count if expected is not None else None
        )
        expected_post_draw = (
            expected.post_draw_count if expected is not None else None
        )
        row.update(
            {
                "observed_pre_draw_count": observed_pre_draw,
                "observed_post_draw_count": observed_post_draw,
                "observed_hidden_draws_before_call": (
                    observed_pre_draw
                    - (
                        previous_observed_post_draw
                        if previous_observed_post_draw is not None
                        else 0
                    )
                    if observed_pre_draw is not None
                    else None
                ),
                "expected_hidden_draws_before_call": (
                    expected_pre_draw
                    - (
                        previous_expected_post_draw
                        if previous_expected_post_draw is not None
                        else 0
                    )
                    if expected_pre_draw is not None
                    else None
                ),
                "pre_draw_count_delta": (
                    observed_pre_draw - expected_pre_draw
                    if observed_pre_draw is not None
                    and expected_pre_draw is not None
                    else None
                ),
                "post_draw_count_delta": (
                    observed_post_draw - expected_post_draw
                    if observed_post_draw is not None
                    and expected_post_draw is not None
                    else None
                ),
            }
        )
        if observed_post_draw is not None:
            previous_observed_post_draw = observed_post_draw
        if expected_post_draw is not None:
            previous_expected_post_draw = expected_post_draw
        rows.append(row)

    hidden_rows: list[dict[str, Any]] = []
    for order, raw_row in enumerate(raw_hidden_draw_rows):
        row = dict(raw_row)
        row["observed_post_draw_count"] = draw_counts.get(
            str(row.get("post_state_sha256"))
        )
        row["receipt_order"] = order
        hidden_rows.append(row)

    hidden_contract_raw = receipt_row.get("hidden_draw_interval")
    hidden_contract = (
        dict(hidden_contract_raw)
        if isinstance(hidden_contract_raw, Mapping)
        else None
    )
    hidden_draw_interval: dict[str, Any] | None = None
    hidden_draw_analysis_passed = hidden_contract is None
    if hidden_contract is not None:
        hidden_failures: list[str] = []
        expected_hidden_draw_count = hidden_contract.get(
            "expected_hidden_draw_count"
        )
        start_post_draw_count = hidden_contract.get(
            "start_post_draw_count"
        )
        stop_entry_order = hidden_contract.get("stop_entry_order")
        observed_stop_pre_draw_count: int | None = None
        if (
            isinstance(stop_entry_order, int)
            and 0 <= stop_entry_order < len(rows)
        ):
            candidate = rows[stop_entry_order].get(
                "observed_pre_draw_count"
            )
            if isinstance(candidate, int):
                observed_stop_pre_draw_count = candidate

        hidden_orders_valid = all(
            row.get("order") == order
            for order, row in enumerate(hidden_rows)
        )
        hidden_draw_counts = [
            row.get("observed_post_draw_count") for row in hidden_rows
        ]
        hidden_draw_counts_complete = all(
            isinstance(value, int) for value in hidden_draw_counts
        )
        hidden_draw_counts_contiguous = (
            hidden_draw_counts_complete
            and all(
                int(current) == int(previous) + 1
                for previous, current in zip(
                    hidden_draw_counts,
                    hidden_draw_counts[1:],
                )
            )
        )
        first_observed_post_draw_count = (
            int(hidden_draw_counts[0])
            if hidden_draw_counts
            and isinstance(hidden_draw_counts[0], int)
            else None
        )
        last_observed_post_draw_count = (
            int(hidden_draw_counts[-1])
            if hidden_draw_counts
            and isinstance(hidden_draw_counts[-1], int)
            else None
        )
        observed_hidden_draw_count = max(0, len(hidden_rows) - 1)
        observed_draw_span = (
            observed_stop_pre_draw_count - int(start_post_draw_count)
            if observed_stop_pre_draw_count is not None
            and isinstance(start_post_draw_count, int)
            else None
        )
        missing_hidden_draw_count = (
            int(expected_hidden_draw_count) - observed_hidden_draw_count
            if isinstance(expected_hidden_draw_count, int)
            else None
        )
        if hidden_contract.get("status") != "PASS":
            hidden_failures.append("runtime_boundary_contract_failed")
        if hidden_contract.get("process_memory_writes") != 0:
            hidden_failures.append("runtime_process_memory_writes_nonzero")
        if hidden_contract.get("process_memory_mutation") is not False:
            hidden_failures.append("runtime_process_memory_mutation")
        if not hidden_rows:
            hidden_failures.append("no_index_write_observations")
        if not hidden_orders_valid:
            hidden_failures.append("observation_order_invalid")
        if not hidden_draw_counts_complete:
            hidden_failures.append("draw_count_reconstruction_incomplete")
        elif not hidden_draw_counts_contiguous:
            hidden_failures.append("draw_counts_not_contiguous")
        if first_observed_post_draw_count != start_post_draw_count:
            hidden_failures.append("start_post_draw_count_mismatch")
        if (
            last_observed_post_draw_count is None
            or observed_stop_pre_draw_count is None
            or last_observed_post_draw_count
            != observed_stop_pre_draw_count
        ):
            hidden_failures.append("stop_pre_draw_count_mismatch")
        if observed_draw_span != observed_hidden_draw_count:
            hidden_failures.append("hidden_draw_span_inconsistent")

        caller_groups: dict[tuple[int | None, int | None], dict[str, Any]] = {}
        for row in hidden_rows[1:]:
            stack_return = row.get("stack_return")
            instruction_pointer = row.get("instruction_pointer")
            key = (
                stack_return if isinstance(stack_return, int) else None,
                (
                    instruction_pointer
                    if isinstance(instruction_pointer, int)
                    else None
                ),
            )
            group = caller_groups.setdefault(
                key,
                {
                    "stack_return": key[0],
                    "stack_return_hex": (
                        f"0x{key[0]:08x}" if key[0] is not None else None
                    ),
                    "instruction_pointer": key[1],
                    "instruction_pointer_hex": (
                        f"0x{key[1]:08x}" if key[1] is not None else None
                    ),
                    "count": 0,
                    "first_observation_order": row.get("order"),
                    "last_observation_order": row.get("order"),
                    "first_framework_update": row.get("framework_update"),
                    "last_framework_update": row.get("framework_update"),
                },
            )
            group["count"] += 1
            group["last_observation_order"] = row.get("order")
            group["last_framework_update"] = row.get("framework_update")
        caller_histogram = sorted(
            caller_groups.values(),
            key=lambda row: (
                -int(row["count"]),
                int(row["stack_return"] or 0),
                int(row["instruction_pointer"] or 0),
            ),
        )
        hidden_draw_analysis_passed = not hidden_failures
        hidden_draw_interval = {
            "status": "PASS" if hidden_draw_analysis_passed else "FAIL",
            "failures": hidden_failures,
            "contract": hidden_contract,
            "summary": {
                "expected_hidden_draw_count": expected_hidden_draw_count,
                "observed_hidden_draw_count": observed_hidden_draw_count,
                "missing_hidden_draw_count": missing_hidden_draw_count,
                "deficit_detected": (
                    missing_hidden_draw_count is not None
                    and missing_hidden_draw_count > 0
                ),
                "index_write_count": len(hidden_rows),
                "expected_total_index_write_count": hidden_contract.get(
                    "expected_total_index_write_count"
                ),
                "start_post_draw_count": start_post_draw_count,
                "first_observed_post_draw_count": (
                    first_observed_post_draw_count
                ),
                "expected_stop_pre_draw_count": hidden_contract.get(
                    "expected_stop_pre_draw_count"
                ),
                "observed_stop_pre_draw_count": (
                    observed_stop_pre_draw_count
                ),
                "last_observed_post_draw_count": (
                    last_observed_post_draw_count
                ),
                "observed_draw_span": observed_draw_span,
                "observation_orders_valid": hidden_orders_valid,
                "draw_counts_complete": hidden_draw_counts_complete,
                "draw_counts_contiguous": hidden_draw_counts_contiguous,
            },
            "caller_histogram": caller_histogram,
            "process_memory_writes": 0,
            "process_memory_mutation": False,
            "process_context_mutation": True,
            "persistent_file_modified": False,
            "observations": hidden_rows,
        }

    def first_order(field: str, expected: Any) -> int | None:
        return next(
            (
                int(row["order"])
                for row in rows
                if row.get(field) is expected
            ),
            None,
        )

    first_draw_count_mismatch = next(
        (
            int(row["order"])
            for row in rows
            if row.get("pre_draw_count_delta") not in (None, 0)
        ),
        None,
    )
    expected_hits = len(oracle.entries)
    passed = (
        result.failure_count == 0
        and result.stopped_at_update
        and receipt_row.get("status") == "PASS"
        and receipt_row.get("oracle_complete") is True
        and receipt_row.get("hardware_breakpoint_restored") is True
        and receipt_row.get("hardware_breakpoint_restore_error") is None
        and receipt_row.get("hit_count") == expected_hits
        and len(rows) == expected_hits
        and receipt_row.get("process_memory_writes") == 0
        and receipt_row.get("process_memory_mutation") is False
        and reconstruction_failure is None
        and hidden_draw_analysis_passed
    )
    payload = {
        "schema": (
            "zuma.popcap_startup_global_mtrand_observation_receipt.v1"
        ),
        "status": "PASS" if passed else "FAIL",
        "classification": (
            "diagnostic_read_only_startup_rng_schedule_observation"
        ),
        "started_perf_counter_ns": started_perf_counter_ns,
        "finished_perf_counter_ns": finished_perf_counter_ns,
        "runtime_process_id": runtime_process_id,
        "runtime_executable": str(runtime_exe.resolve()),
        "source_dmo": {
            "path": str(source),
            "size_bytes": len(source_data),
            "sha256": _sha256_bytes(source_data),
        },
        "oracle": {
            "source_path": str(oracle.source_path),
            "source_sha256": oracle.source_sha256,
            "source_process_id": oracle.source_process_id,
            "source_main_thread_id": oracle.source_main_thread_id,
            "runtime_executable_sha256": oracle.runtime_executable_sha256,
            "seed": oracle.seed,
            "wrapper_address": oracle.wrapper_address,
            "end_at_update": oracle.end_at_update,
            "maximum_draws": oracle.maximum_draws,
            "entry_count": expected_hits,
            "semantic_sha256": oracle.semantic_sha256,
        },
        "receipt": receipt_row,
        "draw_count_reconstruction": {
            "status": (
                "PASS" if reconstruction_failure is None else "FAIL"
            ),
            "failure": reconstruction_failure,
            "maximum_draws": oracle.maximum_draws,
            "unique_state_count": len(state_hashes),
            "reconstructed_state_count": len(draw_counts),
        },
        "comparison": {
            "observed_call_count": len(rows),
            "expected_call_count": expected_hits,
            "exact_match_count": sum(
                bool(row.get("exact_match")) for row in rows
            ),
            "mismatch_count": sum(
                row.get("exact_match") is False for row in rows
            ),
            "first_exact_mismatch_order": first_order(
                "exact_match",
                False,
            ),
            "first_caller_mismatch_order": first_order(
                "caller_match",
                False,
            ),
            "first_framework_update_mismatch_order": first_order(
                "framework_update_match",
                False,
            ),
            "first_pre_state_mismatch_order": first_order(
                "pre_state_match",
                False,
            ),
            "first_transition_mismatch_order": first_order(
                "transition_match",
                False,
            ),
            "first_draw_count_mismatch_order": (
                first_draw_count_mismatch
            ),
        },
        "process_memory_writes": 0,
        "process_memory_mutation": False,
        "process_context_mutation": bool(
            receipt_row.get("process_context_mutation")
        ),
        "persistent_file_modified": False,
        "observations": rows,
        "pre_blackout_trace_result": {
            **asdict(result),
            "failure_count": result.failure_count,
        },
    }
    if hidden_draw_interval is not None:
        payload["hidden_draw_interval"] = hidden_draw_interval
    serialized = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    ) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(serialized)
    return _sha256_bytes(serialized.encode("utf-8"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Launch Steam DMO playback and trace early "
            "PrepareDemoCommand callers."
        )
    )
    parser.add_argument("--dmo", required=True, type=Path)
    parser.add_argument("--steam-exe", type=Path, default=DEFAULT_STEAM_EXE)
    parser.add_argument("--runtime-exe", type=Path, default=DEFAULT_RUNTIME_EXE)
    parser.add_argument(
        "--direct-runtime-exe",
        type=Path,
        help=(
            "Launch this extracted byte-identical retail payload directly "
            "instead of asking Steam to extract a temporary copy."
        ),
    )
    parser.add_argument(
        "--changedir",
        type=Path,
        help="Original retail asset directory for a direct runtime launch.",
    )
    parser.add_argument(
        "--crt-rand-seed",
        type=_non_negative_int,
        help=(
            "One-shot EAX replacement at the retail startup srand call; "
            "requires --direct-runtime-exe and --changedir."
        ),
    )
    parser.add_argument(
        "--direct-natural-seed",
        action="store_true",
        help=(
            "Launch the direct payload suspended, observe the retail EAX "
            "seed at srand without replacing it, and hand off gaplessly "
            "to the command tracer."
        ),
    )
    parser.add_argument(
        "--startup-seed-transport",
        choices=("debugger_register", "iat_stub"),
        default="debugger_register",
        help=(
            "Use the legacy startup debugger register override or a "
            "loader-gated, restored GetTickCount IAT return stub."
        ),
    )
    parser.add_argument(
        "--board-seed-address",
        type=_positive_int,
        default=DEFAULT_BOARD_RESEED_CALL,
        help="Retail CALL instruction that seeds the board MTRand.",
    )
    parser.add_argument(
        "--board-seed",
        type=_non_negative_int,
        help=(
            "One-shot ECX seed replacement before the board MTRand "
            "SRand call; requires direct fixed-seed launch."
        ),
    )
    parser.add_argument(
        "--board-seed-skip-count",
        type=_non_negative_int,
        default=0,
        help=(
            "Single-step this many earlier calls at the board reseed site "
            "without overrides, then observe/override the next call."
        ),
    )
    parser.add_argument(
        "--global-rng-seed",
        type=_non_negative_int,
        help=(
            "Reset the global framework MTRand at the board-seed "
            "breakpoint; requires --board-seed."
        ),
    )
    parser.add_argument(
        "--initial-global-mtrand-oracle",
        type=Path,
        help=(
            "Read-only complete global-call trace whose source order zero "
            "binds the initial framework MTRand state."
        ),
    )
    parser.add_argument(
        "--initial-global-mtrand-seed",
        type=_non_negative_int,
        help="Seed used to reconstruct the source order-zero pre-state.",
    )
    parser.add_argument(
        "--initial-global-mtrand-receipt-json",
        type=Path,
        help=(
            "Exclusively create the completed startup MTRand seed receipt "
            "immediately after the pre-blackout trace."
        ),
    )
    parser.add_argument(
        "--startup-global-mtrand-observation-oracle",
        type=Path,
        help=(
            "Read-only complete global-wrapper trace used to compare every "
            "startup call through a fixed framework update."
        ),
    )
    parser.add_argument(
        "--startup-global-mtrand-observation-seed",
        type=_non_negative_int,
        help="Seed used to reconstruct observed startup MTRand draw counts.",
    )
    parser.add_argument(
        "--startup-global-mtrand-observation-end-at-update",
        type=_non_negative_int,
        help="Inclusive source framework-update boundary for observation.",
    )
    parser.add_argument(
        "--startup-global-mtrand-observation-maximum-draws",
        type=_positive_int,
        default=100_000,
        help="Maximum seeded draws used to reconstruct observed states.",
    )
    parser.add_argument(
        "--startup-global-mtrand-observation-receipt-json",
        type=Path,
        help=(
            "Exclusively create the read-only startup call-schedule receipt "
            "immediately after the pre-blackout trace."
        ),
    )
    parser.add_argument(
        "--startup-global-mtrand-hidden-draw-start-after-source-order",
        type=_non_negative_int,
        help=(
            "Arm a read-only write watchpoint before this observed wrapper "
            "call executes; its wrapper draw is the interval boundary."
        ),
    )
    parser.add_argument(
        "--startup-global-mtrand-hidden-draw-stop-before-source-order",
        type=_non_negative_int,
        help=(
            "Disarm the read-only write watchpoint at this next source call "
            "before its wrapper draw executes."
        ),
    )
    parser.add_argument(
        "--thread-crt-rng-seed",
        type=_non_negative_int,
        help=(
            "Reset the paused board thread's CRT rand state at the "
            "board-seed breakpoint; requires --board-seed."
        ),
    )
    parser.add_argument(
        "--source-bound-board-monitor",
        type=Path,
        help=(
            "Read-only natural retail RNG monitor used to reconstruct the "
            "global MTRand state immediately before one source call."
        ),
    )
    parser.add_argument(
        "--source-bound-board-trace",
        type=Path,
        help=(
            "Read-only natural retail hardware call trace that binds the "
            "global and main-thread CRT source states."
        ),
    )
    parser.add_argument(
        "--source-bound-board-recording-report",
        type=Path,
        help="Natural retail recording report bound to the source trace.",
    )
    parser.add_argument(
        "--source-bound-board-monitor-update",
        type=_non_negative_int,
        help="Unique monitor update used for global state reconstruction.",
    )
    parser.add_argument(
        "--source-bound-board-seed",
        type=_non_negative_int,
        help="Natural recording's global MTRand seed.",
    )
    parser.add_argument(
        "--source-bound-board-rewind-draws",
        type=_non_negative_int,
        help="Draws rewound from the monitor row to the exact source call.",
    )
    parser.add_argument(
        "--source-bound-board-source-order",
        type=_non_negative_int,
        help="Exact source hardware-call order used as the anchor.",
    )
    parser.add_argument(
        "--source-bound-board-framework-update",
        type=_non_negative_int,
        help="Exact source framework update at the anchored call.",
    )
    parser.add_argument(
        "--source-bound-board-caller",
        type=_positive_int,
        help="Exact source return address at the anchored global RNG call.",
    )
    parser.add_argument(
        "--allow-source-bound-board-global-correction",
        action="store_true",
        help=(
            "At the exact source-bound board constructor only, permit a "
            "transactional correction when all 624 global MTRand state words "
            "match and only a verified bounded draw index differs."
        ),
    )
    parser.add_argument(
        "--source-bound-board-precall-global-restore",
        action="store_true",
        help=(
            "Restore the bound natural global MTRand state at the exact "
            "seed-producing retail CALL and suspend competing threads until "
            "the paired Board seed call."
        ),
    )
    parser.add_argument(
        "--gameplay-mtrand-oracle",
        type=Path,
        help=(
            "Verified gameplay-return oracle restored by a hardware "
            "breakpoint inside this strict debugger."
        ),
    )
    parser.add_argument(
        "--gameplay-mtrand-oracle-seed",
        type=_non_negative_int,
        help="Global MTRand seed used to reconstruct the gameplay oracle.",
    )
    parser.add_argument(
        "--gameplay-mtrand-oracle-maximum-draws",
        type=_positive_int,
        default=100_000,
        help="Maximum seeded draws used to validate the gameplay oracle.",
    )
    parser.add_argument(
        "--gameplay-mtrand-receipt-json",
        type=Path,
        help=(
            "Exclusively create the completed synchronization receipt before "
            "the debugger blackout."
        ),
    )
    parser.add_argument(
        "--global-mtrand-restore-log",
        type=Path,
        help=(
            "Read-only RNG monitor whose captured global MTRand state is "
            "reconstructed and restored at one exact DMO boundary."
        ),
    )
    parser.add_argument(
        "--global-mtrand-restore-source-update",
        type=_non_negative_int,
        help=(
            "Unique framework update to select from "
            "--global-mtrand-restore-log."
        ),
    )
    parser.add_argument(
        "--global-mtrand-expected-before-log",
        type=Path,
        help=(
            "Read-only RNG monitor containing the reference global MTRand "
            "state before the restore."
        ),
    )
    parser.add_argument(
        "--global-mtrand-expected-before-source-update",
        type=_non_negative_int,
        help=(
            "Unique framework update to select from "
            "--global-mtrand-expected-before-log."
        ),
    )
    parser.add_argument(
        "--global-mtrand-restore-seed",
        type=_non_negative_int,
        help=(
            "Framework MTRand seed used to reconstruct and verify both "
            "captured state hashes."
        ),
    )
    parser.add_argument(
        "--global-mtrand-restore-rewind-draws",
        type=_non_negative_int,
        default=0,
        help=(
            "Reconstruct this many draws before the captured monitor row "
            "when the exact DMO boundary precedes that row."
        ),
    )
    parser.add_argument(
        "--global-mtrand-restore-maximum-draws",
        type=_positive_int,
        default=DEFAULT_MAXIMUM_RECONSTRUCTION_DRAWS,
        help=(
            "Maximum seeded outputs searched while reconstructing a "
            "captured global MTRand state."
        ),
    )
    parser.add_argument(
        "--global-mtrand-restore-command-order",
        type=_non_negative_int,
        help=(
            "Offline DMO command order at which to restore the shared "
            "global MTRand state."
        ),
    )
    parser.add_argument(
        "--global-mtrand-allow-dynamic-before",
        action="store_true",
        help=(
            "Allow the live global MTRand state to vary from the reference "
            "monitor at the validated restore boundary; the exact live "
            "bytes are still captured for transactional rollback."
        ),
    )
    parser.add_argument(
        "--qrand-restore-log",
        type=Path,
        help=(
            "Read-only RNG monitor whose captured QRand state is restored "
            "at one exact DMO command boundary."
        ),
    )
    parser.add_argument(
        "--qrand-restore-source-update",
        type=_non_negative_int,
        help="Unique framework update to select from --qrand-restore-log.",
    )
    parser.add_argument(
        "--qrand-expected-before-log",
        type=Path,
        help=(
            "Read-only RNG monitor containing the exact live QRand state "
            "required before the restore."
        ),
    )
    parser.add_argument(
        "--qrand-expected-before-source-update",
        type=_non_negative_int,
        help=(
            "Unique framework update to select from "
            "--qrand-expected-before-log."
        ),
    )
    parser.add_argument(
        "--qrand-restore-command-order",
        type=_non_negative_int,
        help="Offline DMO command order at which to restore QRand.",
    )
    parser.add_argument(
        "--thread-crt-restore-log",
        type=Path,
        help=(
            "Thread-aware read-only RNG monitor whose captured main-thread "
            "CRT rand state is restored at one exact DMO boundary."
        ),
    )
    parser.add_argument(
        "--thread-crt-restore-source-update",
        type=_non_negative_int,
        help=(
            "Unique framework update to select from "
            "--thread-crt-restore-log."
        ),
    )
    parser.add_argument(
        "--thread-crt-expected-before-log",
        type=Path,
        help=(
            "Thread-aware read-only RNG monitor containing the exact live "
            "CRT rand state required before restore."
        ),
    )
    parser.add_argument(
        "--thread-crt-expected-before-source-update",
        type=_non_negative_int,
        help=(
            "Unique framework update to select from "
            "--thread-crt-expected-before-log."
        ),
    )
    parser.add_argument(
        "--thread-crt-restore-command-order",
        type=_non_negative_int,
        help=(
            "Offline DMO command order at which to restore the main-thread "
            "CRT rand state."
        ),
    )
    parser.add_argument(
        "--thread-crt-restore-rewind-draws",
        type=_non_negative_int,
        default=0,
        help=(
            "Reconstruct this many MSVC rand calls before the captured "
            "monitor row when the exact DMO boundary precedes generation."
        ),
    )
    parser.add_argument(
        "--thread-crt-allow-dynamic-before",
        action="store_true",
        help=(
            "Allow the exact live CRT value to vary from the reference "
            "monitor at the validated restore boundary; the observed value "
            "is still captured and used for transactional rollback."
        ),
    )
    parser.add_argument("--app-id", type=int, default=3620)
    parser.add_argument(
        "--address",
        type=_positive_int,
        default=DEFAULT_PREPARE_DEMO_COMMAND,
    )
    parser.add_argument("--maximum-hits", type=_positive_int, default=32)
    parser.add_argument("--launch-timeout", type=float, default=20.0)
    parser.add_argument("--trace-timeout", type=float, default=20.0)
    parser.add_argument(
        "--attach-at-update",
        type=_non_negative_int,
        default=0,
        help=(
            "leave startup uninstrumented and attach only after this "
            "framework update"
        ),
    )
    parser.add_argument(
        "--startup-trace-handoff",
        action="store_true",
        help=(
            "Keep the direct-launch main thread suspended across startup "
            "debugger detach, arm strict command and board breakpoints in "
            "the successor debugger, then resume it exactly once."
        ),
    )
    parser.add_argument(
        "--allow-attach-stabilization",
        action="store_true",
        help=(
            "After a nonzero late attach, observe only bounded in-flight "
            "PrepareDemoCommand calls until the first exact complete "
            "original-main-thread offline boundary."
        ),
    )
    parser.add_argument(
        "--allow-pre-attach-file-write-debt",
        action="store_true",
        help=(
            "Allow exact long file-write result rows before a nonzero attach "
            "to serve as a separately labeled bounded late-call debt."
        ),
    )
    parser.add_argument(
        "--allow-font-cache-manifest-completion-debt",
        action="store_true",
        help=(
            "After all explicit startup file-write debt is exhausted, allow "
            "only bounded uniformly false results for unique paths and exact "
            "sizes until the hashed 35-member retail catalog is closed."
        ),
    )
    parser.add_argument(
        "--seed-board-before-attach",
        action="store_true",
        help=(
            "Run a strict service-brokered command phase through the one-shot "
            "board RNG overrides before a nonzero late attachment."
        ),
    )
    parser.add_argument("--attach-timeout", type=float, default=20.0)
    parser.add_argument(
        "--startup-priority-bias-until-update",
        type=_positive_int,
        help=(
            "Temporarily lower the original main thread until this "
            "pre-attach update, leaving worker priorities unchanged, then "
            "restore the exact prior main-thread priority."
        ),
    )
    parser.add_argument(
        "--startup-process-affinity-mask",
        type=_positive_int,
        help=(
            "Apply this process affinity while the direct-launch process is "
            "still suspended, before its first instruction can run."
        ),
    )
    parser.add_argument(
        "--detach-at-update",
        type=_non_negative_int,
        help=(
            "detach after tracing through this update so a later capture "
            "interval runs without debugger breakpoints"
        ),
    )
    parser.add_argument(
        "--reattach-at-update",
        type=_non_negative_int,
        help=(
            "after --detach-at-update, wait read-only and resume strict "
            "tracing at this later framework update"
        ),
    )
    parser.add_argument(
        "--allow-post-blackout-attach-stabilization",
        action="store_true",
        help=(
            "After phased reattachment, ignore only bounded in-flight "
            "PrepareDemoCommand calls until the first exact complete "
            "original-main-thread offline boundary."
        ),
    )
    parser.add_argument(
        "--allow-blackout-file-write-order-rebase",
        action="store_true",
        help=(
            "After phased reattachment, allow one fixed command-order "
            "offset only when every missing entry is an exact long-form "
            "file-write result inside the debugger blackout."
        ),
    )
    parser.add_argument(
        "--blackout-expected-command-order-offset",
        type=_positive_int,
        help=(
            "Preregister one exact post-blackout command-entry deficit; "
            "requires the file-write rebase option and the matching native "
            "timeline offset."
        ),
    )
    parser.add_argument(
        "--blackout-expected-native-timeline-offset",
        type=_non_negative_int,
        help=(
            "Preregister one exact native-update surplus for the bounded "
            "file-write candidate envelope."
        ),
    )
    parser.add_argument(
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
    parser.add_argument(
        "--allow-post-blackout-overdue-idle-reentry",
        action="store_true",
        help=(
            "After phased reattachment, accept only an explicitly audited "
            "successful-write exit that is overdue inside the open interval "
            "between two adjacent saturated long idle commands."
        ),
    )
    parser.add_argument(
        "--broker-service-blocks",
        action="store_true",
        help=(
            "Suspend a racing main-thread call until the original service "
            "threads consume the contiguous offline service run."
        ),
    )
    parser.add_argument(
        "--successful-file-write-padding-row",
        dest="diagnostic_successful_file_write_padding_rows",
        action="append",
        type=_non_negative_int,
        default=[],
        help=(
            "Identify one successful file-write alignment row validated from "
            "the DMO transformation provenance; repeat for multiple rows."
        ),
    )
    parser.add_argument(
        "--service-wait-timeout",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--quiet-nonservice",
        action="store_true",
        help="Suppress ordinary non-service command-entry rows.",
    )
    parser.add_argument(
        "--stop-after-update",
        type=_non_negative_int,
    )
    parser.add_argument(
        "--suspend-main-thread-on-stop",
        action="store_true",
        help=(
            "Leave the stopped main thread suspended across debugger "
            "detach so a successor debugger can arm before resuming it."
        ),
    )
    parser.add_argument(
        "--stop-after-command-order",
        type=_non_negative_int,
        help=(
            "Stop only after the selected offline command's ending bit "
            "has been consumed."
        ),
    )
    parser.add_argument(
        "--close-after-terminal-command",
        action="store_true",
        help=(
            "After exact consumption of a payload-free final idle, post one "
            "normal WM_CLOSE and require exit code zero."
        ),
    )
    parser.add_argument(
        "--progress-every-updates",
        type=_positive_int,
        help="Print a compact progress row after crossing each update bucket.",
    )
    parser.add_argument(
        "--allow-pre-stream-commands",
        action="store_true",
        help=(
            "Allow only the initial negative-order framework command "
            "before the first exact offline DMO boundary."
        ),
    )
    parser.add_argument(
        "--result-json",
        type=Path,
        help="Exclusively create a structured strict-replay result.",
    )
    parser.add_argument(
        "--accept-normal-exit",
        action="store_true",
        help=(
            "Treat a naturally observed process exit with code zero and no "
            "trace failures as success even when a diagnostic hit/stop cap "
            "was not reached."
        ),
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    direct_path_values = (
        args.direct_runtime_exe,
        args.changedir,
    )
    if any(value is not None for value in direct_path_values) and any(
        value is None for value in direct_path_values
    ):
        raise ValueError(
            "--direct-runtime-exe and --changedir must be used together"
        )
    direct_launch = args.direct_runtime_exe is not None
    if args.direct_natural_seed:
        if not direct_launch or args.crt_rand_seed is not None:
            raise ValueError(
                "--direct-natural-seed requires a direct runtime and "
                "forbids --crt-rand-seed"
            )
    elif direct_launch and args.crt_rand_seed is None:
        raise ValueError(
            "direct fixed-seed launch requires --crt-rand-seed"
        )
    initial_global_mtrand_values = (
        args.initial_global_mtrand_oracle,
        args.initial_global_mtrand_seed,
        args.initial_global_mtrand_receipt_json,
    )
    initial_global_mtrand_requested = any(
        value is not None for value in initial_global_mtrand_values
    )
    if initial_global_mtrand_requested and any(
        value is None for value in initial_global_mtrand_values
    ):
        raise ValueError(
            "--initial-global-mtrand-oracle, its seed, and its receipt "
            "path must be used together"
        )
    if initial_global_mtrand_requested and (
        not direct_launch
        or not args.startup_trace_handoff
        or args.attach_at_update != 0
        or args.detach_at_update is None
        or args.reattach_at_update is None
        or args.startup_seed_transport != "debugger_register"
    ):
        raise ValueError(
            "initial global MTRand control requires phased direct startup "
            "debugger handoff at update zero"
        )
    startup_global_mtrand_observation_values = (
        args.startup_global_mtrand_observation_oracle,
        args.startup_global_mtrand_observation_seed,
        args.startup_global_mtrand_observation_end_at_update,
        args.startup_global_mtrand_observation_receipt_json,
    )
    startup_global_mtrand_observation_requested = any(
        value is not None
        for value in startup_global_mtrand_observation_values
    )
    if startup_global_mtrand_observation_requested and any(
        value is None
        for value in startup_global_mtrand_observation_values
    ):
        raise ValueError(
            "all startup global MTRand observation options must be used "
            "together"
        )
    if startup_global_mtrand_observation_requested and (
        not direct_launch
        or not args.startup_trace_handoff
        or args.attach_at_update != 0
        or args.detach_at_update is None
        or args.reattach_at_update is None
        or args.startup_seed_transport != "debugger_register"
        or initial_global_mtrand_requested
    ):
        raise ValueError(
            "startup global MTRand observation requires exclusive phased "
            "direct startup debugger handoff at update zero"
        )
    startup_hidden_draw_values = (
        args.startup_global_mtrand_hidden_draw_start_after_source_order,
        args.startup_global_mtrand_hidden_draw_stop_before_source_order,
    )
    startup_hidden_draw_requested = any(
        value is not None for value in startup_hidden_draw_values
    )
    if startup_hidden_draw_requested and (
        any(value is None for value in startup_hidden_draw_values)
        or not startup_global_mtrand_observation_requested
    ):
        raise ValueError(
            "startup hidden-draw observation requires both source-order "
            "boundaries and the complete startup observation contract"
        )
    source_bound_board_values = (
        args.source_bound_board_monitor,
        args.source_bound_board_trace,
        args.source_bound_board_recording_report,
        args.source_bound_board_monitor_update,
        args.source_bound_board_seed,
        args.source_bound_board_rewind_draws,
        args.source_bound_board_source_order,
        args.source_bound_board_framework_update,
        args.source_bound_board_caller,
    )
    source_bound_board_requested = any(
        value is not None for value in source_bound_board_values
    )
    if source_bound_board_requested and any(
        value is None for value in source_bound_board_values
    ):
        raise ValueError(
            "all source-bound board anchor options must be used together"
        )
    if (
        args.allow_source_bound_board_global_correction
        and not source_bound_board_requested
    ):
        raise ValueError(
            "--allow-source-bound-board-global-correction requires the "
            "complete source-bound board anchor"
        )
    if args.source_bound_board_precall_global_restore and (
        not source_bound_board_requested
        or args.allow_source_bound_board_global_correction
    ):
        raise ValueError(
            "--source-bound-board-precall-global-restore requires the "
            "complete source-bound anchor without post-call correction"
        )
    gameplay_mtrand_values = (
        args.gameplay_mtrand_oracle,
        args.gameplay_mtrand_oracle_seed,
        args.gameplay_mtrand_receipt_json,
    )
    gameplay_mtrand_requested = any(
        value is not None for value in gameplay_mtrand_values
    )
    if gameplay_mtrand_requested and any(
        value is None for value in gameplay_mtrand_values
    ):
        raise ValueError(
            "--gameplay-mtrand-oracle, its seed, and its receipt path must "
            "be used together"
        )
    if gameplay_mtrand_requested and (
        not args.startup_trace_handoff
        or args.detach_at_update is None
        or args.reattach_at_update is None
    ):
        raise ValueError(
            "gameplay MTRand synchronization requires phased startup "
            "trace handoff"
        )
    if (
        startup_global_mtrand_observation_requested
        and gameplay_mtrand_requested
    ):
        raise ValueError(
            "startup global MTRand observation and gameplay synchronization "
            "are mutually exclusive"
        )
    global_mtrand_values = (
        args.global_mtrand_restore_log,
        args.global_mtrand_restore_source_update,
        args.global_mtrand_expected_before_log,
        args.global_mtrand_expected_before_source_update,
        args.global_mtrand_restore_seed,
        args.global_mtrand_restore_command_order,
    )
    global_mtrand_requested = any(
        value is not None for value in global_mtrand_values
    )
    if global_mtrand_requested and any(
        value is None for value in global_mtrand_values
    ):
        raise ValueError(
            "all global MTRand restore options must be used together"
        )
    if (
        args.global_mtrand_allow_dynamic_before
        and not global_mtrand_requested
    ):
        raise ValueError(
            "--global-mtrand-allow-dynamic-before requires a restore"
        )
    qrand_values = (
        args.qrand_restore_log,
        args.qrand_restore_source_update,
        args.qrand_expected_before_log,
        args.qrand_expected_before_source_update,
        args.qrand_restore_command_order,
    )
    qrand_requested = any(value is not None for value in qrand_values)
    if qrand_requested and any(value is None for value in qrand_values):
        raise ValueError("all QRand restore options must be used together")
    thread_crt_values = (
        args.thread_crt_restore_log,
        args.thread_crt_restore_source_update,
        args.thread_crt_expected_before_log,
        args.thread_crt_expected_before_source_update,
        args.thread_crt_restore_command_order,
    )
    thread_crt_requested = any(
        value is not None for value in thread_crt_values
    )
    if thread_crt_requested and any(
        value is None for value in thread_crt_values
    ):
        raise ValueError(
            "all thread CRT restore options must be used together"
        )
    if args.thread_crt_allow_dynamic_before and not thread_crt_requested:
        raise ValueError(
            "--thread-crt-allow-dynamic-before requires thread CRT restore"
        )
    if source_bound_board_requested and (
        global_mtrand_requested
        or thread_crt_requested
        or not direct_launch
        or args.board_seed is None
        or args.board_seed_address != DEFAULT_BOARD_RESEED_CALL
        or args.board_seed_skip_count != 0
        or args.global_rng_seed is not None
        or args.thread_crt_rng_seed is not None
        or args.seed_board_before_attach
    ):
        raise ValueError(
            "source-bound board anchor conflicts with the selected RNG "
            "transport"
        )
    if (
        args.initial_global_mtrand_seed is not None
        and args.initial_global_mtrand_seed > 0xFFFFFFFF
    ):
        raise ValueError(
            "--initial-global-mtrand-seed must fit uint32"
        )
    if (
        args.startup_global_mtrand_observation_seed is not None
        and args.startup_global_mtrand_observation_seed > 0xFFFFFFFF
    ):
        raise ValueError(
            "--startup-global-mtrand-observation-seed must fit uint32"
        )
    if args.crt_rand_seed is not None and args.crt_rand_seed > 0xFFFFFFFF:
        raise ValueError("--crt-rand-seed must fit uint32")
    if args.board_seed is not None and args.board_seed > 0xFFFFFFFF:
        raise ValueError("--board-seed must fit uint32")
    if args.board_seed_skip_count and (
        args.board_seed is None or args.seed_board_before_attach
    ):
        raise ValueError(
            "--board-seed-skip-count requires --board-seed and cannot be "
            "combined with --seed-board-before-attach"
        )
    if (
        args.global_rng_seed is not None
        and args.global_rng_seed > 0xFFFFFFFF
    ):
        raise ValueError("--global-rng-seed must fit uint32")
    if (
        args.global_mtrand_restore_seed is not None
        and args.global_mtrand_restore_seed > 0xFFFFFFFF
    ):
        raise ValueError("--global-mtrand-restore-seed must fit uint32")
    if (
        args.source_bound_board_seed is not None
        and args.source_bound_board_seed > 0xFFFFFFFF
    ):
        raise ValueError("--source-bound-board-seed must fit uint32")
    if (
        args.thread_crt_rng_seed is not None
        and args.thread_crt_rng_seed > 0xFFFFFFFF
    ):
        raise ValueError("--thread-crt-rng-seed must fit uint32")
    if args.board_seed_address > 0xFFFFFFFF:
        raise ValueError("--board-seed-address must fit uint32")
    if args.board_seed is not None and not direct_launch:
        raise ValueError("--board-seed requires direct fixed-seed launch")
    if args.global_rng_seed is not None and args.board_seed is None:
        raise ValueError("--global-rng-seed requires --board-seed")
    if args.thread_crt_rng_seed is not None and args.board_seed is None:
        raise ValueError("--thread-crt-rng-seed requires --board-seed")
    if args.seed_board_before_attach and (
        not direct_launch
        or args.board_seed is None
        or args.attach_at_update <= 0
        or args.startup_seed_transport != "debugger_register"
    ):
        raise ValueError(
            "--seed-board-before-attach requires direct launch, "
            "--board-seed, and a nonzero --attach-at-update"
        )
    if args.startup_priority_bias_until_update is not None and (
        not direct_launch
        or args.seed_board_before_attach
        or (
            args.attach_at_update > 0
            and args.startup_priority_bias_until_update
            >= args.attach_at_update
        )
    ):
        raise ValueError(
            "--startup-priority-bias-until-update requires direct launch, "
            "must precede any nonzero --attach-at-update, and cannot be "
            "combined with --seed-board-before-attach"
        )
    if args.startup_process_affinity_mask is not None and (
        not direct_launch
        or (
            args.startup_seed_transport != "debugger_register"
            and not args.direct_natural_seed
        )
        or not args.startup_trace_handoff
    ):
        raise ValueError(
            "--startup-process-affinity-mask requires a direct debugger "
            "launch with --startup-trace-handoff"
        )
    if (
        args.allow_pre_stream_commands
        and args.attach_at_update != 0
    ):
        raise ValueError(
            "--allow-pre-stream-commands requires "
            "--attach-at-update=0"
        )
    if args.startup_trace_handoff and (
        not direct_launch
        or args.startup_seed_transport != "debugger_register"
        or args.attach_at_update != 0
        or not args.allow_pre_stream_commands
        or args.seed_board_before_attach
        or args.allow_attach_stabilization
        or args.allow_pre_attach_file_write_debt
    ):
        raise ValueError(
            "--startup-trace-handoff requires a direct debugger-register "
            "launch, --attach-at-update=0, --allow-pre-stream-commands, "
            "and no late-attach compatibility phase"
        )
    if args.allow_attach_stabilization and (
        not direct_launch
        or args.attach_at_update <= 0
        or args.seed_board_before_attach
        or args.allow_pre_stream_commands
    ):
        raise ValueError(
            "--allow-attach-stabilization requires a direct nonzero late "
            "attach and cannot be combined with pre-stream or pre-attach "
            "board phases"
        )
    if args.allow_pre_attach_file_write_debt and (
        not direct_launch
        or args.attach_at_update <= 0
        or args.allow_pre_stream_commands
        or args.seed_board_before_attach
    ):
        raise ValueError(
            "--allow-pre-attach-file-write-debt requires a direct nonzero "
            "late attach and cannot be combined with pre-stream or pre-attach "
            "board phases"
        )
    if args.allow_font_cache_manifest_completion_debt:
        legacy_manifest_mode = (
            args.allow_pre_attach_file_write_debt
            and direct_launch
            and args.attach_at_update > 0
            and not args.allow_pre_stream_commands
            and not args.seed_board_before_attach
        )
        startup_handoff_manifest_mode = (
            args.startup_trace_handoff
            and direct_launch
            and args.attach_at_update == 0
            and args.allow_pre_stream_commands
            and not args.allow_pre_attach_file_write_debt
            and not args.seed_board_before_attach
        )
        if not (legacy_manifest_mode or startup_handoff_manifest_mode):
            raise ValueError(
                "--allow-font-cache-manifest-completion-debt requires either "
                "the explicit pre-attach debt mode on a direct nonzero late "
                "attach or a gapless startup trace handoff"
            )
    if direct_launch and args.runtime_exe.resolve() != (
        args.direct_runtime_exe.resolve()
    ):
        raise ValueError(
            "--runtime-exe must identify the direct runtime executable"
        )
    phased = (
        args.detach_at_update is not None
        or args.reattach_at_update is not None
    )
    if phased and (
        args.detach_at_update is None
        or args.reattach_at_update is None
    ):
        raise ValueError(
            "--detach-at-update and --reattach-at-update must be used together"
        )
    if phased and (
        args.stop_after_update is not None
        or args.stop_after_command_order is not None
    ):
        raise ValueError(
            "phased tracing cannot be combined with an external stop target"
        )
    if args.direct_natural_seed and (
        not args.startup_trace_handoff
        or args.attach_at_update != 0
        or not args.allow_pre_stream_commands
        or phased
        or args.stop_after_update is None
        or not args.broker_service_blocks
        or args.board_seed is not None
        or args.global_rng_seed is not None
        or args.thread_crt_rng_seed is not None
        or args.seed_board_before_attach
        or source_bound_board_requested
        or gameplay_mtrand_requested
        or initial_global_mtrand_requested
        or startup_global_mtrand_observation_requested
        or global_mtrand_requested
        or qrand_requested
        or thread_crt_requested
        or args.allow_attach_stabilization
        or args.allow_pre_attach_file_write_debt
        or args.allow_source_bound_board_global_correction
        or args.source_bound_board_precall_global_restore
    ):
        raise ValueError(
            "natural direct command tracing requires a gapless update-zero "
            "service broker with one finite stop and no RNG controls; the "
            "separately validated retail font-cache manifest completion "
            "mechanism is permitted"
        )
    if args.allow_blackout_file_write_order_rebase and (
        not phased or not args.broker_service_blocks
    ):
        raise ValueError(
            "--allow-blackout-file-write-order-rebase requires phased "
            "service-brokered tracing"
        )
    blackout_envelope_values = (
        args.blackout_expected_command_order_offset,
        args.blackout_expected_native_timeline_offset,
    )
    blackout_allowed_offset_pairs = tuple(
        args.blackout_allowed_offset_pairs
    )
    if any(value is not None for value in blackout_envelope_values) and (
        any(value is None for value in blackout_envelope_values)
        or not args.allow_blackout_file_write_order_rebase
    ):
        raise ValueError(
            "blackout expected offsets must be supplied together with "
            "--allow-blackout-file-write-order-rebase"
        )
    if blackout_allowed_offset_pairs and (
        any(value is not None for value in blackout_envelope_values)
        or not args.allow_blackout_file_write_order_rebase
        or len(blackout_allowed_offset_pairs) < 2
        or tuple(sorted(set(blackout_allowed_offset_pairs)))
        != blackout_allowed_offset_pairs
        or not any(
            command_offset > 0
            for command_offset, _ in blackout_allowed_offset_pairs
        )
    ):
        raise ValueError(
            "blackout allowed offset pairs must be canonical, distinct, "
            "used only with the rebase option, and cannot be combined with "
            "the single expected-offset options"
        )
    if (
        args.allow_post_blackout_attach_stabilization
        or args.allow_post_blackout_overdue_idle_reentry
    ) and (
        not phased or not args.broker_service_blocks or not direct_launch
    ):
        raise ValueError(
            "post-blackout stabilization options require phased direct-"
            "runtime service-brokered tracing"
        )
    if phased and not (
        args.attach_at_update
        < args.detach_at_update
        < args.reattach_at_update
    ):
        raise ValueError(
            "trace phase updates must satisfy attach < detach < reattach"
        )
    late_attach_rng_restore = (
        direct_launch
        and not phased
        and args.attach_at_update > 0
        and args.allow_attach_stabilization
        and args.allow_pre_attach_file_write_debt
        and args.allow_font_cache_manifest_completion_debt
        and not args.allow_pre_stream_commands
        and not args.seed_board_before_attach
        and args.startup_priority_bias_until_update is not None
        and args.startup_priority_bias_until_update < args.attach_at_update
    )
    if (
        global_mtrand_requested
        or qrand_requested
        or thread_crt_requested
    ) and not (
        direct_launch
        and not phased
        and (
            args.attach_at_update == 0
            or late_attach_rng_restore
        )
    ):
        raise ValueError(
            "ephemeral RNG restore requires either a direct, unphased "
            "update-zero trace or the fully audited late-attach startup "
            "corridor"
        )
    if (
        args.stop_after_update is not None
        and args.stop_after_command_order is not None
    ):
        raise ValueError(
            "--stop-after-update and --stop-after-command-order "
            "are mutually exclusive"
        )
    if args.close_after_terminal_command and (
        not args.accept_normal_exit
        or args.stop_after_update is not None
        or args.stop_after_command_order is not None
        or args.suspend_main_thread_on_stop
    ):
        raise ValueError(
            "--close-after-terminal-command requires --accept-normal-exit "
            "and cannot use another stop or handoff target"
        )
    required_paths = [(args.dmo, "DMO")]
    if direct_launch:
        required_paths.extend(
            (
                (args.direct_runtime_exe, "direct runtime executable"),
                (args.changedir, "retail asset directory"),
            )
        )
    if source_bound_board_requested:
        required_paths.extend(
            (
                (
                    args.source_bound_board_monitor,
                    "source-bound board RNG monitor",
                ),
                (
                    args.source_bound_board_trace,
                    "source-bound board hardware trace",
                ),
                (
                    args.source_bound_board_recording_report,
                    "source-bound board recording report",
                ),
            )
        )
    if gameplay_mtrand_requested:
        required_paths.append(
            (
                args.gameplay_mtrand_oracle,
                "gameplay MTRand oracle",
            )
        )
    if initial_global_mtrand_requested:
        required_paths.append(
            (
                args.initial_global_mtrand_oracle,
                "initial global MTRand oracle",
            )
        )
    if startup_global_mtrand_observation_requested:
        required_paths.append(
            (
                args.startup_global_mtrand_observation_oracle,
                "startup global MTRand observation oracle",
            )
        )
    if global_mtrand_requested:
        required_paths.extend(
            (
                (
                    args.global_mtrand_restore_log,
                    "global MTRand restore monitor",
                ),
                (
                    args.global_mtrand_expected_before_log,
                    "global MTRand expected-before monitor",
                ),
            )
        )
    if qrand_requested:
        required_paths.extend(
            (
                (args.qrand_restore_log, "QRand restore monitor"),
                (
                    args.qrand_expected_before_log,
                    "QRand expected-before monitor",
                ),
            )
        )
    if thread_crt_requested:
        required_paths.extend(
            (
                (
                    args.thread_crt_restore_log,
                    "thread CRT restore monitor",
                ),
                (
                    args.thread_crt_expected_before_log,
                    "thread CRT expected-before monitor",
                ),
            )
        )
    if not direct_launch:
        required_paths.append(
            (args.steam_exe, "Steam executable")
        )
    for path, label in required_paths:
        if not path.is_file():
            if label == "retail asset directory" and path.is_dir():
                continue
            raise FileNotFoundError(f"{label} does not exist: {path}")
    if args.result_json is not None and args.result_json.exists():
        raise FileExistsError(
            f"refusing to overwrite result JSON: {args.result_json}"
        )
    if gameplay_mtrand_requested:
        assert args.gameplay_mtrand_receipt_json is not None
        if args.gameplay_mtrand_receipt_json.exists():
            raise FileExistsError(
                "refusing to overwrite gameplay MTRand receipt JSON: "
                f"{args.gameplay_mtrand_receipt_json}"
            )
        if not args.gameplay_mtrand_receipt_json.parent.is_dir():
            raise FileNotFoundError(
                "gameplay MTRand receipt parent does not exist: "
                f"{args.gameplay_mtrand_receipt_json.parent}"
            )
    if initial_global_mtrand_requested:
        assert args.initial_global_mtrand_receipt_json is not None
        if args.initial_global_mtrand_receipt_json.exists():
            raise FileExistsError(
                "refusing to overwrite initial global MTRand receipt JSON: "
                f"{args.initial_global_mtrand_receipt_json}"
            )
        if not args.initial_global_mtrand_receipt_json.parent.is_dir():
            raise FileNotFoundError(
                "initial global MTRand receipt parent does not exist: "
                f"{args.initial_global_mtrand_receipt_json.parent}"
            )
    if startup_global_mtrand_observation_requested:
        startup_receipt_path = (
            args.startup_global_mtrand_observation_receipt_json
        )
        assert startup_receipt_path is not None
        if startup_receipt_path.exists():
            raise FileExistsError(
                "refusing to overwrite startup global MTRand observation "
                f"receipt JSON: {startup_receipt_path}"
            )
        if not startup_receipt_path.parent.is_dir():
            raise FileNotFoundError(
                "startup global MTRand observation receipt parent does not "
                f"exist: {startup_receipt_path.parent}"
            )
    font_cache_manifest: dict[str, int] | None = None
    font_cache_manifest_sha256: str | None = None
    font_cache_manifest_main_pak_sha256: str | None = None
    if args.allow_font_cache_manifest_completion_debt:
        assert args.changedir is not None
        (
            font_cache_manifest,
            font_cache_manifest_sha256,
            font_cache_manifest_main_pak_sha256,
        ) = _load_font_cache_manifest(args.changedir)
        print(
            "font_cache_manifest_receipt "
            f"entries={len(font_cache_manifest)} "
            f"manifest_sha256={font_cache_manifest_sha256} "
            f"main_pak_sha256={font_cache_manifest_main_pak_sha256}",
            flush=True,
        )
    source_bound_board_global_state: GlobalMTRandRestoreState | None = None
    source_bound_board_thread_crt_state: ThreadCrtRestoreState | None = None
    if source_bound_board_requested:
        source_bound_board_global_state = (
            load_source_bound_global_mtrand_restore_state(
                args.source_bound_board_monitor,
                args.source_bound_board_trace,
                args.source_bound_board_recording_report,
                monitor_framework_update=(
                    args.source_bound_board_monitor_update
                ),
                seed=args.source_bound_board_seed,
                rewind_draws=args.source_bound_board_rewind_draws,
                source_order=args.source_bound_board_source_order,
                framework_update=(
                    args.source_bound_board_framework_update
                ),
                caller=args.source_bound_board_caller,
            )
        )
        source_bound_board_thread_crt_state = (
            load_source_bound_thread_crt_restore_state(
                args.source_bound_board_trace,
                args.source_bound_board_recording_report,
                source_order=args.source_bound_board_source_order,
                framework_update=(
                    args.source_bound_board_framework_update
                ),
                caller=args.source_bound_board_caller,
            )
        )
        _validate_source_bound_board_anchor_states(
            source_bound_board_global_state,
            source_bound_board_thread_crt_state,
        )
        source_output, _, _, _ = _global_mtrand_call_transition(
            source_bound_board_global_state.payload
        )
        if source_output != args.board_seed:
            raise ValueError(
                "--board-seed must equal the source-bound global call output"
            )
        source_target_update = (
            source_bound_board_thread_crt_state.source_framework_update
        )
        if (
            args.detach_at_update is not None
            and source_target_update >= args.detach_at_update
        ) or (
            args.detach_at_update is None
            and args.stop_after_update is not None
            and source_target_update > args.stop_after_update
        ):
            raise ValueError(
                "source-bound board anchor target is outside the trace phase"
            )

    initial_global_mtrand_oracle: (
        InitialGlobalMTRandCallOracle | None
    ) = None
    if initial_global_mtrand_requested:
        assert args.initial_global_mtrand_oracle is not None
        assert args.initial_global_mtrand_seed is not None
        initial_global_mtrand_oracle = (
            load_initial_global_mtrand_call_oracle(
                args.initial_global_mtrand_oracle,
                seed=args.initial_global_mtrand_seed,
            )
        )

    startup_global_mtrand_observation_oracle: (
        StartupGlobalMTRandObservationOracle | None
    ) = None
    if startup_global_mtrand_observation_requested:
        assert args.startup_global_mtrand_observation_oracle is not None
        assert args.startup_global_mtrand_observation_seed is not None
        assert (
            args.startup_global_mtrand_observation_end_at_update is not None
        )
        startup_global_mtrand_observation_oracle = (
            load_startup_global_mtrand_observation_oracle(
                args.startup_global_mtrand_observation_oracle,
                seed=args.startup_global_mtrand_observation_seed,
                end_at_update=(
                    args.startup_global_mtrand_observation_end_at_update
                ),
                maximum_draws=(
                    args.startup_global_mtrand_observation_maximum_draws
                ),
            )
        )
        assert args.detach_at_update is not None
        if (
            startup_global_mtrand_observation_oracle.entries[-1]
            .framework_update
            >= args.detach_at_update
        ):
            raise ValueError(
                "startup global MTRand observation oracle does not complete "
                "before detach"
            )
        if startup_hidden_draw_requested:
            assert (
                args.startup_global_mtrand_hidden_draw_start_after_source_order
                is not None
            )
            assert (
                args.startup_global_mtrand_hidden_draw_stop_before_source_order
                is not None
            )
            source_order_to_entry_order = {
                entry.source_order: order
                for order, entry in enumerate(
                    startup_global_mtrand_observation_oracle.entries
                )
            }
            start_entry_order = source_order_to_entry_order.get(
                args.startup_global_mtrand_hidden_draw_start_after_source_order
            )
            stop_entry_order = source_order_to_entry_order.get(
                args.startup_global_mtrand_hidden_draw_stop_before_source_order
            )
            if (
                start_entry_order is None
                or stop_entry_order is None
                or stop_entry_order != start_entry_order + 1
            ):
                raise ValueError(
                    "startup hidden-draw boundaries must identify consecutive "
                    "entries in the loaded observation oracle"
                )

    gameplay_mtrand_oracle: GameplayMTRandOracle | None = None
    if gameplay_mtrand_requested:
        assert args.gameplay_mtrand_oracle is not None
        assert args.gameplay_mtrand_oracle_seed is not None
        gameplay_mtrand_oracle = load_gameplay_mtrand_oracle(
            args.gameplay_mtrand_oracle,
            seed=args.gameplay_mtrand_oracle_seed,
            maximum_draws=(
                args.gameplay_mtrand_oracle_maximum_draws
            ),
        )
        assert args.detach_at_update is not None
        if (
            gameplay_mtrand_oracle.entries[-1].framework_update
            >= args.detach_at_update
        ):
            raise ValueError(
                "gameplay MTRand oracle does not complete before detach"
            )

    global_mtrand_restore_state: GlobalMTRandRestoreState | None = None
    global_mtrand_expected_before_state: (
        GlobalMTRandRestoreState | None
    ) = None
    if global_mtrand_requested:
        global_mtrand_restore_state = load_global_mtrand_restore_state(
            args.global_mtrand_restore_log,
            framework_update=args.global_mtrand_restore_source_update,
            seed=args.global_mtrand_restore_seed,
            maximum_draws=args.global_mtrand_restore_maximum_draws,
            rewind_draws=args.global_mtrand_restore_rewind_draws,
        )
        global_mtrand_expected_before_state = (
            load_global_mtrand_restore_state(
                args.global_mtrand_expected_before_log,
                framework_update=(
                    args.global_mtrand_expected_before_source_update
                ),
                seed=args.global_mtrand_restore_seed,
                maximum_draws=args.global_mtrand_restore_maximum_draws,
                rewind_draws=args.global_mtrand_restore_rewind_draws,
            )
        )
    qrand_restore_state: QRandRestoreState | None = None
    qrand_expected_before_state: QRandRestoreState | None = None
    if qrand_requested:
        qrand_restore_state = load_qrand_restore_state(
            args.qrand_restore_log,
            framework_update=args.qrand_restore_source_update,
        )
        qrand_expected_before_state = load_qrand_restore_state(
            args.qrand_expected_before_log,
            framework_update=(
                args.qrand_expected_before_source_update
            ),
        )
    thread_crt_restore_state: ThreadCrtRestoreState | None = None
    thread_crt_expected_before_state: ThreadCrtRestoreState | None = None
    if thread_crt_requested:
        thread_crt_restore_state = load_thread_crt_restore_state(
            args.thread_crt_restore_log,
            framework_update=args.thread_crt_restore_source_update,
            rewind_draws=args.thread_crt_restore_rewind_draws,
        )
        thread_crt_expected_before_state = (
            load_thread_crt_restore_state(
                args.thread_crt_expected_before_log,
                framework_update=(
                    args.thread_crt_expected_before_source_update
                ),
                rewind_draws=args.thread_crt_restore_rewind_draws,
            )
        )
    existing = set(process_ids_by_name(args.runtime_exe.name))
    if existing:
        raise RuntimeError(
            f"refusing to launch with an existing runtime: {sorted(existing)}"
        )
    started_utc = datetime.now(timezone.utc).isoformat()
    trace_started_perf_counter_ns = time.perf_counter_ns()
    startup_rng: (
        FixedSeedLaunchEvidence
        | FixedSeedIatLaunchEvidence
        | NaturalSeedLaunchEvidence
        | None
    ) = None
    startup_handoff_main_thread_id: int | None = None
    startup_handoff_launcher_suspend_previous_count: int | None = None
    board_rng_observations: list[dict[str, Any]] = []
    source_bound_board_observations: list[dict[str, Any]] = []
    global_mtrand_restore_observations: list[dict[str, Any]] = []
    qrand_restore_observations: list[dict[str, Any]] = []
    thread_crt_restore_observations: list[dict[str, Any]] = []
    gameplay_mtrand_sync_observations: list[dict[str, Any]] = []
    gameplay_mtrand_sync_receipt: dict[str, Any] = {}
    initial_global_mtrand_observations: list[dict[str, Any]] = []
    startup_global_mtrand_observations: list[dict[str, Any]] = []
    startup_global_mtrand_observation_receipt: dict[str, Any] = {}
    startup_global_mtrand_hidden_draw_observations: list[
        dict[str, Any]
    ] = []
    board_seed_phase: Mapping[str, Any] | None = None
    startup_priority_phase: Mapping[str, Any] | None = None
    in_trace_startup_priority: StartupPriorityBiasRuntime | None = None
    if direct_launch:
        board_phase_started_ns = (
            time.perf_counter_ns()
            if args.seed_board_before_attach
            else None
        )

        def board_seed_callback(
            process: int,
            thread: int,
            context: WOW64_CONTEXT,
            process_id: int,
            thread_id: int,
        ) -> Mapping[str, Any]:
            return _apply_board_rng_overrides(
                process=process,
                thread=thread,
                context=context,
                process_id=process_id,
                thread_id=thread_id,
                board_seed_address=args.board_seed_address,
                board_seed_override=args.board_seed,
                global_rng_seed_override=args.global_rng_seed,
                thread_crt_rng_seed_override=args.thread_crt_rng_seed,
            )

        if args.direct_natural_seed:
            startup_rng = launch_natural_seed_replay(
                runtime_executable=args.direct_runtime_exe,
                changedir=args.changedir,
                dmo=args.dmo,
                app_id=args.app_id,
                timeout=args.launch_timeout,
                suspend_main_thread_on_detach=(
                    args.startup_trace_handoff
                ),
            )
        elif args.startup_seed_transport == "iat_stub":
            startup_rng = launch_fixed_seed_replay_iat_stub(
                runtime_executable=args.direct_runtime_exe,
                changedir=args.changedir,
                dmo=args.dmo,
                app_id=args.app_id,
                seed=args.crt_rand_seed,
                timeout=args.launch_timeout,
            )
        else:
            startup_rng = launch_fixed_seed_replay(
                runtime_executable=args.direct_runtime_exe,
                changedir=args.changedir,
                dmo=args.dmo,
                app_id=args.app_id,
                seed=args.crt_rand_seed,
                timeout=args.launch_timeout,
                board_seed_address=(
                    args.board_seed_address
                    if args.seed_board_before_attach
                    else None
                ),
                board_seed_callback=(
                    board_seed_callback
                    if args.seed_board_before_attach
                    else None
                ),
                suspend_main_thread_on_detach=(
                    args.startup_trace_handoff
                ),
                process_affinity_mask=(
                    args.startup_process_affinity_mask
                ),
            )
        pid = startup_rng.process_id
        if args.startup_trace_handoff:
            if (
                not isinstance(
                    startup_rng,
                    (FixedSeedLaunchEvidence, NaturalSeedLaunchEvidence),
                )
                or startup_rng.main_thread_suspended_on_detach is not True
                or startup_rng.main_thread_suspend_previous_count != 0
            ):
                raise RuntimeError(
                    "startup trace handoff launch evidence is incomplete"
                )
            startup_handoff_main_thread_id = startup_rng.thread_id
            startup_handoff_launcher_suspend_previous_count = (
                startup_rng.main_thread_suspend_previous_count
            )
        if args.seed_board_before_attach:
            observation = startup_rng.board_seed_observation
            if (
                board_phase_started_ns is None
                or not isinstance(observation, Mapping)
            ):
                raise RuntimeError(
                    "startup board RNG breakpoint did not produce evidence"
                )
            board_rng_observations.append(dict(observation))
            board_phase_finished_ns = time.perf_counter_ns()
            board_seed_phase = {
                "name": "pre_attach_board_seed",
                "mechanism": "startup_dual_breakpoint",
                "started_perf_counter_ns": board_phase_started_ns,
                "finished_perf_counter_ns": board_phase_finished_ns,
                "result": {
                    "hits": 1,
                    "exited": False,
                    "exit_code": None,
                    "brokered_blocks": 0,
                    "broker_bypassed_blocks": 0,
                    "broker_failures": [],
                    "boundary_failures": [],
                    "last_update": int(
                        observation["framework_update"]
                    ),
                    "stopped_at_update": False,
                    "stopped_at_command_order": False,
                    "failure_count": 0,
                },
            }
        if isinstance(startup_rng, NaturalSeedLaunchEvidence):
            print(
                "startup_rng "
                f"observed_seed={startup_rng.observed_seed} "
                "transport=natural_debugger_observation "
                "register_override=None "
                f"breakpoint=0x{startup_rng.breakpoint_address:08X} "
                f"pid={pid} tid={startup_rng.thread_id}",
                flush=True,
            )
        elif isinstance(startup_rng, FixedSeedIatLaunchEvidence):
            print(
                "startup_rng "
                f"seed={startup_rng.seed} "
                "transport=iat_stub "
                f"iat=0x{startup_rng.iat_address:08X} "
                f"pid={pid} tid={startup_rng.thread_id}",
                flush=True,
            )
        else:
            print(
                "startup_rng "
                f"seed={startup_rng.seed} "
                "transport=debugger_register "
                "breakpoint="
                f"0x{startup_rng.breakpoint_address:08X} "
                f"pid={pid} tid={startup_rng.thread_id}",
                flush=True,
            )
    else:
        subprocess.Popen(
            [
                str(args.steam_exe),
                "-silent",
                "-applaunch",
                str(args.app_id),
                "-play",
                f"-demofile={args.dmo.resolve()}",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
        )
        pid = wait_for_new_runtime(
            existing_pids=existing,
            runtime_exe=args.runtime_exe,
            timeout=args.launch_timeout,
        )
    print(f"runtime_pid={pid}", flush=True)
    offline_rows = (
        _offline_rows(args.dmo)
        if (
            args.broker_service_blocks
            or global_mtrand_requested
            or qrand_requested
            or thread_crt_requested
            or args.allow_attach_stabilization
            or args.allow_pre_attach_file_write_debt
            or args.allow_font_cache_manifest_completion_debt
            or args.close_after_terminal_command
            or args.diagnostic_successful_file_write_padding_rows
        )
        else None
    )
    if args.seed_board_before_attach and (
        board_seed_phase is None
        or len(board_rng_observations) != 1
    ):
        raise RuntimeError("pre-attach board RNG phase failed")
    if args.attach_at_update:
        if args.startup_priority_bias_until_update is None:
            base, observed_update = wait_for_update_before_attach(
                pid=pid,
                target_update=args.attach_at_update,
                timeout=args.attach_timeout,
            )
        else:
            if startup_rng is None:
                raise RuntimeError(
                    "startup priority bias requires launch evidence"
                )
            (
                base,
                observed_update,
                startup_priority_phase,
            ) = wait_for_update_with_startup_priority_bias(
                pid=pid,
                main_thread_id=startup_rng.thread_id,
                target_update=args.attach_at_update,
                bias_until_update=(
                    args.startup_priority_bias_until_update
                ),
                timeout=args.attach_timeout,
            )
        print(
            f"attach_gate base=0x{base:08X} update={observed_update}",
            flush=True,
        )
    elif args.startup_priority_bias_until_update is not None:
        if startup_rng is None:
            raise RuntimeError(
                "startup priority bias requires launch evidence"
            )
        if args.startup_trace_handoff:
            if (
                not isinstance(
                    startup_rng,
                    (FixedSeedLaunchEvidence, NaturalSeedLaunchEvidence),
                )
                or startup_rng.main_thread_suspended_on_detach is not True
            ):
                raise RuntimeError(
                    "gapless priority bias requires a suspended startup "
                    "handoff"
                )
            base = 0
            observed_update = 0
        else:
            base, observed_update = wait_for_update_before_attach(
                pid=pid,
                target_update=0,
                timeout=args.attach_timeout,
            )
        in_trace_startup_priority = (
            _begin_in_trace_startup_priority_bias(
                pid=pid,
                main_thread_id=startup_rng.thread_id,
                framework_update=observed_update,
                bias_until_update=(
                    args.startup_priority_bias_until_update
                ),
            )
        )
        print(
            f"attach_gate base=0x{base:08X} update={observed_update} "
            "startup_handoff_suspended="
            f"{args.startup_trace_handoff}",
            flush=True,
        )
    trace_phases: list[Mapping[str, Any]] = []
    debugger_blackout: Mapping[str, Any] | None = None
    board_seed_in_command_trace = (
        args.board_seed is not None
        and not args.seed_board_before_attach
    )
    if phased:
        early_started_ns = time.perf_counter_ns()
        early_result = trace_demo_commands(
            pid=pid,
            executable=args.runtime_exe,
            address=args.address,
            maximum_hits=args.maximum_hits,
            timeout=args.trace_timeout,
            offline_rows=offline_rows,
            broker_service_blocks=args.broker_service_blocks,
            diagnostic_successful_file_write_padding_rows=tuple(
                args.diagnostic_successful_file_write_padding_rows
            ),
            service_wait_timeout=args.service_wait_timeout,
            quiet_nonservice=args.quiet_nonservice,
            stop_after_update=args.detach_at_update,
            progress_every_updates=args.progress_every_updates,
            board_seed_address=(
                args.board_seed_address
                if board_seed_in_command_trace
                else None
            ),
            board_seed_override=(
                args.board_seed
                if board_seed_in_command_trace
                else None
            ),
            board_seed_skip_count=(
                args.board_seed_skip_count
                if board_seed_in_command_trace
                else 0
            ),
            global_rng_seed_override=(
                args.global_rng_seed
                if board_seed_in_command_trace
                else None
            ),
            thread_crt_rng_seed_override=(
                args.thread_crt_rng_seed
                if board_seed_in_command_trace
                else None
            ),
            board_seed_observations=board_rng_observations,
            source_bound_board_global_state=(
                source_bound_board_global_state
            ),
            source_bound_board_thread_crt_state=(
                source_bound_board_thread_crt_state
            ),
            source_bound_board_observations=(
                source_bound_board_observations
            ),
            allow_source_bound_board_global_correction=(
                args.allow_source_bound_board_global_correction
            ),
            source_bound_board_precall_global_restore=(
                args.source_bound_board_precall_global_restore
            ),
            allow_pre_stream_commands=args.allow_pre_stream_commands,
            attach_stabilization_after_update=(
                args.attach_at_update
                if args.allow_attach_stabilization
                else None
            ),
            attach_stabilization_main_thread_id=(
                startup_rng.thread_id
                if args.allow_attach_stabilization
                and startup_rng is not None
                else None
            ),
            pre_attach_file_write_debt_before_update=(
                args.attach_at_update
                if args.allow_pre_attach_file_write_debt
                else None
            ),
            font_cache_manifest=font_cache_manifest,
            font_cache_manifest_sha256=font_cache_manifest_sha256,
            font_cache_manifest_main_pak_sha256=(
                font_cache_manifest_main_pak_sha256
            ),
            startup_priority_bias=in_trace_startup_priority,
            startup_handoff_main_thread_id=(
                startup_handoff_main_thread_id
            ),
            startup_handoff_launcher_suspend_previous_count=(
                startup_handoff_launcher_suspend_previous_count
            ),
            global_mtrand_restore_state=(
                global_mtrand_restore_state
            ),
            global_mtrand_expected_before_state=(
                global_mtrand_expected_before_state
            ),
            global_mtrand_restore_command_order=(
                args.global_mtrand_restore_command_order
            ),
            global_mtrand_restore_observations=(
                global_mtrand_restore_observations
            ),
            global_mtrand_allow_dynamic_before=(
                args.global_mtrand_allow_dynamic_before
            ),
            qrand_restore_state=qrand_restore_state,
            qrand_expected_before_state=qrand_expected_before_state,
            qrand_restore_command_order=(
                args.qrand_restore_command_order
            ),
            qrand_restore_observations=(
                qrand_restore_observations
            ),
            thread_crt_restore_state=thread_crt_restore_state,
            thread_crt_expected_before_state=(
                thread_crt_expected_before_state
            ),
            thread_crt_restore_command_order=(
                args.thread_crt_restore_command_order
            ),
            thread_crt_restore_observations=(
                thread_crt_restore_observations
            ),
            thread_crt_allow_dynamic_before=(
                args.thread_crt_allow_dynamic_before
            ),
            gameplay_mtrand_oracle=gameplay_mtrand_oracle,
            gameplay_mtrand_sync_observations=(
                gameplay_mtrand_sync_observations
                if gameplay_mtrand_requested
                else None
            ),
            gameplay_mtrand_sync_receipt=(
                gameplay_mtrand_sync_receipt
                if gameplay_mtrand_requested
                else None
            ),
            initial_global_mtrand_oracle=(
                initial_global_mtrand_oracle
            ),
            initial_global_mtrand_observations=(
                initial_global_mtrand_observations
                if initial_global_mtrand_requested
                else None
            ),
            startup_global_mtrand_observation_oracle=(
                startup_global_mtrand_observation_oracle
            ),
            startup_global_mtrand_observations=(
                startup_global_mtrand_observations
                if startup_global_mtrand_observation_requested
                else None
            ),
            startup_global_mtrand_observation_receipt=(
                startup_global_mtrand_observation_receipt
                if startup_global_mtrand_observation_requested
                else None
            ),
            startup_global_mtrand_hidden_draw_start_after_source_order=(
                args.startup_global_mtrand_hidden_draw_start_after_source_order
                if startup_hidden_draw_requested
                else None
            ),
            startup_global_mtrand_hidden_draw_stop_before_source_order=(
                args.startup_global_mtrand_hidden_draw_stop_before_source_order
                if startup_hidden_draw_requested
                else None
            ),
            startup_global_mtrand_hidden_draw_observations=(
                startup_global_mtrand_hidden_draw_observations
                if startup_hidden_draw_requested
                else None
            ),
            suspend_main_thread_on_stop=(
                args.suspend_main_thread_on_stop
            ),
        )
        early_finished_ns = time.perf_counter_ns()
        if gameplay_mtrand_requested:
            assert args.gameplay_mtrand_receipt_json is not None
            gameplay_receipt_sha256 = (
                _write_gameplay_mtrand_sync_receipt_json(
                    args.gameplay_mtrand_receipt_json,
                    dmo=args.dmo,
                    runtime_exe=args.runtime_exe,
                    runtime_process_id=pid,
                    started_perf_counter_ns=early_started_ns,
                    finished_perf_counter_ns=early_finished_ns,
                    receipt=gameplay_mtrand_sync_receipt,
                    observations=gameplay_mtrand_sync_observations,
                    result=early_result,
                )
            )
            print(
                "gameplay_mtrand_sync_receipt="
                f"{args.gameplay_mtrand_receipt_json.resolve()} "
                f"sha256={gameplay_receipt_sha256}",
                flush=True,
            )
        if initial_global_mtrand_requested:
            assert args.initial_global_mtrand_receipt_json is not None
            assert initial_global_mtrand_oracle is not None
            initial_receipt_sha256 = (
                _write_initial_global_mtrand_receipt_json(
                    args.initial_global_mtrand_receipt_json,
                    dmo=args.dmo,
                    runtime_exe=args.runtime_exe,
                    runtime_process_id=pid,
                    started_perf_counter_ns=early_started_ns,
                    finished_perf_counter_ns=early_finished_ns,
                    oracle=initial_global_mtrand_oracle,
                    observations=initial_global_mtrand_observations,
                    result=early_result,
                )
            )
            print(
                "initial_global_mtrand_receipt="
                f"{args.initial_global_mtrand_receipt_json.resolve()} "
                f"sha256={initial_receipt_sha256}",
                flush=True,
            )
        if startup_global_mtrand_observation_requested:
            startup_receipt_path = (
                args.startup_global_mtrand_observation_receipt_json
            )
            assert startup_receipt_path is not None
            assert startup_global_mtrand_observation_oracle is not None
            startup_receipt_sha256 = (
                _write_startup_global_mtrand_observation_receipt_json(
                    startup_receipt_path,
                    dmo=args.dmo,
                    runtime_exe=args.runtime_exe,
                    runtime_process_id=pid,
                    started_perf_counter_ns=early_started_ns,
                    finished_perf_counter_ns=early_finished_ns,
                    oracle=startup_global_mtrand_observation_oracle,
                    receipt=startup_global_mtrand_observation_receipt,
                    observations=startup_global_mtrand_observations,
                    result=early_result,
                    hidden_draw_observations=(
                        startup_global_mtrand_hidden_draw_observations
                    ),
                )
            )
            print(
                "startup_global_mtrand_observation_receipt="
                f"{startup_receipt_path.resolve()} "
                f"sha256={startup_receipt_sha256}",
                flush=True,
            )
        trace_phases.append(
            _trace_phase_evidence(
                name="pre_capture",
                started_perf_counter_ns=early_started_ns,
                finished_perf_counter_ns=early_finished_ns,
                result=early_result,
            )
        )
        if (
            early_result.failure_count
            or early_result.exited
            or not early_result.stopped_at_update
        ):
            result = early_result
        else:
            blackout_started_ns = time.perf_counter_ns()
            print(
                "debugger_blackout_start "
                f"detached_after_update={early_result.last_update} "
                f"reattach_target={args.reattach_at_update}",
                flush=True,
            )
            base, reattach_observed_update = wait_for_update_before_attach(
                pid=pid,
                target_update=args.reattach_at_update,
                timeout=args.attach_timeout,
            )
            blackout_finished_ns = time.perf_counter_ns()
            debugger_blackout = {
                "requested_detach_update": args.detach_at_update,
                "detached_after_update": early_result.last_update,
                "requested_reattach_update": args.reattach_at_update,
                "reattach_observed_update": reattach_observed_update,
                "started_perf_counter_ns": blackout_started_ns,
                "finished_perf_counter_ns": blackout_finished_ns,
                "process_id": pid,
                "read_only_wait": True,
            }
            print(
                "debugger_blackout_end "
                f"base=0x{base:08X} update={reattach_observed_update}",
                flush=True,
            )
            late_started_ns = time.perf_counter_ns()
            late_result = trace_demo_commands(
                pid=pid,
                executable=args.runtime_exe,
                address=args.address,
                maximum_hits=args.maximum_hits,
                timeout=args.trace_timeout,
                offline_rows=offline_rows,
                broker_service_blocks=args.broker_service_blocks,
                diagnostic_successful_file_write_padding_rows=tuple(
                    args.diagnostic_successful_file_write_padding_rows
                ),
                initial_service_consumed_file_write_debt_rows=(
                    early_result.service_consumed_file_write_debt_rows
                ),
                service_wait_timeout=args.service_wait_timeout,
                quiet_nonservice=args.quiet_nonservice,
                close_after_terminal_command=(
                    args.close_after_terminal_command
                ),
                progress_every_updates=args.progress_every_updates,
                allow_orphan_prepared_service_block=True,
                allow_post_blackout_overdue_idle_reentry=(
                    args.allow_post_blackout_overdue_idle_reentry
                ),
                allow_initial_file_write_order_rebase=(
                    args.allow_blackout_file_write_order_rebase
                ),
                command_order_rebase_after_update=(
                    early_result.last_update
                    if args.allow_blackout_file_write_order_rebase
                    else None
                ),
                blackout_expected_command_order_offset=(
                    args.blackout_expected_command_order_offset
                ),
                blackout_expected_native_timeline_offset=(
                    args.blackout_expected_native_timeline_offset
                ),
                blackout_allowed_offset_pairs=(
                    blackout_allowed_offset_pairs
                ),
                initial_command_order_offset=(
                    early_result.command_order_offset
                ),
                initial_command_order_rebase_rows=(
                    early_result.command_order_rebase_rows
                ),
                attach_stabilization_after_update=(
                    reattach_observed_update
                    if (
                        args.allow_post_blackout_attach_stabilization
                        or args.allow_blackout_file_write_order_rebase
                        or args.allow_post_blackout_overdue_idle_reentry
                    )
                    else None
                ),
                attach_stabilization_main_thread_id=(
                    startup_rng.thread_id
                    if (
                        args.allow_post_blackout_attach_stabilization
                        or args.allow_blackout_file_write_order_rebase
                        or args.allow_post_blackout_overdue_idle_reentry
                    )
                    and startup_rng is not None
                    else None
                ),
            )
            late_finished_ns = time.perf_counter_ns()
            if (
                args.allow_post_blackout_attach_stabilization
                or args.allow_blackout_file_write_order_rebase
                or args.allow_post_blackout_overdue_idle_reentry
            ) and not late_result.attach_stabilization_verified:
                raise RuntimeError(
                    "post-blackout trace requires one verified complete "
                    "main-thread attach boundary"
                )
            if args.allow_blackout_file_write_order_rebase:
                envelope_set_mode = bool(
                    args.blackout_allowed_offset_pairs
                )
                envelope_mode = envelope_set_mode or (
                    args.blackout_expected_command_order_offset is not None
                )
                receipt_row_indices = (
                    late_result.blackout_file_write_rebase_candidate_rows
                    if envelope_mode
                    else late_result.command_order_rebase_rows
                )
                receipt_rows = []
                for row_index in receipt_row_indices:
                    row = offline_rows[row_index]
                    receipt_rows.append(
                        {
                            "row_index": row_index,
                            "start_bit_position": int(row["start"]),
                            "end_bit_position": int(row["end"]),
                            "update": int(row["update"]),
                            "command_number": int(row["command_number"]),
                            "kind": row["kind"],
                            "short_form": bool(row["short_form"]),
                            "success": bool(row["payload"]["success"]),
                        }
                    )
                rebase_receipt: dict[str, Any] = {
                    "mechanism": (
                        "bounded_long_form_file_write_candidate_envelope_set"
                        if envelope_set_mode
                        else (
                            "bounded_long_form_file_write_candidate_envelope"
                            if envelope_mode
                            else "direct_long_form_file_write_result_consumption"
                        )
                    ),
                    "detached_after_update": early_result.last_update,
                    "observed_offset": late_result.command_order_offset,
                    "first_live_command_order": (
                        late_result.attach_stabilization_verified_command_order
                    ),
                    "first_observed_live_command_order": (
                        late_result.first_command_order
                    ),
                    "first_offline_command_order": (
                        None
                        if late_result.attach_stabilization_verified_command_order
                        is None
                        else (
                            late_result.attach_stabilization_verified_command_order
                            + late_result.command_order_offset
                        )
                    ),
                    "fixed_for_entire_post_capture_phase": True,
                }
                if envelope_mode:
                    rebase_receipt.update(
                        {
                            "observed_native_timeline_offset": (
                                late_result.blackout_native_timeline_offset
                            ),
                            "candidate_row_count": len(receipt_rows),
                            "candidate_rows": receipt_rows,
                            "exact_missing_row_identity_available": False,
                            "verification_mode": (
                                "preregistered_finite_offset_set_and_fixed_suffix"
                                if envelope_set_mode
                                else "preregistered_candidate_envelope_and_fixed_suffix"
                            ),
                        }
                    )
                    if envelope_set_mode:
                        rebase_receipt.update(
                            {
                                "allowed_offset_pairs": [
                                    {
                                        "command_order_offset": command_offset,
                                        "native_timeline_offset": timeline_offset,
                                    }
                                    for command_offset, timeline_offset
                                    in args.blackout_allowed_offset_pairs
                                ],
                                "selected_offset_pair": {
                                    "command_order_offset": (
                                        late_result.command_order_offset
                                    ),
                                    "native_timeline_offset": (
                                        late_result.blackout_native_timeline_offset
                                    ),
                                },
                            }
                        )
                else:
                    rebase_receipt.update(
                        {
                            "accounted_row_count": len(receipt_rows),
                            "accounted_rows": receipt_rows,
                        }
                    )
                debugger_blackout = {
                    **debugger_blackout,
                    "command_order_rebase": rebase_receipt,
                }
            trace_phases.append(
                _trace_phase_evidence(
                    name="post_capture",
                    started_perf_counter_ns=late_started_ns,
                    finished_perf_counter_ns=late_finished_ns,
                    result=late_result,
                )
            )
            result = _merge_trace_results(
                early_result,
                late_result,
            )
    else:
        result = trace_demo_commands(
            pid=pid,
            executable=args.runtime_exe,
            address=args.address,
            maximum_hits=args.maximum_hits,
            timeout=args.trace_timeout,
            offline_rows=offline_rows,
            broker_service_blocks=args.broker_service_blocks,
            diagnostic_successful_file_write_padding_rows=tuple(
                args.diagnostic_successful_file_write_padding_rows
            ),
            service_wait_timeout=args.service_wait_timeout,
            quiet_nonservice=args.quiet_nonservice,
            stop_after_update=args.stop_after_update,
            stop_after_command_order=args.stop_after_command_order,
            close_after_terminal_command=(
                args.close_after_terminal_command
            ),
            progress_every_updates=args.progress_every_updates,
            board_seed_address=(
                args.board_seed_address
                if board_seed_in_command_trace
                else None
            ),
            board_seed_override=(
                args.board_seed
                if board_seed_in_command_trace
                else None
            ),
            board_seed_skip_count=(
                args.board_seed_skip_count
                if board_seed_in_command_trace
                else 0
            ),
            global_rng_seed_override=(
                args.global_rng_seed
                if board_seed_in_command_trace
                else None
            ),
            thread_crt_rng_seed_override=(
                args.thread_crt_rng_seed
                if board_seed_in_command_trace
                else None
            ),
            board_seed_observations=board_rng_observations,
            source_bound_board_global_state=(
                source_bound_board_global_state
            ),
            source_bound_board_thread_crt_state=(
                source_bound_board_thread_crt_state
            ),
            source_bound_board_observations=(
                source_bound_board_observations
            ),
            allow_source_bound_board_global_correction=(
                args.allow_source_bound_board_global_correction
            ),
            source_bound_board_precall_global_restore=(
                args.source_bound_board_precall_global_restore
            ),
            allow_pre_stream_commands=args.allow_pre_stream_commands,
            attach_stabilization_after_update=(
                args.attach_at_update
                if args.allow_attach_stabilization
                else None
            ),
            attach_stabilization_main_thread_id=(
                startup_rng.thread_id
                if args.allow_attach_stabilization
                and startup_rng is not None
                else None
            ),
            pre_attach_file_write_debt_before_update=(
                args.attach_at_update
                if args.allow_pre_attach_file_write_debt
                else None
            ),
            font_cache_manifest=font_cache_manifest,
            font_cache_manifest_sha256=font_cache_manifest_sha256,
            font_cache_manifest_main_pak_sha256=(
                font_cache_manifest_main_pak_sha256
            ),
            startup_priority_bias=in_trace_startup_priority,
            startup_handoff_main_thread_id=(
                startup_handoff_main_thread_id
            ),
            startup_handoff_launcher_suspend_previous_count=(
                startup_handoff_launcher_suspend_previous_count
            ),
            global_mtrand_restore_state=(
                global_mtrand_restore_state
            ),
            global_mtrand_expected_before_state=(
                global_mtrand_expected_before_state
            ),
            global_mtrand_restore_command_order=(
                args.global_mtrand_restore_command_order
            ),
            global_mtrand_restore_observations=(
                global_mtrand_restore_observations
            ),
            global_mtrand_allow_dynamic_before=(
                args.global_mtrand_allow_dynamic_before
            ),
            qrand_restore_state=qrand_restore_state,
            qrand_expected_before_state=qrand_expected_before_state,
            qrand_restore_command_order=(
                args.qrand_restore_command_order
            ),
            qrand_restore_observations=(
                qrand_restore_observations
            ),
            thread_crt_restore_state=thread_crt_restore_state,
            thread_crt_expected_before_state=(
                thread_crt_expected_before_state
            ),
            thread_crt_restore_command_order=(
                args.thread_crt_restore_command_order
            ),
            thread_crt_restore_observations=(
                thread_crt_restore_observations
            ),
            thread_crt_allow_dynamic_before=(
                args.thread_crt_allow_dynamic_before
            ),
            initial_global_mtrand_oracle=(
                initial_global_mtrand_oracle
            ),
            initial_global_mtrand_observations=(
                initial_global_mtrand_observations
                if initial_global_mtrand_requested
                else None
            ),
            suspend_main_thread_on_stop=(
                args.suspend_main_thread_on_stop
            ),
        )
    if in_trace_startup_priority is not None:
        startup_priority_phase = _in_trace_startup_priority_evidence(
            in_trace_startup_priority
        )
    trace_finished_perf_counter_ns = time.perf_counter_ns()
    finished_utc = datetime.now(timezone.utc).isoformat()
    if args.result_json is not None:
        result_sha256 = _write_result_json(
            args.result_json,
            dmo=args.dmo,
            runtime_exe=args.runtime_exe,
            runtime_process_id=pid,
            started_utc=started_utc,
            finished_utc=finished_utc,
            trace_started_perf_counter_ns=(
                trace_started_perf_counter_ns
            ),
            trace_finished_perf_counter_ns=(
                trace_finished_perf_counter_ns
            ),
            args=args,
            result=result,
            trace_phases=trace_phases,
            debugger_blackout=debugger_blackout,
            board_seed_phase=board_seed_phase,
            startup_priority_phase=startup_priority_phase,
            startup_rng=startup_rng,
            board_rng_observations=board_rng_observations,
            source_bound_board_observations=(
                source_bound_board_observations
            ),
            gameplay_mtrand_sync_observations=(
                gameplay_mtrand_sync_observations
            ),
            gameplay_mtrand_sync_receipt=(
                gameplay_mtrand_sync_receipt
                if gameplay_mtrand_requested
                else None
            ),
            initial_global_mtrand_observations=(
                initial_global_mtrand_observations
            ),
            startup_global_mtrand_observations=(
                startup_global_mtrand_observations
            ),
            startup_global_mtrand_hidden_draw_observations=(
                startup_global_mtrand_hidden_draw_observations
            ),
            startup_global_mtrand_observation_receipt=(
                startup_global_mtrand_observation_receipt
                if startup_global_mtrand_observation_requested
                else None
            ),
            global_mtrand_restore_observations=(
                global_mtrand_restore_observations
            ),
            qrand_restore_observations=(
                qrand_restore_observations
            ),
            thread_crt_restore_observations=(
                thread_crt_restore_observations
            ),
        )
        print(
            f"result_json={args.result_json.resolve()} "
            f"sha256={result_sha256}",
            flush=True,
        )
    if result.failure_count:
        return 2
    if args.board_seed is not None:
        if source_bound_board_requested:
            applied_board_rows = [
                row
                for row in board_rng_observations
                if row.get("source_bound_board_anchor_applied") is True
            ]
            if (
                not board_rng_observations
                or len(applied_board_rows) != 1
                or board_rng_observations[-1] is not applied_board_rows[0]
            ):
                return 2
        elif (
            len(board_rng_observations)
            != args.board_seed_skip_count + 1
        ):
            return 2
    if (
        source_bound_board_requested
        and len(source_bound_board_observations) != 1
    ):
        return 2
    if gameplay_mtrand_requested:
        assert gameplay_mtrand_oracle is not None
        expected_gameplay_hits = len(gameplay_mtrand_oracle.entries)
        if (
            gameplay_mtrand_sync_receipt.get("status") != "PASS"
            or gameplay_mtrand_sync_receipt.get("oracle_complete")
            is not True
            or gameplay_mtrand_sync_receipt.get(
                "hardware_breakpoint_restored"
            )
            is not True
            or gameplay_mtrand_sync_receipt.get(
                "hardware_breakpoint_restore_error"
            )
            is not None
            or gameplay_mtrand_sync_receipt.get("hit_count")
            != expected_gameplay_hits
            or len(gameplay_mtrand_sync_observations)
            != expected_gameplay_hits
        ):
            return 2
    if startup_global_mtrand_observation_requested:
        assert startup_global_mtrand_observation_oracle is not None
        startup_receipt_path = (
            args.startup_global_mtrand_observation_receipt_json
        )
        assert startup_receipt_path is not None
        expected_startup_hits = len(
            startup_global_mtrand_observation_oracle.entries
        )
        startup_receipt_payload = json.loads(
            startup_receipt_path.read_text(encoding="utf-8")
        )
        if (
            startup_global_mtrand_observation_receipt.get("status")
            != "PASS"
            or startup_global_mtrand_observation_receipt.get(
                "oracle_complete"
            )
            is not True
            or startup_global_mtrand_observation_receipt.get(
                "hardware_breakpoint_restored"
            )
            is not True
            or startup_global_mtrand_observation_receipt.get(
                "hardware_breakpoint_restore_error"
            )
            is not None
            or startup_global_mtrand_observation_receipt.get("hit_count")
            != expected_startup_hits
            or len(startup_global_mtrand_observations)
            != expected_startup_hits
            or startup_receipt_payload.get("status") != "PASS"
            or startup_receipt_payload.get("process_memory_writes") != 0
            or startup_receipt_payload.get("process_memory_mutation")
            is not False
        ):
            return 2
        if startup_hidden_draw_requested:
            runtime_hidden_receipt = (
                startup_global_mtrand_observation_receipt.get(
                    "hidden_draw_interval"
                )
            )
            persisted_hidden_receipt = startup_receipt_payload.get(
                "hidden_draw_interval"
            )
            if (
                not isinstance(runtime_hidden_receipt, Mapping)
                or runtime_hidden_receipt.get("status") != "PASS"
                or runtime_hidden_receipt.get("process_memory_writes") != 0
                or runtime_hidden_receipt.get("process_memory_mutation")
                is not False
                or runtime_hidden_receipt.get("index_write_count")
                != len(startup_global_mtrand_hidden_draw_observations)
                or not isinstance(persisted_hidden_receipt, Mapping)
                or persisted_hidden_receipt.get("status") != "PASS"
                or persisted_hidden_receipt.get("process_memory_writes") != 0
                or persisted_hidden_receipt.get("process_memory_mutation")
                is not False
            ):
                return 2
    if initial_global_mtrand_requested:
        if (
            len(initial_global_mtrand_observations) != 1
            or initial_global_mtrand_observations[0].get(
                "atomic_window_completed"
            )
            is not True
            or initial_global_mtrand_observations[0].get(
                "call_target_entered"
            )
            is not True
            or initial_global_mtrand_observations[0].get(
                "call_return_observed"
            )
            is not True
            or initial_global_mtrand_observations[0].get(
                "other_threads_resumed_after_call_return"
            )
            is not True
            or initial_global_mtrand_observations[0].get(
                "persistent_file_modified"
            )
            is not False
        ):
            return 2
    if args.source_bound_board_precall_global_restore and (
        len(source_bound_board_observations) != 1
        or not isinstance(
            source_bound_board_observations[0].get(
                "global_precall_restore"
            ),
            dict,
        )
        or source_bound_board_observations[0][
            "global_precall_restore"
        ].get("atomic_window_completed")
        is not True
    ):
        return 2
    if qrand_requested and len(qrand_restore_observations) != 1:
        return 2
    if (
        global_mtrand_requested
        and len(global_mtrand_restore_observations) != 1
    ):
        return 2
    if (
        thread_crt_requested
        and len(thread_crt_restore_observations) != 1
    ):
        return 2
    if phased:
        return (
            0
            if (
                len(trace_phases) == 2
                and result.exited
                and result.exit_code == 0
            )
            else 2
        )
    if (
        args.accept_normal_exit
        and result.exited
        and result.exit_code == 0
    ):
        return 0
    if args.stop_after_command_order is not None:
        return 0 if result.stopped_at_command_order else 2
    if args.stop_after_update is not None:
        return 0 if result.stopped_at_update else 2
    return 0 if result.hits == args.maximum_hits else 2


if __name__ == "__main__":
    raise SystemExit(main())
