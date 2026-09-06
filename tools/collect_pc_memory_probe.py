"""Collect a read-only memory probe from a deterministic retail DMO replay.

This is a reverse-engineering diagnostic, not a PC Golden producer.  It
reuses the proven v4 startup trace, prestate restore, native title-bar repaint
handshake, and exact framework-tick freeze.  While the replay is frozen it
scans committed readable memory for a declared int32 value and records enough
object/pointer context to locate the owning gameplay structure.

The script always restores the caller-supplied host snapshot.  Failed startup
races are retried from the exact replay prestate and never become evidence.
The opt-in formal exact-step mode additionally binds process creation identity,
verified square DWM corners, one discarded warmup frame, and every retained
post-update BGRA snapshot.  The formal full-state mode instead keeps one
lossless visual anchor and every raw Board object at every exact update.  Both
produce formal evidence *sources*; a PC Golden still requires a versioned
package and independent verification.
"""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import time
from typing import Any, Callable, Iterable, Mapping

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tools.collect_pc_golden_v4 import (
    CollectionError,
    DIRECT_FIXED_SEED_LAUNCH_MODE,
    DIRECT_NATURAL_SEED_LAUNCH_MODE,
    DIRECT_NATURAL_STRICT_BROKER_LAUNCH_MODE,
    NONCLIENT_DRAG_REPAINT_MECHANISM,
    SET_WINDOW_POS_REPAINT_MECHANISM,
    STEAM_LAUNCH_MODE,
    _activate_window,
    _child_environment,
    _terminate_exact_runtime,
    _validate_trace_result,
    _wait_for_capture_window,
    _wait_for_repaint_guard,
    _wait_for_runtime,
    _window_position_repaint_handshake,
    _window_repaint_handshake,
)
from tools.capture_dxgi import windows_process_identity
from tools.control_popcap_replay import (
    expected_multiplier,
    main_window_for_pid,
    post_char,
    step_to,
)
from tools.freeze_popcap_replay import freeze_replay
from tools.inspect_popcap_replay import (
    ReplayState,
    _read_region,
    _readable_regions,
    close_process,
    open_process_readonly,
    read_process_bytes,
    read_replay_state,
)
from tools.launch_popcap_replay import process_ids_by_name
from tools.pc_state_transaction import restore_state
from tools.pc_external_input_guard import (
    ExternalInputGuard,
    ExternalInputGuardError,
)
from tools.popcap_global_mtrand_call_oracle import (
    load_initial_global_mtrand_call_oracle,
    load_startup_global_mtrand_observation_oracle,
)
from tools.snapshot_dxgi_window import (
    DIAGNOSTIC_SNAPSHOT_SEMANTICS,
    FORMAL_EXACT_STEP_SNAPSHOT_SEMANTICS,
    snapshot as snapshot_dxgi_window,
)
from tools.trace_popcap_shutdown import (
    PROCESS_ACCESS,
    kernel32,
    read_memory,
    write_memory,
)
from zuma_rl.pc_memory_trajectory import (
    BOARD_FRUIT_ACTIVE_POINT_POINTER_OFFSET,
    BOARD_FRUIT_COLLECTING_OFFSET,
    BOARD_FRUIT_EXPIRY_TIME_OFFSET,
    BOARD_FRUIT_SELECTED_POINT_OFFSET,
    BOARD_NATIVE_GAME_TIME_OFFSET,
    BOARD_UPDATE_COUNT_OFFSET,
    INITIAL_SAMPLE_BARRIER,
    STEPPED_SAMPLE_BARRIER,
    TRAJECTORY_SAMPLE_PHASE,
    TRAJECTORY_TICK_VERSION,
    TRAJECTORY_VERSION,
)
from zuma_rl.pc_memory_evidence import (
    INDEPENDENT_SCORE_BINDING_ACCEPTANCE_RULE,
    INDEPENDENT_SCORE_BINDING_MODE,
    SCORE_BINDING_INDEPENDENT_VERSION,
    SCORE_BINDING_LEGACY_VERSION,
    SCORE_BINDING_SCHEMA,
    canonical_report_bytes,
)
from zuma_rl.pc_external_input import (
    EXTERNAL_INPUT_GUARD_BINDING_SCHEMA,
    EXTERNAL_INPUT_GUARD_BINDING_VERSION,
    canonical_external_input_bytes,
)
from zuma_rl.retail_dmo_provenance import (
    RetailDmoProvenanceError,
    certifying_recording_outcome,
    read_certifying_provenance,
)


NATURAL_STRICT_REPLAY_BINDING_SCHEMA = (
    "zuma-rl.pc-natural-strict-command-replay-binding"
)
NATURAL_STRICT_REPLAY_BINDING_VERSION = 2
SOURCE_BOUND_STRICT_REPLAY_BINDING_SCHEMA = (
    "zuma-rl.pc-source-bound-original-command-replay-binding"
)
SOURCE_BOUND_STRICT_REPLAY_BINDING_VERSION = 1
OUTCOME_SOURCE_BOUND_STRICT_REPLAY_BINDING_VERSION = 2
FORMAL_FULL_STATE_CLASSIFICATION = (
    "formal_pc_full_state_exact_step_source"
)
FORMAL_FULL_STATE_SNAPSHOT_SEMANTICS = (
    "formal_full_state_external_lossless_anchor"
)
from zuma_rl.pc_rng_trajectory import RNG_TRAJECTORY_VERSION
from zuma_rl.pc_protocol_evidence import PcStateSnapshot


TEXT_START = 0x00401000
TEXT_END = 0x0094AA9C
RDATA_START = 0x0094B000
RDATA_END = 0x009E3F28
IMAGE_START = 0x00400000
IMAGE_END = 0x00ECC000
G_SEXY_APP_BASE_ADDRESS = 0x009FC740
G_FRAMEWORK_MTRAND_ADDRESS = 0x00A313C0
G_CURVE_PLAN_EXHAUSTED_ADDRESS = 0x009E8252
MTRAND_STATE_WORDS = 624
MTRAND_STATE_BYTES = (MTRAND_STATE_WORDS + 1) * 4
ACTIVE_BOARD_OFFSET = 0x834
BOARD_VTABLE = 0x0096356C
BOARD_EMBEDDED_VTABLE_OFFSET = 0x88
BOARD_EMBEDDED_VTABLE = 0x0096368C
BOARD_SCORE_OFFSET = 0x104
BOARD_SCORE_TARGET_OFFSET = 0x108
BOARD_COLOR_COUNTS_OFFSET = 0xE4
BOARD_COLOR_COUNT_SLOTS = 6
BOARD_PRIMARY_CHILD_OFFSET = 0x68C
BOARD_CURVE_MANAGER_OFFSET = 0x9C
BOARD_FIRED_BULLET_LIST_OFFSET = 0x698
BOARD_QRAND_POINTER_OFFSET = 0x7A8
BOARD_DISPLAYED_SCORE_OFFSET = 0xEFC
BOARD_MODE_FLAG_1064_OFFSET = 0x1064
BOARD_DUMP_SIZE = 0x1100
BOARD_PRIMARY_CHILD_DUMP_SIZE = 0x32C
SHOOTER_VTABLE = 0x009670E0
SHOOTER_BULLET_POINTER_OFFSETS = (0x130, 0x134)
BULLET_VTABLE = 0x009636E4
BULLET_OBJECT_SIZE = 0x18C
BULLET_GAP_CONTAINER_OFFSET = 0x170
BULLET_GAP_SENTINEL_POINTER_OFFSET = 0x174
BULLET_GAP_COUNT_OFFSET = 0x178
BULLET_CURVE_POINTS_OFFSET = 0x17C
BULLET_CURVE_POINT_COUNT = 4
BULLET_GAP_NODE_SIZE = 0x14
MAX_BULLET_GAP_ENTRIES = 64
CURVE_MANAGER_VTABLE = 0x0096A204
CURVE_MANAGER_DUMP_SIZE = 0x40C
CURVE_MANAGER_CURVE_ARRAY_OFFSET = 0x16C
CURVE_MANAGER_CURVE_COUNT_OFFSET = 0x360
CURVE_MANAGER_POST_ZUMA_TIMER_OFFSET = 0x400
CURVE_MANAGER_POST_ZUMA_RAMP_404_OFFSET = 0x404
CURVE_MANAGER_POST_ZUMA_RAMP_408_OFFSET = 0x408
CURVE_DUMP_SIZE = 0x200
CURVE_PLANNED_VECTOR_BEGIN_OFFSET = 0x34
CURVE_PLANNED_VECTOR_END_OFFSET = 0x38
CURVE_PLANNED_VECTOR_CAPACITY_OFFSET = 0x3C
CURVE_PLANNED_ITEM_SIZE = 0x14
CURVE_ADD_PLAN_ENABLED_OFFSET = 0x1A3
CURVE_INTRUSIVE_LIST_OFFSETS = (0x50, 0x5C, 0x68)
BALL_VTABLE = 0x00960160
BALL_OBJECT_SIZE = 0x134
BALL_COLOR_ID_OFFSET = 0x14
MAX_CURVES = 16
MAX_CURVE_PLANNED_ITEMS = 4096
MAX_CURVE_LIST_ITEMS = 2048
MAX_FIRED_BULLETS = 64
QRAND_OBJECT_SIZE = 0x48
QRAND_VECTOR_LAYOUT = (
    (0x08, "weights", "float32"),
    (0x18, "sways", "float32"),
    (0x28, "last_hit", "int32"),
    (0x38, "previous_hit", "int32"),
)
MAX_QRAND_VECTOR_ITEMS = 64

DIAGNOSTIC_COMPACT_STATE_EXPECTED_PRE_DIFFERENCES = frozenset(
    {
        "/board_color_counts/2",
        "/board_color_counts/3",
        "/bullets/1/ball/color_id",
        "/crt",
        "/curves/0/lists/0/entities/0/ball/color_id",
        "/qrand/selected_index",
        "/qrand/vectors/last_hit/0",
        "/qrand/vectors/last_hit/1",
        "/qrand/vectors/last_hit/2",
        "/qrand/vectors/last_hit/3",
        "/qrand/vectors/previous_hit/1",
        "/qrand/vectors/sways/2",
        "/qrand/vectors/sways/3",
    }
)
DIAGNOSTIC_COMPACT_STATE_ALLOWED_PRE_DIFFERENCES = frozenset(
    [
        "/crt",
        "/bullets/0/ball/color_id",
        "/bullets/1/ball/color_id",
        "/curves/0/lists/0/entities/0/ball/color_id",
        "/qrand/selected_index",
    ]
    + [f"/board_color_counts/{index}" for index in range(4)]
    + [
        f"/qrand/vectors/{name}/{index}"
        for name in ("sways", "last_hit", "previous_hit")
        for index in range(4)
    ]
)
DIAGNOSTIC_BALL_PROJECTION_KEYS = (
    "ball_id",
    "color_id",
    "curve_distance",
    "orientation_radians",
    "previous_orientation_radians",
    "angular_step_radians",
    "position_x",
    "position_y",
    "scale",
    "radius",
    "powerup_previous_type",
    "powerup_primary_type",
    "powerup_secondary_type",
    "flags_b4_c2_hex",
)
DIAGNOSTIC_BULLET_SUBCLASS_PROJECTION_KEYS = (
    "velocity_x",
    "velocity_y",
    "field_148_float",
    "field_14c_float",
    "field_158_float",
    "field_15c_float",
    "heading_radians",
    "field_164_i32",
    "fired",
    "flags_16a_16c_hex",
    "field_178_i32",
    "curve_points",
    "gap_entry_count",
)
LIGHT_BALL_IDENTITY_BYTES = 0x124
LIGHT_BALL_RENDER_FLOAT_FIELDS = (
    "curve_distance",
    "orientation_radians",
    "previous_orientation_radians",
    "angular_step_radians",
    "position_x",
    "position_y",
    "scale",
    "radius",
)
FREEZE_STATE_PREFIX_BYTES = 60
FREEZE_STATE_SUFFIX_BYTES = 108
FREEZE_STATE_BYTES = (
    FREEZE_STATE_PREFIX_BYTES + FREEZE_STATE_SUFFIX_BYTES
)
MAX_TRAJECTORY_TICKS = 2048
TRAJECTORY_SNAPSHOT_TIMEOUT_SECONDS = 2.0
MAX_POINTER_REFERENCE_TARGETS = 256
NATURAL_EXACT_FREEZE_TRIGGER_LEAD_TICKS = 2
NATURAL_EXACT_FREEZE_TOTAL_MINUS_COUNT = 12
FORMAL_FRUIT_TRIGGER_TRANSCRIPT_SCHEMA = (
    "zuma-rl.pc-formal-fruit-lifecycle-trigger-transcript"
)
FORMAL_FRUIT_TRIGGER_BINDING_SCHEMA = (
    "zuma-rl.pc-formal-fruit-lifecycle-trigger-binding"
)
FORMAL_FRUIT_TRIGGER_VERSION = 1


class ProbeError(RuntimeError):
    """A compact diagnostic failure."""


class FormalInputContaminationError(ProbeError):
    """A formal run observed external keyboard or mouse input."""


class _ValidatedCompletedTraceGuard:
    """Tell the repaint guard that a completed trace already passed review."""

    @staticmethod
    def poll() -> None:
        return None


def _uses_natural_exact_freeze(
    launch_mode: object,
    *,
    source_bound_formal_replay: bool,
) -> bool:
    """Return whether a formal capture needs the twelve-minus barrier."""

    return (
        launch_mode
        in {
            STEAM_LAUNCH_MODE,
            DIRECT_NATURAL_SEED_LAUNCH_MODE,
            DIRECT_NATURAL_STRICT_BROKER_LAUNCH_MODE,
        }
        or source_bound_formal_replay
    )


def _natural_exact_freeze(
    *,
    pid: int,
    slowdown_update: int,
    probe_update: int,
    demo_length: int,
    timeout: float,
) -> ReplayState:
    """Reach a completed exact tick without relying on a racing counter.

    ``mUpdateCount`` increments near the start of a retail update.  Natural
    execution can therefore leave the nominal six-minus freeze before the
    collector reaches its first trajectory read.  Stop two ticks early,
    reduce autonomous cadence to a negligible value, and cross the final
    boundary only through the replay's native ``N`` post-update barrier.
    """

    trigger_update = (
        probe_update - NATURAL_EXACT_FREEZE_TRIGGER_LEAD_TICKS
    )
    if trigger_update <= slowdown_update:
        raise ProbeError("natural_exact_freeze_window_invalid")
    preliminary = freeze_replay(
        pid=pid,
        slowdown_at=slowdown_update,
        freeze_update=trigger_update,
        demo_length=demo_length,
        timeout=timeout,
        allow_slowdown_overshoot=True,
    )
    hwnd = main_window_for_pid(pid)
    handle = open_process_readonly(pid)
    try:
        state = preliminary
        for minus_count in range(
            7,
            NATURAL_EXACT_FREEZE_TOTAL_MINUS_COUNT + 1,
        ):
            post_char(hwnd, "-")
            deadline = time.monotonic() + 5.0
            target_multiplier = expected_multiplier(minus_count)
            while True:
                state = read_replay_state(
                    handle,
                    preliminary.multiplier_address,
                )
                if state.update_count >= probe_update:
                    raise ProbeError(
                        "natural_exact_freeze_settle_overshot"
                    )
                if state.update_multiplier == target_multiplier:
                    break
                if time.monotonic() >= deadline:
                    raise ProbeError(
                        "natural_exact_freeze_multiplier_timeout"
                    )
                time.sleep(0.001)
        state = step_to(
            hwnd,
            handle,
            preliminary.multiplier_address,
            probe_update,
            timeout_per_step=5.0,
        )
        if (
            state.update_count != probe_update
            or state.update_multiplier
            != expected_multiplier(
                NATURAL_EXACT_FREEZE_TOTAL_MINUS_COUNT
            )
            or state.fast_forward_target != probe_update
            or state.fast_forward_to_marker
            or state.fast_forward_step
        ):
            raise ProbeError("natural_exact_freeze_barrier_invalid")
        return state
    finally:
        close_process(handle)


def _normalize_formal_fruit_lifecycle_trigger(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the frozen, outcome-independent fruit trigger contract."""

    required = {
        "monitor_start_update",
        "maximum_framework_update",
        "remaining_threshold_ticks",
        "slowdown_lead_updates",
        "freeze_lead_updates",
        "trajectory_tick_count",
        "poll_interval_seconds",
    }
    if set(value) != required:
        raise ProbeError("formal_fruit_trigger_contract_invalid")
    integer_names = required - {"poll_interval_seconds"}
    normalized: dict[str, Any] = {}
    for name in integer_names:
        candidate = value.get(name)
        if (
            isinstance(candidate, bool)
            or not isinstance(candidate, int)
            or candidate < 0
        ):
            raise ProbeError("formal_fruit_trigger_contract_invalid")
        normalized[name] = candidate
    interval = value.get("poll_interval_seconds")
    if (
        isinstance(interval, bool)
        or not isinstance(interval, (int, float))
        or not math.isfinite(float(interval))
        or not 0 < float(interval) <= 0.1
    ):
        raise ProbeError("formal_fruit_trigger_contract_invalid")
    normalized["poll_interval_seconds"] = float(interval)
    if (
        normalized["maximum_framework_update"]
        <= normalized["monitor_start_update"]
        or normalized["slowdown_lead_updates"] < 4
        or normalized["freeze_lead_updates"]
        <= normalized["slowdown_lead_updates"]
        or normalized["remaining_threshold_ticks"]
        <= normalized["freeze_lead_updates"] + 32
        or not 2
        <= normalized["trajectory_tick_count"]
        <= MAX_TRAJECTORY_TICKS
        or normalized["trajectory_tick_count"]
        <= normalized["remaining_threshold_ticks"]
    ):
        raise ProbeError("formal_fruit_trigger_contract_invalid")
    return normalized


def _read_formal_fruit_trigger_observation(
    handle: int,
    *,
    multiplier_address: int,
) -> dict[str, Any] | None:
    """Read one update-coherent replay/Board fruit observation."""

    before = read_replay_state(handle, multiplier_address)
    sexy_app_base = _u32(
        read_process_bytes(handle, G_SEXY_APP_BASE_ADDRESS, 4),
        0,
    )
    board_address = _u32(
        read_process_bytes(
            handle,
            sexy_app_base + ACTIVE_BOARD_OFFSET,
            4,
        ),
        0,
    )
    board = read_process_bytes(handle, board_address, BOARD_DUMP_SIZE)
    after = read_replay_state(handle, multiplier_address)
    if before.update_count != after.update_count:
        return None
    if (
        _u32(board, 0) != BOARD_VTABLE
        or _u32(board, BOARD_EMBEDDED_VTABLE_OFFSET)
        != BOARD_EMBEDDED_VTABLE
    ):
        raise ProbeError("formal_fruit_trigger_board_identity_invalid")
    pointer = _u32(board, BOARD_FRUIT_ACTIVE_POINT_POINTER_OFFSET)
    selected = _i32(board, BOARD_FRUIT_SELECTED_POINT_OFFSET)
    collecting_raw = board[BOARD_FRUIT_COLLECTING_OFFSET]
    if collecting_raw not in {0, 1}:
        raise ProbeError("formal_fruit_trigger_collecting_invalid")
    native_game_time = _i32(board, BOARD_NATIVE_GAME_TIME_OFFSET)
    board_update_count = _i32(board, BOARD_UPDATE_COUNT_OFFSET)
    expiry_time = _i32(board, BOARD_FRUIT_EXPIRY_TIME_OFFSET)
    if (
        native_game_time < 0
        or board_update_count < 0
        or (pointer == 0 and selected not in {-1, 0})
        or (pointer != 0 and selected < 0)
    ):
        raise ProbeError("formal_fruit_trigger_state_invalid")
    return {
        "framework_update": after.update_count,
        "native_game_time": native_game_time,
        "board_update_count": board_update_count,
        "active": pointer != 0,
        "active_point_pointer": pointer,
        "selected_point_index": selected,
        "collecting": bool(collecting_raw),
        "expiry_time": expiry_time,
        "remaining_ticks": (
            expiry_time - native_game_time if pointer != 0 else None
        ),
    }


def _freeze_on_first_fruit_lifecycle_window(
    *,
    pid: int,
    demo_length: int,
    contract: Mapping[str, Any],
) -> tuple[ReplayState, int, Mapping[str, Any]]:
    """Freeze the first observed fruit lifecycle near its natural terminal.

    Every update-coherent observation is retained.  The first active
    lifecycle may not be skipped: collection, disappearance before the
    threshold, or an already-too-late first observation fails the source.
    """

    config = _normalize_formal_fruit_lifecycle_trigger(contract)
    latest_possible_end = (
        config["maximum_framework_update"]
        + config["freeze_lead_updates"]
        + config["trajectory_tick_count"]
        - 1
    )
    if latest_possible_end > demo_length:
        raise ProbeError("formal_fruit_trigger_exceeds_replay")

    handle = open_process_readonly(pid)
    observations: list[dict[str, Any]] = []
    first_active_index: int | None = None
    lifecycle_pointer: int | None = None
    selected: dict[str, Any] | None = None
    try:
        sexy_app_base = _u32(
            read_process_bytes(handle, G_SEXY_APP_BASE_ADDRESS, 4),
            0,
        )
        multiplier_address = sexy_app_base + 0x4D0
        initial = read_replay_state(handle, multiplier_address)
        if (
            initial.update_multiplier != expected_multiplier(0)
            or initial.frame_time_ms != 10
            or initial.update_count
            > config["monitor_start_update"] + 2
        ):
            raise ProbeError("formal_fruit_trigger_monitor_start_missed")
        deadline = time.monotonic() + max(
            180.0,
            (config["maximum_framework_update"] - initial.update_count)
            * 0.05,
        )
        last_update = -1
        while True:
            if time.monotonic() >= deadline:
                raise ProbeError("formal_fruit_trigger_timeout")
            observation = _read_formal_fruit_trigger_observation(
                handle,
                multiplier_address=multiplier_address,
            )
            if observation is None:
                continue
            update = int(observation["framework_update"])
            if update < config["monitor_start_update"]:
                time.sleep(config["poll_interval_seconds"])
                continue
            if update > config["maximum_framework_update"]:
                raise ProbeError("formal_fruit_trigger_not_observed")
            if update == last_update:
                time.sleep(config["poll_interval_seconds"])
                continue
            if update < last_update:
                raise ProbeError("formal_fruit_trigger_update_regressed")
            observations.append(observation)
            last_update = update

            if first_active_index is None:
                if not observation["active"]:
                    time.sleep(config["poll_interval_seconds"])
                    continue
                first_active_index = len(observations) - 1
                lifecycle_pointer = int(
                    observation["active_point_pointer"]
                )
            elif (
                not observation["active"]
                or observation["active_point_pointer"] != lifecycle_pointer
            ):
                raise ProbeError(
                    "formal_fruit_trigger_first_lifecycle_ended_early"
                )

            if observation["collecting"]:
                raise ProbeError(
                    "formal_fruit_trigger_first_lifecycle_collected"
                )
            remaining = observation["remaining_ticks"]
            if not isinstance(remaining, int) or remaining <= 0:
                raise ProbeError("formal_fruit_trigger_expiry_invalid")
            if remaining <= config["freeze_lead_updates"] + 32:
                raise ProbeError(
                    "formal_fruit_trigger_first_observation_too_late"
                )
            if remaining <= config["remaining_threshold_ticks"]:
                selected = observation
                break
            time.sleep(config["poll_interval_seconds"])
    finally:
        close_process(handle)

    if selected is None or first_active_index is None:
        raise ProbeError("formal_fruit_trigger_selection_missing")
    selected_update = int(selected["framework_update"])
    slowdown_update = selected_update + config["slowdown_lead_updates"]
    freeze_update = selected_update + config["freeze_lead_updates"]
    frozen_state = _natural_exact_freeze(
        pid=pid,
        slowdown_update=slowdown_update,
        probe_update=freeze_update,
        demo_length=demo_length,
        timeout=180.0,
    )
    frozen_handle = open_process_readonly(pid)
    try:
        frozen_observation = _read_formal_fruit_trigger_observation(
            frozen_handle,
            multiplier_address=frozen_state.multiplier_address,
        )
    finally:
        close_process(frozen_handle)
    if (
        frozen_observation is None
        or frozen_observation["framework_update"] != freeze_update
        or frozen_observation["active"] is not True
        or frozen_observation["active_point_pointer"] != lifecycle_pointer
        or frozen_observation["collecting"] is not False
        or not isinstance(frozen_observation["remaining_ticks"], int)
        or frozen_observation["remaining_ticks"] <= 0
    ):
        raise ProbeError("formal_fruit_trigger_frozen_state_invalid")
    trajectory_end_update = (
        freeze_update + config["trajectory_tick_count"] - 1
    )
    transcript = {
        "schema": FORMAL_FRUIT_TRIGGER_TRANSCRIPT_SCHEMA,
        "version": FORMAL_FRUIT_TRIGGER_VERSION,
        "status": "PASS",
        "selection_rule": (
            "first_observed_active_fruit_lifecycle_at_or_below_"
            "remaining_threshold_no_skips"
        ),
        "process_id": pid,
        "config": config,
        "observations": observations,
        "selection": {
            "first_active_observation_index": first_active_index,
            "threshold_observation_index": len(observations) - 1,
            "threshold_observation": selected,
            "slowdown_update": slowdown_update,
            "freeze_update": freeze_update,
            "trajectory_end_update": trajectory_end_update,
            "trajectory_tick_count": config["trajectory_tick_count"],
            "frozen_observation": frozen_observation,
        },
    }
    return frozen_state, trajectory_end_update, transcript


def _sha256_path(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _runtime_source_executable_from_plan(
    plan: Mapping[str, Any],
) -> Path:
    """Return the signed retail wrapper that owns the runtime payload."""

    runtime = plan.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ProbeError("runtime_source_identity_invalid")
    source_value = runtime.get("runtime_source_executable")
    expected_sha256 = runtime.get("runtime_source_sha256")
    if (
        not isinstance(source_value, str)
        or not source_value
        or not Path(source_value).is_absolute()
        or not isinstance(expected_sha256, str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", expected_sha256) is None
    ):
        raise ProbeError("runtime_source_identity_invalid")
    try:
        source = Path(source_value).resolve(strict=True)
        if not source.is_file() or _sha256_path(source) != expected_sha256:
            raise ProbeError("runtime_source_identity_invalid")
    except ProbeError:
        raise
    except OSError:
        raise ProbeError("runtime_source_identity_invalid") from None
    return source


def _capture_runtime_process_identity(
    target: Any,
    *,
    runtime_pid: int,
    runtime_executable: Path,
    runtime_source_executable: Path,
    expected_runtime_sha256: str | None = None,
) -> dict[str, Any]:
    """Bind the live unpacked process to its signed retail source."""

    runtime_identity = dict(
        windows_process_identity(
            target,
            process_name=runtime_executable.name,
            runtime_source_executable=runtime_source_executable,
        )
    )
    if expected_runtime_sha256 is None:
        expected_runtime_sha256 = _sha256_path(runtime_executable)
    elif not _is_canonical_sha256(expected_runtime_sha256):
        raise ProbeError("trajectory_process_identity_invalid")
    if (
        runtime_identity.get("process_id") != runtime_pid
        or runtime_identity.get("executable_sha256")
        != expected_runtime_sha256
    ):
        raise ProbeError("trajectory_process_identity_invalid")
    return runtime_identity


def _sha256_bytes(payload: bytes) -> str:
    import hashlib

    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _diagnostic_thread_crt_state_mutator(
    *,
    source_path: Path,
    source_framework_update: int,
) -> Callable[[int, Mapping[str, Any], Path], Mapping[str, Any]]:
    """Build one explicitly diagnostic, source-bound 4-byte CRT restore."""

    source_path = source_path.resolve()
    try:
        source_payload = source_path.read_bytes()
    except OSError as error:
        raise ProbeError(
            "diagnostic_thread_crt_source_unreadable"
        ) from error
    if len(source_payload) != 4:
        raise ProbeError("diagnostic_thread_crt_source_size_invalid")
    if (
        isinstance(source_framework_update, bool)
        or not isinstance(source_framework_update, int)
        or source_framework_update < 0
    ):
        raise ProbeError("diagnostic_thread_crt_source_update_invalid")
    source_sha256 = _sha256_bytes(source_payload)
    source_state = struct.unpack("<I", source_payload)[0]

    def mutate(
        pid: int,
        pre_mutation_board: Mapping[str, Any],
        _attempt_root: Path,
    ) -> Mapping[str, Any]:
        try:
            current_source_payload = source_path.read_bytes()
        except OSError as error:
            raise ProbeError(
                "diagnostic_thread_crt_source_unreadable"
            ) from error
        if (
            current_source_payload != source_payload
            or _sha256_bytes(current_source_payload) != source_sha256
        ):
            raise ProbeError("diagnostic_thread_crt_source_changed")

        thread_crt = pre_mutation_board.get("thread_crt_rand")
        if not isinstance(thread_crt, Mapping):
            raise ProbeError("diagnostic_thread_crt_live_state_missing")
        state_address = thread_crt.get("rand_state_address")
        declared_live_state = thread_crt.get("state")
        if (
            isinstance(state_address, bool)
            or not isinstance(state_address, int)
            or state_address <= 0
            or isinstance(declared_live_state, bool)
            or not isinstance(declared_live_state, int)
            or not 0 <= declared_live_state <= 0xFFFFFFFF
        ):
            raise ProbeError("diagnostic_thread_crt_live_state_invalid")

        process = kernel32.OpenProcess(PROCESS_ACCESS, False, pid)
        if not process:
            raise ctypes.WinError(ctypes.get_last_error())
        changed = declared_live_state != source_state
        try:
            live_payload = read_memory(process, state_address, 4)
            if len(live_payload) != 4:
                raise ProbeError("diagnostic_thread_crt_live_read_invalid")
            live_state = struct.unpack("<I", live_payload)[0]
            if live_state != declared_live_state:
                raise ProbeError("diagnostic_thread_crt_live_state_changed")
            try:
                if changed:
                    write_memory(process, state_address, source_payload)
                observed_payload = read_memory(process, state_address, 4)
                if observed_payload != source_payload:
                    raise ProbeError(
                        "diagnostic_thread_crt_writeback_mismatch"
                    )
            except Exception:
                if changed:
                    try:
                        write_memory(process, state_address, live_payload)
                        if read_memory(process, state_address, 4) != live_payload:
                            raise ProbeError(
                                "diagnostic_thread_crt_rollback_mismatch"
                            )
                    except Exception as rollback_error:
                        raise ProbeError(
                            "diagnostic_thread_crt_restore_and_rollback_failed"
                        ) from rollback_error
                raise
        finally:
            kernel32.CloseHandle(process)

        return {
            "schema": "zuma-rl.pc-diagnostic-thread-crt-state-restore",
            "version": 1,
            "status": "PASS",
            "classification": (
                "diagnostic_process_memory_mutation_not_pc_evidence"
            ),
            "process_id": pid,
            "source_artifact": str(source_path),
            "source_artifact_bytes": len(source_payload),
            "source_artifact_sha256": source_sha256,
            "source_framework_update": source_framework_update,
            "state_address": state_address,
            "live_before_state": declared_live_state,
            "restored_state": source_state,
            "changed": changed,
            "bytes_written": 4 if changed else 0,
            "writeback_verified": True,
            "transactional_rollback_on_failure": True,
            "process_memory_mutation": changed,
            "process_context_mutation": False,
            "persistent_file_modified": False,
        }

    return mutate


def _diagnostic_artifact_payload(
    *,
    artifact_root: Path,
    row: Mapping[str, Any],
    role: str,
    expected_size: int | None = None,
) -> bytes:
    """Read one hash-bound collector artifact without allowing path escape."""

    artifact = row.get("artifact")
    if (
        not isinstance(artifact, str)
        or not artifact
        or Path(artifact).name != artifact
        or "/" in artifact
        or "\\" in artifact
    ):
        raise ProbeError(f"diagnostic_compact_{role}_artifact_invalid")
    root = artifact_root.resolve()
    path = (root / artifact).resolve()
    if path.parent != root:
        raise ProbeError(f"diagnostic_compact_{role}_artifact_escape")
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ProbeError(
            f"diagnostic_compact_{role}_artifact_unreadable"
        ) from error
    declared_bytes = row.get("artifact_bytes")
    if (
        expected_size is not None
        and len(payload) != expected_size
    ) or (
        declared_bytes is not None
        and (
            isinstance(declared_bytes, bool)
            or not isinstance(declared_bytes, int)
            or declared_bytes != len(payload)
        )
    ):
        raise ProbeError(f"diagnostic_compact_{role}_artifact_size_mismatch")
    declared_sha256 = row.get("artifact_sha256")
    if (
        not _is_canonical_sha256(declared_sha256)
        or declared_sha256 != _sha256_bytes(payload)
    ):
        raise ProbeError(f"diagnostic_compact_{role}_artifact_hash_mismatch")
    return payload


def _diagnostic_ball_projection(
    payload: bytes,
    *,
    declared: Mapping[str, Any],
    kind: str,
    expected_vtable: int,
) -> dict[str, Any]:
    decoded = _decode_ball(payload, expected_vtable=expected_vtable)
    if dict(declared) != decoded:
        raise ProbeError("diagnostic_compact_ball_semantics_mismatch")
    return {
        "kind": kind,
        "vtable": expected_vtable,
        **{key: decoded[key] for key in DIAGNOSTIC_BALL_PROJECTION_KEYS},
    }


def _diagnostic_light_ball_projection(
    value: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if value is None:
        return None
    try:
        kind = value["kind"]
        vtable = value["vtable"]
        result = {
            "kind": kind,
            "vtable": vtable,
            **{
                key: value[key]
                for key in DIAGNOSTIC_BALL_PROJECTION_KEYS
            },
        }
    except (KeyError, TypeError) as error:
        raise ProbeError("diagnostic_compact_light_ball_invalid") from error
    if (
        (kind == "ball" and vtable != BALL_VTABLE)
        or (kind == "bullet" and vtable != BULLET_VTABLE)
        or kind not in {"ball", "bullet"}
    ):
        raise ProbeError("diagnostic_compact_light_ball_identity_invalid")
    return result


def _diagnostic_bullet_subclass_projection(
    payload: bytes,
    *,
    declared: Mapping[str, Any],
) -> dict[str, Any]:
    decoded = _decode_bullet_subclass(payload)
    if any(declared.get(key) != value for key, value in decoded.items()):
        raise ProbeError("diagnostic_compact_bullet_semantics_mismatch")
    try:
        return {
            key: declared[key]
            for key in DIAGNOSTIC_BULLET_SUBCLASS_PROJECTION_KEYS
        }
    except KeyError as error:
        raise ProbeError(
            "diagnostic_compact_bullet_projection_field_missing"
        ) from error


def _diagnostic_qrand_projection(
    *,
    board: Mapping[str, Any],
    artifact_root: Path,
) -> dict[str, Any]:
    qrand = board.get("qrand")
    if not isinstance(qrand, Mapping):
        raise ProbeError("diagnostic_compact_qrand_missing")
    raw = _diagnostic_artifact_payload(
        artifact_root=artifact_root,
        row=qrand,
        role="qrand",
        expected_size=QRAND_OBJECT_SIZE,
    )
    if (
        qrand.get("update_count") != _i32(raw, 0)
        or qrand.get("selected_index") != _i32(raw, 4)
    ):
        raise ProbeError("diagnostic_compact_qrand_scalar_mismatch")
    declared_vectors = qrand.get("vectors")
    if not isinstance(declared_vectors, list):
        raise ProbeError("diagnostic_compact_qrand_vectors_invalid")
    by_name = {
        row.get("name"): row
        for row in declared_vectors
        if isinstance(row, Mapping)
    }
    if set(by_name) != {row[1] for row in QRAND_VECTOR_LAYOUT}:
        raise ProbeError("diagnostic_compact_qrand_vector_set_mismatch")
    vectors: dict[str, list[int | float]] = {}
    shapes: dict[str, dict[str, Any]] = {}
    for vector_offset, name, value_type in QRAND_VECTOR_LAYOUT:
        row = by_name[name]
        begin = _u32(raw, vector_offset + 4)
        end = _u32(raw, vector_offset + 8)
        capacity = _u32(raw, vector_offset + 12)
        if (
            row.get("object_offset") != vector_offset
            or row.get("value_type") != value_type
            or row.get("begin") != begin
            or row.get("end") != end
            or row.get("capacity") != capacity
            or begin == 0
            or end < begin
            or capacity < end
            or (end - begin) % 4
            or (capacity - begin) % 4
        ):
            raise ProbeError(
                f"diagnostic_compact_qrand_{name}_header_mismatch"
            )
        item_count = (end - begin) // 4
        capacity_count = (capacity - begin) // 4
        if (
            item_count != BOARD_COLOR_COUNT_SLOTS
            or capacity_count != BOARD_COLOR_COUNT_SLOTS
            or row.get("item_count") != item_count
            or row.get("capacity_count") != capacity_count
        ):
            raise ProbeError(
                f"diagnostic_compact_qrand_{name}_shape_mismatch"
            )
        payload = _diagnostic_artifact_payload(
            artifact_root=artifact_root,
            row=row,
            role=f"qrand_{name}",
            expected_size=item_count * 4,
        )
        if value_type == "float32":
            values: list[int | float] = list(
                struct.unpack(f"<{item_count}f", payload)
            )
        else:
            values = list(struct.unpack(f"<{item_count}i", payload))
        if row.get("values") != values:
            raise ProbeError(
                f"diagnostic_compact_qrand_{name}_values_mismatch"
            )
        vectors[name] = values
        shapes[name] = {
            "item_count": item_count,
            "capacity_count": capacity_count,
            "value_type": value_type,
        }
    return {
        "update_count": _i32(raw, 0),
        "selected_index": _i32(raw, 4),
        "vectors": vectors,
        "vector_shapes": shapes,
    }


def _diagnostic_curve_list_projection(
    *,
    curve_index: int,
    row: Mapping[str, Any],
    artifact_root: Path,
) -> dict[str, Any]:
    container_offset = row.get("container_offset")
    if container_offset not in CURVE_INTRUSIVE_LIST_OFFSETS:
        raise ProbeError("diagnostic_compact_curve_list_offset_invalid")
    records = row.get("records")
    if not isinstance(records, list):
        raise ProbeError("diagnostic_compact_curve_list_records_invalid")
    if (
        row.get("declared_count") != len(records)
        or row.get("traversed_count") != len(records)
        or row.get("payload_count") != len(records)
    ):
        raise ProbeError("diagnostic_compact_curve_list_count_mismatch")
    blob = _diagnostic_artifact_payload(
        artifact_root=artifact_root,
        row=row,
        role=f"curve_{curve_index:02d}_list_{container_offset:03x}",
    )
    entities: list[dict[str, Any]] = []
    cursor = 0
    for expected_index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ProbeError("diagnostic_compact_curve_record_invalid")
        kind = record.get("payload_kind")
        expected_vtable = (
            BALL_VTABLE
            if kind == "ball"
            else BULLET_VTABLE if kind == "bullet" else None
        )
        expected_bytes = (
            BALL_OBJECT_SIZE
            if expected_vtable == BALL_VTABLE
            else BULLET_OBJECT_SIZE
            if expected_vtable == BULLET_VTABLE
            else None
        )
        if (
            expected_vtable is None
            or record.get("index") != expected_index
            or record.get("artifact_offset") != cursor
            or record.get("artifact_bytes") != expected_bytes
            or record.get("payload_head") != expected_vtable
        ):
            raise ProbeError("diagnostic_compact_curve_record_shape_mismatch")
        payload = blob[cursor : cursor + expected_bytes]
        if (
            len(payload) != expected_bytes
            or record.get("payload_sha256") != _sha256_bytes(payload)
        ):
            raise ProbeError("diagnostic_compact_curve_record_hash_mismatch")
        declared_ball = record.get("ball")
        if not isinstance(declared_ball, Mapping):
            raise ProbeError("diagnostic_compact_curve_ball_missing")
        entity: dict[str, Any] = {
            "index": expected_index,
            "payload_kind": kind,
            "payload_head": expected_vtable,
            "ball": _diagnostic_ball_projection(
                payload,
                declared=declared_ball,
                kind=kind,
                expected_vtable=expected_vtable,
            ),
        }
        if expected_vtable == BULLET_VTABLE:
            subclass = record.get("subclass_fields")
            if not isinstance(subclass, Mapping):
                raise ProbeError(
                    "diagnostic_compact_curve_bullet_subclass_missing"
                )
            entity["subclass_fields"] = (
                _diagnostic_bullet_subclass_projection(
                    payload,
                    declared=subclass,
                )
            )
        entities.append(entity)
        cursor += expected_bytes
    if cursor != len(blob):
        raise ProbeError("diagnostic_compact_curve_list_trailing_bytes")
    return {
        "curve_index": curve_index,
        "container_offset": container_offset,
        "declared_count": len(records),
        "traversed_count": len(records),
        "payload_count": len(records),
        "entities": entities,
    }


def _diagnostic_compact_board_projection(
    board: Mapping[str, Any],
    *,
    artifact_root: Path,
) -> dict[str, Any]:
    """Recompute the source-bound compact correction surface from artifacts."""

    board_raw = _diagnostic_artifact_payload(
        artifact_root=artifact_root,
        row=board,
        role="active_board",
        expected_size=BOARD_DUMP_SIZE,
    )
    if (
        _u32(board_raw, 0) != BOARD_VTABLE
        or _u32(board_raw, BOARD_EMBEDDED_VTABLE_OFFSET)
        != BOARD_EMBEDDED_VTABLE
    ):
        raise ProbeError("diagnostic_compact_board_identity_mismatch")
    mode_raw = board_raw[BOARD_MODE_FLAG_1064_OFFSET]
    if mode_raw not in {0, 1}:
        raise ProbeError("diagnostic_compact_board_mode_invalid")
    scalar_pairs = (
        ("score", _i32(board_raw, BOARD_SCORE_OFFSET)),
        ("displayed_score", _i32(board_raw, BOARD_DISPLAYED_SCORE_OFFSET)),
        ("score_target", _i32(board_raw, BOARD_SCORE_TARGET_OFFSET)),
        ("mode_flag_1064", bool(mode_raw)),
    )
    if any(board.get(key) != value for key, value in scalar_pairs):
        raise ProbeError("diagnostic_compact_board_scalar_mismatch")
    board_color_counts = list(
        struct.unpack_from(
            f"<{BOARD_COLOR_COUNT_SLOTS}i",
            board_raw,
            BOARD_COLOR_COUNTS_OFFSET,
        )
    )

    exhausted = board.get("curve_plan_exhausted")
    if not isinstance(exhausted, Mapping):
        raise ProbeError("diagnostic_compact_curve_plan_exhausted_missing")
    exhausted_raw = _diagnostic_artifact_payload(
        artifact_root=artifact_root,
        row=exhausted,
        role="curve_plan_exhausted",
        expected_size=1,
    )
    if (
        exhausted_raw[0] not in {0, 1}
        or exhausted.get("value") is not bool(exhausted_raw[0])
    ):
        raise ProbeError(
            "diagnostic_compact_curve_plan_exhausted_mismatch"
        )

    mtrand = board.get("global_mtrand")
    if not isinstance(mtrand, Mapping):
        raise ProbeError("diagnostic_compact_global_mtrand_missing")
    mtrand_raw = _diagnostic_artifact_payload(
        artifact_root=artifact_root,
        row=mtrand,
        role="global_mtrand",
        expected_size=MTRAND_STATE_BYTES,
    )
    mtrand_index = _u32(mtrand_raw, MTRAND_STATE_WORDS * 4)
    if mtrand.get("index") != mtrand_index:
        raise ProbeError("diagnostic_compact_global_mtrand_index_mismatch")

    crt = board.get("thread_crt_rand")
    if not isinstance(crt, Mapping):
        raise ProbeError("diagnostic_compact_thread_crt_missing")
    crt_raw = _diagnostic_artifact_payload(
        artifact_root=artifact_root,
        row=crt,
        role="thread_crt",
        expected_size=4,
    )
    crt_state = _u32(crt_raw, 0)
    if crt.get("state") != crt_state:
        raise ProbeError("diagnostic_compact_thread_crt_mismatch")

    primary_child = board.get("primary_child")
    if not isinstance(primary_child, Mapping):
        raise ProbeError("diagnostic_compact_shooter_missing")
    bullet_rows = primary_child.get("bullets")
    if not isinstance(bullet_rows, list) or len(bullet_rows) != 2:
        raise ProbeError("diagnostic_compact_shooter_bullets_invalid")
    bullets: list[dict[str, Any]] = []
    for index, expected_offset in enumerate(SHOOTER_BULLET_POINTER_OFFSETS):
        row = bullet_rows[index]
        if (
            not isinstance(row, Mapping)
            or row.get("shooter_pointer_offset") != expected_offset
            or not isinstance(row.get("address"), int)
            or row.get("address") <= 0
        ):
            raise ProbeError("diagnostic_compact_shooter_bullet_shape_invalid")
        payload = _diagnostic_artifact_payload(
            artifact_root=artifact_root,
            row=row,
            role=f"shooter_bullet_{expected_offset:03x}",
            expected_size=BULLET_OBJECT_SIZE,
        )
        declared_ball = row.get("ball")
        subclass = row.get("subclass_fields")
        if not isinstance(declared_ball, Mapping) or not isinstance(
            subclass, Mapping
        ):
            raise ProbeError("diagnostic_compact_shooter_bullet_missing")
        bullets.append(
            {
                "slot": expected_offset,
                "ball": _diagnostic_ball_projection(
                    payload,
                    declared=declared_ball,
                    kind="bullet",
                    expected_vtable=BULLET_VTABLE,
                ),
                "subclass_fields": (
                    _diagnostic_bullet_subclass_projection(
                        payload,
                        declared=subclass,
                    )
                ),
            }
        )

    fired = board.get("fired_bullets")
    if not isinstance(fired, Mapping):
        raise ProbeError("diagnostic_compact_fired_bullets_missing")
    fired_raw = _diagnostic_artifact_payload(
        artifact_root=artifact_root,
        row=fired,
        role="fired_bullets",
        expected_size=0,
    )
    if (
        fired_raw
        or fired.get("declared_count") != 0
        or fired.get("traversed_count") != 0
        or fired.get("records") != []
        or _u32(board_raw, BOARD_FIRED_BULLET_LIST_OFFSET + 8) != 0
    ):
        raise ProbeError("diagnostic_compact_fired_bullets_unsupported")

    manager = board.get("curve_manager")
    if not isinstance(manager, Mapping):
        raise ProbeError("diagnostic_compact_curve_manager_missing")
    manager_raw = _diagnostic_artifact_payload(
        artifact_root=artifact_root,
        row=manager,
        role="curve_manager",
        expected_size=CURVE_MANAGER_DUMP_SIZE,
    )
    curve_rows = manager.get("curves")
    if (
        _u32(manager_raw, 0) != CURVE_MANAGER_VTABLE
        or _i32(manager_raw, CURVE_MANAGER_CURVE_COUNT_OFFSET) != 1
        or manager.get("curve_count") != 1
        or not isinstance(curve_rows, list)
        or len(curve_rows) != 1
    ):
        raise ProbeError("diagnostic_compact_curve_manager_shape_invalid")
    declared_post_zuma = manager.get("post_zuma_diagnostics")
    post_zuma = {
        "timer_remaining": _i32(
            manager_raw,
            CURVE_MANAGER_POST_ZUMA_TIMER_OFFSET,
        ),
        "ramp_404": struct.unpack_from(
            "<f", manager_raw, CURVE_MANAGER_POST_ZUMA_RAMP_404_OFFSET
        )[0],
        "ramp_408": struct.unpack_from(
            "<f", manager_raw, CURVE_MANAGER_POST_ZUMA_RAMP_408_OFFSET
        )[0],
    }
    if not isinstance(declared_post_zuma, Mapping) or any(
        declared_post_zuma.get(key) != value
        for key, value in post_zuma.items()
    ):
        raise ProbeError("diagnostic_compact_post_zuma_mismatch")

    curves: list[dict[str, Any]] = []
    for curve_index, curve in enumerate(curve_rows):
        if not isinstance(curve, Mapping) or curve.get("index") != curve_index:
            raise ProbeError("diagnostic_compact_curve_shape_invalid")
        curve_raw = _diagnostic_artifact_payload(
            artifact_root=artifact_root,
            row=curve,
            role=f"curve_{curve_index:02d}",
            expected_size=CURVE_DUMP_SIZE,
        )
        planned = curve.get("planned_balls")
        if not isinstance(planned, Mapping):
            raise ProbeError("diagnostic_compact_curve_plan_missing")
        plan_raw = _diagnostic_artifact_payload(
            artifact_root=artifact_root,
            row=planned,
            role=f"curve_{curve_index:02d}_plan",
            expected_size=0,
        )
        plan_begin = _u32(curve_raw, CURVE_PLANNED_VECTOR_BEGIN_OFFSET)
        plan_end = _u32(curve_raw, CURVE_PLANNED_VECTOR_END_OFFSET)
        plan_capacity = _u32(
            curve_raw,
            CURVE_PLANNED_VECTOR_CAPACITY_OFFSET,
        )
        if (
            plan_raw
            or planned.get("count") != 0
            or planned.get("begin_address") != plan_begin
            or planned.get("end_address") != plan_end
            or planned.get("capacity_address") != plan_capacity
            or plan_begin != plan_end
            or plan_capacity < plan_end
            or (plan_capacity - plan_begin) % CURVE_PLANNED_ITEM_SIZE
            or planned.get("capacity_count")
            != (plan_capacity - plan_begin) // CURVE_PLANNED_ITEM_SIZE
            or planned.get("add_plan_enabled")
            is not bool(curve_raw[CURVE_ADD_PLAN_ENABLED_OFFSET])
        ):
            raise ProbeError("diagnostic_compact_curve_plan_mismatch")
        list_rows = curve.get("intrusive_lists")
        if not isinstance(list_rows, list) or [
            row.get("container_offset")
            for row in list_rows
            if isinstance(row, Mapping)
        ] != list(CURVE_INTRUSIVE_LIST_OFFSETS):
            raise ProbeError("diagnostic_compact_curve_list_set_mismatch")
        lists: list[dict[str, Any]] = []
        for list_row in list_rows:
            container_offset = int(list_row["container_offset"])
            if (
                _u32(curve_raw, container_offset + 8)
                != list_row.get("declared_count")
            ):
                raise ProbeError("diagnostic_compact_curve_list_raw_mismatch")
            lists.append(
                _diagnostic_curve_list_projection(
                    curve_index=curve_index,
                    row=list_row,
                    artifact_root=artifact_root,
                )
            )
        curves.append(
            {
                "index": curve_index,
                "planned": {
                    "count": 0,
                    "capacity_count": planned["capacity_count"],
                    "item_size": planned["item_size"],
                    "add_plan_enabled": planned["add_plan_enabled"],
                    "payload_sha256": _sha256_bytes(plan_raw),
                },
                "lists": lists,
            }
        )

    return {
        "score": _i32(board_raw, BOARD_SCORE_OFFSET),
        "displayed_score": _i32(board_raw, BOARD_DISPLAYED_SCORE_OFFSET),
        "score_target": _i32(board_raw, BOARD_SCORE_TARGET_OFFSET),
        "mode": bool(mode_raw),
        "curve_plan_exhausted": bool(exhausted_raw[0]),
        "board_color_counts": board_color_counts,
        "mtrand_sha": _sha256_bytes(mtrand_raw),
        "mtrand_index": mtrand_index,
        "crt": crt_state,
        "qrand": _diagnostic_qrand_projection(
            board=board,
            artifact_root=artifact_root,
        ),
        "bullets": bullets,
        "fired_count": 0,
        "post_zuma": post_zuma,
        "curves": curves,
    }


def _diagnostic_runtime_projection_from_full(
    projection: Mapping[str, Any],
) -> dict[str, Any]:
    qrand = projection["qrand"]
    return {
        "score": projection["score"],
        "displayed_score": projection["displayed_score"],
        "score_target": projection["score_target"],
        "board_color_counts": projection["board_color_counts"],
        "mtrand_sha": projection["mtrand_sha"],
        "mtrand_index": projection["mtrand_index"],
        "crt": projection["crt"],
        "qrand": {
            "update_count": qrand["update_count"],
            "selected_index": qrand["selected_index"],
            "vectors": qrand["vectors"],
        },
        "bullets": [
            {"slot": row["slot"], "ball": row["ball"]}
            for row in projection["bullets"]
        ],
        "fired_count": projection["fired_count"],
        "curves": [
            {
                "index": curve["index"],
                "lists": [
                    {
                        "curve_index": row["curve_index"],
                        "container_offset": row["container_offset"],
                        "declared_count": row["declared_count"],
                        "traversed_count": row["traversed_count"],
                        "entities": [
                            {
                                "index": entity["index"],
                                "payload_kind": entity["payload_kind"],
                                "payload_head": entity["payload_head"],
                                "ball": entity["ball"],
                            }
                            for entity in row["entities"]
                        ],
                    }
                    for row in curve["lists"]
                ],
            }
            for curve in projection["curves"]
        ],
    }


def _diagnostic_live_runtime_projection(
    tick: Mapping[str, Any],
    *,
    mtrand_payload: bytes,
) -> dict[str, Any]:
    curve_rows: dict[int, list[dict[str, Any]]] = {}
    for row in tick["curve_lists"]:
        curve_index = int(row["curve_index"])
        entities = row.get("entities")
        if not isinstance(entities, list) or any(
            entity is None for entity in entities
        ):
            raise ProbeError("diagnostic_compact_live_curve_entity_invalid")
        curve_rows.setdefault(curve_index, []).append(
            {
                "curve_index": curve_index,
                "container_offset": int(row["container_offset"]),
                "declared_count": int(row["declared_count"]),
                "traversed_count": int(row["traversed_count"]),
                "entities": [
                    {
                        "index": index,
                        "payload_kind": entity["kind"],
                        "payload_head": entity["vtable"],
                        "ball": _diagnostic_light_ball_projection(entity),
                    }
                    for index, entity in enumerate(entities)
                ],
            }
        )
    curves = [
        {"index": index, "lists": curve_rows[index]}
        for index in sorted(curve_rows)
    ]
    qrand = tick["qrand"]
    return {
        "score": tick["score"],
        "displayed_score": tick["displayed_score"],
        "score_target": tick["score_target"],
        "board_color_counts": tick["board_color_counts"],
        "mtrand_sha": _sha256_bytes(mtrand_payload),
        "mtrand_index": tick["global_mtrand_index"],
        "crt": tick["thread_crt_rand_state"],
        "qrand": {
            "update_count": qrand["update_count"],
            "selected_index": qrand["selected_index"],
            "vectors": qrand["vectors"],
        },
        "bullets": [
            {
                "slot": offset,
                "ball": _diagnostic_light_ball_projection(value),
            }
            for offset, value in zip(
                SHOOTER_BULLET_POINTER_OFFSETS,
                (tick["current_ball"], tick["next_ball"]),
                strict=True,
            )
        ],
        "fired_count": tick["fired_bullet_count"],
        "curves": curves,
    }


def _diagnostic_difference_paths(
    left: Any,
    right: Any,
    *,
    path: str = "",
) -> list[str]:
    if type(left) is not type(right):
        return [path or "/"]
    if isinstance(left, Mapping):
        paths: list[str] = []
        for key in sorted(set(left) | set(right)):
            child = f"{path}/{key}"
            if key not in left or key not in right:
                paths.append(child)
            else:
                paths.extend(
                    _diagnostic_difference_paths(
                        left[key],
                        right[key],
                        path=child,
                    )
                )
        return paths
    if isinstance(left, list):
        if len(left) != len(right):
            return [f"{path}/length"]
        paths = []
        for index, (left_value, right_value) in enumerate(
            zip(left, right, strict=True)
        ):
            paths.extend(
                _diagnostic_difference_paths(
                    left_value,
                    right_value,
                    path=f"{path}/{index}",
                )
            )
        return paths
    return [] if left == right else [path or "/"]


def _load_diagnostic_compact_state_source(
    *,
    source_probe_path: Path,
    expected_probe_sha256: str,
    source_framework_update: int,
) -> dict[str, Any]:
    source_probe_path = source_probe_path.resolve()
    try:
        probe_payload = source_probe_path.read_bytes()
        probe = json.loads(probe_payload.decode("ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProbeError(
            "diagnostic_compact_source_probe_unreadable"
        ) from error
    if (
        not isinstance(probe, Mapping)
        or _sha256_bytes(probe_payload) != expected_probe_sha256
        or probe.get("schema") != "zuma-rl.pc-memory-int32-probe"
        or probe.get("framework_update") != source_framework_update
    ):
        raise ProbeError("diagnostic_compact_source_probe_identity_mismatch")
    board = probe.get("active_board")
    if not isinstance(board, Mapping):
        raise ProbeError("diagnostic_compact_source_board_missing")
    projection = _diagnostic_compact_board_projection(
        board,
        artifact_root=source_probe_path.parent,
    )
    return {
        "probe_payload": probe_payload,
        "probe_sha256": expected_probe_sha256,
        "probe_path": source_probe_path,
        "framework_update": source_framework_update,
        "board": board,
        "artifact_root": source_probe_path.parent,
        "projection": projection,
        "runtime_projection": _diagnostic_runtime_projection_from_full(
            projection
        ),
    }


def _diagnostic_compact_write_operations(
    *,
    source: Mapping[str, Any],
    target_board: Mapping[str, Any],
    target_artifact_root: Path,
) -> list[dict[str, Any]]:
    source_board = source["board"]
    source_root = source["artifact_root"]

    source_board_raw = _diagnostic_artifact_payload(
        artifact_root=source_root,
        row=source_board,
        role="source_active_board_write",
        expected_size=BOARD_DUMP_SIZE,
    )
    target_board_raw = _diagnostic_artifact_payload(
        artifact_root=target_artifact_root,
        row=target_board,
        role="target_active_board_write",
        expected_size=BOARD_DUMP_SIZE,
    )

    source_crt = source_board["thread_crt_rand"]
    target_crt = target_board["thread_crt_rand"]
    source_crt_raw = _diagnostic_artifact_payload(
        artifact_root=source_root,
        row=source_crt,
        role="source_thread_crt_write",
        expected_size=4,
    )
    target_crt_raw = _diagnostic_artifact_payload(
        artifact_root=target_artifact_root,
        row=target_crt,
        role="target_thread_crt_write",
        expected_size=4,
    )

    source_qrand = source_board["qrand"]
    target_qrand = target_board["qrand"]
    source_qrand_raw = _diagnostic_artifact_payload(
        artifact_root=source_root,
        row=source_qrand,
        role="source_qrand_write",
        expected_size=QRAND_OBJECT_SIZE,
    )
    target_qrand_raw = _diagnostic_artifact_payload(
        artifact_root=target_artifact_root,
        row=target_qrand,
        role="target_qrand_write",
        expected_size=QRAND_OBJECT_SIZE,
    )
    source_vectors = {row["name"]: row for row in source_qrand["vectors"]}
    target_vectors = {row["name"]: row for row in target_qrand["vectors"]}

    source_bullets = source_board["primary_child"]["bullets"]
    target_bullets = target_board["primary_child"]["bullets"]
    source_current = source_bullets[0]
    target_current = target_bullets[0]
    source_next = source_bullets[1]
    target_next = target_bullets[1]
    source_current_raw = _diagnostic_artifact_payload(
        artifact_root=source_root,
        row=source_current,
        role="source_current_bullet_write",
        expected_size=BULLET_OBJECT_SIZE,
    )
    target_current_raw = _diagnostic_artifact_payload(
        artifact_root=target_artifact_root,
        row=target_current,
        role="target_current_bullet_write",
        expected_size=BULLET_OBJECT_SIZE,
    )
    source_next_raw = _diagnostic_artifact_payload(
        artifact_root=source_root,
        row=source_next,
        role="source_next_bullet_write",
        expected_size=BULLET_OBJECT_SIZE,
    )
    target_next_raw = _diagnostic_artifact_payload(
        artifact_root=target_artifact_root,
        row=target_next,
        role="target_next_bullet_write",
        expected_size=BULLET_OBJECT_SIZE,
    )

    source_list = source_board["curve_manager"]["curves"][0][
        "intrusive_lists"
    ][0]
    target_list = target_board["curve_manager"]["curves"][0][
        "intrusive_lists"
    ][0]
    source_list_raw = _diagnostic_artifact_payload(
        artifact_root=source_root,
        row=source_list,
        role="source_insertion_list_write",
    )
    target_list_raw = _diagnostic_artifact_payload(
        artifact_root=target_artifact_root,
        row=target_list,
        role="target_insertion_list_write",
    )
    source_record = source_list["records"][0]
    target_record = target_list["records"][0]
    source_record_offset = int(source_record["artifact_offset"])
    target_record_offset = int(target_record["artifact_offset"])

    def operation(
        role: str,
        address: Any,
        before: bytes,
        after: bytes,
    ) -> dict[str, Any]:
        if (
            isinstance(address, bool)
            or not isinstance(address, int)
            or address <= 0
            or not before
            or len(before) != len(after)
        ):
            raise ProbeError(
                f"diagnostic_compact_{role}_write_plan_invalid"
            )
        return {
            "role": role,
            "address": address,
            "before": before,
            "after": after,
            "changed": before != after,
        }

    operations = [
        operation(
            "board_color_counts",
            int(target_board["board_address"]) + BOARD_COLOR_COUNTS_OFFSET,
            target_board_raw[
                BOARD_COLOR_COUNTS_OFFSET : BOARD_COLOR_COUNTS_OFFSET + 24
            ],
            source_board_raw[
                BOARD_COLOR_COUNTS_OFFSET : BOARD_COLOR_COUNTS_OFFSET + 24
            ],
        ),
        operation(
            "thread_crt_rand_state",
            target_crt["rand_state_address"],
            target_crt_raw,
            source_crt_raw,
        ),
        operation(
            "qrand_selected_index",
            int(target_qrand["address"]) + 4,
            target_qrand_raw[4:8],
            source_qrand_raw[4:8],
        ),
    ]
    for name in ("sways", "last_hit", "previous_hit"):
        source_vector = source_vectors[name]
        target_vector = target_vectors[name]
        operations.append(
            operation(
                f"qrand_{name}",
                target_vector["begin"],
                _diagnostic_artifact_payload(
                    artifact_root=target_artifact_root,
                    row=target_vector,
                    role=f"target_qrand_{name}_write",
                    expected_size=24,
                ),
                _diagnostic_artifact_payload(
                    artifact_root=source_root,
                    row=source_vector,
                    role=f"source_qrand_{name}_write",
                    expected_size=24,
                ),
            )
        )
    operations.extend(
        [
            operation(
                "shooter_current_color",
                int(target_current["address"]) + BALL_COLOR_ID_OFFSET,
                target_current_raw[
                    BALL_COLOR_ID_OFFSET : BALL_COLOR_ID_OFFSET + 4
                ],
                source_current_raw[
                    BALL_COLOR_ID_OFFSET : BALL_COLOR_ID_OFFSET + 4
                ],
            ),
            operation(
                "shooter_next_color",
                int(target_next["address"]) + BALL_COLOR_ID_OFFSET,
                target_next_raw[
                    BALL_COLOR_ID_OFFSET : BALL_COLOR_ID_OFFSET + 4
                ],
                source_next_raw[
                    BALL_COLOR_ID_OFFSET : BALL_COLOR_ID_OFFSET + 4
                ],
            ),
            operation(
                "insertion_staging_color",
                int(target_record["payload_address"]) + BALL_COLOR_ID_OFFSET,
                target_list_raw[
                    target_record_offset
                    + BALL_COLOR_ID_OFFSET : target_record_offset
                    + BALL_COLOR_ID_OFFSET
                    + 4
                ],
                source_list_raw[
                    source_record_offset
                    + BALL_COLOR_ID_OFFSET : source_record_offset
                    + BALL_COLOR_ID_OFFSET
                    + 4
                ],
            ),
        ]
    )
    if len(operations) != 9 or sum(
        len(row["after"]) for row in operations
    ) != 116:
        raise ProbeError("diagnostic_compact_write_surface_mismatch")
    spans = sorted(
        (row["address"], row["address"] + len(row["after"]))
        for row in operations
    )
    if any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
        raise ProbeError("diagnostic_compact_write_surface_overlaps")
    return operations


def _diagnostic_compact_state_mutator(
    *,
    source_probe_path: Path,
    expected_probe_sha256: str,
    source_framework_update: int,
) -> Callable[[int, Mapping[str, Any], Path], Mapping[str, Any]]:
    """Build the fixed C111 compact-state correction transaction."""

    source = _load_diagnostic_compact_state_source(
        source_probe_path=source_probe_path,
        expected_probe_sha256=expected_probe_sha256,
        source_framework_update=source_framework_update,
    )

    def mutate(
        pid: int,
        pre_mutation_board: Mapping[str, Any],
        attempt_root: Path,
    ) -> Mapping[str, Any]:
        current_source = _load_diagnostic_compact_state_source(
            source_probe_path=source_probe_path,
            expected_probe_sha256=expected_probe_sha256,
            source_framework_update=source_framework_update,
        )
        if (
            current_source["probe_payload"] != source["probe_payload"]
            or current_source["projection"] != source["projection"]
        ):
            raise ProbeError("diagnostic_compact_source_changed")

        pre_root = attempt_root / "pre-mutation"
        target_projection = _diagnostic_compact_board_projection(
            pre_mutation_board,
            artifact_root=pre_root,
        )
        difference_paths = _diagnostic_difference_paths(
            source["projection"],
            target_projection,
        )
        if not set(difference_paths).issubset(
            DIAGNOSTIC_COMPACT_STATE_ALLOWED_PRE_DIFFERENCES
        ):
            raise ProbeError(
                "diagnostic_compact_pre_difference_contract_mismatch:"
                + json.dumps(difference_paths, separators=(",", ":"))
            )
        target_runtime_projection = (
            _diagnostic_runtime_projection_from_full(target_projection)
        )
        operations = _diagnostic_compact_write_operations(
            source=source,
            target_board=pre_mutation_board,
            target_artifact_root=pre_root,
        )

        process = kernel32.OpenProcess(PROCESS_ACCESS, False, pid)
        if not process:
            raise ctypes.WinError(ctypes.get_last_error())
        applied: list[dict[str, Any]] = []
        try:
            live_before_tick, live_before_mtrand = _read_rng_trajectory_tick(
                process,
                update=source_framework_update,
                thread_crt_state=pre_mutation_board["thread_crt_rand"],
            )
            live_before = _diagnostic_live_runtime_projection(
                live_before_tick,
                mtrand_payload=live_before_mtrand,
            )
            if live_before != target_runtime_projection:
                raise ProbeError(
                    "diagnostic_compact_live_prestate_changed"
                )
            for row in operations:
                observed = read_memory(
                    process,
                    row["address"],
                    len(row["before"]),
                )
                if observed != row["before"]:
                    raise ProbeError(
                        f"diagnostic_compact_{row['role']}_live_before_mismatch"
                    )
            try:
                for row in operations:
                    write_memory(process, row["address"], row["after"])
                    applied.append(row)
                for row in operations:
                    if read_memory(
                        process,
                        row["address"],
                        len(row["after"]),
                    ) != row["after"]:
                        raise ProbeError(
                            f"diagnostic_compact_{row['role']}_writeback_mismatch"
                        )
                live_after_tick, live_after_mtrand = (
                    _read_rng_trajectory_tick(
                        process,
                        update=source_framework_update,
                        thread_crt_state=pre_mutation_board[
                            "thread_crt_rand"
                        ],
                    )
                )
                live_after = _diagnostic_live_runtime_projection(
                    live_after_tick,
                    mtrand_payload=live_after_mtrand,
                )
                if live_after != source["runtime_projection"]:
                    paths = _diagnostic_difference_paths(
                        source["runtime_projection"],
                        live_after,
                    )
                    raise ProbeError(
                        "diagnostic_compact_poststate_mismatch:"
                        + json.dumps(paths, separators=(",", ":"))
                    )
            except BaseException:
                try:
                    for row in reversed(applied):
                        write_memory(
                            process,
                            row["address"],
                            row["before"],
                        )
                    for row in applied:
                        if read_memory(
                            process,
                            row["address"],
                            len(row["before"]),
                        ) != row["before"]:
                            raise ProbeError(
                                "diagnostic_compact_rollback_mismatch"
                            )
                except BaseException as rollback_error:
                    raise ProbeError(
                        "diagnostic_compact_restore_and_rollback_failed"
                    ) from rollback_error
                raise
        finally:
            kernel32.CloseHandle(process)

        projection_sha256 = _sha256_bytes(
            canonical_report_bytes(source["runtime_projection"])
        )
        return {
            "schema": (
                "zuma-rl.pc-diagnostic-source-bound-compact-state-restore"
            ),
            "version": 1,
            "status": "PASS",
            "classification": (
                "diagnostic_process_memory_mutation_not_pc_evidence"
            ),
            "process_id": pid,
            "source_probe": str(source_probe_path.resolve()),
            "source_probe_sha256": expected_probe_sha256,
            "source_framework_update": source_framework_update,
            "pre_difference_paths": difference_paths,
            "pre_difference_path_count": len(difference_paths),
            "allowed_pre_difference_paths": sorted(
                DIAGNOSTIC_COMPACT_STATE_ALLOWED_PRE_DIFFERENCES
            ),
            "write_operation_count": len(operations),
            "bytes_written": sum(len(row["after"]) for row in operations),
            "changed_write_operation_count": sum(
                bool(row["changed"]) for row in operations
            ),
            "changed_byte_count": sum(
                len(row["after"])
                for row in operations
                if row["changed"]
            ),
            "writes": [
                {
                    "role": row["role"],
                    "address": row["address"],
                    "address_hex": f"0x{row['address']:08x}",
                    "bytes": len(row["after"]),
                    "before_hex": row["before"].hex(),
                    "after_hex": row["after"].hex(),
                    "changed": row["changed"],
                }
                for row in operations
            ],
            "writeback_verified": True,
            "post_compact_state_exact_source": True,
            "post_compact_projection_sha256": projection_sha256,
            "transactional_rollback_on_failure": True,
            "write_ranges_non_overlapping": True,
            "process_memory_mutation": True,
            "process_context_mutation": False,
            "persistent_file_modified": False,
        }

    return mutate


def _is_canonical_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and re.fullmatch(r"sha256:[0-9a-f]{64}", value) is not None
    )


def _write_canonical_probe(path: Path, probe: Mapping[str, Any]) -> None:
    """Write the probe in the byte-exact format required by live audits.

    ``Path.write_text`` performs platform newline translation on Windows,
    turning the required final LF into CRLF.  Use bytes so evidence collected
    on the native host has the same canonical representation as WSL/Linux.
    """

    path.write_bytes(canonical_report_bytes(probe))


def _write_canonical_attempts(
    path: Path,
    attempts: Iterable[Mapping[str, Any]],
) -> None:
    """Write the evolving attempt receipt without Windows CRLF translation."""

    payload = (
        json.dumps(
            list(attempts),
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")
    path.write_bytes(payload)


def _write_external_input_guard_receipt(
    path: Path,
    receipt: Mapping[str, Any],
) -> None:
    path.write_bytes(canonical_external_input_bytes(receipt))


def _int32_offsets(data: bytes, value: int, base: int) -> list[int]:
    needle = struct.pack("<i", value)
    offsets: list[int] = []
    start = 0
    while True:
        offset = data.find(needle, start)
        if offset < 0:
            return offsets
        address = base + offset
        if address % 4 == 0:
            offsets.append(offset)
        start = offset + 1


def _valid_vtable(handle: int, address: int) -> bool:
    if not RDATA_START <= address <= RDATA_END - 12 or address % 4:
        return False
    try:
        entries = struct.unpack("<III", read_process_bytes(handle, address, 12))
    except RuntimeError:
        return False
    return all(TEXT_START <= entry < TEXT_END for entry in entries)


def _candidate_objects(
    handle: int,
    *,
    region_base: int,
    region_data: bytes,
    hit_offset: int,
) -> list[dict[str, int | str]]:
    candidates: list[dict[str, int | str]] = []
    start = max(0, hit_offset - 0x1000)
    start += (-start) % 4
    for offset in range(start, hit_offset + 1, 4):
        vtable = struct.unpack_from("<I", region_data, offset)[0]
        if not _valid_vtable(handle, vtable):
            continue
        object_address = region_base + offset
        candidates.append(
            {
                "object_address": object_address,
                "object_address_hex": f"0x{object_address:08x}",
                "value_offset": hit_offset - offset,
                "vtable": vtable,
                "vtable_hex": f"0x{vtable:08x}",
            }
        )
    candidates.sort(key=lambda row: int(row["value_offset"]))
    return candidates[:16]


def _context_rows(
    region_data: bytes,
    hit_offset: int,
) -> list[dict[str, int | float | str | None]]:
    rows: list[dict[str, int | float | str | None]] = []
    start = max(0, hit_offset - 128)
    end = min(len(region_data), hit_offset + 132)
    start += (-start) % 4
    for offset in range(start, end - 3, 4):
        raw = region_data[offset : offset + 4]
        unsigned = int.from_bytes(raw, "little", signed=False)
        signed = int.from_bytes(raw, "little", signed=True)
        floating = struct.unpack("<f", raw)[0]
        rows.append(
            {
                "relative_offset": offset - hit_offset,
                "u32": unsigned,
                "i32": signed,
                "hex": f"0x{unsigned:08x}",
                "float32": floating if math.isfinite(floating) else None,
            }
        )
    return rows


def _pointer_references(
    handle: int,
    targets: Iterable[int],
) -> dict[int, dict[str, Any]]:
    target_set = tuple(sorted(set(targets)))
    results: dict[int, dict[str, Any]] = {
        target: {"count": 0, "addresses": []} for target in target_set
    }
    needles = {target: struct.pack("<I", target) for target in target_set}
    for base, size in _readable_regions(handle):
        data = _read_region(handle, base, size)
        if data is None:
            continue
        for target, needle in needles.items():
            start = 0
            while True:
                offset = data.find(needle, start)
                if offset < 0:
                    break
                address = base + offset
                if address % 4 == 0:
                    row = results[target]
                    row["count"] += 1
                    if len(row["addresses"]) < 128:
                        row["addresses"].append(
                            {
                                "address": address,
                                "address_hex": f"0x{address:08x}",
                                "image": IMAGE_START <= address < IMAGE_END,
                            }
                        )
                start = offset + 1
    return results


def _bounded_reference_targets(
    hits: Iterable[Mapping[str, Any]],
) -> tuple[tuple[int, ...], int]:
    """Select a deterministic diagnostic subset without rejecting a probe."""

    targets: set[int] = set()
    for hit in hits:
        if hit["image"]:
            continue
        candidates = (
            int(hit["address"]),
            *(
                int(candidate["object_address"])
                for candidate in hit["candidate_objects"][:4]
            ),
        )
        targets.update(
            address
            for address in candidates
            if 0 <= address <= 0xFFFFFFFF
        )
    ordered = tuple(sorted(targets))
    return ordered[:MAX_POINTER_REFERENCE_TARGETS], len(ordered)


def _u32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def _i32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<i", data, offset)[0]


def _containing_region(
    regions: Iterable[tuple[int, int]],
    address: int,
    *,
    allow_end: bool = False,
) -> tuple[int, int] | None:
    for base, size in regions:
        end = base + size
        if base <= address < end or (allow_end and address == end):
            return base, size
    return None


def _pointer_rows(
    handle: int,
    data: bytes,
    regions: tuple[tuple[int, int], ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for offset in range(0, len(data) - 3, 4):
        value = _u32(data, offset)
        if value < 0x10000 or _containing_region(regions, value) is None:
            continue
        row: dict[str, Any] = {
            "offset": offset,
            "offset_hex": f"0x{offset:04x}",
            "address": value,
            "address_hex": f"0x{value:08x}",
            "image": IMAGE_START <= value < IMAGE_END,
        }
        try:
            target_head = _u32(read_process_bytes(handle, value, 4), 0)
        except RuntimeError:
            target_head = None
        if target_head is not None:
            row["target_head"] = target_head
            row["target_head_hex"] = f"0x{target_head:08x}"
            row["target_has_vtable"] = _valid_vtable(handle, target_head)
        rows.append(row)
    return rows


def _vector_rows(
    data: bytes,
    regions: tuple[tuple[int, int], ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for offset in range(0, len(data) - 11, 4):
        begin, end, capacity = struct.unpack_from("<III", data, offset)
        if (
            begin < 0x10000
            or begin > end
            or end > capacity
            or capacity - begin > 0x100000
            or (end - begin) % 4
            or (capacity - begin) % 4
            or _containing_region(regions, begin) is None
            or _containing_region(regions, end, allow_end=True) is None
            or _containing_region(regions, capacity, allow_end=True) is None
        ):
            continue
        rows.append(
            {
                "offset": offset,
                "offset_hex": f"0x{offset:04x}",
                "begin": begin,
                "begin_hex": f"0x{begin:08x}",
                "end": end,
                "end_hex": f"0x{end:08x}",
                "capacity": capacity,
                "capacity_hex": f"0x{capacity:08x}",
                "size_bytes": end - begin,
                "capacity_bytes": capacity - begin,
                "pointer_count": (end - begin) // 4,
            }
        )
    return rows


def _object_summary(
    handle: int,
    data: bytes,
    regions: tuple[tuple[int, int], ...],
) -> dict[str, Any]:
    head = _u32(data, 0) if len(data) >= 4 else None
    return {
        "bytes": len(data),
        "head": head,
        "head_hex": f"0x{head:08x}" if head is not None else None,
        "head_is_vtable": (
            _valid_vtable(handle, head) if head is not None else False
        ),
        "pointer_fields": _pointer_rows(handle, data, regions),
        "vector_candidates": _vector_rows(data, regions),
    }


def _decode_ball(
    payload: bytes,
    *,
    expected_vtable: int = BALL_VTABLE,
) -> dict[str, Any]:
    if (
        len(payload) < BALL_OBJECT_SIZE
        or _u32(payload, 0) != expected_vtable
    ):
        raise ProbeError("ball_payload_invalid")
    color_id = _i32(payload, 0x14)
    if not 0 <= color_id <= 5:
        raise ProbeError(f"ball_color_invalid:{color_id}")
    curve_distance = struct.unpack_from("<f", payload, 0x1C)[0]
    orientation = struct.unpack_from("<f", payload, 0x20)[0]
    previous_orientation = struct.unpack_from("<f", payload, 0x24)[0]
    angular_step = struct.unpack_from("<f", payload, 0x28)[0]
    position_x, position_y = struct.unpack_from("<ff", payload, 0x2C)
    scale = struct.unpack_from("<f", payload, 0x34)[0]
    radius = struct.unpack_from("<f", payload, 0x38)[0]
    floats = (
        curve_distance,
        orientation,
        previous_orientation,
        angular_step,
        position_x,
        position_y,
        scale,
        radius,
    )
    if not all(math.isfinite(value) for value in floats):
        raise ProbeError("ball_float_nonfinite")
    return {
        "ball_id": _u32(payload, 0x10),
        "color_id": color_id,
        "curve_distance": curve_distance,
        "orientation_radians": orientation,
        "previous_orientation_radians": previous_orientation,
        "angular_step_radians": angular_step,
        "position_x": position_x,
        "position_y": position_y,
        "scale": scale,
        "radius": radius,
        "powerup_previous_type": _i32(payload, 0xC8),
        "powerup_primary_type": _i32(payload, 0x11C),
        "powerup_secondary_type": _i32(payload, 0x120),
        "flags_b4_c2_hex": payload[0xB4:0xC3].hex(),
    }


def _decode_bullet_subclass(bullet: bytes) -> dict[str, Any]:
    if len(bullet) != BULLET_OBJECT_SIZE or _u32(bullet, 0) != BULLET_VTABLE:
        raise ProbeError("bullet_payload_invalid")
    extra_floats = {
        "velocity_x": struct.unpack_from("<f", bullet, 0x138)[0],
        "velocity_y": struct.unpack_from("<f", bullet, 0x13C)[0],
        "field_148_float": struct.unpack_from("<f", bullet, 0x148)[0],
        "field_14c_float": struct.unpack_from("<f", bullet, 0x14C)[0],
        "field_158_float": struct.unpack_from("<f", bullet, 0x158)[0],
        "field_15c_float": struct.unpack_from("<f", bullet, 0x15C)[0],
        "heading_radians": struct.unpack_from("<f", bullet, 0x160)[0],
    }
    if not all(math.isfinite(value) for value in extra_floats.values()):
        raise ProbeError("bullet_float_nonfinite")
    return {
        "owner_shooter_address": _u32(bullet, 0x130),
        "owner_shooter_address_hex": f"0x{_u32(bullet, 0x130):08x}",
        "field_134_pointer": _u32(bullet, 0x134),
        **extra_floats,
        "field_164_i32": _i32(bullet, 0x164),
        "fired": bool(bullet[0x16A]),
        "flags_16a_16c_hex": bullet[0x16A:0x16D].hex(),
        "field_174_pointer": _u32(bullet, 0x174),
        "field_178_i32": _i32(bullet, 0x178),
        # Retail code at 0x44A1F9-0x44A206 constructs the embedded list,
        # while 0x45C567 and 0x45C67F index these four per-curve latches.
        "gap_list_sentinel_address": _u32(
            bullet,
            BULLET_GAP_SENTINEL_POINTER_OFFSET,
        ),
        "gap_entry_count": _u32(bullet, BULLET_GAP_COUNT_OFFSET),
        "curve_points": list(
            struct.unpack_from(
                f"<{BULLET_CURVE_POINT_COUNT}i",
                bullet,
                BULLET_CURVE_POINTS_OFFSET,
            )
        ),
    }


def _collect_bullet_gap_state(
    handle: int,
    *,
    bullet: bytes,
    regions: tuple[tuple[int, int], ...],
) -> tuple[dict[str, Any], bytes]:
    """Freeze the retail Bullet gap list and verify its intrusive links."""

    if len(bullet) != BULLET_OBJECT_SIZE or _u32(bullet, 0) != BULLET_VTABLE:
        raise ProbeError("bullet_gap_payload_invalid")
    sentinel = _u32(bullet, BULLET_GAP_SENTINEL_POINTER_OFFSET)
    declared_count = _u32(bullet, BULLET_GAP_COUNT_OFFSET)
    if declared_count > MAX_BULLET_GAP_ENTRIES:
        raise ProbeError("bullet_gap_count_invalid")

    def span_is_readable(address: int, size: int) -> bool:
        region = _containing_region(regions, address)
        if region is None:
            return False
        base, region_size = region
        return address + size <= base + region_size

    if not span_is_readable(sentinel, 8):
        raise ProbeError("bullet_gap_sentinel_invalid")
    sentinel_next, sentinel_previous = struct.unpack(
        "<II",
        read_process_bytes(handle, sentinel, 8),
    )
    node = sentinel_next
    expected_previous = sentinel
    visited: set[int] = set()
    records: list[dict[str, Any]] = []
    payload = bytearray()
    while node != sentinel:
        if (
            len(records) >= declared_count
            or len(records) >= MAX_BULLET_GAP_ENTRIES
            or node in visited
        ):
            raise ProbeError("bullet_gap_list_cycle_or_limit")
        if not span_is_readable(node, BULLET_GAP_NODE_SIZE):
            raise ProbeError("bullet_gap_node_invalid")
        visited.add(node)
        raw = read_process_bytes(handle, node, BULLET_GAP_NODE_SIZE)
        (
            next_node,
            previous_node,
            curve_index,
            gap_distance,
            boundary_ball_id,
        ) = struct.unpack("<IIiii", raw)
        if previous_node != expected_previous:
            raise ProbeError("bullet_gap_previous_link_mismatch")
        if (
            not 0 <= curve_index < BULLET_CURVE_POINT_COUNT
            or gap_distance <= 0
            or boundary_ball_id <= 0
        ):
            raise ProbeError("bullet_gap_entry_invalid")
        records.append(
            {
                "index": len(records),
                "node_address": node,
                "node_address_hex": f"0x{node:08x}",
                "next_node_address": next_node,
                "next_node_address_hex": f"0x{next_node:08x}",
                "previous_node_address": previous_node,
                "previous_node_address_hex": f"0x{previous_node:08x}",
                "curve_index": curve_index,
                "gap_distance": gap_distance,
                "boundary_ball_id": boundary_ball_id,
            }
        )
        payload.extend(raw)
        expected_previous = node
        node = next_node

    if len(records) != declared_count:
        raise ProbeError(
            "bullet_gap_count_mismatch:"
            f"declared={declared_count},traversed={len(records)}"
        )
    if sentinel_previous != expected_previous:
        raise ProbeError("bullet_gap_sentinel_previous_mismatch")
    if records and sentinel_next != records[0]["node_address"]:
        raise ProbeError("bullet_gap_sentinel_next_mismatch")
    if not records and (
        sentinel_next != sentinel or sentinel_previous != sentinel
    ):
        raise ProbeError("bullet_gap_empty_sentinel_mismatch")
    boundary_ids = [record["boundary_ball_id"] for record in records]
    if len(set(boundary_ids)) != len(boundary_ids):
        raise ProbeError("bullet_gap_boundary_id_duplicate")

    return (
        {
            "gap_list_sentinel_address": sentinel,
            "gap_list_sentinel_address_hex": f"0x{sentinel:08x}",
            "gap_entry_count": declared_count,
            "curve_points": list(
                struct.unpack_from(
                    f"<{BULLET_CURVE_POINT_COUNT}i",
                    bullet,
                    BULLET_CURVE_POINTS_OFFSET,
                )
            ),
            "gap_entries": records,
        },
        bytes(payload),
    )


def _curve_payload_layout(vtable: int) -> tuple[str, int]:
    if vtable == BALL_VTABLE:
        return "ball", BALL_OBJECT_SIZE
    if vtable == BULLET_VTABLE:
        return "bullet", BULLET_OBJECT_SIZE
    raise ProbeError(
        f"curve_list_payload_vtable_mismatch:0x{vtable:08x}"
    )


def _collect_shooter_bullets(
    handle: int,
    *,
    shooter: bytes,
    regions: tuple[tuple[int, int], ...],
    output_root: Path,
) -> list[dict[str, Any]]:
    if _u32(shooter, 0) != SHOOTER_VTABLE:
        raise ProbeError("shooter_vtable_mismatch")
    rows: list[dict[str, Any]] = []
    for pointer_offset in SHOOTER_BULLET_POINTER_OFFSETS:
        address = _u32(shooter, pointer_offset)
        row: dict[str, Any] = {
            "shooter_pointer_offset": pointer_offset,
            "shooter_pointer_offset_hex": f"0x{pointer_offset:03x}",
            "address": address,
            "address_hex": f"0x{address:08x}",
        }
        if address:
            if _containing_region(regions, address) is None:
                raise ProbeError("shooter_bullet_pointer_invalid")
            bullet = read_process_bytes(
                handle,
                address,
                BULLET_OBJECT_SIZE,
            )
            if _u32(bullet, 0) != BULLET_VTABLE:
                raise ProbeError("shooter_bullet_vtable_mismatch")
            artifact_path = output_root / (
                f"shooter-bullet-{pointer_offset:03x}.bin"
            )
            artifact_path.write_bytes(bullet)
            row.update(
                {
                    "artifact": artifact_path.name,
                    "artifact_sha256": _sha256_path(artifact_path),
                    "ball": _decode_ball(
                        bullet,
                        expected_vtable=BULLET_VTABLE,
                    ),
                    "subclass_fields": _decode_bullet_subclass(bullet),
                    "summary": _object_summary(
                        handle,
                        bullet,
                        regions,
                    ),
                }
            )
        rows.append(row)
    return rows


def _collect_fired_bullets(
    handle: int,
    *,
    board_address: int,
    board: bytes,
    regions: tuple[tuple[int, int], ...],
    output_root: Path,
) -> dict[str, Any]:
    container_offset = BOARD_FIRED_BULLET_LIST_OFFSET
    sentinel = _u32(board, container_offset + 4)
    declared_count = _u32(board, container_offset + 8)
    if declared_count > MAX_FIRED_BULLETS:
        raise ProbeError("fired_bullet_count_invalid")
    if _containing_region(regions, sentinel) is None:
        raise ProbeError("fired_bullet_sentinel_invalid")
    node = _u32(read_process_bytes(handle, sentinel, 4), 0)
    visited: set[int] = set()
    records: list[dict[str, Any]] = []
    payload_blob = bytearray()
    gap_payload_blob = bytearray()
    while node != sentinel:
        if len(records) >= MAX_FIRED_BULLETS or node in visited:
            raise ProbeError("fired_bullet_list_cycle_or_limit")
        if _containing_region(regions, node) is None:
            raise ProbeError("fired_bullet_node_invalid")
        visited.add(node)
        next_node, previous_node, bullet_address = struct.unpack(
            "<III",
            read_process_bytes(handle, node, 12),
        )
        if _containing_region(regions, bullet_address) is None:
            raise ProbeError("fired_bullet_pointer_invalid")
        bullet = read_process_bytes(
            handle,
            bullet_address,
            BULLET_OBJECT_SIZE,
        )
        if _u32(bullet, 0) != BULLET_VTABLE:
            raise ProbeError("fired_bullet_vtable_mismatch")
        gap_state, gap_payload = _collect_bullet_gap_state(
            handle,
            bullet=bullet,
            regions=regions,
        )
        artifact_offset = len(payload_blob)
        gap_artifact_offset = len(gap_payload_blob)
        payload_blob.extend(bullet)
        gap_payload_blob.extend(gap_payload)
        subclass_fields = _decode_bullet_subclass(bullet)
        subclass_fields.update(gap_state)
        records.append(
            {
                "index": len(records),
                "node_address": node,
                "node_address_hex": f"0x{node:08x}",
                "previous_node_address": previous_node,
                "previous_node_address_hex": f"0x{previous_node:08x}",
                "bullet_address": bullet_address,
                "bullet_address_hex": f"0x{bullet_address:08x}",
                "artifact_offset": artifact_offset,
                "artifact_bytes": len(bullet),
                "payload_sha256": _sha256_bytes(bullet),
                "gap_artifact_offset": gap_artifact_offset,
                "gap_artifact_bytes": len(gap_payload),
                "gap_payload_sha256": _sha256_bytes(gap_payload),
                "ball": _decode_ball(
                    bullet,
                    expected_vtable=BULLET_VTABLE,
                ),
                "subclass_fields": subclass_fields,
            }
        )
        node = next_node
    if len(records) != declared_count:
        raise ProbeError(
            "fired_bullet_count_mismatch:"
            f"declared={declared_count},traversed={len(records)}"
        )
    artifact_path = output_root / "active-fired-bullets.bin"
    artifact_path.write_bytes(payload_blob)
    gap_artifact_path = output_root / "active-fired-bullet-gap-nodes.bin"
    gap_artifact_path.write_bytes(gap_payload_blob)
    return {
        "board_container_offset": container_offset,
        "board_container_offset_hex": f"0x{container_offset:03x}",
        "container_address": board_address + container_offset,
        "container_address_hex": (
            f"0x{board_address + container_offset:08x}"
        ),
        "sentinel_address": sentinel,
        "sentinel_address_hex": f"0x{sentinel:08x}",
        "declared_count": declared_count,
        "traversed_count": len(records),
        "artifact": artifact_path.name,
        "artifact_bytes": len(payload_blob),
        "artifact_sha256": _sha256_path(artifact_path),
        "gap_artifact": gap_artifact_path.name,
        "gap_artifact_bytes": len(gap_payload_blob),
        "gap_artifact_sha256": _sha256_path(gap_artifact_path),
        "records": records,
    }


def _collect_curve_list(
    handle: int,
    *,
    curve_address: int,
    curve_index: int,
    container_offset: int,
    regions: tuple[tuple[int, int], ...],
    output_root: Path,
) -> dict[str, Any]:
    container = curve_address + container_offset
    sentinel = _u32(read_process_bytes(handle, container + 4, 4), 0)
    declared_count = _u32(read_process_bytes(handle, container + 8, 4), 0)
    if declared_count > MAX_CURVE_LIST_ITEMS:
        raise ProbeError("curve_list_declared_count_invalid")
    if _containing_region(regions, sentinel) is None:
        raise ProbeError("curve_list_sentinel_invalid")
    node = _u32(read_process_bytes(handle, sentinel, 4), 0)
    visited: set[int] = set()
    records: list[dict[str, Any]] = []
    payload_blob = bytearray()
    while node != sentinel:
        if len(records) >= MAX_CURVE_LIST_ITEMS or node in visited:
            raise ProbeError("curve_list_cycle_or_limit")
        if _containing_region(regions, node) is None:
            raise ProbeError("curve_list_node_invalid")
        visited.add(node)
        node_data = read_process_bytes(handle, node, 12)
        next_node, previous_node, payload_address = struct.unpack(
            "<III", node_data
        )
        row: dict[str, Any] = {
            "index": len(records),
            "node_address": node,
            "node_address_hex": f"0x{node:08x}",
            "previous_node_address": previous_node,
            "previous_node_address_hex": f"0x{previous_node:08x}",
            "payload_address": payload_address,
            "payload_address_hex": f"0x{payload_address:08x}",
        }
        if payload_address:
            payload_region = _containing_region(regions, payload_address)
            if payload_region is None:
                raise ProbeError("curve_list_payload_invalid")
            payload_region_base, payload_region_size = payload_region
            payload_available = (
                payload_region_base + payload_region_size - payload_address
            )
            payload_head = _u32(
                read_process_bytes(handle, payload_address, 4),
                0,
            )
            payload_kind, payload_size = _curve_payload_layout(
                payload_head
            )
            if payload_available < payload_size:
                raise ProbeError("curve_list_payload_unreadable")
            payload = read_process_bytes(
                handle,
                payload_address,
                payload_size,
            )
            payload_offset = len(payload_blob)
            payload_blob.extend(payload)
            row.update(
                {
                    "artifact_offset": payload_offset,
                    "artifact_bytes": len(payload),
                    "payload_sha256": _sha256_bytes(payload),
                    "payload_kind": payload_kind,
                    "payload_head": payload_head,
                    "payload_head_hex": f"0x{payload_head:08x}",
                    "payload_head_is_vtable": _valid_vtable(
                        handle, payload_head
                    ),
                    "ball": _decode_ball(
                        payload,
                        expected_vtable=payload_head,
                    ),
                }
            )
            if payload_kind == "bullet":
                row["subclass_fields"] = _decode_bullet_subclass(payload)
        records.append(row)
        node = next_node
    if len(records) != declared_count:
        raise ProbeError(
            "curve_list_count_mismatch:"
            f"declared={declared_count},traversed={len(records)}"
        )
    artifact_path = output_root / (
        f"curve-{curve_index:02d}-list-{container_offset:03x}-payloads.bin"
    )
    artifact_path.write_bytes(payload_blob)
    frequencies: dict[str, int] = {}
    for row in records:
        head = row.get("payload_head_hex", "null")
        frequencies[head] = frequencies.get(head, 0) + 1
    return {
        "container_offset": container_offset,
        "container_offset_hex": f"0x{container_offset:03x}",
        "sentinel_address": sentinel,
        "sentinel_address_hex": f"0x{sentinel:08x}",
        "declared_count": declared_count,
        "traversed_count": len(records),
        "payload_count": sum(bool(row["payload_address"]) for row in records),
        "payload_vtable_frequencies": frequencies,
        "artifact": artifact_path.name,
        "artifact_bytes": len(payload_blob),
        "artifact_sha256": _sha256_path(artifact_path),
        "records": records,
    }


def _collect_curve_plan(
    handle: int,
    *,
    curve: bytes,
    curve_index: int,
    regions: tuple[tuple[int, int], ...],
    output_root: Path,
) -> dict[str, Any]:
    """Freeze the retail Curve planned-ball vector and its add-plan flag."""

    if len(curve) != CURVE_DUMP_SIZE:
        raise ProbeError("curve_plan_curve_size_invalid")
    begin = _u32(curve, CURVE_PLANNED_VECTOR_BEGIN_OFFSET)
    end = _u32(curve, CURVE_PLANNED_VECTOR_END_OFFSET)
    capacity = _u32(curve, CURVE_PLANNED_VECTOR_CAPACITY_OFFSET)
    if begin == end == capacity == 0:
        count = 0
        capacity_count = 0
        payload = b""
    else:
        if (
            begin == 0
            or end == 0
            or capacity == 0
            or not begin <= end <= capacity
        ):
            raise ProbeError("curve_plan_vector_bounds_invalid")
        used_bytes = end - begin
        capacity_bytes = capacity - begin
        if (
            used_bytes % CURVE_PLANNED_ITEM_SIZE != 0
            or capacity_bytes % CURVE_PLANNED_ITEM_SIZE != 0
        ):
            raise ProbeError("curve_plan_vector_alignment_invalid")
        count = used_bytes // CURVE_PLANNED_ITEM_SIZE
        capacity_count = capacity_bytes // CURVE_PLANNED_ITEM_SIZE
        if (
            count > MAX_CURVE_PLANNED_ITEMS
            or capacity_count > MAX_CURVE_PLANNED_ITEMS
        ):
            raise ProbeError("curve_plan_vector_count_invalid")
        region = _containing_region(
            regions,
            begin,
            allow_end=(capacity == begin),
        )
        if region is None:
            raise ProbeError("curve_plan_vector_unreadable")
        region_base, region_size = region
        if capacity > region_base + region_size:
            raise ProbeError("curve_plan_vector_unreadable")
        payload = (
            read_process_bytes(handle, begin, used_bytes)
            if used_bytes
            else b""
        )

    add_plan_raw = curve[CURVE_ADD_PLAN_ENABLED_OFFSET]
    if add_plan_raw not in {0, 1}:
        raise ProbeError("curve_add_plan_enabled_invalid")
    artifact_path = output_root / f"curve-{curve_index:02d}-plan.bin"
    artifact_path.write_bytes(payload)
    return {
        "vector_begin_offset": CURVE_PLANNED_VECTOR_BEGIN_OFFSET,
        "vector_end_offset": CURVE_PLANNED_VECTOR_END_OFFSET,
        "vector_capacity_offset": CURVE_PLANNED_VECTOR_CAPACITY_OFFSET,
        "begin_address": begin,
        "begin_address_hex": f"0x{begin:08x}",
        "end_address": end,
        "end_address_hex": f"0x{end:08x}",
        "capacity_address": capacity,
        "capacity_address_hex": f"0x{capacity:08x}",
        "item_size": CURVE_PLANNED_ITEM_SIZE,
        "count": count,
        "capacity_count": capacity_count,
        "add_plan_enabled_offset": CURVE_ADD_PLAN_ENABLED_OFFSET,
        "add_plan_enabled": bool(add_plan_raw),
        "artifact": artifact_path.name,
        "artifact_bytes": len(payload),
        "artifact_sha256": _sha256_path(artifact_path),
    }


def _collect_curves(
    handle: int,
    *,
    board: bytes,
    regions: tuple[tuple[int, int], ...],
    output_root: Path,
) -> dict[str, Any]:
    manager_address = _u32(board, BOARD_CURVE_MANAGER_OFFSET)
    if _containing_region(regions, manager_address) is None:
        raise ProbeError("curve_manager_pointer_invalid")
    manager = read_process_bytes(
        handle,
        manager_address,
        CURVE_MANAGER_DUMP_SIZE,
    )
    if _u32(manager, 0) != CURVE_MANAGER_VTABLE:
        raise ProbeError("curve_manager_vtable_mismatch")
    curve_count = _i32(manager, CURVE_MANAGER_CURVE_COUNT_OFFSET)
    if not 0 < curve_count <= MAX_CURVES:
        raise ProbeError(f"curve_count_invalid:{curve_count}")
    manager_path = output_root / "curve-manager.bin"
    manager_path.write_bytes(manager)
    post_zuma_timer = _i32(
        manager,
        CURVE_MANAGER_POST_ZUMA_TIMER_OFFSET,
    )
    post_zuma_ramp_404 = struct.unpack_from(
        "<f",
        manager,
        CURVE_MANAGER_POST_ZUMA_RAMP_404_OFFSET,
    )[0]
    post_zuma_ramp_408 = struct.unpack_from(
        "<f",
        manager,
        CURVE_MANAGER_POST_ZUMA_RAMP_408_OFFSET,
    )[0]
    if not (
        math.isfinite(post_zuma_ramp_404)
        and math.isfinite(post_zuma_ramp_408)
    ):
        raise ProbeError("curve_manager_post_zuma_diagnostics_invalid")
    curves: list[dict[str, Any]] = []
    for curve_index in range(curve_count):
        curve_address = _u32(
            manager,
            CURVE_MANAGER_CURVE_ARRAY_OFFSET + curve_index * 4,
        )
        if _containing_region(regions, curve_address) is None:
            raise ProbeError("curve_pointer_invalid")
        curve = read_process_bytes(handle, curve_address, CURVE_DUMP_SIZE)
        curve_path = output_root / f"curve-{curve_index:02d}.bin"
        curve_path.write_bytes(curve)
        curves.append(
            {
                "index": curve_index,
                "address": curve_address,
                "address_hex": f"0x{curve_address:08x}",
                "artifact": curve_path.name,
                "artifact_bytes": len(curve),
                "artifact_sha256": _sha256_path(curve_path),
                "summary": _object_summary(handle, curve, regions),
                "planned_balls": _collect_curve_plan(
                    handle,
                    curve=curve,
                    curve_index=curve_index,
                    regions=regions,
                    output_root=output_root,
                ),
                "intrusive_lists": [
                    _collect_curve_list(
                        handle,
                        curve_address=curve_address,
                        curve_index=curve_index,
                        container_offset=container_offset,
                        regions=regions,
                        output_root=output_root,
                    )
                    for container_offset in CURVE_INTRUSIVE_LIST_OFFSETS
                ],
            }
        )
    return {
        "board_offset": BOARD_CURVE_MANAGER_OFFSET,
        "board_offset_hex": f"0x{BOARD_CURVE_MANAGER_OFFSET:04x}",
        "address": manager_address,
        "address_hex": f"0x{manager_address:08x}",
        "vtable": _u32(manager, 0),
        "vtable_hex": f"0x{_u32(manager, 0):08x}",
        "curve_count_offset": CURVE_MANAGER_CURVE_COUNT_OFFSET,
        "curve_array_offset": CURVE_MANAGER_CURVE_ARRAY_OFFSET,
        "curve_count": curve_count,
        "artifact": manager_path.name,
        "artifact_bytes": len(manager),
        "artifact_sha256": _sha256_path(manager_path),
        "summary": _object_summary(handle, manager, regions),
        "post_zuma_diagnostics": {
            "timer_remaining_offset": (
                CURVE_MANAGER_POST_ZUMA_TIMER_OFFSET
            ),
            "timer_remaining": post_zuma_timer,
            "ramp_404_offset": CURVE_MANAGER_POST_ZUMA_RAMP_404_OFFSET,
            "ramp_404": post_zuma_ramp_404,
            "ramp_408_offset": CURVE_MANAGER_POST_ZUMA_RAMP_408_OFFSET,
            "ramp_408": post_zuma_ramp_408,
        },
        "curves": curves,
    }


def _collect_qrand(
    handle: int,
    *,
    board: bytes,
    regions: tuple[tuple[int, int], ...],
    output_root: Path,
) -> dict[str, Any]:
    """Dump the retail Board QRand object and all four owned vectors."""

    address = _u32(board, BOARD_QRAND_POINTER_OFFSET)
    region = _containing_region(regions, address)
    if region is None:
        raise ProbeError("qrand_pointer_invalid")
    region_base, region_size = region
    if region_base + region_size - address < QRAND_OBJECT_SIZE:
        raise ProbeError("qrand_object_unreadable")
    raw = read_process_bytes(handle, address, QRAND_OBJECT_SIZE)
    object_path = output_root / "qrand.bin"
    object_path.write_bytes(raw)

    vectors: list[dict[str, Any]] = []
    for vector_offset, name, value_type in QRAND_VECTOR_LAYOUT:
        begin = _u32(raw, vector_offset + 0x04)
        end = _u32(raw, vector_offset + 0x08)
        capacity = _u32(raw, vector_offset + 0x0C)
        if begin == end == capacity == 0:
            item_count = 0
            capacity_count = 0
            payload = b""
        else:
            if (
                begin == 0
                or end < begin
                or capacity < end
                or begin % 4
                or end % 4
                or capacity % 4
                or (end - begin) % 4
                or (capacity - begin) % 4
            ):
                raise ProbeError(f"qrand_{name}_vector_header_invalid")
            item_count = (end - begin) // 4
            capacity_count = (capacity - begin) // 4
            if (
                item_count > MAX_QRAND_VECTOR_ITEMS
                or capacity_count > MAX_QRAND_VECTOR_ITEMS
            ):
                raise ProbeError(f"qrand_{name}_vector_too_large")
            if capacity_count:
                begin_region = _containing_region(regions, begin)
                capacity_region = _containing_region(regions, capacity - 1)
                if (
                    begin_region is None
                    or capacity_region is None
                    or begin_region != capacity_region
                ):
                    raise ProbeError(
                        f"qrand_{name}_vector_storage_invalid"
                    )
            payload = (
                read_process_bytes(handle, begin, item_count * 4)
                if item_count
                else b""
            )

        artifact_path = output_root / f"qrand-{name}.bin"
        artifact_path.write_bytes(payload)
        if value_type == "float32":
            values: list[int | float] = list(
                struct.unpack(f"<{item_count}f", payload)
            )
            if any(not math.isfinite(value) for value in values):
                raise ProbeError(f"qrand_{name}_nonfinite")
        else:
            values = list(struct.unpack(f"<{item_count}i", payload))
        vectors.append(
            {
                "name": name,
                "value_type": value_type,
                "object_offset": vector_offset,
                "object_offset_hex": f"0x{vector_offset:02x}",
                "begin": begin,
                "begin_hex": f"0x{begin:08x}",
                "end": end,
                "end_hex": f"0x{end:08x}",
                "capacity": capacity,
                "capacity_hex": f"0x{capacity:08x}",
                "item_count": item_count,
                "capacity_count": capacity_count,
                "values": values,
                "artifact": artifact_path.name,
                "artifact_bytes": len(payload),
                "artifact_sha256": _sha256_path(artifact_path),
            }
        )

    return {
        "schema": "zuma-rl.pc-qrand-state",
        "version": 1,
        "board_pointer_offset": BOARD_QRAND_POINTER_OFFSET,
        "board_pointer_offset_hex": (
            f"0x{BOARD_QRAND_POINTER_OFFSET:04x}"
        ),
        "address": address,
        "address_hex": f"0x{address:08x}",
        "update_count": _i32(raw, 0),
        "selected_index": _i32(raw, 4),
        "artifact": object_path.name,
        "artifact_bytes": len(raw),
        "artifact_sha256": _sha256_path(object_path),
        "vectors": vectors,
    }


def _collect_crt_rand_state(
    handle: int,
    *,
    thread_state: Mapping[str, int],
    output_root: Path,
) -> dict[str, Any]:
    """Read the current MSVC CRT rand state for the proven board thread."""

    state_address = int(thread_state["rand_state_address"])
    payload = read_process_bytes(handle, state_address, 4)
    state = _u32(payload, 0)
    artifact_path = output_root / "thread-crt-rand-state.bin"
    artifact_path.write_bytes(payload)
    return {
        "schema": "zuma-rl.pc-thread-crt-rand-state",
        "version": 1,
        **{key: int(value) for key, value in thread_state.items()},
        "state": state,
        "state_hex": f"0x{state:08x}",
        "artifact": artifact_path.name,
        "artifact_bytes": len(payload),
        "artifact_sha256": _sha256_path(artifact_path),
    }


def _collect_mtrand_state(
    handle: int,
    *,
    address: int,
    role: str,
    output_root: Path,
) -> dict[str, Any]:
    """Dump one retail 624-word MTRand object and bind its cursor."""

    payload = read_process_bytes(handle, address, MTRAND_STATE_BYTES)
    index = _u32(payload, MTRAND_STATE_WORDS * 4)
    if index > MTRAND_STATE_WORDS:
        raise ProbeError(f"{role}_mtrand_index_invalid:{index}")
    artifact_path = output_root / f"{role}-mtrand.bin"
    artifact_path.write_bytes(payload)
    return {
        "schema": "zuma-rl.pc-mtrand-state",
        "version": 1,
        "role": role,
        "address": address,
        "address_hex": f"0x{address:08x}",
        "state_word_count": MTRAND_STATE_WORDS,
        "index": index,
        "artifact": artifact_path.name,
        "artifact_bytes": len(payload),
        "artifact_sha256": _sha256_path(artifact_path),
    }


def _read_light_ball_identity(
    handle: int,
    address: int,
) -> dict[str, Any] | None:
    """Read compact identity plus the render-driving Ball base state."""

    if address == 0:
        return None
    payload = read_process_bytes(
        handle,
        address,
        LIGHT_BALL_IDENTITY_BYTES,
    )
    vtable = _u32(payload, 0)
    if vtable == BALL_VTABLE:
        kind = "ball"
    elif vtable == BULLET_VTABLE:
        kind = "bullet"
    else:
        raise ProbeError(
            f"light_ball_vtable_mismatch:0x{vtable:08x}"
        )
    color_id = _i32(payload, 0x14)
    if not 0 <= color_id < BOARD_COLOR_COUNT_SLOTS:
        raise ProbeError(f"light_ball_color_invalid:{color_id}")
    powerup_types = (
        _i32(payload, 0xC8),
        _i32(payload, 0x11C),
        _i32(payload, 0x120),
    )
    if any(not 0 <= value <= 14 for value in powerup_types):
        raise ProbeError(
            "light_ball_powerup_type_invalid:"
            f"{powerup_types!r}"
        )
    render_floats = struct.unpack_from("<8f", payload, 0x1C)
    if not all(math.isfinite(value) for value in render_floats):
        raise ProbeError("light_ball_render_float_nonfinite")
    return {
        "address": address,
        "address_hex": f"0x{address:08x}",
        "kind": kind,
        "vtable": vtable,
        "ball_id": _u32(payload, 0x10),
        "color_id": color_id,
        "powerup_previous_type": powerup_types[0],
        "powerup_primary_type": powerup_types[1],
        "powerup_secondary_type": powerup_types[2],
        **dict(zip(LIGHT_BALL_RENDER_FLOAT_FIELDS, render_floats, strict=True)),
        "render_float32_bits_hex": payload[0x1C:0x3C].hex(),
        "flags_b4_c2_hex": payload[0xB4:0xC3].hex(),
    }


def _read_light_qrand_state(
    handle: int,
    *,
    board: bytes,
) -> dict[str, Any]:
    """Read QRand semantics without producing one artifact per vector."""

    address = _u32(board, BOARD_QRAND_POINTER_OFFSET)
    raw = read_process_bytes(handle, address, QRAND_OBJECT_SIZE)
    vectors: dict[str, list[int | float]] = {}
    for vector_offset, name, value_type in QRAND_VECTOR_LAYOUT:
        begin = _u32(raw, vector_offset + 0x04)
        end = _u32(raw, vector_offset + 0x08)
        capacity = _u32(raw, vector_offset + 0x0C)
        if begin == end == capacity == 0:
            item_count = 0
            payload = b""
        else:
            if (
                begin == 0
                or end < begin
                or capacity < end
                or begin % 4
                or end % 4
                or capacity % 4
                or (end - begin) % 4
                or (capacity - begin) % 4
            ):
                raise ProbeError(
                    f"light_qrand_{name}_vector_header_invalid"
                )
            item_count = (end - begin) // 4
            capacity_count = (capacity - begin) // 4
            if (
                item_count > MAX_QRAND_VECTOR_ITEMS
                or capacity_count > MAX_QRAND_VECTOR_ITEMS
            ):
                raise ProbeError(f"light_qrand_{name}_vector_too_large")
            payload = read_process_bytes(handle, begin, item_count * 4)
        if value_type == "float32":
            values: list[int | float] = list(
                struct.unpack(f"<{item_count}f", payload)
            )
            if not all(math.isfinite(float(value)) for value in values):
                raise ProbeError(f"light_qrand_{name}_nonfinite")
        else:
            values = list(struct.unpack(f"<{item_count}i", payload))
        vectors[name] = values
    return {
        "address": address,
        "update_count": _i32(raw, 0),
        "selected_index": _i32(raw, 4),
        "vectors": vectors,
    }


def _read_light_curve_list(
    handle: int,
    *,
    curve_address: int,
    curve_index: int,
    container_offset: int,
) -> dict[str, Any]:
    """Traverse one curve list while retaining only identity and topology."""

    container = read_process_bytes(
        handle,
        curve_address + container_offset,
        12,
    )
    sentinel = _u32(container, 4)
    declared_count = _u32(container, 8)
    if sentinel == 0 or declared_count > MAX_CURVE_LIST_ITEMS:
        raise ProbeError("light_curve_list_header_invalid")
    first_node, last_node = struct.unpack(
        "<II",
        read_process_bytes(handle, sentinel, 8),
    )
    node = first_node
    previous_node = sentinel
    visited: set[int] = set()
    entities: list[dict[str, Any] | None] = []
    topology = bytearray()
    while node != sentinel:
        if len(entities) >= MAX_CURVE_LIST_ITEMS or node in visited:
            raise ProbeError("light_curve_list_cycle_or_limit")
        visited.add(node)
        next_node, linked_previous, payload_address = struct.unpack(
            "<III",
            read_process_bytes(handle, node, 12),
        )
        if linked_previous != previous_node:
            raise ProbeError("light_curve_list_previous_link_mismatch")
        identity = _read_light_ball_identity(handle, payload_address)
        if identity is None:
            topology.extend(
                struct.pack("<IIIII", node, payload_address, 0, 0, 0)
            )
        else:
            topology.extend(
                struct.pack(
                    "<IIIII",
                    node,
                    payload_address,
                    int(identity["vtable"]),
                    int(identity["ball_id"]),
                    int(identity["color_id"]),
                )
            )
        entities.append(identity)
        previous_node = node
        node = next_node
    if len(entities) != declared_count:
        raise ProbeError(
            "light_curve_list_count_mismatch:"
            f"declared={declared_count},traversed={len(entities)}"
        )
    if previous_node != last_node:
        raise ProbeError("light_curve_list_last_link_mismatch")
    return {
        "curve_index": curve_index,
        "container_offset": container_offset,
        "sentinel_address": sentinel,
        "first_node_address": first_node,
        "last_node_address": last_node,
        "declared_count": declared_count,
        "traversed_count": len(entities),
        "topology_sha256": _sha256_bytes(bytes(topology)),
        "entities": entities,
    }


def _read_rng_trajectory_tick(
    handle: int,
    *,
    update: int,
    thread_crt_state: Mapping[str, int] | None,
) -> tuple[dict[str, Any], bytes]:
    """Read the compact gameplay/RNG state needed for a long trajectory."""

    sexy_app_base = _u32(
        read_process_bytes(handle, G_SEXY_APP_BASE_ADDRESS, 4),
        0,
    )
    board_address = _u32(
        read_process_bytes(
            handle,
            sexy_app_base + ACTIVE_BOARD_OFFSET,
            4,
        ),
        0,
    )
    board = read_process_bytes(handle, board_address, BOARD_DUMP_SIZE)
    if (
        _u32(board, 0) != BOARD_VTABLE
        or _u32(board, BOARD_EMBEDDED_VTABLE_OFFSET)
        != BOARD_EMBEDDED_VTABLE
    ):
        raise ProbeError("rng_trajectory_board_identity_mismatch")

    mtrand_payload = read_process_bytes(
        handle,
        G_FRAMEWORK_MTRAND_ADDRESS,
        MTRAND_STATE_BYTES,
    )
    mtrand_index = _u32(
        mtrand_payload,
        MTRAND_STATE_WORDS * 4,
    )
    if mtrand_index > MTRAND_STATE_WORDS:
        raise ProbeError(
            f"rng_trajectory_mtrand_index_invalid:{mtrand_index}"
        )

    shooter_address = _u32(board, BOARD_PRIMARY_CHILD_OFFSET)
    shooter_bytes = read_process_bytes(
        handle,
        shooter_address,
        max(SHOOTER_BULLET_POINTER_OFFSETS) + 4,
    )
    if _u32(shooter_bytes, 0) != SHOOTER_VTABLE:
        raise ProbeError("rng_trajectory_shooter_vtable_mismatch")
    chamber = [
        _read_light_ball_identity(
            handle,
            _u32(shooter_bytes, pointer_offset),
        )
        for pointer_offset in SHOOTER_BULLET_POINTER_OFFSETS
    ]

    fired_bullet_count = _u32(
        board,
        BOARD_FIRED_BULLET_LIST_OFFSET + 8,
    )
    if fired_bullet_count > MAX_FIRED_BULLETS:
        raise ProbeError("rng_trajectory_fired_bullet_count_invalid")

    manager_address = _u32(board, BOARD_CURVE_MANAGER_OFFSET)
    manager = read_process_bytes(
        handle,
        manager_address,
        CURVE_MANAGER_DUMP_SIZE,
    )
    if _u32(manager, 0) != CURVE_MANAGER_VTABLE:
        raise ProbeError("rng_trajectory_curve_manager_vtable_mismatch")
    curve_count = _i32(manager, CURVE_MANAGER_CURVE_COUNT_OFFSET)
    if not 0 < curve_count <= MAX_CURVES:
        raise ProbeError(f"rng_trajectory_curve_count_invalid:{curve_count}")

    lists: list[dict[str, Any]] = []
    for curve_index in range(curve_count):
        curve_address = _u32(
            manager,
            CURVE_MANAGER_CURVE_ARRAY_OFFSET + curve_index * 4,
        )
        for container_offset in CURVE_INTRUSIVE_LIST_OFFSETS:
            lists.append(
                _read_light_curve_list(
                    handle,
                    curve_address=curve_address,
                    curve_index=curve_index,
                    container_offset=container_offset,
                )
            )

    qrand = _read_light_qrand_state(handle, board=board)
    crt_state = None
    if thread_crt_state is not None:
        crt_state = _u32(
            read_process_bytes(
                handle,
                int(thread_crt_state["rand_state_address"]),
                4,
            ),
            0,
        )
    board_color_counts = list(
        struct.unpack_from(
            f"<{BOARD_COLOR_COUNT_SLOTS}i",
            board,
            BOARD_COLOR_COUNTS_OFFSET,
        )
    )
    if any(
        count < 0 or count > MAX_CURVE_LIST_ITEMS
        for count in board_color_counts
    ):
        raise ProbeError("rng_trajectory_board_color_count_invalid")

    chain_ball_count = sum(
        int(row["traversed_count"])
        for row in lists
        if int(row["container_offset"]) == 0x5C
    )
    pending_ball_count = sum(
        int(row["traversed_count"])
        for row in lists
        if int(row["container_offset"]) == 0x68
    )
    inserting_ball_count = sum(
        int(row["traversed_count"])
        for row in lists
        if int(row["container_offset"]) == 0x50
    )
    return (
        {
            "framework_update": update,
            "score": _i32(board, BOARD_SCORE_OFFSET),
            "score_target": _i32(board, BOARD_SCORE_TARGET_OFFSET),
            "displayed_score": _i32(
                board,
                BOARD_DISPLAYED_SCORE_OFFSET,
            ),
            "board_color_counts": board_color_counts,
            "chain_ball_count": chain_ball_count,
            "pending_ball_count": pending_ball_count,
            "inserting_ball_count": inserting_ball_count,
            "fired_bullet_count": fired_bullet_count,
            "current_ball": chamber[0],
            "next_ball": chamber[1],
            "qrand": qrand,
            "thread_crt_rand_state": crt_state,
            "global_mtrand_index": mtrand_index,
            "curve_lists": lists,
        },
        mtrand_payload,
    )


def collect_active_board(
    pid: int,
    *,
    expected_score: int | None,
    expected_displayed_score: int | None = None,
    output_root: Path,
    thread_crt_state: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Resolve and dump the fixed retail active Board object read-only."""

    handle = open_process_readonly(pid)
    try:
        regions = tuple(_readable_regions(handle))
        sexy_app_base = _u32(
            read_process_bytes(handle, G_SEXY_APP_BASE_ADDRESS, 4), 0
        )
        if _containing_region(regions, sexy_app_base) is None:
            raise ProbeError("gSexyAppBase_pointer_invalid")
        board_address = _u32(
            read_process_bytes(
                handle,
                sexy_app_base + ACTIVE_BOARD_OFFSET,
                4,
            ),
            0,
        )
        board = read_process_bytes(handle, board_address, BOARD_DUMP_SIZE)
        if _u32(board, 0) != BOARD_VTABLE:
            raise ProbeError("active_board_vtable_mismatch")
        if (
            _u32(board, BOARD_EMBEDDED_VTABLE_OFFSET)
            != BOARD_EMBEDDED_VTABLE
        ):
            raise ProbeError("active_board_embedded_vtable_mismatch")
        score = _i32(board, BOARD_SCORE_OFFSET)
        displayed_score = _i32(board, BOARD_DISPLAYED_SCORE_OFFSET)
        board_mode_flag_raw = board[BOARD_MODE_FLAG_1064_OFFSET]
        if board_mode_flag_raw not in {0, 1}:
            raise ProbeError("active_board_mode_flag_1064_invalid")
        displayed_target = (
            expected_score
            if expected_displayed_score is None
            else expected_displayed_score
        )
        if (
            expected_score is not None
            and (
                score != expected_score
                or displayed_score != displayed_target
            )
        ):
            raise ProbeError(
                "active_board_score_mismatch:"
                f"score={score},displayed={displayed_score},"
                f"expected_score={expected_score},"
                f"expected_displayed={displayed_target}"
            )

        board_path = output_root / "active-board.bin"
        board_path.write_bytes(board)
        result: dict[str, Any] = {
            "schema": "zuma-rl.pc-active-board-probe",
            "version": 2,
            "g_sexy_app_base_pointer_address": G_SEXY_APP_BASE_ADDRESS,
            "g_sexy_app_base_pointer_address_hex": (
                f"0x{G_SEXY_APP_BASE_ADDRESS:08x}"
            ),
            "sexy_app_base": sexy_app_base,
            "sexy_app_base_hex": f"0x{sexy_app_base:08x}",
            "active_board_pointer_offset": ACTIVE_BOARD_OFFSET,
            "active_board_pointer_offset_hex": (
                f"0x{ACTIVE_BOARD_OFFSET:04x}"
            ),
            "board_address": board_address,
            "board_address_hex": f"0x{board_address:08x}",
            "board_vtable": _u32(board, 0),
            "board_vtable_hex": f"0x{_u32(board, 0):08x}",
            "embedded_vtable_offset": BOARD_EMBEDDED_VTABLE_OFFSET,
            "embedded_vtable": _u32(
                board, BOARD_EMBEDDED_VTABLE_OFFSET
            ),
            "score_offset": BOARD_SCORE_OFFSET,
            "score": score,
            "score_target_offset": BOARD_SCORE_TARGET_OFFSET,
            "score_target": _i32(board, BOARD_SCORE_TARGET_OFFSET),
            "displayed_score_offset": BOARD_DISPLAYED_SCORE_OFFSET,
            "displayed_score": displayed_score,
            "mode_flag_1064_offset": BOARD_MODE_FLAG_1064_OFFSET,
            "mode_flag_1064": bool(board_mode_flag_raw),
            "artifact": board_path.name,
            "artifact_bytes": len(board),
            "artifact_sha256": _sha256_path(board_path),
            "summary": _object_summary(handle, board, regions),
        }
        curve_plan_exhausted_raw = read_process_bytes(
            handle,
            G_CURVE_PLAN_EXHAUSTED_ADDRESS,
            1,
        )
        if curve_plan_exhausted_raw[0] not in {0, 1}:
            raise ProbeError("curve_plan_exhausted_global_invalid")
        curve_plan_exhausted_path = (
            output_root / "curve-plan-exhausted.bin"
        )
        curve_plan_exhausted_path.write_bytes(curve_plan_exhausted_raw)
        result["curve_plan_exhausted"] = {
            "address": G_CURVE_PLAN_EXHAUSTED_ADDRESS,
            "address_hex": f"0x{G_CURVE_PLAN_EXHAUSTED_ADDRESS:08x}",
            "value": bool(curve_plan_exhausted_raw[0]),
            "artifact": curve_plan_exhausted_path.name,
            "artifact_bytes": 1,
            "artifact_sha256": _sha256_path(curve_plan_exhausted_path),
        }
        result["qrand"] = _collect_qrand(
            handle,
            board=board,
            regions=regions,
            output_root=output_root,
        )
        result["global_mtrand"] = _collect_mtrand_state(
            handle,
            address=G_FRAMEWORK_MTRAND_ADDRESS,
            role="global",
            output_root=output_root,
        )
        if thread_crt_state is not None:
            result["thread_crt_rand"] = _collect_crt_rand_state(
                handle,
                thread_state=thread_crt_state,
                output_root=output_root,
            )

        child_address = _u32(board, BOARD_PRIMARY_CHILD_OFFSET)
        child_region = _containing_region(regions, child_address)
        if child_region is None:
            raise ProbeError("active_board_primary_child_pointer_invalid")
        child_base, child_region_size = child_region
        child_available = child_base + child_region_size - child_address
        if child_available < BOARD_PRIMARY_CHILD_DUMP_SIZE:
            raise ProbeError("active_board_primary_child_unreadable")
        child = read_process_bytes(
            handle,
            child_address,
            BOARD_PRIMARY_CHILD_DUMP_SIZE,
        )
        child_path = output_root / "active-board-field-068c.bin"
        child_path.write_bytes(child)
        result["primary_child"] = {
            "board_offset": BOARD_PRIMARY_CHILD_OFFSET,
            "board_offset_hex": f"0x{BOARD_PRIMARY_CHILD_OFFSET:04x}",
            "address": child_address,
            "address_hex": f"0x{child_address:08x}",
            "artifact": child_path.name,
            "artifact_sha256": _sha256_path(child_path),
            "summary": _object_summary(handle, child, regions),
            "bullets": _collect_shooter_bullets(
                handle,
                shooter=child,
                regions=regions,
                output_root=output_root,
            ),
        }
        result["fired_bullets"] = _collect_fired_bullets(
            handle,
            board_address=board_address,
            board=board,
            regions=regions,
            output_root=output_root,
        )
        result["curve_manager"] = _collect_curves(
            handle,
            board=board,
            regions=regions,
            output_root=output_root,
        )
        return result
    finally:
        close_process(handle)


def _replay_state_row(state: ReplayState) -> dict[str, Any]:
    return {
        "multiplier_address": state.multiplier_address,
        "frame_time_ms": state.frame_time_ms,
        "update_multiplier": state.update_multiplier,
        "update_count": state.update_count,
        "draw_count": state.draw_count,
        "non_draw_count": state.non_draw_count,
        "sleep_count": state.sleep_count,
        "paused": state.paused,
        "fast_forward_target": state.fast_forward_target,
        "fast_forward_to_marker": state.fast_forward_to_marker,
        "fast_forward_step": state.fast_forward_step,
        "step_mode": state.step_mode,
        "update_app_state": state.update_app_state,
        "update_app_depth": state.update_app_depth,
        "loading_thread_started": state.loading_thread_started,
        "loading_thread_completed": state.loading_thread_completed,
        "loaded": state.loaded,
    }


def _trajectory_sample_barrier(
    state: ReplayState,
    *,
    update: int,
    start_update: int,
    frozen_state: ReplayState,
) -> str:
    """Fail closed unless a sample is beyond the retail update barrier."""

    if (
        state.update_count != update
        or state.update_multiplier != frozen_state.update_multiplier
        or state.frame_time_ms != frozen_state.frame_time_ms
        or state.fast_forward_to_marker
        or state.fast_forward_step
    ):
        raise ProbeError("trajectory_replay_state_mismatch")
    if update == start_update:
        if not 0 <= state.fast_forward_target <= update:
            raise ProbeError("trajectory_initial_barrier_invalid")
        return INITIAL_SAMPLE_BARRIER
    if state.fast_forward_target != update:
        raise ProbeError("trajectory_step_barrier_invalid")
    return STEPPED_SAMPLE_BARRIER


def _trajectory_tick_summary(
    update: int,
    board: Mapping[str, Any],
) -> dict[str, Any]:
    shooter = board["primary_child"]
    chamber = shooter["bullets"]
    curves = board["curve_manager"]["curves"]
    list_counts: list[dict[str, Any]] = []
    chain_ball_count = 0
    for curve in curves:
        for intrusive in curve["intrusive_lists"]:
            count = int(intrusive["traversed_count"])
            if int(intrusive["container_offset"]) == 0x5C:
                chain_ball_count += count
            list_counts.append(
                {
                    "curve_index": int(curve["index"]),
                    "container_offset": int(
                        intrusive["container_offset"]
                    ),
                    "count": count,
                }
            )
    current_ball = chamber[0].get("ball")
    next_ball = chamber[1].get("ball")
    qrand = board.get("qrand")
    thread_crt_rand = board.get("thread_crt_rand")
    global_mtrand = board.get("global_mtrand")
    result = {
        "framework_update": update,
        "score": int(board["score"]),
        "displayed_score": int(board["displayed_score"]),
        "score_target": int(board["score_target"]),
        "chain_ball_count": chain_ball_count,
        "fired_bullet_count": int(
            board["fired_bullets"]["traversed_count"]
        ),
        "current_ball_id": (
            int(current_ball["ball_id"])
            if current_ball is not None
            else None
        ),
        "current_color_id": (
            int(current_ball["color_id"])
            if current_ball is not None
            else None
        ),
        "next_ball_id": (
            int(next_ball["ball_id"])
            if next_ball is not None
            else None
        ),
        "next_color_id": (
            int(next_ball["color_id"])
            if next_ball is not None
            else None
        ),
        "qrand_update_count": (
            int(qrand["update_count"])
            if isinstance(qrand, Mapping)
            else None
        ),
        "qrand_selected_index": (
            int(qrand["selected_index"])
            if isinstance(qrand, Mapping)
            else None
        ),
        "thread_crt_rand_state": (
            int(thread_crt_rand["state"])
            if isinstance(thread_crt_rand, Mapping)
            else None
        ),
        "global_mtrand_index": (
            int(global_mtrand["index"])
            if isinstance(global_mtrand, Mapping)
            else None
        ),
        "list_counts": list_counts,
    }
    if board.get("version") is not None:
        result["active_board_version"] = int(board["version"])
    if board.get("mode_flag_1064") is not None:
        result["board_mode_flag_1064"] = bool(
            board["mode_flag_1064"]
        )
    exhausted = board.get("curve_plan_exhausted")
    if isinstance(exhausted, Mapping):
        result["curve_plan_exhausted"] = bool(exhausted["value"])
    diagnostics = board["curve_manager"].get("post_zuma_diagnostics")
    if isinstance(diagnostics, Mapping):
        result["post_zuma_timer_remaining"] = int(
            diagnostics["timer_remaining"]
        )
        result["post_zuma_ramp_404"] = float(
            diagnostics["ramp_404"]
        )
        result["post_zuma_ramp_408"] = float(
            diagnostics["ramp_408"]
        )
    curve_plans: list[dict[str, Any]] = []
    for curve in curves:
        planned = curve.get("planned_balls")
        if not isinstance(planned, Mapping):
            continue
        curve_plans.append(
            {
                "curve_index": int(curve["index"]),
                "begin_address": int(planned["begin_address"]),
                "end_address": int(planned["end_address"]),
                "capacity_address": int(planned["capacity_address"]),
                "planned_count": int(planned["count"]),
                "capacity_count": int(planned["capacity_count"]),
                "add_plan_enabled": bool(
                    planned["add_plan_enabled"]
                ),
            }
        )
    if curve_plans:
        result["curve_plans"] = curve_plans
    return result


def collect_board_trajectory(
    pid: int,
    *,
    frozen_state: ReplayState,
    end_update: int,
    output_root: Path,
    thread_crt_state: Mapping[str, int] | None = None,
    process_identity: Mapping[str, Any] | None = None,
    formal_full_state_evidence: bool = False,
) -> Mapping[str, Any]:
    """Single-step a frozen replay and retain raw Board state for every tick."""

    start_update = frozen_state.update_count
    tick_count = end_update - start_update + 1
    if (
        end_update < start_update
        or tick_count < 1
        or tick_count > MAX_TRAJECTORY_TICKS
    ):
        raise ProbeError("trajectory_update_range_invalid")
    if output_root.exists():
        raise ProbeError("trajectory_output_root_exists")
    normalized_process_identity: dict[str, Any] | None = None
    if process_identity is not None:
        try:
            normalized_process_identity = {
                "process_id": int(process_identity["process_id"]),
                "process_creation_filetime_100ns": int(
                    process_identity["process_creation_filetime_100ns"]
                ),
                "executable_bytes": int(
                    process_identity["executable_bytes"]
                ),
                "executable_sha256": str(
                    process_identity["executable_sha256"]
                ),
                "window_handle_hex": str(
                    process_identity["window_handle_hex"]
                ),
                "window_client_region": list(
                    process_identity["window_client_region"]
                ),
            }
            provenance = process_identity.get(
                "executable_hash_provenance"
            )
            if provenance is not None:
                if not isinstance(provenance, Mapping):
                    raise TypeError
                normalized_process_identity[
                    "executable_hash_provenance"
                ] = dict(provenance)
        except (KeyError, TypeError, ValueError) as error:
            raise ProbeError("trajectory_process_identity_invalid") from error
        if (
            normalized_process_identity["process_id"] != pid
            or normalized_process_identity[
                "process_creation_filetime_100ns"
            ]
            <= 0
        ):
            raise ProbeError("trajectory_process_identity_invalid")
    if formal_full_state_evidence and normalized_process_identity is None:
        raise ProbeError("formal_full_state_contract_incomplete")
    output_root.mkdir()

    hwnd = main_window_for_pid(pid)
    handle = open_process_readonly(pid)
    rows: list[dict[str, Any]] = []
    try:
        state = read_replay_state(
            handle,
            frozen_state.multiplier_address,
        )
        if (
            state.update_count != frozen_state.update_count
            or state.update_multiplier != frozen_state.update_multiplier
            or state.frame_time_ms != frozen_state.frame_time_ms
        ):
            raise ProbeError("trajectory_frozen_state_changed")
        for update in range(start_update, end_update + 1):
            if update != start_update:
                state = step_to(
                    hwnd,
                    handle,
                    frozen_state.multiplier_address,
                    update,
                    timeout_per_step=5.0,
                )
            sample_barrier = _trajectory_sample_barrier(
                state,
                update=update,
                start_update=start_update,
                frozen_state=frozen_state,
            )

            tick_root = output_root / f"tick-{update:08d}"
            tick_root.mkdir()
            board = collect_active_board(
                pid,
                expected_score=None,
                output_root=tick_root,
                thread_crt_state=thread_crt_state,
            )
            tick_payload = {
                "schema": "zuma-rl.pc-memory-trajectory-tick",
                "version": TRAJECTORY_TICK_VERSION,
                "framework_update": update,
                "sample_phase": TRAJECTORY_SAMPLE_PHASE,
                "sample_barrier": sample_barrier,
                "replay_state": _replay_state_row(state),
                "active_board": board,
            }
            tick_path = tick_root / "tick.json"
            tick_path.write_bytes(
                (
                    json.dumps(
                        tick_payload,
                        ensure_ascii=True,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode("ascii")
            )
            rows.append(
                {
                    **_trajectory_tick_summary(update, board),
                    "artifact": (
                        f"tick-{update:08d}/tick.json"
                    ),
                    "artifact_sha256": _sha256_path(tick_path),
                }
            )
            print(
                "trajectory_update="
                f"{update} fired={rows[-1]['fired_bullet_count']} "
                f"chain={rows[-1]['chain_ball_count']} "
                f"score={rows[-1]['score']}",
                flush=True,
            )
    finally:
        close_process(handle)

    index = {
        "schema": "zuma-rl.pc-memory-trajectory",
        "version": TRAJECTORY_VERSION,
        "evidence_classification": (
            FORMAL_FULL_STATE_CLASSIFICATION
            if formal_full_state_evidence
            else "diagnostic"
        ),
        "process_identity": normalized_process_identity,
        "sample_phase": TRAJECTORY_SAMPLE_PHASE,
        "freeze_update": start_update,
        "start_update": start_update,
        "end_update": end_update,
        "tick_count": len(rows),
        "ticks": rows,
    }
    index_path = output_root / "index.json"
    index_path.write_bytes(
        (
            json.dumps(
                index,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("ascii")
    )
    return {
        "schema": index["schema"],
        "version": index["version"],
        "artifact": f"{output_root.name}/index.json",
        "artifact_sha256": _sha256_path(index_path),
        "freeze_update": start_update,
        "start_update": start_update,
        "end_update": end_update,
        "tick_count": len(rows),
    }


def _set_square_dwm_corners(window_handle: int) -> Mapping[str, Any]:
    """Disable Windows 11 rounded-corner clipping for one game window."""

    if window_handle <= 0:
        raise ProbeError("dwm_corner_window_invalid")
    try:
        dwmapi = ctypes.WinDLL("dwmapi", use_last_error=True)
        set_attribute = dwmapi.DwmSetWindowAttribute
        get_attribute = dwmapi.DwmGetWindowAttribute
    except (AttributeError, OSError):
        raise ProbeError("dwm_corner_api_unavailable") from None

    attribute = 33  # DWMWA_WINDOW_CORNER_PREFERENCE
    do_not_round = 1  # DWMWCP_DONOTROUND
    set_attribute.argtypes = (
        wintypes.HWND,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
    )
    set_attribute.restype = ctypes.c_long
    get_attribute.argtypes = set_attribute.argtypes
    get_attribute.restype = ctypes.c_long

    requested = ctypes.c_int(do_not_round)
    result = int(
        set_attribute(
            wintypes.HWND(window_handle),
            attribute,
            ctypes.byref(requested),
            ctypes.sizeof(requested),
        )
    )
    if result != 0:
        raise ProbeError("dwm_corner_preference_set_failed")

    observed = ctypes.c_int(-1)
    result = int(
        get_attribute(
            wintypes.HWND(window_handle),
            attribute,
            ctypes.byref(observed),
            ctypes.sizeof(observed),
        )
    )
    if result != 0 or observed.value != do_not_round:
        raise ProbeError("dwm_corner_preference_verify_failed")
    return {
        "schema": "zuma-rl.pc-dwm-corner-preference",
        "version": 1,
        "window_handle_hex": f"0x{window_handle:016x}",
        "attribute": "DWMWA_WINDOW_CORNER_PREFERENCE",
        "requested": "DWMWCP_DONOTROUND",
        "observed_value": observed.value,
        "status": "verified",
    }


def collect_rng_trajectory(
    pid: int,
    *,
    frozen_state: ReplayState,
    end_update: int,
    output_root: Path,
    thread_crt_state: Mapping[str, int] | None = None,
    record_start_update: int | None = None,
    snapshot_trajectory_frames: bool = False,
    snapshot_process_name: str | None = None,
    square_dwm_corners: bool = False,
    snapshot_warmup_frame: bool = False,
    process_identity: Mapping[str, Any] | None = None,
    formal_exact_step_evidence: bool = False,
    snapshot_repaint_mechanism: str = NONCLIENT_DRAG_REPAINT_MECHANISM,
    on_record_start: Callable[[], None] | None = None,
    trajectory_step_timeout_seconds: float = 5.0,
) -> Mapping[str, Any]:
    """Single-step a compact long-form RNG and topology trajectory."""

    freeze_update = frozen_state.update_count
    start_update = (
        freeze_update
        if record_start_update is None
        else record_start_update
    )
    stepped_tick_count = end_update - freeze_update + 1
    tick_count = end_update - start_update + 1
    if (
        isinstance(trajectory_step_timeout_seconds, bool)
        or not isinstance(trajectory_step_timeout_seconds, (int, float))
        or not math.isfinite(float(trajectory_step_timeout_seconds))
        or not 0.0 < float(trajectory_step_timeout_seconds) <= 3600.0
    ):
        raise ProbeError("trajectory_step_timeout_invalid")
    normalized_step_timeout = float(trajectory_step_timeout_seconds)
    if (
        start_update < freeze_update
        or end_update < start_update
        or tick_count < 1
        or stepped_tick_count > MAX_TRAJECTORY_TICKS
    ):
        raise ProbeError("trajectory_update_range_invalid")
    if output_root.exists():
        raise ProbeError("trajectory_output_root_exists")
    if snapshot_trajectory_frames and not snapshot_process_name:
        raise ProbeError("trajectory_snapshot_process_name_missing")
    if square_dwm_corners and not snapshot_trajectory_frames:
        raise ProbeError("trajectory_square_corners_require_snapshots")
    if snapshot_warmup_frame and not snapshot_trajectory_frames:
        raise ProbeError("trajectory_warmup_requires_snapshots")
    if snapshot_repaint_mechanism not in {
        NONCLIENT_DRAG_REPAINT_MECHANISM,
        SET_WINDOW_POS_REPAINT_MECHANISM,
    }:
        raise ProbeError("trajectory_repaint_mechanism_invalid")
    if formal_exact_step_evidence and (
        not snapshot_trajectory_frames
        or not square_dwm_corners
        or not snapshot_warmup_frame
        or process_identity is None
    ):
        raise ProbeError("formal_exact_step_contract_incomplete")
    normalized_process_identity: dict[str, Any] | None = None
    if process_identity is not None:
        try:
            normalized_process_identity = {
                "process_id": int(process_identity["process_id"]),
                "process_creation_filetime_100ns": int(
                    process_identity["process_creation_filetime_100ns"]
                ),
                "executable_bytes": int(
                    process_identity["executable_bytes"]
                ),
                "executable_sha256": str(
                    process_identity["executable_sha256"]
                ),
                "window_handle_hex": str(
                    process_identity["window_handle_hex"]
                ),
                "window_client_region": list(
                    process_identity["window_client_region"]
                ),
            }
            provenance = process_identity.get(
                "executable_hash_provenance"
            )
            if provenance is not None:
                if not isinstance(provenance, Mapping):
                    raise TypeError
                normalized_process_identity[
                    "executable_hash_provenance"
                ] = dict(provenance)
        except (KeyError, TypeError, ValueError) as error:
            raise ProbeError("trajectory_process_identity_invalid") from error
        if (
            normalized_process_identity["process_id"] != pid
            or normalized_process_identity[
                "process_creation_filetime_100ns"
            ]
            <= 0
        ):
            raise ProbeError("trajectory_process_identity_invalid")
    snapshot_semantics = (
        FORMAL_EXACT_STEP_SNAPSHOT_SEMANTICS
        if formal_exact_step_evidence
        else DIAGNOSTIC_SNAPSHOT_SEMANTICS
    )
    output_root.mkdir()
    frames_root = output_root / "frames"
    snapshot_target = None
    if snapshot_trajectory_frames:
        frames_root.mkdir()

    hwnd = main_window_for_pid(pid)
    if snapshot_trajectory_frames:
        snapshot_target = _wait_for_capture_window(pid)
        if (
            snapshot_target.process_id != pid
            or snapshot_target.window_handle != hwnd
        ):
            raise ProbeError("trajectory_snapshot_window_mismatch")
    repaint_handshake = (
        _window_position_repaint_handshake
        if snapshot_repaint_mechanism == SET_WINDOW_POS_REPAINT_MECHANISM
        else _window_repaint_handshake
    )
    corner_receipt = None
    if square_dwm_corners:
        assert snapshot_target is not None
        corner_receipt = dict(
            _set_square_dwm_corners(snapshot_target.window_handle)
        )
    handle = open_process_readonly(pid)
    rows: list[dict[str, Any]] = []
    warmup_snapshot = None
    mtrand_path = output_root / "global-mtrand-frames.bin"
    previous_event_key: tuple[Any, ...] | None = None
    try:
        state = read_replay_state(
            handle,
            frozen_state.multiplier_address,
        )
        if (
            state.update_count != freeze_update
            or state.update_multiplier != frozen_state.update_multiplier
            or state.frame_time_ms != frozen_state.frame_time_ms
        ):
            raise ProbeError("trajectory_frozen_state_changed")
        if snapshot_warmup_frame:
            assert snapshot_target is not None
            warmup_path = frames_root / f"warmup-u{freeze_update:08d}.bmp"
            _activate_window(hwnd)
            repaint = dict(repaint_handshake(snapshot_target))
            capture = snapshot_dxgi_window(
                output=warmup_path,
                process_name=str(snapshot_process_name),
                device_index=0,
                output_index=0,
                timeout=TRAJECTORY_SNAPSHOT_TIMEOUT_SECONDS,
                evidence_semantics=snapshot_semantics,
            )
            if int(capture["process_id"]) != pid:
                raise ProbeError("trajectory_snapshot_process_mismatch")
            warmup_snapshot = {
                "framework_update": freeze_update,
                "artifact": f"frames/{warmup_path.name}",
                "artifact_bytes": warmup_path.stat().st_size,
                "artifact_sha256": _sha256_path(warmup_path),
                "repaint": repaint,
                "capture": {
                    key: value
                    for key, value in capture.items()
                    if key != "output"
                },
                "acceptance_role": "discarded_dxgi_surface_warmup",
            }
        if start_update != freeze_update:
            state = step_to(
                hwnd,
                handle,
                frozen_state.multiplier_address,
                start_update,
                timeout_per_step=normalized_step_timeout,
            )
        if on_record_start is not None:
            on_record_start()
        with mtrand_path.open("xb") as mtrand_stream:
            for update in range(start_update, end_update + 1):
                if update != start_update:
                    state = step_to(
                        hwnd,
                        handle,
                        frozen_state.multiplier_address,
                        update,
                        timeout_per_step=normalized_step_timeout,
                    )
                sample_barrier = _trajectory_sample_barrier(
                    state,
                    update=update,
                    start_update=freeze_update,
                    frozen_state=frozen_state,
                )

                row, mtrand_payload = _read_rng_trajectory_tick(
                    handle,
                    update=update,
                    thread_crt_state=thread_crt_state,
                )
                if snapshot_trajectory_frames:
                    assert snapshot_target is not None
                    frame_path = frames_root / f"u{update:08d}.bmp"
                    _activate_window(hwnd)
                    repaint = dict(repaint_handshake(snapshot_target))
                    capture = snapshot_dxgi_window(
                        output=frame_path,
                        process_name=str(snapshot_process_name),
                        device_index=0,
                        output_index=0,
                        timeout=TRAJECTORY_SNAPSHOT_TIMEOUT_SECONDS,
                        evidence_semantics=snapshot_semantics,
                    )
                    if int(capture["process_id"]) != pid:
                        raise ProbeError("trajectory_snapshot_process_mismatch")
                    row["visual_snapshot"] = {
                        "schema": "zuma-rl.pc-step-visual-snapshot",
                        "version": 1,
                        "artifact": f"frames/{frame_path.name}",
                        "artifact_bytes": frame_path.stat().st_size,
                        "artifact_sha256": _sha256_path(frame_path),
                        "repaint": repaint,
                        "capture": {
                            key: value
                            for key, value in capture.items()
                            if key != "output"
                        },
                    }
                record_offset = mtrand_stream.tell()
                if record_offset != len(rows) * MTRAND_STATE_BYTES:
                    raise ProbeError("rng_trajectory_mtrand_offset_mismatch")
                mtrand_stream.write(mtrand_payload)
                row.update(
                    {
                        "sample_phase": TRAJECTORY_SAMPLE_PHASE,
                        "sample_barrier": sample_barrier,
                        "replay_state": _replay_state_row(state),
                        "global_mtrand_record_offset": record_offset,
                        "global_mtrand_record_bytes": len(mtrand_payload),
                        "global_mtrand_record_sha256": _sha256_bytes(
                            mtrand_payload
                        ),
                    }
                )
                rows.append(row)

                current = row["current_ball"]
                following = row["next_ball"]
                event_key = (
                    row["chain_ball_count"],
                    row["pending_ball_count"],
                    row["inserting_ball_count"],
                    row["fired_bullet_count"],
                    (
                        current["ball_id"],
                        current["color_id"],
                    )
                    if isinstance(current, Mapping)
                    else None,
                    (
                        following["ball_id"],
                        following["color_id"],
                    )
                    if isinstance(following, Mapping)
                    else None,
                    row["qrand"]["update_count"],
                )
                if (
                    event_key != previous_event_key
                    or update == end_update
                    or (update - start_update) % 50 == 0
                ):
                    print(
                        "rng_trajectory_update="
                        f"{update} mt={row['global_mtrand_index']} "
                        f"chain={row['chain_ball_count']} "
                        f"pending={row['pending_ball_count']} "
                        f"fired={row['fired_bullet_count']} "
                        f"score={row['score']}",
                        flush=True,
                    )
                previous_event_key = event_key
    finally:
        close_process(handle)

    expected_mtrand_bytes = len(rows) * MTRAND_STATE_BYTES
    if mtrand_path.stat().st_size != expected_mtrand_bytes:
        raise ProbeError("rng_trajectory_mtrand_stream_size_mismatch")
    index = {
        "schema": "zuma-rl.pc-rng-trajectory",
        "version": RNG_TRAJECTORY_VERSION,
        "evidence_classification": (
            "formal_pc_golden_exact_step_source"
            if formal_exact_step_evidence
            else "diagnostic"
        ),
        "process_identity": normalized_process_identity,
        "sample_phase": TRAJECTORY_SAMPLE_PHASE,
        "freeze_update": freeze_update,
        "start_update": start_update,
        "end_update": end_update,
        "warmup_tick_count": start_update - freeze_update,
        "tick_count": len(rows),
        "visual_snapshots": {
            "enabled": snapshot_trajectory_frames,
            "frame_count": (
                len(rows) if snapshot_trajectory_frames else 0
            ),
            "sample_phase": TRAJECTORY_SAMPLE_PHASE,
            "foreground_activation": (
                "show_restore_attach_input_then_set_foreground_per_frame"
                if snapshot_trajectory_frames
                else None
            ),
            "repaint_handshake": (
                f"{snapshot_repaint_mechanism}_per_frame"
                if snapshot_trajectory_frames
                else None
            ),
            "semantics": snapshot_semantics,
            "warmup_frame": warmup_snapshot,
        },
        "entity_render_state": {
            "schema": "zuma-rl.pc-light-ball-render-state",
            "version": 1,
            "scope": (
                "shooter_current_next_and_all_curve_intrusive_list_entities"
            ),
            "float_fields": list(LIGHT_BALL_RENDER_FLOAT_FIELDS),
            "float_binding": "exact_little_endian_float32_bits",
            "flag_binding": "exact_ball_base_bytes_b4_through_c2",
        },
        "global_mtrand": {
            "address": G_FRAMEWORK_MTRAND_ADDRESS,
            "address_hex": f"0x{G_FRAMEWORK_MTRAND_ADDRESS:08x}",
            "state_word_count": MTRAND_STATE_WORDS,
            "record_bytes": MTRAND_STATE_BYTES,
            "artifact": mtrand_path.name,
            "artifact_bytes": expected_mtrand_bytes,
            "artifact_sha256": _sha256_path(mtrand_path),
        },
        "ticks": rows,
        "window_transport": {
            "square_dwm_corners": corner_receipt,
        },
    }
    index_path = output_root / "index.json"
    index_path.write_bytes(
        (
            json.dumps(
                index,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("ascii")
    )
    return {
        "schema": index["schema"],
        "version": index["version"],
        "artifact": f"{output_root.name}/index.json",
        "artifact_sha256": _sha256_path(index_path),
        "start_update": start_update,
        "end_update": end_update,
        "freeze_update": freeze_update,
        "warmup_tick_count": start_update - freeze_update,
        "tick_count": len(rows),
        "visual_snapshot_count": (
            len(rows) if snapshot_trajectory_frames else 0
        ),
    }


def scan_int32(pid: int, value: int) -> dict[str, Any]:
    """Scan readable target memory and return deterministic structural hints."""

    handle = open_process_readonly(pid)
    hits: list[dict[str, Any]] = []
    scanned_regions = 0
    scanned_bytes = 0
    try:
        for base, size in _readable_regions(handle):
            data = _read_region(handle, base, size)
            if data is None:
                continue
            scanned_regions += 1
            scanned_bytes += len(data)
            for offset in _int32_offsets(data, value, base):
                address = base + offset
                hits.append(
                    {
                        "address": address,
                        "address_hex": f"0x{address:08x}",
                        "region_base": base,
                        "region_base_hex": f"0x{base:08x}",
                        "region_size": len(data),
                        "image": IMAGE_START <= address < IMAGE_END,
                        "context": _context_rows(data, offset),
                        "candidate_objects": _candidate_objects(
                            handle,
                            region_base=base,
                            region_data=data,
                            hit_offset=offset,
                        ),
                    }
                )
        reference_targets, reference_target_count = (
            _bounded_reference_targets(hits)
        )
        references = _pointer_references(handle, reference_targets)
        for hit in hits:
            address = int(hit["address"])
            direct = references.get(address)
            hit["direct_references"] = (
                {**direct, "complete": True}
                if direct is not None
                else {
                    "count": 0,
                    "addresses": [],
                    "complete": False,
                }
            )
            for candidate in hit["candidate_objects"]:
                object_address = int(candidate["object_address"])
                candidate_references = references.get(object_address)
                candidate["references"] = (
                    {**candidate_references, "complete": True}
                    if candidate_references is not None
                    else {
                        "count": 0,
                        "addresses": [],
                        "complete": False,
                    }
                )
    finally:
        close_process(handle)
    return {
        "schema": "zuma-rl.pc-memory-int32-probe",
        "version": 1,
        "process_id": pid,
        "value": value,
        "scanned_regions": scanned_regions,
        "scanned_bytes": scanned_bytes,
        "hit_count": len(hits),
        "non_image_hit_count": sum(not hit["image"] for hit in hits),
        "reference_target_count": reference_target_count,
        "reference_target_limit": MAX_POINTER_REFERENCE_TARGETS,
        "reference_target_analyzed_count": len(reference_targets),
        "reference_target_truncated": (
            reference_target_count > len(reference_targets)
        ),
        "reference_target_selection": "ascending_address",
        "hits": hits,
    }


def _trace_args(
    *,
    project_root: Path,
    plan: Mapping[str, Any],
    dmo: Path,
    result_path: Path,
    detach_at_update: int | None = None,
    reattach_at_update: int | None = None,
    gameplay_mtrand_oracle: Path | None = None,
    gameplay_mtrand_oracle_seed: int | None = None,
    gameplay_mtrand_oracle_maximum_draws: int = 100_000,
    startup_priority_bias_until_update: int | None = None,
    process_affinity_mask: int | None = None,
    suspend_main_thread_on_stop: bool = False,
    disable_source_bound_board_anchor: bool = False,
    initial_global_mtrand_oracle: Path | None = None,
    initial_global_mtrand_seed: int | None = None,
    startup_global_mtrand_observation_oracle: Path | None = None,
    startup_global_mtrand_observation_seed: int | None = None,
    startup_global_mtrand_observation_end_at_update: int | None = None,
    startup_global_mtrand_observation_maximum_draws: int = 100_000,
    startup_global_mtrand_hidden_draw_start_after_source_order: (
        int | None
    ) = None,
    startup_global_mtrand_hidden_draw_stop_before_source_order: (
        int | None
    ) = None,
    diagnostic_startup_crt_seed: int | None = None,
) -> list[str]:
    if (gameplay_mtrand_oracle is None) != (
        gameplay_mtrand_oracle_seed is None
    ):
        raise ProbeError("gameplay_mtrand_oracle_options_incomplete")
    if gameplay_mtrand_oracle_maximum_draws <= 0:
        raise ProbeError("gameplay_mtrand_oracle_maximum_draws_invalid")
    if (initial_global_mtrand_oracle is None) != (
        initial_global_mtrand_seed is None
    ):
        raise ProbeError("initial_global_mtrand_options_incomplete")
    startup_observation_values = (
        startup_global_mtrand_observation_oracle,
        startup_global_mtrand_observation_seed,
        startup_global_mtrand_observation_end_at_update,
    )
    startup_observation_requested = any(
        value is not None for value in startup_observation_values
    )
    if startup_observation_requested and any(
        value is None for value in startup_observation_values
    ):
        raise ProbeError(
            "startup_global_mtrand_observation_options_incomplete"
        )
    if startup_global_mtrand_observation_maximum_draws <= 0:
        raise ProbeError(
            "startup_global_mtrand_observation_maximum_draws_invalid"
        )
    if startup_observation_requested and (
        gameplay_mtrand_oracle is not None
        or initial_global_mtrand_oracle is not None
    ):
        raise ProbeError(
            "startup_global_mtrand_observation_breakpoint_conflict"
        )
    hidden_draw_values = (
        startup_global_mtrand_hidden_draw_start_after_source_order,
        startup_global_mtrand_hidden_draw_stop_before_source_order,
    )
    hidden_draw_requested = any(
        value is not None for value in hidden_draw_values
    )
    if hidden_draw_requested and (
        any(value is None for value in hidden_draw_values)
        or not startup_observation_requested
    ):
        raise ProbeError(
            "startup_global_mtrand_hidden_draw_options_incomplete"
        )
    if diagnostic_startup_crt_seed is not None and (
        isinstance(diagnostic_startup_crt_seed, bool)
        or not isinstance(diagnostic_startup_crt_seed, int)
        or not 0 <= diagnostic_startup_crt_seed <= 0xFFFFFFFF
    ):
        raise ProbeError("diagnostic_startup_crt_seed_invalid")
    runtime = plan["runtime"]
    trace = plan["trace"]
    launch_mode = runtime.get(
        "launch_mode",
        DIRECT_FIXED_SEED_LAUNCH_MODE,
    )
    if (
        diagnostic_startup_crt_seed is not None
        and launch_mode != DIRECT_NATURAL_STRICT_BROKER_LAUNCH_MODE
    ):
        raise ProbeError(
            "diagnostic_startup_crt_seed_requires_natural_strict_plan"
        )
    effective_detach_update = (
        trace.get("detach_at_update")
        if detach_at_update is None
        else detach_at_update
    )
    priority_bias_boundary = (
        trace.get("natural_command_broker_stop_after_update")
        if (
            launch_mode == DIRECT_NATURAL_STRICT_BROKER_LAUNCH_MODE
            and detach_at_update is None
        )
        else effective_detach_update
    )
    if startup_priority_bias_until_update is not None:
        if (
            isinstance(priority_bias_boundary, bool)
            or not isinstance(priority_bias_boundary, int)
            or not (
                0
                < startup_priority_bias_until_update
                < priority_bias_boundary
            )
        ):
            raise ProbeError("startup_priority_bias_update_invalid")
    if process_affinity_mask is not None and process_affinity_mask <= 0:
        raise ProbeError("process_affinity_mask_invalid")
    if launch_mode in {
        STEAM_LAUNCH_MODE,
        DIRECT_NATURAL_SEED_LAUNCH_MODE,
    }:
        natural_seed_controls = (
            runtime.get("crt_rand_seed"),
            runtime.get("board_seed"),
            runtime.get("global_rng_seed"),
            runtime.get("thread_crt_rng_seed"),
            runtime.get("startup_seed_transport"),
        )
        natural_trace_controls = (
            gameplay_mtrand_oracle,
            gameplay_mtrand_oracle_seed,
            initial_global_mtrand_oracle,
            initial_global_mtrand_seed,
            startup_global_mtrand_observation_oracle,
            startup_global_mtrand_observation_seed,
            startup_global_mtrand_observation_end_at_update,
            startup_priority_bias_until_update,
            process_affinity_mask,
            trace.get("source_bound_board_anchor"),
            trace.get("source_bound_board_precall_global_restore"),
        )
        if (
            any(value is not None for value in natural_seed_controls)
            or any(value is not None for value in natural_trace_controls)
            or suspend_main_thread_on_stop
            or disable_source_bound_board_anchor
            or trace.get("seed_board_before_attach", False)
            or trace.get("startup_trace_handoff", False)
            or trace.get(
                "allow_source_bound_board_global_correction",
                False,
            )
        ):
            raise ProbeError("natural_launch_control_invalid")
        args = [
            sys.executable,
            str(project_root / "tools" / "launch_popcap_replay.py"),
            "--dmo",
            str(dmo),
            "--demo-length",
            str(plan["dmo"]["length_updates"]),
            "--minus-count",
            "0",
            "--runtime-exe",
            str(runtime["runtime_executable"]),
            "--timeout",
            str(trace["launch_timeout_seconds"]),
            "--monitor-until-exit",
        ]
        if launch_mode == STEAM_LAUNCH_MODE:
            steam_executable = runtime.get("steam_executable")
            if (
                not isinstance(steam_executable, str)
                or not steam_executable
                or runtime.get("changedir") is not None
            ):
                raise ProbeError("steam_launch_runtime_invalid")
            args.extend(["--steam-exe", steam_executable])
        else:
            changedir = runtime.get("changedir")
            if not isinstance(changedir, str) or not changedir:
                raise ProbeError("direct_natural_launch_runtime_invalid")
            args.extend(
                [
                    "--direct-runtime-exe",
                    str(runtime["runtime_executable"]),
                    "--changedir",
                    changedir,
                ]
            )
        return args
    if launch_mode == DIRECT_NATURAL_STRICT_BROKER_LAUNCH_MODE:
        natural_seed_controls = (
            runtime.get("crt_rand_seed"),
            runtime.get("board_seed_call_address"),
            runtime.get("board_seed"),
            runtime.get("global_rng_seed"),
            runtime.get("thread_crt_rng_seed"),
            runtime.get("startup_seed_transport"),
        )
        stop_after_update = trace.get(
            "natural_command_broker_stop_after_update"
        )
        changedir = runtime.get("changedir")
        forbidden_trace_controls = (
            gameplay_mtrand_oracle,
            gameplay_mtrand_oracle_seed,
            initial_global_mtrand_oracle,
            initial_global_mtrand_seed,
            startup_global_mtrand_observation_oracle,
            startup_global_mtrand_observation_seed,
            startup_global_mtrand_observation_end_at_update,
            trace.get("source_bound_board_anchor"),
            trace.get("source_bound_board_precall_global_restore"),
        )
        if (
            any(value is not None for value in natural_seed_controls)
            or any(
                value is not None for value in forbidden_trace_controls
            )
            or not isinstance(changedir, str)
            or not changedir
            or isinstance(stop_after_update, bool)
            or not isinstance(stop_after_update, int)
            or stop_after_update <= 0
            or trace.get("attach_at_update") != 0
            or trace.get("startup_trace_handoff") is not True
            or trace.get("allow_pre_stream_commands") is not True
            or trace.get(
                "allow_font_cache_manifest_completion_debt"
            ) is not True
            or not isinstance(
                trace.get("font_cache_manifest_receipt"), Mapping
            )
            or trace.get("seed_board_before_attach", False)
            or trace.get(
                "allow_source_bound_board_global_correction",
                False,
            )
            or suspend_main_thread_on_stop
            or disable_source_bound_board_anchor
        ):
            raise ProbeError("natural_strict_broker_control_invalid")
        args = [
            sys.executable,
            str(project_root / "tools" / "trace_popcap_demo_commands.py"),
            "--dmo",
            str(dmo),
            "--runtime-exe",
            str(runtime["runtime_executable"]),
            "--direct-runtime-exe",
            str(runtime["runtime_executable"]),
            "--changedir",
            changedir,
            "--attach-at-update",
            "0",
            "--maximum-hits",
            str(trace["maximum_hits"]),
            "--trace-timeout",
            str(trace["trace_timeout_seconds"]),
            "--attach-timeout",
            "300",
            "--launch-timeout",
            str(trace["launch_timeout_seconds"]),
            "--broker-service-blocks",
            "--service-wait-timeout",
            "5",
            "--quiet-nonservice",
            "--progress-every-updates",
            "1000",
            "--stop-after-update",
            str(stop_after_update),
            "--accept-normal-exit",
            "--allow-pre-stream-commands",
            "--startup-trace-handoff",
            "--allow-font-cache-manifest-completion-debt",
            "--result-json",
            str(result_path),
        ]
        if diagnostic_startup_crt_seed is None:
            args.append("--direct-natural-seed")
        else:
            args.extend(
                [
                    "--crt-rand-seed",
                    str(diagnostic_startup_crt_seed),
                    "--startup-seed-transport",
                    "debugger_register",
                ]
            )
        if startup_priority_bias_until_update is not None:
            args.extend(
                [
                    "--startup-priority-bias-until-update",
                    str(startup_priority_bias_until_update),
                ]
            )
        if process_affinity_mask is not None:
            args.extend(
                [
                    "--startup-process-affinity-mask",
                    str(process_affinity_mask),
                ]
            )
        for row_index in plan.get("dmo", {}).get(
            "diagnostic_successful_file_write_padding_rows",
            [],
        ):
            args.extend(
                ["--successful-file-write-padding-row", str(row_index)]
            )
        return args
    if launch_mode != DIRECT_FIXED_SEED_LAUNCH_MODE:
        raise ProbeError("launch_mode_invalid")
    source_bound_stop_after_update = trace.get(
        "source_bound_command_broker_stop_after_update"
    )
    source_bound_finite_trace = source_bound_stop_after_update is not None
    if source_bound_finite_trace:
        source_bound_anchor = trace.get("source_bound_board_anchor")
        source_bound_precall = trace.get(
            "source_bound_board_precall_global_restore"
        )
        provenance_path = trace.get(
            "source_bound_dmo_provenance_path"
        )
        if (
            isinstance(source_bound_stop_after_update, bool)
            or not isinstance(source_bound_stop_after_update, int)
            or source_bound_stop_after_update <= 0
            or detach_at_update is not None
            or reattach_at_update is not None
            or trace.get("attach_at_update") != 0
            or trace.get("detach_at_update") is not None
            or trace.get("reattach_at_update") is not None
            or not isinstance(source_bound_anchor, Mapping)
            or not isinstance(source_bound_precall, Mapping)
            or source_bound_stop_after_update
            <= int(source_bound_anchor.get("framework_update", -1))
            or not isinstance(provenance_path, str)
            or not provenance_path
            or trace.get("startup_trace_handoff") is not True
            or trace.get("allow_pre_stream_commands") is not True
            or trace.get(
                "allow_font_cache_manifest_completion_debt"
            ) is not True
            or not isinstance(
                trace.get("font_cache_manifest_receipt"), Mapping
            )
            or trace.get("seed_board_before_attach", False)
            or trace.get(
                "allow_source_bound_board_global_correction",
                False,
            )
            or trace.get("allow_blackout_file_write_order_rebase", False)
            or trace.get(
                "allow_post_blackout_attach_stabilization",
                False,
            )
            or trace.get(
                "allow_post_blackout_overdue_idle_reentry",
                False,
            )
            or suspend_main_thread_on_stop
            or disable_source_bound_board_anchor
        ):
            raise ProbeError("source_bound_finite_trace_control_invalid")
    args = [
        sys.executable,
        str(project_root / "tools" / "trace_popcap_demo_commands.py"),
        "--dmo",
        str(dmo),
        "--runtime-exe",
        str(runtime["runtime_executable"]),
    ]
    changedir = runtime.get("changedir")
    crt_rand_seed = runtime.get("crt_rand_seed")
    if (
        not isinstance(changedir, str)
        or not changedir
        or not isinstance(crt_rand_seed, int)
        or isinstance(crt_rand_seed, bool)
        or crt_rand_seed < 0
    ):
        raise ProbeError("direct_launch_runtime_invalid")
    args.extend(
        [
            "--direct-runtime-exe",
            str(runtime["runtime_executable"]),
            "--changedir",
            changedir,
            "--crt-rand-seed",
            str(crt_rand_seed),
        ]
    )
    args.extend(
        [
        "--attach-at-update",
        str(trace["attach_at_update"]),
        "--maximum-hits",
        str(trace["maximum_hits"]),
        "--trace-timeout",
        str(trace["trace_timeout_seconds"]),
        "--attach-timeout",
        "300",
        "--launch-timeout",
        str(trace["launch_timeout_seconds"]),
        "--broker-service-blocks",
        "--service-wait-timeout",
        "5",
        "--quiet-nonservice",
        "--progress-every-updates",
        "1000",
        "--accept-normal-exit",
        "--result-json",
        str(result_path),
        ]
    )
    if source_bound_finite_trace:
        args.extend(
            [
                "--stop-after-update",
                str(source_bound_stop_after_update),
            ]
        )
    else:
        args.extend(
            [
                "--detach-at-update",
                str(effective_detach_update),
                "--reattach-at-update",
                str(
                    trace["reattach_at_update"]
                    if reattach_at_update is None
                    else reattach_at_update
                ),
            ]
        )
    board_seed = runtime.get("board_seed")
    if board_seed is not None:
        args.extend(
            [
                "--board-seed-address",
                str(runtime["board_seed_call_address"]),
                "--board-seed",
                str(board_seed),
            ]
        )
    global_rng_seed = runtime.get("global_rng_seed")
    if global_rng_seed is not None:
        args.extend(["--global-rng-seed", str(global_rng_seed)])
    thread_crt_rng_seed = runtime.get("thread_crt_rng_seed")
    if thread_crt_rng_seed is not None:
        args.extend(
            ["--thread-crt-rng-seed", str(thread_crt_rng_seed)]
        )
    for row_index in plan.get("dmo", {}).get(
        "diagnostic_successful_file_write_padding_rows",
        [],
    ):
        args.extend(
            ["--successful-file-write-padding-row", str(row_index)]
        )
    if trace["allow_pre_stream_commands"]:
        args.append("--allow-pre-stream-commands")
    if trace.get("startup_trace_handoff", False):
        args.append("--startup-trace-handoff")
    if trace.get("allow_attach_stabilization", False):
        args.append("--allow-attach-stabilization")
    if trace.get("allow_pre_attach_file_write_debt", False):
        args.append("--allow-pre-attach-file-write-debt")
    if trace.get(
        "allow_font_cache_manifest_completion_debt",
        False,
    ):
        args.append("--allow-font-cache-manifest-completion-debt")
    if trace.get("seed_board_before_attach", False):
        args.append("--seed-board-before-attach")
    if trace.get(
        "allow_blackout_file_write_order_rebase",
        False,
    ):
        args.append("--allow-blackout-file-write-order-rebase")
    if trace.get(
        "allow_post_blackout_attach_stabilization",
        False,
    ):
        args.append("--allow-post-blackout-attach-stabilization")
    blackout_command_offset = trace.get(
        "blackout_expected_command_order_offset"
    )
    blackout_native_offset = trace.get(
        "blackout_expected_native_timeline_offset"
    )
    if blackout_command_offset is not None:
        args.extend(
            [
                "--blackout-expected-command-order-offset",
                str(blackout_command_offset),
                "--blackout-expected-native-timeline-offset",
                str(blackout_native_offset),
            ]
        )
    for offset_pair in trace.get("blackout_allowed_offset_pairs", []):
        args.extend(
            [
                "--blackout-allowed-offset-pair",
                (
                    f"{offset_pair['command_order_offset']}:"
                    f"{offset_pair['native_timeline_offset']}"
                ),
            ]
        )
    if trace.get(
        "allow_post_blackout_overdue_idle_reentry",
        False,
    ):
        args.append("--allow-post-blackout-overdue-idle-reentry")
    if trace.get("close_after_terminal_command", False):
        args.append("--close-after-terminal-command")
    startup_priority_bias = (
        trace.get("startup_priority_bias_until_update")
        if startup_priority_bias_until_update is None
        else startup_priority_bias_until_update
    )
    if startup_priority_bias is not None:
        args.extend(
            [
                "--startup-priority-bias-until-update",
                str(startup_priority_bias),
            ]
        )
    if process_affinity_mask is not None:
        args.extend(
            [
                "--startup-process-affinity-mask",
                str(process_affinity_mask),
            ]
        )
    if suspend_main_thread_on_stop:
        args.append("--suspend-main-thread-on-stop")
    startup_seed_transport = runtime.get("startup_seed_transport")
    if startup_seed_transport is not None:
        args.extend(
            [
                "--startup-seed-transport",
                str(startup_seed_transport),
            ]
        )
    source_bound_anchor = (
        None
        if disable_source_bound_board_anchor
        else trace.get("source_bound_board_anchor")
    )
    if source_bound_anchor is not None:
        args.extend(
            [
                "--source-bound-board-monitor",
                str(source_bound_anchor["monitor_path"]),
                "--source-bound-board-trace",
                str(source_bound_anchor["trace_path"]),
                "--source-bound-board-recording-report",
                str(source_bound_anchor["recording_report_path"]),
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
        if trace.get(
            "allow_source_bound_board_global_correction",
            False,
        ):
            args.append("--allow-source-bound-board-global-correction")
    if (
        not disable_source_bound_board_anchor
        and trace.get("source_bound_board_precall_global_restore") is not None
    ):
        if source_bound_anchor is None:
            raise ProbeError("source_bound_precall_requires_anchor")
        args.append("--source-bound-board-precall-global-restore")
    if gameplay_mtrand_oracle is not None:
        if gameplay_mtrand_oracle_seed is None:
            raise ProbeError("gameplay_mtrand_oracle_seed_missing")
        args.extend(
            [
                "--gameplay-mtrand-oracle",
                str(gameplay_mtrand_oracle),
                "--gameplay-mtrand-oracle-seed",
                str(gameplay_mtrand_oracle_seed),
                "--gameplay-mtrand-oracle-maximum-draws",
                str(gameplay_mtrand_oracle_maximum_draws),
                "--gameplay-mtrand-receipt-json",
                str(result_path.parent / "gameplay-mtrand-sync.json"),
            ]
        )
    if initial_global_mtrand_oracle is not None:
        if initial_global_mtrand_seed is None:
            raise ProbeError("initial_global_mtrand_seed_missing")
        args.extend(
            [
                "--initial-global-mtrand-oracle",
                str(initial_global_mtrand_oracle),
                "--initial-global-mtrand-seed",
                str(initial_global_mtrand_seed),
                "--initial-global-mtrand-receipt-json",
                str(
                    result_path.parent
                    / "initial-global-mtrand-seed.json"
                ),
            ]
        )
    if startup_global_mtrand_observation_oracle is not None:
        if (
            startup_global_mtrand_observation_seed is None
            or startup_global_mtrand_observation_end_at_update is None
        ):
            raise ProbeError(
                "startup_global_mtrand_observation_options_incomplete"
            )
        args.extend(
            [
                "--startup-global-mtrand-observation-oracle",
                str(startup_global_mtrand_observation_oracle),
                "--startup-global-mtrand-observation-seed",
                str(startup_global_mtrand_observation_seed),
                "--startup-global-mtrand-observation-end-at-update",
                str(startup_global_mtrand_observation_end_at_update),
                "--startup-global-mtrand-observation-maximum-draws",
                str(startup_global_mtrand_observation_maximum_draws),
                "--startup-global-mtrand-observation-receipt-json",
                str(
                    result_path.parent
                    / "startup-global-mtrand-observation.json"
                ),
            ]
        )
        if hidden_draw_requested:
            assert (
                startup_global_mtrand_hidden_draw_start_after_source_order
                is not None
            )
            assert (
                startup_global_mtrand_hidden_draw_stop_before_source_order
                is not None
            )
            args.extend(
                [
                    "--startup-global-mtrand-hidden-draw-"
                    "start-after-source-order",
                    str(
                        startup_global_mtrand_hidden_draw_start_after_source_order
                    ),
                    "--startup-global-mtrand-hidden-draw-"
                    "stop-before-source-order",
                    str(
                        startup_global_mtrand_hidden_draw_stop_before_source_order
                    ),
                ]
            )
    return args


def _natural_strict_trace_binding(
    payload: Mapping[str, Any],
    *,
    artifact_path: Path,
    plan: Mapping[str, Any],
    expected_pid: int,
    expected_dmo: Path,
    expected_runtime: Path,
    expected_startup_priority_bias_until_update: int | None = None,
    expected_startup_process_affinity_mask: int | None = None,
) -> dict[str, Any]:
    """Bind the zero-RNG-write natural command transport fail closed."""

    options = payload.get("options")
    result = payload.get("result")
    startup_rng = payload.get("startup_rng")
    source_dmo = payload.get("source_dmo")
    trace = plan.get("trace")
    if not all(
        isinstance(value, Mapping)
        for value in (options, result, startup_rng, source_dmo, trace)
    ):
        raise ProbeError("natural_strict_trace_contract_failed")
    assert isinstance(options, Mapping)
    assert isinstance(result, Mapping)
    assert isinstance(startup_rng, Mapping)
    assert isinstance(source_dmo, Mapping)
    assert isinstance(trace, Mapping)
    stop_after_update = trace.get(
        "natural_command_broker_stop_after_update"
    )
    manifest = trace.get("font_cache_manifest_receipt")
    observed_seed = startup_rng.get("observed_seed")
    effective_seed = startup_rng.get("effective_seed")
    required_true_options = (
        "direct_runtime",
        "direct_natural_seed",
        "broker_service_blocks",
        "allow_pre_stream_commands",
        "startup_trace_handoff",
        "allow_font_cache_manifest_completion_debt",
    )
    required_false_options = (
        "seed_board_before_attach",
        "source_bound_board_anchor",
        "source_bound_board_precall_global_restore",
        "allow_source_bound_board_global_correction",
        "allow_attach_stabilization",
        "allow_pre_attach_file_write_debt",
        "post_blackout_attach_stabilization",
        "allow_blackout_file_write_order_rebase",
        "allow_post_blackout_overdue_idle_reentry",
        "suspend_main_thread_on_stop",
    )
    required_null_options = (
        "crt_rand_seed",
        "startup_seed_transport",
        "board_seed",
        "global_rng_seed",
        "thread_crt_rng_seed",
        "detach_at_update",
        "reattach_at_update",
        "gameplay_mtrand_oracle",
        "gameplay_mtrand_oracle_seed",
        "initial_global_mtrand_oracle",
        "initial_global_mtrand_seed",
        "startup_global_mtrand_observation_oracle",
        "startup_global_mtrand_observation_seed",
        "global_mtrand_restore_log",
        "qrand_restore_log",
        "thread_crt_restore_log",
    )
    if (
        payload.get("schema") != "zuma.popcap_strict_replay.v3"
        or payload.get("runtime_process_id") != expected_pid
        or not isinstance(payload.get("runtime_executable"), str)
        or Path(str(payload["runtime_executable"])).resolve()
        != expected_runtime.resolve()
        or source_dmo.get("path") is None
        or Path(str(source_dmo["path"])).resolve()
        != expected_dmo.resolve()
        or source_dmo.get("sha256")
        != _sha256_path(expected_dmo).removeprefix("sha256:")
        or source_dmo.get("size_bytes") != expected_dmo.stat().st_size
        or any(options.get(name) is not True for name in required_true_options)
        or any(options.get(name, False) is not False for name in required_false_options)
        or any(options.get(name) is not None for name in required_null_options)
        or options.get("startup_priority_bias_until_update")
        != expected_startup_priority_bias_until_update
        or options.get("startup_process_affinity_mask")
        != expected_startup_process_affinity_mask
        or options.get("rng_seed_override_count") != 0
        or options.get("attach_at_update") != 0
        or options.get("stop_after_update") != stop_after_update
        or options.get("stop_after_command_order") is not None
        or options.get("diagnostic_successful_file_write_padding_rows")
        != []
        or startup_rng.get("mode")
        != "retail_natural_seed_observation"
        or startup_rng.get("process_id") != expected_pid
        or startup_rng.get("seed_source")
        != "retail_eax_before_push_to_srand"
        or startup_rng.get("register_override") is not None
        or startup_rng.get("rng_process_memory_writes") != 0
        or startup_rng.get("persistent_file_modified") is not False
        or startup_rng.get("main_thread_suspended_on_detach") is not True
        or startup_rng.get("main_thread_suspend_previous_count") != 0
        or startup_rng.get("original_instruction_hex") != "50"
        or startup_rng.get("compatibility_layer") != "HIGHDPIAWARE"
        or isinstance(observed_seed, bool)
        or not isinstance(observed_seed, int)
        or not 0 <= observed_seed <= 0xFFFFFFFF
        or effective_seed != observed_seed
        or result.get("failure_count") != 0
        or result.get("boundary_failures") != []
        or result.get("broker_failures") != []
        or result.get("stopped_at_update") is not True
        or result.get("stopped_at_command_order") is not False
        or result.get("last_update") != stop_after_update
        or not isinstance(manifest, Mapping)
        or result.get("font_cache_manifest_entry_count")
        != manifest.get("entry_count")
        or "sha256:" + str(result.get("font_cache_manifest_sha256"))
        != manifest.get("manifest_sha256")
        or "sha256:"
        + str(result.get("font_cache_manifest_main_pak_sha256"))
        != manifest.get("main_pak_sha256")
    ):
        raise ProbeError("natural_strict_trace_contract_failed")

    receipt_groups = {
        name: result.get(name)
        for name in (
            "startup_post_bypass_worker_yields",
            "direct_font_cache_file_write_observations",
            "deferred_file_write_discharges",
            "font_cache_manifest_completion_discharges",
            "terminal_file_write_payload_handoffs",
            "service_file_write_header_claims",
            "service_continuation_verifications",
            "preloading_failed_file_write_tail_short_header_recoveries",
        )
    }
    if any(not isinstance(value, list) for value in receipt_groups.values()):
        raise ProbeError("natural_strict_trace_receipts_invalid")
    worker_yields = receipt_groups["startup_post_bypass_worker_yields"]
    temporary_breakpoint_memory_writes = 0
    for receipt in worker_yields:
        commit = receipt.get("continuation_commit") if isinstance(
            receipt, Mapping
        ) else None
        write_count = receipt.get(
            "temporary_breakpoint_memory_writes"
        ) if isinstance(receipt, Mapping) else None
        if (
            not isinstance(receipt, Mapping)
            or receipt.get("process_id") != expected_pid
            or receipt.get("thread_id") != startup_rng.get("thread_id")
            or receipt.get("memory_writes") != 0
            or receipt.get("failure") is not None
            or receipt.get("probe_status") != "boundary"
            or receipt.get("command_breakpoint_rearmed") is not True
            or receipt.get("resume_count") != 1
            or isinstance(write_count, bool)
            or not isinstance(write_count, int)
            or write_count <= 0
            or not isinstance(commit, Mapping)
            or commit.get("mechanism")
            != "atomic_startup_worker_payload_boundary"
        ):
            raise ProbeError("natural_strict_worker_yield_invalid")
        temporary_breakpoint_memory_writes += write_count

    data = artifact_path.read_bytes()
    callback_context_emulation_count = sum(
        len(receipt_groups[name])
        for name in (
            "deferred_file_write_discharges",
            "font_cache_manifest_completion_discharges",
            "terminal_file_write_payload_handoffs",
            "service_file_write_header_claims",
        )
    )
    short_header_recoveries = receipt_groups[
        "preloading_failed_file_write_tail_short_header_recoveries"
    ]
    if len(short_header_recoveries) > 1:
        raise ProbeError(
            "natural_strict_tail_short_header_recovery_count_invalid"
        )
    recovery_write_count = 0
    recovery_write_bytes = 0
    for receipt in short_header_recoveries:
        expected_before = {
            "buffer_read_bit_position": 1888,
            "last_demo_update": 401,
            "needs_command": 0,
            "is_short": 1,
            "command_number": 1,
            "command_order": 68,
            "command_bit_position": 1882,
            "demo_loading_complete": 0,
        }
        expected_after = {
            "buffer_read_bit_position": 1860,
            "last_demo_update": 391,
            "needs_command": 1,
            "is_short": 0,
            "command_number": 16,
            "command_order": 67,
            "command_bit_position": 1849,
            "demo_loading_complete": 0,
        }
        if (
            not isinstance(receipt, Mapping)
            or set(receipt)
            != {
                "mechanism",
                "process_id",
                "thread_id",
                "framework_update",
                "false_header_row_index",
                "corridor_start_index",
                "corridor_end_index",
                "command_order_offset",
                "command_order_rebase_rows",
                "before",
                "after",
                "false_payload_executed",
                "write_count",
                "bytes_written",
            }
            or receipt.get("mechanism")
            != "audited_false_short_header_pre_payload_recovery"
            or receipt.get("process_id") != expected_pid
            or receipt.get("thread_id") != startup_rng.get("thread_id")
            or receipt.get("framework_update") != 401
            or receipt.get("false_header_row_index") != 74
            or receipt.get("corridor_start_index") != 45
            or receipt.get("corridor_end_index") != 71
            or receipt.get("command_order_offset") != 4
            or receipt.get("command_order_rebase_rows")
            != [64, 65, 68, 69]
            or receipt.get("before") != expected_before
            or receipt.get("after") != expected_after
            or receipt.get("false_payload_executed") is not False
            or receipt.get("write_count") != 8
            or receipt.get("bytes_written") != 23
        ):
            raise ProbeError(
                "natural_strict_tail_short_header_recovery_invalid"
            )
        recovery_write_count += 8
        recovery_write_bytes += 23
    return {
        "schema": NATURAL_STRICT_REPLAY_BINDING_SCHEMA,
        "version": NATURAL_STRICT_REPLAY_BINDING_VERSION,
        "status": "PASS",
        "artifact": artifact_path.name,
        "artifact_bytes": len(data),
        "artifact_sha256": _sha256_bytes(data),
        "runtime_process_id": expected_pid,
        "runtime_executable": str(expected_runtime),
        "runtime_executable_sha256": _sha256_path(expected_runtime),
        "source_dmo": str(expected_dmo),
        "source_dmo_sha256": _sha256_path(expected_dmo),
        "stop_after_update": stop_after_update,
        "failure_count": 0,
        "startup_seed": observed_seed,
        "seed_source": startup_rng["seed_source"],
        "register_override": None,
        "rng_seed_override_count": 0,
        "rng_process_memory_writes": 0,
        "font_cache_manifest": {
            "entry_count": manifest["entry_count"],
            "manifest_sha256": manifest["manifest_sha256"],
            "main_pak_sha256": manifest["main_pak_sha256"],
        },
        "transport": {
            "service_broker": True,
            "startup_worker_yield_count": len(worker_yields),
            "temporary_code_breakpoint_memory_write_count": (
                temporary_breakpoint_memory_writes
            ),
            "callback_context_emulation_count": (
                callback_context_emulation_count
            ),
            "replay_state_recovery_count": len(
                short_header_recoveries
            ),
            "replay_state_recovery_memory_write_count": (
                recovery_write_count
            ),
            "replay_state_recovery_memory_write_bytes": (
                recovery_write_bytes
            ),
            "gameplay_or_rng_process_memory_write_count": 0,
        },
    }


def _source_bound_artifact_descriptor(
    path_value: Any,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Resolve and hash one immutable source-bound replay artifact."""

    if not isinstance(path_value, str) or not path_value:
        raise ProbeError("source_bound_artifact_path_invalid")
    try:
        path = Path(path_value).resolve(strict=True)
        if not path.is_file():
            raise OSError(path)
        size = path.stat().st_size
        digest = _sha256_path(path)
    except OSError as error:
        raise ProbeError("source_bound_artifact_unavailable") from error
    if expected_sha256 is not None and digest != expected_sha256:
        raise ProbeError("source_bound_artifact_sha256_mismatch")
    return {
        "path": str(path),
        "bytes": size,
        "sha256": digest,
    }


def _validate_source_bound_global_natural_state(
    *,
    precall_row: Mapping[str, Any],
    global_row: Mapping[str, Any],
) -> None:
    """Reject any hidden global-MTRand correction around board creation."""

    if (
        precall_row.get("changed") is not False
        or precall_row.get("bytes_written") != 0
        or precall_row.get("process_memory_mutation") is not False
    ):
        raise ProbeError(
            "source_bound_global_precall_natural_state_mismatch"
        )
    if (
        global_row.get("bounded_global_correction_applied") is not False
        or global_row.get("natural_post_state_match") is not True
        or global_row.get("bytes_written") != 0
        or global_row.get("changed") is not False
    ):
        raise ProbeError(
            "source_bound_global_postcall_natural_state_mismatch"
        )


def _source_bound_original_trace_binding(
    payload: Mapping[str, Any],
    *,
    artifact_path: Path,
    plan_path: Path,
    plan: Mapping[str, Any],
    expected_pid: int,
    expected_dmo: Path,
    expected_runtime: Path,
) -> dict[str, Any]:
    """Bind a finite replay to one unmodified natural retail recording.

    This transport is intentionally distinct from the zero-write natural
    strict replay.  It accepts only the source recording's exact constructor
    RNG state: no global-state correction, one disclosed four-byte thread-CRT
    restore, and no mutation after the board-construction boundary.
    """

    runtime = plan.get("runtime")
    trace = plan.get("trace")
    options = payload.get("options")
    result = payload.get("result")
    startup_rng = payload.get("startup_rng")
    source_dmo = payload.get("source_dmo")
    source_receipt = payload.get("source_bound_board_anchor")
    board_rng = payload.get("board_rng")
    if not all(
        isinstance(value, Mapping)
        for value in (
            runtime,
            trace,
            options,
            result,
            startup_rng,
            source_dmo,
            source_receipt,
            board_rng,
        )
    ):
        raise ProbeError("source_bound_trace_contract_failed")
    assert isinstance(runtime, Mapping)
    assert isinstance(trace, Mapping)
    assert isinstance(options, Mapping)
    assert isinstance(result, Mapping)
    assert isinstance(startup_rng, Mapping)
    assert isinstance(source_dmo, Mapping)
    assert isinstance(source_receipt, Mapping)

    stop_after_update = trace.get(
        "source_bound_command_broker_stop_after_update"
    )
    anchor = trace.get("source_bound_board_anchor")
    precall = trace.get("source_bound_board_precall_global_restore")
    manifest = trace.get("font_cache_manifest_receipt")
    observations = source_receipt.get("observations")
    if (
        not isinstance(anchor, Mapping)
        or not isinstance(precall, Mapping)
        or not isinstance(manifest, Mapping)
        or not isinstance(observations, list)
        or len(observations) != 1
        or not isinstance(observations[0], Mapping)
    ):
        raise ProbeError("source_bound_trace_receipt_invalid")
    source_row = observations[0]
    global_row = source_row.get("global")
    crt_row = source_row.get("thread_crt")
    precall_row = source_row.get("global_precall_restore")
    if not all(
        isinstance(value, Mapping)
        for value in (global_row, crt_row, precall_row)
    ):
        raise ProbeError("source_bound_trace_receipt_invalid")
    assert isinstance(global_row, Mapping)
    assert isinstance(crt_row, Mapping)
    assert isinstance(precall_row, Mapping)

    _validate_source_bound_global_natural_state(
        precall_row=precall_row,
        global_row=global_row,
    )

    expected_startup_seed = runtime.get("crt_rand_seed")
    expected_board_seed = runtime.get("board_seed")
    expected_board_call = runtime.get("board_seed_call_address")
    required_false_options = (
        "direct_natural_seed",
        "seed_board_before_attach",
        "allow_source_bound_board_global_correction",
        "allow_attach_stabilization",
        "allow_pre_attach_file_write_debt",
        "post_blackout_attach_stabilization",
        "allow_blackout_file_write_order_rebase",
        "allow_post_blackout_overdue_idle_reentry",
        "suspend_main_thread_on_stop",
    )
    required_null_options = (
        "global_rng_seed",
        "thread_crt_rng_seed",
        "detach_at_update",
        "reattach_at_update",
        "gameplay_mtrand_oracle",
        "gameplay_mtrand_oracle_seed",
        "initial_global_mtrand_oracle",
        "initial_global_mtrand_seed",
        "startup_global_mtrand_observation_oracle",
        "startup_global_mtrand_observation_seed",
        "global_mtrand_restore_log",
        "qrand_restore_log",
        "thread_crt_restore_log",
    )
    if (
        payload.get("schema") != "zuma.popcap_strict_replay.v3"
        or payload.get("runtime_process_id") != expected_pid
        or Path(str(payload.get("runtime_executable"))).resolve()
        != expected_runtime.resolve()
        or Path(str(source_dmo.get("path"))).resolve()
        != expected_dmo.resolve()
        or source_dmo.get("sha256")
        != _sha256_path(expected_dmo).removeprefix("sha256:")
        or source_dmo.get("size_bytes") != expected_dmo.stat().st_size
        or options.get("direct_runtime") is not True
        or options.get("broker_service_blocks") is not True
        or options.get("allow_pre_stream_commands") is not True
        or options.get("startup_trace_handoff") is not True
        or options.get("allow_font_cache_manifest_completion_debt") is not True
        or options.get("attach_at_update") != 0
        or options.get("stop_after_update") != stop_after_update
        or options.get("stop_after_command_order") is not None
        or options.get("crt_rand_seed") != expected_startup_seed
        or options.get("startup_seed_transport") != "debugger_register"
        or options.get("board_seed") != expected_board_seed
        or options.get("board_seed_address") != expected_board_call
        or options.get("source_bound_board_anchor") is not True
        or options.get("source_bound_board_precall_global_restore") is not True
        or any(options.get(name, False) is not False for name in required_false_options)
        or any(options.get(name) is not None for name in required_null_options)
        or startup_rng.get("process_id") != expected_pid
        or startup_rng.get("seed") != expected_startup_seed
        or startup_rng.get("register_override")
        != "eax_before_push_to_srand"
        or startup_rng.get("persistent_file_modified") is not False
        or startup_rng.get("main_thread_suspended_on_detach") is not True
        or startup_rng.get("main_thread_suspend_previous_count") != 0
        or result.get("failure_count") != 0
        or result.get("boundary_failures") != []
        or result.get("broker_failures") != []
        or result.get("stopped_at_update") is not True
        or result.get("stopped_at_command_order") is not False
        or result.get("last_update") != stop_after_update
        or result.get("font_cache_manifest_entry_count")
        != manifest.get("entry_count")
        or "sha256:" + str(result.get("font_cache_manifest_sha256"))
        != manifest.get("manifest_sha256")
        or "sha256:"
        + str(result.get("font_cache_manifest_main_pak_sha256"))
        != manifest.get("main_pak_sha256")
        or source_receipt.get("persistent_file_modified") is not False
        or source_receipt.get("process_memory_mutation") is not True
        or source_row.get("classification")
        != "source-bound-natural-retail-post-call-board-anchor"
        or source_row.get("process_id") != expected_pid
        or source_row.get("framework_update") != anchor.get("framework_update")
        or source_row.get("effective_seed") != expected_board_seed
        or source_row.get("observed_seed") != expected_board_seed
        or source_row.get("observed_seed_matches_source") is not True
        or source_row.get("bytes_written") != 4
        or source_row.get("process_memory_mutation") is not True
        or global_row.get("global_correction_authorized") is not False
        or precall_row.get("classification")
        != "source-bound-natural-retail-global-precall-restore"
        or precall_row.get("framework_update") != precall.get("framework_update")
        or crt_row.get("restored_state") != anchor.get("thread_crt_state")
        or crt_row.get("bytes_written") != 4
        or crt_row.get("changed") is not True
    ):
        raise ProbeError("source_bound_trace_contract_failed")

    plan_descriptor = _source_bound_artifact_descriptor(str(plan_path))
    provenance_descriptor = _source_bound_artifact_descriptor(
        trace.get("source_bound_dmo_provenance_path")
    )
    raw_dmo_descriptor = _source_bound_artifact_descriptor(
        anchor.get("source_dmo_path"),
        expected_sha256=str(anchor.get("source_dmo_sha256")),
    )
    recording_descriptor = _source_bound_artifact_descriptor(
        anchor.get("recording_report_path"),
        expected_sha256=str(anchor.get("recording_report_sha256")),
    )
    monitor_descriptor = _source_bound_artifact_descriptor(
        anchor.get("monitor_path"),
        expected_sha256=str(anchor.get("monitor_sha256")),
    )
    global_trace_descriptor = _source_bound_artifact_descriptor(
        anchor.get("trace_path"),
        expected_sha256=str(anchor.get("trace_sha256")),
    )
    try:
        provenance_summary = read_certifying_provenance(
            source_path=raw_dmo_descriptor["path"],
            output_path=expected_dmo,
            recording_report_path=recording_descriptor["path"],
            collector_plan_path=plan_path,
            provenance_path=provenance_descriptor["path"],
            expected_runtime_sha256=_sha256_path(expected_runtime),
        )
    except RetailDmoProvenanceError as error:
        raise ProbeError("source_bound_dmo_provenance_invalid") from error
    if (
        provenance_summary.get("source_dmo_sha256")
        != raw_dmo_descriptor["sha256"]
        or provenance_summary.get("playback_dmo_sha256")
        != _sha256_path(expected_dmo)
        or provenance_summary.get("recording_natural_outcome") is not True
        or provenance_summary.get("recording_process_memory_writes") != 0
    ):
        raise ProbeError("source_bound_dmo_provenance_invalid")

    try:
        recording_report = json.loads(
            Path(recording_descriptor["path"]).read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProbeError("source_bound_recording_report_invalid") from error
    gameplay = recording_report.get("gameplay")
    monitor = recording_report.get("rng_monitor")
    global_trace = recording_report.get("gameplay_mtrand_trace")
    main_thread = recording_report.get("main_thread")
    try:
        recording_outcome = certifying_recording_outcome(
            recording_report
        )
    except RetailDmoProvenanceError as error:
        raise ProbeError(
            "source_bound_recording_report_invalid"
        ) from error
    if (
        not isinstance(gameplay, Mapping)
        or not isinstance(monitor, Mapping)
        or not isinstance(global_trace, Mapping)
        or not isinstance(main_thread, Mapping)
        or recording_report.get("process_id") != precall.get("source_process_id")
        or gameplay.get("outcome") != recording_outcome
        or provenance_summary.get("recording_outcome")
        != recording_outcome
        or monitor.get("output") != monitor_descriptor["path"]
        or monitor.get("output_sha256") != monitor_descriptor["sha256"]
        or monitor.get("process_memory_writes") != 0
        or global_trace.get("output") != global_trace_descriptor["path"]
        or global_trace.get("output_sha256") != global_trace_descriptor["sha256"]
        or global_trace.get("process_memory_writes") != 0
        or main_thread.get("process_memory_writes") != 0
    ):
        raise ProbeError("source_bound_recording_report_invalid")

    worker_yields = result.get("startup_post_bypass_worker_yields")
    if not isinstance(worker_yields, list):
        raise ProbeError("source_bound_trace_receipts_invalid")
    temporary_writes = sum(
        int(row.get("temporary_breakpoint_memory_writes", 0))
        for row in worker_yields
        if isinstance(row, Mapping)
    )
    callback_count = 0
    for name in (
        "deferred_file_write_discharges",
        "font_cache_manifest_completion_discharges",
        "terminal_file_write_payload_handoffs",
        "service_file_write_header_claims",
    ):
        rows = result.get(name)
        if not isinstance(rows, list):
            raise ProbeError("source_bound_trace_receipts_invalid")
        callback_count += len(rows)

    artifact_data = artifact_path.read_bytes()
    outcome_binding = recording_report.get("version") != 1
    return {
        "schema": SOURCE_BOUND_STRICT_REPLAY_BINDING_SCHEMA,
        "version": (
            OUTCOME_SOURCE_BOUND_STRICT_REPLAY_BINDING_VERSION
            if outcome_binding
            else SOURCE_BOUND_STRICT_REPLAY_BINDING_VERSION
        ),
        "status": "PASS",
        "classification": (
            "source_bound_original_natural_retail_outcome"
            if outcome_binding
            else "source_bound_original_natural_retail_recording"
        ),
        "artifact": artifact_path.name,
        "artifact_bytes": len(artifact_data),
        "artifact_sha256": _sha256_bytes(artifact_data),
        "runtime_process_id": expected_pid,
        "runtime_executable": str(expected_runtime),
        "runtime_executable_sha256": _sha256_path(expected_runtime),
        "source_dmo": str(expected_dmo),
        "source_dmo_sha256": _sha256_path(expected_dmo),
        "stop_after_update": stop_after_update,
        "failure_count": 0,
        "collector_plan": plan_descriptor,
        "retail_dmo_provenance": provenance_descriptor,
        "source_recording": {
            "report": recording_descriptor,
            "raw_dmo": raw_dmo_descriptor,
            "rng_monitor": monitor_descriptor,
            "global_mtrand_trace": global_trace_descriptor,
            "process_id": recording_report["process_id"],
            "outcome": recording_outcome,
            "process_memory_writes": 0,
            "normal_exit": True,
            "host_restored_exactly": True,
        },
        "normalization": {
            "startup_seed": expected_startup_seed,
            "startup_seed_transport": "debugger_register",
            "board_seed": expected_board_seed,
            "board_seed_call_address": expected_board_call,
            "source_bound_framework_update": anchor["framework_update"],
            "global_precall_state_sha256": precall["global_state"][
                "state_sha256"
            ],
            "global_precall_memory_write_bytes": 0,
            "global_postcall_memory_write_bytes": 0,
            "thread_crt_state": anchor["thread_crt_state"],
            "thread_crt_memory_write_bytes": 4,
            "gameplay_or_rng_process_memory_write_count": 1,
            "gameplay_or_rng_process_memory_write_bytes": 4,
            "last_mutation_update": anchor["framework_update"],
            "persistent_file_modified": False,
        },
        "font_cache_manifest": {
            "entry_count": manifest["entry_count"],
            "manifest_sha256": manifest["manifest_sha256"],
            "main_pak_sha256": manifest["main_pak_sha256"],
        },
        "transport": {
            "service_broker": True,
            "startup_worker_yield_count": len(worker_yields),
            "temporary_code_breakpoint_memory_write_count": temporary_writes,
            "callback_context_emulation_count": callback_count,
            "post_board_construction_process_memory_write_count": 0,
        },
    }


def _diagnostic_fixed_startup_strict_trace_binding(
    payload: Mapping[str, Any],
    *,
    artifact_path: Path,
    plan: Mapping[str, Any],
    expected_pid: int,
    expected_dmo: Path,
    expected_runtime: Path,
    expected_startup_seed: int,
    expected_startup_priority_bias_until_update: int | None = None,
    expected_startup_process_affinity_mask: int | None = None,
) -> dict[str, Any]:
    """Bind a disclosed startup-register override as diagnostic evidence."""

    options = payload.get("options")
    result = payload.get("result")
    startup_rng = payload.get("startup_rng")
    source_dmo = payload.get("source_dmo")
    trace = plan.get("trace")
    if not all(
        isinstance(value, Mapping)
        for value in (options, result, startup_rng, source_dmo, trace)
    ):
        raise ProbeError("diagnostic_fixed_startup_trace_contract_failed")
    assert isinstance(options, Mapping)
    assert isinstance(result, Mapping)
    assert isinstance(startup_rng, Mapping)
    assert isinstance(source_dmo, Mapping)
    assert isinstance(trace, Mapping)
    stop_after_update = trace.get(
        "natural_command_broker_stop_after_update"
    )
    manifest = trace.get("font_cache_manifest_receipt")
    required_true_options = (
        "direct_runtime",
        "broker_service_blocks",
        "allow_pre_stream_commands",
        "startup_trace_handoff",
        "allow_font_cache_manifest_completion_debt",
    )
    required_false_options = (
        "direct_natural_seed",
        "seed_board_before_attach",
        "source_bound_board_anchor",
        "source_bound_board_precall_global_restore",
        "allow_source_bound_board_global_correction",
        "allow_attach_stabilization",
        "allow_pre_attach_file_write_debt",
        "post_blackout_attach_stabilization",
        "allow_blackout_file_write_order_rebase",
        "allow_post_blackout_overdue_idle_reentry",
        "suspend_main_thread_on_stop",
    )
    required_null_options = (
        "board_seed",
        "global_rng_seed",
        "thread_crt_rng_seed",
        "detach_at_update",
        "reattach_at_update",
        "gameplay_mtrand_oracle",
        "gameplay_mtrand_oracle_seed",
        "initial_global_mtrand_oracle",
        "initial_global_mtrand_seed",
        "startup_global_mtrand_observation_oracle",
        "startup_global_mtrand_observation_seed",
        "global_mtrand_restore_log",
        "qrand_restore_log",
        "thread_crt_restore_log",
    )
    if (
        payload.get("schema") != "zuma.popcap_strict_replay.v3"
        or payload.get("runtime_process_id") != expected_pid
        or not isinstance(payload.get("runtime_executable"), str)
        or Path(str(payload["runtime_executable"])).resolve()
        != expected_runtime.resolve()
        or source_dmo.get("path") is None
        or Path(str(source_dmo["path"])).resolve()
        != expected_dmo.resolve()
        or source_dmo.get("sha256")
        != _sha256_path(expected_dmo).removeprefix("sha256:")
        or source_dmo.get("size_bytes") != expected_dmo.stat().st_size
        or any(options.get(name) is not True for name in required_true_options)
        or any(options.get(name, False) is not False for name in required_false_options)
        or any(options.get(name) is not None for name in required_null_options)
        or options.get("crt_rand_seed") != expected_startup_seed
        or options.get("startup_seed_transport") != "debugger_register"
        or options.get("rng_seed_override_count") is not None
        or options.get("startup_priority_bias_until_update")
        != expected_startup_priority_bias_until_update
        or options.get("startup_process_affinity_mask")
        != expected_startup_process_affinity_mask
        or options.get("attach_at_update") != 0
        or options.get("stop_after_update") != stop_after_update
        or options.get("stop_after_command_order") is not None
        or options.get("diagnostic_successful_file_write_padding_rows")
        != []
        or startup_rng.get("process_id") != expected_pid
        or startup_rng.get("seed") != expected_startup_seed
        or startup_rng.get("register_override")
        != "eax_before_push_to_srand"
        or startup_rng.get("persistent_file_modified") is not False
        or startup_rng.get("main_thread_suspended_on_detach") is not True
        or startup_rng.get("main_thread_suspend_previous_count") != 0
        or startup_rng.get("original_instruction_hex") != "50"
        or startup_rng.get("compatibility_layer") != "HIGHDPIAWARE"
        or result.get("failure_count") != 0
        or result.get("boundary_failures") != []
        or result.get("broker_failures") != []
        or result.get("stopped_at_update") is not True
        or result.get("stopped_at_command_order") is not False
        or result.get("last_update") != stop_after_update
        or not isinstance(manifest, Mapping)
        or result.get("font_cache_manifest_entry_count")
        != manifest.get("entry_count")
        or "sha256:" + str(result.get("font_cache_manifest_sha256"))
        != manifest.get("manifest_sha256")
        or "sha256:"
        + str(result.get("font_cache_manifest_main_pak_sha256"))
        != manifest.get("main_pak_sha256")
    ):
        raise ProbeError("diagnostic_fixed_startup_trace_contract_failed")

    data = artifact_path.read_bytes()
    return {
        "schema": "zuma-rl.pc-diagnostic-fixed-startup-command-replay-binding",
        "version": 1,
        "status": "PASS",
        "classification": (
            "diagnostic_process_context_mutation_not_pc_evidence"
        ),
        "artifact": artifact_path.name,
        "artifact_bytes": len(data),
        "artifact_sha256": _sha256_bytes(data),
        "runtime_process_id": expected_pid,
        "runtime_executable": str(expected_runtime),
        "runtime_executable_sha256": _sha256_path(expected_runtime),
        "source_dmo": str(expected_dmo),
        "source_dmo_sha256": _sha256_path(expected_dmo),
        "stop_after_update": stop_after_update,
        "startup_seed": expected_startup_seed,
        "seed_source": "formal_c111_retail_eax_before_push_to_srand",
        "register_override": "eax_before_push_to_srand",
        "rng_seed_override_count": 1,
        "rng_process_memory_writes": 0,
        "process_context_mutation": True,
        "persistent_file_modified": False,
        "font_cache_manifest": {
            "entry_count": manifest["entry_count"],
            "manifest_sha256": manifest["manifest_sha256"],
            "main_pak_sha256": manifest["main_pak_sha256"],
        },
        "formal_evidence_eligible": False,
    }


def _finish_natural_strict_trace(
    *,
    process: subprocess.Popen[bytes],
    result_path: Path,
    plan: Mapping[str, Any],
    expected_pid: int,
    expected_dmo: Path,
    expected_runtime: Path,
    expected_startup_priority_bias_until_update: int | None = None,
    expected_startup_process_affinity_mask: int | None = None,
    expected_diagnostic_startup_seed: int | None = None,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    """Wait for, validate, and hash-bind the finite natural replay trace."""

    if timeout_seconds <= 0:
        raise ProbeError("natural_strict_trace_timeout_invalid")
    try:
        return_code = process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as error:
        raise ProbeError("natural_strict_trace_stop_timeout") from error
    if return_code != 0 or not result_path.is_file():
        raise ProbeError("natural_strict_trace_process_failed")
    trace = plan.get("trace")
    dmo = plan.get("dmo")
    if not isinstance(trace, Mapping) or not isinstance(dmo, Mapping):
        raise ProbeError("natural_strict_trace_plan_invalid")
    try:
        payload = _validate_trace_result(
            result_path,
            expected_pid=expected_pid,
            expected_dmo_sha256=str(dmo.get("sha256")),
            expected_attach_at_update=0,
            expected_allow_pre_stream_commands=True,
            expected_startup_trace_handoff=True,
            expected_allow_attach_stabilization=False,
            expected_allow_pre_attach_file_write_debt=False,
            expected_allow_font_cache_manifest_completion_debt=True,
            expected_font_cache_manifest_receipt=trace.get(
                "font_cache_manifest_receipt"
            ),
            expected_seed_board_before_attach=False,
            expected_startup_priority_bias_until_update=(
                expected_startup_priority_bias_until_update
            ),
            expected_startup_process_affinity_mask=(
                expected_startup_process_affinity_mask
            ),
            expected_crt_rand_seed=expected_diagnostic_startup_seed,
            expected_startup_seed_transport=(
                "debugger_register"
                if expected_diagnostic_startup_seed is not None
                else None
            ),
            expected_diagnostic_successful_file_write_padding_rows=[],
            expected_stop_after_update=trace.get(
                "natural_command_broker_stop_after_update"
            ),
        )
    except (CollectionError, OSError, ValueError) as error:
        raise ProbeError("natural_strict_trace_result_invalid") from error
    if expected_diagnostic_startup_seed is not None:
        return _diagnostic_fixed_startup_strict_trace_binding(
            payload,
            artifact_path=result_path,
            plan=plan,
            expected_pid=expected_pid,
            expected_dmo=expected_dmo,
            expected_runtime=expected_runtime,
            expected_startup_seed=expected_diagnostic_startup_seed,
            expected_startup_priority_bias_until_update=(
                expected_startup_priority_bias_until_update
            ),
            expected_startup_process_affinity_mask=(
                expected_startup_process_affinity_mask
            ),
        )
    return _natural_strict_trace_binding(
        payload,
        artifact_path=result_path,
        plan=plan,
        expected_pid=expected_pid,
        expected_dmo=expected_dmo,
        expected_runtime=expected_runtime,
        expected_startup_priority_bias_until_update=(
            expected_startup_priority_bias_until_update
        ),
        expected_startup_process_affinity_mask=(
            expected_startup_process_affinity_mask
        ),
    )


def _finish_source_bound_strict_trace(
    *,
    process: subprocess.Popen[bytes],
    result_path: Path,
    plan_path: Path,
    plan: Mapping[str, Any],
    expected_pid: int,
    expected_dmo: Path,
    expected_runtime: Path,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    """Finish and bind a finite source-recording-normalized replay trace."""

    if timeout_seconds <= 0:
        raise ProbeError("source_bound_strict_trace_timeout_invalid")
    try:
        return_code = process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as error:
        raise ProbeError("source_bound_strict_trace_stop_timeout") from error
    if return_code != 0 or not result_path.is_file():
        raise ProbeError("source_bound_strict_trace_process_failed")
    trace = plan.get("trace")
    runtime = plan.get("runtime")
    dmo = plan.get("dmo")
    if not all(
        isinstance(value, Mapping) for value in (trace, runtime, dmo)
    ):
        raise ProbeError("source_bound_strict_trace_plan_invalid")
    assert isinstance(trace, Mapping)
    assert isinstance(runtime, Mapping)
    assert isinstance(dmo, Mapping)
    try:
        payload = _validate_trace_result(
            result_path,
            expected_pid=expected_pid,
            expected_dmo_sha256=str(dmo.get("sha256")),
            expected_attach_at_update=0,
            expected_allow_pre_stream_commands=True,
            expected_startup_trace_handoff=True,
            expected_allow_attach_stabilization=False,
            expected_allow_pre_attach_file_write_debt=False,
            expected_allow_font_cache_manifest_completion_debt=True,
            expected_font_cache_manifest_receipt=trace.get(
                "font_cache_manifest_receipt"
            ),
            expected_seed_board_before_attach=False,
            expected_startup_priority_bias_until_update=trace.get(
                "startup_priority_bias_until_update"
            ),
            expected_crt_rand_seed=runtime.get("crt_rand_seed"),
            expected_startup_seed_transport=runtime.get(
                "startup_seed_transport"
            ),
            expected_board_seed_address=runtime.get(
                "board_seed_call_address"
            ),
            expected_board_seed=runtime.get("board_seed"),
            expected_global_rng_seed=runtime.get("global_rng_seed"),
            expected_thread_crt_rng_seed=runtime.get(
                "thread_crt_rng_seed"
            ),
            expected_source_bound_board_anchor=trace.get(
                "source_bound_board_anchor"
            ),
            expected_source_bound_board_precall_global_restore=trace.get(
                "source_bound_board_precall_global_restore"
            ),
            expected_allow_source_bound_board_global_correction=False,
            expected_diagnostic_successful_file_write_padding_rows=dmo.get(
                "diagnostic_successful_file_write_padding_rows",
                [],
            ),
            expected_stop_after_update=trace.get(
                "source_bound_command_broker_stop_after_update"
            ),
        )
    except (CollectionError, OSError, ValueError) as error:
        raise ProbeError("source_bound_strict_trace_result_invalid") from error
    return _source_bound_original_trace_binding(
        payload,
        artifact_path=result_path,
        plan_path=plan_path,
        plan=plan,
        expected_pid=expected_pid,
        expected_dmo=expected_dmo,
        expected_runtime=expected_runtime,
    )


def _load_gameplay_mtrand_sync_receipt(
    path: Path,
    *,
    expected_process_id: int,
    expected_oracle: Path,
    expected_dmo: Path,
) -> dict[str, Any]:
    """Validate and compact the pre-blackout synchronization sidecar."""

    try:
        data = path.read_bytes()
        payload = json.loads(data.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProbeError("gameplay_mtrand_sync_receipt_invalid") from error
    if not isinstance(payload, Mapping):
        raise ProbeError("gameplay_mtrand_sync_receipt_invalid")
    receipt = payload.get("receipt")
    observations = payload.get("observations")
    trace_result = payload.get("pre_blackout_trace_result")
    source_dmo = payload.get("source_dmo")
    if (
        payload.get("schema")
        != "zuma.popcap_gameplay_mtrand_sync_receipt.v1"
        or payload.get("status") != "PASS"
        or payload.get("runtime_process_id") != expected_process_id
        or payload.get("persistent_file_modified") is not False
        or payload.get("process_context_mutation") is not True
        or not isinstance(receipt, Mapping)
        or not isinstance(observations, list)
        or not isinstance(trace_result, Mapping)
        or not isinstance(source_dmo, Mapping)
    ):
        raise ProbeError("gameplay_mtrand_sync_receipt_contract_failed")
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
        or not isinstance(expected_hits, int)
        or isinstance(expected_hits, bool)
        or expected_hits <= 0
        or receipt.get("hit_count") != expected_hits
        or receipt.get("register_mutation_count") != expected_hits
        or len(observations) != expected_hits
        or trace_result.get("failure_count") != 0
        or trace_result.get("stopped_at_update") is not True
    ):
        raise ProbeError("gameplay_mtrand_sync_receipt_contract_failed")
    if any(
        not isinstance(row, Mapping) or row.get("order") != order
        for order, row in enumerate(observations)
    ):
        raise ProbeError("gameplay_mtrand_sync_observation_order_invalid")
    oracle_path = oracle.get("source_path")
    oracle_sha256 = oracle.get("source_sha256")
    if (
        not isinstance(oracle_path, str)
        or Path(oracle_path).resolve() != expected_oracle.resolve()
        or oracle_sha256 != _sha256_path(expected_oracle)
        or oracle.get("entry_count") != expected_hits
        or observations[0].get("framework_update")
        != oracle.get("first_update")
        or observations[-1].get("framework_update")
        != oracle.get("last_update")
        or source_dmo.get("sha256") != _sha256_path(expected_dmo)
    ):
        raise ProbeError("gameplay_mtrand_sync_provenance_mismatch")
    correction_count = receipt.get("state_correction_count")
    bytes_written = receipt.get("bytes_written_total")
    if (
        not isinstance(correction_count, int)
        or isinstance(correction_count, bool)
        or not 0 <= correction_count <= expected_hits
        or not isinstance(bytes_written, int)
        or isinstance(bytes_written, bool)
        or bytes_written != correction_count * 2500
    ):
        raise ProbeError("gameplay_mtrand_sync_mutation_count_invalid")
    return {
        "schema": "zuma-rl.pc-gameplay-mtrand-sync-artifact",
        "version": 1,
        "status": "PASS",
        "classification": payload.get("classification"),
        "artifact": path.name,
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


def _load_initial_global_mtrand_seed_receipt(
    path: Path,
    *,
    expected_process_id: int,
    expected_oracle: Path,
    expected_seed: int,
    expected_dmo: Path,
) -> dict[str, Any]:
    """Validate and compact the one-shot source-bound startup seed."""

    try:
        data = path.read_bytes()
        payload = json.loads(data.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProbeError(
            "initial_global_mtrand_seed_receipt_invalid"
        ) from error
    if not isinstance(payload, Mapping):
        raise ProbeError("initial_global_mtrand_seed_receipt_invalid")
    try:
        expected = load_initial_global_mtrand_call_oracle(
            expected_oracle,
            seed=expected_seed,
        )
    except (OSError, ValueError) as error:
        raise ProbeError(
            "initial_global_mtrand_seed_oracle_invalid"
        ) from error
    oracle = payload.get("oracle")
    observations = payload.get("observations")
    trace_result = payload.get("pre_blackout_trace_result")
    source_dmo = payload.get("source_dmo")
    if (
        payload.get("schema")
        != "zuma.popcap_initial_global_mtrand_seed_receipt.v1"
        or payload.get("status") != "PASS"
        or payload.get("classification")
        != "diagnostic_source_bound_rng_initialization"
        or payload.get("runtime_process_id") != expected_process_id
        or payload.get("process_context_mutation") is not True
        or payload.get("persistent_file_modified") is not False
        or not isinstance(oracle, Mapping)
        or not isinstance(observations, list)
        or len(observations) != 1
        or not isinstance(observations[0], Mapping)
        or not isinstance(trace_result, Mapping)
        or not isinstance(source_dmo, Mapping)
        or trace_result.get("failure_count") != 0
        or trace_result.get("stopped_at_update") is not True
    ):
        raise ProbeError(
            "initial_global_mtrand_seed_receipt_contract_failed"
        )
    expected_oracle_row = {
        "source_path": str(expected.source_path),
        "source_sha256": expected.source_sha256,
        "source_process_id": expected.source_process_id,
        "source_main_thread_id": expected.source_main_thread_id,
        "runtime_executable_sha256": (
            expected.runtime_executable_sha256
        ),
        "seed": expected.seed,
        "semantic_sha256": expected.semantic_sha256,
        "wrapper_address": expected.wrapper_address,
        "call_address": expected.call_address,
        "caller": expected.caller,
        "output": expected.output,
        "pre_index": expected.pre_index,
        "pre_state_sha256": expected.pre_state_sha256,
        "post_index": expected.post_index,
        "post_state_sha256": expected.post_state_sha256,
    }
    if dict(oracle) != expected_oracle_row:
        raise ProbeError(
            "initial_global_mtrand_seed_provenance_mismatch"
        )
    try:
        source_dmo_path = Path(str(source_dmo.get("path"))).resolve()
        expected_dmo_path = expected_dmo.resolve()
        expected_dmo_size = expected_dmo.stat().st_size
    except OSError as error:
        raise ProbeError(
            "initial_global_mtrand_seed_provenance_mismatch"
        ) from error
    if (
        source_dmo_path != expected_dmo_path
        or source_dmo.get("size_bytes") != expected_dmo_size
        or source_dmo.get("sha256") != _sha256_path(expected_dmo)
    ):
        raise ProbeError(
            "initial_global_mtrand_seed_provenance_mismatch"
        )
    observation = observations[0]
    changed = observation.get("changed")
    bytes_written = observation.get("bytes_written")
    correction_count = payload.get("state_correction_count")
    bytes_written_total = payload.get("bytes_written_total")
    expected_bytes = MTRAND_STATE_BYTES if changed is True else 0
    if (
        not isinstance(changed, bool)
        or not isinstance(bytes_written, int)
        or isinstance(bytes_written, bool)
        or bytes_written != expected_bytes
        or correction_count != int(changed)
        or bytes_written_total != expected_bytes
        or payload.get("process_memory_mutation") is not changed
        or observation.get("process_memory_mutation") is not changed
        or observation.get("process_context_mutation") is not True
        or observation.get("persistent_file_modified") is not False
        or observation.get("atomic_window_completed") is not True
        or observation.get("wrapper_entry_observed") is not True
        or observation.get("call_target_entered") is not True
        or observation.get("call_return_observed") is not True
        or observation.get(
            "other_threads_resumed_after_call_return"
        ) is not True
        or observation.get("source_path") != str(expected.source_path)
        or observation.get("source_sha256") != expected.source_sha256
        or observation.get("source_process_id")
        != expected.source_process_id
        or observation.get("source_main_thread_id")
        != expected.source_main_thread_id
        or observation.get("semantic_sha256")
        != expected.semantic_sha256
        or observation.get("seed") != expected.seed
        or observation.get("call_address") != expected.call_address
        or observation.get("wrapper_address")
        != expected.wrapper_address
        or observation.get("caller") != expected.caller
        or observation.get("wrapper_entry_stack_return_address")
        != expected.caller
        or observation.get("expected_output") != expected.output
        or observation.get("observed_output") != expected.output
        or observation.get("restored_pre_index") != expected.pre_index
        or observation.get("restored_pre_state_sha256")
        != expected.pre_state_sha256
        or observation.get("expected_post_index") != expected.post_index
        or observation.get("expected_post_state_sha256")
        != expected.post_state_sha256
        or observation.get("observed_post_index")
        != expected.post_index
        or observation.get("observed_post_state_sha256")
        != expected.post_state_sha256
        or not isinstance(observation.get("thread_id"), int)
        or isinstance(observation.get("thread_id"), bool)
        or observation.get("thread_id") <= 0
        or not isinstance(observation.get("live_pre_index"), int)
        or isinstance(observation.get("live_pre_index"), bool)
        or not 0 <= observation.get("live_pre_index") <= MTRAND_STATE_WORDS
        or not isinstance(observation.get("dispatch_target"), int)
        or isinstance(observation.get("dispatch_target"), bool)
        or observation.get("dispatch_target") <= 0
        or not isinstance(
            observation.get("other_threads_suspended_count"),
            int,
        )
        or isinstance(
            observation.get("other_threads_suspended_count"),
            bool,
        )
        or observation.get("other_threads_suspended_count") < 0
        or re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            str(observation.get("live_pre_state_sha256")),
        )
        is None
        or re.fullmatch(
            r"e8[0-9a-f]{8}",
            str(observation.get("call_instruction_hex")),
        )
        is None
        or observation.get("wrapper_instruction_hex")
        != "bac013a300"
    ):
        raise ProbeError(
            "initial_global_mtrand_seed_receipt_contract_failed"
        )
    return {
        "schema": "zuma-rl.pc-initial-global-mtrand-seed-artifact",
        "version": 1,
        "status": "PASS",
        "classification": payload.get("classification"),
        "artifact": path.name,
        "artifact_bytes": len(data),
        "artifact_sha256": _sha256_bytes(data),
        "oracle": expected_oracle_row,
        "state_correction_count": correction_count,
        "bytes_written_total": bytes_written_total,
        "process_memory_mutation": changed,
        "observation": {
            "thread_id": observation.get("thread_id"),
            "live_pre_index": observation.get("live_pre_index"),
            "live_pre_state_sha256": observation.get(
                "live_pre_state_sha256"
            ),
            "changed": changed,
            "bytes_written": bytes_written,
            "other_threads_suspended_count": observation.get(
                "other_threads_suspended_count"
            ),
            "atomic_window_completed": True,
            "call_target_entered": True,
            "call_return_observed": True,
            "other_threads_resumed_after_call_return": True,
            "observed_output": observation.get("observed_output"),
            "observed_post_index": observation.get(
                "observed_post_index"
            ),
            "observed_post_state_sha256": observation.get(
                "observed_post_state_sha256"
            ),
        },
        "pre_blackout_trace": {
            "last_update": trace_result.get("last_update"),
            "stopped_at_update": trace_result.get("stopped_at_update"),
            "failure_count": trace_result.get("failure_count"),
        },
        "persistent_file_modified": False,
    }


def _load_startup_global_mtrand_observation_receipt(
    path: Path,
    *,
    expected_process_id: int,
    expected_oracle: Path,
    expected_seed: int,
    expected_end_at_update: int,
    expected_maximum_draws: int,
    expected_dmo: Path,
    expected_hidden_draw_start_after_source_order: int | None = None,
    expected_hidden_draw_stop_before_source_order: int | None = None,
) -> dict[str, Any]:
    """Validate and compact the read-only startup RNG schedule sidecar."""

    try:
        data = path.read_bytes()
        payload = json.loads(data.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProbeError(
            "startup_global_mtrand_observation_receipt_invalid"
        ) from error
    if not isinstance(payload, Mapping):
        raise ProbeError(
            "startup_global_mtrand_observation_receipt_invalid"
        )
    try:
        expected = load_startup_global_mtrand_observation_oracle(
            expected_oracle,
            seed=expected_seed,
            end_at_update=expected_end_at_update,
            maximum_draws=expected_maximum_draws,
        )
    except (OSError, ValueError) as error:
        raise ProbeError(
            "startup_global_mtrand_observation_oracle_invalid"
        ) from error
    hidden_boundary_values = (
        expected_hidden_draw_start_after_source_order,
        expected_hidden_draw_stop_before_source_order,
    )
    hidden_draw_requested = any(
        value is not None for value in hidden_boundary_values
    )
    if hidden_draw_requested and any(
        value is None for value in hidden_boundary_values
    ):
        raise ProbeError(
            "startup_global_mtrand_hidden_draw_options_incomplete"
        )

    oracle = payload.get("oracle")
    receipt = payload.get("receipt")
    observations = payload.get("observations")
    comparison = payload.get("comparison")
    reconstruction = payload.get("draw_count_reconstruction")
    hidden_draw_payload = payload.get("hidden_draw_interval")
    trace_result = payload.get("pre_blackout_trace_result")
    source_dmo = payload.get("source_dmo")
    expected_count = len(expected.entries)
    if (
        payload.get("schema")
        != "zuma.popcap_startup_global_mtrand_observation_receipt.v1"
        or payload.get("status") != "PASS"
        or payload.get("classification")
        != "diagnostic_read_only_startup_rng_schedule_observation"
        or payload.get("runtime_process_id") != expected_process_id
        or payload.get("process_memory_writes") != 0
        or payload.get("process_memory_mutation") is not False
        or payload.get("process_context_mutation") is not True
        or payload.get("persistent_file_modified") is not False
        or not isinstance(oracle, Mapping)
        or not isinstance(receipt, Mapping)
        or not isinstance(observations, list)
        or len(observations) != expected_count
        or not isinstance(comparison, Mapping)
        or not isinstance(reconstruction, Mapping)
        or not isinstance(trace_result, Mapping)
        or not isinstance(source_dmo, Mapping)
        or trace_result.get("failure_count") != 0
        or trace_result.get("stopped_at_update") is not True
        or reconstruction.get("status") != "PASS"
        or reconstruction.get("failure") is not None
        or reconstruction.get("maximum_draws") != expected.maximum_draws
        or reconstruction.get("reconstructed_state_count")
        != reconstruction.get("unique_state_count")
        or receipt.get("status") != "PASS"
        or receipt.get("failure") is not None
        or receipt.get("oracle_complete") is not True
        or receipt.get("expected_hit_count") != expected_count
        or receipt.get("hit_count") != expected_count
        or receipt.get("hardware_breakpoint_restored") is not True
        or receipt.get("hardware_breakpoint_restore_error") is not None
        or receipt.get("process_memory_writes") != 0
        or receipt.get("process_memory_mutation") is not False
        or receipt.get("process_context_mutation") is not True
        or receipt.get("persistent_file_modified") is not False
        or (
            hidden_draw_requested
            and not isinstance(hidden_draw_payload, Mapping)
        )
        or (
            not hidden_draw_requested
            and hidden_draw_payload is not None
        )
    ):
        raise ProbeError(
            "startup_global_mtrand_observation_receipt_contract_failed"
        )

    expected_oracle_row = {
        "source_path": str(expected.source_path),
        "source_sha256": expected.source_sha256,
        "source_process_id": expected.source_process_id,
        "source_main_thread_id": expected.source_main_thread_id,
        "runtime_executable_sha256": expected.runtime_executable_sha256,
        "seed": expected.seed,
        "wrapper_address": expected.wrapper_address,
        "end_at_update": expected.end_at_update,
        "maximum_draws": expected.maximum_draws,
        "entry_count": expected_count,
        "semantic_sha256": expected.semantic_sha256,
    }
    if dict(oracle) != expected_oracle_row:
        raise ProbeError(
            "startup_global_mtrand_observation_provenance_mismatch"
        )
    try:
        expected_dmo_path = expected_dmo.resolve()
        source_dmo_path = Path(str(source_dmo.get("path"))).resolve()
        expected_dmo_size = expected_dmo.stat().st_size
    except OSError as error:
        raise ProbeError(
            "startup_global_mtrand_observation_provenance_mismatch"
        ) from error
    if (
        source_dmo_path != expected_dmo_path
        or source_dmo.get("size_bytes") != expected_dmo_size
        or source_dmo.get("sha256") != _sha256_path(expected_dmo)
    ):
        raise ProbeError(
            "startup_global_mtrand_observation_provenance_mismatch"
        )

    compact_rows: list[dict[str, Any]] = []
    for order, (row, entry) in enumerate(
        zip(observations, expected.entries, strict=True)
    ):
        if not isinstance(row, Mapping):
            raise ProbeError(
                "startup_global_mtrand_observation_order_invalid"
            )
        observed_pre_draw = row.get("observed_pre_draw_count")
        observed_post_draw = row.get("observed_post_draw_count")
        caller_match = row.get("caller") == entry.caller
        update_match = (
            row.get("framework_update") == entry.framework_update
        )
        pre_state_match = (
            row.get("pre_index") == entry.pre_index
            and row.get("pre_state_sha256")
            == entry.pre_state_sha256
        )
        transition_match = (
            row.get("inferred_output") == entry.output
            and row.get("inferred_post_index") == entry.post_index
            and row.get("inferred_post_state_sha256")
            == entry.post_state_sha256
        )
        exact_match = (
            caller_match
            and update_match
            and pre_state_match
            and transition_match
        )
        if (
            row.get("order") != order
            or row.get("expected_source_order") != entry.source_order
            or row.get("expected_framework_update")
            != entry.framework_update
            or row.get("expected_caller") != entry.caller
            or row.get("expected_output") != entry.output
            or row.get("expected_pre_draw_count")
            != entry.pre_draw_count
            or row.get("expected_pre_index") != entry.pre_index
            or row.get("expected_pre_state_sha256")
            != entry.pre_state_sha256
            or row.get("expected_post_draw_count")
            != entry.post_draw_count
            or row.get("expected_post_index") != entry.post_index
            or row.get("expected_post_state_sha256")
            != entry.post_state_sha256
            or row.get("caller_match") is not caller_match
            or row.get("framework_update_match") is not update_match
            or row.get("pre_state_match") is not pre_state_match
            or row.get("transition_match") is not transition_match
            or row.get("exact_match") is not exact_match
            or not isinstance(observed_pre_draw, int)
            or isinstance(observed_pre_draw, bool)
            or not isinstance(observed_post_draw, int)
            or isinstance(observed_post_draw, bool)
            or observed_post_draw != observed_pre_draw + 1
            or row.get("pre_draw_count_delta")
            != observed_pre_draw - entry.pre_draw_count
            or row.get("post_draw_count_delta")
            != observed_post_draw - entry.post_draw_count
            or row.get("process_memory_writes") != 0
            or row.get("process_memory_mutation") is not False
            or row.get("process_context_mutation") is not True
            or row.get("persistent_file_modified") is not False
            or re.fullmatch(
                r"sha256:[0-9a-f]{64}",
                str(row.get("pre_state_sha256")),
            )
            is None
            or re.fullmatch(
                r"sha256:[0-9a-f]{64}",
                str(row.get("inferred_post_state_sha256")),
            )
            is None
        ):
            raise ProbeError(
                "startup_global_mtrand_observation_row_contract_failed"
            )
        compact_rows.append(
            {
                "order": order,
                "framework_update": row.get("framework_update"),
                "expected_framework_update": entry.framework_update,
                "caller": row.get("caller"),
                "expected_caller": entry.caller,
                "observed_pre_draw_count": observed_pre_draw,
                "expected_pre_draw_count": entry.pre_draw_count,
                "pre_draw_count_delta": row.get("pre_draw_count_delta"),
                "observed_hidden_draws_before_call": row.get(
                    "observed_hidden_draws_before_call"
                ),
                "expected_hidden_draws_before_call": row.get(
                    "expected_hidden_draws_before_call"
                ),
                "caller_match": caller_match,
                "framework_update_match": update_match,
                "pre_state_match": pre_state_match,
                "transition_match": transition_match,
                "exact_match": exact_match,
            }
        )

    def first_mismatch(field: str) -> int | None:
        return next(
            (
                int(row["order"])
                for row in compact_rows
                if row.get(field) is False
            ),
            None,
        )

    first_draw_mismatch = next(
        (
            int(row["order"])
            for row in compact_rows
            if row.get("pre_draw_count_delta") != 0
        ),
        None,
    )
    expected_comparison = {
        "observed_call_count": expected_count,
        "expected_call_count": expected_count,
        "exact_match_count": sum(
            bool(row["exact_match"]) for row in compact_rows
        ),
        "mismatch_count": sum(
            not bool(row["exact_match"]) for row in compact_rows
        ),
        "first_exact_mismatch_order": first_mismatch("exact_match"),
        "first_caller_mismatch_order": first_mismatch("caller_match"),
        "first_framework_update_mismatch_order": first_mismatch(
            "framework_update_match"
        ),
        "first_pre_state_mismatch_order": first_mismatch(
            "pre_state_match"
        ),
        "first_transition_mismatch_order": first_mismatch(
            "transition_match"
        ),
        "first_draw_count_mismatch_order": first_draw_mismatch,
    }
    if dict(comparison) != expected_comparison:
        raise ProbeError(
            "startup_global_mtrand_observation_comparison_invalid"
        )
    compact_hidden_draw_interval: dict[str, Any] | None = None
    if hidden_draw_requested:
        assert isinstance(hidden_draw_payload, Mapping)
        assert expected_hidden_draw_start_after_source_order is not None
        assert expected_hidden_draw_stop_before_source_order is not None
        source_order_to_entry_order = {
            entry.source_order: order
            for order, entry in enumerate(expected.entries)
        }
        start_entry_order = source_order_to_entry_order.get(
            expected_hidden_draw_start_after_source_order
        )
        stop_entry_order = source_order_to_entry_order.get(
            expected_hidden_draw_stop_before_source_order
        )
        if (
            start_entry_order is None
            or stop_entry_order is None
            or stop_entry_order != start_entry_order + 1
        ):
            raise ProbeError(
                "startup_global_mtrand_hidden_draw_boundaries_invalid"
            )
        start_entry = expected.entries[start_entry_order]
        stop_entry = expected.entries[stop_entry_order]
        expected_hidden_draw_count = (
            stop_entry.pre_draw_count - start_entry.post_draw_count
        )
        hidden_contract = hidden_draw_payload.get("contract")
        hidden_summary = hidden_draw_payload.get("summary")
        hidden_histogram = hidden_draw_payload.get("caller_histogram")
        hidden_observations = hidden_draw_payload.get("observations")
        receipt_hidden_contract = receipt.get("hidden_draw_interval")
        if (
            hidden_draw_payload.get("status") != "PASS"
            or hidden_draw_payload.get("failures") != []
            or hidden_draw_payload.get("process_memory_writes") != 0
            or hidden_draw_payload.get("process_memory_mutation") is not False
            or hidden_draw_payload.get("process_context_mutation") is not True
            or hidden_draw_payload.get("persistent_file_modified") is not False
            or not isinstance(hidden_contract, Mapping)
            or not isinstance(receipt_hidden_contract, Mapping)
            or dict(hidden_contract) != dict(receipt_hidden_contract)
            or not isinstance(hidden_summary, Mapping)
            or not isinstance(hidden_histogram, list)
            or not isinstance(hidden_observations, list)
            or not hidden_observations
            or hidden_contract.get("status") != "PASS"
            or hidden_contract.get("failure") is not None
            or hidden_contract.get("start_after_source_order")
            != expected_hidden_draw_start_after_source_order
            or hidden_contract.get("start_entry_order")
            != start_entry_order
            or hidden_contract.get("start_framework_update")
            != start_entry.framework_update
            or hidden_contract.get("start_post_draw_count")
            != start_entry.post_draw_count
            or hidden_contract.get("start_post_state_sha256")
            != start_entry.post_state_sha256
            or hidden_contract.get("stop_before_source_order")
            != expected_hidden_draw_stop_before_source_order
            or hidden_contract.get("stop_entry_order") != stop_entry_order
            or hidden_contract.get("stop_framework_update")
            != stop_entry.framework_update
            or hidden_contract.get("expected_stop_pre_draw_count")
            != stop_entry.pre_draw_count
            or hidden_contract.get("expected_stop_pre_state_sha256")
            != stop_entry.pre_state_sha256
            or hidden_contract.get("expected_hidden_draw_count")
            != expected_hidden_draw_count
            or hidden_contract.get("expected_total_index_write_count")
            != expected_hidden_draw_count + 1
            or hidden_contract.get("start_boundary_observed") is not True
            or hidden_contract.get("stop_boundary_observed") is not True
            or hidden_contract.get("watch_armed") is not True
            or hidden_contract.get("watch_disarmed") is not True
            or hidden_contract.get("process_memory_writes") != 0
            or hidden_contract.get("process_memory_mutation") is not False
            or hidden_contract.get("persistent_file_modified") is not False
        ):
            raise ProbeError(
                "startup_global_mtrand_hidden_draw_contract_failed"
            )

        compact_hidden_rows: list[dict[str, Any]] = []
        for order, row in enumerate(hidden_observations):
            if not isinstance(row, Mapping):
                raise ProbeError(
                    "startup_global_mtrand_hidden_draw_row_invalid"
                )
            post_draw_count = row.get("observed_post_draw_count")
            stack_return = row.get("stack_return")
            instruction_pointer = row.get("instruction_pointer")
            if (
                row.get("order") != order
                or row.get("receipt_order") != order
                or row.get("phase")
                != (
                    "start_boundary_wrapper_draw"
                    if order == 0
                    else "hidden_interval_draw"
                )
                or not isinstance(post_draw_count, int)
                or isinstance(post_draw_count, bool)
                or not isinstance(stack_return, int)
                or isinstance(stack_return, bool)
                or not isinstance(instruction_pointer, int)
                or isinstance(instruction_pointer, bool)
                or row.get("hardware_breakpoint_slot") != 1
                or row.get("hardware_breakpoint_access") != "write"
                or row.get("hardware_breakpoint_length_bytes") != 4
                or row.get("process_memory_writes") != 0
                or row.get("process_memory_mutation") is not False
                or row.get("process_context_mutation") is not True
                or row.get("persistent_file_modified") is not False
                or re.fullmatch(
                    r"sha256:[0-9a-f]{64}",
                    str(row.get("post_state_sha256")),
                )
                is None
            ):
                raise ProbeError(
                    "startup_global_mtrand_hidden_draw_row_invalid"
                )
            if order == 0:
                if (
                    post_draw_count != start_entry.post_draw_count
                    or row.get("post_index") != start_entry.post_index
                    or row.get("post_state_sha256")
                    != start_entry.post_state_sha256
                ):
                    raise ProbeError(
                        "startup_global_mtrand_hidden_draw_start_mismatch"
                    )
            elif post_draw_count != (
                compact_hidden_rows[-1]["observed_post_draw_count"] + 1
            ):
                raise ProbeError(
                    "startup_global_mtrand_hidden_draw_sequence_invalid"
                )
            compact_hidden_rows.append(
                {
                    "order": order,
                    "framework_update": row.get("framework_update"),
                    "stack_return": stack_return,
                    "stack_return_hex": row.get("stack_return_hex"),
                    "instruction_pointer": instruction_pointer,
                    "instruction_pointer_hex": row.get(
                        "instruction_pointer_hex"
                    ),
                    "post_index": row.get("post_index"),
                    "post_state_sha256": row.get("post_state_sha256"),
                    "observed_post_draw_count": post_draw_count,
                }
            )

        observed_hidden_draw_count = len(compact_hidden_rows) - 1
        observed_stop_pre_draw_count = compact_rows[stop_entry_order][
            "observed_pre_draw_count"
        ]
        last_observed_draw_count = compact_hidden_rows[-1][
            "observed_post_draw_count"
        ]
        missing_hidden_draw_count = (
            expected_hidden_draw_count - observed_hidden_draw_count
        )
        if (
            hidden_contract.get("index_write_count")
            != len(compact_hidden_rows)
            or hidden_contract.get("observed_hidden_write_count")
            != observed_hidden_draw_count
            or hidden_summary.get("expected_hidden_draw_count")
            != expected_hidden_draw_count
            or hidden_summary.get("observed_hidden_draw_count")
            != observed_hidden_draw_count
            or hidden_summary.get("missing_hidden_draw_count")
            != missing_hidden_draw_count
            or hidden_summary.get("deficit_detected")
            is not (missing_hidden_draw_count > 0)
            or hidden_summary.get("index_write_count")
            != len(compact_hidden_rows)
            or hidden_summary.get("expected_total_index_write_count")
            != expected_hidden_draw_count + 1
            or hidden_summary.get("start_post_draw_count")
            != start_entry.post_draw_count
            or hidden_summary.get("first_observed_post_draw_count")
            != start_entry.post_draw_count
            or hidden_summary.get("expected_stop_pre_draw_count")
            != stop_entry.pre_draw_count
            or hidden_summary.get("observed_stop_pre_draw_count")
            != observed_stop_pre_draw_count
            or hidden_summary.get("last_observed_post_draw_count")
            != last_observed_draw_count
            or hidden_summary.get("observed_draw_span")
            != observed_hidden_draw_count
            or hidden_summary.get("observation_orders_valid") is not True
            or hidden_summary.get("draw_counts_complete") is not True
            or hidden_summary.get("draw_counts_contiguous") is not True
            or observed_stop_pre_draw_count != last_observed_draw_count
        ):
            raise ProbeError(
                "startup_global_mtrand_hidden_draw_summary_invalid"
            )

        histogram_groups: dict[
            tuple[int, int], dict[str, Any]
        ] = {}
        for row in compact_hidden_rows[1:]:
            key = (row["stack_return"], row["instruction_pointer"])
            group = histogram_groups.setdefault(
                key,
                {
                    "stack_return": key[0],
                    "stack_return_hex": f"0x{key[0]:08x}",
                    "instruction_pointer": key[1],
                    "instruction_pointer_hex": f"0x{key[1]:08x}",
                    "count": 0,
                    "first_observation_order": row["order"],
                    "last_observation_order": row["order"],
                    "first_framework_update": row["framework_update"],
                    "last_framework_update": row["framework_update"],
                },
            )
            group["count"] += 1
            group["last_observation_order"] = row["order"]
            group["last_framework_update"] = row["framework_update"]
        expected_histogram = sorted(
            histogram_groups.values(),
            key=lambda row: (
                -int(row["count"]),
                int(row["stack_return"]),
                int(row["instruction_pointer"]),
            ),
        )
        if hidden_histogram != expected_histogram:
            raise ProbeError(
                "startup_global_mtrand_hidden_draw_histogram_invalid"
            )
        compact_hidden_draw_interval = {
            "status": "PASS",
            "start_after_source_order": (
                expected_hidden_draw_start_after_source_order
            ),
            "stop_before_source_order": (
                expected_hidden_draw_stop_before_source_order
            ),
            "summary": dict(hidden_summary),
            "caller_histogram": expected_histogram,
            "draw_schedule": compact_hidden_rows,
            "process_memory_writes": 0,
            "process_memory_mutation": False,
            "process_context_mutation": True,
            "persistent_file_modified": False,
        }
    first_order = expected_comparison["first_exact_mismatch_order"]
    compact_payload = {
        "schema": "zuma-rl.pc-startup-global-mtrand-observation-artifact",
        "version": 1,
        "status": "PASS",
        "classification": payload.get("classification"),
        "artifact": path.name,
        "artifact_bytes": len(data),
        "artifact_sha256": _sha256_bytes(data),
        "oracle": expected_oracle_row,
        "comparison": expected_comparison,
        "first_mismatch_observation": (
            compact_rows[first_order] if first_order is not None else None
        ),
        "draw_schedule": compact_rows,
        "process_memory_writes": 0,
        "process_memory_mutation": False,
        "process_context_mutation": True,
        "pre_blackout_trace": {
            "last_update": trace_result.get("last_update"),
            "stopped_at_update": trace_result.get("stopped_at_update"),
            "failure_count": trace_result.get("failure_count"),
        },
        "persistent_file_modified": False,
    }
    if compact_hidden_draw_interval is not None:
        compact_payload["hidden_draw_interval"] = (
            compact_hidden_draw_interval
        )
    return compact_payload


def _wait_for_startup_main_thread_id(
    *,
    trace_log_path: Path,
    trace_process: subprocess.Popen[bytes],
    expected_process_id: int,
    timeout_seconds: float,
) -> int:
    """Read the launcher's proven startup thread identity from its log."""

    if expected_process_id <= 0 or timeout_seconds <= 0:
        raise ProbeError("startup_main_thread_wait_invalid")
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            text = trace_log_path.read_text(
                encoding="utf-8",
                errors="replace",
            )
        except OSError:
            text = ""
        matches = {
            (int(process_id), int(thread_id))
            for process_id, thread_id in re.findall(
                r"(?m)^startup_rng\b[^\r\n]*\bpid=(\d+)\s+tid=(\d+)\b",
                text,
            )
        }
        matching_threads = {
            thread_id
            for process_id, thread_id in matches
            if process_id == expected_process_id
        }
        foreign_processes = {
            process_id
            for process_id, _thread_id in matches
            if process_id != expected_process_id
        }
        if foreign_processes or len(matching_threads) > 1:
            raise ProbeError(
                "startup_main_thread_identity_ambiguous:"
                f"expected_pid={expected_process_id}:"
                f"matches={sorted(matches)}"
            )
        if len(matching_threads) == 1:
            thread_id = next(iter(matching_threads))
            if thread_id <= 0:
                raise ProbeError("startup_main_thread_identity_invalid")
            return thread_id
        if trace_process.poll() is not None:
            raise ProbeError(
                "trace_exited_before_startup_main_thread_identity:"
                f"exit={trace_process.returncode}"
            )
        if time.monotonic() >= deadline:
            raise ProbeError("startup_main_thread_identity_timeout")
        time.sleep(0.02)


def _start_global_mtrand_call_sync(
    *,
    project_root: Path,
    pid: int,
    main_thread_id: int,
    runtime_executable: Path,
    oracle_path: Path,
    seed: int,
    start_after_update: int,
    end_at_update: int,
    maximum_draws: int,
    timeout_seconds: float,
    attempt_root: Path,
    observe_boundary_only: bool = False,
    observe_thread_runtime: bool = False,
    observe_handoff_thread_runtime: bool = False,
    observe_handoff_rng_state: bool = False,
    handoff_rng_wait_target_words_sha256: str | None = None,
    handoff_rng_wait_min_index: int | None = None,
    handoff_rng_wait_timeout_seconds: float | None = None,
    handoff_rng_wait_poll_interval_seconds: float | None = None,
) -> dict[str, Any]:
    """Start the post-tracer debugger before the handoff boundary."""

    if (
        pid <= 0
        or main_thread_id <= 0
        or not oracle_path.is_file()
        or not 0 <= seed <= 0xFFFFFFFF
        or start_after_update < 0
        or end_at_update <= start_after_update
        or maximum_draws <= 0
        or timeout_seconds <= 0
    ):
        raise ProbeError("global_mtrand_call_sync_options_invalid")
    if observe_thread_runtime and not observe_boundary_only:
        raise ProbeError(
            "global_mtrand_thread_runtime_requires_boundary_observer"
        )
    if observe_handoff_thread_runtime and not observe_thread_runtime:
        raise ProbeError(
            "global_mtrand_handoff_runtime_requires_thread_runtime"
        )
    if observe_handoff_rng_state and not observe_boundary_only:
        raise ProbeError(
            "global_mtrand_handoff_rng_requires_boundary_observer"
        )
    handoff_rng_wait_values = (
        handoff_rng_wait_target_words_sha256,
        handoff_rng_wait_min_index,
        handoff_rng_wait_timeout_seconds,
        handoff_rng_wait_poll_interval_seconds,
    )
    handoff_rng_wait_enabled = any(
        value is not None for value in handoff_rng_wait_values
    )
    if handoff_rng_wait_enabled and (
        any(value is None for value in handoff_rng_wait_values)
        or not observe_boundary_only
        or not observe_handoff_rng_state
        or not _is_canonical_sha256(
            handoff_rng_wait_target_words_sha256
        )
        or isinstance(handoff_rng_wait_min_index, bool)
        or not isinstance(handoff_rng_wait_min_index, int)
        or not 0 <= handoff_rng_wait_min_index <= MTRAND_STATE_WORDS
        or isinstance(handoff_rng_wait_timeout_seconds, bool)
        or not isinstance(handoff_rng_wait_timeout_seconds, (int, float))
        or handoff_rng_wait_timeout_seconds <= 0
        or isinstance(handoff_rng_wait_poll_interval_seconds, bool)
        or not isinstance(
            handoff_rng_wait_poll_interval_seconds,
            (int, float),
        )
        or handoff_rng_wait_poll_interval_seconds <= 0
        or handoff_rng_wait_poll_interval_seconds
        > handoff_rng_wait_timeout_seconds
    ):
        raise ProbeError("global_mtrand_handoff_rng_wait_options_invalid")
    artifact_stem = (
        "global-mtrand-boundary-observation"
        if observe_boundary_only
        else "global-mtrand-call-sync"
    )
    result_path = attempt_root / f"{artifact_stem}.json"
    ready_path = attempt_root / f"{artifact_stem}.ready.json"
    stop_path = attempt_root / f"{artifact_stem}.stop"
    log_path = attempt_root / f"{artifact_stem}.log"
    log_stream = log_path.open("xb")
    args = [
        sys.executable,
        str(
            project_root
            / "tools"
            / "synchronize_popcap_global_mtrand_calls.py"
        ),
        "--pid",
        str(pid),
        "--main-thread-id",
        str(main_thread_id),
        "--executable",
        str(runtime_executable),
        "--oracle",
        str(oracle_path),
        "--seed",
        str(seed),
        "--start-after-update",
        str(start_after_update),
        "--end-at-update",
        str(end_at_update),
        "--maximum-draws",
        str(maximum_draws),
        "--output",
        str(result_path),
        "--ready",
        str(ready_path),
        "--stop",
        str(stop_path),
        "--resume-main-thread-on-ready",
        "--attach-timeout",
        str(timeout_seconds),
        "--timeout",
        str(timeout_seconds),
    ]
    if observe_boundary_only:
        args.append("--observe-boundary-only")
    if observe_thread_runtime:
        args.append("--observe-boundary-thread-runtime")
    if observe_handoff_thread_runtime:
        args.append("--observe-handoff-thread-runtime")
    if observe_handoff_rng_state:
        args.append("--observe-handoff-rng-state")
    if handoff_rng_wait_enabled:
        args.extend(
            [
                "--handoff-rng-wait-target-words-sha256",
                str(handoff_rng_wait_target_words_sha256),
                "--handoff-rng-wait-min-index",
                str(handoff_rng_wait_min_index),
                "--handoff-rng-wait-timeout-seconds",
                str(handoff_rng_wait_timeout_seconds),
                "--handoff-rng-wait-poll-interval-seconds",
                str(handoff_rng_wait_poll_interval_seconds),
            ]
        )
    try:
        process = subprocess.Popen(
            args,
            cwd=project_root,
            env=_child_environment(project_root),
            stdin=subprocess.DEVNULL,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            close_fds=True,
        )
    except Exception:
        log_stream.close()
        raise
    return {
        "process": process,
        "log_stream": log_stream,
        "log_path": log_path,
        "result_path": result_path,
        "ready_path": ready_path,
        "stop_path": stop_path,
        "pid": pid,
        "main_thread_id": main_thread_id,
        "runtime_executable": runtime_executable,
        "oracle_path": oracle_path,
        "seed": seed,
        "start_after_update": start_after_update,
        "end_at_update": end_at_update,
        "timeout_seconds": timeout_seconds,
        "observe_boundary_only": observe_boundary_only,
        "observe_thread_runtime": observe_thread_runtime,
        "observe_handoff_thread_runtime": (
            observe_handoff_thread_runtime
        ),
        "observe_handoff_rng_state": observe_handoff_rng_state,
        "handoff_rng_wait_enabled": handoff_rng_wait_enabled,
        "handoff_rng_wait_target_words_sha256": (
            handoff_rng_wait_target_words_sha256
        ),
        "handoff_rng_wait_min_index": handoff_rng_wait_min_index,
        "handoff_rng_wait_timeout_seconds": (
            handoff_rng_wait_timeout_seconds
        ),
        "handoff_rng_wait_poll_interval_seconds": (
            handoff_rng_wait_poll_interval_seconds
        ),
    }


def _handoff_rng_wait_ready_contract_passes(
    payload: Mapping[str, Any],
    *,
    bundle: Mapping[str, Any],
    enabled: bool,
) -> bool:
    prefix = "handoff_rng_readiness_wait_"
    wait_keys = (
        "status",
        "target_words_sha256",
        "minimum_index",
        "timeout_seconds",
        "poll_interval_seconds",
        "target_reached",
        "final_state_sha256",
        "final_words_sha256",
        "final_index",
        "final_captured_perf_counter_ns",
    )
    if not enabled:
        return bool(
            payload.get(f"{prefix}enabled", False) is False
            and all(payload.get(f"{prefix}{key}") is None for key in wait_keys)
        )

    status = payload.get(f"{prefix}status")
    target = bundle.get("handoff_rng_wait_target_words_sha256")
    minimum_index = bundle.get("handoff_rng_wait_min_index")
    final_words = payload.get(f"{prefix}final_words_sha256")
    final_index = payload.get(f"{prefix}final_index")
    target_reached = payload.get(f"{prefix}target_reached")
    status_semantics_pass = (
        status == "TARGET_REACHED"
        and target_reached is True
        and final_words == target
        and isinstance(final_index, int)
        and not isinstance(final_index, bool)
        and isinstance(minimum_index, int)
        and final_index >= minimum_index
    ) or (
        status == "TIMEOUT"
        and target_reached is False
        and final_words == target
        and isinstance(final_index, int)
        and not isinstance(final_index, bool)
        and isinstance(minimum_index, int)
        and final_index < minimum_index
    ) or (
        status == "BLOCK_MISMATCH"
        and target_reached is False
        and _is_canonical_sha256(final_words)
        and final_words != target
    )
    return bool(
        payload.get(f"{prefix}enabled") is True
        and status_semantics_pass
        and payload.get(f"{prefix}target_words_sha256") == target
        and payload.get(f"{prefix}minimum_index") == minimum_index
        and payload.get(f"{prefix}timeout_seconds")
        == bundle.get("handoff_rng_wait_timeout_seconds")
        and payload.get(f"{prefix}poll_interval_seconds")
        == bundle.get("handoff_rng_wait_poll_interval_seconds")
        and _is_canonical_sha256(
            payload.get(f"{prefix}final_state_sha256")
        )
        and payload.get(f"{prefix}final_state_sha256")
        == payload.get("handoff_rng_state_sha256")
        and _is_canonical_sha256(final_words)
        and final_words == payload.get("handoff_rng_words_sha256")
        and isinstance(final_index, int)
        and not isinstance(final_index, bool)
        and 0 <= final_index <= MTRAND_STATE_WORDS
        and final_index == payload.get("handoff_rng_index")
        and isinstance(
            payload.get(f"{prefix}final_captured_perf_counter_ns"),
            int,
        )
        and not isinstance(
            payload.get(f"{prefix}final_captured_perf_counter_ns"),
            bool,
        )
        and payload.get(f"{prefix}final_captured_perf_counter_ns")
        == payload.get("handoff_rng_captured_perf_counter_ns")
    )


def _wait_for_global_mtrand_call_sync_ready(
    bundle: Mapping[str, Any],
    *,
    trace_process: subprocess.Popen[bytes],
    timeout_seconds: float,
) -> dict[str, Any]:
    process = bundle["process"]
    ready_path = bundle["ready_path"]
    observe_boundary_only = bool(
        bundle.get("observe_boundary_only", False)
    )
    observe_thread_runtime = bool(
        bundle.get("observe_thread_runtime", False)
    )
    observe_handoff_thread_runtime = bool(
        bundle.get("observe_handoff_thread_runtime", False)
    )
    observe_handoff_rng_state = bool(
        bundle.get("observe_handoff_rng_state", False)
    )
    handoff_rng_wait_enabled = bool(
        bundle.get("handoff_rng_wait_enabled", False)
    )
    if not isinstance(process, subprocess.Popen) or not isinstance(
        ready_path,
        Path,
    ):
        raise ProbeError("global_mtrand_call_sync_bundle_invalid")
    deadline = time.monotonic() + timeout_seconds
    while True:
        if ready_path.is_file():
            try:
                payload = json.loads(
                    ready_path.read_text(encoding="ascii")
                )
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ProbeError(
                    "global_mtrand_call_sync_ready_invalid"
                ) from error
            if (
                not isinstance(payload, Mapping)
                or payload.get("schema")
                != "zuma-rl.pc-global-mtrand-oracle-synchronizer-ready"
                or payload.get("version") != 2
                or payload.get("process_id") != bundle["pid"]
                or payload.get("main_thread_id")
                != bundle["main_thread_id"]
                or payload.get("resume_main_thread_on_ready") is not True
                or payload.get("handoff_verified") is not True
                or payload.get("handoff_main_thread_resumed") is not True
                or payload.get("handoff_resume_previous_suspend_count") != 1
                or payload.get("publication_order")
                != "AFTER_VERIFIED_HANDOFF_RESUME"
                or not isinstance(
                    payload.get("published_perf_counter_ns"), int
                )
                or isinstance(
                    payload.get("published_perf_counter_ns"), bool
                )
                or payload["published_perf_counter_ns"] <= 0
                or payload.get("mode")
                != (
                    "observe_boundary_only"
                    if observe_boundary_only
                    else "synchronize"
                )
                or bool(
                    payload.get(
                        "observe_boundary_thread_runtime",
                        False,
                    )
                )
                != observe_thread_runtime
                or bool(
                    payload.get(
                        "observe_handoff_thread_runtime",
                        False,
                    )
                )
                != observe_handoff_thread_runtime
                or bool(
                    payload.get("observe_handoff_rng_state", False)
                )
                != observe_handoff_rng_state
                or (
                    observe_handoff_rng_state
                    and (
                        not isinstance(
                            payload.get("handoff_rng_state_sha256"),
                            str,
                        )
                        or not str(
                            payload["handoff_rng_state_sha256"]
                        ).startswith("sha256:")
                        or not _is_canonical_sha256(
                            payload.get("handoff_rng_words_sha256")
                        )
                        or not isinstance(
                            payload.get("handoff_rng_index"), int
                        )
                        or isinstance(
                            payload.get("handoff_rng_index"), bool
                        )
                        or not 0 <= payload["handoff_rng_index"] <= 624
                        or not isinstance(
                            payload.get(
                                "handoff_rng_captured_perf_counter_ns"
                            ),
                            int,
                        )
                        or isinstance(
                            payload.get(
                                "handoff_rng_captured_perf_counter_ns"
                            ),
                            bool,
                        )
                        or payload[
                            "handoff_rng_captured_perf_counter_ns"
                        ]
                        <= 0
                    )
                )
                or (
                    not observe_handoff_rng_state
                    and any(
                        payload.get(key) is not None
                        for key in (
                            "handoff_rng_state_sha256",
                            "handoff_rng_words_sha256",
                            "handoff_rng_index",
                            "handoff_rng_captured_perf_counter_ns",
                        )
                    )
                )
                or not _handoff_rng_wait_ready_contract_passes(
                    payload,
                    bundle=bundle,
                    enabled=handoff_rng_wait_enabled,
                )
                or not isinstance(payload.get("expected_call_count"), int)
                or isinstance(payload.get("expected_call_count"), bool)
                or payload["expected_call_count"] <= 0
                or (
                    observe_boundary_only
                    and payload["expected_call_count"] != 1
                )
            ):
                raise ProbeError(
                    "global_mtrand_call_sync_ready_contract_failed"
                )
            return dict(payload)
        if process.poll() is not None:
            raise ProbeError(
                "global_mtrand_call_sync_exited_before_ready:"
                f"exit={process.returncode}"
            )
        if trace_process.poll() is not None:
            raise ProbeError(
                "trace_exited_before_global_mtrand_call_sync_ready:"
                f"exit={trace_process.returncode}"
            )
        if time.monotonic() >= deadline:
            raise ProbeError("global_mtrand_call_sync_ready_timeout")
        time.sleep(0.05)


def _handoff_rng_state_contract_passes(
    handoff_rng_state: object,
    *,
    ready: Mapping[str, Any],
    enabled: bool,
) -> bool:
    """Validate the optional read-only RNG snapshot bound to ready-v2."""

    if not enabled:
        return handoff_rng_state is None
    return bool(
        isinstance(handoff_rng_state, Mapping)
        and handoff_rng_state.get("schema")
        == "zuma-rl.pc-global-mtrand-handoff-state"
        and handoff_rng_state.get("version") == 1
        and handoff_rng_state.get("classification")
        == "diagnostic-read-only-process-memory-snapshot"
        and handoff_rng_state.get("address")
        == G_FRAMEWORK_MTRAND_ADDRESS
        and handoff_rng_state.get("state_bytes") == MTRAND_STATE_BYTES
        and handoff_rng_state.get("state_sha256")
        == ready.get("handoff_rng_state_sha256")
        and handoff_rng_state.get("words_sha256")
        == ready.get("handoff_rng_words_sha256")
        and _is_canonical_sha256(
            handoff_rng_state.get("words_sha256")
        )
        and handoff_rng_state.get("index")
        == ready.get("handoff_rng_index")
        and handoff_rng_state.get("captured_perf_counter_ns")
        == ready.get("handoff_rng_captured_perf_counter_ns")
        and isinstance(handoff_rng_state.get("draw_count"), int)
        and not isinstance(handoff_rng_state.get("draw_count"), bool)
        and handoff_rng_state.get("process_memory_reads") == 1
        and handoff_rng_state.get("process_memory_read_bytes")
        == MTRAND_STATE_BYTES
        and handoff_rng_state.get("process_memory_writes") == 0
        and handoff_rng_state.get("process_memory_mutation") is False
    )


def _handoff_rng_wait_contract_passes(
    readiness_wait: object,
    *,
    ready: Mapping[str, Any],
    bundle: Mapping[str, Any],
    handoff_rng_state: object,
    enabled: bool,
) -> bool:
    """Validate the optional bounded wait without requiring target success."""

    if not enabled:
        return readiness_wait is None
    if not isinstance(readiness_wait, Mapping) or not isinstance(
        handoff_rng_state,
        Mapping,
    ):
        return False

    status = readiness_wait.get("status")
    target = bundle.get("handoff_rng_wait_target_words_sha256")
    minimum_index = bundle.get("handoff_rng_wait_min_index")
    final_words = readiness_wait.get("final_words_sha256")
    final_index = readiness_wait.get("final_index")
    target_reached = readiness_wait.get("target_reached")
    status_semantics_pass = (
        status == "TARGET_REACHED"
        and target_reached is True
        and final_words == target
        and isinstance(final_index, int)
        and not isinstance(final_index, bool)
        and isinstance(minimum_index, int)
        and final_index >= minimum_index
    ) or (
        status == "TIMEOUT"
        and target_reached is False
        and final_words == target
        and isinstance(final_index, int)
        and not isinstance(final_index, bool)
        and isinstance(minimum_index, int)
        and final_index < minimum_index
    ) or (
        status == "BLOCK_MISMATCH"
        and target_reached is False
        and _is_canonical_sha256(final_words)
        and final_words != target
    )
    reads = readiness_wait.get("process_memory_reads")
    started_ns = readiness_wait.get("started_perf_counter_ns")
    finished_ns = readiness_wait.get("finished_perf_counter_ns")
    initial_captured_ns = readiness_wait.get(
        "initial_captured_perf_counter_ns"
    )
    final_captured_ns = readiness_wait.get(
        "final_captured_perf_counter_ns"
    )
    return bool(
        readiness_wait.get("schema")
        == "zuma-rl.pc-global-mtrand-handoff-readiness-wait"
        and readiness_wait.get("version") == 1
        and readiness_wait.get("classification")
        == (
            "diagnostic-read-only-main-thread-suspended-"
            "rng-readiness-wait"
        )
        and status_semantics_pass
        and readiness_wait.get("target_words_sha256") == target
        and readiness_wait.get("minimum_index") == minimum_index
        and readiness_wait.get("timeout_seconds")
        == bundle.get("handoff_rng_wait_timeout_seconds")
        and readiness_wait.get("poll_interval_seconds")
        == bundle.get("handoff_rng_wait_poll_interval_seconds")
        and readiness_wait.get("main_thread_suspended_during_wait") is True
        and isinstance(started_ns, int)
        and not isinstance(started_ns, bool)
        and isinstance(finished_ns, int)
        and not isinstance(finished_ns, bool)
        and isinstance(initial_captured_ns, int)
        and not isinstance(initial_captured_ns, bool)
        and isinstance(final_captured_ns, int)
        and not isinstance(final_captured_ns, bool)
        and 0 < started_ns <= initial_captured_ns
        and initial_captured_ns <= final_captured_ns <= finished_ns
        and readiness_wait.get("elapsed_perf_counter_ns")
        == finished_ns - started_ns
        and _is_canonical_sha256(
            readiness_wait.get("initial_state_sha256")
        )
        and _is_canonical_sha256(
            readiness_wait.get("initial_words_sha256")
        )
        and isinstance(readiness_wait.get("initial_index"), int)
        and not isinstance(readiness_wait.get("initial_index"), bool)
        and 0
        <= readiness_wait.get("initial_index")
        <= MTRAND_STATE_WORDS
        and isinstance(readiness_wait.get("initial_draw_count"), int)
        and not isinstance(
            readiness_wait.get("initial_draw_count"), bool
        )
        and readiness_wait.get("final_state_sha256")
        == handoff_rng_state.get("state_sha256")
        and readiness_wait.get("final_words_sha256")
        == handoff_rng_state.get("words_sha256")
        and readiness_wait.get("final_index")
        == handoff_rng_state.get("index")
        and readiness_wait.get("final_captured_perf_counter_ns")
        == handoff_rng_state.get("captured_perf_counter_ns")
        and readiness_wait.get("final_draw_count")
        == handoff_rng_state.get("draw_count")
        and readiness_wait.get("final_state_sha256")
        == ready.get("handoff_rng_readiness_wait_final_state_sha256")
        and readiness_wait.get("final_words_sha256")
        == ready.get("handoff_rng_readiness_wait_final_words_sha256")
        and readiness_wait.get("final_index")
        == ready.get("handoff_rng_readiness_wait_final_index")
        and readiness_wait.get("final_captured_perf_counter_ns")
        == ready.get(
            "handoff_rng_readiness_wait_final_captured_perf_counter_ns"
        )
        and isinstance(reads, int)
        and not isinstance(reads, bool)
        and reads >= 1
        and readiness_wait.get("poll_count") == reads - 1
        and readiness_wait.get("process_memory_read_bytes")
        == reads * MTRAND_STATE_BYTES
        and readiness_wait.get("process_memory_writes") == 0
        and readiness_wait.get("process_memory_mutation") is False
    )


def _finish_global_mtrand_boundary_observation(
    bundle: Mapping[str, Any],
    *,
    ready: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a one-shot, post-detach, zero-write boundary receipt."""

    process = bundle["process"]
    log_stream = bundle["log_stream"]
    result_path = bundle["result_path"]
    if (
        not isinstance(process, subprocess.Popen)
        or not isinstance(result_path, Path)
        or bundle.get("observe_boundary_only") is not True
    ):
        raise ProbeError("global_mtrand_boundary_observation_bundle_invalid")
    try:
        process.wait(timeout=float(bundle["timeout_seconds"]) + 10.0)
    except subprocess.TimeoutExpired as error:
        raise ProbeError(
            "global_mtrand_boundary_observation_finish_timeout"
        ) from error
    finally:
        if not log_stream.closed:
            log_stream.close()
    if not result_path.is_file():
        raise ProbeError("global_mtrand_boundary_observation_missing")
    try:
        data = result_path.read_bytes()
        payload = json.loads(data.decode("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProbeError(
            "global_mtrand_boundary_observation_invalid"
        ) from error
    oracle = payload.get("oracle") if isinstance(payload, Mapping) else None
    final_post = (
        payload.get("final_post_verification")
        if isinstance(payload, Mapping)
        else None
    )
    reconstruction = (
        payload.get("draw_count_reconstruction")
        if isinstance(payload, Mapping)
        else None
    )
    hits = payload.get("hits") if isinstance(payload, Mapping) else None
    expected_runtime = Path(bundle["runtime_executable"])
    expected_oracle = Path(bundle["oracle_path"])
    first_hit = hits[0] if isinstance(hits, list) and hits else None
    observe_thread_runtime = bool(
        bundle.get("observe_thread_runtime", False)
    )
    observe_handoff_thread_runtime = bool(
        bundle.get("observe_handoff_thread_runtime", False)
    )
    observe_handoff_rng_state = bool(
        bundle.get("observe_handoff_rng_state", False)
    )
    handoff_rng_wait_enabled = bool(
        bundle.get("handoff_rng_wait_enabled", False)
    )
    handoff_rng_state = (
        payload.get("handoff_rng_state_observation")
        if isinstance(payload, Mapping)
        else None
    )
    handoff_rng_readiness_wait = (
        payload.get("handoff_rng_readiness_wait")
        if isinstance(payload, Mapping)
        else None
    )
    handoff_rng_contract_pass = _handoff_rng_state_contract_passes(
        handoff_rng_state,
        ready=ready,
        enabled=observe_handoff_rng_state,
    )
    handoff_rng_wait_contract_pass = (
        _handoff_rng_wait_contract_passes(
            handoff_rng_readiness_wait,
            ready=ready,
            bundle=bundle,
            handoff_rng_state=handoff_rng_state,
            enabled=handoff_rng_wait_enabled,
        )
    )
    thread_runtime = (
        payload.get("thread_runtime_observation")
        if isinstance(payload, Mapping)
        else None
    )
    boundary_thread_snapshot = (
        thread_runtime.get("boundary_snapshot")
        if isinstance(thread_runtime, Mapping)
        else None
    )
    boundary_threads = (
        boundary_thread_snapshot.get("threads")
        if isinstance(boundary_thread_snapshot, Mapping)
        else None
    )
    handoff_thread_snapshot = (
        thread_runtime.get("handoff_snapshot")
        if isinstance(thread_runtime, Mapping)
        else None
    )
    handoff_threads = (
        handoff_thread_snapshot.get("threads")
        if isinstance(handoff_thread_snapshot, Mapping)
        else None
    )
    handoff_main_thread_rows = (
        [
            row
            for row in handoff_threads
            if isinstance(row, Mapping)
            and row.get("thread_id") == bundle["main_thread_id"]
            and row.get("is_main_thread") is True
            and row.get("accessible") is True
        ]
        if isinstance(handoff_threads, list)
        else []
    )
    handoff_to_boundary_delta = (
        thread_runtime.get("handoff_to_boundary_delta")
        if isinstance(thread_runtime, Mapping)
        else None
    )
    handoff_runtime_contract_pass = (
        (
            isinstance(handoff_thread_snapshot, Mapping)
            and handoff_thread_snapshot.get("schema")
            == "zuma-rl.pc-thread-runtime-snapshot"
            and handoff_thread_snapshot.get("version") == 1
            and handoff_thread_snapshot.get("process_id") == bundle["pid"]
            and handoff_thread_snapshot.get("main_thread_id")
            == bundle["main_thread_id"]
            and handoff_thread_snapshot.get("phase")
            == "global_mtrand_handoff_pre_resume"
            and handoff_thread_snapshot.get("process_memory_reads") == 0
            and handoff_thread_snapshot.get("process_memory_writes") == 0
            and handoff_thread_snapshot.get("process_context_reads") == 0
            and handoff_thread_snapshot.get("process_context_writes") == 0
            and isinstance(handoff_threads, list)
            and handoff_thread_snapshot.get("thread_count")
            == len(handoff_threads)
            and len(handoff_main_thread_rows) == 1
            and isinstance(handoff_to_boundary_delta, Mapping)
            and handoff_to_boundary_delta.get("schema")
            == "zuma-rl.pc-thread-runtime-delta"
            and handoff_to_boundary_delta.get("version") == 1
            and handoff_to_boundary_delta.get("process_id") == bundle["pid"]
            and handoff_to_boundary_delta.get("main_thread_id")
            == bundle["main_thread_id"]
            and handoff_to_boundary_delta.get("ready_thread_count")
            == len(handoff_threads)
            and handoff_to_boundary_delta.get("boundary_thread_count")
            == len(boundary_threads or [])
            and isinstance(
                handoff_to_boundary_delta.get("elapsed_perf_counter_ns"),
                int,
            )
            and handoff_to_boundary_delta["elapsed_perf_counter_ns"] >= 0
            and isinstance(handoff_to_boundary_delta.get("threads"), list)
            and isinstance(
                handoff_to_boundary_delta.get("aggregate_by_start_address"),
                list,
            )
        )
        if observe_handoff_thread_runtime
        else (
            handoff_thread_snapshot is None
            and handoff_to_boundary_delta is None
        )
    )
    main_thread_rows = (
        [
            row
            for row in boundary_threads
            if isinstance(row, Mapping)
            and row.get("thread_id") == bundle["main_thread_id"]
            and row.get("is_main_thread") is True
            and row.get("accessible") is True
        ]
        if isinstance(boundary_threads, list)
        else []
    )
    thread_runtime_contract_pass = (
        (
            isinstance(thread_runtime, Mapping)
            and thread_runtime.get("schema")
            == (
                "zuma-rl.pc-global-mtrand-boundary-"
                "thread-runtime-observation"
            )
            and thread_runtime.get("version") == 1
            and thread_runtime.get("enabled") is True
            and thread_runtime.get("process_memory_reads") == 0
            and thread_runtime.get("process_memory_writes") == 0
            and thread_runtime.get("process_context_reads") == 0
            and thread_runtime.get("process_context_writes") == 0
            and thread_runtime.get("handoff_snapshot_enabled")
            is observe_handoff_thread_runtime
            and isinstance(
                thread_runtime.get("lifecycle_events"),
                list,
            )
            and thread_runtime.get("lifecycle_event_count")
            == len(thread_runtime["lifecycle_events"])
            and isinstance(boundary_thread_snapshot, Mapping)
            and boundary_thread_snapshot.get("schema")
            == "zuma-rl.pc-thread-runtime-snapshot"
            and boundary_thread_snapshot.get("version") == 1
            and boundary_thread_snapshot.get("process_id")
            == bundle["pid"]
            and boundary_thread_snapshot.get("main_thread_id")
            == bundle["main_thread_id"]
            and boundary_thread_snapshot.get("phase")
            == "global_mtrand_boundary_pre_call"
            and boundary_thread_snapshot.get("process_memory_reads") == 0
            and boundary_thread_snapshot.get("process_memory_writes") == 0
            and boundary_thread_snapshot.get("process_context_reads") == 0
            and boundary_thread_snapshot.get("process_context_writes") == 0
            and isinstance(boundary_threads, list)
            and boundary_thread_snapshot.get("thread_count")
            == len(boundary_threads)
            and len(main_thread_rows) == 1
            and handoff_runtime_contract_pass
        )
        if observe_thread_runtime
        else thread_runtime is None
    )
    contract_pass = (
        isinstance(payload, Mapping)
        and payload.get("schema")
        == "zuma-rl.pc-global-mtrand-oracle-synchronizer"
        and payload.get("version") == 1
        and payload.get("status") == "PASS"
        and payload.get("failure") is None
        and payload.get("classification")
        == "diagnostic-read-only-global-rng-boundary-observation"
        and payload.get("mode") == "observe_boundary_only"
        and payload.get("process_id") == bundle["pid"]
        and payload.get("main_thread_id") == bundle["main_thread_id"]
        and Path(payload.get("runtime_executable", "")).resolve()
        == expected_runtime.resolve()
        and payload.get("runtime_executable_sha256")
        == _sha256_path(expected_runtime)
        and isinstance(oracle, Mapping)
        and Path(oracle.get("source_path", "")).resolve()
        == expected_oracle.resolve()
        and oracle.get("source_sha256") == _sha256_path(expected_oracle)
        and oracle.get("seed") == bundle["seed"]
        and oracle.get("start_after_update")
        == bundle["start_after_update"]
        and oracle.get("end_at_update") == bundle["end_at_update"]
        and oracle.get("entry_count") == 1
        and ready.get("expected_call_count") == 1
        and ready.get("mode") == "observe_boundary_only"
        and bool(
            ready.get("observe_handoff_thread_runtime", False)
        )
        is observe_handoff_thread_runtime
        and bool(ready.get("observe_handoff_rng_state", False))
        is observe_handoff_rng_state
        and bool(
            ready.get("handoff_rng_readiness_wait_enabled", False)
        )
        is handoff_rng_wait_enabled
        and ready.get("oracle_semantic_sha256")
        == oracle.get("semantic_sha256")
        and payload.get("expected_hit_count") == 1
        and payload.get("hit_count") == 1
        and payload.get("missing_hit_count") == 0
        and payload.get("unexpected_extra_call") is None
        and payload.get("stop_reason") == "boundary_observed"
        and payload.get("state_correction_count") == 0
        and payload.get("bytes_written_total") == 0
        and payload.get("process_memory_writes") == 0
        and payload.get("process_memory_mutation") is False
        and payload.get("process_context_mutation") is True
        and payload.get("persistent_file_modified") is False
        and payload.get("hardware_breakpoint_restored") is True
        and payload.get("hardware_breakpoint_restore_error") is None
        and payload.get("debugger_detach_error") is None
        and payload.get("resume_main_thread_on_ready") is True
        and payload.get("observe_handoff_thread_runtime")
        is observe_handoff_thread_runtime
        and bool(payload.get("observe_handoff_rng_state", False))
        is observe_handoff_rng_state
        and handoff_rng_contract_pass
        and bool(
            payload.get("handoff_rng_readiness_wait_enabled", False)
        )
        is handoff_rng_wait_enabled
        and handoff_rng_wait_contract_pass
        and payload.get("handoff_main_thread_resumed") is True
        and payload.get("handoff_resume_previous_suspend_count") == 1
        and payload.get("ready_receipt_schema")
        == "zuma-rl.pc-global-mtrand-oracle-synchronizer-ready"
        and payload.get("ready_receipt_version") == 2
        and payload.get("ready_receipt_published") is True
        and payload.get("ready_receipt_publication_order")
        == "AFTER_VERIFIED_HANDOFF_RESUME"
        and payload.get("ready_receipt_published_perf_counter_ns")
        == ready.get("published_perf_counter_ns")
        and isinstance(first_hit, Mapping)
        and first_hit.get("bytes_written") == 0
        and first_hit.get("changed") is False
        and first_hit.get("state_correction_applied") is False
        and isinstance(first_hit.get("live_pre_draw_count"), int)
        and not isinstance(first_hit.get("live_pre_draw_count"), bool)
        and isinstance(final_post, Mapping)
        and final_post.get("verified") is True
        and final_post.get("return_address_match") is True
        and isinstance(final_post.get("observed_post_draw_count"), int)
        and not isinstance(
            final_post.get("observed_post_draw_count"), bool
        )
        and isinstance(reconstruction, Mapping)
        and reconstruction.get("status") == "PASS"
        and isinstance(payload.get("natural_boundary_source_exact"), bool)
        and thread_runtime_contract_pass
        and process.returncode == 0
    )
    if not contract_pass:
        raise ProbeError(
            "global_mtrand_boundary_observation_contract_failed:"
            f"status={payload.get('status') if isinstance(payload, Mapping) else None}:"
            f"failure={payload.get('failure') if isinstance(payload, Mapping) else None}:"
            f"exit={process.returncode}"
        )
    evidence = {
        "schema": (
            "zuma-rl.pc-global-mtrand-boundary-observation-artifact"
        ),
        "version": 1,
        "status": "PASS",
        "classification": payload["classification"],
        "artifact": result_path.name,
        "artifact_bytes": len(data),
        "artifact_sha256": _sha256_bytes(data),
        "ready_artifact": Path(bundle["ready_path"]).name,
        "ready_artifact_sha256": _sha256_path(
            Path(bundle["ready_path"])
        ),
        "oracle": dict(oracle),
        "expected_hit_count": 1,
        "hit_count": 1,
        "state_correction_count": 0,
        "bytes_written_total": 0,
        "process_memory_writes": 0,
        "process_memory_mutation": False,
        "source_exact": payload["natural_boundary_source_exact"],
        "semantic_mismatch": payload.get("semantic_mismatch"),
        "first_hit": dict(first_hit),
        "final_post_verification": dict(final_post),
        "draw_count_reconstruction": dict(reconstruction),
        "hardware_breakpoint_restored": True,
        "handoff_main_thread_resumed": True,
        "ready_receipt_version": 2,
        "ready_receipt_publication_order": (
            "AFTER_VERIFIED_HANDOFF_RESUME"
        ),
        "ready_receipt_published_perf_counter_ns": ready[
            "published_perf_counter_ns"
        ],
        "persistent_file_modified": False,
    }
    if observe_thread_runtime:
        evidence["thread_runtime_observation"] = dict(thread_runtime)
    if observe_handoff_rng_state:
        evidence["handoff_rng_state_observation"] = dict(
            handoff_rng_state
        )
    if handoff_rng_wait_enabled:
        evidence["handoff_rng_readiness_wait"] = dict(
            handoff_rng_readiness_wait
        )
    return evidence


def _finish_global_mtrand_call_sync(
    bundle: Mapping[str, Any],
    *,
    ready: Mapping[str, Any],
) -> dict[str, Any]:
    if bundle.get("observe_boundary_only") is True:
        return _finish_global_mtrand_boundary_observation(
            bundle,
            ready=ready,
        )
    process = bundle["process"]
    log_stream = bundle["log_stream"]
    result_path = bundle["result_path"]
    if (
        not isinstance(process, subprocess.Popen)
        or not isinstance(result_path, Path)
    ):
        raise ProbeError("global_mtrand_call_sync_bundle_invalid")
    try:
        process.wait(timeout=float(bundle["timeout_seconds"]) + 10.0)
    except subprocess.TimeoutExpired as error:
        raise ProbeError("global_mtrand_call_sync_finish_timeout") from error
    finally:
        if not log_stream.closed:
            log_stream.close()
    if not result_path.is_file():
        raise ProbeError("global_mtrand_call_sync_result_missing")
    try:
        data = result_path.read_bytes()
        payload = json.loads(data.decode("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProbeError("global_mtrand_call_sync_result_invalid") from error
    oracle = payload.get("oracle") if isinstance(payload, Mapping) else None
    final_post = (
        payload.get("final_post_verification")
        if isinstance(payload, Mapping)
        else None
    )
    expected_hits = payload.get("expected_hit_count")
    correction_count = payload.get("state_correction_count")
    bytes_written = payload.get("bytes_written_total")
    expected_runtime = Path(bundle["runtime_executable"])
    expected_oracle = Path(bundle["oracle_path"])
    contract_pass = (
        isinstance(payload, Mapping)
        and payload.get("schema")
        == "zuma-rl.pc-global-mtrand-oracle-synchronizer"
        and payload.get("version") == 1
        and payload.get("status") == "PASS"
        and payload.get("failure") is None
        and payload.get("classification")
        == "diagnostic-ephemeral-global-rng-synchronization"
        and payload.get("process_id") == bundle["pid"]
        and payload.get("main_thread_id") == bundle["main_thread_id"]
        and Path(payload.get("runtime_executable", "")).resolve()
        == expected_runtime.resolve()
        and payload.get("runtime_executable_sha256")
        == _sha256_path(expected_runtime)
        and isinstance(oracle, Mapping)
        and Path(oracle.get("source_path", "")).resolve()
        == expected_oracle.resolve()
        and oracle.get("source_sha256") == _sha256_path(expected_oracle)
        and oracle.get("seed") == bundle["seed"]
        and oracle.get("start_after_update")
        == bundle["start_after_update"]
        and oracle.get("end_at_update") == bundle["end_at_update"]
        and oracle.get("entry_count") == expected_hits
        and ready.get("expected_call_count") == expected_hits
        and ready.get("oracle_semantic_sha256")
        == oracle.get("semantic_sha256")
        and isinstance(expected_hits, int)
        and not isinstance(expected_hits, bool)
        and expected_hits > 0
        and payload.get("hit_count") == expected_hits
        and payload.get("missing_hit_count") == 0
        and payload.get("semantic_mismatch") is None
        and payload.get("unexpected_extra_call") is None
        and payload.get("stop_reason") == "oracle_complete"
        and isinstance(correction_count, int)
        and not isinstance(correction_count, bool)
        and 0 <= correction_count <= expected_hits
        and isinstance(bytes_written, int)
        and not isinstance(bytes_written, bool)
        and bytes_written == correction_count * MTRAND_STATE_BYTES
        and payload.get("process_memory_mutation") == bool(bytes_written)
        and payload.get("persistent_file_modified") is False
        and payload.get("hardware_breakpoint_restored") is True
        and payload.get("hardware_breakpoint_restore_error") is None
        and payload.get("debugger_detach_error") is None
        and payload.get("resume_main_thread_on_ready") is True
        and payload.get("handoff_main_thread_resumed") is True
        and payload.get("handoff_resume_previous_suspend_count") == 1
        and payload.get("ready_receipt_schema")
        == "zuma-rl.pc-global-mtrand-oracle-synchronizer-ready"
        and payload.get("ready_receipt_version") == 2
        and payload.get("ready_receipt_published") is True
        and payload.get("ready_receipt_publication_order")
        == "AFTER_VERIFIED_HANDOFF_RESUME"
        and payload.get("ready_receipt_published_perf_counter_ns")
        == ready.get("published_perf_counter_ns")
        and isinstance(final_post, Mapping)
        and final_post.get("verified") is True
        and process.returncode == 0
    )
    if not contract_pass:
        raise ProbeError(
            "global_mtrand_call_sync_contract_failed:"
            f"status={payload.get('status') if isinstance(payload, Mapping) else None}:"
            f"failure={payload.get('failure') if isinstance(payload, Mapping) else None}:"
            f"exit={process.returncode}"
        )
    hits = payload.get("hits")
    if not isinstance(hits, list) or len(hits) != expected_hits:
        raise ProbeError("global_mtrand_call_sync_hits_invalid")
    return {
        "schema": "zuma-rl.pc-global-mtrand-call-sync-artifact",
        "version": 1,
        "status": "PASS",
        "classification": payload["classification"],
        "artifact": result_path.name,
        "artifact_bytes": len(data),
        "artifact_sha256": _sha256_bytes(data),
        "ready_artifact": Path(bundle["ready_path"]).name,
        "ready_artifact_sha256": _sha256_path(Path(bundle["ready_path"])),
        "oracle": dict(oracle),
        "expected_hit_count": expected_hits,
        "hit_count": expected_hits,
        "state_correction_count": correction_count,
        "bytes_written_total": bytes_written,
        "process_memory_mutation": bool(bytes_written),
        "first_hit": dict(hits[0]),
        "last_hit": dict(hits[-1]),
        "final_post_verification": dict(final_post),
        "hardware_breakpoint_restored": True,
        "handoff_main_thread_resumed": True,
        "ready_receipt_version": 2,
        "ready_receipt_publication_order": (
            "AFTER_VERIFIED_HANDOFF_RESUME"
        ),
        "ready_receipt_published_perf_counter_ns": ready[
            "published_perf_counter_ns"
        ],
        "persistent_file_modified": False,
    }


def _require_zero_write_global_mtrand_suffix(
    evidence: Mapping[str, Any],
) -> None:
    """Reject any post-initialization correction in the source suffix."""

    if (
        evidence.get("state_correction_count") != 0
        or evidence.get("bytes_written_total") != 0
        or evidence.get("process_memory_mutation") is not False
    ):
        raise ProbeError(
            "initial_global_mtrand_seed_suffix_not_zero_write"
        )


def _abort_global_mtrand_call_sync(bundle: Mapping[str, Any]) -> None:
    process = bundle.get("process")
    log_stream = bundle.get("log_stream")
    stop_path = bundle.get("stop_path")
    try:
        if isinstance(process, subprocess.Popen) and process.poll() is None:
            if isinstance(stop_path, Path) and not stop_path.exists():
                stop_path.write_text("stop\n", encoding="ascii")
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
    finally:
        if log_stream is not None and not log_stream.closed:
            log_stream.close()


def _attempt_failure_disposition(
    *,
    global_mtrand_call_requested: bool,
    global_mtrand_handoff_ready_observed: bool,
) -> tuple[str, str | None]:
    """Make the global-call retry boundary explicit and testable."""

    if global_mtrand_call_requested:
        if global_mtrand_handoff_ready_observed:
            return "FAIL", "post_handoff_nonretryable"
        return "RETRY", "pre_handoff_startup_infrastructure"
    return "RETRY", None


def _stop_children(
    trace_process: subprocess.Popen[bytes] | None,
    runtime_pid: int | None,
    runtime_executable: Path,
) -> None:
    if runtime_pid is not None:
        try:
            _terminate_exact_runtime(runtime_pid, runtime_executable)
        except Exception:
            pass
    if trace_process is not None and trace_process.poll() is None:
        trace_process.terminate()
        try:
            trace_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            trace_process.kill()
            trace_process.wait(timeout=10)
    deadline = time.monotonic() + 10.0
    while process_ids_by_name(runtime_executable.name):
        if time.monotonic() >= deadline:
            break
        time.sleep(0.05)


def _board_thread_id_from_trace_log(path: Path) -> int:
    """Return the unique thread that executed the proven Board reseed call."""

    text = path.read_text(encoding="utf-8", errors="replace")
    matches = {
        int(value)
        for value in re.findall(
            r"(?m)^board_seed_entry tid=(\d+)\b",
            text,
        )
    }
    if len(matches) != 1:
        raise ProbeError(
            "board_thread_id_not_unique:"
            f"observed={sorted(matches)}"
        )
    return next(iter(matches))


def _window_thread_id(window_handle: int, expected_pid: int) -> int:
    if os.name != "nt":
        raise ProbeError("window_thread_requires_windows")
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    get_thread = user32.GetWindowThreadProcessId
    get_thread.argtypes = [
        wintypes.HWND,
        ctypes.POINTER(wintypes.DWORD),
    ]
    get_thread.restype = wintypes.DWORD
    observed_pid = wintypes.DWORD()
    thread_id = int(
        get_thread(
            wintypes.HWND(window_handle),
            ctypes.byref(observed_pid),
        )
    )
    if thread_id <= 0 or int(observed_pid.value) != expected_pid:
        raise ProbeError("window_thread_identity_mismatch")
    return thread_id


def _observe_process_affinity(
    pid: int,
    mask: int,
    *,
    timeout_seconds: float = 5.0,
) -> dict[str, int | bool | str]:
    """Wait for and verify the launcher's pre-resume affinity receipt.

    The parent collector can discover the CREATE_SUSPENDED process in the
    narrow interval between CreateProcessW returning in the child tracer and
    that tracer applying the requested mask.  A single immediate read is
    therefore a race, not evidence that an already-applied mask was reset.
    """

    if os.name != "nt":
        raise ProbeError("process_affinity_requires_windows")
    if (
        pid <= 0
        or mask <= 0
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0.0
    ):
        raise ProbeError("process_affinity_invalid")
    process_query_limited_information = 0x1000
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    open_process = kernel32.OpenProcess
    open_process.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    open_process.restype = wintypes.HANDLE
    get_affinity = kernel32.GetProcessAffinityMask
    get_affinity.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    get_affinity.restype = wintypes.BOOL
    handle = open_process(
        process_query_limited_information,
        False,
        pid,
    )
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        deadline = time.monotonic() + timeout_seconds
        started_ns = time.perf_counter_ns()
        first_observed_mask: int | None = None
        attempt_count = 0
        while True:
            observed = ctypes.c_size_t()
            system = ctypes.c_size_t()
            if not get_affinity(
                handle,
                ctypes.byref(observed),
                ctypes.byref(system),
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            attempt_count += 1
            observed_mask = int(observed.value)
            system_mask = int(system.value)
            if first_observed_mask is None:
                first_observed_mask = observed_mask
            if mask & ~system_mask:
                raise ProbeError("process_affinity_exceeds_system_mask")
            if observed_mask == mask:
                return {
                    "process_id": pid,
                    "requested_process_mask": mask,
                    "first_observed_process_mask": first_observed_mask,
                    "observed_process_mask": observed_mask,
                    "system_mask": system_mask,
                    "observation_attempt_count": attempt_count,
                    "observation_wait_perf_counter_ns": (
                        time.perf_counter_ns() - started_ns
                    ),
                    "single_logical_processor": mask.bit_count() == 1,
                    "application_stage": (
                        "created_suspended_before_first_resume"
                    ),
                    "verification_stage": (
                        "collector_waited_for_startup_affinity_apply"
                    ),
                    "persistent_host_modification": False,
                }
            if time.monotonic() >= deadline:
                raise ProbeError(
                    "process_affinity_verify_failed:"
                    f"pid={pid}:requested=0x{mask:X}:"
                    f"first_observed=0x{first_observed_mask:X}:"
                    f"observed=0x{observed_mask:X}:"
                    f"system=0x{system_mask:X}:"
                    f"attempts={attempt_count}"
                )
            time.sleep(0.01)
    finally:
        kernel32.CloseHandle(handle)


def _resolve_thread_crt_state(
    *,
    pid: int,
    thread_id: int,
) -> dict[str, int]:
    """Resolve one live WOW64 thread's CRT PTD without mutating game state."""

    if os.name != "nt":
        raise ProbeError("thread_crt_state_requires_windows")
    from tools import trace_popcap_demo_commands as trace

    process = open_process_readonly(pid)
    thread = trace.kernel32.OpenThread(
        trace.THREAD_ACCESS | trace.THREAD_SUSPEND_RESUME,
        False,
        thread_id,
    )
    if not thread:
        close_process(process)
        raise ctypes.WinError(ctypes.get_last_error())
    suspended = False
    try:
        previous_suspend_count = int(trace.kernel32.SuspendThread(thread))
        if previous_suspend_count == trace.INVALID_SUSPEND_COUNT:
            raise ctypes.WinError(ctypes.get_last_error())
        suspended = True
        context = trace.WOW64_CONTEXT()
        context.ContextFlags = trace.CONTEXT_FULL
        if not trace.kernel32.Wow64GetThreadContext(
            thread,
            ctypes.byref(context),
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        state = trace._thread_crt_rng_state(
            process=process,
            thread=thread,
            context=context,
            expected_thread_id=thread_id,
        )
        return {
            **{key: int(value) for key, value in state.items()},
            "thread_id": thread_id,
        }
    finally:
        if suspended:
            result = int(trace.kernel32.ResumeThread(thread))
            if result == trace.INVALID_SUSPEND_COUNT:
                trace.kernel32.CloseHandle(thread)
                close_process(process)
                raise ctypes.WinError(ctypes.get_last_error())
        trace.kernel32.CloseHandle(thread)
        close_process(process)


def _resolve_board_thread_crt_state(
    *,
    pid: int,
    window_handle: int,
    trace_log_path: Path,
) -> dict[str, int]:
    """Resolve the replay Board thread and prove its CRT rand-state address."""

    candidates: list[int] = []
    errors: list[str] = []
    try:
        candidates.append(_board_thread_id_from_trace_log(trace_log_path))
    except Exception as error:
        errors.append(
            "trace_log:"
            f"{type(error).__name__}:{str(error)[:160]}"
        )
    try:
        window_thread = _window_thread_id(window_handle, pid)
        if window_thread not in candidates:
            candidates.append(window_thread)
    except Exception as error:
        errors.append(
            "window:"
            f"{type(error).__name__}:{str(error)[:160]}"
        )

    for thread_id in candidates:
        try:
            return _resolve_thread_crt_state(
                pid=pid,
                thread_id=thread_id,
            )
        except Exception as error:
            errors.append(
                f"thread_{thread_id}:"
                f"{type(error).__name__}:{str(error)[:160]}"
            )
    raise ProbeError(
        "board_thread_crt_state_unresolved:"
        + "|".join(errors)
    )


def _start_global_rng_call_trace(
    *,
    project_root: Path,
    pid: int,
    runtime_executable: Path,
    attempt_root: Path,
    start_update: int,
    end_update: int,
    timeout_seconds: float,
) -> tuple[
    subprocess.Popen[bytes],
    Any,
    Path,
    Path,
    Path,
    Path,
]:
    """Start the narrow dynamic caller trace and wait until INT3 is armed."""

    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(float(timeout_seconds))
        or timeout_seconds <= 0
    ):
        raise ProbeError("global_rng_call_trace_timeout_invalid")

    result_path = attempt_root / "global-rng-call-trace.json"
    ready_path = attempt_root / "global-rng-call-trace.ready.json"
    stop_path = attempt_root / "global-rng-call-trace.stop"
    log_path = attempt_root / "global-rng-call-trace.log"
    log_stream = log_path.open("xb")
    process: subprocess.Popen[bytes] | None = None
    try:
        process = subprocess.Popen(
            [
                sys.executable,
                str(
                    project_root
                    / "tools"
                    / "trace_popcap_global_rng_calls.py"
                ),
                "--pid",
                str(pid),
                "--executable",
                str(runtime_executable),
                "--output",
                str(result_path),
                "--ready",
                str(ready_path),
                "--stop",
                str(stop_path),
                "--start-update",
                str(start_update),
                "--end-update",
                str(end_update),
                "--timeout",
                str(float(timeout_seconds)),
            ],
            cwd=project_root,
            env=_child_environment(project_root),
            stdin=subprocess.DEVNULL,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            close_fds=True,
        )
        deadline = time.monotonic() + 15.0
        while not ready_path.is_file():
            if process.poll() is not None:
                raise ProbeError(
                    "global_rng_call_trace_exited_before_ready:"
                    f"returncode={process.returncode}"
                )
            if time.monotonic() >= deadline:
                raise ProbeError("global_rng_call_trace_ready_timeout")
            time.sleep(0.02)
        return (
            process,
            log_stream,
            result_path,
            ready_path,
            stop_path,
            log_path,
        )
    except Exception:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        log_stream.close()
        raise


def _finish_global_rng_call_trace(
    *,
    process: subprocess.Popen[bytes],
    log_stream: Any,
    result_path: Path,
    stop_path: Path,
    log_path: Path,
    start_update: int,
    end_update: int,
) -> dict[str, Any]:
    """Request a clean breakpoint restore, then bind the trace artifacts."""

    stop_path.write_text("stop\n", encoding="ascii")
    try:
        return_code = process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
        raise ProbeError("global_rng_call_trace_stop_timeout")
    finally:
        log_stream.close()
    if return_code != 0 or not result_path.is_file():
        raise ProbeError(
            "global_rng_call_trace_failed:"
            f"returncode={return_code}"
        )
    try:
        result = json.loads(result_path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, ValueError) as error:
        raise ProbeError("global_rng_call_trace_result_invalid") from error
    if (
        result.get("schema")
        != "zuma-rl.pc-global-mtrand-call-trace"
        or result.get("version") != 1
        or result.get("status") != "PASS"
        or result.get("stop_reason") != "stop_requested"
        or result.get("start_update") != start_update
        or result.get("end_update") != end_update
        or result.get("call_count") != len(result.get("calls", ()))
    ):
        raise ProbeError("global_rng_call_trace_contract_invalid")
    return {
        "schema": result["schema"],
        "version": result["version"],
        "status": result["status"],
        "artifact": result_path.name,
        "artifact_sha256": _sha256_path(result_path),
        "log_artifact": log_path.name,
        "log_artifact_sha256": _sha256_path(log_path),
        "start_update": start_update,
        "end_update": end_update,
        "call_count": int(result["call_count"]),
    }


def _start_live_rng_monitor(
    *,
    project_root: Path,
    pid: int,
    attempt_root: Path,
    interval_seconds: float,
    thread_id: int,
) -> dict[str, Any]:
    """Start the read-only live RNG monitor and wait for its first flush."""

    output_path = attempt_root / "live-rng.ndjson"
    stop_path = attempt_root / "live-rng.stop"
    stdout_path = attempt_root / "live-rng-monitor.log"
    stderr_path = attempt_root / "live-rng-monitor.err.log"
    stdout_stream = stdout_path.open("xb")
    stderr_stream = stderr_path.open("xb")
    process: subprocess.Popen[bytes] | None = None
    try:
        process = subprocess.Popen(
            [
                sys.executable,
                str(project_root / "tools" / "monitor_live_zuma_rng.py"),
                "--pid",
                str(pid),
                "--output",
                str(output_path),
                "--interval-seconds",
                str(interval_seconds),
                "--maximum-seconds",
                "900",
                "--thread-id",
                str(thread_id),
                "--stop-file",
                str(stop_path),
            ],
            cwd=project_root,
            env=_child_environment(project_root),
            stdin=subprocess.DEVNULL,
            stdout=stdout_stream,
            stderr=stderr_stream,
            close_fds=True,
        )
        deadline = time.monotonic() + 15.0
        while not output_path.is_file() or output_path.stat().st_size == 0:
            if process.poll() is not None:
                raise ProbeError(
                    "live_rng_monitor_exited_before_ready:"
                    f"returncode={process.returncode}"
                )
            if time.monotonic() >= deadline:
                raise ProbeError("live_rng_monitor_ready_timeout")
            time.sleep(0.02)
        return {
            "process": process,
            "pid": pid,
            "stdout_stream": stdout_stream,
            "stderr_stream": stderr_stream,
            "output_path": output_path,
            "stop_path": stop_path,
            "stdout_path": stdout_path,
            "stderr_path": stderr_path,
            "interval_seconds": interval_seconds,
            "thread_id": thread_id,
        }
    except Exception:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        stdout_stream.close()
        stderr_stream.close()
        raise


def _finish_live_rng_monitor(bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Stop the read-only monitor cleanly and bind all diagnostic artifacts."""

    process = bundle["process"]
    stop_path = bundle["stop_path"]
    stop_path.write_text("stop\n", encoding="ascii")
    try:
        return_code = process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
        raise ProbeError("live_rng_monitor_stop_timeout")
    finally:
        bundle["stdout_stream"].close()
        bundle["stderr_stream"].close()
    output_path = bundle["output_path"]
    stdout_path = bundle["stdout_path"]
    stderr_path = bundle["stderr_path"]
    if return_code != 0 or not output_path.is_file():
        raise ProbeError(
            "live_rng_monitor_failed:"
            f"returncode={return_code}"
        )
    try:
        log_rows = [
            row
            for row in stdout_path.read_text(
                encoding="utf-8",
                errors="strict",
            ).splitlines()
            if row
        ]
        result = json.loads(log_rows[-1])
    except (IndexError, OSError, UnicodeError, ValueError) as error:
        raise ProbeError("live_rng_monitor_result_invalid") from error
    if (
        result.get("status") != "PASS"
        or result.get("process_id") != bundle["pid"]
    ):
        raise ProbeError("live_rng_monitor_result_invalid")
    if result.get("stopped_by_file") is not True:
        raise ProbeError("live_rng_monitor_stop_unverified")
    return {
        "schema": "zuma-rl.pc-live-rng-monitor-evidence",
        "version": 1,
        "classification": "read-only-localization-diagnostic",
        "artifact": output_path.name,
        "artifact_sha256": _sha256_path(output_path),
        "stdout_artifact": stdout_path.name,
        "stdout_sha256": _sha256_path(stdout_path),
        "stderr_artifact": stderr_path.name,
        "stderr_sha256": _sha256_path(stderr_path),
        "stop_artifact": stop_path.name,
        "stop_sha256": _sha256_path(stop_path),
        "interval_seconds": bundle["interval_seconds"],
        "thread_id": bundle["thread_id"],
        "sample_count": int(result["sample_count"]),
        "change_count": int(result["change_count"]),
        "unavailable_count": int(result["unavailable_count"]),
        "process_memory_writes": 0,
    }


def _abort_live_rng_monitor(bundle: Mapping[str, Any]) -> None:
    """Best-effort cleanup used only while an attempt is already failing."""

    process = bundle["process"]
    try:
        if process.poll() is None:
            bundle["stop_path"].write_text("stop\n", encoding="ascii")
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
    finally:
        if not bundle["stdout_stream"].closed:
            bundle["stdout_stream"].close()
        if not bundle["stderr_stream"].closed:
            bundle["stderr_stream"].close()


def collect_probe(
    *,
    plan_path: Path,
    prestate_path: Path,
    host_restore_path: Path,
    output_root: Path,
    probe_update: int,
    slowdown_update: int,
    int32_value: int | None,
    maximum_attempts: int,
    displayed_score_value: int | None = None,
    int32_scan_value: int | None = None,
    skip_int32_scan: bool = False,
    post_mutation_score_value: int | None = None,
    post_mutation_displayed_score_value: int | None = None,
    allow_post_mutation_score_change: bool = False,
    trajectory_end_update: int | None = None,
    trajectory_start_update: int | None = None,
    trajectory_mode: str = "full",
    trace_global_rng_calls: bool = False,
    global_rng_call_trace_timeout_seconds: float = 180.0,
    trajectory_step_timeout_seconds: float = 5.0,
    skip_repaint_guard: bool = False,
    skip_frozen_snapshot: bool = False,
    snapshot_trajectory_frames: bool = False,
    snapshot_square_dwm_corners: bool = False,
    snapshot_trajectory_warmup_frame: bool = False,
    formal_exact_step_evidence: bool = False,
    formal_full_state_evidence: bool = False,
    diagnostic_source_bound_replay: bool = False,
    formal_fruit_lifecycle_trigger: Mapping[str, Any] | None = None,
    trace_detach_update: int | None = None,
    trace_reattach_update: int | None = None,
    allow_discovered_display_mismatch: bool = False,
    formal_independent_score_binding: bool = False,
    live_rng_monitor_interval_seconds: float | None = None,
    gameplay_mtrand_oracle: Path | None = None,
    gameplay_mtrand_oracle_seed: int | None = None,
    gameplay_mtrand_oracle_maximum_draws: int = 100_000,
    global_mtrand_call_oracle: Path | None = None,
    global_mtrand_call_oracle_seed: int | None = None,
    global_mtrand_call_start_after_update: int | None = None,
    global_mtrand_call_end_at_update: int | None = None,
    global_mtrand_call_maximum_draws: int = 100_000,
    global_mtrand_call_sync_timeout_seconds: float = 900.0,
    global_mtrand_call_observe_boundary_only: bool = False,
    global_mtrand_call_observe_thread_runtime: bool = False,
    global_mtrand_call_observe_handoff_thread_runtime: bool = False,
    global_mtrand_call_observe_handoff_rng_state: bool = False,
    global_mtrand_call_handoff_rng_wait_target_words_sha256: (
        str | None
    ) = None,
    global_mtrand_call_handoff_rng_wait_min_index: int | None = None,
    global_mtrand_call_handoff_rng_wait_timeout_seconds: (
        float | None
    ) = None,
    global_mtrand_call_handoff_rng_wait_poll_interval_seconds: (
        float | None
    ) = None,
    initial_global_mtrand_oracle: Path | None = None,
    initial_global_mtrand_seed: int | None = None,
    startup_global_mtrand_observation_oracle: Path | None = None,
    startup_global_mtrand_observation_seed: int | None = None,
    startup_global_mtrand_observation_end_at_update: int | None = None,
    startup_global_mtrand_observation_maximum_draws: int = 100_000,
    startup_global_mtrand_hidden_draw_start_after_source_order: (
        int | None
    ) = None,
    startup_global_mtrand_hidden_draw_stop_before_source_order: (
        int | None
    ) = None,
    diagnostic_disable_source_bound_board_anchor: bool = False,
    startup_priority_bias_until_update: int | None = None,
    process_affinity_mask: int | None = None,
    diagnostic_startup_crt_seed: int | None = None,
    diagnostic_source_bound_compact_restore: bool = False,
    diagnostic_mutator: (
        Callable[
            [int, Mapping[str, Any], Path],
            Mapping[str, Any],
        ]
        | None
    ) = None,
    diagnostic_observer: (
        Callable[
            [int, Mapping[str, Any], Path],
            Mapping[str, Any],
        ]
        | None
    ) = None,
) -> Path:
    if output_root.exists() or not output_root.parent.is_dir():
        raise ProbeError("output_root_invalid")
    if formal_exact_step_evidence and formal_full_state_evidence:
        raise ProbeError("formal_evidence_mode_conflict")
    formal_source_evidence = (
        formal_exact_step_evidence or formal_full_state_evidence
    )
    source_bound_trace_slice = bool(
        diagnostic_source_bound_replay
        and trace_global_rng_calls
        and trajectory_start_update is not None
        and skip_repaint_guard
        and skip_frozen_snapshot
    )
    source_bound_locator = bool(
        diagnostic_source_bound_replay
        and not trace_global_rng_calls
        and trajectory_start_update is None
        and not skip_repaint_guard
        and not skip_frozen_snapshot
    )
    if diagnostic_source_bound_replay and (
        formal_source_evidence
        or trajectory_mode != "rng"
        or trajectory_end_update is None
        or maximum_attempts != 1
        or not (source_bound_locator or source_bound_trace_slice)
        or snapshot_trajectory_frames
        or snapshot_square_dwm_corners
        or snapshot_trajectory_warmup_frame
        or live_rng_monitor_interval_seconds is not None
        or gameplay_mtrand_oracle is not None
        or global_mtrand_call_oracle is not None
        or initial_global_mtrand_oracle is not None
        or startup_global_mtrand_observation_oracle is not None
        or diagnostic_disable_source_bound_board_anchor
        or startup_priority_bias_until_update is not None
        or process_affinity_mask is not None
        or diagnostic_startup_crt_seed is not None
        or diagnostic_mutator is not None
        or diagnostic_observer is not None
    ):
        raise ProbeError("diagnostic_source_bound_replay_contract_invalid")
    normalized_formal_fruit_trigger = (
        _normalize_formal_fruit_lifecycle_trigger(
            formal_fruit_lifecycle_trigger
        )
        if formal_fruit_lifecycle_trigger is not None
        else None
    )
    if normalized_formal_fruit_trigger is not None and (
        not formal_full_state_evidence
        or formal_exact_step_evidence
        or maximum_attempts != 1
        or trajectory_mode != "full"
        or trajectory_start_update is not None
        or trajectory_end_update is None
        or probe_update
        != normalized_formal_fruit_trigger["monitor_start_update"]
        or trajectory_end_update
        != (
            probe_update
            + normalized_formal_fruit_trigger["trajectory_tick_count"]
            - 1
        )
    ):
        raise ProbeError("formal_fruit_trigger_contract_invalid")
    if formal_independent_score_binding and (
        not formal_full_state_evidence
        or int32_value is not None
        or displayed_score_value is not None
        or int32_scan_value is not None
        or not allow_discovered_display_mismatch
        or maximum_attempts != 1
    ):
        raise ProbeError("formal_independent_score_contract_invalid")
    if trajectory_mode not in {"full", "rng"}:
        raise ProbeError("trajectory_mode_invalid")
    if (
        isinstance(global_rng_call_trace_timeout_seconds, bool)
        or not isinstance(
            global_rng_call_trace_timeout_seconds,
            (int, float),
        )
        or not math.isfinite(float(global_rng_call_trace_timeout_seconds))
        or global_rng_call_trace_timeout_seconds <= 0
    ):
        raise ProbeError("global_rng_call_trace_timeout_invalid")
    if (
        isinstance(trajectory_step_timeout_seconds, bool)
        or not isinstance(trajectory_step_timeout_seconds, (int, float))
        or not math.isfinite(float(trajectory_step_timeout_seconds))
        or not 0.0 < float(trajectory_step_timeout_seconds) <= 3600.0
    ):
        raise ProbeError("trajectory_step_timeout_invalid")
    if skip_int32_scan and (
        not formal_exact_step_evidence
        or int32_value is not None
        or int32_scan_value is not None
        or trajectory_mode != "rng"
        or trajectory_start_update is None
        or trajectory_end_update is None
    ):
        raise ProbeError("int32_scan_skip_contract_invalid")
    if trace_global_rng_calls and (
        trajectory_mode != "rng" or trajectory_end_update is None
    ):
        raise ProbeError("global_rng_call_trace_requires_rng_trajectory")
    if trajectory_start_update is not None and (
        trajectory_mode != "rng"
        or trajectory_end_update is None
        or trajectory_start_update < probe_update
        or trajectory_start_update > trajectory_end_update
        or not skip_repaint_guard
        or not skip_frozen_snapshot
    ):
        raise ProbeError(
            "trajectory_record_start_requires_diagnostic_rng_trajectory"
        )
    if startup_priority_bias_until_update is not None and (
        startup_priority_bias_until_update <= 0
        or trajectory_end_update is None
        or not skip_repaint_guard
        or not skip_frozen_snapshot
    ):
        raise ProbeError(
            "startup_priority_bias_requires_diagnostic_trajectory"
        )
    if process_affinity_mask is not None and (
        process_affinity_mask <= 0
        or trajectory_end_update is None
        or not skip_repaint_guard
        or not skip_frozen_snapshot
    ):
        raise ProbeError("process_affinity_requires_diagnostic_trajectory")
    if diagnostic_source_bound_compact_restore and (
        diagnostic_startup_crt_seed is None
        or diagnostic_mutator is None
    ):
        raise ProbeError(
            "source_bound_compact_restore_contract_incomplete"
        )
    if diagnostic_startup_crt_seed is not None and (
        isinstance(diagnostic_startup_crt_seed, bool)
        or not isinstance(diagnostic_startup_crt_seed, int)
        or not 0 <= diagnostic_startup_crt_seed <= 0xFFFFFFFF
        or trajectory_mode != "rng"
        or trajectory_start_update is None
        or trajectory_end_update is None
        or not trace_global_rng_calls
        or not skip_repaint_guard
        or not skip_frozen_snapshot
        or int32_value is None
        or (
            diagnostic_mutator is not None
            and not diagnostic_source_bound_compact_restore
        )
    ):
        raise ProbeError(
            "startup_crt_seed_requires_clean_diagnostic_trajectory"
        )
    if snapshot_trajectory_frames and (
        trajectory_mode != "rng"
        or trajectory_end_update is None
        or not skip_repaint_guard
        or not skip_frozen_snapshot
    ):
        raise ProbeError(
            "trajectory_snapshots_require_diagnostic_rng_trajectory"
        )
    if snapshot_square_dwm_corners and not snapshot_trajectory_frames:
        raise ProbeError("trajectory_square_corners_require_snapshots")
    if snapshot_trajectory_warmup_frame and not snapshot_trajectory_frames:
        raise ProbeError("trajectory_warmup_requires_snapshots")
    global_mtrand_call_values = (
        global_mtrand_call_oracle,
        global_mtrand_call_oracle_seed,
        global_mtrand_call_start_after_update,
        global_mtrand_call_end_at_update,
    )
    global_mtrand_call_requested = any(
        value is not None for value in global_mtrand_call_values
    )
    if (
        global_mtrand_call_observe_boundary_only
        and not global_mtrand_call_requested
    ):
        raise ProbeError(
            "global_mtrand_boundary_observation_requires_call_oracle"
        )
    if (
        global_mtrand_call_observe_thread_runtime
        and not global_mtrand_call_observe_boundary_only
    ):
        raise ProbeError(
            "global_mtrand_thread_runtime_requires_boundary_observer"
        )
    if (
        global_mtrand_call_observe_handoff_thread_runtime
        and not global_mtrand_call_observe_thread_runtime
    ):
        raise ProbeError(
            "global_mtrand_handoff_runtime_requires_thread_runtime"
        )
    if (
        global_mtrand_call_observe_handoff_rng_state
        and not global_mtrand_call_observe_boundary_only
    ):
        raise ProbeError(
            "global_mtrand_handoff_rng_requires_boundary_observer"
        )
    global_mtrand_call_handoff_rng_wait_values = (
        global_mtrand_call_handoff_rng_wait_target_words_sha256,
        global_mtrand_call_handoff_rng_wait_min_index,
        global_mtrand_call_handoff_rng_wait_timeout_seconds,
        global_mtrand_call_handoff_rng_wait_poll_interval_seconds,
    )
    global_mtrand_call_handoff_rng_wait_enabled = any(
        value is not None
        for value in global_mtrand_call_handoff_rng_wait_values
    )
    if global_mtrand_call_handoff_rng_wait_enabled and (
        any(
            value is None
            for value in global_mtrand_call_handoff_rng_wait_values
        )
        or not global_mtrand_call_observe_boundary_only
        or not global_mtrand_call_observe_handoff_rng_state
        or not _is_canonical_sha256(
            global_mtrand_call_handoff_rng_wait_target_words_sha256
        )
        or isinstance(
            global_mtrand_call_handoff_rng_wait_min_index,
            bool,
        )
        or not isinstance(
            global_mtrand_call_handoff_rng_wait_min_index,
            int,
        )
        or not 0
        <= global_mtrand_call_handoff_rng_wait_min_index
        <= MTRAND_STATE_WORDS
        or isinstance(
            global_mtrand_call_handoff_rng_wait_timeout_seconds,
            bool,
        )
        or not isinstance(
            global_mtrand_call_handoff_rng_wait_timeout_seconds,
            (int, float),
        )
        or global_mtrand_call_handoff_rng_wait_timeout_seconds <= 0
        or isinstance(
            global_mtrand_call_handoff_rng_wait_poll_interval_seconds,
            bool,
        )
        or not isinstance(
            global_mtrand_call_handoff_rng_wait_poll_interval_seconds,
            (int, float),
        )
        or global_mtrand_call_handoff_rng_wait_poll_interval_seconds <= 0
        or global_mtrand_call_handoff_rng_wait_poll_interval_seconds
        > global_mtrand_call_handoff_rng_wait_timeout_seconds
    ):
        raise ProbeError("global_mtrand_handoff_rng_wait_options_invalid")
    initial_global_mtrand_values = (
        initial_global_mtrand_oracle,
        initial_global_mtrand_seed,
    )
    initial_global_mtrand_requested = any(
        value is not None for value in initial_global_mtrand_values
    )
    startup_global_mtrand_observation_values = (
        startup_global_mtrand_observation_oracle,
        startup_global_mtrand_observation_seed,
        startup_global_mtrand_observation_end_at_update,
    )
    startup_global_mtrand_observation_requested = any(
        value is not None
        for value in startup_global_mtrand_observation_values
    )
    startup_global_mtrand_hidden_draw_values = (
        startup_global_mtrand_hidden_draw_start_after_source_order,
        startup_global_mtrand_hidden_draw_stop_before_source_order,
    )
    startup_global_mtrand_hidden_draw_requested = any(
        value is not None
        for value in startup_global_mtrand_hidden_draw_values
    )
    if startup_global_mtrand_hidden_draw_requested and (
        any(
            value is None
            for value in startup_global_mtrand_hidden_draw_values
        )
        or not startup_global_mtrand_observation_requested
    ):
        raise ProbeError(
            "startup_global_mtrand_hidden_draw_options_incomplete"
        )
    if formal_exact_step_evidence and (
        trajectory_mode != "rng"
        or trajectory_end_update is None
        or trajectory_start_update is None
        or not skip_repaint_guard
        or not skip_frozen_snapshot
        or not snapshot_trajectory_frames
        or not snapshot_square_dwm_corners
        or not snapshot_trajectory_warmup_frame
        or trace_global_rng_calls
        or trace_detach_update is not None
        or trace_reattach_update is not None
        or (
            allow_discovered_display_mismatch
            and not formal_independent_score_binding
        )
        or live_rng_monitor_interval_seconds is not None
        or gameplay_mtrand_oracle is not None
        or gameplay_mtrand_oracle_seed is not None
        or global_mtrand_call_requested
        or initial_global_mtrand_requested
        or startup_global_mtrand_observation_requested
        or diagnostic_disable_source_bound_board_anchor
        or startup_priority_bias_until_update is not None
        or process_affinity_mask is not None
        or diagnostic_startup_crt_seed is not None
        or diagnostic_mutator is not None
        or diagnostic_observer is not None
    ):
        raise ProbeError("formal_exact_step_contract_incomplete")
    if formal_full_state_evidence and (
        trajectory_mode != "full"
        or trajectory_end_update is None
        or trajectory_start_update is not None
        or skip_repaint_guard
        or skip_frozen_snapshot
        or snapshot_trajectory_frames
        or snapshot_square_dwm_corners
        or snapshot_trajectory_warmup_frame
        or trace_global_rng_calls
        or trace_detach_update is not None
        or trace_reattach_update is not None
        or (
            allow_discovered_display_mismatch
            and not formal_independent_score_binding
        )
        or live_rng_monitor_interval_seconds is not None
        or gameplay_mtrand_oracle is not None
        or gameplay_mtrand_oracle_seed is not None
        or global_mtrand_call_requested
        or initial_global_mtrand_requested
        or startup_global_mtrand_observation_requested
        or diagnostic_disable_source_bound_board_anchor
        or startup_priority_bias_until_update is not None
        or process_affinity_mask is not None
        or diagnostic_startup_crt_seed is not None
        or diagnostic_mutator is not None
        or diagnostic_observer is not None
    ):
        raise ProbeError("formal_full_state_contract_incomplete")
    if skip_frozen_snapshot and (
        not skip_repaint_guard or trajectory_end_update is None
    ):
        raise ProbeError(
            "skip_frozen_snapshot_requires_diagnostic_trajectory"
        )
    if (
        allow_discovered_display_mismatch
        and not formal_independent_score_binding
        and (
            int32_value is not None
            or not skip_repaint_guard
            or not skip_frozen_snapshot
            or trajectory_end_update is None
        )
    ):
        raise ProbeError(
            "rolling_score_discovery_requires_diagnostic_trajectory"
        )
    if live_rng_monitor_interval_seconds is not None and (
        not 0 < live_rng_monitor_interval_seconds <= 1.0
        or not skip_repaint_guard
        or not skip_frozen_snapshot
        or trajectory_end_update is None
    ):
        raise ProbeError(
            "live_rng_monitor_requires_diagnostic_trajectory"
        )
    gameplay_mtrand_requested = (
        gameplay_mtrand_oracle is not None
        or gameplay_mtrand_oracle_seed is not None
    )
    if gameplay_mtrand_requested and (
        gameplay_mtrand_oracle is None
        or gameplay_mtrand_oracle_seed is None
        or not 0 <= gameplay_mtrand_oracle_seed <= 0xFFFFFFFF
        or gameplay_mtrand_oracle_maximum_draws <= 0
        or not skip_repaint_guard
        or not skip_frozen_snapshot
        or trajectory_end_update is None
    ):
        raise ProbeError(
            "gameplay_mtrand_sync_requires_diagnostic_trajectory"
        )
    if (
        gameplay_mtrand_oracle is not None
        and not gameplay_mtrand_oracle.is_file()
    ):
        raise ProbeError("gameplay_mtrand_oracle_missing")
    if global_mtrand_call_requested and (
        any(value is None for value in global_mtrand_call_values)
        or global_mtrand_call_oracle is None
        or not global_mtrand_call_oracle.is_file()
        or global_mtrand_call_oracle_seed is None
        or not 0 <= global_mtrand_call_oracle_seed <= 0xFFFFFFFF
        or global_mtrand_call_start_after_update is None
        or global_mtrand_call_start_after_update < 0
        or global_mtrand_call_end_at_update is None
        or global_mtrand_call_end_at_update
        <= global_mtrand_call_start_after_update
        or global_mtrand_call_maximum_draws <= 0
        or global_mtrand_call_sync_timeout_seconds <= 0
        or trajectory_mode != "rng"
        or trajectory_end_update is None
        or (
            not global_mtrand_call_observe_boundary_only
            and global_mtrand_call_end_at_update
            != trajectory_end_update
        )
        or (
            global_mtrand_call_observe_boundary_only
            and global_mtrand_call_end_at_update > trajectory_end_update
        )
        or trace_detach_update is None
        or trace_detach_update > global_mtrand_call_start_after_update
        or trace_reattach_update is None
        or trace_reattach_update <= trajectory_end_update
        or not skip_repaint_guard
        or not skip_frozen_snapshot
        or gameplay_mtrand_requested
        or trace_global_rng_calls
    ):
        raise ProbeError(
            "global_mtrand_call_sync_requires_bounded_diagnostic_trajectory"
        )
    if initial_global_mtrand_requested and (
        any(value is None for value in initial_global_mtrand_values)
        or initial_global_mtrand_oracle is None
        or not initial_global_mtrand_oracle.is_file()
        or initial_global_mtrand_seed is None
        or not 0 <= initial_global_mtrand_seed <= 0xFFFFFFFF
        or not global_mtrand_call_requested
        or global_mtrand_call_oracle is None
        or initial_global_mtrand_oracle.resolve()
        != global_mtrand_call_oracle.resolve()
        or initial_global_mtrand_seed != global_mtrand_call_oracle_seed
        or not diagnostic_disable_source_bound_board_anchor
    ):
        raise ProbeError(
            "initial_global_mtrand_seed_requires_matching_global_sync"
        )
    if startup_global_mtrand_observation_requested and (
        any(
            value is None
            for value in startup_global_mtrand_observation_values
        )
        or startup_global_mtrand_observation_oracle is None
        or not startup_global_mtrand_observation_oracle.is_file()
        or startup_global_mtrand_observation_seed is None
        or not 0
        <= startup_global_mtrand_observation_seed
        <= 0xFFFFFFFF
        or startup_global_mtrand_observation_end_at_update is None
        or startup_global_mtrand_observation_end_at_update < 0
        or startup_global_mtrand_observation_end_at_update >= probe_update
        or startup_global_mtrand_observation_maximum_draws <= 0
        or trajectory_mode != "rng"
        or trajectory_end_update is None
        or not skip_repaint_guard
        or not skip_frozen_snapshot
        or gameplay_mtrand_requested
        or global_mtrand_call_requested
        or initial_global_mtrand_requested
        or trace_global_rng_calls
        or not diagnostic_disable_source_bound_board_anchor
    ):
        raise ProbeError(
            "startup_global_mtrand_observation_requires_clean_diagnostic_"
            "trajectory"
        )
    if (
        diagnostic_disable_source_bound_board_anchor
        and not (
            global_mtrand_call_requested
            or startup_global_mtrand_observation_requested
        )
    ):
        raise ProbeError(
            "source_bound_board_anchor_disable_requires_rng_diagnostic"
        )
    if diagnostic_mutator is None and (
        post_mutation_score_value is not None
        or post_mutation_displayed_score_value is not None
        or allow_post_mutation_score_change
    ):
        raise ProbeError(
            "post_mutation_score_requires_diagnostic_mutator"
        )
    if int32_value is None and (
        diagnostic_mutator is not None
        or post_mutation_score_value is not None
        or post_mutation_displayed_score_value is not None
        or allow_post_mutation_score_change
        or displayed_score_value is not None
        or int32_scan_value is not None
    ):
        raise ProbeError(
            "score_discovery_requires_unmodified_stable_score"
        )
    if allow_post_mutation_score_change and (
        post_mutation_score_value is not None
    ):
        raise ProbeError(
            "post_mutation_score_contract_is_ambiguous"
        )
    plan = json.loads(plan_path.read_text(encoding="ascii"))
    if plan.get("schema") != "zuma-rl.pc-golden-v4-collection-plan":
        raise ProbeError("plan_invalid")
    diagnostic_runtime = plan.get("runtime")
    diagnostic_trace = plan.get("trace")
    source_bound_plan = (
        isinstance(diagnostic_runtime, Mapping)
        and diagnostic_runtime.get("launch_mode")
        == DIRECT_FIXED_SEED_LAUNCH_MODE
        and isinstance(diagnostic_trace, Mapping)
        and diagnostic_trace.get(
            "source_bound_command_broker_stop_after_update"
        )
        is not None
    )
    source_bound_formal_replay = (
        formal_full_state_evidence and source_bound_plan
    )
    source_bound_diagnostic_replay = (
        diagnostic_source_bound_replay and source_bound_plan
    )
    source_bound_replay = (
        source_bound_formal_replay or source_bound_diagnostic_replay
    )
    if diagnostic_source_bound_replay and not source_bound_plan:
        raise ProbeError("diagnostic_source_bound_replay_binding_missing")
    if formal_full_state_evidence and not (
        isinstance(diagnostic_runtime, Mapping)
        and diagnostic_runtime.get("launch_mode")
        == DIRECT_NATURAL_STRICT_BROKER_LAUNCH_MODE
        or source_bound_formal_replay
    ):
        raise ProbeError("formal_full_state_replay_binding_missing")
    if diagnostic_startup_crt_seed is not None and (
        not isinstance(diagnostic_runtime, Mapping)
        or diagnostic_runtime.get("launch_mode")
        != DIRECT_NATURAL_STRICT_BROKER_LAUNCH_MODE
    ):
        raise ProbeError(
            "startup_crt_seed_requires_natural_strict_plan"
        )
    if diagnostic_disable_source_bound_board_anchor and (
        not isinstance(
            plan.get("trace", {}).get("source_bound_board_anchor"),
            Mapping,
        )
        or not isinstance(
            plan.get("trace", {}).get(
                "source_bound_board_precall_global_restore"
            ),
            Mapping,
        )
    ):
        raise ProbeError("source_bound_board_anchor_disable_plan_invalid")
    project_root = Path(plan["project_root"]).resolve()
    runtime_executable = Path(
        plan["runtime"]["runtime_executable"]
    ).resolve()
    expected_runtime_sha256 = plan["runtime"].get(
        "expected_runtime_sha256"
    )
    declared_runtime_sha256 = plan["runtime"].get(
        "runtime_executable_sha256"
    )
    if (
        not _is_canonical_sha256(expected_runtime_sha256)
        or declared_runtime_sha256 != expected_runtime_sha256
    ):
        raise ProbeError("runtime_executable_identity_invalid")
    runtime_source_executable = _runtime_source_executable_from_plan(plan)
    dmo = plan_path.parent / plan["dmo"]["artifact"]
    replay_prestate = PcStateSnapshot.read(prestate_path)
    host_restore = PcStateSnapshot.read(host_restore_path)
    if initial_global_mtrand_requested:
        assert initial_global_mtrand_oracle is not None
        assert initial_global_mtrand_seed is not None
        try:
            load_initial_global_mtrand_call_oracle(
                initial_global_mtrand_oracle,
                seed=initial_global_mtrand_seed,
            )
        except (OSError, ValueError) as error:
            raise ProbeError(
                "initial_global_mtrand_seed_oracle_invalid"
            ) from error
    if startup_global_mtrand_observation_requested:
        assert startup_global_mtrand_observation_oracle is not None
        assert startup_global_mtrand_observation_seed is not None
        assert startup_global_mtrand_observation_end_at_update is not None
        try:
            loaded_startup_observation_oracle = (
                load_startup_global_mtrand_observation_oracle(
                startup_global_mtrand_observation_oracle,
                seed=startup_global_mtrand_observation_seed,
                end_at_update=(
                    startup_global_mtrand_observation_end_at_update
                ),
                maximum_draws=(
                    startup_global_mtrand_observation_maximum_draws
                ),
            )
            )
        except (OSError, ValueError) as error:
            raise ProbeError(
                "startup_global_mtrand_observation_oracle_invalid"
            ) from error
        if startup_global_mtrand_hidden_draw_requested:
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
                    loaded_startup_observation_oracle.entries
                )
            }
            start_entry_order = source_order_to_entry_order.get(
                startup_global_mtrand_hidden_draw_start_after_source_order
            )
            stop_entry_order = source_order_to_entry_order.get(
                startup_global_mtrand_hidden_draw_stop_before_source_order
            )
            if (
                start_entry_order is None
                or stop_entry_order is None
                or stop_entry_order != start_entry_order + 1
            ):
                raise ProbeError(
                    "startup_global_mtrand_hidden_draw_boundaries_invalid"
                )
    output_root.mkdir()
    attempts: list[dict[str, Any]] = []
    success_path: Path | None = None
    fatal_formal_error: Exception | None = None
    fatal_global_mtrand_error: Exception | None = None
    try:
        for attempt_number in range(1, maximum_attempts + 1):
            attempt_root = output_root / f"attempt-{attempt_number:02d}"
            attempt_root.mkdir()
            restore_state(replay_prestate)
            existing = set(process_ids_by_name(runtime_executable.name))
            if existing:
                deadline = time.monotonic() + 10.0
                while existing and time.monotonic() < deadline:
                    time.sleep(0.05)
                    existing = set(
                        process_ids_by_name(runtime_executable.name)
                    )
                if existing:
                    raise ProbeError("runtime_already_running")
            trace_result = attempt_root / "strict-replay.json"
            trace_log_path = attempt_root / "strict-replay.log"
            trace_process: subprocess.Popen[bytes] | None = None
            rng_trace_bundle: tuple[
                subprocess.Popen[bytes],
                Any,
                Path,
                Path,
                Path,
                Path,
            ] | None = None
            live_rng_monitor_bundle: dict[str, Any] | None = None
            live_rng_monitor_evidence: dict[str, Any] | None = None
            gameplay_mtrand_sync_evidence: dict[str, Any] | None = None
            initial_global_mtrand_seed_evidence: (
                dict[str, Any] | None
            ) = None
            startup_global_mtrand_observation_evidence: (
                dict[str, Any] | None
            ) = None
            global_mtrand_call_sync_bundle: dict[str, Any] | None = None
            global_mtrand_call_sync_ready: dict[str, Any] | None = None
            global_mtrand_call_sync_evidence: dict[str, Any] | None = None
            process_affinity_evidence: dict[str, int | bool] | None = None
            strict_command_replay_evidence: dict[str, Any] | None = None
            formal_fruit_trigger_evidence: dict[str, Any] | None = None
            runtime_identity: dict[str, Any] | None = None
            runtime_pid: int | None = None
            external_input_guard: ExternalInputGuard | None = None
            external_input_guard_receipt: Mapping[str, Any] | None = None
            external_input_guard_path = (
                attempt_root / "external-input-guard.json"
            )
            started_ns = time.perf_counter_ns()
            try:
                if formal_source_evidence or diagnostic_source_bound_replay:
                    external_input_guard = ExternalInputGuard()
                    external_input_guard.start()
                with trace_log_path.open("xb") as trace_log:
                    trace_process = subprocess.Popen(
                        _trace_args(
                            project_root=project_root,
                            plan=plan,
                            dmo=dmo,
                            result_path=trace_result,
                            detach_at_update=trace_detach_update,
                            reattach_at_update=trace_reattach_update,
                            gameplay_mtrand_oracle=(
                                gameplay_mtrand_oracle
                            ),
                            gameplay_mtrand_oracle_seed=(
                                gameplay_mtrand_oracle_seed
                            ),
                            gameplay_mtrand_oracle_maximum_draws=(
                                gameplay_mtrand_oracle_maximum_draws
                            ),
                            initial_global_mtrand_oracle=(
                                initial_global_mtrand_oracle
                            ),
                            initial_global_mtrand_seed=(
                                initial_global_mtrand_seed
                            ),
                            startup_global_mtrand_observation_oracle=(
                                startup_global_mtrand_observation_oracle
                            ),
                            startup_global_mtrand_observation_seed=(
                                startup_global_mtrand_observation_seed
                            ),
                            startup_global_mtrand_observation_end_at_update=(
                                startup_global_mtrand_observation_end_at_update
                            ),
                            startup_global_mtrand_observation_maximum_draws=(
                                startup_global_mtrand_observation_maximum_draws
                            ),
                            startup_global_mtrand_hidden_draw_start_after_source_order=(
                                startup_global_mtrand_hidden_draw_start_after_source_order
                            ),
                            startup_global_mtrand_hidden_draw_stop_before_source_order=(
                                startup_global_mtrand_hidden_draw_stop_before_source_order
                            ),
                            startup_priority_bias_until_update=(
                                startup_priority_bias_until_update
                            ),
                            process_affinity_mask=process_affinity_mask,
                            suspend_main_thread_on_stop=(
                                global_mtrand_call_requested
                            ),
                            disable_source_bound_board_anchor=(
                                diagnostic_disable_source_bound_board_anchor
                            ),
                            diagnostic_startup_crt_seed=(
                                diagnostic_startup_crt_seed
                            ),
                        ),
                        cwd=project_root,
                        env=_child_environment(project_root),
                        stdin=subprocess.DEVNULL,
                        stdout=trace_log,
                        stderr=subprocess.STDOUT,
                        close_fds=True,
                    )
                    runtime_pid, _ = _wait_for_runtime(
                        trace_process,
                        existing=existing,
                        expected_runtime=runtime_executable,
                        timeout=float(
                            plan["trace"]["launch_timeout_seconds"]
                        )
                        + 5,
                    )
                    if global_mtrand_call_requested:
                        assert global_mtrand_call_oracle is not None
                        assert global_mtrand_call_oracle_seed is not None
                        assert (
                            global_mtrand_call_start_after_update
                            is not None
                        )
                        assert global_mtrand_call_end_at_update is not None
                        main_thread_id = _wait_for_startup_main_thread_id(
                            trace_log_path=trace_log_path,
                            trace_process=trace_process,
                            expected_process_id=runtime_pid,
                            timeout_seconds=float(
                                plan["trace"]["launch_timeout_seconds"]
                            )
                            + 5.0,
                        )
                        global_mtrand_call_sync_bundle = (
                            _start_global_mtrand_call_sync(
                                project_root=project_root,
                                pid=runtime_pid,
                                main_thread_id=main_thread_id,
                                runtime_executable=runtime_executable,
                                oracle_path=global_mtrand_call_oracle,
                                seed=global_mtrand_call_oracle_seed,
                                start_after_update=(
                                    global_mtrand_call_start_after_update
                                ),
                                end_at_update=(
                                    global_mtrand_call_end_at_update
                                ),
                                maximum_draws=(
                                    global_mtrand_call_maximum_draws
                                ),
                                timeout_seconds=(
                                    global_mtrand_call_sync_timeout_seconds
                                ),
                                attempt_root=attempt_root,
                                observe_boundary_only=(
                                    global_mtrand_call_observe_boundary_only
                                ),
                                observe_thread_runtime=(
                                    global_mtrand_call_observe_thread_runtime
                                ),
                                observe_handoff_thread_runtime=(
                                    global_mtrand_call_observe_handoff_thread_runtime
                                ),
                                observe_handoff_rng_state=(
                                    global_mtrand_call_observe_handoff_rng_state
                                ),
                                handoff_rng_wait_target_words_sha256=(
                                    global_mtrand_call_handoff_rng_wait_target_words_sha256
                                ),
                                handoff_rng_wait_min_index=(
                                    global_mtrand_call_handoff_rng_wait_min_index
                                ),
                                handoff_rng_wait_timeout_seconds=(
                                    global_mtrand_call_handoff_rng_wait_timeout_seconds
                                ),
                                handoff_rng_wait_poll_interval_seconds=(
                                    global_mtrand_call_handoff_rng_wait_poll_interval_seconds
                                ),
                            )
                        )
                        global_mtrand_call_sync_ready = (
                            _wait_for_global_mtrand_call_sync_ready(
                                global_mtrand_call_sync_bundle,
                                trace_process=trace_process,
                                timeout_seconds=(
                                    global_mtrand_call_sync_timeout_seconds
                                ),
                            )
                        )
                    if process_affinity_mask is not None:
                        process_affinity_evidence = _observe_process_affinity(
                            runtime_pid,
                            process_affinity_mask,
                        )
                    target = _wait_for_capture_window(runtime_pid)
                    if (
                        snapshot_trajectory_frames
                        or formal_full_state_evidence
                        or source_bound_diagnostic_replay
                    ):
                        runtime_identity = _capture_runtime_process_identity(
                            target,
                            runtime_pid=runtime_pid,
                            runtime_executable=runtime_executable,
                            runtime_source_executable=(
                                runtime_source_executable
                            ),
                            expected_runtime_sha256=(
                                expected_runtime_sha256
                            ),
                        )
                    repaint_trace_process: Any = trace_process
                    if (
                        formal_full_state_evidence
                        and plan["runtime"].get("launch_mode")
                        == DIRECT_NATURAL_STRICT_BROKER_LAUNCH_MODE
                    ):
                        if trace_process is None:
                            raise ProbeError(
                                "natural_strict_trace_process_missing"
                            )
                        strict_command_replay_evidence = (
                            _finish_natural_strict_trace(
                                process=trace_process,
                                result_path=trace_result,
                                plan=plan,
                                expected_pid=runtime_pid,
                                expected_dmo=dmo,
                                expected_runtime=runtime_executable,
                                expected_startup_priority_bias_until_update=(
                                    startup_priority_bias_until_update
                                ),
                                expected_startup_process_affinity_mask=(
                                    process_affinity_mask
                                ),
                                expected_diagnostic_startup_seed=(
                                    diagnostic_startup_crt_seed
                                ),
                                timeout_seconds=float(
                                    plan["trace"]["trace_timeout_seconds"]
                                ),
                            )
                        )
                        trace_process = None
                        repaint_trace_process = (
                            _ValidatedCompletedTraceGuard()
                        )
                    elif source_bound_replay:
                        if trace_process is None:
                            raise ProbeError(
                                "source_bound_strict_trace_process_missing"
                            )
                        strict_command_replay_evidence = (
                            _finish_source_bound_strict_trace(
                                process=trace_process,
                                result_path=trace_result,
                                plan_path=plan_path,
                                plan=plan,
                                expected_pid=runtime_pid,
                                expected_dmo=dmo,
                                expected_runtime=runtime_executable,
                                timeout_seconds=float(
                                    plan["trace"]["trace_timeout_seconds"]
                                ),
                            )
                        )
                        trace_process = None
                        repaint_trace_process = (
                            _ValidatedCompletedTraceGuard()
                        )
                    if live_rng_monitor_interval_seconds is not None:
                        live_rng_monitor_bundle = _start_live_rng_monitor(
                            project_root=project_root,
                            pid=runtime_pid,
                            attempt_root=attempt_root,
                            interval_seconds=(
                                live_rng_monitor_interval_seconds
                            ),
                            thread_id=_window_thread_id(
                                target.window_handle,
                                runtime_pid,
                            ),
                        )
                    if skip_repaint_guard:
                        repaint = {
                            "status": (
                                "SUPERSEDED_BY_PER_TICK_REPAINT_HANDSHAKE"
                                if formal_exact_step_evidence
                                else "SKIPPED_DIAGNOSTIC"
                            ),
                            "reason": (
                                "formal_exact_step_per_tick_repaint"
                                if formal_exact_step_evidence
                                else "pre_repaint_memory_trajectory"
                            ),
                            "window_handle": target.window_handle,
                        }
                    else:
                        guard = plan["capture"][
                            "window_repaint_guard"
                        ]
                        repaint = _wait_for_repaint_guard(
                            target=target,
                            trace_process=repaint_trace_process,
                            trigger_update=plan["capture"][
                                "window_repaint_update"
                            ],
                            previous_effectful_command_update=guard[
                                "previous_effectful_command_update"
                            ],
                            next_effectful_command_update=guard[
                                "next_effectful_command_update"
                            ],
                            guarded_idle_command_count=guard[
                                "guarded_idle_command_count"
                            ],
                            # Full command tracing is intentionally much
                            # slower than native replay.  The trace deadline
                            # is the immutable session bound and must also
                            # govern this pre-freeze repaint handshake.
                            timeout=float(
                                plan["trace"]["trace_timeout_seconds"]
                            ),
                        )
                    if normalized_formal_fruit_trigger is not None:
                        (
                            freeze_state,
                            trajectory_end_update,
                            fruit_trigger_transcript,
                        ) = _freeze_on_first_fruit_lifecycle_window(
                            pid=runtime_pid,
                            demo_length=int(plan["dmo"]["length_updates"]),
                            contract=normalized_formal_fruit_trigger,
                        )
                        selection = fruit_trigger_transcript["selection"]
                        probe_update = int(selection["freeze_update"])
                        slowdown_update = int(selection["slowdown_update"])
                        transcript_path = (
                            attempt_root
                            / "formal-fruit-lifecycle-trigger.json"
                        )
                        _write_canonical_probe(
                            transcript_path,
                            fruit_trigger_transcript,
                        )
                        formal_fruit_trigger_evidence = {
                            "schema": FORMAL_FRUIT_TRIGGER_BINDING_SCHEMA,
                            "version": FORMAL_FRUIT_TRIGGER_VERSION,
                            "status": "PASS",
                            "artifact": transcript_path.name,
                            "artifact_sha256": _sha256_path(transcript_path),
                            "selection_rule": fruit_trigger_transcript[
                                "selection_rule"
                            ],
                            "monitor_start_update": (
                                normalized_formal_fruit_trigger[
                                    "monitor_start_update"
                                ]
                            ),
                            "threshold_observation_update": int(
                                selection["threshold_observation"][
                                    "framework_update"
                                ]
                            ),
                            "freeze_update": probe_update,
                            "trajectory_end_update": (
                                trajectory_end_update
                            ),
                            "trajectory_tick_count": int(
                                selection["trajectory_tick_count"]
                            ),
                            "process_id": runtime_pid,
                            "read_only_observation": True,
                            "gameplay_or_rng_process_memory_write_count": 0,
                        }
                    elif _uses_natural_exact_freeze(
                        plan["runtime"].get("launch_mode"),
                        source_bound_formal_replay=(
                            source_bound_replay
                        ),
                    ):
                        freeze_state = _natural_exact_freeze(
                            pid=runtime_pid,
                            slowdown_update=slowdown_update,
                            probe_update=probe_update,
                            demo_length=int(
                                plan["dmo"]["length_updates"]
                            ),
                            timeout=180.0,
                        )
                    else:
                        freeze_state = freeze_replay(
                            pid=runtime_pid,
                            slowdown_at=slowdown_update,
                            freeze_update=probe_update,
                            demo_length=int(
                                plan["dmo"]["length_updates"]
                            ),
                            timeout=180.0,
                            allow_slowdown_overshoot=(
                                skip_repaint_guard
                            ),
                        )
                    if plan["runtime"].get("launch_mode") == (
                        DIRECT_NATURAL_STRICT_BROKER_LAUNCH_MODE
                    ):
                        if strict_command_replay_evidence is None:
                            if trace_process is None:
                                raise ProbeError(
                                    "natural_strict_trace_process_missing"
                                )
                            strict_command_replay_evidence = (
                                _finish_natural_strict_trace(
                                    process=trace_process,
                                    result_path=trace_result,
                                    plan=plan,
                                    expected_pid=runtime_pid,
                                    expected_dmo=dmo,
                                    expected_runtime=runtime_executable,
                                    expected_startup_priority_bias_until_update=(
                                        startup_priority_bias_until_update
                                    ),
                                    expected_startup_process_affinity_mask=(
                                        process_affinity_mask
                                    ),
                                    expected_diagnostic_startup_seed=(
                                        diagnostic_startup_crt_seed
                                    ),
                                )
                            )
                            trace_process = None
                    if gameplay_mtrand_oracle is not None:
                        gameplay_mtrand_sync_evidence = (
                            _load_gameplay_mtrand_sync_receipt(
                                attempt_root / "gameplay-mtrand-sync.json",
                                expected_process_id=runtime_pid,
                                expected_oracle=gameplay_mtrand_oracle,
                                expected_dmo=dmo,
                                expected_hidden_draw_start_after_source_order=(
                                    startup_global_mtrand_hidden_draw_start_after_source_order
                                ),
                                expected_hidden_draw_stop_before_source_order=(
                                    startup_global_mtrand_hidden_draw_stop_before_source_order
                                ),
                            )
                        )
                    if initial_global_mtrand_oracle is not None:
                        if initial_global_mtrand_seed is None:
                            raise ProbeError(
                                "initial_global_mtrand_seed_missing"
                            )
                        initial_global_mtrand_seed_evidence = (
                            _load_initial_global_mtrand_seed_receipt(
                                attempt_root
                                / "initial-global-mtrand-seed.json",
                                expected_process_id=runtime_pid,
                                expected_oracle=(
                                    initial_global_mtrand_oracle
                                ),
                                expected_seed=initial_global_mtrand_seed,
                                expected_dmo=dmo,
                            )
                        )
                    if startup_global_mtrand_observation_oracle is not None:
                        if (
                            startup_global_mtrand_observation_seed is None
                            or startup_global_mtrand_observation_end_at_update
                            is None
                        ):
                            raise ProbeError(
                                "startup_global_mtrand_observation_options_"
                                "incomplete"
                            )
                        startup_global_mtrand_observation_evidence = (
                            _load_startup_global_mtrand_observation_receipt(
                                attempt_root
                                / "startup-global-mtrand-observation.json",
                                expected_process_id=runtime_pid,
                                expected_oracle=(
                                    startup_global_mtrand_observation_oracle
                                ),
                                expected_seed=(
                                    startup_global_mtrand_observation_seed
                                ),
                                expected_end_at_update=(
                                    startup_global_mtrand_observation_end_at_update
                                ),
                                expected_maximum_draws=(
                                    startup_global_mtrand_observation_maximum_draws
                                ),
                                expected_dmo=dmo,
                            )
                        )
                    if live_rng_monitor_bundle is not None:
                        live_rng_monitor_evidence = (
                            _finish_live_rng_monitor(
                                live_rng_monitor_bundle
                            )
                        )
                        live_rng_monitor_bundle = None
                    freeze_handle = open_process_readonly(runtime_pid)
                    try:
                        freeze_payload = read_process_bytes(
                            freeze_handle,
                            (
                                freeze_state.multiplier_address
                                - FREEZE_STATE_PREFIX_BYTES
                            ),
                            FREEZE_STATE_BYTES,
                        )
                    finally:
                        close_process(freeze_handle)
                    freeze_path = (
                        attempt_root / "replay-freeze-state.bin"
                    )
                    freeze_path.write_bytes(freeze_payload)
                    thread_crt_state = _resolve_board_thread_crt_state(
                        pid=runtime_pid,
                        window_handle=target.window_handle,
                        trace_log_path=trace_log_path,
                    )
                    snapshot_path = attempt_root / "frozen-frame.bmp"
                    if skip_frozen_snapshot:
                        frozen_frame: dict[str, Any] = {
                            "status": (
                                "SUPERSEDED_BY_RETAINED_EXACT_STEP_FRAMES"
                                if formal_exact_step_evidence
                                else "SKIPPED_DIAGNOSTIC"
                            ),
                            "reason": (
                                "formal_exact_step_warmup_and_tick_frames"
                                if formal_exact_step_evidence
                                else "memory_only_trajectory"
                            ),
                        }
                    else:
                        snapshot = subprocess.run(
                            [
                                sys.executable,
                                str(
                                    project_root
                                    / "tools"
                                    / "snapshot_dxgi_window.py"
                                ),
                                "--output",
                                str(snapshot_path),
                                "--process",
                                runtime_executable.name,
                                "--timeout",
                                "15",
                            ],
                            cwd=project_root,
                            env=_child_environment(project_root),
                            stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            timeout=30,
                            check=False,
                        )
                        if (
                            snapshot.returncode != 0
                            or not snapshot_path.is_file()
                        ):
                            raise ProbeError("frozen_snapshot_failed")
                        frozen_frame = {
                            "status": "PASS",
                            "artifact": snapshot_path.name,
                            "sha256": _sha256_path(snapshot_path),
                            "semantics": (
                                FORMAL_FULL_STATE_SNAPSHOT_SEMANTICS
                                if formal_full_state_evidence
                                else DIAGNOSTIC_SNAPSHOT_SEMANTICS
                            ),
                            "tool_output": snapshot.stdout.decode(
                                "utf-8", errors="replace"
                            ).strip(),
                        }
                    diagnostic_mutation = None
                    if diagnostic_mutator is not None:
                        pre_mutation_root = attempt_root / "pre-mutation"
                        pre_mutation_root.mkdir()
                        pre_mutation_board = collect_active_board(
                            runtime_pid,
                            expected_score=int32_value,
                            expected_displayed_score=displayed_score_value,
                            output_root=pre_mutation_root,
                            thread_crt_state=thread_crt_state,
                        )
                        diagnostic_mutation = dict(
                            diagnostic_mutator(
                                runtime_pid,
                                pre_mutation_board,
                                attempt_root,
                            )
                        )
                    declared_board_score = (
                        None
                        if int32_value is None
                        else (
                            None
                            if allow_post_mutation_score_change
                            else (
                                int32_value
                                if post_mutation_score_value is None
                                else post_mutation_score_value
                            )
                        )
                    )
                    board_probe = collect_active_board(
                        runtime_pid,
                        expected_score=declared_board_score,
                        expected_displayed_score=(
                            displayed_score_value
                            if post_mutation_displayed_score_value is None
                            else post_mutation_displayed_score_value
                        ),
                        output_root=attempt_root,
                        thread_crt_state=thread_crt_state,
                    )
                    observed_score = int(board_probe["score"])
                    observed_displayed_score = int(
                        board_probe["displayed_score"]
                    )
                    if int32_value is None:
                        if (
                            observed_score != observed_displayed_score
                            and not allow_discovered_display_mismatch
                        ):
                            raise ProbeError(
                                "discovered_score_not_stable:"
                                f"score={observed_score},"
                                f"displayed={observed_displayed_score}"
                            )
                        score_value = observed_score
                        if formal_independent_score_binding:
                            score_binding_mode = (
                                INDEPENDENT_SCORE_BINDING_MODE
                            )
                        elif allow_discovered_display_mismatch:
                            score_binding_mode = (
                                "active_board_read_only_rolling_display_"
                                "diagnostic"
                            )
                        else:
                            score_binding_mode = "active_board_read_only"
                    else:
                        score_value = int32_value
                        score_binding_mode = "declared"
                    scan_value = (
                        score_value
                        if int32_scan_value is None
                        else int32_scan_value
                    )
                    score_binding: dict[str, Any] = {
                        "schema": SCORE_BINDING_SCHEMA,
                        "version": (
                            SCORE_BINDING_INDEPENDENT_VERSION
                            if formal_independent_score_binding
                            else SCORE_BINDING_LEGACY_VERSION
                        ),
                        "mode": score_binding_mode,
                        "score": observed_score,
                        "displayed_score": observed_displayed_score,
                        "scan_value": scan_value,
                    }
                    if formal_independent_score_binding:
                        score_binding.update(
                            {
                                "score_offset": BOARD_SCORE_OFFSET,
                                "displayed_score_offset": (
                                    BOARD_DISPLAYED_SCORE_OFFSET
                                ),
                                "acceptance_rule": (
                                    INDEPENDENT_SCORE_BINDING_ACCEPTANCE_RULE
                                ),
                            }
                        )
                    if skip_int32_scan:
                        probe = {
                            "schema": "zuma-rl.pc-memory-int32-probe",
                            "version": 1,
                            "process_id": runtime_pid,
                            "value": scan_value,
                            "scan_status": (
                                "SKIPPED_EXACT_STEP_SOURCE_NOT_REQUIRED"
                            ),
                            "scanned_regions": 0,
                            "scanned_bytes": 0,
                            "hit_count": 0,
                            "non_image_hit_count": 0,
                            "reference_target_count": 0,
                            "reference_target_limit": (
                                MAX_POINTER_REFERENCE_TARGETS
                            ),
                            "reference_target_analyzed_count": 0,
                            "reference_target_truncated": False,
                            "reference_target_selection": (
                                "not_applicable"
                            ),
                            "hits": [],
                        }
                    else:
                        probe = scan_int32(
                            runtime_pid,
                            scan_value,
                        )
                    diagnostic_observation = None
                    if diagnostic_observer is not None:
                        diagnostic_observation = dict(
                            diagnostic_observer(
                                runtime_pid,
                                board_probe,
                                attempt_root,
                            )
                        )
                    trajectory = None
                    rng_trace_evidence = None
                    if trajectory_end_update is not None:
                        rng_trace_start_update = (
                            probe_update
                            if trajectory_start_update is None
                            else trajectory_start_update
                        )
                        arm_rng_trace: Callable[[], None] | None = None
                        if trace_global_rng_calls:
                            def arm_rng_trace() -> None:
                                nonlocal rng_trace_bundle
                                if rng_trace_bundle is not None:
                                    raise ProbeError(
                                        "global_rng_call_trace_already_started"
                                    )
                                rng_trace_bundle = (
                                    _start_global_rng_call_trace(
                                        project_root=project_root,
                                        pid=runtime_pid,
                                        runtime_executable=(
                                            runtime_executable
                                        ),
                                        attempt_root=attempt_root,
                                        start_update=(
                                            rng_trace_start_update
                                        ),
                                        end_update=(
                                            trajectory_end_update
                                        ),
                                        timeout_seconds=(
                                            global_rng_call_trace_timeout_seconds
                                        ),
                                    )
                                )
                        if trajectory_mode == "full":
                            trajectory = collect_board_trajectory(
                                runtime_pid,
                                frozen_state=freeze_state,
                                end_update=trajectory_end_update,
                                output_root=attempt_root / "trajectory",
                                thread_crt_state=thread_crt_state,
                                process_identity=runtime_identity,
                                formal_full_state_evidence=(
                                    formal_full_state_evidence
                                ),
                            )
                        elif trajectory_mode == "rng":
                            trajectory = collect_rng_trajectory(
                                runtime_pid,
                                frozen_state=freeze_state,
                                end_update=trajectory_end_update,
                                output_root=attempt_root / "trajectory",
                                thread_crt_state=thread_crt_state,
                                record_start_update=(
                                    trajectory_start_update
                                ),
                                snapshot_trajectory_frames=(
                                    snapshot_trajectory_frames
                                ),
                                snapshot_process_name=(
                                    runtime_executable.name
                                ),
                                square_dwm_corners=(
                                    snapshot_square_dwm_corners
                                ),
                                    snapshot_warmup_frame=(
                                        snapshot_trajectory_warmup_frame
                                    ),
                                process_identity=runtime_identity,
                                formal_exact_step_evidence=(
                                    formal_exact_step_evidence
                                ),
                                snapshot_repaint_mechanism=(
                                    SET_WINDOW_POS_REPAINT_MECHANISM
                                    if plan["runtime"].get("launch_mode")
                                    in {
                                        STEAM_LAUNCH_MODE,
                                        DIRECT_NATURAL_SEED_LAUNCH_MODE,
                                        DIRECT_NATURAL_STRICT_BROKER_LAUNCH_MODE,
                                    }
                                    else NONCLIENT_DRAG_REPAINT_MECHANISM
                                ),
                                on_record_start=arm_rng_trace,
                                trajectory_step_timeout_seconds=(
                                    trajectory_step_timeout_seconds
                                ),
                            )
                        else:
                            raise ProbeError("trajectory_mode_invalid")
                        if rng_trace_bundle is not None:
                            (
                                rng_trace_process,
                                rng_trace_log,
                                rng_trace_result,
                                _rng_trace_ready,
                                rng_trace_stop,
                                rng_trace_log_path,
                            ) = rng_trace_bundle
                            rng_trace_evidence = (
                                _finish_global_rng_call_trace(
                                    process=rng_trace_process,
                                    log_stream=rng_trace_log,
                                    result_path=rng_trace_result,
                                    stop_path=rng_trace_stop,
                                    log_path=rng_trace_log_path,
                                    start_update=rng_trace_start_update,
                                    end_update=trajectory_end_update,
                                )
                            )
                            rng_trace_bundle = None
                    if global_mtrand_call_sync_bundle is not None:
                        if global_mtrand_call_sync_ready is None:
                            raise ProbeError(
                                "global_mtrand_call_sync_ready_missing"
                            )
                        global_mtrand_call_sync_evidence = (
                            _finish_global_mtrand_call_sync(
                                global_mtrand_call_sync_bundle,
                                ready=global_mtrand_call_sync_ready,
                            )
                        )
                        global_mtrand_call_sync_bundle = None
                        if initial_global_mtrand_requested:
                            _require_zero_write_global_mtrand_suffix(
                                global_mtrand_call_sync_evidence
                            )
                    external_input_guard_binding = None
                    if formal_source_evidence or diagnostic_source_bound_replay:
                        if external_input_guard is None:
                            raise ExternalInputGuardError(
                                "formal_guard_missing"
                            )
                        external_input_guard_receipt = (
                            external_input_guard.finish()
                        )
                        _write_external_input_guard_receipt(
                            external_input_guard_path,
                            external_input_guard_receipt,
                        )
                        guard_counts = external_input_guard_receipt[
                            "event_counts"
                        ]
                        guard_coverage = external_input_guard_receipt[
                            "coverage"
                        ]
                        external_input_guard_binding = {
                            "schema": (
                                EXTERNAL_INPUT_GUARD_BINDING_SCHEMA
                            ),
                            "version": (
                                EXTERNAL_INPUT_GUARD_BINDING_VERSION
                            ),
                            "status": external_input_guard_receipt[
                                "status"
                            ],
                            "artifact": external_input_guard_path.name,
                            "artifact_bytes": (
                                external_input_guard_path.stat().st_size
                            ),
                            "artifact_sha256": _sha256_path(
                                external_input_guard_path
                            ),
                            "coverage_start_perf_counter_ns": (
                                guard_coverage[
                                    "start_perf_counter_ns"
                                ]
                            ),
                            "coverage_end_perf_counter_ns": (
                                guard_coverage[
                                    "end_perf_counter_ns"
                                ]
                            ),
                            "external_event_count": guard_counts[
                                "external"
                            ],
                            "allowed_repaint_event_count": guard_counts[
                                "allowed_repaint"
                            ],
                        }
                        if (
                            external_input_guard_receipt["status"]
                            != "PASS"
                            or guard_counts["external"] != 0
                        ):
                            raise FormalInputContaminationError(
                                "formal_external_input_guard_failed"
                            )
                    probe.update(
                        {
                            "framework_update": probe_update,
                            "slowdown_update": slowdown_update,
                            "score_binding": score_binding,
                            "freeze_state": {
                                "schema": (
                                    "zuma-rl.pc-replay-freeze-state"
                                ),
                                "version": 1,
                                "artifact": freeze_path.name,
                                "artifact_bytes": len(freeze_payload),
                                "artifact_sha256": _sha256_path(
                                    freeze_path
                                ),
                                "multiplier_offset": (
                                    FREEZE_STATE_PREFIX_BYTES
                                ),
                                "multiplier_address": (
                                    freeze_state.multiplier_address
                                ),
                                "non_draw_count": (
                                    freeze_state.non_draw_count
                                ),
                                "frame_time_ms": (
                                    freeze_state.frame_time_ms
                                ),
                                "sleep_count": freeze_state.sleep_count,
                                "draw_count": freeze_state.draw_count,
                                "update_count": freeze_state.update_count,
                                "update_app_state": (
                                    freeze_state.update_app_state
                                ),
                                "update_app_depth": (
                                    freeze_state.update_app_depth
                                ),
                                "update_multiplier": (
                                    freeze_state.update_multiplier
                                ),
                                "paused": freeze_state.paused,
                                "fast_forward_target": (
                                    freeze_state.fast_forward_target
                                ),
                                "fast_forward_to_marker": (
                                    freeze_state.fast_forward_to_marker
                                ),
                                "fast_forward_step": (
                                    freeze_state.fast_forward_step
                                ),
                                "step_mode": freeze_state.step_mode,
                                "loading_thread_started": (
                                    freeze_state.loading_thread_started
                                ),
                                "loading_thread_completed": (
                                    freeze_state.loading_thread_completed
                                ),
                                "loaded": freeze_state.loaded,
                            },
                            "repaint": repaint,
                            "active_board": board_probe,
                            "diagnostic_mutation": diagnostic_mutation,
                            "diagnostic_observation": (
                                diagnostic_observation
                            ),
                            "trajectory": trajectory,
                            "global_rng_call_trace": rng_trace_evidence,
                            "live_rng_monitor": live_rng_monitor_evidence,
                            "gameplay_mtrand_sync": (
                                gameplay_mtrand_sync_evidence
                            ),
                            "initial_global_mtrand_seed": (
                                initial_global_mtrand_seed_evidence
                            ),
                            "startup_global_mtrand_observation": (
                                startup_global_mtrand_observation_evidence
                            ),
                            "global_mtrand_call_sync": (
                                global_mtrand_call_sync_evidence
                            ),
                            "strict_command_replay": (
                                strict_command_replay_evidence
                            ),
                            "formal_fruit_lifecycle_trigger": (
                                formal_fruit_trigger_evidence
                            ),
                            "diagnostic_source_bound_board_anchor": (
                                {
                                    "disabled": True,
                                    "reason": (
                                        (
                                            "read_only_startup_global_"
                                            "mtrand_observation_requires_"
                                            "unmodified_schedule"
                                        )
                                        if startup_global_mtrand_observation_requested
                                        else (
                                            "complete_global_mtrand_call_"
                                            "oracle_owns_board_call_and_"
                                            "suffix"
                                        )
                                    ),
                                    "persistent_plan_modified": False,
                                }
                                if diagnostic_disable_source_bound_board_anchor
                                else None
                            ),
                            "diagnostic_process_affinity": (
                                process_affinity_evidence
                            ),
                            "runtime_executable_sha256": (
                                runtime_identity["executable_sha256"]
                                if runtime_identity is not None
                                else _sha256_path(runtime_executable)
                            ),
                            "dmo_sha256": _sha256_path(dmo),
                            "process_identity": runtime_identity,
                            "external_input_guard": (
                                external_input_guard_binding
                            ),
                            "evidence_classification": (
                                FORMAL_FULL_STATE_CLASSIFICATION
                                if formal_full_state_evidence
                                else (
                                    "formal_pc_golden_exact_step_source"
                                    if formal_exact_step_evidence
                                    else "diagnostic"
                                )
                            ),
                            "frozen_frame": frozen_frame,
                        }
                    )
                    probe_path = attempt_root / "memory-probe.json"
                    _write_canonical_probe(probe_path, probe)
                    success_path = probe_path
                    attempts.append(
                        {
                            "attempt": attempt_number,
                            "status": "PASS",
                            "process_id": runtime_pid,
                            "process_creation_filetime_100ns": (
                                runtime_identity[
                                    "process_creation_filetime_100ns"
                                ]
                                if runtime_identity is not None
                                else None
                            ),
                            "started_perf_counter_ns": started_ns,
                            "finished_perf_counter_ns": time.perf_counter_ns(),
                            "probe_sha256": _sha256_path(probe_path),
                            **(
                                {
                                    "external_input_guard_sha256": (
                                        external_input_guard_binding[
                                            "artifact_sha256"
                                        ]
                                    ),
                                    "external_input_guard_coverage_start_perf_counter_ns": (
                                        external_input_guard_binding[
                                            "coverage_start_perf_counter_ns"
                                        ]
                                    ),
                                    "external_input_guard_coverage_end_perf_counter_ns": (
                                        external_input_guard_binding[
                                            "coverage_end_perf_counter_ns"
                                        ]
                                    ),
                                }
                                if external_input_guard_binding is not None
                                else {}
                            ),
                        }
                    )
                    print(
                        f"probe_attempt={attempt_number} status=PASS "
                        f"pid={runtime_pid}",
                        flush=True,
                    )
                    break
            except (
                ExternalInputGuardError,
                FormalInputContaminationError,
            ) as error:
                fatal_formal_error = error
                attempts.append(
                    {
                        "attempt": attempt_number,
                        "status": "INVALID",
                        "process_id": runtime_pid,
                        "started_perf_counter_ns": started_ns,
                        "finished_perf_counter_ns": time.perf_counter_ns(),
                        "error_type": type(error).__name__,
                        "error": str(error)[:500],
                    }
                )
                print(
                    f"probe_attempt={attempt_number} status=INVALID "
                    f"error={type(error).__name__}:{str(error)[:200]}",
                    flush=True,
                )
            except Exception as error:
                failure_status, retry_scope = _attempt_failure_disposition(
                    global_mtrand_call_requested=(
                        global_mtrand_call_requested
                    ),
                    global_mtrand_handoff_ready_observed=(
                        global_mtrand_call_sync_ready is not None
                    ),
                )
                if failure_status == "FAIL":
                    fatal_global_mtrand_error = error
                attempts.append(
                    {
                        "attempt": attempt_number,
                        "status": failure_status,
                        "process_id": runtime_pid,
                        "started_perf_counter_ns": started_ns,
                        "finished_perf_counter_ns": time.perf_counter_ns(),
                        "error_type": type(error).__name__,
                        "error": str(error)[:500],
                        **(
                            {"retry_scope": retry_scope}
                            if retry_scope is not None
                            else {}
                        ),
                    }
                )
                print(
                    f"probe_attempt={attempt_number} "
                    f"status={failure_status} "
                    f"error={type(error).__name__}:{str(error)[:200]}",
                    flush=True,
                )
            finally:
                if (
                    external_input_guard is not None
                    and not external_input_guard.finished
                ):
                    try:
                        external_input_guard_receipt = (
                            external_input_guard.finish()
                        )
                        if not external_input_guard_path.exists():
                            _write_external_input_guard_receipt(
                                external_input_guard_path,
                                external_input_guard_receipt,
                            )
                    except Exception:
                        pass
                if live_rng_monitor_bundle is not None:
                    _abort_live_rng_monitor(live_rng_monitor_bundle)
                if rng_trace_bundle is not None:
                    (
                        rng_trace_process,
                        rng_trace_log,
                        _rng_trace_result,
                        _rng_trace_ready,
                        rng_trace_stop,
                        _rng_trace_log_path,
                    ) = rng_trace_bundle
                    try:
                        if not rng_trace_stop.exists():
                            rng_trace_stop.write_text(
                                "stop\n",
                                encoding="ascii",
                            )
                        if rng_trace_process.poll() is None:
                            rng_trace_process.wait(timeout=10)
                    except Exception:
                        if rng_trace_process.poll() is None:
                            rng_trace_process.terminate()
                            try:
                                rng_trace_process.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                rng_trace_process.kill()
                                rng_trace_process.wait(timeout=10)
                    finally:
                        if not rng_trace_log.closed:
                            rng_trace_log.close()
                if global_mtrand_call_sync_bundle is not None:
                    _abort_global_mtrand_call_sync(
                        global_mtrand_call_sync_bundle
                    )
                _stop_children(
                    trace_process,
                    runtime_pid,
                    runtime_executable,
                )
                _write_canonical_attempts(
                    output_root / "attempts.json",
                    attempts,
                )
            if (
                fatal_formal_error is not None
                or fatal_global_mtrand_error is not None
            ):
                break
        if fatal_formal_error is not None:
            raise ProbeError(str(fatal_formal_error))
        if fatal_global_mtrand_error is not None:
            raise ProbeError(
                "nonretryable_global_mtrand_failure:"
                f"{type(fatal_global_mtrand_error).__name__}:"
                f"{str(fatal_global_mtrand_error)}"
            )
        if success_path is None:
            raise ProbeError("all_probe_attempts_failed")
        return success_path
    finally:
        restore_state(host_restore)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--prestate", required=True, type=Path)
    parser.add_argument("--host-restore", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--probe-update", type=int, default=7738)
    parser.add_argument("--slowdown-update", type=int, default=7650)
    parser.add_argument("--int32-value", type=int, default=7950)
    score_mode = parser.add_mutually_exclusive_group()
    score_mode.add_argument(
        "--discover-score",
        action="store_true",
        help=(
            "read the active Board score after freezing instead of requiring "
            "a predeclared score; certification requires score == displayed "
            "score and forbids diagnostic mutation"
        ),
    )
    score_mode.add_argument(
        "--formal-discover-independent-scores",
        action="store_true",
        help=(
            "formal full-state only: read the accumulated and rolling Board "
            "score fields independently and accept every int32 pair without "
            "outcome filtering; requires exactly one startup attempt"
        ),
    )
    score_mode.add_argument(
        "--diagnostic-discover-rolling-score",
        action="store_true",
        help=(
            "diagnostic only: discover Board score and rolling displayed "
            "score independently; requires a memory-only trajectory"
        ),
    )
    parser.add_argument(
        "--displayed-score-value",
        type=int,
        help=(
            "strict expected rolling/displayed score at the frozen frame; "
            "defaults to --int32-value for backward compatibility"
        ),
    )
    parser.add_argument(
        "--int32-scan-value",
        type=int,
        help=(
            "optional value for the auxiliary whole-process int32 scan; "
            "defaults to --int32-value and does not alter strict score checks"
        ),
    )
    parser.add_argument(
        "--skip-int32-scan",
        action="store_true",
        help=(
            "For a formal exact-step source with --discover-score, omit "
            "the unrelated whole-process integer scan while retaining the "
            "complete Board and trajectory evidence."
        ),
    )
    parser.add_argument("--maximum-attempts", type=int, default=12)
    parser.add_argument(
        "--trajectory-end-update",
        type=int,
        help=(
            "after the exact probe, single-step and retain raw Board state "
            "through this inclusive framework update"
        ),
    )
    parser.add_argument(
        "--trajectory-start-update",
        type=int,
        help=(
            "diagnostic RNG mode only: single-step without recording from "
            "the frozen probe through the tick before this first retained "
            "update"
        ),
    )
    parser.add_argument(
        "--trajectory-mode",
        choices=("full", "rng"),
        default="full",
        help=(
            "full retains every raw Board object; rng retains a compact "
            "MTRand, shooter, colour-count, and curve-topology stream"
        ),
    )
    parser.add_argument(
        "--trace-global-rng-calls",
        action="store_true",
        help=(
            "diagnostic only: dynamically record each global MTRand caller "
            "while a compact RNG trajectory is single-stepped"
        ),
    )
    parser.add_argument(
        "--global-rng-call-trace-timeout-seconds",
        type=float,
        default=180.0,
        help=(
            "diagnostic global-MTRand caller tracer wall-clock timeout; "
            "must cover the complete retained update interval"
        ),
    )
    parser.add_argument(
        "--trajectory-step-timeout-seconds",
        type=float,
        default=5.0,
        help=(
            "maximum wait for each compact RNG trajectory post-update "
            "barrier; raise only when a read-only debugger intentionally "
            "slows one retail update"
        ),
    )
    parser.add_argument(
        "--diagnostic-thread-crt-state-source",
        type=Path,
        help=(
            "diagnostic only: at the frozen probe, restore the main-thread "
            "CRT rand state from this exact 4-byte source artifact"
        ),
    )
    parser.add_argument(
        "--diagnostic-thread-crt-state-source-update",
        type=int,
        help=(
            "framework update represented by the diagnostic CRT source; "
            "must equal --probe-update"
        ),
    )
    parser.add_argument(
        "--diagnostic-compact-state-source-probe",
        type=Path,
        help=(
            "diagnostic only: restore the frozen compact gameplay/RNG "
            "surface from this hash-bound source memory probe"
        ),
    )
    parser.add_argument(
        "--diagnostic-compact-state-source-probe-sha256",
        help=(
            "required SHA-256 identity of the compact-state source probe"
        ),
    )
    parser.add_argument(
        "--diagnostic-compact-state-source-update",
        type=int,
        help=(
            "framework update represented by the compact-state source; "
            "must equal --probe-update"
        ),
    )
    parser.add_argument(
        "--diagnostic-startup-crt-seed",
        type=int,
        help=(
            "diagnostic only: replace EAX at the initial retail srand call "
            "with a source-observed seed while preserving the strict "
            "command broker"
        ),
    )
    parser.add_argument(
        "--live-rng-monitor-interval-seconds",
        type=float,
        help=(
            "diagnostic only: sample read-only live Board/RNG state until "
            "the exact probe freeze"
        ),
    )
    parser.add_argument(
        "--gameplay-mtrand-oracle",
        type=Path,
        help=(
            "diagnostic only: synchronize the main-thread gameplay MTRand "
            "return stream to this verified source oracle"
        ),
    )
    parser.add_argument(
        "--gameplay-mtrand-oracle-seed",
        type=int,
        help="global MTRand seed used to reconstruct the gameplay oracle",
    )
    parser.add_argument(
        "--gameplay-mtrand-oracle-maximum-draws",
        type=int,
        default=100_000,
        help="maximum seeded draws used to validate the gameplay oracle",
    )
    parser.add_argument(
        "--global-mtrand-call-oracle",
        type=Path,
        help=(
            "diagnostic only: enforce a complete main-thread global MTRand "
            "call transcript after the strict tracer handoff"
        ),
    )
    parser.add_argument(
        "--global-mtrand-call-oracle-seed",
        type=int,
        help="global MTRand seed used to reconstruct the call oracle",
    )
    parser.add_argument(
        "--global-mtrand-call-start-after-update",
        type=int,
        help="exclusive source-oracle update boundary for the handoff",
    )
    parser.add_argument(
        "--global-mtrand-call-end-at-update",
        type=int,
        help="inclusive source-oracle update bound; must end the trajectory",
    )
    parser.add_argument(
        "--global-mtrand-call-maximum-draws",
        type=int,
        default=100_000,
        help="maximum seeded draws used to validate all oracle states",
    )
    parser.add_argument(
        "--global-mtrand-call-sync-timeout-seconds",
        type=float,
        default=900.0,
        help="timeout for debugger handoff and bounded call synchronization",
    )
    parser.add_argument(
        "--global-mtrand-call-observe-boundary-only",
        action="store_true",
        help=(
            "diagnostic only: after the tracer handoff, observe just the "
            "first selected global MTRand wrapper call and return without "
            "writing process memory"
        ),
    )
    parser.add_argument(
        "--global-mtrand-call-observe-thread-runtime",
        action="store_true",
        help=(
            "diagnostic only: after the selected boundary arrives, retain "
            "read-only Windows thread lifecycle and runtime metadata; "
            "requires --global-mtrand-call-observe-boundary-only"
        ),
    )
    parser.add_argument(
        "--global-mtrand-call-observe-handoff-thread-runtime",
        action="store_true",
        help=(
            "diagnostic only: also snapshot read-only Windows thread runtime "
            "before the handoff resume and retain boundary-minus-handoff "
            "deltas; requires --global-mtrand-call-observe-thread-runtime"
        ),
    )
    parser.add_argument(
        "--global-mtrand-call-observe-handoff-rng-state",
        action="store_true",
        help=(
            "diagnostic only: retain the read-only process-global MTRand "
            "state and seeded draw count immediately before the verified "
            "handoff resume; requires --global-mtrand-call-observe-"
            "boundary-only"
        ),
    )
    parser.add_argument(
        "--global-mtrand-call-handoff-rng-wait-target-words-sha256",
        help=(
            "diagnostic only: exact seeded 624-word MTRand block required "
            "by the bounded, read-only handoff wait"
        ),
    )
    parser.add_argument(
        "--global-mtrand-call-handoff-rng-wait-min-index",
        type=int,
        help="minimum natural MTRand index required before handoff resume",
    )
    parser.add_argument(
        "--global-mtrand-call-handoff-rng-wait-timeout-seconds",
        type=float,
        help="finite maximum duration of the read-only handoff wait",
    )
    parser.add_argument(
        "--global-mtrand-call-handoff-rng-wait-poll-interval-seconds",
        type=float,
        help="read-only MTRand polling interval during the handoff wait",
    )
    parser.add_argument(
        "--initial-global-mtrand-oracle",
        type=Path,
        help=(
            "diagnostic only: initialize the process-global MTRand from "
            "source call order zero before any retail wrapper call"
        ),
    )
    parser.add_argument(
        "--initial-global-mtrand-seed",
        type=int,
        help="uint32 seed used to reconstruct the source order-zero state",
    )
    parser.add_argument(
        "--startup-global-mtrand-observation-oracle",
        type=Path,
        help=(
            "diagnostic only: observe every global wrapper call from startup "
            "against this source trace without writing process memory"
        ),
    )
    parser.add_argument(
        "--startup-global-mtrand-observation-seed",
        type=int,
        help="uint32 seed used to reconstruct observed startup draw counts",
    )
    parser.add_argument(
        "--startup-global-mtrand-observation-end-at-update",
        type=int,
        help="inclusive source update boundary for startup call observation",
    )
    parser.add_argument(
        "--startup-global-mtrand-observation-maximum-draws",
        type=int,
        default=100_000,
        help="maximum seeded draws used to reconstruct observed states",
    )
    parser.add_argument(
        "--startup-global-mtrand-hidden-draw-start-after-source-order",
        type=int,
        help=(
            "diagnostic only: include the boundary wrapper draw and observe "
            "all direct MTRand index writes after this source order"
        ),
    )
    parser.add_argument(
        "--startup-global-mtrand-hidden-draw-stop-before-source-order",
        type=int,
        help=(
            "diagnostic only: stop the index-write watch before this next "
            "source wrapper call"
        ),
    )
    parser.add_argument(
        "--diagnostic-disable-source-bound-board-anchor",
        action="store_true",
        help=(
            "diagnostic only: omit the plan's board-call anchor because the "
            "complete global-call oracle owns that call and its suffix"
        ),
    )
    parser.add_argument(
        "--diagnostic-startup-priority-bias-until-update",
        type=int,
        help=(
            "diagnostic only: temporarily apply the tracer's audited "
            "startup priority bias through this framework update"
        ),
    )
    parser.add_argument(
        "--diagnostic-process-affinity-mask",
        type=lambda value: int(value, 0),
        help=(
            "diagnostic only: pin the launched retail process to this "
            "Windows affinity mask for its lifetime"
        ),
    )
    parser.add_argument(
        "--skip-repaint-guard",
        action="store_true",
        help=(
            "diagnostic only: allow a memory trajectory before the "
            "session's evidence-grade title-bar repaint handshake"
        ),
    )
    parser.add_argument(
        "--skip-frozen-snapshot",
        action="store_true",
        help=(
            "diagnostic only: skip the unrelated DXGI screenshot for a "
            "memory-only trajectory; requires --skip-repaint-guard"
        ),
    )
    parser.add_argument(
        "--snapshot-trajectory-frames",
        action="store_true",
        help=(
            "diagnostic only: capture one lossless client-area frame at "
            "each exact post-update RNG trajectory barrier"
        ),
    )
    parser.add_argument(
        "--snapshot-square-dwm-corners",
        action="store_true",
        help=(
            "diagnostic only: verify DWMWCP_DONOTROUND before exact-step "
            "trajectory snapshots; requires --snapshot-trajectory-frames"
        ),
    )
    parser.add_argument(
        "--snapshot-trajectory-warmup-frame",
        action="store_true",
        help=(
            "diagnostic only: capture and retain one discarded DXGI warmup "
            "frame at the frozen probe before the recorded trajectory"
        ),
    )
    parser.add_argument(
        "--formal-exact-step-evidence",
        action="store_true",
        help=(
            "produce a process-bound formal exact-step evidence source; "
            "requires explicit RNG range, per-tick snapshots, verified "
            "square DWM corners, and a discarded warmup frame"
        ),
    )
    parser.add_argument(
        "--formal-full-state-evidence",
        action="store_true",
        help=(
            "produce a process-bound formal exact-step full-state source; "
            "requires a full trajectory, the evidence-grade single-frame "
            "repaint/snapshot anchor, and external-input isolation"
        ),
    )
    parser.add_argument(
        "--diagnostic-source-bound-replay",
        action="store_true",
        help=(
            "diagnostic locator only: retain a compact RNG trajectory while "
            "using the plan's fully verified source-bound command replay, "
            "natural exact freeze, and external-input isolation"
        ),
    )
    parser.add_argument(
        "--trace-detach-update",
        type=int,
        help="diagnostic override: detach the replay tracer at this update",
    )
    parser.add_argument(
        "--trace-reattach-update",
        type=int,
        help="diagnostic override: reattach the replay tracer at this update",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    score_discovery_requested = (
        args.discover_score
        or args.formal_discover_independent_scores
        or args.diagnostic_discover_rolling_score
    )
    diagnostic_thread_crt_values = (
        args.diagnostic_thread_crt_state_source,
        args.diagnostic_thread_crt_state_source_update,
    )
    diagnostic_thread_crt_requested = any(
        value is not None for value in diagnostic_thread_crt_values
    )
    diagnostic_compact_state_values = (
        args.diagnostic_compact_state_source_probe,
        args.diagnostic_compact_state_source_probe_sha256,
        args.diagnostic_compact_state_source_update,
    )
    diagnostic_compact_state_requested = any(
        value is not None for value in diagnostic_compact_state_values
    )
    gameplay_mtrand_requested = (
        args.gameplay_mtrand_oracle is not None
        or args.gameplay_mtrand_oracle_seed is not None
    )
    global_mtrand_call_values = (
        args.global_mtrand_call_oracle,
        args.global_mtrand_call_oracle_seed,
        args.global_mtrand_call_start_after_update,
        args.global_mtrand_call_end_at_update,
    )
    global_mtrand_call_requested = any(
        value is not None for value in global_mtrand_call_values
    )
    global_mtrand_call_handoff_rng_wait_values = (
        args.global_mtrand_call_handoff_rng_wait_target_words_sha256,
        args.global_mtrand_call_handoff_rng_wait_min_index,
        args.global_mtrand_call_handoff_rng_wait_timeout_seconds,
        args.global_mtrand_call_handoff_rng_wait_poll_interval_seconds,
    )
    global_mtrand_call_handoff_rng_wait_enabled = any(
        value is not None
        for value in global_mtrand_call_handoff_rng_wait_values
    )
    initial_global_mtrand_values = (
        args.initial_global_mtrand_oracle,
        args.initial_global_mtrand_seed,
    )
    initial_global_mtrand_requested = any(
        value is not None for value in initial_global_mtrand_values
    )
    startup_global_mtrand_observation_values = (
        args.startup_global_mtrand_observation_oracle,
        args.startup_global_mtrand_observation_seed,
        args.startup_global_mtrand_observation_end_at_update,
    )
    startup_global_mtrand_observation_requested = any(
        value is not None
        for value in startup_global_mtrand_observation_values
    )
    startup_global_mtrand_hidden_draw_values = (
        args.startup_global_mtrand_hidden_draw_start_after_source_order,
        args.startup_global_mtrand_hidden_draw_stop_before_source_order,
    )
    startup_global_mtrand_hidden_draw_requested = any(
        value is not None
        for value in startup_global_mtrand_hidden_draw_values
    )
    trace_override_invalid = (
        (args.trace_detach_update is None)
        != (args.trace_reattach_update is None)
        or (
            args.trace_detach_update is not None
            and (
                args.trace_detach_update < 0
                or args.trace_detach_update >= args.probe_update
                or args.trace_reattach_update
                <= (
                    args.trajectory_end_update
                    if args.trajectory_end_update is not None
                    else args.probe_update
                )
            )
        )
    )
    if (
        args.maximum_attempts < 1
        or args.slowdown_update < 0
        or args.probe_update <= args.slowdown_update
        or not math.isfinite(
            args.global_rng_call_trace_timeout_seconds
        )
        or args.global_rng_call_trace_timeout_seconds <= 0
        or (
            score_discovery_requested
            and (
                args.displayed_score_value is not None
                or args.int32_scan_value is not None
            )
        )
        or (
            args.skip_int32_scan
            and (
                not args.discover_score
                or not args.formal_exact_step_evidence
                or args.int32_scan_value is not None
                or args.trajectory_mode != "rng"
                or args.trajectory_start_update is None
                or args.trajectory_end_update is None
            )
        )
        or (
            args.diagnostic_discover_rolling_score
            and (
                not args.skip_repaint_guard
                or not args.skip_frozen_snapshot
                or args.trajectory_end_update is None
            )
        )
        or (
            args.formal_discover_independent_scores
            and (
                not args.formal_full_state_evidence
                or args.formal_exact_step_evidence
                or args.maximum_attempts != 1
                or args.trajectory_mode != "full"
                or args.trajectory_end_update is None
                or args.skip_repaint_guard
                or args.skip_frozen_snapshot
            )
        )
        or (
            args.live_rng_monitor_interval_seconds is not None
            and (
                not 0 < args.live_rng_monitor_interval_seconds <= 1.0
                or not args.skip_repaint_guard
                or not args.skip_frozen_snapshot
                or args.trajectory_end_update is None
            )
        )
        or (
            args.diagnostic_startup_priority_bias_until_update is not None
            and (
                args.diagnostic_startup_priority_bias_until_update <= 0
                or not args.skip_repaint_guard
                or not args.skip_frozen_snapshot
                or args.trajectory_end_update is None
            )
        )
        or (
            args.diagnostic_process_affinity_mask is not None
            and (
                args.diagnostic_process_affinity_mask <= 0
                or not args.skip_repaint_guard
                or not args.skip_frozen_snapshot
                or args.trajectory_end_update is None
            )
        )
        or trace_override_invalid
        or (
            args.global_mtrand_call_observe_boundary_only
            and not global_mtrand_call_requested
        )
        or (
            args.global_mtrand_call_observe_thread_runtime
            and not args.global_mtrand_call_observe_boundary_only
        )
        or (
            args.global_mtrand_call_observe_handoff_thread_runtime
            and not args.global_mtrand_call_observe_thread_runtime
        )
        or (
            args.global_mtrand_call_observe_handoff_rng_state
            and not args.global_mtrand_call_observe_boundary_only
        )
        or (
            global_mtrand_call_handoff_rng_wait_enabled
            and (
                any(
                    value is None
                    for value in global_mtrand_call_handoff_rng_wait_values
                )
                or not args.global_mtrand_call_observe_boundary_only
                or not args.global_mtrand_call_observe_handoff_rng_state
                or not _is_canonical_sha256(
                    args.global_mtrand_call_handoff_rng_wait_target_words_sha256
                )
                or isinstance(
                    args.global_mtrand_call_handoff_rng_wait_min_index,
                    bool,
                )
                or not isinstance(
                    args.global_mtrand_call_handoff_rng_wait_min_index,
                    int,
                )
                or not 0
                <= args.global_mtrand_call_handoff_rng_wait_min_index
                <= MTRAND_STATE_WORDS
                or args.global_mtrand_call_handoff_rng_wait_timeout_seconds
                <= 0
                or args.global_mtrand_call_handoff_rng_wait_poll_interval_seconds
                <= 0
                or args.global_mtrand_call_handoff_rng_wait_poll_interval_seconds
                > args.global_mtrand_call_handoff_rng_wait_timeout_seconds
            )
        )
        or (
            gameplay_mtrand_requested
            and (
                args.gameplay_mtrand_oracle is None
                or args.gameplay_mtrand_oracle_seed is None
                or not 0 <= args.gameplay_mtrand_oracle_seed <= 0xFFFFFFFF
                or args.gameplay_mtrand_oracle_maximum_draws <= 0
                or not args.skip_repaint_guard
                or not args.skip_frozen_snapshot
                or args.trajectory_end_update is None
            )
        )
        or (
            global_mtrand_call_requested
            and (
                any(
                    value is None
                    for value in global_mtrand_call_values
                )
                or args.global_mtrand_call_oracle is None
                or args.global_mtrand_call_oracle_seed is None
                or not 0
                <= args.global_mtrand_call_oracle_seed
                <= 0xFFFFFFFF
                or args.global_mtrand_call_start_after_update is None
                or args.global_mtrand_call_start_after_update < 0
                or args.global_mtrand_call_end_at_update is None
                or args.global_mtrand_call_end_at_update
                <= args.global_mtrand_call_start_after_update
                or args.global_mtrand_call_maximum_draws <= 0
                or args.global_mtrand_call_sync_timeout_seconds <= 0
                or args.trajectory_mode != "rng"
                or args.trajectory_end_update is None
                or (
                    not args.global_mtrand_call_observe_boundary_only
                    and args.global_mtrand_call_end_at_update
                    != args.trajectory_end_update
                )
                or (
                    args.global_mtrand_call_observe_boundary_only
                    and args.global_mtrand_call_end_at_update
                    > args.trajectory_end_update
                )
                or args.trace_detach_update is None
                or args.trace_detach_update
                > args.global_mtrand_call_start_after_update
                or args.trace_reattach_update is None
                or not args.skip_repaint_guard
                or not args.skip_frozen_snapshot
                or gameplay_mtrand_requested
                or args.trace_global_rng_calls
            )
        )
        or (
            initial_global_mtrand_requested
            and (
                any(
                    value is None
                    for value in initial_global_mtrand_values
                )
                or args.initial_global_mtrand_oracle is None
                or not args.initial_global_mtrand_oracle.is_file()
                or args.initial_global_mtrand_seed is None
                or not 0
                <= args.initial_global_mtrand_seed
                <= 0xFFFFFFFF
                or not global_mtrand_call_requested
                or args.global_mtrand_call_oracle is None
                or args.initial_global_mtrand_oracle.resolve()
                != args.global_mtrand_call_oracle.resolve()
                or args.initial_global_mtrand_seed
                != args.global_mtrand_call_oracle_seed
                or not args.diagnostic_disable_source_bound_board_anchor
            )
        )
        or (
            startup_global_mtrand_observation_requested
            and (
                any(
                    value is None
                    for value in startup_global_mtrand_observation_values
                )
                or args.startup_global_mtrand_observation_oracle is None
                or not args.startup_global_mtrand_observation_oracle.is_file()
                or args.startup_global_mtrand_observation_seed is None
                or not 0
                <= args.startup_global_mtrand_observation_seed
                <= 0xFFFFFFFF
                or args.startup_global_mtrand_observation_end_at_update
                is None
                or args.startup_global_mtrand_observation_end_at_update < 0
                or args.startup_global_mtrand_observation_end_at_update
                >= args.probe_update
                or args.startup_global_mtrand_observation_maximum_draws <= 0
                or args.trajectory_mode != "rng"
                or args.trajectory_end_update is None
                or not args.skip_repaint_guard
                or not args.skip_frozen_snapshot
                or gameplay_mtrand_requested
                or global_mtrand_call_requested
                or initial_global_mtrand_requested
                or args.trace_global_rng_calls
                or not args.diagnostic_disable_source_bound_board_anchor
            )
        )
        or (
            startup_global_mtrand_hidden_draw_requested
            and (
                any(
                    value is None
                    for value in startup_global_mtrand_hidden_draw_values
                )
                or not startup_global_mtrand_observation_requested
                or any(
                    not isinstance(value, int)
                    or isinstance(value, bool)
                    or value < 0
                    for value in startup_global_mtrand_hidden_draw_values
                )
            )
        )
        or (
            args.diagnostic_disable_source_bound_board_anchor
            and not (
                global_mtrand_call_requested
                or startup_global_mtrand_observation_requested
            )
        )
        or (
            args.trace_global_rng_calls
            and (
                args.trajectory_mode != "rng"
                or args.trajectory_end_update is None
            )
        )
        or (
            diagnostic_thread_crt_requested
            and (
                any(
                    value is None
                    for value in diagnostic_thread_crt_values
                )
                or args.diagnostic_thread_crt_state_source is None
                or not args.diagnostic_thread_crt_state_source.is_file()
                or args.diagnostic_thread_crt_state_source_update
                != args.probe_update
                or args.trajectory_mode != "rng"
                or args.trajectory_start_update is None
                or args.trajectory_end_update is None
                or not args.trace_global_rng_calls
                or not args.skip_repaint_guard
                or not args.skip_frozen_snapshot
                or score_discovery_requested
                or args.formal_exact_step_evidence
                or args.formal_full_state_evidence
                or gameplay_mtrand_requested
                or global_mtrand_call_requested
                or initial_global_mtrand_requested
                or startup_global_mtrand_observation_requested
            )
        )
        or (
            diagnostic_compact_state_requested
            and (
                any(
                    value is None
                    for value in diagnostic_compact_state_values
                )
                or args.diagnostic_compact_state_source_probe is None
                or not args.diagnostic_compact_state_source_probe.is_file()
                or not _is_canonical_sha256(
                    args.diagnostic_compact_state_source_probe_sha256
                )
                or (
                    args.diagnostic_compact_state_source_probe is not None
                    and _sha256_path(
                        args.diagnostic_compact_state_source_probe.resolve()
                    )
                    != args.diagnostic_compact_state_source_probe_sha256
                )
                or args.diagnostic_compact_state_source_update
                != args.probe_update
                or diagnostic_thread_crt_requested
                or args.diagnostic_startup_crt_seed is None
                or args.trajectory_mode != "rng"
                or args.trajectory_start_update is None
                or args.trajectory_end_update is None
                or not args.trace_global_rng_calls
                or not args.skip_repaint_guard
                or not args.skip_frozen_snapshot
                or score_discovery_requested
                or args.formal_exact_step_evidence
                or args.formal_full_state_evidence
                or gameplay_mtrand_requested
                or global_mtrand_call_requested
                or initial_global_mtrand_requested
                or startup_global_mtrand_observation_requested
            )
        )
        or (
            args.diagnostic_startup_crt_seed is not None
            and (
                not 0
                <= args.diagnostic_startup_crt_seed
                <= 0xFFFFFFFF
                or diagnostic_thread_crt_requested
                or args.trajectory_mode != "rng"
                or args.trajectory_start_update is None
                or args.trajectory_end_update is None
                or not args.trace_global_rng_calls
                or not args.skip_repaint_guard
                or not args.skip_frozen_snapshot
                or score_discovery_requested
                or args.formal_exact_step_evidence
                or args.formal_full_state_evidence
                or gameplay_mtrand_requested
                or global_mtrand_call_requested
                or initial_global_mtrand_requested
                or startup_global_mtrand_observation_requested
            )
        )
        or (
            args.trajectory_start_update is not None
            and (
                args.trajectory_mode != "rng"
                or args.trajectory_end_update is None
                or args.trajectory_start_update < args.probe_update
                or args.trajectory_start_update
                > args.trajectory_end_update
                or not args.skip_repaint_guard
                or not args.skip_frozen_snapshot
            )
        )
        or (
            args.snapshot_trajectory_frames
            and (
                args.trajectory_mode != "rng"
                or args.trajectory_end_update is None
                or not args.skip_repaint_guard
                or not args.skip_frozen_snapshot
            )
        )
        or (
            args.snapshot_square_dwm_corners
            and not args.snapshot_trajectory_frames
        )
        or (
            args.snapshot_trajectory_warmup_frame
            and not args.snapshot_trajectory_frames
        )
        or (
            args.formal_exact_step_evidence
            and (
                args.trajectory_mode != "rng"
                or args.trajectory_end_update is None
                or args.trajectory_start_update is None
                or not args.skip_repaint_guard
                or not args.skip_frozen_snapshot
                or not args.snapshot_trajectory_frames
                or not args.snapshot_square_dwm_corners
                or not args.snapshot_trajectory_warmup_frame
                or args.trace_global_rng_calls
                or args.trace_detach_update is not None
                or args.trace_reattach_update is not None
                or args.diagnostic_discover_rolling_score
                or args.formal_discover_independent_scores
                or args.live_rng_monitor_interval_seconds is not None
                or gameplay_mtrand_requested
                or global_mtrand_call_requested
                or initial_global_mtrand_requested
                or startup_global_mtrand_observation_requested
                or args.diagnostic_disable_source_bound_board_anchor
                or args.diagnostic_startup_priority_bias_until_update
                is not None
                or args.diagnostic_process_affinity_mask is not None
            )
        )
        or (
            args.formal_full_state_evidence
            and (
                args.formal_exact_step_evidence
                or args.trajectory_mode != "full"
                or args.trajectory_end_update is None
                or args.trajectory_start_update is not None
                or args.skip_repaint_guard
                or args.skip_frozen_snapshot
                or args.snapshot_trajectory_frames
                or args.snapshot_square_dwm_corners
                or args.snapshot_trajectory_warmup_frame
                or args.trace_global_rng_calls
                or args.trace_detach_update is not None
                or args.trace_reattach_update is not None
                or args.diagnostic_discover_rolling_score
                or args.live_rng_monitor_interval_seconds is not None
                or gameplay_mtrand_requested
                or global_mtrand_call_requested
                or initial_global_mtrand_requested
                or startup_global_mtrand_observation_requested
                or args.diagnostic_disable_source_bound_board_anchor
                or args.diagnostic_startup_priority_bias_until_update
                is not None
                or args.diagnostic_process_affinity_mask is not None
            )
        )
        or (
            args.trajectory_end_update is not None
            and (
                args.trajectory_end_update < args.probe_update
                or args.trajectory_end_update
                - args.probe_update
                + 1
                > MAX_TRAJECTORY_TICKS
            )
        )
    ):
        raise SystemExit("invalid probe timing or retry count")
    if diagnostic_compact_state_requested:
        diagnostic_mutator = _diagnostic_compact_state_mutator(
            source_probe_path=(
                args.diagnostic_compact_state_source_probe.resolve()
            ),
            expected_probe_sha256=(
                args.diagnostic_compact_state_source_probe_sha256
            ),
            source_framework_update=(
                args.diagnostic_compact_state_source_update
            ),
        )
    elif diagnostic_thread_crt_requested:
        diagnostic_mutator = _diagnostic_thread_crt_state_mutator(
            source_path=(
                args.diagnostic_thread_crt_state_source.resolve()
            ),
            source_framework_update=(
                args.diagnostic_thread_crt_state_source_update
            ),
        )
    else:
        diagnostic_mutator = None
    path = collect_probe(
        plan_path=args.plan.resolve(),
        prestate_path=args.prestate.resolve(),
        host_restore_path=args.host_restore.resolve(),
        output_root=args.output_root.resolve(),
        probe_update=args.probe_update,
        slowdown_update=args.slowdown_update,
        int32_value=(
            None
            if score_discovery_requested
            else args.int32_value
        ),
        displayed_score_value=args.displayed_score_value,
        int32_scan_value=args.int32_scan_value,
        skip_int32_scan=args.skip_int32_scan,
        maximum_attempts=args.maximum_attempts,
        trajectory_end_update=args.trajectory_end_update,
        trajectory_start_update=args.trajectory_start_update,
        trajectory_mode=args.trajectory_mode,
        trace_global_rng_calls=args.trace_global_rng_calls,
        global_rng_call_trace_timeout_seconds=(
            args.global_rng_call_trace_timeout_seconds
        ),
        trajectory_step_timeout_seconds=(
            args.trajectory_step_timeout_seconds
        ),
        skip_repaint_guard=args.skip_repaint_guard,
        skip_frozen_snapshot=args.skip_frozen_snapshot,
        snapshot_trajectory_frames=args.snapshot_trajectory_frames,
        snapshot_square_dwm_corners=(
            args.snapshot_square_dwm_corners
        ),
        snapshot_trajectory_warmup_frame=(
            args.snapshot_trajectory_warmup_frame
        ),
        formal_exact_step_evidence=args.formal_exact_step_evidence,
        formal_full_state_evidence=args.formal_full_state_evidence,
        diagnostic_source_bound_replay=(
            args.diagnostic_source_bound_replay
        ),
        trace_detach_update=args.trace_detach_update,
        trace_reattach_update=args.trace_reattach_update,
        allow_discovered_display_mismatch=(
            args.diagnostic_discover_rolling_score
            or args.formal_discover_independent_scores
        ),
        formal_independent_score_binding=(
            args.formal_discover_independent_scores
        ),
        live_rng_monitor_interval_seconds=(
            args.live_rng_monitor_interval_seconds
        ),
        gameplay_mtrand_oracle=(
            args.gameplay_mtrand_oracle.resolve()
            if args.gameplay_mtrand_oracle is not None
            else None
        ),
        gameplay_mtrand_oracle_seed=(
            args.gameplay_mtrand_oracle_seed
        ),
        gameplay_mtrand_oracle_maximum_draws=(
            args.gameplay_mtrand_oracle_maximum_draws
        ),
        global_mtrand_call_oracle=(
            args.global_mtrand_call_oracle.resolve()
            if args.global_mtrand_call_oracle is not None
            else None
        ),
        global_mtrand_call_oracle_seed=(
            args.global_mtrand_call_oracle_seed
        ),
        global_mtrand_call_start_after_update=(
            args.global_mtrand_call_start_after_update
        ),
        global_mtrand_call_end_at_update=(
            args.global_mtrand_call_end_at_update
        ),
        global_mtrand_call_maximum_draws=(
            args.global_mtrand_call_maximum_draws
        ),
        global_mtrand_call_sync_timeout_seconds=(
            args.global_mtrand_call_sync_timeout_seconds
        ),
        global_mtrand_call_observe_boundary_only=(
            args.global_mtrand_call_observe_boundary_only
        ),
        global_mtrand_call_observe_thread_runtime=(
            args.global_mtrand_call_observe_thread_runtime
        ),
        global_mtrand_call_observe_handoff_thread_runtime=(
            args.global_mtrand_call_observe_handoff_thread_runtime
        ),
        global_mtrand_call_observe_handoff_rng_state=(
            args.global_mtrand_call_observe_handoff_rng_state
        ),
        global_mtrand_call_handoff_rng_wait_target_words_sha256=(
            args.global_mtrand_call_handoff_rng_wait_target_words_sha256
        ),
        global_mtrand_call_handoff_rng_wait_min_index=(
            args.global_mtrand_call_handoff_rng_wait_min_index
        ),
        global_mtrand_call_handoff_rng_wait_timeout_seconds=(
            args.global_mtrand_call_handoff_rng_wait_timeout_seconds
        ),
        global_mtrand_call_handoff_rng_wait_poll_interval_seconds=(
            args.global_mtrand_call_handoff_rng_wait_poll_interval_seconds
        ),
        initial_global_mtrand_oracle=(
            args.initial_global_mtrand_oracle.resolve()
            if args.initial_global_mtrand_oracle is not None
            else None
        ),
        initial_global_mtrand_seed=args.initial_global_mtrand_seed,
        startup_global_mtrand_observation_oracle=(
            args.startup_global_mtrand_observation_oracle.resolve()
            if args.startup_global_mtrand_observation_oracle is not None
            else None
        ),
        startup_global_mtrand_observation_seed=(
            args.startup_global_mtrand_observation_seed
        ),
        startup_global_mtrand_observation_end_at_update=(
            args.startup_global_mtrand_observation_end_at_update
        ),
        startup_global_mtrand_observation_maximum_draws=(
            args.startup_global_mtrand_observation_maximum_draws
        ),
        startup_global_mtrand_hidden_draw_start_after_source_order=(
            args.startup_global_mtrand_hidden_draw_start_after_source_order
        ),
        startup_global_mtrand_hidden_draw_stop_before_source_order=(
            args.startup_global_mtrand_hidden_draw_stop_before_source_order
        ),
        diagnostic_disable_source_bound_board_anchor=(
            args.diagnostic_disable_source_bound_board_anchor
        ),
        startup_priority_bias_until_update=(
            args.diagnostic_startup_priority_bias_until_update
        ),
        process_affinity_mask=args.diagnostic_process_affinity_mask,
        diagnostic_startup_crt_seed=(
            args.diagnostic_startup_crt_seed
        ),
        diagnostic_source_bound_compact_restore=(
            diagnostic_compact_state_requested
        ),
        diagnostic_mutator=diagnostic_mutator,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
