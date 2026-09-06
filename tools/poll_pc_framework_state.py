"""High-rate, diagnostic-only observer for PopCap framework draw events."""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

if __package__:
    from .capture_dxgi import (
        CaptureError,
        FrameworkStateSnapshot,
        FrameworkUpdateReader,
        validate_process_name,
    )
else:
    from capture_dxgi import (
        CaptureError,
        FrameworkStateSnapshot,
        FrameworkUpdateReader,
        validate_process_name,
    )


SCHEMA = "zuma-rl.pc-framework-state-poll-diagnostic"
VERSION = 1
READY_SCHEMA = "zuma-rl.pc-framework-state-poll-ready"
STOP_SCHEMA = "zuma-rl.pc-framework-state-poll-stop"
DEFAULT_MAX_DURATION_SECONDS = 20.0
MAX_TRANSITIONS = 1_000_000
_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_TRACKED_FIELDS = (
    "draw_count",
    "update_count",
    "last_draw_tick",
    "next_draw_tick",
    "is_drawing",
    "has_pending_draw",
    "update_app_state",
    "sleep_count",
)


class FrameworkPollError(RuntimeError):
    """A fixed-code high-rate observer failure."""

    def __init__(self, code: str) -> None:
        if re.fullmatch(r"[a-z0-9_]{1,80}", code) is None:
            code = "framework_poll_failed"
        self.code = code
        super().__init__(code)


def _fail(code: str) -> None:
    raise FrameworkPollError(code)


class StateReader(Protocol):
    def sample_state(self) -> FrameworkStateSnapshot: ...


@dataclass(frozen=True)
class PollTransition:
    transition_index: int
    poll_index: int
    previous_sample_after_perf_counter_ns: int
    sample_before_perf_counter_ns: int
    sample_after_perf_counter_ns: int
    changed_fields: tuple[str, ...]
    state: FrameworkStateSnapshot


@dataclass(frozen=True)
class DrawEvent:
    event_index: int
    transition_index: int
    poll_index: int
    previous_draw_count: int
    draw_count: int
    draw_count_delta: int
    update_count: int
    previous_sample_after_perf_counter_ns: int
    sample_before_perf_counter_ns: int
    sample_after_perf_counter_ns: int


@dataclass(frozen=True)
class PollResult:
    ready_sample_before_perf_counter_ns: int
    ready_sample_after_perf_counter_ns: int
    ready_state: FrameworkStateSnapshot
    prearm_sample_count: int
    arm_at_framework_update: int | None
    target_poll_hz: float | None
    poll_start_perf_counter_ns: int
    poll_end_perf_counter_ns: int
    poll_count: int
    initial_sample_before_perf_counter_ns: int
    initial_sample_after_perf_counter_ns: int
    initial_state: FrameworkStateSnapshot
    final_state: FrameworkStateSnapshot
    transitions: tuple[PollTransition, ...]
    draw_events: tuple[DrawEvent, ...]
    sample_latency_ns: tuple[int, ...]
    sample_completion_interval_ns: tuple[int, ...]
    arm_no_later_than_framework_update: int | None = None
    armed_at_framework_update: int | None = None


def _changed_fields(
    previous: FrameworkStateSnapshot,
    current: FrameworkStateSnapshot,
) -> tuple[str, ...]:
    return tuple(
        field
        for field in _TRACKED_FIELDS
        if getattr(previous, field) != getattr(current, field)
    )


def _wait_until_deadline(
    deadline_ns: int,
    *,
    clock_ns: Callable[[], int],
    sleep: Callable[[float], None],
) -> None:
    """Wait without issuing extra cross-process reads near the deadline."""

    while True:
        remaining_ns = deadline_ns - clock_ns()
        if remaining_ns <= 0:
            return
        # Python/Windows short sleeps can overshoot by several milliseconds.
        # Sleep only for long gaps, then spin on this dedicated physical core.
        if remaining_ns > 2_000_000:
            sleep((remaining_ns - 1_000_000) / 1_000_000_000)


def poll_framework_state(
    reader: StateReader,
    *,
    stop_requested: Callable[[], bool],
    on_ready: Callable[[int, int, FrameworkStateSnapshot], None],
    max_duration_seconds: float = DEFAULT_MAX_DURATION_SECONDS,
    arm_at_framework_update: int | None = None,
    arm_no_later_than_framework_update: int | None = None,
    arm_timeout_seconds: float = 600.0,
    target_poll_hz: float | None = None,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
    sleep: Callable[[float], None] = time.sleep,
    control_check_interval: int = 64,
) -> PollResult:
    """Busy-poll scheduling state and retain every tracked transition."""

    if (
        isinstance(max_duration_seconds, bool)
        or not isinstance(max_duration_seconds, (int, float))
        or not math.isfinite(float(max_duration_seconds))
        or not 0.25 <= float(max_duration_seconds) <= 120.0
        or isinstance(control_check_interval, bool)
        or not isinstance(control_check_interval, int)
        or not 1 <= control_check_interval <= 4096
        or (
            arm_at_framework_update is not None
            and (
                isinstance(arm_at_framework_update, bool)
                or not isinstance(arm_at_framework_update, int)
                or arm_at_framework_update < 0
            )
        )
        or (
            arm_no_later_than_framework_update is not None
            and (
                arm_at_framework_update is None
                or isinstance(
                    arm_no_later_than_framework_update,
                    bool,
                )
                or not isinstance(
                    arm_no_later_than_framework_update,
                    int,
                )
                or arm_no_later_than_framework_update
                < arm_at_framework_update
            )
        )
        or isinstance(arm_timeout_seconds, bool)
        or not isinstance(arm_timeout_seconds, (int, float))
        or not math.isfinite(float(arm_timeout_seconds))
        or not 1.0 <= float(arm_timeout_seconds) <= 1800.0
        or (
            target_poll_hz is not None
            and (
                isinstance(target_poll_hz, bool)
                or not isinstance(target_poll_hz, (int, float))
                or not math.isfinite(float(target_poll_hz))
                or not 1_000.0 <= float(target_poll_hz) <= 20_000.0
            )
        )
    ):
        _fail("framework_poll_configuration_invalid")
    maximum_ns = int(float(max_duration_seconds) * 1_000_000_000)

    initial_before = clock_ns()
    try:
        initial_state = reader.sample_state()
    except CaptureError as error:
        raise FrameworkPollError(error.code) from None
    initial_after = clock_ns()
    if (
        initial_before <= 0
        or initial_after < initial_before
        or not isinstance(initial_state, FrameworkStateSnapshot)
    ):
        _fail("framework_poll_clock_invalid")
    on_ready(initial_before, initial_after, initial_state)

    ready_before = initial_before
    ready_after = initial_after
    ready_state = initial_state
    prearm_sample_count = 1
    armed_at_framework_update: int | None = None
    if arm_at_framework_update is not None:
        arm_upper_update = (
            arm_at_framework_update
            if arm_no_later_than_framework_update is None
            else arm_no_later_than_framework_update
        )
        arm_deadline_ns = (
            ready_before + int(float(arm_timeout_seconds) * 1_000_000_000)
        )
        while not (
            arm_at_framework_update
            <= initial_state.update_count
            <= arm_upper_update
            and initial_state.update_app_state == 3
            and not initial_state.is_drawing
            and not initial_state.has_pending_draw
        ):
            if initial_state.update_count > arm_upper_update:
                _fail(
                    "framework_poll_arm_update_missed"
                    if arm_upper_update == arm_at_framework_update
                    else "framework_poll_arm_window_missed"
                )
            if stop_requested():
                _fail("framework_poll_stopped_before_arm")
            if initial_after >= arm_deadline_ns:
                _fail("framework_poll_arm_timeout")
            sleep(0.001)
            initial_before = clock_ns()
            try:
                initial_state = reader.sample_state()
            except CaptureError as error:
                raise FrameworkPollError(error.code) from None
            initial_after = clock_ns()
            prearm_sample_count += 1
            if initial_after < initial_before:
                _fail("framework_poll_clock_invalid")
        armed_at_framework_update = initial_state.update_count

    poll_count = 1
    prior_state = initial_state
    prior_after = initial_after
    transitions: list[PollTransition] = []
    draw_events: list[DrawEvent] = []
    latencies = [initial_after - initial_before]
    intervals: list[int] = []
    period_ns = (
        None
        if target_poll_hz is None
        else max(1, round(1_000_000_000 / float(target_poll_hz)))
    )
    next_sample_ns = (
        None if period_ns is None else initial_after + period_ns
    )
    while True:
        if poll_count % control_check_interval == 0:
            if stop_requested():
                break
            if prior_after - initial_before >= maximum_ns:
                _fail("framework_poll_timeout")

        if next_sample_ns is not None:
            _wait_until_deadline(
                next_sample_ns,
                clock_ns=clock_ns,
                sleep=sleep,
            )
            sample_before = clock_ns()
        else:
            sample_before = clock_ns()
        try:
            state = reader.sample_state()
        except CaptureError as error:
            raise FrameworkPollError(error.code) from None
        sample_after = clock_ns()
        if (
            sample_before < prior_after
            or sample_after < sample_before
            or not isinstance(state, FrameworkStateSnapshot)
        ):
            _fail("framework_poll_clock_invalid")
        if (
            state.draw_count < prior_state.draw_count
            or state.update_count < prior_state.update_count
            or state.sleep_count < prior_state.sleep_count
        ):
            _fail("framework_poll_counter_decreased")

        intervals.append(sample_after - prior_after)
        latencies.append(sample_after - sample_before)
        changed = _changed_fields(prior_state, state)
        if changed:
            transition_index = len(transitions)
            transitions.append(
                PollTransition(
                    transition_index=transition_index,
                    poll_index=poll_count,
                    previous_sample_after_perf_counter_ns=prior_after,
                    sample_before_perf_counter_ns=sample_before,
                    sample_after_perf_counter_ns=sample_after,
                    changed_fields=changed,
                    state=state,
                )
            )
            if len(transitions) > MAX_TRANSITIONS:
                _fail("framework_poll_transition_limit_exceeded")
            if state.draw_count != prior_state.draw_count:
                delta = state.draw_count - prior_state.draw_count
                draw_events.append(
                    DrawEvent(
                        event_index=len(draw_events),
                        transition_index=transition_index,
                        poll_index=poll_count,
                        previous_draw_count=prior_state.draw_count,
                        draw_count=state.draw_count,
                        draw_count_delta=delta,
                        update_count=state.update_count,
                        previous_sample_after_perf_counter_ns=prior_after,
                        sample_before_perf_counter_ns=sample_before,
                        sample_after_perf_counter_ns=sample_after,
                    )
                )
        prior_state = state
        prior_after = sample_after
        poll_count += 1
        if next_sample_ns is not None:
            next_sample_ns += int(period_ns)
            if next_sample_ns <= sample_after:
                missed_periods = (
                    (sample_after - next_sample_ns) // int(period_ns) + 1
                )
                next_sample_ns += missed_periods * int(period_ns)

    return PollResult(
        ready_sample_before_perf_counter_ns=ready_before,
        ready_sample_after_perf_counter_ns=ready_after,
        ready_state=ready_state,
        prearm_sample_count=prearm_sample_count,
        arm_at_framework_update=arm_at_framework_update,
        target_poll_hz=(
            None if target_poll_hz is None else float(target_poll_hz)
        ),
        poll_start_perf_counter_ns=initial_before,
        poll_end_perf_counter_ns=prior_after,
        poll_count=poll_count,
        initial_sample_before_perf_counter_ns=initial_before,
        initial_sample_after_perf_counter_ns=initial_after,
        initial_state=initial_state,
        final_state=prior_state,
        transitions=tuple(transitions),
        draw_events=tuple(draw_events),
        sample_latency_ns=tuple(latencies),
        sample_completion_interval_ns=tuple(intervals),
        arm_no_later_than_framework_update=(
            arm_at_framework_update
            if arm_at_framework_update is not None
            and arm_no_later_than_framework_update is None
            else arm_no_later_than_framework_update
        ),
        armed_at_framework_update=armed_at_framework_update,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except OSError:
        _fail("framework_poll_artifact_read_failed")
    return f"sha256:{digest.hexdigest()}"


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("ascii")
    except (OverflowError, TypeError, UnicodeError, ValueError):
        _fail("framework_poll_encode_failed")


def _write_exclusive(path: Path, payload: bytes, error_code: str) -> str:
    try:
        with path.open("xb") as stream:
            if stream.write(payload) != len(payload):
                _fail(error_code)
            stream.flush()
            os.fsync(stream.fileno())
    except FrameworkPollError:
        raise
    except OSError:
        _fail(error_code)
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _publish_output(path: Path, payload: bytes) -> str:
    part = path.with_name(path.name + ".part")
    try:
        with part.open("xb") as stream:
            if stream.write(payload) != len(payload):
                _fail("framework_poll_output_write_failed")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(part, path)
    except FrameworkPollError:
        raise
    except OSError:
        _fail("framework_poll_output_publish_failed")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _read_json(path: Path, error_code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        _fail(error_code)
    if not isinstance(value, dict):
        _fail(error_code)
    return value


def _prepare_paths(
    *,
    output: Path,
    ready: Path,
    stop: Path,
    capture_metadata: Path,
    framework_update_map: Path,
    framework_state_sidecar: Path,
) -> None:
    outputs = (output, ready, stop)
    if len({path.resolve() for path in outputs}) != len(outputs):
        _fail("framework_poll_path_invalid")
    for path in outputs:
        part = path.with_name(path.name + ".part")
        try:
            if (
                not path.is_absolute()
                or not path.parent.is_dir()
                or path.parent.is_symlink()
                or path.exists()
                or part.exists()
            ):
                _fail("framework_poll_path_invalid")
            if os.name == "nt":
                str(path).encode("ascii", errors="strict")
        except FrameworkPollError:
            raise
        except (OSError, UnicodeError):
            _fail("framework_poll_path_invalid")
    if output.suffix.casefold() != ".json" or ready.suffix.casefold() != ".json":
        _fail("framework_poll_path_invalid")
    for path in (
        capture_metadata,
        framework_update_map,
        framework_state_sidecar,
    ):
        if not path.is_absolute() or path.suffix.casefold() != ".json":
            _fail("framework_poll_path_invalid")


def _hash_executable(path: Path) -> str:
    try:
        if not path.is_file() or not 0 < path.stat().st_size <= 512 * 1024**2:
            _fail("framework_poll_process_identity_invalid")
    except OSError:
        _fail("framework_poll_process_identity_invalid")
    return _sha256_file(path)


def windows_process_identity(process_id: int, process_name: str) -> dict[str, Any]:
    if os.name != "nt":
        _fail("windows_required")
    if (
        isinstance(process_id, bool)
        or not isinstance(process_id, int)
        or process_id <= 0
    ):
        _fail("framework_poll_process_identity_invalid")
    try:
        process_name = validate_process_name(process_name)
    except CaptureError as error:
        raise FrameworkPollError(error.code) from None
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = (
        wintypes.DWORD,
        wintypes.BOOL,
        wintypes.DWORD,
    )
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.QueryFullProcessImageNameW.argtypes = (
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD),
    )
    kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
    kernel32.GetProcessTimes.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME),
    )
    kernel32.GetProcessTimes.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL

    process = kernel32.OpenProcess(0x1000, False, process_id)
    if not process:
        _fail("framework_poll_process_identity_unavailable")
    try:
        capacity = wintypes.DWORD(32768)
        image_buffer = ctypes.create_unicode_buffer(capacity.value)
        creation = wintypes.FILETIME()
        exit_time = wintypes.FILETIME()
        kernel_time = wintypes.FILETIME()
        user_time = wintypes.FILETIME()
        if not kernel32.QueryFullProcessImageNameW(
            process,
            0,
            image_buffer,
            ctypes.byref(capacity),
        ) or not kernel32.GetProcessTimes(
            process,
            ctypes.byref(creation),
            ctypes.byref(exit_time),
            ctypes.byref(kernel_time),
            ctypes.byref(user_time),
        ):
            _fail("framework_poll_process_identity_unavailable")
    finally:
        kernel32.CloseHandle(process)
    image_path = Path(image_buffer.value)
    if image_path.name.casefold() != process_name.casefold():
        _fail("framework_poll_process_identity_mismatch")
    creation_filetime = (
        int(creation.dwHighDateTime) << 32
    ) | int(creation.dwLowDateTime)
    if creation_filetime <= 0:
        _fail("framework_poll_process_identity_invalid")
    return {
        "process_id": process_id,
        "process_creation_filetime_100ns": creation_filetime,
        "executable_path": str(image_path.resolve()),
        "executable_sha256": _hash_executable(image_path),
    }


def set_current_process_affinity(mask: int) -> None:
    if os.name != "nt":
        _fail("windows_required")
    if (
        isinstance(mask, bool)
        or not isinstance(mask, int)
        or mask <= 0
        or mask > 0xFFFFFFFFFFFFFFFF
    ):
        _fail("framework_poll_affinity_invalid")
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.argtypes = ()
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.SetProcessAffinityMask.argtypes = (
        wintypes.HANDLE,
        ctypes.c_size_t,
    )
    kernel32.SetProcessAffinityMask.restype = wintypes.BOOL
    if not kernel32.SetProcessAffinityMask(
        kernel32.GetCurrentProcess(), ctypes.c_size_t(mask)
    ):
        _fail("framework_poll_affinity_failed")


def windows_processor_core_masks() -> tuple[int, ...]:
    """Return group-zero logical masks, one per physical processor core."""

    if os.name != "nt":
        _fail("windows_required")
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetLogicalProcessorInformationEx.argtypes = (
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.POINTER(wintypes.DWORD),
    )
    kernel32.GetLogicalProcessorInformationEx.restype = wintypes.BOOL
    length = wintypes.DWORD()
    kernel32.GetLogicalProcessorInformationEx(
        0, None, ctypes.byref(length)
    )
    if length.value < 32:
        _fail("framework_poll_topology_unavailable")
    payload = ctypes.create_string_buffer(length.value)
    if not kernel32.GetLogicalProcessorInformationEx(
        0, payload, ctypes.byref(length)
    ):
        _fail("framework_poll_topology_unavailable")
    raw = payload.raw[: length.value]
    offset = 0
    masks: list[int] = []
    while offset < len(raw):
        if len(raw) - offset < 8:
            _fail("framework_poll_topology_invalid")
        relationship = int.from_bytes(raw[offset : offset + 4], "little")
        size = int.from_bytes(raw[offset + 4 : offset + 8], "little")
        if size < 32 or offset + size > len(raw):
            _fail("framework_poll_topology_invalid")
        if relationship == 0:
            group_count = int.from_bytes(
                raw[offset + 30 : offset + 32], "little"
            )
            if group_count <= 0 or 32 + group_count * 16 > size:
                _fail("framework_poll_topology_invalid")
            for group_index in range(group_count):
                group_offset = offset + 32 + group_index * 16
                mask = int.from_bytes(
                    raw[group_offset : group_offset + 8], "little"
                )
                group = int.from_bytes(
                    raw[group_offset + 8 : group_offset + 10], "little"
                )
                if group == 0 and mask:
                    masks.append(mask)
        offset += size
    unique = tuple(sorted(set(masks)))
    if not unique or any(
        left & right
        for index, left in enumerate(unique)
        for right in unique[index + 1 :]
    ):
        _fail("framework_poll_topology_invalid")
    return unique


def validate_disjoint_physical_affinity(
    *,
    observer_mask: int,
    game_mask: int,
    core_masks: Sequence[int],
) -> dict[str, Any]:
    if (
        isinstance(observer_mask, bool)
        or not isinstance(observer_mask, int)
        or observer_mask <= 0
        or observer_mask & (observer_mask - 1)
        or isinstance(game_mask, bool)
        or not isinstance(game_mask, int)
        or game_mask <= 0
        or game_mask & (game_mask - 1)
        or not core_masks
    ):
        _fail("framework_poll_affinity_topology_invalid")
    observer_cores = [mask for mask in core_masks if mask & observer_mask]
    game_cores = [mask for mask in core_masks if mask & game_mask]
    if len(observer_cores) != 1 or len(game_cores) != 1:
        _fail("framework_poll_affinity_topology_invalid")
    if observer_cores[0] == game_cores[0]:
        _fail("framework_poll_affinity_physical_core_overlap")
    return {
        "observer_logical_affinity_mask": observer_mask,
        "observer_physical_core_logical_mask": observer_cores[0],
        "game_logical_affinity_mask": game_mask,
        "game_physical_core_logical_mask": game_cores[0],
        "physical_cores_disjoint": True,
        "observed_physical_core_count": len(core_masks),
    }


def _percentiles(values: Sequence[int]) -> dict[str, int | None]:
    if not values:
        return {"minimum": None, "p50": None, "p95": None, "p99": None, "maximum": None}
    ordered = sorted(values)

    def percentile(numerator: int, denominator: int) -> int:
        index = math.ceil((len(ordered) - 1) * numerator / denominator)
        return ordered[index]

    return {
        "minimum": ordered[0],
        "p50": percentile(50, 100),
        "p95": percentile(95, 100),
        "p99": percentile(99, 100),
        "maximum": ordered[-1],
    }


def _transition_row(value: PollTransition) -> dict[str, Any]:
    return {
        "transition_index": value.transition_index,
        "poll_index": value.poll_index,
        "previous_sample_after_perf_counter_ns": (
            value.previous_sample_after_perf_counter_ns
        ),
        "sample_before_perf_counter_ns": value.sample_before_perf_counter_ns,
        "sample_after_perf_counter_ns": value.sample_after_perf_counter_ns,
        "changed_fields": list(value.changed_fields),
        "state": asdict(value.state),
    }


def _draw_event_row(value: DrawEvent) -> dict[str, int]:
    return asdict(value)


def _capture_binding(
    *,
    capture_metadata_path: Path,
    update_map_path: Path,
    state_sidecar_path: Path,
    identity: Mapping[str, Any],
    poll: PollResult,
) -> dict[str, Any]:
    metadata = _read_json(
        capture_metadata_path, "framework_poll_capture_metadata_invalid"
    )
    update_map = _read_json(
        update_map_path, "framework_poll_update_map_invalid"
    )
    state = _read_json(
        state_sidecar_path, "framework_poll_state_sidecar_invalid"
    )
    metadata_sha = _sha256_file(capture_metadata_path)
    update_sha = _sha256_file(update_map_path)
    state_sha = _sha256_file(state_sidecar_path)
    target = metadata.get("target_identity")
    if (
        metadata.get("schema") != "zuma-rl.dxgi-bgra-capture"
        or metadata.get("status") != "acquisition_complete"
        or not isinstance(target, dict)
        or update_map.get("schema") != "zuma-rl.pc-framework-update-map"
        or update_map.get("version") != 1
        or state.get("schema") != "zuma-rl.pc-framework-state-diagnostic"
        or state.get("version") != 1
        or state.get("diagnostic_only") is not True
    ):
        _fail("framework_poll_capture_binding_invalid")
    for source in (target, update_map, state):
        if (
            source.get("process_id") != identity.get("process_id")
            or source.get("process_creation_filetime_100ns")
            != identity.get("process_creation_filetime_100ns")
            or source.get("executable_sha256")
            != identity.get("executable_sha256")
        ):
            _fail("framework_poll_process_binding_mismatch")
    if (
        update_map.get("capture_metadata_sha256") != metadata_sha
        or state.get("capture_metadata_sha256") != metadata_sha
        or state.get("framework_update_map_sha256") != update_sha
    ):
        _fail("framework_poll_capture_binding_invalid")
    capture_start = metadata.get("capture_start_perf_counter_ns")
    capture_end = metadata.get("capture_end_perf_counter_ns")
    if (
        isinstance(capture_start, bool)
        or not isinstance(capture_start, int)
        or isinstance(capture_end, bool)
        or not isinstance(capture_end, int)
        or not poll.poll_start_perf_counter_ns <= capture_start
        < capture_end <= poll.poll_end_perf_counter_ns
    ):
        _fail("framework_poll_capture_interval_not_covered")
    return {
        "capture_metadata_artifact": str(capture_metadata_path.resolve()),
        "capture_metadata_sha256": metadata_sha,
        "framework_update_map_artifact": str(update_map_path.resolve()),
        "framework_update_map_sha256": update_sha,
        "framework_state_sidecar_artifact": str(state_sidecar_path.resolve()),
        "framework_state_sidecar_sha256": state_sha,
        "capture_start_perf_counter_ns": capture_start,
        "capture_end_perf_counter_ns": capture_end,
        "capture_interval_fully_observed": True,
    }


def build_report(
    *,
    identity: Mapping[str, Any],
    affinity_mask: int,
    affinity_topology: Mapping[str, Any],
    ready_marker_path: Path,
    stop_marker_path: Path,
    capture_metadata_path: Path,
    update_map_path: Path,
    state_sidecar_path: Path,
    poll: PollResult,
) -> dict[str, Any]:
    ready = _read_json(ready_marker_path, "framework_poll_ready_marker_invalid")
    stop = _read_json(stop_marker_path, "framework_poll_stop_marker_invalid")
    if (
        ready.get("schema") != READY_SCHEMA
        or ready.get("version") != 1
        or ready.get("process_id") != identity.get("process_id")
        or ready.get("process_creation_filetime_100ns")
        != identity.get("process_creation_filetime_100ns")
        or ready.get("executable_sha256") != identity.get("executable_sha256")
        or stop.get("schema") != STOP_SCHEMA
        or stop.get("version") != 1
        or stop.get("process_id") != identity.get("process_id")
    ):
        _fail("framework_poll_control_marker_invalid")
    binding = _capture_binding(
        capture_metadata_path=capture_metadata_path,
        update_map_path=update_map_path,
        state_sidecar_path=state_sidecar_path,
        identity=identity,
        poll=poll,
    )
    skipped_draw_increments = sum(
        max(0, event.draw_count_delta - 1) for event in poll.draw_events
    )
    duration_ns = (
        poll.poll_end_perf_counter_ns - poll.poll_start_perf_counter_ns
    )
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "classification": "diagnostic-only-not-pc-golden",
        "gate_effect": "none",
        "status": (
            "complete"
            if skipped_draw_increments == 0
            else "complete_with_observed_draw_counter_gaps"
        ),
        "process_identity": dict(identity),
        "observer": {
            "process_affinity_mask": affinity_mask,
            "affinity_topology": dict(affinity_topology),
            "arm_at_framework_update": poll.arm_at_framework_update,
            "arm_no_later_than_framework_update": (
                poll.arm_no_later_than_framework_update
            ),
            "armed_at_framework_update": poll.armed_at_framework_update,
            "target_poll_hz": poll.target_poll_hz,
            "prearm_sample_count": poll.prearm_sample_count,
            "poll_start_perf_counter_ns": poll.poll_start_perf_counter_ns,
            "poll_end_perf_counter_ns": poll.poll_end_perf_counter_ns,
            "duration_ns": duration_ns,
            "poll_count": poll.poll_count,
            "mean_polls_per_second": (
                poll.poll_count * 1_000_000_000 / duration_ns
                if duration_ns > 0
                else None
            ),
            "sample_latency_ns": _percentiles(poll.sample_latency_ns),
            "sample_completion_interval_ns": _percentiles(
                poll.sample_completion_interval_ns
            ),
            "tracked_fields": list(_TRACKED_FIELDS),
        },
        "control": {
            "ready_marker_sha256": _sha256_file(ready_marker_path),
            "stop_marker_sha256": _sha256_file(stop_marker_path),
            "stop_requested_perf_counter_ns": stop.get(
                "requested_perf_counter_ns"
            ),
        },
        "capture_binding": binding,
        "ready_sample": {
            "sample_before_perf_counter_ns": (
                poll.ready_sample_before_perf_counter_ns
            ),
            "sample_after_perf_counter_ns": (
                poll.ready_sample_after_perf_counter_ns
            ),
            "state": asdict(poll.ready_state),
        },
        "initial_sample": {
            "sample_before_perf_counter_ns": (
                poll.initial_sample_before_perf_counter_ns
            ),
            "sample_after_perf_counter_ns": (
                poll.initial_sample_after_perf_counter_ns
            ),
            "state": asdict(poll.initial_state),
        },
        "final_state": asdict(poll.final_state),
        "transition_count": len(poll.transitions),
        "transitions": [_transition_row(row) for row in poll.transitions],
        "draw_event_count": len(poll.draw_events),
        "skipped_draw_increment_count": skipped_draw_increments,
        "draw_events": [_draw_event_row(row) for row in poll.draw_events],
    }


def _positive_int(value: str) -> int:
    try:
        parsed = int(value, 0)
    except ValueError:
        raise argparse.ArgumentTypeError("invalid integer") from None
    if parsed <= 0:
        raise argparse.ArgumentTypeError("invalid integer")
    return parsed


def _duration(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("invalid duration") from None
    if not math.isfinite(parsed) or not 0.25 <= parsed <= 120.0:
        raise argparse.ArgumentTypeError("invalid duration")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", required=True, type=_positive_int)
    parser.add_argument("--process-name", required=True)
    parser.add_argument("--expected-executable-sha256", required=True)
    parser.add_argument("--affinity-mask", type=_positive_int, default=4)
    parser.add_argument("--game-process-affinity-mask", required=True, type=_positive_int)
    parser.add_argument("--arm-at-framework-update", type=_positive_int)
    parser.add_argument(
        "--arm-no-later-than-framework-update",
        type=_positive_int,
    )
    parser.add_argument(
        "--target-poll-hz", type=_positive_int, default=5_000
    )
    parser.add_argument(
        "--maximum-duration-seconds",
        type=_duration,
        default=DEFAULT_MAX_DURATION_SECONDS,
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ready-marker", required=True, type=Path)
    parser.add_argument("--stop-marker", required=True, type=Path)
    parser.add_argument("--capture-metadata", required=True, type=Path)
    parser.add_argument("--framework-update-map", required=True, type=Path)
    parser.add_argument("--framework-state-sidecar", required=True, type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        expected_sha = args.expected_executable_sha256
        if (
            not isinstance(expected_sha, str)
            or _SHA256_RE.fullmatch(expected_sha) is None
        ):
            _fail("framework_poll_expected_identity_invalid")
        output = args.output.resolve()
        ready_marker = args.ready_marker.resolve()
        stop_marker = args.stop_marker.resolve()
        capture_metadata = args.capture_metadata.resolve()
        update_map = args.framework_update_map.resolve()
        state_sidecar = args.framework_state_sidecar.resolve()
        _prepare_paths(
            output=output,
            ready=ready_marker,
            stop=stop_marker,
            capture_metadata=capture_metadata,
            framework_update_map=update_map,
            framework_state_sidecar=state_sidecar,
        )
        identity = windows_process_identity(args.pid, args.process_name)
        if identity["executable_sha256"] != expected_sha:
            _fail("framework_poll_expected_identity_mismatch")
        affinity_topology = validate_disjoint_physical_affinity(
            observer_mask=args.affinity_mask,
            game_mask=args.game_process_affinity_mask,
            core_masks=windows_processor_core_masks(),
        )
        set_current_process_affinity(args.affinity_mask)

        def on_ready(
            sample_before_ns: int,
            sample_after_ns: int,
            state: FrameworkStateSnapshot,
        ) -> None:
            payload = _canonical_bytes(
                {
                    "schema": READY_SCHEMA,
                    "version": 1,
                    "process_id": identity["process_id"],
                    "process_creation_filetime_100ns": identity[
                        "process_creation_filetime_100ns"
                    ],
                    "executable_sha256": identity["executable_sha256"],
                    "sample_before_perf_counter_ns": sample_before_ns,
                    "sample_after_perf_counter_ns": sample_after_ns,
                    "initial_draw_count": state.draw_count,
                    "initial_update_count": state.update_count,
                }
            )
            _publish_output(ready_marker, payload)

        with FrameworkUpdateReader(args.pid) as reader:
            poll = poll_framework_state(
                reader,
                stop_requested=stop_marker.is_file,
                on_ready=on_ready,
                max_duration_seconds=args.maximum_duration_seconds,
                arm_at_framework_update=args.arm_at_framework_update,
                arm_no_later_than_framework_update=(
                    args.arm_no_later_than_framework_update
                ),
                target_poll_hz=args.target_poll_hz,
            )
        report = build_report(
            identity=identity,
            affinity_mask=args.affinity_mask,
            affinity_topology=affinity_topology,
            ready_marker_path=ready_marker,
            stop_marker_path=stop_marker,
            capture_metadata_path=capture_metadata,
            update_map_path=update_map,
            state_sidecar_path=state_sidecar,
            poll=poll,
        )
        digest = _publish_output(output, _canonical_bytes(report))
    except FrameworkPollError as error:
        print(f"framework poll error: {error.code}", file=sys.stderr)
        return 2
    except CaptureError as error:
        print(f"framework poll error: {error.code}", file=sys.stderr)
        return 2
    print(f"{output}\n{digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
