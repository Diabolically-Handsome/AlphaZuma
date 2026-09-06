"""Reconstruct a captured retail global MTRand state for replay restore."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import struct
from typing import Any, Mapping

from zuma_rl.revenge_core import PopCapMTRandom
from tools.popcap_thread_crt_restore import (
    load_source_bound_thread_crt_restore_state,
)


MTRAND_STATE_WORDS = 624
MTRAND_STATE_BYTES = (MTRAND_STATE_WORDS + 1) * 4
DEFAULT_MAXIMUM_RECONSTRUCTION_DRAWS = 100_000


@dataclass(frozen=True, slots=True)
class GlobalMTRandRestoreState:
    source_path: Path
    source_sha256: str
    source_process_id: int
    source_framework_update: int
    source_native_game_time: int
    seed: int
    captured_draw_count: int
    captured_index: int
    captured_state_sha256: str
    rewind_draws: int
    draw_count: int
    index: int
    state_sha256: str
    payload: bytes
    source_kind: str = "live_rng_monitor"
    source_trace_path: Path | None = None
    source_trace_sha256: str | None = None
    source_recording_report_path: Path | None = None
    source_recording_report_sha256: str | None = None
    source_dmo_path: Path | None = None
    source_dmo_sha256: str | None = None
    source_call_order: int | None = None
    source_caller: int | None = None

    def semantic_dict(self) -> dict[str, int | str]:
        return {
            "seed": self.seed,
            "captured_draw_count": self.captured_draw_count,
            "captured_index": self.captured_index,
            "captured_state_sha256": self.captured_state_sha256,
            "rewind_draws": self.rewind_draws,
            "draw_count": self.draw_count,
            "index": self.index,
            "state_sha256": self.state_sha256,
        }

    @property
    def semantic_sha256(self) -> str:
        payload = json.dumps(
            self.semantic_dict(),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        return "sha256:" + hashlib.sha256(payload).hexdigest()


def _strict_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context} must be an integer")
    return value


def _state_payload(rng: PopCapMTRandom) -> bytes:
    return struct.pack(
        f"<{MTRAND_STATE_WORDS + 1}I",
        *rng.words,
        rng.index,
    )


def _state_after_draws(seed: int, draw_count: int) -> bytes:
    rng = PopCapMTRandom(seed)
    for _ in range(draw_count):
        rng.next_u31()
    return _state_payload(rng)


def reconstruct_global_mtrand_state(
    *,
    seed: int,
    index: int,
    state_sha256: str,
    maximum_draws: int = DEFAULT_MAXIMUM_RECONSTRUCTION_DRAWS,
) -> tuple[int, bytes]:
    """Find the unique seeded draw count matching one captured state hash."""

    if not 0 <= seed <= 0xFFFFFFFF:
        raise ValueError("global MTRand seed must fit uint32")
    if not 0 <= index <= MTRAND_STATE_WORDS:
        raise ValueError("global MTRand index is invalid")
    if (
        len(state_sha256) != 64
        or state_sha256.lower() != state_sha256
        or any(value not in "0123456789abcdef" for value in state_sha256)
    ):
        raise ValueError("global MTRand state SHA-256 is invalid")
    if maximum_draws < 0:
        raise ValueError("maximum reconstruction draws must not be negative")

    rng = PopCapMTRandom(seed)
    matches: list[tuple[int, bytes]] = []
    for draw_count in range(maximum_draws + 1):
        if rng.index == index:
            payload = _state_payload(rng)
            if hashlib.sha256(payload).hexdigest() == state_sha256:
                matches.append((draw_count, payload))
        if draw_count != maximum_draws:
            rng.next_u31()
    if len(matches) != 1:
        raise ValueError(
            "captured global MTRand state must have exactly one seeded "
            "reconstruction within the declared draw bound"
        )
    return matches[0]


def load_global_mtrand_restore_state(
    path: Path,
    *,
    framework_update: int,
    seed: int,
    maximum_draws: int = DEFAULT_MAXIMUM_RECONSTRUCTION_DRAWS,
    rewind_draws: int = 0,
) -> GlobalMTRandRestoreState:
    """Load and cryptographically reconstruct one read-only monitor row."""

    if framework_update < 0:
        raise ValueError("global MTRand source update must not be negative")
    if rewind_draws < 0:
        raise ValueError("global MTRand rewind draws must not be negative")
    data = path.read_bytes()
    source_sha256 = "sha256:" + hashlib.sha256(data).hexdigest()
    try:
        lines = data.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("global MTRand monitor log is not UTF-8") from error
    if not lines:
        raise ValueError("global MTRand monitor log is empty")
    try:
        header = json.loads(lines[0])
    except json.JSONDecodeError as error:
        raise ValueError("global MTRand monitor header is invalid") from error
    if (
        not isinstance(header, Mapping)
        or header.get("schema") != "zuma-rl.live-rng-change-monitor"
        or header.get("version") != 1
        or header.get("process_memory_writes") != 0
    ):
        raise ValueError("global MTRand monitor header contract mismatch")
    process_id = _strict_int(
        header.get("process_id"),
        "global MTRand source process ID",
    )
    if process_id <= 0:
        raise ValueError("global MTRand source process ID is invalid")

    matches: list[Mapping[str, Any]] = []
    for line in lines[1:]:
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError("global MTRand monitor row is invalid") from error
        if (
            isinstance(row, Mapping)
            and row.get("type") == "rng"
            and row.get("framework_update") == framework_update
        ):
            matches.append(row)
    if len(matches) != 1:
        raise ValueError(
            "global MTRand monitor must contain exactly one row at the "
            "source update"
        )
    row = matches[0]
    native_game_time = _strict_int(
        row.get("native_game_time"),
        "global MTRand source native game time",
    )
    index = _strict_int(
        row.get("global_mtrand_index"),
        "global MTRand source index",
    )
    state_sha256 = row.get("global_mtrand_sha256")
    if native_game_time < 0 or not isinstance(state_sha256, str):
        raise ValueError("global MTRand monitor state is invalid")
    captured_draw_count, captured_payload = reconstruct_global_mtrand_state(
        seed=seed,
        index=index,
        state_sha256=state_sha256,
        maximum_draws=maximum_draws,
    )
    if rewind_draws > captured_draw_count:
        raise ValueError(
            "global MTRand rewind exceeds the captured draw count"
        )
    draw_count = captured_draw_count - rewind_draws
    payload = (
        captured_payload
        if rewind_draws == 0
        else _state_after_draws(seed, draw_count)
    )
    target_index = struct.unpack_from(
        "<I",
        payload,
        MTRAND_STATE_WORDS * 4,
    )[0]
    target_sha256 = hashlib.sha256(payload).hexdigest()
    return GlobalMTRandRestoreState(
        source_path=path.resolve(),
        source_sha256=source_sha256,
        source_process_id=process_id,
        source_framework_update=framework_update,
        source_native_game_time=native_game_time,
        seed=seed,
        captured_draw_count=captured_draw_count,
        captured_index=index,
        captured_state_sha256="sha256:" + state_sha256,
        rewind_draws=rewind_draws,
        draw_count=draw_count,
        index=target_index,
        state_sha256="sha256:" + target_sha256,
        payload=payload,
    )


def _load_json_object(
    path: Path,
    context: str,
) -> tuple[bytes, Mapping[str, Any]]:
    data = path.read_bytes()
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{context} is not valid JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a JSON object")
    return data, value


def _resolved_path(value: Any, context: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} path is invalid")
    return Path(value).resolve()


def load_source_bound_global_mtrand_restore_state(
    monitor_path: Path,
    trace_path: Path,
    recording_report_path: Path,
    *,
    monitor_framework_update: int,
    seed: int,
    rewind_draws: int,
    source_order: int,
    framework_update: int,
    caller: int,
    maximum_draws: int = DEFAULT_MAXIMUM_RECONSTRUCTION_DRAWS,
) -> GlobalMTRandRestoreState:
    """Bind a reconstructed global state to one natural retail call."""

    monitor_path = monitor_path.resolve()
    trace_path = trace_path.resolve()
    recording_report_path = recording_report_path.resolve()
    state = load_global_mtrand_restore_state(
        monitor_path,
        framework_update=monitor_framework_update,
        seed=seed,
        maximum_draws=maximum_draws,
        rewind_draws=rewind_draws,
    )
    # This validator establishes the natural-win, read-only source trace,
    # recording report, raw DMO, process, main-thread, and exact call selector.
    thread_source = load_source_bound_thread_crt_restore_state(
        trace_path,
        recording_report_path,
        source_order=source_order,
        framework_update=framework_update,
        caller=caller,
    )
    if state.source_process_id != thread_source.source_process_id:
        raise ValueError(
            "source-bound global MTRand process identity mismatch"
        )

    trace_data, trace = _load_json_object(
        trace_path,
        "source-bound global MTRand trace",
    )
    calls = trace.get("calls")
    if not isinstance(calls, list) or source_order >= len(calls):
        raise ValueError("source-bound global MTRand trace calls are invalid")
    row = calls[source_order]
    if not isinstance(row, Mapping):
        raise ValueError("source-bound global MTRand row is invalid")

    unpacked = struct.unpack(
        f"<{MTRAND_STATE_WORDS + 1}I",
        state.payload,
    )
    rng = PopCapMTRandom(1)
    rng.load_state(unpacked[:MTRAND_STATE_WORDS], unpacked[-1])
    expected_output = rng.next_u31()
    post_payload = _state_payload(rng)
    expected_post_sha256 = (
        "sha256:" + hashlib.sha256(post_payload).hexdigest()
    )
    if (
        _strict_int(row.get("pre_index"), "source-bound global pre-index")
        != state.index
        or row.get("pre_state_sha256") != state.state_sha256
        or _strict_int(row.get("output"), "source-bound global output")
        != expected_output
        or _strict_int(
            row.get("post_index"),
            "source-bound global post-index",
        )
        != rng.index
        or row.get("post_state_sha256") != expected_post_sha256
    ):
        raise ValueError(
            "source-bound global MTRand call state contract mismatch"
        )

    report_data, report = _load_json_object(
        recording_report_path,
        "source-bound global MTRand recording report",
    )
    monitor_receipt = report.get("rng_monitor")
    if (
        not isinstance(monitor_receipt, Mapping)
        or monitor_receipt.get("exit_code") != 0
        or monitor_receipt.get("process_memory_writes") != 0
        or _resolved_path(
            monitor_receipt.get("output"),
            "source-bound global MTRand monitor receipt",
        )
        != monitor_path
        or monitor_receipt.get("output_sha256") != state.source_sha256
        or _strict_int(
            monitor_receipt.get("ui_thread_id"),
            "source-bound global MTRand monitor UI thread ID",
        )
        != thread_source.source_thread_id
        or _strict_int(
            monitor_receipt.get("stop_after_native_game_time"),
            "source-bound global MTRand monitor stop time",
        )
        < state.source_native_game_time
    ):
        raise ValueError(
            "source-bound global MTRand monitor receipt mismatch"
        )

    return replace(
        state,
        source_kind="natural_retail_hardware_trace_and_monitor",
        source_trace_path=trace_path,
        source_trace_sha256=(
            "sha256:" + hashlib.sha256(trace_data).hexdigest()
        ),
        source_recording_report_path=recording_report_path,
        source_recording_report_sha256=(
            "sha256:" + hashlib.sha256(report_data).hexdigest()
        ),
        source_dmo_path=thread_source.source_dmo_path,
        source_dmo_sha256=thread_source.source_dmo_sha256,
        source_call_order=source_order,
        source_caller=caller,
    )
