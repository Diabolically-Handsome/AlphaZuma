"""Verify one already-running retail DMO through a natural outcome and exit."""

from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import sys
import time
from typing import Any, Iterable, Mapping

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tools.autoplay_live_zuma import (
    LiveBoardUnavailable,
    read_live_board,
)
from tools.align_popcap_dmo_startup import (
    _command_stream_offset,
    _load_popcap_dmo_module,
    _scan_commands,
)
from tools.inspect_popcap_replay import (
    close_process,
    open_process_readonly,
    read_process_bytes,
)
from tools.launch_popcap_replay import (
    process_image_path,
    same_windows_path,
)
from tools.monitor_popcap_replay import (
    STATE_END_OFFSET,
    STATE_START_OFFSET,
    _decode_state,
    _read_base,
)
from tools.snapshot_dxgi_window import snapshot
from zuma_rl.popcap_dmo import PopCapDemo


SCHEMA = "zuma-rl.existing-popcap-full-replay-monitor"
VERSION = 3
STILL_ACTIVE = 259
_STAGE_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")

kernel32 = (
    ctypes.WinDLL("kernel32", use_last_error=True)
    if os.name == "nt"
    else None
)
if kernel32 is not None:
    kernel32.GetExitCodeProcess.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_ulong),
    ]
    kernel32.GetExitCodeProcess.restype = ctypes.c_int


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _strict_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context} is not an integer")
    return value


def _load_strict_trace_receipt(
    path: Path,
    dmo_path: Path,
) -> dict[str, Any]:
    """Load the exact command-order offset proved by the strict tracer."""

    path = path.resolve()
    dmo_path = dmo_path.resolve()
    data = path.read_bytes()
    try:
        report = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("strict trace result is not valid JSON") from error
    if not isinstance(report, Mapping):
        raise ValueError("strict trace result is not an object")
    source_dmo = report.get("source_dmo")
    result = report.get("result")
    if not isinstance(source_dmo, Mapping) or not isinstance(result, Mapping):
        raise ValueError("strict trace result sections are missing")
    source_path = source_dmo.get("path")
    source_sha256 = source_dmo.get("sha256")
    observed_dmo_sha256 = _sha256_path(dmo_path)
    if (
        report.get("schema") != "zuma.popcap_strict_replay.v3"
        or not isinstance(source_path, str)
        or Path(source_path).resolve() != dmo_path
        or source_sha256
        not in (
            observed_dmo_sha256,
            observed_dmo_sha256.removeprefix("sha256:"),
        )
        or _strict_int(result.get("failure_count"), "failure count") != 0
        or result.get("stopped_at_update") is not True
    ):
        raise ValueError("strict trace result contract mismatch")
    command_order_offset = _strict_int(
        result.get("command_order_offset"),
        "command-order offset",
    )
    rebase_rows = result.get("command_order_rebase_rows")
    candidate_rows = result.get(
        "blackout_file_write_rebase_candidate_rows", []
    )
    native_timeline_offset = _strict_int(
        result.get("blackout_native_timeline_offset", 0),
        "blackout native-timeline offset",
    )
    if (
        command_order_offset < 0
        or not isinstance(rebase_rows, list)
        or any(
            isinstance(row, bool) or not isinstance(row, int) or row < 0
            for row in rebase_rows
        )
        or rebase_rows != sorted(set(rebase_rows))
        or not isinstance(candidate_rows, list)
        or any(
            isinstance(row, bool) or not isinstance(row, int) or row < 0
            for row in candidate_rows
        )
        or candidate_rows != sorted(set(candidate_rows))
        or native_timeline_offset < 0
    ):
        raise ValueError("strict trace command-order receipt is invalid")
    envelope_mode = bool(candidate_rows)
    envelope_set_mode = False
    allowed_offset_pairs: list[dict[str, int]] = []
    if envelope_mode:
        options = report.get("options")
        raw_allowed_offset_pairs = (
            options.get("blackout_allowed_offset_pairs", [])
            if isinstance(options, Mapping)
            else None
        )
        allowed_pairs_valid = (
            isinstance(raw_allowed_offset_pairs, list)
            and all(
                isinstance(pair, Mapping)
                and set(pair)
                == {
                    "command_order_offset",
                    "native_timeline_offset",
                }
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
                for pair in raw_allowed_offset_pairs
            )
        )
        if not allowed_pairs_valid:
            raise ValueError(
                "strict trace command-order receipt is invalid"
            )
        allowed_offset_pairs = [
            {
                "command_order_offset": pair["command_order_offset"],
                "native_timeline_offset": pair[
                    "native_timeline_offset"
                ],
            }
            for pair in raw_allowed_offset_pairs
        ]
        allowed_pair_values = tuple(
            (
                pair["command_order_offset"],
                pair["native_timeline_offset"],
            )
            for pair in allowed_offset_pairs
        )
        envelope_set_mode = bool(allowed_pair_values)
        common_invalid = (
            rebase_rows != []
            or len(candidate_rows) < command_order_offset
            or not isinstance(options, Mapping)
            or options.get("allow_blackout_file_write_order_rebase")
            is not True
        )
        if envelope_set_mode:
            invalid = (
                common_invalid
                or options.get("blackout_expected_command_order_offset")
                is not None
                or options.get(
                    "blackout_expected_native_timeline_offset"
                )
                is not None
                or len(allowed_pair_values) < 2
                or tuple(sorted(set(allowed_pair_values)))
                != allowed_pair_values
                or not any(
                    selected_command_offset > 0
                    for selected_command_offset, _ in allowed_pair_values
                )
                or (command_order_offset, native_timeline_offset)
                not in allowed_pair_values
            )
        else:
            invalid = (
                common_invalid
                or command_order_offset <= 0
                or options.get(
                    "blackout_expected_command_order_offset"
                )
                != command_order_offset
                or options.get(
                    "blackout_expected_native_timeline_offset"
                )
                != native_timeline_offset
            )
        if invalid:
            raise ValueError(
                "strict trace command-order receipt is invalid"
            )
    elif (
        native_timeline_offset != 0
        or len(rebase_rows) != command_order_offset
    ):
        raise ValueError("strict trace command-order receipt is invalid")
    return {
        "path": str(path),
        "sha256": "sha256:" + hashlib.sha256(data).hexdigest(),
        "source_dmo_path": str(dmo_path),
        "source_dmo_sha256": observed_dmo_sha256,
        "command_order_offset": command_order_offset,
        "command_order_rebase_rows": rebase_rows,
        "blackout_file_write_rebase_candidate_rows": candidate_rows,
        "blackout_native_timeline_offset": native_timeline_offset,
        "blackout_file_write_rebase_mode": (
            "candidate_envelope_set"
            if envelope_set_mode
            else (
                "candidate_envelope"
                if envelope_mode
                else "exact_all_rows"
            )
        ),
        "blackout_allowed_offset_pairs": allowed_offset_pairs,
        "blackout_selected_offset_pair": (
            {
                "command_order_offset": command_order_offset,
                "native_timeline_offset": native_timeline_offset,
            }
            if envelope_set_mode
            else None
        ),
        "failure_count": 0,
        "stopped_at_update": True,
    }


def _write_canonical(path: Path, value: Mapping[str, Any]) -> None:
    path.write_bytes(
        (
            json.dumps(
                value,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("ascii")
    )


def _process_exit_code(handle: int) -> int:
    if kernel32 is None:
        raise RuntimeError("process exit monitoring requires Windows")
    value = ctypes.c_ulong()
    if not kernel32.GetExitCodeProcess(
        ctypes.c_void_p(handle),
        ctypes.byref(value),
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    return int(value.value)


def _snapshot_spec(text: str) -> tuple[int, str]:
    try:
        update_text, stage = text.split(":", 1)
        update = int(update_text, 0)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "snapshot must be UPDATE:STAGE"
        ) from error
    if update < 0 or not _STAGE_RE.fullmatch(stage):
        raise argparse.ArgumentTypeError(
            "snapshot update or stage is invalid"
        )
    return update, stage


def _offline_rows(dmo_path: Path) -> list[dict[str, object]]:
    data = dmo_path.read_bytes()
    offset = _command_stream_offset(data)
    length_updates = struct.unpack_from("<I", data, offset - 4)[0]
    module = _load_popcap_dmo_module()
    rows, _ = _scan_commands(
        module,
        data[offset:],
        length_updates,
    )
    return rows


def _compact_board(board: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "native_game_time",
        "score",
        "displayed_score",
        "score_target",
        "active_ball_count",
        "pending_ball_count",
        "inserting_ball_count",
        "fired_ball_count",
        "runtime_active",
        "curve_plan_exhausted",
        "loss_counter",
    )
    return {key: board.get(key) for key in keys}


def _is_natural_win(
    board: Mapping[str, Any],
    *,
    minimum_score: int,
) -> bool:
    score = board.get("score")
    target = board.get("score_target")
    if (
        isinstance(score, bool)
        or not isinstance(score, int)
        or isinstance(target, bool)
        or not isinstance(target, int)
    ):
        return False
    return (
        score >= minimum_score
        and score >= target
        and int(board.get("active_ball_count", -1)) == 0
        and int(board.get("pending_ball_count", -1)) == 0
        and int(board.get("inserting_ball_count", -1)) == 0
        and int(board.get("fired_ball_count", -1)) == 0
        and board.get("runtime_active") is False
    )


def _is_natural_loss(board: Mapping[str, Any]) -> bool:
    """Return whether the live retail Board has entered its loss sequence."""

    loss_counter = board.get("loss_counter")
    return (
        isinstance(loss_counter, int)
        and not isinstance(loss_counter, bool)
        and loss_counter > 0
    )


def _terminal_board_fallback_is_armed(
    *,
    max_score: int | None,
    minimum_win_score: int,
) -> bool:
    """Avoid perturbing active gameplay with terminal-only memory reads."""

    return (
        max_score is not None
        and max_score >= minimum_win_score
    )


def _evaluation_failures(
    *,
    exit_code: int | None,
    timed_out: bool,
    max_update: int,
    expected_update: int,
    max_command_order: int,
    expected_command_order: int,
    max_bit_position: int,
    expected_bit_position: int,
    natural_win: Mapping[str, Any] | None,
    natural_loss: Mapping[str, Any] | None = None,
    expected_outcome: str = "natural_win",
    snapshots: Iterable[Mapping[str, Any]],
    expected_snapshot_count: int,
    command_order_offset: int = 0,
    natural_win_required: bool = True,
) -> list[str]:
    if expected_outcome not in {"natural_loss", "natural_win"}:
        raise ValueError("expected outcome is invalid")
    failures: list[str] = []
    if timed_out:
        failures.append("runtime_exit_timeout")
    if exit_code != 0:
        failures.append("runtime_exit_code")
    if max_update < expected_update:
        failures.append("demo_length_not_reached")
    if (
        max_command_order + command_order_offset < expected_command_order
        and max_bit_position < expected_bit_position
    ):
        failures.append("final_command_not_observed")
    if max_bit_position < expected_bit_position:
        failures.append("final_bit_position_not_observed")
    natural_outcome = (
        natural_win
        if expected_outcome == "natural_win"
        else natural_loss
    )
    if natural_win_required and natural_outcome is None:
        failures.append(f"{expected_outcome}_not_observed")
    captured = list(snapshots)
    if len(captured) != expected_snapshot_count:
        failures.append("ui_snapshot_count")
    if any(item.get("error") is not None for item in captured):
        failures.append("ui_snapshot_failure")
    return failures


def monitor_existing_replay(
    *,
    pid: int,
    executable: Path,
    dmo_path: Path,
    output_path: Path,
    snapshot_root: Path,
    snapshot_specs: tuple[tuple[int, str], ...],
    minimum_win_score: int,
    timeout: float,
    sample_interval: float,
    strict_trace_result_path: Path | None = None,
    defer_natural_win_to_source_bound_parity: bool = False,
    expected_outcome: str = "natural_win",
) -> Path:
    """Observe DMO progress, its natural outcome, UI, and process exit."""

    if (
        pid <= 0
        or minimum_win_score < 0
        or timeout <= 0
        or sample_interval <= 0
        or expected_outcome not in {"natural_loss", "natural_win"}
    ):
        raise ValueError("full replay monitor arguments are invalid")
    if output_path.exists() or not output_path.parent.is_dir():
        raise FileExistsError(
            f"full replay output path is not new: {output_path}"
        )
    if snapshot_root.exists() or not snapshot_root.parent.is_dir():
        raise FileExistsError(
            f"snapshot root is not new: {snapshot_root}"
        )
    if len({stage for _, stage in snapshot_specs}) != len(
        snapshot_specs
    ):
        raise ValueError("snapshot stage names must be unique")
    if tuple(sorted(snapshot_specs)) != snapshot_specs:
        raise ValueError("snapshot specifications must be update-sorted")
    if defer_natural_win_to_source_bound_parity and (
        strict_trace_result_path is None
        or (
            expected_outcome == "natural_win"
            and minimum_win_score <= 0
        )
    ):
        raise ValueError(
            "deferred natural-outcome evidence requires a strict trace "
            "receipt and wins require a positive score floor"
        )

    executable = executable.resolve()
    dmo_path = dmo_path.resolve()
    observed = process_image_path(pid)
    if observed is None or not same_windows_path(observed, executable):
        raise RuntimeError(
            f"PID {pid} path mismatch: observed={observed}, "
            f"expected={executable}"
        )
    demo = PopCapDemo.read(dmo_path)
    offline_rows = _offline_rows(dmo_path)
    if len(offline_rows) != len(demo.commands):
        raise RuntimeError("offline DMO command count mismatch")
    expected_command_order = len(offline_rows) - 1
    expected_bit_position = int(offline_rows[-1]["end"])

    snapshot_root.mkdir()
    handle = open_process_readonly(pid)
    started_utc = datetime.now(timezone.utc).isoformat()
    started_ns = time.perf_counter_ns()
    base = 0
    timed_out = False
    exit_code: int | None = None
    read_error: str | None = None
    sample_count = 0
    board_sample_count = 0
    board_unavailable_count = 0
    terminal_board_fallback_count = 0
    terminal_board_fallback_suppressed_count = 0
    max_update = -1
    max_command_order = -1
    max_bit_position = -1
    max_score: int | None = None
    last_state: dict[str, Any] | None = None
    natural_win: dict[str, Any] | None = None
    natural_loss: dict[str, Any] | None = None
    samples: list[dict[str, Any]] = []
    captured: list[dict[str, Any]] = []
    next_sample_update = 0
    next_snapshot = 0
    next_board_probe = 0.0

    try:
        base = _read_base(handle, min(timeout, 30.0))
        deadline = time.monotonic() + timeout
        while True:
            now = time.monotonic()
            current_exit = _process_exit_code(handle)
            if current_exit != STILL_ACTIVE:
                exit_code = current_exit
                break
            if now >= deadline:
                timed_out = True
                break

            try:
                raw = read_process_bytes(
                    handle,
                    base + STATE_START_OFFSET,
                    STATE_END_OFFSET - STATE_START_OFFSET,
                )
                state = _decode_state(raw)
            except (OSError, RuntimeError, ValueError) as error:
                read_error = f"{type(error).__name__}: {error}"
                time.sleep(sample_interval)
                continue

            sample_count += 1
            update = int(state["update"])
            order = int(state["command_order"])
            bit_position = int(state["buffer_read_bit_position"])
            max_update = max(max_update, update)
            max_command_order = max(max_command_order, order)
            max_bit_position = max(max_bit_position, bit_position)
            last_state = dict(state)
            if update >= next_sample_update:
                samples.append(
                    {
                        "elapsed_seconds": round(
                            now - (deadline - timeout),
                            6,
                        ),
                        **state,
                    }
                )
                samples = samples[-64:]
                next_sample_update = (
                    (max(update, 0) // 250) + 1
                ) * 250

            natural_outcome_missing = (
                natural_win is None
                if expected_outcome == "natural_win"
                else natural_loss is None
            )
            if natural_outcome_missing and now >= next_board_probe:
                next_board_probe = now + max(sample_interval, 0.001)
                try:
                    board = read_live_board(pid)
                except LiveBoardUnavailable:
                    if (
                        not defer_natural_win_to_source_bound_parity
                        and _terminal_board_fallback_is_armed(
                            max_score=max_score,
                            minimum_win_score=minimum_win_score,
                        )
                    ):
                        try:
                            board = read_live_board(
                                pid,
                                require_shooter=False,
                            )
                            terminal_board_fallback_count += 1
                        except (
                            LiveBoardUnavailable,
                            OSError,
                            RuntimeError,
                            ValueError,
                        ):
                            board = None
                            board_unavailable_count += 1
                    else:
                        board = None
                        board_unavailable_count += 1
                        terminal_board_fallback_suppressed_count += 1
                except (OSError, RuntimeError, ValueError):
                    board = None
                    board_unavailable_count += 1
                if board is not None:
                    board_sample_count += 1
                    score = board.get("score")
                    if isinstance(score, int) and not isinstance(
                        score,
                        bool,
                    ):
                        max_score = (
                            score
                            if max_score is None
                            else max(max_score, score)
                        )
                    if (
                        expected_outcome == "natural_win"
                        and _is_natural_win(
                            board,
                            minimum_score=minimum_win_score,
                        )
                    ):
                        natural_win = {
                            "framework_update": update,
                            "perf_counter_ns": time.perf_counter_ns(),
                            "terminal_board_fallback": (
                                terminal_board_fallback_count > 0
                            ),
                            **_compact_board(board),
                        }
                    elif (
                        expected_outcome == "natural_loss"
                        and _is_natural_loss(board)
                    ):
                        natural_loss = {
                            "framework_update": update,
                            "perf_counter_ns": time.perf_counter_ns(),
                            "terminal_board_fallback": (
                                terminal_board_fallback_count > 0
                            ),
                            **_compact_board(board),
                        }

            while (
                next_snapshot < len(snapshot_specs)
                and update >= snapshot_specs[next_snapshot][0]
            ):
                target_update, stage = snapshot_specs[next_snapshot]
                snapshot_path = snapshot_root / f"{stage}.bmp"
                row: dict[str, Any] = {
                    "stage": stage,
                    "target_update": target_update,
                    "observed_update_before_capture": update,
                    "perf_counter_ns": time.perf_counter_ns(),
                }
                try:
                    evidence = snapshot(
                        output=snapshot_path,
                        process_name=executable.name,
                        device_index=0,
                        output_index=0,
                        timeout=2.0,
                    )
                    row.update(evidence)
                    row["sha256"] = _sha256_path(snapshot_path)
                    row["error"] = None
                except Exception as error:
                    row["error"] = (
                        f"{type(error).__name__}: {error}"
                    )
                captured.append(row)
                next_snapshot += 1

            time.sleep(sample_interval)
    finally:
        close_process(handle)

    strict_trace_receipt: dict[str, Any] | None = None
    strict_trace_receipt_error: str | None = None
    command_order_offset = 0
    if strict_trace_result_path is not None:
        try:
            strict_trace_receipt = _load_strict_trace_receipt(
                strict_trace_result_path,
                dmo_path,
            )
            command_order_offset = int(
                strict_trace_receipt["command_order_offset"]
            )
        except (OSError, RuntimeError, ValueError) as error:
            strict_trace_receipt_error = f"{type(error).__name__}: {error}"
    failures = _evaluation_failures(
        exit_code=exit_code,
        timed_out=timed_out,
        max_update=max_update,
        expected_update=demo.length_updates,
        max_command_order=max_command_order,
        expected_command_order=expected_command_order,
        max_bit_position=max_bit_position,
        expected_bit_position=expected_bit_position,
        natural_win=natural_win,
        natural_loss=natural_loss,
        expected_outcome=expected_outcome,
        snapshots=captured,
        expected_snapshot_count=len(snapshot_specs),
        command_order_offset=command_order_offset,
        natural_win_required=(
            not defer_natural_win_to_source_bound_parity
        ),
    )
    if strict_trace_result_path is not None and strict_trace_receipt is None:
        failures.append("strict_trace_receipt_invalid")
    result = {
        "schema": SCHEMA,
        "version": VERSION,
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "classification": "read-only-full-retail-replay-validation",
        "process_id": pid,
        "runtime_executable": str(executable),
        "runtime_executable_sha256": _sha256_path(executable),
        "dmo": {
            "path": str(dmo_path),
            "sha256": demo.artifact_sha256,
            "bytes": demo.artifact_bytes,
            "length_updates": demo.length_updates,
            "command_count": len(demo.commands),
            "input_command_count": len(demo.input_commands),
            "expected_final_command_order": expected_command_order,
            "expected_final_bit_position": expected_bit_position,
        },
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "started_perf_counter_ns": started_ns,
        "finished_perf_counter_ns": time.perf_counter_ns(),
        "base_address": base,
        "base_address_hex": f"0x{base:08x}",
        "timed_out": timed_out,
        "exit_observed": exit_code is not None,
        "exit_code": exit_code,
        "read_error": read_error,
        "sample_count": sample_count,
        "board_sample_count": board_sample_count,
        "board_unavailable_count": board_unavailable_count,
        "terminal_board_fallback_count": terminal_board_fallback_count,
        "terminal_board_fallback_suppressed_count": (
            terminal_board_fallback_suppressed_count
        ),
        "max_framework_update": max_update,
        "max_command_order": max_command_order,
        "strict_trace_receipt": strict_trace_receipt,
        "strict_trace_receipt_error": strict_trace_receipt_error,
        "command_order_offset": command_order_offset,
        "effective_max_command_order": (
            max_command_order + command_order_offset
        ),
        "max_buffer_read_bit_position": max_bit_position,
        "minimum_win_score": minimum_win_score,
        "expected_outcome": expected_outcome,
        "natural_outcome_evidence_deferred": (
            defer_natural_win_to_source_bound_parity
        ),
        "natural_outcome_evidence_mode": (
            "source-bound-exact-gameplay-call-parity"
            if defer_natural_win_to_source_bound_parity
            else "live-board-state"
        ),
        "natural_win_evidence_deferred": (
            defer_natural_win_to_source_bound_parity
            and expected_outcome == "natural_win"
        ),
        "natural_win_evidence_mode": (
            "source-bound-exact-gameplay-call-parity"
            if (
                defer_natural_win_to_source_bound_parity
                and expected_outcome == "natural_win"
            )
            else (
                "live-terminal-board-state"
                if expected_outcome == "natural_win"
                else None
            )
        ),
        "max_score": max_score,
        "natural_win": natural_win,
        "natural_loss": natural_loss,
        "natural_outcome": (
            natural_win
            if expected_outcome == "natural_win"
            else natural_loss
        ),
        "last_state": last_state,
        "snapshot_root": str(snapshot_root.resolve()),
        "snapshot_count": len(captured),
        "snapshots": captured,
        "samples": samples,
        "process_memory_writes": 0,
        "persistent_file_modified": False,
    }
    _write_canonical(output_path, result)
    print(
        f"full_replay_monitor status={result['status']} "
        f"outcome={expected_outcome} "
        f"exit_code={exit_code} update={max_update}/"
        f"{demo.length_updates} command={max_command_order}/"
        f"{expected_command_order} bit={max_bit_position}/"
        f"{expected_bit_position}",
        flush=True,
    )
    print(output_path.resolve(), flush=True)
    return output_path


def _positive_float(text: str) -> float:
    value = float(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _non_negative_int(text: str) -> int:
    value = int(text, 0)
    if value < 0:
        raise argparse.ArgumentTypeError(
            "value must be non-negative"
        )
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", required=True, type=int)
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--dmo", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--snapshot-root", required=True, type=Path)
    parser.add_argument(
        "--snapshot",
        action="append",
        type=_snapshot_spec,
        default=[],
        dest="snapshots",
        metavar="UPDATE:STAGE",
    )
    parser.add_argument(
        "--minimum-win-score",
        type=_non_negative_int,
        default=0,
    )
    parser.add_argument(
        "--expected-outcome",
        choices=("natural_loss", "natural_win"),
        default="natural_win",
    )
    parser.add_argument("--timeout", type=_positive_float, default=180.0)
    parser.add_argument(
        "--sample-interval",
        type=_positive_float,
        default=0.001,
    )
    parser.add_argument("--strict-trace-result", type=Path)
    parser.add_argument(
        "--defer-natural-win-to-source-bound-parity",
        action="store_true",
        help=(
            "Require a separately bound exact source gameplay-call parity "
            "receipt to establish the natural win."
        ),
    )
    parser.add_argument(
        "--defer-natural-outcome-to-source-bound-parity",
        action="store_true",
        help=(
            "Require a separately bound exact source gameplay-call parity "
            "receipt to establish the declared natural outcome."
        ),
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result_path = monitor_existing_replay(
        pid=args.pid,
        executable=args.executable,
        dmo_path=args.dmo,
        output_path=args.output,
        snapshot_root=args.snapshot_root,
        snapshot_specs=tuple(args.snapshots),
        minimum_win_score=args.minimum_win_score,
        timeout=args.timeout,
        sample_interval=args.sample_interval,
        strict_trace_result_path=args.strict_trace_result,
        defer_natural_win_to_source_bound_parity=(
            args.defer_natural_win_to_source_bound_parity
            or args.defer_natural_outcome_to_source_bound_parity
        ),
        expected_outcome=args.expected_outcome,
    )
    result = json.loads(result_path.read_text(encoding="ascii"))
    return 0 if result.get("status") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
