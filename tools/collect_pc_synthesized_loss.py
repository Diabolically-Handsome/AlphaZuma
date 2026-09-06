"""Capture retail skull-entry behavior from one byte-audited state change.

The source DMO and all persistent PC state remain untouched.  At the declared
frozen framework update, this diagnostic moves only the frontmost active
chain ball's curve-distance field to a declared near-skull waypoint.  The
retail executable then owns every subsequent movement, animation, state
transition, and cleanup.  This is diagnostic calibration material, not
unmodified PC evidence.
"""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import json
import math
import os
from pathlib import Path
import struct
import sys
from typing import Any, Iterable, Mapping, Sequence

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT))
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from tools.collect_pc_memory_probe import (
    ProbeError,
    _sha256_path,
    collect_probe,
)


PROCESS_QUERY_INFORMATION = 0x0400
PROCESS_VM_OPERATION = 0x0008
PROCESS_VM_READ = 0x0010
PROCESS_VM_WRITE = 0x0020
PROCESS_ACCESS = (
    PROCESS_QUERY_INFORMATION
    | PROCESS_VM_OPERATION
    | PROCESS_VM_READ
    | PROCESS_VM_WRITE
)

ACTIVE_CHAIN_LIST_OFFSET = 0x5C
BALL_CURVE_DISTANCE_OFFSET = 0x1C


def _mapping(value: Any, code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProbeError(code)
    return value


def _sequence(value: Any, code: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(
        value,
        (str, bytes, bytearray),
    ):
        raise ProbeError(code)
    return value


def _integer(value: Any, code: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProbeError(code)
    return value


def _number(value: Any, code: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ProbeError(code)
    return float(value)


def _kernel32() -> Any:
    if os.name != "nt":
        raise OSError("process-memory synthesis requires Windows")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [
        wintypes.DWORD,
        wintypes.BOOL,
        wintypes.DWORD,
    ]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.ReadProcessMemory.argtypes = [
        wintypes.HANDLE,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    kernel32.ReadProcessMemory.restype = wintypes.BOOL
    kernel32.WriteProcessMemory.argtypes = [
        wintypes.HANDLE,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    kernel32.WriteProcessMemory.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    return kernel32


def _open_process(kernel32: Any, pid: int) -> int:
    handle = kernel32.OpenProcess(PROCESS_ACCESS, False, pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    return int(handle)


def _read_memory(
    kernel32: Any,
    handle: int,
    address: int,
    size: int,
) -> bytes:
    buffer = ctypes.create_string_buffer(size)
    transferred = ctypes.c_size_t()
    if not kernel32.ReadProcessMemory(
        handle,
        ctypes.c_void_p(address),
        buffer,
        size,
        ctypes.byref(transferred),
    ) or transferred.value != size:
        raise ctypes.WinError(ctypes.get_last_error())
    return buffer.raw


def _write_memory(
    kernel32: Any,
    handle: int,
    address: int,
    payload: bytes,
) -> None:
    buffer = ctypes.create_string_buffer(payload)
    transferred = ctypes.c_size_t()
    if not kernel32.WriteProcessMemory(
        handle,
        ctypes.c_void_p(address),
        buffer,
        len(payload),
        ctypes.byref(transferred),
    ) or transferred.value != len(payload):
        raise ctypes.WinError(ctypes.get_last_error())


def _active_records(
    board: Mapping[str, Any],
    *,
    curve_index: int,
) -> tuple[Mapping[str, Any], ...]:
    manager = _mapping(
        board.get("curve_manager"),
        "loss_synthesis_curve_manager_missing",
    )
    curves = _sequence(
        manager.get("curves"),
        "loss_synthesis_curves_missing",
    )
    curve_matches = [
        _mapping(value, "loss_synthesis_curve_invalid")
        for value in curves
        if isinstance(value, Mapping) and value.get("index") == curve_index
    ]
    if len(curve_matches) != 1:
        raise ProbeError("loss_synthesis_curve_not_unique")
    lists = _sequence(
        curve_matches[0].get("intrusive_lists"),
        "loss_synthesis_curve_lists_missing",
    )
    active_matches = [
        _mapping(value, "loss_synthesis_curve_list_invalid")
        for value in lists
        if (
            isinstance(value, Mapping)
            and value.get("container_offset") == ACTIVE_CHAIN_LIST_OFFSET
        )
    ]
    if len(active_matches) != 1:
        raise ProbeError("loss_synthesis_active_list_not_unique")
    records = tuple(
        _mapping(value, "loss_synthesis_ball_record_invalid")
        for value in _sequence(
            active_matches[0].get("records"),
            "loss_synthesis_active_records_missing",
        )
    )
    if not records:
        raise ProbeError("loss_synthesis_active_chain_empty")
    return records


def synthesize_loss_state(
    pid: int,
    board: Mapping[str, Any],
    attempt_root: Path,
    *,
    curve_index: int,
    target_ball_id: int,
    expected_curve_distance: float,
    synthesized_curve_distance: float,
    decoded_curve_end: int,
) -> Mapping[str, Any]:
    """Move only the frontmost ball to a guarded near-skull waypoint."""

    expected = float(expected_curve_distance)
    replacement = float(synthesized_curve_distance)
    if (
        target_ball_id < 0
        or curve_index < 0
        or decoded_curve_end < 1
        or not math.isfinite(expected)
        or not math.isfinite(replacement)
        or expected < 0.0
        or replacement <= expected
        or replacement > decoded_curve_end + 256.0
    ):
        raise ValueError("loss synthesis values are outside safe ranges")
    expected_bytes = struct.pack("<f", expected)
    replacement_bytes = struct.pack("<f", replacement)
    expected_f32 = struct.unpack("<f", expected_bytes)[0]
    replacement_f32 = struct.unpack("<f", replacement_bytes)[0]

    records = _active_records(board, curve_index=curve_index)
    target_matches = []
    decoded_rows: list[tuple[int, int, float, Mapping[str, Any]]] = []
    for record in records:
        ball = _mapping(
            record.get("ball"),
            "loss_synthesis_ball_summary_missing",
        )
        ball_id = _integer(
            ball.get("ball_id"),
            "loss_synthesis_ball_id_invalid",
        )
        index = _integer(
            record.get("index"),
            "loss_synthesis_ball_index_invalid",
        )
        distance = _number(
            ball.get("curve_distance"),
            "loss_synthesis_ball_distance_invalid",
        )
        decoded_rows.append((index, ball_id, distance, record))
        if ball_id == target_ball_id:
            target_matches.append((index, distance, record, ball))
    if len(target_matches) != 1:
        raise ProbeError("loss_synthesis_target_ball_not_unique")

    target_index, observed_distance, target, target_ball = target_matches[0]
    front_index, front_id, front_distance, _ = max(
        decoded_rows,
        key=lambda row: (row[2], row[0]),
    )
    if (
        target_index != front_index
        or target_ball_id != front_id
        or observed_distance != front_distance
    ):
        raise ProbeError("loss_synthesis_target_not_frontmost")
    if observed_distance != expected_f32:
        raise ProbeError("loss_synthesis_distance_precondition_failed")
    flags = target_ball.get("flags_b4_c2_hex")
    if (
        not isinstance(flags, str)
        or len(flags) != 30
        or flags[:2] != "00"
    ):
        raise ProbeError("loss_synthesis_front_contact_flag_invalid")

    target_address = _integer(
        target.get("payload_address"),
        "loss_synthesis_target_address_invalid",
    )
    address = target_address + BALL_CURVE_DISTANCE_OFFSET
    kernel32 = _kernel32()
    handle = _open_process(kernel32, pid)
    try:
        observed = _read_memory(
            kernel32,
            handle,
            address,
            len(expected_bytes),
        )
        if observed != expected_bytes:
            raise ProbeError(
                "loss_synthesis_raw_precondition_failed:"
                f"observed={observed.hex()}:expected={expected_bytes.hex()}"
            )
        _write_memory(kernel32, handle, address, replacement_bytes)
        verified = _read_memory(
            kernel32,
            handle,
            address,
            len(replacement_bytes),
        )
        if verified != replacement_bytes:
            raise ProbeError("loss_synthesis_raw_verification_failed")
    finally:
        kernel32.CloseHandle(handle)

    record = {
        "schema": "zuma-rl.pc-synthesized-loss-state",
        "version": 1,
        "classification": "diagnostic-not-unmodified-pc-evidence",
        "process_id": pid,
        "curve_index": curve_index,
        "decoded_curve_end": decoded_curve_end,
        "target_ball_id": target_ball_id,
        "target_ball_index": target_index,
        "mutation": {
            "label": "frontmost_ball_curve_distance",
            "object_address": target_address,
            "object_address_hex": f"0x{target_address:08x}",
            "field_offset": BALL_CURVE_DISTANCE_OFFSET,
            "field_offset_hex": (
                f"0x{BALL_CURVE_DISTANCE_OFFSET:04x}"
            ),
            "address": address,
            "address_hex": f"0x{address:08x}",
            "bytes": len(expected_bytes),
            "before_hex": observed.hex(),
            "after_hex": verified.hex(),
            "before_value": expected_f32,
            "after_value": replacement_f32,
        },
        "reason": (
            "move only the frontmost active ball to the decoded curve "
            "endpoint and let retail own the skull-entry transition"
        ),
        "persistent_files_modified": False,
    }
    artifact_path = attempt_root / "diagnostic-mutation.json"
    artifact_path.write_text(
        json.dumps(
            record,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="ascii",
    )
    return {
        "schema": record["schema"],
        "version": record["version"],
        "classification": record["classification"],
        "artifact": artifact_path.name,
        "artifact_sha256": _sha256_path(artifact_path),
        "mutation_count": 1,
        "target_ball_id": target_ball_id,
        "decoded_curve_end": decoded_curve_end,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--prestate", required=True, type=Path)
    parser.add_argument("--host-restore", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--probe-update", type=int, default=11190)
    parser.add_argument("--slowdown-update", type=int, default=11090)
    parser.add_argument("--trajectory-end-update", type=int, default=11280)
    parser.add_argument(
        "--trajectory-mode",
        choices=("full", "rng"),
        default="full",
    )
    parser.add_argument("--score", type=int, required=True)
    parser.add_argument("--displayed-score", type=int, required=True)
    parser.add_argument("--int32-scan-value", type=int)
    parser.add_argument("--maximum-attempts", type=int, default=3)
    parser.add_argument("--curve-index", type=int, default=0)
    parser.add_argument("--target-ball-id", required=True, type=int)
    parser.add_argument(
        "--expected-curve-distance",
        required=True,
        type=float,
    )
    parser.add_argument(
        "--synthesized-curve-distance",
        required=True,
        type=float,
    )
    parser.add_argument("--decoded-curve-end", required=True, type=int)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if (
        args.maximum_attempts < 1
        or args.slowdown_update < 0
        or args.probe_update <= args.slowdown_update
        or args.trajectory_end_update < args.probe_update
    ):
        raise SystemExit("invalid synthesis capture timing")

    def mutate(
        pid: int,
        board: Mapping[str, Any],
        attempt_root: Path,
    ) -> Mapping[str, Any]:
        return synthesize_loss_state(
            pid,
            board,
            attempt_root,
            curve_index=args.curve_index,
            target_ball_id=args.target_ball_id,
            expected_curve_distance=args.expected_curve_distance,
            synthesized_curve_distance=(
                args.synthesized_curve_distance
            ),
            decoded_curve_end=args.decoded_curve_end,
        )

    path = collect_probe(
        plan_path=args.plan.resolve(),
        prestate_path=args.prestate.resolve(),
        host_restore_path=args.host_restore.resolve(),
        output_root=args.output_root.resolve(),
        probe_update=args.probe_update,
        slowdown_update=args.slowdown_update,
        int32_value=args.score,
        displayed_score_value=args.displayed_score,
        int32_scan_value=args.int32_scan_value,
        post_mutation_displayed_score_value=args.displayed_score,
        allow_post_mutation_score_change=True,
        maximum_attempts=args.maximum_attempts,
        trajectory_end_update=args.trajectory_end_update,
        trajectory_mode=args.trajectory_mode,
        skip_repaint_guard=True,
        skip_frozen_snapshot=True,
        diagnostic_mutator=mutate,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
