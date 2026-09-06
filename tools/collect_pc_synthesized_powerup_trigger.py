"""Capture a retail power-up trigger from a byte-audited synthesized state.

The source DMO and all persistent PC state remain untouched.  At the declared
frozen framework update, this diagnostic changes only:

* the current shooter ball's colour;
* one existing chain ball's power-up fields; and
* that colour's active-power-up manager count.

The retail executable then performs the shot, collision, insertion, match,
power-up trigger, scoring, and subsequent movement.  Every process-memory
write is guarded by an exact expected value and retained as before/after bytes.
The resulting trajectory is diagnostic calibration material, not unmodified
PC evidence.
"""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import json
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

SHOOTER_CURRENT_POINTER_OFFSET = 0x130
CURVE_ACTIVE_COLOR_COUNTS_OFFSET = 0x15C
CURVE_ACTIVE_COLOR_COUNT_SLOTS = 6
BALL_COLOR_OFFSET = 0x14
BALL_POWERUP_PREVIOUS_TICKS_OFFSET = 0xC4
BALL_POWERUP_PREVIOUS_TYPE_OFFSET = 0xC8
BALL_POWERUP_LIFETIME_OFFSET = 0xF8
BALL_POWERUP_TRANSITION_OFFSET = 0xFC
BALL_POWERUP_VISUAL_SCALE_OFFSET = 0x104
BALL_POWERUP_VISUAL_STEP_OFFSET = 0x108
BALL_POWERUP_VISUAL_INDEX_OFFSET = 0x10C
BALL_POWERUP_PRIMARY_TYPE_OFFSET = 0x11C
BALL_POWERUP_SECONDARY_TYPE_OFFSET = 0x120
POWERUP_NONE_TYPE = 14
ACTIVE_CHAIN_LIST_OFFSET = 0x5C


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


def _curve_record(
    board: Mapping[str, Any],
    *,
    curve_index: int,
) -> Mapping[str, Any]:
    manager = _mapping(
        board.get("curve_manager"),
        "synthesis_curve_manager_missing",
    )
    curves = _sequence(
        manager.get("curves"),
        "synthesis_curves_missing",
    )
    matches = [
        _mapping(value, "synthesis_curve_invalid")
        for value in curves
        if (
            isinstance(value, Mapping)
            and value.get("index") == curve_index
        )
    ]
    if len(matches) != 1:
        raise ProbeError("synthesis_curve_not_unique")
    return matches[0]


def _current_shooter_bullet(
    board: Mapping[str, Any],
) -> Mapping[str, Any]:
    shooter = _mapping(
        board.get("primary_child"),
        "synthesis_shooter_missing",
    )
    bullets = _sequence(
        shooter.get("bullets"),
        "synthesis_shooter_bullets_missing",
    )
    matches = [
        _mapping(value, "synthesis_shooter_bullet_invalid")
        for value in bullets
        if (
            isinstance(value, Mapping)
            and value.get("shooter_pointer_offset")
            == SHOOTER_CURRENT_POINTER_OFFSET
        )
    ]
    if len(matches) != 1:
        raise ProbeError("synthesis_current_shooter_ball_not_unique")
    return matches[0]


def _active_ball_record(
    curve: Mapping[str, Any],
    *,
    ball_id: int,
) -> Mapping[str, Any]:
    lists = _sequence(
        curve.get("intrusive_lists"),
        "synthesis_curve_lists_missing",
    )
    active_lists = [
        _mapping(value, "synthesis_curve_list_invalid")
        for value in lists
        if (
            isinstance(value, Mapping)
            and value.get("container_offset") == ACTIVE_CHAIN_LIST_OFFSET
        )
    ]
    if len(active_lists) != 1:
        raise ProbeError("synthesis_active_list_not_unique")
    records = _sequence(
        active_lists[0].get("records"),
        "synthesis_active_list_records_missing",
    )
    matches = []
    for value in records:
        record = _mapping(value, "synthesis_ball_record_invalid")
        ball = _mapping(
            record.get("ball"),
            "synthesis_ball_summary_missing",
        )
        if ball.get("ball_id") == ball_id:
            matches.append(record)
    if len(matches) != 1:
        raise ProbeError("synthesis_target_ball_not_unique")
    return matches[0]


def synthesize_powerup_trigger_state(
    pid: int,
    board: Mapping[str, Any],
    attempt_root: Path,
    *,
    curve_index: int,
    target_ball_id: int,
    expected_current_color: int,
    synthesized_current_color: int,
    expected_target_color: int,
    powerup_type: int,
    powerup_lifetime: int,
) -> Mapping[str, Any]:
    """Apply and audit the minimal in-process state synthesis."""

    if (
        not 0 <= expected_current_color < CURVE_ACTIVE_COLOR_COUNT_SLOTS
        or not 0
        <= synthesized_current_color
        < CURVE_ACTIVE_COLOR_COUNT_SLOTS
        or not 0 <= expected_target_color < CURVE_ACTIVE_COLOR_COUNT_SLOTS
        or not 0 <= powerup_type < POWERUP_NONE_TYPE
        or powerup_lifetime <= 0
    ):
        raise ValueError("synthesis values are outside retail ranges")

    curve = _curve_record(board, curve_index=curve_index)
    shooter_bullet = _current_shooter_bullet(board)
    target = _active_ball_record(curve, ball_id=target_ball_id)
    shooter_ball = _mapping(
        shooter_bullet.get("ball"),
        "synthesis_current_shooter_ball_summary_missing",
    )
    target_ball = _mapping(
        target.get("ball"),
        "synthesis_target_ball_summary_missing",
    )
    if shooter_ball.get("color_id") != expected_current_color:
        raise ProbeError("synthesis_current_color_precondition_failed")
    if target_ball.get("color_id") != expected_target_color:
        raise ProbeError("synthesis_target_color_precondition_failed")

    shooter_address = _integer(
        shooter_bullet.get("address"),
        "synthesis_current_shooter_address_invalid",
    )
    target_address = _integer(
        target.get("payload_address"),
        "synthesis_target_ball_address_invalid",
    )
    curve_address = _integer(
        curve.get("address"),
        "synthesis_curve_address_invalid",
    )

    fields = [
        (
            "current_shooter_color",
            shooter_address,
            BALL_COLOR_OFFSET,
            struct.pack("<i", expected_current_color),
            struct.pack("<i", synthesized_current_color),
            expected_current_color,
            synthesized_current_color,
        ),
        (
            "target_previous_ticks",
            target_address,
            BALL_POWERUP_PREVIOUS_TICKS_OFFSET,
            struct.pack("<i", 0),
            struct.pack("<i", 0),
            0,
            0,
        ),
        (
            "target_previous_type",
            target_address,
            BALL_POWERUP_PREVIOUS_TYPE_OFFSET,
            struct.pack("<i", POWERUP_NONE_TYPE),
            struct.pack("<i", POWERUP_NONE_TYPE),
            POWERUP_NONE_TYPE,
            POWERUP_NONE_TYPE,
        ),
        (
            "target_powerup_lifetime",
            target_address,
            BALL_POWERUP_LIFETIME_OFFSET,
            struct.pack("<i", 0),
            struct.pack("<i", powerup_lifetime),
            0,
            powerup_lifetime,
        ),
        (
            "target_powerup_transition",
            target_address,
            BALL_POWERUP_TRANSITION_OFFSET,
            struct.pack("<i", 0),
            struct.pack("<i", 0),
            0,
            0,
        ),
        (
            "target_powerup_visual_scale",
            target_address,
            BALL_POWERUP_VISUAL_SCALE_OFFSET,
            struct.pack("<f", 1.0),
            struct.pack("<f", 1.0),
            1.0,
            1.0,
        ),
        (
            "target_powerup_visual_step",
            target_address,
            BALL_POWERUP_VISUAL_STEP_OFFSET,
            struct.pack("<f", 0.0),
            struct.pack("<f", 0.04),
            0.0,
            0.04,
        ),
        (
            "target_powerup_visual_index",
            target_address,
            BALL_POWERUP_VISUAL_INDEX_OFFSET,
            struct.pack("<i", -1),
            struct.pack("<i", -1),
            -1,
            -1,
        ),
        (
            "target_powerup_primary_type",
            target_address,
            BALL_POWERUP_PRIMARY_TYPE_OFFSET,
            struct.pack("<i", POWERUP_NONE_TYPE),
            struct.pack("<i", powerup_type),
            POWERUP_NONE_TYPE,
            powerup_type,
        ),
        (
            "target_powerup_secondary_type",
            target_address,
            BALL_POWERUP_SECONDARY_TYPE_OFFSET,
            struct.pack("<i", POWERUP_NONE_TYPE),
            struct.pack("<i", POWERUP_NONE_TYPE),
            POWERUP_NONE_TYPE,
            POWERUP_NONE_TYPE,
        ),
        (
            "target_color_active_powerup_count",
            curve_address,
            (
                CURVE_ACTIVE_COLOR_COUNTS_OFFSET
                + expected_target_color * 4
            ),
            struct.pack("<i", 0),
            struct.pack("<i", 1),
            0,
            1,
        ),
    ]

    kernel32 = _kernel32()
    handle = _open_process(kernel32, pid)
    mutations: list[dict[str, Any]] = []
    try:
        for (
            label,
            object_address,
            offset,
            expected,
            replacement,
            expected_value,
            replacement_value,
        ) in fields:
            address = object_address + offset
            observed = _read_memory(
                kernel32,
                handle,
                address,
                len(expected),
            )
            if observed != expected:
                raise ProbeError(
                    "synthesis_field_precondition_failed:"
                    f"{label}:observed={observed.hex()}:"
                    f"expected={expected.hex()}"
                )
            _write_memory(kernel32, handle, address, replacement)
            verified = _read_memory(
                kernel32,
                handle,
                address,
                len(replacement),
            )
            if verified != replacement:
                raise ProbeError(
                    f"synthesis_field_verification_failed:{label}"
                )
            mutations.append(
                {
                    "label": label,
                    "object_address": object_address,
                    "object_address_hex": f"0x{object_address:08x}",
                    "field_offset": offset,
                    "field_offset_hex": f"0x{offset:04x}",
                    "address": address,
                    "address_hex": f"0x{address:08x}",
                    "bytes": len(expected),
                    "before_hex": observed.hex(),
                    "after_hex": verified.hex(),
                    "before_value": expected_value,
                    "after_value": replacement_value,
                }
            )
    finally:
        kernel32.CloseHandle(handle)

    record = {
        "schema": "zuma-rl.pc-synthesized-powerup-trigger-state",
        "version": 1,
        "classification": "diagnostic-not-unmodified-pc-evidence",
        "process_id": pid,
        "curve_index": curve_index,
        "target_ball_id": target_ball_id,
        "target_ball_index": _integer(
            target.get("index"),
            "synthesis_target_ball_index_invalid",
        ),
        "target_ball_color": expected_target_color,
        "powerup_type": powerup_type,
        "powerup_lifetime": powerup_lifetime,
        "current_shooter_ball_id": _integer(
            shooter_ball.get("ball_id"),
            "synthesis_current_shooter_ball_id_invalid",
        ),
        "mutations": mutations,
        "reason": (
            "force one otherwise natural retail shot to create a native "
            "colour match containing a synthesized power-up ball"
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
        "mutation_count": len(mutations),
        "target_ball_id": target_ball_id,
        "powerup_type": powerup_type,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--prestate", required=True, type=Path)
    parser.add_argument("--host-restore", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--probe-update", type=int, default=11190)
    parser.add_argument("--slowdown-update", type=int, default=11090)
    parser.add_argument("--trajectory-end-update", type=int, default=11380)
    parser.add_argument("--score", type=int, required=True)
    parser.add_argument("--displayed-score", type=int, required=True)
    parser.add_argument("--int32-scan-value", type=int)
    parser.add_argument("--maximum-attempts", type=int, default=3)
    parser.add_argument("--curve-index", type=int, default=0)
    parser.add_argument("--target-ball-id", type=int, required=True)
    parser.add_argument("--expected-current-color", type=int, required=True)
    parser.add_argument("--synthesized-current-color", type=int, required=True)
    parser.add_argument("--expected-target-color", type=int, required=True)
    parser.add_argument("--powerup-type", type=int, required=True)
    parser.add_argument("--powerup-lifetime", type=int, default=500)
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
        return synthesize_powerup_trigger_state(
            pid,
            board,
            attempt_root,
            curve_index=args.curve_index,
            target_ball_id=args.target_ball_id,
            expected_current_color=args.expected_current_color,
            synthesized_current_color=args.synthesized_current_color,
            expected_target_color=args.expected_target_color,
            powerup_type=args.powerup_type,
            powerup_lifetime=args.powerup_lifetime,
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
        maximum_attempts=args.maximum_attempts,
        trajectory_end_update=args.trajectory_end_update,
        trajectory_mode="full",
        skip_repaint_guard=True,
        skip_frozen_snapshot=True,
        diagnostic_mutator=mutate,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
