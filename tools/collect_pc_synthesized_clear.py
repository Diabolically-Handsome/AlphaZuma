"""Capture retail empty-chain victory behavior from byte-audited ball flags.

The source DMO and every persistent PC file remain untouched.  At one frozen
framework update, this diagnostic marks every active and pending curve ball
for normal removal and moves only the entrance-side active ball far enough
forward for the pending ball to enter the active list.  Retail code then owns
the list transfer, removals, Board outcome transition, and subsequent lock.
This is diagnostic calibration material, not unmodified PC evidence.
"""

from __future__ import annotations

import argparse
import json
import math
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
from tools.collect_pc_synthesized_loss import (
    _integer,
    _kernel32,
    _mapping,
    _number,
    _open_process,
    _read_memory,
    _sequence,
    _write_memory,
)


ACTIVE_CHAIN_LIST_OFFSET = 0x5C
PENDING_CHAIN_LIST_OFFSET = 0x68
BALL_CURVE_DISTANCE_OFFSET = 0x1C
BALL_SHOULD_REMOVE_OFFSET = 0xC0


def _curve_records(
    board: Mapping[str, Any],
    *,
    curve_index: int,
    container_offset: int,
) -> tuple[Mapping[str, Any], ...]:
    manager = _mapping(
        board.get("curve_manager"),
        "clear_synthesis_curve_manager_missing",
    )
    curves = _sequence(
        manager.get("curves"),
        "clear_synthesis_curves_missing",
    )
    curve_matches = [
        _mapping(value, "clear_synthesis_curve_invalid")
        for value in curves
        if isinstance(value, Mapping) and value.get("index") == curve_index
    ]
    if len(curve_matches) != 1:
        raise ProbeError("clear_synthesis_curve_not_unique")
    lists = _sequence(
        curve_matches[0].get("intrusive_lists"),
        "clear_synthesis_curve_lists_missing",
    )
    list_matches = [
        _mapping(value, "clear_synthesis_curve_list_invalid")
        for value in lists
        if (
            isinstance(value, Mapping)
            and value.get("container_offset") == container_offset
        )
    ]
    if len(list_matches) != 1:
        raise ProbeError("clear_synthesis_list_not_unique")
    return tuple(
        _mapping(value, "clear_synthesis_ball_record_invalid")
        for value in _sequence(
            list_matches[0].get("records"),
            "clear_synthesis_records_missing",
        )
    )


def _decoded_record(
    record: Mapping[str, Any],
) -> tuple[int, int, int, float, str]:
    index = _integer(
        record.get("index"),
        "clear_synthesis_ball_index_invalid",
    )
    address = _integer(
        record.get("payload_address"),
        "clear_synthesis_ball_address_invalid",
    )
    ball = _mapping(
        record.get("ball"),
        "clear_synthesis_ball_summary_missing",
    )
    ball_id = _integer(
        ball.get("ball_id"),
        "clear_synthesis_ball_id_invalid",
    )
    distance = _number(
        ball.get("curve_distance"),
        "clear_synthesis_ball_distance_invalid",
    )
    flags = ball.get("flags_b4_c2_hex")
    if not isinstance(flags, str) or len(flags) != 30:
        raise ProbeError("clear_synthesis_ball_flags_invalid")
    return index, address, ball_id, distance, flags


def synthesize_clear_state(
    pid: int,
    board: Mapping[str, Any],
    attempt_root: Path,
    *,
    curve_index: int,
    expected_active_count: int,
    expected_pending_count: int,
    expected_rear_curve_distance: float,
    synthesized_rear_curve_distance: float,
) -> Mapping[str, Any]:
    """Mark all curve balls for removal after one guarded pending transfer."""

    expected_rear = float(expected_rear_curve_distance)
    replacement_rear = float(synthesized_rear_curve_distance)
    if (
        curve_index < 0
        or expected_active_count < 1
        or expected_pending_count < 1
        or not math.isfinite(expected_rear)
        or not math.isfinite(replacement_rear)
        or replacement_rear <= expected_rear
        or replacement_rear - expected_rear > 64.0
    ):
        raise ValueError("clear synthesis values are outside safe ranges")
    active = _curve_records(
        board,
        curve_index=curve_index,
        container_offset=ACTIVE_CHAIN_LIST_OFFSET,
    )
    pending = _curve_records(
        board,
        curve_index=curve_index,
        container_offset=PENDING_CHAIN_LIST_OFFSET,
    )
    if len(active) != expected_active_count:
        raise ProbeError("clear_synthesis_active_count_mismatch")
    if len(pending) != expected_pending_count:
        raise ProbeError("clear_synthesis_pending_count_mismatch")

    active_rows = tuple(_decoded_record(record) for record in active)
    pending_rows = tuple(_decoded_record(record) for record in pending)
    if tuple(row[0] for row in active_rows) != tuple(
        range(len(active_rows))
    ) or tuple(row[0] for row in pending_rows) != tuple(
        range(len(pending_rows))
    ):
        raise ProbeError("clear_synthesis_list_indexes_invalid")
    all_rows = (*active_rows, *pending_rows)
    all_ids = [row[2] for row in all_rows]
    if len(set(all_ids)) != len(all_ids):
        raise ProbeError("clear_synthesis_duplicate_ball_id")
    if any(row[4][24:26] != "00" for row in all_rows):
        raise ProbeError("clear_synthesis_remove_flag_precondition_failed")

    rear = active_rows[0]
    expected_rear_bytes = struct.pack("<f", expected_rear)
    replacement_rear_bytes = struct.pack("<f", replacement_rear)
    expected_rear_f32 = struct.unpack("<f", expected_rear_bytes)[0]
    replacement_rear_f32 = struct.unpack("<f", replacement_rear_bytes)[0]
    if rear[3] != expected_rear_f32:
        raise ProbeError("clear_synthesis_rear_distance_precondition_failed")

    mutations: list[dict[str, Any]] = [
        {
            "label": "entrance_ball_curve_distance",
            "ball_id": rear[2],
            "object_address": rear[1],
            "field_offset": BALL_CURVE_DISTANCE_OFFSET,
            "before": expected_rear_bytes,
            "after": replacement_rear_bytes,
            "before_value": expected_rear_f32,
            "after_value": replacement_rear_f32,
        }
    ]
    for list_name, rows in (("active", active_rows), ("pending", pending_rows)):
        for row in rows:
            mutations.append(
                {
                    "label": f"{list_name}_ball_should_remove",
                    "ball_id": row[2],
                    "object_address": row[1],
                    "field_offset": BALL_SHOULD_REMOVE_OFFSET,
                    "before": b"\x00",
                    "after": b"\x01",
                    "before_value": False,
                    "after_value": True,
                }
            )
    addresses = [
        mutation["object_address"] + mutation["field_offset"]
        for mutation in mutations
    ]
    if len(set(addresses)) != len(addresses):
        raise ProbeError("clear_synthesis_mutation_addresses_overlap")

    kernel32 = _kernel32()
    handle = _open_process(kernel32, pid)
    try:
        for mutation, address in zip(mutations, addresses, strict=True):
            observed = _read_memory(
                kernel32,
                handle,
                address,
                len(mutation["before"]),
            )
            if observed != mutation["before"]:
                raise ProbeError(
                    "clear_synthesis_raw_precondition_failed:"
                    f"{mutation['ball_id']}:{observed.hex()}:"
                    f"{mutation['before'].hex()}"
                )
        for mutation, address in zip(mutations, addresses, strict=True):
            _write_memory(kernel32, handle, address, mutation["after"])
        for mutation, address in zip(mutations, addresses, strict=True):
            verified = _read_memory(
                kernel32,
                handle,
                address,
                len(mutation["after"]),
            )
            if verified != mutation["after"]:
                raise ProbeError(
                    f"clear_synthesis_raw_verification_failed:"
                    f"{mutation['ball_id']}"
                )
            mutation["address"] = address
            mutation["verified"] = verified
    finally:
        kernel32.CloseHandle(handle)

    serialized_mutations = [
        {
            "label": mutation["label"],
            "ball_id": mutation["ball_id"],
            "object_address": mutation["object_address"],
            "object_address_hex": (
                f"0x{mutation['object_address']:08x}"
            ),
            "field_offset": mutation["field_offset"],
            "field_offset_hex": f"0x{mutation['field_offset']:04x}",
            "address": mutation["address"],
            "address_hex": f"0x{mutation['address']:08x}",
            "bytes": len(mutation["before"]),
            "before_hex": mutation["before"].hex(),
            "after_hex": mutation["verified"].hex(),
            "before_value": mutation["before_value"],
            "after_value": mutation["after_value"],
        }
        for mutation in mutations
    ]
    record = {
        "schema": "zuma-rl.pc-synthesized-clear-state",
        "version": 1,
        "classification": "diagnostic-not-unmodified-pc-evidence",
        "process_id": pid,
        "curve_index": curve_index,
        "active_ball_count": len(active_rows),
        "pending_ball_count": len(pending_rows),
        "mutations": serialized_mutations,
        "reason": (
            "let retail transfer its pending ball, remove every curve ball "
            "through the normal should-remove path, and own the empty-chain "
            "Board outcome transition"
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
        "active_ball_count": len(active_rows),
        "pending_ball_count": len(pending_rows),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--prestate", required=True, type=Path)
    parser.add_argument("--host-restore", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--probe-update", type=int, default=11190)
    parser.add_argument("--slowdown-update", type=int, default=11090)
    parser.add_argument("--trajectory-end-update", type=int, default=11220)
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
    parser.add_argument("--expected-active-count", type=int, required=True)
    parser.add_argument("--expected-pending-count", type=int, default=1)
    parser.add_argument(
        "--expected-rear-curve-distance",
        required=True,
        type=float,
    )
    parser.add_argument(
        "--synthesized-rear-curve-distance",
        required=True,
        type=float,
    )
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
        return synthesize_clear_state(
            pid,
            board,
            attempt_root,
            curve_index=args.curve_index,
            expected_active_count=args.expected_active_count,
            expected_pending_count=args.expected_pending_count,
            expected_rear_curve_distance=(
                args.expected_rear_curve_distance
            ),
            synthesized_rear_curve_distance=(
                args.synthesized_rear_curve_distance
            ),
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
        post_mutation_score_value=args.score,
        post_mutation_displayed_score_value=args.displayed_score,
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
