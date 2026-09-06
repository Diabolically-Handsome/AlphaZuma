"""Verify retail pending-ball RNG order against compact memory trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
from typing import Any, Iterable, Mapping

if __package__ in {None, ""}:
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from zuma_rl.pc_rng_trajectory import (
    RngTrajectoryFrame,
    load_rng_trajectory,
    mtrand_outputs_between,
)
from zuma_rl.revenge_core import PopCapMTRandom, RevengeSimulator


REPORT_SCHEMA = "zuma-rl.pc-pending-rng-verification"
REPORT_VERSION = 1
CALL_TRACE_SCHEMA = "zuma-rl.pc-global-mtrand-call-trace"
EXPECTED_CALLERS = (
    (0x004B5ADF, "ambient_before"),
    (0x00458CD1, "repeat_roll_mod_100"),
    (0x004B4B78, "colour_rand_mod"),
    (0x00402111, "ball_visual_frame_mod"),
    (0x0045CEE0, "ambient_after"),
)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _canonical_json(path: Path) -> Mapping[str, Any]:
    payload = path.read_bytes()
    value = json.loads(payload.decode("ascii"))
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object")
    canonical = (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")
    if payload != canonical:
        raise ValueError(f"JSON is not canonical: {path}")
    return value


def _active_colors(
    frame: RngTrajectoryFrame,
    *,
    curve_index: int,
) -> list[int]:
    matches = [
        curve_list
        for curve_list in frame.curve_lists
        if curve_list.curve_index == curve_index
        and curve_list.container_offset == 0x5C
    ]
    if len(matches) != 1 or any(entity is None for entity in matches[0].entities):
        raise ValueError("active curve identity stream is incomplete")
    return [
        entity.color_id
        for entity in matches[0].entities
        if entity is not None
    ]


def _advance_state(
    frame: RngTrajectoryFrame,
    count: int,
) -> tuple[tuple[int, ...], int]:
    rng = PopCapMTRandom(1)
    rng.load_state(frame.mtrand_words, frame.mtrand_index)
    for _ in range(count):
        rng.next_u31()
    return rng.state


def _trace_classification(
    trace_path: Path,
    *,
    reference_outputs: tuple[int, ...],
) -> dict[str, Any]:
    trace = _canonical_json(trace_path)
    if (
        trace.get("schema") != CALL_TRACE_SCHEMA
        or trace.get("version") != 1
        or trace.get("status") != "PASS"
    ):
        raise ValueError("global RNG call trace contract is invalid")
    calls = trace.get("calls")
    if not isinstance(calls, list):
        raise ValueError("global RNG call trace has no calls")
    matching_window = None
    for update in sorted(
        {
            int(row["framework_update"])
            for row in calls
            if isinstance(row, dict)
        }
    ):
        rows = [
            row
            for row in calls
            if isinstance(row, dict)
            and row.get("framework_update") == update
        ]
        outputs = tuple(int(row["mtrand_output"]) for row in rows)
        if outputs == reference_outputs:
            matching_window = rows
            break
    if matching_window is None:
        raise ValueError("call trace does not contain the reference MT sequence")
    callers = tuple(int(row["caller"]) for row in matching_window)
    expected = tuple(address for address, _ in EXPECTED_CALLERS)
    if callers != expected:
        raise ValueError(
            "global RNG caller sequence mismatch:"
            f"observed={callers},expected={expected}"
        )
    return {
        "artifact": trace_path.name,
        "artifact_sha256": _sha256_path(trace_path),
        "framework_update": int(matching_window[0]["framework_update"]),
        "call_count": len(matching_window),
        "calls": [
            {
                "order": index,
                "caller": address,
                "caller_hex": f"0x{address:08x}",
                "role": role,
                "mtrand_output": int(row["mtrand_output"]),
                "mtrand_index_before": int(row["mtrand_index_before"]),
                "mtrand_index_after": int(row["mtrand_index_after"]),
            }
            for index, ((address, role), row) in enumerate(
                zip(EXPECTED_CALLERS, matching_window, strict=True)
            )
        ],
    }


def verify_pending_rng(
    *,
    trajectory_path: Path,
    call_trace_path: Path,
    original_root: Path,
    level_id: str,
    hard: bool,
    curve_index: int,
) -> dict[str, Any]:
    frames = load_rng_trajectory(trajectory_path)
    cases = [
        (before, after)
        for before, after in zip(frames, frames[1:])
        if before.pending_ball_count == 0
        and after.pending_ball_count == 1
        and len(after.pending_colors) == 1
    ]
    if len(cases) < 2:
        raise ValueError("at least two pending-generation transitions are required")

    decoded_cases: list[dict[str, Any]] = []
    for before, after in cases:
        outputs = mtrand_outputs_between(before, after)
        if len(outputs) != 5:
            raise ValueError(
                "pending generation did not consume the proven five-call window"
            )
        simulator = RevengeSimulator.from_installed(
            level_id,
            root=original_root,
            hard=hard,
            curve_index=curve_index,
            seed=1,
        )
        active_colors = _active_colors(
            before,
            curve_index=curve_index,
        )
        simulator.load_state(
            active_colors,
            [float(index) for index in range(len(active_colors))],
            pending_colors=[],
        )

        state_after_leading_ambient = _advance_state(before, 1)
        simulator.load_mtrand_state(
            words=state_after_leading_ambient[0],
            index=state_after_leading_ambient[1],
        )
        predicted_color = simulator._append_pending_color()
        expected_pre_trailing_state = _advance_state(before, 4)
        expected_color = after.pending_colors[0]
        status = (
            "PASS"
            if predicted_color == expected_color
            and simulator.rng.state == expected_pre_trailing_state
            else "FAIL"
        )
        decoded_cases.append(
            {
                "status": status,
                "from_update": before.update,
                "to_update": after.update,
                "active_ball_count": len(active_colors),
                "previous_color": active_colors[0],
                "expected_color": expected_color,
                "predicted_color": predicted_color,
                "mtrand_outputs": list(outputs),
                "ambient_before_output": outputs[0],
                "repeat_roll_output": outputs[1],
                "repeat_roll_mod_100": outputs[1] % 100,
                "colour_candidate_output": outputs[2],
                "colour_candidate_mod": outputs[2] % simulator.num_colors,
                "ball_visual_frame_output": outputs[3],
                "ambient_after_output": outputs[4],
                "simulator_state_matches_before_trailing_ambient": (
                    simulator.rng.state == expected_pre_trailing_state
                ),
            }
        )

    trace_classification = _trace_classification(
        call_trace_path,
        reference_outputs=tuple(decoded_cases[0]["mtrand_outputs"]),
    )
    status = (
        "PASS"
        if all(case["status"] == "PASS" for case in decoded_cases)
        else "FAIL"
    )
    parameters = RevengeSimulator.from_installed(
        level_id,
        root=original_root,
        hard=hard,
        curve_index=curve_index,
        seed=1,
    ).parameters
    return {
        "schema": REPORT_SCHEMA,
        "version": REPORT_VERSION,
        "status": status,
        "trajectory": {
            "artifact": str(trajectory_path),
            "artifact_sha256": _sha256_path(trajectory_path),
            "start_update": frames[0].update,
            "end_update": frames[-1].update,
            "tick_count": len(frames),
        },
        "call_trace": trace_classification,
        "scenario": {
            "level_id": level_id,
            "hard": hard,
            "curve_index": curve_index,
            "colors": int(parameters.colors),
            "ball_repeat_chance": int(parameters.ball_repeat_chance),
            "max_single": int(parameters.max_single),
            "max_clump_size": int(parameters.max_clump_size),
        },
        "proven_call_order": [
            {"caller": address, "caller_hex": f"0x{address:08x}", "role": role}
            for address, role in EXPECTED_CALLERS
        ],
        "case_count": len(decoded_cases),
        "cases": decoded_cases,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--call-trace", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--level", default="Jungle2")
    parser.add_argument("--hard", action="store_true")
    parser.add_argument("--curve-index", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output.resolve()
    if output.exists() or not output.parent.is_dir():
        raise FileExistsError("output path must be new and have an existing parent")
    report = verify_pending_rng(
        trajectory_path=args.trajectory.resolve(),
        call_trace_path=args.call_trace.resolve(),
        original_root=args.root.resolve(),
        level_id=args.level,
        hard=args.hard,
        curve_index=args.curve_index,
    )
    output.write_bytes(
        (
            json.dumps(
                report,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("ascii")
    )
    print(
        f"status={report['status']} cases={report['case_count']} "
        f"call_trace_calls={report['call_trace']['call_count']}"
    )
    print(output)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
