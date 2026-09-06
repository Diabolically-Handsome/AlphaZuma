"""Plan one read-only, memory-assisted shot in a paused retail Zuma process."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.collect_pc_memory_probe import collect_active_board


ACTIVE_CHAIN_LIST_OFFSET = 0x5C
CURRENT_BULLET_POINTER_OFFSET = 0x130
NEXT_BULLET_POINTER_OFFSET = 0x134
COLLISION_RADIUS = 36.0


def _ball(record: Mapping[str, Any]) -> Mapping[str, Any]:
    value = record.get("ball")
    if not isinstance(value, Mapping):
        raise ValueError("active-chain record lacks a decoded ball")
    return value


def _active_records(board: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    manager = board.get("curve_manager")
    if not isinstance(manager, Mapping):
        raise ValueError("curve manager is missing")
    curves = manager.get("curves")
    if (
        not isinstance(curves, Sequence)
        or isinstance(curves, (str, bytes))
        or len(curves) != 1
        or not isinstance(curves[0], Mapping)
    ):
        raise ValueError("planner requires exactly one retail curve")
    lists = curves[0].get("intrusive_lists")
    if not isinstance(lists, Sequence) or isinstance(lists, (str, bytes)):
        raise ValueError("curve list declarations are missing")
    matches = [
        row
        for row in lists
        if (
            isinstance(row, Mapping)
            and row.get("container_offset") == ACTIVE_CHAIN_LIST_OFFSET
        )
    ]
    if len(matches) != 1:
        raise ValueError("active-chain list is missing or duplicated")
    records = matches[0].get("records")
    if (
        not isinstance(records, Sequence)
        or isinstance(records, (str, bytes))
        or any(not isinstance(row, Mapping) for row in records)
    ):
        raise ValueError("active-chain records are invalid")
    if [row.get("index") for row in records] != list(range(len(records))):
        raise ValueError("active-chain indexes are not contiguous")
    return tuple(records)


def _shooter_bullet(
    board: Mapping[str, Any],
    pointer_offset: int,
) -> Mapping[str, Any]:
    primary = board.get("primary_child")
    if not isinstance(primary, Mapping):
        raise ValueError("shooter object is missing")
    bullets = primary.get("bullets")
    if not isinstance(bullets, Sequence) or isinstance(
        bullets,
        (str, bytes),
    ):
        raise ValueError("shooter bullets are missing")
    matches = [
        row
        for row in bullets
        if (
            isinstance(row, Mapping)
            and row.get("shooter_pointer_offset") == pointer_offset
        )
    ]
    if len(matches) != 1:
        raise ValueError("shooter bullet is missing or duplicated")
    return _ball(matches[0])


def _runs(
    records: Sequence[Mapping[str, Any]],
    color_id: int,
) -> tuple[tuple[int, int], ...]:
    result: list[tuple[int, int]] = []
    start: int | None = None
    for index, record in enumerate(records):
        matches = _ball(record).get("color_id") == color_id
        if matches and start is None:
            start = index
        if start is not None and (not matches or index == len(records) - 1):
            end = index if matches and index == len(records) - 1 else index - 1
            result.append((start, end))
            start = None
    return tuple(result)


def _first_hit(
    records: Sequence[Mapping[str, Any]],
    *,
    origin_x: float,
    origin_y: float,
    target_x: float,
    target_y: float,
) -> tuple[int, float] | None:
    dx = target_x - origin_x
    dy = target_y - origin_y
    length = math.hypot(dx, dy)
    if length <= 1e-6:
        return None
    ux = dx / length
    uy = dy / length
    hits: list[tuple[float, int]] = []
    for index, record in enumerate(records):
        ball = _ball(record)
        bx = float(ball["position_x"]) - origin_x
        by = float(ball["position_y"]) - origin_y
        projection = bx * ux + by * uy
        if projection <= 0.0:
            continue
        perpendicular_sq = bx * bx + by * by - projection * projection
        radius_sq = COLLISION_RADIUS * COLLISION_RADIUS
        if perpendicular_sq > radius_sq:
            continue
        entry = projection - math.sqrt(max(0.0, radius_sq - perpendicular_sq))
        hits.append((entry, index))
    if not hits:
        return None
    entry, index = min(hits)
    return index, entry


def _candidate_rows(
    records: Sequence[Mapping[str, Any]],
    *,
    color_id: int,
    origin_x: float,
    origin_y: float,
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for start, end in _runs(records, color_id):
        for target_index in range(start, end + 1):
            target = _ball(records[target_index])
            target_x = float(target["position_x"])
            target_y = float(target["position_y"])
            first = _first_hit(
                records,
                origin_x=origin_x,
                origin_y=origin_y,
                target_x=target_x,
                target_y=target_y,
            )
            if first is None:
                continue
            first_index, entry_distance = first
            if not start <= first_index <= end:
                continue
            first_ball = _ball(records[first_index])
            rows.append(
                {
                    "run_start": start,
                    "run_end": end,
                    "run_length": end - start + 1,
                    "target_index": target_index,
                    "target_ball_id": target["ball_id"],
                    "target_x": target_x,
                    "target_y": target_y,
                    "first_hit_index": first_index,
                    "first_hit_ball_id": first_ball["ball_id"],
                    "first_hit_entry_distance": entry_distance,
                    "shot_distance": math.hypot(
                        target_x - origin_x,
                        target_y - origin_y,
                    ),
                }
            )
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                -int(row["run_length"]),
                float(row["first_hit_entry_distance"]),
                float(row["shot_distance"]),
                int(row["target_index"]),
            ),
        )
    )


def plan_live_shot(
    pid: int,
    *,
    output_root: Path,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"probe output already exists: {output_root}")
    output_root.mkdir(parents=False)
    board = collect_active_board(
        pid,
        expected_score=None,
        output_root=output_root,
    )
    records = _active_records(board)
    current = _shooter_bullet(board, CURRENT_BULLET_POINTER_OFFSET)
    next_ball = _shooter_bullet(board, NEXT_BULLET_POINTER_OFFSET)
    origin_x = float(current["position_x"])
    origin_y = float(current["position_y"])
    current_candidates = _candidate_rows(
        records,
        color_id=int(current["color_id"]),
        origin_x=origin_x,
        origin_y=origin_y,
    )
    next_candidates = _candidate_rows(
        records,
        color_id=int(next_ball["color_id"]),
        origin_x=origin_x,
        origin_y=origin_y,
    )
    if current_candidates:
        recommendation = {
            "action": "fire",
            **current_candidates[0],
        }
    elif next_candidates:
        recommendation = {
            "action": "swap_then_fire",
            **next_candidates[0],
        }
    else:
        fallback: list[dict[str, Any]] = []
        for index, record in enumerate(records):
            ball = _ball(record)
            if ball.get("color_id") != current.get("color_id"):
                continue
            target_x = float(ball["position_x"])
            target_y = float(ball["position_y"])
            first = _first_hit(
                records,
                origin_x=origin_x,
                origin_y=origin_y,
                target_x=target_x,
                target_y=target_y,
            )
            if first is None or first[0] != index:
                continue
            fallback.append(
                {
                    "action": "fire_to_build_pair",
                    "target_index": index,
                    "target_ball_id": ball["ball_id"],
                    "target_x": target_x,
                    "target_y": target_y,
                    "first_hit_index": index,
                    "first_hit_ball_id": ball["ball_id"],
                    "first_hit_entry_distance": first[1],
                    "shot_distance": math.hypot(
                        target_x - origin_x,
                        target_y - origin_y,
                    ),
                }
            )
        if not fallback:
            raise RuntimeError("no unobstructed same-colour target is available")
        recommendation = min(
            fallback,
            key=lambda row: (
                float(row["first_hit_entry_distance"]),
                float(row["shot_distance"]),
            ),
        )

    manager = board["curve_manager"]
    curve = manager["curves"][0]
    active_list = next(
        row
        for row in curve["intrusive_lists"]
        if row["container_offset"] == ACTIVE_CHAIN_LIST_OFFSET
    )
    return {
        "schema": "zuma-rl.live-shot-plan",
        "version": 1,
        "status": "PASS",
        "classification": "read-only-action-planning-diagnostic",
        "process_id": pid,
        "score": board["score"],
        "displayed_score": board["displayed_score"],
        "score_target": board["score_target"],
        "curve_plan_exhausted": board["curve_plan_exhausted"]["value"],
        "active_ball_count": active_list["declared_count"],
        "fired_ball_count": board["fired_bullets"]["declared_count"],
        "current": {
            "ball_id": current["ball_id"],
            "color_id": current["color_id"],
            "origin_x": origin_x,
            "origin_y": origin_y,
        },
        "next": {
            "ball_id": next_ball["ball_id"],
            "color_id": next_ball["color_id"],
        },
        "current_match_candidates": len(current_candidates),
        "next_match_candidates": len(next_candidates),
        "recommendation": recommendation,
        "probe_artifact": str(output_root),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = plan_live_shot(
        args.pid,
        output_root=args.output_root.resolve(),
    )
    print(
        json.dumps(
            report,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
