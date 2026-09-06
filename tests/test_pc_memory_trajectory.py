from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
from dataclasses import replace

import pytest

from zuma_rl import pc_memory_trajectory as trajectory_module
from zuma_rl.pc_memory_trajectory import (
    PcMemoryTrajectoryError,
    TrajectoryEntity,
    TrajectoryFrame,
    derive_trajectory_events,
    load_legacy_memory_trajectory_v1,
    load_memory_trajectory,
)


def _entity(
    ball_id: int,
    color_id: int,
    zone: str,
    *,
    kind: str = "ball",
    index: int = 0,
    distance: float = 100.0,
    x: float = 20.0,
    y: float = 30.0,
    native_address: int | None = None,
) -> TrajectoryEntity:
    return TrajectoryEntity(
        ball_id=ball_id,
        color_id=color_id,
        object_kind=kind,
        zone=zone,
        index=index,
        curve_distance=distance,
        position_x=x,
        position_y=y,
        native_object_address=native_address,
    )


def _frame(
    update: int,
    entities: tuple[TrajectoryEntity, ...],
    *,
    current: tuple[int, int] = (99, 2),
    next_ball: tuple[int, int] = (110, 3),
    staging: int = 0,
    active: int = 1,
) -> TrajectoryFrame:
    return TrajectoryFrame(
        update=update,
        score=7950,
        displayed_score=7950,
        score_target=9650,
        current_ball_id=current[0],
        current_color_id=current[1],
        next_ball_id=next_ball[0],
        next_color_id=next_ball[1],
        list_counts=((0, 0x50, staging), (0, 0x5C, active)),
        entities=entities,
    )


def test_events_recover_projectile_hit_and_object_replacement() -> None:
    active = _entity(1, 2, "curve:0:list:05c")
    current_104 = _entity(
        104,
        0,
        "shooter_current",
        kind="bullet",
    )
    next_99 = _entity(99, 2, "shooter_next", kind="bullet")
    current_99 = _entity(99, 2, "shooter_current", kind="bullet")
    next_110 = _entity(110, 3, "shooter_next", kind="bullet")
    fired_104 = _entity(
        104,
        0,
        "fired",
        kind="bullet",
        distance=0.0,
        x=330.0,
        y=150.0,
    )
    staging_104 = _entity(
        104,
        0,
        "curve:0:list:050",
        kind="bullet",
        distance=2431.0,
        x=333.5,
        y=123.1,
    )
    inserted_111 = _entity(
        111,
        0,
        "curve:0:list:05c",
        index=1,
        distance=2431.0,
        x=333.9,
        y=122.4,
    )
    frames = (
        _frame(
            10,
            (active, current_104, next_99),
            current=(104, 0),
            next_ball=(99, 2),
        ),
        _frame(
            11,
            (active, current_99, next_110, fired_104),
        ),
        _frame(
            12,
            (active, current_99, next_110, staging_104),
            staging=1,
        ),
        _frame(
            13,
            (active, inserted_111, current_99, next_110),
            active=2,
        ),
    )

    events = derive_trajectory_events(frames)

    kinds_by_update = {
        update: {
            event["kind"]
            for event in events
            if event["update"] == update
        }
        for update in (11, 12, 13)
    }
    assert "projectile_spawn" in kinds_by_update[11]
    assert "projectile_hit_staging" in kinds_by_update[12]
    assert "insertion_commit" in kinds_by_update[13]
    commit = next(
        event
        for event in events
        if event["kind"] == "insertion_commit"
    )
    assert commit["source_entity"]["ball_id"] == 104
    assert commit["inserted_entity"]["ball_id"] == 111


def test_events_disambiguate_reused_retail_ids_by_native_address() -> None:
    first = replace(
        _entity(
            178,
            3,
            "curve:0:list:05c",
            index=0,
            native_address=0x1000,
        ),
        suck_count=0,
    )
    second = replace(
        _entity(
            178,
            3,
            "curve:0:list:05c",
            index=1,
            native_address=0x2000,
        ),
        suck_count=0,
    )
    before = _frame(20, (first, second), active=2)
    after = _frame(
        21,
        (first, replace(second, suck_count=1)),
        active=2,
    )

    assert before.entities_by_id == {}
    assert len(before.entities_by_native_identity) == 2
    events = derive_trajectory_events((before, after))
    rollback = [
        event for event in events if event["kind"] == "rollback_started"
    ]
    assert len(rollback) == 1
    assert rollback[0]["ball_id"] == 178


def test_events_reject_duplicate_unresolved_native_identity() -> None:
    duplicated = _entity(178, 3, "curve:0:list:05c")
    frames = (
        _frame(20, (duplicated, duplicated), active=2),
        _frame(21, (duplicated, duplicated), active=2),
    )

    with pytest.raises(
        PcMemoryTrajectoryError,
        match="trajectory_frame_duplicate_native_identity",
    ):
        derive_trajectory_events(frames)


def _canonical(value: dict[str, object]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")


def _ball(ball_id: int, color_id: int) -> dict[str, object]:
    return {
        "ball_id": ball_id,
        "color_id": color_id,
        "curve_distance": 1.0,
        "position_x": 2.0,
        "position_y": 3.0,
    }


def _subclass() -> dict[str, object]:
    return {
        "velocity_x": 0.0,
        "velocity_y": 0.0,
        "field_158_float": 0.0,
        "field_15c_float": 0.025,
        "fired": False,
    }


def _artifact(
    root: Path,
    name: str,
    payload: bytes,
) -> dict[str, object]:
    path = root / name
    path.write_bytes(payload)
    return {
        "artifact": name,
        "artifact_bytes": len(payload),
        "artifact_sha256": (
            "sha256:" + hashlib.sha256(payload).hexdigest()
        ),
    }


def _rng_state(tick_root: Path) -> tuple[dict[str, object], dict[str, object]]:
    layouts = (
        (0x08, "weights", "float32", (0.25,) * 4 + (0.0, 0.0)),
        (
            0x18,
            "sways",
            "float32",
            (0.125, 0.21875, 0.171875, 0.0832500010728836, 0.0, 0.0),
        ),
        (0x28, "last_hit", "int32", (2, 0, 4, 3, 0, 0)),
        (0x38, "previous_hit", "int32", (0, 0, 1, 0, 0, 0)),
    )
    raw = bytearray(0x48)
    struct.pack_into("<ii", raw, 0, 4, 2)
    vectors: list[dict[str, object]] = []
    for index, (offset, name, value_type, values) in enumerate(layouts):
        begin = 0x1000 + index * 0x100
        end = begin + len(values) * 4
        struct.pack_into("<III", raw, offset + 4, begin, end, end)
        format_code = "f" if value_type == "float32" else "i"
        payload = struct.pack(f"<{len(values)}{format_code}", *values)
        vectors.append(
            {
                "name": name,
                "value_type": value_type,
                "object_offset": offset,
                "begin": begin,
                "end": end,
                "capacity": end,
                "item_count": len(values),
                "capacity_count": len(values),
                "values": list(struct.unpack(
                    f"<{len(values)}{format_code}",
                    payload,
                )),
                **_artifact(tick_root, f"qrand-{name}.bin", payload),
            }
        )
    qrand = {
        "schema": "zuma-rl.pc-qrand-state",
        "version": 1,
        "board_pointer_offset": 0x7A8,
        "update_count": 4,
        "selected_index": 2,
        "vectors": vectors,
        **_artifact(tick_root, "qrand.bin", bytes(raw)),
    }
    crt_payload = struct.pack("<I", 1_674_403_832)
    crt = {
        "schema": "zuma-rl.pc-thread-crt-rand-state",
        "version": 1,
        "state": 1_674_403_832,
        **_artifact(
            tick_root,
            "thread-crt-rand-state.bin",
            crt_payload,
        ),
    }
    return qrand, crt


def _write_one_tick_trajectory(
    tmp_path: Path,
    *,
    include_rng: bool = False,
    include_powerup_raw: bool = False,
    include_gap_raw: bool = False,
    include_win_state_raw: bool = False,
) -> Path:
    tick_root = tmp_path / "tick-00000001"
    tick_root.mkdir()
    active_board: dict[str, object] = {
        "score": 10,
        "displayed_score": 10,
        "score_target": 20,
        "primary_child": {
            "bullets": [
                {
                    "address": 100,
                    "ball": _ball(2, 1),
                    "subclass_fields": _subclass(),
                },
                {
                    "address": 200,
                    "ball": _ball(3, 2),
                    "subclass_fields": _subclass(),
                },
            ]
        },
        "fired_bullets": {
            "traversed_count": 0,
            "records": [],
        },
        "curve_manager": {
            "curves": [
                {
                    "index": 0,
                    "intrusive_lists": [
                        {
                            "container_offset": 0x50,
                            "traversed_count": 0,
                            "records": [],
                        },
                        {
                            "container_offset": 0x5C,
                            "traversed_count": 1,
                            "records": [
                                {
                                    "payload_kind": "ball",
                                    "ball": _ball(1, 0),
                                }
                            ],
                        },
                    ],
                }
            ]
        },
    }
    if include_rng:
        qrand, crt = _rng_state(tick_root)
        active_board["qrand"] = qrand
        active_board["thread_crt_rand"] = crt
    if include_powerup_raw:
        board_raw = bytearray(0x1100)
        struct.pack_into("<i", board_raw, 0xEC8, 3_759)
        active_board.update(
            _artifact(
                tick_root,
                "active-board.bin",
                bytes(board_raw),
            )
        )
        curve = active_board["curve_manager"]["curves"][0]
        assert isinstance(curve, dict)
        curve_raw = bytearray(0x200)
        struct.pack_into("<i", curve_raw, 0x78, 2_815)
        struct.pack_into("<14i", curve_raw, 0x7C, *([0] * 14))
        struct.pack_into("<i", curve_raw, 0x88, 1_660)
        cooldowns = [-1_000] * 14
        cooldowns[3] = 3_759
        struct.pack_into("<14i", curve_raw, 0xB4, *cooldowns)
        spawn_counts = [0] * 14
        spawn_counts[3] = 1
        struct.pack_into("<14i", curve_raw, 0xEC, *spawn_counts)
        curve.update(
            _artifact(
                tick_root,
                "curve-00.bin",
                bytes(curve_raw),
            )
        )
        lists = curve["intrusive_lists"]
        assert isinstance(lists, list)
        active_list = lists[1]
        assert isinstance(active_list, dict)
        records = active_list["records"]
        assert isinstance(records, list)
        record = records[0]
        assert isinstance(record, dict)
        ball = record["ball"]
        assert isinstance(ball, dict)
        ball.update(
            {
                "powerup_previous_type": 3,
                "powerup_primary_type": 3,
                "powerup_secondary_type": 14,
            }
        )
        ball_raw = bytearray(0x134)
        struct.pack_into("<I", ball_raw, 0x10, 1)
        struct.pack_into("<i", ball_raw, 0x14, 0)
        struct.pack_into("<f", ball_raw, 0x1C, 1.0)
        struct.pack_into("<ff", ball_raw, 0x2C, 2.0, 3.0)
        struct.pack_into("<f", ball_raw, 0x38, 18.0)
        struct.pack_into("<i", ball_raw, 0xC4, 149)
        struct.pack_into("<i", ball_raw, 0xC8, 3)
        struct.pack_into("<i", ball_raw, 0xF8, 0)
        struct.pack_into("<i", ball_raw, 0xFC, 100)
        struct.pack_into("<f", ball_raw, 0x104, 5.0)
        struct.pack_into("<f", ball_raw, 0x108, 0.04)
        struct.pack_into("<i", ball_raw, 0x10C, -1)
        struct.pack_into("<i", ball_raw, 0x11C, 3)
        struct.pack_into("<i", ball_raw, 0x120, 14)
        record.update(
            {
                "artifact_offset": 0,
                "artifact_bytes": len(ball_raw),
                "payload_sha256": (
                    "sha256:" + hashlib.sha256(ball_raw).hexdigest()
                ),
            }
        )
        active_list.update(
            _artifact(
                tick_root,
                "curve-00-list-05c-payloads.bin",
                bytes(ball_raw),
            )
        )
    if include_gap_raw:
        bullet_raw = bytearray(0x18C)
        struct.pack_into("<I", bullet_raw, 0, 0x009636E4)
        struct.pack_into("<I", bullet_raw, 0x10, 40)
        struct.pack_into("<i", bullet_raw, 0x14, 3)
        struct.pack_into("<f", bullet_raw, 0x1C, 0.0)
        struct.pack_into("<ff", bullet_raw, 0x2C, 108.0, 117.0)
        struct.pack_into("<f", bullet_raw, 0x38, 18.0)
        struct.pack_into("<f", bullet_raw, 0x138, 1.0)
        struct.pack_into("<f", bullet_raw, 0x13C, 0.0)
        struct.pack_into("<f", bullet_raw, 0x15C, 0.025)
        struct.pack_into("<I", bullet_raw, 0x174, 0x1000)
        struct.pack_into("<I", bullet_raw, 0x178, 1)
        struct.pack_into("<4i", bullet_raw, 0x17C, 109, 0, 0, 0)
        gap_raw = struct.pack(
            "<IIiii",
            0x1000,
            0x1000,
            0,
            120,
            2,
        )
        fired_artifact = _artifact(
            tick_root,
            "active-fired-bullets.bin",
            bytes(bullet_raw),
        )
        gap_artifact = _artifact(
            tick_root,
            "active-fired-bullet-gap-nodes.bin",
            gap_raw,
        )
        active_board["fired_bullets"] = {
            "traversed_count": 1,
            "declared_count": 1,
            **fired_artifact,
            "gap_artifact": gap_artifact["artifact"],
            "gap_artifact_bytes": gap_artifact["artifact_bytes"],
            "gap_artifact_sha256": gap_artifact["artifact_sha256"],
            "records": [
                {
                    "index": 0,
                    "artifact_offset": 0,
                    "artifact_bytes": len(bullet_raw),
                    "payload_sha256": (
                        "sha256:" + hashlib.sha256(bullet_raw).hexdigest()
                    ),
                    "gap_artifact_offset": 0,
                    "gap_artifact_bytes": len(gap_raw),
                    "gap_payload_sha256": (
                        "sha256:" + hashlib.sha256(gap_raw).hexdigest()
                    ),
                    "payload_kind": "bullet",
                    "ball": {
                        "ball_id": 40,
                        "color_id": 3,
                        "curve_distance": 0.0,
                        "position_x": 108.0,
                        "position_y": 117.0,
                        "radius": 18.0,
                    },
                    "subclass_fields": {
                        "velocity_x": 1.0,
                        "velocity_y": 0.0,
                        "field_158_float": 0.0,
                        "field_15c_float": 0.025,
                        "fired": False,
                        "field_174_pointer": 0x1000,
                        "field_178_i32": 1,
                        "gap_list_sentinel_address": 0x1000,
                        "gap_entry_count": 1,
                        "curve_points": [109, 0, 0, 0],
                        "gap_entries": [
                            {
                                "index": 0,
                                "node_address": 0x2000,
                                "next_node_address": 0x1000,
                                "previous_node_address": 0x1000,
                                "curve_index": 0,
                                "gap_distance": 120,
                                "boundary_ball_id": 2,
                            }
                        ],
                    },
                }
            ],
        }
    if include_win_state_raw:
        board_raw = bytearray(0x1100)
        struct.pack_into("<I", board_raw, 0x00, 0x0096356C)
        struct.pack_into("<I", board_raw, 0x88, 0x0096368C)
        struct.pack_into("<i", board_raw, 0x104, 10)
        struct.pack_into("<i", board_raw, 0x108, 20)
        struct.pack_into("<i", board_raw, 0xEFC, 10)
        struct.pack_into("<i", board_raw, 0xEC8, 3_759)
        struct.pack_into("<I", board_raw, 0xB4, 0x25000030)
        struct.pack_into("<i", board_raw, 0x118, 2)
        struct.pack_into("<ff", board_raw, 0x130, -2.5, 0.0)
        board_raw[0x153] = 1
        struct.pack_into(
            "<ffff",
            board_raw,
            0xE90,
            0.25,
            0.25,
            -0.01,
            1.5,
        )
        struct.pack_into("<iiiii", board_raw, 0xEAC, 12, -12, 247, 4_759, 25)
        active_board.update(
            {
                "schema": "zuma-rl.pc-active-board-probe",
                "version": 2,
                "mode_flag_1064_offset": 0x1064,
                "mode_flag_1064": False,
                **_artifact(
                    tick_root,
                    "active-board.bin",
                    bytes(board_raw),
                ),
            }
        )
        exhausted = _artifact(
            tick_root,
            "curve-plan-exhausted.bin",
            b"\x00",
        )
        active_board["curve_plan_exhausted"] = {
            "address": 0x009E8252,
            "value": False,
            **exhausted,
        }
        manager = active_board["curve_manager"]
        assert isinstance(manager, dict)
        manager_raw = bytearray(0x40C)
        struct.pack_into("<I", manager_raw, 0x00, 0x0096A204)
        struct.pack_into("<I", manager_raw, 0x16C, 0x31000000)
        struct.pack_into("<i", manager_raw, 0x360, 1)
        struct.pack_into("<i", manager_raw, 0x400, 0)
        struct.pack_into("<f", manager_raw, 0x404, 0.25)
        struct.pack_into("<f", manager_raw, 0x408, 0.5)
        manager.update(
            {
                "curve_count": 1,
                "curve_count_offset": 0x360,
                "curve_array_offset": 0x16C,
                "post_zuma_diagnostics": {
                    "timer_remaining_offset": 0x400,
                    "timer_remaining": 0,
                    "ramp_404_offset": 0x404,
                    "ramp_404": 0.25,
                    "ramp_408_offset": 0x408,
                    "ramp_408": 0.5,
                },
                **_artifact(
                    tick_root,
                    "curve-manager.bin",
                    bytes(manager_raw),
                ),
            }
        )
        curve = manager["curves"][0]
        assert isinstance(curve, dict)
        curve_raw = bytearray(0x200)
        struct.pack_into("<I", curve_raw, 0x00, 0x009641F0)
        struct.pack_into("<i", curve_raw, 0x17C, 20)
        struct.pack_into("<f", curve_raw, 0x190, 3.25)
        struct.pack_into("<i", curve_raw, 0x19C, 1_234)
        curve_raw[0x1A0] = 1
        curve_raw[0x1A3] = 1
        curve_raw[0x1BE] = 1
        curve_raw[0x1BF] = 0
        curve_raw[0x1C0] = 1
        plan_artifact = _artifact(
            tick_root,
            "curve-00-plan.bin",
            b"",
        )
        curve.update(
            {
                "address": 0x31000000,
                "planned_balls": {
                    "vector_begin_offset": 0x34,
                    "vector_end_offset": 0x38,
                    "vector_capacity_offset": 0x3C,
                    "begin_address": 0,
                    "end_address": 0,
                    "capacity_address": 0,
                    "item_size": 0x14,
                    "count": 0,
                    "capacity_count": 0,
                    "add_plan_enabled_offset": 0x1A3,
                    "add_plan_enabled": True,
                    **plan_artifact,
                },
                **_artifact(
                    tick_root,
                    "curve-00.bin",
                    bytes(curve_raw),
                ),
            }
        )
    tick = {
        "schema": "zuma-rl.pc-memory-trajectory-tick",
        "version": 2,
        "framework_update": 1,
        "sample_phase": "frozen_post_replay_update_barrier",
        "sample_barrier": "freeze_multiplier_message",
        "replay_state": {
            "update_count": 1,
            "frame_time_ms": 10,
            "update_multiplier": 0.0877914951989026,
            "fast_forward_target": 0,
            "fast_forward_to_marker": False,
            "fast_forward_step": False,
        },
        "active_board": active_board,
    }
    tick_path = tick_root / "tick.json"
    payload = _canonical(tick)
    tick_path.write_bytes(payload)
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    index_row: dict[str, object] = {
        "framework_update": 1,
        "score": 10,
        "displayed_score": 10,
        "score_target": 20,
        "chain_ball_count": 1,
        "fired_bullet_count": 1 if include_gap_raw else 0,
        "current_ball_id": 2,
        "current_color_id": 1,
        "next_ball_id": 3,
        "next_color_id": 2,
        "list_counts": [
            {
                "curve_index": 0,
                "container_offset": 0x50,
                "count": 0,
            },
            {
                "curve_index": 0,
                "container_offset": 0x5C,
                "count": 1,
            },
        ],
        "artifact": "tick-00000001/tick.json",
        "artifact_sha256": digest,
    }
    if include_rng:
        index_row.update(
            {
                "qrand_update_count": 4,
                "qrand_selected_index": 2,
                "thread_crt_rand_state": 1_674_403_832,
            }
        )
    if include_win_state_raw:
        index_row.update(
            {
                "active_board_version": 2,
                "board_mode_flag_1064": False,
                "curve_plan_exhausted": False,
                "post_zuma_timer_remaining": 0,
                "post_zuma_ramp_404": 0.25,
                "post_zuma_ramp_408": 0.5,
                "curve_plans": [
                    {
                        "curve_index": 0,
                        "begin_address": 0,
                        "end_address": 0,
                        "capacity_address": 0,
                        "planned_count": 0,
                        "capacity_count": 0,
                        "add_plan_enabled": True,
                    }
                ],
            }
        )
    index = {
        "schema": "zuma-rl.pc-memory-trajectory",
        "version": 2,
        "sample_phase": "frozen_post_replay_update_barrier",
        "start_update": 1,
        "end_update": 1,
        "tick_count": 1,
        "ticks": [index_row],
    }
    index_path = tmp_path / "index.json"
    index_path.write_bytes(_canonical(index))
    return index_path


def _write_legacy_one_tick_trajectory(tmp_path: Path) -> Path:
    index_path = _write_one_tick_trajectory(tmp_path)
    tick_path = tmp_path / "tick-00000001" / "tick.json"
    tick = json.loads(tick_path.read_text(encoding="ascii"))
    tick["version"] = 1
    tick["sample_phase"] = "frozen_post_framework_update"
    tick.pop("sample_barrier")
    tick["replay_state"] = {
        "draw_count": 1,
        "fast_forward_target": 0,
        "frame_time_ms": 10,
        "loaded": True,
        "loading_thread_completed": True,
        "loading_thread_started": True,
        "multiplier_address": 0x1234,
        "non_draw_count": 0,
        "paused": False,
        "sleep_count": 0,
        "update_app_depth": 0,
        "update_app_state": 0,
        "update_count": 1,
        "update_multiplier": 0.0877914951989026,
    }
    tick_payload = _canonical(tick)
    tick_path.write_bytes(tick_payload)

    index = json.loads(index_path.read_text(encoding="ascii"))
    index["version"] = 1
    index["sample_phase"] = "frozen_post_framework_update"
    index["ticks"][0]["artifact_sha256"] = (
        "sha256:" + hashlib.sha256(tick_payload).hexdigest()
    )
    index_path.write_bytes(_canonical(index))
    return index_path


def test_loader_recomputes_index_summary_and_hash(tmp_path: Path) -> None:
    index_path = _write_one_tick_trajectory(tmp_path)

    frames = load_memory_trajectory(index_path)

    assert len(frames) == 1
    assert frames[0].list_count(0, 0x5C) == 1
    assert frames[0].entities_by_id[1].zone == "curve:0:list:05c"
    assert frames[0].entities_by_id[2].fired is False

    tick_path = tmp_path / "tick-00000001" / "tick.json"
    tick_path.write_bytes(tick_path.read_bytes() + b" ")
    with pytest.raises(
        PcMemoryTrajectoryError,
        match="trajectory_tick_artifact_sha256_mismatch",
    ):
        load_memory_trajectory(index_path)


def test_legacy_v1_loader_is_explicit_and_hash_bound(tmp_path: Path) -> None:
    index_path = _write_legacy_one_tick_trajectory(tmp_path)

    with pytest.raises(
        PcMemoryTrajectoryError,
        match="trajectory_index_schema_invalid",
    ):
        load_memory_trajectory(index_path)

    frames = load_legacy_memory_trajectory_v1(index_path)
    assert [frame.update for frame in frames] == [1]
    assert frames[0].score == 10
    assert frames[0].entities_by_id[1].zone == "curve:0:list:05c"

    tick_path = tmp_path / "tick-00000001" / "tick.json"
    tick_path.write_bytes(tick_path.read_bytes() + b" ")
    with pytest.raises(
        PcMemoryTrajectoryError,
        match="legacy_trajectory_tick_artifact_sha256_mismatch",
    ):
        load_legacy_memory_trajectory_v1(index_path)


def test_loader_accepts_a_verified_inclusive_window(tmp_path: Path) -> None:
    index_path = _write_one_tick_trajectory(tmp_path)

    frames = load_memory_trajectory(
        index_path,
        start_update=1,
        end_update=1,
    )

    assert [frame.update for frame in frames] == [1]


@pytest.mark.parametrize(
    ("start_update", "end_update"),
    ((0, 1), (1, 2), (2, 2), (2, 1)),
)
def test_loader_rejects_a_window_outside_the_index(
    tmp_path: Path,
    start_update: int,
    end_update: int,
) -> None:
    index_path = _write_one_tick_trajectory(tmp_path)

    with pytest.raises(
        PcMemoryTrajectoryError,
        match="trajectory_window_range_invalid",
    ):
        load_memory_trajectory(
            index_path,
            start_update=start_update,
            end_update=end_update,
        )


def test_loader_binds_qrand_and_thread_crt_artifacts(
    tmp_path: Path,
) -> None:
    index_path = _write_one_tick_trajectory(
        tmp_path,
        include_rng=True,
    )

    frame = load_memory_trajectory(index_path)[0]

    assert frame.thread_crt_rand_state == 1_674_403_832
    assert frame.qrand is not None
    assert frame.qrand.update_count == 4
    assert frame.qrand.selected_index == 2
    assert frame.qrand.weights == (0.25, 0.25, 0.25, 0.25, 0.0, 0.0)
    assert frame.qrand.last_hit == (2, 0, 4, 3, 0, 0)

    sway_path = (
        tmp_path / "tick-00000001" / "qrand-sways.bin"
    )
    payload = bytearray(sway_path.read_bytes())
    payload[0] ^= 1
    sway_path.write_bytes(payload)
    with pytest.raises(
        PcMemoryTrajectoryError,
        match="trajectory_tick_qrand_sways_artifact_sha256_mismatch",
    ):
        load_memory_trajectory(index_path)


def test_qrand_accepts_only_fully_uninitialized_zero_vectors(
    tmp_path: Path,
) -> None:
    raw = bytearray(0x48)
    struct.pack_into("<ii", raw, 0, 0, -1)
    vectors = []
    for offset, name, value_type in trajectory_module.QRAND_VECTOR_LAYOUT:
        vectors.append(
            {
                "name": name,
                "value_type": value_type,
                "object_offset": offset,
                "begin": 0,
                "end": 0,
                "capacity": 0,
                "item_count": 0,
                "capacity_count": 0,
                "values": [],
                **_artifact(tmp_path, f"qrand-{name}.bin", b""),
            }
        )
    qrand = {
        "schema": "zuma-rl.pc-qrand-state",
        "version": 1,
        "board_pointer_offset": 0x7A8,
        "update_count": 0,
        "selected_index": -1,
        "vectors": vectors,
        **_artifact(tmp_path, "qrand.bin", bytes(raw)),
    }

    state = trajectory_module._qrand_from_board(
        {"qrand": qrand},
        tick_root=tmp_path,
    )

    assert state is not None
    assert state.update_count == 0
    assert state.selected_index == -1
    assert state.weights == ()
    assert state.previous_hit == ()

    raw[0:4] = struct.pack("<i", 1)
    qrand["update_count"] = 1
    qrand.update(_artifact(tmp_path, "qrand.bin", bytes(raw)))
    with pytest.raises(
        PcMemoryTrajectoryError,
        match="trajectory_tick_qrand_vector_header_mismatch",
    ):
        trajectory_module._qrand_from_board(
            {"qrand": qrand},
            tick_root=tmp_path,
        )


def test_loader_binds_raw_powerup_and_curve_manager_fields(
    tmp_path: Path,
) -> None:
    index_path = _write_one_tick_trajectory(
        tmp_path,
        include_powerup_raw=True,
    )

    frame = load_memory_trajectory(index_path)[0]
    ball = frame.entities_by_id[1]
    manager = frame.curve_powerup_state(0)

    assert frame.native_game_time == 3_759
    assert ball.powerup_previous_ticks == 149
    assert ball.powerup_previous_type == 3
    assert ball.powerup_primary_type == 3
    assert ball.powerup_secondary_type == 14
    assert ball.powerup_lifetime_ticks == 0
    assert ball.powerup_transition_ticks == 100
    assert ball.powerup_visual_scale == pytest.approx(5.0)
    assert ball.powerup_visual_step == pytest.approx(0.04)
    assert ball.powerup_visual_index == -1
    assert ball.radius == pytest.approx(18.0)
    assert ball.backwards_count == 0
    assert ball.backwards_speed == pytest.approx(0.0)
    assert manager is not None
    assert manager.last_any_spawn_time == 2_815
    assert manager.last_spawn_times[3] == 1_660
    assert manager.cooldown_times[3] == 3_759
    assert manager.spawn_counts[3] == 1
    assert manager.reverse_speed == pytest.approx(0.0)
    assert manager.slow_ticks == 0
    assert manager.reverse_ticks == 0
    assert manager.last_powerup_waypoint == 0
    assert manager.powerup_triggered is False


def test_loader_binds_native_curve_runtime_latches(tmp_path: Path) -> None:
    index_path = _write_one_tick_trajectory(
        tmp_path,
        include_win_state_raw=True,
    )

    frame = load_memory_trajectory(index_path)[0]
    runtime = frame.curve_runtime_state(0)

    assert runtime is not None
    assert runtime.advance_speed == pytest.approx(3.25)
    assert runtime.first_chain_end == 1_234
    assert runtime.stop_adding is True
    assert runtime.has_reached_cruising_speed is False
    assert runtime.has_reached_rollout is True
    assert runtime.stop_time == 20
    assert runtime.first_ball_moved_backwards is True


def test_curve_runtime_latches_fail_closed_on_non_boolean_byte(
    tmp_path: Path,
) -> None:
    raw = bytearray(0x200)
    raw[trajectory_module.CURVE_HAS_REACHED_CRUISING_SPEED_OFFSET] = 2
    curve = _artifact(tmp_path, "curve-00.bin", bytes(raw))

    with pytest.raises(
        PcMemoryTrajectoryError,
        match="trajectory_tick_curve_runtime_state_invalid",
    ):
        trajectory_module._curve_runtime_state(
            curve,
            curve_index=0,
            tick_root=tmp_path,
        )


def test_loader_binds_raw_gap_nodes_and_curve_point_latches(
    tmp_path: Path,
) -> None:
    index_path = _write_one_tick_trajectory(
        tmp_path,
        include_gap_raw=True,
    )

    frame = load_memory_trajectory(index_path)[0]
    projectile = frame.entities_by_id[40]

    assert projectile.zone == "fired"
    assert projectile.gap_list_sentinel_address == 0x1000
    assert projectile.gap_entry_count == 1
    assert projectile.curve_points == (109, 0, 0, 0)
    assert projectile.gap_entries == ((0, 120, 2),)

    gap_path = (
        tmp_path
        / "tick-00000001"
        / "active-fired-bullet-gap-nodes.bin"
    )
    payload = bytearray(gap_path.read_bytes())
    payload[-1] ^= 1
    gap_path.write_bytes(payload)
    with pytest.raises(
        PcMemoryTrajectoryError,
        match="trajectory_tick_gap_artifact_sha256_mismatch",
    ):
        load_memory_trajectory(index_path)


@pytest.mark.parametrize("fired", [False, True])
def test_chamber_bullet_preserves_debug_fill_curve_points(
    fired: bool,
) -> None:
    sentinel = -1163005939  # Signed 0xBAADF00D from untouched retail fields.
    raw = bytearray(0x18C)
    struct.pack_into("<I", raw, 0x10, 40)
    struct.pack_into("<i", raw, 0x14, 3)
    struct.pack_into("<f", raw, 0x1C, 0.0)
    struct.pack_into("<ff", raw, 0x2C, 108.0, 117.0)
    struct.pack_into("<f", raw, 0x38, 18.0)
    struct.pack_into("<I", raw, 0x174, 0x1000)
    struct.pack_into("<I", raw, 0x178, 0)
    struct.pack_into("<4i", raw, 0x17C, *([sentinel] * 4))
    raw[0x16A] = int(fired)
    payload = bytes(raw)
    record = {
        "payload_kind": "bullet",
        "payload_sha256": (
            "sha256:" + hashlib.sha256(payload).hexdigest()
        ),
        "ball": {
            "ball_id": 40,
            "color_id": 3,
            "curve_distance": 0.0,
            "position_x": 108.0,
            "position_y": 117.0,
            "radius": 18.0,
        },
        "subclass_fields": {
            "velocity_x": 0.0,
            "velocity_y": 0.0,
            "field_158_float": 0.0,
            "field_15c_float": 0.025,
            "fired": fired,
            "gap_list_sentinel_address": 0x1000,
            "gap_entry_count": 0,
            "field_174_pointer": 0x1000,
            "field_178_i32": 0,
            "curve_points": [sentinel] * 4,
        },
    }

    entity = trajectory_module._entity(
        record,
        zone="shooter_current",
        index=0,
        default_kind="bullet",
        raw_payload=payload,
    )

    assert entity.fired is fired
    assert entity.gap_entry_count == 0
    assert entity.curve_points == (sentinel,) * 4


def test_debug_fill_is_allowed_only_in_unreferenced_curve_slots() -> None:
    sentinel = -1163005939  # Signed 0xBAADF00D from untouched retail fields.

    assert trajectory_module._bullet_curve_points_valid(
        (397, sentinel, sentinel, sentinel),
        ((0, 36, 7),),
    )
    assert not trajectory_module._bullet_curve_points_valid(
        (397, sentinel, sentinel, sentinel),
        ((1, 36, 7),),
    )
    assert not trajectory_module._bullet_curve_points_valid(
        (397, -1, sentinel, sentinel),
        ((0, 36, 7),),
    )


def test_bullet_rejects_gap_count_without_matching_entries() -> None:
    record = {
        "payload_kind": "bullet",
        "ball": {
            "ball_id": 40,
            "color_id": 3,
            "curve_distance": 0.0,
            "position_x": 108.0,
            "position_y": 117.0,
            "radius": 18.0,
        },
        "subclass_fields": {
            "velocity_x": 0.0,
            "velocity_y": 0.0,
            "fired": False,
            "gap_list_sentinel_address": 0x1000,
            "gap_entry_count": 1,
            "curve_points": [109, 0, 0, 0],
        },
    }

    with pytest.raises(
        PcMemoryTrajectoryError,
        match="trajectory_entity_gap_count_mismatch",
    ):
        trajectory_module._entity(
            record,
            zone="fired",
            index=0,
            default_kind="bullet",
        )


def test_loader_binds_v2_natural_win_state_fields(
    tmp_path: Path,
) -> None:
    frame = load_memory_trajectory(
        _write_one_tick_trajectory(
            tmp_path,
            include_win_state_raw=True,
        )
    )[0]

    assert frame.active_board_version == 2
    assert frame.board_mode_flag_1064 is False
    assert frame.curve_plan_exhausted is False
    assert frame.post_zuma_timer_remaining == 0
    assert frame.post_zuma_ramp_404 == pytest.approx(0.25)
    assert frame.post_zuma_ramp_408 == pytest.approx(0.5)
    plan = frame.curve_plan_state(0)
    assert plan is not None
    assert plan.planned_count == 0
    assert plan.capacity_count == 0
    assert plan.add_plan_enabled is True
    fruit = frame.fruit_state
    assert fruit is not None
    assert fruit.active_point_pointer == 0x25000030
    assert fruit.active is True
    assert fruit.selected_point_index == 2
    assert fruit.collecting is True
    assert fruit.velocity == pytest.approx(0.25)
    assert fruit.max_velocity == pytest.approx(0.25)
    assert fruit.acceleration == pytest.approx(-0.01)
    assert fruit.vertical_offset == pytest.approx(1.5)
    assert fruit.lower_bound == pytest.approx(-2.5)
    assert fruit.upper_bound == pytest.approx(0.0)
    assert fruit.glow_alpha == 12
    assert fruit.glow_step == -12
    assert fruit.alpha == 247
    assert fruit.expiry_time == 4_759
    assert fruit.cell_index == 25


def test_loader_accepts_hash_bound_historical_v2_extended_state(
    tmp_path: Path,
) -> None:
    index_path = _write_one_tick_trajectory(
        tmp_path,
        include_win_state_raw=True,
    )
    index = json.loads(index_path.read_text(encoding="ascii"))
    for key in (
        "active_board_version",
        "board_mode_flag_1064",
        "curve_plan_exhausted",
        "post_zuma_timer_remaining",
        "post_zuma_ramp_404",
        "post_zuma_ramp_408",
        "curve_plans",
    ):
        index["ticks"][0].pop(key)
    index_path.write_bytes(_canonical(index))

    frame = load_memory_trajectory(index_path)[0]

    assert frame.active_board_version == 2
    assert frame.fruit_state is not None
    assert frame.fruit_state.cell_index == 25


def test_loader_requires_extended_summary_for_formal_v2_index(
    tmp_path: Path,
) -> None:
    index_path = _write_one_tick_trajectory(
        tmp_path,
        include_win_state_raw=True,
    )
    index = json.loads(index_path.read_text(encoding="ascii"))
    index["evidence_classification"] = "formal_full_state_exact_step"
    index["ticks"][0].pop("active_board_version")
    index_path.write_bytes(_canonical(index))

    with pytest.raises(
        PcMemoryTrajectoryError,
        match="trajectory_index_summary_mismatch:active_board_version",
    ):
        load_memory_trajectory(index_path)
