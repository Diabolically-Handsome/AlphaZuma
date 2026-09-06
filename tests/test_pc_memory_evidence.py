from __future__ import annotations

import json
from pathlib import Path
import struct
from typing import Any

import pytest

from zuma_rl.pc_memory_evidence import (
    BALL_OBJECT_SIZE,
    BALL_VTABLE,
    BULLET_OBJECT_SIZE,
    BULLET_VTABLE,
    FROZEN_UPDATE_MULTIPLIER,
    FORMAL_FULL_STATE_CLASSIFICATION,
    FORMAL_FULL_STATE_UPDATE_MULTIPLIER,
    INDEPENDENT_SCORE_BINDING_ACCEPTANCE_RULE,
    INDEPENDENT_SCORE_BINDING_MODE,
    PcMemoryTransitionContract,
    PcMemoryEvidenceError,
    _validate_curve_payload_records,
    _validate_fired_gap_records,
    build_memory_curve_geometry_binding,
    decode_ball,
    decode_bullet_subclass,
    sha256_bytes,
    validate_probe_payloads,
)


_RUNTIME_SHA256 = "sha256:" + "12" * 32
_DMO_SHA256 = "sha256:" + "34" * 32
_BOARD_ADDRESS = 0x10000000
_SHOOTER_ADDRESS = 0x20000000
_CURRENT_ADDRESS = 0x21000000
_NEXT_ADDRESS = 0x22000000
_MANAGER_ADDRESS = 0x30000000
_CURVE_ADDRESS = 0x31000000


def _contract(**changes: Any) -> PcMemoryTransitionContract:
    values: dict[str, Any] = {
        "case_id": "case",
        "before_probe_artifact": "memory.before.probe",
        "after_probe_artifact": "memory.after.probe",
        "before_pixel_validation_artifact": "memory.before.pixel",
        "after_pixel_validation_artifact": "memory.after.pixel",
        "transition_validation_artifact": "memory.transition",
        "video_binding_validation_artifact": "memory.video",
        "expected_before_update": 7738,
        "expected_after_update": 7750,
        "expected_score": 7950,
        "expected_chain_distance_delta": 1.5,
        "distance_tolerance": 1e-6,
        "minimum_chain_count": 90,
        "minimum_visible": 90,
        "maximum_mismatches": 0,
        "minimum_visible_after_fired_bullets": 1,
        "maximum_fired_bullet_mismatches": 0,
        "video_binding_excluded_bottom_rows": 1,
        "maximum_excluded_edge_mismatches": 4,
    }
    values.update(changes)
    return PcMemoryTransitionContract(**values)


def _write_artifact(
    root: Path,
    name: str,
    payload: bytes,
) -> dict[str, Any]:
    (root / name).write_bytes(payload)
    return {
        "artifact": name,
        "artifact_bytes": len(payload),
        "artifact_sha256": sha256_bytes(payload),
    }


def _ball_payload(
    *,
    vtable: int,
    size: int,
    ball_id: int,
    color_id: int,
    distance: float,
    x: float,
    y: float,
    owner: int = 0,
    velocity: tuple[float, float] = (0.0, 0.0),
) -> bytes:
    payload = bytearray(size)
    struct.pack_into("<I", payload, 0x00, vtable)
    struct.pack_into("<I", payload, 0x10, ball_id)
    struct.pack_into("<i", payload, 0x14, color_id)
    struct.pack_into("<f", payload, 0x1C, distance)
    struct.pack_into("<ff", payload, 0x2C, x, y)
    struct.pack_into("<f", payload, 0x34, 1.0)
    struct.pack_into("<f", payload, 0x38, 18.0)
    if size == BULLET_OBJECT_SIZE:
        struct.pack_into("<I", payload, 0x130, owner)
        struct.pack_into("<ff", payload, 0x138, *velocity)
    return bytes(payload)


def _ball_record(
    payload: bytes,
    *,
    index: int,
    payload_address: int,
    bullet: bool,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "index": index,
        "payload_address": payload_address,
        "artifact_offset": index * len(payload),
        "artifact_bytes": len(payload),
        "payload_sha256": sha256_bytes(payload),
        "ball": decode_ball(
            payload,
            expected_vtable=BULLET_VTABLE if bullet else BALL_VTABLE,
        ),
    }
    if bullet:
        row["subclass_fields"] = decode_bullet_subclass(payload)
    return row


def _fixture(
    tmp_path: Path,
    *,
    active_board_version: int = 1,
) -> Path:
    freeze = bytearray(168)
    freeze_offset = 60
    struct.pack_into("<i", freeze, freeze_offset - 60, 100)
    struct.pack_into("<i", freeze, freeze_offset - 56, 10)
    struct.pack_into("<i", freeze, freeze_offset - 20, 20)
    struct.pack_into("<i", freeze, freeze_offset - 16, 2174)
    struct.pack_into("<i", freeze, freeze_offset - 12, 7750)
    struct.pack_into("<i", freeze, freeze_offset - 8, 0)
    struct.pack_into("<i", freeze, freeze_offset - 4, 0)
    struct.pack_into(
        "<d",
        freeze,
        freeze_offset,
        FROZEN_UPDATE_MULTIPLIER,
    )
    struct.pack_into("<i", freeze, freeze_offset + 12, 0)
    freeze[freeze_offset + 105] = 1
    freeze[freeze_offset + 106] = 1
    freeze[freeze_offset + 107] = 1
    freeze_artifact = _write_artifact(
        tmp_path,
        "freeze-state.bin",
        bytes(freeze),
    )

    board = bytearray(0x1100)
    struct.pack_into("<I", board, 0x00, 0x0096356C)
    struct.pack_into("<I", board, 0x88, 0x0096368C)
    struct.pack_into("<I", board, 0x9C, _MANAGER_ADDRESS)
    struct.pack_into("<i", board, 0x104, 7950)
    struct.pack_into("<i", board, 0x108, 9650)
    struct.pack_into("<I", board, 0x68C, _SHOOTER_ADDRESS)
    struct.pack_into("<I", board, 0x69C, 0x40000000)
    struct.pack_into("<I", board, 0x6A0, 1)
    struct.pack_into("<i", board, 0xEFC, 7950)

    shooter = bytearray(0x32C)
    struct.pack_into("<I", shooter, 0x00, 0x009670E0)
    struct.pack_into("<I", shooter, 0x130, _CURRENT_ADDRESS)
    struct.pack_into("<I", shooter, 0x134, _NEXT_ADDRESS)
    current = _ball_payload(
        vtable=BULLET_VTABLE,
        size=BULLET_OBJECT_SIZE,
        ball_id=99,
        color_id=2,
        distance=0.0,
        x=385.0,
        y=308.0,
        owner=_SHOOTER_ADDRESS,
    )
    next_bullet = _ball_payload(
        vtable=BULLET_VTABLE,
        size=BULLET_OBJECT_SIZE,
        ball_id=110,
        color_id=3,
        distance=0.0,
        x=0.0,
        y=0.0,
        owner=_SHOOTER_ADDRESS,
    )
    fired = _ball_payload(
        vtable=BULLET_VTABLE,
        size=BULLET_OBJECT_SIZE,
        ball_id=104,
        color_id=0,
        distance=0.0,
        x=357.0,
        y=232.0,
        owner=_SHOOTER_ADDRESS,
        velocity=(-2.75, -7.5),
    )

    manager = bytearray(0x40C if active_board_version == 2 else 0x400)
    struct.pack_into("<I", manager, 0x00, 0x0096A204)
    struct.pack_into("<I", manager, 0x16C, _CURVE_ADDRESS)
    struct.pack_into("<i", manager, 0x360, 1)
    if active_board_version == 2:
        struct.pack_into("<i", manager, 0x400, 0)
        struct.pack_into("<f", manager, 0x404, 0.25)
        struct.pack_into("<f", manager, 0x408, 0.5)
    curve = bytearray(0x200)
    struct.pack_into("<I", curve, 0x00, 0x009641F0)
    if active_board_version == 2:
        curve[0x1A3] = 1
    list_values = {
        0x50: [],
        0x5C: [
            _ball_payload(
                vtable=BALL_VTABLE,
                size=BALL_OBJECT_SIZE,
                ball_id=1,
                color_id=3,
                distance=10.0,
                x=100.0,
                y=100.0,
            ),
            _ball_payload(
                vtable=BALL_VTABLE,
                size=BALL_OBJECT_SIZE,
                ball_id=2,
                color_id=4,
                distance=46.0,
                x=136.0,
                y=100.0,
            ),
        ],
        0x68: [
            _ball_payload(
                vtable=BALL_VTABLE,
                size=BALL_OBJECT_SIZE,
                ball_id=3,
                color_id=1,
                distance=1.0,
                x=-30.0,
                y=78.0,
            )
        ],
    }
    list_rows: list[dict[str, Any]] = []
    for list_index, (offset, values) in enumerate(list_values.items()):
        sentinel = 0x50000000 + list_index * 0x100
        struct.pack_into("<I", curve, offset + 4, sentinel)
        struct.pack_into("<I", curve, offset + 8, len(values))
        blob = b"".join(values)
        artifact = _write_artifact(
            tmp_path,
            f"curve-list-{offset:03x}.bin",
            blob,
        )
        list_rows.append(
            {
                "container_offset": offset,
                "sentinel_address": sentinel,
                "declared_count": len(values),
                "traversed_count": len(values),
                "payload_count": len(values),
                "records": [
                    _ball_record(
                        value,
                        index=index,
                        payload_address=0x60000000 + index * 0x1000,
                        bullet=False,
                    )
                    for index, value in enumerate(values)
                ],
                **artifact,
            }
        )

    current_artifact = _write_artifact(
        tmp_path,
        "shooter-current.bin",
        current,
    )
    next_artifact = _write_artifact(
        tmp_path,
        "shooter-next.bin",
        next_bullet,
    )
    fired_artifact = _write_artifact(
        tmp_path,
        "fired.bin",
        fired,
    )
    shooter_artifact = _write_artifact(
        tmp_path,
        "shooter.bin",
        bytes(shooter),
    )
    curve_artifact = _write_artifact(
        tmp_path,
        "curve.bin",
        bytes(curve),
    )
    manager_artifact = _write_artifact(
        tmp_path,
        "manager.bin",
        bytes(manager),
    )
    board_artifact = _write_artifact(
        tmp_path,
        "board.bin",
        bytes(board),
    )
    v2_board_fields: dict[str, Any] = {}
    v2_manager_fields: dict[str, Any] = {}
    v2_curve_fields: dict[str, Any] = {}
    if active_board_version == 2:
        plan_artifact = _write_artifact(
            tmp_path,
            "curve-plan.bin",
            b"",
        )
        exhausted_artifact = _write_artifact(
            tmp_path,
            "curve-plan-exhausted.bin",
            b"\x00",
        )
        v2_board_fields = {
            "mode_flag_1064_offset": 0x1064,
            "mode_flag_1064": False,
            "curve_plan_exhausted": {
                "address": 0x009E8252,
                "value": False,
                **exhausted_artifact,
            },
        }
        v2_manager_fields = {
            "post_zuma_diagnostics": {
                "timer_remaining_offset": 0x400,
                "timer_remaining": 0,
                "ramp_404_offset": 0x404,
                "ramp_404": 0.25,
                "ramp_408_offset": 0x408,
                "ramp_408": 0.5,
            }
        }
        v2_curve_fields = {
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
            }
        }
    frame = b"BM" + bytes(126)
    (tmp_path / "frame.bmp").write_bytes(frame)

    probe = {
        "schema": "zuma-rl.pc-memory-int32-probe",
        "version": 1,
        "framework_update": 7750,
        "value": 7950,
        "runtime_executable_sha256": _RUNTIME_SHA256,
        "dmo_sha256": _DMO_SHA256,
        "freeze_state": {
            "schema": "zuma-rl.pc-replay-freeze-state",
            "version": 1,
            "multiplier_offset": freeze_offset,
            "multiplier_address": 0x5000003C,
            "non_draw_count": 100,
            "frame_time_ms": 10,
            "sleep_count": 20,
            "draw_count": 2174,
            "update_count": 7750,
            "update_app_state": 0,
            "update_app_depth": 0,
            "update_multiplier": FROZEN_UPDATE_MULTIPLIER,
            "paused": False,
            "fast_forward_target": 0,
            "loading_thread_started": True,
            "loading_thread_completed": True,
            "loaded": True,
            **freeze_artifact,
        },
        "frozen_frame": {
            "artifact": "frame.bmp",
            "sha256": sha256_bytes(frame),
        },
        "active_board": {
            "schema": "zuma-rl.pc-active-board-probe",
            "version": active_board_version,
            "board_vtable": 0x0096356C,
            "embedded_vtable_offset": 0x88,
            "embedded_vtable": 0x0096368C,
            "score_offset": 0x104,
            "score": 7950,
            "score_target_offset": 0x108,
            "score_target": 9650,
            "displayed_score_offset": 0xEFC,
            "displayed_score": 7950,
            **v2_board_fields,
            "primary_child": {
                "board_offset": 0x68C,
                "address": _SHOOTER_ADDRESS,
                "bullets": [
                    {
                        "shooter_pointer_offset": 0x130,
                        "address": _CURRENT_ADDRESS,
                        "ball": decode_ball(
                            current,
                            expected_vtable=BULLET_VTABLE,
                        ),
                        "subclass_fields": decode_bullet_subclass(current),
                        **current_artifact,
                    },
                    {
                        "shooter_pointer_offset": 0x134,
                        "address": _NEXT_ADDRESS,
                        "ball": decode_ball(
                            next_bullet,
                            expected_vtable=BULLET_VTABLE,
                        ),
                        "subclass_fields": decode_bullet_subclass(next_bullet),
                        **next_artifact,
                    },
                ],
                **shooter_artifact,
            },
            "fired_bullets": {
                "board_container_offset": 0x698,
                "sentinel_address": 0x40000000,
                "declared_count": 1,
                "traversed_count": 1,
                "records": [
                    _ball_record(
                        fired,
                        index=0,
                        payload_address=0x70000000,
                        bullet=True,
                    )
                ],
                **fired_artifact,
            },
            "curve_manager": {
                "board_offset": 0x9C,
                "address": _MANAGER_ADDRESS,
                "curve_count_offset": 0x360,
                "curve_array_offset": 0x16C,
                "curve_count": 1,
                "curves": [
                    {
                        "index": 0,
                        "address": _CURVE_ADDRESS,
                        "intrusive_lists": list_rows,
                        **v2_curve_fields,
                        **curve_artifact,
                    }
                ],
                **v2_manager_fields,
                **manager_artifact,
            },
            **board_artifact,
        },
    }
    path = tmp_path / "memory-probe.json"
    path.write_text(
        json.dumps(probe, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
    return path


def _independent_score_fixture(
    tmp_path: Path,
    *,
    displayed_score: int = 7940,
) -> Path:
    path = _fixture(tmp_path)
    probe = json.loads(path.read_text(encoding="ascii"))
    board_path = tmp_path / probe["active_board"]["artifact"]
    board = bytearray(board_path.read_bytes())
    struct.pack_into("<i", board, 0xEFC, displayed_score)
    board_path.write_bytes(board)
    probe["active_board"]["artifact_sha256"] = sha256_bytes(board)
    probe["active_board"]["displayed_score"] = displayed_score
    probe["evidence_classification"] = FORMAL_FULL_STATE_CLASSIFICATION
    freeze_path = tmp_path / probe["freeze_state"]["artifact"]
    freeze = bytearray(freeze_path.read_bytes())
    struct.pack_into(
        "<d",
        freeze,
        probe["freeze_state"]["multiplier_offset"],
        FORMAL_FULL_STATE_UPDATE_MULTIPLIER,
    )
    freeze_path.write_bytes(freeze)
    probe["freeze_state"]["artifact_sha256"] = sha256_bytes(freeze)
    probe["freeze_state"][
        "update_multiplier"
    ] = FORMAL_FULL_STATE_UPDATE_MULTIPLIER
    probe["score_binding"] = {
        "schema": "zuma-rl.pc-memory-score-binding",
        "version": 2,
        "mode": INDEPENDENT_SCORE_BINDING_MODE,
        "score": 7950,
        "displayed_score": displayed_score,
        "scan_value": 7950,
        "score_offset": 0x104,
        "displayed_score_offset": 0xEFC,
        "acceptance_rule": INDEPENDENT_SCORE_BINDING_ACCEPTANCE_RULE,
    }
    path.write_text(
        json.dumps(probe, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
    return path


def test_raw_probe_recomputes_every_transition_field(
    tmp_path: Path,
) -> None:
    result = validate_probe_payloads(
        _fixture(tmp_path),
        expected_runtime_sha256=_RUNTIME_SHA256,
        expected_dmo_sha256=_DMO_SHA256,
        expected_update=7750,
        expected_score=7950,
        require_freeze_state=True,
    )

    assert result["status"] == "PASS"
    assert result["active_chain_count"] == 2
    assert result["fired_bullet_count"] == 1
    assert result["freeze_state"]["update_count"] == 7750
    assert result["shooter_current"]["ball"]["ball_id"] == 99
    assert result["fired_bullets"][0]["ball"]["ball_id"] == 104
    assert result["fired_bullets"][0]["subclass_fields"][
        "velocity_y"
    ] == -7.5


def test_raw_probe_accepts_formal_independent_score_fields(
    tmp_path: Path,
) -> None:
    result = validate_probe_payloads(
        _independent_score_fixture(tmp_path),
        expected_score=7950,
    )

    assert result["score"] == 7950
    assert result["displayed_score"] == 7940


def test_raw_probe_rejects_legacy_score_binding_mismatch(
    tmp_path: Path,
) -> None:
    path = _independent_score_fixture(tmp_path)
    probe = json.loads(path.read_text(encoding="ascii"))
    probe["score_binding"] = {
        "schema": "zuma-rl.pc-memory-score-binding",
        "version": 1,
        "mode": "active_board_read_only",
        "score": 7950,
        "displayed_score": 7940,
        "scan_value": 7950,
    }
    path.write_text(
        json.dumps(probe, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )

    with pytest.raises(
        PcMemoryEvidenceError,
        match="board_score_display_disagree",
    ):
        validate_probe_payloads(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("displayed_score_offset", 0xEF8),
        ("acceptance_rule", "score_fields_must_match"),
    ],
)
def test_raw_probe_rejects_tampered_independent_score_contract(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    path = _independent_score_fixture(tmp_path)
    probe = json.loads(path.read_text(encoding="ascii"))
    probe["score_binding"][field] = value
    path.write_text(
        json.dumps(probe, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )

    with pytest.raises(
        PcMemoryEvidenceError,
        match="score_binding_independent_contract_invalid",
    ):
        validate_probe_payloads(path)


def test_raw_probe_v2_binds_natural_win_diagnostics(
    tmp_path: Path,
) -> None:
    result = validate_probe_payloads(
        _fixture(tmp_path, active_board_version=2),
        expected_score=7950,
    )

    assert result["active_board_version"] == 2
    assert result["board_mode_flag_1064"] is False
    assert result["curve_plan_exhausted"] is False
    assert result["post_zuma_diagnostics"]["timer_remaining"] == 0
    assert result["post_zuma_diagnostics"]["ramp_408"] == 0.5
    assert result["curve_plans"][0]["count"] == 0
    assert result["curve_plans"][0]["add_plan_enabled"] is True


def test_raw_probe_v2_rejects_forged_curve_plan_semantics(
    tmp_path: Path,
) -> None:
    path = _fixture(tmp_path, active_board_version=2)
    probe = json.loads(path.read_text(encoding="ascii"))
    probe["active_board"]["curve_manager"]["curves"][0][
        "planned_balls"
    ]["count"] = 1
    path.write_text(
        json.dumps(probe, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )

    with pytest.raises(
        PcMemoryEvidenceError,
        match="curve_plan_semantics_mismatch:count",
    ):
        validate_probe_payloads(path)


def test_raw_probe_rejects_forged_semantic_json(
    tmp_path: Path,
) -> None:
    path = _fixture(tmp_path)
    probe = json.loads(path.read_text(encoding="ascii"))
    probe["active_board"]["curve_manager"]["curves"][0][
        "intrusive_lists"
    ][1]["records"][0]["ball"]["curve_distance"] = 999.0
    path.write_text(json.dumps(probe), encoding="ascii")

    with pytest.raises(
        PcMemoryEvidenceError,
        match="record_ball_semantics_mismatch:curve_distance",
    ):
        validate_probe_payloads(path)


def test_raw_probe_rejects_tampered_binary_payload(
    tmp_path: Path,
) -> None:
    path = _fixture(tmp_path)
    payload_path = tmp_path / "curve-list-05c.bin"
    payload = bytearray(payload_path.read_bytes())
    struct.pack_into("<i", payload, 0x14, 2)
    payload_path.write_bytes(payload)

    with pytest.raises(
        PcMemoryEvidenceError,
        match="probe_artifact_sha256_mismatch",
    ):
        validate_probe_payloads(path)


def test_raw_probe_requires_every_dependency_in_manifest_set(
    tmp_path: Path,
) -> None:
    path = _fixture(tmp_path)
    allowed = frozenset(
        candidate.resolve()
        for candidate in tmp_path.iterdir()
        if candidate.name != "fired.bin"
    )

    with pytest.raises(
        PcMemoryEvidenceError,
        match="artifact_not_declared_by_manifest",
    ):
        validate_probe_payloads(path, allowed_paths=allowed)


def test_curve_staging_blob_recomputes_mixed_ball_and_bullet_widths() -> None:
    ball = _ball_payload(
        vtable=BALL_VTABLE,
        size=BALL_OBJECT_SIZE,
        ball_id=1,
        color_id=2,
        distance=100.0,
        x=10.0,
        y=20.0,
    )
    bullet = _ball_payload(
        vtable=BULLET_VTABLE,
        size=BULLET_OBJECT_SIZE,
        ball_id=104,
        color_id=0,
        distance=2431.0,
        x=333.5,
        y=123.1,
        velocity=(-2.7, -7.5),
    )
    ball_record = _ball_record(
        ball,
        index=0,
        payload_address=0x10000000,
        bullet=False,
    )
    ball_record["payload_kind"] = "ball"
    bullet_record = _ball_record(
        bullet,
        index=1,
        payload_address=0x20000000,
        bullet=True,
    )
    bullet_record["artifact_offset"] = len(ball)
    bullet_record["payload_kind"] = "bullet"
    row = {
        "declared_count": 2,
        "traversed_count": 2,
        "payload_count": 2,
        "payload_vtable_frequencies": {
            f"0x{BALL_VTABLE:08x}": 1,
            f"0x{BULLET_VTABLE:08x}": 1,
        },
        "records": [ball_record, bullet_record],
    }

    decoded = _validate_curve_payload_records(
        ball + bullet,
        row,
        allow_bullets=True,
        maximum_count=10,
    )

    assert [item["object_kind"] for item in decoded] == [
        "ball",
        "bullet",
    ]
    assert decoded[1]["ball"]["ball_id"] == 104
    assert decoded[1]["subclass_fields"]["velocity_y"] == -7.5
    with pytest.raises(
        PcMemoryEvidenceError,
        match="record_vtable_mismatch",
    ):
        _validate_curve_payload_records(
            ball + bullet,
            row,
            allow_bullets=False,
            maximum_count=10,
        )


def test_memory_transition_contract_round_trips_exactly() -> None:
    contract = _contract()

    parsed = PcMemoryTransitionContract.from_dict(contract.to_dict())

    assert parsed == contract
    assert len(parsed.referenced_artifacts) == 6
    assert parsed.video_binding_excluded_bottom_rows == 1


def test_memory_transition_contract_rejects_alias_and_edge_overreach() -> None:
    with pytest.raises(
        PcMemoryEvidenceError,
        match="artifacts_not_distinct",
    ):
        _contract(
            video_binding_validation_artifact="memory.transition",
        )
    with pytest.raises(
        PcMemoryEvidenceError,
        match="video_edge_budget_invalid",
    ):
        _contract(maximum_excluded_edge_mismatches=801)


def test_memory_curve_geometry_binds_chain_to_claimed_level() -> None:
    class Curve:
        @staticmethod
        def point_at_waypoint(value: float) -> tuple[float, float]:
            return value, value * 2.0

    chain = [
        {
            "object_kind": "ball",
            "ball": {
                "curve_distance": 12.5,
                "position_x": 12.5,
                "position_y": 25.0,
            },
        },
        {
            "object_kind": "ball",
            "ball": {
                "curve_distance": 48.25,
                "position_x": 48.25,
                "position_y": 96.5,
            },
        },
    ]

    report = build_memory_curve_geometry_binding(
        case_id="case",
        level_id="Jungle2",
        hard=False,
        curve_index=0,
        curve=Curve(),
        phases=(("before", 10, chain), ("after", 20, chain)),
    )

    assert report["status"] == "PASS"
    assert report["scenario"]["level_id"] == "Jungle2"
    assert report["phases"][0]["maximum_position_error_px"] == 0.0


def test_memory_curve_geometry_rejects_wrong_level_curve() -> None:
    class WrongCurve:
        @staticmethod
        def point_at_waypoint(value: float) -> tuple[float, float]:
            return value + 10.0, value

    with pytest.raises(
        PcMemoryEvidenceError,
        match="memory_curve_geometry_mismatch",
    ):
        build_memory_curve_geometry_binding(
            case_id="case",
            level_id="Jungle1",
            hard=False,
            curve_index=0,
            curve=WrongCurve(),
            phases=(
                (
                    "before",
                    10,
                    (
                        {
                            "object_kind": "ball",
                            "ball": {
                                "curve_distance": 12.5,
                                "position_x": 12.5,
                                "position_y": 25.0,
                            },
                        },
                    ),
                ),
            ),
        )


def test_fired_gap_artifact_recomputes_node_semantics() -> None:
    bullet = bytearray(
        _ball_payload(
            vtable=BULLET_VTABLE,
            size=BULLET_OBJECT_SIZE,
            ball_id=40,
            color_id=3,
            distance=0.0,
            x=108.0,
            y=117.0,
        )
    )
    struct.pack_into("<I", bullet, 0x174, 0x1000)
    struct.pack_into("<I", bullet, 0x178, 1)
    struct.pack_into("<4i", bullet, 0x17C, 109, 0, 0, 0)
    gap_payload = struct.pack(
        "<IIiii",
        0x1000,
        0x1000,
        0,
        120,
        2,
    )
    record = {
        "gap_artifact_offset": 0,
        "gap_artifact_bytes": len(gap_payload),
        "gap_payload_sha256": sha256_bytes(gap_payload),
        "subclass_fields": {
            **decode_bullet_subclass(bytes(bullet)),
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
    decoded = [{"subclass_fields": decode_bullet_subclass(bytes(bullet))}]

    _validate_fired_gap_records(
        gap_payload,
        bytes(bullet),
        {"records": [record]},
        decoded,
    )

    assert decoded[0]["subclass_fields"]["gap_entry_count"] == 1
    assert decoded[0]["subclass_fields"]["gap_entries"] == [
        {
            "curve_index": 0,
            "gap_distance": 120,
            "boundary_ball_id": 2,
        }
    ]

    tampered = bytearray(gap_payload)
    tampered[12] ^= 1
    with pytest.raises(
        PcMemoryEvidenceError,
        match="fired_gap_payload_sha256_mismatch",
    ):
        _validate_fired_gap_records(
            bytes(tampered),
            bytes(bullet),
            {"records": [record]},
            decoded,
        )


def test_fired_gap_accepts_retail_debug_fill_in_unused_curve_slots() -> None:
    debug_fill = -1163005939  # Signed 0xBAADF00D.
    bullet = bytearray(
        _ball_payload(
            vtable=BULLET_VTABLE,
            size=BULLET_OBJECT_SIZE,
            ball_id=40,
            color_id=3,
            distance=0.0,
            x=108.0,
            y=117.0,
        )
    )
    struct.pack_into("<I", bullet, 0x174, 0x1000)
    struct.pack_into("<I", bullet, 0x178, 0)
    struct.pack_into(
        "<4i",
        bullet,
        0x17C,
        0,
        debug_fill,
        debug_fill,
        debug_fill,
    )
    record = {
        "gap_artifact_offset": 0,
        "gap_artifact_bytes": 0,
        "gap_payload_sha256": sha256_bytes(b""),
        "subclass_fields": {
            **decode_bullet_subclass(bytes(bullet)),
            "gap_list_sentinel_address": 0x1000,
            "gap_entry_count": 0,
            "curve_points": [0, debug_fill, debug_fill, debug_fill],
            "gap_entries": [],
        },
    }
    decoded = [{"subclass_fields": decode_bullet_subclass(bytes(bullet))}]

    _validate_fired_gap_records(
        b"",
        bytes(bullet),
        {"records": [record]},
        decoded,
    )

    assert decoded[0]["subclass_fields"]["curve_points"] == [
        0,
        debug_fill,
        debug_fill,
        debug_fill,
    ]


def test_fired_gap_rejects_debug_fill_in_referenced_curve_slot() -> None:
    debug_fill = -1163005939  # Signed 0xBAADF00D.
    bullet = bytearray(
        _ball_payload(
            vtable=BULLET_VTABLE,
            size=BULLET_OBJECT_SIZE,
            ball_id=40,
            color_id=3,
            distance=0.0,
            x=108.0,
            y=117.0,
        )
    )
    struct.pack_into("<I", bullet, 0x174, 0x1000)
    struct.pack_into("<I", bullet, 0x178, 1)
    struct.pack_into("<4i", bullet, 0x17C, debug_fill, 0, 0, 0)
    gap_payload = struct.pack("<IIiii", 0x1000, 0x1000, 0, 120, 2)
    record = {
        "gap_artifact_offset": 0,
        "gap_artifact_bytes": len(gap_payload),
        "gap_payload_sha256": sha256_bytes(gap_payload),
        "subclass_fields": {
            **decode_bullet_subclass(bytes(bullet)),
            "gap_list_sentinel_address": 0x1000,
            "gap_entry_count": 1,
            "curve_points": [debug_fill, 0, 0, 0],
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
    decoded = [{"subclass_fields": decode_bullet_subclass(bytes(bullet))}]

    with pytest.raises(
        PcMemoryEvidenceError,
        match="fired_gap_entry_semantics_mismatch",
    ):
        _validate_fired_gap_records(
            gap_payload,
            bytes(bullet),
            {"records": [record]},
            decoded,
        )
