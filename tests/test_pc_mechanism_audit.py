"""Tests for machine-derived, source-bound PC mechanism coverage."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from zuma_rl import pc_mechanism_audit
from zuma_rl.pc_memory_trajectory import (
    ACTIVE_CHAIN_LIST_OFFSET,
    INSERTION_STAGING_LIST_OFFSET,
    TrajectoryCurvePowerupState,
    TrajectoryEntity,
    TrajectoryFrame,
    TrajectoryFruitState,
    TrajectoryMTRandState,
    TrajectoryQRandState,
)
from zuma_rl.revenge_core import (
    FruitSpawnCalibration,
    MsvcCRTRandom,
    PopCapMTRandom,
    _BalancedColorChooser,
)


def _frames(start: int, end: int) -> tuple[TrajectoryFrame, ...]:
    return tuple(
        TrajectoryFrame(
            update=update,
            score=0,
            displayed_score=0,
            score_target=100,
            current_ball_id=None,
            current_color_id=None,
            next_ball_id=None,
            next_color_id=None,
            list_counts=(),
            entities=(),
        )
        for update in range(start, end + 1)
    )


def _entity(
    ball_id: int,
    color_id: int,
    *,
    zone: str,
    index: int,
    curve_distance: float,
    x: float,
    y: float = 100.0,
    kind: str = "ball",
) -> TrajectoryEntity:
    return TrajectoryEntity(
        ball_id=ball_id,
        color_id=color_id,
        object_kind=kind,
        zone=zone,
        index=index,
        curve_distance=curve_distance,
        position_x=x,
        position_y=y,
        radius=18.0,
        velocity_x=0.0 if kind == "bullet" else None,
        velocity_y=0.0 if kind == "bullet" else None,
        merge_progress=0.5 if kind == "bullet" else None,
        merge_speed=0.025 if kind == "bullet" else None,
        fired=kind == "bullet" and zone == "fired",
        exploding=False,
        should_remove=False,
        suck_count=0,
    )


def _natural_loss_frames() -> tuple[TrajectoryFrame, ...]:
    chain_states = (
        ((1, 16.0, 0), (2, 19.875, 0)),
        ((1, 16.125, 0), (2, 20.0, 0)),
        ((1, 16.125, 1),),
        ((1, 16.125, 2),),
        ((1, 16.125, 3),),
        ((1, 16.125, 4),),
        ((1, 17.125, 5),),
        ((1, 18.125, 6),),
        ((1, 19.125, 7),),
        ((1, 20.125, 0),),
        (),
    )
    active_zone = f"curve:0:list:{ACTIVE_CHAIN_LIST_OFFSET:03x}"
    result: list[TrajectoryFrame] = []
    for update, chain in enumerate(chain_states):
        before_reset = update == 0
        current_id, current_color = (
            (10, 0) if before_reset else (12, 3)
        )
        next_id, next_color = (
            (11, 1) if before_reset else (13, 3)
        )
        entities = [
            *(
                replace(
                    _entity(
                        ball_id,
                        ball_id % 4,
                        zone=active_zone,
                        index=index,
                        curve_distance=distance,
                        x=distance,
                    ),
                    update_count=5,
                    suck_count=suck_count,
                )
                for index, (ball_id, distance, suck_count) in enumerate(chain)
            ),
            _entity(
                current_id,
                current_color,
                zone="shooter_current",
                index=0,
                curve_distance=0.0,
                x=400.0,
                kind="bullet",
            ),
            _entity(
                next_id,
                next_color,
                zone="shooter_next",
                index=0,
                curve_distance=0.0,
                x=400.0,
                kind="bullet",
            ),
        ]
        result.append(
            TrajectoryFrame(
                update=update,
                score=80,
                displayed_score=70,
                score_target=100,
                current_ball_id=current_id,
                current_color_id=current_color,
                next_ball_id=next_id,
                next_color_id=next_color,
                list_counts=(
                    (0, ACTIVE_CHAIN_LIST_OFFSET, len(chain)),
                    (0, 0x68, 1),
                ),
                entities=tuple(entities),
                native_game_time=100 if update == 0 else update - 1,
                board_runtime_flag_157=update < 2,
                board_runtime_i32_f54=0 if update < 2 else -(update - 1),
            )
        )
    return tuple(result)


def test_natural_loss_requires_exact_native_skull_suction_recurrence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active_zone = f"curve:0:list:{ACTIVE_CHAIN_LIST_OFFSET:03x}"
    monkeypatch.setattr(
        pc_mechanism_audit,
        "_original_curve",
        lambda **kwargs: SimpleNamespace(end_waypoint=20),
    )
    features, proofs = pc_mechanism_audit.derive_native_mechanism_features(
        _natural_loss_frames(),
        original_root=Path("."),
        level_id="Jungle2",
    )

    assert "natural_loss" in features
    proof = next(row for row in proofs if row["feature"] == "natural_loss")
    assert proof["trigger_update"] == 2
    assert proof["empty_update"] == 10
    assert proof["removal_updates"] == {"2": [2], "10": [1]}

    tampered = list(_natural_loss_frames())
    tampered[3] = replace(
        tampered[3],
        entities=tuple(
            replace(entity, suck_count=3)
            if entity.zone == active_zone
            else entity
            for entity in tampered[3].entities
        ),
    )
    tampered_features, _ = (
        pc_mechanism_audit.derive_native_mechanism_features(
            tuple(tampered),
            original_root=Path("."),
            level_id="Jungle2",
        )
    )
    assert "natural_loss" not in tampered_features


def test_probe_trajectory_binding_accepts_matching_freeze_update(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = tmp_path / "index.json"
    index.write_text("{}\n", encoding="ascii")
    frames = _frames(11200, 11201)
    monkeypatch.setattr(
        pc_mechanism_audit,
        "load_memory_trajectory",
        lambda path: frames,
    )
    probe = {
        "trajectory": {
            "artifact": index.name,
            "artifact_sha256": pc_mechanism_audit._sha256_path(index),
            "end_update": 11201,
            "freeze_update": 11200,
            "schema": pc_mechanism_audit.TRAJECTORY_SCHEMA,
            "start_update": 11200,
            "tick_count": 2,
            "version": pc_mechanism_audit.TRAJECTORY_VERSION,
        }
    }

    path, loaded, binding = pc_mechanism_audit._trajectory_from_probe(
        probe,
        probe_path=tmp_path / "memory-probe.json",
    )

    assert path == index.resolve()
    assert loaded == frames
    assert binding["freeze_update"] == 11200


def test_probe_trajectory_binding_rejects_mismatched_freeze_update(
    tmp_path: Path,
) -> None:
    index = tmp_path / "index.json"
    index.write_text("{}\n", encoding="ascii")
    probe = {
        "trajectory": {
            "artifact": index.name,
            "artifact_sha256": pc_mechanism_audit._sha256_path(index),
            "end_update": 11201,
            "freeze_update": 11199,
            "schema": pc_mechanism_audit.TRAJECTORY_SCHEMA,
            "start_update": 11200,
            "tick_count": 2,
            "version": pc_mechanism_audit.TRAJECTORY_VERSION,
        }
    }

    with pytest.raises(
        pc_mechanism_audit.PcMechanismAuditError,
        match="mechanism_audit_trajectory_freeze_mismatch",
    ):
        pc_mechanism_audit._trajectory_from_probe(
            probe,
            probe_path=tmp_path / "memory-probe.json",
        )


def test_formal_projected_bullet_has_nonfired_container_semantics() -> None:
    row = {
        "kind": "bullet",
        "ball_id": 41,
        "color_id": 2,
        "powerup_previous_type": 14,
        "powerup_primary_type": 14,
        "powerup_secondary_type": 14,
        "flags_b4_c2_hex": "00" * 15,
        "radius": 18.0,
        "curve_distance": 0.0,
        "position_x": 400.0,
        "position_y": 300.0,
    }

    entity = pc_mechanism_audit._formal_entity(
        row,
        zone="shooter_current",
        index=0,
    )

    assert entity is not None
    assert entity.object_kind == "bullet"
    assert entity.fired is False


class _HorizontalCurve:
    def point_at_waypoint(self, waypoint: float) -> np.ndarray:
        return np.asarray((waypoint, 100.0), dtype=np.float32)

    def perpendicular_at_waypoint(self, waypoint: float) -> np.ndarray:
        del waypoint
        return np.asarray((0.0, -1.0), dtype=np.float32)

    def is_in_tunnel_at_waypoint(self, waypoint: float) -> np.bool_:
        del waypoint
        return np.bool_(False)


class _TunnelCurve(_HorizontalCurve):
    def is_in_tunnel_at_waypoint(self, waypoint: float) -> np.bool_:
        return np.bool_(waypoint >= 118.0)


class _GapCurve(_HorizontalCurve):
    end_waypoint = 300


@pytest.mark.parametrize(
    ("projectile_x", "active_rows", "after_ids", "expected_feature"),
    (
        (
            103.0,
            ((1, 100.0), (2, 140.0)),
            (1, 41, 2),
            "front_insertion",
        ),
        (
            97.0,
            ((2, 60.0), (1, 100.0)),
            (2, 41, 1),
            "back_insertion",
        ),
    ),
)
def test_direction_uses_curve_cross_product_and_committed_list_order(
    monkeypatch: pytest.MonkeyPatch,
    projectile_x: float,
    active_rows: tuple[tuple[int, float], ...],
    after_ids: tuple[int, ...],
    expected_feature: str,
) -> None:
    monkeypatch.setattr(
        pc_mechanism_audit,
        "_original_curve",
        lambda **kwargs: _HorizontalCurve(),
    )
    active_zone = f"curve:0:list:{ACTIVE_CHAIN_LIST_OFFSET:03x}"
    staging_zone = (
        f"curve:0:list:{INSERTION_STAGING_LIST_OFFSET:03x}"
    )
    active = tuple(
        _entity(
            ball_id,
            ball_id % 4,
            zone=active_zone,
            index=index,
            curve_distance=waypoint,
            x=waypoint,
        )
        for index, (ball_id, waypoint) in enumerate(active_rows)
    )
    fired = _entity(
        40,
        3,
        zone="fired",
        index=0,
        curve_distance=projectile_x,
        x=projectile_x,
        kind="bullet",
    )
    staged = _entity(
        40,
        3,
        zone=staging_zone,
        index=0,
        curve_distance=projectile_x,
        x=projectile_x,
        kind="bullet",
    )
    after_lookup = {
        entity.ball_id: entity for entity in active
    }
    inserted = _entity(
        41,
        3,
        zone=active_zone,
        index=after_ids.index(41),
        curve_distance=projectile_x,
        x=projectile_x,
    )
    after_entities: list[TrajectoryEntity] = []
    for index, ball_id in enumerate(after_ids):
        source = inserted if ball_id == 41 else after_lookup[ball_id]
        after_entities.append(
            _entity(
                source.ball_id,
                source.color_id,
                zone=active_zone,
                index=index,
                curve_distance=source.curve_distance,
                x=source.position_x,
            )
        )
    frames = (
        TrajectoryFrame(
            update=10,
            score=0,
            displayed_score=0,
            score_target=100,
            current_ball_id=None,
            current_color_id=None,
            next_ball_id=None,
            next_color_id=None,
            list_counts=((0, ACTIVE_CHAIN_LIST_OFFSET, 2),),
            entities=(*active, fired),
        ),
        TrajectoryFrame(
            update=11,
            score=0,
            displayed_score=0,
            score_target=100,
            current_ball_id=None,
            current_color_id=None,
            next_ball_id=None,
            next_color_id=None,
            list_counts=(
                (0, ACTIVE_CHAIN_LIST_OFFSET, 2),
                (0, INSERTION_STAGING_LIST_OFFSET, 1),
            ),
            entities=(*active, staged),
        ),
        TrajectoryFrame(
            update=12,
            score=0,
            displayed_score=0,
            score_target=100,
            current_ball_id=None,
            current_color_id=None,
            next_ball_id=None,
            next_color_id=None,
            list_counts=((0, ACTIVE_CHAIN_LIST_OFFSET, 3),),
            entities=tuple(after_entities),
        ),
    )

    features, proofs = (
        pc_mechanism_audit.derive_native_mechanism_features(
            frames,
            original_root=Path("/unused"),
            level_id="Jungle2",
        )
    )

    assert expected_feature in features
    proof = next(
        row for row in proofs if row["feature"] == expected_feature
    )
    assert proof["hit_ball_id"] == 1
    assert proof["inserted_ball_id"] == 41
    assert proof["hit_side_tunnel"] is False


def test_formal_window_start_staging_can_prove_direction_without_claiming_hit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pc_mechanism_audit,
        "_original_curve",
        lambda **kwargs: _HorizontalCurve(),
    )
    active_zone = f"curve:0:list:{ACTIVE_CHAIN_LIST_OFFSET:03x}"
    staging_zone = (
        f"curve:0:list:{INSERTION_STAGING_LIST_OFFSET:03x}"
    )
    first_ball = _entity(
        1,
        0,
        zone=active_zone,
        index=0,
        curve_distance=100.0,
        x=100.0,
    )
    second_ball = _entity(
        2,
        1,
        zone=active_zone,
        index=1,
        curve_distance=140.0,
        x=140.0,
    )
    staged = _entity(
        40,
        3,
        zone=staging_zone,
        index=0,
        curve_distance=103.0,
        x=103.0,
        kind="bullet",
    )
    inserted = _entity(
        41,
        3,
        zone=active_zone,
        index=1,
        curve_distance=103.0,
        x=103.0,
    )
    frames = (
        TrajectoryFrame(
            update=11,
            score=0,
            displayed_score=0,
            score_target=100,
            current_ball_id=None,
            current_color_id=None,
            next_ball_id=None,
            next_color_id=None,
            list_counts=(
                (0, ACTIVE_CHAIN_LIST_OFFSET, 2),
                (0, INSERTION_STAGING_LIST_OFFSET, 1),
            ),
            entities=(first_ball, second_ball, staged),
        ),
        TrajectoryFrame(
            update=12,
            score=0,
            displayed_score=0,
            score_target=100,
            current_ball_id=None,
            current_color_id=None,
            next_ball_id=None,
            next_color_id=None,
            list_counts=((0, ACTIVE_CHAIN_LIST_OFFSET, 3),),
            entities=(first_ball, inserted, replace(second_ball, index=2)),
        ),
    )

    default_features, _ = (
        pc_mechanism_audit.derive_native_mechanism_features(
            frames,
            original_root=Path("/unused"),
            level_id="Jungle2",
        )
    )
    features, proofs = pc_mechanism_audit.derive_native_mechanism_features(
        frames,
        original_root=Path("/unused"),
        level_id="Jungle2",
        allow_initial_staging_observation=True,
    )

    assert "front_insertion" not in default_features
    assert "front_insertion" in features
    assert "projectile_collision" not in features
    proof = next(row for row in proofs if row["feature"] == "front_insertion")
    assert proof["collision_observation_mode"] == (
        "formal_exact_step_window_start_staging"
    )


def _tunnel_frames() -> tuple[TrajectoryFrame, TrajectoryFrame]:
    active_zone = f"curve:0:list:{ACTIVE_CHAIN_LIST_OFFSET:03x}"
    ball = _entity(
        1,
        2,
        zone=active_zone,
        index=0,
        curve_distance=100.0,
        x=100.0,
    )
    projectile = replace(
        _entity(
            40,
            3,
            zone="fired",
            index=0,
            curve_distance=0.0,
            x=103.0,
            kind="bullet",
        ),
        velocity_x=5.0,
        velocity_y=0.0,
        fired=False,
    )
    retained = replace(projectile, position_x=108.0)
    common = {
        "score": 0,
        "displayed_score": 0,
        "score_target": 100,
        "current_ball_id": None,
        "current_color_id": None,
        "next_ball_id": None,
        "next_color_id": None,
        "list_counts": ((0, ACTIVE_CHAIN_LIST_OFFSET, 1),),
    }
    return (
        TrajectoryFrame(
            update=20,
            entities=(ball, projectile),
            **common,
        ),
        TrajectoryFrame(
            update=21,
            entities=(ball, retained),
            **common,
        ),
    )


def test_tunnel_collision_requires_exact_curve_side_and_free_flight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pc_mechanism_audit,
        "_original_curve",
        lambda **kwargs: _TunnelCurve(),
    )

    features, proofs = (
        pc_mechanism_audit.derive_native_mechanism_features(
            _tunnel_frames(),
            original_root=Path("/unused"),
            level_id="Jungle2",
        )
    )

    assert features == ("tunnel_collision",)
    proof = proofs[0]
    assert proof["projectile_remained_free"] is True
    assert proof["non_tunnel_overlap_count"] == 0
    assert proof["tunnel_overlaps"][0]["tunnel_waypoint"] == 118


def test_physical_overlap_without_tunnel_is_not_tunnel_collision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pc_mechanism_audit,
        "_original_curve",
        lambda **kwargs: _HorizontalCurve(),
    )

    features, _ = (
        pc_mechanism_audit.derive_native_mechanism_features(
            _tunnel_frames(),
            original_root=Path("/unused"),
            level_id="Jungle2",
        )
    )

    assert "tunnel_collision" not in features


def test_tunnel_collision_ignores_exploding_chain_balls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pc_mechanism_audit,
        "_original_curve",
        lambda **kwargs: _TunnelCurve(),
    )
    before, after = _tunnel_frames()
    active_zone = f"curve:0:list:{ACTIVE_CHAIN_LIST_OFFSET:03x}"
    exploding = replace(
        _entity(
            2,
            1,
            zone=active_zone,
            index=1,
            curve_distance=300.0,
            x=300.0,
        ),
        exploding=True,
    )
    frames = (
        replace(
            before,
            entities=(*before.entities, exploding),
            list_counts=((0, ACTIVE_CHAIN_LIST_OFFSET, 2),),
        ),
        replace(
            after,
            entities=(*after.entities, exploding),
            list_counts=((0, ACTIVE_CHAIN_LIST_OFFSET, 2),),
        ),
    )

    features, _ = pc_mechanism_audit.derive_native_mechanism_features(
        frames,
        original_root=Path("/unused"),
        level_id="Jungle2",
    )

    assert "tunnel_collision" in features


def _gap_shot_frames(
    *,
    gap_distance: int = 120,
) -> tuple[TrajectoryFrame, TrajectoryFrame]:
    active_zone = f"curve:0:list:{ACTIVE_CHAIN_LIST_OFFSET:03x}"
    chain = (
        _entity(
            1,
            1,
            zone=active_zone,
            index=0,
            curve_distance=50.0,
            x=50.0,
        ),
        _entity(
            2,
            2,
            zone=active_zone,
            index=1,
            curve_distance=170.0,
            x=170.0,
        ),
    )
    projectile = replace(
        _entity(
            40,
            3,
            zone="fired",
            index=0,
            curve_distance=0.0,
            x=108.0,
            y=117.0,
            kind="bullet",
        ),
        velocity_x=1.0,
        velocity_y=0.0,
        fired=False,
        gap_list_sentinel_address=0x12340000,
        gap_entry_count=0,
        curve_points=(0, 0, 0, 0),
        gap_entries=(),
    )
    retained = replace(
        projectile,
        position_x=109.0,
        gap_entry_count=1,
        curve_points=(109, 0, 0, 0),
        gap_entries=((0, gap_distance, 2),),
    )
    common = {
        "score": 0,
        "displayed_score": 0,
        "score_target": 100,
        "current_ball_id": None,
        "current_color_id": None,
        "next_ball_id": None,
        "next_color_id": None,
        "list_counts": ((0, ACTIVE_CHAIN_LIST_OFFSET, 2),),
    }
    return (
        TrajectoryFrame(
            update=30,
            entities=(*chain, projectile),
            **common,
        ),
        TrajectoryFrame(
            update=31,
            entities=(*chain, retained),
            **common,
        ),
    )


def test_gap_shot_replays_native_curve_sample_and_raw_list_growth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pc_mechanism_audit,
        "_original_curve",
        lambda **kwargs: _GapCurve(),
    )

    features, proofs = (
        pc_mechanism_audit.derive_native_mechanism_features(
            _gap_shot_frames(),
            original_root=Path("/unused"),
            level_id="Jungle2",
        )
    )

    assert "gap_shot" in features
    proof = next(row for row in proofs if row["feature"] == "gap_shot")
    assert proof["new_gap_entry"] == {
        "curve_index": 0,
        "gap_distance": 120,
        "boundary_ball_id": 2,
    }
    assert proof["curve_point_after"] == 109
    assert len(proof["simulations"]) == 2


def test_gap_shot_rejects_gap_distance_not_derived_from_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pc_mechanism_audit,
        "_original_curve",
        lambda **kwargs: _GapCurve(),
    )

    features, _ = pc_mechanism_audit.derive_native_mechanism_features(
        _gap_shot_frames(gap_distance=119),
        original_root=Path("/unused"),
        level_id="Jungle2",
    )

    assert "gap_shot" not in features


def _qrand(update_count: int = 7) -> TrajectoryQRandState:
    return TrajectoryQRandState(
        update_count=update_count,
        selected_index=2,
        weights=(0.25, 0.25, 0.25, 0.25, 0.0, 0.0),
        sways=(0.1, 0.2, 0.3, 0.4, 0.0, 0.0),
        last_hit=(1, 2, 3, 4, 0, 0),
        previous_hit=(0, 1, 2, 3, 0, 0),
    )


def test_swap_requires_exact_identity_exchange_without_rng_draw() -> None:
    current = _entity(
        10,
        1,
        zone="shooter_current",
        index=0,
        curve_distance=0.0,
        x=400.0,
        y=320.0,
        kind="bullet",
    )
    next_ball = _entity(
        11,
        2,
        zone="shooter_next",
        index=1,
        curve_distance=0.0,
        x=0.0,
        y=0.0,
        kind="bullet",
    )
    after_current = _entity(
        11,
        2,
        zone="shooter_current",
        index=0,
        curve_distance=0.0,
        x=400.0,
        y=320.0,
        kind="bullet",
    )
    after_next = _entity(
        10,
        1,
        zone="shooter_next",
        index=1,
        curve_distance=0.0,
        x=0.0,
        y=0.0,
        kind="bullet",
    )
    frames = (
        TrajectoryFrame(
            update=20,
            score=50,
            displayed_score=50,
            score_target=100,
            current_ball_id=10,
            current_color_id=1,
            next_ball_id=11,
            next_color_id=2,
            list_counts=(),
            entities=(current, next_ball),
            qrand=_qrand(),
            thread_crt_rand_state=123,
            native_game_time=20,
        ),
        TrajectoryFrame(
            update=21,
            score=50,
            displayed_score=50,
            score_target=100,
            current_ball_id=11,
            current_color_id=2,
            next_ball_id=10,
            next_color_id=1,
            list_counts=(),
            entities=(after_current, after_next),
            qrand=_qrand(),
            thread_crt_rand_state=123,
            native_game_time=21,
        ),
    )

    features, proofs = (
        pc_mechanism_audit.derive_native_mechanism_features(frames)
    )

    assert features == ("swap",)
    assert proofs[0]["qrand_unchanged"] is True


def test_swap_with_qrand_change_is_not_authorized() -> None:
    current = _entity(
        10,
        1,
        zone="shooter_current",
        index=0,
        curve_distance=0.0,
        x=400.0,
        y=320.0,
        kind="bullet",
    )
    next_ball = _entity(
        11,
        2,
        zone="shooter_next",
        index=1,
        curve_distance=0.0,
        x=0.0,
        y=0.0,
        kind="bullet",
    )
    frames = (
        TrajectoryFrame(
            update=20,
            score=0,
            displayed_score=0,
            score_target=100,
            current_ball_id=10,
            current_color_id=1,
            next_ball_id=11,
            next_color_id=2,
            list_counts=(),
            entities=(current, next_ball),
            qrand=_qrand(),
            thread_crt_rand_state=123,
        ),
        TrajectoryFrame(
            update=21,
            score=0,
            displayed_score=0,
            score_target=100,
            current_ball_id=11,
            current_color_id=2,
            next_ball_id=10,
            next_color_id=1,
            list_counts=(),
            entities=(
                _entity(
                    11,
                    2,
                    zone="shooter_current",
                    index=0,
                    curve_distance=0.0,
                    x=400.0,
                    y=320.0,
                    kind="bullet",
                ),
                _entity(
                    10,
                    1,
                    zone="shooter_next",
                    index=1,
                    curve_distance=0.0,
                    x=0.0,
                    y=0.0,
                    kind="bullet",
                ),
            ),
            qrand=_qrand(update_count=8),
            thread_crt_rand_state=123,
        ),
    )

    features, _ = pc_mechanism_audit.derive_native_mechanism_features(
        frames
    )

    assert "swap" not in features


def _qrand_shot_frames() -> tuple[TrajectoryFrame, TrajectoryFrame]:
    active_zone = f"curve:0:list:{ACTIVE_CHAIN_LIST_OFFSET:03x}"
    active = tuple(
        _entity(
            30 + color,
            color,
            zone=active_zone,
            index=color,
            curve_distance=100.0 + 36.0 * color,
            x=100.0 + 36.0 * color,
        )
        for color in range(4)
    )
    before_qrand = _qrand()
    crt = MsvcCRTRandom(1)
    crt.state = 0x12345678
    chooser = _BalancedColorChooser(crt, size=6)
    chooser.load_state(
        update_count=before_qrand.update_count,
        selected_index=before_qrand.selected_index,
        weights=before_qrand.weights,
        sways=before_qrand.sways,
        last_hit=before_qrand.last_hit,
        previous_hit=before_qrand.previous_hit,
    )
    selected = chooser.choose((0, 1, 2, 3, 1))
    assert selected is not None
    after_qrand = TrajectoryQRandState(
        update_count=chooser.update_count,
        selected_index=chooser.selected_index,
        weights=tuple(float(value) for value in chooser.weights),
        sways=tuple(float(value) for value in chooser.sways),
        last_hit=tuple(int(value) for value in chooser.last_hit),
        previous_hit=tuple(
            int(value) for value in chooser.previous_hit
        ),
    )
    current = _entity(
        10,
        1,
        zone="shooter_current",
        index=0,
        curve_distance=0.0,
        x=400.0,
        y=320.0,
        kind="bullet",
    )
    next_ball = _entity(
        11,
        2,
        zone="shooter_next",
        index=1,
        curve_distance=0.0,
        x=0.0,
        y=0.0,
        kind="bullet",
    )
    fired = replace(
        current,
        zone="fired",
        position_x=410.0,
        fired=False,
    )
    promoted = replace(
        next_ball,
        zone="shooter_current",
        index=0,
        position_x=400.0,
        position_y=320.0,
    )
    replacement = _entity(
        12,
        selected,
        zone="shooter_next",
        index=1,
        curve_distance=0.0,
        x=0.0,
        y=0.0,
        kind="bullet",
    )
    common = {
        "score": 100,
        "displayed_score": 100,
        "score_target": 1000,
        "list_counts": ((0, ACTIVE_CHAIN_LIST_OFFSET, len(active)),),
    }
    return (
        TrajectoryFrame(
            update=50,
            current_ball_id=10,
            current_color_id=1,
            next_ball_id=11,
            next_color_id=2,
            entities=(*active, current, next_ball),
            qrand=before_qrand,
            thread_crt_rand_state=0x12345678,
            **common,
        ),
        TrajectoryFrame(
            update=51,
            current_ball_id=11,
            current_color_id=2,
            next_ball_id=12,
            next_color_id=selected,
            entities=(*active, fired, promoted, replacement),
            qrand=after_qrand,
            thread_crt_rand_state=crt.state,
            **common,
        ),
    )


def test_shooter_rng_replays_exact_qrand_and_crt_transition() -> None:
    features, proofs = (
        pc_mechanism_audit.derive_native_mechanism_features(
            _qrand_shot_frames()
        )
    )

    assert features == ("rng_pending", "shot_release")
    rng = next(row for row in proofs if row["feature"] == "rng_pending")
    assert rng["qrand_state_exact"] is True
    assert rng["thread_crt_state_exact"] is True
    assert rng["new_next_ball_id"] == 12


def test_shooter_rng_rejects_tampered_crt_state() -> None:
    frames = _qrand_shot_frames()
    tampered = (
        frames[0],
        replace(
            frames[1],
            thread_crt_rand_state=(
                int(frames[1].thread_crt_rand_state) + 1
            )
            & 0xFFFFFFFF,
        ),
    )

    features, _ = (
        pc_mechanism_audit.derive_native_mechanism_features(tampered)
    )

    assert "rng_pending" not in features
    assert "shot_release" in features


def _rng_rejection_frames() -> tuple[
    tuple[TrajectoryFrame, ...],
    int,
    tuple[int, ...],
]:
    previous_color = 2
    selected_seed = 0
    selected_outputs: tuple[int, ...] = ()
    for seed in range(1, 10_000):
        random = PopCapMTRandom(seed)
        outputs = tuple(random.next_u31() for _ in range(10))
        event = outputs[2:8]
        if (
            event[1] % 100 > 43
            and event[2] % 4 == previous_color
            and event[3] % 4 != previous_color
        ):
            selected_seed = seed
            selected_outputs = outputs
            break
    assert selected_seed > 0
    random = PopCapMTRandom(selected_seed)
    states = [
        TrajectoryMTRandState(index=random.index, words=random.words)
    ]
    for count in (2, 6, 2):
        for _ in range(count):
            random.next_u31()
        states.append(
            TrajectoryMTRandState(
                index=random.index,
                words=random.words,
            )
        )

    active_zone = f"curve:0:list:{ACTIVE_CHAIN_LIST_OFFSET:03x}"
    active = tuple(
        _entity(
            50 + index,
            color,
            zone=active_zone,
            index=index,
            curve_distance=100.0 + 36.0 * index,
            x=100.0 + 36.0 * index,
        )
        for index, color in enumerate((2, 0, 1, 3))
    )
    accepted_color = selected_outputs[5] % 4
    pending = _entity(
        99,
        accepted_color,
        zone="curve:0:list:068",
        index=0,
        curve_distance=0.0,
        x=0.0,
    )
    common = {
        "score": 100,
        "displayed_score": 100,
        "score_target": 1000,
        "current_ball_id": 10,
        "current_color_id": 1,
        "next_ball_id": 11,
        "next_color_id": 3,
        "qrand": _qrand(),
        "thread_crt_rand_state": 123,
    }
    frames: list[TrajectoryFrame] = []
    for index, update in enumerate(range(100, 104)):
        has_pending = index >= 2
        entities = (*active, pending) if has_pending else active
        list_counts = ((0, ACTIVE_CHAIN_LIST_OFFSET, len(active)),)
        if has_pending:
            list_counts += ((0, 0x68, 1),)
        frames.append(
            TrajectoryFrame(
                update=update,
                list_counts=list_counts,
                entities=tuple(entities),
                global_mtrand=states[index],
                native_game_time=1000 + index,
                curve_powerups=(
                    _spawn_state(native_time=1000 + index),
                ),
                **common,
            )
        )
    return tuple(frames), accepted_color, selected_outputs[2:8]


def _rng_rejection_curve() -> SimpleNamespace:
    return SimpleNamespace(
        parameters=SimpleNamespace(
            colors=4,
            ball_repeat_chance=43,
            max_clump_size=6,
            max_single=3,
            powerup_records=(),
        )
    )


def test_pending_rng_proves_rejected_candidate_and_exact_mt_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames, accepted_color, event_outputs = _rng_rejection_frames()
    monkeypatch.setattr(
        pc_mechanism_audit,
        "_original_curve",
        lambda **kwargs: _rng_rejection_curve(),
    )

    features, proofs = (
        pc_mechanism_audit.derive_native_mechanism_features(
            frames,
            original_root=Path("/unused"),
            level_id="Jungle2",
        )
    )

    assert features == ("rng_rejection",)
    proof = proofs[0]
    assert proof["generated_color_id"] == accepted_color
    assert proof["rejected_candidate_colors"] == [2]
    assert proof["accepted_candidate_color"] == accepted_color
    assert proof["mtrand_outputs"] == list(event_outputs)
    assert proof["powerup_rng_prechance_ineligible"] == (
        "before_initial_delay"
    )


def test_pending_rng_proves_zero_ambient_rejection_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    template, _, _ = _rng_rejection_frames()
    previous_color = 2
    selected_seed = 0
    selected_outputs: tuple[int, ...] = ()
    for seed in range(1, 10_000):
        random = PopCapMTRandom(seed)
        outputs = tuple(random.next_u31() for _ in range(4))
        if (
            outputs[0] % 100 > 43
            and outputs[1] % 4 == previous_color
            and outputs[2] % 4 != previous_color
        ):
            selected_seed = seed
            selected_outputs = outputs
            break
    assert selected_seed > 0

    random = PopCapMTRandom(selected_seed)
    before_state = TrajectoryMTRandState(
        index=random.index,
        words=random.words,
    )
    for _ in range(4):
        random.next_u31()
    after_state = TrajectoryMTRandState(
        index=random.index,
        words=random.words,
    )
    states = (before_state, before_state, after_state, after_state)
    accepted_color = selected_outputs[2] % 4
    frames = tuple(
        replace(
            frame,
            global_mtrand=states[index],
            entities=tuple(
                replace(entity, color_id=accepted_color)
                if entity.ball_id == 99
                else entity
                for entity in frame.entities
            ),
        )
        for index, frame in enumerate(template)
    )
    frames = (
        replace(frames[0], entities=frames[0].entities[:-1]),
        *frames[1:],
    )
    monkeypatch.setattr(
        pc_mechanism_audit,
        "_original_curve",
        lambda **kwargs: _rng_rejection_curve(),
    )

    features, proofs = (
        pc_mechanism_audit.derive_native_mechanism_features(
            frames,
            original_root=Path("/unused"),
            level_id="Jungle2",
        )
    )

    assert features == ("rng_rejection",)
    proof = proofs[0]
    assert proof["neighbouring_ambient_draw_count"] == 0
    assert proof["ambient_before_output"] is None
    assert proof["ambient_after_output"] is None
    assert proof["repeat_roll_output"] == selected_outputs[0]
    assert proof["rejected_candidate_outputs"] == [selected_outputs[1]]
    assert proof["accepted_candidate_output"] == selected_outputs[2]
    assert proof["ball_visual_frame_output"] == selected_outputs[3]


def test_pending_rng_rejects_wrong_generated_color(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames, _, _ = _rng_rejection_frames()
    tampered = list(frames)
    for index in (2, 3):
        tampered[index] = replace(
            tampered[index],
            entities=tuple(
                replace(entity, color_id=2)
                if entity.ball_id == 99
                else entity
                for entity in tampered[index].entities
            ),
        )
    monkeypatch.setattr(
        pc_mechanism_audit,
        "_original_curve",
        lambda **kwargs: _rng_rejection_curve(),
    )

    features, _ = (
        pc_mechanism_audit.derive_native_mechanism_features(
            tuple(tampered),
            original_root=Path("/unused"),
            level_id="Jungle2",
        )
    )

    assert "rng_rejection" not in features


def _powerup_ball(
    ball_id: int,
    color_id: int,
    index: int,
) -> TrajectoryEntity:
    return replace(
        _entity(
            ball_id,
            color_id,
            zone=f"curve:0:list:{ACTIVE_CHAIN_LIST_OFFSET:03x}",
            index=index,
            curve_distance=100.0 + 36.0 * index,
            x=100.0 + 36.0 * index,
        ),
        contact_next=index < 7,
        backwards_count=0,
        backwards_speed=0.5,
        combo_count=0,
        combo_score=0,
        powerup_previous_type=14,
        powerup_primary_type=14,
        powerup_secondary_type=14,
        powerup_previous_ticks=0,
        powerup_lifetime_ticks=0,
        powerup_transition_ticks=0,
        powerup_visual_scale=1.0,
        powerup_visual_step=0.0,
        powerup_visual_index=-1,
    )


def _spawn_state(
    *,
    native_time: int,
    selected_type: int | None = None,
    selected_color: int | None = None,
) -> TrajectoryCurvePowerupState:
    last_spawns = [-1000] * 14
    spawn_counts = [0] * 14
    active_colors = [0] * 6
    last_any = 1000
    if selected_type is not None and selected_color is not None:
        last_any = native_time
        last_spawns[selected_type] = native_time
        spawn_counts[selected_type] = 1
        active_colors[selected_color] = 1
    return TrajectoryCurvePowerupState(
        curve_index=0,
        last_any_spawn_time=last_any,
        last_spawn_times=tuple(last_spawns),
        cooldown_times=(-1000,) * 14,
        spawn_counts=tuple(spawn_counts),
        field_124_by_type=(0,) * 14,
        active_color_counts=tuple(active_colors),
        reverse_speed=0.5,
        slow_ticks=0,
        reverse_ticks=0,
        last_powerup_waypoint=0,
        powerup_triggered=False,
    )


def _spawn_frames() -> tuple[
    tuple[TrajectoryFrame, TrajectoryFrame],
    int,
    int,
    tuple[int, ...],
]:
    random = PopCapMTRandom(0xC0FFEE)
    while True:
        before_mt = TrajectoryMTRandState(
            index=random.index,
            words=random.words,
        )
        outputs = tuple(random.next_u31() for _ in range(4))
        if outputs[0] % 617 == 0:
            break
    after_mt = TrajectoryMTRandState(
        index=random.index,
        words=random.words,
    )
    selected_type = 0 if outputs[1] % 200 < 100 else 3
    selected_color = outputs[2] % 4
    before_entities = tuple(
        _powerup_ball(20 + index, index % 4, index)
        for index in range(8)
    )
    candidates = [
        entity
        for entity in before_entities
        if entity.color_id == selected_color
    ]
    selected_ball = candidates[outputs[3] % len(candidates)]
    after_entities = tuple(
        replace(
            entity,
            powerup_secondary_type=selected_type,
            powerup_transition_ticks=100,
            powerup_visual_scale=5.0,
            powerup_visual_step=float(np.float32(4.0) / np.float32(100.0)),
            powerup_visual_index=3 if selected_type == 0 else 4,
        )
        if entity.ball_id == selected_ball.ball_id
        else entity
        for entity in before_entities
    )
    common = {
        "score": 500,
        "displayed_score": 500,
        "score_target": 10_000,
        "current_ball_id": 100,
        "current_color_id": 2,
        "next_ball_id": 101,
        "next_color_id": 1,
        "list_counts": ((0, ACTIVE_CHAIN_LIST_OFFSET, 8),),
        "qrand": _qrand(),
        "thread_crt_rand_state": 12345,
    }
    return (
        (
            TrajectoryFrame(
                update=100,
                entities=before_entities,
                global_mtrand=before_mt,
                native_game_time=2000,
                curve_powerups=(
                    _spawn_state(native_time=2000),
                ),
                **common,
            ),
            TrajectoryFrame(
                update=101,
                entities=after_entities,
                global_mtrand=after_mt,
                native_game_time=2001,
                curve_powerups=(
                    _spawn_state(
                        native_time=2001,
                        selected_type=selected_type,
                        selected_color=selected_color,
                    ),
                ),
                **common,
            ),
        ),
        selected_ball.ball_id,
        selected_type,
        outputs,
    )


def _powerup_curve() -> SimpleNamespace:
    records = [(0, 0)] * 14
    records[0] = (100, 2_147_483_647)
    records[3] = (100, 2_147_483_647)
    return SimpleNamespace(
        parameters=SimpleNamespace(
            colors=4,
            powerup_records=tuple(records),
        )
    )


def test_powerup_spawn_closes_all_four_mtrand_draws(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames, selected_ball_id, selected_type, outputs = _spawn_frames()
    monkeypatch.setattr(
        pc_mechanism_audit,
        "_original_curve",
        lambda **kwargs: _powerup_curve(),
    )

    features, proofs = (
        pc_mechanism_audit.derive_native_mechanism_features(
            frames,
            original_root=Path("/unused"),
            level_id="Jungle2",
        )
    )

    assert features == ("powerup_spawn",)
    assert proofs[0]["selected_ball_id"] == selected_ball_id
    assert proofs[0]["powerup_type"] == selected_type
    assert proofs[0]["mtrand_outputs"] == list(outputs)
    assert proofs[0]["mtrand_state_exact"] is True


def test_powerup_spawn_rejects_one_extra_mtrand_draw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames, _, _, _ = _spawn_frames()
    random = PopCapMTRandom(1)
    assert frames[1].global_mtrand is not None
    random.load_state(
        frames[1].global_mtrand.words,
        frames[1].global_mtrand.index,
    )
    random.next_u31()
    tampered = (
        frames[0],
        replace(
            frames[1],
            global_mtrand=TrajectoryMTRandState(
                index=random.index,
                words=random.words,
            ),
        ),
    )
    monkeypatch.setattr(
        pc_mechanism_audit,
        "_original_curve",
        lambda **kwargs: _powerup_curve(),
    )

    features, _ = (
        pc_mechanism_audit.derive_native_mechanism_features(
            tampered,
            original_root=Path("/unused"),
            level_id="Jungle2",
        )
    )

    assert "powerup_spawn" not in features


def _fruit_state(
    *,
    active_point_pointer: int,
    selected_point_index: int,
    expiry_time: int,
    spawned: bool,
) -> TrajectoryFruitState:
    return TrajectoryFruitState(
        active_point_pointer=active_point_pointer,
        selected_point_index=selected_point_index,
        collecting=False,
        velocity=0.25 if spawned else -0.04,
        max_velocity=0.25,
        acceleration=-0.01,
        vertical_offset=0.0 if spawned else 3.41,
        lower_bound=(
            float(np.finfo(np.float32).max) if spawned else 0.0
        ),
        upper_bound=(
            float(np.finfo(np.float32).max) if spawned else 0.0
        ),
        glow_alpha=0 if spawned else 255,
        glow_step=12 if spawned else -48,
        alpha=255,
        expiry_time=expiry_time,
        cell_index=51,
    )


def _collecting_fruit_state(
    *,
    active: bool,
    collecting: bool,
    alpha: int,
    glow_alpha: int,
    glow_step: int,
    cell_index: int,
    pointer: int = 0x25000030,
    selected_point_index: int = 0,
) -> TrajectoryFruitState:
    return TrajectoryFruitState(
        active_point_pointer=pointer if active else 0,
        selected_point_index=(selected_point_index if active else 0),
        collecting=collecting,
        velocity=0.25,
        max_velocity=0.25,
        acceleration=-0.01,
        vertical_offset=0.0,
        lower_bound=float(np.finfo(np.float32).max),
        upper_bound=float(np.finfo(np.float32).max),
        glow_alpha=glow_alpha,
        glow_step=glow_step,
        alpha=alpha,
        expiry_time=2_000,
        cell_index=cell_index,
    )


def test_fruit_projectile_score_and_collection_animation_are_recomputed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibration = FruitSpawnCalibration(
        frequency_ticks=1_000,
        lifetime_ticks=1_000,
        points=((100.0, 100.0),),
        unlock_percentages=((1,),),
        fruit_type="pineapple",
        collection_animation_frames=3,
        collection_animation_fps=100.0,
        provenance="unit-test installed fruit assets and static flow",
    )
    curve = SimpleNamespace(
        end_waypoint=999,
        parameters=SimpleNamespace(zuma_score=6_000),
    )
    monkeypatch.setattr(
        pc_mechanism_audit,
        "_installed_fruit_scheduler_context",
        lambda **kwargs: ((curve,), calibration),
    )
    projectile = replace(
        _entity(
            77,
            2,
            zone="fired",
            index=0,
            curve_distance=0.0,
            x=126.0,
            y=126.0,
            kind="bullet",
        ),
        native_object_address=0x25001000,
        fired=False,
    )
    common = {
        "displayed_score": 4_600,
        "score_target": 10_000,
        "current_ball_id": None,
        "current_color_id": None,
        "next_ball_id": None,
        "next_color_id": None,
        "list_counts": (),
    }
    frames = (
        TrajectoryFrame(
            update=100,
            score=4_600,
            entities=(projectile,),
            native_game_time=100,
            board_update_count=100,
            fruit_state=_collecting_fruit_state(
                active=True,
                collecting=False,
                alpha=255,
                glow_alpha=0,
                glow_step=12,
                cell_index=0,
            ),
            **common,
        ),
        TrajectoryFrame(
            update=101,
            score=5_100,
            entities=(),
            native_game_time=101,
            board_update_count=101,
            fruit_state=_collecting_fruit_state(
                active=True,
                collecting=True,
                alpha=255,
                glow_alpha=0,
                glow_step=12,
                cell_index=0,
            ),
            **common,
        ),
        TrajectoryFrame(
            update=102,
            score=5_100,
            entities=(),
            native_game_time=102,
            board_update_count=102,
            fruit_state=_collecting_fruit_state(
                active=True,
                collecting=True,
                alpha=247,
                glow_alpha=12,
                glow_step=12,
                cell_index=1,
            ),
            **common,
        ),
        TrajectoryFrame(
            update=103,
            score=5_100,
            entities=(),
            native_game_time=103,
            board_update_count=103,
            fruit_state=_collecting_fruit_state(
                active=False,
                collecting=True,
                alpha=239,
                glow_alpha=24,
                glow_step=12,
                cell_index=1,
            ),
            **common,
        ),
    )

    features, raw_proofs = (
        pc_mechanism_audit.derive_native_mechanism_features(
            frames,
            original_root=Path("/unused"),
            level_id="Jungle2",
        )
    )
    proofs = {proof["feature"]: proof for proof in raw_proofs}

    assert {
        "fruit_projectile_collision",
        "fruit_collection_score",
        "fruit_collection_animation",
    }.issubset(features)
    assert proofs["fruit_projectile_collision"]["projectile_ball_id"] == 77
    assert proofs["fruit_collection_score"]["score_delta"] == 500
    assert proofs["fruit_collection_animation"]["clear_update"] == 103

    tampered = list(frames)
    tampered[2] = replace(
        tampered[2],
        fruit_state=replace(tampered[2].fruit_state, alpha=246),
    )
    tampered_features, _ = (
        pc_mechanism_audit.derive_native_mechanism_features(
            tuple(tampered),
            original_root=Path("/unused"),
            level_id="Jungle2",
        )
    )
    assert "fruit_collection_animation" not in tampered_features


@pytest.mark.parametrize(
    ("fruit_x", "expected"),
    ((207.999, True), (208.0, False)),
)
def test_fruit_powerup_collision_uses_strict_108_pixel_radius(
    monkeypatch: pytest.MonkeyPatch,
    fruit_x: float,
    expected: bool,
) -> None:
    calibration = FruitSpawnCalibration(
        frequency_ticks=1_000,
        lifetime_ticks=1_000,
        points=((fruit_x, 100.0),),
        unlock_percentages=((1,),),
        fruit_type="pineapple",
        provenance="unit-test installed fruit assets and static flow",
    )
    curve = SimpleNamespace(
        end_waypoint=999,
        parameters=SimpleNamespace(zuma_score=6_000),
    )
    monkeypatch.setattr(
        pc_mechanism_audit,
        "_installed_fruit_scheduler_context",
        lambda **kwargs: ((curve,), calibration),
    )
    active_zone = f"curve:0:list:{ACTIVE_CHAIN_LIST_OFFSET:03x}"
    trigger_ball = _entity(
        9,
        1,
        zone=active_zone,
        index=0,
        curve_distance=100.0,
        x=100.0,
        y=100.0,
    )
    common = {
        "score": 4_600,
        "displayed_score": 4_600,
        "score_target": 10_000,
        "current_ball_id": None,
        "current_color_id": None,
        "next_ball_id": None,
        "next_color_id": None,
        "list_counts": ((0, ACTIVE_CHAIN_LIST_OFFSET, 1),),
        "entities": (trigger_ball,),
    }
    frames = (
        TrajectoryFrame(
            update=200,
            native_game_time=200,
            board_update_count=200,
            fruit_state=_collecting_fruit_state(
                active=True,
                collecting=False,
                alpha=255,
                glow_alpha=0,
                glow_step=12,
                cell_index=0,
            ),
            **common,
        ),
        TrajectoryFrame(
            update=201,
            native_game_time=201,
            board_update_count=201,
            fruit_state=_collecting_fruit_state(
                active=True,
                collecting=True,
                alpha=255,
                glow_alpha=0,
                glow_step=12,
                cell_index=0,
            ),
            **common,
        ),
    )
    powerup_proof = {
        "feature": "powerup_proximity_bomb",
        "status": "PASS",
        "framework_update": 201,
        "powerup_type": 0,
        "trigger_ball_id": 9,
    }

    proofs = pc_mechanism_audit._derive_fruit_collection_proofs(
        frames,
        original_root=Path("/unused"),
        level_id="Jungle2",
        hard=False,
        curve_index=0,
        powerup_effect_proofs=(powerup_proof,),
    )
    features = {proof["feature"] for proof in proofs}

    assert ("fruit_powerup_collision" in features) is expected


def test_installed_fruit_scheduler_context_uses_one_catalog_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCatalog:
        def __init__(self, root: Path) -> None:
            assert root == Path("/retail")

        def load_level(self, level_id: str, *, hard: bool) -> object:
            assert level_id == "Jungle2"
            assert hard is False
            definition = SimpleNamespace(
                attributes={"tfreq": "1000"},
                treasure_points=((28.0, 105.0),),
                treasure_point_distances=((60,),),
                fruit_type="pineapple",
            )
            return SimpleNamespace(
                definition=definition,
                curves=(SimpleNamespace(end_waypoint=999),),
            )

        def fruit_assets(self, fruit_type: str) -> object:
            assert fruit_type == "pineapple"
            return SimpleNamespace(
                fruit_type=fruit_type,
                logical_width=52,
                logical_height=52,
                sheet_columns=8,
                sheet_rows=8,
                collection_animation_frames=6,
                collection_animation_fps=24.0,
                provenance="unit-test installed fruit assets",
            )

    monkeypatch.setattr(
        pc_mechanism_audit,
        "OriginalGameCatalog",
        FakeCatalog,
    )

    context = pc_mechanism_audit._installed_fruit_scheduler_context(
        original_root=Path("/retail"),
        level_id="Jungle2",
        hard=False,
        curve_index=0,
    )

    assert context is not None
    curves, calibration = context
    assert len(curves) == 1
    assert calibration.frequency_ticks == 1_000
    assert calibration.points == ((28.0, 105.0),)
    assert calibration.fruit_type == "pineapple"
    assert calibration.collection_animation_frames == 6


def test_fruit_scheduler_spawn_uses_front_ball_and_exact_mt_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibration = FruitSpawnCalibration(
        frequency_ticks=1_000,
        lifetime_ticks=1_000,
        points=((28.0, 105.0), (716.0, 505.0), (602.0, 366.0)),
        unlock_percentages=((60,), (33,), (84,)),
        fruit_type="pineapple",
        provenance="unit-test retail static control flow",
    )
    monkeypatch.setattr(
        pc_mechanism_audit,
        "_installed_fruit_scheduler_context",
        lambda **kwargs: ((SimpleNamespace(end_waypoint=999),), calibration),
    )
    random = PopCapMTRandom(0xC0FFEE)
    while True:
        before_mt = TrajectoryMTRandState(
            index=random.index,
            words=random.words,
        )
        chance = random.next_u31()
        if chance % 1_000 == 0:
            point_choice = random.next_u31()
            trailing = random.next_u31()
            break
    after_mt = TrajectoryMTRandState(
        index=random.index,
        words=random.words,
    )
    active_zone = f"curve:0:list:{ACTIVE_CHAIN_LIST_OFFSET:03x}"
    entities = (
        _entity(
            10,
            0,
            zone=active_zone,
            index=0,
            curve_distance=1.0,
            x=1.0,
        ),
        _entity(
            11,
            1,
            zone=active_zone,
            index=1,
            curve_distance=850.0,
            x=850.0,
        ),
    )
    selected = point_choice % 3
    common = {
        "score": 500,
        "displayed_score": 500,
        "score_target": 10_000,
        "current_ball_id": None,
        "current_color_id": None,
        "next_ball_id": None,
        "next_color_id": None,
        "list_counts": ((0, ACTIVE_CHAIN_LIST_OFFSET, 2),),
        "entities": entities,
        "qrand": _qrand(),
        "thread_crt_rand_state": 12345,
    }
    frames = (
        TrajectoryFrame(
            update=100,
            global_mtrand=before_mt,
            native_game_time=2_000,
            fruit_state=_fruit_state(
                active_point_pointer=0,
                selected_point_index=0,
                expiry_time=500,
                spawned=False,
            ),
            **common,
        ),
        TrajectoryFrame(
            update=101,
            global_mtrand=after_mt,
            native_game_time=2_001,
            fruit_state=_fruit_state(
                active_point_pointer=0x25000030,
                selected_point_index=selected,
                expiry_time=3_001,
                spawned=True,
            ),
            **common,
        ),
    )

    features, proofs = pc_mechanism_audit.derive_native_mechanism_features(
        frames,
        original_root=Path("/unused"),
        level_id="Jungle2",
    )

    assert features == ("fruit_scheduler_spawn",)
    proof = proofs[0]
    assert proof["selected_point_index"] == selected
    assert proof["eligible_point_indices"] == [0, 1, 2]
    assert proof["curve_progress"][0]["front_ball_id"] == 11
    assert proof["curve_progress"][0]["progress_percent"] == 85
    assert proof["mtrand_scheduler_outputs"] == [chance, point_choice]
    assert proof["mtrand_outputs"] == [chance, point_choice, trailing]
    assert proof["mtrand_trailing_draw_count"] == 1


def test_fruit_expiry_requires_exact_native_boundary_without_collection() -> None:
    common = {
        "score": 7950,
        "displayed_score": 7950,
        "score_target": 10_000,
        "current_ball_id": None,
        "current_color_id": None,
        "next_ball_id": None,
        "next_color_id": None,
        "list_counts": (),
        "entities": (),
    }
    frames = (
        TrajectoryFrame(
            update=10289,
            native_game_time=6700,
            fruit_state=_fruit_state(
                active_point_pointer=0x26000010,
                selected_point_index=1,
                expiry_time=6701,
                spawned=False,
            ),
            **common,
        ),
        TrajectoryFrame(
            update=10290,
            native_game_time=6701,
            fruit_state=_fruit_state(
                active_point_pointer=0,
                selected_point_index=0,
                expiry_time=6701,
                spawned=False,
            ),
            **common,
        ),
    )

    features, proofs = pc_mechanism_audit.derive_native_mechanism_features(
        frames
    )

    assert features == ("fruit_expiry",)
    assert proofs[0]["active_pointer_transition"] == "nonzero_to_zero"
    assert proofs[0]["native_game_time_after"] == 6701
    assert proofs[0]["score_delta"] == 0


def _effect_state(
    powerup_type: int,
    update: int,
) -> TrajectoryCurvePowerupState:
    triggered = update >= 102
    cooldowns = [-1000] * 14
    counters = [0] * 14
    active_colors = [0] * 6
    active_colors[3] = 0 if triggered else 1
    if triggered:
        cooldowns[powerup_type] = 1002
        counters[powerup_type] = 1
    return TrajectoryCurvePowerupState(
        curve_index=0,
        last_any_spawn_time=800,
        last_spawn_times=(-1000,) * 14,
        cooldown_times=tuple(cooldowns),
        spawn_counts=(0,) * 14,
        field_124_by_type=tuple(counters),
        active_color_counts=tuple(active_colors),
        reverse_speed=(
            1.0 if powerup_type == 3 and triggered else 0.5
        ),
        slow_ticks=(
            800 - (update - 102)
            if powerup_type == 1 and triggered
            else 0
        ),
        reverse_ticks=(
            300 - (update - 102)
            if powerup_type == 3 and triggered
            else 0
        ),
        last_powerup_waypoint=0,
        powerup_triggered=triggered,
    )


def _effect_ball(
    ball_id: int,
    color_id: int,
    index: int,
    distance: float,
    *,
    exploding: bool,
    contact_next: bool,
    powerup_type: int = 14,
    lifetime: int = 0,
) -> TrajectoryEntity:
    return TrajectoryEntity(
        ball_id=ball_id,
        color_id=color_id,
        object_kind="ball",
        zone=f"curve:0:list:{ACTIVE_CHAIN_LIST_OFFSET:03x}",
        index=index,
        curve_distance=distance,
        position_x=distance,
        position_y=100.0,
        radius=18.0,
        contact_next=contact_next,
        exploding=exploding,
        explode_frame=0,
        should_remove=False,
        update_count=100,
        suck_count=0,
        backwards_count=0,
        backwards_speed=1.0,
        combo_count=0,
        combo_score=0,
        powerup_previous_type=14,
        powerup_primary_type=powerup_type,
        powerup_secondary_type=14,
        powerup_previous_ticks=0,
        powerup_lifetime_ticks=lifetime,
        powerup_transition_ticks=0,
        powerup_visual_scale=1.0,
        powerup_visual_step=0.0,
        powerup_visual_index=-1,
    )


def _effect_frames(powerup_type: int) -> tuple[TrajectoryFrame, ...]:
    frames: list[TrajectoryFrame] = []
    slow_reference = {
        100: 2000.0,
        101: 2001.0,
        102: 2001.25,
        103: 2001.5,
        104: 2001.75,
    }
    reverse_reference = {
        100: 3602.0,
        101: 3603.0,
        102: 3604.0,
        103: 3603.0,
        104: 3602.0,
    }
    for update in range(100, 105):
        triggered = update >= 102
        center = 1000.0
        entities = [
            _effect_ball(
                28,
                3,
                0,
                center - 36.0,
                exploding=triggered,
                contact_next=True,
            ),
            _effect_ball(
                29,
                3,
                1,
                center,
                exploding=triggered,
                contact_next=True,
                powerup_type=powerup_type,
                lifetime=445 if triggered else 446,
            ),
            _effect_ball(
                30,
                3,
                2,
                center + 36.0,
                exploding=triggered,
                contact_next=False,
            ),
            _effect_ball(
                176,
                1,
                3,
                (
                    reverse_reference[update]
                    if powerup_type == 3
                    else slow_reference[update]
                ),
                exploding=False,
                contact_next=False,
            ),
        ]
        if powerup_type == 0:
            entities.extend(
                (
                    _effect_ball(
                        200,
                        2,
                        4,
                        center + 144.0,
                        exploding=triggered,
                        contact_next=False,
                    ),
                    _effect_ball(
                        201,
                        2,
                        5,
                        center + 148.0,
                        exploding=False,
                        contact_next=False,
                    ),
                )
            )
        exploding_count = 4 if powerup_type == 0 else 3
        score = 8120 + (10 * exploding_count if triggered else 0)
        frames.append(
            TrajectoryFrame(
                update=update,
                score=score,
                displayed_score=score,
                score_target=10_000,
                current_ball_id=119,
                current_color_id=3,
                next_ball_id=122,
                next_color_id=1,
                list_counts=(
                    (0, ACTIVE_CHAIN_LIST_OFFSET, len(entities)),
                ),
                entities=tuple(entities),
                native_game_time=900 + update,
                curve_powerups=(
                    _effect_state(powerup_type, update),
                ),
            )
        )
    return tuple(frames)


@pytest.mark.parametrize(
    ("powerup_type", "expected_feature"),
    (
        (0, "powerup_proximity_bomb"),
        (1, "powerup_slow"),
        (3, "powerup_reverse"),
    ),
)
def test_powerup_effect_requires_bookkeeping_timer_and_physics(
    powerup_type: int,
    expected_feature: str,
) -> None:
    features, proofs = (
        pc_mechanism_audit.derive_native_mechanism_features(
            _effect_frames(powerup_type)
        )
    )

    assert features == (expected_feature,)
    proof = proofs[0]
    assert proof["manager_transition_exact"] is True
    assert proof["effect_verification"]["status"] == "PASS"
    if powerup_type == 0:
        assert (
            proof["effect_verification"]["bomb_geometry"][
                "nearest_rejected"
            ]["ball_id"]
            == 201
        )
    elif powerup_type == 1:
        assert proof["slow_physical_movement"]["strictly_slower"] is True
    else:
        assert (
            proof["effect_verification"]["reverse"]["movement"][
                "reference_ball_id"
            ]
            == 176
        )


def _bomb_with_concurrent_fruit_frames(
    *,
    fruit_collects: bool,
    concurrent_score: int,
) -> tuple[TrajectoryFrame, ...]:
    result: list[TrajectoryFrame] = []
    for frame in _effect_frames(0):
        triggered = frame.update >= 102
        result.append(
            replace(
                frame,
                score=(
                    frame.score + concurrent_score
                    if triggered
                    else frame.score
                ),
                displayed_score=(
                    frame.displayed_score + concurrent_score
                    if triggered
                    else frame.displayed_score
                ),
                board_update_count=frame.update,
                fruit_state=_collecting_fruit_state(
                    active=True,
                    collecting=fruit_collects and triggered,
                    alpha=255,
                    glow_alpha=0,
                    glow_step=12,
                    cell_index=0,
                ),
            )
        )
    return tuple(result)


def test_powerup_and_fruit_scores_compose_only_with_exact_native_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibration = FruitSpawnCalibration(
        frequency_ticks=1_000,
        lifetime_ticks=1_000,
        points=((1_000.0, 100.0),),
        unlock_percentages=((1,),),
        fruit_type="pineapple",
        provenance="unit-test installed fruit assets and static flow",
    )
    curve = SimpleNamespace(
        end_waypoint=999,
        parameters=SimpleNamespace(zuma_score=2_000),
    )
    monkeypatch.setattr(
        pc_mechanism_audit,
        "_installed_fruit_scheduler_context",
        lambda **kwargs: ((curve,), calibration),
    )

    frames = _bomb_with_concurrent_fruit_frames(
        fruit_collects=True,
        concurrent_score=500,
    )
    powerup_proofs = pc_mechanism_audit._derive_powerup_effect_proofs(
        frames,
        original_root=Path("/unused"),
        level_id="Jungle2",
        hard=False,
        curve_index=0,
    )
    fruit_proofs = pc_mechanism_audit._derive_fruit_collection_proofs(
        frames,
        original_root=Path("/unused"),
        level_id="Jungle2",
        hard=False,
        curve_index=0,
        powerup_effect_proofs=powerup_proofs,
    )
    raw_proofs = powerup_proofs + fruit_proofs
    features = {proof["feature"] for proof in raw_proofs}
    proofs = {proof["feature"]: proof for proof in raw_proofs}

    assert "powerup_proximity_bomb" in features
    assert "fruit_powerup_collision" in features
    assert "fruit_projectile_collision" not in features
    bomb = proofs["powerup_proximity_bomb"]
    assert bomb["base_explosion_score_delta"] == 40
    assert bomb["concurrent_score_binding"]["expected_score_delta"] == 500
    assert (
        bomb["effect_verification"]["trigger"][
            "concurrent_score_delta"
        ]
        == 500
    )
    assert proofs["fruit_powerup_collision"]["trigger_ball_id"] == 29

    tampered_powerup_proofs = (
        pc_mechanism_audit._derive_powerup_effect_proofs(
            _bomb_with_concurrent_fruit_frames(
                fruit_collects=False,
                concurrent_score=500,
            ),
            original_root=Path("/unused"),
            level_id="Jungle2",
            hard=False,
            curve_index=0,
        )
    )
    tampered_features = {
        proof["feature"] for proof in tampered_powerup_proofs
    }
    assert "powerup_proximity_bomb" not in tampered_features
    assert "fruit_powerup_collision" not in tampered_features


def test_slow_powerup_timer_without_physical_slowdown_is_rejected() -> None:
    frames = list(_effect_frames(1))
    for index, frame in enumerate(frames):
        entities = tuple(
            replace(
                entity,
                curve_distance=2000.0 + index,
                position_x=2000.0 + index,
            )
            if entity.ball_id == 176
            else entity
            for entity in frame.entities
        )
        frames[index] = replace(frame, entities=entities)

    features, _ = (
        pc_mechanism_audit.derive_native_mechanism_features(tuple(frames))
    )

    assert "powerup_slow" not in features


def test_reverse_powerup_timer_without_two_post_trigger_steps_is_rejected() -> None:
    frames = list(_effect_frames(3))
    entities = tuple(
        replace(entity, curve_distance=3603.0, position_x=3603.0)
        if entity.ball_id == 176
        else entity
        for entity in frames[4].entities
    )
    frames[4] = replace(frames[4], entities=entities)

    features, _ = (
        pc_mechanism_audit.derive_native_mechanism_features(tuple(frames))
    )

    assert "powerup_reverse" not in features


def test_shot_and_collision_require_one_tracked_projectile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = (
        {
            "update": 11,
            "kind": "entity_zone_change",
            "ball_id": 42,
            "color_id": 3,
            "from_zone": "shooter_current",
            "to_zone": "fired",
        },
        {
            "update": 11,
            "kind": "projectile_spawn",
            "entity": {"ball_id": 42, "color_id": 3},
        },
        {
            "update": 11,
            "kind": "shooter_chamber_change",
            "before": {
                "current_ball_id": 42,
                "next_ball_id": 43,
            },
            "after": {
                "current_ball_id": 43,
                "next_ball_id": 44,
            },
        },
        {
            "update": 12,
            "kind": "projectile_hit_staging",
            "ball_id": 42,
            "color_id": 3,
            "from_zone": "fired",
            "to_zone": "curve:0:list:050",
        },
    )
    monkeypatch.setattr(
        pc_mechanism_audit,
        "derive_trajectory_events",
        lambda frames: events,
    )

    features, proofs = (
        pc_mechanism_audit.derive_native_mechanism_features(
            _frames(10, 12)
        )
    )

    assert features == ("projectile_collision", "shot_release")
    assert {row["feature"] for row in proofs} == set(features)


def test_matches_and_rollback_are_coupled_to_score_and_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = (
        {
            "update": 11,
            "kind": "insertion_commit",
            "inserted_entity": {"ball_id": 3},
        },
        *(
            {
                "update": 11,
                "kind": "explosion_started",
                "ball_id": ball_id,
                "color_id": 2,
                "combo_count": 0,
            }
            for ball_id in (1, 2, 3)
        ),
        {
            "update": 11,
            "kind": "score_change",
            "score_before": 100,
            "score_after": 130,
        },
        {
            "update": 12,
            "kind": "rollback_started",
            "ball_id": 8,
            "color_id": 4,
        },
        {
            "update": 14,
            "kind": "rollback_stopped",
            "ball_id": 8,
            "color_id": 4,
        },
        *(
            {
                "update": 14,
                "kind": "explosion_started",
                "ball_id": ball_id,
                "color_id": 4,
                "combo_count": 1,
            }
            for ball_id in (7, 8, 9, 10)
        ),
        {
            "update": 14,
            "kind": "score_change",
            "score_before": 130,
            "score_after": 300,
        },
    )
    monkeypatch.setattr(
        pc_mechanism_audit,
        "derive_trajectory_events",
        lambda frames: events,
    )

    features, proofs = (
        pc_mechanism_audit.derive_native_mechanism_features(
            _frames(10, 14)
        )
    )

    assert features == ("match3", "match4", "rollback_chain")
    proof_by_feature = {row["feature"]: row for row in proofs}
    assert proof_by_feature["match3"]["trigger"] == "insertion_commit"
    assert proof_by_feature["match4"]["trigger"] == "rollback_stopped"
    assert proof_by_feature["rollback_chain"]["ball_id"] == 8


def test_simulator_source_feature_proofs_are_recomputed_from_raw_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = tmp_path / "index.json"
    index.write_text("{}\n", encoding="ascii")
    original_root = tmp_path / "original"
    original_root.mkdir()
    features = ("match3",)
    proofs = (
        {
            "feature": "match3",
            "status": "PASS",
            "framework_update": 11,
            "score_delta": 30,
            "exploding_ball_ids": [1, 2, 3],
        },
    )
    monkeypatch.setattr(
        pc_mechanism_audit,
        "load_memory_trajectory",
        lambda *args, **kwargs: _frames(10, 12),
    )
    monkeypatch.setattr(
        pc_mechanism_audit,
        "derive_native_mechanism_features",
        lambda *args, **kwargs: (
            features,
            tuple(dict(proof) for proof in proofs),
        ),
    )
    report = {
        "schema": "zuma-rl.pc-gameplay-simulator-diff",
        "start_update": 10,
        "end_update": 12,
        "curve_index": 0,
        "source_authorized_features": list(features),
        "source_feature_proofs": [dict(proof) for proof in proofs],
        "trajectory": {
            "source_version": 2,
            "artifact": "index.json",
            "artifact_sha256": (
                "sha256:" + pc_mechanism_audit.hashlib.sha256(
                    index.read_bytes()
                ).hexdigest()
            ),
            "selected_start_update": 10,
            "selected_end_update": 12,
        },
    }

    assert (
        pc_mechanism_audit.validate_simulator_source_feature_proofs(
            report,
            evidence_root=tmp_path,
            original_root=original_root,
            level_id="Jungle2",
            hard=False,
        )
        is None
    )

    report["source_feature_proofs"][0]["score_delta"] = 31
    assert (
        pc_mechanism_audit.validate_simulator_source_feature_proofs(
            report,
            evidence_root=tmp_path,
            original_root=original_root,
            level_id="Jungle2",
            hard=False,
        )
        == "simulator source feature proofs differ from raw evidence"
    )


def test_simulator_source_feature_proofs_allow_bound_enclosing_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = tmp_path / "index.json"
    index.write_text("{}\n", encoding="ascii")
    original_root = tmp_path / "original"
    original_root.mkdir()
    loaded_windows: list[tuple[int, int]] = []

    def load_frames(*args, **kwargs):
        loaded_windows.append(
            (kwargs["start_update"], kwargs["end_update"])
        )
        return _frames(kwargs["start_update"], kwargs["end_update"])

    monkeypatch.setattr(
        pc_mechanism_audit,
        "load_memory_trajectory",
        load_frames,
    )
    monkeypatch.setattr(
        pc_mechanism_audit,
        "derive_native_mechanism_features",
        lambda *args, **kwargs: (
            ("powerup_proximity_bomb",),
            (
                {
                    "feature": "powerup_proximity_bomb",
                    "status": "PASS",
                    "framework_update": 12,
                },
            ),
        ),
    )
    report = {
        "schema": "zuma-rl.pc-gameplay-simulator-diff",
        "start_update": 10,
        "end_update": 12,
        "curve_index": 0,
        "source_authorized_features": ["powerup_proximity_bomb"],
        "source_feature_proofs": [
            {
                "feature": "powerup_proximity_bomb",
                "status": "PASS",
                "framework_update": 12,
            }
        ],
        "trajectory": {
            "source_version": 2,
            "artifact": "index.json",
            "artifact_sha256": (
                "sha256:" + pc_mechanism_audit.hashlib.sha256(
                    index.read_bytes()
                ).hexdigest()
            ),
            "captured_start_update": 9,
            "captured_end_update": 14,
            "selected_start_update": 10,
            "selected_end_update": 12,
            "source_proof_start_update": 9,
            "source_proof_end_update": 14,
        },
    }

    assert (
        pc_mechanism_audit.validate_simulator_source_feature_proofs(
            report,
            evidence_root=tmp_path,
            original_root=original_root,
            level_id="Jungle2",
            hard=False,
        )
        is None
    )
    assert loaded_windows == [(9, 14)]

    report["trajectory"]["source_proof_start_update"] = 11
    assert (
        pc_mechanism_audit.validate_simulator_source_feature_proofs(
            report,
            evidence_root=tmp_path,
            original_root=original_root,
            level_id="Jungle2",
            hard=False,
        )
        == "simulator source proof window is invalid"
    )


def test_unscored_explosion_group_does_not_authorize_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = (
        {
            "update": 11,
            "kind": "insertion_commit",
            "inserted_entity": {"ball_id": 3},
        },
        *(
            {
                "update": 11,
                "kind": "explosion_started",
                "ball_id": ball_id,
                "color_id": 2,
                "combo_count": 0,
            }
            for ball_id in (1, 2, 3)
        ),
    )
    monkeypatch.setattr(
        pc_mechanism_audit,
        "derive_trajectory_events",
        lambda frames: events,
    )

    features, _ = pc_mechanism_audit.derive_native_mechanism_features(
        _frames(10, 11)
    )

    assert features == ()


def test_plan_has_no_feature_claim_field(tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema": pc_mechanism_audit.PLAN_SCHEMA,
                "version": pc_mechanism_audit.PLAN_VERSION,
                "cases": [
                    {
                        "id": "case-a",
                        "manifest_path": "cases/a/manifest.json",
                        "memory_probe_path": "diagnostics/a/memory-probe.json",
                        "window": {
                            "start_update": 10,
                            "end_update": 20,
                        },
                        "features": ["projectile_collision"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        pc_mechanism_audit.PcMechanismAuditError,
        match="mechanism_audit_case_fields_invalid",
    ):
        pc_mechanism_audit.read_mechanism_audit_plan(plan)


def test_natural_probe_rejects_diagnostic_mutation() -> None:
    with pytest.raises(
        pc_mechanism_audit.PcMechanismAuditError,
        match="mechanism_audit_diagnostic_mutation_forbidden",
    ):
        pc_mechanism_audit._require_natural_probe(
            {"diagnostic_mutation": {"kind": "score_write"}}
        )


def test_natural_probe_accepts_formal_independent_score_binding() -> None:
    probe = {
        "diagnostic_mutation": None,
        "evidence_classification": (
            "formal_pc_full_state_exact_step_source"
        ),
        "value": 7990,
        "repaint": {
            "schema": "zuma-rl.pc-window-repaint-handshake",
            "version": 1,
            "effectful_command_guard_satisfied": True,
            "geometry_restored": True,
            "foreground_activation_verified": True,
        },
        "frozen_frame": {"status": "PASS"},
        "active_board": {
            "score_offset": 0x104,
            "score": 7990,
            "displayed_score_offset": 0xEFC,
            "displayed_score": 7950,
        },
        "score_binding": {
            "schema": "zuma-rl.pc-memory-score-binding",
            "version": 2,
            "mode": "independent_active_board_fields_read_only",
            "score": 7990,
            "displayed_score": 7950,
            "scan_value": 7990,
            "score_offset": 0x104,
            "displayed_score_offset": 0xEFC,
            "acceptance_rule": (
                "all_int32_pairs_without_outcome_filtering"
            ),
        },
    }

    binding = pc_mechanism_audit._require_natural_probe(probe)

    assert binding["score"] == 7990
    assert binding["displayed_score"] == 7950


def test_formal_projection_preserves_all_rng_state() -> None:
    raw = {
        "framework_update": 42,
        "score": 10,
        "displayed_score": 10,
        "score_target": 100,
        "current_ball": None,
        "next_ball": None,
        "curve_lists": [],
    }
    rng_frame = SimpleNamespace(
        update=42,
        score=10,
        displayed_score=10,
        score_target=100,
        current_ball=None,
        next_ball=None,
        curve_lists=(),
        qrand_update_count=7,
        qrand_selected_index=3,
        qrand_vectors=(
            ("weights", (0.25, 0.25, 0.25, 0.25, 0.0, 0.0)),
            ("sways", (0.1, 0.2, 0.3, 0.4, 0.0, 0.0)),
            ("last_hit", (0, 1, 0, 0, 0, 0)),
            ("previous_hit", (1, 0, 0, 0, 0, 0)),
        ),
        thread_crt_rand_state=12345,
        mtrand_index=17,
        mtrand_words=(1, 2, 3),
    )

    (projected,) = pc_mechanism_audit._formal_exact_step_frames(
        SimpleNamespace(tick_rows=(raw,), frames=(rng_frame,))
    )

    assert projected.qrand is not None
    assert projected.qrand.update_count == 7
    assert projected.qrand.selected_index == 3
    assert projected.thread_crt_rand_state == 12345
    assert projected.global_mtrand == TrajectoryMTRandState(
        index=17,
        words=(1, 2, 3),
    )

def _fingerprinted(payload: dict[str, object]) -> dict[str, object]:
    report = dict(payload)
    report["audit_fingerprint"] = pc_mechanism_audit.canonical_sha256(
        report
    )
    return report


def test_historical_audit_normalization_changes_only_schema_metadata() -> None:
    common: dict[str, object] = {
        "schema": "zuma-rl.actor-audit",
        "status": "PASS",
        "audit_type": "pc_mechanism_coverage",
        "cases": [{"id": "source", "proofs": {"shot_release": {}}}],
        "summary": {"pc_source_count": 1},
    }
    live = _fingerprinted(
        {
            **common,
            "version": 2,
            "scope": {
                "policy": "original-transfer-jungle2-v2",
                "environment_id": "ZumaRevenge-v0",
            },
        }
    )
    historical = _fingerprinted(
        {
            **common,
            "version": 1,
            "scope": {
                "policy": "original-transfer-jungle2-v1",
                "environment_id": "ZumaRevenge-v0",
            },
        }
    )

    assert pc_mechanism_audit._reports_semantically_equivalent(
        historical,
        live,
    )

    tampered = dict(historical)
    tampered["summary"] = {"pc_source_count": 2}
    assert not pc_mechanism_audit._reports_semantically_equivalent(
        tampered,
        live,
    )

    relabeled = dict(historical)
    relabeled_scope = dict(relabeled["scope"])
    relabeled_scope["policy"] = "unrecognized-policy"
    relabeled["scope"] = relabeled_scope
    relabeled.pop("audit_fingerprint")
    relabeled = _fingerprinted(relabeled)
    assert not pc_mechanism_audit._reports_semantically_equivalent(
        relabeled,
        live,
    )
