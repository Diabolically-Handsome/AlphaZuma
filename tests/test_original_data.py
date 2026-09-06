import struct
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from zuma_rl.original_data import (
    OriginalDataError,
    OriginalGameCatalog,
    find_original_installation,
    parse_original_curve,
)


def _synthetic_curve(records: tuple[bytes, ...] | None = None) -> bytes:
    data = bytearray(b"CURV")
    data.extend(struct.pack("<I", 15))
    data.extend(struct.pack("<?", False))
    data.extend(struct.pack("<IIII", 65, 0, 45, 2))
    data.extend(struct.pack("<I", 4))
    data.extend(struct.pack("<f", 0.5))
    data.extend(struct.pack("<IffI", 200, 0.0, 100.0, 1_250))
    data.extend(struct.pack("<III", 75, 300, 1_100))
    data.extend(struct.pack("<fI", 4.0, 6))
    data.extend(struct.pack("<I", 0))
    data.extend(struct.pack("<I", 600))
    data.extend(bytes((1, 1, 1, 1, 1)))
    data.extend(bytes((1, 1)))
    if records is None:
        records = (
            struct.pack("<BBff", 2, 1, 10.0, 20.0),
            struct.pack("<BBbb", 0, 1, 100, 0),
            struct.pack("<BBbb", 1, 2, 0, -100),
            struct.pack("<BBbb", 0, 0, -60, 80),
        )
    data.extend(struct.pack("<I", len(records)))
    for record in records:
        data.extend(record)
    return bytes(data)


def test_parse_version_15_curve() -> None:
    curve = parse_original_curve(_synthetic_curve())
    assert curve.points.dtype == np.dtype(np.float32)
    assert curve.normalized_points.dtype == np.dtype(np.float32)
    assert curve.cumulative_distance.dtype == np.dtype(np.float64)
    np.testing.assert_allclose(
        curve.points,
        (
            (10.0, 20.0),
            (11.0, 20.0),
            (11.0, 19.0),
            (10.4, 19.8),
        ),
    )
    np.testing.assert_array_equal(curve.point_flags, (2, 0, 1, 0))
    np.testing.assert_array_equal(curve.in_tunnel, (False, False, True, False))
    np.testing.assert_array_equal(curve.priorities, (1, 1, 2, 0))
    np.testing.assert_array_equal(
        curve.absolute_anchors,
        (True, False, False, False),
    )
    assert curve.parameters.start_distance_percent == 65
    assert curve.parameters.ball_repeat_chance == 45
    assert curve.parameters.max_single == 2
    assert curve.parameters.colors == 4
    assert curve.parameters.speed == 0.5
    assert curve.parameters.zuma_score == 1_250
    assert curve.parameters.powerup_chance == 600
    assert curve.total_length == pytest.approx(3.0)


def test_relative_points_round_after_every_float32_addition() -> None:
    """A long delta run must not accumulate in float64 and cast only once."""

    delta_count = 4_096
    records = (
        struct.pack("<BBff", 2, 0, 1_000.0, 0.0),
        *(struct.pack("<BBbb", 0, 0, 1, 0) for _ in range(delta_count)),
    )
    curve = parse_original_curve(_synthetic_curve(records))

    sequential = np.float32(1_000.0)
    delta = np.multiply(
        np.float32(1),
        np.float32(0.01),
        dtype=np.float32,
    )
    for _ in range(delta_count):
        sequential = np.add(sequential, delta, dtype=np.float32)

    late_cast = np.float32(1_000.0 + delta_count * 0.01)
    assert sequential == np.float32(1_041.0)
    assert late_cast == np.float32(1_040.96)
    assert sequential != late_cast
    assert curve.points[-1, 0] == sequential
    assert curve.points[-1, 1] == np.float32(0.0)
    assert curve.total_length == pytest.approx(41.0)


def test_curve_interpolation() -> None:
    curve = parse_original_curve(_synthetic_curve())
    audit_point = curve.point_at_distance(0.5)
    assert audit_point.dtype == np.dtype(np.float64)
    np.testing.assert_allclose(audit_point, (10.5, 20.0))
    assert curve.priority_at_distance(0.0) == 1
    assert not curve.is_in_tunnel_at_distance(0.0)
    assert curve.priority_at_distance(1.5) == 1
    assert not curve.is_in_tunnel_at_distance(1.5)
    assert curve.is_in_tunnel_at_distance(2.0)
    points = curve.point_at_distance([0.0, curve.total_length])
    np.testing.assert_allclose(points, (curve.points[0], curve.points[-1]))


def test_waypoint_index_interpolation_and_bounds() -> None:
    curve = parse_original_curve(_synthetic_curve())
    assert curve.end_waypoint == 3
    np.testing.assert_allclose(curve.point_at_waypoint(0.5), (10.5, 20.0))
    np.testing.assert_allclose(curve.point_at_waypoint(-0.5), (9.5, 20.0))
    np.testing.assert_allclose(curve.point_at_waypoint(-1.25), (9.75, 20.0))
    np.testing.assert_allclose(curve.point_at_waypoint(99.5), curve.points[-1])
    np.testing.assert_allclose(
        curve.point_at_waypoint(4.5, loop_at_end=True),
        (10.5, 20.0),
    )
    np.testing.assert_allclose(
        curve.point_at_waypoint([0.0, 1.5, 3.0]),
        ((10.0, 20.0), (11.0, 19.5), (10.4, 19.8)),
    )
    assert curve.point_at_waypoint(0.5).dtype == np.dtype(np.float32)
    # The method's public argument is a game ``float``.  This Python float
    # rounds to exactly 1.0f before the integer cast and interpolation.
    np.testing.assert_array_equal(
        curve.point_at_waypoint(0.99999999),
        curve.points[1],
    )


def test_waypoint_interpolation_uses_float32_expression_order() -> None:
    records = (
        struct.pack("<BBff", 2, 0, 316.7768, 10.0),
        struct.pack("<BBff", 2, 0, 312.57623, 10.0),
    )
    curve = parse_original_curve(_synthetic_curve(records))
    fraction = np.float32(0.79932654)
    actual = curve.point_at_waypoint(float(fraction))

    delta = np.subtract(
        curve.points[1, 0],
        curve.points[0, 0],
        dtype=np.float32,
    )
    product = np.multiply(fraction, delta, dtype=np.float32)
    expected = np.add(product, curve.points[0, 0], dtype=np.float32)
    float64_then_cast = np.float32(
        float(curve.points[0, 0])
        + float(fraction)
        * (float(curve.points[1, 0]) - float(curve.points[0, 0]))
    )

    assert expected != float64_then_cast
    assert actual.dtype == np.dtype(np.float32)
    assert actual[0] == expected
    assert actual[1] == np.float32(10.0)


def test_waypoint_discontinuity_snaps_instead_of_interpolating() -> None:
    records = (
        struct.pack("<BBff", 2, 0, 10.0, 20.0),
        struct.pack("<BBbb", 0, 0, 100, 0),
        struct.pack("<BBff", 2, 2, 100.0, 200.0),
        struct.pack("<BBbb", 1, 2, 100, 0),
    )
    curve = parse_original_curve(_synthetic_curve(records))
    np.testing.assert_allclose(curve.point_at_waypoint(1.75), (11.0, 20.0))
    np.testing.assert_allclose(curve.point_at_waypoint(2.0), (100.0, 200.0))
    np.testing.assert_allclose(
        curve.perpendicular_at_waypoint([0, 1, 2, 3]),
        ((0.0, -1.0), (0.0, -1.0), (0.0, -1.0), (0.0, -1.0)),
    )
    assert curve.total_length > 200.0
    assert curve.end_waypoint == 3


def test_waypoint_tunnel_and_priority_bounds() -> None:
    curve = parse_original_curve(_synthetic_curve())
    np.testing.assert_array_equal(
        curve.priority_at_waypoint([-1, 0, 2, 4]),
        (0, 1, 2, 0),
    )
    np.testing.assert_array_equal(
        curve.is_in_tunnel_at_waypoint([-1, 0, 2, 4]),
        (True, False, True, False),
    )
    perpendicular = curve.perpendicular_at_waypoint([-10, 0, 1, 99])
    assert perpendicular.dtype == np.dtype(np.float32)
    np.testing.assert_allclose(
        perpendicular,
        ((0.0, -1.0), (0.0, -1.0), (-1.0, 0.0), (0.8, 0.6)),
        rtol=2e-6,
        atol=1e-7,
    )


def test_rejects_wrong_curve_version() -> None:
    data = bytearray(_synthetic_curve())
    struct.pack_into("<I", data, 4, 2)
    with pytest.raises(OriginalDataError, match="version"):
        parse_original_curve(bytes(data))


def test_level_parser_preserves_treasure_unlock_percentages() -> None:
    catalog = object.__new__(OriginalGameCatalog)
    catalog.xml_root = ET.fromstring(
        """
        <Levels>
          <Level id="Jungle2" curve1="Jungle2/Jungle2" tfreq="1000">
            <Gun startx="400" starty="300" />
            <TreasurePoint x="28" y="105" dist1="60" />
            <TreasurePoint x="716" y="505" dist1="33" />
            <TreasurePoint x="602" y="366" dist1="84" />
          </Level>
        </Levels>
        """
    )

    definition = catalog._parse_levels()["jungle2"]

    assert definition.treasure_points == (
        (28.0, 105.0),
        (716.0, 505.0),
        (602.0, 366.0),
    )
    assert definition.treasure_point_distances == ((60,), (33,), (84,))


def test_level_parser_binds_zone_fruit_by_adventure_prefix() -> None:
    catalog = object.__new__(OriginalGameCatalog)
    catalog.xml_root = ET.fromstring(
        """
        <Levels>
          <Zone num="1" start="jungle1" boss="boss1" fruit="pineapple" />
          <Zone num="2" start="village1" boss="boss2" fruit="banana" />
          <Level id="Jungle2" curve1="Jungle2/Jungle2">
            <Gun startx="400" starty="300" />
          </Level>
          <Level id="Boss2" curve1="Boss2/Boss2">
            <Gun startx="400" starty="300" />
          </Level>
        </Levels>
        """
    )
    catalog.zones = catalog._parse_zones()

    levels = catalog._parse_levels()

    assert levels["jungle2"].zone_number == 1
    assert levels["jungle2"].fruit_type == "pineapple"
    assert levels["boss2"].zone_number == 2
    assert levels["boss2"].fruit_type == "banana"


def test_installed_jungle1_matches_original_invariants() -> None:
    try:
        root = find_original_installation()
    except OriginalDataError:
        pytest.skip("original game is not installed")
    level = OriginalGameCatalog(root).load_level("Jungle1")
    assert level.definition.display_name == "Shipwrecked!"
    assert level.definition.gun.positions == ((420.0, 290.0),)
    assert len(level.curves) == 1
    curve = level.curves[0]
    assert len(curve.points) == 3_565
    assert curve.total_length == pytest.approx(
        3_582.879828,
        rel=1e-9,
        abs=1e-5,
    )
    assert curve.parameters.colors == 4
    assert curve.parameters.speed == pytest.approx(0.5)
    assert curve.parameters.zuma_score == 1_250
    assert curve.parameters.start_distance_percent == 65
    assert curve.parameters.ball_repeat_chance == 45
    assert curve.parameters.max_single == 2
    assert curve.parameters.skull_rotation_degrees == 75
    assert curve.parameters.zuma_back_distance == 300
    assert curve.parameters.zuma_slow_duration == 1_100
    assert set(map(int, curve.point_flags)) == {0, 1, 3}
    assert set(map(int, curve.priorities)) == {1}
    assert int(curve.in_tunnel.sum()) == 105
    assert int(curve.absolute_anchors.sum()) == 1


def test_installed_fruit_assets_have_exact_logical_geometry_and_main_timing() -> None:
    try:
        root = find_original_installation()
    except OriginalDataError:
        pytest.skip("original game is not installed")
    catalog = OriginalGameCatalog(root)
    expected_frames = {
        "pineapple": 26,
        "banana": 28,
        "cocoa": 26,
        "mango": 26,
        "coconut": 26,
        "acorn": 28,
    }

    for fruit_type, expected in expected_frames.items():
        assets = catalog.fruit_assets(fruit_type)
        assert (assets.logical_width, assets.logical_height) == (52, 52)
        assert assets.collection_animation_frames == expected
        assert assets.collection_animation_fps == 99.0
        assert assets.high_resolution_gif_sha256.startswith("sha256:")
        assert assets.low_resolution_gif_sha256.startswith("sha256:")
        assert assets.collection_animation_sha256.startswith("sha256:")

    jungle2 = catalog.level("Jungle2")
    assert (jungle2.zone_number, jungle2.fruit_type) == (1, "pineapple")


def test_every_referenced_installed_curve_loads_exactly() -> None:
    """Guard the complete Adventure catalog, not only the first level."""

    try:
        root = find_original_installation()
    except OriginalDataError:
        pytest.skip("original game is not installed")
    catalog = OriginalGameCatalog(root)
    assert len(catalog.levels) == 79
    referenced_paths: set[str] = set()
    for level_id in catalog.levels:
        loaded = catalog.load_level(level_id)
        assert loaded.curves
        for curve in loaded.curves:
            assert curve.version == 15
            assert curve.end_waypoint >= 1
            assert curve.source_path is not None
            referenced_paths.add(str(curve.source_path).casefold())
    assert len(referenced_paths) == 89
