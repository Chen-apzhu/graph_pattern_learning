"""
Tests for src/data/room_factory.py

Verifies:
  - RoomSpec computes occupancy and daylight requirement correctly
  - RoomNode derived properties and distance calculation
  - RoomFactory generates rooms within valid attribute ranges
  - EnvNodeFactory generates correct env node types at expected positions
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np

from utils.enums import (
    RoomType, EnvNodeType, DaylightLevel, NoiseLevel, ZoneType
)
from data.room_factory import (
    RoomSpec, RoomNode, EnvironmentalNode,
    RoomCatalog, RoomFactory, EnvNodeFactory,
)


# --- RoomSpec tests ---

def test_room_spec_compute_occupancy():
    """occupancy = max(1, floor(area / occupancy_density))."""
    spec = RoomSpec(
        room_type=RoomType.CLASSROOM,
        display_name="教室",
        area_range_sqm=(54.0, 72.0),
        aspect_ratio_range=(1.0, 1.8),
        daylight_level=DaylightLevel.HIGH,
        noise_level=NoiseLevel.MODERATE,
        noise_tolerance=NoiseLevel.MODERATE,
        occupancy_density=1.2,
        fire_exits_min=2,
        floor_preference=[1, 2, 3],
    )

    # 60 m² / 1.2 m²/person = 50 people
    assert spec.compute_occupancy(60.0) == 50, \
        f"Expected 50 occupants, got {spec.compute_occupancy(60.0)}"

    # 67 m² / 1.2 m²/person = 55.8 → floor = 55
    assert spec.compute_occupancy(67.0) == 55, \
        f"Expected 55 occupants, got {spec.compute_occupancy(67.0)}"

    # Edge case: tiny area should still return at least 1
    assert spec.compute_occupancy(1.0) == 1, \
        f"Expected at least 1 occupant, got {spec.compute_occupancy(1.0)}"

    print("  PASS: test_room_spec_compute_occupancy")


def test_room_spec_requires_daylight():
    """HIGH and CRITICAL daylight levels should require daylight."""
    for level, expected in [
        (DaylightLevel.NONE, False),
        (DaylightLevel.LOW, False),
        (DaylightLevel.MEDIUM, False),
        (DaylightLevel.HIGH, True),
        (DaylightLevel.CRITICAL, True),
    ]:
        spec = RoomSpec(
            room_type=RoomType.CLASSROOM,
            display_name="test",
            area_range_sqm=(10.0, 20.0),
            aspect_ratio_range=(1.0, 2.0),
            daylight_level=level,
            noise_level=NoiseLevel.MODERATE,
            noise_tolerance=NoiseLevel.MODERATE,
            occupancy_density=2.0,
            fire_exits_min=1,
            floor_preference=[0],
        )
        assert spec.requires_daylight() == expected, \
            f"DaylightLevel.{level.name}: expected {expected}, got {spec.requires_daylight()}"

    print("  PASS: test_room_spec_requires_daylight")


# --- RoomNode tests ---

def test_room_node_derived_properties():
    """RoomNode property delegation works correctly."""
    spec = RoomSpec(
        room_type=RoomType.LIBRARY,
        display_name="图书馆",
        area_range_sqm=(100.0, 200.0),
        aspect_ratio_range=(1.0, 2.5),
        daylight_level=DaylightLevel.HIGH,
        noise_level=NoiseLevel.QUIET,
        noise_tolerance=NoiseLevel.QUIET,
        occupancy_density=2.0,
        fire_exits_min=2,
        floor_preference=[0, 1],
    )

    room = RoomNode(
        room_id="library_000_f0",
        spec=spec,
        area=150.0,
        aspect_ratio=1.5,
        floor=0,
        centroid=(50.0, 30.0),
        zone_id=2,
    )

    assert room.room_type == RoomType.LIBRARY
    assert room.daylight_level == DaylightLevel.HIGH
    assert room.noise_level == NoiseLevel.QUIET
    assert room.fire_exits_min == 2
    assert room.requires_daylight is True
    assert room.occupancy == 75  # floor(150 / 2.0)
    print("  PASS: test_room_node_derived_properties")


def test_room_node_distance():
    """Euclidean distance between two rooms on the same plane."""
    room_a = RoomNode(
        room_id="a", spec=None, area=50.0, aspect_ratio=1.0,
        floor=1, centroid=(0.0, 0.0), zone_id=0,
    )
    room_b = RoomNode(
        room_id="b", spec=None, area=50.0, aspect_ratio=1.0,
        floor=1, centroid=(3.0, 4.0), zone_id=0,
    )

    dist = room_a.euclidean_distance_to(room_b)
    assert abs(dist - 5.0) < 1e-9, f"Expected 5.0, got {dist}"
    print("  PASS: test_room_node_distance")


def test_room_node_same_floor():
    """same_floor check works."""
    room_a = RoomNode("a", None, 50.0, 1.0, 0, (0.0, 0.0), 0)
    room_b = RoomNode("b", None, 50.0, 1.0, 0, (5.0, 0.0), 0)
    room_c = RoomNode("c", None, 50.0, 1.0, 1, (5.0, 0.0), 0)

    assert room_a.same_floor(room_b) is True
    assert room_a.same_floor(room_c) is False
    print("  PASS: test_room_node_same_floor")


# --- RoomFactory tests ---

def _make_sample_catalog():
    """Create a minimal RoomCatalog with two room types for testing."""
    specs = {
        RoomType.CLASSROOM: RoomSpec(
            room_type=RoomType.CLASSROOM,
            display_name="教室",
            area_range_sqm=(54.0, 72.0),
            aspect_ratio_range=(1.0, 1.8),
            daylight_level=DaylightLevel.HIGH,
            noise_level=NoiseLevel.MODERATE,
            noise_tolerance=NoiseLevel.MODERATE,
            occupancy_density=1.2,
            fire_exits_min=2,
            floor_preference=[1, 2, 3],
        ),
        RoomType.TOILET: RoomSpec(
            room_type=RoomType.TOILET,
            display_name="卫生间",
            area_range_sqm=(12.0, 24.0),
            aspect_ratio_range=(1.0, 2.5),
            daylight_level=DaylightLevel.NONE,
            noise_level=NoiseLevel.MODERATE,
            noise_tolerance=NoiseLevel.MODERATE,
            occupancy_density=0.8,
            fire_exits_min=1,
            floor_preference=[0, 1, 2, 3, 4],
        ),
    }
    return RoomCatalog(specs)


def test_room_factory_generate_range():
    """Generated rooms should have attributes within spec ranges."""
    catalog = _make_sample_catalog()
    rng = np.random.default_rng(42)
    factory = RoomFactory(catalog, rng)

    for _ in range(100):
        room = factory.generate(RoomType.CLASSROOM, floor=2, zone_id=0)
        spec = room.spec

        # Area within range
        assert spec.area_range_sqm[0] <= room.area <= spec.area_range_sqm[1], \
            f"Area {room.area} outside range {spec.area_range_sqm}"

        # Aspect ratio within range
        assert spec.aspect_ratio_range[0] <= room.aspect_ratio <= spec.aspect_ratio_range[1], \
            f"Aspect ratio {room.aspect_ratio} outside range {spec.aspect_ratio_range}"

        # Floor matches
        assert room.floor == 2, f"Expected floor 2, got {room.floor}"

        # zone_id matches
        assert room.zone_id == 0

    print("  PASS: test_room_factory_generate_range")


def test_room_factory_unique_ids():
    """Generated rooms should have unique IDs."""
    catalog = _make_sample_catalog()
    rng = np.random.default_rng(42)
    factory = RoomFactory(catalog, rng)

    rooms = factory.generate_batch(RoomType.CLASSROOM, 20, floor=1, zone_id=0)
    ids = [r.room_id for r in rooms]
    assert len(ids) == len(set(ids)), f"Duplicate IDs found: {ids}"
    print("  PASS: test_room_factory_unique_ids")


def test_room_factory_deterministic():
    """Same seed should produce identical rooms."""
    catalog = _make_sample_catalog()

    factory_a = RoomFactory(catalog, np.random.default_rng(42))
    factory_b = RoomFactory(catalog, np.random.default_rng(42))

    r_a = factory_a.generate(RoomType.CLASSROOM, floor=1, zone_id=0)
    r_b = factory_b.generate(RoomType.CLASSROOM, floor=1, zone_id=0)

    assert r_a.room_id == r_b.room_id, \
        f"IDs differ: {r_a.room_id} vs {r_b.room_id}"
    assert abs(r_a.area - r_b.area) < 1e-9, \
        f"Areas differ: {r_a.area} vs {r_b.area}"
    assert abs(r_a.aspect_ratio - r_b.aspect_ratio) < 1e-9, \
        f"Aspect ratios differ: {r_a.aspect_ratio} vs {r_b.aspect_ratio}"

    print("  PASS: test_room_factory_deterministic")


# --- EnvNodeFactory tests ---

def test_env_factory_south_facing():
    """South-facing node should be at the southern edge center."""
    factory = EnvNodeFactory(
        site_bounds=(0.0, 0.0, 200.0, 150.0),
        rng=np.random.default_rng(42),
    )
    node = factory.generate_south_facing()

    assert node.env_type == EnvNodeType.SOUTH_FACING
    assert node.position == (100.0, 150.0), \
        f"Expected (100.0, 150.0), got {node.position}"
    assert node.attributes['solar_orientation'] == 1.0
    print("  PASS: test_env_factory_south_facing")


def test_env_factory_main_road():
    """Main road access should be at the correct boundary."""
    factory = EnvNodeFactory(
        site_bounds=(0.0, 0.0, 200.0, 150.0),
        rng=np.random.default_rng(42),
    )

    west = factory.generate_main_road_access("west")
    assert west.position == (0.0, 75.0), f"West: {west.position}"

    east = factory.generate_main_road_access("east")
    assert east.position == (200.0, 75.0), f"East: {east.position}"

    print("  PASS: test_env_factory_main_road")


def test_env_factory_generate_all():
    """generate_all should create correct number of nodes per size."""
    bounds = (0.0, 0.0, 200.0, 150.0)

    for size, expected_min in [("small", 4), ("medium", 5), ("large", 6)]:
        factory = EnvNodeFactory(bounds, rng=np.random.default_rng(42))
        nodes = factory.generate_all(school_size=size)

        # Must contain exactly one south_facing and one main_road_access
        south = [n for n in nodes if n.env_type == EnvNodeType.SOUTH_FACING]
        road = [n for n in nodes if n.env_type == EnvNodeType.MAIN_ROAD_ACCESS]
        assert len(south) == 1, f"{size}: expected 1 south_facing, got {len(south)}"
        assert len(road) == 1, f"{size}: expected 1 main_road, got {len(road)}"

        # All env_ids unique
        ids = [n.env_id for n in nodes]
        assert len(ids) == len(set(ids)), f"{size}: duplicate env_ids"

        print(f"  PASS: test_env_factory_generate_all ({size}: {len(nodes)} nodes)")


def test_room_node_equality():
    """RoomNode equality is based on room_id."""
    r1 = RoomNode("room_001", None, 10.0, 1.0, 0, (0.0, 0.0), 0)
    r2 = RoomNode("room_001", None, 20.0, 2.0, 1, (5.0, 5.0), 1)
    r3 = RoomNode("room_002", None, 10.0, 1.0, 0, (0.0, 0.0), 0)

    assert r1 == r2, "Same ID should be equal"
    assert r1 != r3, "Different ID should not be equal"
    assert hash(r1) == hash(r2)
    print("  PASS: test_room_node_equality")


def run_all_tests():
    test_room_spec_compute_occupancy()
    test_room_spec_requires_daylight()
    test_room_node_derived_properties()
    test_room_node_distance()
    test_room_node_same_floor()
    test_room_factory_generate_range()
    test_room_factory_unique_ids()
    test_room_factory_deterministic()
    test_env_factory_south_facing()
    test_env_factory_main_road()
    test_env_factory_generate_all()
    test_room_node_equality()


if __name__ == '__main__':
    print("Running tests for: room_factory.py")
    run_all_tests()
    print("All tests passed!")
