"""
Tests for src/data/topology_rules.py

Verifies:
  - Each rule produces correct edge types
  - Physical connection rules connect rooms to corridors
  - Acoustic rules trigger when noise gap exceeds threshold
  - Daylight rules connect high-requirement rooms to south node
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import math
import numpy as np

from utils.enums import (
    RoomType, EnvNodeType, EdgeCategory,
    DaylightLevel, NoiseLevel, ZoneType,
)
from data.room_factory import (
    RoomSpec, RoomNode, EnvironmentalNode, RoomCatalog, RoomFactory, EnvNodeFactory,
)
from data.topology_rules import TopologyRuleEngine


def _make_test_rooms():
    """Create a single-floor test school with rooms at known positions."""
    catalog = RoomCatalog({
        RoomType.CLASSROOM: RoomSpec(
            RoomType.CLASSROOM, "教室", (54.0, 72.0), (1.0, 1.8),
            DaylightLevel.HIGH, NoiseLevel.MODERATE, NoiseLevel.MODERATE,
            1.2, 2, [0, 1, 2, 3],
        ),
        RoomType.MUSIC_ROOM: RoomSpec(
            RoomType.MUSIC_ROOM, "音乐教室", (72.0, 90.0), (1.0, 1.6),
            DaylightLevel.MEDIUM, NoiseLevel.LOUD, NoiseLevel.QUIET,
            1.5, 2, [0, 1],
        ),
        RoomType.CORRIDOR: RoomSpec(
            RoomType.CORRIDOR, "走道", (12.0, 48.0), (3.0, 12.0),
            DaylightLevel.LOW, NoiseLevel.NOISY, NoiseLevel.LOUD,
            0.3, 2, [0, 1, 2, 3, 4],
        ),
        RoomType.TOILET: RoomSpec(
            RoomType.TOILET, "卫生间", (12.0, 24.0), (1.0, 2.5),
            DaylightLevel.NONE, NoiseLevel.MODERATE, NoiseLevel.MODERATE,
            0.8, 1, [0, 1, 2, 3, 4],
        ),
        RoomType.STAIRCASE: RoomSpec(
            RoomType.STAIRCASE, "楼梯间", (18.0, 30.0), (1.0, 2.0),
            DaylightLevel.NONE, NoiseLevel.MODERATE, NoiseLevel.LOUD,
            0.5, 2, [0, 1, 2, 3, 4],
        ),
        RoomType.ENTRANCE_HALL: RoomSpec(
            RoomType.ENTRANCE_HALL, "门厅", (40.0, 80.0), (1.0, 3.0),
            DaylightLevel.MEDIUM, NoiseLevel.NOISY, NoiseLevel.LOUD,
            0.5, 2, [0],
        ),
    })

    # Positions: Y+ = south
    rooms = [
        # Classrooms along south side (good daylight)
        RoomNode("classroom_000_f0", catalog.get(RoomType.CLASSROOM),
                 60.0, 1.5, 0, (30.0, 120.0), zone_id=0),
        RoomNode("classroom_001_f0", catalog.get(RoomType.CLASSROOM),
                 60.0, 1.5, 0, (50.0, 120.0), zone_id=0),

        # Music room on north side (isolated)
        RoomNode("music_000_f0", catalog.get(RoomType.MUSIC_ROOM),
                 80.0, 1.3, 0, (30.0, 20.0), zone_id=1),

        # Corridor running east-west in the middle
        RoomNode("corridor_000_f0", catalog.get(RoomType.CORRIDOR),
                 40.0, 8.0, 0, (40.0, 60.0), zone_id=4),

        # Toilet near music room
        RoomNode("toilet_000_f0", catalog.get(RoomType.TOILET),
                 18.0, 1.5, 0, (10.0, 20.0), zone_id=3),

        # Staircase
        RoomNode("staircase_000_f0", catalog.get(RoomType.STAIRCASE),
                 24.0, 1.5, 0, (70.0, 60.0), zone_id=3),

        # Entrance hall at west edge
        RoomNode("entrance_000_f0", catalog.get(RoomType.ENTRANCE_HALL),
                 60.0, 2.0, 0, (5.0, 60.0), zone_id=2),
    ]
    return rooms


def _make_test_env_nodes():
    return [
        EnvironmentalNode("south_00", EnvNodeType.SOUTH_FACING,
                          (40.0, 150.0), {'solar_orientation': 1.0}),
        EnvironmentalNode("road_00", EnvNodeType.MAIN_ROAD_ACCESS,
                          (0.0, 60.0), {'access_type': 1.0}),
        EnvironmentalNode("green_00", EnvNodeType.GREEN_SPACE,
                          (0.0, 150.0), {'view_quality': 0.8}),
    ]


def test_connect_rooms_to_corridors():
    """Every non-corridor room should connect to nearest corridor."""
    engine = TopologyRuleEngine()
    rooms = _make_test_rooms()
    env_nodes = _make_test_env_nodes()

    edges = engine.connect_rooms_to_corridors(rooms, env_nodes)

    non_corridor_count = sum(1 for r in rooms if r.room_type != RoomType.CORRIDOR)
    assert len(edges) == non_corridor_count, \
        f"Expected {non_corridor_count} edges, got {len(edges)}"

    # Each edge should be (room → corridor)
    corridor_ids = {r.room_id for r in rooms if r.room_type == RoomType.CORRIDOR}
    for src, dst, attrs in edges:
        assert dst in corridor_ids, \
            f"Edge {src}→{dst}: destination should be a corridor"
        assert 'distance_weight' in attrs

    print("  PASS: test_connect_rooms_to_corridors")


def test_connect_corridor_network():
    """Corridors on the same floor should form a chain."""
    engine = TopologyRuleEngine()
    rooms = _make_test_rooms()
    env_nodes = _make_test_env_nodes()

    edges = engine.connect_corridor_network(rooms, env_nodes)

    # With 1 corridor, 0 edges expected; with 2+, N-1 edges
    n_corridors = sum(1 for r in rooms if r.room_type == RoomType.CORRIDOR)
    assert len(edges) == max(0, n_corridors - 1), \
        f"Expected {max(0, n_corridors - 1)} edges, got {len(edges)}"

    print("  PASS: test_connect_corridor_network")


def test_connect_staircases():
    """Staircases should connect to corridors."""
    engine = TopologyRuleEngine()
    rooms = _make_test_rooms()
    env_nodes = _make_test_env_nodes()

    edges = engine.connect_staircases_to_corridors(rooms, env_nodes)

    n_stairs = sum(1 for r in rooms if r.room_type == RoomType.STAIRCASE)
    assert len(edges) == n_stairs

    for _src, _dst, attrs in edges:
        assert attrs['is_stair_connection'] == 1.0

    print("  PASS: test_connect_staircases")


def test_entrance_connects_to_road():
    """Entrance hall should connect to main road access env node."""
    engine = TopologyRuleEngine()
    rooms = _make_test_rooms()
    env_nodes = _make_test_env_nodes()

    edges = engine.connect_entrance_to_corridor_and_road(rooms, env_nodes)

    # Should have: 1 corridor connection + 1 road connection
    road_edges = [e for e in edges if e[1] == 'road_00']
    assert len(road_edges) == 1, \
        f"Expected 1 road connection, got {len(road_edges)}"
    assert road_edges[0][2]['access_type'] == 1.0

    print("  PASS: test_entrance_connects_to_road")


def test_acoustic_separation_triggered():
    """Music room (LOUD) near classroom (tolerance=MODERATE) should get acoustic edge."""
    # noise_gap = LOUD(3) - MODERATE(1) = 2 >= threshold(2)
    engine = TopologyRuleEngine({'acoustic': {
        'noise_gap_threshold': 2,
        'proximity_threshold': 100.0,  # large to ensure they're "close"
        'default_attenuation_db': 45.0,
        'music_classroom_attenuation_db': 55.0,
    }})
    rooms = _make_test_rooms()
    env_nodes = _make_test_env_nodes()

    edges = engine.acoustic_separation_noisy_quiet(rooms, env_nodes)

    # The music room and classrooms are within proximity → should have acoustic edge
    assert len(edges) > 0, f"Expected at least 1 acoustic edge, got {len(edges)}"

    # Check music→classroom edge has higher attenuation
    music_classroom_edge = False
    for src, dst, attrs in edges:
        if ('music' in src and 'classroom' in dst) or ('music' in dst and 'classroom' in src):
            music_classroom_edge = True
            assert attrs['attenuation_db'] == 55.0, \
                f"Music↔classroom should get 55dB, got {attrs['attenuation_db']}"

    assert music_classroom_edge, "No music↔classroom acoustic edge found"

    print("  PASS: test_acoustic_separation_triggered")


def test_acoustic_separation_not_triggered():
    """Two quiet rooms should NOT get acoustic edges."""
    engine = TopologyRuleEngine({'acoustic': {
        'noise_gap_threshold': 2,
        'proximity_threshold': 20.0,
    }})

    # Create two classrooms (MODERATE noise, MODERATE tolerance)
    spec = RoomSpec(
        RoomType.CLASSROOM, "教室", (54.0, 72.0), (1.0, 1.8),
        DaylightLevel.HIGH, NoiseLevel.MODERATE, NoiseLevel.MODERATE,
        1.2, 2, [0, 1],
    )
    rooms = [
        RoomNode("c1", spec, 60.0, 1.5, 0, (10.0, 10.0), 0),
        RoomNode("c2", spec, 60.0, 1.5, 0, (15.0, 10.0), 0),
    ]
    env_nodes = []

    edges = engine.acoustic_separation_noisy_quiet(rooms, env_nodes)
    # noise_gap = 1 - 1 = 0 < threshold(2) → no edges
    assert len(edges) == 0, f"Expected 0 acoustic edges, got {len(edges)}"

    print("  PASS: test_acoustic_separation_not_triggered")


def test_daylight_connections():
    """Classrooms (HIGH daylight) should connect to south node."""
    engine = TopologyRuleEngine()
    rooms = _make_test_rooms()
    env_nodes = _make_test_env_nodes()

    edges = engine.daylight_connection_high_requirement(rooms, env_nodes)

    # classrooms require daylight
    n_high = sum(1 for r in rooms if r.requires_daylight)
    assert len(edges) == n_high, \
        f"Expected {n_high} daylight edges, got {len(edges)}"

    # All edges should go to south_facing env node
    for src, dst, attrs in edges:
        assert dst == 'south_00', f"Expected dst=south_00, got {dst}"
        assert 'orientation_preference' in attrs
        assert -1.0 <= attrs['orientation_preference'] <= 1.0

    print("  PASS: test_daylight_connections")


def test_apply_all_rules():
    """Full rule application returns edges keyed by EdgeCategory."""
    engine = TopologyRuleEngine()
    rooms = _make_test_rooms()
    env_nodes = _make_test_env_nodes()

    result = engine.apply_all_rules(rooms, env_nodes)

    # Check that all three edge categories are present
    assert EdgeCategory.PHYSICAL_CONNECTS in result
    assert EdgeCategory.ACOUSTIC_BLOCKS in result
    assert EdgeCategory.SIGHT_LINES in result

    # Physical edges should be the most numerous
    phys_count = len(result[EdgeCategory.PHYSICAL_CONNECTS])
    assert phys_count > 0, "Should have physical edges"

    # Sight lines should exist for high-daylight rooms
    sight_count = len(result[EdgeCategory.SIGHT_LINES])
    assert sight_count > 0, "Should have sight lines"

    print(f"  PASS: test_apply_all_rules (phys={phys_count}, acous={len(result[EdgeCategory.ACOUSTIC_BLOCKS])}, sight={sight_count})")


def test_library_green_connection():
    """Library should connect to green space."""
    engine = TopologyRuleEngine()

    spec = RoomSpec(
        RoomType.LIBRARY, "图书馆", (100.0, 200.0), (1.0, 2.5),
        DaylightLevel.HIGH, NoiseLevel.QUIET, NoiseLevel.QUIET,
        2.0, 2, [0],
    )
    rooms = [
        RoomNode("lib_000", spec, 150.0, 2.0, 0, (50.0, 50.0), 0),
    ]
    env_nodes = [
        EnvironmentalNode("green_00", EnvNodeType.GREEN_SPACE,
                          (10.0, 10.0), {'view_quality': 0.9}),
        EnvironmentalNode("green_01", EnvNodeType.GREEN_SPACE,
                          (90.0, 90.0), {'view_quality': 0.7}),
    ]

    edges = engine.library_green_space_connection(rooms, env_nodes)
    assert len(edges) == 2, f"Library should connect to both green spaces, got {len(edges)}"

    print("  PASS: test_library_green_connection")


def run_all_tests():
    test_connect_rooms_to_corridors()
    test_connect_corridor_network()
    test_connect_staircases()
    test_entrance_connects_to_road()
    test_acoustic_separation_triggered()
    test_acoustic_separation_not_triggered()
    test_daylight_connections()
    test_apply_all_rules()
    test_library_green_connection()


if __name__ == '__main__':
    print("Running tests for: topology_rules.py")
    run_all_tests()
    print("All tests passed!")
