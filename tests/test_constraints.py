"""
Tests for src/data/constraints.py

Verifies:
  - Fire exit constraint catches insufficient physical connections
  - Daylight constraint catches rooms without sight lines
  - Acoustic constraint catches inadequate noisy/quiet separation
  - Connectivity constraint catches isolated components
  - Area bounds and circulation ratio checks
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.enums import (
    RoomType, EnvNodeType, EdgeCategory,
    DaylightLevel, NoiseLevel,
)
from data.room_factory import (
    RoomSpec, RoomNode, EnvironmentalNode, RoomCatalog,
)
from data.constraints import ConstraintValidator


def _make_classroom(room_id, occupancy=50, centroid=(10.0, 10.0), typical_floor='ground'):
    """Helper: create a classroom RoomNode."""
    spec = RoomSpec(
        RoomType.CLASSROOM, "教室", (54.0, 72.0), (1.0, 1.8),
        DaylightLevel.HIGH, NoiseLevel.MODERATE, NoiseLevel.MODERATE,
        occupancy_density=1.0,  # 1:1 for easy occupancy calc
        fire_exits_min=2, floor_preference=[0, 1, 2, 3],
    )
    return RoomNode(room_id, spec, float(occupancy), 1.5, 0,
                    floor_range=(0, 0), typical_floor=typical_floor,
                    centroid=centroid, zone_id=0)


def _make_music_room(room_id, centroid=(10.0, 10.0), typical_floor='ground'):
    """Helper: create a music room."""
    spec = RoomSpec(
        RoomType.MUSIC_ROOM, "音乐教室", (72.0, 90.0), (1.0, 1.6),
        DaylightLevel.MEDIUM, NoiseLevel.LOUD, NoiseLevel.QUIET,
        1.5, 2, [0, 1],
    )
    return RoomNode(room_id, spec, 80.0, 1.3, 0,
                    floor_range=(0, 0), typical_floor=typical_floor,
                    centroid=centroid, zone_id=1)


# --- Fire Exit Tests ---

def test_fire_exit_pass():
    """Room with 50 occupancy and 2 physical connections should pass."""
    validator = ConstraintValidator({'fire_safety': {'occupancy_threshold': 50}})
    room = _make_classroom("room_A", occupancy=50)
    edges = [
        ("room_A", "corridor_1", {}),
        ("room_A", "corridor_2", {}),
    ]

    passed, violations = validator.check_fire_exits([room], edges)
    assert passed, f"Expected pass, got: {violations}"
    print("  PASS: test_fire_exit_pass")


def test_fire_exit_fail_insufficient_degree():
    """Room with 50 occupancy but only 1 connection should fail."""
    validator = ConstraintValidator({'fire_safety': {'occupancy_threshold': 50}})
    room = _make_classroom("room_B", occupancy=50)
    edges = [
        ("room_B", "corridor_1", {}),  # only 1 connection
    ]

    passed, violations = validator.check_fire_exits([room], edges)
    assert not passed, "Should fail due to insufficient degree"
    assert len(violations) >= 1
    assert "fire_exits" in violations[0].lower() or "FIRE" in violations[0]
    print("  PASS: test_fire_exit_fail_insufficient_degree")


def test_fire_exit_low_occupancy_ok():
    """Room with 30 occupancy (<50) can have just 1 connection."""
    validator = ConstraintValidator({'fire_safety': {'occupancy_threshold': 50}})
    room = _make_classroom("room_C", occupancy=30)
    edges = [
        ("room_C", "corridor_1", {}),
    ]

    passed, violations = validator.check_fire_exits([room], edges)
    assert passed, f"Low-occupancy room should pass, got: {violations}"
    print("  PASS: test_fire_exit_low_occupancy_ok")


# --- Daylight Tests ---

def test_daylight_pass():
    """Room with HIGH daylight and 1 sight line should pass."""
    validator = ConstraintValidator()
    room = _make_classroom("room_D")
    room.spec = RoomSpec(
        RoomType.CLASSROOM, "教室", (54.0, 72.0), (1.0, 1.8),
        DaylightLevel.HIGH, NoiseLevel.MODERATE, NoiseLevel.MODERATE,
        1.2, 2, [0, 1],
    )
    edges = [
        ("room_D", "south_00", {}),
    ]

    passed, violations = validator.check_daylight_compliance([room], edges)
    assert passed, f"Expected pass, got: {violations}"
    print("  PASS: test_daylight_pass")


def test_daylight_fail():
    """Room with HIGH daylight but 0 sight lines should fail."""
    validator = ConstraintValidator()
    room = _make_classroom("room_E")
    room.spec = RoomSpec(
        RoomType.CLASSROOM, "教室", (54.0, 72.0), (1.0, 1.8),
        DaylightLevel.HIGH, NoiseLevel.MODERATE, NoiseLevel.MODERATE,
        1.2, 2, [0, 1],
    )

    passed, violations = validator.check_daylight_compliance([room], [])
    assert not passed, "Should fail due to no daylight"
    assert "DAYLIGHT" in violations[0]
    print("  PASS: test_daylight_fail")


def test_daylight_low_requirement_ok():
    """Room with LOW daylight doesn't need sight lines."""
    validator = ConstraintValidator()

    spec = RoomSpec(
        RoomType.TOILET, "卫生间", (12.0, 24.0), (1.0, 2.5),
        DaylightLevel.NONE, NoiseLevel.MODERATE, NoiseLevel.MODERATE,
        0.8, 1, [0, 1],
    )
    room = RoomNode("toilet_1", spec, 18.0, 1.5, 0,
                    floor_range=(0, 0), typical_floor='ground',
                    centroid=(10.0, 10.0), zone_id=3)

    passed, violations = validator.check_daylight_compliance([room], [])
    assert passed, f"Low-daylight room should pass, got: {violations}"
    print("  PASS: test_daylight_low_requirement_ok")


# --- Acoustic Tests ---

def test_acoustic_pass_with_block_edge():
    """Music room near classroom WITH acoustic block edge should pass."""
    validator = ConstraintValidator({'acoustic': {
        'noise_gap_threshold': 2,
        'proximity_threshold': 20.0,
        'min_path_distance': 2,
    }})
    music = _make_music_room("music_1", (10.0, 10.0))
    classroom = _make_classroom("class_1", 30, (12.0, 12.0))

    acoustic_edges = [
        ("music_1", "class_1", {'attenuation_db': 55.0}),
    ]
    phys_edges = [
        ("music_1", "corridor", {}),
        ("class_1", "corridor", {}),
    ]

    passed, violations = validator.check_acoustic_separation(
        [music, classroom], acoustic_edges, phys_edges
    )
    assert passed, f"Expected pass with acoustic edge, got: {violations}"
    print("  PASS: test_acoustic_pass_with_block_edge")


def test_acoustic_fail_no_block_near():
    """Music room near classroom WITHOUT acoustic block should fail."""
    validator = ConstraintValidator({'acoustic': {
        'noise_gap_threshold': 2,
        'proximity_threshold': 20.0,
        'min_path_distance': 3,  # require path >= 3
    }})
    music = _make_music_room("music_2", (10.0, 10.0))
    classroom = _make_classroom("class_2", 30, (12.0, 12.0))

    # Both connect to same corridor → path distance = 2 < min_path_distance(3)
    phys_edges = [
        ("music_2", "corridor", {}),
        ("class_2", "corridor", {}),
    ]

    passed, violations = validator.check_acoustic_separation(
        [music, classroom], [], phys_edges  # no acoustic edges
    )
    assert not passed, "Should fail without acoustic block"
    assert "ACOUSTIC" in violations[0]
    print("  PASS: test_acoustic_fail_no_block_near")


def test_acoustic_far_apart_ok():
    """Music room far from classroom may not need acoustic edge."""
    validator = ConstraintValidator({'acoustic': {
        'noise_gap_threshold': 2,
        'proximity_threshold': 5.0,  # small proximity threshold
        'min_path_distance': 2,
    }})
    music = _make_music_room("music_3", (0.0, 0.0))
    classroom = _make_classroom("class_3", 30, (50.0, 50.0))  # far away

    passed, violations = validator.check_acoustic_separation(
        [music, classroom], [], []
    )
    assert passed, f"Far-apart rooms should pass, got: {violations}"
    print("  PASS: test_acoustic_far_apart_ok")


# --- Connectivity Tests ---

def test_connectivity_pass():
    """Fully connected physical graph should pass."""
    validator = ConstraintValidator()

    rooms = [
        _make_classroom("r_A", centroid=(0.0, 0.0)),
        _make_classroom("r_B", centroid=(5.0, 0.0)),
        _make_classroom("r_C", centroid=(10.0, 0.0)),
    ]
    phys_edges = [
        ("r_A", "r_B", {}),
        ("r_B", "r_C", {}),
    ]

    passed, violations = validator.check_connectivity(rooms, phys_edges)
    assert passed, f"Connected graph should pass, got: {violations}"
    print("  PASS: test_connectivity_pass")


def test_connectivity_fail_isolated():
    """A room with no physical connections should fail."""
    validator = ConstraintValidator()

    rooms = [
        _make_classroom("r_A", centroid=(0.0, 0.0)),
        _make_classroom("r_B", centroid=(5.0, 0.0)),
    ]
    phys_edges = [
        ("r_A", "r_A_corridor", {}),  # r_B has no edges
    ]

    passed, violations = validator.check_connectivity(rooms, phys_edges)
    assert not passed, "Should fail due to isolated room"
    print("  PASS: test_connectivity_fail_isolated")


# --- Area Bounds Tests ---

def test_area_bounds_pass():
    """Room area within spec range should pass."""
    validator = ConstraintValidator()
    room = _make_classroom("r_A", occupancy=60)
    room.area = 60.0  # within [54, 72]

    passed, violations = validator.check_area_bounds([room])
    assert passed, f"Should pass, got: {violations}"
    print("  PASS: test_area_bounds_pass")


def test_area_bounds_fail():
    """Room area outside spec range should fail."""
    validator = ConstraintValidator()
    room = _make_classroom("r_A", occupancy=60)
    room.area = 10.0  # outside [54, 72]

    passed, violations = validator.check_area_bounds([room])
    assert not passed, "Should fail due to area out of range"
    print("  PASS: test_area_bounds_fail")


# --- Circulation Ratio ---

def test_circulation_ratio():
    """Verify corridor ratio calculation."""
    validator = ConstraintValidator()

    # 2 classrooms (60 each = 120) + 1 corridor (30) = 150 total
    # corridor ratio = 30/150 = 20%
    spec_class = RoomSpec(
        RoomType.CLASSROOM, "教室", (54.0, 72.0), (1.0, 1.8),
        DaylightLevel.HIGH, NoiseLevel.MODERATE, NoiseLevel.MODERATE,
        1.2, 2, [0, 1],
    )
    spec_corr = RoomSpec(
        RoomType.CORRIDOR, "走道", (12.0, 48.0), (3.0, 12.0),
        DaylightLevel.LOW, NoiseLevel.NOISY, NoiseLevel.LOUD,
        0.3, 2, [0, 1],
    )

    rooms = [
        RoomNode("c1", spec_class, 60.0, 1.5, 0,
                 floor_range=(0, 0), typical_floor='ground',
                 centroid=(0.0, 0.0), zone_id=0),
        RoomNode("c2", spec_class, 60.0, 1.5, 0,
                 floor_range=(0, 0), typical_floor='ground',
                 centroid=(10.0, 0.0), zone_id=0),
        RoomNode("corr", spec_corr, 30.0, 6.0, 0,
                 floor_range=(0, 0), typical_floor='ground',
                 centroid=(5.0, 5.0), zone_id=4),
    ]

    passed, violations = validator.check_circulation_ratio(rooms)
    assert passed, f"20% ratio should pass, got: {violations}"
    print("  PASS: test_circulation_ratio")


# --- Master Validation ---

def test_validate_all_success():
    """A fully valid graph should pass all constraints."""
    validator = ConstraintValidator({
        'fire_safety': {'occupancy_threshold': 50},
        'acoustic': {'noise_gap_threshold': 2, 'proximity_threshold': 10.0, 'min_path_distance': 2},
    })

    room = _make_classroom("room_X", occupancy=50, centroid=(10.0, 10.0))

    all_edges = {
        EdgeCategory.PHYSICAL_CONNECTS: [
            ("room_X", "corr_1", {'distance_weight': 5.0}),
            ("room_X", "corr_2", {'distance_weight': 8.0}),
        ],
        EdgeCategory.ACOUSTIC_BLOCKS: [],
        EdgeCategory.SIGHT_LINES: [
            ("room_X", "south_node", {'orientation_preference': 0.8}),
        ],
    }

    results = validator.validate_all([room], [], all_edges)
    passed = ConstraintValidator.hard_constraints_passed(results)

    if not passed:
        print(f"  FAILED constraints: {ConstraintValidator.format_violations(results)}")

    assert passed, "All hard constraints should pass"
    print("  PASS: test_validate_all_success")


# --- Area Completeness Tests ---

def _make_room_for_area_test(room_id, room_type, area, floor=0, typical_floor='ground', centroid=(10.0, 5.0)):
    """Helper for area completeness tests."""
    spec = RoomSpec(
        room_type, str(room_type.value), (50.0, 600.0), (1.0, 2.0),
        DaylightLevel.HIGH, NoiseLevel.MODERATE, NoiseLevel.MODERATE,
        1.0, 2, [0, 1, 2],
    )
    return RoomNode(room_id, spec, area, 1.5, floor,
                    floor_range=(floor, floor), typical_floor=typical_floor,
                    centroid=centroid, zone_id=0)


def test_area_completeness_pass():
    """Each typical floor within tolerance of per_floor_area * num_spanned."""
    validator = ConstraintValidator()
    # Each room has num_floors_spanned=1 (single physical floor)
    rooms = [
        _make_room_for_area_test("r_g1", RoomType.CLASSROOM, 500.0, floor=0, typical_floor='ground'),
        _make_room_for_area_test("r_g2", RoomType.CORRIDOR, 500.0, floor=0, typical_floor='ground'),
        _make_room_for_area_test("r_t1", RoomType.CLASSROOM, 980.0, floor=1, typical_floor='teaching'),
        _make_room_for_area_test("r_p1", RoomType.CLASSROOM, 500.0, floor=2, typical_floor='top'),
        _make_room_for_area_test("r_p2", RoomType.CORRIDOR, 500.0, floor=2, typical_floor='top'),
    ]
    per_floor = 1000.0
    passed, violations = validator.check_area_completeness(rooms, num_floors=3, per_floor_area=per_floor)
    assert passed, f"Should pass. Violations: {violations}"
    print("  PASS: test_area_completeness_pass")


def test_area_completeness_fail_under():
    """Total area far below budget should fail."""
    validator = ConstraintValidator({'building_footprint': {'area_tolerance': 0.05}})
    rooms = [
        _make_room_for_area_test("r1", RoomType.CLASSROOM, 350.0, typical_floor='ground'),
        _make_room_for_area_test("r2", RoomType.CLASSROOM, 350.0, typical_floor='ground'),
    ]
    per_floor = 1000.0
    passed, violations = validator.check_area_completeness(rooms, num_floors=3, per_floor_area=per_floor)
    assert not passed, f"Total={700:.0f}, budget={per_floor}, deviation=30% should fail"
    assert len(violations) > 0
    print("  PASS: test_area_completeness_fail_under")


def test_area_completeness_fail_over():
    """Total area far above budget should fail."""
    validator = ConstraintValidator({'building_footprint': {'area_tolerance': 0.05}})
    rooms = [
        _make_room_for_area_test("r1", RoomType.CLASSROOM, 700.0, typical_floor='ground'),
        _make_room_for_area_test("r2", RoomType.CLASSROOM, 600.0, typical_floor='ground'),
    ]
    per_floor = 1000.0
    passed, violations = validator.check_area_completeness(rooms, num_floors=3, per_floor_area=per_floor)
    assert not passed, f"Total={1300:.0f}, budget={per_floor}, deviation=30% should fail"
    assert len(violations) > 0
    print("  PASS: test_area_completeness_fail_over")


def test_area_completeness_per_floor_check():
    """Verify per-typical-floor checking catches one floor deviation."""
    validator = ConstraintValidator({'building_footprint': {'area_tolerance': 0.05}})
    per_floor = 500.0
    rooms = [
        # Ground floor: total = 500 (OK)
        _make_room_for_area_test("r_g1", RoomType.CLASSROOM, 300.0, floor=0, typical_floor='ground'),
        _make_room_for_area_test("r_g2", RoomType.CORRIDOR, 200.0, floor=0, typical_floor='ground'),
        # Teaching floor: total = 300 (60% of budget → FAIL)
        _make_room_for_area_test("r_t1", RoomType.CLASSROOM, 200.0, floor=1, typical_floor='teaching'),
        _make_room_for_area_test("r_t2", RoomType.CORRIDOR, 100.0, floor=1, typical_floor='teaching'),
    ]
    passed, violations = validator.check_area_completeness(rooms, num_floors=2, per_floor_area=per_floor)
    assert not passed, "Teaching floor under-filled should fail per-floor check"
    print("  PASS: test_area_completeness_per_floor_check")


def run_all_tests():
    test_fire_exit_pass()
    test_fire_exit_fail_insufficient_degree()
    test_fire_exit_low_occupancy_ok()
    test_daylight_pass()
    test_daylight_fail()
    test_daylight_low_requirement_ok()
    test_acoustic_pass_with_block_edge()
    test_acoustic_fail_no_block_near()
    test_acoustic_far_apart_ok()
    test_connectivity_pass()
    test_connectivity_fail_isolated()
    test_area_bounds_pass()
    test_area_bounds_fail()
    test_circulation_ratio()
    test_validate_all_success()
    test_area_completeness_pass()
    test_area_completeness_fail_under()
    test_area_completeness_fail_over()
    test_area_completeness_per_floor_check()


if __name__ == '__main__':
    print("Running tests for: constraints.py")
    run_all_tests()
    print("All tests passed!")
