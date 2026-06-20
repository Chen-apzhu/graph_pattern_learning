"""
Tests for src/utils/enums.py

Verifies:
  - Enum value uniqueness (no duplicate string values)
  - ROOM_TO_ZONE mapping completeness (every RoomType maps to a ZoneType)
  - Ordinal consistency of DaylightLevel / NoiseLevel scales
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.enums import (
    RoomType, EnvNodeType, EdgeCategory,
    DaylightLevel, NoiseLevel, ZoneType, ROOM_TO_ZONE
)


def test_room_types_unique():
    """All RoomType string values should be unique."""
    values = [rt.value for rt in RoomType]
    assert len(values) == len(set(values)), f"Duplicate RoomType values: {values}"
    print("  PASS: test_room_types_unique")


def test_env_types_unique():
    """All EnvNodeType string values should be unique."""
    values = [et.value for et in EnvNodeType]
    assert len(values) == len(set(values)), f"Duplicate EnvNodeType values: {values}"
    print("  PASS: test_env_types_unique")


def test_edge_categories_unique():
    """All EdgeCategory string values should be unique."""
    values = [ec.value for ec in EdgeCategory]
    assert len(values) == len(set(values)), f"Duplicate EdgeCategory values: {values}"
    print("  PASS: test_edge_categories_unique")


def test_room_to_zone_completeness():
    """Every RoomType must have a corresponding ZoneType mapping."""
    for rt in RoomType:
        zone = ROOM_TO_ZONE.get(rt)
        assert zone is not None, f"RoomType {rt} has no ROOM_TO_ZONE mapping"
        assert isinstance(zone, ZoneType), f"RoomType {rt} maps to non-ZoneType: {type(zone)}"
    print("  PASS: test_room_to_zone_completeness")


def test_daylight_level_ordinal():
    """DaylightLevel is ordinal: 0=NONE < 1=LOW < 2=MEDIUM < 3=HIGH < 4=CRITICAL."""
    assert DaylightLevel.NONE < DaylightLevel.LOW
    assert DaylightLevel.LOW < DaylightLevel.MEDIUM
    assert DaylightLevel.MEDIUM < DaylightLevel.HIGH
    assert DaylightLevel.HIGH < DaylightLevel.CRITICAL
    print("  PASS: test_daylight_level_ordinal")


def test_noise_level_ordinal():
    """NoiseLevel is ordinal: 0=QUIET < 1=MODERATE < 2=NOISY < 3=LOUD < 4=VERY_LOUD."""
    assert NoiseLevel.QUIET < NoiseLevel.MODERATE
    assert NoiseLevel.MODERATE < NoiseLevel.NOISY
    assert NoiseLevel.NOISY < NoiseLevel.LOUD
    assert NoiseLevel.LOUD < NoiseLevel.VERY_LOUD
    print("  PASS: test_noise_level_ordinal")


def test_edge_category_values_match_pyg_convention():
    """EdgeCategory values use snake_case (PyG edge type convention)."""
    for ec in EdgeCategory:
        assert "_" in ec.value or ec.value.islower(), \
            f"EdgeCategory {ec} value '{ec.value}' should be snake_case"
    print("  PASS: test_edge_category_values_match_pyg_convention")


def test_room_count():
    """Verify we have exactly 13 room types (as defined in task.md)."""
    assert len(list(RoomType)) == 13, \
        f"Expected 13 RoomTypes, got {len(list(RoomType))}"
    print("  PASS: test_room_count")


def test_env_count():
    """Verify we have exactly 4 environment node types."""
    assert len(list(EnvNodeType)) == 4, \
        f"Expected 4 EnvNodeTypes, got {len(list(EnvNodeType))}"
    print("  PASS: test_env_count")


def test_zone_count():
    """Verify we have exactly 6 zone types."""
    assert len(list(ZoneType)) == 6, \
        f"Expected 6 ZoneTypes, got {len(list(ZoneType))}"
    print("  PASS: test_zone_count")


def run_all_tests():
    test_room_types_unique()
    test_env_types_unique()
    test_edge_categories_unique()
    test_room_to_zone_completeness()
    test_daylight_level_ordinal()
    test_noise_level_ordinal()
    test_edge_category_values_match_pyg_convention()
    test_room_count()
    test_env_count()
    test_zone_count()


if __name__ == '__main__':
    print("Running tests for: enums.py")
    run_all_tests()
    print("All tests passed!")
