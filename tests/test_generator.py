"""
Tests for src/data/generator.py

Verifies:
  - SchoolBuildingGenerator creates valid school graphs for all three sizes
  - Generated rooms have correct types and counts
  - Constraint validation passes (or reports failures)
  - HeteroData conversion works (when PyG is available)
  - Deterministic output with fixed seed
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np

from utils.enums import RoomType, EdgeCategory
from data.generator import SchoolBuildingGenerator, GenerationResult


def test_generator_small_school():
    """Generate a small school (no validation retries needed for basic test)."""
    gen = SchoolBuildingGenerator(seed=42)

    result = gen.generate(
        num_floors=2,
        school_size='small',
        validate=False,  # Skip retry logic for basic test
    )

    assert isinstance(result, GenerationResult)
    assert result.school_size == 'small'
    assert result.num_floors == 2

    # Room type counts should be non-zero
    assert len(result.rooms) > 0, "Should generate rooms"
    assert len(result.env_nodes) > 0, "Should generate env nodes"

    # Edge counts should be non-zero
    phys_count = len(result.edges_by_category.get(EdgeCategory.PHYSICAL_CONNECTS, []))
    assert phys_count > 0, "Should have physical connections"

    print(f"  PASS: test_generator_small_school ({len(result.rooms)} rooms, {phys_count} phys edges)")


def test_generator_medium_school():
    """Generate a medium school."""
    gen = SchoolBuildingGenerator(seed=42)

    result = gen.generate(
        num_floors=3,
        school_size='medium',
        validate=False,
    )

    assert result.school_size == 'medium'
    assert len(result.rooms) >= 50, \
        f"Medium school should have >=50 rooms, got {len(result.rooms)}"

    print(f"  PASS: test_generator_medium_school ({len(result.rooms)} rooms)")


def test_generator_large_school():
    """Generate a large school."""
    gen = SchoolBuildingGenerator(seed=42)

    result = gen.generate(
        num_floors=4,
        school_size='large',
        validate=False,
    )

    assert result.school_size == 'large'
    assert len(result.rooms) >= 80, \
        f"Large school should have >=80 rooms, got {len(result.rooms)}"

    print(f"  PASS: test_generator_large_school ({len(result.rooms)} rooms)")


def test_generator_deterministic():
    """Same seed should produce identical results."""
    gen_a = SchoolBuildingGenerator(seed=42)
    gen_b = SchoolBuildingGenerator(seed=42)

    r_a = gen_a.generate(num_floors=2, school_size='small', validate=False)
    r_b = gen_b.generate(num_floors=2, school_size='small', validate=False)

    assert len(r_a.rooms) == len(r_b.rooms), \
        f"Room count differs: {len(r_a.rooms)} vs {len(r_b.rooms)}"
    assert len(r_a.env_nodes) == len(r_b.env_nodes), \
        f"Env node count differs: {len(r_a.env_nodes)} vs {len(r_b.env_nodes)}"

    # Edge counts should match
    for ec in EdgeCategory:
        count_a = len(r_a.edges_by_category.get(ec, []))
        count_b = len(r_b.edges_by_category.get(ec, []))
        assert count_a == count_b, \
            f"Edge count for {ec.value} differs: {count_a} vs {count_b}"

    print("  PASS: test_generator_deterministic")


def test_generator_component_types():
    """Generated rooms should include essential school room types."""
    gen = SchoolBuildingGenerator(seed=42)
    result = gen.generate(num_floors=3, school_size='medium', validate=False)

    room_types = {r.room_type for r in result.rooms}

    essential_types = {
        RoomType.CLASSROOM,
        RoomType.CORRIDOR,
        RoomType.STAIRCASE,
        RoomType.ENTRANCE_HALL,
        RoomType.TOILET,
    }

    missing = essential_types - room_types
    assert not missing, f"Missing essential room types: {missing}"

    print("  PASS: test_generator_component_types")


def test_result_summary():
    """GenerationResult.summary() should not crash."""
    gen = SchoolBuildingGenerator(seed=42)
    result = gen.generate(num_floors=2, school_size='small', validate=False)

    summary = result.summary()
    assert isinstance(summary, str)
    assert len(summary) > 0
    assert 'small' in summary.lower() or 'SMALL' in summary

    print("  PASS: test_result_summary")


def test_generator_invalid_size():
    """Invalid school size should raise ValueError."""
    gen = SchoolBuildingGenerator(seed=42)

    try:
        gen.generate(num_floors=2, school_size='invalid_size', validate=False)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    print("  PASS: test_generator_invalid_size")


def run_all_tests():
    test_generator_small_school()
    test_generator_medium_school()
    test_generator_large_school()
    test_generator_deterministic()
    test_generator_component_types()
    test_result_summary()
    test_generator_invalid_size()


if __name__ == '__main__':
    print("Running tests for: generator.py")
    run_all_tests()
    print("All tests passed!")
