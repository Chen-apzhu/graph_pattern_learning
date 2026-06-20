"""
Tests for src/graph/graph_stats.py

Verifies:
  - Summary statistics are correctly computed
  - Room type distribution matches generation
  - Floor distribution is reasonable
  - Daylight and acoustic compliance rates are in [0, 1]
  - Report formatting
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np

try:
    import torch
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

from utils.enums import EdgeCategory
from data.generator import SchoolBuildingGenerator


def test_graph_stats_summary():
    """Summary dict should contain all expected keys."""
    if not HAS_PYG:
        print("  SKIP: test_graph_stats_summary (PyG not installed)")
        return

    from graph.school_graph import SchoolGraphData
    from graph.graph_stats import GraphStats

    gen = SchoolBuildingGenerator(seed=42)
    result = gen.generate(num_floors=2, school_size='small', validate=False)
    hetero_data = gen.to_hetero_data(result)
    sg = SchoolGraphData(hetero_data)

    stats = GraphStats(sg)
    summary = stats.summary()

    expected_keys = [
        'num_rooms', 'num_env_nodes',
        'num_physical_edges', 'num_acoustic_edges', 'num_sight_edges',
        'avg_degree_physical', 'density_physical',
        'num_connected_components_physical',
        'avg_clustering', 'avg_path_length', 'diameter',
        'graph_is_connected',
    ]
    for key in expected_keys:
        assert key in summary, f"Missing key '{key}' in summary"

    assert summary['num_rooms'] == sg.num_rooms
    assert summary['num_env_nodes'] == sg.num_env_nodes
    assert summary['num_physical_edges'] >= 0

    print(f"  PASS: test_graph_stats_summary (connected={summary['graph_is_connected']})")


def test_room_type_distribution():
    """Room type counts should sum to total rooms."""
    if not HAS_PYG:
        print("  SKIP: test_room_type_distribution (PyG not installed)")
        return

    from graph.school_graph import SchoolGraphData
    from graph.graph_stats import GraphStats

    gen = SchoolBuildingGenerator(seed=42)
    result = gen.generate(num_floors=2, school_size='small', validate=False)
    hetero_data = gen.to_hetero_data(result)
    sg = SchoolGraphData(hetero_data)

    stats = GraphStats(sg)
    dist = stats.room_type_distribution()

    total = sum(dist.values())
    assert total == sg.num_rooms, \
        f"Distribution total {total} != {sg.num_rooms}"

    # Classrooms should exist
    assert dist.get('classroom', 0) > 0, "Should have classrooms"

    print(f"  PASS: test_room_type_distribution ({dist})")


def test_floor_distribution():
    """Floor distribution should cover the generated floors."""
    if not HAS_PYG:
        print("  SKIP: test_floor_distribution (PyG not installed)")
        return

    from graph.school_graph import SchoolGraphData
    from graph.graph_stats import GraphStats

    gen = SchoolBuildingGenerator(seed=42)
    result = gen.generate(num_floors=2, school_size='small', validate=False)
    hetero_data = gen.to_hetero_data(result)
    sg = SchoolGraphData(hetero_data)

    stats = GraphStats(sg)
    dist = stats.floor_distribution()

    total = sum(dist.values())
    assert total == sg.num_rooms

    print(f"  PASS: test_floor_distribution ({dist})")


def test_daylight_compliance_rate():
    """Daylight compliance should be in [0, 1]."""
    if not HAS_PYG:
        print("  SKIP: test_daylight_compliance_rate (PyG not installed)")
        return

    from graph.school_graph import SchoolGraphData
    from graph.graph_stats import GraphStats

    gen = SchoolBuildingGenerator(seed=42)
    result = gen.generate(num_floors=2, school_size='small', validate=False)
    hetero_data = gen.to_hetero_data(result)
    sg = SchoolGraphData(hetero_data)

    stats = GraphStats(sg)
    rate = stats.daylight_compliance_rate()

    assert 0.0 <= rate <= 1.0, f"Rate {rate} not in [0, 1]"

    print(f"  PASS: test_daylight_compliance_rate (rate={rate:.1%})")


def test_acoustic_separation_rate():
    """Acoustic separation rate should be in [0, 1]."""
    if not HAS_PYG:
        print("  SKIP: test_acoustic_separation_rate (PyG not installed)")
        return

    from graph.school_graph import SchoolGraphData
    from graph.graph_stats import GraphStats

    gen = SchoolBuildingGenerator(seed=42)
    result = gen.generate(num_floors=2, school_size='small', validate=False)
    hetero_data = gen.to_hetero_data(result)
    sg = SchoolGraphData(hetero_data)

    stats = GraphStats(sg)
    rate = stats.acoustic_separation_rate()

    assert 0.0 <= rate <= 1.0, f"Rate {rate} not in [0, 1]"

    print(f"  PASS: test_acoustic_separation_rate (rate={rate:.1%})")


def test_report():
    """report() should produce a formatted string."""
    if not HAS_PYG:
        print("  SKIP: test_report (PyG not installed)")
        return

    from graph.school_graph import SchoolGraphData
    from graph.graph_stats import GraphStats

    gen = SchoolBuildingGenerator(seed=42)
    result = gen.generate(num_floors=2, school_size='small', validate=False)
    hetero_data = gen.to_hetero_data(result)
    sg = SchoolGraphData(hetero_data)

    stats = GraphStats(sg)
    report = stats.report()

    assert isinstance(report, str)
    assert len(report) > 0
    assert 'STATISTICS' in report

    print(f"  PASS: test_report ({len(report)} chars)")


def run_all_tests():
    test_graph_stats_summary()
    test_room_type_distribution()
    test_floor_distribution()
    test_daylight_compliance_rate()
    test_acoustic_separation_rate()
    test_report()


if __name__ == '__main__':
    print("Running tests for: graph_stats.py")
    run_all_tests()
    print("All tests passed!")
