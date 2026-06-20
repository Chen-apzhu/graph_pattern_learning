"""
Tests for src/graph/school_graph.py

Verifies:
  - Tensor dimension validation
  - Edge count accessors
  - NetworkX conversion round-trip
  - Summary formatting
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np

try:
    import torch
    from torch_geometric.data import HeteroData
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

from utils.enums import EdgeCategory
from data.generator import SchoolBuildingGenerator


def test_school_graph_construction():
    """Build a SchoolGraphData from a generated HeteroData."""
    if not HAS_PYG:
        print("  SKIP: test_school_graph_construction (PyG not installed)")
        return

    from graph.school_graph import SchoolGraphData

    gen = SchoolBuildingGenerator(seed=42)
    result = gen.generate(num_floors=2, school_size='small', validate=False)
    hetero_data = gen.to_hetero_data(result)

    sg = SchoolGraphData(hetero_data)

    assert sg.num_rooms == len(result.rooms)
    assert sg.num_env_nodes == len(result.env_nodes)
    assert sg.room_features.shape[1] == 27  # ROOM_FEAT_DIM
    assert sg.env_features.shape[1] == 6    # ENV_FEAT_DIM

    print(f"  PASS: test_school_graph_construction ({sg.num_rooms} rooms, {sg.num_env_nodes} env)")


def test_validate_tensor_dimensions():
    """Validation should pass for correctly constructed graphs."""
    if not HAS_PYG:
        print("  SKIP: test_validate_tensor_dimensions (PyG not installed)")
        return

    from graph.school_graph import SchoolGraphData

    gen = SchoolBuildingGenerator(seed=42)
    result = gen.generate(num_floors=2, school_size='small', validate=False)
    hetero_data = gen.to_hetero_data(result)

    sg = SchoolGraphData(hetero_data)
    errors = sg.validate_tensor_dimensions()
    assert len(errors) == 0, f"Should have no errors, got: {errors}"

    print("  PASS: test_validate_tensor_dimensions")


def test_edge_counts():
    """Edge counts should match generation output."""
    if not HAS_PYG:
        print("  SKIP: test_edge_counts (PyG not installed)")
        return

    from graph.school_graph import SchoolGraphData

    gen = SchoolBuildingGenerator(seed=42)
    result = gen.generate(num_floors=2, school_size='small', validate=False)
    hetero_data = gen.to_hetero_data(result)

    sg = SchoolGraphData(hetero_data)
    counts = sg.edge_counts()

    # At least room→room physical edges should exist
    phys_room_key = 'room→physical_connects→room'
    assert counts.get(phys_room_key, 0) > 0, \
        f"No room→room physical edges: {counts}"

    print(f"  PASS: test_edge_counts ({counts})")


def test_to_networkx():
    """NetworkX conversion should preserve node and approximate edge counts."""
    if not HAS_PYG:
        print("  SKIP: test_to_networkx (PyG not installed)")
        return

    from graph.school_graph import SchoolGraphData

    gen = SchoolBuildingGenerator(seed=42)
    result = gen.generate(num_floors=2, school_size='small', validate=False)
    hetero_data = gen.to_hetero_data(result)

    sg = SchoolGraphData(hetero_data)
    G = sg.to_networkx()

    # Node count: rooms + env nodes
    expected_nodes = sg.num_rooms + sg.num_env_nodes
    assert G.number_of_nodes() == expected_nodes, \
        f"NX nodes: {G.number_of_nodes()}, expected: {expected_nodes}"

    # Edge count: NX uses simple graph (multi-edges collapsed)
    # Should have at least physical edges and some sight edges
    assert G.number_of_edges() > 0, "NX graph should have edges"
    # Room→room physical edges should all be present
    phys_count = sg.edge_counts().get('room→physical_connects→room', 0)
    assert G.number_of_edges() >= phys_count, \
        f"NX edges ({G.number_of_edges()}) < physical edges ({phys_count})"

    print(f"  PASS: test_to_networkx ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")


def test_summary():
    """summary() should produce readable output."""
    if not HAS_PYG:
        print("  SKIP: test_summary (PyG not installed)")
        return

    from graph.school_graph import SchoolGraphData

    gen = SchoolBuildingGenerator(seed=42)
    result = gen.generate(num_floors=2, school_size='small', validate=False)
    hetero_data = gen.to_hetero_data(result)

    sg = SchoolGraphData(hetero_data)
    summ = sg.summary()

    assert isinstance(summ, str)
    assert 'Rooms' in summ
    assert 'Env' in summ
    assert 'Edges' in summ

    print("  PASS: test_summary")


def run_all_tests():
    test_school_graph_construction()
    test_validate_tensor_dimensions()
    test_edge_counts()
    test_to_networkx()
    test_summary()


if __name__ == '__main__':
    print("Running tests for: school_graph.py")
    run_all_tests()
    print("All tests passed!")
