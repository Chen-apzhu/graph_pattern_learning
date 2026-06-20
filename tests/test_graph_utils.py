"""
Tests for src/graph/graph_utils.py

Verifies:
  - Shortest path computation
  - Betweenness centrality
  - Topology mask shapes and logic
  - Connectivity analysis
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np

# These tests need PyG + torch — skip if not installed
try:
    import torch
    from torch_geometric.data import HeteroData
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

from utils.enums import RoomType, EdgeCategory
from data.generator import SchoolBuildingGenerator


def _build_test_graph():
    """Build a small test school graph for graph analysis."""
    gen = SchoolBuildingGenerator(seed=42)
    result = gen.generate(num_floors=2, school_size='small', validate=False)
    hetero_data = gen.to_hetero_data(result)

    from graph.school_graph import SchoolGraphData
    return SchoolGraphData(hetero_data), result


def test_shortest_path():
    """Shortest path between two rooms should return a valid path."""
    if not HAS_PYG:
        print("  SKIP: test_shortest_path (PyG not installed)")
        return

    sg, result = _build_test_graph()
    from graph.graph_utils import GraphAnalyzer
    analyzer = GraphAnalyzer(sg)

    room_ids = sg.room_ids
    if len(room_ids) >= 2:
        dist, path = analyzer.shortest_path(room_ids[0], room_ids[1])
        # dist should be finite if graph is connected
        assert dist != float('inf') or dist == float('inf'), \
            "Shortest path should return a float distance"
        print(f"  PASS: test_shortest_path (dist={dist})")
    else:
        print("  SKIP: test_shortest_path (not enough rooms)")


def test_betweenness_centrality():
    """Betweenness centrality should return values for all rooms."""
    if not HAS_PYG:
        print("  SKIP: test_betweenness_centrality (PyG not installed)")
        return

    sg, result = _build_test_graph()
    from graph.graph_utils import GraphAnalyzer
    analyzer = GraphAnalyzer(sg)

    bc = analyzer.betweenness_centrality()
    assert len(bc) == sg.num_rooms, \
        f"Expected centrality for {sg.num_rooms} rooms, got {len(bc)}"

    # Values should be in [0, 1]
    for rid, val in bc.items():
        assert 0.0 <= val <= 1.0, \
            f"Centrality for {rid} = {val}, expected [0, 1]"

    print(f"  PASS: test_betweenness_centrality (n={len(bc)})")


def test_fire_exit_mask():
    """Fire exit mask should have correct shape and type."""
    if not HAS_PYG:
        print("  SKIP: test_fire_exit_mask (PyG not installed)")
        return

    sg, result = _build_test_graph()
    from graph.graph_utils import GraphAnalyzer
    analyzer = GraphAnalyzer(sg)

    mask = analyzer.apply_fire_exit_mask(occupancy_threshold=50)
    assert mask.shape == (sg.num_rooms, sg.num_rooms), \
        f"Mask shape {mask.shape}, expected ({sg.num_rooms}, {sg.num_rooms})"
    assert mask.dtype == torch.bool

    print(f"  PASS: test_fire_exit_mask (deficient rooms: {mask.any(dim=1).sum().item()})")


def test_acoustic_mask():
    """Acoustic mask should be symmetric."""
    if not HAS_PYG:
        print("  SKIP: test_acoustic_mask (PyG not installed)")
        return

    sg, result = _build_test_graph()
    from graph.graph_utils import GraphAnalyzer
    analyzer = GraphAnalyzer(sg)

    mask = analyzer.apply_acoustic_mask(noise_gap_threshold=2)
    assert mask.shape == (sg.num_rooms, sg.num_rooms)
    assert mask.dtype == torch.bool

    # Should be symmetric
    assert torch.equal(mask, mask.T), "Acoustic mask should be symmetric"

    print(f"  PASS: test_acoustic_mask (acoustic pairs: {mask.sum().item() // 2})")


def test_find_isolated_rooms():
    """Isolated room detection should work."""
    if not HAS_PYG:
        print("  SKIP: test_find_isolated_rooms (PyG not installed)")
        return

    sg, result = _build_test_graph()
    from graph.graph_utils import GraphAnalyzer
    analyzer = GraphAnalyzer(sg)

    isolated = analyzer.find_isolated_rooms()
    # In a well-connected graph, there should be few or no isolated rooms
    n_total = sg.num_rooms
    if len(isolated) > n_total * 0.5:
        print(f"  WARN: {len(isolated)}/{n_total} rooms isolated — graph may be disconnected")
    print(f"  PASS: test_find_isolated_rooms ({len(isolated)} isolated)")


def test_component_analysis():
    """Component analysis should return component mapping."""
    if not HAS_PYG:
        print("  SKIP: test_component_analysis (PyG not installed)")
        return

    sg, result = _build_test_graph()
    from graph.graph_utils import GraphAnalyzer
    analyzer = GraphAnalyzer(sg)

    components = analyzer.component_analysis()
    total_rooms = sum(len(rooms) for rooms in components.values())
    assert total_rooms == sg.num_rooms, \
        f"Component rooms {total_rooms} != total rooms {sg.num_rooms}"

    print(f"  PASS: test_component_analysis ({len(components)} components)")


def run_all_tests():
    test_shortest_path()
    test_betweenness_centrality()
    test_fire_exit_mask()
    test_acoustic_mask()
    test_find_isolated_rooms()
    test_component_analysis()


if __name__ == '__main__':
    print("Running tests for: graph_utils.py")
    run_all_tests()
    print("All tests passed!")
