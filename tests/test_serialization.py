"""
Tests for src/utils/serialization.py

Verifies:
  - save_hetero_data / load_hetero_data round-trip integrity
  - File creation and metadata embedding
  - Error handling for missing files
"""

import sys
import os
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.serialization import save_hetero_data, load_hetero_data, HAS_PYG


def test_serialization_round_trip():
    """Save and load a minimal HeteroData object; verify data integrity."""
    if not HAS_PYG:
        print("  SKIP: test_serialization_round_trip (PyG not installed)")
        return

    from torch_geometric.data import HeteroData
    import torch

    # Create a minimal HeteroData
    data = HeteroData()
    data['room'].x = torch.randn(5, 27)
    data['room'].num_nodes = 5
    data['environment'].x = torch.randn(3, 6)
    data['environment'].num_nodes = 3
    data['room', 'physical_connects', 'room'].edge_index = torch.tensor([
        [0, 1, 2], [1, 2, 3]
    ])

    metadata_in = {'school_size': 'small', 'seed': 42}

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'test_graph.pt')
        saved_path = save_hetero_data(data, path, metadata=metadata_in)

        assert os.path.exists(saved_path), f"File was not created: {saved_path}"
        assert saved_path.endswith('.pt'), f"File should end with .pt: {saved_path}"

        loaded_data, loaded_meta = load_hetero_data(saved_path)

        # Verify node counts
        assert loaded_data['room'].num_nodes == 5, \
            f"Expected 5 rooms, got {loaded_data['room'].num_nodes}"
        assert loaded_data['environment'].num_nodes == 3, \
            f"Expected 3 env nodes, got {loaded_data['environment'].num_nodes}"

        # Verify features
        assert torch.allclose(loaded_data['room'].x, data['room'].x), \
            "Room features differ after round-trip"
        assert torch.allclose(loaded_data['environment'].x, data['environment'].x), \
            "Env features differ after round-trip"

        # Verify edge index
        assert torch.equal(
            loaded_data['room', 'physical_connects', 'room'].edge_index,
            data['room', 'physical_connects', 'room'].edge_index
        ), "Edge index differs after round-trip"

        # Verify metadata
        assert loaded_meta is not None, "Metadata should not be None"
        assert loaded_meta['school_size'] == 'small', \
            f"Metadata mismatch: {loaded_meta}"
        assert loaded_meta['seed'] == 42

    print("  PASS: test_serialization_round_trip")


def test_save_without_extension():
    """save_hetero_data should append .pt if no extension given."""
    if not HAS_PYG:
        print("  SKIP: test_save_without_extension (PyG not installed)")
        return

    from torch_geometric.data import HeteroData

    data = HeteroData()
    data['room'].x = None
    data['room'].num_nodes = 1

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'no_ext')
        saved = save_hetero_data(data, path)
        assert saved.endswith('.pt'), f"Expected .pt suffix, got: {saved}"
        assert os.path.exists(saved)

    print("  PASS: test_save_without_extension")


def test_load_nonexistent_file():
    """load_hetero_data should raise FileNotFoundError for missing files."""
    if not HAS_PYG:
        print("  SKIP: test_load_nonexistent_file (PyG not installed)")
        return

    try:
        load_hetero_data('/nonexistent/path/graph.pt')
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        pass  # Expected

    print("  PASS: test_load_nonexistent_file")


def test_metadata_optional():
    """Metadata should be optional — save/load works without it."""
    if not HAS_PYG:
        print("  SKIP: test_metadata_optional (PyG not installed)")
        return

    from torch_geometric.data import HeteroData

    data = HeteroData()
    data['room'].x = None
    data['room'].num_nodes = 2

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'no_meta.pt')
        save_hetero_data(data, path)  # No metadata
        _, meta = load_hetero_data(path)
        assert meta is None, f"Expected None metadata, got: {meta}"

    print("  PASS: test_metadata_optional")


def run_all_tests():
    test_serialization_round_trip()
    test_save_without_extension()
    test_load_nonexistent_file()
    test_metadata_optional()


if __name__ == '__main__':
    print("Running tests for: serialization.py")
    run_all_tests()
    print("All tests passed!")
