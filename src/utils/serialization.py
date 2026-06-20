"""
Serialization helpers for PyG HeteroData objects.

Provides save/load functionality using torch.save and torch.load,
with optional compression and metadata tracking.
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, Dict

import torch

# Try importing PyG — gracefully degrade if not installed yet
try:
    from torch_geometric.data import HeteroData
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    HeteroData = None  # type: ignore


def save_hetero_data(
    data: "HeteroData",
    filepath: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Save a PyG HeteroData object to disk using torch.save.

    The file is saved with a .pt extension. An optional metadata dict
    (e.g., generation parameters, timestamp) is stored alongside the data.

    Args:
        data: PyG HeteroData object to save.
        filepath: Destination path. If no .pt extension, it is appended.
        metadata: Optional dictionary of metadata to embed in the saved file.

    Returns:
        The resolved filepath (with .pt extension).

    Raises:
        ImportError: If PyG is not installed.
    """
    if not HAS_PYG:
        raise ImportError(
            "PyTorch Geometric is required for serialization. "
            "Install with: pip install torch-geometric"
        )

    filepath = Path(filepath)
    if filepath.suffix != '.pt':
        filepath = filepath.with_suffix('.pt')

    # Build the save bundle
    bundle = {
        'hetero_data': data,
        'saved_at': datetime.now().isoformat(),
    }
    if metadata:
        bundle['metadata'] = metadata

    filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, str(filepath))

    return str(filepath)


def load_hetero_data(filepath: str) -> tuple:
    """
    Load a PyG HeteroData object from disk.

    Args:
        filepath: Path to the saved .pt file.

    Returns:
        Tuple of (HeteroData, metadata_dict). metadata_dict may be None
        if no metadata was saved.

    Raises:
        ImportError: If PyG is not installed.
        FileNotFoundError: If the file does not exist.
    """
    if not HAS_PYG:
        raise ImportError(
            "PyTorch Geometric is required for deserialization. "
            "Install with: pip install torch-geometric"
        )

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    bundle = torch.load(str(filepath), weights_only=False)

    data = bundle.get('hetero_data')
    metadata = bundle.get('metadata', None)

    return data, metadata


def save_school_graph(school_graph, filepath: str) -> str:
    """
    Convenience wrapper: save a SchoolGraphData object.

    Args:
        school_graph: SchoolGraphData instance.
        filepath: Destination path.

    Returns:
        Resolved filepath.
    """
    metadata = {
        'num_rooms': school_graph.num_rooms,
        'num_env_nodes': school_graph.num_env_nodes,
    }
    return save_hetero_data(school_graph.hetero_data, filepath, metadata)


def load_school_graph(filepath: str):
    """
    Convenience wrapper: load a SchoolGraphData object.

    Args:
        filepath: Path to the saved .pt file.

    Returns:
        SchoolGraphData instance.

    Note:
        Requires src.graph.school_graph.SchoolGraphData to be importable.
        Import is deferred to avoid circular imports.
    """
    from graph.school_graph import SchoolGraphData

    data, metadata = load_hetero_data(filepath)
    return SchoolGraphData(data)
