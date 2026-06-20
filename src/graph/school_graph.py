"""
School Graph Data Class — PyG HeteroData Wrapper

Wraps a PyG HeteroData object with school-specific accessors, validation,
and utilities. Uses the WRAPPER pattern (not subclassing) because PyG's
HeteroData uses __getattr__ magic that makes subclassing fragile.

=== HeteroData Schema ===

Node types:
  'room'          — spatial nodes (dim=27 features)
  'environment'   — global context nodes (dim=6 features)

Edge types (task.md §3.2):
  ('room', 'physical_connects', 'room')       — doors, passages
  ('room', 'acoustic_blocks', 'room')          — sound-isolating walls
  ('room', 'sight_lines', 'room')              — visual connections (room↔room)
  ('room', 'sight_lines', 'environment')       — visual connections (room↔env)
  ('room', 'physical_connects', 'environment') — entrance/exit (room↔env)
"""

from __future__ import annotations

from typing import Tuple, Dict, Optional, List

import torch

try:
    from torch_geometric.data import HeteroData
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    HeteroData = None  # type: ignore

from utils.enums import EdgeCategory
from utils.constants import ROOM_FEAT_DIM, ENV_FEAT_DIM


class SchoolGraphData:
    """
    Wrapper around PyG HeteroData for school building graphs.

    Provides:
      - Named accessors for node/edge properties
      - Dimension validation
      - NetworkX conversion
      - Serialization helpers
    """

    # Valid edge type triples (as strings for comparison)
    EDGE_TYPES = [
        ('room', 'physical_connects', 'room'),
        ('room', 'acoustic_blocks', 'room'),
        ('room', 'sight_lines', 'room'),
        ('room', 'sight_lines', 'environment'),
        ('room', 'physical_connects', 'environment'),
    ]

    def __init__(self, hetero_data: "HeteroData"):
        """
        Args:
            hetero_data: PyG HeteroData with the school graph schema.

        Raises:
            ImportError: If PyG is not installed.
            ValueError: If the HeteroData has invalid structure.
        """
        if not HAS_PYG:
            raise ImportError(
                "PyTorch Geometric is required. Install with: pip install torch-geometric"
            )

        self._data = hetero_data
        # Validate on construction
        errors = self.validate_tensor_dimensions()
        if errors:
            raise ValueError(
                f"Invalid SchoolGraphData — tensor dimension mismatches:\n" +
                "\n".join(f"  - {e}" for e in errors)
            )

    # ========================================================================
    # Core accessor
    # ========================================================================

    @property
    def hetero_data(self) -> "HeteroData":
        """Direct access to the underlying PyG HeteroData."""
        return self._data

    # ========================================================================
    # Node accessors
    # ========================================================================

    @property
    def num_rooms(self) -> int:
        """Number of room nodes."""
        return self._data['room'].num_nodes

    @property
    def num_env_nodes(self) -> int:
        """Number of environmental nodes."""
        return self._data['environment'].num_nodes

    @property
    def room_features(self) -> torch.Tensor:
        """Room node feature tensor: shape [num_rooms, ROOM_FEAT_DIM]."""
        return self._data['room'].x

    @property
    def env_features(self) -> torch.Tensor:
        """Environment node feature tensor: shape [num_env, ENV_FEAT_DIM]."""
        return self._data['environment'].x

    @property
    def room_ids(self) -> List[str]:
        """List of room_id strings in index order."""
        ids = self._data['room'].room_ids
        return list(ids) if ids is not None else []

    @property
    def env_ids(self) -> List[str]:
        """List of env_id strings in index order."""
        ids = self._data['environment'].env_ids
        return list(ids) if ids is not None else []

    # ========================================================================
    # Edge accessors
    # ========================================================================

    def get_edge_index(self, edge_type: Tuple[str, str, str]) -> torch.Tensor:
        """
        Get the edge_index tensor for a given edge type.

        Args:
            edge_type: Tuple like ('room', 'physical_connects', 'room').

        Returns:
            LongTensor of shape [2, num_edges], or empty [2, 0] if no edges.
        """
        return self._data[edge_type].edge_index

    def get_edge_attr(self, edge_type: Tuple[str, str, str]) -> torch.Tensor:
        """
        Get the edge_attr tensor for a given edge type.

        Returns:
            FloatTensor of shape [num_edges, attr_dim], or empty if no edges.
        """
        return self._data[edge_type].edge_attr

    @property
    def physical_edges(self) -> torch.Tensor:
        """Edge index for physical_connects (room↔room)."""
        return self.get_edge_index(('room', 'physical_connects', 'room'))

    @property
    def acoustic_edges(self) -> torch.Tensor:
        """Edge index for acoustic_blocks (room↔room)."""
        return self.get_edge_index(('room', 'acoustic_blocks', 'room'))

    @property
    def sight_room_edges(self) -> torch.Tensor:
        """Edge index for sight_lines (room↔room)."""
        return self.get_edge_index(('room', 'sight_lines', 'room'))

    @property
    def sight_env_edges(self) -> torch.Tensor:
        """Edge index for sight_lines (room↔environment)."""
        return self.get_edge_index(('room', 'sight_lines', 'environment'))

    def edge_counts(self) -> Dict[str, int]:
        """Return a dict of edge type → count (using full edge type string as key)."""
        counts = {}
        for edge_type in self.EDGE_TYPES:
            # Use full edge type as key to avoid collisions
            key = f"{edge_type[0]}→{edge_type[1]}→{edge_type[2]}"
            try:
                ei = self._data[edge_type].edge_index
                counts[key] = ei.shape[1] if ei.numel() > 0 else 0
            except (KeyError, AttributeError):
                counts[key] = 0
        return counts

    # ========================================================================
    # Dimension validation
    # ========================================================================

    def validate_tensor_dimensions(self) -> List[str]:
        """
        Verify that all feature and edge_index tensors have consistent dimensions.

        Checks:
          1. room.x.shape[1] == ROOM_FEAT_DIM
          2. env.x.shape[1] == ENV_FEAT_DIM
          3. room.x.shape[0] == room.num_nodes
          4. edge_index values are within valid node index ranges
          5. edge_attr rows match edge_index columns

        Returns:
            List of error strings (empty list = valid).
        """
        errors: List[str] = []

        # Check room features
        room_x = self._data['room'].x
        if room_x is not None:
            n_rooms, f_dim = room_x.shape
            if f_dim != ROOM_FEAT_DIM:
                errors.append(
                    f"room.x has feature dim {f_dim}, expected {ROOM_FEAT_DIM}"
                )
            if n_rooms != self._data['room'].num_nodes:
                errors.append(
                    f"room.x has {n_rooms} rows but num_nodes="
                    f"{self._data['room'].num_nodes}"
                )

        # Check env features
        env_x = self._data['environment'].x
        if env_x is not None:
            n_env, f_dim = env_x.shape
            if f_dim != ENV_FEAT_DIM:
                errors.append(
                    f"environment.x has feature dim {f_dim}, expected {ENV_FEAT_DIM}"
                )
            if n_env != self._data['environment'].num_nodes:
                errors.append(
                    f"environment.x has {n_env} rows but num_nodes="
                    f"{self._data['environment'].num_nodes}"
                )

        # Check edge indices
        n_rooms = self._data['room'].num_nodes
        n_env = self._data['environment'].num_nodes

        for edge_type in self.EDGE_TYPES:
            try:
                store = self._data[edge_type]
            except (KeyError, AttributeError):
                continue

            if not hasattr(store, 'edge_index') or store.edge_index is None:
                continue

            ei = store.edge_index
            if ei.numel() == 0:
                continue  # Empty edge set is valid

            if ei.shape[0] != 2:
                errors.append(
                    f"Edge type {edge_type}: edge_index shape {ei.shape}, "
                    f"expected [2, N]"
                )

            # Check that indices are within bounds
            src_type, rel, dst_type = edge_type
            max_src = n_rooms if src_type == 'room' else n_env
            max_dst = n_rooms if dst_type == 'room' else n_env

            if ei[0].max() >= max_src:
                errors.append(
                    f"Edge type {edge_type}: source index {ei[0].max().item()} "
                    f">= node count {max_src}"
                )
            if ei[1].max() >= max_dst:
                errors.append(
                    f"Edge type {edge_type}: dest index {ei[1].max().item()} "
                    f">= node count {max_dst}"
                )

            # Check edge_attr matches edge_index
            if hasattr(store, 'edge_attr') and store.edge_attr is not None:
                ea = store.edge_attr
                if ea.shape[0] != ei.shape[1]:
                    errors.append(
                        f"Edge type {edge_type}: edge_attr has {ea.shape[0]} rows "
                        f"but edge_index has {ei.shape[1]} columns"
                    )

        return errors

    # ========================================================================
    # NetworkX conversion
    # ========================================================================

    def to_networkx(self) -> "nx.Graph":
        """
        Convert to a NetworkX graph for analysis and visualization.

        Returns a single NetworkX graph with:
          - Node attributes: 'type' ('room' or 'environment'), 'features'
          - Edge attributes: 'edge_type', 'attributes'

        Note: Multi-relational edges are flattened into a single graph.
        Edge types are stored as edge attributes.
        """
        import networkx as nx

        G = nx.Graph()

        # Add room nodes
        for i in range(self.num_rooms):
            G.add_node(
                f"room_{i}",
                node_type='room',
                room_id=self.room_ids[i] if i < len(self.room_ids) else f"room_{i}",
                features=self.room_features[i] if self.room_features is not None else None,
            )

        # Add environment nodes
        for i in range(self.num_env_nodes):
            G.add_node(
                f"env_{i}",
                node_type='environment',
                env_id=self.env_ids[i] if i < len(self.env_ids) else f"env_{i}",
                features=self.env_features[i] if self.env_features is not None else None,
            )

        # Add edges for each edge type
        for edge_type in self.EDGE_TYPES:
            try:
                ei = self._data[edge_type].edge_index
                ea = self._data[edge_type].edge_attr
            except (KeyError, AttributeError):
                continue

            if ei is None or ei.numel() == 0:
                continue

            src_type, rel, dst_type = edge_type
            for j in range(ei.shape[1]):
                src_idx = ei[0, j].item()
                dst_idx = ei[1, j].item()

                if src_type == 'room':
                    src_name = f"room_{src_idx}"
                else:
                    src_name = f"env_{src_idx}"

                if dst_type == 'room':
                    dst_name = f"room_{dst_idx}"
                else:
                    dst_name = f"env_{dst_idx}"

                edge_attrs = {'edge_type': rel}
                if ea is not None and j < ea.shape[0]:
                    edge_attrs['attributes'] = ea[j].tolist()

                G.add_edge(src_name, dst_name, **edge_attrs)

        return G

    # ========================================================================
    # Statistics
    # ========================================================================

    def summary(self) -> str:
        """Human-readable summary of the graph structure."""
        lines = [
            f"SchoolGraphData:",
            f"  Rooms:       {self.num_rooms} (feat_dim={ROOM_FEAT_DIM})",
            f"  Env Nodes:   {self.num_env_nodes} (feat_dim={ENV_FEAT_DIM})",
            f"  Edges:",
        ]
        for edge_type, count in self.edge_counts().items():
            lines.append(f"    {edge_type:28s}: {count:5d}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"SchoolGraphData(rooms={self.num_rooms}, "
            f"env={self.num_env_nodes}, "
            f"edges={sum(self.edge_counts().values())})"
        )
