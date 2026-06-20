"""
Feature Engineering — converts RoomNode/EnvironmentalNode object representations
into PyG HeteroData tensor representations.

This module is the bridge between the object-oriented data model (RoomNode,
EnvironmentalNode) and the PyG graph tensor representation. It is the SINGLE
point where feature dimensions are defined — all downstream modules
(gnn models, explainers) will reference these dimensions.

Feature Layout (room, dim=27):
  [0:13]    RoomType one-hot (13 classes)
  [13]      area, normalized to [0, 1]
  [14]      aspect_ratio
  [15]      occupancy, normalized to [0, 1]
  [16]      daylight_level, ordinal normalized to [0, 1]
  [17]      noise_level, ordinal normalized to [0, 1]
  [18]      noise_tolerance, ordinal normalized to [0, 1]
  [19]      floor, normalized to [0, 1]
  [20:26]   ZoneType one-hot (6 zones)
  [26]      fire_exits_min, normalized to [0, 1]

Feature Layout (environment, dim=6):
  [0:4]     EnvNodeType one-hot (4 classes)
  [4]       position_x, normalized to [0, 1]
  [5]       position_y, normalized to [0, 1]
"""

from __future__ import annotations

from typing import List, Dict, Tuple, Optional
import math

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from torch_geometric.data import HeteroData
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    HeteroData = None  # type: ignore

from utils.enums import (
    RoomType, EnvNodeType, EdgeCategory,
    ZoneType, ROOM_TO_ZONE,
)
from utils.constants import (
    NUM_ROOM_TYPES, NUM_ENV_TYPES, NUM_ZONE_TYPES,
    ROOM_FEAT_DIM, ENV_FEAT_DIM,
    DEFAULT_MAX_AREA, DEFAULT_MAX_OCCUPANCY,
    DEFAULT_MAX_FIRE_EXITS, DEFAULT_MAX_FLOORS,
    DEFAULT_SITE_MAX_X, DEFAULT_SITE_MAX_Y,
    DEFAULT_MAX_ATTENUATION,
)


class FeatureEngineer:
    """
    Converts RoomNode and EnvironmentalNode objects into PyG tensor format.

    Responsibilities:
      1. Build node feature tensors (room.x, environment.x)
      2. Build edge_index tensors for each edge type
      3. Build edge_attr tensors for each edge type
      4. Maintain index mappings (room_id / env_id → integer index)
    """

    def __init__(
        self,
        max_area: float = DEFAULT_MAX_AREA,
        max_occupancy: float = DEFAULT_MAX_OCCUPANCY,
        max_fire_exits: int = DEFAULT_MAX_FIRE_EXITS,
        max_floors: int = DEFAULT_MAX_FLOORS,
        site_max_x: float = DEFAULT_SITE_MAX_X,
        site_max_y: float = DEFAULT_SITE_MAX_Y,
        max_attenuation: float = DEFAULT_MAX_ATTENUATION,
        max_aspect_ratio: float = 12.0,
    ):
        self.max_area = max_area
        self.max_occupancy = max_occupancy
        self.max_fire_exits = float(max_fire_exits)
        self.max_floors = float(max_floors)
        self.site_max_x = site_max_x
        self.site_max_y = site_max_y
        self.max_attenuation = max_attenuation
        self.max_aspect_ratio = max_aspect_ratio

    # ========================================================================
    # Node feature construction
    # ========================================================================

    def build_room_features(
        self, rooms: list  # List[RoomNode]
    ) -> Tuple[torch.Tensor, Dict[str, int]]:
        """
        Build the room node feature tensor.

        Args:
            rooms: List of RoomNode instances.

        Returns:
            (features, room_id_to_idx) where:
              - features: tensor of shape [num_rooms, ROOM_FEAT_DIM]
              - room_id_to_idx: mapping from room_id string → integer index
        """
        num_rooms = len(rooms)
        features = torch.zeros(num_rooms, ROOM_FEAT_DIM)
        room_id_to_idx: Dict[str, int] = {}

        for idx, room in enumerate(rooms):
            room_id_to_idx[room.room_id] = idx

            # --- RoomType one-hot [0:13] ---
            rt_idx = list(RoomType).index(room.room_type)
            features[idx, rt_idx] = 1.0

            # --- Scalar features [13:20] ---
            features[idx, 13] = self._normalize(room.area, self.max_area)
            features[idx, 14] = self._normalize(room.aspect_ratio, self.max_aspect_ratio)
            features[idx, 15] = self._normalize(room.occupancy, self.max_occupancy)
            features[idx, 16] = self._normalize(room.daylight_level.value, 4.0)   # max DaylightLevel = 4
            features[idx, 17] = self._normalize(room.noise_level.value, 4.0)       # max NoiseLevel = 4
            features[idx, 18] = self._normalize(room.noise_tolerance.value, 4.0)   # max NoiseLevel = 4
            # Floor range compressed: (floor_range[0] + floor_range[1]) / (2 * max_floors)
            floor_mid = (room.floor_range[0] + room.floor_range[1]) / 2.0
            features[idx, 19] = self._normalize(floor_mid, self.max_floors)

            # --- ZoneType one-hot [20:26] ---
            zone = ROOM_TO_ZONE[room.room_type]
            zi = list(ZoneType).index(zone)
            features[idx, 20 + zi] = 1.0

            # --- fire_exits_min [26] ---
            features[idx, 26] = self._normalize(room.fire_exits_min, self.max_fire_exits)

        return features, room_id_to_idx

    def build_env_features(
        self, env_nodes: list  # List[EnvironmentalNode]
    ) -> Tuple[torch.Tensor, Dict[str, int]]:
        """
        Build the environment node feature tensor.

        Args:
            env_nodes: List of EnvironmentalNode instances.

        Returns:
            (features, env_id_to_idx) where:
              - features: tensor of shape [num_env, ENV_FEAT_DIM]
              - env_id_to_idx: mapping from env_id string → integer index
        """
        num_env = len(env_nodes)
        features = torch.zeros(num_env, ENV_FEAT_DIM)
        env_id_to_idx: Dict[str, int] = {}

        for idx, node in enumerate(env_nodes):
            env_id_to_idx[node.env_id] = idx

            # --- EnvNodeType one-hot [0:4] ---
            et_idx = list(EnvNodeType).index(node.env_type)
            features[idx, et_idx] = 1.0

            # --- Position [4:6] ---
            features[idx, 4] = self._normalize(node.position[0], self.site_max_x)
            features[idx, 5] = self._normalize(node.position[1], self.site_max_y)

        return features, env_id_to_idx

    # ========================================================================
    # Edge index / edge attr construction
    # ========================================================================

    def build_edge_index(
        self,
        edges: List[Tuple[str, str, dict]],
        src_id_to_idx: Dict[str, int],
        dst_id_to_idx: Dict[str, int],
    ) -> torch.Tensor:
        """
        Build edge_index tensor from edge list.

        Args:
            edges: List of (src_id, dst_id, edge_attrs) tuples.
            src_id_to_idx: Mapping from source node ID → integer index.
            dst_id_to_idx: Mapping from destination node ID → integer index.

        Returns:
            edge_index: tensor of shape [2, num_edges], dtype=torch.long.
        """
        if not edges:
            return torch.zeros(2, 0, dtype=torch.long)

        edge_index = torch.zeros(2, len(edges), dtype=torch.long)
        for i, (src_id, dst_id, _attrs) in enumerate(edges):
            edge_index[0, i] = src_id_to_idx[src_id]
            edge_index[1, i] = dst_id_to_idx[dst_id]

        return edge_index

    def build_physical_edge_attr(
        self, edges: List[Tuple[str, str, dict]]
    ) -> torch.Tensor:
        """
        Build edge_attr for physical_connects edges.

        Schema: [distance_weight, is_stair_connection]
        """
        num = len(edges)
        if num == 0:
            return torch.zeros(0, 2)

        attrs = torch.zeros(num, 2)
        for i, (_src, _dst, edge_dict) in enumerate(edges):
            attrs[i, 0] = float(edge_dict.get('distance_weight', 0.0))
            attrs[i, 1] = float(edge_dict.get('is_stair_connection', 0.0))

        return attrs

    def build_acoustic_edge_attr(
        self, edges: List[Tuple[str, str, dict]]
    ) -> torch.Tensor:
        """
        Build edge_attr for acoustic_blocks edges.

        Schema: [attenuation_db]
        """
        num = len(edges)
        if num == 0:
            return torch.zeros(0, 1)

        attrs = torch.zeros(num, 1)
        for i, (_src, _dst, edge_dict) in enumerate(edges):
            raw = float(edge_dict.get('attenuation_db', 0.0))
            attrs[i, 0] = self._normalize(raw, self.max_attenuation)

        return attrs

    def build_sight_room_edge_attr(
        self, edges: List[Tuple[str, str, dict]]
    ) -> torch.Tensor:
        """
        Build edge_attr for room-to-room sight_lines edges.

        Schema: [transparency, sight_distance]
        """
        num = len(edges)
        if num == 0:
            return torch.zeros(0, 2)

        attrs = torch.zeros(num, 2)
        for i, (_src, _dst, edge_dict) in enumerate(edges):
            attrs[i, 0] = float(edge_dict.get('transparency', 0.0))
            # sight_distance normalized by site diagonal
            raw_dist = float(edge_dict.get('sight_distance', 0.0))
            max_dist = math.sqrt(self.site_max_x ** 2 + self.site_max_y ** 2)
            attrs[i, 1] = self._normalize(raw_dist, max_dist)

        return attrs

    def build_sight_env_edge_attr(
        self, edges: List[Tuple[str, str, dict]]
    ) -> torch.Tensor:
        """
        Build edge_attr for room-to-environment sight_lines edges.

        Schema: [orientation_preference, distance]
        """
        num = len(edges)
        if num == 0:
            return torch.zeros(0, 2)

        attrs = torch.zeros(num, 2)
        max_dist = math.sqrt(self.site_max_x ** 2 + self.site_max_y ** 2)
        for i, (_src, _dst, edge_dict) in enumerate(edges):
            # orientation_preference: cosine of angle to south, range [-1, 1]
            attrs[i, 0] = float(edge_dict.get('orientation_preference', 0.0))
            raw_dist = float(edge_dict.get('distance', 0.0))
            attrs[i, 1] = self._normalize(raw_dist, max_dist)

        return attrs

    def build_phys_env_edge_attr(
        self, edges: List[Tuple[str, str, dict]]
    ) -> torch.Tensor:
        """
        Build edge_attr for room-to-environment physical_connects edges.

        Schema: [access_type]
        """
        num = len(edges)
        if num == 0:
            return torch.zeros(0, 1)

        attrs = torch.zeros(num, 1)
        for i, (_src, _dst, edge_dict) in enumerate(edges):
            attrs[i, 0] = float(edge_dict.get('access_type', 0.0))

        return attrs

    # ========================================================================
    # Full HeteroData assembly
    # ========================================================================

    def build_hetero_data(
        self,
        rooms: list,           # List[RoomNode]
        env_nodes: list,       # List[EnvironmentalNode]
        edges_by_category: Dict[EdgeCategory, List[Tuple[str, str, dict]]],
    ) -> "HeteroData":
        """
        Build the complete PyG HeteroData object from nodes and edges.

        Args:
            rooms: List of RoomNode instances.
            env_nodes: List of EnvironmentalNode instances.
            edges_by_category: Dict mapping EdgeCategory → edge list.
                Edge list format: [(src_id, dst_id, {attr_name: value}), ...]

        Returns:
            HeteroData with all node and edge types populated.

        Raises:
            ImportError: If PyG is not installed.
        """
        if not HAS_PYG:
            raise ImportError(
                "PyTorch Geometric is required. Install with: pip install torch-geometric"
            )

        data = HeteroData()

        # --- Node features ---
        room_x, room_id_to_idx = self.build_room_features(rooms)
        data['room'].x = room_x
        data['room'].num_nodes = len(rooms)

        env_x, env_id_to_idx = self.build_env_features(env_nodes)
        data['environment'].x = env_x
        data['environment'].num_nodes = len(env_nodes)

        # --- Store ID mappings for reference ---
        data['room'].room_ids = list(room_id_to_idx.keys())
        data['environment'].env_ids = list(env_id_to_idx.keys())

        # --- Edge types ---
        # Split physical edges into room→room and room→env
        phys_all = edges_by_category.get(EdgeCategory.PHYSICAL_CONNECTS, [])
        phys_room_room, phys_room_env = self._split_phys_edges(
            phys_all, room_id_to_idx, env_id_to_idx,
        )

        # 1. (room, physical_connects, room)
        data['room', 'physical_connects', 'room'].edge_index = \
            self.build_edge_index(phys_room_room, room_id_to_idx, room_id_to_idx)
        data['room', 'physical_connects', 'room'].edge_attr = \
            self.build_physical_edge_attr(phys_room_room)

        # 2. (room, acoustic_blocks, room)
        acous_edges = edges_by_category.get(EdgeCategory.ACOUSTIC_BLOCKS, [])
        data['room', 'acoustic_blocks', 'room'].edge_index = \
            self.build_edge_index(acous_edges, room_id_to_idx, room_id_to_idx)
        data['room', 'acoustic_blocks', 'room'].edge_attr = \
            self.build_acoustic_edge_attr(acous_edges)

        # 3. (room, sight_lines, room) and (room, sight_lines, environment)
        sight_rr = self._split_sight_edges(
            edges_by_category.get(EdgeCategory.SIGHT_LINES, []),
            room_id_to_idx, env_id_to_idx,
        )

        # Room-to-room sight
        data['room', 'sight_lines', 'room'].edge_index = \
            self.build_edge_index(sight_rr['room_to_room'], room_id_to_idx, room_id_to_idx)
        data['room', 'sight_lines', 'room'].edge_attr = \
            self.build_sight_room_edge_attr(sight_rr['room_to_room'])

        # Room-to-env sight
        data['room', 'sight_lines', 'environment'].edge_index = \
            self.build_edge_index(sight_rr['room_to_env'], room_id_to_idx, env_id_to_idx)
        data['room', 'sight_lines', 'environment'].edge_attr = \
            self.build_sight_env_edge_attr(sight_rr['room_to_env'])

        # 4. (room, physical_connects, environment) — entrance/exit connections
        data['room', 'physical_connects', 'environment'].edge_index = \
            self.build_edge_index(phys_room_env, room_id_to_idx, env_id_to_idx)
        data['room', 'physical_connects', 'environment'].edge_attr = \
            self.build_phys_env_edge_attr(phys_room_env)

        return data

    @staticmethod
    def _split_sight_edges(
        edges: List[Tuple[str, str, dict]],
        room_ids: Dict[str, int],
        env_ids: Dict[str, int],
    ) -> Dict[str, List[Tuple[str, str, dict]]]:
        """
        Split sight_lines edges into room→room and room→env based on
        whether the destination ID is a room or an environment node.
        """
        room_to_room: List[Tuple[str, str, dict]] = []
        room_to_env: List[Tuple[str, str, dict]] = []

        for src, dst, attrs in edges:
            if dst in env_ids:
                room_to_env.append((src, dst, attrs))
            else:
                room_to_room.append((src, dst, attrs))

        return {'room_to_room': room_to_room, 'room_to_env': room_to_env}

    @staticmethod
    def _split_phys_edges(
        edges: List[Tuple[str, str, dict]],
        room_ids: Dict[str, int],
        env_ids: Dict[str, int],
    ) -> Tuple[List[Tuple[str, str, dict]], List[Tuple[str, str, dict]]]:
        """
        Split physical_connects edges into (room→room, room→env).
        """
        room_to_room: List[Tuple[str, str, dict]] = []
        room_to_env: List[Tuple[str, str, dict]] = []

        for src, dst, attrs in edges:
            if dst in env_ids:
                room_to_env.append((src, dst, attrs))
            else:
                room_to_room.append((src, dst, attrs))

        return room_to_room, room_to_env

    # ========================================================================
    # Utility
    # ========================================================================

    @staticmethod
    def _normalize(value: float, max_val: float) -> float:
        """
        Min-Max normalize a value to [0, 1].

        Formula: value / max_val, clamped to [0, 1].
        Handles max_val <= 0 by returning 0.0.
        """
        if max_val <= 0:
            return 0.0
        return max(0.0, min(1.0, value / max_val))
