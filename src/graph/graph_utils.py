"""
Graph Utilities — graph analysis on SchoolGraphData.

Provides classical graph algorithms (shortest path, centrality, topological
masks) using NetworkX for analysis and PyG tensors for mask generation.

All algorithms use the physical connection subgraph by default, with options
to analyze other edge types.
"""

from __future__ import annotations

from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict

import torch
import networkx as nx


class GraphAnalyzer:
    """
    Performs graph analysis on SchoolGraphData.

    Combines NetworkX (for classical graph algorithms) with PyG tensor
    operations (for topology masks used in GNN training).

    Usage:
        sg = SchoolGraphData(hetero_data)
        analyzer = GraphAnalyzer(sg)
        paths = analyzer.all_pairs_shortest_path()
        mask = analyzer.apply_fire_exit_mask()
    """

    def __init__(self, school_graph):
        """
        Args:
            school_graph: SchoolGraphData instance.
        """
        self.sg = school_graph
        self._nx_cache: Optional[nx.Graph] = None

    @property
    def nx_graph(self) -> nx.Graph:
        """Lazy-loaded NetworkX conversion (cached)."""
        if self._nx_cache is None:
            self._nx_cache = self.sg.to_networkx()
        return self._nx_cache

    # ========================================================================
    # Shortest path
    # ========================================================================

    def shortest_path(
        self,
        src_id: str,
        dst_id: str,
        edge_type: str = 'physical_connects',
    ) -> Tuple[float, List[str]]:
        """
        Compute the shortest path between two rooms on the specified edge type.

        Args:
            src_id: Source room_id (e.g., 'classroom_000_f1').
            dst_id: Destination room_id.
            edge_type: Which edge type to traverse ('physical_connects',
                       'acoustic_blocks', 'sight_lines').

        Returns:
            Tuple of (path_length, path_node_list).
            If no path exists, returns (inf, []).
        """
        G = self._build_edge_type_subgraph(edge_type)
        src_name = self._room_id_to_nx_name(src_id)
        dst_name = self._room_id_to_nx_name(dst_id)

        try:
            path = nx.shortest_path(G, source=src_name, target=dst_name)
            return (float(len(path) - 1), path)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return (float('inf'), [])

    def all_pairs_shortest_path(
        self,
        edge_type: str = 'physical_connects',
    ) -> Dict[Tuple[str, str], float]:
        """
        Compute all-pairs shortest path lengths on the specified edge type.

        Returns:
            Dict[(src_room_id, dst_room_id), path_length].
            Only includes reachable pairs.
        """
        G = self._build_edge_type_subgraph(edge_type)

        # Build reverse mapping: nx_name → room_id
        nx_to_room = {}
        for i, rid in enumerate(self.sg.room_ids):
            nx_to_room[f"room_{i}"] = rid

        lengths = {}
        for source in G.nodes():
            if not source.startswith('room_'):
                continue
            src_rid = nx_to_room.get(source, source)
            sp = nx.single_source_shortest_path_length(G, source)
            for target, dist in sp.items():
                if target == source or not target.startswith('room_'):
                    continue
                dst_rid = nx_to_room.get(target, target)
                lengths[(src_rid, dst_rid)] = float(dist)

        return lengths

    # ========================================================================
    # Centrality
    # ========================================================================

    def betweenness_centrality(
        self,
        edge_type: str = 'physical_connects',
    ) -> Dict[str, float]:
        """
        Compute node betweenness centrality on the specified edge type.

        Reference: task.md §3.1 — 动线代理: 介数中心性用于预判学校人流效率.

        Formula:
            BC(v) = sum_{s≠v≠t} σ_st(v) / σ_st
            where σ_st = total shortest paths from s to t,
                  σ_st(v) = paths passing through v.

        Returns:
            Dict[room_id, centrality_value]. Higher = more critical for circulation.
        """
        G = self._build_edge_type_subgraph(edge_type)

        nx_bc = nx.betweenness_centrality(G, normalized=True)

        # Map nx node names back to room_ids
        result = {}
        for i, rid in enumerate(self.sg.room_ids):
            nx_name = f"room_{i}"
            result[rid] = nx_bc.get(nx_name, 0.0)

        return result

    # ========================================================================
    # Topology Masks (for GNN training, Phase 2)
    # ========================================================================

    def apply_fire_exit_mask(
        self,
        occupancy_threshold: int = 50,
    ) -> torch.Tensor:
        """
        Generate a fire exit topology mask.

        Returns a boolean tensor of shape [num_rooms, num_rooms] where
        mask[i, j] = True if adding a physical_connects edge between
        room_i and room_j would contribute to satisfying fire exit
        requirements.

        Used in Phase 2 for constrained GNN message passing.

        Formula (§4 硬约束):
            Room i is "deficient" if its physical degree < fire_exits_min
            AND its occupancy >= threshold.
            mask[i, j] = True if i is deficient and j is NOT a corridor
                         (connecting to corridors is the default mechanism,
                         but direct room-room fire exits are also valid).
        """
        n_rooms = self.sg.num_rooms
        mask = torch.zeros(n_rooms, n_rooms, dtype=torch.bool)

        # Get physical degree for each room
        phys_ei = self.sg.physical_edges
        degrees = torch.zeros(n_rooms, dtype=torch.long)
        if phys_ei.numel() > 0:
            for j in range(phys_ei.shape[1]):
                src, dst = phys_ei[0, j].item(), phys_ei[1, j].item()
                degrees[src] += 1
                degrees[dst] += 1

        # Identify deficient rooms from features
        room_x = self.sg.room_features
        for i in range(n_rooms):
            # occupancy at col 15, fire_exits_min at col 26
            occupancy_norm = room_x[i, 15].item()
            fire_exits_norm = room_x[i, 26].item()

            # Denormalize (approximate)
            # occupancy density defaults: occupancy_norm * max_occupancy
            occupancy = occupancy_norm * 300  # DEFAULT_MAX_OCCUPANCY
            fire_exits = max(1, int(fire_exits_norm * 4))  # DEFAULT_MAX_FIRE_EXITS

            if occupancy >= occupancy_threshold and degrees[i].item() < fire_exits:
                # Room is deficient — mask all non-self nodes
                for j in range(n_rooms):
                    if i != j:
                        mask[i, j] = True

        return mask

    def apply_acoustic_mask(
        self,
        noise_gap_threshold: int = 2,
    ) -> torch.Tensor:
        """
        Generate an acoustic separation topology mask.

        Returns a boolean tensor of shape [num_rooms, num_rooms] where
        mask[i, j] = True if an acoustic_blocks edge between i and j
        would be architecturally valid.

        Formula (§4 硬约束):
            mask[i, j] = True if:
                noise_level(i) - noise_tolerance(j) >= noise_gap_threshold
                OR noise_level(j) - noise_tolerance(i) >= noise_gap_threshold
        """
        n_rooms = self.sg.num_rooms
        mask = torch.zeros(n_rooms, n_rooms, dtype=torch.bool)
        room_x = self.sg.room_features

        for i in range(n_rooms):
            for j in range(i + 1, n_rooms):
                # noise_level at col 17, noise_tolerance at col 18
                # Denormalize: 0-4 scale
                ni = int(room_x[i, 17].item() * 4)
                nj = int(room_x[j, 17].item() * 4)
                ti = int(room_x[i, 18].item() * 4)
                tj = int(room_x[j, 18].item() * 4)

                if ni - tj >= noise_gap_threshold or nj - ti >= noise_gap_threshold:
                    mask[i, j] = True
                    mask[j, i] = True

        return mask

    def apply_daylight_mask(self) -> torch.Tensor:
        """
        Generate a daylight connection topology mask.

        Returns a boolean tensor of shape [num_rooms, num_env] where
        mask[i, j] = True if connecting room_i to env_j via sight_lines
        would satisfy daylight requirements.

        Formula (§4 软约束):
            mask[i, j] = True if room_i.requires_daylight()
        """
        n_rooms = self.sg.num_rooms
        n_env = self.sg.num_env_nodes
        mask = torch.zeros(n_rooms, n_env, dtype=torch.bool)
        room_x = self.sg.room_features

        for i in range(n_rooms):
            # daylight_level at col 16
            dl = room_x[i, 16].item() * 4  # denormalize
            if dl >= 3.0:  # HIGH or CRITICAL
                for j in range(n_env):
                    mask[i, j] = True

        return mask

    # ========================================================================
    # Connectivity Analysis
    # ========================================================================

    def find_isolated_rooms(
        self,
        edge_type: str = 'physical_connects',
    ) -> List[str]:
        """
        Find rooms with no edges of the specified type.

        Returns:
            List of room_id strings.
        """
        G = self._build_edge_type_subgraph(edge_type)
        isolated = []

        for i, rid in enumerate(self.sg.room_ids):
            nx_name = f"room_{i}"
            if G.degree(nx_name) == 0:
                isolated.append(rid)

        return isolated

    def find_bridges(
        self,
        edge_type: str = 'physical_connects',
    ) -> List[Tuple[str, str]]:
        """
        Find bridge edges (whose removal disconnects the graph).

        Bridges are critical for fire safety — they represent single points
        of failure in the evacuation network.

        Returns:
            List of (room_id_a, room_id_b) tuples representing bridge edges.
        """
        G = self._build_edge_type_subgraph(edge_type)
        nx_bridges = list(nx.bridges(G))

        # Map back to room_ids
        nx_to_room = {}
        for i, rid in enumerate(self.sg.room_ids):
            nx_to_room[f"room_{i}"] = rid
        for i, rid in enumerate(self.sg.env_ids):
            nx_to_room[f"env_{i}"] = rid

        result = []
        for u, v in nx_bridges:
            ru = nx_to_room.get(u, u)
            rv = nx_to_room.get(v, v)
            result.append((ru, rv))

        return result

    def component_analysis(
        self,
        edge_type: str = 'physical_connects',
    ) -> Dict[int, List[str]]:
        """
        Find connected components in the specified edge type graph.

        Returns:
            Dict[component_id, list_of_room_ids].
        """
        G = self._build_edge_type_subgraph(edge_type)

        nx_to_room = {}
        for i, rid in enumerate(self.sg.room_ids):
            nx_to_room[f"room_{i}"] = rid
        for i, rid in enumerate(self.sg.env_ids):
            nx_to_room[f"env_{i}"] = rid

        components: Dict[int, List[str]] = {}
        for comp_id, nodes in enumerate(nx.connected_components(G)):
            room_list = [nx_to_room.get(n, n) for n in nodes
                         if n.startswith('room_')]
            if room_list:
                components[comp_id] = room_list

        return components

    # ========================================================================
    # Helpers
    # ========================================================================

    def _build_edge_type_subgraph(self, edge_type: str) -> nx.Graph:
        """
        Extract a subgraph from the NetworkX graph containing only edges
        of the specified type.

        Args:
            edge_type: One of 'physical_connects', 'acoustic_blocks', 'sight_lines'.

        Returns:
            NetworkX graph containing only edges of that type.
        """
        full_G = self.nx_graph
        G = nx.Graph()

        # Copy all nodes
        for node, attrs in full_G.nodes(data=True):
            G.add_node(node, **attrs)

        # Copy only edges of the specified type
        for u, v, attrs in full_G.edges(data=True):
            if attrs.get('edge_type') == edge_type:
                G.add_edge(u, v, **attrs)

        return G

    def _room_id_to_nx_name(self, room_id: str) -> str:
        """Convert a room_id string to a NetworkX node name."""
        if room_id in self.sg.room_ids:
            idx = self.sg.room_ids.index(room_id)
            return f"room_{idx}"
        if room_id in self.sg.env_ids:
            idx = self.sg.env_ids.index(room_id)
            return f"env_{idx}"
        return room_id
